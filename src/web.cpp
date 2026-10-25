#include "web.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include "httplib.h"

namespace fs = std::filesystem;

void post_json_async(const std::string& url, const std::string& body) {
    std::thread([url, body] {
        try {
            auto p = url.find("://");
            auto slash = url.find('/', p == std::string::npos ? 0 : p + 3);
            std::string host = url.substr(0, slash), path = slash == std::string::npos ? "/" : url.substr(slash);
            httplib::Client cli(host);
            cli.set_connection_timeout(3);
            cli.set_read_timeout(3);
            auto res = cli.Post(path, body, "application/json");
            if (!res) std::cerr << "[webhook] не доставлено: " << url << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[webhook] " << e.what() << "\n";
        }
    }).detach();
}

namespace {

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::string csv_escape(const std::string& s) {
    std::string r = "\"";
    for (char c : s) r += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return r + "\"";
}

EventQuery parse_query(const httplib::Request& req) {
    EventQuery q;
    auto p = [&](const char* k) { return req.has_param(k) ? req.get_param_value(k) : std::string(); };
    q.type = p("type");
    q.camera = p("camera");
    q.zone = p("zone");
    q.severity = p("severity");
    q.status = p("status");
    if (!p("from").empty()) q.from = std::stoll(p("from"));
    if (!p("to").empty()) q.to = std::stoll(p("to"));
    if (!p("since_id").empty()) q.since_id = std::stoll(p("since_id"));
    if (!p("limit").empty()) q.limit = std::stoi(p("limit"));
    return q;
}

}  // namespace

struct WebServer::Impl {
    Services& svc;
    std::vector<std::unique_ptr<Camera>>& cams;
    std::function<void()> save_config;
    httplib::Server srv;

    Impl(Services& s, std::vector<std::unique_ptr<Camera>>& c, std::function<void()> sc)
        : svc(s), cams(c), save_config(std::move(sc)) {}

    Camera* cam(const std::string& id) {
        for (auto& c : cams)
            if (c->id() == id) return c.get();
        return nullptr;
    }

    void routes() {
        auto json_reply = [](httplib::Response& res, const json& j) {
            res.set_content(j.dump(), "application/json; charset=utf-8");
        };

        srv.Get("/api/cameras", [&, json_reply](const httplib::Request&, httplib::Response& res) {
            json a = json::array();
            for (auto& c : cams) {
                json s = c->status();
                s["zones"] = c->config()["zones"];
                a.push_back(s);
            }
            json_reply(res, a);
        });

        srv.Get(R"(/api/cameras/([^/]+)/(live|frame)\.jpg)", [&](const httplib::Request& req, httplib::Response& res) {
            Camera* c = cam(req.matches[1]);
            std::string jpg;
            if (!c || !c->jpeg(jpg, req.matches[2] == "frame")) {
                res.status = 404;
                return;
            }
            res.set_header("Cache-Control", "no-store");
            res.set_content(jpg, "image/jpeg");
        });

        srv.Post(R"(/api/cameras/([^/]+)/zones)", [&, json_reply](const httplib::Request& req, httplib::Response& res) {
            Camera* c = cam(req.matches[1]);
            if (!c) {
                res.status = 404;
                return;
            }
            try {
                json body = json::parse(req.body);
                std::vector<Zone> zs;
                for (auto& z : body.at("zones")) zs.push_back(Zone::from_json(z));
                c->set_zones(zs);
                save_config();
                json_reply(res, {{"ok", true}, {"zones", body["zones"]}});
            } catch (const std::exception& e) {
                res.status = 400;
                json_reply(res, {{"ok", false}, {"error", e.what()}});
            }
        });

        srv.Get("/api/events", [&, json_reply](const httplib::Request& req, httplib::Response& res) {
            json a = json::array();
            for (auto& e : svc.store->query(parse_query(req))) a.push_back(e.to_json());
            json_reply(res, a);
        });

        srv.Get(R"(/api/events/(\d+))", [&, json_reply](const httplib::Request& req, httplib::Response& res) {
            Event e;
            if (!svc.store->get(std::stoll(req.matches[1]), e)) {
                res.status = 404;
                return;
            }
            json_reply(res, e.to_json());
        });

        srv.Post(R"(/api/events/(\d+)/status)", [&, json_reply](const httplib::Request& req, httplib::Response& res) {
            try {
                json b = json::parse(req.body);
                bool ok = svc.store->set_status(std::stoll(req.matches[1]), b.at("status").get<std::string>(),
                                                jget<std::string>(b, "comment", ""));
                res.status = ok ? 200 : 400;
                json_reply(res, {{"ok", ok}});
            } catch (const std::exception& e) {
                res.status = 400;
                json_reply(res, {{"ok", false}, {"error", e.what()}});
            }
        });

        srv.Get(R"(/api/events/(\d+)/(snapshot|clip_\d+)\.jpg)", [&](const httplib::Request& req, httplib::Response& res) {
            std::string data;
            std::string path = svc.store->event_dir(std::stoll(req.matches[1])) + "/" + std::string(req.matches[2]) + ".jpg";
            if (!read_file(path, data)) {
                res.status = 404;
                return;
            }
            res.set_header("Cache-Control", "max-age=86400");
            res.set_content(data, "image/jpeg");
        });

        srv.Get("/api/stats", [&, json_reply](const httplib::Request&, httplib::Response& res) {
            json_reply(res, svc.store->stats());
        });

        srv.Get("/api/meta", [&, json_reply](const httplib::Request&, httplib::Response& res) {
            json types = json::object();
            for (auto& [k, v] : event_types()) types[k] = {{"title", v.title}, {"severity", v.severity}};
            json models = json::object();
            for (auto& [k, m] : svc.models)
                models[k] = {{"kind", m->meta().kind}, {"backend", m->backend_name()},
                             {"classes", m->meta().classes.size()}, {"name", m->meta().name}};
            json_reply(res, {{"types", types},
                             {"zone_kinds", {{"danger", "Опасная зона"}, {"fence", "Ограждение (перелезание)"},
                                             {"under_load", "Под грузом / конвейером"}}},
                             {"models", models},
                             {"rules", svc.cfg.value("rules", json::object())}});
        });

        srv.Get("/api/bench", [&, json_reply](const httplib::Request&, httplib::Response& res) {
            json a = json::array();
            if (fs::exists(svc.data_dir))
                for (auto& e : fs::directory_iterator(svc.data_dir)) {
                    auto n = e.path().filename().string();
                    if (n.rfind("bench_", 0) == 0 && e.path().extension() == ".json") {
                        std::string s;
                        if (read_file(e.path().string(), s)) try {
                                a.push_back(json::parse(s));
                            } catch (...) {
                            }
                    }
                }
            json_reply(res, a);
        });

        srv.Get("/api/export.csv", [&](const httplib::Request& req, httplib::Response& res) {
            EventQuery q = parse_query(req);
            if (!req.has_param("limit")) q.limit = 1000000;
            std::string out = "\xEF\xBB\xBF";  // BOM, чтобы Excel открыл UTF-8
            out += "id;время;камера;нарушение;важность;зона;детали;статус;комментарий;длительность_с\n";
            for (auto& e : svc.store->query(q)) {
                double dur = e.ts_end ? (e.ts_end - e.ts_start) / 1000.0 : -1;
                out += std::to_string(e.id) + ";" + iso_local(e.ts_alert) + ";" + csv_escape(e.camera_name) + ";" +
                       csv_escape(e.title) + ";" + e.severity + ";" + csv_escape(e.zone) + ";" + csv_escape(e.detail) +
                       ";" + e.status + ";" + csv_escape(e.comment) + ";" + (dur >= 0 ? std::to_string(dur) : "") + "\n";
            }
            res.set_header("Content-Disposition", "attachment; filename=\"violations.csv\"");
            res.set_content(out, "text/csv; charset=utf-8");
        });
    }
};

WebServer::WebServer(Services& svc, std::vector<std::unique_ptr<Camera>>& cams, std::function<void()> save_config)
    : impl_(std::make_unique<Impl>(svc, cams, std::move(save_config))) {
    impl_->routes();
}

WebServer::~WebServer() = default;

bool WebServer::run(const std::string& host, int port, const std::string& web_dir) {
    if (!impl_->srv.set_mount_point("/", web_dir)) std::cerr << "[web] нет папки интерфейса: " << web_dir << "\n";
    std::cerr << "[web] http://" << (host == "0.0.0.0" ? "localhost" : host) << ":" << port << "\n";
    return impl_->srv.listen(host, port);
}

void WebServer::stop() { impl_->srv.stop(); }
