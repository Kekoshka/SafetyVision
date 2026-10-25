// SafetyVision — контроль нарушений промышленной безопасности по видео на LinQ TPU.
//
//   safety_service run    --config config/config.json
//   safety_service detect --config config/config.json --model pose --image img.jpg [--out out.jpg] [--dump-input in.bin]
//   safety_service bench  --config config/config.json --model pose [--image img.jpg] [--iters 200] [--streams 4]
#include <algorithm>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "camera.hpp"
#include "web.hpp"

namespace fs = std::filesystem;

namespace {

WebServer* g_server = nullptr;
void on_signal(int) {
    if (g_server) g_server->stop();
}

struct Args {
    std::string cmd = "run", config = "config/config.json", model, image, out, dump;
    int iters = 200, warmup = 10, streams = 1;
};

Args parse(int argc, char** argv) {
    Args a;
    int i = 1;
    if (argc > 1 && argv[1][0] != '-') a.cmd = argv[i++];
    for (; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (k == "--config") a.config = next();
        else if (k == "--model") a.model = next();
        else if (k == "--image") a.image = next();
        else if (k == "--out") a.out = next();
        else if (k == "--dump-input") a.dump = next();
        else if (k == "--iters") a.iters = std::stoi(next());
        else if (k == "--warmup") a.warmup = std::stoi(next());
        else if (k == "--streams") a.streams = std::stoi(next());
        else if (k == "--help" || k == "-h") {
            std::cout << "Команды: run | detect | bench. См. README.md\n";
            std::exit(0);
        }
    }
    return a;
}

std::string abs_path(const std::string& base, const std::string& p) {
    return (p.empty() || p[0] == '/') ? p : base + "/" + p;
}

void load_services(const json& cfg, const std::string& base, Services& svc, const std::string& only_model = "") {
    svc.cfg = cfg;
    svc.base_dir = base;
    json server = cfg.value("server", json::object());
    svc.data_dir = abs_path(base, jget<std::string>(server, "data_dir", "data"));
    fs::create_directories(svc.data_dir);
    for (auto& v : cfg.value("vehicle_classes", json::array({"truck", "car", "bus", "forklift", "motorcycle"})))
        svc.vehicle_classes.insert(v.get<std::string>());
    json d = cfg.value("dedup", json::object());
    svc.dedup.mode = jget<std::string>(d, "mode", "track");
    svc.dedup.gap_sec = jget<double>(d, "gap_sec", 0.7);
    svc.dedup.release_sec = jget<double>(d, "release_sec", 5.0);
    svc.dedup.cooldown_sec = jget<double>(d, "cooldown_sec", 60.0);
    svc.webhook = jget<std::string>(cfg, "webhook", "");
    json backend = cfg.value("backend", json::object());
    // удалённый инференс: держим несколько кадров в полёте, чтобы FPS не упирался в задержку сети
    svc.pipeline = jget<std::string>(backend, "type", "opencv") == "remote" ? jget<int>(backend, "pipeline", 3) : 1;
    const json models = cfg.value("models", json::object());
    for (auto& [id, mc] : models.items()) {
        if (!only_model.empty() && id != only_model) continue;
        if (!jget<bool>(mc, "enabled", true)) continue;
        try {
            auto m = std::make_shared<YoloModel>(id, mc, backend, base);
            std::cerr << "[model] " << id << ": " << m->meta().name << " (" << m->meta().kind << ", "
                      << m->meta().classes.size() << " классов) на " << m->backend_name() << "\n";
            svc.models[id] = m;
        } catch (const std::exception& e) {
            std::cerr << "[model] " << id << ": НЕ ЗАГРУЖЕНА: " << e.what() << "\n";
        }
    }
}

json load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("не открыть конфиг " + path);
    return json::parse(f, nullptr, true, true);  // разрешены комментарии
}

int cmd_run(const Args& a) {
    json cfg = load_json(a.config);
    const std::string base = fs::absolute(a.config).parent_path().string();
    Services svc;
    load_services(cfg, base, svc);
    if (svc.models.empty()) std::cerr << "[warn] ни одна модель не загружена — будет только видео\n";
    EventStore store(svc.data_dir);
    svc.store = &store;

    std::vector<std::unique_ptr<Camera>> cams;
    for (auto& c : cfg.value("cameras", json::array())) {
        if (!jget<bool>(c, "enabled", true)) continue;
        cams.push_back(std::make_unique<Camera>(c, svc));
    }
    for (auto& c : cams) c->start();

    std::mutex cfg_mu;
    auto save_config = [&] {
        std::lock_guard<std::mutex> lk(cfg_mu);
        json out = cfg;
        out["cameras"] = json::array();
        std::map<std::string, json> live;
        for (auto& c : cams) live[c->id()] = c->config();
        for (auto& c : cfg.value("cameras", json::array())) {
            auto it = live.find(jget<std::string>(c, "id", ""));
            out["cameras"].push_back(it != live.end() ? it->second : c);
        }
        if (!fs::exists(a.config + ".bak")) fs::copy_file(a.config, a.config + ".bak");
        std::ofstream(a.config) << out.dump(2);
        cfg = out;
        std::cerr << "[config] зоны сохранены в " << a.config << "\n";
    };

    json server = cfg.value("server", json::object());
    WebServer web(svc, cams, save_config);
    g_server = &web;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    bool ok = web.run(jget<std::string>(server, "host", "0.0.0.0"), jget<int>(server, "port", 8080),
                      abs_path(base, jget<std::string>(server, "web_dir", "../web")));
    if (!ok) std::cerr << "[web] не удалось запустить сервер (порт занят?)\n";
    for (auto& c : cams) c->stop();
    return ok ? 0 : 1;
}

cv::Mat load_image(const std::string& path) {
    if (path.empty()) return cv::Mat(720, 1280, CV_8UC3, cv::Scalar(90, 90, 90));
    cv::Mat img = cv::imread(path);
    if (img.empty()) throw std::runtime_error("не открыть изображение " + path);
    return img;
}

int cmd_detect(const Args& a) {
    json cfg = load_json(a.config);
    const std::string base = fs::absolute(a.config).parent_path().string();
    Services svc;
    load_services(cfg, base, svc, a.model);
    if (svc.models.empty()) throw std::runtime_error("модель " + a.model + " не загружена");
    auto& m = *svc.models.begin()->second;
    cv::Mat img = load_image(a.image);
    if (!a.dump.empty()) {
        m.dump_input(img, a.dump);
        std::cout << "вход сети сохранён: " << a.dump << "\n";
    }
    Timings t;
    auto dets = m.detect(img, &t);
    std::cout << "Найдено объектов: " << dets.size() << "  (pre " << t.pre << " мс, infer " << t.infer << " мс, post "
              << t.post << " мс" << (t.net > 0 ? ", сеть " + std::to_string(int(t.net)) + " мс" : std::string())
              << (t.tpu > 0 ? ", на TPU " + std::to_string(t.tpu).substr(0, 5) + " мс" : std::string()) << ")\n";
    for (auto& d : dets) {
        std::printf("  %-14s %.2f  [%.1f, %.1f, %.1f, %.1f]  kpts=%zu\n", d.label.c_str(), d.score, d.box.x, d.box.y,
                    d.box.x + d.box.width, d.box.y + d.box.height, d.kpts.size());
        cv::rectangle(img, d.box, {0, 255, 0}, 2);
        cv::putText(img, d.label, {int(d.box.x), int(d.box.y) - 4}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
        for (auto& k : d.kpts)
            if (k.z > 0.4f) cv::circle(img, {int(k.x), int(k.y)}, 3, {0, 0, 255}, cv::FILLED);
    }
    if (!a.out.empty()) cv::imwrite(a.out, img);
    return 0;
}

int cmd_bench(const Args& a) {
    json cfg = load_json(a.config);
    const std::string base = fs::absolute(a.config).parent_path().string();
    Services svc;
    load_services(cfg, base, svc, a.model);
    if (svc.models.empty()) throw std::runtime_error("модель " + a.model + " не загружена");
    auto model = svc.models.begin()->second;
    cv::Mat img = load_image(a.image);
    for (int i = 0; i < a.warmup; ++i) model->detect(img);

    std::vector<double> pre, inf, post, net, tpu, total;
    std::mutex mu;
    const int streams = std::max(1, a.streams);
    const double t0 = mono_sec();
    std::vector<std::thread> th;
    for (int s = 0; s < streams; ++s)
        th.emplace_back([&] {
            for (int i = 0; i < a.iters / streams; ++i) {
                Timings t;
                double s0 = mono_sec();
                model->detect(img, &t);
                double dt = (mono_sec() - s0) * 1e3;
                std::lock_guard<std::mutex> lk(mu);
                pre.push_back(t.pre), inf.push_back(t.infer), post.push_back(t.post), net.push_back(t.net), tpu.push_back(t.tpu), total.push_back(dt);
            }
        });
    for (auto& x : th) x.join();
    const double wall = mono_sec() - t0;

    auto st = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / std::max<size_t>(1, v.size());
        auto pct = [&](double p) { return v.empty() ? 0 : v[std::min(v.size() - 1, size_t(p * v.size()))]; };
        return json{{"mean", mean}, {"p50", pct(0.5)}, {"p95", pct(0.95)}, {"min", v.empty() ? 0 : v.front()}};
    };
    json r = {{"model", a.model},
              {"model_name", model->meta().name},
              {"backend", model->backend_name()},
              {"imgsz", model->meta().imgsz},
              {"image", a.image.empty() ? "synthetic 1280x720" : a.image},
              {"iters", (int)total.size()},
              {"streams", streams},
              {"pre_ms", st(pre)},
              {"infer_ms", st(inf)},
              {"post_ms", st(post)},
              {"net_ms", st(net)},
              {"tpu_ms", st(tpu)},
              {"total_ms", st(total)},
              {"throughput_fps", total.size() / wall},
              {"time", iso_local(wall_ms())}};
    json srv = model->remote_bench(std::max(10, a.iters / 2));
    if (!srv.is_null()) r["server"] = srv;  // чистое время сети на ускорителе (замер на сервере)
    std::cout << r.dump(2) << "\n";
    std::string bname = model->backend_name().rfind("remote", 0) == 0 ? "remote" : model->backend_name();
    std::string out = svc.data_dir + "/bench_" + a.model + "_" + bname + "_s" + std::to_string(streams) + ".json";
    std::ofstream(out) << r.dump(2);
    std::cerr << "результат: " << out << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args a = parse(argc, argv);
        if (a.cmd == "run") return cmd_run(a);
        if (a.cmd == "detect") return cmd_detect(a);
        if (a.cmd == "bench") return cmd_bench(a);
        std::cerr << "неизвестная команда: " << a.cmd << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "ошибка: " << e.what() << "\n";
        return 1;
    }
}
