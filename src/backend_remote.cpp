// Удалённый бэкенд: приложение работает на вашей машине, нейросеть — на ускорителе LinQ.
//
// Кадр после letterbox (imgsz x imgsz) уходит JPEG-ом на python/linq_server.py, который запущен
// на стороне LinQ (Jupyter/ВМ организаторов). Сервер сам декодирует выходы сети и делает NMS,
// обратно приходит компактный JSON с детекциями в координатах входа сети.
//
// https:// работает, если при сборке найден OpenSSL (libssl-dev).
// Конфиг: "backend": {"type": "remote", "url": "http://<адрес>:8500",
//                     "headers": {"Authorization": "token ..."}, "jpeg_quality": 90, "timeout_sec": 5}
#include <mutex>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "httplib.h"
#include "model.hpp"

namespace {

class RemoteBackend : public Backend {
public:
    RemoteBackend(const std::string& model, const json& opts) : model_(model) {
        std::string url = jget<std::string>(opts, "url", "http://127.0.0.1:8500");
        // http://host:port/prefix -> клиент на host:port и префикс пути (для прокси Jupyter)
        auto p = url.find("://");
        auto slash = url.find('/', p == std::string::npos ? 0 : p + 3);
        std::string origin = slash == std::string::npos ? url : url.substr(0, slash);
        prefix_ = slash == std::string::npos ? "" : url.substr(slash);
        while (!prefix_.empty() && prefix_.back() == '/') prefix_.pop_back();
        origin_ = origin;
        timeout_ = jget<int>(opts, "timeout_sec", 5);
        const json hdr = opts.value("headers", json::object());
        for (auto& [k, v] : hdr.items()) headers_.emplace(k, v.get<std::string>());
        quality_ = jget<int>(opts, "jpeg_quality", 90);
        name_ = "remote:" + url;
    }

    bool remote() const override { return true; }

    // httplib::Client выполняет запросы по одному, поэтому держим пул клиентов: кадры разных камер
    // и конвейер одной камеры (camera.pipeline) уходят на сервер одновременно
    struct Lease {
        RemoteBackend* b;
        std::unique_ptr<httplib::Client> c;
        httplib::Client* operator->() { return c.get(); }
        ~Lease() {
            if (!c) return;
            std::lock_guard<std::mutex> lk(b->pool_mu_);
            b->pool_.push_back(std::move(c));
        }
    };
    Lease lease() {
        {
            std::lock_guard<std::mutex> lk(pool_mu_);
            if (!pool_.empty()) {
                auto c = std::move(pool_.back());
                pool_.pop_back();
                return {this, std::move(c)};
            }
        }
        auto c = std::make_unique<httplib::Client>(origin_);
        c->set_connection_timeout(timeout_);
        c->set_read_timeout(timeout_);
        c->set_write_timeout(timeout_);
        c->set_keep_alive(true);
        return {this, std::move(c)};
    }

    std::vector<Det> infer_remote(const cv::Mat& lb, float conf, float iou, Timings& t) override {
        const double t0 = mono_sec();
        std::vector<uchar> jpg;
        cv::imencode(".jpg", lb, jpg, {cv::IMWRITE_JPEG_QUALITY, quality_});
        const double t1 = mono_sec();
        char path[256];
        std::snprintf(path, sizeof path, "%s/infer/%s?conf=%.3f&iou=%.3f", prefix_.c_str(), model_.c_str(), conf, iou);
        auto cli = lease();
        auto res = cli->Post(path, headers_, reinterpret_cast<const char*>(jpg.data()), jpg.size(), "image/jpeg");
        const double t2 = mono_sec();
        if (!res) throw std::runtime_error("LinQ-сервер недоступен (" + name_ + "): " + httplib::to_string(res.error()));
        if (res->status != 200) throw std::runtime_error("LinQ-сервер: HTTP " + std::to_string(res->status) + " " + res->body.substr(0, 200));
        json j = json::parse(res->body);
        std::vector<Det> dets;
        for (auto& d : j.at("dets")) {
            Det x;
            auto b = d.at("b");
            x.box = {b[0].get<float>(), b[1].get<float>(), b[2].get<float>() - b[0].get<float>(), b[3].get<float>() - b[1].get<float>()};
            x.score = d.at("s").get<float>();
            x.cls = d.at("c").get<int>();
            if (d.contains("k")) {
                auto& k = d["k"];
                for (size_t i = 0; i + 2 < k.size(); i += 3) x.kpts.emplace_back(k[i].get<float>(), k[i + 1].get<float>(), k[i + 2].get<float>());
            }
            dets.push_back(std::move(x));
        }
        const double srv_infer = jget<double>(j["t"], "infer", 0.0);
        t.pre = (t1 - t0) * 1e3;
        t.infer = srv_infer;
        t.net = (t2 - t1) * 1e3 - srv_infer;
        t.post = (mono_sec() - t2) * 1e3;
        t.tpu = jget<double>(j["t"], "tpu", 0.0);
        return dets;
    }

    void infer(const cv::Mat&, std::vector<Tensor>&) override {
        throw std::runtime_error("remote: используйте infer_remote");
    }

    json remote_bench(int iters) override {
        auto cli = lease();
        cli->set_read_timeout(600);  // замер на сервере идёт дольше обычного запроса
        auto res = cli->Get(prefix_ + "/bench/" + model_ + "?iters=" + std::to_string(iters), headers_);
        cli->set_read_timeout(timeout_);
        if (!res || res->status != 200) return nullptr;
        return json::parse(res->body);
    }

    std::string name() const override { return name_; }

private:
    std::string model_, prefix_, name_;
    std::string origin_;
    std::mutex pool_mu_;
    std::vector<std::unique_ptr<httplib::Client>> pool_;
    httplib::Headers headers_;
    int quality_ = 90, timeout_ = 5;
};

}  // namespace

std::unique_ptr<Backend> make_remote_backend(const std::string& model, const json& opts) {
    return std::make_unique<RemoteBackend>(model, opts);
}
