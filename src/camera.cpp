#include "camera.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "web.hpp"

namespace fs = std::filesystem;

namespace {

const std::vector<std::pair<int, int>> kSkeleton = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                                                    {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {0, 5},  {0, 6}};

// Подписи на кадре — латиницей (шрифты OpenCV не умеют кириллицу).
std::string code_of(const std::string& type) {
    static const std::map<std::string, std::string> m = {
        {"fall", "FALL"},       {"running", "RUNNING"},     {"vehicle_proximity", "VEHICLE"},
        {"phone", "PHONE"},     {"smoking", "SMOKING"},     {"fire", "FIRE"},
        {"no_helmet", "NO HELMET"}, {"no_vest", "NO VEST"}, {"zone_danger", "DANGER ZONE"},
        {"zone_fence", "FENCE"}, {"zone_under_load", "UNDER LOAD"}};
    auto it = m.find(type);
    return it != m.end() ? it->second : type;
}

cv::Scalar zone_color(const std::string& kind) {
    if (kind == "danger") return {0, 0, 255};
    if (kind == "fence") return {0, 165, 255};
    if (kind == "under_load") return {255, 0, 255};
    return {0, 255, 255};
}

void label(cv::Mat& img, const std::string& text, cv::Point org, const cv::Scalar& bg, double scale) {
    int base = 0;
    const int th = std::max(1, (int)std::round(scale * 1.5));
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, th, &base);
    org.x = std::clamp(org.x, 0, std::max(0, img.cols - sz.width - 6));
    org.y = std::clamp(org.y, sz.height + 4, std::max(sz.height + 4, img.rows - base - 2));
    cv::rectangle(img, {org.x, org.y - sz.height - 4}, {org.x + sz.width + 4, org.y + base - 2}, bg, cv::FILLED);
    cv::putText(img, text, {org.x + 2, org.y - 2}, cv::FONT_HERSHEY_SIMPLEX, scale, {255, 255, 255}, th, cv::LINE_AA);
}

std::string encode_jpeg(const cv::Mat& img, int max_w, int quality) {
    cv::Mat s = img;
    if (img.cols > max_w) cv::resize(img, s, {max_w, int(std::round(img.rows * double(max_w) / img.cols))}, 0, 0, cv::INTER_AREA);
    std::vector<uchar> buf;
    cv::imencode(".jpg", s, buf, {cv::IMWRITE_JPEG_QUALITY, quality});
    return std::string(buf.begin(), buf.end());
}

bool is_live(const std::string& s) {
    return s.rfind("rtsp://", 0) == 0 || s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0 ||
           (!s.empty() && std::all_of(s.begin(), s.end(), ::isdigit));
}

}  // namespace

Camera::Camera(const json& cfg, Services& svc) : cfg_(cfg), svc_(svc) {
    id_ = cfg.at("id").get<std::string>();
    name_ = jget<std::string>(cfg, "name", id_);
    source_ = jget<std::string>(cfg, "source", "");
    if (!source_.empty() && !is_live(source_) && source_.rfind("images:", 0) != 0 && source_[0] != '/')
        source_ = svc.base_dir + "/" + source_;
    if (source_.rfind("images:", 0) == 0 && source_.size() > 7 && source_[7] != '/')
        source_ = "images:" + svc.base_dir + "/" + source_.substr(7);
    loop_ = jget<bool>(cfg, "loop", true);
    realtime_ = jget<bool>(cfg, "realtime", true);
    process_fps_ = jget<double>(cfg, "process_fps", 10.0);
    pipeline_ = std::max(1, jget<int>(cfg, "pipeline", svc_.pipeline));
    image_interval_ = jget<double>(cfg, "image_interval_sec", 3.0);
    model_ids_ = jget<std::vector<std::string>>(cfg, "models", {});
    if (model_ids_.empty())
        for (auto& [k, v] : svc.models) model_ids_.push_back(k);
    for (auto& z : cfg.value("zones", json::array())) zones_.push_back(Zone::from_json(z));
    Deduper::Cfg dc = svc.dedup;
    if (cfg.contains("dedup")) {
        dc.mode = jget<std::string>(cfg["dedup"], "mode", dc.mode);
        dc.release_sec = jget<double>(cfg["dedup"], "release_sec", dc.release_sec);
        dc.cooldown_sec = jget<double>(cfg["dedup"], "cooldown_sec", dc.cooldown_sec);
    }
    dedup_ = std::make_unique<Deduper>(dc);
    build_rules();
}

Camera::~Camera() { stop(); }

void Camera::build_rules() {
    json defaults = svc_.cfg.value("rules", json::object());
    json over = cfg_.value("rules", json::object());
    for (auto& [rid, p0] : defaults.items()) {
        json p = p0;
        if (over.contains(rid)) p.merge_patch(over[rid]);
        if (!jget<bool>(p, "enabled", true)) continue;
        auto r = Rule::create(rid, p);
        if (!r) {
            std::cerr << "[" << id_ << "] неизвестное правило: " << rid << "\n";
            continue;
        }
        rules_.push_back(std::move(r));
    }
    // время подтверждения по типу события
    for (auto& r : rules_) {
        if (r->id == "zone") {
            for (auto& [t, _] : event_types())
                if (t.rfind("zone_", 0) == 0) confirm_[t] = r->confirm_sec;
        } else {
            confirm_[r->id == "vehicle_proximity" ? "vehicle_proximity" : r->id] = r->confirm_sec;
        }
    }
}

void Camera::start() {
    running_ = true;
    grab_th_ = std::thread(&Camera::grab_loop, this);
    proc_th_ = std::thread(&Camera::proc_loop, this);
}

void Camera::stop() {
    if (!running_.exchange(false)) return;
    slot_cv_.notify_all();
    if (grab_th_.joinable()) grab_th_.join();
    if (proc_th_.joinable()) proc_th_.join();
}

// ------------------------------------------------------------------ захват
void Camera::grab_loop() {
    const bool live = is_live(source_);
    const bool block = !realtime_ && !live;  // обработка файла без пропусков кадров
    auto push = [&](Frame f) {
        std::unique_lock<std::mutex> lk(slot_mu_);
        if (block) slot_cv_.wait(lk, [&] { return !has_frame_ || !running_; });
        slot_ = std::move(f);
        has_frame_ = true;
        frames_in_++;
        lk.unlock();
        slot_cv_.notify_all();
    };
    auto sleep_until = [](double target) {
        double d = target - mono_sec();
        if (d > 0) std::this_thread::sleep_for(std::chrono::duration<double>(d));
    };
    const int64_t wall0 = wall_ms();

    if (source_.rfind("images:", 0) == 0) {  // набор стоп-кадров
        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(source_.substr(7))) {
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        state_ = files.empty() ? "error" : "running";
        if (files.empty()) error_ = "нет изображений в " + source_;
        double t = 0, start = mono_sec();
        while (running_ && !files.empty()) {
            for (auto& p : files) {
                cv::Mat img = cv::imread(p.string());
                if (img.empty()) continue;
                for (double k = 0; k < image_interval_ && running_; k += 1.0 / process_fps_) {
                    if (realtime_) sleep_until(start + t);
                    push({img, t, mono_sec(), realtime_ ? wall_ms() : wall0 + int64_t(t * 1000)});
                    t += 1.0 / process_fps_;
                }
                if (!running_) break;
            }
            if (!loop_) break;
        }
    } else {
        double t_off = 0;
        while (running_) {
            cv::VideoCapture cap;
            bool ok = (!source_.empty() && std::all_of(source_.begin(), source_.end(), ::isdigit))
                          ? cap.open(std::stoi(source_))
                          : cap.open(source_);
            if (!ok) {
                state_ = "error";
                error_ = "не открыть источник " + source_;
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }
            state_ = "running";
            error_.clear();
            double fps = cap.get(cv::CAP_PROP_FPS);
            if (!(fps > 1 && fps < 121)) fps = 25;
            const double start = mono_sec();
            int64_t n = 0;
            cv::Mat img;
            double t = t_off;
            while (running_ && cap.read(img)) {
                t = live ? t_off + (mono_sec() - start) : t_off + n / fps;
                ++n;
                if (!live && realtime_) sleep_until(start + (t - t_off));
                push({img.clone(), t, mono_sec(), (realtime_ || live) ? wall_ms() : wall0 + int64_t(t * 1000)});
            }
            t_off = t + 1.0 / fps;
            if (!running_) break;
            if (live) {
                state_ = "reconnecting";
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            if (!loop_) break;
        }
    }
    std::lock_guard<std::mutex> lk(slot_mu_);
    eof_ = true;
    if (state_ == "running") state_ = "finished";
    slot_cv_.notify_all();
}

// ------------------------------------------------------------------ обработка
// Конвейер: при удалённом инференсе (LinQ в облаке) кадр идёт до ускорителя и обратно сотни мс.
// Чтобы FPS не упирался в задержку, к ускорителю одновременно уходят до pipeline_ кадров, а
// трекинг/правила/события получают результаты строго в порядке кадров.
void Camera::proc_loop() {
    struct Job {
        std::shared_ptr<Frame> f;
        std::future<Inferred> fut;
    };
    std::deque<Job> q;
    auto fail = [&](const std::exception& e) {
        {
            std::lock_guard<std::mutex> lk(met_mu_);
            error_ = e.what();
        }
        std::cerr << "[" << id_ << "] ошибка обработки: " << e.what() << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    };
    auto finish_front = [&] {
        Job j = std::move(q.front());
        q.pop_front();
        try {
            Inferred r = j.fut.get();
            process(*j.f, r);
        } catch (const std::exception& e) {
            fail(e);
        }
    };
    double next_proc = -1;
    while (running_) {
        while (!q.empty() && q.front().fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) finish_front();
        auto f = std::make_shared<Frame>();
        {
            std::unique_lock<std::mutex> lk(slot_mu_);
            const bool got = slot_cv_.wait_for(lk, std::chrono::milliseconds(q.empty() ? 500 : 10),
                                               [&] { return has_frame_ || eof_ || !running_; });
            if (!got) continue;
            if (!running_ || (!has_frame_ && eof_)) break;
            *f = std::move(slot_);
            has_frame_ = false;
        }
        slot_cv_.notify_all();
        if (realtime_) {
            if (next_proc >= 0 && f->t + 1e-6 < next_proc && f->t > next_proc - 5) continue;
            next_proc = f->t + 1.0 / process_fps_;
        }
        if (pipeline_ <= 1) {
            try {
                Inferred r = infer(*f);
                process(*f, r);
            } catch (const std::exception& e) {
                fail(e);
            }
            continue;
        }
        while ((int)q.size() >= pipeline_) finish_front();
        q.push_back({f, std::async(std::launch::async, [this, f] { return infer(*f); })});
    }
    while (!q.empty()) {  // дообработать кадры, ушедшие на ускоритель (или дождаться их при остановке)
        if (running_) finish_front();
        else q.front().fut.wait(), q.pop_front();
    }
}

Camera::Inferred Camera::infer(const Frame& f) {
    Inferred r;
    r.t0 = mono_sec();
    // модели одного кадра — параллельно (при удалённом инференсе запросы идут одновременно)
    std::vector<std::pair<std::string, std::future<std::vector<Det>>>> jobs;
    for (auto& mid : model_ids_) {
        auto it = svc_.models.find(mid);
        if (it == svc_.models.end()) continue;
        Timings* tp = &r.tms[mid];
        auto model = it->second;
        jobs.emplace_back(mid, std::async(std::launch::async, [model, tp, &f] { return model->detect(f.img, tp); }));
    }
    for (auto& [mid, fut] : jobs) r.dets.emplace_back(mid, fut.get());
    return r;
}

void Camera::process(Frame& f, Inferred& r) {
    const double t0 = r.t0;
    std::vector<Det> persons, objects;
    bool have_pose = false;
    auto& tms = r.tms;
    for (auto& [mid, d] : r.dets) {
        auto it = svc_.models.find(mid);
        if (it->second->is_pose()) {
            have_pose = true;
            for (auto& x : d) x.label = "person", x.cls = 0, persons.push_back(std::move(x));
        } else {
            for (auto& x : d) objects.push_back(std::move(x));
        }
    }
    std::vector<Det> others, vehicles;
    for (auto& o : objects) {
        if (o.label == "person") {
            if (!have_pose) persons.push_back(o), persons.back().cls = 0;
        } else if (svc_.vehicle_classes.count(o.label)) {
            vehicles.push_back(o);
            vehicles.back().cls = 0;  // метка техники может «мигать» между truck/car — трек общий
        } else {
            others.push_back(o);
        }
    }
    const double t_inf = mono_sec();
    persons_.update(persons, f.t);
    vehicles_.update(vehicles, f.t);

    std::vector<Zone> zones;
    {
        std::lock_guard<std::mutex> lk(zones_mu_);
        zones = zones_;
    }
    SceneCtx ctx;
    ctx.t = f.t;
    ctx.wall = f.wall;
    ctx.frame = f.img.size();
    ctx.persons = &persons_.tracks();
    ctx.vehicles = &vehicles_.tracks();
    ctx.objects = &others;
    ctx.zones = &zones;
    for (auto& mid : model_ids_) {
        auto it = svc_.models.find(mid);
        if (it != svc_.models.end())
            for (auto& c : it->second->meta().classes) ctx.labels.insert(c);
    }
    std::vector<Hit> hits;
    for (auto& r : rules_) r->eval(ctx, hits);

    std::vector<int> alive;
    for (auto& t : persons_.tracks()) alive.push_back(t.id);
    for (auto& t : vehicles_.tracks()) alive.push_back(-1000 - t.id);

    const cv::Mat& frame = f.img;
    dedup_->update(
        f.t, f.wall, hits, confirm_, alive,
        [&](const Hit& h, int64_t wall_start) {
            Hit hh = h;
            int64_t id = 0;
            {
                // кадр-доказательство с выделенным нарушителем
                Event e;
                e.camera = id_;
                e.camera_name = name_;
                e.type = h.type;
                e.title = event_title(h.type);
                e.severity = event_severity(h.type);
                e.zone = h.zone;
                e.detail = h.detail;
                e.track_id = h.track_id;
                e.ts_start = wall_start;
                e.ts_alert = f.wall;
                const float W = (float)frame.cols, H = (float)frame.rows;
                e.box = {h.box.x / W, h.box.y / H, h.box.width / W, h.box.height / H};
                std::deque<std::pair<int64_t, std::string>> ring;
                {
                    std::lock_guard<std::mutex> lk(pub_mu_);
                    ring = ring_;
                }
                e.clip_frames = (int)ring.size() + 1;
                id = svc_.store->add(e);
                const std::string dir = svc_.store->event_dir(id);
                cv::Mat snap = draw(frame, hits, dedup_->active(), &hh);
                std::string snap_jpg = encode_jpeg(snap, 1280, 85);
                std::ofstream(dir + "/snapshot.jpg", std::ios::binary) << snap_jpg;
                int k = 0;
                for (auto& [w, jpg] : ring) {
                    char fn[32];
                    std::snprintf(fn, sizeof fn, "/clip_%02d.jpg", k++);
                    std::ofstream(dir + fn, std::ios::binary) << jpg;
                }
                char fn[32];
                std::snprintf(fn, sizeof fn, "/clip_%02d.jpg", k);
                std::ofstream(dir + fn, std::ios::binary) << encode_jpeg(snap, 640, 75);
                e.id = id;
                std::cerr << "[ALERT] " << iso_local(e.ts_alert) << " " << name_ << ": " << e.title
                          << (e.zone.empty() ? "" : " (" + e.zone + ")") << " #" << id << "\n";
                if (!svc_.webhook.empty()) post_json_async(svc_.webhook, e.to_json().dump());
            }
            return id;
        },
        [&](int64_t id, int64_t wall_end) { svc_.store->close(id, wall_end); });
    const double t_rules = mono_sec();

    // кадр для веб-интерфейса
    cv::Mat annot = draw(frame, dedup_->active_hits(), dedup_->active(), nullptr);
    // объекты (телефон, огонь, ...) — поверх
    const double sc = std::max(0.5, frame.cols / 1280.0);
    for (auto& o : others) {
        cv::rectangle(annot, o.box, {255, 255, 0}, std::max(1, int(2 * sc)));
        char b[64];
        std::snprintf(b, sizeof b, "%s %.2f", o.label.c_str(), o.score);
        label(annot, b, {int(o.box.x), int(o.box.y)}, {160, 120, 0}, 0.5 * sc);
    }
    std::string jpg = encode_jpeg(annot, 960, 75);
    {
        std::lock_guard<std::mutex> lk(pub_mu_);
        jpeg_annot_ = jpg;
        last_raw_ = f.img;
        frame_size_ = f.img.size();
        if (f.t - last_ring_t_ >= 0.25 || f.t < last_ring_t_) {
            ring_.emplace_back(f.wall, encode_jpeg(annot, 640, 70));
            if (ring_.size() > 12) ring_.pop_front();
            last_ring_t_ = f.t;
        }
    }
    const double t_end = mono_sec();

    std::lock_guard<std::mutex> lk(met_mu_);
    for (auto& [mid, tm] : tms) {
        m_pre_[mid].add(tm.pre);
        m_inf_[mid].add(tm.infer);
        m_post_[mid].add(tm.post);
        m_net_[mid].add(tm.net);
        m_tpu_[mid].add(tm.tpu);
    }
    m_rules_.add((t_rules - t_inf) * 1e3);
    m_total_.add((t_end - t0) * 1e3);
    m_latency_.add((t_end - f.grab) * 1e3);
    if (last_proc_mono_ > 0) {
        double dt = t_end - last_proc_mono_;
        if (dt > 0) m_fps_.add(1.0 / dt, 0.05);
    }
    last_proc_mono_ = t_end;
    frames_++;
    n_persons_ = (int)persons.size();
    n_vehicles_ = (int)vehicles.size();
    n_objects_ = (int)others.size();
}

cv::Mat Camera::draw(const cv::Mat& img, const std::vector<Hit>& active, const std::map<std::string, bool>& state,
                     const Hit* focus) {
    cv::Mat out = img.clone();
    const double sc = std::max(0.5, img.cols / 1280.0);
    const int th = std::max(1, int(std::round(2 * sc)));
    const int64_t now = wall_ms();

    std::vector<Zone> zones;
    {
        std::lock_guard<std::mutex> lk(zones_mu_);
        zones = zones_;
    }
    cv::Mat overlay = out.clone();
    int zi = 0;
    for (auto& z : zones) {
        ++zi;
        if (z.poly.size() < 3) continue;
        std::vector<cv::Point> pts;
        for (auto& p : z.poly) pts.emplace_back(int(p.x * img.cols), int(p.y * img.rows));
        cv::Scalar col = z.active_at(now) ? zone_color(z.kind) : cv::Scalar(128, 128, 128);
        cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{pts}, col);
        cv::polylines(out, std::vector<std::vector<cv::Point>>{pts}, true, col, th);
    }
    cv::addWeighted(overlay, 0.18, out, 0.82, 0, out);
    zi = 0;
    for (auto& z : zones) {
        ++zi;
        if (z.poly.empty()) continue;
        cv::Scalar col = z.active_at(now) ? zone_color(z.kind) : cv::Scalar(128, 128, 128);
        std::string k = z.kind;
        std::transform(k.begin(), k.end(), k.begin(), ::toupper);
        label(out, "Z" + std::to_string(zi) + " " + k, {int(z.poly[0].x * img.cols), int(z.poly[0].y * img.rows)}, col,
              0.5 * sc);
    }

    for (auto& v : vehicles_.tracks()) {
        if (!v.confirmed || !v.updated) continue;
        cv::rectangle(out, v.box, {255, 128, 0}, th);
        label(out, v.label + " #" + std::to_string(v.id), {int(v.box.x), int(v.box.y)}, {200, 100, 0}, 0.5 * sc);
    }

    std::map<int, std::pair<std::string, bool>> person_alert;  // track -> (код, подтверждено)
    for (auto& h : active) {
        bool emitted = false;
        auto it = state.find(h.key);
        if (it != state.end()) emitted = it->second;
        if (h.track_id >= 0) {
            auto& pa = person_alert[h.track_id];
            pa.first += (pa.first.empty() ? "" : ", ") + code_of(h.type);
            pa.second = pa.second || emitted;
        } else {
            cv::rectangle(out, h.box, emitted ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 255), th + 1);
            label(out, code_of(h.type), {int(h.box.x), int(h.box.y)}, emitted ? cv::Scalar(0, 0, 200) : cv::Scalar(0, 140, 200),
                  0.6 * sc);
        }
        for (auto& r : h.related) cv::rectangle(out, r, {0, 140, 255}, th + 1);
    }

    for (auto& t : persons_.tracks()) {
        if (!t.confirmed || !t.updated) continue;
        for (auto& [a, b] : kSkeleton)
            if (a < (int)t.kpts.size() && b < (int)t.kpts.size() && t.kpts[a].z > 0.4f && t.kpts[b].z > 0.4f)
                cv::line(out, {int(t.kpts[a].x), int(t.kpts[a].y)}, {int(t.kpts[b].x), int(t.kpts[b].y)}, {80, 220, 80},
                         std::max(1, th - 1), cv::LINE_AA);
        auto it = person_alert.find(t.id);
        cv::Scalar col = {0, 200, 0};
        std::string txt = "#" + std::to_string(t.id);
        if (it != person_alert.end()) {
            col = it->second.second ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 255);
            txt += " " + it->second.first;
        }
        cv::rectangle(out, t.box, col, it != person_alert.end() ? th + 1 : th);
        label(out, txt, {int(t.box.x), int(t.box.y)}, col * 0.8, 0.55 * sc);
    }

    if (focus) {
        cv::rectangle(out, focus->box, {0, 0, 255}, th * 3);
        label(out, "ALERT: " + code_of(focus->type), {int(focus->box.x), int(focus->box.y + focus->box.height + 30 * sc)},
              {0, 0, 220}, 0.8 * sc);
    }
    double fps;
    {
        std::lock_guard<std::mutex> lk(met_mu_);
        fps = m_fps_.v;
    }
    char hud[160];
    std::snprintf(hud, sizeof hud, "%s  %s  %.1f FPS", id_.c_str(), iso_local(now).c_str() + 11, fps);
    label(out, hud, {8, int(28 * sc)}, {40, 40, 40}, 0.6 * sc);
    return out;
}

// ------------------------------------------------------------------ API для веба
json Camera::status() {
    std::lock_guard<std::mutex> lk(met_mu_);
    json models = json::object();
    for (auto& mid : model_ids_) {
        auto it = svc_.models.find(mid);
        if (it == svc_.models.end()) continue;
        models[mid] = {{"backend", it->second->backend_name()},
                       {"pre_ms", m_pre_[mid].v},
                       {"infer_ms", m_inf_[mid].v},
                       {"post_ms", m_post_[mid].v},
                       {"net_ms", m_net_[mid].v},
                       {"tpu_ms", m_tpu_[mid].v}};
    }
    json active = json::array();
    for (auto& h : dedup_->active_hits()) active.push_back({{"type", h.type}, {"title", event_title(h.type)}, {"track", h.track_id}});
    return {{"id", id_},
            {"name", name_},
            {"source", source_},
            {"state", state_},
            {"error", error_},
            {"fps", m_fps_.v},
            {"process_fps_target", process_fps_},
            {"frames", frames_},
            {"frames_in", frames_in_},
            {"width", frame_size_.width},
            {"height", frame_size_.height},
            {"models", models},
            {"rules_ms", m_rules_.v},
            {"total_ms", m_total_.v},
            {"latency_ms", m_latency_.v},
            {"persons", n_persons_},
            {"vehicles", n_vehicles_},
            {"objects", n_objects_},
            {"active", active},
            {"rules", [&] {
                 json r = json::array();
                 for (auto& x : rules_) r.push_back({{"id", x->id}, {"confirm_sec", x->confirm_sec}});
                 return r;
             }()}};
}

json Camera::config() {
    json c = cfg_;
    c["zones"] = json::array();
    std::lock_guard<std::mutex> lk(zones_mu_);
    for (auto& z : zones_) c["zones"].push_back(z.to_json());
    return c;
}

bool Camera::jpeg(std::string& out, bool raw) {
    std::lock_guard<std::mutex> lk(pub_mu_);
    if (raw) {
        if (last_raw_.empty()) return false;
        out = encode_jpeg(last_raw_, 1280, 85);
        return true;
    }
    if (jpeg_annot_.empty()) return false;
    out = jpeg_annot_;
    return true;
}

void Camera::set_zones(const std::vector<Zone>& z) {
    std::lock_guard<std::mutex> lk(zones_mu_);
    zones_ = z;
}

std::vector<Zone> Camera::zones() {
    std::lock_guard<std::mutex> lk(zones_mu_);
    return zones_;
}
