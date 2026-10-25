#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace {

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

std::string join_path(const std::string& base, const std::string& p) {
    if (p.empty() || p[0] == '/' || base.empty()) return p;
    return base + "/" + p;
}

}  // namespace

// ------------------------------------------------------------------ Backend
void Backend::dump_input(const cv::Mat& rgb, const std::string& path) {
    // по умолчанию NCHW float32
    std::vector<cv::Mat> ch;
    cv::split(rgb, ch);
    std::ofstream f(path, std::ios::binary);
    for (auto& c : ch) {
        cv::Mat cc = c.isContinuous() ? c : c.clone();
        f.write(reinterpret_cast<const char*>(cc.ptr<float>()), cc.total() * sizeof(float));
    }
}

// ---------------------------------------------------------------- ModelMeta
ModelMeta ModelMeta::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("не открыть meta: " + path);
    json j = json::parse(f);
    ModelMeta m;
    m.name = jget<std::string>(j, "name", "model");
    m.kind = jget<std::string>(j, "kind", "detect");
    m.classes = jget<std::vector<std::string>>(j, "classes", {});
    m.imgsz = jget<int>(j, "imgsz", 640);
    m.strides = jget<std::vector<int>>(j, "strides", {8, 16, 32});
    m.reg_max = jget<int>(j, "reg_max", 16);
    m.cls_has_sigmoid = jget<bool>(j, "cls_has_sigmoid", true);
    if (j.contains("kpt_shape") && j["kpt_shape"].is_array()) {
        m.nk = j["kpt_shape"][0].get<int>() * j["kpt_shape"][1].get<int>();
        m.kdim = j["kpt_shape"][1].get<int>();
    }
    return m;
}

// ---------------------------------------------------------------- YoloModel
YoloModel::YoloModel(const std::string& id, const json& cfg, const json& backend_cfg, const std::string& base) : id_(id) {
    meta_ = ModelMeta::load(join_path(base, cfg.at("meta").get<std::string>()));
    std::string btype = jget<std::string>(cfg, "backend", jget<std::string>(backend_cfg, "type", "opencv"));
    std::string path = btype == "remote" ? jget<std::string>(cfg, "remote_name", id)
                                         : join_path(base, jget<std::string>(cfg, btype == "linq" ? "tpu" : "onnx", ""));
    if (path.empty()) throw std::runtime_error("модель " + id + ": не задан путь для бэкенда " + btype);
    backend_ = make_backend(btype, path, backend_cfg);
    conf_ = jget<float>(cfg, "conf", 0.3f);
    iou_ = jget<float>(cfg, "iou", 0.5f);
    max_det_ = jget<int>(cfg, "max_det", 300);
    auto keep = jget<std::vector<std::string>>(cfg, "keep", {});
    if (!keep.empty()) {
        keep_.assign(meta_.classes.size(), 0);
        for (size_t i = 0; i < meta_.classes.size(); ++i)
            if (std::find(keep.begin(), keep.end(), meta_.classes[i]) != keep.end()) keep_[i] = 1;
    }
}

cv::Mat YoloModel::letterbox(const cv::Mat& bgr, float& r, cv::Point2f& pad) const {
    const int s = meta_.imgsz;
    const int h = bgr.rows, w = bgr.cols;
    r = std::min(float(s) / h, float(s) / w);
    int nw = (int)std::round(w * r), nh = (int)std::round(h * r);
    float dw = (s - nw) / 2.f, dh = (s - nh) / 2.f;
    cv::Mat resized;
    if (nw != w || nh != h)
        cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    else
        resized = bgr;
    int top = (int)std::round(dh - 0.1f), bottom = (int)std::round(dh + 0.1f);
    int left = (int)std::round(dw - 0.1f), right = (int)std::round(dw + 0.1f);
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    pad = cv::Point2f((float)left, (float)top);
    return padded;
}

cv::Mat YoloModel::preprocess(const cv::Mat& bgr, float& r, cv::Point2f& pad) const {
    cv::Mat rgb, f;
    cv::cvtColor(letterbox(bgr, r, pad), rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);
    return f;
}

// Детекции в координатах входа сети -> координаты кадра, фильтр классов, подписи.
std::vector<Det> YoloModel::finish(std::vector<Det> dets, float r, cv::Point2f pad, cv::Size orig) const {
    std::vector<Det> res;
    const float W = (float)orig.width, H = (float)orig.height;
    for (auto& d : dets) {
        if (d.cls < 0 || d.cls >= (int)meta_.classes.size()) continue;
        if (!keep_.empty() && !keep_[d.cls]) continue;
        float x1 = std::clamp((d.box.x - pad.x) / r, 0.f, W), y1 = std::clamp((d.box.y - pad.y) / r, 0.f, H);
        float x2 = std::clamp((d.box.x + d.box.width - pad.x) / r, 0.f, W);
        float y2 = std::clamp((d.box.y + d.box.height - pad.y) / r, 0.f, H);
        if (x2 - x1 < 2 || y2 - y1 < 2) continue;
        d.box = {x1, y1, x2 - x1, y2 - y1};
        d.label = meta_.classes[d.cls];
        for (auto& k : d.kpts) k = {(k.x - pad.x) / r, (k.y - pad.y) / r, k.z};
        res.push_back(std::move(d));
    }
    return res;
}

std::vector<Det> YoloModel::detect(const cv::Mat& bgr, Timings* t) {
    double t0 = mono_sec();
    if (backend_->remote()) {  // без блокировки: удалённый бэкенд держит пул соединений, запросы идут параллельно
        float r;
        cv::Point2f pad;
        cv::Mat lb = letterbox(bgr, r, pad);
        const double t1 = mono_sec();
        Timings rt;
        auto raw = backend_->infer_remote(lb, conf_, iou_, rt);
        const double t2 = mono_sec();
        auto dets = finish(std::move(raw), r, pad, bgr.size());
        if (t) {
            t->pre = (t1 - t0) * 1e3 + rt.pre;  // letterbox + JPEG
            t->infer = rt.infer;               // сеть на ускорителе (по замеру сервера)
            t->net = rt.net;                   // передача + декодирование JPEG и выходов на сервере
            t->post = (mono_sec() - t2) * 1e3 + rt.post;
            t->tpu = rt.tpu;
        }
        return dets;
    }
    std::lock_guard<std::mutex> lk(mu_);  // локальные бэкенды (cv::dnn, libtpu) — по одному запросу
    t0 = mono_sec();
    float r;
    cv::Point2f pad;
    cv::Mat inp = preprocess(bgr, r, pad);
    double t1 = mono_sec();
    std::vector<Tensor> outs;
    backend_->infer(inp, outs);
    double t2 = mono_sec();
    auto dets = decode(outs, r, pad, bgr.size());
    double t3 = mono_sec();
    if (t) {
        t->pre = (t1 - t0) * 1e3;
        t->infer = (t2 - t1) * 1e3;
        t->post = (t3 - t2) * 1e3;
        if (backend_name().rfind("linq", 0) == 0) t->tpu = t->infer;
    }
    return dets;
}

void YoloModel::dump_input(const cv::Mat& bgr, const std::string& path) {
    float r;
    cv::Point2f pad;
    backend_->dump_input(preprocess(bgr, r, pad), path);
}

std::vector<Det> YoloModel::decode(const std::vector<Tensor>& outs, float r, cv::Point2f pad, cv::Size orig) const {
    const int nc = (int)meta_.classes.size();
    const int nk = meta_.kind == "pose" ? meta_.nk : 0;
    const int kdim = meta_.kdim;
    const int rm = meta_.reg_max;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> cls_ids;
    std::vector<std::vector<cv::Point3f>> kpts;

    auto kept = [&](int c) { return keep_.empty() || keep_[c]; };

    if (outs.size() == 1 && outs[0].shape.size() == 3) {
        // ---- full: (1, 4+nc(+nk), N), всё уже декодировано
        const Tensor& o = outs[0];
        const int C = o.dim(1), N = o.dim(2);
        const float* p = o.data.data();
        for (int i = 0; i < N; ++i) {
            int best = -1;
            float bs = conf_;
            for (int c = 0; c < nc && 4 + c < C; ++c) {
                float s = p[(4 + c) * N + i];
                if (s > bs && kept(c)) bs = s, best = c;
            }
            if (best < 0) continue;
            float cx = p[i], cy = p[N + i], w = p[2 * N + i], h = p[3 * N + i];
            boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
            scores.push_back(bs);
            cls_ids.push_back(best);
            std::vector<cv::Point3f> kp;
            for (int k = 0; k < nk / kdim; ++k) {
                int base = 4 + nc + k * kdim;
                kp.emplace_back(p[base * N + i], p[(base + 1) * N + i], kdim > 2 ? p[(base + 2) * N + i] : 1.f);
            }
            kpts.push_back(std::move(kp));
        }
    } else {
        // ---- raw: по уровням box (4*reg_max), cls (nc), kpt (nk)
        struct Level { const Tensor *box = nullptr, *cls = nullptr, *kpt = nullptr; int H = 0, W = 0; };
        std::map<int, Level, std::greater<int>> lv;  // по убыванию H
        for (const auto& t : outs) {
            if (t.shape.size() != 4) continue;
            Level& L = lv[t.dim(2)];
            L.H = t.dim(2);
            L.W = t.dim(3);
            const int C = t.dim(1);
            const std::string& n = t.name;
            if (n.rfind("box", 0) == 0) L.box = &t;
            else if (n.rfind("cls", 0) == 0) L.cls = &t;
            else if (n.rfind("kpt", 0) == 0) L.kpt = &t;
            else if (C == 4 * rm && nc != 4 * rm) L.box = &t;
            else if (nk && C == nk && nc != nk) L.kpt = &t;
            else L.cls = &t;
        }
        std::vector<float> e(rm);
        for (auto& [H, L] : lv) {
            if (!L.box || !L.cls) continue;
            const int W = L.W, HW = H * W;
            const float s = float(meta_.imgsz) / H;
            const float* pb = L.box->data.data();
            const float* pc = L.cls->data.data();
            const float* pk = L.kpt ? L.kpt->data.data() : nullptr;
            const int ncl = L.cls->dim(1);
            for (int i = 0; i < HW; ++i) {
                int best = -1;
                float bs = conf_;
                for (int c = 0; c < ncl; ++c) {
                    float v = pc[c * HW + i];
                    if (!meta_.cls_has_sigmoid) v = sigmoid(v);
                    if (v > bs && kept(c)) bs = v, best = c;
                }
                if (best < 0) continue;
                float d[4];
                for (int k = 0; k < 4; ++k) {
                    float mx = -1e30f;
                    for (int b = 0; b < rm; ++b) mx = std::max(mx, e[b] = pb[(k * rm + b) * HW + i]);
                    float sum = 0, acc = 0;
                    for (int b = 0; b < rm; ++b) {
                        float ex = std::exp(e[b] - mx);
                        sum += ex;
                        acc += ex * b;
                    }
                    d[k] = acc / sum;
                }
                const int gx = i % W, gy = i / W;
                const float cx = gx + 0.5f, cy = gy + 0.5f;
                float x1 = (cx - d[0]) * s, y1 = (cy - d[1]) * s, x2 = (cx + d[2]) * s, y2 = (cy + d[3]) * s;
                boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
                scores.push_back(bs);
                cls_ids.push_back(best);
                std::vector<cv::Point3f> kp;
                if (pk) {
                    for (int k = 0; k < nk / kdim; ++k) {
                        float kx = pk[(k * kdim) * HW + i], ky = pk[(k * kdim + 1) * HW + i];
                        float kv = kdim > 2 ? sigmoid(pk[(k * kdim + 2) * HW + i]) : 1.f;
                        kp.emplace_back((kx * 2.f + gx) * s, (ky * 2.f + gy) * s, kv);
                    }
                }
                kpts.push_back(std::move(kp));
            }
        }
    }

    auto keep = nms(boxes, scores, cls_ids, iou_, max_det_);
    std::vector<Det> res;
    res.reserve(keep.size());
    const float W = (float)orig.width, H = (float)orig.height;
    for (int k : keep) {
        cv::Rect2f b = boxes[k];
        float x1 = std::clamp((b.x - pad.x) / r, 0.f, W), y1 = std::clamp((b.y - pad.y) / r, 0.f, H);
        float x2 = std::clamp((b.x + b.width - pad.x) / r, 0.f, W), y2 = std::clamp((b.y + b.height - pad.y) / r, 0.f, H);
        if (x2 - x1 < 2 || y2 - y1 < 2) continue;
        Det dt;
        dt.box = {x1, y1, x2 - x1, y2 - y1};
        dt.score = scores[k];
        dt.cls = cls_ids[k];
        dt.label = dt.cls < (int)meta_.classes.size() ? meta_.classes[dt.cls] : std::to_string(dt.cls);
        for (auto p : kpts[k]) dt.kpts.emplace_back((p.x - pad.x) / r, (p.y - pad.y) / r, p.z);
        res.push_back(std::move(dt));
    }
    return res;
}

std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     const std::vector<int>& cls, float iou_thr, int max_det) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<int> keep;
    std::vector<char> removed(boxes.size(), 0);
    for (size_t a = 0; a < order.size() && (int)keep.size() < max_det; ++a) {
        int i = order[a];
        if (removed[i]) continue;
        keep.push_back(i);
        for (size_t b = a + 1; b < order.size(); ++b) {
            int j = order[b];
            if (!removed[j] && cls[j] == cls[i] && iou(boxes[i], boxes[j]) >= iou_thr) removed[j] = 1;
        }
    }
    return keep;
}
