#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

double Track::speed_h(double window) const {
    if (hist.size() < 2) return 0;
    const auto& now = hist.back();
    const TrackPoint* then = &hist.front();
    for (const auto& p : hist)
        if (now.t - p.t <= window) {
            then = &p;
            break;
        }
    double dt = now.t - then->t;
    if (dt < window * 0.5) return 0;
    cv::Point2f d = foot(now.box) - foot(then->box);
    double h = 0;
    int n = 0;
    for (const auto& p : hist)
        if (now.t - p.t <= window) h += p.box.height, ++n;
    h = n ? h / n : now.box.height;
    return h > 1 ? std::hypot(d.x, d.y) / h / dt : 0;
}

double Track::motion(double window) const {
    if (hist.empty()) return 0;
    const auto& now = hist.back();
    double mx = 0;
    double sz = std::max(now.box.width, now.box.height);
    if (hist.front().t > now.t - window * 0.9) return 1e9;  // истории мало — «неподвижность» не доказана
    for (const auto& p : hist) {
        if (now.t - p.t > window) continue;
        cv::Point2f d = center(now.box) - center(p.box);
        mx = std::max(mx, (double)std::hypot(d.x, d.y));
    }
    return sz > 1 ? mx / sz : 0;
}

void Tracker::update(const std::vector<Det>& dets, double t) {
    for (auto& tr : tracks_) tr.updated = false;
    std::vector<char> used(dets.size(), 0);

    // 1) по IoU
    std::vector<std::tuple<float, int, int>> pairs;
    for (size_t i = 0; i < tracks_.size(); ++i)
        for (size_t j = 0; j < dets.size(); ++j)
            if (tracks_[i].cls == dets[j].cls) {
                float v = iou(tracks_[i].box, dets[j].box);
                if (v >= iou_thr_) pairs.emplace_back(v, (int)i, (int)j);
            }
    // 2) по расстоянию центров (для быстрых объектов), с меньшим приоритетом
    for (size_t i = 0; i < tracks_.size(); ++i)
        for (size_t j = 0; j < dets.size(); ++j)
            if (tracks_[i].cls == dets[j].cls) {
                cv::Point2f d = center(tracks_[i].box) - center(dets[j].box);
                float lim = 0.6f * std::max(tracks_[i].box.height, dets[j].box.height);
                float dist = std::hypot(d.x, d.y);
                float hr = tracks_[i].box.height / std::max(1.f, dets[j].box.height);
                if (dist < lim && hr > 0.6f && hr < 1.6f) pairs.emplace_back(0.2f * (1 - dist / lim), (int)i, (int)j);
            }
    std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });

    for (auto& [s, i, j] : pairs) {
        Track& tr = tracks_[i];
        if (tr.updated || used[j]) continue;
        used[j] = 1;
        tr.updated = true;
        tr.box = dets[j].box;
        tr.score = dets[j].score;
        tr.kpts = dets[j].kpts;
        tr.t_last = t;
        if (++tr.hits >= min_hits_) tr.confirmed = true;
        tr.hist.push_back({t, tr.box});
    }
    for (size_t j = 0; j < dets.size(); ++j) {
        if (used[j]) continue;
        Track tr;
        tr.id = next_id_++;
        tr.cls = dets[j].cls;
        tr.label = dets[j].label;
        tr.box = dets[j].box;
        tr.score = dets[j].score;
        tr.kpts = dets[j].kpts;
        tr.t_first = tr.t_last = t;
        tr.hits = 1;
        tr.updated = true;
        tr.confirmed = min_hits_ <= 1;
        tr.hist.push_back({t, tr.box});
        tracks_.push_back(std::move(tr));
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& tr) { return t - tr.t_last > max_age_; }),
                  tracks_.end());
    for (auto& tr : tracks_)
        while (!tr.hist.empty() && t - tr.hist.front().t > 10.0) tr.hist.pop_front();
}
