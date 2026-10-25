#pragma once
// Лёгкий многообъектный трекер: жадное сопоставление по IoU + по расстоянию центров
// (быстрые объекты, например бегущий человек, между кадрами могут не пересекаться).
#include <deque>
#include <vector>

#include "common.hpp"

struct TrackPoint {
    double t;
    cv::Rect2f box;
};

struct Track {
    int id = 0;
    int cls = 0;
    std::string label;
    cv::Rect2f box;
    float score = 0;
    std::vector<cv::Point3f> kpts;
    double t_first = 0, t_last = 0;
    int hits = 0;
    bool confirmed = false;
    bool updated = false;  // обновлён в текущем кадре
    std::deque<TrackPoint> hist;

    // Смещение точки опоры (низ рамки) за окно window_sec, в долях высоты рамки за секунду.
    double speed_h(double window_sec) const;
    // Максимальное смещение центра за окно, в долях max(w,h).
    double motion(double window_sec) const;
};

class Tracker {
public:
    explicit Tracker(float iou_thr = 0.25f, double max_age = 1.5, int min_hits = 3)
        : iou_thr_(iou_thr), max_age_(max_age), min_hits_(min_hits) {}
    void update(const std::vector<Det>& dets, double t);
    const std::vector<Track>& tracks() const { return tracks_; }

private:
    std::vector<Track> tracks_;
    int next_id_ = 1;
    float iou_thr_;
    double max_age_;
    int min_hits_;
};
