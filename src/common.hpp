#pragma once
// Общие типы и утилиты сервиса.
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "json.hpp"

using json = nlohmann::json;

// Детекция в координатах исходного кадра.
struct Det {
    cv::Rect2f box;
    float score = 0.f;
    int cls = 0;
    std::string label;
    std::vector<cv::Point3f> kpts;  // x, y, conf (только у pose-модели)
};

// Выход сети в «логической» раскладке: 4D -> (1, C, H, W), 3D -> (1, C, N).
struct Tensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;
    int dim(int i) const { return i < (int)shape.size() ? shape[i] : 1; }
};

// Замер времени этапов (мс).
struct Timings {
    double pre = 0, infer = 0, post = 0;
    double net = 0;  // удалённый инференс: сеть + JPEG + декодирование на сервере
    double tpu = 0;  // время самого ускорителя LinQ (отчёт воркера облака / замер libtpu), 0 — неизвестно
};

inline double mono_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline int64_t wall_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

inline std::string iso_local(int64_t ms) {
    std::time_t t = ms / 1000;
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return os.str();
}

// Минуты от полуночи по местному времени.
inline int local_minutes(int64_t ms) {
    std::time_t t = ms / 1000;
    std::tm tm{};
    localtime_r(&t, &tm);
    return tm.tm_hour * 60 + tm.tm_min;
}

inline float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}

inline cv::Point2f foot(const cv::Rect2f& r) { return {r.x + r.width / 2, r.y + r.height}; }
inline cv::Point2f center(const cv::Rect2f& r) { return {r.x + r.width / 2, r.y + r.height / 2}; }

template <class T>
T jget(const json& j, const char* key, T def) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    try {
        return it->get<T>();
    } catch (...) {
        return def;
    }
}
