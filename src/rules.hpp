#pragma once
// Правила нарушений. Каждое правило — отдельный класс, регистрируется по имени (REGISTER_RULE).
// Новое нарушение = новый класс + запись в конфиге, без переделки конвейера.
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common.hpp"
#include "tracker.hpp"

struct Zone {
    std::string id, name, kind = "danger";  // danger | fence | under_load | ...
    std::vector<cv::Point2f> poly;          // нормированные координаты 0..1
    int from_min = -1, to_min = -1;         // активность по времени суток (минуты), -1 = всегда
    bool active_at(int64_t wall) const;
    bool contains(cv::Point2f p, cv::Size frame) const;
    static Zone from_json(const json& j);
    json to_json() const;
};

struct SceneCtx {
    double t = 0;
    int64_t wall = 0;
    cv::Size frame;
    const std::vector<Track>* persons = nullptr;
    const std::vector<Track>* vehicles = nullptr;
    const std::vector<Det>* objects = nullptr;
    const std::vector<Zone>* zones = nullptr;
    std::set<std::string> labels;  // какие классы вообще умеют детектировать загруженные модели
};

// Активное в этом кадре нарушение (ещё не событие: событие появляется после подтверждения).
struct Hit {
    std::string key;   // уникален в пределах камеры: правило + трек/зона/ячейка
    std::string type;  // тип события (fall, running, zone_danger, ...)
    int track_id = -1;
    std::string zone, detail;
    cv::Rect2f box;                  // главный объект (человек)
    std::vector<cv::Rect2f> related; // связанные объекты (техника, телефон, огонь)
};

struct EventType {
    std::string title, severity;  // severity: critical | high | warning
};
const std::map<std::string, EventType>& event_types();
std::string event_title(const std::string& type);
std::string event_severity(const std::string& type);

class Rule {
public:
    virtual ~Rule() = default;
    virtual void eval(const SceneCtx& ctx, std::vector<Hit>& out) = 0;
    std::string id;
    json p;              // параметры правила
    double confirm_sec;  // сколько секунд условие должно держаться, чтобы стать событием

    using Factory = std::function<std::unique_ptr<Rule>()>;
    static std::map<std::string, Factory>& registry();
    static std::unique_ptr<Rule> create(const std::string& id, const json& params);
};

#define REGISTER_RULE(NAME, CLASS)                                                           \
    static const bool reg_##CLASS = [] {                                                     \
        Rule::registry()[NAME] = [] { return std::unique_ptr<Rule>(new CLASS()); };          \
        return true;                                                                         \
    }();
