#include "rules.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

// ------------------------------------------------------------------- типы событий
const std::map<std::string, EventType>& event_types() {
    static const std::map<std::string, EventType> t = {
        {"fall", {"Падение / неподвижное лежание", "critical"}},
        {"vehicle_proximity", {"Опасное сближение с техникой", "critical"}},
        {"fire", {"Открытый огонь / дым", "critical"}},
        {"zone_under_load", {"Нахождение под грузом / конвейером", "critical"}},
        {"smoking", {"Курение вне отведённых мест", "high"}},
        {"zone_fence", {"Перелезание через ограждение", "high"}},
        {"zone_danger", {"Нахождение в опасной зоне", "high"}},
        {"running", {"Бег в рабочей зоне", "warning"}},
        {"phone", {"Телефон при работе / перемещении", "warning"}},
        {"no_helmet", {"Нет каски", "warning"}},
        {"no_vest", {"Нет сигнального жилета", "warning"}},
    };
    return t;
}
std::string event_title(const std::string& type) {
    auto it = event_types().find(type);
    return it != event_types().end() ? it->second.title : type;
}
std::string event_severity(const std::string& type) {
    auto it = event_types().find(type);
    return it != event_types().end() ? it->second.severity : "warning";
}

// ------------------------------------------------------------------- зоны
static int parse_hhmm(const std::string& s) {
    int h = 0, m = 0;
    if (std::sscanf(s.c_str(), "%d:%d", &h, &m) >= 1) return h * 60 + m;
    return -1;
}

Zone Zone::from_json(const json& j) {
    Zone z;
    z.id = jget<std::string>(j, "id", "zone");
    z.name = jget<std::string>(j, "name", z.id);
    z.kind = jget<std::string>(j, "kind", "danger");
    for (auto& p : j.value("points", json::array())) z.poly.emplace_back(p[0].get<float>(), p[1].get<float>());
    z.from_min = parse_hhmm(jget<std::string>(j, "active_from", ""));
    z.to_min = parse_hhmm(jget<std::string>(j, "active_to", ""));
    return z;
}

json Zone::to_json() const {
    json j = {{"id", id}, {"name", name}, {"kind", kind}, {"points", json::array()}};
    auto r4 = [](float v) { return std::round(double(v) * 1e4) / 1e4; };
    for (auto& p : poly) j["points"].push_back({r4(p.x), r4(p.y)});
    auto fmt = [](int m) {
        char b[8];
        std::snprintf(b, sizeof b, "%02d:%02d", m / 60, m % 60);
        return std::string(b);
    };
    if (from_min >= 0) j["active_from"] = fmt(from_min);
    if (to_min >= 0) j["active_to"] = fmt(to_min);
    return j;
}

bool Zone::active_at(int64_t wall) const {
    if (from_min < 0 && to_min < 0) return true;
    int m = local_minutes(wall);
    int a = from_min < 0 ? 0 : from_min, b = to_min < 0 ? 24 * 60 : to_min;
    return a <= b ? (m >= a && m < b) : (m >= a || m < b);  // через полночь
}

bool Zone::contains(cv::Point2f p, cv::Size frame) const {
    if (poly.size() < 3) return false;
    std::vector<cv::Point2f> pts;
    for (auto& q : poly) pts.emplace_back(q.x * frame.width, q.y * frame.height);
    return cv::pointPolygonTest(pts, p, false) >= 0;
}

// ------------------------------------------------------------------- реестр
std::map<std::string, Rule::Factory>& Rule::registry() {
    static std::map<std::string, Factory> r;
    return r;
}

std::unique_ptr<Rule> Rule::create(const std::string& id, const json& params) {
    auto it = registry().find(id);
    if (it == registry().end()) return nullptr;
    auto r = it->second();
    r->id = id;
    r->p = params;
    r->confirm_sec = jget<double>(params, "confirm_sec", 2.0);
    return r;
}

// ------------------------------------------------------------------- помощники
namespace {

// COCO-точки: 0 нос, 5/6 плечи, 9/10 запястья, 11/12 бёдра, 13/14 колени, 15/16 лодыжки
bool kp_ok(const Track& t, int i, float c = 0.4f) { return i < (int)t.kpts.size() && t.kpts[i].z >= c; }
cv::Point2f kp(const Track& t, int i) { return {t.kpts[i].x, t.kpts[i].y}; }

// Угол корпуса (плечи -> бёдра) от вертикали, градусы; -1 если точки не видны.
double torso_angle(const Track& t) {
    if (!(kp_ok(t, 5) && kp_ok(t, 6) && kp_ok(t, 11) && kp_ok(t, 12))) return -1;
    cv::Point2f s = (kp(t, 5) + kp(t, 6)) * 0.5f, h = (kp(t, 11) + kp(t, 12)) * 0.5f;
    cv::Point2f d = s - h;
    return std::atan2(std::fabs(d.x), std::fabs(d.y)) * 180.0 / CV_PI;
}

bool is_lying(const Track& t, double ratio, double angle) {
    double a = torso_angle(t);
    if (a >= 0) return a > angle && t.box.width > t.box.height * 0.8;
    return t.box.width > t.box.height * ratio;
}

// Расстояние от объекта до головы/кистей человека в долях его роста (или -1, если не рядом).
double near_head_hands(const Track& person, const cv::Rect2f& obj) {
    cv::Point2f c = center(obj);
    cv::Rect2f grown(person.box.x - person.box.width * 0.15f, person.box.y - person.box.height * 0.1f,
                     person.box.width * 1.3f, person.box.height * 1.2f);
    if (!grown.contains(c)) return -1;
    const float h = std::max(1.f, person.box.height);
    double best = 1e9;
    for (int i : {0, 9, 10})
        if (kp_ok(person, i, 0.3f)) {
            cv::Point2f d = kp(person, i) - c;
            best = std::min(best, (double)std::hypot(d.x, d.y) / h);
        }
    if (best < 1e8) return best;
    return c.y < person.box.y + person.box.height * 0.75f ? 0.2 : -1;  // нет точек — верх рамки
}

bool person_ok(const Track& t, const json& p) {
    return t.confirmed && t.updated && t.box.height >= jget<float>(p, "min_person_px", 40.f);
}

}  // namespace

// ------------------------------------------------------------------- правила

// 07: падение и неподвижное лежание
class FallRule : public Rule {
    void eval(const SceneCtx& c, std::vector<Hit>& out) override {
        const double ratio = jget<double>(p, "lying_ratio", 1.15), angle = jget<double>(p, "torso_angle", 55);
        const double still_sec = jget<double>(p, "still_sec", 2.0), still_frac = jget<double>(p, "still_frac", 0.25);
        for (auto& t : *c.persons) {
            if (!person_ok(t, p) || !is_lying(t, ratio, angle)) continue;
            if (t.motion(still_sec) > still_frac) continue;
            out.push_back({"fall:" + std::to_string(t.id), "fall", t.id, "", "человек лежит и не двигается", t.box, {}});
        }
    }
};
REGISTER_RULE("fall", FallRule)

// 01: бег / резкие перемещения
class RunningRule : public Rule {
    void eval(const SceneCtx& c, std::vector<Hit>& out) override {
        const double win = jget<double>(p, "window_sec", 0.8), thr = jget<double>(p, "speed_h_per_s", 1.6);
        for (auto& t : *c.persons) {
            if (!person_ok(t, p)) continue;
            double v = t.speed_h(win);
            if (v < thr) continue;
            char d[64];
            std::snprintf(d, sizeof d, "скорость %.1f роста/с", v);
            out.push_back({"run:" + std::to_string(t.id), "running", t.id, "", d, t.box, {}});
        }
    }
};
REGISTER_RULE("running", RunningRule)

// 05 + опасные зоны: человек (точка опоры) внутри полигона активной зоны
class ZoneRule : public Rule {
    void eval(const SceneCtx& c, std::vector<Hit>& out) override {
        auto kinds = jget<std::vector<std::string>>(p, "kinds", {});
        for (auto& z : *c.zones) {
            if (!z.active_at(c.wall)) continue;
            if (!kinds.empty() && std::find(kinds.begin(), kinds.end(), z.kind) == kinds.end()) continue;
            for (auto& t : *c.persons) {
                if (!person_ok(t, p) || !z.contains(foot(t.box), c.frame)) continue;
                out.push_back({"zone:" + z.id + ":" + std::to_string(t.id), "zone_" + z.kind, t.id, z.name, "",
                               t.box, {}});
            }
        }
    }
};
REGISTER_RULE("zone", ZoneRule)

// 02: опасное сближение человека с движущейся техникой
class VehicleProximityRule : public Rule {
    void eval(const SceneCtx& c, std::vector<Hit>& out) override {
        const float k = jget<float>(p, "distance_person_heights", 0.5f);
        const bool need_move = jget<bool>(p, "require_moving", true);
        const double min_speed = jget<double>(p, "min_vehicle_speed", 0.15);
        for (auto& v : *c.vehicles) {
            if (!v.confirmed || !v.updated) continue;
            if (need_move && v.speed_h(1.0) < min_speed) continue;
            for (auto& t : *c.persons) {
                if (!person_ok(t, p)) continue;
                // водитель внутри техники — не нарушение
                if ((t.box & v.box).area() > 0.6f * t.box.area()) continue;
                float m = k * t.box.height;
                cv::Rect2f zone(v.box.x - m, v.box.y - m, v.box.width + 2 * m, v.box.height + 2 * m);
                if (!zone.contains(foot(t.box)) && (t.box & v.box).area() <= 0) continue;
                out.push_back({"prox:" + std::to_string(t.id), "vehicle_proximity", t.id, "",
                               "рядом " + v.label + (need_move ? " в движении" : ""), t.box, {v.box}});
            }
        }
    }
};
REGISTER_RULE("vehicle_proximity", VehicleProximityRule)

// 03 / 04: предмет у головы или в руках (телефон, сигарета)
class HandObjectRule : public Rule {
public:
    explicit HandObjectRule(std::string type, std::vector<std::string> def) : type_(std::move(type)), def_(std::move(def)) {}
    void eval(const SceneCtx& c, std::vector<Hit>& out) override {
        auto labels = jget<std::vector<std::string>>(p, "labels", def_);
        const float min_score = jget<float>(p, "min_score", 0.35f);
        const double max_dist = jget<double>(p, "max_dist_h", 0.35);
        for (auto& o : *c.objects) {
            if (o.score < min_score || std::find(labels.begin(), labels.end(), o.label) == labels.end()) continue;
            for (auto& t : *c.persons) {
                if (!person_ok(t, p)) continue;
                double d = near_head_hands(t, o.box);
                if (d < 0 || d > max_dist) continue;
                out.push_back({type_ + ":" + std::to_string(t.id), type_, t.id, "", o.label, t.box, {o.box}});
                break;
            }
        }
    }

private:
    std::string type_;
    std::vector<std::string> def_;
};
struct PhoneRule : HandObjectRule { PhoneRule() : HandObjectRule("phone", {"cell phone"}) {} };
struct SmokingRule : HandObjectRule { SmokingRule() : HandObjectRule("smoking", {"cigarette"}) {} };
REGISTER_RULE("phone", PhoneRule)
REGISTER_RULE("smoking", SmokingRule)

// 04: огонь / дым в кадре
class FireRule : public Rule {
    void eval(const SceneCtx& c, std::vector<Hit>& out) override {
        auto labels = jget<std::vector<std::string>>(p, "labels", {"fire", "smoke", "flame"});
        const float min_score = jget<float>(p, "min_score", 0.4f);
        for (auto& o : *c.objects) {
            if (o.score < min_score || std::find(labels.begin(), labels.end(), o.label) == labels.end()) continue;
            cv::Point2f ce = center(o.box);
            int gx = int(4 * ce.x / std::max(1, c.frame.width)), gy = int(4 * ce.y / std::max(1, c.frame.height));
            out.push_back({"fire:" + o.label + ":" + std::to_string(gx) + "," + std::to_string(gy), "fire", -1, "",
                           o.label, o.box, {}});
        }
    }
};
REGISTER_RULE("fire", FireRule)

// СИЗ: нет каски / жилета в нужной части фигуры (нужна модель СИЗ, например SH17)
class PPERule : public Rule {
public:
    PPERule(std::string type, std::vector<std::string> def, float y0, float y1)
        : type_(std::move(type)), def_(std::move(def)), y0_(y0), y1_(y1) {}
    void eval(const SceneCtx& c, std::vector<Hit>& out) override {
        auto labels = jget<std::vector<std::string>>(p, "labels", def_);
        bool available = false;
        for (auto& l : labels) available |= c.labels.count(l) > 0;
        if (!available) return;  // ни одна модель не умеет искать этот СИЗ
        const float min_px = jget<float>(p, "min_person_px", 90.f);
        for (auto& t : *c.persons) {
            if (!person_ok(t, p) || t.box.height < min_px) continue;
            cv::Rect2f reg(t.box.x - t.box.width * 0.1f, t.box.y + t.box.height * y0_, t.box.width * 1.2f,
                           t.box.height * (y1_ - y0_));
            bool found = false;
            for (auto& o : *c.objects)
                if (std::find(labels.begin(), labels.end(), o.label) != labels.end() && reg.contains(center(o.box)))
                    found = true;
            if (!found) out.push_back({type_ + ":" + std::to_string(t.id), type_, t.id, "", "", t.box, {}});
        }
    }

private:
    std::string type_;
    std::vector<std::string> def_;
    float y0_, y1_;
};
struct NoHelmetRule : PPERule { NoHelmetRule() : PPERule("no_helmet", {"helmet", "hardhat", "safety helmet"}, -0.1f, 0.3f) {} };
struct NoVestRule : PPERule { NoVestRule() : PPERule("no_vest", {"safety-vest", "vest", "safety vest"}, 0.15f, 0.75f) {} };
REGISTER_RULE("no_helmet", NoHelmetRule)
REGISTER_RULE("no_vest", NoVestRule)
