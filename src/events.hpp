#pragma once
// Журнал событий (JSONL на диске + индекс в памяти) и подавление повторов.
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "common.hpp"
#include "rules.hpp"

struct Event {
    int64_t id = 0;
    std::string camera, camera_name, type, title, severity, zone, detail;
    int track_id = -1;
    int64_t ts_start = 0;  // начало нарушения (первый кадр, где условие выполнилось)
    int64_t ts_alert = 0;  // момент оповещения (после подтверждения)
    int64_t ts_end = 0;    // окончание нарушения (0 — длится / не закрыто)
    std::string status = "new";  // new | confirmed | false | forwarded
    std::string comment;
    int64_t status_ts = 0;
    std::vector<float> box;  // x, y, w, h в долях кадра
    int clip_frames = 0;
    json to_json() const;
    static Event from_json(const json& j);
};

struct EventQuery {
    std::string type, camera, zone, severity, status;
    int64_t from = 0, to = 0, since_id = 0;
    int limit = 200;
};

class EventStore {
public:
    explicit EventStore(const std::string& data_dir);
    int64_t add(Event e);  // присваивает id, пишет на диск
    void close(int64_t id, int64_t ts_end);
    bool set_status(int64_t id, const std::string& status, const std::string& comment);
    bool get(int64_t id, Event& out);
    std::vector<Event> query(const EventQuery& q);
    json stats();
    std::string event_dir(int64_t id) const;
    const std::string& data_dir() const { return dir_; }

private:
    void append(const json& rec);
    std::string dir_, path_;
    std::vector<Event> events_;
    std::map<int64_t, size_t> idx_;
    int64_t next_id_ = 1;
    std::mutex mu_;
};

// Конечный автомат «нарушение -> событие» для одной камеры.
//  * условие должно держаться confirm_sec (кратковременные пропуски до gap_sec прощаются);
//  * на пару (трек, тип) — одно событие, пока человек в кадре (mode "track"),
//    или новое не раньше чем через cooldown_sec (mode "cooldown");
//  * событие закрывается, когда условие не выполняется release_sec.
class Deduper {
public:
    struct Cfg {
        std::string mode = "track";
        double gap_sec = 0.7, release_sec = 5.0, cooldown_sec = 60.0;
    };
    using EmitFn = std::function<int64_t(const Hit&, int64_t wall_start)>;
    using CloseFn = std::function<void(int64_t id, int64_t wall_end)>;

    explicit Deduper(Cfg c) : cfg_(c) {}
    void update(double t, int64_t wall, const std::vector<Hit>& hits, const std::map<std::string, double>& confirm,
                const std::vector<int>& alive_tracks, const EmitFn& emit, const CloseFn& close);
    // состояние для отрисовки: ключ -> подтверждено ли событие
    std::map<std::string, bool> active() const;
    std::vector<Hit> active_hits() const;

private:
    struct State {
        Hit hit;
        double first = 0, last = 0;
        int64_t wall_first = 0;
        bool active = false, emitted = false, closed = false;
        int64_t event_id = 0;
    };
    Cfg cfg_;
    std::map<std::string, State> st_;
};
