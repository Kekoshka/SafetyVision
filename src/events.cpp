#include "events.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

namespace fs = std::filesystem;

json Event::to_json() const {
    return {{"id", id},           {"camera", camera},       {"camera_name", camera_name}, {"type", type},
            {"title", title},     {"severity", severity},   {"zone", zone},               {"detail", detail},
            {"track_id", track_id}, {"ts_start", ts_start}, {"ts_alert", ts_alert},       {"ts_end", ts_end},
            {"status", status},   {"comment", comment},     {"status_ts", status_ts},     {"box", box},
            {"clip_frames", clip_frames}, {"time", iso_local(ts_alert)}};
}

Event Event::from_json(const json& j) {
    Event e;
    e.id = jget<int64_t>(j, "id", 0);
    e.camera = jget<std::string>(j, "camera", "");
    e.camera_name = jget<std::string>(j, "camera_name", "");
    e.type = jget<std::string>(j, "type", "");
    e.title = jget<std::string>(j, "title", "");
    e.severity = jget<std::string>(j, "severity", "warning");
    e.zone = jget<std::string>(j, "zone", "");
    e.detail = jget<std::string>(j, "detail", "");
    e.track_id = jget<int>(j, "track_id", -1);
    e.ts_start = jget<int64_t>(j, "ts_start", 0);
    e.ts_alert = jget<int64_t>(j, "ts_alert", 0);
    e.ts_end = jget<int64_t>(j, "ts_end", 0);
    e.status = jget<std::string>(j, "status", "new");
    e.comment = jget<std::string>(j, "comment", "");
    e.status_ts = jget<int64_t>(j, "status_ts", 0);
    e.box = jget<std::vector<float>>(j, "box", {});
    e.clip_frames = jget<int>(j, "clip_frames", 0);
    return e;
}

// ------------------------------------------------------------------ EventStore
EventStore::EventStore(const std::string& dir) : dir_(dir), path_(dir + "/events.jsonl") {
    fs::create_directories(dir_ + "/events");
    std::ifstream f(path_);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        try {
            json r = json::parse(line);
            std::string op = jget<std::string>(r, "op", "");
            if (op == "new") {
                Event e = Event::from_json(r["event"]);
                idx_[e.id] = events_.size();
                next_id_ = std::max(next_id_, e.id + 1);
                events_.push_back(std::move(e));
            } else if (op == "status" || op == "end") {
                auto it = idx_.find(jget<int64_t>(r, "id", 0));
                if (it == idx_.end()) continue;
                Event& e = events_[it->second];
                if (op == "end") {
                    e.ts_end = jget<int64_t>(r, "ts_end", 0);
                } else {
                    e.status = jget<std::string>(r, "status", e.status);
                    e.comment = jget<std::string>(r, "comment", e.comment);
                    e.status_ts = jget<int64_t>(r, "ts", 0);
                }
            }
        } catch (const std::exception& ex) {
            std::cerr << "[events] пропущена строка журнала: " << ex.what() << "\n";
        }
    }
    std::cerr << "[events] загружено событий: " << events_.size() << "\n";
}

void EventStore::append(const json& rec) {
    std::ofstream f(path_, std::ios::app);
    f << rec.dump() << "\n";
}

std::string EventStore::event_dir(int64_t id) const { return dir_ + "/events/" + std::to_string(id); }

int64_t EventStore::add(Event e) {
    std::lock_guard<std::mutex> lk(mu_);
    e.id = next_id_++;
    fs::create_directories(event_dir(e.id));
    append({{"op", "new"}, {"event", e.to_json()}});
    idx_[e.id] = events_.size();
    events_.push_back(e);
    return e.id;
}

void EventStore::close(int64_t id, int64_t ts_end) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = idx_.find(id);
    if (it == idx_.end()) return;
    events_[it->second].ts_end = ts_end;
    append({{"op", "end"}, {"id", id}, {"ts_end", ts_end}});
}

bool EventStore::set_status(int64_t id, const std::string& status, const std::string& comment) {
    static const std::set<std::string> ok = {"new", "confirmed", "false", "forwarded"};
    if (!ok.count(status)) return false;
    std::lock_guard<std::mutex> lk(mu_);
    auto it = idx_.find(id);
    if (it == idx_.end()) return false;
    Event& e = events_[it->second];
    e.status = status;
    e.comment = comment;
    e.status_ts = wall_ms();
    append({{"op", "status"}, {"id", id}, {"status", status}, {"comment", comment}, {"ts", e.status_ts}});
    return true;
}

bool EventStore::get(int64_t id, Event& out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = idx_.find(id);
    if (it == idx_.end()) return false;
    out = events_[it->second];
    return true;
}

std::vector<Event> EventStore::query(const EventQuery& q) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<Event> res;
    for (auto it = events_.rbegin(); it != events_.rend() && (int)res.size() < q.limit; ++it) {
        const Event& e = *it;
        if (q.since_id && e.id <= q.since_id) break;
        if (!q.type.empty() && e.type != q.type) continue;
        if (!q.camera.empty() && e.camera != q.camera) continue;
        if (!q.zone.empty() && e.zone != q.zone) continue;
        if (!q.severity.empty() && e.severity != q.severity) continue;
        if (!q.status.empty() && e.status != q.status) continue;
        if (q.from && e.ts_alert < q.from) continue;
        if (q.to && e.ts_alert > q.to) continue;
        res.push_back(e);
    }
    return res;
}

json EventStore::stats() {
    std::lock_guard<std::mutex> lk(mu_);
    json by_type = json::object(), by_cam = json::object(), by_zone = json::object(), by_day = json::object();
    std::vector<int> by_hour(24, 0);
    std::map<std::string, std::pair<int, int>> fb;  // type -> (confirmed, false)
    double delay_sum = 0;
    int delay_n = 0;
    for (auto& e : events_) {
        by_type[e.type] = jget<int>(by_type, e.type.c_str(), 0) + 1;
        std::string cam = e.camera_name.empty() ? e.camera : e.camera_name;
        by_cam[cam] = jget<int>(by_cam, cam.c_str(), 0) + 1;
        if (!e.zone.empty()) by_zone[e.zone] = jget<int>(by_zone, e.zone.c_str(), 0) + 1;
        std::string day = iso_local(e.ts_alert).substr(0, 10);
        by_day[day] = jget<int>(by_day, day.c_str(), 0) + 1;
        by_hour[local_minutes(e.ts_alert) / 60]++;
        if (e.status == "confirmed") fb[e.type].first++;
        if (e.status == "false") fb[e.type].second++;
        if (e.ts_start && e.ts_alert >= e.ts_start) delay_sum += double(e.ts_alert - e.ts_start), ++delay_n;
    }
    json precision = json::object();
    for (auto& [t, cf] : fb) {
        int n = cf.first + cf.second;
        precision[t] = {{"confirmed", cf.first}, {"false", cf.second}, {"precision", n ? double(cf.first) / n : 0.0}};
    }
    json titles = json::object();
    for (auto& [k, v] : event_types()) titles[k] = {{"title", v.title}, {"severity", v.severity}};
    return {{"total", events_.size()}, {"by_type", by_type},   {"by_camera", by_cam},
            {"by_zone", by_zone},      {"by_day", by_day},     {"by_hour", by_hour},
            {"precision", precision},  {"types", titles},
            {"mean_alert_delay_ms", delay_n ? delay_sum / delay_n : 0.0}};
}

// ------------------------------------------------------------------ Deduper
void Deduper::update(double t, int64_t wall, const std::vector<Hit>& hits, const std::map<std::string, double>& confirm,
                     const std::vector<int>& alive, const EmitFn& emit, const CloseFn& close) {
    for (auto& [k, s] : st_) s.active = false;
    for (const auto& h : hits) {
        auto it = st_.find(h.key);
        if (it == st_.end()) {
            State s;
            s.first = t;
            s.wall_first = wall;
            it = st_.emplace(h.key, s).first;
        }
        State& s = it->second;
        if (s.closed && cfg_.mode == "cooldown" && t - s.last > cfg_.cooldown_sec) {
            s = State{};  // повтор после паузы — новое нарушение
            s.first = t;
            s.wall_first = wall;
        }
        s.hit = h;
        s.last = t;
        s.active = true;
    }
    std::set<int> alive_set(alive.begin(), alive.end());
    for (auto it = st_.begin(); it != st_.end();) {
        State& s = it->second;
        const int tid = s.hit.track_id;
        const bool gone = tid >= 0 && !alive_set.count(tid);
        auto cit = confirm.find(s.hit.type);
        const double need = cit != confirm.end() ? cit->second : 2.0;

        if (s.active && !s.emitted && !s.closed && t - s.first >= need) {
            s.event_id = emit(s.hit, s.wall_first);
            s.emitted = true;
        }
        if (!s.active) {
            if (!s.emitted && t - s.last > cfg_.gap_sec) {  // не подтвердилось — сброс
                it = st_.erase(it);
                continue;
            }
            if (s.emitted && !s.closed && t - s.last > cfg_.release_sec) {
                close(s.event_id, wall - int64_t((t - s.last) * 1000));
                s.closed = true;
            }
        }
        if (gone) {
            if (s.emitted && !s.closed) close(s.event_id, wall - int64_t((t - s.last) * 1000));
            it = st_.erase(it);
            continue;
        }
        if (s.closed && tid < 0 && t - s.last > cfg_.cooldown_sec) {
            it = st_.erase(it);
            continue;
        }
        ++it;
    }
}

std::map<std::string, bool> Deduper::active() const {
    std::map<std::string, bool> m;
    for (auto& [k, s] : st_)
        if (s.active) m[k] = s.emitted;
    return m;
}

std::vector<Hit> Deduper::active_hits() const {
    std::vector<Hit> v;
    for (auto& [k, s] : st_)
        if (s.active) v.push_back(s.hit);
    return v;
}
