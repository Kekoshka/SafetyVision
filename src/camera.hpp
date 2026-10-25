#pragma once
// Конвейер одной камеры: захват -> модели -> трекинг -> правила -> события -> кадр для веб-интерфейса.
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "events.hpp"
#include "model.hpp"
#include "rules.hpp"
#include "tracker.hpp"

struct Services {
    json cfg;
    std::string base_dir, data_dir;
    std::map<std::string, std::shared_ptr<YoloModel>> models;
    EventStore* store = nullptr;
    std::set<std::string> vehicle_classes;
    Deduper::Cfg dedup;
    std::string webhook;
    int pipeline = 1;  // значение camera.pipeline по умолчанию (для удалённого инференса — 3)
};

struct Ema {
    double v = 0;
    bool init = false;
    void add(double x, double a = 0.1) {
        v = init ? v * (1 - a) + x * a : x;
        init = true;
    }
};

class Camera {
public:
    Camera(const json& cfg, Services& svc);
    ~Camera();
    void start();
    void stop();

    const std::string& id() const { return id_; }
    const std::string& name() const { return name_; }
    json status();
    json config();  // текущая конфигурация (с зонами)
    bool jpeg(std::string& out, bool raw);
    void set_zones(const std::vector<Zone>& z);
    std::vector<Zone> zones();

private:
    struct Frame {
        cv::Mat img;
        double t = 0;       // время для правил (видео или монотонное), с
        double grab = 0;    // момент захвата (монотонное), с
        int64_t wall = 0;   // настенное время, мс
    };
    struct Inferred {
        double t0 = 0;
        std::map<std::string, Timings> tms;
        std::vector<std::pair<std::string, std::vector<Det>>> dets;
    };
    void grab_loop();
    void proc_loop();
    Inferred infer(const Frame& f);           // все модели кадра (параллельно), можно из нескольких потоков
    void process(Frame& f, Inferred& r);      // трекинг, правила, события — строго по порядку кадров
    void build_rules();
    cv::Mat draw(const cv::Mat& img, const std::vector<Hit>& active, const std::map<std::string, bool>& state,
                 const Hit* focus);

    std::string id_, name_, source_;
    json cfg_;
    Services& svc_;
    bool loop_ = true, realtime_ = true;
    double process_fps_ = 10, image_interval_ = 2.0;
    int pipeline_ = 1;  // сколько кадров одновременно «в полёте» к ускорителю (для remote > 1)
    std::vector<std::string> model_ids_;

    std::vector<std::unique_ptr<Rule>> rules_;
    std::map<std::string, double> confirm_;
    Tracker persons_{0.25f, 1.5, 3}, vehicles_{0.2f, 2.0, 3};
    std::unique_ptr<Deduper> dedup_;

    std::mutex zones_mu_;
    std::vector<Zone> zones_;

    // одна «ячейка» между захватом и обработкой
    std::mutex slot_mu_;
    std::condition_variable slot_cv_;
    Frame slot_;
    bool has_frame_ = false, eof_ = false;

    std::thread grab_th_, proc_th_;
    std::atomic<bool> running_{false};

    // публикуемое состояние
    std::mutex pub_mu_;
    std::string jpeg_annot_;
    cv::Mat last_raw_;
    std::deque<std::pair<int64_t, std::string>> ring_;  // предыстория для клипа события
    double last_ring_t_ = 0;

    // метрики
    std::mutex met_mu_;
    std::map<std::string, Ema> m_pre_, m_inf_, m_post_, m_net_, m_tpu_;
    Ema m_rules_, m_total_, m_latency_, m_fps_;
    double last_proc_mono_ = 0;
    int64_t frames_ = 0, frames_in_ = 0;
    int n_persons_ = 0, n_vehicles_ = 0, n_objects_ = 0;
    std::string state_ = "starting", error_;
    cv::Size frame_size_;
};
