#pragma once
// Модель YOLO (detect / pose / world) поверх сменного бэкенда инференса.
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common.hpp"

// Бэкенд считает только сеть. На вход — кадр после letterbox: CV_32FC3, RGB, 0..1, HxW.
class Backend {
public:
    virtual ~Backend() = default;
    virtual void infer(const cv::Mat& rgb_f32, std::vector<Tensor>& outs) = 0;
    virtual std::string name() const = 0;
    // Сохранить входной тензор в раскладке бэкенда (для сверки с примерами SDK).
    virtual void dump_input(const cv::Mat& rgb_f32, const std::string& path);

    // Удалённый бэкенд: сервер сам считает сеть, декодирование и NMS, возвращает детекции
    // в координатах входа сети (после letterbox).
    virtual bool remote() const { return false; }
    virtual std::vector<Det> infer_remote(const cv::Mat& letterboxed_bgr, float conf, float iou, Timings& t) {
        (void)letterboxed_bgr, (void)conf, (void)iou, (void)t;
        return {};
    }
    virtual json remote_bench(int iters) {
        (void)iters;
        return nullptr;
    }
};

// type: "opencv" | "linq" | "remote"; path: .onnx, .tpu или имя модели на сервере; opts: параметры бэкенда
std::unique_ptr<Backend> make_backend(const std::string& type, const std::string& path, const json& opts);

struct ModelMeta {
    std::string name, kind = "detect";
    std::vector<std::string> classes;
    int imgsz = 640;
    std::vector<int> strides{8, 16, 32};
    int reg_max = 16;
    int nk = 0, kdim = 3;  // pose: 17 точек x 3
    bool cls_has_sigmoid = true;
    static ModelMeta load(const std::string& path);
};

class YoloModel {
public:
    // cfg: {"meta", "onnx", "tpu", "backend", "conf", "iou", "keep": [...], "max_det"}
    YoloModel(const std::string& id, const json& cfg, const json& backend_cfg, const std::string& base_dir);

    std::vector<Det> detect(const cv::Mat& bgr, Timings* t = nullptr);
    void dump_input(const cv::Mat& bgr, const std::string& path);

    const ModelMeta& meta() const { return meta_; }
    const std::string& id() const { return id_; }
    std::string backend_name() const { return backend_->name(); }
    bool is_pose() const { return meta_.kind == "pose"; }
    json remote_bench(int iters) { return backend_->remote_bench(iters); }

private:
    cv::Mat letterbox(const cv::Mat& bgr, float& r, cv::Point2f& pad) const;
    cv::Mat preprocess(const cv::Mat& bgr, float& r, cv::Point2f& pad) const;
    std::vector<Det> finish(std::vector<Det> dets, float r, cv::Point2f pad, cv::Size orig) const;
    std::vector<Det> decode(const std::vector<Tensor>& outs, float r, cv::Point2f pad, cv::Size orig) const;

    std::string id_;
    ModelMeta meta_;
    std::unique_ptr<Backend> backend_;
    float conf_ = 0.3f, iou_ = 0.5f;
    int max_det_ = 300;
    std::vector<char> keep_;  // какие классы оставлять (пусто = все)
    std::mutex mu_;
};

std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     const std::vector<int>& cls, float iou_thr, int max_det);
