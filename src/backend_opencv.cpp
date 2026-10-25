// Бэкенд OpenCV DNN: ONNX на CPU (или CUDA, если OpenCV собран с ней).
// Нужен для отладки на ноутбуке без ускорителя и для сравнения с LinQ.
#include <stdexcept>

#include <opencv2/dnn.hpp>

#include "model.hpp"

namespace {

class OpenCVBackend : public Backend {
public:
    OpenCVBackend(const std::string& path, const json& opts) {
        net_ = cv::dnn::readNetFromONNX(path);
        if (net_.empty()) throw std::runtime_error("OpenCV DNN: не удалось загрузить " + path);
        if (jget<std::string>(opts, "opencv_target", "cpu") == "cuda") {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        out_names_ = net_.getUnconnectedOutLayersNames();
    }

    void infer(const cv::Mat& rgb, std::vector<Tensor>& outs) override {
        cv::Mat blob = cv::dnn::blobFromImage(rgb);  // NCHW float32
        net_.setInput(blob);
        std::vector<cv::Mat> res;
        net_.forward(res, out_names_);
        outs.clear();
        for (size_t i = 0; i < res.size(); ++i) {
            const cv::Mat& m = res[i];
            Tensor t;
            t.name = out_names_[i];
            for (int d = 0; d < m.dims; ++d) t.shape.push_back(m.size[d]);
            cv::Mat c = m.isContinuous() ? m : m.clone();
            t.data.assign(c.ptr<float>(), c.ptr<float>() + c.total());
            outs.push_back(std::move(t));
        }
    }

    std::string name() const override { return "opencv-dnn"; }

private:
    cv::dnn::Net net_;
    std::vector<std::string> out_names_;
};

}  // namespace

std::unique_ptr<Backend> make_opencv_backend(const std::string& path, const json& opts) {
    return std::make_unique<OpenCVBackend>(path, opts);
}
