// Бэкенд LinQ TPU через C++ API libtpu (по примерам usage_example_cpp_api / usage_example_resnet50_opencv).
//
// program.tpu компилируется SDK из ONNX (python/export_models.py -> *_raw.onnx -> SDK -> *.tpu).
// Режим PackingMode::kFull: libtpu сама упаковывает float32-вход пользователя в int8-формат TPU
// и распаковывает выходы обратно во float32 (масштабы берутся из metadata.json программы).
#include <stdexcept>

#include "model.hpp"

#ifdef HAVE_LINQ

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <new>
#include <string_view>

#include <libtpu/device.hpp>
#include <libtpu/error_message.hpp>
#include <libtpu/program.hpp>
#include <libtpu/tensor_buffer_object.hpp>
#include <libtpu/tpu_types.hpp>
#include <tpu_packer/dtype.hpp>
#include <tpu_packer/tpu_packer.hpp>

namespace {

void alignedDeleter(uint8_t* p) { ::operator delete[](p, std::align_val_t(16)); }

// Одно устройство на имя (/dev/tpu0): несколько моделей загружаются как разные программы.
struct DeviceHolder {
    using Ptr = decltype(tpu::Device::create(std::declval<const std::string&>(), std::declval<tpu::ErrorMessage&>()));
    Ptr dev;
    std::mutex mu;
};

std::shared_ptr<DeviceHolder> get_device(const std::string& name) {
    static std::mutex m;
    static std::map<std::string, std::shared_ptr<DeviceHolder>> devices;
    std::lock_guard<std::mutex> lk(m);
    auto& d = devices[name];
    if (!d) {
        auto h = std::make_shared<DeviceHolder>();
        tpu::ErrorMessage err;
        h->dev = tpu::Device::create(name, err);
        if (!h->dev || h->dev->init(err) != tpu::ErrorCode::kSuccess)
            throw std::runtime_error("LinQ: не открыть устройство " + name + ": " + err.c_str());
        d = h;
    }
    return d;
}

// Описание тензора: порядок осей ("NCHW", "NHWC", "NC", ...) и размеры.
struct Layout {
    std::string order;
    std::vector<size_t> shape;
    bool f32 = true;
};

template <class Fmt>
Layout layout_of(const Fmt& fmt) {
    Layout l;
    for (size_t i = 0; i < fmt.axis_count; ++i) {
        l.order.push_back(static_cast<char>(fmt.order[i]));
        l.shape.push_back(static_cast<size_t>(fmt.shape[i]));
    }
    l.f32 = (fmt.dtype == tpu_util::DType::kFloat32);
    return l;
}

size_t axis(const Layout& l, char a) {
    auto p = l.order.find(a);
    return p == std::string::npos ? size_t(-1) : p;
}

// HWC float (RGB) -> буфер в раскладке входа программы.
void fill_input(const cv::Mat& rgb, const Layout& l, float* dst) {
    const size_t H = rgb.rows, W = rgb.cols, C = 3;
    const size_t ah = axis(l, 'H'), aw = axis(l, 'W'), ac = axis(l, 'C');
    if (ah == size_t(-1) || aw == size_t(-1) || ac == size_t(-1))
        throw std::runtime_error("LinQ: неожиданный порядок осей входа: " + l.order);
    if (l.shape[ah] != H || l.shape[aw] != W || l.shape[ac] != C)
        throw std::runtime_error("LinQ: размер входа программы не совпадает с imgsz модели");
    std::vector<size_t> stride(l.shape.size(), 1);
    for (int i = (int)l.shape.size() - 2; i >= 0; --i) stride[i] = stride[i + 1] * l.shape[i + 1];
    cv::Mat src = rgb.isContinuous() ? rgb : rgb.clone();
    const float* s = src.ptr<float>();
    if (l.order == "NHWC" || l.order == "HWC") {
        std::memcpy(dst, s, H * W * C * sizeof(float));
        return;
    }
    for (size_t y = 0; y < H; ++y)
        for (size_t x = 0; x < W; ++x)
            for (size_t c = 0; c < C; ++c)
                dst[y * stride[ah] + x * stride[aw] + c * stride[ac]] = s[(y * W + x) * C + c];
}

// Буфер выхода -> Tensor в логической раскладке (1, C, H, W) или как есть.
Tensor read_output(const std::string& name, const Layout& l, const float* src) {
    Tensor t;
    t.name = name;
    const size_t an = axis(l, 'N'), ac = axis(l, 'C'), ah = axis(l, 'H'), aw = axis(l, 'W');
    size_t total = 1;
    for (auto v : l.shape) total *= v;
    if (l.shape.size() == 4 && ac != size_t(-1) && ah != size_t(-1) && aw != size_t(-1)) {
        const size_t C = l.shape[ac], H = l.shape[ah], W = l.shape[aw];
        t.shape = {1, (int)C, (int)H, (int)W};
        t.data.resize(C * H * W);
        std::vector<size_t> stride(4, 1);
        for (int i = 2; i >= 0; --i) stride[i] = stride[i + 1] * l.shape[i + 1];
        (void)an;
        for (size_t c = 0; c < C; ++c)
            for (size_t y = 0; y < H; ++y)
                for (size_t x = 0; x < W; ++x)
                    t.data[(c * H + y) * W + x] = src[c * stride[ac] + y * stride[ah] + x * stride[aw]];
    } else {
        for (auto v : l.shape) t.shape.push_back((int)v);
        t.data.assign(src, src + total);
    }
    return t;
}

class LinqBackend : public Backend {
public:
    LinqBackend(const std::string& path, const json& opts) {
        tpu::ErrorMessage err;
        file_ = std::make_unique<tpu::ProgramFile>(path, err);
        if (!file_->isValid()) throw std::runtime_error("LinQ: не открыть " + path + ": " + err.c_str());
        prog_ = std::make_unique<tpu::Program>(*file_, err);
        if (!prog_->isValid()) throw std::runtime_error("LinQ: не создать программу: " + std::string(err.c_str()));

        const size_t nin = prog_->getTensorCount(tpu::TensorType::kInput);
        const size_t nout = prog_->getTensorCount(tpu::TensorType::kOutput);
        if (nin != 1) throw std::runtime_error("LinQ: ожидается 1 вход, в программе " + std::to_string(nin));
        in_layout_ = layout_of(prog_->getTensorFormat(tpu::TensorType::kInput, 0));
        if (!in_layout_.f32) throw std::runtime_error("LinQ: пользовательский тип входа не float32");
        for (size_t n = 0; n < nout; ++n) {
            out_names_.emplace_back(std::string(prog_->getTensorName(tpu::TensorType::kOutput, n)));
            out_layouts_.push_back(layout_of(prog_->getTensorFormat(tpu::TensorType::kOutput, n)));
        }

        in_ = std::make_unique<tpu::TensorBufferObject>(*prog_, tpu::TensorType::kInput);
        out_ = std::make_unique<tpu::TensorBufferObject>(*prog_, tpu::TensorType::kOutput);
        for (size_t n = 0; n < in_->getCount(); ++n)
            in_->setUserBufferPtr(n, new (std::align_val_t(16)) uint8_t[in_->getBufferSize(n)], alignedDeleter);
        for (size_t n = 0; n < out_->getCount(); ++n)
            out_->setUserBufferPtr(n, new (std::align_val_t(16)) uint8_t[out_->getBufferSize(n)], alignedDeleter);

        data_ = std::make_unique<tpu::ProgramData>(*prog_, *file_, err);
        if (!data_->isValid()) throw std::runtime_error("LinQ: не загрузить данные программы: " + std::string(err.c_str()));
        file_->close();

        dev_ = get_device(jget<std::string>(opts, "device", "/dev/tpu0"));
        std::lock_guard<std::mutex> lk(dev_->mu);
        if (dev_->dev->loadProgram(*prog_, *data_, 1, prog_id_, err) != tpu::ErrorCode::kSuccess)
            throw std::runtime_error("LinQ: loadProgram: " + std::string(err.c_str()));
        if (dev_->dev->openInferenceSession(prog_id_, session_id_, err) != tpu::ErrorCode::kSuccess)
            throw std::runtime_error("LinQ: openInferenceSession: " + std::string(err.c_str()));

        std::cerr << "[linq] " << path << ": вход " << in_layout_.order << ", выходов " << nout << "\n";
        for (size_t n = 0; n < nout; ++n) std::cerr << "        " << out_names_[n] << " " << out_layouts_[n].order << "\n";
    }

    void infer(const cv::Mat& rgb, std::vector<Tensor>& outs) override {
        auto ib = in_->getBuffer(0);
        fill_input(rgb, in_layout_, reinterpret_cast<float*>(ib.data()));
        {
            std::lock_guard<std::mutex> lk(dev_->mu);
            tpu::ErrorMessage err;
            if (dev_->dev->runInference(session_id_, *in_, *out_, tpu::PackingMode::kFull, err) != tpu::ErrorCode::kSuccess)
                throw std::runtime_error("LinQ: runInference: " + std::string(err.c_str()));
        }
        outs.clear();
        for (size_t n = 0; n < out_names_.size(); ++n) {
            auto ob = out_->getBuffer(n);
            outs.push_back(read_output(out_names_[n], out_layouts_[n], reinterpret_cast<const float*>(ob.data())));
        }
    }

    void dump_input(const cv::Mat& rgb, const std::string& path) override {
        size_t total = 1;
        for (auto v : in_layout_.shape) total *= v;
        std::vector<float> buf(total);
        fill_input(rgb, in_layout_, buf.data());
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(float));
    }

    std::string name() const override { return "linq-tpu"; }

private:
    std::unique_ptr<tpu::ProgramFile> file_;
    std::unique_ptr<tpu::Program> prog_;
    std::unique_ptr<tpu::ProgramData> data_;
    std::unique_ptr<tpu::TensorBufferObject> in_, out_;
    std::shared_ptr<DeviceHolder> dev_;
    uint64_t prog_id_ = 0, session_id_ = 0;
    Layout in_layout_;
    std::vector<std::string> out_names_;
    std::vector<Layout> out_layouts_;
};

}  // namespace

std::unique_ptr<Backend> make_linq_backend(const std::string& path, const json& opts) {
    return std::make_unique<LinqBackend>(path, opts);
}

#else

std::unique_ptr<Backend> make_linq_backend(const std::string&, const json&) {
    throw std::runtime_error("сервис собран без LinQ: пересоберите на машине с libtpu (cmake -DWITH_LINQ=ON)");
}

#endif

std::unique_ptr<Backend> make_opencv_backend(const std::string& path, const json& opts);
std::unique_ptr<Backend> make_remote_backend(const std::string& model, const json& opts);

std::unique_ptr<Backend> make_backend(const std::string& type, const std::string& path, const json& opts) {
    if (type == "remote") return make_remote_backend(path, opts);
    if (type == "linq") return make_linq_backend(path, opts);
    if (type == "opencv") return make_opencv_backend(path, opts);
    throw std::runtime_error("неизвестный бэкенд: " + type);
}
