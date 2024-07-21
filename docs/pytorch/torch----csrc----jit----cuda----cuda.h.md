# `.\pytorch\torch\csrc\jit\cuda\cuda.h`

```
namespace torch {
namespace jit {

// CUDAEvent class declaration, which wraps around at::cuda::CUDAEvent.
// This class is used because TorchBind does not support all argument types
// for at::cuda::CUDAEvent. Refer to aten/src/ATen/cuda/CUDAEvent.h for details.
class CUDAEvent final : public CustomClassHolder {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Constructor for CUDAEvent class.
  CUDAEvent(
      bool enable_timing = false,
      bool blocking = false,
      bool interprocess = false) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // Initialize flags based on constructor arguments.
    int flags = cudaEventDisableTiming;
    if (enable_timing) {
      flags = cudaEventDefault;
    }
    if (blocking) {
      flags |= cudaEventBlockingSync;
    }
    if (interprocess) {
      // Ensure interprocess flag is set correctly.
      TORCH_CHECK(!enable_timing);
      flags |= cudaEventInterprocess;
    }

    // Create a new CUDA event with specified flags.
    event_ = std::make_unique<at::cuda::CUDAEvent>(flags);
  }

  // Method to compute elapsed time between events.
  double elapsedTime(c10::intrusive_ptr<CUDAEvent> end) {
    return event_->elapsed_time(*end->event_);
  }

  // Method to retrieve IPC handle for the CUDA event.
  std::string ipcHandle() {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    cudaIpcEventHandle_t handle;
    // Obtain IPC handle for the CUDA event.
    CUDA_CHECK(cudaIpcGetEventHandle(&handle, event_->get()));
    // Convert handle to a string representation.
    return std::string(reinterpret_cast<char*>(&handle), sizeof(handle));
  }

 private:
  // Unique pointer to the underlying CUDA event.
  std::unique_ptr<at::cuda::CUDAEvent> event_;
  friend class CUDAStream;
};

// CUDAStream class declaration, which wraps around c10::cuda::CUDAStream.
// This class is used because TorchBind does not support all argument types
// for c10::cuda::CUDAStream. Refer to c10/cuda/CUDAStream.h for details.
class CUDAStream final : public CustomClassHolder {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Constructor for CUDAStream class.
  CUDAStream(
      std::optional<c10::Device> device = c10::nullopt,
      int64_t priority = 0) {
    // Determine CUDA device index based on optional argument or current device.
    c10::DeviceIndex device_index =
        device.has_value() ? device->index() : c10::cuda::current_device();
    
    // Create a new CUDA stream using the device index and priority.
    stream_ = std::make_unique<c10::cuda::CUDAStream>(
        c10::cuda::getStreamFromPool(static_cast<int>(priority), device_index));
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // Constructor for CUDAStream class, using an existing CUDA stream.
  CUDAStream(c10::cuda::CUDAStream s) {
    // Create a new CUDA stream using an existing stream.
    stream_ = std::make_unique<c10::cuda::CUDAStream>(s);
  }

  // Method to query if the stream is in a usable state.
  bool query() {
    return stream_->query();
  }

  // Method to record a CUDA event on the stream.
  c10::intrusive_ptr<CUDAEvent> recordEvent(
      c10::intrusive_ptr<CUDAEvent> event);

  // Method to synchronize the stream.
  void synchronize() {
    stream_->synchronize();
  }

  // Method to wait for a CUDA event on the stream.
  void waitEvent(c10::intrusive_ptr<CUDAEvent> event);

  // Method to wait for another CUDA stream to complete.
  void waitStream(c10::intrusive_ptr<CUDAStream> stream);

  /// Get the CUDA device index associated with this stream.
  int64_t device_index() const {
    return stream_->device_index();
  }

  /// Get the full Device associated with this stream (guaranteed to be CUDA).
  c10::Device device() const {
    return stream_->device();
  }

  /// Return the stream ID corresponding to this particular stream.
  int64_t id() const {
    return stream_->id();
  }

 private:
  // Unique pointer to the underlying CUDA stream.
  std::unique_ptr<c10::cuda::CUDAStream> stream_;
  friend class CUDAEvent;
};

} // namespace jit
} // namespace torch
    // 调用 event_ 对象的 ipc_handle 方法，并将其返回值存储在 handle 中
    event_->ipc_handle(&handle);
    // 将 handle 转换为字符串形式，并存储在 str_handle 中
    std::string str_handle((const char*)&handle, sizeof(handle));
    // 返回字符串形式的 handle
    return str_handle;
  }

  // 查询 event_ 对象的状态，返回查询结果
  bool query() {
    return event_->query();
  }

  // 记录与 CUDA 流相关的事件，声明但未实现具体内容
  void record(c10::intrusive_ptr<CUDAStream> stream);

  // 同步 event_ 对象的状态
  void synchronize() {
    event_->synchronize();
  }

  // 等待与 CUDA 流相关的事件完成
  void wait(c10::intrusive_ptr<CUDAStream> stream);

 private:
  // 内部方法，用于记录 CUDA 流的相关事件
  void recordInternal(CUDAStream* stream);
  // 持有一个独占的 CUDAEvent 对象，用于管理 CUDA 事件
  std::unique_ptr<at::cuda::CUDAEvent> event_;

  // 声明 CUDAStream 类为当前类的友元类，使其可以访问私有成员
  friend class CUDAStream;
};

// 在此处定义了 Torch 库中 CUDA 相关的流和事件类的注册，使其可在 Torch 中使用

TORCH_LIBRARY(cuda, m) {
  // 定义 CUDAStream 类，并设置初始化函数，用于创建 CUDA 流对象
  auto stream_class = m.class_<torch::jit::CUDAStream>("Stream").def(
      torch::init<std::optional<c10::Device>, int64_t>(),
      "",
      {torch::arg("device") = c10::nullopt, torch::arg("priority") = 0});
  
  // 定义 CUDAEvent 类，并设置初始化函数，用于创建 CUDA 事件对象
  auto event_class = m.class_<torch::jit::CUDAEvent>("Event").def(
      torch::init<bool, bool, bool>(),
      "",
      {torch::arg("enable_timing") = false,
       torch::arg("blocking") = false,
       torch::arg("interprocess") = false});

  // 定义 CUDAStream 类中的函数绑定，使其能在 Torch 中调用
  stream_class.def("query", &CUDAStream::query)
      .def("record_event", &CUDAStream::recordEvent)
      .def("synchronize", &CUDAStream::synchronize)
      .def("wait_event", &CUDAStream::waitEvent)
      .def("wait_stream", &CUDAStream::waitStream)
      .def("device_index", &CUDAStream::device_index)
      .def_property("device", &CUDAStream::device)
      .def("id", &CUDAStream::id);

  // 定义 CUDAEvent 类中的函数绑定，使其能在 Torch 中调用
  event_class.def("elapsed_time", &CUDAEvent::elapsedTime)
      .def("query", &CUDAEvent::query)
      .def("record", &CUDAEvent::record)
      .def("synchronize", &CUDAEvent::synchronize)
      .def("wait", &CUDAEvent::wait);
};

} // namespace jit
} // namespace torch
```