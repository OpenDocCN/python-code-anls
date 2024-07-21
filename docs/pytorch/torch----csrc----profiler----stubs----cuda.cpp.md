# `.\pytorch\torch\csrc\profiler\stubs\cuda.cpp`

```
namespace torch {
namespace profiler {
namespace impl {
namespace {

// CUDA 错误检查函数，用于检查 CUDA 操作返回的错误
static inline void cudaCheck(cudaError_t result, const char* file, int line) {
  // 如果 CUDA 操作不成功，生成错误信息并抛出运行时异常
  if (result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": ";
    // 如果是 CUDA 初始化错误，输出相关错误信息和建议解决方案链接
    if (result == cudaErrorInitializationError) {
      ss << "CUDA initialization error. "
         << "This can occur if one runs the profiler in CUDA mode on code "
         << "that creates a DataLoader with num_workers > 0. This operation "
         << "is currently unsupported; potential workarounds are: "
         << "(1) don't use the profiler in CUDA mode or (2) use num_workers=0 "
         << "in the DataLoader or (3) Don't profile the data loading portion "
         << "of your code. https://github.com/pytorch/pytorch/issues/6313 "
         << "tracks profiler support for multi-worker DataLoader.";
    } else {
      // 否则，输出 CUDA 错误字符串
      ss << cudaGetErrorString(result);
    }
    throw std::runtime_error(ss.str());
  }
}

// 宏定义，用于调用 cudaCheck 函数检查 CUDA 操作返回的结果
#define TORCH_CUDA_CHECK(result) cudaCheck(result, __FILE__, __LINE__);

// CUDA 方法的具体实现，继承自 ProfilerStubs 类
struct CUDAMethods : public ProfilerStubs {
  // 记录 CUDA 事件的具体实现
  void record(
      c10::DeviceIndex* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {
    // 如果传入了设备索引，设置当前 CUDA 设备
    if (device) {
      TORCH_CUDA_CHECK(c10::cuda::GetDevice(device));
    }
    // 创建 CUDA 事件对象，并将其封装为 shared_ptr
    CUevent_st* cuda_event_ptr{nullptr};
    TORCH_CUDA_CHECK(cudaEventCreate(&cuda_event_ptr));
    *event = std::shared_ptr<CUevent_st>(cuda_event_ptr, [](CUevent_st* ptr) {
      // 释放 CUDA 事件对象
      TORCH_CUDA_CHECK(cudaEventDestroy(ptr));
    });
    // 获取当前 CUDA 流
    auto stream = at::cuda::getCurrentCUDAStream();
    // 如果传入了 CPU 时间戳变量，记录当前时间戳
    if (cpu_ns) {
      *cpu_ns = c10::getTime();
    }
    // 记录 CUDA 事件到指定流中
    TORCH_CUDA_CHECK(cudaEventRecord(cuda_event_ptr, stream));
  }

  // 计算 CUDA 事件之间的时间间隔
  float elapsed(
      const ProfilerVoidEventStub* event_,
      const ProfilerVoidEventStub* event2_) const override {
    auto event = (const ProfilerEventStub*)(event_);
    auto event2 = (const ProfilerEventStub*)(event2_);
    // 同步等待两个 CUDA 事件完成
    TORCH_CUDA_CHECK(cudaEventSynchronize(event->get()));
    TORCH_CUDA_CHECK(cudaEventSynchronize(event2->get()));
    // 计算两个 CUDA 事件之间的时间差，单位为毫秒
    float ms = 0;
    TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, event->get(), event2->get()));
    // 返回时间差，单位为微秒
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions)
    return ms * 1000.0;
  }

  // 在 NVTX 中标记事件
  void mark(const char* name) const override {
    ::nvtxMark(name);
  }

  // 在 NVTX 中推入一个命名范围
  void rangePush(const char* name) const override {
    ::nvtxRangePushA(name);
  }

  // 在 NVTX 中弹出一个命名范围
  void rangePop() const override {
    ::nvtxRangePop();
  }

  // 针对每个 CUDA 设备执行操作
  void onEachDevice(std::function<void(int)> op) const override {
    at::cuda::OptionalCUDAGuard device_guard;
    // 对于从0到当前CUDA设备数量的范围内的每个设备，执行以下操作
    for (const auto i : c10::irange(at::cuda::device_count())) {
      // 设置当前设备索引为 i
      device_guard.set_index(i);
      // 调用传入的函数 op，并传入当前设备索引 i
      op(i);
    }
  }

  // 同步 CUDA 设备，确保所有设备上的操作完成
  void synchronize() const override {
    // 调用 CUDA 函数 cudaDeviceSynchronize() 同步所有 CUDA 设备
    TORCH_CUDA_CHECK(cudaDeviceSynchronize());
  }

  // 返回当前接口是否启用，始终返回 true
  bool enabled() const override {
    return true;
  }
};

// 结构体 RegisterCUDAMethods 的定义
struct RegisterCUDAMethods {
  // RegisterCUDAMethods 的构造函数
  RegisterCUDAMethods() {
    // 创建静态变量 methods，类型为 CUDAMethods
    static CUDAMethods methods;
    // 调用 registerCUDAMethods 函数，注册 methods 变量
    registerCUDAMethods(&methods);
  }
};

// 创建 RegisterCUDAMethods 的实例 reg

// 结束命名空间 torch
} // namespace torch

// 结束命名空间 profiler
} // namespace profiler

// 结束命名空间 impl
} // namespace impl

// 结束匿名命名空间
} // namespace
```