# `.\pytorch\torch\csrc\distributed\c10d\reducer_cuda.cpp`

```py
// 包含头文件：CUDA 器件事件和设备守卫
#include <torch/csrc/distributed/c10d/reducer_timer.hpp>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/DeviceGuard.h>

// 命名空间 c10d 开始
namespace c10d {

// 匿名命名空间开始
namespace {

// 常量定义：毫秒转换为纳秒的乘数
const int kMilliSecondToNanosSecond = 1000000;

// CUDA 计时器类，继承自 Timer
class CudaTimer : public Timer {
 private:
  // 成员变量：CUDA 计时器所在的设备
  c10::Device device;

  // CUDA 事件对象，用于记录不同阶段的时间点
  at::cuda::CUDAEvent forward_start = at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_compute_start =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_compute_end =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_comm_start =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_comm_end = at::cuda::CUDAEvent(cudaEventDefault);

  // 根据事件类型返回对应的 CUDA 事件对象的引用
  at::cuda::CUDAEvent& getEvent(Event event) {
    switch (event) {
      case Event::kForwardStart:
        return forward_start;
      case Event::kBackwardComputeStart:
        return backward_compute_start;
      case Event::kBackwardComputeEnd:
        return backward_compute_end;
      case Event::kBackwardCommStart:
        return backward_comm_start;
      case Event::kBackwardCommEnd:
        return backward_comm_end;
      default:
        // 如果出现未知事件类型，抛出内部断言错误
        TORCH_INTERNAL_ASSERT(false);
    }
  }

 public:
  // 构造函数：初始化 CUDA 计时器并指定设备
  explicit CudaTimer(c10::Device dev) : device(dev) {}

  // 记录特定事件的时间点
  void record(Event event) override {
    // 调用父类方法设置主机端时间
    Timer::record(event);
    // 设置设备守卫，确保在正确的 CUDA 设备上操作
    c10::DeviceGuard g(device);
    // 记录特定事件的 CUDA 事件
    getEvent(event).record();
  }

  // 计算两个事件之间的时间差
  std::optional<int64_t> measureDifference(Event start, Event end) override {
    // 设置设备守卫，确保在正确的 CUDA 设备上操作
    c10::DeviceGuard g(device);
    // 获取起始和结束事件的 CUDA 事件对象
    at::cuda::CUDAEvent& start_event = getEvent(start);
    at::cuda::CUDAEvent& end_event = getEvent(end);

    // 如果起始或结束事件尚未创建，返回空的可选类型
    if (!start_event.isCreated() || !end_event.isCreated()) {
      return c10::nullopt;
    }

    // 同步起始和结束 CUDA 事件，确保时间点已记录
    start_event.synchronize();
    end_event.synchronize();

    // 计算两个事件之间的时间差，单位为毫秒
    float milliseconds = start_event.elapsed_time(end_event);

    // 如果时间差小于零，表示计算无效，返回空的可选类型
    if (milliseconds < 0) {
      return c10::nullopt;
    }

    // 将毫秒转换为纳秒，并返回时间差的整数值
    return int64_t(milliseconds * kMilliSecondToNanosSecond);
  }
};

// 匿名命名空间结束
} // namespace
} // namespace c10d
C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kCUDA, CudaTimer);



// 注册一个类型化类，将 TimerRegistry 类注册为支持 CUDA 的类型，与 CudaTimer 相关联
C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kCUDA, CudaTimer);



} // namespace
} // namespace c10d



// 结束 namespace c10d 的定义
} // namespace c10d
```