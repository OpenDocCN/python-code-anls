# `.\pytorch\c10\cuda\CUDAFunctions.h`

```py
#pragma once

// 提供对常用 CUDA API 函数的 C++ 封装
// 在这里使用 C++ 的好处是，我们可以在出现错误时抛出异常，而不是显式地传递错误码。
// 这样可以实现更自然的 API 设计。

// 包含必要的头文件
#include <c10/core/Device.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda_runtime_api.h>

namespace c10::cuda {

// 此函数返回当前 CUDA 设备的数量
C10_CUDA_API DeviceIndex device_count() noexcept;

// 如果没有检测到设备，则此函数抛出异常
C10_CUDA_API DeviceIndex device_count_ensure_non_zero();

// 返回当前活动的 CUDA 设备索引
C10_CUDA_API DeviceIndex current_device();

// 设置当前活动的 CUDA 设备
C10_CUDA_API void set_device(DeviceIndex device);

// 同步当前 CUDA 设备
C10_CUDA_API void device_synchronize();

// 如果同步操作出现问题，则发出警告或错误
C10_CUDA_API void warn_or_error_on_sync();

// 获取 CUDA 设备数量的原始 CUDA API 函数
C10_CUDA_API cudaError_t GetDeviceCount(int* dev_count);

// 获取当前设备索引的原始 CUDA API 函数
C10_CUDA_API cudaError_t GetDevice(DeviceIndex* device);

// 设置当前设备索引的原始 CUDA API 函数
C10_CUDA_API cudaError_t SetDevice(DeviceIndex device);

// 尝试设置当前设备索引的原始 CUDA API 函数
C10_CUDA_API cudaError_t MaybeSetDevice(DeviceIndex device);

// 交换当前设备索引的原始 CUDA API 函数
C10_CUDA_API DeviceIndex ExchangeDevice(DeviceIndex device);

// 尝试交换当前设备索引的原始 CUDA API 函数
C10_CUDA_API DeviceIndex MaybeExchangeDevice(DeviceIndex device);

// 设置目标 CUDA 设备的原始 CUDA API 函数
C10_CUDA_API void SetTargetDevice();

// 同步调试模式的枚举类型，包括禁用、警告和错误
enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

// WarningState 类用于存储 CUDA 同步警告状态的全局状态信息
class WarningState {
 public:
  // 设置同步调试模式
  void set_sync_debug_mode(SyncDebugMode l) {
    sync_debug_mode = l;
  }

  // 获取当前同步调试模式
  SyncDebugMode get_sync_debug_mode() {
    return sync_debug_mode;
  }

 private:
  SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

// 返回 WarningState 的静态实例的引用
C10_CUDA_API __inline__ WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}

// 下面的函数在头文件中定义，因为出于性能考虑，我们希望它们被内联
// 执行内存复制并同步的函数
C10_CUDA_API void __inline__ memcpy_and_sync(
    void* dst,
    const void* src,
    int64_t nbytes,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  // 如果同步调试模式不是禁用状态，则发出同步警告或错误
  if (C10_UNLIKELY(
          warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  // 获取当前的 Python 解释器，用于跟踪 GPU 操作
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        c10::kCUDA, reinterpret_cast<uintptr_t>(stream));


// 调用指针 interp 所指向对象的 trace_gpu_stream_synchronization 方法，
// 传入参数 c10::kCUDA 作为 CUDA 类型标志，
// 以及将 stream 转换为 uintptr_t 类型后作为第二个参数传入。
#if defined(TORCH_HIP_VERSION) && (TORCH_HIP_VERSION >= 301)
  // 如果定义了 TORCH_HIP_VERSION 并且其版本号大于等于 301，则使用 hipMemcpyWithStream 进行异步内存拷贝
  C10_CUDA_CHECK(hipMemcpyWithStream(dst, src, nbytes, kind, stream));
#else
  // 否则，使用 cudaMemcpyAsync 进行异步内存拷贝
  C10_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
  // 同步 CUDA 流以确保拷贝操作完成
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
}

C10_CUDA_API void __inline__ stream_synchronize(cudaStream_t stream) {
  // 如果同步调试模式不为禁用状态，则发出同步相关的警告或错误
  if (C10_UNLIKELY(
          warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  // 获取当前 GPU 跟踪的 Python 解释器对象
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  // 如果存在 GPU 跟踪解释器，则追踪 GPU 流同步操作
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        c10::kCUDA, reinterpret_cast<uintptr_t>(stream));
  }
  // 同步 CUDA 流以确保操作完成
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

C10_CUDA_API bool hasPrimaryContext(DeviceIndex device_index);
C10_CUDA_API std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext();

} // namespace c10::cuda


这段代码主要涉及异步内存拷贝和流同步操作，根据不同的条件选择使用不同的 CUDA API 进行内存拷贝，并在操作完成后同步 CUDA 流。
```