# `.\pytorch\aten\src\ATen\native\cuda\linalg\MagmaUtils.h`

```
#pragma once
#include <ATen/cuda/CUDAConfig.h>

#if AT_MAGMA_ENABLED()
#include <magma_types.h>
#include <magma_v2.h>
#endif

namespace at {
namespace native {

#if AT_MAGMA_ENABLED()

// RAII for a MAGMA Queue
struct MAGMAQueue {

  // Default constructor without a device will cause
  // destroying a queue which has not been initialized.
  MAGMAQueue() = delete;

  // Constructor
  explicit MAGMAQueue(int64_t device_id) {
    // 获取当前的 cuBLAS 句柄
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
#if !defined(USE_ROCM)
    // Magma 操作对数值敏感，因此无论全局标志如何，TF32 都应关闭
    // 检查当前 cuBLAS 的数学模式并设置为默认
    TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &original_math_mode));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif
    // 从当前 CUDA 流创建 MAGMA 队列
    magma_queue_create_from_cuda(
      device_id,
      at::cuda::getCurrentCUDAStream(),
      handle,
      at::cuda::getCurrentCUDASparseHandle(),
      &magma_queue_);
  }

  // Getter
  // 返回 MAGMA 队列
  magma_queue_t get_queue() const { return magma_queue_; }

  // Destructor
  ~MAGMAQueue() {
#if !defined(USE_ROCM)
    // 手动设置数学模式为 CUBLAS_DEFAULT_MATH，现在应将原始数学模式恢复
    cublasHandle_t handle = magma_queue_get_cublas_handle(magma_queue_);
    cublasSetMathMode(handle, original_math_mode);
#endif
    // 销毁 MAGMA 队列
    magma_queue_destroy(magma_queue_);
  }

 private:
  magma_queue_t magma_queue_;
#if !defined(USE_ROCM)
  cublasMath_t original_math_mode;  // 原始数学模式
#endif
};

// 将 int64_t 转换为 magma_int_t 的辅助函数
static inline magma_int_t magma_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<magma_int_t>(value);
  // 如果转换后的值与原始值不一致，则抛出错误
  if (static_cast<int64_t>(result) != value) {
    AT_ERROR("magma: The value of ", varname, "(", (long long)value,
             ") is too large to fit into a magma_int_t (", sizeof(magma_int_t), " bytes)");
  }
  return result;
}

// MAGMA 函数如果不接受 magma_queue_t 则不是流安全的
// 通过与默认流同步来解决这个问题
struct MagmaStreamSyncGuard {
  MagmaStreamSyncGuard() {
    // 获取当前 CUDA 流
    auto stream = at::cuda::getCurrentCUDAStream();
    // 如果当前流不是默认流，则同步当前流
    if (stream != at::cuda::getDefaultCUDAStream()) {
      at::cuda::stream_synchronize(stream);
    }
  }

  ~MagmaStreamSyncGuard() noexcept(false) {
    // 获取默认 CUDA 流
    auto default_stream = at::cuda::getDefaultCUDAStream();
    // 如果当前流不是默认流，则同步默认流
    if (at::cuda::getCurrentCUDAStream() != default_stream) {
      at::cuda::stream_synchronize(default_stream);
    }
  }
};
#endif

} // namespace native
} // namespace at
```