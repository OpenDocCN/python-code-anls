# `.\pytorch\torch\csrc\inductor\aoti_runtime\utils_cuda.h`

```
#pragma once

#ifdef USE_CUDA
// 如果使用 CUDA，则包含必要的头文件

// 注意：在此处添加新的包含时要小心。此头文件将用于 model.so，
// 不应引用除了 torch/csrc/inductor/aoti_torch/c/shim.h 中定义的
// 稳定的 C ABI 之外的任何 aten/c10 头文件。torch/csrc/inductor/aoti_runtime/
// 下的其他文件也适用同样的规则。
#include <torch/csrc/inductor/aoti_runtime/utils.h>  // 包含自定义的 CUDA 相关工具函数

#include <cuda.h>             // CUDA 标准头文件
#include <cuda_runtime.h>     // CUDA 运行时头文件

namespace torch::aot_inductor {

// 删除 CUDA guard 的函数，用于释放 CUDA guard 对象的资源
inline void delete_cuda_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_cuda_guard(reinterpret_cast<CUDAGuardHandle>(ptr)));
}

// 删除 CUDA stream guard 的函数，用于释放 CUDA stream guard 对象的资源
inline void delete_cuda_stream_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_cuda_stream_guard(
      reinterpret_cast<CUDAStreamGuardHandle>(ptr)));
}

// 封装 CUDA guard 对象的类
class AOTICudaGuard {
 public:
  // 构造函数，创建指定设备索引的 CUDA guard 对象
  AOTICudaGuard(int32_t device_index) : guard_(nullptr, delete_cuda_guard) {
    CUDAGuardHandle ptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_cuda_guard(device_index, &ptr));
    guard_.reset(ptr);
  }

  // 设置 CUDA guard 对象的设备索引
  void set_index(int32_t device_index) {
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_cuda_guard_set_index(guard_.get(), device_index));
  }

 private:
  std::unique_ptr<CUDAGuardOpaque, DeleterFnPtr> guard_;  // 用于管理 CUDA guard 对象的智能指针
};

// 封装 CUDA stream guard 对象的类
class AOTICudaStreamGuard {
 public:
  // 构造函数，创建指定流和设备索引的 CUDA stream guard 对象
  AOTICudaStreamGuard(cudaStream_t stream, int32_t device_index)
      : guard_(nullptr, delete_cuda_stream_guard) {
    CUDAStreamGuardHandle ptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_cuda_stream_guard(stream, device_index, &ptr));
    guard_.reset(ptr);
  }

 private:
  std::unique_ptr<CUDAStreamGuardOpaque, DeleterFnPtr> guard_;  // 用于管理 CUDA stream guard 对象的智能指针
};

} // namespace torch::aot_inductor
#endif // USE_CUDA
```