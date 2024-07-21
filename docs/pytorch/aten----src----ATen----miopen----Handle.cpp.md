# `.\pytorch\aten\src\ATen\miopen\Handle.cpp`

```
// 包含头文件，ATen/miopen/Exceptions.h 提供了 MIOpen 异常相关功能
// ATen/miopen/Handle.h 提供了 MIOpen 句柄管理功能
// ATen/hip/detail/DeviceThreadHandles.h 提供了设备线程句柄池功能
// c10/hip/HIPStream.h 提供了 HIP 流管理功能
#include <ATen/miopen/Exceptions.h>
#include <ATen/miopen/Handle.h>
#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <c10/hip/HIPStream.h>

// 命名空间 at::native 包裹了本地实现
namespace at { namespace native {
// 命名空间内定义了一个未命名命名空间，用于隐藏函数和类型的全局可见性
namespace {

// 创建 MIOpen 句柄的函数，接收一个 miopenHandle_t 指针
void createMIOpenHandle(miopenHandle_t *handle) {
  // 调用 miopenCreate 函数创建 MIOpen 句柄
  MIOPEN_CHECK(miopenCreate(handle));
}

// 销毁 MIOpen 句柄的函数，接收一个 miopenHandle_t 句柄
void destroyMIOpenHandle(miopenHandle_t handle) {
  // 以下注释是关于销毁句柄的一些问题和工作区。
  // 因为某些不明原因，销毁句柄的顺序可能导致问题。
  // 在 fbcode 设置中，有时候在此之前 CUDA 上下文可能已经被销毁。
  // @colesbury 和我决定不销毁句柄作为一种临时解决方案。
  //   - @soumith
  //
  // 进一步说明：这已经在全局禁用了，因为我们在 CUDA 11 CI 中遇到了与上述相同的问题。
  //   - @zasdfgbnm
  //
  // #ifdef NO_MIOPEN_DESTROY_HANDLE
  // #else
  //   miopenDestroy(handle);
  // #endif
}

// 使用别名定义 MIOpen 句柄池类型，通过 ATen 的 CUDA 设备线程句柄池实现
using MIOpenPoolType = at::cuda::DeviceThreadHandlePool<miopenHandle_t, createMIOpenHandle, destroyMIOpenHandle>;

} // namespace

// 获取当前设备的 MIOpen 句柄的函数
miopenHandle_t getMiopenHandle() {
  int device;
  HIP_CHECK(hipGetDevice(&device));

  // 线程局部的 PoolWindows 是延迟初始化的，以避免在 Windows 上导致的初始化问题导致 hang 问题。
  // 参见：https://github.com/pytorch/pytorch/pull/22405
  // 当线程终止时，此线程局部的 unique_ptrs 将被销毁，将其保留的句柄返回给池。
  static auto pool = std::make_shared<MIOpenPoolType>();
  thread_local std::unique_ptr<MIOpenPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  // 从池中为当前设备获取一个句柄
  auto handle = myPoolWindow->reserve(device);
  // 将该句柄绑定到当前 HIP 流上
  MIOPEN_CHECK(miopenSetStream(handle, at::hip::getCurrentHIPStream()));
  // 返回获取的 MIOpen 句柄
  return handle;
}

}} // namespace at::native
```