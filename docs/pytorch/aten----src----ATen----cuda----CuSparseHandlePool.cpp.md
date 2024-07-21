# `.\pytorch\aten\src\ATen\cuda\CuSparseHandlePool.cpp`

```
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

namespace at::cuda {
namespace {

// 创建一个 cuSparse 句柄
void createCusparseHandle(cusparseHandle_t *handle) {
  TORCH_CUDASPARSE_CHECK(cusparseCreate(handle));
}

// 销毁 cuSparse 句柄
void destroyCusparseHandle(cusparseHandle_t handle) {
// 这是因为在销毁顺序上有些问题。有时在退出时，CUDA 上下文（或其他一些东西）
// 已经被销毁了，导致这里尝试销毁 handle 时出现问题。这在 fbcode 的设置中会发生。
// @colesbury 和 @soumith 决定不销毁句柄作为一种解决方法。
//   - @soumith 的注释，从 cuDNN 句柄池的实现中复制过来
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
    cusparseDestroy(handle);
#endif
}

// 使用 DeviceThreadHandlePool 包装的 cuSparse 句柄池
using CuSparsePoolType = DeviceThreadHandlePool<cusparseHandle_t, createCusparseHandle, destroyCusparseHandle>;

} // namespace

// 获取当前 CUDA 稀疏计算的句柄
cusparseHandle_t getCurrentCUDASparseHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // 线程局部的 PoolWindows 是延迟初始化的，
  // 避免因初始化问题在 Windows 上导致 hang。
  // 参见：https://github.com/pytorch/pytorch/pull/22405
  // 当线程终止时，这些线程局部的 unique_ptrs 将被销毁，
  // 将其保留的句柄释放回池中。
  static auto pool = std::make_shared<CuSparsePoolType>();
  thread_local std::unique_ptr<CuSparsePoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  // 在当前设备上获取一个保留的句柄
  auto handle = myPoolWindow->reserve(device);
  TORCH_CUDASPARSE_CHECK(cusparseSetStream(handle, c10::cuda::getCurrentCUDAStream()));
  return handle;
}

} // namespace at::cuda
```