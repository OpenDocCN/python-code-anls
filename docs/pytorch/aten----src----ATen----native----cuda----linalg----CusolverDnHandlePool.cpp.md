# `.\pytorch\aten\src\ATen\native\cuda\linalg\CusolverDnHandlePool.cpp`

```py
#if defined(CUDART_VERSION) || defined(USE_ROCM)

# 如果定义了CUDART_VERSION或者USE_ROCM，则编译以下代码块

namespace at::cuda {
namespace {

void createCusolverDnHandle(cusolverDnHandle_t *handle) {
  // 创建 cuSolver DN 句柄
  TORCH_CUSOLVER_CHECK(cusolverDnCreate(handle));
}

void destroyCusolverDnHandle(cusolverDnHandle_t handle) {
  // 由于某些销毁顺序问题，有时在 fbcode 设置中，可能在此句柄销毁时 CUDA 上下文（或其他部分）已被销毁
  // 这个问题在 cuDNN 句柄池的实现中也有体现
  // @colesbury 和 @soumith 决定不销毁这个句柄作为一种解决方法
  // - @soumith 的评论，来自 cuDNN 句柄池的实现
#ifdef NO_CUDNN_DESTROY_HANDLE
  (void)handle; // 抑制未使用变量警告
#else
  cusolverDnDestroy(handle);
#endif
}

using CuSolverDnPoolType = DeviceThreadHandlePool<cusolverDnHandle_t, createCusolverDnHandle, destroyCusolverDnHandle>;

} // namespace

cusolverDnHandle_t getCurrentCUDASolverDnHandle() {
  // 获取当前设备索引
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // 线程本地的 PoolWindows 是延迟初始化的，
  // 避免在 Windows 上引发挂起的初始化问题
  // 参考：https://github.com/pytorch/pytorch/pull/22405
  // 当线程终止时，这些线程本地的 unique_ptrs 将被销毁，
  // 将其保留的句柄返回给池
  static auto pool = std::make_shared<CuSolverDnPoolType>();
  thread_local std::unique_ptr<CuSolverDnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  // 在当前设备上保留一个句柄
  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return handle;
}

} // namespace at::cuda

#endif // CUDART_VERSION
```