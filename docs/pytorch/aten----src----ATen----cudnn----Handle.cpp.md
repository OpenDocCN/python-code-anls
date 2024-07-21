# `.\pytorch\aten\src\ATen\cudnn\Handle.cpp`

```py
namespace at { namespace native {
namespace {

// 创建一个新的 CuDNN 句柄并存储在指定的 handle 指针中
void createCuDNNHandle(cudnnHandle_t *handle) {
  AT_CUDNN_CHECK(cudnnCreate(handle));
}

// 销毁 CuDNN 句柄（当前未启用，由于销毁时机问题）
void destroyCuDNNHandle(cudnnHandle_t /*handle*/) {
// 这是由于销毁顺序的一些问题，有时在退出时，CUDA 上下文（或者其他什么东西）
// 可能在此之前已经被销毁了。在 fbcode 环境中发生过。@colesbury 和我决定
// 通过不销毁句柄来解决这个问题。
//   - @soumith
//
// 进一步说明：现在全局上已经禁用了此功能，因为我们在 CUDA 11 CI 中看到了
// 上述问题。
//   - @zasdfgbnm
//
// #ifdef NO_CUDNN_DESTROY_HANDLE
// #else
//   cudnnDestroy(handle);
// #endif
}

// 使用模板创建一个 CUDA 设备线程句柄池，用于管理 CuDNN 句柄
using CudnnPoolType = at::cuda::DeviceThreadHandlePool<cudnnHandle_t, createCuDNNHandle, destroyCuDNNHandle>;

} // namespace

// 获取当前 CUDA 设备的 CuDNN 句柄
cudnnHandle_t getCudnnHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // 线程本地的 PoolWindow 是延迟初始化的，以避免在 Windows 上导致的初始化问题
  // 参见：https://github.com/pytorch/pytorch/pull/22405
  // 这些线程本地的 unique_ptr 将在线程终止时销毁，将其预留的句柄释放回池中
  static auto pool = std::make_shared<CudnnPoolType>();
  thread_local std::unique_ptr<CudnnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  // 从池中获取当前设备的句柄
  auto handle = myPoolWindow->reserve(device);
  AT_CUDNN_CHECK(cudnnSetStream(handle, c10::cuda::getCurrentCUDAStream()));
  return handle;
}

}} // namespace at::native
```