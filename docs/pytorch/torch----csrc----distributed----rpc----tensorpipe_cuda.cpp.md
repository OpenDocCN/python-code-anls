# `.\pytorch\torch\csrc\distributed\rpc\tensorpipe_cuda.cpp`

```
#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>

#if defined(USE_TENSORPIPE) && !defined(USE_ROCM)

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// 忽略特定的编译器警告，因为 tensorpipe 中使用了一些被弃用的特性
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated")
#include <tensorpipe/tensorpipe.h>
#include <tensorpipe/tensorpipe_cuda.h>
C10_DIAGNOSTIC_POP()

namespace torch {
namespace distributed {
namespace rpc {
namespace {

#if TENSORPIPE_HAS_CUDA_IPC_CHANNEL

// 创建并返回一个 cuda_ipc 通道的注册对象
std::unique_ptr<ChannelRegistration> makeCudaIpcChannel() {
  auto context = tensorpipe::channel::cuda_ipc::create();
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCudaIpcChannelPriority});
}

// cuda_ipc 通道使用 cudaMemcpy 在进程间传输 CUDA 张量
C10_REGISTER_CREATOR(TensorPipeChannelRegistry, cuda_ipc, makeCudaIpcChannel);

#endif

#if TENSORPIPE_HAS_CUDA_GDR_CHANNEL

// 创建并返回一个 cuda_gdr 通道的注册对象
std::unique_ptr<ChannelRegistration> makeCudaGdrChannel() {
  auto context = tensorpipe::channel::cuda_gdr::create();
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCudaGdrChannelPriority});
}

// cuda_gdr 通道使用 GPUDirect RDMA 在 InfiniBand 上发送 CUDA 内存
// 它通过 libibverbs 直接注册用户提供的张量，这是昂贵的操作，但会缓存注册信息
// 以便在后续传输中分摊成本并获得低延迟。传输前需要准备好发送/接收握手，
// 确保设备索引匹配并使用正确的队列对。
C10_REGISTER_CREATOR(TensorPipeChannelRegistry, cuda_gdr, makeCudaGdrChannel);

#endif

// 创建并返回一个 cuda_xth 通道的注册对象
std::unique_ptr<ChannelRegistration> makeCudaXthChannel() {
  auto context = tensorpipe::channel::cuda_xth::create();
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCudaXthChannelPriority});
}

// cuda_xth 通道支持同一进程内的 GPU 到 GPU 通信
C10_REGISTER_CREATOR(TensorPipeChannelRegistry, cuda_xth, makeCudaXthChannel);

// 创建并返回一个 cuda_basic 通道的注册对象
std::unique_ptr<ChannelRegistration> makeCudaBasicChannel() {
  auto context = tensorpipe::channel::cuda_basic::create(
      tensorpipe::channel::basic::create());
  return std::make_unique<ChannelRegistration>(
      ChannelRegistration{std::move(context), kCudaBasicChannelPriority});
}

// cuda_basic 是 GPU 到 GPU 通信的备选通道
C10_REGISTER_CREATOR(
    TensorPipeChannelRegistry,
    cuda_basic,
    makeCudaBasicChannel);
class TensorpipeCudaConverter : public TensorpipeDeviceTypeConverter {
 public:
  // 准备张量以便发送，记录在TensorPipe流上，以确保在TensorPipe完成发送之前不会销毁张量。
  std::optional<std::vector<char>> prepareTensorForSending(
      const c10::Storage& storage,
      const std::vector<c10::Stream>& streams,
      tensorpipe::Message& message) const override {
    // 获取与存储设备相关联的CUDA流
    auto stream =
        at::cuda::CUDAStream(getStreamForDevice(streams, storage.device()));
    // 在TensorPipe流上记录张量数据指针，以确保在TensorPipe完成发送之前不会销毁张量。
    c10::cuda::CUDACachingAllocator::recordStream(storage.data_ptr(), stream);

    // 创建CUDA缓冲区对象并设置其指针和流
    tensorpipe::CudaBuffer buffer;
    buffer.ptr = static_cast<char*>(storage.mutable_data());
    buffer.stream = stream.stream();

    // 创建并设置消息中的张量对象，包括缓冲区和长度
    tensorpipe::Message::Tensor tensor;
    tensor.buffer = buffer;
    tensor.length = storage.nbytes();

    // 将张量对象添加到消息的张量列表中
    message.tensors.push_back(std::move(tensor));

    return c10::nullopt;
  }

  // 为接收分配张量，返回数据指针以及记录分配的详细信息
  at::DataPtr allocateTensorForReceiving(
      c10::DeviceIndex deviceIndex,
      size_t length,
      const std::vector<c10::Stream>& streams,
      tensorpipe::Allocation& allocation) const override {
    // 获取与给定设备索引相关联的CUDA设备
    c10::Device device(c10::kCUDA, deviceIndex);
    // 获取与设备相关联的CUDA流
    at::cuda::CUDAStream stream(getStreamForDevice(streams, device));
    // 在当前流上设置CUDA流守卫，以确保在分配期间正确调用记录流
    at::cuda::CUDAStreamGuard guard(stream);
    // 通过CUDA缓存分配器分配内存，并获取分配的数据指针
    at::DataPtr dataPtr =
        c10::cuda::CUDACachingAllocator::get()->allocate(length);

    // 创建CUDA缓冲区对象并设置其指针和流
    tensorpipe::CudaBuffer buffer;
    buffer.ptr = dataPtr.get();
    buffer.stream = stream.stream();

    // 创建并设置分配对象的张量，包括缓冲区
    tensorpipe::Allocation::Tensor tensor;
    tensor.buffer = buffer;

    // 将张量对象添加到分配的张量列表中
    allocation.tensors.push_back(tensor);

    return dataPtr;
  }
};

// 将CUDA设备类型转换器注册到TensorPipe框架中
C10_REGISTER_TENSORPIPE_DEVICE_TYPE_CONVERTER(CUDA, TensorpipeCudaConverter);

} // namespace rpc
} // namespace distributed
} // namespace torch

#endif
```