# `.\pytorch\torch\csrc\distributed\rpc\tensorpipe_utils.h`

```
#pragma once
// 如果定义了 USE_TENSORPIPE 宏，则编译以下代码

#ifdef USE_TENSORPIPE
// 引入 TensorPipe 相关的头文件
#include <torch/csrc/distributed/rpc/utils.h>

// 定义了 tensorpipe 命名空间，包含 Message、Allocation 和 Descriptor 类
namespace tensorpipe {
class Message;
class Allocation;
class Descriptor;
} // namespace tensorpipe

// 定义了 torch 命名空间内的 distributed::rpc 命名空间
namespace torch {
namespace distributed {
namespace rpc {

// 声明并导出的函数，获取适用于特定设备的流对象
TORCH_API const c10::Stream& getStreamForDevice(
    const std::vector<c10::Stream>& streams,
    const c10::Device& device);

// 受 c10/core/impl/DeviceGuardImplInterface.h 启发的类定义

class TensorpipeDeviceTypeConverter {
 public:
  // 准备发送的张量对象，向传入的 tensorpipe::Message 对象的 tensors 字段附加 TensorPipe 张量对象的数据
  virtual std::optional<std::vector<char>> prepareTensorForSending(
      const c10::Storage& storage,
      const std::vector<c10::Stream>& streams,
      tensorpipe::Message& message) const = 0;

  // 分配用于接收的张量对象，向传入的 tensorpipe::Allocation 对象的 tensors 字段附加 TensorPipe 张量对象的数据
  virtual at::DataPtr allocateTensorForReceiving(
      c10::DeviceIndex deviceIndex,
      size_t length,
      const std::vector<c10::Stream>& streams,
      tensorpipe::Allocation& allocation) const = 0;

  virtual ~TensorpipeDeviceTypeConverter() = default;
};

// 定义了一个数组，用于注册 TensorPipe 设备类型转换器
extern TORCH_API std::array<
    std::atomic<const TensorpipeDeviceTypeConverter*>,
    static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
    device_type_converter_registry;

// 定义了 TensorpipeDeviceTypeConverterRegistrar 类
class TORCH_API TensorpipeDeviceTypeConverterRegistrar {
 public:
  // 构造函数，注册 TensorPipe 设备类型转换器
  TensorpipeDeviceTypeConverterRegistrar(
      DeviceType,
      const TensorpipeDeviceTypeConverter*);
};

// 定义了宏，用于注册 TensorPipe 设备类型转换器
#define C10_REGISTER_TENSORPIPE_DEVICE_TYPE_CONVERTER(                     \
    DevType, TensorpipeDeviceTypeConverter)                                \
  static ::torch::distributed::rpc::TensorpipeDeviceTypeConverterRegistrar \
      C10_ANONYMOUS_VARIABLE(g_##DeviceType)(                              \
          ::c10::DeviceType::DevType, new TensorpipeDeviceTypeConverter());

// 返回指定设备类型的 TensorPipe 设备类型转换器对象
inline const TensorpipeDeviceTypeConverter* getDeviceTypeConverter(
    DeviceType type) {
  return device_type_converter_registry[static_cast<size_t>(type)].load();
}

// 一个结构体，包含指针，用于在写操作期间由 TensorPipe 访问的所有内存
struct TensorpipeWriteBuffers {
  // Allocate on heap so pointers stay valid as we move the holder.
  // 使用 unique_ptr 管理 MessageType 对象，确保在移动 holder 时指针仍然有效
  std::unique_ptr<MessageType> type;
  // 使用 unique_ptr 管理 int64_t 对象，确保在移动 holder 时指针仍然有效
  std::unique_ptr<int64_t> id;
  // 存储负载数据的向量
  std::vector<char> payload;
  // 存储 pickle 数据的向量
  std::vector<char> pickle;
  // 包含原始张量和稀疏张量克隆的向量
  std::vector<torch::Tensor> tensors;
  // 包含未拥有其内存数据副本的张量的向量，例如通过 torch::from_blob() 创建且没有使用 deleter 的张量
  std::vector<std::vector<char>> copiedTensors;
};

// 用于在读取操作期间保持所有由 TensorPipe 访问的内存有效的指针持有者
struct TensorpipeReadBuffers {
  // Allocate on heap so pointers stay valid as we move the holder.
  // 使用 unique_ptr 管理 MessageType 对象，确保在移动 holder 时指针仍然有效
  std::unique_ptr<MessageType> type;
  // 使用 unique_ptr 管理 int64_t 对象，确保在移动 holder 时指针仍然有效
  std::unique_ptr<int64_t> id;
  // 存储负载数据的向量
  std::vector<char> payload;
  // 存储 pickle 数据的向量
  std::vector<char> pickle;
  // 存储 c10::DataPtr 对象的向量，用于管理张量的数据指针
  std::vector<c10::DataPtr> tensors;
};

// 将 RPC 消息转换为 TensorPipe 消息，并提供一个持有所有必须在异步写入期间保持有效的数据的 holder
TORCH_API std::tuple<tensorpipe::Message, TensorpipeWriteBuffers>
tensorpipeSerialize(
    c10::intrusive_ptr<Message> rpcMessage,
    std::vector<c10::Device> devices,
    const std::vector<c10::Stream>& streams);

// 分配用于保存传入数据的缓冲区。这些缓冲区将由返回的 holder 管理，holder 必须在异步读取完成之前保持有效。
// 这些缓冲区的指针将存储在返回的 tensorpipe::Allocation 结构中。
TORCH_API std::pair<tensorpipe::Allocation, TensorpipeReadBuffers>
tensorpipeAllocate(
    const tensorpipe::Descriptor& tpDescriptor,
    const std::vector<c10::Stream>& streams);

// 将 TensorPipe 消息反序列化为 RPC 消息。这需要数据可用，因此只能在异步读取完成后执行。
// holder 在此函数返回后可以销毁。
TORCH_API c10::intrusive_ptr<Message> tensorpipeDeserialize(
    tensorpipe::Descriptor&& tpDescriptor,
    TensorpipeReadBuffers&& holder);

} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE
```