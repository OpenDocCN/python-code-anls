# `.\pytorch\torch\csrc\distributed\rpc\tensorpipe_utils.cpp`

```
// 包含TensorPipe工具的头文件
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>

#ifdef USE_TENSORPIPE

// 包含一些必要的头文件
#include <c10/util/irange.h>  // 包含范围迭代工具
#include <limits>             // 包含数值极限定义

// 忽略某些编译器警告，因为使用了已废弃的功能
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated")
#include <tensorpipe/tensorpipe.h>  // 包含TensorPipe库的主头文件
C10_DIAGNOSTIC_POP()

namespace torch {
namespace distributed {
namespace rpc {
namespace {

// TensorPipe代理器将RPC消息的信息分布在多个负载中。这使得代理器能够将数据提供给TensorPipe，
// 而无需将其复制到单一连续缓冲区中，也无需将其存储为元数据，这样效率更高。

// 第一个负载是rpc::Message::type()和::id()。
constexpr int kTpMessageTypeIdx = 0;    // RPC消息类型的索引
constexpr int kTpMessageIdIdx = 1;      // RPC消息ID的索引
// 然后是rpc::Message::payload()。
constexpr int kTpMessagePayloadIdx = 2; // RPC消息负载的索引
// 最后是rpc::Message::tensors()的pickle（其中张量本身以张量形式存储在tensorpipe::Message中）。
constexpr int kTpMessagePickleIdx = 3;  // RPC消息pickle的索引

// 将设备索引转换为设备类型
inline c10::Device indexToDevice(c10::DeviceIndex index) {
  if (index == -1) {
    return c10::Device(at::kCPU);    // 如果索引为-1，返回CPU设备
  } else {
    return c10::Device(at::kCUDA, index);  // 否则返回CUDA设备，索引为index
  }
}

// TensorPipeCpuConverter类，用于在CPU上处理张量转换
class TensorpipeCpuConverter : public TensorpipeDeviceTypeConverter {
 public:
  // 准备张量以便发送
  std::optional<std::vector<char>> prepareTensorForSending(
      const c10::Storage& storage,
      const std::vector<c10::Stream>& /* streams */,
      tensorpipe::Message& message) const override {
    // 如果张量是由torch::from_blob创建的，强制进行内存复制，意味着张量不拥有内存
    bool storageHasDeleter = storage.data_ptr().get_context() != nullptr;
    if (!storageHasDeleter) {
      // 如果没有内存释放器，复制存储区域的数据
      std::vector<char> storageData(
          static_cast<const char*>(storage.data()),
          static_cast<const char*>(storage.data()) + storage.nbytes());

      // 创建CPU缓冲区
      tensorpipe::CpuBuffer buffer;
      buffer.ptr = storageData.data();

      // 创建TensorPipe消息的张量对象
      tensorpipe::Message::Tensor tensor;
      tensor.buffer = buffer;
      tensor.length = storageData.size();

      // 将张量对象添加到消息中
      message.tensors.push_back(std::move(tensor));

      // 返回复制的数据
      return c10::make_optional(std::move(storageData));
    } else {
      // 如果有内存释放器，直接使用存储区域的指针
      tensorpipe::CpuBuffer buffer;
      buffer.ptr = static_cast<char*>(storage.mutable_data());

      // 创建TensorPipe消息的张量对象
      tensorpipe::Message::Tensor tensor;
      tensor.buffer = buffer;
      tensor.length = storage.nbytes();

      // 将张量对象添加到消息中
      message.tensors.push_back(std::move(tensor));

      // 返回空值，表示没有复制的数据
      return c10::nullopt;
    }
  }

  // 为接收分配张量的内存
  at::DataPtr allocateTensorForReceiving(
      c10::DeviceIndex /* deviceIndex */,
      size_t length,
      const std::vector<c10::Stream>& /* streams */,
      tensorpipe::Allocation& allocation) const override {
    // 在CPU上分配接收张量的内存
    at::DataPtr dataPtr = at::getCPUAllocator()->allocate(length);

    // 创建CPU缓冲区
    tensorpipe::CpuBuffer buffer;
    buffer.ptr = dataPtr.get();

    // 创建TensorPipe分配对象的张量
    tensorpipe::Allocation::Tensor tensor;
    tensor.buffer = buffer;

    // 将张量对象添加到分配对象中
    allocation.tensors.push_back(std::move(tensor));

    // 返回分配的数据指针
    return dataPtr;
  }
};
// 注册 TensorPipe 设备类型转换器
C10_REGISTER_TENSORPIPE_DEVICE_TYPE_CONVERTER(CPU, TensorpipeCpuConverter);

// 将 TensorPipe 设备类型转换为 PyTorch 设备类型
c10::DeviceType convertDeviceType(const std::string& tpDeviceType) {
  if (tpDeviceType == tensorpipe::kCpuDeviceType) {
    return c10::kCPU;
  } else if (tpDeviceType == tensorpipe::kCudaDeviceType) {
    return c10::kCUDA;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unrecognized TensorPipe buffer type.");
  }
}

} // namespace

// 根据设备获取对应的流
const c10::Stream& getStreamForDevice(
    const std::vector<c10::Stream>& streams,
    const c10::Device& device) {
  for (const c10::Stream& stream : streams) {
    if (stream.device() == device) {
      return stream;
    }
  }
  TORCH_INTERNAL_ASSERT(false, "No stream found for device ", device);
}

// 注册 TensorPipe 设备类型转换器
std::array<
    std::atomic<const TensorpipeDeviceTypeConverter*>,
    static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
    device_type_converter_registry;

// 注册 TensorPipe 设备类型转换器
TensorpipeDeviceTypeConverterRegistrar::TensorpipeDeviceTypeConverterRegistrar(
    DeviceType type,
    const TensorpipeDeviceTypeConverter* impl) {
  device_type_converter_registry[static_cast<size_t>(type)].store(impl);
}

// 序列化 TensorPipe 消息
std::tuple<tensorpipe::Message, TensorpipeWriteBuffers> tensorpipeSerialize(
    c10::intrusive_ptr<Message> rpcMessage,
    std::vector<c10::Device> devices,
    const std::vector<c10::Stream>& streams) {
  tensorpipe::Message tpMessage;
  TensorpipeWriteBuffers buffers;

  // 元数据
  buffers.type = std::make_unique<MessageType>(rpcMessage->type());
  buffers.id = std::make_unique<int64_t>(rpcMessage->id());
  // kTpMessageTypeIdx = 0
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.type.get(), sizeof(MessageType)});
  // kTpMessageIdIdx = 1
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.id.get(), sizeof(int64_t)});

  // 负载
  buffers.payload = std::move(rpcMessage->payload());
  // TensorPipe 在读写时使用相同的 Message 类，因此即使在写入时不修改它们，也使用非 const 指针。
  char* payloadPtr = buffers.payload.data();
  // kTpMessagePayloadIdx = 2
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{payloadPtr, buffers.payload.size()});

  {
    // 如果存在 Tensor 视图，下面的函数可能会分配新的张量。
    // 在这里应用流保护，以将这些张量分配操作包含到流中。
    c10::MultiStreamGuard guard(streams);
    // 张量
    buffers.tensors = cloneSparseTensors(rpcMessage->tensors()).vec();
  }

  torch::jit::Pickler pickler([&](const void* buf, size_t sz) -> size_t {
    buffers.pickle.insert(
        buffers.pickle.end(),
        static_cast<const char*>(buf),
        static_cast<const char*>(buf) + sz);
    // 返回变量 sz 的值作为函数结果
    return sz;
  });
  // 调用 pickler 对象的 protocol 方法
  pickler.protocol();
  // 将 buffers.tensors 添加到 pickler 对象的 IValue 堆栈中
  pickler.pushIValue(buffers.tensors);
  // 停止 pickler 对象的操作
  pickler.stop();
  // 将 buffers.pickle 的数据作为 payload 添加到 tpMessage 的 payloads 后面
  // kTpMessagePickleIdx = 3
  tpMessage.payloads.push_back(tensorpipe::Message::Payload{
      buffers.pickle.data(), buffers.pickle.size()});
  // 获取 pickler 对象中的 tensorDataVec，这是一个 torch::Tensor 的 vector
  const std::vector<torch::Tensor>& tensorDataVec = pickler.tensorData();
  // 预留足够的空间以容纳 tensorDataVec 中的所有 tensor
  tpMessage.tensors.reserve(tensorDataVec.size());
  // 遍历 tensorDataVec 中的每个 tensor
  for (const auto i : c10::irange(tensorDataVec.size())) {
    // 获取当前循环迭代的 tensor
    const torch::Tensor& tensor = tensorDataVec[i];

    // 获取 tensor 的设备类型转换器
    const TensorpipeDeviceTypeConverter* converter =
        getDeviceTypeConverter(tensor.device().type());
    // 检查转换器是否存在，如果不存在则抛出异常
    TORCH_CHECK(
        converter != nullptr,
        "Attempting to send a Tensor with unexpected device type ",
        tensor.device());

    // 断言确保 tpMessage.tensors 的大小与当前迭代 i 一致
    TORCH_INTERNAL_ASSERT(tpMessage.tensors.size() == i);
    // 准备 tensor 以便发送，返回一个可能拷贝过的 tensor 数据
    std::optional<std::vector<char>> maybeCopiedTensor =
        converter->prepareTensorForSending(
            tensor.storage(), streams, tpMessage);
    // 断言确保 tpMessage.tensors 的大小与当前迭代 i+1 一致
    TORCH_INTERNAL_ASSERT(tpMessage.tensors.size() == i + 1);

    // 根据 tensor 的设备类型确定目标设备，如果 devices 为空或当前设备为 CPU，则使用 kCpuDeviceType
    // 否则使用 kCudaDeviceType，并指定设备索引
    tensorpipe::Device targetDevice = devices.empty() || devices[i].is_cpu()
        ? tensorpipe::Device{tensorpipe::kCpuDeviceType, 0}
        : tensorpipe::Device{tensorpipe::kCudaDeviceType, devices[i].index()};
    // 将目标设备信息移动到当前 tpMessage.tensors 的最后一个元素
    tpMessage.tensors.back().targetDevice = std::move(targetDevice);

    // 如果 maybeCopiedTensor 有值，则将其移动到 buffers.copiedTensors 的末尾
    if (maybeCopiedTensor.has_value()) {
      buffers.copiedTensors.push_back(std::move(maybeCopiedTensor).value());
    }
  }

  // 返回一个 tuple，包含移动后的 tpMessage 和 buffers 对象
  return std::make_tuple(std::move(tpMessage), std::move(buffers));
}

// 分配 Tensorpipe 传输中的内存和缓冲区
std::pair<tensorpipe::Allocation, TensorpipeReadBuffers> tensorpipeAllocate(
    const tensorpipe::Descriptor& tpDescriptor,
    const std::vector<c10::Stream>& streams) {
  // 初始化 Tensorpipe 的内存分配和读取缓冲区
  tensorpipe::Allocation tpAllocation;
  TensorpipeReadBuffers buffers;

  // 断言消息中包含 4 个负载
  TORCH_INTERNAL_ASSERT(
      tpDescriptor.payloads.size() == 4,
      "message expected to contain 4 payloads, whereas it contained ",
      tpDescriptor.payloads.size(),
      " payloads");
  tpAllocation.payloads.resize(tpDescriptor.payloads.size());

  // 断言第一个负载的长度与 MessageType 的大小相等
  TORCH_INTERNAL_ASSERT(
      tpDescriptor.payloads[kTpMessageTypeIdx].length == sizeof(MessageType),
      "first payload expected to contain ",
      sizeof(MessageType),
      " bytes, whereas it contained ",
      tpDescriptor.payloads[kTpMessageTypeIdx].length,
      " bytes");
  // 分配用于存储 MessageType 的缓冲区
  buffers.type = std::make_unique<MessageType>();
  tpAllocation.payloads[kTpMessageTypeIdx].data = buffers.type.get();

  // 断言第二个负载的长度与 int64_t 的大小相等
  TORCH_INTERNAL_ASSERT(
      tpDescriptor.payloads[kTpMessageIdIdx].length == sizeof(int64_t),
      "second payload expected to contain ",
      sizeof(int64_t),
      " bytes, whereas it contained ",
      tpDescriptor.payloads[kTpMessageIdIdx].length,
      " bytes");
  // 分配用于存储 int64_t 的缓冲区
  buffers.id = std::make_unique<int64_t>();
  tpAllocation.payloads[kTpMessageIdIdx].data = buffers.id.get();

  // 分配用于存储消息负载的缓冲区，并设置其大小
  buffers.payload.resize(tpDescriptor.payloads[kTpMessagePayloadIdx].length);
  tpAllocation.payloads[kTpMessagePayloadIdx].data = buffers.payload.data();

  // 分配用于存储 Pickle 数据的缓冲区，并设置其大小
  buffers.pickle.resize(tpDescriptor.payloads[kTpMessagePickleIdx].length);
  tpAllocation.payloads[kTpMessagePickleIdx].data = buffers.pickle.data();

  // 处理每个张量描述并分配接收时所需的内存
  size_t numTensors = tpDescriptor.tensors.size();
  tpAllocation.tensors.reserve(numTensors);
  for (const auto tensorIdx : c10::irange(numTensors)) {
    const tensorpipe::Descriptor::Tensor& tensor =
        tpDescriptor.tensors[tensorIdx];
    // 断言张量的目标设备类型已定义
    TORCH_INTERNAL_ASSERT(tensor.targetDevice.has_value());
    // 转换目标设备类型以便分配内存
    c10::DeviceType targetDeviceType =
        convertDeviceType(tensor.targetDevice->type);

    // 获取设备类型转换器
    const TensorpipeDeviceTypeConverter* converter =
        getDeviceTypeConverter(targetDeviceType);
    // 断言设备类型转换器不为空
    TORCH_INTERNAL_ASSERT(
        converter != nullptr,
        "Attempting to receive a Tensor with unexpected device type ",
        targetDeviceType);

    // 断言分配的张量索引正确
    TORCH_INTERNAL_ASSERT(tpAllocation.tensors.size() == tensorIdx);
    // 断言目标设备索引不超过上限
    TORCH_INTERNAL_ASSERT(
        tensor.targetDevice->index <=
        std::numeric_limits<c10::DeviceIndex>::max());
    // 分配接收张量的内存并返回数据指针
    at::DataPtr dataPtr = converter->allocateTensorForReceiving(
        static_cast<c10::DeviceIndex>(tensor.targetDevice->index),
        tensor.length,
        streams,
        tpAllocation);
    // 断言张量成功分配并索引正确
    TORCH_INTERNAL_ASSERT(tpAllocation.tensors.size() == tensorIdx + 1);

    // 将分配的数据指针移动到张量缓冲区中
    buffers.tensors.push_back(std::move(dataPtr));
  }

  // 返回分配的内存和缓冲区
  return {std::move(tpAllocation), std::move(buffers)};
}

// 反序列化 Tensorpipe 消息
c10::intrusive_ptr<Message> tensorpipeDeserialize(
    // 创建一个函数，接受一个 tensorpipe::Descriptor 对象和一个 TensorpipeReadBuffers 对象作为参数
    // 返回一个 c10::intrusive_ptr<Message> 对象
    auto processMessage(
        tensorpipe::Descriptor&& tpDescriptor,
        TensorpipeReadBuffers&& buffers) {
      // 创建一个空的张量向量
      std::vector<at::Tensor> tensors;
      // 获取 pickle 数据的指针和长度
      const char* pickleData = buffers.pickle.data();
      size_t pickleLen = buffers.pickle.size();
      size_t picklePos = 0;
      // 定义一个 lambda 函数 pickleReadFunc 用于从 pickle 数据中读取数据
      auto pickleReadFunc = [&](char* buf, size_t n) -> size_t {
        // 如果当前位置超出 pickle 数据长度或者 n 为 0，返回 0
        if (picklePos >= pickleLen || n == 0) {
          return 0;
        }
        // 计算需要拷贝的数据长度
        size_t toCopy = std::min(picklePos + n, pickleLen) - picklePos;
        // 将数据从 pickleData 拷贝到 buf 中
        memcpy(buf, pickleData + picklePos, toCopy);
        // 更新 pickle 数据读取位置
        picklePos += toCopy;
        return toCopy;
      };
      // 定义一个 lambda 函数 tensorReadFunc，根据名称从 tensors 中获取对应的数据指针
      auto tensorReadFunc = [&](const std::string& ename) -> at::DataPtr {
        // 将 ename 转换为 unsigned long 类型作为索引，从 buffers.tensors 中获取对应的数据并移动赋值
        unsigned long index = std::stoul(ename);
        return std::move(buffers.tensors.at(index));
      };
    
      // 创建 torch::jit::Unpickler 对象 unpickler，使用 pickleReadFunc 和 tensorReadFunc 作为参数
      // typeResolver 参数为空，因为此处仅处理字符串和张量
      // use_storage_device 参数设置为 true
      torch::jit::Unpickler unpickler(
          pickleReadFunc,
          nullptr,
          nullptr,
          tensorReadFunc,
          {},
          /* use_storage_device*/ true);
    
      // 解析 pickle 数据并获取返回的 IValue
      auto ival = unpickler.parse_ivalue();
      // 将 IValue 中的张量添加到 tensors 中
      for (auto&& t : ival.toTensorList()) {
        tensors.emplace_back(std::move(t));
      }
    
      // 遍历 tensorpipe::Descriptor 中的张量
      for (const auto i : c10::irange(tpDescriptor.tensors.size())) {
        auto& tensor = tpDescriptor.tensors[i];
        // 检查如果目标设备指定了且为 CUDA 设备类型，则进行断言检查
        if (tensor.targetDevice.has_value() &&
            tensor.targetDevice->type == tensorpipe::kCudaDeviceType) {
          // 断言检查：确保接收到的张量位于期望的设备上
          TORCH_INTERNAL_ASSERT(
              tensors[i].device() == indexToDevice(tensor.targetDevice->index),
              "Tensor ",
              i,
              " in message ",
              *buffers.id,
              " was expected to be received on device ",
              tensor.targetDevice->index,
              ", but got it on ",
              tensors[i].device());
        }
      }
    
      // 返回一个包含 payload、tensors、type 和 id 的 Message 对象的 intrusive_ptr
      return c10::make_intrusive<Message>(
          std::move(buffers.payload),
          std::move(tensors),
          *buffers.type,
          *buffers.id);
    }
} // 结束 namespace torch

} // 结束 namespace distributed

} // 结束 namespace rpc

#endif // 使用 TensorPipe
```