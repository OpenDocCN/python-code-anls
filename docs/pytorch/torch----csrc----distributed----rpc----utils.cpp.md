# `.\pytorch\torch\csrc\distributed\rpc\utils.cpp`

```
// 包含头文件，用于远程过程调用（RPC）的工具函数和类
#include <torch/csrc/distributed/rpc/utils.h>

// 包含格式化输出的库
#include <fmt/format.h>

// 包含自动微分相关的头文件
#include <torch/csrc/autograd/profiler.h>

// 包含远程自动微分的RPC消息请求和响应类的头文件
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_resp.h>
#include <torch/csrc/distributed/autograd/utils.h>

// 包含RPC的远程分析器管理类的头文件
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>

// 包含Python调用相关的头文件
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>

// 包含RPC的远程引用（RRef）协议的头文件
#include <torch/csrc/distributed/rpc/rref_proto.h>

// 包含脚本调用相关的头文件
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>

// 包含序列化和反序列化的头文件
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/unpickler.h>

// 包含C++工具类的头文件
#include <c10/util/irange.h>

// 使用torch命名空间下的autograd和profiler子命名空间
using namespace torch::autograd::profiler;

// 使用torch::distributed::rpc命名空间下的所有内容
namespace torch {
namespace distributed {
namespace rpc {

// 匿名命名空间，用于定义本文件内部的私有函数或变量
namespace {

// 处理远程分析事件的函数，接收一个RpcWithProfilingResp对象的引用
void processRemoteProfiledEvents(
    autograd::RpcWithProfilingResp& rpcWithProfilingResp) {

  // 检查分析器是否启用
  auto enabled = profilerEnabled();
  TORCH_CHECK(
      enabled,
      "Profiler was expected to be enabled. This can happen in callback "
      " continuations that run in different threads, and the TLS of the "
      " profiler was not propagated.");

  // 获取RpcWithProfilingResp对象中的分析事件列表
  std::vector<LegacyEvent> events = rpcWithProfilingResp.getProfiledEvents();
  
  // 获取分析ID
  const auto& profilingId = rpcWithProfilingResp.getProfilingId();

  // 获取远程分析器管理器的单例对象
  auto& remoteProfilerManager = RemoteProfilerManager::getInstance();
  
  // 根据分析ID检索RPC分析键
  auto key = remoteProfilerManager.retrieveRPCProfilingKey(profilingId);

  // 删除分析ID对应的键
  remoteProfilerManager.eraseKey(profilingId);

  // 构建键的前缀字符串
  auto keyPrefixStr = key + rpc::REMOTE_PROFILING_KEY_PREFIX;

  // 将每个事件的名称前缀设置为键的前缀
  std::for_each(
      events.begin(), events.end(), [&keyPrefixStr](LegacyEvent& event) {
        std::string name = keyPrefixStr + std::string(event.name());
        event.setName(at::StringView(name));
      });

  // 将事件列表添加到线程本地的分析器中
  addEventList(std::move(events));
}

} // namespace

// 定义全局常量kRPCErrorPrefix，并初始化为"RPCErr"
const std::string kRPCErrorPrefix = std::string("RPCErr");
// 根据给定的 JitFuture 对象获取 RPC 错误类型
RPCErrorType getRPCErrorType(const JitFuture& jitFuture) {
  // 断言确保 jitFuture 中存在错误
  TORCH_INTERNAL_ASSERT(
      jitFuture.hasError(),
      "JitFuture of Message passed to getRPCErrorType does not have an error.");

  // 尝试获取错误信息字符串
  auto err = jitFuture.tryRetrieveErrorMessage();
  // 查找错误信息中的错误前缀 kRPCErrorPrefix 的位置
  size_t pos = err.find(kRPCErrorPrefix);
  if (pos != std::string::npos) {
    // 解析 RPCErrorType 类型
    auto errStartIdx =
        pos + torch::distributed::rpc::kRPCErrorPrefix.size() + 1;
    // 查找错误信息中的冒号位置，确定错误类型的结束位置
    auto errEndIdx = err.find(':', errStartIdx);
    if (errEndIdx == std::string::npos) {
      // 错误格式不正确，返回未知错误类型
      return RPCErrorType::UNKNOWN_ERROR;
    }
    // 提取错误类型字符串并转换为 RPCErrorType 枚举值
    auto errStr = err.substr(errStartIdx, errEndIdx - errStartIdx);
    auto errType = static_cast<RPCErrorType>(std::stoi(errStr));
    return errType;
  } else {
    // 错误信息中未找到错误前缀，返回未知错误类型
    return RPCErrorType::UNKNOWN_ERROR;
  }
}

// 根据给定的 rpcErrorStr 和 errorType 创建 RPC 错误信息字符串
std::string makeRPCError(
    const std::string& rpcErrorStr,
    RPCErrorType errorType) {
  // 格式化错误信息字符串
  return fmt::format(
      "{}:{}:{}",
      torch::distributed::rpc::kRPCErrorPrefix,
      static_cast<int>(errorType),
      rpcErrorStr);
}

// 反序列化给定 Message 对象中的 RpcCommandBase 派生类
std::unique_ptr<RpcCommandBase> deserializeRequest(const Message& request) {
  switch (request.type()) {
    case MessageType::SCRIPT_CALL: {
      // 从 Message 中创建 ScriptCall 对象
      return ScriptCall::fromMessage(request);
    }
    case MessageType::PYTHON_CALL: {
      // 从 Message 中创建 PythonCall 对象
      return PythonCall::fromMessage(request);
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      // 从 Message 中创建 ScriptRemoteCall 对象
      return ScriptRemoteCall::fromMessage(request);
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      // 从 Message 中创建 PythonRemoteCall 对象
      return PythonRemoteCall::fromMessage(request);
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      // 从 Message 中创建 ScriptRRefFetchCall 对象
      return ScriptRRefFetchCall::fromMessage(request);
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      // 从 Message 中创建 PythonRRefFetchCall 对象
      return PythonRRefFetchCall::fromMessage(request);
    }
    case MessageType::RREF_USER_DELETE: {
      // 从 Message 中创建 RRefUserDelete 对象
      return RRefUserDelete::fromMessage(request);
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      // 从 Message 中创建 RRefChildAccept 对象
      return RRefChildAccept::fromMessage(request);
    }
    case MessageType::RREF_FORK_REQUEST: {
      // 从 Message 中创建 RRefForkRequest 对象
      return RRefForkRequest::fromMessage(request);
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      // 从 Message 中创建 RpcWithAutograd 对象
      return autograd::RpcWithAutograd::fromMessage(request);
    }
    case MessageType::BACKWARD_AUTOGRAD_REQ: {
      // 从 Message 中创建 PropagateGradientsReq 对象
      return autograd::PropagateGradientsReq::fromMessage(request);
    }
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: {
      // 从 Message 中创建 CleanupAutogradContextReq 对象
      return autograd::CleanupAutogradContextReq::fromMessage(request);
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
      // 从 Message 中创建 RpcWithProfilingReq 对象
      return autograd::RpcWithProfilingReq::fromMessage(request);
    }
    case MessageType::RREF_BACKWARD_REQ: {
      // 从 Message 中创建 RRefBackwardReq 对象
      return autograd::RRefBackwardReq::fromMessage(request);
    }
    default: {
      // 如果 request.type() 不支持，触发内部断言错误
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", request.type(), " not supported.");
    }
  }
}
// 将消息反序列化为特定类型的 RpcCommandBase 指针
std::unique_ptr<RpcCommandBase> deserializeResponse(
    const Message& response,
    MessageType& wrappedMsgType) {
  switch (response.type()) {
    // 如果响应类型是 SCRIPT_RET
    case MessageType::SCRIPT_RET: {
      // 调用 ScriptResp 类的静态方法从消息中创建对象并返回其 unique_ptr
      return ScriptResp::fromMessage(response);
    }
    // 如果响应类型是 PYTHON_RET
    case MessageType::PYTHON_RET: {
      // 调用 PythonResp 类的静态方法从消息中创建对象并返回其 unique_ptr
      return PythonResp::fromMessage(response);
    }
    // 如果响应类型是 REMOTE_RET
    case MessageType::REMOTE_RET: {
      // 调用 RemoteRet 类的静态方法从消息中创建对象并返回其 unique_ptr
      return RemoteRet::fromMessage(response);
    }
    // 如果响应类型是 SCRIPT_RREF_FETCH_RET
    case MessageType::SCRIPT_RREF_FETCH_RET: {
      // 调用 ScriptRRefFetchRet 类的静态方法从消息中创建对象并返回其 unique_ptr
      return ScriptRRefFetchRet::fromMessage(response);
    }
    // 如果响应类型是 PYTHON_RREF_FETCH_RET
    case MessageType::PYTHON_RREF_FETCH_RET: {
      // 调用 PythonRRefFetchRet 类的静态方法从消息中创建对象并返回其 unique_ptr
      return PythonRRefFetchRet::fromMessage(response);
    }
    // 如果响应类型是 RREF_ACK
    case MessageType::RREF_ACK: {
      // 调用 RRefAck 类的静态方法从消息中创建对象并返回其 unique_ptr
      return RRefAck::fromMessage(response);
    }
    // 如果响应类型是 FORWARD_AUTOGRAD_RESP
    case MessageType::FORWARD_AUTOGRAD_RESP: {
      // 调用 autograd::RpcWithAutograd 类的静态方法从消息中创建对象并返回其 unique_ptr
      std::unique_ptr<RpcCommandBase> rpcPtr =
          autograd::RpcWithAutograd::fromMessage(response);
      RpcCommandBase& rpc = *rpcPtr;
      auto& rpcWithAutograd = static_cast<autograd::RpcWithAutograd&>(rpc);

      // 需要反转设备映射以进行分布式自动微分的反向传播
      DeviceMap reverseDeviceMap;
      for (const auto& mapEntry : rpcWithAutograd.deviceMap()) {
        reverseDeviceMap.insert({mapEntry.second, mapEntry.first});
      }

      // 添加 'recv' 自动微分函数
      addRecvRpcBackward(
          rpcWithAutograd.autogradMetadata(),
          rpcWithAutograd.tensors(),
          rpcWithAutograd.fromWorkerId(),
          reverseDeviceMap);

      // 设置包装的消息类型
      wrappedMsgType = rpcWithAutograd.wrappedMessageType();

      // 移动 rpcWithAutograd 对象的包装 RPC 并返回其 unique_ptr
      return std::move(rpcWithAutograd).moveWrappedRpc();
    }
    // 如果响应类型是 BACKWARD_AUTOGRAD_RESP
    case MessageType::BACKWARD_AUTOGRAD_RESP: {
      // 调用 autograd::PropagateGradientsResp 类的静态方法从消息中创建对象并返回其 unique_ptr
      return autograd::PropagateGradientsResp::fromMessage(response);
    }
    // 如果响应类型是 CLEANUP_AUTOGRAD_CONTEXT_RESP
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP: {
      // 调用 autograd::CleanupAutogradContextResp 类的静态方法从消息中创建对象并返回其 unique_ptr
      return autograd::CleanupAutogradContextResp::fromMessage(response);
    }
    // 如果响应类型是 RUN_WITH_PROFILING_RESP
    case MessageType::RUN_WITH_PROFILING_RESP: {
      // 调用 autograd::RpcWithProfilingResp 类的静态方法从消息中创建对象并返回其 unique_ptr
      std::unique_ptr<RpcCommandBase> rpcPtr =
          autograd::RpcWithProfilingResp::fromMessage(response);
      RpcCommandBase& rpc = *rpcPtr;
      auto& rpcWithProfilingResp =
          static_cast<autograd::RpcWithProfilingResp&>(rpc);

      // 处理远程分析事件
      processRemoteProfiledEvents(rpcWithProfilingResp);

      // 设置包装的消息类型
      wrappedMsgType = rpcWithProfilingResp.wrappedMessageType();

      // 移动 rpcWithProfilingResp 对象的包装 RPC 并返回其 unique_ptr
      return std::move(rpcWithProfilingResp).moveWrappedRpc();
    }
    // 如果响应类型是 RREF_BACKWARD_RESP
    case MessageType::RREF_BACKWARD_RESP: {
      // 调用 autograd::RRefBackwardResp 类的静态方法从消息中创建对象并返回其 unique_ptr
      return autograd::RRefBackwardResp::fromMessage(response);
    }
    // 默认情况下，如果响应类型未知，则断言失败并输出错误信息
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Response type ", response.type(), " not supported.");
    }
  }
}

// 将 RpcCommandBase 对象反序列化为对应的 IValue 对象
IValue deserializeResptoIValueInternal(
    RpcCommandBase& rpc,
    MessageType messageType) {
  switch (messageType) {
    // 如果消息类型是 SCRIPT_RET
    case MessageType::SCRIPT_RET: {
      // 将 rpc 强制转换为 ScriptResp 对象，并返回其值
      auto& ret = static_cast<ScriptResp&>(rpc);
      return ret.value();
    }
    default: {
      // 如果消息类型不在预期范围内，则触发断言错误
      TORCH_INTERNAL_ASSERT(
          false,
          "Response type ",
          messageType,
          " is not supported to be deserialized to IValue.");
    }
}

// Deserialize a response message to an IValue representation.
//
// This function takes a Message object and extracts its type.
// It then calls deserializeResponse() to obtain a pointer to the response object.
// Finally, it invokes deserializeResptoIValueInternal() to convert the response object to an IValue.
IValue deserializeRespToIValue(const Message& message) {
  MessageType msgType = message.type();
  auto response = deserializeResponse(message, msgType);
  return deserializeResptoIValueInternal(*response, msgType);
}

namespace {

// Helper function for parsing wire format sections.
//
// This function parses a given block of data into sections, each represented by a name and size.
// The format of the data block is structured as:
//    section_name_1 size_1\n
//    section_name_2 size_2\n
//    ..
//    \n
//    [sections in order]
//
// Sections include "payload" for payload bits, "meta" for unpickler metadata,
// and numbered sections for tensor data.
//
// Note: This format is subject to change and is primarily used for RPCs rather than disk storage.
std::unordered_map<std::string, std::pair<const char*, size_t>>
parseWireSections(const void* data, size_t data_size) {
  const char* ptr = static_cast<const char*>(data);
  const char* endp = ptr + data_size;

  std::vector<std::pair<std::string, size_t>> headerEnts;
  bool ok = false;
  while (ptr != endp) {
    if (*ptr == '\n') {
      ok = true; // The only "correct" exit point.
      ++ptr;
      break;
    }
    // Parse name
    const char* namePtr = ptr;
    while (ptr != endp && *ptr != ' ') {
      ptr++;
    }
    if (ptr == endp) {
      break;
    }
    std::string name(namePtr, ptr - namePtr);
    if (++ptr == endp) {
      break; // past the ' '
    }
    // Parse size
    const char* sizePtr = ptr;
    while (ptr != endp && *ptr != '\n') {
      ptr++;
    }
    if (ptr == endp) {
      break;
    }
    size_t sz = std::stoll(std::string(sizePtr, ptr - sizePtr));
    headerEnts.emplace_back(name, sz);
    ++ptr; // past the '\n'
  }
  if (!ok) {
    TORCH_CHECK(false, "failed parse");
  }

  std::unordered_map<std::string, std::pair<const char*, size_t>> out;
  for (const auto& headerEnt : headerEnts) {
    out[headerEnt.first] = {ptr, headerEnt.second};
    ptr += headerEnt.second;
  }
  if (ptr != endp) {
    TORCH_CHECK(false, "failed bounds");
  }
  return out;
}

static const char* kMeta = "meta";
static const char* kPayload = "payload";
}; // namespace

// Clone sparse tensors from a vector of tensors.
//
// This function iterates over a vector of tensors and checks if each tensor's storage
// is worth recopying for serialization purposes.
// It creates a new list of cloned tensors (c10::List<at::Tensor> pTensors) and returns it.
c10::List<at::Tensor> cloneSparseTensors(
    const std::vector<at::Tensor>& tensors) {
  auto worthRecopying = [](const at::Tensor& t) -> bool {
    if (!t.has_storage()) {
      return false; // avoid throwing below.
    }
    auto storageSize = t.storage().nbytes();
    auto usefulSize = t.element_size() * t.numel();
    constexpr size_t kMinMultiple = 2;
    constexpr size_t kMinRecopyBytes = 8 * 1024;
    return storageSize >= kMinRecopyBytes &&
        storageSize >= usefulSize * kMinMultiple;
  };
  c10::List<at::Tensor> pTensors;
  pTensors.reserve(tensors.size());
  for (const auto& t : tensors) {
    // 将传入的张量 t 根据条件判断是否需要复制，然后将其添加到 pTensors 的末尾
    // 如果 worthRecopying(t) 返回 true，则通过 t.clone() 创建 t 的副本并添加
    // 否则直接将 t 添加到 pTensors
    pTensors.push_back(worthRecopying(t) ? t.clone() : t);
  }
  // 返回存储所有处理过的张量的 pTensors 容器
  return pTensors;
}

// 将 payload 和 tensors 序列化成字符串表示
std::string wireSerialize(
    const std::vector<char>& payload,  // 输入的 payload 字符串向量
    const std::vector<at::Tensor>& tensors) {  // 输入的 tensor 向量
  // 检查每个 tensor 是否在 CPU 上，RPC 后端只支持 CPU 上的 tensor
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        tensor.device().is_cpu(),
        "ProcessGroup RPC backend only supports",
        " CPU tensors, please move your tensors to CPU before sending ",
        "them over RPC. Found tensor on device: ",
        tensor.device());
  }

  // 定义保存各种条目的结构体
  struct Ent {
    std::string name;     // 条目名称
    const char* data;     // 数据指针
    size_t size;          // 数据大小
  };
  std::vector<Ent> entries;     // 存储所有条目
  std::string metaEntry;        // 存储元数据条目
  std::vector<at::Tensor> tensorData;  // 存储 tensor 数据

  if (!payload.empty()) {
    entries.push_back({kPayload, payload.data(), payload.size()});  // 添加 payload 条目
  }

  if (!tensors.empty()) {
    // 使用 Pickler 序列化 tensors
    torch::jit::Pickler pickler([&](const void* buf, size_t sz) -> size_t {
      metaEntry.append(static_cast<const char*>(buf), sz);  // 添加到 metaEntry 中
      return sz;
    });
    pickler.protocol();  // 设置 Pickler 协议
    pickler.pushIValue(cloneSparseTensors(tensors));  // 将 tensors 序列化到 Pickler 中
    pickler.stop();  // 停止 Pickler 操作
    tensorData = pickler.tensorData();  // 获取 Pickler 中的 tensor 数据
    entries.push_back({kMeta, metaEntry.data(), metaEntry.size()});  // 添加 meta 数据条目
    for (const auto i : c10::irange(tensorData.size())) {
      // 为每个 tensor 构造可写的数据条目
      auto writeableTensorData = jit::getWriteableTensorData(tensorData[i]);
      entries.push_back(
          {std::to_string(i),
           writeableTensorData.data(),
           writeableTensorData.sizeInBytes()});
    }
  }

  std::string header;  // 存储头部信息
  size_t tot = 0;      // 总数据大小
  // 构造头部信息，包括条目名称和大小
  for (const auto& e : entries) {
    tot += e.size;  // 计算总大小
    header.append(e.name)
        .append(" ")
        .append(std::to_string(e.size))
        .append("\n");
  }
  header.push_back('\n');  // 添加空行到头部

  std::string out;  // 最终输出字符串
  out.reserve(header.size() + tot);  // 预留空间
  out.append(header);  // 添加头部信息到输出
  // 将所有条目的数据添加到输出中
  for (const auto& e : entries) {
    out.append(e.data, e.size);
  }
  return out;  // 返回序列化后的字符串表示
}

// 反序列化给定的数据，返回 payload 和 tensors
std::pair<std::vector<char>, std::vector<at::Tensor>> wireDeserialize(
    const void* data,     // 输入的数据指针
    size_t data_size) {   // 输入数据的大小
  auto sections = parseWireSections(data, data_size);  // 解析输入数据的各个部分

  std::vector<char> payload;  // 存储 payload 数据
  // 获取并存储 payload 数据
  auto payloadIt = sections.find(kPayload);
  if (payloadIt != sections.end() && payloadIt->second.second != 0) {
    payload.assign(
        payloadIt->second.first,
        payloadIt->second.first + payloadIt->second.second);
  }

  std::vector<at::Tensor> tensors;  // 存储 tensors 数据
  // 获取并处理 meta 数据
  auto metaIt = sections.find(kMeta);
  if (metaIt != sections.end()) {
    const auto& metaData = metaIt->second;  // 获取 meta 数据
    size_t metaDataPos = 0;

    // TODO: Add further deserialization logic if needed
  }

  // 返回 payload 和 tensors 的 pair
  return {payload, tensors};
}
    // 定义一个 lambda 函数 metaDataReadFunc，用于读取元数据
    auto metaDataReadFunc = [&](char* buf, size_t n) -> size_t {
      // 如果已经读取完所有元数据或者请求读取长度为0，则直接返回0
      if (metaDataPos >= metaData.second || n == 0) {
        return 0;
      }
      // 计算实际需要复制的字节数，避免越界访问
      size_t toCopy = std::min(metaDataPos + n, metaData.second) - metaDataPos;
      // 将元数据复制到指定的缓冲区中
      memcpy(buf, metaData.first + metaDataPos, toCopy);
      // 更新已读取的元数据位置
      metaDataPos += toCopy;
      // 返回实际复制的字节数
      return toCopy;
    };

    // 定义一个 lambda 函数 sectionReadFunc，用于读取特定名称的部分数据
    auto sectionReadFunc = [&](const std::string& ename) -> at::DataPtr {
      // 查找给定名称的部分数据
      auto it = sections.find(ename);
      // 如果未找到对应的部分数据，抛出错误信息
      if (it == sections.end()) {
        TORCH_CHECK(false, "Couldn't find entity " + ename);
      }
      // 获取部分数据的引用
      const auto& idat = it->second;
      // 分配内存用于存储数据
      auto dptr = at::getCPUAllocator()->allocate(idat.second);
      // 如果部分数据大小不为0，则复制数据到分配的内存中
      if (idat.second != 0) {
        memcpy(dptr.get(), idat.first, idat.second);
      }
      // 返回分配的数据指针
      return dptr;
    };

    // 创建一个 Unpickler 对象 unpickler，用于反序列化数据
    // 在这里不需要传递 typeResolver，因为它只处理字符串和张量
    torch::jit::Unpickler unpickler(
        metaDataReadFunc, nullptr, nullptr, sectionReadFunc, {});
    
    // 解析数据并返回一个 IValue 对象 ival
    auto ival = unpickler.parse_ivalue();
    
    // 将 ival 中的张量逐个移动到 tensors 容器中
    for (auto&& t : ival.toTensorList()) {
      tensors.emplace_back(std::move(t));
    }
  }
  // 返回包含 payload 和 tensors 的 std::pair 对象
  return {std::move(payload), std::move(tensors)};
}

// 将额外负载数据写入原始负载的末尾
void writeWrappedPayload(
    std::vector<char>& originalPayload,
    std::vector<char>& additionalPayload) {
  // 将额外负载数据插入到原始负载的末尾
  originalPayload.insert(
      originalPayload.end(),
      additionalPayload.begin(),
      additionalPayload.end());

  // 添加额外负载数据的大小信息
  int64_t indexToWrite = originalPayload.size();
  originalPayload.resize(originalPayload.size() + sizeof(int64_t));
  const int64_t additionalPayloadSize = additionalPayload.size();
  torch::utils::THP_encodeInt64Buffer(
      reinterpret_cast<uint8_t*>(originalPayload.data()) + indexToWrite,
      &additionalPayloadSize,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);
}

// 从负载中读取包装的额外负载数据
std::vector<at::IValue> readWrappedPayload(
    std::vector<char>& payload,
    const rpc::Message& message) {
  // 读取并移除负载中的额外负载数据
  int64_t additionalPayloadSize;
  TORCH_INTERNAL_ASSERT(payload.size() >= sizeof(int64_t));
  size_t indexToRead = payload.size() - sizeof(int64_t);
  torch::utils::THP_decodeInt64Buffer(
      &additionalPayloadSize,
      reinterpret_cast<uint8_t*>(payload.data()) + indexToRead,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);
  payload.resize(indexToRead);

  // 校验负载大小是否正确
  TORCH_INTERNAL_ASSERT(
      additionalPayloadSize > 0 &&
          static_cast<int64_t>(payload.size()) > additionalPayloadSize,
      "Wrong payload sizes: payload.size() is ",
      payload.size(),
      " but additional payload size is ",
      additionalPayloadSize);

  // 获取包装负载的起始位置，并反序列化为 IValue 元组
  auto wrappedPayloadBegin =
      static_cast<const char*>(message.payload().data()) + payload.size() -
      additionalPayloadSize;
  std::vector<torch::Tensor> tensorTable;
  IValue tuple = jit::unpickle(
      wrappedPayloadBegin,
      additionalPayloadSize,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      tensorTable);
  std::vector<at::IValue> tupleElements = tuple.toTupleRef().elements().vec();
  payload.resize(payload.size() - additionalPayloadSize);
  return tupleElements;
}

// 将事件列表添加到远程分析事件中
void populateRemoteProfiledEvents(
    std::vector<LegacyEvent>& profiledEvents,
    const ProfilerConfig& profilingConfig,
    const std::vector<std::vector<LegacyEvent>>& eventLists) {
  // 将所有事件收集到一个向量中
  for (auto& l : eventLists) {
    for (auto& e : l) {
      profiledEvents.push_back(e);
    }
  }
  
  // 查找 "__start_profile" 事件
  bool cudaProfilingEnabled = profilingConfig.state == ProfilerState::CUDA;
  const LegacyEvent* profilerStart = nullptr;

  for (auto& e : profiledEvents) {
    if (std::string(e.name()) == "__start_profile") {
      profilerStart = &e;
      break;
    }
  }
  
  // 必须能够找到 "__start_profile" 事件
  TORCH_CHECK(
      profilerStart != nullptr, "Expected to find __start_profile event.");

  if (cudaProfilingEnabled) {
    // 反序列化的事件缺少相应的 CUDA 事件，无法在接收端使用 cudaEventElapsedTime。为了避免这种情况，
    // 在此未完待续...
    // 创建一个映射表，用于存储 CUDA 事件的推送（Push）事件及其对应的弹出（Pop）事件的映射关系，
    // 以便后续用于计算 CUDA 的经过时间。
    std::unordered_map<at::RecordFunctionHandle, const LegacyEvent*> startEvents;
    
    // 遍历所有的 profiledEvents，找到所有的 CUDA 事件并将推送事件（PushRange）的信息存入映射表中。
    for (auto& e : profiledEvents) {
      // 检查事件是否与 CUDA 相关
      if (e.hasCuda()) {
        // 如果事件类型是推送事件（PushRange），将其记录到映射表中
        if (e.kind() == EventKind::PushRange) {
          startEvents[e.handle()] = &e;
        }
      }
    }
    
    // 再次遍历所有的 profiledEvents，处理弹出事件（PopRange），并根据映射表设置 CUDA 时间。
    for (auto& e : profiledEvents) {
      // 检查事件是否与 CUDA 相关
      if (e.hasCuda()) {
        // 如果事件类型是弹出事件（PopRange）
        if (e.kind() == EventKind::PopRange) {
          // 查找映射表中是否存在对应的推送事件
          auto it = startEvents.find(e.handle());
          if (it != startEvents.end()) {
            // 如果找到对应的推送事件，则设置 CUDA 时间为推送事件到弹出事件的经过时间
            e.setCudaUs(it->second->cudaElapsedUs(e));
          } else {
            // 如果未找到对应的推送事件，记录警告并设置 CUDA 时间为零
            TORCH_WARN("Found a pop event without a corresponding push event");
            e.setCudaUs(0);
          }
        } else {
          // 如果事件不是弹出事件，将 CUDA 时间设置为零
          e.setCudaUs(0);
        }
      }
    }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```