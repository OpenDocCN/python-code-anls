# `.\pytorch\torch\csrc\distributed\rpc\python_remote_call.cpp`

```
// 包含头文件：用于远程调用 Python 函数的相关功能
#include <torch/csrc/distributed/rpc/python_remote_call.h>
// 包含头文件：用于处理 RPC 代理的相关功能
#include <torch/csrc/distributed/rpc/rpc_agent.h>
// 包含头文件：用于 JIT 序列化和反序列化 Python 对象的相关功能
#include <torch/csrc/jit/serialization/pickle.h>

// 声明命名空间：torch -> distributed -> rpc
namespace torch {
namespace distributed {
namespace rpc {

// 构造函数：PythonRemoteCall 类的构造函数，初始化成员变量
PythonRemoteCall::PythonRemoteCall(
    SerializedPyObj&& serializedPyObj, // 移动构造 Python 对象的序列化结果
    at::IValue retRRefId, // RRef 的返回 ID
    at::IValue retForkId, // 分叉 ID 的返回值
    const bool isAsyncExecution) // 是否异步执行的标志
    : serializedPyObj_(std::move(serializedPyObj)), // 初始化序列化 Python 对象的成员变量
      retRRefId_(std::move(retRRefId)), // 初始化 RRef 返回 ID 的成员变量
      retForkId_(std::move(retForkId)), // 初始化分叉 ID 返回值的成员变量
      isAsyncExecution_(isAsyncExecution) {} // 初始化异步执行标志的成员变量

// 方法：将 PythonRemoteCall 对象转换为消息对象的实现
c10::intrusive_ptr<Message> PythonRemoteCall::toMessageImpl() && {
  // 将序列化的 Python 对象转换为 IValue 列表
  std::vector<IValue> ivalues = std::move(serializedPyObj_).toIValues();
  // 将返回的 RRef ID、分叉 ID 和异步执行标志添加到 IValue 列表中
  ivalues.emplace_back(retRRefId_);
  ivalues.emplace_back(retForkId_);
  ivalues.emplace_back(isAsyncExecution_);

  // 创建一个空的张量表
  std::vector<torch::Tensor> tensor_table;
  // 使用 JIT 的 pickle 函数将 IValue 列表转换为序列化数据
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  // 创建并返回一个新的消息对象，包含序列化数据、张量表和消息类型
  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensor_table),
      MessageType::PYTHON_REMOTE_CALL);
}

// 静态方法：从消息对象中反序列化出 PythonRemoteCall 对象
std::unique_ptr<PythonRemoteCall> PythonRemoteCall::fromMessage(
    const Message& message) {
  // 获取消息的有效载荷和大小
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  // 使用 JIT 的 unpickle 函数将有效载荷反序列化为 IValue
  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  // 将反序列化的值转换为元组并获取其元素列表
  auto values = value.toTupleRef().elements().vec();

  // 断言反序列化后的值至少包含四个元素
  TORCH_INTERNAL_ASSERT(
      values.size() > 3,
      "Expect at least 4 elements in the unpickled values, but got ",
      values.size());

  // 从反序列化的值中逆向获取异步执行标志、分叉 ID 和 RRef ID
  bool isAsyncExecution = values.back().toBool();
  values.pop_back();
  auto retForkId = std::move(values.back());
  values.pop_back();
  auto retRRefId = std::move(values.back());
  values.pop_back();
  // 使用反序列化的值创建 SerializedPyObj 对象
  auto serializedPyObj = SerializedPyObj::fromIValues(std::move(values));

  // 创建并返回一个新的 PythonRemoteCall 对象
  return std::make_unique<PythonRemoteCall>(
      std::move(serializedPyObj),
      std::move(retRRefId),
      std::move(retForkId),
      isAsyncExecution);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```