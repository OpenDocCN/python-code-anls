# `.\pytorch\torch\csrc\distributed\rpc\script_remote_call.cpp`

```py
// 引入 Torch RPC 代理和远程调用的必要头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>

// 引入 Torch JIT 序列化 pickle 功能的头文件
#include <torch/csrc/jit/serialization/pickle.h>

// Torch 分布式 RPC 的命名空间
namespace torch {
namespace distributed {
namespace rpc {

// ScriptRemoteCall 类的构造函数，接受操作符指针、堆栈、返回的 RRef ID 和 Fork ID
ScriptRemoteCall::ScriptRemoteCall(
    std::shared_ptr<Operator> op,
    std::vector<at::IValue>&& stack,
    const RRefId& retRRefId,
    const ForkId& retForkId)
    : ScriptCall(std::move(op), std::move(stack)),  // 调用基类 ScriptCall 的构造函数
      retRRefId_(retRRefId),  // 初始化返回的 RRef ID
      retForkId_(retForkId) {}  // 初始化返回的 Fork ID

// ScriptRemoteCall 类的构造函数，接受限定名、堆栈、返回的 RRef ID、Fork ID 和是否异步执行标志
ScriptRemoteCall::ScriptRemoteCall(
    const c10::QualifiedName& qualifiedName,
    std::vector<at::IValue>&& stack,
    const RRefId& retRRefId,
    const ForkId& retForkId,
    const bool isAsyncExecution)
    : ScriptCall(qualifiedName, std::move(stack), isAsyncExecution),  // 调用基类 ScriptCall 的构造函数
      retRRefId_(retRRefId),  // 初始化返回的 RRef ID
      retForkId_(retForkId) {}  // 初始化返回的 Fork ID

// 从 IValues 转换为 ScriptRemoteCall 对象的静态方法
std::unique_ptr<ScriptRemoteCall> ScriptRemoteCall::fromIValues(
    std::vector<at::IValue>& ivalues) {
  // 从值中移除最后一个元素并转换为 RRef
  auto retForkId = RRefId::fromIValue(ivalues.back());
  ivalues.pop_back();
  auto retRRefId = ForkId::fromIValue(ivalues.back());
  ivalues.pop_back();

  // 使用 ScriptCall 的静态方法创建 scriptCallPtr
  auto scriptCallPtr = ScriptCall::fromIValues(ivalues);

  // 根据 scriptCallPtr 是否具有操作符选择调用不同的构造函数创建 ScriptRemoteCall 对象
  if (scriptCallPtr->hasOp()) {
    return std::make_unique<ScriptRemoteCall>(
        scriptCallPtr->op(), std::move(ivalues), retRRefId, retForkId);
  } else {
    return std::make_unique<ScriptRemoteCall>(
        scriptCallPtr->qualifiedName(),
        std::move(ivalues),
        retRRefId,
        retForkId,
        scriptCallPtr->isAsyncExecution());
  }
}

// 转换为消息的实现方法，返回一个消息对象的智能指针
c10::intrusive_ptr<Message> ScriptRemoteCall::toMessageImpl() && {
  std::vector<IValue> ivalues;
  ScriptCall::toIValues(ivalues);  // 调用基类 ScriptCall 的方法将成员转换为 IValues

  // 将 retRRefId 和 retForkId 添加到 ivalues 中
  ivalues.emplace_back(retRRefId_.toIValue());
  ivalues.emplace_back(retForkId_.toIValue());

  std::vector<torch::Tensor> tensor_table;  // 创建一个张量表
  auto payload = jit::pickle(
      c10::ivalue::Tuple::create(std::move(ivalues)), &tensor_table);  // 使用 Torch JIT 的 pickle 序列化 ivalues

  // 创建并返回一个 Message 对象
  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensor_table),
      MessageType::SCRIPT_REMOTE_CALL);
}

// 从消息中创建 ScriptRemoteCall 对象的静态方法
std::unique_ptr<ScriptRemoteCall> ScriptRemoteCall::fromMessage(
    const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());  // 获取消息的有效负载数据
  auto payload_size = message.payload().size();  // 获取有效负载数据的大小

  // 使用 Torch JIT 的 unpickle 方法反序列化消息的有效负载
  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),  // 获取当前 RPC 代理的类型解析器
      message.tensors());  // 获取消息中的张量表
  auto values = value.toTupleRef().elements().vec();  // 将反序列化后的值转换为向量

  TORCH_CHECK(!values.empty(), "Malformed message: empty values unpickled");  // 检查是否有有效的值

  // 调用 fromIValues 方法创建并返回 ScriptRemoteCall 对象
  return fromIValues(values);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```