# `.\pytorch\torch\csrc\distributed\rpc\rref_proto.cpp`

```py
// 包含必要的头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/jit/serialization/pickle.h>

// 包含标准库头文件
#include <limits>

// Torch命名空间
namespace torch {
// 分布式命名空间
namespace distributed {
// RPC命名空间
namespace rpc {

// 匿名命名空间，内部函数和变量不对外可见
namespace {

// 将Message转换为IValues元组元素
c10::ivalue::TupleElements toIValues(const Message& message, MessageType type) {
  // 断言消息类型与期望类型一致
  TORCH_INTERNAL_ASSERT(
      type == message.type(),
      "Expecting message of type ",
      type,
      ", but got ",
      message.type());
  
  // 获取消息的payload和大小
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  // 使用pickle反序列化消息payload
  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  
  // 将反序列化后的值转换为Tuple，并返回其元素
  return std::move(*std::move(value).toTuple()).elements();
}

// 将IValues转换为Message
c10::intrusive_ptr<Message> fromIValues(
    std::vector<IValue> ivalues,
    MessageType type) {
  // 创建Tensor表
  std::vector<torch::Tensor> tensor_table;
  
  // 使用pickle序列化IValues
  auto payload = jit::pickle(
      c10::ivalue::Tuple::create(std::move(ivalues)), &tensor_table);
  
  // 创建新的Message对象
  return c10::make_intrusive<Message>(
      std::move(payload), std::move(tensor_table), type);
}

} // namespace

/////////////////////////// RRefMessageBase //////////////////////////////////

// 返回RRefMessageBase的RRefId
const RRefId& RRefMessageBase::rrefId() {
  return rrefId_;
}

/////////////////////////// ForkMessageBase //////////////////////////////////

// 返回ForkMessageBase的ForkId
const ForkId& ForkMessageBase::forkId() {
  return forkId_;
}

// 将ForkMessageBase转换为Message对象
c10::intrusive_ptr<Message> ForkMessageBase::toMessageImpl() && {
  // 使用fromIValues将rrefId和forkId转换为IValues，并创建Message对象
  return fromIValues({rrefId_.toIValue(), forkId_.toIValue()}, type_);
}

// 从Message对象中提取出RRefId和ForkId
std::pair<RRefId, ForkId> ForkMessageBase::fromMessage(
    const Message& message,
    MessageType type) {
  // 将Message转换为IValues
  auto ivalues = toIValues(message, type);

  // 断言IValues的大小为2
  TORCH_INTERNAL_ASSERT(
      ivalues.size() == 2, "ForkMessageBase expects 2 IValue from message.");

  // 返回RRefId和ForkId的pair
  return std::make_pair(
      RRefId::fromIValue(ivalues[0]), ForkId::fromIValue(ivalues[1]));
}

/////////////////////////// RRef Protocol //////////////////////////////////

// 将ScriptRRefFetchCall转换为Message对象
c10::intrusive_ptr<Message> ScriptRRefFetchCall::toMessageImpl() && {
  // 创建包含rrefId和fromWorkerId的IValues向量
  std::vector<at::IValue> ivalues;
  ivalues.reserve(2);
  ivalues.emplace_back(rrefId_.toIValue());
  ivalues.emplace_back(fromWorkerId_);
  
  // 使用fromIValues创建ScriptRRefFetchCall的Message对象
  return fromIValues(std::move(ivalues), MessageType::SCRIPT_RREF_FETCH_CALL);
}

// 从Message对象中创建ScriptRRefFetchCall对象
std::unique_ptr<ScriptRRefFetchCall> ScriptRRefFetchCall::fromMessage(
    const Message& message) {
  // 将Message转换为IValues
  auto values = toIValues(message, MessageType::SCRIPT_RREF_FETCH_CALL);
  
  // 断言IValues的大小为2
  TORCH_INTERNAL_ASSERT(
      values.size() == 2, "ScriptRRefFetchCall expects 2 IValues from message");
  
  // 将第二个IValue转换为worker_id_t，并进行边界检查
  auto id = values[1].toInt();
  TORCH_INTERNAL_ASSERT(
      id >= std::numeric_limits<worker_id_t>::min() &&
          id <= std::numeric_limits<worker_id_t>::max(),
      "ScriptRRefFetchCall fromWorkerId exceeds worker_id_t limit.")
  
  // 创建并返回ScriptRRefFetchCall对象
  return std::make_unique<ScriptRRefFetchCall>(
      worker_id_t(id), RRefId::fromIValue(values[0]));
}
// 返回一个右值引用类型的 Message 指针，表示将 PythonRRefFetchCall 对象转换为消息对象
c10::intrusive_ptr<Message> PythonRRefFetchCall::toMessageImpl() && {
  // 创建一个空的 IValue 向量
  std::vector<at::IValue> ivalues;
  // 预留空间以存储两个 IValue
  ivalues.reserve(2);
  // 将 rrefId_ 转换为 IValue，并添加到 ivalues 中
  ivalues.emplace_back(rrefId_.toIValue());
  // 将 fromWorkerId_ 添加到 ivalues 中
  ivalues.emplace_back(fromWorkerId_);
  // 使用 fromIValues 函数从 ivalues 创建一个 Message 指针，指定消息类型为 PYTHON_RREF_FETCH_CALL
  return fromIValues(std::move(ivalues), MessageType::PYTHON_RREF_FETCH_CALL);
}

// 从 Message 对象创建 PythonRRefFetchCall 对象的唯一指针
std::unique_ptr<PythonRRefFetchCall> PythonRRefFetchCall::fromMessage(
    const Message& message) {
  // 调用 toIValues 函数从消息中提取 IValue 向量
  auto values = toIValues(message, MessageType::PYTHON_RREF_FETCH_CALL);
  // 断言提取的值数量为 2，确保消息符合预期
  TORCH_INTERNAL_ASSERT(
      values.size() == 2, "PythonRRefFetchCall expects 2 IValues from message");
  // 将第二个 IValue 转换为整数，并进行范围断言
  auto id = values[1].toInt();
  TORCH_INTERNAL_ASSERT(
      id >= std::numeric_limits<worker_id_t>::min() &&
          id <= std::numeric_limits<worker_id_t>::max(),
      "PythonRRefFetchCall fromWorkerId exceeds worker_id_t limit.")
  // 使用提取的值创建 PythonRRefFetchCall 对象的唯一指针
  return std::make_unique<PythonRRefFetchCall>(
      worker_id_t(id), RRefId::fromIValue(values[0]));
}

// 返回 values_ 成员变量的常引用，该变量是存储在 RRefFetchRet 对象中的 IValue 向量
const std::vector<at::IValue>& RRefFetchRet::values() {
  return values_;
}

// 返回一个右值引用类型的 Message 指针，表示将 RRefFetchRet 对象转换为消息对象
c10::intrusive_ptr<Message> RRefFetchRet::toMessageImpl() && {
  // 使用 fromIValues 函数从 values_ 成员变量创建一个 Message 指针，指定消息类型为 type_
  return fromIValues(values_, type_);
}

// 从 Message 对象创建 ScriptRRefFetchRet 对象的唯一指针
std::unique_ptr<ScriptRRefFetchRet> ScriptRRefFetchRet::fromMessage(
    const Message& message) {
  // 调用 toIValues 函数从消息中提取 IValue 向量，指定消息类型为 SCRIPT_RREF_FETCH_RET
  auto values = toIValues(message, MessageType::SCRIPT_RREF_FETCH_RET);
  // 断言提取的值数量为 1，确保消息符合预期
  TORCH_INTERNAL_ASSERT(
      values.size() == 1,
      "RRef of IValue should contain a single IValue, but got ",
      values.size());
  // 使用提取的值创建 ScriptRRefFetchRet 对象的唯一指针
  return std::make_unique<ScriptRRefFetchRet>(std::move(values).vec());
}

// 从 Message 对象创建 PythonRRefFetchRet 对象的唯一指针
std::unique_ptr<PythonRRefFetchRet> PythonRRefFetchRet::fromMessage(
    const Message& message) {
  // 调用 toIValues 函数从消息中提取 IValue 向量，指定消息类型为 PYTHON_RREF_FETCH_RET
  return std::make_unique<PythonRRefFetchRet>(
      toIValues(message, MessageType::PYTHON_RREF_FETCH_RET).vec());
}

// 从 Message 对象创建 RRefUserDelete 对象的唯一指针
std::unique_ptr<RRefUserDelete> RRefUserDelete::fromMessage(
    const Message& message) {
  // 调用 ForkMessageBase::fromMessage 函数从消息中提取一对值，指定消息类型为 RREF_USER_DELETE
  auto pair =
      ForkMessageBase::fromMessage(message, MessageType::RREF_USER_DELETE);
  // 使用提取的值创建 RRefUserDelete 对象的唯一指针
  return std::make_unique<RRefUserDelete>(pair.first, pair.second);
}

// 从 Message 对象创建 RemoteRet 对象的唯一指针
std::unique_ptr<RemoteRet> RemoteRet::fromMessage(const Message& message) {
  // 调用 ForkMessageBase::fromMessage 函数从消息中提取一对值，指定消息类型为 REMOTE_RET
  auto pair = ForkMessageBase::fromMessage(message, MessageType::REMOTE_RET);
  // 使用提取的值创建 RemoteRet 对象的唯一指针
  return std::make_unique<RemoteRet>(pair.first, pair.second);
}

// 返回 forkId_ 成员变量的常引用，该变量是存储在 RRefChildAccept 对象中的 ForkId 对象
const ForkId& RRefChildAccept::forkId() const {
  return forkId_;
}

// 返回一个右值引用类型的 Message 指针，表示将 RRefChildAccept 对象转换为消息对象
c10::intrusive_ptr<Message> RRefChildAccept::toMessageImpl() && {
  // 使用 fromIValues 函数从包含 forkId_.toIValue() 的向量创建一个 Message 指针，指定消息类型为 RREF_CHILD_ACCEPT
  return fromIValues({forkId_.toIValue()}, MessageType::RREF_CHILD_ACCEPT);
}

// 从 Message 对象创建 RRefChildAccept 对象的唯一指针
std::unique_ptr<RRefChildAccept> RRefChildAccept::fromMessage(
    const Message& message) {
  // 调用 toIValues 函数从消息中提取 IValue 向量，指定消息类型为 RREF_CHILD_ACCEPT
  auto values = toIValues(message, MessageType::RREF_CHILD_ACCEPT);
  // 断言提取的值数量为 1，确保消息符合预期
  TORCH_INTERNAL_ASSERT(values.size() == 1, "Expect 1 IValues from message.");

  // 使用提取的值创建 RRefChildAccept 对象的唯一指针
  return std::make_unique<RRefChildAccept>(ForkId::fromIValue(values.back()));
}

// 从 Message 对象创建 RRefForkRequest 对象的唯一指针
std::unique_ptr<RRefForkRequest> RRefForkRequest::fromMessage(
    const Message& message) {
  // 调用 ForkMessageBase::fromMessage 函数从消息中提取一对值，指定消息类型为 RREF_FORK_REQUEST
  auto pair =
      ForkMessageBase::fromMessage(message, MessageType::RREF_FORK_REQUEST);
  // 使用提取的值创建 RRefForkRequest 对象的唯一指针
  return std::make_unique<RRefForkRequest>(pair.first, pair.second);
}
# 创建并返回一个 RRefAck 对象对应的消息对象，使用移动语义
c10::intrusive_ptr<Message> RRefAck::toMessageImpl() && {
  return c10::make_intrusive<Message>(
      std::vector<char>{}, std::vector<torch::Tensor>{}, MessageType::RREF_ACK);
}

# 根据给定的消息创建并返回一个 RRefAck 对象的唯一指针
std::unique_ptr<RRefAck> RRefAck::fromMessage(const Message& message) {
  # 内部断言，检查消息类型是否为 RREF_ACK
  TORCH_INTERNAL_ASSERT(
      message.type() == MessageType::RREF_ACK,
      "Message type miss match, expect ",
      MessageType::RREF_ACK,
      ", but got ",
      message.type());
  # 创建并返回一个 RRefAck 对象的唯一指针
  return std::make_unique<RRefAck>();
}
```