# `.\pytorch\torch\csrc\distributed\rpc\rref_impl.cpp`

```py
#include <torch/csrc/distributed/rpc/rref_impl.h>

#include <ATen/record_function.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/utils.h>

// 匿名命名空间，定义了一个辅助函数和类的实现细节
namespace {

// 如果类型是命名类型的子类型，则返回其限定名称，否则返回其类型字符串。
std::string getTypeStr(const c10::TypePtr& type) {
  switch (type->kind()) {
    case c10::TypeKind::FunctionType:
      return type->castRaw<c10::FunctionType>()->name()->qualifiedName();
    case c10::TypeKind::TupleType:
      return type->castRaw<c10::TupleType>()->name()->qualifiedName();
    case c10::TypeKind::ClassType:
      return type->castRaw<c10::ClassType>()->name()->qualifiedName();
    case c10::TypeKind::InterfaceType:
      return type->castRaw<c10::InterfaceType>()->name()->qualifiedName();
    default:
      return type->annotation_str();
  }
}

} // namespace

namespace torch {
namespace distributed {
namespace rpc {

// 原子类型，用于生成本地 ID
std::atomic<local_id_t> RRefContext::nextLocalId_{0};

//////////////////////////  RRefForkData  /////////////////////////////////

// RRefForkData 类的构造函数，初始化对象的各个成员
RRefForkData::RRefForkData(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    worker_id_t parent,
    std::string typeStr)
    : ownerId_(ownerId),
      rrefId_(rrefId),
      forkId_(forkId),
      parent_(parent),
      typeStr_(std::move(typeStr)) {}

//////////////////////////////  RRef  /////////////////////////////////////

// RRef 类的构造函数，初始化 RRef 实例的各个成员
RRef::RRef(worker_id_t ownerId, const RRefId& rrefId, TypePtr type)
    : RRefInterface(),
      ownerId_(ownerId),
      rrefId_(rrefId),
      type_(std::move(type)) {}

// 派生自 RRefInterface 类的 fork() 函数，返回 RRefForkData 对象
RRefForkData RRef::fork() const {
  auto& ctx = RRefContext::getInstance();
  return RRefForkData(
      ownerId_,
      rrefId_,
      ctx.genGloballyUniqueId(),
      ctx.getWorkerId(),
      getTypeStr(type_));
}

// 处理 RPC 错误的函数，根据不同的错误类型调用相应的处理函数
void RRef::handleError(RPCErrorType errorType, const JitFuture& jitFuture) {
  // 定义不同错误类型的处理函数映射表
  static std::unordered_map<
      RPCErrorType,
      std::function<void(const JitFuture& jitFuture)>,
      std::hash<int>>
      errorHandlers = {
          {RPCErrorType::TIMEOUT,
           [this](const JitFuture& /* unused */) { setTimedOut(); }},
          {RPCErrorType::INTENTIONAL_FAILURE,
           [this](const JitFuture& /* unused */) { setTimedOut(); }},
          {RPCErrorType::UNKNOWN_ERROR, [](const JitFuture& jitFuture) {
             // 默认的错误处理函数
             RRefContext::handleException(jitFuture);
           }}};
  // 根据错误类型查找并执行相应的处理函数
  errorHandlers.find(errorType)->second(jitFuture);
}

//////////////////////////  UserRRef  /////////////////////////////////////

// UserRRef 类的构造函数，初始化用户级别的远程引用对象
UserRRef::UserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    TypePtr type,
    std::shared_ptr<FutureMessage> fm)
    : RRef(ownerId, rrefId, std::move(type)),
      fm_(std::move(fm)),
      confirmedByOwner_(false) {
}
    // UserRRef 的构造函数，继承自 RRef 类，初始化 ownerId, rrefId 和类型信息
    const ForkId& forkId,
    TypePtr type)
    : RRef(ownerId, rrefId, std::move(type)),
      // 初始化 forkId_，表示当前 UserRRef 的 forkId
      forkId_(forkId),
      // 初始化 confirmedByOwner_，标记当前 UserRRef 是否已被所有者确认
      confirmedByOwner_(false) {
  // 什么也不做，
  // (1) 如果这个 UserRRef 是现有 RRef 的分支，RRefContext 将向所有者发送 RREF_FORK_REQUEST 消息。
  // (2) 如果这是创建者的 UserRRef，则 ScriptRemoteCall 或 PythonRemoteCall 将适当地通知所有者。
}

// 在析构函数中调用 tryDel() 方法尝试删除 UserRRef 对象
void UserRRef::tryDel() {
  // 使用互斥锁保护 deletedOnOwnerMutex_，确保线程安全
  std::lock_guard<std::mutex> lockGuard(deletedOnOwnerMutex_);
  // 如果未被 owner 删除，则执行删除操作
  if (!deletedOnOwner_) {
    try {
      // 调用 RRefContext 的实例删除指定的 UserRRef
      RRefContext::getInstance().delUser(ownerId_, rrefId_, forkId_);
      // 标记为已在 owner 端删除
      deletedOnOwner_ = true;
    } catch (const std::exception& ex) {
      // 捕获异常，记录错误日志，显示删除失败的具体信息
      LOG(ERROR) << "Error occurred when deleting" << *this << " : "
                 << ex.what();
    } catch (...) {
      // 捕获未知异常，记录错误日志，显示未知错误
      LOG(ERROR) << "Error occurred when deleting" << *this << " : "
                 << "unknown error";
    }
  }
}

// UserRRef 类的析构函数，调用 tryDel() 方法尝试删除对象
UserRRef::~UserRRef() {
  tryDel();
}

// 释放资源的方法，调用 tryDel() 方法尝试删除对象
void UserRRef::release_resources() {
  tryDel();
}

// 返回 forkId_ 成员变量的引用
const ForkId& UserRRef::forkId() const {
  return forkId_;
}

// 返回对象到当前节点的数据，包括超时检查和删除状态检查
IValue UserRRef::toHere(const float timeoutSeconds) const {
  // 检查是否已超时创建，如果是，则抛出异常
  TORCH_CHECK(
      !getTimedOut(),
      "RRef creation via rpc.remote() timed out, and it "
      "is possible that the RRef on the owner node does not exist.");
  // 检查是否在 owner 端已删除，如果是，则抛出异常
  TORCH_CHECK(
      !deletedOnOwner_,
      *this,
      " has been deleted. Cannot call to_here() on it after deletion.");
  
  // 构造调用的键名，用于性能分析
  auto toHereKey = std::string("");
  if (torch::autograd::profiler::profilerEnabled()) {
    toHereKey = fmt::format(
        "to_here#({})->({})",
        RpcAgent::getCurrentRpcAgent()->getWorkerInfo().name_,
        RpcAgent::getCurrentRpcAgent()->getWorkerInfo(ownerId_).name_);
  }
  // 记录性能范围，用于性能分析
  RECORD_USER_SCOPE(toHereKey);
  
  // 检查是否是 ScriptModule 类型的 RRef，如果是，则抛出异常
  TORCH_CHECK(
      !type_->is_module(),
      *this,
      " is an RRef to a ScriptModule. "
      "It can't be sent through RPC "
      "from owner, ",
      ownerWorkerInfo(),
      ", to user, ",
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo(),
      ".");

  // 获取当前的 RPC 代理
  auto agent = RpcAgent::getCurrentRpcAgent();

  // 如果是 Python 对象，则构造 PythonRRefFetchCall，并转换为消息对象
  c10::intrusive_ptr<Message> msgToSend;
  if (isPyObj()) {
    msgToSend = PythonRRefFetchCall(ownerId_, rrefId()).toMessage();
  } else {
    msgToSend = ScriptRRefFetchCall(ownerId_, rrefId()).toMessage();
  }


// 创建一个消息，用于请求远程对象的数据。ScriptRRefFetchCall 是一个类，
// 它生成一个请求消息，其中包含所有必要的信息（所有者ID和远程引用ID）。
msgToSend = ScriptRRefFetchCall(ownerId_, rrefId()).toMessage();



  // toHere is profiled as a blocking call, and does not execute operations on
  // the remote node. Hence, don't wrap it with a profiling message since we
  // don't need the profiler to be enabled remotely.
  auto jitFuture = autograd::sendMessageWithAutograd(
      *agent,
      agent->getWorkerInfo(ownerId_),
      std::move(msgToSend),
      true /* forceGradRecording */,
      timeoutSeconds,
      true /* forceDisableProfiling */);


// 发送带有自动微分信息的消息到远程节点，并且这个调用是阻塞的。
// 不需要在这个调用上启用远程的性能分析消息，因为这个调用本身不会在远程节点上执行操作。
auto jitFuture = autograd::sendMessageWithAutograd(
    *agent,
    agent->getWorkerInfo(ownerId_),
    std::move(msgToSend),
    true /* forceGradRecording */,
    timeoutSeconds,
    true /* forceDisableProfiling */);



  // TODO: we should ideally be able to interrupt this blocking wait if we check
  // getTimedOut() and it is true
  // (https://github.com/pytorch/pytorch/issues/39411).
  jitFuture->waitAndThrow();
  auto messagePtr = jitFuture->constValue().toCustomClass<Message>();
  MessageType msgType = messagePtr->type();
  auto response = deserializeResponse(*messagePtr, msgType);


// 等待消息发送的结果，并抛出任何异常。
jitFuture->waitAndThrow();
// 获取消息的指针，并将其转换为自定义类 Message 的实例。
auto messagePtr = jitFuture->constValue().toCustomClass<Message>();
// 获取消息的类型。
MessageType msgType = messagePtr->type();
// 反序列化响应消息，根据消息类型返回相应的响应。
auto response = deserializeResponse(*messagePtr, msgType);



  TORCH_INTERNAL_ASSERT(
      msgType == MessageType::SCRIPT_RREF_FETCH_RET ||
          msgType == MessageType::PYTHON_RREF_FETCH_RET,
      "Message type should either be SCRIPT_RREF_FETCH_RET "
      "or PYTHON_RREF_FETCH_RET");
  RpcCommandBase& rpc = *response;
  auto& rrefFetchRet = static_cast<RRefFetchRet&>(rpc);


// 断言消息类型应该是 SCRIPT_RREF_FETCH_RET 或 PYTHON_RREF_FETCH_RET，
// 否则会触发内部错误断言。
TORCH_INTERNAL_ASSERT(
    msgType == MessageType::SCRIPT_RREF_FETCH_RET ||
        msgType == MessageType::PYTHON_RREF_FETCH_RET,
    "Message type should either be SCRIPT_RREF_FETCH_RET "
    "or PYTHON_RREF_FETCH_RET");
// 将响应消息转换为 RpcCommandBase 的引用。
RpcCommandBase& rpc = *response;
// 将 RpcCommandBase 转换为 RRefFetchRet 的引用，以便进一步操作。
auto& rrefFetchRet = static_cast<RRefFetchRet&>(rpc);



  if (isPyObj()) {
    // wrap python serialized vector of ivalues into tuple, this
    // made the C++ toHere interface to return single IValue
    return ivalue::Tuple::create(rrefFetchRet.values());
  } else {
    return rrefFetchRet.values().front();
  }


// 如果是 Python 对象，则将 Python 序列化的 IValues 向量包装成元组，
// 以便 C++ 的 toHere 接口返回单个 IValue。
if (isPyObj()) {
  return ivalue::Tuple::create(rrefFetchRet.values());
} else {
  // 否则直接返回第一个 IValue。
  return rrefFetchRet.values().front();
}
// Note [Best-Effort Check on Deleted UserRRefs]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 这个检查不能保证正确性，因为可能有另一个线程同时试图删除这个 UserRRef。
// 通过这个检查并不意味着在整个函数执行期间这个 RRef 会保持活动状态。
// 这只是我们尽力提供正确错误消息的尝试。使用已删除的 UserRRef 的行为是未定义的。
//
// 不实现严格的检查的原因是：
// 1. 这将需要在 deletedOnOwnerMutex_ 上获取锁，对于大多数正常用例来说会引入不必要的开销。
// 2. 这将引入许多复杂性以确保行为正确。假设我们在这里获取了锁，并且有另一个线程 X 在 tryDel() 上等待锁。
//    退出这个 fork 函数将解除对线程 X 的阻塞。然而，当 X 继续删除这个 UserRRef 时，
//    fork() 的调用点可能已将 UserRRef 添加到 pendingChildren_ 映射中，但到目前为止，没有什么阻止 X
//    删除这个 RRef，即使由于 pendingChildren_ 的状态变化，它不应该这样做。我们可能可以通过在 X 中锁定并检查
//    pendingChildren_ 来正确处理它，但这种复杂性似乎不值得收益。
TORCH_CHECK(
    !deletedOnOwner_,
    *this,
    " has been deleted. Cannot call fork an UserRRef after deletion.");
// 返回通过调用 RRef::fork() 获得的新的 RRef 实例
return RRef::fork();
}



//////////////////////////  OwnerRRef  /////////////////////////////////////

// OwnerRRef 构造函数，初始化一个空的 future_ 对象
OwnerRRef::OwnerRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    TypePtr type,
    std::vector<c10::Device> devices)
    : OwnerRRef(ownerId, rrefId, type, /* value */ {}, std::move(devices)) {}

// OwnerRRef 构造函数，初始化 future_ 对象，并设置值（如果提供了）
OwnerRRef::OwnerRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    TypePtr type,
    std::optional<IValue> value,
    std::vector<c10::Device> devices)
    : RRef(ownerId, rrefId, type) {
  // 创建一个 JitFuture 实例，用于存储类型和设备信息
  future_ = c10::make_intrusive<JitFuture>(type_, std::move(devices));

  // 如果提供了值，则将值标记为已完成
  if (value.has_value()) {
    future_->markCompleted(value.value());
  }
}

// 获取 OwnerRRef 的值，如果超时则抛出错误
const IValue& OwnerRRef::getValue() const {
  TORCH_CHECK(
      !getTimedOut(),
      "RRef creation via rpc.remote() timed out, and it "
      "is possible that the RRef on the owner node does not exist.");
  // 等待 future_ 完成，并抛出异常（如果有）
  future_->waitAndThrow();
  // 返回 future_ 中存储的常量值
  return future_->constValue();
}

// 检查 OwnerRRef 是否有值
bool OwnerRRef::hasValue() const {
  return future_->completed();
}

// 获取 future_ 对象的指针
c10::intrusive_ptr<JitFuture> OwnerRRef::getFuture() {
  return future_;
}

// 设置 OwnerRRef 的值
void OwnerRRef::setValue(IValue&& value) {
  future_->markCompleted(value);
}

// 设置 OwnerRRef 的错误状态
void OwnerRRef::setError(std::exception_ptr eptr) {
  future_->setErrorIfNeeded(std::move(eptr));
}

// 重载 << 运算符，用于打印 OwnerRRef 对象
std::ostream& operator<<(std::ostream& os, const RRef& rref) {
  if (rref.isOwner()) {
    return os << "OwnerRRef("
              << "rref_id=" << rref.rrefId() << ")";
  } else {
    return os << "UserRRef("  // 返回输出流并开始构造字符串 "UserRRef("
              << "rref_id=" << rref.rrefId()  // 添加 rref 对象的 rrefId() 方法返回值到字符串中作为键值对
              << ", fork_id=" << static_cast<const UserRRef*>(&rref)->forkId()  // 添加 rref 对象转换为 UserRRef 类型后调用 forkId() 方法返回值到字符串中作为键值对
              << ")";  // 结束构造字符串并返回输出流
  }
}

// 结束 "rpc" 命名空间
} // namespace rpc

// 结束 "distributed" 命名空间
} // namespace distributed

// 结束 "torch" 命名空间
} // namespace torch
```