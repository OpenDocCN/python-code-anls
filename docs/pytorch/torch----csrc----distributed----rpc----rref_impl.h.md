# `.\pytorch\torch\csrc\distributed\rpc\rref_impl.h`

```
#pragma once

#include <ATen/core/jit_type.h>
#include <ATen/core/rref_interface.h>
#include <c10/core/Event.h>
#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/types.h>

#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {

// Represents a remote reference (RRef) fork data structure to be sent over the wire.
struct TORCH_API RRefForkData {
  const worker_id_t ownerId_;     // ID of the owner of the RRef
  const RRefId rrefId_;           // Globally unique ID of the RRef
  const ForkId forkId_;           // Globally unique ID of the fork of RRef
  const worker_id_t parent_;      // ID of the parent worker
  const std::string typeStr_;     // Type information as a string

  // Constructor to initialize RRefForkData
  RRefForkData(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId,
      worker_id_t parent,
      std::string typeStr);
};

// Note [RRef Protocol]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// [Background]
//
// RRef stands for Remote REFerence. Each RRef is owned by a single worker
// (i.e., owner) and can be used by multiple users. The owner stores the real
// data referenced by its RRefs. RRef needs to support fast and scalable RPC.
// Hence, in the design, we avoid using a single global master to keep RRef
// states, instead owners will keep track of the global reference counts
// for its RRefs. Every RRef can be uniquely identified by a global RRefId,
// which is assigned at the time it is first created either on a user or on the
// owner.
//
// On the owner worker, there is only one OwnerRRef instance, which contains the
// real data, while on user workers, there can be as many UserRRefs as
// necessary, and UserRRef does not hold the data. All usage on the OwnerRRef
// should retrieve the unique OwnerRRef instance using the globally unique
// RRefId. //A UserRRef will be created when it is used as an argument or return
// value in dist.rpc or dist.remote call, but RRef forking and reference
// counting (RC) are completely transparent to applications. Every UserRRef will
// also have its globally unique ForkId.
//
// [Assumptions]
//
// 1. Transient Network Failures
//
// TODO: current RRef implementation does not tolerate failures
//
// The RRef design handles transient network failures by retrying
// messages. Node crashes or permanent network partition is beyond the scope.
// When those incidents occur, the application may take down all workers, revert

// NB: if more fields are added, make sure this field is also bumped
constexpr int RFD_TUPLE_SIZE = 7; // number of RRefForkData fields in py::tuple

// Indices for different fields in the tuple representing RRefForkData
constexpr int OWNER_IDX = 0;      // index of ownerId in the tuple
constexpr int RREFID_ON_IDX = 1;  // index of RRefId.createdOn_ in the tuple
constexpr int RREFID_ID_IDX = 2;  // index of RRefId.localId_ in the tuple
constexpr int FORKID_ON_IDX = 3;  // index of ForkId.createdOn_ in the tuple
constexpr int FORKID_ID_IDX = 4;  // index of ForkId.localId_ in the tuple
constexpr int PARENT_IDX = 5;     // index of parent in the tuple
constexpr int TYPE_IDX = 6;       // index of parent in the tuple

} // namespace rpc
} // namespace distributed
} // namespace torch
# 到前一个检查点，然后恢复训练。
//
// 2. Non-idempotent UDFs
//
// 我们假设用户定义函数（UDF）不是幂等的，因此不能重试。但是，
// 内部的 RRef 控制消息是幂等的，会在消息失败时重试。
//
// TODO: RRef 内部消息目前还不是幂等的
//
// 3. Out of Order Message Delivery
//
// 我们不假设任何节点对之间的消息传递顺序，因为发送方和接收方都使用多个线程。
// 不能保证哪个消息会首先被处理。
//
// [RRef Lifetime]
//
// 协议的目标是在合适的时机删除 OwnerRRef。
// 删除 OwnerRRef 的正确时机是当没有存活的 UserRRefs 时，
// 并且 Python 垃圾回收（GC）也同意在所有者上删除 OwnerRRef 实例。这
// 的棘手之处在于确定是否有任何存活的 UserRRefs。
//
// 用户可以在三种情况下获取 UserRRef：
//
// (1). 从所有者那里接收一个 UserRRef。
// (2). 从另一个用户那里接收一个 UserRRef。
// (3). 创建一个由另一个工作节点拥有的新 UserRRef。
//
// (1) 是最简单的情况，其中所有者启动分叉，因此它可以轻松增加本地的引用计数。
// 唯一的要求是任何 UserRRef 在销毁之前必须通知所有者。因此，我们需要第一个保证：
//
// G1. 当删除任何 UserRRef 时，所有者将会收到通知。
//
// 由于消息可能延迟到达或乱序，我们需要更多的保证来确保删除消息不会太早发送出去。
// 让我们首先引入一个新的概念。如果 A 发送一个涉及 RRef 的 RPC 给 B，我们称 A 上的
// RRef 为父 RRef，B 上的 RRef 为子 RRef。
//
// G2. 父 RRef 在确认子 RRef 被所有者确认之前不能被删除。
//
// 在情况 (1) 下，调用方是 UserRRef，被调用方是 OwnerRRef，这意味着用户只有在所有
// 先前的消息被确认之后才会发送删除消息。请注意，确认并不意味着所有者完成函数的执行，
// 而是仅表示所有者已经获取了其本地的 OwnerRRef 并准备将其传递给函数，这足以使 OwnerRRef
// 保持存活状态，即使用户的删除消息在函数完成执行之前到达所有者。
//
// 对于情况 (2) 和 (3)，可能所有者只部分知道 RRef 的分支图，或者根本不知道。例如，RRef
// 可能在一个用户上被构建，在所有者接收到 RPC 调用之前，创建者用户可能已经与其他用户分享了
// RRef，并且这些用户可能进一步分享了 RRef。一个不变条件是任何 RRef 的分支图始终是以所有者为根
// 的树形结构，因为分叉一个 RRef 总是创建一个新的 RRef 实例，因此每个 RRef 都有一个单一的父节点。
// 一个麻烦的细节是当一个 RRef 在用户上创建时，技术上来说所有者
// ``RRef`` is the base type for both ``UserRRef`` and ``OwnerRRef``.
// Each ``RRef`` has a globally unique ``RRefId``.
class TORCH_API RRef : public RRefInterface {
 public:
  // RRef is made NOT copyable NOT movable to prevent messing up reference
  // counting.
  // 显式删除拷贝构造函数和移动构造函数，确保不可复制、不可移动，以防止引用计数混乱
  explicit RRef(const RRef& other) = delete;
  explicit RRef(RRef&& other) = delete;
  RRef& operator=(RRef&& other) = delete;

  ~RRef() override = default;  // 使用默认析构函数

  // returns the worker id of the owner
  // 返回持有者的 worker id
  inline worker_id_t owner() const override {
    return ownerId_;
  }

  // returns the worker name of the owner
  // 返回持有者的 worker 名称
  inline std::string ownerName() const override {
  // 返回当前 RPC 代理的工作信息中的所有者名称
  return RpcAgent::getCurrentRpcAgent()->getWorkerInfo(ownerId_).name_;
}

// 返回所有者的工作信息
inline WorkerInfo ownerWorkerInfo() const {
  return RpcAgent::getCurrentRpcAgent()->getWorkerInfo(ownerId_);
}

// 返回此 RRef 的全局唯一 RRefId
inline const RRefId& rrefId() const {
  return rrefId_;
}

// 检查此 RRef 是否是 Python 对象类型
inline bool isPyObj() const {
  return type_ == PyObjectType::get();
}

// 返回 RRef 的类型指针
inline const TypePtr type() const override {
  return type_;
}

// 保存创建此 RRef 在远程节点上的 future，
// 仅在处理通过 rpc.remote 调用的请求时设置。
// 用于在性能分析用例中获取对应于 rref 的 future。
inline void registerOwnerCreationFuture(c10::intrusive_ptr<JitFuture> fut) {
  ownerCreationFuture_ = std::move(fut);
}

// 获取创建此 RRef 的 future
inline c10::intrusive_ptr<JitFuture> getOwnerCreationFuture() const {
  return ownerCreationFuture_;
}

// 检查此 RRef 在所有者节点上的创建是否超时
inline bool getTimedOut() const {
  return timedOut_.load();
}

// 根据 RPCErrorType 将错误分派给正确的处理程序
void handleError(RPCErrorType errorType, const JitFuture& JitFuture);

// 向所有者发送删除 UserRRef 请求，
// 如果请求尚未发送。
// 有两种情况会调用它：
// 1. Python GC 决定结束 UserRRef 生命周期，调用析构函数。
// 2. RPC 模块在 RRefContext 中跟踪的所有 UserRRef 上调用，进行优雅关闭。
virtual void tryDel() {}

protected:
// 标记此 RRef 在所有者节点上的创建是否已超时
inline void setTimedOut() {
  timedOut_ = true;
}
friend class RRefContext;

RRef(worker_id_t ownerId, const RRefId& rrefId, TypePtr type);

virtual RRefForkData fork() const;

// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
const worker_id_t ownerId_;
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
const RRefId rrefId_;
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
std::atomic<bool> timedOut_{false};

// 类型字段，用于表示 RRef 所持有的元素类型，
// 可以是 JIT 支持的任何 TypePtr，包括 PyObjectType
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
const TypePtr type_;
// 对应于在远程节点上请求创建 RRef 的 future
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
c10::intrusive_ptr<JitFuture> ownerCreationFuture_;
};

// ``UserRRef`` represents a user of an RRef. Besides the ``RRefId``, each user
// also has a globally unique ``ForkId`` to identify this user. ``UserRRef``
// never owns the real value, the only way to get the value of the ``RRef`` is
// to call ``to_here()`` and get a copy..
class TORCH_API UserRRef final : public RRef {
 public:
  // 禁止复制构造函数和移动构造函数，确保每个UserRRef实例是唯一的
  UserRRef(const UserRRef& other) = delete;
  UserRRef(UserRRef&& other) = delete;
  UserRRef& operator=(const UserRRef& other) = delete;
  UserRRef& operator=(UserRRef&& other) = delete;

  // 构造函数，初始化UserRRef实例
  UserRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      const ForkId& forkId,
      TypePtr type);

  // 返回是否是拥有者的标志，覆盖基类的方法
  inline bool isOwner() const override {
    return false;
  }

  // 返回是否已被拥有者确认的标志，覆盖基类的方法
  inline bool confirmedByOwner() const override {
    return confirmedByOwner_;
  }

  // 返回此RRef的全局唯一ForkId
  const ForkId& forkId() const;

  // 从OwnerRRef获取值的副本。如果值尚未准备好，此调用将阻塞。
  IValue toHere(
      const float timeoutSeconds =
          torch::distributed::rpc::kUnsetRpcTimeout) const;

  // 尝试删除此UserRRef
  void tryDel() override;

  // 当引用计数为0时调用，告知拥有者取消引用
  void release_resources() override;

  // 当引用计数和弱引用计数都为0时调用。详见链接中的文档。
  // 在销毁包装的intrusive_ptr_target实例及其数据成员时调用。
  ~UserRRef() override;

 private:
  friend class RRefContext;

  // 返回当前Fork的数据
  RRefForkData fork() const override;

  // 确认此UserRRef已被其拥有者确认
  inline void confirm() {
    confirmedByOwner_ = true;
  }

  const ForkId forkId_;

  // 指示此用户是否已向其拥有者发送了删除消息的标志
  // 注意，需要线程安全，因为删除消息可以由Python垃圾收集器调用的析构函数发送，
  // 或者由RPC优雅关闭时RRefContext的主动清理发送。
  std::mutex deletedOnOwnerMutex_;
  bool deletedOnOwner_{false};
  // 指示此UserRRef是否已被其拥有者确认的标志
  std::atomic<bool> confirmedByOwner_;
};

// 仅在派生类上保留模板，因为``RRefContext``需要擦除``RRef``上的类型并将它们保存在一个映射中。
class TORCH_API OwnerRRef final : public RRef {
 public:
  // 禁止复制构造函数和移动构造函数，确保每个OwnerRRef实例是唯一的
  OwnerRRef(const OwnerRRef& other) = delete;
  OwnerRRef(OwnerRRef&& other) = delete;
  OwnerRRef& operator=(const OwnerRRef& other) = delete;
  OwnerRRef& operator=(OwnerRRef&& other) = delete;

  // 构造函数，初始化OwnerRRef实例
  OwnerRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      TypePtr type,
      std::vector<c10::Device> devices);

  // 构造函数，初始化OwnerRRef实例，同时提供可选的值
  OwnerRRef(
      worker_id_t ownerId,
      const RRefId& rrefId,
      TypePtr type,
      std::optional<IValue> value,
      std::vector<c10::Device> devices);

  // 返回是否是拥有者的标志，覆盖基类的方法
  inline bool isOwner() const override {
    // 返回 true
      return true;
    }
    
    // OwnerRRef 总是被确认，而 UserRRef 只有在所有者知道它时才被确认。
    inline bool confirmedByOwner() const override {
      return true;
    }
    
    // 获取真实值的常量引用。如果值尚未准备好，则此方法将阻塞。由于它不创建新的 py::object，因此此方法不需要 GIL。如果发生错误，它将抛出异常。
    const IValue& getValue() const;
    
    // 设置此 OwnerRRef 的值。由于它不创建新的 py::object，因此此方法不需要 GIL。
    void setValue(IValue&& value);
    
    // 将此 OwnerRRef 的值设置为包含异常。
    void setError(std::exception_ptr eptr);
    
    // 是否已经设置了值或错误？
    bool hasValue() const;
    
    // 获取一个 Future，在值或错误设置时得到满足。
    c10::intrusive_ptr<JitFuture> getFuture();
    
    private:
    friend class RRefContext;
    
    c10::intrusive_ptr<JitFuture> future_;
};

// 定义输出流操作符重载，用于打印 RRef 对象的信息
TORCH_API std::ostream& operator<<(std::ostream& os, const RRef& rref);

// 从 c10::RRefInterface 转换为 OwnerRRef 的辅助函数
inline TORCH_API c10::intrusive_ptr<OwnerRRef> fromRRefInterface(
    const c10::intrusive_ptr<c10::RRefInterface>& rrefInterface) {
  // 使用 static_intrusive_pointer_cast 将 RRefInterface 转换为 OwnerRRef
  return c10::static_intrusive_pointer_cast<OwnerRRef>(rrefInterface);
}

// 从 OwnerRRef 转换为 c10::RRefInterface 的辅助函数
inline TORCH_API c10::intrusive_ptr<c10::RRefInterface> fromOwnerRRef(
    const c10::intrusive_ptr<RRef>& ownerRRef) {
  // 使用 static_intrusive_pointer_cast 将 OwnerRRef 转换为 c10::RRefInterface
  return c10::static_intrusive_pointer_cast<c10::RRefInterface>(ownerRRef);
}

// 结束 rpc 命名空间
} // namespace rpc

// 结束 distributed 命名空间
} // namespace distributed

// 结束 torch 命名空间
} // namespace torch
```