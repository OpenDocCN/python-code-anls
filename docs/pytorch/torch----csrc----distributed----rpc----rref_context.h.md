# `.\pytorch\torch\csrc\distributed\rpc\rref_context.h`

```py
#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {

namespace callback {
// RemoteCall 的回调函数。
void TORCH_API
confirmPendingUser(const JitFuture& jitFuture, const ForkId& expectedForkId);

// 完成创建 owner rref 的回调函数，返回 deletedRRef，以便在 python_functions.cpp 中处理含有 python 对象的 deletedRRef。
c10::intrusive_ptr<RRef> TORCH_API
finishCreatingOwnerRRef(const JitFuture& jitFuture, const RRefId& rrefId);
} // namespace callback

// 管理 RRef 生命周期并跟踪 RRef 分叉。
class TORCH_API RRefContext {
 public:
  static RRefContext& getInstance();
  // 注意：在析构 RRefContext 单例之前必须调用此方法。
  // 与 delForkOfOwner 类似，此方法返回一个持有 py::object 的 OwnerRRefs 向量。
  // 调用方还需负责在 GIL 下重置这些 shared_ptr 对象。有关详细信息，请参阅 delForkOfOwner() 的注释。
  static std::vector<c10::intrusive_ptr<RRef>> destroyInstance(
      bool ignoreRRefLeak = true);

  static void handleException(const JitFuture& jitFuture);

  // 在不再抛出 ::c10::Error 的情况下处理异常。
  static void handleExceptionSilent(const JitFuture& jitFuture);

  RRefContext(const RRefContext&) = delete;
  RRefContext(RRefContext&& other) = delete;
  void operator=(const RRefContext&) = delete;
  RRefContext& operator=(RRefContext&& other) = delete;

  ~RRefContext();

  // 获取当前 worker 的 worker id
  inline worker_id_t getWorkerId() const {
    return agent_->getWorkerInfo().id_;
  }

  // 获取当前 worker 的 worker name
  inline const std::string& getWorkerName() const {
    return agent_->getWorkerInfo().name_;
  }

  // 生成全局唯一 ID
  inline GloballyUniqueId genGloballyUniqueId() {
    return GloballyUniqueId(getWorkerId(), nextLocalId_++);
  }

  inline const std::shared_ptr<RpcAgent>& agent() const {
    PendingUserState(c10::intrusive_ptr<RRef> rref)
        : rref_(std::move(rref)),
          confirmationFuture_(c10::make_intrusive<JitFuture>(BoolType::get())) {
    }

    inline void confirm() {
      c10::static_intrusive_pointer_cast<UserRRef>(rref_)->confirm();
      confirmationFuture_->markCompleted();
    }

    c10::intrusive_ptr<RRef> rref_;
    // 使用 Future.wait() 和 Future.markCompleted() 阻塞和解除用户函数。
    // future_ 包装的 bool 值未使用。
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```