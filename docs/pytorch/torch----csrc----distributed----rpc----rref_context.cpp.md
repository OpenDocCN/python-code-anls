# `.\pytorch\torch\csrc\distributed\rpc\rref_context.cpp`

```py
// 在torch命名空间中的分布式RPC功能中定义了一些与RRef相关的回调和状态管理。

// 定义一个线程局部变量，用于存储用户状态的表，每个状态是RRefContext::PendingUserState的共享指针的向量。
thread_local std::vector<std::shared_ptr<RRefContext::PendingUserState>> RRefContext::userTable_;

// 定义一个线程局部变量，用于表示是否正在记录RPC调用。
thread_local bool RRefContext::recording_ = false;

// 在命名空间rpc::callback中定义了一些用于处理异步回调的函数。

// 确认挂起的用户状态，当jitFuture完成时调用，检查forkId以确保匹配。
void confirmPendingUser(
    const JitFuture& jitFuture,
    const ForkId& expectedForkId) {
  if (!jitFuture.hasError()) {
    // 如果没有错误，从jitFuture获取消息指针，反序列化响应消息，并检查forkId是否符合预期。
    auto msgPtr = jitFuture.constValue().toCustomClass<Message>();
    auto msgType = msgPtr->type();
    auto rpc = deserializeResponse(*msgPtr, msgType);
    auto& rr = dynamic_cast<RemoteRet&>(*rpc);
    TORCH_INTERNAL_ASSERT(rr.forkId() == expectedForkId);
  } else {
    // 处理错误，例如超时，通过调用RRef的错误处理程序来处理。
    // 详见[Best Effort Error handling for Remote calls]注释说明。
    auto rref_ptr = RRefContext::getInstance().getPendingUser(expectedForkId);
    auto errorType = getRPCErrorType(jitFuture);
    rref_ptr->handleError(errorType, jitFuture);
  }
  // 删除挂起的用户状态。
  RRefContext::getInstance().delPendingUser(expectedForkId);
}

// 完成创建所有者RRef的回调，当jitFuture完成时调用，处理OwnerRRef创建过程中的错误。
c10::intrusive_ptr<RRef> finishCreatingOwnerRRef(
    const JitFuture& jitFuture,
    const RRefId& rrefId) {
  if (jitFuture.hasError()) {
    auto& ctx = RRefContext::getInstance();
    // 如果有错误，获取OwnerRRef的指针，并处理错误。
    auto rref_ptr =
        fromRRefInterface(ctx.getOwnerRRef(rrefId, /* foreCreated */ true)
                              ->constValue()
                              .toRRef());
    auto errorType = getRPCErrorType(jitFuture);
    rref_ptr->handleError(errorType, jitFuture);
    // 删除OwnerRRef的forkId。
    auto deletedRRef =
        ctx.delForkOfOwner(rref_ptr->rrefId(), rref_ptr->rrefId());
    return deletedRRef;
  } else {
    // 如果没有错误，从jitFuture获取消息指针，反序列化响应消息，并检查RRefId和forkId是否匹配。
    auto msgPtr = jitFuture.constValue().toCustomClass<Message>();
    auto msgType = msgPtr->type();
    auto rpc = deserializeResponse(*msgPtr, msgType);
    auto& rr = dynamic_cast<RemoteRet&>(*rpc);
    TORCH_INTERNAL_ASSERT(
        rr.rrefId() == rr.forkId(),
        "Expecting an OwnerRRef as RemoteRet but got a fork.");
    # 获取 RRefContext 的单例实例，并赋值给 ctx 变量
    auto& ctx = RRefContext::getInstance();
    # 调用 RRefContext 实例的 delForkOfOwner 方法，删除指定 rrefId 对应的所有 forked RRef，并将结果保存在 deletedRRef 变量中
    auto deletedRRef = ctx.delForkOfOwner(rr.rrefId(), rr.rrefId());
    # 返回被删除的 RRef 对象
    return deletedRRef;
} // namespace callback

// Keys for RRef-related debug information.
const std::string kNumOwnerRRefs = "num_owner_rrefs"; // 定义所有者 RRef 的数量的键
const std::string kNumPendingFutures = "num_pending_futures"; // 定义待处理 Futures 的数量的键
const std::string kNumPendingUsers = "num_pending_users"; // 定义待处理用户数的键
const std::string kNumForks = "num_forks"; // 定义分支数的键

RRefContext& RRefContext::getInstance() {
  // Leaky singleton to avoid module destructor races.
  // 返回 RRefContext 的单例实例，确保在模块析构竞争时不会出现问题
  static RRefContext* context = new RRefContext(RpcAgent::getCurrentRpcAgent());
  return *context;
}

std::vector<c10::intrusive_ptr<RRef>> RRefContext::destroyInstance(
    bool ignoreRRefLeak) {
  auto& ctx = RRefContext::getInstance();
  {
    std::lock_guard<std::mutex> lock(ctx.destroyedMutex_);
    ctx.destroyed_ = true;
  }
  ctx.checkRRefLeaks(ignoreRRefLeak); // 检查 RRef 泄漏情况
  std::vector<c10::intrusive_ptr<RRef>> deletedRRefs;
  for (auto& entry : ctx.owners_) {
    auto rref = entry.second;
    if (rref->isPyObj()) {
      deletedRRefs.emplace_back(std::move(rref));
    }
  }
  ctx.owners_.clear(); // 清空所有者 RRef 的容器
  ctx.pendingOwners_.clear(); // 清空待处理所有者的容器
  return deletedRRefs; // 返回被删除的 RRef
}

void RRefContext::handleException(const JitFuture& jitFuture) {
  if (jitFuture.hasError()) {
    auto errMsg = jitFuture.tryRetrieveErrorMessage();
    VLOG(1) << "Got exception: " << errMsg;
    TORCH_CHECK(false, errMsg); // 抛出异常并打印错误消息
  }
}

void RRefContext::handleExceptionSilent(const JitFuture& jitFuture) {
  if (jitFuture.hasError()) {
    auto errMsg = jitFuture.tryRetrieveErrorMessage();
    VLOG(1) << "Got exception: " << errMsg;
    TORCH_CHECK_MSG(false, errMsg); // 抛出异常并打印详细错误消息
  }
}

RRefContext::RRefContext(std::shared_ptr<RpcAgent> agent)
    : agent_(std::move(agent)) {} // RRefContext 构造函数，接受一个 RpcAgent 的 shared_ptr

RRefContext::~RRefContext() {
  if (!owners_.empty()) {
    VLOG(1) << "Destructing RRefContext with non-empty OwnerRRef set. "
            << "This would likely cause Python deref error. "
            << "Make sure destroyInstance() is invoked before destruction.";
  }
}

std::unordered_map<std::string, std::string> RRefContext::getDebugInfo() {
  std::unordered_map<std::string, std::string> info; // 创建一个用于调试信息的无序映射
  std::unique_lock<std::mutex> lock(mutex_);
  auto ownerSize = owners_.size(); // 获取所有者 RRef 的数量
  auto numPendingUsers = pendingUsers_.size(); // 获取待处理用户的数量
  int numForks = 0;
  for (const auto& owner : forks_) {
    numForks += owner.second.size(); // 计算分支的总数
  }
  lock.unlock();
  info[kNumOwnerRRefs] = std::to_string(ownerSize); // 存储所有者 RRef 的数量
  info[kNumPendingFutures] = std::to_string(numPendingFutures_.load()); // 存储待处理 Futures 的数量
  info[kNumPendingUsers] = std::to_string(numPendingUsers); // 存储待处理用户的数量
  info[kNumForks] = std::to_string(numForks); // 存储分支数
  return info; // 返回调试信息
}

void RRefContext::checkRRefLeaks(bool ignoreRRefLeak) {
  if (!forks_.empty()) { // 如果存在泄漏的 RRef
    std::stringstream ss;
    for (auto& entry : forks_) {
      const RRefId& rrefId = entry.first;
      for (const auto& forkId : entry.second) {
        ss << "Leaking RRef " << rrefId << " with fork Id " << forkId
           << std::endl; // 打印泄漏的 RRef 和分支 Id
      }
    }
    LOG(WARNING)
        << "Detected RRef Leaks during shutdown. This usually "
        << "occurs when the application code still holds references to RRef "
        << "instances when calling shutdown(). If the program has "
        << "completed correctly and the process is exiting, it is OK to "
        << "ignore these leaks. However, if you program will keep running "
        << "after this, these leaks could result in memory leaks on RRef "
        << "owners. Please make sure all RRefs are out of scope and Python "
        << "GC has deleted them before calling shutdown(): \n"
        << ss.str();
    // 记录警告日志，指出在关闭过程中检测到 RRef 泄漏的情况
    if (!ignoreRRefLeak) {
      // 如果不忽略 RRef 泄漏，则触发 TORCH_CHECK 断言
      TORCH_CHECK(false, ss.str());
    }
  }
}

c10::intrusive_ptr<UserRRef> RRefContext::createUserRRef(
    worker_id_t ownerId,
    const TypePtr& type) {
  // 检查 ownerId 不等于当前 worker 的 ID，确保不在 owner 上创建 UserRRef
  TORCH_CHECK(ownerId != getWorkerId(), "Cannot create UserRRef on owner.");
  // 先生成全局唯一的 rrefId，再生成全局唯一的 forkId，以确保顺序确定性
  const auto rrefId = genGloballyUniqueId();
  const auto forkId = genGloballyUniqueId();
  // 调用另一个重载的方法创建 UserRRef 对象并返回
  return createUserRRef(ownerId, rrefId, forkId, type);
}

c10::intrusive_ptr<UserRRef> RRefContext::createUserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    const TypePtr& type) {
  // 检查 ownerId 不等于当前 worker 的 ID，确保 owner 不能创建自己的 UserRRef
  TORCH_CHECK(ownerId != getWorkerId(), "RRef owner cannot create user RRef.");
  // RRefContext 不追踪用户创建的 UserRRef，只有当没有 shared_ptrs 指向它时才会销毁
  //
  // 注意：这里不能使用 make_shared，因为 UserRRef 的构造函数是私有的。
  // 注意：这个 UserRRef 还未被 owner 确认。调用此函数的地方需要负责将此 UserRRef 添加到 pendingUsers_ 中。
  // 当前有两个调用地点：
  // (1) python_functions.cpp 中的创建用户
  // (2) RRefContext::notifyOwnerAndParentOfFork 中的被调用用户
  //
  // 之所以不在此处添加 pending 用户，是为了将 addPendingUser() 放在 RPC 发生的地方，
  // 并且在调用处的响应回调中与 deletePendingUser() 进行清理，这样更加清晰。
  return c10::make_intrusive<UserRRef>(ownerId, rrefId, forkId, type);
}

void RRefContext::delUser(
    const worker_id_t owner,
    const RRefId& rrefId,
    const ForkId& forkId) {
  {
    std::lock_guard<std::mutex> lock(destroyedMutex_);
    if (!destroyed_) {
      // 发送 RRefUserDelete 消息使接收者运行 delForkOfOwner，现在是幂等的。
      // 更多细节请参阅 RRefContext::delForkOfOwner 处的注释。
      ++numPendingFutures_;
      auto jitFuture = agent_->sendWithRetries(
          agent_->getWorkerInfo(owner),
          RRefUserDelete(rrefId, forkId).toMessage());

      jitFuture->addCallback([this](JitFuture& future) {
        handleExceptionSilent(future);
        --numPendingFutures_;
      });
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  // 从 confirmedUsers_ 中擦除指定的 forkId
  confirmedUsers_.erase(forkId);
}

void RRefContext::delAllUsersAndUnforkedOwners(
    std::chrono::milliseconds timeoutMillis) {
  // 首先等待所有待确认的 UserRRefs，
  // 一种是 pendingUsers_，从 owner 共享的，
  // 另一种是 pendingChildren_，从另一个 User 共享的。
  std::unordered_map<ForkId, c10::weak_intrusive_ptr<RRef>, ForkId::Hash>
      tempConfirmedUsers;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    // 等待直到 pendingUsers_ 和 pendingChildren_ 都为空
    bool noPending = deleteAllUsersCV_.wait_for(lock, timeoutMillis, [this]() {
      return pendingUsers_.empty() && pendingChildren_.empty();
    });
    // 如果存在未处理的待确认项，则记录错误日志
    if (!noPending) {
      LOG(ERROR)
          << "Timed out waiting for pending UserRRefs to be confirmed by owner and parent.";
    }
    // 交换临时确认用户列表与当前确认用户列表
    tempConfirmedUsers.swap(confirmedUsers_);
  }

  // 开始发送 UserRRef 删除消息，在所有待确认的项都被确认之后
  // 注意，在此期间不应有新的分叉发生，因为假定此实用程序在优雅关闭期间调用，
  // 不会再初始化新的用户 RPC 调用。
  for (const auto& user : tempConfirmedUsers) {
    // 获取 UserRRef 的弱引用指针
    c10::intrusive_ptr<RRef> rref_ptr = user.second.lock();
    // 如果弱引用指针为空，则继续下一个循环
    if (!rref_ptr) {
      continue;
    }
    // 调用 tryDel() 方法删除 UserRRef，此处必须释放锁定
    rref_ptr->tryDel();
  }

  // 如果 owners_ 映射中的某个 RRef 从未被分叉，我们将永远不会收到来自分叉节点的删除消息，
  // 因此在此处删除 RRef。当远程调用发送到自身并超时时可能会出现此情况。
  {
    // 获取互斥锁
    std::unique_lock<std::mutex> lock(mutex_);
    // 存储未分叉的 owner 的列表
    std::vector<RRefId> unforkedOwners;
    // 遍历 owners_ 映射
    for (const auto& it : owners_) {
      auto rrefId = it.first;
      // 如果在 forks_ 集合中找不到相应的 rrefId，则表示 owner 的成功分叉从未被处理
      if (forks_.find(rrefId) == forks_.end()) {
        // 将未处理的 owner 添加到列表中
        unforkedOwners.push_back(rrefId);
      }
    }
    // 遍历未分叉的 owner 列表，逐个删除
    for (auto& rrefId : unforkedOwners) {
      LOG(INFO) << "Removing unforked OwnerRRef with RRefId: " << rrefId;
      // 查找并删除 owners_ 中对应的 OwnerRRef
      auto iter = owners_.find(rrefId);
      TORCH_CHECK(
          iter != owners_.end(),
          c10::str("Did not find OwnerRRef with RRefId: ", rrefId));
      owners_.erase(iter);
    }
  }
  // 等待此节点处理所有可能接收到的删除 UserRRef 消息，
  // 这些消息可能是针对此节点上存在的 OwnerRRef 的。
  {
    // 获取互斥锁
    std::unique_lock<std::mutex> lock(mutex_);
    // 等待所有 owner 被删除，直到 owners_ 映射为空或超时
    bool noOwner = deleteAllUsersCV_.wait_for(
        lock, timeoutMillis, [this]() { return owners_.empty(); });
    // 如果超时仍有 owner 存在，则记录错误日志
    if (!noOwner) {
      LOG(ERROR) << "Timed out waiting for pending OwnerRRefs to be deleted.";
    }
  }
}

c10::intrusive_ptr<RRef> RRefContext::getOrCreateRRef(
    const RRefForkData& rrefForkData,
    const TypePtr& type) {
  auto& ownerId = rrefForkData.ownerId_;
  auto& rrefId = rrefForkData.rrefId_;
  auto& forkId = rrefForkData.forkId_;
  if (ownerId == getWorkerId()) {
    // 如果 ownerId 等于当前 Worker 的 ID，则调用 getOrCreateOwnerRRef 方法
    return getOrCreateOwnerRRef(rrefId, type);
  } else {
    // 否则调用 createUserRRef 方法创建用户端的 RRef
    return createUserRRef(ownerId, rrefId, forkId, type);
  }
}

c10::intrusive_ptr<OwnerRRef> RRefContext::getOrCreateOwnerRRef(
    const RRefId& rrefId,
    const TypePtr& type) {
  // 使用互斥锁保护操作
  std::lock_guard<std::mutex> lock(mutex_);
  // 查找 owners_ 中是否已经存在指定 rrefId 的 OwnerRRef
  const auto iter = owners_.find(rrefId);
  if (iter == owners_.end()) {
    // 如果不存在，说明这是首次发现该 RRef 的所有者
    //
    // 注意: 这里不能使用 make_shared，因为 OwnerRRef 的构造函数是私有的
    auto rref = c10::make_intrusive<OwnerRRef>(
        getWorkerId(), rrefId, type, agent_->getDevices());
    // 将新创建的 OwnerRRef 放入 owners_ 中
    owners_[rref->rrefId()] = rref;
    // 检查是否存在该 rrefId 的待处理 OwnerRRef
    const auto pendingOwnerIter = pendingOwners_.find(rrefId);
    if (pendingOwnerIter != pendingOwners_.end()) {
      // 将 OwnerRRef 转换为 RRefInterface 存入 IValue 中
      auto rrefPtr = fromOwnerRRef(rref);
      // 标记待处理的 OwnerRRef 已完成，并移除出 pendingOwners_
      pendingOwnerIter->second->markCompleted(IValue(rrefPtr));
      pendingOwners_.erase(pendingOwnerIter);
    }
    return rref;
  } else {
    // 如果已存在，直接从 owners_ 中获取现有的 OwnerRRef
    auto ownerRRef = fromRRefInterface(iter->second);
    // 双重检查两者的类型是否匹配
    //
    // 为什么在这里特别检查张量类型？
    // 这是因为张量类型可能在将输入传递给函数时进行了特殊化处理，例如填充了特定的形状信息、requires_grad 等。
    // 因此找到的 OwnerRRef 可能已经包含这些信息，但是我们传入的 `type` 是一个普通的 TensorType，它们并不相等:
    // specialized TensorType <: plain TensorType
    //
    // 在 RPC 中，我们不关心这种差异，因为我们只是用普通的 TensorType 进行序列化和反序列化。
    // 这对于用户端创建 UserRRef 也不是问题，因为张量只能在本地 JIT 函数的前一次运行中特殊化，我们不应将专门化的 SubTensorType 信息保留在其他工作进程上，因为这只是信息而已。
    if (type->isSubtypeOf(*TensorType::get())) {
      TORCH_INTERNAL_ASSERT(
          ownerRRef->type()->isSubtypeOf(*TensorType::get()),
          "Expect OwnerRRef to be a sub-type of TensorType, but got ",
          ownerRRef->type()->repr_str());
    } else {
      // 否则，检查 OwnerRRef 的类型是否与传入的 type 相等
      TORCH_INTERNAL_ASSERT(
          *ownerRRef->type() == *type,
          "OwnerRRef type is ",
          ownerRRef->type()->repr_str(),
          ", expected type is ",
          type->repr_str());
    }
    return ownerRRef;
  }
}

c10::intrusive_ptr<OwnerRRef> RRefContext::createOwnerRRef(
    // 返回一个新创建的 OwnerRRef 对象，用于管理指定类型的远程引用
    const TypePtr& type) {
      // 暂时不将此 OwnerRRef 添加到 owners_ 映射中，
      // 否则它将永远不会从那里移除。相反，只在 prepareChildFork 中添加它到映射，
      // 以防这个本地 RRef 被传递给另一个 worker。
      return c10::make_intrusive<OwnerRRef>(
          getWorkerId(), genGloballyUniqueId(), type, agent_->getDevices());
    }
// 获取指定 RRefId 对应的 OwnerRRef 对象
c10::intrusive_ptr<JitFuture> RRefContext::getOwnerRRef(
    const RRefId& rrefId,
    bool forceCreated) {
  // 使用互斥锁锁定当前对象的所有权操作
  std::unique_lock<std::mutex> lock(mutex_);
  // 在 owners_ 中查找给定的 rrefId
  const auto iter = owners_.find(rrefId);
  // 如果找不到对应的 OwnerRRef
  if (iter == owners_.end()) {
    // 如果需要强制创建 OwnerRRef
    if (forceCreated) {
      // 抛出内部断言错误，指示应该已经创建具有指定 rrefId 的 OwnerRRef
      TORCH_INTERNAL_ASSERT(
          false,
          c10::str("Expected OwnerRRef with id ", rrefId, " to be created."));
    }
    // 在 pendingOwners_ 中查找待处理的 OwnerRRef
    const auto pendingOwnerIter = pendingOwners_.find(rrefId);
    // 如果在 pendingOwners_ 中找不到对应的 OwnerRRef
    if (pendingOwnerIter == pendingOwners_.end()) {
      // 创建一个新的 JitFuture 对象，用于表示未来可能会创建的 OwnerRRef
      // 注意：这里传递给 RRefType::create() 的类型并不重要，因为在 getOrCreateOwnerRRef() 中会用具体类型的 RRef 来标记完成
      auto futureOwner = c10::make_intrusive<JitFuture>(
          RRefType::create(c10::AnyType::get()), agent_->getDevices());
      // 将新创建的 JitFuture 添加到 pendingOwners_ 中
      pendingOwners_[rrefId] = futureOwner;
      // 返回新创建的 JitFuture
      return futureOwner;
    } else {
      // 如果在 pendingOwners_ 中找到对应的 OwnerRRef，则直接返回它
      return pendingOwnerIter->second;
    }
  } else {
    // 如果在 owners_ 中找到对应的 OwnerRRef
    auto owner = iter->second;
    // 从 OwnerRRef 获得指针 rrefPtr
    auto rrefPtr = fromOwnerRRef(owner);

    // 创建一个新的 JitFuture 对象，用于表示 OwnerRRef 已完成
    auto futureOwner = c10::make_intrusive<JitFuture>(
        RRefType::create(owner->type()), agent_->getDevices());
    // 将 JitFuture 标记为已完成，并使用 rrefPtr 作为其值
    futureOwner->markCompleted(IValue(rrefPtr));
    // 返回已完成的 JitFuture
    return futureOwner;
  }
}

// 准备子分支的数据，以备后续处理
RRefForkData RRefContext::prepareChildFork(
    const c10::intrusive_ptr<RRef>& rref) {
  // 如果已知在所有者节点上创建 rref 超时，则抛出错误；否则继续序列化处理
  TORCH_CHECK(
      !rref->getTimedOut(),
      "RRef creation via rpc.remote() timed out, and it "
      "is possible that the RRef on the owner node does not exist.");
  // 调用 rref 的 fork() 方法获取分支数据
  auto rrefForkData = rref->fork();
  // 如果当前节点是 rref 的所有者
  if (rref->isOwner()) {
    // 注释 [Early Fork Registration]：
    // 如果父节点（调用方）是所有者，则直接注册分支，而不是等待 RREF_FORK_REQUEST 或 RREF_CHILD_ACCEPT 消息。
    // 另一种方法是在被调方确认之前添加分支。但在此之前，所有者仍需将 OwnerRRef 添加到某些映射中以保持其活动状态（例如 pendingChildren_）。
    // 因此，在此处或在确认中添加分支没有任何区别，只会增加复杂性。
    // 如果在添加失败重试和超时功能时，所有者没有在超时期限内收到确认ACK，
    // 则需要删除此处的 fork。
    addForkOfOwner(rrefForkData.rrefId_, rrefForkData.forkId_);
    // 确保该 RRef 在 owners_ 列表中，以保持其有效状态。
    // 对于本地创建的 OwnerRRefs，这一步是必需的。
    {
      // 使用互斥锁保护临界区域，以防止多线程竞争条件
      std::lock_guard<std::mutex> lock(mutex_);
      // 将 rref 添加到 owners_ 映射中，以便进行管理
      owners_[rref->rrefId()] = rref;
    }
  } else {
    // 注释 [Useful Phantom Fork ID for User to Owner Call]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 如果 dist.remote 或 dist.rpc 的调用方是该 RRef 的所有者，
    // 则调用方不会使用 rrefForkData.forkId_ 创建分支，因为
    // 所有者仅保留一个 OwnerRRef 实例，不会创建任何 UserRRef 实例。
    // 然而，rrefForkData.forkId_ 仍然是必需的，因为调用方需要保持
    // 此 UserRRef 的生命周期，直到它收到所有者的确认ACK。否则，
    // 在 dist.rpc 或 dist.remote 调用之前，删除消息可能会到达所有者，
    // 这可能会在运行用户代码之前触发删除 OwnerRRef 的操作。
    addPendingChild(rrefForkData.forkId_, rref);
  }
  // 返回处理后的 rrefForkData 结构体
  return rrefForkData;
}

void RRefContext::notifyOwnerAndParentOfFork(
    const ForkId& forkId,
    worker_id_t parent,
    const c10::intrusive_ptr<RRef>& rref) {
  // Fork is shared from owner.
  // 如果父进程是 RRef 的所有者
  if (parent == rref->owner()) {
    // 如果父进程是代理的当前工作进程
    if (parent == agent_->getWorkerInfo().id_) {
      // 所有者将 RRef 发送给自身，需要删除在 pickling 过程中添加的 forkId
      auto deletedRRef = delForkOfOwner(rref->rrefId(), forkId);
      // 断言删除的 RRef 是预期的 RRef
      TORCH_INTERNAL_ASSERT(
          deletedRRef->rrefId() == rref->rrefId(),
          "Deleting a fork of ",
          rref->rrefId(),
          " triggered deleting the OwnerRRef of ",
          deletedRRef->rrefId());
      // 注意：不需要重置 deletedRRef，因为 rref 是指向相同 OwnerRRef 的另一个 shared_ptr 实例
    } else {
      // 如果父进程是所有者，则在所有者将消息发送给调用用户时，fork 已经添加到 forks_ map 中
      // 因此，不需要再向所有者发送 RREF_CHILD_ACCEPT 或 RREF_FORK_REQUEST 消息。详见 Note [Early Fork Registration]
      std::lock_guard<std::mutex> lock(mutex_);
      // 将确认的用户添加到 forks_ map 中
      addConfirmedUser(forkId, rref);
    }
    return;
  }

  // Fork is shared from user.
  // 如果 RRef 是从用户共享的
  if (rref->isOwner()) {
    // 查看 Note [Useful Phantom Fork ID for User to Owner Call]
    // 在这种情况下，所有者是调用者，并且它不会将 fork id 添加到 forks_ 中，因为这个 fork id 不会关联任何真实的 UserRRef
    ++numPendingFutures_;
    // 发送 RRefChildAccept 消息给父进程，并带有重试
    auto jitFuture = agent_->sendWithRetries(
        agent_->getWorkerInfo(parent), RRefChildAccept(forkId).toMessage());
    // 添加回调函数处理异常和减少 numPendingFutures_
    jitFuture->addCallback([this](JitFuture& future) {
      handleExceptionSilent(future);
      --numPendingFutures_;
    });
  } else {
    ++numPendingFutures_;
    // 发送 RRefForkRequest 消息给所有者，并带有重试
    auto jitFuture = agent_->sendWithRetries(
        agent_->getWorkerInfo(rref->owner()),
        RRefForkRequest(rref->rrefId(), forkId).toMessage());

    // 将待处理的用户添加到 pendingChildren_ 中
    addPendingUser(forkId, rref);

    // 添加回调函数处理异常，完成 fork 请求，以及减少 numPendingFutures_
    jitFuture->addCallback([this, forkId, parent](JitFuture& future) {
      handleException(future);
      this->finishForkRequest(forkId, parent);
      // Decrease after calling finishForkRequest because, as that creates a new
      // future, it might otherwise cause the count to briefly go to zero.
      --numPendingFutures_;
    });
  }
}

void RRefContext::addPendingChild(
    const ForkId& forkId,
    const c10::intrusive_ptr<RRef>& rref) {
  // see Note [Early Fork Registration]
  // 如果父进程是所有者，则直接将子进程的 UserRRef 添加为一个 fork
  TORCH_INTERNAL_ASSERT(
      !rref->isOwner(), "OwnerRRef should not have a pending child.");
  std::lock_guard<std::mutex> lock(mutex_);
  // 断言 pendingChildren_ 中没有重复添加相同的 child fork
  TORCH_INTERNAL_ASSERT(
      pendingChildren_.find(forkId) == pendingChildren_.end(),
      "Inconsistent states: attempt to add the same child fork twice.");
  // 将 pendingChildren_ 中添加 forkId 对应的 rref
  pendingChildren_[forkId] = rref;
}
// 删除指定 forkId 对应的子用户追踪项
void RRefContext::delPendingChild(const ForkId& forkId) {
  // 创建一个空的 RRef 指针，用于存储将要删除的用户追踪项
  c10::intrusive_ptr<RRef> deletedUser;
  {
    // 使用 mutex_ 进行互斥访问保护
    std::lock_guard<std::mutex> lock(mutex_);
    // 查找指定 forkId 对应的子用户追踪项
    auto iter = pendingChildren_.find(forkId);
    // 检查子用户追踪项是否存在于 pendingChildren_ 中
    // 这确保了触发该函数的消息具有幂等性，即多次触发只会执行一次删除操作
    if (iter != pendingChildren_.end()) {
      // 将找到的用户追踪项赋值给 deletedUser，增加其引用计数
      deletedUser = iter->second; // 增加引用计数
      // 从 pendingChildren_ 中移除该用户追踪项
      pendingChildren_.erase(iter); // 减少引用计数
    } else {
      // 如果不存在对应的追踪项，则记录日志并忽略该请求
      LOG(INFO) << "Ignoring duplicate request to delete child UserRRef with "
                << "ForkId = " << forkId;
    }
  }
  // 通知所有等待线程，以删除所有用户
  deleteAllUsersCV_.notify_all();
  // 减少 deletedUser 的引用计数，可能会触发资源释放操作
  deletedUser.reset(); // 减少引用计数
}

// 添加指定 forkId 对应的待处理用户追踪项
void RRefContext::addPendingUser(
    const ForkId& forkId,
    const c10::intrusive_ptr<RRef>& rref) {
  // 断言待添加的 RRef 不是 OwnerRRef
  TORCH_INTERNAL_ASSERT(
      !rref->isOwner(), "Attempt to add an OwnerRRef as a pending User.");

  // 创建共享状态对象 PendingUserState，并初始化为 rref
  auto state = std::make_shared<PendingUserState>(rref);
  // 如果处于录制状态，将状态添加到 userTable_
  if (recording_) {
    // 由于添加和等待待处理用户是在同一线程中调用的，
    // 但删除待处理用户将从另一个线程调用。
    // 无法通过使 pendingUsers_ 成为 thread_local 来解决此问题，
    // 因此，pendingUsers_ 和 userTable_ 共享相同的 PendingUserState shared_ptr。
    userTable_.push_back(state);
  }

  // 使用 mutex_ 进行互斥访问保护
  std::lock_guard<std::mutex> lock(mutex_);
  // 断言 pendingUsers_ 中不存在指定的 forkId，保证添加操作是幂等的
  TORCH_INTERNAL_ASSERT(
      pendingUsers_.find(forkId) == pendingUsers_.end(),
      "Inconsistent states: attempt to add the same UserRRef twice.");

  // 将 forkId 和对应的 PendingUserState 添加到 pendingUsers_ 中
  pendingUsers_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(forkId),
      std::forward_as_tuple(state));
}

// 删除指定 forkId 对应的待处理用户追踪项
void RRefContext::delPendingUser(const ForkId& forkId) {
  // 创建一个空的 shared_ptr，用于存储将要删除的 PendingUserState
  std::shared_ptr<PendingUserState> deletedState = nullptr;
  {
    // 使用 mutex_ 进行互斥访问保护
    std::lock_guard<std::mutex> lock(mutex_);
    // 查找指定 forkId 对应的待处理用户追踪项
    auto iter = pendingUsers_.find(forkId);
    // 断言找到了指定的待处理用户追踪项
    TORCH_INTERNAL_ASSERT(
        iter != pendingUsers_.end(),
        "Inconsistent states: attempt to delete a non-exist UserRRef.");

    // 保持删除的 PendingUserState 存活的两个原因之一
    // （待处理用户的其他线程可能仍在访问其状态）
    // 创建一个临时 shared_ptr 来增加 PendingUserState 的引用计数，
    // 直到锁释放为止，从而在删除时确保不会销毁其状态
    deletedState = iter->second;
    // 从 pendingUsers_ 中移除该待处理用户追踪项
    pendingUsers_.erase(iter);
  }
}
    // 当退出临界区域后执行以下操作。
    // (1) 因为此 UserRRef 从映射中移除，其引用计数可能会降至 0。
    //     因此，资源析构函数 (`release_resources()`) 可能会被调用，在其中会再次获取锁。
    //     因此，必须在释放锁的情况下销毁它。为了满足这个条件，我们故意创建一个临时指针，
    //     增加已删除的 PendingUserState 的引用计数，延长其生命周期直到锁被释放。
    // (2) 自从 #34497 以来，用户函数只有在确认其所有参数的所有者确认后才运行，
    //     这通过将 RPC 处理逻辑作为回调添加到 UserRRef 就绪的 future 来完成。
    //     因此，在 PendingUserState 上调用 `confirm` 可能会触发待处理的用户函数，
    //     这些函数可能会尝试在 RRefContext 中获取锁。因此，我们必须释放锁以防止死锁。
    // 注意：另一个选项是使用可重入锁。但是，最好让开发人员充分理解锁的行为，而不是使用可重入锁隐藏这些微妙的逻辑。
    deletedState = iter->second; // 增加引用计数

    addConfirmedUser(forkId, iter->second->rref_);
    pendingUsers_.erase(iter); // 减少引用计数。
  }
  deletedState->confirm();
  deleteAllUsersCV_.notify_all();
  deletedState.reset(); // 减少引用计数。
}

// 向确认用户列表中添加用户
void RRefContext::addConfirmedUser(
    const ForkId& forkId,
    const c10::intrusive_ptr<RRef>& rref) {
  // 注意：调用者需持有 confirmedUsers_ 的互斥锁。
  // std::lock_guard<std::mutex> lock(mutex_);
  // 将 forkId 和对应的 RRef 添加到 confirmedUsers_ 中
  confirmedUsers_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(forkId),
      std::forward_as_tuple(rref));
}

// 获取指定 forkId 对应的待处理用户
c10::intrusive_ptr<RRef> RRefContext::getPendingUser(const ForkId& forkId) {
  // 获取 mutex_ 的互斥锁
  std::lock_guard<std::mutex> lock(mutex_);
  // 在 pendingUsers_ 中查找 forkId
  auto it = pendingUsers_.find(forkId);
  // 如果未找到，则抛出内部断言错误
  if (it == pendingUsers_.end()) {
    TORCH_INTERNAL_ASSERT(
        false, "Pending user with forkId ", forkId, " not found");
  }
  // 返回找到的 RRef 指针
  return it->second->rref_;
}

// 记录线程本地待处理的 RRefs
void RRefContext::recordThreadLocalPendingRRefs() {
  // 断言：当开始记录时，用户 RRef 表应为空
  TORCH_INTERNAL_ASSERT(
      userTable_.empty(),
      "User RRef Table should be empty when start recording");
  // 设置 recording_ 为 true，表示正在记录
  recording_ = true;
}

// 等待线程本地待处理的 RRefs 完成
c10::intrusive_ptr<JitFuture> RRefContext::waitForThreadLocalPendingRRefs() {
  // 创建一个 JitFuture 指针，包含一个 Bool 类型的值，传播设备信息
  auto jitFuturePtr =
      c10::make_intrusive<JitFuture>(BoolType::get(), agent_->getDevices());
  // 如果 userTable_ 为空，则标记 jitFuturePtr 为已完成状态
  if (userTable_.empty()) {
    jitFuturePtr->markCompleted(true);
  } else {
    // 创建一个共享的原子计数器 remainingRRefs，初始值为 userTable_ 的大小
    auto remainingRRefs =
        std::make_shared<std::atomic<uint64_t>>(userTable_.size());
    // 遍历 userTable_ 中的每个状态对象
    for (auto& state : userTable_) {
      // 向 confirmationFuture_ 添加回调函数，用于减少 remainingRRefs
      state->confirmationFuture_->addCallback(
          [jitFuturePtr, remainingRRefs](JitFuture& /* unused */) {
            auto localCount = remainingRRefs->fetch_sub(1);
            // 当计数器减少到 1 时，标记 jitFuturePtr 为已完成状态
            if (localCount == 1) {
              jitFuturePtr->markCompleted(true);
            }
          });
    }
    // 清空 userTable_
    userTable_.clear();
  }
  // 设置 recording_ 为 false，表示记录结束
  recording_ = false;
  // 返回 jitFuturePtr
  return jitFuturePtr;
}

// 在错误时清除记录的待处理 RRefs
void RRefContext::clearRecordedPendingRRefsOnError() {
  // 清空 userTable_
  userTable_.clear();
  // 设置 recording_ 为 false，表示记录结束
  recording_ = false;
}

// 处理 fork 请求完成后的操作
void RRefContext::finishForkRequest(const ForkId& forkId, worker_id_t parent) {
  // 删除 pendingUsers_ 中的指定 forkId
  delPendingUser(forkId);
  // 增加 numPendingFutures_ 计数
  ++numPendingFutures_;
  // 发送包含 RRefChildAccept 消息的 jitFuture 到指定 parent 的 agent
  auto jitFuture = agent_->sendWithRetries(
      agent_->getWorkerInfo(parent), RRefChildAccept(forkId).toMessage());

  // 向 jitFuture 添加回调函数
  jitFuture->addCallback([this](JitFuture& future) {
    // 处理异常情况
    handleExceptionSilent(future);
    // 减少 numPendingFutures_ 计数
    --numPendingFutures_;
  });
}

// 将自身作为 fork 添加到 owners_ 和 forks_ 中
void RRefContext::addSelfAsFork(c10::intrusive_ptr<OwnerRRef>& rref) {
  // 获取 mutex_ 的互斥锁
  std::lock_guard<std::mutex> lock(mutex_);
  // 获取 rref 的 rrefId
  const auto& rrefId = rref->rrefId();
  // 将 rref 添加到 owners_ 中
  owners_[rrefId] = rref;
  // 获取 rrefId 对应的 forks_
  auto& rrefForks = forks_[rrefId];
  // 断言：rrefForks 中不应包含 rrefId，以避免重复添加
  TORCH_INTERNAL_ASSERT(
      rrefForks.find(rrefId) == rrefForks.end(),
      "Attempt to add self as fork twice ",
      rrefId);
  // 将 rrefId 添加到 rrefForks 中
  rrefForks.insert(rrefId);
}
// 向指定的 RRefId 添加其对应的 forkId，确保并发安全
void RRefContext::addForkOfOwner(const RRefId& rrefId, const ForkId& forkId) {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> lock(mutex_);
  // 获取 rrefId 对应的所有 forkId 集合
  auto& rrefForks = forks_[rrefId];
  // 断言确保 forkId 在集合中不存在，避免重复添加
  TORCH_INTERNAL_ASSERT(
      rrefForks.find(forkId) == rrefForks.end(),
      "Got fork notification twice on the same RRef ",
      forkId);
  // 将 forkId 添加到集合中
  rrefForks.insert(forkId);
}

// 如果 forkId 不存在于 rrefId 的集合中，则添加该 forkId
void RRefContext::addForkOfOwnerIfNotPresent(
    const RRefId& rrefId,
    const ForkId& forkId) {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> lock(mutex_);
  // 获取 rrefId 对应的所有 forkId 集合
  auto& rrefForks = forks_[rrefId];
  // 检查 forkId 是否已经存在于集合中，确保函数幂等性
  if (rrefForks.find(forkId) == rrefForks.end()) {
    // 如果不存在，则添加 forkId 到集合中
    rrefForks.insert(forkId);
  } else {
    // 如果已经存在，则记录日志，表明忽略重复添加请求
    LOG(INFO) << "Ignoring duplicate request to add Fork of OwnerRRef with "
              << "RRefId = " << rrefId << ", ForkId = " << forkId;
  }
}

// 删除指定的 RRefId 和对应的 forkId，并返回已删除的 RRef 对象
c10::intrusive_ptr<RRef> RRefContext::delForkOfOwner(
    const RRefId& rrefId,
    const ForkId& forkId) {
  c10::intrusive_ptr<RRef> deletedRRef;
  bool ownerReduced = false;
  {
    // 使用互斥锁保护临界区
    std::lock_guard<std::mutex> lock(mutex_);
    // 查找 rrefId 对应的所有 forkId 集合
    auto rrefIter = forks_.find(rrefId);
    if (rrefIter != forks_.end()) {
      auto& rrefForks = rrefIter->second;
      // 查找指定的 forkId 是否存在
      auto forkIter = rrefForks.find(forkId);
      if (forkIter != rrefForks.end()) {
        // 如果存在，则从集合中删除该 forkId
        rrefForks.erase(forkIter);
      } else {
        // 如果不存在，则记录日志，表明找不到对应的 UserRRef 实例
        LOG(INFO)
            << "Could not find UserRRef instance, "
            << "RRefId = " << rrefId << ", ForkId = " << forkId
            << ", likely because it was deleted by a previously retried message";
      }
      // 如果集合为空，则尝试删除 owners_ 中对应的条目
      if (rrefForks.empty()) {
        auto ownerIter = owners_.find(rrefId);
        if (ownerIter != owners_.end()) {
          // 如果找到对应的 ownerRRef，则将其删除，并标记 ownerReduced 为 true
          deletedRRef = ownerIter->second;
          owners_.erase(ownerIter);
          ownerReduced = true;
        }
        // 最后从 forks_ 中删除该条目
        forks_.erase(rrefIter);
      }
    } else {
      // 如果找不到对应的 OwnerRRef，则记录日志，表明可能已经被之前的重试消息删除
      LOG(INFO)
          << "Could not find OwnerRRef with RRefId = " << rrefId
          << ", likely because it was deleted by a previously retried message";
    }
  }
  // 如果成功减少 ownerRRef 的数量，则通知所有等待删除的用户
  if (ownerReduced) {
    deleteAllUsersCV_.notify_all();
  }
  // 返回已删除的 RRef 对象
  return deletedRRef;
}
```