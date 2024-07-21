# `.\pytorch\torch\csrc\distributed\rpc\request_callback_no_python.cpp`

```
// 包含 Torch 分布式 RPC 请求回调所需的头文件

#include <torch/csrc/distributed/rpc/request_callback_no_python.h>

// 包含 C10 核心的流保护功能
#include <c10/core/StreamGuard.h>

// 包含 Torch 分布式自动求导相关的头文件
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>

// 包含 Torch 分布式 RPC 服务器端全局分析器的头文件
#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>

// Torch 分布式 RPC 的命名空间
namespace torch {
namespace distributed {
namespace rpc {

// 使用 Torch 分布式自动求导和 Torch 自动求导分析器的命名空间
using namespace torch::distributed::autograd;
using namespace torch::autograd::profiler;

// DistAutogradContextGuard 结构体用于在 processMessage() 执行完成后清理当前上下文 id
struct DistAutogradContextGuard {
  explicit DistAutogradContextGuard(int64_t ctxId) {
    auto& container = DistAutogradContainer::getInstance();
    prevCtxId_ = container.currentContextId(); // 保存当前自动求导上下文 id
    container.forceCurrentContextId(ctxId);   // 强制设置当前自动求导上下文 id
  }
  ~DistAutogradContextGuard() {
    auto& container = DistAutogradContainer::getInstance();
    container.forceCurrentContextId(prevCtxId_); // 恢复之前保存的自动求导上下文 id
  }

  int64_t prevCtxId_; // 前一个自动求导上下文 id
};

// RequestCallbackNoPython 类的 deserializePythonRpcCommand() 方法的实现
std::unique_ptr<RpcCommandBase> RequestCallbackNoPython::
    deserializePythonRpcCommand(
        std::unique_ptr<RpcCommandBase> rpc,
        const MessageType& messageType) const {
  // 检查消息类型，不支持 Python 调用
  TORCH_CHECK(
      messageType != MessageType::PYTHON_CALL &&
          messageType != MessageType::PYTHON_REMOTE_CALL,
      "Python calls are not supported!");
  return rpc; // 返回反序列化后的 RPC 命令
}

// RequestCallbackNoPython 类的 processMessage() 方法的实现
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processMessage(
    Message& request,
    std::vector<c10::Stream> streams) const {
  // 这里使用两个 future，因为在处理 RPC 消息时可能会暂停两次：
  //  1) 等待所有参数中的 RRef 确认完成；
  //  2) 等待 processRpc 完成。
  auto& rrefContext = RRefContext::getInstance();

  try {
    rrefContext.recordThreadLocalPendingRRefs(); // 记录本地线程待处理的 RRef

    // 反序列化请求中的 PythonUDF，触发 RRef 反序列化过程
    std::unique_ptr<RpcCommandBase> rpc = deserializePythonRpcCommand(
        deserializeRequest(request), request.type());

    auto rrefsReadyFuture = rrefContext.waitForThreadLocalPendingRRefs();
    auto retFuture = rrefsReadyFuture->thenAsync(
        [this,
         // 将 unique_ptr 转换为 shared_ptr，以便在 std::function 中使用
         rpc = (std::shared_ptr<RpcCommandBase>)std::move(rpc),
         // 获取请求消息类型
         messageType = request.type(),
         // 移动 streams 变量到 lambda 函数内部
         streams = std::move(streams)](JitFuture& /* unused */) mutable {
          // 使用 std::shared_lock，预请求检查的成本很小，大约为 10 微秒数量级
          // 获取当前的服务器进程全局分析器状态栈条目指针
          auto serverProcessGlobalProfilerStateStackEntryPtr =
              profiler::processglobal::StateStackEntry::current();
          // 如果服务器全局分析器已启用，则进一步支付线程局部分析器状态初始化的成本
          if (serverProcessGlobalProfilerStateStackEntryPtr) {
            // 从进程全局分析器状态初始化线程局部分析器状态
            enableProfilerLegacy(
                serverProcessGlobalProfilerStateStackEntryPtr->statePtr()
                    ->config());
          }

          // 处理 RPC 请求，并返回处理后的未来对象
          auto retFuture =
              processRpcWithErrors(*rpc, messageType, std::move(streams));

          // 在响应消息发送后执行的后续工作，不影响 RPC 的行程时间
          if (serverProcessGlobalProfilerStateStackEntryPtr) {
            // 恢复线程局部分析器状态
            thread_event_lists event_lists = disableProfilerLegacy();
            // 将线程局部 event_lists 放入进程全局分析器状态中
            profiler::processglobal::pushResultRecursive(
                serverProcessGlobalProfilerStateStackEntryPtr, event_lists);
          }

          // 返回处理后的未来对象
          return retFuture;
        },
        // 获取自定义类消息的类型
        c10::getCustomClassType<c10::intrusive_ptr<Message>>());

    // 对处理后的未来对象执行后续操作，设置消息 ID 并返回处理后的消息对象
    auto retFutureWithMessageId = retFuture->then(
        [id = request.id()](JitFuture& future) {
          // 将未来对象的值转换为自定义类消息
          c10::intrusive_ptr<Message> message =
              future.value().toCustomClass<Message>();
          // 设置消息 ID
          message->setId(id);
          // 执行带有存储的操作并返回结果
          return withStorages(message);
        },
        // 获取自定义类消息的类型
        c10::getCustomClassType<c10::intrusive_ptr<Message>>());

    // 返回具有消息 ID 的处理后的未来对象
    return retFutureWithMessageId;
  } catch (std::exception& e) {
    // 在异常捕获中，清除错误时记录的待处理 RRef
    rrefContext.clearRecordedPendingRRefsOnError();
    // 处理错误并返回作为未来对象的结果
    return asFuture(handleError(e, request.type(), request.id()));
  }
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRpcWithErrors(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    std::vector<c10::Stream> streams) const {
  try {
    // 尝试处理 RPC 请求，返回处理结果的未来对象
    return processRpc(rpc, messageType, std::move(streams));
  } catch (std::exception& e) {
    // 捕获异常并处理，返回处理异常后的未来对象
    // 传递一个虚拟的消息 ID，因为它将被覆盖
    return asFuture(handleError(e, messageType, -1));
  }
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processScriptCall(
    RpcCommandBase& rpc,
    std::vector<c10::Stream> streams) const {
  auto& scriptCall = static_cast<ScriptCall&>(rpc);

  // 检查 ScriptCall 是否包含操作符
  TORCH_CHECK(
      scriptCall.hasOp(), "Only supports the case where ScriptCall has an op");
  
  // 运行 JIT 操作符，并获取其返回的未来对象
  auto future = runJitOperator(
      *scriptCall.op(), scriptCall.stackRef(), std::move(streams));

  // 对未来对象应用 then 函数，处理异步操作结果
  return future->then(
      [](JitFuture& future) {
        // 将 JIT 未来对象的值转换为消息并返回
        return withStorages(ScriptResp(future.value()).toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processPythonCall(
    RpcCommandBase& rpc,
    std::vector<c10::Stream> /* unused */) const {
  // 抛出错误，不支持 Python 调用
  C10_THROW_ERROR(Error, "Python call not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processPythonRemoteCall(
    RpcCommandBase& rpc,
    std::vector<c10::Stream> /* unused */) const {
  // 抛出错误，不支持 Python 远程调用
  C10_THROW_ERROR(Error, "Python call not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::assignOwnerRRef(
    const RRefId& rrefId,
    const RRefId& forkId,
    c10::intrusive_ptr<JitFuture> valueFuture) const {
  auto& ctx = RRefContext::getInstance();

  c10::intrusive_ptr<OwnerRRef> ownerRRef;
  if (rrefId == forkId) {
    // 在本地创建所有者 RRef，应该已经存在于所有者映射中
    ownerRRef =
        fromRRefInterface(ctx.getOwnerRRef(rrefId, /* forceCreated */ true)
                              ->constValue()
                              .toRRef());
  } else {
    // 获取或创建所有者 RRef，并添加分支关系
    ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, valueFuture->elementType());
    // 如果调用方是用户且被调方是所有者，则添加分支
    //
    // 注意：rrefId == forkId 仅在远程调用自身时为真。在这种情况下，调用方和被调方都将访问 OwnerRRef。
    // 因此，在被调方（此处）不应调用 addForkOfOwner，因为这不是一个分支。为了允许被调方区分此请求何时发送到自身，
    // 调用方将使用 rrefId 设置 forkId（无论如何，OwnerRRef 不会有 forkId）。
    ctx.addForkOfOwner(rrefId, forkId);
  }

  // 对值的未来对象应用 then 函数，处理异步操作结果
  return valueFuture->then(
      [ownerRRef, rrefId, forkId](JitFuture& future) {
        // 如果未来对象有错误，则设置所有者 RRef 的错误
        // 否则，设置所有者 RRef 的值
        if (future.hasError()) {
          ownerRRef->setError(future.exception_ptr());
        } else {
          ownerRRef->setValue(future.value());
        }
        // 将远程返回消息封装为存储并返回
        return withStorages(RemoteRet(rrefId, forkId).toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}
// 处理远程脚本调用请求，返回一个 JitFuture 对象
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processScriptRemoteCall(
    RpcCommandBase& rpc,
    std::vector<c10::Stream> streams) const {
  // 将 RpcCommandBase 类型的参数转换为 ScriptRemoteCall 类型引用
  auto& scriptRemoteCall = static_cast<ScriptRemoteCall&>(rpc);

  // 检查 ScriptRemoteCall 是否包含操作符，否则抛出错误信息
  TORCH_CHECK(
      scriptRemoteCall.hasOp(), "ScriptRemoteCall needs to have an op!");
  
  // 调用 runJitOperator 执行 JIT 操作符，并返回执行结果的 JitFuture 对象
  auto future = runJitOperator(
      *scriptRemoteCall.op(), scriptRemoteCall.stackRef(), std::move(streams));

  // 分配 OwnerRRef，并返回其 JitFuture 对象
  return assignOwnerRRef(
      scriptRemoteCall.retRRefId(),
      scriptRemoteCall.retForkId(),
      std::move(future));
}

// 根据 RRefId 获取 OwnerRRef 的 JitFuture 对象
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::retrieveOwnerRRef(
    const RRefId& rrefId) const {
  // 获取 RRefContext 的实例
  auto& ctx = RRefContext::getInstance();

  // 获取指定 RRefId 对应的 OwnerRRef 的 JitFuture 对象
  auto rrefFuture = ctx.getOwnerRRef(rrefId);

  // 获取 OwnerRRef 的元素类型，并进行内部断言验证
  at::TypePtr type = rrefFuture->elementType();
  TORCH_INTERNAL_ASSERT(type->kind() == at::RRefType::Kind);
  
  // 返回 OwnerRRef 的 JitFuture 对象，并异步执行回调函数
  return rrefFuture->thenAsync(
      [](JitFuture& rrefFuture) {
        // 将 RRefInterface 转换为 OwnerRRef 对象，并获取其未来对象
        c10::intrusive_ptr<OwnerRRef> rref =
            fromRRefInterface(rrefFuture.value().toRRef());
        return rref->getFuture();
      },
      // 获取元素类型为 RRefType 的元素类型对象
      type->cast<at::RRefType>()->getElementType());
}

// 处理脚本 RRef 获取调用请求，返回一个 JitFuture 对象
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processScriptRRefFetchCall(RpcCommandBase& rpc) const {
  // 将 RpcCommandBase 类型的参数转换为 ScriptRRefFetchCall 类型引用
  auto& srf = static_cast<ScriptRRefFetchCall&>(rpc);

  // 获取指定 RRefId 的 OwnerRRef 的 JitFuture 对象
  auto future = retrieveOwnerRRef(srf.rrefId());

  // 返回 future 对象的 then 回调，获取带有存储的 ScriptRRefFetchRet 对象
  return future->then(
      [](JitFuture& future) {
        return withStorages(ScriptRRefFetchRet({future.value()}).toMessage());
      },
      // 获取自定义类类型为 Message 类型的对象
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

// 处理 Python RRef 获取调用请求，抛出错误信息
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processPythonRRefFetchCall(RpcCommandBase& rpc) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

// 处理 RRef 用户删除请求，执行删除操作并返回 RRef 确认消息的 JitFuture 对象
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRRefUserDelete(
    RpcCommandBase& rpc) const {
  // 将 RpcCommandBase 类型的参数转换为 RRefUserDelete 类型引用
  auto& rud = static_cast<RRefUserDelete&>(rpc);
  // 获取 RRefContext 的实例
  auto& ctx = RRefContext::getInstance();
  // 删除指定 RRefId 和 ForkId 的 OwnerRRef，返回删除后的 RRef 对象
  auto deletedRRef = ctx.delForkOfOwner(rud.rrefId(), rud.forkId());
  // 处理删除的 RRef 对象
  handleRRefDelete(deletedRRef);
  // 返回 RRefAck 消息的 JitFuture 对象
  return asFuture(RRefAck().toMessage());
}

// 处理 RRef 子对象接受请求，删除待处理的子对象并返回 RRef 确认消息的 JitFuture 对象
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRRefChildAccept(
    RpcCommandBase& rpc) const {
  // 将 RpcCommandBase 类型的参数转换为 RRefChildAccept 类型引用
  auto& rca = static_cast<RRefChildAccept&>(rpc);
  // 获取 RRefContext 的实例
  auto& ctx = RRefContext::getInstance();
  // 删除指定 ForkId 的待处理子对象，并返回 RRefAck 消息的 JitFuture 对象
  ctx.delPendingChild(rca.forkId());
  return asFuture(RRefAck().toMessage());
}

// 处理 RRef 分支请求，如果不存在则添加指定 RRefId 和 ForkId 的 OwnerRRef，并返回 RRef 确认消息的 JitFuture 对象
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRRefForkRequest(
    RpcCommandBase& rpc) const {
  // 将 RpcCommandBase 类型的参数转换为 RRefForkRequest 类型引用
  auto& rfr = static_cast<RRefForkRequest&>(rpc);
  // 获取 RRefContext 的实例
  auto& ctx = RRefContext::getInstance();
  // 添加指定 RRefId 和 ForkId 的 OwnerRRef，如果不存在
  ctx.addForkOfOwnerIfNotPresent(rfr.rrefId(), rfr.forkId());
  // 返回 RRefAck 消息的 JitFuture 对象
  return asFuture(RRefAck().toMessage());
}

// 处理 RRef 删除操作，验证是否存在 Python 对象的 RRef，并抛出错误信息
void RequestCallbackNoPython::handleRRefDelete(
    c10::intrusive_ptr<RRef>& rref) const {
  TORCH_CHECK(!rref->isPyObj(), "RRefs with python objects not supported!");
}
    // 处理前向自动微分请求的方法，接受一个RPC命令和一组流对象
    processForwardAutogradReq(
        RpcCommandBase& rpc,
        std::vector<c10::Stream> streams) const {
  auto& rpcWithAutograd = static_cast<RpcWithAutograd&>(rpc);

  // 需要为分布式自动微分的反向传播反转设备映射。
  DeviceMap reverseDeviceMap;
  // 遍历RPC命令中自动微分的设备映射
  for (const auto& mapEntry : rpcWithAutograd.deviceMap()) {
    reverseDeviceMap.insert({mapEntry.second, mapEntry.first});

// 将键值对 {mapEntry.second, mapEntry.first} 插入到 reverseDeviceMap 中。

  // Attach 'recv' autograd function.
  auto autogradContext = addRecvRpcBackward(
      rpcWithAutograd.autogradMetadata(),
      rpcWithAutograd.tensors(),
      rpcWithAutograd.fromWorkerId(),
      reverseDeviceMap);

  // 将 'recv' 反向自动求导函数附加到当前上下文中。
  // 使用 rpcWithAutograd 的自动求导元数据、张量和发送方的 worker ID，以及 reverseDeviceMap。

  // For this recv thread on server side, before processRpc(),
  // set current_context_id_ to be context_id passed from client.
  // In this way, if there is nested rpc call in python rpc call, original
  // context_id from client can be passed in the chain calls.
  TORCH_INTERNAL_ASSERT(
      autogradContext != nullptr,
      "autogradContext is nullptr, FORWARD_AUTOGRAD_REQ should always get "
      "or create valid autogradContext in addRecvRpcBackward.");

  // 在服务器端的此 recv 线程中，在执行 processRpc() 之前，
  // 将 current_context_id_ 设置为从客户端传递的 context_id。
  // 这样，在 Python RPC 调用中存在嵌套 RPC 调用时，可以在链式调用中传递原始的客户端 context_id。

  // 使用 TORCH_INTERNAL_ASSERT 断言 autogradContext 不为 nullptr，
  // 否则输出错误信息，指出在 addRecvRpcBackward 中 FORWARD_AUTOGRAD_REQ 应该始终获取或创建有效的 autogradContext。

  DistAutogradContextGuard ctxGuard(autogradContext->contextId());

// 使用 autogradContext 的 contextId 创建 DistAutogradContextGuard 对象 ctxGuard。

  // Process the original RPC.
  auto wrappedMessageType = rpcWithAutograd.wrappedMessageType();

  // 处理原始的 RPC 请求。

  // 获取 rpcWithAutograd 的 wrappedMessageType。

  // Kick off processing for the nested RPC command.
  // wrappedRpcResponseFuture will be a Future<T> to the result.
  auto wrappedRpcResponseFuture = processRpc(
      rpcWithAutograd.wrappedRpc(), wrappedMessageType, std::move(streams));

  // 开始处理嵌套的 RPC 命令。
  // wrappedRpcResponseFuture 将是处理结果的 Future<T>。

  // 使用 processRpc 处理 rpcWithAutograd 的 wrappedRpc，
  // 使用 wrappedMessageType 和移动的 streams 进行处理，并将结果保存在 wrappedRpcResponseFuture 中。

  auto fromWorkerId = rpcWithAutograd.fromWorkerId();

  // 获取 rpcWithAutograd 的 fromWorkerId。

  // The original future needs to be marked as completed when the wrapped
  // one completes, with the autograd context information wrapped.

  // 当 wrappedRpcResponseFuture 完成时，原始的 future 需要标记为已完成，
  // 并且包含自动求导上下文信息。

  // The original future needs to be marked as completed when the wrapped
  // one completes, with the autograd context information wrapped.

  auto responseFuture = wrappedRpcResponseFuture->then(
      [fromWorkerId, ctxId = autogradContext->contextId()](
          JitFuture& wrappedRpcResponseFuture) {

  // 创建 responseFuture，它会在 wrappedRpcResponseFuture 完成后被标记为完成。
  // 它会包含自动求导上下文信息。

      // As this callback can be invoked by a different thread, we have to
      // make sure that the thread_local states in the previous thread is
      // correctly propagated.
      // NB: The execution of TorchScript functions can also run on a
      // different thread, which is addressed by
      // https://github.com/pytorch/pytorch/pull/36395
      // NB: when adding async UDF support, we should also propagate
      // thread_local states there.
      // TODO: Land on a general solution for RPC ThreadLocalState. See
      // https://github.com/pytorch/pytorch/issues/38510
      DistAutogradContextGuard cbCtxGuard(ctxId);

      // 由于此回调可以由不同的线程调用，我们必须确保前一个线程的 thread_local 状态得到正确传播。
      // 注意：TorchScript 函数的执行也可能在不同的线程上运行，这通过 https://github.com/pytorch/pytorch/pull/36395 解决。
      // 注意：在添加异步用户定义函数（UDF）支持时，也应在那里传播 thread_local 状态。
      // TODO: 需要一个通用的解决方案来处理 RPC 的 ThreadLocalState。参见 https://github.com/pytorch/pytorch/issues/38510

      if (wrappedRpcResponseFuture.hasError()) {

      // 如果 wrappedRpcResponseFuture 有错误，抛出异常。

          // Propagate error to responseFuture if we had one.
          std::rethrow_exception(wrappedRpcResponseFuture.exception_ptr());
        } else {
          auto msg = getMessageWithAutograd(
              fromWorkerId,
              wrappedRpcResponseFuture.value().toCustomClass<Message>(),
              MessageType::FORWARD_AUTOGRAD_RESP);

          // 否则，使用 getMessageWithAutograd 创建带有自动求导的消息。

          return withStorages(std::move(msg));
        }
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());

      // 返回包含存储的结果消息。

  return responseFuture;

  // 返回 responseFuture。
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processBackwardAutogradReq(
        RpcCommandBase& rpc,
        std::vector<c10::Stream> streams) const {
  // 进入多流上下文保护区域
  c10::MultiStreamGuard guard(streams);
  // 强制类型转换为 PropagateGradientsReq 类型
  auto& gradientsCall = static_cast<PropagateGradientsReq&>(rpc);
  // 获取自动求导元数据
  const auto& autogradMetadata = gradientsCall.getAutogradMetadata();

  // 检索对应的自动求导上下文
  auto autogradContext = DistAutogradContainer::getInstance().retrieveContext(
      autogradMetadata.autogradContextId);

  // 查找适当的“send”函数以排队
  std::shared_ptr<SendRpcBackward> sendFunction =
      autogradContext->retrieveSendFunction(autogradMetadata.autogradMessageId);

  // 将梯度附加到发送函数
  sendFunction->setGrads(gradientsCall.getGrads());

  // 使用“分布式引擎”异步执行自动求导图
  auto execFuture = DistEngine::getInstance().executeSendFunctionAsync(
      autogradContext, sendFunction, gradientsCall.retainGraph());

  // 当 RPC 返回时满足我们的响应
  return execFuture->then(
      [](JitFuture& execFuture) {
        if (execFuture.hasError()) {
          // 如果执行未来发生错误，则重新抛出异常
          std::rethrow_exception(execFuture.exception_ptr());
        } else {
          // 否则返回带存储的 PropagateGradientsResp 消息
          return withStorages(PropagateGradientsResp().toMessage());
        }
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processCleanupAutogradContextReq(RpcCommandBase& rpc) const {
  // 强制类型转换为 CleanupAutogradContextReq 类型
  auto& cleanupContextReq = static_cast<CleanupAutogradContextReq&>(rpc);
  // 获取清理的自动求导上下文ID
  auto cleanupContextId = cleanupContextReq.getContextId();
  // 如果存在的话，在当前线程上释放上下文。需要检查是否存在，因为可能会被正在进行的 RPC 删除
  DistAutogradContainer::getInstance().releaseContextIfPresent(
      cleanupContextId);
  // 返回 CleanupAutogradContextResp 消息的未来
  return asFuture(CleanupAutogradContextResp().toMessage());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processRunWithProfilingReq(RpcCommandBase& rpc) const {
  // 强制类型转换为 RpcWithProfilingReq 类型
  auto& rpcWithProfilingReq = static_cast<RpcWithProfilingReq&>(rpc);
  // 获取包装的消息类型
  auto wrappedMsgType = rpcWithProfilingReq.wrappedMessageType();
  // 获取性能分析配置
  auto profilingConfig = rpcWithProfilingReq.getProfilingConfig();

  // 如果请求使用 KINETO 或 KINETO_GPU_FALLBACK 或 KINETO_PRIVATEUSE1_FALLBACK 进行性能分析，则将配置回退到 CPU
  if (profilingConfig.state == ProfilerState::KINETO ||
      profilingConfig.state == ProfilerState::KINETO_GPU_FALLBACK ||
      profilingConfig.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK) {
    profilingConfig = ProfilerConfig(
        ProfilerState::CPU,
        profilingConfig.report_input_shapes,
        profilingConfig.profile_memory);
  }

  // 如果调用者请求使用 CUDA 但是本机器上没有 CUDA 可用，则回退到 CPU，并记录警告而不是崩溃
  if (profilingConfig.state == ProfilerState::CUDA && !this->cudaAvailable()) {
    // ...
  }
    // 创建一个 ProfilerConfig 对象，指定使用 CPU 进行性能分析
    profilingConfig = ProfilerConfig(
        ProfilerState::CPU,
        profilingConfig.report_input_shapes,
        profilingConfig.profile_memory);

    // 如果在此节点请求启用 CUDA 但 CUDA 不可用，则记录警告信息并回退到仅 CPU 分析
    LOG(WARNING) << "Profiler was requested to be enabled with CUDA on this "
                    "node, but CUDA is not available. "
                 << "Falling back to CPU profiling only.";
  }
  
  // 断言检查，如果配置了使用 CUDA 进行性能分析，则必须确保 CUDA 可用
  TORCH_INTERNAL_ASSERT(
      profilingConfig.state != ProfilerState::CUDA || this->cudaAvailable(),
      "Profiler state set to CUDA but CUDA not available.");

  // 获取传入请求中的性能分析 ID
  const auto profilingKeyId = rpcWithProfilingReq.getProfilingId();

  // 使用来自发送方的配置启用性能分析器
  // 在主线程上启用时，确保清理分析器状态，但推迟到后续的所有分析事件的整合
  ProfilerDisableOptions requestThreadOptions(
      true /* cleanup TLS state */, false /* consolidate events */);

  {
    // 在 TLS 环境下启用旧版性能分析器
    TLSLegacyProfilerGuard g(
        profilingConfig, c10::nullopt, requestThreadOptions);
    
    // 断言检查，确保性能分析器已启用
    TORCH_INTERNAL_ASSERT(
        profilerEnabled(), "Expected profiler to be enabled!");

    // 开始处理嵌套工作并获取 Future<T> 类型的结果在 wrappedRpcResponseFuture 中
    auto wrappedRpcResponseFuture = processRpc(
        rpcWithProfilingReq.wrappedRpc(),
        wrappedMsgType,
        {}); // TODO: https://github.com/pytorch/pytorch/issues/55757

    // 在 wrappedRpcResponseFuture 上注册一个回调，处理异步返回的事件
    auto responseFuture = wrappedRpcResponseFuture->then(
        at::wrapPropagateTLSState([profilingKeyId, profilingConfig](
                                      JitFuture& wrappedRpcResponseFuture) {
          std::vector<LegacyEvent> profiledEvents;
          
          // 推迟分析器事件的整合，直到异步工作完成（如异步 UDF）

          TORCH_INTERNAL_ASSERT(
              profilerEnabled(), "Expected profiler to be enabled!");

          // 在继续线程上，不清理分析器状态，因为它们将由主线程清理，并整合所有事件以异步获取运行的事件
          ProfilerDisableOptions opts(false, true);
          auto event_lists = disableProfilerLegacy(opts);

          if (wrappedRpcResponseFuture.hasError()) {
            // 如果有错误，传播错误
            // 在出错的情况下不需要传播远程事件
            std::rethrow_exception(wrappedRpcResponseFuture.exception_ptr());
          } else {
            // 填充远程分析事件
            populateRemoteProfiledEvents(
                profiledEvents, profilingConfig, event_lists);
            
            // 创建带有性能分析响应的 RpcWithProfilingResp 对象
            auto rpcWithProfilingResp = std::make_unique<RpcWithProfilingResp>(
                MessageType::RUN_WITH_PROFILING_RESP,
                wrappedRpcResponseFuture.value().toCustomClass<Message>(),
                profiledEvents,
                profilingKeyId);
            
            // 返回带有存储的消息
            return withStorages(std::move(*rpcWithProfilingResp).toMessage());
          }
        }),
        c10::getCustomClassType<c10::intrusive_ptr<Message>>());
    return responseFuture;
    // 返回 responseFuture 变量，表示异步操作的未来结果

    // 退出当前作用域将使用上述指定选项禁用此线程上的分析器。
  }
}

// 实现处理 RRef 反向请求的方法
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRRefBackward(
    RpcCommandBase& rpc) const {
  // 抛出错误，不支持 Python 调用
  C10_THROW_ERROR(Error, "Python call not supported!");
}

// 实现处理 RPC 请求的方法
c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRpc(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    std::vector<c10::Stream> streams) const {
  // TODO: RpcCommandBase 应该有一个抽象的 execute() 方法，可以在这里调用，而不是在这里有另一个 switch 语句。
  // 更好的做法是我们可以有抽象类 RpcRequest 和 RpcResp，它们从 RpcCommandBase 和 RpcRequest 继承，
  // RpcRequest 声明了我们可以在这里调用的抽象方法 execute()。RpcResponse 可以有一个将其转换为 Python 对象的抽象方法。
  switch (messageType) {
    case MessageType::SCRIPT_CALL: {
      // 处理脚本调用消息类型
      return processScriptCall(rpc, std::move(streams));
    }
    case MessageType::PYTHON_CALL: {
      // 处理 Python 调用消息类型
      return processPythonCall(rpc, std::move(streams));
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      // 处理远程脚本调用消息类型
      return processScriptRemoteCall(rpc, std::move(streams));
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      // 处理远程 Python 调用消息类型
      return processPythonRemoteCall(rpc, std::move(streams));
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      // 处理脚本 RRef 获取调用消息类型
      return processScriptRRefFetchCall(rpc);
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      // 处理 Python RRef 获取调用消息类型
      return processPythonRRefFetchCall(rpc);
    }
    case MessageType::RREF_USER_DELETE: {
      // 处理 RRef 用户删除消息类型
      return processRRefUserDelete(rpc);
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      // 处理 RRef 子节点接受消息类型
      return processRRefChildAccept(rpc);
    }
    case MessageType::RREF_FORK_REQUEST: {
      // 处理 RRef 分叉请求消息类型
      return processRRefForkRequest(rpc);
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      // 处理自动微分前向请求消息类型
      return processForwardAutogradReq(rpc, std::move(streams));
    }
    case MessageType::BACKWARD_AUTOGRAD_REQ: {
      // 处理自动微分反向请求消息类型
      return processBackwardAutogradReq(rpc, std::move(streams));
    };
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: {
      // 处理清理自动微分上下文请求消息类型
      return processCleanupAutogradContextReq(rpc);
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
      // 处理运行带性能分析请求消息类型
      return processRunWithProfilingReq(rpc);
    }
    case MessageType::RREF_BACKWARD_REQ: {
      // 处理 RRef 反向请求消息类型
      return processRRefBackward(rpc);
    }
    default: {
      // 如果未知请求类型，则抛出内部断言错误
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", messageType, " not supported.");
    }
  }
}

// 处理错误时返回异常响应的方法
c10::intrusive_ptr<Message> RequestCallbackNoPython::handleError(
    const std::exception& e,
    const MessageType messageType,
    int64_t messageId) const {
  // 记录错误日志，包括消息类型和错误信息
  LOG(ERROR) << "Received error while processing request type " << messageType
             << ": " << e.what();
  // 将节点信息添加到错误消息中，因为所有处理过的 RPC 请求应该通过这个函数。
  std::string errorMsg = c10::str(
      "Error on Node ",
      DistAutogradContainer::getInstance().getWorkerId(),
      ": ",
      e.what());
  // 创建异常响应并返回
  return createExceptionResponse(errorMsg, messageId);
}

// 检查 CUDA 是否可用的方法
bool RequestCallbackNoPython::cudaAvailable() const {
#ifdef USE_CUDA
  // 如果编译时启用了 CUDA 支持，返回 true
  return true;
#else
  // 如果编译时未启用 CUDA 支持，返回 false
  return false;
#endif
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::runJitOperator(
    const jit::Operator& op,
    std::vector<at::IValue>& stack,
    std::vector<c10::Stream> streams) const {
  // 在多个流上执行操作，确保操作的并行性
  c10::MultiStreamGuard guard(streams);
  try {
    // 调用操作符的执行函数，传入操作数栈
    op.getOperation()(stack);
  } catch (const std::exception&) {
    // 如果操作执行中发生异常，返回一个异常状态的未来对象
    return asFuture(std::current_exception());
  }
  // 断言操作执行后操作数栈只有一个元素
  TORCH_INTERNAL_ASSERT(
      stack.size() == 1,
      "Return value of a builtin operator or a TorchScript function should be "
      "a single IValue, got a vector of size ",
      stack.size());
  // 获取操作数栈顶元素的类型
  TypePtr type = stack.front().type();
  // 将操作数栈顶元素和其类型封装成一个已完成的未来对象并返回
  return asFuture(std::move(stack.front()), std::move(type));
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    IValue value,
    TypePtr type) const {
  // 创建一个包含给定类型和当前设备的未来对象
  auto future = c10::make_intrusive<JitFuture>(
      std::move(type), RpcAgent::getCurrentRpcAgent()->getDevices());
  // 标记未来对象为已完成状态，设置其返回值
  future->markCompleted(std::move(value));
  // 返回已完成的未来对象
  return future;
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    c10::intrusive_ptr<Message> message) const {
  // 创建一个包含自定义类类型和当前设备的未来对象
  auto future = c10::make_intrusive<JitFuture>(
      at::getCustomClassType<c10::intrusive_ptr<Message>>(),
      RpcAgent::getCurrentRpcAgent()->getDevices());
  // 获取消息中的存储列表
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> storages =
      message->getStorages();
  // 标记未来对象为已完成状态，设置其返回值和存储列表
  future->markCompleted(std::move(message), std::move(storages));
  // 返回已完成的未来对象
  return future;
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    std::exception_ptr err) const {
  // 创建一个包含 None 类型和当前设备的未来对象，用于表示异常
  auto future = c10::make_intrusive<JitFuture>(
      at::NoneType::get(), RpcAgent::getCurrentRpcAgent()->getDevices());
  // 设置未来对象的错误状态
  future->setError(err);
  // 返回已完成的未来对象
  return future;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```