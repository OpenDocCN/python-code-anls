# `.\pytorch\torch\csrc\distributed\rpc\torchscript_functions.cpp`

```py
// 包含头文件：ATen/ThreadLocalState.h 提供了线程局部状态管理的支持
// 包含头文件：fmt/format.h 提供了格式化字符串的支持
// 包含头文件：torch/csrc/autograd/record_function_ops.h 提供了记录函数调用的操作支持
// 包含头文件：torch/csrc/distributed/autograd/utils.h 提供了分布式自动求导的实用函数支持
// 包含头文件：torch/csrc/distributed/rpc/message.h 提供了RPC消息的定义支持
// 包含头文件：torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h 提供了远程性能分析管理的支持
// 包含头文件：torch/csrc/distributed/rpc/rpc_agent.h 提供了RPC代理的定义支持
// 包含头文件：torch/csrc/distributed/rpc/rref_proto.h 提供了RPC远程引用协议的支持
// 包含头文件：torch/csrc/distributed/rpc/script_call.h 提供了RPC脚本调用的支持
// 包含头文件：torch/csrc/distributed/rpc/torchscript_functions.h 提供了TorchScript函数的支持
// 包含头文件：torch/csrc/distributed/rpc/utils.h 提供了RPC实用函数的支持

namespace torch {
namespace distributed {
namespace rpc {

// rpcTorchscript函数定义，用于远程执行TorchScript函数
c10::intrusive_ptr<JitFuture> rpcTorchscript(
    const std::string& dstWorkerName,  // 目标工作节点的名称
    const c10::QualifiedName& qualifiedName,  // TorchScript函数的限定名
    const c10::FunctionSchema& functionSchema,  // 函数的Schema信息
    std::vector<c10::IValue>& stack,  // 函数调用时的输入参数栈
    const float rpcTimeoutSeconds,  // RPC超时时间
    const bool isAsyncExecution) {  // 是否异步执行

  // 记录函数调用信息到profiler，如果启用了profiler并且当前远程性能分析未设置当前键
  c10::intrusive_ptr<torch::autograd::profiler::PythonRecordFunction> record;
  auto shouldProfile = torch::autograd::profiler::profilerEnabled() &&
      !torch::distributed::rpc::RemoteProfilerManager::getInstance()
           .isCurrentKeySet();
  if (shouldProfile) {
    // 构建远程异步JIT调用的键名
    auto rpcAsyncJitKey = fmt::format(
        "rpc_async_jit#{}({} -> {})",
        qualifiedName.qualifiedName(),  // 正在运行的TorchScript函数名称
        RpcAgent::getCurrentRpcAgent()->getWorkerInfo().name_,  // 当前RPC代理的工作节点名称
        dstWorkerName);  // 目标工作节点名称
    // 记录进入新的函数调用记录
    record = torch::autograd::profiler::record_function_enter_new(rpcAsyncJitKey);
    // 获取远程性能分析管理器的单例并设置当前键
    auto& remoteProfilerManager = torch::distributed::rpc::RemoteProfilerManager::getInstance();
    remoteProfilerManager.setCurrentKey(rpcAsyncJitKey);
  }

  // 创建ScriptCall对象，用于封装TorchScript函数调用
  auto scriptCall = std::make_unique<ScriptCall>(
      qualifiedName, std::move(stack), isAsyncExecution);
  // 获取当前RPC代理指针
  auto rpcAgentPtr = RpcAgent::getCurrentRpcAgent();
  // 发送带有自动求导信息的消息，并返回JitFuture对象
  auto jitFuture = autograd::sendMessageWithAutograd(
      *rpcAgentPtr,
      rpcAgentPtr->getWorkerInfo(dstWorkerName),  // 目标工作节点的信息
      std::move(*scriptCall).toMessage(),  // 将ScriptCall对象转换为消息
      true /*forceGradRecording*/,  // 强制梯度记录
      rpcTimeoutSeconds);  // RPC超时时间

  // 获取函数返回类型以构造JitFuture
  auto returns = functionSchema.returns();
  // TorchScript函数只允许返回单个IValue
  TORCH_INTERNAL_ASSERT(
      returns.size() == 1,
      "Return value of an annotated torchScript function should be a single "
      "IValue.",
      returns.size());
  auto returnType = returns.at(0).type();

  // 创建JIT Future，并将其传递给futMessage的回调函数来设置JIT Future的状态
  auto futPtr = jitFuture->createInstance(returnType);
  jitFuture->addCallback(at::wrapPropagateTLSState([futPtr](JitFuture& future) {
    if (future.hasError()) {
      futPtr->setError(future.exception_ptr());
    } else {
      futPtr->markCompleted(
          deserializeRespToIValue(
              *future.constValue().toCustomClass<Message>()),  // 反序列化响应消息为IValue
          future.storages());  // 存储器信息
    }
  }));

  // 如果启用了profiler
  if (shouldProfile) {
    // 下面的代码段被省略
    // 调用 PyTorch 自动微分库中的分析器函数，执行结束时回调函数，并返回新的 Future 指针
    auto profiledFutPtr =
        torch::autograd::profiler::_call_end_callbacks_on_fut_new(
            record, futPtr);
    // 返回经过分析器处理后的 Future 指针
    return profiledFutPtr;
  }
  // 如果没有执行分析器函数，则直接返回原始的 Future 指针
  return futPtr;
} // 结束函数 remoteTorchscript

namespace rpc {
namespace distributed {
namespace torch {
```