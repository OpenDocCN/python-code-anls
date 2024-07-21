# `.\pytorch\torch\csrc\distributed\rpc\request_callback_impl.cpp`

```py
// 包含头文件：请求处理实现的定义
#include <torch/csrc/distributed/rpc/request_callback_impl.h>

// 包含自动微分相关的头文件
#include <torch/csrc/autograd/profiler.h>
// 包含分布式自动微分上下文容器的定义
#include <torch/csrc/distributed/autograd/context/container.h>
// 包含分布式自动微分上下文的定义
#include <torch/csrc/distributed/autograd/context/context.h>
// 包含分布式自动微分引擎的定义
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
// 包含清理自动微分上下文请求的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
// 包含清理自动微分上下文响应的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
// 包含梯度传播请求的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
// 包含梯度传播响应的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
// 包含带自动微分的 RPC 请求的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
// 包含带性能分析的 RPC 请求的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
// 包含带性能分析的 RPC 响应的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>
// 包含远程引用后向传播请求的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h>
// 包含远程引用后向传播响应的消息定义
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_resp.h>
// 包含分布式自动微分实用函数的定义
#include <torch/csrc/distributed/autograd/utils.h>
// 包含 RPC 服务器全局性能分析器的定义
#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>
// 包含 Python 远程引用的定义
#include <torch/csrc/distributed/rpc/py_rref.h>
// 包含 Python 调用的定义
#include <torch/csrc/distributed/rpc/python_call.h>
// 包含 Python 远程调用的定义
#include <torch/csrc/distributed/rpc/python_remote_call.h>
// 包含 Python RPC 响应的定义
#include <torch/csrc/distributed/rpc/python_resp.h>
// 包含 Python RPC 处理器的定义
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
// 包含远程引用上下文的定义
#include <torch/csrc/distributed/rpc/rref_context.h>
// 包含远程引用实现的定义
#include <torch/csrc/distributed/rpc/rref_impl.h>
// 包含远程引用协议的定义
#include <torch/csrc/distributed/rpc/rref_proto.h>
// 包含脚本调用的定义
#include <torch/csrc/distributed/rpc/script_call.h>
// 包含脚本远程调用的定义
#include <torch/csrc/distributed/rpc/script_remote_call.h>
// 包含脚本 RPC 响应的定义
#include <torch/csrc/distributed/rpc/script_resp.h>
// 包含反序列化 Python 调用的定义
#include <torch/csrc/distributed/rpc/unpickled_python_call.h>
// 包含反序列化 Python 远程调用的定义
#include <torch/csrc/distributed/rpc/unpickled_python_remote_call.h>
// 包含 RPC 实用函数的定义
#include <torch/csrc/distributed/rpc/utils.h>
// 包含 Python IValue 的定义
#include <torch/csrc/jit/python/python_ivalue.h>

// 命名空间 torch::distributed::rpc 中定义
namespace torch {
namespace distributed {
namespace rpc {

// 使用命名空间 torch::distributed::autograd 中的内容
using namespace torch::distributed::autograd;

// 匿名命名空间定义，实现了反序列化 Python RPC 命令引用的函数
namespace {

// 反序列化 Python RPC 命令引用的函数
std::unique_ptr<RpcCommandBase> deserializePythonRpcCommandReference(
    RpcCommandBase& rpc, // 传入的 RPC 命令基类的引用
    const MessageType& messageType) { // 消息类型的引用
  switch (messageType) { // 根据消息类型进行分支判断
    case MessageType::PYTHON_CALL: { // 如果消息类型是 Python 调用
      auto& pc = static_cast<PythonCall&>(rpc); // 将 rpc 强制转换为 PythonCall 类型的引用
      // 返回一个指向 UnpickledPythonCall 对象的唯一指针，该对象包含反序列化后的 Python 调用信息
      return std::make_unique<UnpickledPythonCall>(
          pc.serializedPyObj(), pc.isAsyncExecution());
    }
    case MessageType::PYTHON_REMOTE_CALL: { // 如果消息类型是 Python 远程调用
      auto& prc = static_cast<PythonRemoteCall&>(rpc); // 将 rpc 强制转换为 PythonRemoteCall 类型的引用
      // 返回一个指向 UnpickledPythonRemoteCall 对象的唯一指针，该对象包含反序列化后的 Python 远程调用信息
      return std::make_unique<UnpickledPythonRemoteCall>(
          prc.serializedPyObj(),
          prc.retRRefId(),
          prc.retForkId(),
          prc.isAsyncExecution());
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
        // 如果消息类型是 FORWARD_AUTOGRAD_REQ
        // 尝试反序列化包含 Python UDF 的 RPC
        auto& rwa = static_cast<RpcWithAutograd&>(rpc);
        auto& wrappedRpc = rwa.wrappedRpc();
        auto pythonRpc = deserializePythonRpcCommandReference(
            wrappedRpc, rwa.wrappedMessageType());
        // 如果成功反序列化 Python RPC，则更新包装的 RPC 对象
        if (pythonRpc) {
            rwa.setWrappedRpc(std::move(pythonRpc));
        }
        // 返回空指针，表示处理完成
        return nullptr;
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
        // 如果消息类型是 RUN_WITH_PROFILING_REQ
        // 尝试反序列化包含 Python 调用的 RPC
        auto& rpcWithProfilingReq = static_cast<RpcWithProfilingReq&>(rpc);
        auto& wrappedRpc = rpcWithProfilingReq.wrappedRpc();
        auto pythonRpc = deserializePythonRpcCommandReference(
            wrappedRpc, rpcWithProfilingReq.wrappedMessageType());
        // 如果成功反序列化 Python RPC，则更新包装的 RPC 对象
        if (pythonRpc) {
            rpcWithProfilingReq.setWrappedRpc(std::move(pythonRpc));
        }
        // 返回空指针，表示处理完成
        return nullptr;
    }
    default: {
        // 默认情况下，返回空指针，表示未处理该消息类型
        return nullptr;
    }
}
} // anonymous namespace



SerializedPyObj serializePyObject(IValue value) {
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  // 获取全局解释器锁（GIL）以保护 jit::toPyObj 并销毁其返回的 py::object
  py::gil_scoped_acquire acquire;
  try {
    // 将 IValue 序列化为 PyObject，并交给 PythonRpcHandler 进行序列化
    return pythonRpcHandler.serialize(jit::toPyObject(value));
  } catch (py::error_already_set& e) {
    // py::error_already_set 需要 GIL 来析构，需特别注意
    auto err = std::runtime_error(e.what());
    e.restore();
    PyErr_Clear();
    throw err;
  }
}



c10::intrusive_ptr<JitFuture> RequestCallbackImpl::runPythonFunction(
    const py::object& function,
    std::vector<c10::Stream> streams,
    bool isAsyncExecution) const {
  c10::MultiStreamGuard guard(streams);
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  py::gil_scoped_acquire acquire;

  py::object result;
  try {
    // 调用 PythonRpcHandler 的方法执行 Python 用户定义函数（UDF）
    result = pythonRpcHandler.runPythonUdf(function);
  } catch (py::error_already_set& e) {
    // 处理 Python 错误，需要 GIL 来析构异常对象
    auto future =
        asFuture(std::make_exception_ptr(std::runtime_error(e.what())));
    e.restore();
    PyErr_Clear();
    return future;
  } catch (std::exception& e) {
    return asFuture(std::current_exception());
  }

  // 同步执行后或异步执行失败时，直接返回结果值
  if (pythonRpcHandler.isRemoteException(result) || !isAsyncExecution) {
    return asFuture(
        c10::ivalue::ConcretePyObjectHolder::create(result),
        at::PyObjectType::get());
  }

  try {
    // 尝试将结果转换为 jit::PythonFutureWrapper 类型并返回其 future
    return result.cast<jit::PythonFutureWrapper&>().fut;
  } catch (const py::cast_error& e) {
    auto type = result.get_type();
    auto errMsg = c10::str(
        e.what(),
        ". Functions decorated with @rpc.async_function must return a "
        "torch.futures.Future object, but got ",
        type.attr("__module__").cast<std::string>(),
        ".",
        type.attr("__qualname__").cast<std::string>());
    return asFuture(std::make_exception_ptr(std::runtime_error(errMsg)));
  }
}



std::unique_ptr<RpcCommandBase> RequestCallbackImpl::
    deserializePythonRpcCommand(
        std::unique_ptr<RpcCommandBase> rpc,
        const MessageType& messageType) const {
  // 反序列化 Python RPC 命令
  auto pythonRpc = deserializePythonRpcCommandReference(*rpc, messageType);
  // 如果成功反序列化，返回结果；否则返回原始 rpc 对象的所有权
  return pythonRpc ? std::move(pythonRpc) : std::move(rpc);
}



c10::intrusive_ptr<JitFuture> RequestCallbackImpl::processScriptCall(
    RpcCommandBase& rpc,
    std::vector<c10::Stream> streams) const {
  auto& scriptCall = static_cast<ScriptCall&>(rpc);

  c10::intrusive_ptr<JitFuture> future;
  if (scriptCall.hasOp()) {
    // 如果 ScriptCall 包含操作符，则执行 JIT 操作
    future = runJitOperator(
        *scriptCall.op(), scriptCall.stackRef(), std::move(streams));
  } else {
  // 调用 JIT 函数，并传入函数名、栈引用、流对象、以及异步执行标志
  future = runJitFunction(
      scriptCall.qualifiedName(),  // JIT 函数的限定名
      scriptCall.stackRef(),       // 函数调用的栈引用
      std::move(streams),          // 移动流对象的所有权
      scriptCall.isAsyncExecution());  // 标志是否异步执行

} // 结束代码块

// 返回 Future 对象的 then 函数结果，对 JIT 执行结果进行处理
return future->then(
    [](JitFuture& jitFuture) {  // 使用 lambda 表达式处理 Future 中的 JIT 执行结果
      // 将 JIT 执行结果转换为 ScriptResp 对象，并转换为消息格式返回
      return withStorages(ScriptResp(jitFuture.value()).toMessage());
    },
    c10::getCustomClassType<c10::intrusive_ptr<Message>>());  // 指定处理结果的类型
}

// 处理 Python 调用请求，返回异步执行的未来对象
c10::intrusive_ptr<JitFuture> RequestCallbackImpl::processPythonCall(
    RpcCommandBase& rpc,
    std::vector<c10::Stream> streams) const {
  // 将 RpcCommandBase 转换为 UnpickledPythonCall 对象
  auto& upc = static_cast<UnpickledPythonCall&>(rpc);
  // 运行 Python 函数，获取异步执行的未来对象
  auto future = runPythonFunction(
      upc.pythonUdf(), std::move(streams), upc.isAsyncExecution());

  // 返回异步执行的未来对象，执行完成后序列化 Python 对象并封装成消息
  return future->then(
      [](JitFuture& future) {
        return withStorages(
            PythonResp(serializePyObject(future.value())).toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

// 处理脚本远程调用请求，返回异步执行的未来对象
c10::intrusive_ptr<JitFuture> RequestCallbackImpl::processScriptRemoteCall(
    RpcCommandBase& rpc,
    std::vector<c10::Stream> streams) const {
  // 将 RpcCommandBase 转换为 ScriptRemoteCall 对象
  auto& scriptRemoteCall = static_cast<ScriptRemoteCall&>(rpc);

  c10::intrusive_ptr<JitFuture> future;
  // 如果存在操作符，则运行 JIT 操作符
  if (scriptRemoteCall.hasOp()) {
    future = runJitOperator(
        *scriptRemoteCall.op(),
        scriptRemoteCall.stackRef(),
        std::move(streams));
  } else {
    // 否则运行 JIT 函数
    future = runJitFunction(
        scriptRemoteCall.qualifiedName(),
        scriptRemoteCall.stackRef(),
        std::move(streams),
        scriptRemoteCall.isAsyncExecution());
  }

  // 返回拥有者引用对象的未来对象
  return assignOwnerRRef(
      scriptRemoteCall.retRRefId(),
      scriptRemoteCall.retForkId(),
      std::move(future));
}

// 处理 Python 远程调用请求，返回异步执行的未来对象
c10::intrusive_ptr<JitFuture> RequestCallbackImpl::processPythonRemoteCall(
    RpcCommandBase& rpc,
    std::vector<c10::Stream> streams) const {
  // 将 RpcCommandBase 转换为 UnpickledPythonRemoteCall 对象
  auto& uprc = static_cast<UnpickledPythonRemoteCall&>(rpc);
  // 运行 Python 函数，获取异步执行的未来对象
  auto future = runPythonFunction(
      uprc.pythonUdf(), std::move(streams), uprc.isAsyncExecution());

  // 返回分配拥有者引用对象的未来对象
  return assignOwnerRRef(uprc.rrefId(), uprc.forkId(), std::move(future));
}

// 处理 Python RRef 获取请求，返回异步执行的未来对象
c10::intrusive_ptr<JitFuture> RequestCallbackImpl::processPythonRRefFetchCall(
    RpcCommandBase& rpc) const {
  // 将 RpcCommandBase 转换为 PythonRRefFetchCall 对象
  auto& prf = static_cast<PythonRRefFetchCall&>(rpc);

  // 检索拥有者引用对象的未来对象
  auto future = retrieveOwnerRRef(prf.rrefId());

  // 返回异步执行的未来对象，执行完成后序列化 Python 对象并封装成消息
  return future->then(
      [](JitFuture& future) {
        SerializedPyObj result = serializePyObject(future.value());
        return withStorages(
            PythonRRefFetchRet(std::move(result).toIValues()).toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

// 处理 RRef 删除请求，释放 Python 对象的 GIL
void RequestCallbackImpl::handleRRefDelete(
    c10::intrusive_ptr<RRef>& rref) const {
  // 如果 RRef 存在且是 Python 对象，则获取全局解释器锁
  if (rref && rref->isPyObj()) {
    py::gil_scoped_acquire acquire;
    // 重置 RRef 对象
    rref.reset();
  }
}

// 处理带错误的 RPC 请求，捕获 Python 异常并处理
c10::intrusive_ptr<JitFuture> RequestCallbackImpl::processRpcWithErrors(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    std::vector<c10::Stream> streams) const {
  try {
    // 处理 RPC 请求，返回异步执行的未来对象
    return processRpc(rpc, messageType, std::move(streams));
  } catch (py::error_already_set& e) {
    // 捕获 Python 异常，生成错误处理的未来对象
    // 因为 Python 异常可能会被抛出，释放 Python 异常对象需要持有 GIL
    auto future = asFuture(handleError(e, messageType, -1));
    py::gil_scoped_acquire acquire;
    e.restore(); // 释放对 py::objects 的所有权，并且恢复 Python 错误指示器状态
                 // Release ownership on py::objects and also restore
    PyErr_Clear(); // 清除 Python 错误指示器，因为我们已经在响应消息中记录了异常
                   // Clear the Python Error Indicator as we have
                   // recorded the exception in the response message.
    return future;
  } catch (std::exception& e) {
    // 传递一个虚拟的消息 ID，因为它无论如何都会被覆盖
    // Pass a dummy message ID since it will be overwritten anyways.
    return asFuture(handleError(e, messageType, -1));
  }
}

// 结束 RequestCallbackImpl 类的实现

bool RequestCallbackImpl::cudaAvailable() const {
#ifdef USE_CUDA
  // 如果定义了 USE_CUDA 宏，则 CUDA 可用，返回 true
  return true;
#else
  // 否则 CUDA 不可用，返回 false
  return false;
#endif
}

// 处理 RRef 的反向传播请求并返回一个 JitFuture 对象
c10::intrusive_ptr<JitFuture> RequestCallbackImpl::processRRefBackward(
    RpcCommandBase& rpc) const {
  auto& rrefBackwardReq = static_cast<RRefBackwardReq&>(rpc);

  // 获取对应的所有者 RRef 的未来对象
  auto future = retrieveOwnerRRef(rrefBackwardReq.getRRefId());

  // 返回一个新的 future，将其连接到一个 lambda 函数，用于处理反向传播
  return future->then(
      [autogradContextId = rrefBackwardReq.getAutogradContextId(),
       retainGraph = rrefBackwardReq.retainGraph()](JitFuture& future) {
        // 执行反向传播操作（TODO: 可以将其改为异步执行？）
        PyRRef::backwardOwnerRRef(
            autogradContextId, retainGraph, future.value());

        // 返回包含 storages 的 RRefBackwardResp 的消息
        return withStorages(RRefBackwardResp().toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

// 运行指定名称的 JIT 函数，返回一个 JitFuture 对象
c10::intrusive_ptr<JitFuture> RequestCallbackImpl::runJitFunction(
    const c10::QualifiedName& name,
    std::vector<at::IValue>& stack,
    std::vector<c10::Stream> streams,
    bool isAsyncExecution) const {
  c10::MultiStreamGuard guard(streams);
  c10::intrusive_ptr<JitFuture> future;
  try {
    // 调用 runAsync() 启动 JIT 函数执行，返回未完成的 future
    // 对于非异步代码，通常会立即完成
    future = PythonRpcHandler::getInstance()
                 .jitCompilationUnit()
                 ->get_function(name)
                 .runAsync(stack);
  } catch (const std::exception&) {
    // 捕获异常并返回作为 future 对象
    return asFuture(std::current_exception());
  }

  if (isAsyncExecution) {
    // 如果是异步执行，则需要确保返回的类型是 Future 类型的 IValue
    at::TypePtr type = future->elementType();
    if (type->kind() != at::FutureType::Kind) {
      // 如果不是 Future 类型，则抛出异常
      return asFuture(std::make_exception_ptr(std::runtime_error(c10::str(
          "Async functions must return an IValue of Future type, but got ",
          type->str()))));
    }
    // 将 future 转换为其包含的内部 Future 类型
    future = future->thenAsync(
        [](JitFuture& future) { return future.value().toFuture(); },
        type->cast<at::FutureType>()->getElementType());
  }

  // 返回 JIT 函数的 future 对象
  return future;
}

// 结束命名空间 rpc、distributed、torch 的定义
} // namespace rpc
} // namespace distributed
} // namespace torch
```