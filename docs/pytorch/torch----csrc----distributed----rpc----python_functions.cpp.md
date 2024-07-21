# `.\pytorch\torch\csrc\distributed\rpc\python_functions.cpp`

```py
// 包含必要的头文件以及命名空间声明
#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/torchscript_functions.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/python_compat.h>
#include <exception>

namespace torch {
namespace distributed {
namespace rpc {

// 匿名命名空间，用于定义内部实用函数和数据结构
namespace {

// 将传入消息转换为对应的 Python IValue
IValue toPyIValue(const Message& message) {
  // 获取消息类型
  MessageType msgType = message.type();
  // 反序列化消息，根据消息类型选择合适的响应类进行处理
  auto response = deserializeResponse(message, msgType);
  switch (msgType) {
    case MessageType::SCRIPT_RET: {
      // 处理脚本返回消息
      auto& ret = static_cast<ScriptResp&>(*response);
      Stack stack;
      stack.push_back(ret.value());
      // 需要全局解释器锁(GIL)来保护 createPyObjectForStack() 及其返回的 py::object
      py::gil_scoped_acquire acquire;
      return jit::toIValue(
          torch::jit::createPyObjectForStack(std::move(stack)),
          PyObjectType::get());
    }
    case MessageType::PYTHON_RET: {
      // 处理 Python 返回消息
      // TODO: 尝试避免此处的拷贝
      auto& resp = static_cast<PythonResp&>(*response);
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      // 需要全局解释器锁(GIL)来销毁 deserialize() 返回的 py::object
      py::gil_scoped_acquire acquire;
      py::object value = pythonRpcHandler.deserialize(resp.serializedPyObj());
      pythonRpcHandler.handleException(value);
      return jit::toIValue(value, PyObjectType::get());
    }
    default: {
      // 对于未识别的响应消息类型，抛出异常
      TORCH_CHECK(false, "Unrecognized response message type ", msgType);
    }
  }
}

// 匹配内置运算符，返回与指定名称匹配的运算符
std::shared_ptr<Operator> matchBuiltinOp(
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs,
    Stack& stack) {
  // 根据操作名称创建符号
  Symbol symbol = Symbol::fromQualString(opName);

  std::shared_ptr<jit::Operator> matchedOperator;
  if (symbol.is_aten()) {
    // 首选 C10 操作，以通过 C10 调度执行
    auto ops = torch::jit::getAllOperatorsFor(symbol);
    std::vector<std::shared_ptr<torch::jit::Operator>> c10OpsForSymbol;
    // 遍历 ops 容器中的每个操作符指针
    for (auto it = ops.begin(); it != ops.end();) {
      // 获取当前迭代器指向的操作符指针
      std::shared_ptr<jit::Operator> op = *it;
      // 检查操作符是否为 C10 操作
      if (op->isC10Op()) {
        // 将符合条件的 C10 操作移动到 c10OpsForSymbol 容器中
        c10OpsForSymbol.emplace_back(std::move(op));
        // 从 ops 容器中移除当前操作符，并更新迭代器
        it = ops.erase(it);
      } else {
        // 如果不是 C10 操作，则迭代器向前移动
        ++it;
      }
    }

    // 尝试从 c10OpsForSymbol 容器中获取符合条件的操作符及其堆栈信息
    std::pair<std::shared_ptr<torch::jit::Operator>, torch::jit::Stack>
        opWithStack;
    try {
      opWithStack = torch::jit::getOpWithStack(c10OpsForSymbol, args, kwargs);
    } catch (const std::runtime_error& e) {
      // 如果获取失败，则从 ops 容器中尝试获取操作符及其堆栈信息
      opWithStack = torch::jit::getOpWithStack(ops, args, kwargs);
    }
    // 提取匹配到的操作符和堆栈信息
    matchedOperator = std::get<0>(opWithStack);
    stack = std::get<1>(opWithStack);
  }

  // 如果 matchedOperator 为空指针，则抛出错误信息
  // 这种情况理论上不应该发生，因为前面的 getOpWithStack 调用应该返回一个有效的操作符
  TORCH_CHECK(
      matchedOperator != nullptr,
      "Failed to match operator name ",
      opName,
      " and arguments "
      "(args: ",
      args,
      ", kwargs: ",
      kwargs,
      ") to a builtin operator");

  // 返回匹配到的操作符指针
  return matchedOperator;
} // namespace

using namespace torch::distributed::autograd;

// 定义函数 sendPythonRemoteCall，发送远程 Python 调用请求
c10::intrusive_ptr<JitFuture> sendPythonRemoteCall(
    const WorkerInfo& dst,                    // 目标 Worker 的信息
    SerializedPyObj serializedPyObj,          // 序列化的 Python 对象
    const IValue& rrefId,                     // 引用 ID
    const IValue& forkId,                     // 分支 ID
    const float rpcTimeoutSeconds,            // RPC 超时时间（秒）
    const bool isAsyncExecution) {            // 是否异步执行

  // 创建 PythonRemoteCall 对象，包含序列化对象、引用 ID、分支 ID、异步执行标志
  auto pythonRemoteCall = std::make_unique<PythonRemoteCall>(
      std::move(serializedPyObj), rrefId, forkId, isAsyncExecution);

  // 获取当前 RPC 代理
  auto agent = RpcAgent::getCurrentRpcAgent();

  // 调用 torch 分布式自动求导库发送带自动求导的消息
  return torch::distributed::autograd::sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*pythonRemoteCall).toMessage(),
      true /*forceGradRecording*/,            // 强制记录梯度
      rpcTimeoutSeconds);                     // RPC 超时时间
}

// 使用 torch 分布式自动求导命名空间
// 定义函数 toPyJitFuture，将 JitFuture 转换为 PyJitFuture
c10::intrusive_ptr<JitFuture> toPyJitFuture(
    const c10::intrusive_ptr<JitFuture>& messageJitFuture, // 消息 JitFuture 对象
    bool hasValue) {                        // 是否有值标志

  // 如果有值
  if (hasValue) {
    // 创建子 Future，类型为 PyObjectType
    auto child = messageJitFuture->createInstance(PyObjectType::get());

    // 给消息 JitFuture 添加回调函数
    messageJitFuture->addCallback(
        at::wrapPropagateTLSState([child](JitFuture& future) {
          // 如果未来有错误
          if (future.hasError()) {
            // 设置子 Future 的错误状态
            child->setError(future.exception_ptr());
          } else {
            // 获取消息对象
            const Message& message = *future.value().toCustomClass<Message>();

            // 尝试将消息对象转换为 Python IValue
            IValue ivalue;
            try {
              ivalue = toPyIValue(message);
            } catch (py::error_already_set& e) {
              py::gil_scoped_acquire acquire;

              // 处理特定的 Python 异常类型
              if (e.matches(PyExc_ValueError)) {
                child->setErrorIfNeeded(
                    std::make_exception_ptr(pybind11::value_error(e.what())));
              } else if (e.matches(PyExc_TypeError)) {
                child->setErrorIfNeeded(
                    std::make_exception_ptr(pybind11::type_error(e.what())));
              } else {
                // 处理其他类型的异常
                child->setErrorIfNeeded(
                    std::make_exception_ptr(std::runtime_error(e.what())));
              }
              e.restore();
              PyErr_Clear();
              return;
            } catch (std::exception& e) {
              // 处理 C++ 异常
              child->setErrorIfNeeded(std::current_exception());
              return;
            }

            // 标记子 Future 完成，并传递值和存储器状态
            child->markCompleted(ivalue, future.storages());
          }
        }));

    // 返回子 Future
    return child;
  } else {
    return messageJitFuture->then(
        // 调用 messageJitFuture 的 then 方法，传入一个回调函数
        at::wrapPropagateTLSState([](JitFuture& future) {
          // 匿名函数：接收 JitFuture 的引用 future
          if (future.hasError()) {
            // 如果 future 中包含错误
            std::rethrow_exception(future.exception_ptr());
            // 抛出 future 中存储的异常
          } else {
            // 如果 future 没有错误
            return IValue();
            // 返回空的 IValue 对象
          }
        }),
        NoneType::get());
    // 在 messageJitFuture 上调用 then 方法后返回结果，使用 NoneType::get() 作为默认值
}

c10::intrusive_ptr<JitFuture> pyRpcBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs,
    const float rpcTimeoutSeconds) {
  // 断言当前线程已经获取了 GIL
  DCHECK(PyGILState_Check());
  // 创建一个空的堆栈
  Stack stack;
  // 调用 matchBuiltinOp 函数，匹配内置操作的名称、参数和关键字参数，并将结果保存到 op 变量中
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  // 由于参数和关键字参数的处理已经完成，释放 GIL
  py::gil_scoped_release release;
  // 创建一个 ScriptCall 对象，将操作和堆栈移动到该对象中
  auto scriptCall = std::make_unique<ScriptCall>(op, std::move(stack));
  // 获取当前的 RPC 代理
  auto agent = RpcAgent::getCurrentRpcAgent();
  // 将脚本调用消息发送到指定的远程 Worker，并返回一个 JitFuture 对象
  return toPyJitFuture(sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*scriptCall).toMessage(),
      false,
      rpcTimeoutSeconds));
}

c10::intrusive_ptr<JitFuture> pyRpcPythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  // 断言当前线程未获取 GIL
  DCHECK(!PyGILState_Check());
  // 创建一个 SerializedPyObj 对象，包含序列化的 Python 用户定义函数和张量
  auto serializedPyObj =
      SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors));
  // 创建一个 PythonCall 对象，包含序列化的 Python 对象和是否异步执行标志
  auto pythonCall = std::make_unique<PythonCall>(
      std::move(serializedPyObj), isAsyncExecution);

  // 获取当前的 RPC 代理
  auto agent = RpcAgent::getCurrentRpcAgent();
  // 将 Python 调用消息发送到指定的远程 Worker，并返回一个 JitFuture 对象
  return toPyJitFuture(sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*pythonCall).toMessage(),
      true /*forceGradRecording*/,
      rpcTimeoutSeconds));
}

c10::intrusive_ptr<JitFuture> pyRpcTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const py::tuple& argsTuple,
    const py::dict& kwargsDict,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  // 不需要在此处捕获异常，如果找不到函数，get_function() 调用将抛出异常；
  // 如果参数与函数模式不匹配，createStackForSchema() 调用将抛出异常。
  DCHECK(!PyGILState_Check());
  // 创建一个 QualifiedName 对象
  const c10::QualifiedName qualifiedName(qualifiedNameStr);
  // 获取函数的 schema
  auto functionSchema = PythonRpcHandler::getInstance()
                            .jitCompilationUnit()
                            ->get_function(qualifiedName)
                            .getSchema();
  // 创建一个空的堆栈
  Stack stack;
  {
    // 获取 GIL 以处理 py::args 和 py::kwargs
    py::gil_scoped_acquire acquire;
    // 使用函数 schema、py::args 和 py::kwargs 创建堆栈
    stack = torch::jit::createStackForSchema(
        functionSchema,
        argsTuple.cast<py::args>(),
        kwargsDict.cast<py::kwargs>(),
        c10::nullopt);
  }
  // 再次断言当前线程未获取 GIL
  DCHECK(!PyGILState_Check());
  // 调用 rpcTorchscript 函数，将 TorchScript 调用消息发送到指定的远程 Worker，并返回一个 JitFuture 对象
  c10::intrusive_ptr<c10::ivalue::Future> fut = rpcTorchscript(
      dstWorkerName,
      qualifiedName,
      functionSchema,
      stack,
      rpcTimeoutSeconds,
      isAsyncExecution);
  return fut;
}

PyRRef pyRemoteBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const float rpcTimeoutSeconds,
    const py::args& args,
    // 检查是否已经获取了全局解释器锁（GIL）
    DCHECK(PyGILState_Check());
    // 创建一个栈对象用于操作数据
    Stack stack;
    // 调用 matchBuiltinOp 函数匹配内建操作，并传入参数和关键字参数
    auto op = matchBuiltinOp(opName, args, kwargs, stack);
    // 释放 GIL，因为参数和关键字参数的处理已经完成
    py::gil_scoped_release release;
    // 获取操作的返回类型
    TypePtr returnType = op->schema().returns()[0].type();

    // 获取远程引用上下文和当前的 RPC 代理
    auto& ctx = RRefContext::getInstance();
    auto agent = RpcAgent::getCurrentRpcAgent();

    // 如果当前工作进程的 ID 不等于目标的 ID
    if (ctx.getWorkerId() != dst.id_) {
        // 在上下文中创建一个用户远程引用，并指定返回类型
        auto userRRef = ctx.createUserRRef(dst.id_, returnType);

        // 创建一个脚本远程调用对象，用于发送操作、栈和远程引用的 ID
        auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
            op, std::move(stack), userRRef->rrefId(), userRRef->forkId());

        // 发送带自动求导的消息到目标节点，并获取未来对象
        auto jitFuture = sendMessageWithAutograd(
            *agent,
            dst,
            std::move(*scriptRemoteCall).toMessage(),
            /*forceGradRecord */ false,
            /* timeout */ rpcTimeoutSeconds);

        // 注册用户远程引用的创建未来
        userRRef->registerOwnerCreationFuture(jitFuture);
        // 在上下文中添加待处理的用户远程引用
        ctx.addPendingUser(userRRef->forkId(), userRRef);

        // 添加回调函数，用于确认用户远程引用的创建
        jitFuture->addCallback(at::wrapPropagateTLSState(
            [forkId{userRRef->forkId()}](JitFuture& future) {
                callback::confirmPendingUser(future, forkId);
            }));

        // 返回 Python 级别的远程引用对象
        return PyRRef(userRRef);
    } else {
        // 在上下文中创建一个所有者远程引用，并指定返回类型
        auto ownerRRef = ctx.createOwnerRRef(returnType);
        // 防止这个所有者远程引用因为其他分支的删除而被删除
        ctx.addSelfAsFork(ownerRRef);

        // 创建一个脚本远程调用对象，用于发送操作、栈和远程引用的 ID
        auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
            op, std::move(stack), ownerRRef->rrefId(), ownerRRef->rrefId());
        
        // 发送带自动求导的消息到目标节点，并获取未来对象
        auto jitFuture = sendMessageWithAutograd(
            *agent,
            dst,
            std::move(*scriptRemoteCall).toMessage(),
            /* forceGradRecord */ false,
            /* timeout */ rpcTimeoutSeconds);

        // 注册所有者远程引用的创建未来
        ownerRRef->registerOwnerCreationFuture(jitFuture);
        
        // 添加回调函数，用于完成所有者远程引用的创建
        jitFuture->addCallback(at::wrapPropagateTLSState(
            [ownerRRefId = ownerRRef->rrefId()](JitFuture& future) {
                callback::finishCreatingOwnerRRef(future, ownerRRefId);
            }));

        // 返回 Python 级别的远程引用对象
        return PyRRef(ownerRRef);
    }
} // namespace rpc
} // namespace distributed
} // namespace torch
```