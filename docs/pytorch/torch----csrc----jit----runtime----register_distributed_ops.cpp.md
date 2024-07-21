# `.\pytorch\torch\csrc\jit\runtime\register_distributed_ops.cpp`

```py
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含操作注册的头文件
#include <ATen/core/op_registration/op_registration.h>
// 包含分布式自动求导的头文件
#include <torch/csrc/distributed/autograd/autograd.h>
// 包含分布式自动求导上下文容器的头文件
#include <torch/csrc/distributed/autograd/context/container.h>
// 包含分布式引擎的头文件
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
// 包含 RPC 代理的头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>
// 包含远程引用的实现头文件
#include <torch/csrc/distributed/rpc/rref_impl.h>
// 包含 TorchScript 函数的头文件
#include <torch/csrc/distributed/rpc/torchscript_functions.h>
// 包含 Pybind 的 Jit 运行时绑定工具的头文件
#include <torch/csrc/jit/python/pybind_utils.h>
// 包含运行时操作注册工具的头文件
#include <torch/csrc/jit/runtime/register_ops_utils.h>
// 包含 Torch 库的头文件
#include <torch/library.h>

// 包含格式化输出库的头文件
#include <fmt/format.h>
// 包含标准异常的头文件
#include <stdexcept>

// 定义命名空间别名
namespace dist_autograd = torch::distributed::autograd;
namespace dist_rpc = torch::distributed::rpc;

// Torch JIT 模块的命名空间
namespace torch::jit {

// 匿名命名空间，用于实现 RPC 调用的准备和执行
namespace {
// 一次性注册 RPC WorkerInfo
distributed::rpc::RegisterWorkerInfoOnce workerInfo{};

// 准备 RPC 输入参数并调用 C++ 实现
void prepare_and_call_rpc_op(
    Stack& stack,                       // 函数调用堆栈
    int num_inputs,                     // 输入参数数量
    const std::string& rpc_op) {        // RPC 操作名称

  // 从堆栈中获取输入
  auto stackIter = stack.end() - num_inputs;
  auto& dstWorkerIValue = *stackIter++;
  auto& qualifiedNameIValue = *stackIter++;
  
  // 创建空元组和空字典作为默认值
  IValue emptyTuple(c10::ivalue::Tuple::create({}));
  IValue emptyDict{c10::impl::GenericDict(AnyType::get(), AnyType::get())};
  
  // 如果有传入参数，则使用传入的参数，否则使用空元组
  auto& argsTupleIValue = num_inputs >= 3 ? *stackIter++ : emptyTuple;
  // 如果有传入关键字参数，则使用传入的关键字参数，否则使用空字典
  auto& kwargsDictIValue = num_inputs >= 4 ? *stackIter++ : emptyDict;

  // RPC 超时的占位符 IValue
  IValue noTimeout(torch::distributed::rpc::kUnsetRpcTimeout);
  const auto rpcMaxInputs = 5;
  // 如果传入了超过5个参数，则使用传入的超时值，否则使用占位符
  auto& timeoutIValue = num_inputs >= rpcMaxInputs ? *stackIter++ : noTimeout;
  
  // 断言保证参数类型正确
  TORCH_INTERNAL_ASSERT(
      dstWorkerIValue.isString() ||
      c10::getCustomClassType<c10::intrusive_ptr<dist_rpc::WorkerInfo>>() ==
          dstWorkerIValue.type());
  TORCH_INTERNAL_ASSERT(qualifiedNameIValue.isString());
  TORCH_INTERNAL_ASSERT(argsTupleIValue.isTuple());
  TORCH_INTERNAL_ASSERT(kwargsDictIValue.isGenericDict());
  TORCH_INTERNAL_ASSERT(timeoutIValue.isDouble());

  // 获取函数的 FunctionSchema
  auto qualifiedName = c10::QualifiedName(qualifiedNameIValue.toStringRef());
  std::shared_ptr<CompilationUnit> cuPtr;
  {
    py::gil_scoped_acquire acquire;
    cuPtr = get_python_cu();
  }
  auto& functionSchema = cuPtr->get_function(qualifiedName).getSchema();

  // 准备用户可调用函数的堆栈
  Stack userCallableStack;
  userCallableStack.reserve(functionSchema.arguments().size());

  // 将元组 IValue 中的参数移动到堆栈中
  for (auto& elem : argsTupleIValue.toTupleRef().elements()) {
    // 将元素推送到用户可调用堆栈中
    push(userCallableStack, std::move(elem));
  }

  // 将 kwargs 从 Dict IValue 移动到堆栈中
  size_t consumed_kwargs = 0;
  auto kwargsDict = kwargsDictIValue.toGenericDict();
  for (size_t i = userCallableStack.size();
       i < functionSchema.arguments().size();
       ++i) {
    const auto& arg = functionSchema.arguments()[i];
    const auto& argName = arg.name();
    // 检查 kwargsDict 是否包含当前参数名 argName
    if (kwargsDict.contains(argName)) {
      // 将参数值推送到用户可调用堆栈中
      push(userCallableStack, kwargsDict.at(argName));
      consumed_kwargs += 1;
    } else if (arg.default_value()) {
      // 如果参数有默认值，则将默认值推送到用户可调用堆栈中
      push(userCallableStack, *arg.default_value());
    } else {
      // 抛出运行时错误，指示缺少参数值
      throw std::runtime_error(c10::str(
          functionSchema.name(),
          "() is missing value for argument '",
          argName,
          "'. Declaration: ",
          functionSchema));
    }
  }
  // 如果未消耗完所有的 kwargs，则抛出异常显示未预期的 kwargs
  if (consumed_kwargs != kwargsDict.size()) {
    // 收集未消耗的 kwargs 名称
    std::vector<std::string> names;
    for (const auto& entry : kwargsDict) {
      const IValue& keyIValue = entry.key();
      const string& keyStr = keyIValue.toStringRef();
      names.emplace_back(keyStr);
    }
    // 抛出运行时错误，显示在 kwargs 中找到的错误
    throw std::runtime_error(functionSchema.findErrorInKwargs(names));
  }

  // 获取目标 WorkerName
  std::string dstWorkerNameStr;
  if (dstWorkerIValue.isString()) {
    // 如果目标 WorkerName 是字符串，则复制该字符串
    dstWorkerNameStr = dstWorkerIValue.toStringRef();
  } else {
    // 否则，获取自定义类中的 WorkerName
    dstWorkerNameStr =
        dstWorkerIValue.toCustomClass<dist_rpc::WorkerInfo>()->name_;
  }
  // 获取 RPC 超时时间（如果用户指定了）
  const auto rpcTimeout = timeoutIValue.toDouble();

  if (rpc_op == "rpc_async") {
    // 发送异步 RPC 请求
    auto futureIValuePtr = dist_rpc::rpcTorchscript(
        dstWorkerNameStr,
        qualifiedName,
        functionSchema,
        userCallableStack,
        rpcTimeout);
    // 丢弃输入并将输出推送到堆栈中
    drop(stack, num_inputs);
    stack.emplace_back(std::move(futureIValuePtr));
  } else if (rpc_op == "rpc_sync") {
    // 发送同步 RPC 请求
    auto futureIValuePtr = dist_rpc::rpcTorchscript(
        dstWorkerNameStr,
        qualifiedName,
        functionSchema,
        userCallableStack,
        rpcTimeout);
    // 等待 RPC 请求完成
    futureIValuePtr->wait();
    if (futureIValuePtr->hasError()) {
      // 如果 RPC 请求发生错误，则抛出异常
      throw std::runtime_error(futureIValuePtr->tryRetrieveErrorMessage());
    } else {
      // 否则获取 RPC 返回值并推送到堆栈中
      auto res = futureIValuePtr->value();
      drop(stack, num_inputs);
      stack.emplace_back(std::move(res));
    }
  } else if (rpc_op == "rpc_remote") {
    // 发送远程 RPC 请求
    auto rrefPtr = dist_rpc::remoteTorchscript(
        dstWorkerNameStr,
        qualifiedName,
        functionSchema,
        userCallableStack,
        rpcTimeout);
    // 将 RRef 接口的静态指针推送到堆栈中
    drop(stack, num_inputs);
    stack.emplace_back(
        c10::static_intrusive_pointer_cast<c10::RRefInterface>(rrefPtr));
  } else {
    # 抛出异常，指示在 TorchScript 中不支持特定的 RPC 操作
    throw std::runtime_error(
        c10::str(rpc_op, "() is not supported in TorchScript!'"));
  }
}

RegisterOperators reg_rpc_ops(
// 在 torch/csrc/jit/runtime/register_distributed_ops.cpp 中实现的操作注册
// Registering operators implemented in
// torch/csrc/jit/runtime/register_distributed_ops.cpp
TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
    // 注册 "get_gradients" 操作，返回一个 lambda 函数
    m.impl("get_gradients", [](int64_t context_id) {
        // 获取分布式自动求导容器中指定上下文 ID 的自动求导上下文
        const auto& autogradContext =
            dist_autograd::DistAutogradContainer::getInstance().retrieveContext(
                context_id);
        // 返回上下文中的梯度信息
        return autogradContext->getGradients();
    });
}

} // namespace
} // namespace torch::jit
```