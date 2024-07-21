# `.\pytorch\torch\csrc\distributed\rpc\torchscript_functions.h`

```
#pragma once

// 包含必要的头文件：IValue 类型相关、自动微分分析器、分布式自动微分工具、远程引用上下文、脚本远程调用
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>

// 命名空间：torch -> distributed -> rpc
namespace torch {
namespace distributed {
namespace rpc {

// 此函数发送一个 RPC 调用以运行 torchscript 函数，目前支持的 torchscript 函数只能是具有 "@torch.jit.script" 注解的用户定义的 Python 函数。
// torchscript 函数不能是类构造函数、类方法、实例方法或脚本模块。
//   dst: 目标 worker 的名称
//   qualifiedName: torchscript 函数的限定名称字符串，例如 "moduleName::torchscriptFunctionName"，例如 "dist_autograd_test::my_py_add"
//   stack: 传递给 torchscriptFunctionName 的一组 IValue 参数
// 返回值为 c10::intrusive_ptr<ivalue::Future>
c10::intrusive_ptr<c10::ivalue::Future> TORCH_API rpcTorchscript(
    const std::string& dstWorkerName,               // 目标 worker 的名称
    const c10::QualifiedName& qualifiedName,        // torchscript 函数的限定名称
    const c10::FunctionSchema& functionSchema,      // 函数的函数模式
    std::vector<c10::IValue>& stack,                // 传递给函数的参数栈
    const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,  // RPC 超时时间（秒）
    const bool isAsyncExecution = false);           // 是否异步执行的标志

// 此函数发送一个 RPC 调用以远程执行 torchscript 函数。
//   dst: 目标 worker 的名称
//   qualifiedName: torchscript 函数的限定名称字符串，例如 "moduleName::torchscriptFunctionName"，例如 "dist_autograd_test::my_py_add"
//   stack: 传递给 torchscriptFunctionName 的一组 IValue 参数
// 返回值为 c10::intrusive_ptr<RRef>
c10::intrusive_ptr<RRef> TORCH_API remoteTorchscript(
    const std::string& dstWorkerName,               // 目标 worker 的名称
    const c10::QualifiedName& qualifiedName,        // torchscript 函数的限定名称
    const c10::FunctionSchema& functionSchema,      // 函数的函数模式
    std::vector<c10::IValue>& stack,                // 传递给函数的参数栈
    const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,  // RPC 超时时间（秒）
    const bool isAsyncExecution = false);           // 是否异步执行的标志

} // namespace rpc
} // namespace distributed
} // namespace torch
```