# `.\pytorch\torch\csrc\distributed\rpc\python_functions.h`

```
// 防止头文件重复包含，只包含一次
#pragma once

// 包含必要的头文件
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

// torch 命名空间
namespace torch {
// 分布式命名空间
namespace distributed {
// RPC 命名空间
namespace rpc {

// 将内部的 ivalue::Future 转换为用户可见的 ivalue::Future<py::object> 类型，
// 创建一个新的 ivalue::Future 并在给定的 ivalue::Future 上调用 markCompleted 作为回调。
// 如果 hasValue 为 true，则将 Message 转换为 py::object 并用 IValue 包装。
// 如果 hasValue 为 false，则此 ivalue::Future 仅用于信号传递和启动回调。
// 在这种情况下，将丢弃消息并使用空的 IValue 或给定的 FutureError 设置 ivalue::Future。
c10::intrusive_ptr<JitFuture> toPyJitFuture(
    const c10::intrusive_ptr<JitFuture>& messageJitFuture,
    bool hasValue = true);

// 发送内置 RPC 操作请求，返回一个 ivalue::Future，用于异步等待结果。
c10::intrusive_ptr<JitFuture> pyRpcBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs,
    const float rpcTimeoutSeconds);

// 发送 Python 用户定义函数的 RPC 请求，返回一个 ivalue::Future，用于异步等待结果。
c10::intrusive_ptr<JitFuture> pyRpcPythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution);

// 发送 TorchScript 函数的 RPC 请求，返回一个 ivalue::Future，用于异步等待结果。
c10::intrusive_ptr<JitFuture> pyRpcTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const py::tuple& argsTuple,
    const py::dict& kwargsDict,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution);

// 发送内置 RPC 操作请求并返回一个远程引用对象 PyRRef。
PyRRef pyRemoteBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const float rpcTimeoutSeconds,
    const py::args& args,
    const py::kwargs& kwargs);

// 发送 Python 用户定义函数的 RPC 请求并返回一个远程引用对象 PyRRef。
PyRRef pyRemotePythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution);

// 发送 TorchScript 函数的 RPC 请求并返回一个远程引用对象 PyRRef。
PyRRef pyRemoteTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution,
    const py::args& args,
    const py::kwargs& kwargs);

} // namespace rpc
} // namespace distributed
} // namespace torch
```