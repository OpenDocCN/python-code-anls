# `.\pytorch\torch\csrc\distributed\rpc\request_callback_impl.h`

```py
#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback_no_python.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/python/pybind.h>

// 声明 torch 命名空间
namespace torch {
// 声明 distributed 命名空间
namespace distributed {
// 声明 rpc 命名空间
namespace rpc {

// 定义 RequestCallbackImpl 类，继承自 RequestCallbackNoPython 类
class TORCH_API RequestCallbackImpl : public RequestCallbackNoPython {
 public:
  // 重写父类方法，反序列化 Python RPC 命令
  std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
      std::unique_ptr<RpcCommandBase> rpc,
      const MessageType& messageType) const override;

  // 处理 Python 调用的方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> processPythonCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const override;

  // 处理脚本调用的方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> processScriptCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const override;

  // 处理远程脚本调用的方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> processScriptRemoteCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const override;

  // 处理远程 Python 调用的方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> processPythonRemoteCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const override;

  // 处理 Python RRef Fetch 调用的方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> processPythonRRefFetchCall(
      RpcCommandBase& rpc) const override;

  // 处理 RRef 删除的方法，没有返回值
  void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const override;

  // 处理带错误的 RPC 调用的方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      std::vector<c10::Stream> streams) const override;

  // 检查当前是否可用 CUDA 设备，返回布尔值
  bool cudaAvailable() const override;

  // 处理 RRef 反向传播的方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> processRRefBackward(
      RpcCommandBase& rpc) const override;

  // 运行 JIT 函数的辅助方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> runJitFunction(
      const c10::QualifiedName& name,
      std::vector<at::IValue>& stack,
      std::vector<c10::Stream> streams,
      bool isAsyncExecution) const;

  // 运行 Python 函数的辅助方法，返回一个 JitFuture 对象指针
  c10::intrusive_ptr<JitFuture> runPythonFunction(
      const py::object& function,
      std::vector<c10::Stream> streams,
      bool isAsyncExecution) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```