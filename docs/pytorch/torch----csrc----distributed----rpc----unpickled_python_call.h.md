# `.\pytorch\torch\csrc\distributed\rpc\unpickled_python_call.h`

```
// 防止头文件重复包含，只在第一次包含时有效
#pragma once

// 引入必要的头文件：RPC 命令基类、RPC 类型定义、Python 绑定工具
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/pybind.h>

// Torch 的命名空间
namespace torch {
// 分布式命名空间
namespace distributed {
// RPC 命名空间
namespace rpc {

// 这个类将 PythonCall 中的内容转换为 py::object。这是一个辅助类，确保所有参数的反序列化
// 在进入 RequestCallbackImpl::processRpc(...) 之前完成，以便反序列化相关逻辑可以在
// 一个地方完成，而不是在多个消息类型的不同地方散布。
// 注意：不将该类合并到 PythonCall 中的原因是 PythonCall 是一个 libtorch 类型，不应该
// 依赖于 Python 类型。
class TORCH_API UnpickledPythonCall : public RpcCommandBase {
 public:
  // 构造函数，从序列化的 Python 对象和是否异步执行标志初始化
  UnpickledPythonCall(
      const SerializedPyObj& serializedPyObj,
      bool isAsyncExecution);
  // 虚析构函数，用于确保派生类的正确资源释放
  ~UnpickledPythonCall() override;

  // toMessage() 方法未实现，因为不应该直接将该类的对象转换为 Message 对象。
  c10::intrusive_ptr<Message> toMessageImpl() && override;
  // 返回 Python 函数对象的引用
  const py::object& pythonUdf() const;

  // 返回是否异步执行的标志
  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

 private:
  // Python 函数对象成员变量
  py::object pythonUdf_;
  // 是否异步执行标志
  const bool isAsyncExecution_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```