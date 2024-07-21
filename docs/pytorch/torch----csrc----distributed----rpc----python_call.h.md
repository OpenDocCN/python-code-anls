# `.\pytorch\torch\csrc\distributed\rpc\python_call.h`

```py
#pragma once

# 使用 `#pragma once` 预处理指令，确保头文件只被编译一次，防止重复包含。


#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>

# 包含其他头文件，以便在编译时可以访问 `rpc_command_base.h` 和 `types.h` 的定义。


namespace torch {
namespace distributed {
namespace rpc {

# 定义命名空间 `torch::distributed::rpc`，用于组织和限定代码的作用域。


// RPC call representing calling a Python function over RPC.
class TORCH_API PythonCall final : public RpcCommandBase {

# 定义 `PythonCall` 类，表示通过 RPC 调用 Python 函数的 RPC 调用。


 public:
  PythonCall(SerializedPyObj&& serializedPyObj, bool isAsyncExecution);

  c10::intrusive_ptr<Message> toMessageImpl() && override;

  static std::unique_ptr<PythonCall> fromMessage(const Message& message);

  const SerializedPyObj& serializedPyObj() const;

  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

# 公共部分开始，声明构造函数 `PythonCall`，成员函数 `toMessageImpl` 和 `fromMessage`，以及访问 `serializedPyObj` 和 `isAsyncExecution` 的方法。


 private:
  SerializedPyObj serializedPyObj_;
  const bool isAsyncExecution_;
};

# 私有部分开始，存储 `serializedPyObj_` 和 `isAsyncExecution_` 成员变量。


} // namespace rpc
} // namespace distributed
} // namespace torch

# 命名空间结束，分别结束 `rpc`、`distributed` 和 `torch` 命名空间的定义。
```