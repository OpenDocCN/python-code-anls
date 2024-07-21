# `.\pytorch\torch\csrc\distributed\rpc\python_remote_call.h`

```py
#pragma once

// 包含必要的头文件
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <vector>

// 命名空间 torch 下的 distributed 下的 rpc 下的定义
namespace torch {
namespace distributed {
namespace rpc {

// PythonRemoteCall 类，继承自 RpcCommandBase 类
class TORCH_API PythonRemoteCall : public RpcCommandBase {
 public:
  // 构造函数，接受序列化的 Python 对象、返回的 RRef ID、返回的 Fork ID 和是否异步执行标志
  PythonRemoteCall(
      SerializedPyObj&& serializedPyObj,
      at::IValue retRRefId,
      at::IValue retForkId,
      const bool isAsyncExecution);

  // 返回成员变量 serializedPyObj_
  inline const SerializedPyObj& serializedPyObj() const {
    return serializedPyObj_;
  }

  // 返回成员变量 retRRefId_
  inline const at::IValue& retRRefId() const {
    return retRRefId_;
  }

  // 返回成员变量 retForkId_
  inline const at::IValue& retForkId() const {
    return retForkId_;
  }

  // 返回成员变量 isAsyncExecution_
  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

  // 虚函数，返回消息的具体实现，使用右值引用
  c10::intrusive_ptr<Message> toMessageImpl() && override;

  // 静态方法，从消息中构造 PythonRemoteCall 对象
  static std::unique_ptr<PythonRemoteCall> fromMessage(const Message& message);

 private:
  // 成员变量，序列化的 Python 对象
  SerializedPyObj serializedPyObj_;
  // 成员变量，返回的 RRef ID
  const at::IValue retRRefId_;
  // 成员变量，返回的 Fork ID
  const at::IValue retForkId_;
  // 成员变量，是否异步执行
  const bool isAsyncExecution_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```