# `.\pytorch\torch\csrc\distributed\rpc\script_remote_call.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/distributed/rpc/script_call.h>
// 包含远程过程调用（RPC）脚本调用的头文件

#include <torch/csrc/distributed/rpc/types.h>
// 包含RPC相关的类型定义的头文件

#include <torch/csrc/jit/runtime/operator.h>
// 包含运算符的头文件，用于JIT运行时

#include <torch/csrc/jit/serialization/pickler.h>
// 包含用于序列化的Pickler头文件

#include <vector>
// 包含标准向量容器的头文件

namespace torch {
namespace distributed {
namespace rpc {

using torch::jit::Operator;
// 使用torch::jit::Operator命名空间

// ScriptRemoteCall类表示对内置运算符进行dist.remote调用的实例。
// 目前，它还不支持将RRef用作参数。
// 除了运算符和参数向量之外，ScriptRemoteCall还包含返回值RRef的RRefId和ForkId。
class TORCH_API ScriptRemoteCall final : public ScriptCall {
 public:
  // 内置运算符调用的构造函数。
  ScriptRemoteCall(
      std::shared_ptr<Operator> op,
      std::vector<at::IValue>&& stack,
      const RRefId& retRRefId,
      const ForkId& retForkId);

  // TorchScript函数调用的构造函数。
  ScriptRemoteCall(
      const c10::QualifiedName& qualifiedName,
      std::vector<at::IValue>&& stack,
      const RRefId& retRRefId,
      const ForkId& retForkId,
      const bool isAsyncExecution);

  // 返回retRRefId_成员的引用。
  inline const RRefId& retRRefId() const {
    return retRRefId_;
  }

  // 返回retForkId_成员的引用。
  inline const ForkId& retForkId() const {
    return retForkId_;
  }

  // 从IValues创建ScriptRemoteCall的静态方法。
  static std::unique_ptr<ScriptRemoteCall> fromIValues(
      std::vector<at::IValue>& ivalues);

  // 实现ScriptCall类的纯虚函数，生成消息对象。
  c10::intrusive_ptr<Message> toMessageImpl() && override;

  // 从消息对象创建ScriptRemoteCall的静态方法。
  static std::unique_ptr<ScriptRemoteCall> fromMessage(const Message& message);

 private:
  const RRefId retRRefId_;  // 返回值RRef的ID
  const ForkId retForkId_;  // 返回值RRef的Fork ID
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```