# `.\pytorch\torch\csrc\distributed\rpc\script_resp.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次，以避免重复定义错误

#include <torch/csrc/distributed/rpc/message.h>
// 包含 Torch 分布式 RPC 的消息定义头文件

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
// 包含 Torch 分布式 RPC 的 RPC 命令基类定义头文件

#include <torch/csrc/jit/serialization/pickler.h>
// 包含 Torch JIT 序列化的 Pickler 头文件

namespace torch {
namespace distributed {
namespace rpc {

// 表示内置运算符或 TorchScript 函数的返回值。
class TORCH_API ScriptResp final : public RpcCommandBase {
 public:
  explicit ScriptResp(at::IValue&& values);
  // 构造函数，接收移动语义的 at::IValue 参数

  const at::IValue& value();
  // 返回存储的 at::IValue 对象的引用

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  // 转换为消息的实现方法，使用右值引用语义

  static std::unique_ptr<ScriptResp> fromMessage(const Message& message);
  // 从消息创建 ScriptResp 对象的静态方法

 private:
  const at::IValue value_;
  // 存储的 at::IValue 对象
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```