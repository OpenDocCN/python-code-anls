# `.\pytorch\torch\csrc\distributed\rpc\rpc_command_base.h`

```py
// 预处理命令，指定此头文件只包含一次
#pragma once

// 包含消息相关的头文件
#include <torch/csrc/distributed/rpc/message.h>
// 包含 RPC 类型相关的头文件
#include <torch/csrc/distributed/rpc/types.h>

// 命名空间 torch 中的分布式命名空间 distributed 中的 RPC 命名空间
namespace torch {
namespace distributed {
namespace rpc {

// 所有 RPC 请求和响应的基类
class RpcCommandBase {
 public:
  // 必须重写以将 RPC 序列化为消息。此方法应当具有移动语义，因此使用 && 标记。
  c10::intrusive_ptr<Message> toMessage() && {
    // 创建 JitRRefPickleGuard 对象，用于管理 JitRRef 的序列化
    JitRRefPickleGuard jitPickleGuard;
    // 将当前对象以移动方式转换为消息对象并返回
    return std::move(*this).toMessageImpl();
  }
  // 纯虚函数，子类必须实现以完成将对象转换为消息的操作
  virtual c10::intrusive_ptr<Message> toMessageImpl() && = 0;
  // 虚析构函数，使得该类可以作为基类使用
  virtual ~RpcCommandBase() = 0;
};

// 定义虚析构函数的默认实现
inline RpcCommandBase::~RpcCommandBase() = default;

} // namespace rpc
} // namespace distributed
} // namespace torch
```