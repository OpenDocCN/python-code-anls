# `.\pytorch\torch\csrc\distributed\rpc\rref_proto.h`

```py
#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

// 用于暂时解决RRef操作的基类消息。
// TODO: 移除所有这些消息，并改用rpc + 注册函数。
class TORCH_API RRefMessageBase : public RpcCommandBase {
 public:
  // 构造函数，初始化RRef的ID和消息类型
  RRefMessageBase(const RRefId& rrefId, MessageType type)
      : rrefId_(rrefId), type_(type) {}

  // 获取RRef的ID
  const RRefId& rrefId();

 protected:
  // RRef的ID，使用NOLINTNEXTLINE来禁止Lint检查非私有成员变量
  const RRefId rrefId_;
  // 消息类型，使用NOLINTNEXTLINE来禁止Lint检查非私有成员变量
  const MessageType type_;
};

// ForkMessageBase继承自RRefMessageBase，用于处理Fork操作的消息
class TORCH_API ForkMessageBase : public RRefMessageBase {
 public:
  // 构造函数，初始化RRef的ID、Fork的ID和消息类型
  ForkMessageBase(const RRefId& rrefId, const ForkId& forkId, MessageType type)
      : RRefMessageBase(rrefId, type), forkId_(forkId) {}

  // 获取Fork的ID
  const ForkId& forkId();

  // 重载toMessageImpl方法，将对象转换为Message
  c10::intrusive_ptr<Message> toMessageImpl() && override;

  // 从Message中反序列化得到RRef ID和Fork ID
  static std::pair<RRefId, ForkId> fromMessage(
      const Message& message,
      MessageType type);

 protected:
  // Fork的ID，使用NOLINTNEXTLINE来禁止Lint检查非私有成员变量
  const ForkId forkId_;
};

// ScriptRRefFetchCall用于从所有者处获取远程RRef值的消息
class TORCH_API ScriptRRefFetchCall final : public RRefMessageBase {
 public:
  // 构造函数，初始化请求来源的worker ID和RRef的ID
  ScriptRRefFetchCall(worker_id_t fromWorkerId, const RRefId& rrefId)
      : RRefMessageBase(rrefId, MessageType::SCRIPT_RREF_FETCH_CALL),
        fromWorkerId_(fromWorkerId) {}

  // 获取请求来源的worker ID
  inline worker_id_t fromWorkerId() const {
    return fromWorkerId_;
  }

  // 重载toMessageImpl方法，将对象转换为Message
  c10::intrusive_ptr<Message> toMessageImpl() && override;

  // 从Message中反序列化得到ScriptRRefFetchCall对象
  static std::unique_ptr<ScriptRRefFetchCall> fromMessage(
      const Message& message);

 private:
  // 请求来源的worker ID，使用NOLINTNEXTLINE来禁止Lint检查非私有成员变量
  const worker_id_t fromWorkerId_;
};

// PythonRRefFetchCall用于从所有者处获取远程Python RRef值的消息
class TORCH_API PythonRRefFetchCall final : public RRefMessageBase {
 public:
  // 构造函数，初始化请求来源的worker ID和RRef的ID
  PythonRRefFetchCall(worker_id_t fromWorkerId, const RRefId& rrefId)
      : RRefMessageBase(rrefId, MessageType::PYTHON_RREF_FETCH_CALL),
        fromWorkerId_(fromWorkerId) {}

  // 重载toMessageImpl方法，将对象转换为Message
  c10::intrusive_ptr<Message> toMessageImpl() && override;

  // 从Message中反序列化得到PythonRRefFetchCall对象
  static std::unique_ptr<PythonRRefFetchCall> fromMessage(
      const Message& message);

 private:
  // 请求来源的worker ID，使用NOLINTNEXTLINE来禁止Lint检查非私有成员变量
  const worker_id_t fromWorkerId_;
};

// OwnerRRef使用此消息向远程UserRRef发送RRef值
class TORCH_API RRefFetchRet : public RpcCommandBase {
 public:
  // 构造函数，初始化RRef的值和消息类型
  RRefFetchRet(std::vector<at::IValue> values, MessageType type)
      : values_(std::move(values)), type_(type) {}

  // 获取RRef的值
  const std::vector<at::IValue>& values();

  // 重载toMessageImpl方法，将对象转换为Message
  c10::intrusive_ptr<Message> toMessageImpl() && override;

 private:
  // RRef的值，使用NOLINTNEXTLINE来禁止Lint检查非私有成员变量
  std::vector<at::IValue> values_;
  // 消息类型，使用NOLINTNEXTLINE来禁止Lint检查非私有成员变量
  const MessageType type_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
// ScriptRRefFetchRet 类，继承自 RRefFetchRet 类，用于处理脚本类型的远程引用（RRef）获取返回
class TORCH_API ScriptRRefFetchRet final : public RRefFetchRet {
 public:
  // 构造函数，接受一个包含 at::IValue 的向量作为参数，调用基类构造函数初始化数据成员
  explicit ScriptRRefFetchRet(std::vector<at::IValue> values)
      : RRefFetchRet(std::move(values), MessageType::SCRIPT_RREF_FETCH_RET) {}

  // 从 Message 对象中解析得到 ScriptRRefFetchRet 对象的静态方法
  static std::unique_ptr<ScriptRRefFetchRet> fromMessage(
      const Message& message);
};

// PythonRRefFetchRet 类，继承自 RRefFetchRet 类，用于处理 Python 类型的远程引用（RRef）获取返回
class TORCH_API PythonRRefFetchRet final : public RRefFetchRet {
 public:
  // 构造函数，接受一个包含 at::IValue 的向量作为参数，调用基类构造函数初始化数据成员
  explicit PythonRRefFetchRet(std::vector<at::IValue> values)
      : RRefFetchRet(std::move(values), MessageType::PYTHON_RREF_FETCH_RET) {}

  // 从 Message 对象中解析得到 PythonRRefFetchRet 对象的静态方法
  static std::unique_ptr<PythonRRefFetchRet> fromMessage(
      const Message& message);
};

// RRefUserDelete 类，继承自 ForkMessageBase 类，用于通知 OwnerRRef 在删除时的用户引用情况
class TORCH_API RRefUserDelete final : public ForkMessageBase {
 public:
  // 构造函数，接受 RRefId 和 ForkId 作为参数，调用基类构造函数初始化数据成员
  RRefUserDelete(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_USER_DELETE) {}

  // 从 Message 对象中解析得到 RRefUserDelete 对象的静态方法
  static std::unique_ptr<RRefUserDelete> fromMessage(const Message& message);
};

// RemoteRet 类，继承自 ForkMessageBase 类，用于远程返回消息的处理
class TORCH_API RemoteRet final : public ForkMessageBase {
 public:
  // 构造函数，接受 RRefId 和 ForkId 作为参数，调用基类构造函数初始化数据成员
  RemoteRet(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::REMOTE_RET) {}

  // 从 Message 对象中解析得到 RemoteRet 对象的静态方法
  static std::unique_ptr<RemoteRet> fromMessage(const Message& message);
};

// RRefChildAccept 类，继承自 RpcCommandBase 类，用于子 RRef 向其父节点发送接受确认消息
class TORCH_API RRefChildAccept final : public RpcCommandBase {
 public:
  // 构造函数，接受 ForkId 作为参数，初始化数据成员
  explicit RRefChildAccept(const ForkId& forkId) : forkId_(forkId) {}

  // 返回 forkId 数据成员的引用
  const ForkId& forkId() const;

  // 实现 RpcCommandBase 中的虚函数，返回一个 Message 消息对象的智能指针
  c10::intrusive_ptr<Message> toMessageImpl() && override;
  
  // 从 Message 对象中解析得到 RRefChildAccept 对象的静态方法
  static std::unique_ptr<RRefChildAccept> fromMessage(const Message& message);

 private:
  const ForkId forkId_; // 存储 ForkId 的常量引用成员变量
};

// RRefForkRequest 类，继承自 ForkMessageBase 类，用于子 RRef 向所有者发送分叉请求消息
class TORCH_API RRefForkRequest final : public ForkMessageBase {
 public:
  // 构造函数，接受 RRefId 和 ForkId 作为参数，调用基类构造函数初始化数据成员
  RRefForkRequest(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_FORK_REQUEST) {}

  // 从 Message 对象中解析得到 RRefForkRequest 对象的静态方法
  static std::unique_ptr<RRefForkRequest> fromMessage(const Message& message);
};

// RRefAck 类，继承自 RpcCommandBase 类，用于远程引用确认消息的处理
class TORCH_API RRefAck final : public RpcCommandBase {
 public:
  RRefAck() = default; // 默认构造函数

  // 实现 RpcCommandBase 中的虚函数，返回一个 Message 消息对象的智能指针
  c10::intrusive_ptr<Message> toMessageImpl() && override;

  // 从 Message 对象中解析得到 RRefAck 对象的静态方法
  static std::unique_ptr<RRefAck> fromMessage(const Message& message);
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```