# `.\pytorch\torch\csrc\distributed\rpc\python_call.cpp`

```
// 包含 Torch 分布式 RPC 模块的 Python 调用头文件
#include <torch/csrc/distributed/rpc/python_call.h>

// Torch 分布式 RPC 的命名空间
namespace torch {
namespace distributed {
namespace rpc {

// PythonCall 类的构造函数，接受一个 SerializedPyObj 对象和一个是否异步执行的标志
PythonCall::PythonCall(SerializedPyObj&& serializedPyObj, bool isAsyncExecution)
    : serializedPyObj_(std::move(serializedPyObj)),  // 初始化成员变量 serializedPyObj_
      isAsyncExecution_(isAsyncExecution) {}         // 初始化成员变量 isAsyncExecution_

// 转换为 Message 对象的实现，使用移动语义来构造 Message
c10::intrusive_ptr<Message> PythonCall::toMessageImpl() && {
  // 创建用于存储消息负载的向量
  std::vector<char> payload;
  // 预留足够的空间来容纳 serializedPyObj_ 的长度加一（用于存储 isAsyncExecution_ 标志）
  payload.reserve(serializedPyObj_.payload_.length() + 1);
  // 添加 isAsyncExecution_ 的标志（1 表示异步，0 表示同步）
  payload.push_back(isAsyncExecution_ ? 1 : 0);
  // 将 serializedPyObj_ 的数据插入到 payload 后面
  payload.insert(
      payload.end(),
      serializedPyObj_.payload_.begin(),
      serializedPyObj_.payload_.end());

  // 使用 make_intrusive 创建并返回 Message 对象
  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(serializedPyObj_.tensors_),
      MessageType::PYTHON_CALL);
}

// 从 Message 对象构造 PythonCall 对象的静态方法
std::unique_ptr<PythonCall> PythonCall::fromMessage(const Message& message) {
  // 断言消息负载非空
  TORCH_INTERNAL_ASSERT(
      !message.payload().empty(),
      "Failed to convert an RPC message to PythonCall, the payload should at "
      "least contain one byte indicating whether this is an async function, "
      "but got payload of size ",
      message.payload().size());
  // 提取第一个字节作为 isAsyncExecution 标志
  const char& c = message.payload()[0];
  // 断言 isAsyncExecution 的值为 0 或 1
  TORCH_INTERNAL_ASSERT(c == 0 || c == 1);
  bool isAsyncExecution = (c == 1);
  // 提取剩余部分作为 payload 字符串
  std::string payload(message.payload().begin() + 1, message.payload().end());
  // 提取消息中的张量列表
  std::vector<Tensor> tensors = message.tensors();
  // 使用提取的数据构造 SerializedPyObj 对象
  SerializedPyObj serializedPyObj(std::move(payload), std::move(tensors));
  // 使用 make_unique 创建并返回 PythonCall 对象
  return std::make_unique<PythonCall>(
      std::move(serializedPyObj), isAsyncExecution);
}

// 返回 serializedPyObj_ 成员变量的常量引用
const SerializedPyObj& PythonCall::serializedPyObj() const {
  return serializedPyObj_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```