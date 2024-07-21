# `.\pytorch\torch\csrc\distributed\rpc\message.cpp`

```py
//`
// 包含 Torch 分布式 RPC 消息定义的头文件
#include <torch/csrc/distributed/rpc/message.h>
// 包含 Torch 自定义类的头文件
#include <torch/custom_class.h>

// Torch 分布式 RPC 的命名空间
namespace torch {
namespace distributed {
namespace rpc {

// 默认构造函数的实现
Message::Message() = default;

// 带参数的构造函数，初始化消息的 payload、tensors 和 type
Message::Message(
    std::vector<char>&& payload,               // 移动语义的消息内容
    std::vector<torch::Tensor>&& tensors,      // 移动语义的张量数组
    MessageType type)
    : payload_(std::move(payload)),            // 移动赋值消息内容
      tensors_(std::move(tensors)),            // 移动赋值张量数组
      type_(type) {}                           // 初始化消息类型

// 带参数的构造函数，初始化消息的 payload、tensors、type 和 id
Message::Message(
    std::vector<char>&& payload,               // 移动语义的消息内容
    std::vector<torch::Tensor>&& tensors,      // 移动语义的张量数组
    MessageType type,
    int64_t id)
    : payload_(std::move(payload)),            // 移动赋值消息内容
      tensors_(std::move(tensors)),            // 移动赋值张量数组
      type_(type),                             // 初始化消息类型
      id_(id) {}                               // 初始化消息 id

// 移动语义版本的 payload 访问方法
std::vector<char>&& Message::movePayload() && {
  return std::move(payload_);
}

// payload 的访问方法
std::vector<char>& Message::payload() {
  return payload_;
}

// const 版本的 payload 访问方法
const std::vector<char>& Message::payload() const {
  return payload_;
}

// 移动语义版本的 tensors 访问方法
std::vector<torch::Tensor>&& Message::moveTensors() && {
  return std::move(tensors_);
}

// tensors 的访问方法
std::vector<torch::Tensor>& Message::tensors() {
  return tensors_;
}

// const 版本的 tensors 访问方法
const std::vector<torch::Tensor>& Message::tensors() const {
  return tensors_;
}

// 返回消息类型的方法
MessageType Message::type() const {
  return type_;
}

// 判断消息是否为请求类型的方法
bool Message::isRequest() const {
  return MessageTypeFlags::REQUEST_TYPE & type_;
}

// 判断消息是否为响应类型的方法
bool Message::isResponse() const {
  return MessageTypeFlags::RESPONSE_TYPE & type_;
}

// 返回消息 id 的方法
int64_t Message::id() const {
  return id_;
}

// 设置消息 id 的方法
void Message::setId(int64_t id) {
  id_ = id;
}

// 获取消息中所有存储的方法
std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> Message::getStorages()
    const {
  // 稀疏张量不直接包含存储，而是包含两个张量 indices 和 values，它们都包含存储
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> storages;
  storages.reserve(2 * tensors_.size());
  for (const auto& tensor : tensors_) {
    if (tensor.is_sparse()) {
      storages.emplace_back(tensor._indices().storage().getWeakStorageImpl());
      storages.emplace_back(tensor._values().storage().getWeakStorageImpl());
    } else {
      storages.emplace_back(tensor.storage().getWeakStorageImpl());
    }
  }
  return storages;
}

// 创建异常响应消息的辅助函数，接收异常对象和消息 id
c10::intrusive_ptr<Message> createExceptionResponse(
    const std::exception& e,
    int64_t id) {
  return createExceptionResponse(e.what(), id);
}

// 创建异常响应消息的辅助名空间，用于注册 Message 类，以便在 c10::getCustomClassTypeMap() 返回的映射中使用
namespace {

// 注册 Message 类，使其可以在 IValue 中使用
// 注意：此行在此处添加，而不是在 rpc/init.cpp 中添加的原因是：
// 1) 我们有仅运行 C++ 的测试，不会运行 rpc/init.cpp；
// 2) Message 不打算从 Python 可见。
static const auto message = torch::class_<Message>("rpc", "_Message");

} // namespace

} // namespace rpc
} // namespace distributed
} // namespace torch
} // namespace rpc
} // namespace distributed
} // namespace torch
```