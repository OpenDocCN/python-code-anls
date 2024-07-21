# `.\pytorch\torch\csrc\distributed\rpc\message.h`

```py
#pragma once
// 使用 `#pragma once` 指令确保头文件只被编译一次

#include <torch/types.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

// 枚举类型，表示常见的 RPC 错误，用于特定的错误处理
enum RPCErrorType {
  UNKNOWN_ERROR = 0, // 表示无法解析的错误类型
  TIMEOUT = 1, // 表示 RPC 已超时
  INTENTIONAL_FAILURE = 2 // 表示有意的失败，如 FaultyAgent 为测试而注入的失败
};

// 枚举类型，这些值与 MessageType 进行按位 OR 操作
// 它们是从 0x100 开始的位标志，应该具有如 0x100、0x200、0x400、0x800、0xF00 等的值
enum MessageTypeFlags {
  REQUEST_TYPE = 0x100, // 请求类型的标志
  RESPONSE_TYPE = 0x200, // 响应类型的标志
};

// 消息类型的值必须在 0x00 到 0xff 之间
# 定义一个枚举类型 MessageType，用于表示消息类型

// messages for dist.rpc on builtin operators
// 表示针对内置运算符的 dist.rpc 消息
SCRIPT_CALL = 0x00 | MessageTypeFlags::REQUEST_TYPE,

// messages for dist.rpc on Python UDF
// 表示针对 Python UDF 的 dist.rpc 消息
SCRIPT_RET = 0x01 | MessageTypeFlags::RESPONSE_TYPE,

// messages for dist.remote on builtin operators and Python UDF
// 表示针对内置运算符和 Python UDF 的 dist.remote 消息
PYTHON_CALL = 0x02 | MessageTypeFlags::REQUEST_TYPE,
PYTHON_RET = 0x03 | MessageTypeFlags::RESPONSE_TYPE,

// RRef related internal messages
// 与 RRef 相关的内部消息
SCRIPT_REMOTE_CALL = 0x04 |
    MessageTypeFlags::REQUEST_TYPE, // 在内置运算符上进行远程调用
PYTHON_REMOTE_CALL =
    0x05 | MessageTypeFlags::REQUEST_TYPE, // 在 Python UDF 上进行远程调用
REMOTE_RET =
    0x06 | MessageTypeFlags::RESPONSE_TYPE, // 对 UDF、内置函数或脚本的远程调用的响应

SCRIPT_RREF_FETCH_CALL =
    0x07 | MessageTypeFlags::REQUEST_TYPE, // UserRRef<IValue> 从所有者获取值的请求
PYTHON_RREF_FETCH_CALL =
    0x08 | MessageTypeFlags::REQUEST_TYPE, // UserRRef<py::object> 从所有者获取值的请求
SCRIPT_RREF_FETCH_RET = 0x09 |
    MessageTypeFlags::RESPONSE_TYPE, // 所有者发送 IValue 到用户的响应
PYTHON_RREF_FETCH_RET = 0x0a |
    MessageTypeFlags::RESPONSE_TYPE, // 所有者发送 py::object 到用户的响应
RREF_USER_DELETE = 0x0b |
    MessageTypeFlags::REQUEST_TYPE, // 用户告诉所有者解除引用的请求
RREF_FORK_REQUEST =
    0x0c | MessageTypeFlags::REQUEST_TYPE, // 子 UserRRef 告知所有者关于自身的请求
RREF_CHILD_ACCEPT =
    0x0d | MessageTypeFlags::REQUEST_TYPE, // 子 UserRRef 告知父进程所有者知道它的请求
RREF_ACK =
    0x0e | MessageTypeFlags::RESPONSE_TYPE, // 内部 RRef 消息的确认

// Messages with autograd info
// 具有自动求导信息的消息
FORWARD_AUTOGRAD_REQ = 0x0f | MessageTypeFlags::REQUEST_TYPE,
FORWARD_AUTOGRAD_RESP = 0x10 | MessageTypeFlags::RESPONSE_TYPE,

// Messages to propagate gradients on the backward pass.
// 用于在反向传播过程中传播梯度的消息
BACKWARD_AUTOGRAD_REQ = 0x11 | MessageTypeFlags::REQUEST_TYPE,
BACKWARD_AUTOGRAD_RESP = 0x12 | MessageTypeFlags::RESPONSE_TYPE,

// Messages to tell workers to clean up their autograd context.
// 告诉工作进程清理其自动求导上下文的消息
CLEANUP_AUTOGRAD_CONTEXT_REQ = 0x13 | MessageTypeFlags::REQUEST_TYPE,
CLEANUP_AUTOGRAD_CONTEXT_RESP = 0x14 | MessageTypeFlags::RESPONSE_TYPE,

// Messages that tell workers to run requests with profiling enabled.
// 告诉工作进程使用性能分析运行请求的消息
RUN_WITH_PROFILING_REQ = 0x15 | MessageTypeFlags::REQUEST_TYPE,
RUN_WITH_PROFILING_RESP = 0x16 | MessageTypeFlags::RESPONSE_TYPE,

// Messages to support RRef.backward().
// 支持 RRef.backward() 的消息
RREF_BACKWARD_REQ = 0x17 | MessageTypeFlags::REQUEST_TYPE,
RREF_BACKWARD_RESP = 0x18 | MessageTypeFlags::RESPONSE_TYPE,

// Other internal message types
// 其他内部消息类型
EXCEPTION = 0x37 | MessageTypeFlags::RESPONSE_TYPE,
UNKNOWN = 0x3c
};

// A message to be sent/received by an RpcAgent.
// 由 RpcAgent 发送/接收的消息
// A Message object contains 4 fields:
//    payload (std::vector<char>): a binary chunk of data.
//    tensors (std::vector<torch::Tensor>): all tensors. Tensor data are not
//        included in the payload, and it is up to the RpcAgent implementation
//        to determine how to serialize them. This design is helpful for
//        communicating super large tensors where serializing all the data at
//        once leads to excessively large memory footprint. An implementation
//        can then serialize and send tensors chunk-by-chunk, in the streaming
//        fashion.
//    type (MessageType): type of the message.
//    id (int64_t): message id, this is used to match request and response.
//               Other implementation can ignore it if they have their own
//               ways to do matching.
//
// Layers above ``RpcAgent`` only converts ScriptCall, ScriptResp, PythonCall,
// and PythonResp into a Message, and it is up to the RpcAgent
// implementation to determine how to serialize a message.
class TORCH_API Message final : public torch::CustomClassHolder {
 private:
  // Keep these private in order to force users to go through make_intrusive and
  // thus prevent creating a Message that's not held by an intrusive_ptr.
  
  // 默认构造函数，私有化以强制用户通过 make_intrusive 创建 Message，
  // 从而防止创建不被 intrusive_ptr 持有的 Message 对象。
  Message();

  // 构造函数，接收一个 payload、tensors 和 type，移动构造 payload 和 tensors 到成员变量中。
  Message(
      std::vector<char>&& payload,
      std::vector<torch::Tensor>&& tensors,
      MessageType type);

  // 构造函数，接收一个 payload、tensors、type 和 id，移动构造 payload 和 tensors 到成员变量中，
  // 同时设置消息的 id。
  Message(
      std::vector<char>&& payload,
      std::vector<torch::Tensor>&& tensors,
      MessageType type,
      int64_t id);

  // 声明 make_intrusive 函数为友元，允许它访问私有构造函数，用于创建 intrusive_ptr<Message>。
  friend c10::intrusive_ptr<Message>;

 public:
  // 禁用拷贝构造函数和移动构造函数，确保 Message 对象只能通过 make_intrusive 创建并持有。
  Message(const Message& other) = delete;
  Message(Message&& other) = delete;
  Message& operator=(Message const& rhs) = delete;
  Message& operator=(Message&& rhs) = delete;

  // 返回 payload 并将其移动（右值引用），用于在调用方接收并占有 payload。
  std::vector<char>&& movePayload() &&;
  // 返回 payload 的可修改引用。
  std::vector<char>& payload();
  // 返回 payload 的常量引用。
  const std::vector<char>& payload() const;

  // 返回 tensors 并将其移动（右值引用），用于在调用方接收并占有 tensors。
  std::vector<torch::Tensor>&& moveTensors() &&;
  // 返回 tensors 的可修改引用。
  std::vector<torch::Tensor>& tensors();
  // 返回 tensors 的常量引用。
  const std::vector<torch::Tensor>& tensors() const;

  // 返回消息的类型。
  MessageType type() const;

  // 返回消息是否为请求类型。
  bool isRequest() const;
  // 返回消息是否为响应类型。
  bool isResponse() const;
  // 返回消息是否为关闭类型。
  bool isShutdown() const;

  // 返回消息的 id，用于请求和响应的匹配。
  int64_t id() const;
  // 设置消息的 id。
  void setId(int64_t id);

  // 返回消息中所有存储的弱引用列表。
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> getStorages() const;

 private:
  // 存储消息的二进制数据。
  std::vector<char> payload_;
  // 存储消息的张量数据。
  std::vector<torch::Tensor> tensors_;
  // 消息类型，默认为 UNKNOWN。
  MessageType type_ = MessageType::UNKNOWN;
  // 消息 id，默认为 -1。
  int64_t id_ = -1;
};

// 创建一个类型为 Exception 的响应消息。
// 使用异常的字符串表示作为消息的 payload。
// 可以提供与引发此响应的请求对应的消息 id 以便进行请求/响应的匹配。
TORCH_API c10::intrusive_ptr<Message> createExceptionResponse(
    const std::exception& e,
    int64_t id);
// 创建一个异常响应消息，返回类型为 c10::intrusive_ptr<Message>
// 参数 exceptionStr 是异常消息的字符串表示，作为消息的载荷
// 参数 id 是对应请求的消息 ID，用于匹配请求和响应
TORCH_API c10::intrusive_ptr<Message> createExceptionResponse(
    const std::string& exceptionStr,
    int64_t id);

// 定义一个内联函数 withStorages，返回一个 tuple
// 参数 message 是 c10::intrusive_ptr<Message> 类型，表示消息对象
inline std::tuple<
    c10::intrusive_ptr<Message>,                       // 返回的消息对象
    std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>>> // 存储器对象的 vector
withStorages(c10::intrusive_ptr<Message> message) {
  // 调用 message 对象的 getStorages 方法，获取其中的存储器对象
  auto storages = message->getStorages();
  // 返回一个 tuple，包含原始的 message 对象和 storages 的移动构造
  return std::make_tuple(std::move(message), std::move(storages));
}

// 定义 JitFuture 类型别名，表示 c10::ivalue::Future 类型
using JitFuture = c10::ivalue::Future;

// 命名空间声明：rpc
namespace rpc {
  // 命名空间声明：distributed
  namespace distributed {
  } // namespace distributed
} // namespace rpc
```