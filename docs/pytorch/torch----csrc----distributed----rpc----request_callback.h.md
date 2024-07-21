# `.\pytorch\torch\csrc\distributed\rpc\request_callback.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <torch/csrc/distributed/rpc/message.h>
// 包含 Torch 分布式 RPC 消息头文件

namespace torch {
namespace distributed {
namespace rpc {

// Functor which is invoked to process an RPC message. This is an abstract class
// with some common functionality across all request handlers. Users need to
// implement this interface to perform the actual business logic.
// 用于处理 RPC 消息的函数对象。这是一个抽象类，包含所有请求处理程序通用功能。用户需要实现此接口来执行实际的业务逻辑。
class TORCH_API RequestCallback {
 public:
  // Invoke the callback.
  // 调用回调函数。
  c10::intrusive_ptr<JitFuture> operator()(
      Message& request,
      std::vector<c10::Stream> streams) const;

  virtual ~RequestCallback() = default;
  // 虚析构函数，确保多态对象的正确销毁

 protected:
  // RpcAgent implementation should invoke ``RequestCallback`` to process
  // received requests. There is no restriction on the implementation's
  // threading model. This function takes an rvalue reference of the Message
  // object. It is expected to return the future to a response message or
  // message containing an exception. Different rpc agent implementations are
  // expected to ensure delivery of the response/exception based on their
  // implementation specific mechanisms.
  // RpcAgent 实现应该调用 ``RequestCallback`` 来处理接收到的请求。对于实现的线程模型没有限制。
  // 此函数接受 Message 对象的右值引用。预期返回响应消息或包含异常的消息的 future。
  // 不同的 rpc agent 实现应确保根据其特定机制传递响应/异常。
  virtual c10::intrusive_ptr<JitFuture> processMessage(
      Message& request,
      std::vector<c10::Stream> streams) const = 0;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```