# `.\pytorch\torch\csrc\distributed\autograd\context\context.h`

```py
#pragma once


// 使用 pragma once 指令确保头文件只被编译一次，避免重复包含

#include <cstdint>
#include <functional>

#include <ATen/core/Dict.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace autograd {

// DistAutogradContext which stores information for a single distributed
// autograd pass on a worker.
// 存储单个工作节点上分布式自动求导过程的信息的 DistAutogradContext 类
class DistAutogradContext;

};

using ContextPtr = std::shared_ptr<DistAutogradContext>;

// This class stores a shared_ptr to a DistAutogradContext instance in a
// thread local variable. The instance is given by the call site. The class
// doesn't know the current context. It's just a util class.
// 这个类将 DistAutogradContext 实例的 shared_ptr 存储在线程局部变量中。实例由调用方提供。该类不知道当前上下文，仅作为实用类使用。
class TORCH_API ThreadLocalDistAutogradContext {
 public:
  // Store 'new_context' to the thread local variable maintained by this class.
  // 将 'new_context' 存储到该类维护的线程局部变量中
  explicit ThreadLocalDistAutogradContext(ContextPtr&& new_context);

  // Destructor to clean up the stored context when the object goes out of scope.
  // 析构函数，在对象超出作用域时清理存储的上下文
  ~ThreadLocalDistAutogradContext();

  // Retrieve the stored DistAutogradContext instance.
  // 获取存储的 DistAutogradContext 实例
  static ContextPtr getContextPtr();

 private:
  ContextPtr prev_context_ptr_;  // 存储前一个上下文的指针
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```