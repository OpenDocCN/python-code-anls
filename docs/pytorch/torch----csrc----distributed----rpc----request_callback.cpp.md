# `.\pytorch\torch\csrc\distributed\rpc\request_callback.cpp`

```
// 包含请求回调所需的头文件
#include <torch/csrc/distributed/rpc/request_callback.h>

// 包含分布式自动求导相关的头文件
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/utils.h>

// 定义命名空间 torch::distributed::rpc
namespace torch {
namespace distributed {
namespace rpc {

// 使用 torch::distributed::autograd 命名空间中的所有内容
using namespace torch::distributed::autograd;

// 请求回调的运算符重载函数
c10::intrusive_ptr<JitFuture> RequestCallback::operator()(
    Message& request,
    std::vector<c10::Stream> streams) const {
  // 注意：这里不能清除自动求导上下文 ID，因为 processMessage 方法
  // 可能会在等待所有参数中的 RRef 被其所有者确认并在不同线程中继续处理。
  // 因此，线程局部的上下文 ID 需要在确实执行处理逻辑的线程中设置和清除。
  return processMessage(request, std::move(streams));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```