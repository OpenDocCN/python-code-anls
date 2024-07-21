# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\autograd_metadata.cpp`

```py
#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
// 包含 Torch 分布式自动求导模块中的自动求导元数据头文件

namespace torch {
namespace distributed {
namespace autograd {

AutogradMetadata::AutogradMetadata(
    int64_t autogradContextId_,
    int64_t autogradMessageId_)
    : autogradContextId(autogradContextId_),
      autogradMessageId(autogradMessageId_) {}
// AutogradMetadata 类的构造函数实现，接收两个 int64_t 类型参数 autogradContextId_ 和 autogradMessageId_
// 使用成员初始化列表初始化 AutogradMetadata 对象的 autogradContextId 和 autogradMessageId 成员变量

} // namespace autograd
} // namespace distributed
} // namespace torch
// 命名空间结束，包括 torch、distributed 和 autograd 命名空间的关闭
```