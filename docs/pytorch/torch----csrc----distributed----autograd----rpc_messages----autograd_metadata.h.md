# `.\pytorch\torch\csrc\distributed\autograd\rpc_messages\autograd_metadata.h`

```py
#pragma once

#include <torch/csrc/Export.h>  // 包含 Torch 导出宏的头文件

#include <cstdint>  // 包含用于固定宽度整数类型的头文件

namespace torch {
namespace distributed {
namespace autograd {

// This structure represents autograd metadata that we need to pass across
// different nodes when we call an RPC which needs autograd computation.
// 这个结构体表示在调用需要自动求导计算的 RPC 时，我们需要在不同节点之间传递的自动求导元数据。
struct TORCH_API AutogradMetadata {
  
  // Constructor to initialize AutogradMetadata with autogradContextId and autogradMessageId.
  // 构造函数，用于初始化 AutogradMetadata，参数为 autogradContextId 和 autogradMessageId。
  AutogradMetadata(int64_t autogradContextId, int64_t autogradMessageId);

  // autogradContextId_ is a globally unique integer that identifies a
  // particular distributed autograd pass.
  // autogradContextId_ 是一个全局唯一的整数，用于标识特定的分布式自动求导传递。
  int64_t autogradContextId;

  // autogradMessageId_ is a globally unique integer that identifies a pair
  // of send/recv autograd functions.
  // autogradMessageId_ 是一个全局唯一的整数，用于标识发送/接收自动求导函数对。
  int64_t autogradMessageId;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```