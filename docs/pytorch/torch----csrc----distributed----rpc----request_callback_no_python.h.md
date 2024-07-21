# `.\pytorch\torch\csrc\distributed\rpc\request_callback_no_python.h`

```
#pragma once
// 使用 pragma once 指令确保头文件只被编译一次

#include <torch/csrc/distributed/rpc/message.h>
// 包含消息相关的头文件

#include <torch/csrc/distributed/rpc/request_callback.h>
// 包含请求回调相关的头文件

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
// 包含 RPC 命令基类相关的头文件

#include <torch/csrc/distributed/rpc/rref_impl.h>
// 包含远程引用实现相关的头文件

#include <torch/csrc/distributed/rpc/script_call.h>
// 包含脚本调用相关的头文件

#include <torch/csrc/distributed/rpc/script_remote_call.h>
// 包含脚本远程调用相关的头文件

namespace torch {
namespace distributed {
namespace rpc {

// 命名空间 torch::distributed::rpc 中的代码实现了 RPC 请求回调，不依赖于 Python。
};

} // namespace rpc
} // namespace distributed
} // namespace torch
// 命名空间嵌套，定义了 torch::distributed::rpc 命名空间及其内部的实现
```