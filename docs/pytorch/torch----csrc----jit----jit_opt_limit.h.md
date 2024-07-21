# `.\pytorch\torch\csrc\jit\jit_opt_limit.h`

```py
// 预处理指令，指定本头文件在编译时只包含一次
#pragma once

// 包含torch的导出宏定义头文件，用于声明导出符号
#include <torch/csrc/Export.h>

// 包含标准字符串处理库
#include <string>

// 包含无序映射容器的头文件
#include <unordered_map>

// TorchScript 提供了一个简单的优化限制检查器，可以通过环境变量 `PYTORCH_JIT_OPT_LIMIT` 进行配置。
// 它的目的是限制每次优化的数量，有助于调试任何优化过程中的问题。

// 优化限制检查器是基于每个文件启用的（因此是基于每次优化的）。例如，在 `constant_propagation.cpp` 文件中，
// 应将 `PYTORCH_JIT_OPT_LIMIT` 设置为 `constant_propagation=<opt_limit>` 或简单地为
// `constant_propagation=<opt_limit>`，其中 <opt_limit> 是您想要在此优化过程中执行的优化次数。
// （例如 `PYTORCH_JIT_OPT_LIMIT="constant_propagation=<opt_limit>"`）。

// 可以通过调用 JIT_OPT_ALLOWED 来调用优化限制器。如果尚未达到优化限制，则返回true。否则返回false。典型用法：

// if (!JIT_OPT_ALLOWED) {
//     GRAPH_DUMP(...); // 由 jit_log 提供
//     return;
// }

// torch::jit 命名空间，包含了与 JIT 相关的函数和类
namespace torch {
namespace jit {

// TORCH_API 是一个导出宏，声明了该函数在库外可见
TORCH_API bool opt_limit(const char* pass_name);

// 定义了一个宏 JIT_OPT_ALLOWED，用于检查当前是否仍然可以进行优化，调用了 opt_limit 函数
#define JIT_OPT_ALLOWED opt_limit(__FILE__)

} // namespace jit
} // namespace torch
```