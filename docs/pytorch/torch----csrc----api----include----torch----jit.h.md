# `.\pytorch\torch\csrc\api\include\torch\jit.h`

```
#pragma once

#include <torch/csrc/Export.h>  // Torch导出宏定义
#include <torch/csrc/jit/api/module.h>  // Torch JIT模块API

#include <memory>  // 内存管理
#include <string>  // 字符串处理

namespace torch {
namespace jit {

/// 将脚本代码编译为可执行图形式的模块。
///
/// 接受一个包含脚本语法函数的字符串，并将其编译为模块（图形）。返回的模块提供了 `run_method` 函数，
/// 可用于调用编译后的函数。
///
/// 例如：
/// \rst
/// .. code-block:: cpp
///
///   auto module = torch::jit::compile(R"JIT(
///     def relu_script(a, b):
///       return torch.relu(a + b)
///     def test_while(a, i):
///       while i < 10:
///         a += a
///         i += 1
///       return a
///   )JIT");
///   IValue output = module->run_method("relu_script", a, b);
/// \endrst
TORCH_API std::shared_ptr<CompilationUnit> compile(const std::string& source);

} // namespace jit
} // namespace torch
```