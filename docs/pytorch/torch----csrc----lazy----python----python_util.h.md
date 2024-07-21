# `.\pytorch\torch\csrc\lazy\python\python_util.h`

```py
#pragma once
// 使用 #pragma once 确保头文件只被编译一次，避免重复包含

#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional 头文件，用于支持可选的值类型

#include <torch/csrc/Export.h>
// 包含 torch 库中的 Export 头文件，用于导出符号

#include <torch/csrc/lazy/core/ir_metadata.h>
// 包含 torch 库中 lazy 模块的核心 IR 元数据头文件

#include <vector>
// 包含标准库中的 vector 头文件，用于定义和操作动态数组

namespace torch {
namespace lazy {

std::optional<SourceLocation> TORCH_PYTHON_API GetPythonFrameTop();
// 声明一个函数 GetPythonFrameTop，返回一个 std::optional<SourceLocation> 可选值类型，
// 并使用 TORCH_PYTHON_API 指定函数的导出方式（如 DLL 导出）

std::vector<SourceLocation> TORCH_PYTHON_API GetPythonFrames();
// 声明一个函数 GetPythonFrames，返回一个 vector<SourceLocation> 类型，
// 并使用 TORCH_PYTHON_API 指定函数的导出方式（如 DLL 导出）

} // namespace lazy
} // namespace torch
// 命名空间声明，torch::lazy 命名空间包含了 GetPythonFrameTop 和 GetPythonFrames 函数声明
```