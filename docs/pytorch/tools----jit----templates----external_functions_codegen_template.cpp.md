# `.\pytorch\tools\jit\templates\external_functions_codegen_template.cpp`

```
// 包含外部函数声明的头文件
#include <torch/csrc/jit/tensorexpr/external_functions.h>

// 包含 ATen 库的函数声明和定义
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

// 包含用于循环的工具函数
#include <c10/util/irange.h>

// 包含外部函数注册相关的头文件
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

// 声明 torch 命名空间
namespace torch {
// 声明 jit 命名空间
namespace jit {
// 声明 tensorexpr 命名空间
namespace tensorexpr {

// 如果定义了 C10_MOBILE 宏，则定义 extern "C" 语法块
#ifdef C10_MOBILE
extern "C" {
#endif

// 插入外部函数定义（在此处被替换为实际的外部函数声明和定义）
${external_functions}

// 如果未定义 C10_MOBILE 宏，则插入外部函数注册相关代码
#ifndef C10_MOBILE
${external_registrations}
#endif

// 如果定义了 C10_MOBILE 宏，则结束 extern "C" 语法块
#ifdef C10_MOBILE
} // extern "C"
#endif

// 结束 tensorexpr 命名空间
} // namespace tensorexpr
// 结束 jit 命名空间
} // namespace jit
// 结束 torch 命名空间
} // namespace torch
```