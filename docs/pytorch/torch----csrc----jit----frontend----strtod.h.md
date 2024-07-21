# `.\pytorch\torch\csrc\jit\frontend\strtod.h`

```py
#pragma once


// 使用 #pragma once 预处理指令，确保头文件只被编译一次，防止多重包含



#include <c10/macros/Macros.h>


// 包含 c10 库中的 Macros.h 头文件，提供 C++ 的宏定义和预处理器支持



namespace torch {
namespace jit {


// 命名空间声明：进入 torch 命名空间，然后进入 jit 子命名空间



TORCH_API double strtod_c(const char* nptr, char** endptr);
TORCH_API float strtof_c(const char* nptr, char** endptr);


// 声明两个函数 strtod_c 和 strtof_c，它们使用 TORCH_API 宏修饰
// strtod_c: 将字符串转换为 double 类型的数字，并设置 endptr 指向转换后的字符串末尾
// strtof_c: 将字符串转换为 float 类型的数字，并设置 endptr 指向转换后的字符串末尾



} // namespace jit
} // namespace torch


// 命名空间结束：退出 jit 子命名空间，然后退出 torch 命名空间
```