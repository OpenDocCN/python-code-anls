# `.\pytorch\aten\src\ATen\jit_macros.h`

```py
#pragma once
// 在 C++ 中，#pragma once 是一种预处理指令，用于确保头文件只被包含一次

#include <ATen/cuda/CUDAConfig.h>
// 包含 ATen 库中 CUDAConfig.h 头文件，该文件可能包含了与 CUDA 相关的配置信息和声明

#include <string>
// 包含 C++ 标准库中的 string 头文件，用于使用 string 类型

// AT_USE_JITERATOR(), controls whether we jit some elementwise kernels
// 宏定义 AT_USE_JITERATOR() 控制是否对一些逐元素操作的内核进行即时编译

#define AT_USE_JITERATOR() true
// 定义宏 AT_USE_JITERATOR() 的值为 true，表示启用即时编译

#define jiterator_stringify(...) std::string(#__VA_ARGS__);
// 定义宏 jiterator_stringify(...)，将传入的参数转换为字符串，并封装成 std::string 对象
```