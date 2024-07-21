# `.\pytorch\torch\csrc\cuda\device_set.h`

```
#pragma once

// `#pragma once` 是预处理指令，用于确保头文件只被编译一次，提升编译效率。


#include <c10/cuda/CUDAMacros.h>

// 包含 `<c10/cuda/CUDAMacros.h>` 头文件，用于定义 CUDA 相关的宏和功能。


#include <bitset>

// 包含 `<bitset>` 头文件，用于定义和操作固定大小的位集合。


#include <cstddef>

// 包含 `<cstddef>` 头文件，其中定义了 `std::size_t` 和 `nullptr_t` 类型，通常用于表示大小和空指针常量。


namespace torch {

// 定义命名空间 `torch`，用于组织和封装相关的类、函数和变量。


using device_set = std::bitset<C10_COMPILE_TIME_MAX_GPUS>;

// 定义 `device_set` 类型别名，表示一个位集合，其大小由 `C10_COMPILE_TIME_MAX_GPUS` 宏指定，用于管理 GPU 设备集合。


} // namespace torch

// 结束命名空间 `torch` 的定义。
```