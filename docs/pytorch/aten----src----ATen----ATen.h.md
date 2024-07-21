# `.\pytorch\aten\src\ATen\ATen.h`

```
#pragma once
// 如果编译器不是 MSVC 并且 C++ 版本小于 C++17，则抛出错误信息，要求使用兼容 C++17 或更新版本的编译器
#if !defined(_MSC_VER) && __cplusplus < 201703L
#error C++17 or later compatible compiler is required to use ATen.
#endif

#include <ATen/Context.h>            // 包含 ATen 的上下文相关头文件
#include <ATen/Device.h>             // 包含 ATen 的设备相关头文件
#include <ATen/DeviceGuard.h>        // 包含 ATen 的设备守护相关头文件
#include <ATen/DimVector.h>          // 包含 ATen 的维度向量相关头文件
#include <ATen/Dispatch.h>           // 包含 ATen 的分发相关头文件
#include <ATen/Formatting.h>         // 包含 ATen 的格式化相关头文件
#include <ATen/Functions.h>          // 包含 ATen 的函数相关头文件
#include <ATen/NamedTensor.h>        // 包含 ATen 的命名张量相关头文件
#include <ATen/ScalarOps.h>          // 包含 ATen 的标量操作相关头文件
#include <ATen/Tensor.h>             // 包含 ATen 的张量相关头文件
#include <ATen/TensorGeometry.h>     // 包含 ATen 的张量几何相关头文件
#include <ATen/TensorIndexing.h>     // 包含 ATen 的张量索引相关头文件
#include <ATen/TensorOperators.h>    // 包含 ATen 的张量操作相关头文件
#include <ATen/Version.h>            // 包含 ATen 的版本相关头文件
#include <ATen/core/ATenGeneral.h>   // 包含 ATen 核心通用功能相关头文件
#include <ATen/core/Generator.h>     // 包含 ATen 核心生成器相关头文件
#include <ATen/core/Reduction.h>     // 包含 ATen 核心减少相关头文件
#include <ATen/core/Scalar.h>        // 包含 ATen 核心标量相关头文件
#include <ATen/core/UnsafeFromTH.h>  // 包含 ATen 核心不安全转换 TH 相关头文件
#include <ATen/core/ivalue.h>        // 包含 ATen 核心 IValue 相关头文件
#include <ATen/core/jit_type.h>      // 包含 ATen 核心 JIT 类型相关头文件
#include <c10/core/Allocator.h>      // 包含 c10 核心分配器相关头文件
#include <c10/core/InferenceMode.h>  // 包含 c10 核心推理模式相关头文件
#include <c10/core/Layout.h>         // 包含 c10 核心布局相关头文件
#include <c10/core/Storage.h>        // 包含 c10 核心存储相关头文件
#include <c10/core/TensorOptions.h>  // 包含 c10 核心张量选项相关头文件
#include <c10/util/Exception.h>      // 包含 c10 工具异常相关头文件

// TODO: try to remove this
// 有一些背景故事，请参阅 https://github.com/pytorch/pytorch/issues/48684
#include <ATen/NativeFunctions.h>    // 包含 ATen 原生函数相关头文件
```