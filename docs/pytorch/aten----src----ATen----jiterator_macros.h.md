# `.\pytorch\aten\src\ATen\jiterator_macros.h`

```py
#pragma once
// 指令：保证头文件只被编译一次

#include <c10/macros/Macros.h>
// 包含 c10 库的宏定义文件

#include <string>
// 包含标准字符串库

#define JITERATOR_HOST_DEVICE C10_HOST_DEVICE
// 定义 JITERATOR_HOST_DEVICE 宏，用于声明宿主或设备函数

#if defined(_MSC_VER) && defined(__CUDACC__)
// 如果在 Windows 下使用 CUDA 编译器 _MSC_VER，并且定义了 __CUDACC__

// NVRTC on Windows errors if __host__ __device__ attribute is
// present on kernel.
// error: attribute "__host__" does not apply here
// error: attribute "__device__" does not apply here

// 如果在 Windows 下使用 NVRTC 编译器，__host__ __device__ 属性会导致错误
#define JITERATOR_HOST_DEVICE
// 清除 JITERATOR_HOST_DEVICE 宏定义
#endif

// jiterator_also_stringify_as 宏用于定义代码（用于 CPU/ROCm），并为 `jiterator` 生成代码字符串（仅在 CUDA 编译时）
// 用法:
//      jiterator_also_stringify_as(
//          jiterator_code(template <typename T> T identity(T x) { return x; }),
//          identity_string);
// 这将定义模板 `identity`，并将 `std::string identity_string` 定义为代码字符串
// 如果编译目标为 CUDA，则会生成代码字符串。

// `jiterator_code` 宏用于处理内核代码中的逗号 `,`
// 这些逗号会误导预处理器，认为我们正在向宏传递多个参数。
#define jiterator_code(...) __VA_ARGS__
// 定义 jiterator_code 宏，展开参数列表

#if defined(__CUDACC__) || defined(__HIPCC__)
// 如果正在使用 CUDA 或 HIP 编译器

// CPU 和 CUDA 和 ROCm 的情况
#define stringify_code(...) #__VA_ARGS__
// 定义 stringify_code 宏，将参数展开为字符串

#define jiterator_also_stringify_as(code, str_name) \
  code /* define the function */                    \
      const std::string str_name = std::string(stringify_code(code));
// 定义 jiterator_also_stringify_as 宏，生成函数代码，并定义名为 str_name 的字符串常量

#else
// CPU only 或 CPU 和 ROCm 的情况

// 只需要函数定义
#define jiterator_also_stringify_as(code, str_name) code
// 定义 jiterator_also_stringify_as 宏，仅生成函数代码

#endif
```