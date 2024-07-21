# `.\pytorch\c10\util\floating_point_utils.h`

```
#pragma once

#include <c10/macros/Macros.h>  // 包含 C10 宏定义的头文件
#include <cstdint>  // 包含标准整数类型的头文件

namespace c10::detail {

// 定义一个内联函数，将给定的 32 位无符号整数解释为单精度浮点数
C10_HOST_DEVICE inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);  // 如果是 OpenCL 环境，使用 as_float 函数解释为浮点数
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);  // 如果是 CUDA 环境，使用 __uint_as_float 函数解释为浮点数
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);  // 如果是 Intel 编译器，使用 _castu32_f32 函数解释为浮点数
#else
  union {
    uint32_t as_bits;  // 作为比特位的无符号整数
    float as_value;    // 作为浮点值的单精度浮点数
  } fp32 = {w};         // 使用给定的整数 w 初始化联合体
  return fp32.as_value; // 返回联合体中作为浮点值的部分
#endif
}

// 定义一个内联函数，将给定的单精度浮点数转换为对应的 32 位整数表示
C10_HOST_DEVICE inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);  // 如果是 OpenCL 环境，使用 as_uint 函数转换为整数表示
#elif defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);  // 如果是 CUDA 环境，使用 __float_as_uint 函数转换为整数表示
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);  // 如果是 Intel 编译器，使用 _castf32_u32 函数转换为整数表示
#else
  union {
    float as_value;     // 作为浮点值的单精度浮点数
    uint32_t as_bits;   // 作为比特位的无符号整数
  } fp32 = {f};         // 使用给定的浮点数 f 初始化联合体
  return fp32.as_bits;  // 返回联合体中作为整数表示的部分
#endif
}

} // namespace c10::detail
```