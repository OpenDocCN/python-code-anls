# `.\pytorch\aten\src\ATen\cpu\vec\vec_half.h`

```py
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <ATen/cpu/vec/intrinsics.h>
// 包含 ATen 库中 CPU 向量化操作的头文件

namespace at::vec {
// 命名空间 at::vec，包含了向量化操作的实现

inline namespace CPU_CAPABILITY {
// 内联命名空间 CPU_CAPABILITY，见注释 [CPU_CAPABILITY namespace]

#if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
    !defined(__APPLE__)
// 如果编译器支持 AVX2 或 AVX512 并且不是在苹果系统下

static inline uint16_t float2half_scalar(float val) {
// 将单精度浮点数转换为半精度浮点数的函数

#if defined(CPU_CAPABILITY_AVX2)
// 如果支持 AVX2 指令集
#if defined(_MSC_VER)
// 如果是在 Microsoft Visual Studio 编译器下
  __m256 v = _mm256_set1_ps(val);
  // 创建一个包含 val 的 256 位浮点数向量
  __m128i o =
      _mm256_cvtps_ph(v, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  // 将 256 位浮点数向量转换为半精度浮点数向量，向最近整数舍入且不触发异常
  return static_cast<std::uint16_t>(_mm_cvtsi128_si32(o));
  // 将结果转换为 16 位整数并返回
#else
  return _cvtss_sh(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  // 使用硬件指令将单精度浮点数转换为半精度浮点数，向最近整数舍入且不触发异常
#endif
#elif defined(CPU_CAPABILITY_AVX512)
// 如果支持 AVX512 指令集
  __m512 v = _mm512_set1_ps(val);
  // 创建一个包含 val 的 512 位浮点数向量
  __m256i o =
      _mm512_cvtps_ph(v, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  // 将 512 位浮点数向量转换为半精度浮点数向量，向最近整数舍入且不触发异常
  return static_cast<std::uint16_t>(
      _mm_cvtsi128_si32(_mm256_castsi256_si128(o)));
  // 将结果转换为 16 位整数并返回
#endif
}

static inline float half2float_scalar(uint16_t val) {
// 将半精度浮点数转换为单精度浮点数的函数

#if defined(CPU_CAPABILITY_AVX2)
// 如果支持 AVX2 指令集
#if defined(_MSC_VER)
// 如果是在 Microsoft Visual Studio 编译器下
  __m128i v = _mm_cvtsi32_si128(val);
  // 创建一个包含 val 的 128 位整数向量
  __m256 o = _mm256_cvtph_ps(v);
  // 将半精度浮点数向量转换为单精度浮点数向量
  return _mm256_cvtss_f32(o);
  // 提取结果中的单精度浮点数并返回
#else
  return _cvtsh_ss(val);
  // 使用硬件指令将半精度浮点数转换为单精度浮点数
#endif
#elif defined(CPU_CAPABILITY_AVX512)
// 如果支持 AVX512 指令集
  __m256i v =
      _mm256_setr_epi16(val, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  // 创建一个包含 val 和零的 256 位整数向量
  __m512 o = _mm512_cvtph_ps(v);
  // 将半精度浮点数向量转换为单精度浮点数向量
  return _mm512_cvtss_f32(o);
  // 提取结果中的单精度浮点数并返回
#endif
}

#endif
// 结束 AVX2 或 AVX512 的条件编译块

} // namespace CPU_CAPABILITY
// 结束 CPU_CAPABILITY 命名空间

} // namespace at::vec
// 结束 at::vec 命名空间
```