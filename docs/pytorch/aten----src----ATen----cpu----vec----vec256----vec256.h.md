# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256.h`

```py
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>  // 包含矢量化指令的头文件

#include <ATen/cpu/vec/vec_base.h>  // 包含矢量化基类的头文件
#if !(defined(__VSX__)  || defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR))
#include <ATen/cpu/vec/vec256/vec256_float.h>  // 包含单精度浮点矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_float_neon.h>  // 包含NEON单精度浮点矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_half_neon.h>  // 包含NEON半精度浮点矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_bfloat16.h>  // 包含BFLOAT16矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_double.h>  // 包含双精度浮点矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_int.h>  // 包含整型矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_qint.h>  // 包含量化整数矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_complex_float.h>  // 包含复数单精度浮点矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_complex_double.h>  // 包含复数双精度浮点矢量化实现的头文件
#elif defined(__VSX__)  || defined(CPU_CAPABILITY_VSX)
#include <ATen/cpu/vec/vec256/vsx/vec256_common_vsx.h>  // 包含VSX通用矢量化实现的头文件
#else
#include <ATen/cpu/vec/vec256/zarch/vec256_zarch.h>  // 包含z/Architecture矢量化实现的头文件
#include <ATen/cpu/vec/vec256/vec256_bfloat16.h>  // 包含BFLOAT16矢量化实现的头文件
#endif

#include <ATen/cpu/vec/vec256/vec256_convert.h>  // 包含矢量化类型转换的头文件
#include <ATen/cpu/vec/vec256/vec256_mask.h>  // 包含矢量化掩码操作的头文件

#include <algorithm>  // 包含STL算法库的头文件
#include <cstddef>  // 包含标准C库头文件，定义了NULL等宏
#include <cstdint>  // 包含标准整数类型定义的头文件
#include <cstring>  // 包含C字符串处理函数的头文件
#include <ostream>  // 包含输出流操作的头文件

namespace at::vec {

// Note [CPU_CAPABILITY namespace]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This header, and all of its subheaders, will be compiled with
// different architecture flags for each supported set of vector
// intrinsics. So we need to make sure they aren't inadvertently
// linked together. We do this by declaring objects in an `inline
// namespace` which changes the name mangling, but can still be
// accessed as `at::vec`.
inline namespace CPU_CAPABILITY {

inline std::ostream& operator<<(std::ostream& stream, const c10::qint32& val) {
  stream << val.val_;
  return stream;
}
inline std::ostream& operator<<(std::ostream& stream, const c10::qint8& val) {
  stream << static_cast<int>(val.val_);
  return stream;
}
inline std::ostream& operator<<(std::ostream& stream, const c10::quint8& val) {
  stream << static_cast<unsigned int>(val.val_);
  return stream;
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vectorized<T>& vec) {
  T buf[Vectorized<T>::size()];
  vec.store(buf);
  stream << "vec[";
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    if (i != 0) {
      stream << ", ";
    }
    stream << buf[i];
  }
  stream << "]";
  return stream;
}


#if defined(CPU_CAPABILITY_AVX2)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<>
inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
  return _mm256_castpd_ps(src);  // 将双精度浮点向单精度浮点转换的AVX2指令实现
}

template<>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return _mm256_castps_pd(src);  // 将单精度浮点向双精度浮点转换的AVX2指令实现
}

template<>
inline Vectorized<float> cast<float, int32_t>(const Vectorized<int32_t>& src) {
  return _mm256_castsi256_ps(src);  // 将整型向单精度浮点转换的AVX2指令实现
}

template<>
inline Vectorized<double> cast<double, int64_t>(const Vectorized<int64_t>& src) {
  return _mm256_castsi256_pd(src);  // 将长整型向双精度浮点转换的AVX2指令实现
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifndef _MSC_VER
// MSVC is not working well on complex function overload.

// 定义模板函数 gather，从 double 类型的基地址 base_addr 中使用 vindex 所指示的索引进行数据收集
// 这里使用 AVX2 指令集的 _mm256_i64gather_pd 实现
template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline gather(const double* base_addr, const Vectorized<int64_t>& vindex) {
  return _mm256_i64gather_pd(base_addr, vindex, scale);
}

// 定义模板函数 gather，从 float 类型的基地址 base_addr 中使用 vindex 所指示的索引进行数据收集
// 这里使用 AVX2 指令集的 _mm256_i32gather_ps 实现
template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline gather(const float* base_addr, const Vectorized<int32_t>& vindex) {
  return _mm256_i32gather_ps(base_addr, vindex, scale);
}
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifndef _MSC_VER
// MSVC is not working well on complex function overload.

// 定义模板函数 mask_gather，从 double 类型的 base_addr 地址中使用 vindex 索引按照 mask 的条件进行数据收集
// 这里使用 AVX2 指令集的 _mm256_mask_i64gather_pd 实现
template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline mask_gather(const Vectorized<double>& src, const double* base_addr,
                   const Vectorized<int64_t>& vindex, Vectorized<double>& mask) {
  return _mm256_mask_i64gather_pd(src, base_addr, vindex, mask, scale);
}

// 定义模板函数 mask_gather，从 float 类型的 base_addr 地址中使用 vindex 索引按照 mask 的条件进行数据收集
// 这里使用 AVX2 指令集的 _mm256_mask_i32gather_ps 实现
template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline mask_gather(const Vectorized<float>& src, const float* base_addr,
                   const Vectorized<int32_t>& vindex, Vectorized<float>& mask) {
  return _mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale);
}
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 仅适用于范围在 [-2^51, 2^51] 内的输入
// 来自：https://stackoverflow.com/a/41148578
template<>
Vectorized<int64_t>
inline convert_to_int_of_same_size<double>(const Vectorized<double> &src) {
  auto x = _mm256_add_pd(src, _mm256_set1_pd(0x0018000000000000));
  return _mm256_sub_epi64(
      _mm256_castpd_si256(x),
      _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000))
  );
}

// 仅适用于范围在 [-2^51, 2^51] 内的输入
// 来自：https://stackoverflow.com/a/41148578
template<>
Vectorized<int32_t>
inline convert_to_int_of_same_size<float>(const Vectorized<float> &src) {
  return _mm256_cvttps_epi32(src);
}

// 仅适用于范围在 [-2^51, 2^51] 内的输入
// 来自：https://stackoverflow.com/a/41148578
template<>
Vectorized<double>
inline convert_to_fp_of_same_size<double>(const Vectorized<int64_t> &src) {
  auto x = _mm256_add_epi64(src, _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
  return _mm256_sub_pd(
    _mm256_castsi256_pd(x),
    _mm256_set1_pd(0x0018000000000000)
  );
}

// 仅适用于范围在 [-2^51, 2^51] 内的输入
// 来自：https://stackoverflow.com/a/41148578
template<>
Vectorized<float>
inline convert_to_fp_of_same_size<float>(const Vectorized<int32_t> &src) {
  return _mm256_cvtepi32_ps(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 这里应该继续添加代码来完成函数定义的注释
template <>
std::pair<Vectorized<double>, Vectorized<double>>
// 在 AVX2 环境下，交错两个双精度向量的函数
inline interleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // 输入：
  //   a = {a0, a1, a3, a3}
  //   b = {b0, b1, b2, b3}

  // 交换向量的通道：
  //   a_swapped = {a0, a1, b0, b1}
  //   b_swapped = {a2, a3, b2, b3}
  auto a_swapped = _mm256_permute2f128_pd(a, b, 0b0100000);  // 0, 2.   4 位间隔
  auto b_swapped = _mm256_permute2f128_pd(a, b, 0b0110001);  // 1, 3.   4 位间隔

  // 分组列交错到不同通道：
  //   返回 {a0, b0, a1, b1}
  //        {a2, b2, a3, b3}
  return std::make_pair(_mm256_permute4x64_pd(a_swapped, 0b11011000),  // 0, 2, 1, 3
                        _mm256_permute4x64_pd(b_swapped, 0b11011000)); // 0, 2, 1, 3
}

// 对单精度向量进行特化的交错函数
template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline interleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // 输入：
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}

  // 交换向量的通道：
  //   a_swapped = {a0, a1, a2, a3, b0, b1, b2, b3}
  //   b_swapped = {a4, a5, a6, a7, b4, b5, b6, b7}
  // TODO: 是否可以支持缓存此操作？
  auto a_swapped = _mm256_permute2f128_ps(a, b, 0b0100000);  // 0, 2.   4 位间隔
  auto b_swapped = _mm256_permute2f128_ps(a, b, 0b0110001);  // 1, 3.   4 位间隔

  // 分组列交错到不同通道：
  //   返回 {a0, b0, a1, b1, a2, b2, a3, b3}
  //        {a4, b4, a5, b5, a6, b6, a7, b7}
  const __m256i group_ctrl = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  return std::make_pair(_mm256_permutevar8x32_ps(a_swapped, group_ctrl),
                        _mm256_permutevar8x32_ps(b_swapped, group_ctrl));
}

// 双精度向量的反交错函数的特化
template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline deinterleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // 输入：
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}

  // 分组列交错到不同通道：
  //   a_grouped = {a0, a1, b0, b1}
  //   b_grouped = {a2, a3, b2, b3}
  auto a_grouped = _mm256_permute4x64_pd(a, 0b11011000);  // 0, 2, 1, 3
  auto b_grouped = _mm256_permute4x64_pd(b, 0b11011000);  // 0, 2, 1, 3

  // 交换向量的通道：
  //   返回 {a0, a1, a2, a3}
  //        {b0, b1, b2, b3}
  return std::make_pair(_mm256_permute2f128_pd(a_grouped, b_grouped, 0b0100000),  // 0, 2.   4 位间隔
                        _mm256_permute2f128_pd(a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 位间隔
}
// 定义一个内联函数，用于将两个包含浮点数的向量重新交错排列
inline deinterleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5, a6, b6, a7, b7}

  // 创建一个控制变量，用于分组列跨越不同的通道：
  //   a_grouped = {a0, a1, a2, a3, b0, b1, b2, b3}
  //   b_grouped = {a4, a5, a6, a7, b4, b5, b6, b7}
  const __m256i group_ctrl = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  auto a_grouped = _mm256_permutevar8x32_ps(a, group_ctrl);
  auto b_grouped = _mm256_permutevar8x32_ps(b, group_ctrl);

  // 交换通道：
  //   返回 {a0, a1, a2, a3, a4, a5, a6, a7}
  //        {b0, b1, b2, b3, b4, b5, b6, b7}
  return std::make_pair(_mm256_permute2f128_ps(a_grouped, b_grouped, 0b0100000),  // 0, 2.   相距4位
                        _mm256_permute2f128_ps(a_grouped, b_grouped, 0b0110001)); // 1, 3.   相距4位
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FLIP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 模板特化：翻转浮点数向量
template<>
inline Vectorized<float> flip(const Vectorized<float> & v) {
  const __m256i mask_float = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_permutevar8x32_ps(v, mask_float);
}

// 模板特化：翻转双精度浮点数向量
template<>
inline Vectorized<double> flip(const Vectorized<double> & v) {
  return _mm256_permute4x64_pd(v, 27);  // 27 == _MM_SHUFFLE(0, 1, 2, 3)
}

// 模板特化：翻转64位整数向量
template<>
inline Vectorized<int64_t> flip(const Vectorized<int64_t> & v) {
  return _mm256_permute4x64_epi64(v, 27);  // 27 == _MM_SHUFFLE(0, 1, 2, 3)
}

// 模板特化：翻转32位整数向量
template<>
inline Vectorized<int32_t> flip(const Vectorized<int32_t> & v) {
  const __m256i mask_int32 = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_permutevar8x32_epi32(v, mask_int32);
}

// 模板特化：翻转16位整数向量
template<>
inline Vectorized<int16_t> flip(const Vectorized<int16_t> & v) {
  const __m256i mask = _mm256_set_epi8(
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
  );
  auto reversed = _mm256_shuffle_epi8(v, mask);
  return _mm256_permute2x128_si256(reversed, reversed, 1);
}

// 翻转8位整数向量
inline __m256i flip8(const __m256i & v) {
  const __m256i mask_int8 = _mm256_set_epi8(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  );
  auto reversed = _mm256_shuffle_epi8(v, mask_int8);
  return _mm256_permute2x128_si256(reversed, reversed, 1);
}

// 模板特化：翻转8位有符号整数向量
template<>
inline Vectorized<int8_t> flip(const Vectorized<int8_t> & v) {
  return flip8(v);
}

// 模板特化：翻转8位无符号整数向量
template<>
inline Vectorized<uint8_t> flip(const Vectorized<uint8_t> & v) {
  return flip8(v);
}
```