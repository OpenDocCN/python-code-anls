# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512.h`

```py
#pragma once
// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec512/vec512_float.h>
#include <ATen/cpu/vec/vec512/vec512_bfloat16.h>
#include <ATen/cpu/vec/vec512/vec512_double.h>
#include <ATen/cpu/vec/vec512/vec512_int.h>
#include <ATen/cpu/vec/vec512/vec512_qint.h>
#include <ATen/cpu/vec/vec512/vec512_complex_float.h>
#include <ATen/cpu/vec/vec512/vec512_complex_double.h>
#include <ATen/cpu/vec/vec512/vec512_convert.h>
#include <ATen/cpu/vec/vec512/vec512_mask.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>

namespace at {
namespace vec {

// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// Output operator for c10::qint32
inline std::ostream& operator<<(std::ostream& stream, const c10::qint32& val) {
  stream << val.val_;
  return stream;
}

// Output operator for c10::qint8
inline std::ostream& operator<<(std::ostream& stream, const c10::qint8& val) {
  stream << static_cast<int>(val.val_);
  return stream;
}

// Output operator for c10::quint8
inline std::ostream& operator<<(std::ostream& stream, const c10::quint8& val) {
  stream << static_cast<unsigned int>(val.val_);
  return stream;
}

// Template specialization for outputting Vectorized<T> objects
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

// Conditionally define AVX512 specific functionalities
#if defined(CPU_CAPABILITY_AVX512)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX512) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Template specialization for casting Vectorized<double> to Vectorized<float>
template<>
inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
  return _mm512_castpd_ps(src);
}

// Template specialization for casting Vectorized<float> to Vectorized<double>
template<>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return _mm512_castps_pd(src);
}

// Template specialization for casting Vectorized<int32_t> to Vectorized<float>
template<>
inline Vectorized<float> cast<float, int32_t>(const Vectorized<int32_t>& src) {
  return _mm512_castsi512_ps(src);
}

// Template specialization for casting Vectorized<int64_t> to Vectorized<double>
template<>
inline Vectorized<double> cast<double, int64_t>(const Vectorized<int64_t>& src) {
  return _mm512_castsi512_pd(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Define gather function for double using AVX512 instructions
#ifndef _MSC_VER
template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline gather(const double* base_addr, const Vectorized<int64_t>& vindex) {
  return _mm512_i64gather_pd(vindex, base_addr, scale);
}

// Define gather function for float using AVX512 instructions
template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline gather(const float* base_addr, const Vectorized<int32_t>& vindex) {
  return _mm512_i32gather_ps(vindex, base_addr, scale);
}
#endif // _MSC_VER

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifndef _MSC_VER
// MSVC is not working well on complex function overload.
// 定义模板函数 `mask_gather`，用于根据条件进行向量化数据的收集操作
template<int64_t scale = 1>
// 如果 scale 是 1、2、4 或 8，则启用此函数，返回 Vectorized<double> 类型
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
// 内联函数，根据掩码 mask，从 base_addr 指向的内存地址中收集数据到 src 向量中
inline mask_gather(const Vectorized<double>& src, const double* base_addr,
                   const Vectorized<int64_t>& vindex, Vectorized<double>& mask) {
  // 创建一个全为 1 的掩码向量 all_ones
  auto all_ones = _mm512_castsi512_pd(_mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
  // 比较 mask.values 和 all_ones，生成掩码 mask_
  auto mask_ = _mm512_cmp_pd_mask(all_ones, mask.values, _CMP_EQ_OQ);
  // 使用掩码 mask_，按照 vindex 指定的索引，从 base_addr 中读取数据到 src 中，采用指定的数据大小 scale
  return _mm512_mask_i64gather_pd(src, mask_, vindex, base_addr, scale);
}

// 定义模板函数 `mask_gather`，用于根据条件进行向量化数据的收集操作
template<int64_t scale = 1>
// 如果 scale 是 1、2、4 或 8，则启用此函数，返回 Vectorized<float> 类型
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
// 内联函数，根据掩码 mask，从 base_addr 指向的内存地址中收集数据到 src 向量中
inline mask_gather(const Vectorized<float>& src, const float* base_addr,
                   const Vectorized<int32_t>& vindex, Vectorized<float>& mask) {
  // 创建一个全为 1 的掩码向量 all_ones
  auto all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(0xFFFFFFFF));
  // 比较 mask.values 和 all_ones，生成掩码 mask_
  auto mask_ = _mm512_cmp_ps_mask(all_ones, mask.values, _CMP_EQ_OQ);
  // 使用掩码 mask_，按照 vindex 指定的索引，从 base_addr 中读取数据到 src 中，采用指定的数据大小 scale
  return _mm512_mask_i32gather_ps(src, mask_, vindex, base_addr, scale);
}
#endif

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 特化模板函数 convert_to_int_of_same_size，将 Vectorized<double> 类型向量转换为 Vectorized<int64_t> 类型向量
template<>
Vectorized<int64_t>
inline convert_to_int_of_same_size<double>(const Vectorized<double> &src) {
  // 使用 SSE512 指令集将 src 向量中的双精度浮点数转换为对应的整数向量
  return _mm512_cvtpd_epi64(src);
}

// 特化模板函数 convert_to_int_of_same_size，将 Vectorized<float> 类型向量转换为 Vectorized<int32_t> 类型向量
template<>
Vectorized<int32_t>
inline convert_to_int_of_same_size<float>(const Vectorized<float> &src) {
  // 使用 SSE512 指令集将 src 向量中的单精度浮点数转换为对应的整数向量
  return _mm512_cvttps_epi32(src);
}

// 特化模板函数 convert_to_fp_of_same_size，将 Vectorized<int64_t> 类型向量转换为 Vectorized<double> 类型向量
template<>
Vectorized<double>
inline convert_to_fp_of_same_size<double>(const Vectorized<int64_t> &src) {
  // 使用 SSE512 指令集将 src 向量中的整数转换为对应的双精度浮点数向量
  return _mm512_cvtepi64_pd(src);
}

// 特化模板函数 convert_to_fp_of_same_size，将 Vectorized<int32_t> 类型向量转换为 Vectorized<float> 类型向量
template<>
Vectorized<float>
inline convert_to_fp_of_same_size<float>(const Vectorized<int32_t> &src) {
  // 使用 SSE512 指令集将 src 向量中的整数转换为对应的单精度浮点数向量
  return _mm512_cvtepi32_ps(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 特化模板函数 interleave2，将两个 Vectorized<double> 向量按交错顺序合并成一对向量
template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline interleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // 创建两个交错索引向量 idx1 和 idx2
  __m512i idx1 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  __m512i idx2 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  // 使用指定的索引向量 idx1 和 idx2，将向量 a 和 b 按交错顺序合并成两个结果向量
  return std::make_pair(_mm512_mask_permutex2var_pd(a, 0xff, idx1, b),
                        _mm512_mask_permutex2var_pd(a, 0xff, idx2, b));
}

// 特化模板函数 interleave2，将两个 Vectorized<float> 向量按交错顺序合并成一对向量
template <>
std::pair<Vectorized<float>, Vectorized<float>>
// 定义一个内联函数，用于将两个 Vectorized<float> 向量交错排列
inline interleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
  //
  // return:
  //   {a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7}
  //   {a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15}
  __m512i idx1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4,
                                  19, 3, 18, 2, 17, 1, 16, 0);
  __m512i idx2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12,
                                  27, 11, 26, 10, 25, 9, 24, 8);
  return std::make_pair(_mm512_mask_permutex2var_ps(a, 0xffff, idx1, b),
                        _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 模板特化，将两个 Vectorized<double> 向量解交错
template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline deinterleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5, a6, b6, a7, b7}
  // output:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7}
  //          {b0, b1, b2, b3, b4, b5, b6, b7}
  // 以二进制格式列出索引，便于理解
  __m512i idx1 = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  __m512i idx2 = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

  return std::make_pair(_mm512_mask_permutex2var_pd(a, 0xff, idx1, b),
                        _mm512_mask_permutex2var_pd(a, 0xff, idx2, b));
}

// 模板特化，将两个 Vectorized<float> 向量解交错
template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline deinterleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7}
  //   b = {a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15}
  // output:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15}
  //          {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
  __m512i idx1 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                  14, 12, 10, 8, 6, 4, 2, 0);
  __m512i idx2 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                  15, 13, 11, 9, 7, 5, 3, 1);

  return std::make_pair(_mm512_mask_permutex2var_ps(a, 0xffff, idx1, b),
                        _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FLIP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 模板特化，反转 Vectorized<float> 向量中的元素顺序
template<>
inline Vectorized<float> flip(const Vectorized<float> & v) {
  const __m512i mask = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                        8, 9, 10, 11, 12, 13, 14, 15);
  return _mm512_permutexvar_ps(mask, v);
}

// 模板特化，反转 Vectorized<double> 向量中的元素顺序
// 定义一个模板函数，用于将双精度浮点向量按指定顺序反转
inline Vectorized<double> flip(const Vectorized<double> & v) {
  // 创建一个掩码，用于指定反转顺序
  const __m512i mask = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
  // 使用掩码对向量进行反转操作，并返回结果
  return _mm512_permutexvar_pd(mask, v);
}

// 定义一个模板特化函数，用于将64位整型向量按指定顺序反转
template<>
inline Vectorized<int64_t> flip(const Vectorized<int64_t> & v) {
  // 创建一个掩码，用于指定反转顺序
  const __m512i mask = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
  // 使用掩码对向量进行反转操作，并返回结果
  return _mm512_permutexvar_epi64(mask, v);
}

// 定义一个模板特化函数，用于将32位整型向量按指定顺序反转
template<>
inline Vectorized<int32_t> flip(const Vectorized<int32_t> & v) {
  // 创建一个掩码，用于指定反转顺序
  const __m512i mask = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                        8, 9, 10, 11, 12, 13, 14, 15);
  // 使用掩码对向量进行反转操作，并返回结果
  return _mm512_permutexvar_epi32(mask, v);
}

// 定义一个模板特化函数，用于将16位整型向量按指定顺序反转
template<>
inline Vectorized<int16_t> flip(const Vectorized<int16_t> & v) {
  // 创建一个掩码，用于指定反转顺序
  const __m512i mask = _mm512_set_epi16(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
  );
  // 使用掩码对向量进行反转操作，并返回结果
  return _mm512_permutexvar_epi16(mask, v);
}

// 定义一个内联函数，用于将8位整型向量按指定顺序反转
inline __m512i flip8(const __m512i & v) {
  // 创建两个掩码，分别用于指定8位整型向量的反转顺序和对结果向量的再排列
  const __m512i mask1 = _mm512_set_epi8(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  );
  const __m512i mask2 = _mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6);
  // 使用掩码对8位整型向量进行反转和重新排列操作，并返回结果
  auto reversed_vec = _mm512_shuffle_epi8(v, mask1);
  return _mm512_permutexvar_epi64(mask2, reversed_vec);
}

// 定义一个模板特化函数，用于将8位有符号整型向量按指定顺序反转
template<>
inline Vectorized<int8_t> flip(const Vectorized<int8_t> & v) {
  // 调用flip8函数进行具体的反转操作，并返回结果
  return flip8(v);
}

// 定义一个模板特化函数，用于将8位无符号整型向量按指定顺序反转
template<>
inline Vectorized<uint8_t> flip(const Vectorized<uint8_t> & v) {
  // 调用flip8函数进行具体的反转操作，并返回结果
  return flip8(v);
}

// 结束AVX512指令集的条件编译
#endif // defined(CPU_CAPABILITY_AVX512)
```