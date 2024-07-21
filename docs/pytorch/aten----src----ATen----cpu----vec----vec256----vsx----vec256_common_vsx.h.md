# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_common_vsx.h`

```py
#pragma once
// 使用 pragma once 来确保头文件只被包含一次，以防止多重包含的问题

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>

// Note: header order is important here
// 引入向量化操作的头文件，顺序非常重要

#include <ATen/cpu/vec/vec256/vsx/vec256_double_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_float_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_int16_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_int32_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_int64_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_qint32_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_qint8_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_quint8_vsx.h>

#include <ATen/cpu/vec/vec256/vsx/vec256_complex_float_vsx.h>
#include <ATen/cpu/vec/vec256/vsx/vec256_complex_double_vsx.h>

#include <ATen/cpu/vec/vec256/vsx/vec256_bfloat16_vsx.h>

namespace at {
namespace vec {

inline namespace CPU_CAPABILITY {

// 定义向量化操作的命名空间

DEFINE_CLAMP_FUNCS(c10::quint8)
DEFINE_CLAMP_FUNCS(c10::qint8)
DEFINE_CLAMP_FUNCS(c10::qint32)
DEFINE_CLAMP_FUNCS(int16_t)
DEFINE_CLAMP_FUNCS(int32_t)
DEFINE_CLAMP_FUNCS(int64_t)
DEFINE_CLAMP_FUNCS(float)
DEFINE_CLAMP_FUNCS(double)

// 定义各种数据类型的截断函数

template <>
Vectorized<double> C10_ALWAYS_INLINE fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  // 实现双精度浮点数的向量化乘加操作
  return Vectorized<double>{
      vec_madd(a.vec0(), b.vec0(), c.vec0()),
      vec_madd(a.vec1(), b.vec1(), c.vec1())};
}

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b,
    const Vectorized<int64_t>& c) {
  // 实现64位整数的向量化乘加操作
  return Vectorized<int64_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

template <>
Vectorized<int32_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b,
    const Vectorized<int32_t>& c) {
  // 实现32位整数的向量化乘加操作
  return Vectorized<int32_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

template <>
Vectorized<int16_t> C10_ALWAYS_INLINE fmadd(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b,
    const Vectorized<int16_t>& c) {
  // 实现16位整数的向量化乘加操作
  return Vectorized<int16_t>{
      a.vec0() * b.vec0() + c.vec0(), a.vec1() * b.vec1() + c.vec1()};
}

// 定义不同类型之间的向量化数据类型转换函数

DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(float)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(double)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int64_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int32_t)
DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(int16_t)

template <>
Vectorized<int64_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<double>(const Vectorized<double>& src) {
  // 将双精度浮点数向量转换为相同大小的整数向量
  return Vectorized<int64_t>{vec_signed(src.vec0()), vec_signed(src.vec1())};
}

template <>
Vectorized<int32_t> C10_ALWAYS_INLINE
convert_to_int_of_same_size<float>(
    const Vectorized<float>& src) {
  // 将单精度浮点数向量转换为相同大小的整数向量
  return Vectorized<int32_t>{vec_signed(src.vec0()), vec_signed(src.vec1())};
}

template <>
// 将 int32_t 数组转换为 float 数组，元素数量为 n
inline void convert(const int32_t* src, float* dst, int64_t n) {
  // 使用 SIMD 向量化操作处理数组转换
  int64_t i;
  // 以 SIMD 向量的大小为步长进行循环，处理主体数据部分
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    // 指针偏移，加载输入向量数据
    const int32_t* src_a = src + i;
    float* dst_a = dst + i;
    vint32 input_vec0 = vec_vsx_ld(offset0, reinterpret_cast<const vint32*>(src_a));
    vint32 input_vec1 =
        vec_vsx_ld(offset16, reinterpret_cast<const vint32*>(src_a));
    // 转换整型向量到浮点向量
    vfloat32 c0 = vec_float(input_vec0);
    vfloat32 c1 = vec_float(input_vec1);
    // 存储浮点向量到目标数组
    vec_vsx_st(c0, offset0, dst_a);
    vec_vsx_st(c1, offset16, dst_a);
  }

  // 处理剩余的不足一个向量长度的部分
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

// 特化模板，将 int64_t 数组转换为 double 数组，元素数量为 n
template <>
inline void convert(const int64_t* src, double* dst, int64_t n) {
  int64_t i;
  // 以 SIMD 向量的大小为步长进行循环，处理主体数据部分
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    // 指针偏移，加载输入向量数据
    const int64_t* src_a = src + i;
    double* dst_a = dst + i;
    vint64 input_vec0 =
        vec_vsx_ld(offset0, reinterpret_cast<const vint64*>(src_a));
    vint64 input_vec1 =
        vec_vsx_ld(offset16, reinterpret_cast<const vint64*>(src_a));
    // 转换整型向量到双精度浮点向量
    vfloat64 c0 = vec_double(input_vec0);
    vfloat64 c1 = vec_double(input_vec1);
    // 存储双精度浮点向量到目标数组
    vec_vsx_st(c0, offset0, reinterpret_cast<double*>(dst_a));
    vec_vsx_st(c1, offset16, reinterpret_cast<double*>(dst_a));
  }
  // 处理剩余的不足一个向量长度的部分
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);
  }
}

// 通用实现以修复编译器错误
// TO-DO: 为 ppc64 添加优化版本
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(
    const Vectorized<Half>& a) {
  constexpr int64_t K = Vectorized<Half>::size();
  __at_align__ float arr[K];
  __at_align__ Half arr2[K];
  // 将向量 a 存储到 arr2 中
  a.store(arr2);
  // 将 Half 数组 arr2 转换为 float 数组 arr
  convert(arr2, arr, K);
  return std::make_tuple(
       Vectorized<float>::loadu(arr),
       Vectorized<float>::loadu(arr + Vectorized<float>::size()));
}

// 将向量 a 和 b 中的 float 元素转换为 Half 类型的向量
inline Vectorized<Half> convert_float_half(
    const Vectorized<float>& a, const Vectorized<float>& b) {
  constexpr int64_t K = Vectorized<Half>::size();
  __at_align__ float arr[K];
  __at_align__ Half arr2[K];
  // 将向量 a 存储到 arr 中，将向量 b 存储到 arr 的后半部分
  a.store(arr);
  b.store(arr + Vectorized<float>::size());
  // 将 float 数组 arr 转换为 Half 数组 arr2
  convert(arr, arr2, K);
  // 返回转换后的 Half 向量
  return Vectorized<Half>::loadu(arr2);
};

// 特化模板，交错两个双精度向量 a 和 b 的元素
template <>
std::pair<Vectorized<double>, Vectorized<double>> inline interleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  // inputs:
  //   a      = {a0, a1, a2, a3}
  //   b      = {b0, b1, b2, b3}

  // 使用 SIMD 指令交错处理双精度向量的元素
  vfloat64 ab00 = vec_xxpermdi(a.vec0(), b.vec0(), 0);
  vfloat64 ab11 = vec_xxpermdi(a.vec0(), b.vec0(), 3);
  vfloat64 ab2_00 = vec_xxpermdi(a.vec1(), b.vec1(), 0);
  vfloat64 ab2_11 = vec_xxpermdi(a.vec1(), b.vec1(), 3);
  // 返回交错处理后的双精度向量对
  //   return {a0, b0, a1, b1}
  //          {a2, b2, a3, b3}
  return std::make_pair(
      Vectorized<double>{ab00, ab11}, Vectorized<double>{ab2_00, ab2_11});
}

// 特化模板，反交错两个双精度向量 a 和 b 的元素
template <>
std::pair<Vectorized<double>, Vectorized<double>> inline deinterleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}
  // 使用 vec_xxpermdi 函数将 a 的第一个和第二个向量（vec0 和 vec1）按位置 0 进行混洗，结果存入 aa01
  vfloat64 aa01 = vec_xxpermdi(a.vec0(), a.vec1(), 0);
  // 使用 vec_xxpermdi 函数将 b 的第一个和第二个向量（vec0 和 vec1）按位置 0 进行混洗，结果存入 aa23
  vfloat64 aa23 = vec_xxpermdi(b.vec0(), b.vec1(), 0);

  // 使用 vec_xxpermdi 函数将 a 的第一个和第二个向量（vec0 和 vec1）按位置 3 进行混洗，结果存入 bb_01
  vfloat64 bb_01 = vec_xxpermdi(a.vec0(), a.vec1(), 3);
  // 使用 vec_xxpermdi 函数将 b 的第一个和第二个向量（vec0 和 vec1）按位置 3 进行混洗，结果存入 bb_23
  vfloat64 bb_23 = vec_xxpermdi(b.vec0(), b.vec1(), 3);

  // swap lanes:
  //   return {a0, a1, a2, a3}
  //          {b0, b1, b2, b3}
  // 使用 std::make_pair 创建一对 Vectorized<double> 对象，第一个对象包含 aa01 和 aa23，第二个对象包含 bb_01 和 bb_23，并返回
  return std::make_pair(
      Vectorized<double>{aa01, aa23}, Vectorized<double>{bb_01, bb_23});
}
} // 关闭 vec 命名空间

template <> // 模板特化声明，处理 float 类型
std::pair<Vectorized<float>, Vectorized<float>> inline interleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  // 输入：
  //   a = {a0, a1, a2, a3,, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3,, b4, b5, b6, b7}

  // 将 a 的前四个元素和 b 的前四个元素交错合并
  vfloat32 ab0011 = vec_mergeh(a.vec0(), b.vec0());
  // 将 a 的后四个元素和 b 的后四个元素交错合并
  vfloat32 ab2233 = vec_mergel(a.vec0(), b.vec0());

  // 将 a 的第二组前四个元素和 b 的第二组前四个元素交错合并
  vfloat32 ab2_0011 = vec_mergeh(a.vec1(), b.vec1());
  // 将 a 的第二组后四个元素和 b 的第二组后四个元素交错合并
  vfloat32 ab2_2233 = vec_mergel(a.vec1(), b.vec1());
  // 分组列跨越通道：
  //   返回 {a0, b0, a1, b1,, a2, b2, a3, b3}
  //        {a4, b4, a5, b5,, a6, b6, a7, b7}

  return std::make_pair(
      Vectorized<float>{ab0011, ab2233}, Vectorized<float>{ab2_0011, ab2_2233});
}

template <> // 模板特化声明，处理 float 类型
std::pair<Vectorized<float>, Vectorized<float>> inline deinterleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  // 输入：
  //   a = {a0, b0, a1, b1,, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5,, a6, b6, a7, b7}

  // {a0,a2,b0,b2} {a1,a3,b1,b3} 的交错合并
  vfloat32 a0a2b0b2 = vec_mergeh(a.vec0(), a.vec1());
  vfloat32 a1a3b1b3 = vec_mergel(a.vec0(), a.vec1());

  // 对 {a0,a2,b0,b2} {a1,a3,b1,b3} 再次交错合并
  vfloat32 aa0123 = vec_mergeh(a0a2b0b2, a1a3b1b3);
  vfloat32 bb0123 = vec_mergel(a0a2b0b2, a1a3b1b3);

  // 第二组数据的交错合并
  vfloat32 a0a2b0b2_2 = vec_mergeh(b.vec0(), b.vec1());
  vfloat32 a1a3b1b3_2 = vec_mergel(b.vec0(), b.vec1());

  // 对第二组数据的 {a0,a2,b0,b2} {a1,a3,b1,b3} 再次交错合并
  vfloat32 aa0123_2 = vec_mergeh(a0a2b0b2_2, a1a3b1b3_2);
  vfloat32 bb0123_2 = vec_mergel(a0a2b0b2_2, a1a3b1b3_2);

  // 可以使用 vec_perm 进行交换通道：
  //   返回 {a0, a1, a2, a3,, a4, a5, a6, a7}
  //        {b0, b1, b2, b3,, b4, b5, b6, b7}

  return std::make_pair(
      Vectorized<float>{aa0123, aa0123_2}, Vectorized<float>{bb0123, bb0123_2});
}

} // 关闭 at 命名空间
} // 关闭 vec 命名空间
} // 关闭 namespace 命名空间
```