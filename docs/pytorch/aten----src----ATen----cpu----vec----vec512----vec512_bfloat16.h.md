# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_bfloat16.h`

```py
#pragma once
// 防止头文件被多次包含

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// 不要在这个头文件中定义静态数据！
// See Note [Do not compile initializers with AVX]
// 参见注释[不要使用 AVX 编译初始化器]

#include <ATen/cpu/vec/intrinsics.h>
// 包含向量化指令集的头文件
#include <ATen/cpu/vec/vec_base.h>
// 包含向量化基础函数的头文件
#include <c10/util/irange.h>
// 包含用于范围迭代的头文件

#if defined(CPU_CAPABILITY_AVX512)
// 如果支持 AVX512 指令集

#define SLEEF_STATIC_LIBS
// 定义 SLEEF_STATIC_LIBS 用于静态链接 Sleef 库
#include <sleef.h>
// 包含 Sleef 数学库的头文件
#endif

namespace at {
namespace vec {
// 命名空间 at::vec::

// See Note [CPU_CAPABILITY namespace]
// 参见注释[CPU_CAPABILITY 命名空间]
inline namespace CPU_CAPABILITY {
// 内联命名空间 CPU_CAPABILITY

#if defined(CPU_CAPABILITY_AVX512)
// 如果支持 AVX512 指令集

#ifndef SLEEF_CONST
// 如果未定义 SLEEF_CONST

#if (defined(__GNUC__) || defined(__CLANG__)) && !defined(__INTEL_COMPILER)
#define SLEEF_CONST const
// 如果是 GCC 或者 Clang 编译器，定义 SLEEF_CONST 为 const
#else
#define SLEEF_CONST
// 否则为空
#endif

#define SLEEF_CONST_OLD SLEEF_CONST
// 定义 SLEEF_CONST_OLD 为 SLEEF_CONST
#else
#define SLEEF_CONST_OLD
// 否则定义为空
#endif

// bfloat16 conversion
// bfloat16 转换函数
static inline void cvtbf16_fp32(const __m256i& a, __m512& o) {
  o = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16));
  // 将输入的 16 位无符号整数向左移动 16 位，然后转换为 512 位的单精度浮点数
}

// bfloat16 conversion overload
// bfloat16 转换函数的重载
static inline void cvtbf16_fp32(const __m512i& a, __m512& o1, __m512& o2) {
  __m256i lo = _mm512_extracti32x8_epi32(a, 0);
  // 提取 a 的低 256 位整数
  __m256i hi = _mm512_extracti32x8_epi32(a, 1);
  // 提取 a 的高 256 位整数
  cvtbf16_fp32(lo, o1);
  // 调用 cvtbf16_fp32 转换低位数值
  cvtbf16_fp32(hi, o2);
  // 调用 cvtbf16_fp32 转换高位数值
}

// Single precision to bfloat16 conversion
// 单精度浮点数到 bfloat16 转换
static inline __m256i cvtfp32_bf16(const __m512& src) {
  __m512i value = _mm512_castps_si512(src);
  // 将单精度浮点数 src 转换为 512 位整数
  __m512i nan = _mm512_set1_epi32(0xffff);
  // 设置一个 512 位整数，每个 32 位都是 0xffff
  auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
  // 用于比较 src 与自身是否有序，生成掩码
  __m512i ones = _mm512_set1_epi32(0x1);
  // 设置一个 512 位整数，每个 32 位都是 0x1
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // 设置一个 512 位整数，每个 32 位都是 0x7fff
  // uint32_t lsb = (input >> 16) & 1;
  // 计算 lsb
  auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
  // 计算 t_value
  // uint32_t rounding_bias = 0x7fff + lsb;
  // 计算 rounding_bias
  t_value = _mm512_add_epi32(t_value, vec_bias);
  // 计算 t_value
  // input += rounding_bias;
  // 计算 input
  t_value = _mm512_add_epi32(t_value, value);
  // 计算 t_value
  // input = input >> 16;
  // 计算 input
  t_value = _mm512_srli_epi32(t_value, 16);
  // 计算 t_value
  // Check NaN before converting back to bf16
  // 在转换回 bfloat16 之前检查 NaN
  t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
  // 混合 t_value 和 nan 根据 mask_value
  return _mm512_cvtusepi32_epi16(t_value);
  // 将 t_value 转换为 bfloat16
}
// 将两个 __m512 类型的参数 a 和 b 转换为 __m512i 类型
static inline __m512i cvtfp32_bf16(const __m512& a, const __m512& b) {
  // 将 a 和 b 分别转换为 __m512i 类型，即将浮点数向量转换为整数向量
  __m512i lo = _mm512_castps_si512(a);
  __m512i hi = _mm512_castps_si512(b);

  // 创建一个全为 0xffff 的向量，用于标记 NaN
  __m512i nan = _mm512_set1_epi32(0xffff);

  // 检查 a 和 b 是否是有序的（不是 NaN），生成掩码
  auto mask_lo = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
  auto mask_hi = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);

  // 创建一个全为 1 的向量
  __m512i ones = _mm512_set1_epi32(0x1);

  // 创建一个全为 0x7fff 的向量，用于偏置调整
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);

  // 提取 lo 和 hi 向量的低 16 位，并与 ones 向量进行按位与操作
  auto t_lo = _mm512_and_si512(_mm512_srli_epi32(lo, 16), ones);
  auto t_hi = _mm512_and_si512(_mm512_srli_epi32(hi, 16), ones);

  // 将 t_lo 和 t_hi 向量加上 vec_bias 向量
  t_lo = _mm512_add_epi32(t_lo, vec_bias);
  t_hi = _mm512_add_epi32(t_hi, vec_bias);

  // 将 lo 和 hi 向量与 t_lo 和 t_hi 向量相加
  t_lo = _mm512_add_epi32(t_lo, lo);
  t_hi = _mm512_add_epi32(t_hi, hi);

  // 将 t_lo 和 t_hi 向量右移 16 位
  t_lo = _mm512_srli_epi32(t_lo, 16);
  t_hi = _mm512_srli_epi32(t_hi, 16);

  // 根据 mask_lo 和 mask_hi，将 t_lo 和 t_hi 向量与 nan 向量进行混合
  t_lo = _mm512_mask_blend_epi32(mask_lo, nan, t_lo);
  t_hi = _mm512_mask_blend_epi32(mask_hi, nan, t_hi);

  // 将 t_lo 和 t_hi 向量打包成一个新的 __m512i 向量，并按照指定索引 idx 进行重新排列
  t_lo = _mm512_packus_epi32(t_lo, t_hi);
  __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx, t_lo);
}

// 合并比较结果
static inline __m512i merge_compare_result(const __m512& a, const __m512& b) {
  // 将 a 和 b 转换为整数向量 lo 和 hi，并将它们右移 16 位
  __m512i lo = _mm512_castps_si512(a);
  __m512i hi = _mm512_castps_si512(b);
  lo = _mm512_srli_epi32(lo, 16);
  hi = _mm512_srli_epi32(hi, 16);

  // 将 lo 和 hi 向量打包成一个新的 __m512i 向量，并按照指定索引 idx 进行重新排列
  auto out = _mm512_packus_epi32(lo, hi);
  __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx, out);
}

// 将 __m256i 类型的参数 a 转换为 __m512 类型的浮点数向量 o
// 使用 _mm512_cvtph_ps 函数进行转换
static inline void cvtfp16_fp32(const __m256i& a, __m512& o) {
  o = _mm512_cvtph_ps(a);
}

// 将 __m512i 类型的参数 a 分别转换为两个 __m512 类型的浮点数向量 o1 和 o2
// 调用 cvtfp16_fp32 函数进行转换
static inline void cvtfp16_fp32(const __m512i& a, __m512& o1, __m512& o2) {
  // 从 a 向量中提取两个 __m256i 类型的向量 lo 和 hi
  __m256i lo = _mm512_extracti32x8_epi32(a, 0);
  __m256i hi = _mm512_extracti32x8_epi32(a, 1);
  // 分别调用 cvtfp16_fp32 函数，将 lo 和 hi 转换为 o1 和 o2
  cvtfp16_fp32(lo, o1);
  cvtfp16_fp32(hi, o2);
}

// 将 __m512 类型的浮点数向量 src 转换为 __m256i 类型的半精度浮点数向量
static inline __m256i cvtfp32_fp16(const __m512& src) {
  // 使用 _mm512_cvtps_ph 函数将 src 向量转换为半精度浮点数向量
  return _mm512_cvtps_ph(
      src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// 将两个 __m512 类型的参数 a 和 b 分别转换为 __m256i 类型的半精度浮点数向量
// 调用 _mm512_cvtps_ph 函数进行转换，并将结果合并为一个 __m512i 类型的向量
static inline __m512i cvtfp32_fp16(const __m512& a, const __m512& b) {
  // 分别将 a 和 b 转换为半精度浮点数向量 lo 和 hi
  __m256i lo = _mm512_cvtps_ph(
      a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  __m256i hi = _mm512_cvtps_ph(
      b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  
  // 将 lo 和 hi 转换为 __m512 类型的浮点数向量 t_lo 和 t_hi
  __m512 t_lo = _mm512_castsi512_ps(_mm512_castsi256_si512(lo));
  __m256 t_hi = _mm256_castsi256_ps(hi);

  // 将 t_hi 插入到 t_lo 中，返回结果作为 __m512i 类型的向量
  return _mm512_castps_si512(_mm512_insertf32x8(t_lo, t_hi, 1));
}

// 根据模板参数 T，将 __m256i 类型的参数 a 转换为 __m512 类型的浮点数向量 o
// 对于模板参数为 BFloat16 或 Half 的情况，调用对应的转换函数
template <typename T, typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline void cvt_to_fp32(const __m256i& a, __m512& o);

// 对于模板参数为 BFloat16 的情况，调用 cvtbf16_fp32 函数进行转换
template <> inline void cvt_to_fp32<BFloat16>(const __m256i& a, __m512& o) {
  cvtbf16_fp32(a, o);
}

// 对于模板参数为 Half 的情况，调用 cvtfp16_fp32 函数进行转换
template <> inline void cvt_to_fp32<Half>(const __m256i& a, __m512& o) {
  cvtfp16_fp32(a, o);
}
// 定义一个模板函数 cvt_to_fp32，用于将输入 __m512i 转换为 __m512 浮点数向量
template <typename T, typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline void cvt_to_fp32(const __m512i& a, __m512& o1, __m512& o2);

// 专门为 BFloat16 类型实现 cvt_to_fp32 函数，调用 cvtbf16_fp32 进行转换
template <> inline void cvt_to_fp32<BFloat16>(const __m512i& a, __m512& o1, __m512& o2) {
  cvtbf16_fp32(a, o1, o2);
}

// 专门为 Half 类型实现 cvt_to_fp32 函数，调用 cvtfp16_fp32 进行转换
template <> inline void cvt_to_fp32<Half>(const __m512i& a, __m512& o1, __m512& o2) {
  cvtfp16_fp32(a, o1, o2);
}

// 定义一个模板函数 cvt_from_fp32，用于将输入 __m512 浮点数向量转换为 __m512i
template <typename T, bool is_compare_op = false,
          typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline __m512i cvt_from_fp32(const __m512& a, const __m512& b);

// 为 BFloat16 类型实现 cvt_from_fp32 函数，调用 cvtfp32_bf16 进行转换
template <> inline __m512i cvt_from_fp32<BFloat16, false>(const __m512& a, const __m512& b) {
  return cvtfp32_bf16(a, b);
}

// 为 BFloat16 类型实现 cvt_from_fp32 函数，调用 merge_compare_result 进行比较结果合并
template <> inline __m512i cvt_from_fp32<BFloat16, true>(const __m512& a, const __m512& b) {
  return merge_compare_result(a, b);
}

// 为 Half 类型实现 cvt_from_fp32 函数，调用 cvtfp32_fp16 进行转换
template <> inline __m512i cvt_from_fp32<Half, false>(const __m512& a, const __m512& b) {
  return cvtfp32_fp16(a, b);
}

// 为 Half 类型实现 cvt_from_fp32 函数，调用 cvtfp32_fp16 进行转换
template <> inline __m512i cvt_from_fp32<Half, true>(const __m512& a, const __m512& b) {
  return cvtfp32_fp16(a, b);
}

// 定义一个模板类 Vectorized16，支持模板类型 T 为 float16 或 bfloat16
template <typename T>
class Vectorized16 {
  static_assert(
    is_reduced_floating_point_v<T>,
    "Support only float16 and bfloat16.");

private:
  __m512i values; // 私有成员变量，存储 __m512i 向量值

public:
  using value_type = uint16_t;
  using size_type = int;

  // 静态成员函数，返回向量大小为 32
  static constexpr size_type size() {
    return 32;
  }

  // 默认构造函数
  Vectorized16() {}

  // 构造函数，接受 __m512i 作为参数
  Vectorized16(__m512i v) : values(v) {}

  // 构造函数，接受 T 类型的值作为参数，将其转换为 __m512i
  Vectorized16(T val) {
    value_type uw = val.x;
    values = _mm512_set1_epi16(uw);
  }

  // 多参数构造函数，将 T 类型的值按顺序转换为 __m512i
  Vectorized16(T val1, T val2, T val3, T val4,
              T val5, T val6, T val7, T val8,
              T val9, T val10, T val11, T val12,
              T val13, T val14, T val15, T val16,
              T val17, T val18, T val19, T val20,
              T val21, T val22, T val23, T val24,
              T val25, T val26, T val27, T val28,
              T val29, T val30, T val31, T val32) {
    values = _mm512_set_epi16(
        val32.x, val31.x, val30.x, val29.x, val28.x, val27.x, val26.x, val25.x,
        val24.x, val23.x, val22.x, val21.x, val20.x, val19.x, val18.x, val17.x,
        val16.x, val15.x, val14.x, val13.x, val12.x, val11.x, val10.x, val9.x,
        val8.x, val7.x, val6.x, val5.x, val4.x, val3.x, val2.x, val1.x);
  }

  // 类型转换操作符，返回 __m512i 值
  operator __m512i() const {
    return values;
  }

  // 禁用索引操作符 []，确保不支持通过索引访问
  T& operator[](int idx) = delete;
  const T& operator[](int idx) const  = delete;

  // 返回一个整数掩码，其中所有零元素转换为 1 位，其他转换为 0 位
  int zero_mask() const {
    return _mm512_cmpeq_epi16_mask(values, _mm512_set1_epi16(0));
  }

  // 静态成员函数，加载非对齐数据指针到 Vectorized<T> 对象，可选加载元素个数
  static Vectorized<T> loadu(const void* ptr, int16_t count = size()) {
    if (count == size())
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));

    __mmask32 mask = (1ULL << count) - 1;
    return _mm512_maskz_loadu_epi16(mask, ptr);
  }

  // 存储向量值到指定地址，可选存储元素个数
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);

    // 将向量值存储到指定地址
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);



    // 如果存储元素个数不是默认大小，使用掩码加载并存储数据
    _mm512_maskz_loadu_epi16(mask, ptr);



  }
};
    } else if (count > 0) {
      // 计算掩码，将 count 位设置为1
      __mmask32 mask = (1ULL << count) - 1;
      // 使用掩码将 values 中的数据存储到 ptr 指向的内存中
      _mm512_mask_storeu_epi16(ptr, mask, values);
    }
  }
  // 根据给定的掩码 mask，对两个 Vectorized 对象 a 和 b 进行混合操作
  template <int64_t mask>
  static Vectorized<T> blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    // 创建临时数组用于存储数据
    __at_align__ int16_t tmp_values[size()];
    // 将向量 a 中的数据存储到临时数组 tmp_values 中
    a.store(tmp_values);
    // 根据掩码 mask 的每一位进行条件混合操作
    if (mask & 0x01)
      tmp_values[0] = b.values[31];
    if (mask & 0x02)
      tmp_values[1] = b.values[30];
    if (mask & 0x04)
      tmp_values[2] = b.values[29];
    if (mask & 0x08)
      tmp_values[3] = b.values[28];
    if (mask & 0x10)
      tmp_values[4] = b.values[27];
    if (mask & 0x20)
      tmp_values[5] = b.values[26];
    if (mask & 0x40)
      tmp_values[6] = b.values[25];
    if (mask & 0x80)
      tmp_values[7] = b.values[24];
    if (mask & 0x100)
      tmp_values[8] = b.values[23];
    if (mask & 0x200)
      tmp_values[9] = b.values[22];
    if (mask & 0x400)
      tmp_values[10] = b.values[21];
    if (mask & 0x800)
      tmp_values[11] = b.values[20];
    if (mask & 0x1000)
      tmp_values[12] = b.values[19];
    if (mask & 0x2000)
      tmp_values[13] = b.values[18];
    if (mask & 0x4000)
      tmp_values[14] = b.values[17];
    if (mask & 0x8000)
      tmp_values[15] = b.values[16];
    if (mask & 0x10000)
      tmp_values[16] = b.values[15];
    if (mask & 0x20000)
      tmp_values[17] = b.values[14];
    if (mask & 0x40000)
      tmp_values[18] = b.values[13];
    if (mask & 0x80000)
      tmp_values[19] = b.values[12];
    if (mask & 0x100000)
      tmp_values[20] = b.values[11];
    if (mask & 0x200000)
      tmp_values[21] = b.values[10];
    if (mask & 0x400000)
      tmp_values[22] = b.values[9];
    if (mask & 0x800000)
      tmp_values[23] = b.values[8];
    if (mask & 0x1000000)
      tmp_values[24] = b.values[7];
    if (mask & 0x2000000)
      tmp_values[25] = b.values[6];
    if (mask & 0x4000000)
      tmp_values[26] = b.values[5];
    if (mask & 0x8000000)
      tmp_values[27] = b.values[4];
    if (mask & 0x10000000)
      tmp_values[28] = b.values[3];
    if (mask & 0x20000000)
      tmp_values[29] = b.values[2];
    if (mask & 0x40000000)
      tmp_values[30] = b.values[1];
    if (mask & 0x80000000)
      tmp_values[31] = b.values[0];
    // 加载临时数组的数据并返回一个新的 Vectorized 对象
    return loadu(tmp_values);
  }
  // 根据掩码 mask 对两个 Vectorized 对象 a 和 b 进行条件混合操作
  static Vectorized<T> blendv(const Vectorized<T>& a,
      const Vectorized<T>& b, const Vectorized<T>& mask) {
    // 创建一个所有位均为1的向量
    auto all_ones = _mm512_set1_epi16(0xFFFF);
    // 比较 mask 和 all_ones，生成掩码
    auto mask_ = _mm512_cmp_epi16_mask(mask, all_ones, _MM_CMPINT_EQ);
    // 使用掩码将 a 和 b 的数据进行混合，并返回结果向量
    return _mm512_mask_blend_epi16(mask_, a.values, b.values);
  }
  // 根据指定的步长 step_t 和起始值 base，生成一个序列向量
  template<typename step_t>
  static Vectorized<T> arange(T base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
      base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
      base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
      base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
      base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step);
  }


  // 返回一个 Vectorized<T> 对象，其中包含从 base 开始，每隔 step 递增的一系列值
  static Vectorized<T> set(const Vectorized<T>& a,
      const Vectorized<T>& b, int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
      case 8:
        return blend<255>(a, b);
      case 9:
        return blend<511>(a, b);
      case 10:
        return blend<1023>(a, b);
      case 11:
        return blend<2047>(a, b);
      case 12:
        return blend<4095>(a, b);
      case 13:
        return blend<8191>(a, b);
      case 14:
        return blend<16383>(a, b);
      case 15:
        return blend<32767>(a, b);
      case 16:
        return blend<65535>(a, b);
      case 17:
        return blend<131071>(a, b);
      case 18:
        return blend<262143>(a, b);
      case 19:
        return blend<524287>(a, b);
      case 20:
        return blend<1048575>(a, b);
      case 21:
        return blend<2097151>(a, b);
      case 22:
        return blend<4194303>(a, b);
      case 23:
        return blend<8388607>(a, b);
      case 24:
        return blend<16777215>(a, b);
      case 25:
        return blend<33554431>(a, b);
      case 26:
        return blend<67108863>(a, b);
      case 27:
        return blend<134217727>(a, b);
      case 28:
        return blend<268435455>(a, b);
      case 29:
        return blend<536870911>(a, b);
      case 30:
        return blend<1073741823>(a, b);
      case 31:
        return blend<2147483647>(a, b);
    }
    return b;
  }


  // 返回两个 Vectorized<T> 对象按位混合后的结果，混合的位数由 count 决定
  // 如果 count 为 0，则返回 a
  static Vectorized<T> set(const Vectorized<T>& a,
      const Vectorized<T>& b, int64_t count = size()) {
    switch (count) {
      // 混合的位数为 1 时，返回 blend<1>(a, b)
      case 1:
        return blend<1>(a, b);
      // 混合的位数为 2 时，返回 blend<3>(a, b)
      case 2:
        return blend<3>(a, b);
      // 以此类推，根据 count 的不同返回对应位数的混合结果
      // ...
      // 当 count 超出已定义的范围时，返回 b
    }
    return b;
  }


  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wignored-qualifiers"

  Vectorized<T> map(SLEEF_CONST __m512 (*SLEEF_CONST_OLD vop)(__m512)) const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    const auto o1 = vop(lo);
    const auto o2 = vop(hi);
    return cvt_from_fp32<T>(o1, o2);
  }


  // 对当前 Vectorized<T> 对象的每个元素应用函数 vop
  Vectorized<T> map(SLEEF_CONST __m512 (*SLEEF_CONST_OLD vop)(__m512)) const {
    // 将当前对象的值转换为 __m512 类型的 lo 和 hi
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    // 分别将 lo 和 hi 应用函数 vop，得到结果 o1 和 o2
    const auto o1 = vop(lo);
    const auto o2 = vop(hi);
    // 将 o1 和 o2 转换回当前对象的类型 T，并返回结果
    return cvt_from_fp32<T>(o1, o2);
  }


  Vectorized<T> isnan() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    __mmask16 lo_mask, hi_mask;
    __m512 zero = _mm512_set1_ps(0.0);
    __m512i zeroi = _mm512_castps_si512(zero);
    // 比较 lo 中每个元素是否为 NaN，得到掩码 lo_mask
    lo_mask = _mm512_cmp_ps_mask(lo, zero, _CMP_UNORD_Q);
  Vectorized<T> abs() const {
    // 返回当前向量的绝对值，通过清除符号位来实现
    return _mm512_andnot_si512(_mm512_set1_epi16(0x8000), values);
  }

  Vectorized<T> angle() const {
    __m512 lo, hi;
    // 将当前向量转换为单精度浮点数存储在lo和hi中
    cvt_to_fp32<T>(values, lo, hi);

    // 定义计算角度的lambda函数
    auto angle_lambda = [](__m512 values) {
      const auto zero_vec = _mm512_set1_ps(0.f);
      const auto nan_vec = _mm512_set1_ps(NAN);

      // 检查值是否为非NaN
      const auto not_nan_mask = _mm512_cmp_ps_mask(values, values, _CMP_EQ_OQ);
      // 将非NaN位置1，其余位置0
      const auto non_nan_mask_vec = _mm512_mask_set1_epi32(_mm512_castps_si512(zero_vec),
                                                           not_nan_mask, 0xFFFFFFFF);
      // 检查是否为NaN
      const auto nan_mask = _mm512_cmp_ps_mask(_mm512_castsi512_ps(non_nan_mask_vec),
                                               zero_vec, _CMP_EQ_OQ);
      const auto pi = _mm512_set1_ps(c10::pi<float>);

      // 检查是否为负数
      const auto neg_mask = _mm512_cmp_ps_mask(values, zero_vec, _CMP_LT_OQ);
      // 根据条件混合选择角度
      auto angle = _mm512_mask_blend_ps(neg_mask, zero_vec, pi);
      // 根据NaN掩码选择角度
      angle = _mm512_mask_blend_ps(nan_mask, angle, nan_vec);
      return angle;
    };

    // 计算lo和hi的角度
    auto o1 = angle_lambda(lo);
    auto o2 = angle_lambda(hi);
    // 将计算得到的角度转换回向量类型
    return cvt_from_fp32<T>(o1, o2);
  }

  Vectorized<T> real() const {
    // 返回当前向量本身，因为这是实部函数
    return *this;
  }

  Vectorized<T> imag() const {
    // 返回一个向量，所有元素都是0，因为这是虚部函数
    return _mm512_set1_epi16(0);
  }

  Vectorized<T> conj() const {
    // 返回当前向量本身，因为这是共轭函数
    return *this;
  }

  Vectorized<T> acos() const {
    // 使用Sleef库函数计算反余弦值
    return map(Sleef_acosf16_u10);
  }

  Vectorized<T> acosh() const {
    // 使用Sleef库函数计算反双曲余弦值
    return map(Sleef_acoshf16_u10);
  }

  Vectorized<T> asin() const {
    // 使用Sleef库函数计算反正弦值
    return map(Sleef_asinf16_u10);
  }

  Vectorized<T> atan() const {
    // 使用Sleef库函数计算反正切值
    return map(Sleef_atanf16_u10);
  }

  Vectorized<T> atanh() const {
    // 使用Sleef库函数计算反双曲正切值
    return map(Sleef_atanhf16_u10);
  }

  Vectorized<T> atan2(const Vectorized<T> &b) const {
    __m512 lo, hi;
    __m512 b1, b2;
    // 将当前向量和向量b转换为单精度浮点数
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    // 使用Sleef库函数计算atan2(lo, b1)和atan2(hi, b2)，然后转换回向量类型
    auto o1 = Sleef_atan2f16_u10(lo, b1);
    auto o2 = Sleef_atan2f16_u10(hi, b2);
    return cvt_from_fp32<T>(o1, o2);
  }

  Vectorized<T> copysign(const Vectorized<T> &sign) const {
    // 从sign向量复制符号位（0x8000），并将其余位从当前向量values中复制
    __m512i mask_value = _mm512_set1_epi32(~0x80008000);
    __m512i mask_signbit = _mm512_set1_epi32(0x80008000);
    return Vectorized<T>(
      _mm512_or_si512(
        _mm512_and_si512(values, mask_value),
        _mm512_and_si512(sign, mask_signbit)));
  }

  Vectorized<T> erf() const {
    // 使用Sleef库函数计算误差函数值
    return map(Sleef_erff16_u10);
  }

  Vectorized<T> erfc() const {
    // 使用Sleef库函数计算余误差函数值
    return map(Sleef_erfcf16_u15);
  }

  Vectorized<T> erfinv() const {
    __m512 lo, hi;
    // 将当前向量转换为单精度浮点数存储在lo和hi中
    cvt_to_fp32<T>(values, lo, hi);

    // 创建临时数组来存储转换后的值
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
  Vectorized<T> igamma(const Vectorized<T> &x) const {
    // 将当前对象的值转换为单精度浮点数存储在 lo 和 hi 中
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    // 将参数 x 对象的值转换为单精度浮点数存储在 xlo 和 xhi 中
    __m512 xlo, xhi;
    cvt_to_fp32<T>(x.values, xlo, xhi);
    // 创建用于临时存储计算结果的数组 tmp1 和 tmp2
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    // 将 lo 中的数据存储到 tmp1 中
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    // 将 hi 中的数据存储到 tmp2 中
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

    // 遍历 tmp1 和 tmp2 中的数据，分别计算 igamma 函数值
    for (auto i = decltype(size()){0}; i < size() / 2; i++) {
      tmp1[i] = calc_igamma(tmp1[i]);
      tmp2[i] = calc_igamma(tmp2[i]);
    }

    // 将 tmp1 和 tmp2 中的数据加载回 o1 和 o2 中
    const auto o1 = _mm512_loadu_ps(tmp1);
    const auto o2 = _mm512_loadu_ps(tmp2);

    // 将 o1 和 o2 中的数据转换回原始类型 T，返回结果向量
    return cvt_from_fp32<T>(o1, o2);
  }
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    // 将 lo 中的 16 个单精度浮点数存储到 tmp1 数组中
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    // 将 hi 中的 16 个单精度浮点数存储到 tmp2 数组中
    __at_align__ float tmpx1[size() / 2], tmpx2[size() / 2];
    // 定义大小为 size() / 2 的 float 数组 tmpx1 和 tmpx2，并进行内存对齐
    _mm512_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
    // 将 xlo 中的 16 个单精度浮点数存储到 tmpx1 数组中
    _mm512_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
    // 将 xhi 中的 16 个单精度浮点数存储到 tmpx2 数组中
    for (int64_t i = 0; i < size() / 2; ++i) {
      // 遍历数组 tmp1 和 tmpx1 中的元素，计算对应位置的 igamma
      tmp1[i] = calc_igamma(tmp1[i], tmpx1[i]);
      // 遍历数组 tmp2 和 tmpx2 中的元素，计算对应位置的 igamma
      tmp2[i] = calc_igamma(tmp2[i], tmpx2[i]);
    }
    auto o1 = _mm512_loadu_ps(tmp1);
    // 从 tmp1 数组加载 16 个单精度浮点数到 o1 向量
    auto o2 = _mm512_loadu_ps(tmp2);
    // 从 tmp2 数组加载 16 个单精度浮点数到 o2 向量
    return cvt_from_fp32<T>(o1, o2);
    // 将 o1 和 o2 向量转换为指定类型 T 的 Vectorized 并返回
  }

  Vectorized<T> igammac(const Vectorized<T> &x) const {
    __m512 lo, hi;
    // 将当前对象 values 转换为 lo 和 hi 两个 __m512 向量
    cvt_to_fp32<T>(values, lo, hi);
    __m512 xlo, xhi;
    // 将 x.values 转换为 xlo 和 xhi 两个 __m512 向量
    cvt_to_fp32<T>(x.values, xlo, xhi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    // 定义大小为 size() / 2 的 float 数组 tmp1 和 tmp2，并进行内存对齐
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    // 将 lo 中的 16 个单精度浮点数存储到 tmp1 数组中
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    // 将 hi 中的 16 个单精度浮点数存储到 tmp2 数组中
    __at_align__ float tmpx1[size() / 2], tmpx2[size() / 2];
    // 定义大小为 size() / 2 的 float 数组 tmpx1 和 tmpx2，并进行内存对齐
    _mm512_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
    // 将 xlo 中的 16 个单精度浮点数存储到 tmpx1 数组中
    _mm512_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
    // 将 xhi 中的 16 个单精度浮点数存储到 tmpx2 数组中
    for (int64_t i = 0; i < size() / 2; ++i) {
      // 遍历数组 tmp1 和 tmpx1 中的元素，计算对应位置的 igammac
      tmp1[i] = calc_igammac(tmp1[i], tmpx1[i]);
      // 遍历数组 tmp2 和 tmpx2 中的元素，计算对应位置的 igammac
      tmp2[i] = calc_igammac(tmp2[i], tmpx2[i]);
    }
    auto o1 = _mm512_loadu_ps(tmp1);
    // 从 tmp1 数组加载 16 个单精度浮点数到 o1 向量
    auto o2 = _mm512_loadu_ps(tmp2);
    // 从 tmp2 数组加载 16 个单精度浮点数到 o2 向量
    return cvt_from_fp32<T>(o1, o2);
    // 将 o1 和 o2 向量转换为指定类型 T 的 Vectorized 并返回
  }
    auto o1 = _mm512_roundscale_ps(lo, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    auto o2 = _mm512_roundscale_ps(hi, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> lgamma() const {
    return map(Sleef_lgammaf16_u10);
  }
  Vectorized<T> sqrt() const {
    // 将当前对象中的值转换为单精度浮点数，分别存储在 lo 和 hi 中
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    // 计算 lo 和 hi 中每个元素的平方根
    auto o1 = _mm512_sqrt_ps(lo);
    auto o2 = _mm512_sqrt_ps(hi);
    // 将计算结果转换回原始数据类型 T，并返回
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> reciprocal() const {
    // 将当前对象中的值转换为单精度浮点数，分别存储在 lo 和 hi 中
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    // 创建一个全为 1 的向量
    auto ones = _mm512_set1_ps(1);
    // 计算 lo 和 hi 中每个元素的倒数
    auto o1 = _mm512_div_ps(ones, lo);
    auto o2 = _mm512_div_ps(ones, hi);
    // 将计算结果转换回原始数据类型 T，并返回
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> rsqrt() const {
    // 将当前对象中的值转换为单精度浮点数，分别存储在 lo 和 hi 中
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    // 创建一个全为 1 的向量
    auto ones = _mm512_set1_ps(1);
    // 计算 lo 和 hi 中每个元素的平方根的倒数
    auto o1 = _mm512_div_ps(ones, _mm512_sqrt_ps(lo));
    auto o2 = _mm512_div_ps(ones, _mm512_sqrt_ps(hi));
    // 将计算结果转换回原始数据类型 T，并返回
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> pow(const Vectorized<T> &b) const {
    // 将当前对象中的值转换为单精度浮点数，分别存储在 lo 和 hi 中
    __m512 lo, hi;
    __m512 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    // 使用 Sleef 库中的函数计算 lo^b1 和 hi^b2
    auto o1 = Sleef_powf16_u10(lo, b1);
    auto o2 = Sleef_powf16_u10(hi, b2);
    // 将计算结果转换回原始数据类型 T，并返回
    return cvt_from_fp32<T>(o1, o2);
  }
private:
  // 定义一个模板函数，用于进行二进制比较操作，返回一个 Vectorized<T> 对象
  template<typename Op>
  Vectorized<T> inline binary_compare(const Vectorized<T>& b, Op op) const {
    // 定义 SIMD 寄存器用于存储当前对象和参数对象的数据
    __m512 a_lo, a_hi;
    __m512 b_lo, b_hi;
    // 将当前对象的数据转换为单精度浮点数，并存储到相应的 SIMD 寄存器中
    cvt_to_fp32<T>(values, a_lo, a_hi);
    // 将参数对象的数据转换为单精度浮点数，并存储到相应的 SIMD 寄存器中
    cvt_to_fp32<T>(b.values, b_lo, b_hi);
    // 对两个 SIMD 寄存器进行操作，返回结果
    auto o1 = op(a_lo, b_lo);
    auto o2 = op(a_hi, b_hi);
    // 将操作结果转换回原始数据类型 T，并返回一个新的 Vectorized<T> 对象
    return cvt_from_fp32<T, /*is_compare_op*/true>(o1, o2);
  }

public:
  // 重载运算符 >，使用 binary_compare 函数进行向量间的大于比较
  Vectorized<T> inline operator>(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      // 创建一个包含全零的 SIMD 寄存器
      auto zero_vec = _mm512_set1_epi32(0);
      // 使用 _CMP_GT_OQ 模式对 x 和 y 进行比较，返回比较结果的掩码
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ);
      // 根据比较结果掩码设置相应的值并转换为单精度浮点数向量
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  // 重载运算符 <，使用 binary_compare 函数进行向量间的小于比较
  Vectorized<T> inline operator<(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  // 重载运算符 >=，使用 binary_compare 函数进行向量间的大于等于比较
  Vectorized<T> inline operator>=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  // 重载运算符 <=，使用 binary_compare 函数进行向量间的小于等于比较
  Vectorized<T> inline operator<=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  // 重载运算符 ==，使用 binary_compare 函数进行向量间的等于比较
  Vectorized<T> inline operator==(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  // 重载运算符 !=，使用 binary_compare 函数进行向量间的不等于比较
  Vectorized<T> inline operator!=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
};

// 模板函数，用于将两个 Vectorized<T> 对象的数据进行某种操作，并返回一个新的 Vectorized<T> 对象
template<typename T, typename Op>
static inline Vectorized<T> binary_op_as_fp32(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
  // 定义两个 SIMD 寄存器用于存储两个对象的数据
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  // 将两个对象的数据转换为单精度浮点数，并存储到相应的 SIMD 寄存器中
  cvt_to_fp32<T>(__m512i(a), a_lo, a_hi);
  cvt_to_fp32<T>(__m512i(b), b_lo, b_hi);
  // 对两个 SIMD 寄存器进行操作，返回结果
  auto o1 = op(a_lo, b_lo);
  auto o2 = op(a_hi, b_hi);
  // 将操作结果转换回原始数据类型 T，并返回一个新的 Vectorized<T> 对象
  return cvt_from_fp32<T>(o1, o2);
}

template <>
// 特化模板类 Vectorized<BFloat16>，继承自 Vectorized16<BFloat16>
class Vectorized<BFloat16>: public Vectorized16<BFloat16> {
// 定义公共成员函数 frac，返回当前对象与其截断值的差
inline Vectorized<BFloat16> Vectorized<BFloat16>::frac() const {
    // 返回当前对象减去其截断值的结果
    return *this - this->trunc();
}

// 实现相等比较操作符，返回当前对象是否等于给定向量，并返回结果向量中每个元素为1.0的向量
inline Vectorized<BFloat16> Vectorized<BFloat16>::eq(const Vectorized<BFloat16>& other) const {
    // 返回当前对象是否等于给定向量，然后每个元素与1.0进行按位与操作的结果
    return (*this == other) & Vectorized<BFloat16>(1.0f);
}

// 实现不等比较操作符，返回当前对象是否不等于给定向量，并返回结果向量中每个元素为1.0的向量
inline Vectorized<BFloat16> Vectorized<BFloat16>::ne(const Vectorized<BFloat16>& other) const {
    // 返回当前对象是否不等于给定向量，然后每个元素与1.0进行按位与操作的结果
    return (*this != other) & Vectorized<BFloat16>(1.0f);
}

// 实现大于比较操作符，返回当前对象是否大于给定向量，并返回结果向量中每个元素为1.0的向量
inline Vectorized<BFloat16> Vectorized<BFloat16>::gt(const Vectorized<BFloat16>& other) const {
    // 返回当前对象是否大于给定向量，然后每个元素与1.0进行按位与操作的结果
    return (*this > other) & Vectorized<BFloat16>(1.0f);
}

// 实现大于等于比较操作符，返回当前对象是否大于等于给定向量，并返回结果向量中每个元素为1.0的向量
inline Vectorized<BFloat16> Vectorized<BFloat16>::ge(const Vectorized<BFloat16>& other) const {
    // 返回当前对象是否大于等于给定向量，然后每个元素与1.0进行按位与操作的结果
    return (*this >= other) & Vectorized<BFloat16>(1.0f);
}

// 实现小于比较操作符，返回当前对象是否小于给定向量，并返回结果向量中每个元素为1.0的向量
inline Vectorized<BFloat16> Vectorized<BFloat16>::lt(const Vectorized<BFloat16>& other) const {
    // 返回当前对象是否小于给定向量，然后每个元素与1.0进行按位与操作的结果
    return (*this < other) & Vectorized<BFloat16>(1.0f);
}

// 实现小于等于比较操作符，返回当前对象是否小于等于给定向量，并返回结果向量中每个元素为1.0的向量
inline Vectorized<BFloat16> Vectorized<BFloat16>::le(const Vectorized<BFloat16>& other) const {
    // 返回当前对象是否小于等于给定向量，然后每个元素与1.0进行按位与操作的结果
    return (*this <= other) & Vectorized<BFloat16>(1.0f);
}

// 实现加法操作符，返回当前对象与给定向量的加法结果
Vectorized<BFloat16> inline operator+(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    // 调用 binary_op_as_fp32 函数进行加法操作，返回结果
    return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_add_ps(x, y); });
}

// 实现减法操作符，返回当前对象与给定向量的减法结果
Vectorized<BFloat16> inline operator-(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    // 调用 binary_op_as_fp32 函数进行减法操作，返回结果
    return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_sub_ps(x, y); });
}

// 实现乘法操作符，返回当前对象与给定向量的乘法结果
Vectorized<BFloat16> inline operator*(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    // 调用 binary_op_as_fp32 函数进行乘法操作，返回结果
    return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_mul_ps(x, y); });
}

// 实现除法操作符，返回当前对象与给定向量的除法结果
Vectorized<BFloat16> inline operator/(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    // 调用 binary_op_as_fp32 函数进行除法操作，返回结果
    return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_div_ps(x, y); });
}

// 实现按位与操作符，返回当前对象与给定向量的按位与结果
Vectorized<BFloat16> inline operator&(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    // 调用 _mm512_and_si512 函数进行按位与操作，返回结果
    return _mm512_and_si512(a, b);
}

// 实现按位或操作符，返回当前对象与给定向量的按位或结果
Vectorized<BFloat16> inline operator|(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    // 调用 _mm512_or_si512 函数进行按位或操作，返回结果
    return _mm512_or_si512(a, b);
}

// 实现按位异或操作符，返回当前对象与给定向量的按位异或结果
Vectorized<BFloat16> inline operator^(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
    // 调用 _mm512_xor_si512 函数进行按位异或操作，返回结果
    return _mm512_xor_si512(a, b);
}
// 定义一个模板函数，用于计算两个 Vectorized<BFloat16> 向量中每个元素的最大值，并返回结果向量
Vectorized<BFloat16> inline maximum(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  // 声明四个 512 位寄存器变量，用于存储向量 a 和 b 的低位和高位部分
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  // 将 Vectorized<BFloat16> 向量转换为对应的 32 位浮点数向量，并存储在 a_lo, a_hi 和 b_lo, b_hi 中
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(b), b_lo, b_hi);
  // 计算每个元素的最大值，分别在低位和高位部分进行比较
  auto max_lo = _mm512_max_ps(a_lo, b_lo);
  auto max_hi = _mm512_max_ps(a_hi, b_hi);
  // 检测 NaN，并生成相应的掩码
  auto nan_lo_mask = _mm512_cmp_ps_mask(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi_mask = _mm512_cmp_ps_mask(a_hi, b_hi, _CMP_UNORD_Q);
  // 将 NaN 的掩码转换为浮点数向量，利用所有位均为 1 表示 NaN 的特性
  auto nan_lo = _mm512_castsi512_ps(_mm512_set1_epi32(nan_lo_mask));
  auto nan_hi = _mm512_castsi512_ps(_mm512_set1_epi32(nan_hi_mask));
  // 将最大值和 NaN 结果进行按位或操作
  auto o1 = _mm512_or_ps(max_lo, nan_lo);
  auto o2 = _mm512_or_ps(max_hi, nan_hi);
  // 将结果向量 o1 和 o2 转换回 BFloat16 类型并返回
  return cvtfp32_bf16(o1, o2);
}

// 实现 IEEE 754 201X 中的 `minimum` 操作，如果任一输入是 NaN，则传播 NaN
template <>
Vectorized<BFloat16> inline minimum(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  // 声明四个 512 位寄存器变量，用于存储向量 a 和 b 的低位和高位部分
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  // 初始化一个全零的 512 位整数向量，用于生成 NaN 掩码
  __m512i zero_vec = _mm512_set1_epi32(0);
  // 将 Vectorized<BFloat16> 向量转换为对应的 32 位浮点数向量，并存储在 a_lo, a_hi 和 b_lo, b_hi 中
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(b), b_lo, b_hi);
  // 计算每个元素的最小值，分别在低位和高位部分进行比较
  auto min_lo = _mm512_min_ps(a_lo, b_lo);
  auto min_hi = _mm512_min_ps(a_hi, b_hi);
  // 检测 NaN，并生成相应的掩码
  auto nan_lo_mask = _mm512_cmp_ps_mask(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi_mask = _mm512_cmp_ps_mask(a_hi, b_hi, _CMP_UNORD_Q);
  // 将 NaN 的掩码转换为浮点数向量，利用所有位均为 1 表示 NaN 的特性
  auto nan_lo = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, nan_lo_mask,
                                                           0xFFFFFFFF));
  auto nan_hi = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, nan_hi_mask,
                                                           0xFFFFFFFF));
  // 将最小值和 NaN 结果进行按位或操作
  auto o1 = _mm512_or_ps(min_lo, nan_lo);
  auto o2 = _mm512_or_ps(min_hi, nan_hi);
  // 将结果向量 o1 和 o2 转换回 BFloat16 类型并返回
  return cvtfp32_bf16(o1, o2);
}

// 实现 clamp 函数，将向量 a 中的每个元素限制在 min 和 max 之间
template <>
Vectorized<BFloat16> inline clamp(const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min, const Vectorized<BFloat16>& max) {
  // 声明四个 512 位寄存器变量，用于存储向量 a, min 和 max 的低位和高位部分
  __m512 a_lo, a_hi;
  __m512 min_lo, min_hi;
  __m512 max_lo, max_hi;
  // 将 Vectorized<BFloat16> 向量转换为对应的 32 位浮点数向量，并存储在 a_lo, a_hi 中
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(min), min_lo, min_hi);
  cvtbf16_fp32(__m512i(max), max_lo, max_hi);
  // 计算每个元素的限制值，分别在低位和高位部分进行处理
  auto o1 = _mm512_min_ps(max_lo, _mm512_max_ps(min_lo, a_lo));
  auto o2 = _mm512_min_ps(max_hi, _mm512_max_ps(min_hi, a_hi));
  // 将结果向量 o1 和 o2 转换回 BFloat16 类型并返回
  return cvtfp32_bf16(o1, o2);
}

// 实现 clamp_max 函数，将向量 a 中的每个元素限制在 max 以下
template <>
Vectorized<BFloat16> inline clamp_max(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& max) {
  // 声明四个 512 位寄存器变量，用于存储向量 a 和 max 的低位和高位部分
  __m512 a_lo, a_hi;
  __m512 max_lo, max_hi;
  // 将 Vectorized<BFloat16> 向量转换为对应的 32 位浮点数向量，并存储在 a_lo, a_hi 和 max_lo, max_hi 中
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(max), max_lo, max_hi);
  // 计算每个元素的限制值，分别在低位和高位部分进行处理
  auto o1 = _mm512_min_ps(max_lo, a_lo);
  auto o2 = _mm512_min_ps(max_hi, a_hi);
  // 将结果向量 o1 和 o2 转换回 BFloat16 类型并返回
  return cvtfp32_bf16(o1, o2);
}
Vectorized<BFloat16> inline clamp_min(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& min) {
  // 声明两个512位的寄存器变量，用于存储a和min的数据
  __m512 a_lo, a_hi;
  __m512 min_lo, min_hi;
  // 将a和min转换为32位浮点数的512位寄存器表示
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(min), min_lo, min_hi);
  // 对a_lo和min_lo进行每个元素的最大值比较
  auto o1 = _mm512_max_ps(min_lo, a_lo);
  // 对a_hi和min_hi进行每个元素的最大值比较
  auto o2 = _mm512_max_ps(min_hi, a_hi);
  // 将结果转换为BFloat16类型并返回
  return cvtfp32_bf16(o1, o2);
}

template <>
inline void convert(const BFloat16* src, BFloat16* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  // 按照Vectorized<BFloat16>的大小进行循环展开，从src中加载数据并存储到dst中
  for (i = 0; i <= (n - Vectorized<BFloat16>::size()); i += Vectorized<BFloat16>::size()) {
    auto vsrc = _mm512_loadu_si512(reinterpret_cast<__m512i*>((void*)(src + i)));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>((void*)(dst + i)), vsrc);
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  // 处理剩余不足一个Vectorized<BFloat16>大小的元素
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <>
inline void convert(const float* src, BFloat16* dst, int64_t n) {
  int64_t i;
  // 按照Vectorized<BFloat16>的大小，从src中加载数据并转换为BFloat16类型存储到dst中
  for (i = 0; i + Vectorized<BFloat16>::size() <= n; i += Vectorized<BFloat16>::size()) {
    __m512 a = _mm512_loadu_ps(&src[i]);
    __m512 b = _mm512_loadu_ps(&src[i + 16]);

    __m512i bf = cvtfp32_bf16(a, b);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dst[i]), bf);
  }
  // 处理剩余不足一个Vectorized<BFloat16>大小的元素
  for (; i < n; i++) {
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
inline void convert(const double* src, BFloat16* dst, int64_t n) {
  // 定义一个内部函数，从double数组加载一个浮点向量到512位寄存器中
  auto load_float = [](const double *src) -> __m512 {
    __m256 a = _mm512_cvtpd_ps(_mm512_loadu_pd(src));
    __m256 b = _mm512_cvtpd_ps(_mm512_loadu_pd(src + 8));
    return _mm512_insertf32x8(_mm512_castps256_ps512(a), b, 1);
  };

  int64_t i;
  // 按照Vectorized<BFloat16>的大小，从src中加载数据并转换为BFloat16类型存储到dst中
  for (i = 0; i + Vectorized<BFloat16>::size() <= n; i += Vectorized<BFloat16>::size()) {
    __m512 a = load_float(&src[i]);
    __m512 b = load_float(&src[i + 16]);

    __m512i bf = cvtfp32_bf16(a, b);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dst[i]), bf);
  }
  // 处理剩余不足一个Vectorized<BFloat16>大小的元素
  for (; i < n; i++) {
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
Vectorized<BFloat16> inline fmadd(const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b, const Vectorized<BFloat16>& c) {
  // 声明两个512位的寄存器变量，用于存储a、b、c的数据
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  __m512 c_lo, c_hi;
  // 将a、b、c转换为32位浮点数的512位寄存器表示
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(b), b_lo, b_hi);
  cvtbf16_fp32(__m512i(c), c_lo, c_hi);
  // 对a_lo和b_lo进行每个元素的乘加运算，并加上c_lo
  auto o1 = _mm512_fmadd_ps(a_lo, b_lo, c_lo);
  // 对a_hi和b_hi进行每个元素的乘加运算，并加上c_hi
  auto o2 = _mm512_fmadd_ps(a_hi, b_hi, c_hi);
  // 将结果转换为BFloat16类型并返回
  return cvtfp32_bf16(o1, o2);
}

static inline void _transpose_mxn_half_16_16(__m256i t[], __m512i u[]) {
  __m512i r[8];
  // 将四个256位寄存器的数据按特定顺序拼接到512位寄存器中，形成新的512位寄存器数组r
  // a0a1 a2a3 a4a5 a6a7 a8a9 a10a11 a12a13 a14a15   e0e1 e2e3 e4e5 e6e7 e8e9 e10e11 e12e13 e14e15
  // b0-b15  f0-f15
  // c0-c15  g0-g15
  // d0-d15  h0-h15
  // i0-i15  m0-m15
  // j0-j15  n0-n15
  // k0-k15  o0-o15
  // l0-l15  p0-p15
#ifndef __msvc_cl__
#pragma unroll(4)
#endif
  // 对r数组的前四个元素进行循环操作，将t数组的数据拼接到r数组中
  for (int i = 0; i < 4; i++) {
    r[i] = _mm512_inserti64x4(_mm512_castsi256_si512(t[i]), t[i + 4], 0x01);
    r[i + 4] = _mm512_inserti64x4(_mm512_castsi256_si512(t[i + 8]), t[i + 12], 0x01);

// 将 `_mm512_castsi256_si512(t[i + 8])` 插入到 `_mm512_inserti64x4` 中，形成 `r[i + 4]`。


  }

  // u0: a0a1 b0b1 a2a3 b2b3 a8a9 b8b9 a10a11 b10b11   e0e1 f0f1 e2e3 f2f3 e8e9 f8f9 e10e11 f10f11
  // u1: a4a5 b4b5 a6a7 b6b7 a12a13 b12b13 a14a15 b14b15   e4e5 f4f5 e6e7 f6f7 e12e13 f12f13 e14e15 f14f15
  // u2: c0c1 d0d1 c2c3 d2d3 c8c9 d8d9 c10c11 d10d11   g0g1 h0h1 g2g3 h2h3 g8g9 h8h9 g10g11 h10h11
  // u3: c4c5 d4b5 c6c7 d6b7 c12c13 d12d13 c14c15 d14d15   g4g5 h4h5 g6g7 h6h7 g12g13 h12h13 g14g15 h14h15
  // i j  m n
  // k l  o p

// 注释的部分是关于变量 `u0`, `u1`, `u2`, `u3` 的详细描述，描述了每个变量中的具体数据排列方式。
#ifndef __msvc_cl__
#pragma unroll(4)
#endif
// 使用预处理指令，如果不是 MSVC 编译器，对以下循环进行 4 次展开优化
for (int i = 0; i < 8; i += 2) {
  // 使用 AVX-512 指令集中的 _mm512_unpacklo_epi32 和 _mm512_unpackhi_epi32 函数，
  // 将 r[i] 和 r[i+1] 中的每对 32 位整数解压缩成低位和高位，存入 u 数组
  u[i] = _mm512_unpacklo_epi32(r[i], r[i + 1]);
  u[i + 1] = _mm512_unpackhi_epi32(r[i], r[i + 1]);
}

// r0: a0a1 b0b1 c0c1 d0d1 a8a9 b8b9 c8c9 d8d9  e0e1 f0f1 g0g1 h0h1 e8e9 f8f9 g8g9 h8h9
// r1: a2a3 b2b3 c2c3 d2d3 a10a11 b10b11 c10c11 d10d11  e2e3 f2f3 g2g3 h2h3 e10e11 f10f11 g10g11 h10h11
// r2: a4a5 b4b5 c4c5 d4b5 a12a13 b12b13 c12c13 d12d13
// r3: a6a7 b6b7 c6c7 d6b7 a14a15 b14b15 c14c15 d14d15
// r4: i j k l m n o p
// 使用 AVX-512 指令集中的 _mm512_unpacklo_epi64 和 _mm512_unpackhi_epi64 函数，
// 将 u 数组中的元素按照 64 位整数进行解压缩，合并到 r 数组的相应位置
r[0] = _mm512_unpacklo_epi64(u[0], u[2]);
r[1] = _mm512_unpackhi_epi64(u[0], u[2]);
r[2] = _mm512_unpacklo_epi64(u[1], u[3]);
r[3] = _mm512_unpackhi_epi64(u[1], u[3]);
r[4] = _mm512_unpacklo_epi64(u[4], u[6]);
r[5] = _mm512_unpackhi_epi64(u[4], u[6]);
r[6] = _mm512_unpacklo_epi64(u[5], u[7]);
r[7] = _mm512_unpackhi_epi64(u[5], u[7]);

// 定义常量 __m512i const1 和 const2，分别使用 _mm512_set_epi32 函数设置常量值
__m512i const1 = _mm512_set_epi32(
    0x00370035,
    0x00330031,
    0x00270025,
    0x00230021,
    0x00170015,
    0x00130011,
    0x00070005,
    0x00030001,
    0x00360034,
    0x00320030,
    0x00260024,
    0x00220020,
    0x00160014,
    0x00120010,
    0x00060004,
    0x00020000);
__m512i const2 = _mm512_set_epi32(
    0x003f003d,
    0x003b0039,
    0x002f002d,
    0x002b0029,
    0x001f001d,
    0x001b0019,
    0x000f000d,
    0x000b0009,
    0x003e003c,
    0x003a0038,
    0x002e002c,
    0x002a0028,
    0x001e001c,
    0x001a0018,
    0x000e000c,
    0x000a0008);
// 使用 _mm512_permutex2var_epi16 函数，根据 const1 和 const2 的指示，
// 对 r 数组中的相邻元素进行按位混合和交换
#ifndef __msvc_cl__
#pragma unroll(4)
#endif
// 如果不是 MSVC 编译器，对以下循环进行 4 次展开优化
for (int i = 0; i < 4; i++) {
  u[i] = _mm512_permutex2var_epi16(r[i], const1, r[i + 4]);
  u[i + 4] = _mm512_permutex2var_epi16(r[i], const2, r[i + 4]);
}
}
// TODO(Leslie): Add the AVX2 Version of transpose_mxn for BFloat16 and Float16
// Code referred to FBGEMM:
// https://github.com/pytorch/FBGEMM/blob/39a423e4ad1a04b77fea81c7d09c3e6f8984fae9/src/UtilsAvx512.cc#L1483-L1607
template<>
// 定义模板特化版本，用于将 BFloat16 类型的 mxn 矩阵转置为 16x16 矩阵
inline void transpose_mxn<BFloat16, 16, 16>(
    const BFloat16* src,
    int64_t ld_src,
    BFloat16* dst,
    int64_t ld_dst) {
  // 定义一个数组 t，包含 16 个 __m256i 类型的向量
  __m256i t[16];
  // 从 src 加载数据到寄存器中
  // a: a0  a1  a2  a3  a4  a5  a6  a7  a8  a9  a10 a11 a12 a13 a14 a15
  // b: b0  b1  b2  b3  b4  b5  b6  b7  b8  b9  b10 b11 b12 b13 b14 b15
  // c: c0  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10 c11 c12 c13 c14 c15
  // d: d0  d1  d2  d3  d4  d5  d6  d7  d8  d9  d10 d11 d12 d13 d14 d15
  // e: e0  e1  e2  e3  e4  e5  e6  e7  e8  e9  e10 e11 e12 e13 e14 e15
  // f: f0  f1  f2  f3  f4  f5  f6  f7  f8  f9  f10 f11 f12 f13 f14 f15
  // g: g0  g1  g2  g3  g4  g5  g6  g7  g8  g9  g10 g11 g12 g13 g14 g15
  // h: h0  h1  h2  h3  h4  h5  h6  h7  h8  h9  h10 h11 h12 h13 h14 h15
  // i: i0  i1  i2  i3  i4  i5  i6  i7  i8  i9  i10 i11 i12 i13 i14 i15
  // j: j0  j1  j2  j3  j4  j5  j6  j7  j8  j9  j10 j11 j12 j13 j14 j15
  // k: k0  k1  k2  k3  k4  k5  k6  k7  k8  k9  k10 k11 k12 k13 k14 k15
  // l: l0  l1  l2  l3  l4  l5  l6  l7  l8  l9  l10 l11 l12 l13 l14 l15
  // m: m0  m1  m2  m3  m4  m5  m6  m7  m8  m9  m10 m11 m12 m13 m14 m15
  // n: n0  n1  n2  n3  n4  n5  n6  n7  n8  n9  n10 n11 n12 n13 n14 n15
  // o: o0  o1  o2  o3  o4  o5  o6  o7  o8  o9  o10 o11 o12 o13 o14 o15
  // p: p0  p1  p2  p3  p4  p5  p6  p7  p8  p9  p10 p11 p12 p13 p14 p15
#ifndef __msvc_cl__
#pragma unroll(16)
#endif
// 如果不是 MSVC 编译器，则对接下来的循环进行16次展开优化
  for (int i = 0; i < 16; i++) {
    // 从 src 数组中加载数据到 AVX2 寄存器 t[i] 中
    t[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * ld_src));
  }

  __m512i u[8];
  // 调用特定的转置函数将 t 数组中的数据转置存储到 u 数组中
  _transpose_mxn_half_16_16(t, u);

#ifndef __msvc_cl__
#pragma unroll(8)
#endif
// 如果不是 MSVC 编译器，则对接下来的循环进行8次展开优化
  for (int i = 0; i < 8; i++) {
    // 将 u[i] 中的低32位数据存储到 dst 数组对应位置（偶数索引）
    _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(dst + (i * 2) * ld_dst),
      _mm512_extracti32x8_epi32(u[i], 0x0));
    // 将 u[i] 中的高32位数据存储到 dst 数组对应位置（奇数索引）
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + (i * 2 + 1) * ld_dst),
        _mm512_extracti32x8_epi32(u[i], 0x01));
  }
}
// 上述代码片段是 FBGEMM 库中特定的 AVX512 转置函数的实现，详情请见链接：
// https://github.com/pytorch/FBGEMM/blob/39a423e4ad1a04b77fea81c7d09c3e6f8984fae9/src/UtilsAvx512.cc#L1483-L1607
template<>
// 模板特化：转置一个半精度（Half）的 16x16 矩阵
inline void transpose_mxn<Half, 16, 16>(
    const Half* src,
    int64_t ld_src,
    Half* dst,
    int64_t ld_dst) {
  __m256i t[16];
  // 从 src 数组中加载数据到 AVX2 寄存器 t 中
  // 与上面对应的函数 transpose_mxn<BFloat16, 16, 16> 具有相同的矩阵索引
#ifndef __msvc_cl__
#pragma unroll(16)
#endif
  for (int i = 0; i < 16; i++) {
    t[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * ld_src));
  }

  __m512i u[8];
  // 调用特定的转置函数将 t 数组中的数据转置存储到 u 数组中
  _transpose_mxn_half_16_16(t, u);

#ifndef __msvc_cl__
#pragma unroll(8)
#endif
// 如果不是 MSVC 编译器，则对接下来的循环进行8次展开优化
  for (int i = 0; i < 8; i++) {
    // 将 u[i] 中的低32位数据存储到 dst 数组对应位置（偶数索引）
    _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(dst + (i * 2) * ld_dst),
      _mm512_extracti32x8_epi32(u[i], 0x0));
    // 将 u[i] 中的高32位数据存储到 dst 数组对应位置（奇数索引）
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + (i * 2 + 1) * ld_dst),
        _mm512_extracti32x8_epi32(u[i], 0x01));
  }
}
// 对 _transpose_mxn_half_32_32 函数进行了内联定义，该函数用于执行大小为 mxn 的矩阵转置操作
static inline void _transpose_mxn_half_32_32(__m512i r[], __m512i d[]) {
  // 定义了一个循环展开的注释块，每行描述了变量 t 的索引及其对应的元素序号范围
  // t[0]: 0 32 1 33 2 34 3 35 8 40 9 41 10 42 11 43 16 ... 59
  // t[1]: 4 36 5 37 6 38 7 39 12 44 13 45 14 46 15 47 20 ... 63
  // t[2]: 64 96 65 97 66 98 67 99 72 104 73 105 74 106 75 ... 123
  // t[3]: 68 100 69 101 70 102 71 103 76 108 77 109 78 110 79 111 84 ... 127
  // t[4]: 128 160 129 161 130 162 131 163 136 168 137 169 138 170 139 171 144 ... 187
  // t[5]: 132 164 133 165 134 166 135 167 140 172 141 173 142 174 143 175 148 ... 191
  // t[6]: 192 224 193 225 194 226 195 227 200 232 201 233 202 234 203 235 208 ... 251
  // t[7]: 196 228 197 229 198 230 199 231 204 236 205 237 206 238 207 239 212 ... 255
  // t[8]: 256 288 257 289 258 290 259 291 264 296 265 297 266 298 267 299 272 ... 315
  // t[9]: 260 292 261 293 262 294 263 295 268 300 269 301 270 302 271 303 276 ... 319
  // t[10]: 320 352 321 353 322 354 323 355 328 360 329 361 330 362 331 363 336 ... 379
  // t[11]: 324 356 325 357 326 358 327 359 332 364 333 365 334 366 335 367 340 ... 383
  // t[12]: 384 416 385 417 386 418 387 419 392 424 393 425 394 426 395 427 400 ... 443
  // t[13]: 388 420 389 421 390 422 391 423 396 428 397 429 398 430 399 431 404 ... 447
  // t[14]: 448 480 449 481 450 482 451 483 456 488 457 489 458 490 459 491 464 ... 507
  // t[15]: 452 484 453 485 454 486 455 487 460 492 461 493 462 494 463 495 468 ... 511
  // t[16]: 512 544 513 545 514 546 515 547 520 552 521 553 522 554 523 555 528 ... 571
  // ...
  // t[31]: 964 996 965 997 966 998 967 999 972 1004 973 1005 974 1006 975 1007 980 ... 1023

#ifndef __msvc_cl__
#pragma unroll(16)
#endif
  // 使用循环对矩阵进行操作，循环次数为 16 次
  for (int i = 0; i < 16; ++i) {
    // 将 r 数组中第 i*2 和 i*2+1 位置的元素进行解压缩操作，结果存入 d 数组中的对应位置
    d[i * 2] = _mm512_unpacklo_epi16(r[i * 2], r[i * 2 + 1]);
    // 将 r[i * 2] 和 r[i * 2 + 1] 中的每个16位整数进行高低位展开，并存储到 d[i * 2 + 1] 中
    d[i * 2 + 1] = _mm512_unpackhi_epi16(r[i * 2], r[i * 2 + 1]);
  }

  // t[0]: 0 32 64 96 1 33 65 97 8 40 72 104 9 41 73 105 16 ... 121
  // t[1]: 2 34 66 98 3 35 67 99 10 42 74 106 11 43 75 107 18 ... 123
  // t[2]: 4 36 68 100 5 37 69 101 12 44 76 108 13 45 77 109 20 ... 125
  // t[3]: 6 38 70 102 7 39 71 103 14 46 78 110 15 47 79 111 22 ... 127
  // t[4]: 128 160 192 224 129 161 193 225 136 168 200 232 137 169 201 233 144 ... 249
  // t[5]: 130 162 194 226 131 163 195 227 138 170 202 234 139 171 203 235 146 ... 251
  // t[6]: 132 164 196 228 133 165 197 229 140 172 204 236 141 173 205 237 148 ... 253
  // t[7]: 134 166 198 230 135 167 199 231 142 174 206 238 143 175 207 239 150 ... 255
  // t[8]: 256 288 320 352 257 289 321 353 264 296 328 360 265 297 329 361 272 ... 377
  // t[9]: 258 290 322 354 259 291 323 355 266 298 330 362 267 299 331 363 274 ... 379
  // t[10]: 260 292 324 356 261 293 325 357 268 300 332 364 269 301 333 365 276 ... 381
  // t[11]: 262 294 326 358 263 295 327 359 270 302 334 366 271 303 335 367 278 ... 383
  // t[12]: 384 416 448 480 385 417 449 481 392 424 456 488 393 425 457 489 400 ... 505
  // t[13]: 386 418 450 482 387 419 451 483 394 426 458 490 395 427 459 491 402 ... 507
  // t[14]: 388 420 452 484 389 421 453 485 396 428 460 492 397 429 461 493 404 ... 509
  // t[15]: 390 422 454 486 391 423 455 487 398 430 462 494 399 431 463 495 406 ... 511
  // t[16]: 512 544 576 608 513 545 577 609 520 552 584 616 521 553 585 617 528 ... 633
  // ...
  // t[31]: 902 934 966 998 903 935 967 999 910 942 974 1006 911 943 975 1007 918 ... 1023
#ifndef __msvc_cl__
#pragma unroll(8)
#endif
// 循环处理，对于每个 i 从 0 到 7
for (int i = 0; i < 8; ++i) {
    // 将 d[i * 4] 和 d[i * 4 + 2] 进行32位整数的解包，结果存入 r 数组
    r[i * 4] = _mm512_unpacklo_epi32(d[i * 4], d[i * 4 + 2]);
    r[i * 4 + 1] = _mm512_unpackhi_epi32(d[i * 4], d[i * 4 + 2]);
    // 将 d[i * 4 + 1] 和 d[i * 4 + 3] 进行32位整数的解包，结果存入 r 数组
    r[i * 4 + 2] = _mm512_unpacklo_epi32(d[i * 4 + 1], d[i * 4 + 3]);
    r[i * 4 + 3] = _mm512_unpackhi_epi32(d[i * 4 + 1], d[i * 4 + 3]);
}

// t 数组的注释
// t[0]: 0 32 64 96 128 160 192 224 8 40 72 104 136 168 200 232 16 ... 248
// t[1]: 1 33 65 97 129 161 193 225 9 41 73 105 137 169 201 233 17 ... 249
// t[2]: 2 34 66 98 130 162 194 226 10 42 74 106 138 170 202 234 18 ... 250
// t[3]: 3 35 67 99 131 163 195 227 11 43 75 107 139 171 203 235 19 ... 251
// t[4]: 4 36 68 100 132 164 196 228 12 44 76 108 140 172 204 236 20 ... 252
// t[5]: 5 37 69 101 133 165 197 229 13 45 77 109 141 173 205 237 21 ... 253
// t[6]: 6 38 70 102 134 166 198 230 14 46 78 110 142 174 206 238 22 ... 254
// t[7]: 7 39 71 103 135 167 199 231 15 47 79 111 143 175 207 239 23 ... 255
// t[8]: 256 288 320 352 384 416 448 480 264 296 328 360 392 424 456 488 272 ... 504
// t[9]: 257 289 321 353 385 417 449 481 265 297 329 361 393 425 457 489 273 ... 505
// t[10]: 258 290 322 354 386 418 450 482 266 298 330 362 394 426 458 490 274 ... 506
// t[11]: 259 291 323 355 387 419 451 483 267 299 331 363 395 427 459 491 275 ... 507
// t[12]: 260 292 324 356 388 420 452 484 268 300 332 364 396 428 460 492 276 ... 508
// t[13]: 261 293 325 357 389 421 453 485 269 301 333 365 397 429 461 493 277 ... 509
// t[14]: 262 294 326 358 390 422 454 486 270 302 334 366 398 430 462 494 278 ... 510
// t[15]: 263 295 327 359 391 423 455 487 271 303 335 367 399 431 463 495 279 ... 511
// t[16]: 512 544 576 608 640 672 704 736 520 552 584 616 648 680 712 744 528 ... 760
// ...
// t[31]: 775 807 839 871 903 935 967 999 783 815 847 879 911 943 975 1007 791 ... 1023

#ifndef __msvc_cl__
#pragma unroll(4)
#endif
// 循环处理，对于每个 i 从 0 到 3
for (int i = 0; i < 4; ++i) {
    // 将 r[i * 8] 和 r[i * 8 + 4] 进行64位整数的解包，结果存入 d 数组
    d[i * 8] = _mm512_unpacklo_epi64(r[i * 8], r[i * 8 + 4]);
    d[i * 8 + 1] = _mm512_unpackhi_epi64(r[i * 8], r[i * 8 + 4]);
    // 将 r[i * 8 + 1] 和 r[i * 8 + 5] 进行64位整数的解包，结果存入 d 数组
    d[i * 8 + 2] = _mm512_unpacklo_epi64(r[i * 8 + 1], r[i * 8 + 5]);
    d[i * 8 + 3] = _mm512_unpackhi_epi64(r[i * 8 + 1], r[i * 8 + 5]);
    // 将 r[i * 8 + 2] 和 r[i * 8 + 6] 进行64位整数的解包，结果存入 d 数组
    d[i * 8 + 4] = _mm512_unpacklo_epi64(r[i * 8 + 2], r[i * 8 + 6]);
    d[i * 8 + 5] = _mm512_unpackhi_epi64(r[i * 8 + 2], r[i * 8 + 6]);
    // 将 r[i * 8 + 3] 和 r[i * 8 + 7] 进行64位整数的解包，结果存入 d 数组
    d[i * 8 + 6] = _mm512_unpacklo_epi64(r[i * 8 + 3], r[i * 8 + 7]);
    // 使用 SIMD 指令生成常量 __m512i 对象 const1，包含一组特定的64位整数值
    __m512i const1 = _mm512_set_epi64(
        0x000000000000000d,
        0x000000000000000c,
        0x0000000000000005,
        0x0000000000000004,
        0x0000000000000009,
        0x0000000000000008,
        0x0000000000000001,
        0x0000000000000000);
    // 使用 SIMD 指令生成常量 __m512i 对象 const2，包含另一组特定的64位整数值
    __m512i const2 = _mm512_set_epi64(
        0x000000000000000f,
        0x000000000000000e,
        0x0000000000000007,
        0x0000000000000006,
        0x000000000000000b,
        0x000000000000000a,
        0x0000000000000003,
        0x0000000000000002);
#ifndef __msvc_cl__
#pragma unroll(8)
#endif
// 循环展开，对于非 MSVC 编译器
  for (int i = 0; i < 8; ++i) {
    // 使用 _mm512_permutex2var_epi64 函数对数据进行混洗操作，生成结果 r 数组的前半部分
    r[i] = _mm512_permutex2var_epi64(d[i], /*idx*/const1, d[i + 8]);
    // 使用 _mm512_permutex2var_epi64 函数对数据进行混洗操作，生成结果 r 数组的后半部分
    r[i + 8] = _mm512_permutex2var_epi64(d[i], /*idx*/const2, d[i + 8]);
    // 使用 _mm512_permutex2var_epi64 函数对数据进行混洗操作，生成结果 r 数组的第二部分
    r[i + 16] = _mm512_permutex2var_epi64(d[i + 16], /*idx*/const1, d[i + 24]);
    // 使用 _mm512_permutex2var_epi64 函数对数据进行混洗操作，生成结果 r 数组的第四部分
    r[i + 24] = _mm512_permutex2var_epi64(d[i + 16], /*idx*/const2, d[i + 24]);
  }

  // t[0]: 0 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 ... 992
  // t[1]: 1 33 65 97 129 161 193 225 257 289 321 353 385 417 449 481 513 545 ... 993
  // t[2]: 2 34 66 98 130 162 194 226 258 290 322 354 386 418 450 482 514 546 ... 994
  // t[3]: 3 35 67 99 131 163 195 227 259 291 323 355 387 419 451 483 515 547 ... 995
  // t[4]: 4 36 68 100 132 164 196 228 260 292 324 356 388 420 452 484 516 548 ... 996
  // t[5]: 5 37 69 101 133 165 197 229 261 293 325 357 389 421 453 485 517 549 ... 997
  // t[6]: 6 38 70 102 134 166 198 230 262 294 326 358 390 422 454 486 518 550 ... 998
  // t[7]: 7 39 71 103 135 167 199 231 263 295 327 359 391 423 455 487 519 551 ... 999
  // t[8]: 8 40 72 104 136 168 200 232 264 296 328 360 392 424 456 488 520 552 ... 1000
  // t[9]: 9 41 73 105 137 169 201 233 265 297 329 361 393 425 457 489 521 553 ... 1001
  // t[10]: 10 42 74 106 138 170 202 234 266 298 330 362 394 426 458 490 522 554 ... 1002
  // t[11]: 11 43 75 107 139 171 203 235 267 299 331 363 395 427 459 491 523 555 ... 1003
  // t[12]: 12 44 76 108 140 172 204 236 268 300 332 364 396 428 460 492 524 556 ... 1004
  // t[13]: 13 45 77 109 141 173 205 237 269 301 333 365 397 429 461 493 525 557 ... 1005
  // t[14]: 14 46 78 110 142 174 206 238 270 302 334 366 398 430 462 494 526 558 ... 1006
  // t[15]: 15 47 79 111 143 175 207 239 271 303 335 367 399 431 463 495 527 559 ... 1007
  // t[16]: 16 48 80 112 144 176 208 240 272 304 336 368 400 432 464 496 528 560 ... 1008
  // ...
  // t[31]: 31 63 95 127 159 191 223 255 287 319 351 383 415 447 479 511 543 575 ... 1023
  __m512i const3 = _mm512_set_epi64(
      0x000000000000000b,
      0x000000000000000a,
      0x0000000000000009,
      0x0000000000000008,
      0x0000000000000003,
      0x0000000000000002,
      0x0000000000000001,
      0x0000000000000000);
  // 设置常量向量 const3 用于后续的混洗操作
  __m512i const4 = _mm512_set_epi64(
      0x000000000000000f,
      0x000000000000000e,
      0x000000000000000d,
      0x000000000000000c,
      0x0000000000000007,
      0x0000000000000006,
      0x0000000000000005,
      0x0000000000000004);
#ifndef __msvc_cl__
#pragma unroll(16)
#endif
// 循环展开，对于非 MSVC 编译器
  for (int i = 0; i < 16; ++i) {
    // 使用 _mm512_permutex2var_epi64 函数对数据进行混洗操作，生成结果 d 数组的前半部分
    d[i] = _mm512_permutex2var_epi64(r[i], /*idx*/const3, r[i + 16]);
    // 使用 _mm512_permutex2var_epi64 函数对数据进行混洗操作，生成结果 d 数组的后半部分
    d[i + 16] = _mm512_permutex2var_epi64(r[i], /*idx*/const4, r[i + 16]);
  }
}
    // 声明一个包含32个__m512i类型元素的寄存器数组r，用于存储向量数据
    __m512i r[32];
#ifndef __msvc_cl__
#pragma unroll(32)
#endif
// 如果不是在 MSVC 编译环境下，对接下来的循环进行展开优化，展开次数为32次
for (int i = 0; i < 32; ++i) {
  // 将 src 中的数据加载到 r 数组中，每次加载 ld_src 个元素，转换为 __m512i 类型
  r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i* ld_src));
}

__m512i d[32];
// 调用 _transpose_mxn_half_32_32 函数对 r 数组进行转置，结果存储在 d 数组中
_transpose_mxn_half_32_32(r, d);

// 将结果存储到 dst 中
#ifndef __msvc_cl__
#pragma unroll(32)
#endif
// 如果不是在 MSVC 编译环境下，对接下来的循环进行展开优化，展开次数为32次
for (int i = 0; i < 32; ++i) {
  // 将 d 数组中的数据存储到 dst 中，每次存储 ld_dst 个元素，转换为 __m512i 类型
  _mm512_storeu_si512(dst + i* ld_dst, d[i]);
}
}



template<>
inline void transpose_mxn<Half, 32, 32>(
    const Half* src,
    int64_t ld_src,
    Half* dst,
    int64_t ld_dst) {
  // 从内存中加载数据
  __m512i r[32];
#ifndef __msvc_cl__
#pragma unroll(32)
#endif
  // 如果不是在 MSVC 编译环境下，对接下来的循环进行展开优化，展开次数为32次
  for (int i = 0; i < 32; ++i) {
    // 将 src 中的数据加载到 r 数组中，每次加载 ld_src 个元素，转换为 __m512i 类型
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i* ld_src));
  }

  __m512i d[32];
  // 调用 _transpose_mxn_half_32_32 函数对 r 数组进行转置，结果存储在 d 数组中
  _transpose_mxn_half_32_32(r, d);

  // 将结果存储到 dst 中
#ifndef __msvc_cl__
#pragma unroll(32)
#endif
  // 如果不是在 MSVC 编译环境下，对接下来的循环进行展开优化，展开次数为32次
  for (int i = 0; i < 32; ++i) {
    // 将 d 数组中的数据存储到 dst 中，每次存储 ld_dst 个元素，转换为 __m512i 类型
    _mm512_storeu_si512(dst + i* ld_dst, d[i]);
  }
}



template <>
class Vectorized<Half>: public Vectorized16<Half> {
public:
  using Vectorized16::Vectorized16;

  Vectorized<Half> frac() const;

  Vectorized<Half> eq(const Vectorized<Half>& other) const;
  Vectorized<Half> ne(const Vectorized<Half>& other) const;
  Vectorized<Half> gt(const Vectorized<Half>& other) const;
  Vectorized<Half> ge(const Vectorized<Half>& other) const;
  Vectorized<Half> lt(const Vectorized<Half>& other) const;
  Vectorized<Half> le(const Vectorized<Half>& other) const;
};

// 向量加法运算符重载，返回两个向量相加的结果
Vectorized<Half> inline operator+(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_add_ps(x, y); });
}

// 向量减法运算符重载，返回两个向量相减的结果
Vectorized<Half> inline operator-(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_sub_ps(x, y); });
}

// 向量乘法运算符重载，返回两个向量相乘的结果
Vectorized<Half> inline operator*(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_mul_ps(x, y); });
}

// 向量除法运算符重载，返回两个向量相除的结果
Vectorized<Half> inline operator/(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_div_ps(x, y); });
}

// 向量按位与运算符重载，返回两个向量按位与的结果
Vectorized<Half> inline operator&(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm512_and_si512(a, b);
}

// 向量按位或运算符重载，返回两个向量按位或的结果
Vectorized<Half> inline operator|(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm512_or_si512(a, b);
}

// 向量按位异或运算符重载，返回两个向量按位异或的结果
Vectorized<Half> inline operator^(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm512_xor_si512(a, b);
}

// 向量相等比较运算符重载，返回比较结果并将结果转换为 Half 类型向量
inline Vectorized<Half> Vectorized<Half>::eq(const Vectorized<Half>& other) const {
  return (*this == other) & Vectorized<Half>(1.0f);
}

// 向量不等比较运算符重载，返回比较结果并将结果转换为 Half 类型向量
inline Vectorized<Half> Vectorized<Half>::ne(const Vectorized<Half>& other) const {
  return (*this != other) & Vectorized<Half>(1.0f);
}

// 向量大于比较运算符重载，返回比较结果并将结果转换为 Half 类型向量
inline Vectorized<Half> Vectorized<Half>::gt(const Vectorized<Half>& other) const {
  return (*this > other) & Vectorized<Half>(1.0f);
}
// 实现了向量化类 Vectorized<Half> 的大于等于运算符重载函数
inline Vectorized<Half> Vectorized<Half>::ge(const Vectorized<Half>& other) const {
  // 返回一个新的 Vectorized<Half> 对象，包含逐元素比较结果与1.0f的按位与操作结果
  return (*this >= other) & Vectorized<Half>(1.0f);
}

// 实现了向量化类 Vectorized<Half> 的小于运算符重载函数
inline Vectorized<Half> Vectorized<Half>::lt(const Vectorized<Half>& other) const {
  // 返回一个新的 Vectorized<Half> 对象，包含逐元素比较结果与1.0f的按位与操作结果
  return (*this < other) & Vectorized<Half>(1.0f);
}

// 实现了向量化类 Vectorized<Half> 的小于等于运算符重载函数
inline Vectorized<Half> Vectorized<Half>::le(const Vectorized<Half>& other) const {
  // 返回一个新的 Vectorized<Half> 对象，包含逐元素比较结果与1.0f的按位与操作结果
  return (*this <= other) & Vectorized<Half>(1.0f);
}

// 实现了向量化类 Vectorized<Half> 的 frac 函数，返回当前对象减去其截断部分的结果
inline Vectorized<Half> Vectorized<Half>::frac() const {
  return *this - this->trunc();
}

// 特化模板，实现了向量化类 Vectorized<Half> 的 IEEE 754 201X 中的 maximum 操作
// 如果任一输入为 NaN，则传播 NaN
template <>
Vectorized<Half> inline maximum(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(b), b_lo, b_hi);
  auto max_lo = _mm512_max_ps(a_lo, b_lo);
  auto max_hi = _mm512_max_ps(a_hi, b_hi);
  auto nan_lo_mask = _mm512_cmp_ps_mask(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi_mask = _mm512_cmp_ps_mask(a_hi, b_hi, _CMP_UNORD_Q);
  auto nan_lo = _mm512_castsi512_ps(_mm512_set1_epi32(nan_lo_mask));
  auto nan_hi = _mm512_castsi512_ps(_mm512_set1_epi32(nan_hi_mask));
  // 利用全1的特性来表示 NaN
  auto o1 = _mm512_or_ps(max_lo, nan_lo);
  auto o2 = _mm512_or_ps(max_hi, nan_hi);
  return cvtfp32_fp16(o1, o2);
}

// 特化模板，实现了向量化类 Vectorized<Half> 的 IEEE 754 201X 中的 minimum 操作
// 如果任一输入为 NaN，则传播 NaN
template <>
Vectorized<Half> inline minimum(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  __m512i zero_vec = _mm512_set1_epi32(0);
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(b), b_lo, b_hi);
  auto min_lo = _mm512_min_ps(a_lo, b_lo);
  auto min_hi = _mm512_min_ps(a_hi, b_hi);
  auto nan_lo_mask = _mm512_cmp_ps_mask(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi_mask = _mm512_cmp_ps_mask(a_hi, b_hi, _CMP_UNORD_Q);
  auto nan_lo = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, nan_lo_mask,
                                                           0xFFFFFFFF));
  auto nan_hi = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, nan_hi_mask,
                                                           0xFFFFFFFF));
  // 利用全1的特性来表示 NaN
  auto o1 = _mm512_or_ps(min_lo, nan_lo);
  auto o2 = _mm512_or_ps(min_hi, nan_hi);
  return cvtfp32_fp16(o1, o2);
}

// 特化模板，实现了向量化类 Vectorized<Half> 的 clamp 操作
template <>
Vectorized<Half> inline clamp(const Vectorized<Half>& a,
    const Vectorized<Half>& min, const Vectorized<Half>& max) {
  __m512 a_lo, a_hi;
  __m512 min_lo, min_hi;
  __m512 max_lo, max_hi;
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(min), min_lo, min_hi);
  cvtfp16_fp32(__m512i(max), max_lo, max_hi);
  auto o1 = _mm512_min_ps(max_lo, _mm512_max_ps(min_lo, a_lo));
  auto o2 = _mm512_min_ps(max_hi, _mm512_max_ps(min_hi, a_hi));
  return cvtfp32_fp16(o1, o2);
}
Vectorized<Half> inline clamp_max(const Vectorized<Half>& a, const Vectorized<Half>& max) {
  // 声明两个512位寄存器变量，用于存储向量 a 的低64位和高64位数据
  __m512 a_lo, a_hi;
  // 声明两个512位寄存器变量，用于存储向量 max 的低64位和高64位数据
  __m512 max_lo, max_hi;
  // 将向量 a 转换为单精度浮点数，分别存入 a_lo 和 a_hi
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  // 将向量 max 转换为单精度浮点数，分别存入 max_lo 和 max_hi
  cvtfp16_fp32(__m512i(max), max_lo, max_hi);
  // 对比得到每个位置上的最小值，分别存入 o1 和 o2
  auto o1 = _mm512_min_ps(max_lo, a_lo);
  auto o2 = _mm512_min_ps(max_hi, a_hi);
  // 将 o1 和 o2 转换回半精度浮点数，并返回结果向量
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<Half> inline clamp_min(const Vectorized<Half>& a, const Vectorized<Half>& min) {
  // 声明两个512位寄存器变量，用于存储向量 a 的低64位和高64位数据
  __m512 a_lo, a_hi;
  // 声明两个512位寄存器变量，用于存储向量 min 的低64位和高64位数据
  __m512 min_lo, min_hi;
  // 将向量 a 转换为单精度浮点数，分别存入 a_lo 和 a_hi
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  // 将向量 min 转换为单精度浮点数，分别存入 min_lo 和 min_hi
  cvtfp16_fp32(__m512i(min), min_lo, min_hi);
  // 对比得到每个位置上的最大值，分别存入 o1 和 o2
  auto o1 = _mm512_max_ps(min_lo, a_lo);
  auto o2 = _mm512_max_ps(min_hi, a_hi);
  // 将 o1 和 o2 转换回半精度浮点数，并返回结果向量
  return cvtfp32_fp16(o1, o2);
}

template <>
inline void convert(const Half* src, Half* dst, int64_t n) {
  int64_t i;
  // 使用向量化处理的循环，每次处理 Vectorized<Half> 类型的大小
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<Half>::size()); i += Vectorized<Half>::size()) {
    // 加载源数据中的一个向量，并存入 vsrc
    auto vsrc = _mm512_loadu_si512(reinterpret_cast<__m512i*>((void*)(src + i)));
    // 将加载的向量数据存入目标地址中
    _mm512_storeu_si512(reinterpret_cast<__m512i*>((void*)(dst + i)), vsrc);
  }
  // 处理剩余不足一个向量的数据
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    // 将单个数据从源复制到目标，因为不满一个向量大小
    dst[i] = src[i];
  }
}

template <>
inline void convert(const float* src, Half* dst, int64_t n) {
  int64_t i;
  // 使用向量化处理的循环，每次处理 Vectorized<Half> 类型的大小
  for (i = 0; i + Vectorized<Half>::size() <= n; i += Vectorized<Half>::size()) {
    // 加载两个连续的 float 向量
    __m512 a = _mm512_loadu_ps(&src[i]);
    __m512 b = _mm512_loadu_ps(&src[i + 16]);
    // 将加载的 float 向量转换为半精度浮点数，并存入目标地址
    __m512i bf = cvtfp32_fp16(a, b);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dst[i]), bf);
  }
  // 处理剩余不足一个向量大小的数据
  for (; i < n; i++) {
    // 调用 c10::convert 将剩余的 float 数据转换为半精度浮点数
    dst[i] = c10::convert<Half>(src[i]);
  }
}

template <>
inline void convert(const double* src, Half* dst, int64_t n) {
  // 定义一个 lambda 函数，用于加载一个双精度浮点数数组中的一个 float 向量
  auto load_float = [](const double *src) -> __m512 {
    // 加载数组中的一个双精度浮点数向量并转换为四个单精度浮点数向量
    __m256 a = _mm512_cvtpd_ps(_mm512_loadu_pd(src));
    __m256 b = _mm512_cvtpd_ps(_mm512_loadu_pd(src + 8));
    // 将转换后的两个单精度浮点数向量合并成一个512位向量
    return _mm512_insertf32x8(_mm512_castps256_ps512(a), b, 1);
  };

  int64_t i;
  // 使用向量化处理的循环，每次处理 Vectorized<Half> 类型的大小
  for (i = 0; i + Vectorized<Half>::size() <= n; i += Vectorized<Half>::size()) {
    // 加载两个连续的 float 向量，通过 load_float 函数转换为半精度浮点数，并存入目标地址
    __m512 a = load_float(&src[i]);
    __m512 b = load_float(&src[i + 16]);
    // 将加载的半精度浮点数向量存入目标地址
    __m512i bf = cvtfp32_fp16(a, b);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dst[i]), bf);
  }
  // 处理剩余不足一个向量大小的数据
  for (; i < n; i++) {
    // 调用 c10::convert 将剩余的 double 数据转换为半精度浮点数
    dst[i] = c10::convert<Half>(src[i]);
  }
}

template <>
Vectorized<Half> inline fmadd(const Vectorized<Half>& a,
    const Vectorized<Half>& b, const Vectorized<Half>& c) {
  // 声明三个512位寄存器变量，用于存储向量 a、b 和 c 的低64位和高64位数据
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  __m512 c_lo, c_hi;
  // 将向量 a、b、c 转换为单精度浮点数，分别存入对应的寄存器变量
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(b), b_lo, b_hi);
  cvtfp16_fp32(__m512i(c), c_lo, c_hi);
  // 执行 fused multiply-add 操作，将结果存入 o1 和 o2
  auto o1 = _mm512_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm512_fmadd_ps(a_hi, b_hi, c_hi);
  // 将 o1 和 o2 转换回半精度浮点数，并返回结果向量
  return cvtfp32_fp16(o1, o2);
}

#define CONVERT_VECTORIZED_INIT(type, name) \
// 定义一个宏函数，用于将特定类型的向量化数据转换为两个单精度浮点向量，并作为元组返回
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  // 声明两个__m512变量o1和o2，用于存储转换后的结果
  __m512 o1, o2; \
  // 调用模板函数cvt_to_fp32将向量a转换为两个__m512浮点数向量o1和o2
  cvt_to_fp32<type>(__m512i(a), o1, o2); \
  // 使用std::make_tuple将o1和o2打包成元组并返回
  return std::make_tuple(o1, o2); \
} \
\
// 定义一个宏函数，用于将两个单精度浮点向量转换为特定类型的向量化数据并返回
inline Vectorized<type> convert_float_##name(const Vectorized<float>& a, const Vectorized<float>& b) { \
 // 调用模板函数cvt_from_fp32将两个__m512浮点数向量a和b转换为特定类型的向量化数据
 return cvt_from_fp32<type>(__m512(a), __m512(b)); \
}
CONVERT_VECTORIZED_INIT(BFloat16, bfloat16);
CONVERT_VECTORIZED_INIT(Half, half);

#else //defined(CPU_CAPABILITY_AVX512)

// 如果不支持AVX512，则定义另一个宏函数，将特定类型的向量化数据转换为两个单精度浮点向量，并作为元组返回
#define CONVERT_NON_VECTORIZED_INIT(type, name) \
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  // 获取向量的大小K
  constexpr int64_t K = Vectorized<type>::size(); \
  // 声明一个浮点数数组arr和特定类型数组arr2，用于存储转换后的数据
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  // 将向量a的数据存储到arr2数组中
  a.store(arr2); \
  // 使用循环将arr2数组中的数据转换为浮点数存储到arr数组中
  for (const auto k : c10::irange(K)) { \
    arr[k] = c10::convert<float>(arr2[k]); \
  } \
  // 返回转换后的浮点数向量作为元组
  return std::make_tuple( \
      Vectorized<float>::loadu(arr), \
      Vectorized<float>::loadu(arr + Vectorized<float>::size())); \
} \
\
// 定义一个宏函数，将两个单精度浮点向量转换为特定类型的向量化数据并返回
inline Vectorized<type> convert_float_##name(const Vectorized<float>& a, const Vectorized<float>& b) { \
  // 获取向量的大小K
  constexpr int64_t K = Vectorized<type>::size(); \
  // 声明一个浮点数数组arr和特定类型数组arr2，用于存储转换后的数据
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  // 将向量a和b的数据存储到对应的arr和arr+向量大小位置
  a.store(arr); \
  b.store(arr + Vectorized<float>::size()); \
  // 使用循环将arr数组中的数据转换为特定类型存储到arr2数组中
  for (const auto k : c10::irange(K)) { \
    arr2[k] = c10::convert<type>(arr[k]); \
  } \
  // 返回转换后的特定类型的向量化数据
  return Vectorized<type>::loadu(arr2); \
}
CONVERT_NON_VECTORIZED_INIT(BFloat16, bfloat16);
CONVERT_NON_VECTORIZED_INIT(Half, half);

#endif // defined(CPU_CAPABILITY_AVX512)

#if defined(CPU_CAPABILITY_AVX512)
// 如果支持AVX512，则定义一个宏函数，从特定类型的数据加载到向量化的单精度浮点数中
#define LOAD_FP32_VECTORIZED_INIT(type, name) \
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out) { \
  // 使用_mm256_loadu_si256加载data指向的数据到values中
  auto values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data)); \
  // 声明一个__m512类型变量out_values，将values转换为单精度浮点数存储到out中
  __m512 out_values; \
  cvt_to_fp32<type>(values, out_values); \
  // 将转换后的__m512向量赋值给out
  out = out_values; \
} \
\
// 定义一个宏函数，从特定类型的数据加载到两个向量化的单精度浮点数中
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out1, Vectorized<float>& out2) { \
  // 加载data指向的数据到Vectorized<type>类型的vec中
  auto vec = Vectorized<type>::loadu(data); \
  // 声明两个__m512类型变量out1_values和out2_values，将vec转换为单精度浮点数存储到out1和out2中
  __m512 out1_values, out2_values; \
  cvt_to_fp32<type>(vec, out1_values, out2_values); \
  // 将转换后的两个__m512向量赋值给out1和out2
  out1 = out1_values; \
  out2 = out2_values; \
}
LOAD_FP32_VECTORIZED_INIT(BFloat16, bf16);
LOAD_FP32_VECTORIZED_INIT(Half, fp16);

#else // defined(CPU_CAPABILITY_AVX512)
// 如果不支持AVX512，则定义另一个宏函数，从特定类型的数据加载到向量化的单精度浮点数中
#define LOAD_FP32_NON_VECTORIZED_INIT(type, name) \
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out) { \
  // 声明一个长度为Vectorized<float>::size()的浮点数数组values
  __at_align__ float values[Vectorized<float>::size()]; \
  // 使用循环将data指向的数据加载到values数组中
  for (const auto k : c10::irange(Vectorized<float>::size())) { \
    values[k] = data[k]; \
  } \
  // 将values数组中的数据加载到Vectorized<float>类型的out中
  out = Vectorized<float>::loadu(values); \
} \
\
// 定义一个宏函数，从特定类型的数据加载到两个向量化的单精度浮点数中
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out1, Vectorized<float>& out2) { \
  // 调用load_fp32_from_##name函数加载数据到out1
  load_fp32_from_##name(data, out1); \
  // data指针移动Vectorized<float>::size()的位置
  data += Vectorized<float>::size(); \
  // 调用load_fp32_from_##name函数加载数据到out2
  load_fp32_from_##name(data, out2); \
}
LOAD_FP32_NON_VECTORIZED_INIT(BFloat16, bf16);
LOAD_FP32_NON_VECTORIZED_INIT(Half, fp16);

#endif
```