# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_bfloat16.h`

```py
// 预处理指令，指示编译器仅包含本文件一次
#pragma once

// 不要在此头文件中定义静态数据！
// 见注释 [不要使用 AVX 编译初始化器]

// 包含 ATen 库中的向量化指令和基础向量类
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
// 包含 C10 工具库中的整数范围函数
#include <c10/util/irange.h>

// 如果定义了 CPU_CAPABILITY_AVX2 宏，则编译以下代码
#if defined(CPU_CAPABILITY_AVX2)
// 定义 SLEEF_STATIC_LIBS 宏
#define SLEEF_STATIC_LIBS
// 包含 Sleef 库的头文件
#include <sleef.h>
#endif

// 忽略 GCC 编译器的特定警告，将警告状态推入堆栈
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"

// ATen 库的 vec 命名空间
namespace at::vec {
// 内联命名空间 CPU_CAPABILITY，见注释 [CPU_CAPABILITY 命名空间]
inline namespace CPU_CAPABILITY {

// 如果定义了 CPU_CAPABILITY_AVX2 宏，则编译以下代码
#if defined(CPU_CAPABILITY_AVX2)

// 如果未定义 SLEEF_CONST 宏，则根据编译器类型定义 SLEEF_CONST 或 SLEEF_CONST_OLD
#ifndef SLEEF_CONST
#if (defined(__GNUC__) || defined(__CLANG__)) && !defined(__INTEL_COMPILER)
#define SLEEF_CONST const
#else
#define SLEEF_CONST
#endif
#define SLEEF_CONST_OLD SLEEF_CONST
#else
#define SLEEF_CONST_OLD
#endif

// 将 bfloat16 类型转换为 float32 类型的向量操作
static inline void cvtbf16_fp32(const __m128i& a, __m256& o) {
  o = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a), 16));
}

// 将 bfloat16 类型的两个 __m256i 向量转换为两个 __m256 浮点数向量
static inline void cvtbf16_fp32(const __m256i& a, __m256& o1, __m256& o2) {
  // 从 __m256i 向量中提取低128位和高128位
  __m128i lo = _mm256_extractf128_si256(a, 0);
  __m128i hi = _mm256_extractf128_si256(a, 1);
  // 分别对低128位和高128位进行 bfloat16 到 float32 的转换
  cvtbf16_fp32(lo, o1);
  cvtbf16_fp32(hi, o2);
}

// 将 float32 类型的 __m256 向量转换为 bfloat16 类型的 __m128i 向量
static inline __m128i cvtfp32_bf16(const __m256& src) {
  // 将 __m256 向量转换为 __m256i 整数向量
  __m256i value = _mm256_castps_si256(src);
  // 创建全 1 的 bfloat16 NaN 向量
  __m256i nan = _mm256_set1_epi32(0xffff);
  // 比较 src 向量是否有序，生成掩码
  __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(src, src, _CMP_ORD_Q));
  // 创建全 1 的整数向量
  __m256i ones = _mm256_set1_epi32(0x1);
  // 创建全 1 的偏置向量
  __m256i vec_bias = _mm256_set1_epi32(0x7fff);
  // 计算 lsb = (input >> 16) & 1 的结果
  auto t_value = _mm256_and_si256(_mm256_srli_epi32(value, 16), ones);
  // 计算 rounding_bias = 0x7fff + lsb
  t_value = _mm256_add_epi32(t_value, vec_bias);
  // input += rounding_bias
  t_value = _mm256_add_epi32(t_value, value);
  // input = input >> 16
  t_value = _mm256_srli_epi32(t_value, 16);
  // 在转换回 bfloat16 之前检查 NaN
  t_value = _mm256_blendv_epi8(nan, t_value, mask);
  // 将结果打包为 bfloat16
  t_value = _mm256_packus_epi32(t_value, t_value);   // t[4-7] t[4-7] t[0-4] t[0-4]
  t_value = _mm256_permute4x64_epi64(t_value, 0xd8); // 11     01     10     00
  // 转换为 __m128i 类型并返回
  return _mm256_castsi256_si128(t_value);
}
// 将两个 __m256 类型的参数 a 和 b 转换为 __m256i 类型的结果
static inline __m256i cvtfp32_bf16(const __m256& a, const __m256& b) {
  // 将参数 a 转换为 __m256i 类型
  __m256i lo = _mm256_castps_si256(a);
  // 将参数 b 转换为 __m256i 类型
  __m256i hi = _mm256_castps_si256(b);
  // 创建一个全为 0xffff 的 __m256i 类型变量 nan
  __m256i nan = _mm256_set1_epi32(0xffff);
  // 根据参数 a 的有序性生成掩码，转换为 __m256i 类型 mask_lo
  __m256i mask_lo = _mm256_castps_si256(_mm256_cmp_ps(a, a, _CMP_ORD_Q));
  // 根据参数 b 的有序性生成掩码，转换为 __m256i 类型 mask_hi
  __m256i mask_hi = _mm256_castps_si256(_mm256_cmp_ps(b, b, _CMP_ORD_Q));
  // 创建一个全为 1 的 __m256i 类型变量 ones
  __m256i ones = _mm256_set1_epi32(0x1);
  // 创建一个全为 0x7fff 的 __m256i 类型变量 vec_bias
  __m256i vec_bias = _mm256_set1_epi32(0x7fff);

  // 从 lo 中右移 16 位并与 ones 按位与，存入 t_lo
  auto t_lo = _mm256_and_si256(_mm256_srli_epi32(lo, 16), ones);
  // 从 hi 中右移 16 位并与 ones 按位与，存入 t_hi
  auto t_hi = _mm256_and_si256(_mm256_srli_epi32(hi, 16), ones);

  // 将 t_lo 和 vec_bias 相加，结果存回 t_lo
  t_lo = _mm256_add_epi32(t_lo, vec_bias);
  // 将 t_hi 和 vec_bias 相加，结果存回 t_hi
  t_hi = _mm256_add_epi32(t_hi, vec_bias);

  // 将 t_lo 和 lo 相加，结果存回 t_lo
  t_lo = _mm256_add_epi32(t_lo, lo);
  // 将 t_hi 和 hi 相加，结果存回 t_hi
  t_hi = _mm256_add_epi32(t_hi, hi);

  // 将 t_lo 右移 16 位，结果存回 t_lo
  t_lo = _mm256_srli_epi32(t_lo, 16);
  // 将 t_hi 右移 16 位，结果存回 t_hi
  t_hi = _mm256_srli_epi32(t_hi, 16);

  // 根据 mask_lo 将 t_lo 中的 NaN 替换为 nan
  t_lo = _mm256_blendv_epi8(nan, t_lo, mask_lo);
  // 根据 mask_hi 将 t_hi 中的 NaN 替换为 nan
  t_hi = _mm256_blendv_epi8(nan, t_hi, mask_hi);

  // 将 t_lo 和 t_hi 进行打包，形成一个 __m256i 类型的结果
  t_lo = _mm256_packus_epi32(t_lo, t_hi);
  // 对打包结果进行置换，得到最终结果
  return _mm256_permute4x64_epi64(t_lo, 0xd8);
}

// 将两个 __m256 类型的参数 a 和 b 合并比较结果为一个 __m256i 类型的结果
static inline __m256i merge_compare_result(const __m256& a, const __m256& b) {
  // 将参数 a 转换为 __m256i 类型
  __m256i lo = _mm256_castps_si256(a);
  // 将参数 b 转换为 __m256i 类型
  __m256i hi = _mm256_castps_si256(b);
  // 将 lo 和 hi 各右移 16 位，结果存回 lo 和 hi
  lo = _mm256_srli_epi32(lo, 16);
  hi = _mm256_srli_epi32(hi, 16);
  // 将 lo 和 hi 进行打包，形成一个 __m256i 类型的结果 out
  auto out = _mm256_packus_epi32(lo, hi);
  // 对打包结果进行置换，得到最终结果
  return _mm256_permute4x64_epi64(out, 0xd8);
}

// 将 __m128i 类型的参数 a 转换为 __m256 类型的结果 o
// 这里是将 float16 转换为 float32
static inline void cvtfp16_fp32(const __m128i& a, __m256& o) {
  // 使用 SSE 指令将 a 转换为 o，即将 float16 转换为 float32
  o = _mm256_cvtph_ps(a);
}

// 将 __m256i 类型的参数 a 分别转换为两个 __m256 类型的结果 o1 和 o2
// 这里是将两个 float16 转换为两个 float32
static inline void cvtfp16_fp32(const __m256i& a, __m256& o1, __m256& o2) {
  // 从 a 中提取低位和高位分别存入 lo 和 hi
  __m128i lo = _mm256_extractf128_si256(a, 0);
  __m128i hi = _mm256_extractf128_si256(a, 1);
  // 分别将 lo 和 hi 转换为 o1 和 o2，即将 float16 转换为 float32
  cvtfp16_fp32(lo, o1);
  cvtfp16_fp32(hi, o2);
}

// 将 __m256 类型的参数 src 转换为 __m128i 类型的结果
// 这里是将 float32 转换为 float16
static inline __m128i cvtfp32_fp16(const __m256& src) {
  // 使用 AVX 指令将 src 转换为 __m128i 类型，即将 float32 转换为 float16
  return _mm256_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// 将两个 __m256 类型的参数 a 和 b 分别转换为两个 __m128i 类型的结果，再合并为一个 __m256i 类型的结果
// 这里是将两个 float32 转换为两个 float16，再合并为一个 __m256i 类型的结果
static inline __m256i cvtfp32_fp16(const __m256& a, const __m256& b) {
  // 将 a 和 b 分别转换为 __m128i 类型的 lo 和 hi
  __m128i lo = _mm256_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  __m128i hi = _mm256_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  // 将 lo 和 hi 合并为一个 __m256i 类型的结果，其中 lo 存入低 128 位，hi 存入高 128 位
  return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// 模板函数，根据模板类型 T 进行不同的 float16 或 bfloat16 转换为 float32
template <typename T, typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline void cvt_to_fp32(const __m128i& a, __m256& o);

// 模板特化，将 bfloat16 转换为 float32
template <> inline void cvt_to_fp32<BFloat16>(const __m128i& a, __m256& o) {
  // 调用具体的 bfloat16 转换为 float32 的函数 cvtbf16_fp32
  cvtbf16_fp32(a, o);
};

// 模板特化，将 float16 转换为 float32
template <> inline void cvt_to_fp32<Half>(const __m128i& a, __m256& o) {
  // 调用具体的 float16 转换为 float32 的函数 cvtfp16_fp32
  cvtfp16_fp32(a, o);
}

// 模板函数，
template <> inline void cvt_to_fp32<BFloat16>(const __m256i& a, __m256& o1, __m256& o2) {
  // 调用函数 cvtbf16_fp32 将输入的 BFloat16 类型转换为两个 __m256 类型的单精度浮点数
  cvtbf16_fp32(a, o1, o2);
}

template <> inline void cvt_to_fp32<Half>(const __m256i& a, __m256& o1, __m256& o2) {
  // 调用函数 cvtfp16_fp32 将输入的 Half 类型转换为两个 __m256 类型的单精度浮点数
  cvtfp16_fp32(a, o1, o2);
}

template <typename T, bool is_compare_op = false,
          typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline __m256i cvt_from_fp32(const __m256& a, const __m256& b);

template <> inline __m256i cvt_from_fp32<BFloat16, false>(const __m256& a, const __m256& b) {
  // 调用函数 cvtfp32_bf16 将两个 __m256 类型的单精度浮点数转换为一个 __m256i 类型的 BFloat16
  return cvtfp32_bf16(a, b);
}

template <> inline __m256i cvt_from_fp32<BFloat16, true>(const __m256& a, const __m256& b) {
  // 调用函数 merge_compare_result 合并两个 __m256 类型的单精度浮点数的比较结果为一个 __m256i 类型
  return merge_compare_result(a, b);
}

template <> inline __m256i cvt_from_fp32<Half, false>(const __m256& a, const __m256& b) {
  // 调用函数 cvtfp32_fp16 将两个 __m256 类型的单精度浮点数转换为一个 __m256i 类型的 Half
  return cvtfp32_fp16(a, b);
}

template <> inline __m256i cvt_from_fp32<Half, true>(const __m256& a, const __m256& b) {
  // 调用函数 cvtfp32_fp16 将两个 __m256 类型的单精度浮点数转换为一个 __m256i 类型的 Half
  return cvtfp32_fp16(a, b);
}

template <typename T>
class Vectorized16 {
  static_assert(
    is_reduced_floating_point_v<T>,
    "Support only float16 and bfloat16.");
protected:
  __m256i values; // 存储一个 __m256i 类型的值
public:
  using value_type = uint16_t; // 值的类型为 uint16_t
  using size_type = int; // 尺寸的类型为 int
  static constexpr size_type size() {
    return 16; // 返回向量的尺寸，固定为 16
  }
  Vectorized16() {} // 默认构造函数
  Vectorized16(__m256i v) : values(v) {} // 构造函数，使用给定的 __m256i 类型值初始化
  Vectorized16(T val) {
    value_type uw = val.x;
    values = _mm256_set1_epi16(uw); // 使用给定值 val 初始化 __m256i 类型的 values
  }
  Vectorized16(T val1, T val2, T val3, T val4,
         T val5, T val6, T val7, T val8,
         T val9, T val10, T val11, T val12,
         T val13, T val14, T val15, T val16) {
    // 使用给定的 16 个值初始化 __m256i 类型的 values
    values = _mm256_setr_epi16(
        val1.x, val2.x, val3.x, val4.x, val5.x, val6.x, val7.x, val8.x,
        val9.x, val10.x, val11.x, val12.x, val13.x, val14.x, val15.x, val16.x);
  }
  operator __m256i() const {
    return values; // 将 Vectorized16 对象转换为 __m256i 类型
  }
  T& operator[](int idx) = delete; // 删除操作符重载，禁止使用索引访问
  const T& operator[](int idx) const  = delete; // 删除操作符重载，禁止使用索引访问
  int zero_mask() const {
    // 返回一个整数掩码，其中所有零元素转换为 1 位，其他元素转换为 0 位
    __m256i cmp = _mm256_cmpeq_epi16(values, _mm256_set1_epi16(0));
    return _mm256_movemask_epi8(cmp);
  }
  static Vectorized<T> loadu(const void* ptr, int16_t count = size()) {
    if (count == size())
      return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)); // 从地址 ptr 加载一个 __m256i 类型的值

    __at_align__ int16_t tmp_values[size()];
    std::memcpy(tmp_values, ptr, count * sizeof(int16_t)); // 从指针 ptr 复制 count 个 int16_t 到临时数组 tmp_values
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmp_values)); // 从临时数组加载一个 __m256i 类型的值
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values); // 将 values 存储到地址 ptr
    } else if (count > 0) {
      __at_align__ int16_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values); // 将 values 存储到临时数组 tmp_values
      std::memcpy(ptr, tmp_values, count * sizeof(int16_t)); // 将临时数组 tmp_values 复制到地址 ptr
    }
  }
  template <int64_t mask>
  static Vectorized<T> blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    __at_align__ int16_t tmp_values[size()];
    // 将 tmp_values 存储到对象 a 中
    a.store(tmp_values);
    // 根据掩码 mask 分别从 b.values 中提取对应位置的元素，存储到 tmp_values 中
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi16(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi16(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi16(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi16(b.values, 3);
    if (mask & 0x10)
      tmp_values[4] = _mm256_extract_epi16(b.values, 4);
    if (mask & 0x20)
      tmp_values[5] = _mm256_extract_epi16(b.values, 5);
    if (mask & 0x40)
      tmp_values[6] = _mm256_extract_epi16(b.values, 6);
    if (mask & 0x80)
      tmp_values[7] = _mm256_extract_epi16(b.values, 7);
    if (mask & 0x100)
      tmp_values[8] = _mm256_extract_epi16(b.values, 8);
    if (mask & 0x200)
      tmp_values[9] = _mm256_extract_epi16(b.values, 9);
    if (mask & 0x400)
      tmp_values[10] = _mm256_extract_epi16(b.values, 10);
    if (mask & 0x800)
      tmp_values[11] = _mm256_extract_epi16(b.values, 11);
    if (mask & 0x1000)
      tmp_values[12] = _mm256_extract_epi16(b.values, 12);
    if (mask & 0x2000)
      tmp_values[13] = _mm256_extract_epi16(b.values, 13);
    if (mask & 0x4000)
      tmp_values[14] = _mm256_extract_epi16(b.values, 14);
    if (mask & 0x8000)
      tmp_values[15] = _mm256_extract_epi16(b.values, 15);
    // 将更新后的 tmp_values 加载到对象中并返回
    return loadu(tmp_values);
  }
  // 使用掩码 mask 对两个向量 a 和 b 进行混合，根据 mask 中每位的值来选择元素
  static Vectorized<T> blendv(const Vectorized<T>& a,
      const Vectorized<T>& b, const Vectorized<T>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  // 生成一个等差数列向量，从 base 开始，以 step 为步长，共16个元素
  template<typename step_t>
  static Vectorized<T> arange(T base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  // 使用指定的 count 值设置向量 a 的元素，从向量 b 中复制对应元素，未被设置的位置保持 a 的原值
  static Vectorized<T> set(const Vectorized<T>& a,
      const Vectorized<T>& b, int64_t count = size()) {
    switch (count) {
      // 根据 count 的不同值，调用不同的 blend 函数来设置向量 a 的元素
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
    }
    // 默认情况下，返回向量 b
    return b;
  }

  // 对向量中的每个元素应用函数 vop，并返回结果向量
  Vectorized<T> map(SLEEF_CONST __m256 (*SLEEF_CONST_OLD vop)(__m256)) const {
    __m256 lo, hi;
    // 将当前向量 values 转换为单精度浮点数并存储到 lo 和 hi 中
    cvt_to_fp32<T>(values, lo, hi);
    // 继续处理 ...
  Vectorized<T> erfinv() const {
    // 将当前向量转换为两个单精度浮点数向量 lo 和 hi
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);

    // 创建临时数组用于存储转换后的浮点数数据
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

    // 对临时数组中的数据分别计算逆误差函数的值
    for (int64_t i = 0; i < size() / 2; i++) {
      tmp1[i] = calc_erfinv(tmp1[i]);
      tmp2[i] = calc_erfinv(tmp2[i]);
    }

    // 将计算后的逆误差函数值重新加载回 AVX 寄存器
    auto o1 = _mm256_loadu_ps(tmp1);
  Vectorized<T> exp() const {
    // 调用 Sleef 库中的 expf8_u10 函数，对当前向量执行指数函数
    return map(Sleef_expf8_u10);
  }
  Vectorized<T> exp2() const {
    // 调用 Sleef 库中的 exp2f8_u10 函数，对当前向量执行以2为底的指数函数
    return map(Sleef_exp2f8_u10);
  }
  Vectorized<T> expm1() const {
    // 调用 Sleef 库中的 expm1f8_u10 函数，对当前向量执行 exp(x) - 1 的计算
    return map(Sleef_expm1f8_u10);
  }
  Vectorized<T> exp_u20() const {
    // 将 exp_u20 函数简单地重定向到 exp 函数
    return exp();
  }
  Vectorized<T> fmod(const Vectorized<T> & q) const {
    // 将当前向量和参数向量转换为 float 类型，然后调用 Sleef 库中的 fmodf8 函数执行取模运算
    __m256 x_lo, x_hi;
    cvt_to_fp32<T>(values, x_lo, x_hi);
    __m256 q_lo, q_hi;
    cvt_to_fp32<T>(q.values, q_lo, q_hi);
    auto o1 = Sleef_fmodf8(x_lo, q_lo);
    auto o2 = Sleef_fmodf8(x_hi, q_hi);
    // 将结果向量从 float 转换回 T 类型
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> hypot(const Vectorized<T> &b) const {
    // 将当前向量和参数向量转换为 float 类型，然后调用 Sleef 库中的 hypotf8_u05 函数执行求 hypotenuse 运算
    __m256 lo, hi;
    __m256 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    auto o1 = Sleef_hypotf8_u05(lo, b1);
    auto o2 = Sleef_hypotf8_u05(hi, b2);
    // 将结果向量从 float 转换回 T 类型
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> i0() const {
    // 将当前向量转换为 float 类型，然后在数组 tmp1 和 tmp2 中存储每一组的低位和高位
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    // 对 tmp1 和 tmp2 中的每个元素调用 calc_i0 函数，再加载回 __m256 类型的向量
    for (int64_t i = 0; i < size() / 2; i++) {
      tmp1[i] = calc_i0(tmp1[i]);
      tmp2[i] = calc_i0(tmp2[i]);
    }
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    // 将结果向量从 float 转换回 T 类型
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> i0e() const {
    // 将当前向量转换为 float 类型，然后在数组 tmp1 和 tmp2 中存储每一组的低位和高位
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    constexpr auto sz = size();
    __at_align__ float tmp1[sz / 2], tmp2[sz / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

    // 对 tmp1 和 tmp2 中的每个元素调用 calc_i0e 函数，再加载回 __m256 类型的向量
    for (auto i = decltype(sz){0}; i < sz / 2; i++) {
      tmp1[i] = calc_i0e(tmp1[i]);
      tmp2[i] = calc_i0e(tmp2[i]);
    }
    const auto o1 = _mm256_loadu_ps(tmp1);
    const auto o2 = _mm256_loadu_ps(tmp2);
    // 将结果向量从 float 转换回 T 类型
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> digamma() const {
    // 将当前向量转换为 float 类型，然后在数组 tmp1 和 tmp2 中存储每一组的低位和高位
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    constexpr auto sz = size();
    __at_align__ float tmp1[sz / 2], tmp2[sz / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

    // 对 tmp1 和 tmp2 中的每个元素调用 calc_digamma 函数，再加载回 __m256 类型的向量
    for (auto i = decltype(sz){0}; i < sz / 2; i++) {
      tmp1[i] = calc_digamma(tmp1[i]);
      tmp2[i] = calc_digamma(tmp2[i]);
    }
    const auto o1 = _mm256_loadu_ps(tmp1);
    const auto o2 = _mm256_loadu_ps(tmp2);
    // 将结果向量从 float 转换回 T 类型
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> igamma(const Vectorized<T> &x) const {
    // 将当前向量和参数向量转换为 float 类型，然后在数组 tmp1 和 tmp2 中存储每一组的低位和高位
    __m256 lo, hi;
    __m256 xlo, xhi;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(x.values, xlo, xhi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    __at_align__ float tmpx1[size() / 2], tmpx2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
    // 将 AVX 寄存器 xhi 中的数据存储到 tmpx2 数组中
    for (int64_t i = 0; i < size() / 2; ++i) {
      // 对数组 tmp1 和 tmp2 中的每个元素调用 calc_igamma 函数，并更新它们的值
      tmp1[i] = calc_igamma(tmp1[i], tmpx1[i]);
      tmp2[i] = calc_igamma(tmp2[i], tmpx2[i]);
    }
    // 从数组 tmp1 和 tmp2 中加载数据到 AVX 寄存器 o1 和 o2 中
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    // 将 o1 和 o2 中的数据转换回目标类型 T，并返回结果
    return cvt_from_fp32<T>(o1, o2);
  }

  Vectorized<T> igammac(const Vectorized<T> &x) const {
    __m256 lo, hi;
    // 将当前对象 values 中的数据转换为 AVX 寄存器 lo 和 hi 中的浮点数
    cvt_to_fp32<T>(values, lo, hi);
    // 将参数 x 的 values 中的数据转换为 AVX 寄存器 xlo 和 xhi 中的浮点数
    cvt_to_fp32<T>(x.values, xlo, xhi);
    // 声明并初始化大小为当前对象 size() / 2 的临时数组 tmp1 和 tmp2
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    // 将 AVX 寄存器 lo 和 hi 中的数据存储到对应的 tmp1 和 tmp2 数组中
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    // 声明并初始化大小为当前对象 size() / 2 的临时数组 tmpx1 和 tmpx2
    __at_align__ float tmpx1[size() / 2], tmpx2[size() / 2];
    // 将 AVX 寄存器 xlo 和 xhi 中的数据存储到对应的 tmpx1 和 tmpx2 数组中
    _mm256_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
    // 对数组 tmp1 和 tmp2 中的每个元素调用 calc_igammac 函数，并更新它们的值
    for (int64_t i = 0; i < size() / 2; ++i) {
      tmp1[i] = calc_igammac(tmp1[i], tmpx1[i]);
      tmp2[i] = calc_igammac(tmp2[i], tmpx2[i]);
    }
    // 从数组 tmp1 和 tmp2 中加载数据到 AVX 寄存器 o1 和 o2 中
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    // 将 o1 和 o2 中的数据转换回目标类型 T，并返回结果
    return cvt_from_fp32<T>(o1, o2);
  }
    cvt_to_fp32<T>(values, lo, hi);
    // 将当前向量中的值转换为单精度浮点数格式，并存储在 lo 和 hi 中

    auto o1 = _mm256_sqrt_ps(lo);
    // 计算 lo 向量中每个元素的平方根，并存储在 o1 中

    auto o2 = _mm256_sqrt_ps(hi);
    // 计算 hi 向量中每个元素的平方根，并存储在 o2 中

    return cvt_from_fp32<T>(o1, o2);
    // 将 o1 和 o2 中的单精度浮点数值转换回当前向量的数据类型，并返回新的 Vectorized<T> 对象
  }

  Vectorized<T> reciprocal() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    // 将当前向量中的值转换为单精度浮点数格式，并存储在 lo 和 hi 中

    auto ones = _mm256_set1_ps(1);
    // 创建一个包含单精度浮点数 1 的向量

    auto o1 = _mm256_div_ps(ones, lo);
    // 计算 lo 向量中每个元素的倒数，并存储在 o1 中

    auto o2 = _mm256_div_ps(ones, hi);
    // 计算 hi 向量中每个元素的倒数，并存储在 o2 中

    return cvt_from_fp32<T>(o1, o2);
    // 将 o1 和 o2 中的单精度浮点数值转换回当前向量的数据类型，并返回新的 Vectorized<T> 对象
  }

  Vectorized<T> rsqrt() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    // 将当前向量中的值转换为单精度浮点数格式，并存储在 lo 和 hi 中

    auto ones = _mm256_set1_ps(1);
    // 创建一个包含单精度浮点数 1 的向量

    auto o1 = _mm256_div_ps(ones, _mm256_sqrt_ps(lo));
    // 计算 lo 向量中每个元素的平方根的倒数，并存储在 o1 中

    auto o2 = _mm256_div_ps(ones, _mm256_sqrt_ps(hi));
    // 计算 hi 向量中每个元素的平方根的倒数，并存储在 o2 中

    return cvt_from_fp32<T>(o1, o2);
    // 将 o1 和 o2 中的单精度浮点数值转换回当前向量的数据类型，并返回新的 Vectorized<T> 对象
  }

  Vectorized<T> pow(const Vectorized<T> &b) const {
    __m256 lo, hi;
    __m256 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    // 将当前向量中的值转换为单精度浮点数格式，并存储在 lo 和 hi 中

    cvt_to_fp32<T>(b.values, b1, b2);
    // 将向量 b 中的值转换为单精度浮点数格式，并存储在 b1 和 b2 中

    auto o1 = Sleef_powf8_u10(lo, b1);
    // 使用 Sleef 库计算 lo 向量中每个元素的 b1 向量中对应元素次方，并存储在 o1 中

    auto o2 = Sleef_powf8_u10(hi, b2);
    // 使用 Sleef 库计算 hi 向量中每个元素的 b2 向量中对应元素次方，并存储在 o2 中

    return cvt_from_fp32<T>(o1, o2);
    // 将 o1 和 o2 中的单精度浮点数值转换回当前向量的数据类型，并返回新的 Vectorized<T> 对象
  }
private:
  // 定义一个模板方法，用于执行二元比较操作
  template<typename Op>
  Vectorized<T> inline binary_compare(const Vectorized<T>& b, Op op) const {
    // 定义四个 __m256 变量，分别存储向量的低位和高位数据
    __m256 a_lo, a_hi;
    __m256 b_lo, b_hi;
    // 将当前对象的值转换为单精度浮点数，并存储到 a_lo 和 a_hi 中
    cvt_to_fp32<T>(values, a_lo, a_hi);
    // 将参数对象 b 的值转换为单精度浮点数，并存储到 b_lo 和 b_hi 中
    cvt_to_fp32<T>(b.values, b_lo, b_hi);
    // 对 a_lo 和 b_lo 执行给定的比较操作 op，得到结果 o1
    auto o1 = op(a_lo, b_lo);
    // 对 a_hi 和 b_hi 执行给定的比较操作 op，得到结果 o2
    auto o2 = op(a_hi, b_hi);
    // 将 o1 和 o2 的结果转换回原始类型 T 的向量，并返回
    return cvt_from_fp32<T, /*is_compare_op*/true>(o1, o2);
  }

public:
  // 定义重载运算符 >，调用 binary_compare 执行大于比较操作
  Vectorized<T> inline operator>(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_GT_OQ); });
  }
  // 定义重载运算符 <，调用 binary_compare 执行小于比较操作
  Vectorized<T> inline operator<(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_LT_OQ); });
  }
  // 定义重载运算符 >=，调用 binary_compare 执行大于等于比较操作
  Vectorized<T> inline operator>=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_GE_OQ); });
  }
  // 定义重载运算符 <=，调用 binary_compare 执行小于等于比较操作
  Vectorized<T> inline operator<=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_LE_OQ); });
  }
  // 定义重载运算符 ==，调用 binary_compare 执行等于比较操作
  Vectorized<T> inline operator==(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_EQ_OQ); });
  }
  // 定义重载运算符 !=，调用 binary_compare 执行不等于比较操作
  Vectorized<T> inline operator!=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_NEQ_UQ); });
  }
};

// 定义模板方法，执行两个向量的二元操作，并将结果转换为单精度浮点数向量
template<typename T, typename Op>
static inline Vectorized<T> binary_op_as_fp32(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
  // 定义四个 __m256 变量，分别存储向量的低位和高位数据
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  // 将向量 a 转换为单精度浮点数，并存储到 a_lo 和 a_hi 中
  cvt_to_fp32<T>(__m256i(a), a_lo, a_hi);
  // 将向量 b 转换为单精度浮点数，并存储到 b_lo 和 b_hi 中
  cvt_to_fp32<T>(__m256i(b), b_lo, b_hi);
  // 对 a_lo 和 b_lo 执行给定的操作 op，得到结果 o1
  auto o1 = op(a_lo, b_lo);
  // 对 a_hi 和 b_hi 执行给定的操作 op，得到结果 o2
  auto o2 = op(a_hi, b_hi);
  // 将 o1 和 o2 的结果转换回原始类型 T 的向量，并返回
  return cvt_from_fp32<T>(o1, o2);
}

// 特化模板类 Vectorized<BFloat16>，继承自 Vectorized16<BFloat16>
template <>
class Vectorized<BFloat16>: public Vectorized16<BFloat16> {
public:
  using Vectorized16::Vectorized16;

  // 声明方法 frac 的原型，返回类型为 Vectorized<BFloat16>
  Vectorized<BFloat16> frac() const;

  // 声明方法 eq 的原型，返回类型为 Vectorized<BFloat16>
  Vectorized<BFloat16> eq(const Vectorized<BFloat16>& other) const;
  // 声明方法 ne 的原型，返回类型为 Vectorized<BFloat16>
  Vectorized<BFloat16> ne(const Vectorized<BFloat16>& other) const;
  // 声明方法 gt 的原型，返回类型为 Vectorized<BFloat16>
  Vectorized<BFloat16> gt(const Vectorized<BFloat16>& other) const;
  // 声明方法 ge 的原型，返回类型为 Vectorized<BFloat16>
  Vectorized<BFloat16> ge(const Vectorized<BFloat16>& other) const;
  // 声明方法 lt 的原型，返回类型为 Vectorized<BFloat16>
  Vectorized<BFloat16> lt(const Vectorized<BFloat16>& other) const;
  // 声明方法 le 的原型，返回类型为 Vectorized<BFloat16>
  Vectorized<BFloat16> le(const Vectorized<BFloat16>& other) const;
};

// 定义重载运算符 +，调用 binary_op_as_fp32 执行加法操作
Vectorized<BFloat16> inline operator+(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_add_ps(x, y); });
}

// 定义重载运算符 -，调用 binary_op_as_fp32 执行减法操作
Vectorized<BFloat16> inline operator-(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_sub_ps(x, y); });
}

// 定义重载运算符 *，调用 binary_op_as_fp32 执行乘法操作
Vectorized<BFloat16> inline operator*(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_mul_ps(x, y); });
}
// 重载运算符 `/`，用于两个 BFloat16 向量的除法操作，返回结果作为 Vectorized<BFloat16>
Vectorized<BFloat16> inline operator/(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  // 调用 binary_op_as_fp32 函数，将 a 和 b 视为 float32 向量进行除法运算
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_div_ps(x, y); });
}

// 重载运算符 `&`，用于两个 BFloat16 向量的按位与操作，返回结果作为 Vectorized<BFloat16>
Vectorized<BFloat16> inline operator&(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  // 使用 AVX2 指令集的 `_mm256_and_si256` 函数执行向量的按位与操作
  return _mm256_and_si256(a, b);
}

// 重载运算符 `|`，用于两个 BFloat16 向量的按位或操作，返回结果作为 Vectorized<BFloat16>
Vectorized<BFloat16> inline operator|(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  // 使用 AVX2 指令集的 `_mm256_or_si256` 函数执行向量的按位或操作
  return _mm256_or_si256(a, b);
}

// 重载运算符 `^`，用于两个 BFloat16 向量的按位异或操作，返回结果作为 Vectorized<BFloat16>
Vectorized<BFloat16> inline operator^(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  // 使用 AVX2 指令集的 `_mm256_xor_si256` 函数执行向量的按位异或操作
  return _mm256_xor_si256(a, b);
}

// 实现 BFloat16 向量的等于比较操作，返回结果作为 Vectorized<BFloat16>
inline Vectorized<BFloat16> Vectorized<BFloat16>::eq(const Vectorized<BFloat16>& other) const {
  // 使用当前对象和 other 进行等于比较，然后将比较结果与全 1 的 BFloat16 向量按位与
  return (*this == other) & Vectorized<BFloat16>(1.0f);
}

// 实现 BFloat16 向量的不等于比较操作，返回结果作为 Vectorized<BFloat16>
inline Vectorized<BFloat16> Vectorized<BFloat16>::ne(const Vectorized<BFloat16>& other) const {
  // 使用当前对象和 other 进行不等于比较，然后将比较结果与全 1 的 BFloat16 向量按位与
  return (*this != other) & Vectorized<BFloat16>(1.0f);
}

// 实现 BFloat16 向量的大于比较操作，返回结果作为 Vectorized<BFloat16>
inline Vectorized<BFloat16> Vectorized<BFloat16>::gt(const Vectorized<BFloat16>& other) const {
  // 使用当前对象和 other 进行大于比较，然后将比较结果与全 1 的 BFloat16 向量按位与
  return (*this > other) & Vectorized<BFloat16>(1.0f);
}

// 实现 BFloat16 向量的大于等于比较操作，返回结果作为 Vectorized<BFloat16>
inline Vectorized<BFloat16> Vectorized<BFloat16>::ge(const Vectorized<BFloat16>& other) const {
  // 使用当前对象和 other 进行大于等于比较，然后将比较结果与全 1 的 BFloat16 向量按位与
  return (*this >= other) & Vectorized<BFloat16>(1.0f);
}

// 实现 BFloat16 向量的小于比较操作，返回结果作为 Vectorized<BFloat16>
inline Vectorized<BFloat16> Vectorized<BFloat16>::lt(const Vectorized<BFloat16>& other) const {
  // 使用当前对象和 other 进行小于比较，然后将比较结果与全 1 的 BFloat16 向量按位与
  return (*this < other) & Vectorized<BFloat16>(1.0f);
}

// 实现 BFloat16 向量的小于等于比较操作，返回结果作为 Vectorized<BFloat16>
inline Vectorized<BFloat16> Vectorized<BFloat16>::le(const Vectorized<BFloat16>& other) const {
  // 使用当前对象和 other 进行小于等于比较，然后将比较结果与全 1 的 BFloat16 向量按位与
  return (*this <= other) & Vectorized<BFloat16>(1.0f);
}

// 实现 BFloat16 向量的小数部分计算，通过减去截断值实现
inline Vectorized<BFloat16> Vectorized<BFloat16>::frac() const {
  // 使用当前对象减去其截断值（整数部分），返回结果
  return *this - this->trunc();
}

// 实现 IEEE 754 201X 标准的最大值操作，如果输入中有 NaN，则结果也是 NaN
template <>
Vectorized<BFloat16> inline maximum(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  // 将 BFloat16 向量转换为 float32 向量，分别处理低位和高位
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  // 分别找出 a 和 b 的低位和高位的最大值
  auto max_lo = _mm256_max_ps(a_lo, b_lo);
  auto max_hi = _mm256_max_ps(a_hi, b_hi);
  // 检查 a 和 b 的低位和高位是否包含 NaN
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // 将 NaN 标记应用到最大值结果中
  auto o1 = _mm256_or_ps(max_lo, nan_lo);
  auto o2 = _mm256_or_ps(max_hi, nan_hi);
  // 将 float32 结果转换回 BFloat16 向量并返回
  return cvtfp32_bf16(o1, o2);
}
Vectorized<BFloat16> inline minimum(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  // 声明两个 AVX 寄存器来存储 a 和 b 的低位和高位
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  // 将 BFloat16 向量转换为单精度浮点数向量
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  // 分别计算低位和高位的最小值
  auto min_lo = _mm256_min_ps(a_lo, b_lo);
  auto min_hi = _mm256_min_ps(a_hi, b_hi);
  // 检测 NaN 并将其置为全 1
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // 利用全 1 值将 NaN 传播到最小值结果中
  auto o1 = _mm256_or_ps(min_lo, nan_lo);
  auto o2 = _mm256_or_ps(min_hi, nan_hi);
  // 将结果转换为 BFloat16 向量返回
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp(const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min, const Vectorized<BFloat16>& max) {
  // 声明 AVX 寄存器存储 a, min 和 max 的低位和高位
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  __m256 max_lo, max_hi;
  // 将 BFloat16 向量转换为单精度浮点数向量
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(min), min_lo, min_hi);
  cvtbf16_fp32(__m256i(max), max_lo, max_hi);
  // 计算每个向量元素的 clamp 操作结果
  auto o1 = _mm256_min_ps(max_lo, _mm256_max_ps(min_lo, a_lo));
  auto o2 = _mm256_min_ps(max_hi, _mm256_max_ps(min_hi, a_hi));
  // 将结果转换为 BFloat16 向量返回
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_max(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& max) {
  // 声明 AVX 寄存器存储 a 和 max 的低位和高位
  __m256 a_lo, a_hi;
  __m256 max_lo, max_hi;
  // 将 BFloat16 向量转换为单精度浮点数向量
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(max), max_lo, max_hi);
  // 计算每个向量元素的 clamp_max 操作结果
  auto o1 = _mm256_min_ps(max_lo, a_lo);
  auto o2 = _mm256_min_ps(max_hi, a_hi);
  // 将结果转换为 BFloat16 向量返回
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_min(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& min) {
  // 声明 AVX 寄存器存储 a 和 min 的低位和高位
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  // 将 BFloat16 向量转换为单精度浮点数向量
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(min), min_lo, min_hi);
  // 计算每个向量元素的 clamp_min 操作结果
  auto o1 = _mm256_max_ps(min_lo, a_lo);
  auto o2 = _mm256_max_ps(min_hi, a_hi);
  // 将结果转换为 BFloat16 向量返回
  return cvtfp32_bf16(o1, o2);
}

template <>
inline void convert(const BFloat16* src, BFloat16* dst, int64_t n) {
  int64_t i;
  // 循环展开处理 AVX 大小的数据块
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<BFloat16>::size()); i += Vectorized<BFloat16>::size()) {
    // 加载 BFloat16 数据并存储为 AVX 寄存器
    auto vsrc = _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>((void*)(dst + i)), vsrc);
  }
  // 处理余下的不足 AVX 大小的数据
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    // 使用单元素转换函数处理剩余的数据元素
    dst[i] = src[i];
  }
}

template <>
inline void convert(const float* src, BFloat16* dst, int64_t n) {
  int64_t i;
  // 处理连续的 AVX 大小数据块
  for (i = 0; i + Vectorized<BFloat16>::size() <= n; i += Vectorized<BFloat16>::size()) {
    // 分别加载两个连续的 float 向量并转换为 BFloat16
    __m256 a = _mm256_loadu_ps(&src[i]);
    __m256 b = _mm256_loadu_ps(&src[i + 8]);

    __m256i bf = cvtfp32_bf16(a, b);
    // 存储结果到 BFloat16 数组
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), bf);
  }
  // 处理余下的不足 AVX 大小的数据
  for (; i < n; i++) {
    // 使用 C10 库提供的单个元素转换函数处理
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
inline void convert(const double* src, BFloat16* dst, int64_t n) {
  auto load_float = [](const double *src) -> __m256 {
    // 从双精度数组中加载一个 float 向量
  __m128 a = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
  // 将 src 指向的 256 位双精度数据加载为 128 位单精度数据，存储在 a 中
  __m128 b = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
  // 将 src + 4 指向的下一个 256 位双精度数据加载为 128 位单精度数据，存储在 b 中
  return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  // 将 a 和 b 合并成一个 256 位单精度数据，存储在一个新的 AVX 寄存器中并返回
};

int64_t i;
for (i = 0; i + Vectorized<BFloat16>::size() <= n; i += Vectorized<BFloat16>::size()) {
  // 从 src 中加载一组大小为 Vectorized<BFloat16>::size() 的浮点数向量到 a 和 b
  __m256 a = load_float(&src[i]);
  __m256 b = load_float(&src[i + 8]);

  // 将浮点数向量 a 和 b 转换为 bf16 格式
  __m256i bf = cvtfp32_bf16(a, b);
  // 将结果存储到 dst 中
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), bf);
}
for (; i < n; i++) {
  // 将 src[i] 的浮点数值转换为 BFloat16 格式并存储到 dst 中
  dst[i] = c10::convert<BFloat16>(src[i]);
}
}

template <>
Vectorized<BFloat16> inline fmadd(const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b, const Vectorized<BFloat16>& c) {
  // 声明四个 __m256 变量，用于存储向量操作结果的低位和高位部分
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  __m256 c_lo, c_hi;
  // 将向量 a 转换为浮点数向量并分别存储到 a_lo 和 a_hi
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  // 将向量 b 转换为浮点数向量并分别存储到 b_lo 和 b_hi
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  // 将向量 c 转换为浮点数向量并分别存储到 c_lo 和 c_hi
  cvtbf16_fp32(__m256i(c), c_lo, c_hi);
  // 使用 AVX 指令执行 fused multiply-add 操作，将结果存储在 o1 和 o2 中
  auto o1 = _mm256_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm256_fmadd_ps(a_hi, b_hi, c_hi);
  // 将浮点数向量 o1 和 o2 转换为 BFloat16 向量并返回
  return cvtfp32_bf16(o1, o2);
}

template <>
class Vectorized<Half>: public Vectorized16<Half> {
public:
  using Vectorized16::Vectorized16;

  // 下面是各种运算符重载函数，实现 Half 精度的向量运算
  Vectorized<Half> frac() const;

  Vectorized<Half> eq(const Vectorized<Half>& other) const;
  Vectorized<Half> ne(const Vectorized<Half>& other) const;
  Vectorized<Half> gt(const Vectorized<Half>& other) const;
  Vectorized<Half> ge(const Vectorized<Half>& other) const;
  Vectorized<Half> lt(const Vectorized<Half>& other) const;
  Vectorized<Half> le(const Vectorized<Half>& other) const;
};

// 各种算术运算符的实现，通过调用 binary_op_as_fp32 函数实现对应的操作
Vectorized<Half> inline operator+(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_add_ps(x, y); });
}
Vectorized<Half> inline operator-(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_sub_ps(x, y); });
}
Vectorized<Half> inline operator*(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_mul_ps(x, y); });
}
Vectorized<Half> inline operator/(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_div_ps(x, y); });
}
Vectorized<Half> inline operator&(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  // 使用 AVX 指令执行位与操作
  return _mm256_and_si256(a, b);
}
Vectorized<Half> inline operator|(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  // 使用 AVX 指令执行位或操作
  return _mm256_or_si256(a, b);
}
Vectorized<Half> inline operator^(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  // 使用 AVX 指令执行位异或操作
  return _mm256_xor_si256(a, b);
}

// 下面是比较运算符的重载实现，返回 Half 精度的向量结果
inline Vectorized<Half> Vectorized<Half>::eq(const Vectorized<Half>& other) const {
  return (*this == other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::ne(const Vectorized<Half>& other) const {
  return (*this != other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::gt(const Vectorized<Half>& other) const {
  return (*this > other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::ge(const Vectorized<Half>& other) const {
  return (*this >= other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::lt(const Vectorized<Half>& other) const {
  return (*this < other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::le(const Vectorized<Half>& other) const {
  return (*this <= other) & Vectorized<Half>(1.0f);
}
// 实现了 `frac` 函数，用于计算向量中每个元素的小数部分，通过减去其截断部分实现
inline Vectorized<Half> Vectorized<Half>::frac() const {
  return *this - this->trunc();
}

// 实现了 IEEE 754 201X 标准中的 `maximum` 操作，如果任一输入为 NaN，则传播 NaN
template <>
Vectorized<Half> inline maximum(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  // 将向量 a 和 b 的半精度值转换为单精度，并分别存储在 a_lo, a_hi 和 b_lo, b_hi 中
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  // 计算每个向量的最大值，并检查 NaN 的情况
  auto max_lo = _mm256_max_ps(a_lo, b_lo);
  auto max_hi = _mm256_max_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // 如果有 NaN，则结果中对应位设置为 NaN
  auto o1 = _mm256_or_ps(max_lo, nan_lo);
  auto o2 = _mm256_or_ps(max_hi, nan_hi);
  // 将结果从单精度转回半精度
  return cvtfp32_fp16(o1, o2);
}

// 实现了 IEEE 754 201X 标准中的 `minimum` 操作，如果任一输入为 NaN，则传播 NaN
template <>
Vectorized<Half> inline minimum(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  // 将向量 a 和 b 的半精度值转换为单精度，并分别存储在 a_lo, a_hi 和 b_lo, b_hi 中
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  // 计算每个向量的最小值，并检查 NaN 的情况
  auto min_lo = _mm256_min_ps(a_lo, b_lo);
  auto min_hi = _mm256_min_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // 如果有 NaN，则结果中对应位设置为 NaN
  auto o1 = _mm256_or_ps(min_lo, nan_lo);
  auto o2 = _mm256_or_ps(min_hi, nan_hi);
  // 将结果从单精度转回半精度
  return cvtfp32_fp16(o1, o2);
}

// 实现了 clamp 函数，将向量 a 中的每个元素限制在指定的最小和最大值之间
template <>
Vectorized<Half> inline clamp(const Vectorized<Half>& a,
    const Vectorized<Half>& min, const Vectorized<Half>& max) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  __m256 max_lo, max_hi;
  // 将向量 a, min 和 max 的半精度值转换为单精度，并分别存储在对应的向量中
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(min), min_lo, min_hi);
  cvtfp16_fp32(__m256i(max), max_lo, max_hi);
  // 对每个元素进行 clamp 操作，确保它们在 min 和 max 之间
  auto o1 = _mm256_min_ps(max_lo, _mm256_max_ps(min_lo, a_lo));
  auto o2 = _mm256_min_ps(max_hi, _mm256_max_ps(min_hi, a_hi));
  // 将结果从单精度转回半精度
  return cvtfp32_fp16(o1, o2);
}

// 实现了 clamp_max 函数，将向量 a 中的每个元素限制在指定的最大值 max 以下
template <>
Vectorized<Half> inline clamp_max(const Vectorized<Half>& a, const Vectorized<Half>& max) {
  __m256 a_lo, a_hi;
  __m256 max_lo, max_hi;
  // 将向量 a 和 max 的半精度值转换为单精度，并分别存储在对应的向量中
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(max), max_lo, max_hi);
  // 对每个元素进行 clamp_max 操作，确保它们不超过 max
  auto o1 = _mm256_min_ps(max_lo, a_lo);
  auto o2 = _mm256_min_ps(max_hi, a_hi);
  // 将结果从单精度转回半精度
  return cvtfp32_fp16(o1, o2);
}

// 实现了 clamp_min 函数，将向量 a 中的每个元素限制在指定的最小值 min 以上
template <>
Vectorized<Half> inline clamp_min(const Vectorized<Half>& a, const Vectorized<Half>& min) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  // 将向量 a 和 min 的半精度值转换为单精度，并分别存储在对应的向量中
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(min), min_lo, min_hi);
  // 对每个元素进行 clamp_min 操作，确保它们不低于 min
  auto o1 = _mm256_max_ps(min_lo, a_lo);
  auto o2 = _mm256_max_ps(min_hi, a_hi);
  // 将结果从单精度转回半精度
  return cvtfp32_fp16(o1, o2);
}

// 实现了将半精度数组转换为另一个半精度数组的函数，长度为 n
template <>
inline void convert(const Half* src, Half* dst, int64_t n) {
  int64_t i;
  // 循环处理每个向量大小的元素，直到剩余元素不足一个向量大小
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<Half>::size()); i += Vectorized<Half>::size()) {
    # 使用 AVX 指令集加载从源地址开始的一组 256 位数据到寄存器 vsrc
    auto vsrc = _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
    # 将寄存器 vsrc 中的数据存储到目标地址开始的一组 256 位数据中
    _mm256_storeu_si256(reinterpret_cast<__m256i*>((void*)(dst + i)), vsrc);
#ifndef __msvc_cl__
#pragma unroll
#endif
// 如果不是在 MSVC 编译器下，添加 #pragma unroll 指令

for (; i < n; i++) {
  // 循环复制 src 数组的内容到 dst 数组中
  dst[i] = src[i];
}
}

template <>
inline void convert(const float* src, Half* dst, int64_t n) {
  int64_t i;
  // 使用 SIMD 进行数据转换，每次处理 Vectorized<Half>::size() 个元素
  for (i = 0; i + Vectorized<Half>::size() <= n; i += Vectorized<Half>::size()) {
    // 加载 src 数组中的连续 8 个 float 值到 SIMD 寄存器 a 和 b 中
    __m256 a = _mm256_loadu_ps(&src[i]);
    __m256 b = _mm256_loadu_ps(&src[i + 8]);

    // 将加载的 float 数据转换为 Half 类型，并存储到 dst 数组中
    __m256i c = cvtfp32_fp16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), c);
  }
  // 处理剩余不足 Vectorized<Half>::size() 的元素
  for (; i < n; i++) {
    dst[i] = c10::convert<Half>(src[i]);
  }
}

template <>
inline void convert(const double* src, Half* dst, int64_t n) {
  auto load_float = [](const double *src) -> __m256 {
    // 从双精度数组中加载一个 float 向量
    __m128 a = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
    __m128 b = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  };

  int64_t i;
  // 使用 SIMD 进行数据转换，每次处理 Vectorized<Half>::size() 个元素
  for (i = 0; i + Vectorized<Half>::size() <= n; i += Vectorized<Half>::size()) {
    // 加载 src 数组中的连续 8 个 double 值转换为 float，存储到 SIMD 寄存器 a 和 b 中
    __m256 a = load_float(&src[i]);
    __m256 b = load_float(&src[i + 8]);

    // 将加载的 float 数据转换为 Half 类型，并存储到 dst 数组中
    __m256i c = cvtfp32_fp16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), c);
  }
  // 处理剩余不足 Vectorized<Half>::size() 的元素
  for (; i < n; i++) {
    dst[i] = c10::convert<Half>(src[i]);
  }
}

template <>
Vectorized<Half> inline fmadd(const Vectorized<Half>& a,
    const Vectorized<Half>& b, const Vectorized<Half>& c) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  __m256 c_lo, c_hi;
  // 将 Half 类型转换为 float 类型的 SIMD 寄存器
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  cvtfp16_fp32(__m256i(c), c_lo, c_hi);
  // 使用 SIMD 执行 fmadd 操作，并将结果转换为 Half 类型返回
  auto o1 = _mm256_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm256_fmadd_ps(a_hi, b_hi, c_hi);
  return cvtfp32_fp16(o1, o2);
}

#define CONVERT_VECTORIZED_INIT(type, name) \
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  __m256 o1, o2; \
  // 将 Vectorized<type> 类型转换为 float 类型的 SIMD 寄存器 o1 和 o2
  cvt_to_fp32<type>(__m256i(a), o1, o2); \
  return std::make_tuple(o1, o2); \
} \
// 将 float 类型的 SIMD 寄存器转换为 Vectorized<type> 类型
inline Vectorized<type> convert_float_##name(const Vectorized<float>& a, const Vectorized<float>& b) { \
  return cvt_from_fp32<type>(__m256(a), __m256(b)); \
}

// 对于 AVX2 不支持的 CPU 架构，使用非向量化的初始化宏
#else // defined(CPU_CAPABILITY_AVX2)

#define CONVERT_NON_VECTORIZED_INIT(type, name) \
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  constexpr int64_t K = Vectorized<type>::size(); \
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  a.store(arr2); \
  convert(arr2, arr, K); \
  return std::make_tuple( \
      Vectorized<float>::loadu(arr), \
      Vectorized<float>::loadu(arr + Vectorized<float>::size())); \
} \
#define LOAD_FP32_VECTORIZED_INIT(type, name) \  // 定义宏 LOAD_FP32_VECTORIZED_INIT，用于加载特定类型的数据并转换为单精度浮点数向量化格式
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out) { \  // 定义函数 load_fp32_from_name，将数据加载为单精度浮点数向量并存储到 out 中
  auto values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data)); \  // 使用 SSE 指令加载 data 指向的数据到 values
  __m256 out_values; \  // 声明一个 AVX 256 位寄存器用于存储转换后的单精度浮点数数据
  cvt_to_fp32<type>(values, out_values); \  // 调用 cvt_to_fp32 函数将类型为 type 的数据转换为单精度浮点数并存储到 out_values 中
  out = out_values; \  // 将 out_values 赋值给输出向量 out
} \  // 函数定义结束

inline void load_fp32_from_##name(const type *data, Vectorized<float>& out1, Vectorized<float>& out2) { \  // 定义函数 load_fp32_from_name，将数据加载为两个单精度浮点数向量并分别存储到 out1 和 out2 中
  auto vec = Vectorized<type>::loadu(data); \  // 使用 Vectorized<type>::loadu 函数加载 data 指向的数据到向量 vec 中
  __m256 out1_values, out2_values; \  // 声明两个 AVX 256 位寄存器用于存储转换后的单精度浮点数数据
  cvt_to_fp32<type>(vec, out1_values, out2_values); \  // 调用 cvt_to_fp32 函数将类型为 type 的数据转换为两个单精度浮点数向量并分别存储到 out1_values 和 out2_values 中
  out1 = out1_values; \  // 将 out1_values 赋值给 out1
  out2 = out2_values; \  // 将 out2_values 赋值给 out2
} \  // 函数定义结束
// 定义一个加载单精度浮点数向量的内联函数，函数名根据给定的名称动态生成
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out) { \
  // 使用 Vectorized<float> 类定义的大小创建一个 float 类型的数组 values
  __at_align__ float values[Vectorized<float>::size()]; \
  // 遍历 Vectorized<float> 的大小范围，将数据从输入数组 data 复制到 values 数组中
  for (const auto k : c10::irange(Vectorized<float>::size())) { \
    values[k] = data[k]; \
  } \
  // 使用 Vectorized<float> 类的静态方法 loadu 加载未对齐的数据到 out 向量中
  out = Vectorized<float>::loadu(values); \
} \
\
// 定义一个加载两个单精度浮点数向量的内联函数，函数名根据给定的名称动态生成
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out1, Vectorized<float>& out2) { \
  // 调用前面定义的加载单精度浮点数向量的函数来加载第一个向量
  load_fp32_from_##name(data, out1); \
  // 将输入数据指针向后移动一个向量的大小
  data += Vectorized<float>::size(); \
  // 再次调用加载单精度浮点数向量的函数来加载第二个向量
  load_fp32_from_##name(data, out2); \
}
// 使用 BFloat16 类型和名称 "bf16" 生成的加载单精度浮点数向量的初始化函数
LOAD_FP32_NON_VECTORIZED_INIT(BFloat16, bf16);
// 使用 Half 类型和名称 "fp16" 生成的加载单精度浮点数向量的初始化函数
LOAD_FP32_NON_VECTORIZED_INIT(Half, fp16);

#endif
}} // namsepace at::vec::CPU_CAPABILITY

// 恢复之前保存的 GCC 编译器诊断状态
#pragma GCC diagnostic pop
```