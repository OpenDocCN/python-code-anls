# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_convert.h`

```
#pragma once
// 在预处理阶段指定此头文件只包含一次

#include <ATen/cpu/vec/functional_bfloat16.h>
// 包含用于 bfloat16 数据类型的向量化函数声明

#include <ATen/cpu/vec/intrinsics.h>
// 包含与特定 CPU 指令集相关的内联汇编函数声明

#include <ATen/cpu/vec/vec_base.h>
// 包含用于向量化操作的基础定义和函数声明

#include <ATen/cpu/vec/vec_convert.h>
// 包含用于不同数据类型之间向量化转换的函数声明

namespace at::vec {
inline namespace CPU_CAPABILITY {
// 进入 at::vec 命名空间和 CPU_CAPABILITY 内联命名空间

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
// 如果定义了 AVX2 指令集，并且不是在 Microsoft Visual Studio 中

template <>
struct VecConvert<float, 1, BFloat16, 1> {
  // 向量化转换模板特化，从 VectorizedN<BFloat16, 1> 到 VectorizedN<float, 1>
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<BFloat16, 1>& src) {
    // 应用向量化转换，将 BFloat16 向量转换为 float 向量
    VectorizedN<float, 1> result;
    // 声明结果向量
    __m256 value;
    // 声明临时变量 value 作为 AVX 寄存器
    cvtbf16_fp32(_mm256_castsi256_si128(src[0]), value);
    // 调用 AVX 汇编函数 cvtbf16_fp32，将 BFloat16 转换为 float
    result[0] = value;
    // 将转换后的值存入结果向量
    return result;
    // 返回转换后的结果向量
  }
};

template <>
struct VecConvert<float, 1, Half, 1> {
  // 向量化转换模板特化，从 VectorizedN<Half, 1> 到 VectorizedN<float, 1>
  static inline VectorizedN<float, 1> apply(const VectorizedN<Half, 1>& src) {
    // 应用向量化转换，将 Half 向量转换为 float 向量
    VectorizedN<float, 1> result;
    // 声明结果向量
    __m256 value;
    // 声明临时变量 value 作为 AVX 寄存器
    cvtfp16_fp32(_mm256_castsi256_si128(src[0]), value);
    // 调用 AVX 汇编函数 cvtfp16_fp32，将 Half 转换为 float
    result[0] = value;
    // 将转换后的值存入结果向量
    return result;
    // 返回转换后的结果向量
  }
};

template <>
struct VecConvert<BFloat16, 1, float, 1> {
  // 向量化转换模板特化，从 VectorizedN<float, 1> 到 VectorizedN<BFloat16, 1>
  static inline VectorizedN<BFloat16, 1> apply(
      const VectorizedN<float, 1>& src) {
    // 应用向量化转换，将 float 向量转换为 BFloat16 向量
    VectorizedN<BFloat16, 1> result;
    // 声明结果向量
    result[0] = _mm256_castsi128_si256(cvtfp32_bf16(src[0]));
    // 使用 AVX 汇编函数 cvtfp32_bf16，将 float 转换为 BFloat16 并存入结果向量
    return result;
    // 返回转换后的结果向量
  }
};

template <>
struct VecConvert<Half, 1, float, 1> {
  // 向量化转换模板特化，从 VectorizedN<float, 1> 到 VectorizedN<Half, 1>
  static inline VectorizedN<Half, 1> apply(const VectorizedN<float, 1>& src) {
    // 应用向量化转换，将 float 向量转换为 Half 向量
    VectorizedN<Half, 1> result;
    // 声明结果向量
    result[0] = _mm256_castsi128_si256(cvtfp32_fp16(src[0]));
    // 使用 AVX 汇编函数 cvtfp32_fp16，将 float 转换为 Half 并存入结果向量
    return result;
    // 返回转换后的结果向量
  }
};

template <>
inline Vectorized<double> convert_to_fp_of_same_size<double>(
    const Vectorized<int64_t>& src);
// 向量化函数特化声明，将 int64_t 向量转换为相同大小的 double 向量

template <>
struct VecConvert<float, 1, int64_t, 2> {
  // 向量化转换模板特化，从 VectorizedN<int64_t, 2> 到 VectorizedN<float, 1>
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    // 应用向量化转换，将 int64_t 向量转换为 float 向量
    auto low_double = at::vec::convert_to_fp_of_same_size<double>(src[0]);
    // 将低位 int64_t 向量转换为相同大小的 double 向量
    auto low = _mm256_cvtpd_ps(low_double);
    // 使用 AVX 汇编函数 _mm256_cvtpd_ps，将 double 向量转换为 float 向量
    auto high_double = at::vec::convert_to_fp_of_same_size<double>(src[1]);
    // 将高位 int64_t 向量转换为相同大小的 double 向量
    auto high = _mm256_cvtpd_ps(high_double);
    // 使用 AVX 汇编函数 _mm256_cvtpd_ps，将 double 向量转换为 float 向量
    return Vectorized<float>(
        _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1));
    // 将低位和高位转换后的 float 向量合并并返回
  }
};

template <>
inline Vectorized<int32_t> convert_to_int_of_same_size<float>(
    const Vectorized<float>& src);
// 向量化函数特化声明，将 float 向量转换为相同大小的 int32_t 向量

template <>
struct VecConvert<int64_t, 2, float, 1> {
  // 向量化转换模板特化，从 VectorizedN<float, 1> 到 VectorizedN<int64_t, 2>
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<float, 1>& src) {
    // 应用向量化转换，将 float 向量转换为 int64_t 向量
    at::vec::VectorizedN<int64_t, 2> result;
    // 声明结果向量
    auto int32_vec = at::vec::convert_to_int_of_same_size(src[0]);
    // 将 float 向量转换为相同大小的 int32_t 向量
    result[0] = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(int32_vec));
    // 使用 AVX 汇编函数 _mm256_cvtepi32_epi64，将 int32_t 向量转换为 int64_t 向量的低位
    result[1] = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(int32_vec, 1));
    // 使用 AVX 汇编函数 _mm256_cvtepi32_epi64，将 int32_t 向量转换为 int64_t 向量的高位
    return result;
    // 返回转换后的结果向量
  }
};

template <>
struct VecConvert<int32_t, 1, int64_t, 2> {
  // 向量化转换模板特化，从 VectorizedN<int64_t, 2> 到 VectorizedN<int32_t, 1>
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    // 应用向量化转换，将 int64_t 向量转换为 int32_t 向量
    auto low = _mm256_shuffle_epi32(src[0], _MM_SHUFFLE(2, 0, 2,
    return Vectorized<int32_t>(_mm256_blend_epi32(low_perm, high_perm, 0xF0));


    # 使用 AVX2 指令集中的 `_mm256_blend_epi32` 函数，混合低位和高位的 256 位整数向量
    return Vectorized<int32_t>(_mm256_blend_epi32(low_perm, high_perm, 0xF0));
};

// 特化模板，将长度为1的int64_t向量转换为长度为1的int32_t向量
template <>
struct VecConvert<int64_t, 2, int32_t, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<int32_t, 1>& src) {
    // 创建长度为2的int64_t向量
    at::vec::VectorizedN<int64_t, 2> result;
    // 将src的第一个元素转换为int64_t类型，并存入result的第一个位置
    result[0] = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(src[0]));
    // 将src的第一个元素的第二部分（128位）转换为int64_t类型，并存入result的第二个位置
    result[1] = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(src[0], 1));
    return result;
  }
};

// 特化模板，将长度为1的int32_t向量转换为长度为1的int8_t向量
template <>
struct VecConvert<int32_t, 1, int8_t, 1> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<int8_t, 1>& src) {
    // 将src的第一个元素转换为128位int32_t向量，并返回
    auto src128 = _mm256_castsi256_si128(src[0]);
    return Vectorized<int32_t>(_mm256_cvtepi8_epi32(src128));
  }
};

// 特化模板，将长度为1的int32_t向量转换为长度为1的uint8_t向量
template <>
struct VecConvert<int32_t, 1, uint8_t, 1> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<uint8_t, 1>& src) {
    // 将src的第一个元素转换为128位int32_t向量，并返回
    auto src128 = _mm256_castsi256_si128(src[0]);
    return Vectorized<int32_t>(_mm256_cvtepu8_epi32(src128));
  }
};

// 特化模板，当目标类型为长度为1的dst_t，源类型为长度为1的src_t，并满足特定条件时的向量转换
template <typename dst_t, typename src_t>
struct VecConvert<
    dst_t,
    1,
    src_t,
    1,
    typename std::enable_if_t<
        (is_reduced_floating_point_v<dst_t> && is_8bit_integer_v<src_t>) ||
            (is_reduced_floating_point_v<src_t> && is_8bit_integer_v<dst_t>),
        void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<src_t, 1>& src) {
    // 将src_t类型的向量转换为float类型的向量，并返回
    VectorizedN<float, 1> tmp_fp32 = VecConvert<float, 1, src_t, 1>::apply(src);
    // 将float类型的向量再转换为dst_t类型的向量，并返回
    return VecConvert<dst_t, 1, float, 1>::apply(tmp_fp32);
  }
};

// 特化模板，将长度为1的float向量转换为长度为1的dst_t向量
template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    1,
    typename std::enable_if_t<is_8bit_integer_v<dst_t>,
        void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 1>& src) {
    // 将float向量转换为dst_t类型的向量，并返回
    return convert_float_to_int8<dst_t>(src[0]);
  }
};

// 特化模板，将长度为1的src_t向量转换为长度为1的float向量
template <typename src_t>
struct VecConvert<
    float,
    1,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>,
        void>> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<src_t, 1>& src) {
    // 将src_t类型的向量转换为float向量，并返回其中的第一个部分
    auto [res_vec1, res_vec2] = convert_to_float<src_t>(src[0]);
    return res_vec1;
  }
};

// 特化模板，将长度为1的float向量转换为长度为1的dst_t向量
template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    1,
    typename std::enable_if_t<is_reduced_floating_point_v<dst_t>, void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 1>& src) {
    // 将float向量转换为dst_t类型的向量，并返回
    // 该部分代码未完整，后续内容省略

    // 将float向量转换为dst_t类型的向量，并返回
    // 该部分代码未完整，后续内容省略
    // 这里需要根据实际情况补充代码
    # 返回从浮点数 src[0] 转换为目标类型 dst_t 后的结果
    return convert_from_float<dst_t>(src[0], src[0]);
  }
};

// 结束 CPU_CAPABILITY 命名空间
} // namespace CPU_CAPABILITY
// 结束 at::vec 命名空间
} // namespace at::vec
```