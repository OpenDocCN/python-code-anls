# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_convert.h`

```py
#pragma once
// 使用预处理指令#pragma once确保头文件只被包含一次，避免重复定义问题

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec512/vec512_bfloat16.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_convert.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

template <>
// 模板特化：将BFloat16类型向量转换为float类型向量
struct VecConvert<float, 1, BFloat16, 1> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<BFloat16, 1>& src) {
    VectorizedN<float, 1> result;
    __m512 value;
    // 调用AVX512指令将BFloat16向量转换为float向量
    cvtbf16_fp32(_mm512_castsi512_si256(src[0]), value);
    result[0] = value;
    return result;
  }
};

template <>
// 模板特化：将Half类型向量转换为float类型向量
struct VecConvert<float, 1, Half, 1> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<Half, 1>& src) {
    VectorizedN<float, 1> result;
    __m512 value;
    // 调用AVX512指令将Half向量转换为float向量
    cvtfp16_fp32(_mm512_castsi512_si256(src[0]), value);
    result[0] = value;
    return result;
  }
};

template <>
// 模板特化：将float类型向量转换为BFloat16类型向量
struct VecConvert<BFloat16, 1, float, 1> {
  static inline VectorizedN<BFloat16, 1> apply(
      const VectorizedN<float, 1>& src) {
    VectorizedN<BFloat16, 1> result;
    // 调用AVX512指令将float向量转换为BFloat16向量
    result[0] = _mm512_castsi256_si512(cvtfp32_bf16(src[0]));
    return result;
  }
};

template <>
// 模板特化：将float类型向量转换为Half类型向量
struct VecConvert<Half, 1, float, 1> {
  static inline VectorizedN<Half, 1> apply(const VectorizedN<float, 1>& src) {
    VectorizedN<Half, 1> result;
    // 调用AVX512指令将float向量转换为Half向量
    result[0] = _mm512_castsi256_si512(cvtfp32_fp16(src[0]));
    return result;
  }
};

template <>
// 模板特化：将int64_t类型向量转换为float类型向量
struct VecConvert<float, 1, int64_t, 2> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    auto low = _mm512_cvtepi64_ps(src[0]);
    auto high = _mm512_cvtepi64_ps(src[1]);
    // 合并两个AVX512向量并返回一个float向量
    return Vectorized<float>(
        _mm512_insertf32x8(_mm512_castps256_ps512(low), high, 1));
  }
};

template <>
// 模板特化：将float类型向量转换为int64_t类型向量
struct VecConvert<int64_t, 2, float, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<float, 1>& src) {
    at::vec::VectorizedN<int64_t, 2> result;
    // 将AVX512 float向量转换为两个int64_t向量
    result[0] = _mm512_cvt_roundps_epi64(
        _mm512_castps512_ps256(src[0]), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    result[1] = _mm512_cvt_roundps_epi64(
        _mm512_extractf32x8_ps(src[0], 1),
        _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    return result;
  }
};

template <>
// 模板特化：将int64_t类型向量转换为int32_t类型向量
struct VecConvert<int32_t, 1, int64_t, 2> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    auto low = _mm512_cvtepi64_epi32(src[0]);
    auto high = _mm512_cvtepi64_epi32(src[1]);
    // 合并两个AVX512 int32_t向量并返回一个int32_t向量
    return Vectorized<int32_t>(
        _mm512_inserti32x8(_mm512_castsi256_si512(low), high, 1));
  }
};

template <>
// 模板特化：将int32_t类型向量转换为int64_t类型向量
struct VecConvert<int64_t, 2, int32_t, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<int32_t, 1>& src) {
    at::vec::VectorizedN<int64_t, 2> result;
    // 将AVX512 int32_t向量转换为两个int64_t向量
    result[0] = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src[0]));
    result[1] = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src[0], 1));
    return result;
  }
};

#endif // defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

} // inline namespace CPU_CAPABILITY
} // namespace at::vec
// 定义模板结构 VecConvert，用于将 int32_t 向量转换为 int8_t 向量
struct VecConvert<int32_t, 1, int8_t, 1> {
  // 静态方法，接收 int8_t 类型的向量 src，返回 int32_t 类型的向量
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<int8_t, 1>& src) {
    // 将 src[0] 转换为 __m128i 类型
    auto src128 = _mm512_castsi512_si128(src[0]);
    // 返回一个 int32_t 向量，使用 _mm512_cvtepi8_epi32 函数将 src128 转换为 int32_t 向量
    return Vectorized<int32_t>(_mm512_cvtepi8_epi32(src128));
  }
};

// 特化模板结构 VecConvert，将 uint8_t 向量转换为 int32_t 向量
template <>
struct VecConvert<int32_t, 1, uint8_t, 1> {
  // 静态方法，接收 uint8_t 类型的向量 src，返回 int32_t 类型的向量
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<uint8_t, 1>& src) {
    // 将 src[0] 转换为 __m128i 类型
    auto src128 = _mm512_castsi512_si128(src[0]);
    // 返回一个 int32_t 向量，使用 _mm512_cvtepu8_epi32 函数将 src128 转换为 int32_t 向量
    return Vectorized<int32_t>(_mm512_cvtepu8_epi32(src128));
  }
};

// 模板结构 VecConvert 的特化，处理条件为浮点数缩减至整数类型或整数缩减至浮点数类型的情况
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
  // 静态方法，接收 src_t 类型的向量 src，返回 dst_t 类型的向量
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<src_t, 1>& src) {
    // 将 src 转换为 float 类型的向量，tmp_fp32
    VectorizedN<float, 1> tmp_fp32 = VecConvert<float, 1, src_t, 1>::apply(src);
    // 返回将 tmp_fp32 转换为 dst_t 类型的向量
    return VecConvert<dst_t, 1, float, 1>::apply(tmp_fp32);
  }
};

// 模板结构 VecConvert 的特化，处理 float 向量转换为 8 位整数类型的情况
template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    1,
    typename std::enable_if_t<is_8bit_integer_v<dst_t>,
        void>> {
  // 静态方法，接收 float 类型的向量 src，返回 dst_t 类型的向量
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 1>& src) {
    // 调用 convert_float_to_int8 函数将 src[0] 转换为 dst_t 类型的值，并返回其向量形式
    return convert_float_to_int8<dst_t>(src[0]);
  }
};

// 模板结构 VecConvert 的特化，处理 8 位整数类型向量转换为 float 类型的情况
template <typename src_t>
struct VecConvert<
    float,
    1,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>,
        void>> {
  // 静态方法，接收 src_t 类型的向量 src，返回 float 类型的向量
  static inline VectorizedN<float, 1> apply(const VectorizedN<src_t, 1>& src) {
    // 调用 convert_int8_to_float 函数将 src[0] 转换为 float 类型的值，并返回其向量形式
    return convert_int8_to_float<src_t>(src[0]);
  }
};

// 模板结构 VecConvert 的特化，处理 int64_t 向量转换为 1 维 8 位整数类型的情况
template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    int64_t,
    2,
    typename std::enable_if<
        std::is_same_v<dst_t, int8_t> ||
        std::is_same_v<dst_t, uint8_t>>::type> {
  // 静态方法，接收 int64_t 类型的向量 src，返回 dst_t 类型的向量
  static inline VectorizedN<dst_t, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    // 调用 VecConvert<int32_t, 1, int64_t, 2>::apply(src) 转换 src 为 int32_t 类型的向量后，
    // 再调用 VecConvert<dst_t, 1, int32_t, 1>::apply 进行最终转换并返回结果
    return VecConvert<dst_t, 1, int32_t, 1>::apply(
        VecConvert<int32_t, 1, int64_t, 2>::apply(src));
  }
};

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
```