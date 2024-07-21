# `.\pytorch\c10\util\BFloat16.h`

```
#pragma once
// 声明了 bloat16 类型（brain floating-point）。此表示法使用
// 1 位用于符号，8 位用于指数，7 位用于尾数。

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <ostream>

#if defined(__CUDACC__) && !defined(USE_ROCM)
#include <cuda_bf16.h>
#endif

#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
#if defined(CL_SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp> // for SYCL 1.2.1
#else
#include <sycl/sycl.hpp> // for SYCL 2020
#endif
#include <ext/oneapi/bfloat16.hpp>
#endif

namespace c10 {

namespace detail {
// 从 uint16_t 转换为 float
inline C10_HOST_DEVICE float f32_from_bits(uint16_t src) {
  float res = 0;
  uint32_t tmp = src;
  tmp <<= 16;

#if defined(USE_ROCM)
  float* tempRes;
  // 在 HIP 环境中，由于严格别名规则的问题，应使用 memcpy。
  tempRes = reinterpret_cast<float*>(&tmp);
  res = *tempRes;
#else
  std::memcpy(&res, &tmp, sizeof(tmp));
#endif

  return res;
}

// 从 float 转换为 uint16_t
inline C10_HOST_DEVICE uint16_t bits_from_f32(float src) {
  uint32_t res = 0;

#if defined(USE_ROCM)
  // 在 HIP 环境中，由于严格别名规则的问题，应使用 memcpy。
  uint32_t* tempRes = reinterpret_cast<uint32_t*>(&src);
  res = *tempRes;
#else
  std::memcpy(&res, &src, sizeof(res));
#endif

  return res >> 16;
}

// 将 float 四舍五入为最接近的偶数的 bfloat16 格式
inline C10_HOST_DEVICE uint16_t round_to_nearest_even(float src) {
#if defined(USE_ROCM)
  if (src != src) {
#elif defined(_MSC_VER)
  if (isnan(src)) {
#else
  if (std::isnan(src)) {
#endif
    // 如果 src 是 NaN，则返回特定的 NaN 表示
    return UINT16_C(0x7FC0);
  } else {
    // 否则进行四舍五入操作
    union {
      uint32_t U32;
      float F32;
    };

    F32 = src;
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}
} // namespace detail

// 定义了 BFloat16 结构体
struct alignas(2) BFloat16 {
  uint16_t x;

  // 根据编译环境设置构造函数的可访问性标签
#if defined(USE_ROCM)
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  // 定义了 from_bits_t 类型
  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  // 使用指定的 bits 构造 BFloat16 类型
  constexpr C10_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t)
      : x(bits) {}

  // 定义了 BFloat16 类型的隐式转换构造函数
  inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;

  // 根据不同的编译环境定义额外的类型转换函数
#if defined(__CUDACC__) && !defined(USE_ROCM)
  inline C10_HOST_DEVICE BFloat16(const __nv_bfloat16& value);
  explicit inline C10_HOST_DEVICE operator __nv_bfloat16() const;
#endif

#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
  inline C10_HOST_DEVICE BFloat16(const sycl::ext::oneapi::bfloat16& value);
  explicit inline C10_HOST_DEVICE operator sycl::ext::oneapi::bfloat16() const;
#endif
};

// 定义了 BFloat16 类型的输出流操作符重载
C10_API inline std::ostream& operator<<(
    std::ostream& out,
    const BFloat16& value) {
  // 输出 BFloat16 对象转换为 float 后的结果
  out << (float)value;
  return out;
}

} // namespace c10
#include <c10/util/BFloat16-inl.h> // IWYU pragma: keep


// 包含 BFloat16-inl.h 头文件，该文件位于 c10/util/ 目录下
// IWYU pragma: keep 指示编译器保持对该头文件的引用，避免优化掉未使用的头文件
```