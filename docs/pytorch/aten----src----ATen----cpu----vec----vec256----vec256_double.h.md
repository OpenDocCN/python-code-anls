# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_double.h`

```
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#if defined(CPU_CAPABILITY_AVX2)
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at::vec {

// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

// 定义模板特化类 Vectorized<double>
template <> class Vectorized<double> {
private:
  // 使用 AVX 指令集定义存储值的变量
  __m256d values;

public:
  // 类型别名定义
  using value_type = double;
  using size_type = int;

  // 返回向量大小的静态成员函数
  static constexpr size_type size() {
    return 4;
  }

  // 默认构造函数
  Vectorized() {}

  // 接受 __m256d 类型参数的构造函数
  Vectorized(__m256d v) : values(v) {}

  // 根据单个 double 值构造 Vectorized 对象的构造函数
  Vectorized(double val) {
    values = _mm256_set1_pd(val);
  }

  // 根据四个 double 值构造 Vectorized 对象的构造函数
  Vectorized(double val1, double val2, double val3, double val4) {
    values = _mm256_setr_pd(val1, val2, val3, val4);
  }

  // 类型转换运算符，将对象转换为 __m256d 类型
  operator __m256d() const {
    return values;
  }

  // 模板函数，根据掩码 mask 合并两个 Vectorized<double> 对象
  template <int64_t mask>
  static Vectorized<double> blend(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_blend_pd(a.values, b.values, mask);
  }

  // 根据掩码 mask 合并两个 Vectorized<double> 对象的函数
  static Vectorized<double> blendv(const Vectorized<double>& a, const Vectorized<double>& b,
                               const Vectorized<double>& mask) {
    return _mm256_blendv_pd(a.values, b.values, mask.values);
  }

  // 生成等差数列的函数，以 double 类型返回
  template<typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(base, base + step, base + 2 * step, base + 3 * step);
  }

  // 设置 Vectorized<double> 对象的函数，根据 count 参数设定返回值
  static Vectorized<double> set(const Vectorized<double>& a, const Vectorized<double>& b,
                            int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
    }
    return b;
  }

  // 加载未对齐内存中的数据到 Vectorized<double> 对象的函数
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));

    // 对于不满足 size() 的情况，手动加载数据到临时数组并返回
    __at_align__ double tmp_values[size()];
    // 用循环初始化临时数组，避免未初始化内存的问题，参见链接注释
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(double));
    return _mm256_load_pd(tmp_values);
  }

  // 将 Vectorized<double> 对象的数据存储到内存中的函数
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      // 对于不满足 size() 的情况，先将数据存储到临时数组，再复制到目标地址
      double tmp_values[size()];
      _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(double));
    }
  }
};
#endif // CPU_CAPABILITY_AVX2

} // inline namespace CPU_CAPABILITY
} // namespace at::vec
  Vectorized<double> expm1() const {
    // 计算 exp(x) - 1 的向量化操作
    return Vectorized<double>(Sleef_expm1d4_u10(values));
  }
  Vectorized<double> exp_u20() const {
    // 返回一个使用 exp 函数计算的 Vectorized<double> 对象
    return exp();
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    // 返回一个使用 Sleef 库中 fmod 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_fmodd4(values, q));
  }
  Vectorized<double> hypot(const Vectorized<double> &b) const {
    // 返回一个使用 Sleef 库中 hypot 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_hypotd4_u05(values, b));
  }
  Vectorized<double> i0() const {
    // 返回一个通过 map 函数计算的 Vectorized<double> 对象，使用 calc_i0 函数
    return map(calc_i0);
  }
  Vectorized<double> i0e() const {
    // 返回一个通过 map 函数计算的 Vectorized<double> 对象，使用 calc_i0e 函数
    return map(calc_i0e);
  }
  Vectorized<double> digamma() const {
    // 返回一个通过 map 函数计算的 Vectorized<double> 对象，使用 calc_digamma 函数
    return map(calc_digamma);
  }
  Vectorized<double> igamma(const Vectorized<double> &x) const {
    // 将当前对象的值存储到临时数组 tmp 中，并将 x 的值存储到临时数组 tmp_x 中
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    // 对于每个索引 i 在 [0, size()) 范围内，计算 calc_igamma(tmp[i], tmp_x[i])
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    // 返回从临时数组 tmp 中加载的 Vectorized<double> 对象
    return loadu(tmp);
  }
  Vectorized<double> igammac(const Vectorized<double> &x) const {
    // 将当前对象的值存储到临时数组 tmp 中，并将 x 的值存储到临时数组 tmp_x 中
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    // 对于每个索引 i 在 [0, size()) 范围内，计算 calc_igammac(tmp[i], tmp_x[i])
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    // 返回从临时数组 tmp 中加载的 Vectorized<double> 对象
    return loadu(tmp);
  }
  Vectorized<double> log() const {
    // 返回一个使用 Sleef 库中 log 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_logd4_u10(values));
  }
  Vectorized<double> log2() const {
    // 返回一个使用 Sleef 库中 log2 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_log2d4_u10(values));
  }
  Vectorized<double> log10() const {
    // 返回一个使用 Sleef 库中 log10 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_log10d4_u10(values));
  }
  Vectorized<double> log1p() const {
    // 返回一个使用 Sleef 库中 log1p 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_log1pd4_u10(values));
  }
  Vectorized<double> sin() const {
    // 返回一个使用 Sleef 库中 sin 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_sind4_u10(values));
  }
  Vectorized<double> sinh() const {
    // 返回一个使用 Sleef 库中 sinh 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_sinhd4_u10(values));
  }
  Vectorized<double> cos() const {
    // 返回一个使用 Sleef 库中 cos 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_cosd4_u10(values));
  }
  Vectorized<double> cosh() const {
    // 返回一个使用 Sleef 库中 cosh 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_coshd4_u10(values));
  }
  Vectorized<double> ceil() const {
    // 返回一个使用 AVX 指令集中 _mm256_ceil_pd 函数计算的 Vectorized<double> 对象
    return _mm256_ceil_pd(values);
  }
  Vectorized<double> floor() const {
    // 返回一个使用 AVX 指令集中 _mm256_floor_pd 函数计算的 Vectorized<double> 对象
    return _mm256_floor_pd(values);
  }
  Vectorized<double> frac() const;
  Vectorized<double> neg() const {
    // 返回一个将当前值按位异或来实现的 Vectorized<double> 对象，用于取负数
    return _mm256_xor_pd(_mm256_set1_pd(-0.), values);
  }
  Vectorized<double> nextafter(const Vectorized<double> &b) const {
    // 返回一个使用 Sleef 库中 nextafter 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_nextafterd4(values, b));
  }
  Vectorized<double> round() const {
    // 返回一个使用 AVX 指令集中 _mm256_round_pd 函数计算的 Vectorized<double> 对象，执行向最近整数舍入
    return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> tan() const {
    // 返回一个使用 Sleef 库中 tan 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_tand4_u10(values));
  }
  Vectorized<double> tanh() const {
    // 返回一个使用 Sleef 库中 tanh 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_tanhd4_u10(values));
  }
  Vectorized<double> trunc() const {
    // 返回一个使用 AVX 指令集中 _mm256_round_pd 函数计算的 Vectorized<double> 对象，执行向零舍入
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> lgamma() const {
    // 返回一个使用 Sleef 库中 lgamma 函数计算的 Vectorized<double> 对象
    return Vectorized<double>(Sleef_lgammad4_u10(values));
  }
  Vectorized<double> sqrt() const {
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    // 使用 AVX 指令比较当前向量与另一个向量是否相等，不会抛出异常，返回布尔向量
    return _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ);
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    // 使用 AVX 指令比较当前向量与另一个向量是否不相等，不会抛出异常，返回布尔向量
    return _mm256_cmp_pd(values, other.values, _CMP_NEQ_UQ);
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    // 使用 AVX 指令比较当前向量是否小于另一个向量，不会抛出异常，返回布尔向量
    return _mm256_cmp_pd(values, other.values, _CMP_LT_OQ);
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    // 使用 AVX 指令比较当前向量是否小于等于另一个向量，不会抛出异常，返回布尔向量
    return _mm256_cmp_pd(values, other.values, _CMP_LE_OQ);
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    // 使用 AVX 指令比较当前向量是否大于另一个向量，不会抛出异常，返回布尔向量
    return _mm256_cmp_pd(values, other.values, _CMP_GT_OQ);
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    // 使用 AVX 指令比较当前向量是否大于等于另一个向量，不会抛出异常，返回布尔向量
    return _mm256_cmp_pd(values, other.values, _CMP_GE_OQ);
  }

  Vectorized<double> eq(const Vectorized<double>& other) const;
  Vectorized<double> ne(const Vectorized<double>& other) const;
  Vectorized<double> lt(const Vectorized<double>& other) const;
  Vectorized<double> le(const Vectorized<double>& other) const;
  Vectorized<double> gt(const Vectorized<double>& other) const;
  Vectorized<double> ge(const Vectorized<double>& other) const;
};

// 模板特化：双精度向量的加法运算符重载
template <>
Vectorized<double> inline operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_add_pd(a, b);
}

// 模板特化：双精度向量的减法运算符重载
template <>
Vectorized<double> inline operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_sub_pd(a, b);
}

// 模板特化：双精度向量的乘法运算符重载
template <>
Vectorized<double> inline operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_mul_pd(a, b);
}

// 模板特化：双精度向量的除法运算符重载
template <>
Vectorized<double> inline operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_div_pd(a, b);
}

// 实现 frac 函数，利用减法实现
// 返回当前向量减去截断后的向量
inline Vectorized<double> Vectorized<double>::frac() const {
  return *this - this->trunc();
}

// 实现 IEEE 754 201X 中的 maximum 操作，若任一输入为 NaN，则传播 NaN
template <>
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  Vectorized<double> max = _mm256_max_pd(a, b);
  Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // 利用所有位为 1 表示 NaN 的特性
  return _mm256_or_pd(max, isnan);
}

// 实现 IEEE 754 201X 中的 minimum 操作，若任一输入为 NaN，则传播 NaN
template <>
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  Vectorized<double> min = _mm256_min_pd(a, b);
  Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // 利用所有位为 1 表示 NaN 的特性
  return _mm256_or_pd(min, isnan);
}

// 模板特化：双精度向量的 clamp 函数，将向量限制在指定范围内
template <>
Vectorized<double> inline clamp(const Vectorized<double>& a, const Vectorized<double>& min, const Vectorized<double>& max) {
  return _mm256_min_pd(max, _mm256_max_pd(min, a));
}

// 模板特化：双精度向量的 clamp_min 函数，将向量限制在最小值以上
template <>
Vectorized<double> inline clamp_min(const Vectorized<double>& a, const Vectorized<double>& min) {
  return _mm256_max_pd(min, a);
}

// 模板特化：双精度向量的 clamp_max 函数，将向量限制在最大值以下
template <>
Vectorized<double> inline clamp_max(const Vectorized<double>& a, const Vectorized<double>& max) {
  return _mm256_min_pd(max, a);
}

// 模板特化：双精度向量的按位与运算符重载
template <>
Vectorized<double> inline operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_and_pd(a, b);
}

// 模板特化：双精度向量的按位或运算符重载
template <>
Vectorized<double> inline operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_or_pd(a, b);
}

// 模板特化：双精度向量的按位异或运算符重载
template <>
Vectorized<double> inline operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_xor_pd(a, b);
}

// 实现向量的相等比较，返回比较结果的向量
inline Vectorized<double> Vectorized<double>::eq(const Vectorized<double>& other) const {
  return (*this == other) & Vectorized<double>(1.0);
}

// 实现向量的不等比较，返回比较结果的向量
inline Vectorized<double> Vectorized<double>::ne(const Vectorized<double>& other) const {
  return (*this != other) & Vectorized<double>(1.0);
}

// 实现向量的大于比较，返回比较结果的向量
inline Vectorized<double> Vectorized<double>::gt(const Vectorized<double>& other) const {
  return (*this > other) & Vectorized<double>(1.0);
}
// 返回当前向量对象与另一个向量对象按元素进行大于或等于比较的结果，并与值为1.0的向量对象进行按位与操作
inline Vectorized<double> Vectorized<double>::ge(const Vectorized<double>& other) const {
  return (*this >= other) & Vectorized<double>(1.0);
}

// 返回当前向量对象与另一个向量对象按元素进行小于比较的结果，并与值为1.0的向量对象进行按位与操作
inline Vectorized<double> Vectorized<double>::lt(const Vectorized<double>& other) const {
  return (*this < other) & Vectorized<double>(1.0);
}

// 返回当前向量对象与另一个向量对象按元素进行小于或等于比较的结果，并与值为1.0的向量对象进行按位与操作
inline Vectorized<double> Vectorized<double>::le(const Vectorized<double>& other) const {
  return (*this <= other) & Vectorized<double>(1.0);
}

// 特化模板，用于将双精度浮点数数组从源指针转换到目标指针，处理长度为n的数组
template <>
inline void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
  // 非 Microsoft Visual C++ 编译器环境下，对下面的循环进行展开优化
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    // 使用 AVX2 指令集将长度为 Vectorized<double>::size() 的双精度浮点数数组从src加载到dst
    _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
  }
  // 处理剩余部分，长度小于 Vectorized<double>::size() 的双精度浮点数数组从src复制到dst
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

#ifdef CPU_CAPABILITY_AVX2
// AVX2 指令集下的特化模板，执行向量的 Fused Multiply-Add (FMA) 操作
template <>
Vectorized<double> inline fmadd(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return _mm256_fmadd_pd(a, b, c);
}

// AVX2 指令集下的特化模板，执行向量的 Fused Multiply-Subtract (FMS) 操作
template <>
Vectorized<double> inline fmsub(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return _mm256_fmsub_pd(a, b, c);
}
#endif

#endif

}} // namespace at::vec::CPU_CAPABILITY
```