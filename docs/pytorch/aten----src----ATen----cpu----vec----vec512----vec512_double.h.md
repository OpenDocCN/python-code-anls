# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_double.h`

```
#pragma once
// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#if (defined(CPU_CAPABILITY_AVX512))
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at {
namespace vec {

// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512)

template <> class Vectorized<double> {
private:
  // 静态成员变量，用于存储全零的 __m512i 向量
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
public:
  // 存储一个 512 位的 AVX 向量
  __m512d values;
  using value_type = double;
  using size_type = int;
  
  // 返回向量长度为 8
  static constexpr size_type size() {
    return 8;
  }
  
  // 默认构造函数
  Vectorized() {}
  
  // 使用给定的 __m512d 向量初始化的构造函数
  Vectorized(__m512d v) : values(v) {}
  
  // 使用单个 double 值初始化的构造函数
  Vectorized(double val) {
    values = _mm512_set1_pd(val);
  }
  
  // 使用多个 double 值初始化的构造函数
  Vectorized(double val1, double val2, double val3, double val4,
         double val5, double val6, double val7, double val8) {
    values = _mm512_setr_pd(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  
  // 将 Vectorized 转换为 __m512d 向量的隐式转换函数
  operator __m512d() const {
    return values;
  }
  
  // 根据 mask 合并两个 Vectorized<double> 向量
  template <int64_t mask>
  static Vectorized<double> blend(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_mask_blend_pd(mask, a.values, b.values);
  }
  
  // 根据 mask 合并两个 Vectorized<double> 向量（变长版本）
  static Vectorized<double> blendv(const Vectorized<double>& a, const Vectorized<double>& b,
                               const Vectorized<double>& mask) {
    auto all_ones = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    auto mmask = _mm512_cmp_epi64_mask(_mm512_castpd_si512(mask.values), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_blend_pd(mmask, a.values, b.values);
  }
  
  // 生成以 base 为起始，以 step 为步长的等差序列的 Vectorized<double> 向量
  template<typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(base, base + step, base + 2 * step, base + 3 * step,
                          base + 4 * step, base + 5 * step, base + 6 * step,
                          base + 7 * step);
  }
  
  // 使用 a 和 b 生成一个长度为 count 的 Vectorized<double> 向量
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
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }
    return b;
  }
  
  // 从内存地址 ptr 处加载一个长度为 count 的 Vectorized<double> 向量
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm512_loadu_pd(reinterpret_cast<const double*>(ptr));

    // 创建一个掩码，用于加载不定长的 Vectorized<double> 向量
    __mmask8 mask = (1ULL << count) - 1;
    return _mm512_maskz_loadu_pd(mask, ptr);
  }
  
  // 将当前 Vectorized<double> 向量存储到内存地址 ptr 处，长度为 count
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_pd(reinterpret_cast<double*>(ptr), values);
  Vectorized<double> acos() const {
    // 使用 Sleef 库计算反余弦函数，返回结果向量化 double 类型
    return Vectorized<double>(Sleef_acosd8_u10(values));
  }
  Vectorized<double> acosh() const {
    // 使用 Sleef 库计算反双曲余弦函数，返回结果向量化 double 类型
    return Vectorized<double>(Sleef_acoshd8_u10(values));
  }
  Vectorized<double> asin() const {
    // 使用 Sleef 库计算反正弦函数，返回结果向量化 double 类型
    return Vectorized<double>(Sleef_asind8_u10(values));
  }
  Vectorized<double> atan() const {
    // 使用 Sleef 库计算反正切函数，返回结果向量化 double 类型
    return Vectorized<double>(Sleef_atand8_u10(values));
  }
  Vectorized<double> atanh() const {
    // 使用 Sleef 库计算反双曲正切函数，返回结果向量化 double 类型
    return Vectorized<double>(Sleef_atanhd8_u10(values));
  }
  Vectorized<double> atan2(const Vectorized<double> &b) const {
    // 使用 Sleef 库计算两个向量之间的反正切函数，返回结果向量化 double 类型
    return Vectorized<double>(Sleef_atan2d8_u10(values, b));
  }
  return Vectorized<double>(Sleef_copysignd8(values, sign));


  // 返回一个新的 Vectorized 对象，其中每个元素是 values 对应位置使用 sign 的符号的结果
  return Vectorized<double>(Sleef_copysignd8(values, sign));



  Vectorized<double> erf() const {
    return Vectorized<double>(Sleef_erfd8_u10(values));
  }


  // 计算每个元素的误差函数 erf，返回新的 Vectorized 对象
  Vectorized<double> erf() const {
    return Vectorized<double>(Sleef_erfd8_u10(values));
  }



  Vectorized<double> erfc() const {
    return Vectorized<double>(Sleef_erfcd8_u15(values));
  }


  // 计算每个元素的余误差函数 erfc，返回新的 Vectorized 对象
  Vectorized<double> erfc() const {
    return Vectorized<double>(Sleef_erfcd8_u15(values));
  }



  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }


  // 对每个元素应用逆误差函数 erfinv，返回新的 Vectorized 对象
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }



  Vectorized<double> exp() const {
    return Vectorized<double>(Sleef_expd8_u10(values));
  }


  // 计算每个元素的指数函数 exp，返回新的 Vectorized 对象
  Vectorized<double> exp() const {
    return Vectorized<double>(Sleef_expd8_u10(values));
  }



  Vectorized<double> exp2() const {
    return Vectorized<double>(Sleef_exp2d8_u10(values));
  }


  // 计算每个元素的 2 的指数函数 exp2，返回新的 Vectorized 对象
  Vectorized<double> exp2() const {
    return Vectorized<double>(Sleef_exp2d8_u10(values));
  }



  Vectorized<double> expm1() const {
    return Vectorized<double>(Sleef_expm1d8_u10(values));
  }


  // 计算每个元素的 expm1 函数（exp(x) - 1），返回新的 Vectorized 对象
  Vectorized<double> expm1() const {
    return Vectorized<double>(Sleef_expm1d8_u10(values));
  }



  Vectorized<double> exp_u20() const {
    return exp();
  }


  // 返回 exp 函数的结果，用于兼容性，不执行额外的操作
  Vectorized<double> exp_u20() const {
    return exp();
  }



  Vectorized<double> fmod(const Vectorized<double>& q) const {
    return Vectorized<double>(Sleef_fmodd8(values, q));
  }


  // 对每个元素执行 fmod 操作，返回新的 Vectorized 对象
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    return Vectorized<double>(Sleef_fmodd8(values, q));
  }



  Vectorized<double> hypot(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_hypotd8_u05(values, b));
  }


  // 计算每个元素与向量 b 对应位置元素之间的直角三角形斜边长度，返回新的 Vectorized 对象
  Vectorized<double> hypot(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_hypotd8_u05(values, b));
  }



  Vectorized<double> i0() const {
    return map(calc_i0);
  }


  // 对每个元素应用修正的 0 阶贝塞尔函数 i0，返回新的 Vectorized 对象
  Vectorized<double> i0() const {
    return map(calc_i0);
  }



  Vectorized<double> i0e() const {
    return map(calc_i0e);
  }


  // 对每个元素应用修正的指数形式 0 阶贝塞尔函数 i0e，返回新的 Vectorized 对象
  Vectorized<double> i0e() const {
    return map(calc_i0e);
  }



  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }


  // 对每个元素应用 digamma 函数，返回新的 Vectorized 对象
  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }



  Vectorized<double> igamma(const Vectorized<double> &x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }


  // 对每个元素应用正则化不完全 Gamma 函数 igamma，返回新的 Vectorized 对象
  Vectorized<double> igamma(const Vectorized<double> &x) const {
    // 创建临时数组用于存储当前对象的值和参数 x 的值
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    // 将当前对象的值存储到 tmp 中
    store(tmp);
    // 将参数 x 的值存储到 tmp_x 中
    x.store(tmp_x);
    // 遍历并计算每个元素的正则化不完全 Gamma 函数
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    // 从临时数组加载结果并返回新的 Vectorized 对象
    return loadu(tmp);
  }



  Vectorized<double> igammac(const Vectorized<double> &x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }


  // 对每个元素应用补充正则化不完全 Gamma 函数 igammac，返回新的 Vectorized 对象
  Vectorized<double> igammac(const Vectorized<double> &x) const {
    // 创建临时数组用于存储当前对象的值和参数 x 的值
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    // 将当前对象的值存储到 tmp 中
    store(tmp);
    // 将参数 x 的值存储到 tmp_x 中
    x.store(tmp_x);
    // 遍历并计算每个元素的补充正则化不完全 Gamma 函数
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    // 从临时数组加载结果并返回新的 Vectorized 对象
    return loadu(tmp);
  }



  Vectorized<double> log() const {
    return Vectorized<double>(Sleef_logd8_u10(values));
  }


  // 计算每个元素的自然对数 log，返回新的 Vectorized 对象
  Vectorized<double> log() const {
    return Vectorized<double>(Sleef_logd8_u10(values));
  }



  Vectorized<double> log2() const {
    return Vectorized<double>(Sleef_log2d8_u10(values));
  }


  // 计算每个元素的以 2 为底的对数 log2，返回新的 Vectorized 对象
  Vectorized<double> log2() const {
    return Vectorized<double>(Sleef_log2d8_u10(values));
  }



  Vectorized<double> log10() const {
    return Vectorized<double>(Sleef_log10d8_u10(values));
  }


  // 计算每个元素的以 10 为底的对数 log10，返回新的 Vectorized 对象
  Vectorized<double> log10() const {
    return Vectorized<double
  }
  // 返回经过 MM512 舍入和缩放后的结果，使用最接近整数舍入模式，无异常
  Vectorized<double> tan() const {
    // 返回通过 Sleef 库计算的 tan 值
    return Vectorized<double>(Sleef_tand8_u10(values));
  }
  // 返回经过 MM512 舍入和缩放后的结果，使用朝零舍入模式，无异常
  Vectorized<double> tanh() const {
    // 返回通过 Sleef 库计算的 tanh 值
    return Vectorized<double>(Sleef_tanhd8_u10(values));
  }
  // 返回经过 MM512 舍入和缩放后的结果，使用朝零舍入模式，无异常
  Vectorized<double> trunc() const {
    // 返回 values 的朝零舍入结果
    return _mm512_roundscale_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  // 返回通过 Sleef 库计算的 lgamma 值
  Vectorized<double> lgamma() const {
    // 返回通过 Sleef 库计算的 lgamma 值
    return Vectorized<double>(Sleef_lgammad8_u10(values));
  }
  // 返回 values 向量中各元素的平方根
  Vectorized<double> sqrt() const {
    // 返回 values 向量中各元素的平方根
    return _mm512_sqrt_pd(values);
  }
  // 返回 values 向量中各元素的倒数
  Vectorized<double> reciprocal() const {
    // 返回 values 向量中各元素的倒数
    return _mm512_div_pd(_mm512_set1_pd(1), values);
  }
  // 返回 values 向量中各元素的平方根的倒数
  Vectorized<double> rsqrt() const {
    // 返回 values 向量中各元素的平方根的倒数
    return _mm512_div_pd(_mm512_set1_pd(1), _mm512_sqrt_pd(values));
  }
  // 返回 values 向量和 b 向量对应元素的幂次方结果
  Vectorized<double> pow(const Vectorized<double> &b) const {
    // 返回 values 向量和 b 向量对应元素的幂次方结果
    return Vectorized<double>(Sleef_powd8_u10(values, b));
  }
  // 使用 _CMP_EQ_OQ 谓词进行比较
  //   `O`: 如果操作数是 NaN，则得到 false
  //   `Q`: 如果操作数是 NaN，则不引发异常
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    // 比较 values 和 other.values 的相等性，生成掩码
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_EQ_OQ);
    // 根据掩码设置返回结果向量
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  // 使用 _CMP_NEQ_UQ 谓词进行比较
  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    // 比较 values 和 other.values 的不等性，生成掩码
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_NEQ_UQ);
    // 根据掩码设置返回结果向量
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  // 使用 _CMP_LT_OQ 谓词进行比较
  Vectorized<double> operator<(const Vectorized<double>& other) const {
    // 比较 values 和 other.values 的小于关系，生成掩码
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_LT_OQ);
    // 根据掩码设置返回结果向量
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  // 使用 _CMP_LE_OQ 谓词进行比较
  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    // 比较 values 和 other.values 的小于等于关系，生成掩码
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_LE_OQ);
    // 根据掩码设置返回结果向量
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  // 使用 _CMP_GT_OQ 谓词进行比较
  Vectorized<double> operator>(const Vectorized<double>& other) const {
    // 比较 values 和 other.values 的大于关系，生成掩码
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_GT_OQ);
    // 根据掩码设置返回结果向量
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  // 使用 _CMP_GE_OQ 谓词进行比较
  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    // 比较 values 和 other.values 的大于等于关系，生成掩码
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_GE_OQ);
    // 根据掩码设置返回结果向量
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }
    // 将所有元素置为 0xFFFFFFFFFFFFFFFF 或 0，取决于比较掩码 cmp_mask
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  // 返回与另一个 Vectorized<double> 对象进行相等比较的结果向量
  Vectorized<double> eq(const Vectorized<double>& other) const;

  // 返回与另一个 Vectorized<double> 对象进行不等比较的结果向量
  Vectorized<double> ne(const Vectorized<double>& other) const;

  // 返回与另一个 Vectorized<double> 对象进行小于比较的结果向量
  Vectorized<double> lt(const Vectorized<double>& other) const;

  // 返回与另一个 Vectorized<double> 对象进行小于等于比较的结果向量
  Vectorized<double> le(const Vectorized<double>& other) const;

  // 返回与另一个 Vectorized<double> 对象进行大于比较的结果向量
  Vectorized<double> gt(const Vectorized<double>& other) const;

  // 返回与另一个 Vectorized<double> 对象进行大于等于比较的结果向量
  Vectorized<double> ge(const Vectorized<double>& other) const;
};

// 特化模板，重载向量双精度加法运算符
template <>
Vectorized<double> inline operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_add_pd(a, b);
}

// 特化模板，重载向量双精度减法运算符
template <>
Vectorized<double> inline operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_sub_pd(a, b);
}

// 特化模板，重载向量双精度乘法运算符
template <>
Vectorized<double> inline operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_mul_pd(a, b);
}

// 特化模板，重载向量双精度除法运算符
template <>
Vectorized<double> inline operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_div_pd(a, b);
}

// frac. 在这里实现，以便使用减法。
// 返回向量的小数部分，即该向量减去其截断部分
inline Vectorized<double> Vectorized<double>::frac() const {
  return *this - this->trunc();
}

// 实现 IEEE 754 201X 的 `maximum` 操作，如果任一输入为 NaN，则传播 NaN。
template <>
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  auto zero_vec = _mm512_set1_epi64(0);
  Vectorized<double> max = _mm512_max_pd(a, b);
  auto isnan_mask = _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q);
  auto isnan = _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vec, isnan_mask,
                                                          0xFFFFFFFFFFFFFFFF));
  // 利用全1表示 NaN 的特性
  return _mm512_or_pd(max, isnan);
}

// 实现 IEEE 754 201X 的 `minimum` 操作，如果任一输入为 NaN，则传播 NaN。
template <>
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  auto zero_vec = _mm512_set1_epi64(0);
  Vectorized<double> min = _mm512_min_pd(a, b);
  auto isnan_mask = _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q);
  auto isnan = _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vec, isnan_mask,
                                                          0xFFFFFFFFFFFFFFFF));
  // 利用全1表示 NaN 的特性
  return _mm512_or_pd(min, isnan);
}

// 特化模板，实现 clamp 函数，限制向量的取值范围在[min, max]之间
template <>
Vectorized<double> inline clamp(const Vectorized<double>& a, const Vectorized<double>& min, const Vectorized<double>& max) {
  return _mm512_min_pd(max, _mm512_max_pd(min, a));
}

// 特化模板，实现 clamp_min 函数，将向量的值限制在最小值 min 以上
template <>
Vectorized<double> inline clamp_min(const Vectorized<double>& a, const Vectorized<double>& min) {
  return _mm512_max_pd(min, a);
}

// 特化模板，实现 clamp_max 函数，将向量的值限制在最大值 max 以下
template <>
Vectorized<double> inline clamp_max(const Vectorized<double>& a, const Vectorized<double>& max) {
  return _mm512_min_pd(max, a);
}

// 特化模板，重载向量双精度按位与运算符
template <>
Vectorized<double> inline operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_and_pd(a, b);
}

// 特化模板，重载向量双精度按位或运算符
template <>
Vectorized<double> inline operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_or_pd(a, b);
}

// 特化模板，重载向量双精度按位异或运算符
template <>
Vectorized<double> inline operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_xor_pd(a, b);
}

// 实现向量双精度的相等比较函数，返回每个元素与 other 相等的向量
inline Vectorized<double> Vectorized<double>::eq(const Vectorized<double>& other) const {
  return (*this == other) & Vectorized<double>(1.0);
}
inline Vectorized<double> Vectorized<double>::ne(const Vectorized<double>& other) const {
    // 返回当前向量与另一个向量按元素比较不相等的结果向量，并与所有元素为1.0的向量进行按位与操作
    return (*this != other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::gt(const Vectorized<double>& other) const {
    // 返回当前向量与另一个向量按元素比较大于的结果向量，并与所有元素为1.0的向量进行按位与操作
    return (*this > other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ge(const Vectorized<double>& other) const {
    // 返回当前向量与另一个向量按元素比较大于等于的结果向量，并与所有元素为1.0的向量进行按位与操作
    return (*this >= other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::lt(const Vectorized<double>& other) const {
    // 返回当前向量与另一个向量按元素比较小于的结果向量，并与所有元素为1.0的向量进行按位与操作
    return (*this < other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::le(const Vectorized<double>& other) const {
    // 返回当前向量与另一个向量按元素比较小于等于的结果向量，并与所有元素为1.0的向量进行按位与操作
    return (*this <= other) & Vectorized<double>(1.0);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
    int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
    // 使用 SIMD 指令集将 src 中的双精度数据转换并存储到 dst 中，处理长度为 Vectorized<double>::size() 的块
    for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
        _mm512_storeu_pd(dst + i, _mm512_loadu_pd(src + i));
    }
#ifndef __msvc_cl__
#pragma unroll
#endif
    // 处理剩余的不足一个 SIMD 块长度的数据
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

template <>
// 使用 SIMD 指令集执行向量的 Fused Multiply-Add 操作，返回结果向量
Vectorized<double> inline fmadd(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
    return _mm512_fmadd_pd(a, b, c);
}

template <>
// 使用 SIMD 指令集执行向量的 Fused Multiply-Subtract 操作，返回结果向量
Vectorized<double> inline fmsub(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
    return _mm512_fmsub_pd(a, b, c);
}

#endif
```