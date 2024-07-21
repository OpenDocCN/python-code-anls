# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_float.h`

```py
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>
#if defined(CPU_CAPABILITY_AVX512)
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512)

// 定义 AVX-512 下的 float 类型向量化类模板
template <> class Vectorized<float> {
private:
  // 零向量，用于初始化
  static constexpr __m512i zero_vec {0, 0, 0, 0, 0, 0, 0, 0};
public:
  // SIMD 寄存器中的值
  __m512 values;
  using value_type = float;
  using size_type = int;
  // 向量大小为 16
  static constexpr size_type size() {
    return 16;
  }
  // 默认构造函数
  Vectorized() {}
  // 使用给定的 SIMD 寄存器初始化
  Vectorized(__m512 v) : values(v) {}
  // 使用单个值初始化所有 SIMD 寄存器
  Vectorized(float val) {
    values = _mm512_set1_ps(val);
  }
  // 使用多个值初始化 SIMD 寄存器
  Vectorized(float val1, float val2, float val3, float val4,
         float val5, float val6, float val7, float val8,
         float val9, float val10, float val11, float val12,
         float val13, float val14, float val15, float val16) {
    values = _mm512_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8,
                            val9, val10, val11, val12, val13, val14, val15, val16);
  }
  // 类型转换操作符，返回 SIMD 寄存器中的值
  operator __m512() const {
    return values;
  }
  // 按照掩码合并两个向量
  template <int64_t mask>
  static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_mask_blend_ps(mask, a.values, b.values);
  }
  // 根据掩码向量合并两个向量
  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask.values), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_blend_ps(mmask, a.values, b.values);
  }
  // 创建一个按步长递增的向量
  template<typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step,
      base + 8 * step, base + 9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  // 将一个向量中的前若干个元素设置为另一个向量中的值
  static Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
                           int64_t count = size()) {


This comment block covers the required annotations for the given C++ code snippet. Each line is explained in the context of its functionality within the class template `Vectorized<float>` for AVX-512 operations.
    // 根据 count 的不同情况进行不同的混合操作
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
    }
    // 如果 count 超出了范围，则返回 b
    return b;
  }
  // 从给定指针加载 count 个元素的向量化浮点数，如果 count 等于 size()，则加载所有元素
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      // 使用非对齐方式加载向量化浮点数
      return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));

    // 根据 count 创建掩码，只加载指定数量的元素
    __mmask16 mask = (1ULL << count) - 1;
    return _mm512_maskz_loadu_ps(mask, ptr);
  }
  // 将当前向量的值存储到指定指针中，如果 count 等于 size()，则存储所有元素
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      // 使用非对齐方式存储向量化浮点数
      _mm512_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      // 根据 count 创建掩码，只存储指定数量的元素
      __mmask16 mask = (1ULL << count) - 1;
      _mm512_mask_storeu_ps(reinterpret_cast<float*>(ptr), mask, values);
    }
  }
  // 禁用索引运算符 [] 的使用
  const float& operator[](int idx) const  = delete;
  float& operator[](int idx) = delete;
  // 返回一个整数掩码，其中所有零元素转换为 1 位，其他元素转换为 0 位
  int zero_mask() const {
    __mmask16 cmp = _mm512_cmp_ps_mask(values, _mm512_set1_ps(0.0), _CMP_EQ_OQ);
    return static_cast<int32_t>(cmp);
  }
  // 返回一个向量，表示每个元素是否为 NaN
  Vectorized<float> isnan() const {
    auto mask =  _mm512_cmp_ps_mask(values, _mm512_set1_ps(0.0), _CMP_UNORD_Q);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }
  // 检查向量中是否包含无穷大或 NaN
  bool has_inf_nan() const {
    __m512 self_sub  = _mm512_sub_ps(values, values);
    return (_mm512_movepi8_mask(_mm512_castps_si512(self_sub)) & 0x7777777777777777) != 0;
  }
  // 对向量中的每个元素应用函数 f，并返回结果向量
  Vectorized<float> map(float (*const f)(float)) const {
    // 将向量值存储到临时数组中
    __at_align__ float tmp[size()];
    store(tmp);
    // 对临时数组中的每个元素应用函数 f
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    // 加载经函数 f 处理后的临时数组为向量
    return loadu(tmp);
  }
  // 返回向量的绝对值
  Vectorized<float> abs() const {
    auto mask = _mm512_set1_ps(-0.f);
    return _mm512_andnot_ps(mask, values);
  }
  // 返回向量中每个元素的角度
  Vectorized<float> angle() const {
    __m512 zero_vec = _mm512_set1_ps(0.f);
    const auto nan_vec = _mm512_set1_ps(NAN);
    const auto not_nan_mask = _mm512_cmp_ps_mask(values, values, _CMP_EQ_OQ);
    const auto not_nan_vec = _mm512_mask_set1_epi32(_mm512_castps_si512(zero_vec),
                                                    not_nan_mask, 0xFFFFFFFF);
    const auto nan_mask = _mm512_cmp_ps_mask(_mm512_castsi512_ps(not_nan_vec),
                                             zero_vec, _CMP_EQ_OQ);
  Vectorized<float> expm1() const {
    // 计算常量
    const auto neg_zero_vec = _mm512_set1_ps(-0.f);
    const auto one_vec = _mm512_set1_ps(1.0f);
    const auto ln2_vec = _mm512_set1_ps(0.6931471805599453f);
    const auto p1 = _mm512_set1_ps(6.931530797412468e-1f);
    const auto p2 = _mm512_set1_ps(1.42860682030941723212e-6f);
    const auto p3 = _mm512_set1_ps(1.54010762792771901396e-13f);
    // x = values
    auto x = values;
    // mask for values < -log(2)
    auto mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(-0.6931471805599453f), _CMP_LT_OQ);
    // y = abs(x)
    auto y = _mm512_abs_ps(x);
    // t = 1 + y + y^2 * (p1 + y * (p2 + y * p3))
    auto t = _mm512_fmadd_ps(y, p1, one_vec);
    auto y2 = _mm512_mul_ps(y, y);
    auto tmp = _mm512_fmadd_ps(y, p2, p3);
    tmp = _mm512_fmadd_ps(y2, tmp, p1);
    t = _mm512_fmadd_ps(y2, t, tmp);
    // mask for x < -log(2)
    auto x_mask = _mm512_mask_blend_ps(mask, t, x);
    // if x < -log(2) then return -1, else return 1
    return _mm512_xor_ps(neg_zero_vec, x_mask);
  }
    // 返回一个 Vectorized 对象，其值为 Sleef_expm1f16_u10 函数处理 values 后的结果
    return Vectorized<float>(Sleef_expm1f16_u10(values));
  }
  Vectorized<float> exp_u20() const {
    // 一个更快的 exp 函数，精度为 ULP=20

    // 预定义常量，逆阶乘的值，用于泰勒展开的系数
    static __m512 vec_factorial_1 = _mm512_set1_ps(0.999999701f); // 1/factorial(1)
    static __m512 vec_factorial_2 = _mm512_set1_ps(0.499991506f); // 1/factorial(2)
    static __m512 vec_factorial_3 = _mm512_set1_ps(0.166676521f); // 1/factorial(3)
    static __m512 vec_factorial_4 = _mm512_set1_ps(0.0418978221f); // 1/factorial(4)
    static __m512 vec_factorial_5 = _mm512_set1_ps(0.00828929059f); // 1/factorial(5)

    // 预定义常量，log2(e)、0.5、1.0、0.0、2.0、ln(2) 等常用的 SIMD 寄存器
    static __m512 vec_exp_log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b)); // log2(e)
    static __m512 vec_half = _mm512_set1_ps(0.5f);
    static __m512 vec_one = _mm512_set1_ps(1.f);
    static __m512 vec_zero = _mm512_set1_ps(0.f);
    static __m512 vec_two = _mm512_set1_ps(2.f);
    static __m512 vec_ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218)); // ln(2)
    static __m512 vec_ln_flt_min = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
    static __m512 vec_ln_flt_max = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
    static __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
    static int n_mantissa_bits = 23;

    // exp(x) 的计算流程注释如下：
    // = exp(n * ln(2) + r) // 将 x 除以 ln(2)，得到商和余数
    // = 2^n * exp(r) // 简化 exp(n*ln(2)) 表达式

    // 根据 values 和 ln_flt_min 的比较生成掩码
    auto less_ln_flt_min_mask = _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);

    // 将 values 限制在 ln_flt_min 和 ln_flt_max 之间
    auto vec_src = _mm512_min_ps(values, vec_ln_flt_max);
    vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

    // fx = floorf(x * log2ef + 0.5)
    auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
    auto vec_fx_i = _mm512_cvt_roundps_epi32(
        vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

    // x = x - fx * ln2
    auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

    // 计算多项式
    auto vec_res =
        _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

    // 计算 2^(n-1)
    auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
    auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
    auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
    vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
    auto vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);
    vec_two_pow_n =
        _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

    // y = y * 2^n
    vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
    vec_res = _mm512_mul_ps(vec_res, vec_two);

    // 返回最终计算结果

    return vec_res;
  }
  }
  // 计算向量中每个元素的 fmod 操作结果
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    return Vectorized<float>(Sleef_fmodf16(values, q));
  }
  // 计算向量中每个元素的自然对数
  Vectorized<float> log() const {
    return Vectorized<float>(Sleef_logf16_u10(values));
  }
  // 计算向量中每个元素的以2为底的对数
  Vectorized<float> log2() const {
    return Vectorized<float>(Sleef_log2f16_u10(values));
  }
  // 计算向量中每个元素的以10为底的对数
  Vectorized<float> log10() const {
    return Vectorized<float>(Sleef_log10f16_u10(values));
  }
  // 计算向量中每个元素的 log(1+x) 操作结果
  Vectorized<float> log1p() const {
    return Vectorized<float>(Sleef_log1pf16_u10(values));
  }
  // 计算向量中每个元素的小数部分
  Vectorized<float> frac() const;
  // 计算向量中每个元素的正弦值
  Vectorized<float> sin() const {
    return Vectorized<float>(Sleef_sinf16_u35(values));
  }
  // 计算向量中每个元素的双曲正弦值
  Vectorized<float> sinh() const {
    return Vectorized<float>(Sleef_sinhf16_u10(values));
  }
  // 计算向量中每个元素的余弦值
  Vectorized<float> cos() const {
    return Vectorized<float>(Sleef_cosf16_u35(values));
  }
  // 计算向量中每个元素的双曲余弦值
  Vectorized<float> cosh() const {
    return Vectorized<float>(Sleef_coshf16_u10(values));
  }
  // 对向量中每个元素进行向上取整
  Vectorized<float> ceil() const {
    return _mm512_ceil_ps(values);
  }
  // 对向量中每个元素进行向下取整
  Vectorized<float> floor() const {
    return _mm512_floor_ps(values);
  }
  // 计算向量与另一个向量之间每个对应元素的 hypot 结果
  Vectorized<float> hypot(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_hypotf16_u05(values, b));
  }
  // 计算向量中每个元素的 i0 函数值
  Vectorized<float> i0() const {
    return map(calc_i0);
  }
  // 计算向量中每个元素的 i0e 函数值
  Vectorized<float> i0e() const {
    return map(calc_i0e);
  }
  // 计算向量中每个元素的 digamma 函数值
  Vectorized<float> digamma() const {
    return map(calc_digamma);
  }
  // 计算向量与另一个向量之间每个对应元素的 igamma 函数值
  Vectorized<float> igamma(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    // 循环计算每个元素的 igamma 函数值
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  // 计算向量与另一个向量之间每个对应元素的 igammac 函数值
  Vectorized<float> igammac(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    // 循环计算每个元素的 igammac 函数值
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  // 计算向量中每个元素的负数
  Vectorized<float> neg() const {
    return _mm512_xor_ps(_mm512_set1_ps(-0.f), values);
  }
  // 计算向量与另一个向量之间每个对应元素的 nextafter 函数值
  Vectorized<float> nextafter(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_nextafterf16(values, b));
  }
  // 对向量中每个元素进行四舍五入
  Vectorized<float> round() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  // 计算向量中每个元素的正切值
  Vectorized<float> tan() const {
    return Vectorized<float>(Sleef_tanf16_u10(values));
  }
  // 计算向量中每个元素的双曲正切值
  Vectorized<float> tanh() const {
    return Vectorized<float>(Sleef_tanhf16_u10(values));
  }
  // 对向量中每个元素进行向零取整
  Vectorized<float> trunc() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  // 计算向量中每个元素的 lgamma 函数值
  Vectorized<float> lgamma() const {
    return Vectorized<float>(Sleef_lgammaf16_u10(values));
  }
  // 计算向量中每个元素的平方根
  Vectorized<float> sqrt() const {
    return _mm512_sqrt_ps(values);
  }
  // 计算向量中每个元素的倒数
  Vectorized<float> reciprocal() const {
    return _mm512_div_ps(_mm512_set1_ps(1), values);
  }
  // 计算向量中每个元素的平方根的倒数
  Vectorized<float> rsqrt() const {
  // 返回向量中每个元素的倒数平方根
  return _mm512_div_ps(_mm512_set1_ps(1), _mm512_sqrt_ps(values));
}

// 返回向量中每个元素的 b 次方
Vectorized<float> pow(const Vectorized<float> &b) const {
  return Vectorized<float>(Sleef_powf16_u10(values, b));
}

// 使用 _CMP_EQ_OQ 谓词进行比较
//   `O`: 如果操作数为 NaN，则返回 false
//   `Q`: 如果操作数为 NaN，则不触发异常
Vectorized<float> operator==(const Vectorized<float>& other) const {
  auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_EQ_OQ);
  return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                    0xFFFFFFFF));
}

// 使用 _CMP_NEQ_UQ 谓词进行比较
//   `U`: 如果操作数为 NaN，则返回 true
//   `Q`: 如果操作数为 NaN，则不触发异常
Vectorized<float> operator!=(const Vectorized<float>& other) const {
  auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_NEQ_UQ);
  return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                    0xFFFFFFFF));
}

// 使用 _CMP_LT_OQ 谓词进行比较
//   `O`: 如果操作数为 NaN，则返回 false
//   `Q`: 如果操作数为 NaN，则不触发异常
Vectorized<float> operator<(const Vectorized<float>& other) const {
  auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_LT_OQ);
  return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                    0xFFFFFFFF));
}

// 使用 _CMP_LE_OQ 谓词进行比较
//   `O`: 如果操作数为 NaN，则返回 false
//   `Q`: 如果操作数为 NaN，则不触发异常
Vectorized<float> operator<=(const Vectorized<float>& other) const {
  auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_LE_OQ);
  return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                    0xFFFFFFFF));
}

// 使用 _CMP_GT_OQ 谓词进行比较
//   `O`: 如果操作数为 NaN，则返回 false
//   `Q`: 如果操作数为 NaN，则不触发异常
Vectorized<float> operator>(const Vectorized<float>& other) const {
  auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_GT_OQ);
  return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                    0xFFFFFFFF));
}

// 使用 _CMP_GE_OQ 谓词进行比较
//   `O`: 如果操作数为 NaN，则返回 false
//   `Q`: 如果操作数为 NaN，则不触发异常
Vectorized<float> operator>=(const Vectorized<float>& other) const {
  auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_GE_OQ);
  return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                    0xFFFFFFFF));
}
// 结束模板特化定义
};

// 定义模板特化，实现两个 Vectorized<float> 向量的加法
template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_add_ps(a, b);
}

// 定义模板特化，实现两个 Vectorized<float> 向量的减法
template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_sub_ps(a, b);
}

// 定义模板特化，实现两个 Vectorized<float> 向量的乘法
template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_mul_ps(a, b);
}

// 定义模板特化，实现两个 Vectorized<float> 向量的除法
template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_div_ps(a, b);
}

// 定义成员函数，计算当前向量的小数部分
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// 实现 IEEE 754 201X 标准的最大值操作，若有 NaN 则传播 NaN
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  auto zero_vec = _mm512_set1_epi32(0);
  auto max = _mm512_max_ps(a, b);
  auto isnan_mask = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
  auto isnan = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, isnan_mask,
                                                          0xFFFFFFFF));
  // 利用全为1的向量表示 NaN
  return _mm512_or_ps(max, isnan);
}

// 实现 IEEE 754 201X 标准的最小值操作，若有 NaN 则传播 NaN
template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  auto zero_vec = _mm512_set1_epi32(0);
  auto min = _mm512_min_ps(a, b);
  auto isnan_mask = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
  auto isnan = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, isnan_mask,
                                                          0xFFFFFFFF));
  // 利用全为1的向量表示 NaN
  return _mm512_or_ps(min, isnan);
}

// 实现 clamp 函数的模板特化，将向量 a 限制在最小值 min 和最大值 max 之间
template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return _mm512_min_ps(max, _mm512_max_ps(min, a));
}

// 实现 clamp_max 函数的模板特化，将向量 a 限制在最大值 max 之内
template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return _mm512_min_ps(max, a);
}

// 实现 clamp_min 函数的模板特化，将向量 a 限制在最小值 min 之上
template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return _mm512_max_ps(min, a);
}

// 实现按位与运算的模板特化，对两个向量进行按位与操作
template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_and_ps(a, b);
}

// 实现按位或运算的模板特化，对两个向量进行按位或操作
template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_or_ps(a, b);
}

// 实现按位异或运算的模板特化，对两个向量进行按位异或操作
template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_xor_ps(a, b);
}

// 实现等于运算的成员函数，返回当前向量与另一个向量的比较结果
inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}
// 定义返回向量化 float 类型的不等于操作结果的方法，使用当前对象和传入的向量进行逻辑与操作后返回一个新的向量
inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

// 定义返回向量化 float 类型的大于操作结果的方法，使用当前对象和传入的向量进行逻辑与操作后返回一个新的向量
inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

// 定义返回向量化 float 类型的大于等于操作结果的方法，使用当前对象和传入的向量进行逻辑与操作后返回一个新的向量
inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}

// 定义返回向量化 float 类型的小于操作结果的方法，使用当前对象和传入的向量进行逻辑与操作后返回一个新的向量
inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

// 定义返回向量化 float 类型的小于等于操作结果的方法，使用当前对象和传入的向量进行逻辑与操作后返回一个新的向量
inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

// 定义模板特化版本，将 float 数组 src 转换为 float 数组 dst，转换元素个数为 n
template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
  // 针对非 Microsoft Visual C++ 编译器，使用指令展开优化循环
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    // 使用 AVX-512 指令集将 src 中的向量加载并存储到 dst 中
    _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
  }
  // 处理剩余的不足一个向量长度的元素
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    // 直接复制单个 float 元素
    dst[i] = src[i];
  }
}

// 定义模板特化版本，实现向量化 float 类型的 fused multiply-add 操作
template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  // 使用 AVX-512 指令集进行 fused multiply-add 操作
  return _mm512_fmadd_ps(a, b, c);
}

// 定义模板特化版本，实现向量化 float 类型的 fused multiply-subtract 操作
template <>
Vectorized<float> inline fmsub(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  // 使用 AVX-512 指令集进行 fused multiply-subtract 操作
  return _mm512_fmsub_ps(a, b, c);
}

// TODO(jgong5): rewrite with ATEN vectorized (need to add unpack and shuffle)
// Used by Inductor CPP codegen
// Code referred to FBGEMM:
// https://github.com/pytorch/FBGEMM/blob/39a423e4ad1a04b77fea81c7d09c3e6f8984fae9/src/UtilsAvx512.cc#LL19C6-L19C6
// 16 * 6 = 96 instructions
// 定义模板特化版本，实现 float 类型的矩阵转置操作，源矩阵 src，行宽 ld_src，目标矩阵 dst
template<>
inline void transpose_mxn<float, 16, 16>(
    const float* src,
    int64_t ld_src,
    float* dst,
    // 函数未完成的注释，需要添加具体描述
}
```