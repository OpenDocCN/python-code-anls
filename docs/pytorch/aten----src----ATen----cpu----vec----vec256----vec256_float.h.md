# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_float.h`

```py
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

// 定义模板类 Vectorized<float>，用于实现 AVX2 指令集下的向量化操作
template <> class Vectorized<float> {
private:
  __m256 values;  // 内部存储的 AVX 寄存器

public:
  using value_type = float;  // 向量元素类型为 float
  using size_type = int;     // 向量长度类型为 int

  // 返回向量长度为 8
  static constexpr size_type size() {
    return 8;
  }

  // 默认构造函数，未初始化 AVX 寄存器
  Vectorized() {}

  // 使用给定的 AVX 寄存器初始化向量
  Vectorized(__m256 v) : values(v) {}

  // 使用单个 float 值初始化向量的构造函数
  Vectorized(float val) {
    // 使用 _mm256_set1_ps 设置所有元素为同一个值
    values = _mm256_set1_ps(val);
  }

  // 使用八个 float 值初始化向量的构造函数
  Vectorized(float val1, float val2, float val3, float val4,
             float val5, float val6, float val7, float val8) {
    // 使用 _mm256_setr_ps 设置 AVX 寄存器的每个元素
    values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
  }

  // 类型转换操作符，将向量转换为 __m256 类型
  operator __m256() const {
    return values;
  }

  // 按位混合操作，根据 mask 对两个向量 a 和 b 进行混合
  template <int64_t mask>
  static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_blend_ps(a.values, b.values, mask);
  }

  // 按位条件混合操作，根据 mask 对两个向量 a 和 b 进行条件混合
  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask) {
    return _mm256_blendv_ps(a.values, b.values, mask.values);
  }

  // 生成等差序列的向量，从 base 开始，步长为 step
  template<typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
  }

  // 设置向量的前 count 个元素为 a，后面的元素为 b
  static Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
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

  // 从内存地址 ptr 处加载 count 个元素到向量中
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
    
    // 若加载数量不为向量长度，使用备用内存 tmp_values
    __at_align__ float tmp_values[size()];
    // 确保未初始化的内存不会改变输出值，详见链接 https://github.com/pytorch/pytorch/issues/32502
    // 不使用 "={0}" 初始化数组为零，因为 gcc 可能会编译为两条指令，而循环只编译为一条指令。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
  return _mm256_loadu_ps(tmp_values);
}
// 将临时存储的值加载到 AVX 寄存器中并返回

void store(void* ptr, int64_t count = size()) const {
  if (count == size()) {
    _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
  } else if (count > 0) {
    float tmp_values[size()];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
    std::memcpy(ptr, tmp_values, count * sizeof(float));
  }
}
// 将向量存储到内存中，支持部分存储或完整存储，根据 count 参数决定

const float& operator[](int idx) const  = delete;
float& operator[](int idx) = delete;
// 禁用了 [] 操作符的 const 和非 const 版本

int zero_mask() const {
  // 返回一个整数掩码，其中所有零元素被转换为 1 位，其他元素被转换为 0 位
  __m256 cmp = _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_EQ_OQ);
  return _mm256_movemask_ps(cmp);
}

Vectorized<float> isnan() const {
  return _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_UNORD_Q);
}

bool has_inf_nan() const {
  __m256 self_sub  = _mm256_sub_ps(values, values);
  return (_mm256_movemask_epi8(_mm256_castps_si256(self_sub)) & 0x77777777) != 0;
}

Vectorized<float> map(float (*const f)(float)) const {
  __at_align__ float tmp[size()];
  store(tmp);
  for (const auto i : c10::irange(size())) {
    tmp[i] = f(tmp[i]);
  }
  return loadu(tmp);
}
// 对每个向量元素执行函数 f，并返回结果

Vectorized<float> abs() const {
  auto mask = _mm256_set1_ps(-0.f);
  return _mm256_andnot_ps(mask, values);
}
// 返回向量的绝对值

Vectorized<float> angle() const {
  const auto zero_vec = _mm256_set1_ps(0.f);
  const auto nan_vec = _mm256_set1_ps(NAN);
  const auto not_nan_mask = _mm256_cmp_ps(values, values, _CMP_EQ_OQ);
  const auto nan_mask = _mm256_cmp_ps(not_nan_mask, zero_vec, _CMP_EQ_OQ);
  const auto pi = _mm256_set1_ps(c10::pi<float>);

  const auto neg_mask = _mm256_cmp_ps(values, zero_vec, _CMP_LT_OQ);
  auto angle = _mm256_blendv_ps(zero_vec, pi, neg_mask);
  angle = _mm256_blendv_ps(angle, nan_vec, nan_mask);
  return angle;
}
// 返回向量元素的角度值，处理 NaN 和负数的情况

Vectorized<float> real() const {
  return *this;
}
// 返回实部，对应于自身向量

Vectorized<float> imag() const {
  return _mm256_set1_ps(0);
}
// 返回虚部，对应于零向量

Vectorized<float> conj() const {
  return *this;
}
// 返回共轭向量，对应于自身向量

Vectorized<float> acos() const {
  return Vectorized<float>(Sleef_acosf8_u10(values));
}
// 返回向量元素的反余弦值

Vectorized<float> acosh() const {
  return Vectorized<float>(Sleef_acoshf8_u10(values));
}
// 返回向量元素的反双曲余弦值

Vectorized<float> asin() const {
  return Vectorized<float>(Sleef_asinf8_u10(values));
}
// 返回向量元素的反正弦值

Vectorized<float> atan() const {
  return Vectorized<float>(Sleef_atanf8_u10(values));
}
// 返回向量元素的反正切值

Vectorized<float> atanh() const {
  return Vectorized<float>(Sleef_atanhf8_u10(values));
}
// 返回向量元素的反双曲正切值

Vectorized<float> atan2(const Vectorized<float> &b) const {
  return Vectorized<float>(Sleef_atan2f8_u10(values, b));
}
// 返回向量元素的反正切二元函数值

Vectorized<float> copysign(const Vectorized<float> &sign) const {
  return Vectorized<float>(Sleef_copysignf8(values, sign));
}
// 返回以 sign 向量元素的符号为准的向量副本

Vectorized<float> erf() const {
  // 常数
  const auto neg_zero_vec = _mm256_set1_ps(-0.f);
  const auto one_vec = _mm256_set1_ps(1.0f);
    const auto p = _mm256_set1_ps(0.3275911f);
    // 设置常数 p = 0.3275911 的 AVX 向量
    const auto p1 = _mm256_set1_ps(0.254829592f);
    // 设置常数 p1 = 0.254829592 的 AVX 向量
    const auto p2 = _mm256_set1_ps(-0.284496736f);
    // 设置常数 p2 = -0.284496736 的 AVX 向量
    const auto p3 = _mm256_set1_ps(1.421413741f);
    // 设置常数 p3 = 1.421413741 的 AVX 向量
    const auto p4 = _mm256_set1_ps(-1.453152027f);
    // 设置常数 p4 = -1.453152027 的 AVX 向量
    const auto p5 = _mm256_set1_ps(1.061405429f);
    // 设置常数 p5 = 1.061405429 的 AVX 向量
    // sign(x)
    auto sign_mask = _mm256_and_ps(neg_zero_vec, values);
    // 计算 sign(x)，即提取 values 中每个元素的符号位
    auto abs_vec = _mm256_xor_ps(sign_mask, values);
    // 计算 abs(x)，即取 values 中每个元素的绝对值
    // t = 1 / (p * abs(x) + 1)
    auto tmp0 = _mm256_fmadd_ps(p, abs_vec, one_vec);
    // 计算 p * abs(x) + 1
    auto t = _mm256_div_ps(one_vec, tmp0);
    // 计算 t = 1 / (p * abs(x) + 1)
    // r = p5 * t ^ 4 + p4 * t ^ 3 + p3 * t ^ 2 + p2 * t + p1
    auto tmp1 = _mm256_fmadd_ps(p5, t, p4);
    // 计算 p5 * t + p4
    auto tmp2 = _mm256_fmadd_ps(tmp1, t, p3);
    // 计算 (p5 * t + p4) * t + p3
    auto tmp3 = _mm256_fmadd_ps(tmp2, t, p2);
    // 计算 ((p5 * t + p4) * t + p3) * t + p2
    auto r = _mm256_fmadd_ps(tmp3, t, p1);
    // 计算 (((p5 * t + p4) * t + p3) * t + p2) * t + p1
    // - exp(- x * x)
    auto pow_2 = _mm256_mul_ps(values, values);
    // 计算 x * x
    auto neg_pow_2 = _mm256_xor_ps(neg_zero_vec, pow_2);
    // 计算 - (x * x)
    // auto tmp4 = exp(neg_pow_2);
    auto tmp4 = Vectorized<float>(Sleef_expf8_u10(neg_pow_2));
    // 计算 exp(- (x * x))，使用 Sleef 库进行向量化计算
    auto tmp5 = _mm256_xor_ps(neg_zero_vec, tmp4);
    // 计算 - exp(- (x * x))
    // erf(x) = sign(x) * (1 - r * t * exp(- x * x))
    auto tmp6 = _mm256_mul_ps(tmp5, t);
    // 计算 exp(- (x * x)) * t
    auto tmp7 = _mm256_fmadd_ps(tmp6, r, one_vec);
    // 计算 1 + exp(- (x * x)) * t * r
    return _mm256_xor_ps(sign_mask, tmp7);
    // 返回 sign(x) * (1 + exp(- (x * x)) * t * r)，即 erf(x) 的近似值
}
Vectorized<float> erfc() const {
    return Vectorized<float>(Sleef_erfcf8_u15(values));
    // 使用 Sleef 库计算 values 中每个元素的 erfc(x)，并返回结果
}
Vectorized<float> erfinv() const {
    return map(calc_erfinv);
    // 使用 calc_erfinv 函数映射计算 erfinv(values) 并返回结果
}
Vectorized<float> exp() const {
    return Vectorized<float>(Sleef_expf8_u10(values));
    // 使用 Sleef 库计算 values 中每个元素的 exp(x)，并返回结果
}
Vectorized<float> exp2() const {
    return Vectorized<float>(Sleef_exp2f8_u10(values));
    // 使用 Sleef 库计算 values 中每个元素的 exp2(x)，并返回结果
}
Vectorized<float> expm1() const {
    return Vectorized<float>(Sleef_expm1f8_u10(values));
    // 使用 Sleef 库计算 values 中每个元素的 expm1(x)，并返回结果
}
Vectorized<float> exp_u20() const {
    // A faster version of exp with ULP=20
    static __m256 vec_factorial_1 =
        _mm256_set1_ps(0.999999701f); // 1/factorial(1)
    // 设置常数 1/factorial(1) 的 AVX 向量
    static __m256 vec_factorial_2 =
        _mm256_set1_ps(0.499991506f); // 1/factorial(2)
    // 设置常数 1/factorial(2) 的 AVX 向量
    static __m256 vec_factorial_3 =
        _mm256_set1_ps(0.166676521f); // 1/factorial(3)
    // 设置常数 1/factorial(3) 的 AVX 向量
    static __m256 vec_factorial_4 =
        _mm256_set1_ps(0.0418978221f); // 1/factorial(4)
    // 设置常数 1/factorial(4) 的 AVX 向量
    static __m256 vec_factorial_5 =
        _mm256_set1_ps(0.00828929059f); // 1/factorial(5)
    // 设置常数 1/factorial(5) 的 AVX 向量
    static __m256 vec_exp_log2ef =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b)); // log2(e)
    // 设置常数 log2(e) 的 AVX 向量
    static __m256 vec_half = _mm256_set1_ps(0.5f);
    // 设置常数 0.5 的 AVX 向量
    static __m256 vec_one = _mm256_set1_ps(1.f);
    // 设置常数 1.0 的 AVX 向量
    static __m256 vec_zero = _mm256_set1_ps(0.f);
    // 设置常数 0.0 的 AVX 向量
    static __m256 vec_two = _mm256_set1_ps(2.f);
    // 设置常数 2.0 的 AVX 向量
    static __m256 vec_ln2f = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218)); // ln(2)
    // 设置常数 ln(2) 的 AVX 向量
    static __m256 vec_ln_flt_min = _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50));
    // 设置常数 float 最小值的 ln 的 AVX 向量
    static __m256 vec_ln_flt_max = _mm256_castsi256_ps(_mm256_set1_epi32(0x42b17218));
    // 设置常数 float 最大值的 ln 的 AVX 向量
    static __m256i vec_127 = _mm256_set1_epi32(0x0000007f);
    // 设置常数 127 的 AVX 向量，用于处理指数部分

    // exp(x) =
    // 比较每个元素是否小于vec_ln_flt_min，生成掩码
    auto less_ln_flt_min_mask =
        _mm256_cmp_ps(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
    // 将values向量中的每个元素限制在vec_ln_flt_max和vec_ln_flt_min之间
    auto vec_src = _mm256_min_ps(values, vec_ln_flt_max);
    vec_src = _mm256_max_ps(vec_src, vec_ln_flt_min);

    // 计算fx = floorf(x * log2ef + 0.5)
    auto vec_fx = _mm256_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
    vec_fx = _mm256_floor_ps(vec_fx);

    // x = x - fx * ln2
    auto vec_exp_poly = _mm256_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

    // 计算多项式
    auto vec_res =
        _mm256_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
    vec_res = _mm256_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
    vec_res = _mm256_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
    vec_res = _mm256_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
    vec_res = _mm256_fmadd_ps(vec_exp_poly, vec_res, vec_one);

    // 计算2^(n-1)
    auto vec_exp_number = _mm256_sub_ps(vec_fx, vec_one);
    auto vec_exp_number_i = _mm256_cvtps_epi32(vec_exp_number);
    auto vec_two_pow_n_i = _mm256_add_epi32(vec_exp_number_i, vec_127);
    vec_two_pow_n_i = _mm256_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
    auto vec_two_pow_n = _mm256_castsi256_ps(vec_two_pow_n_i);
    vec_two_pow_n =
        _mm256_blendv_ps(vec_two_pow_n, vec_zero, less_ln_flt_min_mask);

    // y = y * 2^n
    vec_res = _mm256_mul_ps(vec_res, vec_two_pow_n);
    vec_res = _mm256_mul_ps(vec_res, vec_two);
    return vec_res;
}
  // 返回一个新的 Vectorized<float>，其中每个元素应用 calc_digamma 函数
  return map(calc_digamma);
}

// 计算每个元素的逆伽马函数，返回结果作为新的 Vectorized<float>
Vectorized<float> igamma(const Vectorized<float> &x) const {
  // 创建临时数组 tmp 和 tmp_x 来存储当前对象和输入向量的值
  __at_align__ float tmp[size()];
  __at_align__ float tmp_x[size()];
  // 将当前对象的值存储到 tmp 中
  store(tmp);
  // 将输入向量的值存储到 tmp_x 中
  x.store(tmp_x);
  // 对于每个元素，计算其逆伽马函数并存储到 tmp 数组中
  for (const auto i : c10::irange(size())) {
    tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
  }
  // 将计算结果加载到新的 Vectorized<float> 中并返回
  return loadu(tmp);
}

// 计算每个元素的逆伽马函数的补函数，返回结果作为新的 Vectorized<float>
Vectorized<float> igammac(const Vectorized<float> &x) const {
  // 创建临时数组 tmp 和 tmp_x 来存储当前对象和输入向量的值
  __at_align__ float tmp[size()];
  __at_align__ float tmp_x[size()];
  // 将当前对象的值存储到 tmp 中
  store(tmp);
  // 将输入向量的值存储到 tmp_x 中
  x.store(tmp_x);
  // 对于每个元素，计算其逆伽马函数的补函数并存储到 tmp 数组中
  for (const auto i : c10::irange(size())) {
    tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
  }
  // 将计算结果加载到新的 Vectorized<float> 中并返回
  return loadu(tmp);
}

// 返回当前对象值的负值
Vectorized<float> neg() const {
  return _mm256_xor_ps(_mm256_set1_ps(-0.f), values);
}

// 返回当前对象值与参数向量之间的下一个可表示浮点数
Vectorized<float> nextafter(const Vectorized<float> &b) const {
  return Vectorized<float>(Sleef_nextafterf8(values, b));
}

// 返回当前对象值的四舍五入结果
Vectorized<float> round() const {
  return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// 返回当前对象值的正切值
Vectorized<float> tan() const {
  return Vectorized<float>(Sleef_tanf8_u10(values));
}

// 返回当前对象值的双曲正切值
Vectorized<float> tanh() const {
  return Vectorized<float>(Sleef_tanhf8_u10(values));
}

// 返回当前对象值的截断整数部分
Vectorized<float> trunc() const {
  return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}

// 返回当前对象值的自然对数的伽玛函数值
Vectorized<float> lgamma() const {
  return Vectorized<float>(Sleef_lgammaf8_u10(values));
}

// 返回当前对象值的平方根
Vectorized<float> sqrt() const {
  return _mm256_sqrt_ps(values);
}

// 返回当前对象值的倒数
Vectorized<float> reciprocal() const {
  return _mm256_div_ps(_mm256_set1_ps(1), values);
}

// 返回当前对象值的平方根的倒数
Vectorized<float> rsqrt() const {
  return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
}

// 返回当前对象值与参数向量值的幂运算结果
Vectorized<float> pow(const Vectorized<float> &b) const {
  return Vectorized<float>(Sleef_powf8_u10(values, b));
}

// 比较当前对象值与参数向量值是否相等，使用 _CMP_EQ_OQ 谓词
Vectorized<float> operator==(const Vectorized<float>& other) const {
  return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);
}

// 比较当前对象值与参数向量值是否不相等，使用 _CMP_NEQ_UQ 谓词
Vectorized<float> operator!=(const Vectorized<float>& other) const {
  return _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ);
}

// 比较当前对象值是否小于参数向量值，使用 _CMP_LT_OQ 谓词
Vectorized<float> operator<(const Vectorized<float>& other) const {
  return _mm256_cmp_ps(values, other.values, _CMP_LT_OQ);
}

// 比较当前对象值是否小于等于参数向量值，使用 _CMP_LE_OQ 谓词
Vectorized<float> operator<=(const Vectorized<float>& other) const {
  return _mm256_cmp_ps(values, other.values, _CMP_LE_OQ);
}

// 比较当前对象值是否大于参数向量值，使用 _CMP_GT_OQ 谓词
Vectorized<float> operator>(const Vectorized<float>& other) const {
  return _mm256_cmp_ps(values, other.values, _CMP_GT_OQ);
}

// 比较当前对象值是否大于等于参数向量值，使用 _CMP_GE_OQ 谓词
Vectorized<float> operator>=(const Vectorized<float>& other) const {
    # 返回一个使用 AVX 指令集进行比较操作的结果，比较当前 Vectorized 对象的每个元素是否大于或等于另一个 Vectorized 对象的对应元素
    return _mm256_cmp_ps(values, other.values, _CMP_GE_OQ);
  }

  # 声明一个函数，用于比较当前 Vectorized 对象和另一个 Vectorized<float> 对象是否相等
  Vectorized<float> eq(const Vectorized<float>& other) const;
  
  # 声明一个函数，用于比较当前 Vectorized 对象和另一个 Vectorized<float> 对象是否不相等
  Vectorized<float> ne(const Vectorized<float>& other) const;
  
  # 声明一个函数，用于比较当前 Vectorized 对象和另一个 Vectorized<float> 对象是否大于
  Vectorized<float> gt(const Vectorized<float>& other) const;
  
  # 声明一个函数，用于比较当前 Vectorized 对象和另一个 Vectorized<float> 对象是否大于或等于
  Vectorized<float> ge(const Vectorized<float>& other) const;
  
  # 声明一个函数，用于比较当前 Vectorized 对象和另一个 Vectorized<float> 对象是否小于
  Vectorized<float> lt(const Vectorized<float>& other) const;
  
  # 声明一个函数，用于比较当前 Vectorized 对象和另一个 Vectorized<float> 对象是否小于或等于
  Vectorized<float> le(const Vectorized<float>& other) const;
};

// 特化模板函数，实现两个 Vectorized<float> 向量的加法
template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_add_ps(a, b);
}

// 特化模板函数，实现两个 Vectorized<float> 向量的减法
template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_sub_ps(a, b);
}

// 特化模板函数，实现两个 Vectorized<float> 向量的乘法
template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_mul_ps(a, b);
}

// 特化模板函数，实现两个 Vectorized<float> 向量的除法
template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_div_ps(a, b);
}

// 实现向量的 frac 函数，返回当前向量减去其截断后的值
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// 实现 IEEE 754 201X 的 maximum 操作，如果有 NaN，则返回 NaN
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  Vectorized<float> max = _mm256_max_ps(a, b);
  Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // 利用所有位为 1 的向量表示 NaN
  return _mm256_or_ps(max, isnan);
}

// 实现 IEEE 754 201X 的 minimum 操作，如果有 NaN，则返回 NaN
template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  Vectorized<float> min = _mm256_min_ps(a, b);
  Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // 利用所有位为 1 的向量表示 NaN
  return _mm256_or_ps(min, isnan);
}

// 实现 clamp 函数，将向量 a 中的值限制在 min 和 max 之间
template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return _mm256_min_ps(max, _mm256_max_ps(min, a));
}

// 实现 clamp_max 函数，将向量 a 中的值限制在不超过 max 的范围内
template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return _mm256_min_ps(max, a);
}

// 实现 clamp_min 函数，将向量 a 中的值限制在不低于 min 的范围内
template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return _mm256_max_ps(min, a);
}

// 实现按位与运算符 & 的重载，对两个向量进行按位与操作
template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_and_ps(a, b);
}

// 实现按位或运算符 | 的重载，对两个向量进行按位或操作
template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_or_ps(a, b);
}

// 实现按位异或运算符 ^ 的重载，对两个向量进行按位异或操作
template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_xor_ps(a, b);
}

// 实现向量的相等比较，返回与 other 向量元素相等的布尔向量
inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}

// 实现向量的不等比较，返回与 other 向量元素不相等的布尔向量
inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

// 实现向量的大于比较，返回当前向量元素大于 other 向量元素的布尔向量
inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

// 实现向量的大于等于比较，返回当前向量元素大于等于 other 向量元素的布尔向量
inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}
// 返回一个新的 Vectorized<float> 对象，其中包含当前对象和另一个对象的小于比较结果，并与包含 1.0f 的 Vectorized<float> 对象按位与
inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

// 返回一个新的 Vectorized<float> 对象，其中包含当前对象和另一个对象的小于等于比较结果，并与包含 1.0f 的 Vectorized<float> 对象按位与
inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

// 特化模板函数 convert<float>，用于将 float 类型数组 src 中的数据转换并存储到 dst 数组中，长度为 n
template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  // 使用 Vectorized<float> 的大小作为步长进行循环，转换和存储数据
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  // 处理剩余的不足一个 Vectorized<float> 大小的数据
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

// 特化模板函数 fmadd<float>，使用 AVX 指令执行乘加运算并返回结果
template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return _mm256_fmadd_ps(a, b, c);
}

// 特化模板函数 fmsub<float>，使用 AVX 指令执行乘减运算并返回结果
template <>
Vectorized<float> inline fmsub(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return _mm256_fmsub_ps(a, b, c);
}

// 用于 Inductor CPP 代码生成的特化模板函数 transpose_mxn<float, 8, 8>
template<>
inline void transpose_mxn<float, 8, 8>(
    const float* src,
    int64_t ld_src,
    float* dst,
) {
  // 在 ld_src 的基础上进行转置操作，将结果存储到 dst 中
}

#endif

}} // namespace at::vec::CPU_CAPABILITY
```