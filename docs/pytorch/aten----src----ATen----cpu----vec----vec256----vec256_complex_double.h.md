# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_complex_double.h`

```
#pragma once
// 防止头文件中定义静态数据！
// 参见注释 [不要使用 AVX 编译初始化器]

#include <c10/util/complex.h>      // 包含复数操作的实用函数
#include <c10/util/irange.h>       // 包含整数范围的实用函数
#include <ATen/cpu/vec/intrinsics.h>  // 包含向量操作的低级指令
#include <ATen/cpu/vec/vec_base.h>     // 包含向量操作的基础定义

#if defined(CPU_CAPABILITY_AVX2)
#define SLEEF_STATIC_LIBS
#include <sleef.h>                 // 使用 AVX2 编译时包含 SLEEF 库
#endif

namespace at::vec {
// 参见注释 [CPU_CAPABILITY 命名空间]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

template <> class Vectorized<c10::complex<double>> {
private:
  __m256d values;  // 使用 AVX2 提供的 256 位双精度寄存器存储数据
public:
  using value_type = c10::complex<double>;  // 定义类型别名
  using size_type = int;  // 定义大小类型为整数
  static constexpr size_type size() {  // 静态成员函数，返回向量的大小
    return 2;
  }
  Vectorized() {}  // 默认构造函数
  Vectorized(__m256d v) : values(v) {}  // 构造函数，使用给定的 __m256d 初始化
  Vectorized(c10::complex<double> val) {
    // 构造函数，将 c10::complex<double> 转换为 __m256d 向量
    double real_value = val.real();
    double imag_value = val.imag();
    values = _mm256_setr_pd(real_value, imag_value,
                            real_value, imag_value);
  }
  Vectorized(c10::complex<double> val1, c10::complex<double> val2) {
    // 构造函数，用两个 c10::complex<double> 值填充 __m256d 向量
    values = _mm256_setr_pd(val1.real(), val1.imag(),
                            val2.real(), val2.imag());
  }
  operator __m256d() const {  // 类型转换操作符，将 Vectorized 转换为 __m256d
    return values;
  }
  template <int64_t mask>
  static Vectorized<c10::complex<double>> blend(const Vectorized<c10::complex<double>>& a, const Vectorized<c10::complex<double>>& b) {
    // 模板函数，根据掩码合并两个 Vectorized<c10::complex<double>> 向量
    // 将 c10::complex<V> 的索引掩码转换为 V 的索引掩码：xy -> xxyy
    static_assert (mask > -1 && mask < 4, "Unexpected mask value");
    switch (mask) {
      case 0:
        return a;
      case 1:
        return _mm256_blend_pd(a.values, b.values, 0x03);
      case 2:
        return _mm256_blend_pd(a.values, b.values, 0x0c);
      case 3: break;
    }
    return b;
  }
  static Vectorized<c10::complex<double>> blendv(const Vectorized<c10::complex<double>>& a, const Vectorized<c10::complex<double>>& b,
                               const Vectorized<c10::complex<double>>& mask) {
    // 根据掩码向量 mask 合并两个 Vectorized<c10::complex<double>> 向量
    // 将 c10::complex<V> 的索引掩码转换为 V 的索引掩码：xy -> xxyy
    auto mask_ = _mm256_unpacklo_pd(mask.values, mask.values);
    return _mm256_blendv_pd(a.values, b.values, mask_);
  }
  template<typename step_t>
  static Vectorized<c10::complex<double>> arange(c10::complex<double> base = 0., step_t step = static_cast<step_t>(1)) {
    // 创建一个 arange 向量，填充以 base 为起始，步长为 step 的 c10::complex<double> 值
    return Vectorized<c10::complex<double>>(base,
                                        base + step);
  }
  static Vectorized<c10::complex<double>> set(const Vectorized<c10::complex<double>>& a, const Vectorized<c10::complex<double>>& b,
                            int64_t count = size()) {
    // 设置向量的前 count 个元素为 b，其余为 a
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
    }
    return b;
  }
  static Vectorized<c10::complex<double>> loadu(const void* ptr, int64_t count = size()) {
    // 从未对齐内存地址 ptr 处加载双精度浮点数值到向量，加载 count 个元素
    if (count == size())
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));

    __at_align__ double tmp_values[2*size()];
    // 确保未初始化的内存不会改变输出值，请参考 https://github.com/pytorch/pytorch/issues/32502
    // 有关更多详细信息。我们不使用"={0}"将数组初始化为零，因为gcc会将其编译成两条指令，
    // 而使用循环会编译成一条指令。

    // 使用循环将临时值数组中的所有元素初始化为0.0
    for (const auto i : c10::irange(2*size())) {
      tmp_values[i] = 0.0;
    }

    // 将 ptr 指向的数据按照 c10::complex<double> 的大小拷贝到 tmp_values 中
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(c10::complex<double>));

    // 返回从 tmp_values 加载的 AVX 寄存器值
    return _mm256_load_pd(tmp_values);
  }

  // 将 Vectorized 对象的数据存储到指定的 ptr 中
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 如果 count 等于 size()，直接将 values 存储到 ptr 中
      _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      // 否则，创建临时数组 tmp_values，将 values 存储到 tmp_values 中，再将 tmp_values 拷贝到 ptr 中
      double tmp_values[2*size()];
      _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<double>));
    }
  }

  // 禁用下标操作符[]的 const 版本
  const c10::complex<double>& operator[](int idx) const  = delete;
  // 禁用下标操作符[]的非 const 版本
  c10::complex<double>& operator[](int idx) = delete;

  // 对 Vectorized 对象中的每个元素应用函数 f，并返回结果
  Vectorized<c10::complex<double>> map(c10::complex<double> (*const f)(const c10::complex<double> &)) const {
    // 创建临时数组 tmp，并将 Vectorized 对象的数据存储到 tmp 中
    __at_align__ c10::complex<double> tmp[size()];
    store(tmp);

    // 对 tmp 数组中的每个元素应用函数 f
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }

    // 将经过函数 f 处理后的 tmp 数组加载到新的 Vectorized 对象中并返回
    return loadu(tmp);
  }

  // 计算 Vectorized 对象中每个复数的模的平方
  __m256d abs_2_() const {
    auto val_2 = _mm256_mul_pd(values, values);     // a*a     b*b
    return _mm256_hadd_pd(val_2, val_2);            // a*a+b*b a*a+b*b
  }

  // 计算 Vectorized 对象中每个复数的模
  __m256d abs_() const {
    auto real = _mm256_movedup_pd(values);       // real real
    // movehdup_pd 函数不存在...
    auto imag = _mm256_permute_pd(values, 0xf);  // imag imag
    return Sleef_hypotd4_u05(real, imag);        // abs  abs
  }

  // 计算 Vectorized 对象中每个复数的模，并返回一个新的 Vectorized 对象
  Vectorized<c10::complex<double>> abs() const {
    const __m256d real_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                     0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    return _mm256_and_pd(abs_(), real_mask);        // abs     0
  }

  // 计算 Vectorized 对象中每个复数的幅角
  __m256d angle_() const {
    // angle = atan2(b/a)
    auto b_a = _mm256_permute_pd(values, 0x05);     // b        a
    return Sleef_atan2d4_u10(values, b_a);          // 90-angle angle
  }

  // 计算 Vectorized 对象中每个复数的幅角，并返回一个新的 Vectorized 对象
  Vectorized<c10::complex<double>> angle() const {
    const __m256d real_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                     0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    auto angle = _mm256_permute_pd(angle_(), 0x05); // angle    90-angle
    return _mm256_and_pd(angle, real_mask);         // angle    0
  }

  // 计算 Vectorized 对象中每个复数的符号函数值
  Vectorized<c10::complex<double>> sgn() const {
    auto abs = abs_();
    auto zero = _mm256_setzero_pd();
    auto mask = _mm256_cmp_pd(abs, zero, _CMP_EQ_OQ);
    auto div = _mm256_div_pd(values, abs);
    return _mm256_blendv_pd(div, zero, mask);
  }

  // 获取 Vectorized 对象中每个复数的实部
  __m256d real_() const {
  Vectorized<c10::complex<double>> atan() const;


注释：

  // 返回atan()的Vectorized<c10::complex<double>>版本，实现未提供在此处
  Vectorized<c10::complex<double>> atan() const;



  Vectorized<c10::complex<double>> atanh() const {
    return map(std::atanh);
  }


注释：

  // 返回atanh()函数的Vectorized<c10::complex<double>>版本
  Vectorized<c10::complex<double>> atanh() const {
    return map(std::atanh);
  }



  Vectorized<c10::complex<double>> exp() const {


注释：

  // 返回exp()函数的Vectorized<c10::complex<double>>版本
  Vectorized<c10::complex<double>> exp() const {
  Vectorized<c10::complex<double>> exp2() const {
    // 使用恒等式 2**x = exp(log(2) * x) 计算复数的指数函数
    const __m256d ln_2 = _mm256_set1_pd(c10::ln_2<double>);
    // 将当前向量乘以 ln(2)
    Vectorized<c10::complex<double>> scaled_values = _mm256_mul_pd(values, ln_2);
    // 返回指数函数的结果
    return scaled_values.exp();
  }
  Vectorized<c10::complex<double>> expm1() const {
    // 对当前向量中的每个复数执行 expm1 函数（exp(x) - 1）
    return map(std::expm1);
  }
  Vectorized<c10::complex<double>> sin() const {
    // 对当前向量中的每个复数执行正弦函数
    return map(std::sin);
  }
  Vectorized<c10::complex<double>> sinh() const {
    // 对当前向量中的每个复数执行双曲正弦函数
    return map(std::sinh);
  }
  Vectorized<c10::complex<double>> cos() const {
    // 对当前向量中的每个复数执行余弦函数
    return map(std::cos);
  }
  Vectorized<c10::complex<double>> cosh() const {
    // 对当前向量中的每个复数执行双曲余弦函数
    return map(std::cosh);
  }
  Vectorized<c10::complex<double>> ceil() const {
    // 对当前向量中的每个复数执行向上取整操作
    return _mm256_ceil_pd(values);
  }
  Vectorized<c10::complex<double>> floor() const {
    // 对当前向量中的每个复数执行向下取整操作
    return _mm256_floor_pd(values);
  }
  Vectorized<c10::complex<double>> neg() const {
    auto zero = _mm256_setzero_pd();
    // 对当前向量中的每个复数执行取负操作
    return _mm256_sub_pd(zero, values);
  }
  Vectorized<c10::complex<double>> round() const {
    // 对当前向量中的每个复数执行四舍五入操作
    return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<c10::complex<double>> tan() const {
    // 对当前向量中的每个复数执行正切函数
    return map(std::tan);
  }
  Vectorized<c10::complex<double>> tanh() const {
    // 对当前向量中的每个复数执行双曲正切函数
    return map(std::tanh);
  }
  Vectorized<c10::complex<double>> trunc() const {
    // 对当前向量中的每个复数执行截断操作（向零取整）
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<c10::complex<double>> sqrt() const {
    // 对当前向量中的每个复数执行平方根函数
    return map(std::sqrt);
  }
  Vectorized<c10::complex<double>> reciprocal() const;
  Vectorized<c10::complex<double>> rsqrt() const {
    // 对当前向量中的每个复数执行平方根的倒数函数
    return sqrt().reciprocal();
  }
  Vectorized<c10::complex<double>> pow(const Vectorized<c10::complex<double>> &exp) const {
    // 对当前向量中的每个复数执行指数函数的指定次幂
    __at_align__ c10::complex<double> x_tmp[size()];
    __at_align__ c10::complex<double> y_tmp[size()];
    // 将当前向量和指数向量的值存储到临时数组中
    store(x_tmp);
    exp.store(y_tmp);
    // 对临时数组中的每对复数执行幂运算
    for (const auto i : c10::irange(size())) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    // 加载结果并返回
    return loadu(x_tmp);
  }
  // 使用 _CMP_EQ_OQ 谓词进行比较
  //   `O`: 如果操作数为 NaN，则返回 false
  //   `Q`: 如果操作数为 NaN，则不引发异常
  Vectorized<c10::complex<double>> operator==(const Vectorized<c10::complex<double>>& other) const {
    // 对当前向量和另一个向量进行相等比较
    return _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ);
  }
  Vectorized<c10::complex<double>> operator!=(const Vectorized<c10::complex<double>>& other) const {
  // 使用 AVX 指令集比较两个 Vectorized 对象的双精度复数元素是否不相等，并返回比较结果向量
  return _mm256_cmp_pd(values, other.values, _CMP_NEQ_UQ);
}

// 复数类型的 Vectorized 对象的小于运算符重载，抛出错误信息，复数不支持该操作
Vectorized<c10::complex<double>> operator<(const Vectorized<c10::complex<double>>&) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 复数类型的 Vectorized 对象的小于等于运算符重载，抛出错误信息，复数不支持该操作
Vectorized<c10::complex<double>> operator<=(const Vectorized<c10::complex<double>>&) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 复数类型的 Vectorized 对象的大于运算符重载，抛出错误信息，复数不支持该操作
Vectorized<c10::complex<double>> operator>(const Vectorized<c10::complex<double>>&) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 复数类型的 Vectorized 对象的大于等于运算符重载，抛出错误信息，复数不支持该操作
Vectorized<c10::complex<double>> operator>=(const Vectorized<c10::complex<double>>&) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 返回与另一个复数类型的 Vectorized 对象进行相等比较后的结果
Vectorized<c10::complex<double>> eq(const Vectorized<c10::complex<double>>& other) const;

// 返回与另一个复数类型的 Vectorized 对象进行不等比较后的结果
Vectorized<c10::complex<double>> ne(const Vectorized<c10::complex<double>>& other) const;
};

// 重载加法运算符，对两个复数向量进行逐元素加法
template <> Vectorized<c10::complex<double>> inline operator+(const Vectorized<c10::complex<double>> &a, const Vectorized<c10::complex<double>> &b) {
  return _mm256_add_pd(a, b);
}

// 重载减法运算符，对两个复数向量进行逐元素减法
template <> Vectorized<c10::complex<double>> inline operator-(const Vectorized<c10::complex<double>> &a, const Vectorized<c10::complex<double>> &b) {
  return _mm256_sub_pd(a, b);
}

// 重载乘法运算符，对两个复数向量进行逐元素乘法
template <> Vectorized<c10::complex<double>> inline operator*(const Vectorized<c10::complex<double>> &a, const Vectorized<c10::complex<double>> &b) {
  // 使用 AVX 指令集进行复数乘法运算 (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
  const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm256_mul_pd(a, b);         // ac bd

  auto d_c = _mm256_permute_pd(b, 0x05);    // d c
  d_c = _mm256_xor_pd(sign_mask, d_c);      // d -c
  auto ad_bc = _mm256_mul_pd(a, d_c);       // ad -bc

  auto ret = _mm256_hsub_pd(ac_bd, ad_bc);  // ac - bd  ad + bc
  return ret;
}

// 重载除法运算符，对两个复数向量进行逐元素除法
template <> Vectorized<c10::complex<double>> inline operator/(const Vectorized<c10::complex<double>> &a, const Vectorized<c10::complex<double>> &b) {
  // 使用 AVX 指令集进行复数除法 (a + bi) / (c + di)
  auto mask = _mm256_set1_pd(-0.f);
  auto fabs_cd = _mm256_andnot_pd(mask, b);     // |c| |d|
  auto fabs_dc = _mm256_permute_pd(fabs_cd, 0x05);   // |d| |c|
  auto scale = _mm256_div_pd(_mm256_set1_pd(1.0f), _mm256_max_pd(fabs_cd, fabs_dc));  // 1/sc 1/sc
  auto a2 = _mm256_mul_pd(a, scale);         // a/sc b/sc
  auto b2 = _mm256_mul_pd(b, scale);         // c/sc d/sc
  auto acbd2 = _mm256_mul_pd(a2, b2);

  const __m256d sign_mask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
  auto dc2 = _mm256_permute_pd(b2, 0x05);    // d/sc c/sc
  dc2 = _mm256_xor_pd(sign_mask, dc2);       // -d/|c,d| c/sc
  auto adbc2 = _mm256_mul_pd(a2, dc2);       // -ad/sc^2 bc/sc^2
  auto res2 = _mm256_hadd_pd(acbd2, adbc2);  // (ac+bd)/sc^2 (bc-ad)/sc^2

  // 计算分母
  auto denom2 = Vectorized<c10::complex<double>>(b2).abs_2_();  // (c^2+d^2)/sc^2 (c^2+d^2)/sc^2
  res2 = _mm256_div_pd(res2, denom2);
  return res2;
}

// 求复数向量的倒数，用于复数除法的实现
inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::reciprocal() const {
  // 使用 AVX 指令集进行复数的倒数运算 (a + bi) / (c + di)
  // 实部：re = c / abs_2()
  // 虚部：im = d / abs_2()
  const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm256_xor_pd(sign_mask, values);    // c -d
  return _mm256_div_pd(c_d, abs_2_());
}
inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::atan() const {
  // atan(x) = i/2 * ln((i + z)/(i - z))
  // 设置复数单位i，作为常量向量
  const __m256d i = _mm256_setr_pd(0.0, 1.0, 0.0, 1.0);
  // 设置常量向量 i_half = 0.5*i，用于后续计算
  const Vectorized i_half = _mm256_setr_pd(0.0, 0.5, 0.0, 0.5);

  // 计算 sum = i + values，其中 values 是当前对象的值
  auto sum = Vectorized(_mm256_add_pd(i, values));                      // a        1+b
  // 计算 sub = i - values，其中 values 是当前对象的值
  auto sub = Vectorized(_mm256_sub_pd(i, values));                      // -a       1-b
  // 计算 ln = log((i + z)/(i - z))，其中 z 是当前对象的值
  auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
  // 返回结果 i_half * ln，即 i/2 * ln((i + z)/(i - z))
  return i_half * ln;                                                 // i/2*ln()
}

template <>
Vectorized<c10::complex<double>> inline maximum(const Vectorized<c10::complex<double>>& a, const Vectorized<c10::complex<double>>& b) {
  // 计算向量 a 和 b 的模的平方
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // 使用比较指令比较 abs_a 和 abs_b，生成掩码 mask
  auto mask = _mm256_cmp_pd(abs_a, abs_b, _CMP_LT_OQ);
  // 根据掩码选择较大的值构成 max 向量
  auto max = _mm256_blendv_pd(a, b, mask);
  // 使用无序比较指令比较 abs_a 和 abs_b，生成 NaN 判断的掩码
  auto isnan = _mm256_cmp_pd(abs_a, abs_b, _CMP_UNORD_Q);
  // 返回 max 或 isnan，利用掩码处理 NaN
  return _mm256_or_pd(max, isnan);
}

template <>
Vectorized<c10::complex<double>> inline minimum(const Vectorized<c10::complex<double>>& a, const Vectorized<c10::complex<double>>& b) {
  // 计算向量 a 和 b 的模的平方
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // 使用比较指令比较 abs_a 和 abs_b，生成掩码 mask
  auto mask = _mm256_cmp_pd(abs_a, abs_b, _CMP_GT_OQ);
  // 根据掩码选择较小的值构成 min 向量
  auto min = _mm256_blendv_pd(a, b, mask);
  // 使用无序比较指令比较 abs_a 和 abs_b，生成 NaN 判断的掩码
  auto isnan = _mm256_cmp_pd(abs_a, abs_b, _CMP_UNORD_Q);
  // 返回 min 或 isnan，利用掩码处理 NaN
  return _mm256_or_pd(min, isnan);
}

template <>
Vectorized<c10::complex<double>> inline operator&(const Vectorized<c10::complex<double>>& a, const Vectorized<c10::complex<double>>& b) {
  // 返回 a 和 b 的按位与操作结果
  return _mm256_and_pd(a, b);
}

template <>
Vectorized<c10::complex<double>> inline operator|(const Vectorized<c10::complex<double>>& a, const Vectorized<c10::complex<double>>& b) {
  // 返回 a 和 b 的按位或操作结果
  return _mm256_or_pd(a, b);
}

template <>
Vectorized<c10::complex<double>> inline operator^(const Vectorized<c10::complex<double>>& a, const Vectorized<c10::complex<double>>& b) {
  // 返回 a 和 b 的按位异或操作结果
  return _mm256_xor_pd(a, b);
}

inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::eq(const Vectorized<c10::complex<double>>& other) const {
  // 比较当前对象和 other 对象的实部和虚部是否相等
  auto eq = (*this == other);  // 比较实部和虚部
  // 如果实部和虚部都相等，则返回向量元素值为 1.0 的结果向量
  return (eq.real() & eq.imag()) & Vectorized<c10::complex<double>>(_mm256_set1_pd(1.0));
}

inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::ne(const Vectorized<c10::complex<double>>& other) const {
  // 比较当前对象和 other 对象的实部和虚部是否不相等
  auto ne = (*this != other);  // 比较实部和虚部
  // 如果实部和虚部有任何一个不相等，则返回向量元素值为 1.0 的结果向量
  return (ne.real() | ne.imag()) & Vectorized<c10::complex<double>>(_mm256_set1_pd(1.0));
}

#endif

}} // namespace at::vec::CPU_CAPABILITY
```