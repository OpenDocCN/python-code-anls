# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_complex_double.h`

```py
#pragma once
// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <c10/util/complex.h>
#include <c10/util/irange.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#if defined(CPU_CAPABILITY_AVX512)
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512)

template <> class Vectorized<c10::complex<double>> {
private:
  __m512d values; // SIMD vector that holds eight double-precision values
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0}; // Zero-initialized SIMD integer vector
public:
  using value_type = c10::complex<double>; // Type alias for complex doubles
  using size_type = int; // Type alias for size, using int
  static constexpr size_type size() {
    return 4; // Returns the size of the vector (number of elements it holds)
  }
  Vectorized() {} // Default constructor
  Vectorized(__m512d v) : values(v) {} // Constructor initializing with a SIMD vector
  Vectorized(c10::complex<double> val) {
    // Constructor initializing with a single complex double value
    double real_value = val.real();
    double imag_value = val.imag();
    values = _mm512_setr_pd(real_value, imag_value, real_value, imag_value,
                            real_value, imag_value, real_value, imag_value);
  }
  Vectorized(c10::complex<double> val1, c10::complex<double> val2,
            c10::complex<double> val3, c10::complex<double> val4) {
    // Constructor initializing with four complex double values
    values = _mm512_setr_pd(val1.real(), val1.imag(),
                            val2.real(), val2.imag(),
                            val3.real(), val3.imag(),
                            val4.real(), val4.imag());
  }
  operator __m512d() const {
    return values; // Conversion operator to retrieve the SIMD vector
  }
  template <int64_t mask>
  static Vectorized<c10::complex<double>> blend(const Vectorized<c10::complex<double>>& a,
                                               const Vectorized<c10::complex<double>>& b) {
     // Blend two Vectorized<c10::complex<double>> instances based on a mask
    // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    // NOLINTNEXTLINE(clang-diagnostic-warning)
    switch (mask) {
      case 0:
        return a;
      case 1:
        // 使用掩码 0x03 混合 a 和 b 的值，掩码的二进制形式为 b0000 0011
        return _mm512_mask_blend_pd(0x03, a.values, b.values);
      case 2:
        // 使用掩码 0x0C 混合 a 和 b 的值，掩码的二进制形式为 b0000 1100
        return _mm512_mask_blend_pd(0x0C, a.values, b.values);
      case 3:
        // 使用掩码 0x0F 混合 a 和 b 的值，掩码的二进制形式为 b0000 1111
        return _mm512_mask_blend_pd(0x0F, a.values, b.values);
      case 4:
        // 使用掩码 0x30 混合 a 和 b 的值，掩码的二进制形式为 b0011 0000
        return _mm512_mask_blend_pd(0x30, a.values, b.values);
      case 5:
        // 使用掩码 0x33 混合 a 和 b 的值，掩码的二进制形式为 b0011 0011
        return _mm512_mask_blend_pd(0x33, a.values, b.values);
      case 6:
        // 使用掩码 0x3C 混合 a 和 b 的值，掩码的二进制形式为 b0011 1100
        return _mm512_mask_blend_pd(0x3C, a.values, b.values);
      case 7:
        // 使用掩码 0x3F 混合 a 和 b 的值，掩码的二进制形式为 b0011 1111
        return _mm512_mask_blend_pd(0x3F, a.values, b.values);
      case 8:
        // 使用掩码 0xC0 混合 a 和 b 的值，掩码的二进制形式为 b1100 0000
        return _mm512_mask_blend_pd(0xC0, a.values, b.values);
      case 9:
        // 使用掩码 0xC3 混合 a 和 b 的值，掩码的二进制形式为 b1100 0011
        return _mm512_mask_blend_pd(0xC3, a.values, b.values);
      case 10:
        // 使用掩码 0xCC 混合 a 和 b 的值，掩码的二进制形式为 b1100 1100
        return _mm512_mask_blend_pd(0xCC, a.values, b.values);
      case 11:
        // 使用掩码 0xCF 混合 a 和 b 的值，掩码的二进制形式为 b1100 1111
        return _mm512_mask_blend_pd(0xCF, a.values, b.values);
      case 12:
        // 使用掩码 0xF0 混合 a 和 b 的值，掩码的二进制形式为 b1111 0000
        return _mm512_mask_blend_pd(0xF0, a.values, b.values);
      case 13:
        // 使用掩码 0xF3 混合 a 和 b 的值，掩码的二进制形式为 b1111 0011
        return _mm512_mask_blend_pd(0xF3, a.values, b.values);
      case 14:
        // 使用掩码 0xFC 混合 a 和 b 的值，掩码的二进制形式为 b1111 1100
        return _mm512_mask_blend_pd(0xFC, a.values, b.values);
      case 15:
        // 使用掩码 0xFF 混合 a 和 b 的值，掩码的二进制形式为 b1111 1111
        return _mm512_mask_blend_pd(0xFF, a.values, b.values);
    }
    // 如果没有匹配的掩码，返回 b 的值
    return b;
  }

  static Vectorized<c10::complex<double>> blendv(const Vectorized<c10::complex<double>>& a,
                                                const Vectorized<c10::complex<double>>& b,
                                                const Vectorized<c10::complex<double>>& mask) {
    // 将 c10::complex<V> 的掩码转换为 V 的掩码：xy -> xxyy
    auto mask_ = _mm512_unpacklo_pd(mask.values, mask.values);
    // 创建一个全 1 的掩码
    auto all_ones = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    // 使用比较操作创建掩码 mmask
    auto mmask = _mm512_cmp_epi64_mask(_mm512_castpd_si512(mask_), all_ones, _MM_CMPINT_EQ);
    // 使用 mmask 混合 a 和 b 的值
    return _mm512_mask_blend_pd(mmask, a.values, b.values);
  }

  template<typename step_t>
  static Vectorized<c10::complex<double>> arange(c10::complex<double> base = 0.,
                                                step_t step = static_cast<step_t>(1)) {
    // 返回一个 Vectorized 对象，表示从 base 开始，以 step 为步长的一系列值
    return Vectorized<c10::complex<double>>(base,
                                           base + c10::complex<double>(1)*step,
                                           base + c10::complex<double>(2)*step,
                                           base + c10::complex<double>(3)*step);
  }

  static Vectorized<c10::complex<double>> set(const Vectorized<c10::complex<double>>& a,
                                             const Vectorized<c10::complex<double>>& b,
                                             int64_t count = size()) {
    // 根据 count 的不同情况进行不同的处理
    switch (count) {
      // 如果 count 等于 0，返回 a
      case 0:
        return a;
      // 如果 count 等于 1，使用 blend<1> 混合函数处理 a 和 b，并返回结果
      case 1:
        return blend<1>(a, b);
      // 如果 count 等于 2，使用 blend<3> 混合函数处理 a 和 b，并返回结果
      case 2:
        return blend<3>(a, b);
      // 如果 count 等于 3，使用 blend<7> 混合函数处理 a 和 b，并返回结果
      case 3:
        return blend<7>(a, b);
    }
    // 如果 count 不在以上情况中，则返回 b
    return b;
  }
  // 从给定指针 ptr 处加载 Vectorized 对象，数量为 count，默认为 size()
  static Vectorized<c10::complex<double>> loadu(const void* ptr, int64_t count = size()) {
    // 如果 count 等于 size()，直接加载内存中的数据到 Vectorized 对象
    if (count == size())
      return _mm512_loadu_pd(reinterpret_cast<const double*>(ptr));

    // 否则，创建临时数组 tmp_values 来存储加载的数据，确保未初始化的内存不影响输出值
    __at_align__ double tmp_values[2*size()];
    // 使用循环初始化数组元素为 0.0
    for (const auto i : c10::irange(2*size())) {
      tmp_values[i] = 0.0;
    }
    // 将 ptr 指向的数据复制到 tmp_values 中，只复制 count 大小的数据
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(c10::complex<double>));
    // 返回加载 tmp_values 数据后的 Vectorized 对象
    return _mm512_load_pd(tmp_values);
  }
  // 将 Vectorized 对象的数据存储到给定指针 ptr 处，数量为 count，默认为 size()
  void store(void* ptr, int count = size()) const {
    // 如果 count 等于 size()，直接将 values 中的数据存储到 ptr 指向的内存中
    if (count == size()) {
      _mm512_storeu_pd(reinterpret_cast<double*>(ptr), values);
    // 如果 count 大于 0，创建临时数组 tmp_values 来存储 values 中的数据
    } else if (count > 0) {
      double tmp_values[2*size()];
      // 将 values 中的数据存储到 tmp_values 中
      _mm512_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      // 将 tmp_values 中的数据复制到 ptr 指向的内存中，只复制 count 大小的数据
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<double>));
    }
  }
  // 禁用操作符重载，不允许通过索引访问 Vectorized 对象中的元素
  const c10::complex<double>& operator[](int idx) const  = delete;
  // 禁用操作符重载，不允许通过索引访问 Vectorized 对象中的元素
  c10::complex<double>& operator[](int idx) = delete;
  // 对 Vectorized 对象中的每个元素应用函数 f，并返回结果
  Vectorized<c10::complex<double>> map(c10::complex<double> (*const f)(const c10::complex<double> &)) const {
    // 创建临时数组 tmp 来存储 Vectorized 对象中的数据
    __at_align__ c10::complex<double> tmp[size()];
    // 将 Vectorized 对象中的数据存储到 tmp 中
    store(tmp);
    // 对 tmp 中的每个元素应用函数 f，并更新 tmp 中的元素值
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    // 返回应用函数 f 后的新的 Vectorized 对象
    return loadu(tmp);
  }
  // 使用 AVX512 指令集中的指令计算向量的平方和
  // 返回值为两个向量的每个元素的平方和
  __m512d abs_2_() const {
    auto val_2 = _mm512_mul_pd(values, values);     // 计算每个元素的平方
    return hadd_pd(val_2, val_2);            // 水平加法：计算两个向量的每个元素的平方和
  }
  // 使用 AVX512 指令集中的指令计算向量的绝对值
  // 返回值为向量的每个元素的绝对值
  __m512d abs_() const {
    auto real = _mm512_movedup_pd(values);        // 复制向量中的实部到每个元素
    auto imag = _mm512_permute_pd(values, 0xff);  // 将向量中的虚部进行排列
    return Sleef_hypotd8_u05(real, imag);         // 调用Sleef库中的hypot函数，计算复数的模
  }

  Vectorized<c10::complex<double>> abs() const {
    const __m512d real_mask = _mm512_castsi512_pd(_mm512_setr_epi64(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    return _mm512_and_pd(abs_(), real_mask);        // 返回当前向量的绝对值，实部取绝对值
  }

  __m512d angle_() const {
    // 计算复数的幅角，即atan2(b/a)
    auto b_a = _mm512_permute_pd(values, 0x55);     // 将复数向量按顺序取出两个部分
    return Sleef_atan2d8_u10(values, b_a);          // 返回计算得到的幅角
  }

  Vectorized<c10::complex<double>> angle() const {
    const __m512d real_mask = _mm512_castsi512_pd(_mm512_setr_epi64(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    auto angle = _mm512_permute_pd(angle_(), 0x55); // 获取复数的幅角
    return _mm512_and_pd(angle, real_mask);         // 返回幅角的实部
  }

  Vectorized<c10::complex<double>> sgn() const {
    auto abs = abs_();                             // 计算复数的绝对值
    auto zero = _mm512_setzero_pd();                // 创建一个全零向量
    auto mask = _mm512_cmp_pd_mask(abs, zero, _CMP_EQ_OQ); // 生成一个标志位向量，标识复数是否为零
    auto div = _mm512_div_pd(values, abs);          // 计算复数的标准化向量
    return _mm512_mask_blend_pd(mask, div, zero);   // 返回标准化后的复数向量
  }

  __m512d real_() const {
    const __m512d real_mask = _mm512_castsi512_pd(_mm512_setr_epi64(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    return _mm512_and_pd(values, real_mask);        // 提取复数向量的实部
  }

  Vectorized<c10::complex<double>> real() const {
    return real_();                                 // 返回复数向量的实部
  }

  __m512d imag_() const {
    const __m512d imag_mask = _mm512_castsi512_pd(_mm512_setr_epi64(0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                                                    0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                                                    0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                                                    0x0000000000000000, 0xFFFFFFFFFFFFFFFF));
    return _mm512_and_pd(values, imag_mask);        // 提取复数向量的虚部
  }

  Vectorized<c10::complex<double>> imag() const {
    return _mm512_permute_pd(imag_(), 0x55);        // 返回复数向量的虚部
  }
    const __m512d sign_mask = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    // 创建一个包含特定值的 AVX-512 双精度浮点向量，用于掩码操作
    return _mm512_xor_pd(values, sign_mask);           // 对输入的 AVX-512 双精度浮点向量执行按位异或操作，用来实现值的符号翻转
  }
  Vectorized<c10::complex<double>> conj() const {
    return conj_();                                    // 返回调用类内部的 conj_() 方法的结果，用于计算共轭复数
  }
  Vectorized<c10::complex<double>> log() const {
    // 大多数三角函数操作使用 log() 函数来提高复数性能
    return map(std::log);                              // 返回应用 std::log 函数映射到每个元素的结果向量，用于计算复数的自然对数
  }
  Vectorized<c10::complex<double>> log2() const {
    const __m512d log2_ = _mm512_set1_pd(std::log(2));  // 创建一个 AVX-512 双精度浮点向量，其中所有元素值为 log(2)
    return _mm512_div_pd(log(), log2_);                // 返回 log() 函数计算结果与 log2_ 的每个元素相除的结果向量，用于计算复数的以2为底的对数
  }
  Vectorized<c10::complex<double>> log10() const {
    const __m512d log10_ = _mm512_set1_pd(std::log(10));  // 创建一个 AVX-512 双精度浮点向量，其中所有元素值为 log(10)
    return _mm512_div_pd(log(), log10_);                // 返回 log() 函数计算结果与 log10_ 的每个元素相除的结果向量，用于计算复数的以10为底的对数
  }
  Vectorized<c10::complex<double>> log1p() const {
    return map(std::log1p);                            // 返回应用 std::log1p 函数映射到每个元素的结果向量，用于计算复数的 ln(1+x)
  }
  Vectorized<c10::complex<double>> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    const __m512d one = _mm512_set1_pd(1);              // 创建一个 AVX-512 双精度浮点向量，其中所有元素值为 1

    auto conj = conj_();                                // 计算复共轭
    auto b_a = _mm512_permute_pd(conj, 0x55);           // 从复共轭中创建新的 AVX-512 双精度浮点向量，重新排列元素得到 -b 和 a
    auto ab = _mm512_mul_pd(conj, b_a);                 // 计算复共轭乘积，得到 -ab

    auto im = _mm512_add_pd(ab, ab);                    // 将 -ab 向量的每个元素相加，得到 -2ab

    auto val_2 = _mm512_mul_pd(values, values);         // 对输入值的每个元素分别平方
    auto re = hsub_pd(val_2, _mm512_permute_pd(val_2, 0x55));  // 计算水平减法，得到 a*a-b*b 和 b*b-a*a
    re = _mm512_sub_pd(one, re);                        // 计算 1 - (a**2 - b**2)

    auto root = Vectorized(_mm512_mask_blend_pd(0xAA, re, im)).sqrt();  // 使用掩码混合运算得到平方根，用于计算复数的平方根
    auto ln = Vectorized(_mm512_add_pd(b_a, root)).log();  // 计算对数，用于计算复数的自然对数
    return Vectorized(_mm512_permute_pd(ln.values, 0x55)).conj();  // 返回调整元素顺序的结果向量的共轭，用于计算复数的最终 asin() 结果
  }
  Vectorized<c10::complex<double>> acos() const {
    // acos(x) = pi/2 - asin(x)
    constexpr auto pi_2d = c10::pi<double> / 2;        // 创建一个常量，表示 pi/2
    const __m512d pi_2 = _mm512_setr_pd(pi_2d, 0.0, pi_2d, 0.0, pi_2d, 0.0, pi_2d, 0.0);  // 创建一个 AVX-512 双精度浮点向量，其中每个元素值为 pi/2
    return _mm512_sub_pd(pi_2, asin());                 // 返回 pi/2 减去 asin(x) 的结果向量，用于计算复数的 arccosine
  }
  Vectorized<c10::complex<double>> atan() const;
  Vectorized<c10::complex<double>> atanh() const {
    return map(std::atanh);                            // 返回应用 std::atanh 函数映射到每个元素的结果向量，用于计算复数的双曲正切反函数
  }
  Vectorized<c10::complex<double>> exp() const {
    //exp(a + bi)
    // = exp(a)*(cos(b) + sin(b)i)
    auto exp = Sleef_expd8_u10(values);                 // 使用 Sleef 库计算指数函数的结果，用于计算复数的指数函数
    exp = _mm512_mask_blend_pd(0xAA, exp, _mm512_permute_pd(exp, 0x55));  // 使用掩码混合运算，重新排列元素，得到复数的指数函数结果

    auto sin_cos = Sleef_sincosd8_u10(values);          // 使用 Sleef 库计算正弦和余弦函数的结果
    auto cos_sin = _mm512_mask_blend_pd(0xAA, _mm512_permute_pd(sin_cos.y, 0x55),
                                   sin_cos.x);         // 使用掩码混合运算，重新排列元素，得到复数的正弦和余弦函数结果
    return _mm512_mul_pd(exp, cos_sin);                 // 返回复数的指数函数结果乘以正弦和余弦函数结果，用于计算复数的指数函数
  }
  Vectorized<c10::complex<double>> exp2() const {
    // Use identity 2**x = exp(log(2) * x)
    const __m512d ln_2 = _mm512_set1_pd(c10::ln_2<double>);  // 创建一个 AVX-512 双精度浮点向量，其中所有元素值为 ln(2)
    // 计算向量中每个复数值与 ln_2 的乘积，并返回结果向量
    Vectorized<c10::complex<double>> scaled_values = _mm512_mul_pd(values, ln_2);
    // 对乘积向量中每个复数值执行指数函数，返回指数化后的向量
    return scaled_values.exp();
    }
    
    // 返回当前向量中每个复数值执行 expm1 函数后的结果向量
    Vectorized<c10::complex<double>> expm1() const {
        return map(std::expm1);
    }
    
    // 返回当前向量中每个复数值执行 sin 函数后的结果向量
    Vectorized<c10::complex<double>> sin() const {
        return map(std::sin);
    }
    
    // 返回当前向量中每个复数值执行 sinh 函数后的结果向量
    Vectorized<c10::complex<double>> sinh() const {
        return map(std::sinh);
    }
    
    // 返回当前向量中每个复数值执行 cos 函数后的结果向量
    Vectorized<c10::complex<double>> cos() const {
        return map(std::cos);
    }
    
    // 返回当前向量中每个复数值执行 cosh 函数后的结果向量
    Vectorized<c10::complex<double>> cosh() const {
        return map(std::cosh);
    }
    
    // 返回当前向量中每个复数值执行向上取整后的结果向量
    Vectorized<c10::complex<double>> ceil() const {
        return _mm512_ceil_pd(values);
    }
    
    // 返回当前向量中每个复数值执行向下取整后的结果向量
    Vectorized<c10::complex<double>> floor() const {
        return _mm512_floor_pd(values);
    }
    
    // 返回当前向量中每个复数值取负后的结果向量
    Vectorized<c10::complex<double>> neg() const {
        auto zero = _mm512_setzero_pd();
        return _mm512_sub_pd(zero, values);
    }
    
    // 返回当前向量中每个复数值执行四舍五入后的结果向量
    Vectorized<c10::complex<double>> round() const {
        return _mm512_roundscale_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    
    // 返回当前向量中每个复数值执行 tan 函数后的结果向量
    Vectorized<c10::complex<double>> tan() const {
        return map(std::tan);
    }
    
    // 返回当前向量中每个复数值执行 tanh 函数后的结果向量
    Vectorized<c10::complex<double>> tanh() const {
        return map(std::tanh);
    }
    
    // 返回当前向量中每个复数值执行向零取整后的结果向量
    Vectorized<c10::complex<double>> trunc() const {
        return _mm512_roundscale_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    }
    
    // 返回当前向量中每个复数值执行平方根函数后的结果向量
    Vectorized<c10::complex<double>> sqrt() const {
        return map(std::sqrt);
    }
    
    // 返回当前向量中每个复数值的倒数向量
    Vectorized<c10::complex<double>> reciprocal() const;
    
    // 返回当前向量中每个复数值的平方根的倒数向量
    Vectorized<c10::complex<double>> rsqrt() const {
        return sqrt().reciprocal();
    }
    
    // 返回当前向量中每个复数值与给定向量中对应值的幂次方结果向量
    Vectorized<c10::complex<double>> pow(const Vectorized<c10::complex<double>> &exp) const {
        __at_align__ c10::complex<double> x_tmp[size()];
        __at_align__ c10::complex<double> y_tmp[size()];
        store(x_tmp);
        exp.store(y_tmp);
        for (const auto i : c10::irange(size())) {
            x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
        }
        return loadu(x_tmp);
    }
    
    // 通过使用 _CMP_EQ_OQ 谓词进行比较，返回与另一个向量相等的掩码向量
    Vectorized<c10::complex<double>> operator==(const Vectorized<c10::complex<double>>& other) const {
        auto mask = _mm512_cmp_pd_mask(values, other.values, _CMP_EQ_OQ);
        return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, mask,
                                                          0xFFFFFFFFFFFFFFFF));
    }
    
    // 通过使用 _CMP_NEQ_UQ 谓词进行比较，返回与另一个向量不等的掩码向量
    Vectorized<c10::complex<double>> operator!=(const Vectorized<c10::complex<double>>& other) const {
        auto mask = _mm512_cmp_pd_mask(values, other.values, _CMP_NEQ_UQ);
        return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, mask,
                                                          0xFFFFFFFFFFFFFFFF));
    }
    
    // 抛出错误，因为复数不支持小于操作符
    Vectorized<c10::complex<double>> operator<(const Vectorized<c10::complex<double>>& other) const {
        TORCH_CHECK(false, "not supported for complex numbers");
    }
    
    // 抛出错误，因为复数不支持小于等于操作符
    Vectorized<c10::complex<double>> operator<=(const Vectorized<c10::complex<double>>& other) const {
        TORCH_CHECK(false, "not supported for complex numbers");
    }
  // 检查条件为 false，抛出错误信息 "not supported for complex numbers"
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 定义复数类型的向量化操作符 >，不支持复数，抛出错误信息
Vectorized<c10::complex<double>> operator>(const Vectorized<c10::complex<double>>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 定义复数类型的向量化操作符 >=，不支持复数，抛出错误信息
Vectorized<c10::complex<double>> operator>=(const Vectorized<c10::complex<double>>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 声明复数类型的向量化操作符 eq，实现未提供在此处
Vectorized<c10::complex<double>> eq(const Vectorized<c10::complex<double>>& other) const;

// 声明复数类型的向量化操作符 ne，实现未提供在此处
Vectorized<c10::complex<double>> ne(const Vectorized<c10::complex<double>>& other) const;
};

// 重载运算符+：复数向量的加法
template <> Vectorized<c10::complex<double>> inline operator+(const Vectorized<c10::complex<double>> &a,
                                                             const Vectorized<c10::complex<double>> &b) {
  return _mm512_add_pd(a, b);  // 使用 SIMD 指令执行复数向量的加法
}

// 重载运算符-：复数向量的减法
template <> Vectorized<c10::complex<double>> inline operator-(const Vectorized<c10::complex<double>> &a,
                                                             const Vectorized<c10::complex<double>> &b) {
  return _mm512_sub_pd(a, b);  // 使用 SIMD 指令执行复数向量的减法
}

// 重载运算符*：复数向量的乘法
template <> Vectorized<c10::complex<double>> inline operator*(const Vectorized<c10::complex<double>> &a,
                                                             const Vectorized<c10::complex<double>> &b) {
  // 计算复数向量乘法 (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
  const __m512d sign_mask = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm512_mul_pd(a, b);         // 计算 ac 和 bd 的乘积
  auto d_c = _mm512_permute_pd(b, 0x55);    // 交换 b 中元素的位置得到 d 和 c
  d_c = _mm512_xor_pd(sign_mask, d_c);      // 将 d 的符号取反
  auto ad_bc = _mm512_mul_pd(a, d_c);       // 计算 ad 和 bc 的乘积
  auto ret = Vectorized<c10::complex<double>>::hsub_pd(ac_bd, ad_bc);  // 返回结果 (ac - bd) + (ad + bc)i
  return ret;
}

// 重载运算符/：复数向量的除法
template <> Vectorized<c10::complex<double>> inline operator/(const Vectorized<c10::complex<double>> &a,
                                                             const Vectorized<c10::complex<double>> &b) {
  // 计算复数向量除法 (a + bi) / (c + di)
  auto mask = _mm512_set1_pd(-0.f);
  auto fabs_cd = _mm512_andnot_pd(mask, b);     // 计算 b 的绝对值
  auto fabs_dc = _mm512_permute_pd(fabs_cd, 0x55);   // 交换 fabs_cd 中元素的位置得到 |d| 和 |c|
  auto scale = _mm512_rcp14_pd(_mm512_max_pd(fabs_cd, fabs_dc));  // 计算除法的缩放因子
  auto a2 = _mm512_mul_pd(a, scale);         // 对 a 进行缩放
  auto b2 = _mm512_mul_pd(b, scale);         // 对 b 进行缩放
  auto acbd2 = _mm512_mul_pd(a2, b2);

  const __m512d sign_mask = _mm512_setr_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
  auto dc2 = _mm512_permute_pd(b2, 0x55);    // 交换 b2 中元素的位置得到 d/sc 和 c/sc
  dc2 = _mm512_xor_pd(sign_mask, dc2);       // 将 d 的符号取反
  auto adbc2 = _mm512_mul_pd(a2, dc2);       // 计算 -ad/sc^2 和 bc/sc^2
  auto res2 = Vectorized<c10::complex<double>>::hadd_pd(acbd2, adbc2);  // 返回结果 (ac+bd)/sc^2 和 (bc-ad)/sc^2

  auto denom2 = Vectorized<c10::complex<double>>(b2).abs_2_();  // 计算分母 (c^2+d^2)/sc^2
  res2 = _mm512_div_pd(res2, denom2);  // 对结果进行除法运算
  return res2;
}

// 求复数向量的倒数
inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::reciprocal() const {
  const __m512d sign_mask = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm512_xor_pd(sign_mask, values);    // 将 values 中复数部分的符号取反
  return _mm512_div_pd(c_d, abs_2_());  // 返回复数向量的倒数
}
inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::atan() const {
  // atan(x) = i/2 * ln((i + z)/(i - z))
  // 定义常数向量 i，包含复数单位 i
  const __m512d i = _mm512_setr_pd(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  // 定义常数向量 i_half，表示 i/2
  const Vectorized i_half = _mm512_setr_pd(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);

  // 将 values 向量与 i 向量逐元素相加，得到 sum 向量
  auto sum = Vectorized(_mm512_add_pd(i, values));                      // a        1+b
  // 将 values 向量与 i 向量逐元素相减，得到 sub 向量
  auto sub = Vectorized(_mm512_sub_pd(i, values));                      // -a       1-b
  // 计算 sum/sub 的自然对数，表示 ln((i + z)/(i - z))
  auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
  // 返回 i_half 与 ln 相乘的结果，表示 i/2*ln()
  return i_half*ln;                                                 // i/2*ln()
}

template <>
Vectorized<c10::complex<double>> inline maximum(const Vectorized<c10::complex<double>>& a,
                                               const Vectorized<c10::complex<double>>& b) {
  // 创建一个所有元素为零的掩码向量 zero_vec
  auto zero_vec = _mm512_set1_epi64(0);
  // 分别计算 a 和 b 的绝对值的平方
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // 使用 _CMP_LT_OQ 比较 abs_a 和 abs_b，生成掩码 mask，标记 abs_a < abs_b 的位置
  auto mask = _mm512_cmp_pd_mask(abs_a, abs_b, _CMP_LT_OQ);
  // 根据 mask，在 a 和 b 之间选择最大值，生成 max 向量
  auto max = _mm512_mask_blend_pd(mask, a, b);
  // 利用 _CMP_UNORD_Q 检测 abs_a 和 abs_b 是否包含 NaN，生成对应的掩码 isnan_mask
  auto isnan_mask = _mm512_cmp_pd_mask(abs_a, abs_b, _CMP_UNORD_Q);
  // 使用 isnan_mask 设置一个所有元素均为 1 的向量 isnan
  auto isnan = _mm512_mask_set1_epi64(zero_vec, isnan_mask,
                                      0xFFFFFFFFFFFFFFFF);
  // 将 max 与 isnan 向量逐元素或操作，处理 NaN
  return _mm512_or_pd(max, _mm512_castsi512_pd(isnan));
}

template <>
Vectorized<c10::complex<double>> inline minimum(const Vectorized<c10::complex<double>>& a,
                                               const Vectorized<c10::complex<double>>& b) {
  // 创建一个所有元素为零的掩码向量 zero_vec
  auto zero_vec = _mm512_set1_epi64(0);
  // 分别计算 a 和 b 的绝对值的平方
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // 使用 _CMP_GT_OQ 比较 abs_a 和 abs_b，生成掩码 mask，标记 abs_a > abs_b 的位置
  auto mask = _mm512_cmp_pd_mask(abs_a, abs_b, _CMP_GT_OQ);
  // 根据 mask，在 a 和 b 之间选择最小值，生成 min 向量
  auto min = _mm512_mask_blend_pd(mask, a, b);
  // 利用 _CMP_UNORD_Q 检测 abs_a 和 abs_b 是否包含 NaN，生成对应的掩码 isnan_mask
  auto isnan_mask = _mm512_cmp_pd_mask(abs_a, abs_b, _CMP_UNORD_Q);
  // 使用 isnan_mask 设置一个所有元素均为 1 的向量 isnan
  auto isnan = _mm512_mask_set1_epi64(zero_vec, isnan_mask,
                                      0xFFFFFFFFFFFFFFFF);
  // 将 min 与 isnan 向量逐元素或操作，处理 NaN
  return _mm512_or_pd(min, _mm512_castsi512_pd(isnan));
}

template <>
Vectorized<c10::complex<double>> inline operator&(const Vectorized<c10::complex<double>>& a,
                                                 const Vectorized<c10::complex<double>>& b) {
  // 对 a 和 b 向量逐元素进行按位与操作，返回结果
  return _mm512_and_pd(a, b);
}

template <>
Vectorized<c10::complex<double>> inline operator|(const Vectorized<c10::complex<double>>& a,
                                                 const Vectorized<c10::complex<double>>& b) {
  // 对 a 和 b 向量逐元素进行按位或操作，返回结果
  return _mm512_or_pd(a, b);
}

template <>
Vectorized<c10::complex<double>> inline operator^(const Vectorized<c10::complex<double>>& a,
                                                 const Vectorized<c10::complex<double>>& b) {
  // 对 a 和 b 向量逐元素进行按位异或操作，返回结果
  return _mm512_xor_pd(a, b);
}
// 实现复数向量的相等比较运算符重载
inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::eq(const Vectorized<c10::complex<double>>& other) const {
  auto eq = (*this == other);  // 比较实部和虚部是否分别相等
  // 如果实部和虚部都相等，则复数相等
  return (eq.real() & eq.imag()) & Vectorized<c10::complex<double>>(_mm512_set1_pd(1.0));
}

// 实现复数向量的不等比较运算符重载
inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::ne(const Vectorized<c10::complex<double>>& other) const {
  auto ne = (*this != other);  // 比较实部和虚部是否分别不等
  // 如果实部或虚部有任意一个不相等，则复数不相等
  return (ne.real() | ne.imag()) & Vectorized<c10::complex<double>>(_mm512_set1_pd(1.0));
}
```