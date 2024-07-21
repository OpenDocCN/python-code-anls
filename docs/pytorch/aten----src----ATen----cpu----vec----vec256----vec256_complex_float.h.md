# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_complex_float.h`

```
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <c10/util/complex.h>
#include <c10/util/irange.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#if defined(CPU_CAPABILITY_AVX2)
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

// 特化模板 Vectorized<c10::complex<float>> 的定义
template <> class Vectorized<c10::complex<float>> {
private:
  // 内部成员变量，使用 AVX 指令集的 256 位寄存器来存储数据
  __m256 values;
public:
  // 类型别名
  using value_type = c10::complex<float>;
  using size_type = int;
  // 返回向量长度为 4
  static constexpr size_type size() {
    return 4;
  }
  // 默认构造函数
  Vectorized() {}
  // 构造函数，根据给定的 __m256 向量初始化
  Vectorized(__m256 v) : values(v) {}
  // 构造函数，根据给定的 c10::complex<float> 初始化向量
  Vectorized(c10::complex<float> val) {
    float real_value = val.real();
    float imag_value = val.imag();
    // 使用 _mm256_setr_ps 函数将实部和虚部填充到 AVX 向量中
    values = _mm256_setr_ps(real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value
                            );
  }
  // 构造函数，根据给定的四个 c10::complex<float> 初始化向量
  Vectorized(c10::complex<float> val1, c10::complex<float> val2, c10::complex<float> val3, c10::complex<float> val4) {
    // 使用 _mm256_setr_ps 函数将四组实部和虚部填充到 AVX 向量中
    values = _mm256_setr_ps(val1.real(), val1.imag(),
                            val2.real(), val2.imag(),
                            val3.real(), val3.imag(),
                            val4.real(), val4.imag()
                            );
  }
  // 类型转换操作符，将 Vectorized 转换为 __m256 向量
  operator __m256() const {
    return values;
  }
  // 模板函数，根据指定的掩码 mask 进行混合操作
  template <int64_t mask>
  static Vectorized<c10::complex<float>> blend(const Vectorized<c10::complex<float>>& a, const Vectorized<c10::complex<float>>& b) {
     // 将 c10::complex<V> 索引掩码转换为 V 索引掩码：xy -> xxyy
    static_assert(mask > -1 && mask < 16, "Unexpected mask range");


这段代码定义了一个特化模板 `Vectorized<c10::complex<float>>`，实现了使用 AVX2 指令集进行复数向量化操作。
    switch (mask) {
      case 0:
        return a;
      case 1:
        // 使用掩码 0x03 进行混合（blend），将 a 和 b 向量的低两个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0x03); //b0000 0001 = b0000 0011
      case 2:
        // 使用掩码 0x0C 进行混合（blend），将 a 和 b 向量的第二和第三个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0x0C); //b0000 0010 = b0000 1100
      case 3:
        // 使用掩码 0x0F 进行混合（blend），将 a 和 b 向量的所有单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0x0F); //b0000 0011 = b0000 1111
      case 4:
        // 使用掩码 0x30 进行混合（blend），将 a 和 b 向量的第四和第五个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0x30); //b0000 0100 = b0011 0000
      case 5:
        // 使用掩码 0x33 进行混合（blend），将 a 和 b 向量的前六个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0x33); //b0000 0101 = b0011 0011
      case 6:
        // 使用掩码 0x3C 进行混合（blend），将 a 和 b 向量的第六和第七个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0x3C); //b0000 0110 = b0011 1100
      case 7:
        // 使用掩码 0x3F 进行混合（blend），将 a 和 b 向量的所有单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0x3F); //b0000 0111 = b0011 1111
      case 8:
        // 使用掩码 0xC0 进行混合（blend），将 a 和 b 向量的第八和第九个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0xC0); //b0000 1000 = b1100 0000
      case 9:
        // 使用掩码 0xC3 进行混合（blend），将 a 和 b 向量的前十个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0xC3); //b0000 1001 = b1100 0011
      case 10:
        // 使用掩码 0xCC 进行混合（blend），将 a 和 b 向量的第十和第十一个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0xCC); //b0000 1010 = b1100 1100
      case 11:
        // 使用掩码 0xCF 进行混合（blend），将 a 和 b 向量的前十二个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0xCF); //b0000 1011 = b1100 1111
      case 12:
        // 使用掩码 0xF0 进行混合（blend），将 a 和 b 向量的第十二和第十三个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0xF0); //b0000 1100 = b1111 0000
      case 13:
        // 使用掩码 0xF3 进行混合（blend），将 a 和 b 向量的前十四个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0xF3); //b0000 1101 = b1111 0011
      case 14:
        // 使用掩码 0xFC 进行混合（blend），将 a 和 b 向量的第十四和第十五个单精度浮点数值进行混合
        return _mm256_blend_ps(a.values, b.values, 0xFC); //b0000 1110 = b1111 1100
      default:
        break;
    }
    // 如果 mask 超出预期范围，默认返回 b 向量
    return b;
  }
  static Vectorized<c10::complex<float>> blendv(const Vectorized<c10::complex<float>>& a, const Vectorized<c10::complex<float>>& b,
                               const Vectorized<c10::complex<float>>& mask) {
    // 将复数向量 mask 的每个值复制并展开，形成一个 V 进行混合的掩码
    auto mask_ = _mm256_unpacklo_ps(mask.values, mask.values);
    // 使用 mask_ 对 a 和 b 进行单精度浮点数值的混合
    return _mm256_blendv_ps(a.values, b.values, mask_);
  }
  template<typename step_t>
  static Vectorized<c10::complex<float>> arange(c10::complex<float> base = 0., step_t step = static_cast<step_t>(1)) {
    // 返回一个包含四个复数的向量，从 base 开始，步长为 step
    return Vectorized<c10::complex<float>>(base,
                                        base + step,
                                        base + c10::complex<float>(2)*step,
                                        base + c10::complex<float>(3)*step);
  }
  static Vectorized<c10::complex<float>> set(const Vectorized<c10::complex<float>>& a, const Vectorized<c10::complex<float>>& b,
                            int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        // 使用 blend<1> 对 a 和 b 进行混合（blend）
        return blend<1>(a, b);
      case 2:
        // 使用 blend<3> 对 a 和 b 进行混合（blend）
        return blend<3>(a, b);
      case 3:
        // 使用 blend<7> 对 a 和 b 进行混合（blend）
        return blend<7>(a, b);
    }
    // 如果 count 超出预期范围，默认返回 b
    return b;
  }
  static Vectorized<c10::complex<float>> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      // 加载未对齐的单精度浮点数值，转换为复数向量
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));

    // 如果 count 不等于预期的大小，则分配一个临时的 float 数组
    __at_align__ float tmp_values[2*size()];
    // 确保未初始化的内存不会改变输出值，参考 https://github.com/pytorch/pytorch/issues/32502 获取更多细节。
    // 不使用"={0}"初始化数组到零，因为gcc会将其编译成两条指令，而循环只会编译成一条指令。
    for (const auto i : c10::irange(2*size())) {
      // 将临时数组的每个元素初始化为0.0
      tmp_values[i] = 0.0;
    }
    // 将从ptr指向的内存解释为float型的指针，并复制count个c10::complex<float>的数据到tmp_values中
    std::memcpy(
        tmp_values,
        reinterpret_cast<const float*>(ptr),
        count * sizeof(c10::complex<float>));
    // 返回tmp_values加载的结果作为__m256类型的数据
    return _mm256_load_ps(tmp_values);
  }
  // 将数据存储到ptr指向的内存中，count表示要存储的元素数量，默认为size()
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 如果count等于size()，直接存储values到ptr中
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      // 否则，创建临时数组tmp_values来存储values中的数据，并将其复制到ptr指向的内存中
      float tmp_values[2*size()];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<float>));
    }
  }
  // 删除索引操作符，禁止对该类型进行索引访问
  const c10::complex<float>& operator[](int idx) const  = delete;
  c10::complex<float>& operator[](int idx) = delete;
  // 对每个元素应用函数f，并返回Vectorized<c10::complex<float>>类型的结果
  Vectorized<c10::complex<float>> map(c10::complex<float> (*const f)(const c10::complex<float> &)) const {
    // 创建临时数组tmp来存储当前对象的数据
    __at_align__ c10::complex<float> tmp[size()];
    store(tmp);
    // 对tmp中的每个元素应用函数f，并返回加载tmp的Vectorized对象
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  // 计算每个元素的模的平方，并返回结果作为__m256类型的数据
  __m256 abs_2_() const {
    auto val_2 = _mm256_mul_ps(values, values);     // 计算实部和虚部的平方
    auto ret = _mm256_hadd_ps(val_2, val_2);        // 水平加法求和
    return _mm256_permute_ps(ret, 0xD8);            // 重新排列结果
  }
  // 计算每个元素的模，并返回结果作为__m256类型的数据
  __m256 abs_() const {
    auto real = _mm256_moveldup_ps(values);   // 提取实部
    auto imag = _mm256_movehdup_ps(values);   // 提取虚部
    return Sleef_hypotf8_u05(real, imag);     // 计算模的结果
  }
  // 返回Vectorized对象的每个元素的模，实部保留，虚部置零
  Vectorized<c10::complex<float>> abs() const {
    const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    return _mm256_and_ps(abs_(), real_mask);        // 返回每个元素的模，虚部为0
  }
  // 计算每个元素的幅角，并返回结果作为__m256类型的数据
  __m256 angle_() const {
    // 计算每个元素的幅角，即atan2(values.imag, values.real)
    auto b_a = _mm256_permute_ps(values, 0xB1);     // 提取虚部和实部
    return Sleef_atan2f8_u10(values, b_a);          // 计算幅角
  }
  // 返回Vectorized对象的每个元素的幅角，虚部保留，实部置零
  Vectorized<c10::complex<float>> angle() const {
    const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    auto angle = _mm256_permute_ps(angle_(), 0xB1); // 提取每个元素的幅角
    return _mm256_and_ps(angle, real_mask);         // 返回每个元素的幅角，实部为0
  }
  // 返回Vectorized对象的每个元素的符号函数，即values / abs(values)，并处理分母为零的情况
  Vectorized<c10::complex<float>> sgn() const {
    auto abs = abs_();
    auto zero = _mm256_setzero_ps();
    auto mask = _mm256_cmp_ps(abs, zero, _CMP_EQ_OQ); // 检查abs是否为零
    auto div = _mm256_div_ps(values, abs);            // 计算每个元素的符号函数
    return _mm256_blendv_ps(div, zero, mask);         // 分母为零时返回零向量，否则返回符号函数结果
  }
  // 返回Vectorized对象的每个元素的实部，虚部置零
  __m256 real_() const {
  // 创建一个实部掩码，用于从复数向量中提取实部
  const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
  // 使用实部掩码对复数向量进行按位与运算，提取实部
  return _mm256_and_ps(values, real_mask);
}

Vectorized<c10::complex<float>> real() const {
  // 调用实部提取函数
  return real_();
}

// 创建一个虚部掩码，用于从复数向量中提取虚部
__m256 imag_() const {
  const __m256 imag_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF));
  // 使用虚部掩码对复数向量进行按位与运算，提取虚部
  return _mm256_and_ps(values, imag_mask);
}

Vectorized<c10::complex<float>> imag() const {
  // 返回虚部向量，通过置换操作使得虚部部分在正确的位置
  return _mm256_permute_ps(imag_(), 0xB1);        //b        a
}

// 创建一个共轭掩码，用于计算复数的共轭
__m256 conj_() const {
  // 创建一个符号掩码，用于取反虚部
  const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  // 对复数向量进行按位异或操作，得到其共轭
  return _mm256_xor_ps(values, sign_mask);        // a       -b
}

Vectorized<c10::complex<float>> conj() const {
  // 返回复数向量的共轭
  return conj_();
}

Vectorized<c10::complex<float>> log() const {
  // 对复数向量的每个元素取对数，用于提高复数运算性能
  // 返回对数后的复数向量
  return map(std::log);
}

Vectorized<c10::complex<float>> log2() const {
  const __m256 log2_ = _mm256_set1_ps(std::log(2));
  // 对复数向量中每个元素取自然对数后，再除以以2为底的对数
  return _mm256_div_ps(log(), log2_);
}

Vectorized<c10::complex<float>> log10() const {
  const __m256 log10_ = _mm256_set1_ps(std::log(10));
  // 对复数向量中每个元素取自然对数后，再除以以10为底的对数
  return _mm256_div_ps(log(), log10_);
}

Vectorized<c10::complex<float>> log1p() const {
  // 对复数向量中每个元素取log1p函数的结果
  return map(std::log1p);
}

Vectorized<c10::complex<float>> asin() const {
  // 计算复数向量中每个元素的反正弦值
  // 根据公式：asin(z) = -i * ln(iz + sqrt(1 - z^2))
  const __m256 one = _mm256_set1_ps(1);

  auto conj = conj_();
  auto b_a = _mm256_permute_ps(conj, 0xB1);                         //-b        a
  auto ab = _mm256_mul_ps(conj, b_a);                               //-ab       -ab
  auto im = _mm256_add_ps(ab, ab);                                  //-2ab      -2ab

  auto val_2 = _mm256_mul_ps(values, values);                       // a*a      b*b
  auto re = _mm256_hsub_ps(val_2, _mm256_permute_ps(val_2, 0xB1));  // a*a-b*b  b*b-a*a
  re = _mm256_permute_ps(re, 0xD8);
  re = _mm256_sub_ps(one, re);

  auto root = Vectorized(_mm256_blend_ps(re, im, 0xAA)).sqrt();         //sqrt(re + i*im)
  auto ln = Vectorized(_mm256_add_ps(b_a, root)).log();                 //ln(iz + sqrt())
  return Vectorized(_mm256_permute_ps(ln.values, 0xB1)).conj();         //-i*ln()
}

Vectorized<c10::complex<float>> acos() const {
  // 计算复数向量中每个元素的反余弦值
  return map(std::acos);
}

Vectorized<c10::complex<float>> atan() const;
// 计算复数向量中每个元素的反正切值
Vectorized<c10::complex<float>> atanh() const {
  return map(std::atanh);
}

Vectorized<c10::complex<float>> exp() const {
  // 计算复数向量中每个元素的指数值
  // exp(a + bi) = exp(a) * (cos(b) + sin(b)i)
    auto exp = Sleef_expf8_u10(values);                               // 计算values中每个元素的指数函数exp(a), exp(b)
    exp = _mm256_blend_ps(exp, _mm256_permute_ps(exp, 0xB1), 0xAA);   // 使用融合和置换操作，保留exp(a)的部分结果，得到exp(a), exp(b)

    auto sin_cos = Sleef_sincosf8_u10(values);                        // 计算values中每个元素的正弦和余弦函数[sin(a), cos(a)], [sin(b), cos(b)]
    auto cos_sin = _mm256_blend_ps(_mm256_permute_ps(sin_cos.y, 0xB1),
                                   sin_cos.x, 0xAA);                  // 从sincos函数的结果中提取cos(b)和sin(b)

    return _mm256_mul_ps(exp, cos_sin);                                // 返回exp(a) * cos(b), exp(b) * sin(b)的乘积
  }
  Vectorized<c10::complex<float>> exp2() const {
    // 使用恒等式 2**x = exp(log(2) * x) 来计算2的values次方
    const __m256 ln_2 = _mm256_set1_ps(c10::ln_2<float>);
    Vectorized<c10::complex<float>> scaled_values = _mm256_mul_ps(values, ln_2);
    return scaled_values.exp();                                       // 返回2的values次方的复数形式的向量化结果
  }
  Vectorized<c10::complex<float>> expm1() const {
    return map(std::expm1);                                           // 对values中的每个元素应用expm1函数，并返回结果向量
  }
  Vectorized<c10::complex<float>> sin() const {
    return map(std::sin);                                             // 对values中的每个元素应用sin函数，并返回结果向量
  }
  Vectorized<c10::complex<float>> sinh() const {
    return map(std::sinh);                                            // 对values中的每个元素应用sinh函数，并返回结果向量
  }
  Vectorized<c10::complex<float>> cos() const {
    return map(std::cos);                                             // 对values中的每个元素应用cos函数，并返回结果向量
  }
  Vectorized<c10::complex<float>> cosh() const {
    return map(std::cosh);                                            // 对values中的每个元素应用cosh函数，并返回结果向量
  }
  Vectorized<c10::complex<float>> ceil() const {
    return _mm256_ceil_ps(values);                                     // 对values中的每个元素执行向上取整操作，并返回结果向量
  }
  Vectorized<c10::complex<float>> floor() const {
    return _mm256_floor_ps(values);                                    // 对values中的每个元素执行向下取整操作，并返回结果向量
  }
  Vectorized<c10::complex<float>> neg() const {
    auto zero = _mm256_setzero_ps();                                   // 创建一个全零的向量
    return _mm256_sub_ps(zero, values);                                // 返回零向量减去values向量的结果，即取负操作
  }
  Vectorized<c10::complex<float>> round() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));  // 对values中的每个元素执行四舍五入操作，并返回结果向量
  }
  Vectorized<c10::complex<float>> tan() const {
    return map(std::tan);                                             // 对values中的每个元素应用tan函数，并返回结果向量
  }
  Vectorized<c10::complex<float>> tanh() const {
    return map(std::tanh);                                            // 对values中的每个元素应用tanh函数，并返回结果向量
  }
  Vectorized<c10::complex<float>> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));  // 对values中的每个元素执行截断操作，并返回结果向量
  }
  Vectorized<c10::complex<float>> sqrt() const {
    return map(std::sqrt);                                            // 对values中的每个元素应用sqrt函数，并返回结果向量
  }
  Vectorized<c10::complex<float>> reciprocal() const;
  Vectorized<c10::complex<float>> rsqrt() const {
    return sqrt().reciprocal();                                       // 计算每个元素的平方根后取倒数，并返回结果向量
  }
  Vectorized<c10::complex<float>> pow(const Vectorized<c10::complex<float>> &exp) const {
    __at_align__ c10::complex<float> x_tmp[size()];
    __at_align__ c10::complex<float> y_tmp[size()];
    store(x_tmp);                                                     // 将当前对象的值存储到临时数组x_tmp中
    exp.store(y_tmp);                                                 // 将传入的exp对象的值存储到临时数组y_tmp中
    for (const auto i : c10::irange(size())) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);                        // 对x_tmp和y_tmp中对应位置的元素执行幂运算
    }
    return loadu(x_tmp);                                              // 从临时数组x_tmp加载数据并返回作为向量化结果
  }
  // 使用_CMP_**_OQ谓词进行比较
  //   `O`: 如果操作数为NaN则返回false
  //   `Q`: 如果操作数为NaN则不引发异常
  Vectorized<c10::complex<float>> operator==(const Vectorized<c10::complex<float>>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);           // 比较两个向量values和other.values中的每个元素是否相等，并返回比较结果向量
  }
  Vectorized<c10::complex<float>> operator!=(const Vectorized<c10::complex<float>>& other) const {
    // 使用 AVX 指令集比较两个向量 values 和 other.values 的不相等性，返回比较结果向量
    return _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ);
    }
    
    // 当前操作不支持复数类型，抛出错误信息
    Vectorized<c10::complex<float>> operator<(const Vectorized<c10::complex<float>>& /*other*/) const {
      TORCH_CHECK(false, "not supported for complex numbers");
    }
    
    // 当前操作不支持复数类型，抛出错误信息
    Vectorized<c10::complex<float>> operator<=(const Vectorized<c10::complex<float>>& /*other*/) const {
      TORCH_CHECK(false, "not supported for complex numbers");
    }
    
    // 当前操作不支持复数类型，抛出错误信息
    Vectorized<c10::complex<float>> operator>(const Vectorized<c10::complex<float>>& /*other*/) const {
      TORCH_CHECK(false, "not supported for complex numbers");
    }
    
    // 当前操作不支持复数类型，抛出错误信息
    Vectorized<c10::complex<float>> operator>=(const Vectorized<c10::complex<float>>& /*other*/) const {
      TORCH_CHECK(false, "not supported for complex numbers");
    }
    
    // 比较当前复数向量与另一个复数向量 other 的相等性，返回比较结果向量
    Vectorized<c10::complex<float>> eq(const Vectorized<c10::complex<float>>& other) const;
    
    // 比较当前复数向量与另一个复数向量 other 的不相等性，返回比较结果向量
    Vectorized<c10::complex<float>> ne(const Vectorized<c10::complex<float>>& other) const;
// 重载模板特化，定义复数向量的加法运算符重载
template <> Vectorized<c10::complex<float>> inline operator+(const Vectorized<c10::complex<float>> &a, const Vectorized<c10::complex<float>> &b) {
  return _mm256_add_ps(a, b);  // 使用 AVX 指令集执行复数向量的加法
}

// 重载模板特化，定义复数向量的减法运算符重载
template <> Vectorized<c10::complex<float>> inline operator-(const Vectorized<c10::complex<float>> &a, const Vectorized<c10::complex<float>> &b) {
  return _mm256_sub_ps(a, b);  // 使用 AVX 指令集执行复数向量的减法
}

// 重载模板特化，定义复数向量的乘法运算符重载
template <> Vectorized<c10::complex<float>> inline operator*(const Vectorized<c10::complex<float>> &a, const Vectorized<c10::complex<float>> &b) {
  // 复数乘法公式：(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
  const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm256_mul_ps(a, b);         // 计算 ac 和 bd
  auto d_c = _mm256_permute_ps(b, 0xB1);    // 交换 b 向量的高低位，得到 d 和 c
  d_c = _mm256_xor_ps(sign_mask, d_c);      // 对 d_c 向量取反，变为 d 和 -c
  auto ad_bc = _mm256_mul_ps(a, d_c);       // 计算 ad 和 -bc
  auto ret = _mm256_hsub_ps(ac_bd, ad_bc);  // 水平减半求和 ac-bd 和 ad+bc
  ret = _mm256_permute_ps(ret, 0xD8);       // 调整结果向量顺序
  return ret;                               // 返回复数向量乘法结果
}

// 重载模板特化，定义复数向量的除法运算符重载
template <> Vectorized<c10::complex<float>> inline operator/(const Vectorized<c10::complex<float>> &a, const Vectorized<c10::complex<float>> &b) {
  // 复数除法公式：(a + bi) / (c + di) = ((ac + bd) / (c^2 + d^2)) + ((bc - ad) / (c^2 + d^2))i
  auto mask = _mm256_set1_ps(-0.f);                            // 创建掩码，用于处理复数向量的符号
  auto fabs_cd = _mm256_andnot_ps(mask, b);                    // 取 b 向量绝对值
  auto fabs_dc = _mm256_permute_ps(fabs_cd, 0xB1);             // 交换 fabs_cd 向量的高低位，得到 |d| 和 |c|
  auto scale = _mm256_rcp_ps(_mm256_max_ps(fabs_cd, fabs_dc)); // 计算除法的缩放因子
  auto a2 = _mm256_mul_ps(a, scale);                           // 对 a 向量进行缩放
  auto b2 = _mm256_mul_ps(b, scale);                           // 对 b 向量进行缩放
  auto acbd2 = _mm256_mul_ps(a2, b2);                          // 计算 a2 和 b2 的乘积

  const __m256 sign_mask = _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
  auto dc2 = _mm256_permute_ps(b2, 0xB1);                       // 交换 b2 向量的高低位，得到 d/sc 和 c/sc
  dc2 = _mm256_xor_ps(sign_mask, dc2);                          // 对 dc2 向量取反
  auto adbc2 = _mm256_mul_ps(a2, dc2);                          // 计算 -ad/sc^2 和 bc/sc^2
  auto res2 = _mm256_hadd_ps(acbd2, adbc2);                     // 水平加半求和 ac+bd 和 bc-ad
  res2 = _mm256_permute_ps(res2, 0xD8);                         // 调整结果向量顺序

  auto denom2 = Vectorized<c10::complex<float>>(b2).abs_2_();   // 计算分母的平方
  res2 = _mm256_div_ps(res2, denom2);                           // 对结果向量进行除法运算
  return res2;                                                  // 返回复数向量除法结果
}

// 实现复数向量的倒数函数，用于复数向量的乘法运算
inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::reciprocal() const {
  const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm256_xor_ps(sign_mask, values);   // 对 values 向量取反，得到复数向量的负部分
  return _mm256_div_ps(c_d, abs_2_());           // 返回复数向量的倒数
}
inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::atan() const {
  // atan(x) = i/2 * ln((i + z)/(i - z))
  // 定义复数i为向量(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
  const __m256 i = _mm256_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  // 定义复数i_half为向量(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)，表示i/2
  const Vectorized i_half = _mm256_setr_ps(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);

  // 计算sum = i + values
  auto sum = Vectorized(_mm256_add_ps(i, values));                      // a        1+b
  // 计算sub = i - values
  auto sub = Vectorized(_mm256_sub_ps(i, values));                      // -a       1-b
  // 计算ln = ln((i + z)/(i - z))
  auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
  // 返回i_half乘以ln的结果，即i/2 * ln()
  return i_half*ln;                                                 // i/2*ln()
}

template <>
Vectorized<c10::complex<float>> inline maximum(const Vectorized<c10::complex<float>>& a, const Vectorized<c10::complex<float>>& b) {
  // 计算a和b的模平方
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // 比较abs_a和abs_b的大小，生成掩码
  auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_LT_OQ);
  // 根据掩码选择较大的值
  auto max = _mm256_blendv_ps(a, b, mask);
  // 利用所有位为NaN的特性
  auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // 返回max和isnan的按位或结果
  return _mm256_or_ps(max, isnan);
}

template <>
Vectorized<c10::complex<float>> inline minimum(const Vectorized<c10::complex<float>>& a, const Vectorized<c10::complex<float>>& b) {
  // 计算a和b的模平方
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // 比较abs_a和abs_b的大小，生成掩码
  auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_GT_OQ);
  // 根据掩码选择较小的值
  auto min = _mm256_blendv_ps(a, b, mask);
  // 利用所有位为NaN的特性
  auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // 返回min和isnan的按位或结果
  return _mm256_or_ps(min, isnan);
}

template <>
Vectorized<c10::complex<float>> inline operator&(const Vectorized<c10::complex<float>>& a, const Vectorized<c10::complex<float>>& b) {
  // 返回a和b的按位与操作结果
  return _mm256_and_ps(a, b);
}

template <>
Vectorized<c10::complex<float>> inline operator|(const Vectorized<c10::complex<float>>& a, const Vectorized<c10::complex<float>>& b) {
  // 返回a和b的按位或操作结果
  return _mm256_or_ps(a, b);
}

template <>
Vectorized<c10::complex<float>> inline operator^(const Vectorized<c10::complex<float>>& a, const Vectorized<c10::complex<float>>& b) {
  // 返回a和b的按位异或操作结果
  return _mm256_xor_ps(a, b);
}

inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::eq(
    const Vectorized<c10::complex<float>>& other) const {
  // 比较实部和虚部是否分别相等
  auto eq = (*this == other);  // compares real and imag individually
  // 如果实部和虚部均相等，则复数相等
  return (eq.real() & eq.imag()) & Vectorized<c10::complex<float>>(_mm256_set1_ps(1.0f));
}

inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::ne(
    const Vectorized<c10::complex<float>>& other) const {
  // 比较实部和虚部是否分别不相等
  auto ne = (*this != other);  // compares real and imag individually
  // 如果实部和虚部有任一不相等，则复数不相等
  return (ne.real() | ne.imag()) & Vectorized<c10::complex<float>>(_mm256_set1_ps(1.0f));
}
```