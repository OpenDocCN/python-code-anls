# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_complex_float.h`

```
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

// 特化模板类 Vectorized，处理 c10::complex<float> 类型
template <> class Vectorized<c10::complex<float>> {
private:
  __m512 values;  // 使用 AVX-512 指令集处理的 512 位寄存器
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};  // 零向量的静态成员

public:
  using value_type = c10::complex<float>;  // 类型别名
  using size_type = int;  // 大小类型别名，这里为整数

  // 返回向量长度为 8
  static constexpr size_type size() {
    return 8;
  }

  // 默认构造函数
  Vectorized() {}

  // 使用给定的 __m512 寄存器构造向量化对象
  Vectorized(__m512 v) : values(v) {}

  // 使用单个 c10::complex<float> 值构造向量化对象
  Vectorized(c10::complex<float> val) {
    float real_value = val.real();
    float imag_value = val.imag();
    // 使用 AVX-512 指令集创建包含多个复数值的向量
    values = _mm512_setr_ps(real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value);
  }

  // 使用多个 c10::complex<float> 值构造向量化对象
  Vectorized(c10::complex<float> val1, c10::complex<float> val2,
            c10::complex<float> val3, c10::complex<float> val4,
            c10::complex<float> val5, c10::complex<float> val6,
            c10::complex<float> val7, c10::complex<float> val8) {
    // 使用 AVX-512 指令集创建包含多个复数值的向量
    values = _mm512_setr_ps(val1.real(), val1.imag(),
                            val2.real(), val2.imag(),
                            val3.real(), val3.imag(),
                            val4.real(), val4.imag(),
                            val5.real(), val5.imag(),
                            val6.real(), val6.imag(),
                            val7.real(), val7.imag(),
                            val8.real(), val8.imag());
  }

  // 类型转换运算符，将向量化对象转换为 __m512 类型
  operator __m512() const {
    return values;
  }

  // 模板方法，根据掩码 mask 进行混合操作
  template <int64_t mask>
  static Vectorized<c10::complex<float>> blend(const Vectorized<c10::complex<float>>& a,
                                              const Vectorized<c10::complex<float>>& b) {
    // 将 c10::complex<V> 索引掩码转换为 V 索引掩码：xy -> xxyy
    static_assert(mask > -1 && mask < 256, "Unexpected mask value");
    // 编译器有望将此 switch 条件转换为跳转表
    // 返回混合后的结果向量 b
    return b;
  }

  // 根据掩码 mask 进行可变混合操作
  static Vectorized<c10::complex<float>> blendv(const Vectorized<c10::complex<float>>& a,
                                               const Vectorized<c10::complex<float>>& b,
                                               const Vectorized<c10::complex<float>>& mask) {
    // 将 c10::complex<V> 索引掩码转换为 V 索引掩码：xy -> xxyy
    auto mask_ = _mm512_unpacklo_ps(mask.values, mask.values);
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    // 使用 AVX-512 指令集比较两个 32 位整数向量，生成掩码
    auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask_), all_ones, _MM_CMPINT_EQ);
    // 使用掩码合并两个 AVX-512 单精度浮点向量
    return _mm512_mask_blend_ps(mmask, a.values, b.values);
  }
  template<typename step_t>
  // 创建一个包含复数浮点数的 Vectorized 对象，按给定的基数和步长
  static Vectorized<c10::complex<float>> arange(c10::complex<float> base = 0.,
                                               step_t step = static_cast<step_t>(1)) {
    // 返回一个 Vectorized 对象，包含从 base 开始，以复数步长递增的一系列值
    return Vectorized<c10::complex<float>>(base,
                                        base + step,
                                        base + c10::complex<float>(2)*step,
                                        base + c10::complex<float>(3)*step,
                                        base + c10::complex<float>(4)*step,
                                        base + c10::complex<float>(5)*step,
                                        base + c10::complex<float>(6)*step,
                                        base + c10::complex<float>(7)*step);
  }
  // 创建一个包含复数浮点数的 Vectorized 对象，按给定的起始向量 a 和目标向量 b 进行混合，混合数量由参数 count 决定
  static Vectorized<c10::complex<float>> set(const Vectorized<c10::complex<float>>& a,
                                            const Vectorized<c10::complex<float>>& b,
                            int64_t count = size()) {
    switch (count) {
      case 0:
        // 如果 count 为 0，返回向量 a
        return a;
      case 1:
        // 如果 count 为 1，使用模板函数 blend<1> 对 a 和 b 进行混合，并返回结果
        return blend<1>(a, b);
      case 2:
        // 如果 count 为 2，使用模板函数 blend<3> 对 a 和 b 进行混合，并返回结果
        return blend<3>(a, b);
      case 3:
        // 如果 count 为 3，使用模板函数 blend<7> 对 a 和 b 进行混合，并返回结果
        return blend<7>(a, b);
      case 4:
        // 如果 count 为 4，使用模板函数 blend<15> 对 a 和 b 进行混合，并返回结果
        return blend<15>(a, b);
      case 5:
        // 如果 count 为 5，使用模板函数 blend<31> 对 a 和 b 进行混合，并返回结果
        return blend<31>(a, b);
      case 6:
        // 如果 count 为 6，使用模板函数 blend<63> 对 a 和 b 进行混合，并返回结果
        return blend<63>(a, b);
      case 7:
        // 如果 count 为 7，使用模板函数 blend<127> 对 a 和 b 进行混合，并返回结果
        return blend<127>(a, b);
    }
    // 默认情况下，返回向量 b
    return b;
  }
  // 从给定的内存地址 ptr 加载复数浮点数，加载数量由参数 count 决定
  static Vectorized<c10::complex<float>> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      // 如果 count 等于 size()，直接加载 AVX-512 浮点数向量
      return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));

    // 否则，创建临时数组存储加载数据的复数浮点数值，并确保未初始化的内存不会影响输出值
    __at_align__ float tmp_values[2*size()];
    for (const auto i : c10::irange(2*size())) {
      // 初始化临时数组为零，避免未初始化内存的问题，参考 https://github.com/pytorch/pytorch/issues/32502
      tmp_values[i] = 0.0;
    }
    // 将 ptr 指向的数据复制到临时数组 tmp_values 中，复制长度为 count 个复数浮点数
    std::memcpy(
        tmp_values,
        reinterpret_cast<const float*>(ptr),
        count * sizeof(c10::complex<float>));
    // 返回加载的 AVX-512 浮点数向量
    return _mm512_load_ps(tmp_values);
  }
  // 将向量值存储到给定的内存地址 ptr，存储数量由参数 count 决定
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 如果 count 等于 size()，直接存储 AVX-512 浮点数向量到 ptr 指向的内存地址
      _mm512_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      // 否则，创建临时数组存储向量值，并将其复制到 ptr 指向的内存地址，复制长度为 count 个复数浮点数
      float tmp_values[2*size()];
      _mm512_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<float>));
      // 注意：未处理 count 小于 0 的情况
  // AVX512不支持水平加法和水平减法指令。
  // TODO: hadd_pd() 和 hsub_pd() 可能有优化空间。
  static inline __m512 hadd_ps(__m512 a, __m512 b) {
  // 创建索引1，用于执行 hadd_ps 操作
  __m512i idx1 = _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0);
  // 创建索引2，用于执行 hadd_ps 操作
  __m512i idx2 = _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1);
  // 执行 hadd_ps 操作：a 和 b 的对应元素相加并按照指定索引重排
  return _mm512_add_ps(_mm512_mask_permutex2var_ps(a, 0xffff, idx1, b),
                       _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b));
  }

  static inline __m512 hsub_ps(__m512 a, __m512 b) {
  // 创建索引1，用于执行 hsub_ps 操作
  __m512i idx1 = _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0);
  // 创建索引2，用于执行 hsub_ps 操作
  __m512i idx2 = _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1);
  // 执行 hsub_ps 操作：a 和 b 的对应元素相减并按照指定索引重排
  return _mm512_sub_ps(_mm512_mask_permutex2var_ps(a, 0xffff, idx1, b),
                       _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b));
  }

  // 删除操作符重载，不允许访问复数向量的指定索引
  const c10::complex<float>& operator[](int idx) const  = delete;
  // 删除操作符重载，不允许修改复数向量的指定索引
  c10::complex<float>& operator[](int idx) = delete;

  // 对复数向量中的每个元素执行映射操作
  Vectorized<c10::complex<float>> map(c10::complex<float> (*const f)(const c10::complex<float> &)) const {
    // 创建临时数组存储当前向量的复数值
    __at_align__ c10::complex<float> tmp[size()];
    // 将向量值存储到临时数组中
    store(tmp);
    // 对临时数组中的每个元素应用指定的映射函数 f
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    // 加载映射后的临时数组到向量对象中并返回
    return loadu(tmp);
  }

  // 计算向量的绝对值的平方
  __m512 abs_2_() const {
    auto val_2 = _mm512_mul_ps(values, values);     // a*a     b*b
    auto ret = hadd_ps(val_2, val_2);               // a*a+b*b a*a+b*b
    return ret;
  }

  // 计算向量的绝对值
  __m512 abs_() const {
    auto real = _mm512_moveldup_ps(values);    // real real
    auto imag = _mm512_movehdup_ps(values);    // imag imag
    return Sleef_hypotf16_u05(real, imag);     // abs  abs
  }

  // 对复数向量中的每个元素计算绝对值
  Vectorized<c10::complex<float>> abs() const {
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    // 计算复数向量的绝对值，并将虚部清零
    return _mm512_and_ps(abs_(), real_mask);        // abs     0
  }

  // 计算向量的幅角
  __m512 angle_() const {
    // angle = atan2(b/a)
    auto b_a = _mm512_permute_ps(values, 0xB1);     // b        a
    return Sleef_atan2f16_u10(values, b_a);          // 90-angle angle
  }

  // 对复数向量中的每个元素计算幅角
  Vectorized<c10::complex<float>> angle() const {
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    // 计算复数向量的幅角，并将虚部清零
    return _mm512_and_ps(angle_(), real_mask);
  }
    // 计算角度的 90 度差，使用 MM512 指令集的 permute_ps 函数
    auto angle = _mm512_permute_ps(angle_(), 0xB1); // angle    90-angle
    
    // 对 angle 和 real_mask 执行按位与操作，使用 MM512 指令集的 and_ps 函数
    return _mm512_and_ps(angle, real_mask);         // angle    0
    }
    Vectorized<c10::complex<float>> sgn() const {
    // 计算绝对值，使用成员函数 abs_()
    auto abs = abs_();
    // 创建一个全零的向量
    auto zero = _mm512_setzero_ps();
    // 比较 abs 和 zero 是否相等，并生成比较结果的掩码
    auto mask = _mm512_cmp_ps_mask(abs, zero, _CMP_EQ_OQ);
    // 将 values 向量除以 abs 向量中的每个元素
    auto div = _mm512_div_ps(values, abs);
    // 使用掩码合并 div 和 zero 向量中的元素
    return _mm512_mask_blend_ps(mask, div, zero);
    }
    __m512 real_() const {
    // 创建一个全一的掩码向量，用于提取复数向量 values 的实部部分
    const __m512 real_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    return _mm512_and_ps(values, real_mask);
    }
    Vectorized<c10::complex<float>> real() const {
    // 返回实部函数 real_()
    return real_();
    }
    __m512 imag_() const {
    // 创建一个掩码向量，用于提取复数向量 values 的虚部部分
    const __m512 imag_mask = _mm512_castsi512_ps(_mm512_setr_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF));
    return _mm512_and_ps(values, imag_mask);
    }
    Vectorized<c10::complex<float>> imag() const {
    // 对虚部函数 imag_() 返回值执行 permute_ps 函数，以获得正确顺序的虚部向量
    return _mm512_permute_ps(imag_(), 0xB1);        //b        a
    }
    __m512 conj_() const {
    // 创建一个符号掩码向量，用于执行复数向量 values 的共轭运算
    const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                            0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    // 使用 XOR 操作符执行共轭运算
    return _mm512_xor_ps(values, sign_mask);        // a       -b
    }
    Vectorized<c10::complex<float>> conj() const {
    // 返回共轭函数 conj_()
    return conj_();
    }
    Vectorized<c10::complex<float>> log() const {
    // 使用 log() 函数映射复数值以提高性能
    // 大多数三角函数操作使用 log() 函数来实现
    return map(std::log);
    }
    Vectorized<c10::complex<float>> log2() const {
    // 创建一个 log2 的常数向量
    const __m512 log2_ = _mm512_set1_ps(std::log(2));
    // 返回 log() 函数返回值与 log2_ 常数向量的除法结果
    return _mm512_div_ps(log(), log2_);
    }
    Vectorized<c10::complex<float>> log10() const {
    // 创建一个 log10 的常数向量
    const __m512 log10_ = _mm512_set1_ps(std::log(10));
    // 返回 log() 函数返回值与 log10_ 常数向量的除法结果
    return _mm512_div_ps(log(), log10_);
    }
    Vectorized<c10::complex<float>> log1p() const {
    // 使用 log1p 函数映射复数值
    return map(std::log1p);
    }
    Vectorized<c10::complex<float>> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    const __m512 one = _mm512_set1_ps(1);
    
    auto conj = conj_();
    // 对 conj 执行 permute_ps 函数，以获取正确顺序的虚部向量
    auto b_a = _mm512_permute_ps(conj, 0xB1);                         //-b        a
    auto ab = _mm512_mul_ps(conj, b_a);                               // 计算 ab = conj * b_a，这里使用 AVX-512 指令集进行向量乘法
    auto im = _mm512_add_ps(ab, ab);                                  // 计算 im = ab + ab，即 im = 2 * ab

    auto val_2 = _mm512_mul_ps(values, values);                       // 计算 val_2 = values * values，即复数值的平方
    auto re = hsub_ps(val_2, _mm512_permute_ps(val_2, 0xB1));         // 计算 re = hsub_ps(val_2, permute(val_2, 0xB1))，hsub_ps 是一个 AVX-512 指令，此处为水平减法
    re = _mm512_sub_ps(one, re);                                      // 计算 re = 1 - re

    auto root = Vectorized(_mm512_mask_blend_ps(0xAAAA, re, im)).sqrt();  // 计算根号(re + i*im)，使用 AVX-512 指令集进行混合和平方根操作
    auto ln = Vectorized(_mm512_add_ps(b_a, root)).log();             // 计算 ln(iz + sqrt())，其中 ln 是自然对数函数
    return Vectorized(_mm512_permute_ps(ln.values, 0xB1)).conj();     // 返回 -i * ln() 的复共轭值
  }
  Vectorized<c10::complex<float>> acos() const {
    return map(std::acos);                                           // 返回 acos(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> atan() const;
  Vectorized<c10::complex<float>> atanh() const {
    return map(std::atanh);                                          // 返回 atanh(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> exp() const {
    // 计算 exp(a + bi) = exp(a) * (cos(b) + sin(b)i)，其中 exp 是指数函数，cos 和 sin 是三角函数
    auto exp = Sleef_expf16_u10(values);                              // 计算 exp(values) 的向量化结果
    exp = _mm512_mask_blend_ps(0xAAAA, exp, _mm512_permute_ps(exp, 0xB1));  // 对 exp 进行混合，以得到正确的复指数值

    auto sin_cos = Sleef_sincosf16_u10(values);                       // 计算 values 的 sin 和 cos 的向量化结果
    auto cos_sin = _mm512_mask_blend_ps(0xAAAA, _mm512_permute_ps(sin_cos.y, 0xB1),
                                   sin_cos.x);                        // 对 sin_cos 进行混合，得到正确的 cos 和 sin 值
    return _mm512_mul_ps(exp, cos_sin);                               // 返回 exp(a) * (cos(b) + sin(b)i) 的结果
  }
  Vectorized<c10::complex<float>> exp2() const {
    // 使用恒等式 2**x = exp(log(2) * x) 计算 2 的 values 次幂的向量化结果
    const __m512 ln_2 = _mm512_set1_ps(c10::ln_2<float>);             // 设置 ln(2) 的 AVX-512 向量
    Vectorized<c10::complex<float>> scaled_values = _mm512_mul_ps(values, ln_2);  // 计算 values * ln(2)
    return scaled_values.exp();                                       // 返回 exp(values * ln(2))
  }
  Vectorized<c10::complex<float>> expm1() const {
    return map(std::expm1);                                           // 返回 expm1(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> sin() const {
    return map(std::sin);                                             // 返回 sin(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> sinh() const {
    return map(std::sinh);                                            // 返回 sinh(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> cos() const {
    return map(std::cos);                                             // 返回 cos(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> cosh() const {
    return map(std::cosh);                                            // 返回 cosh(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> ceil() const {
    return _mm512_ceil_ps(values);                                    // 返回 values 的向上取整的 AVX-512 向量化结果
  }
  Vectorized<c10::complex<float>> floor() const {
    return _mm512_floor_ps(values);                                   // 返回 values 的向下取整的 AVX-512 向量化结果
  }
  Vectorized<c10::complex<float>> neg() const {
    auto zero = _mm512_setzero_ps();                                  // 设置一个全零的 AVX-512 向量
    return _mm512_sub_ps(zero, values);                               // 返回零减去 values 的 AVX-512 向量化结果，即取负值
  }
  Vectorized<c10::complex<float>> round() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));  // 返回 values 的四舍五入到最近整数的 AVX-512 向量化结果
  }
  Vectorized<c10::complex<float>> tan() const {
    return map(std::tan);                                             // 返回 tan(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> tanh() const {
    return map(std::tanh);                                            // 返回 tanh(values) 的向量化结果
  }
  Vectorized<c10::complex<float>> trunc() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));  // 返回 values 的截断到零的 AVX-512 向量化结果
  }
  Vectorized<c10::complex<float>> sqrt() const {
  return map(std::sqrt);
}

// 返回一个 Vectorized 对象，其中每个元素是当前对象中每个复数的倒数
Vectorized<c10::complex<float>> reciprocal() const;

// 返回一个 Vectorized 对象，其中每个元素是当前对象中每个复数的平方根的倒数
Vectorized<c10::complex<float>> rsqrt() const {
  return sqrt().reciprocal();
}

// 返回一个 Vectorized 对象，其中每个元素是当前对象中每个复数的指数运算结果
Vectorized<c10::complex<float>> pow(const Vectorized<c10::complex<float>> &exp) const {
  // 临时存储当前对象和参数对象的复数值
  __at_align__ c10::complex<float> x_tmp[size()];
  __at_align__ c10::complex<float> y_tmp[size()];
  store(x_tmp);   // 将当前对象的复数值存储到 x_tmp 数组中
  exp.store(y_tmp);   // 将参数对象的复数值存储到 y_tmp 数组中
  // 对每个复数进行指数运算
  for (const auto i : c10::irange(size())) {
    x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
  }
  // 返回处理后的 Vectorized 对象
  return loadu(x_tmp);
}

// 返回一个 Vectorized 对象，其中每个元素是当前对象中每个复数与另一个对象中对应元素的相等性比较结果
// 使用 _CMP_EQ_OQ 谓词进行比较：
//   `O`: 如果操作数是 NaN 则返回 false
//   `Q`: 如果操作数是 NaN 则不引发异常
Vectorized<c10::complex<float>> operator==(const Vectorized<c10::complex<float>>& other) const {
  auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_EQ_OQ);
  return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF));
}

// 返回一个 Vectorized 对象，其中每个元素是当前对象中每个复数与另一个对象中对应元素的不等性比较结果
// 使用 _CMP_NEQ_UQ 谓词进行比较：
//   `U`: 如果任一操作数是 NaN 则返回 true
//   `Q`: 如果操作数是 NaN 则不引发异常
Vectorized<c10::complex<float>> operator!=(const Vectorized<c10::complex<float>>& other) const {
  auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_NEQ_UQ);
  return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF));
}

// 抛出异常，因为复数对象不支持小于操作
Vectorized<c10::complex<float>> operator<(const Vectorized<c10::complex<float>>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 抛出异常，因为复数对象不支持小于等于操作
Vectorized<c10::complex<float>> operator<=(const Vectorized<c10::complex<float>>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 抛出异常，因为复数对象不支持大于操作
Vectorized<c10::complex<float>> operator>(const Vectorized<c10::complex<float>>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 抛出异常，因为复数对象不支持大于等于操作
Vectorized<c10::complex<float>> operator>=(const Vectorized<c10::complex<float>>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 返回一个 Vectorized 对象，其中每个元素是当前对象中每个复数与另一个对象中对应元素的相等性比较结果
Vectorized<c10::complex<float>> eq(const Vectorized<c10::complex<float>>& other) const;

// 返回一个 Vectorized 对象，其中每个元素是当前对象中每个复数与另一个对象中对应元素的不等性比较结果
Vectorized<c10::complex<float>> ne(const Vectorized<c10::complex<float>>& other) const;
};

template <> Vectorized<c10::complex<float>> inline operator+(const Vectorized<c10::complex<float>> &a,
                                                            const Vectorized<c10::complex<float>> &b) {
  // 重载 + 运算符，实现复数向量的加法
  return _mm512_add_ps(a, b);
}

template <> Vectorized<c10::complex<float>> inline operator-(const Vectorized<c10::complex<float>> &a,
                                                            const Vectorized<c10::complex<float>> &b) {
  // 重载 - 运算符，实现复数向量的减法
  return _mm512_sub_ps(a, b);
}

template <> Vectorized<c10::complex<float>> inline operator*(const Vectorized<c10::complex<float>> &a,
                                                            const Vectorized<c10::complex<float>> &b) {
  // 重载 * 运算符，实现复数向量的乘法
  //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
  const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                          0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm512_mul_ps(a, b);         // 计算 ac 和 bd
  auto d_c = _mm512_permute_ps(b, 0xB1);    // 获取 b 向量的反向排列 d 和 c
  d_c = _mm512_xor_ps(sign_mask, d_c);      // 将 d 部分取反，得到 d 和 -c
  auto ad_bc = _mm512_mul_ps(a, d_c);       // 计算 ad 和 -bc
  auto ret = Vectorized<c10::complex<float>>::hsub_ps(ac_bd, ad_bc);  // 返回 ac - bd 和 ad + bc
  return ret;
}

template <> Vectorized<c10::complex<float>> inline operator/(const Vectorized<c10::complex<float>> &a,
                                                            const Vectorized<c10::complex<float>> &b) {
  // 重载 / 运算符，实现复数向量的除法
  // re + im*i = (a + bi)  / (c + di)
  auto mask = _mm512_set1_ps(-0.f);
  auto fabs_cd = _mm512_andnot_ps(mask, b);     // 取 b 向量绝对值的实部和虚部
  auto fabs_dc = _mm512_permute_ps(fabs_cd, 0xB1);   // 取 b 向量绝对值的虚部和实部
  auto scale = _mm512_rcp14_ps(_mm512_max_ps(fabs_cd, fabs_dc));  // 计算比例因子的倒数
  auto a2 = _mm512_mul_ps(a, scale);         // 对 a 向量进行缩放
  auto b2 = _mm512_mul_ps(b, scale);         // 对 b 向量进行缩放
  auto acbd2 = _mm512_mul_ps(a2, b2);        // 计算 ac 和 bd
  const __m512 sign_mask = _mm512_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0,
                                          -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
  auto dc2 = _mm512_permute_ps(b2, 0xB1);    // 获取 b 向量的反向排列 d/sc 和 c/sc
  dc2 = _mm512_xor_ps(sign_mask, dc2);       // 将 d 部分取反，得到 -d/sc 和 c/sc
  auto adbc2 = _mm512_mul_ps(a2, dc2);       // 计算 -ad/sc^2 和 bc/sc^2
  auto res2 = Vectorized<c10::complex<float>>::hadd_ps(acbd2, adbc2);  // 返回 (ac+bd)/sc^2 和 (bc-ad)/sc^2
  auto denom2 = Vectorized<c10::complex<float>>(b2).abs_2_();  // 计算分母 (c^2+d^2)/sc^2
  res2 = _mm512_div_ps(res2, denom2);         // 返回最终结果
  return res2;
}

// reciprocal. Implement this here so we can use multiplication.
// 定义复数向量化类的逆操作
inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::reciprocal() const {
  // 设置用于反转符号的掩码，将复数值的实部变为-c，虚部变为-d
  const __m512 sign_mask = _mm512_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                                          0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm512_xor_ps(sign_mask, values);    // 对复数值进行异或操作，获取其逆
  return _mm512_div_ps(c_d, abs_2_());            // 返回逆的结果，通过除以模的平方实现
}

// 定义复数向量化类的反正切操作
inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::atan() const {
  // 定义虚数单位i
  const __m512 i = _mm512_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                  0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  // 定义i/2
  const Vectorized i_half = _mm512_setr_ps(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
                                          0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);

  auto sum = Vectorized(_mm512_add_ps(i, values));  // 计算i + z
  auto sub = Vectorized(_mm512_sub_ps(i, values));  // 计算i - z
  auto ln = (sum/sub).log();                       // 计算ln((i + z)/(i - z))
  return i_half*ln;                                // 返回i/2乘以ln的结果，即atan(z)
}

// 实现复数向量化类的最大值操作
template <>
Vectorized<c10::complex<float>> inline maximum(const Vectorized<c10::complex<float>>& a,
                                              const Vectorized<c10::complex<float>>& b) {
  auto zero_vector = _mm512_set1_epi32(0);
  auto abs_a = a.abs_2_();                         // 计算向量a的模的平方
  auto abs_b = b.abs_2_();                         // 计算向量b的模的平方
  auto mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_LT_OQ);  // 比较模的平方，获取比较结果的掩码
  auto max = _mm512_mask_blend_ps(mask, a, b);      // 使用掩码选择模的平方较大的向量元素
  // 利用所有元素为NaN的事实，构建NaN掩码
  auto isnan_mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_UNORD_Q);
  auto isnan = _mm512_mask_set1_epi32(zero_vector, isnan_mask, 0xFFFFFFFF);
  return _mm512_or_ps(max, _mm512_castsi512_ps(isnan));  // 返回最大值向量，并将NaN掩码应用
}

// 实现复数向量化类的最小值操作
template <>
Vectorized<c10::complex<float>> inline minimum(const Vectorized<c10::complex<float>>& a,
                                              const Vectorized<c10::complex<float>>& b) {
  auto zero_vector = _mm512_set1_epi32(0);
  auto abs_a = a.abs_2_();                         // 计算向量a的模的平方
  auto abs_b = b.abs_2_();                         // 计算向量b的模的平方
  auto mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_GT_OQ);  // 比较模的平方，获取比较结果的掩码
  auto min = _mm512_mask_blend_ps(mask, a, b);      // 使用掩码选择模的平方较小的向量元素
  // 利用所有元素为NaN的事实，构建NaN掩码
  auto isnan_mask = _mm512_cmp_ps_mask(abs_a, abs_b, _CMP_UNORD_Q);
  auto isnan = _mm512_mask_set1_epi32(zero_vector, isnan_mask, 0xFFFFFFFF);
  return _mm512_or_ps(min, _mm512_castsi512_ps(isnan));  // 返回最小值向量，并将NaN掩码应用
}

// 实现复数向量化类的按位与操作
template <>
Vectorized<c10::complex<float>> inline operator&(const Vectorized<c10::complex<float>>& a,
                                                const Vectorized<c10::complex<float>>& b) {
  return _mm512_and_ps(a, b);  // 返回a和b按位与的结果
}
// 定义一个复数向量化运算符重载，执行逻辑或操作
Vectorized<c10::complex<float>> inline operator|(const Vectorized<c10::complex<float>>& a,
                                                const Vectorized<c10::complex<float>>& b) {
  return _mm512_or_ps(a, b);  // 使用 SIMD 指令执行复数向量的逻辑或运算
}

// 特化模板，定义复数向量化运算符重载，执行逻辑异或操作
template <>
Vectorized<c10::complex<float>> inline operator^(const Vectorized<c10::complex<float>>& a,
                                                const Vectorized<c10::complex<float>>& b) {
  return _mm512_xor_ps(a, b);  // 使用 SIMD 指令执行复数向量的逻辑异或运算
}

// 类方法，实现复数向量化对象与另一对象的相等比较
inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::eq(
    const Vectorized<c10::complex<float>>& other) const {
  auto eq = (*this == other);  // 比较实部和虚部是否分别相等
  // 如果实部和虚部均相等，则复数对象相等
  return (eq.real() & eq.imag()) & Vectorized<c10::complex<float>>(_mm512_set1_ps(1.0f));
}

// 类方法，实现复数向量化对象与另一对象的不等比较
inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::ne(
    const Vectorized<c10::complex<float>>& other) const {
  auto ne = (*this != other);  // 比较实部和虚部是否分别不等
  // 如果实部或虚部有任一不相等，则复数对象不相等
  return (ne.real() | ne.imag()) & Vectorized<c10::complex<float>>(_mm512_set1_ps(1.0f));
}
```