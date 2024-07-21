# `.\pytorch\aten\src\ATen\cpu\vec\vec512\vec512_int.h`

```
#pragma once
// 防止头文件中定义静态数据！
// 参见注释 [不要使用 AVX 编译初始化器]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX512

// 定义 AVX512 指令集下的 Vectorizedi 结构体
struct Vectorizedi {
protected:
  // 内部变量，存储 __m512i 类型的数据
  __m512i values;
  // 零向量的静态常量定义
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
  // 反转向量的静态方法定义
  static inline __m512i invert(const __m512i& v) {
    const auto ones = _mm512_set1_epi64(-1);  // 创建所有位为1的向量
    return _mm512_xor_si512(ones, v);  // 使用 XOR 实现反转
  }
public:
  Vectorizedi() {}  // 默认构造函数
  Vectorizedi(__m512i v) : values(v) {}  // 初始化构造函数
  operator __m512i() const {  // 类型转换运算符，将对象转为 __m512i 类型
    return values;
  }
};

#else

// 如果不支持 AVX512，定义一个空的 Vectorizedi 结构体
struct Vectorizedi {};  // 用于定义 Vectorizedi，保证其总是被定义

#endif // CPU_CAPABILITY_AVX512

#ifdef CPU_CAPABILITY_AVX512

// 特化模板类 Vectorized<int64_t>，继承自 Vectorizedi
template <>
class Vectorized<int64_t> : public Vectorizedi {
private:
  static const Vectorized<int64_t> ones;  // 静态常量 ones 的声明
public:
  using value_type = int64_t;  // 定义值类型为 int64_t
  using size_type = int;  // 定义大小类型为 int
  static constexpr size_type size() {  // 返回固定大小 8
    return 8;
  }
  using Vectorizedi::Vectorizedi;  // 继承基类的构造函数
  Vectorized() {}  // 默认构造函数
  Vectorized(int64_t v) { values = _mm512_set1_epi64(v); }  // 根据给定值创建向量
  Vectorized(int64_t val1, int64_t val2, int64_t val3, int64_t val4,
         int64_t val5, int64_t val6, int64_t val7, int64_t val8) {
    values = _mm512_setr_epi64(val1, val2, val3, val4,
                                val5, val6, val7, val8);  // 使用给定值初始化向量
  }
  template <int64_t mask>
  static Vectorized<int64_t> blend(Vectorized<int64_t> a, Vectorized<int64_t> b) {
    return _mm512_mask_blend_epi64(mask, a.values, b.values);  // 按掩码融合两个向量
  }
  static Vectorized<int64_t> blendv(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b,
                                const Vectorized<int64_t>& mask) {
    auto msb_one = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);  // 创建所有位为1的向量
    auto mask_ = _mm512_cmp_epi64_mask(mask, msb_one, _MM_CMPINT_EQ);  // 比较生成掩码
    return _mm512_mask_blend_epi64(mask_, a.values, b.values);  // 按掩码融合两个向量
  }
  template <typename step_t>
  static Vectorized<int64_t> arange(int64_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(base,            base + step,     base + 2 * step, base + 3 * step,
                           base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
    // 返回基于步长的连续向量
  }
  static Vectorized<int64_t>
  set(Vectorized<int64_t> a, Vectorized<int64_t> b, int64_t count = size()) {
    switch (count) {
      case 0:
        return a;  // 返回第一个向量
      case 1:
        return blend<1>(a, b);  // 按掩码融合两个向量
      case 2:
        return blend<3>(a, b);  // 按掩码融合两个向量
      case 3:
        return blend<7>(a, b);  // 按掩码融合两个向量
      case 4:
        return blend<15>(a, b);  // 按掩码融合两个向量
      case 5:
        return blend<31>(a, b);  // 按掩码融合两个向量
      case 6:
        return blend<63>(a, b);  // 按掩码融合两个向量
      case 7:
        return blend<127>(a, b);  // 按掩码融合两个向量
    }
    return b;  // 返回第二个向量
  }
  static Vectorized<int64_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  // 加载未对齐的 512 位整数向量数据，转换给定指针 ptr 所指向的数据
  static Vectorized<int64_t> loadu(const void* ptr, int64_t count) {
    if (count == size()) {
      // 如果 count 等于向量大小，则直接加载整个向量数据
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    } else {
      // 否则根据 count 创建掩码 mask，并加载部分数据
      __mmask8 mask = (1ULL << count) - 1;
      return _mm512_maskz_loadu_epi64(mask, ptr);
    }
  }
  // 存储向量数据到指定的内存位置 ptr，可选存储数量 count 默认为向量大小
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 如果 count 等于向量大小，使用 _mm512_storeu_si512 存储未对齐的 512 位整数向量数据
      // 此处不需要对 ptr 进行对齐。参考链接：https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      // 如果 count 大于 0，则根据 count 创建掩码 mask，使用 _mm512_mask_storeu_epi64 存储部分数据
      __mmask8 mask = (1ULL << count) - 1;
      _mm512_mask_storeu_epi64(ptr, mask, values);
    }
  }
  // 禁用下标运算符 []，不允许直接访问向量中的元素
  const int64_t& operator[](int idx) const  = delete;
  // 禁用下标运算符 []，不允许直接访问向量中的元素
  int64_t& operator[](int idx)  = delete;
  // 计算向量中各元素的绝对值并返回新向量
  Vectorized<int64_t> abs() const {
    auto is_larger_mask = _mm512_cmpgt_epi64_mask(zero_vector, values);
    auto is_larger = _mm512_mask_set1_epi64(zero_vector, is_larger_mask, 0xFFFFFFFFFFFFFFFF);
    auto inverse = _mm512_xor_si512(values, is_larger);
    return _mm512_sub_epi64(inverse, is_larger);
  }
  // 返回当前向量，即实部
  Vectorized<int64_t> real() const {
    return *this;
  }
  // 返回一个零填充的向量，表示虚部
  Vectorized<int64_t> imag() const {
    return _mm512_set1_epi64(0);
  }
  // 返回当前向量的共轭，即不改变任何值
  Vectorized<int64_t> conj() const {
    return *this;
  }
  // 返回当前向量的负值
  Vectorized<int64_t> neg() const;
  // 比较两个向量是否相等，返回相等位置的掩码
  Vectorized<int64_t> operator==(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmpeq_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  // 比较两个向量是否不相等，返回不相等位置的掩码
  Vectorized<int64_t> operator!=(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmpneq_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  // 比较当前向量是否小于另一个向量，返回小于位置的掩码
  Vectorized<int64_t> operator<(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmplt_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  // 比较当前向量是否小于等于另一个向量，返回小于等于位置的掩码
  Vectorized<int64_t> operator<=(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmple_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  // 比较当前向量是否大于另一个向量，返回大于位置的掩码
  Vectorized<int64_t> operator>(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmpgt_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  // 比较当前向量是否大于等于另一个向量，返回大于等于位置的掩码
  Vectorized<int64_t> operator>=(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmpge_epi64_mask(values, other.values);
    # 使用 SIMD 指令 `_mm512_mask_set1_epi64` 将指定条件下的元素设置为全 1 的 64 位整数向量
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }

  # 返回一个新的向量对象，包含当前向量和另一个向量按元素比较后的相等结果
  Vectorized<int64_t> eq(const Vectorized<int64_t>& other) const;

  # 返回一个新的向量对象，包含当前向量和另一个向量按元素比较后的不等结果
  Vectorized<int64_t> ne(const Vectorized<int64_t>& other) const;

  # 返回一个新的向量对象，包含当前向量和另一个向量按元素比较后的大于结果
  Vectorized<int64_t> gt(const Vectorized<int64_t>& other) const;

  # 返回一个新的向量对象，包含当前向量和另一个向量按元素比较后的大于等于结果
  Vectorized<int64_t> ge(const Vectorized<int64_t>& other) const;

  # 返回一个新的向量对象，包含当前向量和另一个向量按元素比较后的小于结果
  Vectorized<int64_t> lt(const Vectorized<int64_t>& other) const;

  # 返回一个新的向量对象，包含当前向量和另一个向量按元素比较后的小于等于结果
  Vectorized<int64_t> le(const Vectorized<int64_t>& other) const;
};
// 结束了模板类 Vectorized 的定义

template <>
class Vectorized<int32_t> : public Vectorizedi {
private:
  // 静态成员变量，定义了一个全零的 __m512i 向量
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
  // 静态成员变量，定义了一个全一的 Vectorized<int32_t> 向量
  static const Vectorized<int32_t> ones;
public:
  // 声明类型别名 value_type 为 int32_t
  using value_type = int32_t;
  // 静态方法，返回向量大小为 16
  static constexpr int size() {
    return 16;
  }
  // 使用基类 Vectorizedi 的构造函数
  using Vectorizedi::Vectorizedi;
  // 默认构造函数
  Vectorized() {}
  // 根据指定的 int32_t 值构造向量
  Vectorized(int32_t v) { values = _mm512_set1_epi32(v); }
  // 根据16个指定的 int32_t 值构造向量
  Vectorized(int32_t val1, int32_t val2, int32_t val3, int32_t val4,
            int32_t val5, int32_t val6, int32_t val7, int32_t val8,
            int32_t val9, int32_t val10, int32_t val11, int32_t val12,
            int32_t val13, int32_t val14, int32_t val15, int32_t val16) {
    values = _mm512_setr_epi32(val1, val2, val3, val4, val5, val6, val7, val8,
                               val9, val10, val11, val12, val13, val14, val15, val16);
  }
  // 模板方法，根据 mask 混合两个向量 a 和 b
  template <int64_t mask>
  static Vectorized<int32_t> blend(Vectorized<int32_t> a, Vectorized<int32_t> b) {
    return _mm512_mask_blend_epi32(mask, a.values, b.values);
  }
  // 混合向量 a 和 b，根据 mask 向量的每个位的值来决定混合的方式
  static Vectorized<int32_t> blendv(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b,
                                const Vectorized<int32_t>& mask) {
    // 创建一个全 1 的掩码向量
    auto msb_one = _mm512_set1_epi32(0xFFFFFFFF);
    // 使用 mask 向量和全 1 向量比较，生成混合掩码
    auto mask_ = _mm512_cmp_epi32_mask(mask, msb_one, _MM_CMPINT_EQ);
    // 根据混合掩码混合向量 a 和 b
    return _mm512_mask_blend_epi32(mask_, a.values, b.values);
  }
  // 模板方法，返回一个从 base 开始，以 step 为步长的向量
  template <typename step_t>
  static Vectorized<int32_t> arange(int32_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  // 静态方法，根据 count 的值选择适当的混合操作来设置向量的值
  static Vectorized<int32_t>
  set(Vectorized<int32_t> a, Vectorized<int32_t> b, int32_t count = size()) {
    switch (count) {
      // 根据 count 的不同值，选择不同的混合操作
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
    return b;
  }
  // 静态方法，从给定的内存地址加载向量数据
  static Vectorized<int32_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  // 静态方法，从给定的内存地址加载指定数量的向量数据
  static Vectorized<int32_t> loadu(const void* ptr, int32_t count) {
    if (count == size()) {
      // 如果 count 等于当前向量大小，则直接加载整个向量数据
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    } else {
      // 否则，根据 count 创建掩码，加载部分向量数据
      __mmask16 mask = (1ULL << count) - 1;
      return _mm512_maskz_loadu_epi32(mask, ptr);
    }
  }
  
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 当 count 等于当前向量大小时，存储整个向量数据到 ptr 指向的内存
      // 这里的 ptr 不需要对齐，参考链接：https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      // 当 count 大于 0 时，创建掩码，存储部分向量数据到 ptr 指向的内存
      __mmask16 mask = (1ULL << count) - 1;
      _mm512_mask_storeu_epi32(ptr, mask, values);
    }
  }
  
  const int32_t& operator[](int idx) const  = delete;
  int32_t& operator[](int idx)  = delete;
  
  Vectorized<int32_t> abs() const {
    // 返回当前向量数据的绝对值
    return _mm512_abs_epi32(values);
  }
  
  Vectorized<int32_t> real() const {
    // 返回当前向量的实部，即返回当前向量本身
    return *this;
  }
  
  Vectorized<int32_t> imag() const {
    // 返回当前向量的虚部，即返回全零的向量
    return _mm512_set1_epi32(0);
  }
  
  Vectorized<int32_t> conj() const {
    // 返回当前向量的共轭，即返回当前向量本身
    return *this;
  }
  
  Vectorized<int32_t> neg() const;
  
  Vectorized<int32_t> operator==(const Vectorized<int32_t>& other) const {
    // 比较当前向量和另一个向量的元素是否相等，返回结果为掩码
    auto mask = _mm512_cmpeq_epi32_mask(values, other.values);
    // 将掩码的真值部分设为全 1，作为返回向量的数据
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  
  Vectorized<int32_t> operator!=(const Vectorized<int32_t>& other) const {
    // 比较当前向量和另一个向量的元素是否不相等，返回结果为掩码
    auto mask = _mm512_cmpneq_epi32_mask(values, other.values);
    // 将掩码的真值部分设为全 1，作为返回向量的数据
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  
  Vectorized<int32_t> operator<(const Vectorized<int32_t>& other) const {
    // 比较当前向量和另一个向量的元素是否小于，返回结果为掩码
    auto mask = _mm512_cmplt_epi32_mask(values, other.values);
    // 将掩码的真值部分设为全 1，作为返回向量的数据
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  
  Vectorized<int32_t> operator<=(const Vectorized<int32_t>& other) const {
    // 比较当前向量和另一个向量的元素是否小于等于，返回结果为掩码
    auto mask = _mm512_cmple_epi32_mask(values, other.values);
    // 将掩码的真值部分设为全 1，作为返回向量的数据
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  
  Vectorized<int32_t> operator>(const Vectorized<int32_t>& other) const {
    // 比较当前向量和另一个向量的元素是否大于，返回结果为掩码
    auto mask = _mm512_cmpgt_epi32_mask(values, other.values);
    // 将掩码的真值部分设为全 1，作为返回向量的数据
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  
  Vectorized<int32_t> operator>=(const Vectorized<int32_t>& other) const {
    // 比较当前向量和另一个向量的元素是否大于等于，返回结果为掩码
    auto mask = _mm512_cmpge_epi32_mask(values, other.values);
    // 将掩码的真值部分设为全 1，作为返回向量的数据
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  
  Vectorized<int32_t> eq(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ne(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> gt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ge(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> lt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> le(const Vectorized<int32_t>& other) const;
};

// 模板特化：将 int32_t 数组转换为 float 数组
template <>
inline void convert(const int32_t *src, float *dst, int64_t n) {
  int64_t i;
  // 如果不是 Microsoft Visual Studio 编译器，则执行循环展开
#ifndef _MSC_VER
# pragma unroll
#endif
  // 使用 SIMD 指令处理数据，每次处理 Vectorized<int32_t>::size() 个元素
  for (i = 0; i <= (n - Vectorized<int32_t>::size()); i += Vectorized<int32_t>::size()) {
    auto input_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
    auto output_vec = _mm512_cvtepi32_ps(input_vec); // 将 int32_t 转换为 float
    _mm512_storeu_ps(reinterpret_cast<float*>(dst + i), output_vec); // 存储转换后的 float 数据
  }
#ifndef _MSC_VER
# pragma unroll
#endif
  // 处理剩余不足一次 SIMD 处理量的数据
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]); // 将剩余的 int32_t 数据转换为 float
  }
}

// 模板特化：将 int32_t 数组转换为 double 数组
template <>
inline void convert(const int32_t *src, double *dst, int64_t n) {
  int64_t i;
  // 如果不是 Microsoft Visual Studio 编译器，则执行循环展开
#ifndef _MSC_VER
# pragma unroll
#endif
  // 使用 SIMD 指令处理数据，每次处理 Vectorized<double>::size() 个元素
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    auto input_256_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    auto output_vec = _mm512_cvtepi32_pd(input_256_vec); // 将 int32_t 转换为 double
    _mm512_storeu_pd(reinterpret_cast<double*>(dst + i), output_vec); // 存储转换后的 double 数据
  }
#ifndef _MSC_VER
# pragma unroll
#endif
  // 处理剩余不足一次 SIMD 处理量的数据
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]); // 将剩余的 int32_t 数据转换为 double
  }
}

// 模板特化：定义 Vectorized<int16_t> 类
template <>
class Vectorized<int16_t> : public Vectorizedi {
private:
  static const Vectorized<int16_t> ones; // 静态成员变量，全为 1 的向量
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0}; // 零向量

public:
  using value_type = int16_t; // 类型别名，表示向量中元素的类型为 int16_t
  static constexpr int size() {
    return 32; // 向量的大小为 32
  }
  using Vectorizedi::Vectorizedi; // 继承基类的构造函数
  Vectorized() {} // 默认构造函数
  // 构造函数，用指定的 int16_t 值初始化向量
  Vectorized(int16_t v) { values = _mm512_set1_epi16(v); }
  // 构造函数，用指定的 int16_t 值列表初始化向量
  Vectorized(int16_t val1, int16_t val2, int16_t val3, int16_t val4,
         int16_t val5, int16_t val6, int16_t val7, int16_t val8,
         int16_t val9, int16_t val10, int16_t val11, int16_t val12,
         int16_t val13, int16_t val14, int16_t val15, int16_t val16,
         int16_t val17, int16_t val18, int16_t val19, int16_t val20,
         int16_t val21, int16_t val22, int16_t val23, int16_t val24,
         int16_t val25, int16_t val26, int16_t val27, int16_t val28,
         int16_t val29, int16_t val30, int16_t val31, int16_t val32) {
    // 使用 SIMD 指令设置向量的元素
    values = _mm512_set_epi16(val32, val31, val30, val29, val28, val27, val26, val25,
                              val24, val23, val22, val21, val20, val19, val18, val17,
                              val16, val15, val14, val13, val12, val11, val10, val9,
                              val8, val7, val6, val5, val4, val3, val2, val1);
  }
  // 模板函数，根据掩码 blend 两个向量
  template <int64_t mask>
  static Vectorized<int16_t> blend(Vectorized<int16_t> a, Vectorized<int16_t> b) {
    return _mm512_mask_blend_epi16(mask, a.values, b.values);
  }
  // blendv 函数，根据掩码向量 mask 来 blend 两个向量
  static Vectorized<int16_t> blendv(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b,
                                const Vectorized<int16_t>& mask) {
    auto msb_one = _mm512_set1_epi16(0xFFFF); // 生成全 1 的掩码向量
    auto mask_ = _mm512_cmp_epi16_mask(mask, msb_one, _MM_CMPINT_EQ); // 比较 mask 和全 1 向量，生成掩码
  }
  // 生成一个 int16_t 类型的向量，按一定步长从指定的基数开始排列
  template <typename step_t>
  static Vectorized<int16_t> arange(int16_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int16_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
      base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
      base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
      base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
      base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step
    );
  }
  // 使用给定的两个 Vectorized<int16_t> 类型的向量 a 和 b，根据 count 的值选择相应的混合策略
  static Vectorized<int16_t>
  set(Vectorized<int16_t> a, Vectorized<int16_t> b, int16_t count = size()) {
    switch (count) {
      // 根据 count 的不同值选择不同的混合模式，然后返回混合后的结果向量
      case 0:
        return a;
      case 1:
        return blend<0x1>(a, b);
      case 2:
        return blend<0x3>(a, b);
      case 3:
        return blend<0x7>(a, b);
      case 4:
        return blend<0xF>(a, b);
      case 5:
        return blend<0x1F>(a, b);
      case 6:
        return blend<0x3F>(a, b);
      case 7:
        return blend<0x7F>(a, b);
      case 8:
        return blend<0xFF>(a, b);
      case 9:
        return blend<0x1FF>(a, b);
      case 10:
        return blend<0x3FF>(a, b);
      case 11:
        return blend<0x7FF>(a, b);
      case 12:
        return blend<0xFFF>(a, b);
      case 13:
        return blend<0x1FFF>(a, b);
      case 14:
        return blend<0x3FFF>(a, b);
      case 15:
        return blend<0x7FFF>(a, b);
      case 16:
        return blend<0xFFFF>(a, b);
      case 17:
        return blend<0x1FFFF>(a, b);
      case 18:
        return blend<0x3FFFF>(a, b);
      case 19:
        return blend<0x7FFFF>(a, b);
      case 20:
        return blend<0xFFFFF>(a, b);
      case 21:
        return blend<0x1FFFFF>(a, b);
      case 22:
        return blend<0x3FFFFF>(a, b);
      case 23:
        return blend<0x7FFFFF>(a, b);
      case 24:
        return blend<0xFFFFFF>(a, b);
      case 25:
        return blend<0x1FFFFFF>(a, b);
      case 26:
        return blend<0x3FFFFFF>(a, b);
      case 27:
        return blend<0x7FFFFFF>(a, b);
      case 28:
        return blend<0xFFFFFFF>(a, b);
      case 29:
        return blend<0x1FFFFFFF>(a, b);
      case 30:
        return blend<0x3FFFFFFF>(a, b);
      case 31:
        return blend<0x7FFFFFFF>(a, b);
    }
    // 默认情况下返回向量 b
    return b;
  }
  // 从给定地址 ptr 处加载一个 Vectorized<int16_t> 类型的向量
  static Vectorized<int16_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  // 从给定地址 ptr 处加载一个长度为 count 的 Vectorized<int16_t> 类型的向量
  static Vectorized<int16_t> loadu(const void* ptr, int16_t count) {
    // 如果 count 等于当前向量的大小，直接加载并返回向量
    if (count == size()) {
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    }
    // 如果 count 不等于向量大小，这里应该有错误处理的代码或者默认返回的逻辑，但当前代码片段不包含此部分
  }
  Vectorized<int16_t> operator==(const Vectorized<int16_t>& other) const {
    // 使用 AVX-512 指令比较当前向量和另一个向量的每个元素是否相等，并生成比较结果的掩码
    auto mask = _mm512_cmpeq_epi16_mask(values, other.values);
    // 根据掩码将所有元素设置为 0xFFFF 或 0x0000，表示相等或不相等
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator!=(const Vectorized<int16_t>& other) const {
    // 使用 AVX-512 指令比较当前向量和另一个向量的每个元素是否不相等，并生成比较结果的掩码
    auto mask = _mm512_cmpneq_epi16_mask(values, other.values);
    // 根据掩码将所有元素设置为 0xFFFF 或 0x0000，表示不相等或相等
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator<(const Vectorized<int16_t>& other) const {
    // 使用 AVX-512 指令比较当前向量和另一个向量的每个元素是否小于，并生成比较结果的掩码
    auto mask = _mm512_cmplt_epi16_mask(values, other.values);
    // 根据掩码将所有元素设置为 0xFFFF 或 0x0000，表示小于或不小于
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator<=(const Vectorized<int16_t>& other) const {
    // 使用 AVX-512 指令比较当前向量和另一个向量的每个元素是否小于等于，并生成比较结果的掩码
    auto mask = _mm512_cmple_epi16_mask(values, other.values);
    // 根据掩码将所有元素设置为 0xFFFF 或 0x0000，表示小于等于或大于
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator>(const Vectorized<int16_t>& other) const {
    // 使用 AVX-512 指令比较当前向量和另一个向量的每个元素是否大于，并生成比较结果的掩码
    auto mask = _mm512_cmpgt_epi16_mask(values, other.values);
    // 根据掩码将所有元素设置为 0xFFFF 或 0x0000，表示大于或不大于
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator>=(const Vectorized<int16_t>& other) const {
    // 使用 AVX-512 指令比较当前向量和另一个向量的每个元素是否大于等于，并生成比较结果的掩码
    auto mask = _mm512_cmpge_epi16_mask(values, other.values);
    // 根据掩码将所有元素设置为 0xFFFF 或 0x0000，表示大于等于或小于
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
};

// 模板类 Vectorized8，继承自 Vectorizedi
template <typename T>
class Vectorized8 : public Vectorizedi {
  // 静态断言，确保 T 类型为 int8_t 或 uint8_t
  static_assert(
    std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
    "Only int8_t/uint8_t are supported");
protected:
  // 静态常量，用于表示全零的 __m512i 向量
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
  // 静态常量，用于表示全一的 Vectorized<T> 向量
  static const Vectorized<T> ones;
public:
  // 类型别名 value_type 为 T
  using value_type = T;
  // 静态 constexpr 方法，返回向量的大小为 64
  static constexpr int size() {
    return 64;
  }
  // 继承基类 Vectorizedi 的构造函数
  using Vectorizedi::Vectorizedi;
  // 默认构造函数
  Vectorized8() {}
  // 构造函数，初始化所有值为 v 的 Vectorized8 对象
  Vectorized8(T v) { values = _mm512_set1_epi8(v); }
  // 复杂构造函数，设置每个元素的值，构造 Vectorized8 对象
  Vectorized8(T val1, T val2, T val3, T val4,
         T val5, T val6, T val7, T val8,
         T val9, T val10, T val11, T val12,
         T val13, T val14, T val15, T val16,
         T val17, T val18, T val19, T val20,
         T val21, T val22, T val23, T val24,
         T val25, T val26, T val27, T val28,
         T val29, T val30, T val31, T val32,
         T val33, T val34, T val35, T val36,
         T val37, T val38, T val39, T val40,
         T val41, T val42, T val43, T val44,
         T val45, T val46, T val47, T val48,
         T val49, T val50, T val51, T val52,
         T val53, T val54, T val55, T val56,
         T val57, T val58, T val59, T val60,
         T val61, T val62, T val63, T val64){
    values = _mm512_set_epi8(val64, val63, val62, val61, val60, val59, val58, val57,
                              val56, val55, val54, val53, val52, val51, val50, val49,
                              val48, val47, val46, val45, val44, val43, val42, val41,
                              val40, val39, val38, val37, val36, val35, val34, val33,
                              val32, val31, val30, val29, val28, val27, val26, val25,
                              val24, val23, val22, val21, val20, val19, val18, val17,
                              val16, val15, val14, val13, val12, val11, val10, val9,
                              val8, val7, val6, val5, val4, val3, val2, val1);
  }
  // 模板方法，根据 mask 对两个向量 a 和 b 进行融合操作
  template <int64_t mask>
  static Vectorized<T> blend(Vectorized<T> a, Vectorized<T> b) {
    return _mm512_mask_blend_epi8(mask, a.values, b.values);
  }
  // 模板方法，生成一个从 base 开始，步长为 step 的 arange 向量
  template <typename step_t>
  static Vectorized<T> arange(T base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
      base,             // 构造一个 Vectorized<T> 对象，初始化第一个元素为 base
      base +      step, // 初始化第二个元素为 base + step
      base +  2 * step, // 初始化第三个元素为 base + 2 * step
      base +  3 * step, // 初始化第四个元素为 base + 3 * step
      base +  4 * step, // 初始化第五个元素为 base + 4 * step
      base +  5 * step, // 初始化第六个元素为 base + 5 * step
      base +  6 * step, // 初始化第七个元素为 base + 6 * step
      base +  7 * step, // 初始化第八个元素为 base + 7 * step
      base +  8 * step, // 初始化第九个元素为 base + 8 * step
      base +  9 * step, // 初始化第十个元素为 base + 9 * step
      base + 10 * step, // 初始化第十一个元素为 base + 10 * step
      base + 11 * step, // 初始化第十二个元素为 base + 11 * step
      base + 12 * step, // 初始化第十三个元素为 base + 12 * step
      base + 13 * step, // 初始化第十四个元素为 base + 13 * step
      base + 14 * step, // 初始化第十五个元素为 base + 14 * step
      base + 15 * step, // 初始化第十六个元素为 base + 15 * step
      base + 16 * step, // 初始化第十七个元素为 base + 16 * step
      base + 17 * step, // 初始化第十八个元素为 base + 17 * step
      base + 18 * step, // 初始化第十九个元素为 base + 18 * step
      base + 19 * step, // 初始化第二十个元素为 base + 19 * step
      base + 20 * step, // 初始化第二十一个元素为 base + 20 * step
      base + 21 * step, // 初始化第二十二个元素为 base + 21 * step
      base + 22 * step, // 初始化第二十三个元素为 base + 22 * step
      base + 23 * step, // 初始化第二十四个元素为 base + 23 * step
      base + 24 * step, // 初始化第二十五个元素为 base + 24 * step
      base + 25 * step, // 初始化第二十六个元素为 base + 25 * step
      base + 26 * step, // 初始化第二十七个元素为 base + 26 * step
      base + 27 * step, // 初始化第二十八个元素为 base + 27 * step
      base + 28 * step, // 初始化第二十九个元素为 base + 28 * step
      base + 29 * step, // 初始化第三十个元素为 base + 29 * step
      base + 30 * step, // 初始化第三十一个元素为 base + 30 * step
      base + 31 * step, // 初始化第三十二个元素为 base + 31 * step
      base + 32 * step, // 初始化第三十三个元素为 base + 32 * step
      base + 33 * step, // 初始化第三十四个元素为 base + 33 * step
      base + 34 * step, // 初始化第三十五个元素为 base + 34 * step
      base + 35 * step, // 初始化第三十六个元素为 base + 35 * step
      base + 36 * step, // 初始化第三十七个元素为 base + 36 * step
      base + 37 * step, // 初始化第三十八个元素为 base + 37 * step
      base + 38 * step, // 初始化第三十九个元素为 base + 38 * step
      base + 39 * step, // 初始化第四十个元素为 base + 39 * step
      base + 40 * step, // 初始化第四十一个元素为 base + 40 * step
      base + 41 * step, // 初始化第四十二个元素为 base + 41 * step
      base + 42 * step, // 初始化第四十三个元素为 base + 42 * step
      base + 43 * step, // 初始化第四十四个元素为 base + 43 * step
      base + 44 * step, // 初始化第四十五个元素为 base + 44 * step
      base + 45 * step, // 初始化第四十六个元素为 base + 45 * step
      base + 46 * step, // 初始化第四十七个元素为 base + 46 * step
      base + 47 * step, // 初始化第四十八个元素为 base + 47 * step
      base + 48 * step, // 初始化第四十九个元素为 base + 48 * step
      base + 49 * step, // 初始化第五十个元素为 base + 49 * step
      base + 50 * step, // 初始化第五十一个元素为 base + 50 * step
      base + 51 * step, // 初始化第五十二个元素为 base + 51 * step
      base + 52 * step, // 初始化第五十三个元素为 base + 52 * step
      base + 53 * step, // 初始化第五十四个元素为 base + 53 * step
      base + 54 * step, // 初始化第五十五个元素为 base + 54 * step
      base + 55 * step, // 初始化第五十六个元素为 base + 55 * step
      base + 56 * step, // 初始化第五十七个元素为 base + 56 * step
      base + 57 * step, // 初始化第五十八个元素为 base + 57 * step
      base + 58 * step, // 初始化第五十九个元素为 base + 58 * step
      base + 59 * step, // 初始化第六十个元素为 base + 59 * step
      base + 60 * step, // 初始化第六十一个元素为 base + 60 * step
      base + 61 * step, // 初始化第六十二个元素为 base + 61 * step
      base + 62 * step, // 初始化第六十三个元素为 base + 62 * step
      base + 63 * step  // 初始化第六十四个元素为 base + 63 * step
    );
  }
    if (count == size()) {
      // 如果 count 等于向量大小，说明要存储整个向量
      // ptr 在这里不需要对齐。参见链接中的说明：
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      if (count == 16) {
        // 如果只有 16 个元素要存储，使用快速路径
        _mm_storeu_si128(
          reinterpret_cast<__m128i*>(ptr),
          _mm512_castsi512_si128(values));
      } else {
        // 创建一个掩码，用来标识存储的元素
        __mmask64 mask = (1ULL << count) - 1;
        // 根据掩码存储数据到 ptr 指向的内存位置
        _mm512_mask_storeu_epi8(ptr, mask, values);
      }
    }
  }
  const T& operator[](int idx) const  = delete;
  T& operator[](int idx)  = delete;
  // 返回当前向量化对象的实部，即本身
  Vectorized<T> real() const {
    return *this;
  }
  // 返回当前向量化对象的虚部，使用全 0 向量表示
  Vectorized<T> imag() const {
    return _mm512_set1_epi8(0);
  }
  // 返回当前向量化对象的共轭，即本身
  Vectorized<T> conj() const {
    return *this;
  }
};

// 特化模板类 Vectorized<int8_t> 的实现
template<>
class Vectorized<int8_t>: public Vectorized8<int8_t> {
public:
  // 继承父类构造函数
  using Vectorized8::Vectorized8;

  // 静态方法，根据掩码 mask 对两个 int8_t 向量 a 和 b 进行混合
  static Vectorized<int8_t> blendv(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b,
                               const Vectorized<int8_t>& mask) {
    // 创建一个所有位都是 1 的掩码
    auto msb_one = _mm512_set1_epi8(0xFF);
    // 比较 mask 和 msb_one，生成比较结果的掩码
    auto mask_ = _mm512_cmp_epi8_mask(mask, msb_one, _MM_CMPINT_EQ);
    // 使用掩码 mask_ 对 a 和 b 进行混合
    return _mm512_mask_blend_epi8(mask_, a.values, b.values);
  }

  // 返回当前向量的负向量
  Vectorized<int8_t> neg() const;

  // 返回当前向量的绝对值向量
  Vectorized<int8_t> abs() const {
    return _mm512_abs_epi8(values);
  }

  // 重载 == 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<int8_t> operator==(const Vectorized<int8_t>& other) const {
    // 比较当前向量和另一个向量的每个元素是否相等，生成比较结果的掩码
    auto mask = _mm512_cmpeq_epi8_mask(values, other.values);
    // 使用掩码 mask 设置全 1 的向量
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }

  // 重载 != 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<int8_t> operator!=(const Vectorized<int8_t>& other) const {
    // 比较当前向量和另一个向量的每个元素是否不相等，生成比较结果的掩码
    auto mask = _mm512_cmpneq_epi8_mask(values, other.values);
    // 使用掩码 mask 设置全 1 的向量
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }

  // 重载 < 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<int8_t> operator<(const Vectorized<int8_t>& other) const {
    // 比较当前向量和另一个向量的每个元素是否小于，生成比较结果的掩码
    auto mask = _mm512_cmplt_epi8_mask(values, other.values);
    // 使用掩码 mask 设置全 1 的向量
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }

  // 重载 <= 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<int8_t> operator<=(const Vectorized<int8_t>& other) const {
    // 比较当前向量和另一个向量的每个元素是否小于等于，生成比较结果的掩码
    auto mask = _mm512_cmple_epi8_mask(values, other.values);
    // 使用掩码 mask 设置全 1 的向量
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }

  // 重载 > 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<int8_t> operator>(const Vectorized<int8_t>& other) const {
    // 使用另一个向量的 < 运算符实现
    return other < *this;
  }

  // 重载 >= 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<int8_t> operator>=(const Vectorized<int8_t>& other) const {
    // 使用另一个向量的 <= 运算符实现
    return other <= *this;
  }

  // 返回当前向量与另一个向量的相等比较结果
  Vectorized<int8_t> eq(const Vectorized<int8_t>& other) const;
  // 返回当前向量与另一个向量的不等比较结果
  Vectorized<int8_t> ne(const Vectorized<int8_t>& other) const;
  // 返回当前向量与另一个向量的大于比较结果
  Vectorized<int8_t> gt(const Vectorized<int8_t>& other) const;
  // 返回当前向量与另一个向量的大于等于比较结果
  Vectorized<int8_t> ge(const Vectorized<int8_t>& other) const;
  // 返回当前向量与另一个向量的小于比较结果
  Vectorized<int8_t> lt(const Vectorized<int8_t>& other) const;
  // 返回当前向量与另一个向量的小于等于比较结果
  Vectorized<int8_t> le(const Vectorized<int8_t>& other) const;
};

// 特化模板类 Vectorized<uint8_t> 的实现
template<>
class Vectorized<uint8_t>: public Vectorized8<uint8_t> {
public:
  // 继承父类构造函数
  using Vectorized8::Vectorized8;

  // 静态方法，根据掩码 mask 对两个 uint8_t 向量 a 和 b 进行混合
  static Vectorized<uint8_t> blendv(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b,
                               const Vectorized<uint8_t>& mask) {
    // 创建一个所有位都是 1 的掩码
    auto msb_one = _mm512_set1_epi8(0xFF);
    // 比较 mask 和 msb_one，生成比较结果的掩码
    auto mask_ = _mm512_cmp_epu8_mask(mask, msb_one, _MM_CMPINT_EQ);
    // 使用掩码 mask_ 对 a 和 b 进行混合
    return _mm512_mask_blend_epi8(mask_, a.values, b.values);
  }

  // 返回当前向量的负向量
  Vectorized<uint8_t> neg() const;

  // 返回当前向量的绝对值向量
  Vectorized<uint8_t> abs() const {
    // 返回当前向量自身，因为 uint8_t 类型没有负数
    return *this;
  }

  // 重载 == 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<uint8_t> operator==(const Vectorized<uint8_t>& other) const {
    // 比较当前向量和另一个向量的每个元素是否相等，生成比较结果的掩码
    auto mask = _mm512_cmpeq_epu8_mask(values, other.values);
    // 使用掩码 mask 设置全 1 的向量
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }

  // 重载 != 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<uint8_t> operator!=(const Vectorized<uint8_t>& other) const {
    // 比较当前向量和另一个向量的每个元素是否不相等，生成比较结果的掩码
    auto mask = _mm512_cmpneq_epu8_mask(values, other.values);
    // 使用掩码 mask 设置全 1 的向量
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }

  // 重载 < 运算符，返回比较结果为真的位设置为全 1 的向量
  Vectorized<uint8_t> operator<(const Vectorized<uint8_t>& other) const {
   `
  // 使用 `_mm512_cmplt_epu8_mask` 函数比较 `values` 与 `other.values` 的大小，生成一个掩码，掩码中小于的位为1，其余为0
  auto mask = _mm512_cmplt_epu8_mask(values, other.values);
  // 使用 `_mm512_mask_set1_epi8` 函数，将 `zero_vector` 中掩码为1的位设置为0xFF，其余位保持为原值
  return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
}

// 定义 `operator<=` 运算符，比较当前对象与 `other` 对象的值，返回一个 `Vectorized<uint8_t>` 对象
Vectorized<uint8_t> operator<=(const Vectorized<uint8_t>& other) const {
  // 使用 `_mm512_cmple_epu8_mask` 函数比较 `values` 与 `other.values` 的大小，生成一个掩码，掩码中小于等于的位为1，其余为0
  auto mask = _mm512_cmple_epu8_mask(values, other.values);
  // 使用 `_mm512_mask_set1_epi8` 函数，将 `zero_vector` 中掩码为1的位设置为0xFF，其余位保持为原值
  return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
}

// 重载 `operator>` 运算符，返回 `other < *this` 的结果
Vectorized<uint8_t> operator>(const Vectorized<uint8_t>& other) const {
  return other < *this;
}

// 重载 `operator>=` 运算符，返回 `other <= *this` 的结果
Vectorized<uint8_t> operator>=(const Vectorized<uint8_t>& other) const {
  return other <= *this;
}

// 定义一个方法 `eq`，用于比较当前对象与 `other` 对象的值是否相等
Vectorized<uint8_t> eq(const Vectorized<uint8_t>& other) const;
// 定义一个方法 `ne`，用于比较当前对象与 `other` 对象的值是否不相等
Vectorized<uint8_t> ne(const Vectorized<uint8_t>& other) const;
// 定义一个方法 `gt`，用于比较当前对象与 `other` 对象的值是否大于
Vectorized<uint8_t> gt(const Vectorized<uint8_t>& other) const;
// 定义一个方法 `ge`，用于比较当前对象与 `other` 对象的值是否大于或等于
Vectorized<uint8_t> ge(const Vectorized<uint8_t>& other) const;
// 定义一个方法 `lt`，用于比较当前对象与 `other` 对象的值是否小于
Vectorized<uint8_t> lt(const Vectorized<uint8_t>& other) const;
// 定义一个方法 `le`，用于比较当前对象与 `other` 对象的值是否小于或等于
Vectorized<uint8_t> le(const Vectorized<uint8_t>& other) const;
// 结束模板特化定义，这里是一个分号，用于结束前一个特化定义
};

// 整数64位向量化加法操作符的模板特化定义，返回两个向量相加的结果
template <>
Vectorized<int64_t> inline operator+(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_add_epi64(a, b);
}

// 整数32位向量化加法操作符的模板特化定义，返回两个向量相加的结果
template <>
Vectorized<int32_t> inline operator+(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_add_epi32(a, b);
}

// 整数16位向量化加法操作符的模板特化定义，返回两个向量相加的结果
template <>
Vectorized<int16_t> inline operator+(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_add_epi16(a, b);
}

// 整数8位向量化加法操作符的模板特化定义，返回两个向量相加的结果
template <>
Vectorized<int8_t> inline operator+(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm512_add_epi8(a, b);
}

// 无符号8位向量化加法操作符的模板特化定义，返回两个向量相加的结果
template <>
Vectorized<uint8_t> inline operator+(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm512_add_epi8(a, b);
}

// 整数64位向量化减法操作符的模板特化定义，返回两个向量相减的结果
template <>
Vectorized<int64_t> inline operator-(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_sub_epi64(a, b);
}

// 整数32位向量化减法操作符的模板特化定义，返回两个向量相减的结果
template <>
Vectorized<int32_t> inline operator-(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_sub_epi32(a, b);
}

// 整数16位向量化减法操作符的模板特化定义，返回两个向量相减的结果
template <>
Vectorized<int16_t> inline operator-(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_sub_epi16(a, b);
}

// 整数8位向量化减法操作符的模板特化定义，返回两个向量相减的结果
template <>
Vectorized<int8_t> inline operator-(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm512_sub_epi8(a, b);
}

// 无符号8位向量化减法操作符的模板特化定义，返回两个向量相减的结果
template <>
Vectorized<uint8_t> inline operator-(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm512_sub_epi8(a, b);
}

// 负号操作。在此定义，以便能够利用操作符-
inline Vectorized<int64_t> Vectorized<int64_t>::neg() const {
  return Vectorized<int64_t>(0) - *this;
}

// 整数32位向量的负号操作，返回当前向量的负数
inline Vectorized<int32_t> Vectorized<int32_t>::neg() const {
  return Vectorized<int32_t>(0) - *this;
}

// 整数16位向量的负号操作，返回当前向量的负数
inline Vectorized<int16_t> Vectorized<int16_t>::neg() const {
  return Vectorized<int16_t>(0) - *this;
}

// 整数8位向量的负号操作，返回当前向量的负数
inline Vectorized<int8_t> Vectorized<int8_t>::neg() const {
  return Vectorized<int8_t>(0) - *this;
}

// 无符号8位向量的负号操作，返回当前向量的负数
inline Vectorized<uint8_t> Vectorized<uint8_t>::neg() const {
  return Vectorized<uint8_t>(0) - *this;
}

// 整数64位向量化乘法操作符的模板特化定义，返回两个向量相乘的结果
template <>
Vectorized<int64_t> inline operator*(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_mullo_epi64(a, b);
}

// 整数32位向量化乘法操作符的模板特化定义，返回两个向量相乘的结果
template <>
Vectorized<int32_t> inline operator*(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_mullo_epi32(a, b);
}

// 整数16位向量化乘法操作符的模板特化定义，返回两个向量相乘的结果
template <>
Vectorized<int16_t> inline operator*(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_mullo_epi16(a, b);
}

// 通用的512位向量整数元素级二元操作函数模板，执行给定操作符的操作
template <typename T, typename Op>
Vectorized<T> inline int_elementwise_binary_512(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
  T values_a[Vectorized<T>::size()];
  T values_b[Vectorized<T>::size()];
  a.store(values_a);
  b.store(values_b);
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    values_a[i] = op(values_a[i], values_b[i]);
  }
  return Vectorized<T>::loadu(values_a);
}

// 模板特化定义，以支持特定的整数类型
template <>
// 重载运算符 *，用于两个 int8_t 向量的乘法运算
Vectorized<int8_t> inline operator*(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  // 如果 CPU 不支持 AVX512 指令集，使用 int_elementwise_binary_512 函数进行 int8_t 向量的逐元素乘法
#ifndef CPU_CAPABILITY_AVX512
  return int_elementwise_binary_512(a, b, std::multiplies<int8_t>());
#else
  // 创建一个掩码，用于处理 int8_t 向量的低位字节
  __m512i mask00FF = _mm512_set1_epi16(0x00FF);
  // 取 a 和 b 向量的低位字节
  __m512i a_lo = _mm512_srai_epi16(_mm512_slli_epi16(a, 8), 8);
  __m512i b_lo = _mm512_srai_epi16(_mm512_slli_epi16(b, 8), 8);
  // 取 a 和 b 向量的高位字节
  __m512i a_hi = _mm512_srai_epi16(a, 8);
  __m512i b_hi = _mm512_srai_epi16(b, 8);
  // 计算低位和高位乘积，并通过掩码处理结果
  __m512i res_lo = _mm512_and_si512(_mm512_mullo_epi16(a_lo, b_lo), mask00FF);
  __m512i res_hi = _mm512_slli_epi16(_mm512_mullo_epi16(a_hi, b_hi), 8);
  // 合并低位和高位结果得到最终乘法结果向量
  __m512i res = _mm512_or_si512(res_hi, res_lo);
  return res;
#endif
}

// 重载模板特化版本的运算符 *，用于两个 uint8_t 向量的乘法运算
template <>
Vectorized<uint8_t> inline operator*(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  // 如果 CPU 不支持 AVX512 指令集，使用 int_elementwise_binary_512 函数进行 uint8_t 向量的逐元素乘法
#ifndef CPU_CAPABILITY_AVX512
  return int_elementwise_binary_512(a, b, std::multiplies<uint8_t>());
#else
  // 创建一个掩码，用于处理 uint8_t 向量的低位字节
  __m512i mask00FF = _mm512_set1_epi16(0x00FF);
  // 取 a 和 b 向量的低位字节
  __m512i a_lo = _mm512_and_si512(a, mask00FF);
  __m512i b_lo = _mm512_and_si512(b, mask00FF);
  // 取 a 和 b 向量的高位字节
  __m512i a_hi = _mm512_srli_epi16(a, 8);
  __m512i b_hi = _mm512_srli_epi16(b, 8);
  // 计算低位和高位乘积，并通过掩码处理结果
  __m512i res_lo = _mm512_and_si512(_mm512_mullo_epi16(a_lo, b_lo), mask00FF);
  __m512i res_hi = _mm512_slli_epi16(_mm512_mullo_epi16(a_hi, b_hi), 8);
  // 合并低位和高位结果得到最终乘法结果向量
  __m512i res = _mm512_or_si512(res_hi, res_lo);
  return res;
#endif
}

// 模板特化版本的 minimum 函数，用于 int64_t 向量的最小值计算
template <>
Vectorized<int64_t> inline minimum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_min_epi64(a, b); // 使用 AVX512 指令计算 int64_t 向量的最小值
}

// 模板特化版本的 minimum 函数，用于 int32_t 向量的最小值计算
template <>
Vectorized<int32_t> inline minimum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_min_epi32(a, b); // 使用 AVX512 指令计算 int32_t 向量的最小值
}

// 模板特化版本的 minimum 函数，用于 int16_t 向量的最小值计算
template <>
Vectorized<int16_t> inline minimum(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_min_epi16(a, b); // 使用 AVX512 指令计算 int16_t 向量的最小值
}

// 模板特化版本的 minimum 函数，用于 int8_t 向量的最小值计算
template <>
Vectorized<int8_t> inline minimum(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm512_min_epi8(a, b); // 使用 AVX512 指令计算 int8_t 向量的最小值
}

// 模板特化版本的 minimum 函数，用于 uint8_t 向量的最小值计算
template <>
Vectorized<uint8_t> inline minimum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm512_min_epu8(a, b); // 使用 AVX512 指令计算 uint8_t 向量的最小值
}

// 模板特化版本的 maximum 函数，用于 int64_t 向量的最大值计算
template <>
Vectorized<int64_t> inline maximum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_max_epi64(a, b); // 使用 AVX512 指令计算 int64_t 向量的最大值
}

// 模板特化版本的 maximum 函数，用于 int32_t 向量的最大值计算
template <>
Vectorized<int32_t> inline maximum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_max_epi32(a, b); // 使用 AVX512 指令计算 int32_t 向量的最大值
}

// 模板特化版本的 maximum 函数，用于 int16_t 向量的最大值计算
template <>
Vectorized<int16_t> inline maximum(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_max_epi16(a, b); // 使用 AVX512 指令计算 int16_t 向量的最大值
}

// 模板特化版本的 maximum 函数，用于 int8_t 向量的最大值计算
template <>
Vectorized<int8_t> inline maximum(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm512_max_epi8(a, b); // 使用 AVX512 指令计算 int8_t 向量的最大值
}

// 模板特化版本的 maximum 函数，用于 uint8_t 向量的最大值计算
template <>
Vectorized<uint8_t> inline maximum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm512_max_epu8(a, b); // 使用 AVX512 指令计算 uint8_t 向量的最大值
}
// 使用 SIMD 指令对 int64_t 向量进行范围限制，返回限制后的向量
Vectorized<int64_t> inline clamp(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val, const Vectorized<int64_t>& max_val) {
  return _mm512_min_epi64(max_val, _mm512_max_epi64(a, min_val));
}

// 使用 SIMD 指令对 int32_t 向量进行范围限制，返回限制后的向量
template <>
Vectorized<int32_t> inline clamp(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val, const Vectorized<int32_t>& max_val) {
  return _mm512_min_epi32(max_val, _mm512_max_epi32(a, min_val));
}

// 使用 SIMD 指令对 int16_t 向量进行范围限制，返回限制后的向量
template <>
Vectorized<int16_t> inline clamp(const Vectorized<int16_t>& a, const Vectorized<int16_t>& min_val, const Vectorized<int16_t>& max_val) {
  return _mm512_min_epi16(max_val, _mm512_max_epi16(a, min_val));
}

// 使用 SIMD 指令对 int8_t 向量进行范围限制，返回限制后的向量
template <>
Vectorized<int8_t> inline clamp(const Vectorized<int8_t>& a, const Vectorized<int8_t>& min_val, const Vectorized<int8_t>& max_val) {
  return _mm512_min_epi8(max_val, _mm512_max_epi8(a, min_val));
}

// 使用 SIMD 指令对 uint8_t 向量进行范围限制，返回限制后的向量
template <>
Vectorized<uint8_t> inline clamp(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& min_val, const Vectorized<uint8_t>& max_val) {
  return _mm512_min_epu8(max_val, _mm512_max_epu8(a, min_val));
}

// 使用 SIMD 指令对 int64_t 向量进行最大值限制，返回限制后的向量
template <>
Vectorized<int64_t> inline clamp_max(const Vectorized<int64_t>& a, const Vectorized<int64_t>& max_val) {
  return _mm512_min_epi64(max_val, a);
}

// 使用 SIMD 指令对 int32_t 向量进行最大值限制，返回限制后的向量
template <>
Vectorized<int32_t> inline clamp_max(const Vectorized<int32_t>& a, const Vectorized<int32_t>& max_val) {
  return _mm512_min_epi32(max_val, a);
}

// 使用 SIMD 指令对 int16_t 向量进行最大值限制，返回限制后的向量
template <>
Vectorized<int16_t> inline clamp_max(const Vectorized<int16_t>& a, const Vectorized<int16_t>& max_val) {
  return _mm512_min_epi16(max_val, a);
}

// 使用 SIMD 指令对 int8_t 向量进行最大值限制，返回限制后的向量
template <>
Vectorized<int8_t> inline clamp_max(const Vectorized<int8_t>& a, const Vectorized<int8_t>& max_val) {
  return _mm512_min_epi8(max_val, a);
}

// 使用 SIMD 指令对 uint8_t 向量进行最大值限制，返回限制后的向量
template <>
Vectorized<uint8_t> inline clamp_max(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& max_val) {
  return _mm512_min_epu8(max_val, a);
}

// 使用 SIMD 指令对 int64_t 向量进行最小值限制，返回限制后的向量
template <>
Vectorized<int64_t> inline clamp_min(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val) {
  return _mm512_max_epi64(min_val, a);
}

// 使用 SIMD 指令对 int32_t 向量进行最小值限制，返回限制后的向量
template <>
Vectorized<int32_t> inline clamp_min(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val) {
  return _mm512_max_epi32(min_val, a);
}

// 使用 SIMD 指令对 int16_t 向量进行最小值限制，返回限制后的向量
template <>
Vectorized<int16_t> inline clamp_min(const Vectorized<int16_t>& a, const Vectorized<int16_t>& min_val) {
  return _mm512_max_epi16(min_val, a);
}

// 使用 SIMD 指令对 int8_t 向量进行最小值限制，返回限制后的向量
template <>
Vectorized<int8_t> inline clamp_min(const Vectorized<int8_t>& a, const Vectorized<int8_t>& min_val) {
  return _mm512_max_epi8(min_val, a);
}

// 使用 SIMD 指令对 uint8_t 向量进行最小值限制，返回限制后的向量
template <>
Vectorized<uint8_t> inline clamp_min(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& min_val) {
  return _mm512_max_epu8(min_val, a);
}

// 将任意类型 T 的数据指针转换为 int32_t 向量，加载数据并返回
template<typename T>
Vectorized<int32_t> inline convert_to_int32(const T* ptr) {
  return Vectorized<int32_t>::loadu(ptr);
}

// 将 int8_t 类型数据指针转换为 int32_t 向量，使用 SIMD 指令加载数据并返回
template<>
Vectorized<int32_t> inline convert_to_int32<int8_t>(const int8_t* ptr) {
  return _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
}
// 定义一个模板函数，将指向 uint8_t 类型的指针转换为 Vectorized<int32_t> 类型
Vectorized<int32_t> inline convert_to_int32<uint8_t>(const uint8_t* ptr) {
  // 使用 SIMD 指令加载并将 ptr 指向的内存数据转换为 128 位整数类型，再将其转换为 int32_t 类型并返回
  return _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
}

// 重载模板特化，定义 int64_t 类型的向量除法运算符
template <>
Vectorized<int64_t> inline operator/(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  // 调用 int_elementwise_binary_512 函数，使用 std::divides<int64_t>() 进行逐元素除法运算并返回结果
  return int_elementwise_binary_512(a, b, std::divides<int64_t>());
}

// 以下为 int32_t、int16_t、int8_t 和 uint8_t 类型的向量除法运算符重载模板特化，实现方法与上述相似

// 按位与运算符重载模板，要求 T 类型必须是 Vectorizedi 的派生类
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 使用 SIMD 指令对 a 和 b 进行按位与操作并返回结果
  return _mm512_and_si512(a, b);
}

// 按位或运算符重载模板，要求 T 类型必须是 Vectorizedi 的派生类
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 使用 SIMD 指令对 a 和 b 进行按位或操作并返回结果
  return _mm512_or_si512(a, b);
}

// 按位异或运算符重载模板，要求 T 类型必须是 Vectorizedi 的派生类
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  // 使用 SIMD 指令对 a 和 b 进行按位异或操作并返回结果
  return _mm512_xor_si512(a, b);
}

// 按位取反运算符重载模板，要求 T 类型必须是 Vectorizedi 的派生类
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  // 使用 SIMD 指令对 a 进行按位取反操作并返回结果
  return _mm512_xor_si512(a, _mm512_set1_epi32(-1));
}

// int64_t 类型向量的相等比较方法，返回相等比较结果的向量
inline Vectorized<int64_t> Vectorized<int64_t>::eq(const Vectorized<int64_t>& other) const {
  // 使用 SIMD 指令进行逐元素比较是否相等，然后将结果与 Vectorized<int64_t>(1) 按位与返回
  return (*this == other) & Vectorized<int64_t>(1);
}

// int64_t 类型向量的不等比较方法，返回不等比较结果的向量
inline Vectorized<int64_t> Vectorized<int64_t>::ne(const Vectorized<int64_t>& other) const {
  // 使用 SIMD 指令进行逐元素比较是否不等，然后将结果与 Vectorized<int64_t>(1) 按位与返回
  return (*this != other) & Vectorized<int64_t>(1);
}

// int64_t 类型向量的大于比较方法，返回大于比较结果的向量
inline Vectorized<int64_t> Vectorized<int64_t>::gt(const Vectorized<int64_t>& other) const {
  // 使用 SIMD 指令进行逐元素比较是否大于，然后将结果与 Vectorized<int64_t>(1) 按位与返回
  return (*this > other) & Vectorized<int64_t>(1);
}

// int64_t 类型向量的大于等于比较方法，返回大于等于比较结果的向量
inline Vectorized<int64_t> Vectorized<int64_t>::ge(const Vectorized<int64_t>& other) const {
  // 使用 SIMD 指令进行逐元素比较是否大于等于，然后将结果与 Vectorized<int64_t>(1) 按位与返回
  return (*this >= other) & Vectorized<int64_t>(1);
}

// int64_t 类型向量的小于比较方法，返回小于比较结果的向量
inline Vectorized<int64_t> Vectorized<int64_t>::lt(const Vectorized<int64_t>& other) const {
  // 使用 SIMD 指令进行逐元素比较是否小于，然后将结果与 Vectorized<int64_t>(1) 按位与返回
  return (*this < other) & Vectorized<int64_t>(1);
}

// int64_t 类型向量的小于等于比较方法，返回小于等于比较结果的向量
inline Vectorized<int64_t> Vectorized<int64_t>::le(const Vectorized<int64_t>& other) const {
  // 使用 SIMD 指令进行逐元素比较是否小于等于，然后将结果与 Vectorized<int64_t>(1) 按位与返回
  return (*this <= other) & Vectorized<int64_t>(1);
}

// int32_t 类型向量的相等比较方法，返回相等比较结果的向量
inline Vectorized<int32_t> Vectorized<int32_t>::eq(const Vectorized<int32_t>& other) const {
  // 使用 SIMD 指令进行逐元素比较是否相等，然后将结果与 Vectorized<int32_t>(1) 按位与返回
  return (*this == other) & Vectorized<int32_t>(1);
}
# 返回当前向量与另一个向量逐元素进行不等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int32_t> Vectorized<int32_t>::ne(const Vectorized<int32_t>& other) const {
    return (*this != other) & Vectorized<int32_t>(1);
}

# 返回当前向量与另一个向量逐元素进行大于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int32_t> Vectorized<int32_t>::gt(const Vectorized<int32_t>& other) const {
    return (*this > other) & Vectorized<int32_t>(1);
}

# 返回当前向量与另一个向量逐元素进行大于等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int32_t> Vectorized<int32_t>::ge(const Vectorized<int32_t>& other) const {
    return (*this >= other) & Vectorized<int32_t>(1);
}

# 返回当前向量与另一个向量逐元素进行小于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int32_t> Vectorized<int32_t>::lt(const Vectorized<int32_t>& other) const {
    return (*this < other) & Vectorized<int32_t>(1);
}

# 返回当前向量与另一个向量逐元素进行小于等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int32_t> Vectorized<int32_t>::le(const Vectorized<int32_t>& other) const {
    return (*this <= other) & Vectorized<int32_t>(1);
}

# 返回当前向量与另一个向量逐元素进行等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int16_t> Vectorized<int16_t>::eq(const Vectorized<int16_t>& other) const {
    return (*this == other) & Vectorized<int16_t>(1);
}

# 返回当前向量与另一个向量逐元素进行不等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int16_t> Vectorized<int16_t>::ne(const Vectorized<int16_t>& other) const {
    return (*this != other) & Vectorized<int16_t>(1);
}

# 返回当前向量与另一个向量逐元素进行大于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int16_t> Vectorized<int16_t>::gt(const Vectorized<int16_t>& other) const {
    return (*this > other) & Vectorized<int16_t>(1);
}

# 返回当前向量与另一个向量逐元素进行大于等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int16_t> Vectorized<int16_t>::ge(const Vectorized<int16_t>& other) const {
    return (*this >= other) & Vectorized<int16_t>(1);
}

# 返回当前向量与另一个向量逐元素进行小于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int16_t> Vectorized<int16_t>::lt(const Vectorized<int16_t>& other) const {
    return (*this < other) & Vectorized<int16_t>(1);
}

# 返回当前向量与另一个向量逐元素进行小于等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int16_t> Vectorized<int16_t>::le(const Vectorized<int16_t>& other) const {
    return (*this <= other) & Vectorized<int16_t>(1);
}

# 返回当前向量与另一个向量逐元素进行等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int8_t> Vectorized<int8_t>::eq(const Vectorized<int8_t>& other) const {
    return (*this == other) & Vectorized<int8_t>(1);
}

# 返回当前向量与另一个向量逐元素进行不等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int8_t> Vectorized<int8_t>::ne(const Vectorized<int8_t>& other) const {
    return (*this != other) & Vectorized<int8_t>(1);
}

# 返回当前向量与另一个向量逐元素进行大于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int8_t> Vectorized<int8_t>::gt(const Vectorized<int8_t>& other) const {
    return (*this > other) & Vectorized<int8_t>(1);
}

# 返回当前向量与另一个向量逐元素进行大于等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int8_t> Vectorized<int8_t>::ge(const Vectorized<int8_t>& other) const {
    return (*this >= other) & Vectorized<int8_t>(1);
}

# 返回当前向量与另一个向量逐元素进行小于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int8_t> Vectorized<int8_t>::lt(const Vectorized<int8_t>& other) const {
    return (*this < other) & Vectorized<int8_t>(1);
}

# 返回当前向量与另一个向量逐元素进行小于等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<int8_t> Vectorized<int8_t>::le(const Vectorized<int8_t>& other) const {
    return (*this <= other) & Vectorized<int8_t>(1);
}

# 返回当前向量与另一个向量逐元素进行等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<uint8_t> Vectorized<uint8_t>::eq(const Vectorized<uint8_t>& other) const {
    return (*this == other) & Vectorized<uint8_t>(1);
}

# 返回当前向量与另一个向量逐元素进行不等于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<uint8_t> Vectorized<uint8_t>::ne(const Vectorized<uint8_t>& other) const {
    return (*this != other) & Vectorized<uint8_t>(1);
}

# 返回当前向量与另一个向量逐元素进行大于比较的结果，并将每个比较结果与向量 [1] 逻辑与
inline Vectorized<uint8_t> Vectorized<uint8_t>::gt(const Vectorized<uint8_t>& other) const {
    return (*this > other) & Vectorized<uint8_t>(1);
}
// 返回当前向量与另一个向量的每个元素是否大于或等于的结果，结果以向量形式返回
inline Vectorized<uint8_t> Vectorized<uint8_t>::ge(const Vectorized<uint8_t>& other) const {
  return (*this >= other) & Vectorized<uint8_t>(1);
}

// 返回当前向量与另一个向量的每个元素是否小于的结果，结果以向量形式返回
inline Vectorized<uint8_t> Vectorized<uint8_t>::lt(const Vectorized<uint8_t>& other) const {
  return (*this < other) & Vectorized<uint8_t>(1);
}

// 返回当前向量与另一个向量的每个元素是否小于或等于的结果，结果以向量形式返回
inline Vectorized<uint8_t> Vectorized<uint8_t>::le(const Vectorized<uint8_t>& other) const {
  return (*this <= other) & Vectorized<uint8_t>(1);
}

// 根据输入类型和左移标志位，对输入向量执行不同的位移操作
template <bool left_shift, typename T, typename std::enable_if_t<std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>, int> = 0>
c0 = _mm512_sllv_epi16(a0, b0);
else
if constexpr (std::is_same_v<T, int8_t>)
  c0 = _mm512_srav_epi16(a0, b0);
else
  c0 = _mm512_srlv_epi16(a0, b0);
c0 = _mm512_shuffle_epi8(c0, ctl_1_0);

// 对输入数组元素进行位移，仅处理 idx%2==1 的情况
__m512i a1 = _mm512_and_si512(a, keep_1);
__m512i b1 = _mm512_shuffle_epi8(b, ctl_1_0);
__m512i c1;
if (left_shift)
  c1 = _mm512_sllv_epi16(a1, b1);
else
if constexpr (std::is_same_v<T, int8_t>)
  c1 = _mm512_srav_epi16(a1, b1);
else
  c1 = _mm512_srlv_epi16(a1, b1);
c1 = _mm512_and_si512(c1, keep_1);

// 合并部分结果以得到最终结果
__m512i c = _mm512_or_si512(c0, c1);

return c;
}

// 下面是特化模板，分别处理不同数据类型的左移操作

template <>
Vectorized<int64_t> inline operator<<(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_sllv_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator<<(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_sllv_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator<<(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_sllv_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator<<(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return shift_512_8<true>(a, b);
}

template <>
Vectorized<uint8_t> inline operator<<(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return shift_512_8<true>(a, b);
}

template <>
Vectorized<int64_t> inline operator>>(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_srav_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator>>(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_srav_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator>>(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_srav_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator>>(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return shift_512_8<false>(a, b);
}

template <>
Vectorized<uint8_t> inline operator>>(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return shift_512_8<false>(a, b);
}
```