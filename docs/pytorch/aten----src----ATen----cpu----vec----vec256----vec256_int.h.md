# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vec256_int.h`

```py
#pragma once
// 表示只编译一次这个头文件

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]
// 在这个头文件中不要定义静态数据！
// 参见注释[不要使用 AVX 编译初始化器]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX2

// 定义一个结构体 Vectorizedi，用于处理 AVX2 指令集
struct Vectorizedi {
protected:
  __m256i values; // 256 位整数类型

  // 反转向量 v 中的所有位
  static inline __m256i invert(const __m256i& v) {
    const auto ones = _mm256_set1_epi64x(-1); // 创建一个所有位为 1 的向量
    return _mm256_xor_si256(ones, v); // 通过异或操作实现反转
  }
public:
  Vectorizedi() {} // 默认构造函数
  Vectorizedi(__m256i v) : values(v) {} // 使用给定的 __m256i 类型初始化
  operator __m256i() const { // 类型转换操作符，将当前对象转换为 __m256i 类型
    return values;
  }
};

#else

struct Vectorizedi {};  // dummy definition to make Vectorizedi always defined
// 为了始终定义 Vectorizedi 而提供的虚拟定义

#endif // CPU_CAPABILITY_AVX2

#ifdef CPU_CAPABILITY_AVX2

// 针对 int64_t 类型的模板化向量化类 Vectorized，继承自 Vectorizedi
template <>
class Vectorized<int64_t> : public Vectorizedi {
private:
  static const Vectorized<int64_t> ones; // 静态成员变量，存储全 1 的向量
public:
  using value_type = int64_t; // 向量中元素的类型
  using size_type = int; // 向量中元素的数量类型
  static constexpr size_type size() { // 返回向量中元素的数量（固定为 4）
    return 4;
  }
  using Vectorizedi::Vectorizedi; // 继承自 Vectorizedi 的构造函数
  Vectorized() {} // 默认构造函数
  Vectorized(int64_t v) { values = _mm256_set1_epi64x(v); } // 使用给定值初始化所有元素
  Vectorized(int64_t val1, int64_t val2, int64_t val3, int64_t val4) { // 使用四个值初始化向量
    values = _mm256_setr_epi64x(val1, val2, val3, val4);
  }
  template <int64_t mask>
  static Vectorized<int64_t> blend(Vectorized<int64_t> a, Vectorized<int64_t> b) {
    __at_align__ int64_t tmp_values[size()]; // 临时存储向量元素的数组
    a.store(tmp_values); // 将向量 a 中的值存储到 tmp_values 中
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi64(b.values, 0); // 根据掩码 mask 替换 tmp_values 的值
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi64(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi64(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi64(b.values, 3);
    return loadu(tmp_values); // 载入 tmp_values 中的值到向量中并返回
  }
  static Vectorized<int64_t> blendv(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b,
                                const Vectorized<int64_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values); // 使用 mask 对 a 和 b 进行混合
  }
  template <typename step_t>
  static Vectorized<int64_t> arange(int64_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(base, base + step, base + 2 * step, base + 3 * step); // 生成等差数列
  }
  static Vectorized<int64_t>
  set(Vectorized<int64_t> a, Vectorized<int64_t> b, int64_t count = size()) { // 设置向量中的值
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b); // 根据不同的 count 调用 blend 方法
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
    }
    return b;
  }
  static Vectorized<int64_t> loadu(const void* ptr) { // 从未对齐的内存地址加载向量
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vectorized<int64_t> loadu(const void* ptr, int64_t count) {
    __at_align__ int64_t tmp_values[size()]; // 临时存储向量元素的数组
    // 确保未初始化的内存不会改变输出值，参见 https://github.com/pytorch/pytorch/issues/32502
    // 循环遍历从 0 到 size()-1 的范围，将 tmp_values 数组中的元素初始化为 0
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    // 使用 memcpy 将 ptr 指向的内存区域的数据复制到 tmp_values 数组中，复制的字节数为 count * sizeof(int64_t)
    std::memcpy(tmp_values, ptr, count * sizeof(int64_t));
    // 调用 loadu 函数加载 tmp_values 数组中的数据，返回加载后的结果
    return loadu(tmp_values);
  }
  // 将 Vectorized 对象的值存储到 ptr 指向的内存区域中，存储的元素个数由 count 指定
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 当 count 等于 size() 时，使用 _mm256_storeu_si256 将 values 寄存器的值存储到 ptr 指向的内存区域中
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      // 当 count 大于 0 且小于 size() 时，在栈上分配临时数组 tmp_values，将 values 寄存器的值存储到 tmp_values 数组中
      __at_align__ int64_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      // 使用 memcpy 将 tmp_values 数组的数据复制到 ptr 指向的内存区域中，复制的字节数为 count * sizeof(int64_t)
      std::memcpy(ptr, tmp_values, count * sizeof(int64_t));
    }
  }
  // 禁用操作符 [] 的引用版本
  const int64_t& operator[](int idx) const  = delete;
  // 禁用操作符 [] 的非引用版本
  int64_t& operator[](int idx)  = delete;
  // 返回对当前 Vectorized 对象调用 abs 函数的结果，即取绝对值后的结果
  Vectorized<int64_t> abs() const {
    auto zero = _mm256_set1_epi64x(0);
    auto is_larger = _mm256_cmpgt_epi64(zero, values);
    auto inverse = _mm256_xor_si256(values, is_larger);
    return _mm256_sub_epi64(inverse, is_larger);
  }
  // 返回当前 Vectorized 对象本身，即返回实部
  Vectorized<int64_t> real() const {
    return *this;
  }
  // 返回一个所有元素均为零的 Vectorized 对象，即返回虚部
  Vectorized<int64_t> imag() const {
    return _mm256_set1_epi64x(0);
  }
  // 返回当前 Vectorized 对象本身，即返回共轭复数
  Vectorized<int64_t> conj() const {
    return *this;
  }
  // 返回对当前 Vectorized 对象调用 neg 函数的结果，即取负数后的结果
  Vectorized<int64_t> neg() const;
  // 返回当前 Vectorized 对象与另一个 Vectorized 对象比较相等的结果
  Vectorized<int64_t> operator==(const Vectorized<int64_t>& other) const {
    return _mm256_cmpeq_epi64(values, other.values);
  }
  // 返回当前 Vectorized 对象与另一个 Vectorized 对象比较不等的结果
  Vectorized<int64_t> operator!=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpeq_epi64(values, other.values));
  }
  // 返回当前 Vectorized 对象与另一个 Vectorized 对象比较大小（大于）的结果
  Vectorized<int64_t> operator<(const Vectorized<int64_t>& other) const {
    return _mm256_cmpgt_epi64(other.values, values);
  }
  // 返回当前 Vectorized 对象与另一个 Vectorized 对象比较大小（小于等于）的结果
  Vectorized<int64_t> operator<=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpgt_epi64(values, other.values));
  }
  // 返回当前 Vectorized 对象与另一个 Vectorized 对象比较大小（小于）的结果
  Vectorized<int64_t> operator>(const Vectorized<int64_t>& other) const {
    return _mm256_cmpgt_epi64(values, other.values);
  }
  // 返回当前 Vectorized 对象与另一个 Vectorized 对象比较大小（大于等于）的结果
  Vectorized<int64_t> operator>=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpgt_epi64(other.values, values));
  }

  // 返回对当前 Vectorized 对象调用 eq 函数的结果，即比较相等的结果
  Vectorized<int64_t> eq(const Vectorized<int64_t>& other) const;
  // 返回对当前 Vectorized 对象调用 ne 函数的结果，即比较不等的结果
  Vectorized<int64_t> ne(const Vectorized<int64_t>& other) const;
  // 返回对当前 Vectorized 对象调用 gt 函数的结果，即比较大于的结果
  Vectorized<int64_t> gt(const Vectorized<int64_t>& other) const;
  // 返回对当前 Vectorized 对象调用 ge 函数的结果，即比较大于等于的结果
  Vectorized<int64_t> ge(const Vectorized<int64_t>& other) const;
  // 返回对当前 Vectorized 对象调用 lt 函数的结果，即比较小于的结果
  Vectorized<int64_t> lt(const Vectorized<int64_t>& other) const;
  // 返回对当前 Vectorized 对象调用 le 函数的结果，即比较小于等于的结果
  Vectorized<int64_t> le(const Vectorized<int64_t>& other) const;
};


// 结束类模板 Vectorized<int32_t> 的定义

template <>
class Vectorized<int32_t> : public Vectorizedi {
private:
  // 静态成员变量，表示所有元素为1的 Vectorized<int32_t> 实例
  static const Vectorized<int32_t> ones;

public:
  // value_type 类型定义为 int32_t
  using value_type = int32_t;

  // 返回 Vectorized<int32_t> 类型对象的大小，固定返回值为8
  static constexpr int size() {
    return 8;
  }

  // 继承 Vectorizedi 类的构造函数
  using Vectorizedi::Vectorizedi;

  // 默认构造函数
  Vectorized() {}

  // 初始化所有元素为 v 的构造函数
  Vectorized(int32_t v) { values = _mm256_set1_epi32(v); }

  // 按照给定的 val1 到 val8 初始化成员变量 values
  Vectorized(int32_t val1, int32_t val2, int32_t val3, int32_t val4,
             int32_t val5, int32_t val6, int32_t val7, int32_t val8) {
    values = _mm256_setr_epi32(val1, val2, val3, val4, val5, val6, val7, val8);
  }

  // 根据掩码 mask 对 a 和 b 进行混合操作，返回混合后的 Vectorized<int32_t> 对象
  template <int64_t mask>
  static Vectorized<int32_t> blend(Vectorized<int32_t> a, Vectorized<int32_t> b) {
    return _mm256_blend_epi32(a, b, mask);
  }

  // 根据掩码 mask 对 a 和 b 进行逐元素混合操作，返回混合后的 Vectorized<int32_t> 对象
  static Vectorized<int32_t> blendv(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b,
                                    const Vectorized<int32_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }

  // 生成一组值，从 base 开始，步长为 step，返回对应的 Vectorized<int32_t> 对象
  template <typename step_t>
  static Vectorized<int32_t> arange(int32_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
  }

  // 根据给定的 count 设置 a 和 b 的值，返回对应的 Vectorized<int32_t> 对象
  static Vectorized<int32_t> set(Vectorized<int32_t> a, Vectorized<int32_t> b, int32_t count = size()) {
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

  // 从未对齐的内存地址 ptr 处加载数据，返回对应的 Vectorized<int32_t> 对象
  static Vectorized<int32_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }

  // 从未对齐的内存地址 ptr 处加载 count 个 int32_t 类型的数据，返回对应的 Vectorized<int32_t> 对象
  static Vectorized<int32_t> loadu(const void* ptr, int32_t count) {
    __at_align__ int32_t tmp_values[size()];
    // 确保未初始化的内存不会改变输出值，详细内容参见 https://github.com/pytorch/pytorch/issues/32502
    // 不使用"={0}"来初始化数组为零，因为 gcc 会将其编译成两条指令，而循环则会编译成一条指令。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, ptr, count * sizeof(int32_t));
    return loadu(tmp_values);
  }

  // 将当前对象的值存储到 ptr 指向的内存地址中，存储的元素个数为 count
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 这里不需要对 ptr 进行对齐处理。参见 https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
  Vectorized<int32_t> abs() const {
    // 返回该向量的绝对值向量
    return _mm256_abs_epi32(values);
  }
  Vectorized<int32_t> real() const {
    // 返回该向量的实部向量，实部即向量本身
    return *this;
  }
  Vectorized<int32_t> imag() const {
    // 返回该向量的虚部向量，虚部为全零向量
    return _mm256_set1_epi32(0);
  }
  Vectorized<int32_t> conj() const {
    // 返回该向量的共轭向量，共轭即向量本身
    return *this;
  }
  Vectorized<int32_t> neg() const;
  Vectorized<int32_t> operator==(const Vectorized<int32_t>& other) const {
    // 返回一个向量，其元素为对应位置上的比较结果是否相等
    return _mm256_cmpeq_epi32(values, other.values);
  }
  Vectorized<int32_t> operator!=(const Vectorized<int32_t>& other) const {
    // 返回一个向量，其元素为对应位置上的比较结果是否不相等
    return invert(_mm256_cmpeq_epi32(values, other.values));
  }
  Vectorized<int32_t> operator<(const Vectorized<int32_t>& other) const {
    // 返回一个向量，其元素为对应位置上的比较结果是否小于
    return _mm256_cmpgt_epi32(other.values, values);
  }
  Vectorized<int32_t> operator<=(const Vectorized<int32_t>& other) const {
    // 返回一个向量，其元素为对应位置上的比较结果是否小于等于
    return invert(_mm256_cmpgt_epi32(values, other.values));
  }
  Vectorized<int32_t> operator>(const Vectorized<int32_t>& other) const {
    // 返回一个向量，其元素为对应位置上的比较结果是否大于
    return _mm256_cmpgt_epi32(values, other.values);
  }
  Vectorized<int32_t> operator>=(const Vectorized<int32_t>& other) const {
    // 返回一个向量，其元素为对应位置上的比较结果是否大于等于
    return invert(_mm256_cmpgt_epi32(other.values, values));
  }
  Vectorized<int32_t> eq(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ne(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> gt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ge(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> lt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> le(const Vectorized<int32_t>& other) const;
};

// 特化模板，将 int32_t 数组转换为 float 数组
template <>
inline void convert(const int32_t *src, float *dst, int64_t n) {
  int64_t i;
  // int32_t 和 float 具有相同的大小

#ifndef _MSC_VER
# pragma unroll
#endif
  // 使用 SIMD 指令逐步处理 int32_t 数组并转换为 float 数组
  for (i = 0; i <= (n - Vectorized<int32_t>::size()); i += Vectorized<int32_t>::size()) {
    auto input_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    auto output_vec = _mm256_cvtepi32_ps(input_vec); // 将 int32_t 向量转换为 float 向量
    _mm256_storeu_ps(reinterpret_cast<float*>(dst + i), output_vec); // 存储转换后的 float 向量
  }

#ifndef _MSC_VER
# pragma unroll
#endif
  // 处理剩余部分，一次转换一个元素
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]); // 将 int32_t 转换为 float
  }
}

// 特化模板，将 int32_t 数组转换为 double 数组
template <>
inline void convert(const int32_t *src, double *dst, int64_t n) {
  int64_t i;
  // int32_t 的大小是 double 的一半

#ifndef _MSC_VER
# pragma unroll
#endif
  // 使用 SIMD 指令逐步处理 int32_t 数组并转换为 double 数组
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    auto input_128_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    auto output_vec = _mm256_cvtepi32_pd(input_128_vec); // 将 int32_t 向量转换为 double 向量
    _mm256_storeu_pd(reinterpret_cast<double*>(dst + i), output_vec); // 存储转换后的 double 向量
  }

#ifndef _MSC_VER
# pragma unroll
#endif
  // 处理剩余部分，一次转换一个元素
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]); // 将 int32_t 转换为 double
  }
}

// int16_t 类型的 SIMD 向量化模板类
template <>
class Vectorized<int16_t> : public Vectorizedi {
private:
  static const Vectorized<int16_t> ones; // 静态成员变量，表示所有元素为1的向量
public:
  using value_type = int16_t;
  static constexpr int size() {
    return 16; // 向量大小为16
  }
  using Vectorizedi::Vectorizedi; // 使用基类的构造函数
  Vectorized() {} // 默认构造函数
  // 根据给定的16个 int16_t 值构造向量
  Vectorized(int16_t val1, int16_t val2, int16_t val3, int16_t val4,
         int16_t val5, int16_t val6, int16_t val7, int16_t val8,
         int16_t val9, int16_t val10, int16_t val11, int16_t val12,
         int16_t val13, int16_t val14, int16_t val15, int16_t val16) {
    values = _mm256_setr_epi16(val1, val2, val3, val4, val5, val6, val7, val8,
                               val9, val10, val11, val12, val13, val14, val15, val16); // 设置向量的值
  }
  // 根据掩码 mask 混合两个 int16_t 向量 a 和 b 的元素
  template <int64_t mask>
  static Vectorized<int16_t> blend(Vectorized<int16_t> a, Vectorized<int16_t> b) {
    __at_align__ int16_t tmp_values[size()];
    a.store(tmp_values); // 将向量 a 的值存储到临时数组中
    // 根据掩码 mask 选择混合元素
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi16(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi16(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi16(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi16(b.values, 3);
    if (mask & 0x10)
      tmp_values[4] = _mm256_extract_epi16(b.values, 4);
    if (mask & 0x20)
      tmp_values[5] = _mm256_extract_epi16(b.values, 5);
    if (mask & 0x40)
      tmp_values[6] = _mm256_extract_epi16(b.values, 6);
    if (mask & 0x80)
      tmp_values[7] = _mm256_extract_epi16(b.values, 7);
    if (mask & 0x100)
      tmp_values[8] = _mm256_extract_epi16(b.values, 8);
    if (mask & 0x200)
      tmp_values[9] = _mm256_extract_epi16(b.values, 9);
    if (mask & 0x400)
      tmp_values[10] = _mm256_extract_epi16(b.values, 10);
    // 检查掩码中的每个位，如果为真，则从向量 b 中提取相应的值并存储在临时数组 tmp_values 中
    if (mask & 0x800)
      tmp_values[11] = _mm256_extract_epi16(b.values, 11);
    if (mask & 0x1000)
      tmp_values[12] = _mm256_extract_epi16(b.values, 12);
    if (mask & 0x2000)
      tmp_values[13] = _mm256_extract_epi16(b.values, 13);
    if (mask & 0x4000)
      tmp_values[14] = _mm256_extract_epi16(b.values, 14);
    if (mask & 0x8000)
      tmp_values[15] = _mm256_extract_epi16(b.values, 15);
    // 将临时数组 tmp_values 加载到向量并返回
    return loadu(tmp_values);
  }

  // 使用掩码 mask 对向量 a 和 b 进行混合操作，返回混合后的向量
  static Vectorized<int16_t> blendv(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b,
                                const Vectorized<int16_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }

  // 创建一个包含从 base 开始、步长为 step 的等差数列的向量
  template <typename step_t>
  static Vectorized<int16_t> arange(int16_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int16_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }

  // 根据指定的 count 值对向量 a 和 b 进行设置操作，返回设置后的向量
  static Vectorized<int16_t>
  set(Vectorized<int16_t> a, Vectorized<int16_t> b, int16_t count = size()) {
    switch (count) {
      // 根据 count 的不同值调用不同的 blend 模板函数进行混合操作
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
    // 默认返回向量 b
    return b;
  }

  // 从指针 ptr 处加载 int16_t 类型数据到向量中并返回
  static Vectorized<int16_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }

  // 从指针 ptr 处加载 count 个 int16_t 类型数据到向量中并返回
  static Vectorized<int16_t> loadu(const void* ptr, int16_t count) {
    __at_align__ int16_t tmp_values[size()];
    // 确保未初始化的内存不会改变输出值。参考 https://github.com/pytorch/pytorch/issues/32502
    // 详细信息。我们不使用 "={0}" 来初始化数组为零，因为 gcc 会将其编译成两条指令，而使用循环只需要一条指令。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    // 将 ptr 处开始的 count 个 int16_t 数据复制到 tmp_values 中
    std::memcpy(tmp_values, ptr, count * sizeof(int16_t));
    // 调用 loadu 函数加载 tmp_values 到向量中并返回
    return loadu(tmp_values);
  }
    if (count == size()) {
      // 如果 count 等于 size()，则说明要存储的元素数量与向量大小相等
      // 此时 ptr 不需要对齐。参考链接：
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      // 否则如果 count 大于 0，则需要处理部分元素
      __at_align__ int16_t tmp_values[size()];
      // 将 AVX 寄存器中的值存储到临时数组 tmp_values 中
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      // 使用 memcpy 将临时数组中的数据拷贝到 ptr 指向的地址，拷贝 count 个元素的大小
      std::memcpy(ptr, tmp_values, count * sizeof(int16_t));
    }
  }
  const int16_t& operator[](int idx) const  = delete;
  int16_t& operator[](int idx)  = delete;
  // 返回当前向量的绝对值向量
  Vectorized<int16_t> abs() const {
    return _mm256_abs_epi16(values);
  }
  // 返回当前向量的实部，即自身
  Vectorized<int16_t> real() const {
    return *this;
  }
  // 返回当前向量的虚部，即全为零的向量
  Vectorized<int16_t> imag() const {
    return _mm256_set1_epi16(0);
  }
  // 返回当前向量的共轭，即自身
  Vectorized<int16_t> conj() const {
    return *this;
  }
  // 返回当前向量的负向量
  Vectorized<int16_t> neg() const;
  // 检查当前向量与另一个向量是否相等，并返回比较结果
  Vectorized<int16_t> operator==(const Vectorized<int16_t>& other) const {
    return _mm256_cmpeq_epi16(values, other.values);
  }
  // 检查当前向量与另一个向量是否不相等，并返回比较结果
  Vectorized<int16_t> operator!=(const Vectorized<int16_t>& other) const {
    return invert(_mm256_cmpeq_epi16(values, other.values));
  }
  // 检查当前向量是否小于另一个向量，并返回比较结果
  Vectorized<int16_t> operator<(const Vectorized<int16_t>& other) const {
    return _mm256_cmpgt_epi16(other.values, values);
  }
  // 检查当前向量是否小于等于另一个向量，并返回比较结果
  Vectorized<int16_t> operator<=(const Vectorized<int16_t>& other) const {
    return invert(_mm256_cmpgt_epi16(values, other.values));
  }
  // 检查当前向量是否大于另一个向量，并返回比较结果
  Vectorized<int16_t> operator>(const Vectorized<int16_t>& other) const {
    return _mm256_cmpgt_epi16(values, other.values);
  }
  // 检查当前向量是否大于等于另一个向量，并返回比较结果
  Vectorized<int16_t> operator>=(const Vectorized<int16_t>& other) const {
    return invert(_mm256_cmpgt_epi16(other.values, values));
  }

  // 以下几个函数声明未提供实现，分别用于比较当前向量与另一个向量的大小关系
  Vectorized<int16_t> eq(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> ne(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> gt(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> ge(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> lt(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> le(const Vectorized<int16_t>& other) const;
};

template <typename T>
class Vectorized8 : public Vectorizedi {
  // 检查模板参数 T 是否为 int8_t 或 uint8_t，否则编译时报错
  static_assert(
    std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
    "Only int8_t/uint8_t are supported");
protected:
  // 声明静态成员变量 ones，用于存储全为 1 的向量
  static const Vectorized<T> ones;
public:
  // 定义 value_type 作为 T 的别名
  using value_type = T;
  // 返回向量的大小，这里固定为 32
  static constexpr int size() {
    return 32;
  }
  // 继承 Vectorizedi 类的构造函数
  using Vectorizedi::Vectorizedi;
  // 默认构造函数，未做额外的初始化操作
  Vectorized8() {}
  // 构造函数，使用给定的值 v 初始化所有向量元素
  Vectorized8(T v) { values = _mm256_set1_epi8(v); }
  // 构造函数，使用给定的 32 个值初始化向量
  Vectorized8(T val1, T val2, T val3, T val4,
         T val5, T val6, T val7, T val8,
         T val9, T val10, T val11, T val12,
         T val13, T val14, T val15, T val16,
         T val17, T val18, T val19, T val20,
         T val21, T val22, T val23, T val24,
         T val25, T val26, T val27, T val28,
         T val29, T val30, T val31, T val32) {
    // 使用给定的值构造一个向量
    values = _mm256_setr_epi8(val1, val2, val3, val4, val5, val6, val7, val8,
                              val9, val10, val11, val12, val13, val14, val15, val16,
                              val17, val18, val19, val20, val21, val22, val23, val24,
                              val25, val26, val27, val28, val29, val30, val31, val32);
  }
  // 模板函数，根据掩码 mask 来混合两个向量 a 和 b 的元素
  template <int64_t mask>
  static Vectorized<T> blend(Vectorized<T> a, Vectorized<T> b) {
    // 创建一个临时数组 tmp_values 来存储向量元素
    __at_align__ T tmp_values[size()];
    // 将向量 a 的值存储到 tmp_values 数组中
    a.store(tmp_values);
    // 根据掩码 mask 分别将向量 b 的部分元素混合到 tmp_values 数组中
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi8(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi8(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi8(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi8(b.values, 3);
    if (mask & 0x10)
      tmp_values[4] = _mm256_extract_epi8(b.values, 4);
    if (mask & 0x20)
      tmp_values[5] = _mm256_extract_epi8(b.values, 5);
    if (mask & 0x40)
      tmp_values[6] = _mm256_extract_epi8(b.values, 6);
    if (mask & 0x80)
      tmp_values[7] = _mm256_extract_epi8(b.values, 7);
    if (mask & 0x100)
      tmp_values[8] = _mm256_extract_epi8(b.values, 8);
    if (mask & 0x200)
      tmp_values[9] = _mm256_extract_epi8(b.values, 9);
    if (mask & 0x400)
      tmp_values[10] = _mm256_extract_epi8(b.values, 10);
    if (mask & 0x800)
      tmp_values[11] = _mm256_extract_epi8(b.values, 11);
    if (mask & 0x1000)
      tmp_values[12] = _mm256_extract_epi8(b.values, 12);
    if (mask & 0x2000)
      tmp_values[13] = _mm256_extract_epi8(b.values, 13);
    if (mask & 0x4000)
      tmp_values[14] = _mm256_extract_epi8(b.values, 14);
    if (mask & 0x8000)
      tmp_values[15] = _mm256_extract_epi8(b.values, 15);
    if (mask & 0x010000)
      tmp_values[16] = _mm256_extract_epi8(b.values, 16);
    if (mask & 0x020000)
      tmp_values[17] = _mm256_extract_epi8(b.values, 17);
    if (mask & 0x040000)
      tmp_values[18] = _mm256_extract_epi8(b.values, 18);
    if (mask & 0x080000)
      tmp_values[19] = _mm256_extract_epi8(b.values, 19);
    if (mask & 0x100000)
      tmp_values[20] = _mm256_extract_epi8(b.values, 20);
    # 检查掩码中的每个位，判断是否需要从 b.values 中提取值并存储到 tmp_values 数组中对应的位置
    if (mask & 0x200000)
      tmp_values[21] = _mm256_extract_epi8(b.values, 21);
    if (mask & 0x400000)
      tmp_values[22] = _mm256_extract_epi8(b.values, 22);
    if (mask & 0x800000)
      tmp_values[23] = _mm256_extract_epi8(b.values, 23);
    if (mask & 0x1000000)
      tmp_values[24] = _mm256_extract_epi8(b.values, 24);
    if (mask & 0x2000000)
      tmp_values[25] = _mm256_extract_epi8(b.values, 25);
    if (mask & 0x4000000)
      tmp_values[26] = _mm256_extract_epi8(b.values, 26);
    if (mask & 0x8000000)
      tmp_values[27] = _mm256_extract_epi8(b.values, 27);
    if (mask & 0x10000000)
      tmp_values[28] = _mm256_extract_epi8(b.values, 28);
    if (mask & 0x20000000)
      tmp_values[29] = _mm256_extract_epi8(b.values, 29);
    if (mask & 0x40000000)
      tmp_values[30] = _mm256_extract_epi8(b.values, 30);
    if (mask & 0x80000000)
      tmp_values[31] = _mm256_extract_epi8(b.values, 31);
    # 返回从 tmp_values 数组中加载的向量
    return loadu(tmp_values);
  }
  # 使用掩码 mask 对 a 和 b 的值进行混合操作，返回混合后的向量
  static Vectorized<T> blendv(const Vectorized<T>& a, const Vectorized<T>& b,
                               const Vectorized<T>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  # 创建一个以步长 step_t 递增的从 base 开始的序列，长度为 32 的向量
  template <typename step_t>
  static Vectorized<T> arange(T base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
      base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
      base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
      base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
      base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step);
  }
  # 将向量 a 的元素设置为向量 b 的元素，长度为 size()（默认情况下）的向量
  static Vectorized<T>
  set(Vectorized<T> a, Vectorized<T> b, T count = size()) {
    // 根据 count 的值选择不同的混合模式进行返回
    switch (count) {
      // 如果 count 为 0，返回 a
      case 0:
        return a;
      // 如果 count 为 1，使用 blend<0x1> 模式混合 a 和 b 后返回
      case 1:
        return blend<0x1>(a, b);
      // 如果 count 为 2，使用 blend<0x3> 模式混合 a 和 b 后返回
      case 2:
        return blend<0x3>(a, b);
      // 依此类推，每个 case 语句对应不同的混合模式
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
    // 如果 count 超出了范围（大于 31），则直接返回 b
    return b;
  }
  
  // 加载给定地址 ptr 的向量数据并返回
  static Vectorized<T> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  
  // 加载给定地址 ptr 的向量数据的低 128 位并返回为 256 位向量
  static Vectorized<T> loadu_one_fourth(const void* ptr) {
      // 如果仅加载了 8 个元素，则使用快速路径
      // 注意：我们没有将其合并为 loadu(const void* ptr, T count) 的快速路径，
      // 因为 loadu(const void* ptr, T count) 需要对上128位进行零初始化。
      // 然而，通过使用 _mm256_castsi128_si256，结果的上128位是未定义的。
      // TODO<leslie> 在将来我们可以使用 _mm256_zextsi128_si256，
      // 因为目前 gcc 9.3 不支持这一点。
      __m128i input_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
      return _mm256_castsi128_si256(input_128);
  }
  
  // 加载给定地址 ptr 开始的 count 个元素的向量数据并返回
  static Vectorized<T> loadu(const void* ptr, T count) {
    __at_align__ T tmp_values[size()];
    // 确保未初始化的内存不会改变输出值，详细信息请参见 https://github.com/pytorch/pytorch/issues/32502
    // 我们不使用 "= {0}" 来将数组初始化为零，因为 gcc 会将其编译为两条指令，而使用循环只需一条指令。
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    // 使用 memcpy 将 ptr 处开始的 count 个元素拷贝到 tmp_values 中
    std::memcpy(tmp_values, ptr, count * sizeof(T));
    // 调用 loadu 函数加载 tmp_values 数组内容，并返回结果
    return loadu(tmp_values);
  }
  // 存储函数，将向量数据存储到指针 ptr 所指向的内存位置，元素个数为 count，默认为 size()
  void store(void* ptr, int count = size()) const {
    // 如果 count 等于 size()，则执行以下操作
    if (count == size()) {
      // ptr 在此处无需对齐。参考 Intel AVX 存储操作文档
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      // 如果 count 大于 0，则根据 count 的不同值执行不同的存储操作
      if (count == 8) {
        // 如果只存储 8 个元素，采用快速路径
        _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr), _mm256_castsi256_si128(values));
      } else {
        // 否则，创建临时数组 tmp_values，将向量数据存储到 tmp_values 中，然后使用 memcpy 将数据复制到 ptr 指向的内存位置
        __at_align__ T tmp_values[size()];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
        std::memcpy(ptr, tmp_values, count * sizeof(T));
      }
    }
  }
  // 禁用 [] 运算符的 const 版本
  const T& operator[](int idx) const  = delete;
  // 禁用 [] 运算符的非 const 版本
  T& operator[](int idx)  = delete;
  // 返回当前对象的实部，即返回自身
  Vectorized<T> real() const {
    return *this;
  }
  // 返回当前对象的虚部，使用 _mm256_set1_epi8(0) 返回全 0 向量
  Vectorized<T> imag() const {
    return _mm256_set1_epi8(0);
  }
  // 返回当前对象的共轭，即返回自身
  Vectorized<T> conj() const {
    return *this;
  }
};

template<>
class Vectorized<int8_t>: public Vectorized8<int8_t> {
public:
  using Vectorized8::Vectorized8;

  // 返回当前向量的负数向量
  Vectorized<int8_t> neg() const;

  // 返回当前向量的绝对值向量
  Vectorized<int8_t> abs() const {
   return _mm256_abs_epi8(values);
  }

  // 判断当前向量是否等于另一个向量
  Vectorized<int8_t> operator==(const Vectorized<int8_t>& other) const {
    return _mm256_cmpeq_epi8(values, other.values);
  }

  // 判断当前向量是否不等于另一个向量
  Vectorized<int8_t> operator!=(const Vectorized<int8_t>& other) const {
    return invert(_mm256_cmpeq_epi8(values, other.values));
  }

  // 判断当前向量是否小于另一个向量
  Vectorized<int8_t> operator<(const Vectorized<int8_t>& other) const {
    return _mm256_cmpgt_epi8(other.values, values);
  }

  // 判断当前向量是否小于等于另一个向量
  Vectorized<int8_t> operator<=(const Vectorized<int8_t>& other) const {
    return invert(_mm256_cmpgt_epi8(values, other.values));
  }

  // 判断当前向量是否大于另一个向量
  Vectorized<int8_t> operator>(const Vectorized<int8_t>& other) const {
    return other < *this;
  }

  // 判断当前向量是否大于等于另一个向量
  Vectorized<int8_t> operator>=(const Vectorized<int8_t>& other) const {
    return other <= *this;
  }

  // 比较当前向量与另一个向量是否相等
  Vectorized<int8_t> eq(const Vectorized<int8_t>& other) const;

  // 比较当前向量与另一个向量是否不相等
  Vectorized<int8_t> ne(const Vectorized<int8_t>& other) const;

  // 比较当前向量是否大于另一个向量
  Vectorized<int8_t> gt(const Vectorized<int8_t>& other) const;

  // 比较当前向量是否大于等于另一个向量
  Vectorized<int8_t> ge(const Vectorized<int8_t>& other) const;

  // 比较当前向量是否小于另一个向量
  Vectorized<int8_t> lt(const Vectorized<int8_t>& other) const;

  // 比较当前向量是否小于等于另一个向量
  Vectorized<int8_t> le(const Vectorized<int8_t>& other) const;
};

template<>
class Vectorized<uint8_t>: public Vectorized8<uint8_t> {
public:
  using Vectorized8::Vectorized8;

  // 返回当前向量的负数向量
  Vectorized<uint8_t> neg() const;

  // 返回当前向量的绝对值向量，对于无符号整数向量与自身相同
  Vectorized<uint8_t> abs() const {
    return *this;
  }

  // 判断当前向量是否等于另一个向量
  Vectorized<uint8_t> operator==(const Vectorized<uint8_t>& other) const {
    return _mm256_cmpeq_epi8(values, other.values);
  }

  // 判断当前向量是否不等于另一个向量
  Vectorized<uint8_t> operator!=(const Vectorized<uint8_t>& other) const {
    return invert(_mm256_cmpeq_epi8(values, other.values));
  }

  // 判断当前向量是否小于另一个向量
  Vectorized<uint8_t> operator<(const Vectorized<uint8_t>& other) const {
    // 取当前向量与另一个向量的最大值，然后判断最大值是否等于当前向量，返回相反值
    __m256i max = _mm256_max_epu8(values, other.values);
    return invert(_mm256_cmpeq_epi8(max, values));
  }

  // 判断当前向量是否小于等于另一个向量
  Vectorized<uint8_t> operator<=(const Vectorized<uint8_t>& other) const {
    // 取当前向量与另一个向量的最大值，然后判断最大值是否等于另一个向量
    __m256i max = _mm256_max_epu8(values, other.values);
    return _mm256_cmpeq_epi8(max, other.values);
  }

  // 判断当前向量是否大于另一个向量
  Vectorized<uint8_t> operator>(const Vectorized<uint8_t>& other) const {
    return other < *this;
  }

  // 判断当前向量是否大于等于另一个向量
  Vectorized<uint8_t> operator>=(const Vectorized<uint8_t>& other) const {
    return other <= *this;
  }

  // 比较当前向量与另一个向量是否相等
  Vectorized<uint8_t> eq(const Vectorized<uint8_t>& other) const;

  // 比较当前向量与另一个向量是否不相等
  Vectorized<uint8_t> ne(const Vectorized<uint8_t>& other) const;

  // 比较当前向量是否大于另一个向量
  Vectorized<uint8_t> gt(const Vectorized<uint8_t>& other) const;

  // 比较当前向量是否大于等于另一个向量
  Vectorized<uint8_t> ge(const Vectorized<uint8_t>& other) const;

  // 比较当前向量是否小于另一个向量
  Vectorized<uint8_t> lt(const Vectorized<uint8_t>& other) const;

  // 比较当前向量是否小于等于另一个向量
  Vectorized<uint8_t> le(const Vectorized<uint8_t>& other) const;
};

template <>
// 实现两个 int64_t 类型向量的加法
Vectorized<int64_t> inline operator+(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm256_add_epi64(a, b);
}

template <>
// 定义整型向量加法运算符重载，使用 AVX 指令集进行向量加法
Vectorized<int32_t> inline operator+(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_add_epi32(a, b);
}

// 特化模板：定义短整型向量加法运算符重载，使用 AVX 指令集进行向量加法
template <>
Vectorized<int16_t> inline operator+(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm256_add_epi16(a, b);
}

// 特化模板：定义字符型向量加法运算符重载，使用 AVX 指令集进行向量加法
template <>
Vectorized<int8_t> inline operator+(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm256_add_epi8(a, b);
}

// 特化模板：定义无符号字符型向量加法运算符重载，使用 AVX 指令集进行向量加法
template <>
Vectorized<uint8_t> inline operator+(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm256_add_epi8(a, b);
}

// 特化模板：定义长整型向量减法运算符重载，使用 AVX 指令集进行向量减法
template <>
Vectorized<int64_t> inline operator-(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm256_sub_epi64(a, b);
}

// 特化模板：定义整型向量减法运算符重载，使用 AVX 指令集进行向量减法
template <>
Vectorized<int32_t> inline operator-(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_sub_epi32(a, b);
}

// 特化模板：定义短整型向量减法运算符重载，使用 AVX 指令集进行向量减法
template <>
Vectorized<int16_t> inline operator-(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm256_sub_epi16(a, b);
}

// 特化模板：定义字符型向量减法运算符重载，使用 AVX 指令集进行向量减法
template <>
Vectorized<int8_t> inline operator-(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm256_sub_epi8(a, b);
}

// 特化模板：定义无符号字符型向量减法运算符重载，使用 AVX 指令集进行向量减法
template <>
Vectorized<uint8_t> inline operator-(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm256_sub_epi8(a, b);
}

// 定义长整型向量的取负操作，通过减去当前向量本身实现
inline Vectorized<int64_t> Vectorized<int64_t>::neg() const {
  return Vectorized<int64_t>(0) - *this;
}

// 定义整型向量的取负操作，通过减去当前向量本身实现
inline Vectorized<int32_t> Vectorized<int32_t>::neg() const {
  return Vectorized<int32_t>(0) - *this;
}

// 定义短整型向量的取负操作，通过减去当前向量本身实现
inline Vectorized<int16_t> Vectorized<int16_t>::neg() const {
  return Vectorized<int16_t>(0) - *this;
}

// 定义字符型向量的取负操作，通过减去当前向量本身实现
inline Vectorized<int8_t> Vectorized<int8_t>::neg() const {
  return Vectorized<int8_t>(0) - *this;
}

// 定义无符号字符型向量的取负操作，通过减去当前向量本身实现
inline Vectorized<uint8_t> Vectorized<uint8_t>::neg() const {
  return Vectorized<uint8_t>(0) - *this;
}

// 模拟操作，用于处理 AVX 中无法直接支持的 64 位整数运算，逐个元素进行操作后合并结果为向量
template <typename op_t>
Vectorized<int64_t> inline emulate(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b, const op_t& op) {
  // 提取向量 a 中的每个元素到标量变量
  int64_t a0 = _mm256_extract_epi64(a, 0);
  int64_t a1 = _mm256_extract_epi64(a, 1);
  int64_t a2 = _mm256_extract_epi64(a, 2);
  int64_t a3 = _mm256_extract_epi64(a, 3);

  // 提取向量 b 中的每个元素到标量变量
  int64_t b0 = _mm256_extract_epi64(b, 0);
  int64_t b1 = _mm256_extract_epi64(b, 1);
  int64_t b2 = _mm256_extract_epi64(b, 2);
  int64_t b3 = _mm256_extract_epi64(b, 3);

  // 使用 op 函数逐个元素进行操作
  int64_t c0 = op(a0, b0);
  int64_t c1 = op(a1, b1);
  int64_t c2 = op(a2, b2);
  int64_t c3 = op(a3, b3);

  // 将操作结果合并为一个 AVX 向量并返回
  return _mm256_set_epi64x(c3, c2, c1, c0);
}

// 模板函数，用于模拟 AVX 不支持的操作，操作类型由模板参数 op_t 指定
template <typename op_t>
// 定义一个函数 `emulate`，用于模拟 AVX2 下 int64_t 类型的向量乘法操作
Vectorized<int64_t> inline emulate(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b, const Vectorized<int64_t>& c, const op_t& op) {
  // 从向量 `a` 中提取第 0、1、2、3 个元素
  int64_t a0 = _mm256_extract_epi64(a, 0);
  int64_t a1 = _mm256_extract_epi64(a, 1);
  int64_t a2 = _mm256_extract_epi64(a, 2);
  int64_t a3 = _mm256_extract_epi64(a, 3);

  // 从向量 `b` 中提取第 0、1、2、3 个元素
  int64_t b0 = _mm256_extract_epi64(b, 0);
  int64_t b1 = _mm256_extract_epi64(b, 1);
  int64_t b2 = _mm256_extract_epi64(b, 2);
  int64_t b3 = _mm256_extract_epi64(b, 3);

  // 从向量 `c` 中提取第 0、1、2、3 个元素
  int64_t c0 = _mm256_extract_epi64(c, 0);
  int64_t c1 = _mm256_extract_epi64(c, 1);
  int64_t c2 = _mm256_extract_epi64(c, 2);
  int64_t c3 = _mm256_extract_epi64(c, 3);

  // 调用指定的操作 `op` 对各个元素进行计算
  int64_t d0 = op(a0, b0, c0);
  int64_t d1 = op(a1, b1, c1);
  int64_t d2 = op(a2, b2, c2);
  int64_t d3 = op(a3, b3, c3);

  // 将计算结果重新组合成 AVX 向量并返回
  return _mm256_set_epi64x(d3, d2, d1, d0);
}

// 对于 int64_t 类型的向量乘法运算符的特化，使用 `emulate` 函数实现
// 这里处理 AVX2 中没有的 int64_t 类型的乘法操作
template <>
Vectorized<int64_t> inline operator*(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  // 使用 lambda 函数调用 `emulate` 函数，传入乘法操作
  return emulate(a, b, [](int64_t a_point, int64_t b_point) __ubsan_ignore_undefined__ {return a_point * b_point;});
}

// 对于 int32_t 类型的向量乘法运算符的特化，直接使用 AVX2 中的 `_mm256_mullo_epi32` 函数
template <>
Vectorized<int32_t> inline operator*(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_mullo_epi32(a, b);
}

// 对于 int16_t 类型的向量乘法运算符的特化，直接使用 AVX2 中的 `_mm256_mullo_epi16` 函数
template <>
Vectorized<int16_t> inline operator*(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm256_mullo_epi16(a, b);
}

// 泛化版本的向量元素二元操作函数，处理任意类型 T 的向量和操作 Op
template <typename T, typename Op>
Vectorized<T> inline int_elementwise_binary_256(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
  // 创建两个数组，用于暂存向量 `a` 和 `b` 中的元素
  T values_a[Vectorized<T>::size()];
  T values_b[Vectorized<T>::size()];
  // 将向量 `a` 和 `b` 中的元素存储到数组中
  a.store(values_a);
  b.store(values_b);
  // 对数组中的元素逐个应用操作 `op`
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    values_a[i] = op(values_a[i], values_b[i]);
  }
  // 将处理后的数组重新加载为向量并返回
  return Vectorized<T>::loadu(values_a);
}

// 对于 int8_t 类型的向量乘法运算符的特化
template <>
Vectorized<int8_t> inline operator*(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  // 如果不支持 AVX2，使用 `int_elementwise_binary_256` 函数逐元素乘法
#ifndef CPU_CAPABILITY_AVX2
  return int_elementwise_binary_256(a, b, std::multiplies<int8_t>());
#else
  // 使用 AVX2 指令实现 int8_t 类型向量的乘法
  __m256i mask00FF = _mm256_set1_epi16(0x00FF);
  __m256i a_lo = _mm256_srai_epi16(_mm256_slli_epi16(a, 8), 8);
  __m256i b_lo = _mm256_srai_epi16(_mm256_slli_epi16(b, 8), 8);
  __m256i a_hi = _mm256_srai_epi16(a, 8);
  __m256i b_hi = _mm256_srai_epi16(b, 8);
  __m256i res_lo = _mm256_and_si256(_mm256_mullo_epi16(a_lo, b_lo), mask00FF);
  __m256i res_hi = _mm256_slli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
  __m256i res = _mm256_or_si256(res_hi, res_lo);
  return res;
#endif
}

// 对于 uint8_t 类型的向量乘法运算符的特化
template <>
Vectorized<uint8_t> inline operator*(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  // 对于 uint8_t 类型，同样使用 `int_elementwise_binary_256` 函数逐元素乘法
#ifndef CPU_CAPABILITY_AVX2
  return int_elementwise_binary_256(a, b, std::multiplies<uint8_t>());
#else
  // 这里可以添加 AVX2 指令实现，但在注释中没有提供具体实现
  // ...
#endif
}
#ifndef CPU_CAPABILITY_AVX2
  // 如果不支持 AVX2，使用标准库函数执行 int 元素级别的二进制操作并返回结果
  return int_elementwise_binary_256(a, b, std::multiplies<uint8_t>());
#else
  // 创建掩码，用于提取低位字节（0xFF）并将其应用到向量 a 和 b
  __m256i mask00FF = _mm256_set1_epi16(0x00FF);
  // 提取 a 的低位字节
  __m256i a_lo = _mm256_and_si256 (a, mask00FF);
  // 提取 b 的低位字节
  __m256i b_lo = _mm256_and_si256 (b, mask00FF);
  // 将 a 右移 8 位，获取高位字节
  __m256i a_hi = _mm256_srli_epi16(a, 8);
  // 将 b 右移 8 位，获取高位字节
  __m256i b_hi = _mm256_srli_epi16(b, 8);
  // 计算低位字节乘积后与掩码相与，并存储在 res_lo 中
  __m256i res_lo = _mm256_and_si256(_mm256_mullo_epi16(a_lo, b_lo), mask00FF);
  // 计算高位字节乘积后左移 8 位，并存储在 res_hi 中
  __m256i res_hi = _mm256_slli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
  // 将高位和低位结果进行或运算得到最终结果 res
  __m256i res = _mm256_or_si256(res_hi, res_lo);
  // 返回计算结果
  return res;
#endif
}

template <>
Vectorized<int64_t> inline minimum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
#ifndef CPU_CAPABILITY_AVX2
  // 如果不支持 AVX2，模拟并返回 a 和 b 中每个元素的最小值向量
  return emulate(a, b, [](int64_t a_point, int64_t b_point) {return std::min(a_point, b_point);});
#else
  // 使用 AVX2 指令集进行比较，生成比较结果向量 cmp
  __m256i cmp = _mm256_cmpgt_epi64(a, b);
  // 根据比较结果选择最小值，并返回结果向量
  return _mm256_blendv_epi8(a, b, cmp);
#endif
}

template <>
Vectorized<int32_t> inline minimum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  // 使用 AVX2 指令集返回 a 和 b 中每个元素的最小值向量
  return _mm256_min_epi32(a, b);
}

template <>
Vectorized<int16_t> inline minimum(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  // 使用 AVX2 指令集返回 a 和 b 中每个元素的最小值向量
  return _mm256_min_epi16(a, b);
}

template <>
Vectorized<int8_t> inline minimum(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  // 使用 AVX2 指令集返回 a 和 b 中每个元素的最小值向量
  return _mm256_min_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline minimum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  // 使用 AVX2 指令集返回 a 和 b 中每个元素的无符号最小值向量
  return _mm256_min_epu8(a, b);
}

template <>
Vectorized<int64_t> inline maximum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
#ifndef CPU_CAPABILITY_AVX2
  // 如果不支持 AVX2，模拟并返回 a 和 b 中每个元素的最大值向量
  return emulate(a, b, [](int64_t a_point, int64_t b_point) {return std::max(a_point, b_point);});
#else
  // 使用 AVX2 指令集进行比较，生成比较结果向量 cmp
  __m256i cmp = _mm256_cmpgt_epi64(a, b);
  // 根据比较结果选择最大值，并返回结果向量
  return _mm256_blendv_epi8(b, a, cmp);
#endif
}

template <>
Vectorized<int32_t> inline maximum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  // 使用 AVX2 指令集返回 a 和 b 中每个元素的最大值向量
  return _mm256_max_epi32(a, b);
}

template <>
Vectorized<int16_t> inline maximum(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  // 使用 AVX2 指令集返回 a 和 b 中每个元素的最大值向量
  return _mm256_max_epi16(a, b);
}

template <>
Vectorized<int8_t> inline maximum(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  // 使用 AVX2 指令集返回 a 和 b 中每个元素的最大值向量
  return _mm256_max_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline maximum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  // 使用 AVX2 指令集返回 a 和 b 中每个元素的无符号最大值向量
  return _mm256_max_epu8(a, b);
}

template <>
Vectorized<int64_t> inline clamp(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val, const Vectorized<int64_t>& max_val) {
#ifndef CPU_CAPABILITY_AVX2
  // 如果不支持 AVX2，模拟并返回 a 和 min_val、max_val 中每个元素的 clamp 值向量
  return emulate(a, min_val, max_val, [](int64_t a_point, int64_t min_point, int64_t max_point) {return std::min(max_point, std::max(a_point, min_point));});
#else
  // 使用 AVX2 指令集实现 clamp 操作，先计算 a 和 min_val 的最大值，再与 max_val 求最小值
  return minimum(maximum(a, min_val), max_val);
#endif
}

template <>
Vectorized<int32_t> inline clamp(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val, const Vectorized<int32_t>& max_val) {
  // 使用 AVX2 指令集实现 clamp 操作，先计算 a 和 min_val 的最大值，再与 max_val 求最小值
  return _mm256_min_epi32(max_val, _mm256_max_epi32(a, min_val));
}
template <>
// 定义模板特化函数 clamp，用于限制 int16_t 类型向量的取值范围
Vectorized<int16_t> inline clamp(const Vectorized<int16_t>& a, const Vectorized<int16_t>& min_val, const Vectorized<int16_t>& max_val) {
  // 使用 AVX2 指令集中的 _mm256_min_epi16 和 _mm256_max_epi16 函数进行向量化最小值和最大值的计算
  return _mm256_min_epi16(max_val, _mm256_max_epi16(a, min_val));
}

template <>
// 定义模板特化函数 clamp，用于限制 int8_t 类型向量的取值范围
Vectorized<int8_t> inline clamp(const Vectorized<int8_t>& a, const Vectorized<int8_t>& min_val, const Vectorized<int8_t>& max_val) {
  // 使用 AVX2 指令集中的 _mm256_min_epi8 和 _mm256_max_epi8 函数进行向量化最小值和最大值的计算
  return _mm256_min_epi8(max_val, _mm256_max_epi8(a, min_val));
}

template <>
// 定义模板特化函数 clamp，用于限制 uint8_t 类型向量的取值范围
Vectorized<uint8_t> inline clamp(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& min_val, const Vectorized<uint8_t>& max_val) {
  // 使用 AVX2 指令集中的 _mm256_min_epu8 和 _mm256_max_epu8 函数进行向量化无符号最小值和最大值的计算
  return _mm256_min_epu8(max_val, _mm256_max_epu8(a, min_val));
}

template <>
// 定义模板特化函数 clamp_max，用于限制 int64_t 类型向量的最大值
Vectorized<int64_t> inline clamp_max(const Vectorized<int64_t>& a, const Vectorized<int64_t>& max_val) {
  // 根据 CPU 支持的指令集选择不同的实现路径
#ifndef CPU_CAPABILITY_AVX2
  // 如果不支持 AVX2，则调用 emulate 函数模拟实现
  return emulate(a, max_val, [](int64_t a_point, int64_t max_point) {return std::min(max_point, a_point);});
#else
  // 如果支持 AVX2，则调用 minimum 函数进行向量化最小值计算
  return minimum(max_val, a);
#endif
}

template <>
// 定义模板特化函数 clamp_max，用于限制 int32_t 类型向量的最大值
Vectorized<int32_t> inline clamp_max(const Vectorized<int32_t>& a, const Vectorized<int32_t>& max_val) {
  // 使用 AVX2 指令集中的 _mm256_min_epi32 函数进行向量化最小值的计算
  return _mm256_min_epi32(max_val, a);
}

template <>
// 定义模板特化函数 clamp_max，用于限制 int16_t 类型向量的最大值
Vectorized<int16_t> inline clamp_max(const Vectorized<int16_t>& a, const Vectorized<int16_t>& max_val) {
  // 使用 AVX2 指令集中的 _mm256_min_epi16 函数进行向量化最小值的计算
  return _mm256_min_epi16(max_val, a);
}

template <>
// 定义模板特化函数 clamp_max，用于限制 int8_t 类型向量的最大值
Vectorized<int8_t> inline clamp_max(const Vectorized<int8_t>& a, const Vectorized<int8_t>& max_val) {
  // 使用 AVX2 指令集中的 _mm256_min_epi8 函数进行向量化最小值的计算
  return _mm256_min_epi8(max_val, a);
}

template <>
// 定义模板特化函数 clamp_max，用于限制 uint8_t 类型向量的最大值
Vectorized<uint8_t> inline clamp_max(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& max_val) {
  // 使用 AVX2 指令集中的 _mm256_min_epu8 函数进行向量化无符号最小值的计算
  return _mm256_min_epu8(max_val, a);
}

template <>
// 定义模板特化函数 clamp_min，用于限制 int64_t 类型向量的最小值
Vectorized<int64_t> inline clamp_min(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val) {
  // 根据 CPU 支持的指令集选择不同的实现路径
#ifndef CPU_CAPABILITY_AVX2
  // 如果不支持 AVX2，则调用 emulate 函数模拟实现
  return emulate(a, min_val, [](int64_t a_point, int64_t min_point) {return std::max(min_point, a_point);});
#else
  // 如果支持 AVX2，则调用 maximum 函数进行向量化最大值计算
  return maximum(min_val, a);
#endif
}

template <>
// 定义模板特化函数 clamp_min，用于限制 int32_t 类型向量的最小值
Vectorized<int32_t> inline clamp_min(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val) {
  // 使用 AVX2 指令集中的 _mm256_max_epi32 函数进行向量化最大值的计算
  return _mm256_max_epi32(min_val, a);
}

template <>
// 定义模板特化函数 clamp_min，用于限制 int16_t 类型向量的最小值
Vectorized<int16_t> inline clamp_min(const Vectorized<int16_t>& a, const Vectorized<int16_t>& min_val) {
  // 使用 AVX2 指令集中的 _mm256_max_epi16 函数进行向量化最大值的计算
  return _mm256_max_epi16(min_val, a);
}

template <>
// 定义模板特化函数 clamp_min，用于限制 int8_t 类型向量的最小值
Vectorized<int8_t> inline clamp_min(const Vectorized<int8_t>& a, const Vectorized<int8_t>& min_val) {
  // 使用 AVX2 指令集中的 _mm256_max_epi8 函数进行向量化最大值的计算
  return _mm256_max_epi8(min_val, a);
}

template <>
// 定义模板特化函数 clamp_min，用于限制 uint8_t 类型向量的最小值
Vectorized<uint8_t> inline clamp_min(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& min_val) {
  // 使用 AVX2 指令集中的 _mm256_max_epu8 函数进行向量化无符号最大值的计算
  return _mm256_max_epu8(min_val, a);
}

template<typename T>
// 定义通用模板函数 convert_to_int32，用于将类型 T 指针转换为 Vectorized<int32_t> 向量
Vectorized<int32_t> inline convert_to_int32(const T* ptr) {
  // 调用 Vectorized<int32_t> 类的 loadu 静态方法加载未对齐的数据指针
  return Vectorized<int32_t>::loadu(ptr);
}

template<>
// 定义模板特化函数 convert_to_int32，用于将 int8_t 类型指针转换为 Vectorized<int32_t> 向量
Vectorized<int32_t> inline convert_to_int32<int8_t>(const int8_t* ptr) {
  // 使用 AVX2 指令集中的 _mm256_cvtepi8_epi32 和 _mm_loadl_epi64 函数进行向量化的 int8_t 到 int32_t 类型转换
  return _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
}

template<>
// 定义模板特化函数 convert_to_int32，用于将 uint8_t 类型指针转换为 Vectorized<int32_t> 向量
Vectorized<int32_t> inline convert_to_int32<uint8_t>(const uint8_t* ptr) {
  // 使用 AVX2 指令集中的 _mm256_cvtepu8_epi32 和 _mm_loadl_epi64 函数进行向量化的 uint8_t 到 int32_t 类型转换
  return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
}
// 定义整型向量化运算符 '/'，接受两个同类型的向量，返回对应位置元素相除的结果向量
Vectorized<int64_t> inline operator/(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int64_t>());
}

// 特化模板，定义整型向量化运算符 '/'，用于 int32_t 类型向量
template <>
Vectorized<int32_t> inline operator/(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int32_t>());
}

// 特化模板，定义整型向量化运算符 '/'，用于 int16_t 类型向量
template <>
Vectorized<int16_t> inline operator/(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int16_t>());
}

// 特化模板，定义整型向量化运算符 '/'，用于 int8_t 类型向量
template <>
Vectorized<int8_t> inline operator/(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int8_t>());
}

// 特化模板，定义整型向量化运算符 '/'，用于 uint8_t 类型向量
template <>
Vectorized<uint8_t> inline operator/(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<uint8_t>());
}

// 定义位运算符 '&'，接受两个相同类型的向量，返回按位与操作的结果向量
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_and_si256(a, b);
}

// 定义位运算符 '|'，接受两个相同类型的向量，返回按位或操作的结果向量
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_or_si256(a, b);
}

// 定义位运算符 '^'，接受两个相同类型的向量，返回按位异或操作的结果向量
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_xor_si256(a, b);
}

// 定义位运算符 '~'，接受一个向量，返回按位取反操作的结果向量
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
}

// 成员函数，返回当前整型向量是否等于另一个向量的每个元素，结果向量元素为 1 或 0
inline Vectorized<int64_t> Vectorized<int64_t>::eq(const Vectorized<int64_t>& other) const {
  return (*this == other) & Vectorized<int64_t>(1);
}

// 成员函数，返回当前整型向量是否不等于另一个向量的每个元素，结果向量元素为 1 或 0
inline Vectorized<int64_t> Vectorized<int64_t>::ne(const Vectorized<int64_t>& other) const {
  return (*this != other) & Vectorized<int64_t>(1);
}

// 成员函数，返回当前整型向量是否大于另一个向量的每个元素，结果向量元素为 1 或 0
inline Vectorized<int64_t> Vectorized<int64_t>::gt(const Vectorized<int64_t>& other) const {
  return (*this > other) & Vectorized<int64_t>(1);
}

// 成员函数，返回当前整型向量是否大于等于另一个向量的每个元素，结果向量元素为 1 或 0
inline Vectorized<int64_t> Vectorized<int64_t>::ge(const Vectorized<int64_t>& other) const {
  return (*this >= other) & Vectorized<int64_t>(1);
}

// 成员函数，返回当前整型向量是否小于另一个向量的每个元素，结果向量元素为 1 或 0
inline Vectorized<int64_t> Vectorized<int64_t>::lt(const Vectorized<int64_t>& other) const {
  return (*this < other) & Vectorized<int64_t>(1);
}

// 成员函数，返回当前整型向量是否小于等于另一个向量的每个元素，结果向量元素为 1 或 0
inline Vectorized<int64_t> Vectorized<int64_t>::le(const Vectorized<int64_t>& other) const {
  return (*this <= other) & Vectorized<int64_t>(1);
}

// 成员函数，返回当前整型向量是否等于另一个向量的每个元素，结果向量元素为 1 或 0
inline Vectorized<int32_t> Vectorized<int32_t>::eq(const Vectorized<int32_t>& other) const {
  return (*this == other) & Vectorized<int32_t>(1);
}

// 成员函数，返回当前整型向量是否不等于另一个向量的每个元素，结果向量元素为 1 或 0
inline Vectorized<int32_t> Vectorized<int32_t>::ne(const Vectorized<int32_t>& other) const {
  return (*this != other) & Vectorized<int32_t>(1);
}
// 返回一个新的 Vectorized<int32_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否大于的结果
inline Vectorized<int32_t> Vectorized<int32_t>::gt(const Vectorized<int32_t>& other) const {
    return (*this > other) & Vectorized<int32_t>(1);
}

// 返回一个新的 Vectorized<int32_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否大于等于的结果
inline Vectorized<int32_t> Vectorized<int32_t>::ge(const Vectorized<int32_t>& other) const {
    return (*this >= other) & Vectorized<int32_t>(1);
}

// 返回一个新的 Vectorized<int32_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否小于的结果
inline Vectorized<int32_t> Vectorized<int32_t>::lt(const Vectorized<int32_t>& other) const {
    return (*this < other) & Vectorized<int32_t>(1);
}

// 返回一个新的 Vectorized<int32_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否小于等于的结果
inline Vectorized<int32_t> Vectorized<int32_t>::le(const Vectorized<int32_t>& other) const {
    return (*this <= other) & Vectorized<int32_t>(1);
}

// 返回一个新的 Vectorized<int16_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否等于的结果
inline Vectorized<int16_t> Vectorized<int16_t>::eq(const Vectorized<int16_t>& other) const {
    return (*this == other) & Vectorized<int16_t>(1);
}

// 返回一个新的 Vectorized<int16_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否不等于的结果
inline Vectorized<int16_t> Vectorized<int16_t>::ne(const Vectorized<int16_t>& other) const {
    return (*this != other) & Vectorized<int16_t>(1);
}

// 返回一个新的 Vectorized<int16_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否大于的结果
inline Vectorized<int16_t> Vectorized<int16_t>::gt(const Vectorized<int16_t>& other) const {
    return (*this > other) & Vectorized<int16_t>(1);
}

// 返回一个新的 Vectorized<int16_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否大于等于的结果
inline Vectorized<int16_t> Vectorized<int16_t>::ge(const Vectorized<int16_t>& other) const {
    return (*this >= other) & Vectorized<int16_t>(1);
}

// 返回一个新的 Vectorized<int16_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否小于的结果
inline Vectorized<int16_t> Vectorized<int16_t>::lt(const Vectorized<int16_t>& other) const {
    return (*this < other) & Vectorized<int16_t>(1);
}

// 返回一个新的 Vectorized<int16_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否小于等于的结果
inline Vectorized<int16_t> Vectorized<int16_t>::le(const Vectorized<int16_t>& other) const {
    return (*this <= other) & Vectorized<int16_t>(1);
}

// 返回一个新的 Vectorized<int8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否等于的结果
inline Vectorized<int8_t> Vectorized<int8_t>::eq(const Vectorized<int8_t>& other) const {
    return (*this == other) & Vectorized<int8_t>(1);
}

// 返回一个新的 Vectorized<int8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否不等于的结果
inline Vectorized<int8_t> Vectorized<int8_t>::ne(const Vectorized<int8_t>& other) const {
    return (*this != other) & Vectorized<int8_t>(1);
}

// 返回一个新的 Vectorized<int8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否大于的结果
inline Vectorized<int8_t> Vectorized<int8_t>::gt(const Vectorized<int8_t>& other) const {
    return (*this > other) & Vectorized<int8_t>(1);
}

// 返回一个新的 Vectorized<int8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否大于等于的结果
inline Vectorized<int8_t> Vectorized<int8_t>::ge(const Vectorized<int8_t>& other) const {
    return (*this >= other) & Vectorized<int8_t>(1);
}

// 返回一个新的 Vectorized<int8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否小于的结果
inline Vectorized<int8_t> Vectorized<int8_t>::lt(const Vectorized<int8_t>& other) const {
    return (*this < other) & Vectorized<int8_t>(1);
}

// 返回一个新的 Vectorized<int8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否小于等于的结果
inline Vectorized<int8_t> Vectorized<int8_t>::le(const Vectorized<int8_t>& other) const {
    return (*this <= other) & Vectorized<int8_t>(1);
}

// 返回一个新的 Vectorized<uint8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否等于的结果
inline Vectorized<uint8_t> Vectorized<uint8_t>::eq(const Vectorized<uint8_t>& other) const {
    return (*this == other) & Vectorized<uint8_t>(1);
}

// 返回一个新的 Vectorized<uint8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否不等于的结果
inline Vectorized<uint8_t> Vectorized<uint8_t>::ne(const Vectorized<uint8_t>& other) const {
    return (*this != other) & Vectorized<uint8_t>(1);
}

// 返回一个新的 Vectorized<uint8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否大于的结果
inline Vectorized<uint8_t> Vectorized<uint8_t>::gt(const Vectorized<uint8_t>& other) const {
    return (*this > other) & Vectorized<uint8_t>(1);
}

// 返回一个新的 Vectorized<uint8_t> 对象，其中每个元素是对应位置上 this 对象和参数 other 对象对比是否大于等于的结果
inline Vectorized<uint8_t> Vectorized<uint8_t>::ge(const Vectorized<uint8_t>& other) const {
    return (*this >= other) & Vectorized<uint8_t>(1);
}
# 实现 Vectorized<uint8_t> 类的 lt 方法，用于比较当前对象与另一个对象是否小于
inline Vectorized<uint8_t> Vectorized<uint8_t>::lt(const Vectorized<uint8_t>& other) const {
    # 返回当前对象是否小于另一个对象，并且使用 Vectorized<uint8_t>(1) 来构造结果向量
    return (*this < other) & Vectorized<uint8_t>(1);
}

# 实现 Vectorized<uint8_t> 类的 le 方法，用于比较当前对象与另一个对象是否小于等于
inline Vectorized<uint8_t> Vectorized<uint8_t>::le(const Vectorized<uint8_t>& other) const {
    # 返回当前对象是否小于等于另一个对象，并且使用 Vectorized<uint8_t>(1) 来构造结果向量
    return (*this <= other) & Vectorized<uint8_t>(1);
}

# 开始定义一个模板，该模板用于处理左移操作
// 定义一个函数，名称为 shift_256_16，参数类型为 Vectorized<int16_t> 类型的引用 a 和 b
Vectorized<int16_t> inline shift_256_16(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  // 由于 int16_t 类型没有用于向量化的移位指令，因此通过模拟来实现移位操作。

  // 控制用于洗牌操作的掩码，将 256 位视为包含 16 位元素的数组，并考虑相邻元素对。
  // 名为 "ctl_0_1" 的掩码设置使得洗牌操作将输入对中索引为 M 的元素移动到输出对中索引为 N 的位置，
  // 同时将输出对中索引为 M 的元素设置为全 0。
  __m256i ctl_0_1 = _mm256_set_epi8(29, 28, 0x80, 0x80, 25, 24, 0x80, 0x80,
                                    21, 20, 0x80, 0x80, 17, 16, 0x80, 0x80,
                                    13, 12, 0x80, 0x80, 9, 8, 0x80, 0x80,
                                    5, 4, 0x80, 0x80, 1, 0, 0x80, 0x80);
  // 另一个洗牌掩码，将元素 M 移到元素 N，同时将元素 M 设置为全 0。
  __m256i ctl_1_0 = _mm256_set_epi8(0x80, 0x80, 31, 30, 0x80, 0x80, 27, 26,
                                    0x80, 0x80, 23, 22, 0x80, 0x80, 19, 18,
                                    0x80, 0x80, 15, 14, 0x80, 0x80, 11, 10,
                                    0x80, 0x80, 7, 6, 0x80, 0x80, 3, 2);

  // 位与操作的掩码，将 256 位视为包含 16 位元素的数组，并对相邻元素对进行操作。
  // 名为 "keep_0" 的掩码设置使得位与操作将输入对中索引为 M 的元素复制到输出对中相同索引的位置，
  // 同时将输出对中的另一个元素设置为全 0。
  __m256i keep_0 = _mm256_set1_epi32(0xFFFF);
  __m256i keep_1 = _mm256_set1_epi32(0xFFFF0000);

  // 从待移位的输入数组中取出每个索引为偶数的 16 位元素，并将其扩展为 32 位，以在右侧添加 0。
  // 然后对这个 32 位数字进行移位操作。上半部分 16 位将是移位后的结果，将其写入结果数组，
  // 写入位置与对应的输入元素取出位置相同。同时，确保结果数组中索引为奇数的元素设置为全 0。
  //
  // 注意，要移位的位数通过在左侧添加 0 来扩展为 32 位。这意味着对于负值，移位数不会进行适当的符号扩展。
  // 然而，移位指令按无符号整数处理移位数，因此如果是负数，则无论是否进行适当的符号扩展，
  // 它都会被解释为大于 32 的数字，移位结果将是相同的。
  __m256i a0 = _mm256_shuffle_epi8(a, ctl_0_1);
  __m256i b0 = _mm256_and_si256(b, keep_0);
  __m256i c0;
  if (left_shift)
    c0 = _mm256_sllv_epi32(a0, b0);
  else
    // 使用 AVX2 指令集中的 _mm256_srav_epi32 函数对 a0 中的每个元素右移动 b0 中相应元素的位数，结果存储在 c0 中
    c0 = _mm256_srav_epi32(a0, b0);

    // 使用 AVX2 指令集中的 _mm256_shuffle_epi8 函数，根据掩码 ctl_1_0 对 c0 进行字节级别的重新排列
    c0 = _mm256_shuffle_epi8(c0, ctl_1_0);

    // 对于输入数组中 idx%2==1 的元素，使用 AVX2 指令集中的函数执行相同的位移操作
    __m256i a1 = _mm256_and_si256(a, keep_1);  // 使用掩码 keep_1 对 a 进行按位与操作，得到 a1
    __m256i b1 = _mm256_shuffle_epi8(b, ctl_1_0);  // 使用掩码 ctl_1_0 对 b 进行字节级别的重新排列，得到 b1
    __m256i c1;
    if (left_shift)
        c1 = _mm256_sllv_epi32(a1, b1);  // 如果 left_shift 为真，则使用 _mm256_sllv_epi32 函数对 a1 中的每个元素左移动 b1 中相应元素的位数
    else
        c1 = _mm256_srav_epi32(a1, b1);  // 否则，使用 _mm256_srav_epi32 函数对 a1 中的每个元素右移动 b1 中相应元素的位数
    c1 = _mm256_and_si256(c1, keep_1);  // 使用掩码 keep_1 对 c1 进行按位与操作，确保结果在指定的位宽内

    // 将部分结果 c0 和 c1 合并到最终结果 c 中
    __m256i c = _mm256_or_si256(c0, c1);

    return c;  // 返回最终结果向量 c
}

// 结束前面的模板定义块

template <bool left_shift, typename T, typename std::enable_if_t<std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>, int> = 0>
// 定义模板函数，接受一个布尔型参数 left_shift，一个类型参数 T，只有当 T 是 int8_t 或 uint8_t 时才有效
c0 = _mm256_sllv_epi32(a0, b0);
// 如果 left_shift 是 true，使用 _mm256_sllv_epi32 对 a0 和 b0 进行位移操作，结果赋给 c0
else
  // 否则
  if constexpr (std::is_same_v<T, int8_t>)
    // 如果 T 是 int8_t 类型
    c0 = _mm256_srav_epi32(a0, b0);
    // 使用 _mm256_srav_epi32 对 a0 和 b0 进行算术右移操作，结果赋给 c0
  else
    // 否则（即 T 是 uint8_t 类型）
    c0 = _mm256_srlv_epi32(a0, b0);
    // 使用 _mm256_srlv_epi32 对 a0 和 b0 进行逻辑右移操作，结果赋给 c0
c0 = _mm256_shuffle_epi8(c0, ctl_3_0);
// 使用 _mm256_shuffle_epi8 根据 ctl_3_0 的掩码对 c0 进行字节级重排

// Peform shifting the same way for input array elements with
// idx%4==1.
// 对于下标模 4 余 1 的输入数组元素执行相同的位移操作
__m256i a1 = _mm256_shuffle_epi8(a, ctl_1_3);
// 使用 _mm256_shuffle_epi8 根据 ctl_1_3 的掩码对 a 进行字节级重排，结果赋给 a1
__m256i b1 = _mm256_shuffle_epi8(b, ctl_1_0);
// 使用 _mm256_shuffle_epi8 根据 ctl_1_0 的掩码对 b 进行字节级重排，结果赋给 b1
__m256i c1;
// 声明 __m256i 类型变量 c1
if (left_shift)
  // 如果 left_shift 是 true
  c1 = _mm256_sllv_epi32(a1, b1);
  // 使用 _mm256_sllv_epi32 对 a1 和 b1 进行位移操作，结果赋给 c1
else
  // 否则
  if constexpr (std::is_same_v<T, int8_t>)
    // 如果 T 是 int8_t 类型
    c1 = _mm256_srav_epi32(a1, b1);
    // 使用 _mm256_srav_epi32 对 a1 和 b1 进行算术右移操作，结果赋给 c1
  else
    // 否则（即 T 是 uint8_t 类型）
    c1 = _mm256_srlv_epi32(a1, b1);
    // 使用 _mm256_srlv_epi32 对 a1 和 b1 进行逻辑右移操作，结果赋给 c1
c1 = _mm256_shuffle_epi8(c1, ctl_3_1);
// 使用 _mm256_shuffle_epi8 根据 ctl_3_1 的掩码对 c1 进行字节级重排

// Peform shifting the same way for input array elements with
// idx%4==2.
// 对于下标模 4 余 2 的输入数组元素执行相同的位移操作
__m256i a2 = _mm256_shuffle_epi8(a, ctl_2_3);
// 使用 _mm256_shuffle_epi8 根据 ctl_2_3 的掩码对 a 进行字节级重排，结果赋给 a2
__m256i b2 = _mm256_shuffle_epi8(b, ctl_2_0);
// 使用 _mm256_shuffle_epi8 根据 ctl_2_0 的掩码对 b 进行字节级重排，结果赋给 b2
__m256i c2;
// 声明 __m256i 类型变量 c2
if (left_shift)
  // 如果 left_shift 是 true
  c2 = _mm256_sllv_epi32(a2, b2);
  // 使用 _mm256_sllv_epi32 对 a2 和 b2 进行位移操作，结果赋给 c2
else
  // 否则
  if constexpr (std::is_same_v<T, int8_t>)
    // 如果 T 是 int8_t 类型
    c2 = _mm256_srav_epi32(a2, b2);
    // 使用 _mm256_srav_epi32 对 a2 和 b2 进行算术右移操作，结果赋给 c2
  else
    // 否则（即 T 是 uint8_t 类型）
    c2 = _mm256_srlv_epi32(a2, b2);
    // 使用 _mm256_srlv_epi32 对 a2 和 b2 进行逻辑右移操作，结果赋给 c2
c2 = _mm256_shuffle_epi8(c2, ctl_3_2);
// 使用 _mm256_shuffle_epi8 根据 ctl_3_2 的掩码对 c2 进行字节级重排

// Peform shifting the same way for input array elements with
// idx%4==3.
// 对于下标模 4 余 3 的输入数组元素执行相同的位移操作
__m256i a3 =  _mm256_and_si256(a, keep_3);
// 使用 _mm256_and_si256 对 a 和 keep_3 进行按位与操作，结果赋给 a3
__m256i b3 = _mm256_shuffle_epi8(b, ctl_3_0);
// 使用 _mm256_shuffle_epi8 根据 ctl_3_0 的掩码对 b 进行字节级重排，结果赋给 b3
__m256i c3;
// 声明 __m256i 类型变量 c3
if (left_shift)
  // 如果 left_shift 是 true
  c3 = _mm256_sllv_epi32(a3, b3);
  // 使用 _mm256_sllv_epi32 对 a3 和 b3 进行位移操作，结果赋给 c3
else
  // 否则
  if constexpr (std::is_same_v<T, int8_t>)
    // 如果 T 是 int8_t 类型
    c3 = _mm256_srav_epi32(a3, b3);
    // 使用 _mm256_srav_epi32 对 a3 和 b3 进行算术右移操作，结果赋给 c3
  else
    // 否则（即 T 是 uint8_t 类型）
    c3 = _mm256_srlv_epi32(a3, b3);
    // 使用 _mm256_srlv_epi32 对 a3 和 b3 进行逻辑右移操作，结果赋给 c3
c3 = _mm256_and_si256(c3, keep_3);
// 使用 _mm256_and_si256 对 c3 和 keep_3 进行按位与操作，结果赋给 c3

// Merge partial results into the final result.
// 将部分结果合并成最终结果
__m256i c01 = _mm256_or_si256(c0, c1);
// 使用 _mm256_or_si256 对 c0 和 c1 进行按位或操作，结果赋给 c01
__m256i c23 = _mm256_or_si256(c2, c3);
// 使用 _mm256_or_si256 对 c2 和 c3 进行按位或操作，结果赋给 c23
__m256i c = _mm256_or_si256(c01, c23);
// 使用 _mm256_or_si256 对 c01 和 c23 进行按位或操作，结果赋给 c

return c;
// 返回最终的结果 c
}

template <>
// 特化模板函数，当 T 类型为 int64_t 时
Vectorized<int64_t> inline operator<<(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm256_sllv_epi64(a, b);
  // 使用 _mm256_sllv_epi64 对 a 和 b 进行位移操作，返回结果
}

template <>
// 特化模板函数，当 T 类型为 int32_t 时
Vectorized<int32_t> inline operator<<(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_sllv_epi32(a, b);
  // 使用 _mm256_sllv_epi32 对 a 和 b 进
// 定义一个模板特化的右移运算符重载函数，用于处理 int64_t 类型的向量操作
Vectorized<int64_t> inline operator>>(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  // 由于 int64_t 类型没有直接的矢量右移指令，因此在此模拟实现。
  
  // 设置一个全 0 的 256 位整数向量
  __m256i zero = _mm256_set1_epi64x(0);
  // 设置一个全 64 的 256 位整数向量，用于限制右移位数的范围
  __m256i max_shift = _mm256_set1_epi64x(64);
  // 创建一个掩码，用于将小于 0 或大于 64 的移位值限制为 64，从而使负输入为 -1，非负输入为 0
  __m256i mask = _mm256_or_si256(_mm256_cmpgt_epi64(zero, b), _mm256_cmpgt_epi64(b, max_shift));
  // 使用掩码选择正确的移位值
  __m256i shift = _mm256_blendv_epi8(b, max_shift, mask);
  // 计算符号位扩展所需的操作
  __m256i sign_bits = _mm256_cmpgt_epi64(zero, a);
  __m256i sign_shift = _mm256_sub_epi64(max_shift, shift);
  __m256i sign_ext = _mm256_sllv_epi64(sign_bits, sign_shift);
  // 执行逻辑右移操作，并用符号位替换最高位
  __m256i c = _mm256_srlv_epi64(a, shift);
  c = _mm256_or_si256(c, sign_ext);

  return c;
}

// 模板特化的右移运算符重载函数，处理 int32_t 类型的向量操作
template <>
Vectorized<int32_t> inline operator>>(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_srav_epi32(a, b);
}

// 模板特化的右移运算符重载函数，处理 int16_t 类型的向量操作
template <>
Vectorized<int16_t> inline operator>>(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return shift_256_16<false>(a, b);
}

// 模板特化的右移运算符重载函数，处理 int8_t 类型的向量操作
template <>
Vectorized<int8_t> inline operator>>(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return shift_256_8<false>(a, b);
}

// 模板特化的右移运算符重载函数，处理 uint8_t 类型的向量操作
template <>
Vectorized<uint8_t> inline operator>>(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return shift_256_8<false>(a, b);
}

#endif

}} // namespace at::vec::CPU_CAPABILITY
```