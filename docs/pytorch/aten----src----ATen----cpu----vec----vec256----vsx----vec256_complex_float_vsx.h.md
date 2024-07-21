# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_complex_float_vsx.h`

```py
#pragma once
// 包含 ATen 库中有关 CPU 矢量化的头文件
#include <ATen/cpu/vec/intrinsics.h>
// 包含 ATen 库中的矢量化基类定义
#include <ATen/cpu/vec/vec_base.h>
// 包含 ATen 库中针对 VSX 指令集的矢量化助手函数
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
// 包含 C10 实用工具库中关于复数的定义
#include <c10/util/complex.h>
// 包含 C10 实用工具库中关于整数范围的定义
#include <c10/util/irange.h>

// 在 at 命名空间内定义 vec 命名空间，用于向量化操作
namespace at {
namespace vec {
// 内联命名空间 CPU_CAPABILITY，用于封装特定 CPU 指令集的功能
inline namespace CPU_CAPABILITY {
// 使用 ComplexFlt 作为 float 复数的别名
using ComplexFlt = c10::complex<float>;

// Vectorized<ComplexFlt> 模板特化类定义
template <>
class Vectorized<ComplexFlt> {
 private:
  // 匿名联合体，用于处理内部矢量数据和掩码数据
  union {
    // 结构体成员，包含两个 vfloat32 类型的矢量数据
    struct {
      vfloat32 _vec0;
      vfloat32 _vec1;
    };
    // 结构体成员，包含两个 vbool32 类型的掩码数据
    struct {
      vbool32 _vecb0;
      vbool32 _vecb1;
    };

  } __attribute__((__may_alias__)); // 指定联合体可以使用别名

 public:
  // 类型定义
  using value_type = ComplexFlt;
  using vec_internal_type = vfloat32;
  using vec_internal_mask_type = vbool32;
  using size_type = int;

  // 返回向量化对象的大小为 4
  static constexpr size_type size() {
    return 4;
  }
  
  // 默认构造函数
  Vectorized() {}

  // 构造函数，使用单个 vfloat32 类型初始化
  C10_ALWAYS_INLINE Vectorized(vfloat32 v) : _vec0{v}, _vec1{v} {}
  
  // 构造函数，使用单个 vbool32 类型初始化
  C10_ALWAYS_INLINE Vectorized(vbool32 vmask) : _vecb0{vmask}, _vecb1{vmask} {}

  // 构造函数，使用两个 vfloat32 类型初始化
  C10_ALWAYS_INLINE Vectorized(vfloat32 v1, vfloat32 v2) : _vec0{v1}, _vec1{v2} {}

  // 构造函数，使用两个 vbool32 类型初始化
  C10_ALWAYS_INLINE Vectorized(vbool32 v1, vbool32 v2) : _vecb0{v1}, _vecb1{v2} {}

  // 构造函数，使用 ComplexFlt 类型初始化
  Vectorized(ComplexFlt val) {
    float real_value = val.real();
    float imag_value = val.imag();
    _vec0 = vfloat32{real_value, imag_value, real_value, imag_value};
    _vec1 = vfloat32{real_value, imag_value, real_value, imag_value};
  }

  // 构造函数，使用四个 ComplexFlt 类型初始化
  Vectorized(ComplexFlt val1, ComplexFlt val2, ComplexFlt val3, ComplexFlt val4) {
    _vec0 = vfloat32{val1.real(), val1.imag(), val2.real(), val2.imag()};
    _vec1 = vfloat32{val3.real(), val3.imag(), val4.real(), val4.imag()};
  }

  // 模板函数，根据掩码 mask 来进行矢量化数据的混合操作
  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 0, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    return a;
  }

  // 模板函数，根据掩码 mask 来进行矢量化数据的混合操作
  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 1, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    return b;
  }

  // 模板函数，根据掩码 mask 来进行矢量化数据的混合操作
  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 2, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    return {b._vec0, a._vec1};
  }

  // 模板函数，根据掩码 mask 来进行矢量化数据的混合操作
  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 3, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    return {a._vec0, b._vec1};
  }

  // 模板函数，根据掩码 mask 来进行矢量化数据的混合操作
  template <uint64_t mask>
  static std::enable_if_t<blendChoiceComplex(mask) == 4, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    // 使用掩码 mask 构建第一个掩码
    const vbool32 mask_1st = VsxComplexMask1(mask);
  template <uint64_t mask>
  // 如果模板参数 mask 对应的 blendChoiceComplex 结果为 4，返回复数向量的混合结果
  static std::enable_if_t<blendChoiceComplex(mask) == 4, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    // 使用生成的掩码 mask_1st 混合向量 a 和 b 的 _vec0 部分
    const vbool32 mask_1st = VsxComplexMask1(mask);
    // 返回混合结果，其中_vec0 使用 mask_1st 控制
    return {(vfloat32)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1};
  }

  template <uint64_t mask>
  // 如果模板参数 mask 对应的 blendChoiceComplex 结果为 5，返回复数向量的混合结果
  static std::enable_if_t<blendChoiceComplex(mask) == 5, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    // 使用生成的掩码 mask_1st 混合向量 a 和 b 的 _vec0 部分
    const vbool32 mask_1st = VsxComplexMask1(mask);
    // 返回混合结果，其中_vec0 使用 mask_1st 控制，_vec1 直接使用向量 b 的 _vec1
    return {(vfloat32)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1};
  }

  template <uint64_t mask>
  // 如果模板参数 mask 对应的 blendChoiceComplex 结果为 6，返回复数向量的混合结果
  static std::enable_if_t<blendChoiceComplex(mask) == 6, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    // 使用生成的掩码 mask_2nd 混合向量 a 和 b 的 _vec1 部分
    const vbool32 mask_2nd = VsxComplexMask2(mask);
    // 返回混合结果，其中_vec0 直接使用向量 a 的 _vec0，_vec1 使用 mask_2nd 控制
    return {a._vec0, (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <uint64_t mask>
  // 如果模板参数 mask 对应的 blendChoiceComplex 结果为 7，返回复数向量的混合结果
  static std::enable_if_t<blendChoiceComplex(mask) == 7, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    // 使用生成的掩码 mask_2nd 混合向量 a 和 b 的 _vec1 部分
    const vbool32 mask_2nd = VsxComplexMask2(mask);
    // 返回混合结果，其中_vec0 直接使用向量 b 的 _vec0，_vec1 使用 mask_2nd 控制
    return {b._vec0, (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <uint64_t mask>
  // 如果模板参数 mask 对应的 blendChoiceComplex 结果为 8，返回复数向量的混合结果
  static std::enable_if_t<blendChoiceComplex(mask) == 8, Vectorized<ComplexFlt>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    // 使用生成的掩码 mask_1st 混合向量 a 和 b 的 _vec0 部分
    const vbool32 mask_1st = VsxComplexMask1(mask);
    // 使用生成的掩码 mask_2nd 混合向量 a 和 b 的 _vec1 部分
    const vbool32 mask_2nd = VsxComplexMask2(mask);
    // 返回混合结果，_vec0 和 _vec1 分别使用 mask_1st 和 mask_2nd 控制
    return {
        (vfloat32)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  template <int64_t mask>
  // 返回逐元素混合的复数向量的结果
  static Vectorized<ComplexFlt> C10_ALWAYS_INLINE
  el_blend(const Vectorized<ComplexFlt>& a, const Vectorized<ComplexFlt>& b) {
    // 使用生成的掩码 mask_1st 混合向量 a 和 b 的 _vec0 部分
    const vbool32 mask_1st = VsxMask1(mask);
    // 使用生成的掩码 mask_2nd 混合向量 a 和 b 的 _vec1 部分
    const vbool32 mask_2nd = VsxMask2(mask);
    // 返回逐元素混合的结果，_vec0 和 _vec1 分别使用 mask_1st 和 mask_2nd 控制
    return {
        (vfloat32)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  static Vectorized<ComplexFlt> blendv(
      const Vectorized<ComplexFlt>& a,
      const Vectorized<ComplexFlt>& b,
      const Vectorized<ComplexFlt>& mask) {
    // 将 std::complex<V> 类型的掩码 mask 转换为 V 类型的掩码 mask_complex: xy -> xxyy
    auto mask_complex = Vectorized<ComplexFlt>(
        vec_mergeh(mask._vec0, mask._vec0), vec_mergeh(mask._vec1, mask._vec1));
    // 返回混合结果，使用 mask_complex 控制 a 和 b 的各个元素
    return {
        vec_sel(a._vec0, b._vec0, reinterpret_cast<vbool32>(mask_complex._vec0)),
        vec_sel(a._vec1, b._vec1, reinterpret_cast<vbool32>(mask_complex._vec1)),
    };
  }

  static Vectorized<ComplexFlt> elwise_blendv(
      const Vectorized<ComplexFlt>& a,
      const Vectorized<ComplexFlt>& b,
      const Vectorized<ComplexFlt>& mask) {
    // 返回逐元素混合的结果，使用 mask 控制 a 和 b 的各个元素
    return {
        vec_sel(a._vec0, b._vec0, reinterpret_cast<vbool32>(mask._vec0)),
        vec_sel(a._vec1, b._vec1, reinterpret_cast<vbool32>(mask._vec1)),
    };
  }
  // 返回一个 Vectorized<ComplexFlt> 对象，其中包含四个元素，每个元素是 base 到 base + 3 * step 的复数序列
  return Vectorized<ComplexFlt>(
      base,
      base + step,
      base + ComplexFlt(2) * step,
      base + ComplexFlt(3) * step);
}

// 静态方法，根据 count 的值返回一个混合后的 Vectorized<ComplexFlt> 对象
static Vectorized<ComplexFlt> set(
    const Vectorized<ComplexFlt>& a,
    const Vectorized<ComplexFlt>& b,
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

// 静态方法，加载未对齐的数据指针到 Vectorized<value_type> 对象
static Vectorized<value_type> C10_ALWAYS_INLINE
loadu(const void* ptr, int count = size()) {
  if (count == size()) {
    // 直接从未对齐的数据指针 ptr 中加载数据到 Vectorized 对象，使用 vec_vsx_ld 加载到两个向量中
    return {
        vec_vsx_ld(offset0, reinterpret_cast<const float*>(ptr)),
        vec_vsx_ld(offset16, reinterpret_cast<const float*>(ptr))};
  }

  // 如果 count 小于 size()，则先将数据复制到临时数组 tmp_values，再加载到 Vectorized 对象中
  __at_align__ value_type tmp_values[size()] = {};
  std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

  return {
      vec_vsx_ld(offset0, reinterpret_cast<const float*>(tmp_values)),
      vec_vsx_ld(offset16, reinterpret_cast<const float*>(tmp_values))};
}

// 将 Vectorized 对象的数据存储到内存地址 ptr 处
void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
  if (count == size()) {
    // 将 _vec0 和 _vec1 中的数据存储到 ptr 指向的内存地址中
    vec_vsx_st(_vec0, offset0, reinterpret_cast<float*>(ptr));
    vec_vsx_st(_vec1, offset16, reinterpret_cast<float*>(ptr));
  } else if (count > 0) {
    // 如果 count 小于 size()，则先存储到临时数组 tmp_values 中，再将其复制到 ptr 指向的内存地址中
    __at_align__ value_type tmp_values[size()];
    vec_vsx_st(_vec0, offset0, reinterpret_cast<float*>(tmp_values));
    vec_vsx_st(_vec1, offset16, reinterpret_cast<float*>(tmp_values));
    std::memcpy(
        ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
  }
}

// 删除操作符重载，禁止通过索引访问元素
const ComplexFlt& operator[](int idx) const = delete;
ComplexFlt& operator[](int idx) = delete;

// 对 Vectorized<ComplexFlt> 中的每个元素应用函数 f，并返回结果
Vectorized<ComplexFlt> map(ComplexFlt (*const f)(ComplexFlt)) const {
  __at_align__ ComplexFlt tmp[size()];
  store(tmp);
  for (const auto i : c10::irange(size())) {
    tmp[i] = f(tmp[i]);
  }
  return loadu(tmp);
}

// 对 Vectorized<ComplexFlt> 中的每个元素应用函数 f，并返回结果
Vectorized<ComplexFlt> map(ComplexFlt (*const f)(const ComplexFlt&)) const {
  __at_align__ ComplexFlt tmp[size()];
  store(tmp);
  for (const auto i : c10::irange(size())) {
    tmp[i] = f(tmp[i]);
  }
  return loadu(tmp);
}

// 静态方法，对两个 Vectorized<ComplexFlt> 对象进行水平相加操作
static Vectorized<ComplexFlt> horizontal_add(
    Vectorized<ComplexFlt>& first,
    Vectorized<ComplexFlt>& second) {
  // 使用 el_mergee 和 el_mergeo 函数对两个向量进行水平相加操作
  // 结果向量的每个元素是对应元素实部和虚部的和
  return el_mergee(first, second) + el_mergeo(first, second);
}

// 静态方法，通过指定的方式对两个 Vectorized<ComplexFlt> 对象进行水平减法
static Vectorized<ComplexFlt> horizontal_sub_permD8(
    Vectorized<ComplexFlt>& first,
    Vectorized<ComplexFlt>& second) {
  // 使用 el_swapped 函数对第一个向量进行重新排列，然后将第二个向量加到重新排列后的第一个向量上
  // 以模拟水平减法操作
  auto first_perm = first.el_swapped(); // 2perm
    auto second_perm = second.el_swapped(); // 生成第二个向量的元素交换版本，称为2perm
    // 计算第一个向量与其元素交换版本的差，称为2sub
    auto first_ret = first - first_perm;
    // 计算第二个向量与其元素交换版本的差，称为2sub
    auto second_ret = second - second_perm;
    // 将计算得到的两个向量的结果进行合并
    return el_mergee(first_ret, second_ret); // 返回两个向量合并后的结果，称为2 mergee's
  }

  Vectorized<ComplexFlt> abs_2_() const {
    auto a = (*this).elwise_mult(*this); // 计算当前向量与自身的逐元素乘积，称为a
    auto permuted = a.el_swapped(); // 获取a的元素交换版本
    a = a + permuted; // 将a与其元素交换版本相加
    return a.el_mergee(); // 返回a与其元素交换版本合并后的结果
  }

  Vectorized<ComplexFlt> abs_() const {
    auto vi = el_mergeo(); // 获取当前向量的奇数索引元素
    auto vr = el_mergee(); // 获取当前向量的偶数索引元素
    // 返回向量中每个复数的绝对值
    return {Sleef_hypotf4_u05vsx(vr._vec0, vi._vec0), Sleef_hypotf4_u05vsx(vr._vec1, vi._vec1)};
  }

  Vectorized<ComplexFlt> abs() const {
    return abs_() & real_mask; // 返回绝对值，并按位与实部掩码
  }

  Vectorized<ComplexFlt> real_() const {
    return *this & real_mask; // 返回当前向量的实部
  }

  Vectorized<ComplexFlt> real() const {
    return *this & real_mask; // 返回当前向量的实部
  }

  Vectorized<ComplexFlt> imag_() const {
    return *this & imag_mask; // 返回当前向量的虚部
  }

  Vectorized<ComplexFlt> imag() const {
    // 使用swap_mask或sldwi进行虚部的处理
    auto ret = imag_(); // 获取当前向量的虚部
    // 返回虚部每个元素向左移动3位后的结果
    return {
        vec_sldw(ret._vec0, ret._vec0, 3), vec_sldw(ret._vec1, ret._vec1, 3)};
  }

  Vectorized<ComplexFlt> conj_() const {
    return *this ^ isign_mask; // 返回当前向量的共轭
  }

  Vectorized<ComplexFlt> conj() const {
    return *this ^ isign_mask; // 返回当前向量的共轭
  }

  Vectorized<ComplexFlt> log() const {
    // 大多数三角函数操作使用log()函数来提高复数性能
    return map(std::log); // 对当前向量的每个元素应用对数函数，并返回结果向量
  }

  Vectorized<ComplexFlt> log2() const {
    // log2eB_inv
    auto ret = log(); // 计算当前向量的每个元素的自然对数
    // 返回结果与log2e_inv逐元素相乘的向量
    return ret.elwise_mult(log2e_inv);
  }

  Vectorized<ComplexFlt> log10() const {
    auto ret = log(); // 计算当前向量的每个元素的自然对数
    // 返回结果与log10e_inv逐元素相乘的向量
    return ret.elwise_mult(log10e_inv);
  }

  Vectorized<ComplexFlt> log1p() const {
    return map(std::log1p); // 对当前向量的每个元素应用log1p()函数，并返回结果向量
  }

  Vectorized<ComplexFlt> el_swapped() const {
    // 使用swap_mask对向量的元素进行交换
    vfloat32 v0 = vec_perm(_vec0, _vec0, swap_mask); // 对_vec0进行元素交换
    vfloat32 v1 = vec_perm(_vec1, _vec1, swap_mask); // 对_vec1进行元素交换
    // 返回交换元素后的向量
    return {v0, v1};
  }

  Vectorized<ComplexFlt> el_mergee() const {
    // 在mergee阶段中，使用vec_perm和掩码进行合并
    return {vec_mergee(_vecb0, _vecb0), vec_mergee(_vecb1, _vecb1)}; // 返回两个向量的偶数索引元素合并后的结果
  }

  Vectorized<ComplexFlt> el_mergeo() const {
    // 在mergeo阶段中，使用vec_perm和掩码进行合并
    return {vec_mergeo(_vecb0, _vecb0), vec_mergeo(_vecb1, _vecb1)}; // 返回两个向量的奇数索引元素合并后的结果
  }

  Vectorized<ComplexFlt> el_madd(
      const Vectorized<ComplexFlt>& multiplier,
      const Vectorized<ComplexFlt>& val) const {
    // 对当前向量的每个元素进行乘法和加法操作
    return {
        vec_madd(_vec0, multiplier._vec0, val._vec0), // 对_vec0进行乘法和加法操作
        vec_madd(_vec1, multiplier._vec1, val._vec1)}; // 对_vec1进行乘法和加法操作
  }

  static Vectorized<ComplexFlt> el_mergee(
      Vectorized<ComplexFlt>& first,
      Vectorized<ComplexFlt>& second) {
    // 在mergee阶段中，使用vec_perm和掩码对两个向量进行合并
    return {
        vec_mergee(first._vecb0, second._vecb0), // 对first和second的第一个向量的偶数索引元素进行合并
        vec_mergee(first._vecb1, second._vecb1)}; // 对first和second的第二个向量的偶数索引元素进行合并
  }

  static Vectorized<ComplexFlt> el_mergeo(
      Vectorized<ComplexFlt>& first,
      Vectorized<ComplexFlt>& second) {
  Vectorized<ComplexFlt> angle_() const {
    // 计算复数向量的幅角
    // angle = atan2(b/a)
    Vectorized<ComplexFlt> ret;
    for (int i = 0; i < 4; i += 2) {
      // 计算每个复数向量元素的幅角
      ret._vec0[i] = std::atan2(_vec0[i + 1], _vec0[i]);
      ret._vec1[i] = std::atan2(_vec1[i + 1], _vec1[i]);
    }
    return ret;
  }

  Vectorized<ComplexFlt> angle() const {
    // 返回复数向量的幅角，并与实部掩码进行按位与操作
    return angle_() & real_mask;
  }

  Vectorized<ComplexFlt> sin() const {
    // 返回复数向量的正弦值
    return map(std::sin);
  }

  Vectorized<ComplexFlt> sinh() const {
    // 返回复数向量的双曲正弦值
    return map(std::sinh);
  }

  Vectorized<ComplexFlt> cos() const {
    // 返回复数向量的余弦值
    return map(std::cos);
  }

  Vectorized<ComplexFlt> cosh() const {
    // 返回复数向量的双曲余弦值
    return map(std::cosh);
  }

  Vectorized<ComplexFlt> ceil() const {
    // 返回向上取整后的复数向量
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }

  Vectorized<ComplexFlt> floor() const {
    // 返回向下取整后的复数向量
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }

  Vectorized<ComplexFlt> neg() const {
    // 返回复数向量的负数
    auto z = Vectorized<ComplexFlt>(zero);
    return z - *this;
  }

  Vectorized<ComplexFlt> round() const {
    // 返回四舍五入后的复数向量
    return {vec_round(_vec0), vec_round(_vec1)};
  }

  Vectorized<ComplexFlt> tan() const {
    // 返回复数向量的正切值
    return map(std::tan);
  }

  Vectorized<ComplexFlt> tanh() const {
    // 返回复数向量的双曲正切值
    return map(std::tanh);
  }

  Vectorized<ComplexFlt> trunc() const {
    // 返回截断后的复数向量
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<ComplexFlt> elwise_sqrt() const {
    // 返回复数向量元素的平方根
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }

  Vectorized<ComplexFlt> sqrt() const {
    // 返回复数向量的平方根
    return map(std::sqrt);
  }

  Vectorized<ComplexFlt> reciprocal() const {
    // 返回复数向量的倒数
    // re + im*i = (a + bi)  / (c + di)
    // re = (ac + bd)/abs_2() = c/abs_2()
    // im = (bc - ad)/abs_2() = d/abs_2()
    auto c_d = *this ^ isign_mask; // c       -d
    auto abs = abs_2_();
    return c_d.elwise_div(abs);
  }

  Vectorized<ComplexFlt> rsqrt() const {
    // 返回复数向量的平方根的倒数
    return sqrt().reciprocal();
  }

  Vectorized<ComplexFlt> pow(const Vectorized<ComplexFlt>& exp) const {
    // 返回复数向量的指数幂
    __at_align__ ComplexFlt x_tmp[size()];
    __at_align__ ComplexFlt y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (const auto i : c10::irange(size())) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }

  Vectorized<ComplexFlt> atan() const {
    // 返回复数向量的反正切值
    // atan(x) = i/2 * ln((i + z)/(i - z))
    auto ione = Vectorized(imag_one);
    auto sum = ione + *this;
    auto sub = ione - *this;
    auto ln = (sum / sub).log(); // ln((i + z)/(i - z))
    return ln * imag_half; // i/2*ln()
  }

  Vectorized<ComplexFlt> atanh() const {
    // 返回复数向量的反双曲正切值
    return map(std::atanh);
  }

  Vectorized<ComplexFlt> acos() const {
    // 返回复数向量的反余弦值
    // acos(x) = pi/2 - asin(x)
    return Vectorized(pi_2) - asin();
  }

  Vectorized<ComplexFlt> inline operator*(const Vectorized<ComplexFlt>& b) const {
    // 返回复数向量的乘积
    //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
#if 1
    // 如果条件满足，则执行以下代码块

    // 计算 el_mergeo() 和 el_mergee() 的结果并存储到 vi 和 vr 中
    auto vi = b.el_mergeo();
    auto vr = b.el_mergee();
    
    // vi 应用 rsign_mask 的按位异或操作
    vi = vi ^ rsign_mask;

    // 计算 elwise_mult(vr) 的结果并存储到 ret 中
    auto ret = elwise_mult(vr);

    // 获取 el_swapped() 的结果并存储到 vx_swapped 中
    auto vx_swapped = el_swapped();

    // 使用 el_madd(vi, ret) 计算最终结果并存储到 ret 中
    ret = vx_swapped.el_madd(vi, ret);

    // 返回计算结果 ret
    return ret;

#else

    // 如果条件不满足，则执行以下代码块

    // 计算 elwise_mult(b) 的结果并存储到 ac_bd 中
    auto ac_bd = elwise_mult(b);

    // 获取 el_swapped() 的结果并存储到 d_c 中
    auto d_c = b.el_swapped();

    // d_c 应用 isign_mask 的按位异或操作
    d_c = d_c ^ isign_mask;

    // 计算 elwise_mult(d_c) 的结果并存储到 ad_bc 中
    auto ad_bc = elwise_mult(d_c);

    // 调用 horizontal_sub_permD8(ac_bd, ad_bc) 计算最终结果并返回
    auto ret = horizontal_sub_permD8(ac_bd, ad_bc);
    return ret;

#endif
  }

  Vectorized<ComplexFlt> inline operator/(const Vectorized<ComplexFlt>& b) const {
    // 重载除法运算符，实现复数向量的除法操作
    // 计算公式为: (a + bi) / (c + di)

    // 构造 Vectorized 对象 fabs_cd 和 fabs_dc，存储 |c| 和 |d| 的绝对值
    auto fabs_cd = Vectorized{
      vec_andc(b._vec0, sign_mask),   // |c| 
      vec_andc(b._vec1, sign_mask)};  // |d|
      
    // 获取 fabs_cd 的交换元素存储到 fabs_dc 中，即 |d| 和 |c|
    auto fabs_dc = fabs_cd.el_swapped();

    // 计算最大值 scale = max(|c|, |d|)
    auto scale = fabs_cd.elwise_max(fabs_dc);

    // 分别计算 a2 和 b2，即将当前对象和 b 向量除以 scale
    auto a2 = elwise_div(scale);    // a/sc, b/sc
    auto b2 = b.elwise_div(scale);  // c/sc, d/sc

    // 计算 acbd2 = (ac/sc^2, bd/sc^2)
    auto acbd2 = a2.elwise_mult(b2);

    // 获取 b2 的交换元素并应用 rsign_mask 进行按位异或操作
    auto dc2 = b2.el_swapped();               
    dc2 = dc2 ^ rsign_mask;

    // 计算 adbc2 = (-ad/sc^2, bc/sc^2)
    auto adbc2 = a2.elwise_mult(dc2);

    // 调用 horizontal_add(acbd2, adbc2) 计算 (ac+bd)/sc^2 和 (bc-ad)/sc^2
    auto ret = horizontal_add(acbd2, adbc2);

    // 计算分母的平方 denom2 = (c^2+d^2)/sc^2
    auto denom2 = b2.abs_2_();

    // 计算最终结果 ret = ret / denom2
    ret = ret.elwise_div(denom2);

    // 返回计算结果 ret
    return ret;
  }

  Vectorized<ComplexFlt> asin() const {
    // 计算反正弦函数 asin(x) 的值

    // 计算共轭向量并存储到 conj 中
    auto conj = conj_();

    // 获取 conj 的交换元素并存储到 b_a 中
    auto b_a = conj.el_swapped();

    // 计算 conj 和 b_a 的逐元素乘积存储到 ab 中
    auto ab = conj.elwise_mult(b_a);

    // 计算 im = 2 * ab，即 bc - ad
    auto im = ab + ab;

    // 计算 val_2 = (*this) * (*this)，即当前对象的模的平方
    auto val_2 = (*this).elwise_mult(*this);

    // 获取 val_2 的交换元素并存储到 val_2_swapped 中
    auto val_2_swapped = val_2.el_swapped();

    // 计算 re = horizontal_sub_permD8(val_2, val_2_swapped)，即 a^2 - b^2
    auto re = horizontal_sub_permD8(val_2, val_2_swapped);
    re = Vectorized<ComplexFlt>(one) - re;

    // 计算平方根 root = sqrt(re + im)
    auto root = el_blend<0xAA>(re, im).sqrt();

    // 计算 ln = ln(b_a + root)
    auto ln = (b_a + root).log();

    // 返回 ln 的共轭向量的交换元素
    return ln.el_swapped().conj();
#else
    // 如果条件不满足，则调用标准库函数 std::asin
    return map(std::asin);
#endif
  }

  Vectorized<ComplexFlt> exp() const {
    // 计算复数向量的指数函数 exp(x)
    return map(std::exp);
  }

  Vectorized<ComplexFlt> exp2() const {
    // 计算复数向量的2的指数函数 exp2(x)
    return map(exp2_impl);
  }

  Vectorized<ComplexFlt> expm1() const {
    // 计算复数向量的 exp(x) - 1 函数 expm1(x)
    return map(std::expm1);
  }

  Vectorized<ComplexFlt> eq(const Vectorized<ComplexFlt>& other) const {
    // 判断两个复数向量是否相等

    // 获取当前对象和 other 的相等比较结果
    auto eq = (*this == other);

    // 如果实部和虚部都相等，则返回全1向量，否则返回全0向量
    return (eq.real() & eq.imag()) & one;
  }

  Vectorized<ComplexFlt> ne(const Vectorized<ComplexFlt>& other) const {
    // 判断两个复数向量是否不相等

    // 获取当前对象和 other 的不相等比较结果
    auto ne = (*this != other);

    // 如果实部和虚部有任意一个不相等，则返回全1向量，否则返回全0向量
    return (ne.real() | ne.imag()) & one;
  }
  // 如果实部或虚部任一不相等，则复数不相等
  return (ne.real() | ne.imag()) & one;
}

// 返回一个向量，其中每个元素是对当前复数向量的每个元素取符号函数后的结果
Vectorized<ComplexFlt> sgn() const {
  return map(at::native::sgn_impl);
}

// 比较当前复数向量与另一个复数向量的元素，如果小于则抛出错误
Vectorized<ComplexFlt> operator<(const Vectorized<ComplexFlt>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 比较当前复数向量与另一个复数向量的元素，如果小于等于则抛出错误
Vectorized<ComplexFlt> operator<=(const Vectorized<ComplexFlt>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 比较当前复数向量与另一个复数向量的元素，如果大于则抛出错误
Vectorized<ComplexFlt> operator>(const Vectorized<ComplexFlt>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 比较当前复数向量与另一个复数向量的元素，如果大于等于则抛出错误
Vectorized<ComplexFlt> operator>=(const Vectorized<ComplexFlt>& other) const {
  TORCH_CHECK(false, "not supported for complex numbers");
}

// 定义复数向量的相等运算，使用特定的向量化比较函数
DEFINE_MEMBER_OP(operator==, ComplexFlt, vec_cmpeq)
// 定义复数向量的不等运算，使用特定的向量化比较函数
DEFINE_MEMBER_OP(operator!=, ComplexFlt, vec_cmpne)

// 定义复数向量的加法操作，使用特定的向量化加法函数
DEFINE_MEMBER_OP(operator+, ComplexFlt, vec_add)
// 定义复数向量的减法操作，使用特定的向量化减法函数
DEFINE_MEMBER_OP(operator-, ComplexFlt, vec_sub)
// 定义复数向量的按位与操作，使用特定的向量化按位与函数
DEFINE_MEMBER_OP(operator&, ComplexFlt, vec_and)
// 定义复数向量的按位或操作，使用特定的向量化按位或函数
DEFINE_MEMBER_OP(operator|, ComplexFlt, vec_or)
// 定义复数向量的按位异或操作，使用特定的向量化按位异或函数
DEFINE_MEMBER_OP(operator^, ComplexFlt, vec_xor)

// 定义复数向量的逐元素乘法操作，使用特定的向量化乘法函数
DEFINE_MEMBER_OP(elwise_mult, ComplexFlt, vec_mul)
// 定义复数向量的逐元素除法操作，使用特定的向量化除法函数
DEFINE_MEMBER_OP(elwise_div, ComplexFlt, vec_div)
// 定义复数向量的逐元素大于操作，使用特定的向量化大于比较函数
DEFINE_MEMBER_OP(elwise_gt, ComplexFlt, vec_cmpgt)
// 定义复数向量的逐元素大于等于操作，使用特定的向量化大于等于比较函数
DEFINE_MEMBER_OP(elwise_ge, ComplexFlt, vec_cmpge)
// 定义复数向量的逐元素小于操作，使用特定的向量化小于比较函数
DEFINE_MEMBER_OP(elwise_lt, ComplexFlt, vec_cmplt)
// 定义复数向量的逐元素小于等于操作，使用特定的向量化小于等于比较函数
DEFINE_MEMBER_OP(elwise_le, ComplexFlt, vec_cmple)
// 定义复数向量的逐元素最大值操作，使用特定的向量化最大值函数
DEFINE_MEMBER_OP(elwise_max, ComplexFlt, vec_max)
};

// 特化模板，计算两个复数向量化对象的最大值
template <>
Vectorized<ComplexFlt> inline maximum(
    const Vectorized<ComplexFlt>& a,
    const Vectorized<ComplexFlt>& b) {
  // 计算向量 a 和 b 的模长的平方
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  
  // 使用元素级别比较操作得到比较结果的掩码
  auto mask = abs_a.elwise_lt(abs_b);
  // 根据掩码选择较大的值构成结果向量
  auto max = Vectorized<ComplexFlt>::elwise_blendv(a, b, mask);

  // 返回计算得到的最大值向量
  return max;
  // 利用所有位均为1表示NaN的特性
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(max, isnan);
}

// 特化模板，计算两个复数向量化对象的最小值
template <>
Vectorized<ComplexFlt> inline minimum(
    const Vectorized<ComplexFlt>& a,
    const Vectorized<ComplexFlt>& b) {
  // 计算向量 a 和 b 的模长的平方
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  
  // 使用元素级别比较操作得到比较结果的掩码
  auto mask = abs_a.elwise_gt(abs_b);
  // 根据掩码选择较小的值构成结果向量
  auto min = Vectorized<ComplexFlt>::elwise_blendv(a, b, mask);
  
  // 返回计算得到的最小值向量
  return min;
  // 利用所有位均为1表示NaN的特性
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(min, isnan);
}

// 命名空间闭合
} // namespace
} // namespace vec
} // namespace at
```