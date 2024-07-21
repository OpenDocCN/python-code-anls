# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_float_vsx.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/cpu/vec/intrinsics.h>
// 包含 ATen 库中 CPU 矢量操作的内部函数声明

#include <ATen/cpu/vec/vec_base.h>
// 包含 ATen 库中 CPU 矢量操作的基础类声明

#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
// 包含 ATen 库中 CPU 256 位矢量操作的 VSX 架构助手函数声明

#include <sleef.h>
// 包含 Sleef 库，提供了一些数学函数的 SIMD 实现

namespace at {
namespace vec {
// 进入 ATen 库中矢量操作的命名空间

// See Note [CPU_CAPABILITY namespace]
// 参见文档中有关 CPU 能力的命名空间的说明

inline namespace CPU_CAPABILITY {
// 进入 CPU_CAPABILITY 命名空间

template <>
class Vectorized<float> {
// Vectorized 模板特化为 float 类型

 private:
  union {
    struct {
      vfloat32 _vec0;
      vfloat32 _vec1;
    };
    // 使用联合体定义两个 vfloat32 类型的成员 _vec0 和 _vec1
    struct {
      vbool32 _vecb0;
      vbool32 _vecb1;
    };
    // 使用联合体定义两个 vbool32 类型的成员 _vecb0 和 _vecb1
  } __attribute__((__may_alias__));
  // 指定联合体可以使用可能的别名优化

 public:
  using value_type = float;
  using vec_internal_type = vfloat32;
  using vec_internal_mask_type = vbool32;
  using size_type = int;

  static constexpr size_type size() {
    return 8;
  }
  // 返回矢量长度常量 8

  Vectorized() {}
  // 默认构造函数

  C10_ALWAYS_INLINE Vectorized(vfloat32 v) : _vec0{v}, _vec1{v} {}
  // 构造函数，接受一个 vfloat32 类型参数 v，并将其分别赋值给 _vec0 和 _vec1

  C10_ALWAYS_INLINE Vectorized(vbool32 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  // 构造函数，接受一个 vbool32 类型参数 vmask，并将其分别赋值给 _vecb0 和 _vecb1

  C10_ALWAYS_INLINE Vectorized(vfloat32 v1, vfloat32 v2) : _vec0{v1}, _vec1{v2} {}
  // 构造函数，接受两个 vfloat32 类型参数 v1 和 v2，并分别赋值给 _vec0 和 _vec1

  C10_ALWAYS_INLINE Vectorized(vbool32 v1, vbool32 v2) : _vecb0{v1}, _vecb1{v2} {}
  // 构造函数，接受两个 vbool32 类型参数 v1 和 v2，并分别赋值给 _vecb0 和 _vecb1

  C10_ALWAYS_INLINE Vectorized(float scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  // 构造函数，接受一个 float 标量参数 scalar，并使用 vec_splats 将其广播为两个 vfloat32 向量 _vec0 和 _vec1

  C10_ALWAYS_INLINE Vectorized(
      float scalar1,
      float scalar2,
      float scalar3,
      float scalar4,
      float scalar5,
      float scalar6,
      float scalar7,
      float scalar8)
      : _vec0{vfloat32{scalar1, scalar2, scalar3, scalar4}},
        _vec1{vfloat32{scalar5, scalar6, scalar7, scalar8}} {}
  // 构造函数，接受八个 float 标量参数，分别构造两个 vfloat32 向量 _vec0 和 _vec1

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  // 返回 _vec0 的引用作为 vec_internal_type 类型

  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }
  // 返回 _vec1 的引用作为 vec_internal_type 类型

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 0, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return a;
  }
  // blend 函数模板的特化实现，当 blendChoice(mask) 等于 0 时返回 a

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 1, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return b;
  }
  // blend 函数模板的特化实现，当 blendChoice(mask) 等于 1 时返回 b

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 2, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return {b._vec0, a._vec1};
  }
  // blend 函数模板的特化实现，当 blendChoice(mask) 等于 2 时返回一个新的 Vectorized<float> 对象，其中 _vec0 取自 b，_vec1 取自 a

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 3, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return {a._vec0, b._vec1};
  }
  // blend 函数模板的特化实现，当 blendChoice(mask) 等于 3 时返回一个新的 Vectorized<float> 对象，其中 _vec0 取自 a，_vec1 取自 b

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 4, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_1st = VsxMask1(mask);
    return {(vfloat32)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1};
  }
  // blend 函数模板的特化实现，当 blendChoice(mask) 等于 4 时返回一个新的 Vectorized<float> 对象，根据 mask 混合 a 和 b 的 _vec0，并保留 a 的 _vec1

  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 5, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_1st = VsxMask1(mask);
    // 使用 mask 构造一个 vbool32 类型的 mask_1st
  return {(vfloat32)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1};



  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 6, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_2nd = VsxMask2(mask);
    // 根据模板参数生成特定掩码
    return {a._vec0, (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 7, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_2nd = VsxMask2(mask);
    // 根据模板参数生成特定掩码
    return {b._vec0, (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoice(mask) == 8, Vectorized<float>> C10_ALWAYS_INLINE
  blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    const vbool32 mask_1st = VsxMask1(mask);
    const vbool32 mask_2nd = VsxMask2(mask);
    // 根据模板参数生成特定掩码，并将两个向量按位混合
    return {
        (vfloat32)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat32)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }



  static Vectorized<float> C10_ALWAYS_INLINE blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask) {
    // 使用给定的掩码向量，对两个向量的对应元素进行混合
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }



  template <typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    // 生成一系列等差排列的浮点数向量
    return Vectorized<float>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
  }



  static Vectorized<float> set(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      size_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        // 混合两个向量，使用掩码1
        return blend<1>(a, b);
      case 2:
        // 混合两个向量，使用掩码3
        return blend<3>(a, b);
      case 3:
        // 混合两个向量，使用掩码7
        return blend<7>(a, b);
      case 4:
        // 混合两个向量，使用掩码15
        return blend<15>(a, b);
      case 5:
        // 混合两个向量，使用掩码31
        return blend<31>(a, b);
      case 6:
        // 混合两个向量，使用掩码63
        return blend<63>(a, b);
      case 7:
        // 混合两个向量，使用掩码127
        return blend<127>(a, b);
    }

    // 默认情况下返回向量 b
    return b;
  }



  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      // 从指针加载数据到两个向量中，使用指定的偏移量
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
    }

    // 如果加载数量小于向量大小，使用临时数组加载数据
    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }



  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 如果向量中元素个数与 size() 相等，则直接将 _vec0 和 _vec1 的数据存储到 ptr 所指向的内存位置
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      // 否则，创建一个临时数组 tmp_values，将 _vec0 和 _vec1 的数据存储到这个数组中
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, tmp_values);
      vec_vsx_st(_vec1, offset16, tmp_values);
      // 将 tmp_values 中的部分数据（最多 count 个）复制到 ptr 所指向的内存位置
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }

  const float& operator[](int idx) const = delete;
  float& operator[](int idx) = delete;

  Vectorized<float> map(float (*const f)(float)) const {
    // 对当前对象中的每一半元素应用函数 f，并将结果存储到新创建的 Vectorized<float> 对象中返回
    Vectorized<float> ret;
    for (int i = 0; i < size() / 2; i++) {
      ret._vec0[i] = f(_vec0[i]);
    }
    for (int i = 0; i < size() / 2; i++) {
      ret._vec1[i] = f(_vec1[i]);
    }
    return ret;
  }

  Vectorized<float> mapbi(float (*const f)(float, float), const Vectorized<float>& other)
      const {
    // 对当前对象和另一个给定的 Vectorized<float> 对象中的每一半元素应用函数 f，并将结果存储到新创建的 Vectorized<float> 对象中返回
    Vectorized<float> ret;
    for (int i = 0; i < size() / 2; i++) {
      ret._vec0[i] = f(_vec0[i], other._vec0[i]);
    }
    for (int i = 0; i < size() / 2; i++) {
      ret._vec1[i] = f(_vec1[i], other._vec1[i]);
    }
    return ret;
  }

  Vectorized<float> _nor() const {
    // 返回当前对象中每一半元素的逻辑 NOR 操作结果组成的新的 Vectorized<float> 对象
    return {vec_nor(_vec0, _vec0), vec_nor(_vec1, _vec1)};
  }

  Vectorized<float> isnan() const {
    // 返回一个新的 Vectorized<float> 对象，表示当前对象中每一半元素是否为 NaN 的检测结果
    auto x = *this;
    auto ret = (x == x);  // 检测每一半元素是否等于自身，用于检测 NaN
    return ret._nor();   // 对结果执行逻辑 NOR 操作，得到最终返回结果
  }

  bool has_inf_nan() const {
    // 检查当前对象中每一半元素是否包含无穷大或 NaN
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec0[i]) || _isinf(_vec0[i])) {
        return true;
      }
    }
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec1[i]) || _isinf(_vec1[i])) {
        return true;
      }
    }
    return false;  // 如果没有发现无穷大或 NaN，则返回 false
  }

  int zero_mask() const {
    // 返回一个整数掩码，其中所有零元素被转换为 1 位，其他元素被转换为 0 位
    auto cmp = (*this == zero);  // 检查当前对象中每一半元素是否等于零
    vuint64 result0 = vec_vbpermq((vuint8)cmp._vecb0, mask_zero_bits);
    vuint64 result1 = vec_vbpermq((vuint8)cmp._vecb1, mask_zero_bits);
    return (result0[1] >> 12 | (result1[1] >> 8));  // 返回最终的整数掩码
  }

  Vectorized<float> C10_ALWAYS_INLINE abs() const {
    // 返回当前对象中每一半元素的绝对值组成的新的 Vectorized<float> 对象
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  Vectorized<float> C10_ALWAYS_INLINE acos() const {
    // 返回当前对象中每一半元素的反余弦值组成的新的 Vectorized<float> 对象
    return {Sleef_acosf4_u10(_vec0), Sleef_acosf4_u10(_vec1)};
  }

  Vectorized<float> C10_ALWAYS_INLINE asin() const {
    // 返回当前对象中每一半元素的反正弦值组成的新的 Vectorized<float> 对象
    return {Sleef_asinf4_u10(_vec0), Sleef_asinf4_u10(_vec1)};
  }

  Vectorized<float> atan() const {
    // 返回当前对象中每一半元素的反正切值组成的新的 Vectorized<float> 对象
    return {Sleef_atanf4_u10(_vec0), Sleef_atanf4_u10(_vec1)};
  }

  Vectorized<float> atanh() const {
    // 返回当前对象中每一半元素的反双曲正切值组成的新的 Vectorized<float> 对象
    return {Sleef_atanhf4_u10(_vec0), Sleef_atanhf4_u10(_vec1)};
  }

  Vectorized<float> atan2(const Vectorized<float>& b) const {
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量和参数向量每个元素 atan2 计算结果
  Vectorized<float> atan2(const Vectorized<float> &b) const {
    return {Sleef_atan2f4_u10(_vec0, b._vec0), Sleef_atan2f4_u10(_vec1, b._vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量和参数向量每个元素按位复制符号的结果
  Vectorized<float> copysign(const Vectorized<float> &sign) const {
    return {Sleef_copysignf4(_vec0, sign._vec0), Sleef_copysignf4(_vec1, sign._vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 lgamma 计算结果
  Vectorized<float> lgamma() const {
    return {Sleef_lgammaf4_u10(_vec0), Sleef_lgammaf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 erf 计算结果
  Vectorized<float> erf() const {
    return {Sleef_erff4_u10(_vec0), Sleef_erff4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 erfc 计算结果
  Vectorized<float> erfc() const {
    return {Sleef_erfcf4_u15(_vec0), Sleef_erfcf4_u15(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 erfinv 计算结果
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的角度计算结果
  Vectorized<float> angle() const {
    auto tmp = blendv(
      Vectorized<float>(0), Vectorized<float>(c10::pi<float>), *this < Vectorized<float>(0));
    return blendv(tmp, *this, isnan());
  }
  // 返回当前向量本身，即返回一个新的 Vectorized<float> 对象，包含当前向量的实部
  Vectorized<float> real() const {
    return *this;
  }
  // 返回一个新的 Vectorized<float> 对象，其所有元素为零，即返回当前向量的虚部
  Vectorized<float> imag() const {
    return Vectorized<float>{0};
  }
  // 返回当前向量本身，即返回一个新的 Vectorized<float> 对象，包含当前向量的共轭
  Vectorized<float> conj() const {
    return *this;
  }

  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 exp 计算结果
  Vectorized<float> C10_ALWAYS_INLINE exp() const {
    return {Sleef_expf4_u10(_vec0), Sleef_expf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 exp2 计算结果
  Vectorized<float> C10_ALWAYS_INLINE exp2() const {
    return {Sleef_exp2f4_u10(_vec0), Sleef_exp2f4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 expm1 计算结果
  Vectorized<float> expm1() const {
    return {Sleef_expm1f4_u10(_vec0), Sleef_expm1f4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 exp 计算结果（别名函数）
  Vectorized<float> C10_ALWAYS_INLINE exp_u20() const {
    return exp();
  }

  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 log 计算结果
  Vectorized<float> C10_ALWAYS_INLINE log() const {
    return {Sleef_logf4_u10(_vec0), Sleef_logf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 log10 计算结果
  Vectorized<float> C10_ALWAYS_INLINE log10() const {
    return {Sleef_log10f4_u10(_vec0), Sleef_log10f4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 log1p 计算结果
  Vectorized<float> C10_ALWAYS_INLINE log1p() const {
    return {Sleef_log1pf4_u10(_vec0), Sleef_log1pf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 log2 计算结果
  Vectorized<float> C10_ALWAYS_INLINE log2() const {
    return {Sleef_log2f4_u10(_vec0), Sleef_log2f4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的向上取整结果
  Vectorized<float> C10_ALWAYS_INLINE ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 cos 计算结果
  Vectorized<float> C10_ALWAYS_INLINE cos() const {
    return {Sleef_cosf4_u10(_vec0), Sleef_cosf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 cosh 计算结果
  Vectorized<float> C10_ALWAYS_INLINE cosh() const {
    return {Sleef_coshf4_u10(_vec0), Sleef_coshf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的向下取整结果
  Vectorized<float> C10_ALWAYS_INLINE floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的取负结果
  Vectorized<float> C10_ALWAYS_INLINE neg() const {
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }

  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的四舍五入结果
  Vectorized<float> C10_ALWAYS_INLINE round() const {
    return {vec_round(_vec0), vec_round(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 sin 计算结果
  Vectorized<float> C10_ALWAYS_INLINE sin() const {
    return {Sleef_sinf4_u10(_vec0), Sleef_sinf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 sinh 计算结果
  Vectorized<float> C10_ALWAYS_INLINE sinh() const {
    return {Sleef_sinhf4_u10(_vec0), Sleef_sinhf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 tan 计算结果
  Vectorized<float> C10_ALWAYS_INLINE tan() const {
    return {Sleef_tanf4_u10(_vec0), Sleef_tanf4_u10(_vec1)};
  }
  // 返回一个新的 Vectorized<float> 对象，包含当前向量每个元素的 tanh 计算结果
  Vectorized<float> C10_ALWAYS_INLINE tanh() const {
  Vectorized<float> C10_ALWAYS_INLINE trunc() const {
    // 返回向下取整的结果，即返回每个向量元素的整数部分
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<float> C10_ALWAYS_INLINE frac() const {
    // 返回向下取整后的小数部分，即返回每个向量元素减去其整数部分的结果
    return *this - trunc();
  }

  Vectorized<float> C10_ALWAYS_INLINE sqrt() const {
    // 返回每个向量元素的平方根
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }

  Vectorized<float> C10_ALWAYS_INLINE reciprocal() const {
    // 返回每个向量元素的倒数
    return Vectorized<float>(one) / (*this);
  }

  Vectorized<float> C10_ALWAYS_INLINE rsqrt() const {
    // 返回每个向量元素的平方根的倒数
    return sqrt().reciprocal();
  }

  Vectorized<float> C10_ALWAYS_INLINE pow(const Vectorized<float>& exp) const {
    // 返回每个向量元素的 exp 次幂
    return {Sleef_powf4_u10(_vec0, exp._vec0), Sleef_powf4_u10(_vec1, exp._vec1)};
  }

  Vectorized<float> fmod(const Vectorized<float>& b) const {
    // 返回每个向量元素与 b 对应元素取模的结果
    return {Sleef_fmodf4(_vec0, b._vec0), Sleef_fmodf4(_vec1, b._vec1)};
  }

  Vectorized<float> hypot(const Vectorized<float>& b) const {
    // 返回每个向量元素与 b 对应元素的直角三角形斜边长度的结果
    return {Sleef_hypotf4_u05(_vec0, b._vec0), Sleef_hypotf4_u05(_vec1, b._vec1)};
  }

  Vectorized<float> nextafter(const Vectorized<float>& b) const {
    // 返回每个向量元素在 b 方向上最接近的浮点数
    return {Sleef_nextafterf4(_vec0, b._vec0), Sleef_nextafterf4(_vec1, b._vec1)};
  }

  Vectorized<float> igamma(const Vectorized<float>& x) const {
    // 返回每个向量元素的完全伽马函数值
    return mapbi(calc_igamma, x);
  }

  Vectorized<float> igammac(const Vectorized<float>& x) const {
    // 返回每个向量元素的伽马函数的补函数值
    return mapbi(calc_igammac, x);
  }

  Vectorized<float> i0() const {
    // 返回每个向量元素的修正贝塞尔函数 I0
    return map(calc_i0);
  }

  Vectorized<float> i0e() const {
    // 返回每个向量元素的指数修正贝塞尔函数 I0e
    return map(calc_i0e);
  }

  Vectorized<float> digamma() const {
    // 返回每个向量元素的对数伽马函数值
    return map(calc_digamma);
  }

  // 下面是一系列宏定义的运算符重载和函数操作

  DEFINE_MEMBER_OP(operator==, float, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, float, vec_cmpne)
  DEFINE_MEMBER_OP(operator<, float, vec_cmplt)
  DEFINE_MEMBER_OP(operator<=, float, vec_cmple)
  DEFINE_MEMBER_OP(operator>, float, vec_cmpgt)
  DEFINE_MEMBER_OP(operator>=, float, vec_cmpge)
  DEFINE_MEMBER_OP_AND_ONE(eq, float, vec_cmpeq)
  DEFINE_MEMBER_OP_AND_ONE(ne, float, vec_cmpne)
  DEFINE_MEMBER_OP_AND_ONE(lt, float, vec_cmplt)
  DEFINE_MEMBER_OP_AND_ONE(le, float, vec_cmple)
  DEFINE_MEMBER_OP_AND_ONE(gt, float, vec_cmpgt)
  DEFINE_MEMBER_OP_AND_ONE(ge, float, vec_cmpge)
  DEFINE_MEMBER_OP(operator+, float, vec_add)
  DEFINE_MEMBER_OP(operator-, float, vec_sub)
  DEFINE_MEMBER_OP(operator*, float, vec_mul)
  DEFINE_MEMBER_OP(operator/, float, vec_div)
  DEFINE_MEMBER_OP(maximum, float, vec_max_nan2)
  DEFINE_MEMBER_OP(minimum, float, vec_min_nan2)
  DEFINE_MEMBER_OP(operator&, float, vec_and)
  DEFINE_MEMBER_OP(operator|, float, vec_or)
  DEFINE_MEMBER_OP(operator^, float, vec_xor)
  DEFINE_MEMBER_TERNARY_OP(madd, float, vec_madd)
};

// 结束 vec 命名空间
template <>
// 特化模板：返回两个 Vectorized<float> 中的最大值
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
    return a.maximum(b);
}

// 结束 vec 命名空间
template <>
// 特化模板：返回两个 Vectorized<float> 中的最小值
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
    return a.minimum(b);
}

// 结束 at 命名空间
} // namespace
// 结束 vec 命名空间
} // namespace vec
// 结束 at 命名空间
} // namespace at
```