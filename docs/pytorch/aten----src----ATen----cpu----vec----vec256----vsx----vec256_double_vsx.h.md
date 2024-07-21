# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_double_vsx.h`

```py
#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <c10/util/irange.h>

#include <sleef.h>

namespace at {
namespace vec {

// 向量化操作的命名空间，适用于 CPU_CAPABILITY 特性
inline namespace CPU_CAPABILITY {

// 模板特化，用于双精度浮点数 double 的向量化操作
template <>
class Vectorized<double> {
 private:
  union {
    // 双精度浮点数的两个向量
    struct {
      vfloat64 _vec0;
      vfloat64 _vec1;
    };
    // 布尔掩码的两个向量
    struct {
      vbool64 _vecb0;
      vbool64 _vecb1;
    };

  } __attribute__((__may_alias__));  // 使用 __may_alias__ 属性允许联合使用不同类型的成员

 public:
  using value_type = double;  // 向量化的值类型为 double
  using vec_internal_type = vfloat64;  // 向量内部类型为 vfloat64
  using vec_internal_mask_type = vbool64;  // 向量布尔掩码类型为 vbool64
  using size_type = int;  // 大小类型为 int
  static constexpr size_type size() {  // 返回向量大小为 4
    return 4;
  }
  Vectorized() {}  // 默认构造函数

  // 构造函数，接受一个向量作为参数，初始化两个内部向量
  C10_ALWAYS_INLINE Vectorized(vfloat64 v) : _vec0{v}, _vec1{v} {}
  // 构造函数，接受一个布尔掩码向量作为参数，初始化两个布尔掩码内部向量
  C10_ALWAYS_INLINE Vectorized(vbool64 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  // 构造函数，接受两个向量作为参数，分别初始化两个内部向量
  C10_ALWAYS_INLINE Vectorized(vfloat64 v1, vfloat64 v2) : _vec0{v1}, _vec1{v2} {}
  // 构造函数，接受两个布尔掩码向量作为参数，分别初始化两个布尔掩码内部向量
  C10_ALWAYS_INLINE Vectorized(vbool64 v1, vbool64 v2) : _vecb0{v1}, _vecb1{v2} {}
  // 构造函数，接受一个标量 double 参数，使用 vec_splats 创建一个相同值的向量初始化两个内部向量
  C10_ALWAYS_INLINE Vectorized(double scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  // 构造函数，接受四个标量 double 参数，分别初始化两个内部向量
  C10_ALWAYS_INLINE Vectorized(
      double scalar1,
      double scalar2,
      double scalar3,
      double scalar4)
      : _vec0{vfloat64{scalar1, scalar2}}, _vec1{vfloat64{scalar3, scalar4}} {}

  // 返回第一个内部向量的引用
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  // 返回第二个内部向量的引用
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  // 返回零掩码
  int zero_mask() const {
    // 比较当前向量是否等于全零向量 vd_zero
    auto cmp = (*this == vd_zero);
    return (cmp._vecb0[0] & 1) | (cmp._vecb0[1] & 2) | (cmp._vecb1[0] & 4) |
        (cmp._vecb1[1] & 8);


    // 使用位运算将来自四个向量（_vecb0 和 _vecb1）的特定位合并成一个整数返回
    return (cmp._vecb0[0] & 1) | (cmp._vecb0[1] & 2) | (cmp._vecb1[0] & 4) |
        (cmp._vecb1[1] & 8);



  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 0, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return a;
  }


  // 如果 blendChoiceDbl(mask) 的结果为 0，则返回向量 a；否则返回向量 b
  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 0, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return a;
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 1, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return b;
  }


  // 如果 blendChoiceDbl(mask) 的结果为 1，则返回向量 b；否则返回向量 a
  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 1, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return b;
  }


（以下模板函数 blend 类似，根据不同的 blendChoiceDbl(mask) 结果返回不同的向量组合。根据示例，每个模板函数应添加类似的注释解释其作用。）


  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 2, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return { b._vec0, a._vec1 };
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 3, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      return { a._vec0, b._vec1 };
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 4, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_1st = VsxDblMask1(mask);
      return { (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st), a._vec1 };
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 5, Vectorized<double>> C10_ALWAYS_INLINE
      blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_1st = VsxDblMask1(mask);
      return { (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st), b._vec1 };
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 6,
      Vectorized<double>>
      C10_ALWAYS_INLINE blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_2nd = VsxDblMask2(mask);
      // 生成的掩码用于向量选择操作
      return { a._vec0,
          (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd) };
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 7,
      Vectorized<double>>
      C10_ALWAYS_INLINE blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_2nd = VsxDblMask2(mask);
      // 生成的掩码用于向量选择操作
      return { b._vec0,
          (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd) };
  }



  template <int64_t mask>
  static std::enable_if_t<blendChoiceDbl(mask) == 8, Vectorized<double>>
      C10_ALWAYS_INLINE blend(const Vectorized<double>& a, const Vectorized<double>& b) {
      const vbool64 mask_1st = VsxDblMask1(mask);
      const vbool64 mask_2nd = VsxDblMask2(mask);
      // 生成的掩码用于两个向量的选择操作
      return {
          (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st),
          (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd) };
  }



  static Vectorized<double> C10_ALWAYS_INLINE blendv(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      const Vectorized<double>& mask) {
    // 这里使用的掩码是通过比较 vec256 生成的
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }



// 返回两个向量根据掩码选择的结果向量
template <typename step_t>
static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
  // 返回一个包含四个 double 类型元素的向量，分别为 base, base+step, base+2*step, base+3*step
  return Vectorized<double>(base, base + step, base + 2 * step, base + 3 * step);
}



static Vectorized<double> C10_ALWAYS_INLINE
set(const Vectorized<double>& a, const Vectorized<double>& b, size_t count = size()) {
  switch (count) {
    case 0:
      return a;
    case 1:
      // 使用 blend 函数根据掩码 1 返回混合后的向量
      return blend<1>(a, b);
    case 2:
      // 使用 blend 函数根据掩码 3 返回混合后的向量
      return blend<3>(a, b);
    case 3:
      // 使用 blend 函数根据掩码 7 返回混合后的向量
      return blend<7>(a, b);
  }

  // 默认返回 b 向量
  return b;
}



static Vectorized<value_type> C10_ALWAYS_INLINE
loadu(const void* ptr, int count = size()) {
  if (count == size()) {
    // 从内存地址 ptr 处加载两个向量，使用不对齐加载方式
    return {
        vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
        vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
  }

  // 如果 count 小于 size，创建临时数组 tmp_values 并从 ptr 复制数据，再加载两个向量
  __at_align__ value_type tmp_values[size()] = {};
  std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

  return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
}



void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
  if (count == size()) {
    // 将当前对象的两个向量分别存储到 ptr 所指向的内存地址中，使用不对齐存储方式
    vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
    vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
  } else if (count > 0) {
    // 如果 count 小于 size，创建临时数组 tmp_values，存储两个向量数据，并复制到 ptr
    __at_align__ value_type tmp_values[size()];
    vec_vsx_st(_vec0, offset0, tmp_values);
    vec_vsx_st(_vec1, offset16, tmp_values);
    std::memcpy(
        ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
  }
}



const double& operator[](int idx) const = delete;
double& operator[](int idx) = delete;



Vectorized<double> map(double (*const f)(double)) const {
  // 使用函数 f 映射当前对象的两个向量中的每个元素，并返回结果向量
  Vectorized<double> ret;
  for (const auto i : c10::irange(size()/2)) {
      ret._vec0[i] = f(_vec0[i]);
  }
  for (const auto i : c10::irange(size()/2)) {
      ret._vec1[i] = f(_vec1[i]);
  }
  return ret;
}



Vectorized<double> mapbi(double (*const f)(double, double), const Vectorized<double>& other)
    const {
  // 使用函数 f 映射当前对象的两个向量和另一个向量 other 中对应元素，并返回结果向量
  Vectorized<double> ret;
  for (const auto i : c10::irange(size()/2)) {
      ret._vec0[i] = f(_vec0[i], other._vec0[i]);
  }
  for (const auto i : c10::irange(size()/2)) {
      ret._vec1[i] = f(_vec1[i], other._vec1[i]);
  }
  return ret;
}



Vectorized<double> C10_ALWAYS_INLINE abs() const {
 `
# 返回一个由向量中各元素绝对值组成的向量
return {vec_abs(_vec0), vec_abs(_vec1)};



# 返回一个由向量中各元素计算 acos 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE acos() const {
   return {Sleef_acosd2_u10(_vec0), Sleef_acosd2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 asin 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE asin() const {
   return {Sleef_asind2_u10(_vec0), Sleef_asind2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 atan 函数值组成的向量
Vectorized<double> atan() const {
   return {Sleef_atand2_u10(_vec0), Sleef_atand2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 atanh 函数值组成的向量
Vectorized<double> atanh() const {
   return {Sleef_atanhd2_u10(_vec0), Sleef_atanhd2_u10(_vec1)};
}



# 返回两个向量各元素对应位置计算 atan2 函数值组成的向量
Vectorized<double> atan2(const Vectorized<double>& b) const {
   return {Sleef_atan2d2_u10(_vec0, b._vec0), Sleef_atan2d2_u10(_vec1, b._vec1)};
}



# 返回两个向量各元素对应位置执行 copysign 操作后的向量
Vectorized<double> copysign(const Vectorized<double> &sign) const {
  return {Sleef_copysignd2(_vec0, sign._vec0), Sleef_copysignd2(_vec1, sign._vec1)};
}



# 返回一个由向量中各元素计算 erf 函数值组成的向量
Vectorized<double> erf() const {
   return {Sleef_erfd2_u10(_vec0), Sleef_erfd2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 erfc 函数值组成的向量
Vectorized<double> erfc() const {
   return {Sleef_erfcd2_u15(_vec0), Sleef_erfcd2_u15(_vec1)};
}



# 返回一个由向量中各元素计算 exp 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE exp() const {
   return {Sleef_expd2_u10(_vec0), Sleef_expd2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 exp2 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE exp2() const {
  return {Sleef_exp2d2_u10(_vec0), Sleef_exp2d2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 expm1 函数值组成的向量
Vectorized<double> expm1() const {
   return {Sleef_expm1d2_u10(_vec0), Sleef_expm1d2_u10(_vec1)};
}



# 返回调用 exp 函数计算结果的向量
Vectorized<double> C10_ALWAYS_INLINE exp_u20() const {
   return exp();
}



# 返回一个由向量中各元素计算 lgamma 函数值组成的向量，忽略未定义行为检查
Vectorized<double> lgamma() const __ubsan_ignore_undefined__ {
   return {Sleef_lgammad2_u10(_vec0), Sleef_lgammad2_u10(_vec1)};
}



# 返回调用 calc_erfinv 函数处理后的结果向量
Vectorized<double> erfinv() const {
  return map(calc_erfinv);
}



# 返回根据向量元素值进行条件混合处理后的结果向量
Vectorized<double> angle() const {
  auto tmp = blendv(
    Vectorized<double>(0), Vectorized<double>(c10::pi<double>), *this < Vectorized<double>(0));
  return blendv(tmp, *this, isnan());
}



# 返回向量本身，表示取实部操作
Vectorized<double> real() const {
  return *this;
}



# 返回一个零向量，表示取虚部操作
Vectorized<double> imag() const {
  return Vectorized<double>{0};
}



# 返回向量本身，表示复共轭操作
Vectorized<double> conj() const {
  return *this;
}



# 返回一个由向量中各元素计算 log 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE log() const {
   return {Sleef_logd2_u10(_vec0), Sleef_logd2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 log10 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE log10() const {
   return {Sleef_log10d2_u10(_vec0), Sleef_log10d2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 log1p 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE log1p() const {
   return {Sleef_log1pd2_u10(_vec0), Sleef_log1pd2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 log2 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE log2() const {
   return {Sleef_log2d2_u10(_vec0), Sleef_log2d2_u10(_vec1)};
}



# 返回一个由向量中各元素向上取整后组成的向量
Vectorized<double> C10_ALWAYS_INLINE ceil() const {
  return {vec_ceil(_vec0), vec_ceil(_vec1)};
}



# 返回一个由向量中各元素计算 cos 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE cos() const {
   return {Sleef_cosd2_u10(_vec0), Sleef_cosd2_u10(_vec1)};
}



# 返回一个由向量中各元素计算 cosh 函数值组成的向量
Vectorized<double> C10_ALWAYS_INLINE cosh() const {
   return {Sleef_coshd2_u10(_vec0), Sleef_coshd2_u10(_vec1)};
}



# 返回一个由向量中各元素向下取整后组成的向量
Vectorized<double> C10_ALWAYS_INLINE floor() const {
  Vectorized<double> C10_ALWAYS_INLINE neg() const {
    // 返回当前向量每个元素的相反数
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE round() const {
    // 返回当前向量每个元素的四舍五入值
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE sin() const {
    // 返回当前向量每个元素的正弦值
    return {Sleef_sind2_u10(_vec0), Sleef_sind2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE sinh() const {
    // 返回当前向量每个元素的双曲正弦值
    return {Sleef_sinhd2_u10(_vec0), Sleef_sinhd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE tan() const {
    // 返回当前向量每个元素的正切值
    return {Sleef_tand2_u10(_vec0), Sleef_tand2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE tanh() const {
    // 返回当前向量每个元素的双曲正切值
    return {Sleef_tanhd2_u10(_vec0), Sleef_tanhd2_u10(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE trunc() const {
    // 返回当前向量每个元素的截断值
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }

  Vectorized<double> C10_ALWAYS_INLINE frac() const {
    // 返回当前向量每个元素的小数部分
    return *this - trunc();
  }

  Vectorized<double> C10_ALWAYS_INLINE sqrt() const {
    // 返回当前向量每个元素的平方根
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE reciprocal() const {
    // 返回当前向量每个元素的倒数
    return {
        vec_div(vd_one, _vec0), // 估算 _vec0 的倒数
        vec_div(vd_one, _vec1)}; // 估算 _vec1 的倒数
  }
  Vectorized<double> C10_ALWAYS_INLINE rsqrt() const {
    // 返回当前向量每个元素的平方根的倒数
    return sqrt().reciprocal();
  }

  Vectorized<double> C10_ALWAYS_INLINE pow(const Vectorized<double>& b) const {
    // 返回当前向量每个元素的指数运算结果
    return {Sleef_powd2_u10(_vec0, b._vec0), Sleef_powd2_u10(_vec1, b._vec1)};
  }
  Vectorized<double> C10_ALWAYS_INLINE fmod(const Vectorized<double>& b) const {
    // 返回当前向量每个元素对 b 中对应元素取模的结果
    return {Sleef_fmodd2(_vec0, b._vec0), Sleef_fmodd2(_vec1, b._vec1)};
  }

  Vectorized<double> hypot(const Vectorized<double>& b) const {
    // 返回当前向量和 b 向量对应元素的 hypot 函数结果
    return {Sleef_hypotd2_u05(_vec0, b._vec0), Sleef_hypotd2_u05(_vec1, b._vec1)};
  }

  Vectorized<double> nextafter(const Vectorized<double>& b) const {
    // 返回当前向量每个元素向 b 中对应元素靠近的下一个浮点数
    return {Sleef_nextafterd2(_vec0, b._vec0), Sleef_nextafterd2(_vec1, b._vec1)};
  }

  Vectorized<double> igamma(const Vectorized<double>& x) const {
    // 对当前向量和 x 向量分别应用 calc_igamma 函数
    return mapbi(calc_igamma, x);
  }

  Vectorized<double> igammac(const Vectorized<double>& x) const {
    // 对当前向量和 x 向量分别应用 calc_igammac 函数
    return mapbi(calc_igammac, x);
  }

  Vectorized<double> i0() const {
    // 对当前向量应用 calc_i0 函数
    return map(calc_i0);
  }

  Vectorized<double> i0e() const {
    // 对当前向量应用 calc_i0e 函数
    return map(calc_i0e);
  }

  Vectorized<double> digamma() const {
    // 对当前向量应用 calc_digamma 函数
    return map(calc_digamma);
  }

  Vectorized<double> _nor() const {
    // 返回当前向量每个元素的按位取反结果
    return {vec_nor(_vec0, _vec0), vec_nor(_vec1, _vec1)};
  }

  Vectorized<double> isnan() const {
    auto x = *this;
    auto ret = (x == x);  // 检查当前向量中是否有 NaN（非数字）
    return ret._nor();    // 返回所有元素的按位取反结果
  }
  bool has_inf_nan() const {
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec0[i]) || _isinf(_vec0[i])) {
        return true;    // 如果_vec0 中存在 NaN 或无穷大，则返回 true
      }
    }
    for (const auto i : c10::irange(size()/2)) {
      if(_isnan(_vec1[i]) || _isinf(_vec1[i])) {
        return true;    // 如果_vec1 中存在 NaN 或无穷大，则返回 true
      }
    }
    return false;       // 如果没有发现 NaN 或无穷大，则返回 false
  }
    // 返回 false
    return false;
  }

  // 定义成员运算符重载函数 operator==，返回 double 类型，对应 vec_cmpeq 函数
  DEFINE_MEMBER_OP(operator==, double, vec_cmpeq)
  // 定义成员运算符重载函数 operator!=，返回 double 类型，对应 vec_cmpne 函数
  DEFINE_MEMBER_OP(operator!=, double, vec_cmpne)
  // 定义成员运算符重载函数 operator<，返回 double 类型，对应 vec_cmplt 函数
  DEFINE_MEMBER_OP(operator<, double, vec_cmplt)
  // 定义成员运算符重载函数 operator<=，返回 double 类型，对应 vec_cmple 函数
  DEFINE_MEMBER_OP(operator<=, double, vec_cmple)
  // 定义成员运算符重载函数 operator>，返回 double 类型，对应 vec_cmpgt 函数
  DEFINE_MEMBER_OP(operator>, double, vec_cmpgt)
  // 定义成员运算符重载函数 operator>=，返回 double 类型，对应 vec_cmpge 函数
  DEFINE_MEMBER_OP(operator>=, double, vec_cmpge)
  // 定义成员运算符重载函数 eq，返回 double 类型，对应 vec_cmpeq 函数，并附加一个固定的参数
  DEFINE_MEMBER_OP_AND_ONE(eq, double, vec_cmpeq)
  // 定义成员运算符重载函数 ne，返回 double 类型，对应 vec_cmpne 函数，并附加一个固定的参数
  DEFINE_MEMBER_OP_AND_ONE(ne, double, vec_cmpne)
  // 定义成员运算符重载函数 lt，返回 double 类型，对应 vec_cmplt 函数，并附加一个固定的参数
  DEFINE_MEMBER_OP_AND_ONE(lt, double, vec_cmplt)
  // 定义成员运算符重载函数 le，返回 double 类型，对应 vec_cmple 函数，并附加一个固定的参数
  DEFINE_MEMBER_OP_AND_ONE(le, double, vec_cmple)
  // 定义成员运算符重载函数 gt，返回 double 类型，对应 vec_cmpgt 函数，并附加一个固定的参数
  DEFINE_MEMBER_OP_AND_ONE(gt, double, vec_cmpgt)
  // 定义成员运算符重载函数 ge，返回 double 类型，对应 vec_cmpge 函数，并附加一个固定的参数
  DEFINE_MEMBER_OP_AND_ONE(ge, double, vec_cmpge)
  // 定义成员运算符重载函数 operator+，返回 double 类型，对应 vec_add 函数
  DEFINE_MEMBER_OP(operator+, double, vec_add)
  // 定义成员运算符重载函数 operator-，返回 double 类型，对应 vec_sub 函数
  DEFINE_MEMBER_OP(operator-, double, vec_sub)
  // 定义成员运算符重载函数 operator*，返回 double 类型，对应 vec_mul 函数
  DEFINE_MEMBER_OP(operator*, double, vec_mul)
  // 定义成员运算符重载函数 operator/，返回 double 类型，对应 vec_div 函数
  DEFINE_MEMBER_OP(operator/, double, vec_div)
  // 定义成员运算符重载函数 maximum，返回 double 类型，对应 vec_max_nan2 函数
  DEFINE_MEMBER_OP(maximum, double, vec_max_nan2)
  // 定义成员运算符重载函数 minimum，返回 double 类型，对应 vec_min_nan2 函数
  DEFINE_MEMBER_OP(minimum, double, vec_min_nan2)
  // 定义成员运算符重载函数 operator&，返回 double 类型，对应 vec_and 函数
  DEFINE_MEMBER_OP(operator&, double, vec_and)
  // 定义成员运算符重载函数 operator|，返回 double 类型，对应 vec_or 函数
  DEFINE_MEMBER_OP(operator|, double, vec_or)
  // 定义成员运算符重载函数 operator^，返回 double 类型，对应 vec_xor 函数
  DEFINE_MEMBER_OP(operator^, double, vec_xor)
  // 定义三元成员运算符重载函数 madd，返回 double 类型，对应 vec_madd 函数
  DEFINE_MEMBER_TERNARY_OP(madd, double, vec_madd)
};  // 结束当前代码块

template <>  // 显式特化模板，重载 maximum 函数，处理 Vectorized<double> 类型参数
Vectorized<double> inline maximum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return a.maximum(b);  // 调用 Vectorized 类的 maximum 方法，返回两个向量中每个元素的最大值
}

template <>  // 显式特化模板，重载 minimum 函数，处理 Vectorized<double> 类型参数
Vectorized<double> inline minimum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return a.minimum(b);  // 调用 Vectorized 类的 minimum 方法，返回两个向量中每个元素的最小值
}
} // namespace  // 结束 vec 命名空间
} // namespace at  // 结束 at 命名空间
```