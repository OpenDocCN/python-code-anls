# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_complex_double_vsx.h`

```py
#pragma once
// 引入ATen库的CPU向量化指令集
#include <ATen/cpu/vec/intrinsics.h>
// 引入ATen库的向量基类
#include <ATen/cpu/vec/vec_base.h>
// 引入ATen库的VSX辅助函数
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
// 引入c10库的复数类型支持
#include <c10/util/complex.h>
// 引入c10库的范围迭代工具
#include <c10/util/irange.h>

// 命名空间：at下的vec命名空间
namespace at {
namespace vec {
// 内联命名空间：CPU_CAPABILITY命名空间
inline namespace CPU_CAPABILITY {
// 使用ComplexDbl作为double类型复数的别名
using ComplexDbl = c10::complex<double>;

// ComplexDbl类型的模板特化类Vectorized
template <>
class Vectorized<ComplexDbl> {
  // 使用联合体进行内部存储，支持不同的视角
  union {
    // 第一个视角：包含两个vfloat64向量
    struct {
      vfloat64 _vec0;
      vfloat64 _vec1;
    };
    // 第二个视角：包含两个vbool64向量
    struct {
      vbool64 _vecb0;
      vbool64 _vecb1;
    };

  } __attribute__((__may_alias__)); // 使用__may_alias__属性确保正确对齐

 public:
  // 类型定义
  using value_type = ComplexDbl;
  using vec_internal_type = vfloat64;
  using vec_internal_mask_type = vbool64;
  using size_type = int;
  
  // 返回向量大小为2
  static constexpr size_type size() {
    return 2;
  }

  // 默认构造函数
  Vectorized() {}

  // 构造函数：根据vfloat64向量构造
  C10_ALWAYS_INLINE Vectorized(vfloat64 v) : _vec0{v}, _vec1{v} {}

  // 构造函数：根据vbool64向量构造
  C10_ALWAYS_INLINE Vectorized(vbool64 vmask) : _vecb0{vmask}, _vecb1{vmask} {}

  // 构造函数：根据两个vfloat64向量构造
  C10_ALWAYS_INLINE Vectorized(vfloat64 v1, vfloat64 v2) : _vec0{v1}, _vec1{v2} {}

  // 构造函数：根据两个vbool64向量构造
  C10_ALWAYS_INLINE Vectorized(vbool64 v1, vbool64 v2) : _vecb0{v1}, _vecb1{v2} {}

  // 构造函数：根据ComplexDbl类型构造，将实部和虚部存储为vfloat64向量
  Vectorized(ComplexDbl val) {
    double real_value = val.real();
    double imag_value = val.imag();
    _vec0 = vfloat64{real_value, imag_value};
    _vec1 = vfloat64{real_value, imag_value};
  }

  // 构造函数：根据两个ComplexDbl类型构造，分别存储为_vec0和_vec1的实部和虚部
  Vectorized(ComplexDbl val1, ComplexDbl val2) {
    _vec0 = vfloat64{val1.real(), val1.imag()};
    _vec1 = vfloat64{val2.real(), val2.imag()};
  }

  // 返回_vec0的引用
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }

  // 返回_vec1的引用
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  // 模板函数：根据掩码mask选择a或b向量，返回Vectorized<ComplexDbl>类型
  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 0, Vectorized<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    return a;
  }

  // 模板函数：根据掩码mask选择a或b向量，返回Vectorized<ComplexDbl>类型
  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 1, Vectorized<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    return b;
  }

  // 模板函数：根据掩码mask选择a或b向量，返回Vectorized<ComplexDbl>类型
  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 2, Vectorized<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    return {b._vec0, a._vec1};
  }

  // 模板函数：根据掩码mask选择a或b向量，返回Vectorized<ComplexDbl>类型
  template <int64_t mask>
  static std::enable_if_t<blendChoiceComplexDbl(mask) == 3, Vectorized<ComplexDbl>>
      C10_ALWAYS_INLINE
      blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    return {a._vec0, b._vec1};
  }

  // 模板函数：元素级别根据掩码mask选择a或b向量的对应元素构成新的Vectorized<ComplexDbl>向量
  template <int64_t mask>
  static Vectorized<ComplexDbl> C10_ALWAYS_INLINE
  el_blend(const Vectorized<ComplexDbl>& a, const Vectorized<ComplexDbl>& b) {
    // 获取掩码mask的第一部分和第二部分对应的vbool64掩码
    const vbool64 mask_1st = VsxDblMask1(mask);
    const vbool64 mask_2nd = VsxDblMask2(mask);
    return {
        (vfloat64)vec_sel(a._vec0, b._vec0, mask_1st),
        (vfloat64)vec_sel(a._vec1, b._vec1, mask_2nd)};
  }

  static Vectorized<ComplexDbl> blendv(
      const Vectorized<ComplexDbl>& a,
      const Vectorized<ComplexDbl>& b,
      const Vectorized<ComplexDbl>& mask) {
    // 将复杂类型的掩码转换为简单类型的掩码：xy -> xxyy
    auto mask_complex =
        Vectorized<ComplexDbl>(vec_splat(mask._vec0, 0), vec_splat(mask._vec1, 0));
    return {
        vec_sel(a._vec0, b._vec0, mask_complex._vecb0),  // 使用掩码混合向量的第一个部分
        vec_sel(a._vec1, b._vec1, mask_complex._vecb1)}; // 使用掩码混合向量的第二个部分
  }

  static Vectorized<ComplexDbl> C10_ALWAYS_INLINE elwise_blendv(
      const Vectorized<ComplexDbl>& a,
      const Vectorized<ComplexDbl>& b,
      const Vectorized<ComplexDbl>& mask) {
    // 使用给定的掩码对向量的每个元素进行混合
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),  // 使用掩码混合向量的第一个部分
        vec_sel(a._vec1, b._vec1, mask._vecb1)}; // 使用掩码混合向量的第二个部分
  }

  template <typename step_t>
  static Vectorized<ComplexDbl> arange(
      ComplexDbl base = 0.,
      step_t step = static_cast<step_t>(1)) {
    // 返回一个包含起始值和步长的向量
    return Vectorized<ComplexDbl>(base, base + step);
  }

  static Vectorized<ComplexDbl> set(
      const Vectorized<ComplexDbl>& a,
      const Vectorized<ComplexDbl>& b,
      int64_t count = size()) {
    switch (count) {
      case 0:
        return a;  // 返回向量 a，因为 count 为 0
      case 1:
        return blend<1>(a, b);  // 返回使用 blend 函数混合后的向量
    }
    return b;  // 返回向量 b，默认情况
  }

  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      // 从内存中加载向量数据，假设数据在内存中对齐
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const double*>(ptr)),  // 加载向量的第一个部分
          vec_vsx_ld(offset16, reinterpret_cast<const double*>(ptr))};  // 加载向量的第二个部分
    }

    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    // 从临时数组加载向量数据
    return {
        vec_vsx_ld(offset0, reinterpret_cast<const double*>(tmp_values)),  // 加载向量的第一个部分
        vec_vsx_ld(offset16, reinterpret_cast<const double*>(tmp_values))};  // 加载向量的第二个部分
  }

  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 将向量数据存储到内存中，假设数据在内存中对齐
      vec_vsx_st(_vec0, offset0, reinterpret_cast<double*>(ptr));  // 存储向量的第一个部分
      vec_vsx_st(_vec1, offset16, reinterpret_cast<double*>(ptr));  // 存储向量的第二个部分
    } else if (count > 0) {
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, reinterpret_cast<double*>(tmp_values));  // 存储向量的第一个部分到临时数组
      vec_vsx_st(_vec1, offset16, reinterpret_cast<double*>(tmp_values));  // 存储向量的第二个部分到临时数组
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));  // 将临时数组数据复制到目标内存
    }
  }

  const ComplexDbl& operator[](int idx) const = delete;
  ComplexDbl& operator[](int idx) = delete;

  Vectorized<ComplexDbl> map(ComplexDbl (*const f)(ComplexDbl)) const {
    __at_align__ ComplexDbl tmp[size()];
    store(tmp);  // 将向量数据存储到临时数组
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);  // 对临时数组中的每个元素应用函数 f
    }
    return loadu(tmp);  // 从处理后的临时数组加载数据到向量
  }

  Vectorized<ComplexDbl> map(ComplexDbl (*const f)(const ComplexDbl&)) const {
    __at_align__ ComplexDbl tmp[size()];
    store(tmp);  // 将向量数据存储到临时数组
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);  // 对临时数组中的每个元素应用函数 f
    }
    return loadu(tmp);  // 从处理后的临时数组加载数据到向量
  }
  Vectorized<ComplexDbl> abs_2_() const {
    // 计算每个元素的平方
    auto a = (*this).elwise_mult(*this);
    // 对元素进行交换
    auto permuted = a.el_swapped();
    // 对应元素相加
    a = a + permuted;
    return a;
  }

  Vectorized<ComplexDbl> abs_() const {
    // 分别提取实部和虚部
    auto vi = el_mergeo();
    auto vr = el_mergee();
    // 计算绝对值并返回
    return {Sleef_hypotd2_u05vsx(vr._vec0, vi._vec0), Sleef_hypotd2_u05vsx(vr._vec1, vi._vec1)};
  }

  Vectorized<ComplexDbl> abs() const {
    // 返回绝对值并应用实部掩码
    return abs_() & vd_real_mask;
  }

  Vectorized<ComplexDbl> angle_() const {
    // 计算复数的角度 angle = atan2(b/a)
    Vectorized<ComplexDbl> ret;
    ret._vec0[0] = std::atan2(_vec0[1], _vec0[0]);
    ret._vec1[0] = std::atan2(_vec1[1], _vec1[0]);
    return ret;
  }

  Vectorized<ComplexDbl> angle() const {
    // 返回角度并应用实部掩码
    return angle_() & vd_real_mask;
  }

  Vectorized<ComplexDbl> real_() const {
    // 返回实部并应用实部掩码
    return *this & vd_real_mask;
  }

  Vectorized<ComplexDbl> real() const {
    // 返回实部并应用实部掩码
    return *this & vd_real_mask;
  }

  Vectorized<ComplexDbl> imag_() const {
    // 返回虚部并应用虚部掩码
    return *this & vd_imag_mask;
  }

  Vectorized<ComplexDbl> imag() const {
    // 返回虚部并交换元素顺序
    return imag_().el_swapped();
  }

  Vectorized<ComplexDbl> conj_() const {
    // 返回共轭并应用虚部符号掩码
    return *this ^ vd_isign_mask;
  }

  Vectorized<ComplexDbl> conj() const {
    // 返回共轭并应用虚部符号掩码
    return *this ^ vd_isign_mask;
  }

  Vectorized<ComplexDbl> log() const {
    // 使用对数操作来改善复数的性能
    return map(std::log);
  }

  Vectorized<ComplexDbl> log2() const {
    // 计算以2为底的对数
    auto ret = log();
    return ret.elwise_mult(vd_log2e_inv);
  }

  Vectorized<ComplexDbl> log10() const {
    // 计算以10为底的对数
    auto ret = log();
    ```

    return ret.elwise_mult(vd_log10e_inv);
  }
  Vectorized<ComplexDbl> log1p() const {
    // 返回 log(1 + x)，其中 x 为当前向量的每个元素
    return map(std::log1p);
  }

  Vectorized<ComplexDbl> asin() const {
    // 返回 arcsin(x)，其中 x 为当前向量的每个元素
    // 使用公式 -i*ln((ai - b) + sqrt(1 - (a**2 - b**2) - 2*abi)) 计算
    auto conj = conj_();
    auto b_a = conj.el_swapped();
    auto ab = conj.elwise_mult(b_a);
    auto im = ab + ab;
    auto val_2 = (*this).elwise_mult(*this);
    auto val_2_swapped = val_2.el_swapped();
    auto re = horizontal_sub(val_2, val_2_swapped);
    re = Vectorized<ComplexDbl>(vd_one) - re;
    auto root = el_blend<0x0A>(re, im).sqrt();
    auto ln = (b_a + root).log();
    return ln.el_swapped().conj();
  }

  Vectorized<ComplexDbl> acos() const {
    // 返回 arccos(x)，其中 x 为当前向量的每个元素
    // 使用公式 pi/2 - asin(x) 计算
    return Vectorized(vd_pi_2) - asin();
  }

  Vectorized<ComplexDbl> atan() const {
    // 返回 arctan(x)，其中 x 为当前向量的每个元素
    // 使用公式 i/2 * ln((i + z)/(i - z)) 计算
    auto ione = Vectorized(vd_imag_one);
    auto sum = ione + *this;
    auto sub = ione - *this;
    auto ln = (sum / sub).log(); // ln((i + z)/(i - z))
    return ln * vd_imag_half; // i/2*ln()
  }
    // 返回一个包含水平求和结果的向量，其中每个元素由两个向量对应位置元素的和构成
    return el_mergee(first, second) + el_mergeo(first, second);
  }

  static Vectorized<ComplexDbl> horizontal_sub(
      Vectorized<ComplexDbl>& first,
      Vectorized<ComplexDbl>& second) {
    // 以总共6条指令模拟不同的方式进行操作
    // 将第二个向量进行排列，以便进行求和得到水平求和
    auto first_perm = first.el_swapped(); // 对 first 进行元素交换（2perm）
    auto second_perm = second.el_swapped(); // 对 second 进行元素交换（2perm）
    // 计算差值
    auto first_ret = first - first_perm; // 对 first 和其排列结果进行元素减法操作（2sub）
    auto second_ret = second - second_perm; // 对 second 和其排列结果进行元素减法操作（2sub）
    // 选择偶数索引位置的元素进行合并
    return el_mergee(first_ret, second_ret); // 对 first_ret 和 second_ret 执行 el_mergee 操作（2 mergee's）
  }

  Vectorized<ComplexDbl> inline operator*(const Vectorized<ComplexDbl>& b) const {
    // 计算两个复数向量的乘积
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
#if 1
    // 如果条件为真，则执行以下代码块
    // 将当前对象的虚部与实部分别合并为一个新向量 vi 和 vr
    auto vi = b.el_mergeo();
    auto vr = b.el_mergee();
    // 将 vi 中的元素按位异或操作，使用 vd_rsign_mask 进行掩码操作
    vi = vi ^ vd_rsign_mask;
    // 计算 vi 和 vr 的逐元素乘积
    auto ret = elwise_mult(vr);
    // 对 vi 和 ret 进行逐元素乘法，结果存储在 vx_swapped 中
    auto vx_swapped = el_swapped();
    ret = vx_swapped.el_madd(vi, ret);
#else
    // 如果条件为假，则执行以下代码块
    // 计算当前对象与 b 的逐元素乘积
    auto ac_bd = elwise_mult(b);
    // 将 b 的虚部和实部交换，存储在 d_c 中
    auto d_c = b.el_swapped();
    // 对 d_c 的虚部进行按位异或操作，使用 vd_isign_mask 进行掩码操作
    d_c = d_c ^ vd_isign_mask;
    // 计算当前对象与 d_c 的逐元素乘积
    auto ad_bc = elwise_mult(d_c);
    // 计算 ac_bd 和 ad_bc 的逐元素水平减法
    auto ret = horizontal_sub(ac_bd, ad_bc);
#endif
    // 返回计算结果 ret
    return ret;
  }

  // 重载运算符，实现复数除法
  Vectorized<ComplexDbl> inline operator/(const Vectorized<ComplexDbl>& b) const {
    // 实部 re + 虚部 im * i = (a + bi) / (c + di)
    // 计算分母的模的平方 abs_2()
    auto fabs_cd =  Vectorized{
      vec_andc(b._vec0, vd_sign_mask),
      vec_andc(b._vec1, vd_sign_mask)};       // |c|            |d|
    // 交换 fabs_cd 中的元素，得到 fabs_dc
    auto fabs_dc =  fabs_cd.el_swapped();     // |d|            |c|
    // 计算 fabs_cd 和 fabs_dc 中对应元素的最大值，得到 scale
    auto scale = fabs_cd.elwise_max(fabs_dc); // sc = max(|c|, |d|)
    // 将当前对象按照 scale 进行逐元素除法，得到 a2 和 b2
    auto a2 = elwise_div(scale);              // a/sc           b/sc
    auto b2 = b.elwise_div(scale);            // c/sc           d/sc
    // 计算 a2 和 b2 的逐元素乘积 acbd2 和 b2 的虚部按位异或操作后的结果 dc2
    auto acbd2 = a2.elwise_mult(b2);          // ac/sc^2        bd/sc^2
    auto dc2 = b2.el_swapped();               // d/sc           c/sc
    dc2 = dc2 ^ vd_rsign_mask;                // -d/sc          c/sc
    // 计算 a2 和 dc2 的逐元素乘积 adbc2，以及 acbd2 和 adbc2 的逐元素水平加法结果 ret
    auto adbc2 = a2.elwise_mult(dc2);         // -ad/sc^2       bc/sc^2
    auto ret = horizontal_add(acbd2, adbc2);  // (ac+bd)/sc^2   (bc-ad)/sc^2
    // 计算分母的模的平方 denom2，并将 ret 按照 denom2 进行逐元素除法
    auto denom2 = b2.abs_2_();                // (c^2+d^2)/sc^2 (c^2+d^2)/sc^2
    ret = ret.elwise_div(denom2);
    // 返回计算结果 ret
    return ret;
  }

  // 对复数向量进行指数运算
  Vectorized<ComplexDbl> exp() const {
    return map(std::exp);
  }

  // 对复数向量进行 2 的指数运算
  Vectorized<ComplexDbl> exp2() const {
    return map(exp2_impl);
  }

  // 对复数向量进行 expm1 运算
  Vectorized<ComplexDbl> expm1() const {
    return map(std::expm1);
  }

  // 对复数向量进行幂运算
  Vectorized<ComplexDbl> pow(const Vectorized<ComplexDbl>& exp) const {
    // 定义临时数组 x_tmp 和 y_tmp，用于存储当前对象和 exp 的值
    __at_align__ ComplexDbl x_tmp[size()];
    __at_align__ ComplexDbl y_tmp[size()];
    // 将当前对象和 exp 的值存储到对应的临时数组中
    store(x_tmp);
    exp.store(y_tmp);
    // 遍历数组中的元素，对 x_tmp 和 y_tmp 中对应位置的元素进行幂运算
    for (const auto i : c10::irange(size())) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    // 将计算结果加载到新的 Vectorized<ComplexDbl> 对象中并返回
    return loadu(x_tmp);
  }

  // 对复数向量进行符号函数运算
  Vectorized<ComplexDbl> sgn() const {
    return map(at::native::sgn_impl);
  }

  // 不支持复数向量的比较运算符，抛出错误信息
  Vectorized<ComplexDbl> operator<(const Vectorized<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<ComplexDbl> operator<=(const Vectorized<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<ComplexDbl> operator>(const Vectorized<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<ComplexDbl> operator>=(const Vectorized<ComplexDbl>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  // 对复数向量进行相等比较运算
  Vectorized<ComplexDbl> eq(const Vectorized<ComplexDbl>& other) const {
    // 比较当前对象和 other 中实部和虚部是否相等
    auto eq = (*this == other);  // compares real and imag individually
    // 返回比较结果
  // 如果实部和虚部都相等，则复数相等
  return (eq.real() & eq.imag()) & vd_one;
}

// 返回一个向量，指示当前向量中的每个复数是否与另一个向量中对应位置的复数不相等
Vectorized<ComplexDbl> ne(const Vectorized<ComplexDbl>& other) const {
  // 对每个实部和虚部分别进行比较
  auto ne = (*this != other);
  // 如果实部或虚部中有任何一个不相等，则复数不相等
  return (ne.real() | ne.imag()) & vd_one;
}

// 定义成员运算符重载，实现复数相等比较
DEFINE_MEMBER_OP(operator==, ComplexDbl, vec_cmpeq)
// 定义成员运算符重载，实现复数不相等比较
DEFINE_MEMBER_OP(operator!=, ComplexDbl, vec_cmpne)

// 定义成员运算符重载，实现复数加法
DEFINE_MEMBER_OP(operator+, ComplexDbl, vec_add)
// 定义成员运算符重载，实现复数减法
DEFINE_MEMBER_OP(operator-, ComplexDbl, vec_sub)
// 定义成员运算符重载，实现按位与操作
DEFINE_MEMBER_OP(operator&, ComplexDbl, vec_and)
// 定义成员运算符重载，实现按位或操作
DEFINE_MEMBER_OP(operator|, ComplexDbl, vec_or)
// 定义成员运算符重载，实现按位异或操作
DEFINE_MEMBER_OP(operator^, ComplexDbl, vec_xor)

// 定义成员运算符重载，实现按元素乘法
DEFINE_MEMBER_OP(elwise_mult, ComplexDbl, vec_mul)
// 定义成员运算符重载，实现按元素除法
DEFINE_MEMBER_OP(elwise_div, ComplexDbl, vec_div)
// 定义成员运算符重载，实现按元素大于比较
DEFINE_MEMBER_OP(elwise_gt, ComplexDbl, vec_cmpgt)
// 定义成员运算符重载，实现按元素大于等于比较
DEFINE_MEMBER_OP(elwise_ge, ComplexDbl, vec_cmpge)
// 定义成员运算符重载，实现按元素小于比较
DEFINE_MEMBER_OP(elwise_lt, ComplexDbl, vec_cmplt)
// 定义成员运算符重载，实现按元素小于等于比较
DEFINE_MEMBER_OP(elwise_le, ComplexDbl, vec_cmple)
// 定义成员运算符重载，实现按元素取最大值
DEFINE_MEMBER_OP(elwise_max, ComplexDbl, vec_max)
};

template <>
// 返回两个复数向量中每个元素的绝对值平方，作为比较的基础
Vectorized<ComplexDbl> inline maximum(
    const Vectorized<ComplexDbl>& a,
    const Vectorized<ComplexDbl>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // 创建一个比较掩码，用于标识 abs_a 中每个元素是否小于 abs_b 中对应的元素
  auto mask = abs_a.elwise_lt(abs_b);
  // 使用掩码选择每个位置上较大的复数元素，构成新的复数向量
  auto max = Vectorized<ComplexDbl>::elwise_blendv(a, b, mask);

  return max;
  // 利用所有位都是 1 的情况表示 NaN
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(max, isnan);
}

template <>
// 返回两个复数向量中每个元素的绝对值平方，作为比较的基础
Vectorized<ComplexDbl> inline minimum(
    const Vectorized<ComplexDbl>& a,
    const Vectorized<ComplexDbl>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  // 创建一个比较掩码，用于标识 abs_a 中每个元素是否大于 abs_b 中对应的元素
  auto mask = abs_a.elwise_gt(abs_b);
  // 使用掩码选择每个位置上较小的复数元素，构成新的复数向量
  auto min = Vectorized<ComplexDbl>::elwise_blendv(a, b, mask);
  return min;
  // 利用所有位都是 1 的情况表示 NaN
  // auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  // return _mm256_or_ps(min, isnan);
}


} // namespace
} // namespace vec
} // namespace at
```