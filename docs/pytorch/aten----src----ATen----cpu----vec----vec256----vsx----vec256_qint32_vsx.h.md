# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_qint32_vsx.h`

```
// 预处理命令，指示编译器只包含一次本文件内容
#pragma once

// 包含ATen库的向量化指令和基础定义
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
#include <c10/util/qint32.h>
#include <array>

// 声明命名空间at::vec::CPU_CAPABILITY
namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

// Vectorized模板特化，用于c10::qint32类型
template <>
struct Vectorized<c10::qint32> {
 private:
  // 使用联合体定义内部数据结构，支持向量操作
  union {
    struct {
      vint32 _vec0;   // 第一个vint32向量
      vint32 _vec1;   // 第二个vint32向量
    };
    struct {
      vbool32 _vecb0; // 第一个vbool32掩码向量
      vbool32 _vecb1; // 第二个vbool32掩码向量
    };

  } __attribute__((__may_alias__)); // 指示编译器允许联合体成员别名访问

 public:
  // 默认构造函数
  Vectorized() {}

  // 定义向量大小的静态成员函数
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }

  // 定义返回浮点数向量个数的静态成员函数
  static constexpr size_t float_num_vecs() {
    return 1;
  }
  // 定义返回整数向量个数的静态成员函数
  static constexpr int int_num_vecs() {
    return 1;
  }

  // 定义浮点数向量的返回类型
  using float_vec_return_type = std::array<Vectorized<float>, 1>;
  // 定义整数向量的返回类型
  using int_vec_return_type = std::array<Vectorized<c10::qint32>, 1>;
  // 定义值类型为c10::qint32的底层类型
  using value_type = c10::qint32::underlying;
  // 定义向量内部操作类型为vint32
  using vec_internal_type = vint32;
  // 定义向量内部掩码类型为vbool32
  using vec_internal_mask_type = vbool32;

  // 构造函数，接受vint32类型参数，初始化内部向量
  C10_ALWAYS_INLINE Vectorized(vint32 v) : _vec0{v}, _vec1{v} {}
  // 构造函数，接受vbool32类型参数，初始化内部掩码向量
  C10_ALWAYS_INLINE Vectorized(vbool32 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  // 构造函数，接受两个vint32类型参数，初始化两个内部向量
  C10_ALWAYS_INLINE Vectorized(vint32 v1, vint32 v2) : _vec0{v1}, _vec1{v2} {}
  // 构造函数，接受两个vbool32类型参数，初始化两个内部掩码向量
  C10_ALWAYS_INLINE Vectorized(vbool32 v1, vbool32 v2) : _vecb0{v1}, _vecb1{v2} {}

  // 构造函数，接受c10::qint32类型的引用，通过vec_splats转换为vint32向量，并初始化两个内部向量
  Vectorized(const c10::qint32& val)
      : _vec0(vec_splats(val.val_)), _vec1(vec_splats(val.val_)) {}

  // 静态成员函数，从内存加载向量数据，并根据count参数选择部分加载或全部加载
  static Vectorized<c10::qint32> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
    }

    // 创建临时数组存储加载的数据，并转换为向量
    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }

  // 存储向量数据到内存，根据count参数选择部分存储或全部存储
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      // 如果 count 大于 0，则执行以下操作
      __at_align__ value_type tmp_values[size()];
      // 创建临时数组 tmp_values，大小为 size()
      vec_vsx_st(_vec0, offset0, tmp_values);
      // 将 _vec0 中的值存储到 tmp_values 中，偏移量为 offset0
      vec_vsx_st(_vec1, offset16, tmp_values);
      // 将 _vec1 中的值存储到 tmp_values 中，偏移量为 offset16
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
      // 使用 std::memcpy 将 tmp_values 中的数据复制到 ptr 所指向的地址，
      // 复制的字节数为 std::min(count, size()) * sizeof(value_type)
    }
  }

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    // 返回成员变量 _vec0 的引用
    return _vec0;
  }
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    // 返回成员变量 _vec1 的引用
    return _vec1;
  }

  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point,
      Vectorized<float> scale_zp_premul) const {
    // 将 _vec0 和 _vec1 中的值转换为 float 类型
    vfloat32 float_vals0 = vec_float(_vec0);
    vfloat32 float_vals1 = vec_float(_vec1);
    // 分别获取 scale、zero_point 和 scale_zp_premul 的 vec0 和 vec1
    vfloat32 scale_vec0 = scale.vec0();
    vfloat32 scale_vec1 = scale.vec1();
    vfloat32 scale_zp_premul0 = scale_zp_premul.vec0();
    vfloat32 scale_zp_premul1 = scale_zp_premul.vec1();
    // 返回 dequantize 结果，将 _vec0 和 _vec1 中的值进行加权和修正
    return {Vectorized<float>{
        vec_madd(scale_vec0, float_vals0, scale_zp_premul0),
        vec_madd(scale_vec1, float_vals1, scale_zp_premul1)}};
  }

  float_vec_return_type dequantize(
      Vectorized<float> scale,
      Vectorized<float> zero_point) const {
    // 将 _vec0 和 _vec1 中的值转换为 float 类型
    vfloat32 float_vals0 = vec_float(_vec0);
    vfloat32 float_vals1 = vec_float(_vec1);
    // 分别获取 scale 的 vec0 和 vec1
    vfloat32 scale_vec0 = scale.vec0();
    vfloat32 scale_vec1 = scale.vec1();
    // 分别获取 zero_point 的 vec0 和 vec1
    vfloat32 zero_point0 = zero_point.vec0();
    vfloat32 zero_point1 = zero_point.vec1();
    // 返回 dequantize 结果，将 _vec0 和 _vec1 中的值进行反量化处理
    return {Vectorized<float>{
        (float_vals0 - zero_point0) * scale_vec0,
        (float_vals1 - zero_point1) * scale_vec1}};
  }

  static Vectorized<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    // 创建返回值对象
    Vectorized<c10::qint32> retval;
    // 获取 value_type 的最小值和最大值，并转换为向量形式
    const vint32 vmin = vec_splats(std::numeric_limits<value_type>::min());
    const vint32 vmax = vec_splats(std::numeric_limits<value_type>::max());
    // 将 inverse_scale 扩展为向量
    vfloat32 inverse_scale_v = vec_splats(inverse_scale);
    // 将 zero_point 转换为向量
    vfloat32 vec_zero_point = vec_splats((float)(zero_point));
    // 获取 rhs 的第一个元素并转换为向量
    Vectorized<float> vf0 = rhs[0];

    // 将 vf0 的 vec0 和 vec1 分别存储到 vecf0 和 vecf1 中
    vfloat32 vecf0 = vf0.vec0();
    vfloat32 vecf1 = vf0.vec1();
    // 将 vecf0 和 vecf1 分别乘以 inverse_scale_v
    vecf0 = vec_mul(vecf0, inverse_scale_v);
    vecf1 = vec_mul(vecf1, inverse_scale_v);
    // 四舍五入并加上 zero_point
    vecf0 = vec_add(vec_rint(vecf0), vec_zero_point);
    vecf1 = vec_add(vec_rint(vecf1), vec_zero_point);
    // 将 vecf0 和 vecf1 转换为整型向量
    vint32 veci0  = vec_signed(vecf0);
    vint32 veci1  = vec_signed(vecf1);
    // 将 veci0 和 veci1 限制在 value_type 的取值范围内
    veci0 = vec_max(veci0, vmin);
    veci1 = vec_max(veci1, vmin);
    veci0 = vec_min(veci0, vmax);
    veci1 = vec_min(veci1, vmax);
    // 返回量化后的结果
    return {veci0, veci1};
  }

  Vectorized<c10::qint32> relu(Vectorized<c10::qint32> zero_point) const {
    // 返回 relu 操作后的结果向量
    return {vec_max(_vec0, zero_point._vec0), vec_max(_vec1, zero_point._vec1)};
  }

  Vectorized<c10::qint32> relu6(
      Vectorized<c10::qint32> zero_point,
      Vectorized<c10::qint32> q_six) const {
    // 对 _vec0 和 _vec1 分别执行 relu6 操作
    vint32 max0 = vec_max(_vec0, zero_point._vec0);
    vint32 max1 = vec_max(_vec1, zero_point._vec1);
    // 返回两个向量的最小值，其中第一个向量限制在max0内，第二个向量限制在max1内
    return {vec_min(max0, q_six._vec0), vec_min(max1, q_six._vec1)};
  }

  // 执行向量化减法操作，返回结果作为int_vec_return_type类型
  int_vec_return_type widening_subtract(Vectorized<c10::qint32> b) const {
    return {*this - b};
  }

  // 从整数向量类型inp重新量化为c10::qint32类型向量
  static Vectorized<c10::qint32> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    // 创建最小值和最大值的向量
    const vint32 vmin = vec_splats(std::numeric_limits<value_type>::min());
    const vint32 vmax = vec_splats(std::numeric_limits<value_type>::max());
    // 创建乘法因子和零点的向量
    vfloat32 vec_mult = vec_splats(multiplier);
    vint32 vec_zero_point = vec_splats(zero_point);
    // 从inp获取c10::qint32类型的向量vi
    Vectorized<c10::qint32> vi = inp[0];
    // 将vi的向量转换为浮点向量
    vfloat32 vecf0 = vec_float(vi.vec0());
    vfloat32 vecf1 = vec_float(vi.vec1());

    // 对浮点向量应用乘法因子
    vecf0 = vec_mul(vecf0, vec_mult);
    vecf1 = vec_mul(vecf1, vec_mult);

    // 对浮点向量进行四舍五入
    vecf0 = vec_rint(vecf0);
    vecf1 = vec_rint(vecf1);

    // 将浮点向量转换为带有零点偏移的整数向量
    vint32 veci0  = vec_add(vec_signed(vecf0), vec_zero_point);
    vint32 veci1  = vec_add(vec_signed(vecf1), vec_zero_point);

    // 将整数向量限制在[vmin, vmax]范围内
    veci0 = vec_max(veci0, vmin);
    veci1 = vec_max(veci1, vmin);
    veci0 = vec_min(veci0, vmax);
    veci1 = vec_min(veci1, vmax);

    // 返回包含更新后整数向量的c10::qint32类型向量
    return {veci0, veci1};
  }

  // 定义成员运算符，应用于c10::qint32类型的向量，例如相等比较、不等比较等
  DEFINE_MEMBER_OP(operator==, c10::qint32, vec_cmpeq)
  DEFINE_MEMBER_OP(operator!=, c10::qint32, vec_cmpne)
  DEFINE_MEMBER_OP(operator<, c10::qint32, vec_cmplt)
  DEFINE_MEMBER_OP(operator<=, c10::qint32, vec_cmple)
  DEFINE_MEMBER_OP(operator>, c10::qint32, vec_cmpgt)
  DEFINE_MEMBER_OP(operator>=, c10::qint32, vec_cmpge)
  DEFINE_MEMBER_OP(operator+, c10::qint32, vec_add)
  DEFINE_MEMBER_OP(operator-, c10::qint32, vec_sub)
  DEFINE_MEMBER_OP(operator*, c10::qint32, vec_mul)
  DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, c10::qint32, /)
  DEFINE_MEMBER_OP(maximum, c10::qint32, vec_max)
  DEFINE_MEMBER_OP(minimum, c10::qint32, vec_min)
  DEFINE_MEMBER_OP(operator&, c10::qint32, vec_and)
  DEFINE_MEMBER_OP(operator|, c10::qint32, vec_or)
  DEFINE_MEMBER_OP(operator^, c10::qint32, vec_xor)
};

template <>
// 定义模板特化，计算两个 Vectorized<c10::qint32> 对象的最大值
Vectorized<c10::qint32> inline maximum(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  // 调用 Vectorized 类的 maximum 方法，返回两个对象的最大值
  return a.maximum(b);
}

template <>
// 定义模板特化，计算两个 Vectorized<c10::qint32> 对象的最小值
Vectorized<c10::qint32> inline minimum(
    const Vectorized<c10::qint32>& a,
    const Vectorized<c10::qint32>& b) {
  // 调用 Vectorized 类的 minimum 方法，返回两个对象的最小值
  return a.minimum(b);
}
} // namespace
} // namespace vec
} // namespace at


这段代码是C++中的模板特化实现，针对 `Vectorized<c10::qint32>` 类型的对象进行了最大值和最小值的计算。
```