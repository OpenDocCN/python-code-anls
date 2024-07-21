# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_int32_vsx.h`

```py
// 预处理指令，指示编译器只包含该头文件一次
#pragma once

// 引入ATen库的相关头文件
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>

// 命名空间at::vec中的命名空间CPU_CAPABILITY的内联声明
namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {

// Vectorized类的特化模板，针对int32_t类型
template <>
class Vectorized<int32_t> {
 private:
  // 使用联合体以便在同一内存位置定义多种视图
  union {
    struct {
      vint32 _vec0; // 第一个int32_t向量
      vint32 _vec1; // 第二个int32_t向量
    };
    struct {
      vbool32 _vecb0; // 第一个bool向量
      vbool32 _vecb1; // 第二个bool向量
    };
  } __attribute__((__may_alias__)); // 允许使用类型别名来引用联合体成员

 public:
  using value_type = int32_t; // 向量中元素的类型
  using vec_internal_type = vint32; // 内部向量类型
  using vec_internal_mask_type = vbool32; // 内部掩码类型
  using size_type = int; // 大小类型为int
  static constexpr size_type size() { // 返回向量的固定大小，这里为8
    return 8;
  }
  
  // 默认构造函数
  Vectorized() {}
  
  // 向量构造函数，使用单个vint32向量初始化
  C10_ALWAYS_INLINE Vectorized(vint32 v) : _vec0{v}, _vec1{v} {}
  
  // 向量构造函数，使用单个vbool32向量初始化
  C10_ALWAYS_INLINE Vectorized(vbool32 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  
  // 向量构造函数，使用两个vint32向量分别初始化_vec0和_vec1
  C10_ALWAYS_INLINE Vectorized(vint32 v1, vint32 v2) : _vec0{v1}, _vec1{v2} {}
  
  // 向量构造函数，使用两个vbool32向量分别初始化_vecb0和_vecb1
  C10_ALWAYS_INLINE Vectorized(vbool32 v1, vbool32 v2) : _vecb0{v1}, _vecb1{v2} {}
  
  // 向量构造函数，使用一个int32_t标量初始化_vec0和_vec1
  C10_ALWAYS_INLINE Vectorized(int32_t scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  
  // 向量构造函数，使用8个int32_t标量依次初始化_vec0和_vec1的元素
  C10_ALWAYS_INLINE Vectorized(
      int32_t scalar1,
      int32_t scalar2,
      int32_t scalar3,
      int32_t scalar4,
      int32_t scalar5,
      int32_t scalar6,
      int32_t scalar7,
      int32_t scalar8)
      : _vec0{vint32{scalar1, scalar2, scalar3, scalar4}},
        _vec1{vint32{scalar5, scalar6, scalar7, scalar8}} {}
  
  // 返回_vec0的引用
  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  
  // 返回_vec1的引用
  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }

  // blend函数模板，根据掩码mask进行向量混合，此处为mask为0时的特化版本
  template <uint64_t mask>
  static std::enable_if_t<mask == 0, Vectorized<int32_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return a;
  }

  // blend函数模板，mask中所有位都为1时的特化版本
  template <uint64_t mask>
  static std::enable_if_t<(mask & 255) == 255, Vectorized<int32_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return b;
  }

  // blend函数模板，mask为15时的特化版本
  template <uint64_t mask>
  static std::enable_if_t<mask == 15, Vectorized<int32_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return {b._vec0, a._vec1};
  }

  // blend函数模板，其他情况下的通用版本，根据mask生成对应的vbool32掩码并进行向量混合
  template <uint64_t mask>
  static std::enable_if_t<(mask > 0 && mask < 15), Vectorized<int32_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    // 根据mask生成用于第一向量的掩码
    constexpr uint32_t g0 = (mask & 1) * 0xffffffff;
    constexpr uint32_t g1 = ((mask & 2) >> 1) * 0xffffffff;
    constexpr uint32_t g2 = ((mask & 4) >> 2) * 0xffffffff;
    constexpr uint32_t g3 = ((mask & 8) >> 3) * 0xffffffff;
    const vbool32 mask_1st = (vbool32){g0, g1, g2, g3};
    // 根据给定的掩码（mask）选择并混合两个 Vectorized<int32_t> 对象的元素
    return {(vint32)vec_sel(a._vec0, b._vec0, (vbool32)mask_1st), a._vec1};
  }

  template <uint64_t mask>
  // 当 mask 大于 15 且低 8 位中不全为 1 且低 4 位全为 1 时，启用模板
  static std::enable_if_t<
      (mask > 15 && (mask & 255) != 255 && ((mask & 15) == 15)),
      Vectorized<int32_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    constexpr uint32_t mask2 = (mask & 255) >> 4;
    constexpr uint32_t g0_2 = (mask2 & 1) * 0xffffffff;
    constexpr uint32_t g1_2 = ((mask2 & 2) >> 1) * 0xffffffff;
    constexpr uint32_t g2_2 = ((mask2 & 4) >> 2) * 0xffffffff;
    constexpr uint32_t g3_2 = ((mask2 & 8) >> 3) * 0xffffffff;

    // 根据生成的掩码创建第二组掩码向量
    const vbool32 mask_2nd = (vbool32){g0_2, g1_2, g2_2, g3_2};
    // 生成的掩码
    return {b._vec0, (vint32)vec_sel(a._vec1, b._vec1, (vbool32)mask_2nd)};
  }

  template <uint64_t mask>
  // 当 mask 大于 15 且低 8 位中不全为 1 且低 4 位全为 0 时，启用模板
  static std::enable_if_t<
      (mask > 15 && ((mask & 255) != 255) && ((mask & 15) == 0)),
      Vectorized<int32_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    constexpr uint32_t mask2 = (mask & 255) >> 4;
    constexpr uint32_t g0_2 = (mask2 & 1) * 0xffffffff;
    constexpr uint32_t g1_2 = ((mask2 & 2) >> 1) * 0xffffffff;
    constexpr uint32_t g2_2 = ((mask2 & 4) >> 2) * 0xffffffff;
    constexpr uint32_t g3_2 = ((mask2 & 8) >> 3) * 0xffffffff;

    // 根据生成的掩码创建第二组掩码向量
    const vbool32 mask_2nd = (vbool32){g0_2, g1_2, g2_2, g3_2};
    // 生成的掩码
    return {a, (vint32)vec_sel(a._vec1, b._vec1, (vbool32)mask_2nd)};
  }

  template <uint64_t mask>
  // 当 mask 大于 15 且低 8 位中不全为 1 且低 4 位既不全为 0 也不全为 1 时，启用模板
  static std::enable_if_t<
      (mask > 15 && ((mask & 255) != 255) && ((mask & 15) != 0) &&
       ((mask & 15) != 15)),
      Vectorized<int32_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    constexpr uint32_t g0 = (mask & 1) * 0xffffffff;
    constexpr uint32_t g1 = ((mask & 2) >> 1) * 0xffffffff;
    constexpr uint32_t g2 = ((mask & 4) >> 2) * 0xffffffff;
    constexpr uint32_t g3 = ((mask & 8) >> 3) * 0xffffffff;
    constexpr uint32_t mask2 = (mask & 255) >> 4;
    constexpr uint32_t g0_2 = (mask2 & 1) * 0xffffffff;
    constexpr uint32_t g1_2 = ((mask2 & 2) >> 1) * 0xffffffff;
    constexpr uint32_t g2_2 = ((mask2 & 4) >> 2) * 0xffffffff;
    constexpr uint32_t g3_2 = ((mask2 & 8) >> 3) * 0xffffffff;

    // 根据生成的掩码创建第一组和第二组掩码向量
    const vbool32 mask_1st = (vbool32){g0, g1, g2, g3};
    const vbool32 mask_2nd = (vbool32){g0_2, g1_2, g2_2, g3_2};
    // 生成的掩码
    return {
        (vint32)vec_sel(a._vec0, b._vec0, (vbool32)mask_1st),
        (vint32)vec_sel(a._vec1, b._vec1, (vbool32)mask_2nd)};
  }

  static Vectorized<int32_t> C10_ALWAYS_INLINE blendv(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      const Vectorized<int32_t>& mask) {
    // 使用比较得到的掩码进行混合
    // 假设我们可以直接使用相同的掩码与 vec_sel
    // 警告：Intel 风格掩码可能无法正确工作

    // 使用比较得到的掩码进行混合
    // 假设我们可以直接使用相同的掩码与 vec_sel
    // 警告：Intel 风格掩码可能无法正确工作
  // 返回两个向量中每个元素根据掩码值进行选择后的结果向量
  return {
      vec_sel(a._vec0, b._vec0, mask._vecb0),
      vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }

  // 创建一个整数类型的向量，按照给定的步长从指定的基数开始填充
  template <typename step_t>
  static Vectorized<int32_t> arange(int32_t base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
  }

  // 根据给定的计数值，设置向量的内容
  static Vectorized<int32_t> set(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      size_t count = size()) {
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

    // 默认情况返回向量 b
    return b;
  }

  // 从内存中加载未对齐的数据到向量中
  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
    }

    // 如果请求加载的数量小于 size，则先复制数据到临时数组，再加载到向量中
    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }

  // 将向量中的数据存储到内存中
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      // 如果请求存储的数量小于 size，则先存储数据到临时数组，再复制到内存中
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, tmp_values);
      vec_vsx_st(_vec1, offset16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }

  // 禁止通过索引访问向量中的元素
  const int32_t& operator[](int idx) const = delete;
  int32_t& operator[](int idx) = delete;

  // 计算向量中每个元素的角度值
  Vectorized<int32_t> angle() const {
    return blendv(
      Vectorized<int32_t>(0), Vectorized<int32_t>(c10::pi<int32_t>), *this < Vectorized<int32_t>(0));
  }

  // 返回当前向量的实部，即向量本身
  Vectorized<int32_t> real() const {
    return *this;
  }

  // 返回当前向量的虚部，即元素都为 0 的向量
  Vectorized<int32_t> imag() const {
    return Vectorized<int32_t>{0};
  }

  // 返回当前向量的共轭，即向量本身
  Vectorized<int32_t> conj() const {
    return *this;
  }

  // 计算当前向量中每个元素的绝对值
  Vectorized<int32_t> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  // 计算当前向量中每个元素的负值
  Vectorized<int32_t> C10_ALWAYS_INLINE neg() const {
    // 返回一个包含 _vec0 和 _vec1 元素取负值的数组
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }

  // 定义一个按位取反操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_UNARY_OP(operator~, int32_t, vec_not)
  // 定义一个相等比较操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator==, int32_t, vec_cmpeq)
  // 定义一个不等比较操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator!=, int32_t, vec_cmpne)
  // 定义一个小于比较操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator<, int32_t, vec_cmplt)
  // 定义一个小于等于比较操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator<=, int32_t, vec_cmple)
  // 定义一个大于比较操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator>, int32_t, vec_cmpgt)
  // 定义一个大于等于比较操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator>=, int32_t, vec_cmpge)
  // 定义一个与操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP_AND_ONE(eq, int32_t, vec_cmpeq)
  // 定义一个或操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP_AND_ONE(ne, int32_t, vec_cmpne)
  // 定义一个按位与操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP_AND_ONE(lt, int32_t, vec_cmplt)
  // 定义一个按位或操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP_AND_ONE(le, int32_t, vec_cmple)
  // 定义一个按位异或操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP_AND_ONE(gt, int32_t, vec_cmpgt)
  // 定义一个按位非操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP_AND_ONE(ge, int32_t, vec_cmpge)
  // 定义一个加法操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator+, int32_t, vec_add)
  // 定义一个减法操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator-, int32_t, vec_sub)
  // 定义一个乘法操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator*, int32_t, vec_mul)
  // 定义一个除法操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, int32_t, /)
  // 定义一个取最大值操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(maximum, int32_t, vec_max)
  // 定义一个取最小值操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(minimum, int32_t, vec_min)
  // 定义一个按位与操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator&, int32_t, vec_and)
  // 定义一个按位或操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator|, int32_t, vec_or)
  // 定义一个按位异或操作的宏，对 int32_t 类型的成员函数进行封装
  DEFINE_MEMBER_OP(operator^, int32_t, vec_xor)
// 模板特化：重载位左移运算符，对两个整数向量进行逐元素左移操作
template <>
Vectorized<int32_t> inline operator<<(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    // 将第二个向量的每个元素转换为无符号整数类型
    vuint32 shift_vec0 = reinterpret_cast<vuint32>(b.vec0());
    vuint32 shift_vec1 = reinterpret_cast<vuint32>(b.vec1());
    // 返回一个新的整数向量，其中每个元素等于第一个向量对应元素左移第二个向量对应元素指定的位数
    return Vectorized<int32_t>{vec_sl(a.vec0(), shift_vec0), vec_sl(a.vec1(), shift_vec1)};
}

// 模板特化：重载位右移运算符，对两个整数向量进行逐元素右移操作
template <>
Vectorized<int32_t> inline operator>>(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    // 将第二个向量的每个元素转换为无符号整数类型
    vuint32 shift_vec0 = reinterpret_cast<vuint32>(b.vec0());
    vuint32 shift_vec1 = reinterpret_cast<vuint32>(b.vec1());
    // 返回一个新的整数向量，其中每个元素等于第一个向量对应元素右移第二个向量对应元素指定的位数
    return Vectorized<int32_t>{vec_sr(a.vec0(), shift_vec0), vec_sr(a.vec1(), shift_vec1)};
}

// 模板特化：求两个整数向量中每个元素的最大值
template <>
Vectorized<int32_t> inline maximum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return a.maximum(b);
}

// 模板特化：求两个整数向量中每个元素的最小值
template <>
Vectorized<int32_t> inline minimum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return a.minimum(b);
}

// 结束 vec 命名空间
} // namespace vec

// 结束 at 命名空间
} // namespace at
```