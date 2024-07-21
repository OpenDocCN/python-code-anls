# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vec256_int16_vsx.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/cpu/vec/intrinsics.h>
// 包含 ATen 库中的 CPU 向量化指令头文件

#include <ATen/cpu/vec/vec_base.h>
// 包含 ATen 库中的向量基础操作头文件

#include <ATen/cpu/vec/vec256/vsx/vsx_helpers.h>
// 包含 ATen 库中 VSX（Vector Scalar Extensions）辅助函数头文件

namespace at {
namespace vec {
// 命名空间 at::vec，用于封装向量化相关功能

// See Note [CPU_CAPABILITY namespace]
// CPU_CAPABILITY 命名空间的注意事项

inline namespace CPU_CAPABILITY {
// 内联定义 CPU_CAPABILITY 命名空间

template <>
class Vectorized<int16_t> {
// int16_t 类型的向量化类模板特化声明

 private:
  union {
    struct {
      vint16 _vec0;
      vint16 _vec1;
    };
    struct {
      vbool16 _vecb0;
      vbool16 _vecb1;
    };

  } __attribute__((__may_alias__));
  // 匿名联合体，允许对不同类型的数据进行别名访问，并且可能进行指针优化

 public:
  using value_type = int16_t;
  // 类型别名，指定向量化类的值类型为 int16_t
  using vec_internal_type = vint16;
  // 类型别名，指定向量内部数据类型为 vint16（int16_t 向量）
  using vec_internal_mask_type = vbool16;
  // 类型别名，指定向量内部掩码类型为 vbool16（布尔向量）
  using size_type = int;
  // 类型别名，指定大小类型为 int

  static constexpr size_type size() {
    return 16;
  }
  // 静态成员函数，返回向量大小为 16

  Vectorized() {}
  // 默认构造函数，创建一个空的 Vectorized 对象

  C10_ALWAYS_INLINE Vectorized(vint16 v) : _vec0{v}, _vec1{v} {}
  // 构造函数，使用 vint16 类型的参数 v 初始化 _vec0 和 _vec1

  C10_ALWAYS_INLINE Vectorized(vbool16 vmask) : _vecb0{vmask}, _vecb1{vmask} {}
  // 构造函数，使用 vbool16 类型的参数 vmask 初始化 _vecb0 和 _vecb1

  C10_ALWAYS_INLINE Vectorized(vint16 v1, vint16 v2) : _vec0{v1}, _vec1{v2} {}
  // 构造函数，使用 vint16 类型的参数 v1 和 v2 初始化 _vec0 和 _vec1

  C10_ALWAYS_INLINE Vectorized(vbool16 v1, vbool16 v2) : _vecb0{v1}, _vecb1{v2} {}
  // 构造函数，使用 vbool16 类型的参数 v1 和 v2 初始化 _vecb0 和 _vecb1

  C10_ALWAYS_INLINE Vectorized(int16_t scalar)
      : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  // 构造函数，使用 int16_t 类型的标量参数 scalar 分别初始化 _vec0 和 _vec1

  C10_ALWAYS_INLINE Vectorized(
      int16_t scalar1,
      int16_t scalar2,
      int16_t scalar3,
      int16_t scalar4,
      int16_t scalar5,
      int16_t scalar6,
      int16_t scalar7,
      int16_t scalar8,
      int16_t scalar9,
      int16_t scalar10,
      int16_t scalar11,
      int16_t scalar12,
      int16_t scalar13,
      int16_t scalar14,
      int16_t scalar15,
      int16_t scalar16)
      : _vec0{vint16{
            scalar1,
            scalar2,
            scalar3,
            scalar4,
            scalar5,
            scalar6,
            scalar7,
            scalar8}},
        _vec1{vint16{
            scalar9,
            scalar10,
            scalar11,
            scalar12,
            scalar13,
            scalar14,
            scalar15,
            scalar16}} {}
  // 构造函数，使用 16 个 int16_t 类型的标量参数分别初始化 _vec0 和 _vec1

  C10_ALWAYS_INLINE const vec_internal_type& vec0() const {
    return _vec0;
  }
  // 返回 _vec0 的引用，作为 vint16 类型的向量

  C10_ALWAYS_INLINE const vec_internal_type& vec1() const {
    return _vec1;
  }
  // 返回 _vec1 的引用，作为 vint16 类型的向量

  template <uint64_t mask>
  static std::enable_if_t<mask == 0, Vectorized<int16_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    return a;
  }
  // 模板函数，根据掩码 mask，如果 mask 等于 0，则返回向量 a，否则返回向量 b

  template <uint64_t mask>
  static std::enable_if_t<(mask & 65535) == 65535, Vectorized<int16_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    return b;
  }
  // 模板函数，如果 mask 的低 16 位全为 1，则返回向量 b，否则返回向量 a

  template <uint64_t mask>
  static std::enable_if_t<mask == 255, Vectorized<int16_t>> C10_ALWAYS_INLINE
  blend(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    return {b._vec0, a._vec1};
  }
  // 模板函数，如果 mask 的值为 255，则返回使用 b 的 _vec0 和 a 的 _vec1 组成的新向量

  template <uint64_t mask>
  static std::enable_if_t<(mask > 0 && mask < 255), Vectorized<int16_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    constexpr int16_t g0 = (mask & 1) * 0xffff;
    constexpr int16_t g1 = ((mask & 2) >> 1) * 0xffff;
  }
  // 模板函数，如果 mask 的值大于 0 且小于 255，则根据 mask 的不同位组合，计算出 g0 和 g1 的值
    // 使用位掩码(mask)来生成g2到g7，每个掩码位都右移并乘以0xffff
    constexpr int16_t g2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7 = ((mask & 128) >> 7) * 0xffff;
    // 创建一个16位整数向量mask_1st，其中包含生成的g0到g7
    const vint16 mask_1st = vint16{g0, g1, g2, g3, g4, g5, g6, g7};

    // 返回一个新的Vectorized<int16_t>对象，使用vec_sel函数根据mask_1st选择a或b的_vec0，a的_vec1
    return {(vint16)vec_sel(a._vec0, b._vec0, (vbool16)mask_1st), a._vec1};
  }

  // 当mask大于255且低16位不全为1且低8位全为1时，启用此blend函数
  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && (mask & 65535) != 65535 && ((mask & 255) == 255)),
      Vectorized<int16_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    // 使用位掩码(mask)来生成g0到g7，每个掩码位都右移并乘以0xffff
    constexpr int16_t g0_2 = (mask & 1) * 0xffff;
    constexpr int16_t g1_2 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2_2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3_2 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4_2 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5_2 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6_2 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7_2 = ((mask & 128) >> 7) * 0xffff;

    // 创建一个16位整数向量mask_2nd，其中包含生成的g0_2到g7_2
    const vint16 mask_2nd =
        vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    // 返回一个新的Vectorized<int16_t>对象，选择b的_vec0和a的_vec1，使用mask_2nd
    return {b._vec0, (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  // 当mask大于255且低16位不全为1且低8位全为0时，启用此blend函数
  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && ((mask & 65535) != 65535) && ((mask & 255) == 0)),
      Vectorized<int16_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    // 将低16位移到高16位并取整
    constexpr int16_t mask2 = (mask & 65535) >> 16;
    // 使用位掩码(mask)来生成g0到g7，每个掩码位都右移并乘以0xffff
    constexpr int16_t g0_2 = (mask & 1) * 0xffff;
    constexpr int16_t g1_2 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2_2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3_2 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4_2 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5_2 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6_2 = ((mask & 64) >> 6) * 0xffff;
    constexpr int16_t g7_2 = ((mask & 128) >> 7) * 0xffff;

    // 创建一个16位整数向量mask_2nd，其中包含生成的g0_2到g7_2
    const vint16 mask_2nd =
        vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    // 返回一个新的Vectorized<int16_t>对象，选择a，使用vec_sel函数根据mask_2nd选择a的_vec1或b的_vec1
    return {a, (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  // 当mask大于255且低16位不全为1且低8位不全为0和全为1时，启用此blend函数
  template <uint64_t mask>
  static std::enable_if_t<
      (mask > 255 && ((mask & 65535) != 65535) && ((mask & 255) != 0) &&
       ((mask & 255) != 255)),
      Vectorized<int16_t>>
      C10_ALWAYS_INLINE blend(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    // 使用位掩码(mask)来生成g0到g6，每个掩码位都右移并乘以0xffff
    constexpr int16_t g0 = (mask & 1) * 0xffff;
    constexpr int16_t g1 = ((mask & 2) >> 1) * 0xffff;
    constexpr int16_t g2 = ((mask & 4) >> 2) * 0xffff;
    constexpr int16_t g3 = ((mask & 8) >> 3) * 0xffff;
    constexpr int16_t g4 = ((mask & 16) >> 4) * 0xffff;
    constexpr int16_t g5 = ((mask & 32) >> 5) * 0xffff;
    constexpr int16_t g6 = ((mask & 64) >> 6) * 0xffff;
    // 创建一个16位整数向量mask_2nd，其中包含生成的g0到g6
    const vint16 mask_2nd =
        vint16{g0, g1, g2, g3, g4, g5, g6, 0};  // g7为0
    // 计算 g7 的值，根据 mask 中第 8 位的值进行设置，并将结果左移 7 位，然后转换为 int16_t 类型
    constexpr int16_t g7 = ((mask & 128) >> 7) * 0xffff;
    // 计算 mask 的低 16 位，然后将结果右移 16 位，得到 mask2，并转换为 int16_t 类型
    constexpr int16_t mask2 = (mask & 65535) >> 16;
    // 计算 g0_2，根据 mask 中第 1 位的值进行设置，并将结果转换为 int16_t 类型
    constexpr int16_t g0_2 = (mask & 1) * 0xffff;
    // 计算 g1_2，根据 mask 中第 2 位的值进行设置，并将结果右移 1 位，然后转换为 int16_t 类型
    constexpr int16_t g1_2 = ((mask & 2) >> 1) * 0xffff;
    // 计算 g2_2，根据 mask 中第 3 位的值进行设置，并将结果右移 2 位，然后转换为 int16_t 类型
    constexpr int16_t g2_2 = ((mask & 4) >> 2) * 0xffff;
    // 计算 g3_2，根据 mask 中第 4 位的值进行设置，并将结果右移 3 位，然后转换为 int16_t 类型
    constexpr int16_t g3_2 = ((mask & 8) >> 3) * 0xffff;
    // 计算 g4_2，根据 mask 中第 5 位的值进行设置，并将结果右移 4 位，然后转换为 int16_t 类型
    constexpr int16_t g4_2 = ((mask & 16) >> 4) * 0xffff;
    // 计算 g5_2，根据 mask 中第 6 位的值进行设置，并将结果右移 5 位，然后转换为 int16_t 类型
    constexpr int16_t g5_2 = ((mask & 32) >> 5) * 0xffff;
    // 计算 g6_2，根据 mask 中第 7 位的值进行设置，并将结果右移 6 位，然后转换为 int16_t 类型
    constexpr int16_t g6_2 = ((mask & 64) >> 6) * 0xffff;
    // 计算 g7_2，根据 mask 中第 8 位的值进行设置，并将结果右移 7 位，然后转换为 int16_t 类型
    constexpr int16_t g7_2 = ((mask & 128) >> 7) * 0xffff;

    // 创建第一个 256 位整型向量 mask_1st，使用之前计算的 g0 到 g7 的值
    const vint16 mask_1st = vint16{g0, g1, g2, g3, g4, g5, g6, g7};
    // 创建第二个 256 位整型向量 mask_2nd，使用之前计算的 g0_2 到 g7_2 的值
    const vint16 mask_2nd = vint16{g0_2, g1_2, g2_2, g3_2, g4_2, g5_2, g6_2, g7_2};
    // 返回由两个向量 a 和 b 根据 mask_1st 和 mask_2nd 进行混合后的结果向量
    return {
        (vint16)vec_sel(a._vec0, b._vec0, (vbool16)mask_1st),
        (vint16)vec_sel(a._vec1, b._vec1, (vbool16)mask_2nd)};
  }

  // 静态函数：blendv
  static Vectorized<int16_t> C10_ALWAYS_INLINE blendv(
      const Vectorized<int16_t>& a,
      const Vectorized<int16_t>& b,
      const Vectorized<int16_t>& mask) {
    // 使用由比较 vec256 返回的掩码，在 vec_sel 中直接使用相同的掩码
    // 警告：Intel 风格的掩码可能无法正常工作
    return {
        vec_sel(a._vec0, b._vec0, mask._vecb0),
        vec_sel(a._vec1, b._vec1, mask._vecb1)};
  }

  // 模板函数：arange
  template <typename step_t>
  static Vectorized<int16_t> arange(int16_t base = 0, step_t step = static_cast<step_t>(1)) {
    // 返回一个 Vectorized<int16_t> 对象，其中包含根据 base 和 step 计算得到的连续整数序列
    return Vectorized<int16_t>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
  }

  // 静态函数：set
  static Vectorized<int16_t> set(
      const Vectorized<int16_t>& a,
      const Vectorized<int16_t>& b,
      size_t count = size()) {
    // 根据 count 的不同值，返回 a 和 b 的混合结果
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
    // 默认情况，返回 b
    return b;
  }

  // 静态函数：loadu
  static Vectorized<value_type> C10_ALWAYS_INLINE
  loadu(const void* ptr, int count = size()) {
    // 如果 count 等于 size()，则直接返回两个向量的加载结果
    if (count == size()) {
      return {
          vec_vsx_ld(offset0, reinterpret_cast<const value_type*>(ptr)),
          vec_vsx_ld(offset16, reinterpret_cast<const value_type*>(ptr))};
    }

    // 创建临时数组 tmp_values 来存储部分数据，并将部分数据拷贝到 tmp_values 中
    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(tmp_values, ptr, std::min(count, size()) * sizeof(value_type));

    // 返回使用 tmp_values 加载的两个向量结果
    return {vec_vsx_ld(offset0, tmp_values), vec_vsx_ld(offset16, tmp_values)};
  }
  // 将向量数据存储到指定的内存位置，支持不完全存储（count < size()）情况
  void C10_ALWAYS_INLINE store(void* ptr, int count = size()) const {
    if (count == size()) {
      // 如果 count 等于 size()，则同时存储两个向量到同一内存位置
      vec_vsx_st(_vec0, offset0, reinterpret_cast<value_type*>(ptr));
      vec_vsx_st(_vec1, offset16, reinterpret_cast<value_type*>(ptr));
    } else if (count > 0) {
      // 否则，创建临时数组 tmp_values，将向量数据存储到 tmp_values 中，并将部分数据拷贝到 ptr 所指向的内存位置
      __at_align__ value_type tmp_values[size()];
      vec_vsx_st(_vec0, offset0, tmp_values);
      vec_vsx_st(_vec1, offset16, tmp_values);
      std::memcpy(ptr, tmp_values, std::min(count, size()) * sizeof(value_type));
    }
  }
  // 禁止访问特定索引的操作符重载
  const int16_t& operator[](int idx) const = delete;
  int16_t& operator[](int idx) = delete;

  // 返回向量的角度，根据向量的正负来决定返回的值
  Vectorized<int16_t> angle() const {
    return blendv(
      Vectorized<int16_t>(0), Vectorized<int16_t>(c10::pi<int16_t>), *this < Vectorized<int16_t>(0));
  }
  // 返回向量的实部，即返回自身向量
  Vectorized<int16_t> real() const {
    return *this;
  }
  // 返回向量的虚部，所有元素设置为0
  Vectorized<int16_t> imag() const {
    return Vectorized<int16_t>{0};
  }
  // 返回向量的共轭，即返回自身向量
  Vectorized<int16_t> conj() const {
    return *this;
  }

  // 返回向量每个元素的绝对值组成的向量
  Vectorized<int16_t> C10_ALWAYS_INLINE abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  // 返回向量每个元素的负值组成的向量
  Vectorized<int16_t> C10_ALWAYS_INLINE neg() const {
    return {vec_neg(_vec0), vec_neg(_vec1)};
  }

  // 定义按位取反的一元运算符重载
  DEFINE_MEMBER_UNARY_OP(operator~, int16_t, vec_not)
  // 定义相等比较的成员运算符重载
  DEFINE_MEMBER_OP(operator==, int16_t, vec_cmpeq)
  // 定义不等比较的成员运算符重载
  DEFINE_MEMBER_OP(operator!=, int16_t, vec_cmpne)
  // 定义小于比较的成员运算符重载
  DEFINE_MEMBER_OP(operator<, int16_t, vec_cmplt)
  // 定义小于等于比较的成员运算符重载
  DEFINE_MEMBER_OP(operator<=, int16_t, vec_cmple)
  // 定义大于比较的成员运算符重载
  DEFINE_MEMBER_OP(operator>, int16_t, vec_cmpgt)
  // 定义大于等于比较的成员运算符重载
  DEFINE_MEMBER_OP(operator>=, int16_t, vec_cmpge)
  // 定义相等比较并且与一个值比较的成员运算符重载
  DEFINE_MEMBER_OP_AND_ONE(eq, int16_t, vec_cmpeq)
  // 定义不等比较并且与一个值比较的成员运算符重载
  DEFINE_MEMBER_OP_AND_ONE(ne, int16_t, vec_cmpne)
  // 定义小于比较并且与一个值比较的成员运算符重载
  DEFINE_MEMBER_OP_AND_ONE(lt, int16_t, vec_cmplt)
  // 定义小于等于比较并且与一个值比较的成员运算符重载
  DEFINE_MEMBER_OP_AND_ONE(le, int16_t, vec_cmple)
  // 定义大于比较并且与一个值比较的成员运算符重载
  DEFINE_MEMBER_OP_AND_ONE(gt, int16_t, vec_cmpgt)
  // 定义大于等于比较并且与一个值比较的成员运算符重载
  DEFINE_MEMBER_OP_AND_ONE(ge, int16_t, vec_cmpge)
  // 定义加法的成员运算符重载
  DEFINE_MEMBER_OP(operator+, int16_t, vec_add)
  // 定义减法的成员运算符重载
  DEFINE_MEMBER_OP(operator-, int16_t, vec_sub)
  // 定义乘法的成员运算符重载
  DEFINE_MEMBER_OP(operator*, int16_t, vec_mul)
  // 定义除法的成员运算符重载，模拟二元操作符
  DEFINE_MEMBER_EMULATE_BINARY_OP(operator/, int16_t, /)
  // 定义取最大值的成员运算符重载
  DEFINE_MEMBER_OP(maximum, int16_t, vec_max)
  // 定义取最小值的成员运算符重载
  DEFINE_MEMBER_OP(minimum, int16_t, vec_min)
  // 定义按位与的成员运算符重载
  DEFINE_MEMBER_OP(operator&, int16_t, vec_and)
  // 定义按位或的成员运算符重载
  DEFINE_MEMBER_OP(operator|, int16_t, vec_or)
  // 定义按位异或的成员运算符重载
  DEFINE_MEMBER_OP(operator^, int16_t, vec_xor)
};

// 结束 vec 命名空间和 at 命名空间

template <>
Vectorized<int16_t> inline operator<<(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    // 将 b 的向量数据转换为 vuint16 类型
    vuint16 shift_vec0 = reinterpret_cast<vuint16>(b.vec0());
    vuint16 shift_vec1 = reinterpret_cast<vuint16>(b.vec1());
    // 返回 a 向左移动 b 的位数后的结果向量
    return Vectorized<int16_t>{vec_sl(a.vec0(), shift_vec0), vec_sl(a.vec1(), shift_vec1)};
}

template <>
Vectorized<int16_t> inline operator>>(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
    // 将 b 的向量数据转换为 vuint16 类型
    vuint16 shift_vec0 = reinterpret_cast<vuint16>(b.vec0());
    vuint16 shift_vec1 = reinterpret_cast<vuint16>(b.vec1()) ;
    // 返回 a 向右移动 b 的位数后的结果向量
    return Vectorized<int16_t>{vec_sr(a.vec0(), shift_vec0), vec_sr(a.vec1(), shift_vec1)};
}

template <>
Vectorized<int16_t> inline maximum(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  // 调用 Vectorized 类的 maximum 方法，返回 a 和 b 各位置的最大值向量
  return a.maximum(b);
}

template <>
Vectorized<int16_t> inline minimum(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  // 调用 Vectorized 类的 minimum 方法，返回 a 和 b 各位置的最小值向量
  return a.minimum(b);
}

// 结束 vec 命名空间和 at 命名空间
} // namespace
} // namespace vec
} // namespace at
```