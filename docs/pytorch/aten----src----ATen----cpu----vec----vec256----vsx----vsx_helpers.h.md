# `.\pytorch\aten\src\ATen\cpu\vec\vec256\vsx\vsx_helpers.h`

```
#pragma once
#include <cstdint>
#include <c10/macros/Macros.h>
#include <ATen/cpu/vec/intrinsics.h>

#if defined(__clang__)
// 定义 Clang 下的向量布尔类型
typedef __vector __bool char vbool8;
typedef __vector __bool short vbool16;
typedef __vector __bool int vbool32;
typedef __vector __bool long long vbool64;
// 定义 Clang 下的向量类型
using vint8    = __attribute__((vector_size(16))) signed char;
using vint16   = __attribute__((vector_size(16))) signed short;
using vint32   = __attribute__((vector_size(16))) signed int;
using vint64   = __attribute__((vector_size(16))) signed long long;
using vuint8   = __attribute__((vector_size(16))) unsigned char;
using vuint16  = __attribute__((vector_size(16))) unsigned short;
using vuint32  = __attribute__((vector_size(16))) unsigned int;
using vuint64  = __attribute__((vector_size(16))) unsigned long long;
using vfloat32 = __attribute__((vector_size(16))) float;
using vfloat64 = __attribute__((vector_size(16))) double;
#else
// 定义非 Clang 编译器下的向量布尔类型和向量类型（使用 altivec）
using vbool8   =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) char;
using vbool16  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) short;
using vbool32  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) int;
using vbool64  =  __attribute__((altivec(vector__))) __attribute__((altivec(bool__))) long long;
using vint8    =  __attribute__((altivec(vector__)))  signed char;
using vint16   =  __attribute__((altivec(vector__)))  signed short;
using vint32   =  __attribute__((altivec(vector__)))  signed int;
using vint64   =  __attribute__((altivec(vector__)))  signed long long;
using vuint8   =  __attribute__((altivec(vector__)))  unsigned char;
using vuint16  =  __attribute__((altivec(vector__)))  unsigned short;
using vuint32  =  __attribute__((altivec(vector__)))  unsigned int;
using vuint64  =  __attribute__((altivec(vector__)))  unsigned long long;
using vfloat32 =  __attribute__((altivec(vector__)))  float;
using vfloat64 =  __attribute__((altivec(vector__)))  double;
#endif

#if !defined(vec_float)
// 如果 vec_float 函数未定义，则定义为将 vint32 向量转换为 vfloat32 向量的函数
C10_ALWAYS_INLINE vfloat32 vec_float(const vint32& vec_in) {
  vfloat32 vec_out;
  // 使用 asm 内联汇编执行向量整数到向量浮点数的转换
  __asm__("xvcvsxwsp %x0,%x1" : "=wf"(vec_out) : "wa"(vec_in));
  return vec_out;
}
#endif

#if !defined(vec_signed)
// 如果 vec_signed 函数未定义，则定义两个重载函数，分别用于 vfloat32 和 vfloat64 的转换
C10_ALWAYS_INLINE vint32 vec_signed(const vfloat32& vec_in) {
  vint32 vec_out;
  // 使用 asm 内联汇编执行向量浮点数到向量整数（有符号 32 位）的转换
  __asm__("xvcvspsxws %x0,%x1" : "=wa"(vec_out) : "wf"(vec_in));
  return vec_out;
}

C10_ALWAYS_INLINE vint64 vec_signed(const vfloat64& vec_in) {
  vint64 vec_out;
  // 使用 asm 内联汇编执行向量浮点数到向量整数（有符号 64 位）的转换
  __asm__("xvcvdpsxds %x0,%x1" : "=wa"(vec_out) : "wd"(vec_in));
  return vec_out;
}
#endif

#if !defined(vec_neg)
// 如果 vec_neg 函数未定义，则定义两个重载函数，分别用于 vfloat32 和 vfloat64 的取负操作
C10_ALWAYS_INLINE vfloat32 vec_neg(const vfloat32& vec_in) {
  vfloat32 vec_out;
  // 使用 asm 内联汇编执行向量浮点数取负的操作
  __asm__("xvnegsp %x0,%x1" : "=wf"(vec_out) : "wf"(vec_in));
  return vec_out;
}

C10_ALWAYS_INLINE vfloat64 vec_neg(const vfloat64& vec_in) {
  vfloat64 vec_out;
  // 使用 asm 内联汇编执行向量双精度浮点数取负的操作
  __asm__("xvnegdp %x0,%x1" : "=wd"(vec_out) : "wd"(vec_in));
  return vec_out;
}
#endif


这段代码是一些向量处理的宏定义和函数声明，使用了特定的汇编语言指令来实现向量操作。
// 定义一个宏，用于创建一个 vint16 类型的全零向量，并对输入向量取负
C10_ALWAYS_INLINE vint16 vec_neg(const vint16& vec_in) {
  vint16 vint0 = {0, 0, 0, 0 ,0, 0, 0, 0};
  return vec_vsubuhm(vint0, vec_in);
}

// 定义一个宏，用于创建一个 vint32 类型的全零向量，并对输入向量取负
C10_ALWAYS_INLINE vint32 vec_neg(const vint32& vec_in) {
  vint32 vint0 = {0, 0, 0, 0};
  return vec_vsubuwm(vint0, vec_in);
}

// 对于输入的 vint64 向量，直接对其取负
C10_ALWAYS_INLINE vint64 vec_neg(const vint64& vec_in) {
  return -vec_in;
}
#endif

#if !defined(vec_sldw)
// 一个模板函数，用于将两个 vfloat32 向量进行字节级别的向左位移
template <unsigned int C>
C10_ALWAYS_INLINE vfloat32
vec_sldw_aux(const vfloat32& vec_in0, const vfloat32& vec_in1) {
  vfloat32 vec_out;
  __asm("xxsldwi %x0, %x1, %x2, %3 "
        : "=wa"(vec_out)
        : "wa"(vec_in0), "wa"(vec_in1), "I"(C));
  return vec_out;
}

// 定义一个宏，用于调用 vec_sldw_aux 函数，参数 C 为位移量
#define vec_sldw(a, b, c) vec_sldw_aux<c>(a, b)
#endif

// 定义一个宏，用于对输入向量中的每个位进行逻辑非操作
#define vec_not(a) vec_nor(a, a)

// 当编译器为 clang 且未定义 vec_splats 时，定义一个模板函数用于生成所有元素均为给定值的 vint64 向量
#if defined(__clang__) && !defined(vec_splats)
C10_ALWAYS_INLINE vint64 vec_splats(const int64_t& a) {
  return vec_splats(a);
}
#endif

// 一个模板函数，用于对两个数进行比较，并返回较小值（如果其中有 NaN，则返回 NaN）
template <class T>
C10_ALWAYS_INLINE T vec_min_nan(const T& a, const T& b) {
  return vec_min(a, b);
}

// 一个模板函数，用于对两个数进行比较，并返回较大值（如果其中有 NaN，则返回 NaN）
template <class T>
C10_ALWAYS_INLINE T vec_max_nan(const T& a, const T& b) {
  return vec_max(a, b);
}

// 为 vfloat32 类型特化 vec_min_nan 模板函数，处理 NaN 的情况
template<>
C10_ALWAYS_INLINE vfloat32 vec_min_nan<vfloat32>(const vfloat32& a, const vfloat32& b)
{
  // 使用 SIMD 指令处理 NaN 的情况，返回较小值（如果其中有 NaN，则返回 NaN）
  vfloat32 ret;
  __asm__ ("xvcmpgesp %x0,%x1,%x2\n\txxsel %x0,%x1,%x2,%x0" : "=&wa" (ret) : "wa" (a), "wa" (b));
  return ret;
}

// 为 vfloat32 类型特化 vec_max_nan 模板函数，处理 NaN 的情况
template<>
C10_ALWAYS_INLINE vfloat32 vec_max_nan<vfloat32>(const vfloat32& a, const vfloat32& b)
{
  // 使用 SIMD 指令处理 NaN 的情况，返回较大值（如果其中有 NaN，则返回 NaN）
  vfloat32 ret;
   __asm__ ("xvcmpgtsp %x0,%x2,%x1\n\txxsel %x0,%x1,%x2,%x0" : "=&wa" (ret) : "wa" (a), "wa" (b));
  return ret;
}

// 为 vfloat64 类型特化 vec_min_nan 模板函数，处理 NaN 的情况
template<>
C10_ALWAYS_INLINE vfloat64 vec_min_nan<vfloat64>(const vfloat64& a, const vfloat64& b)
{
  // 使用 SIMD 指令处理 NaN 的情况，返回较小值（如果其中有 NaN，则返回 NaN）
  vfloat64 ret;
  __asm__ ("xvcmpgedp %x0,%x1,%x2\n\txxsel %x0,%x1,%x2,%x0" : "=&wa" (ret) : "wa" (a), "wa" (b));
  return ret;
}

// 为 vfloat64 类型特化 vec_max_nan 模板函数，处理 NaN 的情况
template<>
C10_ALWAYS_INLINE vfloat64 vec_max_nan<vfloat64>(const vfloat64& a, const vfloat64& b)
{
  // 使用 SIMD 指令处理 NaN 的情况，返回较大值（如果其中有 NaN，则返回 NaN）
  vfloat64 ret;
  __asm__ ("xvcmpgtdp %x0,%x2,%x1\n\txxsel %x0,%x1,%x2,%x0" : "=&wa" (ret) : "wa" (a), "wa" (b));
  return ret;
}

// 定义一个宏，用于比较两个向量并返回 NaN（非数值） 的处理结果
#define C10_VSX_VEC_NAN_PROPAG(name, type, btype, func)       \
  C10_ALWAYS_INLINE type name(const type& a, const type& b) { \
    type tmp = func(a, b);                                    \
    btype nan_a = vec_cmpne(a, a);                            \
    btype nan_b = vec_cmpne(b, b);                            \
    tmp = vec_sel(tmp, a, nan_a);                             \
    return vec_sel(tmp, b, nan_b);                            \
  }


注释：

    // 返回一个通过 vec_sel 函数选择处理后的向量 tmp，使用向量 b 和 nan_b 进行选择
    // 这是一个函数结束的语句
C10_VSX_VEC_NAN_PROPAG(vec_min_nan2, vfloat32, vbool32, vec_min)
// 宏展开：定义一个用于处理 NaN 传播的向量操作，针对 vfloat32 类型，使用 vec_min 函数

C10_VSX_VEC_NAN_PROPAG(vec_max_nan2, vfloat32, vbool32, vec_max)
// 宏展开：定义一个用于处理 NaN 传播的向量操作，针对 vfloat32 类型，使用 vec_max 函数

C10_VSX_VEC_NAN_PROPAG(vec_min_nan2, vfloat64, vbool64, vec_min)
// 宏展开：定义一个用于处理 NaN 传播的向量操作，针对 vfloat64 类型，使用 vec_min 函数

C10_VSX_VEC_NAN_PROPAG(vec_max_nan2, vfloat64, vbool64, vec_max)
// 宏展开：定义一个用于处理 NaN 传播的向量操作，针对 vfloat64 类型，使用 vec_max 函数

#undef C10_VSX_VEC_NAN_PROPAG
// 取消之前定义的 C10_VSX_VEC_NAN_PROPAG 宏的定义

#define DEFINE_MEMBER_UNARY_OP(op, op_type, func)     \
  Vectorized<op_type> C10_ALWAYS_INLINE op() const {      \
    return Vectorized<op_type>{func(_vec0), func(_vec1)}; \
  }
// 定义一个用于成员函数形式的一元操作符宏，操作符由 func 指定，返回一个新的 Vectorized 对象

#define DEFINE_MEMBER_OP(op, op_type, func)                                  \
  Vectorized<op_type> C10_ALWAYS_INLINE op(const Vectorized<op_type>& other) const { \
    return Vectorized<op_type>{                                                  \
        func(_vec0, other._vec0), func(_vec1, other._vec1)};                 \
  }
// 定义一个用于成员函数形式的二元操作符宏，操作符由 func 指定，接受另一个 Vectorized 对象作为参数

#define DEFINE_MEMBER_BITWISE_OP(op, op_type, func)                          \
  Vectorized<op_type> C10_ALWAYS_INLINE op(const Vectorized<op_type>& other) const { \
    return Vectorized<op_type>{                                                  \
        func(_vecb0, other._vecb0), func(_vecb1, other._vecb1)};             \
  }
// 定义一个用于成员函数形式的按位二元操作符宏，操作符由 func 指定，接受另一个 Vectorized 对象作为参数

#define DEFINE_MEMBER_TERNARY_OP(op, op_type, func)                    \
  Vectorized<op_type> C10_ALWAYS_INLINE op(                                \
      const Vectorized<op_type>& b, const Vectorized<op_type>& c) const {      \
    return Vectorized<op_type>{                                            \
        func(_vec0, b._vec0, c._vec0), func(_vec1, b._vec1, c._vec1)}; \
  }
// 定义一个用于成员函数形式的三元操作符宏，操作符由 func 指定，接受另外两个 Vectorized 对象作为参数

#define DEFINE_MEMBER_EMULATE_BINARY_OP(op, op_type, binary_op)          \
  Vectorized<op_type> C10_ALWAYS_INLINE op(const Vectorized<op_type>& b) const { \
    Vectorized<op_type>::vec_internal_type ret_0;                         \
    Vectorized<op_type>::vec_internal_type ret_1;                         \
    for (int i = 0; i < Vectorized<op_type>::size() / 2; i++) {           \
      ret_0[i] = _vec0[i] binary_op b._vec0[i];                       \
      ret_1[i] = _vec1[i] binary_op b._vec1[i];                       \
    }                                                                 \
    return Vectorized<op_type>{ret_0, ret_1};                             \
  }
// 定义一个用于成员函数形式的仿真二元操作符宏，二元操作符由 binary_op 指定，接受另一个 Vectorized 对象作为参数

#define DEFINE_MEMBER_OP_AND_ONE(op, op_type, func)                          \
  Vectorized<op_type> C10_ALWAYS_INLINE op(const Vectorized<op_type>& other) const { \
    using vvtype = Vectorized<op_type>::vec_internal_type;                       \
    const vvtype v_one = vec_splats(static_cast<op_type>(1.0));              \
    vvtype ret0 = (vvtype)func(_vec0, other._vec0);                          \
    vvtype ret1 = (vvtype)func(_vec1, other._vec1);                          \
    return Vectorized<op_type>{vec_and(ret0, v_one), vec_and(ret1, v_one)};      \
  }
// 定义一个用于成员函数形式的二元操作符宏，操作符由 func 指定，接受另一个 Vectorized 对象作为参数，并与常数 1 进行按位与操作
# 定义用于限制操作数类型的模板函数
#define DEFINE_CLAMP_FUNCS(operand_type)                                        \
  # 在特化的模板中定义clamp函数，限制向量化操作数为operand_type类型
  template <>                                                                   \
  Vectorized<operand_type> C10_ALWAYS_INLINE clamp(                             \
      const Vectorized<operand_type>& a,                                        \
      const Vectorized<operand_type>& min,                                      \
      const Vectorized<operand_type>& max) {                                    \
    return Vectorized<operand_type>{                                            \
        # 对每个向量元素执行vec_max_nan和vec_min_nan操作，确保在[min, max]范围内
        vec_min_nan(vec_max_nan(a.vec0(), min.vec0()), max.vec0()),             \
        vec_min_nan(vec_max_nan(a.vec1(), min.vec1()), max.vec1())};            \
  }                                                                             \
  # 在特化的模板中定义clamp_min函数，限制向量化操作数为operand_type类型
  template <>                                                                   \
  Vectorized<operand_type> C10_ALWAYS_INLINE clamp_min(                         \
      const Vectorized<operand_type>& a, const Vectorized<operand_type>& min) { \
    return Vectorized<operand_type>{                                            \
        # 对每个向量元素执行vec_max_nan操作，确保不小于min
        vec_max_nan(a.vec0(), min.vec0()),                                      \
        vec_max_nan(a.vec1(), min.vec1())};                                     \
  }                                                                             \
  # 在特化的模板中定义clamp_max函数，限制向量化操作数为operand_type类型
  template <>                                                                   \
  Vectorized<operand_type> C10_ALWAYS_INLINE clamp_max(                         \
      const Vectorized<operand_type>& a, const Vectorized<operand_type>& max) { \
    return Vectorized<operand_type>{                                            \
        # 对每个向量元素执行vec_min_nan操作，确保不大于max
        vec_min_nan(a.vec0(), max.vec0()),                                      \
        vec_min_nan(a.vec1(), max.vec1())};                                     \
  }

# 定义用于重新解释类型转换的模板函数
#define DEFINE_REINTERPRET_CAST_FUNCS(                             \
    first_type, cast_type, cast_inner_vector_type)                 \
  # 在特化的模板中定义cast函数，将Vectorized<first_type>转换为Vectorized<cast_type>
  template <>                                                      \
  C10_ALWAYS_INLINE Vectorized<cast_type> cast<cast_type, first_type>( \
      const Vectorized<first_type>& src) {                                 \
    return Vectorized<cast_type>{                                      \
        # 将每个向量元素强制类型转换为cast_inner_vector_type类型
        (cast_inner_vector_type)src.vec0(),                            \
        (cast_inner_vector_type)src.vec1()};                           \
  }

# 定义将特定类型first_type转换为所有支持类型的模板函数
#define DEFINE_REINTERPRET_CAST_TO_ALL_FUNCS(first_type)     \
  # 将first_type类型转换为double类型的模板特化
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, double, vfloat64)    \
  # 将first_type类型转换为float类型的模板特化
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, float, vfloat32)     \
  # 将first_type类型转换为int64_t类型的模板特化
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, int64_t, vint64) \
  # 将first_type类型转换为int32_t类型的模板特化
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, int32_t, vint32)   \
  # 将first_type类型转换为int16_t类型的模板特化
  DEFINE_REINTERPRET_CAST_FUNCS(first_type, int16_t, vint16)

# 可用于加快模拟混合操作的说明
// 定义一个 constexpr 函数 blendChoice，用于根据给定掩码 mask 选择合适的混合方式
constexpr int blendChoice(uint32_t mask, uint32_t half1 = 0xF, uint32_t half2 = 0xF0) {
  // 定义掩码值 none 和 both
  uint32_t none = 0;
  uint32_t both = half1 | half2;
  // 将 mask 限制在 0 和 both 之间
  mask = mask & both;
  // 根据 mask 的不同值返回不同的混合方式编号
  if (mask == none) return 0;
  else if (mask == both) return 1;
  else if (mask == half1) return 2;
  else if (mask == half2) return 3;
  else if (mask > 0 && mask < half1) return 4;
  else if ((mask & half2) == half2) return 5;
  else if ((mask & half1) == 0 && mask > half1) return 6;
  else if ((mask & half1) == half1 && mask > half1) return 7;
  else return 8;
}

// 定义一个 constexpr 函数 blendChoiceDbl，使用 blendChoice 来快速模拟双倍混合
constexpr int blendChoiceDbl(uint32_t mask) {
  // 将 mask 限制在 0 和 0xF 之间，然后调用 blendChoice 函数
  return blendChoice(mask, 0x3, 0xC);
}

// 定义一个 constexpr 函数 VsxMask1，根据 mask 创建一个 vbool32 结构的向量掩码
constexpr vbool32 VsxMask1(uint32_t mask) {
  // 根据 mask 的不同位设置 g0, g1, g2, g3 的值，并返回 vbool32 结构
  uint32_t g0 = (mask & 1) * 0xffffffff;
  uint32_t g1 = ((mask & 2) >> 1) * 0xffffffff;
  uint32_t g2 = ((mask & 4) >> 2) * 0xffffffff;
  uint32_t g3 = ((mask & 8) >> 3) * 0xffffffff;
  return (vbool32){g0, g1, g2, g3};
}

// 定义一个 constexpr 函数 VsxMask2，根据 mask 创建一个 vbool32 结构的向量掩码
constexpr vbool32 VsxMask2(uint32_t mask) {
  // 将 mask 中的低 8 位提取出来作为 mask2，然后调用 VsxMask1 函数
  uint32_t mask2 = (mask & 0xFF) >> 4;
  return VsxMask1(mask2);
}

// 定义一个 constexpr 函数 VsxDblMask1，根据 mask 创建一个 vbool64 结构的双精度向量掩码
constexpr vbool64 VsxDblMask1(uint32_t mask) {
  // 根据 mask 的不同位设置 g0, g1 的值，并返回 vbool64 结构
  uint64_t g0 = (mask & 1) * 0xffffffffffffffff;
  uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffff;
  return (vbool64){g0, g1};
}

// 定义一个 constexpr 函数 VsxDblMask2，根据 mask 创建一个 vbool64 结构的双精度向量掩码
constexpr vbool64 VsxDblMask2(uint32_t mask) {
  // 将 mask 中的低 4 位提取出来作为 mask2，然后调用 VsxDblMask1 函数
  uint32_t mask2 = (mask & 0xF) >> 2;
  return VsxDblMask1(mask2);
}

// 定义一个 constexpr 函数 maskForComplex，根据 mask 创建一个复杂的整数掩码
constexpr int maskForComplex(uint32_t mask) {
  // 将 mask 限制在 0 和 0xF 之间，然后根据不同位设置复杂掩码 complex_mask
  mask = mask & 0xF;
  int complex_mask = 0;
  if (mask & 1) complex_mask |= 3;
  if (mask & 2) complex_mask |= (3 << 2);
  if (mask & 4) complex_mask |= (3 << 4);
  if (mask & 8) complex_mask |= (3 << 6);
  return complex_mask;
}

// 定义一个 constexpr 函数 maskForComplexDbl，根据 mask 创建一个复杂的整数掩码
constexpr int maskForComplexDbl(uint32_t mask) {
  // 将 mask 限制在 0 和 0x3 之间，然后根据不同位设置复杂掩码 complex_mask
  mask = mask & 0x3;
  int complex_mask = 0;
  if (mask & 1) complex_mask |= 3;
  if (mask & 2) complex_mask |= (3 << 2);
  return complex_mask;
}

// 定义一个 constexpr 函数 blendChoiceComplex，根据 mask 创建一个复杂的混合选择
constexpr int blendChoiceComplex(uint32_t mask) {
  // 调用 blendChoice 函数，使用 maskForComplex 函数生成的复杂掩码
  return blendChoice(maskForComplex(mask));
}

// 定义一个 constexpr 函数 blendChoiceComplexDbl，根据 mask 创建一个复杂的双倍混合选择
constexpr int blendChoiceComplexDbl(uint32_t mask) {
  // 调用 blendChoiceDbl 函数，使用 maskForComplexDbl 函数生成的复杂掩码
  return blendChoiceDbl(maskForComplexDbl(mask));
}

// 定义一个 constexpr 函数 VsxComplexMask1，根据 mask 创建一个复杂的 vbool32 结构的向量掩码
constexpr vbool32 VsxComplexMask1(uint32_t mask) {
  // 调用 VsxMask1 函数，使用 maskForComplex 函数生成的复杂掩码
  return VsxMask1(maskForComplex(mask));
}

// 定义一个 constexpr 函数 VsxComplexMask2，根据 mask 创建一个复杂的 vbool32 结构的向量掩码
constexpr vbool32 VsxComplexMask2(uint32_t mask) {
  // 将 mask 中的低 4 位提取出来作为 mask2，然后调用 VsxMask1 函数
  uint32_t mask2 = (mask & 0xF) >> 2;
  return VsxMask1(maskForComplex(mask2));
}

// 定义一个 constexpr 函数 VsxComplexDblMask1，根据 mask 创建一个复杂的 vbool64 结构的双精度向量掩码
constexpr vbool64 VsxComplexDblMask1(uint32_t mask) {
  // 调用 VsxDblMask1 函数，使用 maskForComplex 函数生成的复杂掩码
  return VsxDblMask1(maskForComplex(mask));
}

// 定义一个 constexpr 函数 VsxComplexDblMask2，根据 mask 创建一个复杂的 vbool64 结构的双精度向量掩码
constexpr vbool64 VsxComplexDblMask2(uint32_t mask) {
  // 将 mask 中的低 4 位提取出来作为 mask2，然后调用 VsxDblMask1 函数
  uint32_t mask2 = (mask & 0xF) >> 2;
  return VsxDblMask1(mask2);
}

// 命名空间常量 at::vec::CPU_CAPABILITY 下的常量定义
namespace at {
namespace vec {
inline namespace CPU_CAPABILITY {
// 偏移常量定义
constexpr int offset0 = 0;
constexpr int offset16 = 16;
// #Constants
const vuint8 mask_zero_bits = vuint8{128, 128, 128, 128, 128, 128, 128, 128,
                                128, 128, 128, 128, 96,  64,  32,  0};
// 定义一个包含特定值的 vuint8 类型的常量，用于掩码操作

const vuint8 swap_mask =
    vuint8{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
// 定义一个用于交换操作的 vuint8 类型的常量，对 SIMD 操作有特定用途

const vint32 v0x7f = vec_splats(0x7f);
// 创建一个 vint32 类型的常量，所有元素初始化为十六进制数 0x7f

const vint32 vi_0 = vec_splats((int)(0));
// 创建一个 vint32 类型的常量，所有元素初始化为整数 0

const vint32 vi_1 = vec_splats((int)1);
// 创建一个 vint32 类型的常量，所有元素初始化为整数 1

const vint32 vi_2 = vec_splats((int)2);
// 创建一个 vint32 类型的常量，所有元素初始化为整数 2

const vint32 vi_4 = vec_splats((int)4);
// 创建一个 vint32 类型的常量，所有元素初始化为整数 4

const vint32 vi_inv1 = vec_splats((int)~1);
// 创建一个 vint32 类型的常量，所有元素初始化为整数 ~1 的补码

const vuint32 vu_29 = vec_splats(29u);
// 创建一个 vuint32 类型的常量，所有元素初始化为无符号整数 29

const vuint32 vu_23 = vec_splats(23u);
// 创建一个 vuint32 类型的常量，所有元素初始化为无符号整数 23

const vbool32 inv_mant_mask = (vbool32)vec_splats((unsigned int)~0xff800000);
// 创建一个 vbool32 类型的常量，用于对浮点数的指数位进行掩码操作

const vbool32 sign_mask = (vbool32)vec_splats((int)0x80000000);
// 创建一个 vbool32 类型的常量，用于对浮点数的符号位进行掩码操作

const vbool32 real_mask = vbool32{0xFFFFFFFF, 0x0, 0xFFFFFFFF, 0x0};
// 创建一个 vbool32 类型的常量，用于指定实部在 SIMD 操作中的掩码

const vbool32 imag_mask = vbool32{0x0, 0xFFFFFFFF, 0x0, 0xFFFFFFFF};
// 创建一个 vbool32 类型的常量，用于指定虚部在 SIMD 操作中的掩码

const vbool32 isign_mask = vbool32{0x0, 0x80000000, 0x0, 0x80000000};
// 创建一个 vbool32 类型的常量，用于对浮点数的复数符号位进行掩码操作

const vbool32 rsign_mask = vbool32{0x80000000, 0x0, 0x80000000, 0x0};
// 创建一个 vbool32 类型的常量，用于对浮点数的实数符号位进行掩码操作

const vbool64 vd_sign_mask  = vbool64{0x8000000000000000, 0x8000000000000000};
// 创建一个 vbool64 类型的常量，用于对双精度浮点数的符号位进行掩码操作

const vbool64 vd_imag_mask  = vbool64{0x0, 0xFFFFFFFFFFFFFFFF};
// 创建一个 vbool64 类型的常量，用于指定双精度浮点数的虚部在 SIMD 操作中的掩码

const vbool64 vd_real_mask  = vbool64{0xFFFFFFFFFFFFFFFF, 0x0};
// 创建一个 vbool64 类型的常量，用于指定双精度浮点数的实部在 SIMD 操作中的掩码

const vbool64 vd_isign_mask = vbool64{0x0, 0x8000000000000000};
// 创建一个 vbool64 类型的常量，用于对双精度浮点数的复数符号位进行掩码操作

const vbool64 vd_rsign_mask = vbool64{0x8000000000000000, 0x0};
// 创建一个 vbool64 类型的常量，用于对双精度浮点数的实数符号位进行掩码操作

const vfloat32 zero = vec_splats(0.f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为浮点数 0.0

const vfloat32 half = vec_splats(0.5f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为浮点数 0.5

const vfloat32 one = vec_splats(1.f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为浮点数 1.0

const vfloat32 two = vec_splats(2.0f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为浮点数 2.0

const vfloat32 _4div_pi = vec_splats(1.27323954473516f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为浮点数 4/π 的值

const vfloat32 v_inf = (vfloat32)vec_splats(0x7f800000u);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 IEEE 754 浮点数正无穷大

const vfloat32 v_minus_inf = vfloat32{ 0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u };
// 创建一个 vfloat32 类型的常量，所有元素初始化为 IEEE 754 浮点数负无穷大

const vfloat32 v_nan = (vfloat32)vec_splats(0x7fffffff);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 IEEE 754 浮点数 NaN

const vfloat32 log10e_inv = vec_splats(0.43429448190325176f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 log10(e) 的倒数

const vfloat32 log2e_inv = vec_splats(1.4426950408889634f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 log2(e) 的倒数

const vfloat32 log2eB_inv = vec_splats(1.442695036924675f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 log2(eB) 的倒数

const vfloat32 cephes_SQRTHF = vec_splats(0.707106781186547524f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为平方根的倒数的一半

const vfloat32 coscof_p0 = vec_splats(2.443315711809948E-005f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 cos 系列函数的系数 p0

const vfloat32 coscof_p1 = vec_splats(-1.388731625493765E-003f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 cos 系列函数的系数 p1

const vfloat32 coscof_p2 = vec_splats(4.166664568298827E-002f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 cos 系列函数的系数 p2

const vfloat32 exp_hi = vec_splats(104.f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 exp 函数的系数 hi

const vfloat32 exp_lo = vec_splats(-104.f);
// 创建一个 vfloat32 类型的常量，所有元素初始化为 exp 函数的系数 lo

const vfloat32 exp_p0 = vec_splats(0.000198527617612853646278
// 定义常量 log_p6，值为单精度浮点数 2.0000714765E-1
const vfloat32 log_p6 = vec_splats(+2.0000714765E-1f);
// 定义常量 log_p7，值为单精度浮点数 -2.4999993993E-1
const vfloat32 log_p7 = vec_splats(-2.4999993993E-1f);
// 定义常量 log_p8，值为单精度浮点数 3.3333331174E-1
const vfloat32 log_p8 = vec_splats(+3.3333331174E-1f);
// 定义常量 log_q1，值为单精度浮点数 -2.12194440e-4
const vfloat32 log_q1 = vec_splats(-2.12194440e-4f);
// 定义常量 log_q2，值为单精度浮点数 0.693359375
const vfloat32 log_q2 = vec_splats(0.693359375f);
// 定义常量 max_logf，值为单精度浮点数 88.02969187150841
const vfloat32 max_logf = vec_splats(88.02969187150841f);
// 定义常量 max_numf，值为单精度浮点数 1.7014117331926442990585209174225846272e38
const vfloat32 max_numf = vec_splats(1.7014117331926442990585209174225846272e38f);
// 定义常量 min_inf，值为单精度浮点数 0xff800000u 对应的值
const vfloat32 min_inf = (vfloat32)vec_splats(0xff800000u);
// 定义常量 min_norm_pos，值为单精度浮点数 0x0800000u 对应的值
const vfloat32 min_norm_pos = (vfloat32)vec_splats(0x0800000u);
// 定义常量 minus_cephes_dp1，值为单精度浮点数 -0.78515625
const vfloat32 minus_cephes_dp1 = vec_splats(-0.78515625f);
// 定义常量 minus_cephes_dp2，值为单精度浮点数 -2.4187564849853515625e-4
const vfloat32 minus_cephes_dp2 = vec_splats(-2.4187564849853515625e-4f);
// 定义常量 minus_cephes_dp3，值为单精度浮点数 -3.77489497744594108e-8
const vfloat32 minus_cephes_dp3 = vec_splats(-3.77489497744594108e-8f);
// 定义常量 negln2f_hi，值为单精度浮点数 -0.693145751953125
const vfloat32 negln2f_hi = vec_splats(-0.693145751953125f);
// 定义常量 negln2f_lo，值为单精度浮点数 -1.428606765330187045e-06
const vfloat32 negln2f_lo = vec_splats(-1.428606765330187045e-06f);
// 定义常量 p0，值为单精度浮点数 2.03721912945E-4
const vfloat32 p0 = vec_splats(2.03721912945E-4f);
// 定义常量 p1，值为单精度浮点数 8.33028376239E-3
const vfloat32 p1 = vec_splats(8.33028376239E-3f);
// 定义常量 p2，值为单精度浮点数 1.66667160211E-1
const vfloat32 p2 = vec_splats(1.66667160211E-1f);
// 定义常量 sincof_p0，值为单精度浮点数 -1.9515295891E-4
const vfloat32 sincof_p0 = vec_splats(-1.9515295891E-4f);
// 定义常量 sincof_p1，值为单精度浮点数 8.3321608736E-3
const vfloat32 sincof_p1 = vec_splats(8.3321608736E-3f);
// 定义常量 sincof_p2，值为单精度浮点数 -1.6666654611E-1
const vfloat32 sincof_p2 = vec_splats(-1.6666654611E-1f);
// 定义常量 tanh_0p625，值为单精度浮点数 0.625
const vfloat32 tanh_0p625 = vec_splats(0.625f);
// 定义常量 tanh_half_max，值为单精度浮点数 44.014845935754205
const vfloat32 tanh_half_max = vec_splats(44.014845935754205f);
// 定义常量 tanh_p0，值为单精度浮点数 -5.70498872745E-3
const vfloat32 tanh_p0 = vec_splats(-5.70498872745E-3f);
// 定义常量 tanh_p1，值为单精度浮点数 2.06390887954E-2
const vfloat32 tanh_p1 = vec_splats(2.06390887954E-2f);
// 定义常量 tanh_p2，值为单精度浮点数 -5.37397155531E-2
const vfloat32 tanh_p2 = vec_splats(-5.37397155531E-2f);
// 定义常量 tanh_p3，值为单精度浮点数 1.33314422036E-1
const vfloat32 tanh_p3 = vec_splats(1.33314422036E-1f);
// 定义常量 tanh_p4，值为单精度浮点数 -3.33332819422E-1
const vfloat32 tanh_p4 = vec_splats(-3.33332819422E-1f);
// 定义常量 vcheck，值为单精度浮点数 1LL 左移 24 位对应的值
const vfloat32 vcheck = vec_splats((float)(1LL << 24));
// 定义常量 imag_one，值为包含四个单精度浮点数的向量{0.f, 1.f, 0.f, 1.f}
const vfloat32 imag_one = vfloat32{0.f, 1.f, 0.f, 1.f};
// 定义常量 imag_half，值为包含四个单精度浮点数的向量{0.f, 0.5f, 0.f, 0.5f}
const vfloat32 imag_half = vfloat32{0.f, 0.5f, 0.f, 0.5f};
// 定义常量 sqrt2_2，值为包含四个单精度浮点数的向量{0.70710676908493042f, 0.70710676908493042, 0.70710676908493042, 0.70710676908493042}
const vfloat32 sqrt2_2 = vfloat32{0.70710676908493042f, 0.70710676908493042f, 0.70710676908493042f, 0.70710676908493042f};
// 定义常量 pi_2，值为包含四个单精度浮点数的向量{M_PI / 2, 0.0, M_PI / 2, 0.0}
const vfloat32 pi_2 = vfloat32{M_PI / 2, 0.0f, M_PI / 2, 0.0f};
// 定义常量 vf_89，值为包含四个单精度浮点数的向量{89.f, 89.f, 89.f, 89.f}
const vfloat32 vf_89 = vfloat32{89.f, 89.f, 89.f, 89.f};
// 定义常量 vd_one，值为双精度浮点数 1.0
const vfloat64 vd_one = vec_splats(1.0);
//
```