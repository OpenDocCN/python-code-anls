# `.\pytorch\c10\core\ScalarType.h`

```py
#pragma once

#include <c10/util/BFloat16.h> // 引入BFloat16类型的头文件
#include <c10/util/Deprecated.h> // 引入Deprecated工具函数的头文件
#include <c10/util/Exception.h> // 引入异常处理工具函数的头文件
#include <c10/util/Float8_e4m3fn.h> // 引入Float8_e4m3fn类型的头文件
#include <c10/util/Float8_e4m3fnuz.h> // 引入Float8_e4m3fnuz类型的头文件
#include <c10/util/Float8_e5m2.h> // 引入Float8_e5m2类型的头文件
#include <c10/util/Float8_e5m2fnuz.h> // 引入Float8_e5m2fnuz类型的头文件
#include <c10/util/Half.h> // 引入Half类型的头文件
#include <c10/util/bits.h> // 引入位操作工具函数的头文件
#include <c10/util/complex.h> // 引入复数类型的头文件
#include <c10/util/qint32.h> // 引入qint32类型的头文件
#include <c10/util/qint8.h> // 引入qint8类型的头文件
#include <c10/util/quint2x4.h> // 引入quint2x4类型的头文件
#include <c10/util/quint4x2.h> // 引入quint4x2类型的头文件
#include <c10/util/quint8.h> // 引入quint8类型的头文件

#include <array> // 引入array标准库
#include <cstddef> // 引入C标准库stddef.h，定义了各种常量和类型
#include <cstdint> // 引入C标准库cstdint.h，定义了特定宽度的整数类型
#include <limits> // 引入limits标准库，提供数值类型的极限值信息
#include <ostream> // 引入ostream标准库，提供输出流类及其相关操作
#include <type_traits> // 引入type_traits标准库，提供类型特性的工具类
#include <unordered_map> // 引入unordered_map标准库，提供哈希表的实现

namespace c10 {

// dummy struct for uint1 to uint7, actual functionality
// of these dtypes will be implemented in python with Tensor subclass
// 用于uint1到uint7的虚拟结构体，实际功能将在Python中通过Tensor子类实现

template <unsigned int N>
struct dummy_uint1_7_t {};

// For the macros below:
//
// For users: If you want to macro some code for all non-QInt scalar types
// (i.e. types with complete information, you probably want one of the
// AT_FORALL_SCALAR_TYPES / AT_FORALL_SCALAR_TYPES_AND macros below, which are
// designed to behave similarly to the Dispatch macros with the same name.
//
// For adding a new dtype: In the beginning, we had an idea that there was a
// list of all scalar types, and you could use AT_FORALL_SCALAR_TYPES to
// iterate over them.  But over the years we added weird types which couldn't
// be handled uniformly everywhere and so in the end we ended up with some
// mish-mosh of some helper macros, but mostly use sites making a call about
// what dtypes they can or can't support.  So if you want to add a new dtype,
// the preferred resolution is to find a dtype similar to what you want,
// grep for it and edit all the sites you find this way.  If you need to add
// a completely new kind of dtype, you're going to have to laboriously audit
// all of the sites everywhere to figure out how it should work.  Consulting
// some old PRs where we added new dtypes (check history of this file) can
// help give you an idea where to start.

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
// 以下宏的顺序很重要，在_promoteTypesLookup和序列化格式中被依赖
// 定义一个宏，用于列举所有标量类型及其对应的名称，包括基本整数类型、浮点数类型、复数类型和量化整数类型
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \  // 无符号8位整数类型，对应的名称为Byte
  _(int8_t, Char) /* 1 */                                \  // 有符号8位整数类型，对应的名称为Char
  _(int16_t, Short) /* 2 */                              \  // 有符号16位整数类型，对应的名称为Short
  _(int, Int) /* 3 */                                    \  // 有符号整数类型，对应的名称为Int
  _(int64_t, Long) /* 4 */                               \  // 有符号64位整数类型，对应的名称为Long
  _(at::Half, Half) /* 5 */                              \  // 半精度浮点数类型，对应的名称为Half
  _(float, Float) /* 6 */                                \  // 单精度浮点数类型，对应的名称为Float
  _(double, Double) /* 7 */                              \  // 双精度浮点数类型，对应的名称为Double
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */        \  // 半精度复数类型，对应的名称为ComplexHalf
  _(c10::complex<float>, ComplexFloat) /* 9 */           \  // 单精度复数类型，对应的名称为ComplexFloat
  _(c10::complex<double>, ComplexDouble) /* 10 */        \  // 双精度复数类型，对应的名称为ComplexDouble
  _(bool, Bool) /* 11 */                                 \  // 布尔类型，对应的名称为Bool
  _(c10::qint8, QInt8) /* 12 */                          \  // 8位量化整数类型，对应的名称为QInt8
  _(c10::quint8, QUInt8) /* 13 */                        \  // 无符号8位量化整数类型，对应的名称为QUInt8
  _(c10::qint32, QInt32) /* 14 */                        \  // 32位量化整数类型，对应的名称为QInt32
  _(at::BFloat16, BFloat16) /* 15 */                     \  // BF16浮点数类型，对应的名称为BFloat16
  _(c10::quint4x2, QUInt4x2) /* 16 */                    \  // 4位乘以2位的无符号整数类型，对应的名称为QUInt4x2
  _(c10::quint2x4, QUInt2x4) /* 17 */                    \  // 2位乘以4位的无符号整数类型，对应的名称为QUInt2x4
  _(c10::bits1x8, Bits1x8) /* 18 */                      \  // 1位乘以8位的位类型，对应的名称为Bits1x8
  _(c10::bits2x4, Bits2x4) /* 19 */                      \  // 2位乘以4位的位类型，对应的名称为Bits2x4
  _(c10::bits4x2, Bits4x2) /* 20 */                      \  // 4位乘以2位的位类型，对应的名称为Bits4x2
  _(c10::bits8, Bits8) /* 21 */                          \  // 8位的位类型，对应的名称为Bits8
  _(c10::bits16, Bits16) /* 22 */                        \  // 16位的位类型，对应的名称为Bits16
  _(c10::Float8_e5m2, Float8_e5m2) /* 23 */              \  // 8位乘以5位减2的浮点数类型，对应的名称为Float8_e5m2
  _(c10::Float8_e4m3fn, Float8_e4m3fn) /* 24 */          \  // 8位乘以4位减3个隐含位的浮点数类型，对应的名称为Float8_e4m3fn
  _(c10::Float8_e5m2fnuz, Float8_e5m2fnuz) /* 25 */      \  // 8位乘以5位减2个隐含位且未指定标记的浮点数类型，对应的名称为Float8_e5m2fnuz
  _(c10::Float8_e4m3fnuz, Float8_e4m3fnuz) /* 26 */      \  // 8位乘以4位减3个隐含位且未指定标记的浮点数类型，对应的名称为Float8_e4m3fnuz
  _(uint16_t, UInt16) /* 27 */                           \  // 无符号16位整数类型，对应的名称为UInt16
  _(uint32_t, UInt32) /* 28 */                           \  // 无符号32位整数类型，对应的名称为UInt32
  _(uint64_t, UInt64) /* 29 */                           \  // 无符号64位整数类型，对应的名称为UInt64
  _(c10::dummy_uint1_7_t<1>, UInt1) /* 30 */             \  // 1位到7位的虚拟无符号整数类型，对应的名称为UInt1
  _(c10::dummy_uint1_7_t<2>, UInt2) /* 31 */             \  // 2位到7位的虚拟无符号整数类型，对应的名称为UInt2
  _(c10::dummy_uint1_7_t<3>, UInt3) /* 32 */             \  // 3位到7位的虚拟无符号整数类型，对应的名称为UInt3
  _(c10::dummy_uint1_7_t<4>, UInt4) /* 33 */             \  // 4位到7位的虚拟无符号整数类型，对应的名称为UInt4
  _(c10::dummy_uint1_7_t<5>, UInt5) /* 34 */             \  // 5位到7位的虚拟无符号整数类型，对应的名称为UInt5
  _(c10::dummy_uint1_7_t<6>, UInt6) /* 35 */             \  // 6位到7位的虚拟无符号整数类型，对应的名称为UInt6
  _(c10::dummy_uint1_7_t<7>, UInt7) /* 36 */             \  // 7位的虚拟无符号整数类型，对应的名称为UInt7
// 定义一个宏，展开为多个类型和对应的描述符，用于标识所有标量类型（不包括复数类型和特殊半精度类型）
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(_) \
  _(uint8_t, Byte)                                                      \
  _(int8_t, Char)                                                       \
  _(int16_t, Short)                                                     \
  _(int, Int)                                                           \
  _(int64_t, Long)                                                      \
  _(at::Half, Half)                                                     \
  _(float, Float)                                                       \
  _(double, Double)                                                     \
  _(c10::complex<float>, ComplexFloat)                                  \
  _(c10::complex<double>, ComplexDouble)                                \
  _(bool, Bool)                                                         \
  _(at::BFloat16, BFloat16)                                             \
  _(at::Float8_e5m2, Float8_e5m2)                                       \
  _(at::Float8_e4m3fn, Float8_e4m3fn)

// 定义一个宏，展开为多个类型和对应的描述符，用于标识所有标量类型（包括复数类型和特殊半精度类型）
// 该宏控制许多 C++ API，包括 Scalar 的构造函数以及 Tensor 上的 data() 和 item() 访问器
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
  _(uint8_t, Byte)                             \
  _(int8_t, Char)                              \
  _(int16_t, Short)                            \
  _(int, Int)                                  \
  _(int64_t, Long)                             \
  _(at::Half, Half)                            \
  _(float, Float)                              \
  _(double, Double)                            \
  _(c10::complex<c10::Half>, ComplexHalf)      \
  _(c10::complex<float>, ComplexFloat)         \
  _(c10::complex<double>, ComplexDouble)       \
  _(bool, Bool)                                \
  _(at::BFloat16, BFloat16)                    \
  _(at::Float8_e5m2, Float8_e5m2)              \
  _(at::Float8_e4m3fn, Float8_e4m3fn)          \
  _(at::Float8_e5m2fnuz, Float8_e5m2fnuz)      \
  _(at::Float8_e4m3fnuz, Float8_e4m3fnuz)

// 枚举类型 ScalarType 定义，包含所有标量类型的枚举值，以及 Undefined 和 NumOptions 两个特殊值
enum class ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
  Undefined,
  NumOptions
};

// 常量表达式，表示标量类型的数量
constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);

namespace impl {

// 用于将 ScalarType 映射到 C++ 类型的模板
template <c10::ScalarType N>
struct ScalarTypeToCPPType;

// 特化模板 ScalarTypeToCPPType，将特定的 ScalarType 映射到对应的 C++ 类型
#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type)                \
  template <>                                                                \
  struct ScalarTypeToCPPType<c10::ScalarType::scalar_type> {                 \
    using type = cpp_type;                                                   \
                                                                             \
    /* This is a workaround for the CUDA bug which prevents */
    /* ::detail::ScalarTypeToCType<T>::type being used directly due to */    \
    /* ambiguous reference which can't to be resolved. For some reason it */ \
    /* can't pick between at::detail and at::cuda::detail. */                \
    /* For repro example, please see: */                                     \
    /* https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba */    \
    /* TODO: remove once the bug is fixed. */                                \
    static type t;                                                           \
  };
// 定义宏 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS，展开为一系列类型映射模板特化
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)

// 取消宏 SPECIALIZE_ScalarTypeToCPPType 的定义
#undef SPECIALIZE_ScalarTypeToCPPType

// 定义模板别名 ScalarTypeToCPPTypeT，用于从标量类型到对应的 C++ 类型的映射
template <c10::ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

// 实现 C++ 类型到标量类型的映射结构体模板，使用偏特化来实现具体的映射
} // namespace impl

// 定义宏 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS，用于展开一系列 C++ 类型到标量类型的特化
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_CppTypeToScalarType)

// 取消宏 SPECIALIZE_CppTypeToScalarType 的定义
#undef SPECIALIZE_CppTypeToScalarType

// 宏定义 AT_FORALL_INT_TYPES，展开为一系列整数类型及其对应的标量类型
#define AT_FORALL_INT_TYPES(_) \
  _(uint8_t, Byte)             \
  _(int8_t, Char)              \
  _(int16_t, Short)            \
  _(int, Int)                  \
  _(int64_t, Long)

// 宏定义 AT_FORALL_SCALAR_TYPES，展开为一系列标量类型及其对应的标量类型
#define AT_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte)                \
  _(int8_t, Char)                 \
  _(int16_t, Short)               \
  _(int, Int)                     \
  _(int64_t, Long)                \
  _(float, Float)                 \
  _(double, Double)

// 宏定义 AT_FORALL_SCALAR_TYPES_AND，展开为一系列标量类型及其对应的标量类型，并包括额外的自定义标量类型
#define AT_FORALL_SCALAR_TYPES_AND(SCALARTYPE, _) \
  _(uint8_t, Byte)                                \
  _(int8_t, Char)                                 \
  _(int16_t, Short)                               \
  _(int, Int)                                     \
  _(int64_t, Long)                                \
  _(float, Float)                                 \
  _(double, Double)                               \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE>::t),  \
    SCALARTYPE)

// 宏定义 AT_FORALL_SCALAR_TYPES_AND2，展开为一系列标量类型及其对应的标量类型，并包括额外的两个自定义标量类型
#define AT_FORALL_SCALAR_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, _) \
  _(uint8_t, Byte)                                               \
  _(int8_t, Char)                                                \
  _(int16_t, Short)                                              \
  _(int, Int)                                                    \
  _(int64_t, Long)                                               \
  _(float, Float)                                                \
  _(double, Double)                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \
             ::c10::ScalarType::SCALARTYPE1>::t),                \
    SCALARTYPE1)                                                 \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \
             ::c10::ScalarType::SCALARTYPE2>::t),                \
    SCALARTYPE2)
    SCALARTYPE1)                                                 \

在宏定义中，插入 `SCALARTYPE1` 的文本，并且在行尾添加反斜杠，用于连接下一行的代码。


  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \

通过 `decltype` 关键字获取 `::c10::impl::ScalarTypeToCPPType< ::c10::ScalarType::SCALARTYPE2 >::t` 的类型，并在宏中进行使用。


             ::c10::ScalarType::SCALARTYPE2>::t),                \

使用 `SCALARTYPE2` 替换模板中的类型参数，并获取其对应的 `t` 类型。


    SCALARTYPE2)

在宏定义中插入 `SCALARTYPE2` 的文本。

这段代码是一个宏定义，用于根据两个标量类型 `SCALARTYPE1` 和 `SCALARTYPE2` 生成相应的代码。
# 定义宏，用于遍历所有标量类型和额外的三种标量类型
#define AT_FORALL_SCALAR_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, _) \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE1>::t),                             \
    SCALARTYPE1)                                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE2>::t),                             \
    SCALARTYPE2)                                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE3>::t),                             \
    SCALARTYPE3)

# 定义宏，用于遍历所有标量类型和额外的七种标量类型
#define AT_FORALL_SCALAR_TYPES_AND7(              \
    SCALARTYPE1,                                  \
    SCALARTYPE2,                                  \
    SCALARTYPE3,                                  \
    SCALARTYPE4,                                  \
    SCALARTYPE5,                                  \
    SCALARTYPE6,                                  \
    SCALARTYPE7,                                  \
    _)                                            \
  _(uint8_t, Byte)                                \
  _(int8_t, Char)                                 \
  _(int16_t, Short)                               \
  _(int, Int)                                     \
  _(int64_t, Long)                                \
  _(float, Float)                                 \
  _(double, Double)                               \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE1>::t), \
    SCALARTYPE1)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE2>::t), \
    SCALARTYPE2)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE3>::t), \
    SCALARTYPE3)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE4>::t), \
    SCALARTYPE4)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE5>::t), \
    SCALARTYPE5)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE6>::t), \
    SCALARTYPE6)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE7>::t), \
    SCALARTYPE7)
    SCALARTYPE5)                                  \  # 宏定义中的一部分，以 SCALARTYPE5 结尾
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \  # 使用 decltype 获取 SCALARTYPE6 对应的 C++ 类型
             ::c10::ScalarType::SCALARTYPE6>::t), \  # 根据 SCALARTYPE6 查找其对应的 c10::ScalarType，并获取其 C++ 类型
    SCALARTYPE6)                                  \  # 宏定义中的一部分，以 SCALARTYPE6 结尾
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \  # 使用 decltype 获取 SCALARTYPE7 对应的 C++ 类型
             ::c10::ScalarType::SCALARTYPE7>::t), \  # 根据 SCALARTYPE7 查找其对应的 c10::ScalarType，并获取其 C++ 类型
    SCALARTYPE7)                                  \  # 宏定义中的一部分，以 SCALARTYPE7 结尾
#define AT_FORALL_QINT_TYPES(_) \
  _(c10::qint8, QInt8)          \  // 定义宏，展开为一系列 qint8 相关声明
  _(c10::quint8, QUInt8)        \
  _(c10::qint32, QInt32)        \
  _(c10::quint4x2, QUInt4x2)    \
  _(c10::quint2x4, QUInt2x4)

#define AT_FORALL_COMPLEX_TYPES(_)     \  // 定义宏，展开为一系列 complex 类型相关声明
  _(c10::complex<float>, ComplexFloat) \
  _(c10::complex<double>, ComplexDouble)

#define DEFINE_CONSTANT(_, name) \  // 定义宏，用于声明常量
  constexpr ScalarType k##name = ScalarType::name;

// NOLINTNEXTLINE(clang-diagnostic-unused-const-variable)
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)  // 展开所有标量类型的常量声明

#undef DEFINE_CONSTANT  // 取消之前的常量声明宏定义

inline const char* toString(ScalarType t) {  // 返回标量类型对应的字符串表示
#define DEFINE_CASE(_, name) \  // 定义宏，返回不同标量类型的名称
  case ScalarType::name:     \
    return #name;

  switch (t) {  // 根据标量类型选择不同的返回值
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)  // 展开所有标量类型的名称定义
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE  // 取消之前的标量类型名称定义宏

}

inline size_t elementSize(ScalarType t) {  // 返回标量类型的元素大小
#define CASE_ELEMENTSIZE_CASE(ctype, name) \  // 定义宏，返回不同标量类型的元素大小
  case ScalarType::name:                   \
    return sizeof(ctype);

  switch (t) {  // 根据标量类型选择不同的元素大小
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CASE_ELEMENTSIZE_CASE)  // 展开所有标量类型的元素大小定义
    default:
      TORCH_CHECK(false, "Unknown ScalarType");  // 未知的标量类型错误检查
  }
#undef CASE_ELEMENTSIZE_CASE  // 取消之前的标量类型元素大小定义宏
}

inline bool isIntegralType(ScalarType t, bool includeBool) {  // 检查标量类型是否为整数类型
  bool isIntegral =
      (t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
       t == ScalarType::Long || t == ScalarType::Short ||
       t == ScalarType::UInt16 || t == ScalarType::UInt32 ||
       t == ScalarType::UInt64);

  return isIntegral || (includeBool && t == ScalarType::Bool);  // 返回是否为整数类型或者包含布尔类型的结果
}

C10_DEPRECATED_MESSAGE(
    "isIntegralType is deprecated. Please use the overload with 'includeBool' parameter instead.")
inline bool isIntegralType(ScalarType t) {  // 已弃用的函数，用于检查标量类型是否为整数类型
  return isIntegralType(t, /*includeBool=*/false);  // 调用包含布尔类型的重载函数
}

inline bool isFloat8Type(ScalarType t) {  // 检查标量类型是否为 Float8 类型
  return t == ScalarType::Float8_e5m2 || t == ScalarType::Float8_e5m2fnuz ||
      t == ScalarType::Float8_e4m3fn || t == ScalarType::Float8_e4m3fnuz;  // 返回检查结果
}

inline bool isReducedFloatingType(ScalarType t) {  // 检查标量类型是否为减少的浮点数类型
  return t == ScalarType::Half || t == ScalarType::BFloat16 || isFloat8Type(t);  // 返回检查结果
}

inline bool isFloatingType(ScalarType t) {  // 检查标量类型是否为浮点数类型
  return t == ScalarType::Double || t == ScalarType::Float ||
      isReducedFloatingType(t);  // 返回检查结果
}

inline bool isComplexType(ScalarType t) {  // 检查标量类型是否为复数类型
  return (
      t == ScalarType::ComplexHalf || t == ScalarType::ComplexFloat ||
      t == ScalarType::ComplexDouble);  // 返回检查结果
}

inline bool isQIntType(ScalarType t) {  // 检查标量类型是否为 QInt 类型
  // Don't forget to extend this when adding new QInt types
  return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
      t == ScalarType::QInt32 || t == ScalarType::QUInt4x2 ||
      t == ScalarType::QUInt2x4;  // 返回检查结果
}

inline bool isBitsType(ScalarType t) {  // 检查标量类型是否为 Bits 类型
  return t == ScalarType::Bits1x8 || t == ScalarType::Bits2x4 ||
      t == ScalarType::Bits4x2 || t == ScalarType::Bits8 ||
      t == ScalarType::Bits16;  // 返回检查结果
}
inline bool isBarebonesUnsignedType(ScalarType t) {
  // 检查给定的标量类型是否为裸露的无符号类型，包括各种位宽的无符号整数
  return t == ScalarType::UInt1 || t == ScalarType::UInt2 ||
      t == ScalarType::UInt3 || t == ScalarType::UInt4 ||
      t == ScalarType::UInt5 || t == ScalarType::UInt6 ||
      t == ScalarType::UInt7 || t == ScalarType::UInt16 ||
      t == ScalarType::UInt32 || t == ScalarType::UInt64;
}

inline ScalarType toQIntType(ScalarType t) {
  // 将标量类型转换为对应的量化整数类型，例如将字节类型转换为8位无符号量化整数类型
  switch (t) {
    case ScalarType::Byte:
      return ScalarType::QUInt8;
    case ScalarType::Char:
      return ScalarType::QInt8;
    case ScalarType::Int:
      return ScalarType::QInt32;
    default:
      return t;
  }
}

inline ScalarType toUnderlying(ScalarType t) {
  // 将量化整数类型转换为其底层类型，例如将8位无符号量化整数类型转换为字节类型
  switch (t) {
    case ScalarType::QUInt8:
    case ScalarType::QUInt4x2:
      [[fallthrough]];  // 继续执行下一个 case 语句
    case ScalarType::QUInt2x4:
      return ScalarType::Byte;
    case ScalarType::QInt8:
      return ScalarType::Char;
    case ScalarType::QInt32:
      return ScalarType::Int;
    default:
      return t;
  }
}

inline bool isSignedType(ScalarType t) {
#define CASE_ISSIGNED(name)     \
  case ScalarType::name:        \
    return std::numeric_limits< \
        ::c10::impl::ScalarTypeToCPPTypeT<ScalarType::name>>::is_signed;

  switch (t) {
    case ScalarType::QInt8:
    case ScalarType::QUInt8:
    case ScalarType::QInt32:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
      // 对于量化类型，抛出错误，不支持有符号类型判断
      TORCH_CHECK(false, "isSignedType not supported for quantized types");
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Bits16:
      // 对于位类型，抛出错误，未定义行为
      TORCH_CHECK(false, "Bits types are undefined");
      // 判断其他类型是否有符号，使用宏展开 CASE_ISSIGNED(name) 定义的逻辑
      CASE_ISSIGNED(UInt16);
      CASE_ISSIGNED(UInt32);
      CASE_ISSIGNED(UInt64);
      CASE_ISSIGNED(BFloat16);
      CASE_ISSIGNED(Float8_e5m2);
      CASE_ISSIGNED(Float8_e5m2fnuz);
      CASE_ISSIGNED(Float8_e4m3fn);
      CASE_ISSIGNED(Float8_e4m3fnuz);
      CASE_ISSIGNED(Byte);
      CASE_ISSIGNED(Char);
      CASE_ISSIGNED(Short);
      CASE_ISSIGNED(Int);
      CASE_ISSIGNED(Long);
      CASE_ISSIGNED(Half);
      CASE_ISSIGNED(Float);
      CASE_ISSIGNED(Double);
      CASE_ISSIGNED(ComplexHalf);
      CASE_ISSIGNED(ComplexFloat);
      CASE_ISSIGNED(ComplexDouble);
      CASE_ISSIGNED(Bool);
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
      return true;
    case ScalarType::Undefined:
    case ScalarType::NumOptions:
      break;
      // 不要在这里添加 default，而是在每个新条目的 case 语句中定义行为，避免 -Wswitch-enum 引发警告
  }
  // 对于未知的标量类型，抛出错误
  TORCH_CHECK(false, "Unknown ScalarType ", t);
#undef CASE_ISSIGNED
}

inline bool isUnderlying(ScalarType type, ScalarType qtype) {
  // 检查给定的标量类型是否与给定量化类型的底层类型相匹配
  return type == toUnderlying(qtype);
}

inline ScalarType toRealValueType(ScalarType t) {
  // 将标量类型转换为实际值类型，这里只展示了部分的 case 语句
  switch (t) {
    # 根据输入的复数类型，返回其对应的实部类型
    case ScalarType::ComplexHalf:
      # 如果输入类型是复数半精度，则返回实部类型为半精度
      return ScalarType::Half;
    case ScalarType::ComplexFloat:
      # 如果输入类型是复数单精度，则返回实部类型为单精度浮点数
      return ScalarType::Float;
    case ScalarType::ComplexDouble:
      # 如果输入类型是复数双精度，则返回实部类型为双精度浮点数
      return ScalarType::Double;
    default:
      # 如果输入类型不是复数类型，则直接返回输入类型本身
      return t;
  }
}

// 定义一个函数，将给定的标量类型转换为复数类型
inline ScalarType toComplexType(ScalarType t) {
  // 根据输入的标量类型进行选择
  switch (t) {
    case ScalarType::BFloat16:
      // BFloat16 的范围等同于 Float，
      // 因此将其映射为 ComplexFloat。
      return ScalarType::ComplexFloat;
    case ScalarType::Half:
      return ScalarType::ComplexHalf;
    case ScalarType::Float:
      return ScalarType::ComplexFloat;
    case ScalarType::Double:
      return ScalarType::ComplexDouble;
    case ScalarType::ComplexHalf:
      return ScalarType::ComplexHalf;
    case ScalarType::ComplexFloat:
      return ScalarType::ComplexFloat;
    case ScalarType::ComplexDouble:
      return ScalarType::ComplexDouble;
    default:
      // 如果标量类型未知，则抛出错误信息
      TORCH_CHECK(false, "Unknown Complex ScalarType for ", t);
  }
}

// 查看 tensor_attributes.rst 以获取详细的解释和示例
// 关于类型转换规则。
inline bool canCast(const ScalarType from, const ScalarType to) {
  // 禁止复数类型转换为非复数类型，例如 float_tensor *= complex 被禁止。
  if (isComplexType(from) && !isComplexType(to)) {
    return false;
  }
  // 禁止浮点类型转换为整数类型，例如 int_tensor *= float 被禁止。
  if (isFloatingType(from) && isIntegralType(to, false)) {
    return false;
  }

  // 将 bool 视为不同的“类别”，以保持与类型提升规则的一致性
  // （例如 `bool_tensor + 5 -> int64_tensor`）。如果 `5` 与 `bool_tensor` 在同一类别中，
  // 我们将不进行提升。不同类别意味着 `bool_tensor += 5` 被禁止。
  //
  // 注意：numpy 将“unsigned”视为一个类别，以获得所需的 `bool_tensor + 5 -> int64_tensor` 行为。
  // 我们不这样做，因为：
  // * 我们不希望检查 Scalars 的运行时符号而产生性能损失。
  // * `uint8_tensor + 5 -> int64_tensor` 是不期望的。
  if (from != ScalarType::Bool && to == ScalarType::Bool) {
    return false;
  }
  return true;
}

// 用于提升两个标量类型的函数
C10_API ScalarType promoteTypes(ScalarType a, ScalarType b);

// 重载运算符<<，用于将标量类型输出到流中
inline std::ostream& operator<<(
    std::ostream& stream,
    at::ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

// 返回一对字符串，分别表示每种数据类型的名称
// 返回的对是 (name, legacy_name_if_applicable)
C10_API std::pair<std::string, std::string> getDtypeNames(
    c10::ScalarType scalarType);

// 返回一个映射，将字符串名称映射到数据类型
C10_API const std::unordered_map<std::string, ScalarType>& getStringToDtypeMap();

} // namespace c10
```