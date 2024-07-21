# `.\pytorch\aten\src\ATen\Dispatch.h`

```
#pragma once

#include <ATen/core/DeprecatedTypeProperties.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/complex.h>
#include <c10/util/string_view.h>

#ifdef __CUDACC__
#include <cuda.h> // For CUDA_VERSION
#endif

#ifdef TEMPLATE_SELECTIVE_BUILD
#include <ATen/selected_mobile_ops.h>
#else
namespace at {
/**
 * The method should_include_kernel_dtype() returns true/false
 * based on whether the switching code for a specific dtype should be
 * included based on build time constants generated from tracing model
 * execution. This method will be implmeneted via code-generation and
 * included in this file when code-gen is ready.
 */
inline constexpr bool should_include_kernel_dtype(
    const char* /*kernel_tag_str*/,
    at::ScalarType /*scalar_type*/
) {
  return true;
}
} // namespace at
#endif

/**
 * In the Facebook internal build (using BUCK), this macro is enabled by
 * passing in -c pt.enable_record_kernel_dtype=1 when building the tracer
 * binary.
 */
#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
namespace at {
namespace detail {
TORCH_API void record_kernel_function_dtype(std::string name);
}
} // namespace at

/**
 * Macro RECORD_KERNEL_FUNCTION_DTYPE(NAME, enum_type) records the kernel
 * function's dtype for tracing purposes, if enabled.
 */
#define RECORD_KERNEL_FUNCTION_DTYPE(NAME, enum_type) \
  at::detail::record_kernel_function_dtype(           \
      std::string(NAME) + "$" + toString(enum_type));
#else
/**
 * Macro RECORD_KERNEL_FUNCTION_DTYPE(NAME, enum_type) is empty if
 * ENABLE_RECORD_KERNEL_FUNCTION_DTYPE is not defined.
 */
#define RECORD_KERNEL_FUNCTION_DTYPE(NAME, enum_type)
#endif

/**
 * AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type) verifies if a specific dtype
 * (enum_type) is included based on build-time constants. If not included,
 * it raises an error.
 */
#define AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type)   \
  do {                                                \
    if constexpr (!at::should_include_kernel_dtype(   \
                      at_dispatch_name, enum_type)) { \
      AT_ERROR(                                       \
          "dtype '",                                  \
          toString(enum_type),                        \
          "' not selected for kernel tag ",           \
          at_dispatch_name);                          \
    }                                                 \
  } while (0)

/**
 * AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...) defines a case for
 * type dispatch based on enum_type. It checks if the dtype should be included,
 * defines a type alias using HINT, and executes the provided code (__VA_ARGS__).
 */
#define AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...)           \
  case enum_type: {                                                     \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);                        \
    using HINT C10_UNUSED = c10::impl::ScalarTypeToCPPTypeT<enum_type>; \
    return __VA_ARGS__();                                               \
  }

/**
 * AT_DISPATCH_CASE(enum_type, ...) dispatches based on enum_type,
 * using scalar_t as a type hint for the dispatch, and executes __VA_ARGS__.
 */
#define AT_DISPATCH_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

/**
 * AT_DISPATCH_CASE_QINT(enum_type, scalar_type, ...) dispatches based on enum_type,
 * defines scalar_t as scalar_type, and executes __VA_ARGS__.
 */
#define AT_DISPATCH_CASE_QINT(enum_type, scalar_type, ...)            \
  case enum_type: {                                                   \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);                      \
    using scalar_t = scalar_type;                                     \
    using underlying_t C10_UNUSED = typename scalar_t::underlying;    \
    const auto& SCALAR_TYPE C10_UNUSED = enum_type;                   \
    // 使用 C10_UNUSED 宏将 enum_type 转换为其底层类型，并赋值给 UNDERLYING_TYPE 常量引用
    const auto& UNDERLYING_TYPE C10_UNUSED = toUnderlying(enum_type); \
    // 返回使用宏展开的参数列表 __VA_ARGS__ 所表示的值
    return __VA_ARGS__();                                             \
  }
// 定义一个宏，用于在AT_DISPATCH宏中处理整数类型和定量类型的特殊情况
#define AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                           \
    enum_type, scalar_type, bitwidth, qmin, qmax, ...)                \
  case enum_type: {                                                   \
    // 检查当前枚举类型是否被选择性地构建
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);                      \
    // 定义标量类型为传入的标量类型
    using scalar_t = scalar_type;                                     \
    // 定义底层类型为标量类型的底层类型
    using underlying_t C10_UNUSED = typename scalar_t::underlying;    \
    // 定义常量SCALAR_TYPE并将其与当前枚举类型关联
    const auto& SCALAR_TYPE C10_UNUSED = enum_type;                   \
    // 定义常量UNDERLYING_TYPE并将其与当前枚举类型的底层类型的值关联
    const auto& UNDERLYING_TYPE C10_UNUSED = toUnderlying(enum_type); \
    // 定义并初始化bit_width为传入的位宽
    C10_UNUSED int bit_width = bitwidth;                              \
    // 定义并初始化quant_min为传入的最小量化值
    C10_UNUSED int64_t quant_min = qmin;                              \
    // 定义并初始化quant_max为传入的最大量化值
    C10_UNUSED int64_t quant_max = qmax;                              \
    // 调用并返回传入的可变参数宏
    return __VA_ARGS__();                                             \
  }

namespace detail {

// 返回传入标量类型的值
inline at::ScalarType scalar_type(at::ScalarType s) {
  return s;
}

// 已弃用的消息，不再建议在AT_DISPATCH宏中传入at::DeprecatedTypeProperties，建议传入at::ScalarType
C10_DEPRECATED_MESSAGE(
    "passing at::DeprecatedTypeProperties to an AT_DISPATCH macro is deprecated, "
    "pass an at::ScalarType instead")
inline at::ScalarType scalar_type(const at::DeprecatedTypeProperties& t) {
  return t.scalarType();
}

// 已弃用的消息，AT_DISPATCH_ALL_TYPES_AND_HALF已弃用，建议使用AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, ...)代替
C10_DEPRECATED_MESSAGE(
    "AT_DISPATCH_ALL_TYPES_AND_HALF is deprecated, "
    "use AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, ...) instead")
inline void deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF() {}

// 已弃用的消息，AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX已弃用，建议使用AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, ...)代替
C10_DEPRECATED_MESSAGE(
    "AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX is deprecated, "
    "use AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, ...) "
    "instead")
inline void deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX() {}

} // namespace detail

// AT_DISPATCH_*系列宏提供了一种方便地生成在PyTorch中所有我们关心的数据类型上特例化的方法。
// 我们称之为“dispatch”，因为我们正在“分派”到正确的dtype特定的内核。
//
// 标准用法如下：
//
//      AT_DISPATCH_ALL_TYPES(self.scalar_type(), "op_name", [&] {
//          // 在这里编写您的代码，现在'scalar_t'被定义为相关的dtype
//      });
//
// 此宏有许多变体，因此重要的是准确理解您想要实例化的哪些dtypes，以及“默认”集合是什么。
//
// 被实例化的默认dtype集合（例如，通过AT_DISPATCH_ALL_TYPES）包括浮点类型（float，double）和整数类型
// （int32_t，int64_t，int16_t，int8_t，uint8_t），但不包括布尔值（bool）、半精度浮点数（Half）或复数
// （c10::complex<float>，c10::complex<double>）。这种“切割”在一定程度上是历史性的（默认类型是TH历史上支持的类型），
// 但也反映了非默认类型的“行为较差”（布尔值不是模2的整数，半精度操作大致相等
//
/*
// don't exist on CPU, complex numbers are an experimental application).
//
// Here are the questions you should generally ask to decide which
// dispatch you want:
//
// 1. Is this an integral or floating point specific operation?
//    (If so, you'll want one of the FLOATING or INTEGRAL macros.)
//
// 2. Should half be supported?  (If you're on CPU, the answer is almost
//    definitely no.  If you do want support, use one of the AND_HALF
//    macros)
//
// Much rarer situations:
//
// 3. Should bool be supported?  (You often have to write your kernel
//    differently if arithmetic operations are involved.)  If so,
//    Use AT_DISPATCH_ALL_TYPES_AND along with ScalarType::Bool
//
// 4. Should complex be supported?  The answer is almost always no,
//    unless you are working on "generic" code that should work on
//    all dtypes.
//
// Parameters:
// -----------
//
// 1. The NAME argument is a "tag" that is used to trace and then
//    conditionally compile fragments of the case statements such
//    that the kernel functions are specialized only for the dtypes
//    that are needed. The NAME parameter *must* be a build time
//    const char* (can't be std::string, etc...)
//
// Please ensure that the NAME is unique for every implementation
// or you run the risk of over-including code for the kernel
// functions. There is no risk of missing out on any code, so
// it's mostly a risk of a Type-2 error, and not a Type-1 error.
//
// Switch-like syntax:
// -------------------
// There is also a switch-case like syntax which is useful if a kernel
// needs to be specialized for particular scalar types
//
//      AT_DISPATCH_SWITCH(self.scalar_type(), "op_name",
//          AT_DISPATCH_CASE_INTEGRAL_TYPES([&] {
//            op_integral<scalar_t>(iter);
//          })
//          AT_DISPATCH_CASE_FLOATING_TYPES([&] {
//            op_floating<scalar_t>(iter);
//          })
//          AT_DISPATCH_CASE(kBool, [&] {
//            op_bool(iter);
//          })
//      );
//
// For each AT_DISPATCH_FOO macro, there is a corresponding
// AT_DISPATCH_CASE_FOO macro which can be used inside of an
// AT_DISPATCH_SWITCH block.

// NB: the the_type variable is not used, but we have kept it for
// backwards compatibility.  It's probably not used by anyone though;
// but we're just being safe (and it doesn't hurt.)  Note we must
// use it to shut up warnings about unused store.
*/

#define AT_DISPATCH_SWITCH(TYPE, NAME, ...)                                 \
  [&] {                                                                     \
    // 引用类型 TYPE
    const auto& the_type = TYPE;                                            \
    // 定义常量字符串指针 at_dispatch_name 为 NAME
    constexpr const char* at_dispatch_name = NAME;                          \
    /* 不要再次使用 TYPE，以防它是昂贵或具有副作用的操作 */                   \
    // 获取 TYPE 的标量类型并赋值给 _st
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    // 记录使用 at_dispatch_name 和 _st 的内核函数的数据类型
    RECORD_KERNEL_FUNCTION_DTYPE(at_dispatch_name, _st);                    \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        AT_ERROR(                                                           \
            '"',                                                            \
            at_dispatch_name,                                               \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }()



// 使用宏定义的 switch 语句进行多条件分支判断，_st 是判断条件
switch (_st) {                                                          \
  // __VA_ARGS__ 在此处展开，表示各个 case 分支的具体实现
  __VA_ARGS__                                                           \
  // 默认分支：如果 _st 不匹配任何已知条件，则执行下面的默认操作
  default:                                                              \
    // 调用 AT_ERROR 宏，输出错误信息，指出不支持当前 _st 的操作
    AT_ERROR(                                                           \
        '"',                                                            \
        at_dispatch_name,                                               \
        "\" not implemented for '",                                     \
        toString(_st),                                                  \
        "'");                                                           \
}()


这段代码是一个宏定义，实现了类似于 switch-case 的多条件分支判断，当 _st 不匹配任何已知条件时，会调用 AT_ERROR 宏输出相应的错误信息。
# 定义一个宏，用于根据浮点类型分发任务，仅限于双精度和单精度浮点数
#define AT_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

# 定义一个宏，用于根据给定的浮点类型分发任务，包括双精度和单精度浮点数
#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

# 定义一个宏，用于根据浮点类型和半精度浮点数分发任务，包括双精度、单精度和半精度
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

# 定义一个宏，用于根据给定的浮点类型和半精度浮点数分发任务，包括双精度、单精度和半精度
#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(__VA_ARGS__))

# 定义一个宏，用于根据减少的浮点类型分发任务，仅限于半精度和BFloat16类型
#define AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(...)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

# 定义一个宏，用于根据给定的减少的浮点类型分发任务，仅限于半精度和BFloat16类型
#define AT_DISPATCH_REDUCED_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE, NAME, AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(__VA_ARGS__))

# 定义一个宏，用于根据浮点类型和指定的标量类型分发任务
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

# 定义一个宏，用于根据给定的浮点类型和指定的标量类型分发任务
#define AT_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, __VA_ARGS__))

# 定义一个宏，用于根据两种浮点类型和指定的标量类型分发任务
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

# 定义一个宏，用于根据两种浮点类型和指定的标量类型分发任务
#define AT_DISPATCH_FLOATING_TYPES_AND2(       \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                          \
      TYPE,                                    \
      NAME,                                    \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(    \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

# 定义一个宏，用于根据三种浮点类型和指定的标量类型分发任务
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND3(   \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)  \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)    \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)    \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

# 定义一个宏，用于根据三种浮点类型和指定的标量类型分发任务
#define AT_DISPATCH_FLOATING_TYPES_AND3(                    \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND3(                 \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND4(                \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)               \  # 调用 AT_DISPATCH_CASE_FLOATING_TYPES 宏，处理所有浮点类型参数
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                 \  # 调用 AT_DISPATCH_CASE 宏，处理 SCALARTYPE1 类型的情况
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                 \  # 调用 AT_DISPATCH_CASE 宏，处理 SCALARTYPE2 类型的情况
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                 \  # 调用 AT_DISPATCH_CASE 宏，处理 SCALARTYPE3 类型的情况
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                 \  # 调用 AT_DISPATCH_CASE 宏，处理 SCALARTYPE4 类型的情况

#define AT_DISPATCH_FLOATING_TYPES_AND4(                                 \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \  # 调用 AT_DISPATCH_SWITCH 宏，处理类型和名称
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND4(                              \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__)) \  # 调用 AT_DISPATCH_CASE_FLOATING_TYPES_AND4 处理浮点类型和其他参数

#define AT_DISPATCH_CASE_COMPLEX_TYPES(...)                    \
  AT_DISPATCH_CASE(at::ScalarType::ComplexDouble, __VA_ARGS__) \  # 调用 AT_DISPATCH_CASE 处理复杂类型为 ComplexDouble 的情况
  AT_DISPATCH_CASE(at::ScalarType::ComplexFloat, __VA_ARGS__)   \  # 调用 AT_DISPATCH_CASE 处理复杂类型为 ComplexFloat 的情况

#define AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)) \  # 调用 AT_DISPATCH_SWITCH 处理复杂类型，类型和名称

#define AT_DISPATCH_CASE_COMPLEX_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)               \  # 调用 AT_DISPATCH_CASE_COMPLEX_TYPES 处理复杂类型和其他参数
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)                 \  # 调用 AT_DISPATCH_CASE 处理特定复杂类型

#define AT_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                              \  # 调用 AT_DISPATCH_SWITCH 处理复杂类型，类型和名称
      TYPE, NAME, AT_DISPATCH_CASE_COMPLEX_TYPES_AND(SCALARTYPE, __VA_ARGS__)) \  # 调用 AT_DISPATCH_CASE_COMPLEX_TYPES_AND 处理复杂类型和其他参数

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(...) \  # 调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES 处理浮点和复杂类型
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)           \  # 调用 AT_DISPATCH_CASE_FLOATING_TYPES 处理所有浮点类型
  AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)           \  # 调用 AT_DISPATCH_CASE_COMPLEX_TYPES 处理所有复杂类型

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...) \  # 调用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 处理浮点和复杂类型，类型和名称
  AT_DISPATCH_SWITCH(                                           \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__)) \  # 调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES 处理浮点和复杂类型和其他参数

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1(SCALARTYPE, ...) \  # 调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1 处理浮点和复杂类型和其他参数
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__)                \  # 调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES 处理浮点和复杂类型和其他参数
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)                              \  # 调用 AT_DISPATCH_CASE 处理特定类型的情况

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(    \  # 调用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1 处理浮点和复杂类型和其他参数
    SCALARTYPE, TYPE, NAME, ...)                        \
  AT_DISPATCH_SWITCH(                                   \  # 调用 AT_DISPATCH_SWITCH 处理类型和名称
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1( \  # 调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1 处理浮点和复杂类型和其他参数
          SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2(  \  # 调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2 处理浮点和复杂类型和其他参数
    SCALARTYPE1, SCALARTYPE2, ...)                         \
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__) \  # 调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES 处理浮点和复杂类型和其他参数
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)               \  # 调用 AT_DISPATCH_CASE 处理特定类型的情况
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)               \  # 调用 AT_DISPATCH_CASE 处理特定类型的情况

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(    \  # 调用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 处理浮点和复杂类型和其他参数
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)          \
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2( \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))



    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)          \  # 宏定义的参数，用于传递不定数量的类型和名称参数
  AT_DISPATCH_SWITCH(                                   \  # 调用一个宏，根据类型进行分发和调度
      TYPE,                                             \  # 第一个参数：类型，用于选择不同的分发函数
      NAME,                                             \  # 第二个参数：名称，用于选择不同的分发函数
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2( \  # 调用一个宏，根据类型进行进一步的分发和选择
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))        # 将前面定义的参数传递给另一个宏进行处理
# 定义一个宏，用于处理浮点数和复数类型以及三个额外的类型参数
#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3(  \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...)            \
  # 调用另一个宏处理浮点数和复数类型，并传递所有变长参数
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__) \
  # 分发到处理单一类型的宏，处理第一个类型参数
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)               \
  # 分发到处理单一类型的宏，处理第二个类型参数
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)               \
  # 分发到处理单一类型的宏，处理第三个类型参数
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

# 定义一个宏，用于处理浮点数和复数类型以及三个额外的类型参数，并指定类型和名称
#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(        \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  # 调用一个宏来选择处理的类型
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      # 调用处理三个参数的宏
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3(     \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

# 定义一个宏，用于处理浮点数和复数类型以及四个额外的类型参数
#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4(    \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  # 调用另一个宏处理浮点数和复数类型，并传递所有变长参数
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__)   \
  # 分发到处理单一类型的宏，处理第一个类型参数
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                 \
  # 分发到处理单一类型的宏，处理第二个类型参数
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                 \
  # 分发到处理单一类型的宏，处理第三个类型参数
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                 \
  # 分发到处理单一类型的宏，处理第四个类型参数
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

# 定义一个宏，用于处理浮点数和复数类型以及四个额外的类型参数，并指定类型和名称
#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND4(                     \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  # 调用一个宏来选择处理的类型
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      # 调用处理四个参数的宏
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4(                  \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__))

# 定义一个宏，用于处理浮点数和复数类型以及五个额外的类型参数
#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5(                 \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
  # 调用另一个宏处理浮点数和复数类型，并传递所有变长参数
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__)                \
  # 分发到处理单一类型的宏，处理第一个类型参数
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                              \
  # 分发到处理单一类型的宏，处理第二个类型参数
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                              \
  # 分发到处理单一类型的宏，处理第三个类型参数
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                              \
  # 分发到处理单一类型的宏，处理第四个类型参数
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                              \
  # 分发到处理单一类型的宏，处理第五个类型参数
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)

# 定义一个宏，用于处理浮点数和复数类型以及五个额外的类型参数，并指定类型和名称
#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND5(    \
    SCALARTYPE1,                                        \
    SCALARTYPE2,                                        \
    SCALARTYPE3,                                        \
    SCALARTYPE4,                                        \
    SCALARTYPE5,                                        \
    TYPE,                                               \
    NAME,                                               \
    ...
    ...)                                                \  # 继续宏定义的多行语句，这里是一个行连接符
  AT_DISPATCH_SWITCH(                                   \  # 使用宏定义开始一个分发调度的开关
      TYPE,                                             \  # 宏定义的参数：类型 TYPE
      NAME,                                             \  # 宏定义的参数：名称 NAME
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5( \  # 调用另一个宏定义，处理浮点数和复数类型以及另外5个类型的情况
          SCALARTYPE1,                                  \  # 第一个类型参数 SCALARTYPE1
          SCALARTYPE2,                                  \  # 第二个类型参数 SCALARTYPE2
          SCALARTYPE3,                                  \  # 第三个类型参数 SCALARTYPE3
          SCALARTYPE4,                                  \  # 第四个类型参数 SCALARTYPE4
          SCALARTYPE5,                                  \  # 第五个类型参数 SCALARTYPE5
          __VA_ARGS__))                                  # 可变数量的额外参数，传递给宏定义 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5
# 定义一个宏，用于根据参数中的多种浮点数和复数类型分发调用
#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6(  \
    SCALARTYPE1,                                           \
    SCALARTYPE2,                                           \
    SCALARTYPE3,                                           \
    SCALARTYPE4,                                           \
    SCALARTYPE5,                                           \
    SCALARTYPE6,                                           \
    ...)                                                   \
  # 先调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES 宏，处理所有的浮点数和复数类型
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__) \
  # 分别调用 AT_DISPATCH_CASE 宏，处理每个 SCALARTYPE 对应的类型

# 定义一个宏，用于根据参数中的多种浮点数和复数类型和6个额外类型参数分发调用
#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND6(    \
    SCALARTYPE1,                                        \
    SCALARTYPE2,                                        \
    SCALARTYPE3,                                        \
    SCALARTYPE4,                                        \
    SCALARTYPE5,                                        \
    SCALARTYPE6,                                        \
    TYPE,                                               \
    NAME,                                               \
    ...)                                                \
  # 调用 AT_DISPATCH_SWITCH 宏，根据类型 TYPE 和 NAME 处理分发
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      # 调用 AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6 宏，处理所有的浮点数和复数类型和6个额外类型参数
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6( \
          SCALARTYPE1,                                  \
          SCALARTYPE2,                                  \
          SCALARTYPE3,                                  \
          SCALARTYPE4,                                  \
          SCALARTYPE5,                                  \
          SCALARTYPE6,                                  \
          __VA_ARGS__))

# 定义一个宏，用于根据参数中的多种整数类型分发调用
#define AT_DISPATCH_CASE_INTEGRAL_TYPES(...)          \
  # 调用 AT_DISPATCH_CASE 宏，处理所有的整数类型，分别为 Byte, Char, Int, Long, Short
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)

# 定义一个宏，用于根据参数中的整数类型和额外类型参数分发调用
#define AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  # 调用 AT_DISPATCH_SWITCH 宏，根据类型 TYPE 和 NAME 处理分发
  AT_DISPATCH_SWITCH(TYPE, NAME, \
    # 调用 AT_DISPATCH_CASE_INTEGRAL_TYPES 宏，处理所有的整数类型和额外类型参数
    AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

# 定义一个宏，用于根据参数中的整数类型和一个额外的 SCALARTYPE 类型分发调用
#define AT_DISPATCH_CASE_INTEGRAL_TYPES_AND(SCALARTYPE, ...) \
  # 调用 AT_DISPATCH_CASE_INTEGRAL_TYPES 宏，处理所有的整数类型和额外类型参数
  AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__)               \
  # 调用 AT_DISPATCH_CASE 宏，处理额外的 SCALARTYPE 类型
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)
#define AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_DISPATCH_CASE_INTEGRAL_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES(...)        \
  AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_TYPES(...)                      \
  AT_DISPATCH_CASE_QINT(at::kQInt8, at::qint8, __VA_ARGS__)   \
  AT_DISPATCH_CASE_QINT(at::kQUInt8, at::quint8, __VA_ARGS__) \
  AT_DISPATCH_CASE_QINT(at::kQInt32, at::qint32, __VA_ARGS__)

#define AT_DISPATCH_QINT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_QINT_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_QINT_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                           \
      TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_BYTE_TYPES(...)               \
  AT_DISPATCH_CASE_QINT(at::kQInt8, at::qint8, __VA_ARGS__) \
  AT_DISPATCH_CASE_QINT(at::kQUInt8, at::quint8, __VA_ARGS__)

#define AT_DISPATCH_QINT_BYTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_BYTE_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(...)                     \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQInt8, at::qint8, CHAR_BIT, SCHAR_MIN, SCHAR_MAX, __VA_ARGS__) \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQUInt8, at::quint8, CHAR_BIT, 0, UCHAR_MAX, __VA_ARGS__)       \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQInt32,                                                        \
      at::qint32,                                                         \
      CHAR_BIT * sizeof(int),                                             \
      INT_MIN,                                                            \
      INT_MAX,                                                            \
      __VA_ARGS__)                                                        \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQUInt4x2, at::quint4x2, 4, 0, 15, __VA_ARGS__)                 \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQUInt2x4, at::quint2x4, 2, 0, 3, __VA_ARGS__)



// 宏定义，用于处理整数类型和指定类型的情况
#define AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  // 调用AT_DISPATCH_SWITCH宏，处理类型TYPE，调用NAME，传递参数__VA_ARGS__
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      // 调用AT_DISPATCH_CASE_INTEGRAL_TYPES_AND宏，处理整数类型SCALARTYPE和__VA_ARGS__
      AT_DISPATCH_CASE_INTEGRAL_TYPES_AND(SCALARTYPE, __VA_ARGS__))

// 宏定义，处理所有类型的情况
#define AT_DISPATCH_CASE_ALL_TYPES(...)        \
  // 调用AT_DISPATCH_CASE_INTEGRAL_TYPES宏，处理所有整数类型__VA_ARGS__
  AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__) \
  // 调用AT_DISPATCH_CASE_FLOATING_TYPES宏，处理所有浮点类型__VA_ARGS__

// 宏定义，处理所有类型的情况
#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  // 调用AT_DISPATCH_SWITCH宏，处理类型TYPE，调用NAME，传递参数__VA_ARGS__
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

// 宏定义，处理量化整数类型的情况
#define AT_DISPATCH_CASE_QINT_TYPES(...)                      \
  // 调用AT_DISPATCH_CASE_QINT宏，处理kQInt8和qint8类型，传递参数__VA_ARGS__处理其他参数
  AT_DISPATCH_CASE_QINT(at::kQInt8, at::qint8, __VA_ARGS__)   \
  // 调用AT_DISPATCH_CASE_QINT宏，处理kQUInt8和quint8类型，传递参数__VA_ARGS__处理其他参数
  AT_DISPATCH_CASE_QINT(at::kQUInt8, at::quint8, __VA_ARGS__) \
  // 调用AT_DISPATCH_CASE_QINT宏，处理kQInt32和qint32类型，传递参数__VA_ARGS__处理其他参数

// 宏定义，处理量化整数类型的情况
#define AT_DISPATCH_QINT_TYPES(TYPE, NAME, ...) \
  // 调用AT_DISPATCH_SWITCH宏，处理类型TYPE，调用NAME，传递参数__VA_ARGS__
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES(__VA_ARGS__))

// 宏定义，处理量化整数类型和其他情况的情况
#define AT_DISPATCH_CASE_QINT_TYPES_AND(SCALARTYPE, ...) \
  // 调用AT_DISPATCH_CASE_QINT_TYPES宏，处理__VA_ARGS__中的参数
  AT_DISPATCH_CASE_QINT_TYPES(__VA_ARGS__)               \
  // 调用AT_DISPATCH_CASE宏，处理SCALARTYPE和__VA_ARGS__中的参数

// 宏定义，处理量化整数类型和其他情况的情况
#define AT_DISPATCH_QINT_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  // 调用AT_DISPATCH_SWITCH宏，处理类型TYPE，调用NAME，传递参数__VA_ARGS__
  AT_DISPATCH_SWITCH(                                           \
      TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES_AND(SCALARTYPE, __VA_ARGS__))

// 宏定义，处理量化整数和子字节类型的情况
#define AT_DISPATCH_CASE_QINT_BYTE_TYPES(...)               \
  // 调用AT_DISPATCH_CASE_QINT宏，处理kQInt8和qint8类型，传递参数__VA_ARGS__处理其他参数
  AT_DISPATCH_CASE_QINT(at::kQInt8, at::qint8, __VA_ARGS__) \
  // 调用AT_DISPATCH_CASE_QINT宏，处理kQUInt8和quint8类型，传递参数__VA_ARGS__处理其他参数

// 宏定义，处理量化整数和子字节类型的情况
#define AT_DISPATCH_QINT_BYTE_TYPES(TYPE, NAME, ...) \
  // 调用AT_DISPATCH_SWITCH宏，处理类型TYPE，调用NAME，传递参数__VA_ARGS__
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_BYTE_TYPES(__VA_ARGS__))

// 宏定义，处理量化整数和子字节类型的情况
#define AT_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(...)                     \
  // 调用AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE宏，处理kQInt8和qint8类型，传递参数__VA_ARGS__处理其他参数
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQInt8, at::qint8, CHAR_BIT, SCHAR_MIN, SCHAR_MAX, __VA_ARGS__) \
  // 调用AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE宏，处理kQUInt8和quint8类型，传递参数__VA_ARGS__处理其他参数
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQUInt8, at::quint8, CHAR_BIT, 0, UCHAR_MAX, __VA_ARGS__)       \
  // 调用AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE宏，处理kQInt32和qint32类型，传递参数__VA_ARGS__处理其他参数
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQInt32,                                                        \
      at::qint32,                                                         \
      CHAR_BIT * sizeof(int),                                             \
      INT_MIN,                                                            \
      INT_MAX,                                                            \
      __VA_ARGS__)                                                        \
  // 调用AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE宏，处理kQUInt4x2和quint4x2类型，传递参数__VA_ARGS__处理其他参数
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQUInt4x2, at::quint4x2, 4, 0, 15, __VA_ARGS__)                 \
  // 调用AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE宏，处理kQUInt2x4和quint2x4类型，传递参数__VA_ARGS__处理其他参数
  AT_QINT_SUB
// 定义一个宏，用于根据类型调度处理宏，支持量化整数和字节子类型
#define AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(TYPE, NAME, ...) \
  // 调度宏的开关，根据给定的类型、名称和参数调度处理宏
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(__VA_ARGS__))

// 定义一个宏，用于调度处理所有类型和复数类型
#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(...) \
  // 调度处理所有类型的情况
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)           \
  // 调度处理复数类型的情况
  AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)

// 定义一个宏，用于根据类型调度处理所有类型和复数类型
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...) \
  // 调度宏的开关，根据给定的类型、名称和参数调度处理所有类型和复数类型的情况
  AT_DISPATCH_SWITCH(                                      \
      TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__))

// 定义一个宏，用于调度处理所有类型和给定标量类型
#define AT_DISPATCH_CASE_ALL_TYPES_AND(SCALARTYPE, ...) \
  // 调度处理所有类型的情况
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)               \
  // 调度处理给定标量类型的情况
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

// 定义一个宏，用于根据类型调度处理所有类型和给定标量类型
#define AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  // 调度宏的开关，根据给定的类型、名称和参数调度处理所有类型和给定标量类型的情况
  AT_DISPATCH_SWITCH(                                          \
      TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES_AND(SCALARTYPE, __VA_ARGS__))

// 定义一个宏，用于调度处理所有类型、复数类型和给定标量类型
#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, ...) \
  // 调度处理所有类型和复数类型的情况
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__)               \
  // 调度处理给定标量类型的情况
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

// 定义一个宏，用于根据类型调度处理所有类型、复数类型和给定标量类型
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...) \
  // 调度宏的开关，根据给定的类型、名称和参数调度处理所有类型、复数类型和给定标量类型的情况
  AT_DISPATCH_SWITCH(                                                      \
      TYPE,                                                                \
      NAME,                                                                \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, __VA_ARGS__))

// 定义一个宏，用于调度处理所有类型和两个给定标量类型
#define AT_DISPATCH_CASE_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  // 调度处理所有类型的情况
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)                              \
  // 调度处理两个给定标量类型的情况
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

// 定义一个宏，用于根据类型调度处理所有类型和两个给定标量类型
#define AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  // 调度宏的开关，根据给定的类型、名称和参数调度处理所有类型和两个给定标量类型的情况
  AT_DISPATCH_SWITCH(                                                         \
      TYPE,                                                                   \
      NAME,                                                                   \
      AT_DISPATCH_CASE_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

// 定义一个宏，用于调度处理所有类型、复数类型和两个给定标量类型
#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2(  \
    SCALARTYPE1, SCALARTYPE2, ...)                    \
  // 调度处理所有类型和复数类型的情况
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \
  // 调度处理两个给定标量类型的情况
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

// 定义一个宏，用于根据类型调度处理所有类型、复数类型和两个给定标量类型
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(    \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)     \
  // 调度宏的开关，根据给定的类型、名称和参数调度处理所有类型、复数类型和两个给定标量类型的情况
  AT_DISPATCH_SWITCH(                              \
      TYPE,                                        \
      NAME,                                        \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2( \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \   # 定义多个标量类型参数，以逗号分隔，表示各种数据类型
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)       \   # 调用宏，根据传入的所有类型分发处理各种情况
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)    \   # 调用宏，处理第一种标量类型的情况
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)    \   # 调用宏，处理第二种标量类型的情况
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)          # 调用宏，处理第三种标量类型的情况
#define AT_DISPATCH_ALL_TYPES_AND3(                         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_ALL_TYPES_AND3(                      \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))  

# 宏定义：用于分派和处理包括三种标量类型的所有类型
#define AT_DISPATCH_CASE_ALL_TYPES_AND3(  \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...)       \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

# 宏定义：用于分派和处理包括三种标量类型及复合类型的所有类型
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(             \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND3(          \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

# 宏定义：用于分派和处理包括四种标量类型及复合类型的所有类型
#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__)        \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

# 宏定义：用于分派和处理包括四种标量类型及复合类型的所有类型
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                          \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(                       \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__))

# 宏定义：用于分派和处理包括五种标量类型及复合类型的所有类型
#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5(                      \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__)                     \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)

# 宏定义：用于分派和处理包括五种标量类型及复合类型的所有类型
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5(    \
    SCALARTYPE1,                                   \
    SCALARTYPE2,                                   \
    SCALARTYPE3,                                   \
    SCALARTYPE4,                                   \
    SCALARTYPE5, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5(                       \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, __VA_ARGS__))
    SCALARTYPE5,                                   \  # 定义宏展开时的第五个标量类型参数
    TYPE,                                          \  # 定义宏展开时的类型参数
    NAME,                                          \  # 定义宏展开时的名称参数
    ...)                                           \  # 定义宏展开时的额外可变参数
  AT_DISPATCH_SWITCH(                              \  # 使用宏 AT_DISPATCH_SWITCH 开始一个分发调度的语句块
      TYPE,                                        \  # 将类型参数传递给 AT_DISPATCH_SWITCH 宏
      NAME,                                        \  # 将名称参数传递给 AT_DISPATCH_SWITCH 宏
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5( \  # 使用宏 AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5 开始一个多类型和复数类型的 case 匹配
          SCALARTYPE1,                             \  # 宏 AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5 的第一个标量类型参数
          SCALARTYPE2,                             \  # 宏的第二个标量类型参数
          SCALARTYPE3,                             \  # 宏的第三个标量类型参数
          SCALARTYPE4,                             \  # 宏的第四个标量类型参数
          SCALARTYPE5,                             \  # 宏的第五个标量类型参数
          __VA_ARGS__))                            \  # 宏的额外可变参数
#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND6(  \
    SCALARTYPE1,                                      \  # 定义一个宏，用于处理六个不同类型的标量，以及额外参数
    SCALARTYPE2,                                      \  # 第二个标量类型
    SCALARTYPE3,                                      \  # 第三个标量类型
    SCALARTYPE4,                                      \  # 第四个标量类型
    SCALARTYPE5,                                      \  # 第五个标量类型
    SCALARTYPE6,                                      \  # 第六个标量类型
    ...)                                              \  # 省略参数，表示可接受更多参数
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \  # 调用宏，处理所有类型和复数，并传递剩余参数
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \  # 调用宏，处理第一个标量类型，并传递剩余参数
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)          \  # 同上，处理第二个标量类型
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)          \  # 同上，处理第三个标量类型
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)          \  # 同上，处理第四个标量类型
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)          \  # 同上，处理第五个标量类型
  AT_DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)          \  # 同上，处理第六个标量类型

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(    \  # 定义一个宏，用于处理所有类型和复数以及六个标量类型
    SCALARTYPE1,                                   \  # 第一个标量类型
    SCALARTYPE2,                                   \  # 第二个标量类型
    SCALARTYPE3,                                   \  # 第三个标量类型
    SCALARTYPE4,                                   \  # 第四个标量类型
    SCALARTYPE5,                                   \  # 第五个标量类型
    SCALARTYPE6,                                   \  # 第六个标量类型
    TYPE,                                          \  # 类型参数
    NAME,                                          \  # 名称参数
    ...)                                           \  # 省略参数，表示可接受更多参数
  AT_DISPATCH_SWITCH(                              \  # 调用宏，根据类型参数分发到不同的处理分支
      TYPE,                                        \  # 类型参数
      NAME,                                        \  # 名称参数
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND6( \  # 调用宏，处理六个不同类型的标量以及剩余参数
          SCALARTYPE1,                             \  # 第一个标量类型
          SCALARTYPE2,                             \  # 第二个标量类型
          SCALARTYPE3,                             \  # 第三个标量类型
          SCALARTYPE4,                             \  # 第四个标量类型
          SCALARTYPE5,                             \  # 第五个标量类型
          SCALARTYPE6,                             \  # 第六个标量类型
          __VA_ARGS__))                            \  # 剩余参数

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND7(  \  # 定义一个宏，用于处理七个不同类型的标量，以及额外参数
    SCALARTYPE1,                                      \  # 第一个标量类型
    SCALARTYPE2,                                      \  # 第二个标量类型
    SCALARTYPE3,                                      \  # 第三个标量类型
    SCALARTYPE4,                                      \  # 第四个标量类型
    SCALARTYPE5,                                      \  # 第五个标量类型
    SCALARTYPE6,                                      \  # 第六个标量类型
    SCALARTYPE7,                                      \  # 第七个标量类型
    ...)                                              \  # 省略参数，表示可接受更多参数
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \  # 调用宏，处理所有类型和复数，并传递剩余参数
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \  # 调用宏，处理第一个标量类型，并传递剩余参数
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)          \  # 同上，处理第二个标量类型
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)          \  # 同上，处理第三个标量类型
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)          \  # 同上，处理第四个标量类型
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)          \  # 同上，处理第五个标量类型
  AT_DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)          \  # 同上，处理第六个标量类型
  AT_DISPATCH_CASE(SCALARTYPE7, __VA_ARGS__)          \  # 同上，处理第七个标量类型

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND7(    \  # 定义一个宏，用于处理所有类型和复数以及七个标量类型
    SCALARTYPE1,                                   \  # 第一个标量类型
    ...                                            \  # 省略参数，表示可接受更多参数
    SCALARTYPE2,                                   \  # 定义宏：第二种标量类型
    SCALARTYPE3,                                   \  # 定义宏：第三种标量类型
    SCALARTYPE4,                                   \  # 定义宏：第四种标量类型
    SCALARTYPE5,                                   \  # 定义宏：第五种标量类型
    SCALARTYPE6,                                   \  # 定义宏：第六种标量类型
    SCALARTYPE7,                                   \  # 定义宏：第七种标量类型
    TYPE,                                          \  # 定义宏：类型
    NAME,                                          \  # 定义宏：名称
    ...)                                           \  # 定义宏：额外参数
  AT_DISPATCH_SWITCH(                              \  # 调用宏：根据类型和名称分派到对应的分发函数
      TYPE,                                        \  # 分派的类型参数
      NAME,                                        \  # 分派的名称参数
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND7( \  # 调用宏：处理所有类型和复杂类型及七个标量类型的分发函数
          SCALARTYPE1,                             \  # 第一种标量类型
          SCALARTYPE2,                             \  # 第二种标量类型
          SCALARTYPE3,                             \  # 第三种标量类型
          SCALARTYPE4,                             \  # 第四种标量类型
          SCALARTYPE5,                             \  # 第五种标量类型
          SCALARTYPE6,                             \  # 第六种标量类型
          SCALARTYPE7,                             \  # 第七种标量类型
          __VA_ARGS__))                            \  # 传递的额外参数
// 定义一个宏，用于处理所有标量类型和复数类型，并接受至少8个类型参数
#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8(  \
    SCALARTYPE1,                                      \
    SCALARTYPE2,                                      \
    SCALARTYPE3,                                      \
    SCALARTYPE4,                                      \
    SCALARTYPE5,                                      \
    SCALARTYPE6,                                      \
    SCALARTYPE7,                                      \
    SCALARTYPE8,                                      \
    ...)                                              \
  // 调用宏AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX处理可变数量的参数
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \
  // 依次调用宏AT_DISPATCH_CASE，处理每一个标量类型参数
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE7, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE8, __VA_ARGS__)

// 定义一个宏，用于处理所有类型和复数类型，并接受至少8个类型参数，同时包含类型和名称参数
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND8(    \
    SCALARTYPE1,                                   \
    SCALARTYPE2,                                   \
    SCALARTYPE3,                                   \
    SCALARTYPE4,                                   \
    SCALARTYPE5,                                   \
    SCALARTYPE6,                                   \
    SCALARTYPE7,                                   \
    SCALARTYPE8,                                   \
    TYPE,                                          \
    NAME,                                          \
    ...)                                           \
  // 调用宏AT_DISPATCH_SWITCH处理类型和名称参数，传递给宏AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8
  AT_DISPATCH_SWITCH(                              \
      TYPE,                                        \
      NAME,                                        \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8( \
          SCALARTYPE1,                             \
          SCALARTYPE2,                             \
          SCALARTYPE3,                             \
          SCALARTYPE4,                             \
          SCALARTYPE5,                             \
          SCALARTYPE6,                             \
          SCALARTYPE7,                             \
          SCALARTYPE8,                             \
          __VA_ARGS__))

// 定义一个宏，用于处理所有比特类型，并接受至少一个参数
#define AT_DISPATCH_CASE_BIT_TYPES(...)                  \
  // 调用宏AT_DISPATCH_CASE处理at::ScalarType::BitsX类型和可变数量的参数
  AT_DISPATCH_CASE(at::ScalarType::Bits1x8, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Bits2x4, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Bits4x2, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Bits8, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Bits16, __VA_ARGS__)

// 定义一个宏，用于处理比特类型，并包含类型和名称参数
#define AT_DISPATCH_BIT_TYPES(TYPE, NAME, ...) \
  // 调用宏AT_DISPATCH_SWITCH处理类型和名称参数，传递给宏AT_DISPATCH_CASE_BIT_TYPES
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_BIT_TYPES(__VA_ARGS__))
#define AT_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)     \
// 宏定义，用于根据索引类型分发操作，参数包括类型(TYPE)和名称(NAME)，以及额外的参数(__VA_ARGS__)
  AT_DISPATCH_SWITCH(                                \
// 调用另一个宏，执行根据类型分发的操作
      TYPE,                                          \
// 使用传入的类型参数(TYPE)作为分发的依据
      NAME,                                          \
// 使用传入的名称参数(NAME)
      AT_PRIVATE_CASE_TYPE_USING_HINT(               \
// 调用具体的分发宏，根据提示使用Int类型执行分发操作
          at::ScalarType::Int, index_t, __VA_ARGS__) \
// 传递Int类型的标量类型作为提示，并使用参数(__VA_ARGS__)执行操作
          AT_PRIVATE_CASE_TYPE_USING_HINT(           \
// 再次调用具体的分发宏，这次使用Long类型进行分发操作
              at::ScalarType::Long, index_t, __VA_ARGS__))

// ----------------------------------------------------------------------------
// DEPRECATED MACROS, DON'T USE THESE
// ----------------------------------------------------------------------------
// 下面是已弃用的宏，不要使用这些

#define AT_DISPATCH_ALL_TYPES_AND_HALF(TYPE, NAME, ...) \
// 宏定义，用于分发所有类型和Half类型的操作，包括类型(TYPE)和名称(NAME)，以及额外的参数(__VA_ARGS__)
  detail::deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF();  \
// 调用已弃用的详细宏
  AT_DISPATCH_SWITCH(                                   \
// 调用另一个宏，执行根据类型分发的操作
      TYPE,                                             \
// 使用传入的类型参数(TYPE)作为分发的依据
      NAME,                                             \
// 使用传入的名称参数(NAME)
      AT_DISPATCH_CASE_ALL_TYPES_AND(at::ScalarType::Half, __VA_ARGS__))
// 使用Half类型作为特例，执行分发操作，并传递参数(__VA_ARGS__)
```