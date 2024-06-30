# `D:\src\scipysrc\scipy\scipy\special\special\third_party\kokkos\mdspan.hpp`

```
#ifndef _MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_
#define _MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_

//BEGIN_FILE_INCLUDE: mdspan/include/mdspan/mdspan.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef MDSPAN_HPP_
#define MDSPAN_HPP_

#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
  #define MDSPAN_IMPL_STANDARD_NAMESPACE std
#endif

#ifndef MDSPAN_IMPL_PROPOSED_NAMESPACE
  #define MDSPAN_IMPL_PROPOSED_NAMESPACE experimental
#endif

//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/default_accessor.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/macros.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER


//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/config.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef __has_include
#  define __has_include(x) 0
#endif

#if __has_include(<version>)
#  include <version>


注释：


#ifndef _MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_
#define _MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_

//BEGIN_FILE_INCLUDE: mdspan/include/mdspan/mdspan.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef MDSPAN_HPP_
#define MDSPAN_HPP_

#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
  #define MDSPAN_IMPL_STANDARD_NAMESPACE std
#endif

#ifndef MDSPAN_IMPL_PROPOSED_NAMESPACE
  #define MDSPAN_IMPL_PROPOSED_NAMESPACE experimental
#endif

//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/default_accessor.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/macros.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER


//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/config.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef __has_include
#  define __has_include(x) 0
#endif

#if __has_include(<version>)
#  include <version>
#else
#  include <type_traits>  // 引入类型特性库，用于类型判断和转换
#  include <utility>      // 引入实用工具库，包含各种实用功能和模板

#endif

#ifdef _MSVC_LANG
#define _MDSPAN_CPLUSPLUS _MSVC_LANG   // 如果定义了 _MSVC_LANG，则使用之作为 _MDSPAN_CPLUSPLUS
#else
#define _MDSPAN_CPLUSPLUS __cplusplus  // 否则使用编译器预定义的 __cplusplus 作为 _MDSPAN_CPLUSPLUS
#endif

#define MDSPAN_CXX_STD_14 201402L       // 定义 C++14 的标准版本号
#define MDSPAN_CXX_STD_17 201703L       // 定义 C++17 的标准版本号
#define MDSPAN_CXX_STD_20 202002L       // 定义 C++20 的标准版本号
// 注意：GCC 在版本 13 中未更新此信息
#ifdef __clang__
#define MDSPAN_CXX_STD_23 202302L       // 对于 Clang，定义 C++23 的标准版本号
#else
#define MDSPAN_CXX_STD_23 202100L       // 否则定义 C++23 的标准版本号为 202100L
#endif

#define MDSPAN_HAS_CXX_14 (_MDSPAN_CPLUSPLUS >= MDSPAN_CXX_STD_14)  // 检查是否支持 C++14 或更高版本
#define MDSPAN_HAS_CXX_17 (_MDSPAN_CPLUSPLUS >= MDSPAN_CXX_STD_17)  // 检查是否支持 C++17 或更高版本
#define MDSPAN_HAS_CXX_20 (_MDSPAN_CPLUSPLUS >= MDSPAN_CXX_STD_20)  // 检查是否支持 C++20 或更高版本
#define MDSPAN_HAS_CXX_23 (_MDSPAN_CPLUSPLUS >= MDSPAN_CXX_STD_23)  // 检查是否支持 C++23 或更高版本

static_assert(_MDSPAN_CPLUSPLUS >= MDSPAN_CXX_STD_14, "mdspan requires C++14 or later.");  // 断言当前编译器至少支持 C++14

#ifndef _MDSPAN_COMPILER_CLANG
#  if defined(__clang__)
#    define _MDSPAN_COMPILER_CLANG __clang__  // 如果未定义 _MDSPAN_COMPILER_CLANG 且定义了 __clang__，则将其定义为 __clang__
#  endif
#endif

#if !defined(_MDSPAN_COMPILER_MSVC) && !defined(_MDSPAN_COMPILER_MSVC_CLANG)
#  if defined(_MSC_VER)
#    if !defined(_MDSPAN_COMPILER_CLANG)
#      define _MDSPAN_COMPILER_MSVC _MSC_VER         // 如果未定义 _MDSPAN_COMPILER_MSVC 且定义了 _MSC_VER，则将其定义为 _MSC_VER
#    else
#      define _MDSPAN_COMPILER_MSVC_CLANG _MSC_VER    // 否则将 _MDSPAN_COMPILER_MSVC_CLANG 定义为 _MSC_VER
#    endif
#  endif
#endif

#ifndef _MDSPAN_COMPILER_INTEL
#  ifdef __INTEL_COMPILER
#    define _MDSPAN_COMPILER_INTEL __INTEL_COMPILER   // 如果未定义 _MDSPAN_COMPILER_INTEL 且定义了 __INTEL_COMPILER，则将其定义为 __INTEL_COMPILER
#  endif
#endif

#ifndef _MDSPAN_COMPILER_APPLECLANG
#  ifdef __apple_build_version__
#    define _MDSPAN_COMPILER_APPLECLANG __apple_build_version__  // 如果未定义 _MDSPAN_COMPILER_APPLECLANG 且定义了 __apple_build_version__，则将其定义为 __apple_build_version__
#  endif
#endif

#ifndef _MDSPAN_HAS_CUDA
#  if defined(__CUDACC__)
#    define _MDSPAN_HAS_CUDA __CUDACC__   // 如果未定义 _MDSPAN_HAS_CUDA 且定义了 __CUDACC__，则将其定义为 __CUDACC__
#  endif
#endif

#ifndef _MDSPAN_HAS_HIP
#  if defined(__HIPCC__)
#    define _MDSPAN_HAS_HIP __HIPCC__     // 如果未定义 _MDSPAN_HAS_HIP 且定义了 __HIPCC__，则将其定义为 __HIPCC__
#  endif
#endif

#ifndef _MDSPAN_HAS_SYCL
#  if defined(SYCL_LANGUAGE_VERSION)
#    define _MDSPAN_HAS_SYCL SYCL_LANGUAGE_VERSION  // 如果未定义 _MDSPAN_HAS_SYCL 且定义了 SYCL_LANGUAGE_VERSION，则将其定义为 SYCL_LANGUAGE_VERSION
#  endif
#endif

#ifndef __has_cpp_attribute
#  define __has_cpp_attribute(x) 0         // 如果未定义 __has_cpp_attribute 宏，则定义为 0
#endif

#ifndef _MDSPAN_PRESERVE_STANDARD_LAYOUT
// 默认情况下保留标准布局，但在确定这不会对编译器或优化器造成不合理的开销之前，不会删除关闭此选项的旧版本。
#  define _MDSPAN_PRESERVE_STANDARD_LAYOUT 1  // 默认开启标准布局的保留
#endif

#if !defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
#  if ((__has_cpp_attribute(no_unique_address) >= 201803L) && \
       (!defined(__NVCC__) || MDSPAN_HAS_CXX_20) && \
       (!defined(_MDSPAN_COMPILER_MSVC) || MDSPAN_HAS_CXX_20))
#    define _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS 1        // 如果满足条件，则定义为 1，表示支持 no_unique_address 特性
#    define _MDSPAN_NO_UNIQUE_ADDRESS [[no_unique_address]]  // 定义 no_unique_address 的属性
#  else
#    define _MDSPAN_NO_UNIQUE_ADDRESS  // 否则定义为空
#  endif
#endif

// NVCC 版本低于 11.6 会在 no-unique-address 模拟上失败，因此假装使用它（避免完整的 EBO 工作，
// 同样 NVCC 也不喜欢...），并将宏留空
#ifndef _MDSPAN_NO_UNIQUE_ADDRESS
#  if defined(__NVCC__)
#    define _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS 1        // 如果定义了 __NVCC__，则定义为 1，表示支持 no_unique_address 特性
#    define _MDSPAN_USE_FAKE_ATTRIBUTE_NO_UNIQUE_ADDRESS    // 定义假的 no_unique_address 属性
#  endif
#  define _MDSPAN_NO_UNIQUE_ADDRESS  // 定义为空
#endif
// 如果不是使用 HIPCC 编译器
#ifndef __HIPCC__
    // 如果未定义 _MDSPAN_USE_CONCEPTS，并且编译器支持 C++20 概念特性
    #ifndef _MDSPAN_USE_CONCEPTS
        #if defined(__cpp_concepts) && __cpp_concepts >= 201507L
            // 定义 _MDSPAN_USE_CONCEPTS 为 1
            #define _MDSPAN_USE_CONCEPTS 1
        #endif
    #endif
#endif

// 如果未定义 _MDSPAN_USE_FOLD_EXPRESSIONS
#ifndef _MDSPAN_USE_FOLD_EXPRESSIONS
    // 如果编译器支持折叠表达式（fold expressions）特性，或者如果不支持折叠表达式但支持 C++17
    #if (defined(__cpp_fold_expressions) && __cpp_fold_expressions >= 201603L) \
            || (!defined(__cpp_fold_expressions) && MDSPAN_HAS_CXX_17)
        // 定义 _MDSPAN_USE_FOLD_EXPRESSIONS 为 1
        #define _MDSPAN_USE_FOLD_EXPRESSIONS 1
    #endif
#endif

// 如果未定义 _MDSPAN_USE_INLINE_VARIABLES
#ifndef _MDSPAN_USE_INLINE_VARIABLES
    // 如果编译器支持内联变量（inline variables）特性，或者如果不支持内联变量但支持 C++17
    #if defined(__cpp_inline_variables) && __cpp_inline_variables >= 201606L \
            || (!defined(__cpp_inline_variables) && MDSPAN_HAS_CXX_17)
        // 定义 _MDSPAN_USE_INLINE_VARIABLES 为 1
        #define _MDSPAN_USE_INLINE_VARIABLES 1
    #endif
#endif

// 如果未定义 _MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS
#ifndef _MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS
    // 如果不支持类型特征变量模板（type trait variable templates）或者不支持 C++17
    #if (!(defined(__cpp_lib_type_trait_variable_templates) && __cpp_lib_type_trait_variable_templates >= 201510L) \
            || !MDSPAN_HAS_CXX_17)
        // 如果不是苹果的 Clang 编译器，并且不支持 C++17，则定义 _MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS 为 1
        #if !(defined(_MDSPAN_COMPILER_APPLECLANG) && MDSPAN_HAS_CXX_17)
            #define _MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS 1
        #endif
    #endif
#endif

// 如果未定义 _MDSPAN_USE_VARIABLE_TEMPLATES
#ifndef _MDSPAN_USE_VARIABLE_TEMPLATES
    // 如果编译器支持变量模板（variable templates）特性，或者如果不支持变量模板但支持 C++17
    #if (defined(__cpp_variable_templates) && __cpp_variable_templates >= 201304 && MDSPAN_HAS_CXX_17) \
            || (!defined(__cpp_variable_templates) && MDSPAN_HAS_CXX_17)
        // 定义 _MDSPAN_USE_VARIABLE_TEMPLATES 为 1
        #define _MDSPAN_USE_VARIABLE_TEMPLATES 1
    #endif
#endif // _MDSPAN_USE_VARIABLE_TEMPLATES

// 如果未定义 _MDSPAN_USE_CONSTEXPR_14
#ifndef _MDSPAN_USE_CONSTEXPR_14
    // 如果编译器支持 C++14 constexpr 特性，或者如果不支持 C++14 但支持 C++14 特定情况下
    #if (defined(__cpp_constexpr) && __cpp_constexpr >= 201304) \
            || (!defined(__cpp_constexpr) && MDSPAN_HAS_CXX_14) \
            && (!(defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1700))
        // 定义 _MDSPAN_USE_CONSTEXPR_14 为 1
        #define _MDSPAN_USE_CONSTEXPR_14 1
    #endif
#endif

// 如果未定义 _MDSPAN_USE_INTEGER_SEQUENCE
#ifndef _MDSPAN_USE_INTEGER_SEQUENCE
    // 如果使用 MSVC 编译器，并且支持整数序列（integer sequence）特性
    #if defined(_MDSPAN_COMPILER_MSVC)
        #if (defined(__cpp_lib_integer_sequence) && __cpp_lib_integer_sequence >= 201304)
            // 定义 _MDSPAN_USE_INTEGER_SEQUENCE 为 1
            #define _MDSPAN_USE_INTEGER_SEQUENCE 1
        #endif
    #endif
#endif

// 如果未定义 _MDSPAN_USE_INTEGER_SEQUENCE
#ifndef _MDSPAN_USE_INTEGER_SEQUENCE
    // 如果编译器支持整数序列特性，或者如果不支持整数序列但支持 C++14，或者特定情况下需要 C++14 支持
    /* 根据我的了解，libc++ 似乎认为这是一个 C++11 特性... */
    || (defined(__GLIBCXX__) && __GLIBCXX__ > 20150422 && __GNUC__ < 5 && !defined(__INTEL_CXX11_MODE__))
        // 定义 _MDSPAN_USE_INTEGER_SEQUENCE 为 1
        #define _MDSPAN_USE_INTEGER_SEQUENCE 1
    // 如果使用苹果的 Clang 编译器，并且支持 C++14
    #elif defined(_MDSPAN_COMPILER_APPLECLANG) && MDSPAN_HAS_CXX_14
        // 定义 _MDSPAN_USE_INTEGER_SEQUENCE 为 1
        #define _MDSPAN_USE_INTEGER_SEQUENCE 1
#endif

// 如果未定义 _MDSPAN_USE_RETURN_TYPE_DEDUCTION
#ifndef _MDSPAN_USE_RETURN_TYPE_DEDUCTION
    // 如果编译器支持返回类型推导（return type deduction）特性，或者如果不支持返回类型推导但支持 C++14
    #if (defined(__cpp_return_type_deduction) && __cpp_return_type_deduction >= 201304) \
            || (!defined(__cpp_return_type_deduction) && MDSPAN_HAS_CXX_14)
#ifndef _MDSPAN_USE_RETURN_TYPE_DEDUCTION
#    define _MDSPAN_USE_RETURN_TYPE_DEDUCTION 1
#  endif
#endif


// 如果未定义 _MDSPAN_USE_RETURN_TYPE_DEDUCTION，则定义为 1
#ifndef _MDSPAN_USE_RETURN_TYPE_DEDUCTION
#    define _MDSPAN_USE_RETURN_TYPE_DEDUCTION 1
#  endif
#endif



#ifndef _MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
#  if (!defined(__NVCC__) || (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10 >= 1170)) && \
      ((defined(__cpp_deduction_guides) && __cpp_deduction_guides >= 201703) || \
       (!defined(__cpp_deduction_guides) && MDSPAN_HAS_CXX_17))
#    define _MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION 1
#  endif
#endif


// 如果未定义 _MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
// 根据条件判断是否满足设定条件，若满足则定义为 1
#ifndef _MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
#  if (!defined(__NVCC__) || (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10 >= 1170)) && \
      ((defined(__cpp_deduction_guides) && __cpp_deduction_guides >= 201703) || \
       (!defined(__cpp_deduction_guides) && MDSPAN_HAS_CXX_17))
#    define _MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION 1
#  endif
#endif



#ifndef _MDSPAN_USE_STANDARD_TRAIT_ALIASES
#  if (defined(__cpp_lib_transformation_trait_aliases) && __cpp_lib_transformation_trait_aliases >= 201304) \
          || (!defined(__cpp_lib_transformation_trait_aliases) && MDSPAN_HAS_CXX_14)
#    define _MDSPAN_USE_STANDARD_TRAIT_ALIASES 1
#  elif defined(_MDSPAN_COMPILER_APPLECLANG) && MDSPAN_HAS_CXX_14
     // appleclang seems to be missing the __cpp_lib_... macros, but doesn't seem to lie about C++14
#    define _MDSPAN_USE_STANDARD_TRAIT_ALIASES 1
#  endif
#endif


// 如果未定义 _MDSPAN_USE_STANDARD_TRAIT_ALIASES
// 根据条件判断是否满足设定条件，若满足则定义为 1
#ifndef _MDSPAN_USE_STANDARD_TRAIT_ALIASES
#  if (defined(__cpp_lib_transformation_trait_aliases) && __cpp_lib_transformation_trait_aliases >= 201304) \
          || (!defined(__cpp_lib_transformation_trait_aliases) && MDSPAN_HAS_CXX_14)
#    define _MDSPAN_USE_STANDARD_TRAIT_ALIASES 1
#  elif defined(_MDSPAN_COMPILER_APPLECLANG) && MDSPAN_HAS_CXX_14
     // appleclang 似乎缺少 __cpp_lib_... 宏，但不会误报 C++14
#    define _MDSPAN_USE_STANDARD_TRAIT_ALIASES 1
#  endif
#endif



#ifndef _MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND
#  ifdef __GNUC__
#    if __GNUC__ < 9
#      define _MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND 1
#    endif
#  endif
#endif


// 如果未定义 _MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND
// 根据条件判断是否满足设定条件，若满足则定义为 1
#ifndef _MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND
#  ifdef __GNUC__
#    if __GNUC__ < 9
#      define _MDSPAN_DEFAULTED_CONSTRUCTORS_INHERITANCE_WORKAROUND 1
#    endif
#  endif
#endif



#ifndef MDSPAN_CONDITIONAL_EXPLICIT
#  if MDSPAN_HAS_CXX_20
#    define MDSPAN_CONDITIONAL_EXPLICIT(COND) explicit(COND)
#  else
#    define MDSPAN_CONDITIONAL_EXPLICIT(COND)
#  endif
#endif


// 如果未定义 MDSPAN_CONDITIONAL_EXPLICIT
// 根据条件判断是否满足设定条件，若满足则定义为 explicit(COND)，否则为空
#ifndef MDSPAN_CONDITIONAL_EXPLICIT
#  if MDSPAN_HAS_CXX_20
#    define MDSPAN_CONDITIONAL_EXPLICIT(COND) explicit(COND)
#  else
#    define MDSPAN_CONDITIONAL_EXPLICIT(COND)
#  endif
#endif



#ifndef MDSPAN_USE_BRACKET_OPERATOR
#  if defined(__cpp_multidimensional_subscript)
#    define MDSPAN_USE_BRACKET_OPERATOR 1
#  else
#    define MDSPAN_USE_BRACKET_OPERATOR 0
#  endif
#endif


// 如果未定义 MDSPAN_USE_BRACKET_OPERATOR
// 根据条件判断是否满足设定条件，若满足则定义为 1，否则定义为 0
#ifndef MDSPAN_USE_BRACKET_OPERATOR
#  if defined(__cpp_multidimensional_subscript)
#    define MDSPAN_USE_BRACKET_OPERATOR 1
#  else
#    define MDSPAN_USE_BRACKET_OPERATOR 0
#  endif
#endif



#ifndef MDSPAN_USE_PAREN_OPERATOR
#  if !MDSPAN_USE_BRACKET_OPERATOR
#    define MDSPAN_USE_PAREN_OPERATOR 1
#  else
#    define MDSPAN_USE_PAREN_OPERATOR 0
#  endif
#endif


// 如果未定义 MDSPAN_USE_PAREN_OPERATOR
// 根据条件判断是否满足设定条件，若 MDSPAN_USE_BRACKET_OPERATOR 未定义，则定义为 1，否则定义为 0
#ifndef MDSPAN_USE_PAREN_OPERATOR
#  if !MDSPAN_USE_BRACKET_OPERATOR
#    define MDSPAN_USE_PAREN_OPERATOR 1
#  else
#    define MDSPAN_USE_PAREN_OPERATOR 0
#  endif
#endif



#if MDSPAN_USE_BRACKET_OPERATOR
#  define __MDSPAN_OP(mds,...) mds[__VA_ARGS__]
// Corentins demo compiler for subscript chokes on empty [] call,
// though I believe the proposal supports it?
#ifdef MDSPAN_NO_EMPTY_BRACKET_OPERATOR
#  define __MDSPAN_OP0(mds) mds.accessor().access(mds.data_handle(),0)
#else
#  define __MDSPAN_OP0(mds) mds[]
#endif
#  define __MDSPAN_OP1(mds, a) mds[a]
#  define __MDSPAN_OP2(mds, a, b) mds[a,b]
#  define __MDSPAN_OP3(mds, a, b, c) mds[a,b,c]
#  define __MDSPAN_OP4(mds, a, b, c, d) mds[a,b,c,d]
#  define __MDSPAN_OP5(mds, a, b, c, d, e) mds[a,b,c,d,e]
#  define __MDSPAN_OP6(mds, a, b, c, d, e, f) mds[a,b,c,d,e,f]
#else
#  define __MDSPAN_OP(mds,...) mds(__VA_ARGS__)
#  define __MDSPAN_OP0(mds) mds()
#  define __MDSPAN_OP1(mds, a) mds(a)
#  define __MDSPAN_OP2(mds, a, b) mds(a,b)
#  define __MDSPAN_OP3(mds, a, b, c) mds(a,b,c)
#  define __MDSPAN_OP4(mds, a, b, c, d) mds(a,b,c,d)
#  define __MDSPAN_OP5(mds, a, b, c, d, e) mds(a,b,c,d,e)
#  define __MDSPAN_OP6(mds, a, b, c, d, e, f) mds(a,b,c,d,e,f)
#endif


// 如果 MDSPAN_USE_BRACKET_OPERATOR 已定义
// 定义宏 __MDSPAN_OP 为 mds[__VA_ARGS__]
#if MDSPAN_USE_BRACKET_OPERATOR
#  define __MDSPAN_OP(mds,...) mds[__VA_ARGS__]
// 对于 MDSPAN_NO_EMPTY_BRACKET_OPERATOR 宏的条件判断，根据情况定义 __MDSPAN_OP0
// Corentins 演示编译器对于空的 [] 调用有问题，尽管我相信该提案支持它？
#ifdef MDSPAN_NO_EMPTY_BRACKET_OPERATOR
#  define __MDSPAN_OP0(mds) mds.accessor().access(mds.data_handle(),0)
#else
#  define __MDSPAN_OP0(mds) mds[]
#endif
// 定义宏 __MDSPAN_OP1 至 __MDSPAN_OP6，分别
#if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_SYCL)
#include "assert.h"
#endif


#ifndef _MDSPAN_HOST_DEVICE
#  if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP)
#    define _MDSPAN_HOST_DEVICE __host__ __device__
#  else
#    define _MDSPAN_HOST_DEVICE
#  endif
#endif


#ifndef MDSPAN_FORCE_INLINE_FUNCTION
#  ifdef _MDSPAN_COMPILER_MSVC // Microsoft compilers
#    define MDSPAN_FORCE_INLINE_FUNCTION __forceinline _MDSPAN_HOST_DEVICE
#  else
#    define MDSPAN_FORCE_INLINE_FUNCTION __attribute__((always_inline)) _MDSPAN_HOST_DEVICE
#  endif
#endif


#ifndef MDSPAN_INLINE_FUNCTION
#  define MDSPAN_INLINE_FUNCTION inline _MDSPAN_HOST_DEVICE
#endif


#ifndef MDSPAN_FUNCTION
#  define MDSPAN_FUNCTION _MDSPAN_HOST_DEVICE
#endif


#ifdef _MDSPAN_HAS_HIP
#  define MDSPAN_DEDUCTION_GUIDE _MDSPAN_HOST_DEVICE
#else
#  define MDSPAN_DEDUCTION_GUIDE
#endif


// In CUDA defaulted functions do not need host device markup
#ifndef MDSPAN_INLINE_FUNCTION_DEFAULTED
#  define MDSPAN_INLINE_FUNCTION_DEFAULTED
#endif


//==============================================================================
// <editor-fold desc="Preprocessor helpers"> {{{1


#define MDSPAN_PP_COUNT(...) \
  _MDSPAN_PP_INTERNAL_EXPAND_ARGS_PRIVATE( \
    _MDSPAN_PP_INTERNAL_ARGS_AUGMENTER(__VA_ARGS__) \
  )


#define _MDSPAN_PP_INTERNAL_ARGS_AUGMENTER(...) unused, __VA_ARGS__


#define _MDSPAN_PP_INTERNAL_EXPAND(x) x
#define _MDSPAN_PP_INTERNAL_EXPAND_ARGS_PRIVATE(...) \
  _MDSPAN_PP_INTERNAL_EXPAND( \
    _MDSPAN_PP_INTERNAL_COUNT_PRIVATE( \
      __VA_ARGS__, 69, 68, 67, 66, 65, 64, 63, 62, 61, \
      60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,  \
      48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37,  \
      36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,  \
      24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,  \
      12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 \
    ) \
  )


#define _MDSPAN_PP_INTERNAL_COUNT_PRIVATE( \
         _1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, \
    _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, \
    _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, \
    _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, \
    _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, \
    _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, \
    _70, count, ...) count \
    /**/


# define MDSPAN_PP_STRINGIFY_IMPL(x) #x
# define MDSPAN_PP_STRINGIFY(x) MDSPAN_PP_STRINGIFY_IMPL(x)


# define MDSPAN_PP_CAT_IMPL(x, y) x ## y
# define MDSPAN_PP_CAT(x, y) MDSPAN_PP_CAT_IMPL(x, y)


# define MDSPAN_PP_EVAL(X, ...) X(__VA_ARGS__)


# define MDSPAN_PP_REMOVE_PARENS_IMPL(...) __VA_ARGS__
# define MDSPAN_PP_REMOVE_PARENS(...) MDSPAN_PP_REMOVE_PARENS_IMPL __VA_ARGS__


# define MDSPAN_IMPL_STANDARD_NAMESPACE_STRING MDSPAN_PP_STRINGIFY(MDSPAN_IMPL_STANDARD_NAMESPACE)
// 定义一个字符串，包含命名空间的字符串表示，用于命名空间组合
#define MDSPAN_IMPL_PROPOSED_NAMESPACE_STRING MDSPAN_PP_STRINGIFY(MDSPAN_IMPL_STANDARD_NAMESPACE) "::" MDSPAN_PP_STRINGIFY(MDSPAN_IMPL_PROPOSED_NAMESPACE)

// 定义 MDSPAN_IMPL_STANDARD_NAMESPACE 命名空间内的 detail 命名空间
namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

// 根据不同的编译器宏选择不同的预处理函数，处理预条件违规情况
#if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP)
// 如果定义了 _MDSPAN_HAS_CUDA 或 _MDSPAN_HAS_HIP，使用 printf 输出错误信息，并断言失败
MDSPAN_FUNCTION inline void default_precondition_violation_handler(const char* cond, const char* file, unsigned line)
{
  printf("%s:%u: precondition failure: `%s`\n", file, line, cond);
  assert(0);
}
#elif defined(_MDSPAN_HAS_SYCL)
// 如果定义了 _MDSPAN_HAS_SYCL，使用 sycl::ext::oneapi::experimental::printf 输出错误信息，并断言失败
MDSPAN_FUNCTION inline void default_precondition_violation_handler(const char* cond, const char* file, unsigned line)
{
  sycl::ext::oneapi::experimental::printf("%s:%u: precondition failure: `%s`\n", file, line, cond);
  assert(0);
}
#else
// 默认情况下，使用 std::fprintf 输出错误信息到标准错误流，并调用 std::abort 终止程序
MDSPAN_FUNCTION inline void default_precondition_violation_handler(const char* cond, const char* file, unsigned line)
{
  std::fprintf(stderr, "%s:%u: precondition failure: `%s`\n", file, line, cond);
  std::abort();
}
#endif

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

// 如果未定义 MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER 宏，则定义为默认的处理器宏
#ifndef MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER
#define MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER(cond, file, line) \
  MDSPAN_IMPL_STANDARD_NAMESPACE::detail::default_precondition_violation_handler(cond, file, line)
#endif

// 如果未定义 MDSPAN_IMPL_CHECK_PRECONDITION 宏
#ifndef MDSPAN_IMPL_CHECK_PRECONDITION
  // 如果处于调试模式（NDEBUG 未定义），定义 MDSPAN_IMPL_CHECK_PRECONDITION 为 0
  #ifndef NDEBUG
    #define MDSPAN_IMPL_CHECK_PRECONDITION 0
  // 否则定义为 1
  #else
    #define MDSPAN_IMPL_CHECK_PRECONDITION 1
  #endif
#endif

// 在 MDSPAN_IMPL_STANDARD_NAMESPACE 命名空间内的 detail 命名空间定义预条件检查模板
namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

// 定义模板函数 precondition，根据 MDSPAN_IMPL_CHECK_PRECONDITION 编译时常量进行编译时检查
template <bool check = MDSPAN_IMPL_CHECK_PRECONDITION>
MDSPAN_FUNCTION constexpr void precondition(const char* cond, const char* file, unsigned line)
{
  // 如果 check 为 false，则直接返回，不进行检查
  if (not check) { return; }
  // 用于确保宏未使用参数的情况下的编译警告消除
  (void) cond;
  (void) file;
  (void) line;
  // 调用 MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER 处理预条件失败
  MDSPAN_IMPL_PRECONDITION_VIOLATION_HANDLER(cond, file, line);
}

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

// 定义 MDSPAN_IMPL_PRECONDITION 宏，用于执行预条件检查
#define MDSPAN_IMPL_PRECONDITION(...) \
  do { \
    // 如果条件不满足，则调用 MDSPAN_IMPL_STANDARD_NAMESPACE::detail::precondition 函数处理
    if (not (__VA_ARGS__)) { \
      MDSPAN_IMPL_STANDARD_NAMESPACE::detail::precondition(#__VA_ARGS__, __FILE__, __LINE__); \
    } \
  } while (0)

// </editor-fold> end Preprocessor helpers }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Concept emulation"> {{{1

// 这些兼容性宏在 mdspan 中模拟概念的需求
#ifdef _MDSPAN_USE_CONCEPTS
// 如果定义了 _MDSPAN_USE_CONCEPTS，定义 MDSPAN_CLOSE_ANGLE_REQUIRES 宏为 > requires REQ
#  define MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) > requires REQ
// 定义 MDSPAN_FUNCTION_REQUIRES 宏，在函数声明中包含 requires 语句
#  define MDSPAN_FUNCTION_REQUIRES(PAREN_PREQUALS, FNAME, PAREN_PARAMS, QUALS, REQ) \
     MDSPAN_PP_REMOVE_PARENS(PAREN_PREQUALS) FNAME PAREN_PARAMS QUALS requires REQ \
     /**/
#else
// 如果未定义 _MDSPAN_USE_CONCEPTS
// 定义 MDSPAN_CLOSE_ANGLE_REQUIRES 宏为 , typename ::std::enable_if<(REQ), int>::type = 0>
#  define MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) , typename ::std::enable_if<(REQ), int>::type = 0>
// 如果宏 MDSPAN_FUNCTION_REQUIRES 被定义，则根据参数展开为模板要求宏 MDSPAN_TEMPLATE_REQUIRES 和函数声明
#define MDSPAN_FUNCTION_REQUIRES(PAREN_PREQUALS, FNAME, PAREN_PARAMS, QUALS, REQ) \
 MDSPAN_TEMPLATE_REQUIRES( \
   class __function_requires_ignored=void, \  // 定义一个无关紧要的类，用于检测是否满足模板要求
   (std::is_void<__function_requires_ignored>::value && REQ) \  // 判断模板要求是否满足
 ) MDSPAN_PP_REMOVE_PARENS(PAREN_PREQUALS) FNAME PAREN_PARAMS QUALS \  // 展开为模板要求和函数声明
 /**/
#endif

// 如果编译器为 Microsoft Visual C++ 并且不是传统模式或未定义 _MSVC_TRADITIONAL，则使用以下宏定义
#if defined(_MDSPAN_COMPILER_MSVC) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL)
// 定义模板要求宏，参数数量由宏参数个数决定
#  define MDSPAN_TEMPLATE_REQUIRES(...) \
   MDSPAN_PP_CAT( \
     MDSPAN_PP_CAT(MDSPAN_TEMPLATE_REQUIRES_, MDSPAN_PP_COUNT(__VA_ARGS__))\
     (__VA_ARGS__), \
   ) \
 /**/
// 否则使用以下宏定义
#else
// 定义模板要求宏，参数数量由宏参数个数决定，使用 MDSPAN_PP_EVAL 展开参数
#  define MDSPAN_TEMPLATE_REQUIRES(...) \
   MDSPAN_PP_EVAL( \
       MDSPAN_PP_CAT(MDSPAN_TEMPLATE_REQUIRES_, MDSPAN_PP_COUNT(__VA_ARGS__)), \
       __VA_ARGS__ \
   ) \
   /**/
#endif

// 定义各个具体参数数量的模板要求宏，每个宏定义一个模板声明
#define MDSPAN_TEMPLATE_REQUIRES_2(TP1, REQ) \
 template<TP1 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_3(TP1, TP2, REQ) \
 template<TP1, TP2 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_4(TP1, TP2, TP3, REQ) \
 template<TP1, TP2, TP3 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_5(TP1, TP2, TP3, TP4, REQ) \
 template<TP1, TP2, TP3, TP4 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_6(TP1, TP2, TP3, TP4, TP5, REQ) \
 template<TP1, TP2, TP3, TP4, TP5 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_7(TP1, TP2, TP3, TP4, TP5, TP6, REQ) \
 template<TP1, TP2, TP3, TP4, TP5, TP6 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_8(TP1, TP2, TP3, TP4, TP5, TP6, TP7, REQ) \
 template<TP1, TP2, TP3, TP4, TP5, TP6, TP7 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_9(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, REQ) \
 template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_10(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, REQ) \
 template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_11(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, REQ) \
 template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_12(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, REQ) \
 template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
#define MDSPAN_TEMPLATE_REQUIRES_13(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, REQ) \
 template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12 \
   MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
   /**/
// 定义一个模板宏，用于指定要求的模板参数数量为14，并设置条件需求
#define MDSPAN_TEMPLATE_REQUIRES_14(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, REQ) \
  // 使用模板定义，接受14个模板参数及条件需求
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

// 定义一个模板宏，用于指定要求的模板参数数量为15，并设置条件需求
#define MDSPAN_TEMPLATE_REQUIRES_15(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, REQ) \
  // 使用模板定义，接受15个模板参数及条件需求
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

// 定义一个模板宏，用于指定要求的模板参数数量为16，并设置条件需求
#define MDSPAN_TEMPLATE_REQUIRES_16(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, REQ) \
  // 使用模板定义，接受16个模板参数及条件需求
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

// 定义一个模板宏，用于指定要求的模板参数数量为17，并设置条件需求
#define MDSPAN_TEMPLATE_REQUIRES_17(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, REQ) \
  // 使用模板定义，接受17个模板参数及条件需求
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

// 定义一个模板宏，用于指定要求的模板参数数量为18，并设置条件需求
#define MDSPAN_TEMPLATE_REQUIRES_18(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, REQ) \
  // 使用模板定义，接受18个模板参数及条件需求
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

// 定义一个模板宏，用于指定要求的模板参数数量为19，并设置条件需求
#define MDSPAN_TEMPLATE_REQUIRES_19(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, REQ) \
  // 使用模板定义，接受19个模板参数及条件需求
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

// 定义一个模板宏，用于指定要求的模板参数数量为20，并设置条件需求
#define MDSPAN_TEMPLATE_REQUIRES_20(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, TP19, REQ) \
  // 使用模板定义，接受20个模板参数及条件需求
  template<TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, TP11, TP12, TP13, TP14, TP15, TP16, TP17, TP18, TP19 \
    MDSPAN_CLOSE_ANGLE_REQUIRES(REQ) \
    /**/

// 定义一个宏，用于只在使用时实例化
#define MDSPAN_INSTANTIATE_ONLY_IF_USED \
  // 使用模板定义，接受一个模板参数并进行条件判断是否为空
  MDSPAN_TEMPLATE_REQUIRES( \
    class __instantiate_only_if_used_tparam=void, \
    ( _MDSPAN_TRAIT(std::is_void, __instantiate_only_if_used_tparam) ) \
  ) \
  /**/

// </editor-fold> end Concept emulation }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="inline variables"> {{{1

#ifdef _MDSPAN_USE_INLINE_VARIABLES
// 如果定义了宏 _MDSPAN_USE_INLINE_VARIABLES，则定义 _MDSPAN_INLINE_VARIABLE 为 inline
#  define _MDSPAN_INLINE_VARIABLE inline
#else
// 否则定义 _MDSPAN_INLINE_VARIABLE 为空
#  define _MDSPAN_INLINE_VARIABLE
#endif

// </editor-fold> end inline variables }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Return type deduction"> {{{1

#if _MDSPAN_USE_RETURN_TYPE_DEDUCTION
// 如果定义了宏 _MDSPAN_USE_RETURN_TYPE_DEDUCTION，则定义一个宏 _MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE
#  define _MDSPAN_DEDUCE_RETURN_TYPE_SINGLE_LINE(SIGNATURE, BODY) \
    // 使用 auto 进行返回类型推导，返回经去除括号的 BODY
    auto MDSPAN_PP_REMOVE_PARENS(SIGNATURE) { return MDSPAN_PP_REMOVE_PARENS(BODY); }
// <editor-fold desc="fold expressions"> {{{1
// 定义用于折叠表达式的宏，适配使用折叠表达式的平台
struct __mdspan_enable_fold_comma { };

#ifdef _MDSPAN_USE_FOLD_EXPRESSIONS
// 定义逻辑与折叠表达式宏
#  define _MDSPAN_FOLD_AND(...) ((__VA_ARGS__) && ...)
// 定义模板化的逻辑与折叠表达式宏
#  define _MDSPAN_FOLD_AND_TEMPLATE(...) ((__VA_ARGS__) && ...)
// 定义逻辑或折叠表达式宏
#  define _MDSPAN_FOLD_OR(...) ((__VA_ARGS__) || ...)
// 定义左值赋值折叠表达式宏
#  define _MDSPAN_FOLD_ASSIGN_LEFT(INIT, ...) (INIT = ... = (__VA_ARGS__))
// 定义右值赋值折叠表达式宏
#  define _MDSPAN_FOLD_ASSIGN_RIGHT(PACK, ...) (PACK = ... = (__VA_ARGS__))
// 定义右值乘法折叠表达式宏
#  define _MDSPAN_FOLD_TIMES_RIGHT(PACK, ...) (PACK * ... * (__VA_ARGS__))
// 定义右值加法折叠表达式宏
#  define _MDSPAN_FOLD_PLUS_RIGHT(PACK, ...) (PACK + ... + (__VA_ARGS__))
// 定义逗号折叠表达式宏
#  define _MDSPAN_FOLD_COMMA(...) ((__VA_ARGS__), ...)
#else

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

namespace __fold_compatibility_impl {

// 对于不支持折叠表达式的情况，提供兼容实现
// 针对逻辑与的右折叠实现
MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) __fold_right_and_impl() {
  return true;
}

// 递归模板处理逻辑与右折叠实现
template <class Arg, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) __fold_right_and_impl(Arg&& arg, Args&&... args) {
  return ((Arg&&)arg) && __fold_compatibility_impl::__fold_right_and_impl((Args&&)args...);
}

// 针对逻辑或的右折叠实现
MDSPAN_FORCE_INLINE_FUNCTION
constexpr decltype(auto) __fold_right_or_impl() {
  return false;
}

// 递归模板处理逻辑或右折叠实现
template <class Arg, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_or_impl(Arg&& arg, Args&&... args) {
  return ((Arg&&)arg) || __fold_compatibility_impl::__fold_right_or_impl((Args&&)args...);
}

// 左值赋值右折叠实现
template <class Arg1>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_left_assign_impl(Arg1&& arg1) {
  return (Arg1&&)arg1;
}

// 递归模板处理左值赋值右折叠实现
template <class Arg1, class Arg2, class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_left_assign_impl(Arg1&& arg1, Arg2&& arg2, Args&&... args) {
  return __fold_compatibility_impl::__fold_left_assign_impl((((Arg1&&)arg1) = ((Arg2&&)arg2)), (Args&&)args...);
}

// 右值赋值右折叠实现
template <class Arg1>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr auto __fold_right_assign_impl(Arg1&& arg1) {
  return (Arg1&&)arg1;
}


这段代码是为了支持折叠表达式而定义的一系列宏和函数模板，同时也提供了在不支持折叠表达式的环境下的兼容实现。
// <editor-fold desc="right and"> {{{2
// 定义一个模板结构体 __fold_right_and_impl_，用于实现右折叠的逻辑与操作
template <class... Args>
struct __fold_right_and_impl_;

// 特化模板结构体 __fold_right_and_impl_，当没有参数时返回布尔类型的 true
template <>
struct __fold_right_and_impl_<> {
    using __rv = bool;
    // 静态成员函数 __impl 实现，始终返回 true
    MDSPAN_FORCE_INLINE_FUNCTION
    static constexpr __rv
    __impl() noexcept {
        return true;
    }
};

// 模板结构体 __fold_right_and_impl_ 的部分特化，处理一个及多个参数的情况
template <class Arg, class... Args>
struct __fold_right_and_impl_<Arg, Args...> {
    // 嵌套类型别名 __next_t 表示下一个递归结构
    using __next_t = __fold_right_and_impl_<Args...>;
    // 嵌套类型别名 __rv 为 Arg 和下一个递归结构的逻辑与操作结果的类型
    using __rv = decltype(std::declval<Arg>() && std::declval<typename __next_t::__rv>());
    // 静态成员函数 __impl 实现右折叠的逻辑与操作
    MDSPAN_FORCE_INLINE_FUNCTION
    static constexpr __rv
    __impl(Arg&& arg, Args&&... args) noexcept {
        // 返回当前参数和下一个递归结构的逻辑与结果
        return ((Arg&&)arg) && __next_t::__impl((Args&&)args...);
    }
};

// 模板函数 __fold_right_and_impl 实现右折叠的逻辑与操作
template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_and_impl_<Args...>::__rv
__fold_right_and_impl(Args&&... args) {
    // 调用模板结构体 __fold_right_and_impl_ 的静态成员函数 __impl
    return __fold_right_and_impl_<Args...>::__impl((Args&&)args...);
}
// </editor-fold> end right and }}}2



// <editor-fold desc="right or"> {{{2
// 定义一个模板结构体 __fold_right_or_impl_，用于实现右折叠的逻辑或操作
template <class... Args>
struct __fold_right_or_impl_;

// 特化模板结构体 __fold_right_or_impl_，当没有参数时返回布尔类型的 false
template <>
struct __fold_right_or_impl_<> {
    using __rv = bool;
    // 静态成员函数 __impl 实现，始终返回 false
    MDSPAN_FORCE_INLINE_FUNCTION
    static constexpr __rv
    __impl() noexcept {
        return false;
    }
};

// 模板结构体 __fold_right_or_impl_ 的部分特化，处理一个及多个参数的情况
template <class Arg, class... Args>
struct __fold_right_or_impl_<Arg, Args...> {
    // 嵌套类型别名 __next_t 表示下一个递归结构
    using __next_t = __fold_right_or_impl_<Args...>;
    // 嵌套类型别名 __rv 为 Arg 和下一个递归结构的逻辑或操作结果的类型
    using __rv = decltype(std::declval<Arg>() || std::declval<typename __next_t::__rv>());
    // 静态成员函数 __impl 实现右折叠的逻辑或操作
    MDSPAN_FORCE_INLINE_FUNCTION
    static constexpr __rv
    __impl(Arg&& arg, Args&&... args) noexcept {
        // 返回当前参数和下一个递归结构的逻辑或结果
        return ((Arg&&)arg) || __next_t::__impl((Args&&)args...);
    }
};

// 模板函数 __fold_right_or_impl 实现右折叠的逻辑或操作
template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_or_impl_<Args...>::__rv
__fold_right_or_impl(Args&&... args) {
    // 调用模板结构体 __fold_right_or_impl_ 的静态成员函数 __impl
    return __fold_right_or_impl_<Args...>::__impl((Args&&)args...);
}
// </editor-fold> end right or }}}2
// 返回折叠右折叠操作的结果，使用可变参数模板 Args... 和对应的实现类 __fold_right_or_impl_
constexpr typename __fold_right_or_impl_<Args...>::__rv
__fold_right_or_impl(Args&&... args) {
  return __fold_right_or_impl_<Args...>::__impl((Args&&)args...);
}

// </editor-fold> end right or }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right plus"> {{{2

// 右折叠加法的实现结构体模板，处理单一参数情况
template <class Arg>
struct __fold_right_plus_impl_<Arg> {
  using __rv = Arg&&;
  // 强制内联函数
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(Arg&& arg) noexcept {
    return (Arg&&)arg;
  }
};

// 右折叠加法的实现结构体模板，处理多参数情况
template <class Arg1, class Arg2, class... Args>
struct __fold_right_plus_impl_<Arg1, Arg2, Args...> {
  // 下一个递归实现结构体类型
  using __next_t = __fold_right_plus_impl_<Arg2, Args...>;
  // 结果类型为第一个参数类型与下一个递归实现结果类型相加后的类型
  using __rv = decltype(std::declval<Arg1>() + std::declval<typename __next_t::__rv>());
  // 强制内联函数
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(Arg1&& arg, Arg2&& arg2, Args&&... args) noexcept {
    // 返回第一个参数加上后续参数的递归加法结果
    return ((Arg1&&)arg) + __next_t::__impl((Arg2&&)arg2, (Args&&)args...);
  }
};

// 右折叠加法的函数模板，调用对应实现类的实现函数
template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_plus_impl_<Args...>::__rv
__fold_right_plus_impl(Args&&... args) {
  return __fold_right_plus_impl_<Args...>::__impl((Args&&)args...);
}

// </editor-fold> end right plus }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right times"> {{{2

// 右折叠乘法的实现结构体模板，处理单一参数情况
template <class Arg>
struct __fold_right_times_impl_<Arg> {
  using __rv = Arg&&;
  // 强制内联函数
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(Arg&& arg) noexcept {
    return (Arg&&)arg;
  }
};

// 右折叠乘法的实现结构体模板，处理多参数情况
template <class Arg1, class Arg2, class... Args>
struct __fold_right_times_impl_<Arg1, Arg2, Args...> {
  // 下一个递归实现结构体类型
  using __next_t = __fold_right_times_impl_<Arg2, Args...>;
  // 结果类型为第一个参数类型与下一个递归实现结果类型相乘后的类型
  using __rv = decltype(std::declval<Arg1>() * std::declval<typename __next_t::__rv>());
  // 强制内联函数
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(Arg1&& arg, Arg2&& arg2, Args&&... args) noexcept {
    // 返回第一个参数乘以后续参数的递归乘法结果
    return ((Arg1&&)arg) * __next_t::__impl((Arg2&&)arg2, (Args&&)args...);
  }
};

// 右折叠乘法的函数模板，调用对应实现类的实现函数
template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_times_impl_<Args...>::__rv
__fold_right_times_impl(Args&&... args) {
  return __fold_right_times_impl_<Args...>::__impl((Args&&)args...);
}

// </editor-fold> end right times }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="right assign"> {{{2

// 右折叠赋值的实现结构体模板，处理单一参数情况
template <class Arg>
struct __fold_right_assign_impl_<Arg> {
  using __rv = Arg&&;
  // 强制内联函数
  MDSPAN_FORCE_INLINE_FUNCTION
  static constexpr __rv
  __impl(Arg&& arg) noexcept {
    // 直接返回参数本身
    return (Arg&&)arg;
  }
}
    return (Arg&&)arg;


// 返回一个右值引用类型的参数 arg
return (Arg&&)arg;
// 结构模板 __fold_right_assign_impl_ 的特化，处理多个模板参数的右折叠赋值操作
template <class Arg1, class Arg2, class... Args>
struct __fold_right_assign_impl_ {
    // 下一个右折叠赋值操作的实现类类型
    using __next_t = __fold_right_assign_impl_<Arg2, Args...>;
    // 右折叠赋值操作结果的类型
    using __rv = decltype(std::declval<Arg1>() = std::declval<typename __next_t::__rv>());
    
    // 强制内联函数，执行右折叠赋值操作
    MDSPAN_FORCE_INLINE_FUNCTION
    static constexpr __rv
    __impl(Arg1&& arg, Arg2&& arg2, Args&&... args) noexcept {
        return ((Arg1&&)arg) = __next_t::__impl((Arg2&&)arg2, (Args&&)args...);
    }
};

// 函数模板 __fold_right_assign_impl 的实例化，用于启动右折叠赋值操作的过程
template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_right_assign_impl_<Args...>::__rv
__fold_right_assign_impl(Args&&... args) {
    return __fold_right_assign_impl_<Args...>::__impl((Args&&)args...);
}

// </editor-fold> end right assign }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="left assign"> {{{2

// 结构模板 __fold_left_assign_impl_，处理多个模板参数的左折叠赋值操作
template <class... Args>
struct __fold_left_assign_impl_;

// 特化：当只有一个模板参数时的左折叠赋值操作
template <class Arg>
struct __fold_left_assign_impl_<Arg> {
    // 左折叠赋值操作结果的类型
    using __rv = Arg&&;
    
    // 强制内联函数，执行左折叠赋值操作
    MDSPAN_FORCE_INLINE_FUNCTION
    static constexpr __rv
    __impl(Arg&& arg) noexcept {
        return (Arg&&)arg;
    }
};

// 特化：当有多个模板参数时的左折叠赋值操作
template <class Arg1, class Arg2, class... Args>
struct __fold_left_assign_impl_<Arg1, Arg2, Args...> {
    // 左折叠赋值操作结果的类型
    using __assign_result_t = decltype(std::declval<Arg1>() = std::declval<Arg2>());
    // 下一个左折叠赋值操作的实现类类型
    using __next_t = __fold_left_assign_impl_<__assign_result_t, Args...>;
    // 左折叠赋值操作结果的类型
    using __rv = typename __next_t::__rv;
    
    // 强制内联函数，执行左折叠赋值操作
    MDSPAN_FORCE_INLINE_FUNCTION
    static constexpr __rv
    __impl(Arg1&& arg, Arg2&& arg2, Args&&... args) noexcept {
        return __next_t::__impl(((Arg1&&)arg) = (Arg2&&)arg2, (Args&&)args...);
    }
};

// 函数模板 __fold_left_assign_impl 的实例化，用于启动左折叠赋值操作的过程
template <class... Args>
MDSPAN_FORCE_INLINE_FUNCTION
constexpr typename __fold_left_assign_impl_<Args...>::__rv
__fold_left_assign_impl(Args&&... args) {
    return __fold_left_assign_impl_<Args...>::__impl((Args&&)args...);
}

// </editor-fold> end left assign }}}2
//------------------------------------------------------------------------------

// 结束 ifdef 保护区域

// 函数模板 __fold_comma_impl，用于启动逗号操作的折叠
template <class... Args>
constexpr __mdspan_enable_fold_comma __fold_comma_impl(Args&&... args) noexcept { return { }; }

// 定义 __bools 结构体，用于处理多个布尔类型参数
template <bool... Bs>
struct __bools;

// 结束命名空间 MDSPAN_IMPL_STANDARD_NAMESPACE
} // __fold_compatibility_impl

// 结束命名空间 MDSPAN_IMPL_STANDARD_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

// 定义 _MDSPAN_FOLD_AND 宏，用于执行逻辑与操作的折叠
#  define _MDSPAN_FOLD_AND(...) MDSPAN_IMPL_STANDARD_NAMESPACE::__fold_compatibility_impl::__fold_right_and_impl((__VA_ARGS__)...)
// 定义 _MDSPAN_FOLD_OR 宏，用于执行逻辑或操作的折叠
#  define _MDSPAN_FOLD_OR(...) MDSPAN_IMPL_STANDARD_NAMESPACE::__fold_compatibility_impl::__fold_right_or_impl((__VA_ARGS__)...)
// 定义 _MDSPAN_FOLD_ASSIGN_LEFT 宏，用于执行左折叠赋值操作的折叠
#  define _MDSPAN_FOLD_ASSIGN_LEFT(INIT, ...) MDSPAN_IMPL_STANDARD_NAMESPACE::__fold_compatibility_impl::__fold_left_assign_impl(INIT, (__VA_ARGS__)...)
// 定义 _MDSPAN_FOLD_ASSIGN_RIGHT 宏，用于执行右折叠赋值操作的折叠
#  define _MDSPAN_FOLD_ASSIGN_RIGHT(PACK, ...) MDSPAN_IMPL_STANDARD_NAMESPACE::__fold_compatibility_impl::__fold_right_assign_impl((PACK)..., __VA_ARGS__)
// 定义宏_MDSPAN_FOLD_TIMES_RIGHT(PACK, ...)，调用右折叠乘法实现
#define _MDSPAN_FOLD_TIMES_RIGHT(PACK, ...) MDSPAN_IMPL_STANDARD_NAMESPACE::__fold_compatibility_impl::__fold_right_times_impl((PACK)..., __VA_ARGS__)
// 定义宏_MDSPAN_FOLD_PLUS_RIGHT(PACK, ...)，调用右折叠加法实现
#define _MDSPAN_FOLD_PLUS_RIGHT(PACK, ...) MDSPAN_IMPL_STANDARD_NAMESPACE::__fold_compatibility_impl::__fold_right_plus_impl((PACK)..., __VA_ARGS__)
// 定义宏_MDSPAN_FOLD_COMMA(...)，调用逗号操作实现
#define _MDSPAN_FOLD_COMMA(...) MDSPAN_IMPL_STANDARD_NAMESPACE::__fold_compatibility_impl::__fold_comma_impl((__VA_ARGS__)...)

// 定义宏_MDSPAN_FOLD_AND_TEMPLATE(...)，根据条件是否使用变量模板选择性地调用标准库中的模板
#define _MDSPAN_FOLD_AND_TEMPLATE(...) \
  _MDSPAN_TRAIT(std::is_same, __fold_compatibility_impl::__bools<(__VA_ARGS__)..., true>, __fold_compatibility_impl::__bools<true, (__VA_ARGS__)...>)

#endif

// </editor-fold> end fold expressions }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Variable template compatibility"> {{{1

#if _MDSPAN_USE_VARIABLE_TEMPLATES
// 如果使用变量模板，则使用标准库中的变量模板_TRAIT##_v
#  define _MDSPAN_TRAIT(TRAIT, ...) TRAIT##_v<__VA_ARGS__>
#else
// 否则，使用标准库中的模板_TRAIT<__VA_ARGS__>::value
#  define _MDSPAN_TRAIT(TRAIT, ...) TRAIT<__VA_ARGS__>::value
#endif

// </editor-fold> end Variable template compatibility }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Pre-C++14 constexpr"> {{{1

#if _MDSPAN_USE_CONSTEXPR_14
// 如果支持 C++14 constexpr，则定义为 constexpr
#  define _MDSPAN_CONSTEXPR_14 constexpr
// 用于 EDG 前端编译器的 bug 修复
#  ifdef __EDG__
#    define _MDSPAN_CONSTEXPR_14_DEFAULTED
#  else
// 否则，默认为 constexpr
#    define _MDSPAN_CONSTEXPR_14_DEFAULTED constexpr
#  endif
#else
// 如果不支持 C++14 constexpr，则定义为空
#  define _MDSPAN_CONSTEXPR_14
#  define _MDSPAN_CONSTEXPR_14_DEFAULTED
#endif

// </editor-fold> end Pre-C++14 constexpr }}}1
//==============================================================================
//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/macros.hpp

#include <cstddef> // size_t

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

// ElementType 默认访问器结构体定义
template <class ElementType>
struct default_accessor {

  using offset_policy = default_accessor; // 偏移策略为 default_accessor
  using element_type = ElementType; // 元素类型为 ElementType
  using reference = ElementType&; // 引用类型为 ElementType 的引用
  using data_handle_type = ElementType*; // 数据处理类型为 ElementType 指针

  // 默认构造函数，使用默认实现，并声明为 constexpr
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr default_accessor() noexcept = default;

  // 转换构造函数模板，允许从其他类型的 default_accessor 隐式转换
  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    /* requires */ (
      _MDSPAN_TRAIT(std::is_convertible, OtherElementType(*)[], element_type(*)[])
    )
  )
  MDSPAN_INLINE_FUNCTION
  // constexpr 构造函数
  constexpr default_accessor(default_accessor<OtherElementType>) noexcept {}

  // 偏移函数，返回偏移后的指针
  MDSPAN_INLINE_FUNCTION
  constexpr data_handle_type
  offset(data_handle_type p, size_t i) const noexcept {
    return p + i;
  }

  // 访问函数，返回引用，允许通过指针和索引进行访问
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return p[i];
  }

};

} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE
//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/default_accessor.hpp
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/full_extent_t.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// 包含命名空间 MDSPAN_IMPL_STANDARD_NAMESPACE
namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

// 定义结构体 full_extent_t，显式默认构造函数
struct full_extent_t { explicit full_extent_t() = default; };

// 定义 _MDSPAN_INLINE_VARIABLE 宏，并创建 constexpr full_extent 变量
_MDSPAN_INLINE_VARIABLE constexpr auto full_extent = full_extent_t{ };

} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/full_extent_t.hpp
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/mdspan.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/layout_right.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/trait_backports.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef MDSPAN_INCLUDE_EXPERIMENTAL_BITS_TRAIT_BACKPORTS_HPP_
#define MDSPAN_INCLUDE_EXPERIMENTAL_BITS_TRAIT_BACKPORTS_HPP_

#include <type_traits>
#include <utility> // integer_sequence

//==============================================================================
// <editor-fold desc="Variable template trait backports (e.g., is_void_v)"> {{{1

#ifdef _MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS

#if _MDSPAN_USE_VARIABLE_TEMPLATES
namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

#define _MDSPAN_BACKPORT_TRAIT(TRAIT) \
  // 定义模板变量 TRAIT##_v，用于获取 TRAIT<Args...>::value 的值
  template <class... Args> _MDSPAN_INLINE_VARIABLE constexpr auto TRAIT##_v = TRAIT<Args...>::value;

_MDSPAN_BACKPORT_TRAIT(is_assignable)
_MDSPAN_BACKPORT_TRAIT(is_constructible)
_MDSPAN_BACKPORT_TRAIT(is_convertible)
_MDSPAN_BACKPORT_TRAIT(is_default_constructible)
_MDSPAN_BACKPORT_TRAIT(is_trivially_destructible)
_MDSPAN_BACKPORT_TRAIT(is_same)
_MDSPAN_BACKPORT_TRAIT(is_empty)
_MDSPAN_BACKPORT_TRAIT(is_void)

#undef _MDSPAN_BACKPORT_TRAIT

} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif // _MDSPAN_USE_VARIABLE_TEMPLATES

#endif // _MDSPAN_NEEDS_TRAIT_VARIABLE_TEMPLATE_BACKPORTS

// </editor-fold> end Variable template trait backports (e.g., is_void_v) }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="integer sequence (ugh...)"> {{{1

#if !defined(_MDSPAN_USE_INTEGER_SEQUENCE) || !_MDSPAN_USE_INTEGER_SEQUENCE

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

// 定义整数序列模板
template <class T, T... Vals>
struct integer_sequence {
  // 返回序列的大小
  static constexpr size_t size() noexcept { return sizeof...(Vals); }
  using value_type = T;
};

// 使用 std::integer_sequence 作为 index_sequence 的别名模板
template <size_t... Vals>
using index_sequence = std::integer_sequence<size_t, Vals...>;

namespace __detail {

// 辅助模板，生成整数序列的实现
template <class T, T N, T I, class Result>
struct __make_int_seq_impl;

// 达到边界 N 时终止递归，返回生成的整数序列类型
template <class T, T N, T... Vals>
struct __make_int_seq_impl<T, N, N, integer_sequence<T, Vals...>>
{
  using type = integer_sequence<T, Vals...>;
};

// 递归生成整数序列的模板
template <class T, T N, T I, T... Vals>
struct __make_int_seq_impl<
  T, N, I, integer_sequence<T, Vals...>
> : __make_int_seq_impl<T, N, I+1, integer_sequence<T, Vals..., I>>
{ };

} // end namespace __detail

// 生成整数序列的模板别名
template <class T, T N>
using make_integer_sequence = typename __detail::__make_int_seq_impl<T, N, 0, integer_sequence<T>>::type;

// 使用 size_t 生成索引序列的别名模板
template <size_t N>
using make_index_sequence = typename __detail::__make_int_seq_impl<size_t, N, 0, integer_sequence<size_t>>::type;

// 为类型序列 T... 生成索引序列的别名模板
template <class... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif

// </editor-fold> end integer sequence (ugh...) }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="standard trait aliases"> {{{1

#if !defined(_MDSPAN_USE_STANDARD_TRAIT_ALIASES) || !_MDSPAN_USE_STANDARD_TRAIT_ALIASES

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {


注释：
以上代码片段是一个C++头文件，实现了一系列的模板和宏定义，用于提供对C++标准特性的后向兼容和支持。主要功能包括：

- 定义和实现了一系列的变量模板（如 `is_void_v`），用于获取对应类型特性的布尔值。
- 提供了整数序列（`integer_sequence`）和索引序列（`index_sequence`）模板，用于在编译时生成一系列整数。
- 实现了生成整数序列的辅助模板 `__make_int_seq_impl`，支持不同类型的整数序列生成。
- 最后一部分代码尚未注释，继续完善。
#define _MDSPAN_BACKPORT_TRAIT_ALIAS(TRAIT) \
  template <class... Args> using TRAIT##_t = typename TRAIT<Args...>::type;
// 定义一个宏，用于简化类型特征模板的别名，生成一个模板别名，将模板 TRAIT<Args...>::type 替换为 TRAIT##_t<Args...>

_MDSPAN_BACKPORT_TRAIT_ALIAS(remove_cv)
// 使用上述宏定义 remove_cv_t，表示去除 cv 修饰符后的类型

_MDSPAN_BACKPORT_TRAIT_ALIAS(remove_reference)
// 使用上述宏定义 remove_reference_t，表示去除引用后的类型

template <bool _B, class _T=void>
using enable_if_t = typename enable_if<_B, _T>::type;
// 定义一个模板别名 enable_if_t，用于根据 _B 的值选择类型 _T 或 void

#undef _MDSPAN_BACKPORT_TRAIT_ALIAS
// 取消之前定义的宏 _MDSPAN_BACKPORT_TRAIT_ALIAS

} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE
// 结束 MDSPAN_IMPL_STANDARD_NAMESPACE 命名空间

#endif
// 结束条件编译指令，结束 trait_backports.hpp 文件的包含

// </editor-fold> end standard trait aliases }}}1
//==============================================================================

#endif //MDSPAN_INCLUDE_EXPERIMENTAL_BITS_TRAIT_BACKPORTS_HPP_
// 结束条件编译指令，结束 trait_backports.hpp 文件的包含，并且指明条件编译符合

//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/trait_backports.hpp
//END_FILE_INCLUDE 表示 mdspan/include/experimental/__p0009_bits/trait_backports.hpp 文件包含结束
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/extents.hpp
//BEGIN_FILE_INCLUDE 表示 mdspan/include/experimental/__p0009_bits/extents.hpp 文件包含开始
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/dynamic_extent.hpp
//BEGIN_FILE_INCLUDE 表示 mdspan/include/experimental/__p0009_bits/dynamic_extent.hpp 文件包含开始
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER


#if defined(__cpp_lib_span)
#include <span>
#endif

#include <cstddef>  // size_t
#include <limits>   // numeric_limits

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
#if defined(__cpp_lib_span)
using std::dynamic_extent;
#else
_MDSPAN_INLINE_VARIABLE constexpr auto dynamic_extent = std::numeric_limits<size_t>::max();
#endif
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
// 在 MDSPAN_IMPL_STANDARD_NAMESPACE 命名空间中定义 dynamic_extent，如果支持 std::span，则使用 std::dynamic_extent，否则使用 std::numeric_limits<size_t>::max()

//==============================================================================================================
//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/dynamic_extent.hpp
//END_FILE_INCLUDE 表示 mdspan/include/experimental/__p0009_bits/dynamic_extent.hpp 文件包含结束
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/utility.hpp
//BEGIN_FILE_INCLUDE 表示 mdspan/include/experimental/__p0009_bits/utility.hpp 文件包含开始

#include <cstddef>
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

// type alias used for rank-based tag dispatch
//
// this is used to enable alternatives to constexpr if when building for C++14
//
template <std::size_t N>
using with_rank = std::integral_constant<std::size_t, N>;
// 定义一个类型别名 with_rank，用于基于排列标签分派

template <class I1, class I2>
// 定义了一个常量表达式函数，用于比较两个整数型变量是否相等
constexpr bool common_integral_compare(I1 x, I2 y)
{
  // 确保 I1 和 I2 都是整数类型，否则静态断言失败
  static_assert(std::is_integral<I1>::value and
                std::is_integral<I2>::value, "");

  // 使用 std::common_type_t<I1, I2> 获取两个整数类型的共同类型
  using I = std::common_type_t<I1, I2>;
  // 将 x 和 y 转换为共同类型 I 后比较它们的值是否相等并返回结果
  return static_cast<I>(x) == static_cast<I>(y);
}

// 定义了一个模板函数 rankwise_equal，用于比较两个对象的元素在给定维度上是否相等
template <class T1, class T2, class F>
constexpr bool rankwise_equal(with_rank<0>, const T1&, const T2&, F)
{
  // 当维度为 0 时，直接返回 true，表示相等
  return true;
}

// 定义了一个模板函数 rankwise_equal，用于比较两个对象的元素在给定维度上是否相等
template <std::size_t N, class T1, class T2, class F>
constexpr bool rankwise_equal(with_rank<N>, const T1& x, const T2& y, F func)
{
  bool match = true;

  // 遍历给定的维度 N，比较 x 和 y 对应维度上的元素是否相等
  for (std::size_t r = 0; r < N; r++) {
    // 使用 common_integral_compare 函数比较 x 和 y 在第 r 维度上的元素是否相等
    match = match && common_integral_compare(func(x, r), func(y, r));
  }

  // 返回比较结果
  return match;
}

// 匿名命名空间 detail 中定义了一个结构体，实现了一个 constexpr 运算符
constexpr struct
{
  // 模板运算符重载，接受一个对象 x 和一个索引 i，调用 x.extent(i) 函数返回结果
  template <class T, class I>
  constexpr auto operator()(const T& x, I i) const
  {
    return x.extent(i);
  }
} extent;

// 匿名命名空间 detail 中定义了一个结构体，实现了一个 constexpr 运算符
constexpr struct
{
  // 模板运算符重载，接受一个对象 x 和一个索引 i，调用 x.stride(i) 函数返回结果
  template <class T, class I>
  constexpr auto operator()(const T& x, I i) const
  {
    return x.stride(i);
  }
} stride;

// 结束命名空间 detail
} // namespace detail

// 匿名命名空间 MDSPAN_IMPL_STANDARD_NAMESPACE 中定义了一个结构体 mdspan_non_standard_tag
constexpr struct mdspan_non_standard_tag {
} mdspan_non_standard;

// 结束命名空间 MDSPAN_IMPL_STANDARD_NAMESPACE
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

// 包含 MDSPAN_IMPL_STANDARD_NAMESPACE 命名空间中的头文件 __p0009_bits/utility.hpp
//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/utility.hpp

// 如果定义了 __cpp_lib_span，包含 <span> 头文件
#ifdef __cpp_lib_span
#include <span>
#endif

// 包含 <array>、<type_traits>、<cassert>、<cinttypes> 头文件
#include <array>
#include <type_traits>
#include <cassert>
#include <cinttypes>

// 定义了命名空间 MDSPAN_IMPL_STANDARD_NAMESPACE 中的 detail 命名空间
namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

// 定义了一个函数模板 __check_compatible_extents，用于检查两个整数序列是否兼容
template <size_t... Extents, size_t... OtherExtents>
static constexpr std::integral_constant<bool, false> __check_compatible_extents(
    std::integral_constant<bool, false>,
    std::integer_sequence<size_t, Extents...>,
    std::integer_sequence<size_t, OtherExtents...>) noexcept {
  return {};
}

// 辅助结构 __compare_extent_compatible，用于比较两个大小是否兼容
template <size_t Lhs, size_t Rhs>
struct __compare_extent_compatible : std::integral_constant<bool,
     Lhs == dynamic_extent ||
     Rhs == dynamic_extent ||
     Lhs == Rhs>
{};

// 定义了一个函数模板 __check_compatible_extents，用于检查两个整数序列是否兼容
template <size_t... Extents, size_t... OtherExtents>
static constexpr std::integral_constant<
    bool, _MDSPAN_FOLD_AND(__compare_extent_compatible<Extents, OtherExtents>::value)>
__check_compatible_extents(
    std::integral_constant<bool, true>,
    std::integer_sequence<size_t, Extents...>,
    std::integer_sequence<size_t, OtherExtents...>) noexcept {
  return {};
}

// 定义了一个函数模板 are_valid_indices，用于检查给定的索引类型是否有效
template<class IndexType, class ... Arguments>
MDSPAN_INLINE_FUNCTION
static constexpr bool are_valid_indices() {
    return
      // 检查 Arguments 是否可以转换为 IndexType 类型
      _MDSPAN_FOLD_AND(std::is_convertible<Arguments, IndexType>::value) &&
      // 检查 IndexType 是否可以从 Arguments 构造而不抛出异常
      _MDSPAN_FOLD_AND(std::is_nothrow_constructible<IndexType, Arguments>::value);
}

// ------------------------------------------------------------------
// ------------ static_array ----------------------------------------
// ------------------------------------------------------------------

// 类似于数组的类，提供了静态值数组，包含 get 函数和 [] 运算符。
// Implementation of Static Array with recursive implementation of get.

// Template struct for static_array_impl with recursive specialization
template <size_t R, class T, T... Extents>
struct static_array_impl;

// Recursive template specialization for static_array_impl
template <size_t R, class T, T FirstExt, T... Extents>
struct static_array_impl<R, T, FirstExt, Extents...> {
  // Inline function to get the value at index r recursively
  MDSPAN_INLINE_FUNCTION
  constexpr static T get(size_t r) {
    // Check if r matches the current recursion level R, return FirstExt if true
    if (r == R)
      return FirstExt;
    else
      // Recursively call get() on the next recursion level
      return static_array_impl<R + 1, T, Extents...>::get(r);
  }

  // Template function to get the value at index r (constexpr version)
  template <size_t r>
  MDSPAN_INLINE_FUNCTION constexpr static T get() {
    // Use C++17 constexpr if to conditionally return based on r
#if MDSPAN_HAS_CXX_17
    if constexpr (r == R)
      return FirstExt;
    else
      return static_array_impl<R + 1, T, Extents...>::template get<r>();
#else
    // Fallback to non-constexpr if not using C++17
    get(r);
#endif
  }
};

// End the recursion for static_array_impl
template <size_t R, class T, T FirstExt>
struct static_array_impl<R, T, FirstExt> {
  // Inline function to get the value at any index (which is always FirstExt)
  MDSPAN_INLINE_FUNCTION
  constexpr static T get(size_t) { return FirstExt; }

  // Template function to get the value at any index (constexpr version)
  template <size_t>
  MDSPAN_INLINE_FUNCTION constexpr static T get() {
    return FirstExt;
  }
};

// Specialization to handle size 0, prevents recursion from starting
template <class T>
struct static_array_impl<0, T> {
  // Inline function to get the value at any index (returns default constructed T)
  MDSPAN_INLINE_FUNCTION
  constexpr static T get(size_t) { return T(); }

  // Template function to get the value at any index (constexpr version)
  template <size_t>
  MDSPAN_INLINE_FUNCTION constexpr static T get() {
    return T();
  }
};

// Struct representing a static array with variadic template parameters
template <class T, T... Values>
struct static_array : public static_array_impl<0, T, Values...> {
public:
  using value_type = T;

  // Inline function to return the size of the static array
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t size() { return sizeof...(Values); }
};

// ------------------------------------------------------------------
// ------------ index_sequence_scan ---------------------------------
// ------------------------------------------------------------------

// index_sequence_scan provides get(r) and get<r>() returning the sum of the first r-1 values.

// Recursive implementation for get in index_sequence_scan
template <size_t R, size_t... Values>
struct index_sequence_scan_impl;

template <size_t R, size_t FirstVal, size_t... Values>
struct index_sequence_scan_impl<R, FirstVal, Values...> {
  // Inline function to compute the sum of the first r-1 values
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t get(size_t r) {
    // If r exceeds R, recursively add FirstVal to the sum of the rest
    if (r > R)
      return FirstVal + index_sequence_scan_impl<R + 1, Values...>::get(r);
    else
      return 0; // Return 0 if r is less than or equal to R
  }
};

// End the recursion for index_sequence_scan_impl
template <size_t R, size_t FirstVal>
struct index_sequence_scan_impl<R, FirstVal> {
  // Inline function to get the value at any index (returns FirstVal if R > r, otherwise 0)
#if defined(__NVCC__) || defined(__NVCOMPILER) || defined(_MDSPAN_COMPILER_INTEL)
  // For specific compilers, handle a special case with signed integer comparison
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t get(size_t r) {
    return static_cast<int64_t>(R) > static_cast<int64_t>(r) ? FirstVal : 0;
  }
#else
  // Default case: return FirstVal if R > r, otherwise 0
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t get(size_t r) { return R > r ? FirstVal : 0; }
#endif
};

// Specialization for index_sequence_scan_impl when R is 0
template <>
struct index_sequence_scan_impl<0> {
  // Inline function to get the value at any index (always returns 0)
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t get(size_t) { return 0; }
};
// ------------------------------------------------------------------
// ------------ possibly_empty_array  -------------------------------
// ------------------------------------------------------------------

// array like class which provides get function and operator [], and
// has a specialization for the size 0 case.
// This is needed to make the maybe_static_array be truly empty, for
// all static values.

template <class T, size_t N> struct possibly_empty_array {
  T vals[N]{};  // Array to store elements of type T with size N

  // Get mutable reference to element at index r
  MDSPAN_INLINE_FUNCTION
  constexpr T &operator[](size_t r) { return vals[r]; }

  // Get const reference to element at index r
  MDSPAN_INLINE_FUNCTION
  constexpr const T &operator[](size_t r) const { return vals[r]; }
};

// Specialization for possibly_empty_array when N is 0
template <class T> struct possibly_empty_array<T, 0> {
  
  // Returns a default-constructed T for any index access
  MDSPAN_INLINE_FUNCTION
  constexpr T operator[](size_t) { return T(); }
  
  // Returns a default-constructed T for any index access (const version)
  MDSPAN_INLINE_FUNCTION
  constexpr const T operator[](size_t) const { return T(); }
};

// ------------------------------------------------------------------
// ------------ maybe_static_array ----------------------------------
// ------------------------------------------------------------------

// array like class which has a mix of static and runtime values but
// only stores the runtime values.
// The type of the static and the runtime values can be different.
// The position of a dynamic value is indicated through a tag value.
template <class TDynamic, class TStatic, TStatic dyn_tag, TStatic... Values>
struct maybe_static_array {

  static_assert(std::is_convertible<TStatic, TDynamic>::value, "maybe_static_array: TStatic must be convertible to TDynamic");
  static_assert(std::is_convertible<TDynamic, TStatic>::value, "maybe_static_array: TDynamic must be convertible to TStatic");

private:
  // Static values member
  using static_vals_t = static_array<TStatic, Values...>;  // Type alias for static_array with TStatic values
  constexpr static size_t m_size = sizeof...(Values);     // Number of static values
  constexpr static size_t m_size_dynamic =
      _MDSPAN_FOLD_PLUS_RIGHT((Values == dyn_tag), 0);     // Count of dynamic values based on dyn_tag

  // Dynamic values member
  _MDSPAN_NO_UNIQUE_ADDRESS possibly_empty_array<TDynamic, m_size_dynamic>
      m_dyn_vals;  // Array for dynamic values of type TDynamic with size m_size_dynamic

  // static mapping of indices to the position in the dynamic values array
  using dyn_map_t = index_sequence_scan_impl<0, static_cast<size_t>(Values == dyn_tag)...>;  // Type alias for index_sequence_scan_impl
// 定义公共部分
public:

  // 定义两种值类型：动态值和静态值
  using value_type = TDynamic;
  using static_value_type = TStatic;
  // 指定动态值的标记值
  constexpr static static_value_type tag_value = dyn_tag;

  // 默认构造函数
  constexpr maybe_static_array() = default;

  // 针对所有静态值的构造函数
  // TODO: 是否需要添加前置条件检查？
  MDSPAN_TEMPLATE_REQUIRES(class... Vals,
                           /* requires */ ((m_size_dynamic == 0) &&
                                           (sizeof...(Vals) > 0)))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(Vals...) : m_dyn_vals{} {}

  // 只接受动态值的构造函数
  MDSPAN_TEMPLATE_REQUIRES(class... DynVals,
                           /* requires */ (sizeof...(DynVals) ==
                                               m_size_dynamic &&
                                           m_size_dynamic > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(DynVals... vals)
      : m_dyn_vals{static_cast<TDynamic>(vals)...} {}


  // 从动态值的std::array构造
  MDSPAN_TEMPLATE_REQUIRES(class T, size_t N,
                           /* requires */ (N == m_size_dynamic && N > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::array<T, N> &vals) {
    for (size_t r = 0; r < N; r++)
      m_dyn_vals[r] = static_cast<TDynamic>(vals[r]);
  }

  // 当N为0时，从std::array构造
  MDSPAN_TEMPLATE_REQUIRES(class T, size_t N,
                           /* requires */ (N == m_size_dynamic && N == 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::array<T, N> &) : m_dyn_vals{} {}

#ifdef __cpp_lib_span
  // 从std::span构造，当N > 0时
  MDSPAN_TEMPLATE_REQUIRES(class T, size_t N,
                           /* requires */ (N == m_size_dynamic && N > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::span<T, N> &vals) {
    for (size_t r = 0; r < N; r++)
      m_dyn_vals[r] = static_cast<TDynamic>(vals[r]);
  }

  // 从std::span构造，当N为0时
  MDSPAN_TEMPLATE_REQUIRES(class T, size_t N,
                           /* requires */ (N == m_size_dynamic && N == 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::span<T, N> &) : m_dyn_vals{} {}
#endif

  // 从所有值构造的构造函数
  MDSPAN_TEMPLATE_REQUIRES(class... DynVals,
                           /* requires */ (sizeof...(DynVals) !=
                                               m_size_dynamic &&
                                           m_size_dynamic > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(DynVals... vals)
    : m_dyn_vals{} {
    static_assert((sizeof...(DynVals) == m_size), "Invalid number of values.");
    TDynamic values[m_size]{static_cast<TDynamic>(vals)...};
    for (size_t r = 0; r < m_size; r++) {
      // 获取静态值
      TStatic static_val = static_vals_t::get(r);
      // 如果静态值为动态标记，则将对应位置的动态值赋值为对应的值
      if (static_val == dyn_tag) {
        m_dyn_vals[dyn_map_t::get(r)] = values[r];
      }
      // 预置条件检查
#ifdef _MDSPAN_DEBUG
      else {
        assert(values[r] == static_cast<TDynamic>(static_val));
      }
#endif
    }
  }

  // 定义模板要求，T 为类型，N 为大小，要求 N 不等于 m_size_dynamic 且 m_size_dynamic 大于 0
  MDSPAN_TEMPLATE_REQUIRES(
      class T, size_t N,
      /* requires */ (N != m_size_dynamic && m_size_dynamic > 0))
  // 内联函数，constexpr 修饰，接受 std::array<T, N> 的引用作为参数
  MDSPAN_INLINE_FUNCTION
  constexpr maybe_static_array(const std::array<T, N> &vals) {
    // 静态断言，确保 N 等于 m_size，否则触发静态断言错误信息
    static_assert((N == m_size), "Invalid number of values.");
// Precondition check
#ifdef _MDSPAN_DEBUG
    // 断言：N（数组大小）应与 m_size 相等
    assert(N == m_size);
#endif
    // 遍历数组中的每个元素
    for (size_t r = 0; r < m_size; r++) {
      // 获取静态值
      TStatic static_val = static_vals_t::get(r);
      // 如果静态值等于动态标记 dyn_tag，则将 vals[r] 转换为动态类型存入 m_dyn_vals
      if (static_val == dyn_tag) {
        m_dyn_vals[dyn_map_t::get(r)] = static_cast<TDynamic>(vals[r]);
      }
// Precondition check
#ifdef _MDSPAN_DEBUG
      // 否则，断言：vals[r] 和 static_val 应该相等
      else {
        assert(static_cast<TDynamic>(vals[r]) ==
               static_cast<TDynamic>(static_val));
      }
#endif
    }
  }

#ifdef __cpp_lib_span
  // 如果支持 C++20 中的 span 类型
  MDSPAN_TEMPLATE_REQUIRES(
      class T, size_t N,
      /* requires */ (N != m_size_dynamic && m_size_dynamic > 0))
  MDSPAN_INLINE_FUNCTION
  // 构造函数：根据给定的 span<T, N> 对象初始化 maybe_static_array
  constexpr maybe_static_array(const std::span<T, N> &vals) {
    // 静态断言：N 应等于 m_size 或 m_size 应为动态维度
    static_assert((N == m_size) || (m_size == dynamic_extent));
#ifdef _MDSPAN_DEBUG
    // 断言：N 应等于 m_size
    assert(N == m_size);
#endif
    // 遍历 span 中的每个元素
    for (size_t r = 0; r < m_size; r++) {
      // 获取静态值
      TStatic static_val = static_vals_t::get(r);
      // 如果静态值等于动态标记 dyn_tag，则将 vals[r] 转换为动态类型存入 m_dyn_vals
      if (static_val == dyn_tag) {
        m_dyn_vals[dyn_map_t::get(r)] = static_cast<TDynamic>(vals[r]);
      }
#ifdef _MDSPAN_DEBUG
      // 否则，断言：vals[r] 和 static_val 应该相等
      else {
        assert(static_cast<TDynamic>(vals[r]) ==
               static_cast<TDynamic>(static_val));
      }
#endif
    }
  }
#endif

  // 访问函数
  // 返回第 r 个静态值
  MDSPAN_INLINE_FUNCTION
  constexpr static TStatic static_value(size_t r) { return static_vals_t::get(r); }

  // 返回第 r 个值（静态或动态）
  MDSPAN_INLINE_FUNCTION
  constexpr TDynamic value(size_t r) const {
    // 获取第 r 个静态值
    TStatic static_val = static_vals_t::get(r);
    // 如果静态值为 dyn_tag，则返回 m_dyn_vals 中相应索引的值；否则返回静态值的动态类型转换
    return static_val == dyn_tag ? m_dyn_vals[dyn_map_t::get(r)]
                                        : static_cast<TDynamic>(static_val);
  }
  // 重载操作符 []，返回第 r 个值
  MDSPAN_INLINE_FUNCTION
  constexpr TDynamic operator[](size_t r) const { return value(r); }


  // 观察函数
  // 返回 m_size（数组大小）
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t size() { return m_size; }
  // 返回 m_size_dynamic（动态维度大小）
  MDSPAN_INLINE_FUNCTION
  constexpr static size_t size_dynamic() { return m_size_dynamic; }
};

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

// ------------------------------------------------------------------
// ------------ extents ---------------------------------------------
// ------------------------------------------------------------------

// Class to describe the extents of a multi dimensional array.
// Used by mdspan, mdarray and layout mappings.
// See ISO C++ standard [mdspan.extents]

// 描述多维数组的维度的类
template <class IndexType, size_t... Extents> class extents {
public:
  // typedefs 用于使用的整数类型
  using index_type = IndexType;
  using size_type = std::make_unsigned_t<index_type>;
  using rank_type = size_t;

  // 静态断言：index_type 必须是有符号或无符号整数类型，不能是布尔类型
  static_assert(std::is_integral<index_type>::value && !std::is_same<index_type, bool>::value,
                MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::extents::index_type must be a signed or unsigned integer type");
private:
  // constexpr 成员变量，存储 Extents 的数量
  constexpr static rank_type m_rank = sizeof...(Extents);
  // constexpr 成员变量，存储动态 Extents 的数量
  constexpr static rank_type m_rank_dynamic =
      _MDSPAN_FOLD_PLUS_RIGHT((Extents == dynamic_extent), /* + ... + */ 0);

  // 使用 maybe_static_array 内部存储类型
  // IndexType 是元素类型，size_t 是动态维度的大小，Extents 是静态维度
  using vals_t =
      detail::maybe_static_array<IndexType, size_t, dynamic_extent, Extents...>;
  _MDSPAN_NO_UNIQUE_ADDRESS vals_t m_vals;

public:
  // 多维索引空间的观察器
  MDSPAN_INLINE_FUNCTION
  // 返回静态维度的数量
  constexpr static rank_type rank() noexcept { return m_rank; }
  MDSPAN_INLINE_FUNCTION
  // 返回动态维度的数量
  constexpr static rank_type rank_dynamic() noexcept { return m_rank_dynamic; }

  MDSPAN_INLINE_FUNCTION
  // 返回特定维度的大小
  constexpr index_type extent(rank_type r) const noexcept { return m_vals.value(r); }
  MDSPAN_INLINE_FUNCTION
  // 返回静态维度的大小
  constexpr static size_t static_extent(rank_type r) noexcept {
    return vals_t::static_value(r);
  }

  // 构造函数
  // 默认构造函数
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr extents() noexcept = default;

  // 根据动态或所有值进行构造
  // 前提条件检查延迟到 maybe_static_array 构造函数
  MDSPAN_TEMPLATE_REQUIRES(
      class... OtherIndexTypes,
      /* requires */ (
          _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(std::is_convertible, OtherIndexTypes,
                                         index_type) /* && ... */) &&
          _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(std::is_nothrow_constructible, index_type,
                                         OtherIndexTypes) /* && ... */) &&
          (sizeof...(OtherIndexTypes) == m_rank ||
           sizeof...(OtherIndexTypes) == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  constexpr explicit extents(OtherIndexTypes... dynvals) noexcept
      : m_vals(static_cast<index_type>(dynvals)...) {}

  // 使用 std::array 进行构造
  MDSPAN_TEMPLATE_REQUIRES(
      class OtherIndexType, size_t N,
      /* requires */
      (
          _MDSPAN_TRAIT(std::is_convertible, const OtherIndexType&, index_type) &&
          _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type,
              const OtherIndexType&) &&
          (N == m_rank || N == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  // 显式构造函数，条件取决于 N 是否等于 m_rank_dynamic
  MDSPAN_CONDITIONAL_EXPLICIT(N != m_rank_dynamic)
  constexpr extents(const std::array<OtherIndexType, N> &exts) noexcept
      : m_vals(std::move(exts)) {}

#ifdef __cpp_lib_span
  // 使用 std::span 进行构造
  MDSPAN_TEMPLATE_REQUIRES(
      class OtherIndexType, size_t N,
      /* requires */
      (_MDSPAN_TRAIT(std::is_convertible, const OtherIndexType&, index_type) &&
       _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, const OtherIndexType&) &&
       (N == m_rank || N == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  // 显式构造函数，条件取决于 N 是否等于 m_rank_dynamic
  MDSPAN_CONDITIONAL_EXPLICIT(N != m_rank_dynamic)
  constexpr extents(const std::span<OtherIndexType, N> &exts) noexcept
      : m_vals(std::move(exts)) {}
#endif
private:
  // 从其他 extents 构造 extents 存储的函数
  // 在 C++ 17 中，可以使用 if constexpr 将前两个变体合并，
  // 在这种情况下，不需要所有的 requires 子句。
  // 在 C++ 14 模式下，由于无限递归，这种方式不起作用。
  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R < m_rank) && (static_extent(R) == dynamic_extent)))
  MDSPAN_INLINE_FUNCTION
  constexpr
  vals_t __construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &exts,
                                       DynamicValues... dynamic_values) noexcept {
    return __construct_vals_from_extents(
        std::integral_constant<size_t, DynCount + 1>(),
        std::integral_constant<size_t, R + 1>(), exts, dynamic_values...,
        exts.extent(R));
  }

  // 从其他 extents 构造 extents 存储的函数，用于静态 extents
  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R < m_rank) && (static_extent(R) != dynamic_extent)))
  MDSPAN_INLINE_FUNCTION
  constexpr
  vals_t __construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &exts,
                                       DynamicValues... dynamic_values) noexcept {
    return __construct_vals_from_extents(
        std::integral_constant<size_t, DynCount>(),
        std::integral_constant<size_t, R + 1>(), exts, dynamic_values...);
  }

  // 从其他 extents 构造 extents 存储的函数，用于最后的静态 extents
  MDSPAN_TEMPLATE_REQUIRES(
      size_t DynCount, size_t R, class OtherExtents, class... DynamicValues,
      /* requires */ ((R == m_rank) && (DynCount == m_rank_dynamic)))
  MDSPAN_INLINE_FUNCTION
  constexpr
  vals_t __construct_vals_from_extents(std::integral_constant<size_t, DynCount>,
                                       std::integral_constant<size_t, R>,
                                       const OtherExtents &,
                                       DynamicValues... dynamic_values) noexcept {
    return vals_t{static_cast<index_type>(dynamic_values)...};
  }

public:

  // 从其他 extents 的特化版本进行转换的构造函数
    // MDSPAN_TEMPLATE_REQUIRES 宏，用于指定模板参数的约束条件
    MDSPAN_TEMPLATE_REQUIRES(
        // 模板参数：其他索引类型及其大小
        class OtherIndexType, size_t... OtherExtents,
        /* requires */
        (
            // 多阶段检查，防止在大小不匹配时发生无效的包展开错误
            decltype(detail::__check_compatible_extents(
              // 使用 std::integral_constant<bool, extents<int, Extents...>::rank() == extents<int, OtherExtents...>::rank()> 作为第二个参数在 MSVC+NVCC 中会导致某些复杂的扩展错误
              // MSVC: 19.38.33133 NVCC: 12.0
              std::integral_constant<bool, extents<int, Extents...>::rank() == extents<int, OtherExtents...>::rank()>{},
              std::integer_sequence<size_t, Extents...>{},
              std::integer_sequence<size_t, OtherExtents...>{}))::value
      )
  )
  // MDSPAN_INLINE_FUNCTION 宏，定义内联函数
  MDSPAN_INLINE_FUNCTION
  // MDSPAN_CONDITIONAL_EXPLICIT 宏，条件显式化，用于确定 extents 类的构造函数是否显式
  MDSPAN_CONDITIONAL_EXPLICIT((((Extents != dynamic_extent) &&
                                (OtherExtents == dynamic_extent)) ||
                               ...) ||
                              (std::numeric_limits<index_type>::max() <
                               std::numeric_limits<OtherIndexType>::max()))
  // 构造函数：从另一个 extents 实例构造当前 extents 实例
  constexpr extents(const extents<OtherIndexType, OtherExtents...> &other) noexcept
      : m_vals(__construct_vals_from_extents(
            std::integral_constant<size_t, 0>(),
            std::integral_constant<size_t, 0>(), other)) {}

  // 比较运算符的友元模板函数
  template <class OtherIndexType, size_t... OtherExtents>
  // MDSPAN_INLINE_FUNCTION 宏，定义内联函数
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator==(const extents &lhs,
             const extents<OtherIndexType, OtherExtents...> &rhs) noexcept {
    // 返回比较结果：当前 extents 的秩等于另一个 extents 的秩，并且在秩逐个比较时相等
    return
      rank() == extents<OtherIndexType, OtherExtents...>::rank() &&
      detail::rankwise_equal(detail::with_rank<rank()>{}, rhs, lhs, detail::extent);
  }
#if !(MDSPAN_HAS_CXX_20)
  // 如果不支持 C++20，则定义友元函数 operator!= 用于比较 extents 对象的不等性
  template <class OtherIndexType, size_t... OtherExtents>
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator!=(extents const &lhs,
             extents<OtherIndexType, OtherExtents...> const &rhs) noexcept {
    return !(lhs == rhs);  // 返回两个 extents 对象是否不相等的结果
  }
#endif
};

// 递归辅助类，用于实现 extents 的别名 dextents
namespace detail {

template <class IndexType, size_t Rank,
          class Extents = ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType>>
struct __make_dextents;

template <class IndexType, size_t Rank, size_t... ExtentsPack>
struct __make_dextents<
    IndexType, Rank, ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>>
{
  // 定义类型 type 作为 dextents 的实现，Rank 减小 1
  using type = typename __make_dextents<
      IndexType, Rank - 1,
      ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType,
                                                ::MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent,
                                                ExtentsPack...>>::type;
};

template <class IndexType, size_t... ExtentsPack>
struct __make_dextents<
    IndexType, 0, ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>>
{
  // 当 Rank 为 0 时，type 是 extents 类型本身
  using type = ::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>;
};

} // end namespace detail

// [mdspan.extents.dextents], 别名模板用于创建 dextents
template <class IndexType, size_t Rank>
using dextents = typename detail::__make_dextents<IndexType, Rank>::type;

// extents 的推导指南
#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
template <class... IndexTypes>
extents(IndexTypes...)
    -> extents<size_t,
               ((void) sizeof(IndexTypes), ::MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent)...>;
#endif

// 用于识别 extents 类型的辅助类型特征
namespace detail {

template <class T> struct __is_extents : ::std::false_type {};

template <class IndexType, size_t... ExtentsPack>
struct __is_extents<::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, ExtentsPack...>>
    : ::std::true_type {};

template <class T>
#if MDSPAN_HAS_CXX_17
inline
#else
static
#endif
constexpr bool __is_extents_v = __is_extents<T>::value;

template<class InputIndexType, class ExtentsIndexType>
MDSPAN_INLINE_FUNCTION
constexpr void
check_lower_bound(InputIndexType user_index,
                  ExtentsIndexType /* current_extent */,
                  std::true_type /* is_signed */)
{
  (void) user_index; // 防止未使用变量的警告
#ifdef _MDSPAN_DEBUG
  assert(static_cast<ExtentsIndexType>(user_index) >= 0);  // 检查用户索引是否大于等于 0
#endif
}

template<class InputIndexType, class ExtentsIndexType>
MDSPAN_INLINE_FUNCTION
constexpr void
check_lower_bound(InputIndexType /* user_index */,
                  ExtentsIndexType /* current_extent */,
                  std::false_type /* is_signed */)
{}

template<class InputIndexType, class ExtentsIndexType>
MDSPAN_INLINE_FUNCTION
constexpr void
check_upper_bound(InputIndexType user_index,
                  ExtentsIndexType current_extent)
{
  (void) user_index; // 防止未使用变量的警告
  (void) current_extent; // 防止未使用变量的警告
#ifdef _MDSPAN_DEBUG
  assert(static_cast<ExtentsIndexType>(user_index) < current_extent);
#endif
}

// 返回 true 以使用 AND 折叠而非逗号
// 在 CPP14 模式下，void 表达式的使用方式与 _MDSPAN_FOLD_AND 设置不兼容
template<class InputIndex, class ExtentsIndexType>
MDSPAN_INLINE_FUNCTION
constexpr bool
check_one_index(InputIndex user_index,
                ExtentsIndexType current_extent)
{
  // 检查下界
  check_lower_bound(user_index, current_extent,
    std::integral_constant<bool, std::is_signed<ExtentsIndexType>::value>{});
  // 检查上界
  check_upper_bound(user_index, current_extent);
  return true; // 返回 true
}

template<size_t ... RankIndices,
         class ExtentsIndexType, size_t ... Exts,
         class ... Indices>
MDSPAN_INLINE_FUNCTION
constexpr void
check_all_indices_helper(std::index_sequence<RankIndices...>,
                         const extents<ExtentsIndexType, Exts...>& exts,
                         Indices... indices)
{
  // 抑制关于语句无效的警告
  (void) _MDSPAN_FOLD_AND(
    (check_one_index(indices, exts.extent(RankIndices)))
  );
}

template<class ExtentsIndexType, size_t ... Exts,
         class ... Indices>
MDSPAN_INLINE_FUNCTION
constexpr void
check_all_indices(const extents<ExtentsIndexType, Exts...>& exts,
                  Indices... indices)
{
  // 调用辅助函数来检查所有索引
  check_all_indices_helper(std::make_index_sequence<sizeof...(Indices)>(),
                           exts, indices...);
}

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/extents.hpp
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/layout_stride.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// 根据合同 DE-NA0003525 与 NTESS，美国政府在本软件中保留特定权利
//
// Kokkos 的一部分，遵循 Apache License v2.0 与 LLVM Exceptions
// 有关许可信息，请参见 https://kokkos.org/LICENSE
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/compressed_pair.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// 根据合同 DE-NA0003525 与 NTESS，美国政府在本软件中保留特定权利
//
// Kokkos 的一部分，遵循 Apache License v2.0 与 LLVM Exceptions
// 有关许可信息，请参见 https://kokkos.org/LICENSE
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#if !defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
// 如果未定义宏_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，则执行以下内容
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/no_unique_address.hpp
// 开始包含文件: mdspan/include/experimental/__p0009_bits/no_unique_address.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// 声明开始

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

//==============================================================================

// 模板结构 __no_unique_address_emulation
// _T：类型参数，_Disambiguator：用于消除歧义的值，默认为0，_Enable：启用条件
template <class _T, size_t _Disambiguator = 0, class _Enable = void>
struct __no_unique_address_emulation {
  using __stored_type = _T;  // 定义存储类型为_T
  _T __v;  // 成员变量__v，类型为_T
  // 强制内联函数，返回常引用__v
  MDSPAN_FORCE_INLINE_FUNCTION constexpr _T const &__ref() const noexcept {
    return __v;
  }
  // 强制内联函数，返回可变引用__v
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 _T &__ref() noexcept {
    return __v;
  }
};

// 空情况的特化版本
// 当_T为空类型且_T是平凡析构的时候使用该特化版本
template <class _T, size_t _Disambiguator>
struct __no_unique_address_emulation<
    _T, _Disambiguator,
    std::enable_if_t<_MDSPAN_TRAIT(std::is_empty, _T) &&
                // 如果类型不是平凡析构的，则不使用该特化版本
                _MDSPAN_TRAIT(std::is_trivially_destructible, _T)>> :
#ifdef _MDSPAN_COMPILER_MSVC
    // 如果使用MSVC编译器，因为无法访问私有继承的公共静态成员函数，所以使用protected继承
    protected
#else
    // 其他情况下使用private继承，以避免直接访问_T的成员而应该使用__ref()函数
    private
#endif
    _T {
  using __stored_type = _T;  // 定义存储类型为_T
  // 强制内联函数，返回常引用，通过强制类型转换调用this指针
  MDSPAN_FORCE_INLINE_FUNCTION constexpr _T const &__ref() const noexcept {
    return *static_cast<_T const *>(this);
  }
  // 强制内联函数，返回可变引用，通过强制类型转换调用this指针
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 _T &__ref() noexcept {
    return *static_cast<_T *>(this);
  }


**注释：**
这段代码主要是在定义和实现了一个模板结构 `__no_unique_address_emulation`，用于模拟C++20中的`[[no_unique_address]]`属性。根据条件不同，有两个主要实现：一个是一般情况下的实现，另一个是针对空类型且平凡析构的特化版本。
  // 返回指向当前对象的指针，通过 static_cast 进行类型转换为 _T* 类型
  return *static_cast<_T *>(this);
}

MDSPAN_INLINE_FUNCTION_DEFAULTED
// 默认生成 noexcept 的默认构造函数
constexpr __no_unique_address_emulation() noexcept = default;

MDSPAN_INLINE_FUNCTION_DEFAULTED
// 默认生成 noexcept 的复制构造函数
constexpr __no_unique_address_emulation(
    __no_unique_address_emulation const &) noexcept = default;

MDSPAN_INLINE_FUNCTION_DEFAULTED
// 默认生成 noexcept 的移动构造函数
constexpr __no_unique_address_emulation(
    __no_unique_address_emulation &&) noexcept = default;

MDSPAN_INLINE_FUNCTION_DEFAULTED
// 默认生成 noexcept 的复制赋值运算符
_MDSPAN_CONSTEXPR_14_DEFAULTED __no_unique_address_emulation &
operator=(__no_unique_address_emulation const &) noexcept = default;

MDSPAN_INLINE_FUNCTION_DEFAULTED
// 默认生成 noexcept 的移动赋值运算符
_MDSPAN_CONSTEXPR_14_DEFAULTED __no_unique_address_emulation &
operator=(__no_unique_address_emulation &&) noexcept = default;

MDSPAN_INLINE_FUNCTION_DEFAULTED
// 默认生成 noexcept 的析构函数
~__no_unique_address_emulation() noexcept = default;

// 显式声明此构造函数不是引用，以确保复制或移动构造函数仍然被调用
MDSPAN_INLINE_FUNCTION
// 使用给定的引用 __v，生成一个显式的 constexpr 构造函数
explicit constexpr __no_unique_address_emulation(_T const& __v) noexcept : _T(__v) {}

MDSPAN_INLINE_FUNCTION
// 使用给定的右值引用 __v，生成一个显式的 constexpr 构造函数，并将 __v 移动到 _T 对象中
explicit constexpr __no_unique_address_emulation(_T&& __v) noexcept : _T(::std::move(__v)) {}
  return __t2_val;
}

// Empty _T2 specialization for __compressed_pair, when _T2 is empty.
template <class _T1, class _T2>
struct __compressed_pair<
    _T1, _T2,
    std::enable_if_t<!_MDSPAN_TRAIT(std::is_empty, _T1) && _MDSPAN_TRAIT(std::is_empty, _T2)>>
    : private _T2 {
  _T1 __t1_val{};
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 _T1 &__first() noexcept {
    return __t1_val;
  }
  MDSPAN_FORCE_INLINE_FUNCTION constexpr _T1 const &__first() const noexcept {
    return __t1_val;
  }
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 _T2 &__second() noexcept {
    return *static_cast<_T2 *>(this);
  }
  MDSPAN_FORCE_INLINE_FUNCTION constexpr _T2 const &__second() const noexcept {
    return *static_cast<_T2 const *>(this);
  }
};

#endif // !_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS

} // end namespace detail
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE
//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/no_unique_address.hpp
#endif



// 这是一个用于模拟没有唯一地址的情况下的压缩对（compressed pair）实现。
// 如果_T2不为空，这个结构将_T1和_T2组合在一起。
// 如果_T1为空，而_T2不为空，这个结构将_T2放在_T1的位置上。
// 如果_T2为空，而_T1不为空，这个结构将_T1放在_T2的位置上。
// 这个实现依赖于_T1和_T2是否为空的判断，以实现最小化空间占用。
    // 返回 __t2_val 成员变量的值
        return __t2_val;
      }
    
    // 默认构造函数，使用默认构造函数初始化 __compressed_pair 对象
      MDSPAN_INLINE_FUNCTION_DEFAULTED
      constexpr __compressed_pair() = default;
    
    // 拷贝构造函数，使用默认方式复制构造 __compressed_pair 对象
      MDSPAN_INLINE_FUNCTION_DEFAULTED
      constexpr __compressed_pair(__compressed_pair const &) = default;
    
    // 移动构造函数，使用默认方式移动构造 __compressed_pair 对象
      MDSPAN_INLINE_FUNCTION_DEFAULTED
      constexpr __compressed_pair(__compressed_pair &&) = default;
    
    // 拷贝赋值运算符，使用默认方式拷贝赋值 __compressed_pair 对象
      MDSPAN_INLINE_FUNCTION_DEFAULTED
      _MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
      operator=(__compressed_pair const &) = default;
    
    // 移动赋值运算符，使用默认方式移动赋值 __compressed_pair 对象
      MDSPAN_INLINE_FUNCTION_DEFAULTED
      _MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
      operator=(__compressed_pair &&) = default;
    
    // 默认析构函数，使用默认方式析构 __compressed_pair 对象
      MDSPAN_INLINE_FUNCTION_DEFAULTED
      ~__compressed_pair() = default;
    
    // 模板构造函数，接受两个参数 __t1 和 __t2，并分别初始化 _T1 和 __t2_val 成员变量
      template <class _T1Like, class _T2Like>
      MDSPAN_INLINE_FUNCTION constexpr __compressed_pair(_T1Like &&__t1, _T2Like &&__t2)
          : _T1((_T1Like &&) __t1), __t2_val((_T2Like &&) __t2) {}
```cpp`
};

// Second empty.
// 定义一个模板结构体 __compressed_pair，用于存储两个类型 _T1 和 _T2，当两者均非空时生效
template <class _T1, class _T2>
struct __compressed_pair<
    _T1, _T2,
    std::enable_if_t<!_MDSPAN_TRAIT(std::is_empty, _T1) && _MDSPAN_TRAIT(std::is_empty, _T2)>>
    : private _T2 {
  
  // 第一个数据成员 __t1_val，类型为 _T1，初始化为默认构造的值
  _T1 __t1_val{};

  // 返回第一个成员 __t1_val 的引用，内联函数
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 _T1 &__first() noexcept { return __t1_val; }

  // 返回第一个成员 __t1_val 的常量引用，内联函数
  MDSPAN_FORCE_INLINE_FUNCTION constexpr _T1 const &__first() const noexcept {
    return __t1_val;
  }

  // 返回第二个成员（基类 _T2 对象的引用），内联函数
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 _T2 &__second() noexcept {
    return *static_cast<_T2 *>(this);
  }

  // 返回第二个成员（基类 _T2 对象的常量引用），内联函数
  MDSPAN_FORCE_INLINE_FUNCTION constexpr _T2 const &__second() const noexcept {
    return *static_cast<_T2 const *>(this);
  }

  // 默认构造函数，内联函数
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair() = default;

  // 拷贝构造函数，内联函数
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair const &) = default;

  // 移动构造函数，内联函数
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair &&) = default;

  // 拷贝赋值运算符，内联函数
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  _MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair const &) = default;

  // 移动赋值运算符，内联函数
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  _MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair &&) = default;

  // 析构函数，内联函数
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__compressed_pair() = default;

  // 构造函数模板，接受两个参数 _T1Like 和 _T2Like，并分别转发给 __t1_val 和 _T2 的构造函数
  template <class _T1Like, class _T2Like>
  MDSPAN_INLINE_FUNCTION constexpr __compressed_pair(_T1Like &&__t1, _T2Like &&__t2)
      : _T2((_T2Like &&) __t2), __t1_val((_T1Like &&) __t1) {}
};

// Both empty.
// 定义一个模板结构体 __compressed_pair，用于存储两个类型 _T1 和 _T2，当两者均为空时生效
template <class _T1, class _T2>
struct __compressed_pair<
    _T1, _T2,
    std::enable_if_t<_MDSPAN_TRAIT(std::is_empty, _T1) && _MDSPAN_TRAIT(std::is_empty, _T2)>>
    // 使用 __no_unique_address_emulation 包装器来避免基类冲突
#ifdef _MDSPAN_COMPILER_MSVC
// MSVC 不允许在私有继承某个类型时，通过基类访问其公共静态成员函数
    : protected __no_unique_address_emulation<_T1, 0>,
      protected __no_unique_address_emulation<_T2, 1>
#else
    : private __no_unique_address_emulation<_T1, 0>,
      private __no_unique_address_emulation<_T2, 1>
#endif
{
  using __first_base_t = __no_unique_address_emulation<_T1, 0>;
  using __second_base_t = __no_unique_address_emulation<_T2, 1>;

  // 返回第一个成员（__no_unique_address_emulation<_T1, 0> 对象的引用），内联函数
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 _T1 &__first() noexcept {
    return this->__first_base_t::__ref();
  }

  // 返回第一个成员（__no_unique_address_emulation<_T1, 0> 对象的常量引用），内联函数
  MDSPAN_FORCE_INLINE_FUNCTION constexpr _T1 const &__first() const noexcept {
    return this->__first_base_t::__ref();
  }

  // 返回第二个成员（__no_unique_address_emulation<_T2, 1> 对象的引用），内联函数
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 _T2 &__second() noexcept {
    return this->__second_base_t::__ref();
  }

  // 返回第二个成员（__no_unique_address_emulation<_T2, 1> 对象的常量引用），内联函数
  MDSPAN_FORCE_INLINE_FUNCTION constexpr _T2 const &__second() const noexcept {
    return this->__second_base_t::__ref();
  }
    返回 this 指针调用 __second_base_t 类的 __ref() 成员函数的结果
  }

  // 默认构造函数，使用默认方式构造对象
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair() = default;

  // 拷贝构造函数，使用默认方式拷贝构造对象
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair const &) = default;

  // 移动构造函数，使用默认方式移动构造对象
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  constexpr __compressed_pair(__compressed_pair &&) = default;

  // 拷贝赋值运算符，使用默认方式拷贝赋值对象
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  _MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair const &) = default;

  // 移动赋值运算符，使用默认方式移动赋值对象
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  _MDSPAN_CONSTEXPR_14_DEFAULTED __compressed_pair &
  operator=(__compressed_pair &&) = default;

  // 析构函数，使用默认方式析构对象
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__compressed_pair() = default;

  // 模板构造函数，根据给定的 _T1Like 和 _T2Like 参数构造对象
  template <class _T1Like, class _T2Like>
  MDSPAN_INLINE_FUNCTION constexpr __compressed_pair(_T1Like &&__t1, _T2Like &&__t2) noexcept
    : __first_base_t(_T1((_T1Like &&) __t1)), // 使用 __t1 参数构造 __first_base_t 成员
      __second_base_t(_T2((_T2Like &&) __t2)) // 使用 __t2 参数构造 __second_base_t 成员
  { }
#if !defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
#endif


// 如果未定义 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，则执行以下代码段



#include <array>
#include <type_traits>
#include <utility>


// 包含标准库头文件 <array>, <type_traits>, <utility>



#ifdef __cpp_lib_span
#include <span>
#endif


// 如果支持 C++20 的 std::span 库，则包含头文件 <span>



#if defined(_MDSPAN_USE_CONCEPTS) && MDSPAN_HAS_CXX_20 && defined(__cpp_lib_concepts)
#  include <concepts>
#endif


// 如果定义了 _MDSPAN_USE_CONCEPTS，并且编译器支持 C++20 并且支持 std::concepts 库，则包含头文件 <concepts>



namespace MDSPAN_IMPL_STANDARD_NAMESPACE {


// 进入 MDSPAN_IMPL_STANDARD_NAMESPACE 命名空间



struct layout_left {
  template<class Extents>
  class mapping;
};
struct layout_right {
  template<class Extents>
  class mapping;
};


// 定义 layout_left 和 layout_right 结构体，每个结构体都包含一个模板类 mapping



namespace detail {
  template<class Layout, class Mapping>
  constexpr bool __is_mapping_of =
    std::is_same<typename Layout::template mapping<typename Mapping::extents_type>, Mapping>::value;


// 在 detail 命名空间中定义模板 __is_mapping_of，用于检查 Layout 和 Mapping 是否匹配



#if defined(_MDSPAN_USE_CONCEPTS) && MDSPAN_HAS_CXX_20
#  if !defined(__cpp_lib_concepts)
  namespace internal {
  namespace detail {
  template <typename _Tp, typename _Up>
  concept __same_as = std::is_same_v<_Tp, _Up>;
  } // namespace detail
  template <class T, class U>
  concept __same_as = detail::__same_as<T, U> && detail::__same_as<U, T>;
  } // namespace internal
#  endif


// 如果定义了 _MDSPAN_USE_CONCEPTS 并且支持 C++20，如果不支持 std::concepts，则定义内部概念 __same_as



  template<class M>
  concept __layout_mapping_alike = requires {
    requires __is_extents<typename M::extents_type>::value;
#if defined(__cpp_lib_concepts)
    { M::is_always_strided() } -> std::same_as<bool>;
    { M::is_always_exhaustive() } -> std::same_as<bool>;
    { M::is_always_unique() } -> std::same_as<bool>;
#else
    { M::is_always_strided() } -> internal::__same_as<bool>;
    { M::is_always_exhaustive() } -> internal::__same_as<bool>;
    { M::is_always_unique() } -> internal::__same_as<bool>;
#endif
    std::bool_constant<M::is_always_strided()>::value;
    std::bool_constant<M::is_always_exhaustive()>::value;
    std::bool_constant<M::is_always_unique()>::value;
  };


// 定义 __layout_mapping_alike 概念，要求 M 类型满足特定的要求，包括检查 is_always_strided()、is_always_exhaustive() 和 is_always_unique() 方法的返回类型



} // namespace detail


// 结束 detail 命名空间



struct layout_stride {
  template <class Extents>
  class mapping
#if !defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : private detail::__no_unique_address_emulation<
        detail::__compressed_pair<
          Extents,
          detail::possibly_empty_array<typename Extents::index_type, Extents::rank()>
        >
      >
#endif
  {


// 定义 layout_stride 结构体，包含模板类 mapping，在未定义 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS 时，基于 detail::__no_unique_address_emulation 进行特化



  public:
    using extents_type = Extents;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using layout_type = layout_stride;


// 公共成员声明，定义多个类型别名



    // This could be a `requires`, but I think it's better and clearer as a `static_assert`.
    static_assert(detail::__is_extents_v<Extents>,
                  MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::layout_stride::mapping must be instantiated with a specialization of " MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::extents.");


// 静态断言，确保 mapping 类被实例化时，Extents 类型满足 __is_extents_v 的要求



  private:


// 私有部分开始
    //----------------------------------------------------------------------------
    // 使用 __strides_storage_t 定义了一个可能为空的数组类型，其元素类型为 index_type，
    // 数组的维度由 extents_type::rank() 决定。
    using __strides_storage_t = detail::possibly_empty_array<index_type, extents_type::rank()>;
    
    // 使用 __member_pair_t 定义了一个压缩对（compressed pair），包含 extents_type 和
    // __strides_storage_t 两个成员，用来表示数组的维度信息和步长信息。
    using __member_pair_t = detail::__compressed_pair<extents_type, __strides_storage_t>;
```cpp`
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    // 如果定义了宏 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，使用 __member_pair_t 类型的成员，避免地址唯一性问题
    _MDSPAN_NO_UNIQUE_ADDRESS __member_pair_t __members;
#else
    // 否则，定义一个类型 __base_t，用于模拟没有唯一地址的情况，采用 detail::__no_unique_address_emulation<__member_pair_t> 类型
    using __base_t = detail::__no_unique_address_emulation<__member_pair_t>;
#endif

    // 定义一个常量函数，返回 __strides_storage_t 类型的引用，使用 noexcept 表示不会抛出异常
    MDSPAN_FORCE_INLINE_FUNCTION constexpr __strides_storage_t const&
    __strides_storage() const noexcept {
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      // 如果定义了宏 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，直接返回 __members 的 __second() 方法
      return __members.__second();
#else
      // 否则，返回通过 this->__base_t::__ref() 获取的 __second() 方法
      return this->__base_t::__ref().__second();
#endif
    }

    // 定义一个非 const 的常量函数，返回 __strides_storage_t 类型的引用，使用 noexcept 表示不会抛出异常
    MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 __strides_storage_t&
    __strides_storage() noexcept {
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      // 如果定义了宏 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，直接返回 __members 的 __second() 方法
      return __members.__second();
#else
      // 否则，返回通过 this->__base_t::__ref() 获取的 __second() 方法
      return this->__base_t::__ref().__second();
#endif
    }

    // 定义一个模板函数，接受 SizeType 类型和整数序列 Ep、Idx，返回 index_type 类型的值，使用 _MDSPAN_HOST_DEVICE 和 constexpr 修饰
    template<class SizeType, size_t ... Ep, size_t ... Idx>
    _MDSPAN_HOST_DEVICE
    constexpr index_type __get_size(::MDSPAN_IMPL_STANDARD_NAMESPACE::extents<SizeType, Ep...>,std::integer_sequence<size_t, Idx...>) const {
      // 使用 _MDSPAN_FOLD_TIMES_RIGHT 对 extents().extent(Idx) 的值进行右折叠，乘积求值
      return _MDSPAN_FOLD_TIMES_RIGHT( static_cast<index_type>(extents().extent(Idx)), 1 );
    }

    //----------------------------------------------------------------------------

    // 声明一个模板类 mapping，作为当前类的友元类
    template <class>
    friend class mapping;

    //----------------------------------------------------------------------------

    // 定义一个模板结构体 __deduction_workaround，用于解决在顶层参数化时，索引序列模板参数的不可推导问题
    template <class>
    struct __deduction_workaround;

    // 特化 __deduction_workaround 结构体，模板参数为 std::index_sequence<Idxs...>
    template <size_t... Idxs>
    struct __deduction_workaround<std::index_sequence<Idxs...>>
    {
      // 比较两个映射对象的 stride 是否相等，使用通用类型 common_t 进行比较
      template <class OtherExtents>
      MDSPAN_INLINE_FUNCTION
      static constexpr bool _eq_impl(mapping const& self, mapping<OtherExtents> const& other) noexcept {
        using common_t = std::common_type_t<index_type, typename OtherExtents::index_type>;
        // 检查所有索引位置上的 stride 是否相等，并返回比较结果
        return    _MDSPAN_FOLD_AND((static_cast<common_t>(self.stride(Idxs)) == static_cast<common_t>(other.stride(Idxs))) /* && ... */)
               && _MDSPAN_FOLD_AND((static_cast<common_t>(self.extents().extent(Idxs)) == static_cast<common_t>(other.extents().extent(Idxs))) /* || ... */);
      }
      
      // 比较两个映射对象的 stride 是否不相等，使用通用类型 common_t 进行比较
      template <class OtherExtents>
      MDSPAN_INLINE_FUNCTION
      static constexpr bool _not_eq_impl(mapping const& self, mapping<OtherExtents> const& other) noexcept {
        using common_t = std::common_type_t<index_type, typename OtherExtents::index_type>;
        // 检查所有索引位置上的 stride 是否不相等，并返回比较结果
        return    _MDSPAN_FOLD_OR((static_cast<common_t>(self.stride(Idxs)) != static_cast<common_t>(other.stride(Idxs))) /* || ... */)
               || _MDSPAN_FOLD_OR((static_cast<common_t>(self.extents().extent(Idxs)) != static_cast<common_t>(other.extents().extent(Idxs))) /* || ... */);
      }
    
      // 计算映射对象在给定索引位置上的操作结果
      template <class... Integral>
      MDSPAN_FORCE_INLINE_FUNCTION
      static constexpr size_t _call_op_impl(mapping const& self, Integral... idxs) noexcept {
        // 将每个索引乘以对应的 stride，并累加求和
        return _MDSPAN_FOLD_PLUS_RIGHT((idxs * self.stride(Idxs)), /* + ... + */ 0);
      }
    
      // 计算映射对象所需的 span 大小
      MDSPAN_INLINE_FUNCTION
      static constexpr size_t _req_span_size_impl(mapping const& self) noexcept {
        // 假设没有负的 stride；不确定是否可以假设这一点
        return __impl::_call_op_impl(self, (self.extents().template __extent<Idxs>() - 1)...) + 1;
      }
    
      // 使用其他映射对象填充 strides_storage_t 结构
      template<class OtherMapping>
      MDSPAN_INLINE_FUNCTION
      static constexpr const __strides_storage_t fill_strides(const OtherMapping& map) {
        return __strides_storage_t{static_cast<index_type>(map.stride(Idxs))...};
      }
    
      // 使用给定的 strides_storage_t 结构填充 strides_storage_t
      MDSPAN_INLINE_FUNCTION
      static constexpr const __strides_storage_t& fill_strides(const __strides_storage_t& s) {
        return s;
      }
    
      // 使用 std::array 填充 strides_storage_t 结构
      template<class IntegralType>
      MDSPAN_INLINE_FUNCTION
      static constexpr const __strides_storage_t fill_strides(const std::array<IntegralType,extents_type::rank()>& s) {
        return __strides_storage_t{static_cast<index_type>(s[Idxs])...};
      }
    
      // 使用非标准标签 mdspan_non_standard_tag 和 C 数组填充 strides_storage_t 结构
      template<class IntegralType>
      MDSPAN_INLINE_FUNCTION
      static constexpr const __strides_storage_t fill_strides(mdspan_non_standard_tag, const IntegralType (&s)[extents_type::rank()]) {
        return __strides_storage_t{static_cast<index_type>(s[Idxs])...};
      }
    }
#ifdef __cpp_lib_span
      // 如果支持 C++20 的 std::span，则定义静态 constexpr 函数 fill_strides
      template<class IntegralType>
      MDSPAN_INLINE_FUNCTION
      static constexpr const __strides_storage_t fill_strides(const std::span<IntegralType,extents_type::rank()>& s) {
        // 使用 std::span 中的元素填充 __strides_storage_t 结构体，返回结果
        return __strides_storage_t{static_cast<index_type>(s[Idxs])...};
      }
#endif

      // 定义静态 constexpr 函数 return_strides，返回长度为 extents_type::rank() 的 std::array
      MDSPAN_INLINE_FUNCTION
      static constexpr std::array<index_type, extents_type::rank()> return_strides(const __strides_storage_t& s) {
        // 使用 __strides_storage_t 结构体中的元素初始化 std::array，并返回
        return std::array<index_type, extents_type::rank()>{s[Idxs]...};
      }

      // 定义模板函数 __return_zero，返回常量 0
      template<size_t K>
      MDSPAN_INLINE_FUNCTION
      static constexpr size_t __return_zero() { return 0; }

      // 定义静态 constexpr 函数 __OFFSET，接受一个 Mapping 对象并返回其索引类型
      template<class Mapping>
      MDSPAN_INLINE_FUNCTION
      static constexpr typename Mapping::index_type
        __OFFSET(const Mapping& m) { return m(__return_zero<Idxs>()...); }
    };

    // 使用 std::make_index_sequence<Extents::rank()> 实例化 __deduction_workaround 并命名为 __impl
    using __impl = __deduction_workaround<std::make_index_sequence<Extents::rank()>>;

    // 定义静态 constexpr 函数 strides_storage，对于 rank 为 0 的情况返回空的 __strides_storage_t
    static constexpr __strides_storage_t strides_storage(detail::with_rank<0>) {
      return {};
    }
    // 定义模板函数 strides_storage，对于指定的非零 rank 返回计算后的 __strides_storage_t
    template <std::size_t N>
    static constexpr __strides_storage_t strides_storage(detail::with_rank<N>) {
      __strides_storage_t s{};

      extents_type e;
      index_type stride = 1;
      // 根据 extents_type 的 rank 计算每个维度的 stride，并存储在 s 中
      for(int r = static_cast<int>(extents_type::rank() - 1); r >= 0; r--) {
        s[r] = stride;
        stride *= e.extent(r);
      }

      return s;
    }

    //----------------------------------------------------------------------------

#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    // 如果定义了 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，则使用 __member_pair_t 初始化 mapping
    MDSPAN_INLINE_FUNCTION constexpr explicit
    mapping(__member_pair_t&& __m) : __members(::std::move(__m)) {}
#else
    // 否则使用 __base_t 初始化 mapping
    MDSPAN_INLINE_FUNCTION constexpr explicit
    mapping(__base_t&& __b) : __base_t(::std::move(__b)) {}
#endif

  public:

    //--------------------------------------------------------------------------------

    // 默认构造函数 mapping，根据编译器支持情况选择初始化方式
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping() noexcept
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __members{
#else
      : __base_t(__base_t{__member_pair_t(
#endif
          extents_type(),
          // 使用 strides_storage 计算的 stride 初始化 __strides_storage_t
          __strides_storage_t(strides_storage(detail::with_rank<extents_type::rank()>{}))
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
    {}

    // 默认拷贝构造函数 mapping，使用编译器生成的默认实现
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;
    // 定义一个宏 MDSPAN_TEMPLATE_REQUIRES，用于约束模板参数 IntegralTypes
    MDSPAN_TEMPLATE_REQUIRES(
      class IntegralTypes,
      /* requires */ (
        // 在 MSVC 19.32 中，无法在此处使用 index_type，需要使用 typename Extents::index_type
        // 错误 C2641: 无法为 'MDSPAN_IMPL_STANDARD_NAMESPACE::layout_stride::mapping' 推断模板参数
        // 检查 IntegralTypes 是否可以隐式转换为 Extents 类型的 index_type，使用 _MDSPAN_TRAIT 判断
        _MDSPAN_TRAIT(std::is_convertible, const std::remove_const_t<IntegralTypes>&, typename Extents::index_type) &&
        // 检查是否可以用 IntegralTypes 构造 typename Extents::index_type，使用 _MDSPAN_TRAIT 判断
        _MDSPAN_TRAIT(std::is_nothrow_constructible, typename Extents::index_type, const std::remove_const_t<IntegralTypes>&)
      )
    )
    // 定义一个内联的 constexpr 函数 MDSPAN_INLINE_FUNCTION，名称为 mapping
    MDSPAN_INLINE_FUNCTION
    constexpr
    // 函数 mapping 的声明，接受 extents_type 类型的引用 e 和 std::array<IntegralTypes, extents_type::rank()> 类型的引用 s
    mapping(
      extents_type const& e,
      std::array<IntegralTypes, extents_type::rank()> const& s
    ) noexcept
    // 如果定义了宏 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，则使用 __members 初始化器列表
      : __members{
    // 否则，使用 __base_t 初始化器列表，并在其中包含一个 __member_pair_t 初始化器列表
#else
      : __base_t(__base_t{__member_pair_t(
#endif
          // 根据给定的步长数组 s 填充 strides_storage_t 对象，并传递给构造函数
          e, __strides_storage_t(__impl::fill_strides(s))
    // 如果定义了宏 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，则关闭 __members 初始化器列表
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
    // 否则，关闭 __base_t 初始化器列表
#else
        )})
#endif
    {
      /*
       * TODO: 检查前提条件
       * - 对于所有的 i 在范围 [0, rank_) 中，s[i] > 0 必须为真。
       * - REQUIRED-SPAN-SIZE(e, s) 必须是 index_type 类型的可表示值 ([basic.fundamental])。
       * - 如果 rank_ 大于 0，则存在一个排列 P，使得对于范围 [1, rank_) 中的所有 i，s[ pi ] >= s[ pi − 1 ] * e.extent( pi − 1 ) 必须为真，
       *   其中 pi 是 P 的第 i 个元素。
       */
    }

    // 使用 MDSPAN_TEMPLATE_REQUIRES 宏，定义模板参数要求
    MDSPAN_TEMPLATE_REQUIRES(
      class IntegralTypes,
      /* requires */ (
        // MSVC 19.32 不允许在此处使用 index_type，而要求使用 typename Extents::index_type
        // error C2641: cannot deduce template arguments for 'MDSPAN_IMPL_STANDARD_NAMESPACE::layout_stride::mapping'
        _MDSPAN_TRAIT(std::is_convertible, const std::remove_const_t<IntegralTypes>&, typename Extents::index_type) &&
        _MDSPAN_TRAIT(std::is_nothrow_constructible, typename Extents::index_type, const std::remove_const_t<IntegralTypes>&)
      )
    )
    // 定义 constexpr 的 mapping 构造函数，使用 mdspan_non_standard_tag、extents_type 和 IntegralTypes 数组 s
    MDSPAN_INLINE_FUNCTION
    constexpr
    mapping(
      mdspan_non_standard_tag,
      extents_type const& e,
      IntegralTypes (&s)[extents_type::rank()]
    ) noexcept
    // 如果定义了宏 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，则使用 __members 初始化器列表
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __members{
    // 否则，使用 __base_t 初始化器列表，并在其中包含一个 __member_pair_t 初始化器列表
#else
      : __base_t(__base_t{__member_pair_t(
#endif
          // 根据给定的步长数组 s 和 mdspan_non_standard 标签，填充 strides_storage_t 对象，并传递给构造函数
          e, __strides_storage_t(__impl::fill_strides(mdspan_non_standard, s))
    // 如果定义了宏 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS，则关闭 __members 初始化器列表
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
    // 否则，关闭 __base_t 初始化器列表
#else
        )})
#endif
    {
      /*
       * TODO: 检查前提条件
       * - 对于所有的 i 在范围 [0, rank_) 中，s[i] > 0 必须为真。
       * - REQUIRED-SPAN-SIZE(e, s) 必须是 index_type 类型的可表示值 ([basic.fundamental])。
       * - 如果 rank_ 大于 0，则存在一个排列 P，使得对于范围 [1, rank_) 中的所有 i，s[ pi ] >= s[ pi − 1 ] * e.extent( pi − 1 ) 必须为真，
       *   其中 pi 是 P 的第 i 个元素。
       */
    }
    mapping(
      extents_type const& e,
      std::span<IntegralTypes, extents_type::rank()> const& s
    ) noexcept



    # 调用名为 mapping 的函数，接受两个参数 e 和 s，并声明为 noexcept
    mapping(
      # 参数 e 是一个常量引用，表示 extents_type 类型的对象
      extents_type const& e,
      # 参数 s 是一个常量引用，表示一个 std::span 对象，其元素类型是 IntegralTypes，长度为 extents_type::rank()
      std::span<IntegralTypes, extents_type::rank()> const& s
    ) noexcept
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __members{
#else
      : __base_t(__base_t{__member_pair_t(
#endif
          e, __strides_storage_t(__impl::fill_strides(s))
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
    {
      /*
       * TODO: check preconditions
       * - s[i] > 0 is true for all i in the range [0, rank_ ).
       * - REQUIRED-SPAN-SIZE(e, s) is a representable value of type index_type ([basic.fundamental]).
       * - If rank_ is greater than 0, then there exists a permutation P of the integers in the
       *   range [0, rank_), such that s[ pi ] >= s[ pi − 1 ] * e.extent( pi − 1 ) is true for
       *   all i in the range [1, rank_ ), where pi is the ith element of P.
       */
    }
#endif // __cpp_lib_span

#if !(defined(_MDSPAN_USE_CONCEPTS) && MDSPAN_HAS_CXX_20)
    // 定义模板条件：要求StridedLayoutMapping类具有特定的特性
    MDSPAN_TEMPLATE_REQUIRES(
      class StridedLayoutMapping,
      /* requires */ (
        // 检查是否可以从StridedLayoutMapping的extents_type类型构造extents_type类型
        _MDSPAN_TRAIT(std::is_constructible, extents_type, typename StridedLayoutMapping::extents_type) &&
        // 检查StridedLayoutMapping是否是指定类型的映射
        detail::__is_mapping_of<typename StridedLayoutMapping::layout_type, StridedLayoutMapping> &&
        // StridedLayoutMapping是否总是唯一的
        StridedLayoutMapping::is_always_unique() &&
        // StridedLayoutMapping是否总是分布的
        StridedLayoutMapping::is_always_strided()
      )
    )
#else
    // 如果支持概念和C++20，则使用requires子句
    template<class StridedLayoutMapping>
    requires(
         // 检查是否是指定布局映射的类别
         detail::__layout_mapping_alike<StridedLayoutMapping> &&
         // 检查是否可以从StridedLayoutMapping的extents_type类型构造extents_type类型
         _MDSPAN_TRAIT(std::is_constructible, extents_type, typename StridedLayoutMapping::extents_type) &&
         // StridedLayoutMapping是否总是唯一的
         StridedLayoutMapping::is_always_unique() &&
         // StridedLayoutMapping是否总是分布的
         StridedLayoutMapping::is_always_strided()
    )
#endif
    // 定义条件性显式构造函数：检查是否需要拷贝或移动构造
    MDSPAN_CONDITIONAL_EXPLICIT(
      !(std::is_convertible<typename StridedLayoutMapping::extents_type, extents_type>::value &&
       (detail::__is_mapping_of<layout_left, StridedLayoutMapping> ||
        detail::__is_mapping_of<layout_right, StridedLayoutMapping> ||
        detail::__is_mapping_of<layout_stride, StridedLayoutMapping>))
    ) // 需要两个括号，因为有逗号
    // 内联函数定义：构造函数，用于从另一个StridedLayoutMapping构造
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    mapping(StridedLayoutMapping const& other) noexcept // NOLINT(google-explicit-constructor)
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __members{
#else
      : __base_t(__base_t{__member_pair_t(
#endif
          // 使用other的extents()和填充其strides的存储来构造对象
          other.extents(), __strides_storage_t(__impl::fill_strides(other))
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
    {
      /*
       * TODO: check preconditions
       * - other.stride(i) > 0 is true for all i in the range [0, rank_ ).
       * - other.required_span_size() is a representable value of type index_type ([basic.fundamental]).
       * - OFFSET(other) == 0
       */
    }

    //--------------------------------------------------------------------------------

    // 默认定义内联函数：拷贝赋值运算符
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED
    mapping& operator=(mapping const&) noexcept = default;
    # 声明一个内联函数（在编译时进行替换优化的函数），返回成员函数返回类型为 extents_type 的常量引用
    MDSPAN_INLINE_FUNCTION constexpr const extents_type& extents() const noexcept {
    // 如果定义了 _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS 宏，则返回 __members 的第一个元素
    return __members.__first();
#else
    // 否则返回 this->__base_t::__ref() 的第一个元素
    return this->__base_t::__ref().__first();
#endif
    };

    // 返回当前对象的步长数组，其长度等于 Extents::rank()
    MDSPAN_INLINE_FUNCTION
    constexpr std::array< index_type, extents_type::rank() > strides() const noexcept {
      return __impl::return_strides(__strides_storage());
    }

    // 计算当前对象所需的跨度大小并返回，考虑各维度的范围和步长
    MDSPAN_INLINE_FUNCTION
    constexpr index_type required_span_size() const noexcept {
      index_type span_size = 1;
      for(unsigned r = 0; r < extents_type::rank(); r++) {
        // 如果任何一个维度的长度为零，立即返回零
        if(extents().extent(r)==0) return 0;
        // 计算跨度大小的累加值，考虑每一维度的范围和步长
        span_size += ( static_cast<index_type>(extents().extent(r) - 1 ) * __strides_storage()[r]);
      }
      return span_size;
    }

    // 对于给定的索引，调用操作符 () 返回相应位置的元素值，索引的数量必须等于 Extents::rank()
    MDSPAN_TEMPLATE_REQUIRES(
      class... Indices,
      /* requires */ (
        sizeof...(Indices) == Extents::rank() &&
        (detail::are_valid_indices<index_type, Indices...>())
      )
    )
    MDSPAN_FORCE_INLINE_FUNCTION
    constexpr index_type operator()(Indices... idxs) const noexcept {
#if ! defined(NDEBUG)
      // 在调试模式下，检查所有索引是否在有效范围内
      detail::check_all_indices(this->extents(), idxs...);
#endif // ! NDEBUG
      // 调用实际的操作符 () 实现，并返回结果转换为 index_type 类型
      return static_cast<index_type>(__impl::_call_op_impl(*this, static_cast<index_type>(idxs)...));
    }

    // 返回是否总是唯一的标志，始终返回 true
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    // 返回是否总是耗尽的标志，始终返回 false
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept {
      return false;
    }
    // 返回是否总是步进的标志，始终返回 true
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }

    // 返回是否当前对象总是唯一的标志，始终返回 true
    MDSPAN_INLINE_FUNCTION static constexpr bool is_unique() noexcept { return true; }

  private:
    // 计算当前对象是否对于非零跨度大小总是耗尽
    constexpr bool exhaustive_for_nonzero_span_size() const
    {
      return required_span_size() == __get_size(extents(), std::make_index_sequence<extents_type::rank()>());
    }

    // 辅助函数，根据不同的维度情况计算是否耗尽
    constexpr bool is_exhaustive_impl(detail::with_rank<0>) const
    {
      return true;
    }
    // 辅助函数，根据不同的维度情况计算是否耗尽
    constexpr bool is_exhaustive_impl(detail::with_rank<1>) const
    {
      if (required_span_size() != static_cast<index_type>(0)) {
        return exhaustive_for_nonzero_span_size();
      }
      return stride(0) == 1;
    }
    // 辅助函数，根据不同的维度情况计算是否耗尽
    template <std::size_t N>
    constexpr bool is_exhaustive_impl(detail::with_rank<N>) const
    {
      if (required_span_size() != static_cast<index_type>(0)) {
        return exhaustive_for_nonzero_span_size();
      }

      rank_type r_largest = 0;
      for (rank_type r = 1; r < extents_type::rank(); r++) {
        if (stride(r) > stride(r_largest)) {
          r_largest = r;
        }
      }
      for (rank_type r = 0; r < extents_type::rank(); r++) {
        if (extents().extent(r) == 0 && r != r_largest) {
          return false;
        }
      }
      return true;
    }

  public:
    // 返回当前对象是否总是耗尽的标志
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 bool is_exhaustive() const noexcept {
      return is_exhaustive_impl(detail::with_rank<extents_type::rank()>{});
    }
    # 定义静态内联函数，返回常量表达式 true，表示此函数总是返回 true
    MDSPAN_INLINE_FUNCTION static constexpr bool is_strided() noexcept { return true; }

    # 定义内联函数，返回在给定维度上的步长（stride）
    MDSPAN_INLINE_FUNCTION
    constexpr index_type stride(rank_type r) const noexcept {
      # 调用 __strides_storage() 函数获取步长数组，并返回第 r 维的步长
      return __strides_storage()[r];
    }
#if !(defined(_MDSPAN_USE_CONCEPTS) && MDSPAN_HAS_CXX_20)
// 如果不支持概念和 C++20，则使用旧版的 MDSPAN_TEMPLATE_REQUIRES 宏定义
  MDSPAN_TEMPLATE_REQUIRES(
    class StridedLayoutMapping,
    /* requires */ (
      // 要求 StridedLayoutMapping 是 StridedLayoutMapping::layout_type 的映射类型，
      // 并且它的秩与 extents_type::rank() 相同，并且总是按步进排布
      detail::__is_mapping_of<typename StridedLayoutMapping::layout_type, StridedLayoutMapping> &&
      (extents_type::rank() == StridedLayoutMapping::extents_type::rank()) &&
      StridedLayoutMapping::is_always_strided()
    )
  )
#else
// 否则，使用新的概念约束来定义模板
  template<class StridedLayoutMapping>
  requires(
       // 要求 StridedLayoutMapping 类型与 __layout_mapping_alike 的返回值相似，
       // 并且其秩与 extents_type::rank() 相同，并且总是按步进排布
       detail::__layout_mapping_alike<StridedLayoutMapping> &&
       (extents_type::rank() == StridedLayoutMapping::extents_type::rank()) &&
       StridedLayoutMapping::is_always_strided()
  )
#endif
  MDSPAN_INLINE_FUNCTION
  // 定义一个友元函数 operator==，用于比较 mapping 对象和 StridedLayoutMapping 对象是否相等
  friend constexpr bool operator==(const mapping& x, const StridedLayoutMapping& y) noexcept {
    return (x.extents() == y.extents()) &&
           (__impl::__OFFSET(y) == static_cast<typename StridedLayoutMapping::index_type>(0)) &&
           detail::rankwise_equal(detail::with_rank<extents_type::rank()>{}, x, y, detail::stride);
  }

  // 这部分并不是提案的技术内容，只是为了优化实现而添加的
  MDSPAN_TEMPLATE_REQUIRES(
    class OtherExtents,
    /* requires */ (
      // 要求 OtherExtents 的秩与 extents_type::rank() 相同
      (extents_type::rank() == OtherExtents::rank())
    )
  )
  MDSPAN_INLINE_FUNCTION
  // 定义一个友元函数 operator==，用于比较两个 mapping 对象是否相等
  friend constexpr bool operator==(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
    return __impl::_eq_impl(lhs, rhs);
  }

#if !MDSPAN_HAS_CXX_20
// 如果不支持 C++20，使用旧版的 MDSPAN_TEMPLATE_REQUIRES 宏定义
  MDSPAN_TEMPLATE_REQUIRES(
    class StridedLayoutMapping,
    /* requires */ (
      // 要求 StridedLayoutMapping 是 StridedLayoutMapping::layout_type 的映射类型，
      // 并且其秩与 extents_type::rank() 相同，并且总是按步进排布
      detail::__is_mapping_of<typename StridedLayoutMapping::layout_type, StridedLayoutMapping> &&
      (extents_type::rank() == StridedLayoutMapping::extents_type::rank()) &&
      StridedLayoutMapping::is_always_strided()
    )
  )
  MDSPAN_INLINE_FUNCTION
  // 定义一个友元函数 operator!=，用于比较 mapping 对象和 StridedLayoutMapping 对象是否不相等
  friend constexpr bool operator!=(const mapping& x, const StridedLayoutMapping& y) noexcept {
    return not (x == y);
  }

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherExtents,
    /* requires */ (
      // 要求 OtherExtents 的秩与 extents_type::rank() 相同
      (extents_type::rank() == OtherExtents::rank())
    )
  )
  MDSPAN_INLINE_FUNCTION
  // 定义一个友元函数 operator!=，用于比较两个 mapping 对象是否不相等
  friend constexpr bool operator!=(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
    return __impl::_not_eq_impl(lhs, rhs);
  }
#endif

 // [mdspan.submdspan.mapping], submdspan mapping specialization
// 定义 submdspan_mapping_impl 函数模板，用于生成子映射
 template<class... SliceSpecifiers>
 MDSPAN_INLINE_FUNCTION
 constexpr auto submdspan_mapping_impl(
     SliceSpecifiers... slices) const;

// 定义 submdspan_mapping 函数模板，生成 mapping 对象的子映射
 template<class... SliceSpecifiers>
   friend constexpr auto submdspan_mapping(
     const mapping& src, SliceSpecifiers... slices) {
    return src.submdspan_mapping_impl(slices...);
  }
};
};

namespace detail {

// 模板函数 validate_strides 的部分特化，处理秩为 0 的情况
template <class Layout, class Extents, class Mapping>
constexpr void validate_strides(with_rank<0>, Layout, const Extents&, const Mapping&)
{}
constexpr void validate_strides(with_rank<N>, Layout, const Extents& ext, const Mapping& other)
{
  // 静态断言，验证 Mapping 类型的 layout_type 是否为 layout_stride，并且 Layout 是 layout_left 或 layout_right
  static_assert(std::is_same<typename Mapping::layout_type, layout_stride>::value and
                (std::is_same<Layout, layout_left>::value or
                 std::is_same<Layout, layout_right>::value)
                , "This function is only intended to validate construction of "
                  "a layout_left or layout_right mapping from a layout_stride mapping.");

  // 是否为 layout_left 的布尔值，用于后续的条件判断
  constexpr auto is_left = std::is_same<Layout, layout_left>::value;

  // 初始化步长为 1
  typename Extents::index_type stride = 1;

  // 循环遍历维度 N
  for (std::size_t r = 0; r < N; r++) {
    // 如果是 layout_left，则 s = r；否则 s = N - 1 - r
    const std::size_t s = is_left ? r : N - 1 - r;

    // 断言检查当前步长和 other 对象的步长是否一致，若不一致则抛出错误信息
    MDSPAN_IMPL_PRECONDITION(common_integral_compare(stride, other.stride(s))
                             and "invalid strides for layout_{left,right}");

    // 更新步长，乘以当前维度的大小
    stride *= ext.extent(s);
  }
}

} // namespace detail
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE
//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/layout_stride.hpp
#if MDSPAN_HAS_CXX_17
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p2642_bits/layout_padded_fwd.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <cassert>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {

template <size_t padding_value = dynamic_extent>
struct layout_left_padded {
  template <class _Extents>
  class mapping;
};

template <size_t padding_value = dynamic_extent>
struct layout_right_padded {
  template <class _Extents>
  class mapping;
};

namespace detail {
// The layout_padded_constants structs are only useful if rank > 1, otherwise they may wrap
template <class _Layout, class _ExtentsType>
struct layout_padded_constants;

template <class _ExtentsType, size_t _PaddingStride>
struct layout_padded_constants<layout_left_padded<_PaddingStride>, _ExtentsType>
{
  using rank_type = typename _ExtentsType::rank_type;
  static constexpr rank_type padded_stride_idx = 1;
  static constexpr rank_type extent_to_pad_idx = 0;
};

template <class _ExtentsType, size_t _PaddingStride>
struct layout_padded_constants<layout_right_padded<_PaddingStride>, _ExtentsType>
{
  using rank_type = typename _ExtentsType::rank_type;
  static constexpr rank_type padded_stride_idx = _ExtentsType::rank() - 2;
  static constexpr rank_type extent_to_pad_idx = _ExtentsType::rank() - 1;
};

template <class _Layout>
struct is_layout_left_padded : std::false_type {};

template <size_t _PaddingStride>
// 如果给定的 _Mapping 是 layout_left_padded 类型，则继承自 std::true_type
struct is_layout_left_padded<layout_left_padded<_PaddingStride>> : std::true_type {};

// 对于任何 _Mapping 类型，默认为 std::false_type
template <class _Mapping, class _Enabled = void>
struct is_layout_left_padded_mapping : std::false_type {};

// 如果 _Mapping 是 layout_left_padded 类型的特化，并且其 extents_type 类型与 _Mapping::extents_type 相同，则为 std::true_type
template <class _Mapping>
struct is_layout_left_padded_mapping<_Mapping,
  std::enable_if_t<std::is_same<_Mapping, typename layout_left_padded<_Mapping::padding_value>::template mapping<typename _Mapping::extents_type>>::value>>
    : std::true_type {};

// 如果给定的 _Layout 不是 layout_right_padded 类型，则继承自 std::false_type
template <class _Layout>
struct is_layout_right_padded : std::false_type {};

// 如果给定的 _Layout 是 layout_right_padded 类型，则继承自 std::true_type
template <size_t _PaddingStride>
struct is_layout_right_padded<layout_right_padded<_PaddingStride>> : std::true_type {};

// 对于任何 _Mapping 类型，默认为 std::false_type
template <class _Mapping, class _Enabled = void>
struct is_layout_right_padded_mapping : std::false_type {};

// 如果 _Mapping 是 layout_right_padded 类型的特化，并且其 extents_type 类型与 _Mapping::extents_type 相同，则为 std::true_type
template <class _Mapping>
struct is_layout_right_padded_mapping<_Mapping,
  std::enable_if_t<std::is_same<_Mapping, typename layout_right_padded<_Mapping::padding_value>::template mapping<typename _Mapping::extents_type>>::value>>
    : std::true_type {};

// 检查是否满足特定条件的模板函数，当 _LayoutExtentsType 的秩为 0 时调用
template <class _LayoutExtentsType, class _PaddedLayoutMappingType>
constexpr void check_padded_layout_converting_constructor_mandates(MDSPAN_IMPL_STANDARD_NAMESPACE::detail::with_rank<0>) {}

// 检查是否满足特定条件的模板函数，当 _LayoutExtentsType 的秩为 1 时调用
template <class _LayoutExtentsType, class _PaddedLayoutMappingType>
constexpr void check_padded_layout_converting_constructor_mandates(MDSPAN_IMPL_STANDARD_NAMESPACE::detail::with_rank<1>) {}

// 检查是否满足特定条件的模板函数，当 _LayoutExtentsType 的秩为 N 时调用，其中 N > 1
template <class _LayoutExtentsType, class _PaddedLayoutMappingType, std::size_t N>
constexpr void check_padded_layout_converting_constructor_mandates(MDSPAN_IMPL_STANDARD_NAMESPACE::detail::with_rank<N>)
{
  // 提取 _PaddedLayoutMappingType 的 extents_type 类型
  using extents_type = typename _PaddedLayoutMappingType::extents_type;
  // 提取 _PaddedLayoutMappingType 的 padding_value 常量
  constexpr auto padding_value = _PaddedLayoutMappingType::padding_value;
  // 计算 _LayoutExtentsType 中与 padding 相关的索引值
  constexpr auto idx = layout_padded_constants<typename _PaddedLayoutMappingType::layout_type, _LayoutExtentsType >::extent_to_pad_idx;

  // 静态确定性条件：要求 _LayoutExtentsType 和 extents_type 在 idx 处的静态大小不为动态大小（dynamic_extent），并且 padding_value 也不是动态大小
  constexpr auto statically_determinable =
    (_LayoutExtentsType::static_extent(idx) != dynamic_extent) &&
    (extents_type::static_extent(idx) != dynamic_extent) &&
    (padding_value != dynamic_extent);

  // 断言：如果静态确定性条件不满足，则输出空字符串，否则验证 _LayoutExtentsType 在 idx 处的静态大小是否符合特定模式
  static_assert(not statically_determinable or
                (padding_value == 0
                 ? _LayoutExtentsType::static_extent(idx) == 0
                 : _LayoutExtentsType::static_extent(idx) % padding_value == 0),
                "");
}

// 检查是否满足特定条件的模板函数，当 _ExtentsType 的秩为 0 时调用，但此处未使用 _ExtentsType 和 _OtherMapping 参数
template <typename _ExtentsType, typename _OtherMapping>
constexpr void check_padded_layout_converting_constructor_preconditions(MDSPAN_IMPL_STANDARD_NAMESPACE::detail::with_rank<0>,
                                                                        const _OtherMapping&) {}

// 检查是否满足特定条件的模板函数，当 _ExtentsType 的秩为 1 时调用，但此处未使用 _ExtentsType 和 _OtherMapping 参数
template <typename _ExtentsType, typename _OtherMapping>
constexpr void check_padded_layout_converting_constructor_preconditions(MDSPAN_IMPL_STANDARD_NAMESPACE::detail::with_rank<1>,
                                                                        const _OtherMapping&) {}

// 检查是否满足特定条件的模板函数，当 _ExtentsType 的秩为 N 时调用，其中 N > 1，但此处未使用 _ExtentsType 和 _OtherMapping 参数
template <typename _ExtentsType, typename _OtherMapping, std::size_t N>
// 检查使用标准命名空间中的指定秩和其他映射进行填充布局转换构造函数的前提条件
constexpr void check_padded_layout_converting_constructor_preconditions(MDSPAN_IMPL_STANDARD_NAMESPACE::detail::with_rank<N>,
                                                                        const _OtherMapping &other_mapping) {
  // 确定填充步长的索引
  constexpr auto padded_stride_idx =
    layout_padded_constants<typename _OtherMapping::layout_type,
                            _ExtentsType>::padded_stride_idx;
  // 确定需要填充的范围的索引
  constexpr auto extent_to_pad_idx = layout_padded_constants<typename _OtherMapping::layout_type, _ExtentsType>::extent_to_pad_idx;
  // 断言：其他映射的填充步长等于其范围中指定索引的大小
  MDSPAN_IMPL_PRECONDITION(other_mapping.stride(padded_stride_idx) == other_mapping.extents().extent(extent_to_pad_idx));
}

// 结束条件：文件包含结束，导入 mdspan/include/experimental/__p2642_bits/layout_padded_fwd.hpp 文件完成
// END_FILE_INCLUDE: mdspan/include/experimental/__p2642_bits/layout_padded_fwd.hpp
#endif

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

//==============================================================================
// layout_right 类的映射模板
template <class Extents>
class layout_right::mapping {
  public:
    // 类型别名定义
    using extents_type = Extents;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using layout_type = layout_right;

  private:
    // 静态断言：确保 extents_type 是 __is_extents_v 的专用实例
    static_assert(detail::__is_extents_v<extents_type>,
                  MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::layout_right::mapping must be instantiated with a specialization of " MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::extents.");

    template <class>
    friend class mapping;

    // 计算偏移量的模板函数
    // i0+(i1 + E(1)*(i2 + E(2)*i3))
    template <size_t r, size_t Rank>
    struct __rank_count {};

    // 递归计算偏移量的实现
    template <size_t r, size_t Rank, class I, class... Indices>
    _MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(
      index_type offset, __rank_count<r,Rank>, const I& i, Indices... idx) const {
      return __compute_offset(offset * __extents.extent(r) + i,__rank_count<r+1,Rank>(),  idx...);
    }

    // 计算偏移量的终止条件：到达最后一个维度
    template<class I, class ... Indices>
    _MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(
      __rank_count<0,extents_type::rank()>, const I& i, Indices... idx) const {
      return __compute_offset(i,__rank_count<1,extents_type::rank()>(),idx...);
    }

    // 计算偏移量的终止条件：处理最后一个维度
    _MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(size_t offset, __rank_count<extents_type::rank(), extents_type::rank()>) const {
      return static_cast<index_type>(offset);
    }

    // 计算偏移量的特化终止条件：处理零秩
    _MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(__rank_count<0,0>) const { return 0; }

  public:

    //--------------------------------------------------------------------------------

    // 默认构造函数
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping() noexcept = default;
    // 拷贝构造函数
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;

    // 构造函数：使用给定的范围初始化映射
    _MDSPAN_HOST_DEVICE
    constexpr mapping(extents_type const& __exts) noexcept
      :__extents(__exts)
    { }
    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        _MDSPAN_TRAIT(std::is_constructible, extents_type, OtherExtents)
      )
    )
    // 定义了一个模板需求，要求类型 OtherExtents 可以构造成 extents_type 类型
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible<OtherExtents, extents_type>::value)) // 需要两个 ()，因为有逗号
    // 如果 OtherExtents 不能隐式转换为 extents_type，则显式构造该映射对象
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    // 内联函数定义，constexpr 14，函数是 noexcept 的
    mapping(mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    {
       /*
        * TODO: check precondition
        * other.required_span_size() is a representable value of type index_type
        */
       // TODO: 检查前置条件，确保 other.required_span_size() 是 index_type 类型可以表示的值
    }

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        _MDSPAN_TRAIT(std::is_constructible, extents_type, OtherExtents) &&
        (extents_type::rank() <= 1)
      )
    )
    // 定义了另一个模板需求，要求类型 OtherExtents 可以构造成 extents_type 类型，并且 extents_type 的 rank 小于等于 1
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible<OtherExtents, extents_type>::value)) // 需要两个 ()，因为有逗号
    // 如果 OtherExtents 不能隐式转换为 extents_type，则显式构造该映射对象
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    // 内联函数定义，constexpr 14，函数是 noexcept 的
    mapping(layout_left::mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    {
       /*
        * TODO: check precondition
        * other.required_span_size() is a representable value of type index_type
        */
       // TODO: 检查前置条件，确保 other.required_span_size() 是 index_type 类型可以表示的值
    }

    /**
     * Converting constructor from `layout_right_padded::mapping`.
     *
     * This overload participates in overload resolution only if _Mapping is a layout_right_padded mapping and
     * extents_type is constructible from _Mapping::extents_type.
     *
     * \note There is currently a difference from p2642r2, where this function is specified as taking
     * `layout_right_padded< padding_value >::mapping< Extents>`. However, this makes `padding_value` non-deducible.
     */
    // 从 layout_right_padded::mapping 转换的构造函数
#if MDSPAN_HAS_CXX_17
    // 如果编译器支持 C++17
    MDSPAN_TEMPLATE_REQUIRES(
        class _Mapping,
        /* requires */ (
        MDSPAN_IMPL_PROPOSED_NAMESPACE::detail::is_layout_right_padded_mapping<_Mapping>::value
        && std::is_constructible_v<extents_type, typename _Mapping::extents_type>))
    // 如果 _Mapping 类型符合 is_layout_right_padded_mapping 要求，并且可以构造 extents_type
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible_v<typename _Mapping::extents_type, extents_type>))
    // 如果 _Mapping::extents_type 不能隐式转换为 extents_type
    mapping(const _Mapping &__other) noexcept
        : __extents(__other.extents())
    {
      // 使用提议的命名空间下的函数，检查填充布局的转换构造函数的必要条件
      MDSPAN_IMPL_PROPOSED_NAMESPACE::detail::
          check_padded_layout_converting_constructor_mandates<
            extents_type, _Mapping>(detail::with_rank<extents_type::rank()>{});
      // 使用提议的命名空间下的函数，检查填充布局的转换构造函数的前提条件
      MDSPAN_IMPL_PROPOSED_NAMESPACE::detail::
          check_padded_layout_converting_constructor_preconditions<
            extents_type>(detail::with_rank<extents_type::rank()>{}, __other);
    }
#endif

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        _MDSPAN_TRAIT(std::is_constructible, extents_type, OtherExtents)
      )
    )
    // 如果 extents_type 可以从 OtherExtents 构造
    MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    // 构造函数，从 layout_stride::mapping<OtherExtents> 构造， noexcept 表示不抛出异常
    mapping(layout_stride::mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    {
       /*
        * TODO: check precondition
        * other.required_span_size() is a representable value of type index_type
        */
       // 检查前提条件：other.required_span_size() 是 index_type 可表示的值
       detail::validate_strides(detail::with_rank<extents_type::rank()>{}, layout_right{}, __extents, other);
    }

    // 默认复制赋值运算符
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(mapping const&) noexcept = default;

    // 返回 extents 成员变量的常量引用
    MDSPAN_INLINE_FUNCTION
    constexpr const extents_type& extents() const noexcept {
      return __extents;
    }

    // 返回 required_span_size，即所有维度尺寸的乘积
    MDSPAN_INLINE_FUNCTION
    constexpr index_type required_span_size() const noexcept {
      index_type value = 1;
      for(rank_type r=0; r != extents_type::rank(); ++r) value*=__extents.extent(r);
      return value;
    }

    //--------------------------------------------------------------------------------

    MDSPAN_TEMPLATE_REQUIRES(
      class ... Indices,
      /* requires */ (
      (sizeof...(Indices) == extents_type::rank()) &&
      (detail::are_valid_indices<index_type, Indices...>())
      )
    )
    // 运算符重载，返回多维数组的线性偏移量
    _MDSPAN_HOST_DEVICE
    constexpr index_type operator()(Indices... idxs) const noexcept {
#if ! defined(NDEBUG)
      // 如果不是调试模式，检查所有索引是否有效
      detail::check_all_indices(this->extents(), idxs...);
#endif // ! NDEBUG
      // 计算多维索引对应的线性偏移量
      return __compute_offset(__rank_count<0, extents_type::rank()>(), static_cast<index_type>(idxs)...);
    }

    // 返回常量 true，表明总是唯一的
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    // 返回常量 true，表明总是穷尽的
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept { return true; }
    // 返回常量 true，表明总是分段的
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }
    # 返回常量值 true，表示该函数始终返回唯一值
    MDSPAN_INLINE_FUNCTION static constexpr bool is_unique() noexcept { return true; }
    
    # 返回常量值 true，表示该函数始终返回穷尽值
    MDSPAN_INLINE_FUNCTION static constexpr bool is_exhaustive() noexcept { return true; }
    
    # 返回常量值 true，表示该函数始终返回步进值
    MDSPAN_INLINE_FUNCTION static constexpr bool is_strided() noexcept { return true; }
    
    # 返回该对象的索引 `i` 对应的步进值，这是一个常量表达式函数
    constexpr index_type stride(rank_type i) const noexcept
#if MDSPAN_HAS_CXX_20
      requires ( Extents::rank() > 0 )
#endif
    {
      // 如果支持 C++ 20 并且 Extents 的秩大于 0，则执行以下代码块

      index_type value = 1;
      // 初始化 value 为 1，用于存储计算结果

      for(rank_type r=extents_type::rank()-1; r>i; r--) value*=__extents.extent(r);
      // 循环计算从 extents_type 的秩减一到 i 的各个维度的尺度乘积，并将结果赋给 value

      return value;
      // 返回计算结果
    }

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ ( Extents::rank() == OtherExtents::rank())
    )
    MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator==(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
      // 模板友元函数，比较两个 mapping 对象是否相等
      return lhs.extents() == rhs.extents();
      // 调用 extents() 方法比较左右操作数的 extents 对象是否相等，并返回比较结果
    }

    // In C++ 20 the not equal exists if equal is found
#if !(MDSPAN_HAS_CXX_20)
    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (Extents::rank() == OtherExtents::rank())
    )
    MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator!=(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
      // 如果不支持 C++ 20，定义 operator!= 模板友元函数，比较两个 mapping 对象是否不相等
      return lhs.extents() != rhs.extents();
      // 调用 extents() 方法比较左右操作数的 extents 对象是否不相等，并返回比较结果
    }
#endif

    // Not really public, but currently needed to implement fully constexpr useable submdspan:
    template<size_t N, class SizeType, size_t ... E, size_t ... Idx>
    constexpr index_type __get_stride(MDSPAN_IMPL_STANDARD_NAMESPACE::extents<SizeType, E...>,std::integer_sequence<size_t, Idx...>) const {
      // 私有成员函数，计算指定维度 N 的步长
      return _MDSPAN_FOLD_TIMES_RIGHT((Idx>N? __extents.template __extent<Idx>():1),1);
      // 调用宏 _MDSPAN_FOLD_TIMES_RIGHT 计算右折叠乘积，生成步长值
    }
    template<size_t N>
    constexpr index_type __stride() const noexcept {
      // 公有成员函数，返回指定维度 N 的步长
      return __get_stride<N>(__extents, std::make_index_sequence<extents_type::rank()>());
      // 调用私有成员函数 __get_stride，计算并返回维度 N 的步长值
    }

private:
   _MDSPAN_NO_UNIQUE_ADDRESS extents_type __extents{};

   // [mdspan.submdspan.mapping], submdspan mapping specialization
   template<class... SliceSpecifiers>
   MDSPAN_INLINE_FUNCTION
   constexpr auto submdspan_mapping_impl(
       SliceSpecifiers... slices) const;
   // 子类型映射实现函数声明

   template<class... SliceSpecifiers>
     friend constexpr auto submdspan_mapping(
       const mapping& src, SliceSpecifiers... slices) {
         return src.submdspan_mapping_impl(slices...);
     }
   // 友元函数模板，用于创建子类型映射
};

} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/layout_right.hpp

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
template <
  class ElementType,
  class Extents,
  class LayoutPolicy = layout_right,
  class AccessorPolicy = default_accessor<ElementType>
>
class mdspan
{
  // 省略部分 mdspan 类的定义
private:
  // 确保模板参数 Extents 是 extents 的特化类型
  static_assert(detail::__is_extents_v<Extents>,
                MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::mdspan's Extents template parameter must be a specialization of " MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::extents.");
  // 确保模板参数 ElementType 与 AccessorPolicy::element_type 相同
  static_assert(std::is_same<ElementType, typename AccessorPolicy::element_type>::value,
                MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::mdspan's ElementType template parameter must be the same as its AccessorPolicy::element_type.");

  // 用于解决在顶层模板参数中索引序列模板参数不可推导的问题
  template <class>
  struct __deduction_workaround;

  // 索引序列模板参数的特化
  template <size_t... Idxs>
  struct __deduction_workaround<std::index_sequence<Idxs...>>
  {
    // 计算映射的总大小，通过右折叠表达式计算
    MDSPAN_FORCE_INLINE_FUNCTION static constexpr
    size_t __size(mdspan const& __self) noexcept {
      return _MDSPAN_FOLD_TIMES_RIGHT((__self.__mapping_ref().extents().extent(Idxs)), /* * ... * */ size_t(1));
    }
    // 检查映射是否为空，通过右折叠表达式检查所有维度是否为零
    MDSPAN_FORCE_INLINE_FUNCTION static constexpr
    bool __empty(mdspan const& __self) noexcept {
      return (__self.rank()>0) && _MDSPAN_FOLD_OR((__self.__mapping_ref().extents().extent(Idxs)==index_type(0)));
    }
    // 访问映射中给定索引处的元素
    template <class ReferenceType, class SizeType, size_t N>
    MDSPAN_FORCE_INLINE_FUNCTION static constexpr
    ReferenceType __callop(mdspan const& __self, const std::array<SizeType, N>& indices) noexcept {
      return __self.__accessor_ref().access(__self.__ptr_ref(), __self.__mapping_ref()(indices[Idxs]...));
    }
    // 当支持 std::span 时，访问映射中给定索引处的元素
#ifdef __cpp_lib_span
    template <class ReferenceType, class SizeType, size_t N>
    MDSPAN_FORCE_INLINE_FUNCTION static constexpr
    ReferenceType __callop(mdspan const& __self, const std::span<SizeType, N>& indices) noexcept {
      return __self.__accessor_ref().access(__self.__ptr_ref(), __self.__mapping_ref()(indices[Idxs]...));
    }
#endif
  };
public:

  //--------------------------------------------------------------------------------
  // Domain and codomain types

  // 定义类型别名，表示范围和布局策略
  using extents_type = Extents;
  using layout_type = LayoutPolicy;
  using accessor_type = AccessorPolicy;
  // 使用布局策略定义映射类型
  using mapping_type = typename layout_type::template mapping<extents_type>;
  // 元素类型和值类型去除cv限定符
  using element_type = ElementType;
  using value_type = std::remove_cv_t<element_type>;
  // 索引类型和大小类型
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;
  // 数据处理句柄类型和引用类型
  using data_handle_type = typename accessor_type::data_handle_type;
  using reference = typename accessor_type::reference;

  // 返回维度的静态常量表达式
  MDSPAN_INLINE_FUNCTION static constexpr size_t rank() noexcept { return extents_type::rank(); }
  // 返回动态维度的数量
  MDSPAN_INLINE_FUNCTION static constexpr size_t rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  // 返回指定维度的静态范围
  MDSPAN_INLINE_FUNCTION static constexpr size_t static_extent(size_t r) noexcept { return extents_type::static_extent(r); }
  // 返回指定维度的动态范围
  MDSPAN_INLINE_FUNCTION constexpr index_type extent(size_t r) const noexcept { return __mapping_ref().extents().extent(r); };

private:

  // 因为 MSVC 存在 bug（警告 C4348），无法在 __deduction_workaround 模板中使用默认参数。
  // 使用 __deduction_workaround 模板定义内部实现类型
  using __impl = __deduction_workaround<std::make_index_sequence<extents_type::rank()>>;

  // 使用 __compressed_pair 包装映射类型和访问器类型
  using __map_acc_pair_t = detail::__compressed_pair<mapping_type, accessor_type>;

public:

  //--------------------------------------------------------------------------------
  // [mdspan.basic.cons], mdspan constructors, assignment, and destructor

  // 默认构造函数，如果不支持 C++20，则显式声明为默认
#if !MDSPAN_HAS_CXX_20
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdspan() = default;
#else
  // 在满足特定条件下使用默认构造函数
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdspan()
    requires(
       // nvhpc 存在 bug，这里不能仅使用 rank_dynamic()
       (extents_type::rank_dynamic() > 0) &&
       _MDSPAN_TRAIT(std::is_default_constructible, data_handle_type) &&
       _MDSPAN_TRAIT(std::is_default_constructible, mapping_type) &&
       _MDSPAN_TRAIT(std::is_default_constructible, accessor_type)
     ) = default;
#endif
  // 复制构造函数声明为默认
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdspan(const mdspan&) = default;
  // 移动构造函数声明为默认
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdspan(mdspan&&) = default;

  // 模板构造函数，用于根据给定的数据处理句柄和动态维度创建对象
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      // 维度数量必须等于静态维度数量或者动态维度数量
      ((sizeof...(SizeTypes) == rank()) || (sizeof...(SizeTypes) == rank_dynamic())) &&
      // 确保索引类型有效
      (detail::are_valid_indices<index_type, SizeTypes...>()) &&
      // 映射类型可构造
      _MDSPAN_TRAIT(std::is_constructible, mapping_type, extents_type) &&
      // 访问器类型可默认构造
      _MDSPAN_TRAIT(std::is_default_constructible, accessor_type)
    )
  )
  // 显式构造函数，使用数据处理句柄和动态维度创建对象
  MDSPAN_INLINE_FUNCTION
  explicit constexpr mdspan(data_handle_type p, SizeTypes... dynamic_extents)
    // TODO @proposal-bug shouldn't I be allowed to do `move(p)` here?
    // 构造函数，接受一个数据句柄 p 和一个大小为 N 的动态长度数组 dynamic_extents
      : __members(std::move(p), __map_acc_pair_t(mapping_type(extents_type(static_cast<index_type>(std::move(dynamic_extents))...)), accessor_type()))
      { }
    
    // 模板化的构造函数，要求模板参数 SizeType 和 N，以及一些编译时要求
    MDSPAN_TEMPLATE_REQUIRES(
        class SizeType, size_t N,
        /* requires */ (
          // SizeType 必须可以转换为 index_type 类型
          _MDSPAN_TRAIT(std::is_convertible, const SizeType&, index_type) &&
          // index_type 的构造必须是无异常的
          _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, const SizeType&) &&
          // N 必须等于 rank() 或者 rank_dynamic()
          ((N == rank()) || (N == rank_dynamic())) &&
          // mapping_type 必须可以用 extents_type 构造
          _MDSPAN_TRAIT(std::is_constructible, mapping_type, extents_type) &&
          // accessor_type 必须是默认可构造的
          _MDSPAN_TRAIT(std::is_default_constructible, accessor_type)
        )
      )
      // 如果 N 不等于 rank_dynamic()，则使用条件显式转换
      MDSPAN_CONDITIONAL_EXPLICIT(N != rank_dynamic())
      // 内联函数，constexpr 修饰，接受一个数据句柄 p 和一个大小为 N 的动态长度数组 dynamic_extents
      MDSPAN_INLINE_FUNCTION
      constexpr mdspan(data_handle_type p, const std::array<SizeType, N>& dynamic_extents)
        // 初始化成员 __members，使用 mapping_type 和 accessor_type 构造
        : __members(std::move(p), __map_acc_pair_t(mapping_type(extents_type(dynamic_extents)), accessor_type()))
      { }
#ifdef __cpp_lib_span
  // 如果支持 C++20 中的 std::span 类型

  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType, size_t N,
    /* requires */ (
      // 要求 SizeType 类型能转换为 index_type 类型
      _MDSPAN_TRAIT(std::is_convertible, const SizeType&, index_type) &&
      // 要求 index_type 类型可以从 SizeType 类型无异常构造
      _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, const SizeType&) &&
      // 要求 N 要么等于 rank()，要么等于 rank_dynamic()
      ((N == rank()) || (N == rank_dynamic())) &&
      // 要求 mapping_type 类型可以从 extents_type 类型构造
      _MDSPAN_TRAIT(std::is_constructible, mapping_type, extents_type) &&
      // 要求 accessor_type 类型可以默认构造
      _MDSPAN_TRAIT(std::is_default_constructible, accessor_type)
    )
  )

  MDSPAN_CONDITIONAL_EXPLICIT(N != rank_dynamic())
  // 在行内声明的函数 mdspan 的 constexpr 构造函数，接受一个数据句柄 p 和一个动态维度的 std::span<SizeType, N>
  MDSPAN_INLINE_FUNCTION
  constexpr mdspan(data_handle_type p, std::span<SizeType, N> dynamic_extents)
    : __members(std::move(p), __map_acc_pair_t(mapping_type(extents_type(as_const(dynamic_extents))), accessor_type()))
  { }
#endif

// 函数声明，要求 mdspan 构造函数必须是行内的、constexpr 的
MDSPAN_FUNCTION_REQUIRES(
  (MDSPAN_INLINE_FUNCTION constexpr),
  mdspan, (data_handle_type p, const extents_type& exts), ,
  /* requires */ (_MDSPAN_TRAIT(std::is_default_constructible, accessor_type) &&
                  _MDSPAN_TRAIT(std::is_constructible, mapping_type, const extents_type&))
)
// mdspan 的构造函数，接受一个数据句柄 p 和 extents_type 类型的 exts
: __members(std::move(p), __map_acc_pair_t(mapping_type(exts), accessor_type()))
{ }

// 函数声明，要求 mdspan 构造函数必须是行内的、constexpr 的
MDSPAN_FUNCTION_REQUIRES(
  (MDSPAN_INLINE_FUNCTION constexpr),
  mdspan, (data_handle_type p, const mapping_type& m), ,
  /* requires */ (_MDSPAN_TRAIT(std::is_default_constructible, accessor_type))
)
// mdspan 的构造函数，接受一个数据句柄 p 和 mapping_type 类型的 m
: __members(std::move(p), __map_acc_pair_t(m, accessor_type()))
{ }

// 行内声明的函数 mdspan 的 constexpr 构造函数，接受一个数据句柄 p、一个 mapping_type 类型的 m 和一个 accessor_type 类型的 a
MDSPAN_INLINE_FUNCTION
constexpr mdspan(data_handle_type p, const mapping_type& m, const accessor_type& a)
  : __members(std::move(p), __map_acc_pair_t(m, a))
{ }

// 如果满足以下条件，则声明函数 mdspan 的 constexpr 拷贝构造函数
MDSPAN_TEMPLATE_REQUIRES(
  class OtherElementType, class OtherExtents, class OtherLayoutPolicy, class OtherAccessor,
  /* requires */ (
    // 要求 mapping_type 类型可以从 typename OtherLayoutPolicy::template mapping<OtherExtents> 类型构造
    _MDSPAN_TRAIT(std::is_constructible, mapping_type, const typename OtherLayoutPolicy::template mapping<OtherExtents>&) &&
    // 要求 accessor_type 类型可以从 OtherAccessor 类型构造
    _MDSPAN_TRAIT(std::is_constructible, accessor_type, const OtherAccessor&)
  )
)
// 如果不满足以下条件，则声明函数 mdspan 的 constexpr 拷贝构造函数为 explicit
MDSPAN_CONDITIONAL_EXPLICIT(
  !_MDSPAN_TRAIT(std::is_convertible, const typename OtherLayoutPolicy::template mapping<OtherExtents>&, mapping_type) ||
  !_MDSPAN_TRAIT(std::is_convertible, const OtherAccessor&, accessor_type)
)
// 在行内声明的函数 mdspan 的 constexpr 拷贝构造函数，接受一个 mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherAccessor> 的引用 other
MDSPAN_INLINE_FUNCTION
constexpr mdspan(const mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherAccessor>& other)
  : __members(other.__ptr_ref(), __map_acc_pair_t(other.__mapping_ref(), other.__accessor_ref()))
  {
      // 静态断言，检查是否满足特定条件，如果条件不成立会编译失败
      static_assert(_MDSPAN_TRAIT(std::is_constructible, data_handle_type, typename OtherAccessor::data_handle_type),"Incompatible data_handle_type for mdspan construction");
      static_assert(_MDSPAN_TRAIT(std::is_constructible, extents_type, OtherExtents),"Incompatible extents for mdspan construction");
      /*
       * TODO: Check precondition
       * 对于每个秩索引 r，条件 static_extent(r) == dynamic_extent || static_extent(r) == other.extent(r) 必须成立。
       * 静态断言用于在编译时检查此条件。
       */
  }

  /* Might need this on NVIDIA?
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~mdspan() = default;
  */

  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mdspan& operator=(const mdspan&) = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mdspan& operator=(mdspan&&) = default;


  //--------------------------------------------------------------------------------
  // [mdspan.basic.mapping], mdspan mapping domain multidimensional index to access codomain element

  #if MDSPAN_USE_BRACKET_OPERATOR
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      // 折叠表达式，检查是否可以将 SizeTypes 类型转换为 index_type 类型
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(std::is_convertible, SizeTypes, index_type) /* && ... */) &&
      // 折叠表达式，检查是否可以通过 SizeTypes 类型构造 index_type 类型
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, SizeTypes) /* && ... */) &&
      // 检查秩是否与 SizeTypes 参数数量相同
      (rank() == sizeof...(SizeTypes))
    )
  )
  // 强制内联函数，constexpr 函数，返回 mdspan 的引用
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator[](SizeTypes... indices) const
  {
    // 调用访问器对象的 access 方法，传入 multidimensional indices，获取访问的元素引用
    return __accessor_ref().access(__ptr_ref(), __mapping_ref()(static_cast<index_type>(std::move(indices))...));
  }
  #endif

  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType,
    /* requires */ (
      // 检查是否可以将 SizeType 类型转换为 index_type 类型
      _MDSPAN_TRAIT(std::is_convertible, const SizeType&, index_type) &&
      // 检查是否可以通过 SizeType 类型构造 index_type 类型
      _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, const SizeType&)
    )
  )
  // 强制内联函数，constexpr 函数，返回 mdspan 的引用
  constexpr reference operator[](const std::array< SizeType, rank()>& indices) const
  {
    // 调用内部实现的 __callop 函数模板，传入 *this 和 indices，返回访问的元素引用
    return __impl::template __callop<reference>(*this, indices);
  }

  #ifdef __cpp_lib_span
  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType,
    /* requires */ (
      // 检查是否可以将 SizeType 类型转换为 index_type 类型
      _MDSPAN_TRAIT(std::is_convertible, const SizeType&, index_type) &&
      // 检查是否可以通过 SizeType 类型构造 index_type 类型
      _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, const SizeType&)
    )
  )
  // 强制内联函数，constexpr 函数，返回 mdspan 的引用
  constexpr reference operator[](std::span<SizeType, rank()> indices) const
  {
    // 调用内部实现的 __callop 函数模板，传入 *this 和 indices，返回访问的元素引用
    return __impl::template __callop<reference>(*this, indices);
  }
  #endif // __cpp_lib_span

  #if !MDSPAN_USE_BRACKET_OPERATOR
  MDSPAN_TEMPLATE_REQUIRES(
    class Index,
    /* requires */ (
      // 检查是否可以将 Index 类型转换为 index_type 类型
      _MDSPAN_TRAIT(std::is_convertible, Index, index_type) &&
      // 检查是否可以通过 Index 类型构造 index_type 类型
      _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, Index) &&
      // 检查秩是否为 1
      extents_type::rank() == 1
    )
  )
  // 强制内联函数，constexpr 函数，返回 mdspan 的引用
  constexpr reference operator[](Index idx) const
  {
    // 调用内部实现的 __callop 函数模板，传入 *this 和 idx，返回访问的元素引用
  // 如果定义了 MDSPAN_USE_PAREN_OPERATOR 宏，则定义一个可调用对象，允许通过括号操作符访问多维数组元素
  #if MDSPAN_USE_PAREN_OPERATOR
  // 使用模板参数推断机制，要求 SizeTypes 是有效的维度大小类型，并且索引数量匹配数组的维度
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      extents_type::rank() == sizeof...(SizeTypes) &&
      (detail::are_valid_indices<index_type, SizeTypes...>())
    )
  )
  // 强制内联函数声明，返回通过调用 accessor 对象访问指定索引的数组元素的引用
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator()(SizeTypes... indices) const
  {
    // 调用 accessor 对象的 access 方法，使用 mapping 对象将索引映射到底层数据的指针
    return __accessor_ref().access(__ptr_ref(), __mapping_ref()(static_cast<index_type>(std::move(indices))...));
  }

  // 使用 std::array 作为参数的调用操作符重载，要求 SizeType 类型可以转换为 index_type 类型，并且能以此构造 index_type 对象
  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType,
    /* requires */ (
      _MDSPAN_TRAIT(std::is_convertible, const SizeType&, index_type) &&
      _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, const SizeType&)
    )
  )
  // 强制内联函数声明，返回通过调用 __impl::__callop<reference> 处理器访问指定索引的数组元素的引用
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator()(const std::array<SizeType, rank()>& indices) const
  {
    // 调用 __impl 命名空间下的 __callop 函数模板，使用当前对象和传入的索引数组访问数组元素
    return __impl::template __callop<reference>(*this, indices);
  }

  // 如果定义了 __cpp_lib_span 宏，则使用 std::span 作为参数的调用操作符重载
  #ifdef __cpp_lib_span
  // 要求 SizeType 类型可以转换为 index_type 类型，并且能以此构造 index_type 对象
  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType,
    /* requires */ (
      _MDSPAN_TRAIT(std::is_convertible, const SizeType&, index_type) &&
      _MDSPAN_TRAIT(std::is_nothrow_constructible, index_type, const SizeType&)
    )
  )
  // 强制内联函数声明，返回通过调用 __impl::__callop<reference> 处理器访问指定索引的数组元素的引用
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator()(std::span<SizeType, rank()> indices) const
  {
    // 调用 __impl 命名空间下的 __callop 函数模板，使用当前对象和传入的索引 span 访问数组元素
    return __impl::template __callop<reference>(*this, indices);
  }
  // 结束 __cpp_lib_span 宏的条件编译
  #endif // __cpp_lib_span
  // 结束 MDSPAN_USE_PAREN_OPERATOR 宏的条件编译
  #endif // MDSPAN_USE_PAREN_OPERATOR

  // 返回当前多维数组的大小，使用 __impl::__size 函数
  MDSPAN_INLINE_FUNCTION constexpr size_type size() const noexcept {
    return __impl::__size(*this);
  };

  // 返回当前多维数组是否为空，使用 __impl::__empty 函数
  MDSPAN_INLINE_FUNCTION constexpr bool empty() const noexcept {
    return __impl::__empty(*this);
  };

  // 友元函数声明，用于交换两个 mdspan 对象的状态，如果未定义 _MDSPAN_HAS_HIP 和 _MDSPAN_HAS_CUDA 宏
  MDSPAN_INLINE_FUNCTION
  friend constexpr void swap(mdspan& x, mdspan& y) noexcept {
    // 不能在 HIP 或 CUDA 下调用 std::swap
    #if !defined(_MDSPAN_HAS_HIP) && !defined(_MDSPAN_HAS_CUDA)
    using std::swap;
    // 交换 x 和 y 的 __ptr_ref、__mapping_ref 和 __accessor_ref 成员
    swap(x.__ptr_ref(), y.__ptr_ref());
    swap(x.__mapping_ref(), y.__mapping_ref());
    swap(x.__accessor_ref(), y.__accessor_ref());
    // 在 HIP 或 CUDA 下，通过复制和赋值交换 x 和 y 的状态
    #else
    mdspan tmp = y;
    y = x;
    x = tmp;
    // 结束 HIP 或 CUDA 宏的条件编译
    #endif
  }
    #endif
  }



  //--------------------------------------------------------------------------------
  // [mdspan.basic.domobs], mdspan observers of the domain multidimensional index space



  MDSPAN_INLINE_FUNCTION constexpr const extents_type& extents() const noexcept { return __mapping_ref().extents(); };


  MDSPAN_INLINE_FUNCTION constexpr const data_handle_type& data_handle() const noexcept { return __ptr_ref(); };


  MDSPAN_INLINE_FUNCTION constexpr const mapping_type& mapping() const noexcept { return __mapping_ref(); };


  MDSPAN_INLINE_FUNCTION constexpr const accessor_type& accessor() const noexcept { return __accessor_ref(); };



  //--------------------------------------------------------------------------------
  // [mdspan.basic.obs], mdspan observers of the mapping



  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() { return mapping_type::is_always_unique(); };


  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() { return mapping_type::is_always_exhaustive(); };


  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() { return mapping_type::is_always_strided(); };



  MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const { return __mapping_ref().is_unique(); };


  MDSPAN_INLINE_FUNCTION constexpr bool is_exhaustive() const { return __mapping_ref().is_exhaustive(); };


  MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const { return __mapping_ref().is_strided(); };


  MDSPAN_INLINE_FUNCTION constexpr index_type stride(size_t r) const { return __mapping_ref().stride(r); };
// 私有部分开始

// 数据成员__members是一个__compressed_pair对象，包含data_handle_type和__map_acc_pair_t类型的成员
detail::__compressed_pair<data_handle_type, __map_acc_pair_t> __members{};

// 获取和返回数据句柄的引用，用于访问数据
MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 data_handle_type& __ptr_ref() noexcept { return __members.__first(); }
// 获取和返回常量数据句柄的引用，用于只读访问数据
MDSPAN_FORCE_INLINE_FUNCTION constexpr data_handle_type const& __ptr_ref() const noexcept { return __members.__first(); }

// 获取和返回映射类型的引用，用于访问数据的映射方式
MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 mapping_type& __mapping_ref() noexcept { return __members.__second().__first(); }
// 获取和返回常量映射类型的引用，用于只读访问数据的映射方式
MDSPAN_FORCE_INLINE_FUNCTION constexpr mapping_type const& __mapping_ref() const noexcept { return __members.__second().__first(); }

// 获取和返回访问器类型的引用，用于访问数据的访问器
MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 accessor_type& __accessor_ref() noexcept { return __members.__second().__second(); }
// 获取和返回常量访问器类型的引用，用于只读访问数据的访问器
MDSPAN_FORCE_INLINE_FUNCTION constexpr accessor_type const& __accessor_ref() const noexcept { return __members.__second().__second(); }

// 友元类模板mdspan的定义
template <class, class, class, class>
friend class mdspan;

// 私有部分结束
// 类模板 `layout_left::mapping` 的定义，使用模板参数 `Extents` 表示维度信息
template <class Extents>
class layout_left::mapping {
  public:
    // 定义使用的类型别名
    using extents_type = Extents;          // 维度类型
    using index_type = typename extents_type::index_type;   // 索引类型
    using size_type = typename extents_type::size_type;     // 大小类型
    using rank_type = typename extents_type::rank_type;     // 秩类型
    using layout_type = layout_left;        // 布局类型为 `layout_left`

  private:
    // 断言，确保 `extents_type` 是 `MDSPAN` 库的合法维度类型
    static_assert(detail::__is_extents_v<extents_type>,
                  MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::layout_left::mapping must be instantiated with a specialization of " MDSPAN_IMPL_STANDARD_NAMESPACE_STRING "::extents.");

    template <class>
    friend class mapping;  // 允许其它 `mapping` 类模板访问私有成员

    // 递归计算偏移量的函数模板，用于计算多维索引在扁平化存储中的偏移量
    // i0 + (i1 + E(1) * (i2 + E(2) * i3))
    template <size_t r, size_t Rank>
    struct __rank_count {};

    // 递归计算偏移量的实现函数，使用可变参数模板展开多维索引
    template <size_t r, size_t Rank, class I, class... Indices>
    _MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(
      __rank_count<r, Rank>, const I& i, Indices... idx) const {
      return __compute_offset(__rank_count<r+1, Rank>(), idx...) *
                 __extents.extent(r) + i;
    }

    // 递归计算偏移量的结束条件，处理最后一个维度的索引
    template<class I>
    _MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(
      __rank_count<extents_type::rank()-1, extents_type::rank()>, const I& i) const {
      return i;
    }

    // 特化处理最外层维度的递归计算，初始偏移量为0
    _MDSPAN_HOST_DEVICE
    constexpr index_type __compute_offset(__rank_count<0, 0>) const { return 0; }

  public:
    //--------------------------------------------------------------------------------

    // 默认构造函数，使用 `MDSPAN_INLINE_FUNCTION_DEFAULTED` 进行声明
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping() noexcept = default;
    // 默认使用复制构造函数，拷贝另一个 mapping 对象的内容
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;
    
    // 以给定的 extents_type 构造 mapping 对象，初始化 __extents 成员
    _MDSPAN_HOST_DEVICE
    constexpr mapping(extents_type const& __exts) noexcept
      :__extents(__exts)
    { }
    
    // 如果满足 OtherExtents 类型可构造为 extents_type 类型，则使用 OtherExtents 构造 mapping 对象
    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        _MDSPAN_TRAIT(std::is_constructible, extents_type, OtherExtents)
      )
    )
    // 如果 OtherExtents 不可转换为 extents_type，则显式构造函数
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible<OtherExtents, extents_type>::value))
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    // 从另一个 mapping 对象 other 复制 extents 并初始化
    mapping(mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    {
       /*
        * TODO: check precondition
        * other.required_span_size() is a representable value of type index_type
        */
    }
    
    // 如果满足 OtherExtents 类型可构造为 extents_type 类型，并且 extents_type 的秩小于等于 1，则使用 OtherExtents 构造 mapping 对象
    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        _MDSPAN_TRAIT(std::is_constructible, extents_type, OtherExtents) &&
        (extents_type::rank() <= 1)
      )
    )
    // 如果 OtherExtents 不可转换为 extents_type，则显式构造函数
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible<OtherExtents, extents_type>::value))
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    // 从另一个 layout_right::mapping 对象 other 复制 extents 并初始化
    mapping(layout_right::mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    {
       /*
        * TODO: check precondition
        * other.required_span_size() is a representable value of type index_type
        */
    }
#if MDSPAN_HAS_CXX_17
    /**
     * Converting constructor from `layout_left_padded::mapping`.
     *
     * This overload participates in overload resolution only if _Mapping is a layout_left_padded mapping and
     * extents_type is constructible from _Mapping::extents_type.
     *
     * \note There is currently a difference from p2642r2, where this function is specified as taking
     * `layout_left_padded< padding_value >::mapping< Extents>`. However, this makes `padding_value` non-deducible.
     */
    MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (
        MDSPAN_IMPL_PROPOSED_NAMESPACE::detail::is_layout_left_padded_mapping<_Mapping>::value
        && std::is_constructible_v<extents_type, typename _Mapping::extents_type>
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible_v<typename _Mapping::extents_type, extents_type>))
    mapping(const _Mapping& __other) noexcept
      : __extents(__other.extents())
    {
      MDSPAN_IMPL_PROPOSED_NAMESPACE::detail::
          check_padded_layout_converting_constructor_mandates<
            extents_type, _Mapping>(detail::with_rank<extents_type::rank()>{});
      MDSPAN_IMPL_PROPOSED_NAMESPACE::detail::
          check_padded_layout_converting_constructor_preconditions<
              extents_type>(detail::with_rank<extents_type::rank()>{}, __other);
    }
#endif

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        _MDSPAN_TRAIT(std::is_constructible, extents_type, OtherExtents)
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    mapping(layout_stride::mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    {
       /*
        * TODO: check precondition
        * other.required_span_size() is a representable value of type index_type
        */
       detail::validate_strides(detail::with_rank<extents_type::rank()>{}, layout_left{}, __extents, other);
    }

    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(mapping const&) noexcept = default;

    MDSPAN_INLINE_FUNCTION
    constexpr const extents_type& extents() const noexcept {
      return __extents;
    }

    MDSPAN_INLINE_FUNCTION
    constexpr index_type required_span_size() const noexcept {
      index_type value = 1;
      for(rank_type r=0; r<extents_type::rank(); r++) value*=__extents.extent(r);
      return value;
    }

    //--------------------------------------------------------------------------------

    MDSPAN_TEMPLATE_REQUIRES(
      class... Indices,
      /* requires */ (
        (sizeof...(Indices) == extents_type::rank()) &&
        (detail::are_valid_indices<index_type, Indices...>())
      )
    )
    _MDSPAN_HOST_DEVICE
    constexpr index_type operator()(Indices... idxs) const noexcept {
#if ! defined(NDEBUG)
      detail::check_all_indices(this->extents(), idxs...);
#endif
      return detail::linearize_index(__extents, idxs...);
    }
#endif // ! NDEBUG
      // 如果未定义 NDEBUG 宏，则执行下面的语句
      return __compute_offset(__rank_count<0, extents_type::rank()>(), static_cast<index_type>(idxs)...);
    }



    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    // 返回始终为真的 constexpr 布尔值，表示该类型始终唯一
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept { return true; }
    // 返回始终为真的 constexpr 布尔值，表示该类型始终穷尽
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }
    // 返回始终为真的 constexpr 布尔值，表示该类型始终步进

    MDSPAN_INLINE_FUNCTION static constexpr bool is_unique() noexcept { return true; }
    // 返回始终为真的 constexpr 布尔值，表示当前实例唯一
    MDSPAN_INLINE_FUNCTION static constexpr bool is_exhaustive() noexcept { return true; }
    // 返回始终为真的 constexpr 布尔值，表示当前实例穷尽
    MDSPAN_INLINE_FUNCTION static constexpr bool is_strided() noexcept { return true; }
    // 返回始终为真的 constexpr 布尔值，表示当前实例步进

    MDSPAN_INLINE_FUNCTION
    constexpr index_type stride(rank_type i) const noexcept
#if MDSPAN_HAS_CXX_20
      // 如果支持 C++20 的约束语法，并且 Extents 的秩大于 0
      requires ( Extents::rank() > 0 )
#endif
    {
      index_type value = 1;
      // 计算给定维度的步进值
      for(rank_type r=0; r<i; r++) value*=__extents.extent(r);
      return value;
    }

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ ( Extents::rank() == OtherExtents::rank())
    )
    MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator==(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
      // 比较两个映射是否相等，基于 Extents 的秩和大小
      return lhs.extents() == rhs.extents();
    }

    // 在 C++20 中，如果发现相等，则存在不等操作
#if !(MDSPAN_HAS_CXX_20)
    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ ( Extents::rank() == OtherExtents::rank())
    )
    MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator!=(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
      // 比较两个映射是否不相等，基于 Extents 的秩和大小
      return lhs.extents() != rhs.extents();
    }
#endif

    // 实现完全的 constexpr 可用 submdspan 所需，当前不是公共的
    template<size_t N, class SizeType, size_t ... E, size_t ... Idx>
    constexpr index_type __get_stride(MDSPAN_IMPL_STANDARD_NAMESPACE::extents<SizeType, E...>,std::integer_sequence<size_t, Idx...>) const {
      // 计算特定维度的步进值，用于实现子范围视图
      return _MDSPAN_FOLD_TIMES_RIGHT((Idx<N? __extents.template __extent<Idx>():1),1);
    }
    template<size_t N>
    constexpr index_type __stride() const noexcept {
      // 获取特定维度的步进值
      return __get_stride<N>(__extents, std::make_index_sequence<extents_type::rank()>());
    }

private:
   _MDSPAN_NO_UNIQUE_ADDRESS extents_type __extents{};

   // [mdspan.submdspan.mapping], submdspan mapping specialization
   template<class... SliceSpecifiers>
    MDSPAN_INLINE_FUNCTION
    constexpr auto submdspan_mapping_impl(
       SliceSpecifiers... slices) const;
       // 子 mdspan 映射的实现，基于给定的切片规范

   template<class... SliceSpecifiers>
     friend constexpr auto submdspan_mapping(
       const mapping& src, SliceSpecifiers... slices) {
         // 友元函数，用于生成子 mdspan 映射
         return src.submdspan_mapping_impl(slices...);
     }
};


} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

//END_FILE_INCLUDE: mdspan/include/experimental/__p0009_bits/layout_left.hpp
#if MDSPAN_HAS_CXX_17
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p2642_bits/layout_padded.hpp
//@HEADER
//@HEADER

#include <cassert>  // 包含 C++ 标准断言库头文件

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {

namespace detail {

// 返回大于或等于 offset 的下一个 alignment 的倍数
template<class _T>
MDSPAN_INLINE_FUNCTION
constexpr _T
find_next_multiple(_T alignment, _T offset)
{
  if ( alignment == 0 ) {  // 如果 alignment 为 0，则直接返回 0
    return _T(0);
  } else {
    return ( ( offset + alignment - 1 ) / alignment) * alignment;  // 计算下一个 alignment 的倍数
  }
}

// 获取实际的静态填充值，用于填充数组的某一维度
template <class _ExtentsType, size_t _PaddingValue, size_t _ExtentToPadIdx>
MDSPAN_INLINE_FUNCTION constexpr size_t get_actual_static_padding_value() {
  constexpr auto rank = _ExtentsType::rank();

  if constexpr (rank <= typename _ExtentsType::rank_type(1)) {  // 如果维度数小于等于 1，则返回 0
    return 0;
  } else if constexpr (_PaddingValue != dynamic_extent &&
                       _ExtentsType::static_extent(_ExtentToPadIdx) !=
                           dynamic_extent) {
    static_assert(
        (_PaddingValue != 0) ||
            (_ExtentsType::static_extent(_ExtentToPadIdx) == 0),
        "padding stride can be 0 only if "
        "extents_type::static_extent(extent-to-pad) is 0 or dynamic_extent");
    return find_next_multiple(_PaddingValue,
                                _ExtentsType::static_extent(_ExtentToPadIdx));  // 返回填充值的下一个倍数
  } else {
    return dynamic_extent;  // 返回动态填充值
  }
  // 缺少返回语句的警告来自于 NVCC
#ifdef __NVCC__
  return 0;
#endif
}

// 对于具有填充需求的静态数组，确定其类型
template <size_t _PaddingValue, typename _Extents, size_t _ExtentToPadIdx, size_t _Rank, typename Enabled = void>
struct static_array_type_for_padded_extent
{
  static constexpr size_t padding_value = _PaddingValue;
  using index_type = typename _Extents::index_type;
  using extents_type = _Extents;
  using type = ::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::maybe_static_array<
      index_type, size_t, dynamic_extent,
      detail::get_actual_static_padding_value<extents_type, padding_value,
                                                _ExtentToPadIdx>()>;
};

// 如果维度数小于等于 1，直接返回静态数组类型
template <size_t _PaddingValue, typename _Extents, size_t _ExtentToPadIdx, size_t Rank>
struct static_array_type_for_padded_extent<_PaddingValue, _Extents,
                                             _ExtentToPadIdx, Rank, std::enable_if_t<Rank <= 1>> {
  using index_type = typename _Extents::index_type;
  using extents_type = _Extents;
  using type =
      ::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::maybe_static_array<
          index_type, size_t, dynamic_extent, 0>;
};

// 以下内容未完，需要继续补充
struct padded_extent {
  // 定义静态常量 padding_value，其值为 _PaddingValue
  static constexpr size_t padding_value = _PaddingValue;
  // 使用 _Extents 的 index_type 定义别名 index_type
  using index_type = typename _Extents::index_type;
  // 使用 _Extents 的 extents_type 定义别名 extents_type
  using extents_type = _Extents;
  // 根据 static_array_type_for_padded_extent 模板参数初始化静态数组类型 static_array_type
  using static_array_type = typename static_array_type_for_padded_extent<
      padding_value, _Extents, _ExtentToPadIdx, _Extents::rank()>::type;

  // 返回 static_array_type 的静态值的静态成员函数 static_value
  static constexpr auto static_value() { return static_array_type::static_value(0); }

  // 初始化填充数据的静态数组类型 static_array_type
  MDSPAN_INLINE_FUNCTION
  static constexpr static_array_type
  init_padding(const _Extents &exts) {
    // 如果 _Extents 的秩大于 1 并且 padding_value 是动态扩展，则返回包含 _ExtentToPadIdx 维度大小的数组
    if constexpr ((_Extents::rank() > 1) && (padding_value == dynamic_extent)) {
      return {exts.extent(_ExtentToPadIdx)};
    } else {
      // 否则，使用指定的 padding_value 初始化填充数组
      return init_padding(exts, padding_value);
    }
  // 缺少 NVCC 的返回语句警告
#ifdef __NVCC__
  return {};
#endif
  }

  // 使用给定的 padding_value 和 exts 初始化填充数组的静态数组类型 static_array_type
  MDSPAN_INLINE_FUNCTION static constexpr static_array_type
  init_padding([[maybe_unused]] const _Extents &exts,
               [[maybe_unused]] index_type pv) {
    // 如果 _Extents 的秩大于 1，则返回包含 _ExtentToPadIdx 维度的下一个倍数的数组
    if constexpr (_Extents::rank() > 1) {
      return {find_next_multiple(pv,
                                   exts.extent(_ExtentToPadIdx))};
    } else {
      // 否则返回空数组
      return {};
    }
  // 缺少 NVCC 的返回语句警告
#ifdef __NVCC__
  return {};
#endif
  }

  // 根据其他映射 _Mapping 和 _PaddingStrideIdx 初始化填充数组的静态数组类型 static_array_type
  template <typename _Mapping, size_t _PaddingStrideIdx>
  MDSPAN_INLINE_FUNCTION static constexpr static_array_type
  init_padding([[maybe_unused]] const _Mapping &other_mapping,
                      std::integral_constant<size_t, _PaddingStrideIdx>) {
    // 如果 _Extents 的秩大于 1，则返回包含 _PaddingStrideIdx 维度的步长的数组
    if constexpr (_Extents::rank() > 1) {
      return {other_mapping.stride(_PaddingStrideIdx)};
    } else {
      // 否则返回空数组
      return {};
    }
  // 缺少 NVCC 的返回语句警告
#ifdef __NVCC__
  return {};
#endif
  }
};
} // namespace detail

template <size_t PaddingValue>
template <class Extents>
class layout_left_padded<PaddingValue>::mapping {
public:
  // 定义静态常量 padding_value，其值为 PaddingValue
  static constexpr size_t padding_value = PaddingValue;

  // 使用 Extents 的类型定义别名 extents_type
  using extents_type = Extents;
  // 使用 Extents 的 index_type 定义别名 index_type
  using index_type = typename extents_type::index_type;
  // 使用 Extents 的 size_type 定义别名 size_type
  using size_type = typename extents_type::size_type;
  // 使用 Extents 的 rank_type 定义别名 rank_type
  using rank_type = typename extents_type::rank_type;
  // 使用 layout_left_padded<padding_value> 定义别名 layout_type
  using layout_type = layout_left_padded<padding_value>;

#ifndef MDSPAN_INTERNAL_TEST
private:
// 结束条件指令，用于标记内部测试的结束
#endif // MDSPAN_INTERNAL_TEST

// 获取填充后步长的索引
static constexpr rank_type padded_stride_idx = detail::layout_padded_constants<layout_type, extents_type>::padded_stride_idx;
// 获取要填充的尺度索引
static constexpr rank_type extent_to_pad_idx = detail::layout_padded_constants<layout_type, extents_type>::extent_to_pad_idx;

// 静态断言，检查是否满足以下任一条件：
// 1. padding_value 不为 0
// 2. extent_to_pad_idx 对应的静态尺度为 0
// 3. extent_to_pad_idx 对应的静态尺度为动态尺度
static_assert((padding_value != 0)
              || (extents_type::static_extent(extent_to_pad_idx) == 0)
              || (extents_type::static_extent(extent_to_pad_idx) == dynamic_extent),
              "out of bounds access for rank 0");

// 使用 detail::padded_extent 创建带填充的尺度类型
using padded_stride_type = detail::padded_extent< padding_value, extents_type, extent_to_pad_idx >;

// 获取静态填充步长值
static constexpr size_t static_padding_stride = padded_stride_type::static_value();

// 创建静态填充步长的静态数组
typename padded_stride_type::static_array_type padded_stride = {};
// 创建尺度类型的对象
extents_type exts = {};

// 计算偏移量的函数，无参数情况下返回0
MDSPAN_INLINE_FUNCTION constexpr index_type
compute_offset(std::index_sequence<>) const {
  return 0;
}

// 计算偏移量的函数模板，给定Rank和IndexOffset，返回IndexOffset
template <size_t Rank, class IndexOffset>
MDSPAN_INLINE_FUNCTION constexpr index_type
compute_offset(std::index_sequence<Rank>, IndexOffset index_offset) const {
  return index_offset;
}

// 计算偏移量的函数模板，给定Ranks和IndexOffsets，返回计算后的偏移量
template <size_t... Ranks, class... IndexOffsets>
MDSPAN_INLINE_FUNCTION constexpr index_type
compute_offset(std::index_sequence<Ranks...>,
               IndexOffsets... index_offsets) const {
  // 将参数转换为索引数组
  index_type indices[] = {static_cast<index_type>(index_offsets)...};
  // 使用自递归折叠技巧计算偏移量，参考自：
  // https://github.com/llvm/llvm-project/blob/96e1914aa2e6d8966acbfbe2f4d184201f1aa318/libcxx/include/mdspan/layout_left.h#L144
  index_type res = 0;
  // 逐个计算偏移量
  ((res = indices[extents_type::rank() - 1 - Ranks] +
          // 如果是 extent_to_pad_idx 对应的尺度索引，则使用填充步长值
          ((extents_type::rank() - 1 - Ranks) == extent_to_pad_idx
               ? padded_stride.value(0)
               : exts.extent(extents_type::rank() - 1 - Ranks)) *
              res),
   ...);
  return res;
}

public:
// 如果不支持 C++20，则默认构造函数委托给空 extents_type 对象的构造函数
#if !MDSPAN_HAS_CXX_20
MDSPAN_INLINE_FUNCTION_DEFAULTED
constexpr mapping()
    : mapping(extents_type{})
{}
#else
// 如果支持 C++20 并且静态填充步长不是动态尺度，则使用默认构造函数
MDSPAN_INLINE_FUNCTION_DEFAULTED
constexpr mapping()
  requires(static_padding_stride != dynamic_extent) = default;

// 如果支持 C++20 并且静态填充步长是动态尺度，则使用空 extents_type 对象的构造函数
MDSPAN_INLINE_FUNCTION
constexpr mapping()
  requires(static_padding_stride == dynamic_extent)
    : mapping(extents_type{})
{}
#endif

// 复制构造函数，默认使用默认的复制构造函数
MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(const mapping&) noexcept = default;
// 赋值运算符，默认使用默认的赋值运算符
MDSPAN_INLINE_FUNCTION_DEFAULTED mapping& operator=(const mapping&) noexcept = default;

/**
 * 使用给定的 extents 初始化映射。
 *
 * \param ext 给定的 extents
 */
MDSPAN_INLINE_FUNCTION
constexpr mapping(const extents_type& ext)
  /**
   * 构造函数：使用给定的 extents 和指定的填充值初始化映射。
   *
   * 此重载仅在 `is_convertible_v<Size, index_type>` 为 `true` 和
   * `is_nothrow_constructible_v<index_type, Size>` 为 `true` 时参与重载解析。
   *
   * \param ext 给定的 extents
   * \param dynamic_padding_value 动态填充值
   */
  MDSPAN_TEMPLATE_REQUIRES(
    class _Size,
    /* requires */ (
      std::is_convertible_v<_Size, index_type>
      && std::is_nothrow_constructible_v<index_type, _Size>
    )
  )
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const extents_type &ext, _Size dynamic_padding_value)
      : padded_stride(padded_stride_type::init_padding(ext, dynamic_padding_value)), exts(ext)
  {
    // 确保填充值是动态或与动态填充值类型匹配
    assert((padding_value == dynamic_extent) || (static_cast<index_type>(padding_value) == static_cast<index_type>(dynamic_padding_value)));
  }

  /**
   * 从 `layout_left::mapping` 转换构造函数。
   *
   * 此重载仅在 `is_constructible_v<extents_type, OtherExtents>` 为 true 时参与重载解析。
   * 如果 `OtherExtents::rank() > 1`，则 `padding_value`、`static_extent(0)` 或 `OtherExtents::static_extent(0)`
   * 必须是 `dynamic_extent`；否则，`OtherExtents::static_extent(0)` 必须等于大于等于 `extents_type::static_extent(0)`
   * 的最小填充值的倍数。
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _OtherExtents,
      /* requires */ (std::is_constructible_v<extents_type, _OtherExtents>))
  MDSPAN_CONDITIONAL_EXPLICIT(
      (!std::is_convertible_v<_OtherExtents, extents_type>))
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const layout_left::mapping<_OtherExtents> &other_mapping)
      : padded_stride(padded_stride_type::init_padding(
            other_mapping,
            std::integral_constant<size_t, padded_stride_idx>{})),
        exts(other_mapping.extents()) {
    static_assert(
        (_OtherExtents::rank() > 1) ||  // 静态断言：要求_OtherExtents的秩大于1，或者以下条件之一成立
        (static_padding_stride != dynamic_extent) ||  // 静态填充步长不是动态值
        (_OtherExtents::static_extent(extent_to_pad_idx) != dynamic_extent) ||  // _OtherExtents的特定静态维度不是动态值
        (static_padding_stride ==
         _OtherExtents::static_extent(extent_to_pad_idx)));  // 静态填充步长等于_OtherExtents的特定静态维度

  }

  /**
   * 从`layout_stride::mapping`转换的构造函数。
   *
   * 只有当`is_constructible_v<extents_type, OtherExtents>`为true时，此重载才参与重载解析。
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _OtherExtents,
      /* requires */ (std::is_constructible_v<extents_type, _OtherExtents>))
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const layout_stride::mapping<_OtherExtents> &other_mapping)
      : padded_stride(padded_stride_type::init_padding(
            other_mapping,
            std::integral_constant<size_t, padded_stride_idx>{})),  // 初始化填充步长
        exts(other_mapping.extents()) {}  // 使用other_mapping的维度构造mapping对象的exts

  /**
   * 从`layout_left_padded::mapping`转换的构造函数。
   *
   * 只有当`is_constructible_v<extents_type, OtherExtents>`为true时，此重载才参与重载解析。
   * `padding_value`或`OtherPaddingStride`必须是`std::dynamic_extent`，或者`padding_value == OtherPaddingStride`。
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (detail::is_layout_left_padded_mapping<_Mapping>::value
                          && std::is_constructible_v<
                              extents_type, typename _Mapping::extents_type>))
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 1 &&
                               (padding_value == dynamic_extent ||
                                _Mapping::padding_value == dynamic_extent)))
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const _Mapping &other_mapping)
      : padded_stride(padded_stride_type::init_padding(
            other_mapping,
            std::integral_constant<size_t, padded_stride_idx>{})),  // 初始化填充步长
        exts(other_mapping.extents()) {  // 使用other_mapping的维度构造mapping对象的exts
    static_assert(padding_value == dynamic_extent ||
                  _Mapping::padding_value == dynamic_extent ||
                  padding_value == _Mapping::padding_value);
  }

  /**
   * Converting constructor from `layout_right_padded::mapping`.
   *
   * This overload participates in overload resolution only if
   * `extents_type::rank()` is 0 or 1 and `is_constructible_v<extents_type,
   * OtherExtents>` is `true`.
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (detail::is_layout_right_padded_mapping<_Mapping>::value
                              && extents_type::rank() <= 1 &&
                      std::is_constructible_v<extents_type,
                                              typename _Mapping::extents_type>))
  MDSPAN_CONDITIONAL_EXPLICIT(
      (!std::is_convertible_v<typename _Mapping::extents_type, extents_type>))
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const _Mapping &other_mapping) noexcept
      : padded_stride(padded_stride_type::init_padding(
            other_mapping.extents(),
            other_mapping.extents().extent(extent_to_pad_idx))),
        exts(other_mapping.extents()) {}

  /**
   * Returns the extents of the mapping.
   */
  MDSPAN_INLINE_FUNCTION constexpr const extents_type &
  extents() const noexcept {
    return exts;
  }

  /**
   * Returns the strides of the mapping.
   *
   * Depending on the rank of extents_type, calculates and returns an array of
   * strides.
   */
  MDSPAN_INLINE_FUNCTION constexpr std::array<index_type, extents_type::rank()>
  strides() const noexcept {
    if constexpr (extents_type::rank() == 0) {
      return {};
    } else if constexpr (extents_type::rank() == 1) {
      return {1};
    } else {
      index_type value = 1;
      std::array<index_type, extents_type::rank()> s{};
      s[extent_to_pad_idx] = value;
      value *= padded_stride.value(0);
      for (rank_type r = extent_to_pad_idx + 1; r < extents_type::rank() - 1;
           ++r) {
        s[r] = value;
        value *= exts.extent(r);
      }
      s[extents_type::rank() - 1] = value;
      return s;
    }
  }

  /**
   * Returns the required span size of the mapping.
   *
   * Calculates and returns the total size required by the mapping across all
   * dimensions.
   */
  MDSPAN_INLINE_FUNCTION constexpr index_type
  required_span_size() const noexcept {
    if constexpr (extents_type::rank() == 0) {
      return 1;
    } else if constexpr (extents_type::rank() == 1) {
      return exts.extent(0);
    } else {
      index_type value = padded_stride.value(0);
      for (rank_type r = 1; r < extents_type::rank(); ++r) {
        value *= exts.extent(r);
      }
      return value;
    }
  }
  }
  // 结束类定义

  /**
   * Return the mapping given the provided indices per rank.
   *
   * This overload participates in overload resolution only if:
   * - `sizeof...(Indices) == extents_type::rank()`,
   * - `(is_convertible_v<Indices, index_type> && ...) is true`, and
   * - (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
   */
  // 定义函数模板，根据给定的索引返回映射
  MDSPAN_TEMPLATE_REQUIRES(
      class... _Indices,
      /* requires */ (sizeof...(_Indices) == extents_type::rank() &&
                      (::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::
                           are_valid_indices<index_type, _Indices...>())))
  // 内联函数声明，constexpr 修饰，保证在编译期求值，且不抛出异常
  MDSPAN_INLINE_FUNCTION constexpr size_t
  operator()(_Indices... idxs) const noexcept {
#if !defined(NDEBUG)
    // 如果未定义 NDEBUG 宏，则执行索引范围检查
    ::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::check_all_indices(this->extents(),
                                                                idxs...);
#endif // ! NDEBUG
    // 调用 compute_offset 函数计算偏移量，并返回结果
    return compute_offset(std::index_sequence_for<_Indices...>{}, idxs...);
  }

  // 返回常量表达式，表示该类型总是具有唯一索引
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept {
    return true;
  }
  // 返回常量表达式，表示该类型总是具有穷尽的索引
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept {
    return (extents_type::rank() <= rank_type(1)) ||
           (extents_type::static_extent(extent_to_pad_idx) != dynamic_extent &&
            extents_type::static_extent(extent_to_pad_idx) ==
                padded_stride_type::static_value());
  }
  // 返回常量表达式，表示该类型总是具有步长索引
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept {
    return true;
  }

  // 返回常量表达式，表示该类型具有唯一索引
  MDSPAN_INLINE_FUNCTION static constexpr bool is_unique() noexcept {
    return true;
  }
  // 返回该实例是否具有穷尽索引的非常量成员函数
  MDSPAN_INLINE_FUNCTION constexpr bool is_exhaustive() const noexcept {
    return (extents_type::rank() < 2) ||
           (exts.extent(extent_to_pad_idx) == padded_stride.value(0));
  }
  // 返回常量表达式，表示该类型具有步长索引
  MDSPAN_INLINE_FUNCTION static constexpr bool is_strided() noexcept {
    return true;
  }

  // 返回步长的索引类型的值
  MDSPAN_INLINE_FUNCTION
  constexpr index_type stride(rank_type r) const noexcept {
    assert(r < extents_type::rank());  // 断言：排名小于维度数
    if (r == 0)
      return index_type(1);  // 如果 r 为 0，则返回步长为 1

    index_type value = padded_stride.value(0);  // 初始化步长值为第一个填充步长的值
    for (rank_type k = 1; k < r; k++)
      value *= exts.extent(k);  // 计算步长乘积

    return value;  // 返回计算后的步长值
  }

  /**
   * Equality operator between `layout_left_padded`s
   *
   * This overload only participates in overload resolution if
   * `OtherExtents::rank() == extents_type::rank()`.
   *
   * \note There is currently a difference from p2642r2, where this function is
   * specified as taking `layout_left_padded< padding_value >::mapping<
   * Extents>`. However, this makes `padding_value` non-deducible.
   */
  // 等号操作符，比较两个 layout_left_padded 类型的对象是否相等
  MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (detail::is_layout_left_padded_mapping<_Mapping>::value &&
                      (_Mapping::extents_type::rank() == extents_type::rank())))
  MDSPAN_INLINE_FUNCTION friend constexpr bool
  operator==(const mapping &left, const _Mapping &right) noexcept {
    // 编译时检查的一些编译器不正确短路操作的回避方法，例如我们不能访问 rank 0 映射的 stride(_padding_stride_idx)
    bool strides_equal = true;
    if constexpr (extents_type::rank() > rank_type(1)) {
      strides_equal =
          left.stride(padded_stride_idx) == right.stride(padded_stride_idx);
    }
    return (left.extents() == right.extents()) && strides_equal;  // 返回比较结果
  }
#if !MDSPAN_HAS_CXX_20
/**
 * Inequality operator between `layout_left_padded`s
 *
 * This overload only participates in overload resolution if
 * `OtherExtents::rank() == extents_type::rank()`.
 */
MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (detail::is_layout_left_padded_mapping<_Mapping>::value &&
                    (_Mapping::extents_type::rank() == extents_type::rank())))
MDSPAN_INLINE_FUNCTION friend constexpr bool
operator!=(const mapping &left, const _Mapping &right) noexcept {
  return !(left == right);
}
#endif



template <size_t PaddingValue>
template <class Extents>
class layout_right_padded<PaddingValue>::mapping {
public:
  static constexpr size_t padding_value = PaddingValue;

  using extents_type = Extents;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;
  using layout_type = layout_right_padded<padding_value>;

#ifndef MDSPAN_INTERNAL_TEST
private:
#endif // MDSPAN_INTERNAL_TEST

  static constexpr rank_type padded_stride_idx = detail::layout_padded_constants<layout_type, extents_type>::padded_stride_idx;
  static constexpr rank_type extent_to_pad_idx = detail::layout_padded_constants<layout_type, extents_type>::extent_to_pad_idx;

  static_assert((padding_value != 0)
                || (extents_type::static_extent(extent_to_pad_idx) == 0)
                || (extents_type::static_extent(extent_to_pad_idx) == dynamic_extent),
                "if padding stride is 0, static_extent(extent-to-pad-rank) must also be 0 or dynamic_extent");

  using padded_stride_type = detail::padded_extent< padding_value, extents_type, extent_to_pad_idx >;
  static constexpr size_t static_padding_stride = padded_stride_type::static_value();

  typename padded_stride_type::static_array_type padded_stride = {};
  extents_type exts = {};

  /**
   * Compute the offset for a multi-dimensional index sequence.
   *
   * This function uses a self-recursive fold trick to calculate the offset based on
   * the provided index offsets and dimensions of the extents.
   *
   * @tparam Rank The current rank of the index sequence being processed
   * @tparam IndexOffset The type of index offset used
   * @param index_offsets The offsets for each dimension in the sequence
   * @return The computed offset based on the provided indices and extents
   */
  MDSPAN_INLINE_FUNCTION constexpr index_type
  compute_offset(std::index_sequence<>) const {
    return 0;
  }

  /**
   * Compute the offset for a multi-dimensional index sequence with at least one rank.
   *
   * This overload of compute_offset handles the computation of offsets using a fold trick.
   * It adjusts the computation based on whether the current rank matches the extent index
   * requiring padding.
   *
   * @tparam Rank The current rank of the index sequence being processed
   * @tparam IndexOffset The type of index offset used
   * @param index_offset The offset for the current rank
   * @return The computed offset based on the provided indices and extents
   */
  template <size_t Rank, class IndexOffset>
  MDSPAN_INLINE_FUNCTION constexpr index_type
  compute_offset(std::index_sequence<Rank>, IndexOffset index_offset) const {
    return index_offset;
  }

  /**
   * Compute the offset for a multi-dimensional index sequence with multiple ranks.
   *
   * This function uses a self-recursive fold trick to calculate the offset based on
   * the provided index offsets and dimensions of the extents. It iterates over each rank
   * in the sequence and accumulates the offset using the padded stride for the specified rank.
   *
   * @tparam Ranks Parameter pack containing the ranks of the index sequence
   * @tparam IndexOffsets Parameter pack containing the offsets for each rank
   * @param index_offsets The offsets for each dimension in the sequence
   * @return The computed offset based on the provided indices and extents
   */
  template <size_t... Ranks, class... IndexOffsets>
  MDSPAN_INLINE_FUNCTION constexpr index_type
  compute_offset(std::index_sequence<Ranks...>,
                 IndexOffsets... index_offsets) const {
    // self-recursive fold trick from
    // https://github.com/llvm/llvm-project/blob/4d9771741d40cc9cfcccb6b033f43689d36b705a/libcxx/include/mdspan/layout_right.h#L141
    index_type res = 0;
    ((res = static_cast<index_type>(index_offsets) +
            (Ranks == extent_to_pad_idx ? padded_stride.value(0)
                                        : exts.extent(Ranks)) *
                res),
     ...);
    return res;
  }

public:
#if !MDSPAN_HAS_CXX_20
// 如果不支持 C++20 特性，则默认构造函数 `mapping()` 使用空的 extents 初始化
MDSPAN_INLINE_FUNCTION_DEFAULTED
    constexpr mapping()
    : mapping(extents_type{})
{}
#else
// 如果支持 C++20，并且 static_padding_stride 不等于 dynamic_extent，则使用默认的映射构造函数
MDSPAN_INLINE_FUNCTION_DEFAULTED
    constexpr mapping()
  requires(static_padding_stride != dynamic_extent) = default;

// 如果支持 C++20，并且 static_padding_stride 等于 dynamic_extent，则使用空 extents 初始化映射
MDSPAN_INLINE_FUNCTION
    constexpr mapping()
  requires(static_padding_stride == dynamic_extent)
    : mapping(extents_type{})
{}
#endif

// 默认拷贝构造函数，使用默认的复制语义
MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(const mapping&) noexcept = default;

// 默认赋值运算符重载，使用默认的赋值语义
MDSPAN_INLINE_FUNCTION_DEFAULTED mapping& operator=(const mapping&) noexcept = default;

/**
 * 使用给定的 extents 初始化映射。
 *
 * \param ext 给定的 extents
 */
MDSPAN_INLINE_FUNCTION
constexpr mapping(const extents_type &ext)
    : padded_stride(padded_stride_type::init_padding(ext)), exts(ext) {}

/**
 * 使用给定的 extents 和指定的填充值初始化映射。
 *
 * 仅当 `is_convertible_v<Size, index_type>` 和 `is_nothrow_constructible_v<index_type, Size>` 都为 `true` 时，此重载才参与重载解析。
 *
 * \param ext 给定的 extents
 * \param dynamic_padding_value 指定的填充值
 */
MDSPAN_TEMPLATE_REQUIRES(
    class _Size,
    /* requires */ (
        std::is_convertible_v<_Size, index_type>
            && std::is_nothrow_constructible_v<index_type, _Size>
        )
    )
MDSPAN_INLINE_FUNCTION
constexpr mapping(const extents_type &ext, _Size dynamic_padding_value)
    : padded_stride(padded_stride_type::init_padding(ext, static_cast<index_type>(dynamic_padding_value))),
      exts(ext) {
  // 断言，确保填充值为 dynamic_extent 或者与 dynamic_padding_value 类型转换后相等
  assert((padding_value == dynamic_extent) ||
         (static_cast<index_type>(padding_value) == static_cast<index_type>(dynamic_padding_value)));
}

/**
 * 从 `layout_right::mapping` 转换构造函数。
 *
 * 仅当 `is_constructible_v<extents_type, OtherExtents>` 为真时，此重载才参与重载解析。
 * 如果 `OtherExtents::rank() > 1`，则 `padding_value`、`static_extent(0)` 或 `OtherExtents::static_extent(0)` 必须为 `dynamic_extent`；
 * 否则，`OtherExtents::static_extent(0)` 必须等于大于等于 `extents_type::static_extent(0)` 的最小填充值倍数。
 */
MDSPAN_TEMPLATE_REQUIRES(
    class _OtherExtents,
    /* requires */ (std::is_constructible_v<extents_type, _OtherExtents>))
MDSPAN_CONDITIONAL_EXPLICIT(
    (!std::is_convertible_v<_OtherExtents, extents_type>))
MDSPAN_INLINE_FUNCTION
constexpr mapping(const layout_right::mapping<_OtherExtents> &other_mapping)
    : padded_stride(padded_stride_type::init_padding(
          other_mapping,
          std::integral_constant<size_t, padded_stride_idx>{})),
      exts(other_mapping.extents()) {
    // 静态断言，验证以下条件之一：
    // 1. _OtherExtents 的秩大于1
    // 2. padded_stride_type 的静态值不等于 dynamic_extent
    // 3. _OtherExtents 在 extent_to_pad_idx 索引上的静态值不等于 dynamic_extent
    // 4. padded_stride_type 的静态值等于 _OtherExtents 在 extent_to_pad_idx 索引上的静态值
    static_assert(
        (_OtherExtents::rank() > 1) ||
        (padded_stride_type::static_value() != dynamic_extent) ||
        (_OtherExtents::static_extent(extent_to_pad_idx) != dynamic_extent) ||
        (padded_stride_type::static_value() ==
         _OtherExtents::static_extent(extent_to_pad_idx)));
  }

  /**
   * 从 layout_stride::mapping 转换的构造函数。
   *
   * 如果 is_constructible_v<extents_type, OtherExtents> 为 true，此重载参与重载解析。
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _OtherExtents,
      /* requires */ (std::is_constructible_v<extents_type, _OtherExtents>))
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const layout_stride::mapping<_OtherExtents> &other_mapping)
      : padded_stride(padded_stride_type::init_padding(
            other_mapping,
            std::integral_constant<size_t, padded_stride_idx>{})),
        exts(other_mapping.extents()) {}

  /**
   * 从 layout_right_padded::mapping 转换的构造函数。
   *
   * 如果 is_constructible_v<extents_type, OtherExtents> 为 true，且以下条件之一成立：
   * - padding_value 或 OtherPaddingStride 必须为 std::dynamic_extent
   * - padding_value 等于 OtherPaddingStride
   * 此重载参与重载解析。
   */
  MDSPAN_TEMPLATE_REQUIRES(
      class _Mapping,
      /* requires */ (detail::is_layout_right_padded_mapping<_Mapping>::value
                          && std::is_constructible_v<
                              extents_type, typename _Mapping::extents_type>))
  MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 1 &&
                               (padding_value == dynamic_extent ||
                                _Mapping::padding_value == dynamic_extent)))
  MDSPAN_INLINE_FUNCTION
  constexpr mapping(const _Mapping &other_mapping)
      : padded_stride(padded_stride_type::init_padding(
            other_mapping,
            std::integral_constant<size_t, padded_stride_idx>{})),
        exts(other_mapping.extents()) {
  // 断言检查：确保填充值等于动态范围或者映射的填充值，用于静态验证条件
  static_assert(padding_value == dynamic_extent ||
                _Mapping::padding_value == dynamic_extent ||
                padding_value == _Mapping::padding_value);
}

/**
 * `layout_left_padded::mapping`的转换构造函数。
 *
 * 只有在 `extents_type::rank()` 为 0 或 1 且 `is_constructible_v<extents_type, OtherExtents>` 为 `true` 时才参与重载解析。
 */
MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (detail::is_layout_left_padded_mapping<_Mapping>::value
                            && extents_type::rank() <= 1 &&
                    std::is_constructible_v<extents_type,
                                            typename _Mapping::extents_type>))
MDSPAN_CONDITIONAL_EXPLICIT(
    (!std::is_convertible_v<typename _Mapping::extents_type, extents_type>))
MDSPAN_INLINE_FUNCTION
constexpr mapping(const _Mapping &other_mapping) noexcept
    : padded_stride(padded_stride_type::init_padding(
          other_mapping.extents(),
          other_mapping.extents().extent(extent_to_pad_idx))),
      exts(other_mapping.extents()) {}

/**
 * 返回常量引用的 `extents_type`。
 */
MDSPAN_INLINE_FUNCTION constexpr const extents_type &
extents() const noexcept {
  return exts;
}

/**
 * 返回 `extents_type::rank()` 长度的 `std::array<index_type>`。
 * 如果 `extents_type::rank()` 为 0，则返回空数组。
 * 如果 `extents_type::rank()` 为 1，则返回值为 {1} 的数组。
 * 否则，根据 `extent_to_pad_idx` 生成填充后的步长数组。
 */
MDSPAN_INLINE_FUNCTION constexpr std::array<index_type, extents_type::rank()>
strides() const noexcept {
  if constexpr (extents_type::rank() == 0) {
    return {};
  } else if constexpr (extents_type::rank() == 1) {
    return {1};
  } else {
    index_type value = 1;
    std::array<index_type, extents_type::rank()> s{};
    s[extent_to_pad_idx] = value;
    value *= padded_stride.value(0);
    for (rank_type r = extent_to_pad_idx - 1; r > 0; --r) {
      s[r] = value;
      value *= exts.extent(r);
    }
    s[0] = value;
    return s;
  }
}

/**
 * 返回所需的跨度大小。
 * 如果 `extents_type::rank()` 为 0，则返回 1。
 * 如果 `extents_type::rank()` 为 1，则返回第 0 维度的大小。
 * 否则，根据 `extent_to_pad_idx` 计算所需的跨度大小。
 */
MDSPAN_INLINE_FUNCTION constexpr index_type
required_span_size() const noexcept {
  if constexpr (extents_type::rank() == 0) {
    return 1;
  } else if constexpr (extents_type::rank() == 1) {
    return exts.extent(0);
  } else {
    index_type value = 1;
    for (rank_type r = 0; r < extent_to_pad_idx; ++r) {
      value *= exts.extent(r);
    }
    return value * padded_stride.value(0);
  }
}

/**
 * 给定每个维度的索引，返回映射值。
 *
 * 只有当以下条件满足时，该重载参与重载解析：
 * - `sizeof...(Indices) == extents_type::rank()`
 * - `(is_convertible_v<Indices, index_type> && ...)` 为 `true`
 * - `(is_nothrow_constructible_v<index_type, Indices> && ...)` 为 `true`
 */
MDSPAN_TEMPLATE_REQUIRES(
    class... _Indices,
    /* requires */ (sizeof...(_Indices) == extents_type::rank() &&
                    (::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::
                         are_valid_indices<index_type, _Indices...>())))
MDSPAN_INLINE_FUNCTION constexpr size_t
operator()(_Indices... idxs) const noexcept {
  /**
   * 返回使用给定索引序列和索引计算偏移量。
   *
   * \tparam _Indices... 索引序列类型
   * \param idxs... 索引参数
   */
  return compute_offset(std::index_sequence_for<_Indices...>{}, idxs...);
}

/**
 * 永远返回true的静态成员函数，指示此布局是否始终是唯一的。
 */
MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept {
  return true;
}

/**
 * 永远返回true的静态成员函数，指示此布局是否始终是完全穷尽的。
 *
 * \note 当维度数小于等于1时，或者指定维度的静态大小与填充步长的静态值相等时返回true。
 */
MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept {
  return (extents_type::rank() <= rank_type(1)) ||
         (extents_type::static_extent(extent_to_pad_idx) != dynamic_extent &&
          extents_type::static_extent(extent_to_pad_idx) ==
              padded_stride_type::static_value());
}

/**
 * 永远返回true的静态成员函数，指示此布局是否始终是分步的。
 */
MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept {
  return true;
}

/**
 * 永远返回true的成员函数，指示此布局是否是唯一的。
 */
MDSPAN_INLINE_FUNCTION static constexpr bool is_unique() noexcept {
  return true;
}

/**
 * 返回此布局是否完全穷尽的成员函数。
 *
 * \note 当维度数小于2时，或者指定维度的大小等于填充步长的第一个值时返回true。
 */
MDSPAN_INLINE_FUNCTION constexpr bool is_exhaustive() const noexcept {
  return (extents_type::rank() < 2) ||
         (exts.extent(extent_to_pad_idx) == padded_stride.value(0));
}

/**
 * 永远返回true的静态成员函数，指示此布局是否始终是分步的。
 */
MDSPAN_INLINE_FUNCTION static constexpr bool is_strided() noexcept {
  return true;
}

/**
 * 返回指定秩的步长。
 *
 * \param r 秩
 * \returns 步长值
 * \pre r 必须小于维度的秩
 * \post 如果 r 等于维度的秩减1，则返回值为1；否则计算并返回适当的步长值。
 */
MDSPAN_INLINE_FUNCTION constexpr index_type
stride(rank_type r) const noexcept {
  assert(r < extents_type::rank());
  if (r == extents_type::rank() - 1)
    return index_type(1);

  index_type value = padded_stride.value(0);
  for (rank_type k = extents_type::rank() - 2; k > r; k--)
    value *= exts.extent(k);

  return value;
}

/**
 * `layout_right_padded`之间的相等运算符重载。
 *
 * 仅在 `OtherExtents::rank() == extents_type::rank()` 的情况下参与重载解析。
 *
 * \note 目前与 p2642r2 有所不同，该函数被指定为接受 `layout_right_padded< padding_value >::mapping< Extents>`。然而，这使得 `padding_value` 不可推导。
 *
 * \tparam _Mapping 另一个布局映射类型
 * \param left 一个布局映射对象
 * \param right 另一个布局映射对象
 * \returns 如果两个布局映射对象的维度和步长相等，则返回true；否则返回false。
 */
MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (detail::is_layout_right_padded_mapping<_Mapping>::value &&
                    (_Mapping::extents_type::rank() == extents_type::rank())))
MDSPAN_INLINE_FUNCTION friend constexpr bool
operator==(const mapping &left, const _Mapping &right) noexcept {
  // 一些编译器不正确地进行编译时检查的短路处理的回避方法，即不能访问秩为0的映射的 `_padding_stride_idx` 步长
  bool strides_equal = true;
  if constexpr (extents_type::rank() > rank_type(1)) {
    strides_equal =
        left.stride(padded_stride_idx) == right.stride(padded_stride_idx);
  }
  return (left.extents() == right.extents()) && strides_equal;
}
#if !MDSPAN_HAS_CXX_20
/**
 * Inequality operator between `layout_right_padded`s
 *
 * This overload only participates in overload resolution if
 * `OtherExtents::rank() == extents_type::rank()`.
 */
MDSPAN_TEMPLATE_REQUIRES(
    class _Mapping,
    /* requires */ (detail::is_layout_right_padded_mapping<_Mapping>::value &&
                    (_Mapping::extents_type::rank() == extents_type::rank())))
MDSPAN_INLINE_FUNCTION friend constexpr bool
operator!=(const mapping &left, const _Mapping &right) noexcept {
  return !(left == right);
}
#endif



//END_FILE_INCLUDE: mdspan/include/experimental/__p2642_bits/layout_padded.hpp
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p2630_bits/submdspan.hpp



//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER



//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p2630_bits/submdspan_extents.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER



#include <tuple>



//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p2630_bits/strided_slice.hpp

//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER



#include <type_traits>



namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

namespace {
  template<class T>
  struct __mdspan_is_integral_constant: std::false_type {};

  template<class T, T val>
  struct __mdspan_is_integral_constant<std::integral_constant<T,val>>: std::true_type {};
}



// Slice Specifier allowing for strides and compile time extent
// 定义一个模板结构体 `strided_slice`，用于描述带有偏移量、范围和步长的切片信息
template <class OffsetType, class ExtentType, class StrideType>
struct strided_slice {
  using offset_type = OffsetType; // 偏移量类型
  using extent_type = ExtentType; // 范围类型
  using stride_type = StrideType; // 步长类型

  _MDSPAN_NO_UNIQUE_ADDRESS OffsetType offset{}; // 偏移量，默认初始化
  _MDSPAN_NO_UNIQUE_ADDRESS ExtentType extent{}; // 范围，默认初始化
  _MDSPAN_NO_UNIQUE_ADDRESS StrideType stride{}; // 步长，默认初始化

  // 断言确保 OffsetType、ExtentType 和 StrideType 是整数类型或者是整数常量
  static_assert(std::is_integral_v<OffsetType> || __mdspan_is_integral_constant<OffsetType>::value);
  static_assert(std::is_integral_v<ExtentType> || __mdspan_is_integral_constant<ExtentType>::value);
  static_assert(std::is_integral_v<StrideType> || __mdspan_is_integral_constant<StrideType>::value);
};

} // MDSPAN_IMPL_STANDARD_NAMESPACE

// 包含结束，即文件 mdspan/include/experimental/__p2630_bits/strided_slice.hpp 的最后
// END_FILE_INCLUDE: mdspan/include/experimental/__p2630_bits/strided_slice.hpp

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

// 映射子映射等级到源映射等级
// InvMapRank 是一个索引序列，我们通过递归构建它来包含映射后的索引
// 递归终止的特化，包含最终的索引序列
template <size_t Counter, size_t... MapIdxs>
MDSPAN_INLINE_FUNCTION
constexpr auto inv_map_rank(std::integral_constant<size_t, Counter>, std::index_sequence<MapIdxs...>) {
  return std::index_sequence<MapIdxs...>();
}

// 减少一个等级的特化（即整数切片说明符）
template<size_t Counter, class Slice, class... SliceSpecifiers, size_t... MapIdxs>
MDSPAN_INLINE_FUNCTION
constexpr auto inv_map_rank(std::integral_constant<size_t, Counter>, std::index_sequence<MapIdxs...>, Slice,
                  SliceSpecifiers... slices) {
  // 使用条件类型转换，根据 Slice 是否可转换为 size_t 决定是否增加 Counter 到映射索引序列
  using next_idx_seq_t = std::conditional_t<std::is_convertible_v<Slice, size_t>,
                                       std::index_sequence<MapIdxs...>,
                                       std::index_sequence<MapIdxs..., Counter>>;

  return inv_map_rank(std::integral_constant<size_t,Counter + 1>(), next_idx_seq_t(),
                                     slices...);
}

// 用于识别 strided_slice 的辅助结构体模板
template <class T> struct is_strided_slice : std::false_type {};

// 特化用于识别 strided_slice 类型
template <class OffsetType, class ExtentType, class StrideType>
struct is_strided_slice<
    strided_slice<OffsetType, ExtentType, StrideType>> : std::true_type {};

// 获取切片规范范围的起始位置的辅助函数 first_of(slice)
MDSPAN_TEMPLATE_REQUIRES(
  class Integral,
  /* requires */(std::is_convertible_v<Integral, size_t>)
)
MDSPAN_INLINE_FUNCTION
constexpr Integral first_of(const Integral &i) {
  return i;
}

// 返回整数常量 0，用于全范围切片的起始位置获取
MDSPAN_INLINE_FUNCTION
constexpr std::integral_constant<size_t, 0>
first_of(const ::MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent_t &) {
  return std::integral_constant<size_t, 0>();
}

// 获取 std::tuple<size_t, size_t> 类型切片规范的起始位置
MDSPAN_TEMPLATE_REQUIRES(
  class Slice,
  /* requires */(std::is_convertible_v<Slice, std::tuple<size_t, size_t>>)
)
MDSPAN_INLINE_FUNCTION
constexpr auto first_of(const Slice &i) {
  return std::get<0>(i);
}

// 继续定义的函数，用于返回 OffsetType 类型的值
template <class OffsetType, class ExtentType, class StrideType>
MDSPAN_INLINE_FUNCTION
constexpr OffsetType
// 获取 strided_slice 结构体中的 offset 成员变量
first_of(const strided_slice<OffsetType, ExtentType, StrideType> &r) {
  return r.offset;
}

// 获取 slice 的末尾范围，需要考虑原始视图的 extents 和其维度
// 当 slice 是 full_extent_t 的情况下，还需要知道 extents 的 rank
MDSPAN_TEMPLATE_REQUIRES(
  size_t k, class Extents, class Integral,
  /* requires */(std::is_convertible_v<Integral, size_t>)
)
MDSPAN_INLINE_FUNCTION
constexpr Integral
    last_of(std::integral_constant<size_t, k>, const Extents &, const Integral &i) {
  return i;
}

// 获取 slice 的末尾范围，需要考虑原始视图的 extents 和 slice 的 tuple 类型
MDSPAN_TEMPLATE_REQUIRES(
  size_t k, class Extents, class Slice,
  /* requires */(std::is_convertible_v<Slice, std::tuple<size_t, size_t>>)
)
MDSPAN_INLINE_FUNCTION
constexpr auto last_of(std::integral_constant<size_t, k>, const Extents &,
                       const Slice &i) {
  return std::get<1>(i);
}

// 屏蔽关于 NVCC 对于没有返回语句的警告
// 这是 NVCC 和 NVC++ 中已知的问题
#if defined __NVCC__
    #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
        #pragma nv_diagnostic push
        #pragma nv_diag_suppress = implicit_return_from_non_void_function
    #else
      #ifdef __CUDA_ARCH__
        #pragma diagnostic push
        #pragma diag_suppress implicit_return_from_non_void_function
      #endif
    #endif
#elif defined __NVCOMPILER
    #pragma diagnostic push
    #pragma diag_suppress = implicit_return_from_non_void_function
#endif
// 获取 full_extent_t 对应维度的范围
template <size_t k, class Extents>
MDSPAN_INLINE_FUNCTION
constexpr auto last_of(std::integral_constant<size_t, k>, const Extents &ext,
                       ::MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent_t) {
  if constexpr (Extents::static_extent(k) == dynamic_extent) {
    return ext.extent(k);
  } else {
    return std::integral_constant<size_t, Extents::static_extent(k)>();
  }
#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
  // 即使使用 CUDA_ARCH 保护，这里也会警告调用主机函数
  __builtin_unreachable();
#endif
}
#if defined __NVCC__
    #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
        #pragma nv_diagnostic pop
    #else
      #ifdef __CUDA_ARCH__
        #pragma diagnostic pop
      #endif
    #endif
#elif defined __NVCOMPILER
    #pragma diagnostic pop
#endif

// 获取 strided_slice 结构体中的 extent 成员变量
template <size_t k, class Extents, class OffsetType, class ExtentType,
          class StrideType>
MDSPAN_INLINE_FUNCTION
constexpr OffsetType
last_of(std::integral_constant<size_t, k>, const Extents &,
        const strided_slice<OffsetType, ExtentType, StrideType> &r) {
  return r.extent;
}

// 获取 slice 的 stride
template <class T>
MDSPAN_INLINE_FUNCTION
constexpr auto stride_of(const T &) {
  return std::integral_constant<size_t, 1>();
}
// MDSPAN_INLINE_FUNCTION宏定义，用于声明内联函数
MDSPAN_INLINE_FUNCTION
// 计算给定 strided_slice 对象的步长值并返回
constexpr auto stride_of(const strided_slice<OffsetType, ExtentType, StrideType> &r) {
    return r.stride;
}

// 带有整数常量保留的除法函数模板
template <class IndexT, class T0, class T1>
MDSPAN_INLINE_FUNCTION
// 对两个值进行除法操作，并返回结果
constexpr auto divide(const T0 &v0, const T1 &v1) {
    return IndexT(v0) / IndexT(v1);
}

// 带有整数常量保留的除法函数模板，处理整数常量的情况
template <class IndexT, class T0, T0 v0, class T1, T1 v1>
MDSPAN_INLINE_FUNCTION
// 当除数为零时，返回0；用于处理 strided_slice 的零范围/步长的情况
constexpr auto divide(const std::integral_constant<T0, v0> &,
                      const std::integral_constant<T1, v1> &) {
    // 防止除零操作，如果 v0 为零，则结果为零，否则为 v0 除以 v1
    return std::integral_constant<IndexT, v0 == 0 ? 0 : v0 / v1>();
}

// 带有整数常量保留的乘法函数模板
template <class IndexT, class T0, class T1>
MDSPAN_INLINE_FUNCTION
// 对两个值进行乘法操作，并返回结果
constexpr auto multiply(const T0 &v0, const T1 &v1) {
    return IndexT(v0) * IndexT(v1);
}

// 带有整数常量保留的乘法函数模板，处理整数常量的情况
template <class IndexT, class T0, T0 v0, class T1, T1 v1>
MDSPAN_INLINE_FUNCTION
// 对整数常量 v0 和 v1 进行乘法操作，并返回结果
constexpr auto multiply(const std::integral_constant<T0, v0> &,
                        const std::integral_constant<T1, v1> &) {
    return std::integral_constant<IndexT, v0 * v1>();
}

// 根据范围计算新的静态长度，保留静态信息
template <class Arg0, class Arg1> struct StaticExtentFromRange {
    // 默认情况下，返回动态长度 dynamic_extent
    constexpr static size_t value = dynamic_extent;
};

// 根据整数常量范围计算新的静态长度
template <class Integral0, Integral0 val0, class Integral1, Integral1 val1>
struct StaticExtentFromRange<std::integral_constant<Integral0, val0>,
                             std::integral_constant<Integral1, val1>> {
    // 计算静态长度为 val1 - val0
    constexpr static size_t value = val1 - val0;
};

// 根据 strided_slice 计算新的静态长度，保留静态信息
template <class Arg0, class Arg1> struct StaticExtentFromStridedRange {
    // 默认情况下，返回动态长度 dynamic_extent
    constexpr static size_t value = dynamic_extent;
};

// 根据整数常量 strided_slice 计算新的静态长度
template <class Integral0, Integral0 val0, class Integral1, Integral1 val1>
struct StaticExtentFromStridedRange<std::integral_constant<Integral0, val0>,
                                    std::integral_constant<Integral1, val1>> {
    // 计算静态长度，如果步长大于零，则计算为 1 + (val0 - 1) / val1；否则返回0
    constexpr static size_t value = val0 > 0 ? 1 + (val0 - 1) / val1 : 0;
};

// 通过递归调用 next_extent 成员函数创建新的 extents
template <size_t K, class Extents, size_t... NewExtents>
struct extents_constructor {
    // 类模板的静态断言，要求 Slice 类型不能转换为 size_t，且不是 strided_slice
    MDSPAN_TEMPLATE_REQUIRES(
      class Slice, class... SlicesAndExtents,
      /* requires */(!std::is_convertible_v<Slice, size_t> &&
                     !is_strided_slice<Slice>::value)
    )
    // 内联函数，通过调用 next_extent 计算下一个 extent
    MDSPAN_INLINE_FUNCTION
    constexpr static auto next_extent(const Extents &ext, const Slice &sl,
                                      SlicesAndExtents... slices_and_extents) {
    // 计算新的静态维度，根据给定的 Slice 参数推断
    constexpr size_t new_static_extent = StaticExtentFromRange<
        decltype(first_of(std::declval<Slice>())),
        decltype(last_of(std::integral_constant<size_t, Extents::rank() - K>(),
                         std::declval<Extents>(),
                         std::declval<Slice>()))>::value;

    // 定义下一级的 extents_constructor 类型，添加新计算的静态维度到 NewExtents 中
    using next_t =
        extents_constructor<K - 1, Extents, NewExtents..., new_static_extent>;

    // 使用最后一个 Slice 计算新的索引范围，并调整 Extents，返回下一级的 next_extent 结果
    using index_t = typename Extents::index_type;
    return next_t::next_extent(
        ext, slices_and_extents...,
        index_t(last_of(std::integral_constant<size_t, Extents::rank() - K>(), ext,
                        sl)) -
            index_t(first_of(sl)));
  }

  // 当 Slice 类型可以转换为 size_t 时，处理下一级的 next_extent 调用
  MDSPAN_TEMPLATE_REQUIRES(
    class Slice, class... SlicesAndExtents,
    /* requires */ (std::is_convertible_v<Slice, size_t>)
  )
  MDSPAN_INLINE_FUNCTION
  // 推断下一级的 next_extent 返回类型，继续处理剩余的 slices_and_extents
  constexpr static auto next_extent(const Extents &ext, const Slice &,
                                    SlicesAndExtents... slices_and_extents) {
    using next_t = extents_constructor<K - 1, Extents, NewExtents...>;
    return next_t::next_extent(ext, slices_and_extents...);
  }

  // 处理 strided_slice 类型的情况，计算新的静态维度或者动态维度
  template <class OffsetType, class ExtentType, class StrideType,
            class... SlicesAndExtents>
  MDSPAN_INLINE_FUNCTION
  constexpr static auto
  // 当前级别处理 strided_slice 的 next_extent 计算
  next_extent(const Extents &ext,
              const strided_slice<OffsetType, ExtentType, StrideType> &r,
              SlicesAndExtents... slices_and_extents) {
    using index_t = typename Extents::index_type;
    using new_static_extent_t =
        StaticExtentFromStridedRange<ExtentType, StrideType>;

    // 根据新的静态维度类型决定下一级的处理方式
    if constexpr (new_static_extent_t::value == dynamic_extent) {
      // 处理动态维度的情况，根据 strided_slice 计算新的动态维度
      using next_t =
          extents_constructor<K - 1, Extents, NewExtents..., dynamic_extent>;
      return next_t::next_extent(
          ext, slices_and_extents...,
          r.extent > 0 ? 1 + divide<index_t>(r.extent - 1, r.stride) : 0);
    } else {
      // 处理静态维度的情况，计算新的静态维度并继续处理
      constexpr size_t new_static_extent = new_static_extent_t::value;
      using next_t =
          extents_constructor<K - 1, Extents, NewExtents..., new_static_extent>;
      return next_t::next_extent(
          ext, slices_and_extents..., index_t(divide<index_t>(ExtentType(), StrideType())));
    }
  }
};

template <class Extents, size_t... NewStaticExtents>
struct extents_constructor<0, Extents, NewStaticExtents...> {

  template <class... NewExtents>
  MDSPAN_INLINE_FUNCTION
  constexpr static auto next_extent(const Extents &, NewExtents... new_exts) {
    return extents<typename Extents::index_type, NewStaticExtents...>(
        new_exts...);
  }
};

} // namespace detail

// submdspan_extents creates new extents given src extents and submdspan slice
// specifiers
template <class IndexType, size_t... Extents, class... SliceSpecifiers>
MDSPAN_INLINE_FUNCTION
constexpr auto submdspan_extents(const extents<IndexType, Extents...> &src_exts,
                                 SliceSpecifiers... slices) {

  using ext_t = extents<IndexType, Extents...>;
  return detail::extents_constructor<ext_t::rank(), ext_t>::next_extent(
      src_exts, slices...);
}
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
//END_FILE_INCLUDE: mdspan/include/experimental/__p2630_bits/submdspan_extents.hpp
//BEGIN_FILE_INCLUDE: mdspan/include/experimental/__p2630_bits/submdspan_mapping.hpp
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER


#include <array>
#include <type_traits>
#include <tuple>
#include <utility> // index_sequence

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
//******************************************
// Return type of submdspan_mapping overloads
//******************************************
template <class LayoutMapping> struct submdspan_mapping_result {
  _MDSPAN_NO_UNIQUE_ADDRESS LayoutMapping mapping{};
  size_t offset;
};

namespace detail {

// We use const Slice& and not Slice&& because the various
// submdspan_mapping_impl overloads use their slices arguments
// multiple times.  This makes perfect forwarding not useful, but we
// still don't want to pass those (possibly of size 64 x 3 bits)
// objects by value.
template<class IndexType,
         class Slice>
MDSPAN_INLINE_FUNCTION
constexpr bool
one_slice_out_of_bounds(const IndexType& extent, const Slice& slice)
{
  using common_t = std::common_type_t<decltype(detail::first_of(slice)), IndexType>;
  return static_cast<common_t>(detail::first_of(slice)) == static_cast<common_t>(extent);
}

template<size_t ... RankIndices,
         class IndexType, size_t ... Exts,
         class ... Slices>
MDSPAN_INLINE_FUNCTION
constexpr bool
// 辅助函数，用于检查任意索引是否超出范围
template <size_t... RankIndices, class IndexType, size_t... Exts, class... Slices>
MDSPAN_INLINE_FUNCTION
constexpr bool
any_slice_out_of_bounds_helper(std::index_sequence<RankIndices...>,
                               const extents<IndexType, Exts...>& exts,
                               const Slices& ... slices)
{
  // 调用单个索引超出范围检查函数，并通过折叠表达式连接结果
  return _MDSPAN_FOLD_OR(
    (one_slice_out_of_bounds(exts.extent(RankIndices), slices))
  );
}

// 主模板函数，检查是否有任何切片超出边界
template<class IndexType, size_t ... Exts, class ... Slices>
MDSPAN_INLINE_FUNCTION
constexpr bool
any_slice_out_of_bounds(const extents<IndexType, Exts...>& exts,
                        const Slices& ... slices)
{
  // 调用辅助函数，传递切片参数和索引序列
  return any_slice_out_of_bounds_helper(
    std::make_index_sequence<sizeof...(Slices)>(),
    exts, slices...);
}
  
// 构造子步长函数
template <class SrcMapping, class... slice_strides, size_t... InvMapIdxs>
MDSPAN_INLINE_FUNCTION
constexpr auto
construct_sub_strides(const SrcMapping &src_mapping,
                      std::index_sequence<InvMapIdxs...>,
                      const std::tuple<slice_strides...> &slices_stride_factor) {
  using index_type = typename SrcMapping::index_type;
  // 使用源映射和切片步长因子创建步长数组
  return std::array<typename SrcMapping::index_type, sizeof...(InvMapIdxs)>{
      (static_cast<index_type>(src_mapping.stride(InvMapIdxs)) *
       static_cast<index_type>(std::get<InvMapIdxs>(slices_stride_factor)))...};
}

} // namespace detail

// layout_left submdspan_mapping
namespace detail {

// 确定是否保留 layout_left 的映射
template <class IndexSequence, size_t SubRank, class... SliceSpecifiers>
struct preserve_layout_left_mapping;

template <class... SliceSpecifiers, size_t... Idx, size_t SubRank>
struct preserve_layout_left_mapping<std::index_sequence<Idx...>, SubRank,
                                    SliceSpecifiers...> {
  constexpr static bool value =
      // 对于 rank 0，保留 layout
      (SubRank == 0) ||
      (
          // 切片规范需要保留到 subrank 之前为 full_extent_t - 除了最后一个可能是 tuple 但不是 strided index range
          // subrank 之后的切片规范必须是整数
          ((Idx > SubRank - 1) || // 这些是仅整数切片规范
           (std::is_same_v<SliceSpecifiers, full_extent_t>) ||
           ((Idx == SubRank - 1) &&
            std::is_convertible_v<SliceSpecifiers, std::tuple<size_t, size_t>>)) &&
          ...);
};
} // namespace detail

// 抑制 NVCC 关于无返回语句的警告
#if defined __NVCC__
    #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
        #pragma nv_diagnostic push
        #pragma nv_diag_suppress = implicit_return_from_non_void_function
    #else
      #ifdef __CUDA_ARCH__
        #pragma diagnostic push
        #pragma diag_suppress implicit_return_from_non_void_function
      #endif
    #endif
#elif defined __NVCOMPILER
// 如果定义了 __NVCOMPILER，执行以下代码段
    #pragma diagnostic push
    // 压栈保存当前诊断状态
    #pragma diag_suppress = implicit_return_from_non_void_function
    // 抑制隐式从非 void 函数返回的诊断信息
#endif

// 实际的 submdspan 映射调用
template <class Extents>
template <class... SliceSpecifiers>
// 内联函数定义，返回类型自动推断为 auto
MDSPAN_INLINE_FUNCTION
constexpr auto
layout_left::mapping<Extents>::submdspan_mapping_impl(SliceSpecifiers... slices) const {

  // 计算子范围
  using src_ext_t = Extents;
  // 计算目标范围
  auto dst_ext = submdspan_extents(extents(), slices...);
  using dst_ext_t = decltype(dst_ext);

  // 确定子布局类型
  constexpr bool preserve_layout = detail::preserve_layout_left_mapping<
      decltype(std::make_index_sequence<src_ext_t::rank()>()), dst_ext_t::rank(),
      SliceSpecifiers...>::value;
  using dst_layout_t =
      std::conditional_t<preserve_layout, layout_left, layout_stride>;
  using dst_mapping_t = typename dst_layout_t::template mapping<dst_ext_t>;

  // 确定是否有任何切片的下界等于相应的范围
  // 如果是，则绕过评估布局映射。这修复了 LWG Issue 4060。
  const bool out_of_bounds =
    detail::any_slice_out_of_bounds(this->extents(), slices...);
  auto offset = static_cast<size_t>(
    out_of_bounds ?
    this->required_span_size() :
    this->operator()(detail::first_of(slices)...)
  );

  if constexpr (std::is_same_v<dst_layout_t, layout_left>) {
    // 布局为 layout_left 的情况
    return submdspan_mapping_result<dst_mapping_t>{dst_mapping_t(dst_ext), offset};
  } else {
    // 布局为 layout_stride 的情况
    auto inv_map = detail::inv_map_rank(
      std::integral_constant<size_t,0>(),
      std::index_sequence<>(),
      slices...
    );
    return submdspan_mapping_result<dst_mapping_t>{
        dst_mapping_t(dst_ext, detail::construct_sub_strides(
                                   *this, inv_map,
    // HIP 需要推导指南来有标记，所以我们需要明确说明
    // NVCC 11.0 在这里有一个推导指南的 bug，测试显示 11.2 没有这个问题
    // 但是 Clang-CUDA 也不接受推导指南的使用，所以在 CUDA 中要完全禁用它
    #if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
                                   std::tuple<decltype(detail::stride_of(slices))...>{detail::stride_of(slices)...})),
    #else
                                   std::tuple{detail::stride_of(slices)...})),
    #endif
        offset};
  }
#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
  // 如果满足条件，指示编译器此处不可达
  __builtin_unreachable();
#endif
}

#if defined __NVCC__
    #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
        #pragma nv_diagnostic pop
    #else
      #ifdef __CUDA_ARCH__
        #pragma diagnostic pop
      #endif
    #endif
#elif defined __NVCOMPILER
    #pragma diagnostic pop
#endif
// 弹出并恢复之前的诊断状态

//**********************************
// layout_right submdspan_mapping
//*********************************
namespace detail {

// 确定是否保留 layout_right
template <class IndexSequence, size_t SubRank, class... SliceSpecifiers>
// 声明一个未定义的结构体 preserve_layout_right_mapping
struct preserve_layout_right_mapping;

// 模板定义：保持右侧布局映射
template <class... SliceSpecifiers, size_t... Idx, size_t SubRank>
struct preserve_layout_right_mapping<std::index_sequence<Idx...>, SubRank,
                                     SliceSpecifiers...> {
  // 源维度数
  constexpr static size_t SrcRank = sizeof...(SliceSpecifiers);
  // 是否保持布局的布尔值：
  // - 对于子维度为0的情况，始终保持布局
  // - 对于其他情况，根据规则判断是否保持布局
  constexpr static bool value =
      (SubRank == 0) ||
      (
          // 根据规则判断：最后的子维度切片规范应为 full_extent_t
          // 除了 srcrank-subrank 外，还可以是元组但不能是 strided index range
          // 在 srcrank-subrank 之前的切片规范应为整数
          ((Idx < SrcRank - SubRank) ||
           (std::is_same_v<SliceSpecifiers, full_extent_t>) ||
           ((Idx == SrcRank - SubRank) &&
            std::is_convertible_v<SliceSpecifiers, std::tuple<size_t, size_t>>)) &&
          ...);
};
} // namespace detail

// 命名空间结束

// 禁止 NVCC 对于无返回语句的警告，这是 NVCC 和 NVC++ 中已知的问题
#if defined __NVCC__
    #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
        #pragma nv_diagnostic push
        #pragma nv_diag_suppress = implicit_return_from_non_void_function
    #else
      #ifdef __CUDA_ARCH__
        #pragma diagnostic push
        #pragma diag_suppress implicit_return_from_non_void_function
      #endif
    #endif
// 对于其他编译器的类似处理
#elif defined __NVCOMPILER
    #pragma diagnostic push
    #pragma diag_suppress = implicit_return_from_non_void_function
#endif

// 模板方法定义：submdspan_mapping_impl
template <class Extents>
template <class... SliceSpecifiers>
// constexpr 修饰的内联函数
MDSPAN_INLINE_FUNCTION
constexpr auto
layout_right::mapping<Extents>::submdspan_mapping_impl(
                  SliceSpecifiers... slices) const {
  // 获取源扩展
  using src_ext_t = Extents;
  // 计算目标扩展
  auto dst_ext = submdspan_extents(extents(), slices...);
  using dst_ext_t = decltype(dst_ext);

  // 确定新的布局类型
  constexpr bool preserve_layout = detail::preserve_layout_right_mapping<
      decltype(std::make_index_sequence<src_ext_t::rank()>()), dst_ext_t::rank(),
      SliceSpecifiers...>::value;
  // 根据是否保持布局选择不同的目标布局类型
  using dst_layout_t =
      std::conditional_t<preserve_layout, layout_right, layout_stride>;
  // 目标布局映射类型
  using dst_mapping_t = typename dst_layout_t::template mapping<dst_ext_t>;

  // 判断是否有任何切片的下限等于相应的扩展。如果是，则绕过评估布局映射。这修复了 LWG Issue 4060。
  const bool out_of_bounds =
    detail::any_slice_out_of_bounds(this->extents(), slices...);
  // 计算偏移量，如果切片越界，则使用所需的跨度大小
  auto offset = static_cast<size_t>(
    out_of_bounds ?
    this->required_span_size() :
    this->operator()(detail::first_of(slices)...)
  );
  
  // 如果目标布局类型是 layout_right
  if constexpr (std::is_same_v<dst_layout_t, layout_right>) {
    // layout_right 的情况



    // 在 layout_right 的情况下，返回计算得到的目标映射
    return dst_mapping_t{dst_ext, offset};
  } else {
    // 其他情况，返回计算得到的目标映射，并指定偏移量
    return dst_mapping_t{dst_ext, offset, this->required_span_size()};
  }
}


这样就完成了对给定 C++ 代码块的注释。
    return submdspan_mapping_result<dst_mapping_t>{dst_mapping_t(dst_ext), offset};
  } else {
    // layout_stride case
    auto inv_map = detail::inv_map_rank(
      std::integral_constant<size_t,0>(),
      std::index_sequence<>(),
      slices...);
    return submdspan_mapping_result<dst_mapping_t>{
        dst_mapping_t(dst_ext, detail::construct_sub_strides(
                                   *this, inv_map,
                                   // HIP needs deduction guides to have markups so we need to be explicit
                                   // NVCC 11.0 has a bug with deduction guide here, tested that 11.2 does not have the issue
                                   // But Clang-CUDA also doesn't accept the use of deduction guide so disable it for CUDA alltogether
    #if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
                                   std::tuple<decltype(detail::stride_of(slices))...>{detail::stride_of(slices)...})),
    #else
                                   std::tuple{detail::stride_of(slices)...})),
    #endif
        offset};
  }



    return submdspan_mapping_result<dst_mapping_t>{dst_mapping_t(dst_ext), offset};
    // 如果条件为真，返回一个以目标映射类型为键的结果字典，其中包含目标扩展和偏移量

  } else {
    // layout_stride case
    // 计算逆映射，使用零大小常量和空索引序列初始化
    auto inv_map = detail::inv_map_rank(
      std::integral_constant<size_t,0>(),
      std::index_sequence<>(),
      slices...);
    // 返回一个以目标映射类型为键的结果字典，包括目标扩展、子数组步长和偏移量
    return submdspan_mapping_result<dst_mapping_t>{
        dst_mapping_t(dst_ext, detail::construct_sub_strides(
                                   *this, inv_map,
                                   // 如果是 HIP 环境，需要显式指定推导指南的标记
                                   // NVCC 11.0 在此处有一个推导指南的 bug，已测试 11.2 版本没有此问题
                                   // 但是 Clang-CUDA 也不接受推导指南的使用，因此对 CUDA 完全禁用它
    #if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
                                   std::tuple<decltype(detail::stride_of(slices))...>{detail::stride_of(slices)...})),
    #else
                                   std::tuple{detail::stride_of(slices)...})),
    #endif
        offset};
  }


这段代码根据条件返回不同的 `submdspan_mapping_result` 结果，具体取决于条件表达式的值。
#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
  // 如果是在 NVCC 编译环境中，且不是在 CUDA 设备代码中，并且使用的是 GCC 编译器
  __builtin_unreachable();
#endif
}
#if defined __NVCC__
    #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
        // 如果支持 NVCC 的诊断指令弹出
        #pragma nv_diagnostic pop
    #else
      #ifdef __CUDA_ARCH__
        // 如果在 CUDA 设备代码中
        #pragma diagnostic pop
      #endif
    #endif
#elif defined __NVCOMPILER
    // 如果在 NVCOMPILER 编译环境中
    #pragma    diagnostic pop
#endif

//**********************************
// layout_stride submdspan_mapping
//*********************************
template <class Extents>
template <class... SliceSpecifiers>
MDSPAN_INLINE_FUNCTION
constexpr auto
layout_stride::mapping<Extents>::submdspan_mapping_impl(
                  SliceSpecifiers... slices) const {
  // 计算子多维跨步映射的实现
  auto dst_ext = submdspan_extents(extents(), slices...);
  using dst_ext_t = decltype(dst_ext);
  // 计算反向映射
  auto inv_map = detail::inv_map_rank(
      std::integral_constant<size_t,0>(),
      std::index_sequence<>(),
      slices...);
  using dst_mapping_t = typename layout_stride::template mapping<dst_ext_t>;

  // 解决 LWG Issue 4060：检查是否有任何切片的下界等于对应的维度大小，如果是，则绕过布局映射的评估
  const bool out_of_bounds =
    detail::any_slice_out_of_bounds(this->extents(), slices...);
  // 计算偏移量
  auto offset = static_cast<size_t>(
    out_of_bounds ?
    this->required_span_size() :
    this->operator()(detail::first_of(slices)...));

  // 返回子多维跨步映射结果
  return submdspan_mapping_result<dst_mapping_t>{
      dst_mapping_t(dst_ext, detail::construct_sub_strides(
                                 *this, inv_map,
    // HIP 需要推导指南来有标记，因此需要显式指定
    // NVCC 11.0 在这里存在推导指南的 bug，已测试 11.2 不再有此问题
    #if defined(_MDSPAN_HAS_HIP) || (defined(__NVCC__) && (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10) < 1120)
                                 std::tuple<decltype(detail::stride_of(slices))...>(detail::stride_of(slices)...))),
#else
                                 std::tuple(detail::stride_of(slices)...))),
#endif
      offset};
}

} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
//END_FILE_INCLUDE: mdspan/include/experimental/__p2630_bits/submdspan_mapping.hpp

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
// 对于 ElementType、Extents、LayoutPolicy、AccessorPolicy 和 SliceSpecifiers 参数的多维跨步映射实现
template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy, class... SliceSpecifiers>
MDSPAN_INLINE_FUNCTION
constexpr auto
// 定义名为 submdspan 的函数模板，接受一个常量引用 src，以及变长模板参数 SliceSpecifiers
submdspan(const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> &src,
          SliceSpecifiers... slices) {
  // 调用 submdspan_mapping 函数，传入 src 的映射和 slices 参数，获取子范围映射的结果
  const auto sub_submdspan_mapping_result = submdspan_mapping(src.mapping(), slices...);
  
  // 由于 NVCC 存在类型推断问题，因此明确 sub_mapping_t 的类型
  using sub_mapping_t = std::remove_cv_t<decltype(sub_submdspan_mapping_result.mapping)>;
  // 定义子范围的 extents_type 类型为 sub_mapping_t 的 extents_type
  using sub_extents_t = typename sub_mapping_t::extents_type;
  // 定义子范围的 layout_type 类型为 sub_mapping_t 的 layout_type
  using sub_layout_t = typename sub_mapping_t::layout_type;
  // 定义子范围的 offset_policy 类型为 AccessorPolicy 的 offset_policy
  using sub_accessor_t = typename AccessorPolicy::offset_policy;
  
  // 创建一个新的 mdspan 对象，用子范围的相关信息进行初始化，并返回该对象
  return mdspan<ElementType, sub_extents_t, sub_layout_t, sub_accessor_t>(
      // 调用 src.accessor().offset 函数，传入 src 的 data_handle 和子范围的偏移量，获取新的偏移量
      src.accessor().offset(src.data_handle(), sub_submdspan_mapping_result.offset),
      // 使用子范围映射的结果作为新 mdspan 对象的 mapping
      sub_submdspan_mapping_result.mapping,
      // 用 src 的 accessor 初始化新 mdspan 对象的 accessor
      sub_accessor_t(src.accessor()));
}
```