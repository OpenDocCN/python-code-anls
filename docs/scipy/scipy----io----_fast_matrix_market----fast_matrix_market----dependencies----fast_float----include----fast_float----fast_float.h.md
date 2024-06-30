# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\fast_float\include\fast_float\fast_float.h`

```
// fast_float by Daniel Lemire
// fast_float by João Paulo Magalhaes
//
// with contributions from Eugene Golushkov
// with contributions from Maksim Kita
// with contributions from Marcin Wojdyr
// with contributions from Neal Richardson
// with contributions from Tim Paine
// with contributions from Fabio Pellacini
// with contributions from Lénárd Szolnoki
// with contributions from Jan Pharago
//
// Licensed under the Apache License, Version 2.0, or the
// MIT License or the Boost License. This file may not be copied,
// modified, or distributed except according to those terms.
//
// MIT License Notice
//
//    MIT License
//
//    Copyright (c) 2021 The fast_float authors
//
//    Permission is hereby granted, free of charge, to any
//    person obtaining a copy of this software and associated
//    documentation files (the "Software"), to deal in the
//    Software without restriction, including without
//    limitation the rights to use, copy, modify, merge,
//    publish, distribute, sublicense, and/or sell copies of
//    the Software, and to permit persons to whom the Software
//    is furnished to do so, subject to the following
//    conditions:
//
//    The above copyright notice and this permission notice
//    shall be included in all copies or substantial portions
//    of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
//    ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
//    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
//    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
//    SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//    OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//    IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//    DEALINGS IN THE SOFTWARE.
//
// Apache License (Version 2.0) Notice
//
//    Copyright 2021 The fast_float authors
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//
// BOOST License Notice
//
//    Boost Software License - Version 1.0 - August 17th, 2003
//
//    Permission is hereby granted, free of charge, to any person or organization
//    obtaining a copy of the software and accompanying documentation covered by
//    this license (the "Software") to use, reproduce, display, distribute,
//    execute, and transmit the Software, and to prepare derivative works of the
//    Software, and to permit third-parties to whom the Software is furnished to
// 定义头文件防卫式声明，避免头文件被多次包含
#ifndef FASTFLOAT_CONSTEXPR_FEATURE_DETECT_H
#define FASTFLOAT_CONSTEXPR_FEATURE_DETECT_H

// 检查是否支持 __has_include 宏，如果支持且包含 <version> 头文件，则尝试包含该头文件
#ifdef __has_include
#if __has_include(<version>)
#include <version>
#endif
#endif

// 根据 C++ 标准版本检查 constexpr 的支持情况，设置 FASTFLOAT_CONSTEXPR14 宏
#if __cpp_constexpr >= 201304
#define FASTFLOAT_CONSTEXPR14 constexpr
#else
#define FASTFLOAT_CONSTEXPR14
#endif

// 检查是否支持 std::bit_cast 特性，并设置 FASTFLOAT_HAS_BIT_CAST 宏
#if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
#define FASTFLOAT_HAS_BIT_CAST 1
#else
#define FASTFLOAT_HAS_BIT_CAST 0
#endif

// 检查是否支持 std::is_constant_evaluated 特性，并设置 FASTFLOAT_HAS_IS_CONSTANT_EVALUATED 宏
#if defined(__cpp_lib_is_constant_evaluated) && __cpp_lib_is_constant_evaluated >= 201811L
#define FASTFLOAT_HAS_IS_CONSTANT_EVALUATED 1
#else
#define FASTFLOAT_HAS_IS_CONSTANT_EVALUATED 0
#endif

// 根据 C++20 constexpr 库特性的支持情况，设置 FASTFLOAT_CONSTEXPR20 和 FASTFLOAT_IS_CONSTEXPR 宏
#if FASTFLOAT_HAS_IS_CONSTANT_EVALUATED \
    && FASTFLOAT_HAS_BIT_CAST \
    && __cpp_lib_constexpr_algorithms >= 201806L /*For std::copy and std::fill*/
#define FASTFLOAT_CONSTEXPR20 constexpr
#define FASTFLOAT_IS_CONSTEXPR 1
#else
#define FASTFLOAT_CONSTEXPR20
#define FASTFLOAT_IS_CONSTEXPR 0
#endif

// 结束头文件防卫式声明
#endif // FASTFLOAT_CONSTEXPR_FEATURE_DETECT_H

// 定义头文件防卫式声明，避免头文件被多次包含
#ifndef FASTFLOAT_FLOAT_COMMON_H
#define FASTFLOAT_FLOAT_COMMON_H

// 包含 C++ 标准库头文件
#include <cfloat>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <system_error>

// fast_float 命名空间
namespace fast_float {

// 枚举类型，表示字符格式选项
enum chars_format {
  scientific = 1 << 0,
  fixed = 1 << 2,
  hex = 1 << 3,
  general = fixed | scientific
};

// 解析结果结构模板，用于 from_chars 函数的返回值
template <typename UC>
struct from_chars_result_t {
  UC const* ptr; // 指向解析后的字符的指针
  std::errc ec;  // 解析操作的错误码
};

// 使用 char 类型作为模板参数的解析结果类型别名
using from_chars_result = from_chars_result_t<char>;

// 解析选项结构模板，用于指定解析函数的行为选项
template <typename UC>
struct parse_options_t {
  constexpr explicit parse_options_t(chars_format fmt = chars_format::general,
    UC dot = UC('.'))
    : format(fmt), decimal_point(dot) {}

  chars_format format; // 数字格式选项
  UC decimal_point;    // 小数点字符
};

// 使用 char 类型作为模板参数的解析选项类型别名
using parse_options = parse_options_t<char>;

}

// 如果支持 std::bit_cast 特性，则包含 <bit> 头文件
#if FASTFLOAT_HAS_BIT_CAST
#include <bit>
#endif

// 结束头文件防卫式声明
#endif // FASTFLOAT_FLOAT_COMMON_H
#if (defined(__x86_64) || defined(__x86_64__) || defined(_M_X64)   \
       || defined(__amd64) || defined(__aarch64__) || defined(_M_ARM64) \
       || defined(__MINGW64__)                                          \
       || defined(__s390x__)                                            \
       || (defined(__ppc64__) || defined(__PPC64__) || defined(__ppc64le__) || defined(__PPC64LE__)) )
// 如果目标平台被定义为64位架构的一种（x86_64, amd64, aarch64, ARM64等），则定义FASTFLOAT_64BIT为1
#define FASTFLOAT_64BIT 1
#elif (defined(__i386) || defined(__i386__) || defined(_M_IX86)   \
     || defined(__arm__) || defined(_M_ARM) || defined(__ppc__)   \
     || defined(__MINGW32__) || defined(__EMSCRIPTEN__))
// 如果目标平台被定义为32位架构的一种（x86, ARM, ppc等），则定义FASTFLOAT_32BIT为1
#define FASTFLOAT_32BIT 1
#else
// 如果无法确定平台位数，则逐步检查SIZE_MAX以避免溢出，并选择对应的位数定义
  // 无法判断寄存器位数，但SIZE_MAX是一个很好的近似值
  // UINTPTR_MAX和INTPTR_MAX是可选的，为了最大的可移植性，避免使用它们
  #if SIZE_MAX == 0xffff
    #error Unknown platform (16-bit, unsupported)
  #elif SIZE_MAX == 0xffffffff
    #define FASTFLOAT_32BIT 1
  #elif SIZE_MAX == 0xffffffffffffffff
    #define FASTFLOAT_64BIT 1
  #else
    #error Unknown platform (not 32-bit, not 64-bit?)
  #endif
#endif

#if ((defined(_WIN32) || defined(_WIN64)) && !defined(__clang__))
#include <intrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
// 如果是Visual Studio编译器，则定义FASTFLOAT_VISUAL_STUDIO为1
#define FASTFLOAT_VISUAL_STUDIO 1
#endif

#if defined __BYTE_ORDER__ && defined __ORDER_BIG_ENDIAN__
// 如果定义了__BYTE_ORDER__和__ORDER_BIG_ENDIAN__，则根据它们的值定义FASTFLOAT_IS_BIG_ENDIAN
#define FASTFLOAT_IS_BIG_ENDIAN (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#elif defined _WIN32
// 如果在Windows平台上，则定义FASTFLOAT_IS_BIG_ENDIAN为0（小端序）
#define FASTFLOAT_IS_BIG_ENDIAN 0
#else
// 否则，根据平台的字节序定义FASTFLOAT_IS_BIG_ENDIAN
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <machine/endian.h>
#elif defined(sun) || defined(__sun)
#include <sys/byteorder.h>
#else
#ifdef __has_include
#if __has_include(<endian.h>)
#include <endian.h>
#endif //__has_include(<endian.h>)
#endif //__has_include
#endif
#
#ifndef __BYTE_ORDER__
// 如果未定义__BYTE_ORDER__，则安全地将FASTFLOAT_IS_BIG_ENDIAN定义为0
#define FASTFLOAT_IS_BIG_ENDIAN 0
#endif
#
#ifndef __ORDER_LITTLE_ENDIAN__
// 如果未定义__ORDER_LITTLE_ENDIAN__，则安全地将FASTFLOAT_IS_BIG_ENDIAN定义为0
#define FASTFLOAT_IS_BIG_ENDIAN 0
#endif
#
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
// 根据平台的字节序定义FASTFLOAT_IS_BIG_ENDIAN
#define FASTFLOAT_IS_BIG_ENDIAN 0
#else
#define FASTFLOAT_IS_BIG_ENDIAN 1
#endif
#endif

#ifdef FASTFLOAT_VISUAL_STUDIO
// 如果是Visual Studio编译器，则定义fastfloat_really_inline为__forceinline
#define fastfloat_really_inline __forceinline
#else
// 否则定义fastfloat_really_inline为inline并标记为总是内联
#define fastfloat_really_inline inline __attribute__((always_inline))
#endif

#ifndef FASTFLOAT_ASSERT
// 如果未定义FASTFLOAT_ASSERT宏，则定义为空语句块
#define FASTFLOAT_ASSERT(x)  { ((void)(x)); }
#endif

#ifndef FASTFLOAT_DEBUG_ASSERT
// 如果未定义FASTFLOAT_DEBUG_ASSERT宏，则定义为空语句块
#define FASTFLOAT_DEBUG_ASSERT(x) { ((void)(x)); }
#endif

// rust style `try!()` macro, or `?` operator
// 定义一个类似于Rust的`try!()`宏的宏，用于检查条件并在条件不满足时返回false
#define FASTFLOAT_TRY(x) { if (!(x)) return false; }

namespace fast_float {

fastfloat_really_inline constexpr bool cpp20_and_in_constexpr() {
// 如果支持C++20的is_constant_evaluated函数，则返回其结果，否则返回false
#if FASTFLOAT_HAS_IS_CONSTANT_EVALUATED
  return std::is_constant_evaluated();
#else
  return false;
#endif
}

// Compares two ASCII strings in a case insensitive manner.
// 在不区分大小写的情况下比较两个ASCII字符串的模板函数声明
template <typename UC>
inline FASTFLOAT_CONSTEXPR14 bool
// 比较两个以 null 结尾的字符数组 input1 和 input2 的前 length 个字符是否相等（不区分大小写）
fastfloat_strncasecmp(UC const * input1, UC const * input2, size_t length) {
  // 初始化 running_diff 为 0，用于记录字符比较的差异
  char running_diff{0};
  // 遍历长度为 length 的字符数组
  for (size_t i = 0; i < length; ++i) {
    // 使用按位异或操作比较 input1 和 input2 的字符，并将结果存入 running_diff
    running_diff |= (char(input1[i]) ^ char(input2[i]));
  }
  // 返回比较结果：running_diff 是否为 0 或者 32
  return (running_diff == 0) || (running_diff == 32);
}

#ifndef FLT_EVAL_METHOD
// 如果 FLT_EVAL_METHOD 未定义，则编译错误，要求包含 cfloat 头文件
#error "FLT_EVAL_METHOD should be defined, please include cfloat."
#endif

// 表示指向连续内存块的指针及其长度
template <typename T>
struct span {
  const T* ptr;      // 指向内存块的指针
  size_t length;     // 内存块的长度

  // 构造函数：初始化指针和长度
  constexpr span(const T* _ptr, size_t _length) : ptr(_ptr), length(_length) {}
  // 默认构造函数：指针为空，长度为 0
  constexpr span() : ptr(nullptr), length(0) {}

  // 返回内存块的长度
  constexpr size_t len() const noexcept {
    return length;
  }

  // 重载操作符 []，返回指定索引位置的元素引用
  FASTFLOAT_CONSTEXPR14 const T& operator[](size_t index) const noexcept {
    // 断言索引在有效范围内
    FASTFLOAT_DEBUG_ASSERT(index < length);
    return ptr[index];
  }
};

// 128 位整数类型
struct value128 {
  uint64_t low;     // 低位 64 位
  uint64_t high;    // 高位 64 位

  // 构造函数：初始化低位和高位
  constexpr value128(uint64_t _low, uint64_t _high) : low(_low), high(_high) {}
  // 默认构造函数：低位和高位均为 0
  constexpr value128() : low(0), high(0) {}
};

/* Helper C++11 constexpr generic implementation of leading_zeroes */
// 通用的计算输入数的前导零个数的 constexpr 函数实现
fastfloat_really_inline constexpr
int leading_zeroes_generic(uint64_t input_num, int last_bit = 0) {
  return (
    // 逐步检查 input_num 中从高位到低位每个字节，确定前导零个数
    ((input_num & uint64_t(0xffffffff00000000)) && (input_num >>= 32, last_bit |= 32)),
    ((input_num & uint64_t(        0xffff0000)) && (input_num >>= 16, last_bit |= 16)),
    ((input_num & uint64_t(            0xff00)) && (input_num >>=  8, last_bit |=  8)),
    ((input_num & uint64_t(              0xf0)) && (input_num >>=  4, last_bit |=  4)),
    ((input_num & uint64_t(               0xc)) && (input_num >>=  2, last_bit |=  2)),
    ((input_num & uint64_t(               0x2)) && (input_num >>=  1, last_bit |=  1)),
    // 返回前导零的数目
    63 - last_bit
  );
}

/* result might be undefined when input_num is zero */
// 当 input_num 为零时，结果可能未定义的快速计算前导零的 constexpr 函数
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
int leading_zeroes(uint64_t input_num) {
  // 断言输入数大于 0
  assert(input_num > 0);
  // 如果编译环境支持 C++20 constexpr
  if (cpp20_and_in_constexpr()) {
    // 调用通用的前导零计算函数
    return leading_zeroes_generic(input_num);
  }
  // 对于其他编译环境，根据平台使用不同的方法计算前导零
#ifdef FASTFLOAT_VISUAL_STUDIO
  // 如果是 Visual Studio 编译环境
  #if defined(_M_X64) || defined(_M_ARM64)
  unsigned long leading_zero = 0;
  // 使用 _BitScanReverse64 函数从高位到低位查找设置位（1）
  // 返回位于 input_num 中最高设置位的索引
  _BitScanReverse64(&leading_zero, input_num);
  // 返回前导零的数目
  return (int)(63 - leading_zero);
  #else
  // 其他平台使用通用的前导零计算函数
  return leading_zeroes_generic(input_num);
  #endif
#else
  // 使用内置函数 __builtin_clzll 计算前导零的数目
  return __builtin_clzll(input_num);
#endif
}

// 用于 32 位乘法的慢速模拟例程
fastfloat_really_inline constexpr uint64_t emulu(uint32_t x, uint32_t y) {
    // 返回 x 和 y 的乘积，结果类型为 uint64_t
    return x * (uint64_t)y;
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR14
// 定义函数 umul128_generic，计算两个 64 位整数的乘积，同时返回高位结果
uint64_t umul128_generic(uint64_t ab, uint64_t cd, uint64_t *hi) {
  // 使用 emulu 函数计算高位和低位部分的乘积
  uint64_t ad = emulu((uint32_t)(ab >> 32), (uint32_t)cd);
  uint64_t bd = emulu((uint32_t)ab, (uint32_t)cd);
  // 计算跨越两个 32 位乘法结果的高位和低位相加的中间结果
  uint64_t adbc = ad + emulu((uint32_t)ab, (uint32_t)(cd >> 32));
  // 检查 adbc 是否产生了进位，如果产生了则设置 adbc_carry 为 1，否则为 0
  uint64_t adbc_carry = !!(adbc < ad);
  // 计算最终的低位结果
  uint64_t lo = bd + (adbc << 32);
  // 计算最终的高位结果，包括剩余的乘法结果和可能的进位
  *hi = emulu((uint32_t)(ab >> 32), (uint32_t)(cd >> 32)) + (adbc >> 32) +
        (adbc_carry << 32) + !!(lo < bd);
  // 返回低位结果
  return lo;
}

#ifdef FASTFLOAT_32BIT

// 为 32 位平台提供的 _umul128 函数的慢速模拟实现
#if !defined(__MINGW64__)
// 在非 MinGW64 编译环境下定义 _umul128 函数
fastfloat_really_inline FASTFLOAT_CONSTEXPR14
uint64_t _umul128(uint64_t ab, uint64_t cd, uint64_t *hi) {
  return umul128_generic(ab, cd, hi);
}
#endif // !__MINGW64__

#endif // FASTFLOAT_32BIT

// 计算两个 64 位整数的完整乘积，并以 value128 结构体返回
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
value128 full_multiplication(uint64_t a, uint64_t b) {
  // 在 constexpr 环境下使用 cpp20_and_in_constexpr 函数
  if (cpp20_and_in_constexpr()) {
    // 如果在 constexpr 环境下，使用 umul128_generic 计算低位结果，并初始化高位
    value128 answer;
    answer.low = umul128_generic(a, b, &answer.high);
    return answer;
  }
  // 否则，在非 constexpr 环境下，根据不同平台执行不同的乘法操作
  value128 answer;
#if defined(_M_ARM64) && !defined(__MINGW32__)
  // 对于 ARM64 平台，使用 __umulh 和乘法指令计算乘积
  answer.high = __umulh(a, b);
  answer.low = a * b;
#elif defined(FASTFLOAT_32BIT) || (defined(_WIN64) && !defined(__clang__))
  // 对于 32 位或特定 Windows 64 位平台，调用 _umul128 函数计算乘积
  answer.low = _umul128(a, b, &answer.high); // ARM64 平台不支持 _umul128
#elif defined(FASTFLOAT_64BIT)
  // 对于 64 位平台，使用 __uint128_t 类型计算乘积
  __uint128_t r = ((__uint128_t)a) * b;
  answer.low = uint64_t(r);
  answer.high = uint64_t(r >> 64);
#else
  // 默认情况下使用 umul128_generic 函数计算乘积
  answer.low = umul128_generic(a, b, &answer.high);
#endif
  // 返回计算结果
  return answer;
}

// 结构体 adjusted_mantissa，包含一个 64 位的 mantissa 和一个 32 位的 power2
struct adjusted_mantissa {
  uint64_t mantissa{0};
  int32_t power2{0}; // 负值表示无效结果
  adjusted_mantissa() = default;
  // 重载相等运算符，用于比较两个 adjusted_mantissa 结构体是否相等
  constexpr bool operator==(const adjusted_mantissa &o) const {
    return mantissa == o.mantissa && power2 == o.power2;
  }
  // 重载不等运算符，用于比较两个 adjusted_mantissa 结构体是否不相等
  constexpr bool operator!=(const adjusted_mantissa &o) const {
    return mantissa != o.mantissa || power2 != o.power2;
  }
};

// 用于 binary_format_lookup_tables<T>::max_mantissa 的常量定义
constexpr uint64_t constant_55555 = 5 * 5 * 5 * 5 * 5;

// 为 binary_format_lookup_tables 结构模板提供特化，使用默认模板参数 U = void
template <typename T, typename U = void>
struct binary_format_lookup_tables;
// 定义一个模板结构体 binary_format，继承自 binary_format_lookup_tables<T>
template <typename T> struct binary_format : binary_format_lookup_tables<T> {
    // 使用条件语句确定 equiv_uint 类型为 uint32_t 或 uint64_t，具体取决于 T 的大小
    using equiv_uint = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;

    // 声明静态成员函数，返回明确指定的尾数位数
    static inline constexpr int mantissa_explicit_bits();
    
    // 声明静态成员函数，返回最小指数
    static inline constexpr int minimum_exponent();
    
    // 声明静态成员函数，返回无穷大的幂
    static inline constexpr int infinite_power();
    
    // 声明静态成员函数，返回符号位的索引
    static inline constexpr int sign_index();
    
    // 声明静态成员函数，返回在 fegetround() == FE_TONEAREST 时使用的最小指数快速路径
    static inline constexpr int min_exponent_fast_path();
    
    // 声明静态成员函数，返回在 fegetround() == FE_TONEAREST 时使用的最大指数快速路径
    static inline constexpr int max_exponent_fast_path();
    
    // 声明静态成员函数，返回在偶数舍入模式下使用的最大指数
    static inline constexpr int max_exponent_round_to_even();
    
    // 声明静态成员函数，返回在偶数舍入模式下使用的最小指数
    static inline constexpr int min_exponent_round_to_even();
    
    // 声明静态成员函数，返回在给定幂时使用的最大尾数快速路径
    static inline constexpr uint64_t max_mantissa_fast_path(int64_t power);
    
    // 声明静态成员函数，返回在 fegetround() == FE_TONEAREST 时使用的最大尾数快速路径
    static inline constexpr uint64_t max_mantissa_fast_path();
    
    // 声明静态成员函数，返回最大的十的幂
    static inline constexpr int largest_power_of_ten();
    
    // 声明静态成员函数，返回最小的十的幂
    static inline constexpr int smallest_power_of_ten();
    
    // 声明静态成员函数，返回指定幂的精确十的幂
    static inline constexpr T exact_power_of_ten(int64_t power);
    
    // 声明静态成员函数，返回最大位数
    static inline constexpr size_t max_digits();
    
    // 声明静态成员函数，返回指数掩码
    static inline constexpr equiv_uint exponent_mask();
    
    // 声明静态成员函数，返回尾数掩码
    static inline constexpr equiv_uint mantissa_mask();
    
    // 声明静态成员函数，返回隐藏位掩码
    static inline constexpr equiv_uint hidden_bit_mask();
};
# 定义一个模板结构体，用于双精度浮点数和模板类型 U 的二进制格式查找表
struct binary_format_lookup_tables<double, U> {

  # 存储十进制的幂的数组，对应于10的不同次幂
  static constexpr double powers_of_ten[] = {
      1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
      1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

  # 最大的整数值 v，使得 (5**index * v) <= 1<<53
  # 0x10000000000000 == 1 << 53
  static constexpr uint64_t max_mantissa[] = {
      0x10000000000000,
      0x10000000000000 / 5,
      0x10000000000000 / (5 * 5),
      0x10000000000000 / (5 * 5 * 5),
      0x10000000000000 / (5 * 5 * 5 * 5),
      0x10000000000000 / (constant_55555),  # 假设 constant_55555 是某个常数
      0x10000000000000 / (constant_55555 * 5),
      0x10000000000000 / (constant_55555 * 5 * 5),
      0x10000000000000 / (constant_55555 * 5 * 5 * 5),
      0x10000000000000 / (constant_55555 * 5 * 5 * 5 * 5),
      0x10000000000000 / (constant_55555 * constant_55555),
      0x10000000000000 / (constant_55555 * constant_55555 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * 5 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * 5 * 5 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * 5 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * 5 * 5 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * 5 * 5 * 5 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * constant_55555),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * constant_55555 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * constant_55555 * 5 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * constant_55555 * 5 * 5 * 5),
      0x10000000000000 / (constant_55555 * constant_55555 * constant_55555 * constant_55555 * 5 * 5 * 5 * 5)};
};

# 实例化模板中的静态成员 powers_of_ten
template <typename U>
constexpr double binary_format_lookup_tables<double, U>::powers_of_ten[];

# 实例化模板中的静态成员 max_mantissa
template <typename U>
constexpr uint64_t binary_format_lookup_tables<double, U>::max_mantissa[];

# 模板的结尾
template <typename U>
// 定义模板结构体 binary_format_lookup_tables，处理 float 类型和模板参数 U
template <typename U>
struct binary_format_lookup_tables<float, U> {
    // 存储十次幂的常量数组
    static constexpr float powers_of_ten[] = {1e0f, 1e1f, 1e2f, 1e3f, 1e4f, 1e5f,
                                     1e6f, 1e7f, 1e8f, 1e9f, 1e10f};

    // 存储最大尾数的常量数组，使得 (5**index * v) <= 1<<24
    // 0x1000000 == 1<<24
    static constexpr uint64_t max_mantissa[] = {
        0x1000000,
        0x1000000 / 5,
        0x1000000 / (5 * 5),
        0x1000000 / (5 * 5 * 5),
        0x1000000 / (5 * 5 * 5 * 5),
        0x1000000 / (constant_55555),  // 使用常量 constant_55555 计算
        0x1000000 / (constant_55555 * 5),
        0x1000000 / (constant_55555 * 5 * 5),
        0x1000000 / (constant_55555 * 5 * 5 * 5),
        0x1000000 / (constant_55555 * 5 * 5 * 5 * 5),
        0x1000000 / (constant_55555 * constant_55555),
        0x1000000 / (constant_55555 * constant_55555 * 5)
    };
};

// 实例化模板特化，定义常量数组 powers_of_ten
template <typename U>
constexpr float binary_format_lookup_tables<float, U>::powers_of_ten[];

// 实例化模板特化，定义常量数组 max_mantissa
template <typename U>
constexpr uint64_t binary_format_lookup_tables<float, U>::max_mantissa[];

// 对 double 类型特化的 binary_format 结构体的 min_exponent_fast_path 方法的 constexpr 实现
template <> inline constexpr int binary_format<double>::min_exponent_fast_path() {
    // 如果浮点数评估方法不是 1 也不是 0，则返回 0
#if (FLT_EVAL_METHOD != 1) && (FLT_EVAL_METHOD != 0)
    return 0;
#else
    // 否则返回 -22
    return -22;
#endif
}

// 对 float 类型特化的 binary_format 结构体的 min_exponent_fast_path 方法的 constexpr 实现
template <> inline constexpr int binary_format<float>::min_exponent_fast_path() {
    // 如果浮点数评估方法不是 1 也不是 0，则返回 0
#if (FLT_EVAL_METHOD != 1) && (FLT_EVAL_METHOD != 0)
    return 0;
#else
    // 否则返回 -10
    return -10;
#endif
}

// 对 double 类型特化的 binary_format 结构体的 mantissa_explicit_bits 方法的 constexpr 实现，返回 52
template <> inline constexpr int binary_format<double>::mantissa_explicit_bits() {
    return 52;
}

// 对 float 类型特化的 binary_format 结构体的 mantissa_explicit_bits 方法的 constexpr 实现，返回 23
template <> inline constexpr int binary_format<float>::mantissa_explicit_bits() {
    return 23;
}

// 对 double 类型特化的 binary_format 结构体的 max_exponent_round_to_even 方法的 constexpr 实现，返回 23
template <> inline constexpr int binary_format<double>::max_exponent_round_to_even() {
    return 23;
}

// 对 float 类型特化的 binary_format 结构体的 max_exponent_round_to_even 方法的 constexpr 实现，返回 10
template <> inline constexpr int binary_format<float>::max_exponent_round_to_even() {
    return 10;
}

// 对 double 类型特化的 binary_format 结构体的 min_exponent_round_to_even 方法的 constexpr 实现，返回 -4
template <> inline constexpr int binary_format<double>::min_exponent_round_to_even() {
    return -4;
}

// 对 float 类型特化的 binary_format 结构体的 min_exponent_round_to_even 方法的 constexpr 实现，返回 -17
template <> inline constexpr int binary_format<float>::min_exponent_round_to_even() {
    return -17;
}

// 对 double 类型特化的 binary_format 结构体的 minimum_exponent 方法的 constexpr 实现，返回 -1023
template <> inline constexpr int binary_format<double>::minimum_exponent() {
    return -1023;
}

// 对 float 类型特化的 binary_format 结构体的 minimum_exponent 方法的 constexpr 实现，返回 -127
template <> inline constexpr int binary_format<float>::minimum_exponent() {
    return -127;
}

// 对 double 类型特化的 binary_format 结构体的 infinite_power 方法的 constexpr 实现，返回 0x7FF
template <> inline constexpr int binary_format<double>::infinite_power() {
    return 0x7FF;
}

// 对 float 类型特化的 binary_format 结构体的 infinite_power 方法的 constexpr 实现，返回 0xFF
template <> inline constexpr int binary_format<float>::infinite_power() {
    return 0xFF;
}

// 对 double 类型特化的 binary_format 结构体的 sign_index 方法的 constexpr 实现，返回 63
template <> inline constexpr int binary_format<double>::sign_index() { return 63; }

// 对 float 类型特化的 binary_format 结构体的 sign_index 方法的 constexpr 实现，返回 31
template <> inline constexpr int binary_format<float>::sign_index() { return 31; }

// 对 double 类型特化的 binary_format 结构体的 max_exponent_fast_path 方法的 constexpr 实现，返回 22
template <> inline constexpr int binary_format<double>::max_exponent_fast_path() {
    return 22;
}

// 对 float 类型特化的 binary_format 结构体的 max_exponent_fast_path 方法的 constexpr 实现，返回 10
template <> inline constexpr int binary_format<float>::max_exponent_fast_path() {
    return 10;
}

// 对 double 类型特化的 binary_format 结构体的 max_mantissa_fast_path 方法的 constexpr 实现
template <> inline constexpr uint64_t binary_format<double>::max_mantissa_fast_path() {
    // 返回 2 左移 mantissa_explicit_bits() 位的结果
    return uint64_t(2) << mantissa_explicit_bits();
}
// 返回一个常量表达式，表示给定浮点数类型的最大有效位数，依赖于模板参数 T
template <> inline constexpr size_t binary_format<double>::max_digits() {
    return 769;
}
// 返回一个常量表达式，表示给定浮点数类型的最大有效位数，依赖于模板参数 T
template <> inline constexpr size_t binary_format<float>::max_digits() {
    return 114;
}

// 返回一个常量表达式，表示给定浮点数类型的最大尾数（mantissa）的快速路径，依赖于模板参数 T
template <> inline constexpr uint64_t binary_format<double>::max_mantissa_fast_path(int64_t power) {
    // 调用者需确保 power >= 0 && power <= 22
    //
    // 绕过 clang 的 bug https://godbolt.org/z/zedh7rrhc
    return (void)max_mantissa[0], max_mantissa[power];
}

// 返回一个常量表达式，表示给定浮点数类型的最大尾数（mantissa）的快速路径，依赖于模板参数 T
template <> inline constexpr uint64_t binary_format<float>::max_mantissa_fast_path() {
    return uint64_t(2) << mantissa_explicit_bits();
}

// 返回一个常量表达式，表示给定浮点数类型的最大尾数（mantissa）的快速路径，依赖于模板参数 T
template <> inline constexpr uint64_t binary_format<float>::max_mantissa_fast_path(int64_t power) {
    // 调用者需确保 power >= 0 && power <= 10
    //
    // 绕过 clang 的 bug https://godbolt.org/z/zedh7rrhc
    return (void)max_mantissa[0], max_mantissa[power];
}

// 返回一个常量表达式，表示给定浮点数类型的确切十的幂，依赖于模板参数 T
template <>
inline constexpr double binary_format<double>::exact_power_of_ten(int64_t power) {
    // 绕过 clang 的 bug https://godbolt.org/z/zedh7rrhc
    return (void)powers_of_ten[0], powers_of_ten[power];
}

// 返回一个常量表达式，表示给定浮点数类型的确切十的幂，依赖于模板参数 T
template <>
inline constexpr float binary_format<float>::exact_power_of_ten(int64_t power) {
    // 绕过 clang 的 bug https://godbolt.org/z/zedh7rrhc
    return (void)powers_of_ten[0], powers_of_ten[power];
}

// 返回一个常量表达式，表示给定浮点数类型的最大的十的幂，依赖于模板参数 T
template <>
inline constexpr int binary_format<double>::largest_power_of_ten() {
    return 308;
}

// 返回一个常量表达式，表示给定浮点数类型的最大的十的幂，依赖于模板参数 T
template <>
inline constexpr int binary_format<float>::largest_power_of_ten() {
    return 38;
}

// 返回一个常量表达式，表示给定浮点数类型的最小的十的幂，依赖于模板参数 T
template <>
inline constexpr int binary_format<double>::smallest_power_of_ten() {
    return -342;
}

// 返回一个常量表达式，表示给定浮点数类型的最小的十的幂，依赖于模板参数 T
template <>
inline constexpr int binary_format<float>::smallest_power_of_ten() {
    return -65;
}

// 返回一个常量表达式，表示给定浮点数类型的指数掩码，依赖于模板参数 T
template <> inline constexpr binary_format<float>::equiv_uint
    binary_format<float>::exponent_mask() {
    return 0x7F800000;
}

// 返回一个常量表达式，表示给定浮点数类型的指数掩码，依赖于模板参数 T
template <> inline constexpr binary_format<double>::equiv_uint
    binary_format<double>::exponent_mask() {
    return 0x7FF0000000000000;
}

// 返回一个常量表达式，表示给定浮点数类型的尾数掩码，依赖于模板参数 T
template <> inline constexpr binary_format<float>::equiv_uint
    binary_format<float>::mantissa_mask() {
    return 0x007FFFFF;
}

// 返回一个常量表达式，表示给定浮点数类型的尾数掩码，依赖于模板参数 T
template <> inline constexpr binary_format<double>::equiv_uint
    binary_format<double>::mantissa_mask() {
    return 0x000FFFFFFFFFFFFF;
}

// 返回一个常量表达式，表示给定浮点数类型的隐藏位掩码，依赖于模板参数 T
template <> inline constexpr binary_format<float>::equiv_uint
    binary_format<float>::hidden_bit_mask() {
    return 0x00800000;
}

// 返回一个常量表达式，表示给定浮点数类型的隐藏位掩码，依赖于模板参数 T
template <> inline constexpr binary_format<double>::equiv_uint
    binary_format<double>::hidden_bit_mask() {
    return 0x0010000000000000;
}

// 将调整后的尾数和幂次转换为浮点数值，依赖于模板参数 T
template<typename T>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
void to_float(bool negative, adjusted_mantissa am, T &value) {
    // 使用模板参数 T 中定义的等价整数类型
    using uint = typename binary_format<T>::equiv_uint;
    uint word = (uint)am.mantissa;
    // 将尾数与幂次信息编码到一个整数中
    word |= uint(am.power2) << binary_format<T>::mantissa_explicit_bits();
    // 将符号信息编码到整数中
    word |= uint(negative) << binary_format<T>::sign_index();
    // 如果支持 std::bit_cast，则直接使用，否则通过内存拷贝实现类型转换
#if FASTFLOAT_HAS_BIT_CAST
    value = std::bit_cast<T>(word);
#else
    ::memcpy(&value, &word, sizeof(T));
#endif
}
#ifdef FASTFLOAT_SKIP_WHITE_SPACE // disabled by default
// 空白字符查找表模板，用于快速判断字符是否为空白字符
template <typename = void>
struct space_lut {
  // ASCII 码中的空白字符用 1 表示，非空白字符用 0 表示
  static constexpr bool value[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

// 判断一个字符是否为空白字符
template <typename T>
constexpr bool space_lut<T>::value[];

// 内联函数，判断一个字符是否为空白字符
inline constexpr bool is_space(uint8_t c) { return space_lut<>::value[c]; }
#endif

// 没有包含 FASTFLOAT_FAST_FLOAT_H 的情况下，定义 fast_float 命名空间
#ifndef FASTFLOAT_FAST_FLOAT_H
namespace fast_float {
/**
 * This function checks if the character `c` represents a digit between '0' and '9'.
 * Returns true if `c` is a digit, false otherwise.
 */
fastfloat_really_inline constexpr bool is_integer(UC c) noexcept {
  return !(c > UC('9') || c < UC('0'));
}

/**
 * This function performs byte swapping on a 64-bit unsigned integer `val`.
 * Byte swapping reverses the byte order of `val` from big-endian to little-endian or vice versa.
 * Returns the byte-swapped value.
 */
fastfloat_really_inline constexpr uint64_t byteswap(uint64_t val) {
  return (val & 0xFF00000000000000) >> 56
    | (val & 0x00FF000000000000) >> 40
    | (val & 0x0000FF0000000000) >> 24
    | (val & 0x000000FF00000000) >> 8
    | (val & 0x00000000FF000000) << 8
    | (val & 0x0000000000FF0000) << 24
    | (val & 0x000000000000FF00) << 40
    | (val & 0x00000000000000FF) << 56;
}

/**
 * This function reads an unsigned 64-bit integer from the character array `chars`.
 * It interprets the first 8 characters of `chars` as a little-endian encoded 64-bit integer.
 * Returns the decoded 64-bit unsigned integer.
 */
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
uint64_t read_u64(const char *chars) {
  if (cpp20_and_in_constexpr()) {
    uint64_t val = 0;
    for(int i = 0; i < 8; ++i) {
      val |= uint64_t(*chars) << (i*8);
      ++chars;
    }
    return val;
  } else {
    // Non-constexpr fallback (should not occur in constexpr context)
    return reinterpret_cast<const uint64_t*>(chars)[0];
  }
}
    }
    // 返回变量 val 的值
    return val;
  }
  // 声明一个名为 val 的 uint64_t 类型变量
  uint64_t val;
  // 将 chars 指向的内存中的数据复制到 val 变量中，复制的长度为 uint64_t 的大小
  ::memcpy(&val, chars, sizeof(uint64_t));
#if FASTFLOAT_IS_BIG_ENDIAN == 1
  // 如果 FASTFLOAT_IS_BIG_ENDIAN 等于 1，则需要按照小端序读取数字。
  val = byteswap(val);
#endif
  // 返回 uint64_t 类型的值 val。
  return val;
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
void write_u64(uint8_t *chars, uint64_t val) {
  if (cpp20_and_in_constexpr()) {
    // 如果当前处于 constexpr 上下文中，则按字节写入 uint64_t 值到 chars 数组。
    for(int i = 0; i < 8; ++i) {
      *chars = uint8_t(val);
      val >>= 8;
      ++chars;
    }
    return;
  }
#if FASTFLOAT_IS_BIG_ENDIAN == 1
  // 如果 FASTFLOAT_IS_BIG_ENDIAN 等于 1，则需要按照小端序写入数字。
  val = byteswap(val);
#endif
  // 否则，使用 memcpy 将 val 的字节表示直接复制到 chars 数组。
  ::memcpy(chars, &val, sizeof(uint64_t));
}

// credit  @aqrit
fastfloat_really_inline FASTFLOAT_CONSTEXPR14
uint32_t parse_eight_digits_unrolled(uint64_t val) {
  const uint64_t mask = 0x000000FF000000FF;
  const uint64_t mul1 = 0x000F424000000064; // 100 + (1000000ULL << 32)
  const uint64_t mul2 = 0x0000271000000001; // 1 + (10000ULL << 32)
  // 对 val 执行偏移和乘法操作，解析并返回其中八位数字的值。
  val -= 0x3030303030303030;
  val = (val * 10) + (val >> 8); // val = (val * 2561) >> 8;
  val = (((val & mask) * mul1) + (((val >> 16) & mask) * mul2)) >> 32;
  return uint32_t(val);
}

fastfloat_really_inline constexpr
uint32_t parse_eight_digits_unrolled(const char16_t *)  noexcept  {
  // 对于 char16_t 类型的输入，直接返回 0。
  return 0;
}

fastfloat_really_inline constexpr
uint32_t parse_eight_digits_unrolled(const char32_t *)  noexcept  {
  // 对于 char32_t 类型的输入，直接返回 0。
  return 0;
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
uint32_t parse_eight_digits_unrolled(const char *chars)  noexcept  {
  // 调用 read_u64 函数解析 chars 所指向的字符串，然后调用前面定义的函数解析其中的八位数字。
  return parse_eight_digits_unrolled(read_u64(chars));
}

// credit @aqrit
fastfloat_really_inline constexpr bool is_made_of_eight_digits_fast(uint64_t val)  noexcept  {
  // 检查 val 是否由八位数字组成，如果是则返回 true，否则返回 false。
  return !((((val + 0x4646464646464646) | (val - 0x3030303030303030)) &
     0x8080808080808080));
}

fastfloat_really_inline constexpr
bool is_made_of_eight_digits_fast(const char16_t *)  noexcept  {
  // 对于 char16_t 类型的输入，直接返回 false。
  return false;
}

fastfloat_really_inline constexpr
bool is_made_of_eight_digits_fast(const char32_t *)  noexcept  {
  // 对于 char32_t 类型的输入，直接返回 false。
  return false;
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
bool is_made_of_eight_digits_fast(const char *chars)  noexcept  {
  // 调用 read_u64 函数解析 chars 所指向的字符串，然后调用前面定义的函数检查其中是否全部由八位数字组成。
  return is_made_of_eight_digits_fast(read_u64(chars));
}

template <typename UC>
struct parsed_number_string_t {
  int64_t exponent{0};
  uint64_t mantissa{0};
  UC const * lastmatch{nullptr};
  bool negative{false};
  bool valid{false};
  bool too_many_digits{false};
  // 包含重要数字范围的 span
  span<const UC> integer{};  // 不可为空
  span<const UC> fraction{}; // 可为空
};
using byte_span = span<char>;
using parsed_number_string = parsed_number_string_t<char>;
// 假设不超过 19 位数字，这段代码将解析 ASCII 字符串。
template <typename UC>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
// 解析数字字符串，返回解析结果结构体 parsed_number_string_t<UC>
parsed_number_string_t<UC> parse_number_string(UC const *p, UC const * pend, parse_options_t<UC> options) noexcept {
    // 获取解析选项中的格式和小数点字符
    chars_format const fmt = options.format;
    UC const decimal_point = options.decimal_point;

    // 初始化解析结果结构体，设定 valid 和 too_many_digits 初始值为 false
    parsed_number_string_t<UC> answer;
    answer.valid = false;
    answer.too_many_digits = false;
    // 判断是否为负数
    answer.negative = (*p == UC('-'));

    // 如果允许有前置加号，则 '+' 或 '-' 后移一位
#ifdef FASTFLOAT_ALLOWS_LEADING_PLUS // 默认禁用
    if ((*p == UC('-')) || (*p == UC('+'))) {
#else
    // 否则只允许 '-'，C++17 20.19.3.(7.1) 明确禁止 '+' 号在此位置
    if (*p == UC('-')) {
#endif
        ++p;
        // 若移动后已到达 pend，则返回初始结果
        if (p == pend) {
            return answer;
        }
        // 若不是整数或小数点，则返回初始结果
        if (!is_integer(*p) && (*p != decimal_point)) {
            return answer;
        }
    }

    // 记录整数部分起始位置
    UC const * const start_digits = p;

    uint64_t i = 0; // 使用无符号整数，避免有符号溢出

    // 解析整数部分，乘以 10 比任意整数乘法更便宜
    while ((p != pend) && is_integer(*p)) {
        i = 10 * i + uint64_t(*p - UC('0')); // 可能会溢出，稍后处理溢出情况
        ++p;
    }
    UC const * const end_of_integer_part = p;
    int64_t digit_count = int64_t(end_of_integer_part - start_digits);
    // 将整数部分存入 span
    answer.integer = span<const UC>(start_digits, size_t(digit_count));

    int64_t exponent = 0;
    // 如果遇到小数点
    if ((p != pend) && (*p == decimal_point)) {
        ++p;
        UC const * before = p;
        // 在满足条件下，快速解析八位数字的连续循环
        if (std::is_same<UC,char>::value) {
            while ((std::distance(p, pend) >= 8) && is_made_of_eight_digits_fast(p)) {
                i = i * 100000000 + parse_eight_digits_unrolled(p); // 在稀有情况下会溢出，但没关系
                p += 8;
            }
        }
        // 继续解析剩余整数部分
        while ((p != pend) && is_integer(*p)) {
            uint8_t digit = uint8_t(*p - UC('0'));
            ++p;
            i = i * 10 + digit; // 在稀有情况下会溢出，但没关系
        }
        // 计算小数点后的位数
        exponent = before - p;
        // 将小数部分存入 span
        answer.fraction = span<const UC>(before, size_t(p - before));
        digit_count -= exponent;
    }

    // 必须至少遇到一个整数！
    if (digit_count == 0) {
        return answer;
    }

    int64_t exp_number = 0; // 显式指数部分
    // 如果遇到科学计数法
    if ((fmt & chars_format::scientific) && (p != pend) && ((UC('e') == *p) || (UC('E') == *p))) {
        UC const * location_of_e = p;
        ++p;
        bool neg_exp = false;
        // 若指数部分有负号
        if ((p != pend) && (UC('-') == *p)) {
            neg_exp = true;
            ++p;
        } else if ((p != pend) && (UC('+') == *p)) {
            ++p;
        }
        // 若下一位不是整数，则根据格式判断是否报错
        if ((p == pend) || !is_integer(*p)) {
            if (!(fmt & chars_format::fixed)) {
                // 出现错误，返回初始结果
                return answer;
            }
            // 否则忽略 'e'
            p = location_of_e;
``
  } else {
    // 如果是科学计数法但不是固定格式，则返回当前答案
    if((fmt & chars_format::scientific) && !(fmt & chars_format::fixed)) { return answer; }
  }
  // 设置最后匹配的位置为当前解析指针
  answer.lastmatch = p;
  // 标记解析成功
  answer.valid = true;

  // 如果经常需要处理长数字字符串，
  // 我们可以通过使用128位整数而不是64位整数来扩展我们的代码。不过，这种情况不常见。
  //
  // 我们可以处理多达19位数字。
  if (digit_count > 19) { // 这种情况不常见
    // 可能整数溢出的情况。
    // 我们必须处理像0.0000somenumber这样的情况。
    // 我们需要注意只有零的情况...
    // 例如，0.000000000...000。
    UC const * start = start_digits;
    while ((start != pend) && (*start == UC('0') || *start == decimal_point)) {
      if(*start == UC('0')) { digit_count --; }
      start++;
    }
    if (digit_count > 19) {
      // 标记有过多的数字
      answer.too_many_digits = true;
      // 让我们重新开始，这次避免溢出。
      // 我们无需检查is_integer，因为我们使用上面预分词的跨度。
      i = 0;
      p = answer.integer.ptr;
      UC const * int_end = p + answer.integer.len();
      // 最小的19位数字整数常量
      const uint64_t minimal_nineteen_digit_integer{1000000000000000000};
      while((i < minimal_nineteen_digit_integer) && (p != int_end)) {
        i = i * 10 + uint64_t(*p - UC('0'));
        ++p;
      }
      if (i >= minimal_nineteen_digit_integer) { // 我们有一个大整数
        exponent = end_of_integer_part - p + exp_number;
      } else { // 我们有一个带有小数部分的值
          p = answer.fraction.ptr;
          UC const * frac_end = p + answer.fraction.len();
          while((i < minimal_nineteen_digit_integer) && (p != frac_end)) {
            i = i * 10 + uint64_t(*p - UC('0'));
            ++p;
          }
          exponent = answer.fraction.ptr - p + exp_number;
      }
      // 现在我们已经修正了指数和i，将其截断为一个值
    }
  }
  // 设置答案的指数和尾数
  answer.exponent = exponent;
  answer.mantissa = i;
  // 返回解析的结果
  return answer;
/**
 * Header guard to prevent multiple inclusion of this file.
 */
#ifndef FASTFLOAT_FAST_TABLE_H
#define FASTFLOAT_FAST_TABLE_H

#include <cstdint>  // Include for standard integer types

namespace fast_float {

/**
 * Templated struct to define powers of five for decimal to binary mapping.
 * This struct handles conversions from decimal to binary by managing powers of five.
 */
template <class unused = void>
struct powers_template {

/**
 * The smallest power of five needed for binary64 representation.
 * This value is retrieved from the binary_format<double> template.
 */
constexpr static int smallest_power_of_five = binary_format<double>::smallest_power_of_ten();

/**
 * The largest power of five needed for binary64 representation.
 * This value is retrieved from the binary_format<double> template.
 */
constexpr static int largest_power_of_five = binary_format<double>::largest_power_of_ten();

/**
 * Total number of entries needed for powers of five table.
 * This is calculated based on the range between smallest and largest powers of five.
 */
constexpr static int number_of_entries = 2 * (largest_power_of_five - smallest_power_of_five + 1);

/**
 * Array containing precomputed powers of five (5^k) rounded toward one,
 * stored as 128-bit unsigned integers (uint64_t).
 * These values are used for efficient conversion from decimal to binary.
 */
constexpr static uint64_t power_of_five_128[number_of_entries] = {
    0xeef453d6923bd65a, 0x113faa2906a13b3f,
    0x9558b4661b6565f8, 0x4ac7ca59a424c507,
    0xbaaee17fa23ebf76, 0x5d79bcf00d2df649,
    0xe95a99df8ace6f53, 0xf4d82c2c107973dc,
    0x91d8a02bb6c10594, 0x79071b9b8a4be869,
    0xb64ec836a47146f9, 0x9748e2826cdee284,
    0xe3e27a444d8d98b7, 0xfd1b1b2308169b25,
    0x8e6d8c6ab0787f72, 0xfe30f0f5e50e20f7,
    0xb208ef855c969f4f, 0xbdbd2d335e51a935,
    0xde8b2b66b3bc4723, 0xad2c788035e61382,
    0x8b16fb203055ac76, 0x4c3bcb5021afcc31,
    0xaddcb9e83c6b1793, 0xdf4abe242a1bbf3d,
    0xd953e8624b85dd78, 0xd71d6dad34a2af0d,
    0x87d4713d6f33aa6b, 0x8672648c40e5ad68,
    0xa9c98d8ccb009506, 0x680efdaf511f18c2,
    0xd43bf0effdc0ba48, 0x212bd1b2566def2,
    0x84a57695fe98746d, 0x14bb630f7604b57,
    0xa5ced43b7e3e9188, 0x419ea3bd35385e2d,
    0xcf42894a5dce35ea, 0x52064cac828675b9,
    0x818995ce7aa0e1b2, 0x7343efebd1940993,
    0xa1ebfb4219491a1f, 0x1014ebe6c5f90bf8,
    0xca66fa129f9b60a6, 0xd41a26e077774ef6,
    0xfd00b897478238d0, 0x8920b098955522b4,
    0x9e20735e8cb16382, 0x55b46e5f5d5535b0,
    0xc5a890362fddbc62, 0xeb2189f734aa831d,
    0xf712b443bbd52b7b, 0xa5e9ec7501d523e4,
    0x9a6bb0aa55653b2d, 0x47b233c92125366e,
    0xc1069cd4eabe89f8, 0x999ec0bb696e840a,
    0xf148440a256e2c76, 0xc00670ea43ca250d,
    0x96cd2a865764dbca, 0x380406926a5e5728,
    0xbc807527ed3e12bc, 0xc605083704f5ecf2,
    0xeba09271e88d976b, 0xf7864a44c633682e,
    0x93445b8731587ea3, 0x7ab3ee6afbe0211d,
};

}; // struct powers_template

} // namespace fast_float

#endif // FASTFLOAT_FAST_TABLE_H
    # 下面是一组十六进制数对，每一对都表示一个64位整数，以逗号分隔
    0xb8157268fdae9e4c,0x5960ea05bad82964,
    0xe61acf033d1a45df,0x6fb92487298e33bd,
    0x8fd0c16206306bab,0xa5d3b6d479f8e056,
    0xb3c4f1ba87bc8696,0x8f48a4899877186c,
    0xe0b62e2929aba83c,0x331acdabfe94de87,
    0x8c71dcd9ba0b4925,0x9ff0c08b7f1d0b14,
    0xaf8e5410288e1b6f,0x07ecf0ae5ee44dd9,
    0xdb71e91432b1a24a,0xc9e82cd9f69d6150,
    0x892731ac9faf056e,0xbe311c083a225cd2,
    0xab70fe17c79ac6ca,0x6dbd630a48aaf406,
    0xd64d3d9db981787d,0x092cbbccdad5b108,
    0x85f0468293f0eb4e,0x25bbf56008c58ea5,
    0xa76c582338ed2621,0xaf2af2b80af6f24e,
    0xd1476e2c07286faa,0x1af5af660db4aee1,
    0x82cca4db847945ca,0x50d98d9fc890ed4d,
    0xa37fce126597973c,0xe50ff107bab528a0,
    0xcc5fc196fefd7d0c,0x1e53ed49a96272c8,
    0xff77b1fcbebcdc4f,0x25e8e89c13bb0f7a,
    0x9faacf3df73609b1,0x77b191618c54e9ac,
    0xc795830d75038c1d,0xd59df5b9ef6a2417,
    0xf97ae3d0d2446f25,0x4b0573286b44ad1d,
    0x9becce62836ac577,0x4ee367f9430aec32,
    0xc2e801fb244576d5,0x229c41f793cda73f,
    0xf3a20279ed56d48a,0x6b43527578c1110f,
    0x9845418c345644d6,0x830a13896b78aaa9,
    0xbe5691ef416bd60c,0x23cc986bc656d553,
    0xedec366b11c6cb8f,0x2cbfbe86b7ec8aa8,
    0x94b3a202eb1c3f39,0x7bf7d71432f3d6a9,
    0xb9e08a83a5e34f07,0xdaf5ccd93fb0cc53,
    0xe858ad248f5c22c9,0xd1b3400f8f9cff68,
    0x91376c36d99995be,0x23100809b9c21fa1,
    0xb58547448ffffb2d,0xabd40a0c2832a78a,
    0xe2e69915b3fff9f9,0x16c90c8f323f516c,
    0x8dd01fad907ffc3b,0xae3da7d97f6792e3,
    0xb1442798f49ffb4a,0x99cd11cfdf41779c,
    0xdd95317f31c7fa1d,0x40405643d711d583,
    0x8a7d3eef7f1cfc52,0x482835ea666b2572,
    0xad1c8eab5ee43b66,0xda3243650005eecf,
    0xd863b256369d4a40,0x90bed43e40076a82,
    0x873e4f75e2224e68,0x5a7744a6e804a291,
    0xa90de3535aaae202,0x711515d0a205cb36,
    0xd3515c2831559a83,0xd5a5b44ca873e03,
    0x8412d9991ed58091,0xe858790afe9486c2,
    0xa5178fff668ae0b6,0x626e974dbe39a872,
    0xce5d73ff402d98e3,0xfb0a3d212dc8128f,
    0x80fa687f881c7f8e,0x7ce66634bc9d0b99,
    0xa139029f6a239f72,0x1c1fffc1ebc44e80,
    0xc987434744ac874e,0xa327ffb266b56220,
    0xfbe9141915d7a922,0x4bf1ff9f0062baa8,
    0x9d71ac8fada6c9b5,0x6f773fc3603db4a9,
    0xc4ce17b399107c22,0xcb550fb4384d21d3,
    0xf6019da07f549b2b,0x7e2a53a146606a48,
    0x99c102844f94e0fb,0x2eda7444cbfc426d,
    0xc0314325637a1939,0xfa911155fefb5308,
    0xf03d93eebc589f88,0x793555ab7eba27ca,
    0x96267c7535b763b5,0x4bc1558b2f3458de,
    0xbbb01b9283253ca2,0x9eb1aaedfb016f16,
    0xea9c227723ee8bcb,0x465e15a979c1cadc,
    0x92a1958a7675175f,0x0bfacd89ec191ec9,
    0xb749faed14125d36,0xcef980ec671f667b,
    0xe51c79a85916f484,0x82b7e12780e7401a,
    0x8f31cc0937ae58d2,0xd1b2ecb8b0908810,
    0xb2fe3f0b8599ef07,0x861fa7e6dcb4aa15,
    0xdfbdcece67006ac9,0x67a791e093e1d49a,
    0x8bd6a141006042bd,0xe0c8bb2c5c6d24e0,
    0xaecc49914078536d,0x58fae9f773886e18,
    0xda7f5bf590966848,0xaf39a475506a899e,
    0x888f99797a5e012d,0x6d8406c952429603,
    0xaab37fd7d8f58178,0xc8e5087ba6d33b83,
    # 以下是一组十六进制数对，似乎是用逗号分隔的 64 位整数的表示形式
    0xd5605fcdcf32e1d6,0xfb1e4a9a90880a64,
    0x855c3be0a17fcd26,0x5cf2eea09a55067f,
    0xa6b34ad8c9dfc06f,0xf42faa48c0ea481e,
    0xd0601d8efc57b08b,0xf13b94daf124da26,
    0x823c12795db6ce57,0x76c53d08d6b70858,
    0xa2cb1717b52481ed,0x54768c4b0c64ca6e,
    0xcb7ddcdda26da268,0xa9942f5dcf7dfd09,
    0xfe5d54150b090b02,0xd3f93b35435d7c4c,
    0x9efa548d26e5a6e1,0xc47bc5014a1a6daf,
    0xc6b8e9b0709f109a,0x359ab6419ca1091b,
    0xf867241c8cc6d4c0,0xc30163d203c94b62,
    0x9b407691d7fc44f8,0x79e0de63425dcf1d,
    0xc21094364dfb5636,0x985915fc12f542e4,
    0xf294b943e17a2bc4,0x3e6f5b7b17b2939d,
    0x979cf3ca6cec5b5a,0xa705992ceecf9c42,
    0xbd8430bd08277231,0x50c6ff782a838353,
    0xece53cec4a314ebd,0xa4f8bf5635246428,
    0x940f4613ae5ed136,0x871b7795e136be99,
    0xb913179899f68584,0x28e2557b59846e3f,
    0xe757dd7ec07426e5,0x331aeada2fe589cf,
    0x9096ea6f3848984f,0x3ff0d2c85def7621,
    0xb4bca50b065abe63,0xfed077a756b53a9,
    0xe1ebce4dc7f16dfb,0xd3e8495912c62894,
    0x8d3360f09cf6e4bd,0x64712dd7abbbd95c,
    0xb080392cc4349dec,0xbd8d794d96aacfb3,
    0xdca04777f541c567,0xecf0d7a0fc5583a0,
    0x89e42caaf9491b60,0xf41686c49db57244,
    0xac5d37d5b79b6239,0x311c2875c522ced5,
    0xd77485cb25823ac7,0x7d633293366b828b,
    0x86a8d39ef77164bc,0xae5dff9c02033197,
    0xa8530886b54dbdeb,0xd9f57f830283fdfc,
    0xd267caa862a12d66,0xd072df63c324fd7b,
    0x8380dea93da4bc60,0x4247cb9e59f71e6d,
    0xa46116538d0deb78,0x52d9be85f074e608,
    0xcd795be870516656,0x67902e276c921f8b,
    0x806bd9714632dff6,0xba1cd8a3db53b6,
    0xa086cfcd97bf97f3,0x80e8a40eccd228a4,
    0xc8a883c0fdaf7df0,0x6122cd128006b2cd,
    0xfad2a4b13d1b5d6c,0x796b805720085f81,
    0x9cc3a6eec6311a63,0xcbe3303674053bb0,
    0xc3f490aa77bd60fc,0xbedbfc4411068a9c,
    0xf4f1b4d515acb93b,0xee92fb5515482d44,
    0x991711052d8bf3c5,0x751bdd152d4d1c4a,
    0xbf5cd54678eef0b6,0xd262d45a78a0635d,
    0xef340a98172aace4,0x86fb897116c87c34,
    0x9580869f0e7aac0e,0xd45d35e6ae3d4da0,
    0xbae0a846d2195712,0x8974836059cca109,
    0xe998d258869facd7,0x2bd1a438703fc94b,
    0x91ff83775423cc06,0x7b6306a34627ddcf,
    0xb67f6455292cbf08,0x1a3bc84c17b1d542,
    0xe41f3d6a7377eeca,0x20caba5f1d9e4a93,
    0x8e938662882af53e,0x547eb47b7282ee9c,
    0xb23867fb2a35b28d,0xe99e619a4f23aa43,
    0xdec681f9f4c31f31,0x6405fa00e2ec94d4,
    0x8b3c113c38f9f37e,0xde83bc408dd3dd04,
    0xae0b158b4738705e,0x9624ab50b148d445,
    0xd98ddaee19068c76,0x3badd624dd9b0957,
    0x87f8a8d4cfa417c9,0xe54ca5d70a80e5d6,
    0xa9f6d30a038d1dbc,0x5e9fcf4ccd211f4c,
    0xd47487cc8470652b,0x7647c3200069671f,
    0x84c8d4dfd2c63f3b,0x29ecd9f40041e073,
    0xa5fb0a17c777cf09,0xf468107100525890,
    0xcf79cc9db955c2cc,0x7182148d4066eeb4,
    0x81ac1fe293d599bf,0xc6f14cd848405530,
    0xa21727db38cb002f,0xb8ada00e5a506a7c,
    0xca9cf1d206fdc03b,0xa6d90811f0e4851c,
    0xfd442e4688bd304a,0x908f4a166d1da663,
    0x9e4a9cec15763e2e,0x9a598e4e043287fe,
    0xc5dd44271ad3cdba,0x40eff1e1853f29fd,
    # 这是一个长列表的十六进制数对，每个数对占用一个整数位置
    0xf7549530e188c128,0xd12bee59e68ef47c,
    0x9a94dd3e8cf578b9,0x82bb74f8301958ce,
    0xc13a148e3032d6e7,0xe36a52363c1faf01,
    0xf18899b1bc3f8ca1,0xdc44e6c3cb279ac1,
    0x96f5600f15a7b7e5,0x29ab103a5ef8c0b9,
    0xbcb2b812db11a5de,0x7415d448f6b6f0e7,
    0xebdf661791d60f56,0x111b495b3464ad21,
    0x936b9fcebb25c995,0xcab10dd900beec34,
    0xb84687c269ef3bfb,0x3d5d514f40eea742,
    0xe65829b3046b0afa,0xcb4a5a3112a5112,
    0x8ff71a0fe2c2e6dc,0x47f0e785eaba72ab,
    0xb3f4e093db73a093,0x59ed216765690f56,
    0xe0f218b8d25088b8,0x306869c13ec3532c,
    0x8c974f7383725573,0x1e414218c73a13fb,
    0xafbd2350644eeacf,0xe5d1929ef90898fa,
    0xdbac6c247d62a583,0xdf45f746b74abf39,
    0x894bc396ce5da772,0x6b8bba8c328eb783,
    0xab9eb47c81f5114f,0x66ea92f3f326564,
    0xd686619ba27255a2,0xc80a537b0efefebd,
    0x8613fd0145877585,0xbd06742ce95f5f36,
    0xa798fc4196e952e7,0x2c48113823b73704,
    0xd17f3b51fca3a7a0,0xf75a15862ca504c5,
    0x82ef85133de648c4,0x9a984d73dbe722fb,
    0xa3ab66580d5fdaf5,0xc13e60d0d2e0ebba,
    0xcc963fee10b7d1b3,0x318df905079926a8,
    0xffbbcfe994e5c61f,0xfdf17746497f7052,
    0x9fd561f1fd0f9bd3,0xfeb6ea8bedefa633,
    0xc7caba6e7c5382c8,0xfe64a52ee96b8fc0,
    0xf9bd690a1b68637b,0x3dfdce7aa3c673b0,
    0x9c1661a651213e2d,0x6bea10ca65c084e,
    0xc31bfa0fe5698db8,0x486e494fcff30a62,
    0xf3e2f893dec3f126,0x5a89dba3c3efccfa,
    0x986ddb5c6b3a76b7,0xf89629465a75e01c,
    0xbe89523386091465,0xf6bbb397f1135823,
    0xee2ba6c0678b597f,0x746aa07ded582e2c,
    0x94db483840b717ef,0xa8c2a44eb4571cdc,
    0xba121a4650e4ddeb,0x92f34d62616ce413,
    0xe896a0d7e51e1566,0x77b020baf9c81d17,
    0x915e2486ef32cd60,0xace1474dc1d122e,
    0xb5b5ada8aaff80b8,0xd819992132456ba,
    0xe3231912d5bf60e6,0x10e1fff697ed6c69,
    0x8df5efabc5979c8f,0xca8d3ffa1ef463c1,
    0xb1736b96b6fd83b3,0xbd308ff8a6b17cb2,
    0xddd0467c64bce4a0,0xac7cb3f6d05ddbde,
    0x8aa22c0dbef60ee4,0x6bcdf07a423aa96b,
    0xad4ab7112eb3929d,0x86c16c98d2c953c6,
    0xd89d64d57a607744,0xe871c7bf077ba8b7,
    0x87625f056c7c4a8b,0x11471cd764ad4972,
    0xa93af6c6c79b5d2d,0xd598e40d3dd89bcf,
    0xd389b47879823479,0x4aff1d108d4ec2c3,
    0x843610cb4bf160cb,0xcedf722a585139ba,
    0xa54394fe1eedb8fe,0xc2974eb4ee658828,
    0xce947a3da6a9273e,0x733d226229feea32,
    0x811ccc668829b887,0x806357d5a3f525f,
    0xa163ff802a3426a8,0xca07c2dcb0cf26f7,
    0xc9bcff6034c13052,0xfc89b393dd02f0b5,
    0xfc2c3f3841f17c67,0xbbac2078d443ace2,
    0x9d9ba7832936edc0,0xd54b944b84aa4c0d,
    0xc5029163f384a931,0xa9e795e65d4df11,
    0xf64335bcf065d37d,0x4d4617b5ff4a16d5,
    0x99ea0196163fa42e,0x504bced1bf8e4e45,
    0xc06481fb9bcf8d39,0xe45ec2862f71e1d6,
    0xf07da27a82c37088,0x5d767327bb4e5a4c,
    0x964e858c91ba2655,0x3a6a07f8d510f86f,
    0xbbe226efb628afea,0x890489f70a55368b,
    0xeadab0aba3b2dbe5,0x2b45ac74ccea842e,
    0x92c8ae6b464fc96f,0x3b0b8bc90012929d,
    0xb77ada0617e3bbcb,0x9ce6ebb40173744,
    0xe55990879ddcaabd,0xcc420a6a101d0515,
    # 下面是一系列十六进制数值，可能是用于某种加密或哈希算法的常量或密钥
    0x8f57fa54c2a9eab6,0x9fa946824a12232d,
    0xb32df8e9f3546564,0x47939822dc96abf9,
    0xdff9772470297ebd,0x59787e2b93bc56f7,
    0x8bfbea76c619ef36,0x57eb4edb3c55b65a,
    0xaefae51477a06b03,0xede622920b6b23f1,
    0xdab99e59958885c4,0xe95fab368e45eced,
    0x88b402f7fd75539b,0x11dbcb0218ebb414,
    0xaae103b5fcd2a881,0xd652bdc29f26a119,
    0xd59944a37c0752a2,0x4be76d3346f0495f,
    0x857fcae62d8493a5,0x6f70a4400c562ddb,
    0xa6dfbd9fb8e5b88e,0xcb4ccd500f6bb952,
    0xd097ad07a71f26b2,0x7e2000a41346a7a7,
    0x825ecc24c873782f,0x8ed400668c0c28c8,
    0xa2f67f2dfa90563b,0x728900802f0f32fa,
    0xcbb41ef979346bca,0x4f2b40a03ad2ffb9,
    0xfea126b7d78186bc,0xe2f610c84987bfa8,
    0x9f24b832e6b0f436,0xdd9ca7d2df4d7c9,
    0xc6ede63fa05d3143,0x91503d1c79720dbb,
    0xf8a95fcf88747d94,0x75a44c6397ce912a,
    0x9b69dbe1b548ce7c,0xc986afbe3ee11aba,
    0xc24452da229b021b,0xfbe85badce996168,
    0xf2d56790ab41c2a2,0xfae27299423fb9c3,
    0x97c560ba6b0919a5,0xdccd879fc967d41a,
    0xbdb6b8e905cb600f,0x5400e987bbc1c920,
    0xed246723473e3813,0x290123e9aab23b68,
    0x9436c0760c86e30b,0xf9a0b6720aaf6521,
    0xb94470938fa89bce,0xf808e40e8d5b3e69,
    0xe7958cb87392c2c2,0xb60b1d1230b20e04,
    0x90bd77f3483bb9b9,0xb1c6f22b5e6f48c2,
    0xb4ecd5f01a4aa828,0x1e38aeb6360b1af3,
    0xe2280b6c20dd5232,0x25c6da63c38de1b0,
    0x8d590723948a535f,0x579c487e5a38ad0e,
    0xb0af48ec79ace837,0x2d835a9df0c6d851,
    0xdcdb1b2798182244,0xf8e431456cf88e65,
    0x8a08f0f8bf0f156b,0x1b8e9ecb641b58ff,
    0xac8b2d36eed2dac5,0xe272467e3d222f3f,
    0xd7adf884aa879177,0x5b0ed81dcc6abb0f,
    0x86ccbb52ea94baea,0x98e947129fc2b4e9,
    0xa87fea27a539e9a5,0x3f2398d747b36224,
    0xd29fe4b18e88640e,0x8eec7f0d19a03aad,
    0x83a3eeeef9153e89,0x1953cf68300424ac,
    0xa48ceaaab75a8e2b,0x5fa8c3423c052dd7,
    0xcdb02555653131b6,0x3792f412cb06794d,
    0x808e17555f3ebf11,0xe2bbd88bbee40bd0,
    0xa0b19d2ab70e6ed6,0x5b6aceaeae9d0ec4,
    0xc8de047564d20a8b,0xf245825a5a445275,
    0xfb158592be068d2e,0xeed6e2f0f0d56712,
    0x9ced737bb6c4183d,0x55464dd69685606b,
    0xc428d05aa4751e4c,0xaa97e14c3c26b886,
    0xf53304714d9265df,0xd53dd99f4b3066a8,
    0x993fe2c6d07b7fab,0xe546a8038efe4029,
    0xbf8fdb78849a5f96,0xde98520472bdd033,
    0xef73d256a5c0f77c,0x963e66858f6d4440,
    0x95a8637627989aad,0xdde7001379a44aa8,
    0xbb127c53b17ec159,0x5560c018580d5d52,
    0xe9d71b689dde71af,0xaab8f01e6e10b4a6,
    0x9226712162ab070d,0xcab3961304ca70e8,
    0xb6b00d69bb55c8d1,0x3d607b97c5fd0d22,
    0xe45c10c42a2b3b05,0x8cb89a7db77c506a,
    0x8eb98a7a9a5b04e3,0x77f3608e92adb242,
    0xb267ed1940f1c61c,0x55f038b237591ed3,
    0xdf01e85f912e37a3,0x6b6c46dec52f6688,
    0x8b61313bbabce2c6,0x2323ac4b3b3da015,
    0xae397d8aa96c1b77,0xabec975e0a0d081a,
    0xd9c7dced53c72255,0x96e7bd358c904a21,
    0x881cea14545c7575,0x7e50d64177da2e54,
    0xaa242499697392d2,0xdde50bd1d5d0b9e9,
    0xd4ad2dbfc3d07787,0x955e4ec64b44e864,
    0x84ec3c97da624ab4,0xbd5af13bef0b113e,
    # 下面是一组十六进制数对，可能用于某种特定的数据结构或算法
    0xa6274bbdd0fadd61,0xecb1ad8aeacdd58e,
    0xcfb11ead453994ba,0x67de18eda5814af2,
    0x81ceb32c4b43fcf4,0x80eacf948770ced7,
    0xa2425ff75e14fc31,0xa1258379a94d028d,
    0xcad2f7f5359a3b3e,0x96ee45813a04330,
    0xfd87b5f28300ca0d,0x8bca9d6e188853fc,
    0x9e74d1b791e07e48,0x775ea264cf55347e,
    0xc612062576589dda,0x95364afe032a819e,
    0xf79687aed3eec551,0x3a83ddbd83f52205,
    0x9abe14cd44753b52,0xc4926a9672793543,
    0xc16d9a0095928a27,0x75b7053c0f178294,
    0xf1c90080baf72cb1,0x5324c68b12dd6339,
    0x971da05074da7bee,0xd3f6fc16ebca5e04,
    0xbce5086492111aea,0x88f4bb1ca6bcf585,
    0xec1e4a7db69561a5,0x2b31e9e3d06c32e6,
    0x9392ee8e921d5d07,0x3aff322e62439fd0,
    0xb877aa3236a4b449,0x9befeb9fad487c3,
    0xe69594bec44de15b,0x4c2ebe687989a9b4,
    0x901d7cf73ab0acd9,0xf9d37014bf60a11,
    0xb424dc35095cd80f,0x538484c19ef38c95,
    0xe12e13424bb40e13,0x2865a5f206b06fba,
    0x8cbccc096f5088cb,0xf93f87b7442e45d4,
    0xafebff0bcb24aafe,0xf78f69a51539d749,
    0xdbe6fecebdedd5be,0xb573440e5a884d1c,
    0x89705f4136b4a597,0x31680a88f8953031,
    0xabcc77118461cefc,0xfdc20d2b36ba7c3e,
    0xd6bf94d5e57a42bc,0x3d32907604691b4d,
    0x8637bd05af6c69b5,0xa63f9a49c2c1b110,
    0xa7c5ac471b478423,0xfcf80dc33721d54,
    0xd1b71758e219652b,0xd3c36113404ea4a9,
    0x83126e978d4fdf3b,0x645a1cac083126ea,
    0xa3d70a3d70a3d70a,0x3d70a3d70a3d70a4,
    0xcccccccccccccccc,0xcccccccccccccccd,
    0x8000000000000000,0x0,
    0xa000000000000000,0x0,
    0xc800000000000000,0x0,
    0xfa00000000000000,0x0,
    0x9c40000000000000,0x0,
    0xc350000000000000,0x0,
    0xf424000000000000,0x0,
    0x9896800000000000,0x0,
    0xbebc200000000000,0x0,
    0xee6b280000000000,0x0,
    0x9502f90000000000,0x0,
    0xba43b74000000000,0x0,
    0xe8d4a51000000000,0x0,
    0x9184e72a00000000,0x0,
    0xb5e620f480000000,0x0,
    0xe35fa931a0000000,0x0,
    0x8e1bc9bf04000000,0x0,
    0xb1a2bc2ec5000000,0x0,
    0xde0b6b3a76400000,0x0,
    0x8ac7230489e80000,0x0,
    0xad78ebc5ac620000,0x0,
    0xd8d726b7177a8000,0x0,
    0x878678326eac9000,0x0,
    0xa968163f0a57b400,0x0,
    0xd3c21bcecceda100,0x0,
    0x84595161401484a0,0x0,
    0xa56fa5b99019a5c8,0x0,
    0xcecb8f27f4200f3a,0x0,
    0x813f3978f8940984,0x4000000000000000,
    0xa18f07d736b90be5,0x5000000000000000,
    0xc9f2c9cd04674ede,0xa400000000000000,
    0xfc6f7c4045812296,0x4d00000000000000,
    0x9dc5ada82b70b59d,0xf020000000000000,
    0xc5371912364ce305,0x6c28000000000000,
    0xf684df56c3e01bc6,0xc732000000000000,
    0x9a130b963a6c115c,0x3c7f400000000000,
    0xc097ce7bc90715b3,0x4b9f100000000000,
    0xf0bdc21abb48db20,0x1e86d40000000000,
    0x96769950b50d88f4,0x1314448000000000,
    0xbc143fa4e250eb31,0x17d955a000000000,
    0xeb194f8e1ae525fd,0x5dcfab0800000000,
    0x92efd1b8d0cf37be,0x5aa1cae500000000,
    0xb7abc627050305ad,0xf14a3d9e40000000,
    0xe596b7b0c643c719,0x6d9ccd05d0000000,
    0x8f7e32ce7bea5c6f,0xe4820023a2000000,
    0xb35dbf821ae4f38b,0xdda2802c8a800000,
    # 下面是一系列十六进制数对，可能是某种数据集或配置的一部分
    0xe0352f62a19e306e,0xd50b2037ad200000,
    0x8c213d9da502de45,0x4526f422cc340000,
    0xaf298d050e4395d6,0x9670b12b7f410000,
    0xdaf3f04651d47b4c,0x3c0cdd765f114000,
    0x88d8762bf324cd0f,0xa5880a69fb6ac800,
    0xab0e93b6efee0053,0x8eea0d047a457a00,
    0xd5d238a4abe98068,0x72a4904598d6d880,
    0x85a36366eb71f041,0x47a6da2b7f864750,
    0xa70c3c40a64e6c51,0x999090b65f67d924,
    0xd0cf4b50cfe20765,0xfff4b4e3f741cf6d,
    0x82818f1281ed449f,0xbff8f10e7a8921a4,
    0xa321f2d7226895c7,0xaff72d52192b6a0d,
    0xcbea6f8ceb02bb39,0x9bf4f8a69f764490,
    0xfee50b7025c36a08,0x2f236d04753d5b4,
    0x9f4f2726179a2245,0x1d762422c946590,
    0xc722f0ef9d80aad6,0x424d3ad2b7b97ef5,
    0xf8ebad2b84e0d58b,0xd2e0898765a7deb2,
    0x9b934c3b330c8577,0x63cc55f49f88eb2f,
    0xc2781f49ffcfa6d5,0x3cbf6b71c76b25fb,
    0xf316271c7fc3908a,0x8bef464e3945ef7a,
    0x97edd871cfda3a56,0x97758bf0e3cbb5ac,
    0xbde94e8e43d0c8ec,0x3d52eeed1cbea317,
    0xed63a231d4c4fb27,0x4ca7aaa863ee4bdd,
    0x945e455f24fb1cf8,0x8fe8caa93e74ef6a,
    0xb975d6b6ee39e436,0xb3e2fd538e122b44,
    0xe7d34c64a9c85d44,0x60dbbca87196b616,
    0x90e40fbeea1d3a4a,0xbc8955e946fe31cd,
    0xb51d13aea4a488dd,0x6babab6398bdbe41,
    0xe264589a4dcdab14,0xc696963c7eed2dd1,
    0x8d7eb76070a08aec,0xfc1e1de5cf543ca2,
    0xb0de65388cc8ada8,0x3b25a55f43294bcb,
    0xdd15fe86affad912,0x49ef0eb713f39ebe,
    0x8a2dbf142dfcc7ab,0x6e3569326c784337,
    0xacb92ed9397bf996,0x49c2c37f07965404,
    0xd7e77a8f87daf7fb,0xdc33745ec97be906,
    0x86f0ac99b4e8dafd,0x69a028bb3ded71a3,
    0xa8acd7c0222311bc,0xc40832ea0d68ce0c,
    0xd2d80db02aabd62b,0xf50a3fa490c30190,
    0x83c7088e1aab65db,0x792667c6da79e0fa,
    0xa4b8cab1a1563f52,0x577001b891185938,
    0xcde6fd5e09abcf26,0xed4c0226b55e6f86,
    0x80b05e5ac60b6178,0x544f8158315b05b4,
    0xa0dc75f1778e39d6,0x696361ae3db1c721,
    0xc913936dd571c84c,0x3bc3a19cd1e38e9,
    0xfb5878494ace3a5f,0x4ab48a04065c723,
    0x9d174b2dcec0e47b,0x62eb0d64283f9c76,
    0xc45d1df942711d9a,0x3ba5d0bd324f8394,
    0xf5746577930d6500,0xca8f44ec7ee36479,
    0x9968bf6abbe85f20,0x7e998b13cf4e1ecb,
    0xbfc2ef456ae276e8,0x9e3fedd8c321a67e,
    0xefb3ab16c59b14a2,0xc5cfe94ef3ea101e,
    0x95d04aee3b80ece5,0xbba1f1d158724a12,
    0xbb445da9ca61281f,0x2a8a6e45ae8edc97,
    0xea1575143cf97226,0xf52d09d71a3293bd,
    0x924d692ca61be758,0x593c2626705f9c56,
    0xb6e0c377cfa2e12e,0x6f8b2fb00c77836c,
    0xe498f455c38b997a,0xb6dfb9c0f956447,
    0x8edf98b59a373fec,0x4724bd4189bd5eac,
    0xb2977ee300c50fe7,0x58edec91ec2cb657,
    0xdf3d5e9bc0f653e1,0x2f2967b66737e3ed,
    0x8b865b215899f46c,0xbd79e0d20082ee74,
    0xae67f1e9aec07187,0xecd8590680a3aa11,
    0xda01ee641a708de9,0xe80e6f4820cc9495,
    0x884134fe908658b2,0x3109058d147fdcdd,
    0xaa51823e34a7eede,0xbd4b46f0599fd415,
    0xd4e5e2cdc1d1ea96,0x6c9e18ac7007c91a,
    0x850fadc09923329e,0x3e2cf6bc604ddb0,
    0xa6539930bf6bff45,0x84db8346b786151c,
    0xcfe87f7cef46ff16,0xe612641865679a63,
    0x81f14fae158c5f6e,0x4fcb7e8f3f60c07e,
    0xa26da3999aef7749,0xe3be5e330f38f09d,
    0xcb090c8001ab551c,0x5cadf5bfd3072cc5,
    0xfdcb4fa002162a63,0x73d9732fc7c8f7f6,
    0x9e9f11c4014dda7e,0x2867e7fddcdd9afa,
    0xc646d63501a1511d,0xb281e1fd541501b8,
    0xf7d88bc24209a565,0x1f225a7ca91a4226,
    0x9ae757596946075f,0x3375788de9b06958,
    0xc1a12d2fc3978937,0x52d6b1641c83ae,
    0xf209787bb47d6b84,0xc0678c5dbd23a49a,
    0x9745eb4d50ce6332,0xf840b7ba963646e0,
    0xbd176620a501fbff,0xb650e5a93bc3d898,
    0xec5d3fa8ce427aff,0xa3e51f138ab4cebe,
    0x93ba47c980e98cdf,0xc66f336c36b10137,
    0xb8a8d9bbe123f017,0xb80b0047445d4184,
    0xe6d3102ad96cec1d,0xa60dc059157491e5,
    0x9043ea1ac7e41392,0x87c89837ad68db2f,
    0xb454e4a179dd1877,0x29babe4598c311fb,
    0xe16a1dc9d8545e94,0xf4296dd6fef3d67a,
    0x8ce2529e2734bb1d,0x1899e4a65f58660c,
    0xb01ae745b101e9e4,0x5ec05dcff72e7f8f,
    0xdc21a1171d42645d,0x76707543f4fa1f73,
    0x899504ae72497eba,0x6a06494a791c53a8,
    0xabfa45da0edbde69,0x487db9d17636892,
    0xd6f8d7509292d603,0x45a9d2845d3c42b6,
    0x865b86925b9bc5c2,0xb8a2392ba45a9b2,
    0xa7f26836f282b732,0x8e6cac7768d7141e,
    0xd1ef0244af2364ff,0x3207d795430cd926,
    0x8335616aed761f1f,0x7f44e6bd49e807b8,
    0xa402b9c5a8d3a6e7,0x5f16206c9c6209a6,
    0xcd036837130890a1,0x36dba887c37a8c0f,
    0x802221226be55a64,0xc2494954da2c9789,
    0xa02aa96b06deb0fd,0xf2db9baa10b7bd6c,
    0xc83553c5c8965d3d,0x6f92829494e5acc7,
    0xfa42a8b73abbf48c,0xcb772339ba1f17f9,
    0x9c69a97284b578d7,0xff2a760414536efb,
    0xc38413cf25e2d70d,0xfef5138519684aba,
    0xf46518c2ef5b8cd1,0x7eb258665fc25d69,
    0x98bf2f79d5993802,0xef2f773ffbd97a61,
    0xbeeefb584aff8603,0xaafb550ffacfd8fa,
    0xeeaaba2e5dbf6784,0x95ba2a53f983cf38,
    0x952ab45cfa97a0b2,0xdd945a747bf26183,
    0xba756174393d88df,0x94f971119aeef9e4,
    0xe912b9d1478ceb17,0x7a37cd5601aab85d,
    0x91abb422ccb812ee,0xac62e055c10ab33a,
    0xb616a12b7fe617aa,0x577b986b314d6009,
    0xe39c49765fdf9d94,0xed5a7e85fda0b80b,
    0x8e41ade9fbebc27d,0x14588f13be847307,
    0xb1d219647ae6b31c,0x596eb2d8ae258fc8,
    0xde469fbd99a05fe3,0x6fca5f8ed9aef3bb,
    0x8aec23d680043bee,0x25de7bb9480d5854,
    0xada72ccc20054ae9,0xaf561aa79a10ae6a,
    0xd910f7ff28069da4,0x1b2ba1518094da04,
    0x87aa9aff79042286,0x90fb44d2f05d0842,
    0xa99541bf57452b28,0x353a1607ac744a53,
    0xd3fa922f2d1675f2,0x42889b8997915ce8,
    0x847c9b5d7c2e09b7,0x69956135febada11,
    0xa59bc234db398c25,0x43fab9837e699095,
    0xcf02b2c21207ef2e,0x94f967e45e03f4bb,
    0x8161afb94b44f57d,0x1d1be0eebac278f5,
    0xa1ba1ba79e1632dc,0x6462d92a69731732,
    0xca28a291859bbf93,0x7d7b8f7503cfdcfe,
    0xfcb2cb35e702af78,0x5cda735244c3d43e,
    0x9defbf01b061adab,0x3a0888136afa64a7,
    0xc56baec21c7a1916,0x88aaa1845b8fdd0,
    0xf6c69a72a3989f5b,0x8aad549e57273d45,
    0x9a3c2087a63f6399,0x36ac54e2f678864b,
    0xc0cb28a98fcf3c7f,0x84576a1bb416a7dd,
    0xf0fdf2d3f3c30b9f,0x656d44a2a11c51d5,
    # 下面的代码看起来像是一系列十六进制数对，可能是用作某种数据的参考值或者密钥。
    0x969eb7c47859e743,0x9f644ae5a4b1b325,
    0xbc4665b596706114,0x873d5d9f0dde1fee,
    0xeb57ff22fc0c7959,0xa90cb506d155a7ea,
    0x9316ff75dd87cbd8,0x9a7f12442d588f2,
    0xb7dcbf5354e9bece,0xc11ed6d538aeb2f,
    0xe5d3ef282a242e81,0x8f1668c8a86da5fa,
    0x8fa475791a569d10,0xf96e017d694487bc,
    0xb38d92d760ec4455,0x37c981dcc395a9ac,
    0xe070f78d3927556a,0x85bbe253f47b1417,
    0x8c469ab843b89562,0x93956d7478ccec8e,
    0xaf58416654a6babb,0x387ac8d1970027b2,
    0xdb2e51bfe9d0696a,0x6997b05fcc0319e,
    0x88fcf317f22241e2,0x441fece3bdf81f03,
    0xab3c2fddeeaad25a,0xd527e81cad7626c3,
    0xd60b3bd56a5586f1,0x8a71e223d8d3b074,
    0x85c7056562757456,0xf6872d5667844e49,
    0xa738c6bebb12d16c,0xb428f8ac016561db,
    0xd106f86e69d785c7,0xe13336d701beba52,
    0x82a45b450226b39c,0xecc0024661173473,
    0xa34d721642b06084,0x27f002d7f95d0190,
    0xcc20ce9bd35c78a5,0x31ec038df7b441f4,
    0xff290242c83396ce,0x7e67047175a15271,
    0x9f79a169bd203e41,0xf0062c6e984d386,
    0xc75809c42c684dd1,0x52c07b78a3e60868,
    0xf92e0c3537826145,0xa7709a56ccdf8a82,
    0x9bbcc7a142b17ccb,0x88a66076400bb691,
    0xc2abf989935ddbfe,0x6acff893d00ea435,
    0xf356f7ebf83552fe,0x583f6b8c4124d43,
    0x98165af37b2153de,0xc3727a337a8b704a,
    0xbe1bf1b059e9a8d6,0x744f18c0592e4c5c,
    0xeda2ee1c7064130c,0x1162def06f79df73,
    0x9485d4d1c63e8be7,0x8addcb5645ac2ba8,
    0xb9a74a0637ce2ee1,0x6d953e2bd7173692,
    0xe8111c87c5c1ba99,0xc8fa8db6ccdd0437,
    0x910ab1d4db9914a0,0x1d9c9892400a22a2,
    0xb54d5e4a127f59c8,0x2503beb6d00cab4b,
    0xe2a0b5dc971f303a,0x2e44ae64840fd61d,
    0x8da471a9de737e24,0x5ceaecfed289e5d2,
    0xb10d8e1456105dad,0x7425a83e872c5f47,
    0xdd50f1996b947518,0xd12f124e28f77719,
    0x8a5296ffe33cc92f,0x82bd6b70d99aaa6f,
    0xace73cbfdc0bfb7b,0x636cc64d1001550b,
    0xd8210befd30efa5a,0x3c47f7e05401aa4e,
    0x8714a775e3e95c78,0x65acfaec34810a71,
    0xa8d9d1535ce3b396,0x7f1839a741a14d0d,
    0xd31045a8341ca07c,0x1ede48111209a050,
    0x83ea2b892091e44d,0x934aed0aab460432,
    0xa4e4b66b68b65d60,0xf81da84d5617853f,
    0xce1de40642e3f4b9,0x36251260ab9d668e,
    0x80d2ae83e9ce78f3,0xc1d72b7c6b426019,
    0xa1075a24e4421730,0xb24cf65b8612f81f,
    0xc94930ae1d529cfc,0xdee033f26797b627,
    0xfb9b7cd9a4a7443c,0x169840ef017da3b1,
    0x9d412e0806e88aa5,0x8e1f289560ee864e,
    0xc491798a08a2ad4e,0xf1a6f2bab92a27e2,
    0xf5b5d7ec8acb58a2,0xae10af696774b1db,
    0x9991a6f3d6bf1765,0xacca6da1e0a8ef29,
    0xbff610b0cc6edd3f,0x17fd090a58d32af3,
    0xeff394dcff8a948e,0xddfc4b4cef07f5b0,
    0x95f83d0a1fb69cd9,0x4abdaf101564f98e,
    0xbb764c4ca7a4440f,0x9d6d1ad41abe37f1,
    0xea53df5fd18d5513,0x84c86189216dc5ed,
    0x92746b9be2f8552c,0x32fd3cf5b4e49bb4,
    0xb7118682dbb66a77,0x3fbc8c33221dc2a1,
    0xe4d5e82392a40515,0xfabaf3feaa5334a,
    0x8f05b1163ba6832d,0x29cb4d87f2a7400e,
    0xb2c71d5bca9023f8,0x743e20e9ef511012,
    0xdf78e4b2bd342cf6,0x914da9246b255416,
    0x8bab8eefb6409c1a,0x1ad089b6c2f7548e,
    # 下面是一系列十六进制数，可能代表一个数组或者其他数据结构，但由于缺乏上下文，无法确切知道其用途和含义。
    0xae9672aba3d0c320,0xa184ac2473b529b1,
    0xda3c0f568cc4f3e8,0xc9e5d72d90a2741e,
    0x8865899617fb1871,0x7e2fa67c7a658892,
    0xaa7eebfb9df9de8d,0xddbb901b98feeab7,
    0xd51ea6fa85785631,0x552a74227f3ea565,
    0x8533285c936b35de,0xd53a88958f87275f,
    0xa67ff273b8460356,0x8a892abaf368f137,
    0xd01fef10a657842c,0x2d2b7569b0432d85,
    0x8213f56a67f6b29b,0x9c3b29620e29fc73,
    0xa298f2c501f45f42,0x8349f3ba91b47b8f,
    0xcb3f2f7642717713,0x241c70a936219a73,
    0xfe0efb53d30dd4d7,0xed238cd383aa0110,
    0x9ec95d1463e8a506,0xf4363804324a40aa,
    0xc67bb4597ce2ce48,0xb143c6053edcd0d5,
    0xf81aa16fdc1b81da,0xdd94b7868e94050a,
    0x9b10a4e5e9913128,0xca7cf2b4191c8326,
    0xc1d4ce1f63f57d72,0xfd1c2f611f63a3f0,
    0xf24a01a73cf2dccf,0xbc633b39673c8cec,
    0x976e41088617ca01,0xd5be0503e085d813,
    0xbd49d14aa79dbc82,0x4b2d8644d8a74e18,
    0xec9c459d51852ba2,0xddf8e7d60ed1219e,
    0x93e1ab8252f33b45,0xcabb90e5c942b503,
    0xb8da1662e7b00a17,0x3d6a751f3b936243,
    0xe7109bfba19c0c9d,0xcc512670a783ad4,
    0x906a617d450187e2,0x27fb2b80668b24c5,
    0xb484f9dc9641e9da,0xb1f9f660802dedf6,
    0xe1a63853bbd26451,0x5e7873f8a0396973,
    0x8d07e33455637eb2,0xdb0b487b6423e1e8,
    0xb049dc016abc5e5f,0x91ce1a9a3d2cda62,
    0xdc5c5301c56b75f7,0x7641a140cc7810fb,
    0x89b9b3e11b6329ba,0xa9e904c87fcb0a9d,
    0xac2820d9623bf429,0x546345fa9fbdcd44,
    0xd732290fbacaf133,0xa97c177947ad4095,
    0x867f59a9d4bed6c0,0x49ed8eabcccc485d,
    0xa81f301449ee8c70,0x5c68f256bfff5a74,
    0xd226fc195c6a2f8c,0x73832eec6fff3111,
    0x83585d8fd9c25db7,0xc831fd53c5ff7eab,
    0xa42e74f3d032f525,0xba3e7ca8b77f5e55,
    0xcd3a1230c43fb26f,0x28ce1bd2e55f35eb,
    0x80444b5e7aa7cf85,0x7980d163cf5b81b3,
    0xa0555e361951c366,0xd7e105bcc332621f,
    0xc86ab5c39fa63440,0x8dd9472bf3fefaa7,
    0xfa856334878fc150,0xb14f98f6f0feb951,
    0x9c935e00d4b9d8d2,0x6ed1bf9a569f33d3,
    0xc3b8358109e84f07,0xa862f80ec4700c8,
    0xf4a642e14c6262c8,0xcd27bb612758c0fa,
    0x98e7e9cccfbd7dbd,0x8038d51cb897789c,
    0xbf21e44003acdd2c,0xe0470a63e6bd56c3,
    0xeeea5d5004981478,0x1858ccfce06cac74,
    0x95527a5202df0ccb,0xf37801e0c43ebc8,
    0xbaa718e68396cffd,0xd30560258f54e6ba,
    0xe950df20247c83fd,0x47c6b82ef32a2069,
    0x91d28b7416cdd27e,0x4cdc331d57fa5441,
    0xb6472e511c81471d,0xe0133fe4adf8e952,
    0xe3d8f9e563a198e5,0x58180fddd97723a6,
    0x8e679c2f5e44ff8f,0x570f09eaa7ea7648,
// 结束命名空间 fast_float
} // namespace fast_float

// 如果 FASTFLOAT_DECIMAL_TO_BINARY_H 未定义，则定义 FASTFLOAT_DECIMAL_TO_BINARY_H
#ifndef FASTFLOAT_DECIMAL_TO_BINARY_H
#define FASTFLOAT_DECIMAL_TO_BINARY_H

// 包含 C 标准库头文件
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// 进入命名空间 fast_float
namespace fast_float {

// 计算或者近似计算 w * 5**q 的结果，并返回一个近似结果的 128 位值，高位对应最高有效位，低位对应最低有效位。
template <int bit_precision>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
value128 compute_product_approximation(int64_t q, uint64_t w) {
  // 根据 q 计算对应的 powers 数组的索引
  const int index = 2 * int(q - powers::smallest_power_of_five);
  // 对于较小的 q 值，例如 q 在 [0,27] 范围内，由于 full_multiplication(w, power_of_five_128[index]) 可以给出精确的结果。
  value128 firstproduct = full_multiplication(w, powers::power_of_five_128[index]);
  
  // 确保 bit_precision 在有效范围内，即 (0,64]
  static_assert((bit_precision >= 0) && (bit_precision <= 64), " precision should  be in (0,64]");
  constexpr uint64_t precision_mask = (bit_precision < 64) ?
               (uint64_t(0xFFFFFFFFFFFFFFFF) >> bit_precision)
               : uint64_t(0xFFFFFFFFFFFFFFFF);
  
  // 如果满足精度要求，考虑进行第二个乘积的计算
  if((firstproduct.high & precision_mask) == precision_mask) { // 可能还需进一步判断 (lower + w < lower)
    // 计算第二个乘积
    value128 secondproduct = full_multiplication(w, powers::power_of_five_128[index + 1]);
    // 更新 firstproduct 的低位
    firstproduct.low += secondproduct.high;
    // 如果第二个乘积的高位大于 firstproduct 的低位，则进位
    if(secondproduct.high > firstproduct.low) {
      firstproduct.high++;
    }
  }
  
  // 返回计算结果
  return firstproduct;
}

// 进入命名空间 detail
namespace detail {
/**
 * 对于 q 在 (0,350) 范围内，
 *  f = (((152170 + 65536) * q ) >> 16);
 * 等价于 floor(p) + q，其中 p = log(5**q)/log(2) = q * log(5)/log(2)
 *
 * 对于 q 在 (-400,0) 范围内的负值，
 *  f = (((152170 + 65536) * q ) >> 16);
 * 等价于 -ceil(p) + q，其中 p = log(5**-q)/log(2) = -q * log(5)/log(2)
 */
  constexpr fastfloat_really_inline int32_t power(int32_t q)  noexcept  {
    return (((152170 + 65536) * q) >> 16) + 63;
  }
} // namespace detail

// 创建一个根据无效的 power2 调整的尾数，用于已经乘以 10 ** q 的有效位数。
template <typename binary>
fastfloat_really_inline FASTFLOAT_CONSTEXPR14
adjusted_mantissa compute_error_scaled(int64_t q, uint64_t w, int lz) noexcept  {
  // 计算 w 的最高位的值
  int hilz = int(w >> 63) ^ 1;
  adjusted_mantissa answer;
  // 将 w 的值左移 hilz 位，并存入 answer 的尾数字段
  answer.mantissa = w << hilz;
  // 计算偏置值，用于 binary 类型的尾数显式位数与最小指数的偏差
  int bias = binary::mantissa_explicit_bits() - binary::minimum_exponent();
  // 计算 power2 的值，考虑了 detail::power 的返回值，以及 hilz、lz 和无效的 am_bias 的偏差
  answer.power2 = int32_t(detail::power(int32_t(q)) + bias - hilz - lz - 62 + invalid_am_bias);
  // 返回调整后的结果
  return answer;
}
// 计算浮点数的误差，并根据情况调整指数
template <typename binary>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
adjusted_mantissa compute_error(int64_t q, uint64_t w)  noexcept  {
  // 计算 w * 10 ** q 的结果，不进行舍入
  int lz = leading_zeroes(w); // 计算 w 的前导零位数
  w <<= lz; // 将 w 左移，使得最高有效位为1
  value128 product = compute_product_approximation<binary::mantissa_explicit_bits() + 3>(q, w);
  // 根据计算结果进一步计算误差并按比例调整
  return compute_error_scaled<binary>(q, product.high, lz);
}

// 计算浮点数的值 w * 10 ** q
// 返回的值应该是一个有效的 ieee64 数字，只需打包即可
// 然而，在一些非常罕见的情况下，计算可能失败。在这种情况下，返回一个带有负的2的幂的 adjusted_mantissa：
// 调用者应在这种情况下重新计算。
template <typename binary>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
adjusted_mantissa compute_float(int64_t q, uint64_t w)  noexcept  {
  adjusted_mantissa answer;
  if ((w == 0) || (q < binary::smallest_power_of_ten())) {
    answer.power2 = 0;
    answer.mantissa = 0;
    // 结果应该为零
    return answer;
  }
  if (q > binary::largest_power_of_ten()) {
    // 我们想要得到无穷大：
    answer.power2 = binary::infinite_power();
    answer.mantissa = 0;
    return answer;
  }
  // 此时 q 处于 [powers::smallest_power_of_five, powers::largest_power_of_five] 范围内。

  // 我们希望 i 的最高有效位为1。如果需要，进行位移。
  int lz = leading_zeroes(w);
  w <<= lz;

  // 由于以下三个原因，所需的精度为 binary::mantissa_explicit_bits() + 3：
  // 1. 我们需要隐式位
  // 2. 我们需要额外的位进行舍入
  // 3. 由于 "upperbit" 算法（结果太小，需要位移），我们可能会失去一个位

  value128 product = compute_product_approximation<binary::mantissa_explicit_bits() + 3>(q, w);
  // 计算得到的 'product' 总是足够的。
  // 数学证明：
  // Noble Mushtak 和 Daniel Lemire, Fast Number Parsing Without Fallback (to appear)
  // 参见 script/mushtak_lemire.py

  // 如果 "compute_product_approximation" 函数稍慢于无分支方法：
  // value128 product = compute_product(q, w);
  // 但实际上，如果能够轻松预测其额外分支，通过 compute_product_approximation 可以大幅提升性能。数据具体情况决定哪个更好。
  int upperbit = int(product.high >> 63);

  // 将结果右移以得到有效的 mantissa
  answer.mantissa = product.high >> (upperbit + 64 - binary::mantissa_explicit_bits() - 3);

  // 计算最终的指数值
  answer.power2 = int32_t(detail::power(int32_t(q)) + upperbit - lz - binary::minimum_exponent());
  if (answer.power2 <= 0) { // 如果是一个次标准数？
    // 在这里，有 answer.power2 <= 0，所以 -answer.power2 >= 0
    if(-answer.power2 + 1 >= 64) { // 如果最小指数以下有超过64位，结果肯定为零。
      answer.power2 = 0;
      answer.mantissa = 0;
      // 结果应该为零
      return answer;
    }
    // 下一行是安全的，因为 -answer.power2 + 1 < 64
    // 将 answer.mantissa 向右移动，移动位数为 -answer.power2 + 1
    answer.mantissa >>= -answer.power2 + 1;

    // 如果 "round-to-even" 和 subnormals 同时存在是不可能的，因为
    // "round-to-even" 只在接近0的幂次时发生。
    answer.mantissa += (answer.mantissa & 1); // 向上舍入

    // 将 answer.mantissa 向右移动一位
    answer.mantissa >>= 1;

    // 存在一种奇怪的情况，我们没有 subnormal，仅仅是
    // 假设我们从 2.2250738585072013e-308 开始，最终得到的结果
    // 是 0x3fffffffffffff x 2^-1023-53，这在技术上是 subnormal 的，
    // 而 0x40000000000000 x 2^-1023-53 是正常数。现在，我们需要舍入
    // 0x3fffffffffffff x 2^-1023-53，并且一旦舍入完成，我们不再是
    // subnormal，但是我们只能在舍入后才能知道这一点。
    // 因此，只有在小于阈值时我们才声明为 subnormal。
    answer.power2 = (answer.mantissa < (uint64_t(1) << binary::mantissa_explicit_bits())) ? 0 : 1;

    return answer;
  }

  // 通常情况下，我们向上舍入，但是如果恰好处于中间位置并且基数是偶数，
  // 我们需要向下舍入。
  // 我们只关注 5**q 可以放入单个64位字中的情况。
  if ((product.low <= 1) && (q >= binary::min_exponent_round_to_even()) && (q <= binary::max_exponent_round_to_even()) &&
      ((answer.mantissa & 3) == 1)) { // 可能处于两个浮点数之间！
    // 要处于两个浮点数之间，我们需要在执行
    //   answer.mantissa = product.high >> (upperbit + 64 - binary::mantissa_explicit_bits() - 3);
    // ...时，丢失的只能是零。但如果发生了这种情况，那么我们可以回退！！！
    if ((answer.mantissa << (upperbit + 64 - binary::mantissa_explicit_bits() - 3)) == product.high) {
      answer.mantissa &= ~uint64_t(1);          // 翻转它，以便我们不向上舍入
    }
  }

  // 向上舍入
  answer.mantissa += (answer.mantissa & 1);
  // 将 answer.mantissa 向右移动一位
  answer.mantissa >>= 1;

  // 如果 answer.mantissa 大于或等于 (uint64_t(2) << binary::mantissa_explicit_bits())，则进行处理
  // 将 answer.mantissa 设置为 (uint64_t(1) << binary::mantissa_explicit_bits())
  // 并撤销之前的加法
  if (answer.mantissa >= (uint64_t(2) << binary::mantissa_explicit_bits())) {
    answer.mantissa = (uint64_t(1) << binary::mantissa_explicit_bits());
    answer.power2++; // 撤销之前的加法
  }

  // 清除 answer.mantissa 中的特定位
  answer.mantissa &= ~(uint64_t(1) << binary::mantissa_explicit_bits());

  // 如果 answer.power2 大于或等于 binary::infinite_power()，则表示无穷大
  // 设置 answer.power2 和 answer.mantissa
  if (answer.power2 >= binary::infinite_power()) {
    answer.power2 = binary::infinite_power();
    answer.mantissa = 0;
  }

  return answer;
// 关闭 FASTFLOAT 命名空间
}

// 结束命名空间 fast_float

} // namespace fast_float

// 结束条件编译指令，防止头文件被重复包含
#endif

#ifndef FASTFLOAT_BIGINT_H
#define FASTFLOAT_BIGINT_H

// 包含必要的标准库头文件
#include <algorithm>    // 包含算法库，用于提供各种算法操作
#include <cstdint>      // 包含整数类型定义，如 uint64_t, uint32_t 等
#include <climits>      // 包含常量定义，如 CHAR_BIT（每字节位数）
#include <cstring>      // 包含字符串处理函数，如 memcpy 等

// FASTFLOAT 命名空间开始
namespace fast_float {

// limb 宽度定义：选择合适的 limb 类型和位数
#if defined(FASTFLOAT_64BIT) && !defined(__sparc)
#define FASTFLOAT_64BIT_LIMB 1
typedef uint64_t limb;     // 定义 limb 类型为 uint64_t
constexpr size_t limb_bits = 64;    // limb 的位数为 64
#else
#define FASTFLOAT_32BIT_LIMB     // 定义 32 位 limb
typedef uint32_t limb;     // 定义 limb 类型为 uint32_t
constexpr size_t limb_bits = 32;    // limb 的位数为 32
#endif

typedef span<limb> limb_span;    // 定义 limb_span 类型

// bigint 的位数定义：至少能容纳最大的 bigint 的位数
constexpr size_t bigint_bits = 4000;    // bigint 的位数为 4000
constexpr size_t bigint_limbs = bigint_bits / limb_bits;   // bigint 的 limb 数量

// stackvec 结构体定义：类似于向量类型，但在栈上分配
template <uint16_t size>
struct stackvec {
  limb data[size];    // 存储 limb 的数组
  uint16_t length{0};    // 当前存储的 limb 数量，默认为 0

  stackvec() = default;   // 默认构造函数
  stackvec(const stackvec &) = delete;   // 禁用拷贝构造函数
  stackvec &operator=(const stackvec &) = delete;   // 禁用赋值运算符
  stackvec(stackvec &&) = delete;    // 禁用移动构造函数
  stackvec &operator=(stackvec &&other) = delete;   // 禁用移动赋值运算符

  // 从现有 limb_span 创建 stackvec
  FASTFLOAT_CONSTEXPR20 stackvec(limb_span s) {
    FASTFLOAT_ASSERT(try_extend(s));    // 扩展 stackvec，确保长度足够
  }

  // 下标运算符重载，用于访问指定位置的 limb
  FASTFLOAT_CONSTEXPR14 limb& operator[](size_t index) noexcept {
    FASTFLOAT_DEBUG_ASSERT(index < length);   // 断言确保下标在有效范围内
    return data[index];
  }
  FASTFLOAT_CONSTEXPR14 const limb& operator[](size_t index) const noexcept {
    FASTFLOAT_DEBUG_ASSERT(index < length);   // 断言确保下标在有效范围内
    return data[index];
  }
  
  // 从末尾索引的运算符重载
  FASTFLOAT_CONSTEXPR14 const limb& rindex(size_t index) const noexcept {
    FASTFLOAT_DEBUG_ASSERT(index < length);   // 断言确保下标在有效范围内
    size_t rindex = length - index - 1;
    return data[rindex];
  }

  // 设置长度，无需边界检查
  FASTFLOAT_CONSTEXPR14 void set_len(size_t len) noexcept {
    length = uint16_t(len);
  }
  constexpr size_t len() const noexcept {
    return length;
  }
  constexpr bool is_empty() const noexcept {
    return length == 0;
  }
  constexpr size_t capacity() const noexcept {
    return size;
  }

  // 追加元素到向量，无需边界检查
  FASTFLOAT_CONSTEXPR14 void push_unchecked(limb value) noexcept {
    data[length] = value;
    length++;
  }
  
  // 尝试追加元素到向量，返回是否添加成功
  FASTFLOAT_CONSTEXPR14 bool try_push(limb value) noexcept {
    if (len() < capacity()) {
      push_unchecked(value);
      return true;
    }
    // 如果容量不足，返回失败
    return false;
  }
  // 如果条件为真，则返回 false；否则返回 false。
  } else {
    return false;
  }
}

// 从 span 中添加项目到向量中，不进行边界检查
FASTFLOAT_CONSTEXPR20 void extend_unchecked(limb_span s) noexcept {
  // 将 span 的内容复制到向量的末尾
  limb* ptr = data + length;
  std::copy_n(s.ptr, s.len(), ptr);
  // 更新向量的长度
  set_len(len() + s.len());
}

// 尝试从 span 中添加项目到向量中，并返回是否添加成功
FASTFLOAT_CONSTEXPR20 bool try_extend(limb_span s) noexcept {
  // 如果向量的当前长度加上 span 的长度不超过容量，则执行添加并返回 true；否则返回 false
  if (len() + s.len() <= capacity()) {
    extend_unchecked(s);
    return true;
  } else {
    return false;
  }
}

// 调整向量的大小，不进行边界检查
// 如果新的大小大于当前向量的长度，则为每个添加的项赋值
FASTFLOAT_CONSTEXPR20
void resize_unchecked(size_t new_len, limb value) noexcept {
  // 如果新的长度大于当前长度，则为新增的项赋予指定的值
  if (new_len > len()) {
    size_t count = new_len - len();
    limb* first = data + len();
    limb* last = first + count;
    ::std::fill(first, last, value);
    // 更新向量的长度
    set_len(new_len);
  } else {
    // 直接更新向量的长度
    set_len(new_len);
  }
}

// 尝试调整向量的大小，并返回是否成功调整了向量的大小
FASTFLOAT_CONSTEXPR20 bool try_resize(size_t new_len, limb value) noexcept {
  // 如果新的长度大于容量，则返回 false；否则调用 resize_unchecked 调整向量的大小，并返回 true
  if (new_len > capacity()) {
    return false;
  } else {
    resize_unchecked(new_len, value);
    return true;
  }
}

// 检查给定索引之后是否存在非零的 limb
// 由于索引是相对于最高有效 limb 的，因此需倒序进行检查
FASTFLOAT_CONSTEXPR14 bool nonzero(size_t index) const noexcept {
  // 从指定索引开始向向量的最高有效 limb 扫描，如果发现非零 limb，则返回 true；否则返回 false
  while (index < len()) {
    if (rindex(index) != 0) {
      return true;
    }
    index++;
  }
  return false;
}

// 规范化大整数，删除最高位为零的 limb
FASTFLOAT_CONSTEXPR14 void normalize() noexcept {
  // 从最低位开始，删除最高位为零的 limb，直到遇到第一个非零 limb 或者向量为空
  while (len() > 0 && rindex(0) == 0) {
    length--;
  }
}
};

// 结束一个代码块，不包含任何参数或返回值。

fastfloat_really_inline FASTFLOAT_CONSTEXPR14
// 使用 fastfloat_really_inline 宏定义一个在编译时展开的内联函数，返回值类型为 uint64_t，C++14 标准兼容性。
uint64_t empty_hi64(bool& truncated) noexcept {
  truncated = false;
  return 0;
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
// 使用 fastfloat_really_inline 宏定义一个在编译时展开的内联函数，返回值类型为 uint64_t，C++20 标准兼容性。
uint64_t uint64_hi64(uint64_t r0, bool& truncated) noexcept {
  truncated = false;
  int shl = leading_zeroes(r0);
  return r0 << shl;
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
// 使用 fastfloat_really_inline 宏定义一个在编译时展开的内联函数，返回值类型为 uint64_t，C++20 标准兼容性。
uint64_t uint64_hi64(uint64_t r0, uint64_t r1, bool& truncated) noexcept {
  int shl = leading_zeroes(r0);
  if (shl == 0) {
    truncated = r1 != 0;
    return r0;
  } else {
    int shr = 64 - shl;
    truncated = (r1 << shl) != 0;
    return (r0 << shl) | (r1 >> shr);
  }
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
// 使用 fastfloat_really_inline 宏定义一个在编译时展开的内联函数，返回值类型为 uint64_t，C++20 标准兼容性。
uint64_t uint32_hi64(uint32_t r0, bool& truncated) noexcept {
  return uint64_hi64(r0, truncated);
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
// 使用 fastfloat_really_inline 宏定义一个在编译时展开的内联函数，返回值类型为 uint64_t，C++20 标准兼容性。
uint64_t uint32_hi64(uint32_t r0, uint32_t r1, bool& truncated) noexcept {
  uint64_t x0 = r0;
  uint64_t x1 = r1;
  return uint64_hi64((x0 << 32) | x1, truncated);
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
// 使用 fastfloat_really_inline 宏定义一个在编译时展开的内联函数，返回值类型为 uint64_t，C++20 标准兼容性。
uint64_t uint32_hi64(uint32_t r0, uint32_t r1, uint32_t r2, bool& truncated) noexcept {
  uint64_t x0 = r0;
  uint64_t x1 = r1;
  uint64_t x2 = r2;
  return uint64_hi64(x0, (x1 << 32) | x2, truncated);
}

// add two small integers, checking for overflow.
// we want an efficient operation. for msvc, where
// we don't have built-in intrinsics, this is still
// pretty fast.
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
// 使用 fastfloat_really_inline 宏定义一个在编译时展开的内联函数，返回值类型为 limb，C++20 标准兼容性。
limb scalar_add(limb x, limb y, bool& overflow) noexcept {
  limb z;
// gcc and clang
#if defined(__has_builtin)
  #if __has_builtin(__builtin_add_overflow)
    if (!cpp20_and_in_constexpr()) {
      overflow = __builtin_add_overflow(x, y, &z);
      return z;
    }
  #endif
#endif

  // generic, this still optimizes correctly on MSVC.
  z = x + y;
  overflow = z < x;
  return z;
}

// multiply two small integers, getting both the high and low bits.
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
// 使用 fastfloat_really_inline 宏定义一个在编译时展开的内联函数，返回值类型为 limb，C++20 标准兼容性。
limb scalar_mul(limb x, limb y, limb& carry) noexcept {
#ifdef FASTFLOAT_64BIT_LIMB
  #if defined(__SIZEOF_INT128__)
  // GCC and clang both define it as an extension.
  __uint128_t z = __uint128_t(x) * __uint128_t(y) + __uint128_t(carry);
  carry = limb(z >> limb_bits);
  return limb(z);
  #else
  // fallback, no native 128-bit integer multiplication with carry.
  // on msvc, this optimizes identically, somehow.
  value128 z = full_multiplication(x, y);
  bool overflow;
  z.low = scalar_add(z.low, carry, overflow);
  z.high += uint64_t(overflow);  // cannot overflow
  carry = z.high;
  return z.low;
  #endif
#else
  uint64_t z = uint64_t(x) * uint64_t(y) + uint64_t(carry);
  carry = limb(z >> limb_bits);
  return limb(z);
#endif
}

// add scalar value to bigint starting from offset.
// used in grade school multiplication
template <uint16_t size>
inline FASTFLOAT_CONSTEXPR20
// 从栈向量的指定位置开始，将一个小整数添加到大整数中
bool small_add_from(stackvec<size>& vec, limb y, size_t start) noexcept {
  size_t index = start;  // 初始化索引为给定的起始位置
  limb carry = y;        // 初始化进位为传入的整数
  bool overflow;         // 溢出标志位

  // 循环直到进位为零或者遍历完栈向量
  while (carry != 0 && index < vec.len()) {
    vec[index] = scalar_add(vec[index], carry, overflow);  // 将当前位置的元素与进位相加
    carry = limb(overflow);  // 更新进位为加法的溢出结果
    index += 1;  // 移动到下一个位置
  }

  // 如果还有进位且超出了栈向量范围，则尝试向栈向量中添加进位
  if (carry != 0) {
    FASTFLOAT_TRY(vec.try_push(carry));
  }

  return true;  // 操作成功返回true
}

// 向大整数添加一个标量值
template <uint16_t size>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
bool small_add(stackvec<size>& vec, limb y) noexcept {
  return small_add_from(vec, y, 0);  // 调用small_add_from函数，从头开始添加标量值
}

// 将大整数乘以一个标量值
template <uint16_t size>
inline FASTFLOAT_CONSTEXPR20
bool small_mul(stackvec<size>& vec, limb y) noexcept {
  limb carry = 0;  // 初始化进位为零

  // 遍历栈向量中的每个元素，将其与标量值相乘
  for (size_t index = 0; index < vec.len(); index++) {
    vec[index] = scalar_mul(vec[index], y, carry);  // 执行乘法运算
  }

  // 如果最后有进位，则尝试将进位添加到栈向量中
  if (carry != 0) {
    FASTFLOAT_TRY(vec.try_push(carry));
  }

  return true;  // 操作成功返回true
}

// 从指定位置开始，将一个大整数加到另一个大整数上，用于普通乘法中
template <uint16_t size>
FASTFLOAT_CONSTEXPR20
bool large_add_from(stackvec<size>& x, limb_span y, size_t start) noexcept {
  // 如果x的长度小于起始位置，或者y的长度超过从start位置开始的剩余x的长度，则尝试调整x的大小
  if (x.len() < start || y.len() > x.len() - start) {
      FASTFLOAT_TRY(x.try_resize(y.len() + start, 0));
  }

  bool carry = false;  // 初始化进位标志为false
  for (size_t index = 0; index < y.len(); index++) {
    limb xi = x[index + start];  // 获取x中对应位置的元素
    limb yi = y[index];          // 获取y中对应位置的元素
    bool c1 = false;             // 用于标记加法操作的溢出情况
    bool c2 = false;             // 用于处理进位时的溢出情况
    xi = scalar_add(xi, yi, c1); // 执行加法操作
    if (carry) {
      xi = scalar_add(xi, 1, c2);  // 如果有进位，则再加1
    }
    x[index + start] = xi;   // 更新x中的元素
    carry = c1 | c2;         // 更新进位状态
  }

  // 处理最后的溢出情况
  if (carry) {
    FASTFLOAT_TRY(small_add_from(x, 1, y.len() + start));
  }

  return true;  // 操作成功返回true
}

// 将一个大整数加到另一个大整数上，从起始位置0开始
template <uint16_t size>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
bool large_add_from(stackvec<size>& x, limb_span y) noexcept {
  return large_add_from(x, y, 0);  // 调用large_add_from函数，从头开始相加
}

// 标准的竖式乘法算法
template <uint16_t size>
FASTFLOAT_CONSTEXPR20
bool long_mul(stackvec<size>& x, limb_span y) noexcept {
  limb_span xs = limb_span(x.data, x.len());  // 创建x的视图
  stackvec<size> z(xs);                      // 创建临时的栈向量z
  limb_span zs = limb_span(z.data, z.len()); // 创建z的视图

  if (y.len() != 0) {
    limb y0 = y[0];                       // 获取y中的第一个元素
    FASTFLOAT_TRY(small_mul(x, y0));      // 将x乘以y的第一个元素
    for (size_t index = 1; index < y.len(); index++) {
      limb yi = y[index];                 // 获取y中的当前元素
      stackvec<size> zi;                  // 创建一个新的栈向量zi
      if (yi != 0) {
        zi.set_len(0);                    // 将zi的长度设置为0
        FASTFLOAT_TRY(zi.try_extend(zs)); // 尝试将z的内容复制到zi中
        FASTFLOAT_TRY(small_mul(zi, yi)); // 将zi乘以yi
        limb_span zis = limb_span(zi.data, zi.len());  // 创建zi的视图
        FASTFLOAT_TRY(large_add_from(x, zis, index));  // 将zi加到x上，从index位置开始
      }
    }
  }

  x.normalize();  // 规范化x
  return true;    // 操作成功返回true
}

// 标准的竖式乘法算法
template <uint16_t size>
FASTFLOAT_CONSTEXPR20
bool large_mul(stackvec<size>& x, limb_span y) noexcept {
  if (y.len() == 1) {
    FASTFLOAT_TRY(small_mul(x, y[0]));  // 如果y的长度为1，则将x乘以y的第一个元素
  } else {
    // 省略部分内容
    FASTFLOAT_TRY(long_mul(x, y));
  }
  return true;


注释：


// 尝试执行长整数乘法，使用 FASTFLOAT_TRY 宏来处理可能的异常
FASTFLOAT_TRY(long_mul(x, y));
// 如果上述操作成功完成，返回 true，表示函数执行成功
}
return true;


这段代码看起来是在一个函数或者代码块中的一部分。`FASTFLOAT_TRY` 是一个宏，用于尝试执行 `long_mul(x, y)` 这个长整数乘法操作，并处理可能发生的异常。如果 `long_mul(x, y)` 执行成功，这个宏可能会有一些内部逻辑来处理成功的情况。最后，函数返回 `true`，表示整个操作成功完成。
// 结构模板 `pow5_tables`，定义了一些与5的幂相关的常量
template <typename = void>
struct pow5_tables {
  // 大步长常量，值为135
  static constexpr uint32_t large_step = 135;
  // 小于64位的5的幂数组
  static constexpr uint64_t small_power_of_5[] = {
    1UL, 5UL, 25UL, 125UL, 625UL, 3125UL, 15625UL, 78125UL, 390625UL,
    1953125UL, 9765625UL, 48828125UL, 244140625UL, 1220703125UL,
    6103515625UL, 30517578125UL, 152587890625UL, 762939453125UL,
    3814697265625UL, 19073486328125UL, 95367431640625UL, 476837158203125UL,
    2384185791015625UL, 11920928955078125UL, 59604644775390625UL,
    298023223876953125UL, 1490116119384765625UL, 7450580596923828125UL,
  };
#ifdef FASTFLOAT_64BIT_LIMB
  // 如果定义了64位枝干，使用64位的大于5的幂数组
  constexpr static limb large_power_of_5[] = {
    1414648277510068013UL, 9180637584431281687UL, 4539964771860779200UL,
    10482974169319127550UL, 198276706040285095UL};
#else
  // 否则使用32位的大于5的幂数组
  constexpr static limb large_power_of_5[] = {
    4279965485U, 329373468U, 4020270615U, 2137533757U, 4287402176U,
    1057042919U, 1071430142U, 2440757623U, 381945767U, 46164893U};
#endif
};

// 大整数类型 `bigint`，继承自 `pow5_tables<>`
struct bigint : pow5_tables<> {
  // 存储大整数的枝干，按小端序排列
  stackvec<bigint_limbs> vec;

  // 默认构造函数，创建一个空的大整数
  FASTFLOAT_CONSTEXPR20 bigint(): vec() {}

  // 构造函数，从一个64位无符号整数值构造大整数
  FASTFLOAT_CONSTEXPR20 bigint(uint64_t value): vec() {
#ifdef FASTFLOAT_64BIT_LIMB
    // 如果是64位枝干，直接插入值
    vec.push_unchecked(value);
#else
    // 否则将64位值拆分为两个32位插入
    vec.push_unchecked(uint32_t(value));
    vec.push_unchecked(uint32_t(value >> 32));
#endif
    // 规范化大整数
    vec.normalize();
  }

  // 获取向量中的高64位，并返回是否截断了位数
  FASTFLOAT_CONSTEXPR20 uint64_t hi64(bool& truncated) const noexcept {
#ifdef FASTFLOAT_64BIT_LIMB
    if (vec.len() == 0) {
      return empty_hi64(truncated);
    } else if (vec.len() == 1) {
      return uint64_hi64(vec.rindex(0), truncated);
    } else {
      uint64_t result = uint64_hi64(vec.rindex(0), vec.rindex(1), truncated);
      truncated |= vec.nonzero(2);
      return result;
    }
#else
    if (vec.len() == 0) {
      return empty_hi64(truncated);
    } else if (vec.len() == 1) {
      return uint32_hi64(vec.rindex(0), truncated);
    } else if (vec.len() == 2) {
      return uint32_hi64(vec.rindex(0), vec.rindex(1), truncated);
    } else {
      uint64_t result = uint32_hi64(vec.rindex(0), vec.rindex(1), vec.rindex(2), truncated);
      truncated |= vec.nonzero(3);
      return result;
    }
#endif
#endif
  }

  // 比较两个大整数，返回较大的值。
  // 假设两者都已归一化。如果返回值为负数，则表示参数 other 较大，
  // 如果返回值为正数，则表示当前对象 (*this) 较大，否则它们相等。
  // limbs 以小端顺序存储，因此必须以逆序比较它们。
  FASTFLOAT_CONSTEXPR20 int compare(const bigint& other) const noexcept {
    if (vec.len() > other.vec.len()) {
      return 1;
    } else if (vec.len() < other.vec.len()) {
      return -1;
    } else {
      for (size_t index = vec.len(); index > 0; index--) {
        limb xi = vec[index - 1];
        limb yi = other.vec[index - 1];
        if (xi > yi) {
          return 1;
        } else if (xi < yi) {
          return -1;
        }
      }
      return 0;
    }
  }

  // 将每个 limb 左移 n 位，将进位加到新的 limb 上。
  // 如果能够成功移动所有数字，则返回 true。
  FASTFLOAT_CONSTEXPR20 bool shl_bits(size_t n) noexcept {
    // 在内部，对于每个项目，我们左移 n 位，并将前一个右移的 limb-bits 添加进去。
    // 例如，我们将左移 2 位（对于 u8）转换为：
    //      b10100100 b01000010
    //      b10 b10010001 b00001000
    FASTFLOAT_DEBUG_ASSERT(n != 0);
    FASTFLOAT_DEBUG_ASSERT(n < sizeof(limb) * 8);

    size_t shl = n;
    size_t shr = limb_bits - shl;
    limb prev = 0;
    for (size_t index = 0; index < vec.len(); index++) {
      limb xi = vec[index];
      vec[index] = (xi << shl) | (prev >> shr);
      prev = xi;
    }

    limb carry = prev >> shr;
    if (carry != 0) {
      return vec.try_push(carry);
    }
    return true;
  }

  // 将 limb 左移 n 个 limbs。
  FASTFLOAT_CONSTEXPR20 bool shl_limbs(size_t n) noexcept {
    FASTFLOAT_DEBUG_ASSERT(n != 0);
    if (n + vec.len() > vec.capacity()) {
      return false;
    } else if (!vec.is_empty()) {
      // 移动 limbs
      limb* dst = vec.data + n;
      const limb* src = vec.data;
      std::copy_backward(src, src + vec.len(), dst + vec.len());
      // 填充空 limbs
      limb* first = vec.data;
      limb* last = first + n;
      ::std::fill(first, last, 0);
      vec.set_len(n + vec.len());
      return true;
    } else {
      return true;
    }
  }

  // 将 limb 左移 n 位。
  FASTFLOAT_CONSTEXPR20 bool shl(size_t n) noexcept {
    size_t rem = n % limb_bits;
    size_t div = n / limb_bits;
    if (rem != 0) {
      FASTFLOAT_TRY(shl_bits(rem));
    }
    if (div != 0) {
      FASTFLOAT_TRY(shl_limbs(div));
    }
    return true;
  }

  // 获取 bigint 中前导零的数量。
  FASTFLOAT_CONSTEXPR20 int ctlz() const noexcept {
    if (vec.is_empty()) {
      return 0;
    } else {
#ifdef FASTFLOAT_64BIT_LIMB
      return leading_zeroes(vec.rindex(0));
#else
      // 为 32 位类型没有定义专用的 leading_zeroes，因此此处无用。
      uint64_t r0 = vec.rindex(0);
      return leading_zeroes(r0 << 32);
#endif
  }
}

// 获取大整数的位数。
FASTFLOAT_CONSTEXPR20 int bit_length() const noexcept {
  // 计算大整数的前导零数。
  int lz = ctlz();
  // 返回大整数的位长度。
  return int(limb_bits * vec.len()) - lz;
}

FASTFLOAT_CONSTEXPR20 bool mul(limb y) noexcept {
  // 使用小整数乘法操作。
  return small_mul(vec, y);
}

FASTFLOAT_CONSTEXPR20 bool add(limb y) noexcept {
  // 使用小整数加法操作。
  return small_add(vec, y);
}

// 按2的幂乘法运算。
FASTFLOAT_CONSTEXPR20 bool pow2(uint32_t exp) noexcept {
  // 调用左移操作，相当于乘以2的幂。
  return shl(exp);
}

// 按5的幂乘法运算。
FASTFLOAT_CONSTEXPR20 bool pow5(uint32_t exp) noexcept {
  // 乘以5的幂
  size_t large_length = sizeof(large_power_of_5) / sizeof(limb);
  limb_span large = limb_span(large_power_of_5, large_length);
  while (exp >= large_step) {
    // 使用大整数乘法操作，直到指数变小。
    FASTFLOAT_TRY(large_mul(vec, large));
    exp -= large_step;
  }
#ifdef FASTFLOAT_64BIT_LIMB
    // 如果定义了 FASTFLOAT_64BIT_LIMB 宏，则使用以下参数
    uint32_t small_step = 27;  // 设置步长为 27
    limb max_native = 7450580596923828125UL;  // 设置最大本地变量为 7450580596923828125UL
#else
    // 如果未定义 FASTFLOAT_64BIT_LIMB 宏，则使用以下参数
    uint32_t small_step = 13;  // 设置步长为 13
    limb max_native = 1220703125U;  // 设置最大本地变量为 1220703125U
#endif
    while (exp >= small_step) {
      FASTFLOAT_TRY(small_mul(vec, max_native));
      exp -= small_step;
    }
    if (exp != 0) {
      // 解决 Clang 编译器的 bug https://godbolt.org/z/zedh7rrhc
      // 类似于 https://github.com/llvm/llvm-project/issues/47746 的问题，
      // 但该问题的已知解决方法在此并不适用
      FASTFLOAT_TRY(
        small_mul(vec, limb(((void)small_power_of_5[0], small_power_of_5[exp])))
      );
    }

    return true;
  }

  // 模拟以 10 的指数进行乘法运算
  FASTFLOAT_CONSTEXPR20 bool pow10(uint32_t exp) noexcept {
    FASTFLOAT_TRY(pow5(exp));
    return pow2(exp);
  }
};

} // namespace fast_float

#endif

#ifndef FASTFLOAT_DIGIT_COMPARISON_H
#define FASTFLOAT_DIGIT_COMPARISON_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>


namespace fast_float {

// 1e0 to 1e19
constexpr static uint64_t powers_of_ten_uint64[] = {
    1UL, 10UL, 100UL, 1000UL, 10000UL, 100000UL, 1000000UL, 10000000UL, 100000000UL,
    1000000000UL, 10000000000UL, 100000000000UL, 1000000000000UL, 10000000000000UL,
    100000000000000UL, 1000000000000000UL, 10000000000000000UL, 100000000000000000UL,
    1000000000000000000UL, 10000000000000000000UL};

// 计算一个数字的科学计数法指数
// 该算法虽然未经优化，但不会对性能产生实际影响：
// 要获得更快的算法，我们需要减慢对更快算法的性能，而这个算法仍然很快
template <typename UC>
fastfloat_really_inline FASTFLOAT_CONSTEXPR14
int32_t scientific_exponent(parsed_number_string_t<UC> & num) noexcept {
  uint64_t mantissa = num.mantissa;
  int32_t exponent = int32_t(num.exponent);
  while (mantissa >= 10000) {
    mantissa /= 10000;
    exponent += 4;
  }
  while (mantissa >= 100) {
    mantissa /= 100;
    exponent += 2;
  }
  while (mantissa >= 10) {
    mantissa /= 10;
    exponent += 1;
  }
  return exponent;
}

// 将本地浮点数转换为扩展精度浮点数
template <typename T>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
adjusted_mantissa to_extended(T value) noexcept {
  using equiv_uint = typename binary_format<T>::equiv_uint;
  constexpr equiv_uint exponent_mask = binary_format<T>::exponent_mask();
  constexpr equiv_uint mantissa_mask = binary_format<T>::mantissa_mask();
  constexpr equiv_uint hidden_bit_mask = binary_format<T>::hidden_bit_mask();

  adjusted_mantissa am;
  int32_t bias = binary_format<T>::mantissa_explicit_bits() - binary_format<T>::minimum_exponent();
  equiv_uint bits;
#if FASTFLOAT_HAS_BIT_CAST
  bits = std::bit_cast<equiv_uint>(value);
#else
  ::memcpy(&bits, &value, sizeof(T));
#endif
  if ((bits & exponent_mask) == 0) {
    // 非规范化数
    am.power2 = 1 - bias;
    // 如果该浮点数是特殊情况（例如无穷大、NaN或零）
    if (bits & special_mask) {
        // 设置为特殊值，例如无穷大、NaN或零
        am.special = true;
        // 将尾数部分设置为位掩码后的结果
        am.mantissa = bits & mantissa_mask;
    } else {
        // 如果是普通浮点数
        // 计算指数部分，右移得到指数值
        am.power2 = int32_t((bits & exponent_mask) >> binary_format<T>::mantissa_explicit_bits());
        // 减去偏移值得到真实的指数
        am.power2 -= bias;
        // 将尾数部分设置为位掩码和隐藏位的结果
        am.mantissa = (bits & mantissa_mask) | hidden_bit_mask;
    }

    // 返回解析后的浮点数表示
    return am;
// 结束前一段代码块的大括号
}

// 获取b和b+u之间一半的扩展精度值。
// 给定一个代表b的本机浮点数，我们需要调整它在b和b+u之间的中点。
template <typename T>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
adjusted_mantissa to_extended_halfway(T value) noexcept {
  // 将value转换为扩展格式
  adjusted_mantissa am = to_extended(value);
  // 将尾数左移一位，然后加一，表示在b和b+u之间的中点
  am.mantissa <<= 1;
  am.mantissa += 1;
  // 减少二的幂指数，因为左移操作增加了指数
  am.power2 -= 1;
  return am;
}

// 将扩展精度浮点数四舍五入到最接近的机器浮点数。
template <typename T, typename callback>
fastfloat_really_inline FASTFLOAT_CONSTEXPR14
void round(adjusted_mantissa& am, callback cb) noexcept {
  // 计算尾数的位移量
  int32_t mantissa_shift = 64 - binary_format<T>::mantissa_explicit_bits() - 1;
  if (-am.power2 >= mantissa_shift) {
    // 处理非规格化浮点数的情况
    int32_t shift = -am.power2 + 1;
    // 调用回调函数处理非规格化浮点数的舍入
    cb(am, std::min<int32_t>(shift, 64));
    // 检查是否需要进位：如果四舍五入到隐藏位，则将幂指数设置为0
    am.power2 = (am.mantissa < (uint64_t(1) << binary_format<T>::mantissa_explicit_bits())) ? 0 : 1;
    return;
  }

  // 处理普通浮点数的情况，使用默认的位移量
  cb(am, mantissa_shift);

  // 检查是否需要进位
  if (am.mantissa >= (uint64_t(2) << binary_format<T>::mantissa_explicit_bits())) {
    am.mantissa = (uint64_t(1) << binary_format<T>::mantissa_explicit_bits());
    am.power2++;
  }

  // 检查是否为无穷大：可能会导致幂指数无穷大
  am.mantissa &= ~(uint64_t(1) << binary_format<T>::mantissa_explicit_bits());
  if (am.power2 >= binary_format<T>::infinite_power()) {
    am.power2 = binary_format<T>::infinite_power();
    am.mantissa = 0;
  }
}

template <typename callback>
fastfloat_really_inline FASTFLOAT_CONSTEXPR14
void round_nearest_tie_even(adjusted_mantissa& am, int32_t shift, callback cb) noexcept {
  const uint64_t mask
  = (shift == 64)
    ? UINT64_MAX
    : (uint64_t(1) << shift) - 1;
  const uint64_t halfway
  = (shift == 0)
    ? 0
    : uint64_t(1) << (shift - 1);
  uint64_t truncated_bits = am.mantissa & mask;
  bool is_above = truncated_bits > halfway;
  bool is_halfway = truncated_bits == halfway;

  // 将数字移入正确的位置
  if (shift == 64) {
    am.mantissa = 0;
  } else {
    am.mantissa >>= shift;
  }
  am.power2 += shift;

  bool is_odd = (am.mantissa & 1) == 1;
  // 调用回调函数处理舍入到最接近的偶数模式
  am.mantissa += uint64_t(cb(is_odd, is_halfway, is_above));
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR14
void round_down(adjusted_mantissa& am, int32_t shift) noexcept {
  // 将数字移入正确的位置
  if (shift == 64) {
    am.mantissa = 0;
  } else {
    am.mantissa >>= shift;
  }
  am.power2 += shift;
}

template <typename UC>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
void skip_zeros(UC const * & first, UC const * last) noexcept {
  uint64_t val;
  while (!cpp20_and_in_constexpr() && std::distance(first, last) >= int_cmp_len<UC>()) {
    ::memcpy(&val, first, sizeof(uint64_t));
    // 如果不是零，退出循环
    if (val != int_cmp_zeros<UC>()) {
      break;
    }
    first += int_cmp_len<UC>();
  }
  // 跳过零之后的字符
  while (first != last) {
    # 检查指针 `first` 所指向的字符是否不是字符 '0'
    if (*first != UC('0')) {
      # 如果不是 '0'，跳出循环
      break;
    }
    # 指针 `first` 向前移动一个位置，继续下一次循环
    first++;
// 结束前面的 C++ 函数定义和模板
}

// 判断是否存在被截断的非零数字，所有字符必须是有效数字
// 使用模板参数 UC，表示通用字符类型
template <typename UC>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
bool is_truncated(UC const * first, UC const * last) noexcept {
  // 当不是在 constexpr 环境下，并且要比较的距离大于 UC 类型的整数比较长度
  while (!cpp20_and_in_constexpr() && std::distance(first, last) >= int_cmp_len<UC>()) {
    uint64_t val;
    ::memcpy(&val, first, sizeof(uint64_t));
    // 如果值不等于 UC 类型的整数比较 0 值，返回 true
    if (val != int_cmp_zeros<UC>()) {
      return true;
    }
    // 移动 first 指针，增加整数比较长度的距离
    first += int_cmp_len<UC>();
  }
  // 处理剩余的字符
  while (first != last) {
    // 如果当前字符不是 '0'，返回 true
    if (*first != UC('0')) {
      return true;
    }
    // 移动 first 指针到下一个字符
    ++first;
  }
  // 没有截断，返回 false
  return false;
}

// 使用模板参数 UC，重载 is_truncated 函数，接受 span<const UC> 类型参数
template <typename UC>
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
bool is_truncated(span<const UC> s) noexcept {
  // 调用前面定义的 is_truncated 函数，传入 span 的指针和长度
  return is_truncated(s.ptr, s.ptr + s.len());
}

// 以下几个函数都是未使用的函数，这里不需要添加注释说明其作用

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
void parse_eight_digits(const char16_t*& , limb& , size_t& , size_t& ) noexcept {
  // 当前未使用
}

fastfloat_really_inline FASTFLOAT_CONSTEXPR20
void parse_eight_digits(const char32_t*& , limb& , size_t& , size_t& ) noexcept {
  // 当前未使用
}

// 解析八位数字字符串，转换为 limb 类型，并更新计数器
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
void parse_eight_digits(const char*& p, limb& value, size_t& counter, size_t& count) noexcept {
  // 将当前八位字符转换为 limb 类型的值，乘以 100000000 并累加到 value
  value = value * 100000000 + parse_eight_digits_unrolled(p);
  // 移动 p 指针 8 位字符
  p += 8;
  // 更新计数器
  counter += 8;
  count += 8;
}

// 解析单个数字字符，转换为 limb 类型，并更新计数器
template <typename UC>
fastfloat_really_inline FASTFLOAT_CONSTEXPR14
void parse_one_digit(UC const *& p, limb& value, size_t& counter, size_t& count) noexcept {
  // 将当前字符转换为 limb 类型的值，乘以 10 并累加到 value
  value = value * 10 + limb(*p - UC('0'));
  // 移动 p 指针到下一个字符
  p++;
  // 更新计数器
  counter++;
  count++;
}

// 向 bigint 添加一个 limb 类型的值，乘以指定的 power 参数，然后加上 value
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
void add_native(bigint& big, limb power, limb value) noexcept {
  // 调用 bigint 类的 mul 方法，乘以 power
  big.mul(power);
  // 调用 bigint 类的 add 方法，加上 value
  big.add(value);
}

// 对 bigint 进行四舍五入，增加到 count 参数
fastfloat_really_inline FASTFLOAT_CONSTEXPR20
void round_up_bigint(bigint& big, size_t& count) noexcept {
  // 添加 1 到 bigint，实现四舍五入
  add_native(big, 10, 1);
  // 增加 count 计数
  count++;
}

// 解析有效数字位数到 bigint 中
template <typename UC>
inline FASTFLOAT_CONSTEXPR20
void parse_mantissa(bigint& result, parsed_number_string_t<UC>& num, size_t max_digits, size_t& digits) noexcept {
  // 尝试最小化大整数和标量乘法的数量
  // 因此，尝试每次解析 8 位数字，并乘以最大的标量值（9 或 19 位数字）进行每一步
  size_t counter = 0;
  digits = 0;
  limb value = 0;
#ifdef FASTFLOAT_64BIT_LIMB
  size_t step = 19;
#else
  size_t step = 9;
#endif

  // 处理所有整数位数字
  UC const * p = num.integer.ptr;
  UC const * pend = p + num.integer.len();
  skip_zeros(p, pend);
  // 处理所有数字，每次循环增加 step
  while (p != pend) {
    // 如果 UC 和 char 是相同类型
    if (std::is_same<UC,char>::value) {
      // 当剩余的字符数、步长和最大数字位数均大于等于 8 时，处理八位数字的情况
      while ((std::distance(p, pend) >= 8) && (step - counter >= 8) && (max_digits - digits >= 8)) {
        // 解析八位数字
        parse_eight_digits(p, value, counter, digits);
      }
    }
    // 处理单个数字的情况，直到达到步长、结束指针或最大数字位数
    while (counter < step && p != pend && digits < max_digits) {
      // 解析单个数字
      parse_one_digit(p, value, counter, digits);
    }
    // 如果已解析的数字等于最大数字位数
    if (digits == max_digits) {
      // 将临时值添加到结果中，然后检查是否截断了任何数字
      add_native(result, limb(powers_of_ten_uint64[counter]), value);
      // 检查是否发生了截断
      bool truncated = is_truncated(p, pend);
      // 如果存在小数部分，再次检查是否截断
      if (num.fraction.ptr != nullptr) {
        truncated |= is_truncated(num.fraction);
      }
      // 如果发生了截断，则对结果进行四舍五入
      if (truncated) {
        round_up_bigint(result, digits);
      }
      return;
    } else {
      // 将临时值添加到结果中
      add_native(result, limb(powers_of_ten_uint64[counter]), value);
      // 重置计数器和数值
      counter = 0;
      value = 0;
    }
  }

  // 如果存在小数部分，则添加小数部分的数字
  // 如果小数部分的指针不为空
  if (num.fraction.ptr != nullptr) {
    // 将指针设置为小数部分的起始位置，将结束指针设置为小数部分的末尾位置
    p = num.fraction.ptr;
    pend = p + num.fraction.len();
    // 如果当前没有解析任何数字，则跳过前导零
    if (digits == 0) {
      skip_zeros(p, pend);
    }
    // 处理所有数字，以步长为单位进行循环处理
    while (p != pend) {
      // 如果 UC 和 char 是相同类型
      if (std::is_same<UC,char>::value) {
        // 当剩余的字符数、步长和最大数字位数均大于等于 8 时，处理八位数字的情况
        while ((std::distance(p, pend) >= 8) && (step - counter >= 8) && (max_digits - digits >= 8)) {
          // 解析八位数字
          parse_eight_digits(p, value, counter, digits);
        }
      }
      // 处理单个数字的情况，直到达到步长、结束指针或最大数字位数
      while (counter < step && p != pend && digits < max_digits) {
        // 解析单个数字
        parse_one_digit(p, value, counter, digits);
      }
      // 如果已解析的数字等于最大数字位数
      if (digits == max_digits) {
        // 将临时值添加到结果中，然后检查是否截断了任何数字
        add_native(result, limb(powers_of_ten_uint64[counter]), value);
        // 检查是否发生了截断
        bool truncated = is_truncated(p, pend);
        // 如果发生了截断，则对结果进行四舍五入
        if (truncated) {
          round_up_bigint(result, digits);
        }
        return;
      } else {
        // 将临时值添加到结果中
        add_native(result, limb(powers_of_ten_uint64[counter]), value);
        // 重置计数器和数值
        counter = 0;
        value = 0;
      }
    }
  }

  // 如果计数器不为零，则将临时值添加到结果中
  if (counter != 0) {
    add_native(result, limb(powers_of_ten_uint64[counter]), value);
  }
// 用于计算正数位的调整后尾数。根据给定的大整数和指数，计算尾数和二进制指数。
template <typename T>
inline FASTFLOAT_CONSTEXPR20
adjusted_mantissa positive_digit_comp(bigint& bigmant, int32_t exponent) noexcept {
  // 确保大整数以10的指数形式计算
  FASTFLOAT_ASSERT(bigmant.pow10(uint32_t(exponent)));

  // 初始化调整后尾数对象
  adjusted_mantissa answer;
  bool truncated;

  // 获取大整数的高64位作为尾数
  answer.mantissa = bigmant.hi64(truncated);

  // 计算偏置值，用于校准尾数
  int bias = binary_format<T>::mantissa_explicit_bits() - binary_format<T>::minimum_exponent();
  answer.power2 = bigmant.bit_length() - 64 + bias;

  // 调用round函数进行四舍五入，根据截断情况调整
  round<T>(answer, [truncated](adjusted_mantissa& a, int32_t shift) {
    round_nearest_tie_even(a, shift, [truncated](bool is_odd, bool is_halfway, bool is_above) -> bool {
      // 根据奇偶性、截断和四舍五入情况返回布尔值
      return is_above || (is_halfway && truncated) || (is_odd && is_halfway);
    });
  });

  return answer; // 返回调整后的尾数对象
}

// 缩放操作很简单：我们有实数位`m * 10^e`，理论位`n * 2^f`。
// 由于`e`总是负数，为了使它们按相同比例缩放，我们做`n * 2^f * 5^-f`，这样我们就有了`m * 2^e`。
// 然后我们需要按`2^(f-e)`缩放，这样两个显著数字就具有相同的数量级。
template <typename T>
inline FASTFLOAT_CONSTEXPR20
adjusted_mantissa negative_digit_comp(bigint& bigmant, adjusted_mantissa am, int32_t exponent) noexcept {
  bigint& real_digits = bigmant;
  int32_t real_exp = exponent;

  // 获取`b`的值，并向下舍入，然后获取`b+h`的bigint表示
  adjusted_mantissa am_b = am;
  round<T>(am_b, [](adjusted_mantissa&a, int32_t shift) { round_down(a, shift); });
  T b;
  to_float(false, am_b, b);
  adjusted_mantissa theor = to_extended_halfway(b);
  bigint theor_digits(theor.mantissa);
  int32_t theor_exp = theor.power2;

  // 将实数位和理论位缩放到相同的幂次
  int32_t pow2_exp = theor_exp - real_exp;
  uint32_t pow5_exp = uint32_t(-real_exp);
  if (pow5_exp != 0) {
    FASTFLOAT_ASSERT(theor_digits.pow5(pow5_exp));
  }
  if (pow2_exp > 0) {
    FASTFLOAT_ASSERT(theor_digits.pow2(uint32_t(pow2_exp)));
  } else if (pow2_exp < 0) {
    FASTFLOAT_ASSERT(real_digits.pow2(uint32_t(-pow2_exp)));
  }

  // 比较数字，并根据比较结果进行四舍五入
  int ord = real_digits.compare(theor_digits);
  adjusted_mantissa answer = am;
  round<T>(answer, [ord](adjusted_mantissa& a, int32_t shift) {
    round_nearest_tie_even(a, shift, [ord](bool is_odd, bool _, bool __) -> bool {
      (void)_;  // 我们已经完成了比较，所以不需要这些参数
      (void)__; // 我们已经完成了比较，所以不需要这些参数
      if (ord > 0) {
        return true;
      } else if (ord < 0) {
        return false;
      } else {
        return is_odd;
      }
    });
  });

  return answer; // 返回调整后的尾数对象
}

// 将显著数字解析为大整数，以便明确地四舍五入显著数字。
// 在这里，我们试图确定如何四舍五入接近`b+h`的扩展浮点表示，
// 介于`b`（舍入为下一个最接近的`b+u`，下一个正浮点数。
// 算法始终正确，并采用两种方法之一。当指数相对于有效数字为正数时（例如1234），我们创建一个大整数表示，获取高64位，
// 确定是否截断了任何低位，并使用它来指导舍入。如果指数相对于有效数字为负数（例如1.2345），我们将`b`的理论表示创建为大整数类型，
// 缩放为与实际数字相同的二进制指数。然后我们比较两者的大整数表示，并使用它来指导舍入。
template <typename T, typename UC>
inline FASTFLOAT_CONSTEXPR20
adjusted_mantissa digit_comp(parsed_number_string_t<UC>& num, adjusted_mantissa am) noexcept {
  // 移除无效指数偏差
  am.power2 -= invalid_am_bias;

  // 获取科学计数法指数
  int32_t sci_exp = scientific_exponent(num);
  size_t max_digits = binary_format<T>::max_digits();
  size_t digits = 0;
  bigint bigmant;
  // 解析尾数
  parse_mantissa(bigmant, num, max_digits, digits);
  // 由于digits最多为max_digits，不会下溢。
  int32_t exponent = sci_exp + 1 - int32_t(digits);
  // 如果指数大于等于0，则使用正数比较函数
  if (exponent >= 0) {
    return positive_digit_comp<T>(bigmant, exponent);
  } else { // 否则使用负数比较函数
    return negative_digit_comp<T>(bigmant, am, exponent);
  }
}

} // namespace fast_float

#endif

#ifndef FASTFLOAT_PARSE_NUMBER_H
#define FASTFLOAT_PARSE_NUMBER_H


#include <cmath>
#include <cstring>
#include <limits>
#include <system_error>

namespace fast_float {


namespace detail {
/**
 * Special case +inf, -inf, nan, infinity, -infinity.
 * The case comparisons could be made much faster given that we know that the
 * strings a null-free and fixed.
 **/
template <typename T, typename UC>
from_chars_result_t<UC> FASTFLOAT_CONSTEXPR14
parse_infnan(UC const * first, UC const * last, T &value)  noexcept  {
  from_chars_result_t<UC> answer{};
  answer.ptr = first;
  answer.ec = std::errc(); // be optimistic
  bool minusSign = false;
  // 如果第一个字符是负号，则标记为负数
  if (*first == UC('-')) { // 假设first < last，所以可以直接解引用，不需要检查；根据C++17 20.19.3.(7.1)，在这里明确禁止使用'+'
      minusSign = true;
      ++first;
  }
#ifdef FASTFLOAT_ALLOWS_LEADING_PLUS // 默认禁用
  // 如果第一个字符是加号，则跳过
  if (*first == UC('+')) {
      ++first;
  }
#endif
  // 如果剩余字符长度大于等于3
  if (last - first >= 3) {
    // 检查是否为NaN（不区分大小写），若是则处理
    if (fastfloat_strncasecmp(first, str_const_nan<UC>(), 3)) {
      // 将指针移动到跳过NaN标识
      answer.ptr = (first += 3);
      // 根据符号标志设置NaN的值
      value = minusSign ? -std::numeric_limits<T>::quiet_NaN() : std::numeric_limits<T>::quiet_NaN();
      // 检查是否存在可能的nan(n-char-seq-opt)形式，参考C++17 20.19.3.7，C11 7.20.1.3.3。
      // 至少MSVC生成nan(ind)和nan(snan)。
      if(first != last && *first == UC('(')) {
        // 遍历可能的字符序列，确定是否是有效的nan(n-char-seq-opt)
        for(UC const * ptr = first + 1; ptr != last; ++ptr) {
          if (*ptr == UC(')')) {
            // 找到了匹配的结束括号，指针移动到括号后
            answer.ptr = ptr + 1; // valid nan(n-char-seq-opt)
            break;
          }
          else if(!((UC('a') <= *ptr && *ptr <= UC('z')) || (UC('A') <= *ptr && *ptr <= UC('Z')) || (UC('0') <= *ptr && *ptr <= UC('9')) || *ptr == UC('_')))
            break; // forbidden char, not nan(n-char-seq-opt)
        }
      }
      return answer;
    }
    // 检查是否为无穷大（不区分大小写），若是则处理
    if (fastfloat_strncasecmp(first, str_const_inf<UC>(), 3)) {
      // 如果长度足够长并且匹配了无穷大的后续部分，则移动指针跳过整个表示无穷大的部分
      if ((last - first >= 8) && fastfloat_strncasecmp(first + 3, str_const_inf<UC>() + 3, 5)) {
        answer.ptr = first + 8;
      } else {
        answer.ptr = first + 3;
      }
      // 根据符号标志设置无穷大的值
      value = minusSign ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
      return answer;
    }
  }
  // 若以上条件均不满足，则将错误码设置为invalid_argument
  answer.ec = std::errc::invalid_argument;
  return answer;
} // namespace detail



/**
 * Ends the namespace 'detail' block.
 */

/**
 * Returns true if the floating-pointing rounding mode is to 'nearest'.
 * It is the default on most system. This function is meant to be inexpensive.
 * Credit : @mwalcott3
 */
fastfloat_really_inline bool rounds_to_nearest() noexcept {
  // https://lemire.me/blog/2020/06/26/gcc-not-nearest/
#if (FLT_EVAL_METHOD != 1) && (FLT_EVAL_METHOD != 0)
  return false;
#endif
  // See
  // A fast function to check your floating-point rounding mode
  // https://lemire.me/blog/2022/11/16/a-fast-function-to-check-your-floating-point-rounding-mode/
  //
  // This function is meant to be equivalent to :
  // prior: #include <cfenv>
  //  return fegetround() == FE_TONEAREST;
  // However, it is expected to be much faster than the fegetround()
  // function call.
  //
  // The volatile keyword prevents the compiler from computing the function
  // at compile-time.
  // There might be other ways to prevent compile-time optimizations (e.g., asm).
  // The value does not need to be std::numeric_limits<float>::min(), any small
  // value so that 1 + x should round to 1 would do (after accounting for excess
  // precision, as in 387 instructions).
  static volatile float fmin = std::numeric_limits<float>::min();
  float fmini = fmin; // we copy it so that it gets loaded at most once.
  //
  // Explanation:
  // Only when fegetround() == FE_TONEAREST do we have that
  // fmin + 1.0f == 1.0f - fmin.
  //
  // FE_UPWARD:
  //  fmin + 1.0f > 1
  //  1.0f - fmin == 1
  //
  // FE_DOWNWARD or  FE_TOWARDZERO:
  //  fmin + 1.0f == 1
  //  1.0f - fmin < 1
  //
  // Note: This may fail to be accurate if fast-math has been
  // enabled, as rounding conventions may not apply.
  #ifdef FASTFLOAT_VISUAL_STUDIO
  #   pragma warning(push)
  //  todo: is there a VS warning?
  //  see https://stackoverflow.com/questions/46079446/is-there-a-warning-for-floating-point-equality-checking-in-visual-studio-2013
  #elif defined(__clang__)
  #   pragma clang diagnostic push
  #   pragma clang diagnostic ignored "-Wfloat-equal"
  #elif defined(__GNUC__)
  #   pragma GCC diagnostic push
  #   pragma GCC diagnostic ignored "-Wfloat-equal"
  #endif
  return (fmini + 1.0f == 1.0f - fmini);
  #ifdef FASTFLOAT_VISUAL_STUDIO
  #   pragma warning(pop)
  #elif defined(__clang__)
  #   pragma clang diagnostic pop
  #elif defined(__GNUC__)
  #   pragma GCC diagnostic pop
  #endif
}

/**
 * Begins the definition of the 'from_chars' function template.
 * Converts characters to a numeric value of type T, using specified formatting.
 * It delegates to 'from_chars_advanced' with default parsing options.
 *
 * @tparam T         The type to convert the characters to.
 * @tparam UC        The type of characters (usually char or wchar_t).
 * @param first      Pointer to the beginning of the character range.
 * @param last       Pointer to the end of the character range.
 * @param value      Reference to store the parsed numeric value.
 * @param fmt        Format specifier for parsing (default: chars_format::general).
 * @return           Result of the conversion operation.
 */
FASTFLOAT_CONSTEXPR20
from_chars_result_t<UC> from_chars(UC const * first, UC const * last,
                                   T &value, chars_format fmt /*= chars_format::general*/)  noexcept  {
  return from_chars_advanced(first, last, value, parse_options_t<UC>{fmt});
}



/**
 * Ends the definition of the 'from_chars' function template.
 */
FASTFLOAT_CONSTEXPR20



/**
 * Begins a new definition (presumably another function template) after 'from_chars'.
 */
// 定义函数模板 from_chars_advanced，用于将字符序列解析为浮点数值，并返回解析结果的结构体
template<typename UC>
from_chars_result_t<UC> from_chars_advanced(UC const * first, UC const * last,
                                      T &value, parse_options_t<UC> options)  noexcept  {

  // 断言检查模板参数 T 必须是 double 或 float 类型，否则静态断言失败
  static_assert (std::is_same<T, double>::value || std::is_same<T, float>::value, "only float and double are supported");
  // 断言检查模板参数 UC 必须是 char、wchar_t、char16_t 或 char32_t 类型，否则静态断言失败
  static_assert (std::is_same<UC, char>::value ||
                 std::is_same<UC, wchar_t>::value ||
                 std::is_same<UC, char16_t>::value ||
                 std::is_same<UC, char32_t>::value , "only char, wchar_t, char16_t and char32_t are supported");

  // 如果定义了 FASTFLOAT_SKIP_WHITE_SPACE 宏，则跳过空白字符
#ifdef FASTFLOAT_SKIP_WHITE_SPACE  // disabled by default
  while ((first != last) && fast_float::is_space(uint8_t(*first))) {
    first++;
  }
#endif

  // 如果字符序列为空，则返回无效参数错误
  if (first == last) {
    answer.ec = std::errc::invalid_argument;
    answer.ptr = first;
    return answer;
  }

  // 解析字符序列为数字字符串结构体
  parsed_number_string_t<UC> pns = parse_number_string<UC>(first, last, options);

  // 如果解析失败，则调用 parse_infnan 函数处理无穷大或 NaN 的情况
  if (!pns.valid) {
    return detail::parse_infnan(first, last, value);
  }

  // 初始化返回结果的错误码为无错误，指针指向最后匹配的位置
  answer.ec = std::errc(); // be optimistic
  answer.ptr = pns.lastmatch;

  // 实现 Clinger 的快速路径，保证在任何舍入模式下都采用最接近舍入
  // 详细说明快速路径实现的复杂性，假定 detail::rounds_to_nearest() 返回 true
  if (binary_format<T>::min_exponent_fast_path() <= pns.exponent && pns.exponent <= binary_format<T>::max_exponent_fast_path() && !pns.too_many_digits) {
    // 在系统支持舍入至最接近模式时，采用传统的 Clinger 快速路径
    if(!cpp20_and_in_constexpr() && detail::rounds_to_nearest())  {
      // 当 fegetround() == FE_TONEAREST 时采用 Clinger 的快速路径
      if (pns.mantissa <= binary_format<T>::max_mantissa_fast_path()) {
        // 根据快速路径计算浮点数值
        value = T(pns.mantissa);
        if (pns.exponent < 0) { value = value / binary_format<T>::exact_power_of_ten(-pns.exponent); }
        else { value = value * binary_format<T>::exact_power_of_ten(pns.exponent); }
        if (pns.negative) { value = -value; }
        return answer;
      }
    } else {
      // 当 fegetround() != FE_TONEAREST 时，采用修改后的 Clinger 快速路径，受 Jakub Jelínek 提议启发
      if (pns.exponent >= 0 && pns.mantissa <= binary_format<T>::max_mantissa_fast_path(pns.exponent)) {
#if defined(__clang__)
        // 对于 Clang 编译器，当 fegetround() == FE_DOWNWARD 时，将 0 映射为 -0.0
        if(pns.mantissa == 0) {
          value = pns.negative ? -0. : 0.;
          return answer;
        }
#endif
        // 将解析出来的数值乘以十的指数次方，得到最终的浮点数值
        value = T(pns.mantissa) * binary_format<T>::exact_power_of_ten(pns.exponent);
        // 如果解析结果为负数，则取负值
        if (pns.negative) { value = -value; }
        // 返回解析结果
        return answer;
      }
    }
  }
  // 计算浮点数的调整后的尾数和二进制指数
  adjusted_mantissa am = compute_float<binary_format<T>>(pns.exponent, pns.mantissa);
  // 如果存在过多的数字，并且浮点数的二进制指数大于等于零
  if(pns.too_many_digits && am.power2 >= 0) {
    // 如果增加一个单位后的浮点数不等于原浮点数，则重新计算错误浮点数
    if(am != compute_float<binary_format<T>>(pns.exponent, pns.mantissa + 1)) {
      am = compute_error<binary_format<T>>(pns.exponent, pns.mantissa);
    }
  }
  // 如果计算浮点数时得到的二进制指数无效（小于零），则采用更复杂的数字比较方法
  if(am.power2 < 0) { am = digit_comp<T>(pns, am); }
  // 将调整后的浮点数转换为实际的浮点值
  to_float(pns.negative, am, value);
  // 检测是否发生了浮点数的溢出或下溢
  if ((pns.mantissa != 0 && am.mantissa == 0 && am.power2 == 0) || am.power2 == binary_format<T>::infinite_power()) {
    // 如果溢出或下溢，则设置错误码为结果超出范围
    answer.ec = std::errc::result_out_of_range;
  }
  // 返回最终解析结果
  return answer;
}

} // namespace fast_float

#endif


这段代码是 C++ 中的一段函数实现，涉及浮点数解析和处理。
```