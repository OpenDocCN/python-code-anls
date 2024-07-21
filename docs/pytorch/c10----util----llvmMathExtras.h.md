# `.\pytorch\c10\util\llvmMathExtras.h`

```
/// LLVM Project的头文件，包含一些有用的数学函数声明
//===-- llvm/Support/MathExtras.h - Useful math functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some functions that are useful for math stuff.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <c10/util/bit_cast.h>  // 引入bit_cast工具函数

#include <algorithm>   // 标准算法库
#include <cassert>     // 断言
#include <climits>     // 包含整数类型的常量
#include <cmath>       // 数学函数
#include <cstdint>     // C99标准整数类型
#include <cstring>     // C风格字符串操作
#include <limits>      // 数值极限
#include <type_traits> // 类型特性元编程工具

#ifdef __ANDROID_NDK__
#include <android/api-level.h>  // Android平台API级别
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0  // 定义__has_builtin宏，检查内建函数的支持
#endif

#ifndef LLVM_GNUC_PREREQ
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define LLVM_GNUC_PREREQ(maj, min, patch)                             \
  ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) + __GNUC_PATCHLEVEL__ >= \
   ((maj) << 20) + ((min) << 10) + (patch))  // 检查GNU编译器版本宏定义
#elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#define LLVM_GNUC_PREREQ(maj, min, patch) \
  ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) >= ((maj) << 20) + ((min) << 10))
#else
#define LLVM_GNUC_PREREQ(maj, min, patch) 0  // 默认为0
#endif
#endif

#ifdef _MSC_VER
// Declare these intrinsics manually rather including intrin.h. It's very
// expensive, and MathExtras.h is popular.
// #include <intrin.h>
extern "C" {
unsigned char _BitScanForward(unsigned long* _Index, unsigned long _Mask);  // 声明_BitScanForward函数
unsigned char _BitScanForward64(unsigned long* _Index, unsigned __int64 _Mask);  // 声明_BitScanForward64函数
unsigned char _BitScanReverse(unsigned long* _Index, unsigned long _Mask);  // 声明_BitScanReverse函数
unsigned char _BitScanReverse64(unsigned long* _Index, unsigned __int64 _Mask);  // 声明_BitScanReverse64函数
}
#endif

namespace c10::llvm {
/// 定义操作在输入为0时的行为
enum ZeroBehavior {
  ZB_Undefined,  // 返回值未定义
  ZB_Max,        // 返回值为numeric_limits<T>::max()
  ZB_Width       // 返回值为numeric_limits<T>::digits
};

namespace detail {
/// 模板类，用于计算类型T的尾部零的数量
template <typename T, std::size_t SizeOfT>
struct TrailingZerosCounter {
  /// 计算尾部零的数量
  static std::size_t count(T Val, ZeroBehavior ZB) {
    if (!Val)  // 如果Val为0
      return std::numeric_limits<T>::digits;  // 返回T类型的位数

    if (Val & 0x1)  // 如果Val的最低位为1
      return 0;

    // 二分法计算
    std::size_t ZeroBits = 0;  // 零位的数量
    T Shift = std::numeric_limits<T>::digits >> 1;  // 右移一半位数
    T Mask = std::numeric_limits<T>::max() >> Shift;  // 最大值右移半数位数
    while (Shift) {  // 循环直到Shift为0
      if ((Val & Mask) == 0) {  // 如果Val与Mask按位与结果为0
        Val >>= Shift;  // 右移Val
        ZeroBits |= Shift;  // 或运算
      }
      Shift >>= 1;  // 右移一位
      Mask >>= Shift;  // 右移Shift位
    }
    return ZeroBits;  // 返回零位数量
  }
};

#if (defined(__GNUC__) && __GNUC__ >= 4) || defined(_MSC_VER)
/// 模板特化，用于32位类型T的尾部零的数量计算
template <typename T>
struct TrailingZerosCounter<T, 4> {
  /// 计算尾部零的数量
  static std::size_t count(T Val, ZeroBehavior ZB) {
    if (ZB != ZB_Undefined && Val == 0)
      return 32;  // 返回32
#if __has_builtin(__builtin_ctz) || LLVM_GNUC_PREREQ(4, 0, 0)
    // 如果编译器支持 __builtin_ctz 函数或者是 LLVM 版本大于等于 4.0.0，则使用该函数计算 Val 的末尾零的数量
    return __builtin_ctz(Val);
#elif defined(_MSC_VER)
    // 如果是在 MSC 编译器下，则声明一个 unsigned long 变量 Index
    unsigned long Index;
    // 调用 _BitScanForward 函数查找 Val 的最低位的零的位置，并将结果存入 Index
    _BitScanForward(&Index, Val);
    // 返回 Index，即 Val 的最低位零的位置
    return Index;
#endif
  }
};

#if !defined(_MSC_VER) || defined(_M_X64)
template <typename T>
struct TrailingZerosCounter<T, 8> {
  static std::size_t count(T Val, ZeroBehavior ZB) {
    // 如果 ZB 不是 ZB_Undefined 并且 Val 等于 0，则返回 64
    if (ZB != ZB_Undefined && Val == 0)
      return 64;

#if __has_builtin(__builtin_ctzll) || LLVM_GNUC_PREREQ(4, 0, 0)
    // 如果编译器支持 __builtin_ctzll 函数或者是 LLVM 版本大于等于 4.0.0，则使用该函数计算 Val 的末尾零的数量
    return __builtin_ctzll(Val);
#elif defined(_MSC_VER)
    // 如果是在 MSC 编译器下，则声明一个 unsigned long 变量 Index
    unsigned long Index;
    // 调用 _BitScanForward64 函数查找 Val 的最低位的零的位置，并将结果存入 Index
    _BitScanForward64(&Index, Val);
    // 返回 Index，即 Val 的最低位零的位置
    return Index;
#endif
  }
};
#endif
#endif
} // namespace detail

/// Count number of 0's from the least significant bit to the most
///   stopping at the first 1.
///
/// Only unsigned integral types are allowed.
///
/// \param ZB the behavior on an input of 0. Only ZB_Width and ZB_Undefined are
///   valid arguments.
template <typename T>
std::size_t countTrailingZeros(T Val, ZeroBehavior ZB = ZB_Width) {
  // 静态断言：确保 T 是无符号整数类型
  static_assert(
      std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed,
      "Only unsigned integral types are allowed.");
  // 调用 TrailingZerosCounter 结构体的 count 方法来计算 Val 的末尾零的数量
  return llvm::detail::TrailingZerosCounter<T, sizeof(T)>::count(Val, ZB);
}

namespace detail {
template <typename T, std::size_t SizeOfT>
struct LeadingZerosCounter {
  static std::size_t count(T Val, ZeroBehavior) {
    // 如果 Val 等于 0，则返回 T 类型的位数（全为零）
    if (!Val)
      return std::numeric_limits<T>::digits;

    // 二分法计算最高位零的数量
    std::size_t ZeroBits = 0;
    for (T Shift = std::numeric_limits<T>::digits >> 1; Shift; Shift >>= 1) {
      T Tmp = Val >> Shift;
      if (Tmp)
        Val = Tmp;
      else
        ZeroBits |= Shift;
    }
    return ZeroBits;
  }
};

#if (defined(__GNUC__) && __GNUC__ >= 4) || defined(_MSC_VER)
template <typename T>
struct LeadingZerosCounter<T, 4> {
  static std::size_t count(T Val, ZeroBehavior ZB) {
    // 如果 ZB 不是 ZB_Undefined 并且 Val 等于 0，则返回 32
    if (ZB != ZB_Undefined && Val == 0)
      return 32;

#if __has_builtin(__builtin_clz) || LLVM_GNUC_PREREQ(4, 0, 0)
    // 如果编译器支持 __builtin_clz 函数或者是 LLVM 版本大于等于 4.0.0，则使用该函数计算 Val 的前导零的数量
    return __builtin_clz(Val);
#elif defined(_MSC_VER)
    // 如果是在 MSC 编译器下，则声明一个 unsigned long 变量 Index
    unsigned long Index;
    // 调用 _BitScanReverse 函数查找 Val 的最高位的零的位置，并将结果存入 Index
    _BitScanReverse(&Index, Val);
    // 返回 Index 异或 31，即 Val 的最高位零的位置
    return Index ^ 31;
#endif
  }
};

#if !defined(_MSC_VER) || defined(_M_X64)
template <typename T>
struct LeadingZerosCounter<T, 8> {
  static std::size_t count(T Val, ZeroBehavior ZB) {
    // 如果 ZB 不是 ZB_Undefined 并且 Val 等于 0，则返回 64
    if (ZB != ZB_Undefined && Val == 0)
      return 64;

#if __has_builtin(__builtin_clzll) || LLVM_GNUC_PREREQ(4, 0, 0)
    // 如果编译器支持 __builtin_clzll 函数或者是 LLVM 版本大于等于 4.0.0，则使用该函数计算 Val 的前导零的数量
    return __builtin_clzll(Val);
#elif defined(_MSC_VER)
    // 如果是在 MSC 编译器下，则声明一个 unsigned long 变量 Index
    unsigned long Index;
    // 调用 _BitScanReverse64 函数查找 Val 的最高位的零的位置，并将结果存入 Index
    _BitScanReverse64(&Index, Val);
    // 返回 Index 异或 63，即 Val 的最高位零的位置
    return Index ^ 63;
#endif
  }
};
#endif
#endif
} // namespace detail

/// Count number of 0's from the most significant bit to the least
///   stopping at the first 1.
///
/// Only unsigned integral types are allowed.
///
/// \param ZB the behavior on an input of 0. Only ZB_Width and ZB_Undefined are
///   valid arguments.
template <typename T>
/// Count the number of leading zeros in the integer `Val`.
///
/// This function uses template specialization to ensure that only unsigned
/// integral types are accepted.
///
/// \tparam T The type of the input value `Val`.
/// \param Val The value for which leading zeros are to be counted.
/// \param ZB Specifies the behavior when `Val` is zero, with valid options
///   being `ZB_Width`.
/// \return The count of leading zeros in `Val`.
template <typename T>
std::size_t countLeadingZeros(T Val, ZeroBehavior ZB = ZB_Width) {
  // Ensure the type T is an unsigned integral type.
  static_assert(
      std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed,
      "Only unsigned integral types are allowed.");
  // Delegate the counting of leading zeros to a specialized helper function
  return llvm::detail::LeadingZerosCounter<T, sizeof(T)>::count(Val, ZB);
}

/// Find the index of the first set bit starting from the least significant bit.
///
/// Only unsigned integral types are allowed for `T`.
///
/// \tparam T The type of the input value `Val`.
/// \param Val The value for which to find the first set bit.
/// \param ZB Specifies the behavior when `Val` is zero, with valid options
///   being `ZB_Max` and `ZB_Undefined`.
/// \return The index of the first set bit, or the maximum value of `T` if `Val` is zero and `ZB` is `ZB_Max`.
template <typename T>
T findFirstSet(T Val, ZeroBehavior ZB = ZB_Max) {
  // Return the maximum value of T if Val is zero and ZB is ZB_Max
  if (ZB == ZB_Max && Val == 0)
    return std::numeric_limits<T>::max();

  // Delegate the operation to count trailing zeros, which is suitable for finding
  // the first set bit.
  return countTrailingZeros(Val, ZB_Undefined);
}

/// Create a bitmask with the N right-most bits set to 1, and all other bits set to 0.
///
/// Only unsigned types are allowed for `T`.
///
/// \tparam T The type of the bitmask to create.
/// \param N The number of right-most bits to set to 1.
/// \return The bitmask with the specified number of right-most bits set to 1.
template <typename T>
T maskTrailingOnes(unsigned N) {
  static_assert(std::is_unsigned_v<T>, "Invalid type!");
  const unsigned Bits = CHAR_BIT * sizeof(T);
  // Ensure N is within valid bit index range
  assert(N <= Bits && "Invalid bit index");
  // Calculate and return the bitmask with the specified number of trailing ones
  return N == 0 ? 0 : (T(-1) >> (Bits - N));
}

/// Create a bitmask with the N left-most bits set to 1, and all other bits set to 0.
///
/// Only unsigned types are allowed for `T`.
///
/// \tparam T The type of the bitmask to create.
/// \param N The number of left-most bits to set to 1.
/// \return The bitmask with the specified number of left-most bits set to 1.
template <typename T>
T maskLeadingOnes(unsigned N) {
  // Calculate and return the bitmask with the specified number of leading ones
  return ~maskTrailingOnes<T>(CHAR_BIT * sizeof(T) - N);
}

/// Create a bitmask with the N right-most bits set to 0, and all other bits set to 1.
///
/// Only unsigned types are allowed for `T`.
///
/// \tparam T The type of the bitmask to create.
/// \param N The number of right-most bits to set to 0.
/// \return The bitmask with the specified number of trailing zeros.
template <typename T>
T maskTrailingZeros(unsigned N) {
  // Calculate and return the bitmask with the specified number of trailing zeros
  return maskLeadingOnes<T>(CHAR_BIT * sizeof(T) - N);
}

/// Create a bitmask with the N left-most bits set to 0, and all other bits set to 1.
///
/// Only unsigned types are allowed for `T`.
///
/// \tparam T The type of the bitmask to create.
/// \param N The number of left-most bits to set to 0.
/// \return The bitmask with the specified number of leading zeros.
template <typename T>
T maskLeadingZeros(unsigned N) {
  // Calculate and return the bitmask with the specified number of leading zeros
  return maskTrailingOnes<T>(CHAR_BIT * sizeof(T) - N);
}

/// Get the index of the last set bit starting from the least significant bit.
///
/// Only unsigned integral types are allowed for `T`.
///
/// \tparam T The type of the input value `Val`.
/// \param Val The value for which to find the last set bit.
/// \param ZB Specifies the behavior when `Val` is zero, with valid options
///   being `ZB_Max` and `ZB_Undefined`.
/// \return The index of the last set bit, or the maximum value of `T` minus one.
template <typename T>
T findLastSet(T Val, ZeroBehavior ZB = ZB_Max) {
  // Return the maximum value of T if Val is zero and ZB is ZB_Max
  if (ZB == ZB_Max && Val == 0)
    return std::numeric_limits<T>::max();

  // Use ^ instead of - because both gcc and llvm can remove the associated ^
  // in the __builtin_clz intrinsic on x86.
  // Calculate and return the index of the last set bit
  return countLeadingZeros(Val, ZB_Undefined) ^
      (std::numeric_limits<T>::digits - 1);
}

/// Macro compressed bit reversal table for 256 bits.
///
/// This table is a macro for a precomputed bit reversal table for 256 bits.
/// It is utilized for efficient bit reversal operations.
/// NOLINTNEXTLINE(*c-arrays*)
static constexpr unsigned char BitReverseTable256[256] = {
#define R2(n) n, n + 2 * 64, n + 1 * 64, n + 3 * 64
#define R4(n) R2(n), R2(n + 2 * 16), R2(n + 1 * 16), R2(n + 3 * 16)
#define R6(n) R4(n), R4(n + 2 * 4), R4(n + 1 * 4), R4(n + 3 * 4)
    R6(0),
    R6(2),
    R6(1),
    R6(3)
#undef R2
#undef R4
#undef R6
};

/// Reverse the bits in `Val`.
///
/// \tparam T The type of the input value `Val`.
/// \param Val The value whose bits are to be reversed.
/// \return The value with its bits reversed.
template <typename T>
// 反转给定值的比特位顺序
T reverseBits(T Val) {
  // NOLINTNEXTLINE(*c-arrays*)
  unsigned char in[sizeof(Val)];  // 声明一个数组，用于存储给定值的字节表示
  // NOLINTNEXTLINE(*c-arrays*)
  unsigned char out[sizeof(Val)];  // 声明一个数组，用于存储反转后的字节表示
  std::memcpy(in, &Val, sizeof(Val));  // 将给定值复制到字节数组中
  for (unsigned i = 0; i < sizeof(Val); ++i)
    out[(sizeof(Val) - i) - 1] = BitReverseTable256[in[i]];  // 使用预先定义的表格反转每个字节的比特位顺序
  std::memcpy(&Val, out, sizeof(Val));  // 将反转后的字节数组复制回给定值的内存表示
  return Val;  // 返回反转后的值
}

// NOTE: The following support functions use the _32/_64 extensions instead of
// type overloading so that signed and unsigned integers can be used without
// ambiguity.

/// 返回64位值的高32位。
constexpr inline uint32_t Hi_32(uint64_t Value) {
  return static_cast<uint32_t>(Value >> 32);  // 返回给定64位值的高32位
}

/// 返回64位值的低32位。
constexpr inline uint32_t Lo_32(uint64_t Value) {
  return static_cast<uint32_t>(Value);  // 返回给定64位值的低32位
}

/// 使用高低32位整数构造64位整数。
constexpr inline uint64_t Make_64(uint32_t High, uint32_t Low) {
  return ((uint64_t)High << 32) | (uint64_t)Low;  // 使用给定的高低32位整数构造一个64位整数
}

/// 检查一个整数是否适合于给定的位宽度。
template <unsigned N>
constexpr inline bool isInt(int64_t x) {
  return N >= 64 ||
      (-(INT64_C(1) << (N - 1)) <= x && x < (INT64_C(1) << (N - 1)));  // 检查整数是否在指定的有符号位宽度范围内
}
// Template specializations to get better code for common cases.
template <>
constexpr inline bool isInt<8>(int64_t x) {
  return static_cast<int8_t>(x) == x;  // 特化版本：检查整数是否为8位有符号整数
}
template <>
constexpr inline bool isInt<16>(int64_t x) {
  return static_cast<int16_t>(x) == x;  // 特化版本：检查整数是否为16位有符号整数
}
template <>
constexpr inline bool isInt<32>(int64_t x) {
  return static_cast<int32_t>(x) == x;  // 特化版本：检查整数是否为32位有符号整数
}

/// 检查一个有符号整数是否为通过左移S位得到的N位数。
template <unsigned N, unsigned S>
constexpr inline bool isShiftedInt(int64_t x) {
  static_assert(
      N > 0, "isShiftedInt<0> doesn't make sense (refers to a 0-bit number.");
  static_assert(N + S <= 64, "isShiftedInt<N, S> with N + S > 64 is too wide.");
  return isInt<N + S>(x) && (x % (UINT64_C(1) << S) == 0);  // 检查整数是否是通过左移S位得到的N位有符号整数
}

/// 检查一个无符号整数是否适合于给定的位宽度。
///
/// This is written as two functions rather than as simply
///
///   return N >= 64 || X < (UINT64_C(1) << N);
///
/// to keep MSVC from (incorrectly) warning on isUInt<64> that we're shifting
/// left too many places.
template <unsigned N>
constexpr inline std::enable_if_t<(N < 64), bool> isUInt(uint64_t X) {
  static_assert(N > 0, "isUInt<0> doesn't make sense");
  return X < (UINT64_C(1) << (N));  // 检查无符号整数是否在指定的位宽度范围内
}
template <unsigned N>
constexpr inline std::enable_if_t<N >= 64, bool> isUInt(uint64_t /*X*/) {
  return true;  // 如果位宽度超过64位，则始终返回true
}

// Template specializations to get better code for common cases.
template <>
constexpr inline bool isUInt<8>(uint64_t x) {
  return static_cast<uint8_t>(x) == x;  // 特化版本：检查整数是否为8位无符号整数
}
template <>
constexpr inline bool isUInt<16>(uint64_t x) {
  return static_cast<uint16_t>(x) == x;  // 特化版本：检查整数是否为16位无符号整数
}
template <>
constexpr inline bool isUInt<32>(uint64_t x) {
  return static_cast<uint32_t>(x) == x;  // 特化版本：检查整数是否为32位无符号整数
}

/// 检查一个无符号整数是否为通过左移S位得到的N位数。
/// Check if the unsigned integer x can be represented as a shifted N-bit unsigned integer.
/// This function is constexpr, allowing it to be evaluated at compile-time.
/// It uses static_assert to ensure N > 0 and N + S <= 64 are satisfied for valid shifts.
constexpr inline bool isShiftedUInt(uint64_t x) {
  static_assert(
      N > 0, "isShiftedUInt<0> doesn't make sense (refers to a 0-bit number)");
  static_assert(
      N + S <= 64, "isShiftedUInt<N, S> with N + S > 64 is too wide.");
  // Per the two static_asserts above, S must be strictly less than 64.  So
  // 1 << S is not undefined behavior.
  return isUInt<N + S>(x) && (x % (UINT64_C(1) << S) == 0);
}

/// Gets the maximum value for a N-bit unsigned integer.
/// This function returns UINT64_MAX right shifted by (64 - N) bits.
inline uint64_t maxUIntN(uint64_t N) {
  assert(N > 0 && N <= 64 && "integer width out of range");

  // uint64_t(1) << 64 is undefined behavior, so we can't do
  //   (uint64_t(1) << N) - 1
  // without checking first that N != 64.  But this works and doesn't have a
  // branch.
  return UINT64_MAX >> (64 - N);
}

// Ignore the false warning "Arithmetic overflow" for MSVC
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)
#endif

/// Gets the minimum value for a N-bit signed integer.
/// This function returns -(1 << (N - 1)), ensuring proper two's complement for signed integers.
inline int64_t minIntN(int64_t N) {
  assert(N > 0 && N <= 64 && "integer width out of range");
  // NOLINTNEXTLINE(*-narrowing-conversions)
  return -(UINT64_C(1) << (N - 1));
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

/// Gets the maximum value for a N-bit signed integer.
/// This function returns (1 << (N - 1)) - 1, relying on two's complement wraparound when N == 64.
inline int64_t maxIntN(int64_t N) {
  assert(N > 0 && N <= 64 && "integer width out of range");

  // This relies on two's complement wraparound when N == 64, so we convert to
  // int64_t only at the very end to avoid UB.
  // NOLINTNEXTLINE(*-narrowing-conversions)
  return (UINT64_C(1) << (N - 1)) - 1;
}

/// Checks if an unsigned integer fits into the given (dynamic) bit width N.
/// This function returns true if N is greater than or equal to 64 or if x <= maxUIntN(N).
inline bool isUIntN(unsigned N, uint64_t x) {
  return N >= 64 || x <= maxUIntN(N);
}

/// Checks if a signed integer fits into the given (dynamic) bit width N.
/// This function returns true if N is greater than or equal to 64 or if minIntN(N) <= x <= maxIntN(N).
inline bool isIntN(unsigned N, int64_t x) {
  return N >= 64 || (minIntN(N) <= x && x <= maxIntN(N));
}

/// Return true if the argument Value is a non-empty sequence of ones starting at the
/// least significant bit with the remainder zero (32 bit version).
/// Ex. isMask_32(0x0000FFFFU) == true.
constexpr inline bool isMask_32(uint32_t Value) {
  return Value && ((Value + 1) & Value) == 0;
}

/// Return true if the argument Value is a non-empty sequence of ones starting at the
/// least significant bit with the remainder zero (64 bit version).
constexpr inline bool isMask_64(uint64_t Value) {
  return Value && ((Value + 1) & Value) == 0;
}

/// Return true if the argument Value contains a non-empty sequence of ones with the
/// remainder zero (32 bit version.)
/// Ex. isShiftedMask_32(0x0000FF00U) == true.
constexpr inline bool isShiftedMask_32(uint32_t Value) {
  return Value && isMask_32((Value - 1) | Value);
}

/// Return true if the argument Value contains a non-empty sequence of ones with the
/// remainder zero (64 bit version.)
constexpr inline bool isShiftedMask_64(uint64_t Value) {
  return Value && isMask_64((Value - 1) | Value);
}
/// Return true if the argument is a power of two > 0.
/// Ex. isPowerOf2_32(0x00100000U) == true (32 bit edition.)
constexpr inline bool isPowerOf2_32(uint32_t Value) {
  // Check if Value is non-zero and has only one bit set (using bitwise trick)
  return Value && !(Value & (Value - 1));
}

/// Return true if the argument is a power of two > 0 (64 bit edition.)
constexpr inline bool isPowerOf2_64(uint64_t Value) {
  // Check if Value is non-zero and has only one bit set (using bitwise trick)
  return Value && !(Value & (Value - 1));
}

/// Count the number of ones from the most significant bit to the first
/// zero bit.
///
/// Ex. countLeadingOnes(0xFF0FFF00) == 8.
/// Only unsigned integral types are allowed.
///
/// \param ZB the behavior on an input of all ones. Only ZB_Width and
/// ZB_Undefined are valid arguments.
template <typename T>
std::size_t countLeadingOnes(T Value, ZeroBehavior ZB = ZB_Width) {
  // Ensure T is unsigned integral type
  static_assert(
      std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed,
      "Only unsigned integral types are allowed.");
  // Invert Value and count leading zeros using countLeadingZeros function
  return countLeadingZeros<T>(~Value, ZB);
}

/// Count the number of ones from the least significant bit to the first
/// zero bit.
///
/// Ex. countTrailingOnes(0x00FF00FF) == 8.
/// Only unsigned integral types are allowed.
///
/// \param ZB the behavior on an input of all ones. Only ZB_Width and
/// ZB_Undefined are valid arguments.
template <typename T>
std::size_t countTrailingOnes(T Value, ZeroBehavior ZB = ZB_Width) {
  // Ensure T is unsigned integral type
  static_assert(
      std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed,
      "Only unsigned integral types are allowed.");
  // Invert Value and count trailing zeros using countTrailingZeros function
  return countTrailingZeros<T>(~Value, ZB);
}

namespace detail {
template <typename T, std::size_t SizeOfT>
struct PopulationCounter {
  static unsigned count(T Value) {
    // Check if SizeOfT is within 4 bytes, otherwise raise an error
    static_assert(SizeOfT <= 4, "Not implemented!");
    // Use built-in function __builtin_popcount for GCC
#if defined(__GNUC__) && __GNUC__ >= 4
    return __builtin_popcount(Value);
#else
    // Fallback implementation for non-GCC compilers
    uint32_t v = Value;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
  }
};

template <typename T>
struct PopulationCounter<T, 8> {
  static unsigned count(T Value) {
    // Use built-in function __builtin_popcountll for GCC
#if defined(__GNUC__) && __GNUC__ >= 4
    return __builtin_popcountll(Value);
#else
    // Fallback implementation for non-GCC compilers
    uint64_t v = Value;
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return unsigned((uint64_t)(v * 0x0101010101010101ULL) >> 56);
#endif
  }
};
} // namespace detail

/// Count the number of set bits in a value.
/// Ex. countPopulation(0xF000F000) = 8
/// Returns 0 if the word is zero.
template <typename T>
inline unsigned countPopulation(T Value) {
  // Ensure T is unsigned integral type
  static_assert(
      std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed,
      "Only unsigned integral types are allowed.");
  // Call specialized PopulationCounter based on sizeof(T)
  return detail::PopulationCounter<T, sizeof(T)>::count(Value);
}

/// Return the log base 2 of the specified value.
inline double Log2(double Value) {
#if defined(__ANDROID_API__) && __ANDROID_API__ < 18
  // 如果目标平台是 Android 且 API 版本低于 18，使用内置函数计算以 2 为底的对数
  return __builtin_log(Value) / __builtin_log(2.0);
#else
  // 否则，调用标准库函数计算以 2 为底的对数
  return log2(Value);
#endif
}

/// Return the floor log base 2 of the specified value, -1 if the value is zero.
/// (32 bit edition.)
/// Ex. Log2_32(32) == 5, Log2_32(1) == 0, Log2_32(0) == -1, Log2_32(6) == 2
inline unsigned Log2_32(uint32_t Value) {
  // 使用位操作和计数前导零的函数，返回指定值的以 2 为底的对数向下取整
  return static_cast<unsigned>(31 - countLeadingZeros(Value));
}

/// Return the floor log base 2 of the specified value, -1 if the value is zero.
/// (64 bit edition.)
inline unsigned Log2_64(uint64_t Value) {
  // 使用位操作和计数前导零的函数，返回指定值的以 2 为底的对数向下取整
  return static_cast<unsigned>(63 - countLeadingZeros(Value));
}

/// Return the ceil log base 2 of the specified value, 32 if the value is zero.
/// (32 bit edition).
/// Ex. Log2_32_Ceil(32) == 5, Log2_32_Ceil(1) == 0, Log2_32_Ceil(6) == 3
inline unsigned Log2_32_Ceil(uint32_t Value) {
  // 使用位操作和计数前导零的函数，返回指定值的以 2 为底的对数向上取整
  return static_cast<unsigned>(32 - countLeadingZeros(Value - 1));
}

/// Return the ceil log base 2 of the specified value, 64 if the value is zero.
/// (64 bit edition.)
inline unsigned Log2_64_Ceil(uint64_t Value) {
  // 使用位操作和计数前导零的函数，返回指定值的以 2 为底的对数向上取整
  return static_cast<unsigned>(64 - countLeadingZeros(Value - 1));
}

/// Return the greatest common divisor of the values using Euclid's algorithm.
inline uint64_t GreatestCommonDivisor64(uint64_t A, uint64_t B) {
  // 使用欧几里得算法计算两个数的最大公约数
  while (B) {
    uint64_t T = B;
    B = A % B;
    A = T;
  }
  return A;
}

/// This function takes a 64-bit integer and returns the bit equivalent double.
inline double BitsToDouble(uint64_t Bits) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double D;
  // 使用内存拷贝将 64 位整数的位表示转换为双精度浮点数
  static_assert(sizeof(uint64_t) == sizeof(double), "Unexpected type sizes");
  memcpy(&D, &Bits, sizeof(Bits));
  return D;
}

/// This function takes a 32-bit integer and returns the bit equivalent float.
inline float BitsToFloat(uint32_t Bits) {
  // TODO: Use std::bit_cast once C++20 becomes available.
  // 使用位转换函数将 32 位整数的位表示转换为单精度浮点数
  return c10::bit_cast<float>(Bits);
}

/// This function takes a double and returns the bit equivalent 64-bit integer.
/// Note that copying doubles around changes the bits of NaNs on some hosts,
/// notably x86, so this routine cannot be used if these bits are needed.
inline uint64_t DoubleToBits(double Double) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint64_t Bits;
  // 使用内存拷贝将双精度浮点数的位表示转换为 64 位整数
  static_assert(sizeof(uint64_t) == sizeof(double), "Unexpected type sizes");
  memcpy(&Bits, &Double, sizeof(Double));
  return Bits;
}

/// This function takes a float and returns the bit equivalent 32-bit integer.
/// Note that copying floats around changes the bits of NaNs on some hosts,
/// notably x86, so this routine cannot be used if these bits are needed.
inline uint32_t FloatToBits(float Float) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t Bits;
  // 使用内存拷贝将单精度浮点数的位表示转换为 32 位整数
  static_assert(sizeof(uint32_t) == sizeof(float), "Unexpected type sizes");
  memcpy(&Bits, &Float, sizeof(Float));
  return Bits;
}

/// A and B are either alignments or offsets. Return the minimum alignment that
/// Calculates the minimum alignment that ensures both A and B are properly aligned.
constexpr inline uint64_t MinAlign(uint64_t A, uint64_t B) {
  // 返回能够同时对齐 A 和 B 的最大的2的幂次方。
  // 将以下注释代码中的 "-Value" 替换为 "1+~Value"，以避免 MSVC 警告 C4146
  //    return (A | B) & -(A | B);
  return (A | B) & (1 + ~(A | B));
}

/// 将 Addr 地址对齐到 Alignment 字节，向上舍入。
///
/// Alignment 应为2的幂。此方法向上舍入，因此 alignAddr(7, 4) == 8 和 alignAddr(8, 4) == 8。
inline uintptr_t alignAddr(const void* Addr, size_t Alignment) {
  assert(
      Alignment && isPowerOf2_64((uint64_t)Alignment) &&
      "Alignment is not a power of two!");
  
  // 确保地址增加 Alignment - 1 后不会溢出
  assert((uintptr_t)Addr + Alignment - 1 >= (uintptr_t)Addr);

  // 返回将 Addr 地址向上舍入到 Alignment 的倍数后的地址
  return (((uintptr_t)Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1));
}

/// 返回将 Ptr 对齐到 Alignment 字节所需的调整量，向上舍入。
inline size_t alignmentAdjustment(const void* Ptr, size_t Alignment) {
  // 调用 alignAddr 函数并计算地址对齐后的调整量
  return alignAddr(Ptr, Alignment) - (uintptr_t)Ptr;
}

/// 返回严格大于 A 的下一个2的幂（64位）。
/// 溢出时返回零。
inline uint64_t NextPowerOf2(uint64_t A) {
  A |= (A >> 1);
  A |= (A >> 2);
  A |= (A >> 4);
  A |= (A >> 8);
  A |= (A >> 16);
  A |= (A >> 32);
  return A + 1;
}

/// 返回不大于给定值 A 的2的幂。
/// 实质上是对2的幂的域执行 floor 操作。
inline uint64_t PowerOf2Floor(uint64_t A) {
  if (!A)
    return 0;
  return 1ull << (63 - countLeadingZeros(A, ZB_Undefined));
}

/// 返回不小于给定值 A 的2的幂。
/// 实质上是对2的幂的域执行 ceil 操作。
inline uint64_t PowerOf2Ceil(uint64_t A) {
  if (!A)
    return 0;
  return NextPowerOf2(A - 1);
}

/// 返回大于或等于 \p Value 并且是 \p Align 的倍数的下一个整数（模2**64）。
///
/// 如果指定了非零 \p Skew，则返回值将是大于或等于 \p Value 并且等于 \p Align * N + \p Skew 的最小整数 N。
/// 如果 \p Skew 大于 \p Align，则其值调整为 '\p Skew mod \p Align'。
///
/// 示例：
/// \code
///   alignTo(5, 8) = 8
///   alignTo(17, 8) = 24
///   alignTo(~0LL, 8) = 0
///   alignTo(321, 255) = 510
///
///   alignTo(5, 8, 7) = 7
///   alignTo(17, 8, 1) = 17
///   alignTo(~0LL, 8, 3) = 3
///   alignTo(321, 255, 42) = 552
/// \endcode
inline uint64_t alignTo(uint64_t Value, uint64_t Align, uint64_t Skew = 0) {
  assert(Align != 0u && "Align can't be 0.");
  Skew %= Align;
  // 返回大于或等于 Value 的下一个 Align 的倍数
  return (Value + Align - 1 - Skew) / Align * Align + Skew;
}

/// 返回大于或等于 \p Value 并且是非零 \c Align 的倍数的下一个整数（模2**64）。
template <uint64_t Align>
/// Computes the aligned value of `Value` based on the template parameter `Align`.
/// Throws a static assertion error if `Align` is zero.
constexpr inline uint64_t alignTo(uint64_t Value) {
  static_assert(Align != 0u, "Align must be non-zero");
  return (Value + Align - 1) / Align * Align;
}

/// Returns the integer ceiling of the division `Numerator / Denominator`.
inline uint64_t divideCeil(uint64_t Numerator, uint64_t Denominator) {
  return alignTo(Numerator, Denominator) / Denominator;
}

/// Template struct for computing aligned values where `Align` is a template parameter.
/// Throws a static assertion error if `Align` is zero.
template <uint64_t Align>
struct AlignTo {
  static_assert(Align != 0u, "Align must be non-zero");
  template <uint64_t Value>
  struct from_value {
    static const uint64_t value = (Value + Align - 1) / Align * Align;
  };
};

/// Computes the largest integer less than or equal to `Value` that is aligned to `Align`
/// with an optional `Skew`. Throws an assertion error if `Align` is zero.
inline uint64_t alignDown(uint64_t Value, uint64_t Align, uint64_t Skew = 0) {
  assert(Align != 0u && "Align can't be 0.");
  Skew %= Align;
  return (Value - Skew) / Align * Align + Skew;
}

/// Computes the offset needed to align `Value` to the next multiple of `Align`.
/// Throws an assertion error if `Align` is zero.
inline uint64_t OffsetToAlignment(uint64_t Value, uint64_t Align) {
  return alignTo(Value, Align) - Value;
}

/// Sign-extends the bottom `B` bits of `X` to a 32-bit integer.
/// Requires that 0 < B <= 32.
template <unsigned B>
constexpr inline int32_t SignExtend32(uint32_t X) {
  static_assert(B > 0, "Bit width can't be 0.");
  static_assert(B <= 32, "Bit width out of range.");
  return int32_t(X << (32 - B)) >> (32 - B);
}

/// Sign-extends the bottom `B` bits of `X` to a 32-bit integer.
/// Requires that 0 < B <= 32.
inline int32_t SignExtend32(uint32_t X, unsigned B) {
  assert(B > 0 && "Bit width can't be 0.");
  assert(B <= 32 && "Bit width out of range.");
  return int32_t(X << (32 - B)) >> (32 - B);
}

/// Sign-extends the bottom `B` bits of `X` to a 64-bit integer.
/// Requires that 0 < B <= 64.
template <unsigned B>
constexpr inline int64_t SignExtend64(uint64_t x) {
  static_assert(B > 0, "Bit width can't be 0.");
  static_assert(B <= 64, "Bit width out of range.");
  return int64_t(x << (64 - B)) >> (64 - B);
}

/// Sign-extends the bottom `B` bits of `X` to a 64-bit integer.
/// Requires that 0 < B <= 64.
inline int64_t SignExtend64(uint64_t X, unsigned B) {
  assert(B > 0 && "Bit width can't be 0.");
  assert(B <= 64 && "Bit width out of range.");
  return int64_t(X << (64 - B)) >> (64 - B);
}

/// Computes the absolute difference between two unsigned integers `X` and `Y` of type `T`.
/// Returns the result of `std::max(X, Y) - std::min(X, Y)`.
template <typename T>
std::enable_if_t<std::is_unsigned_v<T>, T> AbsoluteDifference(T X, T Y) {
  return std::max(X, Y) - std::min(X, Y);
}
/// Perform a saturating addition of two unsigned integers, X and Y, of type T.
/// If ResultOverflowed is provided, set it to true if overflow occurs.
template <typename T>
std::enable_if_t<std::is_unsigned_v<T>, T> SaturatingAdd(
    T X,
    T Y,
    bool* ResultOverflowed = nullptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool Dummy;
  bool& Overflowed = ResultOverflowed ? *ResultOverflowed : Dummy;
  
  // Calculate the sum of X and Y
  T Z = X + Y;
  
  // Determine if overflow occurred by comparing Z to X and Y
  Overflowed = (Z < X || Z < Y);
  
  // If overflow occurred, return the maximum value representable by type T
  if (Overflowed)
    return std::numeric_limits<T>::max();
  else
    return Z;
}

/// Perform a saturating multiplication of two unsigned integers, X and Y, of type T.
/// If ResultOverflowed is provided, set it to true if overflow occurs.
template <typename T>
std::enable_if_t<std::is_unsigned_v<T>, T> SaturatingMultiply(
    T X,
    T Y,
    bool* ResultOverflowed = nullptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool Dummy;
  bool& Overflowed = ResultOverflowed ? *ResultOverflowed : Dummy;

  // Initialize Overflowed to false
  Overflowed = false;

  // Calculate Log2(Z) where Z = X * Y
  int Log2Z = Log2_64(X) + Log2_64(Y);
  const T Max = std::numeric_limits<T>::max();
  int Log2Max = Log2_64(Max);

  // Check if Log2(Z) is less than Log2Max
  if (Log2Z < Log2Max) {
    return X * Y;
  }

  // Check if Log2(Z) is greater than Log2Max, indicating overflow
  if (Log2Z > Log2Max) {
    Overflowed = true;
    return Max;
  }

  // Perform multiplication with consideration for potential overflow
  T Z = (X >> 1) * Y;
  if (Z & ~(Max >> 1)) {
    Overflowed = true;
    return Max;
  }
  Z <<= 1;
  if (X & 1)
    return SaturatingAdd(Z, Y, ResultOverflowed);

  return Z;
}

/// Perform a saturating multiply-add operation with unsigned integers X, Y, and A of type T.
/// If ResultOverflowed is provided, set it to true if overflow occurs.
template <typename T>
std::enable_if_t<std::is_unsigned_v<T>, T> SaturatingMultiplyAdd(
    T X,
    T Y,
    T A,
    bool* ResultOverflowed = nullptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool Dummy;
  bool& Overflowed = ResultOverflowed ? *ResultOverflowed : Dummy;

  // Perform saturating multiplication of X and Y
  T Product = SaturatingMultiply(X, Y, &Overflowed);
  if (Overflowed)
    return Product;

  // Perform saturating addition of A and the product X * Y
  return SaturatingAdd(A, Product, &Overflowed);
}
```