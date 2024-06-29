# `D:\src\scipysrc\numpy\numpy\_core\src\common\utils.hpp`

```py
#ifndef NUMPY_CORE_SRC_COMMON_UTILS_HPP
#define NUMPY_CORE_SRC_COMMON_UTILS_HPP

#include "npdef.hpp"

#if NP_HAS_CPP20
    #include <bit>
#endif

#include <type_traits>
#include <string.h>
#include <cstdint>
#include <cassert>

namespace np {

using std::uint32_t;
using std::uint64_t;

/** Create a value of type `To` from the bits of `from`.
 *
 * similar to `std::bit_cast` but compatible with C++17,
 * should perform similar to `*reinterpret_cast<To*>(&from)`
 * or through punning without expecting any undefined behaviors.
 */
template<typename To, typename From>
#if NP_HAS_BUILTIN(__builtin_bit_cast) || NP_HAS_CPP20
[[nodiscard]] constexpr
#else
inline
#endif
To BitCast(const From &from) noexcept
{
    // 静态断言：确保目标类型和源类型具有相同的大小
    static_assert(
        sizeof(To) == sizeof(From),
        "both data types must have the same size");

    // 静态断言：确保目标类型和源类型都是平凡可复制的
    static_assert(
        std::is_trivially_copyable_v<To> &&
        std::is_trivially_copyable_v<From>,
        "both data types must be trivially copyable");

    // 根据编译器支持情况选择实现方式
#if NP_HAS_CPP20
    return std::bit_cast<To>(from);
#elif NP_HAS_BUILTIN(__builtin_bit_cast)
    return __builtin_bit_cast(To, from);
#else
    // 使用 memcpy 进行位拷贝
    To to;
    memcpy(&to, &from, sizeof(from));
    return to;
#endif
}

/// Bit-scan reverse for non-zeros.
/// Returns the index of the highest set bit. Equivalent to floor(log2(a))
template <typename T>
inline int BitScanReverse(uint32_t a)
{
    // 根据编译器支持情况选择实现方式
#if NP_HAS_CPP20
    return std::countl_one(a);
#else
    // 如果 a 为 0，返回 0
    if (a == 0) {
        // 由于使用了 __builtin_clz，可能导致未定义行为
        return 0;
    }
    int r;
    // 根据编译器选择特定的实现方式
    #ifdef _MSC_VER
    unsigned long rl;
    (void)_BitScanReverse(&rl, (unsigned long)a);
    r = static_cast<int>(rl);
    #elif (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)) \
        &&  (defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64))
    __asm__("bsr %1, %0" : "=r" (r) : "r"(a));
    #elif defined(__GNUC__) || defined(__clang__)
    r = 31 - __builtin_clz(a); // 执行于 arm -> clz, ppc -> cntlzw
    #else
    r = 0;
    while (a >>= 1) {
        r++;
    }
    #endif
    return r;
#endif
}

/// Bit-scan reverse for non-zeros.
/// Returns the index of the highest set bit. Equivalent to floor(log2(a))
inline int BitScanReverse(uint64_t a)
{
    // 根据编译器支持情况选择实现方式
#if NP_HAS_CPP20
    return std::countl_one(a);
#else
    // 如果 a 为 0，返回 0
    if (a == 0) {
        // 由于使用了 __builtin_clzll，可能导致未定义行为
        return 0;
    }
    // 根据平台和编译器选择特定的实现方式
    #if defined(_M_AMD64) && defined(_MSC_VER)
    unsigned long rl;
    (void)_BitScanReverse64(&rl, a);
    return static_cast<int>(rl);
    #elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER))
    uint64_t r;
    __asm__("bsrq %1, %0" : "=r"(r) : "r"(a));
    return static_cast<int>(r);
    #elif defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(a);
    #else
    uint64_t a_hi = a >> 32;
    if (a_hi == 0) {
        return BitScanReverse(static_cast<uint32_t>(a));
    }
    return 32 + BitScanReverse(static_cast<uint32_t>(a_hi));
    #endif
#endif
}
    #endif
#endif

这行代码是预处理器指令，用于结束一个条件编译指令块。在这里，`#endif` 结束了之前由 `#ifndef` 或 `#if` 开始的条件编译块。


}

这是 C++ 中的闭合花括号 `}`，用于结束 `namespace np` 命名空间的定义。


} // namespace np

这行代码注释结束了 `namespace np` 命名空间的定义，提供了命名空间结束的说明性注释。


#endif // NUMPY_CORE_SRC_COMMON_UTILS_HPP

这是条件编译指令的结尾，`#endif` 用于结束条件编译块，后面的 `// NUMPY_CORE_SRC_COMMON_UTILS_HPP` 是注释，指出这个 `#endif` 是为了匹配开始于 `NUMPY_CORE_SRC_COMMON_UTILS_HPP` 的条件编译指令。
```