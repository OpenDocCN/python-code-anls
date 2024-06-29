# `.\numpy\numpy\_core\include\numpy\libdivide\libdivide.h`

```
// libdivide.h - 优化的整数除法
// https://libdivide.com
//
// 版权所有 (C) 2010 - 2019 ridiculous_fish, <libdivide@ridiculousfish.com>
// 版权所有 (C) 2016 - 2019 Kim Walisch, <kim.walisch@gmail.com>
//
// libdivide 根据 Boost 或 zlib 许可双重许可
// 您可以根据这两种许可的条款使用 libdivide。
// 更多详情请参阅 LICENSE.txt。

#ifndef NUMPY_CORE_INCLUDE_NUMPY_LIBDIVIDE_LIBDIVIDE_H_
#define NUMPY_CORE_INCLUDE_NUMPY_LIBDIVIDE_LIBDIVIDE_H_

#define LIBDIVIDE_VERSION "3.0"        // 定义 libdivide 的版本号字符串
#define LIBDIVIDE_VERSION_MAJOR 3      // 定义 libdivide 的主版本号
#define LIBDIVIDE_VERSION_MINOR 0      // 定义 libdivide 的次版本号

#include <stdint.h>                    // 包含 C 标准库的头文件，用于整数类型

#if defined(__cplusplus)
    #include <cstdlib>                 // 如果是 C++ 环境，包含 C++ 标准库的头文件
    #include <cstdio>                  // 包含 C 标准输入输出库的头文件
    #include <type_traits>             // 包含类型特性库的头文件
#else
    #include <stdlib.h>                // 如果是 C 环境，包含 C 标准库的头文件
    #include <stdio.h>                 // 包含 C 标准输入输出库的头文件
#endif

#if defined(LIBDIVIDE_AVX512)
    #include <immintrin.h>             // 如果启用 AVX512，包含 AVX512 指令集头文件
#elif defined(LIBDIVIDE_AVX2)
    #include <immintrin.h>             // 如果启用 AVX2，包含 AVX2 指令集头文件
#elif defined(LIBDIVIDE_SSE2)
    #include <emmintrin.h>             // 如果启用 SSE2，包含 SSE2 指令集头文件
#endif

#if defined(_MSC_VER)
    #include <intrin.h>                // 如果是 Microsoft Visual C++ 编译器，包含内部函数头文件
    // 禁用警告 C4146: 对无符号类型应用一元减号运算符，结果仍然是无符号数
    #pragma warning(disable: 4146)
    #define LIBDIVIDE_VC               // 定义标记 LIBDIVIDE_VC，表示在 VC 编译器下编译
#endif

#if !defined(__has_builtin)
    #define __has_builtin(x) 0         // 如果没有定义 __has_builtin 宏，则定义为 0
#endif

#if defined(__SIZEOF_INT128__)
    #define HAS_INT128_T               // 如果支持 __int128_t 类型，定义 HAS_INT128_T
    // clang-cl 在 Windows 上尚不支持 128 位整数除法
    #if !(defined(__clang__) && defined(LIBDIVIDE_VC))
        #define HAS_INT128_DIV         // 如果不是 clang-cl 并且不是 VC 编译器，定义 HAS_INT128_DIV
    #endif
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define LIBDIVIDE_X86_64           // 如果是 x86_64 架构，定义 LIBDIVIDE_X86_64
#endif

#if defined(__i386__)
    #define LIBDIVIDE_i386             // 如果是 i386 架构，定义 LIBDIVIDE_i386
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define LIBDIVIDE_GCC_STYLE_ASM    // 如果是 GCC 或 Clang 编译器，定义 LIBDIVIDE_GCC_STYLE_ASM
#endif

#if defined(__cplusplus) || defined(LIBDIVIDE_VC)
    #define LIBDIVIDE_FUNCTION __FUNCTION__  // 如果是 C++ 或 VC 编译器，使用 __FUNCTION__
#else
    #define LIBDIVIDE_FUNCTION __func__      // 否则使用 __func__
#endif

#define LIBDIVIDE_ERROR(msg) \
    do { \
        fprintf(stderr, "libdivide.h:%d: %s(): Error: %s\n", \
            __LINE__, LIBDIVIDE_FUNCTION, msg); \
        abort(); \
    } while (0)                         // 定义宏 LIBDIVIDE_ERROR，输出错误信息并中止程序

#if defined(LIBDIVIDE_ASSERTIONS_ON)
    #define LIBDIVIDE_ASSERT(x) \
        do { \
            if (!(x)) { \
                fprintf(stderr, "libdivide.h:%d: %s(): Assertion failed: %s\n", \
                    __LINE__, LIBDIVIDE_FUNCTION, #x); \
                abort(); \
            } \
        } while (0)                     // 如果定义了 LIBDIVIDE_ASSERTIONS_ON，定义宏 LIBDIVIDE_ASSERT，用于断言检查
#else
    #define LIBDIVIDE_ASSERT(x)          // 否则定义为空
#endif

#ifdef __cplusplus
namespace libdivide {
#endif

// 为防止编译器填充，对分隔符结构体进行打包。
// 当使用大量 libdivide 分隔符数组时，这将减少内存使用量高达 43%，
// 并通过减少内存带宽提高最多 10% 的性能。
#pragma pack(push, 1)

struct libdivide_u32_t {
    uint32_t magic;     // 魔数，用于快速整数除法
    uint8_t more;       // 其他数据
};

struct libdivide_s32_t {
    int32_t magic;      // 魔数，用于快速整数除法
    uint8_t more;       // 其他数据
};

struct libdivide_u64_t {
    uint64_t magic;     // 魔数，用于快速整数除法
    uint8_t more;       // 其他数据
};

struct libdivide_s64_t {
    int64_t magic;      // 魔数，用于快速整数除法
    uint8_t more;       // 其他数据
};

struct libdivide_u32_branchfree_t {
    // 定义一个无符号32位整数变量magic，用于存储魔数或特定标识符
    uint32_t magic;
    // 定义一个无符号8位整数变量more，通常用于表示额外的标志或状态信息
    uint8_t more;
};

// 结构体定义，用于存储带有分支优化的32位有符号整数除法信息
struct libdivide_s32_branchfree_t {
    int32_t magic; // 魔数，用于除法优化
    uint8_t more; // 更多信息，用于指示除法的具体实现方式
};

// 结构体定义，用于存储带有分支优化的64位无符号整数除法信息
struct libdivide_u64_branchfree_t {
    uint64_t magic; // 魔数，用于除法优化
    uint8_t more; // 更多信息，用于指示除法的具体实现方式
};

// 结构体定义，用于存储带有分支优化的64位有符号整数除法信息
struct libdivide_s64_branchfree_t {
    int64_t magic; // 魔数，用于除法优化
    uint8_t more; // 更多信息，用于指示除法的具体实现方式
};

#pragma pack(pop)

// "more"字段的解释：
//
// * 位 0-5 是移位值（用于移位路径或乘法路径）。
// * 位 6 是乘法路径的加法指示器。
// * 位 7 如果被设置表示除数为负数。我们使用位 7 作为负除数指示器，
//   这样我们可以有效地使用符号扩展来创建一个所有位均设置为 1 的位掩码
//   （如果除数为负数），或者为 0（如果除数为正数）。
//
// u32: [0-4] 移位值
//      [5] 忽略
//      [6] 加法指示器
//      魔数为 0 表示移位路径
//
// s32: [0-4] 移位值
//      [5] 忽略
//      [6] 加法指示器
//      [7] 表示负除数
//      魔数为 0 表示移位路径
//
// u64: [0-5] 移位值
//      [6] 加法指示器
//      魔数为 0 表示移位路径
//
// s64: [0-5] 移位值
//      [6] 加法指示器
//      [7] 表示负除数
//      魔数为 0 表示移位路径
//
// 在 s32 和 s64 分支优化模式下，根据除数是否为负数，魔数会被取反。
// 在分支优化策略中，不对魔数进行取反。

// 枚举常量定义
enum {
    LIBDIVIDE_32_SHIFT_MASK = 0x1F, // 32位移位掩码
    LIBDIVIDE_64_SHIFT_MASK = 0x3F, // 64位移位掩码
    LIBDIVIDE_ADD_MARKER = 0x40,    // 加法指示器
    LIBDIVIDE_NEGATIVE_DIVISOR = 0x80 // 负除数指示器
};

// 静态内联函数声明，用于生成带有分支优化的32位有符号整数除法信息
static inline struct libdivide_s32_t libdivide_s32_gen(int32_t d);

// 静态内联函数声明，用于生成带有分支优化的32位无符号整数除法信息
static inline struct libdivide_u32_t libdivide_u32_gen(uint32_t d);

// 静态内联函数声明，用于生成带有分支优化的64位有符号整数除法信息
static inline struct libdivide_s64_t libdivide_s64_gen(int64_t d);

// 静态内联函数声明，用于生成带有分支优化的64位无符号整数除法信息
static inline struct libdivide_u64_t libdivide_u64_gen(uint64_t d);

// 静态内联函数声明，用于执行带有分支优化的32位有符号整数除法
static inline int32_t  libdivide_s32_do(int32_t numer, const struct libdivide_s32_t *denom);

// 静态内联函数声明，用于执行带有分支优化的32位无符号整数除法
static inline uint32_t libdivide_u32_do(uint32_t numer, const struct libdivide_u32_t *denom);

// 静态内联函数声明，用于执行带有分支优化的64位有符号整数除法
static inline int64_t  libdivide_s64_do(int64_t numer, const struct libdivide_s64_t *denom);

// 静态内联函数声明，用于执行带有分支优化的64位无符号整数除法
static inline uint64_t libdivide_u64_do(uint64_t numer, const struct libdivide_u64_t *denom);

// 静态内联函数声明，用于执行带有分支优化的32位有符号整数除法（分支优化模式）
static inline int32_t  libdivide_s32_branchfree_do(int32_t numer, const struct libdivide_s32_branchfree_t *denom);

// 静态内联函数声明，用于执行带有分支优化的32位无符号整数除法（分支优化模式）
static inline uint32_t libdivide_u32_branchfree_do(uint32_t numer, const struct libdivide_u32_branchfree_t *denom);

// 静态内联函数声明，用于执行带有分支优化的64位有符号整数除法（分支优化模式）
static inline int64_t  libdivide_s64_branchfree_do(int64_t numer, const struct libdivide_s64_branchfree_t *denom);

// 静态内联函数声明，用于执行带有分支优化的64位无符号整数除法（分支优化模式）
static inline uint64_t libdivide_u64_branchfree_do(uint64_t numer, const struct libdivide_u64_branchfree_t *denom);
// 恢复 libdivide_s32_t 结构体指针的实际值，用于除法运算
static inline int32_t  libdivide_s32_recover(const struct libdivide_s32_t *denom);
// 恢复 libdivide_u32_t 结构体指针的实际值，用于无符号整数除法运算
static inline uint32_t libdivide_u32_recover(const struct libdivide_u32_t *denom);
// 恢复 libdivide_s64_t 结构体指针的实际值，用于64位有符号整数除法运算
static inline int64_t  libdivide_s64_recover(const struct libdivide_s64_t *denom);
// 恢复 libdivide_u64_t 结构体指针的实际值，用于64位无符号整数除法运算
static inline uint64_t libdivide_u64_recover(const struct libdivide_u64_t *denom);

// 恢复 libdivide_s32_branchfree_t 结构体指针的实际值，用于无分支的32位有符号整数除法运算
static inline int32_t  libdivide_s32_branchfree_recover(const struct libdivide_s32_branchfree_t *denom);
// 恢复 libdivide_u32_branchfree_t 结构体指针的实际值，用于无分支的32位无符号整数除法运算
static inline uint32_t libdivide_u32_branchfree_recover(const struct libdivide_u32_branchfree_t *denom);
// 恢复 libdivide_s64_branchfree_t 结构体指针的实际值，用于无分支的64位有符号整数除法运算
static inline int64_t  libdivide_s64_branchfree_recover(const struct libdivide_s64_branchfree_t *denom);
// 恢复 libdivide_u64_branchfree_t 结构体指针的实际值，用于无分支的64位无符号整数除法运算
static inline uint64_t libdivide_u64_branchfree_recover(const struct libdivide_u64_branchfree_t *denom);

//////// Internal Utility Functions

// 32位无符号整数乘法的高32位计算
static inline uint32_t libdivide_mullhi_u32(uint32_t x, uint32_t y) {
    uint64_t xl = x, yl = y; // 将参数转换为64位整数以避免溢出
    uint64_t rl = xl * yl; // 计算乘积
    return (uint32_t)(rl >> 32); // 返回高32位结果
}

// 32位有符号整数乘法的高32位计算
static inline int32_t libdivide_mullhi_s32(int32_t x, int32_t y) {
    int64_t xl = x, yl = y; // 将参数转换为64位整数以避免溢出
    int64_t rl = xl * yl; // 计算乘积
    // 需要算术右移来处理符号
    return (int32_t)(rl >> 32); // 返回高32位结果
}

// 64位无符号整数乘法的高64位计算
static inline uint64_t libdivide_mullhi_u64(uint64_t x, uint64_t y) {
#if defined(LIBDIVIDE_VC) && \
    defined(LIBDIVIDE_X86_64)
    return __umulh(x, y); // 使用硬件提供的无符号64位乘法高64位计算
#elif defined(HAS_INT128_T)
    __uint128_t xl = x, yl = y; // 使用128位整数类型计算
    __uint128_t rl = xl * yl; // 计算乘积
    return (uint64_t)(rl >> 64); // 返回高64位结果
#else
    // 使用32位乘法计算完整的128位乘积，以处理平台不支持128位整数的情况
    uint32_t mask = 0xFFFFFFFF;
    uint32_t x0 = (uint32_t)(x & mask);
    uint32_t x1 = (uint32_t)(x >> 32);
    uint32_t y0 = (uint32_t)(y & mask);
    uint32_t y1 = (uint32_t)(y >> 32);
    uint32_t x0y0_hi = libdivide_mullhi_u32(x0, y0); // 计算低位乘积的高32位
    uint64_t x0y1 = x0 * (uint64_t)y1;
    uint64_t x1y0 = x1 * (uint64_t)y0;
    uint64_t x1y1 = x1 * (uint64_t)y1;
    uint64_t temp = x1y0 + x0y0_hi;
    uint64_t temp_lo = temp & mask;
    uint64_t temp_hi = temp >> 32;

    return x1y1 + temp_hi + ((temp_lo + x0y1) >> 32); // 计算最终的高64位结果
#endif
}

// 64位有符号整数乘法的高64位计算
static inline int64_t libdivide_mullhi_s64(int64_t x, int64_t y) {
#if defined(LIBDIVIDE_VC) && \
    defined(LIBDIVIDE_X86_64)
    return __mulh(x, y); // 使用硬件提供的有符号64位乘法高64位计算
#elif defined(HAS_INT128_T)
    __int128_t xl = x, yl = y; // 使用128位整数类型计算
    __int128_t rl = xl * yl; // 计算乘积
    return (int64_t)(rl >> 64); // 返回高64位结果
#else
    // 使用32位乘法计算完整的128位乘积，以处理平台不支持128位整数的情况
    uint32_t mask = 0xFFFFFFFF;
    uint32_t x0 = (uint32_t)(x & mask);
    uint32_t y0 = (uint32_t)(y & mask);
    int32_t x1 = (int32_t)(x >> 32);
    int32_t y1 = (int32_t)(y >> 32);
    uint32_t x0y0_hi = libdivide_mullhi_u32(x0, y0); // 计算低位乘积的高32位
    int64_t t = x1 * (int64_t)y0 + x0y0_hi;
    int64_t w1 = x0 * (int64_t)y1 + (t & mask);

    return x1 * (int64_t)y1 + (t >> 32) + (w1 >> 32); // 计算最终的高64位结果
#endif
}

// 计算32位无符号整数的前导零数
static inline int32_t libdivide_count_leading_zeros32(uint32_t val) {
#if defined(__GNUC__) || \
    __has_builtin(__builtin_clz)
    // 使用快速计算前导零数的内建函数
    # 调用内建函数 __builtin_clz() 来计算参数 val 的前导零的数量，并将结果返回
    return __builtin_clz(val);
#elif defined(LIBDIVIDE_VC)
    // 如果使用的是 Visual C++ 编译器
    unsigned long result;
    // 使用 _BitScanReverse 函数查找 val 的最高位索引，将结果保存在 result 中
    if (_BitScanReverse(&result, val)) {
        // 如果找到最高位索引，则返回 31 减去 result 的值
        return 31 - result;
    }
    // 如果未找到最高位索引，则返回 0
    return 0;
#else
    // 如果使用的是其他编译器
    if (val == 0)
        // 如果 val 为 0，则返回 32（因为 0 的最高位索引为 31，再加上 1）
        return 32;
    int32_t result = 8;
    // hi 初始化为 0xFF 左移 24 位，即 0xFF000000，用于逐步检测 val 的最高位
    uint32_t hi = 0xFFU << 24;
    // 当 val 的高位为 0 时，向右移动 hi，并增加 result 的值
    while ((val & hi) == 0) {
        hi >>= 8;
        result += 8;
    }
    // 当 val 的高位不为 0 时，向左移动 hi，并减少 result 的值，直到找到最高位
    while (val & hi) {
        result -= 1;
        hi <<= 1;
    }
    // 返回找到的最高位索引
    return result;
#endif
}

static inline int32_t libdivide_count_leading_zeros64(uint64_t val) {
#if defined(__GNUC__) || \
    __has_builtin(__builtin_clzll)
    // 如果使用的是 GCC 编译器或者支持 __builtin_clzll 内置函数
    // 使用 __builtin_clzll 快速计算前导零的个数
    return __builtin_clzll(val);
#elif defined(LIBDIVIDE_VC) && defined(_WIN64)
    // 如果使用的是 Visual C++ 编译器且为 64 位 Windows
    unsigned long result;
    // 使用 _BitScanReverse64 函数查找 val 的最高位索引，将结果保存在 result 中
    if (_BitScanReverse64(&result, val)) {
        // 如果找到最高位索引，则返回 63 减去 result 的值
        return 63 - result;
    }
    // 如果未找到最高位索引，则返回 0
    return 0;
#else
    // 其他情况，分别处理 val 的高位和低位
    uint32_t hi = val >> 32;
    uint32_t lo = val & 0xFFFFFFFF;
    // 如果高位不为 0，则递归调用 libdivide_count_leading_zeros32 处理高位
    if (hi != 0) return libdivide_count_leading_zeros32(hi);
    // 否则返回 32 加上 libdivide_count_leading_zeros32 处理低位的结果
    return 32 + libdivide_count_leading_zeros32(lo);
#endif
}

// libdivide_64_div_32_to_32: divides a 64-bit uint {u1, u0} by a 32-bit
// uint {v}. The result must fit in 32 bits.
// Returns the quotient directly and the remainder in *r
static inline uint32_t libdivide_64_div_32_to_32(uint32_t u1, uint32_t u0, uint32_t v, uint32_t *r) {
#if (defined(LIBDIVIDE_i386) || defined(LIBDIVIDE_X86_64)) && \
     defined(LIBDIVIDE_GCC_STYLE_ASM)
    // 如果使用的是支持 GCC 风格内联汇编的 i386 或 X86_64 架构
    uint32_t result;
    // 使用 inline assembly 执行除法操作，结果保存在 result 中，余数保存在 *r 中
    __asm__("divl %[v]"
            : "=a"(result), "=d"(*r)
            : [v] "r"(v), "a"(u0), "d"(u1)
            );
    // 返回除法结果
    return result;
#else
    // 其他情况，将 u1 和 u0 合并成一个 64 位数 n
    uint64_t n = ((uint64_t)u1 << 32) | u0;
    // 执行除法操作，将结果强制转换为 32 位整数，保存在 result 中
    uint32_t result = (uint32_t)(n / v);
    // 计算余数，保存在 *r 中
    *r = (uint32_t)(n - result * (uint64_t)v);
    // 返回除法结果
    return result;
#endif
}

// libdivide_128_div_64_to_64: divides a 128-bit uint {u1, u0} by a 64-bit
// uint {v}. The result must fit in 64 bits.
// Returns the quotient directly and the remainder in *r
static uint64_t libdivide_128_div_64_to_64(uint64_t u1, uint64_t u0, uint64_t v, uint64_t *r) {
#if defined(LIBDIVIDE_X86_64) && \
    defined(LIBDIVIDE_GCC_STYLE_ASM)
    // 如果使用的是 X86_64 架构并且支持 GCC 风格内联汇编
    uint64_t result;
    // 使用 inline assembly 执行除法操作，结果保存在 result 中，余数保存在 *r 中
    __asm__("divq %[v]"
            : "=a"(result), "=d"(*r)
            : [v] "r"(v), "a"(u0), "d"(u1)
            );
    // 返回除法结果
    return result;
#elif defined(HAS_INT128_T) && \
      defined(HAS_INT128_DIV)
    // 如果支持 __uint128_t 类型和 __uint128_t 的除法操作
    __uint128_t n = ((__uint128_t)u1 << 64) | u0;
    // 执行除法操作，将结果强制转换为 64 位整数，保存在 result 中
    uint64_t result = (uint64_t)(n / v);
    // 计算余数，保存在 *r 中
    *r = (uint64_t)(n - result * (__uint128_t)v);
    // 返回除法结果
    return result;
#else
    // 其他情况，使用 Hacker's Delight 中的代码进行处理
    const uint64_t b = (1ULL << 32); // Number base (32 bits)
    uint64_t un1, un0; // Norm. dividend LSD's
    uint64_t vn1, vn0; // Norm. divisor digits
    uint64_t q1, q0; // Quotient digits
    uint64_t un64, un21, un10; // Dividend digit pairs
    uint64_t rhat; // A remainder
    int32_t s; // Shift amount for norm

    // If overflow, set rem. to an impossible value,
    // 检查除数是否大于等于被除数的高位部分，如果是则返回最大可能的商
    if (u1 >= v) {
        *r = (uint64_t) -1;  // 将余数指针指向最大的无符号整数，表示无法整除
        return (uint64_t) -1;  // 返回最大的无符号整数，表示无法整除
    }

    // 计算除数的前导零位数
    s = libdivide_count_leading_zeros64(v);
    if (s > 0) {
        // 将除数标准化
        v = v << s;  // 将除数左移 s 位，使得除数的高位非零
        un64 = (u1 << s) | (u0 >> (64 - s));  // 将被除数高位左移 s 位，并将低位右移 s 位后与高位进行或运算，得到标准化后的被除数高位部分
        un10 = u0 << s;  // 将被除数整体左移 s 位
    } else {
        // 处理当 s = 0 的情况，即除数的前导零位数为 0
        un64 = u1;
        un10 = u0;
    }

    // 将除数分解为两个 32 位的数字
    vn1 = v >> 32;  // 获取除数的高 32 位
    vn0 = v & 0xFFFFFFFF;  // 获取除数的低 32 位

    // 将被除数右半部分分解为两个 32 位的数字
    un1 = un10 >> 32;  // 获取被除数的右半部分的高 32 位
    un0 = un10 & 0xFFFFFFFF;  // 获取被除数的右半部分的低 32 位

    // 计算第一个商数 q1
    q1 = un64 / vn1;  // 计算高位商数的估计值
    rhat = un64 - q1 * vn1;  // 计算余数的估计值

    while (q1 >= b || q1 * vn0 > b * rhat + un1) {
        q1 = q1 - 1;  // 如果估计的 q1 过大，则减小 q1
        rhat = rhat + vn1;  // 调整余数的估计值
        if (rhat >= b)
            break;
    }

     // 乘法和减法操作
    un21 = un64 * b + un1 - q1 * v;

    // 计算第二个商数 q0
    q0 = un21 / vn1;  // 计算低位商数的估计值
    rhat = un21 - q0 * vn1;  // 计算余数的估计值

    while (q0 >= b || q0 * vn0 > b * rhat + un0) {
        q0 = q0 - 1;  // 如果估计的 q0 过大，则减小 q0
        rhat = rhat + vn1;  // 调整余数的估计值
        if (rhat >= b)
            break;
    }

    *r = (un21 * b + un0 - q0 * v) >> s;  // 计算最终余数并将其右移 s 位
    return q1 * b + q0;  // 返回最终的商
#endif
}

// Bitshift a u128 in place, left (signed_shift > 0) or right (signed_shift < 0)
static inline void libdivide_u128_shift(uint64_t *u1, uint64_t *u0, int32_t signed_shift) {
    if (signed_shift > 0) {
        // 如果 signed_shift 大于 0，则左移 u1 和 u0
        uint32_t shift = signed_shift;
        *u1 <<= shift;  // 左移 u1
        *u1 |= *u0 >> (64 - shift);  // 将 u0 右移以合并到 u1 中
        *u0 <<= shift;  // 左移 u0
    }
    else if (signed_shift < 0) {
        // 如果 signed_shift 小于 0，则右移 u0 和 u1
        uint32_t shift = -signed_shift;
        *u0 >>= shift;  // 右移 u0
        *u0 |= *u1 << (64 - shift);  // 将 u1 左移以合并到 u0 中
        *u1 >>= shift;  // 右移 u1
    }
}

// Computes a 128 / 128 -> 64 bit division, with a 128 bit remainder.
static uint64_t libdivide_128_div_128_to_64(uint64_t u_hi, uint64_t u_lo, uint64_t v_hi, uint64_t v_lo, uint64_t *r_hi, uint64_t *r_lo) {
#if defined(HAS_INT128_T) && \
    defined(HAS_INT128_DIV)
    __uint128_t ufull = u_hi;
    __uint128_t vfull = v_hi;
    ufull = (ufull << 64) | u_lo;
    vfull = (vfull << 64) | v_lo;
    uint64_t res = (uint64_t)(ufull / vfull);  // 计算 ufull / vfull 的整数部分
    __uint128_t remainder = ufull - (vfull * res);  // 计算余数
    *r_lo = (uint64_t)remainder;  // 余数的低 64 位
    *r_hi = (uint64_t)(remainder >> 64);  // 余数的高 64 位
    return res;  // 返回整数部分
#else
    // Adapted from "Unsigned Doubleword Division" in Hacker's Delight
    // We want to compute u / v
    typedef struct { uint64_t hi; uint64_t lo; } u128_t;
    u128_t u = {u_hi, u_lo};
    u128_t v = {v_hi, v_lo};

    if (v.hi == 0) {
        // divisor v is a 64 bit value, so we just need one 128/64 division
        // Note that we are simpler than Hacker's Delight here, because we know
        // the quotient fits in 64 bits whereas Hacker's Delight demands a full
        // 128 bit quotient
        *r_hi = 0;
        return libdivide_128_div_64_to_64(u.hi, u.lo, v.lo, r_lo);  // 执行 128/64 位的除法计算
    }
    // Here v >= 2**64
    // We know that v.hi != 0, so count leading zeros is OK
    // We have 0 <= n <= 63
    uint32_t n = libdivide_count_leading_zeros64(v.hi);  // 计算 v.hi 前导零的数量

    // Normalize the divisor so its MSB is 1
    u128_t v1t = v;
    libdivide_u128_shift(&v1t.hi, &v1t.lo, n);  // 将 v1t 规范化，使其最高位为 1
    uint64_t v1 = v1t.hi; // i.e. v1 = v1t >> 64

    // To ensure no overflow
    u128_t u1 = u;
    libdivide_u128_shift(&u1.hi, &u1.lo, -1);  // 将 u1 规范化，防止溢出

    // Get quotient from divide unsigned insn.
    uint64_t rem_ignored;
    uint64_t q1 = libdivide_128_div_64_to_64(u1.hi, u1.lo, v1, &rem_ignored);  // 执行无符号 128/64 位的除法计算

    // Undo normalization and division of u by 2.
    u128_t q0 = {0, q1};
    libdivide_u128_shift(&q0.hi, &q0.lo, n);  // 恢复 q0 的结果
    libdivide_u128_shift(&q0.hi, &q0.lo, -63);  // 反向移位，相当于除以 2^63

    // Make q0 correct or too small by 1
    // Equivalent to `if (q0 != 0) q0 = q0 - 1;`
    if (q0.hi != 0 || q0.lo != 0) {
        q0.hi -= (q0.lo == 0); // borrow
        q0.lo -= 1;
    }

    // Now q0 is correct.
    // Compute q0 * v as q0v
    // = (q0.hi << 64 + q0.lo) * (v.hi << 64 + v.lo)
    // = (q0.hi * v.hi << 128) + (q0.hi * v.lo << 64) +
    //   (q0.lo * v.hi <<  64) + q0.lo * v.lo)
    // Each term is 128 bit
    // High half of full product (upper 128 bits!) are dropped
    u128_t q0v = {0, 0};
    // 计算 q0v.hi，使用 q0 和 v 的部分乘积，以及 q0.lo 和 v.lo 的高位乘积
    q0v.hi = q0.hi * v.lo + q0.lo * v.hi + libdivide_mullhi_u64(q0.lo, v.lo);
    // 计算 q0v.lo，即 q0.lo 和 v.lo 的乘积
    q0v.lo = q0.lo * v.lo;

    // 计算 u - q0v 得到余数 u_q0v
    // 这就是余数
    u128_t u_q0v = u;
    // 减去 q0v.hi，并处理借位（如果有）
    u_q0v.hi -= q0v.hi + (u.lo < q0v.lo); // 第二项是借位
    // 减去 q0v.lo
    u_q0v.lo -= q0v.lo;

    // 检查 u_q0v 是否大于等于 v
    // 这检查余数是否大于等于除数
    if ((u_q0v.hi > v.hi) ||
        (u_q0v.hi == v.hi && u_q0v.lo >= v.lo)) {
        // 增加 q0
        q0.lo += 1;
        // 处理进位
        q0.hi += (q0.lo == 0);

        // 从余数中减去 v
        u_q0v.hi -= v.hi + (u_q0v.lo < v.lo);
        u_q0v.lo -= v.lo;
    }

    // 将余数的结果写入 r_hi 和 r_lo
    *r_hi = u_q0v.hi;
    *r_lo = u_q0v.lo;

    // 断言 q0.hi 必须为 0
    LIBDIVIDE_ASSERT(q0.hi == 0);
    // 返回商的低位
    return q0.lo;
#endif
}

////////// UINT32

// 定义一个内联函数，用于生成 libdivide_u32_t 结构体对象
static inline struct libdivide_u32_t libdivide_internal_u32_gen(uint32_t d, int branchfree) {
    // 如果除数为0，抛出错误
    if (d == 0) {
        LIBDIVIDE_ERROR("divider must be != 0");
    }

    struct libdivide_u32_t result;
    // 计算除数 d 的 floor(log2(d))
    uint32_t floor_log_2_d = 31 - libdivide_count_leading_zeros32(d);

    // 如果 d 是2的幂次方
    if ((d & (d - 1)) == 0) {
        // 如果是无分支版本的除法，需要在移位值中减去1，因为算法中有一个固定的右移1位
        // 在恢复算法中需要将这个1加回来
        result.magic = 0;
        result.more = (uint8_t)(floor_log_2_d - (branchfree != 0));
    } else {
        uint8_t more;
        uint32_t rem, proposed_m;
        // 通过调用 libdivide_64_div_32_to_32 函数计算 2^floor_log_2_d / d，并得到余数 rem
        proposed_m = libdivide_64_div_32_to_32(1U << floor_log_2_d, 0, d, &rem);

        // 确保余数 rem 大于0且小于d
        LIBDIVIDE_ASSERT(rem > 0 && rem < d);
        const uint32_t e = d - rem;

        // 如果不是无分支版本，并且 e < 2^floor_log_2_d，则选择当前的幂次方
        if (!branchfree && (e < (1U << floor_log_2_d))) {
            more = floor_log_2_d;
        } else {
            // 否则需要使用一般的33位算法，通过对两倍的 rem 进行调整来计算较大的除法
            proposed_m += proposed_m;
            const uint32_t twice_rem = rem + rem;
            if (twice_rem >= d || twice_rem < rem) proposed_m += 1;
            more = floor_log_2_d | LIBDIVIDE_ADD_MARKER;
        }
        result.magic = 1 + proposed_m;
        result.more = more;
        // result.more 的移位通常应为 ceil(log2(d))。但如果使用较小的幂次方，则从移位中减去1，
        // 因为我们使用了较小的幂次方。如果使用较大的幂次方，则通过添加指示符来处理移位。
        // 所以在这两种情况下，floor_log_2_d 都是正确的值。
    }
    return result;
}

// 生成 libdivide_u32_t 结构体对象，使用默认的非无分支版本
struct libdivide_u32_t libdivide_u32_gen(uint32_t d) {
    return libdivide_internal_u32_gen(d, 0);
}

// 生成 libdivide_u32_branchfree_t 结构体对象，用于无分支版本的除法
struct libdivide_u32_branchfree_t libdivide_u32_branchfree_gen(uint32_t d) {
    // 如果除数为1，抛出错误
    if (d == 1) {
        LIBDIVIDE_ERROR("branchfree divider must be != 1");
    }
    // 调用内部函数生成 libdivide_u32_t 结构体对象，然后构造 libdivide_u32_branchfree_t 对象返回
    struct libdivide_u32_t tmp = libdivide_internal_u32_gen(d, 1);
    struct libdivide_u32_branchfree_t ret = {tmp.magic, (uint8_t)(tmp.more & LIBDIVIDE_32_SHIFT_MASK)};
    return ret;
}

// 执行无分支或非无分支版本的32位除法
uint32_t libdivide_u32_do(uint32_t numer, const struct libdivide_u32_t *denom) {
    uint8_t more = denom->more;
    // 如果 magic 为0，表示使用右移来实现除法
    if (!denom->magic) {
        return numer >> more;
    }
    else {
        // 使用 libdivide_mullhi_u32 函数计算 numer 与 denom->magic 的乘积的高32位
        uint32_t q = libdivide_mullhi_u32(denom->magic, numer);
        
        // 检查 more 变量是否包含 LIBDIVIDE_ADD_MARKER 标志位
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 如果包含标志位，则计算 ((numer - q) >> 1) + q，并右移 (more & LIBDIVIDE_32_SHIFT_MASK) 位
            uint32_t t = ((numer - q) >> 1) + q;
            return t >> (more & LIBDIVIDE_32_SHIFT_MASK);
        }
        else {
            // 如果没有包含标志位，说明所有高位都为0，可以直接右移 more 位
            // 所有上位位都为0，不需要屏蔽掉它们。
            return q >> more;
        }
    }
}

// 使用无分支方法实现的32位无符号整数除法
uint32_t libdivide_u32_branchfree_do(uint32_t numer, const struct libdivide_u32_branchfree_t *denom) {
    // 计算商 q = magic * numer 的高32位
    uint32_t q = libdivide_mullhi_u32(denom->magic, numer);
    // 计算 t = ((numer - q) >> 1) + q
    uint32_t t = ((numer - q) >> 1) + q;
    // 返回 t 右移 denom->more 位后的结果
    return t >> denom->more;
}

// 恢复函数，用于处理32位无符号整数的除法结果恢复
uint32_t libdivide_u32_recover(const struct libdivide_u32_t *denom) {
    // 从 denom 结构中提取 more 字段
    uint8_t more = denom->more;
    // 从 more 字段中提取 shift 值
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;

    // 如果 magic 为零，返回 2^shift
    if (!denom->magic) {
        return 1U << shift;
    } else if (!(more & LIBDIVIDE_ADD_MARKER)) {
        // 计算 hi_dividend = 2^shift
        uint32_t hi_dividend = 1U << shift;
        uint32_t rem_ignored;
        // 返回 ceil(2^shift / magic)，其中 magic 不是2的幂
        return 1 + libdivide_64_div_32_to_32(hi_dividend, 0, denom->magic, &rem_ignored);
    } else {
        // 计算 d = (2^(32+shift) + magic)，注意 magic 是一个32位数
        uint64_t half_n = 1ULL << (32 + shift);
        uint64_t d = (1ULL << 32) | denom->magic;
        // 计算半商 half_q = 2^(32+shift) / d
        uint32_t half_q = (uint32_t)(half_n / d);
        uint64_t rem = half_n % d;
        // 计算全商 full_q = 2^(32+shift) / d * 2，并考虑是否需要向上取整
        uint32_t full_q = half_q + half_q + ((rem << 1) >= d);

        // 返回 full_q + 1，用于恢复精确的商值
        return full_q + 1;
    }
}

// 使用无分支方法实现的32位无符号整数除法结果恢复函数
uint32_t libdivide_u32_branchfree_recover(const struct libdivide_u32_branchfree_t *denom) {
    // 从 denom 结构中提取 more 字段
    uint8_t more = denom->more;
    // 从 more 字段中提取 shift 值
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;

    // 如果 magic 为零，返回 2^(shift+1)
    if (!denom->magic) {
        return 1U << (shift + 1);
    } else {
        // 这里我们希望计算 d = 2^(32+shift+1)/(m+2^32)。
        // 注意 (m + 2^32) 是一个 33 位数字。暂时使用 64 位除法。
        // 还要注意 shift 可能最大为 31，所以 shift + 1 将会溢出。
        // 因此，我们先计算 2^(32+shift)/(m+2^32)，然后将商和余数各自加倍。
        uint64_t half_n = 1ULL << (32 + shift);  // 计算 2^(32+shift)
        uint64_t d = (1ULL << 32) | denom->magic;  // 计算 (m + 2^32)
        
        // 注意商保证 <= 32 位，但余数可能需要 33 位！
        uint32_t half_q = (uint32_t)(half_n / d);  // 计算一半的商
        uint64_t rem = half_n % d;  // 计算余数
        
        // 我们计算了 2^(32+shift)/(m+2^32)
        // 需要将其加倍，如果加倍后的余数会使商增加，则将商加1。
        // 注意 rem<<1 不会溢出，因为 rem < d 且 d 是 33 位数字。
        uint32_t full_q = half_q + half_q + ((rem << 1) >= d);

        // 在 gen 中我们向下舍入了（因此 +1）
        return full_q + 1;  // 返回舍入后的结果
    }
}

/////////// UINT64

// 生成用于除法的数据结构，包含魔数和额外信息
static inline struct libdivide_u64_t libdivide_internal_u64_gen(uint64_t d, int branchfree) {
    // 如果除数为0，抛出错误
    if (d == 0) {
        LIBDIVIDE_ERROR("divider must be != 0");
    }

    // 结果数据结构
    struct libdivide_u64_t result;
    // 计算除数的二进制中最高位1之前的0的个数
    uint32_t floor_log_2_d = 63 - libdivide_count_leading_zeros64(d);

    // 如果除数是2的幂次方
    if ((d & (d - 1)) == 0) {
        // 如果是无分支优化的除法，需要调整额外信息
        if (branchfree != 0) {
            result.magic = 0;
            result.more = (uint8_t)(floor_log_2_d - 1);
        } else {
            result.magic = 0;
            result.more = (uint8_t)floor_log_2_d;
        }
    } else {
        uint64_t proposed_m, rem;
        uint8_t more;
        
        // 计算 (1 << (64 + floor_log_2_d)) / d，并返回余数
        proposed_m = libdivide_128_div_64_to_64(1ULL << floor_log_2_d, 0, d, &rem);

        // 确保余数在0到d之间
        LIBDIVIDE_ASSERT(rem > 0 && rem < d);
        const uint64_t e = d - rem;

        // 如果不是无分支优化且 e < 2**floor_log_2_d，则使用此幂次方
        if (!branchfree && e < (1ULL << floor_log_2_d)) {
            more = floor_log_2_d;
        } else {
            // 否则使用通用的65位算法，通过双倍调整来计算更大的除法
            proposed_m += proposed_m;
            const uint64_t twice_rem = rem + rem;
            if (twice_rem >= d || twice_rem < rem) proposed_m += 1;
            more = floor_log_2_d | LIBDIVIDE_ADD_MARKER;
        }
        result.magic = 1 + proposed_m;
        result.more = more;
    }
    return result;
}

// 生成用于除法的数据结构，无分支优化版本
struct libdivide_u64_t libdivide_u64_gen(uint64_t d) {
    return libdivide_internal_u64_gen(d, 0);
}

// 生成用于无分支优化除法的数据结构
struct libdivide_u64_branchfree_t libdivide_u64_branchfree_gen(uint64_t d) {
    // 如果除数为1，抛出错误
    if (d == 1) {
        LIBDIVIDE_ERROR("branchfree divider must be != 1");
    }
    // 调用内部生成函数，获取数据结构，并截取需要的位数信息
    struct libdivide_u64_t tmp = libdivide_internal_u64_gen(d, 1);
    struct libdivide_u64_branchfree_t ret = {tmp.magic, (uint8_t)(tmp.more & LIBDIVIDE_64_SHIFT_MASK)};
    return ret;
}

// 执行64位无符号整数的除法操作
uint64_t libdivide_u64_do(uint64_t numer, const struct libdivide_u64_t *denom) {
    uint8_t more = denom->more;
    // 如果魔数为0，直接右移操作数
    if (!denom->magic) {
        return numer >> more;
    }
    else {
        // 使用 libdivide_mullhi_u64 函数计算 denom->magic 和 numer 的乘积的高位
        uint64_t q = libdivide_mullhi_u64(denom->magic, numer);
        // 如果 more 包含 LIBDIVIDE_ADD_MARKER 标记
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 计算 t 的值，其中 ((numer - q) >> 1) + q
            uint64_t t = ((numer - q) >> 1) + q;
            // 返回 t 右移 (more & LIBDIVIDE_64_SHIFT_MASK) 位后的结果
            return t >> (more & LIBDIVIDE_64_SHIFT_MASK);
        }
        else {
             // 所有的高位都是 0，
             // 不需要屏蔽它们。
            // 直接返回 q 右移 more 位后的结果
            return q >> more;
        }
    }
}

// 使用分支无关的方法计算64位无符号整数的除法，denom是除数结构体指针
uint64_t libdivide_u64_branchfree_do(uint64_t numer, const struct libdivide_u64_branchfree_t *denom) {
    // 计算商q，使用乘法高位返回乘积
    uint64_t q = libdivide_mullhi_u64(denom->magic, numer);
    // 计算t = ((numer - q) >> 1) + q
    uint64_t t = ((numer - q) >> 1) + q;
    // 返回t右移denom->more位的结果
    return t >> denom->more;
}

// 恢复64位无符号整数的除数，denom是除数结构体指针
uint64_t libdivide_u64_recover(const struct libdivide_u64_t *denom) {
    uint8_t more = denom->more;
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;

    // 如果magic为0，返回2^(shift+1)
    if (!denom->magic) {
        return 1ULL << (shift + 1);
    }
    // 如果more不包含LIBDIVIDE_ADD_MARKER标记
    else if (!(more & LIBDIVIDE_ADD_MARKER)) {
        // 计算hi_dividend = 2^shift
        uint64_t hi_dividend = 1ULL << shift;
        uint64_t rem_ignored;
        // 返回1 + libdivide_128_div_64_to_64(hi_dividend, 0, denom->magic, &rem_ignored)的结果
        return 1 + libdivide_128_div_64_to_64(hi_dividend, 0, denom->magic, &rem_ignored);
    }
    // 否则
    else {
        // 计算half_n_hi = 2^shift, half_n_lo = 0
        uint64_t half_n_hi = 1ULL << shift, half_n_lo = 0;
        // d_hi = 1, d_lo = denom->magic
        const uint64_t d_hi = 1, d_lo = denom->magic;
        uint64_t r_hi, r_lo;
        // 计算half_q = libdivide_128_div_128_to_64(half_n_hi, half_n_lo, d_hi, d_lo, &r_hi, &r_lo)的结果
        uint64_t half_q = libdivide_128_div_128_to_64(half_n_hi, half_n_lo, d_hi, d_lo, &r_hi, &r_lo);
        // 计算2^(64+shift)/(m+2^64)，并检查余数是否超过除数
        uint64_t dr_lo = r_lo + r_lo;
        uint64_t dr_hi = r_hi + r_hi + (dr_lo < r_lo); // 最后一项是进位
        int dr_exceeds_d = (dr_hi > d_hi) || (dr_hi == d_hi && dr_lo >= d_lo);
        // 计算full_q = half_q + half_q + (dr_exceeds_d ? 1 : 0)
        uint64_t full_q = half_q + half_q + (dr_exceeds_d ? 1 : 0);
        // 返回full_q + 1的结果
        return full_q + 1;
    }
}

// 使用分支无关的方法恢复64位无符号整数的除数，denom是除数结构体指针
uint64_t libdivide_u64_branchfree_recover(const struct libdivide_u64_branchfree_t *denom) {
    uint8_t more = denom->more;
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;

    // 如果magic为0，返回2^(shift+1)
    if (!denom->magic) {
        return 1ULL << (shift + 1);
    } else {
        // 在这里，我们希望计算 d = 2^(64+shift+1)/(m+2^64)。
        // 注意 (m + 2^64) 是一个 65 位数。这变得复杂了。请看 libdivide_u32_recover 以了解我们在这里做了什么。
        // TODO: 做一些比 128 位数学更好的事情

        // 完整的 n 是一个（可能）129 位值
        // half_n 是一个 128 位值
        // 计算 half_n 的高 64 位。低 64 位为 0。
        uint64_t half_n_hi = 1ULL << shift, half_n_lo = 0;
        // d 是一个 65 位值。最高位始终设为 1。
        const uint64_t d_hi = 1, d_lo = denom->magic;
        // 请注意，商保证 <= 64 位，但余数可能需要 65 位！
        uint64_t r_hi, r_lo;
        uint64_t half_q = libdivide_128_div_128_to_64(half_n_hi, half_n_lo, d_hi, d_lo, &r_hi, &r_lo);
        // 我们计算了 2^(64+shift)/(m+2^64)
        // 将余数加倍 ('dr') 并检查它是否大于 d
        // 请注意，d 是一个 65 位值，因此r1 很小，因此 r1 + r1
        // 无法溢出
        uint64_t dr_lo = r_lo + r_lo;
        uint64_t dr_hi = r_hi + r_hi + (dr_lo < r_lo); // 最后一个项是进位
        int dr_exceeds_d = (dr_hi > d_hi) || (dr_hi == d_hi && dr_lo >= d_lo);
        uint64_t full_q = half_q + half_q + (dr_exceeds_d ? 1 : 0);
        // 返回完整的商加 1
        return full_q + 1;
    }
}

/////////// SINT32

// 生成用于32位有符号整数的除法信息，根据给定的除数和是否分支优化来生成
static inline struct libdivide_s32_t libdivide_internal_s32_gen(int32_t d, int branchfree) {
    // 如果除数为0，则抛出错误
    if (d == 0) {
        LIBDIVIDE_ERROR("divider must be != 0");
    }

    struct libdivide_s32_t result;

    // 如果除数是2的幂或负数的2的幂，则必须使用移位操作
    // 这尤其重要，因为对于-1，魔术算法无法正常工作。
    // 要检查除数是否是2的幂或其倒数，只需检查其绝对值是否恰好有一个位设置为1。
    // 即使对于INT_MIN，这也适用，因为abs(INT_MIN) == INT_MIN，而INT_MIN有一个位设置为1且是2的幂。
    uint32_t ud = (uint32_t)d;
    uint32_t absD = (d < 0) ? -ud : ud;
    uint32_t floor_log_2_d = 31 - libdivide_count_leading_zeros32(absD);
    // 检查是否恰好有一个位设置为1，
    // 不关心absD是否为0，因为那会导致除以0
    if ((absD & (absD - 1)) == 0) {
        // 分支优化和普通路径完全相同
        result.magic = 0;
        result.more = floor_log_2_d | (d < 0 ? LIBDIVIDE_NEGATIVE_DIVISOR : 0);
    } else {
        LIBDIVIDE_ASSERT(floor_log_2_d >= 1);

        uint8_t more;
        // 这里的被除数是2 ** (floor_log_2_d + 31)，因此低32位为0，高位为floor_log_2_d - 1
        uint32_t rem, proposed_m;
        proposed_m = libdivide_64_div_32_to_32(1U << (floor_log_2_d - 1), 0, absD, &rem);
        const uint32_t e = absD - rem;

        // 如果不是分支优化且e < 2 ** floor_log_2_d，则这个幂次可以使用
        if (!branchfree && e < (1U << floor_log_2_d)) {
            // 这个幂次有效
            more = floor_log_2_d - 1;
        } else {
            // 我们需要再高一点。这不应使得proposed_m溢出，但当作为int32_t解释时会使其变负。
            proposed_m += proposed_m;
            const uint32_t twice_rem = rem + rem;
            if (twice_rem >= absD || twice_rem < rem) proposed_m += 1;
            more = floor_log_2_d | LIBDIVIDE_ADD_MARKER;
        }

        proposed_m += 1;
        int32_t magic = (int32_t)proposed_m;

        // 如果除数为负数，则标记为负数。注意在分支完整情况下只有魔术数会被取反。
        if (d < 0) {
            more |= LIBDIVIDE_NEGATIVE_DIVISOR;
            if (!branchfree) {
                magic = -magic;
            }
        }

        result.more = more;
        result.magic = magic;
    }
    return result;
}

// 生成用于32位有符号整数的除法信息，使用普通的生成函数
struct libdivide_s32_t libdivide_s32_gen(int32_t d) {
    return libdivide_internal_s32_gen(d, 0);
}

// 生成用于32位有符号整数的除法信息，使用分支优化的生成函数
struct libdivide_s32_branchfree_t libdivide_s32_branchfree_gen(int32_t d) {
    struct libdivide_s32_t tmp = libdivide_internal_s32_gen(d, 1);
    struct libdivide_s32_branchfree_t result = {tmp.magic, tmp.more};
    return result;
}

// 执行32位有符号整数的除法运算
int32_t libdivide_s32_do(int32_t numer, const struct libdivide_s32_t *denom) {
    // 从结构体指针 denom 中读取 more 字段，并将其转换为 uint8_t 类型的变量 more
    uint8_t more = denom->more;
    // 从 more 中提取 shift 值，使用与操作和预定义的掩码 LIBDIVIDE_32_SHIFT_MASK
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;

    // 如果 denom->magic 为假（即为0）
    if (!denom->magic) {
        // 将 more 的最高位作为符号位，转换为 uint32_t 类型的变量 sign
        uint32_t sign = (int8_t)more >> 7;
        // 创建掩码 mask，用来屏蔽 uq 中超出位移范围的位
        uint32_t mask = (1U << shift) - 1;
        // 计算 uq，将 numer 和根据符号位 mask 进行调整后相加
        uint32_t uq = numer + ((numer >> 31) & mask);
        // 将 uq 转换为 int32_t 类型的变量 q
        int32_t q = (int32_t)uq;
        // 右移 shift 位，对 q 进行修正
        q >>= shift;
        // 根据符号位 sign 对 q 进行调整
        q = (q ^ sign) - sign;
        // 返回计算结果 q
        return q;
    } else {
        // 使用 denom->magic 和 numer 调用 libdivide_mullhi_s32 函数，将结果保存到 uq 中
        uint32_t uq = (uint32_t)libdivide_mullhi_s32(denom->magic, numer);
        // 如果 more 中包含 LIBDIVIDE_ADD_MARKER 标记
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 从 more 中提取符号位，转换为 int32_t 类型的变量 sign
            int32_t sign = (int8_t)more >> 7;
            // 根据符号位 sign 调整 uq 的值，以保证符号扩展正确
            uq += ((uint32_t)numer ^ sign) - sign;
        }
        // 将 uq 转换为 int32_t 类型的变量 q
        int32_t q = (int32_t)uq;
        // 右移 shift 位，对 q 进行修正
        q >>= shift;
        // 如果 q 小于0，则将其增加1
        q += (q < 0);
        // 返回计算结果 q
        return q;
    }
}

// 对应于分支自由的 libdivide_s32_do 函数，用于执行带分支的除法操作
int32_t libdivide_s32_branchfree_do(int32_t numer, const struct libdivide_s32_branchfree_t *denom) {
    // 获取更多信息字节，其中包含了移位数量
    uint8_t more = denom->more;
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
    // 必须进行算术右移并进行符号扩展
    int32_t sign = (int8_t)more >> 7;
    int32_t magic = denom->magic;
    // 使用 libdivide_mullhi_s32 计算乘法高位结果
    int32_t q = libdivide_mullhi_s32(magic, numer);
    // 加上被除数本身
    q += numer;

    // 如果 q 是非负数，无需进一步处理
    // 如果 q 是负数，根据是否为 2 的幂次方，添加 (2**shift)-1 或 (2**shift)
    uint32_t is_power_of_2 = (magic == 0);
    uint32_t q_sign = (uint32_t)(q >> 31);
    q += q_sign & ((1U << shift) - is_power_of_2);

    // 算术右移
    q >>= shift;
    // 根据需要取反
    q = (q ^ sign) - sign;

    return q;
}

// 根据分支全的 libdivide_s32_recover 函数，用于恢复原始除数
int32_t libdivide_s32_recover(const struct libdivide_s32_t *denom) {
    // 获取更多信息字节，其中包含了移位数量
    uint8_t more = denom->more;
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
    if (!denom->magic) {
        // 如果 magic 为 0，说明是 2 的幂次方
        uint32_t absD = 1U << shift;
        // 如果是负除数，取其相反数
        if (more & LIBDIVIDE_NEGATIVE_DIVISOR) {
            absD = -absD;
        }
        return (int32_t)absD;
    } else {
        // 无符号数的运算更为简单
        // 在分支全的情况下，我们只对魔数取反，不知道具体情况
        // 但有足够信息确定魔数的符号性质。如果除数是负数，LIBDIVIDE_NEGATIVE_DIVISOR 标志被设置。
        // 如果 ADD_MARKER 被设置，魔数的符号与除数相反。
        int negative_divisor = (more & LIBDIVIDE_NEGATIVE_DIVISOR);
        int magic_was_negated = (more & LIBDIVIDE_ADD_MARKER)
            ? denom->magic > 0 : denom->magic < 0;

        // 处理 2 的幂次方的情况（包括分支自由）
        if (denom->magic == 0) {
            int32_t result = 1U << shift;
            return negative_divisor ? -result : result;
        }

        uint32_t d = (uint32_t)(magic_was_negated ? -denom->magic : denom->magic);
        uint64_t n = 1ULL << (32 + shift); // 这个移位不超过 30
        uint32_t q = (uint32_t)(n / d);
        int32_t result = (int32_t)q;
        result += 1;
        return negative_divisor ? -result : result;
    }
}

// 对应于分支自由的 libdivide_s32_recover 函数，用于恢复原始除数
int32_t libdivide_s32_branchfree_recover(const struct libdivide_s32_branchfree_t *denom) {
    return libdivide_s32_recover((const struct libdivide_s32_t *)denom);
}

///////////// SINT64

// 内部函数，生成带分支的 libdivide_s64_t 结构
static inline struct libdivide_s64_t libdivide_internal_s64_gen(int64_t d, int branchfree) {
    if (d == 0) {
        LIBDIVIDE_ERROR("divider must be != 0");
    }

    struct libdivide_s64_t result;

    // 如果 d 是 2 的幂次方，或者是负数的 2 的幂次方，必须使用移位。
    // 这对于 magic 算法无法处理 -1 特别重要。
    // 要检查 d 是否是 2 的幂次方或其倒数，仅需检查
    // 将浮点数转换为无符号64位整数
    uint64_t ud = (uint64_t)d;
    // 计算 d 的绝对值
    uint64_t absD = (d < 0) ? -ud : ud;
    // 计算 absD 的 floor(log2(absD))，即 absD 的二进制位数减一
    uint32_t floor_log_2_d = 63 - libdivide_count_leading_zeros64(absD);
    // 检查 absD 是否恰好只有一位为1，即是否为2的幂
    if ((absD & (absD - 1)) == 0) {
        // 如果 absD 是2的幂，设置 result 的 magic 为 0，more 为 floor_log_2_d 或者带有符号位的标记
        result.magic = 0;
        result.more = floor_log_2_d | (d < 0 ? LIBDIVIDE_NEGATIVE_DIVISOR : 0);
    } else {
        // 如果 absD 不是2的幂，则需要进一步计算更多信息
        uint8_t more;
        uint64_t rem, proposed_m;
        // 使用 libdivide_128_div_64_to_64 计算商 proposed_m 和余数 rem
        proposed_m = libdivide_128_div_64_to_64(1ULL << (floor_log_2_d - 1), 0, absD, &rem);
        const uint64_t e = absD - rem;

        // 判断是否需要分支执行非分支化路径
        if (!branchfree && e < (1ULL << floor_log_2_d)) {
            // 如果不需要分支执行且 e < 2^floor_log_2_d，则选择 floor_log_2_d - 1 作为 more
            more = floor_log_2_d - 1;
        } else {
            // 否则，选择更高的位数，可能会导致 proposed_m 为负数
            proposed_m += proposed_m;
            const uint64_t twice_rem = rem + rem;
            if (twice_rem >= absD || twice_rem < rem) proposed_m += 1;
            // 在非分支化情况下设置 LIBDIVIDE_NEGATIVE_DIVISOR 位
            more = floor_log_2_d | LIBDIVIDE_ADD_MARKER;
        }
        proposed_m += 1;
        // 将 proposed_m 转换为 int64_t 类型作为 magic
        int64_t magic = (int64_t)proposed_m;

        // 如果 d 是负数，设置 more 的 LIBDIVIDE_NEGATIVE_DIVISOR 位，并根据情况调整 magic
        if (d < 0) {
            more |= LIBDIVIDE_NEGATIVE_DIVISOR;
            if (!branchfree) {
                magic = -magic;
            }
        }

        // 设置 result 的 more 和 magic
        result.more = more;
        result.magic = magic;
    }
    // 返回计算结果 result
    return result;
}

// 生成一个 libdivide_s64_t 结构体，通过调用内部函数 libdivide_internal_s64_gen
struct libdivide_s64_t libdivide_s64_gen(int64_t d) {
    return libdivide_internal_s64_gen(d, 0);
}

// 生成一个 libdivide_s64_branchfree_t 结构体，通过调用内部函数 libdivide_internal_s64_gen
struct libdivide_s64_branchfree_t libdivide_s64_branchfree_gen(int64_t d) {
    // 调用 libdivide_internal_s64_gen 获取 libdivide_s64_t 结构体
    struct libdivide_s64_t tmp = libdivide_internal_s64_gen(d, 1);
    // 构造 libdivide_s64_branchfree_t 结构体并返回
    struct libdivide_s64_branchfree_t ret = {tmp.magic, tmp.more};
    return ret;
}

// 执行 libdivide_s64_t 结构体定义的除法操作
int64_t libdivide_s64_do(int64_t numer, const struct libdivide_s64_t *denom) {
    uint8_t more = denom->more;
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;

    if (!denom->magic) { // 如果 magic 字段为 0，使用移位路径
        uint64_t mask = (1ULL << shift) - 1;
        uint64_t uq = numer + ((numer >> 63) & mask);
        int64_t q = (int64_t)uq;
        q >>= shift;
        int64_t sign = (int8_t)more >> 7; // 必须是算术右移并且符号扩展
        q = (q ^ sign) - sign;
        return q;
    } else {
        uint64_t uq = (uint64_t)libdivide_mullhi_s64(denom->magic, numer);
        if (more & LIBDIVIDE_ADD_MARKER) {
            int64_t sign = (int8_t)more >> 7; // 必须是算术右移并且符号扩展
            uq += ((uint64_t)numer ^ sign) - sign;
        }
        int64_t q = (int64_t)uq;
        q >>= shift;
        q += (q < 0); // 如果 q 小于 0，则加 1
        return q;
    }
}

// 执行 libdivide_s64_branchfree_t 结构体定义的分支消除除法操作
int64_t libdivide_s64_branchfree_do(int64_t numer, const struct libdivide_s64_branchfree_t *denom) {
    uint8_t more = denom->more;
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
    int64_t sign = (int8_t)more >> 7; // 必须是算术右移并且符号扩展
    int64_t magic = denom->magic;
    int64_t q = libdivide_mullhi_s64(magic, numer);
    q += numer;

    uint64_t is_power_of_2 = (magic == 0);
    uint64_t q_sign = (uint64_t)(q >> 63);
    q += q_sign & ((1ULL << shift) - is_power_of_2);

    q >>= shift; // 算术右移
    q = (q ^ sign) - sign; // 根据符号扩展修正 q
    return q;
}

// 根据 libdivide_s64_t 结构体恢复被除数
int64_t libdivide_s64_recover(const struct libdivide_s64_t *denom) {
    uint8_t more = denom->more;
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
    if (denom->magic == 0) { // 如果 magic 字段为 0，使用移位路径
        uint64_t absD = 1ULL << shift;
        if (more & LIBDIVIDE_NEGATIVE_DIVISOR) {
            absD = -absD;
        }
        return (int64_t)absD;

        }
        uint64_t uabsD = denom->magic;
        return (int64_t)uabsD;
    }
    } else {
        // 如果条件不满足，则执行以下操作，处理除法操作

        // 检查是否为无符号数，无符号数的处理更加简单
        int negative_divisor = (more & LIBDIVIDE_NEGATIVE_DIVISOR);

        // 检查魔数是否被否定
        int magic_was_negated = (more & LIBDIVIDE_ADD_MARKER)
            ? denom->magic > 0 : denom->magic < 0;

        // 将魔数转换为无符号64位整数
        uint64_t d = (uint64_t)(magic_was_negated ? -denom->magic : denom->magic);

        // 左移操作，设置64位整数的高位
        uint64_t n_hi = 1ULL << shift, n_lo = 0;

        // 忽略的余数变量
        uint64_t rem_ignored;

        // 调用libdivide库中的128位除法函数，计算商q，其中参数为64位整数
        uint64_t q = libdivide_128_div_64_to_64(n_hi, n_lo, d, &rem_ignored);

        // 将q + 1转换为int64_t类型作为结果
        int64_t result = (int64_t)(q + 1);

        // 如果是负数除数，将结果取反
        if (negative_divisor) {
            result = -result;
        }

        // 返回最终结果
        return result;
    }
// 结束函数 libdivide_s64_branchfree_recover 的定义，它接受一个指向 libdivide_s64_branchfree_t 结构体的指针参数，并调用 libdivide_s64_recover 来恢复相同类型的 libdivide_s64_t 结构体。
int64_t libdivide_s64_branchfree_recover(const struct libdivide_s64_branchfree_t *denom) {
    return libdivide_s64_recover((const struct libdivide_s64_t *)denom);
}

// 如果定义了 LIBDIVIDE_AVX512 宏，则以下是针对 AVX512 指令集的函数定义：

// 以下四个函数是对于 AVX512 指令集的向量化除法运算函数，分别对应不同的数据类型和分支预测方式。
static inline __m512i libdivide_u32_do_vector(__m512i numers, const struct libdivide_u32_t *denom);
static inline __m512i libdivide_s32_do_vector(__m512i numers, const struct libdivide_s32_t *denom);
static inline __m512i libdivide_u64_do_vector(__m512i numers, const struct libdivide_u64_t *denom);
static inline __m512i libdivide_s64_do_vector(__m512i numers, const struct libdivide_s64_t *denom);

// 以下四个函数是对于 AVX512 指令集的分支预测优化后的向量化除法运算函数，同样对应不同的数据类型。
static inline __m512i libdivide_u32_branchfree_do_vector(__m512i numers, const struct libdivide_u32_branchfree_t *denom);
static inline __m512i libdivide_s32_branchfree_do_vector(__m512i numers, const struct libdivide_s32_branchfree_t *denom);
static inline __m512i libdivide_u64_branchfree_do_vector(__m512i numers, const struct libdivide_u64_branchfree_t *denom);
static inline __m512i libdivide_s64_branchfree_do_vector(__m512i numers, const struct libdivide_s64_branchfree_t *denom);

// 下面是一些内部实用函数的定义：

// 以下函数用于计算 __m512i 类型的向量中每个元素的符号位。它通过算术右移 63 位来获得每个元素的最高位的复制。
static inline __m512i libdivide_s64_signbits(__m512i v) {;
    return _mm512_srai_epi64(v, 63);
}

// 以下函数将 __m512i 类型的向量 v 中的每个元素右移 amt 位。
static inline __m512i libdivide_s64_shift_right_vector(__m512i v, int amt) {
    return _mm512_srai_epi64(v, amt);
}

// 这里假设 b 中包含一个重复的 32 位值。函数执行两个 __m512i 类型向量 a 和 b 的无符号整数乘法，并返回结果的高位部分。
static inline __m512i libdivide_mullhi_u32_vector(__m512i a, __m512i b) {
    __m512i hi_product_0Z2Z = _mm512_srli_epi64(_mm512_mul_epu32(a, b), 32);
    __m512i a1X3X = _mm512_srli_epi64(a, 32);
    __m512i mask = _mm512_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0);
    __m512i hi_product_Z1Z3 = _mm512_and_si512(_mm512_mul_epu32(a1X3X, b), mask);
    return _mm512_or_si512(hi_product_0Z2Z, hi_product_Z1Z3);
}

// 假设 b 中包含一个重复的 32 位值。函数执行两个 __m512i 类型向量 a 和 b 的有符号整数乘法，并返回结果的高位部分。
static inline __m512i libdivide_mullhi_s32_vector(__m512i a, __m512i b) {
    __m512i hi_product_0Z2Z = _mm512_srli_epi64(_mm512_mul_epi32(a, b), 32);
    __m512i a1X3X = _mm512_srli_epi64(a, 32);
    __m512i mask = _mm512_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0);
    __m512i hi_product_Z1Z3 = _mm512_and_si512(_mm512_mul_epi32(a1X3X, b), mask);
    return _mm512_or_si512(hi_product_0Z2Z, hi_product_Z1Z3);
}

// 这里假设 y 中包含一个重复的 64 位值。函数执行两个 __m512i 类型向量 x 和 y 的无符号整数乘法，并返回结果的高位部分。
static inline __m512i libdivide_mullhi_u64_vector(__m512i x, __m512i y) {
    __m512i lomask = _mm512_set1_epi64(0xffffffff);
    __m512i xh = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM) 0xB1);
    __m512i yh = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM) 0xB1);
    __m512i w0 = _mm512_mul_epu32(x, y);
    __m512i w1 = _mm512_mul_epu32(x, yh);
    __m512i w2 = _mm512_mul_epu32(xh, y);
    __m512i w3 = _mm512_mul_epu32(xh, yh);
    __m512i w0h = _mm512_srli_epi64(w0, 32);
    __m512i s1 = _mm512_add_epi64(w1, w0h);
    __m512i s1l = _mm512_and_si512(s1, lomask);
    // 将 s1 的每个元素逻辑右移 32 位，并存储在 s1h 中
    __m512i s1h = _mm512_srli_epi64(s1, 32);
    // 将 w2 和 s1l 的对应元素相加，并存储在 s2 中
    __m512i s2 = _mm512_add_epi64(w2, s1l);
    // 将 s2 的每个元素逻辑右移 32 位，并存储在 s2h 中
    __m512i s2h = _mm512_srli_epi64(s2, 32);
    // 将 w3 和 s1h 的对应元素相加，并存储在 hi 中
    __m512i hi = _mm512_add_epi64(w3, s1h);
    // 继续将 hi 和 s2h 的对应元素相加，并更新 hi
    hi = _mm512_add_epi64(hi, s2h);

    // 返回最终结果 hi
    return hi;
}

// 结束 libdivide_mullhi_s64_vector 函数的定义

static inline __m512i libdivide_mullhi_s64_vector(__m512i x, __m512i y) {
    // 调用 libdivide_mullhi_u64_vector 函数计算无符号整数 x 和 y 的乘积的高位
    __m512i p = libdivide_mullhi_u64_vector(x, y);
    // 计算 x 和 y 的符号位并求与，得到 t1
    __m512i t1 = _mm512_and_si512(libdivide_s64_signbits(x), y);
    // 计算 y 和 x 的符号位并求与，得到 t2
    __m512i t2 = _mm512_and_si512(libdivide_s64_signbits(y), x);
    // 从 p 中减去 t1
    p = _mm512_sub_epi64(p, t1);
    // 从 p 中减去 t2
    p = _mm512_sub_epi64(p, t2);
    // 返回 p
    return p;
}

////////// UINT32

// libdivide_u32_do_vector 函数的定义
__m512i libdivide_u32_do_vector(__m512i numers, const struct libdivide_u32_t *denom) {
    // 从 denom 结构中读取 more 字段
    uint8_t more = denom->more;
    // 如果 denom->magic 为 0
    if (!denom->magic) {
        // 对 numers 中的每个元素逻辑右移 more 位并返回结果
        return _mm512_srli_epi32(numers, more);
    }
    else {
        // 计算 numers 与 denom->magic 的乘积的高位并存入 q
        __m512i q = libdivide_mullhi_u32_vector(numers, _mm512_set1_epi32(denom->magic));
        // 如果 more 的 LIBDIVIDE_ADD_MARKER 标记为真
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 从 numers 减去 q，结果右移 1 位后加上 q，并将结果右移 shift 位并返回
            uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
            __m512i t = _mm512_add_epi32(_mm512_srli_epi32(_mm512_sub_epi32(numers, q), 1), q);
            return _mm512_srli_epi32(t, shift);
        }
        else {
            // 对 q 中的每个元素逻辑右移 more 位并返回结果
            return _mm512_srli_epi32(q, more);
        }
    }
}

// libdivide_u32_branchfree_do_vector 函数的定义
__m512i libdivide_u32_branchfree_do_vector(__m512i numers, const struct libdivide_u32_branchfree_t *denom) {
    // 计算 numers 与 denom->magic 的乘积的高位并存入 q
    __m512i q = libdivide_mullhi_u32_vector(numers, _mm512_set1_epi32(denom->magic));
    // 计算 numers 减去 q，结果右移 1 位后加上 q，并将结果右移 denom->more 位并返回
    __m512i t = _mm512_add_epi32(_mm512_srli_epi32(_mm512_sub_epi32(numers, q), 1), q);
    return _mm512_srli_epi32(t, denom->more);
}

////////// UINT64

// libdivide_u64_do_vector 函数的定义
__m512i libdivide_u64_do_vector(__m512i numers, const struct libdivide_u64_t *denom) {
    // 从 denom 结构中读取 more 字段
    uint8_t more = denom->more;
    // 如果 denom->magic 为 0
    if (!denom->magic) {
        // 对 numers 中的每个元素逻辑右移 more 位并返回结果
        return _mm512_srli_epi64(numers, more);
    }
    else {
        // 计算 numers 与 denom->magic 的乘积的高位并存入 q
        __m512i q = libdivide_mullhi_u64_vector(numers, _mm512_set1_epi64(denom->magic));
        // 如果 more 的 LIBDIVIDE_ADD_MARKER 标记为真
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 从 numers 减去 q，结果右移 1 位后加上 q，并将结果右移 shift 位并返回
            uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
            __m512i t = _mm512_add_epi64(_mm512_srli_epi64(_mm512_sub_epi64(numers, q), 1), q);
            return _mm512_srli_epi64(t, shift);
        }
        else {
            // 对 q 中的每个元素逻辑右移 more 位并返回结果
            return _mm512_srli_epi64(q, more);
        }
    }
}

// libdivide_u64_branchfree_do_vector 函数的定义
__m512i libdivide_u64_branchfree_do_vector(__m512i numers, const struct libdivide_u64_branchfree_t *denom) {
    // 计算 numers 与 denom->magic 的乘积的高位并存入 q
    __m512i q = libdivide_mullhi_u64_vector(numers, _mm512_set1_epi64(denom->magic));
    // 计算 numers 减去 q，结果右移 1 位后加上 q，并将结果右移 denom->more 位并返回
    __m512i t = _mm512_add_epi64(_mm512_srli_epi64(_mm512_sub_epi64(numers, q), 1), q);
    return _mm512_srli_epi64(t, denom->more);
}

////////// SINT32

// libdivide_s32_do_vector 函数的定义
__m512i libdivide_s32_do_vector(__m512i numers, const struct libdivide_s32_t *denom) {
    // 从 denom 结构中读取 more 字段
    uint8_t more = denom->more;
    // 如果分母的 magic 值为零，执行以下代码块
    if (!denom->magic) {
        // 提取更多位的值中的 LIBDIVIDE_32_SHIFT_MASK，并赋给 shift
        uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
        // 创建一个 mask，用于掩码操作
        uint32_t mask = (1U << shift) - 1;
        // 创建一个 roundToZeroTweak 的 SIMD 寄存器，用于舍入到零的调整
        __m512i roundToZeroTweak = _mm512_set1_epi32(mask);
        // 计算 q = numer + ((numer >> 31) & roundToZeroTweak);
        __m512i q = _mm512_add_epi32(numers, _mm512_and_si512(_mm512_srai_epi32(numers, 31), roundToZeroTweak));
        // 对 q 进行算术右移操作
        q = _mm512_srai_epi32(q, shift);
        // 创建一个 sign 寄存器，其中包含 more 的最高位符号位
        __m512i sign = _mm512_set1_epi32((int8_t)more >> 7);
        // 执行 q = (q ^ sign) - sign 的计算
        q = _mm512_sub_epi32(_mm512_xor_si512(q, sign), sign);
        // 返回计算结果 q
        return q;
    }
    // 如果分母的 magic 值非零，执行以下代码块
    else {
        // 使用 libdivide_mullhi_s32_vector 计算 q = numer * denom->magic 的高位结果
        __m512i q = libdivide_mullhi_s32_vector(numers, _mm512_set1_epi32(denom->magic));
        // 如果 more 中包含 LIBDIVIDE_ADD_MARKER 标记
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 创建一个 sign 寄存器，其中包含 more 的最高位符号位
            __m512i sign = _mm512_set1_epi32((int8_t)more >> 7);
            // 执行 q += ((numer ^ sign) - sign) 的计算，进行算术右移操作
            q = _mm512_add_epi32(q, _mm512_sub_epi32(_mm512_xor_si512(numers, sign), sign));
        }
        // 对 q 进行算术右移操作，shift 由 more 的低位表示
        q = _mm512_srai_epi32(q, more & LIBDIVIDE_32_SHIFT_MASK);
        // 对 q 进行修正，q += (q < 0)，即当 q 小于零时加 1
        q = _mm512_add_epi32(q, _mm512_srli_epi32(q, 31));
        // 返回计算结果 q
        return q;
    }
}

__m512i libdivide_s32_branchfree_do_vector(__m512i numers, const struct libdivide_s32_branchfree_t *denom) {
    int32_t magic = denom->magic;  // 从结构体中获取魔数
    uint8_t more = denom->more;  // 从结构体中获取更多信息
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;  // 从更多信息中提取出位移量

    // 必须是算术右移
    __m512i sign = _mm512_set1_epi32((int8_t)more >> 7);  // 创建一个包含符号位的向量

    // 计算乘法的高位结果
    __m512i q = libdivide_mullhi_s32_vector(numers, _mm512_set1_epi32(magic));
    q = _mm512_add_epi32(q, numers); // q += numers

    // 如果 q 是非负数，无需处理
    // 如果 q 是负数，根据是否是2的幂，要添加 (2**shift)-1 或 2**shift
    uint32_t is_power_of_2 = (magic == 0);
    __m512i q_sign = _mm512_srai_epi32(q, 31); // q_sign = q >> 31
    __m512i mask = _mm512_set1_epi32((1U << shift) - is_power_of_2);
    q = _mm512_add_epi32(q, _mm512_and_si512(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm512_srai_epi32(q, shift); // q >>= shift
    q = _mm512_sub_epi32(_mm512_xor_si512(q, sign), sign); // q = (q ^ sign) - sign
    return q;
}

////////// SINT64

__m512i libdivide_s64_do_vector(__m512i numers, const struct libdivide_s64_t *denom) {
    uint8_t more = denom->more;  // 从结构体中获取更多信息
    int64_t magic = denom->magic;  // 从结构体中获取魔数

    if (magic == 0) { // 如果是位移路径
        uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;  // 从更多信息中提取出位移量
        uint64_t mask = (1ULL << shift) - 1;  // 创建掩码
        __m512i roundToZeroTweak = _mm512_set1_epi64(mask);

        // q = numer + ((numer >> 63) & roundToZeroTweak);
        __m512i q = _mm512_add_epi64(numers, _mm512_and_si512(libdivide_s64_signbits(numers), roundToZeroTweak));
        q = libdivide_s64_shift_right_vector(q, shift);

        __m512i sign = _mm512_set1_epi32((int8_t)more >> 7);  // 创建一个包含符号位的向量

        // q = (q ^ sign) - sign;
        q = _mm512_sub_epi64(_mm512_xor_si512(q, sign), sign);

        return q;
    } else {
        __m512i q = libdivide_mullhi_s64_vector(numers, _mm512_set1_epi64(magic));

        if (more & LIBDIVIDE_ADD_MARKER) {  // 如果有加法标记
            // 必须是算术右移
            __m512i sign = _mm512_set1_epi32((int8_t)more >> 7);

            // q += ((numer ^ sign) - sign);
            q = _mm512_add_epi64(q, _mm512_sub_epi64(_mm512_xor_si512(numers, sign), sign));
        }

        // q >>= denom->mult_path.shift
        q = libdivide_s64_shift_right_vector(q, more & LIBDIVIDE_64_SHIFT_MASK);
        q = _mm512_add_epi64(q, _mm512_srli_epi64(q, 63)); // q += (q < 0)

        return q;
    }
}

__m512i libdivide_s64_branchfree_do_vector(__m512i numers, const struct libdivide_s64_branchfree_t *denom) {
    int64_t magic = denom->magic;  // 从结构体中获取魔数
    uint8_t more = denom->more;  // 从结构体中获取更多信息
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;  // 从更多信息中提取出位移量

    // 必须是算术右移
    __m512i sign = _mm512_set1_epi32((int8_t)more >> 7);  // 创建一个包含符号位的向量

    // 计算乘法的高位结果
    __m512i q = libdivide_mullhi_s64_vector(numers, _mm512_set1_epi64(magic));
    q = _mm512_add_epi64(q, numers); // q += numers

    // 如果 q 是非负数，无需处理.
    // 如果 q 是负数，我们希望根据 d 是否为 2 的幂来添加 (2**shift)-1 或 (2**shift)
    uint32_t is_power_of_2 = (magic == 0);
    // 使用 libdivide 库函数计算 q 的符号位
    __m512i q_sign = libdivide_s64_signbits(q); // q_sign = q >> 63
    // 创建一个掩码，根据 d 是否为 2 的幂来确定要添加的值
    __m512i mask = _mm512_set1_epi64((1ULL << shift) - is_power_of_2);
    // 将 q 的值增加 (q_sign & mask) 的结果
    q = _mm512_add_epi64(q, _mm512_and_si512(q_sign, mask)); // q = q + (q_sign & mask)
    // 使用 libdivide 库函数将 q 右移 shift 位
    q = libdivide_s64_shift_right_vector(q, shift); // q >>= shift
    // 对 q 应用逐位异或和减法操作，以获取最终的结果
    q = _mm512_sub_epi64(_mm512_xor_si512(q, sign), sign); // q = (q ^ sign) - sign
    // 返回计算结果 q
    return q;
}

#elif defined(LIBDIVIDE_AVX2)

// 声明一系列静态内联函数，用于处理 AVX2 指令集下的向量化除法操作

// 处理无符号32位整数的向量化除法操作
static inline __m256i libdivide_u32_do_vector(__m256i numers, const struct libdivide_u32_t *denom);

// 处理有符号32位整数的向量化除法操作
static inline __m256i libdivide_s32_do_vector(__m256i numers, const struct libdivide_s32_t *denom);

// 处理无符号64位整数的向量化除法操作
static inline __m256i libdivide_u64_do_vector(__m256i numers, const struct libdivide_u64_t *denom);

// 处理有符号64位整数的向量化除法操作
static inline __m256i libdivide_s64_do_vector(__m256i numers, const struct libdivide_s64_t *denom);

// 处理无符号32位整数的无分支向量化除法操作
static inline __m256i libdivide_u32_branchfree_do_vector(__m256i numers, const struct libdivide_u32_branchfree_t *denom);

// 处理有符号32位整数的无分支向量化除法操作
static inline __m256i libdivide_s32_branchfree_do_vector(__m256i numers, const struct libdivide_s32_branchfree_t *denom);

// 处理无符号64位整数的无分支向量化除法操作
static inline __m256i libdivide_u64_branchfree_do_vector(__m256i numers, const struct libdivide_u64_branchfree_t *denom);

// 处理有符号64位整数的无分支向量化除法操作
static inline __m256i libdivide_s64_branchfree_do_vector(__m256i numers, const struct libdivide_s64_branchfree_t *denom);

//////// Internal Utility Functions

// 实现 _mm256_srai_epi64(v, 63) 的功能（来自 AVX512）
static inline __m256i libdivide_s64_signbits(__m256i v) {
    // 复制高位，并生成符号位掩码
    __m256i hiBitsDuped = _mm256_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i signBits = _mm256_srai_epi32(hiBitsDuped, 31); // 右移获得符号位
    return signBits;
}

// 实现 _mm256_srai_epi64 的功能（来自 AVX512）
static inline __m256i libdivide_s64_shift_right_vector(__m256i v, int amt) {
    const int b = 64 - amt;
    __m256i m = _mm256_set1_epi64x(1ULL << (b - 1)); // 创建掩码
    __m256i x = _mm256_srli_epi64(v, amt); // 右移指定位数
    __m256i result = _mm256_sub_epi64(_mm256_xor_si256(x, m), m); // 计算结果
    return result;
}

// 这里假定 b 包含一个重复的32位值
static inline __m256i libdivide_mullhi_u32_vector(__m256i a, __m256i b) {
    // 计算高位乘积
    __m256i hi_product_0Z2Z = _mm256_srli_epi64(_mm256_mul_epu32(a, b), 32);
    __m256i a1X3X = _mm256_srli_epi64(a, 32);
    __m256i mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
    __m256i hi_product_Z1Z3 = _mm256_and_si256(_mm256_mul_epu32(a1X3X, b), mask);
    return _mm256_or_si256(hi_product_0Z2Z, hi_product_Z1Z3);
}

// 假定 b 是一个重复的32位值
static inline __m256i libdivide_mullhi_s32_vector(__m256i a, __m256i b) {
    // 计算有符号整数的高位乘积
    __m256i hi_product_0Z2Z = _mm256_srli_epi64(_mm256_mul_epi32(a, b), 32);
    __m256i a1X3X = _mm256_srli_epi64(a, 32);
    __m256i mask = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
    __m256i hi_product_Z1Z3 = _mm256_and_si256(_mm256_mul_epi32(a1X3X, b), mask);
    return _mm256_or_si256(hi_product_0Z2Z, hi_product_Z1Z3);
}

// 这里假定 y 包含一个重复的64位值
// 参考：https://stackoverflow.com/a/28827013
static inline __m256i libdivide_mullhi_u64_vector(__m256i x, __m256i y) {
    __m256i lomask = _mm256_set1_epi64x(0xffffffff);
    __m256i xh = _mm256_shuffle_epi32(x, 0xB1); // x0l, x0h, x1l, x1h
    __m256i yh = _mm256_shuffle_epi32(y, 0xB1); // y0l, y0h, y1l, y1h
    __m256i w0 = _mm256_mul_epu32(x, y); // x0l*y0l, x1l*y1l
    # 计算低位乘积 x0l*y0h 和 x1l*y1h，并将结果存储在 w1 中
    __m256i w1 = _mm256_mul_epu32(x, yh);
    
    # 计算高位乘积 x0h*y0l 和 x1h*y0l，并将结果存储在 w2 中
    __m256i w2 = _mm256_mul_epu32(xh, y);
    
    # 计算高位乘积 x0h*y0h 和 x1h*y1h，并将结果存储在 w3 中
    __m256i w3 = _mm256_mul_epu32(xh, yh);
    
    # 将 w0 向右移动 32 位，获取高位部分存储在 w0h 中
    __m256i w0h = _mm256_srli_epi64(w0, 32);
    
    # 将 w1 和 w0h 相加，得到 s1
    __m256i s1 = _mm256_add_epi64(w1, w0h);
    
    # 将 s1 与 lomask 按位与，获取低位部分存储在 s1l 中
    __m256i s1l = _mm256_and_si256(s1, lomask);
    
    # 将 s1 向右移动 32 位，获取高位部分存储在 s1h 中
    __m256i s1h = _mm256_srli_epi64(s1, 32);
    
    # 将 w2 和 s1l 相加，得到 s2
    __m256i s2 = _mm256_add_epi64(w2, s1l);
    
    # 将 s2 向右移动 32 位，获取高位部分存储在 s2h 中
    __m256i s2h = _mm256_srli_epi64(s2, 32);
    
    # 将 w3 和 s1h 相加，然后加上 s2h，得到最终结果存储在 hi 中
    __m256i hi = _mm256_add_epi64(w3, s1h);
    hi = _mm256_add_epi64(hi, s2h);
    
    # 返回最终计算结果 hi
    return hi;
}

// 结束静态内联函数 libdivide_mullhi_s64_vector 的定义

// 使用无符号 64 位整数向量进行乘法高位运算，返回一个 256 位整数向量
static inline __m256i libdivide_mullhi_s64_vector(__m256i x, __m256i y) {
    // 调用 libdivide_mullhi_u64_vector 函数执行无符号 64 位整数向量的乘法高位运算
    __m256i p = libdivide_mullhi_u64_vector(x, y);
    // 计算 x 和 y 的符号位，并与乘法结果相与，存储到 t1 和 t2 中
    __m256i t1 = _mm256_and_si256(libdivide_s64_signbits(x), y);
    __m256i t2 = _mm256_and_si256(libdivide_s64_signbits(y), x);
    // 从乘法结果中减去 t1 和 t2，得到最终结果 p
    p = _mm256_sub_epi64(p, t1);
    p = _mm256_sub_epi64(p, t2);
    // 返回乘法高位运算结果
    return p;
}

////////// UINT32

// 使用无符号 32 位整数向量执行除法运算，返回一个 256 位整数向量
__m256i libdivide_u32_do_vector(__m256i numers, const struct libdivide_u32_t *denom) {
    // 获取结构体 denom 中的 more 字段
    uint8_t more = denom->more;
    // 如果 magic 字段为 0，则执行逻辑右移操作
    if (!denom->magic) {
        return _mm256_srli_epi32(numers, more);
    }
    else {
        // 否则，执行乘法高位运算并右移操作
        __m256i q = libdivide_mullhi_u32_vector(numers, _mm256_set1_epi32(denom->magic));
        // 如果 more 中包含 LIBDIVIDE_ADD_MARKER 标志位
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 计算 t = ((numer - q) >> 1) + q
            // 然后再右移 denom->shift 位
            uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
            __m256i t = _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(numers, q), 1), q);
            return _mm256_srli_epi32(t, shift);
        }
        else {
            // 否则，直接右移 more 位
            return _mm256_srli_epi32(q, more);
        }
    }
}

// 使用无符号 32 位整数向量执行分支无关的除法运算，返回一个 256 位整数向量
__m256i libdivide_u32_branchfree_do_vector(__m256i numers, const struct libdivide_u32_branchfree_t *denom) {
    // 执行乘法高位运算
    __m256i q = libdivide_mullhi_u32_vector(numers, _mm256_set1_epi32(denom->magic));
    // 计算 t = ((numer - q) >> 1) + q，然后右移 denom->more 位
    __m256i t = _mm256_add_epi32(_mm256_srli_epi32(_mm256_sub_epi32(numers, q), 1), q);
    return _mm256_srli_epi32(t, denom->more);
}

////////// UINT64

// 使用无符号 64 位整数向量执行除法运算，返回一个 256 位整数向量
__m256i libdivide_u64_do_vector(__m256i numers, const struct libdivide_u64_t *denom) {
    // 获取结构体 denom 中的 more 字段
    uint8_t more = denom->more;
    // 如果 magic 字段为 0，则执行逻辑右移操作
    if (!denom->magic) {
        return _mm256_srli_epi64(numers, more);
    }
    else {
        // 否则，执行乘法高位运算并右移操作
        __m256i q = libdivide_mullhi_u64_vector(numers, _mm256_set1_epi64x(denom->magic));
        // 如果 more 中包含 LIBDIVIDE_ADD_MARKER 标志位
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 计算 t = ((numer - q) >> 1) + q
            // 然后再右移 denom->shift 位
            uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
            __m256i t = _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q);
            return _mm256_srli_epi64(t, shift);
        }
        else {
            // 否则，直接右移 more 位
            return _mm256_srli_epi64(q, more);
        }
    }
}

// 使用无符号 64 位整数向量执行分支无关的除法运算，返回一个 256 位整数向量
__m256i libdivide_u64_branchfree_do_vector(__m256i numers, const struct libdivide_u64_branchfree_t *denom) {
    // 执行乘法高位运算
    __m256i q = libdivide_mullhi_u64_vector(numers, _mm256_set1_epi64x(denom->magic));
    // 计算 t = ((numer - q) >> 1) + q，然后右移 denom->more 位
    __m256i t = _mm256_add_epi64(_mm256_srli_epi64(_mm256_sub_epi64(numers, q), 1), q);
    return _mm256_srli_epi64(t, denom->more);
}

////////// SINT32

// 使用有符号 32 位整数向量执行除法运算，返回一个 256 位整数向量
__m256i libdivide_s32_do_vector(__m256i numers, const struct libdivide_s32_t *denom) {
    // 获取结构体 denom 中的 more 字段
    uint8_t more = denom->more;

// 如果 more 中包含 LIBDIVIDE_ADD_MARKER 标志位
if (!denom->magic) {
    // 如果 magic 字段为 0，则执行逻辑右移操作
    return _mm256_srli_epi32(numers, more);
} else {
    // 否则，执行乘法高位运算并右移操作
__m256i q = libdivide_mullhi_u32_vector(numers, _mm256_set1_epi32(denom->magic));
//  uint32_t t = ((numer - q) >> 1) + q
    // 检查分母的魔数是否为零，如果为零则执行以下操作
    if (!denom->magic) {
        // 从 more 中提取 LIBDIVIDE_32_SHIFT_MASK，表示移位数
        uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
        // 创建一个掩码，用于取低 shift 位的数
        uint32_t mask = (1U << shift) - 1;
        // 创建一个全为 mask 的 __m256i 向量，用于向下取整使用
        __m256i roundToZeroTweak = _mm256_set1_epi32(mask);
        // 计算 q = numer + ((numer >> 31) & roundToZeroTweak);
        __m256i q = _mm256_add_epi32(numers, _mm256_and_si256(_mm256_srai_epi32(numers, 31), roundToZeroTweak));
        // 对 q 进行右移操作
        q = _mm256_srai_epi32(q, shift);
        // 创建一个符号向量，用于处理负数的情况
        __m256i sign = _mm256_set1_epi32((int8_t)more >> 7);
        // 执行 q = (q ^ sign) - sign;
        q = _mm256_sub_epi32(_mm256_xor_si256(q, sign), sign);
        // 返回计算结果 q
        return q;
    }
    else {
        // 使用 libdivide_mullhi_s32_vector 函数计算乘法高位结果，结果存入 q 中
        __m256i q = libdivide_mullhi_s32_vector(numers, _mm256_set1_epi32(denom->magic));
        // 如果 more 中包含 LIBDIVIDE_ADD_MARKER 标记，则执行以下操作
        if (more & LIBDIVIDE_ADD_MARKER) {
             // more 的高位表示算术右移
            __m256i sign = _mm256_set1_epi32((int8_t)more >> 7);
             // q += ((numer ^ sign) - sign);
            q = _mm256_add_epi32(q, _mm256_sub_epi32(_mm256_xor_si256(numers, sign), sign));
        }
        // 对 q 进行右移操作
        q = _mm256_srai_epi32(q, more & LIBDIVIDE_32_SHIFT_MASK);
        // q += (q < 0)，处理负数情况下的修正
        q = _mm256_add_epi32(q, _mm256_srli_epi32(q, 31));
        // 返回计算结果 q
        return q;
    }
}

__m256i libdivide_s32_branchfree_do_vector(__m256i numers, const struct libdivide_s32_branchfree_t *denom) {
    int32_t magic = denom->magic;  // 从结构体中获取魔数 magic
    uint8_t more = denom->more;    // 从结构体中获取更多信息字段
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;  // 计算需要移位的位数，通过与操作获取

    // 必须是算术右移
    __m256i sign = _mm256_set1_epi32((int8_t)more >> 7);  // 根据 more 的最高位设置符号位

    __m256i q = libdivide_mullhi_s32_vector(numers, _mm256_set1_epi32(magic));  // 计算高位乘积
    q = _mm256_add_epi32(q, numers);  // q += numers

    // 如果 q 是非负数，无需处理
    // 如果 q 是负数，根据是否是2的幂，添加 (2**shift)-1 或者 (2**shift)
    uint32_t is_power_of_2 = (magic == 0);
    __m256i q_sign = _mm256_srai_epi32(q, 31);  // q_sign = q >> 31
    __m256i mask = _mm256_set1_epi32((1U << shift) - is_power_of_2);
    q = _mm256_add_epi32(q, _mm256_and_si256(q_sign, mask));  // q = q + (q_sign & mask)
    q = _mm256_srai_epi32(q, shift);  // q >>= shift
    q = _mm256_sub_epi32(_mm256_xor_si256(q, sign), sign);  // q = (q ^ sign) - sign

    return q;
}

////////// SINT64

__m256i libdivide_s64_do_vector(__m256i numers, const struct libdivide_s64_t *denom) {
    uint8_t more = denom->more;  // 从结构体中获取更多信息字段
    int64_t magic = denom->magic;  // 从结构体中获取魔数

    if (magic == 0) {  // 如果魔数为0，使用移位路径
        uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;  // 计算需要移位的位数
        uint64_t mask = (1ULL << shift) - 1;  // 计算掩码
        __m256i roundToZeroTweak = _mm256_set1_epi64x(mask);

        // q = numer + ((numer >> 63) & roundToZeroTweak);
        __m256i q = _mm256_add_epi64(numers, _mm256_and_si256(libdivide_s64_signbits(numers), roundToZeroTweak));
        q = libdivide_s64_shift_right_vector(q, shift);

        __m256i sign = _mm256_set1_epi32((int8_t)more >> 7);  // 根据 more 的最高位设置符号位

        q = _mm256_sub_epi64(_mm256_xor_si256(q, sign), sign);  // q = (q ^ sign) - sign

        return q;
    }
    else {  // 非移位路径
        __m256i q = libdivide_mullhi_s64_vector(numers, _mm256_set1_epi64x(magic));  // 计算高位乘积

        if (more & LIBDIVIDE_ADD_MARKER) {
            // 必须是算术右移
            __m256i sign = _mm256_set1_epi32((int8_t)more >> 7);
            q = _mm256_add_epi64(q, _mm256_sub_epi64(_mm256_xor_si256(numers, sign), sign));
        }

        q = libdivide_s64_shift_right_vector(q, more & LIBDIVIDE_64_SHIFT_MASK);  // q >>= denom->mult_path.shift
        q = _mm256_add_epi64(q, _mm256_srli_epi64(q, 63));  // q += (q < 0)

        return q;
    }
}

__m256i libdivide_s64_branchfree_do_vector(__m256i numers, const struct libdivide_s64_branchfree_t *denom) {
    int64_t magic = denom->magic;  // 从结构体中获取魔数
    uint8_t more = denom->more;    // 从结构体中获取更多信息字段
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;  // 计算需要移位的位数

    // 必须是算术右移
    __m256i sign = _mm256_set1_epi32((int8_t)more >> 7);  // 根据 more 的最高位设置符号位

    __m256i q = libdivide_mullhi_s64_vector(numers, _mm256_set1_epi64x(magic));  // 计算高位乘积
    q = _mm256_add_epi64(q, numers);  // q += numers

    // 如果 q 是非负数，无需处理.
    // 如果 q 是负数，我们希望根据 d 是否为2的幂来添加 (2**shift)-1 或 (2**shift)
    uint32_t is_power_of_2 = (magic == 0);  // 检查 magic 是否为零，以确定 d 是否为2的幂
    // 计算 q 的符号位
    __m256i q_sign = libdivide_s64_signbits(q);
    // 创建一个掩码，用于根据 d 是否为2的幂来选择要添加的值
    __m256i mask = _mm256_set1_epi64x((1ULL << shift) - is_power_of_2);
    // 将 q 增加 (q_sign & mask) 的结果
    q = _mm256_add_epi64(q, _mm256_and_si256(q_sign, mask));
    // 使用 libdivide_s64_shift_right_vector 函数将 q 右移 shift 位
    q = libdivide_s64_shift_right_vector(q, shift);
    // 执行 q 的按位异或和减法操作，计算最终的结果
    q = _mm256_sub_epi64(_mm256_xor_si256(q, sign), sign);
    // 返回计算后的结果 q
    return q;
// 结束前面的条件分支代码块
}

// 如果定义了 LIBDIVIDE_SSE2，以下是针对 SSE2 的函数声明

// 定义了 libdivide_u32_do_vector 函数，处理无符号 32 位整数向量除法
static inline __m128i libdivide_u32_do_vector(__m128i numers, const struct libdivide_u32_t *denom);

// 定义了 libdivide_s32_do_vector 函数，处理有符号 32 位整数向量除法
static inline __m128i libdivide_s32_do_vector(__m128i numers, const struct libdivide_s32_t *denom);

// 定义了 libdivide_u64_do_vector 函数，处理无符号 64 位整数向量除法
static inline __m128i libdivide_u64_do_vector(__m128i numers, const struct libdivide_u64_t *denom);

// 定义了 libdivide_s64_do_vector 函数，处理有符号 64 位整数向量除法
static inline __m128i libdivide_s64_do_vector(__m128i numers, const struct libdivide_s64_t *denom);

// 定义了 libdivide_u32_branchfree_do_vector 函数，处理无符号 32 位整数向量分支无关除法
static inline __m128i libdivide_u32_branchfree_do_vector(__m128i numers, const struct libdivide_u32_branchfree_t *denom);

// 定义了 libdivide_s32_branchfree_do_vector 函数，处理有符号 32 位整数向量分支无关除法
static inline __m128i libdivide_s32_branchfree_do_vector(__m128i numers, const struct libdivide_s32_branchfree_t *denom);

// 定义了 libdivide_u64_branchfree_do_vector 函数，处理无符号 64 位整数向量分支无关除法
static inline __m128i libdivide_u64_branchfree_do_vector(__m128i numers, const struct libdivide_u64_branchfree_t *denom);

// 定义了 libdivide_s64_branchfree_do_vector 函数，处理有符号 64 位整数向量分支无关除法
static inline __m128i libdivide_s64_branchfree_do_vector(__m128i numers, const struct libdivide_s64_branchfree_t *denom);

//////// 内部实用函数

// 实现了 _mm_srai_epi64(v, 63) 的功能（来自 AVX512）
static inline __m128i libdivide_s64_signbits(__m128i v) {
    // 复制高位到每个位置
    __m128i hiBitsDuped = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));
    // 右移并提取符号位
    __m128i signBits = _mm_srai_epi32(hiBitsDuped, 31);
    return signBits;
}

// 实现了 _mm_srai_epi64 的功能（来自 AVX512）
static inline __m128i libdivide_s64_shift_right_vector(__m128i v, int amt) {
    const int b = 64 - amt;
    __m128i m = _mm_set1_epi64x(1ULL << (b - 1));
    __m128i x = _mm_srli_epi64(v, amt);
    __m128i result = _mm_sub_epi64(_mm_xor_si128(x, m), m);
    return result;
}

// 这里假设 b 包含一个重复的 32 位值
static inline __m128i libdivide_mullhi_u32_vector(__m128i a, __m128i b) {
    // 高位乘法结果，右移并合并
    __m128i hi_product_0Z2Z = _mm_srli_epi64(_mm_mul_epu32(a, b), 32);
    __m128i a1X3X = _mm_srli_epi64(a, 32);
    __m128i mask = _mm_set_epi32(-1, 0, -1, 0);
    __m128i hi_product_Z1Z3 = _mm_and_si128(_mm_mul_epu32(a1X3X, b), mask);
    return _mm_or_si128(hi_product_0Z2Z, hi_product_Z1Z3);
}

// SSE2 没有带符号乘法指令，但我们可以将无符号转换为带符号。这里假设 b 是一个重复的 32 位值
static inline __m128i libdivide_mullhi_s32_vector(__m128i a, __m128i b) {
    __m128i p = libdivide_mullhi_u32_vector(a, b);
    // t1 = (a >> 31) & y，算术右移
    __m128i t1 = _mm_and_si128(_mm_srai_epi32(a, 31), b);
    __m128i t2 = _mm_and_si128(_mm_srai_epi32(b, 31), a);
    p = _mm_sub_epi32(p, t1);
    p = _mm_sub_epi32(p, t2);
    return p;
}

// 这里假设 y 包含一个重复的 64 位值
static inline __m128i libdivide_mullhi_u64_vector(__m128i x, __m128i y) {
    __m128i lomask = _mm_set1_epi64x(0xffffffff);
    __m128i xh = _mm_shuffle_epi32(x, 0xB1);        // x0l, x0h, x1l, x1h
    __m128i yh = _mm_shuffle_epi32(y, 0xB1);        // y0l, y0h, y1l, y1h
    __m128i w0 = _mm_mul_epu32(x, y);               // x0l*y0l, x1l*y1l
    // 计算两个 64 位整数向量 x 和 y 的乘积的高位部分
    __m128i w1 = _mm_mul_epu32(x, yh);              // 计算 x0l*y0h, x1l*y1h
    __m128i w2 = _mm_mul_epu32(xh, y);              // 计算 x0h*y0l, x1h*y0l
    __m128i w3 = _mm_mul_epu32(xh, yh);             // 计算 x0h*y0h, x1h*y1h
    
    // 将 w0 向右移动 32 位，得到高位部分 w0h
    __m128i w0h = _mm_srli_epi64(w0, 32);
    
    // 计算 s1 = w1 + w0h
    __m128i s1 = _mm_add_epi64(w1, w0h);
    
    // 取 s1 的低位部分并与 lomask 进行按位与操作，得到 s1 的低位 s1l
    __m128i s1l = _mm_and_si128(s1, lomask);
    
    // 将 s1 向右移动 32 位，得到 s1 的高位 s1h
    __m128i s1h = _mm_srli_epi64(s1, 32);
    
    // 计算 s2 = w2 + s1l
    __m128i s2 = _mm_add_epi64(w2, s1l);
    
    // 将 s2 向右移动 32 位，得到 s2 的高位 s2h
    __m128i s2h = _mm_srli_epi64(s2, 32);
    
    // 计算 hi = w3 + s1h + s2h
    __m128i hi = _mm_add_epi64(w3, s1h);
    hi = _mm_add_epi64(hi, s2h);
    
    // 返回计算结果 hi
    return hi;
}

// 结束函数 libdivide_mullhi_s64_vector 的定义

// 计算有符号64位整数的乘法高位结果，返回128位整数结果
static inline __m128i libdivide_mullhi_s64_vector(__m128i x, __m128i y) {
    // 调用无符号64位整数乘法高位计算函数得到结果
    __m128i p = libdivide_mullhi_u64_vector(x, y);
    // 计算 x 和 y 的符号位，并与之前结果进行与运算
    __m128i t1 = _mm_and_si128(libdivide_s64_signbits(x), y);
    __m128i t2 = _mm_and_si128(libdivide_s64_signbits(y), x);
    // 结果减去符号位处理后的值
    p = _mm_sub_epi64(p, t1);
    p = _mm_sub_epi64(p, t2);
    // 返回最终结果
    return p;
}

////////// UINT32

// 执行32位无符号整数除法的向量化计算
__m128i libdivide_u32_do_vector(__m128i numers, const struct libdivide_u32_t *denom) {
    uint8_t more = denom->more;
    // 如果 magic 为零，直接右移操作数
    if (!denom->magic) {
        return _mm_srli_epi32(numers, more);
    }
    else {
        // 计算乘法高位结果
        __m128i q = libdivide_mullhi_u32_vector(numers, _mm_set1_epi32(denom->magic));
        // 如果 more 的标记位为 LIBDIVIDE_ADD_MARKER
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 计算 t = ((numer - q) >> 1) + q
            // 然后右移 shift 位
            uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
            __m128i t = _mm_add_epi32(_mm_srli_epi32(_mm_sub_epi32(numers, q), 1), q);
            return _mm_srli_epi32(t, shift);
        }
        else {
            // 否则直接右移 q
            return _mm_srli_epi32(q, more);
        }
    }
}

// 执行32位无符号整数分支无关的向量化除法计算
__m128i libdivide_u32_branchfree_do_vector(__m128i numers, const struct libdivide_u32_branchfree_t *denom) {
    // 计算乘法高位结果
    __m128i q = libdivide_mullhi_u32_vector(numers, _mm_set1_epi32(denom->magic));
    // 计算 t = ((numer - q) >> 1) + q，并右移 denom->more 位
    __m128i t = _mm_add_epi32(_mm_srli_epi32(_mm_sub_epi32(numers, q), 1), q);
    return _mm_srli_epi32(t, denom->more);
}

////////// UINT64

// 执行64位无符号整数除法的向量化计算
__m128i libdivide_u64_do_vector(__m128i numers, const struct libdivide_u64_t *denom) {
    uint8_t more = denom->more;
    // 如果 magic 为零，直接右移操作数
    if (!denom->magic) {
        return _mm_srli_epi64(numers, more);
    }
    else {
        // 计算乘法高位结果
        __m128i q = libdivide_mullhi_u64_vector(numers, _mm_set1_epi64x(denom->magic));
        // 如果 more 的标记位为 LIBDIVIDE_ADD_MARKER
        if (more & LIBDIVIDE_ADD_MARKER) {
            // 计算 t = ((numer - q) >> 1) + q
            // 然后右移 shift 位
            uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;
            __m128i t = _mm_add_epi64(_mm_srli_epi64(_mm_sub_epi64(numers, q), 1), q);
            return _mm_srli_epi64(t, shift);
        }
        else {
            // 否则直接右移 q
            return _mm_srli_epi64(q, more);
        }
    }
}

// 执行64位无符号整数分支无关的向量化除法计算
__m128i libdivide_u64_branchfree_do_vector(__m128i numers, const struct libdivide_u64_branchfree_t *denom) {
    // 计算乘法高位结果
    __m128i q = libdivide_mullhi_u64_vector(numers, _mm_set1_epi64x(denom->magic));
    // 计算 t = ((numer - q) >> 1) + q，并右移 denom->more 位
    __m128i t = _mm_add_epi64(_mm_srli_epi64(_mm_sub_epi64(numers, q), 1), q);
    return _mm_srli_epi64(t, denom->more);
}

////////// SINT32

// 执行32位有符号整数除法的向量化计算
__m128i libdivide_s32_do_vector(__m128i numers, const struct libdivide_s32_t *denom) {
    uint8_t more = denom->more;
    // 检查分母的魔数是否为零
    if (!denom->magic) {
        // 从 more 中获取 LIBDIVIDE_32_SHIFT_MASK，用作移位操作的位数
        uint32_t shift = more & LIBDIVIDE_32_SHIFT_MASK;
        // 创建一个 mask，用于获取最低 shift 位的掩码
        uint32_t mask = (1U << shift) - 1;
        // 创建一个 __m128i 类型的常数，用 mask 初始化，用于舍入向零
        __m128i roundToZeroTweak = _mm_set1_epi32(mask);
        // 计算 q = numer + ((numer >> 31) & roundToZeroTweak);
        __m128i q = _mm_add_epi32(numers, _mm_and_si128(_mm_srai_epi32(numers, 31), roundToZeroTweak));
        // 对 q 进行算术右移 shift 位
        q = _mm_srai_epi32(q, shift);
        // 创建一个 sign 常数，从 more 中获取并符号扩展
        __m128i sign = _mm_set1_epi32((int8_t)more >> 7);
        // 计算 q = (q ^ sign) - sign;
        q = _mm_sub_epi32(_mm_xor_si128(q, sign), sign);
        // 返回计算结果 q
        return q;
    }
    else {
        // 使用 libdivide_mullhi_s32_vector 计算 q = numer * denom->magic 的高位
        __m128i q = libdivide_mullhi_s32_vector(numers, _mm_set1_epi32(denom->magic));
        // 检查是否需要添加标记
        if (more & LIBDIVIDE_ADD_MARKER) {
             // 从 more 中获取符号位并扩展为整数
            __m128i sign = _mm_set1_epi32((int8_t)more >> 7);
             // q += ((numer ^ sign) - sign);
            q = _mm_add_epi32(q, _mm_sub_epi32(_mm_xor_si128(numers, sign), sign));
        }
        // 对 q 进行算术右移，从 more 中获取移位的位数
        q = _mm_srai_epi32(q, more & LIBDIVIDE_32_SHIFT_MASK);
        // 如果 q < 0，则 q += 1（将符号位扩展到更高位）
        q = _mm_add_epi32(q, _mm_srli_epi32(q, 31)); // q += (q < 0)
        // 返回计算结果 q
        return q;
    }
}

__m128i libdivide_s32_branchfree_do_vector(__m128i numers, const struct libdivide_s32_branchfree_t *denom) {
    int32_t magic = denom->magic;  // 从结构体中获取魔数
    uint8_t more = denom->more;  // 从结构体中获取更多信息
    uint8_t shift = more & LIBDIVIDE_32_SHIFT_MASK;  // 使用位掩码提取移位量

     // 必须是算术右移
    __m128i sign = _mm_set1_epi32((int8_t)more >> 7);  // 设置符号位掩码
    __m128i q = libdivide_mullhi_s32_vector(numers, _mm_set1_epi32(magic));  // 计算乘法的高位部分
    q = _mm_add_epi32(q, numers); // q += numers

    // 如果 q 是非负数，则不需要处理
    // 如果 q 是负数，我们希望根据 d 是否为2的幂，添加 (2**shift)-1 或者 (2**shift)
    uint32_t is_power_of_2 = (magic == 0);  // 判断魔数是否为0
    __m128i q_sign = _mm_srai_epi32(q, 31); // q_sign = q >> 31
    __m128i mask = _mm_set1_epi32((1U << shift) - is_power_of_2);  // 根据移位量生成掩码
    q = _mm_add_epi32(q, _mm_and_si128(q_sign, mask)); // q = q + (q_sign & mask)
    q = _mm_srai_epi32(q, shift); // q >>= shift
    q = _mm_sub_epi32(_mm_xor_si128(q, sign), sign); // q = (q ^ sign) - sign
    return q;
}

////////// SINT64

__m128i libdivide_s64_do_vector(__m128i numers, const struct libdivide_s64_t *denom) {
    uint8_t more = denom->more;  // 从结构体中获取更多信息
    int64_t magic = denom->magic;  // 从结构体中获取魔数
    if (magic == 0) { // shift path 如果魔数为0，则使用移位路径
        uint32_t shift = more & LIBDIVIDE_64_SHIFT_MASK;  // 使用位掩码提取移位量
        uint64_t mask = (1ULL << shift) - 1;  // 根据移位量生成掩码
        __m128i roundToZeroTweak = _mm_set1_epi64x(mask);  // 设置舍入到零的调整值

        // q = numer + ((numer >> 63) & roundToZeroTweak);
        __m128i q = _mm_add_epi64(numers, _mm_and_si128(libdivide_s64_signbits(numers), roundToZeroTweak));  // 执行加法

        q = libdivide_s64_shift_right_vector(q, shift);  // 右移操作
        __m128i sign = _mm_set1_epi32((int8_t)more >> 7);  // 设置符号位掩码
         // q = (q ^ sign) - sign;
        q = _mm_sub_epi64(_mm_xor_si128(q, sign), sign);  // 执行减法
        return q;
    }
    else {
        __m128i q = libdivide_mullhi_s64_vector(numers, _mm_set1_epi64x(magic));  // 执行乘法的高位计算
        if (more & LIBDIVIDE_ADD_MARKER) {  // 如果设置了加法标记
            // 必须是算术右移
            __m128i sign = _mm_set1_epi32((int8_t)more >> 7);  // 设置符号位掩码
            // q += ((numer ^ sign) - sign);
            q = _mm_add_epi64(q, _mm_sub_epi64(_mm_xor_si128(numers, sign), sign));  // 执行加法
        }
        // q >>= denom->mult_path.shift
        q = libdivide_s64_shift_right_vector(q, more & LIBDIVIDE_64_SHIFT_MASK);  // 右移操作
        q = _mm_add_epi64(q, _mm_srli_epi64(q, 63)); // q += (q < 0)
        return q;
    }
}

__m128i libdivide_s64_branchfree_do_vector(__m128i numers, const struct libdivide_s64_branchfree_t *denom) {
    int64_t magic = denom->magic;  // 从结构体中获取魔数
    uint8_t more = denom->more;  // 从结构体中获取更多信息
    uint8_t shift = more & LIBDIVIDE_64_SHIFT_MASK;  // 使用位掩码提取移位量
    // 必须是算术右移
    __m128i sign = _mm_set1_epi32((int8_t)more >> 7);  // 设置符号位掩码

     // libdivide_mullhi_s64(numers, magic);
    __m128i q = libdivide_mullhi_s64_vector(numers, _mm_set1_epi64x(magic));  // 执行乘法的高位计算
    q = _mm_add_epi64(q, numers); // q += numers

    // 如果 q 是非负数，则不需要处理。
    // 如果 q 是负数，我们希望根据 d 是否为2的幂，添加 (2**shift)-1 if d is
    // 检查 magic 是否为 0，如果是，则 is_power_of_2 为 1，否则为 0
    uint32_t is_power_of_2 = (magic == 0);
    // 计算 q 的符号位，并右移 63 位得到 q_sign
    __m128i q_sign = libdivide_s64_signbits(q);
    // 创建一个掩码 mask，用于对 q 进行修正，使其为 2 的 shift 次幂
    __m128i mask = _mm_set1_epi64x((1ULL << shift) - is_power_of_2);
    // 将 q 和 (q_sign & mask) 进行按位与操作后加到 q 上
    q = _mm_add_epi64(q, _mm_and_si128(q_sign, mask));
    // 对 q 进行右移 shift 位操作
    q = libdivide_s64_shift_right_vector(q, shift);
    // 对 q 执行异或运算后减去 sign，得到最终结果
    q = _mm_sub_epi64(_mm_xor_si128(q, sign), sign);
    // 返回处理后的 q
    return q;
#ifdef __cplusplus

// 如果正在使用 C++ 编译器，则进入 C++ 相关的部分


// The C++ divider class is templated on both an integer type
// (like uint64_t) and an algorithm type.
// * BRANCHFULL is the default algorithm type.
// * BRANCHFREE is the branchfree algorithm type.

// 定义了一个 C++ 分割器类，该类模板化于整数类型（如 uint64_t）和算法类型。
// * BRANCHFULL 是默认的算法类型。
// * BRANCHFREE 是无分支算法类型。


enum {
    BRANCHFULL,
    BRANCHFREE
};

// 定义了两个枚举常量 BRANCHFULL 和 BRANCHFREE，分别代表默认算法和无分支算法。


#if defined(LIBDIVIDE_AVX512)
    #define LIBDIVIDE_VECTOR_TYPE __m512i
#elif defined(LIBDIVIDE_AVX2)
    #define LIBDIVIDE_VECTOR_TYPE __m256i
#elif defined(LIBDIVIDE_SSE2)
    #define LIBDIVIDE_VECTOR_TYPE __m128i
#endif

// 根据编译器定义的 SIMD 扩展，选择适当的向量类型。


#if !defined(LIBDIVIDE_VECTOR_TYPE)
    #define LIBDIVIDE_DIVIDE_VECTOR(ALGO)
#else
    #define LIBDIVIDE_DIVIDE_VECTOR(ALGO) \
        LIBDIVIDE_VECTOR_TYPE divide(LIBDIVIDE_VECTOR_TYPE n) const { \
            return libdivide_##ALGO##_do_vector(n, &denom); \
        }
#endif

// 如果未定义向量类型，定义一个空的宏 LIBDIVIDE_DIVIDE_VECTOR(ALGO)，否则定义一个返回 SIMD 向量的宏，使用 libdivide 库中指定算法 ALGO 处理向量 n。


#define DISPATCHER_GEN(T, ALGO) \
    libdivide_##ALGO##_t denom; \
    dispatcher() { } \
    dispatcher(T d) \
        : denom(libdivide_##ALGO##_gen(d)) \
    { } \
    T divide(T n) const { \
        return libdivide_##ALGO##_do(n, &denom); \
    } \
    LIBDIVIDE_DIVIDE_VECTOR(ALGO) \
    T recover() const { \
        return libdivide_##ALGO##_recover(&denom); \
    }

// 宏 DISPATCHER_GEN(T, ALGO) 生成基于类型 T 和算法 ALGO 的 C++ 方法，这些方法重定向到 libdivide 的 C API。


template<bool IS_INTEGRAL, bool IS_SIGNED, int SIZEOF, int ALGO> struct dispatcher { };

// dispatcher 结构模板，根据 IS_INTEGRAL（是否整数）、IS_SIGNED（是否有符号）、SIZEOF（字节大小）、ALGO（算法类型）选择特定的分派器。


template<> struct dispatcher<true, true, sizeof(int32_t), BRANCHFULL> { DISPATCHER_GEN(int32_t, s32) };
template<> struct dispatcher<true, true, sizeof(int32_t), BRANCHFREE> { DISPATCHER_GEN(int32_t, s32_branchfree) };
template<> struct dispatcher<true, false, sizeof(uint32_t), BRANCHFULL> { DISPATCHER_GEN(uint32_t, u32) };
template<> struct dispatcher<true, false, sizeof(uint32_t), BRANCHFREE> { DISPATCHER_GEN(uint32_t, u32_branchfree) };
template<> struct dispatcher<true, true, sizeof(int64_t), BRANCHFULL> { DISPATCHER_GEN(int64_t, s64) };
template<> struct dispatcher<true, true, sizeof(int64_t), BRANCHFREE> { DISPATCHER_GEN(int64_t, s64_branchfree) };
template<> struct dispatcher<true, false, sizeof(uint64_t), BRANCHFULL> { DISPATCHER_GEN(uint64_t, u64) };
template<> struct dispatcher<true, false, sizeof(uint64_t), BRANCHFREE> { DISPATCHER_GEN(uint64_t, u64_branchfree) };

// 部分模板特化，根据整数和算法类型选择相应的 dispatcher 结构模板，并使用 DISPATCHER_GEN 宏生成相应的方法。


template<typename T, int ALGO = BRANCHFULL>
class divider {
public:
    divider() { }
    divider(T d) : div(d) { }
    T divide(T n) const {
        return libdivide_##ALGO##_do(n, &denom);
    }
    LIBDIVIDE_DIVIDE_VECTOR(ALGO)
    T recover() const {
        return libdivide_##ALGO##_recover(&denom);
    }

// divider 类模板，用于用户使用（C++ API），根据整数类型 T 和算法 ALGO 选择分派器，提供除法和向量除法功能。
    // 调用 divide 方法，使用当前对象中的 div 对象对 n 进行除法运算，返回结果
    T divide(T n) const {
        return div.divide(n);
    }

    // 调用 recover 方法，返回当前对象中的 div 对象所使用的初始化值
    // 这个值被用于初始化这个 divider 对象
    T recover() const {
        return div.recover();
    }

    // 重载 == 操作符，比较两个 divider 对象是否相等
    // 当且仅当两个对象的 div 对象的 denom 成员的 magic 和 more 成员都相等时返回 true
    bool operator==(const divider<T, ALGO>& other) const {
        return div.denom.magic == other.denom.magic &&
               div.denom.more == other.denom.more;
    }

    // 重载 != 操作符，比较两个 divider 对象是否不相等
    // 当两个对象使用 == 操作符返回 false 时返回 true，否则返回 false
    bool operator!=(const divider<T, ALGO>& other) const {
        return !(*this == other);
    }
#if defined(LIBDIVIDE_VECTOR_TYPE)
    // 如果定义了 LIBDIVIDE_VECTOR_TYPE 宏，则编译以下代码块

    // Treats the vector as packed integer values with the same type as
    // the divider (e.g. s32, u32, s64, u64) and divides each of
    // them by the divider, returning the packed quotients.
    // 将向量视为打包的整数值，其类型与除数相同（例如 s32, u32, s64, u64），
    // 并将每个值除以除数，返回打包后的商。

    LIBDIVIDE_VECTOR_TYPE divide(LIBDIVIDE_VECTOR_TYPE n) const {
        return div.divide(n);
        // 调用 div 对象的 divide 方法来执行向量的除法操作，并返回结果
    }
#endif

private:
    // Storage for the actual divisor
    // 实际除数的存储
    dispatcher<std::is_integral<T>::value,
               std::is_signed<T>::value, sizeof(T), ALGO> div;
    // 使用模板类 dispatcher 存储实际的除数，根据模板参数 T、ALGO 确定具体类型和算法。
};

// Overload of operator / for scalar division
// 标量除法运算符重载
template<typename T, int ALGO>
T operator/(T n, const divider<T, ALGO>& div) {
    return div.divide(n);
    // 调用 div 对象的 divide 方法执行标量的除法操作，并返回结果
}

// Overload of operator /= for scalar division
// 标量除法赋值运算符重载
template<typename T, int ALGO>
T& operator/=(T& n, const divider<T, ALGO>& div) {
    n = div.divide(n);
    return n;
    // 调用 div 对象的 divide 方法执行标量的除法操作，并将结果赋值给 n 后返回 n
}

#if defined(LIBDIVIDE_VECTOR_TYPE)
    // Overload of operator / for vector division
    // 向量除法运算符重载
    template<typename T, int ALGO>
    LIBDIVIDE_VECTOR_TYPE operator/(LIBDIVIDE_VECTOR_TYPE n, const divider<T, ALGO>& div) {
        return div.divide(n);
        // 调用 div 对象的 divide 方法执行向量的除法操作，并返回结果
    }
    // Overload of operator /= for vector division
    // 向量除法赋值运算符重载
    template<typename T, int ALGO>
    LIBDIVIDE_VECTOR_TYPE& operator/=(LIBDIVIDE_VECTOR_TYPE& n, const divider<T, ALGO>& div) {
        n = div.divide(n);
        return n;
        // 调用 div 对象的 divide 方法执行向量的除法操作，并将结果赋值给 n 后返回 n
    }
#endif

// libdivdie::branchfree_divider<T>
// libdivide 命名空间中的 branchfree_divider<T> 别名定义
template <typename T>
using branchfree_divider = divider<T, BRANCHFREE>;

}  // namespace libdivide

#endif  // __cplusplus

#endif  // NUMPY_CORE_INCLUDE_NUMPY_LIBDIVIDE_LIBDIVIDE_H_
```