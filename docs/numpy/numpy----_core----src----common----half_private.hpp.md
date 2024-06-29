# `D:\src\scipysrc\numpy\numpy\_core\src\common\half_private.hpp`

```
#ifndef NUMPY_CORE_SRC_COMMON_HALF_PRIVATE_HPP
#define NUMPY_CORE_SRC_COMMON_HALF_PRIVATE_HPP

#include "npstd.hpp"
#include "float_status.hpp"

/*
 * The following functions that emulating float/double/half conversions
 * are copied from npymath without any changes to its functionality.
 */
namespace np { namespace half_private {

template<bool gen_overflow=true, bool gen_underflow=true, bool round_even=true>
inline uint16_t FromFloatBits(uint32_t f)
{
    uint32_t f_exp, f_sig;
    uint16_t h_sgn, h_exp, h_sig;

    // Extract the sign bit from the float representation and move it to the appropriate position in half precision
    h_sgn = (uint16_t) ((f&0x80000000u) >> 16);
    // Extract the exponent bits from the float representation
    f_exp = (f&0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            // Extract the significand bits from the float representation
            f_sig = (f&0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                // Convert NaN to half precision format while preserving the NaN flag
                uint16_t ret = (uint16_t) (0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (uint16_t) (h_sgn + 0x7c00u);
            }
        } else {
            if constexpr (gen_overflow) {
                /* overflow to signed inf */
                // Raise overflow exception if enabled
                FloatStatus::RaiseOverflow();
            }
            return (uint16_t) (h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    // 检查浮点数是否小于或等于 0x38000000u
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        // 处理有符号零、次正规化浮点数和小指数浮点数，它们都转换为有符号零的半精度浮点数
        if (f_exp < 0x33000000u) {
            if constexpr (gen_underflow) {
                /* 如果 f 不等于 0，则发生下溢到 0 */
                if ((f&0x7fffffff) != 0) {
                    FloatStatus::RaiseUnderflow();
                }
            }
            // 返回半精度浮点数的符号部分
            return h_sgn;
        }
        /* 构造次正规化数的尾数 */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f&0x007fffffu));
        if constexpr (gen_underflow) {
            /* 如果它不是精确表示，则发生下溢 */
            if ((f_sig&(((uint32_t)1 << (126 - f_exp)) - 1)) != 0) {
                FloatStatus::RaiseUnderflow();
            }
        }
        /*
         * 通常尾数向右移动 13 位。对于次正规化数，需要额外的移动。
         * 这个移动是为了在最大指数给出次正规化的情况下 `f_exp = 0x38000000 >> 23 = 112`，
         * 这会偏移新的第一位。最多移动 1+10 位。
         */
        f_sig >>= (113 - f_exp);
        /* 处理四舍五入，通过将超过半精度的位数加 1 */
        if constexpr (round_even) {
            /*
             * 如果半精度尾数的最后一位是 0（已经是偶数），并且剩余的位模式是 1000...0，
             * 则我们不会向半精度尾数后的位添加一。然而，(113 - f_exp) 移位最多可能丢失 11 位，
             * 所以 || 检查它们在原始中。在所有其他情况下，我们可以直接加一。
             */
            if (((f_sig&0x00003fffu) != 0x00001000u) || (f&0x000007ffu)) {
                f_sig += 0x00001000u;
            }
        }
        else {
            f_sig += 0x00001000u;
        }
        h_sig = (uint16_t) (f_sig >> 13);
        /*
         * 如果舍入导致一个位溢出到 h_exp，它将从零增加到一，并且 h_sig 将为零。
         * 这是正确的结果。
         */
        return (uint16_t) (h_sgn + h_sig);
    }

    /* 常规情况，没有溢出或下溢 */
    h_exp = (uint16_t) ((f_exp - 0x38000000u) >> 13);
    /* 处理四舍五入，通过将超过半精度的位数加 1 */
    f_sig = (f&0x007fffffu);
    if constexpr (round_even) {
        /*
         * 如果半精度尾数的最后一位是 0（已经是偶数），并且剩余的位模式是 1000...0，
         * 则我们不会向半精度尾数后的位添加一。在所有其他情况下，我们会添加一。
         */
        if ((f_sig&0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }
    }
    else {
        f_sig += 0x00001000u;
    }
    h_sig = (uint16_t) (f_sig >> 13);
    /*
     * 如果四舍五入导致一个位溢出到 h_exp 中，h_exp 将增加一，并且 h_sig 将变为零。
     * 这是正确的结果。h_exp 最多可能增加到 15，此时结果将溢出到有符号的无穷大。
     */
    if constexpr (gen_overflow) {
        // 如果生成的结果溢出，将 h_sig 加到 h_exp 上
        h_sig += h_exp;
        // 如果 h_sig 等于 0x7c00u（十进制为 31744），表示结果溢出到正无穷大
        if (h_sig == 0x7c00u) {
            // 触发浮点溢出状态
            FloatStatus::RaiseOverflow();
        }
        // 返回带符号的结果：符号位 + h_sig
        return h_sgn + h_sig;
    }
    else {
        // 返回带符号的结果：符号位 + h_exp + h_sig
        return h_sgn + h_exp + h_sig;
    }
    // 从双精度浮点数表示中提取出半精度浮点数
    template<bool gen_overflow=true, bool gen_underflow=true, bool round_even=true>
    inline uint16_t FromDoubleBits(uint64_t d)
    {
        uint64_t d_exp, d_sig;
        uint16_t h_sgn, h_exp, h_sig;

        // 提取符号位并右移至对应半精度浮点数的位置
        h_sgn = (d & 0x8000000000000000ULL) >> 48;
        // 提取双精度浮点数的指数位
        d_exp = (d & 0x7ff0000000000000ULL);

        /* Exponent overflow/NaN converts to signed inf/NaN */
        // 如果指数大于等于最大半精度浮点数的指数范围，则处理溢出或 NaN
        if (d_exp >= 0x40f0000000000000ULL) {
            if (d_exp == 0x7ff0000000000000ULL) {
                /* Inf or NaN */
                // 提取双精度浮点数的有效位
                d_sig = (d & 0x000fffffffffffffULL);
                if (d_sig != 0) {
                    /* NaN - 将标志传播到尾数... */
                    uint16_t ret = (uint16_t)(0x7c00u + (d_sig >> 42));
                    /* ...但确保它仍然是 NaN */
                    if (ret == 0x7c00u) {
                        ret++;
                    }
                    return h_sgn + ret;
                } else {
                    /* signed inf */
                    // 返回带符号的无穷大
                    return h_sgn + 0x7c00u;
                }
            } else {
                /* overflow to signed inf */
                // 指数溢出转换为带符号的无穷大
                if constexpr (gen_overflow) {
                    FloatStatus::RaiseOverflow();  // 如果允许溢出生成，则引发溢出状态
                }
                return h_sgn + 0x7c00u;
            }
        }

        /* Exponent underflow converts to subnormal half or signed zero */
        // 指数下溢转换为次正规化的半精度浮点数或带符号零
    // 如果双精度浮点数的指数小于或等于 0x3f00000000000000ULL
    if (d_exp <= 0x3f00000000000000ULL) {
        /*
         * 对于有符号零、次标准浮点数和指数较小的浮点数，它们都转换为有符号零的半精度浮点数。
         */
        if (d_exp < 0x3e60000000000000ULL) {
            // 如果生成下溢时为真
            if constexpr (gen_underflow) {
                /* 如果 d 不等于 0，则它下溢到 0 */
                if ((d&0x7fffffffffffffffULL) != 0) {
                    FloatStatus::RaiseUnderflow();
                }
            }
            // 返回有符号的零半精度浮点数
            return h_sgn;
        }
        /* 构造次标准的尾数 */
        d_exp >>= 52;
        d_sig = (0x0010000000000000ULL + (d&0x000fffffffffffffULL));
        // 如果生成下溢时为真
        if constexpr (gen_underflow) {
            /* 如果它不能被精确表示，则它下溢了 */
            if ((d_sig&(((uint64_t)1 << (1051 - d_exp)) - 1)) != 0) {
                FloatStatus::RaiseUnderflow();
            }
        }
        /*
         * 不像浮点数，双精度浮点数有足够的空间将尾数左移以对齐次标准尾数，不会丢失最后的位。
         * 给出次标准的最小可能指数是：
         * `d_exp = 0x3e60000000000000 >> 52 = 998`。所有更大的次标准都相对于它做了偏移。
         * 这在与正常分支中的右移比较时增加了 10+1 位的偏移。
         */
        assert(d_exp - 998 >= 0);
        d_sig <<= (d_exp - 998);
        /* 通过在超过半精度后添加 1 来处理舍入 */
        if constexpr (round_even) {
            /*
             * 如果半精度尾数中的最后一位是 0（已经是偶数），并且剩余的位模式是 1000...0，
             * 那么我们不在半精度尾数后面加一。在所有其他情况下，我们加一。
             */
            if ((d_sig&0x003fffffffffffffULL) != 0x0010000000000000ULL) {
                d_sig += 0x0010000000000000ULL;
            }
        }
        else {
            d_sig += 0x0010000000000000ULL;
        }
        h_sig = (uint16_t) (d_sig >> 53);
        /*
         * 如果舍入导致位溢出到 h_exp，它将从零增加到一，并且 h_sig 将为零。
         * 这是正确的结果。
         */
        return h_sgn + h_sig;
    }

    /* 普通情况，没有溢出或下溢 */
    h_exp = (uint16_t) ((d_exp - 0x3f00000000000000ULL) >> 42);
    /* 通过在超过半精度后添加 1 来处理舍入 */
    d_sig = (d&0x000fffffffffffffULL);
    if constexpr (round_even) {
        /*
         * 如果半精度尾数中的最后一位是 0（已经是偶数），并且剩余的位模式是 1000...0，
         * 那么我们不在半精度尾数后面加一。在所有其他情况下，我们加一。
         */
        if ((d_sig&0x000007ffffffffffULL) != 0x0000020000000000ULL) {
            d_sig += 0x0000020000000000ULL;
        }
    }
    else {
        d_sig += 0x0000020000000000ULL;
    }
    // 将 d_sig 右移 42 位，并将结果转换为 uint16_t 类型，存入 h_sig 中
    h_sig = (uint16_t) (d_sig >> 42);

    /*
     * 如果舍入导致一个比特溢出到 h_exp 中，h_exp 将增加一，并且 h_sig 将变为零。
     * 这是正确的结果。h_exp 最多可能增加到 15，此时结果会溢出为带符号的无穷大。
     */
    // 如果编译时条件 gen_overflow 为真
    if constexpr (gen_overflow) {
        // 将 h_sig 与 h_exp 相加
        h_sig += h_exp;
        // 如果 h_sig 等于 0x7c00u
        if (h_sig == 0x7c00u) {
            // 触发浮点溢出异常
            FloatStatus::RaiseOverflow();
        }
        // 返回 h_sgn + h_sig 的结果
        return h_sgn + h_sig;
    }
    else {
        // 返回 h_sgn + h_exp + h_sig 的结果
        return h_sgn + h_exp + h_sig;
    }
constexpr uint32_t ToFloatBits(uint16_t h)
{
    // 提取 h 的指数部分
    uint16_t h_exp = (h&0x7c00u);
    // 提取 h 的符号位，并左移 16 位
    uint32_t f_sgn = ((uint32_t)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: { // 0 或者亚正规化数
            // 提取 h 的尾数部分
            uint16_t h_sig = (h&0x03ffu);
            // 如果尾数部分为零，返回符号位
            if (h_sig == 0) {
                return f_sgn;
            }
            // 亚正规化数，尾数部分左移一位直到最高位为 1
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            // 计算浮点数的指数部分，尾数部分，和符号位
            uint32_t f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
            uint32_t f_sig = ((uint32_t)(h_sig&0x03ffu)) << 13;
            return f_sgn + f_exp + f_sig;
        }
        case 0x7c00u: // 无穷大或者 NaN
            // 全 1 的指数部分和尾数部分的拷贝
            return f_sgn + 0x7f800000u + (((uint32_t)(h&0x03ffu)) << 13);
        default: // 规格化数
            // 只需要调整指数并左移
            return f_sgn + (((uint32_t)(h&0x7fffu) + 0x1c000u) << 13);
    }
}

constexpr uint64_t ToDoubleBits(uint16_t h)
{
    // 提取 h 的指数部分
    uint16_t h_exp = (h&0x7c00u);
    // 提取 h 的符号位，并左移 48 位
    uint64_t d_sgn = ((uint64_t)h&0x8000u) << 48;
    switch (h_exp) {
        case 0x0000u: { // 0 或者亚正规化数
            // 提取 h 的尾数部分
            uint16_t h_sig = (h&0x03ffu);
            // 如果尾数部分为零，返回符号位
            if (h_sig == 0) {
                return d_sgn;
            }
            // 亚正规化数，尾数部分左移一位直到最高位为 1
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            // 计算双精度浮点数的指数部分，尾数部分，和符号位
            uint64_t d_exp = ((uint64_t)(1023 - 15 - h_exp)) << 52;
            uint64_t d_sig = ((uint64_t)(h_sig&0x03ffu)) << 42;
            return d_sgn + d_exp + d_sig;
        }
        case 0x7c00u: // 无穷大或者 NaN
            // 全 1 的指数部分和尾数部分的拷贝
            return d_sgn + 0x7ff0000000000000ULL + (((uint64_t)(h&0x03ffu)) << 42);
        default: // 规格化数
            // 只需要调整指数并左移
            return d_sgn + (((uint64_t)(h&0x7fffu) + 0xfc000u) << 42);
    }
}
```