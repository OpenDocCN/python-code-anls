# `D:\src\scipysrc\numpy\numpy\_core\src\common\float_status.hpp`

```
#ifndef NUMPY_CORE_SRC_COMMON_FLOAT_STATUS_HPP
#define NUMPY_CORE_SRC_COMMON_FLOAT_STATUS_HPP

#include "npstd.hpp"

#include <fenv.h>

namespace np {

/// @addtogroup cpp_core_utility
/// @{
/**
 * Class wraps floating-point environment operations,
 * provides lazy access to its functionality.
 */
class FloatStatus {
 public:
/*
 * According to the C99 standard FE_DIVBYZERO, etc. may not be provided when
 * unsupported.  In such cases NumPy will not report these correctly, but we
 * should still allow compiling (whether tests pass or not).
 * By defining them as 0 locally, we make them no-ops.  Unlike these defines,
 * for example `musl` still defines all of the functions (as no-ops):
 *     https://git.musl-libc.org/cgit/musl/tree/src/fenv/fenv.c
 * and does similar replacement in its tests:
 * http://nsz.repo.hu/git/?p=libc-test;a=blob;f=src/common/mtest.h;h=706c1ba23ea8989b17a2f72ed1a919e187c06b6a;hb=HEAD#l30
 */

#ifdef FE_DIVBYZERO
    // If FE_DIVBYZERO is defined in <fenv.h>, use its value; otherwise, use 0
    static constexpr int kDivideByZero = FE_DIVBYZERO;
#else
    static constexpr int kDivideByZero = 0;
#endif

#ifdef FE_INVALID
    // If FE_INVALID is defined in <fenv.h>, use its value; otherwise, use 0
    static constexpr int kInvalid = FE_INVALID;
#else
    static constexpr int kInvalid = 0;
#endif

#ifdef FE_INEXACT
    // If FE_INEXACT is defined in <fenv.h>, use its value; otherwise, use 0
    static constexpr int kInexact = FE_INEXACT;
#else
    static constexpr int kInexact = 0;
#endif

#ifdef FE_OVERFLOW
    // If FE_OVERFLOW is defined in <fenv.h>, use its value; otherwise, use 0
    static constexpr int kOverflow = FE_OVERFLOW;
#else
    static constexpr int kOverflow = 0;
#endif

#ifdef FE_UNDERFLOW
    // If FE_UNDERFLOW is defined in <fenv.h>, use its value; otherwise, use 0
    static constexpr int kUnderflow = FE_UNDERFLOW;
#else
    static constexpr int kUnderflow = 0;
#endif

    // Calculate the bitwise OR of all supported floating-point exceptions
    static constexpr int kAllExcept = (kDivideByZero | kInvalid | kInexact |
                                       kOverflow | kUnderflow);

    // Constructor initializes the floating-point status
    FloatStatus(bool clear_on_dst=true)
        : clear_on_dst_(clear_on_dst)
    {
        // If any floating-point exceptions are supported, fetch the current status
        if constexpr (kAllExcept != 0) {
            fpstatus_ = fetestexcept(kAllExcept);
        }
        else {
            fpstatus_ = 0;
        }
    }

    // Destructor clears floating-point exceptions if required
    ~FloatStatus()
    {
        // If any floating-point exceptions are supported and set, clear them
        if constexpr (kAllExcept != 0) {
            if (fpstatus_ != 0 && clear_on_dst_) {
                feclearexcept(kAllExcept);
            }
        }
    }

    // Check if Divide By Zero exception is set
    constexpr bool IsDivideByZero() const
    {
        return (fpstatus_ & kDivideByZero) != 0;
    }

    // Check if Inexact exception is set
    constexpr bool IsInexact() const
    {
        return (fpstatus_ & kInexact) != 0;
    }

    // Check if Invalid exception is set
    constexpr bool IsInvalid() const
    {
        return (fpstatus_ & kInvalid) != 0;
    }

    // Check if Overflow exception is set
    constexpr bool IsOverFlow() const
    {
        return (fpstatus_ & kOverflow) != 0;
    }

    // Check if Underflow exception is set
    constexpr bool IsUnderFlow() const
    {
        return (fpstatus_ & kUnderflow) != 0;
    }

    // Raise Divide By Zero exception
    static void RaiseDivideByZero()
    {
        if constexpr (kDivideByZero != 0) {
            feraiseexcept(kDivideByZero);
        }
    }

    // Raise Inexact exception
    static void RaiseInexact()
    {
        if constexpr (kInexact != 0) {
            feraiseexcept(kInexact);
        }
    }

    // Raise Invalid exception
    static void RaiseInvalid()
    {
        if constexpr (kInvalid != 0) {
            feraiseexcept(kInvalid);
        }
    }

    // End of class definition for FloatStatus
#endif
    }
    # 静态函数：触发溢出异常
    static void RaiseOverflow()
    {
        # 如果 kOverflow 不为零，则触发对应的浮点异常
        if constexpr (kOverflow != 0) {
            feraiseexcept(kOverflow);
        }
    }
    # 静态函数：触发下溢异常
    static void RaiseUnderflow()
    {
        # 如果 kUnderflow 不为零，则触发对应的浮点异常
        if constexpr (kUnderflow != 0) {
            feraiseexcept(kUnderflow);
        }
    }

  private:
    # 清除目标上的状态标志
    bool clear_on_dst_;
    # 浮点运算状态
    int fpstatus_;
};

/// @} cpp_core_utility
// 结束命名空间 np

} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_FLOAT_STATUS_HPP
// 结束条件编译指令 NUMPY_CORE_SRC_COMMON_FLOAT_STATUS_HPP
```