# `.\pytorch\c10\util\MathConstants.cpp`

```
// 如果在使用 MSVC 编译器环境下，并且没有定义 _USE_MATH_DEFINES 宏，则定义 _USE_MATH_DEFINES 宏
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif

// 引入 c10 库中的 MathConstants.h 文件，该文件包含了常量如 M_PI 和 C 的定义
#include <c10/util/MathConstants.h>

// NOLINTNEXTLINE(modernize-deprecated-headers)
// 引入 math.h 头文件，提供数学函数和宏定义，虽然被标记为过时但仍然使用
#include <math.h>

// 静态断言：确保 c10::pi<double> 的值等于 M_PI
static_assert(M_PI == c10::pi<double>, "c10::pi<double> must be equal to M_PI");

// 静态断言：确保 c10::frac_sqrt_2<double> 的值等于 M_SQRT1_2
static_assert(
    M_SQRT1_2 == c10::frac_sqrt_2<double>,
    "c10::frac_sqrt_2<double> must be equal to M_SQRT1_2");
```