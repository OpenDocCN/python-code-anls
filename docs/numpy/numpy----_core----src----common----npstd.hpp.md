# `D:\src\scipysrc\numpy\numpy\_core\src\common\npstd.hpp`

```py
#ifndef NUMPY_CORE_SRC_COMMON_NPSTD_HPP
#define NUMPY_CORE_SRC_COMMON_NPSTD_HPP

#include <cstddef>      // C标准库定义的常用宏和类型
#include <cstring>      // C字符串操作函数
#include <cctype>       // C字符处理函数
#include <cstdint>      // C标准整数类型

#include <string>       // 字符串类
#include <algorithm>    // STL算法
#include <utility>      // STL实用程序组件
#include <cstdlib>      // C标准库的通用工具函数
#include <cmath>        // 数学函数
#include <complex>      // 复数类和函数
#include <type_traits>  // 类型特性元编程

#include <numpy/npy_common.h>   // NumPy的公共头文件
#include "npy_config.h"         // NumPy的配置文件

namespace np {
/// @addtogroup cpp_core_types
/// @{
using std::uint8_t;             // 使用std命名空间中的uint8_t
using std::int8_t;              // 使用std命名空间中的int8_t
using std::uint16_t;            // 使用std命名空间中的uint16_t
using std::int16_t;             // 使用std命名空间中的int16_t
using std::uint32_t;            // 使用std命名空间中的uint32_t
using std::int32_t;             // 使用std命名空间中的int32_t
using std::uint64_t;            // 使用std命名空间中的uint64_t
using std::int64_t;             // 使用std命名空间中的int64_t
using std::uintptr_t;           // 使用std命名空间中的uintptr_t
using std::intptr_t;            // 使用std命名空间中的intptr_t
using std::complex;             // 使用std命名空间中的complex
using std::uint_fast16_t;       // 使用std命名空间中的uint_fast16_t
using std::uint_fast32_t;       // 使用std命名空间中的uint_fast32_t
using SSize = Py_ssize_t;       // 定义SSize为Py_ssize_t类型，用于表示大小或索引

/** Guard for long double.
 *
 * The C implementation defines long double as double
 * on MinGW to provide compatibility with MSVC to unify
 * one behavior under Windows OS, which makes npy_longdouble
 * not fit to be used with template specialization or overloading.
 *
 * This type will be set to `void` when `npy_longdouble` is not defined
 * as `long double`.
 */
using LongDouble = typename std::conditional<
    !std::is_same<npy_longdouble, long double>::value,
     void, npy_longdouble
>::type;
/// @} cpp_core_types

} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_NPSTD_HPP
```