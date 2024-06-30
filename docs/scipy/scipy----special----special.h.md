# `D:\src\scipysrc\scipy\scipy\special\special.h`

```
#pragma once
// 只允许本头文件被编译一次

#include "Python.h"
// 包含 Python.h 头文件，提供 Python C API 支持

#include "special/legendre.h"
// 包含特殊函数的 Legendre 相关函数声明

#include "special/specfun.h"
// 包含特殊函数的通用函数声明

#include "special/sph_harm.h"
// 包含特殊函数的球谐函数声明

namespace {
// 匿名命名空间，限制声明的作用域在当前文件内有效

template <typename T, typename OutputVec1, typename OutputVec2>
void lpn(T z, OutputVec1 p, OutputVec2 p_jac) {
    // 计算 Legendre 函数和其导数
    special::legendre_all(z, p);
    special::legendre_all_jac(z, p, p_jac);
}

template <typename T, typename OutputMat1, typename OutputMat2>
void lpmn(T x, bool m_signbit, OutputMat1 p, OutputMat2 p_jac) {
    // 计算关联 Legendre 函数和其导数
    special::assoc_legendre_all(x, m_signbit, p);
    special::assoc_legendre_all_jac(x, m_signbit, p, p_jac);
}

template <typename T>
std::complex<T> sph_harm(long m, long n, T theta, T phi) {
    // 计算球谐函数
    if (std::abs(m) > n) {
        // 检查参数是否合理，若不合理则设置错误状态并返回 NaN
        special::set_error("sph_harm", SF_ERROR_ARG, "m should not be greater than n");
        return NAN;
    }

    return special::sph_harm(m, n, theta, phi);
}

template <typename T>
std::complex<T> sph_harm(T m, T n, T theta, T phi) {
    // 计算球谐函数（处理浮点数参数情况）
    if (static_cast<long>(m) != m || static_cast<long>(n) != n) {
        // 若浮点数被截断为整数，则发出运行时警告
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyErr_WarnEx(PyExc_RuntimeWarning, "floating point number truncated to an integer", 1);
        PyGILState_Release(gstate);
    }

    return sph_harm(static_cast<long>(m), static_cast<long>(n), theta, phi);
}

} // namespace
// 结束匿名命名空间
```