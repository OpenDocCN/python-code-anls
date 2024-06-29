# `.\numpy\numpy\_core\src\umath\_umath_tests.dispatch.c`

```
/**
 * Testing the utilities of the CPU dispatcher
 *
 * @targets $werror baseline
 * SSE2 SSE41 AVX2
 * VSX VSX2 VSX3
 * NEON ASIMD ASIMDHP
 */

// 包含 Python 核心头文件
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// 包含 NumPy 的 CPU 分发相关头文件
#include "npy_cpu_dispatch.h"
#include "numpy/utils.h" // NPY_TOSTRING

// 如果未禁用优化，则包含特定的优化测试分发头文件
#ifndef NPY_DISABLE_OPTIMIZATION
    #include "_umath_tests.dispatch.h"
#endif

// 声明并定义 CPU 分发相关函数和变量
NPY_CPU_DISPATCH_DECLARE(const char *_umath_tests_dispatch_func, (void))
NPY_CPU_DISPATCH_DECLARE(extern const char *_umath_tests_dispatch_var)
NPY_CPU_DISPATCH_DECLARE(void _umath_tests_dispatch_attach, (PyObject *list))

// 初始化 CPU 分发变量并定义相应函数
const char *NPY_CPU_DISPATCH_CURFX(_umath_tests_dispatch_var) = NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(var));
const char *NPY_CPU_DISPATCH_CURFX(_umath_tests_dispatch_func)(void)
{
    static const char *current = NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(func));
    return current;
}

// 将当前 CPU 分发函数名称添加到给定的 Python 列表中
void NPY_CPU_DISPATCH_CURFX(_umath_tests_dispatch_attach)(PyObject *list)
{
    PyObject *item = PyUnicode_FromString(NPY_TOSTRING(NPY_CPU_DISPATCH_CURFX(func)));
    if (item) {
        PyList_Append(list, item);
        Py_DECREF(item);
    }
}
```