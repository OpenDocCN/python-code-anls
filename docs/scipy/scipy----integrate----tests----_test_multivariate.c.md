# `D:\src\scipysrc\scipy\scipy\integrate\tests\_test_multivariate.c`

```
#include <Python.h>

#include "math.h"

const double PI = 3.141592653589793238462643383279502884;

static double
_multivariate_typical(int n, double *args)
{
    // 计算多变量典型函数值，根据参数计算结果并返回
    return cos(args[1] * args[0] - args[2] * sin(args[0])) / PI;
}

static double
_multivariate_indefinite(int n, double *args)
{
    // 计算多变量不定积分函数值，根据参数计算结果并返回
    return -exp(-args[0]) * log(args[0]);
}

static double
_multivariate_sin(int n, double *args)
{
    // 计算多变量正弦函数值，根据参数计算结果并返回
    return sin(args[0]);
}

static double
_sin_0(double x, void *user_data)
{
    // 计算正弦函数值，接受一个参数并返回其正弦值
    return sin(x);
}

static double
_sin_1(int ndim, double *x, void *user_data)
{
    // 计算多维正弦函数值，接受一个参数数组并返回第一个元素的正弦值
    return sin(x[0]);
}

static double
_sin_2(double x)
{
    // 计算正弦函数值，接受一个参数并返回其正弦值
    return sin(x);
}

static double
_sin_3(int ndim, double *x)
{
    // 计算多维正弦函数值，接受一个参数数组并返回第一个元素的正弦值
    return sin(x[0]);
}


typedef struct {
    char *name;
    void *ptr;
} routine_t;


static const routine_t routines[] = {
    {"_multivariate_typical", &_multivariate_typical},
    {"_multivariate_indefinite", &_multivariate_indefinite},
    {"_multivariate_sin", &_multivariate_sin},
    {"_sin_0", &_sin_0},
    {"_sin_1", &_sin_1},
    {"_sin_2", &_sin_2},
    {"_sin_3", &_sin_3}
};


static int create_pointers(PyObject *module)
{
    PyObject *d, *obj = NULL;
    size_t i;

    d = PyModule_GetDict(module);
    if (d == NULL) {
        goto fail;
    }

    for (i = 0; i < sizeof(routines) / sizeof(routine_t); ++i) {
        obj = PyLong_FromVoidPtr(routines[i].ptr);
        if (obj == NULL) {
            goto fail;
        }

        if (PyDict_SetItemString(d, routines[i].name, obj)) {
            goto fail;
        }

        Py_DECREF(obj);
        obj = NULL;
    }

    Py_XDECREF(obj);
    return 0;

fail:
    Py_XDECREF(obj);
    return -1;
}


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_test_multivariate",
    NULL,
    -1,
    NULL, /* Empty methods section */
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__test_multivariate(void)
{
    // 创建一个 Python 模块对象
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }
    // 将函数指针添加到模块字典中
    if (create_pointers(m)) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
```