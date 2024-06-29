# `.\numpy\numpy\_core\src\umath\_operand_flag_tests.c`

```py
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include "numpy/npy_3kcompat.h"
#include <math.h>
#include <structmember.h>


static PyMethodDef TestMethods[] = {
        {NULL, NULL, 0, NULL}
};


static void
inplace_add(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];  // 指向第一个输入数组的指针
    char *in2 = args[1];  // 指向第二个输入数组的指针
    npy_intp in1_step = steps[0];  // 第一个输入数组的步长
    npy_intp in2_step = steps[1];  // 第二个输入数组的步长

    for (i = 0; i < n; i++) {
        (*(npy_intp *)in1) = *(npy_intp*)in1 + *(npy_intp*)in2;  // 将第二个数组的值加到第一个数组中
        in1 += in1_step;  // 更新第一个数组的指针位置
        in2 += in2_step;  // 更新第二个数组的指针位置
    }
}


/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&inplace_add};

/* These are the input and return dtypes of logit.*/
static const char types[2] = {NPY_INTP, NPY_INTP};  // 输入和返回的数据类型为整型

static void *const data[1] = {NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_operand_flag_tests",  // 模块名
    NULL,  // 模块文档字符串
    -1,  // 模块状态，-1表示使用默认状态
    TestMethods,  // 模块的方法定义
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__operand_flag_tests(void)
{
    PyObject *m = NULL;
    PyObject *ufunc;

    m = PyModule_Create(&moduledef);  // 创建Python模块对象
    if (m == NULL) {
        goto fail;
    }

    import_array();  // 导入NumPy的数组对象API
    import_umath();  // 导入NumPy的数学函数API

    // 创建并初始化ufunc对象，用于执行inplace_add函数
    ufunc = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 0,
                                    PyUFunc_None, "inplace_add",
                                    "inplace_add_docstring", 0);

    /*
     * Set flags to turn off buffering for first input operand,
     * so that result can be written back to input operand.
     */
    // 设置操作标志，以关闭第一个输入操作数的缓冲区，从而可以将结果写回到输入操作数中
    ((PyUFuncObject*)ufunc)->op_flags[0] = NPY_ITER_READWRITE;
    ((PyUFuncObject*)ufunc)->iter_flags = NPY_ITER_REDUCE_OK;
    PyModule_AddObject(m, "inplace_add", (PyObject*)ufunc);  // 将ufunc对象添加到模块中

    return m;

fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load _operand_flag_tests module.");
    }
    if (m) {
        Py_DECREF(m);
        m = NULL;
    }
    return m;
}
```