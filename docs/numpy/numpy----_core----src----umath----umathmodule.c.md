# `.\numpy\numpy\_core\src\umath\umathmodule.c`

```
/*
 * _UMATHMODULE IS needed in __ufunc_api.h, included from numpy/ufuncobject.h.
 * This is a mess and it would be nice to fix it. It has nothing to do with
 * __ufunc_api.c
 */
/* 定义宏 _UMATHMODULE，用于 __ufunc_api.h 中，从 numpy/ufuncobject.h 中引入 */
#define _UMATHMODULE

/* 使用最新的 NPY API 版本，禁用所有过时的 API */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* 定义宏 _MULTIARRAYMODULE，用于多维数组模块 */
#define _MULTIARRAYMODULE

/* 定义宏 _UMATHMODULE，用于通用数学函数模块 */
#define _UMATHMODULE

/* 清除 PY_SSIZE_T_CLEAN 宏定义 */
#define PY_SSIZE_T_CLEAN

/* 包含 Python 标准头文件 */
#include <Python.h>

/* 包含 numpy 的配置文件 */
#include "npy_config.h"

/* 包含 numpy CPU 特性检测和分发 */
#include "npy_cpu_features.h"
#include "npy_cpu_dispatch.h"

/* 包含 numpy CPU 相关头文件 */
#include "numpy/npy_cpu.h"

/* 包含 numpy 数组对象头文件 */
#include "numpy/arrayobject.h"

/* 包含 numpy 通用函数对象头文件 */
#include "numpy/ufuncobject.h"

/* 包含 numpy 3k 兼容性头文件 */
#include "numpy/npy_3kcompat.h"

/* 包含 numpy Python 兼容性头文件 */
#include "npy_pycompat.h"

/* 包含 numpy 抽象对象头文件 */
#include "abstract.h"

/* 包含 numpy 数学函数头文件 */
#include "numpy/npy_math.h"

/* 包含数字相关头文件 */
#include "number.h"

/* 包含分发相关头文件 */
#include "dispatching.h"

/* 包含字符串通用函数头文件 */
#include "string_ufuncs.h"

/* 包含字符串数据类型通用函数头文件 */
#include "stringdtype_ufuncs.h"

/* 包含特殊整数比较头文件 */
#include "special_integer_comparisons.h"

/* 包含外部对象头文件，用于 _extobject_contextvar 的暴露 */
#include "extobj.h"

/* 包含通用函数类型解析头文件 */
#include "ufunc_type_resolution.h"

/* 包含 funcs.inc 自动生成的所有通用函数的代码 */
#include "funcs.inc"

/* 包含 __umath_generated.c 自动生成的代码 */
#include "__umath_generated.c"

/* 定义 pyfunc_functions 数组，包含 PyUFunc_On_Om 函数 */
static PyUFuncGenericFunction pyfunc_functions[] = {PyUFunc_On_Om};

/* 定义对象类型解析函数 object_ufunc_type_resolver */
static int
object_ufunc_type_resolver(PyUFuncObject *ufunc,
                            NPY_CASTING casting,
                            PyArrayObject **operands,
                            PyObject *type_tup,
                            PyArray_Descr **out_dtypes)
{
    int i, nop = ufunc->nin + ufunc->nout;

    /* 将输出数据类型设置为 NPY_OBJECT 类型 */
    out_dtypes[0] = PyArray_DescrFromType(NPY_OBJECT);
    if (out_dtypes[0] == NULL) {
        return -1;
    }

    /* 复制第一个输出数据类型给所有操作数 */
    for (i = 1; i < nop; ++i) {
        Py_INCREF(out_dtypes[0]);
        out_dtypes[i] = out_dtypes[0];
    }

    return 0;
}

/* 定义从 Python 函数创建通用函数的函数 ufunc_frompyfunc */
PyObject *
ufunc_frompyfunc(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds) {
    PyObject *function, *pyname = NULL;
    int nin, nout, i, nargs;
    PyUFunc_PyFuncData *fdata;
    PyUFuncObject *self;
    const char *fname = NULL;
    char *str, *types, *doc;
    Py_ssize_t fname_len = -1;
    void * ptr, **data;
    int offset[2];
    PyObject *identity = NULL;  /* 注意：语义不同于 Py_None */
    static char *kwlist[] = {"", "nin", "nout", "identity", NULL};

    /* 解析参数列表和关键字参数 */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii|$O:frompyfunc", kwlist,
                &function, &nin, &nout, &identity)) {
        return NULL;
    }
    /* 检查 function 是否可调用 */
    if (!PyCallable_Check(function)) {
        PyErr_SetString(PyExc_TypeError, "function must be callable");
        return NULL;
    }

    /* 计算参数个数 */
    nargs = nin + nout;

    /* 获取 function 的名称 */
    pyname = PyObject_GetAttrString(function, "__name__");
    if (pyname) {
        fname = PyUnicode_AsUTF8AndSize(pyname, &fname_len);
    }
    /* 如果获取名称失败，则使用默认名称 "?" */
    if (fname == NULL) {
        PyErr_Clear();
        fname = "?";
        fname_len = 1;
    }

    /* 这里为函数体的起始部分，后续代码未提供，可能包括函数体实现和返回语句 */
}
    /*
     * ptr will be assigned to self->ptr, which holds a pointer to memory allocated for various data structures:
     * self->data[0] (fdata)
     * self->data
     * self->name
     * self->types
     *
     * To ensure memory alignment on void * pointers, additional space may be allocated.
     * Therefore, offsets are calculated to ensure alignment.
     */
    offset[0] = sizeof(PyUFunc_PyFuncData);
    i = (sizeof(PyUFunc_PyFuncData) % sizeof(void *));
    if (i) {
        offset[0] += (sizeof(void *) - i);
    }
    offset[1] = nargs;
    i = (nargs % sizeof(void *));
    if (i) {
        offset[1] += (sizeof(void *) - i);
    }
    
    // Allocate memory for ptr, considering offsets and additional space for fname.
    ptr = PyArray_malloc(offset[0] + offset[1] + sizeof(void *) + (fname_len + 14));
    if (ptr == NULL) {
        Py_XDECREF(pyname);
        return PyErr_NoMemory();
    }
    
    // Point fdata to the allocated memory at ptr.
    fdata = (PyUFunc_PyFuncData *)(ptr);
    fdata->callable = function;
    fdata->nin = nin;
    fdata->nout = nout;
    
    // Set data to point to fdata and initialize types array.
    data = (void **)(((char *)ptr) + offset[0]);
    data[0] = (void *)fdata;
    types = (char *)data + sizeof(void *);
    for (i = 0; i < nargs; i++) {
        types[i] = NPY_OBJECT;
    }
    
    // Set str to contain fname followed by " (vectorized)".
    str = types + offset[1];
    memcpy(str, fname, fname_len);
    memcpy(str + fname_len, " (vectorized)", 14);
    Py_XDECREF(pyname);
    
    /* Do a better job someday */
    doc = "dynamic ufunc based on a python function";
    
    // Create a PyUFuncObject using PyUFunc_FromFuncAndDataAndSignatureAndIdentity.
    self = (PyUFuncObject *)PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
            (PyUFuncGenericFunction *)pyfunc_functions, data,
            types, /* ntypes */ 1, nin, nout, identity ? PyUFunc_IdentityValue : PyUFunc_None,
            str, doc, /* unused */ 0, NULL, identity);
    
    if (self == NULL) {
        PyArray_free(ptr);
        return NULL;
    }
    
    // Increment the reference count for function, assign object and ptr to self.
    Py_INCREF(function);
    self->obj = function;
    self->ptr = ptr;
    
    // Set type_resolver and track self for garbage collection.
    self->type_resolver = &object_ufunc_type_resolver;
    PyObject_GC_Track(self);
    
    // Return the constructed PyUFuncObject.
    return (PyObject *)self;
/* docstring in numpy.add_newdocs.py */
/* 定义一个函数，用于为新的ufunc对象添加文档字符串 */

PyObject *
add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    // 声明变量
    PyUFuncObject *ufunc;
    PyObject *str;

    // 解析传入的参数，期望参数为 PyUFunc_Type 和 PyUnicode_Type
    if (!PyArg_ParseTuple(args, "O!O!:_add_newdoc_ufunc", &PyUFunc_Type, &ufunc,
                                        &PyUnicode_Type, &str)) {
        return NULL;
    }

    // 如果ufunc对象已经有文档字符串，不允许修改，返回错误
    if (ufunc->doc != NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot change docstring of ufunc with non-NULL docstring");
        return NULL;
    }

    // 将传入的字符串对象转换为UTF-8编码的字节串
    PyObject *tmp = PyUnicode_AsUTF8String(str);
    if (tmp == NULL) {
        return NULL;
    }
    char *docstr = PyBytes_AS_STRING(tmp);

    /*
     * 这里存在内存泄漏风险，因为分配的文档字符串内存不会在删除ufunc对象时被释放。
     * 不过通常情况下不会有问题，因为用户需要重复创建、文档化和丢弃ufunc对象才会触发。
     */
    
    // 为新文档字符串分配内存空间，并复制内容
    char *newdocstr = malloc(strlen(docstr) + 1);
    if (!newdocstr) {
        Py_DECREF(tmp);
        return PyErr_NoMemory();
    }
    strcpy(newdocstr, docstr);

    // 将新的文档字符串赋值给ufunc对象
    ufunc->doc = newdocstr;

    // 释放临时字节串对象
    Py_DECREF(tmp);

    // 返回None表示成功
    Py_RETURN_NONE;
}

/*
 *****************************************************************************
 **                            SETUP UFUNCS                                 **
 *****************************************************************************
 */

/* 设置ufunc模块的初始化函数 */

int initumath(PyObject *m)
{
    // 声明变量
    PyObject *d, *s, *s2;
    int UFUNC_FLOATING_POINT_SUPPORT = 1;

    // 检查是否定义了 NO_UFUNC_FLOATING_POINT_SUPPORT 宏
#ifdef NO_UFUNC_FLOATING_POINT_SUPPORT
    UFUNC_FLOATING_POINT_SUPPORT = 0;
#endif

    // 将模块m的字典对象赋值给变量d
    d = PyModule_GetDict(m);

    // 初始化操作符，如果失败则返回-1
    if (InitOperators(d) < 0) {
        return -1;
    }

    // 向模块字典中添加一些符号常量
    PyDict_SetItemString(d, "pi", s = PyFloat_FromDouble(NPY_PI));
    Py_DECREF(s);
    PyDict_SetItemString(d, "e", s = PyFloat_FromDouble(NPY_E));
    Py_DECREF(s);
    PyDict_SetItemString(d, "euler_gamma", s = PyFloat_FromDouble(NPY_EULER));
    Py_DECREF(s);

    // 定义宏来添加整型常量和字符串常量
#define ADDCONST(str) PyModule_AddIntConstant(m, #str, UFUNC_##str)
#define ADDSCONST(str) PyModule_AddStringConstant(m, "UFUNC_" #str, UFUNC_##str)

    // 添加浮点异常常量
    ADDCONST(FPE_DIVIDEBYZERO);
    ADDCONST(FPE_OVERFLOW);
    ADDCONST(FPE_UNDERFLOW);
    ADDCONST(FPE_INVALID);

    // 添加浮点支持宏
    ADDCONST(FLOATING_POINT_SUPPORT);

    // 添加Python值的名称常量
    ADDSCONST(PYVALS_NAME);

    // 清除宏定义
#undef ADDCONST
#undef ADDSCONST

    // 添加整型常量 UFUC_BUFSIZE_DEFAULT
    PyModule_AddIntConstant(m, "UFUNC_BUFSIZE_DEFAULT", (long)NPY_BUFSIZE);

    // 增加对静态数据的引用，并添加到模块中
    Py_INCREF(npy_static_pydata.npy_extobj_contextvar);
    PyModule_AddObject(m, "_extobj_contextvar", npy_static_pydata.npy_extobj_contextvar);

    // 向模块添加一些浮点数对象常量
    PyModule_AddObject(m, "PINF", PyFloat_FromDouble(NPY_INFINITY));
    PyModule_AddObject(m, "NINF", PyFloat_FromDouble(-NPY_INFINITY));
    PyModule_AddObject(m, "PZERO", PyFloat_FromDouble(NPY_PZERO));
    PyModule_AddObject(m, "NZERO", PyFloat_FromDouble(NPY_NZERO));
    PyModule_AddObject(m, "NAN", PyFloat_FromDouble(NPY_NAN));
}
    # 从字典 d 中获取键为 "divide" 的值，并赋给变量 s
    s = PyDict_GetItemString(d, "divide");
    # 将变量 s 添加到字典 d 中，键为 "true_divide"
    PyDict_SetItemString(d, "true_divide", s);

    # 从字典 d 中获取键为 "conjugate" 和 "remainder" 的值，并分别赋给变量 s 和 s2
    s = PyDict_GetItemString(d, "conjugate");
    s2 = PyDict_GetItemString(d, "remainder");

    # 设置数组对象的数值结构，将适当的 ufuncs 添加到字典 d 中
    if (_PyArray_SetNumericOps(d) < 0) {
        return -1;
    }

    # 将变量 s 添加到字典 d 中，键为 "conj"
    PyDict_SetItemString(d, "conj", s);
    # 将变量 s2 添加到字典 d 中，键为 "mod"
    PyDict_SetItemString(d, "mod", s2);

    """
     * 设置逻辑函数的提升器
     * TODO: 可能应该在更合适的位置进行，甚至可以直接在代码生成器中完成。
     """
    # 从字典 d 中获取键为 "logical_and" 的值，并将结果存储在变量 s 中
    int res = PyDict_GetItemStringRef(d, "logical_and", &s);
    if (res <= 0) {
        return -1;
    }
    # 安装逻辑 ufunc 提升器，如果失败则释放变量 s，并返回 -1
    if (install_logical_ufunc_promoter(s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    # 释放变量 s
    Py_DECREF(s);

    # 依次处理 "logical_or" 和 "logical_xor" 的逻辑与上述步骤类似
    res = PyDict_GetItemStringRef(d, "logical_or", &s);
    if (res <= 0) {
        return -1;
    }
    if (install_logical_ufunc_promoter(s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    res = PyDict_GetItemStringRef(d, "logical_xor", &s);
    if (res <= 0) {
        return -1;
    }
    if (install_logical_ufunc_promoter(s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    # 初始化字符串 ufuncs，如果失败则返回 -1
    if (init_string_ufuncs(d) < 0) {
        return -1;
    }

    # 初始化字符串数据类型的 ufuncs，如果失败则返回 -1
    if (init_stringdtype_ufuncs(m) < 0) {
        return -1;
    }

    # 初始化特殊整数比较操作的 ufuncs，如果失败则返回 -1
    if (init_special_int_comparisons(d) < 0) {
        return -1;
    }

    # 函数执行成功，返回 0
    return 0;
}



# 这行代码表示一个代码块的结束
```