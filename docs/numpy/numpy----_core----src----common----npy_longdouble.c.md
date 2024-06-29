# `.\numpy\numpy\_core\src\common\npy_longdouble.c`

```
/*
 * 定义宏，指定使用的 NumPy API 版本，禁用过时的 API
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 定义宏，用于多维数组模块
 */
#define _MULTIARRAYMODULE

/*
 * 清理 PY_SSIZE_T 的定义，确保使用最新的 API
 */
#define PY_SSIZE_T_CLEAN

/*
 * 包含 Python.h 头文件，提供 Python C API 的基本功能
 */
#include <Python.h>

/*
 * 包含 NumPy 的数组类型定义头文件
 */
#include "numpy/ndarraytypes.h"

/*
 * 包含 NumPy 的数学函数头文件
 */
#include "numpy/npy_math.h"

/*
 * 包含 NumPy 的操作系统相关头文件
 */
#include "numpyos.h"

/*
 * 将 longdouble 转换为 Python 的长整型对象 PyLong 的函数。
 * 这个函数是基于 PyLong_FromDouble 修改的，由于不能直接设置数字的位数，
 * 因此必须进行位移和按位或操作。
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_longdouble_to_PyLong(npy_longdouble ldval)
{
    PyObject *v;
    PyObject *l_chunk_size;
    /*
     * 每次提取的比特位数。CPython 使用 30，但这是因为它与内部长整型表示相关
     */
    const int chunk_size = NPY_BITSOF_LONGLONG;
    npy_longdouble frac;
    int i, ndig, expo, neg;
    neg = 0;

    /*
     * 如果 ldval 是无穷大，则设置 OverflowError 异常
     */
    if (npy_isinf(ldval)) {
        PyErr_SetString(PyExc_OverflowError,
                        "cannot convert longdouble infinity to integer");
        return NULL;
    }
    /*
     * 如果 ldval 是 NaN，则设置 ValueError 异常
     */
    if (npy_isnan(ldval)) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot convert longdouble NaN to integer");
        return NULL;
    }
    /*
     * 如果 ldval 是负数，则设置 neg 标志并取其绝对值
     */
    if (ldval < 0.0) {
        neg = 1;
        ldval = -ldval;
    }
    /*
     * 分解 ldval 为尾数 frac 和指数 expo，ldval = frac * 2**expo; 0.0 <= frac < 1.0
     */
    frac = npy_frexpl(ldval, &expo);
    v = PyLong_FromLong(0L);
    if (v == NULL)
        return NULL;
    /*
     * 如果指数 expo 小于等于 0，则直接返回 v
     */
    if (expo <= 0)
        return v;

    /*
     * 计算所需的位数 ndig
     */
    ndig = (expo - 1) / chunk_size + 1;

    /*
     * 创建长整型对象 l_chunk_size 以表示每次位移的比特位数
     */
    l_chunk_size = PyLong_FromLong(chunk_size);
    if (l_chunk_size == NULL) {
        Py_DECREF(v);
        return NULL;
    }

    /*
     * 获取浮点数的整数部分的最高有效位
     */
    frac = npy_ldexpl(frac, (expo - 1) % chunk_size + 1);
    for (i = ndig; --i >= 0;) {
        npy_ulonglong chunk = (npy_ulonglong) frac;
        PyObject *l_chunk;
        /*
         * v = v << chunk_size，将 v 左移 chunk_size 位
         */
        Py_SETREF(v, PyNumber_Lshift(v, l_chunk_size));
        if (v == NULL) {
            goto done;
        }
        /*
         * 创建表示 chunk 的长整型对象 l_chunk
         */
        l_chunk = PyLong_FromUnsignedLongLong(chunk);
        if (l_chunk == NULL) {
            Py_DECREF(v);
            v = NULL;
            goto done;
        }
        /*
         * v = v | chunk，将 chunk 或运算到 v 上
         */
        Py_SETREF(v, PyNumber_Or(v, l_chunk));
        Py_DECREF(l_chunk);
        if (v == NULL) {
            goto done;
        }

        /*
         * 去除最高位并重复
         */
        frac = frac - (npy_longdouble) chunk;
        frac = npy_ldexpl(frac, chunk_size);
    }

    /*
     * 如果是负数，取 v 的相反数
     */
    if (neg) {
        Py_SETREF(v, PyNumber_Negative(v));
        if (v == NULL) {
            goto done;
        }
    }

done:
    /*
     * 清理临时使用的 l_chunk_size 对象
     */
    Py_DECREF(l_chunk_size);
    return v;
}

/*
 * 辅助函数，用于将 PyLong 对象转换为 UTF-8 编码的字节串
 */
static PyObject *
_PyLong_Bytes(PyObject *long_obj) {
    PyObject *bytes;
    PyObject *unicode = PyObject_Str(long_obj);
    if (unicode == NULL) {
        return NULL;
    }
    bytes = PyUnicode_AsUTF8String(unicode);
    Py_DECREF(unicode);
    return bytes;
}
/**
 * 从 Python 的 long 对象转换为 npy_longdouble 类型的数值。
 * 
 * 使用一个字符串表示的 long 值进行转换，这种方法是正确的但速度较慢。
 * 另一种方法是通过数值方式转换，类似于 PyLong_AsDouble 的方式。
 * 然而，为了正确处理舍入模式，这需要知道尾数的大小，这是依赖于平台的。
 */
NPY_VISIBILITY_HIDDEN npy_longdouble
npy_longdouble_from_PyLong(PyObject *long_obj) {
    npy_longdouble result = 1234;  // 初始化结果为 1234
    char *end;  // 字符串结束位置指针
    char *cstr;  // 字符串指针
    PyObject *bytes;  // Python 字节对象

    /* convert the long to a string */
    bytes = _PyLong_Bytes(long_obj);  // 将 Python long 对象转换为字节对象
    if (bytes == NULL) {
        return -1;  // 转换失败，返回错误码 -1
    }

    cstr = PyBytes_AsString(bytes);  // 获取字节对象的字符串表示
    if (cstr == NULL) {
        goto fail;  // 如果转换为字符串失败，则跳转到失败处理
    }
    end = NULL;  // 初始化结束位置指针为 NULL

    /* convert the string to a long double and capture errors */
    errno = 0;  // 清空错误码
    result = NumPyOS_ascii_strtold(cstr, &end);  // 将字符串转换为长双精度浮点数，并捕获错误
    if (errno == ERANGE) {
        /* strtold 返回正确符号的无穷大。 */
        if (PyErr_Warn(PyExc_RuntimeWarning,
                "overflow encountered in conversion from python long") < 0) {
            goto fail;  // 发出溢出警告失败，跳转到失败处理
        }
    }
    else if (errno) {
        PyErr_Format(PyExc_RuntimeError,
                     "Could not parse python long as longdouble: %s (%s)",
                     cstr,
                     strerror(errno));
        goto fail;  // 其他错误情况，抛出运行时错误，跳转到失败处理
    }

    /* Extra characters at the end of the string, or nothing parsed */
    if (end == cstr || *end != '\0') {
        PyErr_Format(PyExc_RuntimeError,
                     "Could not parse long as longdouble: %s",
                     cstr);
        goto fail;  // 字符串末尾有额外字符或未解析任何内容，抛出运行时错误，跳转到失败处理
    }

    /* finally safe to decref now that we're done with `end` */
    Py_DECREF(bytes);  // 安全地释放字节对象的引用计数
    return result;  // 返回转换后的长双精度浮点数

fail:
    Py_DECREF(bytes);  // 失败处理：释放字节对象的引用计数
    return -1;  // 返回错误码 -1
}
```