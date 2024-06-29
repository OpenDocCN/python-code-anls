# `.\numpy\numpy\_core\src\common\ucsnarrow.c`

```
/*
 * 定义宏，禁用已弃用的 NumPy API，并设置为当前 API 版本
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 定义宏，指示该文件属于 MultiArray 模块
 */
#define _MULTIARRAYMODULE

/*
 * 引入 Python 头文件
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
 * 引入 NumPy 数组对象头文件
 */
#include "numpy/arrayobject.h"

/*
 * 引入 NumPy 的数学函数和宏定义头文件
 */
#include "numpy/npy_math.h"

/*
 * 引入 NumPy 的配置头文件
 */
#include "npy_config.h"

/*
 * 引入自定义的 ctors.h 头文件
 */
#include "ctors.h"

/*
 * 以下部分的代码原本包含在窄版 Python 构建中，用于在 NumPy Unicode 数据类型（总是 4 字节）和 Python Unicode 标量（在窄版中为 2 字节）之间进行转换。
 * 这个 "narrow" 接口现在在 Python 中已被弃用，并且在 NumPy 中未被使用。
 */

/*
 * 从包含 UCS4 Unicode 的缓冲区初始化 PyUnicodeObject 对象。
 *
 * Parameters
 * ----------
 *  src: char *
 *      指向包含 UCS4 Unicode 的缓冲区的指针。
 *  size: Py_ssize_t
 *      缓冲区的字节大小。
 *  swap: int
 *      如果为真，则对数据进行交换。
 *  align: int
 *      如果为真，则对数据进行对齐。
 *
 * Returns
 * -------
 * new_reference: PyUnicodeObject
 *      初始化后的 PyUnicodeObject 对象。
 */
NPY_NO_EXPORT PyUnicodeObject *
PyUnicode_FromUCS4(char const *src_char, Py_ssize_t size, int swap, int align)
{
    Py_ssize_t ucs4len = size / sizeof(npy_ucs4);
    npy_ucs4 const *src = (npy_ucs4 const *)src_char;
    npy_ucs4 *buf = NULL;

    /* 如果需要，进行数据交换和对齐 */
    if (swap || align) {
        buf = (npy_ucs4 *)malloc(size);
        if (buf == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(buf, src, size);
        if (swap) {
            byte_swap_vector(buf, ucs4len, sizeof(npy_ucs4));
        }
        src = buf;
    }

    /* 去除尾部的零 */
    while (ucs4len > 0 && src[ucs4len - 1] == 0) {
        ucs4len--;
    }

    /* 使用 PyUnicode_FromKindAndData 函数创建 PyUnicodeObject 对象 */
    PyUnicodeObject *ret = (PyUnicodeObject *)PyUnicode_FromKindAndData(
        PyUnicode_4BYTE_KIND, src, ucs4len);
    
    /* 释放申请的内存 */
    free(buf);

    /* 返回创建的 PyUnicodeObject 对象 */
    return ret;
}
```