# `.\numpy\numpy\_core\src\multiarray\convert.c`

```
/*
 * 设置 NPY_NO_DEPRECATED_API 到 NPY_API_VERSION，避免使用过时的 NumPy API
 * 定义 _MULTIARRAYMODULE，用于多维数组模块
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T_CLEAN 宏定义，确保使用最新的 Python 对象大小 API
 * 包含 Python.h 头文件，提供 Python C API 功能
 * 包含 structmember.h 头文件，用于定义 C 结构体的成员
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

/*
 * 包含 numpy 库配置相关头文件
 * 包含 numpy 数组对象头文件
 * 包含 numpy 数组标量头文件
 */
#include "npy_config.h"
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

/*
 * 包含通用功能头文件
 * 包含数组对象功能头文件
 * 包含构造函数功能头文件
 * 包含数据类型元数据头文件
 * 包含映射头文件
 * 包含低级分步循环头文件
 * 包含标量类型头文件
 * 包含数组赋值头文件
 */
#include "common.h"
#include "arrayobject.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "mapping.h"
#include "lowlevel_strided_loops.h"
#include "scalartypes.h"
#include "array_assign.h"

/*
 * 包含类型转换功能头文件
 * 包含数组强制转换功能头文件
 * 包含引用计数头文件
 */
#include "convert.h"
#include "array_coercion.h"
#include "refcount.h"

/*
 * 如果定义了 HAVE_FALLOCATE 并且在 Linux 平台上
 * 包含 fcntl.h 头文件，提供文件控制相关功能
 */
#if defined(HAVE_FALLOCATE) && defined(__linux__)
#include <fcntl.h>
#endif

/*
 * npy_fallocate 函数：为文件 fp 分配 nbytes 大小的磁盘空间
 * 允许文件系统进行更智能的分配决策，并在空间不足时快速退出
 * 返回 -1 并引发异常表示空间不足，忽略其他所有错误
 */
static int
npy_fallocate(npy_intp nbytes, FILE * fp)
{
    /*
     * 如果定义了 HAVE_FALLOCATE 并且在 Linux 平台上
     */
#if defined(HAVE_FALLOCATE) && defined(__linux__)
    int r;
    /* 对于小文件不值得进行系统调用 */
    if (nbytes < 16 * 1024 * 1024) {
        return 0;
    }

    /*
     * 刷新文件流，以防在 fallocate 调用和描述符中的未写数据之间存在意外交互
     */
    NPY_BEGIN_ALLOW_THREADS;
    r = fallocate(fileno(fp), 1, npy_ftell(fp), nbytes);
    NPY_END_ALLOW_THREADS;

    /*
     * 如果空间不足，则提前退出，并在写入过程中发现其他错误
     */
    if (r == -1 && errno == ENOSPC) {
        PyErr_Format(PyExc_OSError, "Not enough free space to write "
                     "%"NPY_INTP_FMT" bytes", nbytes);
        return -1;
    }
#endif
    return 0;
}

/*
 * recursive_tolist 函数：将 self 数组的子数组转换为列表
 * 从数据指针 dataptr 和维度 startdim 开始，直到 self 的最后一个维度
 * 返回新的引用对象
 */
static PyObject *
recursive_tolist(PyArrayObject *self, char *dataptr, int startdim)
{
    npy_intp i, n, stride;
    PyObject *ret, *item;

    /* 基本情况 */
    if (startdim >= PyArray_NDIM(self)) {
        return PyArray_GETITEM(self, dataptr);
    }

    n = PyArray_DIM(self, startdim);
    stride = PyArray_STRIDE(self, startdim);

    ret = PyList_New(n);
    if (ret == NULL) {
        return NULL;
    }

    for (i = 0; i < n; ++i) {
        item = recursive_tolist(self, dataptr, startdim+1);
        if (item == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyList_SET_ITEM(ret, i, item);

        dataptr += stride;
    }

    return ret;
}

/*
 * PyArray_ToList 函数：将数组 self 转换为 Python 列表
 * 调用 recursive_tolist 函数完成转换
 * 返回新的 Python 对象引用
 */
NPY_NO_EXPORT PyObject *
PyArray_ToList(PyArrayObject *self)
{
    return recursive_tolist(self, PyArray_DATA(self), 0);
}
/* XXX: FIXME --- add ordering argument to
   Allow Fortran ordering on write
   This will need the addition of a Fortran-order iterator.
 */
/* 在写入时添加排序参数
   允许使用Fortran排序
   这将需要添加Fortran排序迭代器。
 */

/*NUMPY_API
  To File
*/
/* NUMPY_API
   写入文件
*/

NPY_NO_EXPORT int
/* 不导出API */
PyArray_ToFile(PyArrayObject *self, FILE *fp, char *sep, char *format)
{
    npy_intp size;
    npy_intp n, n2;
    size_t n3, n4;
    PyArrayIterObject *it;
    PyObject *obj, *strobj, *tupobj, *byteobj;

    n3 = (sep ? strlen((const char *)sep) : 0);
    /* 计算分隔符的长度，如果不存在则为0 */

    if (n3 == 0) {
        /* binary data */
        /* 二进制数据 */

        if (PyDataType_FLAGCHK(PyArray_DESCR(self), NPY_LIST_PICKLE)) {
            PyErr_SetString(PyExc_OSError,
                    "cannot write object arrays to a file in binary mode");
            return -1;
        }
        /* 如果数组描述符标志包含NPY_LIST_PICKLE，抛出异常不能以二进制模式写入对象数组 */

        if (PyArray_ITEMSIZE(self) == 0) {
            /* For zero-width data types there's nothing to write */
            /* 对于零宽度的数据类型，没有内容需要写入 */
            return 0;
        }

        if (npy_fallocate(PyArray_NBYTES(self), fp) != 0) {
            /* 使用npy_fallocate分配内存失败时返回-1 */
            return -1;
        }

        if (PyArray_ISCONTIGUOUS(self)) {
            /* 如果数组是连续存储的 */

            size = PyArray_SIZE(self);
            /* 获取数组的大小 */
            NPY_BEGIN_ALLOW_THREADS;
            /* 开始允许多线程 */
#if defined(_WIN64)
            /*
             * 解决 Win64 fwrite() Bug。问题详见 gh-2256
             * 本地 64 位 Windows 运行时存在此问题，上述代码也将触发 UCRT（未触发的情况也可能更精确）。
             *
             * 如果您修改了此代码，请运行以下测试，该测试因速度慢而已从测试套件中移除。
             * 原始失败模式涉及在 tofile() 过程中的无限循环。
             *
             * import tempfile, numpy as np
             * from numpy.testing import (assert_)
             * fourgbplus = 2**32 + 2**16
             * testbytes = np.arange(8, dtype=np.int8)
             * n = len(testbytes)
             * flike = tempfile.NamedTemporaryFile()
             * f = flike.file
             * np.tile(testbytes, fourgbplus // testbytes.nbytes).tofile(f)
             * flike.seek(0)
             * a = np.fromfile(f, dtype=np.int8)
             * flike.close()
             * assert_(len(a) == fourgbplus)
             * # check only start and end for speed:
             * assert_((a[:n] == testbytes).all())
             * assert_((a[-n:] == testbytes).all())
             */
            {
                // 计算每次写入的最大字节数，避免 Win64 下 fwrite() 的问题
                size_t maxsize = 2147483648 / (size_t)PyArray_ITEMSIZE(self);
                size_t chunksize;

                n = 0;
                // 循环写入直到所有数据被处理完毕
                while (size > 0) {
                    // 确定当前循环可以写入的数据块大小
                    chunksize = (size > maxsize) ? maxsize : size;
                    // 调用 fwrite() 写入数据
                    n2 = fwrite((const void *)
                                 ((char *)PyArray_DATA(self) + (n * PyArray_ITEMSIZE(self))),
                                 (size_t) PyArray_ITEMSIZE(self),
                                 chunksize, fp);
                    // 检查写入是否完整，若不完整则跳出循环
                    if (n2 < chunksize) {
                        break;
                    }
                    // 更新已写入数据的总量和剩余数据的大小
                    n += n2;
                    size -= chunksize;
                }
                // 重置 size 为数组的总大小
                size = PyArray_SIZE(self);
            }
#else
            // 非 Win64 平台下直接调用 fwrite() 写入数据
            n = fwrite((const void *)PyArray_DATA(self),
                    (size_t) PyArray_ITEMSIZE(self),
                    (size_t) size, fp);
#endif
#else
            // 结束任何可能存在的线程访问
            NPY_END_ALLOW_THREADS;
            // 如果写入的字节数小于请求的字节数，抛出异常并返回-1
            if (n < size) {
                PyErr_Format(PyExc_OSError,
                        "%ld requested and %ld written",
                        (long) size, (long) n);
                return -1;
            }
        }
        else {
            // 定义线程开始的宏
            NPY_BEGIN_THREADS_DEF;

            // 创建数组迭代器对象
            it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)self);
            // 开始线程
            NPY_BEGIN_THREADS;
            // 当迭代器的索引小于其大小时循环
            while (it->index < it->size) {
                // 将当前迭代器位置的数据写入文件
                if (fwrite((const void *)it->dataptr,
                            (size_t) PyArray_ITEMSIZE(self),
                            1, fp) < 1) {
                    // 结束线程
                    NPY_END_THREADS;
                    // 格式化异常信息，并返回-1
                    PyErr_Format(PyExc_OSError,
                            "problem writing element %" NPY_INTP_FMT
                            " to file", it->index);
                    Py_DECREF(it);
                    return -1;
                }
                // 移动迭代器到下一个元素
                PyArray_ITER_NEXT(it);
            }
            // 结束线程
            NPY_END_THREADS;
            // 释放迭代器对象的引用
            Py_DECREF(it);
        }
    }
    else:
        """
        * text data
        """

        # 创建一个 PyArrayIterObject 迭代器对象，用于迭代数组 self
        it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)self)
        # 如果定义了格式字符串 format，则计算其长度；否则长度为 0
        n4 = (format ? strlen((const char *)format) : 0)
        # 当迭代器的索引小于数组的大小时，执行循环
        while (it->index < it->size):
            """
            * This is as documented.  If we have a low precision float value
            * then it may convert to float64 and store unnecessary digits.
            * TODO: This could be fixed, by not using `arr.item()` or using
            *       the array printing/formatting functionality.
            """
            # 获取数组中当前迭代位置的元素对象 obj
            obj = PyArray_GETITEM(self, it->dataptr)
            # 如果获取失败，则释放迭代器并返回 -1
            if (obj == NULL):
                Py_DECREF(it)
                return -1
            # 如果没有指定格式字符串
            if (n4 == 0):
                """
                * standard writing
                """
                # 将 obj 转换为字符串对象 strobj
                strobj = PyObject_Str(obj)
                # 释放 obj 对象的引用计数
                Py_DECREF(obj)
                # 如果转换失败，则释放迭代器并返回 -1
                if (strobj == NULL):
                    Py_DECREF(it)
                    return -1
            else:
                """
                * use format string
                """
                # 创建一个包含 obj 的单元素元组对象 tupobj
                tupobj = PyTuple_New(1)
                # 如果创建失败，则释放迭代器并返回 -1
                if (tupobj == NULL):
                    Py_DECREF(it)
                    return -1
                # 将 obj 设置为元组的第一个元素
                PyTuple_SET_ITEM(tupobj, 0, obj)
                # 根据格式字符串 format 创建 Unicode 字符串对象 obj
                obj = PyUnicode_FromString((const char *)format)
                # 如果创建失败，则释放 tupobj 和迭代器并返回 -1
                if (obj == NULL):
                    Py_DECREF(tupobj)
                    Py_DECREF(it)
                    return -1
                # 格式化 Unicode 字符串 obj 和 tupobj，返回格式化后的字符串 strobj
                strobj = PyUnicode_Format(obj, tupobj)
                # 释放 obj 和 tupobj 对象的引用计数
                Py_DECREF(obj)
                Py_DECREF(tupobj)
                # 如果格式化失败，则释放迭代器并返回 -1
                if (strobj == NULL):
                    Py_DECREF(it)
                    return -1
            # 将 Unicode 字符串 strobj 转换为 ASCII 字符串 byteobj
            byteobj = PyUnicode_AsASCIIString(strobj)
            # 在多线程环境下允许线程，获取 byteobj 的大小 n2
            NPY_BEGIN_ALLOW_THREADS
            n2 = PyBytes_GET_SIZE(byteobj)
            # 将 byteobj 的内容写入文件 fp
            n = fwrite(PyBytes_AS_STRING(byteobj), 1, n2, fp)
            NPY_END_ALLOW_THREADS
            # 释放 byteobj 对象的引用计数
            Py_DECREF(byteobj)
            # 如果写入的字节数小于应写入的字节数 n2，则抛出异常并释放 strobj 和迭代器，返回 -1
            if (n < n2):
                PyErr_Format(PyExc_OSError,
                             "problem writing element %" NPY_INTP_FMT " to file", it->index)
                Py_DECREF(strobj)
                Py_DECREF(it)
                return -1
            # 如果不是最后一个元素，则在文件中写入分隔符 sep
            if (it->index != it->size - 1):
                if (fwrite(sep, 1, n3, fp) < n3):
                    PyErr_Format(PyExc_OSError,
                                 "problem writing separator to file")
                    Py_DECREF(strobj)
                    Py_DECREF(it)
                    return -1
            # 释放 strobj 对象的引用计数
            Py_DECREF(strobj)
            # 移动到迭代器的下一个位置
            PyArray_ITER_NEXT(it)
        # 释放迭代器对象 it 的引用计数
        Py_DECREF(it)
    # 返回 0 表示成功
    return 0
/*NUMPY_API*/
NPY_NO_EXPORT PyObject *
PyArray_ToString(PyArrayObject *self, NPY_ORDER order)
{
    npy_intp numbytes;              // 存储数组总字节数
    npy_intp i;                     // 循环计数器
    char *dptr;                     // 指向目标字符串的指针
    int elsize;                     // 数组元素大小
    PyObject *ret;                  // 返回的Python对象
    PyArrayIterObject *it;          // 数组迭代器对象

    if (order == NPY_ANYORDER)
        order = PyArray_ISFORTRAN(self) ? NPY_FORTRANORDER : NPY_CORDER;

    /*        if (PyArray_TYPE(self) == NPY_OBJECT) {
              PyErr_SetString(PyExc_ValueError, "a string for the data" \
              "in an object array is not appropriate");
              return NULL;
              }
    */
    // 如果数组类型为对象数组，抛出错误并返回空指针
    numbytes = PyArray_NBYTES(self); // 计算数组的总字节数
    // 如果数组是C连续的且按顺序为C，或者是Fortran连续的且按顺序为Fortran，直接从数组数据创建字符串对象
    if ((PyArray_IS_C_CONTIGUOUS(self) && (order == NPY_CORDER))
        || (PyArray_IS_F_CONTIGUOUS(self) && (order == NPY_FORTRANORDER))) {
        ret = PyBytes_FromStringAndSize(PyArray_DATA(self), (Py_ssize_t) numbytes);
    }
    else {
        PyObject *new;
        if (order == NPY_FORTRANORDER) {
            /* iterators are always in C-order */
            // 如果按Fortran顺序，需要先转置数组为C顺序
            new = PyArray_Transpose(self, NULL);
            if (new == NULL) {
                return NULL;
            }
        }
        else {
            Py_INCREF(self);
            new = (PyObject *)self;
        }
        // 创建数组迭代器
        it = (PyArrayIterObject *)PyArray_IterNew(new);
        Py_DECREF(new);
        if (it == NULL) {
            return NULL;
        }
        // 根据数组总字节数创建新的字符串对象
        ret = PyBytes_FromStringAndSize(NULL, (Py_ssize_t) numbytes);
        if (ret == NULL) {
            Py_DECREF(it);
            return NULL;
        }
        // 将数组数据复制到字符串中
        dptr = PyBytes_AS_STRING(ret);
        i = it->size;
        elsize = PyArray_ITEMSIZE(self);
        while (i--) {
            memcpy(dptr, it->dataptr, elsize);
            dptr += elsize;
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return ret;  // 返回字符串对象
}

/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_FillWithScalar(PyArrayObject *arr, PyObject *obj)
{

    if (PyArray_FailUnlessWriteable(arr, "assignment destination") < 0) {
        return -1;
    }

    /*
     * If we knew that the output array has at least one element, we would
     * not actually need a helping buffer, we always null it, just in case.
     *
     * (The longlong here should help with alignment.)
     */
    // 用于存储标量值的缓冲区
    npy_longlong value_buffer_stack[4] = {0};
    char *value_buffer_heap = NULL;
    char *value = (char *)value_buffer_stack;
    PyArray_Descr *descr = PyArray_DESCR(arr);

    // 如果数组元素大小超过堆栈缓冲区大小，需要分配堆内存
    if ((size_t)descr->elsize > sizeof(value_buffer_stack)) {
        /* We need a large temporary buffer... */
        // 分配足够大的临时缓冲区
        value_buffer_heap = PyObject_Calloc(1, descr->elsize);
        if (value_buffer_heap == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        value = value_buffer_heap;
    }
    // 将Python对象obj打包成数组元素
    if (PyArray_Pack(descr, value, obj) < 0) {
        PyMem_FREE(value_buffer_heap);
        return -1;
    }

    /*
     * There is no cast anymore, the above already coerced using scalar
     * coercion rules
     */
    // 不再需要强制转换，上述操作已经使用标量强制转换规则执行了
    // 返回成功状态

    return 0;
}
    # 调用 raw_array_assign_scalar 函数执行数组的标量赋值操作，并将返回值赋给 retcode
    int retcode = raw_array_assign_scalar(
            PyArray_NDIM(arr), PyArray_DIMS(arr), descr,
            PyArray_BYTES(arr), PyArray_STRIDES(arr),
            descr, value);

    # 检查数据类型 descr 是否需要引用计数检查，如果需要，则调用 PyArray_ClearBuffer 清除缓冲区
    if (PyDataType_REFCHK(descr)) {
        PyArray_ClearBuffer(descr, value, 0, 1, 1);
    }
    # 释放 value_buffer_heap 指向的内存块
    PyMem_FREE(value_buffer_heap);

    # 返回 retcode 作为函数的返回值
    return retcode;
/*
 * Internal function to fill an array with zeros.
 * Used in einsum and dot, which ensures the dtype is, in some sense, numerical
 * and not a str or struct
 *
 * dst: The destination array.
 * wheremask: If non-NULL, a boolean mask specifying where to set the values.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignZero(PyArrayObject *dst,
                   PyArrayObject *wheremask)
{
    // 初始化返回码
    int retcode = 0;
    
    // 如果目标数组是对象数组
    if (PyArray_ISOBJECT(dst)) {
        // 创建一个表示整数0的 Python 对象
        PyObject * pZero = PyLong_FromLong(0);
        // 使用 PyArray_AssignRawScalar 函数将整数0赋值给目标数组
        retcode = PyArray_AssignRawScalar(dst, PyArray_DESCR(dst),
                                     (char *)&pZero, wheremask, NPY_SAFE_CASTING);
        // 释放 Python 对象 pZero 的引用计数
        Py_DECREF(pZero);
    }
    else {
        /* 创建一个原始的布尔标量，其值为 False */
        // 从 NPY_BOOL 类型创建一个描述符
        PyArray_Descr *bool_dtype = PyArray_DescrFromType(NPY_BOOL);
        if (bool_dtype == NULL) {
            return -1;  // 如果创建描述符失败则返回 -1
        }
        npy_bool value = 0;  // 初始化布尔值为 False

        // 使用 PyArray_AssignRawScalar 函数将布尔值赋值给目标数组
        retcode = PyArray_AssignRawScalar(dst, bool_dtype, (char *)&value,
                                          wheremask, NPY_SAFE_CASTING);

        // 释放布尔类型的描述符
        Py_DECREF(bool_dtype);
    }
    // 返回操作结果码
    return retcode;
}



/*NUMPY_API
 * Copy an array.
 */
NPY_NO_EXPORT PyObject *
PyArray_NewCopy(PyArrayObject *obj, NPY_ORDER order)
{
    // 定义返回的数组对象指针
    PyArrayObject *ret;

    // 如果传入的数组对象为空，则抛出数值错误异常并返回空指针
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "obj is NULL in PyArray_NewCopy");
        return NULL;
    }

    // 使用 PyArray_NewLikeArray 函数根据原数组对象创建一个相似的新数组对象
    ret = (PyArrayObject *)PyArray_NewLikeArray(obj, order, NULL, 1);
    if (ret == NULL) {
        return NULL;  // 如果创建新数组失败则返回空指针
    }

    // 使用 PyArray_AssignArray 函数将原数组的数据复制到新数组中
    if (PyArray_AssignArray(ret, obj, NULL, NPY_UNSAFE_CASTING) < 0) {
        Py_DECREF(ret);
        return NULL;  // 如果复制数据失败则释放新数组并返回空指针
    }

    // 返回新创建的数组对象
    return (PyObject *)ret;
}



/*NUMPY_API
 * View
 * steals a reference to type -- accepts NULL
 */
NPY_NO_EXPORT PyObject *
PyArray_View(PyArrayObject *self, PyArray_Descr *type, PyTypeObject *pytype)
{
    // 定义返回的数组对象指针
    PyArrayObject *ret = NULL;
    PyArray_Descr *dtype;
    PyTypeObject *subtype;
    int flags;

    // 如果传入的类型非空，则使用该类型，否则使用 self 的类型
    if (pytype) {
        subtype = pytype;
    }
    else {
        subtype = Py_TYPE(self);
    }

    // 获取 self 的数据描述符和标志位
    dtype = PyArray_DESCR(self);
    flags = PyArray_FLAGS(self);

    // 增加数据描述符的引用计数，并使用 PyArray_NewFromDescr_int 函数创建新的数组对象
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr_int(
            subtype, dtype,
            PyArray_NDIM(self), PyArray_DIMS(self), PyArray_STRIDES(self),
            PyArray_DATA(self),
            flags, (PyObject *)self, (PyObject *)self,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    if (ret == NULL) {
        Py_XDECREF(type);
        return NULL;  // 如果创建新数组对象失败则释放 type 并返回空指针
    }

    // 如果传入的类型非空，则将该类型设置为新数组对象的 dtype 属性
    if (type != NULL) {
        if (PyObject_SetAttrString((PyObject *)ret, "dtype",
                                   (PyObject *)type) < 0) {
            Py_DECREF(ret);
            Py_DECREF(type);
            return NULL;  // 如果设置 dtype 属性失败则释放 ret 和 type 并返回空指针
        }
        Py_DECREF(type);
    }
    // 返回新创建的数组对象
    return (PyObject *)ret;
}
```