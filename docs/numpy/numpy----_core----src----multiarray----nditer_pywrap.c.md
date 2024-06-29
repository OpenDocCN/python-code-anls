# `.\numpy\numpy\_core\src\multiarray\nditer_pywrap.c`

```py
/*
 * This file implements the CPython wrapper of NpyIter
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 */

/* Define to prevent deprecated NumPy API usage */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/* Define to enable multiarray module */
#define _MULTIARRAYMODULE

/* Clean PY_SSIZE_T */
#define PY_SSIZE_T_CLEAN

#include <Python.h>         /* Python header file */
#include <structmember.h>   /* Defines struct members */

#include "numpy/arrayobject.h"  /* NumPy's array object */
#include "npy_config.h"         /* NumPy configuration */

#include "alloc.h"              /* Allocation utilities */
#include "common.h"             /* Common utilities */
#include "conversion_utils.h"   /* Conversion utilities */
#include "ctors.h"              /* Constructors */

/* Function declarations not part of the public NumPy C API */
npy_bool npyiter_has_writeback(NpyIter *iter);

/* Structure definition for the Python object */
typedef struct NewNpyArrayIterObject_tag NewNpyArrayIterObject;

struct NewNpyArrayIterObject_tag {
    PyObject_HEAD               /* Python object header */
    NpyIter *iter;              /* NpyIter object for iteration */
    char started, finished;     /* Flags indicating iteration state */
    NewNpyArrayIterObject *nested_child;    /* Child iterator for nested iteration */
    NpyIter_IterNextFunc *iternext;         /* Function pointer for iteration */
    NpyIter_GetMultiIndexFunc *get_multi_index;  /* Function pointer for multi-index */
    char **dataptrs;            /* Data pointers */
    PyArray_Descr **dtypes;     /* Array descriptors */
    PyArrayObject **operands;   /* Array operands */
    npy_intp *innerstrides;     /* Inner strides */
    npy_intp *innerloopsizeptr; /* Inner loop size pointer */
    char readflags[NPY_MAXARGS];    /* Read flags */
    char writeflags[NPY_MAXARGS];   /* Write flags */
};

/* Function to cache values from NpyIter into NewNpyArrayIterObject */
static int npyiter_cache_values(NewNpyArrayIterObject *self)
{
    NpyIter *iter = self->iter;

    /* Retrieve iternext function */
    self->iternext = NpyIter_GetIterNext(iter, NULL);
    if (self->iternext == NULL) {
        return -1;
    }

    /* Retrieve get_multi_index function if available */
    if (NpyIter_HasMultiIndex(iter) && !NpyIter_HasDelayedBufAlloc(iter)) {
        self->get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);
    }
    else {
        self->get_multi_index = NULL;
    }

    /* Retrieve data pointers, descriptors, and operands */
    self->dataptrs = NpyIter_GetDataPtrArray(iter);
    self->dtypes = NpyIter_GetDescrArray(iter);
    self->operands = NpyIter_GetOperandArray(iter);

    /* Retrieve inner strides and inner loop size pointer if external loop exists */
    if (NpyIter_HasExternalLoop(iter)) {
        self->innerstrides = NpyIter_GetInnerStrideArray(iter);
        self->innerloopsizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    }
    else {
        self->innerstrides = NULL;
        self->innerloopsizeptr = NULL;
    }

    /* Retrieve read and write flags */
    NpyIter_GetReadFlags(iter, self->readflags);
    NpyIter_GetWriteFlags(iter, self->writeflags);
    return 0;
}

/* Function to create a new instance of NewNpyArrayIterObject */
static PyObject *
npyiter_new(PyTypeObject *subtype, PyObject *NPY_UNUSED(args),
            PyObject *NPY_UNUSED(kwds))
{
    NewNpyArrayIterObject *self;

    /* Allocate memory for NewNpyArrayIterObject */
    self = (NewNpyArrayIterObject *)subtype->tp_alloc(subtype, 0);
    if (self != NULL) {
        self->iter = NULL;
        self->nested_child = NULL;
    }

    return (PyObject *)self;
}

/* Function to convert global flags */
static int
NpyIter_GlobalFlagsConverter(PyObject *flags_in, npy_uint32 *flags)
{
    npy_uint32 tmpflags = 0;
    int iflags, nflags;

    PyObject *f;
    char *str = NULL;
    Py_ssize_t length = 0;
    npy_uint32 flag;

    /* Check if flags_in is None or NULL */
    if (flags_in == NULL || flags_in == Py_None) {
        return 1;
    }
    # 如果 flags_in 不是 Python 元组或列表，则抛出值错误异常
    if (!PyTuple_Check(flags_in) && !PyList_Check(flags_in)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator global flags must be a list or tuple of strings");
        return 0;
    }

    # 获取 flags_in 中元素的数量
    nflags = PySequence_Size(flags_in);

    # 将 tmpflags 的值按位或运算到 flags 指针所指向的变量中
    *flags |= tmpflags;
    
    # 返回操作成功标志
    return 1;
}

static int
NpyIter_OpFlagsConverter(PyObject *op_flags_in,
                         npy_uint32 *op_flags)
{
    int iflags, nflags;
    npy_uint32 flag;

    // 检查输入的 op_flags_in 是否为元组或列表，否则报错
    if (!PyTuple_Check(op_flags_in) && !PyList_Check(op_flags_in)) {
        PyErr_SetString(PyExc_ValueError,
                "op_flags must be a tuple or array of per-op flag-tuples");
        return 0;
    }

    // 获取 op_flags_in 的长度
    nflags = PySequence_Size(op_flags_in);

    // 将 op_flags 初始化为 0
    *op_flags = 0;
    }

    // 返回成功
    return 1;
}

static int
npyiter_convert_op_flags_array(PyObject *op_flags_in,
                         npy_uint32 *op_flags_array, npy_intp nop)
{
    npy_intp iop;

    // 检查输入的 op_flags_in 是否为元组或列表，否则报错
    if (!PyTuple_Check(op_flags_in) && !PyList_Check(op_flags_in)) {
        PyErr_SetString(PyExc_ValueError,
                "op_flags must be a tuple or array of per-op flag-tuples");
        return 0;
    }

    // 如果 op_flags_in 的长度不等于 nop，跳转到 try_single_flags 标签处处理
    if (PySequence_Size(op_flags_in) != nop) {
        goto try_single_flags;
    }

    // 遍历 op_flags_in 中的每个元素
    for (iop = 0; iop < nop; ++iop) {
        PyObject *f = PySequence_GetItem(op_flags_in, iop);
        if (f == NULL) {
            return 0;
        }
        /* 如果第一个元素是字符串，则尝试作为一组标志 */
        if (iop == 0 && (PyBytes_Check(f) || PyUnicode_Check(f))) {
            Py_DECREF(f);
            goto try_single_flags;
        }
        // 转换操作标志并存入 op_flags_array[iop] 中
        if (NpyIter_OpFlagsConverter(f,
                        &op_flags_array[iop]) != 1) {
            Py_DECREF(f);
            return 0;
        }

        Py_DECREF(f);
    }

    // 返回成功
    return 1;

try_single_flags:
    // 尝试将 op_flags_in 转换为单一的操作标志，并复制给所有操作数
    if (NpyIter_OpFlagsConverter(op_flags_in,
                        &op_flags_array[0]) != 1) {
        return 0;
    }

    // 将第一个操作标志复制给其余的操作标志
    for (iop = 1; iop < nop; ++iop) {
        op_flags_array[iop] = op_flags_array[0];
    }

    // 返回成功
    return 1;
}

static int
npyiter_convert_dtypes(PyObject *op_dtypes_in,
                        PyArray_Descr **op_dtypes,
                        npy_intp nop)
{
    npy_intp iop;

    /*
     * 如果输入不是 dtype 的元组，尝试直接将其转换为一个 dtype，并复制给所有操作数。
     */
    if ((!PyTuple_Check(op_dtypes_in) && !PyList_Check(op_dtypes_in)) ||
                                    PySequence_Size(op_dtypes_in) != nop) {
        // 跳转到 try_single_dtype 标签处处理
        goto try_single_dtype;
    }

    // 遍历 op_dtypes_in 中的每个元素
    for (iop = 0; iop < nop; ++iop) {
        PyObject *dtype = PySequence_GetItem(op_dtypes_in, iop);
        if (dtype == NULL) {
            npy_intp i;
            for (i = 0; i < iop; ++i ) {
                Py_XDECREF(op_dtypes[i]);
            }
            return 0;
        }

        // 尝试将对象转换为 dtype 描述符
        if (PyArray_DescrConverter2(dtype, &op_dtypes[iop]) != 1) {
            npy_intp i;
            for (i = 0; i < iop; ++i ) {
                Py_XDECREF(op_dtypes[i]);
            }
            Py_DECREF(dtype);
            PyErr_Clear();
            // 转到 try_single_dtype 标签处处理
            goto try_single_dtype;
        }

        Py_DECREF(dtype);
    }

    // 返回成功
    return 1;

try_single_dtype:
    # 如果成功将 op_dtypes_in 转换为 PyArray_Descr 结构，并且返回值为 1，则执行以下操作
    if (PyArray_DescrConverter2(op_dtypes_in, &op_dtypes[0]) == 1) {
        # 对于每个操作数（除第一个外），将其数据类型设置为第一个操作数的数据类型
        for (iop = 1; iop < nop; ++iop) {
            op_dtypes[iop] = op_dtypes[0];
            # 增加对 op_dtypes[iop] 的引用计数，以确保其内存不会在使用期间被释放
            Py_XINCREF(op_dtypes[iop]);
        }
        # 返回成功标志 1，表示操作成功
        return 1;
    }
    
    # 如果未能成功转换 op_dtypes_in 或者返回值不为 1，则返回失败标志 0
    return 0;
# 定义一个静态函数，用于将传入的操作轴数据转换为适合的格式，并存储在op_axes中，同时设置oa_ndim
static int
npyiter_convert_op_axes(PyObject *op_axes_in, int nop,
                        int **op_axes, int *oa_ndim)
{
    PyObject *a;
    int iop;

    # 检查op_axes_in是否为元组或列表，并且长度与nop相等
    if ((!PyTuple_Check(op_axes_in) && !PyList_Check(op_axes_in)) ||
                                PySequence_Size(op_axes_in) != nop) {
        # 设置异常信息，表明op_axes必须是与操作数个数相匹配的元组或列表
        PyErr_SetString(PyExc_ValueError,
                "op_axes must be a tuple/list matching the number of ops");
        return 0;  // 返回0表示出错
    }

    *oa_ndim = -1;  // 初始化oa_ndim为-1

    /* Copy the tuples into op_axes */
    # 将元组或列表中的每个元素复制到op_axes中
    for (iop = 0; iop < nop; ++iop) {
        int idim;
        a = PySequence_GetItem(op_axes_in, iop);  // 获取op_axes_in中的第iop个元素
        if (a == NULL) {
            return 0;  // 如果获取失败，返回0表示出错
        }
        if (a == Py_None) {
            op_axes[iop] = NULL;  // 如果元素是None，则在op_axes中对应位置置为NULL
        } else {
            if (!PyTuple_Check(a) && !PyList_Check(a)) {
                PyErr_SetString(PyExc_ValueError,
                        "Each entry of op_axes must be None "
                        "or a tuple/list");
                Py_DECREF(a);
                return 0;  // 如果元素不是None也不是元组或列表，则返回0表示出错
            }
            if (*oa_ndim == -1) {
                *oa_ndim = PySequence_Size(a);  // 第一次设置oa_ndim为a的长度
                if (*oa_ndim > NPY_MAXDIMS) {
                    PyErr_SetString(PyExc_ValueError,
                            "Too many dimensions in op_axes");
                    Py_DECREF(a);
                    return 0;  // 如果oa_ndim超过了NPY_MAXDIMS，则返回0表示出错
                }
            }
            if (PySequence_Size(a) != *oa_ndim) {
                PyErr_SetString(PyExc_ValueError,
                        "Each entry of op_axes must have the same size");
                Py_DECREF(a);
                return 0;  // 如果a的长度与oa_ndim不相等，则返回0表示出错
            }
            for (idim = 0; idim < *oa_ndim; ++idim) {
                PyObject *v = PySequence_GetItem(a, idim);  // 获取a中的第idim个元素
                if (v == NULL) {
                    Py_DECREF(a);
                    return 0;  // 如果获取失败，返回0表示出错
                }
                /* numpy.newaxis is None */
                // numpy.newaxis对应的是None
                if (v == Py_None) {
                    op_axes[iop][idim] = -1;  // 如果v是None，则在op_axes中设置为-1
                }
                else {
                    op_axes[iop][idim] = PyArray_PyIntAsInt(v);  // 将v转换为整数并存储在op_axes中
                    if (op_axes[iop][idim] == -1 &&
                                                PyErr_Occurred()) {
                        Py_DECREF(a);
                        Py_DECREF(v);
                        return 0;  // 如果转换失败，则返回0表示出错
                    }
                }
                Py_DECREF(v);  // 释放v的引用计数
            }
        }
        Py_DECREF(a);  // 释放a的引用计数
    }

    if (*oa_ndim == -1) {
        PyErr_SetString(PyExc_ValueError,
                "If op_axes is provided, at least one list of axes "
                "must be contained within it");
        return 0;  // 如果oa_ndim仍为-1，则设置异常信息并返回0表示出错
    }

    return 1;  // 成功转换返回1
}
    int iop, nop;

    /* 定义整数变量 iop 和 nop，用于迭代和存储操作数的数量 */

    /* 检查输入参数 op_in 是否为元组或列表 */
    if (PyTuple_Check(op_in) || PyList_Check(op_in)) {
        nop = PySequence_Size(op_in);  // 获取 op_in 中元素的数量
        if (nop == 0) {
            PyErr_SetString(PyExc_ValueError,
                    "Must provide at least one operand");
            return 0;  // 如果没有操作数，返回错误
        }
        if (nop > NPY_MAXARGS) {
            PyErr_SetString(PyExc_ValueError, "Too many operands");
            return 0;  // 如果操作数超过最大限制，返回错误
        }

        /* 遍历操作数列表 op_in */
        for (iop = 0; iop < nop; ++iop) {
            PyObject *item = PySequence_GetItem(op_in, iop);  // 获取列表中的每个元素
            if (item == NULL) {
                npy_intp i;
                for (i = 0; i < iop; ++i) {
                    Py_XDECREF(op[i]);  // 如果获取元素失败，释放已获取的元素内存
                }
                return 0;  // 返回错误
            }
            else if (item == Py_None) {
                Py_DECREF(item);
                item = NULL;  // 如果元素为 None，则释放并置为 NULL
            }
            /* 将获取的元素转换为 PyArrayObject 类型，存入 op 数组 */
            op[iop] = (PyArrayObject *)item;
        }
    }
    else {
        nop = 1;  // 如果 op_in 不是元组或列表，则操作数为 1
        /* 将 op_in 转换为 PyArrayObject 类型，存入 op 数组 */
        Py_INCREF(op_in);
        op[0] = (PyArrayObject *)op_in;
    }

    *nop_out = nop;  // 将操作数数量存入输出参数 nop_out

    /* 处理操作标志 op_flags */
    if (op_flags_in == NULL || op_flags_in == Py_None) {
        /* 如果未提供 op_flags_in，则根据 op 是否为 NULL 设置默认标志 */
        for (iop = 0; iop < nop; ++iop) {
            /*
             * 默认情况下，将 NULL 操作数设置为只写并标记为分配内存，其他情况设置为只读。
             * 若要写入提供的操作数，必须手动指定写标志。
             */
            if (op[iop] == NULL) {
                op_flags[iop] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
            }
            else {
                op_flags[iop] = NPY_ITER_READONLY;
            }
        }
    }
    else if (npyiter_convert_op_flags_array(op_flags_in,
                                      op_flags, nop) != 1) {
        /* 如果提供了 op_flags_in，则调用函数将其转换为 op_flags 数组 */
        for (iop = 0; iop < nop; ++iop) {
            Py_XDECREF(op[iop]);  // 如果转换失败，释放已获取的元素内存
        }
        *nop_out = 0;
        return 0;  // 返回错误
    }

    /* 现在我们有了标志，将所有操作数转换为数组形式 */
    // 遍历操作数数组，处理每个操作数
    for (iop = 0; iop < nop; ++iop) {
        // 检查操作数是否为非空
        if (op[iop] != NULL) {
            // 声明一个 PyArrayObject 对象指针
            PyArrayObject *ao;
            // 初始化 fromanyflags 标志为 0
            int fromanyflags = 0;

            // 如果操作标志表明可读写或仅写入，则设置 NPY_ARRAY_WRITEBACKIFCOPY 标志
            if (op_flags[iop] & (NPY_ITER_READWRITE | NPY_ITER_WRITEONLY)) {
                fromanyflags |= NPY_ARRAY_WRITEBACKIFCOPY;
            }
            // 将 Python 对象 op[iop] 转换为 PyArrayObject 对象
            ao = (PyArrayObject *)PyArray_FROM_OF((PyObject *)op[iop],
                                                  fromanyflags);
            // 如果转换失败，处理异常情况
            if (ao == NULL) {
                // 如果发生异常且异常匹配 TypeError 类型
                if (PyErr_Occurred() &&
                    PyErr_ExceptionMatches(PyExc_TypeError)) {
                    // 设置错误消息，指出迭代器操作数标记为可写，但是无法通过 WRITEBACKIFCOPY 进行写入
                    PyErr_SetString(PyExc_TypeError,
                            "Iterator operand is flagged as writeable, "
                            "but is an object which cannot be written "
                            "back to via WRITEBACKIFCOPY");
                }
                // 释放之前创建的 PyArrayObject 对象
                for (iop = 0; iop < nop; ++iop) {
                    Py_DECREF(op[iop]);
                }
                // 设置输出参数 nop_out 为 0，返回失败
                *nop_out = 0;
                return 0;
            }
            // 释放原来的 Python 对象 op[iop]
            Py_DECREF(op[iop]);
            // 将转换后的 PyArrayObject 对象赋值回原来的 op[iop]
            op[iop] = ao;
        }
    }

    // 处理完所有操作数后，返回成功
    return 1;
    }
    // 静态函数npyiter_init的实现，用于初始化NpyArrayIter对象
    static int
    npyiter_init(NewNpyArrayIterObject *self, PyObject *args, PyObject *kwds)
    {
        // 静态字符串数组，用于解析函数参数
        static char *kwlist[] = {"op", "flags", "op_flags", "op_dtypes",
                                 "order", "casting", "op_axes", "itershape",
                                 "buffersize",
                                 NULL};

        // 初始化PyObject指针变量
        PyObject *op_in = NULL, *op_flags_in = NULL,
                    *op_dtypes_in = NULL, *op_axes_in = NULL;

        // 初始化整型变量和数组
        int iop, nop = 0;
        PyArrayObject *op[NPY_MAXARGS];
        npy_uint32 flags = 0;
        NPY_ORDER order = NPY_KEEPORDER;
        NPY_CASTING casting = NPY_SAFE_CASTING;
        npy_uint32 op_flags[NPY_MAXARGS];
        PyArray_Descr *op_request_dtypes[NPY_MAXARGS];
        int oa_ndim = -1;
        int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
        int *op_axes[NPY_MAXARGS];
        PyArray_Dims itershape = {NULL, -1};
        int buffersize = 0;

        // 如果迭代器已经初始化，则返回错误
        if (self->iter != NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator was already initialized");
            return -1;
        }

        // 解析输入参数并进行类型转换，若解析失败则返回错误
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&OOO&O&OO&i:nditer", kwlist,
                        &op_in,
                        NpyIter_GlobalFlagsConverter, &flags,
                        &op_flags_in,
                        &op_dtypes_in,
                        PyArray_OrderConverter, &order,
                        PyArray_CastingConverter, &casting,
                        &op_axes_in,
                        PyArray_OptionalIntpConverter, &itershape,
                        &buffersize)) {
            npy_free_cache_dim_obj(itershape);
            return -1;
        }

        /* Set the dtypes and ops to all NULL to start */
        // 将op_request_dtypes数组初始化为NULL
        memset(op_request_dtypes, 0, sizeof(op_request_dtypes));

        /* op and op_flags */
        // 转换操作数和操作标志
        if (npyiter_convert_ops(op_in, op_flags_in, op, op_flags, &nop)
                                                            != 1) {
            goto fail;
        }

        /* op_request_dtypes */
        // 转换操作数据类型
        if (op_dtypes_in != NULL && op_dtypes_in != Py_None &&
                npyiter_convert_dtypes(op_dtypes_in,
                                       op_request_dtypes, nop) != 1) {
            goto fail;
        }

        /* op_axes */
        // 转换操作轴信息
        if (op_axes_in != NULL && op_axes_in != Py_None) {
            // 初始化op_axes数组指向op_axes_arrays的各个数组
            for (iop = 0; iop < nop; ++iop) {
                op_axes[iop] = op_axes_arrays[iop];
            }

            // 执行操作轴转换函数
            if (npyiter_convert_op_axes(op_axes_in, nop,
                                        op_axes, &oa_ndim) != 1) {
                goto fail;
            }
        }

        // 检查itershape的长度，与op_axes的维度信息进行匹配
        if (itershape.len != -1) {
            if (oa_ndim == -1) {
                oa_ndim = itershape.len;
                memset(op_axes, 0, sizeof(op_axes[0]) * nop);
            }
            else if (oa_ndim != itershape.len) {
                PyErr_SetString(PyExc_ValueError,
                            "'op_axes' and 'itershape' must have the same number "
                            "of entries equal to the iterator ndim");
                goto fail;
            }
        }
    // 使用 NpyIter_AdvancedNew 函数创建高级迭代器对象，并初始化 self->iter
    self->iter = NpyIter_AdvancedNew(nop, op, flags, order, casting, op_flags,
                                     op_request_dtypes,
                                     oa_ndim, oa_ndim >= 0 ? op_axes : NULL,
                                     itershape.ptr,
                                     buffersize);

    // 检查 self->iter 是否为 NULL，如果是则跳转到失败处理代码块
    if (self->iter == NULL) {
        goto fail;
    }

    /* 缓存一些值，供成员函数使用 */
    // 如果 npyiter_cache_values 函数返回值小于 0，则跳转到失败处理代码块
    if (npyiter_cache_values(self) < 0) {
        goto fail;
    }

    // 检查 self->iter 的迭代大小是否为 0，根据结果设置 self->started 和 self->finished
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    // 释放 itershape 所持有的对象引用
    npy_free_cache_dim_obj(itershape);

    /* 释放对 ops 和 dtypes 的引用 */
    // 遍历释放 op 和 op_request_dtypes 数组中的引用对象
    for (iop = 0; iop < nop; ++iop) {
        Py_XDECREF(op[iop]);
        Py_XDECREF(op_request_dtypes[iop]);
    }

    // 成功执行，返回 0 表示没有错误
    return 0;
fail:
    // 释放缓存中的迭代形状对象
    npy_free_cache_dim_obj(itershape);
    // 循环释放操作数数组中的对象引用
    for (iop = 0; iop < nop; ++iop) {
        Py_XDECREF(op[iop]);
        Py_XDECREF(op_request_dtypes[iop]);
    }
    // 返回错误状态
    return -1;
}



NPY_NO_EXPORT PyObject *
NpyIter_NestedIters(PyObject *NPY_UNUSED(self),
                    PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"op", "axes", "flags", "op_flags",
                             "op_dtypes", "order",
                             "casting", "buffersize",
                             NULL};

    PyObject *op_in = NULL, *axes_in = NULL,
            *op_flags_in = NULL, *op_dtypes_in = NULL;

    int iop, nop = 0, inest, nnest = 0;
    PyArrayObject *op[NPY_MAXARGS];
    npy_uint32 flags = 0, flags_inner;
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_SAFE_CASTING;
    npy_uint32 op_flags[NPY_MAXARGS], op_flags_inner[NPY_MAXARGS];
    PyArray_Descr *op_request_dtypes[NPY_MAXARGS],
                  *op_request_dtypes_inner[NPY_MAXARGS];
    int op_axes_data[NPY_MAXDIMS];
    int *nested_op_axes[NPY_MAXDIMS];
    int nested_naxes[NPY_MAXDIMS], iaxes, naxes;
    int negones[NPY_MAXDIMS];
    char used_axes[NPY_MAXDIMS];
    int buffersize = 0;

    PyObject *ret = NULL;

    // 解析传入参数并进行类型转换
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O&OOO&O&i", kwlist,
                    &op_in,
                    &axes_in,
                    NpyIter_GlobalFlagsConverter, &flags,
                    &op_flags_in,
                    &op_dtypes_in,
                    PyArray_OrderConverter, &order,
                    PyArray_CastingConverter, &casting,
                    &buffersize)) {
        return NULL;
    }

    /* axes */
    // 检查传入的 axes 是否为元组或列表
    if (!PyTuple_Check(axes_in) && !PyList_Check(axes_in)) {
        PyErr_SetString(PyExc_ValueError,
                "axes must be a tuple of axis arrays");
        return NULL;
    }
    // 获取传入 axes 的长度
    nnest = PySequence_Size(axes_in);
    // axes 至少需要两个条目才能进行嵌套迭代
    if (nnest < 2) {
        PyErr_SetString(PyExc_ValueError,
                "axes must have at least 2 entries for nested iteration");
        return NULL;
    }
    // 初始化轴数计数器和轴使用标志数组
    naxes = 0;
    memset(used_axes, 0, NPY_MAXDIMS);
    for (inest = 0; inest < nnest; ++inest) {
        // 从 axes_in 序列中获取第 inest 个元素
        PyObject *item = PySequence_GetItem(axes_in, inest);
        npy_intp i;
        // 如果获取失败，返回空指针
        if (item == NULL) {
            return NULL;
        }
        // 检查 item 是否为元组或列表类型，如果不是，返回错误信息并释放 item
        if (!PyTuple_Check(item) && !PyList_Check(item)) {
            PyErr_SetString(PyExc_ValueError,
                    "Each item in axes must be a an integer tuple");
            Py_DECREF(item);
            return NULL;
        }
        // 记录当前 item 的长度到 nested_naxes[inest]
        nested_naxes[inest] = PySequence_Size(item);
        // 检查当前轴数与之前的总轴数是否超过 NPY_MAXDIMS，如果是，返回错误信息并释放 item
        if (naxes + nested_naxes[inest] > NPY_MAXDIMS) {
            PyErr_SetString(PyExc_ValueError,
                    "Too many axes given");
            Py_DECREF(item);
            return NULL;
        }
        // 遍历 item 中的元素
        for (i = 0; i < nested_naxes[inest]; ++i) {
            // 获取 item 中第 i 个元素
            PyObject *v = PySequence_GetItem(item, i);
            npy_intp axis;
            // 如果获取失败，释放 item 并返回空指针
            if (v == NULL) {
                Py_DECREF(item);
                return NULL;
            }
            // 将 v 转换为长整型并赋值给 axis
            axis = PyLong_AsLong(v);
            Py_DECREF(v);
            // 检查 axis 是否在有效范围内，如果不是，返回错误信息并释放 item
            if (axis < 0 || axis >= NPY_MAXDIMS) {
                PyErr_SetString(PyExc_ValueError,
                        "An axis is out of bounds");
                Py_DECREF(item);
                return NULL;
            }
            // 检查当前 axis 是否已被使用，如果是，返回错误信息并释放 item
            if (used_axes[axis] != 0) {
                PyErr_SetString(PyExc_ValueError,
                        "An axis is used more than once");
                Py_DECREF(item);
                return NULL;
            }
            // 标记当前 axis 已被使用
            used_axes[axis] = 1;
            // 将 axis 添加到 op_axes_data 中，从 naxes 开始的位置
            op_axes_data[naxes+i] = axis;
        }
        // 将当前 op_axes_data 的子数组赋值给 nested_op_axes[inest]
        nested_op_axes[inest] = &op_axes_data[naxes];
        // 更新总轴数 naxes
        naxes += nested_naxes[inest];
        // 释放 item
        Py_DECREF(item);
    }

    /* op and op_flags */
    // 调用 npyiter_convert_ops 函数处理 op 和 op_flags
    if (npyiter_convert_ops(op_in, op_flags_in, op, op_flags, &nop)
                                                        != 1) {
        return NULL;
    }

    /* Set the dtypes to all NULL to start as well */
    // 将 op_request_dtypes 数组初始化为全 NULL
    memset(op_request_dtypes, 0, sizeof(op_request_dtypes[0])*nop);
    // 将 op_request_dtypes_inner 数组初始化为全 NULL
    memset(op_request_dtypes_inner, 0,
                        sizeof(op_request_dtypes_inner[0])*nop);

    /* op_request_dtypes */
    // 如果 op_dtypes_in 不为空且不是 Py_None，则调用 npyiter_convert_dtypes 处理 op_dtypes_in
    if (op_dtypes_in != NULL && op_dtypes_in != Py_None &&
            npyiter_convert_dtypes(op_dtypes_in,
                                   op_request_dtypes, nop) != 1) {
        goto fail;
    }

    // 创建一个包含 nnest 个元素的新元组 ret
    ret = PyTuple_New(nnest);
    // 如果创建失败，跳转到 fail 标签处理错误
    if (ret == NULL) {
        goto fail;
    }

    /* For broadcasting allocated arrays */
    // 将 negones 数组的前 naxes 个元素初始化为 -1
    for (iaxes = 0; iaxes < naxes; ++iaxes) {
        negones[iaxes] = -1;
    }

    /*
     * Clear any unnecessary ALLOCATE flags, so we can use them
     * to indicate exactly the allocated outputs.  Also, separate
     * the inner loop flags.
     */
    for (iop = 0; iop < nop; ++iop) {
        // 检查是否需要分配内存，并且操作数组不为NULL时，清除分配标志位
        if ((op_flags[iop] & NPY_ITER_ALLOCATE) && op[iop] != NULL) {
            op_flags[iop] &= ~NPY_ITER_ALLOCATE;
        }

        /*
         * 清除允许在内部循环中进行复制或输出分配的标志位。
         */
        op_flags_inner[iop] = op_flags[iop] & ~(NPY_ITER_COPY |
                                                NPY_ITER_UPDATEIFCOPY |
                                                NPY_ITER_ALLOCATE);
        /*
         * 如果启用了缓冲，并且没有复制操作，
         * 则清除 nbo_aligned 标志位并且去除外部循环的数据类型。
         */
        if ((flags & (NPY_ITER_BUFFERED)) &&
                !(op_flags[iop] & (NPY_ITER_COPY |
                                   NPY_ITER_UPDATEIFCOPY |
                                   NPY_ITER_ALLOCATE))) {
            op_flags[iop] &= ~(NPY_ITER_NBO | NPY_ITER_ALIGNED | NPY_ITER_CONTIG);
            op_request_dtypes_inner[iop] = op_request_dtypes[iop];
            op_request_dtypes[iop] = NULL;
        }
    }

    /* 只有内部循环可以使用缓冲，而且没有内部标志 */
    flags_inner = flags & ~NPY_ITER_COMMON_DTYPE;
    flags &= ~(NPY_ITER_EXTERNAL_LOOP |
               NPY_ITER_BUFFERED);

    }

    /* 释放对操作数组和数据类型的引用 */
    for (iop = 0; iop < nop; ++iop) {
        Py_XDECREF(op[iop]);
        Py_XDECREF(op_request_dtypes[iop]);
        Py_XDECREF(op_request_dtypes_inner[iop]);
    }

    /* 设置嵌套子引用 */
    for (inest = 0; inest < nnest-1; ++inest) {
        NewNpyArrayIterObject *iter;
        iter = (NewNpyArrayIterObject *)PyTuple_GET_ITEM(ret, inest);
        /*
         * 指示在每次迭代步骤中重置哪个迭代器的基指针。
         */
        iter->nested_child =
                (NewNpyArrayIterObject *)PyTuple_GET_ITEM(ret, inest+1);
        Py_INCREF(iter->nested_child);
        /*
         * 需要进行嵌套重置，以便所有迭代器指向正确的数据。
         */
        if (NpyIter_ResetBasePointers(iter->nested_child->iter,
                                      iter->dataptrs, NULL) != NPY_SUCCEED) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    return ret;
/*
 * Free resources held by the iterator upon encountering a failure condition.
 * This function decrements references to Python objects and returns NULL.
 */
fail:
    for (iop = 0; iop < nop; ++iop) {
        Py_XDECREF(op[iop]);  // Decrement reference count for op[iop]
        Py_XDECREF(op_request_dtypes[iop]);  // Decrement reference count for op_request_dtypes[iop]
        Py_XDECREF(op_request_dtypes_inner[iop]);  // Decrement reference count for op_request_dtypes_inner[iop]
    }
    return NULL;  // Return NULL to indicate failure
}



/*
 * Deallocates memory and resources held by a NewNpyArrayIterObject instance.
 * If an iterator is present, it handles error handling and cleanup of resources.
 */
static void
npyiter_dealloc(NewNpyArrayIterObject *self)
{
    if (self->iter) {
        /* Store current exception state to preserve it */
        PyObject *exc, *val, *tb;
        PyErr_Fetch(&exc, &val, &tb);

        // Check if writeback is needed
        if (npyiter_has_writeback(self->iter)) {
            // Issue a runtime warning if temporary data hasn't been written back
            if (PyErr_WarnEx(PyExc_RuntimeWarning,
                    "Temporary data has not been written back to one of the "
                    "operands. Typically nditer is used as a context manager "
                    "otherwise 'close' must be called before reading iteration "
                    "results.", 1) < 0) {
                PyObject *s;

                s = PyUnicode_FromString("npyiter_dealloc");
                if (s) {
                    PyErr_WriteUnraisable(s);  // Write an unraisable error with a custom message
                    Py_DECREF(s);
                }
                else {
                    PyErr_WriteUnraisable(Py_None);  // Write an unraisable error with a default message
                }
            }
        }

        // Deallocate the iterator
        if (!NpyIter_Deallocate(self->iter)) {
            PyErr_WriteUnraisable(Py_None);  // Write an unraisable error if deallocation fails
        }

        // Reset instance variables
        self->iter = NULL;
        Py_XDECREF(self->nested_child);  // Decrement reference count for nested_child
        self->nested_child = NULL;

        // Restore previous exception state
        PyErr_Restore(exc, val, tb);
    }

    // Free memory allocated for the object
    Py_TYPE(self)->tp_free((PyObject*)self);
}



/*
 * Resets base pointers for nested iterators and checks for success.
 * Updates the status of iteration for the current iterator.
 * Returns NPY_SUCCEED on success, NPY_FAIL on failure.
 */
static int
npyiter_resetbasepointers(NewNpyArrayIterObject *self)
{
    while (self->nested_child) {
        // Reset base pointers for the nested iterator
        if (NpyIter_ResetBasePointers(self->nested_child->iter,
                                        self->dataptrs, NULL) != NPY_SUCCEED) {
            return NPY_FAIL;  // Return failure status if reset fails
        }
        self = self->nested_child;

        // Update iteration status based on iterator size
        if (NpyIter_GetIterSize(self->iter) == 0) {
            self->started = 1;
            self->finished = 1;
        }
        else {
            self->started = 0;
            self->finished = 0;
        }
    }

    return NPY_SUCCEED;  // Return success status
}



/*
 * Resets the iterator to its initial state.
 * Checks for validity of the iterator and performs reset operations.
 * Returns None on success, raises ValueError on invalid iterator.
 */
static PyObject *
npyiter_reset(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    // Check if iterator is valid
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;  // Return NULL indicating error
    }

    // Reset the iterator
    if (NpyIter_Reset(self->iter, NULL) != NPY_SUCCEED) {
        return NULL;  // Return NULL indicating error
    }

    // Update iteration status based on iterator size
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    // Get multi-index function if not already retrieved
    if (self->get_multi_index == NULL && NpyIter_HasMultiIndex(self->iter)) {
        self->get_multi_index = NpyIter_GetGetMultiIndex(self->iter, NULL);
    }

    /* If there is nesting, reset nested iterators */
    if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
        return NULL;  // Return NULL indicating error
    }

    Py_RETURN_NONE;  // Return Python None object on success
}
/*
 * 复制给定的 NpyArray 迭代器对象并返回其 Python 对象表示
 */
PyObject *npyiter_copy(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    NewNpyArrayIterObject *iter;

    // 检查迭代器是否有效，若无效则返回 ValueError 异常
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    /* 分配新的迭代器对象 */
    iter = (NewNpyArrayIterObject *)npyiter_new(&NpyIter_Type, NULL, NULL);
    if (iter == NULL) {
        return NULL;
    }

    /* 复制 C 语言迭代器 */
    iter->iter = NpyIter_Copy(self->iter);
    if (iter->iter == NULL) {
        Py_DECREF(iter);
        return NULL;
    }

    /* 缓存一些值以供成员函数使用 */
    if (npyiter_cache_values(iter) < 0) {
        Py_DECREF(iter);
        return NULL;
    }

    // 复制起始和结束标记
    iter->started = self->started;
    iter->finished = self->finished;

    return (PyObject *)iter;
}

/*
 * 迭代器的下一个操作，返回一个布尔值的 Python 对象表示
 */
static PyObject *
npyiter_iternext(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    // 检查迭代器是否有效，迭代函数是否可用，并且迭代未完成且成功迭代下一个元素
    if (self->iter != NULL && self->iternext != NULL &&
                        !self->finished && self->iternext(self->iter)) {
        /* 如果有嵌套迭代，需要重置嵌套的迭代器 */
        if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
            return NULL;
        }

        Py_RETURN_TRUE;
    }
    else {
        // 如果发生错误，返回 NULL
        if (PyErr_Occurred()) {
            /* 类型转换错误，缓冲区将在重置或释放时清理 */
            return NULL;
        }
        // 设置迭代完成标记并返回 False
        self->finished = 1;
        Py_RETURN_FALSE;
    }
}

/*
 * 移除迭代器中的指定轴
 */
static PyObject *
npyiter_remove_axis(NewNpyArrayIterObject *self, PyObject *args)
{
    int axis = 0;

    // 检查迭代器是否有效，若无效则返回 ValueError 异常
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 解析参数，获取要移除的轴的索引
    if (!PyArg_ParseTuple(args, "i:remove_axis", &axis)) {
        return NULL;
    }

    // 在迭代器中移除指定的轴
    if (NpyIter_RemoveAxis(self->iter, axis) != NPY_SUCCEED) {
        return NULL;
    }
    /* 移除轴后使缓存的值无效 */
    if (npyiter_cache_values(self) < 0) {
        return NULL;
    }
    /* 移除轴也会重置迭代器 */
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    Py_RETURN_NONE;
}

/*
 * 移除迭代器中的多重索引
 */
static PyObject *
npyiter_remove_multi_index(
    NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    // 检查迭代器是否有效，若无效则返回 ValueError 异常
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 移除迭代器中的多重索引
    NpyIter_RemoveMultiIndex(self->iter);
    /* 移除多重索引后使缓存的值无效 */
    npyiter_cache_values(self);
    /* 移除多重索引也会重置迭代器 */
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    else {
        self->started = 0;
        self->finished = 0;
    }

    Py_RETURN_NONE;
}

/*
 * 启用迭代器的外部循环模式
 */
static PyObject *
npyiter_enable_external_loop(
    NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    // 检查迭代器是否为 NULL，如果是则设置一个值错误异常并返回 NULL
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 启用迭代器的外部循环功能
    NpyIter_EnableExternalLoop(self->iter);
    /* EnableExternalLoop invalidates cached values */
    // 启用外部循环功能会使缓存的数值无效
    npyiter_cache_values(self);
    /* EnableExternalLoop also resets the iterator */
    // 启用外部循环功能也会重置迭代器
    // 如果迭代器的大小为 0，则将 started 和 finished 标记为 1
    if (NpyIter_GetIterSize(self->iter) == 0) {
        self->started = 1;
        self->finished = 1;
    }
    // 否则将 started 和 finished 标记为 0
    else {
        self->started = 0;
        self->finished = 0;
    }

    // 返回 Python 中的 None 对象，表示成功执行但不返回任何值
    Py_RETURN_NONE;
static PyObject *
npyiter_debug_print(NewNpyArrayIterObject *self, PyObject *NPY_UNUSED(args))
{
    // 检查迭代器是否存在，如果存在则调用 NpyIter_DebugPrint 输出调试信息
    if (self->iter != NULL) {
        NpyIter_DebugPrint(self->iter);
    }
    // 如果迭代器不存在，打印信息表示迭代器为空
    else {
        printf("Iterator: (nil)\n");
    }

    // 返回 None
    Py_RETURN_NONE;
}

NPY_NO_EXPORT PyObject *
npyiter_seq_item(NewNpyArrayIterObject *self, Py_ssize_t i);

static PyObject *
npyiter_value_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;

    npy_intp iop, nop;

    // 检查迭代器是否为 NULL 或者已经结束
    if (self->iter == NULL || self->finished) {
        // 设置错误信息并返回 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    nop = NpyIter_GetNOp(self->iter);

    /* Return an array  or tuple of arrays with the values */
    // 如果操作数个数为 1，则调用 npyiter_seq_item 获取第一个元素
    if (nop == 1) {
        ret = npyiter_seq_item(self, 0);
    }
    // 否则创建一个元组，并逐个获取每个操作数的值
    else {
        ret = PyTuple_New(nop);
        if (ret == NULL) {
            return NULL;
        }
        for (iop = 0; iop < nop; ++iop) {
            PyObject *a = npyiter_seq_item(self, iop);
            if (a == NULL) {
                Py_DECREF(ret);
                return NULL;
            }
            PyTuple_SET_ITEM(ret, iop, a);
        }
    }

    return ret;
}

static PyObject *
npyiter_operands_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;

    npy_intp iop, nop;
    PyArrayObject **operands;

    // 检查迭代器是否为 NULL
    if (self->iter == NULL) {
        // 设置错误信息并返回 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }
    nop = NpyIter_GetNOp(self->iter);
    operands = self->operands;

    // 创建一个元组，并将每个操作数对象添加到元组中
    ret = PyTuple_New(nop);
    if (ret == NULL) {
        return NULL;
    }
    for (iop = 0; iop < nop; ++iop) {
        PyObject *operand = (PyObject *)operands[iop];

        Py_INCREF(operand);
        PyTuple_SET_ITEM(ret, iop, operand);
    }

    return ret;
}

static PyObject *
npyiter_itviews_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;

    npy_intp iop, nop;

    // 检查迭代器是否为 NULL
    if (self->iter == NULL) {
        // 设置错误信息并返回 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    // 创建一个元组，并将每个迭代视图对象添加到元组中
    ret = PyTuple_New(nop);
    if (ret == NULL) {
        return NULL;
    }
    for (iop = 0; iop < nop; ++iop) {
        PyArrayObject *view = NpyIter_GetIterView(self->iter, iop);

        if (view == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(ret, iop, (PyObject *)view);
    }

    return ret;
}

static PyObject *
npyiter_next(NewNpyArrayIterObject *self)
{
    // 检查迭代器是否为 NULL，或者迭代过程已经结束，或者 iternext 函数为空
    if (self->iter == NULL || self->iternext == NULL ||
                self->finished) {
        // 返回 NULL，表示迭代结束或者出现错误
        return NULL;
    }

    /*
     * Use the started flag for the Python iteration protocol to work
     * when buffering is enabled.
     */

    // 这里通常用来实现迭代器的下一个值获取逻辑，但是具体细节未给出，需要根据实际情况补充实现

    // 返回下一个迭代值，具体逻辑应根据实际需求补充
}
    # 检查迭代器是否已经开始迭代
    if (self->started) {
        # 如果已经开始迭代，尝试获取下一个元素
        if (!self->iternext(self->iter)) {
            /*
             * 如果出现类型转换错误，可能会在这里设置错误（或者没有错误导致 StopIteration）。
             * 缓冲区可能只能在稍后清理。
             */
            # 如果无法获取下一个元素，标记迭代结束
            self->finished = 1;
            # 返回空指针表示迭代结束
            return NULL;
        }

        /* 如果存在嵌套迭代器，应该重置这些嵌套迭代器 */
        # 如果有嵌套迭代器，重置它们的基指针
        if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
            # 如果重置基指针失败，返回空指针
            return NULL;
        }
    }
    # 设置迭代器已经开始迭代的标志
    self->started = 1;

    # 返回当前迭代器位置的值
    return npyiter_value_get(self, NULL);
# 获取迭代器的形状信息并返回作为 Python 元组
static PyObject *
npyiter_shape_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    npy_intp ndim, shape[NPY_MAXDIMS];  // 定义变量：维度数和形状数组

    if (self->iter == NULL || self->finished) {  // 检查迭代器是否为空或已完成
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");  // 设置异常信息并返回 NULL
        return NULL;
    }

    if (NpyIter_GetShape(self->iter, shape) == NPY_SUCCEED) {  // 获取迭代器的形状信息
        ndim = NpyIter_GetNDim(self->iter);  // 获取迭代器的维度数
        return PyArray_IntTupleFromIntp(ndim, shape);  // 将形状数组转换为 Python 元组并返回
    }

    return NULL;  // 获取形状信息失败，返回 NULL
}

# 获取迭代器的多重索引并返回作为 Python 元组
static PyObject *
npyiter_multi_index_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    npy_intp ndim, multi_index[NPY_MAXDIMS];  // 定义变量：维度数和多重索引数组

    if (self->iter == NULL || self->finished) {  // 检查迭代器是否为空或已完成
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");  // 设置异常信息并返回 NULL
        return NULL;
    }

    if (self->get_multi_index != NULL) {  // 检查是否定义了获取多重索引的函数
        ndim = NpyIter_GetNDim(self->iter);  // 获取迭代器的维度数
        self->get_multi_index(self->iter, multi_index);  // 调用获取多重索引的函数
        return PyArray_IntTupleFromIntp(ndim, multi_index);  // 将多重索引数组转换为 Python 元组并返回
    }
    else {  // 若未定义获取多重索引的函数
        if (!NpyIter_HasMultiIndex(self->iter)) {  // 检查迭代器是否未跟踪多重索引
            PyErr_SetString(PyExc_ValueError,
                    "Iterator is not tracking a multi-index");  // 设置异常信息并返回 NULL
            return NULL;
        }
        else if (NpyIter_HasDelayedBufAlloc(self->iter)) {  // 检查迭代器是否使用了延迟缓冲区分配
            PyErr_SetString(PyExc_ValueError,
                    "Iterator construction used delayed buffer allocation, "
                    "and no reset has been done yet");  // 设置异常信息并返回 NULL
            return NULL;
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator is in an invalid state");  // 其他情况下设置异常信息并返回 NULL
            return NULL;
        }
    }
}

# 设置迭代器的多重索引
static int
npyiter_multi_index_set(
        NewNpyArrayIterObject *self, PyObject *value, void *NPY_UNUSED(ignored))
{
    npy_intp idim, ndim, multi_index[NPY_MAXDIMS];  // 定义变量：单个索引、维度数和多重索引数组

    if (value == NULL) {  // 检查值是否为空
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete nditer multi_index");  // 设置异常信息并返回 -1
        return -1;
    }
    if (self->iter == NULL) {  // 检查迭代器是否为空
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");  // 设置异常信息并返回 -1
        return -1;
    }
    # 检查迭代器是否具有多索引功能
    if (NpyIter_HasMultiIndex(self->iter)) {
        # 获取迭代器的维度数
        ndim = NpyIter_GetNDim(self->iter);
        
        # 检查传入的值是否为序列类型
        if (!PySequence_Check(value)) {
            PyErr_SetString(PyExc_ValueError,
                    "multi_index must be set with a sequence");
            return -1;
        }
        
        # 检查传入的序列长度是否与迭代器的维度数一致
        if (PySequence_Size(value) != ndim) {
            PyErr_SetString(PyExc_ValueError,
                    "Wrong number of indices");
            return -1;
        }
        
        # 遍历每个维度，从序列中获取对应的索引值并存入multi_index数组
        for (idim = 0; idim < ndim; ++idim) {
            PyObject *v = PySequence_GetItem(value, idim);
            multi_index[idim] = PyLong_AsLong(v);
            Py_DECREF(v);
            # 检查转换过程中是否出现错误
            if (error_converting(multi_index[idim])) {
                return -1;
            }
        }
        
        # 将迭代器定位到指定的多索引位置
        if (NpyIter_GotoMultiIndex(self->iter, multi_index) != NPY_SUCCEED) {
            return -1;
        }
        
        # 重置迭代器的状态
        self->started = 0;
        self->finished = 0;

        /* 如果存在嵌套迭代器，应该重置嵌套的迭代器指针 */
        if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
            return -1;
        }

        # 操作成功，返回0表示没有错误
        return 0;
    }
    else {
        # 如果迭代器不支持多索引，则返回错误信息
        PyErr_SetString(PyExc_ValueError,
                "Iterator is not tracking a multi-index");
        return -1;
    }
static PyObject *
npyiter_index_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    // 检查迭代器是否为空或已经结束
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    // 检查迭代器是否具有索引
    if (NpyIter_HasIndex(self->iter)) {
        // 获取迭代器当前索引值并返回作为 Python 整数对象
        npy_intp ind = *NpyIter_GetIndexPtr(self->iter);
        return PyLong_FromLong(ind);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator does not have an index");
        return NULL;
    }
}

static int
npyiter_index_set(
        NewNpyArrayIterObject *self, PyObject *value, void *NPY_UNUSED(ignored))
{
    // 检查是否试图删除迭代器的索引
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete nditer index");
        return -1;
    }
    // 检查迭代器是否为空
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return -1;
    }

    // 检查迭代器是否具有索引
    if (NpyIter_HasIndex(self->iter)) {
        npy_intp ind;
        // 尝试将 Python 对象转换为长整型索引值
        ind = PyLong_AsLong(value);
        if (error_converting(ind)) {
            return -1;
        }
        // 跳转至指定索引位置
        if (NpyIter_GotoIndex(self->iter, ind) != NPY_SUCCEED) {
            return -1;
        }
        // 重置迭代器状态
        self->started = 0;
        self->finished = 0;

        /* 如果存在嵌套迭代器，则需要重置嵌套迭代器的基本指针 */
        if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
            return -1;
        }

        return 0;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                "Iterator does not have an index");
        return -1;
    }
}

static PyObject *
npyiter_iterindex_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    // 检查迭代器是否为空或已经结束
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    // 返回当前迭代器的迭代索引作为 Python 整数对象
    return PyLong_FromLong(NpyIter_GetIterIndex(self->iter));
}

static int
npyiter_iterindex_set(
        NewNpyArrayIterObject *self, PyObject *value, void *NPY_UNUSED(ignored))
{
    npy_intp iterindex;

    // 检查是否试图删除迭代器的迭代索引
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete nditer iterindex");
        return -1;
    }
    // 检查迭代器是否为空
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return -1;
    }

    // 尝试将 Python 对象转换为长整型迭代索引值
    iterindex = PyLong_AsLong(value);
    if (error_converting(iterindex)) {
        return -1;
    }
    // 跳转至指定迭代索引位置
    if (NpyIter_GotoIterIndex(self->iter, iterindex) != NPY_SUCCEED) {
        return -1;
    }
    // 重置迭代器状态
    self->started = 0;
    self->finished = 0;

    /* 如果存在嵌套迭代器，则需要重置嵌套迭代器的基本指针 */
    if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
        return -1;
    }

    return 0;
}

static PyObject *
npyiter_iterrange_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    npy_intp istart = 0, iend = 0;
    PyObject *ret;
    # 检查迭代器是否为NULL，如果是则抛出值错误异常并返回NULL
    if (self->iter == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    # 获取迭代器的索引范围，存储在istart和iend中
    NpyIter_GetIterIndexRange(self->iter, &istart, &iend);

    # 创建一个包含两个元素的元组对象ret
    ret = PyTuple_New(2);
    if (ret == NULL) {
        return NULL;  # 如果创建失败，则返回NULL
    }

    # 将istart和iend分别转换为Python的长整型对象，并设置到元组ret的对应位置
    PyTuple_SET_ITEM(ret, 0, PyLong_FromLong(istart));
    PyTuple_SET_ITEM(ret, 1, PyLong_FromLong(iend));

    # 返回构建好的元组对象ret
    return ret;
static int
npyiter_iterrange_set(
        NewNpyArrayIterObject *self, PyObject *value, void *NPY_UNUSED(ignored))
{
    npy_intp istart = 0, iend = 0;

    if (value == NULL) {
        // 如果传入的值为 NULL，设置错误信息并返回 -1
        PyErr_SetString(PyExc_AttributeError,
                "Cannot delete nditer iterrange");
        return -1;
    }
    if (self->iter == NULL) {
        // 如果迭代器为 NULL，设置错误信息并返回 -1
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return -1;
    }

    // 解析传入的 value，期望格式为 "nn"，分别存入 istart 和 iend
    if (!PyArg_ParseTuple(value, "nn", &istart, &iend)) {
        return -1;
    }

    // 调用 NpyIter_ResetToIterIndexRange 重置迭代器范围，如果失败则返回 -1
    if (NpyIter_ResetToIterIndexRange(self->iter, istart, iend, NULL)
                                                    != NPY_SUCCEED) {
        return -1;
    }

    // 根据 istart 和 iend 的关系设置 started 和 finished
    if (istart < iend) {
        self->started = self->finished = 0;
    }
    else {
        self->started = self->finished = 1;
    }

    // 如果 get_multi_index 为 NULL 且迭代器支持多索引，获取多索引函数
    if (self->get_multi_index == NULL && NpyIter_HasMultiIndex(self->iter)) {
        self->get_multi_index = NpyIter_GetGetMultiIndex(self->iter, NULL);
    }

    /* 如果存在嵌套迭代器，应当重置嵌套迭代器的基础指针 */
    // 调用 npyiter_resetbasepointers 重置嵌套迭代器的基础指针，如果失败返回 -1
    if (npyiter_resetbasepointers(self) != NPY_SUCCEED) {
        return -1;
    }

    // 返回成功标志
    return 0;
}

static PyObject *
npyiter_has_delayed_bufalloc_get(
        NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        // 如果迭代器为 NULL，设置错误信息并返回 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 检查迭代器是否有延迟缓冲区分配，返回相应的 Python 布尔对象
    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_iterationneedsapi_get(
        NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        // 如果迭代器为 NULL，设置错误信息并返回 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 检查迭代器是否需要 API 支持，返回相应的 Python 布尔对象
    if (NpyIter_IterationNeedsAPI(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_has_multi_index_get(
        NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        // 如果迭代器为 NULL，设置错误信息并返回 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 检查迭代器是否有多索引，返回相应的 Python 布尔对象
    if (NpyIter_HasMultiIndex(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_has_index_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    if (self->iter == NULL) {
        // 如果迭代器为 NULL，设置错误信息并返回 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 检查迭代器是否有索引，返回相应的 Python 布尔对象
    if (NpyIter_HasIndex(self->iter)) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
npyiter_dtypes_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    PyObject *ret;

    npy_intp iop, nop;
    PyArray_Descr **dtypes;

    if (self->iter == NULL) {
        // 如果迭代器为 NULL，设置错误信息并返回 NULL
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 获取迭代器的操作数和操作描述符数组
    nop = NpyIter_GetNOp(self->iter);
    dtypes = NpyIter_GetDescrArray(self->iter);

    // 构造返回值，包含迭代器中所有操作的数据类型
    ret = PyTuple_New(nop);
    for (iop = 0; iop < nop; ++iop) {
        PyTuple_SET_ITEM(ret, iop, PyArray_Descr_WRAP(dtypes[iop]));
    }

    // 返回包含操作数据类型的元组对象
    return ret;
}
    # 创建一个新的元组对象，元组长度为 nop
    ret = PyTuple_New(nop);
    # 检查元组对象是否创建成功，如果为 NULL，则返回 NULL
    if (ret == NULL) {
        return NULL;
    }
    # 获取 self 对象的 dtypes 属性
    dtypes = self->dtypes;
    # 遍历从 dtypes 中获取的数据类型数组
    for (iop = 0; iop < nop; ++iop) {
        # 获取当前索引 iop 处的数据类型对象指针
        PyArray_Descr *dtype = dtypes[iop];

        # 增加数据类型对象的引用计数，以防止在元组中被销毁
        Py_INCREF(dtype);
        # 将数据类型对象转换为 PyObject* 并设置为元组 ret 的第 iop 个元素
        PyTuple_SET_ITEM(ret, iop, (PyObject *)dtype);
    }

    # 返回填充好的元组对象 ret
    return ret;
static PyObject *
npyiter_ndim_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    // 检查迭代器是否为空
    if (self->iter == NULL) {
        // 如果迭代器为空，设置错误信息并返回空对象
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 返回迭代器的维度作为一个 Python 长整型对象
    return PyLong_FromLong(NpyIter_GetNDim(self->iter));
}

static PyObject *
npyiter_nop_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    // 检查迭代器是否为空
    if (self->iter == NULL) {
        // 如果迭代器为空，设置错误信息并返回空对象
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 返回迭代器操作数的数量作为一个 Python 长整型对象
    return PyLong_FromLong(NpyIter_GetNOp(self->iter));
}

static PyObject *
npyiter_itersize_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    // 检查迭代器是否为空
    if (self->iter == NULL) {
        // 如果迭代器为空，设置错误信息并返回空对象
        PyErr_SetString(PyExc_ValueError,
                "Iterator is invalid");
        return NULL;
    }

    // 返回迭代器的迭代大小作为一个 Python 长整型对象
    return PyLong_FromLong(NpyIter_GetIterSize(self->iter));
}

static PyObject *
npyiter_finished_get(NewNpyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    // 检查迭代器是否为空或者是否已经完成迭代
    if (self->iter == NULL || !self->finished) {
        // 如果迭代器为空或者未完成迭代，返回 Python False 对象
        Py_RETURN_FALSE;
    }
    else {
        // 否则返回 Python True 对象
        Py_RETURN_TRUE;
    }
}

NPY_NO_EXPORT Py_ssize_t
npyiter_seq_length(NewNpyArrayIterObject *self)
{
    // 检查迭代器是否为空
    if (self->iter == NULL) {
        // 如果迭代器为空，返回长度为 0
        return 0;
    }
    else {
        // 返回迭代器操作数的数量
        return NpyIter_GetNOp(self->iter);
    }
}

NPY_NO_EXPORT PyObject *
npyiter_seq_item(NewNpyArrayIterObject *self, Py_ssize_t i)
{
    npy_intp ret_ndim;
    npy_intp nop, innerloopsize, innerstride;
    char *dataptr;
    PyArray_Descr *dtype;
    int has_external_loop;
    Py_ssize_t i_orig = i;

    // 检查迭代器是否为空或者是否已经完成迭代
    if (self->iter == NULL || self->finished) {
        // 如果迭代器为空或者已经完成迭代，设置错误信息并返回空对象
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return NULL;
    }

    // 检查是否有延迟的缓冲区分配
    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        // 如果有延迟的缓冲区分配，设置错误信息并返回空对象
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return NULL;
    }
    nop = NpyIter_GetNOp(self->iter);

    /* Negative indexing */
    // 处理负索引
    if (i < 0) {
        i += nop;
    }

    // 检查索引是否超出范围
    if (i < 0 || i >= nop) {
        // 如果索引超出范围，设置错误信息并返回空对象
        PyErr_Format(PyExc_IndexError,
                "Iterator operand index %zd is out of bounds", i_orig);
        return NULL;
    }

#if 0
    /*
     * This check is disabled because it prevents things like
     * np.add(it[0], it[1], it[2]), where it[2] is a write-only
     * parameter.  When write-only, the value of it[i] is
     * likely random junk, as if it were allocated with an
     * np.empty(...) call.
     */
    // 这个检查被禁用，因为它阻止像 np.add(it[0], it[1], it[2]) 这样的操作，
    // 其中 it[2] 是一个只写的参数。当只写时，it[i] 的值可能是随机垃圾，
    // 就像它是通过 np.empty(...) 调用分配的一样。
#endif

    // 获取数据指针和数据类型
    dataptr = self->dataptrs[i];
    dtype = self->dtypes[i];
    has_external_loop = NpyIter_HasExternalLoop(self->iter);

    // 如果有外部循环
    if (has_external_loop) {
        // 获取内部循环大小和内部步长，并设置返回维度为 1
        innerloopsize = *self->innerloopsizeptr;
        innerstride = self->innerstrides[i];
        ret_ndim = 1;
    }
    else {
        innerloopsize = 1;
        innerstride = 0;
        /* 如果迭代器遍历每个元素，则返回数组标量 */
        ret_ndim = 0;
    }



    Py_INCREF(dtype);
    /* 根据给定的描述符和基础数据创建一个新的数组对象 */
    return PyArray_NewFromDescrAndBase(
            &PyArray_Type, dtype,
            ret_ndim, &innerloopsize, &innerstride, dataptr,
            self->writeflags[i] ? NPY_ARRAY_WRITEABLE : 0,
            NULL, (PyObject *)self);
    // 返回一个 Python 对象，该对象代表从 self 迭代器中切片获取的子序列
    NPY_NO_EXPORT PyObject *
    npyiter_seq_slice(NewNpyArrayIterObject *self,
                        Py_ssize_t ilow, Py_ssize_t ihigh)
    {
        PyObject *ret;  // 用于存储返回的 Python 元组对象
        npy_intp nop;   // 迭代器中操作数的数量
        Py_ssize_t i;   // 循环变量

        // 如果迭代器为空或已经完成迭代，抛出 ValueError 异常
        if (self->iter == NULL || self->finished) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator is past the end");
            return NULL;
        }

        // 如果迭代器使用了延迟的缓冲区分配，抛出 ValueError 异常
        if (NpyIter_HasDelayedBufAlloc(self->iter)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator construction used delayed buffer allocation, "
                    "and no reset has been done yet");
            return NULL;
        }
        nop = NpyIter_GetNOp(self->iter);  // 获取迭代器操作数的数量
        if (ilow < 0) {
            ilow = 0;  // 如果 ilow 小于 0，则将其设为 0
        }
        else if (ilow >= nop) {
            ilow = nop-1;  // 如果 ilow 大于或等于操作数的数量，将其设为 nop-1
        }
        if (ihigh < ilow) {
            ihigh = ilow;  // 如果 ihigh 小于 ilow，则将其设为 ilow
        }
        else if (ihigh > nop) {
            ihigh = nop;  // 如果 ihigh 大于操作数的数量，则将其设为 nop
        }

        // 创建一个元组对象 ret，长度为 ihigh-ilow
        ret = PyTuple_New(ihigh-ilow);
        if (ret == NULL) {
            return NULL;
        }
        // 填充元组对象 ret，循环从 ilow 到 ihigh-1
        for (i = ilow; i < ihigh ; ++i) {
            // 调用 npyiter_seq_item 获取索引 i 对应的元素，并将其放入元组 ret 中
            PyObject *item = npyiter_seq_item(self, i);
            if (item == NULL) {
                Py_DECREF(ret);
                return NULL;
            }
            PyTuple_SET_ITEM(ret, i-ilow, item);
        }
        return ret;  // 返回填充好的元组对象
    }

    // 设置 self 迭代器中索引 i 处的元素为 v
    NPY_NO_EXPORT int
    npyiter_seq_ass_item(NewNpyArrayIterObject *self, Py_ssize_t i, PyObject *v)
    {

        npy_intp nop, innerloopsize, innerstride;  // 迭代器中操作数的数量，内部循环大小和步长
        char *dataptr;  // 指向数据的指针
        PyArray_Descr *dtype;  // 数组元素的描述符
        PyArrayObject *tmp;  // 临时数组对象
        int ret, has_external_loop;  // 返回值，是否有外部循环标志
        Py_ssize_t i_orig = i;  // 原始索引 i

        // 如果 v 为 NULL，抛出 TypeError 异常，无法删除迭代器的元素
        if (v == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "Cannot delete iterator elements");
            return -1;
        }

        // 如果迭代器为空或已经完成迭代，抛出 ValueError 异常
        if (self->iter == NULL || self->finished) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator is past the end");
            return -1;
        }

        // 如果迭代器使用了延迟的缓冲区分配，抛出 ValueError 异常
        if (NpyIter_HasDelayedBufAlloc(self->iter)) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator construction used delayed buffer allocation, "
                    "and no reset has been done yet");
            return -1;
        }
        nop = NpyIter_GetNOp(self->iter);  // 获取迭代器操作数的数量

        /* Negative indexing */
        // 处理负索引
        if (i < 0) {
            i += nop;
        }

        // 检查索引 i 是否在有效范围内
        if (i < 0 || i >= nop) {
            PyErr_Format(PyExc_IndexError,
                    "Iterator operand index %zd is out of bounds", i_orig);
            return -1;
        }
        // 检查索引 i 对应的操作数是否可写
        if (!self->writeflags[i]) {
            PyErr_Format(PyExc_RuntimeError,
                    "Iterator operand %zd is not writeable", i_orig);
            return -1;
        }

        // 获取数据指针和数据类型描述符
        dataptr = self->dataptrs[i];
        dtype = self->dtypes[i];
        // 检查是否有外部循环
        has_external_loop = NpyIter_HasExternalLoop(self->iter);

        // 如果有外部循环，设置内部循环大小和步长
        if (has_external_loop) {
            innerloopsize = *self->innerloopsizeptr;
            innerstride = self->innerstrides[i];
        }
        else {
            innerloopsize = 1;
            innerstride = 0;
        }

        // 增加数据类型描述符的引用计数
        Py_INCREF(dtype);
    // 创建一个新的 NumPy 数组对象 tmp，使用给定的数据类型 dtype，
    // 维度为 1，大小为 innerloopsize，步长为 innerstride，数据指针为 dataptr，
    // 设置为可写模式 NPY_ARRAY_WRITEABLE，没有基础对象（NULL）。
    tmp = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                                1, &innerloopsize,
                                &innerstride, dataptr,
                                NPY_ARRAY_WRITEABLE, NULL);
    // 如果创建失败，返回 -1
    if (tmp == NULL) {
        return -1;
    }

    // 将数组对象 v 复制到 tmp 中，返回结果
    ret = PyArray_CopyObject(tmp, v);
    // 释放 tmp 的引用计数
    Py_DECREF(tmp);
    // 返回复制操作的结果
    return ret;
static int
npyiter_seq_ass_slice(NewNpyArrayIterObject *self, Py_ssize_t ilow,
                Py_ssize_t ihigh, PyObject *v)
{
    npy_intp nop;  // 定义变量nop，用于存储迭代器操作数的数量
    Py_ssize_t i;  // 定义变量i，用于迭代器操作

    if (v == NULL) {  // 如果传入的对象v为空
        PyErr_SetString(PyExc_TypeError,
                "Cannot delete iterator elements");  // 抛出类型错误异常，指明不能删除迭代器元素
        return -1;  // 返回-1表示操作失败
    }

    if (self->iter == NULL || self->finished) {  // 如果迭代器指针为空或者迭代器已经完成
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");  // 抛出值错误异常，指明迭代器已经超出末尾
        return -1;  // 返回-1表示操作失败
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {  // 如果迭代器构造过程使用了延迟缓冲区分配
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");  // 抛出值错误异常，指明迭代器构造过程中使用了延迟缓冲区分配且尚未重置
        return -1;  // 返回-1表示操作失败
    }
    nop = NpyIter_GetNOp(self->iter);  // 获取迭代器操作数的数量
    if (ilow < 0) {  // 如果传入的ilow小于0
        ilow = 0;  // 将ilow设为0
    }
    else if (ilow >= nop) {  // 如果传入的ilow大于等于迭代器操作数的数量
        ilow = nop-1;  // 将ilow设为迭代器操作数的数量减1
    }
    if (ihigh < ilow) {  // 如果传入的ihigh小于ilow
        ihigh = ilow;  // 将ihigh设为ilow
    }
    else if (ihigh > nop) {  // 如果传入的ihigh大于迭代器操作数的数量
        ihigh = nop;  // 将ihigh设为迭代器操作数的数量
    }

    if (!PySequence_Check(v) || PySequence_Size(v) != ihigh-ilow) {  // 如果v不是一个序列对象或者其大小不等于ihigh-ilow
        PyErr_SetString(PyExc_ValueError,
                "Wrong size to assign to iterator slice");  // 抛出值错误异常，指明分配给迭代器切片的大小不正确
        return -1;  // 返回-1表示操作失败
    }

    for (i = ilow; i < ihigh ; ++i) {  // 循环遍历ilow到ihigh的范围
        PyObject *item = PySequence_GetItem(v, i-ilow);  // 获取v中索引为i-ilow的元素
        if (item == NULL) {  // 如果获取元素失败
            return -1;  // 返回-1表示操作失败
        }
        if (npyiter_seq_ass_item(self, i, item) < 0) {  // 调用npyiter_seq_ass_item函数，将索引为i的迭代器位置赋值为item
            Py_DECREF(item);  // 减少item的引用计数
            return -1;  // 返回-1表示操作失败
        }
        Py_DECREF(item);  // 减少item的引用计数
    }

    return 0;  // 返回0表示操作成功
}

static PyObject *
npyiter_subscript(NewNpyArrayIterObject *self, PyObject *op)
{
    if (self->iter == NULL || self->finished) {  // 如果迭代器指针为空或者迭代器已经完成
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");  // 抛出值错误异常，指明迭代器已经超出末尾
        return NULL;  // 返回NULL表示操作失败
    }

    if (NpyIter_HasDelayedBufAlloc(self->iter)) {  // 如果迭代器构造过程使用了延迟缓冲区分配
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");  // 抛出值错误异常，指明迭代器构造过程中使用了延迟缓冲区分配且尚未重置
        return NULL;  // 返回NULL表示操作失败
    }

    if (PyLong_Check(op) ||  // 如果op是一个长整型对象或者
                    (PyIndex_Check(op) && !PySequence_Check(op))) {  // op是一个索引对象但不是序列对象
        npy_intp i = PyArray_PyIntAsIntp(op);  // 将op转换为npy_intp类型的整数
        if (error_converting(i)) {  // 如果转换出错
            return NULL;  // 返回NULL表示操作失败
        }
        return npyiter_seq_item(self, i);  // 返回调用npyiter_seq_item函数得到的结果
    }
    else if (PySlice_Check(op)) {  // 如果op是一个切片对象
        Py_ssize_t istart = 0, iend = 0, istep = 0, islicelength;  // 定义几个变量用于存储切片的起始、结束、步长和切片长度
        if (PySlice_GetIndicesEx(op, NpyIter_GetNOp(self->iter),
                                 &istart, &iend, &istep, &islicelength) < 0) {  // 调用PySlice_GetIndicesEx函数获取切片的索引范围
            return NULL;  // 返回NULL表示操作失败
        }
        if (istep != 1) {  // 如果切片的步长不为1
            PyErr_SetString(PyExc_ValueError,
                    "Iterator slicing only supports a step of 1");  // 抛出值错误异常，指明迭代器切片只支持步长为1
            return NULL;  // 返回NULL表示操作失败
        }
        return npyiter_seq_slice(self, istart, iend);  // 返回调用npyiter_seq_slice函数得到的结果
    }

    PyErr_SetString(PyExc_TypeError,
            "invalid index type for iterator indexing");  // 抛出类型错误异常，指明迭代器索引类型无效
    return NULL;  // 返回NULL表示操作失败
}

static int
npyiter_ass_subscript(NewNpyArrayIterObject *self, PyObject *op,
                        PyObject *value)
{
    # 如果值为 NULL，则设置类型错误异常，指示无法删除迭代器元素
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot delete iterator elements");
        return -1;
    }

    # 如果迭代器为 NULL 或已经完成迭代，则设置值错误异常，指示迭代器已经超出末尾
    if (self->iter == NULL || self->finished) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator is past the end");
        return -1;
    }

    # 如果迭代器使用了延迟的缓冲区分配，并且还没有进行重置，则设置值错误异常
    if (NpyIter_HasDelayedBufAlloc(self->iter)) {
        PyErr_SetString(PyExc_ValueError,
                "Iterator construction used delayed buffer allocation, "
                "and no reset has been done yet");
        return -1;
    }

    # 如果操作对象是 PyLong 或者 PyIndex 但不是 PySequence，则尝试将操作对象转换为 npy_intp 类型，并进行序列赋值操作
    if (PyLong_Check(op) ||
                    (PyIndex_Check(op) && !PySequence_Check(op))) {
        npy_intp i = PyArray_PyIntAsIntp(op);
        if (error_converting(i)) {
            return -1;
        }
        return npyiter_seq_ass_item(self, i, value);
    }
    # 如果操作对象是 PySlice，则获取切片的起始、结束、步长和切片长度，并进行切片赋值操作
    else if (PySlice_Check(op)) {
        Py_ssize_t istart = 0, iend = 0, istep = 0, islicelength = 0;
        if (PySlice_GetIndicesEx(op, NpyIter_GetNOp(self->iter),
                                 &istart, &iend, &istep, &islicelength) < 0) {
            return -1;
        }
        # 如果切片的步长不为 1，则设置值错误异常，因为迭代器切片赋值仅支持步长为 1
        if (istep != 1) {
            PyErr_SetString(PyExc_ValueError,
                    "Iterator slice assignment only supports a step of 1");
            return -1;
        }
        return npyiter_seq_ass_slice(self, istart, iend, value);
    }

    # 如果操作对象不是合法的索引类型，则设置类型错误异常
    PyErr_SetString(PyExc_TypeError,
            "invalid index type for iterator indexing");
    return -1;
    {"reset",
        (PyCFunction)npyiter_reset,
        METH_NOARGS, NULL},
    # 定义名为 "reset" 的方法，使用 npyiter_reset 函数实现，不接受参数，无额外信息
    {"copy",
        (PyCFunction)npyiter_copy,
        METH_NOARGS, NULL},
    # 定义名为 "copy" 的方法，使用 npyiter_copy 函数实现，不接受参数，无额外信息
    {"__copy__",
        (PyCFunction)npyiter_copy,
        METH_NOARGS, NULL},
    # 定义名为 "__copy__" 的方法，使用 npyiter_copy 函数实现，不接受参数，无额外信息
    {"iternext",
        (PyCFunction)npyiter_iternext,
        METH_NOARGS, NULL},
    # 定义名为 "iternext" 的方法，使用 npyiter_iternext 函数实现，不接受参数，无额外信息
    {"remove_axis",
        (PyCFunction)npyiter_remove_axis,
        METH_VARARGS, NULL},
    # 定义名为 "remove_axis" 的方法，使用 npyiter_remove_axis 函数实现，接受可变参数，无额外信息
    {"remove_multi_index",
        (PyCFunction)npyiter_remove_multi_index,
        METH_NOARGS, NULL},
    # 定义名为 "remove_multi_index" 的方法，使用 npyiter_remove_multi_index 函数实现，不接受参数，无额外信息
    {"enable_external_loop",
        (PyCFunction)npyiter_enable_external_loop,
        METH_NOARGS, NULL},
    # 定义名为 "enable_external_loop" 的方法，使用 npyiter_enable_external_loop 函数实现，不接受参数，无额外信息
    {"debug_print",
        (PyCFunction)npyiter_debug_print,
        METH_NOARGS, NULL},
    # 定义名为 "debug_print" 的方法，使用 npyiter_debug_print 函数实现，不接受参数，无额外信息
    {"__enter__", (PyCFunction)npyiter_enter,
         METH_NOARGS,  NULL},
    # 定义名为 "__enter__" 的方法，使用 npyiter_enter 函数实现，不接受参数，无额外信息
    {"__exit__",  (PyCFunction)npyiter_exit,
         METH_VARARGS, NULL},
    # 定义名为 "__exit__" 的方法，使用 npyiter_exit 函数实现，接受可变参数，无额外信息
    {"close",  (PyCFunction)npyiter_close,
         METH_NOARGS, NULL},
    # 定义名为 "close" 的方法，使用 npyiter_close 函数实现，不接受参数，无额外信息
    {NULL, NULL, 0, NULL},
    # 结束方法定义的标志
    {"has_multi_index",
        // 用于获取是否具有多重索引的属性，对应的 getter 函数为 npyiter_has_multi_index_get
        (getter)npyiter_has_multi_index_get,
        // 没有 setter 函数，因此为 NULL
        NULL, 
        // 删除时的函数，这里为 NULL
        NULL, 
        // 文档字符串，这里为 NULL
        NULL},
    {"has_index",
        // 用于获取是否具有索引的属性，对应的 getter 函数为 npyiter_has_index_get
        (getter)npyiter_has_index_get,
        NULL, NULL, NULL},
    {"dtypes",
        // 用于获取迭代器数据类型的属性，对应的 getter 函数为 npyiter_dtypes_get
        (getter)npyiter_dtypes_get,
        NULL, NULL, NULL},
    {"ndim",
        // 用于获取迭代器维度的属性，对应的 getter 函数为 npyiter_ndim_get
        (getter)npyiter_ndim_get,
        NULL, NULL, NULL},
    {"nop",
        // 用于获取 nop 的属性，对应的 getter 函数为 npyiter_nop_get
        (getter)npyiter_nop_get,
        NULL, NULL, NULL},
    {"itersize",
        // 用于获取迭代器尺寸的属性，对应的 getter 函数为 npyiter_itersize_get
        (getter)npyiter_itersize_get,
        NULL, NULL, NULL},
    {"finished",
        // 用于获取迭代器是否完成的属性，对应的 getter 函数为 npyiter_finished_get
        (getter)npyiter_finished_get,
        NULL, NULL, NULL},

    {NULL, NULL, NULL, NULL, NULL}
# 定义结构体 PySequenceMethods，并初始化其成员
NPY_NO_EXPORT PySequenceMethods npyiter_as_sequence = {
    (lenfunc)npyiter_seq_length,            /*sq_length*/  # 设置序列长度函数指针
    (binaryfunc)NULL,                       /*sq_concat*/  # 没有定义序列连接操作，置为 NULL
    (ssizeargfunc)NULL,                     /*sq_repeat*/  # 没有定义序列重复操作，置为 NULL
    (ssizeargfunc)npyiter_seq_item,         /*sq_item*/    # 设置获取序列元素的函数指针
    (ssizessizeargfunc)NULL,                /*sq_slice*/   # 没有定义序列切片操作，置为 NULL
    (ssizeobjargproc)npyiter_seq_ass_item,  /*sq_ass_item*/  # 设置设置序列元素的函数指针
    (ssizessizeobjargproc)NULL,             /*sq_ass_slice*/  # 没有定义序列切片设置操作，置为 NULL
    (objobjproc)NULL,                       /*sq_contains */  # 没有定义序列包含操作，置为 NULL
    (binaryfunc)NULL,                       /*sq_inplace_concat */  # 没有定义序列原地连接操作，置为 NULL
    (ssizeargfunc)NULL,                     /*sq_inplace_repeat */  # 没有定义序列原地重复操作，置为 NULL
};

# 定义结构体 PyMappingMethods，并初始化其成员
NPY_NO_EXPORT PyMappingMethods npyiter_as_mapping = {
    (lenfunc)npyiter_seq_length,          /*mp_length*/  # 设置映射长度函数指针
    (binaryfunc)npyiter_subscript,        /*mp_subscript*/  # 设置映射获取元素的函数指针
    (objobjargproc)npyiter_ass_subscript, /*mp_ass_subscript*/  # 设置映射设置元素的函数指针
};

# 定义结构体 PyTypeObject，并初始化其成员
NPY_NO_EXPORT PyTypeObject NpyIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)  # 初始化基类对象
    .tp_name = "numpy.nditer",  # 设置类型对象的名称
    .tp_basicsize = sizeof(NewNpyArrayIterObject),  # 设置类型对象的基本大小
    .tp_dealloc = (destructor)npyiter_dealloc,  # 设置析构函数指针
    .tp_as_sequence = &npyiter_as_sequence,  # 设置序列操作方法集合
    .tp_as_mapping = &npyiter_as_mapping,  # 设置映射操作方法集合
    .tp_flags = Py_TPFLAGS_DEFAULT,  # 设置类型对象的标志
    .tp_iternext = (iternextfunc)npyiter_next,  # 设置迭代器的下一个函数指针
    .tp_methods = npyiter_methods,  # 设置类型对象的方法集合
    .tp_members = npyiter_members,  # 设置类型对象的成员变量集合
    .tp_getset = npyiter_getsets,  # 设置类型对象的属性访问器集合
    .tp_init = (initproc)npyiter_init,  # 设置初始化函数指针
    .tp_new = npyiter_new,  # 设置新建对象函数指针
};
```