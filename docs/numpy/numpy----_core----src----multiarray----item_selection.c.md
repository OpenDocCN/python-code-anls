# `.\numpy\numpy\_core\src\multiarray\item_selection.c`

```py
/*
 * 定义 NPY_NO_DEPRECATED_API 以及 _MULTIARRAYMODULE 宏
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 定义 PY_SSIZE_T_CLEAN 宏，并包含必要的头文件
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"
#include "numpy/npy_cpu.h"

#include "npy_config.h"

/*
 * 包含其他头文件，用于数组操作和类型处理
 */
#include "npy_static_data.h"
#include "common.h"
#include "dtype_transfer.h"
#include "dtypemeta.h"
#include "arrayobject.h"
#include "ctors.h"
#include "lowlevel_strided_loops.h"
#include "array_assign.h"
#include "refcount.h"

#include "npy_sort.h"
#include "npy_partition.h"
#include "npy_binsearch.h"
#include "alloc.h"
#include "arraytypes.h"
#include "array_coercion.h"
#include "simd/simd.h"

/*
 * 定义静态内联函数 npy_fasttake_impl
 * 实现快速取值操作，支持多线程处理
 */
static NPY_GCC_OPT_3 inline int
npy_fasttake_impl(
        char *dest, char *src, const npy_intp *indices,
        npy_intp n, npy_intp m, npy_intp max_item,
        npy_intp nelem, npy_intp chunk,
        NPY_CLIPMODE clipmode, npy_intp itemsize, int needs_refcounting,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype, int axis)
{
    NPY_BEGIN_THREADS_DEF;  /* 定义多线程开始 */

    NPY_cast_info cast_info;  /* 定义类型转换信息结构体 */
    NPY_ARRAYMETHOD_FLAGS flags;  /* 定义数组方法标志 */

    NPY_cast_info_init(&cast_info);  /* 初始化类型转换信息 */

    if (!needs_refcounting) {
        /* 如果不需要引用计数，直接使用 memcpy 进行简单的拷贝 */
        NPY_BEGIN_THREADS;  /* 开始多线程处理 */
    }
    else {
        /* 如果需要引用计数 */
        if (PyArray_GetDTypeTransferFunction(
                1, itemsize, itemsize, src_dtype, dst_dtype, 0,
                &cast_info, &flags) < 0) {
            return -1;  /* 获取数据类型转换函数失败，返回错误 */
        }
        if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
            NPY_BEGIN_THREADS;  /* 开始多线程处理 */
        }
    }

    /* 多线程结束 */
    NPY_END_THREADS;

    NPY_cast_info_xfree(&cast_info);  /* 释放类型转换信息结构体内存 */
    return 0;  /* 返回成功 */
    
fail:
    /* 失败时，已经确保多线程结束 */
    NPY_cast_info_xfree(&cast_info);  /* 释放类型转换信息结构体内存 */
    return -1;  /* 返回失败 */
}


/*
 * 辅助函数，实例化 npy_fasttake_impl 在不同分支中以优化每个特定的 itemsize
 */
static NPY_GCC_OPT_3 int
npy_fasttake(
        char *dest, char *src, const npy_intp *indices,
        npy_intp n, npy_intp m, npy_intp max_item,
        npy_intp nelem, npy_intp chunk,
        NPY_CLIPMODE clipmode, npy_intp itemsize, int needs_refcounting,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype, int axis)
{
    # 如果不需要引用计数（refcounting），则执行以下逻辑
    if (!needs_refcounting) {
        # 如果 chunk 大小为 1，调用快速取值实现函数
        if (chunk == 1) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        # 如果 chunk 大小为 2，调用快速取值实现函数
        if (chunk == 2) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        # 如果 chunk 大小为 4，调用快速取值实现函数
        if (chunk == 4) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        # 如果 chunk 大小为 8，调用快速取值实现函数
        if (chunk == 8) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        # 如果 chunk 大小为 16，调用快速取值实现函数
        if (chunk == 16) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
        # 如果 chunk 大小为 32，调用快速取值实现函数
        if (chunk == 32) {
            return npy_fasttake_impl(
                    dest, src, indices, n, m, max_item, nelem, chunk,
                    clipmode, itemsize, needs_refcounting, src_dtype,
                    dst_dtype, axis);
        }
    }

    # 对于其他 chunk 大小或者需要引用计数的情况，调用快速取值实现函数
    return npy_fasttake_impl(
            dest, src, indices, n, m, max_item, nelem, chunk,
            clipmode, itemsize, needs_refcounting, src_dtype,
            dst_dtype, axis);
    }

    /*NUMPY_API
     * Take
     */
    /* 定义 PyArray_TakeFrom 函数，接受一个 NumPy 数组对象 self0，一个索引数组对象 indices0，
     * 一个轴 axis，一个输出数组对象 out，以及一个剪切模式 clipmode */
    NPY_NO_EXPORT PyObject *
    PyArray_TakeFrom(PyArrayObject *self0, PyObject *indices0, int axis,
                     PyArrayObject *out, NPY_CLIPMODE clipmode)
    {
        PyArray_Descr *dtype;  /* PyArray_TakeFrom 函数中用到的数组描述符 */
        PyArrayObject *obj = NULL, *self, *indices;  /* 定义 PyArrayObject 类型的指针变量 obj, self, indices */
        npy_intp nd, i, n, m, max_item, chunk, itemsize, nelem;  /* 定义 numpy 的整型数据类型变量 */
        npy_intp shape[NPY_MAXDIMS];  /* 定义数组形状的数组 */

        npy_bool needs_refcounting;  /* 是否需要引用计数 */

        indices = NULL;  /* 将 indices 初始化为 NULL */
        /* 将 self0 转换为 PyArrayObject 类型，并检查轴，返回只读的 C 数组类型的 self，若失败则返回 NULL */
        self = (PyArrayObject *)PyArray_CheckAxis(self0, &axis,
                                        NPY_ARRAY_CARRAY_RO);
        if (self == NULL) {
            return NULL;  /* 如果 self 为 NULL，则直接返回 NULL */
        }

        /* 将 indices0 转换为 PyArrayObject 类型，要求数据类型为 NPY_INTP，若失败则跳转到 fail 标签处 */
        indices = (PyArrayObject *)PyArray_FromAny(indices0,
                    PyArray_DescrFromType(NPY_INTP),
                    0, 0,
                    NPY_ARRAY_SAME_KIND_CASTING | NPY_ARRAY_DEFAULT,
                    NULL);
        if (indices == NULL) {
            goto fail;  /* 如果 indices 为 NULL，则跳转到 fail 标签处 */
        }

        n = m = chunk = 1;  /* 初始化 n, m, chunk 为 1 */
        nd = PyArray_NDIM(self) + PyArray_NDIM(indices) - 1;  /* 计算结果数组的维度 */
        /* 遍历计算结果数组的形状 */
        for (i = 0; i < nd; i++) {
            if (i < axis) {
                shape[i] = PyArray_DIMS(self)[i];  /* 如果 i 小于 axis，则取 self 对应维度的大小 */
                n *= shape[i];  /* 计算 n */
            }
            else {
                if (i < axis+PyArray_NDIM(indices)) {
                    shape[i] = PyArray_DIMS(indices)[i-axis];  /* 计算 indices 对应维度的大小 */
                    m *= shape[i];  /* 计算 m */
                }
                else {
                    shape[i] = PyArray_DIMS(self)[i-PyArray_NDIM(indices)+1];  /* 计算剩余维度的大小 */
                    chunk *= shape[i];  /* 计算 chunk */
                }
            }
        }
        /* 如果没有指定输出数组 out */
        if (!out) {
            dtype = PyArray_DESCR(self);  /* 获取 self 的数据类型描述符 */
            Py_INCREF(dtype);  /* 增加数据类型描述符的引用计数 */
            /* 使用给定的描述符创建新的数组对象 obj */
            obj = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                                                        dtype,
                                                        nd, shape,
                                                        NULL, NULL, 0,
                                                        (PyObject *)self);
            if (obj == NULL) {
                goto fail;  /* 如果创建 obj 失败，则跳转到 fail 标签处 */
            }

        }
        else {
            int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY;  /* 设置数组的标志 */

            /* 检查输出数组 out 的维度是否与结果数组的维度相同，若不同则抛出 ValueError 异常 */
            if ((PyArray_NDIM(out) != nd) ||
                !PyArray_CompareLists(PyArray_DIMS(out), shape, nd)) {
                PyErr_SetString(PyExc_ValueError,
                            "output array does not match result of ndarray.take");
                goto fail;  /* 如果维度不匹配，则跳转到 fail 标签处 */
            }

            /* 如果 out 和 self 有重叠的部分，则设置标志为 NPY_ARRAY_ENSURECOPY */
            if (arrays_overlap(out, self)) {
                flags |= NPY_ARRAY_ENSURECOPY;
            }

            /* 如果剪切模式为 NPY_RAISE，则需要确保获取副本 */
            if (clipmode == NPY_RAISE) {
                /*
                 * 我们需要确保获取一个副本
                 * 这样在调用错误之前不会改变输入数组
                 */
                flags |= NPY_ARRAY_ENSURECOPY;
            }
            dtype = PyArray_DESCR(self);  /* 获取 self 的数据类型描述符 */
            Py_INCREF(dtype);  /* 增加数据类型描述符的引用计数 */
            /* 使用给定的数组 out、描述符 dtype 和标志 flags 创建新的数组对象 obj */
            obj = (PyArrayObject *)PyArray_FromArray(out, dtype, flags);
            if (obj == NULL) {
                goto fail;  /* 如果创建 obj 失败，则跳转到 fail 标签处 */
            }
        }

        max_item = PyArray_DIMS(self)[axis];  /* 获取 self 在轴 axis 上的大小 */
        nelem = chunk;  /* 设置 nelem 为 chunk */
        itemsize = PyArray_ITEMSIZE(obj);  /* 获取 obj 的单个元素大小 */
        chunk = chunk * itemsize;  /* 计算 chunk 的大小 */
        char *src = PyArray_DATA(self);  /* 获取 self 的数据指针 */
        char *dest = PyArray_DATA(obj);  /* 获取 obj 的数据指针 */
    // 获取源数组的描述符
    PyArray_Descr *src_descr = PyArray_DESCR(self);
    // 获取目标数组的描述符
    PyArray_Descr *dst_descr = PyArray_DESCR(obj);
    // 检查是否需要引用计数，基于源数组的数据类型描述符
    needs_refcounting = PyDataType_REFCHK(PyArray_DESCR(self));
    // 将索引数组的数据转换为 npy_intp 类型指针
    npy_intp *indices_data = (npy_intp *)PyArray_DATA(indices);

    // 如果 max_item 为 0 且目标数组不为空，则抛出索引错误
    if ((max_item == 0) && (PyArray_SIZE(obj) != 0)) {
        /* Index error, since that is the usual error for raise mode */
        PyErr_SetString(PyExc_IndexError,
                    "cannot do a non-empty take from an empty axes.");
        // 跳转到失败处理部分
        goto fail;
    }

    // 调用 npy_fasttake 函数执行快速取值操作
    if (npy_fasttake(
            dest, src, indices_data, n, m, max_item, nelem, chunk,
            clipmode, itemsize, needs_refcounting, src_descr, dst_descr,
            axis) < 0) {
        // 如果操作失败，跳转到失败处理部分
        goto fail;
    }

    // 如果指定了输出数组且输出数组不等于原始数组
    if (out != NULL && out != obj) {
        // 尝试解析写回（writeback）操作，如果失败则跳转到失败处理部分
        if (PyArray_ResolveWritebackIfCopy(obj) < 0) {
            goto fail;
        }
        // 释放原始数组的引用
        Py_DECREF(obj);
        // 增加输出数组的引用
        Py_INCREF(out);
        // 更新 obj 指向输出数组
        obj = out;
    }
    // 释放索引数组的引用
    Py_XDECREF(indices);
    // 释放 self 指向的数组的引用
    Py_XDECREF(self);
    // 返回 obj 对象（PyObject 类型的指针）
    return (PyObject *)obj;

 fail:
    // 放弃写回操作（如果是复制的情况）
    PyArray_DiscardWritebackIfCopy(obj);
    // 释放 obj 对象的引用
    Py_XDECREF(obj);
    // 释放索引数组的引用
    Py_XDECREF(indices);
    // 释放 self 指向的数组的引用
    Py_XDECREF(self);
    // 返回 NULL 表示操作失败
    return NULL;
    }

    /*NUMPY_API
     * Put values into an array
     */
    /* 将值放入数组中的函数定义，是 NumPy C API 中的一部分 */

    NPY_NO_EXPORT PyObject *
    PyArray_PutTo(PyArrayObject *self, PyObject* values0, PyObject *indices0,
                  NPY_CLIPMODE clipmode)
    {
        PyArrayObject  *indices, *values;
        npy_intp i, itemsize, ni, max_item, nv, tmp;
        char *src, *dest;
        int copied = 0;
        int overlap = 0;

        NPY_BEGIN_THREADS_DEF;  /* 定义 NumPy 线程支持 */

        NPY_cast_info cast_info;  /* 定义类型转换信息结构体 */
        NPY_ARRAYMETHOD_FLAGS flags;  /* 定义数组方法的标志 */

        NPY_cast_info_init(&cast_info);  /* 初始化类型转换信息结构体 */

        indices = NULL;
        values = NULL;
        if (!PyArray_Check(self)) {
            PyErr_SetString(PyExc_TypeError,
                            "put: first argument must be an array");
            return NULL;
        }
        /* 检查第一个参数是否为数组，若不是则设置类型错误并返回空指针 */

        if (PyArray_FailUnlessWriteable(self, "put: output array") < 0) {
            return NULL;
        }
        /* 确保输出数组可写，若不可写则返回空指针 */

        indices = (PyArrayObject *)PyArray_ContiguousFromAny(indices0,
                                                             NPY_INTP, 0, 0);
        /* 将 indices0 转换为连续的整型数组对象 */
        if (indices == NULL) {
            goto fail;
        }
        /* 若转换失败则跳转到 fail 标签处 */

        ni = PyArray_SIZE(indices);  /* 获取 indices 数组的大小 */
        if ((ni > 0) && (PyArray_Size((PyObject *)self) == 0)) {
            PyErr_SetString(PyExc_IndexError, 
                            "cannot replace elements of an empty array");
            goto fail;
        }
        /* 若 indices 非空且 self 是空数组，则设置索引错误并跳转到 fail 标签处 */

        Py_INCREF(PyArray_DESCR(self));  /* 增加 self 数组的类型描述的引用计数 */

        values = (PyArrayObject *)PyArray_FromAny(values0, PyArray_DESCR(self), 0, 0,
                                  NPY_ARRAY_DEFAULT | NPY_ARRAY_FORCECAST, NULL);
        /* 将 values0 转换为具有指定描述符的数组对象 */
        if (values == NULL) {
            goto fail;
        }
        /* 若转换失败则跳转到 fail 标签处 */

        nv = PyArray_SIZE(values);  /* 获取 values 数组的大小 */
        if (nv <= 0) {
            goto finish;
        }
        /* 若 values 为空则跳转到 finish 标签处 */

        overlap = arrays_overlap(self, values) || arrays_overlap(self, indices);
        /* 检查 self 与 values 或 indices 是否有重叠 */

        if (overlap || !PyArray_ISCONTIGUOUS(self)) {
            /* 若有重叠或 self 不是连续的数组 */
            PyArrayObject *obj;
            int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY |
                        NPY_ARRAY_ENSURECOPY;

            Py_INCREF(PyArray_DESCR(self));  /* 增加 self 数组的类型描述的引用计数 */

            obj = (PyArrayObject *)PyArray_FromArray(self,
                                                     PyArray_DESCR(self), flags);
            /* 根据给定数组创建新的数组对象 */
            copied = 1;
            assert(self != obj);  /* 断言 self 与 obj 不相同 */
            self = obj;
        }

        max_item = PyArray_SIZE(self);  /* 获取 self 数组的大小 */
        dest = PyArray_DATA(self);  /* 获取 self 数组的数据指针 */
        itemsize = PyArray_ITEMSIZE(self);  /* 获取 self 数组的单个元素大小 */

        int has_references = PyDataType_REFCHK(PyArray_DESCR(self));
        /* 检查 self 数组的数据类型是否包含引用 */

        if (!has_references) {
            /* 若数据类型不包含引用，则直接使用 memcpy 进行简单复制 */
            NPY_BEGIN_THREADS_THRESHOLDED(ni);  /* 根据线程阈值启动线程 */
        }
        else {
            PyArray_Descr *dtype = PyArray_DESCR(self);
            if (PyArray_GetDTypeTransferFunction(
                    PyArray_ISALIGNED(self), itemsize, itemsize, dtype, dtype, 0,
                    &cast_info, &flags) < 0) {
                goto fail;
            }
            /* 获取数据类型之间的传输函数信息 */

            if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
                NPY_BEGIN_THREADS_THRESHOLDED(ni);  /* 根据线程阈值启动线程 */
            }
        }
        ```
    // 如果有引用存在时执行以下代码块
    if (has_references) {
        // 定义常量 one 为 npy_intp 类型的 1
        const npy_intp one = 1;
        // 定义步长数组 strides，包含两个元素，每个元素都是 itemsize
        const npy_intp strides[2] = {itemsize, itemsize};

        // 根据 clipmode 的不同情况执行不同的处理
        switch(clipmode) {
        // 如果 clipmode 是 NPY_RAISE
        case NPY_RAISE:
            // 遍历索引数组中的元素
            for (i = 0; i < ni; i++) {
                // 计算 src 指针指向的位置
                src = PyArray_BYTES(values) + itemsize*(i % nv);
                // 获取索引数组中的值作为 tmp
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                // 检查并调整索引 tmp，如果失败则跳转至 fail 标签处
                if (check_and_adjust_index(&tmp, max_item, 0, _save) < 0) {
                    goto fail;
                }
                // 定义包含两个指针的 data 数组
                char *data[2] = {src, dest + tmp*itemsize};
                // 调用 cast_info.func 进行类型转换，处理 data 数据
                if (cast_info.func(
                        &cast_info.context, data, &one, strides,
                        cast_info.auxdata) < 0) {
                    NPY_END_THREADS;
                    goto fail;
                }
            }
            break;
        
        // 如果 clipmode 是 NPY_WRAP
        case NPY_WRAP:
            // 遍历索引数组中的元素
            for (i = 0; i < ni; i++) {
                // 计算 src 指针指向的位置
                src = PyArray_BYTES(values) + itemsize * (i % nv);
                // 获取索引数组中的值作为 tmp
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                // 如果 tmp 小于 0，则循环增加 tmp 直到大于等于 0
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                // 如果 tmp 大于等于 max_item，则循环减小 tmp 直到小于 max_item
                else if (tmp >= max_item) {
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                // 定义包含两个指针的 data 数组
                char *data[2] = {src, dest + tmp*itemsize};
                // 调用 cast_info.func 进行类型转换，处理 data 数据
                if (cast_info.func(
                        &cast_info.context, data, &one, strides,
                        cast_info.auxdata) < 0) {
                    NPY_END_THREADS;
                    goto fail;
                }
            }
            break;
        
        // 如果 clipmode 是 NPY_CLIP
        case NPY_CLIP:
            // 遍历索引数组中的元素
            for (i = 0; i < ni; i++) {
                // 计算 src 指针指向的位置
                src = PyArray_BYTES(values) + itemsize * (i % nv);
                // 获取索引数组中的值作为 tmp
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                // 如果 tmp 小于 0，则将 tmp 设为 0
                if (tmp < 0) {
                    tmp = 0;
                }
                // 如果 tmp 大于等于 max_item，则将 tmp 设为 max_item - 1
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                // 定义包含两个指针的 data 数组
                char *data[2] = {src, dest + tmp*itemsize};
                // 调用 cast_info.func 进行类型转换，处理 data 数据
                if (cast_info.func(
                        &cast_info.context, data, &one, strides,
                        cast_info.auxdata) < 0) {
                    NPY_END_THREADS;
                    goto fail;
                }
            }
            break;
        }
    }
    else {
        switch(clipmode) {
        case NPY_RAISE:
            // 如果 clipmode 是 NPY_RAISE，进行严格模式处理
            for (i = 0; i < ni; i++) {
                // 计算源数据的位置
                src = PyArray_BYTES(values) + itemsize * (i % nv);
                // 获取索引值并进行调整
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (check_and_adjust_index(&tmp, max_item, 0, _save) < 0) {
                    // 若索引越界则跳转至失败处理标签
                    goto fail;
                }
                // 将数据从 src 复制到 dest 中的指定位置
                memmove(dest + tmp * itemsize, src, itemsize);
            }
            break;
        case NPY_WRAP:
            // 如果 clipmode 是 NPY_WRAP，进行环绕模式处理
            for (i = 0; i < ni; i++) {
                // 计算源数据的位置
                src = PyArray_BYTES(values) + itemsize * (i % nv);
                // 获取索引值
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    // 处理负索引，使其在范围内
                    while (tmp < 0) {
                        tmp += max_item;
                    }
                }
                else if (tmp >= max_item) {
                    // 处理超出最大索引，使其在范围内
                    while (tmp >= max_item) {
                        tmp -= max_item;
                    }
                }
                // 将数据从 src 复制到 dest 中的指定位置
                memmove(dest + tmp * itemsize, src, itemsize);
            }
            break;
        case NPY_CLIP:
            // 如果 clipmode 是 NPY_CLIP，进行截断模式处理
            for (i = 0; i < ni; i++) {
                // 计算源数据的位置
                src = PyArray_BYTES(values) + itemsize * (i % nv);
                // 获取索引值并将其限制在合法范围内
                tmp = ((npy_intp *)(PyArray_DATA(indices)))[i];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= max_item) {
                    tmp = max_item - 1;
                }
                // 将数据从 src 复制到 dest 中的指定位置
                memmove(dest + tmp * itemsize, src, itemsize);
            }
            break;
        }
    }
    // 结束多线程操作
    NPY_END_THREADS;

 finish:
    // 释放类型转换信息资源
    NPY_cast_info_xfree(&cast_info);

    // 释放引用的对象
    Py_XDECREF(values);
    Py_XDECREF(indices);
    if (copied) {
        // 若有复制操作，解析写回（writeback）并释放数组对象
        PyArray_ResolveWritebackIfCopy(self);
        Py_DECREF(self);
    }
    // 返回 None
    Py_RETURN_NONE;

 fail:
    // 失败处理，释放类型转换信息资源
    NPY_cast_info_xfree(&cast_info);

    // 释放引用的对象
    Py_XDECREF(indices);
    Py_XDECREF(values);
    if (copied) {
        // 若有复制操作，丢弃写回（writeback）并释放数组对象
        PyArray_DiscardWritebackIfCopy(self);
        Py_XDECREF(self);
    }
    // 返回 NULL 表示失败
    return NULL;
/*NUMPY_API
 * Put values into an array according to a mask.
 */
NPY_NO_EXPORT PyObject *
PyArray_PutMask(PyArrayObject *self, PyObject* values0, PyObject* mask0)
{
    PyArrayObject *mask, *values;
    PyArray_Descr *dtype;
    npy_intp itemsize, ni, nv;
    char *src, *dest;
    npy_bool *mask_data;
    int copied = 0;
    int overlap = 0;
    NPY_BEGIN_THREADS_DEF;

    mask = NULL;  // 初始化 mask 为 NULL
    values = NULL;  // 初始化 values 为 NULL
    // 检查 self 是否为一个数组，如果不是，设置错误信息并返回 NULL
    if (!PyArray_Check(self)) {
        PyErr_SetString(PyExc_TypeError,
                        "putmask: first argument must "
                        "be an array");
        return NULL;
    }

    // 检查 self 是否可写，如果不可写，返回 NULL
    if (PyArray_FailUnlessWriteable(self, "putmask: output array") < 0) {
        return NULL;
    }

    // 将 mask0 转换为 NPY_BOOL 类型的 PyArrayObject，要求 C 连续存储，并强制转换
    mask = (PyArrayObject *)PyArray_FROM_OTF(mask0, NPY_BOOL,
                                NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST);
    // 如果转换失败，跳转到 fail 标签处理错误
    if (mask == NULL) {
        goto fail;
    }
    // 获取 mask 的元素个数 ni，与 self 的元素个数进行比较，如果不相等，设置错误信息并跳转到 fail 标签
    ni = PyArray_SIZE(mask);
    if (ni != PyArray_SIZE(self)) {
        PyErr_SetString(PyExc_ValueError,
                        "putmask: mask and data must be "
                        "the same size");
        goto fail;
    }
    // 获取 mask 数据的指针
    mask_data = PyArray_DATA(mask);
    // 获取 self 的数据类型描述符，并增加其引用计数
    dtype = PyArray_DESCR(self);
    Py_INCREF(dtype);
    // 将 values0 转换为 PyArrayObject 类型的数组对象，使用给定的 dtype
    // 如果转换失败，则跳转到错误处理标签 fail
    values = (PyArrayObject *)PyArray_FromAny(values0, dtype,
                                    0, 0, NPY_ARRAY_CARRAY, NULL);
    if (values == NULL) {
        goto fail;
    }
    
    // 获取数组 values 的元素个数 nv，如果数组为空，则 nv 为零
    nv = PyArray_SIZE(values); /* 如果数组为空则为零 */
    if (nv <= 0) {
        // 如果 nv 小于等于 0，释放 values 和 mask 的引用，并返回 None
        Py_XDECREF(values);
        Py_XDECREF(mask);
        Py_RETURN_NONE;
    }
    
    // 获取数组 values 的数据指针 src
    src = PyArray_DATA(values);

    // 检查 self 和 values 或 mask 是否有重叠，或者 self 是否非连续存储
    overlap = arrays_overlap(self, values) || arrays_overlap(self, mask);
    if (overlap || !PyArray_ISCONTIGUOUS(self)) {
        // 如果有重叠或者 self 不是连续存储，则创建一个新的数组对象 obj
        // 使用 flags 标志指定数组属性，保留原始数据或者创建数据的拷贝
        int flags = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY;
        PyArrayObject *obj;

        if (overlap) {
            flags |= NPY_ARRAY_ENSURECOPY;
        }

        // 获取 self 的数据类型 dtype，并增加其引用计数
        dtype = PyArray_DESCR(self);
        Py_INCREF(dtype);
        
        // 根据 self 的数据类型和 flags 创建一个新的数组对象 obj
        obj = (PyArrayObject *)PyArray_FromArray(self, dtype, flags);
        
        // 如果 obj 不等于 self，则表示进行了数据拷贝
        if (obj != self) {
            copied = 1;
        }
        
        // 将 self 指向新创建的数组对象 obj
        self = obj;
    }

    // 获取 self 中每个元素的字节大小 itemsize 和数据指针 dest
    itemsize = PyArray_ITEMSIZE(self);
    dest = PyArray_DATA(self);

    // 检查 self 的数据类型是否需要引用计数检查
    if (PyDataType_REFCHK(PyArray_DESCR(self))) {
        NPY_cast_info cast_info;
        NPY_ARRAYMETHOD_FLAGS flags;
        const npy_intp one = 1;
        const npy_intp strides[2] = {itemsize, itemsize};

        // 初始化类型转换信息 cast_info
        NPY_cast_info_init(&cast_info);
        
        // 获取数据类型之间的转换函数，如果失败则跳转到错误处理标签 fail
        if (PyArray_GetDTypeTransferFunction(
                PyArray_ISALIGNED(self), itemsize, itemsize, dtype, dtype, 0,
                &cast_info, &flags) < 0) {
            goto fail;
        }
        
        // 如果转换函数不需要 Python API，则启动线程
        if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
            NPY_BEGIN_THREADS;
        }

        // 遍历 mask_data，根据 mask_data[i] 决定是否进行类型转换
        for (npy_intp i = 0, j = 0; i < ni; i++, j++) {
            // 如果 j 超过了 values 的长度 nv，则重置为 0
            if (j >= nv) {
                j = 0;
            }
            // 如果 mask_data[i] 为真，则进行类型转换
            if (mask_data[i]) {
                char *data[2] = {src + j*itemsize, dest + i*itemsize};
                // 调用转换函数进行数据转换，如果失败则跳转到错误处理标签 fail
                if (cast_info.func(
                        &cast_info.context, data, &one, strides,
                        cast_info.auxdata) < 0) {
                    NPY_END_THREADS;
                    NPY_cast_info_xfree(&cast_info);
                    goto fail;
                }
            }
        }
        // 释放类型转换信息 cast_info
        NPY_cast_info_xfree(&cast_info);
    }
    else {
        // 如果 self 的数据类型不需要引用计数检查，则启动线程进行快速的数据更新
        NPY_BEGIN_THREADS;
        npy_fastputmask(dest, src, mask_data, ni, nv, itemsize);
    }

    // 结束线程
    NPY_END_THREADS;

    // 释放 values 和 mask 的引用
    Py_XDECREF(values);
    Py_XDECREF(mask);
    
    // 如果进行了数据拷贝，则解析写回数据并释放 self
    if (copied) {
        PyArray_ResolveWritebackIfCopy(self);
        Py_DECREF(self);
    }
    
    // 返回 None
    Py_RETURN_NONE;

 fail:
    // 错误处理，释放 mask 和 values 的引用，并根据是否拷贝了数据进行处理
    Py_XDECREF(mask);
    Py_XDECREF(values);
    if (copied) {
        PyArray_DiscardWritebackIfCopy(self);
        Py_XDECREF(self);
    }
    // 返回 NULL 指示出现错误
    return NULL;
}

static NPY_GCC_OPT_3 inline int
npy_fastrepeat_impl(
    npy_intp n_outer, npy_intp n, npy_intp nel, npy_intp chunk,
    npy_bool broadcast, npy_intp* counts, char* new_data, char* old_data,
    npy_intp elsize, NPY_cast_info cast_info, int needs_refcounting)
{
    // 外层循环，循环次数为 n_outer
    npy_intp i, j, k;
    for (i = 0; i < n_outer; i++) {
        // 内层循环，循环次数为 n
        for (j = 0; j < n; j++) {
            // 计算重复次数 tmp，如果是广播则取 counts[0]，否则取 counts[j]
            npy_intp tmp = broadcast ? counts[0] : counts[j];
            // 根据 tmp 执行数据复制操作的循环
            for (k = 0; k < tmp; k++) {
                // 如果不需要引用计数，直接使用 memcpy 复制数据块
                if (!needs_refcounting) {
                    memcpy(new_data, old_data, chunk);
                }
                // 否则，执行类型转换并复制数据块
                else {
                    char *data[2] = {old_data, new_data};
                    npy_intp strides[2] = {elsize, elsize};
                    // 调用类型转换函数
                    if (cast_info.func(&cast_info.context, data, &nel,
                                       strides, cast_info.auxdata) < 0) {
                        return -1;
                    }
                }
                // 更新 new_data 的位置到下一个数据块的起始位置
                new_data += chunk;
            }
            // 更新 old_data 的位置到下一个数据块的起始位置
            old_data += chunk;
        }
    }
    // 函数执行成功返回 0
    return 0;
}

static NPY_GCC_OPT_3 int
npy_fastrepeat(
    npy_intp n_outer, npy_intp n, npy_intp nel, npy_intp chunk,
    npy_bool broadcast, npy_intp* counts, char* new_data, char* old_data,
    npy_intp elsize, NPY_cast_info cast_info, int needs_refcounting)
{
    // 如果不需要引用计数，直接调用 npy_fastrepeat_impl 函数
    if (!needs_refcounting) {
        // 根据不同的 chunk 大小调用 npy_fastrepeat_impl 函数
        if (chunk == 1 || chunk == 2 || chunk == 4 || chunk == 8 ||
            chunk == 16 || chunk == 32) {
            return npy_fastrepeat_impl(
                n_outer, n, nel, chunk, broadcast, counts, new_data, old_data,
                elsize, cast_info, needs_refcounting);
        }
    }

    // 否则，无论 chunk 大小，都调用 npy_fastrepeat_impl 函数
    return npy_fastrepeat_impl(
        n_outer, n, nel, chunk, broadcast, counts, new_data, old_data, elsize,
        cast_info, needs_refcounting);
}


/*NUMPY_API
 * Repeat the array.
 */
NPY_NO_EXPORT PyObject *
PyArray_Repeat(PyArrayObject *aop, PyObject *op, int axis)
{
    // 声明和初始化变量
    npy_intp *counts;
    npy_intp i, j, n, n_outer, chunk, elsize, nel;
    npy_intp total = 0;
    # 初始化广播标志为假
    npy_bool broadcast = NPY_FALSE;
    # 初始化重复对象为空
    PyArrayObject *repeats = NULL;
    # 初始化变量指针为NULL
    PyObject *ap = NULL;
    # 初始化返回数组对象为空
    PyArrayObject *ret = NULL;
    # 初始化新旧数据的字符指针
    char *new_data, *old_data;
    # 初始化类型转换信息
    NPY_cast_info cast_info;
    # 初始化数组方法标志
    NPY_ARRAYMETHOD_FLAGS flags;
    # 初始化需要引用计数的标志
    int needs_refcounting;

    # 从任何类型的输入数组创建连续的重复对象
    repeats = (PyArrayObject *)PyArray_ContiguousFromAny(op, NPY_INTP, 0, 1);
    if (repeats == NULL) {
        return NULL;
    }

    /*
     * 标量和大小为1的'repeat'数组可以广播到任何形状，对于所有其他输入，维度必须完全匹配。
     */
    if (PyArray_NDIM(repeats) == 0 || PyArray_SIZE(repeats) == 1) {
        broadcast = NPY_TRUE;
    }

    # 获取重复数组的数据指针
    counts = (npy_intp *)PyArray_DATA(repeats);

    # 检查输入数组是否满足指定轴条件，并返回C语言风格的对象
    if ((ap = PyArray_CheckAxis(aop, &axis, NPY_ARRAY_CARRAY)) == NULL) {
        Py_DECREF(repeats);
        return NULL;
    }

    # 将返回的对象转换为PyArrayObject类型
    aop = (PyArrayObject *)ap;
    # 获取指定轴的维度大小
    n = PyArray_DIM(aop, axis);
    # 初始化类型转换信息
    NPY_cast_info_init(&cast_info);
    # 检查数据类型是否需要引用计数
    needs_refcounting = PyDataType_REFCHK(PyArray_DESCR(aop));

    # 如果不是广播，并且重复数组的大小与指定轴的维度大小不匹配，则引发值错误
    if (!broadcast && PyArray_SIZE(repeats) != n) {
        PyErr_Format(PyExc_ValueError,
                     "operands could not be broadcast together "
                     "with shape (%zd,) (%zd,)", n, PyArray_DIM(repeats, 0));
        goto fail;
    }
    # 如果是广播，则计算总数
    if (broadcast) {
        total = counts[0] * n;
    }
    else {
        # 否则遍历重复计数数组并累加总数，同时检查重复计数是否为负数
        for (j = 0; j < n; j++) {
            if (counts[j] < 0) {
                PyErr_SetString(PyExc_ValueError,
                                "repeats may not contain negative values.");
                goto fail;
            }
            total += counts[j];
        }
    }

    /* 构建新的数组 */
    # 设置指定轴的维度为总数
    PyArray_DIMS(aop)[axis] = total;
    # 增加数组描述符的引用计数
    Py_INCREF(PyArray_DESCR(aop));
    # 根据给定描述符和参数创建新的数组对象
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(aop),
                                                PyArray_DESCR(aop),
                                                PyArray_NDIM(aop),
                                                PyArray_DIMS(aop),
                                                NULL, NULL, 0,
                                                (PyObject *)aop);
    # 恢复指定轴的维度为原始值
    PyArray_DIMS(aop)[axis] = n;
    # 如果创建数组对象失败，则跳转到fail标签处
    if (ret == NULL) {
        goto fail;
    }
    # 获取新数组的数据指针
    new_data = PyArray_DATA(ret);
    # 获取原始数组的数据指针
    old_data = PyArray_DATA(aop);

    # 初始化元素数量为1
    nel = 1;
    # 初始化每个元素的大小
    elsize = PyArray_ITEMSIZE(aop);
    # 计算从指定轴之后的维度的元素数量乘积
    for(i = axis + 1; i < PyArray_NDIM(aop); i++) {
        nel *= PyArray_DIMS(aop)[i];
    }
    # 计算每块数据的大小
    chunk = nel*elsize;

    # 初始化外部循环计数为1
    n_outer = 1;
    # 计算指定轴之前的维度的元素数量乘积
    for (i = 0; i < axis; i++) {
        n_outer *= PyArray_DIMS(aop)[i];
    }

    # 如果需要引用计数，则获取数据类型的转移函数
    if (needs_refcounting) {
        if (PyArray_GetDTypeTransferFunction(
                1, elsize, elsize, PyArray_DESCR(aop), PyArray_DESCR(aop), 0,
                &cast_info, &flags) < 0) {
            goto fail;
        }
    }

    # 使用快速重复函数对数组进行重复操作
    if (npy_fastrepeat(n_outer, n, nel, chunk, broadcast, counts, new_data,
                       old_data, elsize, cast_info, needs_refcounting) < 0) {
        goto fail;
    }

    # 释放重复对象的引用
    Py_DECREF(repeats);
    # 释放aop对象的引用
    Py_XDECREF(aop);
    # 释放类型转换信息的内存
    NPY_cast_info_xfree(&cast_info);
    return (PyObject *)ret;

# 将 ret 转换为 PyObject 指针后返回，函数执行成功时返回结果对象。


 fail:
    Py_DECREF(repeats);
    Py_XDECREF(aop);
    Py_XDECREF(ret);
    NPY_cast_info_xfree(&cast_info);
    return NULL;

# 处理函数执行失败的情况：逐一清理之前分配的资源（减少 repeats 的引用计数，释放 aop 的引用，释放 ret 的引用，释放 cast_info 的内存），然后返回空指针表示失败。
    /*
     * 转换所有输入为相同类型的数组
     * 同时使它们变为 C 连续数组
     */
    mps = PyArray_ConvertToCommonType(op, &n);
    if (mps == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    ap = (PyArrayObject *)PyArray_FROM_OT((PyObject *)ip, NPY_INTP);
    if (ap == NULL) {
        goto fail;
    }
    /* 将所有数组广播到彼此，最后是索引数组 */
    multi = (PyArrayMultiIterObject *)
        PyArray_MultiIterFromObjects((PyObject **)mps, n, 1, ap);
    if (multi == NULL) {
        goto fail;
    }
    dtype = PyArray_DESCR(mps[0]);

    /* 设置返回数组 */
    if (out == NULL) {
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(ap),
                                                    dtype,
                                                    multi->nd,
                                                    multi->dimensions,
                                                    NULL, NULL, 0,
                                                    (PyObject *)ap);
    }
    else {
        int flags = NPY_ARRAY_CARRAY |
                    NPY_ARRAY_WRITEBACKIFCOPY |
                    NPY_ARRAY_FORCECAST;

        if ((PyArray_NDIM(out) != multi->nd)
                    || !PyArray_CompareLists(PyArray_DIMS(out),
                                             multi->dimensions,
                                             multi->nd)) {
            PyErr_SetString(PyExc_TypeError,
                            "choose: invalid shape for output array.");
            goto fail;
        }

        for (i = 0; i < n; i++) {
            if (arrays_overlap(out, mps[i])) {
                flags |= NPY_ARRAY_ENSURECOPY;
            }
        }

        if (clipmode == NPY_RAISE) {
            /*
             * 需要确保并获取一个副本，
             * 以便在调用错误之前不更改输入数组
             */
            flags |= NPY_ARRAY_ENSURECOPY;
        }
        Py_INCREF(dtype);
        obj = (PyArrayObject *)PyArray_FromArray(out, dtype, flags);
    }

    if (obj == NULL) {
        goto fail;
    }
    elsize = dtype->elsize;
    ret_data = PyArray_DATA(obj);
    npy_intp transfer_strides[2] = {elsize, elsize};
    npy_intp one = 1;
    NPY_ARRAYMETHOD_FLAGS transfer_flags = 0;
    // 检查数据类型是否需要引用计数，如果是则执行以下操作
    if (PyDataType_REFCHK(dtype)) {
        // 检查对象是否按无符号整数对齐
        int is_aligned = IsUintAligned(obj);
        // 获取数据类型转换函数，并设置转换的相关信息
        PyArray_GetDTypeTransferFunction(
                    is_aligned,
                    dtype->elsize,
                    dtype->elsize,
                    dtype,
                    dtype, 0, &cast_info,
                    &transfer_flags);
    }

    // 循环遍历多重迭代器，直到迭代完成
    while (PyArray_MultiIter_NOTDONE(multi)) {
        // 获取当前迭代的索引
        mi = *((npy_intp *)PyArray_MultiIter_DATA(multi, n));
        // 如果索引超出了有效范围，则根据剪切模式进行处理
        if (mi < 0 || mi >= n) {
            switch(clipmode) {
            case NPY_RAISE:
                // 如果剪切模式为 NPY_RAISE，则引发值错误异常
                PyErr_SetString(PyExc_ValueError,
                        "invalid entry in choice "\
                        "array");
                // 跳转到失败处理标签
                goto fail;
            case NPY_WRAP:
                // 如果剪切模式为 NPY_WRAP，则根据循环方式处理超出范围的索引
                if (mi < 0) {
                    while (mi < 0) {
                        mi += n;
                    }
                }
                else {
                    while (mi >= n) {
                        mi -= n;
                    }
                }
                break;
            case NPY_CLIP:
                // 如果剪切模式为 NPY_CLIP，则将超出范围的索引调整到合法范围内
                if (mi < 0) {
                    mi = 0;
                }
                else if (mi >= n) {
                    mi = n - 1;
                }
                break;
            }
        }
        // 如果转换信息中的函数为空，则使用 memcpy 复制数据，因为内存不重叠
        if (cast_info.func == NULL) {
            /* We ensure memory doesn't overlap, so can use memcpy */
            memcpy(ret_data, PyArray_MultiIter_DATA(multi, mi), elsize);
        }
        else {
            // 否则，调用转换函数进行数据转换
            char *args[2] = {PyArray_MultiIter_DATA(multi, mi), ret_data};
            if (cast_info.func(&cast_info.context, args, &one,
                                transfer_strides, cast_info.auxdata) < 0) {
                // 如果转换失败，则跳转到失败处理标签
                goto fail;
            }
        }
        // 更新返回数据的指针位置
        ret_data += elsize;
        // 更新多重迭代器的状态，移动到下一个迭代位置
        PyArray_MultiIter_NEXT(multi);
    }

    // 释放转换信息所占用的内存
    NPY_cast_info_xfree(&cast_info);
    // 释放多重迭代器对象
    Py_DECREF(multi);
    // 释放所有的缓冲数组对象
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    // 释放输入数组对象
    Py_DECREF(ap);
    // 释放缓冲数组对象的内存
    PyDataMem_FREE(mps);
    // 如果有输出对象且不是输入对象的引用，则增加输出对象的引用计数
    if (out != NULL && out != obj) {
        Py_INCREF(out);
        // 解析写回（如果有必要）
        PyArray_ResolveWritebackIfCopy(obj);
        // 释放输入对象的引用
        Py_DECREF(obj);
        // 将输出对象设置为当前操作对象
        obj = out;
    }
    // 返回 Python 对象指针类型的对象
    return (PyObject *)obj;

 fail:
    // 在失败的情况下，释放转换信息所占用的内存
    NPY_cast_info_xfree(&cast_info);
    // 减少多重迭代器对象的引用计数
    Py_XDECREF(multi);
    // 释放所有的缓冲数组对象
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    // 减少输入数组对象的引用计数
    Py_XDECREF(ap);
    // 释放缓冲数组对象的内存
    PyDataMem_FREE(mps);
    // 放弃写回（如果有必要）
    PyArray_DiscardWritebackIfCopy(obj);
    // 减少当前操作对象的引用计数
    Py_XDECREF(obj);
    // 返回空指针，表示操作失败
    return NULL;
/*
 * These algorithms use special sorting.  They are not called unless the
 * underlying sort function for the type is available.  Note that axis is
 * already valid. The sort functions require 1-d contiguous and well-behaved
 * data.  Therefore, a copy will be made of the data if needed before handing
 * it to the sorting routine.  An iterator is constructed and adjusted to walk
 * over all but the desired sorting axis.
 */
static int
_new_sortlike(PyArrayObject *op, int axis, PyArray_SortFunc *sort,
              PyArray_PartitionFunc *part, npy_intp const *kth, npy_intp nkth)
{
    // 获取指定轴向的数组维度大小
    npy_intp N = PyArray_DIM(op, axis);
    // 获取数组元素的大小（字节数）
    npy_intp elsize = (npy_intp)PyArray_ITEMSIZE(op);
    // 获取指定轴向的数组步长
    npy_intp astride = PyArray_STRIDE(op, axis);
    // 检查数组是否被字节交换过
    int swap = PyArray_ISBYTESWAPPED(op);
    // 检查数组是否内存对齐
    int is_aligned = IsAligned(op);
    // 判断是否需要复制数据
    int needcopy = !is_aligned || swap || astride != elsize;
    // 检查数组描述符是否需要 Python API 支持
    int needs_api = PyDataType_FLAGCHK(PyArray_DESCR(op), NPY_NEEDS_PYAPI);

    // 缓冲区指针初始化为空
    char *buffer = NULL;

    // 迭代器对象初始化为空
    PyArrayIterObject *it;
    // 迭代器对象大小初始化
    npy_intp size;

    // 返回值初始化为 0
    int ret = 0;

    // 获取数组描述符
    PyArray_Descr *descr = PyArray_DESCR(op);
    // 原数组描述符初始化为空
    PyArray_Descr *odescr = NULL;

    // 转换信息初始化
    NPY_cast_info to_cast_info = {.func = NULL};
    NPY_cast_info from_cast_info = {.func = NULL};

    // 多线程操作开始
    NPY_BEGIN_THREADS_DEF;

    /* Check if there is any sorting to do */
    // 检查是否需要进行排序操作
    if (N <= 1 || PyArray_SIZE(op) == 0) {
        return 0;
    }

    // 获取内存处理器句柄
    PyObject *mem_handler = PyDataMem_GetHandler();
    // 如果内存处理器句柄为空，返回错误
    if (mem_handler == NULL) {
        return -1;
    }
    // 构建除指定轴外的所有维度的迭代器
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    // 如果迭代器为空，释放内存处理器句柄并返回错误
    if (it == NULL) {
        Py_DECREF(mem_handler);
        return -1;
    }
    // 获取迭代器的大小
    size = it->size;

    // 如果需要复制数据
    if (needcopy) {
        // 根据内存处理器分配缓冲区
        buffer = PyDataMem_UserNEW(N * elsize, mem_handler);
        // 如果分配缓冲区失败，设置返回值为 -1 并跳转到失败处理标签
        if (buffer == NULL) {
            ret = -1;
            goto fail;
        }
        // 如果数组描述符标记需要初始化，将缓冲区初始化为零
        if (PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            memset(buffer, 0, N * elsize);
        }

        // 如果数组被交换过字节顺序，创建新的字节顺序描述符
        if (swap) {
            odescr = PyArray_DescrNewByteorder(descr, NPY_SWAP);
        }
        // 否则直接使用原数组描述符
        else {
            odescr = descr;
            Py_INCREF(odescr);
        }

        // 获取数据类型转换函数和转换标志信息
        NPY_ARRAYMETHOD_FLAGS to_transfer_flags;
        if (PyArray_GetDTypeTransferFunction(
                is_aligned, astride, elsize, descr, odescr, 0, &to_cast_info,
                &to_transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }

        // 获取数据类型转换函数和转换标志信息（反向转换）
        NPY_ARRAYMETHOD_FLAGS from_transfer_flags;
        if (PyArray_GetDTypeTransferFunction(
                is_aligned, elsize, astride, odescr, descr, 0, &from_cast_info,
                &from_transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
    }

    // 多线程操作开始（使用数组描述符）
    NPY_BEGIN_THREADS_DESCR(descr);
    // 循环，每次迭代减小 size 的值
    while (size--) {
        // 获取当前迭代器指向的数据指针
        char *bufptr = it->dataptr;

        // 如果需要复制数据
        if (needcopy) {
            // 设置函数参数和步幅以进行类型转换
            char *args[2] = {it->dataptr, buffer};
            npy_intp strides[2] = {astride, elsize};

            // 调用类型转换函数，如果返回值小于 0，则跳转到失败标签
            if (NPY_UNLIKELY(to_cast_info.func(
                                 &to_cast_info.context, args, &N, strides,
                                 to_cast_info.auxdata) < 0)) {
                goto fail;
            }
            // 更新 bufptr 指向 buffer
            bufptr = buffer;
        }

        /*
         * TODO: 如果输入数组进行了字节交换，但是是连续且对齐的，
         * 可以在不复制到缓冲区的情况下直接进行交换（稍后再进行还原）。
         * 在调用 sort 或 part 函数时，需要确保即使调用出错，仍然能够在返回之前还原交换。
         */

        // 如果 part 函数为 NULL，则调用 sort 函数对 bufptr 所指向的数据进行排序
        if (part == NULL) {
            ret = sort(bufptr, N, op);
            // 如果需要 API 并且发生了异常，则设置返回值为 -1
            if (needs_api && PyErr_Occurred()) {
                ret = -1;
            }
            // 如果 sort 函数返回值小于 0，则跳转到失败标签
            if (ret < 0) {
                goto fail;
            }
        }
        // 否则，调用 part 函数对 bufptr 所指向的数据进行分区处理
        else {
            npy_intp pivots[NPY_MAX_PIVOT_STACK];
            npy_intp npiv = 0;
            npy_intp i;
            // 对 kth 中的每个值调用 part 函数
            for (i = 0; i < nkth; ++i) {
                ret = part(bufptr, N, kth[i], pivots, &npiv, nkth, op);
                // 如果需要 API 并且发生了异常，则设置返回值为 -1
                if (needs_api && PyErr_Occurred()) {
                    ret = -1;
                }
                // 如果 part 函数返回值小于 0，则跳转到失败标签
                if (ret < 0) {
                    goto fail;
                }
            }
        }

        // 如果需要复制数据，则进行逆类型转换
        if (needcopy) {
            // 设置函数参数和步幅以进行逆类型转换
            char *args[2] = {buffer, it->dataptr};
            npy_intp strides[2] = {elsize, astride};

            // 调用逆类型转换函数，如果返回值小于 0，则跳转到失败标签
            if (NPY_UNLIKELY(from_cast_info.func(
                                 &from_cast_info.context, args, &N, strides,
                                 from_cast_info.auxdata) < 0)) {
                goto fail;
            }
        }

        // 更新迭代器以处理下一个元素
        PyArray_ITER_NEXT(it);
    }
fail:
    # 结束可能存在的线程
    NPY_END_THREADS_DESCR(descr);
    /* cleanup internal buffer */
    # 如果需要拷贝数据，清理内部缓冲区
    if (needcopy) {
        PyArray_ClearBuffer(odescr, buffer, elsize, N, 1);
        PyDataMem_UserFREE(buffer, N * elsize, mem_handler);
        Py_DECREF(odescr);
    }
    # 如果返回值小于0且没有发生Python错误
    if (ret < 0 && !PyErr_Occurred()) {
        /* Out of memory during sorting or buffer creation */
        # 在排序或缓冲区创建期间内存不足
        PyErr_NoMemory();
    }
    // if an error happened with a dtype that doesn't hold the GIL, need
    // to make sure we return an error value from this function.
    // note: only the first error is ever reported, subsequent errors
    // must *not* set the error handler.
    // 如果使用了不持有GIL的数据类型，并且发生了错误，确保从该函数返回错误值。
    // 注意：只有第一个错误会被报告，后续错误不应设置错误处理程序。
    if (PyErr_Occurred() && ret == 0) {
        ret = -1;
    }
    # 释放迭代器和内存处理器
    Py_DECREF(it);
    Py_DECREF(mem_handler);
    # 释放类型转换信息
    NPY_cast_info_xfree(&to_cast_info);
    NPY_cast_info_xfree(&from_cast_info);

    # 返回函数结果
    return ret;
}

static PyObject*
_new_argsortlike(PyArrayObject *op, int axis, PyArray_ArgSortFunc *argsort,
                 PyArray_ArgPartitionFunc *argpart,
                 npy_intp const *kth, npy_intp nkth)
{
    # 获取操作数组的维度大小
    npy_intp N = PyArray_DIM(op, axis);
    # 获取元素大小
    npy_intp elsize = (npy_intp)PyArray_ITEMSIZE(op);
    # 获取操作数组在指定轴上的步长
    npy_intp astride = PyArray_STRIDE(op, axis);
    # 检查是否进行了字节交换
    int swap = PyArray_ISBYTESWAPPED(op);
    # 检查数组是否对齐
    int is_aligned = IsAligned(op);
    # 判断是否需要拷贝数据
    int needcopy = !is_aligned || swap || astride != elsize;
    # 检查是否需要使用 Python C API
    int needs_api = PyDataType_FLAGCHK(PyArray_DESCR(op), NPY_NEEDS_PYAPI);
    # 检查是否需要索引缓冲区
    int needidxbuffer;

    char *valbuffer = NULL;
    npy_intp *idxbuffer = NULL;

    # 创建一个新的数组对象
    PyArrayObject *rop;
    # 目标数组的步长
    npy_intp rstride;

    # 迭代器对象和反向迭代器对象
    PyArrayIterObject *it, *rit;
    # 迭代器对象的大小
    npy_intp size;

    # 返回值的初始化
    int ret = 0;

    # 源数组的数据类型描述符
    PyArray_Descr *descr = PyArray_DESCR(op);
    # 目标数组的数据类型描述符
    PyArray_Descr *odescr = NULL;

    # 数组方法传输标志
    NPY_ARRAYMETHOD_FLAGS transfer_flags;
    # 类型转换信息
    NPY_cast_info cast_info = {.func = NULL};

    # 启动线程（如果已定义）
    NPY_BEGIN_THREADS_DEF;

    # 获取内存处理器
    PyObject *mem_handler = PyDataMem_GetHandler();
    if (mem_handler == NULL) {
        return NULL;
    }
    # 从数据描述符创建一个新的数组对象
    rop = (PyArrayObject *)PyArray_NewFromDescr(
            Py_TYPE(op), PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(op), PyArray_DIMS(op), NULL, NULL,
            0, (PyObject *)op);
    if (rop == NULL) {
        Py_DECREF(mem_handler);
        return NULL;
    }
    # 获取目标数组在指定轴上的步长
    rstride = PyArray_STRIDE(rop, axis);
    # 检查是否需要索引缓冲区
    needidxbuffer = rstride != sizeof(npy_intp);

    /* Check if there is any argsorting to do */
    # 检查是否需要进行排序操作
    if (N <= 1 || PyArray_SIZE(op) == 0) {
        # 释放内存处理器
        Py_DECREF(mem_handler);
        # 如果数组大小小于等于1或者数组大小为0，则直接返回全0的目标数组
        memset(PyArray_DATA(rop), 0, PyArray_NBYTES(rop));
        return (PyObject *)rop;
    }

    # 获取操作数组的所有轴的迭代器，除了指定的轴
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)op, &axis);
    rit = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)rop, &axis);
    if (it == NULL || rit == NULL) {
        # 设置返回值为-1，并跳转到fail标签处
        ret = -1;
        goto fail;
    }
    # 获取迭代器对象的大小
    size = it->size;
    // 如果需要进行复制操作
    if (needcopy) {
        // 使用自定义的内存分配函数分配空间给valbuffer
        valbuffer = PyDataMem_UserNEW(N * elsize, mem_handler);
        // 检查分配是否成功
        if (valbuffer == NULL) {
            ret = -1;
            goto fail;
        }
        // 如果描述符需要初始化，使用memset将valbuffer清零
        if (PyDataType_FLAGCHK(descr, NPY_NEEDS_INIT)) {
            memset(valbuffer, 0, N * elsize);
        }

        // 如果需要进行字节顺序转换
        if (swap) {
            // 创建一个新的描述符，指定为需要交换字节顺序
            odescr = PyArray_DescrNewByteorder(descr, NPY_SWAP);
        }
        else {
            // 否则直接使用当前的描述符
            odescr = descr;
            Py_INCREF(odescr);
        }

        // 获取适合描述符转换的函数，初始化转换信息
        if (PyArray_GetDTypeTransferFunction(
                is_aligned, astride, elsize, descr, odescr, 0, &cast_info,
                &transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
    }

    // 如果需要索引缓冲区
    if (needidxbuffer) {
        // 使用自定义的内存分配函数分配空间给idxbuffer
        idxbuffer = (npy_intp *)PyDataMem_UserNEW(N * sizeof(npy_intp),
                                                  mem_handler);
        // 检查分配是否成功
        if (idxbuffer == NULL) {
            ret = -1;
            goto fail;
        }
    }

    // 开始线程安全操作，使用给定的描述符
    NPY_BEGIN_THREADS_DESCR(descr);

    // 迭代处理数据的每一项
    while (size--) {
        // 获取当前迭代器指向的值指针和索引指针
        char *valptr = it->dataptr;
        npy_intp *idxptr = (npy_intp *)rit->dataptr;
        npy_intp *iptr, i;

        // 如果需要复制操作
        if (needcopy) {
            // 设置参数数组和步长数组，调用类型转换函数
            char *args[2] = {it->dataptr, valbuffer};
            npy_intp strides[2] = {astride, elsize};

            // 如果转换失败，则跳转到错误处理步骤
            if (NPY_UNLIKELY(cast_info.func(
                                 &cast_info.context, args, &N, strides,
                                 cast_info.auxdata) < 0)) {
                goto fail;
            }
            // 更新值指针为valbuffer，表示使用复制后的数据
            valptr = valbuffer;
        }

        // 如果需要索引缓冲区，更新索引指针
        if (needidxbuffer) {
            idxptr = idxbuffer;
        }

        // 初始化iptr为idxptr，然后为每个元素设置递增索引
        iptr = idxptr;
        for (i = 0; i < N; ++i) {
            *iptr++ = i;
        }

        // 如果未提供argpart函数，调用argsort函数进行排序
        if (argpart == NULL) {
            ret = argsort(valptr, idxptr, N, op);
            /* 在Python 3中，对象比较可能引发异常 */
            if (needs_api && PyErr_Occurred()) {
                ret = -1;
            }
            // 如果排序操作返回小于0的值，跳转到错误处理步骤
            if (ret < 0) {
                goto fail;
            }
        }
        // 否则，使用argpart函数进行部分排序
        else {
            npy_intp pivots[NPY_MAX_PIVOT_STACK];
            npy_intp npiv = 0;

            for (i = 0; i < nkth; ++i) {
                ret = argpart(valptr, idxptr, N, kth[i], pivots, &npiv, nkth, op);
                /* 在Python 3中，对象比较可能引发异常 */
                if (needs_api && PyErr_Occurred()) {
                    ret = -1;
                }
                // 如果排序操作返回小于0的值，跳转到错误处理步骤
                if (ret < 0) {
                    goto fail;
                }
            }
        }

        // 如果需要索引缓冲区，将排序后的索引写回原始数据中
        if (needidxbuffer) {
            char *rptr = rit->dataptr;
            iptr = idxbuffer;

            for (i = 0; i < N; ++i) {
                *(npy_intp *)rptr = *iptr++;
                rptr += rstride;
            }
        }

        // 更新迭代器，使其指向下一个元素
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(rit);
    }
fail:
    // 调用描述符的结束线程函数，完成线程操作
    NPY_END_THREADS_DESCR(descr);
    
    /* 清理内部缓冲区 */
    // 如果需要复制数据，则清理值缓冲区，并释放相关内存
    if (needcopy) {
        PyArray_ClearBuffer(odescr, valbuffer, elsize, N, 1);
        PyDataMem_UserFREE(valbuffer, N * elsize, mem_handler);
        Py_DECREF(odescr);
    }
    
    // 释放索引缓冲区的内存
    PyDataMem_UserFREE(idxbuffer, N * sizeof(npy_intp), mem_handler);
    
    // 如果返回值小于0，表示出现错误
    if (ret < 0) {
        // 如果没有设置异常，则设置内存不足的异常
        if (!PyErr_Occurred()) {
            /* 在排序或缓冲区创建过程中内存不足 */
            PyErr_NoMemory();
        }
        // 释放结果对象的引用
        Py_XDECREF(rop);
        rop = NULL;
    }
    
    // 释放迭代器对象的引用
    Py_XDECREF(it);
    Py_XDECREF(rit);
    
    // 释放内存处理器对象的引用
    Py_DECREF(mem_handler);
    
    // 释放类型转换信息的内存
    NPY_cast_info_xfree(&cast_info);

    // 返回排序后的结果对象
    return (PyObject *)rop;
}



/*NUMPY_API
 * 对数组进行原地排序
 */
NPY_NO_EXPORT int
PyArray_Sort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArray_SortFunc *sort = NULL;
    int n = PyArray_NDIM(op);

    // 检查并调整轴的值
    if (check_and_adjust_axis(&axis, n) < 0) {
        return -1;
    }

    // 确保数组可写
    if (PyArray_FailUnlessWriteable(op, "sort array") < 0) {
        return -1;
    }

    // 检查排序类型的有效性
    if (which < 0 || which >= NPY_NSORTS) {
        PyErr_SetString(PyExc_ValueError, "not a valid sort kind");
        return -1;
    }

    // 获取排序函数
    sort = PyDataType_GetArrFuncs(PyArray_DESCR(op))->sort[which];

    // 如果排序函数为空
    if (sort == NULL) {
        // 如果类型有比较函数，则根据排序类型选择默认的排序算法
        if (PyDataType_GetArrFuncs(PyArray_DESCR(op))->compare) {
            switch (which) {
                default:
                case NPY_QUICKSORT:
                    sort = npy_quicksort;
                    break;
                case NPY_HEAPSORT:
                    sort = npy_heapsort;
                    break;
                case NPY_STABLESORT:
                    sort = npy_timsort;
                    break;
            }
        }
        else {
            // 类型没有比较函数则设置类型错误异常
            PyErr_SetString(PyExc_TypeError,
                            "type does not have compare function");
            return -1;
        }
    }

    // 调用新的排序函数进行排序
    return _new_sortlike(op, axis, sort, NULL, NULL, 0);
}



/*
 * 使第k个数组元素为正数，展平并排序
 */
static PyArrayObject *
partition_prep_kth_array(PyArrayObject * ktharray,
                         PyArrayObject * op,
                         int axis)
{
    const npy_intp * shape = PyArray_SHAPE(op);
    PyArrayObject * kthrvl;
    npy_intp * kth;
    npy_intp nkth, i;

    // 如果ktharray是布尔类型则发出警告，并返回NULL
    if (PyArray_ISBOOL(ktharray)) {
        /* 2021-09-29, NumPy 1.22 */
        if (DEPRECATE(
                "Passing booleans as partition index is deprecated"
                " (warning added in NumPy 1.22)") < 0) {
            return NULL;
        }
    }
    // 如果ktharray不是整数类型则设置类型错误并返回NULL
    else if (!PyArray_ISINTEGER(ktharray)) {
        PyErr_Format(PyExc_TypeError, "Partition index must be integer");
        return NULL;
    }

    // 如果ktharray的维度大于1则设置值错误并返回NULL
    if (PyArray_NDIM(ktharray) > 1) {
        PyErr_Format(PyExc_ValueError, "kth array must have dimension <= 1");
        return NULL;
    }
    // 将ktharray转换为整型数组
    kthrvl = (PyArrayObject *)PyArray_Cast(ktharray, NPY_INTP);

    if (kthrvl == NULL)
        return NULL;

    // 获取kth数组的数据指针和大小
    kth = PyArray_DATA(kthrvl);
    nkth = PyArray_SIZE(kthrvl);
    # 对每一个 kth 值进行检查和修正，确保其在合法范围内
    for (i = 0; i < nkth; i++) {
        # 如果 kth 值小于 0，则加上数组在指定轴上的长度，使其非负
        if (kth[i] < 0) {
            kth[i] += shape[axis];
        }
        # 如果操作数组 op 的大小不为 0，并且 kth[i] 超出了指定轴的边界
        if (PyArray_SIZE(op) != 0 &&
                    (kth[i] < 0 || kth[i] >= shape[axis])) {
            # 报错：kth 值超出边界，给出具体数值和边界信息
            PyErr_Format(PyExc_ValueError, "kth(=%zd) out of bounds (%zd)",
                         kth[i], shape[axis]);
            # 释放 kthrvl 对象
            Py_XDECREF(kthrvl);
            # 返回空指针，表示出错
            return NULL;
        }
    }

    /*
     * 对 kthrvl 数组进行排序，以确保分区不会相互重叠
     */
    if (PyArray_SIZE(kthrvl) > 1) {
        # 使用快速排序算法对 kthrvl 数组进行降序排序
        PyArray_Sort(kthrvl, -1, NPY_QUICKSORT);
    }

    # 返回排序后的 kthrvl 数组
    return kthrvl;
/*NUMPY_API
 * Partition an array in-place
 */
NPY_NO_EXPORT int
PyArray_Partition(PyArrayObject *op, PyArrayObject * ktharray, int axis,
                  NPY_SELECTKIND which)
{
    PyArrayObject *kthrvl; // 声明 kthrvl 变量，用于存储处理后的 ktharray 数组对象
    PyArray_PartitionFunc *part; // 声明 part 变量，用于存储分区函数指针
    PyArray_SortFunc *sort; // 声明 sort 变量，用于存储排序函数指针
    int n = PyArray_NDIM(op); // 获取数组 op 的维度数，存入 n 中
    int ret; // 声明 ret 变量，用于存储函数返回值

    if (check_and_adjust_axis(&axis, n) < 0) { // 检查并调整轴的值，确保其有效性
        return -1; // 如果检查失败，返回错误码
    }

    if (PyArray_FailUnlessWriteable(op, "partition array") < 0) { // 检查数组 op 是否可写
        return -1; // 如果不可写，返回错误码
    }

    if (which < 0 || which >= NPY_NSELECTS) { // 检查 which 参数是否有效
        PyErr_SetString(PyExc_ValueError, "not a valid partition kind"); // 设置错误信息
        return -1; // 返回错误码
    }
    part = get_partition_func(PyArray_TYPE(op), which); // 获取分区函数，并存入 part 中
    if (part == NULL) { // 如果获取失败
        /* Use sorting, slower but equivalent */ // 使用排序代替分区，虽然更慢但功能相同
        if (PyDataType_GetArrFuncs(PyArray_DESCR(op))->compare) { // 检查类型是否具有比较函数
            sort = npy_quicksort; // 设置快速排序函数指针
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "type does not have compare function"); // 设置类型不具有比较函数的错误信息
            return -1; // 返回错误码
        }
    }

    /* Process ktharray even if using sorting to do bounds checking */
    kthrvl = partition_prep_kth_array(ktharray, op, axis); // 准备 ktharray 数组的处理，进行边界检查
    if (kthrvl == NULL) { // 如果处理失败
        return -1; // 返回错误码
    }

    ret = _new_sortlike(op, axis, sort, part,
                        PyArray_DATA(kthrvl), PyArray_SIZE(kthrvl)); // 执行排序或分区操作

    Py_DECREF(kthrvl); // 释放 kthrvl 对象的引用计数

    return ret; // 返回操作结果
}


/*NUMPY_API
 * ArgSort an array
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgSort(PyArrayObject *op, int axis, NPY_SORTKIND which)
{
    PyArrayObject *op2; // 声明 op2 变量，用于存储处理后的数组对象
    PyArray_ArgSortFunc *argsort = NULL; // 声明 argsort 变量，用于存储排序函数指针
    PyObject *ret; // 声明 ret 变量，用于存储函数返回值

    argsort = PyDataType_GetArrFuncs(PyArray_DESCR(op))->argsort[which]; // 获取指定排序类型的函数指针

    if (argsort == NULL) { // 如果未找到相应的排序函数
        if (PyDataType_GetArrFuncs(PyArray_DESCR(op))->compare) { // 检查类型是否具有比较函数
            switch (which) { // 根据排序类型选择排序函数
                default:
                case NPY_QUICKSORT:
                    argsort = npy_aquicksort; // 快速排序函数
                    break;
                case NPY_HEAPSORT:
                    argsort = npy_aheapsort; // 堆排序函数
                    break;
                case NPY_STABLESORT:
                    argsort = npy_atimsort; // 稳定排序函数
                    break;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "type does not have compare function"); // 设置类型不具有比较函数的错误信息
            return NULL; // 返回空指针表示错误
        }
    }

    op2 = (PyArrayObject *)PyArray_CheckAxis(op, &axis, 0); // 检查并返回处理后的数组对象 op2
    if (op2 == NULL) { // 如果检查失败
        return NULL; // 返回空指针表示错误
    }

    ret = _new_argsortlike(op2, axis, argsort, NULL, NULL, 0); // 执行类似于排序的操作

    Py_DECREF(op2); // 释放 op2 对象的引用计数
    return ret; // 返回操作结果
}


/*NUMPY_API
 * ArgPartition an array
 */
NPY_NO_EXPORT PyObject *
PyArray_ArgPartition(PyArrayObject *op, PyArrayObject *ktharray, int axis,
                     NPY_SELECTKIND which)
{
    PyArrayObject *op2, *kthrvl; // 声明 op2 和 kthrvl 变量，用于存储处理后的数组对象
    PyArray_ArgPartitionFunc *argpart; // 声明 argpart 变量，用于存储分区函数指针
    PyArray_ArgSortFunc *argsort; // 声明 argsort 变量，用于存储排序函数指针
    PyObject *ret; // 声明 ret 变量，用于存储函数返回值

    /*
     * As a C-exported function, enum NPY_SELECTKIND loses its enum property
     * Check the values to make sure they are in range
     */
    // 作为 C 导出函数，枚举 NPY_SELECTKIND 失去其枚举属性
    // 检查值以确保其在有效范围内
    # 检查输入的 which 参数是否在有效范围内
    if ((int)which < 0 || (int)which >= NPY_NSELECTS) {
        # 设置一个异常，指示无效的分区类型
        PyErr_SetString(PyExc_ValueError,
                        "not a valid partition kind");
        # 返回 NULL 表示出错
        return NULL;
    }

    # 根据操作数组的类型和指定的 which 参数获取相应的分区函数
    argpart = get_argpartition_func(PyArray_TYPE(op), which);
    if (argpart == NULL) {
        /* 如果没有找到相应的分区函数，则使用排序（更慢但等效的方法） */
        if (PyDataType_GetArrFuncs(PyArray_DESCR(op))->compare) {
            # 如果操作数组的数据类型有比较函数，则使用快速排序
            argsort = npy_aquicksort;
        }
        else {
            # 否则，设置一个异常，指示数据类型没有比较函数
            PyErr_SetString(PyExc_TypeError,
                            "type does not have compare function");
            # 返回 NULL 表示出错
            return NULL;
        }
    }

    # 检查并获取经过检查后的操作数组 op2，并设定轴参数
    op2 = (PyArrayObject *)PyArray_CheckAxis(op, &axis, 0);
    if (op2 == NULL) {
        # 如果 op2 为 NULL，表示检查失败，返回 NULL 表示出错
        return NULL;
    }

    # 对 ktharray 进行预处理，即使使用排序也要做边界检查
    kthrvl = partition_prep_kth_array(ktharray, op2, axis);
    if (kthrvl == NULL) {
        # 如果 kthrvl 为 NULL，表示预处理失败，释放 op2 并返回 NULL 表示出错
        Py_DECREF(op2);
        return NULL;
    }

    # 调用 _new_argsortlike 函数创建排序结果或分区结果
    ret = _new_argsortlike(op2, axis, argsort, argpart,
                           PyArray_DATA(kthrvl), PyArray_SIZE(kthrvl));

    # 释放 kthrvl 和 op2，它们已经不再需要
    Py_DECREF(kthrvl);
    Py_DECREF(op2);

    # 返回排序或分区的结果
    return ret;
/*NUMPY_API
 *LexSort an array providing indices that will sort a collection of arrays
 *lexicographically.  The first key is sorted on first, followed by the second key
 *-- requires that arg"merge"sort is available for each sort_key
 *
 *Returns an index array that shows the indexes for the lexicographic sort along
 *the given axis.
 */
NPY_NO_EXPORT PyObject *
PyArray_LexSort(PyObject *sort_keys, int axis)
{
    PyArrayObject **mps;    // 指向排序键的数组对象的指针数组
    PyArrayIterObject **its;    // 指向排序键的迭代器对象的指针数组
    PyArrayObject *ret = NULL;    // 返回的排序后的索引数组对象
    PyArrayIterObject *rit = NULL;    // 返回的排序后的索引数组的迭代器对象
    npy_intp n, N, size, i, j;    // 整数变量声明
    npy_intp astride, rstride, *iptr;    // 整数变量声明
    int nd;    // 数组的维数
    int needcopy = 0;    // 是否需要复制的标志位
    int elsize;    // 元素大小
    int maxelsize;    // 最大元素大小
    int object = 0;    // 是否包含对象数组的标志位
    PyArray_ArgSortFunc *argsort;    // 排序函数指针
    NPY_BEGIN_THREADS_DEF;    // 多线程宏定义的开始

    if (!PySequence_Check(sort_keys)    // 检查排序键是否是序列对象
           || ((n = PySequence_Size(sort_keys)) <= 0)) {    // 获取排序键的长度并检查是否大于0
        PyErr_SetString(PyExc_TypeError,
                "need sequence of keys with len > 0 in lexsort");    // 设置错误信息并返回空指针
        return NULL;
    }
    mps = (PyArrayObject **) PyArray_malloc(n * sizeof(PyArrayObject *));    // 分配排序键数组对象的指针数组内存
    if (mps == NULL) {    // 内存分配失败处理
        return PyErr_NoMemory();    // 返回内存错误信息
    }
    its = (PyArrayIterObject **) PyArray_malloc(n * sizeof(PyArrayIterObject *));    // 分配排序键迭代器对象的指针数组内存
    if (its == NULL) {    // 内存分配失败处理
        PyArray_free(mps);    // 释放之前分配的mps内存
        return PyErr_NoMemory();    // 返回内存错误信息
    }
    for (i = 0; i < n; i++) {    // 遍历排序键数目
        mps[i] = NULL;    // 初始化每个排序键数组对象指针为空
        its[i] = NULL;    // 初始化每个排序键迭代器对象指针为空
    }
    for (i = 0; i < n; i++) {    // 再次遍历排序键数目
        PyObject *obj;    // Python对象指针声明
        obj = PySequence_GetItem(sort_keys, i);    // 获取排序键序列中第i个对象
        if (obj == NULL) {    // 获取失败处理
            goto fail;    // 跳转到错误处理标签
        }
        mps[i] = (PyArrayObject *)PyArray_FROM_O(obj);    // 从Python对象创建排序键的数组对象
        Py_DECREF(obj);    // 减少对象的引用计数
        if (mps[i] == NULL) {    // 创建失败处理
            goto fail;    // 跳转到错误处理标签
        }
        if (i > 0) {    // 对于非第一个排序键
            if ((PyArray_NDIM(mps[i]) != PyArray_NDIM(mps[0]))    // 检查维度是否相同
                || (!PyArray_CompareLists(PyArray_DIMS(mps[i]),    // 检查维度列表是否相同
                                       PyArray_DIMS(mps[0]),
                                       PyArray_NDIM(mps[0])))) {
                PyErr_SetString(PyExc_ValueError,
                                "all keys need to be the same shape");    // 设置错误信息并返回空指针
                goto fail;    // 跳转到错误处理标签
            }
        }
        if (!PyDataType_GetArrFuncs(PyArray_DESCR(mps[i]))->argsort[NPY_STABLESORT]    // 检查是否支持稳定排序
                && !PyDataType_GetArrFuncs(PyArray_DESCR(mps[i]))->compare) {    // 检查是否有比较函数
            PyErr_Format(PyExc_TypeError,
                         "item %zd type does not have compare function", i);    // 设置错误信息并返回空指针
            goto fail;    // 跳转到错误处理标签
        }
        if (!object    // 如果不是对象数组
            && PyDataType_FLAGCHK(PyArray_DESCR(mps[i]), NPY_NEEDS_PYAPI)) {    // 检查是否需要Python API
            object = 1;    // 设置对象标志位
        }
    }

    /* Now we can check the axis */
    nd = PyArray_NDIM(mps[0]);    // 获取第一个排序键的维数
    /*
    * Special case letting axis={-1,0} slip through for scalars,
    * for backwards compatibility reasons.
    */
    if (nd == 0 && (axis == 0 || axis == -1)) {
        /* TODO: can we deprecate this? */
    }
    else if (check_and_adjust_axis(&axis, nd) < 0) {    // 检查并调整轴的有效性
        goto fail;    // 跳转到错误处理标签
    }
    if ((nd == 0) || (PyArray_SIZE(mps[0]) <= 1)) {
        /* empty/single element case */
        // 如果输入数组的维度为0或者第一个数组的元素数量小于等于1，则处理空数组或单元素情况

        // 创建一个新的数组对象，用于返回结果
        ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(mps[0]), PyArray_DIMS(mps[0]), NULL, NULL,
            0, NULL);

        // 检查数组对象是否创建成功
        if (ret == NULL) {
            goto fail;
        }

        // 如果第一个数组的元素数量大于0，将第一个元素设为0
        if (PyArray_SIZE(mps[0]) > 0) {
            *((npy_intp *)(PyArray_DATA(ret))) = 0;
        }

        // 跳转到完成处理的标签位置
        goto finish;
    }

    // 为每个输入数组创建迭代器对象
    for (i = 0; i < n; i++) {
        its[i] = (PyArrayIterObject *)PyArray_IterAllButAxis(
                (PyObject *)mps[i], &axis);
        // 检查迭代器对象是否创建成功
        if (its[i] == NULL) {
            goto fail;
        }
    }

    /* Now do the sorting */
    // 创建一个新的整数数组对象用于排序结果
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(mps[0]), PyArray_DIMS(mps[0]), NULL, NULL,
            0, NULL);
    if (ret == NULL) {
        goto fail;
    }

    // 创建一个迭代器对象用于排序结果
    rit = (PyArrayIterObject *)
            PyArray_IterAllButAxis((PyObject *)ret, &axis);
    if (rit == NULL) {
        goto fail;
    }

    // 如果不是对象数组，则开始线程处理
    if (!object) {
        NPY_BEGIN_THREADS;
    }

    // 初始化变量
    size = rit->size;
    N = PyArray_DIMS(mps[0])[axis];
    rstride = PyArray_STRIDE(ret, axis);
    maxelsize = PyArray_ITEMSIZE(mps[0]);
    needcopy = (rstride != sizeof(npy_intp));

    // 检查是否需要复制数据
    for (j = 0; j < n; j++) {
        needcopy = needcopy
            || PyArray_ISBYTESWAPPED(mps[j])
            || !(PyArray_FLAGS(mps[j]) & NPY_ARRAY_ALIGNED)
            || (PyArray_STRIDES(mps[j])[axis] != (npy_intp)PyArray_ITEMSIZE(mps[j]));

        // 更新最大元素大小
        if (PyArray_ITEMSIZE(mps[j]) > maxelsize) {
            maxelsize = PyArray_ITEMSIZE(mps[j]);
        }
    }
    if (needcopy) {
        // 如果需要进行复制操作，则进入此条件分支

        char *valbuffer, *indbuffer;
        int *swaps;

        // 确保 N 大于 0，这是由 indbuffer 保证的前提条件
        assert(N > 0);

        // 计算要分配的 valbuffer 的大小，并确保至少为 1，避免空的分配
        npy_intp valbufsize = N * maxelsize;
        if (NPY_UNLIKELY(valbufsize) == 0) {
            valbufsize = 1;
        }

        // 分配 valbuffer 内存空间
        valbuffer = PyDataMem_NEW(valbufsize);
        if (valbuffer == NULL) {
            // 分配失败时跳转到 fail 标签处处理
            goto fail;
        }

        // 分配 indbuffer 内存空间
        indbuffer = PyDataMem_NEW(N * sizeof(npy_intp));
        if (indbuffer == NULL) {
            // 分配失败时释放之前分配的 valbuffer，并跳转到 fail 标签处处理
            PyDataMem_FREE(valbuffer);
            goto fail;
        }

        // 分配 swaps 数组的内存空间
        swaps = malloc(NPY_LIKELY(n > 0) ? n * sizeof(int) : 1);
        if (swaps == NULL) {
            // 分配失败时释放之前分配的 valbuffer 和 indbuffer，并跳转到 fail 标签处处理
            PyDataMem_FREE(valbuffer);
            PyDataMem_FREE(indbuffer);
            goto fail;
        }

        // 对 swaps 数组进行初始化，判断是否需要字节交换
        for (j = 0; j < n; j++) {
            swaps[j] = PyArray_ISBYTESWAPPED(mps[j]);
        }

        // 处理每一个元素
        while (size--) {
            iptr = (npy_intp *)indbuffer;
            // 初始化 indbuffer 数组
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }

            // 对每一个数组进行排序操作
            for (j = 0; j < n; j++) {
                int rcode;
                elsize = PyArray_ITEMSIZE(mps[j]);
                astride = PyArray_STRIDES(mps[j])[axis];
                // 获取排序函数，如果未找到则使用默认的排序函数 npy_atimsort
                argsort = PyDataType_GetArrFuncs(PyArray_DESCR(mps[j]))->argsort[NPY_STABLESORT];
                if(argsort == NULL) {
                    argsort = npy_atimsort;
                }

                // 复制数据到 valbuffer，并进行可能的字节交换
                _unaligned_strided_byte_copy(valbuffer, (npy_intp) elsize,
                                             its[j]->dataptr, astride, N, elsize);
                if (swaps[j]) {
                    _strided_byte_swap(valbuffer, (npy_intp) elsize, N, elsize);
                }

                // 调用排序函数进行排序
                rcode = argsort(valbuffer, (npy_intp *)indbuffer, N, mps[j]);
                if (rcode < 0 || (PyDataType_REFCHK(PyArray_DESCR(mps[j]))
                            && PyErr_Occurred())) {
                    // 排序失败时释放所有内存，并跳转到 fail 标签处处理
                    PyDataMem_FREE(valbuffer);
                    PyDataMem_FREE(indbuffer);
                    free(swaps);
                    goto fail;
                }

                // 移动到下一个数组元素
                PyArray_ITER_NEXT(its[j]);
            }

            // 将排序后的索引数据复制到结果数组中
            _unaligned_strided_byte_copy(rit->dataptr, rstride, indbuffer,
                                         sizeof(npy_intp), N, sizeof(npy_intp));

            // 移动到结果数组的下一个位置
            PyArray_ITER_NEXT(rit);
        }

        // 完成所有操作后释放内存
        PyDataMem_FREE(valbuffer);
        PyDataMem_FREE(indbuffer);
        free(swaps);
    }
    else {
        while (size--) {
            // 获取当前迭代器的数据指针，并将其转换为整数指针
            iptr = (npy_intp *)rit->dataptr;
            // 对当前迭代器的数据指针进行赋值操作，从0到N-1
            for (i = 0; i < N; i++) {
                *iptr++ = i;
            }
            // 遍历mps数组，对每个元素进行排序操作
            for (j = 0; j < n; j++) {
                int rcode;
                // 获取排序函数，如果为NULL，则使用默认排序函数npy_atimsort
                argsort = PyDataType_GetArrFuncs(PyArray_DESCR(mps[j]))->argsort[NPY_STABLESORT];
                if(argsort == NULL) {
                    argsort = npy_atimsort;
                }
                // 调用排序函数进行排序
                rcode = argsort(its[j]->dataptr, (npy_intp *)rit->dataptr, N, mps[j]);
                // 检查排序操作是否成功，如果失败则跳转到fail标签处理
                if (rcode < 0 || (PyDataType_REFCHK(PyArray_DESCR(mps[j]))
                            && PyErr_Occurred())) {
                    goto fail;
                }
                // 移动到下一个迭代器
                PyArray_ITER_NEXT(its[j]);
            }
            // 移动到下一个迭代器
            PyArray_ITER_NEXT(rit);
        }
    }

    // 如果object为假值，结束多线程状态
    if (!object) {
        NPY_END_THREADS;
    }

 finish:
    // 释放mps和its数组的每个元素的引用计数
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    // 释放rit迭代器的引用计数
    Py_XDECREF(rit);
    // 释放mps和its数组的内存
    PyArray_free(mps);
    PyArray_free(its);
    // 返回ret对象
    return (PyObject *)ret;

 fail:
    // 失败处理：结束多线程状态
    NPY_END_THREADS;
    // 如果没有设置错误状态，则设置内存分配失败的错误状态
    if (!PyErr_Occurred()) {
        /* Out of memory during sorting or buffer creation */
        PyErr_NoMemory();
    }
    // 释放rit迭代器的引用计数
    Py_XDECREF(rit);
    // 释放ret对象的引用计数
    Py_XDECREF(ret);
    // 释放mps和its数组的每个元素的引用计数
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
        Py_XDECREF(its[i]);
    }
    // 释放mps和its数组的内存
    PyArray_free(mps);
    PyArray_free(its);
    // 返回空值
    return NULL;
/*NUMPY_API
 *
 * Search the sorted array op1 for the location of the items in op2. The
 * result is an array of indexes, one for each element in op2, such that if
 * the item were to be inserted in op1 just before that index the array
 * would still be in sorted order.
 *
 * Parameters
 * ----------
 * op1 : PyArrayObject *
 *     Array to be searched, must be 1-D.
 * op2 : PyObject *
 *     Array of items whose insertion indexes in op1 are wanted
 * side : {NPY_SEARCHLEFT, NPY_SEARCHRIGHT}
 *     If NPY_SEARCHLEFT, return first valid insertion indexes
 *     If NPY_SEARCHRIGHT, return last valid insertion indexes
 * perm : PyObject *
 *     Permutation array that sorts op1 (optional)
 *
 * Returns
 * -------
 * ret : PyObject *
 *   New reference to npy_intp array containing indexes where items in op2
 *   could be validly inserted into op1. NULL on error.
 *
 * Notes
 * -----
 * Binary search is used to find the indexes.
 */
NPY_NO_EXPORT PyObject *
PyArray_SearchSorted(PyArrayObject *op1, PyObject *op2,
                     NPY_SEARCHSIDE side, PyObject *perm)
{
    PyArrayObject *ap1 = NULL;
    PyArrayObject *ap2 = NULL;
    PyArrayObject *ap3 = NULL;
    PyArrayObject *sorter = NULL;
    PyArrayObject *ret = NULL;
    PyArray_Descr *dtype;
    int ap1_flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED;
    PyArray_BinSearchFunc *binsearch = NULL;
    PyArray_ArgBinSearchFunc *argbinsearch = NULL;
    NPY_BEGIN_THREADS_DEF;
    
    // 寻找与 op2 的共同类型
    dtype = PyArray_DescrFromObject((PyObject *)op2, PyArray_DESCR(op1));
    if (dtype == NULL) {
        return NULL;
    }
    /* refs to dtype we own = 1 */

    // 查找二分搜索函数
    if (perm) {
        argbinsearch = get_argbinsearch_func(dtype, side);
    }
    else {
        binsearch = get_binsearch_func(dtype, side);
    }
    if (binsearch == NULL && argbinsearch == NULL) {
        PyErr_SetString(PyExc_TypeError, "compare not supported for type");
        /* refs to dtype we own = 1 */
        Py_DECREF(dtype);
        /* refs to dtype we own = 0 */
        return NULL;
    }

    // 需要将 ap2 转换为连续数组并且是正确的类型
    /* refs to dtype we own = 1 */
    Py_INCREF(dtype);
    /* refs to dtype we own = 2 */
    ap2 = (PyArrayObject *)PyArray_CheckFromAny(op2, dtype,
                                0, 0,
                                NPY_ARRAY_CARRAY_RO | NPY_ARRAY_NOTSWAPPED,
                                NULL);
    /* refs to dtype we own = 1, array creation steals one even on failure */
    if (ap2 == NULL) {
        Py_DECREF(dtype);
        /* refs to dtype we own = 0 */
        return NULL;
    }

    /*
     * 如果要查找的元素 (ap2) 大于待查找的数组 (op1)，我们将待查找数组复制到一个连续的数组以提高缓存利用率。
     */
    if (PyArray_SIZE(ap2) > PyArray_SIZE(op1)) {
        ap1_flags |= NPY_ARRAY_CARRAY_RO;
    }
    ap1 = (PyArrayObject *)PyArray_CheckFromAny((PyObject *)op1, dtype,
                                1, 1, ap1_flags, NULL);
    /* 检查 op1 是否可以转换为指定的 NumPy 数组对象，要求一维数组，数据类型由 dtype 指定 */
    /* 当 ap1 为 NULL 时，跳转到错误处理部分 */
    if (ap1 == NULL) {
        goto fail;
    }

    if (perm) {
        /* 将 perm 转换为一维的、对齐的、未交换字节序的 NumPy 数组对象 */
        ap3 = (PyArrayObject *)PyArray_CheckFromAny(perm, NULL,
                                    1, 1,
                                    NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED,
                                    NULL);
        /* 当 ap3 为 NULL 时，设置类型错误异常，并跳转到错误处理部分 */
        if (ap3 == NULL) {
            PyErr_SetString(PyExc_TypeError,
                        "could not parse sorter argument");
            goto fail;
        }
        /* 当 ap3 不是整数类型的数组时，设置类型错误异常，并跳转到错误处理部分 */
        if (!PyArray_ISINTEGER(ap3)) {
            PyErr_SetString(PyExc_TypeError,
                        "sorter must only contain integers");
            goto fail;
        }
        /* 将 ap3 转换为已知的整数类型数组 */
        sorter = (PyArrayObject *)PyArray_FromArray(ap3,
                                    PyArray_DescrFromType(NPY_INTP),
                                    NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
        /* 当 sorter 为 NULL 时，设置数值错误异常，并跳转到错误处理部分 */
        if (sorter == NULL) {
            PyErr_SetString(PyExc_ValueError,
                        "could not parse sorter argument");
            goto fail;
        }
        /* 检查 sorter 的大小是否与 ap1 的大小相等，不相等则设置数值错误异常，并跳转到错误处理部分 */
        if (PyArray_SIZE(sorter) != PyArray_SIZE(ap1)) {
            PyErr_SetString(PyExc_ValueError,
                        "sorter.size must equal a.size");
            goto fail;
        }
    }

    /* 创建一个整数类型的连续数组 ret，用于存储返回的索引 */
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            PyArray_NDIM(ap2), PyArray_DIMS(ap2), NULL, NULL,
            0, (PyObject *)ap2);
    /* 当 ret 为 NULL 时，跳转到错误处理部分 */
    if (ret == NULL) {
        goto fail;
    }

    if (ap3 == NULL) {
        /* 执行常规的二分查找 */
        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
        binsearch((const char *)PyArray_DATA(ap1),
                  (const char *)PyArray_DATA(ap2),
                  (char *)PyArray_DATA(ret),
                  PyArray_SIZE(ap1), PyArray_SIZE(ap2),
                  PyArray_STRIDES(ap1)[0], PyArray_ITEMSIZE(ap2),
                  NPY_SIZEOF_INTP, ap2);
        NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
    }
    else {
        /* 使用排序数组进行二分查找 */
        // 定义错误变量
        int error = 0;
        // 开始线程安全操作，根据 ap2 的描述符
        NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
        // 调用二分查找函数 argbinsearch，处理 ap1、ap2、sorter 和 ret 的数据
        error = argbinsearch((const char *)PyArray_DATA(ap1),
                             (const char *)PyArray_DATA(ap2),
                             (const char *)PyArray_DATA(sorter),
                             (char *)PyArray_DATA(ret),
                             PyArray_SIZE(ap1), PyArray_SIZE(ap2),
                             PyArray_STRIDES(ap1)[0],
                             PyArray_ITEMSIZE(ap2),
                             PyArray_STRIDES(sorter)[0], NPY_SIZEOF_INTP, ap2);
        // 结束线程安全操作，根据 ap2 的描述符
        NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));
        // 如果二分查找出错，设置异常并跳转到 fail 标签
        if (error < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "Sorter index out of range.");
            goto fail;
        }
        // 释放对象引用：ap3 和 sorter
        Py_DECREF(ap3);
        Py_DECREF(sorter);
    }
    // 释放对象引用：ap1 和 ap2
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    // 返回 ret 对象的 PyObject 指针
    return (PyObject *)ret;

 fail:
    // 在发生失败时释放对象引用：ap1、ap2、ap3、sorter 和 ret
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ap3);
    Py_XDECREF(sorter);
    Py_XDECREF(ret);
    // 返回 NULL 指针表示函数执行失败
    return NULL;
/*NUMPY_API
 * Diagonal
 *
 * In NumPy versions prior to 1.7,  this function always returned a copy of
 * the diagonal array. In 1.7, the code has been updated to compute a view
 * onto 'self', but it still copies this array before returning, as well as
 * setting the internal WARN_ON_WRITE flag. In a future version, it will
 * simply return a view onto self.
 */
NPY_NO_EXPORT PyObject *
PyArray_Diagonal(PyArrayObject *self, int offset, int axis1, int axis2)
{
    int i, idim, ndim = PyArray_NDIM(self);
    npy_intp *strides;
    npy_intp stride1, stride2, offset_stride;
    npy_intp *shape, dim1, dim2;

    char *data;
    npy_intp diag_size;
    PyArray_Descr *dtype;
    PyObject *ret;
    npy_intp ret_shape[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];

    if (ndim < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "diag requires an array of at least two dimensions");
        return NULL;
    }

    /* Handle negative axes with standard Python indexing rules */
    if (check_and_adjust_axis_msg(&axis1, ndim, npy_interned_str.axis1) < 0) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(&axis2, ndim, npy_interned_str.axis2) < 0) {
        return NULL;
    }
    if (axis1 == axis2) {
        PyErr_SetString(PyExc_ValueError,
                    "axis1 and axis2 cannot be the same");
        return NULL;
    }

    /* Get the shape and strides of the two axes */
    shape = PyArray_SHAPE(self);
    dim1 = shape[axis1];
    dim2 = shape[axis2];
    strides = PyArray_STRIDES(self);
    stride1 = strides[axis1];
    stride2 = strides[axis2];

    /* Compute the data pointers and diag_size for the view */
    data = PyArray_DATA(self);
    if (offset >= 0) {
        offset_stride = stride2;
        dim2 -= offset;
    }
    else {
        offset = -offset;
        offset_stride = stride1;
        dim1 -= offset;
    }
    diag_size = dim2 < dim1 ? dim2 : dim1;
    if (diag_size < 0) {
        diag_size = 0;
    }
    else {
        data += offset * offset_stride;
    }

    /* Build the new shape and strides for the main data */
    i = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (idim != axis1 && idim != axis2) {
            ret_shape[i] = shape[idim];
            ret_strides[i] = strides[idim];
            ++i;
        }
    }
    ret_shape[ndim-2] = diag_size;
    ret_strides[ndim-2] = stride1 + stride2;

    /* Create the diagonal view */
    dtype = PyArray_DTYPE(self);
    Py_INCREF(dtype);
    ret = PyArray_NewFromDescrAndBase(
            Py_TYPE(self), dtype,
            ndim-1, ret_shape, ret_strides, data,
            PyArray_FLAGS(self), (PyObject *)self, (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }

    /*
     * For numpy 1.9 the diagonal view is not writeable.
     * This line needs to be removed in 1.10.
     */
    PyArray_CLEARFLAGS((PyArrayObject *)ret, NPY_ARRAY_WRITEABLE);

    return ret;
}
/* 压缩数组元素，将满足条件的元素压缩成一个数组，并返回结果 */
PyArray_Compress(PyArrayObject *self, PyObject *condition, int axis,
                 PyArrayObject *out)
{
    PyArrayObject *cond;  // 条件数组对象
    PyObject *res, *ret;  // 结果对象和返回对象

    if (PyArray_Check(condition)) {
        cond = (PyArrayObject *)condition;  // 如果条件是数组，则直接使用
        Py_INCREF(cond);  // 增加条件数组的引用计数
    }
    else {
        // 如果条件不是数组，则创建一个布尔类型的数组
        PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            return NULL;  // 如果创建描述符失败则返回空
        }
        cond = (PyArrayObject *)PyArray_FromAny(condition, dtype,
                                    0, 0, 0, NULL);  // 将条件转换为布尔类型数组
        if (cond == NULL) {
            return NULL;  // 如果转换失败则返回空
        }
    }

    if (PyArray_NDIM(cond) != 1) {
        Py_DECREF(cond);  // 如果条件数组维度不是1，则释放条件数组
        PyErr_SetString(PyExc_ValueError,
                        "condition must be a 1-d array");  // 抛出值错误异常
        return NULL;  // 返回空
    }

    res = PyArray_Nonzero(cond);  // 找出条件数组中非零元素的索引
    Py_DECREF(cond);  // 释放条件数组
    if (res == NULL) {
        return res;  // 如果结果为空，则直接返回空
    }
    ret = PyArray_TakeFrom(self, PyTuple_GET_ITEM(res, 0), axis,
                           out, NPY_RAISE);  // 从数组中按索引取出元素形成新数组
    Py_DECREF(res);  // 释放结果对象
    return ret;  // 返回结果对象
}

/*
 * 计算 48 字节块中非零字节的数量
 * w 必须按 8 字节对齐
 *
 * 即使它使用 64 位类型，它比 32 位平台上的逐字节求和更快
 * 但是在这些平台上，使用 32 位类型版本将使其更快
 */
#if !NPY_SIMD
static inline npy_intp
count_nonzero_bytes_384(const npy_uint64 * w)
{
    const npy_uint64 w1 = w[0];
    const npy_uint64 w2 = w[1];
    const npy_uint64 w3 = w[2];
    const npy_uint64 w4 = w[3];
    const npy_uint64 w5 = w[4];
    const npy_uint64 w6 = w[5];
    npy_intp r;

    /*
     * 最后部分的横向加法和 popcount，前三个二分可以跳过，因为我们正在处理字节。
     * 乘法等同于 (x + (x>>8) + (x>>16) + (x>>24)) & 0xFF
     * 无符号类型的乘法溢出在定义上是良好的。
     * w1 + w2 确保不会溢出，因为数据只有 0 和 1。
     */
    r = ((w1 + w2 + w3 + w4 + w5 + w6) * 0x0101010101010101ULL) >> 56ULL;

    /*
     * 字节不全为 0 或 1，则逐个求和。
     * 只有在视图或外部缓冲区中做了奇怪的操作时才会发生。
     * 在乐观计算之后执行此操作允许节省寄存器并实现更好的流水线处理。
     */
    if (NPY_UNLIKELY(
             ((w1 | w2 | w3 | w4 | w5 | w6) & 0xFEFEFEFEFEFEFEFEULL) != 0)) {
        /* 重新加载指针以避免与 gcc 的不必要的堆栈溢出 */
        const char * c = (const char *)w;
        npy_uintp i, count = 0;
        for (i = 0; i < 48; i++) {
            count += (c[i] != 0);  // 统计非零字节的数量
        }
        return count;  // 返回统计结果
    }

    return r;  // 返回快速计算的结果
}
#endif

#if NPY_SIMD
/* 计算 `*d` 和 `end` 之间的零字节数量，更新 `*d` 指向下一个要计算的位置 */
NPY_FINLINE NPY_GCC_OPT_3 npyv_u8
count_zero_bytes_u8(const npy_uint8 **d, const npy_uint8 *end, npy_uint8 max_count)
{
    const npyv_u8 vone = npyv_setall_u8(1);  // 创建所有元素为 1 的向量
    const npyv_u8 vzero = npyv_zero_u8();    // 创建所有元素为 0 的向量

    npy_intp lane_max = 0;  // 最大车道数
    # 初始化一个全零的无符号8位整数向量vsum8
    npyv_u8 vsum8 = npyv_zero_u8();
    # 当指针d指向的位置小于end，并且lane_max小于等于max_count - 1时，执行循环
    while (*d < end && lane_max <= max_count - 1) {
        # 使用npyv_load_u8从指针d处加载一个8位整数向量，并与全零向量vzero比较相等，生成一个布尔值向量
        npyv_u8 vt = npyv_cvt_u8_b8(npyv_cmpeq_u8(npyv_load_u8(*d), vzero));
        # 将布尔值向量vt与全一向量vone进行按位与操作，将布尔值转换为0或1
        vt = npyv_and_u8(vt, vone);
        # 将结果向量vt加到vsum8上，用于计算非零元素的数量
        vsum8 = npyv_add_u8(vsum8, vt);
        # 指针d向后移动npyv_nlanes_u8个字节（即向量中元素的数量）
        *d += npyv_nlanes_u8;
        # lane_max加1，用于记录当前处理的向量元素的索引
        lane_max += 1;
    }
    # 返回累加后的向量vsum8，其中每个元素代表对应位置上非零元素的个数
    return vsum8;
/*
 * Counts the number of non-zero values in a raw array of unsigned 16-bit integers.
 * Depending on SIMD availability, it uses vectorized operations for efficient counting.
 */
static inline NPY_GCC_OPT_3 npy_intp
count_nonzero_u16(const char *data, npy_intp bstride, npy_uintp len)
{
    npy_intp count = 0;
    // Check if SIMD (Single Instruction, Multiple Data) optimization is available
#if NPY_SIMD
    // If bstride is 1, perform SIMD operations for optimal counting
    if (bstride == 1) {
        npy_uintp len_m = len & -npyv_nlanes_u8;
        npy_uintp zcount = 0;
        // Process the data in chunks based on SIMD vectorization
        for (const char *end = data + len_m; data < end;) {
            // Count zero bytes using SIMD for 16-bit integers
            npyv_u16x2 vsum16 = count_zero_bytes_u16((const npy_uint8**)&data, (const npy_uint8*)end, NPY_MAX_UINT16);
            // Expand the 16-bit sums to 32-bit integers
            npyv_u32x2 sum_32_0 = npyv_expand_u32_u16(vsum16.val[0]);
            npyv_u32x2 sum_32_1 = npyv_expand_u32_u16(vsum16.val[1]);
            // Sum the 32-bit values to get the count of non-zero elements
            zcount += npyv_sum_u32(npyv_add_u32(
                    npyv_add_u32(sum_32_0.val[0], sum_32_0.val[1]),
                    npyv_add_u32(sum_32_1.val[0], sum_32_1.val[1])
            ));
        }
        // Adjust the remaining length after SIMD processing
        len  -= len_m;
        // Calculate the total count of non-zero elements
        count = len_m - zcount;
    } else {
        // If bstride is not 1 and SIMD is available, but the stride is not optimal,
        // fall back to non-SIMD approach if alignment conditions are not met
        if (!NPY_ALIGNMENT_REQUIRED || npy_is_aligned(data, sizeof(npy_uint64))) {
            // Define a step size to process data in chunks of 6 * sizeof(npy_uint64)
            int step = 6 * sizeof(npy_uint64);
            // Calculate the remaining bytes to process after the aligned chunks
            int left_bytes = len % step;
            // Process the aligned chunks using a specialized function
            for (const char *end = data + len; data < end - left_bytes; data += step) {
                 count += count_nonzero_bytes_384((const npy_uint64 *)data);
            }
            // Process the remaining bytes
            len = left_bytes;
        }
    }
    // If SIMD is not available, or bstride != 1 and alignment is required, use a fallback
#else
    // If SIMD is not available, or if bstride != 1 and alignment is required,
    // fall back to a sequential non-SIMD approach
#endif // NPY_SIMD
    // Count non-zero elements sequentially for the remaining data
    for (; len > 0; --len, data += bstride) {
        count += (*data != 0);
    }
    // Return the total count of non-zero elements in the array
    return count;
}
    # 检查每个元素的字节步长是否等于 np.uint16 的大小
    if (bstride == sizeof(npy_uint16)) {
        # 初始化零值变量和长度调整
        npy_uintp zcount = 0, len_m = len & -npyv_nlanes_u16;
        # 创建一个所有元素为 1 的 npyv_u16 向量
        const npyv_u16 vone  = npyv_setall_u16(1);
        # 创建一个所有元素为 0 的 npyv_u16 向量
        const npyv_u16 vzero = npyv_zero_u16();

        # 循环处理每个长度为 npyv_nlanes_u16 的片段
        for (npy_uintp lenx = len_m; lenx > 0;) {
            # 初始化一个全零 npyv_u16 向量用于累加
            npyv_u16 vsum16 = npyv_zero_u16();
            # 计算本轮最多能处理的元素数，考虑 SIMD 寄存器的宽度
            npy_uintp max16 = PyArray_MIN(lenx, NPY_MAX_UINT16*npyv_nlanes_u16);

            # 遍历数据，每次处理 npyv_nlanes_u16 个元素
            for (const char *end = data + max16*bstride; data < end; data += NPY_SIMD_WIDTH) {
                # 加载当前数据段的元素为 npyv_u16 向量，并检查是否为零
                npyv_u16 mask = npyv_cvt_u16_b16(npyv_cmpeq_u16(npyv_load_u16((npy_uint16*)data), vzero));
                # 将 mask 中所有元素与 vone 相与，得到非零元素的计数
                mask = npyv_and_u16(mask, vone);
                # 将非零计数累加到 vsum16 中
                vsum16 = npyv_add_u16(vsum16, mask);
            }
            # 更新剩余长度
            lenx   -= max16;
            # 累加本轮非零元素计数到 zcount
            zcount += npyv_sumup_u16(vsum16);
        }
        # 更新总长度和非零元素计数之差，即得到零元素的数量
        len  -= len_m;
        count = len_m - zcount;
    }
/*
 * 在条件编译指令结束后，开始定义函数count_nonzero_u32，用于计算非零元素个数
 */
static inline NPY_GCC_OPT_3 npy_intp
count_nonzero_u32(const char *data, npy_intp bstride, npy_uintp len)
{
    npy_intp count = 0;
#if NPY_SIMD
    // 如果步长等于4字节（即sizeof(npy_uint32)），则启用SIMD优化
    if (bstride == sizeof(npy_uint32)) {
        // 计算最大迭代次数
        const npy_uintp max_iter = NPY_MAX_UINT32 * npyv_nlanes_u32;
        // 计算实际处理的数据长度
        const npy_uintp len_m = (len > max_iter ? max_iter : len) & -npyv_nlanes_u32;
        // 创建SIMD向量，所有元素初始化为1和0
        const npyv_u32 vone   = npyv_setall_u32(1);
        const npyv_u32 vzero  = npyv_zero_u32();

        npyv_u32 vsum32 = npyv_zero_u32();
        // 使用SIMD进行循环，逐步处理数据
        for (const char *end = data + len_m * bstride; data < end; data += NPY_SIMD_WIDTH) {
            // 加载数据并进行比较，生成掩码
            npyv_u32 mask = npyv_cvt_u32_b32(npyv_cmpeq_u32(npyv_load_u32((npy_uint32*)data), vzero));
            // 掩码与全1向量进行与运算
            mask = npyv_and_u32(mask, vone);
            // 向量加法，累加掩码结果
            vsum32 = npyv_add_u32(vsum32, mask);
        }
        // 对奇偶向量进行处理，计算总的非零元素个数
        const npyv_u32 maskevn = npyv_reinterpret_u32_u64(npyv_setall_u64(0xffffffffULL));
        npyv_u64 odd  = npyv_shri_u64(npyv_reinterpret_u64_u32(vsum32), 32);
        npyv_u64 even = npyv_reinterpret_u64_u32(npyv_and_u32(vsum32, maskevn));
        count = len_m - npyv_sum_u64(npyv_add_u64(odd, even));
        // 更新剩余长度
        len  -= len_m;
    }
#endif
    // 普通循环，处理剩余数据
    for (; len > 0; --len, data += bstride) {
        count += (*(npy_uint32*)data != 0);
    }
    // 返回非零元素个数
    return count;
}

/*
 * 在条件编译指令结束后，开始定义函数count_nonzero_u64，用于计算非零元素个数
 */
static inline NPY_GCC_OPT_3 npy_intp
count_nonzero_u64(const char *data, npy_intp bstride, npy_uintp len)
{
    npy_intp count = 0;
#if NPY_SIMD
    // 如果步长等于8字节（即sizeof(npy_uint64)），则启用SIMD优化
    if (bstride == sizeof(npy_uint64)) {
        // 计算实际处理的数据长度
        const npy_uintp len_m = len & -npyv_nlanes_u64;
        // 创建SIMD向量，所有元素初始化为1和0
        const npyv_u64 vone   = npyv_setall_u64(1);
        const npyv_u64 vzero  = npyv_zero_u64();

        npyv_u64 vsum64 = npyv_zero_u64();
        // 使用SIMD进行循环，逐步处理数据
        for (const char *end = data + len_m * bstride; data < end; data += NPY_SIMD_WIDTH) {
            // 加载数据并进行比较，生成掩码
            npyv_u64 mask = npyv_cvt_u64_b64(npyv_cmpeq_u64(npyv_load_u64((npy_uint64*)data), vzero));
            // 掩码与全1向量进行与运算
            mask = npyv_and_u64(mask, vone);
            // 向量加法，累加掩码结果
            vsum64 = npyv_add_u64(vsum64, mask);
        }
        // 计算总的非零元素个数
        count = len_m - npyv_sum_u64(vsum64);
        // 更新剩余长度
        len  -= len_m;
    }
#endif
    // 普通循环，处理剩余数据
    for (; len > 0; --len, data += bstride) {
        count += (*(npy_uint64*)data != 0);
    }
    // 返回非零元素个数
    return count;
}

/*
 * 在函数定义之前添加注释，描述该函数的功能和返回值含义
 */
static NPY_GCC_OPT_3 npy_intp
count_nonzero_int(int ndim, char *data, const npy_intp *ashape, const npy_intp *astrides, int elsize)
{
    assert(elsize <= 8);
    int idim;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    // 使用原始迭代处理，无堆内存分配
    if (PyArray_PrepareOneRawArrayIter(
                    ndim, ashape,
                    data, astrides,
                    &ndim, shape,
                    &data, strides) < 0) {
        return -1;
    }
    // 处理长度为零的数组情况，如果数组第一个维度的长度为零，则直接返回计数为零
    if (shape[0] == 0) {
        return 0;
    }

    // 开始多线程操作的宏定义，根据条件决定是否开启多线程
    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS_THRESHOLDED(shape[0]);

    // 定义宏 NONZERO_CASE，根据元素大小不同进行不同的非零计数操作
    #define NONZERO_CASE(LEN, SFX) \
        case LEN: \
            // 使用原始迭代器开始迭代，遍历数组中的元素，计算非零元素的个数
            NPY_RAW_ITER_START(idim, ndim, coord, shape) { \
                count += count_nonzero_##SFX(data, strides[0], shape[0]); \
            } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides); \
            // 每个元素大小情况下的操作结束
            break

    // 初始化计数器 count
    npy_intp count = 0;
    // 根据元素大小 elsize 的不同，选择不同的 NONZERO_CASE 宏处理
    switch(elsize) {
        NONZERO_CASE(1, u8);
        NONZERO_CASE(2, u16);
        NONZERO_CASE(4, u32);
        NONZERO_CASE(8, u64);
    }
    // 取消 NONZERO_CASE 宏的定义
    #undef NONZERO_CASE

    // 结束多线程操作
    NPY_END_THREADS;
    // 返回计数值
    return count;
/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT NPY_GCC_OPT_3 npy_intp
count_boolean_trues(int ndim, char *data, npy_intp const *ashape, npy_intp const *astrides)
{
    // 使用 count_nonzero_int 函数计算布尔数组中的 True 值数量
    return count_nonzero_int(ndim, data, ashape, astrides, 1);
}

/*NUMPY_API
 * Counts the number of non-zero elements in the array.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
PyArray_CountNonzero(PyArrayObject *self)
{
    PyArray_NonzeroFunc *nonzero;
    char *data;
    npy_intp stride, count;
    npy_intp nonzero_count = 0;
    int needs_api = 0;
    PyArray_Descr *dtype;

    // 获取数组的数据类型描述符
    dtype = PyArray_DESCR(self);
    /* Special low-overhead version specific to the boolean/int types */
    // 如果数组对齐且是布尔或整数类型，则调用特定的低开销版本
    if (PyArray_ISALIGNED(self) && (
            PyDataType_ISBOOL(dtype) || PyDataType_ISINTEGER(dtype))) {
        return count_nonzero_int(
            PyArray_NDIM(self), PyArray_BYTES(self), PyArray_DIMS(self),
            PyArray_STRIDES(self), dtype->elsize
        );
    }

    // 获取非零元素计数的函数指针
    nonzero = PyDataType_GetArrFuncs(PyArray_DESCR(self))->nonzero;
    /* If it's a trivial one-dimensional loop, don't use an iterator */
    // 如果是简单的一维循环，则不使用迭代器
    if (PyArray_TRIVIALLY_ITERABLE(self)) {
        // 检查是否需要 Python API 支持
        needs_api = PyDataType_FLAGCHK(dtype, NPY_NEEDS_PYAPI);
        // 准备简单迭代器
        PyArray_PREPARE_TRIVIAL_ITERATION(self, count, data, stride);

        // 根据需要使用多线程处理
        if (needs_api) {
            while (count--) {
                // 调用非零元素判定函数，统计非零元素数量
                if (nonzero(data, self)) {
                    ++nonzero_count;
                }
                // 检查是否有 Python 异常发生
                if (PyErr_Occurred()) {
                    return -1;
                }
                // 移动数据指针到下一个元素
                data += stride;
            }
        } else {
            // 多线程处理非零元素判定
            NPY_BEGIN_THREADS_THRESHOLDED(count);
            while (count--) {
                if (nonzero(data, self)) {
                    ++nonzero_count;
                }
                data += stride;
            }
            NPY_END_THREADS;
        }

        // 返回统计到的非零元素数量
        return nonzero_count;
    }

    /*
     * If the array has size zero, return zero (the iterator rejects
     * size zero arrays)
     */
    // 如果数组大小为零，则直接返回零
    if (PyArray_SIZE(self) == 0) {
        return 0;
    }

    /*
     * Otherwise create and use an iterator to count the nonzeros.
     */
    // 创建迭代器来统计非零元素数量
    iter = NpyIter_New(self, NPY_ITER_READONLY |
                             NPY_ITER_EXTERNAL_LOOP |
                             NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return -1;
    }
    // 检查是否需要 Python API 支持
    needs_api = NpyIter_IterationNeedsAPI(iter);

    // 获取迭代器的下一步函数指针
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }

    // 开始多线程处理迭代器
    NPY_BEGIN_THREADS_NDITER(iter);
    # 获取指向数据指针数组的指针
    dataptr = NpyIter_GetDataPtrArray(iter);
    # 获取指向内部步长数组的指针
    strideptr = NpyIter_GetInnerStrideArray(iter);
    # 获取指向内部循环大小的指针
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    /* 遍历所有元素以计算非零元素数量 */
    do {
        # 获取当前数据指针的内容
        data = *dataptr;
        # 获取当前步长的内容
        stride = *strideptr;
        # 获取当前内部循环大小的内容
        count = *innersizeptr;

        # 遍历内部循环大小的次数
        while (count--) {
            # 如果当前数据非零，增加非零计数
            if (nonzero(data, self)) {
                ++nonzero_count;
            }
            # 如果需要 API 并且发生了异常，则设置非零计数为-1，并跳转到完成标签
            if (needs_api && PyErr_Occurred()) {
                nonzero_count = -1;
                goto finish;
            }
            # 数据指针根据步长增加
            data += stride;
        }

    } while(iternext(iter));
finish:
    NPY_END_THREADS;  // 结束线程

    NpyIter_Deallocate(iter);  // 释放迭代器资源

    return nonzero_count;  // 返回非零元素的数量
}

/*NUMPY_API
 * Nonzero
 *
 * TODO: In NumPy 2.0, should make the iteration order a parameter.
 */
NPY_NO_EXPORT PyObject *
PyArray_Nonzero(PyArrayObject *self)
{
    int i, ndim = PyArray_NDIM(self);  // 获取数组的维度数

    // 检查数组是否为零维，如果是，设置错误信息并返回 NULL
    if (ndim == 0) {
        char const* msg;
        if (PyArray_ISBOOL(self)) {
            msg =
                "Calling nonzero on 0d arrays is not allowed. "
                "Use np.atleast_1d(scalar).nonzero() instead. "
                "If the context of this error is of the form "
                "`arr[nonzero(cond)]`, just use `arr[cond]`.";
        } else {
            msg =
                "Calling nonzero on 0d arrays is not allowed. "
                "Use np.atleast_1d(scalar).nonzero() instead.";
        }
        PyErr_SetString(PyExc_ValueError, msg);  // 设置错误类型和消息
        return NULL;  // 返回 NULL 表示错误
    }

    PyArrayObject *ret = NULL;  // 结果数组对象指针
    PyObject *ret_tuple;  // 结果元组对象
    npy_intp ret_dims[2];  // 结果数组的维度

    PyArray_NonzeroFunc *nonzero;  // 非零元素查找函数指针
    PyArray_Descr *dtype;  // 数组的数据类型描述符

    npy_intp nonzero_count;  // 非零元素的数量
    npy_intp added_count = 0;  // 已添加的计数
    int needs_api;  // 是否需要 Python API
    int is_bool;  // 数组是否为布尔类型

    NpyIter *iter;  // 迭代器对象指针
    NpyIter_IterNextFunc *iternext;  // 迭代器下一步函数指针
    NpyIter_GetMultiIndexFunc *get_multi_index;  // 获取多重索引函数指针
    char **dataptr;  // 数据指针数组

    dtype = PyArray_DESCR(self);  // 获取数组的数据类型描述符
    nonzero = PyDataType_GetArrFuncs(dtype)->nonzero;  // 获取非零元素查找函数
    needs_api = PyDataType_FLAGCHK(dtype, NPY_NEEDS_PYAPI);  // 检查是否需要 Python API

    /*
     * First count the number of non-zeros in 'self'.
     */
    nonzero_count = PyArray_CountNonzero(self);  // 计算数组中非零元素的数量
    if (nonzero_count < 0) {
        return NULL;  // 如果计算出错，返回 NULL
    }

    is_bool = PyArray_ISBOOL(self);  // 检查数组是否为布尔类型

    /* Allocate the result as a 2D array */
    ret_dims[0] = nonzero_count;  // 第一维度是非零元素的数量
    ret_dims[1] = ndim;  // 第二维度是数组的维度数
    ret = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            2, ret_dims, NULL, NULL,
            0, NULL);  // 从描述符创建一个新的数组对象作为结果
    if (ret == NULL) {
        return NULL;  // 如果创建失败，返回 NULL
    }

    /* If it's a one-dimensional result, don't use an iterator */
    # 如果数组是一维的情况下
    if (ndim == 1) {
        # 获取返回数组的多维索引指针
        npy_intp * multi_index = (npy_intp *)PyArray_DATA(ret);
        # 获取当前数组的字节数据指针
        char * data = PyArray_BYTES(self);
        # 获取当前数组在第一个轴上的步长
        npy_intp stride = PyArray_STRIDE(self, 0);
        # 获取当前数组在第一个轴上的大小
        npy_intp count = PyArray_DIM(self, 0);
        # 定义多线程开始宏（未展开）
        NPY_BEGIN_THREADS_DEF;

        /* 无需处理 */
        # 如果非零元素个数为0，直接跳转到结束标签
        if (nonzero_count == 0) {
            goto finish;
        }

        # 如果不需要 GIL（全局解释器锁）
        if (!needs_api) {
            # 使用阈值条件开始多线程操作
            NPY_BEGIN_THREADS_THRESHOLDED(count);
        }

        /* 针对布尔类型避免函数调用 */
        # 如果是布尔类型数组
        if (is_bool) {
            /*
             * 对于稀疏数据，使用快速的 memchr 变体，参见 gh-4370
             * 在这种稀疏路径后的快速布尔计数比结合两个循环更快，即使对于更大的数组也是如此
             */
            # 如果非零元素比例小于等于0.1
            if (((double)nonzero_count / count) <= 0.1) {
                # 定义子大小和循环索引 j
                npy_intp subsize;
                npy_intp j = 0;
                # 循环直到结束条件
                while (1) {
                    # 使用快速 memchr 函数查找零元素位置
                    npy_memchr(data + j * stride, 0, stride, count - j,
                               &subsize, 1);
                    # 更新 j
                    j += subsize;
                    # 如果 j 超过数组大小，则退出循环
                    if (j >= count) {
                        break;
                    }
                    # 将 j 添加到多维索引中并更新 j
                    *multi_index++ = j++;
                }
            }
            /*
             * 为了避免分支预测错误导致的性能下降，在这里使用无分支策略
             */
            else {
                # 定义多维索引结束位置
                npy_intp *multi_index_end = multi_index + nonzero_count;
                npy_intp j = 0;

                /* 手动展开循环以便于 GCC 和可能的其他编译器 */
                # 循环展开，每次处理四个元素
                while (multi_index + 4 < multi_index_end) {
                    # 设置索引值并根据 data[0] 的值更新 multi_index
                    *multi_index = j;
                    multi_index += data[0] != 0;
                    # 设置索引值并根据 data[stride] 的值更新 multi_index
                    *multi_index = j + 1;
                    multi_index += data[stride] != 0;
                    # 设置索引值并根据 data[stride * 2] 的值更新 multi_index
                    *multi_index = j + 2;
                    multi_index += data[stride * 2] != 0;
                    # 设置索引值并根据 data[stride * 3] 的值更新 multi_index
                    *multi_index = j + 3;
                    multi_index += data[stride * 3] != 0;
                    # 更新 data 指针
                    data += stride * 4;
                    # 更新 j
                    j += 4;
                }

                # 处理剩余的索引
                while (multi_index < multi_index_end) {
                    # 设置索引值并根据 data 的值更新 multi_index
                    *multi_index = j;
                    multi_index += *data != 0;
                    # 更新 data 指针
                    data += stride;
                    # 更新 j
                    ++j;
                }
            }
        }
        else {
            # 处理一般情况下的索引设置
            npy_intp j;
            for (j = 0; j < count; ++j) {
                # 如果当前元素非零，则将索引 j 添加到 multi_index 中
                if (nonzero(data, self)) {
                    # 如果添加的索引超过了非零元素的数量，则退出循环
                    if (++added_count > nonzero_count) {
                        break;
                    }
                    # 将索引 j 添加到 multi_index 中
                    *multi_index++ = j;
                }
                # 如果需要 API 并且出现了异常，则中断循环
                if (needs_api && PyErr_Occurred()) {
                    break;
                }
                # 更新 data 指针
                data += stride;
            }
        }

        # 结束多线程操作
        NPY_END_THREADS;

        # 跳转到结束标签
        goto finish;
    }

    /*
     * 以 C 顺序构建一个迭代器来跟踪多维索引。
     */
    iter = NpyIter_New(self, NPY_ITER_READONLY |
                             NPY_ITER_MULTI_INDEX |
                             NPY_ITER_ZEROSIZE_OK |
                             NPY_ITER_REFS_OK,
                        NPY_CORDER, NPY_NO_CASTING,
                        NULL);

创建一个 Numpy 迭代器对象 `iter`，用于遍历数组元素。


    if (iter == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

检查迭代器是否成功创建，如果创建失败，则释放之前分配的资源并返回空指针。


    if (NpyIter_GetIterSize(iter) != 0) {

检查迭代器中的元素数量是否不为零，即数组不为空。


        npy_intp * multi_index;
        NPY_BEGIN_THREADS_DEF;

声明多索引数组 `multi_index` 和 Numpy 线程宏 `NPY_BEGIN_THREADS_DEF`。


        /* Get the pointers for inner loop iteration */
        iternext = NpyIter_GetIterNext(iter, NULL);

获取迭代器的下一个迭代函数 `iternext`，用于迭代内部循环。


        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(ret);
            return NULL;
        }

如果获取迭代函数失败，则释放迭代器资源、释放之前分配的 Python 对象并返回空指针。


        get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);

获取获取多索引函数 `get_multi_index`，用于获取多维数组中元素的索引。


        if (get_multi_index == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(ret);
            return NULL;
        }

如果获取获取多索引函数失败，则释放迭代器资源、释放之前分配的 Python 对象并返回空指针。


        needs_api = NpyIter_IterationNeedsAPI(iter);

检查迭代器是否需要 Python API 支持，并将结果存储在 `needs_api` 中。


        NPY_BEGIN_THREADS_NDITER(iter);

开始 Numpy 线程化迭代器循环。


        dataptr = NpyIter_GetDataPtrArray(iter);

获取数据指针数组 `dataptr`，用于访问数组元素数据。


        multi_index = (npy_intp *)PyArray_DATA(ret);

将返回对象 `ret` 中的数据指针类型转换为 `npy_intp*` 类型，并赋值给 `multi_index`。


        /* Get the multi-index for each non-zero element */
        if (is_bool) {
            /* avoid function call for bool */
            do {
                if (**dataptr != 0) {
                    get_multi_index(iter, multi_index);
                    multi_index += ndim;
                }
            } while(iternext(iter));
        }
        else {
            do {
                if (nonzero(*dataptr, self)) {
                    if (++added_count > nonzero_count) {
                        break;
                    }
                    get_multi_index(iter, multi_index);
                    multi_index += ndim;
                }
                if (needs_api && PyErr_Occurred()) {
                    break;
                }
            } while(iternext(iter));
        }

根据数组元素的情况，获取每个非零元素的多索引：
- 如果数组是布尔类型，则直接检查元素值是否不为零，然后获取其多索引。
- 否则，调用 `nonzero` 函数检查元素是否非零，并根据需要获取多索引。同时检查是否需要 Python API 支持和是否发生了异常。


        NPY_END_THREADS;

结束 Numpy 线程化迭代器循环。


    }

    NpyIter_Deallocate(iter);

循环结束后，释放 Numpy 迭代器资源。
finish:
    // 检查是否有 Python 异常发生，如果有则清理返回空并释放之前创建的对象
    if (PyErr_Occurred()) {
        Py_DECREF(ret);
        return NULL;
    }

    /* if executed `nonzero()` check for miscount due to side-effect */
    // 如果执行了 `nonzero()` 函数检查由于副作用导致的计数错误
    if (!is_bool && added_count != nonzero_count) {
        PyErr_SetString(PyExc_RuntimeError,
            "number of non-zero array elements "
            "changed during function execution.");
        Py_DECREF(ret);
        return NULL;
    }

    // 创建一个包含 ndim 个元素的元组对象 ret_tuple
    ret_tuple = PyTuple_New(ndim);
    // 如果创建元组对象失败则清理返回空并释放之前创建的对象
    if (ret_tuple == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    /* Create views into ret, one for each dimension */
    // 为 ret 中的每个维度创建视图
    for (i = 0; i < ndim; ++i) {
        npy_intp stride = ndim * NPY_SIZEOF_INTP;
        /* the result is an empty array, the view must point to valid memory */
        // 如果结果是一个空数组，视图必须指向有效的内存
        npy_intp data_offset = nonzero_count == 0 ? 0 : i * NPY_SIZEOF_INTP;

        // 创建一个 PyArrayObject 类型的视图对象 view
        PyArrayObject *view = (PyArrayObject *)PyArray_NewFromDescrAndBase(
            Py_TYPE(ret), PyArray_DescrFromType(NPY_INTP),
            1, &nonzero_count, &stride, PyArray_BYTES(ret) + data_offset,
            PyArray_FLAGS(ret), (PyObject *)ret, (PyObject *)ret);
        // 如果创建视图对象失败则清理返回空并释放之前创建的对象
        if (view == NULL) {
            Py_DECREF(ret);
            Py_DECREF(ret_tuple);
            return NULL;
        }
        // 将视图对象 view 添加到元组 ret_tuple 的第 i 个位置
        PyTuple_SET_ITEM(ret_tuple, i, (PyObject *)view);
    }
    // 清理返回对象 ret，因为其引用已经被传递给视图对象
    Py_DECREF(ret);

    // 返回包含视图对象的元组 ret_tuple
    return ret_tuple;
}

/*
 * Gets a single item from the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 */
NPY_NO_EXPORT PyObject *
PyArray_MultiIndexGetItem(PyArrayObject *self, const npy_intp *multi_index)
{
    int idim, ndim = PyArray_NDIM(self);
    char *data = PyArray_DATA(self);
    npy_intp *shape = PyArray_SHAPE(self);
    npy_intp *strides = PyArray_STRIDES(self);

    /* Get the data pointer */
    // 获取数据指针，根据多重索引 multi_index 访问数组元素
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp shapevalue = shape[idim];
        npy_intp ind = multi_index[idim];

        // 检查并调整索引 ind，确保在有效范围内
        if (check_and_adjust_index(&ind, shapevalue, idim, NULL) < 0) {
            return NULL;
        }
        // 根据索引计算数据指针的偏移量
        data += ind * strides[idim];
    }

    // 返回数组中指定位置的元素对象
    return PyArray_GETITEM(self, data);
}

/*
 * Sets a single item in the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_MultiIndexSetItem(PyArrayObject *self, const npy_intp *multi_index,
                                                PyObject *obj)
{
    int idim, ndim = PyArray_NDIM(self);
    char *data = PyArray_DATA(self);
    npy_intp *shape = PyArray_SHAPE(self);
    npy_intp *strides = PyArray_STRIDES(self);

    /* Get the data pointer */
    // 获取数据指针，根据多重索引 multi_index 访问数组元素
    for (idim = 0; idim < ndim; ++idim) {
        npy_intp shapevalue = shape[idim];
        npy_intp ind = multi_index[idim];

        // 检查并调整索引 ind，确保在有效范围内
        if (check_and_adjust_index(&ind, shapevalue, idim, NULL) < 0) {
            return -1;
        }
        // 根据索引计算数据指针的偏移量
        data += ind * strides[idim];
    }

    // 将对象 obj 设置到数组中指定位置，并返回操作结果
    return PyArray_Pack(PyArray_DESCR(self), data, obj);
}
```