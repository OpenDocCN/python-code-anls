# `.\numpy\numpy\_core\src\multiarray\arrayobject.c`

```py
/*
  提供多维数组作为Python的基本对象类型。

  基于原始的Numeric实现
  版权所有 (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  还包括了来自1995-2004年间许多Numeric Python开发者的贡献

  2005年进行了大幅修改，受到Numarray的启发

  作者：
  Travis Oliphant,  oliphant@ee.byu.edu
  布里格姆扬大学

  维护者邮箱： oliphant.travis@ieee.org

  Numarray的设计（提供了指导）由
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "common.h"

#include "number.h"
#include "usertypes.h"
#include "arraywrap.h"
#include "arraytypes.h"
#include "scalartypes.h"
#include "arrayobject.h"
#include "convert_datatype.h"
#include "conversion_utils.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "methods.h"
#include "descriptor.h"
#include "iterators.h"
#include "mapping.h"
#include "getset.h"
#include "sequence.h"
#include "npy_buffer.h"
#include "array_assign.h"
#include "alloc.h"
#include "mem_overlap.h"
#include "numpyos.h"
#include "refcount.h"
#include "strfuncs.h"

#include "binop_override.h"
#include "array_coercion.h"
#include "multiarraymodule.h"

/*NUMPY_API
  计算数组的大小（以项数表示）
*/
NPY_NO_EXPORT npy_intp
PyArray_Size(PyObject *op)
{
    if (PyArray_Check(op)) {
        return PyArray_SIZE((PyArrayObject *)op);
    }
    else {
        return 0;
    }
}

/*NUMPY_API */
NPY_NO_EXPORT int
PyArray_SetUpdateIfCopyBase(PyArrayObject *arr, PyArrayObject *base)
{
    /* 2021-Dec-15 1.23*/
    PyErr_SetString(PyExc_RuntimeError,
        "PyArray_SetUpdateIfCopyBase is disabled, use "
        "PyArray_SetWritebackIfCopyBase instead, and be sure to call "
        "PyArray_ResolveWritebackIfCopy before the array is deallocated, "
        "i.e. before the last call to Py_DECREF. If cleaning up from an "
        "error, PyArray_DiscardWritebackIfCopy may be called instead to "
        "throw away the scratch buffer.");
    return -1;
}

/*NUMPY_API
 *
 * 前提条件：'arr' 是 'base' 的一个副本（尽管可能具有不同的步幅、顺序等）。
 * 此函数设置WRITEBACKIFCOPY标志和->base指针在'arr'上，
 * 调用PyArray_ResolveWritebackIfCopy在释放数组之前将任何更改复制回'base'。
 *
 * 引用计数 'base'。
 *
 * 成功返回0，失败返回-1。
 */
NPY_NO_EXPORT int
PyArray_SetWritebackIfCopyBase(PyArrayObject *arr, PyArrayObject *base)
{
    if (base == NULL) {
        PyErr_SetString(PyExc_ValueError,
                  "Cannot WRITEBACKIFCOPY to NULL array");
        return -1;
    }
    # 检查是否已经有基础对象关联到数组，若有则报错并跳转到失败处理
    if (PyArray_BASE(arr) != NULL) {
        PyErr_SetString(PyExc_ValueError,
                  "Cannot set array with existing base to WRITEBACKIFCOPY");
        goto fail;
    }
    # 检查基础对象是否可写，若不可写则跳转到失败处理
    if (PyArray_FailUnlessWriteable(base, "WRITEBACKIFCOPY base") < 0) {
        goto fail;
    }

    /*
     * Any writes to 'arr' will magically turn into writes to 'base', so we
     * should warn if necessary.
     */
    # 如果基础对象设置了 NPY_ARRAY_WARN_ON_WRITE 标志，则将此标志设置到 arr 上
    if (PyArray_FLAGS(base) & NPY_ARRAY_WARN_ON_WRITE) {
        PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WARN_ON_WRITE);
    }

    /*
     * Unlike PyArray_SetBaseObject, we do not compress the chain of base
     * references.
     */
    # 将 arr 的 base 属性设置为 base 对象，并启用 NPY_ARRAY_WRITEBACKIFCOPY 标志
    ((PyArrayObject_fields *)arr)->base = (PyObject *)base;
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WRITEBACKIFCOPY);
    # 清除 base 对象的 NPY_ARRAY_WRITEABLE 标志
    PyArray_CLEARFLAGS(base, NPY_ARRAY_WRITEABLE);

    # 操作成功，返回 0
    return 0;

  fail:
    # 失败处理：释放 base 对象的引用计数并返回 -1
    Py_DECREF(base);
    return -1;
/*NUMPY_API
 * Sets the 'base' attribute of the array. This steals a reference
 * to 'obj'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_SetBaseObject(PyArrayObject *arr, PyObject *obj)
{
    // 如果传入的对象是空指针，抛出数值错误异常并返回 -1
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency to NULL after initialization");
        return -1;
    }
    
    /*
     * 允许仅设置一次基础对象。一旦设置了拥有数据的对象，再次更改就不合理了。
     */
    if (PyArray_BASE(arr) != NULL) {
        // 引用计数减一，清理对象，并抛出数值错误异常
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency more than once");
        return -1;
    }

    /*
     * 不允许视图之间无限的链式依赖，始终将基础对象设置为数据的第一个拥有者。
     * 即，第一个不是数组的对象，或者第一个拥有自己数据的对象。
     */
    while (PyArray_Check(obj) && (PyObject *)arr != obj) {
        PyArrayObject *obj_arr = (PyArrayObject *)obj;
        PyObject *tmp;

        /* 如果视图具有 NPY_ARRAY_WARN_ON_WRITE 标志，传播该警告 */
        if (PyArray_FLAGS(obj_arr) & NPY_ARRAY_WARN_ON_WRITE) {
            PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WARN_ON_WRITE);
        }

        /* 如果此数组拥有自己的数据，则停止合并 */
        if (PyArray_CHKFLAGS(obj_arr, NPY_ARRAY_OWNDATA)) {
            break;
        }

        tmp = PyArray_BASE(obj_arr);
        /* 如果没有基础对象，则停止合并 */
        if (tmp == NULL) {
            break;
        }
        /* 当新基础对象的类型不同时停止合并（即，不同的子类） */
        if (Py_TYPE(tmp) != Py_TYPE(arr)) {
            break;
        }

        Py_INCREF(tmp);
        Py_DECREF(obj);
        obj = tmp;
    }

    /* 禁止循环引用 */
    if ((PyObject *)arr == obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError,
                "Cannot create a circular NumPy array 'base' dependency");
        return -1;
    }

    // 设置数组对象的 base 属性为 obj
    ((PyArrayObject_fields *)arr)->base = obj;

    return 0;
}

/**
 * Assign an arbitrary object a NumPy array. This is largely basically
 * identical to PyArray_FromAny, but assigns directly to the output array.
 *
 * @param dest Array to be written to
 * @param src_object Object to be assigned, array-coercion rules apply.
 * @return 0 on success -1 on failures.
 */
/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_CopyObject(PyArrayObject *dest, PyObject *src_object)
{
    int ret = 0;
    PyArrayObject *view;
    PyArray_Descr *dtype = NULL;
    int ndim;
    npy_intp dims[NPY_MAXDIMS];
    coercion_cache_obj *cache = NULL;

    /*
     * We have to set the maximum number of dimensions here to support
     * sequences within object arrays.
     */




    /*
     * 在这里设置最大的维度数以支持对象数组中的序列。
     */
    # 调用 PyArray_DiscoverDTypeAndShape 函数，用于发现源对象的数据类型和形状，并根据目标数组的维度指定形状和缓存
    ndim = PyArray_DiscoverDTypeAndShape(src_object,
            PyArray_NDIM(dest), dims, &cache,
            NPY_DTYPE(PyArray_DESCR(dest)), PyArray_DESCR(dest), &dtype, 1, NULL);
    # 如果返回值小于 0，表示出现错误，直接返回 -1
    if (ndim < 0) {
        return -1;
    }

    # 如果缓存不为空且不是序列，则直接分配给视图
    if (cache != NULL && !(cache->sequence)) {
        /* 输入是数组或数组对象，因此可以直接赋值 */
        assert(cache->converted_obj == src_object);
        view = (PyArrayObject *)cache->arr_or_sequence;
        Py_DECREF(dtype);
        # 使用 PyArray_AssignArray 函数将视图数组赋值给目标数组
        ret = PyArray_AssignArray(dest, view, NULL, NPY_UNSAFE_CASTING);
        npy_free_coercion_cache(cache);
        return ret;
    }

    /*
     * 可能需要广播，因为形状不匹配，此时先创建临时数组，并填充后再赋值
     */
    if (ndim != PyArray_NDIM(dest) ||
            !PyArray_CompareLists(PyArray_DIMS(dest), dims, ndim)) {
        /*
         * 可能需要广播，因此先分配给一个视图。
         * 这种情况可能会导致后面的形状不匹配错误。
         */
        assert (ndim <= PyArray_NDIM(dest));  /* 在发现期间可能会出错 */
        # 使用 PyArray_NewFromDescr 函数创建一个新的数组对象作为视图
        view = (PyArrayObject *) PyArray_NewFromDescr(
                &PyArray_Type, dtype, ndim, dims, NULL, NULL,
                PyArray_FLAGS(dest) & NPY_ARRAY_F_CONTIGUOUS, NULL);
        # 如果创建视图失败，释放缓存并返回 -1
        if (view == NULL) {
            npy_free_coercion_cache(cache);
            return -1;
        }
    }
    else {
        # 如果形状匹配，直接使用目标数组作为视图
        Py_DECREF(dtype);
        view = dest;
    }

    # 将值分配给 `view`（无论是哪个数组）
    if (cache == NULL) {
        /* 单个（非数组）项，立即赋值 */
        if (PyArray_Pack(
                PyArray_DESCR(view), PyArray_DATA(view), src_object) < 0) {
            goto fail;
        }
    }
    else {
        # 使用 PyArray_AssignFromCache 函数将缓存中的值分配给视图
        if (PyArray_AssignFromCache(view, cache) < 0) {
            goto fail;
        }
    }
    # 如果视图与目标数组相同，直接返回 0
    if (view == dest) {
        return 0;
    }
    # 将视图数组赋值给目标数组，使用不安全的转换方式（NPY_UNSAFE_CASTING）
    ret = PyArray_AssignArray(dest, view, NULL, NPY_UNSAFE_CASTING);
    # 减少视图的引用计数
    Py_DECREF(view);
    return ret;

  fail:
    # 如果视图与目标数组不同，减少视图的引用计数
    if (view != dest) {
        Py_DECREF(view);
    }
    # 返回 -1 表示失败
    return -1;
/*NUMPY_API
 *
 * If WRITEBACKIFCOPY and self has data, reset the base WRITEABLE flag,
 * copy the local data to base, release the local data, and set flags
 * appropriately. Return 0 if not relevant, 1 if success, < 0 on failure
 */
NPY_NO_EXPORT int
PyArray_ResolveWritebackIfCopy(PyArrayObject * self)
{
    /* 将输入的 PyArrayObject 转换为 PyArrayObject_fields 类型 */
    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;
    
    /* 检查 fa 和 fa->base 是否存在 */
    if (fa && fa->base) {
        /* 检查是否设置了 NPY_ARRAY_WRITEBACKIFCOPY 标志 */
        if (fa->flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            /*
             * WRITEBACKIFCOPY 意味着 fa->base 的数据
             * 应该更新为 self 的内容。
             * fa->base->flags 不是 WRITEABLE 以保护这种关系，
             * 解锁它。
             */
            int retval = 0;
            /* 启用 fa->base 的 WRITEABLE 标志 */
            PyArray_ENABLEFLAGS(((PyArrayObject *)fa->base),
                                NPY_ARRAY_WRITEABLE);
            /* 清除 self 的 NPY_ARRAY_WRITEBACKIFCOPY 标志 */
            PyArray_CLEARFLAGS(self, NPY_ARRAY_WRITEBACKIFCOPY);
            /* 将 self 的数据复制到 fa->base */
            retval = PyArray_CopyAnyInto((PyArrayObject *)fa->base, self);
            /* 释放 fa->base 的引用 */
            Py_DECREF(fa->base);
            fa->base = NULL;
            /* 如果复制失败，返回负值 */
            if (retval < 0) {
                /* 这不应该发生，两份数据怎么会不同步？ */
                return retval;
            }
            /* 成功复制数据，返回 1 */
            return 1;
        }
    }
    /* 没有满足条件的情况，返回 0 */
    return 0;
}

/*********************** end C-API functions **********************/

/* dealloc must not raise an error, best effort try to write
   to stderr and clear the error
*/

/* 在 dealloc 中不应该引发错误，尽力尝试写入 stderr 并清除错误 */

static inline void
WARN_IN_DEALLOC(PyObject* warning, const char * msg) {
    /* 如果发出警告时出错 */
    if (PyErr_WarnEx(warning, msg, 1) < 0) {
        PyObject * s;

        /* 创建字符串 "array_dealloc" */
        s = PyUnicode_FromString("array_dealloc");
        /* 如果成功创建字符串 */
        if (s) {
            /* 写入无法处理的异常 */
            PyErr_WriteUnraisable(s);
            /* 释放字符串对象的引用 */
            Py_DECREF(s);
        }
        else {
            /* 如果创建字符串失败，写入无法处理的异常 */
            PyErr_WriteUnraisable(Py_None);
        }
    }
}

/* array object functions */

/* 数组对象的函数 */

static void
array_dealloc(PyArrayObject *self)
{
    /* 将输入的 PyArrayObject 转换为 PyArrayObject_fields 类型 */
    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;

    /* 如果释放缓冲区信息时出错 */
    if (_buffer_info_free(fa->_buffer_info, (PyObject *)self) < 0) {
        /* 写入无法处理的异常 */
        PyErr_WriteUnraisable(NULL);
    }

    /* 清除弱引用列表 */
    if (fa->weakreflist != NULL) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
}
    # 如果数组对象的 `base` 属性存在
    if (fa->base) {
        int retval;
        # 如果数组对象标记中包含 `NPY_ARRAY_WRITEBACKIFCOPY` 标志
        if (PyArray_FLAGS(self) & NPY_ARRAY_WRITEBACKIFCOPY)
        {
            # 提示消息
            char const * msg = "WRITEBACKIFCOPY detected in array_dealloc. "
                " Required call to PyArray_ResolveWritebackIfCopy or "
                "PyArray_DiscardWritebackIfCopy is missing.";
            /*
             * 防止两次达到引用计数为0，从而递归进入dealloc。
             * 增加 sys.gettotalrefcount，但实际路径不应被执行。
             */
            # 增加引用计数，发出运行时警告
            Py_INCREF(self);
            WARN_IN_DEALLOC(PyExc_RuntimeWarning, msg);
            # 解决 WRITEBACKIFCOPY 问题
            retval = PyArray_ResolveWritebackIfCopy(self);
            if (retval < 0)
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        /*
         * 如果 fa->base 非空，则需要 DECREF 它 —— 可能是视图或缓冲区对象
         */
        # DECREF fa->base
        Py_XDECREF(fa->base);
    }

    # 如果数组对象标记中包含 `NPY_ARRAY_OWNDATA` 并且 `fa->data` 非空
    if ((fa->flags & NPY_ARRAY_OWNDATA) && fa->data) {
        /* 释放任何内部引用 */
        # 如果数组描述符中需要引用检查
        if (PyDataType_REFCHK(fa->descr)) {
            # 尝试清除数组的所有引用
            if (PyArray_ClearArray(self) < 0) {
                PyErr_WriteUnraisable(NULL);
            }
        }
        # 如果内存处理器为空
        if (fa->mem_handler == NULL) {
            # 如果未设置内存策略，发出警告
            char const *msg = "Trying to dealloc data, but a memory policy "
                "is not set. If you take ownership of the data, you must "
                "set a base owning the data (e.g. a PyCapsule).";
            WARN_IN_DEALLOC(PyExc_RuntimeWarning, msg);
            // 猜测使用 malloc/free ???
            # 释放 fa->data 内存
            free(fa->data);
        }
        else {
            # 计算数据大小
            size_t nbytes = PyArray_NBYTES(self);
            if (nbytes == 0) {
                nbytes = 1;
            }
            # 使用 PyDataMem_UserFREE 释放 fa->data
            PyDataMem_UserFREE(fa->data, nbytes, fa->mem_handler);
            # DECREF fa->mem_handler
            Py_DECREF(fa->mem_handler);
        }
    }

    /* 必须匹配 PyArray_NewFromDescr 中的分配 */
    # 释放数组的维度缓存
    npy_free_cache_dim(fa->dimensions, 2 * fa->nd);
    # DECREF fa->descr
    Py_DECREF(fa->descr);
    # 调用类型对象的 tp_free 方法释放 self 对象
    Py_TYPE(self)->tp_free((PyObject *)self);
/*NUMPY_API
 * Prints the raw data of the ndarray in a form useful for debugging
 * low-level C issues.
 */
// 定义一个函数 PyArray_DebugPrint，用于打印 NumPy ndarray 的原始数据，便于低级别 C 问题的调试
NPY_NO_EXPORT void
PyArray_DebugPrint(PyArrayObject *obj)
{
    int i;
    // 将输入的 PyArrayObject 转换为 PyArrayObject_fields 类型的指针 fobj
    PyArrayObject_fields *fobj = (PyArrayObject_fields *)obj;

    // 打印一条分隔线，指示 ndarray 的内存地址
    printf("-------------------------------------------------------\n");
    printf(" Dump of NumPy ndarray at address %p\n", obj);

    // 如果 obj 为 NULL，则输出相应的消息并返回
    if (obj == NULL) {
        printf(" It's NULL!\n");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
        return;
    }

    // 打印 ndarray 的维度信息
    printf(" ndim   : %d\n", fobj->nd);
    // 打印 ndarray 的形状信息
    printf(" shape  :");
    for (i = 0; i < fobj->nd; ++i) {
        printf(" %" NPY_INTP_FMT, fobj->dimensions[i]);
    }
    printf("\n");

    // 打印 ndarray 的数据类型信息
    printf(" dtype  : ");
    PyObject_Print((PyObject *)fobj->descr, stdout, 0);
    printf("\n");
    // 打印 ndarray 的数据起始地址
    printf(" data   : %p\n", fobj->data);
    // 打印 ndarray 的跨步信息
    printf(" strides:");
    for (i = 0; i < fobj->nd; ++i) {
        printf(" %" NPY_INTP_FMT, fobj->strides[i]);
    }
    printf("\n");

    // 打印 ndarray 的基类对象的地址
    printf(" base   : %p\n", fobj->base);

    // 打印 ndarray 的 flags 信息，包括内存布局等属性
    printf(" flags :");
    if (fobj->flags & NPY_ARRAY_C_CONTIGUOUS)
        printf(" NPY_C_CONTIGUOUS");
    if (fobj->flags & NPY_ARRAY_F_CONTIGUOUS)
        printf(" NPY_F_CONTIGUOUS");
    if (fobj->flags & NPY_ARRAY_OWNDATA)
        printf(" NPY_OWNDATA");
    if (fobj->flags & NPY_ARRAY_ALIGNED)
        printf(" NPY_ALIGNED");
    if (fobj->flags & NPY_ARRAY_WRITEABLE)
        printf(" NPY_WRITEABLE");
    if (fobj->flags & NPY_ARRAY_WRITEBACKIFCOPY)
        printf(" NPY_WRITEBACKIFCOPY");
    printf("\n");

    // 如果 ndarray 的基类不为 NULL 且是一个 PyArrayObject，则进行递归打印其基类的信息
    if (fobj->base != NULL && PyArray_Check(fobj->base)) {
        printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
        printf("Dump of array's BASE:\n");
        PyArray_DebugPrint((PyArrayObject *)fobj->base);
        printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    }

    // 打印结束的分隔线，并刷新输出缓冲区
    printf("-------------------------------------------------------\n");
    fflush(stdout);
}

/* Call this from contexts where an array might be written to, but we have no
 * way to tell. (E.g., when converting to a read-write buffer.)
 */
// 当可能写入数组但没有明确方法告知时调用此函数（例如，在转换为读写缓冲区时）
NPY_NO_EXPORT int
array_might_be_written(PyArrayObject *obj)
{
    // 提示消息的内容，用于警告可能写入的情况
    const char *msg =
        "Numpy has detected that you (may be) writing to an array with\n"
        "overlapping memory from np.broadcast_arrays. If this is intentional\n"
        "set the WRITEABLE flag True or make a copy immediately before writing.";

    // 如果 obj 的标志中包含 NPY_ARRAY_WARN_ON_WRITE
    if (PyArray_FLAGS(obj) & NPY_ARRAY_WARN_ON_WRITE) {
        // 弃用警告消息，并在失败时返回 -1
        if (DEPRECATE(msg) < 0) {
            return -1;
        }
        // 一次性只发出一次警告
        while (1) {
            // 清除数组的 WARN_ON_WRITE 标志
            PyArray_CLEARFLAGS(obj, NPY_ARRAY_WARN_ON_WRITE);
            // 如果 obj 的基类不为空且是一个 PyArrayObject，则继续处理其基类
            if (!PyArray_BASE(obj) || !PyArray_Check(PyArray_BASE(obj))) {
                break;
            }
            obj = (PyArrayObject *)PyArray_BASE(obj);
        }
    }
    return 0;
}
/*
 * NUMPY_API
 *
 *  如果 *obj* 是可写的，则此函数不执行任何操作并返回 0。
 *  如果 *obj* 不可写，则引发异常并返回 -1。
 *  它可能还会执行其他的一些工作，例如在数组转变为视图时发出警告。
 *  在写入数组之前一定要调用此函数。
 *
 *  *name* 是数组的名称，用于提供更好的错误消息。
 *  可以是诸如 "assignment destination"、"output array"，甚至只是 "array"。
 */
NPY_NO_EXPORT int
PyArray_FailUnlessWriteable(PyArrayObject *obj, const char *name)
{
    // 检查 *obj* 是否为可写数组，如果不可写则设置异常并返回 -1
    if (!PyArray_ISWRITEABLE(obj)) {
        PyErr_Format(PyExc_ValueError, "%s is read-only", name);
        return -1;
    }
    // 检查数组是否可能被写入，如果是则返回 -1
    if (array_might_be_written(obj) < 0) {
        return -1;
    }
    // 如果 *obj* 可写，返回 0
    return 0;
}


/*
 * 从 umath/string_ufuncs.cpp/h 中引入
 */
NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip);

/*
 * VOID 类型的数组只能进行相等和不相等比较
 * 在这种情况下，所有字段都会被提取并逐个进行测试...
 * 相等性测试通过对所有字段使用逻辑与进行。
 * 不等性测试通过对所有字段使用逻辑或进行。
 *
 * 没有字段的 VOID 类型数组通过直接比较每个位置上的内存进行相等性比较。
 */
static PyObject *
_void_compare(PyArrayObject *self, PyArrayObject *other, int cmp_op)
{
    // 检查是否支持除了相等和不相等之外的比较操作，不支持则设置异常并返回 NULL
    if (!(cmp_op == Py_EQ || cmp_op == Py_NE)) {
        PyErr_SetString(PyExc_TypeError,
                "Void-arrays can only be compared for equality.");
        return NULL;
    }
    // 检查 *other* 是否为 NPY_VOID 类型，不是则设置异常并返回 NULL
    if (PyArray_TYPE(other) != NPY_VOID) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot compare structured or void to non-void arrays.");
        return NULL;
    }
    // 如果 *self* 或 *other* 具有字段，则设置异常并返回 NULL
    else if (PyArray_HASFIELDS(self) || PyArray_HASFIELDS(other)) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot compare structured with unstructured void arrays. "
                "(unreachable error, please report to NumPy devs.)");
        return NULL;
    }
    else {
        /*
         * 由于数组吸收了子数组描述符，只有在两个数组都具有非结构化 VOID "V<len>" 数据类型时才会进入此路径。
         */
        // 检查 *self* 和 *other* 的 ITEMSIZE 是否相同，如果不同则设置异常并返回 NULL
        if (PyArray_ITEMSIZE(self) != PyArray_ITEMSIZE(other)) {
            PyErr_SetString(PyExc_TypeError,
                    "cannot compare unstructured voids of different length. "
                    "Use bytes to compare. "
                    "(This may return array of False in the future.)");
            return NULL;
        }
        /* 使用字符串比较。假设 *self* 和 *other* 具有相同的 descr->type */
        // 调用 _umath_strings_richcompare 函数进行字符串比较，并返回结果
        return _umath_strings_richcompare(self, other, cmp_op, 0);
    }
}
/*
 * Silence the current error and emit a deprecation warning instead.
 *
 * If warnings are raised as errors, this sets the warning __cause__ to the
 * silenced error.
 */
NPY_NO_EXPORT int
DEPRECATE_silence_error(const char *msg) {
    // 声明保存异常信息的对象指针
    PyObject *exc, *val, *tb;
    // 捕获当前异常
    PyErr_Fetch(&exc, &val, &tb);
    // 发出过时警告信息，并检查是否发生错误
    if (DEPRECATE(msg) < 0) {
        // 如果发生错误，将之前捕获的异常设置为警告信息的 __cause__
        npy_PyErr_ChainExceptionsCause(exc, val, tb);
        return -1;
    }
    // 清除异常对象的引用计数
    Py_XDECREF(exc);
    Py_XDECREF(val);
    Py_XDECREF(tb);
    return 0;
}


NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op)
{
    // 声明其他用于比较的数组对象和结果对象
    PyArrayObject *array_other;
    PyObject *obj_self = (PyObject *)self;
    PyObject *result = NULL;

    switch (cmp_op) {
    case Py_LT:
        // 如果需要，放弃富比较
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        // 执行小于比较操作
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, other, n_ops.less);
        break;
    case Py_LE:
        // 如果需要，放弃富比较
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        // 执行小于等于比较操作
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, other, n_ops.less_equal);
        break;
    case Py_EQ:
        // 如果需要，放弃富比较
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        /*
         * The ufunc does not support void/structured types, so these
         * need to be handled specifically. Only a few cases are supported.
         */
        
        // 如果数组类型为 NPY_VOID
        if (PyArray_TYPE(self) == NPY_VOID) {
            // 将 Python 对象 other 转换为 PyArrayObject 类型
            array_other = (PyArrayObject *)PyArray_FROM_O(other);
            // 如果转换不成功，发出过时警告并返回 NotImplemented
            if (array_other == NULL) {
                /* 2015-05-07, 1.10 */
                if (DEPRECATE_silence_error(
                        "elementwise == comparison failed and returning scalar "
                        "instead; this will raise an error in the future.") < 0) {
                    return NULL;
                }
                // 增加 NotImplemented 对象的引用计数并返回
                Py_INCREF(Py_NotImplemented);
                return Py_NotImplemented;
            }

            // 对 VOID 类型进行比较操作
            result = _void_compare(self, array_other, cmp_op);
            Py_DECREF(array_other);
            return result;
        }

        // 执行一般的相等比较操作
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, (PyObject *)other, n_ops.equal);
        break;
    # 处理 Python 对象的不等于运算符操作
    case Py_NE:
        # 在必要时放弃比较
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        
        """
         * 该 ufunc 不支持 void/structured 类型，因此需要特别处理。
         * 仅支持少数情况。
         """
        
        # 如果 self 是 NPY_VOID 类型
        if (PyArray_TYPE(self) == NPY_VOID) {
            # 将 other 转换为 PyArrayObject 类型
            array_other = (PyArrayObject *)PyArray_FROM_O(other);
            
            """
             * 如果转换不成功，则表示无法以这种方式比较项目。
             """
            
            if (array_other == NULL) {
                /* 2015-05-07, 1.10 */
                # 如果无法成功，发出警告并返回 Py_NotImplemented
                if (DEPRECATE_silence_error(
                        "elementwise != comparison failed and returning scalar "
                        "instead; this will raise an error in the future.") < 0) {
                    return NULL;
                }
                Py_INCREF(Py_NotImplemented);
                return Py_NotImplemented;
            }

            # 执行 void 类型的比较操作
            result = _void_compare(self, array_other, cmp_op);
            Py_DECREF(array_other);
            return result;
        }

        # 对于其他类型的对象，使用通用的二元函数
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, (PyObject *)other, n_ops.not_equal);
        break;
        
    # 处理 Python 对象的大于运算符操作
    case Py_GT:
        # 在必要时放弃比较
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, other, n_ops.greater);
        break;
        
    # 处理 Python 对象的大于等于运算符操作
    case Py_GE:
        # 在必要时放弃比较
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, other, n_ops.greater_equal);
        break;
        
    # 处理其他未知的运算符操作
    default:
        # 返回 Py_NotImplemented 表示操作不支持
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    /*
     * 现在，`self` 可以通过将 `other` 转换为数组来控制操作
     * （它有机会接管操作）。
     * 如果我们不在 `==` 和 `!=` 中，这将是一个错误，
     * 我们希望现有的错误能够有意义，并且源自 `TypeError`
     * （当 Python 应该为 `NotImplemented` 抛出时）。
     *
     * 然而，如果问题是给定 dtypes 没有匹配的循环，
     * 并且我们在 `==` 和 `!=` 中，那么返回一个 True 或 False 的数组是有意义的
     * （遵循 Python 对于 `==` 和 `!=` 的行为）。
     * 实际上：两个 *dtypes* 告诉我们它们无法比较。
     *
     * 理论上，错误可能会在对象循环内部引发，解决这个问题的方法可能是将其推入到 ufunc 中
     * （在那里我们可以轻松区分这两种情况）。
     * 实际上，这似乎不应该是一个巨大的问题：
     * ufunc 循环本身将调用 `==`，它可能永远不会引发 UFuncNoLoopError。
     *
     * TODO: 如果/一旦我们正确地将结构化比较推入 ufunc，
     *       我们可以考虑将这条路径作为回退循环推入 ufunc 本身
     *       （忽略输入数组）。
     *       这将有一个优势，即实现 `__array_ufunc__` 的子类不需要显式地实现 `__eq__` 和 `__ne__`。
     */
    if (result == NULL
            && (cmp_op == Py_EQ || cmp_op == Py_NE)
            && PyErr_ExceptionMatches(
                    npy_static_pydata._UFuncNoLoopError)) {
        PyErr_Clear();

        PyArrayObject *array_other = (PyArrayObject *)PyArray_FROM_O(other);
        if (PyArray_TYPE(array_other) == NPY_VOID) {
            /*
            * 对于空值数组（void arrays），ufunc 目前不处理，所以如果 `other` 是空值数组，
            * 我们会将操作推迟给它（会引发 TypeError）。
            */
            Py_DECREF(array_other);
            Py_RETURN_NOTIMPLEMENTED;
        }

        if (PyArray_NDIM(self) == 0 && PyArray_NDIM(array_other) == 0) {
            /*
             * （seberg）我不确定这是否最佳做法，但是我们暂时通过返回 `NotImplemented` 来保留
             * “标量”输入的 Python 布尔结果。
             */
            Py_DECREF(array_other);
            Py_RETURN_NOTIMPLEMENTED;
        }

        /* Hack warning: using NpyIter to allocate broadcasted result. */
        PyArrayObject *ops[3] = {self, array_other, NULL};
        npy_uint32 flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_REFS_OK;
        npy_uint32 op_flags[3] = {
            NPY_ITER_READONLY, NPY_ITER_READONLY,
            NPY_ITER_ALLOCATE | NPY_ITER_WRITEONLY};

        PyArray_Descr *bool_descr = PyArray_DescrFromType(NPY_BOOL);
        PyArray_Descr *op_descrs[3] = {
            PyArray_DESCR(self), PyArray_DESCR(array_other), bool_descr};

        // 使用 NpyIter_MultiNew 创建多重迭代器，用于处理多个操作数
        NpyIter *iter = NpyIter_MultiNew(
                    3, ops, flags, NPY_KEEPORDER, NPY_NO_CASTING,
                    op_flags, op_descrs);

        Py_CLEAR(bool_descr);
        Py_CLEAR(array_other);
        if (iter == NULL) {
            return NULL;
        }
        // 从迭代器中获取结果数组，这里是第三个操作数
        PyArrayObject *res = NpyIter_GetOperandArray(iter)[2];
        Py_INCREF(res);
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(res);
            return NULL;
        }

        /*
         * 数组保证是新分配的，因此是连续的，可以简单地用 0 或 1 填充它。
         */
        memset(PyArray_BYTES(res), cmp_op == Py_EQ ? 0 : 1, PyArray_NBYTES(res));

        /* Ensure basic subclass support by wrapping: */
        if (!PyArray_CheckExact(self)) {
            /*
             * 如果 `other` 也是一个子类（优先级更高），我们已经推迟处理了。所以使用 `self` 进行包装。
             * 如果用户需要更多，他们需要重写 `==` 和 `!=`。
             */
            PyObject *wrapped = npy_apply_wrap_simple(self, res);
            Py_DECREF(res);
            return wrapped;
        }
        return (PyObject *)res;
    }
    return result;
}

/*NUMPY_API
 */
# 定义一个名为 PyArray_ElementStrides 的函数，接受一个 PyObject 对象作为参数
NPY_NO_EXPORT int
PyArray_ElementStrides(PyObject *obj)
{
    PyArrayObject *arr;  // 定义一个 PyArrayObject 指针变量 arr
    int itemsize;  // 定义一个整型变量 itemsize
    int i, ndim;  // 定义两个整型变量 i 和 ndim
    npy_intp *strides;  // 定义一个 npy_intp 类型的指针变量 strides

    // 如果 obj 不是一个 NumPy 数组对象，则返回 0
    if (!PyArray_Check(obj)) {
        return 0;
    }

    arr = (PyArrayObject *)obj;  // 将 obj 强制转换为 PyArrayObject 类型的指针并赋值给 arr

    itemsize = PyArray_ITEMSIZE(arr);  // 调用 PyArray_ITEMSIZE 宏，获取 arr 对象的每个元素大小
    ndim = PyArray_NDIM(arr);  // 调用 PyArray_NDIM 宏，获取 arr 对象的维度数
    strides = PyArray_STRIDES(arr);  // 调用 PyArray_STRIDES 宏，获取 arr 对象的步长数组

    // 遍历数组的每一个维度
    for (i = 0; i < ndim; i++) {
        // 检查当前维度的步长是否是 itemsize 的整数倍，如果不是则返回 0
        if ((strides[i] % itemsize) != 0) {
            return 0;
        }
    }
    return 1;  // 如果所有维度的步长都符合条件，则返回 1
}

/*
 * This routine checks to see if newstrides (of length nd) will not
 * ever be able to walk outside of the memory implied numbytes and offset.
 *
 * The available memory is assumed to start at -offset and proceed
 * to numbytes-offset.  The strides are checked to ensure
 * that accessing memory using striding will not try to reach beyond
 * this memory for any of the axes.
 *
 * If numbytes is 0 it will be calculated using the dimensions and
 * element-size.
 *
 * This function checks for walking beyond the beginning and right-end
 * of the buffer and therefore works for any integer stride (positive
 * or negative).
 */

/*NUMPY_API*/
# 定义一个名为 PyArray_CheckStrides 的函数，接受多个参数
NPY_NO_EXPORT npy_bool
PyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset,
                     npy_intp const *dims, npy_intp const *newstrides)
{
    npy_intp begin, end;  // 定义两个 npy_intp 类型的变量 begin 和 end
    npy_intp lower_offset;  // 定义一个 npy_intp 类型的变量 lower_offset
    npy_intp upper_offset;  // 定义一个 npy_intp 类型的变量 upper_offset

    // 如果 numbytes 为 0，则根据维度和元素大小计算 numbytes
    if (numbytes == 0) {
        numbytes = PyArray_MultiplyList(dims, nd) * elsize;
    }

    begin = -offset;  // 计算可用内存的起始位置
    end = numbytes - offset;  // 计算可用内存的结束位置

    // 调用 offset_bounds_from_strides 函数，计算根据新步长数组 newstrides 计算的下限和上限偏移量
    offset_bounds_from_strides(elsize, nd, dims, newstrides,
                                        &lower_offset, &upper_offset);

    // 检查上限偏移量是否超出了可用内存的结束位置，或者下限偏移量是否超出了可用内存的起始位置
    if ((upper_offset > end) || (lower_offset < begin)) {
        return NPY_FALSE;  // 如果超出范围则返回 NPY_FALSE
    }
    return NPY_TRUE;  // 如果在范围内则返回 NPY_TRUE
}


static PyObject *
# 定义一个名为 array_new 的静态函数，接受 PyObject 类型的指针变量作为参数
array_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"shape", "dtype", "buffer", "offset", "strides",
                             "order", NULL};
    PyArray_Descr *descr = NULL;  // 定义一个 PyArray_Descr 指针变量 descr
    int itemsize;  // 定义一个整型变量 itemsize
    PyArray_Dims dims = {NULL, 0};  // 定义一个 PyArray_Dims 结构体变量 dims，初始化为 NULL 和 0
    PyArray_Dims strides = {NULL, -1};  // 定义一个 PyArray_Dims 结构体变量 strides，初始化为 NULL 和 -1
    PyArray_Chunk buffer;  // 定义一个 PyArray_Chunk 结构体变量 buffer
    npy_longlong offset = 0;  // 定义一个 npy_longlong 类型的变量 offset，初始化为 0
    NPY_ORDER order = NPY_CORDER;  // 定义一个 NPY_ORDER 类型的变量 order，初始化为 NPY_CORDER
    int is_f_order = 0;  // 定义一个整型变量 is_f_order，初始化为 0
    PyArrayObject *ret;  // 定义一个 PyArrayObject 指针变量 ret

    buffer.ptr = NULL;  // 将 buffer.ptr 成员变量设置为 NULL
    /*
     * Usually called with shape and type but can also be called with buffer,
     * strides, and swapped info For now, let's just use this to create an
     * empty, contiguous array of a specific type and shape.
     */
    # 使用 PyArg_ParseTupleAndKeywords 解析传入的参数，格式为 "O&|O&O&LO&O&:ndarray"，
    # 对应的解析器分别为 PyArray_IntpConverter, PyArray_DescrConverter,
    # PyArray_BufferConverter, PyArray_OptionalIntpConverter, PyArray_OrderConverter。
    # 如果解析失败，则跳转到失败处理块。
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&LO&O&:ndarray",
                                     kwlist, PyArray_IntpConverter,
                                     &dims,
                                     PyArray_DescrConverter,
                                     &descr,
                                     PyArray_BufferConverter,
                                     &buffer,
                                     &offset,
                                     &PyArray_OptionalIntpConverter,
                                     &strides,
                                     &PyArray_OrderConverter,
                                     &order)) {
        goto fail;
    }

    # 如果数组的存储顺序是 NPY_FORTRANORDER，则设置为列优先存储。
    if (order == NPY_FORTRANORDER) {
        is_f_order = 1;
    }

    # 如果没有提供数组的描述符，则使用默认类型创建一个描述符。
    if (descr == NULL) {
        descr = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    # 获取数组元素的大小。
    itemsize = descr->elsize;

    # 如果 strides.len 不等于 -1，表示用户提供了步幅信息。
    if (strides.len != -1) {
        npy_intp nb, off;

        # 如果步幅的长度不等于维度的长度，则抛出值错误并跳转到失败处理块。
        if (strides.len != dims.len) {
            PyErr_SetString(PyExc_ValueError,
                            "strides, if given, must be "   \
                            "the same length as shape");
            goto fail;
        }

        # 如果缓冲区的指针为 NULL，则将 nb 和 off 设置为 0。
        # 否则，分别使用 buffer.len 和 offset 赋值给 nb 和 off。
        if (buffer.ptr == NULL) {
            nb = 0;
            off = 0;
        }
        else {
            nb = buffer.len;
            off = (npy_intp) offset;
        }

        # 检查给定的步幅是否与请求数组的形状和缓冲区大小兼容，
        # 如果不兼容，则抛出值错误并跳转到失败处理块。
        if (!PyArray_CheckStrides(itemsize, dims.len,
                                  nb, off,
                                  dims.ptr, strides.ptr)) {
            PyErr_SetString(PyExc_ValueError,
                            "strides is incompatible "      \
                            "with shape of requested "      \
                            "array and size of buffer");
            goto fail;
        }
    }

    # 如果缓冲区的指针为 NULL，则创建一个新的数组对象，
    # 使用提供的描述符和其他参数。
    if (buffer.ptr == NULL) {
        ret = (PyArrayObject *)
            PyArray_NewFromDescr_int(subtype, descr,
                                     (int)dims.len,
                                     dims.ptr,
                                     strides.ptr, NULL, is_f_order, NULL, NULL,
                                     _NPY_ARRAY_ALLOW_EMPTY_STRING);
        # 如果创建失败，则将描述符设置为 NULL，并跳转到失败处理块。
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
        
        # 如果数组描述符有参考检查的需求，
        # 则将对象位置设置为 Py_None。
        if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
            if (PyArray_SetObjectsToNone(ret) < 0) {
                descr = NULL;
                goto fail;
            }
        }
    }
    else {
        /* 如果给定了缓冲区，则使用它 */

        // 如果维度长度为1且第一个元素为-1，则根据缓冲区长度和每个元素大小计算第一个维度的大小
        if (dims.len == 1 && dims.ptr[0] == -1) {
            dims.ptr[0] = (buffer.len-(npy_intp)offset) / itemsize;
        }
        // 如果未指定步长并且缓冲区长度小于所需数组的总大小，则抛出类型错误异常
        else if ((strides.ptr == NULL) &&
                 (buffer.len < (offset + (((npy_intp)itemsize)*
                                          PyArray_MultiplyList(dims.ptr,
                                                               dims.len))))) {
            PyErr_SetString(PyExc_TypeError,
                            "buffer is too small for "      \
                            "requested array");
            goto fail;
        }
        
        /* 获取可写和对齐的数组 */

        // 如果数组应按F顺序存储，则设置数组的标志以指示F连续性
        if (is_f_order) {
            buffer.flags |= NPY_ARRAY_F_CONTIGUOUS;
        }

        // 从描述符创建新的数组对象，返回PyArrayObject指针
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                subtype, descr,
                dims.len, dims.ptr, strides.ptr, offset + (char *)buffer.ptr,
                buffer.flags, NULL, buffer.base,
                _NPY_ARRAY_ALLOW_EMPTY_STRING);
        
        // 如果创建失败，则清空描述符并跳转到失败处理标签
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
    }

    // 释放维度对象的缓存
    npy_free_cache_dim_obj(dims);
    // 释放步长对象的缓存
    npy_free_cache_dim_obj(strides);
    // 返回成功创建的数组对象的PyObject指针
    return (PyObject *)ret;

 fail:
    // 清空描述符对象，释放维度和步长对象的缓存，返回NULL指针表示创建数组失败
    Py_XDECREF(descr);
    npy_free_cache_dim_obj(dims);
    npy_free_cache_dim_obj(strides);
    return NULL;
# 定义静态全局变量 PyArray_Type，表示 numpy.ndarray 类型对象
NPY_NO_EXPORT PyTypeObject PyArray_Type = {
    # 使用 PyVarObject_HEAD_INIT 宏初始化变量头部信息
    PyVarObject_HEAD_INIT(NULL, 0)
    # 设置类型名称为 "numpy.ndarray"
    .tp_name = "numpy.ndarray",
    # 设置对象基本大小为 PyArrayObject_fields 结构体的大小
    .tp_basicsize = sizeof(PyArrayObject_fields),
    # 方法定义部分
    # 定义对象销毁时的函数为 array_dealloc
    .tp_dealloc = (destructor)array_dealloc,
    # 定义对象的字符串表示方法为 array_repr
    .tp_repr = (reprfunc)array_repr,
    # 定义对象作为数字的接口为 array_as_number
    .tp_as_number = &array_as_number,
    # 定义对象作为序列的接口为 array_as_sequence
    .tp_as_sequence = &array_as_sequence,
    # 定义对象作为映射的接口为 array_as_mapping
    .tp_as_mapping = &array_as_mapping,
    # 定义对象的字符串转换方法为 array_str
    .tp_str = (reprfunc)array_str,
    # 定义对象作为缓冲区的接口为 array_as_buffer
    .tp_as_buffer = &array_as_buffer,
    # 设置对象的默认标志为 Py_TPFLAGS_DEFAULT 和 Py_TPFLAGS_BASETYPE 的按位或
    .tp_flags = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE),

    # 定义对象的富比较函数为 array_richcompare
    .tp_richcompare = (richcmpfunc)array_richcompare,
    # 设置弱引用列表的偏移量为 PyArrayObject_fields 结构体中 weakreflist 的偏移量
    .tp_weaklistoffset = offsetof(PyArrayObject_fields, weakreflist),
    # 定义对象的迭代器函数为 array_iter
    .tp_iter = (getiterfunc)array_iter,
    # 定义对象的方法集合为 array_methods
    .tp_methods = array_methods,
    # 定义对象的属性获取与设置列表为 array_getsetlist
    .tp_getset = array_getsetlist,
    # 定义对象的构造函数为 array_new
    .tp_new = (newfunc)array_new,
};
```