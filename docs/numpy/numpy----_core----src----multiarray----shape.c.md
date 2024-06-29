# `.\numpy\numpy\_core\src\multiarray\shape.c`

```
/* 定义 NPY_NO_DEPRECATED_API 为 NPY_API_VERSION */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/* 定义 _MULTIARRAYMODULE */

#define _MULTIARRAYMODULE

/* 定义 PY_SSIZE_T_CLEAN */
#define PY_SSIZE_T_CLEAN

/* 包含 Python 标准头文件 */
#include <Python.h>
/* 包含结构成员的头文件 */
#include <structmember.h>

/* 包含 NumPy 数组对象的头文件 */
#include "numpy/arrayobject.h"
/* 包含 NumPy 数组标量的头文件 */
#include "numpy/arrayscalars.h"

/* 包含 NumPy 的数学函数头文件 */
#include "numpy/npy_math.h"

/* 包含 NumPy 配置文件头文件 */
#include "npy_config.h"

/* 包含数组包装相关的头文件 */
#include "arraywrap.h"
/* 包含构造函数相关的头文件 */
#include "ctors.h"

/* 包含数组形状相关的头文件 */
#include "shape.h"

/* 包含 NumPy 的静态数据头文件，用于内部化字符串 */
#include "npy_static_data.h"

/* 包含模板通用函数头文件，用于 npy_mul_sizes_with_overflow */
#include "templ_common.h"

/* 包含通用函数头文件，用于 convert_shape_to_string */
#include "common.h"

/* 包含分配相关的头文件 */
#include "alloc.h"

/* 定义修复未知维度的静态函数 */
static int
_fix_unknown_dimension(PyArray_Dims *newshape, PyArrayObject *arr);

/* 定义尝试无拷贝重塑的静态函数 */
static int
_attempt_nocopy_reshape(PyArrayObject *self, int newnd, const npy_intp *newdims,
                        npy_intp *newstrides, int is_f_order);

/* 定义将零值放入指定位置的静态函数 */
static void
_putzero(char *optr, PyObject *zero, PyArray_Descr *dtype);

/* NUMPY_API
 * 调整大小（重新分配数据）。仅在数组是单一段且没有其他引用的情况下才有效。
 * 如果 refcheck 是 0，则不检查引用计数，假设引用计数为 1。
 * 仍然需要拥有数据并且没有弱引用和基对象。
 */
NPY_NO_EXPORT PyObject *
PyArray_Resize(PyArrayObject *self, PyArray_Dims *newshape, int refcheck,
               NPY_ORDER NPY_UNUSED(order))
{
    npy_intp oldnbytes, newnbytes;
    npy_intp oldsize, newsize;
    int new_nd = newshape->len, k, elsize;
    int refcnt;
    npy_intp* new_dimensions = newshape->ptr;
    npy_intp new_strides[NPY_MAXDIMS];
    npy_intp *dimptr;
    char *new_data;

    /* 如果数组不是单一段，报错并返回空指针 */
    if (!PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_ValueError,
                "resize only works on single-segment arrays");
        return NULL;
    }

    /* 计算旧数组和新数组的总大小。新大小可能会溢出 */
    oldsize = PyArray_SIZE(self);
    newsize = 1;
    for(k = 0; k < new_nd; k++) {
        /* 如果维度为 0，则新大小为 0 */
        if (new_dimensions[k] == 0) {
            newsize = 0;
            break;
        }
        /* 如果维度为负数，报错并返回空指针 */
        if (new_dimensions[k] < 0) {
            PyErr_SetString(PyExc_ValueError,
                    "negative dimensions not allowed");
            return NULL;
        }
        /* 计算新大小，防止溢出 */
        if (npy_mul_sizes_with_overflow(&newsize, newsize, new_dimensions[k])) {
            return PyErr_NoMemory();
        }
    }

    /* 将大小转换为字节数。新的计数可能会溢出 */
    elsize = PyArray_ITEMSIZE(self);
    oldnbytes = oldsize * elsize;
    if (npy_mul_sizes_with_overflow(&newnbytes, newsize, elsize)) {
        return PyErr_NoMemory();
    }
    # 如果旧字节大小不等于新字节大小，则执行以下代码块
    if (oldnbytes != newnbytes) {
        # 如果数组不拥有自己的数据（即不是内存所有者），则抛出数值错误异常并返回空
        if (!(PyArray_FLAGS(self) & NPY_ARRAY_OWNDATA)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize this array: it does not own its data");
            return NULL;
        }

        # 如果数组有基础对象（base）或者有弱引用列表，则抛出数值错误异常并返回空
        if (PyArray_BASE(self) != NULL
              || (((PyArrayObject_fields *)self)->weakreflist != NULL)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize an array that "
                    "references or is referenced\n"
                    "by another array in this way. Use the np.resize function.");
            return NULL;
        }
        
        # 如果启用了引用检查（refcheck 参数为真），则执行以下代码块
        if (refcheck) {
#ifdef PYPY_VERSION
            /* 如果在 PyPy 上且 refcheck=True，则不能调整数组大小 */
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize an array with refcheck=True on PyPy.\n"
                    "Use the np.resize function or refcheck=False");
            return NULL;
#else
            /* 否则获取对象的引用计数 */
            refcnt = Py_REFCNT(self);
#endif /* PYPY_VERSION */
        }
        else {
            /* 对于不可变对象，引用计数为 1 */
            refcnt = 1;
        }
        if (refcnt > 2) {
            /* 如果引用计数大于 2，则无法调整大小 */
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize an array that "
                    "references or is referenced\n"
                    "by another array in this way.\n"
                    "Use the np.resize function or refcheck=False");
            return NULL;
        }

        /* 如果需要重新分配空间，则进行重新分配 - 分配 0 大小被禁止 */
        PyObject *handler = PyArray_HANDLER(self);
        if (handler == NULL) {
            /* 如果没有找到内存处理器但设置了 NPY_ARRAY_OWNDATA 标志，则报错 */
            PyErr_SetString(PyExc_RuntimeError,
                            "no memory handler found but OWNDATA flag set");
            return NULL;
        }
        /* 使用 PyDataMem_UserRENEW 重新分配数据 */
        new_data = PyDataMem_UserRENEW(PyArray_DATA(self),
                                       newnbytes == 0 ? elsize : newnbytes,
                                       handler);
        if (new_data == NULL) {
            /* 如果无法分配内存，则报内存错误 */
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate memory for array");
            return NULL;
        }
        /* 更新数组对象的数据指针 */
        ((PyArrayObject_fields *)self)->data = new_data;
    }

    /* 如果新分配的大小大于旧大小且数组可写，则填充新分配的内存区域为零 */
    if (newnbytes > oldnbytes && PyArray_ISWRITEABLE(self)) {
        /* 如果数组的数据类型包含引用计数，则使用 _putzero 函数填充 */
        if (PyDataType_FLAGCHK(PyArray_DESCR(self), NPY_ITEM_REFCOUNT)) {
            PyObject *zero = PyLong_FromLong(0);
            char *optr;
            /* 指向新内存的起始位置 */
            optr = PyArray_BYTES(self) + oldnbytes;
            /* 计算需要填充零的元素数量 */
            npy_intp n_new = newsize - oldsize;
            /* 循环填充零 */
            for (npy_intp i = 0; i < n_new; i++) {
                _putzero((char *)optr, zero, PyArray_DESCR(self));
                optr += elsize;
            }
            /* 释放零对象 */
            Py_DECREF(zero);
        }
        else {
            /* 否则使用 memset 函数填充零 */
            memset(PyArray_BYTES(self) + oldnbytes, 0, newnbytes - oldnbytes);
        }
    }
    # 如果新的维度数大于零，则需要进行维度和步长的调整
    if (new_nd > 0) {
        # 如果当前数组的维度数与新的维度数不同
        if (PyArray_NDIM(self) != new_nd) {
            /* 不同的维度数。*/
            # 更新数组对象的维度数为新的维度数
            ((PyArrayObject_fields *)self)->nd = new_nd;
            /* 需要新的维度和步长数组 */
            # 重新分配维度和步长数组的内存空间
            dimptr = PyDimMem_RENEW(PyArray_DIMS(self), 3*new_nd);
            # 如果内存分配失败
            if (dimptr == NULL) {
                # 报告内存错误
                PyErr_SetString(PyExc_MemoryError,
                                "cannot allocate memory for array");
                return NULL;
            }
            # 更新数组对象的维度和步长指针
            ((PyArrayObject_fields *)self)->dimensions = dimptr;
            ((PyArrayObject_fields *)self)->strides = dimptr + new_nd;
        }
        /* 生成新的步长变量 */
        # 填充新的步长数组
        _array_fill_strides(new_strides, new_dimensions, new_nd,
                            PyArray_ITEMSIZE(self), PyArray_FLAGS(self),
                            &(((PyArrayObject_fields *)self)->flags));
        # 移动新维度数据到数组对象的维度数组中
        memmove(PyArray_DIMS(self), new_dimensions, new_nd*sizeof(npy_intp));
        # 移动新步长数据到数组对象的步长数组中
        memmove(PyArray_STRIDES(self), new_strides, new_nd*sizeof(npy_intp));
    }
    else {
        # 如果新的维度数不大于零，则清理数组对象的维度和步长数组
        PyDimMem_FREE(((PyArrayObject_fields *)self)->dimensions);
        ((PyArrayObject_fields *)self)->nd = 0;
        ((PyArrayObject_fields *)self)->dimensions = NULL;
        ((PyArrayObject_fields *)self)->strides = NULL;
    }
    # 返回 None
    Py_RETURN_NONE;
/*
 * 返回一个新的数组
 * 根据旧数组的数据和顺序参数创建具有新形状的数组
 * 只在必要时复制数据
 */

/*NUMPY_API
 * 为数组创建新形状
 */
NPY_NO_EXPORT PyObject *
PyArray_Newshape(PyArrayObject *self, PyArray_Dims *newdims,
                 NPY_ORDER order)
{
    // 调用内部函数进行重塑，根据需要复制数据
    return _reshape_with_copy_arg(self, newdims, order, NPY_COPY_IF_NEEDED);
}


NPY_NO_EXPORT PyObject *
_reshape_with_copy_arg(PyArrayObject *array, PyArray_Dims *newdims,
                       NPY_ORDER order, NPY_COPYMODE copy)
{
    npy_intp i;
    npy_intp *dimensions = newdims->ptr;  // 新形状的维度数组
    PyArrayObject *ret;  // 返回的重塑后的数组对象
    int ndim = newdims->len;  // 新形状的维度数
    npy_bool same;  // 标志变量，表示新旧形状是否相同
    npy_intp *strides = NULL;  // 步长数组，默认为空
    npy_intp newstrides[NPY_MAXDIMS];  // 新数组的步长数组
    int flags;  // 标志变量

    // 如果顺序为 NPY_ANYORDER，则根据数组是否为 Fortran 风格或 C 风格决定顺序
    if (order == NPY_ANYORDER) {
        order = PyArray_ISFORTRAN(array) ? NPY_FORTRANORDER : NPY_CORDER;
    }
    // 如果顺序为 NPY_KEEPORDER，不允许使用 'K' 来进行重塑，抛出异常
    else if (order == NPY_KEEPORDER) {
        PyErr_SetString(PyExc_ValueError,
                "order 'K' is not permitted for reshaping");
        return NULL;
    }
    // 快速检查是否需要执行重塑操作
    if (ndim == PyArray_NDIM(array) && copy != NPY_COPY_ALWAYS) {
        same = NPY_TRUE;
        i = 0;
        while (same && i < ndim) {
            // 检查每个维度是否相同
            if (PyArray_DIM(array, i) != dimensions[i]) {
                same = NPY_FALSE;
            }
            i++;
        }
        // 如果形状相同，则返回数组的视图
        if (same) {
            return PyArray_View(array, NULL, NULL);
        }
    }

    /*
     * 修正任何 -1 的维度，并检查新形状与旧数组大小的匹配情况
     */
    if (_fix_unknown_dimension(newdims, array) < 0) {
        return NULL;
    }
    /*
     * 内存顺序不依赖于复制/非复制的上下文
     * 始终根据 'order' 参数决定顺序
     */
    // 如果需要始终复制数据，则创建数组的副本
    if (copy == NPY_COPY_ALWAYS) {
        PyObject *newcopy = PyArray_NewCopy(array, order);
        if (newcopy == NULL) {
            return NULL;
        }
        array = (PyArrayObject *)newcopy;
    }
    else {
        /*
         * 有时我们必须创建数组的新副本
         * 以便获取正确的方向和
         * 因为我们不能简单地重用带有
         * 数据顺序的缓冲区。
         */
        Py_INCREF(array);  // 增加数组的引用计数，以防止其被意外释放
        if (((order == NPY_CORDER && !PyArray_IS_C_CONTIGUOUS(array)) ||
                (order == NPY_FORTRANORDER && !PyArray_IS_F_CONTIGUOUS(array)))) {
            int success = 0;
            success = _attempt_nocopy_reshape(array, ndim, dimensions,
                                              newstrides, order);
            if (success) {
                /* 原地重塑成功，不需要复制数组 */
                strides = newstrides;  // 更新步幅为新计算的步幅
            }
            else if (copy == NPY_COPY_NEVER) {
                PyErr_SetString(PyExc_ValueError,
                                "Unable to avoid creating a copy while reshaping.");
                Py_DECREF(array);  // 减少数组的引用计数
                return NULL;  // 返回空指针表示错误
            }
            else {
                PyObject *newcopy = PyArray_NewCopy(array, order);
                Py_DECREF(array);  // 减少原始数组的引用计数
                if (newcopy == NULL) {
                    return NULL;  // 如果复制失败，返回空指针
                }
                array = (PyArrayObject *)newcopy;  // 更新数组为新复制的数组
            }
        }
    }
    /* 我们总是必须正确地解释连续的缓冲区 */

    /* 确保 flags 参数被设置。*/
    flags = PyArray_FLAGS(array);  // 获取数组的标志位
    if (ndim > 1) {
        if (order == NPY_FORTRANORDER) {
            flags &= ~NPY_ARRAY_C_CONTIGUOUS;  // 清除 C 连续标志位
            flags |= NPY_ARRAY_F_CONTIGUOUS;   // 设置 Fortran 连续标志位
        }
        else {
            flags &= ~NPY_ARRAY_F_CONTIGUOUS;  // 清除 Fortran 连续标志位
            flags |= NPY_ARRAY_C_CONTIGUOUS;   // 设置 C 连续标志位
        }
    }

    Py_INCREF(PyArray_DESCR(array));  // 增加数组描述符的引用计数
    ret = (PyArrayObject *)PyArray_NewFromDescr_int(
            Py_TYPE(array), PyArray_DESCR(array),
            ndim, dimensions, strides, PyArray_DATA(array),
            flags, (PyObject *)array, (PyObject *)array,
            _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    Py_DECREF(array);  // 减少数组的引用计数
    return (PyObject *)ret;  // 返回处理后的 Python 对象
/* For backward compatibility -- Not recommended */
/*NUMPY_API
 * Reshape
 */
/* 定义 PyArray_Reshape 函数，用于对数组进行重新形状化操作 */
NPY_NO_EXPORT PyObject *
PyArray_Reshape(PyArrayObject *self, PyObject *shape)
{
    PyObject *ret;
    PyArray_Dims newdims;

    /* 使用 PyArray_IntpConverter 将 shape 转换为 PyArray_Dims 结构 */
    if (!PyArray_IntpConverter(shape, &newdims)) {
        return NULL;
    }
    /* 调用 PyArray_Newshape 函数重新构造数组的形状 */
    ret = PyArray_Newshape(self, &newdims, NPY_CORDER);
    /* 释放 newdims 占用的内存 */
    npy_free_cache_dim_obj(newdims);
    return ret;
}

/* 定义 _putzero 函数，用于在数组中置零操作 */
static void
_putzero(char *optr, PyObject *zero, PyArray_Descr *dtype)
{
    /* 如果 dtype 不是引用计数类型，则使用 memset 将 optr 置零 */
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        memset(optr, 0, dtype->elsize);
    }
    /* 如果 dtype 有字段 */
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        /* 遍历 dtype 的字段 */
        while (PyDict_Next(PyDataType_FIELDS(dtype), &pos, &key, &value)) {
            /* 如果字段是标题字段，则跳过 */
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            /* 解析字段的信息 */
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            /* 在 optr + offset 处置零 */
            _putzero(optr + offset, zero, new);
        }
    }
    /* 否则按 dtype 的元素大小置零 */
    else {
        npy_intp i;
        npy_intp nsize = dtype->elsize / sizeof(zero);

        /* 遍历元素并置零 */
        for (i = 0; i < nsize; i++) {
            Py_INCREF(zero);
            memcpy(optr, &zero, sizeof(zero));
            optr += sizeof(zero);
        }
    }
    return;
}

/*
 * attempt to reshape an array without copying data
 *
 * The requested newdims are not checked, but must be compatible with
 * the size of self, which must be non-zero. Other than that this
 * function should correctly handle all reshapes, including axes of
 * length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills newstrides
 *     with appropriate strides
 *
 * The "is_f_order" argument describes how the array should be viewed
 * during the reshape, not how it is stored in memory (that
 * information is in PyArray_STRIDES(self)).
 *
 * If some output dimensions have length 1, the strides assigned to
 * them are arbitrary. In the current implementation, they are the
 * stride of the next-fastest index.
 */
/* 定义 _attempt_nocopy_reshape 函数，尝试在不复制数据的情况下重新形状化数组 */
static int
_attempt_nocopy_reshape(PyArrayObject *self, int newnd, const npy_intp *newdims,
                        npy_intp *newstrides, int is_f_order)
{
    int oldnd;
    npy_intp olddims[NPY_MAXDIMS];
    npy_intp oldstrides[NPY_MAXDIMS];
    npy_intp last_stride;
    int oi, oj, ok, ni, nj, nk;

    oldnd = 0;
    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
    /* 从旧数组中移除维度为 1 的轴 */
    for (oi = 0; oi < PyArray_NDIM(self); oi++) {
        if (PyArray_DIMS(self)[oi]!= 1) {
            olddims[oldnd] = PyArray_DIMS(self)[oi];
            oldstrides[oldnd] = PyArray_STRIDES(self)[oi];
            oldnd++;
        }
    }

    /* oi to oj and ni to nj give the axis ranges currently worked with */
    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while (ni < newnd && oi < oldnd) {
        npy_intp np = newdims[ni];  // 获取新维度数组中的当前维度大小
        npy_intp op = olddims[oi];  // 获取旧维度数组中的当前维度大小

        while (np != op) {
            if (np < op) {
                /* Misses trailing 1s, these are handled later */
                np *= newdims[nj++];  // 如果新维度小于旧维度，则将当前维度乘以下一个新维度以处理后续的尾随1
            } else {
                op *= olddims[oj++];  // 否则，将当前维度乘以下一个旧维度
            }
        }

        /* Check whether the original axes can be combined */
        for (ok = oi; ok < oj - 1; ok++) {
            if (is_f_order) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                     /* not contiguous enough */
                    return 0;  // 如果不是足够连续的，返回0
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;  // 如果不是足够连续的，返回0
                }
            }
        }

        /* Calculate new strides for all axes currently worked with */
        if (is_f_order) {
            newstrides[ni] = oldstrides[oi];  // 如果是Fortran顺序，则将新步幅设置为旧步幅
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];  // 对于后续的轴，根据前一个轴的步幅和维度计算新步幅
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];  // 如果是C顺序，则将最后一个新步幅设置为最后一个旧步幅
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];  // 对于前面的轴，根据后一个轴的步幅和维度计算新步幅
            }
        }
        ni = nj++;  // 更新新维度索引并增加下一个新维度索引
        oi = oj++;  // 更新旧维度索引并增加下一个旧维度索引
    }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
    if (ni >= 1) {
        last_stride = newstrides[ni - 1];  // 如果新维度索引大于等于1，设置最后一个步幅为前一个新步幅
    }
    else {
        last_stride = PyArray_ITEMSIZE(self);  // 否则，设置最后一个步幅为数组中的每个元素的大小
    }
    if (is_f_order) {
        last_stride *= newdims[ni - 1];  // 如果是Fortran顺序，根据最后一个新维度调整最后一个步幅
    }
    for (nk = ni; nk < newnd; nk++) {
        newstrides[nk] = last_stride;  // 设置新维度中剩余轴的步幅为最后一个步幅
    }

    return 1;  // 返回成功
}

static void
raise_reshape_size_mismatch(PyArray_Dims *newshape, PyArrayObject *arr)
{
    // 将新形状转换为字符串表示
    PyObject *tmp = convert_shape_to_string(newshape->len, newshape->ptr, "");
    // 如果转换成功
    if (tmp != NULL) {
        // 抛出格式化的 ValueError 异常
        PyErr_Format(PyExc_ValueError,
                "cannot reshape array of size %zd into shape %S",
                PyArray_SIZE(arr), tmp);
        // 释放临时对象
        Py_DECREF(tmp);
    }
}

static int
_fix_unknown_dimension(PyArray_Dims *newshape, PyArrayObject *arr)
{
    npy_intp *dimensions;
    npy_intp s_original = PyArray_SIZE(arr);
    npy_intp i_unknown, s_known;
    int i, n;

    dimensions = newshape->ptr;
    n = newshape->len;
    s_known = 1;
    i_unknown = -1;

    // 遍历新形状的维度数组
    for (i = 0; i < n; i++) {
        // 处理未知维度的情况
        if (dimensions[i] < 0) {
            // 如果已经有未知维度
            if (i_unknown == -1) {
                i_unknown = i;
            }
            else {
                // 抛出值错误异常，只能指定一个未知维度
                PyErr_SetString(PyExc_ValueError,
                                "can only specify one unknown dimension");
                return -1;
            }
        }
        else if (npy_mul_sizes_with_overflow(&s_known, s_known,
                                            dimensions[i])) {
            // 如果计算维度乘积溢出，抛出重塑大小不匹配异常
            raise_reshape_size_mismatch(newshape, arr);
            return -1;
        }
    }

    // 处理存在未知维度的情况
    if (i_unknown >= 0) {
        // 如果已知维度为 0 或者原始数组大小不能整除已知维度
        if (s_known == 0 || s_original % s_known != 0) {
            // 抛出重塑大小不匹配异常
            raise_reshape_size_mismatch(newshape, arr);
            return -1;
        }
        // 计算未知维度的值
        dimensions[i_unknown] = s_original / s_known;
    }
    else {
        // 如果没有未知维度，检查原始大小是否等于已知维度
        if (s_original != s_known) {
            // 抛出重塑大小不匹配异常
            raise_reshape_size_mismatch(newshape, arr);
            return -1;
        }
    }
    return 0;
}

/*NUMPY_API
 *
 * return a new view of the array object with all of its unit-length
 * dimensions squeezed out if needed, otherwise
 * return the same array.
 */
NPY_NO_EXPORT PyObject *
PyArray_Squeeze(PyArrayObject *self)
{
    PyArrayObject *ret;
    npy_bool unit_dims[NPY_MAXDIMS];
    int idim, ndim, any_ones;
    npy_intp *shape;

    // 获取数组的维度数和形状
    ndim = PyArray_NDIM(self);
    shape = PyArray_SHAPE(self);

    any_ones = 0;
    // 遍历数组的每个维度
    for (idim = 0; idim < ndim; ++idim) {
        // 如果维度为 1，则表示为单位维度
        if (shape[idim] == 1) {
            unit_dims[idim] = 1;
            any_ones = 1;
        }
        else {
            unit_dims[idim] = 0;
        }
    }

    /* 如果没有单位维度需要挤压，直接返回原数组 */
    if (!any_ones) {
        // 增加原数组的引用计数并返回
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // 创建数组的视图，去除单位维度
    ret = (PyArrayObject *)PyArray_View(self, NULL, &PyArray_Type);
    // 如果创建视图失败，返回空指针
    if (ret == NULL) {
        return NULL;
    }

    // 在视图中移除指定的单位维度
    PyArray_RemoveAxesInPlace(ret, unit_dims);

    /*
     * 如果 self 不是基类 ndarray，调用其 __array_wrap__ 方法
     */
    if (Py_TYPE(self) != &PyArray_Type) {
        // 应用简单的数组包装方法，并返回包装后的对象
        PyObject *wrapped = npy_apply_wrap_simple(self, ret);
        Py_DECREF(ret);
        return wrapped;
    }

    // 返回视图对象
    return (PyObject *)ret;
}

/*
 * Just like PyArray_Squeeze, but allows the caller to select
 * a subset of the size-one dimensions to squeeze out.
 */
/*
 * Squeeze selected axes in a NumPy array.
 */
NPY_NO_EXPORT PyObject *
PyArray_SqueezeSelected(PyArrayObject *self, npy_bool *axis_flags)
{
    PyArrayObject *ret;
    int idim, ndim, any_ones;
    npy_intp *shape;

    ndim = PyArray_NDIM(self);      // 获取数组的维度数
    shape = PyArray_SHAPE(self);    // 获取数组的形状

    /* Verify that the axes requested are all of size one */
    any_ones = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim] != 0) {    // 如果axis_flags[idim]为真
            if (shape[idim] == 1) {     // 如果该轴的大小为1
                any_ones = 1;           // 设置标志位，表示存在可以压缩的轴
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                        "cannot select an axis to squeeze out "
                        "which has size not equal to one");  // 抛出值错误异常，说明无法压缩大小不为1的轴
                return NULL;    // 返回空指针
            }
        }
    }

    /* If there were no axes to squeeze out, return the same array */
    if (!any_ones) {    // 如果没有需要压缩的轴
        Py_INCREF(self);    // 增加对self的引用计数
        return (PyObject *)self;    // 返回self数组的Python对象指针
    }

    ret = (PyArrayObject *)PyArray_View(self, NULL, &PyArray_Type);    // 创建一个self的视图
    if (ret == NULL) {    // 如果视图创建失败
        return NULL;    // 返回空指针
    }

    PyArray_RemoveAxesInPlace(ret, axis_flags);    // 在ret中就地移除指定的轴

    /*
     * If self isn't not a base class ndarray, call its
     * __array_wrap__ method
     */
    if (Py_TYPE(self) != &PyArray_Type) {    // 如果self不是基类ndarray
        PyObject *wrapped = npy_apply_wrap_simple(self, ret);    // 调用self的__array_wrap__方法
        Py_DECREF(ret);    // 减少对ret的引用计数
        return wrapped;    // 返回调用结果
    }

    return (PyObject *)ret;    // 返回视图ret的Python对象指针
}

/*
 * SwapAxes in a NumPy array.
 */
/*NUMPY_API
 * SwapAxes
 */
NPY_NO_EXPORT PyObject *
PyArray_SwapAxes(PyArrayObject *ap, int a1, int a2)
{
    PyArray_Dims new_axes;
    npy_intp dims[NPY_MAXDIMS];
    int n = PyArray_NDIM(ap);
    int i;

    if (check_and_adjust_axis_msg(&a1, n, npy_interned_str.axis1) < 0) {    // 检查并调整轴a1
        return NULL;    // 返回空指针
    }
    if (check_and_adjust_axis_msg(&a2, n, npy_interned_str.axis2) < 0) {    // 检查并调整轴a2
        return NULL;    // 返回空指针
    }

    for (i = 0; i < n; ++i) {    // 设置初始维度顺序
        dims[i] = i;
    }
    dims[a1] = a2;    // 交换轴a1和a2
    dims[a2] = a1;

    new_axes.ptr = dims;    // 设置新轴顺序
    new_axes.len = n;

    return PyArray_Transpose(ap, &new_axes);    // 返回转置后的数组
}


/*
 * Transpose a NumPy array.
 */
/*NUMPY_API
 * Return Transpose.
 */
NPY_NO_EXPORT PyObject *
PyArray_Transpose(PyArrayObject *ap, PyArray_Dims *permute)
{
    npy_intp *axes;
    int i, n;
    int permutation[NPY_MAXDIMS], reverse_permutation[NPY_MAXDIMS];
    PyArrayObject *ret = NULL;
    int flags;

    if (permute == NULL) {    // 如果未指定置换顺序
        n = PyArray_NDIM(ap);    // 获取数组的维度数
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;    // 设置默认逆序置换顺序
        }
    }

    // 更多代码...
}
    else {
        // 获取置换数组的长度和指针
        n = permute->len;
        axes = permute->ptr;
        // 如果长度与数组维度不匹配，设置错误信息并返回空
        if (n != PyArray_NDIM(ap)) {
            PyErr_SetString(PyExc_ValueError,
                            "axes don't match array");
            return NULL;
        }
        // 初始化逆置换数组，全部置为-1
        for (i = 0; i < n; i++) {
            reverse_permutation[i] = -1;
        }
        // 对每个轴进行检查和调整
        for (i = 0; i < n; i++) {
            int axis = axes[i];
            // 检查并调整轴的有效性，若无效则返回空
            if (check_and_adjust_axis(&axis, PyArray_NDIM(ap)) < 0) {
                return NULL;
            }
            // 如果在逆置换数组中发现重复的轴，设置错误信息并返回空
            if (reverse_permutation[axis] != -1) {
                PyErr_SetString(PyExc_ValueError,
                                "repeated axis in transpose");
                return NULL;
            }
            // 更新置换和逆置换数组
            reverse_permutation[axis] = i;
            permutation[i] = axis;
        }
    }

    // 获取数组的标志位
    flags = PyArray_FLAGS(ap);

    /*
     * 分配内存给维度和步幅（但填充不正确），设置描述符，并将数据指向 PyArray_DATA(ap)。
     */
    // 增加描述符的引用计数
    Py_INCREF(PyArray_DESCR(ap));
    // 使用描述符和基础数据创建新的数组对象
    ret = (PyArrayObject *) PyArray_NewFromDescrAndBase(
            Py_TYPE(ap), PyArray_DESCR(ap),
            n, PyArray_DIMS(ap), NULL, PyArray_DATA(ap),
            flags, (PyObject *)ap, (PyObject *)ap);
    // 若创建失败，返回空
    if (ret == NULL) {
        return NULL;
    }

    /* 调整返回数组的维度和步幅 */
    for (i = 0; i < n; i++) {
        // 根据置换数组更新维度
        PyArray_DIMS(ret)[i] = PyArray_DIMS(ap)[permutation[i]];
        // 根据置换数组更新步幅
        PyArray_STRIDES(ret)[i] = PyArray_STRIDES(ap)[permutation[i]];
    }
    // 更新返回数组的标志位
    PyArray_UpdateFlags(ret, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS |
                        NPY_ARRAY_ALIGNED);
    // 返回转置后的数组对象
    return (PyObject *)ret;
/*
 * Return matrix transpose (swap last two dimensions).
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixTranspose(PyArrayObject *ap)
{
    int ndim = PyArray_NDIM(ap); // 获取数组的维度数

    if (ndim < 2) { // 如果维度数小于2，抛出数值错误异常
        PyErr_SetString(PyExc_ValueError,
                        "matrix transpose with ndim < 2 is undefined");
        return NULL;
    }
    return PyArray_SwapAxes(ap, ndim - 2, ndim - 1); // 返回交换了最后两个维度的数组对象
}

/*
 * Sorts items so stride is descending, because C-order
 * is the default in the face of ambiguity.
 */
static int _npy_stride_sort_item_comparator(const void *a, const void *b)
{
    npy_intp astride = ((const npy_stride_sort_item *)a)->stride, // 获取结构体元素 a 的步幅
            bstride = ((const npy_stride_sort_item *)b)->stride; // 获取结构体元素 b 的步幅

    /* Sort the absolute value of the strides */
    if (astride < 0) { // 如果步幅为负数，取其绝对值
        astride = -astride;
    }
    if (bstride < 0) { // 如果步幅为负数，取其绝对值
        bstride = -bstride;
    }

    if (astride == bstride) { // 如果步幅相等，则按照排列顺序比较
        /*
         * Make the qsort stable by next comparing the perm order.
         * (Note that two perm entries will never be equal)
         */
        npy_intp aperm = ((const npy_stride_sort_item *)a)->perm, // 获取结构体元素 a 的排列顺序
                bperm = ((const npy_stride_sort_item *)b)->perm; // 获取结构体元素 b 的排列顺序
        return (aperm < bperm) ? -1 : 1; // 返回根据排列顺序比较的结果
    }
    if (astride > bstride) { // 如果 a 的步幅大于 b 的步幅，返回 -1
        return -1;
    }
    return 1; // 否则返回 1
}

/*NUMPY_API
 *
 * This function populates the first ndim elements
 * of strideperm with sorted descending by their absolute values.
 * For example, the stride array (4, -2, 12) becomes
 * [(2, 12), (0, 4), (1, -2)].
 */
NPY_NO_EXPORT void
PyArray_CreateSortedStridePerm(int ndim, npy_intp const *strides,
                        npy_stride_sort_item *out_strideperm)
{
    int i;

    /* Set up the strideperm values */
    for (i = 0; i < ndim; ++i) { // 初始化 strideperm 结构体数组，设置排列顺序和步幅
        out_strideperm[i].perm = i; // 设置排列顺序为当前索引 i
        out_strideperm[i].stride = strides[i]; // 设置步幅为 strides 数组中对应的值
    }

    /* Sort them */
    qsort(out_strideperm, ndim, sizeof(npy_stride_sort_item),
                                    &_npy_stride_sort_item_comparator); // 使用 qsort 对 strideperm 结构体数组进行排序
}

static inline npy_intp
s_intp_abs(npy_intp x)
{
    return (x < 0) ? -x : x; // 返回 x 的绝对值
}

/*
 * Creates a sorted stride perm matching the KEEPORDER behavior
 * of the NpyIter object. Because this operates based on multiple
 * input strides, the 'stride' member of the npy_stride_sort_item
 * would be useless and we simply argsort a list of indices instead.
 *
 * The caller should have already validated that 'ndim' matches for
 * every array in the arrays list.
 */
NPY_NO_EXPORT void
PyArray_CreateMultiSortedStridePerm(int narrays, PyArrayObject **arrays,
                        int ndim, int *out_strideperm)
{
    int i0, i1, ipos, ax_j0, ax_j1, iarrays;

    /* Initialize the strideperm values to the identity. */
    for (i0 = 0; i0 < ndim; ++i0) { // 初始化 strideperm 数组，使其对应于标识
        out_strideperm[i0] = i0; // 设置当前索引处的值为当前索引 i0
    }
}
    /*
     * 这段代码实现了一种定制的稳定插入排序，用于NpyIter对象，但以与迭代器相反的顺序排序。
     * 迭代器按照最小步长到最大步长的顺序排序（Fortran顺序），而这里按照最大步长到最小步长的顺序排序（C顺序）。
     */
    for (i0 = 1; i0 < ndim; ++i0) {

        ipos = i0;
        ax_j0 = out_strideperm[i0];

        for (i1 = i0 - 1; i1 >= 0; --i1) {
            int ambig = 1, shouldswap = 0;

            ax_j1 = out_strideperm[i1];

            for (iarrays = 0; iarrays < narrays; ++iarrays) {
                // 检查当前轴上数组的形状是否不是1，如果都不是1，则可能需要交换
                if (PyArray_SHAPE(arrays[iarrays])[ax_j0] != 1 &&
                            PyArray_SHAPE(arrays[iarrays])[ax_j1] != 1) {
                    // 比较当前两个步长的绝对值大小，决定是否需要交换
                    if (s_intp_abs(PyArray_STRIDES(arrays[iarrays])[ax_j0]) <=
                            s_intp_abs(PyArray_STRIDES(arrays[iarrays])[ax_j1])) {
                        /*
                         * 即使还不是明确的歧义情况，也设置为需要交换，
                         * 因为在不同操作数之间的冲突情况下，C顺序优先。
                         */
                        shouldswap = 0;
                    }
                    else {
                        /* 只有在仍然存在歧义时才设置为需要交换 */
                        if (ambig) {
                            shouldswap = 1;
                        }
                    }

                    /*
                     * 已经进行了比较，因此不再是歧义的
                     */
                    ambig = 0;
                }
            }
            /*
             * 如果比较是明确的，要么将 'ipos' 移动到 'i1'，要么停止寻找插入点
             */
            if (!ambig) {
                if (shouldswap) {
                    ipos = i1;
                }
                else {
                    break;
                }
            }
        }

        /* 将 out_strideperm[i0] 插入到正确的位置 */
        if (ipos != i0) {
            for (i1 = i0; i1 > ipos; --i1) {
                out_strideperm[i1] = out_strideperm[i1-1];
            }
            out_strideperm[ipos] = ax_j0;
        }
    }
/*NUMPY_API
 * Ravel
 * Returns a contiguous array
 */
NPY_NO_EXPORT PyObject *
PyArray_Ravel(PyArrayObject *arr, NPY_ORDER order)
{
    // 新维度结构体，用于重塑数组形状
    PyArray_Dims newdim = {NULL,1};
    // 初始的尺寸值为-1，用于自动计算新形状
    npy_intp val[1] = {-1};

    newdim.ptr = val;

    // 如果指定保持原始顺序
    if (order == NPY_KEEPORDER) {
        /* This handles some corner cases, such as 0-d arrays as well */
        // 处理一些特殊情况，比如0维数组
        if (PyArray_IS_C_CONTIGUOUS(arr)) {
            order = NPY_CORDER; // 如果是C连续的，使用C顺序
        }
        else if (PyArray_IS_F_CONTIGUOUS(arr)) {
            order = NPY_FORTRANORDER; // 如果是Fortran连续的，使用Fortran顺序
        }
    }
    // 如果指定任意顺序
    else if (order == NPY_ANYORDER) {
        // 根据数组是否Fortran连续决定顺序
        order = PyArray_ISFORTRAN(arr) ? NPY_FORTRANORDER : NPY_CORDER;
    }

    // 如果顺序是C顺序且数组是C连续的
    if (order == NPY_CORDER && PyArray_IS_C_CONTIGUOUS(arr)) {
        // 返回一个新的数组，按指定的C顺序重塑
        return PyArray_Newshape(arr, &newdim, NPY_CORDER);
    }
    // 如果顺序是Fortran顺序且数组是Fortran连续的
    else if (order == NPY_FORTRANORDER && PyArray_IS_F_CONTIGUOUS(arr)) {
        // 返回一个新的数组，按指定的Fortran顺序重塑
        return PyArray_Newshape(arr, &newdim, NPY_FORTRANORDER);
    }
    /* For KEEPORDER, check if we can make a flattened view */
    // 对于保持原始顺序，检查是否可以创建扁平视图
    else if (order == NPY_KEEPORDER) {
        // 创建排序后的步幅列表
        npy_stride_sort_item strideperm[NPY_MAXDIMS];
        npy_intp stride;
        int i, ndim = PyArray_NDIM(arr);

        // 创建按步幅排序的排列
        PyArray_CreateSortedStridePerm(PyArray_NDIM(arr),
                                PyArray_STRIDES(arr), strideperm);

        // 输出数组必须是连续的，因此第一个步幅是固定的
        stride = PyArray_ITEMSIZE(arr);

        // 从最后一个维度开始检查
        for (i = ndim-1; i >= 0; --i) {
            // 大小为1的维度不重要
            if (PyArray_DIM(arr, strideperm[i].perm) == 1) {
                continue;
            }
            // 如果步幅不匹配，中断循环
            if (strideperm[i].stride != stride) {
                break;
            }
            // 更新步幅
            stride *= PyArray_DIM(arr, strideperm[i].perm);
        }

        // 如果所有步幅匹配连续布局，则返回视图
        if (i < 0) {
            stride = PyArray_ITEMSIZE(arr);
            val[0] = PyArray_SIZE(arr);

            // 增加数组的引用计数并返回新数组
            Py_INCREF(PyArray_DESCR(arr));
            return PyArray_NewFromDescrAndBase(
                    Py_TYPE(arr), PyArray_DESCR(arr),
                    1, val, &stride, PyArray_BYTES(arr),
                    PyArray_FLAGS(arr), (PyObject *)arr, (PyObject *)arr);
        }
    }

    // 否则，返回数组的扁平视图
    return PyArray_Flatten(arr, order);
}
    }
    // 将 ret 转换为 PyObject 指针并返回
    return (PyObject *)ret;
/*NUMPY_API
 *
 * Removes the axes flagged as True from the array,
 * modifying it in place. If an axis flagged for removal
 * has a shape entry bigger than one, this effectively selects
 * index zero for that axis.
 *
 * WARNING: If an axis flagged for removal has a shape equal to zero,
 *          the array will point to invalid memory. The caller must
 *          validate this!
 *          If an axis flagged for removal has a shape larger than one,
 *          the aligned flag (and in the future the contiguous flags),
 *          may need explicit update.
 *
 * For example, this can be used to remove the reduction axes
 * from a reduction result once its computation is complete.
 */
NPY_NO_EXPORT void
PyArray_RemoveAxesInPlace(PyArrayObject *arr, const npy_bool *flags)
{
    // 转换数组对象的字段到具体类型
    PyArrayObject_fields *fa = (PyArrayObject_fields *)arr;
    // 获取数组的维度和步长信息
    npy_intp *shape = fa->dimensions, *strides = fa->strides;
    int idim, ndim = fa->nd, idim_out = 0;

    /* Compress the dimensions and strides */
    // 压缩维度和步长信息
    for (idim = 0; idim < ndim; ++idim) {
        // 如果标志为真，则跳过该维度
        if (!flags[idim]) {
            // 将非跳过的维度信息复制到输出的维度和步长数组中
            shape[idim_out] = shape[idim];
            strides[idim_out] = strides[idim];
            // 更新输出维度索引
            ++idim_out;
        }
    }

    /* The final number of dimensions */
    // 更新数组对象的维度数
    fa->nd = idim_out;

    /* NOTE: This is only necessary if a dimension with size != 1 was removed */
    // 如果移除了大小不为1的维度，则需要更新连续性标志
    PyArray_UpdateFlags(arr, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
}
```