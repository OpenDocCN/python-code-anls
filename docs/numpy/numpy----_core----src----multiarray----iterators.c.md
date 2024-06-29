# `.\numpy\numpy\_core\src\multiarray\iterators.c`

```
/*
 * 定义宏，禁用过时的 NumPy API 并设置为当前版本
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * 定义宏，用于 MultiArray 模块
 */
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T_CLEAN 宏定义
 */
#define PY_SSIZE_T_CLEAN

/*
 * 包含 Python.h 头文件，这是所有 Python C API 的入口点
 */
#include <Python.h>

/*
 * 包含 structmember.h 头文件，用于定义结构体成员访问的宏
 */
#include <structmember.h>

/*
 * 包含 NumPy 的 arrayobject.h 头文件，用于 NumPy 数组对象的操作
 */
#include "numpy/arrayobject.h"

/*
 * 包含 NumPy 的 arrayscalars.h 头文件，定义数组标量对象的相关操作
 */
#include "numpy/arrayscalars.h"

/*
 * 包含 NumPy 的 npy_math.h 头文件，提供数学运算相关的宏和函数声明
 */
#include "numpy/npy_math.h"

/*
 * 包含 numpy 的配置文件 npy_config.h
 */
#include "npy_config.h"

/*
 * 包含 NumPy 的 arrayobject.h 头文件，定义数组对象的操作
 */
#include "arrayobject.h"

/*
 * 包含 iterators.h 头文件，定义了与迭代器相关的操作
 */
#include "iterators.h"

/*
 * 包含 ctors.h 头文件，用于定义构造函数和析构函数
 */
#include "ctors.h"

/*
 * 包含 common.h 头文件，提供了通用的宏和函数定义
 */
#include "common.h"

/*
 * 包含 conversion_utils.h 头文件，包含了数据类型转换的工具函数
 */
#include "conversion_utils.h"

/*
 * 包含 dtypemeta.h 头文件，用于处理数据类型元信息
 */
#include "dtypemeta.h"

/*
 * 包含 array_coercion.h 头文件，包含了数组强制转换相关的操作
 */
#include "array_coercion.h"

/*
 * 包含 item_selection.h 头文件，定义了数组元素选择的操作
 */
#include "item_selection.h"

/*
 * 包含 lowlevel_strided_loops.h 头文件，提供了低级别的分块循环操作
 */
#include "lowlevel_strided_loops.h"

/*
 * 包含 array_assign.h 头文件，定义了数组赋值的操作
 */
#include "array_assign.h"

/*
 * 定义宏 NEWAXIS_INDEX 为 -1，用于表示新轴索引
 */
#define NEWAXIS_INDEX -1

/*
 * 定义宏 ELLIPSIS_INDEX 为 -2，用于表示省略号索引
 */
#define ELLIPSIS_INDEX -2

/*
 * 定义宏 SINGLE_INDEX 为 -3，用于表示单一索引
 */
#define SINGLE_INDEX -3

/*
 * 尝试将 Python 对象 o 转换为 npy_intp 类型的索引值 v，不设置异常
 * 如果转换成功返回 1，否则返回 0
 */
static int
coerce_index(PyObject *o, npy_intp *v)
{
    *v = PyArray_PyIntAsIntp(o);

    if ((*v) == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

/*
 * 将索引元组的一个元素转换为步长和步数，返回起始索引
 * 非切片操作通过 n_steps 返回 NEWAXIS_INDEX、ELLIPSIS_INDEX 或 SINGLE_INDEX
 */
NPY_NO_EXPORT npy_intp
parse_index_entry(PyObject *op, npy_intp *step_size,
                  npy_intp *n_steps, npy_intp max,
                  int axis, int check_index)
{
    npy_intp i;

    if (op == Py_None) {
        *n_steps = NEWAXIS_INDEX;
        i = 0;
    }
    else if (op == Py_Ellipsis) {
        *n_steps = ELLIPSIS_INDEX;
        i = 0;
    }
    else if (PySlice_Check(op)) {
        npy_intp stop;
        if (PySlice_GetIndicesEx(op, max, &i, &stop, step_size, n_steps) < 0) {
            goto fail;
        }
        if (*n_steps <= 0) {
            *n_steps = 0;
            *step_size = 1;
            i = 0;
        }
    }
    else if (coerce_index(op, &i)) {
        *n_steps = SINGLE_INDEX;
        *step_size = 0;
        if (check_index) {
            if (check_and_adjust_index(&i, max, axis, NULL) < 0) {
                goto fail;
            }
        }
    }
    else {
        PyErr_SetString(PyExc_IndexError,
                        "each index entry must be either a "
                        "slice, an integer, Ellipsis, or "
                        "newaxis");
        goto fail;
    }
    return i;

 fail:
    return -1;
}

/*
 * 根据迭代器和坐标获取简单迭代器的数据指针
 */
static char*
get_ptr_simple(PyArrayIterObject* iter, const npy_intp *coordinates)
{
    npy_intp i;
    char *ret;

    ret = PyArray_DATA(iter->ao);

    for(i = 0; i < PyArray_NDIM(iter->ao); ++i) {
        ret += coordinates[i] * iter->strides[i];
    }

    return ret;
}
/*
 * This is common initialization code between PyArrayIterObject and
 * PyArrayNeighborhoodIterObject
 *
 * Steals a reference to the array object which gets removed at deallocation,
 * if the iterator is allocated statically and its dealloc not called, it
 * can be thought of as borrowing the reference.
 */
NPY_NO_EXPORT void
PyArray_RawIterBaseInit(PyArrayIterObject *it, PyArrayObject *ao)
{
    int nd, i;

    nd = PyArray_NDIM(ao);  /* 获取数组对象 ao 的维度数 */
    /* The legacy iterator only supports 32 dimensions */
    assert(nd <= NPY_MAXDIMS_LEGACY_ITERS);  /* 断言：维度数不超过 NPY_MAXDIMS_LEGACY_ITERS */
    PyArray_UpdateFlags(ao, NPY_ARRAY_C_CONTIGUOUS);  /* 更新数组对象 ao 的标志位，表示其是否是 C 连续的数组 */

    if (PyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;  /* 如果 ao 是 C 连续的，则设置迭代器 it 的 contiguous 属性为 1 */
    }
    else {
        it->contiguous = 0;  /* 否则设置为 0 */
    }

    it->ao = ao;  /* 设置迭代器 it 的 ao 属性为输入的数组对象 ao */
    it->size = PyArray_SIZE(ao);  /* 设置迭代器 it 的 size 属性为数组 ao 的总元素数 */
    it->nd_m1 = nd - 1;  /* 设置迭代器 it 的 nd_m1 属性为数组 ao 的维度数减一 */

    if (nd != 0) {
        it->factors[nd-1] = 1;  /* 初始化迭代器 it 的 factors 数组，最后一个元素为 1 */
    }

    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = PyArray_DIMS(ao)[i] - 1;  /* 设置迭代器 it 的 dims_m1 数组，存储数组 ao 每个维度的上限 */
        it->strides[i] = PyArray_STRIDES(ao)[i];  /* 设置迭代器 it 的 strides 数组，存储数组 ao 每个维度的步长 */
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];  /* 计算迭代器 it 的 backstrides 数组，表示反向步长 */
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * PyArray_DIMS(ao)[nd-i];  /* 计算迭代器 it 的 factors 数组，用于索引转换 */
        }
        it->bounds[i][0] = 0;  /* 设置迭代器 it 的 bounds 数组，表示每个维度的下限 */
        it->bounds[i][1] = PyArray_DIMS(ao)[i] - 1;  /* 设置迭代器 it 的 bounds 数组，表示每个维度的上限 */
        it->limits[i][0] = 0;  /* 设置迭代器 it 的 limits 数组，表示每个维度的限制下限 */
        it->limits[i][1] = PyArray_DIMS(ao)[i] - 1;  /* 设置迭代器 it 的 limits 数组，表示每个维度的限制上限 */
        it->limits_sizes[i] = it->limits[i][1] - it->limits[i][0] + 1;  /* 计算迭代器 it 的 limits_sizes 数组，表示限制的尺寸 */
    }

    it->translate = &get_ptr_simple;  /* 设置迭代器 it 的 translate 属性为 get_ptr_simple 函数的指针 */
    PyArray_ITER_RESET(it);  /* 调用 PyArray_ITER_RESET 函数初始化迭代器 it */

    return;
}

static void
array_iter_base_dealloc(PyArrayIterObject *it)
{
    Py_XDECREF(it->ao);  /* 释放迭代器 it 的数组对象 ao 的引用 */
}

/*NUMPY_API
 * Get Iterator.
 */
NPY_NO_EXPORT PyObject *
PyArray_IterNew(PyObject *obj)
{
    /*
     * Note that internally PyArray_RawIterBaseInit may be called directly on a
     * statically allocated PyArrayIterObject.
     */
    PyArrayIterObject *it;
    PyArrayObject *ao;

    if (!PyArray_Check(obj)) {
        PyErr_BadInternalCall();  /* 如果输入对象不是数组对象，则引发错误 */
        return NULL;
    }

    ao = (PyArrayObject *)obj;  /* 将输入对象强制转换为数组对象 ao */

    if (PyArray_NDIM(ao) > NPY_MAXDIMS_LEGACY_ITERS) {
        PyErr_Format(PyExc_RuntimeError,
                "this function only supports up to 32 dimensions but "
                "the array has %d.", PyArray_NDIM(ao));  /* 如果数组对象 ao 的维度数超过限制，引发运行时错误 */
        return NULL;
    }

    it = (PyArrayIterObject *)PyArray_malloc(sizeof(PyArrayIterObject));  /* 分配内存以存储 PyArrayIterObject 结构 */
    PyObject_Init((PyObject *)it, &PyArrayIter_Type);  /* 初始化 PyObject 结构体，设置类型为 PyArrayIter_Type */
    /* it = PyObject_New(PyArrayIterObject, &PyArrayIter_Type); */
    if (it == NULL) {
        return NULL;  /* 如果内存分配失败，则返回 NULL */
    }

    Py_INCREF(ao);  /* 增加数组对象 ao 的引用计数，因为 PyArray_RawIterBaseInit 会偷走该引用 */
    PyArray_RawIterBaseInit(it, ao);  /* 初始化迭代器 it，并传入数组对象 ao */
    return (PyObject *)it;  /* 返回初始化后的迭代器 it */
}

/*NUMPY_API
 * Get Iterator broadcast to a particular shape
 */
NPY_NO_EXPORT PyObject *
PyArray_BroadcastToShape(PyObject *obj, npy_intp *dims, int nd)
{
    PyArrayIterObject *it;
    int i, diff, j, compat, k;
    PyArrayObject *ao = (PyArrayObject *)obj;

    if (PyArray_NDIM(ao) > nd) {
        goto err;  /* 如果输入数组对象 ao 的维度数大于指定的维度数 nd，跳转到错误处理标签 */
    }

    compat = 1;  /* 兼容标志置为 1 */
    diff = j = nd - PyArray_NDIM(ao);  /* 计算输入数组对象 ao 和指定维度数 nd 的差值 */
    # 对数组对象 ao 进行迭代，检查每一个维度的大小是否兼容
    for (i = 0; i < PyArray_NDIM(ao); i++, j++) {
        # 如果数组 ao 的当前维度大小为 1，则跳过
        if (PyArray_DIMS(ao)[i] == 1) {
            continue;
        }
        # 如果数组 ao 的当前维度大小与给定的 dims 数组中的大小不匹配，则设置 compat 为 0 并跳出循环
        if (PyArray_DIMS(ao)[i] != dims[j]) {
            compat = 0;
            break;
        }
    }
    # 如果不兼容，则跳转到错误处理部分
    if (!compat) {
        goto err;
    }
    # 分配内存并初始化一个 PyArrayIterObject 迭代器对象
    it = (PyArrayIterObject *)PyArray_malloc(sizeof(PyArrayIterObject));
    if (it == NULL) {
        return NULL;
    }
    # 初始化迭代器对象，设定其类型为 PyArrayIter_Type
    PyObject_Init((PyObject *)it, &PyArrayIter_Type);

    # 更新数组 ao 的标志位，表示它是 C 连续存储的数组
    PyArray_UpdateFlags(ao, NPY_ARRAY_C_CONTIGUOUS);
    # 如果数组 ao 是 C 连续存储的，则设置迭代器对象的 contiguous 属性为 1，否则为 0
    if (PyArray_ISCONTIGUOUS(ao)) {
        it->contiguous = 1;
    }
    else {
        it->contiguous = 0;
    }
    # 增加数组 ao 的引用计数
    Py_INCREF(ao);
    # 将数组 ao 赋值给迭代器对象的 ao 属性
    it->ao = ao;
    # 计算迭代器对象的总大小
    it->size = PyArray_MultiplyList(dims, nd);
    # 设置迭代器对象的最大维度数减一
    it->nd_m1 = nd - 1;
    # 如果维度数不为 0，则初始化 factors 数组中的最后一个元素为 1
    if (nd != 0) {
        it->factors[nd-1] = 1;
    }
    # 遍历各个维度，设置相关属性
    for (i = 0; i < nd; i++) {
        # 设置 dims_m1 数组的值为 dims[i] - 1
        it->dims_m1[i] = dims[i] - 1;
        # 计算 k 值，用于检查是否为 C 连续存储的数组
        k = i - diff;
        # 如果 k 小于 0 或者数组 ao 在索引 k 处的维度大小与 dims[i] 不匹配，则设置非连续存储并将 strides[i] 设置为 0
        if ((k < 0) || PyArray_DIMS(ao)[k] != dims[i]) {
            it->contiguous = 0;
            it->strides[i] = 0;
        }
        else {
            # 否则，将 strides[i] 设置为数组 ao 在索引 k 处的步长值
            it->strides[i] = PyArray_STRIDES(ao)[k];
        }
        # 设置 backstrides[i] 为 strides[i] 乘以 dims_m1[i]，用于反向遍历
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];
        # 如果 i 大于 0，则计算 factors 数组中的对应元素值
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * dims[nd-i];
        }
    }
    # 将迭代器对象重置为起始状态
    PyArray_ITER_RESET(it);
    # 返回迭代器对象的 PyObject 指针类型
    return (PyObject *)it;

 err:
    # 设置错误信息，数组无法广播到正确的形状
    PyErr_SetString(PyExc_ValueError, "array is not broadcastable to "\
                    "correct shape");
    # 返回空指针，表示错误
    return NULL;
/*NUMPY_API
 * Get Iterator that iterates over all but one axis (don't use this with
 * PyArray_ITER_GOTO1D).  The axis will be over-written if negative
 * with the axis having the smallest stride.
 */
NPY_NO_EXPORT PyObject *
PyArray_IterAllButAxis(PyObject *obj, int *inaxis)
{
    PyArrayObject *arr;
    PyArrayIterObject *it;
    int axis;

    // 检查输入参数是否为 ndarray 类型
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_ValueError,
                "Numpy IterAllButAxis requires an ndarray");
        return NULL;
    }
    // 将输入参数转换为 PyArrayObject 类型
    arr = (PyArrayObject *)obj;

    // 创建一个新的迭代器对象
    it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
    if (it == NULL) {
        return NULL;
    }
    // 如果数组是零维的，直接返回迭代器对象
    if (PyArray_NDIM(arr)==0) {
        return (PyObject *)it;
    }
    // 如果输入的轴数小于 0，则选择具有最小步幅的轴
    if (*inaxis < 0) {
        int i, minaxis = 0;
        npy_intp minstride = 0;
        i = 0;
        // 找到第一个非零步幅的轴
        while (minstride == 0 && i < PyArray_NDIM(arr)) {
            minstride = PyArray_STRIDE(arr,i);
            i++;
        }
        // 遍历数组的所有轴，找到步幅最小的轴
        for (i = 1; i < PyArray_NDIM(arr); i++) {
            if (PyArray_STRIDE(arr,i) > 0 &&
                PyArray_STRIDE(arr, i) < minstride) {
                minaxis = i;
                minstride = PyArray_STRIDE(arr,i);
            }
        }
        // 更新输入的轴参数为步幅最小的轴
        *inaxis = minaxis;
    }
    // 获取当前要迭代的轴
    axis = *inaxis;
    // 设置迭代器不是连续的，以便跳过特定轴
    it->contiguous = 0;
    // 调整迭代器的大小，以便不迭代指定轴上的元素
    if (it->size != 0) {
        it->size /= PyArray_DIM(arr,axis);
    }
    it->dims_m1[axis] = 0;
    it->backstrides[axis] = 0;

    /*
     * (won't fix factors so don't use
     * PyArray_ITER_GOTO1D with this iterator)
     */
    return (PyObject *)it;
}

/*NUMPY_API
 * Adjusts previously broadcasted iterators so that the axis with
 * the smallest sum of iterator strides is not iterated over.
 * Returns dimension which is smallest in the range [0,multi->nd).
 * A -1 is returned if multi->nd == 0.
 *
 * don't use with PyArray_ITER_GOTO1D because factors are not adjusted
 */
NPY_NO_EXPORT int
PyArray_RemoveSmallest(PyArrayMultiIterObject *multi)
{
    PyArrayIterObject *it;
    int i, j;
    int axis;
    npy_intp smallest;
    npy_intp sumstrides[NPY_MAXDIMS];

    // 如果 multi->nd 等于 0，则返回 -1
    if (multi->nd == 0) {
        return -1;
    }
    // 计算每个迭代器轴上步幅的总和
    for (i = 0; i < multi->nd; i++) {
        sumstrides[i] = 0;
        for (j = 0; j < multi->numiter; j++) {
            sumstrides[i] += multi->iters[j]->strides[i];
        }
    }
    // 初始化最小步幅的轴为第一个轴
    axis = 0;
    smallest = sumstrides[0];
    // 找到步幅最小的轴
    for (i = 1; i < multi->nd; i++) {
        if (sumstrides[i] < smallest) {
            axis = i;
            smallest = sumstrides[i];
        }
    }
    // 调整每个迭代器，使其不迭代最小步幅的轴
    for(i = 0; i < multi->numiter; i++) {
        it = multi->iters[i];
        it->contiguous = 0;
        if (it->size != 0) {
            it->size /= (it->dims_m1[axis]+1);
        }
        it->dims_m1[axis] = 0;
        it->backstrides[axis] = 0;
    }
    // 更新 multi 对象的大小为第一个迭代器的大小
    multi->size = multi->iters[0]->size;
    return axis;
}
static void
arrayiter_dealloc(PyArrayIterObject *it)
{
    /*
     * 注意：可以静态分配 PyArrayIterObject，这种情况下不会调用此函数。
     */
    // 调用基类的销毁函数
    array_iter_base_dealloc(it);
    // 释放 PyArrayIterObject 对象
    PyArray_free(it);
}

static Py_ssize_t
iter_length(PyArrayIterObject *self)
{
    // 返回迭代器中的元素数量
    return self->size;
}


static PyArrayObject *
iter_subscript_Bool(PyArrayIterObject *self, PyArrayObject *ind,
                    NPY_cast_info *cast_info)
{
    npy_intp counter, strides;
    int itemsize;
    npy_intp count = 0;
    char *dptr, *optr;
    PyArrayObject *ret;

    if (PyArray_NDIM(ind) != 1) {
        // 如果布尔索引数组不是一维的，抛出 ValueError 异常
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array should have 1 dimension");
        return NULL;
    }
    // 获取布尔索引数组的大小
    counter = PyArray_DIMS(ind)[0];
    if (counter > self->size) {
        // 如果布尔索引数量超过迭代器的元素数量，抛出 ValueError 异常
        PyErr_SetString(PyExc_ValueError,
                        "too many boolean indices");
        return NULL;
    }

    // 获取布尔索引数组的步长
    strides = PyArray_STRIDES(ind)[0];
    /* 获取返回数组的大小 */
    // 计算布尔真值的数量
    count = count_boolean_trues(PyArray_NDIM(ind), PyArray_DATA(ind),
                                PyArray_DIMS(ind), PyArray_STRIDES(ind));
    // 获取迭代器的元素大小
    itemsize = PyArray_ITEMSIZE(self->ao);
    // 复制迭代器的描述符
    PyArray_Descr *dtype = PyArray_DESCR(self->ao);
    Py_INCREF(dtype);
    // 从描述符创建新的数组对象
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                             dtype, 1, &count,
                             NULL, NULL,
                             0, (PyObject *)self->ao);
    if (ret == NULL) {
        return NULL;
    }
    if (count > 0) {
        /* 设置循环 */
        // 设置输出指针
        optr = PyArray_DATA(ret);
        // 获取布尔索引数组的大小
        counter = PyArray_DIMS(ind)[0];
        // 获取布尔索引数组的数据指针
        dptr = PyArray_DATA(ind);
        npy_intp one = 1;
        while (counter--) {
            if (*((npy_bool *)dptr) != 0) {
                char *args[2] = {self->dataptr, optr};
                npy_intp transfer_strides[2] = {itemsize, itemsize};
                // 调用类型转换函数，转换数据并写入输出数组
                if (cast_info->func(&cast_info->context, args, &one,
                                    transfer_strides, cast_info->auxdata) < 0) {
                    return NULL;
                }
                optr += itemsize;
            }
            // 移动布尔索引数组的数据指针
            dptr += strides;
            // 移动迭代器到下一个元素
            PyArray_ITER_NEXT(self);
        }
        // 重置迭代器的位置
        PyArray_ITER_RESET(self);
    }
    return ret;
}

static PyObject *
iter_subscript_int(PyArrayIterObject *self, PyArrayObject *ind,
                   NPY_cast_info *cast_info)
{
    npy_intp num;
    PyArrayObject *ret;
    PyArrayIterObject *ind_it;
    int itemsize;
    char *optr;
    npy_intp counter;
    // 检查输入数组 `ind` 的维度是否为0
    if (PyArray_NDIM(ind) == 0) {
        // 如果是0维数组，获取其数据指针指向的整数值
        num = *((npy_intp *)PyArray_DATA(ind));
        // 检查并调整索引 `num`，确保在有效范围内
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            // 若索引无效，重置迭代器并返回 NULL
            PyArray_ITER_RESET(self);
            return NULL;
        }
        else {
            // 否则，跳转到一维索引 `num` 处
            PyObject *tmp;
            PyArray_ITER_GOTO1D(self, num);
            // 将该位置的数据转换为标量对象
            tmp = PyArray_ToScalar(self->dataptr, self->ao);
            // 重置迭代器并返回临时对象 `tmp`
            PyArray_ITER_RESET(self);
            return tmp;
        }
    }

    // 获取数组 `self->ao` 的数据类型描述符并增加其引用计数
    PyArray_Descr *dtype = PyArray_DESCR(self->ao);
    Py_INCREF(dtype);
    // 从描述符创建一个新的数组对象 `ret`
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                             dtype,
                             PyArray_NDIM(ind),
                             PyArray_DIMS(ind),
                             NULL, NULL,
                             0, (PyObject *)self->ao);
    // 若创建失败，返回 NULL
    if (ret == NULL) {
        return NULL;
    }

    // 获取 `ret` 的数据指针
    optr = PyArray_DATA(ret);
    // 创建 `ind` 的迭代器 `ind_it`
    ind_it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)ind);
    // 若迭代器创建失败，释放 `ret` 并返回 NULL
    if (ind_it == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    // 初始化变量
    npy_intp one = 1;
    itemsize = dtype->elsize;
    counter = ind_it->size;
    // 遍历迭代器中的每个元素
    while (counter--) {
        // 获取当前索引值 `num`
        num = *((npy_intp *)(ind_it->dataptr));
        // 检查并调整索引 `num`，确保在有效范围内
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            // 如果索引无效，清除 `ret` 并跳转到 `finish` 标签
            Py_CLEAR(ret);
            goto finish;
        }
        // 跳转到 `self` 的一维索引 `num` 处
        PyArray_ITER_GOTO1D(self, num);
        // 构建参数数组和传输步长数组，并调用 `cast_info` 中的函数进行数据类型转换
        char *args[2] = {self->dataptr, optr};
        npy_intp transfer_strides[2] = {itemsize, itemsize};
        if (cast_info->func(&cast_info->context, args, &one,
                            transfer_strides, cast_info->auxdata) < 0) {
            // 若转换失败，清除 `ret` 并跳转到 `finish` 标签
            Py_CLEAR(ret);
            goto finish;
        }
        // 更新 `optr` 指向下一个数据位置
        optr += itemsize;
        // 移动 `ind_it` 到下一个元素
        PyArray_ITER_NEXT(ind_it);
    }

 finish:
    // 释放 `ind_it` 迭代器
    Py_DECREF(ind_it);
    // 重置 `self` 迭代器
    PyArray_ITER_RESET(self);
    // 返回 `ret` 对象转换为 PyObject 类型
    return (PyObject *)ret;
}

/* Always returns arrays */
NPY_NO_EXPORT PyObject *
iter_subscript(PyArrayIterObject *self, PyObject *ind)
{
    PyArray_Descr *indtype = NULL;  // 声明一个指向数组描述符的指针变量，初始为NULL
    PyArray_Descr *dtype;  // 声明一个指向数组描述符的指针变量
    npy_intp start, step_size;  // 声明用于存储整数索引起始位置和步长的变量
    npy_intp n_steps;  // 声明用于存储步数的变量
    PyArrayObject *ret;  // 声明一个指向数组对象的指针变量，用于存储返回结果
    char *dptr;  // 声明一个指向字符的指针变量
    int size;  // 声明一个整型变量，用于存储元素大小
    PyObject *obj = NULL;  // 声明一个Python对象指针变量，初始为NULL
    PyObject *new;  // 声明一个Python对象指针变量
    NPY_cast_info cast_info = {.func = NULL};  // 声明一个结构体变量，用于存储转换函数信息，初始值为NULL

    if (ind == Py_Ellipsis) {
        ind = PySlice_New(NULL, NULL, NULL);  // 创建一个新的切片对象，用于索引操作
        obj = iter_subscript(self, ind);  // 递归调用当前函数，处理切片对象作为索引
        Py_DECREF(ind);  // 减少切片对象的引用计数
        return obj;  // 返回处理结果
    }
    if (PyTuple_Check(ind)) {
        int len;  // 声明一个整型变量，用于存储元组长度
        len = PyTuple_GET_SIZE(ind);  // 获取元组的长度
        if (len > 1) {
            goto fail;  // 如果元组长度大于1，则跳转到错误处理标签
        }
        if (len == 0) {
            Py_INCREF(self->ao);  // 增加数组对象的引用计数
            return (PyObject *)self->ao;  // 返回数组对象的Python对象指针
        }
        ind = PyTuple_GET_ITEM(ind, 0);  // 获取元组中的第一个元素作为新的索引对象
    }

    /*
     * Tuples >1d not accepted --- i.e. no newaxis
     * Could implement this with adjusted strides and dimensions in iterator
     * Check for Boolean -- this is first because Bool is a subclass of Int
     */
    PyArray_ITER_RESET(self);  // 重置数组迭代器的状态

    if (PyBool_Check(ind)) {
        int istrue = PyObject_IsTrue(ind);  // 检查布尔值是否为True
        if (istrue == -1) {
            goto fail;  // 如果检查失败，则跳转到错误处理标签
        }
        if (istrue) {
            return PyArray_ToScalar(self->dataptr, self->ao);  // 将数组中的数据转换为标量并返回
        }
        else { /* empty array */
            npy_intp ii = 0;  // 声明一个整型变量，表示空数组的长度
            dtype = PyArray_DESCR(self->ao);  // 获取数组对象的描述符
            Py_INCREF(dtype);  // 增加描述符的引用计数
            ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                                     dtype,
                                     1, &ii,
                                     NULL, NULL, 0,
                                     (PyObject *)self->ao);  // 创建一个新的数组对象
            return (PyObject *)ret;  // 返回新创建的数组对象的Python对象指针
        }
    }

    dtype = PyArray_DESCR(self->ao);  // 获取数组对象的描述符
    size = dtype->elsize;  // 获取数组元素的大小

    /* set up a cast to handle item copying */

    NPY_ARRAYMETHOD_FLAGS transfer_flags = 0;  // 声明一个标志位变量，用于存储数组操作的标志
    npy_intp one = 1;  // 声明一个整型变量，表示步长为1
    /* We can assume the newly allocated output array is aligned */
    int is_aligned = IsUintAligned(self->ao);  // 检查数组是否按照整数对齐
    if (PyArray_GetDTypeTransferFunction(
                is_aligned, size, size, dtype, dtype, 0, &cast_info,
                &transfer_flags) < 0) {
        goto fail;  // 如果获取数据类型转换函数失败，则跳转到错误处理标签
    }

    /* Check for Integer or Slice */
    # 检查索引是否为 PyLong 或 PySlice 对象
    if (PyLong_Check(ind) || PySlice_Check(ind)) {
        # 解析索引，获取起始位置、步长和步数
        start = parse_index_entry(ind, &step_size, &n_steps,
                                  self->size, 0, 1);
        # 解析失败，跳转到错误处理
        if (start == -1) {
            goto fail;
        }
        # 若步数为省略号索引或新轴索引，抛出索引错误并跳转到错误处理
        if (n_steps == ELLIPSIS_INDEX || n_steps == NEWAXIS_INDEX) {
            PyErr_SetString(PyExc_IndexError,
                            "cannot use Ellipsis or newaxes here");
            goto fail;
        }
        # 将迭代器移动到一维数组的起始位置
        PyArray_ITER_GOTO1D(self, start);
        # 若步数为单一索引（整数）
        if (n_steps == SINGLE_INDEX) { /* Integer */
            PyObject *tmp;
            # 将当前位置的数据转换为标量对象
            tmp = PyArray_ToScalar(self->dataptr, self->ao);
            # 重置迭代器到初始状态
            PyArray_ITER_RESET(self);
            # 释放类型转换信息并返回临时对象
            NPY_cast_info_xfree(&cast_info);
            return tmp;
        }
        # 增加引用计数以防止类型被释放
        Py_INCREF(dtype);
        # 根据指定的类型和步数创建新的数组对象
        ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self->ao),
                                 dtype,
                                 1, &n_steps,
                                 NULL, NULL,
                                 0, (PyObject *)self->ao);
        # 创建失败，跳转到错误处理
        if (ret == NULL) {
            goto fail;
        }
        # 获取新数组对象的数据指针
        dptr = PyArray_DATA(ret);
        # 循环处理每一个步数
        while (n_steps--) {
            char *args[2] = {self->dataptr, dptr};
            npy_intp transfer_strides[2] = {size, size};
            # 如果类型转换函数执行失败，跳转到错误处理
            if (cast_info.func(&cast_info.context, args, &one,
                               transfer_strides, cast_info.auxdata) < 0) {
                goto fail;
            }
            # 更新起始位置
            start += step_size;
            # 将迭代器移动到新的一维位置
            PyArray_ITER_GOTO1D(self, start);
            # 更新数据指针位置
            dptr += size;
        }
        # 重置迭代器到初始状态
        PyArray_ITER_RESET(self);
        # 释放类型转换信息
        NPY_cast_info_xfree(&cast_info);
        # 返回新数组对象
        return (PyObject *)ret;
    }

    /* convert to INTP array if Integer array scalar or List */
    # 获取整数类型描述符
    indtype = PyArray_DescrFromType(NPY_INTP);
    # 如果索引是整数数组标量或列表
    if (PyArray_IsScalar(ind, Integer) || PyList_Check(ind)) {
        # 增加类型描述符的引用计数
        Py_INCREF(indtype);
        # 将输入对象转换为任意类型数组
        obj = PyArray_FromAny(ind, indtype, 0, 0, NPY_ARRAY_FORCECAST, NULL);
        # 转换失败，跳转到错误处理
        if (obj == NULL) {
            goto fail;
        }
    }
    else {
        # 增加索引对象的引用计数
        Py_INCREF(ind);
        # 直接使用索引对象
        obj = ind;
    }

    /* Any remaining valid input is an array or has been turned into one */
    # 如果对象不是数组，跳转到错误处理
    if (!PyArray_Check(obj)) {
        goto fail;
    }

    /* Check for Boolean array */
    # 检查是否为布尔数组
    if (PyArray_TYPE((PyArrayObject *)obj) == NPY_BOOL) {
        # 使用布尔数组进行迭代下标操作
        ret = iter_subscript_Bool(self, (PyArrayObject *)obj, &cast_info);
        # 跳转到结束处理
        goto finish;
    }

    /* Only integer arrays left */
    # 只剩下整数数组
    if (!PyArray_ISINTEGER((PyArrayObject *)obj)) {
        # 跳转到错误处理
        goto fail;
    }

    # 增加整数类型描述符的引用计数
    Py_INCREF(indtype);
    # 将对象转换为整数类型数组
    new = PyArray_FromAny(obj, indtype, 0, 0,
                      NPY_ARRAY_FORCECAST | NPY_ARRAY_ALIGNED, NULL);
    # 转换失败，跳转到错误处理
    if (new == NULL) {
        goto fail;
    }
    # 使用整数类型数组进行迭代下标操作
    ret = (PyArrayObject *)iter_subscript_int(self, (PyArrayObject *)new,
                                              &cast_info);
    # 释放临时数组对象
    Py_DECREF(new);

 finish:
    # 释放整数类型描述符的引用
    Py_DECREF(indtype);
    # 释放输入对象的引用
    Py_DECREF(obj);
    # 释放类型转换信息
    NPY_cast_info_xfree(&cast_info);
    # 返回操作结果对象
    return (PyObject *)ret;

 fail:
    # 错误处理标签，用于跳转到错误处理部分
    // 检查是否存在 Python 异常状态
    if (!PyErr_Occurred()) {
        // 如果没有异常，设置一个索引错误异常并设定错误消息
        PyErr_SetString(PyExc_IndexError, "unsupported iterator index");
    }
    // 释放 Python 对象引用，避免内存泄漏
    Py_XDECREF(indtype);
    Py_XDECREF(obj);
    // 释放 NumPy 的类型转换信息资源，包括内存空间的释放
    NPY_cast_info_xfree(&cast_info);

    // 返回 NULL 表示函数执行失败
    return NULL;
static`
# 定义一个静态函数，用于处理布尔索引的赋值操作，将给定的值赋给迭代器指向的位置
static int
iter_ass_sub_Bool(PyArrayIterObject *self, PyArrayObject *ind,
                  PyArrayIterObject *val, NPY_cast_info *cast_info)
{
    npy_intp counter, strides;  // 定义计数器和步长变量
    char *dptr;  // 定义指向布尔索引数组数据的指针

    if (PyArray_NDIM(ind) != 1) {  // 检查布尔索引数组的维度是否为1
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array should have 1 dimension");  // 抛出错误信息
        return -1;  // 返回-1，表示出错
    }

    counter = PyArray_DIMS(ind)[0];  // 获取布尔索引数组的长度
    if (counter > self->size) {  // 检查布尔索引数组长度是否超过迭代器的大小
        PyErr_SetString(PyExc_ValueError,
                        "boolean index array has too many values");  // 抛出错误信息
        return -1;  // 返回-1，表示出错
    }

    strides = PyArray_STRIDES(ind)[0];  // 获取布尔索引数组的步长
    dptr = PyArray_DATA(ind);  // 获取布尔索引数组的数据指针
    PyArray_ITER_RESET(self);  // 重置迭代器的状态
    /* Loop over Boolean array */  // 循环遍历布尔数组
    npy_intp one = 1;  // 定义一个大小为1的整数
    PyArray_Descr *dtype = PyArray_DESCR(self->ao);  // 获取迭代器所操作数组的数据类型描述符
    int itemsize = dtype->elsize;  // 获取数组元素的大小
    npy_intp transfer_strides[2] = {itemsize, itemsize};  // 定义数据传输的步长数组
    while (counter--) {  // 循环，直到计数器减为0
        if (*((npy_bool *)dptr) != 0) {  // 如果布尔数组当前位置的值不为0
            char *args[2] = {val->dataptr, self->dataptr};  // 创建一个参数数组，包含值迭代器和目标迭代器的数据指针
            if (cast_info->func(&cast_info->context, args, &one,
                                transfer_strides, cast_info->auxdata) < 0) {
                return -1;  // 如果类型转换函数返回小于0的值，表示出错，返回-1
            }
            PyArray_ITER_NEXT(val);  // 移动值迭代器到下一个位置
            if (val->index == val->size) {  // 如果值迭代器的索引等于其大小
                PyArray_ITER_RESET(val);  // 重置值迭代器的状态
            }
        }
        dptr += strides;  // 移动布尔索引数组数据指针到下一个位置
        PyArray_ITER_NEXT(self);  // 移动迭代器到下一个位置
    }
    PyArray_ITER_RESET(self);  // 最后重置迭代器的状态
    return 0;  // 返回0，表示操作成功完成
}
    # 循环直到 counter 变为 0
    while (counter--) {
        # 从指针中获取整数值，作为索引 num
        num = *((npy_intp *)(ind_it->dataptr));
        # 检查并调整索引 num，确保在有效范围内，如果不合法则返回 -1
        if (check_and_adjust_index(&num, self->size, -1, NULL) < 0) {
            # 释放索引迭代器对象并返回 -1
            Py_DECREF(ind_it);
            return -1;
        }
        # 将迭代器移动到一维数组中的 num 位置
        PyArray_ITER_GOTO1D(self, num);
        # 准备参数数组，用于类型转换函数调用
        char *args[2] = {val->dataptr, self->dataptr};
        # 调用类型转换函数，将 self 的数据转换为 val 的数据类型
        if (cast_info->func(&cast_info->context, args, &one,
                            transfer_strides, cast_info->auxdata) < 0) {
            # 如果类型转换函数调用失败，释放索引迭代器对象并返回 -1
            Py_DECREF(ind_it);
            return -1;
        }
        # 移动索引迭代器到下一个位置
        PyArray_ITER_NEXT(ind_it);
        # 移动 val 迭代器到下一个位置
        PyArray_ITER_NEXT(val);
        # 如果 val 迭代器的当前索引达到其大小，则重置为起始位置
        if (val->index == val->size) {
            PyArray_ITER_RESET(val);
        }
    }
    # 释放索引迭代器对象
    Py_DECREF(ind_it);
    # 循环执行完毕，返回 0 表示成功
    return 0;
    /*
     * 以下是iter_ass_subscript函数的具体实现，用于处理NumPy数组迭代器的元素赋值操作。
     * 根据ind参数类型的不同，实现不同的赋值逻辑。
     */

    NPY_NO_EXPORT int
    iter_ass_subscript(PyArrayIterObject *self, PyObject *ind, PyObject *val)
    {
        PyArrayObject *arrval = NULL;
        PyArrayIterObject *val_it = NULL;
        PyArray_Descr *type;
        PyArray_Descr *indtype = NULL;
        int retval = -1;
        npy_intp start, step_size;
        npy_intp n_steps;
        PyObject *obj = NULL;
        NPY_cast_info cast_info = {.func = NULL};

        // 如果val为NULL，无法删除迭代器元素，抛出TypeError异常
        if (val == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot delete iterator elements");
            return -1;
        }

        // 确保底层数组可写
        if (PyArray_FailUnlessWriteable(self->ao, "underlying array") < 0)
            return -1;

        // 处理ind为省略号（Ellipsis）的情况，递归调用自身处理
        if (ind == Py_Ellipsis) {
            ind = PySlice_New(NULL, NULL, NULL);
            retval = iter_ass_subscript(self, ind, val);
            Py_DECREF(ind);
            return retval;
        }

        // 处理ind为元组的情况，只取第一个元素处理
        if (PyTuple_Check(ind)) {
            int len;
            len = PyTuple_GET_SIZE(ind);
            if (len > 1) {
                goto finish;
            }
            ind = PyTuple_GET_ITEM(ind, 0);
        }

        // 获取底层数组的描述符
        type = PyArray_DESCR(self->ao);

        /*
         * 检查是否为布尔类型 -- 这是因为
         * 布尔类型是整数类型的子类
         */
        if (PyBool_Check(ind)) {
            retval = 0;
            int istrue = PyObject_IsTrue(ind);
            if (istrue == -1) {
                return -1;
            }
            // 如果布尔值为真，则将val打包到数组中
            if (istrue) {
                retval = PyArray_Pack(
                        PyArray_DESCR(self->ao), self->dataptr, val);
            }
            goto finish;
        }

        // 如果ind为序列或切片对象，则跳过后续处理
        if (PySequence_Check(ind) || PySlice_Check(ind)) {
            goto skip;
        }

        // 尝试将ind转换为npy_intp类型作为起始索引
        start = PyArray_PyIntAsIntp(ind);
        if (error_converting(start)) {
            PyErr_Clear();
        }
        else {
            // 检查并调整索引，然后定位到一维位置
            if (check_and_adjust_index(&start, self->size, -1, NULL) < 0) {
                goto finish;
            }
            PyArray_ITER_GOTO1D(self, start);
            // 将val打包到数组的当前位置
            retval = PyArray_Pack(PyArray_DESCR(self->ao), self->dataptr, val);
            PyArray_ITER_RESET(self);
            if (retval < 0) {
                PyErr_SetString(PyExc_ValueError,
                                "Error setting single item of array.");
            }
            goto finish;
        }

    skip:
        // 增加对类型的引用计数，并尝试从val创建一个PyArrayObject对象
        Py_INCREF(type);
        arrval = (PyArrayObject *)PyArray_FromAny(val, type, 0, 0,
                                                  NPY_ARRAY_FORCECAST, NULL);
        if (arrval == NULL) {
            return -1;
        }
        // 创建val的迭代器对象
        val_it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arrval);
        if (val_it == NULL) {
            goto finish;
        }
        // 如果val_it的大小为0，则直接返回0
        if (val_it->size == 0) {
            retval = 0;
            goto finish;
        }

        /* 设置转换以处理将单个元素复制到arrval中 */
        NPY_ARRAYMETHOD_FLAGS transfer_flags = 0;
        npy_intp one = 1;
        int itemsize = type->elsize;
        /* 我们可以假设新分配的数组是对齐的 */
        int is_aligned = IsUintAligned(self->ao);
        // 获取数据类型转换函数
        if (PyArray_GetDTypeTransferFunction(
                    is_aligned, itemsize, itemsize, type, type, 0,
                    &cast_info, &transfer_flags) < 0) {
            goto finish;
        }

        /* 检查切片 */
    # 检查是否为切片对象
    if (PySlice_Check(ind)) {
        # 解析切片对象，获取起始位置、步长和步数
        start = parse_index_entry(ind, &step_size, &n_steps, self->size, 0, 0);
        # 如果解析失败，跳转至结束标签
        if (start == -1) {
            goto finish;
        }
        # 如果步数为省略号或新轴索引，抛出索引错误并跳转至结束标签
        if (n_steps == ELLIPSIS_INDEX || n_steps == NEWAXIS_INDEX) {
            PyErr_SetString(PyExc_IndexError,
                            "cannot use Ellipsis or newaxes here");
            goto finish;
        }
        # 将迭代器移到一维数组的指定位置
        PyArray_ITER_GOTO1D(self, start);
        # 设置数据传输步长
        npy_intp transfer_strides[2] = {itemsize, itemsize};
        # 如果步数为单一索引
        if (n_steps == SINGLE_INDEX) {
            # 设置参数数组，并进行类型转换和数据传输
            char *args[2] = {PyArray_DATA(arrval), self->dataptr};
            if (cast_info.func(&cast_info.context, args, &one,
                               transfer_strides, cast_info.auxdata) < 0) {
                goto finish;
            }
            # 重置迭代器状态
            PyArray_ITER_RESET(self);
            # 返回成功标志
            retval = 0;
            # 跳转至结束标签
            goto finish;
        }
        # 多步索引时循环处理
        while (n_steps--) {
            # 设置参数数组，并进行类型转换和数据传输
            char *args[2] = {val_it->dataptr, self->dataptr};
            if (cast_info.func(&cast_info.context, args, &one,
                               transfer_strides, cast_info.auxdata) < 0) {
                goto finish;
            }
            # 更新起始位置
            start += step_size;
            # 将迭代器移到更新后的位置
            PyArray_ITER_GOTO1D(self, start);
            # 移动值迭代器到下一个位置
            PyArray_ITER_NEXT(val_it);
            # 如果值迭代器超出范围，重置它
            if (val_it->index == val_it->size) {
                PyArray_ITER_RESET(val_it);
            }
        }
        # 重置主迭代器状态
        PyArray_ITER_RESET(self);
        # 返回成功标志
        retval = 0;
        # 跳转至结束标签
        goto finish;
    }

    /* 将对象转换为 INTP 数组，如果是整数数组标量或列表 */
    indtype = PyArray_DescrFromType(NPY_INTP);
    if (PyList_Check(ind)) {
        # 增加引用计数，将对象转换为数组
        Py_INCREF(indtype);
        obj = PyArray_FromAny(ind, indtype, 0, 0, NPY_ARRAY_FORCECAST, NULL);
    }
    else {
        # 增加引用计数
        Py_INCREF(ind);
        obj = ind;
    }

    # 检查对象是否为数组
    if (obj != NULL && PyArray_Check(obj)) {
        /* 检查是否为布尔数组 */
        if (PyArray_TYPE((PyArrayObject *)obj)==NPY_BOOL) {
            # 如果是布尔数组，执行布尔索引赋值操作
            if (iter_ass_sub_Bool(self, (PyArrayObject *)obj,
                                  val_it, &cast_info) < 0) {
                goto finish;
            }
            # 设置返回成功标志
            retval=0;
        }
        /* 检查是否为整数数组 */
        else if (PyArray_ISINTEGER((PyArrayObject *)obj)) {
            # 增加引用计数，将对象转换为整数数组
            PyObject *new;
            Py_INCREF(indtype);
            new = PyArray_CheckFromAny(obj, indtype, 0, 0,
                           NPY_ARRAY_FORCECAST | NPY_ARRAY_BEHAVED_NS, NULL);
            # 减少原对象的引用计数
            Py_DECREF(obj);
            obj = new;
            # 如果转换失败，跳转至结束标签
            if (new == NULL) {
                goto finish;
            }
            # 如果是整数数组，执行整数索引赋值操作
            if (iter_ass_sub_int(self, (PyArrayObject *)obj,
                                 val_it, &cast_info) < 0) {
                goto finish;
            }
            # 设置返回成功标志
            retval = 0;
        }
    }

 finish:
    # 如果没有发生异常且返回值小于0，设置索引错误异常
    if (!PyErr_Occurred() && retval < 0) {
        PyErr_SetString(PyExc_IndexError, "unsupported iterator index");
    }
    # 释放对象类型描述符的引用
    Py_XDECREF(indtype);
    # 释放对象的引用
    Py_XDECREF(obj);
    # 释放值迭代器的引用
    Py_XDECREF(val_it);
    # 释放数组值的引用
    Py_XDECREF(arrval);
    # 调用函数 NPY_cast_info_xfree，释放 cast_info 指向的内存
    NPY_cast_info_xfree(&cast_info);
    # 返回函数的结果值给调用者
    return retval;
}

static PyMappingMethods iter_as_mapping = {
    (lenfunc)iter_length,                   /* mp_length */
    (binaryfunc)iter_subscript,             /* mp_subscript */
    (objobjargproc)iter_ass_subscript,      /* mp_ass_subscript */
};

/* Two options:
 *  1) underlying array is contiguous
 *     -- return 1-d wrapper around it
 *  2) underlying array is not contiguous
 *     -- make new 1-d contiguous array with updateifcopy flag set
 *        to copy back to the old array
 *
 *  If underlying array is readonly, then we make the output array readonly
 *     and updateifcopy does not apply.
 *
 *  Changed 2017-07-21, 1.14.0.
 *
 *  In order to start the process of removing UPDATEIFCOPY, see gh-7054, the
 *  behavior is changed to always return an non-writeable copy when the base
 *  array is non-contiguous. Doing that will hopefully smoke out those few
 *  folks who assign to the result with the expectation that the base array
 *  will be changed. At a later date non-contiguous arrays will always return
 *  writeable copies.
 *
 *  Note that the type and argument expected for the __array__ method is
 *  ignored.
 */
static PyArrayObject *
iter_array(PyArrayIterObject *it, PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    PyArrayObject *ret;
    npy_intp size;

    size = PyArray_SIZE(it->ao);  // 获取数组对象 `it` 的大小

    Py_INCREF(PyArray_DESCR(it->ao));  // 增加数组描述符的引用计数

    if (PyArray_ISCONTIGUOUS(it->ao)) {
        // 如果数组是连续的，则创建一个基于原数组的一维包装器
        ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
                &PyArray_Type, PyArray_DESCR(it->ao),
                1, &size, NULL, PyArray_DATA(it->ao),
                PyArray_FLAGS(it->ao), (PyObject *)it->ao, (PyObject *)it->ao);
        if (ret == NULL) {
            return NULL;
        }
    }
    else {
        // 如果数组不是连续的，则创建一个新的一维连续数组，并设置 updateifcopy 标志
        ret = (PyArrayObject *)PyArray_NewFromDescr(
                &PyArray_Type, PyArray_DESCR(it->ao), 1, &size,
                NULL, NULL, 0, (PyObject *)it->ao);
        if (ret == NULL) {
            return NULL;
        }
        if (PyArray_CopyAnyInto(ret, it->ao) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
        // 清除可写标志，使得输出数组为只读状态
        PyArray_CLEARFLAGS(ret, NPY_ARRAY_WRITEABLE);
    }
    return ret;  // 返回创建的数组对象
}

static PyObject *
iter_copy(PyArrayIterObject *it, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    // 返回数组的扁平化视图
    return PyArray_Flatten(it->ao, 0);
}

static PyMethodDef iter_methods[] = {
    /* to get array */
    {"__array__",
        (PyCFunction)iter_array,           // 对应的 C 函数为 iter_array
        METH_VARARGS | METH_KEYWORDS, NULL},  // 方法接受位置参数和关键字参数

    {"copy",
        (PyCFunction)iter_copy,            // 对应的 C 函数为 iter_copy
        METH_VARARGS, NULL},                // 方法接受位置参数

    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyObject *
iter_richcompare(PyArrayIterObject *self, PyObject *other, int cmp_op)
{
    PyArrayObject *new;
    PyObject *ret;

    new = (PyArrayObject *)iter_array(self, NULL, NULL);  // 获取数组的新视图
    if (new == NULL) {
        return NULL;
    }

    // 使用数组的比较方法进行比较
    ret = array_richcompare(new, other, cmp_op);

    // 解析写回复制标志
    PyArray_ResolveWritebackIfCopy(new);

    Py_DECREF(new);  // 减少数组视图的引用计数

    // 返回比较的结果
    return ret;
}
    return ret;
/** END of Array Iterator **/

// 设置迭代器成员变量的定义
static PyMemberDef iter_members[] = {
    // base 成员，对象类型，偏移量，只读，无特殊处理函数
    {"base",
        T_OBJECT,
        offsetof(PyArrayIterObject, ao),
        READONLY, NULL},
    // 数组结束标志
    {NULL, 0, 0, 0, NULL},
};

// 定义获取迭代器索引的函数
static PyObject *
iter_index_get(PyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    // 返回迭代器当前的索引值
    return PyArray_PyIntFromIntp(self->index);
}

// 定义获取迭代器坐标的函数
static PyObject *
iter_coords_get(PyArrayIterObject *self, void *NPY_UNUSED(ignored))
{
    int nd;
    nd = PyArray_NDIM(self->ao);
    // 如果数组是连续存储的
    if (self->contiguous) {
        /*
         * 坐标不被跟踪 ---
         * 需要根据索引生成
         */
        npy_intp val;
        int i;
        val = self->index;
        for (i = 0; i < nd; i++) {
            // 如果因子不为零，计算坐标
            if (self->factors[i] != 0) {
                self->coordinates[i] = val / self->factors[i];
                val = val % self->factors[i];
            } else {
                self->coordinates[i] = 0;
            }
        }
    }
    // 返回坐标的整数元组对象
    return PyArray_IntTupleFromIntp(nd, self->coordinates);
}

// 定义获取器的获取和设置函数
static PyGetSetDef iter_getsets[] = {
    // 获取索引的属性
    {"index",
        (getter)iter_index_get,
        NULL, NULL, NULL},
    // 获取坐标的属性
    {"coords",
        (getter)iter_coords_get,
        NULL, NULL, NULL},
    // 结束标志
    {NULL, NULL, NULL, NULL, NULL},
};

// 定义数组迭代器类型对象
NPY_NO_EXPORT PyTypeObject PyArrayIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    // 对象类型名
    .tp_name = "numpy.flatiter",
    // 基本大小
    .tp_basicsize = sizeof(PyArrayIterObject),
    // 析构函数
    .tp_dealloc = (destructor)arrayiter_dealloc,
    // 映射接口
    .tp_as_mapping = &iter_as_mapping,
    // 标志
    .tp_flags = Py_TPFLAGS_DEFAULT,
    // 富比较函数
    .tp_richcompare = (richcmpfunc)iter_richcompare,
    // 迭代下一个函数
    .tp_iternext = (iternextfunc)arrayiter_next,
    // 方法
    .tp_methods = iter_methods,
    // 成员变量
    .tp_members = iter_members,
    // 获取器
    .tp_getset = iter_getsets,
};

// 调整索引对象迭代器的维度和步长
/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_Broadcast(PyArrayMultiIterObject *mit)
{
    int i, nd, k, j;
    int src_iter = -1;  /* Initializing avoids a compiler warning. */
    npy_intp tmp;
    PyArrayIterObject *it;

    // 发现广播的维度数量
    /* 遍历所有迭代器，确定迭代器中最大的维度数 */
    for (i = 0, nd = 0; i < mit->numiter; i++) {
        nd = PyArray_MAX(nd, PyArray_NDIM(mit->iters[i]->ao));
    }
    mit->nd = nd;

    /* 在每个维度中发现广播形状 */
    for (i = 0; i < nd; i++) {
        mit->dimensions[i] = 1;
        for (j = 0; j < mit->numiter; j++) {
            it = mit->iters[j];
            /* 如果数组的维度小于最大维度nd，就在其前面添加1 */
            k = i + PyArray_NDIM(it->ao) - nd;
            if (k >= 0) {
                tmp = PyArray_DIMS(it->ao)[k];
                if (tmp == 1) {
                    continue;
                }
                /* 如果mit->dimensions[i]为1，将其设置为tmp */
                if (mit->dimensions[i] == 1) {
                    mit->dimensions[i] = tmp;
                    src_iter = j;
                }
                /* 如果mit->dimensions[i]不等于tmp，则设置形状不匹配的异常 */
                else if (mit->dimensions[i] != tmp) {
                    set_shape_mismatch_exception(mit, src_iter, j);
                    return -1;
                }
            }
        }
    }

    /*
     * 重设每个迭代器对象的迭代器维度和步长 -- 使用值为0的步长进行广播
     * 需要检查溢出
     */
    tmp = PyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    if (tmp < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "broadcast dimensions too large.");
        return -1;
    }
    mit->size = tmp;
    for (i = 0; i < mit->numiter; i++) {
        it = mit->iters[i];
        it->nd_m1 = mit->nd - 1;
        it->size = tmp;
        nd = PyArray_NDIM(it->ao);
        if (nd != 0) {
            it->factors[mit->nd-1] = 1;
        }
        for (j = 0; j < mit->nd; j++) {
            it->dims_m1[j] = mit->dimensions[j] - 1;
            k = j + nd - mit->nd;
            /*
             * 如果此维度是添加的或底层数组的形状为1
             */
            if ((k < 0) ||
                PyArray_DIMS(it->ao)[k] != mit->dimensions[j]) {
                it->contiguous = 0;
                it->strides[j] = 0;
            }
            else {
                it->strides[j] = PyArray_STRIDES(it->ao)[k];
            }
            it->backstrides[j] = it->strides[j] * it->dims_m1[j];
            if (j > 0)
                it->factors[mit->nd-j-1] =
                    it->factors[mit->nd-j] * mit->dimensions[mit->nd-j];
        }
        PyArray_ITER_RESET(it);
    }
    return 0;
/*
 * 返回一个错误信息对象，指示传递的参数数量不正确。
 * 参数数量需要至少为0且最多为NPY_MAXARGS个数组对象。
 */
static inline PyObject*
multiiter_wrong_number_of_args(void)
{
    return PyErr_Format(PyExc_ValueError,
                        "Need at least 0 and at most %d "
                        "array objects.", NPY_MAXARGS);
}

/*
 * PyArrayMultiIterObject 构造函数的通用实现。
 * 创建一个 PyArrayMultiIterObject 对象，用于迭代多个数组。
 */
static PyObject*
multiiter_new_impl(int n_args, PyObject **args)
{
    PyArrayMultiIterObject *multi;
    int i;

    // 分配 PyArrayMultiIterObject 结构的内存空间
    multi = PyArray_malloc(sizeof(PyArrayMultiIterObject));
    if (multi == NULL) {
        return PyErr_NoMemory();
    }
    // 初始化 multi 对象，设置其类型为 PyArrayMultiIter_Type
    PyObject_Init((PyObject *)multi, &PyArrayMultiIter_Type);
    multi->numiter = 0;

    for (i = 0; i < n_args; ++i) {
        PyObject *obj = args[i];
        PyObject *arr;
        PyArrayIterObject *it;

        // 如果 obj 是 PyArrayMultiIterObject 类型的实例
        if (PyObject_IsInstance(obj, (PyObject *)&PyArrayMultiIter_Type)) {
            PyArrayMultiIterObject *mit = (PyArrayMultiIterObject *)obj;
            int j;

            // 检查将要添加的迭代器数量是否超过 NPY_MAXARGS
            if (multi->numiter + mit->numiter > NPY_MAXARGS) {
                multiiter_wrong_number_of_args();
                goto fail;
            }
            // 将 mit 中的每个迭代器添加到 multi 中
            for (j = 0; j < mit->numiter; ++j) {
                arr = (PyObject *)mit->iters[j]->ao;
                it = (PyArrayIterObject *)PyArray_IterNew(arr);
                if (it == NULL) {
                    goto fail;
                }
                multi->iters[multi->numiter++] = it;
            }
        }
        // 如果 multi 中迭代器的数量还未达到 NPY_MAXARGS，且 obj 可以转换为数组对象
        else if (multi->numiter < NPY_MAXARGS) {
            arr = PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);
            if (arr == NULL) {
                goto fail;
            }
            it = (PyArrayIterObject *)PyArray_IterNew(arr);
            Py_DECREF(arr);
            if (it == NULL) {
                goto fail;
            }
            multi->iters[multi->numiter++] = it;
        }
        else {
            multiiter_wrong_number_of_args();
            goto fail;
        }
    }

    // 检查是否成功添加了迭代器
    if (multi->numiter < 0) {
        multiiter_wrong_number_of_args();
        goto fail;
    }
    // 对 multi 进行广播处理
    if (PyArray_Broadcast(multi) < 0) {
        goto fail;
    }
    // 重置 multi 的多重迭代器状态
    PyArray_MultiIter_RESET(multi);

    return (PyObject *)multi;

fail:
    // 在失败时释放 multi 对象
    Py_DECREF(multi);

    return NULL;
}

/*NUMPY_API
 * 从 Python 对象数组和任何额外数组中获取 MultiIterator
 *
 * PyObject **mps - PyObjects 数组
 * int n - 数组中的 PyObjects 数量
 * int nadd - 要包含在迭代器中的额外数组数量
 *
 * 返回一个 MultiIterator 对象。
 */
NPY_NO_EXPORT PyObject*
PyArray_MultiIterFromObjects(PyObject **mps, int n, int nadd, ...)
{
    PyObject *args_impl[NPY_MAXARGS];
    int ntot = n + nadd;
    int i;
    va_list va;

    // 检查总参数数量是否在合理范围内
    if ((ntot > NPY_MAXARGS) || (ntot < 0)) {
        return multiiter_wrong_number_of_args();
    }

    // 复制 mps 数组中的 Python 对象到 args_impl 中
    for (i = 0; i < n; ++i) {
        args_impl[i] = mps[i];
    }

    // 处理可变参数列表，将其添加到 args_impl 中
    va_start(va, nadd);
    for (; i < ntot; ++i) {
        args_impl[i] = va_arg(va, PyObject *);
    }
    va_end(va);

    // 调用 multiiter_new_impl 函数创建 MultiIterator 对象并返回
    return multiiter_new_impl(ntot, args_impl);
}
/*NUMPY_API
 * Get MultiIterator,
 */
/* 定义 PyArray_MultiIterNew 函数，返回 PyObject 指针类型 */
NPY_NO_EXPORT PyObject*
PyArray_MultiIterNew(int n, ...)
{
    /* 定义存储 PyObject 指针的数组 */
    PyObject *args_impl[NPY_MAXARGS];
    /* 定义循环计数器 */
    int i;
    /* 定义 va_list 变量 */
    va_list va;

    /* 如果参数个数超过 NPY_MAXARGS 或小于 0，则调用 multiiter_wrong_number_of_args 函数返回错误信息 */
    if ((n > NPY_MAXARGS) || (n < 0)) {
        return multiiter_wrong_number_of_args();
    }

    /* 开始使用可变参数列表 */
    va_start(va, n);
    /* 将可变参数列表中的参数依次存入 args_impl 数组中 */
    for (i = 0; i < n; ++i) {
        args_impl[i] = va_arg(va, PyObject *);
    }
    /* 结束可变参数列表的使用 */
    va_end(va);

    /* 调用 multiiter_new_impl 函数，返回多迭代器对象的 PyObject 指针 */
    return multiiter_new_impl(n, args_impl);
}

/* 定义 arraymultiter_new 函数 */
static PyObject*
arraymultiter_new(PyTypeObject *NPY_UNUSED(subtype), PyObject *args,
                  PyObject *kwds)
{
    /* 定义返回值、快速序列和序列长度 */
    PyObject *ret, *fast_seq;
    Py_ssize_t n;

    /* 如果关键字参数不为空且大于 0，抛出 ValueError 异常 */
    if (kwds != NULL && PyDict_Size(kwds) > 0) {
        PyErr_SetString(PyExc_ValueError,
                        "keyword arguments not accepted.");
        return NULL;
    }

    /* 将传入的参数 args 转换为快速序列 fast_seq */
    fast_seq = PySequence_Fast(args, "");  // needed for pypy
    /* 如果转换失败，返回 NULL */
    if (fast_seq == NULL) {
        return NULL;
    }
    /* 获取快速序列的长度 */
    n = PySequence_Fast_GET_SIZE(fast_seq);
    /* 如果长度大于 NPY_MAXARGS，释放 fast_seq 并返回错误信息 */
    if (n > NPY_MAXARGS) {
        Py_DECREF(fast_seq);
        return multiiter_wrong_number_of_args();
    }
    /* 调用 multiiter_new_impl 函数，返回多迭代器对象的 PyObject 指针 */
    ret = multiiter_new_impl(n, PySequence_Fast_ITEMS(fast_seq));
    /* 释放 fast_seq 对象 */
    Py_DECREF(fast_seq);
    return ret;
}

/* 定义 arraymultiter_next 函数 */
static PyObject *
arraymultiter_next(PyArrayMultiIterObject *multi)
{
    /* 定义返回值、迭代器数目和循环计数器 */
    PyObject *ret;
    int i, n;

    /* 获取迭代器的数目 */
    n = multi->numiter;
    /* 创建一个元组对象作为返回值 */
    ret = PyTuple_New(n);
    /* 如果创建失败，返回 NULL */
    if (ret == NULL) {
        return NULL;
    }
    /* 如果 multi->index 小于 multi->size */
    if (multi->index < multi->size) {
        /* 遍历迭代器数组 */
        for (i = 0; i < n; i++) {
            /* 获取当前迭代器对象 */
            PyArrayIterObject *it=multi->iters[i];
            /* 将迭代器指向的数据转换为标量对象，存入元组中 */
            PyTuple_SET_ITEM(ret, i,
                             PyArray_ToScalar(it->dataptr, it->ao));
            /* 移动迭代器到下一个位置 */
            PyArray_ITER_NEXT(it);
        }
        /* 增加 multi 的索引值 */
        multi->index++;
        /* 返回元组对象 */
        return ret;
    }
    /* 如果 multi->index 大于等于 multi->size，释放 ret 对象并返回 NULL */
    Py_DECREF(ret);
    return NULL;
}

/* 定义 arraymultiter_dealloc 函数 */
static void
arraymultiter_dealloc(PyArrayMultiIterObject *multi)
{
    /* 定义循环计数器 */
    int i;

    /* 释放迭代器数组中的每一个迭代器对象 */
    for (i = 0; i < multi->numiter; i++) {
        Py_XDECREF(multi->iters[i]);
    }
    /* 释放 multi 对象内存 */
    Py_TYPE(multi)->tp_free((PyObject *)multi);
}

/* 定义 arraymultiter_size_get 函数 */
static PyObject *
arraymultiter_size_get(PyArrayMultiIterObject *self, void *NPY_UNUSED(ignored))
{
    /* 返回 self 对象的 size 属性 */
    return PyArray_PyIntFromIntp(self->size);
}

/* 定义 arraymultiter_index_get 函数 */
static PyObject *
arraymultiter_index_get(PyArrayMultiIterObject *self, void *NPY_UNUSED(ignored))
{
    /* 返回 self 对象的 index 属性 */
    return PyArray_PyIntFromIntp(self->index);
}

/* 定义 arraymultiter_shape_get 函数 */
static PyObject *
arraymultiter_shape_get(PyArrayMultiIterObject *self, void *NPY_UNUSED(ignored))
{
    /* 返回 self 对象的 dimensions 属性 */
    return PyArray_IntTupleFromIntp(self->nd, self->dimensions);
}

/* 定义 arraymultiter_iters_get 函数 */
static PyObject *
arraymultiter_iters_get(PyArrayMultiIterObject *self, void *NPY_UNUSED(ignored))
{
    /* 返回一个元组对象，包含 self 对象的所有迭代器 */
    PyObject *res;
    int i, n;

    /* 获取迭代器的数目 */
    n = self->numiter;
    /* 创建一个包含 n 个元素的元组对象 */
    res = PyTuple_New(n);
    /* 如果创建失败，返回 NULL */
    if (res == NULL) {
        return res;
    }
    /* 遍历 self 对象的迭代器数组，将每个迭代器对象添加到元组中 */
    for (i = 0; i < n; i++) {
        Py_INCREF(self->iters[i]);
        PyTuple_SET_ITEM(res, i, (PyObject *)self->iters[i]);
    }
    /* 返回包含迭代器的元组对象 */
    return res;
}

/* 定义 arraymultiter_getsetlist 数组，包含 size 属性的 getter 函数 */
static PyGetSetDef arraymultiter_getsetlist[] = {
    {"size",
        (getter)arraymultiter_size_get,
        NULL,
        NULL, NULL},
    {"index",
        // 键名为 "index"，对应的 getter 函数为 arraymultiter_index_get
        (getter)arraymultiter_index_get,
        // setter 为 NULL
        NULL,
        // docstring 为 NULL
        NULL, NULL},
    {"shape",
        // 键名为 "shape"，对应的 getter 函数为 arraymultiter_shape_get
        (getter)arraymultiter_shape_get,
        // setter 为 NULL
        NULL,
        // docstring 为 NULL
        NULL, NULL},
    {"iters",
        // 键名为 "iters"，对应的 getter 函数为 arraymultiter_iters_get
        (getter)arraymultiter_iters_get,
        // setter 为 NULL
        NULL,
        // docstring 为 NULL
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
        // 最后一行为 NULL 结束符
};

static PyMemberDef arraymultiter_members[] = {
    {"numiter",
        T_INT,
        offsetof(PyArrayMultiIterObject, numiter),
        READONLY, NULL},
    {"nd",
        T_INT,
        offsetof(PyArrayMultiIterObject, nd),
        READONLY, NULL},
    {"ndim",
        T_INT,
        offsetof(PyArrayMultiIterObject, nd),
        READONLY, NULL},
    {NULL, 0, 0, 0, NULL},
};

static PyObject *
arraymultiter_reset(PyArrayMultiIterObject *self, PyObject *args)
{
    // 解析参数元组，如果解析失败则返回 NULL
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    // 调用 NumPy C API 函数 PyArray_MultiIter_RESET 重置多重迭代器对象
    PyArray_MultiIter_RESET(self);
    // 返回 Python 中的 None 对象
    Py_RETURN_NONE;
}

static PyMethodDef arraymultiter_methods[] = {
    // reset 方法的定义
    {"reset",
        (PyCFunction) arraymultiter_reset,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},      /* sentinel */
};

NPY_NO_EXPORT PyTypeObject PyArrayMultiIter_Type = {
    // PyArrayMultiIter_Type 对象的类型定义
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.broadcast",   // 对象类型名称
    .tp_basicsize = sizeof(PyArrayMultiIterObject),   // 基本大小
    .tp_dealloc = (destructor)arraymultiter_dealloc,  // 析构函数
    .tp_flags = Py_TPFLAGS_DEFAULT,   // 默认标志
    .tp_iternext = (iternextfunc)arraymultiter_next, // 迭代下一个函数
    .tp_methods = arraymultiter_methods,   // 方法列表
    .tp_members = arraymultiter_members,   // 成员列表
    .tp_getset = arraymultiter_getsetlist, // 获取和设置方法列表
    .tp_new = arraymultiter_new,   // 新建对象的方法
};

/*========================= Neighborhood iterator ======================*/

static void neighiter_dealloc(PyArrayNeighborhoodIterObject* iter);

static char* _set_constant(PyArrayNeighborhoodIterObject* iter,
        PyArrayObject *fill)
{
    char *ret;
    PyArrayIterObject *ar = iter->_internal_iter;
    int storeflags, st;

    // 分配内存给 ret，大小为数组元素的大小
    ret = PyDataMem_NEW(PyArray_ITEMSIZE(ar->ao));
    if (ret == NULL) {
        // 分配内存失败，设置内存错误并返回 NULL
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    if (PyArray_ISOBJECT(ar->ao)) {
        // 如果数组元素是对象类型，则复制填充数组中的 PyObject* 数据到 ret
        memcpy(ret, PyArray_DATA(fill), sizeof(PyObject*));
        // 增加引用计数
        Py_INCREF(*(PyObject**)ret);
    } else {
        /* 非对象类型 */

        // 存储原始标志
        storeflags = PyArray_FLAGS(ar->ao);
        // 设置数组行为标志
        PyArray_ENABLEFLAGS(ar->ao, NPY_ARRAY_BEHAVED);
        // 设置数组元素为 fill
        st = PyArray_SETITEM(ar->ao, ret, (PyObject*)fill);
        // 恢复原始标志
        ((PyArrayObject_fields *)ar->ao)->flags = storeflags;

        if (st < 0) {
            // 设置数组元素失败，释放内存并返回 NULL
            PyDataMem_FREE(ret);
            return NULL;
        }
    }

    return ret;
}

#define _INF_SET_PTR(c) \
    bd = coordinates[c] + p->coordinates[c]; \
    if (bd < p->limits[c][0] || bd > p->limits[c][1]) { \
        return niter->constant; \
    } \
    _coordinates[c] = bd;

/* 根据当前坐标设置数据指针 */
static char*
get_ptr_constant(PyArrayIterObject* _iter, const npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS];
    PyArrayNeighborhoodIterObject *niter = (PyArrayNeighborhoodIterObject*)_iter;
    PyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR(i)
    }

    // 调用 translate 方法返回数据指针
    return p->translate(p, _coordinates);
}
#undef _INF_SET_PTR

#define _NPY_IS_EVEN(x) ((x) % 2 == 0)
/*
 * For an array x of dimension n, and given index i, returns j, 0 <= j < n
 * such as x[i] = x[j], with x assumed to be mirrored. For example, for x =
 * {1, 2, 3} (n = 3)
 *
 * index -5 -4 -3 -2 -1 0 1 2 3 4 5 6
 * value  2  3  3  2  1 1 2 3 3 2 1 1
 *
 * __npy_pos_remainder(4, 3) will return 1, because x[4] = x[1]
 */
static inline npy_intp
__npy_pos_remainder(npy_intp i, npy_intp n)
{
    npy_intp k, l, j;

    /* Mirror i such that it is guaranteed to be positive */
    if (i < 0) {
        i = - i - 1;
    }

    /* Compute k and l such that i = k * n + l, 0 <= l < k */
    k = i / n;
    l = i - k * n;

    if (_NPY_IS_EVEN(k)) {
        j = l;
    } else {
        j = n - 1 - l;
    }
    return j;
}
#undef _NPY_IS_EVEN

/*
 * Macro to set _coordinates[c] using mirrored indexing based on bounds
 * and current coordinates for neighborhood iteration.
 */
#define _INF_SET_PTR_MIRROR(c) \
    lb = p->limits[c][0]; \
    bd = coordinates[c] + p->coordinates[c] - lb; \
    _coordinates[c] = lb + __npy_pos_remainder(bd, p->limits_sizes[c]);

/*
 * Sets the data pointer (_coordinates) based on mirrored indexing for
 * neighborhood iteration.
 */
static char*
get_ptr_mirror(PyArrayIterObject* _iter, const npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS], lb;
    PyArrayNeighborhoodIterObject *niter = (PyArrayNeighborhoodIterObject*)_iter;
    PyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR_MIRROR(i)
    }

    return p->translate(p, _coordinates);
}
#undef _INF_SET_PTR_MIRROR

/*
 * Compute l such that i = k * n + l, 0 <= l < |k|
 */
static inline npy_intp
__npy_euclidean_division(npy_intp i, npy_intp n)
{
    npy_intp l;

    l = i % n;
    if (l < 0) {
        l += n;
    }
    return l;
}

/*
 * Macro to set _coordinates[c] using circular indexing based on bounds
 * and current coordinates for neighborhood iteration.
 */
#define _INF_SET_PTR_CIRCULAR(c) \
    lb = p->limits[c][0]; \
    bd = coordinates[c] + p->coordinates[c] - lb; \
    _coordinates[c] = lb + __npy_euclidean_division(bd, p->limits_sizes[c]);

/*
 * Sets the data pointer (_coordinates) based on circular indexing for
 * neighborhood iteration.
 */
static char*
get_ptr_circular(PyArrayIterObject* _iter, const npy_intp *coordinates)
{
    int i;
    npy_intp bd, _coordinates[NPY_MAXDIMS], lb;
    PyArrayNeighborhoodIterObject *niter = (PyArrayNeighborhoodIterObject*)_iter;
    PyArrayIterObject *p = niter->_internal_iter;

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR_CIRCULAR(i)
    }
    return p->translate(p, _coordinates);
}
#undef _INF_SET_PTR_CIRCULAR

/*
 * Create a new neighborhood iterator object.
 */
/*NUMPY_API
 * A Neighborhood Iterator object.
 */
NPY_NO_EXPORT PyObject*
PyArray_NeighborhoodIterNew(PyArrayIterObject *x, const npy_intp *bounds,
                            int mode, PyArrayObject* fill)
{
    int i;
    PyArrayNeighborhoodIterObject *ret;

    ret = PyArray_malloc(sizeof(*ret));
    if (ret == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)ret, &PyArrayNeighborhoodIter_Type);

    Py_INCREF(x->ao);  /* PyArray_RawIterBaseInit steals a reference */
    PyArray_RawIterBaseInit((PyArrayIterObject*)ret, x->ao);
    Py_INCREF(x);
    ret->_internal_iter = x;

    ret->nd = PyArray_NDIM(x->ao);

    /* Additional initialization and setup code continues beyond this point */
    for (i = 0; i < ret->nd; ++i) {
        ret->dimensions[i] = PyArray_DIMS(x->ao)[i];
    }


    /* 设置返回结构体中的维度信息为输入数组的维度信息 */
    ret->dimensions[i] = PyArray_DIMS(x->ao)[i];



    /* 计算邻域的大小并复制形状 */
    ret->size = 1;
    for (i = 0; i < ret->nd; ++i) {
        ret->bounds[i][0] = bounds[2 * i];
        ret->bounds[i][1] = bounds[2 * i + 1];
        ret->size *= (ret->bounds[i][1] - ret->bounds[i][0]) + 1;

        /* limits 用于跟踪邻域的有效范围：如果邻域的边界超出数组范围，则 limits 等于 boundaries。
         * 相反，如果边界严格在数组内部，则 limits 对应于数组范围。例如，对于数组 [1, 2, 3]，
         * 如果边界是 [-1, 3]，则 limits 是 [-1, 3]；但如果边界是 [1, 2]，则 limits 是 [0, 2]。
         *
         * 这在叠加在此迭代器之上的邻域迭代器中使用 */
        ret->limits[i][0] = ret->bounds[i][0] < 0 ? ret->bounds[i][0] : 0;
        ret->limits[i][1] = ret->bounds[i][1] >= ret->dimensions[i] - 1 ?
                            ret->bounds[i][1] :
                            ret->dimensions[i] - 1;
        ret->limits_sizes[i] = (ret->limits[i][1] - ret->limits[i][0]) + 1;
    }


    /* 计算邻域的大小并复制形状 */
    ret->size = 1;
    for (i = 0; i < ret->nd; ++i) {
        /* 设置邻域边界 */
        ret->bounds[i][0] = bounds[2 * i];
        ret->bounds[i][1] = bounds[2 * i + 1];
        /* 计算邻域大小 */
        ret->size *= (ret->bounds[i][1] - ret->bounds[i][0]) + 1;

        /* limits 用于跟踪邻域的有效范围 */
        ret->limits[i][0] = ret->bounds[i][0] < 0 ? ret->bounds[i][0] : 0;
        ret->limits[i][1] = ret->bounds[i][1] >= ret->dimensions[i] - 1 ?
                            ret->bounds[i][1] :
                            ret->dimensions[i] - 1;
        ret->limits_sizes[i] = (ret->limits[i][1] - ret->limits[i][0]) + 1;
    }



    switch (mode) {
        case NPY_NEIGHBORHOOD_ITER_ZERO_PADDING:
            ret->constant = PyArray_Zero(x->ao);
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_ONE_PADDING:
            ret->constant = PyArray_One(x->ao);
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING:
            /* _set_constant 返回值中的新引用，如果数组对象 */
            assert(PyArray_EquivArrTypes(x->ao, fill) == NPY_TRUE);
            ret->constant = _set_constant(ret, fill);
            if (ret->constant == NULL) {
                goto clean_x;
            }
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING:
            ret->mode = mode;
            ret->constant = NULL;
            ret->translate = &get_ptr_mirror;
            break;
        case NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING:
            ret->mode = mode;
            ret->constant = NULL;
            ret->translate = &get_ptr_circular;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Unsupported padding mode");
            goto clean_x;
    }


    /* 根据不同的填充模式进行设置 */
    switch (mode) {
        case NPY_NEIGHBORHOOD_ITER_ZERO_PADDING:
            /* 使用零填充模式 */
            ret->constant = PyArray_Zero(x->ao);
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_ONE_PADDING:
            /* 使用一填充模式 */
            ret->constant = PyArray_One(x->ao);
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING:
            /* 常数填充模式，_set_constant 返回值中的新引用 */
            assert(PyArray_EquivArrTypes(x->ao, fill) == NPY_TRUE);
            ret->constant = _set_constant(ret, fill);
            if (ret->constant == NULL) {
                goto clean_x;
            }
            ret->mode = mode;
            ret->translate = &get_ptr_constant;
            break;
        case NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING:
            /* 镜像填充模式 */
            ret->mode = mode;
            ret->constant = NULL;
            ret->translate = &get_ptr_mirror;
            break;
        case NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING:
            /* 循环填充模式 */
            ret->mode = mode;
            ret->constant = NULL;
            ret->translate = &get_ptr_circular;
            break;
        default:
            /* 不支持的填充模式 */
            PyErr_SetString(PyExc_ValueError, "Unsupported padding mode");
            goto clean_x;
    }



    /*
     * XXX: we force x iterator to be non contiguous because we need
     * coordinates... Modifying the iterator here is not great
     */
    x->contiguous = 0;


    /* 强制 x 迭代器为非连续，因为我们需要坐标... 在这里修改迭代器并不是最佳做法 */
    x->contiguous = 0;



    PyArrayNeighborhoodIter_Reset(ret);

    return (PyObject*)ret;


    /* 重置邻域迭代器并返回其 PyObject* 类型的指针 */
    PyArrayNeighborhoodIter_Reset(ret);

    return (PyObject*)ret;
// 释放 clean_x 函数中 ret 结构体的内部迭代器资源
Py_DECREF(ret->_internal_iter);
// 调用 array_iter_base_dealloc 函数释放 ret 结构体的基类迭代器资源
array_iter_base_dealloc((PyArrayIterObject*)ret);
// 释放 ret 结构体本身的内存资源
PyArray_free((PyArrayObject*)ret);
// 返回空指针表示操作失败
return NULL;
}

// 释放 PyArrayNeighborhoodIterObject 结构体的资源
static void neighiter_dealloc(PyArrayNeighborhoodIterObject* iter)
{
    // 如果迭代器模式是常量填充模式
    if (iter->mode == NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING) {
        // 如果数组对象是对象数组，则释放常量指针指向的对象
        if (PyArray_ISOBJECT(iter->_internal_iter->ao)) {
            Py_DECREF(*(PyObject**)iter->constant);
        }
    }
    // 释放常量数组的内存资源
    PyDataMem_FREE(iter->constant);
    // 释放内部迭代器资源
    Py_DECREF(iter->_internal_iter);

    // 调用 array_iter_base_dealloc 函数释放 iter 结构体的基类迭代器资源
    array_iter_base_dealloc((PyArrayIterObject*)iter);
    // 释放 iter 结构体本身的内存资源
    PyArray_free((PyArrayObject*)iter);
}

// 定义 PyArrayNeighborhoodIter_Type 类型的静态属性
NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    // 类型名称为 "numpy.neigh_internal_iter"
    .tp_name = "numpy.neigh_internal_iter",
    // 类型对象的基本大小为 PyArrayNeighborhoodIterObject 结构体大小
    .tp_basicsize = sizeof(PyArrayNeighborhoodIterObject),
    // 设置析构函数为 neighiter_dealloc 函数
    .tp_dealloc = (destructor)neighiter_dealloc,
    // 设置默认标志位
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
```