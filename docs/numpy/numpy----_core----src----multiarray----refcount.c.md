# `.\numpy\numpy\_core\src\multiarray\refcount.c`

```py
/*
 * This module corresponds to the `Special functions for NPY_OBJECT`
 * section in the numpy reference for C-API.
 */
#include "array_method.h"
#include "dtype_traversal.h"
#include "lowlevel_strided_loops.h"
#include "pyerrors.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"
#include "iterators.h"
#include "dtypemeta.h"
#include "refcount.h"

#include "npy_config.h"



/*
 * Helper function to clear a strided memory (normally or always contiguous)
 * from all Python (or other) references.  The function does nothing if the
 * array dtype does not indicate holding references.
 *
 * It is safe to call this function more than once, failing here is usually
 * critical (during cleanup) and should be set up to minimize the risk or
 * avoid it fully.
 */
NPY_NO_EXPORT int
PyArray_ClearBuffer(
        PyArray_Descr *descr, char *data,
        npy_intp stride, npy_intp size, int aligned)
{
    // Check if the array's dtype holds references; if not, return early
    if (!PyDataType_REFCHK(descr)) {
        return 0;
    }

    NPY_traverse_info clear_info;
    /* Flags unused: float errors do not matter and we do not release GIL */
    NPY_ARRAYMETHOD_FLAGS flags_unused;
    
    // Retrieve the function pointer to clear the buffer
    if (PyArray_GetClearFunction(
            aligned, stride, descr, &clear_info, &flags_unused) < 0) {
        return -1;
    }

    // Call the function to clear the memory
    int res = clear_info.func(
            NULL, clear_info.descr, data, size, stride, clear_info.auxdata);
    NPY_traverse_info_xfree(&clear_info);
    return res;
}


/*
 * Helper function to clear whole array.  It seems plausible that we should
 * be able to get away with assuming the array is contiguous.
 *
 * Must only be called on arrays which own their data (and asserts this).
 */
NPY_NO_EXPORT int
PyArray_ClearArray(PyArrayObject *arr)
{
    // Assert that the array owns its data
    assert(PyArray_FLAGS(arr) & NPY_ARRAY_OWNDATA);

    // Get the descriptor of the array
    PyArray_Descr *descr = PyArray_DESCR(arr);

    // Check if the array's dtype holds references; if not, return early
    if (!PyDataType_REFCHK(descr)) {
        return 0;
    }

    /*
     * The contiguous path should cover practically all important cases since
     * it is difficult to create a non-contiguous array which owns its memory
     * and only arrays which own their memory should clear it.
     */
    int aligned = PyArray_ISALIGNED(arr);
    if (PyArray_ISCONTIGUOUS(arr)) {
        // Clear the array buffer directly if it is contiguous
        return PyArray_ClearBuffer(
                descr, PyArray_BYTES(arr), descr->elsize,
                PyArray_SIZE(arr), aligned);
    }

    // Handle the case for non-contiguous arrays
    int idim, ndim;
    npy_intp shape_it[NPY_MAXDIMS], strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];
    char *data_it;

    // Prepare iterator for the array to clear all parts
    if (PyArray_PrepareOneRawArrayIter(
                    PyArray_NDIM(arr), PyArray_DIMS(arr),
                    PyArray_BYTES(arr), PyArray_STRIDES(arr),
                    &ndim, shape_it, &data_it, strides_it) < 0) {
        return -1;
    }

    npy_intp inner_stride = strides_it[0];
    npy_intp inner_shape = shape_it[0];
    NPY_traverse_info clear_info;
    # 定义未使用的数组方法标志，用于指示浮点错误不重要且不释放全局解释器锁（GIL）
    NPY_ARRAYMETHOD_FLAGS flags_unused;
    # 获取清除函数及其相关信息，如果失败则返回-1
    if (PyArray_GetClearFunction(
            aligned, inner_stride, descr, &clear_info, &flags_unused) < 0) {
        return -1;
    }
    # 开始原始迭代，处理多维数组的数据
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        # 处理最内层维度的数据
        if (clear_info.func(NULL, clear_info.descr,
                data_it, inner_shape, inner_stride, clear_info.auxdata) < 0) {
            return -1;
        }
    } NPY_RAW_ITER_ONE_NEXT(idim, ndim, coord,
                            shape_it, data_it, strides_it);
    # 执行成功，返回0
    return 0;
/*NUMPY_API
 * 增加单个数组项中所有对象的引用计数。这对于结构化数据类型来说比较复杂，因为需要提取对象的位置。
 * 函数会递归地执行每个嵌套字段或子数组数据类型，例如 `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    // 如果数据类型不需要引用计数检查，则直接返回
    if (!PyDataType_REFCHK(descr)) {
        return;
    }

    // 如果数据类型是 NPY_OBJECT（对象类型）
    if (descr->type_num == NPY_OBJECT) {
        // 从数据中拷贝对象的指针到临时变量
        memcpy(&temp, data, sizeof(temp));
        // 增加对象的引用计数
        Py_XINCREF(temp);
    }
    // 如果数据类型有字段（结构化类型）
    else if (PyDataType_HASFIELDS(descr)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        // 遍历结构化数据类型的字段
        while (PyDict_Next(PyDataType_FIELDS(descr), &pos, &key, &value)) {
            // 如果是标题键，则跳过
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            // 解析字段值，获取新的数据描述符、偏移量和标题
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                return;
            }
            // 递归增加字段数据的引用计数
            PyArray_Item_INCREF(data + offset, new);
        }
    }
    // 如果数据类型有子数组（子数组类型）
    else if (PyDataType_HASSUBARRAY(descr)) {
        int size, i, inner_elsize;

        // 获取子数组中单个元素的大小
        inner_elsize = PyDataType_SUBARRAY(descr)->base->elsize;
        // 如果内部元素大小为 0，则直接返回
        if (inner_elsize == 0) {
            /* 没有任何元素，因此直接返回 */
            return;
        }
        // 子数组在内存中是连续存储的
        size = descr->elsize / inner_elsize;

        // 遍历子数组中的每个元素，递归增加其引用计数
        for (i = 0; i < size; i++){
            /* 递归地增加子数组元素的引用计数 */
            PyArray_Item_INCREF(data + i * inner_elsize,
                                PyDataType_SUBARRAY(descr)->base);
        }
    }
    else {
        /* 此分支不应该被执行到，如果执行到这里，表示存在问题 */
        assert(0);
    }
    // 函数执行完毕，返回
    return;
}


/*NUMPY_API
 *
 * 减少单个数组项中所有对象的引用计数。这对于结构化数据类型来说比较复杂，因为需要提取对象的位置。
 * 函数会递归地执行每个嵌套字段或子数组数据类型，例如 `np.dtype([("field1", "O"), ("field2", "f,O", (3,2))])`
 */
NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr)
{
    PyObject *temp;

    // 如果数据类型不需要引用计数检查，则直接返回
    if (!PyDataType_REFCHK(descr)) {
        return;
    }

    // 如果数据类型是 NPY_OBJECT（对象类型）
    if (descr->type_num == NPY_OBJECT) {
        // 从数据中拷贝对象的指针到临时变量
        memcpy(&temp, data, sizeof(temp));
        // 减少对象的引用计数
        Py_XDECREF(temp);
    }
    // 剩余的分支（结构化类型、子数组类型）在这里不需要处理，因为这些情况不会涉及到减少引用计数的操作
}
    else if (PyDataType_HASFIELDS(descr)) {
        // 如果描述符包含字段信息
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        // 遍历字段字典
        while (PyDict_Next(PyDataType_FIELDS(descr), &pos, &key, &value)) {
            // 如果是字段的标题键，则跳过
            if (NPY_TITLE_KEY(key, value)) {
                continue;
            }
            // 解析字段值元组，获取新的描述符和偏移量
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
                                  &title)) {
                // 解析失败，返回
                return;
            }
            // 释放数据中偏移位置的项目
            PyArray_Item_XDECREF(data + offset, new);
        }
    }
    else if (PyDataType_HASSUBARRAY(descr)) {
        // 如果描述符有子数组
        int size, i, inner_elsize;

        // 获取子数组的元素大小
        inner_elsize = PyDataType_SUBARRAY(descr)->base->elsize;
        // 如果子数组的元素大小为0，说明没有元素，直接返回
        if (inner_elsize == 0) {
            /* There cannot be any elements, so return */
            return;
        }
        // 计算子数组在内存中的大小
        size = descr->elsize / inner_elsize;

        // 逐个减少子数组元素的引用计数
        for (i = 0; i < size; i++){
            /* Recursively decrement the reference count of subarray elements */
            PyArray_Item_XDECREF(data + i * inner_elsize,
                                 PyDataType_SUBARRAY(descr)->base);
        }
    }
    else {
        // 否则，这条路径不应该被执行到，断言失败
        /* This path should not be reachable. */
        assert(0);
    }
    // 返回
    return;
/* Used for arrays of python objects to increment the reference count of */
/* every python object in the array. */
/*NUMPY_API
  For object arrays, increment all internal references.
*/
NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp)
{
    npy_intp i, n;
    PyObject **data;
    PyObject *temp;
    PyArrayIterObject *it;

    // 检查数据类型是否需要引用计数
    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    // 如果数据类型不是对象类型，使用迭代器递增引用计数
    if (PyArray_DESCR(mp)->type_num != NPY_OBJECT) {
        // 创建数组迭代器
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        // 遍历数组并递增每个元素的引用计数
        while(it->index < it->size) {
            PyArray_Item_INCREF(it->dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
        return 0;
    }

    // 如果数组是一段连续的内存块
    if (PyArray_ISONESEGMENT(mp)) {
        // 获取数组数据指针和大小
        data = (PyObject **)PyArray_DATA(mp);
        n = PyArray_SIZE(mp);
        // 如果数据是对齐的，逐个增加引用计数
        if (PyArray_ISALIGNED(mp)) {
            for (i = 0; i < n; i++, data++) {
                Py_XINCREF(*data);
            }
        }
        // 如果数据未对齐，使用临时变量逐个增加引用计数
        else {
            for( i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                Py_XINCREF(temp);
            }
        }
    }
    else { /* 处理未对齐的数据 */
        // 创建数组迭代器
        it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)mp);
        if (it == NULL) {
            return -1;
        }
        // 遍历数组并递增每个元素的引用计数
        while(it->index < it->size) {
            memcpy(&temp, it->dataptr, sizeof(temp));
            Py_XINCREF(temp);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);
    }
    return 0;
}

/*NUMPY_API
  Decrement all internal references for object arrays.
  (or arrays with object fields)

  The use of this function is strongly discouraged, within NumPy
  use PyArray_Clear, which DECREF's and sets everything to NULL and can
  work with any dtype.
*/
NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp)
{
    npy_intp i, n;
    PyObject **data;
    PyObject *temp;
    /*
     * 静态分配可以在销毁期间不修改数组的引用计数。
     * （静态分配本质上并非必需）
     */
    PyArrayIterObject it;

    // 检查数据类型是否需要引用计数
    if (!PyDataType_REFCHK(PyArray_DESCR(mp))) {
        return 0;
    }
    // 如果数据类型不是对象类型，使用迭代器递减引用计数
    if (PyArray_DESCR(mp)->type_num != NPY_OBJECT) {
        // 检查数组维度是否超出支持范围
        if (PyArray_NDIM(mp) > NPY_MAXDIMS_LEGACY_ITERS) {
            PyErr_Format(PyExc_RuntimeError,
                    "this function only supports up to 32 dimensions but "
                    "the array has %d.", PyArray_NDIM(mp));
            return -1;
        }

        // 初始化原始迭代器并递减每个元素的引用计数
        PyArray_RawIterBaseInit(&it, mp);
        while(it.index < it.size) {
            PyArray_Item_XDECREF(it.dataptr, PyArray_DESCR(mp));
            PyArray_ITER_NEXT(&it);
        }
        return 0;
    }
    # 检查是否数组是单一连续段
    if (PyArray_ISONESEGMENT(mp)) {
        # 获取指向数据指针的指针数组
        data = (PyObject **)PyArray_DATA(mp);
        # 获取数组元素的数量
        n = PyArray_SIZE(mp);
        # 如果数组是对齐的
        if (PyArray_ISALIGNED(mp)) {
            # 逐个释放每个数据指针指向的对象
            for (i = 0; i < n; i++, data++) Py_XDECREF(*data);
        }
        else {
            # 如果数组不对齐，使用临时变量逐个释放数据指针指向的对象
            for (i = 0; i < n; i++, data++) {
                memcpy(&temp, data, sizeof(temp));
                Py_XDECREF(temp);
            }
        }
    }
    else { /* 处理不对齐的数据 */
        # 如果数组的维度超过了 Legacy 迭代器支持的最大维度（32维）
        if (PyArray_NDIM(mp) > NPY_MAXDIMS_LEGACY_ITERS) {
            # 抛出运行时错误，说明函数最多支持32维数组
            PyErr_Format(PyExc_RuntimeError,
                    "this function only supports up to 32 dimensions but "
                    "the array has %d.", PyArray_NDIM(mp));
            # 返回错误代码
            return -1;
        }

        # 初始化一个原始迭代器
        PyArray_RawIterBaseInit(&it, mp);
        # 遍历数组中的每个元素
        while(it.index < it.size) {
            # 使用临时变量释放当前迭代器指向的对象
            memcpy(&temp, it.dataptr, sizeof(temp));
            Py_XDECREF(temp);
            # 移动迭代器到下一个元素
            PyArray_ITER_NEXT(&it);
        }
    }
    # 返回成功代码
    return 0;
/*
 * 此函数作为一个入口点，确保 `np.empty()` 填充 dtype=object（包括字段）为 `None`，
 * 而不是保留为 NULL。因为未显式支持 NULL（尽管 Cython 现在支持了，我们从未严格保证过）。
 *
 * 假设连续内存布局
 *
 * TODO: 对于结构体而言，这个函数非常荒谬，应该使用 dtype_traversal 函数来代替...
 */
NPY_NO_EXPORT int
PyArray_SetObjectsToNone(PyArrayObject *arr)
{
    // 获取数组的描述符
    PyArray_Descr* descr = PyArray_DESCR(arr);

    // 非遗留数据类型需要负责初始化其内部引用
    if (!NPY_DT_is_legacy(NPY_DTYPE(descr))) {
        return 0;
    }

    npy_intp i,n;
    n = PyArray_SIZE(arr);
    // 如果数据类型是对象类型（dtype=object）
    if (descr->type_num == NPY_OBJECT) {
        PyObject **optr;
        // optr 指向数组数据的起始位置，这是一个 PyObject 指针数组
        optr = (PyObject **)(PyArray_DATA(arr));
        for (i = 0; i < n; i++) {
            // 增加 Py_None 的引用计数，并将其赋值给当前位置的指针
            Py_INCREF(Py_None);
            *optr++ = Py_None;
        }
    }
    else {
        char *optr;
        // optr 指向数组数据的起始位置，这是一个字符指针（字节流）
        optr = PyArray_DATA(arr);
        for (i = 0; i < n; i++) {
            // 根据数据类型描述符填充 optr 指向的数据区域为 None
            if (_fill_with_none(optr, descr) < 0) {
                return -1;
            }
            optr += descr->elsize;
        }
    }
    return 0;
}

/*
 * 根据数据类型描述符填充 optr 指向的数据区域为 None
 */
static int
_fill_with_none(char *optr, PyArray_Descr *dtype)
{
    // 如果数据类型不具有 NPY_ITEM_REFCOUNT 标志，则直接返回
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        return 0;
    }

    PyObject *None = Py_None;
    // 如果数据类型是对象类型（dtype=object）
    if (dtype->type_num == NPY_OBJECT) {
        // 增加 Py_None 的引用计数，并将其拷贝到 optr 指向的位置
        Py_XINCREF(Py_None);
        memcpy(optr, &None, sizeof(PyObject *));
    }
    // 如果数据类型具有字段（结构体）
    else if (PyDataType_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;

        // 遍历字段字典，填充各个字段为 None
        while (PyDict_Next(PyDataType_FIELDS(dtype), &pos, &key, &value)) {
            if (NPY_TITLE_KEY(key, value)) {
                continue; // 忽略标题字段
            }
            // 解析字段值为新的数据类型描述符、偏移量和标题
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return -1;
            }
            // 递归调用 _fill_with_none 填充字段数据区域为 None
            if (_fill_with_none(optr + offset, new) < 0) {
                return -1;
            }
        }
    }
    // 如果数据类型具有子数组
    else if (PyDataType_HASSUBARRAY(dtype)) {
        int size, i, inner_elsize;

        // 获取子数组的元素大小
        inner_elsize = PyDataType_SUBARRAY(dtype)->base->elsize;
        if (inner_elsize == 0) {
            /* 没有任何元素，直接返回 */
            return 0;
        }
        // 子数组在内存中是连续的
        size = dtype->elsize / inner_elsize;

        // 逐个元素递归调用 _fill_with_none
        for (i = 0; i < size; i++) {
            if (_fill_with_none(optr, PyDataType_SUBARRAY(dtype)->base) < 0) {
                return -1;
            }
            optr += inner_elsize;
        }
    }
    else {
        /* 不应该到达的路径，断言错误 */
        assert(0);
    }
    return 0;
}
```