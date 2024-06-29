# `.\numpy\numpy\_core\src\multiarray\mapping.c`

```
/*
 * Define macro to specify the NPY API version without deprecated features
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * Define macros to specify modules for multiarray and umath
 */
#define _MULTIARRAYMODULE
#define _UMATHMODULE

/*
 * Clean PY_SSIZE_T type usage to ensure compatibility
 */
#define PY_SSIZE_T_CLEAN

/*
 * Include Python core header files and structmember.h for member descriptors
 */
#include <Python.h>
#include <structmember.h>

/*
 * Include numpy array object and math utility headers
 */
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

/*
 * Include necessary custom numpy headers
 */
#include "arrayobject.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "common.h"
#include "ctors.h"
#include "descriptor.h"
#include "iterators.h"
#include "mapping.h"
#include "lowlevel_strided_loops.h"
#include "item_selection.h"
#include "mem_overlap.h"
#include "array_assign.h"
#include "array_coercion.h"

/*
 * Define code for iterator implementation for nditer until made public
 */
#define NPY_ITERATOR_IMPLEMENTATION_CODE
#include "nditer_impl.h"

/*
 * Include umathmodule header for universal functions in numpy
 */
#include "umathmodule.h"

/*
 * Define constants to indicate presence of different index types in numpy arrays
 */
#define HAS_INTEGER 1
#define HAS_NEWAXIS 2
#define HAS_SLICE 4
#define HAS_ELLIPSIS 8
#define HAS_FANCY 16
#define HAS_BOOL 32
#define HAS_SCALAR_ARRAY 64
#define HAS_0D_BOOL (HAS_FANCY | 128)

/*
 * Implementation of array_length function to provide length of array
 * Supports arrays with non-zero dimensions
 * Throws TypeError for unsized objects (0-dimensional arrays)
 */
NPY_NO_EXPORT Py_ssize_t
array_length(PyArrayObject *self)
{
    if (PyArray_NDIM(self) != 0) {
        return PyArray_DIMS(self)[0]; // Return the length of the first dimension
    } else {
        PyErr_SetString(PyExc_TypeError, "len() of unsized object"); // Raise TypeError for unsized objects
        return -1;
    }
}

/*
 * Helper function for PyArray_MapIterSwapAxes and related functions
 * Generates a tuple for transpose based on specified dimensions
 */
static void
_get_transpose(int fancy_ndim, int consec, int ndim, int getmap, npy_intp *dims)
{
    /*
     * For getting the array the tuple for transpose is
     * (n1,...,n1+n2-1,0,...,n1-1,n1+n2,...,n3-1)
     * n1 is the number of dimensions of the broadcast index array
     * n2 is the number of dimensions skipped at the start
     * n3 is the number of dimensions of the result
     */

    /*
     * For setting the array the tuple for transpose is
     * (n2,...,n1+n2-1,0,...,n2-1,n1+n2,...n3-1)
     */
    int n1 = fancy_ndim;
    int n2 = consec;  // axes to insert at
    int n3 = ndim;

    // Determine the boundary value based on operation type (get or set)
    int bnd = getmap ? n1 : n2;
    int val = bnd;
    int i = 0;

    // Generate the tuple for transpose based on calculated dimensions
    while (val < n1 + n2) {
        dims[i++] = val++;
    }
    val = 0;
    while (val < bnd) {
        dims[i++] = val++;
    }
    val = n1 + n2;
    while (val < n3) {
        dims[i++] = val++;
    }
}
/*
 * Swap the axes to or from their inserted form. MapIter always puts the
 * advanced (array) indices first in the iteration. But if they are
 * consecutive, will insert/transpose them back before returning.
 * This is stored as `mit->consec != 0` (the place where they are inserted)
 * For assignments, the opposite happens: The values to be assigned are
 * transposed (getmap=1 instead of getmap=0). `getmap=0` and `getmap=1`
 * undo the other operation.
 */
NPY_NO_EXPORT void
PyArray_MapIterSwapAxes(PyArrayMapIterObject *mit, PyArrayObject **ret, int getmap)
{
    PyObject *new;
    PyArray_Dims permute;
    npy_intp d[NPY_MAXDIMS];
    PyArrayObject *arr;

    permute.ptr = d;
    permute.len = mit->nd;

    /*
     * arr might not have the right number of dimensions
     * and need to be reshaped first by prepending ones
     */
    arr = *ret;
    if (PyArray_NDIM(arr) != mit->nd) {
        // Calculate dimensions for reshaping arr
        for (int i = 1; i <= PyArray_NDIM(arr); i++) {
            permute.ptr[mit->nd-i] = PyArray_DIMS(arr)[PyArray_NDIM(arr)-i];
        }
        // Prepend ones if necessary
        for (int i = 0; i < mit->nd-PyArray_NDIM(arr); i++) {
            permute.ptr[i] = 1;
        }
        // Reshape arr with the new dimensions
        new = PyArray_Newshape(arr, &permute, NPY_ANYORDER);
        Py_DECREF(arr);
        *ret = (PyArrayObject *)new;
        if (new == NULL) {
            return;
        }
    }

    // Perform transpose operation based on getmap flag
    _get_transpose(mit->nd_fancy, mit->consec, mit->nd, getmap, permute.ptr);

    // Transpose *ret array using permute dimensions
    new = PyArray_Transpose(*ret, &permute);
    Py_DECREF(*ret);
    *ret = (PyArrayObject *)new;
}

static inline void
multi_DECREF(PyObject **objects, npy_intp n)
{
    npy_intp i;
    for (i = 0; i < n; i++) {
        Py_DECREF(objects[i]);
    }
}

/**
 * Unpack a tuple into an array of new references. Returns the number of objects
 * unpacked.
 *
 * Useful if a tuple is being iterated over multiple times, or for a code path
 * that doesn't always want the overhead of allocating a tuple.
 */
static inline npy_intp
unpack_tuple(PyTupleObject *index, PyObject **result, npy_intp result_n)
{
    npy_intp n, i;
    n = PyTuple_GET_SIZE(index);
    if (n > result_n) {
        PyErr_SetString(PyExc_IndexError,
                        "too many indices for array");
        return -1;
    }
    // Unpack tuple into result array and increment references
    for (i = 0; i < n; i++) {
        result[i] = PyTuple_GET_ITEM(index, i);
        Py_INCREF(result[i]);
    }
    return n;
}

/* Unpack a single scalar index, taking a new reference to match unpack_tuple */
static inline npy_intp
unpack_scalar(PyObject *index, PyObject **result, npy_intp NPY_UNUSED(result_n))
{
    // Increment reference for the scalar index
    Py_INCREF(index);
    result[0] = index;
    return 1;
}
/**
 * Turn an index argument into a c-array of `PyObject *`s, one for each index.
 *
 * When a tuple is passed, the tuple elements are unpacked into the buffer.
 * Anything else is handled by unpack_scalar().
 *
 * @param  index     The index object, which may or may not be a tuple. This is
 *                   a borrowed reference.
 * @param  result    An empty buffer of PyObject* to write each index component
 *                   to. The references written are new.
 * @param  result_n  The length of the result buffer
 *
 * @returns          The number of items in `result`, or -1 if an error occurred.
 *                   The entries in `result` at and beyond this index should be
 *                   assumed to contain garbage, even if they were initialized
 *                   to NULL, so are not safe to Py_XDECREF. Use multi_DECREF to
 *                   dispose of them.
 */
NPY_NO_EXPORT npy_intp
unpack_indices(PyObject *index, PyObject **result, npy_intp result_n)
{
    /* It is likely that the logic here can be simplified. See the discussion
     * on https://github.com/numpy/numpy/pull/21029
     */

    /* Fast route for passing a tuple */
    if (PyTuple_CheckExact(index)) {
        // 如果传入的是元组，则调用 unpack_tuple 函数解压元组到 result 中
        return unpack_tuple((PyTupleObject *)index, result, result_n);
    }

    /*
     * Passing a tuple subclass - coerce to the base type. This incurs an
     * allocation, but doesn't need to be a fast path anyway. Note that by
     * calling `PySequence_Tuple`, we ensure that the subclass `__iter__` is
     * called.
     */
    if (PyTuple_Check(index)) {
        // 如果传入的是元组的子类，则强制转换为基础类型元组，并解压到 result 中
        PyTupleObject *tup = (PyTupleObject *) PySequence_Tuple(index);
        if (tup == NULL) {
            return -1;
        }
        npy_intp n = unpack_tuple(tup, result, result_n);
        Py_DECREF(tup);
        return n;
    }

    // 对于其他情况，调用 unpack_scalar 处理单个索引
    return unpack_scalar(index, result, result_n);
}

/**
 * Prepare an npy_index_object from the python slicing object.
 *
 * This function handles all index preparations with the exception
 * of field access. It fills the array of index_info structs correctly.
 * It already handles the boolean array special case for fancy indexing,
 * i.e. if the index type is boolean, it is exactly one matching boolean
 * array. If the index type is fancy, the boolean array is already
 * converted to integer arrays. There is (as before) no checking of the
 * boolean dimension.
 *
 * Checks everything but the bounds.
 *
 * @param the array being indexed
 * @param the index object
 * @param index info struct being filled (size of NPY_MAXDIMS * 2 + 1)
 * @param number of indices found
 * @param dimension of the indexing result
 * @param dimension of the fancy/advanced indices part
 * @param whether to allow the boolean special case
 *
 * @returns the index_type or -1 on failure and fills the number of indices.
 */
NPY_NO_EXPORT int
/*
 * 准备索引函数，根据给定的索引对象和数组信息，解析并准备索引信息
 */
prepare_index(PyArrayObject *self, PyObject *index,
              npy_index_info *indices,
              int *num, int *ndim, int *out_fancy_ndim, int allow_boolean)
{
    int new_ndim, fancy_ndim, used_ndim, index_ndim;
    int curr_idx, get_idx;

    int i;
    npy_intp n;

    PyObject *obj = NULL;
    PyArrayObject *arr;

    int index_type = 0;
    int ellipsis_pos = -1;

    /*
     * 选择解包 `2*NPY_MAXDIMS` 项的历史原因。
     * 最长的“合理”索引，生成的结果维度最多为 32，
     * 是 `(0,)*ncu.MAXDIMS + (None,)*ncu.MAXDIMS`。
     * 更长的索引可能存在，但并不常见。
     */
    PyObject *raw_indices[NPY_MAXDIMS*2];

    // 解包索引对象，填充 raw_indices 数组，返回索引的维度
    index_ndim = unpack_indices(index, raw_indices, NPY_MAXDIMS*2);
    if (index_ndim == -1) {
        return -1;
    }

    /*
     * 将所有索引解析到 indices 数组中的 index_info 结构体中
     */
    used_ndim = 0;
    new_ndim = 0;
    fancy_ndim = 0;
    get_idx = 0;
    curr_idx = 0;

    /*
     * 比较索引的维度和实际的维度。这是为了找到省略号值或在必要时添加省略号。
     */
    if (used_ndim < PyArray_NDIM(self)) {
        if (index_type & HAS_ELLIPSIS) {
            // 设置省略号的值并更新维度计数
            indices[ellipsis_pos].value = PyArray_NDIM(self) - used_ndim;
            used_ndim = PyArray_NDIM(self);
            new_ndim += indices[ellipsis_pos].value;
        }
        else {
            /*
             * 尚未有省略号，但索引不完整，因此在末尾添加省略号。
             */
            index_type |= HAS_ELLIPSIS;
            indices[curr_idx].object = NULL;
            indices[curr_idx].type = HAS_ELLIPSIS;
            indices[curr_idx].value = PyArray_NDIM(self) - used_ndim;
            ellipsis_pos = curr_idx;

            used_ndim = PyArray_NDIM(self);
            new_ndim += indices[curr_idx].value;
            curr_idx += 1;
        }
    }
    else if (used_ndim > PyArray_NDIM(self)) {
        // 索引的维度超过数组的维度，抛出 IndexError 异常
        PyErr_Format(PyExc_IndexError,
                     "too many indices for array: "
                     "array is %d-dimensional, but %d were indexed",
                     PyArray_NDIM(self),
                     used_ndim);
        goto failed_building_indices;
    }
    else if (index_ndim == 0) {
        /*
         * 0 维度索引到 0 维度数组，即 array[()]。
         * 我们将其视为整数索引，返回标量值。
         * 这是有道理的，因为 array[...] 返回数组，而 array[()] 返回标量。
         */
        used_ndim = 0;
        index_type = HAS_INTEGER;
    }

    /* HAS_SCALAR_ARRAY 需要清理 index_type */
    if (index_type & HAS_SCALAR_ARRAY) {
        /* 如果索引类型包含标志 HAS_SCALAR_ARRAY */
        /* 清除这个信息，因为后续处理中不需要它，会增加复杂度 */
        if (index_type & HAS_FANCY) {
            index_type -= HAS_SCALAR_ARRAY;
        }
        /* 对于完整的整数索引，标量数组被视为整数索引的一部分 */
        else if (index_type == (HAS_INTEGER | HAS_SCALAR_ARRAY)) {
            index_type -= HAS_SCALAR_ARRAY;
        }
    }

    /*
     * 到这一步，索引已经全部正确设置，没有进行边界检查，
     * 新的数组可能仍然具有比可能的维度更多的维度，
     * 并且布尔索引数组的形状可能不正确。
     *
     * 现在检查这些，这样我们以后就不必担心了。
     * 这可能发生在使用 fancy indexing 或者 newaxis 时。
     * 这意味着在维度过多时，广播错误的情况会更少发生。
     */
    if (index_type & (HAS_NEWAXIS | HAS_FANCY)) {
        if (new_ndim + fancy_ndim > NPY_MAXDIMS) {
            /* 如果新的数组维度和 fancy 索引维度总和超过了 NPY_MAXDIMS */
            /* 报错，指数的数量必须在 [0, NPY_MAXDIMS] 范围内，索引结果会有 %d 维度 */
            PyErr_Format(PyExc_IndexError,
                         "number of dimensions must be within [0, %d], "
                         "indexing result would have %d",
                         NPY_MAXDIMS, (new_ndim + fancy_ndim));
            goto failed_building_indices;
        }

        /*
         * 如果我们有一个 fancy 索引，可能有一个布尔数组索引。
         * 现在检查它的形状是否正确，因为我们可以找出它作用的轴。
         */
        used_ndim = 0;
        for (i = 0; i < curr_idx; i++) {
            if ((indices[i].type == HAS_FANCY) && indices[i].value > 0) {
                if (indices[i].value != PyArray_DIM(self, used_ndim)) {
                    char err_msg[174];

                    PyOS_snprintf(err_msg, sizeof(err_msg),
                        "boolean index did not match indexed array along "
                        "axis %d; size of axis is %" NPY_INTP_FMT
                        " but size of corresponding boolean axis is %" NPY_INTP_FMT,
                        used_ndim, PyArray_DIM(self, used_ndim),
                        indices[i].value);
                    PyErr_SetString(PyExc_IndexError, err_msg);
                    goto failed_building_indices;
                }
            }

            if (indices[i].type == HAS_ELLIPSIS) {
                used_ndim += indices[i].value;
            }
            else if ((indices[i].type == HAS_NEWAXIS) ||
                     (indices[i].type == HAS_0D_BOOL)) {
                used_ndim += 0;
            }
            else {
                used_ndim += 1;
            }
        }
    }

    /* 将 curr_idx 赋值给 num */
    *num = curr_idx;
    /* 将 new_ndim + fancy_ndim 赋值给 ndim */
    *ndim = new_ndim + fancy_ndim;
    /* 将 fancy_ndim 赋值给 out_fancy_ndim */
    *out_fancy_ndim = fancy_ndim;

    /* 减少 raw_indices 的引用计数，释放内存 */
    multi_DECREF(raw_indices, index_ndim);

    /* 返回索引类型 */
    return index_type;

  failed_building_indices:
    /* 处理构建索引失败的情况 */
    for (i=0; i < curr_idx; i++) {
        Py_XDECREF(indices[i].object);
    }
    /* 减少 raw_indices 的引用计数，释放内存 */
    multi_DECREF(raw_indices, index_ndim);
    /* 返回错误状态 */
    return -1;
/**
 * Check if self has memory overlap with one of the index arrays, or with extra_op.
 *
 * @returns 1 if memory overlap found, 0 if not.
 */
NPY_NO_EXPORT int
index_has_memory_overlap(PyArrayObject *self,
                         int index_type, npy_index_info *indices, int num,
                         PyObject *extra_op)
{
    int i;

    // 如果索引类型包含花式索引或布尔索引
    if (index_type & (HAS_FANCY | HAS_BOOL)) {
        // 遍历索引数组
        for (i = 0; i < num; ++i) {
            // 如果索引对象存在且为数组，并且与 self 存在内存重叠
            if (indices[i].object != NULL &&
                    PyArray_Check(indices[i].object) &&
                    solve_may_share_memory(self,
                                           (PyArrayObject *)indices[i].object,
                                           1) != 0) {
                return 1;
            }
        }
    }

    // 如果存在额外操作对象，并且额外操作对象是数组，并且与 self 存在内存重叠
    if (extra_op != NULL && PyArray_Check(extra_op) &&
            solve_may_share_memory(self, (PyArrayObject *)extra_op, 1) != 0) {
        return 1;
    }

    // 没有发现内存重叠
    return 0;
}


/**
 * Get pointer for an integer index.
 *
 * For a purely integer index, set ptr to the memory address.
 * Returns 0 on success, -1 on failure.
 * The caller must ensure that the index is a full integer
 * one.
 *
 * @param Array being indexed
 * @param result pointer
 * @param parsed index information
 * @param number of indices
 *
 * @return 0 on success -1 on failure
 */
static int
get_item_pointer(PyArrayObject *self, char **ptr,
                    npy_index_info *indices, int index_num) {
    int i;
    // 设置指针为数组的起始地址
    *ptr = PyArray_BYTES(self);
    // 遍历所有索引
    for (i=0; i < index_num; i++) {
        // 检查并调整索引值，确保在有效范围内
        if ((check_and_adjust_index(&(indices[i].value),
                               PyArray_DIMS(self)[i], i, NULL)) < 0) {
            return -1;
        }
        // 计算指针偏移量
        *ptr += PyArray_STRIDE(self, i) * indices[i].value;
    }
    return 0;
}


/**
 * Get view into an array using all non-array indices.
 *
 * For any index, get a view of the subspace into the original
 * array. If there are no fancy indices, this is the result of
 * the indexing operation.
 * Ensure_array allows to fetch a safe subspace view for advanced
 * indexing.
 *
 * @param Array being indexed
 * @param resulting array (new reference)
 * @param parsed index information
 * @param number of indices
 * @param Whether result should inherit the type from self
 *
 * @return 0 on success -1 on failure
 */
static int
get_view_from_index(PyArrayObject *self, PyArrayObject **view,
                    npy_index_info *indices, int index_num, int ensure_array) {
    npy_intp new_strides[NPY_MAXDIMS];
    npy_intp new_shape[NPY_MAXDIMS];
    int i, j;
    int new_dim = 0;
    int orig_dim = 0;
    // 获取数组的起始地址
    char *data_ptr = PyArray_BYTES(self);

    /* for slice parsing */
    npy_intp start, stop, step, n_steps;

    // 更多功能索引解析...
    for (i=0; i < index_num; i++) {
        // 根据索引类型进行处理
        switch (indices[i].type) {
            case HAS_INTEGER:
                // 如果索引类型是整数，检查并调整索引值
                if ((check_and_adjust_index(&indices[i].value,
                                PyArray_DIMS(self)[orig_dim], orig_dim,
                                NULL)) < 0) {
                    return -1;
                }
                // 根据数组在原始维度上的步长和调整后的索引值更新数据指针位置
                data_ptr += PyArray_STRIDE(self, orig_dim) * indices[i].value;

                // 增加新维度的计数，原始维度加一
                new_dim += 0;
                orig_dim += 1;
                break;
            case HAS_ELLIPSIS:
                // 如果索引类型是省略号，根据省略号值设置新的步长和形状
                for (j=0; j < indices[i].value; j++) {
                    new_strides[new_dim] = PyArray_STRIDE(self, orig_dim);
                    new_shape[new_dim] = PyArray_DIMS(self)[orig_dim];
                    new_dim += 1;
                    orig_dim += 1;
                }
                break;
            case HAS_SLICE:
                // 如果索引类型是切片，获取切片的起始、停止、步长等参数
                if (PySlice_GetIndicesEx(indices[i].object,
                                         PyArray_DIMS(self)[orig_dim],
                                         &start, &stop, &step, &n_steps) < 0) {
                    return -1;
                }
                // 处理步长非正数的情况
                if (n_steps <= 0) {
                    /* TODO: Always points to start then, could change that */
                    n_steps = 0;
                    step = 1;
                    start = 0;
                }

                // 根据切片的起始位置更新数据指针位置
                data_ptr += PyArray_STRIDE(self, orig_dim) * start;
                // 设置新维度的步长和形状
                new_strides[new_dim] = PyArray_STRIDE(self, orig_dim) * step;
                new_shape[new_dim] = n_steps;
                new_dim += 1;
                orig_dim += 1;
                break;
            case HAS_NEWAXIS:
                // 如果索引类型是新轴，设置新轴的步长为0，形状为1
                new_strides[new_dim] = 0;
                new_shape[new_dim] = 1;
                new_dim += 1;
                break;
            /* Fancy and 0-d boolean indices are ignored here */
            case HAS_0D_BOOL:
                // 忽略特殊情况：精确索引和0维布尔索引
                break;
            default:
                // 默认情况下，增加新维度计数，原始维度加一
                new_dim += 0;
                orig_dim += 1;
                break;
        }
    }

    /* 创建新视图并设置基础数组 */
    // 增加基础数组的引用计数
    Py_INCREF(PyArray_DESCR(self));
    // 使用给定的描述符和参数创建新的数组视图
    *view = (PyArrayObject *)PyArray_NewFromDescr_int(
            ensure_array ? &PyArray_Type : Py_TYPE(self),
            PyArray_DESCR(self),
            new_dim, new_shape, new_strides, data_ptr,
            PyArray_FLAGS(self),
            ensure_array ? NULL : (PyObject *)self,
            (PyObject *)self, _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
    // 检查视图创建是否成功
    if (*view == NULL) {
        return -1;
    }

    // 返回成功
    return 0;
}

/*
 * Implements boolean indexing. This produces a one-dimensional
 * array which picks out all of the elements of 'self' for which
 * the corresponding element of 'op' is True.
 *
 * This operation is somewhat unfortunate, because to produce
 * a one-dimensional output array, it has to choose a particular
 * iteration order, in the case of NumPy that is always C order even
 * though this function allows different choices.
 */
NPY_NO_EXPORT PyArrayObject *
array_boolean_subscript(PyArrayObject *self,
                        PyArrayObject *bmask, NPY_ORDER order)
{
    npy_intp size, itemsize;
    char *ret_data;
    PyArray_Descr *dtype;
    PyArray_Descr *ret_dtype;
    PyArrayObject *ret;

    size = count_boolean_trues(PyArray_NDIM(bmask), PyArray_DATA(bmask),
                                PyArray_DIMS(bmask), PyArray_STRIDES(bmask));

    /* Allocate the output of the boolean indexing */
    dtype = PyArray_DESCR(self);
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1, &size,
                                NULL, NULL, 0, NULL);
    if (ret == NULL) {
        return NULL;
    }
    /* not same as *dtype* if the DType class replaces dtypes */
    ret_dtype = PyArray_DESCR(ret);

    itemsize = dtype->elsize;
    ret_data = PyArray_DATA(ret);

    /* Create an iterator for the data */
    }

    if (!PyArray_CheckExact(self)) {
        PyArrayObject *tmp = ret;

        Py_INCREF(ret_dtype);
        ret = (PyArrayObject *)PyArray_NewFromDescrAndBase(
                Py_TYPE(self), ret_dtype,
                1, &size, PyArray_STRIDES(ret), PyArray_BYTES(ret),
                PyArray_FLAGS(self), (PyObject *)self, (PyObject *)tmp);

        Py_DECREF(tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    return ret;
}

/*
 * Implements boolean indexing assignment. This takes the one-dimensional
 * array 'v' and assigns its values to all of the elements of 'self' for which
 * the corresponding element of 'op' is True.
 *
 * This operation is somewhat unfortunate, because to match up with
 * a one-dimensional output array, it has to choose a particular
 * iteration order, in the case of NumPy that is always C order even
 * though this function allows different choices.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
array_assign_boolean_subscript(PyArrayObject *self,
                    PyArrayObject *bmask, PyArrayObject *v, NPY_ORDER order)
{
    npy_intp size, v_stride;
    char *v_data;
    npy_intp bmask_size;

    if (PyArray_DESCR(bmask)->type_num != NPY_BOOL) {
        PyErr_SetString(PyExc_TypeError,
                "NumPy boolean array indexing assignment "
                "requires a boolean index");
        return -1;
    }
}
    // 检查输入数组 v 是否为多维数组（大于1维）
    if (PyArray_NDIM(v) > 1) {
        // 报错并返回 -1，要求 NumPy 布尔数组索引赋值需要输入为 0 或 1 维
        PyErr_Format(PyExc_TypeError,
                "NumPy boolean array indexing assignment "
                "requires a 0 or 1-dimensional input, input "
                "has %d dimensions", PyArray_NDIM(v));
        return -1;
    }

    // 检查布尔掩码数组 bmask 的维度是否与被索引数组 self 的维度相同
    if (PyArray_NDIM(bmask) != PyArray_NDIM(self)) {
        // 报错并返回 -1，要求布尔掩码索引数组必须与被索引数组 self 的维度相同
        PyErr_SetString(PyExc_ValueError,
                "The boolean mask assignment indexing array "
                "must have the same number of dimensions as "
                "the array being indexed");
        return -1;
    }

    // 计算布尔掩码中 True 的数量
    size = count_boolean_trues(PyArray_NDIM(bmask), PyArray_DATA(bmask),
                                PyArray_DIMS(bmask), PyArray_STRIDES(bmask));
    /* 用于调整广播 'bmask' 到 'self' 的修正因子 */
    bmask_size = PyArray_SIZE(bmask);
    if (bmask_size > 0) {
        size *= PyArray_SIZE(self) / bmask_size;
    }

    // 调整用于 0 维和广播情况的步长
    if (PyArray_NDIM(v) > 0 && PyArray_DIMS(v)[0] != 1) {
        // 如果 v 是一维数组且长度与 size 不一致，则报错
        if (size != PyArray_DIMS(v)[0]) {
            PyErr_Format(PyExc_ValueError,
                    "NumPy boolean array indexing assignment "
                    "cannot assign %" NPY_INTP_FMT " input values to "
                    "the %" NPY_INTP_FMT " output values where the mask is true",
                    PyArray_DIMS(v)[0], size);
            return -1;
        }
        // 获取 v 的步长
        v_stride = PyArray_STRIDES(v)[0];
    }
    else {
        // 对于 0 维数组，步长设为 0
        v_stride = 0;
    }

    // 获取数组 v 的数据指针
    v_data = PyArray_DATA(v);

    /* 为数据创建迭代器 */
    // 初始化结果变量为 0
    int res = 0;
    // 返回结果
    return res;
}
/*
 * C-level integer indexing always returning an array and never a scalar.
 * Works also for subclasses, but it will not be called on one from the
 * Python API.
 *
 * This function does not accept negative indices because it is called by
 * PySequence_GetItem (through array_item) and that converts them to
 * positive indices.
 */
NPY_NO_EXPORT PyObject *
array_item_asarray(PyArrayObject *self, npy_intp i)
{
    npy_index_info indices[2];  // 定义索引信息结构体数组
    PyObject *result;  // 结果对象指针

    if (PyArray_NDIM(self) == 0) {
        PyErr_SetString(PyExc_IndexError,
                        "too many indices for array");
        return NULL;  // 如果数组维度为0，抛出索引错误并返回空
    }
    if (i < 0) {
        /* This is an error, but undo PySequence_GetItem fix for message */
        i -= PyArray_DIM(self, 0);  // 如果索引为负数，将其转换为正数
    }

    indices[0].value = i;  // 设置第一个索引的值为 i
    indices[0].type = HAS_INTEGER;  // 表示第一个索引是整数类型
    indices[1].value = PyArray_NDIM(self) - 1;  // 设置第二个索引的值为数组的最后一个维度
    indices[1].type = HAS_ELLIPSIS;  // 表示第二个索引是省略号类型
    if (get_view_from_index(self, (PyArrayObject **)&result,
                            indices, 2, 0) < 0) {
        return NULL;  // 根据索引获取视图对象，失败则返回空
    }
    return result;  // 返回获取的结果对象
}


/*
 * Python C-Api level item subscription (implementation for PySequence_GetItem)
 *
 * Negative indices are not accepted because PySequence_GetItem converts
 * them to positive indices before calling this.
 */
NPY_NO_EXPORT PyObject *
array_item(PyArrayObject *self, Py_ssize_t i)
{
    if (PyArray_NDIM(self) == 1) {  // 如果数组维度为1
        char *item;  // 字符指针 item
        npy_index_info index;  // 索引信息结构体

        if (i < 0) {
            /* This is an error, but undo PySequence_GetItem fix for message */
            i -= PyArray_DIM(self, 0);  // 如果索引为负数，将其转换为正数
        }

        index.value = i;  // 设置索引值为 i
        index.type = HAS_INTEGER;  // 表示索引为整数类型
        if (get_item_pointer(self, &item, &index, 1) < 0) {
            return NULL;  // 获取数组项指针，失败则返回空
        }
        return PyArray_Scalar(item, PyArray_DESCR(self), (PyObject *)self);  // 根据数组项创建标量对象并返回
    }
    else {
        return array_item_asarray(self, i);  // 如果数组维度不为1，调用 array_item_asarray 函数处理
    }
}


/* make sure subscript always returns an array object */
NPY_NO_EXPORT PyObject *
array_subscript_asarray(PyArrayObject *self, PyObject *op)
{
    return PyArray_EnsureAnyArray(array_subscript(self, op));  // 确保使用下标访问时总是返回一个数组对象
}

/*
 * Attempts to subscript an array using a field name or list of field names.
 *
 * ret =  0, view != NULL: view points to the requested fields of arr
 * ret =  0, view == NULL: an error occurred
 * ret = -1, view == NULL: unrecognized input, this is not a field index.
 */
NPY_NO_EXPORT int
_get_field_view(PyArrayObject *arr, PyObject *ind, PyArrayObject **view)
{
    assert(PyDataType_ISLEGACY(PyArray_DESCR(arr)));  // 断言数组的数据类型是遗留类型
    *view = NULL;  // 将视图对象指针初始化为空

    /* first check for a single field name */
    /* 检查是否为 Unicode 对象 */
    if (PyUnicode_Check(ind)) {
        PyObject *tup;
        PyArray_Descr *fieldtype;
        npy_intp offset;

        /* 获取字段偏移量和数据类型 */
        tup = PyDict_GetItemWithError(PyDataType_FIELDS(PyArray_DESCR(arr)), ind);
        if (tup == NULL && PyErr_Occurred()) {
            return 0;
        }
        else if (tup == NULL){
            PyErr_Format(PyExc_ValueError, "no field of name %S", ind);
            return 0;
        }
        if (_unpack_field(tup, &fieldtype, &offset) < 0) {
            return 0;
        }

        /* 在新的偏移量和数据类型下查看数组 */
        Py_INCREF(fieldtype);
        *view = (PyArrayObject*)PyArray_NewFromDescr_int(
                Py_TYPE(arr),
                fieldtype,
                PyArray_NDIM(arr),
                PyArray_SHAPE(arr),
                PyArray_STRIDES(arr),
                PyArray_BYTES(arr) + offset,
                PyArray_FLAGS(arr),
                (PyObject *)arr, (PyObject *)arr,
                /* 仅对字符串进行子数组时不保留数据类型 */
                _NPY_ARRAY_ALLOW_EMPTY_STRING);
        if (*view == NULL) {
            return 0;
        }
        return 0;
    }

    /* 检查是否为字段名列表 */
    else if (PySequence_Check(ind) && !PyTuple_Check(ind)) {
        npy_intp seqlen, i;
        PyArray_Descr *view_dtype;

        seqlen = PySequence_Size(ind);

        /* 如果是虚假的类似序列且调用 len() 时出错则退出 */
        if (seqlen == -1) {
            PyErr_Clear();
            return -1;
        }
        /* 处理空列表作为整数索引的情况 */
        if (seqlen == 0) {
            return -1;
        }

        /* 检查所有项是否为字符串 */
        for (i = 0; i < seqlen; i++) {
            npy_bool is_string;
            PyObject *item = PySequence_GetItem(ind, i);
            if (item == NULL) {
                PyErr_Clear();
                return -1;
            }
            is_string = PyUnicode_Check(item);
            Py_DECREF(item);
            if (!is_string) {
                return -1;
            }
        }

        /* 调用 dtype 的下标操作 */
        view_dtype = arraydescr_field_subset_view(
                (_PyArray_LegacyDescr *)PyArray_DESCR(arr), ind);
        if (view_dtype == NULL) {
            return 0;
        }

        *view = (PyArrayObject*)PyArray_NewFromDescr_int(
                Py_TYPE(arr),
                view_dtype,
                PyArray_NDIM(arr),
                PyArray_SHAPE(arr),
                PyArray_STRIDES(arr),
                PyArray_DATA(arr),
                PyArray_FLAGS(arr),
                (PyObject *)arr, (PyObject *)arr,
                /* 仅对字符串进行子数组时不保留数据类型 */
                _NPY_ARRAY_ALLOW_EMPTY_STRING);

        if (*view == NULL) {
            return 0;
        }

        return 0;
    }
    /* 默认情况返回 -1 */
    return -1;
}

/*
 * General function for indexing a NumPy array with a Python object.
 */
NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op)
{
    // 定义索引类型、索引数量以及循环变量等
    int index_type;
    int index_num;
    int i, ndim, fancy_ndim;
    // 初始化类型为 NPY_cast_info 的结构体 cast_info，并设置 func 字段为 NULL
    NPY_cast_info cast_info = {.func = NULL};

    /*
     * Index info array. We can have twice as many indices as dimensions
     * (because of None). The + 1 is to not need to check as much.
     */
    // 定义索引信息数组，数组大小为 NPY_MAXDIMS * 2 + 1，用于存储索引信息
    npy_index_info indices[NPY_MAXDIMS * 2 + 1];

    // 定义视图和结果对象，并初始化为 NULL
    PyArrayObject *view = NULL;
    PyObject *result = NULL;

    // 定义 PyArrayMapIterObject 类型指针 mit，并初始化为 NULL
    PyArrayMapIterObject * mit = NULL;

    /* return fields if op is a string index */
    // 如果 op 是字符串索引，并且数组描述符有字段信息
    if (PyDataType_HASFIELDS(PyArray_DESCR(self))) {
        // 声明视图对象 view，并调用 _get_field_view 函数
        PyArrayObject *view;
        // 调用 _get_field_view 函数尝试获取字段视图，结果存入 view
        int ret = _get_field_view(self, op, &view);
        // 如果 ret 等于 0，则成功获取视图
        if (ret == 0){
            // 如果视图为 NULL，返回 NULL
            if (view == NULL) {
                return NULL;
            }
            // 返回视图对象
            return (PyObject*)view;
        }
    }

    // 准备索引信息
    index_type = prepare_index(self, op, indices, &index_num,
                               &ndim, &fancy_ndim, 1);

    // 如果准备索引失败，返回 NULL
    if (index_type < 0) {
        return NULL;
    }

    // 如果索引类型是完全整数索引
    else if (index_type == HAS_INTEGER) {
        // 获取数组中元素的指针
        char *item;
        if (get_item_pointer(self, &item, indices, index_num) < 0) {
            // 如果获取失败，跳转到 finish 标签
            goto finish;
        }
        // 创建并返回数组标量对象
        result = (PyObject *) PyArray_Scalar(item, PyArray_DESCR(self),
                                             (PyObject *)self);
        /* Because the index is full integer, we do not need to decref */
        // 因为索引是完全整数，不需要减少引用计数
        return result;
    }

    // 如果索引类型是布尔数组索引
    else if (index_type == HAS_BOOL) {
        // 使用布尔数组索引进行数组子脚本操作，并返回结果对象
        result = (PyObject *)array_boolean_subscript(self,
                                    (PyArrayObject *)indices[0].object,
                                    NPY_CORDER);
        // 跳转到 finish 标签
        goto finish;
    }

    // 如果索引类型是单个省略号索引
    else if (index_type == HAS_ELLIPSIS) {
        /*
         * TODO: Should this be a view or not? The only reason not would be
         *       optimization (i.e. of array[...] += 1) I think.
         *       Before, it was just self for a single ellipsis.
         */
        // 创建并返回数组的视图对象
        result = PyArray_View(self, NULL, NULL);
        /* A single ellipsis, so no need to decref */
        // 单个省略号索引，不需要减少引用计数
        return result;
    }

    /*
     * View based indexing.
     * There are two cases here. First we need to create a simple view,
     * second we need to create a (possibly invalid) view for the
     * subspace to the fancy index. This procedure is identical.
     */
    # 如果索引类型包含切片、新轴、省略号或整数
    else if (index_type & (HAS_SLICE | HAS_NEWAXIS |
                           HAS_ELLIPSIS | HAS_INTEGER)) {
        # 尝试从索引中获取视图对象，若失败则跳转到结束
        if (get_view_from_index(self, &view, indices, index_num,
                                (index_type & HAS_FANCY)) < 0) {
            goto finish;
        }

        '''
         * 存在标量数组，需要强制复制以模拟花式索引。
         '''
        if (index_type & HAS_SCALAR_ARRAY) {
            result = PyArray_NewCopy(view, NPY_KEEPORDER);
            goto finish;
        }
    }

    '''
     * 如果没有花式索引，直接使用视图对象作为结果。
     '''
    # 如果没有花式索引，则结果是视图对象本身
    if (!(index_type & HAS_FANCY)) {
        result = (PyObject *)view;
        Py_INCREF(result);
        goto finish;
    }

    '''
     * 特殊情况：非常简单的一维花式索引，尽管如此常见。
     * 这不仅节省了迭代器的设置时间，而且更快（必须完全是花式索引，
     * 因为这里不支持0维布尔值）。
     '''
    if (index_type == HAS_FANCY && index_num == 1) {
        /* 如果索引类型为 HAS_FANCY 并且索引数量为 1 */

        PyArrayObject *ind = (PyArrayObject*)indices[0].object;
        /* 将索引对象转换为 PyArrayObject 类型 */

        if (PyArray_TRIVIALLY_ITERABLE(ind) &&
                /* 检查索引是否足够简单 */
                PyArray_ITEMSIZE(ind) == sizeof(npy_intp) &&
                PyArray_DESCR(ind)->kind == 'i' &&
                IsUintAligned(ind) &&
                PyDataType_ISNOTSWAPPED(PyArray_DESCR(ind))) {
            /* 检查类型是否等同于 INTP */
            
            Py_INCREF(PyArray_DESCR(self));
            /* 增加对数组描述符的引用计数 */

            result = PyArray_NewFromDescr(&PyArray_Type,
                                          PyArray_DESCR(self),
                                          PyArray_NDIM(ind),
                                          PyArray_SHAPE(ind),
                                          NULL, NULL,
                                          /* 与索引顺序相同 */
                                          PyArray_ISFORTRAN(ind) ?
                                              NPY_ARRAY_F_CONTIGUOUS : 0,
                                          NULL);
            /* 使用给定的描述符和形状创建新的数组对象 */

            if (result == NULL) {
                goto finish;
            }

            NPY_ARRAYMETHOD_FLAGS transfer_flags;
            npy_intp itemsize = PyArray_ITEMSIZE(self);
            /* 获取数组元素的大小 */
            int is_aligned = IsUintAligned(self);
            /* 检查数组是否按无符号整数对齐 */

            if (PyArray_GetDTypeTransferFunction(is_aligned,
                    itemsize, itemsize,
                    PyArray_DESCR(self), PyArray_DESCR((PyArrayObject *)result),
                    0, &cast_info, &transfer_flags) != NPY_SUCCEED) {
                goto finish;
            }

            if (mapiter_trivial_get(
                    self, ind, (PyArrayObject *)result, is_aligned, &cast_info) < 0) {
                /* 调用 mapiter_trivial_get 函数获取数据 */
                Py_DECREF(result);
                result = NULL;
                goto finish;
            }

            goto wrap_out_array;
            /* 跳转到 wrap_out_array 标签处 */
        }
    }

    /* 必须使用花式索引。视图是子空间。 */
    mit = (PyArrayMapIterObject *)PyArray_MapIterNew(indices, index_num,
                                                     index_type,
                                                     ndim, fancy_ndim,
                                                     self, view, 0,
                                                     NPY_ITER_READONLY,
                                                     NPY_ITER_WRITEONLY,
                                                     NULL, PyArray_DESCR(self));
    /* 创建数组映射迭代器对象 */

    if (mit == NULL) {
        goto finish;
    }
    if (mit->num_fancy > 1 || mit->size == 0) {
        /*
         * 如果 num_fancy 大于 1 或者 size 等于 0，
         * 则需要进行内部循环检查索引；否则，
         * 在广播发生时先进行索引检查，因为这样速度更快，
         * 而且大多数情况下没有太大的开销。
         * 然而，对于 size == 0 的情况，内部循环优化会跳过索引检查。
         */
        if (PyArray_MapIterCheckIndices(mit) < 0) {
            goto finish;
        }
    }

    /* 重置外部迭代器 */
    if (NpyIter_Reset(mit->outer, NULL) < 0) {
        goto finish;
    }

    /*
     * 对齐信息（由于我们使用缓冲区，不需要进行交换），
     * 还可以检查 extra_op 是否已经缓冲，但这通常不重要。
     */
    int is_aligned = IsUintAligned(self) && IsUintAligned(mit->extra_op);
    /*
     * 注意：获取数据类型转换函数时实际上不会执行类型转换，
     *       因此我们目前不必进行完整的检查（例如浮点数错误）（不像赋值操作那样）。
     */
    int meth_flags = NpyIter_GetTransferFlags(mit->outer);
    if (mit->extra_op_iter) {
        int extra_op_flags = NpyIter_GetTransferFlags(mit->extra_op_iter);
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, extra_op_flags);
    }

    if (mit->subspace_iter != NULL) {
        int extra_op_flags = NpyIter_GetTransferFlags(mit->subspace_iter);
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, extra_op_flags);

        NPY_ARRAYMETHOD_FLAGS transfer_flags;
        npy_intp fixed_strides[2];
        /*
         * 获取 dtype 转换函数，由于没有缓冲区，这是安全的。
         */
        NpyIter_GetInnerFixedStrideArray(mit->subspace_iter, fixed_strides);

        if (PyArray_GetDTypeTransferFunction(is_aligned,
                fixed_strides[0], fixed_strides[1],
                PyArray_DESCR(self), PyArray_DESCR(mit->extra_op),
                0, &cast_info, &transfer_flags) != NPY_SUCCEED) {
            goto finish;
        }
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, transfer_flags);
    }
    else {
        /* 可能需要一个通用的复制函数（仅用于引用和奇怪的大小） */
        NPY_ARRAYMETHOD_FLAGS transfer_flags;
        npy_intp itemsize = PyArray_ITEMSIZE(self);

        if (PyArray_GetDTypeTransferFunction(1,
                itemsize, itemsize,
                PyArray_DESCR(self), PyArray_DESCR(self),
                0, &cast_info, &transfer_flags) != NPY_SUCCEED) {
            goto finish;
        }
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, transfer_flags);
    }

    if (mapiter_get(mit, &cast_info, meth_flags, is_aligned) < 0) {
        goto finish;
    }

    result = (PyObject *)mit->extra_op;
    Py_INCREF(result);

    if (mit->consec) {
        PyArray_MapIterSwapAxes(mit, (PyArrayObject **)&result, 1);
    }

  wrap_out_array:
    if (!PyArray_CheckExact(self)) {
        /*
         * 如果 self 不是一个确切的 PyArray 对象，需要创建一个新的数组，
         * 就好像旧数组从未存在过。
         */
        PyArrayObject *tmp_arr = (PyArrayObject *)result;

        // 增加临时数组的描述符的引用计数
        Py_INCREF(PyArray_DESCR(tmp_arr));

        // 使用临时数组的信息创建一个新的 PyArray 对象
        result = PyArray_NewFromDescrAndBase(
                Py_TYPE(self),
                PyArray_DESCR(tmp_arr),
                PyArray_NDIM(tmp_arr),
                PyArray_SHAPE(tmp_arr),
                PyArray_STRIDES(tmp_arr),
                PyArray_BYTES(tmp_arr),
                PyArray_FLAGS(tmp_arr),
                (PyObject *)self, (PyObject *)tmp_arr);
        Py_DECREF(tmp_arr);
        // 如果创建失败，跳转至结束标签 finish
        if (result == NULL) {
            goto finish;
        }
    }

  finish:
    // 释放类型转换信息结构体的内存
    NPY_cast_info_xfree(&cast_info);
    // 释放 mit 对象的引用
    Py_XDECREF(mit);
    // 释放 view 对象的引用
    Py_XDECREF(view);
    /* 清理索引数组中的对象引用 */
    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }
    // 返回 result 对象
    return result;
/*
 * Python C-Api level item assignment (implementation for PySequence_SetItem)
 *
 * Negative indices are not accepted because PySequence_SetItem converts
 * them to positive indices before calling this.
 */
NPY_NO_EXPORT int
array_assign_item(PyArrayObject *self, Py_ssize_t i, PyObject *op)
{
    npy_index_info indices[2];  // 定义索引信息结构体数组，用于存储索引信息

    if (op == NULL) {  // 如果操作对象为空指针
        PyErr_SetString(PyExc_ValueError,
                        "cannot delete array elements");  // 设置错误消息：无法删除数组元素
        return -1;  // 返回错误代码
    }
    if (PyArray_FailUnlessWriteable(self, "assignment destination") < 0) {
        return -1;  // 如果数组不可写，则返回错误代码
    }
    if (PyArray_NDIM(self) == 0) {  // 如果数组维度为0
        PyErr_SetString(PyExc_IndexError,
                        "too many indices for array");  // 设置错误消息：数组索引过多
        return -1;  // 返回错误代码
    }

    if (i < 0) {  // 如果索引值为负数
        /* This is an error, but undo PySequence_SetItem fix for message */
        i -= PyArray_DIM(self, 0);  // 对负数索引进行修正
    }

    indices[0].value = i;  // 设置第一个索引值
    indices[0].type = HAS_INTEGER;  // 设置索引类型为整数索引
    if (PyArray_NDIM(self) == 1) {  // 如果数组维度为1
        char *item;
        if (get_item_pointer(self, &item, indices, 1) < 0) {
            return -1;  // 获取数组元素指针失败，则返回错误代码
        }
        if (PyArray_Pack(PyArray_DESCR(self), item, op) < 0) {
            return -1;  // 将 Python 对象打包为数组元素失败，则返回错误代码
        }
    }
    else {  // 如果数组维度大于1
        PyArrayObject *view;

        indices[1].value = PyArray_NDIM(self) - 1;  // 设置第二个索引值为最后一个维度索引
        indices[1].type = HAS_ELLIPSIS;  // 设置第二个索引类型为省略号索引
        if (get_view_from_index(self, &view, indices, 2, 0) < 0) {
            return -1;  // 从索引获取视图对象失败，则返回错误代码
        }
        if (PyArray_CopyObject(view, op) < 0) {
            Py_DECREF(view);
            return -1;  // 复制对象到视图对象失败，则返回错误代码
        }
        Py_DECREF(view);
    }
    return 0;  // 返回成功代码
}
    /* Full integer index */
    // 如果索引类型为整数数组
    if (index_type == HAS_INTEGER) {
        char *item;
        // 获取索引的指针
        if (get_item_pointer(self, &item, indices, index_num) < 0) {
            return -1;
        }
        // 将 item 打包到 self 中
        if (PyArray_Pack(PyArray_DESCR(self), item, op) < 0) {
            return -1;
        }
        /* integers do not store objects in indices */
        // 整数索引不存储对象
        return 0;
    }

    /* Single boolean array */
    // 如果索引类型为布尔数组
    if (index_type == HAS_BOOL) {
        if (!PyArray_Check(op)) {
            // 如果 op 不是数组，则创建一个新的数组 tmp_arr
            Py_INCREF(PyArray_DESCR(self));
            tmp_arr = (PyArrayObject *)PyArray_FromAny(op,
                                                   PyArray_DESCR(self), 0, 0,
                                                   NPY_ARRAY_FORCECAST, NULL);
            if (tmp_arr == NULL) {
                goto fail;
            }
        }
        else {
            // 如果 op 已经是数组，则直接增加引用计数
            Py_INCREF(op);
            tmp_arr = (PyArrayObject *)op;
        }

        // 使用布尔子脚本进行数组分配
        if (array_assign_boolean_subscript(self,
                                           (PyArrayObject *)indices[0].object,
                                           tmp_arr, NPY_CORDER) < 0) {
            goto fail;
        }
        // 成功时跳转到 success 标签
        goto success;
    }


    /*
     * Single ellipsis index, no need to create a new view.
     * Note that here, we do *not* go through self.__getitem__ for subclasses
     * (defchar array failed then, due to uninitialized values...)
     */
    // 如果索引类型为省略号
    else if (index_type == HAS_ELLIPSIS) {
        if ((PyObject *)self == op) {
            /*
             * CopyObject does not handle this case gracefully and
             * there is nothing to do. Removing the special case
             * will cause segfaults, though it is unclear what exactly
             * happens.
             */
            // 如果 self 和 op 相同，则无需处理，直接返回
            return 0;
        }
        /* we can just use self, but incref for error handling */
        // 我们可以直接使用 self，但需要增加引用计数以处理错误
        Py_INCREF((PyObject *)self);
        view = self;
    }

    /*
     * WARNING: There is a huge special case here. If this is not a
     *          base class array, we have to get the view through its
     *          very own index machinery.
     *          Many subclasses should probably call __setitem__
     *          with a base class ndarray view to avoid this.
     */
    // 如果索引类型既不是花式索引也不是标量数组，并且 self 不是精确的数组类型
    else if (!(index_type & (HAS_FANCY | HAS_SCALAR_ARRAY))
                && !PyArray_CheckExact(self)) {
        // 通过索引获取视图
        view = (PyArrayObject *)PyObject_GetItem((PyObject *)self, ind);
        if (view == NULL) {
            goto fail;
        }
        // 确保返回的是一个数组
        if (!PyArray_Check(view)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Getitem not returning array");
            goto fail;
        }
    }

    /*
     * View based indexing.
     * There are two cases here. First we need to create a simple view,
     * second we need to create a (possibly invalid) view for the
     * subspace to the fancy index. This procedure is identical.
     */
    // 基于视图的索引操作
    # 如果索引类型中包含切片、新轴、省略或整数
    else if (index_type & (HAS_SLICE | HAS_NEWAXIS |
                           HAS_ELLIPSIS | HAS_INTEGER)) {
        # 如果需要从索引中获取视图，并且不是花式索引，则执行以下逻辑
        if (get_view_from_index(self, &view, indices, index_num,
                                (index_type & HAS_FANCY)) < 0) {
            # 获取视图失败，则跳转到失败处
            goto fail;
        }
    }
    else {
        # 否则，将视图设为NULL
        view = NULL;
    }

    /* 如果没有花式索引，直接将视图拷贝给操作数 */
    if (!(index_type & HAS_FANCY)) {
        # 如果能够成功拷贝视图给操作数
        if (PyArray_CopyObject(view, op) < 0) {
            # 拷贝失败，则跳转到失败处
            goto fail;
        }
        # 跳转到成功处
        goto success;
    }

    # 如果操作数不是一个数组
    if (!PyArray_Check(op)) {
        /*
         * 如果数组是对象数组，并且操作数是一个序列，
         * 尽管普通赋值可行，但转换值到数组可能不合法。
         * 因此，分配一个正确大小的临时数组，并使用普通赋值处理这种情况。
         */
        if (PyDataType_REFCHK(descr) && PySequence_Check(op)) {
            # 临时数组设为NULL
            tmp_arr = NULL;
        }
        else {
            /* 没有可能使用花式索引，因此只需创建一个数组 */
            # 增加描述符的引用计数
            Py_INCREF(descr);
            # 从操作数创建一个数组，强制转换类型
            tmp_arr = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0,
                                                    NPY_ARRAY_FORCECAST, NULL);
            # 如果创建临时数组失败，则跳转到失败处
            if (tmp_arr == NULL) {
                goto fail;
            }
        }
    }
    else {
        # 增加操作数的引用计数
        Py_INCREF(op);
        # 将操作数强制转换为数组
        tmp_arr = (PyArrayObject *)op;
    }

    /*
     * 特殊情况处理非常简单的一维花式索引，这种情况相当常见。
     * 这不仅节省了迭代器的设置时间，而且速度更快（必须完全是花式的，
     * 因为这里不支持0维布尔值）
     */
    if (index_type == HAS_FANCY &&
            index_num == 1 && tmp_arr) {
        /* 当索引类型为HAS_FANCY且索引数量为1，并且tmp_arr不为空时进入条件判断 */

        PyArrayObject *ind = (PyArrayObject*)indices[0].object;
        /* 将索引数组转换为PyArrayObject类型 */

        /* 检查类型是否等效 */
        if (PyArray_EquivTypes(PyArray_DESCR(self),
                                   PyArray_DESCR(tmp_arr)) &&
                /*
                 * 类型要么等效，要么值必须是标量
                 */
                (PyArray_EQUIVALENTLY_ITERABLE(ind, tmp_arr,
                                               PyArray_TRIVIALLY_ITERABLE_OP_READ,
                                               PyArray_TRIVIALLY_ITERABLE_OP_READ) ||
                 (PyArray_NDIM(tmp_arr) == 0 &&
                        PyArray_TRIVIALLY_ITERABLE(ind))) &&
                /* 检查类型是否等效于INTP */
                PyArray_ITEMSIZE(ind) == sizeof(npy_intp) &&
                PyArray_DESCR(ind)->kind == 'i' &&
                IsUintAligned(ind) &&
                PyDataType_ISNOTSWAPPED(PyArray_DESCR(ind))) {

            NPY_ARRAYMETHOD_FLAGS transfer_flags;
            npy_intp itemsize = PyArray_ITEMSIZE(self);
            int is_aligned = IsUintAligned(self) && IsUintAligned(tmp_arr);

            if (PyArray_GetDTypeTransferFunction(is_aligned,
                    itemsize, itemsize,
                    PyArray_DESCR(self), PyArray_DESCR(self),
                    0, &cast_info, &transfer_flags) != NPY_SUCCEED) {
                goto fail;
            }

            /* trivial_set函数检查索引，然后进行设置 */
            if (mapiter_trivial_set(
                    self, ind, tmp_arr, is_aligned, &cast_info) < 0) {
                goto fail;
            }
            goto success;
        }
    }

    /*
     * 注意：如果tmp_arr尚未分配，则应由mit处理分配。
     *       NPY_ITER_READWRITE 对于自动分配是必要的。
     *       Readwrite 模式不允许正确地进行广播，但这样的操作数总是具有完整的大小。
     */
    mit = (PyArrayMapIterObject *)PyArray_MapIterNew(indices,
                                             index_num, index_type,
                                             ndim, fancy_ndim, self,
                                             view, 0,
                                             NPY_ITER_WRITEONLY,
                                             ((tmp_arr == NULL) ?
                                                  NPY_ITER_READWRITE :
                                                  NPY_ITER_READONLY),
                                             tmp_arr, descr);

    if (mit == NULL) {
        goto fail;
    }
    if (tmp_arr == NULL) {
        /* 如果 tmp_arr 为空指针，则需要填充额外的操作，首先需要交换 */
        tmp_arr = mit->extra_op;
        // 增加 tmp_arr 的引用计数，确保其不被释放
        Py_INCREF(tmp_arr);
        // 如果 mit->consec 为真，调用 PyArray_MapIterSwapAxes 函数交换轴
        if (mit->consec) {
            PyArray_MapIterSwapAxes(mit, &tmp_arr, 1);
            // 如果交换后 tmp_arr 为空，跳转到 fail 标签处处理错误
            if (tmp_arr == NULL) {
                goto fail;
            }
        }
        // 将 tmp_arr 的内容复制到 op 中
        if (PyArray_CopyObject(tmp_arr, op) < 0) {
             goto fail;
        }
    }

    // 检查 MapIter 对象的索引是否有效
    if (PyArray_MapIterCheckIndices(mit) < 0) {
        goto fail;
    }

    /*
     * 对齐信息（由于我们使用缓冲区，不需要交换），可以检查 extra_op 是否已缓冲，
     * 但这通常很少有影响。
     */
    int is_aligned = IsUintAligned(self) && IsUintAligned(mit->extra_op);
    // 获取迭代器的传输标志
    int meth_flags = NpyIter_GetTransferFlags(mit->outer);

    // 如果存在额外操作的迭代器
    if (mit->extra_op_iter) {
        int extra_op_flags = NpyIter_GetTransferFlags(mit->extra_op_iter);
        // 合并方法标志
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, extra_op_flags);
    }

    // 如果存在子空间迭代器
    if (mit->subspace_iter != NULL) {
        int extra_op_flags = NpyIter_GetTransferFlags(mit->subspace_iter);
        // 合并方法标志
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, extra_op_flags);

        NPY_ARRAYMETHOD_FLAGS transfer_flags;
        npy_intp fixed_strides[2];

        /*
         * 获取 dtype 传输函数，由于没有缓冲区，这是安全的。
         */
        NpyIter_GetInnerFixedStrideArray(mit->subspace_iter, fixed_strides);

        // 获取数据类型的传输函数，设置转换标志
        if (PyArray_GetDTypeTransferFunction(is_aligned,
                fixed_strides[1], fixed_strides[0],
                PyArray_DESCR(mit->extra_op), PyArray_DESCR(self),
                0, &cast_info, &transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
        // 合并方法标志
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, transfer_flags);
    }
    else {
        /* 可能需要一个通用的复制函数（仅适用于引用和奇怪的大小） */
        NPY_ARRAYMETHOD_FLAGS transfer_flags;
        npy_intp itemsize = PyArray_ITEMSIZE(self);

        // 获取数据类型的传输函数，设置转换标志
        if (PyArray_GetDTypeTransferFunction(1,
                itemsize, itemsize,
                PyArray_DESCR(self), PyArray_DESCR(self),
                0, &cast_info, &transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
        // 合并方法标志
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, transfer_flags);
    }

    // 如果方法标志不包含 NPY_METH_NO_FLOATINGPOINT_ERRORS，则清除浮点状态
    if (!(meth_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)mit);
    }

    // 现在可以重置外部迭代器（延迟 bufalloc）
    if (NpyIter_Reset(mit->outer, NULL) < 0) {
        goto fail;
    }

    /*
     * 可能需要一个类型转换检查，但显然大多数赋值操作不关心安全转换。
     */
    // 如果设置 MapIter 的函数失败，则跳转到 fail 处理错误
    if (mapiter_set(mit, &cast_info, meth_flags, is_aligned) < 0) {
        goto fail;
    }

    // 如果方法标志不包含 NPY_METH_NO_FLOATINGPOINT_ERRORS，则获取浮点错误状态
    if (!(meth_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier((char *)mit);
        // 如果获取到浮点错误并且处理浮点错误失败，则跳转到 fail 处理错误
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            goto fail;
        }

    // 如果 tmp_arr 为空指针，则需要填充额外的操作，首先需要交换
    if (tmp_arr == NULL) {
        tmp_arr = mit->extra_op;
        // 增加 tmp_arr 的引用计数，确保其不被释放
        Py_INCREF(tmp_arr);
        // 如果 mit->consec 为真，调用 PyArray_MapIterSwapAxes 函数交换轴
        if (mit->consec) {
            PyArray_MapIterSwapAxes(mit, &tmp_arr, 1);
            // 如果交换后 tmp_arr 为空，跳转到 fail 标签处处理错误
            if (tmp_arr == NULL) {
                goto fail;
            }
        }
        // 将 tmp_arr 的内容复制到 op 中
        if (PyArray_CopyObject(tmp_arr, op) < 0) {
             goto fail;
        }
    }

    // 检查 MapIter 对象的索引是否有效
    if (PyArray_MapIterCheckIndices(mit) < 0) {
        goto fail;
    }

    /*
     * 对齐信息（由于我们使用缓冲区，不需要交换），可以检查 extra_op 是否已缓冲，
     * 但这通常很少有影响。
     */
    int is_aligned = IsUintAligned(self) && IsUintAligned(mit->extra_op);
    // 获取迭代器的传输标志
    int meth_flags = NpyIter_GetTransferFlags(mit->outer);

    // 如果存在额外操作的迭代器
    if (mit->extra_op_iter) {
        int extra_op_flags = NpyIter_GetTransferFlags(mit->extra_op_iter);
        // 合并方法标志
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, extra_op_flags);
    }

    // 如果存在子空间迭代器
    if (mit->subspace_iter != NULL) {
        int extra_op_flags = NpyIter_GetTransferFlags(mit->subspace_iter);
        // 合并方法标志
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, extra_op_flags);

        NPY_ARRAYMETHOD_FLAGS transfer_flags;
        npy_intp fixed_strides[2];

        /*
         * 获取 dtype 传输函数，由于没有缓冲区，这是安全的。
         */
        NpyIter_GetInnerFixedStrideArray(mit->subspace_iter, fixed_strides);

        // 获取数据类型的传输函数，设置转换标志
        if (PyArray_GetDTypeTransferFunction(is_aligned,
                fixed_strides[1], fixed_strides[0],
                PyArray_DESCR(mit->extra_op), PyArray_DESCR(self),
                0, &cast_info, &transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
        // 合并方法标志
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, transfer_flags);
    }
    else {
        /* 可能需要一个通用的复制函数（仅适用于引用和奇怪的大小） */
        NPY_ARRAYMETHOD_FLAGS transfer_flags;
        npy_intp itemsize = PyArray_ITEMSIZE(self);

        // 获取数据类型的传输函数，设置转换标志
        if (PyArray_GetDTypeTransferFunction(1,
                itemsize, itemsize,
                PyArray_DESCR(self), PyArray_DESCR(self),
                0, &cast_info, &transfer_flags) != NPY_SUCCEED) {
            goto fail;
        }
        // 合并方法标志
        meth_flags = PyArrayMethod_COMBINED_FLAGS(meth_flags, transfer_flags);
    }

    // 如果方法标志不包含 NPY_METH_NO_FLOATINGPOINT_ERRORS，则清除浮点状态
    if (!(meth_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)mit);
    }

    // 现在可以重置外部迭代器（延迟 bufalloc）
    if (NpyIter_Reset(mit->outer, NULL) < 0) {
        goto fail;
    }

    /*
     * 可能需要一个类型转换检查，但显然大多数赋值操作不关心安全转换。
     */
    // 如果设置 MapIter 的函数失败，则跳转到 fail 处理错误
    if (mapiter_set(mit, &cast_info, meth_flags, is_aligned) < 0) {
        goto fail;
    }

    // 如果方法标志不包含 NPY_METH_NO_FLOATINGPOINT_ERRORS，则获取浮点错误状态
    if (!(meth_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier((char *)mit);
        // 如果获取到浮点错误并且处理浮点错误失败，则跳转到 fail 处理错误
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            goto fail;
        }
    }
    }

    // 减少 mit 的引用计数，释放其占用的内存
    Py_DECREF(mit);
    // 跳转到成功的标签，表示函数成功执行
    goto success;

    /* Clean up temporary variables and indices */
  fail:
    // 释放视图对象的内存
    Py_XDECREF((PyObject *)view);
    // 释放临时数组对象的内存
    Py_XDECREF((PyObject *)tmp_arr);
    // 释放 mit 对象的内存
    Py_XDECREF((PyObject *)mit);
    // 释放类型转换信息结构体的内存
    NPY_cast_info_xfree(&cast_info);

    // 循环释放索引数组中每个元素对象的内存
    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }
    // 返回 -1 表示执行失败
    return -1;

  success:
    // 释放视图对象的内存
    Py_XDECREF((PyObject *)view);
    // 释放临时数组对象的内存
    Py_XDECREF((PyObject *)tmp_arr);
    // 释放类型转换信息结构体的内存
    NPY_cast_info_xfree(&cast_info);

    // 循环释放索引数组中每个元素对象的内存
    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }
    // 返回 0 表示执行成功
    return 0;
/*********************** Subscript Array Iterator *************************
 *                                                                        *
 * This object handles subscript behavior for array objects.              *
 * It is an iterator object with a next method                            *
 * It abstracts the n-dimensional mapping behavior to make the looping    *
 * code more understandable (maybe)                                      *
 * and so that indexing can be set up ahead of time                      *
 */

/*
 * This function takes a Boolean array and constructs index objects and
 * iterators as if nonzero(Bool) had been called
 *
 * Must not be called on a 0-d array.
 */
static int
_nonzero_indices(PyObject *myBool, PyArrayObject **arrays)
{
    PyArray_Descr *typecode;
    PyArrayObject *ba = NULL, *new = NULL;
    int nd, j;
    npy_intp size, i, count;
    npy_bool *ptr;
    npy_intp coords[NPY_MAXDIMS], dims_m1[NPY_MAXDIMS];
    npy_intp *dptr[NPY_MAXDIMS];
    static npy_intp one = 1;
    NPY_BEGIN_THREADS_DEF;

    // 创建一个描述符类型为 NPY_BOOL 的 PyArray_Descr 对象
    typecode = PyArray_DescrFromType(NPY_BOOL);
    // 从任意对象 myBool 创建一个布尔数组 ba
    ba = (PyArrayObject *)PyArray_FromAny(myBool, typecode, 0, 0,
                                          NPY_ARRAY_CARRAY, NULL);
    // 如果创建失败，返回错误码 -1
    if (ba == NULL) {
        return -1;
    }
    // 获取数组 ba 的维度数
    nd = PyArray_NDIM(ba);

    // 将数组指针初始化为空
    for (j = 0; j < nd; j++) {
        arrays[j] = NULL;
    }
    // 获取数组 ba 的总大小
    size = PyArray_SIZE(ba);
    // 获取数组 ba 的数据指针，并转换为布尔型指针
    ptr = (npy_bool *)PyArray_DATA(ba);

    /*
     * 预先确定有多少个非零条目，
     * 忽略输入的维度信息，因为它是一个 CARRAY
     */
    // 调用 count_boolean_trues 函数计算布尔数组中的真值数量
    count = count_boolean_trues(1, (char*)ptr, &size, &one);

    /* 为每个维度创建大小为 count 的索引数组 */
    for (j = 0; j < nd; j++) {
        // 创建一个新的 intp 类型数组 new，大小为 count
        new = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
            1, &count, NULL, NULL,
            0, NULL);
        // 如果创建失败，跳转到失败标签
        if (new == NULL) {
            goto fail;
        }
        // 将新创建的数组赋值给 arrays[j]
        arrays[j] = new;

        // 获取新数组 new 的数据指针，并转换为 intp 型指针
        dptr[j] = (npy_intp *)PyArray_DATA(new);
        // 初始化坐标数组的当前维度为 0
        coords[j] = 0;
        // 将 ba 在当前维度上的最大索引保存到 dims_m1 数组中
        dims_m1[j] = PyArray_DIMS(ba)[j] - 1;
    }
    // 如果 count 为 0，直接跳转到结束标签
    if (count == 0) {
        goto finish;
    }

    /*
     * 遍历布尔数组，并复制非零条目的坐标
     */
    // 启动多线程处理，根据 size 的大小决定是否启动多线程
    NPY_BEGIN_THREADS_THRESHOLDED(size);
    // 循环遍历从 0 到 size-1 的索引 i
    for (i = 0; i < size; i++) {
        // 检查 ptr 指向的值是否为真，如果是则执行内部代码块
        if (*(ptr++)) {
            // 遍历从 0 到 nd-1 的索引 j
            for (j = 0; j < nd; j++) {
                // 将 coords[j] 的值赋给 dptr[j] 所指向的位置，并移动 dptr[j] 指针
                *(dptr[j]++) = coords[j];
            }
        }
        /* Borrowed from ITER_NEXT macro */
        // 从 nd-1 开始到 0 的索引 j 进行遍历
        for (j = nd - 1; j >= 0; j--) {
            // 如果 coords[j] 小于 dims_m1[j]，则将 coords[j] 增加 1 并跳出循环
            if (coords[j] < dims_m1[j]) {
                coords[j]++;
                break;
            }
            // 否则将 coords[j] 设为 0
            else {
                coords[j] = 0;
            }
        }
    }
    NPY_END_THREADS;

finish:
    // 释放 ba 对象的引用
    Py_DECREF(ba);
    // 返回 nd 的值
    return nd;

fail:
    // 失败时释放所有 arrays[j] 的引用
    for (j = 0; j < nd; j++) {
        Py_XDECREF(arrays[j]);
    }
    // 释放 ba 对象的引用
    Py_XDECREF(ba);
    // 返回 -1 表示失败
    return -1;
/* 重置映射迭代器到开始位置 */
NPY_NO_EXPORT int
PyArray_MapIterReset(PyArrayMapIterObject *mit)
{
    npy_intp indval;  // 定义整数索引值
    char *baseptrs[2];  // 定义两个字符指针数组
    int i;  // 定义循环计数变量

    if (mit->size == 0) {  // 如果迭代器大小为0，则直接返回0
        return 0;
    }

    if (!NpyIter_Reset(mit->outer, NULL)) {  // 重置外部迭代器，如果失败则返回-1
        return -1;
    }
    if (mit->extra_op_iter) {  // 如果存在额外操作迭代器
        if (!NpyIter_Reset(mit->extra_op_iter, NULL)) {  // 重置额外操作迭代器，如果失败则返回-1
            return -1;
        }

        baseptrs[1] = mit->extra_op_ptrs[0];  // 设置第二个基础指针
    }

    baseptrs[0] = mit->baseoffset;  // 设置第一个基础指针为偏移量

    for (i = 0; i < mit->num_fancy; i++) {  // 遍历所有的特殊索引
        indval = *((npy_intp*)mit->outer_ptrs[i]);  // 获取外部指针的索引值
        if (indval < 0) {  // 如果索引值小于0，则加上特殊维度值
            indval += mit->fancy_dims[i];
        }
        baseptrs[0] += indval * mit->fancy_strides[i];  // 根据索引值和步幅计算基础指针偏移量
    }
    mit->dataptr = baseptrs[0];  // 设置数据指针为第一个基础指针

    if (mit->subspace_iter) {  // 如果存在子空间迭代器
        if (!NpyIter_ResetBasePointers(mit->subspace_iter, baseptrs, NULL)) {  // 重置子空间迭代器的基础指针，如果失败则返回-1
            return -1;
        }
        mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->subspace_iter);  // 获取子空间迭代器的内部循环大小
    }
    else {
        mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->outer);  // 否则获取外部迭代器的内部循环大小
    }

    return 0;  // 返回0表示成功
}


/*
 * 这个函数需要更新映射迭代器的状态，并将 mit->dataptr 指向下一个对象的内存位置
 *
 * 需要注意，此函数不处理额外操作数，但为旧的（已暴露的）API提供兼容性。
 */
NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *mit)
{
    int i;  // 定义循环计数变量
    char *baseptr;  // 定义字符指针变量
    npy_intp indval;  // 定义整数索引值

    if (mit->subspace_iter) {  // 如果存在子空间迭代器
        if (--mit->iter_count > 0) {  // 如果内部循环计数大于0
            mit->subspace_ptrs[0] += mit->subspace_strides[0];  // 子空间指针加上子空间步幅
            mit->dataptr = mit->subspace_ptrs[0];  // 设置数据指针为子空间指针
            return;
        }
        else if (mit->subspace_next(mit->subspace_iter)) {  // 否则如果子空间迭代器的下一个有效
            mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->subspace_iter);  // 获取子空间迭代器的内部循环大小
            mit->dataptr = mit->subspace_ptrs[0];  // 设置数据指针为子空间指针
        }
        else {  // 否则
            if (!mit->outer_next(mit->outer)) {  // 如果外部迭代器的下一个无效，则返回
                return;
            }

            baseptr = mit->baseoffset;  // 设置基础指针为偏移量

            for (i = 0; i < mit->num_fancy; i++) {  // 遍历所有的特殊索引
                indval = *((npy_intp*)mit->outer_ptrs[i]);  // 获取外部指针的索引值
                if (indval < 0) {  // 如果索引值小于0，则加上特殊维度值
                    indval += mit->fancy_dims[i];
                }
                baseptr += indval * mit->fancy_strides[i];  // 根据索引值和步幅计算基础指针偏移量
            }
            NpyIter_ResetBasePointers(mit->subspace_iter, &baseptr, NULL);  // 重置子空间迭代器的基础指针
            mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->subspace_iter);  // 获取子空间迭代器的内部循环大小

            mit->dataptr = mit->subspace_ptrs[0];  // 设置数据指针为子空间指针
        }
    }
    else {
        // 如果迭代计数大于0，执行以下操作
        if (--mit->iter_count > 0) {
            // 将基础指针设置为迭代器的基础偏移量
            baseptr = mit->baseoffset;

            // 遍历所有的“花式”索引
            for (i = 0; i < mit->num_fancy; i++) {
                // 增加外部指针的步长
                mit->outer_ptrs[i] += mit->outer_strides[i];

                // 读取当前外部指针的整数值
                indval = *((npy_intp*)mit->outer_ptrs[i]);
                // 如果该值小于0，加上“花式”维度的大小
                if (indval < 0) {
                    indval += mit->fancy_dims[i];
                }
                // 基础指针加上计算出的索引值乘以“花式”步长
                baseptr += indval * mit->fancy_strides[i];
            }

            // 设置数据指针为新的基础指针位置，并返回
            mit->dataptr = baseptr;
            return;
        }
        else {
            // 如果迭代计数不大于0，检查是否可以获取下一个外部迭代器
            if (!mit->outer_next(mit->outer)) {
                // 如果无法获取下一个外部迭代器，直接返回
                return;
            }
            // 重新设置迭代计数为当前外部迭代器的内部循环大小
            mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->outer);
            // 将基础指针设置为迭代器的基础偏移量
            baseptr = mit->baseoffset;

            // 再次遍历所有的“花式”索引
            for (i = 0; i < mit->num_fancy; i++) {
                // 读取当前外部指针的整数值
                indval = *((npy_intp*)mit->outer_ptrs[i]);
                // 如果该值小于0，加上“花式”维度的大小
                if (indval < 0) {
                    indval += mit->fancy_dims[i];
                }
                // 基础指针加上计算出的索引值乘以“花式”步长
                baseptr += indval * mit->fancy_strides[i];
            }

            // 设置数据指针为新的基础指针位置
            mit->dataptr = baseptr;
        }
    }
/**
 * Fill information about the iterator. The MapIterObject does not
 * need to have any information set for this function to work.
 * (PyArray_MapIterSwapAxes requires also nd and nd_fancy info)
 *
 * Sets the following information:
 *    * mit->consec: The axis where the fancy indices need transposing to.
 *    * mit->iteraxes: The axis which the fancy index corresponds to.
 *    * mit->fancy_dims: the dimension of `arr` along the indexed dimension
 *          for each fancy index.
 *    * mit->fancy_strides: the strides for the dimension being indexed
 *          by each fancy index.
 *    * mit->dimensions: Broadcast dimension of the fancy indices and
 *          the subspace iteration dimension.
 *
 * @param MapIterObject The iterator object to fill with information.
 * @param indices The parsed indices object containing index information.
 * @param index_num Number of indices.
 * @param arr The array that is being iterated.
 *
 * @return 0 on success, -1 on failure (broadcasting or too many fancy indices).
 */
static int
mapiter_fill_info(PyArrayMapIterObject *mit, npy_index_info *indices,
                  int index_num, PyArrayObject *arr)
{
    int j = 0, i;
    int curr_dim = 0;
    /* dimension of index result (up to first fancy index) */
    int result_dim = 0;
    /* -1 init; 0 found fancy; 1 fancy stopped; 2 found not consecutive fancy */
    int consec_status = -1;
    int axis, broadcast_axis;
    npy_intp dimension;

    // Initialize dimensions of fancy indices to 1
    for (i = 0; i < mit->nd_fancy; i++) {
        mit->dimensions[i] = 1;
    }

    // Set consec to 0
    mit->consec = 0;

    // Fill dimension of subspace if it exists
    if (mit->subspace) {
        for (i = 0; i < PyArray_NDIM(mit->subspace); i++) {
            mit->dimensions[mit->nd_fancy + i] = PyArray_DIM(mit->subspace, i);
        }
    }

    // Successful completion
    return 0;

broadcast_error: ;  // Declarations cannot follow labels, add empty statement.
    /*
     * Attempt to set a meaningful exception. Could also find out
     * if a boolean index was converted.
     */
    
    // Initialize error message
    PyObject *errmsg = PyUnicode_FromString("");
    if (errmsg == NULL) {
        return -1;
    }

    // Iterate over indices to build error message
    for (i = 0; i < index_num; i++) {
        // Skip non-fancy indices
        if (!(indices[i].type & HAS_FANCY)) {
            continue;
        }

        // Get dimensions and shape of the index object
        int ndim = PyArray_NDIM((PyArrayObject *)indices[i].object);
        npy_intp *shape = PyArray_SHAPE((PyArrayObject *)indices[i].object);
        // Convert shape information to string
        PyObject *tmp = convert_shape_to_string(ndim, shape, " ");
        if (tmp == NULL) {
            Py_DECREF(errmsg);
            return -1;
        }

        // Concatenate shape information to error message
        Py_SETREF(errmsg, PyUnicode_Concat(errmsg, tmp));
        Py_DECREF(tmp);
        if (errmsg == NULL) {
            return -1;
        }
    }

    // Format and set PyErr exception for broadcasting error
    PyErr_Format(PyExc_IndexError,
            "shape mismatch: indexing arrays could not "
            "be broadcast together with shapes %S", errmsg);
    Py_DECREF(errmsg);
    return -1;
}
# 检查索引是否符合要求的函数，针对PyArrayMapIterObject结构体的实例mit
def PyArray_MapIterCheckIndices(PyArrayMapIterObject *mit):
    # op是PyArrayObject类型的指针，用于存储当前操作的数组对象
    PyArrayObject *op;
    # op_iter是NpyIter类型的指针，用于存储操作数组的迭代器对象
    NpyIter *op_iter;
    # op_iternext是NpyIter_IterNextFunc类型的指针，用于存储操作数组迭代器的迭代函数
    NpyIter_IterNextFunc *op_iternext;
    # outer_dim是npy_intp类型的变量，表示外层维度的大小
    npy_intp outer_dim, indval;
    # outer_axis是整型变量，表示外层轴的索引
    int outer_axis;
    # itersize是npy_intp类型的变量，表示迭代器的大小
    npy_intp itersize, *iterstride;
    # iterptr是字符指针的数组，用于存储迭代器的指针
    char **iterptr;
    # intp_type是PyArray_Descr类型的指针，用于存储表示整型的数组描述符对象
    PyArray_Descr *intp_type;
    # i是整型变量，用于迭代
    int i;
    # NPY_BEGIN_THREADS_DEF宏定义，用于开启线程（Python多线程环境下使用）

    if (NpyIter_GetIterSize(mit->outer) == 0) {
        """
         * 当外层迭代为空时，索引广播到一个空形状，此时我们不检查是否存在越界索引。
         * 下面的代码在不进行广播的情况下使用索引，因为广播只是重复值。
         """
        # 返回0，表示没有越界索引的检查
        return 0;
    }

    # intp_type为NPY_INTP类型的PyArray_Descr对象，表示整型数组描述符
    intp_type = PyArray_DescrFromType(NPY_INTP);

    # 开启线程环境
    NPY_BEGIN_THREADS;
    // 遍历mit结构体中的每个fancy索引操作
    for (i=0; i < mit->num_fancy; i++) {
        // 获取当前操作数
        op = NpyIter_GetOperandArray(mit->outer)[i];

        // 获取外部维度和迭代轴
        outer_dim = mit->fancy_dims[i];
        outer_axis = mit->iteraxes[i];

        /* 查看是否可以简单迭代数组 */
        if (PyArray_TRIVIALLY_ITERABLE(op) &&
                /* 检查类型是否等同于INTP */
                PyArray_ITEMSIZE(op) == sizeof(npy_intp) &&
                PyArray_DESCR(op)->kind == 'i' &&
                IsUintAligned(op) &&
                PyDataType_ISNOTSWAPPED(PyArray_DESCR(op))) {
            char *data;
            npy_intp stride;
            /* 如果GIL被下面的nditer占用，则释放它 */
            if (_save == NULL) {
                NPY_BEGIN_THREADS;
            }

            // 准备简单迭代操作
            PyArray_PREPARE_TRIVIAL_ITERATION(op, itersize, data, stride);

            // 迭代操作数据
            while (itersize--) {
                indval = *((npy_intp*)data);
                // 检查并调整索引值
                if (check_and_adjust_index(&indval,
                                           outer_dim, outer_axis, _save) < 0) {
                    Py_DECREF(intp_type);
                    // 跳转到索引错误处理
                    goto indexing_error;
                }
                data += stride;
            }
            /* 在函数结束或需要nditer路径时重新获取GIL */
            continue;
        }

        /* 如果无法简单迭代，则使用NpyIter */
        NPY_END_THREADS;
        // 创建NpyIter对象
        op_iter = NpyIter_New(op,
                        NPY_ITER_BUFFERED | NPY_ITER_NBO | NPY_ITER_ALIGNED |
                        NPY_ITER_EXTERNAL_LOOP | NPY_ITER_GROWINNER |
                        NPY_ITER_READONLY | NPY_ITER_ZEROSIZE_OK,
                        NPY_KEEPORDER, NPY_SAME_KIND_CASTING, intp_type);

        // 检查NpyIter对象是否创建成功
        if (op_iter == NULL) {
            Py_DECREF(intp_type);
            return -1;
        }
        // 如果迭代大小为0，则释放NpyIter对象并继续下一次循环
        if (NpyIter_GetIterSize(op_iter) == 0) {
            NpyIter_Deallocate(op_iter);
            continue;
        }

        // 获取NpyIter对象的迭代器和数据指针
        op_iternext = NpyIter_GetIterNext(op_iter, NULL);
        if (op_iternext == NULL) {
            Py_DECREF(intp_type);
            NpyIter_Deallocate(op_iter);
            return -1;
        }

        // 开始多线程迭代操作
        NPY_BEGIN_THREADS_NDITER(op_iter);
        iterptr = NpyIter_GetDataPtrArray(op_iter);
        iterstride = NpyIter_GetInnerStrideArray(op_iter);
        do {
            // 获取内部循环的大小
            itersize = *NpyIter_GetInnerLoopSizePtr(op_iter);
            while (itersize--) {
                indval = *((npy_intp*)*iterptr);
                // 检查并调整索引值
                if (check_and_adjust_index(&indval,
                                           outer_dim, outer_axis, _save) < 0) {
                    Py_DECREF(intp_type);
                    NpyIter_Deallocate(op_iter);
                    // 跳转到索引错误处理
                    goto indexing_error;
                }
                *iterptr += *iterstride;
            }
        } while (op_iternext(op_iter));

        // 结束多线程迭代
        NPY_END_THREADS;
        // 释放NpyIter对象
        NpyIter_Deallocate(op_iter);
    }

    // 结束多线程迭代
    NPY_END_THREADS;
    // 释放intp_type类型对象的引用计数
    Py_DECREF(intp_type);
    // 返回成功状态
    return 0;
indexing_error:

    if (mit->size == 0) {
        // 检查迭代器的大小是否为0，即迭代结果中没有元素
        PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
        // 检索最近的 Python 异常
        PyErr_Fetch(&err_type, &err_value, &err_traceback);
        /* 2020-05-27, NumPy 1.20 */
        // 发出弃用警告，说明索引越界将不再被忽略，而是引发异常
        if (DEPRECATE(
                "Out of bound index found. This was previously ignored "
                "when the indexing result contained no elements. "
                "In the future the index error will be raised. This error "
                "occurs either due to an empty slice, or if an array has zero "
                "elements even before indexing.\n"
                "(Use `warnings.simplefilter('error')` to turn this "
                "DeprecationWarning into an error and get more details on "
                "the invalid index.)") < 0) {
            // 将之前检索到的异常链入当前 Python 异常
            npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
            return -1;
        }
        // 释放异常对象的引用
        Py_DECREF(err_type);
        Py_DECREF(err_value);
        Py_XDECREF(err_traceback);
        return 0;
    }

    return -1;
}


/*
 * Create new mapiter.
 *
 * NOTE: The outer iteration (and subspace if requested buffered) is
 *       created with DELAY_BUFALLOC. It must be reset before usage!
 *
 * @param Index information filled by prepare_index.
 *        由 prepare_index 填充的索引信息
 * @param Number of indices (gotten through prepare_index).
 *        索引的数量（通过 prepare_index 获得）
 * @param Kind of index (gotten through preprare_index).
 *        索引的类型（通过 prepare_index 获得）
 * @param NpyIter flags for an extra array. If 0 assume that there is no
 *        extra operand. NPY_ITER_ALLOCATE can make sense here.
 *        用于额外数组的 NpyIter 标志。如果为 0，则假设没有额外操作数。
 *        NPY_ITER_ALLOCATE 在这里可能有意义。
 * @param Array being indexed
 *        被索引的数组
 * @param subspace (result of getting view for the indices)
 *        subspace（通过索引获取视图的结果）
 * @param Subspace iterator flags can be used to enable buffering.
 *        NOTE: When no subspace is necessary, the extra operand will
 *              always be buffered! Buffering the subspace when not
 *              necessary is very slow when the subspace is small.
 *        子空间迭代器标志可用于启用缓冲。
 *        注意：当不需要子空间时，额外操作数始终会被缓冲！
 *              当子空间很小时，缓冲子空间不必要地慢。
 * @param Subspace operand flags (should just be 0 normally)
 *        子空间操作数标志（通常应为0）
 * @param Operand iteration flags for the extra operand, this must not be
 *        0 if an extra operand should be used, otherwise it must be 0.
 *        Should be at least READONLY, WRITEONLY or READWRITE.
 *        用于额外操作数的操作数迭代标志，如果应使用额外操作数，则此值不能为0，否则必须为0。
 *        应至少为 READONLY、WRITEONLY 或 READWRITE。
 * @param Extra operand. For getmap, this would be the result, for setmap
 *        this would be the arrays to get from.
 *        Can be NULL, and will be allocated in that case. However,
 *        it matches the mapiter iteration, so you have to call
 *        MapIterSwapAxes(mit, &extra_op, 1) on it.
 *        The operand has no effect on the shape.
 *        额外的操作数。对于 getmap，这将是结果，对于 setmap，这将是要获取的数组。
 *        可以为 NULL，在这种情况下将会分配内存。但是，它匹配 mapiter 迭代，因此您必须在其上调用 MapIterSwapAxes(mit, &extra_op, 1)。
 *        操作数对形状没有影响。
 * @param Dtype for the extra operand, borrows the reference and must not
 *        be NULL (if extra_op_flags is not 0).
 *        用于额外操作数的 Dtype，借用引用并且不能为 NULL（如果 extra_op_flags 不为 0）。
 *
 * @return A new MapIter (PyObject *) or NULL.
 *         返回新的 MapIter（PyObject *）或 NULL。
 */
NPY_NO_EXPORT PyObject *
    /* 用于在错误报告中报告形状 */
    PyArrayObject *original_extra_op = extra_op;

    /* 注意：MAXARGS 是实际限制（2*NPY_MAXDIMS 是第一个索引号） */
    PyArrayObject *index_arrays[NPY_MAXDIMS];
    PyArray_Descr *intp_descr;
    PyArray_Descr *dtypes[NPY_MAXDIMS];  /* 借用的引用 */

    npy_uint32 op_flags[NPY_MAXDIMS];
    npy_uint32 outer_flags;

    PyArrayMapIterObject *mit;

    int single_op_axis[NPY_MAXDIMS];
    int *op_axes[NPY_MAXDIMS] = {NULL};
    int i, j, dummy_array = 0;
    int nops;
    int uses_subspace;

    /* 从 NPY_INTP 类型创建描述符 */
    intp_descr = PyArray_DescrFromType(NPY_INTP);
    if (intp_descr == NULL) {
        return NULL;
    }

    /* 创建新的 MapIter 对象 */
    mit = (PyArrayMapIterObject *)PyArray_malloc(
            sizeof(PyArrayMapIterObject) + sizeof(NPY_cast_info));
    if (mit == NULL) {
        Py_DECREF(intp_descr);
        return NULL;
    }
    /* 将 mapiter 的所有属性设置为零 */
    memset(mit, 0, sizeof(PyArrayMapIterObject) + sizeof(NPY_cast_info));
    PyObject_Init((PyObject *)mit, &PyArrayMapIter_Type);

    Py_INCREF(arr);
    mit->array = arr;
    Py_XINCREF(subspace);
    mit->subspace = subspace;

    /*
     * subspace 是数组未索引部分，在 subspace 大小大于 1 时需要迭代。
     * 如果大小为 1，则仅影响结果形状。（优化例如 np.newaxis 的用法）
     */
    if ((subspace == NULL) || PyArray_SIZE(subspace) == 1) {
        uses_subspace = 0;
    }
    else {
        uses_subspace = 1;
    }

    /* 填充 mapiter 的基本信息 */
    mit->nd = ndim;
    mit->nd_fancy = fancy_ndim;
    if (mapiter_fill_info(mit, indices, index_num, arr) < 0) {
        Py_DECREF(mit);
        Py_DECREF(intp_descr);
        return NULL;
    }

    /*
     * 设置索引数组的迭代信息。
     */
    for (i=0; i < index_num; i++) {
        if (indices[i].type & HAS_FANCY) {
            index_arrays[mit->num_fancy] = (PyArrayObject *)indices[i].object;
            dtypes[mit->num_fancy] = intp_descr;

            op_flags[mit->num_fancy] = (NPY_ITER_NBO |
                                      NPY_ITER_ALIGNED |
                                      NPY_ITER_READONLY);
            mit->num_fancy += 1;
        }
    }
    if (mit->num_fancy == 0) {
        /*
         * 对于 MapIterArray，可能没有使用 fancy 索引。
         * 为了支持这种情况，添加一个虚拟迭代器。
         * 由于它是零维的，其转置等操作并不重要。
         */

        /* 信号需要减少引用计数... */
        dummy_array = 1;

        // 创建一个包含零个元素的整数类型数组作为索引数组
        index_arrays[0] = (PyArrayObject *)PyArray_Zeros(0, NULL,
                                        PyArray_DescrFromType(NPY_INTP), 0);
        // 如果创建失败，释放资源并返回空指针
        if (index_arrays[0] == NULL) {
            Py_DECREF(mit);
            Py_DECREF(intp_descr);
            return NULL;
        }
        // 设置数据类型为整数类型描述符
        dtypes[0] = intp_descr;
        // 设置操作标志为字节序、对齐和只读
        op_flags[0] = NPY_ITER_NBO | NPY_ITER_ALIGNED | NPY_ITER_READONLY;

        // 设置 fancy_dims 的第一个维度为 1，并标记有 fancy 索引存在
        mit->fancy_dims[0] = 1;
        mit->num_fancy = 1;
    }

    /*
     * 现在有两种一般情况下额外操作（extra_op）的使用方式：
     *   1. 不需要子空间迭代，因此额外操作可以包含在索引迭代器中（会被缓冲）。
     *   2. 需要子空间迭代，因此额外操作会独立迭代，并且迭代顺序固定为 C 顺序
     *      （如果数组是 Fortran 顺序，也可以使用 Fortran 顺序）。
     *      在这种情况下，子空间迭代器不会被缓冲。
     *
     * 如果需要子空间迭代，并且给定了额外操作（extra_op），可能还需要对额外操作进行转置
     * （或者通知高级迭代器进行转置）。
     */
    if (extra_op != NULL) {
        /*
         * 如果存在额外操作数，需要对其进行准备。
         *   1. 子类可能会影响形状，因此需要一个基类。
         *   2. 需要确保形状是兼容的。
         *   3. 可能需要移除前导的1并转置维度。
         *      普通的赋值操作允许广播去除前导的1，但是转置代码不允许这样做。
         */
        
        if (!PyArray_CheckExact(extra_op)) {
            // 如果额外操作数不是精确的PyArray对象，将其视图化为PyArray对象
            extra_op = (PyArrayObject *)PyArray_View(extra_op, NULL,
                                                     &PyArray_Type);
            if (extra_op == NULL) {
                goto fail;
            }
        }
        else {
            // 如果额外操作数已经是精确的PyArray对象，增加其引用计数
            Py_INCREF(extra_op);
        }

        if (PyArray_NDIM(extra_op) > mit->nd) {
            /*
             * 普通赋值操作允许移除前导的单维度（或等效地添加单维度到被赋值的数组）。
             * 为了实现这一点，重新整形数组。
             */
            PyArrayObject *tmp_arr;
            PyArray_Dims permute;

            permute.len = mit->nd;
            permute.ptr = &PyArray_DIMS(extra_op)[
                                            PyArray_NDIM(extra_op) - mit->nd];
            tmp_arr = (PyArrayObject*)PyArray_Newshape(extra_op, &permute,
                                                       NPY_CORDER);
            if (tmp_arr == NULL) {
                goto broadcast_error;
            }
            Py_DECREF(extra_op);
            extra_op = tmp_arr;
        }

        /*
         * 如果需要前置维度（并且不需要swapaxis），在确保分配额外操作数后使用op_axes。
         */
        if (mit->consec) {
            PyArray_MapIterSwapAxes(mit, &extra_op, 0);
            if (extra_op == NULL) {
                goto fail;
            }
        }

        if (subspace && !uses_subspace) {
            /*
             * 我们没有使用子空间，因此其大小为1。
             * 额外操作数对应于子空间的所有维度必须等于1。
             */
            if (PyArray_NDIM(subspace) <= PyArray_NDIM(extra_op)) {
                j = PyArray_NDIM(subspace);
            }
            else {
                j = PyArray_NDIM(extra_op);
            }
            for (i = 1; i < j + 1; i++) {
                if (PyArray_DIM(extra_op, PyArray_NDIM(extra_op) - i) != 1) {
                    goto broadcast_error;
                }
            }
        }
    }

    /*
     * 如果子空间不为NULL，NpyIter不能为我们分配额外操作数。
     * 这有点笨拙。创建一个虚拟迭代器来找到正确的输出形状和步幅排列。
     * TODO: 这可以至少部分地被替换，因为形状已经找到以处理广播错误。
     */
        else if (extra_op_flags && (subspace != NULL)) {
            # 如果额外操作标志位非零且子空间非空

            npy_uint32 tmp_op_flags[NPY_MAXDIMS];
            # 声明一个用于临时操作标志的数组，长度为最大维度数

            NpyIter *tmp_iter;
            # 声明一个 NpyIter 指针变量

            npy_intp stride;
            npy_intp strides[NPY_MAXDIMS];
            # 声明一个步幅和一个步幅数组，长度均为最大维度数

            npy_stride_sort_item strideperm[NPY_MAXDIMS];
            # 声明一个用于步幅排序的结构体数组，长度为最大维度数

            for (i=0; i < mit->num_fancy; i++) {
                tmp_op_flags[i] = NPY_ITER_READONLY;
                # 初始化临时操作标志数组，所有元素设置为 NPY_ITER_READONLY
            }

            if (PyArray_SIZE(subspace) == 1) {
                /* Create an iterator, just to broadcast the arrays?! */
                # 如果子空间大小为1，创建一个迭代器，用于广播数组？！

                tmp_iter = NpyIter_MultiNew(mit->num_fancy, index_arrays,
                                            NPY_ITER_ZEROSIZE_OK |
                                            NPY_ITER_REFS_OK |
                                            NPY_ITER_MULTI_INDEX |
                                            NPY_ITER_DONT_NEGATE_STRIDES,
                                            NPY_KEEPORDER,
                                            NPY_UNSAFE_CASTING,
                                            tmp_op_flags, NULL);
                # 使用 NpyIter_MultiNew 函数创建一个多迭代器
                # 设置迭代器的选项包括 NPY_ITER_ZEROSIZE_OK，NPY_ITER_REFS_OK，
                # NPY_ITER_MULTI_INDEX 和 NPY_ITER_DONT_NEGATE_STRIDES
                # 还传入了临时操作标志数组和空指针作为额外参数

                if (tmp_iter == NULL) {
                    goto fail;
                    # 如果迭代器创建失败，则跳转到 fail 标签处处理错误
                }

                /*
                 * nditer allows itemsize with npy_intp type, so it works
                 * here, but it would *not* work directly, since elsize
                 * is limited to int.
                 */
                # nditer 允许使用 npy_intp 类型的 itemsize，所以这里可以工作
                # 但直接使用时不会工作，因为 elsize 限制为 int 类型

                if (!NpyIter_CreateCompatibleStrides(tmp_iter,
                            extra_op_dtype->elsize * PyArray_SIZE(subspace),
                            strides)) {
                    PyErr_SetString(PyExc_ValueError,
                            "internal error: failed to find output array strides");
                    goto fail;
                    # 如果无法创建兼容的步幅，则设置异常并跳转到 fail 标签处处理错误
                }
                NpyIter_Deallocate(tmp_iter);
                # 释放迭代器资源
            }
            else {
                /* Just use C-order strides (TODO: allow also F-order) */
                # 否则，使用 C 阶序的步幅（TODO：也允许 F 阶序）

                stride = extra_op_dtype->elsize * PyArray_SIZE(subspace);
                # 计算步幅为额外操作数据类型的元素大小乘以子空间大小

                for (i=mit->nd_fancy - 1; i >= 0; i--) {
                    strides[i] = stride;
                    stride *= mit->dimensions[i];
                    # 根据迭代器的高级维度计算步幅数组
                }
            }

            /* shape is set, and strides is set up to mit->nd, set rest */
            # 设置形状和步幅到 mit->nd，设置剩余部分

            PyArray_CreateSortedStridePerm(PyArray_NDIM(subspace),
                                    PyArray_STRIDES(subspace), strideperm);
            # 使用子空间的维度和步幅，以及步幅排序结构体数组，创建排好序的步幅排列

            stride = extra_op_dtype->elsize;
            # 步幅设置为额外操作数据类型的元素大小

            for (i=PyArray_NDIM(subspace) - 1; i >= 0; i--) {
                strides[mit->nd_fancy + strideperm[i].perm] = stride;
                stride *= PyArray_DIM(subspace, (int)strideperm[i].perm);
                # 根据排序后的步幅排列，计算最终的步幅数组
            }

            /*
             * Allocate new array. Note: Always base class, because
             * subclasses might mess with the shape.
             */
            # 分配新的数组。注意：始终使用基类，因为子类可能会改变形状。

            Py_INCREF(extra_op_dtype);
            # 增加额外操作数据类型的引用计数

            extra_op = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                               extra_op_dtype,
                                               mit->nd_fancy + PyArray_NDIM(subspace),
                                               mit->dimensions, strides,
                                               NULL, 0, NULL);
            # 使用 PyArray_NewFromDescr 函数创建一个新的数组对象
            # 设置对象类型为 PyArray_Type，传入额外操作数据类型、维度数、形状、步幅等参数

            if (extra_op == NULL) {
                goto fail;
                # 如果创建数组对象失败，则跳转到 fail 标签处处理错误
            }
        }
    /*
     * 如果额外操作 extra_op 存在，则要么已分配，可以由 NpyIter 分配（无子空间），或者根本未使用。
     * 需要为 extra_op 设置轴重映射。这将导致忽略子空间维度，并在广播时添加 -1。
     */
    if (extra_op) {
        for (j=0; j < mit->nd - PyArray_NDIM(extra_op); j++) {
            single_op_axis[j] = -1;  // 将 -1 添加到轴映射中，以忽略子空间维度
        }
        for (i=0; i < PyArray_NDIM(extra_op); i++) {
            /* （填充子空间维度，但它们不是未使用的）*/
            single_op_axis[j++] = i;  // 将额外操作的轴映射添加到数组末尾
        }
    }

    /*
     * 注意：如果出于某种原因有人希望使用 REDUCE_OK，请小心并修复末尾的错误消息替换。
     */
    outer_flags = NPY_ITER_ZEROSIZE_OK |
                  NPY_ITER_REFS_OK |
                  NPY_ITER_BUFFERED |
                  NPY_ITER_DELAY_BUFALLOC |
                  NPY_ITER_GROWINNER;

    /*
     * 对于单个一维操作数，保证迭代顺序（scipy 使用这个）。注意子空间可能被使用。
     */
    if ((mit->num_fancy == 1) && (PyArray_NDIM(index_arrays[0]) == 1)) {
        outer_flags |= NPY_ITER_DONT_NEGATE_STRIDES;  // 设置不反转步长的标志
    }

    /* 如果外部数组被迭代，并且不需要子空间 */
    nops = mit->num_fancy;  // 设置操作数的数量

    if (!uses_subspace) {
        outer_flags |= NPY_ITER_EXTERNAL_LOOP;  // 设置外部循环标志
    }

    if (extra_op_flags && !uses_subspace) {
        /*
         * 注意：这个小限制实际上应该没有关系。
         *       （替换 npyiter 错误）
         */
        if (mit->num_fancy > NPY_MAXDIMS - 1) {
            PyErr_Format(PyExc_IndexError,
                         "when no subspace is given, the number of index "
                         "arrays cannot be above %d, but %d index arrays found",
                         NPY_MAXDIMS - 1, mit->num_fancy);
            goto fail;  // 如果索引数组数量超过限制，触发错误并跳转到失败处理
        }

        nops += 1;  // 增加操作数的数量
        index_arrays[mit->num_fancy] = extra_op;  // 设置额外操作的索引数组

        dtypes[mit->num_fancy] = extra_op_dtype;  // 设置额外操作的数据类型
        op_flags[mit->num_fancy] = (extra_op_flags |
                                  NPY_ITER_ALLOCATE |
                                  NPY_ITER_NO_SUBTYPE);  // 设置额外操作的标志

        if (extra_op) {
            /* 使用轴重映射 */
            op_axes[mit->num_fancy] = single_op_axis;  // 设置额外操作的轴映射
            mit->outer = NpyIter_AdvancedNew(nops, index_arrays, outer_flags,
                             NPY_KEEPORDER, NPY_UNSAFE_CASTING, op_flags, dtypes,
                             mit->nd_fancy, op_axes, mit->dimensions, 0);  // 创建高级迭代器
        }
        else {
            mit->outer = NpyIter_MultiNew(nops, index_arrays, outer_flags,
                             NPY_KEEPORDER, NPY_UNSAFE_CASTING, op_flags, dtypes);  // 创建多迭代器
        }

    }
    else {
        /* TODO: 可能添加 CORDER 的测试，并且也许允许 F */
        mit->outer = NpyIter_MultiNew(nops, index_arrays, outer_flags,
                         NPY_CORDER, NPY_UNSAFE_CASTING, op_flags, dtypes);  // 创建多迭代器
    }
    /* NpyIter cleanup and information: */
    // 如果 dummy_array 存在，则减少 index_arrays[0] 的引用计数
    if (dummy_array) {
        Py_DECREF(index_arrays[0]);
    }
    // 如果 mit->outer 为 NULL，则跳转到失败标签
    if (mit->outer == NULL) {
        goto fail;
    }

    // 获取 mit->outer 的下一个迭代器函数
    mit->outer_next = NpyIter_GetIterNext(mit->outer, NULL);
    // 如果获取失败，则跳转到失败标签
    if (mit->outer_next == NULL) {
        goto fail;
    }
    // 获取 mit->outer 的数据指针数组
    mit->outer_ptrs = NpyIter_GetDataPtrArray(mit->outer);
    // 如果不使用子空间，则获取 mit->outer 的内部步幅数组
    if (!uses_subspace) {
        mit->outer_strides = NpyIter_GetInnerStrideArray(mit->outer);
    }

    /* Get the allocated extra_op */
    // 如果 extra_op_flags 存在
    if (extra_op_flags) {
        // 如果 extra_op 为 NULL，则将 mit->extra_op 设置为 mit->outer 的操作数数组中的第 mit->num_fancy 个操作数
        if (extra_op == NULL) {
            mit->extra_op = NpyIter_GetOperandArray(mit->outer)[mit->num_fancy];
        }
        else {
            // 否则，将 mit->extra_op 设置为 extra_op
            mit->extra_op = extra_op;
        }
        // 增加 mit->extra_op 的引用计数
        Py_INCREF(mit->extra_op);
    }

    /*
     * If extra_op is being tracked but subspace is used, we need
     * to create a dedicated iterator for the outer iteration of
     * the extra operand.
     */
    // 如果 extra_op_flags 存在并且使用子空间
    if (extra_op_flags && uses_subspace) {
        // 设置操作轴数组中的第一个元素为 single_op_axis
        op_axes[0] = single_op_axis;
        // 创建 mit->extra_op_iter 作为 mit->extra_op 的高级新迭代器
        mit->extra_op_iter = NpyIter_AdvancedNew(1, &extra_op,
                                                 NPY_ITER_ZEROSIZE_OK |
                                                 NPY_ITER_REFS_OK |
                                                 NPY_ITER_GROWINNER,
                                                 NPY_CORDER,
                                                 NPY_NO_CASTING,
                                                 &extra_op_flags,
                                                 NULL,
                                                 mit->nd_fancy, op_axes,
                                                 mit->dimensions, 0);

        // 如果创建失败，则跳转到失败标签
        if (mit->extra_op_iter == NULL) {
            goto fail;
        }

        // 获取 mit->extra_op_iter 的下一个迭代器函数
        mit->extra_op_next = NpyIter_GetIterNext(mit->extra_op_iter, NULL);
        // 如果获取失败，则跳转到失败标签
        if (mit->extra_op_next == NULL) {
            goto fail;
        }
        // 获取 mit->extra_op_iter 的数据指针数组
        mit->extra_op_ptrs = NpyIter_GetDataPtrArray(mit->extra_op_iter);
    }

    /* Get the full dimension information */
    // 如果 subspace 不为 NULL，则将 mit->baseoffset 设置为 subspace 的字节偏移量
    if (subspace != NULL) {
        mit->baseoffset = PyArray_BYTES(subspace);
    }
    else {
        // 否则，将 mit->baseoffset 设置为 arr 的字节偏移量
        mit->baseoffset = PyArray_BYTES(arr);
    }

    /* Calculate total size of the MapIter */
    // 计算 MapIter 的总大小
    mit->size = PyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    // 如果计算结果为负数，则设置异常并跳转到失败标签
    if (mit->size < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "advanced indexing operation result is too large");
        goto fail;
    }

    /* Can now return early if no subspace is being used */
    // 如果不使用子空间，则减少 extra_op 和 intp_descr 的引用计数，并返回 mit 指针
    if (!uses_subspace) {
        Py_XDECREF(extra_op);
        Py_DECREF(intp_descr);
        return (PyObject *)mit;
    }

    /* Fill in the last bit of mapiter information needed */

    /*
     * Now just need to create the correct subspace iterator.
     */
    // 将 index_arrays[0] 设置为 subspace，并初始化其他相关变量
    index_arrays[0] = subspace;
    dtypes[0] = NULL;
    op_flags[0] = subspace_flags;
    op_axes[0] = NULL;
    if (extra_op_flags) {
        /* 如果存在额外操作标志，表示需要迭代额外操作 */
        nops = 2;
        index_arrays[1] = extra_op;  // 将额外操作数组加入索引数组中

        op_axes[1] = &single_op_axis[mit->nd_fancy];  // 设置额外操作的轴

        /*
         * Buffering is never used here, but in case someone plugs it in
         * somewhere else, set the type correctly then.
         */
        // 如果启用了缓冲，设置额外操作的数据类型
        if ((subspace_iter_flags & NPY_ITER_BUFFERED)) {
            dtypes[1] = extra_op_dtype;
        }
        else {
            dtypes[1] = NULL;  // 否则将数据类型设为 NULL
        }
        op_flags[1] = extra_op_flags;  // 设置额外操作的操作标志
    }
    else {
        nops = 1;  // 如果不存在额外操作，设置操作数为 1
    }

    // 创建高级迭代器对象，用于处理索引操作
    mit->subspace_iter = NpyIter_AdvancedNew(nops, index_arrays,
                                    NPY_ITER_ZEROSIZE_OK |
                                    NPY_ITER_REFS_OK |
                                    NPY_ITER_GROWINNER |
                                    NPY_ITER_EXTERNAL_LOOP |
                                    NPY_ITER_DELAY_BUFALLOC |
                                    subspace_iter_flags,
                                    (nops == 1 ? NPY_CORDER : NPY_KEEPORDER),
                                    NPY_UNSAFE_CASTING,
                                    op_flags, dtypes,
                                    PyArray_NDIM(subspace), op_axes,
                                    &mit->dimensions[mit->nd_fancy], 0);

    // 检查迭代器对象是否创建成功
    if (mit->subspace_iter == NULL) {
        goto fail;  // 如果创建失败，跳转到失败处理
    }

    // 获取迭代器的下一个迭代函数
    mit->subspace_next = NpyIter_GetIterNext(mit->subspace_iter, NULL);
    // 检查获取下一个迭代函数是否成功
    if (mit->subspace_next == NULL) {
        goto fail;  // 如果获取失败，跳转到失败处理
    }
    // 获取迭代器的数据指针数组
    mit->subspace_ptrs = NpyIter_GetDataPtrArray(mit->subspace_iter);
    // 获取迭代器的内部步长数组
    mit->subspace_strides = NpyIter_GetInnerStrideArray(mit->subspace_iter);

    // 减少额外操作的引用计数
    Py_XDECREF(extra_op);
    // 减少 intp 描述符的引用计数
    Py_DECREF(intp_descr);
    // 返回迭代器对象指针，表示成功
    return (PyObject *)mit;

  fail:
    /*
     * Check whether the operand could not be broadcast and replace the error
     * in that case. This should however normally be found early with a
     * direct goto to broadcast_error
     */
    // 检查是否无法对操作数进行广播，并在这种情况下替换错误
    if (extra_op == NULL) {
        goto finish;  // 如果额外操作为 NULL，跳转到结束处理
    }

    // 逆序检查额外操作的维度是否可以广播到索引结果
    j = mit->nd;
    for (i = PyArray_NDIM(extra_op) - 1; i >= 0; i--) {
        j--;
        if ((PyArray_DIM(extra_op, i) != 1) &&
                /* (j < 0 is currently impossible, extra_op is reshaped) */
                j >= 0 &&
                PyArray_DIM(extra_op, i) != mit->dimensions[j]) {
            /* extra_op cannot be broadcast to the indexing result */
            goto broadcast_error;  // 如果无法广播，跳转到广播错误处理
        }
    }
    goto finish;  // 检查完成，跳转到结束处理

  broadcast_error:
    /* Report the shape of the original array if it exists */
    // 如果存在原始数组，报告其形状
    if (original_extra_op == NULL) {
        original_extra_op = extra_op;  // 如果原始额外操作为 NULL，设置为额外操作
    }

    // 获取原始额外操作的维度信息
    int extra_ndim = PyArray_NDIM(original_extra_op);
    npy_intp *extra_dims = PyArray_DIMS(original_extra_op);
    // 将维度转换为字符串形式
    PyObject *shape1 = convert_shape_to_string(extra_ndim, extra_dims, "");
    // 如果转换失败，跳转到结束处理
    if (shape1 == NULL) {
        goto finish;
    }

    // 在使用 `mit->consec` 时，反转迭代器形状以进行报告
    npy_intp transposed[NPY_MAXDIMS];
    // 使用自定义函数获取转置操作后的索引数组，此处mit->nd_fancy可能是用于描述高级索引的数据结构
    _get_transpose(mit->nd_fancy, mit->consec, mit->nd, 1, transposed);

    // 根据转置后的索引数组，获取相应维度的大小，存储在transposed数组中
    for (i = 0; i < mit->nd; i++) {
        transposed[i] = mit->dimensions[transposed[i]];
    }

    // 将转置后的维度信息转换为字符串表示形式，用于错误信息中的格式化输出
    PyObject *shape2 = convert_shape_to_string(mit->nd, transposed, "");
    if (shape2 == NULL) {
        // 如果转换失败，释放shape1并跳转到finish标签处
        Py_DECREF(shape1);
        goto finish;
    }

    // 格式化错误信息，指示形状不匹配的问题，包括value array的形状和索引结果的形状
    PyErr_Format(PyExc_ValueError,
            "shape mismatch: value array of shape %S could not be broadcast "
            "to indexing result of shape %S", shape1, shape2);

    // 释放shape1和shape2对象
    Py_DECREF(shape1);
    Py_DECREF(shape2);

  finish:
    // 释放额外操作对象、整数描述对象和mit对象
    Py_XDECREF(extra_op);
    Py_DECREF(intp_descr);
    Py_DECREF(mit);
    // 返回NULL表示函数执行失败
    return NULL;
# 关闭函数的定义
}

# 利用高级索引来迭代数组。
# 如果 copy_if_overlap != 0，则检查 `a` 是否与 `index` 中的任何数组以及 `extra_op` 存在内存重叠。如果是，适当进行复制以避免在迭代期间修改 `a` 时出现问题。
# `iter->array` 可能包含已复制的数组（设置了 WRITEBACKIFCOPY）。
NPY_NO_EXPORT PyObject *
PyArray_MapIterArrayCopyIfOverlap(PyArrayObject * a, PyObject * index,
                                  int copy_if_overlap, PyArrayObject *extra_op)
{
    PyArrayMapIterObject * mit = NULL;  # 初始化 PyArrayMapIterObject 指针
    PyArrayObject *subspace = NULL;  # 初始化 PyArrayObject 指针
    npy_index_info indices[NPY_MAXDIMS * 2 + 1];  # 创建索引信息数组
    int i, index_num, ndim, fancy_ndim, index_type;  # 初始化整型变量
    PyArrayObject *a_copy = NULL;  # 初始化 PyArrayObject 指针

    index_type = prepare_index(a, index, indices, &index_num,  # 调用 prepare_index 函数准备索引
                               &ndim, &fancy_ndim, 0);

    if (index_type < 0) {  # 如果索引类型小于0，则返回NULL
        return NULL;
    }

    if (copy_if_overlap && index_has_memory_overlap(a, index_type, indices,  # 如果需要复制且存在内存重叠
                                                    index_num,
                                                    (PyObject *)extra_op)) {
        /* Make a copy of the input array */
        a_copy = (PyArrayObject *)PyArray_NewLikeArray(a, NPY_ANYORDER,  # 创建输入数组的副本
                                                       NULL, 0);
        if (a_copy == NULL) {
            goto fail;
        }

        if (PyArray_CopyInto(a_copy, a) != 0) {  # 如果复制失败，则跳转到失败
            goto fail;
        }

        Py_INCREF(a);  # 增加input数组的引用计数
        if (PyArray_SetWritebackIfCopyBase(a_copy, a) < 0) {  # 如果设置WRITEBACKIFCOPY出错，则跳转到失败
            goto fail;
        }

        a = a_copy;  # 将input数组指向副本
    }

    # 如果不是纯粹的fancy索引，需要获取子空间
    if (index_type != HAS_FANCY) {
        if (get_view_from_index(a, &subspace, indices, index_num, 1) < 0) {  # 如果获取子空间出现错误，则跳转到失败
            goto fail;
        }
    }

    # 创建MapIter对象
    mit = (PyArrayMapIterObject *)PyArray_MapIterNew(indices, index_num,
                                                     index_type, ndim,
                                                     fancy_ndim,
                                                     a, subspace, 0,
                                                     NPY_ITER_READWRITE,
                                                     0, NULL, NULL);
    if (mit == NULL) {  # 如果创建MapIter对象出现错误，则跳转到失败
        goto fail;
    }

    if (PyArray_MapIterCheckIndices(mit) < 0) {  # 检查MapIter的索引
        goto fail;
    }

    if (PyArray_MapIterReset(mit) < 0) {  # 重置MapIter
        goto fail;
    }

    Py_XDECREF(a_copy);  # 释放内存
    Py_XDECREF(subspace);  # 释放内存

    for (i=0; i < index_num; i++) {  # 释放索引对象的内存
        Py_XDECREF(indices[i].object);
    }

    return (PyObject *)mit;  # 返回mit对象

 fail:  # 失败时释放所有分配的内存
    Py_XDECREF(a_copy);  # 释放内存
    Py_XDECREF(subspace);  # 释放内存
    Py_XDECREF((PyObject *)mit);  # 释放内存
    for (i = 0; i < index_num; i++) {  # 释放索引对象的内存
        Py_XDECREF(indices[i].object);
    }
    return NULL;  # 返回NULL
}


# 取消定义多个宏
#undef HAS_INTEGER
#undef HAS_NEWAXIS
#undef HAS_SLICE
#undef HAS_ELLIPSIS
#undef HAS_FANCY
#undef HAS_BOOL
#undef HAS_SCALAR_ARRAY
#undef HAS_0D_BOOL


static void  # 静态函数定义
arraymapiter_dealloc(PyArrayMapIterObject *mit)  # arraymapiter_dealloc函数需要PyArrayMapIterObject指针作为参数
    # 解析并解决可能存在的写回副本问题
    PyArray_ResolveWritebackIfCopy(mit->array);
    # 释放 mit 结构体中的 array 成员的引用
    Py_XDECREF(mit->array);
    # 释放 mit 结构体中的 subspace 成员的引用
    Py_XDECREF(mit->subspace);
    # 释放 mit 结构体中的 extra_op 成员的引用
    Py_XDECREF(mit->extra_op);
    # 如果 mit 结构体中的 outer 成员不为 NULL，则释放其内存空间
    if (mit->outer != NULL) {
        NpyIter_Deallocate(mit->outer);
    }
    # 如果 mit 结构体中的 subspace_iter 成员不为 NULL，则释放其内存空间
    if (mit->subspace_iter != NULL) {
        NpyIter_Deallocate(mit->subspace_iter);
    }
    # 如果 mit 结构体中的 extra_op_iter 成员不为 NULL，则释放其内存空间
    if (mit->extra_op_iter != NULL) {
        NpyIter_Deallocate(mit->extra_op_iter);
    }
    # 释放 mit 结构体本身占用的内存空间
    PyArray_free(mit);
/*
 * The mapiter object must be created new each time.  It does not work
 * to bind to a new array, and continue.
 *
 * This comment explains that the `mapiter` object in NumPy must be 
 * instantiated anew for each use. Attempting to rebind it to a new 
 * array and continue using it does not function correctly.
 */

/*
 * This was the original intention, but currently that does not work.
 * Do not expose the MapIter_Type to Python.
 *
 * This comment notes that the original design intention to expose 
 * `MapIter_Type` to Python has been abandoned due to current 
 * functionality issues.
 */

/*
 * The original mapiter(indexobj); mapiter.bind(a); idea is now fully
 * removed. This is not very useful anyway, since mapiter is equivalent
 * to a[indexobj].flat but the latter gets to use slice syntax.
 *
 * This comment clarifies that the concept of using `mapiter(indexobj); 
 * mapiter.bind(a);` has been completely removed from the implementation. 
 * It states that using `mapiter` was not particularly useful compared 
 * to `a[indexobj].flat`, which benefits from more readable slice syntax.
 */
NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.mapiter",
    .tp_basicsize = sizeof(PyArrayMapIterObject),
    .tp_dealloc = (destructor)arraymapiter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};
```