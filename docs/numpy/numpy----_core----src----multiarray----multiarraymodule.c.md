# `.\numpy\numpy\_core\src\multiarray\multiarraymodule.c`

```py
/*
  Python Multiarray Module -- A useful collection of functions for creating and
  using ndarrays

  Original file
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  Modified for numpy in 2005

  Travis E. Oliphant
  oliphant@ee.byu.edu
  Brigham Young University
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "multiarraymodule.h"
#include "numpy/npy_math.h"
#include "npy_argparse.h"
#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_import.h"
#include "convert_datatype.h"
#include "legacy_dtype_implementation.h"

NPY_NO_EXPORT int NPY_NUMUSERTYPES = 0;

/* Internal APIs */
#include "alloc.h"
#include "abstractdtypes.h"
#include "array_coercion.h"
#include "arrayfunction_override.h"
#include "arraytypes.h"
#include "arrayobject.h"
#include "array_converter.h"
#include "hashdescr.h"
#include "descriptor.h"
#include "dragon4.h"
#include "flagsobject.h"
#include "calculation.h"
#include "number.h"
#include "scalartypes.h"
#include "convert_datatype.h"
#include "conversion_utils.h"
#include "nditer_pywrap.h"
#define NPY_ITERATOR_IMPLEMENTATION_CODE
#include "nditer_impl.h"
#include "methods.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "datetime_busday.h"
#include "datetime_busdaycal.h"
#include "item_selection.h"
#include "shape.h"
#include "ctors.h"
#include "array_assign.h"
#include "common.h"
#include "npy_static_data.h"
#include "cblasfuncs.h"
#include "vdot.h"
#include "templ_common.h" /* for npy_mul_sizes_with_overflow */
#include "compiled_base.h"
#include "mem_overlap.h"
#include "convert.h" /* for PyArray_AssignZero */
#include "lowlevel_strided_loops.h"
#include "dtype_transfer.h"
#include "stringdtype/dtype.h"

#include "get_attr_string.h"
#include "public_dtype_api.h"  /* _fill_dtype_api */
#include "textreading/readtext.h"  /* _readtext_from_file_object */

#include "npy_dlpack.h"

#include "umathmodule.h"

/*
 *****************************************************************************
 **                    INCLUDE GENERATED CODE                               **
 *****************************************************************************
 */
/* __ufunc_api.c define is the PyUFunc_API table: */
#include "__ufunc_api.c"

// 定义全局变量 NPY_NUMUSERTYPES，初始化为 0
NPY_NO_EXPORT int NPY_NUMUSERTYPES = 0;

// 声明初始化标量数学函数的函数
NPY_NO_EXPORT int initscalarmath(PyObject *);

// 声明设置矩阵乘法标志的函数，实现在 ufunc_object.c 中
NPY_NO_EXPORT int set_matmul_flags(PyObject *d);

// 声明在 umath/string_ufuncs.cpp/h 中实现的字符串比较函数
NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip);

// 定义设置遗留打印模式的函数
static PyObject *
set_legacy_print_mode(PyObject *NPY_UNUSED(self), PyObject *args)
{
    // 解析参数，设置遗留打印模式
    if (!PyArg_ParseTuple(args, "i", &npy_thread_unsafe_state.legacy_print_mode)) {
        return NULL;
    }
    // 成功设置模式后返回 NULL
    return NULL;
}
    # 如果不处于 legacy_print_mode（即旧版打印模式），则设置为 INT_MAX
    if (!npy_thread_unsafe_state.legacy_print_mode) {
        npy_thread_unsafe_state.legacy_print_mode = INT_MAX;
    }
    # 返回 Python 中的 None 对象，表示函数执行成功但无需返回其他数值
    Py_RETURN_NONE;
/*NUMPY_API
 * Get Priority from object
 */
NPY_NO_EXPORT double
PyArray_GetPriority(PyObject *obj, double default_)
{
    PyObject *ret;
    double priority = NPY_PRIORITY;

    // 检查对象是否为精确的 NumPy 数组对象
    if (PyArray_CheckExact(obj)) {
        return priority;  // 如果是，则返回默认优先级
    }
    else if (PyArray_CheckAnyScalarExact(obj)) {
        return NPY_SCALAR_PRIORITY;  // 如果是精确的标量对象，则返回标量优先级
    }

    // 尝试在对象实例上查找特定属性 np.array_priority
    ret = PyArray_LookupSpecial_OnInstance(obj, npy_interned_str.array_priority);
    if (ret == NULL) {
        if (PyErr_Occurred()) {
            /* TODO[gh-14801]: propagate crashes during attribute access? */
            PyErr_Clear();  // 如果出现错误，清除异常状态
        }
        return default_;  // 返回默认优先级
    }

    // 将返回的属性值转换为双精度浮点数
    priority = PyFloat_AsDouble(ret);
    Py_DECREF(ret);  // 减少属性对象的引用计数
    if (error_converting(priority)) {
        /* TODO[gh-14801]: propagate crashes for bad priority? */
        PyErr_Clear();  // 如果转换出错，清除异常状态
        return default_;  // 返回默认优先级
    }
    return priority;  // 返回计算得到的优先级
}

/*NUMPY_API
 * Multiply a List of ints
 */
NPY_NO_EXPORT int
PyArray_MultiplyIntList(int const *l1, int n)
{
    int s = 1;

    // 逐个将列表中的整数相乘
    while (n--) {
        s *= (*l1++);
    }
    return s;  // 返回乘积
}

/*NUMPY_API
 * Multiply a List
 */
NPY_NO_EXPORT npy_intp
PyArray_MultiplyList(npy_intp const *l1, int n)
{
    npy_intp s = 1;

    // 逐个将列表中的整数相乘
    while (n--) {
        s *= (*l1++);
    }
    return s;  // 返回乘积
}

/*NUMPY_API
 * Multiply a List of Non-negative numbers with over-flow detection.
 */
NPY_NO_EXPORT npy_intp
PyArray_OverflowMultiplyList(npy_intp const *l1, int n)
{
    npy_intp prod = 1;
    int i;

    // 逐个将列表中的非负数相乘，检测溢出
    for (i = 0; i < n; i++) {
        npy_intp dim = l1[i];

        if (dim == 0) {
            return 0;  // 如果遇到零，直接返回零
        }
        if (npy_mul_sizes_with_overflow(&prod, prod, dim)) {
            return -1;  // 如果乘法溢出，返回负一
        }
    }
    return prod;  // 返回乘积
}

/*NUMPY_API
 * Produce a pointer into array
 */
NPY_NO_EXPORT void *
PyArray_GetPtr(PyArrayObject *obj, npy_intp const* ind)
{
    int n = PyArray_NDIM(obj);
    npy_intp *strides = PyArray_STRIDES(obj);
    char *dptr = PyArray_DATA(obj);

    // 计算多维数组中指定索引的指针位置
    while (n--) {
        dptr += (*strides++) * (*ind++);
    }
    return (void *)dptr;  // 返回指针
}

/*NUMPY_API
 * Compare Lists
 */
NPY_NO_EXPORT int
PyArray_CompareLists(npy_intp const *l1, npy_intp const *l2, int n)
{
    int i;

    // 比较两个列表中的元素是否相等
    for (i = 0; i < n; i++) {
        if (l1[i] != l2[i]) {
            return 0;  // 如果有不相等的元素，返回零
        }
    }
    return 1;  // 如果所有元素都相等，返回一
}

/*
 * simulates a C-style 1-3 dimensional array which can be accessed using
 * ptr[i]  or ptr[i][j] or ptr[i][j][k] -- requires pointer allocation
 * for 2-d and 3-d.
 *
 * For 2-d and up, ptr is NOT equivalent to a statically defined
 * 2-d or 3-d array.  In particular, it cannot be passed into a
 * function that requires a true pointer to a fixed-size array.
 */

/*NUMPY_API
 * Simulate a C-array
 * steals a reference to typedescr -- can be NULL
 */
NPY_NO_EXPORT int
PyArray_AsCArray(PyObject **op, void *ptr, npy_intp *dims, int nd,
                 PyArray_Descr* typedescr)
{
    PyArrayObject *ap;
    npy_intp n, m, i, j;
    char **ptr2;
    char ***ptr3;
    # 检查数组的维度是否在1到3之间，如果不是，则设置异常并返回-1
    if ((nd < 1) || (nd > 3)) {
        PyErr_SetString(PyExc_ValueError,
                        "C arrays of only 1-3 dimensions available");
        Py_XDECREF(typedescr);
        return -1;
    }
    
    # 尝试将对象 *op 转换为 PyArrayObject 类型，使用指定的描述符和维度，存储为 C 风格数组
    if ((ap = (PyArrayObject*)PyArray_FromAny(*op, typedescr, nd, nd,
                                      NPY_ARRAY_CARRAY, NULL)) == NULL) {
        return -1;
    }
    
    # 根据数组的维度进行不同的处理
    switch(nd) {
    case 1:
        # 对于一维数组，将指针 ptr 指向数组的数据
        *((char **)ptr) = PyArray_DATA(ap);
        break;
    case 2:
        # 对于二维数组，获取数组的第一维长度 n
        n = PyArray_DIMS(ap)[0];
        # 分配内存以存储指向每行数据的指针
        ptr2 = (char **)PyArray_malloc(n * sizeof(char *));
        if (!ptr2) {
            PyErr_NoMemory();
            return -1;
        }
        # 遍历数组每行，设置指针数组 ptr2[i] 指向每行数据的起始位置
        for (i = 0; i < n; i++) {
            ptr2[i] = PyArray_BYTES(ap) + i*PyArray_STRIDES(ap)[0];
        }
        # 将指针数组的地址赋给 ptr
        *((char ***)ptr) = ptr2;
        break;
    case 3:
        # 对于三维数组，获取数组的第一维长度 n 和第二维长度 m
        n = PyArray_DIMS(ap)[0];
        m = PyArray_DIMS(ap)[1];
        # 分配内存以存储指向每个元素数据的指针
        ptr3 = (char ***)PyArray_malloc(n*(m+1) * sizeof(char *));
        if (!ptr3) {
            PyErr_NoMemory();
            return -1;
        }
        # 使用两层循环遍历数组每个元素，设置指针数组 ptr3[i][j] 指向每个元素的数据起始位置
        for (i = 0; i < n; i++) {
            ptr3[i] = (char **) &ptr3[n + m * i];
            for (j = 0; j < m; j++) {
                ptr3[i][j] = PyArray_BYTES(ap) + i*PyArray_STRIDES(ap)[0] + j*PyArray_STRIDES(ap)[1];
            }
        }
        # 将指针数组的地址赋给 ptr
        *((char ****)ptr) = ptr3;
    }
    
    # 如果数组的维度大于0，则复制数组的维度信息到 dims 数组中
    if (nd) {
        memcpy(dims, PyArray_DIMS(ap), nd*sizeof(npy_intp));
    }
    
    # 将处理后的数组对象 ap 赋给 *op
    *op = (PyObject *)ap;
    
    # 返回操作成功的标志
    return 0;
/*NUMPY_API
 * Free pointers created if As2D is called
 */
NPY_NO_EXPORT int
PyArray_Free(PyObject *op, void *ptr)
{
    PyArrayObject *ap = (PyArrayObject *)op;

    // 如果数组维度小于1或大于3，则返回错误
    if ((PyArray_NDIM(ap) < 1) || (PyArray_NDIM(ap) > 3)) {
        return -1;
    }
    // 如果数组维度大于等于2，则释放指针
    if (PyArray_NDIM(ap) >= 2) {
        PyArray_free(ptr);
    }
    // 减少数组对象的引用计数
    Py_DECREF(ap);
    return 0;
}

/*
 * Get the ndarray subclass with the highest priority
 */
NPY_NO_EXPORT PyTypeObject *
PyArray_GetSubType(int narrays, PyArrayObject **arrays) {
    PyTypeObject *subtype = &PyArray_Type;
    double priority = NPY_PRIORITY;
    int i;

    /* 获取具有最高优先级的子类 */
    for (i = 0; i < narrays; ++i) {
        if (Py_TYPE(arrays[i]) != subtype) {
            double pr = PyArray_GetPriority((PyObject *)(arrays[i]), 0.0);
            // 如果当前数组的优先级高于已知的最高优先级，则更新子类
            if (pr > priority) {
                priority = pr;
                subtype = Py_TYPE(arrays[i]);
            }
        }
    }

    return subtype;
}


/*
 * Concatenates a list of ndarrays.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ConcatenateArrays(int narrays, PyArrayObject **arrays, int axis,
                          PyArrayObject* ret, PyArray_Descr *dtype,
                          NPY_CASTING casting)
{
    int iarrays, idim, ndim;
    npy_intp shape[NPY_MAXDIMS];
    PyArrayObject_fields *sliding_view = NULL;

    // 至少需要一个数组来进行连接
    if (narrays <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need at least one array to concatenate");
        return NULL;
    }

    // 所有数组必须具有相同的维度
    ndim = PyArray_NDIM(arrays[0]);
    if (ndim == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "zero-dimensional arrays cannot be concatenated");
        return NULL;
    }

    // 处理标准的负索引
    if (check_and_adjust_axis(&axis, ndim) < 0) {
        return NULL;
    }

    /*
     * 从第一个数组的形状开始计算最终连接的形状
     */
    memcpy(shape, PyArray_SHAPE(arrays[0]), ndim * sizeof(shape[0]));
    for (iarrays = 1; iarrays < narrays; ++iarrays) {
        npy_intp *arr_shape;

        // 检查当前数组与第一个数组的维度是否一致
        if (PyArray_NDIM(arrays[iarrays]) != ndim) {
            PyErr_Format(PyExc_ValueError,
                         "all the input arrays must have same number of "
                         "dimensions, but the array at index %d has %d "
                         "dimension(s) and the array at index %d has %d "
                         "dimension(s)",
                         0, ndim, iarrays, PyArray_NDIM(arrays[iarrays]));
            return NULL;
        }
        // 获取当前数组的形状
        arr_shape = PyArray_SHAPE(arrays[iarrays]);

        // 遍历每个维度
        for (idim = 0; idim < ndim; ++idim) {
            /* Build up the size of the concatenation axis */
            // 如果当前维度是连接轴（axis），增加该维度的大小
            if (idim == axis) {
                shape[idim] += arr_shape[idim];
            }
            /* Validate that the rest of the dimensions match */
            // 否则，验证当前维度与第一个数组的对应维度是否相等
            else if (shape[idim] != arr_shape[idim]) {
                PyErr_Format(PyExc_ValueError,
                             "all the input array dimensions except for the "
                             "concatenation axis must match exactly, but "
                             "along dimension %d, the array at index %d has "
                             "size %d and the array at index %d has size %d",
                             idim, 0, shape[idim], iarrays, arr_shape[idim]);
                return NULL;
            }
        }
    }

    // 如果输出数组 ret 不为空，进行进一步验证
    if (ret != NULL) {
        assert(dtype == NULL);
        // 检查输出数组的维度是否正确
        if (PyArray_NDIM(ret) != ndim) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array has wrong dimensionality");
            return NULL;
        }
        // 检查输出数组的形状是否与期望的形状相符
        if (!PyArray_CompareLists(shape, PyArray_SHAPE(ret), ndim)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array is the wrong shape");
            return NULL;
        }
        // 增加输出数组的引用计数
        Py_INCREF(ret);
    }
    else {
        npy_intp s, strides[NPY_MAXDIMS];
        int strideperm[NPY_MAXDIMS];

        /* 获取数组的优先子类型 */
        PyTypeObject *subtype = PyArray_GetSubType(narrays, arrays);
        /* 找到合并后数组的描述符 */
        PyArray_Descr *descr = PyArray_FindConcatenationDescriptor(
                narrays, arrays, dtype);
        if (descr == NULL) {
            return NULL;
        }

        /*
         * 计算需要对步幅进行的排列，以匹配输入数组的内存布局，
         * 使用与 NpyIter 类似的歧义解析规则。
         */
        PyArray_CreateMultiSortedStridePerm(narrays, arrays, ndim, strideperm);
        s = descr->elsize;
        for (idim = ndim-1; idim >= 0; --idim) {
            int iperm = strideperm[idim];
            strides[iperm] = s;
            s *= shape[iperm];
        }

        /* 分配结果数组的空间。这里会窃取 'dtype' 的引用。 */
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                subtype, descr, ndim, shape, strides, NULL, 0, NULL,
                NULL, _NPY_ARRAY_ALLOW_EMPTY_STRING);
        if (ret == NULL) {
            return NULL;
        }
    }

    /*
     * 创建一个视图，用于在 ret 中滑动以赋值各个输入数组。
     */
    sliding_view = (PyArrayObject_fields *)PyArray_View(ret,
                                                        NULL, &PyArray_Type);
    if (sliding_view == NULL) {
        Py_DECREF(ret);
        return NULL;
    }
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        /* 设置维度以匹配输入数组的维度 */
        sliding_view->dimensions[axis] = PyArray_SHAPE(arrays[iarrays])[axis];

        /* 复制当前数组的数据 */
        if (PyArray_AssignArray((PyArrayObject *)sliding_view, arrays[iarrays],
                            NULL, casting) < 0) {
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* 滑动到下一个窗口的起始位置 */
        sliding_view->data += sliding_view->dimensions[axis] *
                                 sliding_view->strides[axis];
    }

    Py_DECREF(sliding_view);
    return ret;
/*
 * Concatenates a list of ndarrays, flattening each in the specified order.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ConcatenateFlattenedArrays(int narrays, PyArrayObject **arrays,
                                   NPY_ORDER order, PyArrayObject *ret,
                                   PyArray_Descr *dtype, NPY_CASTING casting,
                                   npy_bool casting_not_passed)
{
    int iarrays;
    npy_intp shape = 0;
    PyArrayObject_fields *sliding_view = NULL;

    // 检查输入的数组数量是否合法，至少需要一个数组
    if (narrays <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need at least one array to concatenate");
        return NULL;
    }

    /*
     * 计算最终连接后的数组形状，从第一个数组的形状开始计算。
     */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        shape += PyArray_SIZE(arrays[iarrays]);
        /* 检查是否溢出 */
        if (shape < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "total number of elements "
                            "too large to concatenate");
            return NULL;
        }
    }

    int out_passed = 0;
    // 如果已经提供了输出数组ret，则验证其合法性
    if (ret != NULL) {
        assert(dtype == NULL);
        out_passed = 1;
        // 输出数组必须是一维的
        if (PyArray_NDIM(ret) != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array must be 1D");
            return NULL;
        }
        // 输出数组的大小必须与计算得到的shape相等
        if (shape != PyArray_SIZE(ret)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output array is the wrong size");
            return NULL;
        }
        // 增加输出数组的引用计数
        Py_INCREF(ret);
    }
    else {
        npy_intp stride;

        /* 获取数组的优先子类型 */
        PyTypeObject *subtype = PyArray_GetSubType(narrays, arrays);

        // 查找连接数组时需要的描述符
        PyArray_Descr *descr = PyArray_FindConcatenationDescriptor(
                narrays, arrays, dtype);
        if (descr == NULL) {
            return NULL;
        }

        stride = descr->elsize;

        /*
         * 分配结果数组的内存空间。这里会使用描述符来分配，并且会偷窃描述符的引用。
         */
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                subtype, descr,  1, &shape, &stride, NULL, 0, NULL,
                NULL, _NPY_ARRAY_ALLOW_EMPTY_STRING);
        if (ret == NULL) {
            return NULL;
        }
        assert(PyArray_DESCR(ret) == descr);
    }

    /*
     * 创建一个视图，通过该视图可以在结果数组ret中滑动并逐个赋值输入数组。
     */
    sliding_view = (PyArrayObject_fields *)PyArray_View(ret,
                                                        NULL, &PyArray_Type);
    if (sliding_view == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    // 是否需要给出弃用警告，仅用于第一个输入数组
    int give_deprecation_warning = 1;  /* To give warning for just one input array. */
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        /* Adjust the window dimensions for this array */
        // 更新滑动视图的第一个维度为当前数组的大小
        sliding_view->dimensions[0] = PyArray_SIZE(arrays[iarrays]);

        if (!PyArray_CanCastArrayTo(
                arrays[iarrays], PyArray_DESCR(ret), casting)) {
            /* This should be an error, but was previously allowed here. */
            // 如果无法将数组 `arrays[iarrays]` 强制转换为 `ret` 的数据类型，则处理错误
            if (casting_not_passed && out_passed) {
                /* NumPy 1.20, 2020-09-03 */
                // 如果需要发出弃用警告，并且发出警告时出现错误，则清理资源并返回 NULL
                if (give_deprecation_warning && DEPRECATE(
                        "concatenate() with `axis=None` will use same-kind "
                        "casting by default in the future. Please use "
                        "`casting='unsafe'` to retain the old behaviour. "
                        "In the future this will be a TypeError.") < 0) {
                    Py_DECREF(sliding_view);
                    Py_DECREF(ret);
                    return NULL;
                }
                give_deprecation_warning = 0;
            }
            else {
                // 设置类型转换错误并返回 NULL
                npy_set_invalid_cast_error(
                        PyArray_DESCR(arrays[iarrays]), PyArray_DESCR(ret),
                        casting, PyArray_NDIM(arrays[iarrays]) == 0);
                Py_DECREF(sliding_view);
                Py_DECREF(ret);
                return NULL;
            }
        }

        /* Copy the data for this array */
        // 复制当前数组的数据到滑动视图
        if (PyArray_CopyAsFlat((PyArrayObject *)sliding_view, arrays[iarrays],
                            order) < 0) {
            // 如果复制失败，则清理资源并返回 NULL
            Py_DECREF(sliding_view);
            Py_DECREF(ret);
            return NULL;
        }

        /* Slide to the start of the next window */
        // 将滑动视图的数据指针滑动到下一个窗口的起始位置
        sliding_view->data +=
            sliding_view->strides[0] * PyArray_SIZE(arrays[iarrays]);
    }

    // 释放滑动视图的引用并返回结果数组
    Py_DECREF(sliding_view);
    return ret;
/**
 * Implementation for np.concatenate
 *
 * @param op Sequence of arrays to concatenate
 * @param axis Axis to concatenate along
 * @param ret output array to fill
 * @param dtype Forced output array dtype (cannot be combined with ret)
 * @param casting Casting mode used
 * @param casting_not_passed Deprecation helper
 */
NPY_NO_EXPORT PyObject *
PyArray_ConcatenateInto(PyObject *op,
        int axis, PyArrayObject *ret, PyArray_Descr *dtype,
        NPY_CASTING casting, npy_bool casting_not_passed)
{
    int iarrays, narrays;
    PyArrayObject **arrays;

    // 检查输入参数op是否为序列类型，如果不是则返回类型错误
    if (!PySequence_Check(op)) {
        PyErr_SetString(PyExc_TypeError,
                        "The first input argument needs to be a sequence");
        return NULL;
    }
    // 如果同时提供了ret和dtype参数，则返回类型错误
    if (ret != NULL && dtype != NULL) {
        PyErr_SetString(PyExc_TypeError,
                "concatenate() only takes `out` or `dtype` as an "
                "argument, but both were provided.");
        return NULL;
    }

    /* Convert the input list into arrays */
    // 获取序列op的长度，如果获取失败则返回NULL
    narrays = PySequence_Size(op);
    if (narrays < 0) {
        return NULL;
    }
    // 分配存储PyArrayObject指针的数组空间
    arrays = PyArray_malloc(narrays * sizeof(arrays[0]));
    if (arrays == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    // 遍历序列op中的每个元素，将其转换为PyArrayObject
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        PyObject *item = PySequence_GetItem(op, iarrays);
        if (item == NULL) {
            // 如果获取元素失败，则记录当前处理到的元素个数，跳转到失败处理部分
            narrays = iarrays;
            goto fail;
        }
        // 将Python对象item转换为PyArrayObject
        arrays[iarrays] = (PyArrayObject *)PyArray_FROM_O(item);
        if (arrays[iarrays] == NULL) {
            // 如果转换失败，则释放前面成功转换的对象并记录失败位置，跳转到失败处理部分
            Py_DECREF(item);
            narrays = iarrays;
            goto fail;
        }
        // 标记item对象为临时数组，用于在处理时避免误操作
        npy_mark_tmp_array_if_pyscalar(item, arrays[iarrays], NULL);
        Py_DECREF(item);
    }

    // 根据axis参数确定调用不同的数组拼接函数
    if (axis == NPY_RAVEL_AXIS) {
        ret = PyArray_ConcatenateFlattenedArrays(
                narrays, arrays, NPY_CORDER, ret, dtype,
                casting, casting_not_passed);
    }
    else {
        ret = PyArray_ConcatenateArrays(
                narrays, arrays, axis, ret, dtype, casting);
    }

    // 释放分配的PyArrayObject指针数组及其元素
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        Py_DECREF(arrays[iarrays]);
    }
    PyArray_free(arrays);

    // 返回拼接后的结果数组对象
    return (PyObject *)ret;

fail:
    /* 'narrays' was set to how far we got in the conversion */
    // 处理转换失败时释放已成功转换的对象，并释放数组空间
    for (iarrays = 0; iarrays < narrays; ++iarrays) {
        Py_DECREF(arrays[iarrays]);
    }
    PyArray_free(arrays);

    return NULL;
}

/*NUMPY_API
 * Concatenate
 *
 * Concatenate an arbitrary Python sequence into an array.
 * op is a python object supporting the sequence interface.
 * Its elements will be concatenated together to form a single
 * multidimensional array. If axis is NPY_MAXDIMS or bigger, then
 * each sequence object will be flattened before concatenation
*/
NPY_NO_EXPORT PyObject *
PyArray_Concatenate(PyObject *op, int axis)
{
    // 保留旧版的类型转换行为
    NPY_CASTING casting;
    // 如果axis大于或等于NPY_MAXDIMS，则设置类型转换模式为NPY_UNSAFE_CASTING
    if (axis >= NPY_MAXDIMS) {
        casting = NPY_UNSAFE_CASTING;
    }
    # 如果不是上述条件，将 casting 设置为 NPY_SAME_KIND_CASTING
    else {
        casting = NPY_SAME_KIND_CASTING;
    }
    # 调用 PyArray_ConcatenateInto 函数进行数组拼接操作
    return PyArray_ConcatenateInto(
            op, axis, NULL, NULL, casting, 0);
}

static int
_signbit_set(PyArrayObject *arr)
{
    static char bitmask = (char) 0x80;  // 定义一个静态字符变量，用于表示最高位为1的位掩码
    char *ptr;  /* points to the npy_byte to test */  // 指向要测试的 npy_byte 的指针
    char byteorder;  // 存储数组的字节顺序信息
    int elsize;  // 存储数组元素的大小

    elsize = PyArray_ITEMSIZE(arr);  // 获取数组元素的大小
    byteorder = PyArray_DESCR(arr)->byteorder;  // 获取数组的字节顺序
    ptr = PyArray_DATA(arr);  // 获取数组数据的指针
    if (elsize > 1 &&
        (byteorder == NPY_LITTLE ||
         (byteorder == NPY_NATIVE &&
          PyArray_ISNBO(NPY_LITTLE)))) {
        ptr += elsize - 1;  // 如果数组元素大小大于1且字节顺序为小端或者本机字节顺序是小端并且数组也是小端，将指针移动到倒数第二个元素
    }
    return ((*ptr & bitmask) != 0);  // 返回指针指向的字节与位掩码进行按位与运算后的结果是否为非零
}


/*NUMPY_API
 * ScalarKind
 *
 * Returns the scalar kind of a type number, with an
 * optional tweak based on the scalar value itself.
 * If no scalar is provided, it returns INTPOS_SCALAR
 * for both signed and unsigned integers, otherwise
 * it checks the sign of any signed integer to choose
 * INTNEG_SCALAR when appropriate.
 */
NPY_NO_EXPORT NPY_SCALARKIND
PyArray_ScalarKind(int typenum, PyArrayObject **arr)
{
    NPY_SCALARKIND ret = NPY_NOSCALAR;  // 初始化返回值，默认为 NPY_NOSCALAR

    if ((unsigned int)typenum < NPY_NTYPES_LEGACY) {
        ret = _npy_scalar_kinds_table[typenum];  // 从预定义的标量类型表中获取标量类型
        /* Signed integer types are INTNEG in the table */
        if (ret == NPY_INTNEG_SCALAR) {  // 如果标量类型为负整数
            if (!arr || !_signbit_set(*arr)) {  // 如果没有提供数组或者数组的最高位不是1
                ret = NPY_INTPOS_SCALAR;  // 返回正整数标量类型
            }
        }
    } else if (PyTypeNum_ISUSERDEF(typenum)) {  // 如果类型号是用户定义的类型
        PyArray_Descr* descr = PyArray_DescrFromType(typenum);  // 根据类型号获取数据类型描述符

        if (PyDataType_GetArrFuncs(descr)->scalarkind) {
            ret = PyDataType_GetArrFuncs(descr)->scalarkind((arr ? *arr : NULL));  // 获取数据类型描述符的标量类型
        }
        Py_DECREF(descr);  // 减少数据类型描述符的引用计数
    }

    return ret;  // 返回标量类型
}

/*NUMPY_API
 *
 * Determines whether the data type 'thistype', with
 * scalar kind 'scalar', can be coerced into 'neededtype'.
 */
NPY_NO_EXPORT int
PyArray_CanCoerceScalar(int thistype, int neededtype,
                        NPY_SCALARKIND scalar)
{
    PyArray_Descr* from;
    int *castlist;

    /* If 'thistype' is not a scalar, it must be safely castable */
    if (scalar == NPY_NOSCALAR) {
        return PyArray_CanCastSafely(thistype, neededtype);  // 如果不是标量类型，直接检查是否能够安全地转换
    }
    if ((unsigned int)neededtype < NPY_NTYPES_LEGACY) {
        NPY_SCALARKIND neededscalar;

        if (scalar == NPY_OBJECT_SCALAR) {
            return PyArray_CanCastSafely(thistype, neededtype);  // 如果目标类型是对象类型标量，直接检查是否能够安全地转换
        }

        /*
         * The lookup table gives us exactly what we need for
         * this comparison, which PyArray_ScalarKind would not.
         *
         * The rule is that positive scalars can be coerced
         * to a signed ints, but negative scalars cannot be coerced
         * to unsigned ints.
         *   _npy_scalar_kinds_table[int]==NEGINT > POSINT,
         *      so 1 is returned, but
         *   _npy_scalar_kinds_table[uint]==POSINT < NEGINT,
         *      so 0 is returned, as required.
         *
         */
        neededscalar = _npy_scalar_kinds_table[neededtype];  // 获取目标类型的标量类型
        if (neededscalar >= scalar) {
            return 1;  // 如果目标标量类型大于等于当前标量类型，返回1
        }
        if (!PyTypeNum_ISUSERDEF(thistype)) {
            return 0;  // 如果当前类型不是用户定义的类型，返回0
        }
    }

这是一段代码的结尾，表示一个函数或者一个代码块的结束。


    from = PyArray_DescrFromType(thistype);

调用 `PyArray_DescrFromType` 函数，根据 `thistype` 参数获取一个描述符对象，并将其赋值给 `from` 变量。


    if (PyDataType_GetArrFuncs(from)->cancastscalarkindto

检查从描述符对象 `from` 获取的数组函数集合中是否存在 `cancastscalarkindto` 字段或属性。


        && (castlist = PyDataType_GetArrFuncs(from)->cancastscalarkindto[scalar])) {

如果 `cancastscalarkindto` 存在，则将其与 `scalar` 索引相关联的 `castlist` 变量进行比较。


        while (*castlist != NPY_NOTYPE) {

进入一个循环，该循环将检查 `castlist` 中的每个元素，直到遇到 `NPY_NOTYPE` 结束循环。


            if (*castlist++ == neededtype) {

检查当前 `castlist` 指向的元素是否等于 `neededtype`。


                Py_DECREF(from);

如果找到了匹配的 `neededtype`，则释放 `from` 对象，并返回 `1` 表示找到匹配。


                return 1;
            }
        }
    }

结束循环后，如果没有找到匹配的 `neededtype`，继续执行下面的代码。


    Py_DECREF(from);

在函数返回之前，释放 `from` 对象。


    return 0;

如果没有找到匹配的 `neededtype`，则返回 `0` 表示未找到匹配。
}

/* Could perhaps be redone to not make contiguous arrays */

/*NUMPY_API
 * Numeric.innerproduct(a,v)
 */
NPY_NO_EXPORT PyObject *
PyArray_InnerProduct(PyObject *op1, PyObject *op2)
{
    PyArrayObject *ap1 = NULL;  // 定义数组对象指针 ap1，初始化为 NULL
    PyArrayObject *ap2 = NULL;  // 定义数组对象指针 ap2，初始化为 NULL
    int typenum;  // 定义整型变量 typenum，用于存储数组元素的数据类型编号
    PyArray_Descr *typec = NULL;  // 定义数组描述符指针 typec，初始化为 NULL
    PyObject* ap2t = NULL;  // 定义 Python 对象指针 ap2t，初始化为 NULL
    npy_intp dims[NPY_MAXDIMS];  // 定义整型数组 dims，用于存储数组维度信息
    PyArray_Dims newaxes = {dims, 0};  // 定义 PyArray_Dims 结构 newaxes，并初始化其维度信息为 dims，长度为 0
    int i;  // 定义整型变量 i，用于循环计数
    PyObject* ret = NULL;  // 定义 Python 对象指针 ret，初始化为 NULL

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);  // 调用 PyArray_ObjectType 函数获取 op1 的数组元素数据类型编号
    if (typenum == NPY_NOTYPE) {  // 如果获取的数据类型编号为 NPY_NOTYPE，返回 NULL
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);  // 调用 PyArray_ObjectType 函数获取 op2 的数组元素数据类型编号，并与之前的 typenum 进行匹配
    if (typenum == NPY_NOTYPE) {  // 如果获取的数据类型编号为 NPY_NOTYPE，返回 NULL
        return NULL;
    }

    typec = PyArray_DescrFromType(typenum);  // 根据 typenum 获取数组描述符 typec
    if (typec == NULL) {  // 如果获取的数组描述符为 NULL
        if (!PyErr_Occurred()) {  // 如果没有发生 Python 异常
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");  // 设置类型错误异常信息
        }
        goto fail;  // 跳转到 fail 标签处，执行清理操作
    }

    Py_INCREF(typec);  // 增加 typec 的引用计数
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, NULL);  // 根据 op1 和 typec 创建 PyArrayObject 对象 ap1
    if (ap1 == NULL) {  // 如果创建的 ap1 为 NULL
        Py_DECREF(typec);  // 减少 typec 的引用计数
        goto fail;  // 跳转到 fail 标签处，执行清理操作
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                           NPY_ARRAY_ALIGNED, NULL);  // 根据 op2 和 typec 创建 PyArrayObject 对象 ap2
    if (ap2 == NULL) {  // 如果创建的 ap2 为 NULL
        goto fail;  // 跳转到 fail 标签处，执行清理操作
    }

    newaxes.len = PyArray_NDIM(ap2);  // 设置 newaxes 结构体的长度为 ap2 的维度数
    if ((PyArray_NDIM(ap1) >= 1) && (newaxes.len >= 2)) {  // 如果 ap1 的维度大于等于 1 并且 newaxes 的长度大于等于 2
        for (i = 0; i < newaxes.len - 2; i++) {  // 循环遍历 newaxes 长度减去 2 的次数
            dims[i] = (npy_intp)i;  // 设置 dims 数组的值为循环变量 i 的值
        }
        dims[newaxes.len - 2] = newaxes.len - 1;  // 设置 dims 数组倒数第二个元素为 newaxes 长度减 1 的值
        dims[newaxes.len - 1] = newaxes.len - 2;  // 设置 dims 数组最后一个元素为 newaxes 长度减 2 的值

        ap2t = PyArray_Transpose(ap2, &newaxes);  // 对 ap2 进行转置操作，结果存储在 ap2t 中
        if (ap2t == NULL) {  // 如果转置操作返回 NULL
            goto fail;  // 跳转到 fail 标签处，执行清理操作
        }
    }
    else {  // 如果 ap1 的维度小于 1 或者 newaxes 的长度小于 2
        ap2t = (PyObject *)ap2;  // 直接将 ap2 赋给 ap2t
        Py_INCREF(ap2);  // 增加 ap2 的引用计数
    }

    ret = PyArray_MatrixProduct2((PyObject *)ap1, ap2t, NULL);  // 调用 PyArray_MatrixProduct2 函数进行矩阵乘法运算，结果存储在 ret 中
    if (ret == NULL) {  // 如果 ret 为 NULL
        goto fail;  // 跳转到 fail 标签处，执行清理操作
    }


    Py_DECREF(ap1);  // 减少 ap1 的引用计数
    Py_DECREF(ap2);  // 减少 ap2 的引用计数
    Py_DECREF(ap2t);  // 减少 ap2t 的引用计数
    return ret;  // 返回 ret

fail:  // 定义 fail 标签处，用于处理错误清理操作
    Py_XDECREF(ap1);  // 安全地减少 ap1 的引用计数
    Py_XDECREF(ap2);  // 安全地减少 ap2 的引用计数
    Py_XDECREF(ap2t);  // 安全地减少 ap2t 的引用计数
    Py_XDECREF(ret);  // 安全地减少 ret 的引用计数
    return NULL;  // 返回 NULL
}

/*NUMPY_API
 * Numeric.matrixproduct(a,v)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct(PyObject *op1, PyObject *op2)
{
    return PyArray_MatrixProduct2(op1, op2, NULL);  // 调用 PyArray_MatrixProduct2 函数进行矩阵乘法运算，结果返回
}

/*NUMPY_API
 * Numeric.matrixproduct2(a,v,out)
 * just like inner product but does the swapaxes stuff on the fly
 */
NPY_NO_EXPORT PyObject *
PyArray_MatrixProduct2(PyObject *op1, PyObject *op2, PyArrayObject* out)
{
    PyArrayObject *ap1, *ap2, *out_buf = NULL, *result = NULL;  // 定义多个 PyArrayObject 指针，初始化为 NULL
    PyArrayIterObject *it1, *it2;  // 定义 PyArrayIterObject 迭代器指针
    npy_intp i, j, l;  // 定义多个整型变量
    int typenum, nd, axis, matchDim;  // 定义多个整型变量
    npy_intp is1, is2, os;  // 定义多个整型变量
    char *op;  // 定义字符指针 op
    npy_intp dimensions[NPY_MAXDIMS];  // 定义整型数组 dimensions，用于存储数组维度信息
    PyArray_DotFunc *dot;  // 定义 PyArray_DotFunc 结构指针 dot
    PyArray_Descr *typec = NULL;  // 定义数组描述符指针 typec，初始化为 NULL
    NPY_BEGIN_THREADS_DEF;  // 定义多线程宏

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);  // 调用 PyArray_ObjectType 函数获取 op1 的数组元素数据类型编号
    if (typenum == NPY_NOTYPE) {  // 如果获取的数据类型编号为 NPY_NOTYPE，返回 NULL
        return NULL;
    }
    # 使用 op2 的数据类型号码获取相应的对象数据类型
    typenum = PyArray_ObjectType(op2, typenum);
    # 如果获取的数据类型号码是 NPY_NOTYPE，表示未找到有效的数据类型，返回空指针
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    # 根据数据类型号码获取对应的数据描述符
    typec = PyArray_DescrFromType(typenum);
    # 如果获取数据描述符失败，并且没有发生异常，设置一个类型错误的异常信息
    if (typec == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot find a common data type.");
        }
        return NULL;
    }

    # 增加数据描述符的引用计数，以防止其被释放
    Py_INCREF(typec);
    # 从 op1 创建一个 PyArrayObject 对象，要求其数据类型与 typec 一致，要求对齐，不指定其他标志
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    # 如果创建失败，释放之前增加的数据描述符的引用计数，返回空指针
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    # 从 op2 创建一个 PyArrayObject 对象，要求其数据类型与 typec 一致，要求对齐，不指定其他标志
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 0, 0,
                                        NPY_ARRAY_ALIGNED, NULL);
    # 如果创建失败，释放 ap1 对象和之前增加的数据描述符的引用计数，返回空指针
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }
#if defined(HAVE_CBLAS)
    // 检查是否定义了 CBLAS，并且数组维度都不超过2，且数据类型为双精度、单精度或复数类型之一
    if (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2 &&
            (NPY_DOUBLE == typenum || NPY_CDOUBLE == typenum ||
             NPY_FLOAT == typenum || NPY_CFLOAT == typenum)) {
        // 调用 cblas_matrixproduct 函数进行矩阵乘积计算
        return cblas_matrixproduct(typenum, ap1, ap2, out);
    }
#endif

// 处理当 ap1 或 ap2 至少有一个是零维数组的情况
if (PyArray_NDIM(ap1) == 0 || PyArray_NDIM(ap2) == 0) {
    // 调用 multiply 函数计算两个数组的乘积，并返回结果
    PyObject *mul_res = PyObject_CallFunctionObjArgs(
            n_ops.multiply, ap1, ap2, out, NULL);
    // 减少数组 ap1 和 ap2 的引用计数
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    // 返回乘积结果
    return mul_res;
}

// 获取 ap1 最后一个维度的大小
l = PyArray_DIMS(ap1)[PyArray_NDIM(ap1) - 1];

// 确定匹配维度 matchDim 的值
if (PyArray_NDIM(ap2) > 1) {
    matchDim = PyArray_NDIM(ap2) - 2;
} else {
    matchDim = 0;
}

// 检查维度是否匹配
if (PyArray_DIMS(ap2)[matchDim] != l) {
    // 引发维度不匹配错误，并跳转到 fail 标签处理错误
    dot_alignment_error(ap1, PyArray_NDIM(ap1) - 1, ap2, matchDim);
    goto fail;
}

// 计算输出数组的维度数
nd = PyArray_NDIM(ap1) + PyArray_NDIM(ap2) - 2;
// 检查是否超过最大维度限制
if (nd > NPY_MAXDIMS) {
    PyErr_SetString(PyExc_ValueError, "dot: too many dimensions in result");
    goto fail;
}

// 初始化 dimensions 数组以存储输出数组的维度信息
j = 0;
for (i = 0; i < PyArray_NDIM(ap1) - 1; i++) {
    dimensions[j++] = PyArray_DIMS(ap1)[i];
}
for (i = 0; i < PyArray_NDIM(ap2) - 2; i++) {
    dimensions[j++] = PyArray_DIMS(ap2)[i];
}
if (PyArray_NDIM(ap2) > 1) {
    dimensions[j++] = PyArray_DIMS(ap2)[PyArray_NDIM(ap2)-1];
}

// 获取数组 ap1 和 ap2 的步长信息
is1 = PyArray_STRIDES(ap1)[PyArray_NDIM(ap1)-1];
is2 = PyArray_STRIDES(ap2)[matchDim];

/* 选择要返回的子类型 */
// 根据参数创建新的输出数组，并返回其数据缓冲区
out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
if (out_buf == NULL) {
    goto fail;
}

/* 确保当 ap1 和 ap2 均为空数组时，返回全零数组 */
// 如果 ap1 和 ap2 均为空数组，则将输出数组设为全零数组
if (PyArray_SIZE(ap1) == 0 && PyArray_SIZE(ap2) == 0) {
    if (PyArray_AssignZero(out_buf, NULL) < 0) {
        goto fail;
    }
}

// 获取输出数组的 dot 函数
dot = PyDataType_GetArrFuncs(PyArray_DESCR(out_buf))->dotfunc;
if (dot == NULL) {
    // 如果 dot 函数为 NULL，则引发错误并跳转到 fail 标签处理错误
    PyErr_SetString(PyExc_ValueError,
                    "dot not available for this type");
    goto fail;
}

// 初始化输出数组的数据指针和每个元素的字节大小
op = PyArray_DATA(out_buf);
os = PyArray_ITEMSIZE(out_buf);
// 设置 axis 为 ap1 最后一个维度的索引
axis = PyArray_NDIM(ap1)-1;

// 创建迭代器 it1，用于遍历除了最后一个维度外的所有维度
it1 = (PyArrayIterObject *)
    PyArray_IterAllButAxis((PyObject *)ap1, &axis);
if (it1 == NULL) {
    goto fail;
}

// 创建迭代器 it2，用于遍历除了匹配维度外的所有维度
it2 = (PyArrayIterObject *)
    PyArray_IterAllButAxis((PyObject *)ap2, &matchDim);
if (it2 == NULL) {
    Py_DECREF(it1);
    goto fail;
}

// 开始多线程处理，保护 ap2 数组的描述信息
NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap2));
while (it1->index < it1->size) {
    while (it2->index < it2->size) {
        // 对每对迭代器指向的数据执行 dot 操作
        dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
        // 更新输出数据指针到下一个位置
        op += os;
        // 移动 it2 迭代器到下一个元素
        PyArray_ITER_NEXT(it2);
    }
    // 移动 it1 迭代器到下一个元素
    PyArray_ITER_NEXT(it1);
    // 重置 it2 迭代器到起始位置
    PyArray_ITER_RESET(it2);
}
// 结束多线程处理，释放 ap2 数组的描述信息
NPY_END_THREADS_DESCR(PyArray_DESCR(ap2));

// 减少迭代器 it1 和 it2 的引用计数
Py_DECREF(it1);
Py_DECREF(it2);

// 检查是否有 Python 异常发生，如果有，则跳转到 fail 标签处理异常
if (PyErr_Occurred()) {
    /* 仅适用于 OBJECT 类型的数组 */
    goto fail;
}

// 减少数组 ap1 的引用计数
Py_DECREF(ap1);
    // 释放对象引用，减少 `ap2` 的引用计数
    Py_DECREF(ap2);

    // 如果需要，触发将 `out_buf` 中的数据复制回 `result`
    PyArray_ResolveWritebackIfCopy(out_buf);
    
    // 释放对象引用，减少 `out_buf` 的引用计数
    Py_DECREF(out_buf);

    // 返回 `result` 对象作为 Python 对象的指针
    return (PyObject *)result;
fail:
    // 释放指针 ap1 所指向的对象，减少其引用计数
    Py_XDECREF(ap1);
    // 释放指针 ap2 所指向的对象，减少其引用计数
    Py_XDECREF(ap2);
    // 释放指针 out_buf 所指向的对象，减少其引用计数
    Py_XDECREF(out_buf);
    // 释放指针 result 所指向的对象，减少其引用计数
    Py_XDECREF(result);
    // 返回 NULL 指针，表示函数执行失败
    return NULL;
}



/*
 * Implementation which is common between PyArray_Correlate
 * and PyArray_Correlate2.
 *
 * inverted is set to 1 if computed correlate(ap2, ap1), 0 otherwise
 */
static PyArrayObject*
_pyarray_correlate(PyArrayObject *ap1, PyArrayObject *ap2, int typenum,
                   int mode, int *inverted)
{
    PyArrayObject *ret;
    npy_intp length;
    npy_intp i, n1, n2, n, n_left, n_right;
    npy_intp is1, is2, os;
    char *ip1, *ip2, *op;
    PyArray_DotFunc *dot;

    NPY_BEGIN_THREADS_DEF;

    // 获取数组 ap1 和 ap2 的长度
    n1 = PyArray_DIMS(ap1)[0];
    n2 = PyArray_DIMS(ap2)[0];
    // 如果数组 ap1 长度为 0，抛出异常并返回 NULL 指针
    if (n1 == 0) {
        PyErr_SetString(PyExc_ValueError, "first array argument cannot be empty");
        return NULL;
    }
    // 如果数组 ap2 长度为 0，抛出异常并返回 NULL 指针
    if (n2 == 0) {
        PyErr_SetString(PyExc_ValueError, "second array argument cannot be empty");
        return NULL;
    }
    // 如果数组 ap1 长度小于数组 ap2，交换它们的引用，设置 *inverted 为 1
    if (n1 < n2) {
        ret = ap1;
        ap1 = ap2;
        ap2 = ret;
        ret = NULL;
        i = n1;
        n1 = n2;
        n2 = i;
        *inverted = 1;
    } else {
        *inverted = 0;
    }

    length = n1;
    n = n2;
    switch(mode) {
    // 根据 mode 的值设置相关参数
    case 0:
        length = length - n + 1;
        n_left = n_right = 0;
        break;
    case 1:
        n_left = (npy_intp)(n/2);
        n_right = n - n_left - 1;
        break;
    case 2:
        n_right = n - 1;
        n_left = n - 1;
        length = length + n - 1;
        break;
    default:
        // 如果 mode 不是 0、1 或 2，抛出异常并返回 NULL 指针
        PyErr_SetString(PyExc_ValueError, "mode must be 0, 1, or 2");
        return NULL;
    }

    /*
     * 需要选择一个能容纳总和的输出数组
     * -- 使用优先级确定子类型。
     */
    // 根据给定的参数创建一个新的数组，用于保存和的结果
    ret = new_array_for_sum(ap1, ap2, NULL, 1, &length, typenum, NULL);
    if (ret == NULL) {
        return NULL;
    }
    // 获取 ret 数组的 dot 函数
    dot = PyDataType_GetArrFuncs(PyArray_DESCR(ret))->dotfunc;
    if (dot == NULL) {
        // 如果 dot 函数为 NULL，抛出异常并清理 ret 后返回 NULL 指针
        PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
        goto clean_ret;
    }

    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ret));
    // 获取数组 ap1 和 ap2 的步长
    is1 = PyArray_STRIDES(ap1)[0];
    is2 = PyArray_STRIDES(ap2)[0];
    // 获取输出数组的起始地址和单元大小
    op = PyArray_DATA(ret);
    os = PyArray_ITEMSIZE(ret);
    // 获取数组 ap1 和 ap2 的数据地址
    ip1 = PyArray_DATA(ap1);
    ip2 = PyArray_BYTES(ap2) + n_left*is2;
    n = n - n_left;
    // 对左边的部分进行 dot 运算
    for (i = 0; i < n_left; i++) {
        dot(ip1, is1, ip2, is2, op, n, ret);
        n++;
        ip2 -= is2;
        op += os;
    }
    // 如果可以使用 small_correlate 函数，则调用它计算结果
    if (small_correlate(ip1, is1, n1 - n2 + 1, PyArray_TYPE(ap1),
                        ip2, is2, n, PyArray_TYPE(ap2),
                        op, os)) {
        ip1 += is1 * (n1 - n2 + 1);
        op += os * (n1 - n2 + 1);
    }
    else {
        // 否则，使用 dot 函数计算结果
        for (i = 0; i < (n1 - n2 + 1); i++) {
            dot(ip1, is1, ip2, is2, op, n, ret);
            ip1 += is1;
            op += os;
        }
    }


这样，你的代码现在每行都有了详细的注释，解释了每个语句的作用和意图。
    # 对右侧数组中的元素进行循环处理，逐个执行以下操作：
    for (i = 0; i < n_right; i++) {
        # 减少计数器 n 的值，指向下一个元素
        n--;
        # 调用 dot 函数计算两个数组的点积，将结果写入 ret 中
        dot(ip1, is1, ip2, is2, op, n, ret);
        # 移动 ip1 指针，以便下次计算使用下一个元素
        ip1 += is1;
        # 移动 op 指针，以便下次写入结果到正确位置
        op += os;
    }

    # 结束多线程环境，清理可能的线程状态
    NPY_END_THREADS_DESCR(PyArray_DESCR(ret));
    # 检查是否有 Python 异常发生，如果有，则跳转到清理 ret 的代码段
    if (PyErr_Occurred()) {
        goto clean_ret;
    }

    # 返回计算结果 ret
    return ret;
/*
 * Clean up and return NULL on failure
 */
clean_ret:
    Py_DECREF(ret);
    return NULL;
}

/*
 * Revert a one dimensional array in-place
 *
 * Return 0 on success, other value on failure
 */
static int
_pyarray_revert(PyArrayObject *ret)
{
    npy_intp length = PyArray_DIM(ret, 0);  // 获取数组的长度
    npy_intp os = PyArray_ITEMSIZE(ret);    // 获取数组元素的大小
    char *op = PyArray_DATA(ret);           // 获取数组的数据指针
    char *sw1 = op;                         // 指向数组起始位置的指针
    char *sw2;

    if (PyArray_ISNUMBER(ret) && !PyArray_ISCOMPLEX(ret)) {
        /* Optimization for unstructured dtypes */
        PyArray_CopySwapNFunc *copyswapn = PyDataType_GetArrFuncs(PyArray_DESCR(ret))->copyswapn;  // 获取类型特定的数据交换函数
        sw2 = op + length * os - 1;  // 指向数组末尾的指针
        /* First reverse the whole array byte by byte... */
        while(sw1 < sw2) {
            const char tmp = *sw1;
            *sw1++ = *sw2;
            *sw2-- = tmp;
        }
        /* ...then swap in place every item */
        copyswapn(op, os, NULL, 0, length, 1, NULL);  // 在数组中交换每个元素
    }
    else {
        char *tmp = PyArray_malloc(PyArray_ITEMSIZE(ret));  // 分配临时空间用于交换元素
        if (tmp == NULL) {
            PyErr_NoMemory();  // 内存分配失败
            return -1;         // 返回错误值
        }
        sw2 = op + (length - 1) * os;  // 指向数组末尾的指针
        while (sw1 < sw2) {
            memcpy(tmp, sw1, os);   // 将 sw1 处的元素拷贝到 tmp
            memcpy(sw1, sw2, os);   // 将 sw2 处的元素拷贝到 sw1
            memcpy(sw2, tmp, os);   // 将 tmp 中的元素拷贝到 sw2
            sw1 += os;              // 移动指针 sw1
            sw2 -= os;              // 移动指针 sw2
        }
        PyArray_free(tmp);  // 释放临时空间
    }

    return 0;  // 返回成功标志
}

/*NUMPY_API
 * correlate(a1,a2,mode)
 *
 * This function computes the usual correlation (correlate(a1, a2) !=
 * correlate(a2, a1), and conjugate the second argument for complex inputs
 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate2(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    PyArray_Descr *typec;
    int inverted;
    int st;

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);  // 获取第一个输入对象的数据类型
    if (typenum == NPY_NOTYPE) {
        return NULL;  // 数据类型获取失败，返回 NULL
    }
    typenum = PyArray_ObjectType(op2, typenum);  // 获取第二个输入对象的数据类型
    if (typenum == NPY_NOTYPE) {
        return NULL;  // 数据类型获取失败，返回 NULL
    }

    typec = PyArray_DescrFromType(typenum);  // 根据数据类型创建描述符
    Py_INCREF(typec);  // 增加描述符的引用计数
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
                                        NPY_ARRAY_DEFAULT, NULL);  // 将第一个输入对象转换为数组对象
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;  // 转换失败，返回 NULL
    }
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
                                        NPY_ARRAY_DEFAULT, NULL);  // 将第二个输入对象转换为数组对象
    if (ap2 == NULL) {
        goto clean_ap1;  // 转换失败，跳转到清理 ap1 的标签
    }

    if (PyArray_ISCOMPLEX(ap2)) {
        PyArrayObject *cap2;
        cap2 = (PyArrayObject *)PyArray_Conjugate(ap2, NULL);  // 对第二个数组对象取共轭
        if (cap2 == NULL) {
            goto clean_ap2;  // 共轭操作失败，跳转到清理 ap2 的标签
        }
        Py_DECREF(ap2);
        ap2 = cap2;  // 将共轭后的数组对象赋给 ap2
    }

    ret = _pyarray_correlate(ap1, ap2, typenum, mode, &inverted);  // 调用相关函数计算相关性
    if (ret == NULL) {
        goto clean_ap2;  // 相关性计算失败，跳转到清理 ap2 的标签
    }

    /*
     * If we inverted input orders, we need to reverse the output array (i.e.
     * ret = ret[::-1])
     */
    if (inverted) {
        st = _pyarray_revert(ret);  // 如果输入顺序反转，需要反转输出数组
        if (st) {
            goto clean_ret;  // 反转操作失败，跳转到清理 ret 的标签
        }
    }
    // 递减引用计数，减少对ap1指向对象的引用
    Py_DECREF(ap1);
    // 递减引用计数，减少对ap2指向对象的引用
    Py_DECREF(ap2);
    // 返回ret对象的PyObject指针类型
    return (PyObject *)ret;
/* 清理变量和返回 NULL */
clean_ret:
    Py_DECREF(ret);
/* 清理变量 ap2 */
clean_ap2:
    Py_DECREF(ap2);
/* 清理变量 ap1 */
clean_ap1:
    Py_DECREF(ap1);
    return NULL;
}

/*NUMPY_API
 * Numeric.correlate(a1,a2,mode)
 */
/* 导出的 NumPy API 函数，计算数组 a1 和 a2 的相关性 */
NPY_NO_EXPORT PyObject *
PyArray_Correlate(PyObject *op1, PyObject *op2, int mode)
{
    PyArrayObject *ap1, *ap2, *ret = NULL;
    int typenum;
    int unused;
    PyArray_Descr *typec;

    /* 确定 op1 的类型编号 */
    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }
    /* 根据 op1 的类型编号确定 op2 的类型编号 */
    typenum = PyArray_ObjectType(op2, typenum);
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    /* 根据类型编号创建描述符对象 */
    typec = PyArray_DescrFromType(typenum);
    Py_INCREF(typec);
    /* 将 op1 转换为 PyArrayObject 类型 */
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, typec, 1, 1,
                                            NPY_ARRAY_DEFAULT, NULL);
    if (ap1 == NULL) {
        Py_DECREF(typec);
        return NULL;
    }
    /* 将 op2 转换为 PyArrayObject 类型 */
    ap2 = (PyArrayObject *)PyArray_FromAny(op2, typec, 1, 1,
                                           NPY_ARRAY_DEFAULT, NULL);
    if (ap2 == NULL) {
        goto fail;
    }

    /* 调用底层的 _pyarray_correlate 函数计算相关性 */
    ret = _pyarray_correlate(ap1, ap2, typenum, mode, &unused);
    if (ret == NULL) {
        goto fail;
    }
    /* 释放 ap1 和 ap2 对象的引用计数 */
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    return (PyObject *)ret;

fail:
    /* 出错时释放 ap1、ap2 和 ret 对象的引用计数 */
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ret);
    return NULL;
}

/* 静态函数，用于实现 NumPy 中的 array_putmask 函数 */
static PyObject *
array_putmask(PyObject *NPY_UNUSED(module), PyObject *const *args,
                Py_ssize_t len_args, PyObject *kwnames )
{
    PyObject *mask, *values;
    PyObject *array;

    NPY_PREPARE_ARGPARSER;
    /* 解析参数，验证参数类型和个数 */
    if (npy_parse_arguments("putmask", args, len_args, kwnames,
            "", NULL, &array,
            "mask", NULL, &mask,
            "values", NULL, &values,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    /* 检查 array 是否为 NumPy 数组 */
    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_TypeError,
                "argument a of putmask must be a numpy array");
    }

    /* 调用 PyArray_PutMask 函数处理数据 */
    return PyArray_PutMask((PyArrayObject *)array, values, mask);
}


/*NUMPY_API
 *
 * 判断两个数据类型描述符是否等价（基本种类和大小相同）
 */
NPY_NO_EXPORT unsigned char
PyArray_EquivTypes(PyArray_Descr *type1, PyArray_Descr *type2)
{
    /* 如果两个描述符指针相同，说明描述符相等 */
    if (type1 == type2) {
        return 1;
    }

    /*
     * 使用 PyArray_GetCastInfo 而不是 PyArray_CanCastTypeTo，因为它支持
     * 旧的灵活数据类型作为输入。
     */
    npy_intp view_offset;
    /* 获取类型转换的安全性 */
    NPY_CASTING safety = PyArray_GetCastInfo(type1, type2, NULL, &view_offset);
    if (safety < 0) {
        PyErr_Clear();
        return 0;
    }
    /* 如果转换安全性为 "无转换"，则认为这两种类型等价 */
    return PyArray_MinCastSafety(safety, NPY_NO_CASTING) == NPY_NO_CASTING;
}


/*NUMPY_API*/
NPY_NO_EXPORT unsigned char
PyArray_EquivTypenums(int typenum1, int typenum2)
{
    PyArray_Descr *d1, *d2;
    npy_bool ret;

    /* 如果两个类型编号相同，返回成功 */
    if (typenum1 == typenum2) {
        return NPY_SUCCEED;
    }

    /* 根据类型编号创建描述符对象 */
    d1 = PyArray_DescrFromType(typenum1);
    d2 = PyArray_DescrFromType(typenum2);

    /* 如果两个类型编号相同，返回成功 */
    if (typenum1 == typenum2) {
        return NPY_SUCCEED;
    }

    /* 根据类型编号创建描述符对象 */
    d1 = PyArray_DescrFromType(typenum1);
    d2 = PyArray_DescrFromType(typenum2);
    # 比较两个 NumPy 数组描述符的等效性并返回结果
    ret = PyArray_EquivTypes(d1, d2);
    # 减少数组描述符 d1 的引用计数，释放其内存
    Py_DECREF(d1);
    # 减少数组描述符 d2 的引用计数，释放其内存
    Py_DECREF(d2);
    # 返回两个数组描述符的等效性比较结果
    return ret;
/*** END C-API FUNCTIONS **/
/*
 * NOTE: The order specific stride setting is not necessary to preserve
 *       contiguity and could be removed.  However, this way the resulting
 *       strides strides look better for fortran order inputs.
 */
static NPY_STEALS_REF_TO_ARG(1) PyObject *
_prepend_ones(PyArrayObject *arr, int nd, int ndmin, NPY_ORDER order)
{
    npy_intp newdims[NPY_MAXDIMS];  // 用于存放新维度大小的数组
    npy_intp newstrides[NPY_MAXDIMS];  // 用于存放新步幅大小的数组
    npy_intp newstride;  // 新步幅值
    int i, k, num;  // 循环计数器和辅助变量
    PyObject *ret;  // 返回的 Python 对象
    PyArray_Descr *dtype;  // 数组的数据类型描述符

    // 根据输入的顺序和数组属性，确定新的步幅值
    if (order == NPY_FORTRANORDER || PyArray_ISFORTRAN(arr) || PyArray_NDIM(arr) == 0) {
        newstride = PyArray_ITEMSIZE(arr);  // 对于 Fortran 顺序或者已经是 Fortran 的数组，步幅为元素大小
    }
    else {
        newstride = PyArray_STRIDES(arr)[0] * PyArray_DIMS(arr)[0];  // 否则，步幅为首元素步幅乘以首元素大小
    }

    num = ndmin - nd;  // 计算需要添加的新维度数量
    for (i = 0; i < num; i++) {
        newdims[i] = 1;  // 前 num 个维度大小设为 1
        newstrides[i] = newstride;  // 对应的步幅设为新步幅值
    }
    for (i = num; i < ndmin; i++) {
        k = i - num;
        newdims[i] = PyArray_DIMS(arr)[k];  // 后面的维度大小保持与原数组一致
        newstrides[i] = PyArray_STRIDES(arr)[k];  // 对应的步幅也保持与原数组一致
    }
    dtype = PyArray_DESCR(arr);  // 获取原数组的数据类型描述符
    Py_INCREF(dtype);  // 增加数据类型描述符的引用计数
    ret = PyArray_NewFromDescrAndBase(
            Py_TYPE(arr), dtype,
            ndmin, newdims, newstrides, PyArray_DATA(arr),
            PyArray_FLAGS(arr), (PyObject *)arr, (PyObject *)arr);  // 根据给定参数创建新的数组对象
    Py_DECREF(arr);  // 减少原数组对象的引用计数

    return ret;  // 返回新创建的数组对象
}

#define STRIDING_OK(op, order) \
                ((order) == NPY_ANYORDER || \
                 (order) == NPY_KEEPORDER || \
                 ((order) == NPY_CORDER && PyArray_IS_C_CONTIGUOUS(op)) || \
                 ((order) == NPY_FORTRANORDER && PyArray_IS_F_CONTIGUOUS(op)))

static inline PyObject *
_array_fromobject_generic(
        PyObject *op, PyArray_Descr *in_descr, PyArray_DTypeMeta *in_DType,
        NPY_COPYMODE copy, NPY_ORDER order, npy_bool subok, int ndmin)
{
    PyArrayObject *oparr = NULL, *ret = NULL;  // 输入和输出的数组对象
    PyArray_Descr *oldtype = NULL;  // 旧的数据类型描述符
    int nd, flags = 0;  // 数组维度和标志位

    /* Hold on to `in_descr` as `dtype`, since we may also set it below. */
    Py_XINCREF(in_descr);  // 增加输入的数据类型描述符的引用计数
    PyArray_Descr *dtype = in_descr;  // 设置数组的数据类型描述符

    if (ndmin > NPY_MAXDIMS) {  // 如果指定的最小维度大于最大允许维度
        PyErr_Format(PyExc_ValueError,
                "ndmin bigger than allowable number of dimensions "
                "NPY_MAXDIMS (=%d)", NPY_MAXDIMS);
        goto finish;  // 报错并跳到结束标签
    }
    /* fast exit if simple call */
    }

    if (copy == NPY_COPY_ALWAYS) {  // 如果指定总是复制数据
        flags = NPY_ARRAY_ENSURECOPY;  // 设置标志位，确保复制数据
    }
    else if (copy == NPY_COPY_NEVER) {  // 如果指定从不复制数据
        flags = NPY_ARRAY_ENSURENOCOPY;  // 设置标志位，确保不复制数据
    }
    if (order == NPY_CORDER) {  // 如果指定 C 顺序
        flags |= NPY_ARRAY_C_CONTIGUOUS;  // 设置标志位，确保 C 连续
    }
    else if ((order == NPY_FORTRANORDER)
                 /* order == NPY_ANYORDER && */
                 || (PyArray_Check(op) &&
                     PyArray_ISFORTRAN((PyArrayObject *)op))) {  // 如果指定 Fortran 顺序或者输入数组为 Fortran 连续
        flags |= NPY_ARRAY_F_CONTIGUOUS;  // 设置标志位，确保 Fortran 连续
    }
    if (!subok) {  // 如果不允许创建子类数组
        flags |= NPY_ARRAY_ENSUREARRAY;  // 设置标志位，确保创建数组的副本
    }

    flags |= NPY_ARRAY_FORCECAST;  // 强制进行类型转换

    ret = (PyArrayObject *)PyArray_CheckFromAny_int(
            op, dtype, in_DType, 0, 0, flags, NULL);  // 检查输入对象并返回相应的数组对象

finish:
    // 释放输入数据类型描述符的引用
    Py_XDECREF(dtype);

    return ret;  // 返回最终的数组对象
}
finish:
    // 释放 dtype 所指向的 Python 对象的资源
    Py_XDECREF(dtype);

    // 如果 ret 为 NULL，则返回 NULL
    if (ret == NULL) {
        return NULL;
    }

    // 获取 ret 的维度数
    nd = PyArray_NDIM(ret);
    // 如果 ret 的维度数大于等于 ndmin，则直接返回 ret
    if (nd >= ndmin) {
        return (PyObject *)ret;
    }

    /*
     * 创建一个新的数组，使用相同的数据，但在形状上添加 ones
     * 这里会获取 ret 的引用
     */
    return _prepend_ones(ret, nd, ndmin, order);
}

#undef STRIDING_OK

static PyObject *
array_array(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *op;
    npy_bool subok = NPY_FALSE;
    NPY_COPYMODE copy = NPY_COPY_ALWAYS;
    int ndmin = 0;
    npy_dtype_info dt_info = {NULL, NULL};
    NPY_ORDER order = NPY_KEEPORDER;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    // 如果参数数量不为1或者关键字参数不为空，则解析参数
    if (len_args != 1 || (kwnames != NULL)) {
        if (npy_parse_arguments("array", args, len_args, kwnames,
                "object", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "$copy", &PyArray_CopyConverter, &copy,
                "$order", &PyArray_OrderConverter, &order,
                "$subok", &PyArray_BoolConverter, &subok,
                "$ndmin", &PyArray_PythonPyIntFromInt, &ndmin,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            // 解析参数失败时释放内存并返回 NULL
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        // 如果 like 不是 Py_None，则尝试创建一个新的数组函数
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "array", like, NULL, NULL, args, len_args, kwnames);
            // 如果成功创建，则返回新的数组对象
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        // 参数数量为1且没有关键字参数时，快速路径，直接使用第一个参数作为 op
        op = args[0];
    }

    // 调用通用的对象转换为数组函数，返回转换后的结果
    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, copy, order, subok, ndmin);
    // 释放描述符和数据类型信息的引用
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return res;
}

static PyObject *
array_asarray(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *op;
    NPY_COPYMODE copy = NPY_COPY_IF_NEEDED;
    npy_dtype_info dt_info = {NULL, NULL};
    NPY_ORDER order = NPY_KEEPORDER;
    NPY_DEVICE device = NPY_DEVICE_CPU;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;
    // 检查传入参数个数是否为1，且关键字参数列表是否为空
    if (len_args != 1 || (kwnames != NULL)) {
        // 尝试解析函数参数，如果失败则释放资源并返回空
        if (npy_parse_arguments("asarray", args, len_args, kwnames,
                "a", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "|order", &PyArray_OrderConverter, &order,
                "$device", &PyArray_DeviceConverterOptional, &device,
                "$copy", &PyArray_CopyConverter, &copy,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        // 如果指定了 'like' 参数，则调用函数创建一个基于 'like' 的数组
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "asarray", like, NULL, NULL, args, len_args, kwnames);
            // 如果创建成功，返回创建的数组对象
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        // 如果参数个数为1且没有关键字参数，则直接将第一个参数作为 'op'
        op = args[0];
    }

    // 调用通用的从 Python 对象创建数组的函数
    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, copy, order, NPY_FALSE, 0);
    // 释放 'dt_info' 中的资源
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    // 返回创建的数组对象
    return res;
# 定义名为 array_asanyarray 的静态函数，用于将输入对象转换为 NumPy 数组
static PyObject *
array_asanyarray(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 定义操作对象
    PyObject *op;
    // 指定复制模式，默认为 NPY_COPY_IF_NEEDED
    NPY_COPYMODE copy = NPY_COPY_IF_NEEDED;
    // 定义 dtype 信息结构体，初始为 NULL
    npy_dtype_info dt_info = {NULL, NULL};
    // 指定数组的存储顺序，默认为 NPY_KEEPORDER
    NPY_ORDER order = NPY_KEEPORDER;
    // 指定设备类型，默认为 NPY_DEVICE_CPU
    NPY_DEVICE device = NPY_DEVICE_CPU;
    // 用于指定类似数组对象，默认为 Py_None
    PyObject *like = Py_None;
    // 定义参数解析器
    NPY_PREPARE_ARGPARSER;

    // 如果传入的参数个数不为1，或者存在关键字参数
    if (len_args != 1 || (kwnames != NULL)) {
        // 解析参数，可选参数有 'dtype', 'order', 'device', 'copy', 'like'
        if (npy_parse_arguments("asanyarray", args, len_args, kwnames,
                "a", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "|order", &PyArray_OrderConverter, &order,
                "$device", &PyArray_DeviceConverterOptional, &device,
                "$copy", &PyArray_CopyConverter, &copy,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            // 解析失败时释放 dtype 相关资源并返回 NULL
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        // 如果 like 不是 Py_None，尝试创建类似数组的延迟对象
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "asanyarray", like, NULL, NULL, args, len_args, kwnames);
            // 如果成功创建延迟对象，释放 dtype 相关资源并返回延迟对象
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        // 如果参数个数为1且没有关键字参数，直接取第一个参数作为操作对象
        op = args[0];
    }

    // 调用通用的对象转数组函数，返回结果赋给 res
    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, copy, order, NPY_TRUE, 0);
    // 释放 dtype 相关资源
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    // 返回转换后的数组对象
    return res;
}


``````py
// 定义名为 array_ascontiguousarray 的静态函数，用于将输入对象转换为 NumPy 连续数组
static PyObject *
array_ascontiguousarray(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 定义操作对象
    PyObject *op;
    // 定义 dtype 信息结构体，初始为 NULL
    npy_dtype_info dt_info = {NULL, NULL};
    // 用于指定类似数组对象，默认为 Py_None
    PyObject *like = Py_None;
    // 定义参数解析器
    NPY_PREPARE_ARGPARSER;

    // 如果传入的参数个数不为1，或者存在关键字参数
    if (len_args != 1 || (kwnames != NULL)) {
        // 解析参数，可选参数有 'dtype', 'like'
        if (npy_parse_arguments("ascontiguousarray", args, len_args, kwnames,
                "a", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            // 解析失败时释放 dtype 相关资源并返回 NULL
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        // 如果 like 不是 Py_None，尝试创建类似数组的延迟对象
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "ascontiguousarray", like, NULL, NULL, args, len_args, kwnames);
            // 如果成功创建延迟对象，释放 dtype 相关资源并返回延迟对象
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        // 如果参数个数为1且没有关键字参数，直接取第一个参数作为操作对象
        op = args[0];
    }

    // 调用通用的对象转数组函数，返回结果赋给 res
    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, NPY_COPY_IF_NEEDED, NPY_CORDER, NPY_FALSE,
            1);
    // 释放 dtype 相关资源
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    // 返回转换后的连续数组对象
    return res;
}
/*
PyObject 是 Python 中表示任意对象的 C 结构体指针类型
NPY_UNUSED 宏用于标记未使用的参数，通常用于编译器静默未使用参数的警告
ignored 参数命名未使用，可以在函数体内不使用它们
*/
array_asfortranarray(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    /*
    op 用于存储传递给函数的第一个参数
    npy_dtype_info 结构体用于存储 dtype 相关信息的结构体
    like 变量用于存储传递给函数的关键字参数 "$like" 的值，默认为 Py_None
    NPY_PREPARE_ARGPARSER 宏用于为参数解析做准备
    */
    PyObject *op;
    npy_dtype_info dt_info = {NULL, NULL};
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    /*
    检查参数数量是否为 1 或关键字参数是否为 NULL，若不是则进行参数解析
    */
    if (len_args != 1 || (kwnames != NULL)) {
        /*
        如果参数解析失败，则释放资源并返回 NULL
        */
        if (npy_parse_arguments("asfortranarray", args, len_args, kwnames,
                "a", NULL, &op,
                "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
                "$like", NULL, &like,
                NULL, NULL, NULL) < 0) {
            Py_XDECREF(dt_info.descr);
            Py_XDECREF(dt_info.dtype);
            return NULL;
        }
        /*
        如果 like 不等于 Py_None，则调用 array_implement_c_array_function_creation 函数创建一个延迟对象
        */
        if (like != Py_None) {
            PyObject *deferred = array_implement_c_array_function_creation(
                    "asfortranarray", like, NULL, NULL, args, len_args, kwnames);
            /*
            如果创建成功，则释放资源并返回延迟对象
            */
            if (deferred != Py_NotImplemented) {
                Py_XDECREF(dt_info.descr);
                Py_XDECREF(dt_info.dtype);
                return deferred;
            }
        }
    }
    else {
        /*
        如果参数数量为 1 并且关键字参数为 NULL，则直接将第一个参数赋给 op
        */
        op = args[0];
    }

    /*
    调用 _array_fromobject_generic 函数创建一个新的数组对象
    */
    PyObject *res = _array_fromobject_generic(
            op, dt_info.descr, dt_info.dtype, NPY_COPY_IF_NEEDED, NPY_FORTRANORDER,
            NPY_FALSE, 1);
    Py_XDECREF(dt_info.descr);
    Py_XDECREF(dt_info.dtype);
    return res;
}

/*
PyObject 是 Python 中表示任意对象的 C 结构体指针类型
NPY_UNUSED 宏用于标记未使用的参数，通常用于编译器静默未使用参数的警告
ignored 参数命名未使用，可以在函数体内不使用它们
*/
static PyObject *
array_copyto(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{
    /*
    kwlist 是一个静态的字符指针数组，用于 PyArg_ParseTupleAndKeywords 函数的关键字参数列表
    wheremask_in, dst, src, wheremask 是用于存储传递给函数的参数和中间变量的 PyObject 和 PyArrayObject 类型指针
    casting 是用于存储传递给函数的参数 "casting" 的值，默认为 NPY_SAME_KIND_CASTING
    */
    static char *kwlist[] = {"dst", "src", "casting", "where", NULL};
    PyObject *wheremask_in = NULL;
    PyArrayObject *dst = NULL, *src = NULL, *wheremask = NULL;
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;

    /*
    使用 PyArg_ParseTupleAndKeywords 函数解析传递给函数的参数
    */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O&|O&O:copyto", kwlist,
                &PyArray_Type, &dst,
                &PyArray_Converter, &src,
                &PyArray_CastingConverter, &casting,
                &wheremask_in)) {
        goto fail;
    }

    /*
    如果 wheremask_in 不为 NULL，则创建一个布尔类型的 where mask 数组
    */
    if (wheremask_in != NULL) {
        /* Get the boolean where mask */
        PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            goto fail;
        }
        wheremask = (PyArrayObject *)PyArray_FromAny(wheremask_in,
                                        dtype, 0, 0, 0, NULL);
        if (wheremask == NULL) {
            goto fail;
        }
    }

    /*
    调用 PyArray_AssignArray 函数执行数组赋值操作
    */
    if (PyArray_AssignArray(dst, src, wheremask, casting) < 0) {
        goto fail;
    }

    /*
    释放 src 和 wheremask 资源，并返回 None 对象
    */
    Py_XDECREF(src);
    Py_XDECREF(wheremask);

    Py_RETURN_NONE;

fail:
    /*
    如果发生错误，释放 src 和 wheremask 资源，并返回 NULL
    */
    Py_XDECREF(src);
    Py_XDECREF(wheremask);
    return NULL;
}

/*
PyObject 是 Python 中表示任意对象的 C 结构体指针类型
NPY_UNUSED 宏用于标记未使用的参数，通常用于编译器静默未使用参数的警告
ignored 参数命名未使用，可以在函数体内不使用它们
*/
static PyObject *
array_empty(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    /*
    dt_info 用于存储 dtype 相关信息的结构体
    shape 用于存储创建数组的形状信息的结构体
    order 用于存储数组存储顺序的枚举值，默认为 NPY_CORDER
    is_f_order 用于判断数组是否以 Fortran 顺序存储的布尔值
    ret 是返回的 PyArrayObject 类型指针
    device 用于存储数组存储设备类型的枚举值，默认为 NPY_DEVICE_CPU
    like 变量用于存储传递给函数的关键字参数 "$like" 的值，默认为 Py_None
    NPY_PREPARE_ARGPARSER 宏用于为参数解析做准备
    */
    npy_dtype_info dt_info = {NULL, NULL};
    PyArray_Dims shape = {NULL, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_bool is_f_order;
    PyArrayObject *ret = NULL;
    NPY_DEVICE device = NPY_DEVICE_CPU;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    /*
    此处需要继续添加注释，以解释接下来的代码
    */
    # 解析传入的参数，填充相应的结构体或变量
    if (npy_parse_arguments("empty", args, len_args, kwnames,
            "shape", &PyArray_IntpConverter, &shape,
            "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            "|order", &PyArray_OrderConverter, &order,
            "$device", &PyArray_DeviceConverterOptional, &device,
            "$like", NULL, &like,
            NULL, NULL, NULL) < 0) {
        # 解析参数失败，跳转到失败处理部分
        goto fail;
    }

    # 如果存在 like 参数，尝试创建延迟执行的数组操作函数
    if (like != Py_None) {
        # 调用数组实现的 C 数组函数创建
        PyObject *deferred = array_implement_c_array_function_creation(
                "empty", like, NULL, NULL, args, len_args, kwnames);
        # 如果成功创建了延迟执行对象，则清理资源并返回延迟对象
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(dt_info.descr);  # 释放描述符对象的引用计数
            Py_XDECREF(dt_info.dtype);   # 释放数据类型对象的引用计数
            npy_free_cache_dim_obj(shape);  # 释放形状结构的内存
            return deferred;  # 返回延迟执行对象
        }
    }

    # 根据指定的存储顺序设置是否为 Fortran order
    switch (order) {
        case NPY_CORDER:
            is_f_order = NPY_FALSE;  # C order，非 Fortran order
            break;
        case NPY_FORTRANORDER:
            is_f_order = NPY_TRUE;   # Fortran order
            break;
        default:
            # 存储顺序不合法，设置错误信息并跳转到失败处理部分
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    # 调用 PyArray_Empty_int 函数创建一个 PyArrayObject 对象
    ret = (PyArrayObject *)PyArray_Empty_int(
        shape.len, shape.ptr, dt_info.descr, dt_info.dtype, is_f_order);
fail:
    // 释放描述符对象的引用计数
    Py_XDECREF(dt_info.descr);
    // 释放数据类型对象的引用计数
    Py_XDECREF(dt_info.dtype);
    // 释放缓存的维度对象
    npy_free_cache_dim_obj(shape);
    // 返回已构造的对象
    return (PyObject *)ret;
}

static PyObject *
array_empty_like(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyArrayObject *prototype = NULL;
    npy_dtype_info dt_info = {NULL, NULL};
    NPY_ORDER order = NPY_KEEPORDER;
    PyArrayObject *ret = NULL;
    int subok = 1;
    /* -1 is a special value meaning "not specified" */
    // 初始化形状参数，-1 表示未指定
    PyArray_Dims shape = {NULL, -1};
    NPY_DEVICE device = NPY_DEVICE_CPU;

    NPY_PREPARE_ARGPARSER;

    // 解析传入的参数
    if (npy_parse_arguments("empty_like", args, len_args, kwnames,
            "prototype", &PyArray_Converter, &prototype,
            "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,
            "|order", &PyArray_OrderConverter, &order,
            "|subok", &PyArray_PythonPyIntFromInt, &subok,
            "|shape", &PyArray_OptionalIntpConverter, &shape,
            "$device", &PyArray_DeviceConverterOptional, &device,
            NULL, NULL, NULL) < 0) {
        // 解析失败，跳转到失败处理标签
        goto fail;
    }
    // 若描述符对象不为 NULL，则增加其引用计数
    if (dt_info.descr != NULL) {
        Py_INCREF(dt_info.descr);
    }
    // 创建一个新的数组对象，形状由 shape 指定
    ret = (PyArrayObject *)PyArray_NewLikeArrayWithShape(
            prototype, order, dt_info.descr, dt_info.dtype,
            shape.len, shape.ptr, subok);
    // 释放缓存的维度对象
    npy_free_cache_dim_obj(shape);

fail:
    // 释放原型对象的引用计数
    Py_XDECREF(prototype);
    // 释放数据类型对象的引用计数
    Py_XDECREF(dt_info.dtype);
    // 释放描述符对象的引用计数
    Py_XDECREF(dt_info.descr);
    // 返回已构造的对象
    return (PyObject *)ret;
}

/*
 * This function is needed for supporting Pickles of
 * numpy scalar objects.
 */
static PyObject *
array_scalar(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"dtype", "obj", NULL};
    PyArray_Descr *typecode;
    PyObject *obj = NULL, *tmpobj = NULL;
    int alloc = 0;
    void *dptr;
    PyObject *ret;
    PyObject *base = NULL;

    // 解析参数，包括数据类型和对象
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O:scalar", kwlist,
                &PyArrayDescr_Type, &typecode, &obj)) {
        // 解析失败，返回 NULL 表示错误
        return NULL;
    }
    # 检查是否标记为 NPY_LIST_PICKLE 的类型
    if (PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
        # 如果是 NPY_OBJECT 类型，则警告已废弃
        if (typecode->type_num == NPY_OBJECT) {
            /* Deprecated 2020-11-24, NumPy 1.20 */
            # 如果警告执行失败（返回值小于0），返回空指针
            if (DEPRECATE(
                    "Unpickling a scalar with object dtype is deprecated. "
                    "Object scalars should never be created. If this was a "
                    "properly created pickle, please open a NumPy issue. In "
                    "a best effort this returns the original object.") < 0) {
                return NULL;
            }
            # 增加对象的引用计数并返回对象
            Py_INCREF(obj);
            return obj;
        }
        # 存储完整数组以解压缩
        /* We store the full array to unpack it here: */
        # 如果对象不是 PyArray 的确切类型，报错
        if (!PyArray_CheckExact(obj)) {
            /* We pickle structured voids as arrays currently */
            PyErr_SetString(PyExc_RuntimeError,
                    "Unpickling NPY_LIST_PICKLE (structured void) scalar "
                    "requires an array.  The pickle file may be corrupted?");
            return NULL;
        }
        # 如果对象类型与请求的类型不兼容，报错
        if (!PyArray_EquivTypes(PyArray_DESCR((PyArrayObject *)obj), typecode)) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Pickled array is not compatible with requested scalar "
                    "dtype.  The pickle file may be corrupted?");
            return NULL;
        }
        # 将对象设置为基础对象，并获取其字节指针
        base = obj;
        dptr = PyArray_BYTES((PyArrayObject *)obj);
    }

    # 否则，如果类型标记为 NPY_ITEM_IS_POINTER
    else if (PyDataType_FLAGCHK(typecode, NPY_ITEM_IS_POINTER)) {
        # 如果对象为空，则设置为 Py_None
        if (obj == NULL) {
            obj = Py_None;
        }
        # 设置指针指向对象地址
        dptr = &obj;
    }
    else {
        // 如果对象为 NULL
        if (obj == NULL) {
            // 如果类型码的元素大小为 0，则将其设置为 1
            if (typecode->elsize == 0) {
                typecode->elsize = 1;
            }
            // 分配内存以存储指定大小的数据块
            dptr = PyArray_malloc(typecode->elsize);
            // 如果内存分配失败，则返回内存错误异常
            if (dptr == NULL) {
                return PyErr_NoMemory();
            }
            // 将分配的内存块清零
            memset(dptr, '\0', typecode->elsize);
            // 设置分配标志为真
            alloc = 1;
        }
        else {
            /* 与 Python 2 NumPy pickle 的向后兼容性 */
            // 如果对象是 Unicode 字符串，则尝试将其转换为 Latin-1 编码
            if (PyUnicode_Check(obj)) {
                tmpobj = PyUnicode_AsLatin1String(obj);
                obj = tmpobj;
                // 如果转换失败，则设置更详细的错误消息并返回空值
                if (tmpobj == NULL) {
                    PyErr_SetString(PyExc_ValueError,
                            "Failed to encode Numpy scalar data string to "
                            "latin1,\npickle.load(a, encoding='latin1') is "
                            "assumed if unpickling.");
                    return NULL;
                }
            }
            // 如果对象不是字节对象，则设置类型错误并返回空值
            if (!PyBytes_Check(obj)) {
                PyErr_SetString(PyExc_TypeError,
                        "initializing object must be a bytes object");
                Py_XDECREF(tmpobj);
                return NULL;
            }
            // 如果字节对象大小小于类型码指定的大小，则设置值错误并返回空值
            if (PyBytes_GET_SIZE(obj) < typecode->elsize) {
                PyErr_SetString(PyExc_ValueError,
                        "initialization string is too small");
                Py_XDECREF(tmpobj);
                return NULL;
            }
            // 否则，直接使用对象中的数据指针
            dptr = PyBytes_AS_STRING(obj);
        }
    }
    // 调用 PyArray_Scalar 函数来创建一个新的标量对象
    ret = PyArray_Scalar(dptr, typecode, base);

    /* 释放包含零值的 dptr 内存块 */
    // 如果分配标志为真，则释放 dptr 指向的内存
    if (alloc) {
        PyArray_free(dptr);
    }
    // 释放临时对象的引用
    Py_XDECREF(tmpobj);
    // 返回创建的标量对象或者 NULL
    return ret;
static PyObject *
array_zeros(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    npy_dtype_info dt_info = {NULL, NULL};  // 定义一个包含 dtype 信息的结构体
    PyArray_Dims shape = {NULL, 0};  // 定义一个数组维度结构体
    NPY_ORDER order = NPY_CORDER;  // 设置数组存储顺序，默认为 C 风格顺序
    npy_bool is_f_order = NPY_FALSE;  // 标志位，表示是否使用 Fortran 风格顺序
    PyArrayObject *ret = NULL;  // 定义返回的 NumPy 数组对象指针
    NPY_DEVICE device = NPY_DEVICE_CPU;  // 设置设备类型，默认为 CPU
    PyObject *like = Py_None;  // 用于指定类似对象，默认为 None
    NPY_PREPARE_ARGPARSER;  // 宏定义，准备参数解析器

    if (npy_parse_arguments("zeros", args, len_args, kwnames,
            "shape", &PyArray_IntpConverter, &shape,  // 解析 shape 参数为整型数组
            "|dtype", &PyArray_DTypeOrDescrConverterOptional, &dt_info,  // 解析 dtype 参数，可选
            "|order", &PyArray_OrderConverter, &order,  // 解析 order 参数，指定存储顺序
            "$device", &PyArray_DeviceConverterOptional, &device,  // 解析 device 参数，可选
            "$like", NULL, &like,  // 解析 like 参数，可选
            NULL, NULL, NULL) < 0) {  // 处理参数解析错误
        goto finish;  // 如果解析失败，跳转到 finish 标签处
    }

    if (like != Py_None) {  // 如果 like 参数不为空
        PyObject *deferred = array_implement_c_array_function_creation(
                "zeros", like, NULL, NULL, args, len_args, kwnames);  // 调用函数处理类似对象
        if (deferred != Py_NotImplemented) {  // 如果成功处理类似对象
            Py_XDECREF(dt_info.descr);  // 释放 dtype 信息
            Py_XDECREF(dt_info.dtype);  // 释放 dtype
            npy_free_cache_dim_obj(shape);  // 释放 shape 对象内存
            return deferred;  // 返回处理结果
        }
    }

    switch (order) {  // 根据指定的存储顺序进行处理
        case NPY_CORDER:  // 如果是 C 风格顺序
            is_f_order = NPY_FALSE;  // 设置为非 Fortran 风格
            break;
        case NPY_FORTRANORDER:  // 如果是 Fortran 风格顺序
            is_f_order = NPY_TRUE;  // 设置为 Fortran 风格
            break;
        default:  // 如果不是上述两种顺序
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");  // 报错，只能使用 'C' 或 'F' 顺序
            goto finish;  // 跳转到 finish 标签处
    }

    ret = (PyArrayObject *)PyArray_Zeros_int(
        shape.len, shape.ptr, dt_info.descr, dt_info.dtype, (int) is_f_order);  // 创建全零数组对象

finish:
    npy_free_cache_dim_obj(shape);  // 释放 shape 对象内存
    Py_XDECREF(dt_info.descr);  // 释放 dtype 信息
    Py_XDECREF(dt_info.dtype);  // 释放 dtype
    return (PyObject *)ret;  // 返回创建的数组对象
}

static PyObject *
array_count_nonzero(PyObject *NPY_UNUSED(self), PyObject *const *args, Py_ssize_t len_args)
{
    PyArrayObject *array;  // 定义 NumPy 数组对象指针
    npy_intp count;  // 定义用于存储非零元素个数的变量

    NPY_PREPARE_ARGPARSER;  // 宏定义，准备参数解析器
    if (npy_parse_arguments("count_nonzero", args, len_args, NULL,
            "", PyArray_Converter, &array,  // 解析必需的数组参数
            NULL, NULL, NULL) < 0) {  // 处理参数解析错误
        return NULL;  // 返回空指针表示错误
    }

    count =  PyArray_CountNonzero(array);  // 计算数组中非零元素的个数

    Py_DECREF(array);  // 释放数组对象的引用计数

    if (count == -1) {  // 如果计数返回 -1，表示出错
        return NULL;  // 返回空指针表示错误
    }
    return PyLong_FromSsize_t(count);  // 返回计数结果
}

static PyObject *
array_fromstring(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    char *data;  // 定义字符串数据指针
    Py_ssize_t nin = -1;  // 定义默认的 count 参数值
    char *sep = NULL;  // 定义分隔符字符串指针
    Py_ssize_t s;  // 定义字符串数据的长度
    static char *kwlist[] = {"string", "dtype", "count", "sep", "like", NULL};  // 定义关键字列表
    PyObject *like = Py_None;  // 定义用于指定类似对象的变量，默认为 None
    PyArray_Descr *descr = NULL;  // 定义 NumPy 数据类型描述符指针

    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "s#|O&" NPY_SSIZE_T_PYFMT "s$O:fromstring", kwlist,
                &data, &s, PyArray_DescrConverter, &descr, &nin, &sep, &like)) {
        Py_XDECREF(descr);  // 解析参数失败时释放描述符对象
        return NULL;  // 返回空指针表示错误
    }
    // 如果 `like` 不是 Python 的 None 对象
    if (like != Py_None) {
        // 调用 array_implement_c_array_function_creation 函数创建一个名为 "fromstring" 的函数对象
        PyObject *deferred = array_implement_c_array_function_creation(
                "fromstring", like, args, keywds, NULL, 0, NULL);
        // 如果返回的结果不是 Py_NotImplemented，说明函数创建成功
        if (deferred != Py_NotImplemented) {
            // 释放之前可能存在的描述符对象
            Py_XDECREF(descr);
            // 返回创建的函数对象
            return deferred;
        }
    }

    /* 二进制模式，条件复制自 PyArray_FromString */
    // 如果分隔符 `sep` 为 NULL 或长度为 0
    if (sep == NULL || strlen(sep) == 0) {
        /* Numpy 1.14, 2017-10-19 */
        // 发出警告信息表明二进制模式的 `fromstring` 方法已经被弃用，并建议使用 `frombuffer` 替代
        if (DEPRECATE(
                "The binary mode of fromstring is deprecated, as it behaves "
                "surprisingly on unicode inputs. Use frombuffer instead") < 0) {
            // 释放之前可能存在的描述符对象
            Py_XDECREF(descr);
            // 返回 NULL 表示出错
            return NULL;
        }
    }
    // 调用 PyArray_FromString 函数，将 `data` 转换为数组对象
    return PyArray_FromString(data, (npy_intp)s, descr, (npy_intp)nin, sep);
static PyObject *
array_fromfile(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *keywds)
{
    // 定义变量
    PyObject *file = NULL, *ret = NULL;
    PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
    char *sep = "";
    Py_ssize_t nin = -1;
    static char *kwlist[] = {"file", "dtype", "count", "sep", "offset", "like", NULL};
    PyObject *like = Py_None;
    PyArray_Descr *type = NULL;
    int own;
    npy_off_t orig_pos = 0, offset = 0;
    FILE *fp;

    // 解析参数
    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "O|O&" NPY_SSIZE_T_PYFMT "s" NPY_OFF_T_PYFMT "$O:fromfile", kwlist,
                &file, PyArray_DescrConverter, &type, &nin, &sep, &offset, &like)) {
        Py_XDECREF(type);
        return NULL;
    }

    // 如果有类似对象，则尝试创建类似对象的文件
    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "fromfile", like, args, keywds, NULL, 0, NULL);
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(type);
            return deferred;
        }
    }

    // 将路径类对象转换为文件系统路径
    file = NpyPath_PathlikeToFspath(file);
    if (file == NULL) {
        Py_XDECREF(type);
        return NULL;
    }

    // 检查是否指定了偏移量但同时指定了分隔符，只有在处理二进制文件时允许偏移量参数
    if (offset != 0 && strcmp(sep, "") != 0) {
        PyErr_SetString(PyExc_TypeError, "'offset' argument only permitted for binary files");
        Py_XDECREF(type);
        Py_DECREF(file);
        return NULL;
    }

    // 如果 file 是字节对象或 Unicode 对象，则尝试打开为二进制文件
    if (PyBytes_Check(file) || PyUnicode_Check(file)) {
        Py_SETREF(file, npy_PyFile_OpenFile(file, "rb"));
        if (file == NULL) {
            Py_XDECREF(type);
            return NULL;
        }
        own = 1;
    }
    else {
        own = 0;
    }

    // 复制文件描述符，以便于在可能的异常情况下恢复文件位置
    fp = npy_PyFile_Dup2(file, "rb", &orig_pos);
    if (fp == NULL) {
        Py_DECREF(file);
        Py_XDECREF(type);
        return NULL;
    }

    // 若设置了偏移量，则移动文件指针到指定位置
    if (npy_fseek(fp, offset, SEEK_CUR) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto cleanup;
    }

    // 如果未指定数据类型，则使用默认数据类型创建描述符
    if (type == NULL) {
        type = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    // 从文件中读取数据并创建 NumPy 数组
    ret = PyArray_FromFile(fp, type, (npy_intp) nin, sep);

    /* 如果在调用 PyArray_FromFile 时抛出异常，
     * 我们需要清除异常，并稍后恢复以确保可以正确清理复制的文件描述符。
     */
cleanup:
    PyErr_Fetch(&err_type, &err_value, &err_traceback);

    // 关闭文件和恢复异常状态
    if (npy_PyFile_DupClose2(file, fp, orig_pos) < 0) {
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        goto fail;
    }
    if (own && npy_PyFile_CloseFile(file) < 0) {
        npy_PyErr_ChainExceptions(err_type, err_value, err_traceback);
        goto fail;
    }
    PyErr_Restore(err_type, err_value, err_traceback);

    // 释放资源并返回结果
    Py_DECREF(file);
    return ret;

fail:
    // 处理失败情况
    Py_DECREF(file);
    Py_XDECREF(ret);
    return NULL;
}
    # 初始化一个 PyObject 指针变量 like，设置为 Python 中的 None 对象
    PyObject *like = Py_None;
    # 初始化一个 PyArray_Descr 指针变量 descr，设置为 NULL
    PyArray_Descr *descr = NULL;

    # 使用 PyArg_ParseTupleAndKeywords 解析传入的参数，并处理关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, keywds,
                "OO&|" NPY_SSIZE_T_PYFMT "$O:fromiter", kwlist,
                &iter, PyArray_DescrConverter, &descr, &nin, &like)) {
        // 如果解析参数失败，释放 descr，并返回空指针
        Py_XDECREF(descr);
        return NULL;
    }

    # 如果 like 不是 Python 中的 None 对象
    if (like != Py_None) {
        # 调用 array_implement_c_array_function_creation 函数创建一个延迟执行的对象
        PyObject *deferred = array_implement_c_array_function_creation(
                "fromiter", like, args, keywds, NULL, 0, NULL);
        # 如果创建成功，返回 deferred 对象，释放 descr
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(descr);
            return deferred;
        }
    }

    # 使用 PyArray_FromIter 函数根据传入的迭代器 iter、描述符 descr 和 nin 创建一个新的 PyArray 对象
    return PyArray_FromIter(iter, descr, (npy_intp)nin);
static PyObject *
array_outerproduct(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len_args)
{
    PyObject *b0, *a0;

    NPY_PREPARE_ARGPARSER;  // 准备解析函数参数
    if (npy_parse_arguments("outerproduct", args, len_args, NULL,
            "", NULL, &a0,  // 解析第一个参数
            "", NULL, &b0,  // 解析第二个参数
            NULL, NULL, NULL) < 0) {  // 如果解析失败，返回空指针
        return NULL;
    }

    return PyArray_Return((PyArrayObject *)PyArray_OuterProduct(a0, b0));  // 返回外积计算结果
}
/*
 * 计算两个数组的矩阵乘积。
 * 包括参数解析和异常处理。
 */
static PyObject *
array_matrixproduct(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *v, *a, *o = NULL;
    PyArrayObject *ret;

    NPY_PREPARE_ARGPARSER;  // 准备参数解析器
    if (npy_parse_arguments("dot", args, len_args, kwnames,
            "a", NULL, &a,  // 解析参数 'a'
            "b", NULL, &v,  // 解析参数 'b'
            "|out", NULL, &o,  // 解析可选参数 'out'
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    if (o != NULL) {
        if (o == Py_None) {
            o = NULL;
        }
        else if (!PyArray_Check(o)) {
            PyErr_SetString(PyExc_TypeError, "'out' must be an array");
            return NULL;
        }
    }

    // 调用底层函数计算矩阵乘积并返回结果
    ret = (PyArrayObject *)PyArray_MatrixProduct2(a, v, (PyArrayObject *)o);
    return PyArray_Return(ret);
}


/*
 * 计算向量的共轭点积，使用BLAS进行计算。
 * 对op1和op2进行扁平化处理后进行点积计算。
 */
static PyObject *
array_vdot(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len_args)
{
    int typenum;
    char *ip1, *ip2, *op;
    npy_intp n, stride1, stride2;
    PyObject *op1, *op2;
    npy_intp newdimptr[1] = {-1};
    PyArray_Dims newdims = {newdimptr, 1};
    PyArrayObject *ap1 = NULL, *ap2  = NULL, *ret = NULL;
    PyArray_Descr *type;
    PyArray_DotFunc *vdot;
    NPY_BEGIN_THREADS_DEF;

    NPY_PREPARE_ARGPARSER;  // 准备参数解析器
    if (npy_parse_arguments("vdot", args, len_args, NULL,
            "", NULL, &op1,  // 解析参数
            "", NULL, &op2,  // 解析参数
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    typenum = PyArray_ObjectType(op1, NPY_NOTYPE);  // 获取op1的数组类型
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }
    typenum = PyArray_ObjectType(op2, typenum);  // 根据op1的类型获取op2的类型
    if (typenum == NPY_NOTYPE) {
        return NULL;
    }

    type = PyArray_DescrFromType(typenum);  // 根据类型编号获取描述符
    Py_INCREF(type);
    ap1 = (PyArrayObject *)PyArray_FromAny(op1, type, 0, 0, 0, NULL);  // 将op1转换为数组
    if (ap1 == NULL) {
        Py_DECREF(type);
        goto fail;
    }

    op1 = PyArray_Newshape(ap1, &newdims, NPY_CORDER);  // 对op1进行新维度的重塑
    if (op1 == NULL) {
        Py_DECREF(type);
        goto fail;
    }
    Py_DECREF(ap1);
    ap1 = (PyArrayObject *)op1;

    ap2 = (PyArrayObject *)PyArray_FromAny(op2, type, 0, 0, 0, NULL);  // 将op2转换为数组
    if (ap2 == NULL) {
        goto fail;
    }
    op2 = PyArray_Newshape(ap2, &newdims, NPY_CORDER);  // 对op2进行新维度的重塑
    if (op2 == NULL) {
        goto fail;
    }
    Py_DECREF(ap2);
    ap2 = (PyArrayObject *)op2;

    if (PyArray_DIM(ap2, 0) != PyArray_DIM(ap1, 0)) {  // 检查向量长度是否一致
        PyErr_SetString(PyExc_ValueError,
                "vectors have different lengths");
        goto fail;
    }

    // 创建输出数组用于存储结果
    ret = new_array_for_sum(ap1, ap2, NULL, 0, (npy_intp *)NULL, typenum, NULL);
    if (ret == NULL) {
        goto fail;
    }

    n = PyArray_DIM(ap1, 0);  // 获取向量长度
    stride1 = PyArray_STRIDE(ap1, 0);  // 获取ap1的步长
    stride2 = PyArray_STRIDE(ap2, 0);  // 获取ap2的步长
    ip1 = PyArray_DATA(ap1);  // 获取ap1的数据指针
    ip2 = PyArray_DATA(ap2);  // 获取ap2的数据指针
    op = PyArray_DATA(ret);  // 获取输出数据的指针
    # 根据给定的 typenum 执行不同的操作
    switch (typenum) {
        # 如果 typenum 是 NPY_CFLOAT 类型
        case NPY_CFLOAT:
            # 设置 vdot 函数指针为 CFLOAT_vdot 函数的指针
            vdot = (PyArray_DotFunc *)CFLOAT_vdot;
            break;
        # 如果 typenum 是 NPY_CDOUBLE 类型
        case NPY_CDOUBLE:
            # 设置 vdot 函数指针为 CDOUBLE_vdot 函数的指针
            vdot = (PyArray_DotFunc *)CDOUBLE_vdot;
            break;
        # 如果 typenum 是 NPY_CLONGDOUBLE 类型
        case NPY_CLONGDOUBLE:
            # 设置 vdot 函数指针为 CLONGDOUBLE_vdot 函数的指针
            vdot = (PyArray_DotFunc *)CLONGDOUBLE_vdot;
            break;
        # 如果 typenum 是 NPY_OBJECT 类型
        case NPY_OBJECT:
            # 设置 vdot 函数指针为 OBJECT_vdot 函数的指针
            vdot = (PyArray_DotFunc *)OBJECT_vdot;
            break;
        # 对于其它所有情况
        default:
            # 获取给定数据类型的 dotfunc 函数指针
            vdot = PyDataType_GetArrFuncs(type)->dotfunc;
            # 如果函数指针为空
            if (vdot == NULL) {
                # 设置异常信息为指定的错误字符串
                PyErr_SetString(PyExc_ValueError,
                        "function not available for this data type");
                # 跳转到失败处理的标签
                goto fail;
            }
    }

    # 如果 n 小于 500
    if (n < 500) {
        # 调用 vdot 函数指针执行点积计算，不使用多线程
        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
    }
    else {
        # 启动线程安全区域描述符以进行多线程操作
        NPY_BEGIN_THREADS_DESCR(type);
        # 调用 vdot 函数指针执行点积计算，使用多线程
        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
        # 结束线程安全区域描述符
        NPY_END_THREADS_DESCR(type);
    }

    # 释放 Python 对象的引用计数，避免内存泄漏
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    # 返回处理结果的 Python 对象
    return PyArray_Return(ret);
/*
 * Decrements the reference count of three Python objects and returns NULL.
 * This is typically used when an error occurs in the function.
 */
Py_XDECREF(ap1);
Py_XDECREF(ap2);
Py_XDECREF(ret);
return NULL;
}

/*
 * Parses the arguments for einsum operation from a Python tuple,
 * extracts the subscripts string and operands.
 *
 * Returns:
 *  - Number of operands on success
 *  - -1 on error, with appropriate Python exceptions set
 */
static int
einsum_sub_op_from_str(PyObject *args, PyObject **str_obj, char **subscripts,
                       PyArrayObject **op)
{
    int i, nop;
    PyObject *subscripts_str;

    nop = PyTuple_GET_SIZE(args) - 1;
    if (nop <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand");
        return -1;
    }
    else if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError, "too many operands");
        return -1;
    }

    /* Get the subscripts string */
    subscripts_str = PyTuple_GET_ITEM(args, 0);
    if (PyUnicode_Check(subscripts_str)) {
        *str_obj = PyUnicode_AsASCIIString(subscripts_str);
        if (*str_obj == NULL) {
            return -1;
        }
        subscripts_str = *str_obj;
    }

    *subscripts = PyBytes_AsString(subscripts_str);
    if (*subscripts == NULL) {
        Py_XDECREF(*str_obj);
        *str_obj = NULL;
        return -1;
    }

    /* Set the operands to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
    }

    /* Get the operands */
    for (i = 0; i < nop; ++i) {
        PyObject *obj = PyTuple_GET_ITEM(args, i+1);

        op[i] = (PyArrayObject *)PyArray_FROM_OF(obj, NPY_ARRAY_ENSUREARRAY);
        if (op[i] == NULL) {
            goto fail;
        }
    }

    return nop;

fail:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }

    return -1;
}

/*
 * Converts a Python sequence of subscripts into a C string.
 *
 * Returns:
 *  -1 on error (Python exception set),
 *  - Number of characters placed in `subscripts` on success.
 */
static int
einsum_list_to_subscripts(PyObject *obj, char *subscripts, int subsize)
{
    int ellipsis = 0, subindex = 0;
    npy_intp i, size;
    PyObject *item;

    obj = PySequence_Fast(obj, "the subscripts for each operand must "
                               "be a list or a tuple");
    if (obj == NULL) {
        return -1;
    }
    size = PySequence_Size(obj);
    # 遍历给定的对象，使用索引 i 从 0 到 size-1
    for (i = 0; i < size; ++i) {
        # 获取序列对象 obj 在索引 i 处的元素
        item = PySequence_Fast_GET_ITEM(obj, i);
        
        # 如果当前元素是省略符号 Ellipsis
        /* Ellipsis */
        if (item == Py_Ellipsis) {
            # 如果已经存在省略符号，报错并释放对象
            if (ellipsis) {
                PyErr_SetString(PyExc_ValueError,
                        "each subscripts list may have only one ellipsis");
                Py_DECREF(obj);
                return -1;
            }
            
            # 如果添加省略符号后的索引超过了预设的大小 subsize，报错并释放对象
            if (subindex + 3 >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                Py_DECREF(obj);
                return -1;
            }
            
            # 将省略符号添加到 subscripts 数组中，并标记存在省略符号
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            subscripts[subindex++] = '.';
            ellipsis = 1;
        }
        # 如果当前元素是普通的下标值
        /* Subscript */
        else {
            # 将 Python 中的整数对象转换为 C 的 npy_intp 类型
            npy_intp s = PyArray_PyIntAsIntp(item);
            
            # 如果转换失败，报类型错误并释放对象
            /* Invalid */
            if (error_converting(s)) {
                PyErr_SetString(PyExc_TypeError,
                        "each subscript must be either an integer "
                        "or an ellipsis");
                Py_DECREF(obj);
                return -1;
            }
            
            # 检查 subindex 是否越界，如果超过预设大小 subsize，报错并释放对象
            if (subindex + 1 >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                Py_DECREF(obj);
                return -1;
            }
            
            # 检查整数值 s 的有效性和范围
            npy_bool bad_input = 0;
            
            # 如果 s 小于 0，则标记为无效输入
            if (s < 0) {
                bad_input = 1;
            }
            # 如果 s 在 0 到 25 之间，将其转换为大写字母添加到 subscripts 数组中
            else if (s < 26) {
                subscripts[subindex++] = 'A' + (char)s;
            }
            # 如果 s 在 26 到 51 之间，将其转换为小写字母添加到 subscripts 数组中
            else if (s < 2*26) {
                subscripts[subindex++] = 'a' + (char)s - 26;
            }
            # 如果 s 超出有效范围 [0, 52)，标记为无效输入
            else {
                bad_input = 1;
            }
            
            # 如果标记为无效输入，报错并释放对象
            if (bad_input) {
                PyErr_SetString(PyExc_ValueError,
                        "subscript is not within the valid range [0, 52)");
                Py_DECREF(obj);
                return -1;
            }
        }

    }

    # 释放对象 obj 的引用计数
    Py_DECREF(obj);

    # 返回成功添加到 subscripts 数组中的索引数
    return subindex;
/*
 * Fills in the subscripts, with maximum size subsize, and op,
 * with the values in the tuple 'args'.
 *
 * Returns -1 on error, number of operands placed in op otherwise.
 */
static int
einsum_sub_op_from_lists(PyObject *args,
                char *subscripts, int subsize, PyArrayObject **op)
{
    int subindex = 0;
    npy_intp i, nop;

    // Calculate the number of operands (nop) from the size of the input tuple 'args'
    nop = PyTuple_Size(args) / 2;

    // Check if there are at least two elements in 'args'; otherwise, raise an error
    if (nop == 0) {
        PyErr_SetString(PyExc_ValueError, "must provide at least an "
                        "operand and a subscripts list to einsum");
        return -1;
    }
    // Check if the number of operands exceeds the maximum allowed (NPY_MAXARGS)
    else if (nop >= NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError, "too many operands");
        return -1;
    }

    // Set all operands in the array 'op' to NULL initially
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
    }

    // Iterate through each operand in 'args' and build the subscript string
    for (i = 0; i < nop; ++i) {
        PyObject *obj = PyTuple_GET_ITEM(args, 2*i);
        int n;

        // Insert a comma between subscripts of each operand (except the first one)
        if (i != 0) {
            subscripts[subindex++] = ',';
            if (subindex >= subsize) {
                PyErr_SetString(PyExc_ValueError,
                        "subscripts list is too long");
                goto fail;
            }
        }

        // Convert the operand to a PyArrayObject, ensuring it's an array
        op[i] = (PyArrayObject *) PyArray_FROM_OF(obj, NPY_ARRAY_ENSUREARRAY);
        if (op[i] == NULL) {
            goto fail;
        }

        // Get the subscript list for the current operand and append it to 'subscripts'
        obj = PyTuple_GET_ITEM(args, 2*i+1);
        n = einsum_list_to_subscripts(obj, subscripts + subindex,
                                      subsize - subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    // If provided, add the '->' specifier to the subscript string
    if (PyTuple_Size(args) == 2*nop + 1) {
        PyObject *obj;
        int n;

        // Ensure there's enough space in 'subscripts' for '->'
        if (subindex + 2 >= subsize) {
            PyErr_SetString(PyExc_ValueError,
                    "subscripts list is too long");
            goto fail;
        }
        subscripts[subindex++] = '-';
        subscripts[subindex++] = '>';

        // Append the output subscript list to 'subscripts'
        obj = PyTuple_GET_ITEM(args, 2*nop);
        n = einsum_list_to_subscripts(obj, subscripts + subindex,
                                      subsize - subindex);
        if (n < 0) {
            goto fail;
        }
        subindex += n;
    }

    // Null-terminate the 'subscripts' string
    subscripts[subindex] = '\0';

    return nop;

fail:
    // Clean up on failure: release all allocated operands
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }

    return -1;
}
    // 检查参数元组的大小，确保至少包含一个子集合字符串和至少一个操作数，或者至少一个操作数和其对应的子集合列表
    if (PyTuple_GET_SIZE(args) < 1) {
        // 抛出值错误异常，说明必须指定爱因斯坦求和子集合字符串和至少一个操作数，或者至少一个操作数和其对应的子集合列表
        PyErr_SetString(PyExc_ValueError,
                        "must specify the einstein sum subscripts string "
                        "and at least one operand, or at least one operand "
                        "and its corresponding subscripts list");
        return NULL;
    }
    // 获取参数元组中的第一个参数
    arg0 = PyTuple_GET_ITEM(args, 0);

    /* einsum('i,j', a, b), einsum('i,j->ij', a, b) */
    // 如果第一个参数是字节字符串或者Unicode字符串，则根据字符串解析爱因斯坦求和的子集合和操作
    if (PyBytes_Check(arg0) || PyUnicode_Check(arg0)) {
        nop = einsum_sub_op_from_str(args, &str_obj, &subscripts, op);
    }
    /* einsum(a, [0], b, [1]), einsum(a, [0], b, [1], [0,1]) */
    else {
        // 否则，根据参数列表和子集合列表解析爱因斯坦求和的子集合和操作
        nop = einsum_sub_op_from_lists(args, subscripts_buffer,
                                    sizeof(subscripts_buffer), op);
        subscripts = subscripts_buffer;
    }
    // 如果没有有效的操作数，则跳转到结束位置
    if (nop <= 0) {
        goto finish;
    }

    /* Get the keyword arguments */
    // 获取关键字参数
    if (kwds != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        // 遍历关键字字典
        while (PyDict_Next(kwds, &pos, &key, &value)) {
            char *str = NULL;

            // 释放之前的字符串键对象，并尝试将键转换为ASCII字符串
            Py_XDECREF(str_key_obj);
            str_key_obj = PyUnicode_AsASCIIString(key);
            // 如果成功转换，则将key更新为ASCII字符串对象
            if (str_key_obj != NULL) {
                key = str_key_obj;
            }

            // 获取键的C风格字符串表示
            str = PyBytes_AsString(key);

            // 如果字符串为空，清除错误并设置类型错误异常，然后跳转到结束位置
            if (str == NULL) {
                PyErr_Clear();
                PyErr_SetString(PyExc_TypeError, "invalid keyword");
                goto finish;
            }

            // 根据关键字字符串的值进行不同的处理
            if (strcmp(str,"out") == 0) {
                // 如果关键字是"out"，则检查值是否为数组，是则赋给out
                if (PyArray_Check(value)) {
                    out = (PyArrayObject *)value;
                }
                else {
                    // 否则设置类型错误异常，要求"out"参数必须是一个数组
                    PyErr_SetString(PyExc_TypeError,
                                "keyword parameter out must be an "
                                "array for einsum");
                    goto finish;
                }
            }
            else if (strcmp(str,"order") == 0) {
                // 如果关键字是"order"，则转换其值为数组排序方式
                if (!PyArray_OrderConverter(value, &order)) {
                    goto finish;
                }
            }
            else if (strcmp(str,"casting") == 0) {
                // 如果关键字是"casting"，则转换其值为数组类型转换方式
                if (!PyArray_CastingConverter(value, &casting)) {
                    goto finish;
                }
            }
            else if (strcmp(str,"dtype") == 0) {
                // 如果关键字是"dtype"，则转换其值为数组数据类型描述符
                if (!PyArray_DescrConverter2(value, &dtype)) {
                    goto finish;
                }
            }
            else {
                // 否则设置类型错误异常，说明关键字不是有效的einsum关键字
                PyErr_Format(PyExc_TypeError,
                            "'%s' is an invalid keyword for einsum",
                            str);
                goto finish;
            }
        }
    }

    // 执行爱因斯坦求和操作，返回结果对象
    ret = (PyObject *)PyArray_EinsteinSum(subscripts, nop, op, dtype,
                                        order, casting, out);

    // 如果返回结果不为空且没有提供输出数组，则尝试将结果转换为标量
    if (ret != NULL && out == NULL) {
        ret = PyArray_Return((PyArrayObject *)ret);
    }
finish:
    for (i = 0; i < nop; ++i) {
        // 释放 Python 对象数组中第 i 个对象的引用计数
        Py_XDECREF(op[i]);
    }
    // 释放 dtype 对象的引用计数
    Py_XDECREF(dtype);
    // 释放 str_obj 对象的引用计数
    Py_XDECREF(str_obj);
    // 释放 str_key_obj 对象的引用计数
    Py_XDECREF(str_key_obj);
    /* out is a borrowed reference */

    // 返回函数结果
    return ret;
}


static PyObject *
array_correlate(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *shape, *a0;
    int mode = 0;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("correlate", args, len_args, kwnames,
            // 解析函数参数，获取数组 a0 和 shape
            "a", NULL, &a0,
            "v", NULL, &shape,
            // 可选参数 mode，指定相关运算的模式
            "|mode", &PyArray_CorrelatemodeConverter, &mode,
            NULL, NULL, NULL) < 0) {
        // 参数解析失败，返回 NULL
        return NULL;
    }
    // 调用 PyArray_Correlate 函数进行相关运算，并返回其结果
    return PyArray_Correlate(a0, shape, mode);
}

static PyObject*
array_correlate2(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *shape, *a0;
    int mode = 0;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("correlate2", args, len_args, kwnames,
            // 解析函数参数，获取数组 a0 和 shape
            "a", NULL, &a0,
            "v", NULL, &shape,
            // 可选参数 mode，指定相关运算的模式
            "|mode", &PyArray_CorrelatemodeConverter, &mode,
            NULL, NULL, NULL) < 0) {
        // 参数解析失败，返回 NULL
        return NULL;
    }
    // 调用 PyArray_Correlate2 函数进行相关运算，并返回其结果
    return PyArray_Correlate2(a0, shape, mode);
}

static PyObject *
array_arange(PyObject *NPY_UNUSED(ignored),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *o_start = NULL, *o_stop = NULL, *o_step = NULL, *range=NULL;
    PyArray_Descr *typecode = NULL;
    NPY_DEVICE device = NPY_DEVICE_CPU;
    PyObject *like = Py_None;
    NPY_PREPARE_ARGPARSER;

    if (npy_parse_arguments("arange", args, len_args, kwnames,
            // 解析函数参数，可选参数 start, stop, step, dtype, device, like
            "|start", NULL, &o_start,
            "|stop", NULL, &o_stop,
            "|step", NULL, &o_step,
            "|dtype", &PyArray_DescrConverter2, &typecode,
            "$device", &PyArray_DeviceConverterOptional, &device,
            "$like", NULL, &like,
            NULL, NULL, NULL) < 0) {
        // 参数解析失败，释放 typecode 对象的引用计数并返回 NULL
        Py_XDECREF(typecode);
        return NULL;
    }
    // 如果指定了 like 参数，尝试使用数组函数创建函数
    if (like != Py_None) {
        PyObject *deferred = array_implement_c_array_function_creation(
                "arange", like, NULL, NULL, args, len_args, kwnames);
        // 如果成功创建，返回创建结果
        if (deferred != Py_NotImplemented) {
            Py_XDECREF(typecode);
            return deferred;
        }
    }

    // 如果没有指定 stop 参数且没有其他位置参数，抛出类型错误
    if (o_stop == NULL) {
        if (len_args == 0){
            PyErr_SetString(PyExc_TypeError,
                "arange() requires stop to be specified.");
            Py_XDECREF(typecode);
            return NULL;
        }
    }
    else if (o_start == NULL) {
        // 如果没有指定 start 参数但指定了 stop 参数，则调整参数位置
        o_start = o_stop;
        o_stop = NULL;
    }

    // 调用 PyArray_ArangeObj 函数生成 range 对象
    range = PyArray_ArangeObj(o_start, o_stop, o_step, typecode);
    Py_XDECREF(typecode);

    // 返回生成的 range 对象
    return range;
}

/*NUMPY_API
 *
 * Included at the very first so not auto-grabbed and thus not labeled.
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCVersion(void)
{
    // 返回当前 NumPy 库的 ABI 版本号
    return (unsigned int)NPY_ABI_VERSION;
}
/*NUMPY_API
 * Returns the built-in (at compilation time) C API version
 */
NPY_NO_EXPORT unsigned int
PyArray_GetNDArrayCFeatureVersion(void)
{
    // 返回编译时内置的 NumPy C API 版本号
    return (unsigned int)NPY_API_VERSION;
}

static PyObject *
array__get_ndarray_c_version(
        PyObject *NPY_UNUSED(dummy), PyObject *NPY_UNUSED(arg))
{
    // 返回 NumPy ndarray C 版本的 Python 封装函数
    return PyLong_FromLong( (long) PyArray_GetNDArrayCVersion() );
}

/*NUMPY_API
*/
NPY_NO_EXPORT int
PyArray_GetEndianness(void)
{
    const union {
        npy_uint32 i;
        char c[4];
    } bint = {0x01020304};

    // 检测当前平台的字节序，并返回对应的常量
    if (bint.c[0] == 1) {
        return NPY_CPU_BIG;    // 大端字节序
    }
    else if (bint.c[0] == 4) {
        return NPY_CPU_LITTLE; // 小端字节序
    }
    else {
        return NPY_CPU_UNKNOWN_ENDIAN; // 未知字节序
    }
}

static PyObject *
array__reconstruct(PyObject *NPY_UNUSED(dummy), PyObject *args)
{

    PyObject *ret;
    PyTypeObject *subtype;
    PyArray_Dims shape = {NULL, 0};
    PyArray_Descr *dtype = NULL;

    evil_global_disable_warn_O4O8_flag = 1;

    // 解析参数，重建 ndarray 对象
    if (!PyArg_ParseTuple(args, "O!O&O&:_reconstruct",
                &PyType_Type, &subtype,
                PyArray_IntpConverter, &shape,
                PyArray_DescrConverter, &dtype)) {
        goto fail;
    }
    // 检查 subtype 是否为 ndarray 的子类型
    if (!PyType_IsSubtype(subtype, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError,
                "_reconstruct: First argument must be a sub-type of ndarray");
        goto fail;
    }
    // 根据给定的 shape 和 dtype 创建新的 ndarray 对象
    ret = PyArray_NewFromDescr(subtype, dtype,
            (int)shape.len, shape.ptr, NULL, NULL, 0, NULL);
    npy_free_cache_dim_obj(shape);

    evil_global_disable_warn_O4O8_flag = 0;

    return ret;

fail:
    evil_global_disable_warn_O4O8_flag = 0;

    Py_XDECREF(dtype);
    npy_free_cache_dim_obj(shape);
    return NULL;
}

static PyObject *
array_set_datetimeparse_function(PyObject *NPY_UNUSED(self),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    // 设置 datetime 解析函数，但已被移除，因此直接抛出 RuntimeError
    PyErr_SetString(PyExc_RuntimeError, "This function has been removed");
    return NULL;
}

/*
 * inner loop with constant size memcpy arguments
 * this allows the compiler to replace function calls while still handling the
 * alignment requirements of the platform.
 */
#define INNER_WHERE_LOOP(size) \
    do { \
        npy_intp i; \
        for (i = 0; i < n; i++) { \
            if (*csrc) { \
                memcpy(dst, xsrc, size); \
            } \
            else { \
                memcpy(dst, ysrc, size); \
            } \
            dst += size; \
            xsrc += xstride; \
            ysrc += ystride; \
            csrc += cstride; \
        } \
    } while(0)

/*NUMPY_API
 * Where
 */
NPY_NO_EXPORT PyObject *
PyArray_Where(PyObject *condition, PyObject *x, PyObject *y)
{
    PyArrayObject *arr = NULL, *ax = NULL, *ay = NULL;
    PyObject *ret = NULL;
    PyArray_Descr *common_dt = NULL;

    // 将 condition 转换为 PyArrayObject
    arr = (PyArrayObject *)PyArray_FROM_O(condition);
    if (arr == NULL) {
        return NULL;
    }
    // 如果 x 和 y 都为 NULL，则返回 condition 的非零元素索引
    if ((x == NULL) && (y == NULL)) {
        ret = PyArray_Nonzero(arr);
        Py_DECREF(arr);
        return ret;
    }
    // 检查 x 和 y 是否都为空，如果是，则释放 arr 对象，设置错误信息并返回空指针
    if ((x == NULL) || (y == NULL)) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError,
                "either both or neither of x and y should be given");
        return NULL;
    }

    // 定义 x 和 y 的类型信息结构体，初始设置函数指针为 NULL
    NPY_cast_info x_cast_info = {.func = NULL};
    NPY_cast_info y_cast_info = {.func = NULL};

    // 将 x 转换为 PyArrayObject 类型，如果失败则跳转到错误处理标签 fail
    ax = (PyArrayObject*)PyArray_FROM_O(x);
    if (ax == NULL) {
        goto fail;
    }
    // 将 y 转换为 PyArrayObject 类型，如果失败则跳转到错误处理标签 fail
    ay = (PyArrayObject*)PyArray_FROM_O(y);
    if (ay == NULL) {
        goto fail;
    }
    // 标记 x 和 y 如果是 Python 标量，则标记为临时数组
    npy_mark_tmp_array_if_pyscalar(x, ax, NULL);
    npy_mark_tmp_array_if_pyscalar(y, ay, NULL);

    // 定义迭代器的 flags
    npy_uint32 flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED |
                        NPY_ITER_REFS_OK | NPY_ITER_ZEROSIZE_OK;
    // 定义操作输入对象数组
    PyArrayObject * op_in[4] = {
        NULL, arr, ax, ay
    };
    // 定义操作输入对象的 flags 数组
    npy_uint32 op_flags[4] = {
        NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NO_SUBTYPE,
        NPY_ITER_READONLY,
        NPY_ITER_READONLY | NPY_ITER_ALIGNED,
        NPY_ITER_READONLY | NPY_ITER_ALIGNED
    };

    // 获取 x 和 y 的共同数据类型
    common_dt = PyArray_ResultType(2, &op_in[2], 0, NULL);
    if (common_dt == NULL) {
        goto fail;
    }
    // 获取共同数据类型的元素大小
    npy_intp itemsize = common_dt->elsize;

    // 如果 x 和 y 没有引用，且元素大小适中，则使用简单的循环进行快速复制
    // 否则，在循环中逐项处理类型转换
    PyArray_Descr *x_dt, *y_dt;
    int trivial_copy_loop = !PyDataType_REFCHK(common_dt) &&
            ((itemsize == 16) || (itemsize == 8) || (itemsize == 4) ||
             (itemsize == 2) || (itemsize == 1));
    if (trivial_copy_loop) {
        x_dt = common_dt;
        y_dt = common_dt;
    }
    else {
        x_dt = PyArray_DESCR(op_in[2]);
        y_dt = PyArray_DESCR(op_in[3]);
    }
    /* `PyArray_DescrFromType` cannot fail for simple builtin types: */
    // 定义操作的数据类型数组
    PyArray_Descr * op_dt[4] = {common_dt, PyArray_DescrFromType(NPY_BOOL), x_dt, y_dt};

    // 定义迭代器对象和线程管理相关的变量
    NpyIter * iter;
    NPY_BEGIN_THREADS_DEF;

    // 创建多输入迭代器对象
    iter =  NpyIter_MultiNew(
            4, op_in, flags, NPY_KEEPORDER, NPY_UNSAFE_CASTING,
            op_flags, op_dt);
    // 释放多余的 bool 类型的数据描述符对象
    Py_DECREF(op_dt[1]);
    if (iter == NULL) {
        goto fail;
    }

    /* Get the result from the iterator object array */
    // 从迭代器对象数组中获取结果对象
    ret = (PyObject*)NpyIter_GetOperandArray(iter)[0];
    // 获取结果对象的数据描述符
    PyArray_Descr *ret_dt = PyArray_DESCR((PyArrayObject *)ret);

    // 定义数据传输的标志
    NPY_ARRAYMETHOD_FLAGS transfer_flags = 0;

    // 定义 x 和 y 的步长数组
    npy_intp x_strides[2] = {x_dt->elsize, itemsize};
    npy_intp y_strides[2] = {y_dt->elsize, itemsize};
    npy_intp one = 1;
    // 如果没有简单的复制循环（trivial_copy_loop 为假），则执行以下操作
    if (!trivial_copy_loop) {
        // 迭代器具有 NPY_ITER_ALIGNED 标志，因此无需检查输入数组的对齐方式。
        // 对输入数组 op_in[2] 的数据类型和返回数据类型 ret_dt 进行转换函数的获取，
        // 0 表示不进行对齐检查，将结果保存在 x_cast_info 和 transfer_flags 中。
        if (PyArray_GetDTypeTransferFunction(
                    1, x_strides[0], x_strides[1],
                    PyArray_DESCR(op_in[2]), ret_dt, 0,
                    &x_cast_info, &transfer_flags) != NPY_SUCCEED) {
            // 如果获取转换函数失败，则跳转到失败处理标签。
            goto fail;
        }
        // 对输入数组 op_in[3] 的数据类型和返回数据类型 ret_dt 进行转换函数的获取，
        // 0 表示不进行对齐检查，将结果保存在 y_cast_info 和 transfer_flags 中。
        if (PyArray_GetDTypeTransferFunction(
                    1, y_strides[0], y_strides[1],
                    PyArray_DESCR(op_in[3]), ret_dt, 0,
                    &y_cast_info, &transfer_flags) != NPY_SUCCEED) {
            // 如果获取转换函数失败，则跳转到失败处理标签。
            goto fail;
        }
    }

    // 合并 transfer_flags 和迭代器的传输标志，保存在 transfer_flags 中。
    transfer_flags = PyArrayMethod_COMBINED_FLAGS(
        transfer_flags, NpyIter_GetTransferFlags(iter));

    // 如果 transfer_flags 不需要 Python API 支持，则启动多线程处理。
    if (!(transfer_flags & NPY_METH_REQUIRES_PYAPI)) {
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
    }
    # 检查迭代器的大小是否不为零
    if (NpyIter_GetIterSize(iter) != 0) {
        # 获取迭代器的下一个迭代函数
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        # 获取内部循环大小的指针
        npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        # 获取数据指针数组
        char **dataptrarray = NpyIter_GetDataPtrArray(iter);
        # 获取内部步长数组
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);

        # 迭代执行循环操作
        do {
            # 获取当前内部循环的大小
            npy_intp n = (*innersizeptr);
            # 获取目标数据指针
            char *dst = dataptrarray[0];
            # 获取C源数据指针
            char *csrc = dataptrarray[1];
            # 获取X源数据指针
            char *xsrc = dataptrarray[2];
            # 获取Y源数据指针
            char *ysrc = dataptrarray[3];

            // 迭代器可能会改变这些指针，
            // 所以每次迭代都需要更新它们
            # 获取C数据的步长
            npy_intp cstride = strides[1];
            # 获取X数据的步长
            npy_intp xstride = strides[2];
            # 获取Y数据的步长
            npy_intp ystride = strides[3];

            /* 常量大小，因此编译器会替换成memcpy */
            # 如果是平凡的复制循环并且itemsize为16
            if (trivial_copy_loop && itemsize == 16) {
                INNER_WHERE_LOOP(16);
            }
            # 如果是平凡的复制循环并且itemsize为8
            else if (trivial_copy_loop && itemsize == 8) {
                INNER_WHERE_LOOP(8);
            }
            # 如果是平凡的复制循环并且itemsize为4
            else if (trivial_copy_loop && itemsize == 4) {
                INNER_WHERE_LOOP(4);
            }
            # 如果是平凡的复制循环并且itemsize为2
            else if (trivial_copy_loop && itemsize == 2) {
                INNER_WHERE_LOOP(2);
            }
            # 如果是平凡的复制循环并且itemsize为1
            else if (trivial_copy_loop && itemsize == 1) {
                INNER_WHERE_LOOP(1);
            }
            else {
                # 否则进行普通的循环操作
                npy_intp i;
                for (i = 0; i < n; i++) {
                    # 如果C源数据非零
                    if (*csrc) {
                        # 创建参数数组args，指向X源数据和目标数据
                        char *args[2] = {xsrc, dst};

                        # 调用X类型转换函数
                        if (x_cast_info.func(
                                &x_cast_info.context, args, &one,
                                x_strides, x_cast_info.auxdata) < 0) {
                            # 转换失败则跳转到失败处理标签
                            goto fail;
                        }
                    }
                    else {
                        # 否则创建参数数组args，指向Y源数据和目标数据
                        char *args[2] = {ysrc, dst};

                        # 调用Y类型转换函数
                        if (y_cast_info.func(
                                &y_cast_info.context, args, &one,
                                y_strides, y_cast_info.auxdata) < 0) {
                            # 转换失败则跳转到失败处理标签
                            goto fail;
                        }
                    }
                    # 更新目标数据指针位置
                    dst += itemsize;
                    # 更新X源数据指针位置
                    xsrc += xstride;
                    # 更新Y源数据指针位置
                    ysrc += ystride;
                    # 更新C源数据指针位置
                    csrc += cstride;
                }
            }
        } while (iternext(iter));  // 继续迭代直到结束

    }

    # 结束多线程区域
    NPY_END_THREADS;

    # 增加返回对象的引用计数
    Py_INCREF(ret);
    # 减少数组对象的引用计数
    Py_DECREF(arr);
    # 减少ax对象的引用计数
    Py_DECREF(ax);
    # 减少ay对象的引用计数
    Py_DECREF(ay);
    # 减少common_dt对象的引用计数
    Py_DECREF(common_dt);
    # 释放X类型转换信息的内存
    NPY_cast_info_xfree(&x_cast_info);
    # 释放Y类型转换信息的内存
    NPY_cast_info_xfree(&y_cast_info);

    # 如果释放迭代器失败，则释放返回对象并返回空指针
    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        Py_DECREF(ret);
        return NULL;
    }

    # 返回结果对象
    return ret;
fail:
    // 释放 arr 对象的引用计数
    Py_DECREF(arr);
    // 释放 ax 对象的引用计数
    Py_XDECREF(ax);
    // 释放 ay 对象的引用计数
    Py_XDECREF(ay);
    // 释放 common_dt 对象的引用计数
    Py_XDECREF(common_dt);
    // 释放 x_cast_info 结构体占用的内存
    NPY_cast_info_xfree(&x_cast_info);
    // 释放 y_cast_info 结构体占用的内存
    NPY_cast_info_xfree(&y_cast_info);
    // 返回 NULL 指针，表示函数执行失败
    return NULL;
}

#undef INNER_WHERE_LOOP

static PyObject *
array_where(PyObject *NPY_UNUSED(ignored), PyObject *const *args, Py_ssize_t len_args)
{
    // 声明函数变量
    PyObject *obj = NULL, *x = NULL, *y = NULL;

    // 解析传入的参数
    NPY_PREPARE_ARGPARSER;
    // 如果参数解析失败，则返回 NULL 指针
    if (npy_parse_arguments("where", args, len_args, NULL,
            "", NULL, &obj,
            "|x", NULL, &x,
            "|y", NULL, &y,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 调用 PyArray_Where 函数处理传入的参数并返回结果
    return PyArray_Where(obj, x, y);
}

static PyObject *
array_lexsort(PyObject *NPY_UNUSED(ignored), PyObject *const *args, Py_ssize_t len_args,
                             PyObject *kwnames)
{
    // 初始化 axis 变量为默认值 -1
    int axis = -1;
    // 声明 keys 对象
    PyObject *obj;

    // 解析传入的参数
    NPY_PREPARE_ARGPARSER;
    // 如果参数解析失败，则返回 NULL 指针
    if (npy_parse_arguments("lexsort", args, len_args, kwnames,
            "keys", NULL, &obj,
            "|axis", PyArray_PythonPyIntFromInt, &axis,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    // 调用 PyArray_LexSort 函数进行排序操作，并返回结果
    return PyArray_Return((PyArrayObject *)PyArray_LexSort(obj, axis));
}

static PyObject *
array_can_cast_safely(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 声明变量
    PyObject *from_obj = NULL;
    PyArray_Descr *d1 = NULL;
    PyArray_Descr *d2 = NULL;
    int ret;
    PyObject *retobj = NULL;
    // 初始化 casting 变量为 NPY_SAFE_CASTING
    NPY_CASTING casting = NPY_SAFE_CASTING;

    // 解析传入的参数
    NPY_PREPARE_ARGPARSER;
    // 如果参数解析失败，则跳转到 finish 标签
    if (npy_parse_arguments("can_cast", args, len_args, kwnames,
            "from_", NULL, &from_obj,
            "to", &PyArray_DescrConverter2, &d2,
            "|casting", &PyArray_CastingConverter, &casting,
            NULL, NULL, NULL) < 0) {
        goto finish;
    }
    // 如果 d2 是 NULL 指针，则设置类型错误并跳转到 finish 标签
    if (d2 == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "did not understand one of the types; 'None' not accepted");
        goto finish;
    }

    // 如果 from_obj 是数组对象，则使用 PyArray_CanCastArrayTo 函数进行类型转换判断
    if (PyArray_Check(from_obj)) {
        ret = PyArray_CanCastArrayTo((PyArrayObject *)from_obj, d2, casting);
    }
    else if (PyArray_IsScalar(from_obj, Generic)) {
        /*
         * TODO: `PyArray_IsScalar` 在新的数据类型中不应该被要求。
         *       weak-promotion 分支实际上与 dtype 分支相同。
         */
        if (get_npy_promotion_state() == NPY_USE_WEAK_PROMOTION) {
            // 获取标量对象的描述符
            PyObject *descr = PyObject_GetAttr(from_obj, npy_interned_str.dtype);
            if (descr == NULL) {
                goto finish;  // 如果获取失败，直接结束
            }
            if (!PyArray_DescrCheck(descr)) {
                Py_DECREF(descr);
                PyErr_SetString(PyExc_TypeError,
                    "numpy_scalar.dtype did not return a dtype instance.");
                goto finish;  // 如果不是有效的描述符类型，设置错误并结束
            }
            // 检查是否可以将描述符类型转换为目标类型
            ret = PyArray_CanCastTypeTo((PyArray_Descr *)descr, d2, casting);
            Py_DECREF(descr);
        }
        else {
            /* 需要将标量对象转换为数组对象，以便考虑旧的基于值的逻辑 */
            PyArrayObject *arr;
            arr = (PyArrayObject *)PyArray_FROM_O(from_obj);
            if (arr == NULL) {
                goto finish;  // 如果转换失败，直接结束
            }
            // 检查是否可以将数组对象转换为目标类型
            ret = PyArray_CanCastArrayTo(arr, d2, casting);
            Py_DECREF(arr);
        }
    }
    else if (PyArray_IsPythonNumber(from_obj)) {
        PyErr_SetString(PyExc_TypeError,
                "can_cast() does not support Python ints, floats, and "
                "complex because the result used to depend on the value.\n"
                "This change was part of adopting NEP 50, we may "
                "explicitly allow them again in the future.");
        goto finish;  // 不支持 Python 的整数、浮点数和复数类型，设置错误并结束
    }
    /* 否则使用 CanCastTypeTo */
    else {
        // 尝试将输入对象转换为描述符类型
        if (!PyArray_DescrConverter2(from_obj, &d1) || d1 == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "did not understand one of the types; 'None' not accepted");
            goto finish;  // 如果转换失败或者返回空描述符，设置错误并结束
        }
        // 检查是否可以将描述符类型转换为目标类型
        ret = PyArray_CanCastTypeTo(d1, d2, casting);
    }

    // 根据返回值设置结果对象为 True 或 False
    retobj = ret ? Py_True : Py_False;
    Py_INCREF(retobj);  // 增加结果对象的引用计数

 finish:
    Py_XDECREF(d1);  // 释放描述符对象 d1 的引用
    Py_XDECREF(d2);  // 释放目标描述符对象 d2 的引用
    return retobj;  // 返回结果对象
static PyObject *
array_promote_types(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len_args)
{
    PyArray_Descr *d1 = NULL;
    PyArray_Descr *d2 = NULL;
    PyObject *ret = NULL;

    NPY_PREPARE_ARGPARSER;
    // 解析函数参数，将解析结果保存在d1和d2中
    if (npy_parse_arguments("promote_types", args, len_args, NULL,
            "", PyArray_DescrConverter2, &d1,
            "", PyArray_DescrConverter2, &d2,
            NULL, NULL, NULL) < 0) {
        goto finish;
    }

    // 如果d1或d2为空，则设置错误并跳转到结束标签
    if (d1 == NULL || d2 == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "did not understand one of the types");
        goto finish;
    }

    // 调用NumPy C API函数PyArray_PromoteTypes进行类型提升，并将结果保存在ret中
    ret = (PyObject *)PyArray_PromoteTypes(d1, d2);

 finish:
    // 释放d1和d2的引用计数
    Py_XDECREF(d1);
    Py_XDECREF(d2);
    return ret;
}

static PyObject *
array_min_scalar_type(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *array_in = NULL;
    PyArrayObject *array;
    PyObject *ret = NULL;

    // 解析输入参数，期望一个参数，将结果保存在array_in中
    if (!PyArg_ParseTuple(args, "O:min_scalar_type", &array_in)) {
        return NULL;
    }

    // 将array_in转换为PyArrayObject对象
    array = (PyArrayObject *)PyArray_FROM_O(array_in);
    if (array == NULL) {
        return NULL;
    }

    // 调用NumPy C API函数PyArray_MinScalarType获取数组的最小标量类型，并将结果保存在ret中
    ret = (PyObject *)PyArray_MinScalarType(array);
    // 释放array的引用计数
    Py_DECREF(array);
    return ret;
}

static PyObject *
array_result_type(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len)
{
    npy_intp i, narr = 0, ndtypes = 0;
    PyArrayObject **arr = NULL;
    PyArray_Descr **dtypes = NULL;
    PyObject *ret = NULL;

    // 如果没有输入参数，则设置错误并跳转到结束标签
    if (len == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "at least one array or dtype is required");
        goto finish;
    }

    // 分配内存以存储PyArrayObject指针和PyArray_Descr指针数组
    arr = PyArray_malloc(2 * len * sizeof(void *));
    if (arr == NULL) {
        return PyErr_NoMemory();
    }
    dtypes = (PyArray_Descr**)&arr[len];

    for (i = 0; i < len; ++i) {
        PyObject *obj = args[i];
        // 如果参数是数组对象，则增加其引用计数并存储在arr数组中
        if (PyArray_Check(obj)) {
            Py_INCREF(obj);
            arr[narr] = (PyArrayObject *)obj;
            ++narr;
        }
        // 如果参数是标量对象或Python数值，则转换为PyArrayObject对象存储在arr数组中
        else if (PyArray_IsScalar(obj, Generic) ||
                                    PyArray_IsPythonNumber(obj)) {
            arr[narr] = (PyArrayObject *)PyArray_FROM_O(obj);
            if (arr[narr] == NULL) {
                goto finish;
            }
            /*
             * 如果参数是Python标量，则标记数组为临时数组
             * （此时不需要实际的DType，这在ResultType函数内部会处理）
             */
            npy_mark_tmp_array_if_pyscalar(obj, arr[narr], NULL);
            ++narr;
        }
        // 如果参数是描述符对象，则通过PyArray_DescrConverter函数进行转换并存储在dtypes数组中
        else {
            if (!PyArray_DescrConverter(obj, &dtypes[ndtypes])) {
                goto finish;
            }
            ++ndtypes;
        }
    }

    // 调用NumPy C API函数PyArray_ResultType计算数组的结果类型，并将结果保存在ret中
    ret = (PyObject *)PyArray_ResultType(narr, arr, ndtypes, dtypes);

finish:
    // 释放arr数组中所有PyArrayObject对象的引用计数
    for (i = 0; i < narr; ++i) {
        Py_DECREF(arr[i]);
    }
    // 释放dtypes数组中所有PyArray_Descr对象的引用计数
    for (i = 0; i < ndtypes; ++i) {
        Py_DECREF(dtypes[i]);
    }
    // 释放arr数组的内存
    PyArray_free(arr);
    return ret;
}

static PyObject *
/*
 * Returns datetime metadata as a tuple for a given NumPy dtype object.
 * Parses arguments to extract and convert the dtype object.
 * If parsing fails, returns NULL.
 */
PyObject *array_datetime_data(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyArray_Descr *dtype;
    PyArray_DatetimeMetaData *meta;

    // 解析参数，获取并转换 NumPy dtype 对象
    if (!PyArg_ParseTuple(args, "O&:datetime_data",
                PyArray_DescrConverter, &dtype)) {
        return NULL;
    }

    // 获取 dtype 对象的日期时间元数据
    meta = get_datetime_metadata_from_dtype(dtype);
    if (meta == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    // 将日期时间元数据转换为元组并返回
    PyObject *res = convert_datetime_metadata_to_tuple(meta);
    Py_DECREF(dtype);
    return res;
}


/*
 * Converts a Python object to a TrimMode enum value.
 * Checks if the object is a single-character Unicode string and assigns the corresponding TrimMode.
 * If the object does not match any expected characters, raises a TypeError.
 */
static int
trimmode_converter(PyObject *obj, TrimMode *trim)
{
    if (!PyUnicode_Check(obj) || PyUnicode_GetLength(obj) != 1) {
        // 如果对象不是单字符 Unicode 字符串，或长度不为 1，则跳转到错误处理部分
        goto error;
    }
    const char *trimstr = PyUnicode_AsUTF8AndSize(obj, NULL);

    if (trimstr != NULL) {
        // 根据字符内容设置对应的 TrimMode
        if (trimstr[0] == 'k') {
            *trim = TrimMode_None;
        }
        else if (trimstr[0] == '.') {
            *trim = TrimMode_Zeros;
        }
        else if (trimstr[0] ==  '0') {
            *trim = TrimMode_LeaveOneZero;
        }
        else if (trimstr[0] ==  '-') {
            *trim = TrimMode_DptZeros;
        }
        else {
            // 字符不在预期范围内，抛出类型错误异常
            goto error;
        }
    }
    return NPY_SUCCEED;

error:
    PyErr_Format(PyExc_TypeError,
            "if supplied, trim must be 'k', '.', '0' or '-' found `%100S`",
            obj);
    return NPY_FAIL;
}


/*
 * Implements the Dragon4 algorithm in scientific mode for floating-point scalars.
 * Parses and validates arguments related to formatting the output.
 * Returns a Python object representing the formatted value.
 */
static PyObject *
dragon4_scientific(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;
    int precision=-1, pad_left=-1, exp_digits=-1, min_digits=-1;
    DigitMode digit_mode;
    TrimMode trim = TrimMode_None;
    int sign=0, unique=1;
    NPY_PREPARE_ARGPARSER;

    // 解析并验证参数，设置 Dragon4 算法的相关参数
    if (npy_parse_arguments("dragon4_scientific", args, len_args, kwnames,
            "x", NULL , &obj,
            "|precision", &PyArray_PythonPyIntFromInt, &precision,
            "|unique", &PyArray_PythonPyIntFromInt, &unique,
            "|sign", &PyArray_PythonPyIntFromInt, &sign,
            "|trim", &trimmode_converter, &trim,
            "|pad_left", &PyArray_PythonPyIntFromInt, &pad_left,
            "|exp_digits", &PyArray_PythonPyIntFromInt, &exp_digits,
            "|min_digits", &PyArray_PythonPyIntFromInt, &min_digits,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    digit_mode = unique ? DigitMode_Unique : DigitMode_Exact;

    // 在非唯一模式下，如果未提供精度参数，抛出类型错误异常
    if (unique == 0 && precision < 0) {
        PyErr_SetString(PyExc_TypeError,
            "in non-unique mode `precision` must be supplied");
        return NULL;
    }

    // 调用 Dragon4 算法，返回格式化后的结果对象
    return Dragon4_Scientific(obj, digit_mode, precision, min_digits, sign, trim,
                              pad_left, exp_digits);
}
/*
 * 使用 Dragon4 算法以位置模式打印浮点数标量。
 * 参见 `np.format_float_positional` 的文档字符串以获取参数描述。
 * 不同之处在于 pad_left、pad_right、precision 的值为 -1 表示未指定，等效于 `None`。
 */
static PyObject *
dragon4_positional(PyObject *NPY_UNUSED(dummy),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj;
    int precision=-1, pad_left=-1, pad_right=-1, min_digits=-1;
    CutoffMode cutoff_mode;
    DigitMode digit_mode;
    TrimMode trim = TrimMode_None;
    int sign=0, unique=1, fractional=0;
    NPY_PREPARE_ARGPARSER;

    // 解析函数参数
    if (npy_parse_arguments("dragon4_positional", args, len_args, kwnames,
            "x", NULL , &obj,
            "|precision", &PyArray_PythonPyIntFromInt, &precision,
            "|unique", &PyArray_PythonPyIntFromInt, &unique,
            "|fractional", &PyArray_PythonPyIntFromInt, &fractional,
            "|sign", &PyArray_PythonPyIntFromInt, &sign,
            "|trim", &trimmode_converter, &trim,
            "|pad_left", &PyArray_PythonPyIntFromInt, &pad_left,
            "|pad_right", &PyArray_PythonPyIntFromInt, &pad_right,
            "|min_digits", &PyArray_PythonPyIntFromInt, &min_digits,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 根据 unique 的值确定 digit_mode
    digit_mode = unique ? DigitMode_Unique : DigitMode_Exact;
    // 根据 fractional 的值确定 cutoff_mode
    cutoff_mode = fractional ? CutoffMode_FractionLength :
                               CutoffMode_TotalLength;

    // 在非 unique 模式下，如果未提供 precision，报错
    if (unique == 0 && precision < 0) {
        PyErr_SetString(PyExc_TypeError,
            "in non-unique mode `precision` must be supplied");
        return NULL;
    }

    // 调用 Dragon4_Positional 函数处理参数并返回结果
    return Dragon4_Positional(obj, digit_mode, cutoff_mode, precision,
                              min_digits, sign, trim, pad_left, pad_right);
}

/*
 * 格式化长浮点数对象。
 * 要求精度值 precision，参见 `obj` 的类型为 LongDouble。
 */
static PyObject *
format_longfloat(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    unsigned int precision;
    static char *kwlist[] = {"x", "precision", NULL};

    // 解析参数列表
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OI:format_longfloat", kwlist,
                &obj, &precision)) {
        return NULL;
    }

    // 如果 obj 不是 LongDouble 类型，则报错
    if (!PyArray_IsScalar(obj, LongDouble)) {
        PyErr_SetString(PyExc_TypeError,
                "not a longfloat");
        return NULL;
    }

    // 调用 Dragon4_Scientific 函数处理参数并返回结果
    return Dragon4_Scientific(obj, DigitMode_Unique, precision, -1, 0,
                              TrimMode_LeaveOneZero, -1, -1);
}

/*
 * 检查数组是否是用户定义的字符串数据类型。
 * 如果是，则返回 1；否则设置错误并返回 0。
 */
static int _is_user_defined_string_array(PyArrayObject* array)
{
    // 省略部分代码
}
    # 检查给定的 NumPy 数组是否具有用户定义的数据类型描述符
    if (NPY_DT_is_user_defined(PyArray_DESCR(array))) {
        # 获取数组元素的标量类型对象
        PyTypeObject* scalar_type = NPY_DTYPE(PyArray_DESCR(array))->scalar_type;
        # 检查标量类型是否是 bytes 或者 unicode 类型的子类型
        if (PyType_IsSubtype(scalar_type, &PyBytes_Type) ||
            PyType_IsSubtype(scalar_type, &PyUnicode_Type)) {
            # 如果是，则返回 1 表示允许字符串比较
            return 1;
        }
        else {
            # 如果不是，则设置类型错误异常，说明只有标量类型是 str 或 bytes 的子类型才能进行字符串比较
            PyErr_SetString(
                PyExc_TypeError,
                "string comparisons are only allowed for dtypes with a "
                "scalar type that is a subtype of str or bytes.");
            return 0;
        }
    }
    else {
        # 如果数组的数据类型不是用户定义的，则设置类型错误异常，表示非字符串数组上的字符串操作
        PyErr_SetString(
            PyExc_TypeError,
            "string operation on non-string array");
        return 0;
    }
/*
 * The only purpose of this function is that it allows the "rstrip".
 * From my (@seberg's) perspective, this function should be deprecated
 * and I do not think it matters if it is not particularly fast.
 */
static PyObject *
compare_chararrays(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *kwds)
{
    PyObject *array;
    PyObject *other;
    PyArrayObject *newarr, *newoth;
    int cmp_op;
    npy_bool rstrip;
    char *cmp_str;
    Py_ssize_t strlength;
    PyObject *res = NULL;
    static char msg[] = "comparison must be '==', '!=', '<', '>', '<=', '>='";
    static char *kwlist[] = {"a1", "a2", "cmp", "rstrip", NULL};

    // 解析传入的参数和关键字参数，设置关键字列表
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOs#O&:compare_chararrays",
                kwlist,
                &array, &other, &cmp_str, &strlength,
                PyArray_BoolConverter, &rstrip)) {
        return NULL;
    }

    // 检查比较操作符字符串长度的有效性
    if (strlength < 1 || strlength > 2) {
        goto err;
    }

    // 根据比较操作符字符串长度的不同，确定比较操作符类型
    if (strlength > 1) {
        if (cmp_str[1] != '=') {
            goto err;
        }
        if (cmp_str[0] == '=') {
            cmp_op = Py_EQ;
        }
        else if (cmp_str[0] == '!') {
            cmp_op = Py_NE;
        }
        else if (cmp_str[0] == '<') {
            cmp_op = Py_LE;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GE;
        }
        else {
            goto err;
        }
    }
    else {
        if (cmp_str[0] == '<') {
            cmp_op = Py_LT;
        }
        else if (cmp_str[0] == '>') {
            cmp_op = Py_GT;
        }
        else {
            goto err;
        }
    }

    // 将输入参数转换为 NumPy 数组对象
    newarr = (PyArrayObject *)PyArray_FROM_O(array);
    if (newarr == NULL) {
        return NULL;
    }
    newoth = (PyArrayObject *)PyArray_FROM_O(other);
    if (newoth == NULL) {
        Py_DECREF(newarr);
        return NULL;
    }

    // 如果输入的数组都是字符串数组，则调用字符串比较函数
    if (PyArray_ISSTRING(newarr) && PyArray_ISSTRING(newoth)) {
        res = _umath_strings_richcompare(newarr, newoth, cmp_op, rstrip != 0);
    }
    else {
        // 如果输入的数组不是字符串数组，则抛出类型错误异常
        PyErr_SetString(PyExc_TypeError,
                "comparison of non-string arrays");
        Py_DECREF(newarr);
        Py_DECREF(newoth);
        return NULL;
    }

    // 释放数组对象的引用计数
    Py_DECREF(newarr);
    Py_DECREF(newoth);
    return res;

 err:
    // 如果出现错误，设置值错误异常并返回 NULL
    PyErr_SetString(PyExc_ValueError, msg);
    return NULL;
}

static PyObject *
_vec_string_with_args(PyArrayObject* char_array, PyArray_Descr* type,
                      PyObject* method, PyObject* args)
{
    PyObject* broadcast_args[NPY_MAXARGS];
    PyArrayMultiIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;
    Py_ssize_t i, n, nargs;

    // 计算参数序列的长度，并检查其有效性
    nargs = PySequence_Size(args) + 1;
    if (nargs == -1 || nargs > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "len(args) must be < %d", NPY_MAXARGS - 1);
        Py_DECREF(type);
        goto err;
    }

    // 将第一个参数设置为输入的字符数组对象
    broadcast_args[0] = (PyObject*)char_array;
    for (i = 1; i < nargs; i++) {
        // 从参数元组中获取第 i-1 个参数对象
        PyObject* item = PySequence_GetItem(args, i-1);
        // 如果获取失败，释放之前申请的 type 对象并跳转到错误处理标签 err
        if (item == NULL) {
            Py_DECREF(type);
            goto err;
        }
        // 将获取的参数对象保存到 broadcast_args 数组中
        broadcast_args[i] = item;
        // 释放参数对象的引用计数
        Py_DECREF(item);
    }

    // 根据 broadcast_args 数组创建多迭代器对象 in_iter
    in_iter = (PyArrayMultiIterObject*)PyArray_MultiIterFromObjects
        (broadcast_args, nargs, 0);
    // 如果创建失败，释放之前申请的 type 对象并跳转到错误处理标签 err
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }
    // 获取 in_iter 中的迭代器数量
    n = in_iter->numiter;

    // 根据 in_iter 的维度和类型创建新的 PyArrayObject 对象 result
    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(in_iter->nd,
            in_iter->dimensions, type);
    // 如果创建失败，跳转到错误处理标签 err
    if (result == NULL) {
        goto err;
    }

    // 根据 result 创建输出迭代器对象 out_iter
    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    // 如果创建失败，跳转到错误处理标签 err
    if (out_iter == NULL) {
        goto err;
    }

    // 循环处理多迭代器 in_iter 中的元素
    while (PyArray_MultiIter_NOTDONE(in_iter)) {
        PyObject* item_result;
        // 创建一个包含 n 个元素的元组 args_tuple
        PyObject* args_tuple = PyTuple_New(n);
        // 如果创建失败，跳转到错误处理标签 err
        if (args_tuple == NULL) {
            goto err;
        }

        // 遍历 in_iter 的迭代器并将每个元素转换为标量对象存入 args_tuple 中
        for (i = 0; i < n; i++) {
            PyArrayIterObject* it = in_iter->iters[i];
            PyObject* arg = PyArray_ToScalar(PyArray_ITER_DATA(it), it->ao);
            // 如果转换失败，释放 args_tuple 并跳转到错误处理标签 err
            if (arg == NULL) {
                Py_DECREF(args_tuple);
                goto err;
            }
            /* Steals ref to arg */
            // 将 arg 添加到 args_tuple 中，注意 PyTuple_SetItem 会接管 arg 的引用计数
            PyTuple_SetItem(args_tuple, i, arg);
        }

        // 调用 method 对象，并传入 args_tuple 中的参数进行计算
        item_result = PyObject_CallObject(method, args_tuple);
        // 释放 args_tuple 对象的引用计数
        Py_DECREF(args_tuple);
        // 如果调用失败，跳转到错误处理标签 err
        if (item_result == NULL) {
            goto err;
        }

        // 将 item_result 存入 result 的当前迭代位置
        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            // 释放 item_result 对象
            Py_DECREF(item_result);
            // 设置错误信息并跳转到错误处理标签 err
            PyErr_SetString( PyExc_TypeError,
                    "result array type does not match underlying function");
            goto err;
        }
        // 释放 item_result 对象的引用计数
        Py_DECREF(item_result);

        // 移动 in_iter 和 out_iter 到下一个元素
        PyArray_MultiIter_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    // 释放 in_iter 和 out_iter 对象的引用计数
    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    // 返回 result 对象作为函数执行的结果
    return (PyObject*)result;

 err:
    // 释放 in_iter、out_iter 和 result 对象的引用计数
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    // 返回 0 表示函数执行失败
    return 0;
static PyObject *
_vec_string_no_args(PyArrayObject* char_array,
                                   PyArray_Descr* type, PyObject* method)
{
    /*
     * This is a faster version of _vec_string_args to use when there
     * are no additional arguments to the string method.  This doesn't
     * require a broadcast iterator (and broadcast iterators don't work
     * with 1 argument anyway).
     */
    // 初始化输入迭代器和输出结果对象的迭代器为NULL
    PyArrayIterObject* in_iter = NULL;
    PyArrayObject* result = NULL;
    PyArrayIterObject* out_iter = NULL;

    // 创建输入迭代器，如果失败则跳转到错误处理
    in_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)char_array);
    if (in_iter == NULL) {
        Py_DECREF(type);
        goto err;
    }

    // 根据给定的描述符创建结果数组对象，如果失败则跳转到错误处理
    result = (PyArrayObject*)PyArray_SimpleNewFromDescr(
            PyArray_NDIM(char_array), PyArray_DIMS(char_array), type);
    if (result == NULL) {
        goto err;
    }

    // 创建输出结果对象的迭代器，如果失败则跳转到错误处理
    out_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)result);
    if (out_iter == NULL) {
        goto err;
    }

    // 迭代处理输入数组中的每个元素
    while (PyArray_ITER_NOTDONE(in_iter)) {
        PyObject* item_result;
        PyObject* item = PyArray_ToScalar(in_iter->dataptr, in_iter->ao);
        if (item == NULL) {
            goto err;
        }

        // 调用给定的方法处理当前元素，如果失败则跳转到错误处理
        item_result = PyObject_CallFunctionObjArgs(method, item, NULL);
        Py_DECREF(item);
        if (item_result == NULL) {
            goto err;
        }

        // 将处理结果放入输出数组中的当前位置，如果类型不匹配则设置错误信息并跳转到错误处理
        if (PyArray_SETITEM(result, PyArray_ITER_DATA(out_iter), item_result)) {
            Py_DECREF(item_result);
            PyErr_SetString( PyExc_TypeError,
                "result array type does not match underlying function");
            goto err;
        }
        Py_DECREF(item_result);

        // 移动到下一个输入和输出数组的位置
        PyArray_ITER_NEXT(in_iter);
        PyArray_ITER_NEXT(out_iter);
    }

    // 释放迭代器对象
    Py_DECREF(in_iter);
    Py_DECREF(out_iter);

    // 返回处理结果数组对象
    return (PyObject*)result;

 err:
    // 出错时释放所有可能已分配的资源，并返回0
    Py_XDECREF(in_iter);
    Py_XDECREF(out_iter);
    Py_XDECREF(result);

    return 0;
}
    else {
        // 如果条件不成立，则执行以下操作
        if (_is_user_defined_string_array(char_array)) {
            // 检查 char_array 是否为用户定义的字符串数组，若是，则执行以下操作
            PyTypeObject* scalar_type =
                NPY_DTYPE(PyArray_DESCR(char_array))->scalar_type;
            // 获取 char_array 对应的数据类型描述符的标量类型
            method = PyObject_GetAttr((PyObject*)scalar_type, method_name);
            // 获取标量类型对应的方法对象
        }
        else {
            // 如果不是用户定义的字符串数组，则执行以下操作
            Py_DECREF(type);
            // 减少对 type 的引用计数
            goto err;
            // 跳转到错误处理部分
        }
    }
    if (method == NULL) {
        // 如果 method 为 NULL，则执行以下操作
        Py_DECREF(type);
        // 减少对 type 的引用计数
        goto err;
        // 跳转到错误处理部分
    }

    if (args_seq == NULL
            || (PySequence_Check(args_seq) && PySequence_Size(args_seq) == 0)) {
        // 如果 args_seq 为 NULL 或者是一个空序列，则执行以下操作
        result = _vec_string_no_args(char_array, type, method);
        // 调用 _vec_string_no_args 函数，处理没有参数的情况
    }
    else if (PySequence_Check(args_seq)) {
        // 如果 args_seq 是一个序列，则执行以下操作
        result = _vec_string_with_args(char_array, type, method, args_seq);
        // 调用 _vec_string_with_args 函数，处理带有参数的情况
    }
    else {
        // 如果 args_seq 不是一个序列，则执行以下操作
        Py_DECREF(type);
        // 减少对 type 的引用计数
        PyErr_SetString(PyExc_TypeError,
                "'args' must be a sequence of arguments");
        // 设置一个类型错误的异常字符串
        goto err;
        // 跳转到错误处理部分
    }
    if (result == NULL) {
        // 如果 result 为 NULL，则执行以下操作
        goto err;
        // 跳转到错误处理部分
    }

    Py_DECREF(char_array);
    // 减少对 char_array 的引用计数
    Py_DECREF(method);
    // 减少对 method 的引用计数

    return (PyObject*)result;
    // 返回 result 对象的 PyObject 指针

 err:
    // 错误处理部分
    Py_XDECREF(char_array);
    // 减少或清除 char_array 的引用计数
    Py_XDECREF(method);
    // 减少或清除 method 的引用计数

    return 0;
    // 返回 0，表示函数执行失败
static PyObject *
array_shares_memory_impl(PyObject *args, PyObject *kwds, Py_ssize_t default_max_work,
                         int raise_exceptions)
{
    PyObject * self_obj = NULL;  // 定义指向第一个数组对象的 Python 对象指针
    PyObject * other_obj = NULL;  // 定义指向第二个数组对象的 Python 对象指针
    PyArrayObject * self = NULL;  // 定义指向第一个数组对象的 NumPy 数组对象指针
    PyArrayObject * other = NULL;  // 定义指向第二个数组对象的 NumPy 数组对象指针
    PyObject *max_work_obj = NULL;  // 定义指向 max_work 参数的 Python 对象指针
    static char *kwlist[] = {"self", "other", "max_work", NULL};  // 定义关键字列表，用于参数解析

    mem_overlap_t result;  // 定义存储内存重叠检测结果的变量
    Py_ssize_t max_work;  // 定义存储最大工作单元数的变量
    NPY_BEGIN_THREADS_DEF;  // 定义 NumPy 线程处理开始的宏

    max_work = default_max_work;  // 将默认最大工作单元数赋值给 max_work

    // 解析传入的参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:shares_memory_impl", kwlist,
                                     &self_obj, &other_obj, &max_work_obj)) {
        return NULL;  // 参数解析失败时返回空指针
    }

    // 检查第一个对象是否为 NumPy 数组
    if (PyArray_Check(self_obj)) {
        self = (PyArrayObject*)self_obj;  // 将 self_obj 转换为 PyArrayObject 指针
        Py_INCREF(self);  // 增加 self 的引用计数
    }
    else {
        /* Use FromAny to enable checking overlap for objects exposing array
           interfaces etc. */
        self = (PyArrayObject*)PyArray_FROM_O(self_obj);  // 使用 PyArray_FROM_O 转换对象
        if (self == NULL) {
            goto fail;  // 转换失败时跳转到 fail 标签处理错误
        }
    }

    // 检查第二个对象是否为 NumPy 数组
    if (PyArray_Check(other_obj)) {
        other = (PyArrayObject*)other_obj;  // 将 other_obj 转换为 PyArrayObject 指针
        Py_INCREF(other);  // 增加 other 的引用计数
    }
    else {
        other = (PyArrayObject*)PyArray_FROM_O(other_obj);  // 使用 PyArray_FROM_O 转换对象
        if (other == NULL) {
            goto fail;  // 转换失败时跳转到 fail 标签处理错误
        }
    }

    // 处理 max_work_obj 参数
    if (max_work_obj == NULL || max_work_obj == Py_None) {
        /* noop */  // 如果 max_work_obj 为空或者为 None，则什么都不做
    }
    else if (PyLong_Check(max_work_obj)) {
        max_work = PyLong_AsSsize_t(max_work_obj);  // 将 max_work_obj 转换为 Py_ssize_t 类型
        if (PyErr_Occurred()) {
            goto fail;  // 转换过程中出错时跳转到 fail 标签处理错误
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "max_work must be an integer");  // 设置错误信息
        goto fail;  // 跳转到 fail 标签处理错误
    }

    // 检查 max_work 的值是否合法
    if (max_work < -2) {
        PyErr_SetString(PyExc_ValueError, "Invalid value for max_work");  // 设置错误信息
        goto fail;  // 跳转到 fail 标签处理错误
    }

    NPY_BEGIN_THREADS;  // 开始 NumPy 线程处理

    // 调用 solve_may_share_memory 函数进行内存重叠检测
    result = solve_may_share_memory(self, other, max_work);

    NPY_END_THREADS;  // 结束 NumPy 线程处理

    Py_XDECREF(self);  // 释放 self 对象的引用
    Py_XDECREF(other);  // 释放 other 对象的引用

    // 根据检测结果返回相应的 Python 对象
    if (result == MEM_OVERLAP_NO) {
        Py_RETURN_FALSE;  // 返回 Python 中的 False
    }
    else if (result == MEM_OVERLAP_YES) {
        Py_RETURN_TRUE;  // 返回 Python 中的 True
    }
    else if (result == MEM_OVERLAP_OVERFLOW) {
        if (raise_exceptions) {
            PyErr_SetString(PyExc_OverflowError,
                            "Integer overflow in computing overlap");  // 设置错误信息
            return NULL;  // 返回空指针
        }
        else {
            /* Don't know, so say yes */
            Py_RETURN_TRUE;  // 不确定时，默认返回 Python 中的 True
        }
    }
    else if (result == MEM_OVERLAP_TOO_HARD) {
        if (raise_exceptions) {
            PyErr_SetString(npy_static_pydata.TooHardError,
                            "Exceeded max_work");  // 设置错误信息
            return NULL;  // 返回空指针
        }
        else {
            /* Don't know, so say yes */
            Py_RETURN_TRUE;  // 不确定时，默认返回 Python 中的 True
        }
    }
    else {
        /* Doesn't happen usually */
        PyErr_SetString(PyExc_RuntimeError,
                        "Error in computing overlap");  // 设置错误信息
        return NULL;  // 返回空指针
    }

fail:
    Py_XDECREF(self);  // 处理错误时释放 self 对象的引用
    Py_XDECREF(other);  // 处理错误时释放 other 对象的引用
    return NULL;  // 返回空指针
}
static PyObjectc
// 定义函数 `array_shares_memory`，接收三个参数：无用参数 `ignored`，`args` 和 `kwds`
{
    // 调用 `array_shares_memory_impl` 函数，使用参数 `args` 和 `kwds`，指定共享内存的精确性标志为 `NPY_MAY_SHARE_EXACT`，并返回结果
    return array_shares_memory_impl(args, kwds, NPY_MAY_SHARE_EXACT, 1);
}


// 定义静态函数 `array_may_share_memory`，接收三个参数：无用参数 `ignored`，`args` 和 `kwds`
{
    // 调用 `array_shares_memory_impl` 函数，使用参数 `args` 和 `kwds`，指定共享内存的边界标志为 `NPY_MAY_SHARE_BOUNDS`，并返回结果
    return array_shares_memory_impl(args, kwds, NPY_MAY_SHARE_BOUNDS, 0);
}


// 定义静态函数 `normalize_axis_index`，接收四个参数：无用参数 `self`，`args` 数组，`len_args` 参数数组长度，`kwnames` 关键字参数
{
    // 声明整型变量 `axis` 和 `ndim`
    int axis;
    int ndim;
    // 声明对象 `msg_prefix`，初始化为 `Py_None`
    PyObject *msg_prefix = Py_None;
    // 定义宏 `NPY_PREPARE_ARGPARSER`，准备解析函数参数

    // 解析参数，如果解析失败则返回 `NULL`
    if (npy_parse_arguments("normalize_axis_index", args, len_args, kwnames,
            "axis", &PyArray_PythonPyIntFromInt, &axis,
            "ndim", &PyArray_PythonPyIntFromInt, &ndim,
            "|msg_prefix", NULL, &msg_prefix,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    // 检查和调整 `axis`，如果失败则返回 `NULL`
    if (check_and_adjust_axis_msg(&axis, ndim, msg_prefix) < 0) {
        return NULL;
    }

    // 返回 `axis` 的 `PyLong` 对象
    return PyLong_FromLong(axis);
}


// 定义静态函数 `_set_numpy_warn_if_no_mem_policy`，接收两个参数：无用参数 `self` 和 `arg`
{
    // 判断 `arg` 是否为真值，失败则返回 `NULL`
    int res = PyObject_IsTrue(arg);
    if (res < 0) {
        return NULL;
    }
    // 保存 `npy_thread_unsafe_state.warn_if_no_mem_policy` 的旧值
    int old_value = npy_thread_unsafe_state.warn_if_no_mem_policy;
    // 设置 `npy_thread_unsafe_state.warn_if_no_mem_policy` 为 `res` 的值
    npy_thread_unsafe_state.warn_if_no_mem_policy = res;
    // 如果旧值为真，返回 `True`；否则返回 `False`
    if (old_value) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}


// 定义静态函数 `_reload_guard`，接收两个参数：无用参数 `self` 和 `args`
{
    // 如果不是 PyPy 版本，并且当前解释器不是主解释器状态
    if (PyThreadState_Get()->interp != PyInterpreterState_Main()) {
        // 发出警告，说明 NumPy 是从 Python 的子解释器中导入的，而 NumPy 并不完全支持子解释器
        // 设置 `npy_thread_unsafe_state.reload_guard_initialized` 为 1
        // 返回 `None`
        if (PyErr_WarnEx(PyExc_UserWarning,
                "NumPy was imported from a Python sub-interpreter but "
                "NumPy does not properly support sub-interpreters. "
                "This will likely work for most users but might cause hard to "
                "track down issues or subtle bugs. "
                "A common user of the rare sub-interpreter feature is wsgi "
                "which also allows single-interpreter mode.\n"
                "Improvements in the case of bugs are welcome, but is not "
                "on the NumPy roadmap, and full support may require "
                "significant effort to achieve.", 2) < 0) {
            return NULL;
        }
        // 返回 `None`
        npy_thread_unsafe_state.reload_guard_initialized = 1;
        Py_RETURN_NONE;
    }

    // 如果 `npy_thread_unsafe_state.reload_guard_initialized` 已经初始化
    if (npy_thread_unsafe_state.reload_guard_initialized) {
        // 发出警告，说明 NumPy 模块被重新加载（第二次导入）
        // 返回 `None`
        if (PyErr_WarnEx(PyExc_UserWarning,
                "The NumPy module was reloaded (imported a second time). "
                "This can in some cases result in small but subtle issues "
                "and is discouraged.", 2) < 0) {
            return NULL;
        }
    }
    // 设置 `npy_thread_unsafe_state.reload_guard_initialized` 为 1
    // 返回 `None`
    npy_thread_unsafe_state.reload_guard_initialized = 1;
    Py_RETURN_NONE;
}


// 定义静态结构体 `array_module_methods`，包含一系列方法定义
    {"_get_implementing_args",
        (PyCFunction)array__get_implementing_args,
        METH_VARARGS, NULL},
    # 注册一个名为 "_get_implementing_args" 的方法，对应的 C 函数是 array__get_implementing_args，接受位置参数，并且没有关键字参数
    {"_get_ndarray_c_version",
        (PyCFunction)array__get_ndarray_c_version,
        METH_NOARGS, NULL},
    # 注册一个名为 "_get_ndarray_c_version" 的方法，对应的 C 函数是 array__get_ndarray_c_version，不接受任何参数
    {"_reconstruct",
        (PyCFunction)array__reconstruct,
        METH_VARARGS, NULL},
    # 注册一个名为 "_reconstruct" 的方法，对应的 C 函数是 array__reconstruct，接受位置参数，并且没有关键字参数
    {"set_datetimeparse_function",
        (PyCFunction)array_set_datetimeparse_function,
        METH_VARARGS|METH_KEYWORDS, NULL},
    # 注册一个名为 "set_datetimeparse_function" 的方法，对应的 C 函数是 array_set_datetimeparse_function，接受位置参数和关键字参数
    {"set_typeDict",
        (PyCFunction)array_set_typeDict,
        METH_VARARGS, NULL},
    # 注册一个名为 "set_typeDict" 的方法，对应的 C 函数是 array_set_typeDict，接受位置参数，并且没有关键字参数
    {"array",
        (PyCFunction)array_array,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "array" 的方法，对应的 C 函数是 array_array，接受快速调用和关键字参数
    {"asarray",
        (PyCFunction)array_asarray,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "asarray" 的方法，对应的 C 函数是 array_asarray，接受快速调用和关键字参数
    {"asanyarray",
        (PyCFunction)array_asanyarray,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "asanyarray" 的方法，对应的 C 函数是 array_asanyarray，接受快速调用和关键字参数
    {"ascontiguousarray",
        (PyCFunction)array_ascontiguousarray,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "ascontiguousarray" 的方法，对应的 C 函数是 array_ascontiguousarray，接受快速调用和关键字参数
    {"asfortranarray",
        (PyCFunction)array_asfortranarray,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "asfortranarray" 的方法，对应的 C 函数是 array_asfortranarray，接受快速调用和关键字参数
    {"copyto",
        (PyCFunction)array_copyto,
        METH_VARARGS|METH_KEYWORDS, NULL},
    # 注册一个名为 "copyto" 的方法，对应的 C 函数是 array_copyto，接受位置参数和关键字参数
    {"nested_iters",
        (PyCFunction)NpyIter_NestedIters,
        METH_VARARGS|METH_KEYWORDS, NULL},
    # 注册一个名为 "nested_iters" 的方法，对应的 C 函数是 NpyIter_NestedIters，接受位置参数和关键字参数
    {"arange",
        (PyCFunction)array_arange,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "arange" 的方法，对应的 C 函数是 array_arange，接受快速调用和关键字参数
    {"zeros",
        (PyCFunction)array_zeros,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "zeros" 的方法，对应的 C 函数是 array_zeros，接受快速调用和关键字参数
    {"count_nonzero",
        (PyCFunction)array_count_nonzero,
        METH_FASTCALL, NULL},
    # 注册一个名为 "count_nonzero" 的方法，对应的 C 函数是 array_count_nonzero，接受快速调用，不接受关键字参数
    {"empty",
        (PyCFunction)array_empty,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "empty" 的方法，对应的 C 函数是 array_empty，接受快速调用和关键字参数
    {"empty_like",
        (PyCFunction)array_empty_like,
        METH_FASTCALL|METH_KEYWORDS, NULL},
    # 注册一个名为 "empty_like" 的方法，对应的 C 函数是 array_empty_like，接受快速调用和关键字参数
    {"scalar",
        (PyCFunction)array_scalar,
        METH_VARARGS|METH_KEYWORDS, NULL},
    # 注册一个名为 "scalar" 的方法，对应的 C 函数是 array_scalar，接受位置参数和关键字参数
    {"where",
        (PyCFunction)array_where,
        METH_FASTCALL, NULL},
    # 注册一个名为 "where" 的方法，对应的 C 函数是 array_where，接受快速调用，不接受关键字参数
    {"lexsort",
        (PyCFunction)array_lexsort,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "lexsort" 的方法，对应的 C 函数是 array_lexsort，接受快速调用和关键字参数
    {"putmask",
        (PyCFunction)array_putmask,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "putmask" 的方法，对应的 C 函数是 array_putmask，接受快速调用和关键字参数
    {"fromstring",
        (PyCFunction)array_fromstring,
        METH_VARARGS|METH_KEYWORDS, NULL},
    # 注册一个名为 "fromstring" 的方法，对应的 C 函数是 array_fromstring，接受位置参数和关键字参数
    {"fromiter",
        (PyCFunction)array_fromiter,
        METH_VARARGS|METH_KEYWORDS, NULL},
    # 注册一个名为 "fromiter" 的方法，对应的 C 函数是 array_fromiter，接受位置参数和关键字参数
    {"concatenate",
        (PyCFunction)array_concatenate,
        METH_FASTCALL|METH_KEYWORDS, NULL},
    # 注册一个名为 "concatenate" 的方法，对应的 C 函数是 array_concatenate，接受快速调用和关键字参数
    {"inner",
        (PyCFunction)array_innerproduct,
        METH_FASTCALL, NULL},
    # 注册一个名为 "inner" 的方法，对应的 C 函数是 array_innerproduct，接受快速调用，不接受关键字参数
    {"dot",
        (PyCFunction)array_matrixproduct,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "dot" 的方法，对应的 C 函数是 array_matrixproduct，接受快速调用和关键字参数
    {"vdot",
        (PyCFunction)array_vdot,
        METH_FASTCALL, NULL},
    # 注册一个名为 "vdot" 的方法，对应的 C 函数是 array_vdot，接受快速调用，不接受关键字参数
    {"c_einsum",
        (PyCFunction)array_einsum,
        METH_VARARGS|METH_KEYWORDS, NULL},
    # 注册一个名为 "c_einsum" 的方法，对应的 C 函数是 array_einsum，接受位置参数和关键字参数
    {"correlate",
        (PyCFunction)array_correlate,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "correlate" 的方法，对应的 C 函数是 array_correlate，接受快速调用和关键字参数
    {"correlate2",
        (PyCFunction)array_correlate2,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 注册一个名为 "correlate2" 的方法，对应的 C 函数是 array_correlate2，接受快速调用和关键字参数
    {"frombuffer",
        (PyCFunction)array_frombuffer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "frombuffer" 的键值对，值为指向 array_frombuffer 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"fromfile",
        (PyCFunction)array_fromfile,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "fromfile" 的键值对，值为指向 array_fromfile 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"can_cast",
        (PyCFunction)array_can_cast_safely,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "can_cast" 的键值对，值为指向 array_can_cast_safely 函数的指针，
    # 支持快速调用和关键字参数的调用方式，没有额外的说明信息
    {"promote_types",
        (PyCFunction)array_promote_types,
        METH_FASTCALL, NULL},
    # 定义名为 "promote_types" 的键值对，值为指向 array_promote_types 函数的指针，
    # 支持快速调用的调用方式，没有额外的说明信息
    {"min_scalar_type",
        (PyCFunction)array_min_scalar_type,
        METH_VARARGS, NULL},
    # 定义名为 "min_scalar_type" 的键值对，值为指向 array_min_scalar_type 函数的指针，
    # 支持位置参数的调用方式，没有额外的说明信息
    {"result_type",
        (PyCFunction)array_result_type,
        METH_FASTCALL, NULL},
    # 定义名为 "result_type" 的键值对，值为指向 array_result_type 函数的指针，
    # 支持快速调用的调用方式，没有额外的说明信息
    {"shares_memory",
        (PyCFunction)array_shares_memory,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "shares_memory" 的键值对，值为指向 array_shares_memory 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"may_share_memory",
        (PyCFunction)array_may_share_memory,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "may_share_memory" 的键值对，值为指向 array_may_share_memory 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    /* Datetime-related functions */
    # 下面是与日期时间相关的函数
    {"datetime_data",
        (PyCFunction)array_datetime_data,
        METH_VARARGS, NULL},
    # 定义名为 "datetime_data" 的键值对，值为指向 array_datetime_data 函数的指针，
    # 支持位置参数的调用方式，没有额外的说明信息
    {"datetime_as_string",
        (PyCFunction)array_datetime_as_string,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "datetime_as_string" 的键值对，值为指向 array_datetime_as_string 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    /* Datetime business-day API */
    # 下面是与工作日相关的日期时间 API
    {"busday_offset",
        (PyCFunction)array_busday_offset,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "busday_offset" 的键值对，值为指向 array_busday_offset 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"busday_count",
        (PyCFunction)array_busday_count,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "busday_count" 的键值对，值为指向 array_busday_count 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"is_busday",
        (PyCFunction)array_is_busday,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "is_busday" 的键值对，值为指向 array_is_busday 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"format_longfloat",
        (PyCFunction)format_longfloat,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "format_longfloat" 的键值对，值为指向 format_longfloat 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"dragon4_positional",
        (PyCFunction)dragon4_positional,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "dragon4_positional" 的键值对，值为指向 dragon4_positional 函数的指针，
    # 支持快速调用和关键字参数的调用方式，没有额外的说明信息
    {"dragon4_scientific",
        (PyCFunction)dragon4_scientific,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "dragon4_scientific" 的键值对，值为指向 dragon4_scientific 函数的指针，
    # 支持快速调用和关键字参数的调用方式，没有额外的说明信息
    {"compare_chararrays",
        (PyCFunction)compare_chararrays,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "compare_chararrays" 的键值对，值为指向 compare_chararrays 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"_vec_string",
        (PyCFunction)_vec_string,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "_vec_string" 的键值对，值为指向 _vec_string 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"_place", (PyCFunction)arr_place,
        METH_VARARGS | METH_KEYWORDS,
        "Insert vals sequentially into equivalent 1-d positions "
        "indicated by mask."},
    # 定义名为 "_place" 的键值对，值为指向 arr_place 函数的指针，
    # 支持位置参数和关键字参数的调用方式，说明信息为 "Insert vals sequentially..." 到末尾
    {"bincount", (PyCFunction)arr_bincount,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "bincount" 的键值对，值为指向 arr_bincount 函数的指针，
    # 支持快速调用和关键字参数的调用方式，没有额外的说明信息
    {"_monotonicity", (PyCFunction)arr__monotonicity,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "_monotonicity" 的键值对，值为指向 arr__monotonicity 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"interp", (PyCFunction)arr_interp,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "interp" 的键值对，值为指向 arr_interp 函数的指针，
    # 支持快速调用和关键字参数的调用方式，没有额外的说明信息
    {"interp_complex", (PyCFunction)arr_interp_complex,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "interp_complex" 的键值对，值为指向 arr_interp_complex 函数的指针，
    # 支持快速调用和关键字参数的调用方式，没有额外的说明信息
    {"ravel_multi_index", (PyCFunction)arr_ravel_multi_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "ravel_multi_index" 的键值对，值为指向 arr_ravel_multi_index 函数的指针，
    # 支持位置参数和关键字参数的调用方式，没有额外的说明信息
    {"unravel
    {"normalize_axis_index", (PyCFunction)normalize_axis_index,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "normalize_axis_index" 的 C 函数，使用 METH_FASTCALL 和 METH_KEYWORDS 标志
    {"set_legacy_print_mode", (PyCFunction)set_legacy_print_mode,
        METH_VARARGS, NULL},
    # 定义名为 "set_legacy_print_mode" 的 C 函数，使用 METH_VARARGS 标志
    {"_discover_array_parameters", (PyCFunction)_discover_array_parameters,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "_discover_array_parameters" 的 C 函数，使用 METH_FASTCALL 和 METH_KEYWORDS 标志
    {"_get_castingimpl",  (PyCFunction)_get_castingimpl,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "_get_castingimpl" 的 C 函数，使用 METH_VARARGS 和 METH_KEYWORDS 标志
    {"_load_from_filelike", (PyCFunction)_load_from_filelike,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "_load_from_filelike" 的 C 函数，使用 METH_FASTCALL 和 METH_KEYWORDS 标志
    /* from umath */
    # 以下几行是注释，说明接下来的函数来自 umath 模块
    {"frompyfunc",
        (PyCFunction) ufunc_frompyfunc,
        METH_VARARGS | METH_KEYWORDS, NULL},
    # 定义名为 "frompyfunc" 的 C 函数，使用 METH_VARARGS 和 METH_KEYWORDS 标志
    {"_make_extobj",
        (PyCFunction)extobj_make_extobj,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "_make_extobj" 的 C 函数，使用 METH_FASTCALL 和 METH_KEYWORDS 标志
    {"_get_extobj_dict",
        (PyCFunction)extobj_get_extobj_dict,
        METH_NOARGS, NULL},
    # 定义名为 "_get_extobj_dict" 的 C 函数，使用 METH_NOARGS 标志
    {"get_handler_name",
        (PyCFunction) get_handler_name,
        METH_VARARGS, NULL},
    # 定义名为 "get_handler_name" 的 C 函数，使用 METH_VARARGS 标志
    {"get_handler_version",
        (PyCFunction) get_handler_version,
        METH_VARARGS, NULL},
    # 定义名为 "get_handler_version" 的 C 函数，使用 METH_VARARGS 标志
    {"_get_promotion_state",
        (PyCFunction)npy__get_promotion_state,
        METH_NOARGS, "Get the current NEP 50 promotion state."},
    # 定义名为 "_get_promotion_state" 的 C 函数，使用 METH_NOARGS 标志，附带说明字符串
    {"_set_promotion_state",
         (PyCFunction)npy__set_promotion_state,
         METH_O, "Set the NEP 50 promotion state.  This is not thread-safe.\n"
                 "The optional warnings can be safely silenced using the \n"
                 "`np._no_nep50_warning()` context manager."},
    # 定义名为 "_set_promotion_state" 的 C 函数，使用 METH_O 标志，附带详细说明字符串
    {"_set_numpy_warn_if_no_mem_policy",
         (PyCFunction)_set_numpy_warn_if_no_mem_policy,
         METH_O, "Change the warn if no mem policy flag for testing."},
    # 定义名为 "_set_numpy_warn_if_no_mem_policy" 的 C 函数，使用 METH_O 标志，附带简要说明字符串
    {"_add_newdoc_ufunc", (PyCFunction)add_newdoc_ufunc,
        METH_VARARGS, NULL},
    # 定义名为 "_add_newdoc_ufunc" 的 C 函数，使用 METH_VARARGS 标志
    {"_get_sfloat_dtype",
        get_sfloat_dtype, METH_NOARGS, NULL},
    # 定义名为 "_get_sfloat_dtype" 的 C 函数，使用 METH_NOARGS 标志
    {"_get_madvise_hugepage", (PyCFunction)_get_madvise_hugepage,
        METH_NOARGS, NULL},
    # 定义名为 "_get_madvise_hugepage" 的 C 函数，使用 METH_NOARGS 标志
    {"_set_madvise_hugepage", (PyCFunction)_set_madvise_hugepage,
        METH_O, NULL},
    # 定义名为 "_set_madvise_hugepage" 的 C 函数，使用 METH_O 标志
    {"_reload_guard", (PyCFunction)_reload_guard,
        METH_NOARGS,
        "Give a warning on reload and big warning in sub-interpreters."},
    # 定义名为 "_reload_guard" 的 C 函数，使用 METH_NOARGS 标志，附带详细说明字符串
    {"from_dlpack", (PyCFunction)from_dlpack,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    # 定义名为 "from_dlpack" 的 C 函数，使用 METH_FASTCALL 和 METH_KEYWORDS 标志
    {NULL, NULL, 0, NULL}                /* sentinel */
    # 最后一个条目，标志着结尾的 sentinel，不定义任何函数
/* Establish scalar-type hierarchy
 *
 *  For dual inheritance we need to make sure that the objects being
 *  inherited from have the tp->mro object initialized.  This is
 *  not necessarily true for the basic type objects of Python (it is
 *  checked for single inheritance but not dual in PyType_Ready).
 *
 *  Thus, we call PyType_Ready on the standard Python Types, here.
 */
static int
setup_scalartypes(PyObject *NPY_UNUSED(dict))
{
    // 检查并初始化标准 Python 类型对象，确保可以进行双重继承
    if (PyType_Ready(&PyBool_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyFloat_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyComplex_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyBytes_Type) < 0) {
        return -1;
    }
    if (PyType_Ready(&PyUnicode_Type) < 0) {
        return -1;
    }

#define SINGLE_INHERIT(child, parent)                                   \
    // 设置单一继承关系宏，初始化子类的类型对象
    Py##child##ArrType_Type.tp_base = &Py##parent##ArrType_Type;        \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

    // 初始化单一继承关系的各个类型对象
    if (PyType_Ready(&PyGenericArrType_Type) < 0) {
        return -1;
    }
    SINGLE_INHERIT(Number, Generic);
    SINGLE_INHERIT(Integer, Number);
    SINGLE_INHERIT(Inexact, Number);
    SINGLE_INHERIT(SignedInteger, Integer);
    SINGLE_INHERIT(UnsignedInteger, Integer);
    SINGLE_INHERIT(Floating, Inexact);
    SINGLE_INHERIT(ComplexFloating, Inexact);
    SINGLE_INHERIT(Flexible, Generic);
    SINGLE_INHERIT(Character, Flexible);

#define DUAL_INHERIT(child, parent1, parent2)                           \
    // 设置双重继承关系宏，初始化子类的类型对象
    Py##child##ArrType_Type.tp_base = &Py##parent2##ArrType_Type;       \
    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent2##ArrType_Type,               \
                      &Py##parent1##_Type);                             \
    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;       \
    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }

#define DUAL_INHERIT2(child, parent1, parent2)                          \
    // 设置另一种双重继承关系宏，将子类的类型对象基类设置为 parent1
    Py##child##ArrType_Type.tp_base = &Py##parent1##_Type;              \
    Py##child##ArrType_Type = PyType_Type;
        // 将 Py##child##ArrType_Type 设置为 PyType_Type

    Py##child##ArrType_Type.tp_bases =                                  \
        Py_BuildValue("(OO)", &Py##parent1##_Type,                      \
                      &Py##parent2##ArrType_Type);                      \
        // 设置 Py##child##ArrType_Type 的基类为 Py##parent1##_Type 和 Py##parent2##ArrType_Type

    Py##child##ArrType_Type.tp_richcompare =                            \
        Py##parent1##_Type.tp_richcompare;
        // 设置 Py##child##ArrType_Type 的富比较方法为 Py##parent1##_Type 的富比较方法

    Py##child##ArrType_Type.tp_hash = Py##parent1##_Type.tp_hash;       \
        // 设置 Py##child##ArrType_Type 的哈希方法为 Py##parent1##_Type 的哈希方法

    if (PyType_Ready(&Py##child##ArrType_Type) < 0) {                   \
        PyErr_Print();                                                  \
        PyErr_Format(PyExc_SystemError,                                 \
                     "could not initialize Py%sArrType_Type",           \
                     #child);                                           \
        return -1;                                                      \
    }
        // 如果初始化 Py##child##ArrType_Type 失败，则打印错误信息并返回 -1

    SINGLE_INHERIT(Bool, Generic);
        // 单继承：Bool 继承自 Generic

    SINGLE_INHERIT(Byte, SignedInteger);
        // 单继承：Byte 继承自 SignedInteger

    SINGLE_INHERIT(Short, SignedInteger);
        // 单继承：Short 继承自 SignedInteger

    SINGLE_INHERIT(Int, SignedInteger);
        // 单继承：Int 继承自 SignedInteger

    SINGLE_INHERIT(Long, SignedInteger);
        // 单继承：Long 继承自 SignedInteger

    SINGLE_INHERIT(LongLong, SignedInteger);
        // 单继承：LongLong 继承自 SignedInteger

    /* Datetime doesn't fit in any category */
    SINGLE_INHERIT(Datetime, Generic);
        // 单继承：Datetime 继承自 Generic

    /* Timedelta is an integer with an associated unit */
    SINGLE_INHERIT(Timedelta, SignedInteger);
        // 单继承：Timedelta 继承自 SignedInteger

    SINGLE_INHERIT(UByte, UnsignedInteger);
        // 单继承：UByte 继承自 UnsignedInteger

    SINGLE_INHERIT(UShort, UnsignedInteger);
        // 单继承：UShort 继承自 UnsignedInteger

    SINGLE_INHERIT(UInt, UnsignedInteger);
        // 单继承：UInt 继承自 UnsignedInteger

    SINGLE_INHERIT(ULong, UnsignedInteger);
        // 单继承：ULong 继承自 UnsignedInteger

    SINGLE_INHERIT(ULongLong, UnsignedInteger);
        // 单继承：ULongLong 继承自 UnsignedInteger

    SINGLE_INHERIT(Half, Floating);
        // 单继承：Half 继承自 Floating

    SINGLE_INHERIT(Float, Floating);
        // 单继承：Float 继承自 Floating

    DUAL_INHERIT(Double, Float, Floating);
        // 双继承：Double 同时继承自 Float 和 Floating

    SINGLE_INHERIT(LongDouble, Floating);
        // 单继承：LongDouble 继承自 Floating

    SINGLE_INHERIT(CFloat, ComplexFloating);
        // 单继承：CFloat 继承自 ComplexFloating

    DUAL_INHERIT(CDouble, Complex, ComplexFloating);
        // 双继承：CDouble 同时继承自 Complex 和 ComplexFloating

    SINGLE_INHERIT(CLongDouble, ComplexFloating);
        // 单继承：CLongDouble 继承自 ComplexFloating

    DUAL_INHERIT2(String, String, Character);
        // 双继承：String 同时继承自 String 和 Character

    DUAL_INHERIT2(Unicode, Unicode, Character);
        // 双继承：Unicode 同时继承自 Unicode 和 Character

    SINGLE_INHERIT(Void, Flexible);
        // 单继承：Void 继承自 Flexible

    SINGLE_INHERIT(Object, Generic);
        // 单继承：Object 继承自 Generic

    return 0;
        // 函数返回 0，表示初始化成功
/*
 * Clean up string and unicode array types so they act more like
 * strings -- get their tables from the standard types.
 */
}

/* place a flag dictionary in d */

static void
set_flaginfo(PyObject *d)
{
    PyObject *s;
    PyObject *newd;

    newd = PyDict_New();

#define _addnew(key, val, one)                                       \
    PyDict_SetItemString(newd, #key, s=PyLong_FromLong(val));    \
    Py_DECREF(s);                                               \
    PyDict_SetItemString(newd, #one, s=PyLong_FromLong(val));    \
    Py_DECREF(s)

#define _addone(key, val)                                            \
    PyDict_SetItemString(newd, #key, s=PyLong_FromLong(val));    \
    Py_DECREF(s)

    /* Define flag constants and populate them in the dictionary */
    _addnew(OWNDATA, NPY_ARRAY_OWNDATA, O);
    _addnew(FORTRAN, NPY_ARRAY_F_CONTIGUOUS, F);
    _addnew(CONTIGUOUS, NPY_ARRAY_C_CONTIGUOUS, C);
    _addnew(ALIGNED, NPY_ARRAY_ALIGNED, A);
    _addnew(WRITEBACKIFCOPY, NPY_ARRAY_WRITEBACKIFCOPY, X);
    _addnew(WRITEABLE, NPY_ARRAY_WRITEABLE, W);
    _addone(C_CONTIGUOUS, NPY_ARRAY_C_CONTIGUOUS);
    _addone(F_CONTIGUOUS, NPY_ARRAY_F_CONTIGUOUS);

#undef _addone
#undef _addnew

    /* Add the flag dictionary to the input dictionary `d` */
    PyDict_SetItemString(d, "_flagdict", newd);
    Py_DECREF(newd);
    return;
}

// static variables are automatically zero-initialized
NPY_VISIBILITY_HIDDEN npy_thread_unsafe_state_struct npy_thread_unsafe_state;

static int
initialize_thread_unsafe_state(void) {
    char *env = getenv("NUMPY_WARN_IF_NO_MEM_POLICY");
    if ((env != NULL) && (strncmp(env, "1", 1) == 0)) {
        npy_thread_unsafe_state.warn_if_no_mem_policy = 1;
    }
    else {
        npy_thread_unsafe_state.warn_if_no_mem_policy = 0;
    }

    /* Initialize legacy print mode to INT_MAX */
    npy_thread_unsafe_state.legacy_print_mode = INT_MAX;

    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_multiarray_umath",
        NULL,
        -1,
        array_module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__multiarray_umath(void) {
    PyObject *m, *d, *s;
    PyObject *c_api;

    /* Create the module and add the functions */
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    /* Initialize CPU features */
    if (npy_cpu_init() < 0) {
        goto err;
    }
    /* Initialize CPU dispatch tracer */
    if (npy_cpu_dispatch_tracer_init(m) < 0) {
        goto err;
    }

#if defined(MS_WIN64) && defined(__GNUC__)
  PyErr_WarnEx(PyExc_Warning,
        "Numpy built with MINGW-W64 on Windows 64 bits is experimental, " \
        "and only available for \n" \
        "testing. You are advised not to use it for production. \n\n" \
        "CRASHES ARE TO BE EXPECTED - PLEASE REPORT THEM TO NUMPY DEVELOPERS",
        1);
#endif

    /* Initialize access to the PyDateTime API */
    numpy_pydatetime_import();

    if (PyErr_Occurred()) {
        goto err;
    /* 将一些符号常量添加到模块中 */
    /* 获取模块的字典 */
    d = PyModule_GetDict(m);
    if (!d) {
        goto err;
    }

    /* 初始化内部字符串 */
    if (intern_strings() < 0) {
        goto err;
    }

    /* 初始化静态全局变量 */
    if (initialize_static_globals() < 0) {
        goto err;
    }

    /* 初始化线程不安全状态 */
    if (initialize_thread_unsafe_state() < 0) {
        goto err;
    }

    /* 初始化扩展对象 */
    if (init_extobj() < 0) {
        goto err;
    }

    /* 准备PyUFunc_Type类型 */
    if (PyType_Ready(&PyUFunc_Type) < 0) {
        goto err;
    }

    /* 设置PyArrayDTypeMeta_Type的基类为PyType_Type，并准备该类型 */
    PyArrayDTypeMeta_Type.tp_base = &PyType_Type;
    if (PyType_Ready(&PyArrayDTypeMeta_Type) < 0) {
        goto err;
    }

    /* 设置PyArrayDescr_Type的哈希函数为PyArray_DescrHash，并设置其类型为PyArrayDTypeMeta_Type，然后准备该类型 */
    PyArrayDescr_Type.tp_hash = PyArray_DescrHash;
    Py_SET_TYPE(&PyArrayDescr_Type, &PyArrayDTypeMeta_Type);
    if (PyType_Ready(&PyArrayDescr_Type) < 0) {
        goto err;
    }

    /* 初始化类型转换表 */
    initialize_casting_tables();

    /* 初始化数值类型 */
    initialize_numeric_types();

    /* 初始化标量数学函数 */
    if (initscalarmath(m) < 0) {
        goto err;
    }

    /* 准备PyArray_Type类型 */
    if (PyType_Ready(&PyArray_Type) < 0) {
        goto err;
    }

    /* 设置标量类型 */
    if (setup_scalartypes(d) < 0) {
        goto err;
    }

    /* 设置迭代器类型并准备 */
    PyArrayIter_Type.tp_iter = PyObject_SelfIter;
    if (PyType_Ready(&PyArrayIter_Type) < 0) {
        goto err;
    }

    /* 准备PyArrayMapIter_Type类型 */
    if (PyType_Ready(&PyArrayMapIter_Type) < 0) {
        goto err;
    }

    /* 准备PyArrayMultiIter_Type类型 */
    if (PyType_Ready(&PyArrayMultiIter_Type) < 0) {
        goto err;
    }

    /* 设置PyArrayNeighborhoodIter_Type的构造函数为PyType_GenericNew，并准备该类型 */
    PyArrayNeighborhoodIter_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyArrayNeighborhoodIter_Type) < 0) {
        goto err;
    }

    /* 准备NpyIter_Type类型 */
    if (PyType_Ready(&NpyIter_Type) < 0) {
        goto err;
    }

    /* 准备PyArrayFlags_Type类型 */
    if (PyType_Ready(&PyArrayFlags_Type) < 0) {
        goto err;
    }

    /* 设置NpyBusDayCalendar_Type的构造函数为PyType_GenericNew，并准备该类型 */
    NpyBusDayCalendar_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&NpyBusDayCalendar_Type) < 0) {
        goto err;
    }

    /*
     * PyExc_Exception应该捕获所有标准错误，而不是字符串异常"multiarray.error"
     * 这是为了向后兼容现有代码
     */
    /* 将PyExc_Exception设置为"error"键的值 */
    PyDict_SetItemString(d, "error", PyExc_Exception);

    /* 设置"tracemalloc_domain"键的值为NPY_TRACE_DOMAIN的长整型 */
    s = PyLong_FromLong(NPY_TRACE_DOMAIN);
    PyDict_SetItemString(d, "tracemalloc_domain", s);
    Py_DECREF(s);

    /* 设置"__version__"键的值为"3.1"的Unicode字符串 */
    s = PyUnicode_FromString("3.1");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    /* 获取CPU特性字典 */
    s = npy_cpu_features_dict();
    if (s == NULL) {
        goto err;
    }
    /* 将CPU特性字典设置为"__cpu_features__"键的值 */
    if (PyDict_SetItemString(d, "__cpu_features__", s) < 0) {
        Py_DECREF(s);
        goto err;
    }
    Py_DECREF(s);

    /* 获取CPU基线列表 */
    s = npy_cpu_baseline_list();
    if (s == NULL) {
        goto err;
    }
    /* 将CPU基线列表设置为"__cpu_baseline__"键的值 */
    if (PyDict_SetItemString(d, "__cpu_baseline__", s) < 0) {
        Py_DECREF(s);
        goto err;
    }
    Py_DECREF(s);

    /* 获取CPU分派列表 */
    s = npy_cpu_dispatch_list();
    if (s == NULL) {
        goto err;
    }
    /* 将CPU分派列表设置为"__cpu_dispatch__"键的值 */
    if (PyDict_SetItemString(d, "__cpu_dispatch__", s) < 0) {
        Py_DECREF(s);
        goto err;
    }
    Py_DECREF(s);
    // 使用 PyCapsule_New 函数创建一个包含 _datetime_strings 的指针的 PyCapsule 对象，并赋值给变量 s
    s = PyCapsule_New((void *)_datetime_strings, NULL, NULL);
    // 检查 s 是否为空，如果为空则跳转到 err 标签处理错误
    if (s == NULL) {
        goto err;
    }
    // 将 PyCapsule 对象 s 以 "DATETIMEUNITS" 为键添加到字典 d 中
    PyDict_SetItemString(d, "DATETIMEUNITS", s);
    // 减少 PyCapsule 对象 s 的引用计数，释放其占用的资源
    Py_DECREF(s);
#define ADDCONST(NAME)                          \
    s = PyLong_FromLong(NPY_##NAME);             \
    PyDict_SetItemString(d, #NAME, s);          \
    Py_DECREF(s)

// 添加常量 ALLOW_THREADS 到字典 d 中
ADDCONST(ALLOW_THREADS);
// 添加常量 BUFSIZE 到字典 d 中
ADDCONST(BUFSIZE);
// 添加常量 CLIP 到字典 d 中
ADDCONST(CLIP);

// 添加常量 ITEM_HASOBJECT 到字典 d 中
ADDCONST(ITEM_HASOBJECT);
// 添加常量 LIST_PICKLE 到字典 d 中
ADDCONST(ITEM_IS_POINTER);
// 添加常量 NEEDS_INIT 到字典 d 中
ADDCONST(NEEDS_INIT);
// 添加常量 NEEDS_PYAPI 到字典 d 中
ADDCONST(NEEDS_PYAPI);
// 添加常量 USE_GETITEM 到字典 d 中
ADDCONST(USE_GETITEM);
// 添加常量 USE_SETITEM 到字典 d 中
ADDCONST(USE_SETITEM);

// 添加常量 RAISE 到字典 d 中
ADDCONST(RAISE);
// 添加常量 WRAP 到字典 d 中
ADDCONST(WRAP);
// 添加常量 MAXDIMS 到字典 d 中
ADDCONST(MAXDIMS);

// 添加常量 MAY_SHARE_BOUNDS 到字典 d 中
ADDCONST(MAY_SHARE_BOUNDS);
// 添加常量 MAY_SHARE_EXACT 到字典 d 中
ADDCONST(MAY_SHARE_EXACT);
#undef ADDCONST

// 将 ndarray 类型的对象添加到字典 d 中
PyDict_SetItemString(d, "ndarray", (PyObject *)&PyArray_Type);
// 将 flatiter 类型的对象添加到字典 d 中
PyDict_SetItemString(d, "flatiter", (PyObject *)&PyArrayIter_Type);
// 将 nditer 类型的对象添加到字典 d 中
PyDict_SetItemString(d, "nditer", (PyObject *)&NpyIter_Type);
// 将 broadcast 类型的对象添加到字典 d 中
PyDict_SetItemString(d, "broadcast",
                     (PyObject *)&PyArrayMultiIter_Type);
// 将 dtype 类型的对象添加到字典 d 中
PyDict_SetItemString(d, "dtype", (PyObject *)&PyArrayDescr_Type);
// 将 flagsobj 类型的对象添加到字典 d 中
PyDict_SetItemString(d, "flagsobj", (PyObject *)&PyArrayFlags_Type);

// 将 busdaycalendar 类型的对象添加到字典 d 中
/* Business day calendar object */
PyDict_SetItemString(d, "busdaycalendar",
                        (PyObject *)&NpyBusDayCalendar_Type);
// 设置标志信息
set_flaginfo(d);

// 完成标量类型的初始化并通过命名空间或 typeinfo 字典公开它们
// 如果设置类型信息失败，则跳转到 err 标签
if (set_typeinfo(d) != 0) {
    goto err;
}

// 准备 PyArrayFunctionDispatcher_Type 类型，若失败则跳转到 err 标签
if (PyType_Ready(&PyArrayFunctionDispatcher_Type) < 0) {
    goto err;
}
// 将 PyArrayFunctionDispatcher_Type 类型添加到字典 d 中
PyDict_SetItemString(
        d, "_ArrayFunctionDispatcher",
        (PyObject *)&PyArrayFunctionDispatcher_Type);

// 准备 PyArrayArrayConverter_Type 类型，若失败则跳转到 err 标签
if (PyType_Ready(&PyArrayArrayConverter_Type) < 0) {
    goto err;
}
// 将 PyArrayArrayConverter_Type 类型添加到字典 d 中
PyDict_SetItemString(
        d, "_array_converter",
        (PyObject *)&PyArrayArrayConverter_Type);

// 准备 PyArrayMethod_Type 类型，若失败则跳转到 err 标签
if (PyType_Ready(&PyArrayMethod_Type) < 0) {
    goto err;
}
// 准备 PyBoundArrayMethod_Type 类型，若失败则跳转到 err 标签
if (PyType_Ready(&PyBoundArrayMethod_Type) < 0) {
    goto err;
}
// 初始化并映射 Python 类型到 dtype，若失败则跳转到 err 标签
if (initialize_and_map_pytypes_to_dtypes() < 0) {
    goto err;
}

// 初始化类型转换
if (PyArray_InitializeCasts() < 0) {
    goto err;
}

// 初始化字符串 dtype，若失败则跳转到 err 标签
if (init_string_dtype() < 0) {
    goto err;
}

// 初始化 umath 模块，若失败则跳转到 err 标签
if (initumath(m) != 0) {
    goto err;
}

// 设置矩阵乘法标志，若失败则跳转到 err 标签
if (set_matmul_flags(d) < 0) {
    goto err;
}

// 初始化静态引用到 ndarray.__array_*__ 特殊方法
npy_static_pydata.ndarray_array_finalize = PyObject_GetAttrString(
        (PyObject *)&PyArray_Type, "__array_finalize__");
// 若获取 ndarray_array_finalize 失败，则跳转到 err 标签
if (npy_static_pydata.ndarray_array_finalize == NULL) {
    goto err;
}
npy_static_pydata.ndarray_array_ufunc = PyObject_GetAttrString(
        (PyObject *)&PyArray_Type, "__array_ufunc__");
// 若获取 ndarray_array_ufunc 失败，则跳转到 err 标签
if (npy_static_pydata.ndarray_array_ufunc == NULL) {
    goto err;
}
npy_static_pydata.ndarray_array_function = PyObject_GetAttrString(
        (PyObject *)&PyArray_Type, "__array_function__");
// 若获取 ndarray_array_function 失败，则跳转到 err 标签
if (npy_static_pydata.ndarray_array_function == NULL) {
    goto err;
}
    /*
     * 初始化 np.dtypes.StringDType
     *
     * 注意，这里在初始化旧版内置 DTypes 之后进行，
     * 以避免在 NumPy 设置期间出现循环依赖。
     * 这是因为需要在 init_string_dtype() 之后执行此操作，
     * 而 init_string_dtype() 需要在旧版 dtypemeta 类可用之后执行。
     */
    npy_cache_import("numpy.dtypes", "_add_dtype_helper",
                     &npy_thread_unsafe_state._add_dtype_helper);
    if (npy_thread_unsafe_state._add_dtype_helper == NULL) {
        goto err;
    }

    if (PyObject_CallFunction(
            npy_thread_unsafe_state._add_dtype_helper,
            "Os", (PyObject *)&PyArray_StringDType, NULL) == NULL) {
        goto err;
    }
    PyDict_SetItemString(d, "StringDType", (PyObject *)&PyArray_StringDType);

    /*
     * 初始化默认的 PyDataMem_Handler capsule 单例。
     */
    PyDataMem_DefaultHandler = PyCapsule_New(
            &default_handler, MEM_HANDLER_CAPSULE_NAME, NULL);
    if (PyDataMem_DefaultHandler == NULL) {
        goto err;
    }
    /*
     * 使用默认的 PyDataMem_Handler capsule 初始化上下文本地的当前 handler。
     */
    current_handler = PyContextVar_New("current_allocator", PyDataMem_DefaultHandler);
    if (current_handler == NULL) {
        goto err;
    }

    // 初始化静态引用的零类似数组
    npy_static_pydata.zero_pyint_like_arr = PyArray_ZEROS(
            0, NULL, NPY_LONG, NPY_FALSE);
    if (npy_static_pydata.zero_pyint_like_arr == NULL) {
        goto err;
    }
    ((PyArrayObject_fields *)npy_static_pydata.zero_pyint_like_arr)->flags |=
            (NPY_ARRAY_WAS_PYTHON_INT|NPY_ARRAY_WAS_INT_AND_REPLACED);

    if (verify_static_structs_initialized() < 0) {
        goto err;
    }

    /*
     * 导出 API 表
     */
    c_api = PyCapsule_New((void *)PyArray_API, NULL, NULL);
    /* dtype API 不是通过 Python 脚本自动填充/生成的： */
    _fill_dtype_api(PyArray_API);
    if (c_api == NULL) {
        goto err;
    }
    PyDict_SetItemString(d, "_ARRAY_API", c_api);
    Py_DECREF(c_api);

    c_api = PyCapsule_New((void *)PyUFunc_API, NULL, NULL);
    if (c_api == NULL) {
        goto err;
    }
    PyDict_SetItemString(d, "_UFUNC_API", c_api);
    Py_DECREF(c_api);
    if (PyErr_Occurred()) {
        goto err;
    }

    return m;

 err:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load multiarray module.");
    }
    Py_DECREF(m);
    return NULL;
}



# 这行代码是一个单独的右大括号 '}'，用于结束一个代码块或语句块。
# 在程序中，右大括号通常与左大括号 '{' 成对出现，用于定义代码块的起始和结束位置。
# 在这段示例中，单独的右大括号可能是某个控制结构、函数定义或类定义的结束标志，但缺少上下文难以具体确定其用途。
# 一般情况下，右大括号用于结束代码块，但单独的右大括号出现在示例中，需要更多上下文才能准确解释其作用。
```