# `.\numpy\numpy\_core\src\multiarray\compiled_base.c`

```py
/*
 * 定义 NPY_NO_DEPRECATED_API 为 NPY_API_VERSION，避免使用已弃用的 API
 * 定义 _MULTIARRAYMODULE
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 引入必要的头文件：
 * - Python.h：Python C API 的核心头文件
 * - structmember.h：定义结构体成员的相关宏和函数
 * - arrayobject.h：NumPy 多维数组对象的头文件
 * - npy_3kcompat.h：NumPy 兼容 Python 3 的头文件
 * - npy_math.h：NumPy 数学运算的头文件
 * - npy_argparse.h：NumPy 参数解析相关的头文件
 * - npy_config.h：NumPy 的配置文件
 * - templ_common.h：用于模板通用操作的头文件，如 npy_mul_sizes_with_overflow
 * - lowlevel_strided_loops.h：低级别分块循环的头文件，如 npy_bswap8
 * - alloc.h：内存分配相关的头文件
 * - ctors.h：构造函数相关的头文件
 * - common.h：通用工具函数的头文件
 * - dtypemeta.h：数据类型元信息的头文件
 * - simd/simd.h：SIMD（单指令多数据）操作的头文件
 * - string.h：C 标准字符串操作的头文件
 */
#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "numpy/npy_math.h"
#include "npy_argparse.h"
#include "npy_config.h"
#include "templ_common.h" /* for npy_mul_sizes_with_overflow */
#include "lowlevel_strided_loops.h" /* for npy_bswap8 */
#include "alloc.h"
#include "ctors.h"
#include "common.h"
#include "dtypemeta.h"
#include "simd/simd.h"
#include <string.h>

/*
 * 定义枚举 PACK_ORDER，用于表示打包顺序：
 * - PACK_ORDER_LITTLE：小端顺序
 * - PACK_ORDER_BIG：大端顺序
 */
typedef enum {
    PACK_ORDER_LITTLE = 0,
    PACK_ORDER_BIG
} PACK_ORDER;

/*
 * 检查数组是否单调的函数
 * 返回值：
 * -1：数组单调递减
 * +1：数组单调递增
 *  0：数组不单调
 */
static int
check_array_monotonic(const double *a, npy_intp lena)
{
    npy_intp i;
    double next;
    double last;

    if (lena == 0) {
        /* 如果数组长度为0，所有的元素都相同，认为是单调递增 */
        return 1;
    }
    last = a[0];

    /* 跳过数组开头的重复值 */
    for (i = 1; (i < lena) && (a[i] == last); i++);

    if (i == lena) {
        /* 如果跳过重复值后数组长度为0，所有的元素都相同，认为是单调递增 */
        return 1;
    }

    next = a[i];
    if (last < next) {
        /* 可能是单调递增 */
        for (i += 1; i < lena; i++) {
            last = next;
            next = a[i];
            if (last > next) {
                return 0;
            }
        }
        return 1;
    }
    else {
        /* last > next，可能是单调递减 */
        for (i += 1; i < lena; i++) {
            last = next;
            next = a[i];
            if (last < next) {
                return 0;
            }
        }
        return -1;
    }
}

/*
 * 找到整数数组的最小值和最大值的函数
 */
static void
minmax(const npy_intp *data, npy_intp data_len, npy_intp *mn, npy_intp *mx)
{
    npy_intp min = *data;
    npy_intp max = *data;

    while (--data_len) {
        const npy_intp val = *(++data);
        if (val < min) {
            min = val;
        }
        else if (val > max) {
            max = val;
        }
    }

    *mn = min;
    *mx = max;
}

/*
 * 注册为 bincount 的 arr_bincount 函数
 * bincount 接受一个、两个或三个参数：
 * - 第一个参数是非负整数数组
 * - 第二个参数是权重数组（如果有），必须能够提升为 double 类型
 * - 第三个参数（如果有）是期望输出数组的最小长度
 * 如果没有权重数组，bincount(list)[i] 表示在 list 中出现 i 的次数；
 * 如果有权重数组，则 bincount(self, list, weight)[i] 表示所有 list[j] == i 的权重和。
 * 不使用 Self 参数。
 */
NPY_NO_EXPORT PyObject *
arr_bincount(PyObject *NPY_UNUSED(self), PyObject *const *args,
                            Py_ssize_t len_args, PyObject *kwnames)
{
    // 声明变量，用于存储函数参数和结果
    PyObject *list = NULL, *weight = Py_None, *mlength = NULL;
    PyArrayObject *lst = NULL, *ans = NULL, *wts = NULL;
    npy_intp *numbers, *ians, len, mx, mn, ans_size;
    npy_intp minlength = 0;
    npy_intp i;
    double *weights , *dans;

    // 解析函数参数
    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("bincount", args, len_args, kwnames,
                "list", NULL, &list,
                "|weights", NULL, &weight,
                "|minlength", NULL, &mlength,
                NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 将传入的列表参数转换为一维整型数组对象
    lst = (PyArrayObject *)PyArray_ContiguousFromAny(list, NPY_INTP, 1, 1);
    if (lst == NULL) {
        goto fail;
    }
    len = PyArray_SIZE(lst);

    /*
     * This if/else if can be removed by changing the argspec to O|On above,
     * once we retire the deprecation
     */
    // 处理 minlength 参数的特殊情况
    if (mlength == Py_None) {
        /* NumPy 1.14, 2017-06-01 */
        // 发出弃用警告，建议传入 0 作为 minlength 而不是 None
        if (DEPRECATE("0 should be passed as minlength instead of None; "
                      "this will error in future.") < 0) {
            goto fail;
        }
    }
    else if (mlength != NULL) {
        // 将传入的 minlength 参数转换为 npy_intp 类型
        minlength = PyArray_PyIntAsIntp(mlength);
        if (error_converting(minlength)) {
            goto fail;
        }
    }

    // 检查 minlength 是否为负数，若是则报错并跳转到错误处理部分
    if (minlength < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "'minlength' must not be negative");
        goto fail;
    }

    // 处理空列表的情况
    if (len == 0) {
        // 创建一个长度为 minlength 的全零数组对象
        ans = (PyArrayObject *)PyArray_ZEROS(1, &minlength, NPY_INTP, 0);
        if (ans == NULL){
            goto fail;
        }
        // 释放 lst 对象，并返回结果数组对象
        Py_DECREF(lst);
        return (PyObject *)ans;
    }

    // 获取列表数据的指针，并计算列表中的最小值和最大值
    numbers = (npy_intp *)PyArray_DATA(lst);
    minmax(numbers, len, &mn, &mx);
    // 检查列表中是否有负数，若有则报错并跳转到错误处理部分
    if (mn < 0) {
        PyErr_SetString(PyExc_ValueError,
                "'list' argument must have no negative elements");
        goto fail;
    }
    // 计算结果数组的大小
    ans_size = mx + 1;
    // 如果传入了 minlength 参数且 ans_size 小于 minlength，则以 minlength 为准
    if (mlength != Py_None) {
        if (ans_size < minlength) {
            ans_size = minlength;
        }
    }
    // 处理未传入权重参数的情况
    if (weight == Py_None) {
        // 创建一个长度为 ans_size 的全零数组对象
        ans = (PyArrayObject *)PyArray_ZEROS(1, &ans_size, NPY_INTP, 0);
        if (ans == NULL) {
            goto fail;
        }
        // 获取结果数组的数据指针，开启线程，并根据列表中的元素值进行计数
        ians = (npy_intp *)PyArray_DATA(ans);
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < len; i++)
            ians[numbers[i]] += 1;
        NPY_END_ALLOW_THREADS;
        // 释放 lst 对象
        Py_DECREF(lst);
    }
    // 如果不是首次调用该函数，则执行以下操作
    else {
        // 从给定对象 `weight` 创建一个连续的 `PyArrayObject` 对象，数据类型为双精度浮点数
        wts = (PyArrayObject *)PyArray_ContiguousFromAny(
                                                weight, NPY_DOUBLE, 1, 1);
        // 如果 `wts` 为 NULL，则跳转到失败处理标签
        if (wts == NULL) {
            goto fail;
        }
        // 获取 `wts` 对象的数据指针，并赋给 `weights`
        weights = (double *)PyArray_DATA(wts);
        // 如果 `wts` 对象的大小与 `len` 不相等，则设置错误信息并跳转到失败处理标签
        if (PyArray_SIZE(wts) != len) {
            PyErr_SetString(PyExc_ValueError,
                    "The weights and list don't have the same length.");
            goto fail;
        }
        // 创建一个元素个数为 `ans_size` 的双精度浮点数类型的零数组 `ans`
        ans = (PyArrayObject *)PyArray_ZEROS(1, &ans_size, NPY_DOUBLE, 0);
        // 如果 `ans` 为 NULL，则跳转到失败处理标签
        if (ans == NULL) {
            goto fail;
        }
        // 获取 `ans` 对象的数据指针，并赋给 `dans`
        dans = (double *)PyArray_DATA(ans);
        // 开始线程允许，用于多线程环境下的线程安全操作
        NPY_BEGIN_ALLOW_THREADS;
        // 遍历长度为 `len` 的循环
        for (i = 0; i < len; i++) {
            // 将 `weights[i]` 的值加到 `dans[numbers[i]]` 上
            dans[numbers[i]] += weights[i];
        }
        // 结束线程允许，恢复线程锁状态
        NPY_END_ALLOW_THREADS;
        // 释放 `lst` 和 `wts` 对象的引用
        Py_DECREF(lst);
        Py_DECREF(wts);
    }
    // 返回 `ans` 对象的 PyObject 指针形式
    return (PyObject *)ans;
fail:
    // 释放 lst 对象的引用计数
    Py_XDECREF(lst);
    // 释放 wts 对象的引用计数
    Py_XDECREF(wts);
    // 释放 ans 对象的引用计数
    Py_XDECREF(ans);
    // 返回 NULL 表示函数执行失败
    return NULL;
}

/* Internal function to expose check_array_monotonic to python */
// 定义一个不导出的函数，将 check_array_monotonic 暴露给 Python
NPY_NO_EXPORT PyObject *
arr__monotonicity(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"x", NULL};
    PyObject *obj_x = NULL;
    PyArrayObject *arr_x = NULL;
    long monotonic;
    npy_intp len_x;
    NPY_BEGIN_THREADS_DEF;

    // 解析参数，期望参数是一个对象和一个关键字参数列表
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:_monotonicity", kwlist,
                                     &obj_x)) {
        // 解析失败，返回 NULL
        return NULL;
    }

    /*
     * TODO:
     *  `x` could be strided, needs change to check_array_monotonic
     *  `x` is forced to double for this check
     */
    // 将 obj_x 转换为一个一维的 NPY_DOUBLE 类型的 PyArrayObject，以供后续的单调性检查使用
    arr_x = (PyArrayObject *)PyArray_FROMANY(
        obj_x, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO);
    if (arr_x == NULL) {
        // 转换失败，返回 NULL
        return NULL;
    }

    len_x = PyArray_SIZE(arr_x);  // 获取数组 arr_x 的大小
    NPY_BEGIN_THREADS_THRESHOLDED(len_x)
    monotonic = check_array_monotonic(
        (const double *)PyArray_DATA(arr_x), len_x);  // 调用 check_array_monotonic 函数进行单调性检查
    NPY_END_THREADS
    Py_DECREF(arr_x);  // 释放 arr_x 对象的引用计数

    // 将检查结果 monotonic 转换为 Python 的 long 类型，并返回
    return PyLong_FromLong(monotonic);
}

/*
 * Returns input array with values inserted sequentially into places
 * indicated by the mask
 */
// 返回一个在 mask 指示的位置上顺序插入值的输入数组
NPY_NO_EXPORT PyObject *
arr_place(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{
    char *src, *dest;
    npy_bool *mask_data;
    PyArray_Descr *dtype;
    PyArray_CopySwapFunc *copyswap;
    PyObject *array0, *mask0, *values0;
    PyArrayObject *array, *mask, *values;
    npy_intp i, j, chunk, nm, ni, nv;

    static char *kwlist[] = {"input", "mask", "vals", NULL};
    NPY_BEGIN_THREADS_DEF;
    values = mask = NULL;

    // 解析参数，期望参数是一个 PyArray_Type 类型的对象，以及两个任意对象
    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O!OO:place", kwlist,
                &PyArray_Type, &array0, &mask0, &values0)) {
        // 解析失败，返回 NULL
        return NULL;
    }

    // 将 array0 转换为一个 NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY 类型的 PyArrayObject 对象
    array = (PyArrayObject *)PyArray_FromArray((PyArrayObject *)array0, NULL,
                                    NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (array == NULL) {
        goto fail;  // 转换失败，跳转到 fail 标签处理异常
    }

    ni = PyArray_SIZE(array);  // 获取数组 array 的大小
    dest = PyArray_DATA(array);  // 获取数组 array 的数据起始地址
    chunk = PyArray_ITEMSIZE(array);  // 获取数组 array 的每个元素大小
    // 将 mask0 转换为一个 NPY_BOOL 类型的 PyArrayObject 对象
    mask = (PyArrayObject *)PyArray_FROM_OTF(mask0, NPY_BOOL,
                                NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST);
    if (mask == NULL) {
        goto fail;  // 转换失败，跳转到 fail 标签处理异常
    }

    nm = PyArray_SIZE(mask);  // 获取数组 mask 的大小
    if (nm != ni) {
        // mask 和 array 大小不匹配，抛出 ValueError 异常
        PyErr_SetString(PyExc_ValueError,
                        "place: mask and data must be "
                        "the same size");
        goto fail;  // 跳转到 fail 标签处理异常
    }

    mask_data = PyArray_DATA(mask);  // 获取数组 mask 的数据起始地址
    dtype = PyArray_DESCR(array);  // 获取数组 array 的数据描述符
    Py_INCREF(dtype);  // 增加数据描述符的引用计数

    // 将 values0 转换为一个 NPY_ARRAY_CARRAY 类型的 PyArrayObject 对象
    values = (PyArrayObject *)PyArray_FromAny(values0, dtype,
                                    0, 0, NPY_ARRAY_CARRAY, NULL);
    if (values == NULL) {
        goto fail;  // 转换失败，跳转到 fail 标签处理异常
    }

    nv = PyArray_SIZE(values);  // 获取数组 values 的大小（如果是空数组，则为零）
    # 检查插入值数量是否小于等于零
    if (nv <= 0):
        # 初始化一个布尔变量，用于标记所有值是否都为假
        npy_bool allFalse = 1;
        i = 0;

        # 遍历掩码数据，查找是否存在非零值
        while (allFalse && i < ni):
            if (mask_data[i]):
                allFalse = 0;
            else:
                i++;

        # 如果存在非零值，则抛出值错误异常
        if (!allFalse):
            PyErr_SetString(PyExc_ValueError,
                            "Cannot insert from an empty array!");
            # 跳转到错误处理标签
            goto fail;
        else:
            # 清理内存并返回 None
            Py_XDECREF(values);
            Py_XDECREF(mask);
            PyArray_ResolveWritebackIfCopy(array);
            Py_XDECREF(array);
            Py_RETURN_NONE;

    # 获取源数据的指针
    src = PyArray_DATA(values);
    j = 0;

    # 获取目标数组元素的复制交换函数
    copyswap = PyDataType_GetArrFuncs(PyArray_DESCR(array))->copyswap;
    # 启动线程（如果支持）
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(array));
    # 遍历输入数据
    for (i = 0; i < ni; i++) {
        if (mask_data[i]):
            # 确保复制的索引不超出目标数组的长度
            if (j >= nv):
                j = 0;

            # 执行复制交换操作
            copyswap(dest + i*chunk, src + j*chunk, 0, array);
            j++;
    }
    # 结束线程
    NPY_END_THREADS;

    # 清理内存并返回 None
    Py_XDECREF(values);
    Py_XDECREF(mask);
    PyArray_ResolveWritebackIfCopy(array);
    Py_DECREF(array);
    Py_RETURN_NONE;

fail:
    # 清理内存并返回 NULL，表示失败
    Py_XDECREF(mask);
    PyArray_ResolveWritebackIfCopy(array);
    Py_XDECREF(array);
    Py_XDECREF(values);
    return NULL;
}

#define LIKELY_IN_CACHE_SIZE 8

#ifdef __INTEL_COMPILER
#pragma intel optimization_level 0
#endif

/**
 * @brief Perform a linear search in a sorted array to find the largest index
 * such that the array element is less than or equal to the given key.
 *
 * @param key Key value to search for.
 * @param arr Sorted array to search within.
 * @param len Length of the array.
 * @param i0 Starting index for the search.
 * @return Largest index i such that arr[i] <= key.
 */
static inline npy_intp
_linear_search(const npy_double key, const npy_double *arr, const npy_intp len, const npy_intp i0)
{
    npy_intp i;

    for (i = i0; i < len && key >= arr[i]; i++);
    return i - 1;
}

/**
 * @brief Perform a binary search in a sorted array to find the index such that
 * arr[i] <= key < arr[i + 1].
 *
 * If a starting index guess is provided, it checks nearby values first.
 * Otherwise, it defaults to bisection method for finding the index.
 *
 * @param key Key value to search for.
 * @param arr Sorted array to search within.
 * @param len Length of the array.
 * @param guess Initial guess for the index.
 * @return Index i such that arr[i] <= key < arr[i + 1].
 */
static npy_intp
binary_search_with_guess(const npy_double key, const npy_double *arr,
                         npy_intp len, npy_intp guess)
{
    npy_intp imin = 0;
    npy_intp imax = len;

    /* Handle keys outside of the arr range first */
    if (key > arr[len - 1]) {
        return len;
    }
    else if (key < arr[0]) {
        return -1;
    }

    /*
     * If len <= 4 use linear search.
     * From above we know key >= arr[0] when we start.
     */
    if (len <= 4) {
        return _linear_search(key, arr, len, 1);
    }

    if (guess > len - 3) {
        guess = len - 3;
    }
    if (guess < 1)  {
        guess = 1;
    }

    /* check most likely values: guess - 1, guess, guess + 1 */
    if (key < arr[guess]) {
        if (key < arr[guess - 1]) {
            imax = guess - 1;
            /* last attempt to restrict search to items in cache */
            if (guess > LIKELY_IN_CACHE_SIZE &&
                        key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
                imin = guess - LIKELY_IN_CACHE_SIZE;
            }
        }
        else {
            /* key >= arr[guess - 1] */
            return guess - 1;
        }
    }
    else {
        /* key >= arr[guess] */
        if (key < arr[guess + 1]) {
            return guess;
        }
        else {
            /* key >= arr[guess + 1] */
            if (key < arr[guess + 2]) {
                return guess + 1;
            }
            else {
                /* key >= arr[guess + 2] */
                imin = guess + 2;
                /* last attempt to restrict search to items in cache */
                if (guess < len - LIKELY_IN_CACHE_SIZE - 1 &&
                            key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
                    imax = guess + LIKELY_IN_CACHE_SIZE;
                }
            }
        }
    }

    /* finally, find index by bisection */
    # 当最小索引小于最大索引时，执行循环
    while (imin < imax) {
        # 计算中间索引，避免溢出风险
        const npy_intp imid = imin + ((imax - imin) >> 1);
        # 如果目标值大于等于中间元素，调整最小索引到中间索引的下一个位置
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        # 否则，调整最大索引到中间索引处
        else {
            imax = imid;
        }
    }
    # 返回找到的目标值的位置，因为可能出现key小于arr[0]的情况，所以返回imin-1
    return imin - 1;
    }

#undef LIKELY_IN_CACHE_SIZE

NPY_NO_EXPORT PyObject *
arr_interp(PyObject *NPY_UNUSED(self), PyObject *const *args, Py_ssize_t len_args,
                             PyObject *kwnames)
{
    // 指针声明
    PyObject *fp, *xp, *x;
    // 左右边界初始化为 NULL
    PyObject *left = NULL, *right = NULL;
    // 数组对象声明
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    // 索引变量声明
    npy_intp i, lenx, lenxp;
    // 左右边界值声明
    npy_double lval, rval;
    // 指向数据的指针声明
    const npy_double *dy, *dx, *dz;
    npy_double *dres, *slopes = NULL;

    // 线程操作宏定义
    NPY_BEGIN_THREADS_DEF;

    // 参数解析准备
    NPY_PREPARE_ARGPARSER;
    // 解析函数参数，如果失败返回 NULL
    if (npy_parse_arguments("interp", args, len_args, kwnames,
                "x", NULL, &x,
                "xp", NULL, &xp,
                "fp", NULL, &fp,
                "|left", NULL, &left,
                "|right", NULL, &right,
                NULL, NULL, NULL) < 0) {
        return NULL;
    }

    // 将 fp 转换为 NPY_DOUBLE 类型的连续数组对象
    afp = (PyArrayObject *)PyArray_ContiguousFromAny(fp, NPY_DOUBLE, 1, 1);
    if (afp == NULL) {
        return NULL;
    }
    // 将 xp 转换为 NPY_DOUBLE 类型的连续数组对象
    axp = (PyArrayObject *)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (axp == NULL) {
        goto fail;
    }
    // 将 x 转换为 NPY_DOUBLE 类型的数组对象
    ax = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 0, 0);
    if (ax == NULL) {
        goto fail;
    }
    // 获取 axp 的长度
    lenxp = PyArray_SIZE(axp);
    // 如果长度为 0，抛出异常并跳转到失败处理标签
    if (lenxp == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    // 检查 afp 和 axp 的长度是否相同，如果不同则抛出异常并跳转到失败处理标签
    if (PyArray_SIZE(afp) != lenxp) {
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    // 创建一个与 ax 具有相同维度和形状的 NPY_DOUBLE 类型的新数组对象 af
    af = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ax),
                                            PyArray_DIMS(ax), NPY_DOUBLE);
    if (af == NULL) {
        goto fail;
    }
    // 获取 ax 的长度
    lenx = PyArray_SIZE(ax);

    // 获取 afp、axp、ax、af 数组对象的数据指针
    dy = (const npy_double *)PyArray_DATA(afp);
    dx = (const npy_double *)PyArray_DATA(axp);
    dz = (const npy_double *)PyArray_DATA(ax);
    dres = (npy_double *)PyArray_DATA(af);
    /* 获取左右填充值。*/
    if ((left == NULL) || (left == Py_None)) {
        lval = dy[0];
    }
    else {
        lval = PyFloat_AsDouble(left);
        // 如果转换失败则跳转到失败处理标签
        if (error_converting(lval)) {
            goto fail;
        }
    }
    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp - 1];
    }
    else {
        rval = PyFloat_AsDouble(right);
        // 如果转换失败则跳转到失败处理标签
        if (error_converting(rval)) {
            goto fail;
        }
    }

    /* binary_search_with_guess 至少需要一个长度为 3 的数组 */
    if (lenxp == 1) {
        const npy_double xp_val = dx[0];
        const npy_double fp_val = dy[0];

        // 多线程处理，根据 lenx 的大小决定是否启动多线程
        NPY_BEGIN_THREADS_THRESHOLDED(lenx);
        // 对于每个 x 中的元素进行插值计算
        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];
            dres[i] = (x_val < xp_val) ? lval :
                                         ((x_val > xp_val) ? rval : fp_val);
        }
        // 结束多线程处理
        NPY_END_THREADS;
    }
    else {
        npy_intp j = 0;

        /* only pre-calculate slopes if there are relatively few of them. */
        如果斜率相对较少，则预先计算斜率
        if (lenxp <= lenx) {
            // 分配内存以存储斜率数组，长度为 (lenxp - 1)
            slopes = PyArray_malloc((lenxp - 1) * sizeof(npy_double));
            // 如果分配失败，报告内存错误并跳转到失败处理部分
            if (slopes == NULL) {
                PyErr_NoMemory();
                goto fail;
            }
        }

        NPY_BEGIN_THREADS;  // 开始线程安全区域

        // 如果斜率数组非空，计算每段斜率
        if (slopes != NULL) {
            for (i = 0; i < lenxp - 1; ++i) {
                slopes[i] = (dy[i+1] - dy[i]) / (dx[i+1] - dx[i]);
            }
        }

        // 对每个需要插值的点进行处理
        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];

            // 如果 x_val 是 NaN，直接将结果设为 x_val 并继续下一个点
            if (npy_isnan(x_val)) {
                dres[i] = x_val;
                continue;
            }

            // 使用二分查找找到 x_val 在 dx 中的位置，j 是初始猜测位置
            j = binary_search_with_guess(x_val, dx, lenxp, j);

            // 根据查找结果决定如何插值
            if (j == -1) {
                dres[i] = lval;  // 如果找不到合适位置，结果设为左边界值 lval
            }
            else if (j == lenxp) {
                dres[i] = rval;  // 如果超出右边界，结果设为右边界值 rval
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];  // 如果在最后一个点上，结果直接设为对应的 dy 值
            }
            else if (dx[j] == x_val) {
                /* Avoid potential non-finite interpolation */
                // 如果精确找到了点，避免潜在的非有限插值问题，结果直接设为对应的 dy 值
                dres[i] = dy[j];
            }
            else {
                const npy_double slope =
                        (slopes != NULL) ? slopes[j] :
                        (dy[j+1] - dy[j]) / (dx[j+1] - dx[j]);

                // 使用线性插值计算结果
                dres[i] = slope*(x_val - dx[j]) + dy[j];

                // 如果插值结果是 NaN，尝试使用相邻点进行插值
                if (NPY_UNLIKELY(npy_isnan(dres[i]))) {
                    dres[i] = slope*(x_val - dx[j+1]) + dy[j+1];

                    // 如果再次插值结果仍然是 NaN，并且相邻点的 dy 值相同，则结果设为相邻点的 dy 值
                    if (NPY_UNLIKELY(npy_isnan(dres[i])) && dy[j] == dy[j+1]) {
                        dres[i] = dy[j];
                    }
                }
            }
        }

        NPY_END_THREADS;  // 结束线程安全区域
    }

    // 释放斜率数组内存
    PyArray_free(slopes);

    // 释放引用的数组对象
    Py_DECREF(afp);
    Py_DECREF(axp);
    Py_DECREF(ax);

    // 返回插值结果数组对象
    return PyArray_Return(af);
fail:
    // 释放 afp 指针所指向的对象，并将其引用计数减少，避免内存泄漏
    Py_XDECREF(afp);
    // 释放 axp 指针所指向的对象，并将其引用计数减少，避免内存泄漏
    Py_XDECREF(axp);
    // 释放 ax 指针所指向的对象，并将其引用计数减少，避免内存泄漏
    Py_XDECREF(ax);
    // 释放 af 指针所指向的对象，并将其引用计数减少，避免内存泄漏
    Py_XDECREF(af);
    // 返回 NULL 表示函数执行失败
    return NULL;
}

/* As for arr_interp but for complex fp values */
NPY_NO_EXPORT PyObject *
arr_interp_complex(PyObject *NPY_UNUSED(self), PyObject *const *args, Py_ssize_t len_args,
                             PyObject *kwnames)
{
    // 声明需要使用的变量
    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp;

    const npy_double *dx, *dz;
    const npy_cdouble *dy;
    npy_cdouble lval, rval;
    npy_cdouble *dres, *slopes = NULL;

    NPY_BEGIN_THREADS_DEF;

    // 准备解析参数
    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("interp_complex", args, len_args, kwnames,
                "x", NULL, &x,
                "xp", NULL, &xp,
                "fp", NULL, &fp,
                "|left", NULL, &left,
                "|right", NULL, &right,
                NULL, NULL, NULL) < 0) {
        // 解析参数失败，返回 NULL 表示函数执行失败
        return NULL;
    }

    // 将 fp 转换为 NPY_CDOUBLE 类型的连续数组对象
    afp = (PyArrayObject *)PyArray_ContiguousFromAny(fp, NPY_CDOUBLE, 1, 1);

    if (afp == NULL) {
        // 转换失败，返回 NULL 表示函数执行失败
        return NULL;
    }

    // 将 xp 转换为 NPY_DOUBLE 类型的连续数组对象
    axp = (PyArrayObject *)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (axp == NULL) {
        // 转换失败，跳转到 fail 标签处处理错误
        goto fail;
    }
    // 将 x 转换为 NPY_DOUBLE 类型的连续数组对象
    ax = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 0, 0);
    if (ax == NULL) {
        // 转换失败，跳转到 fail 标签处处理错误
        goto fail;
    }
    lenxp = PyArray_SIZE(axp);
    if (lenxp == 0) {
        // xp 数组长度为 0，抛出 ValueError 异常并跳转到 fail 标签处处理错误
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    if (PyArray_SIZE(afp) != lenxp) {
        // fp 和 xp 数组长度不相等，抛出 ValueError 异常并跳转到 fail 标签处处理错误
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    lenx = PyArray_SIZE(ax);
    dx = (const npy_double *)PyArray_DATA(axp);
    dz = (const npy_double *)PyArray_DATA(ax);

    // 创建一个新的 NPY_CDOUBLE 类型的数组对象 af
    af = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ax),
                                            PyArray_DIMS(ax), NPY_CDOUBLE);
    if (af == NULL) {
        // 创建失败，跳转到 fail 标签处处理错误
        goto fail;
    }

    dy = (const npy_cdouble *)PyArray_DATA(afp);
    dres = (npy_cdouble *)PyArray_DATA(af);
    /* 获取 left 和 right 填充值 */
    if ((left == NULL) || (left == Py_None)) {
        // 如果 left 为 NULL 或者 Py_None，则将 lval 设置为 dy[0]
        lval = dy[0];
    }
    else {
        // 否则，从 left 中获取实部和虚部，并进行类型转换，如果转换失败则跳转到 fail 标签处理错误
        npy_csetreal(&lval, PyComplex_RealAsDouble(left));
        if (error_converting(npy_creal(lval))) {
            goto fail;
        }
        npy_csetimag(&lval, PyComplex_ImagAsDouble(left));
        if (error_converting(npy_cimag(lval))) {
            goto fail;
        }
    }

    if ((right == NULL) || (right == Py_None)) {
        // 如果 right 为 NULL 或者 Py_None，则将 rval 设置为 dy[lenxp - 1]
        rval = dy[lenxp - 1];
    }
    else {
        // 否则，从 right 中获取实部和虚部，并进行类型转换，如果转换失败则跳转到 fail 标签处理错误
        npy_csetreal(&rval, PyComplex_RealAsDouble(right));
        if (error_converting(npy_creal(rval))) {
            goto fail;
        }
        npy_csetimag(&rval, PyComplex_ImagAsDouble(right));
        if (error_converting(npy_cimag(rval))) {
            goto fail;
        }
    }

    // binary_search_with_guess 需要至少一个包含 3 个元素的数组
    # 如果 lenxp 等于 1，则执行以下操作
    if (lenxp == 1) {
        # 将 dx 数组的第一个元素赋给 xp_val
        const npy_double xp_val = dx[0];
        # 将 dy 数组的第一个元素赋给 fp_val
        const npy_cdouble fp_val = dy[0];

        # 使用线程阈值启动多线程
        NPY_BEGIN_THREADS_THRESHOLDED(lenx);
        # 遍历 dz 数组中的元素
        for (i = 0; i < lenx; ++i) {
            # 将 dz 数组中第 i 个元素赋给 x_val
            const npy_double x_val = dz[i];
            # 根据 x_val 和 xp_val 的比较结果，选择合适的值赋给 dres[i]
            dres[i] = (x_val < xp_val) ? lval :
                      ((x_val > xp_val) ? rval : fp_val);
        }
        # 结束多线程区块
        NPY_END_THREADS;
    }
    else {
        // 初始化变量 j 为 0
        npy_intp j = 0;
    
        /* only pre-calculate slopes if there are relatively few of them. */
        // 仅在斜率相对较少时预先计算斜率
        if (lenxp <= lenx) {
            // 分配内存以存储斜率数组，长度为 (lenxp - 1)
            slopes = PyArray_malloc((lenxp - 1) * sizeof(npy_cdouble));
            // 如果内存分配失败，设置内存错误并跳转到失败标签
            if (slopes == NULL) {
                PyErr_NoMemory();
                goto fail;
            }
        }
    
        NPY_BEGIN_THREADS; // 开始线程安全操作
    
        // 如果斜率数组非空，则计算斜率
        if (slopes != NULL) {
            for (i = 0; i < lenxp - 1; ++i) {
                // 计算斜率的实部和虚部
                const double inv_dx = 1.0 / (dx[i+1] - dx[i]);
                npy_csetreal(&slopes[i], (npy_creal(dy[i+1]) - npy_creal(dy[i])) * inv_dx);
                npy_csetimag(&slopes[i], (npy_cimag(dy[i+1]) - npy_cimag(dy[i])) * inv_dx);
            }
        }
    
        // 遍历输入数组的元素
        for (i = 0; i < lenx; ++i) {
            // 获取当前输入值 x_val
            const npy_double x_val = dz[i];
    
            // 如果 x_val 是 NaN，则将结果设置为 x_val 的实部为 NaN，虚部为 0
            if (npy_isnan(x_val)) {
                npy_csetreal(&dres[i], x_val);
                npy_csetimag(&dres[i], 0.0);
                continue; // 继续下一个循环
            }
    
            // 使用二分搜索找到 x_val 在 dx 中的位置 j
            j = binary_search_with_guess(x_val, dx, lenxp, j);
    
            // 根据搜索的结果 j 进行插值操作或者直接返回边界值
            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (dx[j] == x_val) {
                /* Avoid potential non-finite interpolation */
                dres[i] = dy[j];
            }
            else {
                npy_cdouble slope;
                // 如果斜率数组非空，则使用预先计算的斜率
                if (slopes != NULL) {
                    slope = slopes[j];
                }
                // 否则根据两个相邻点的差值计算斜率
                else {
                    const npy_double inv_dx = 1.0 / (dx[j+1] - dx[j]);
                    npy_csetreal(&slope, (npy_creal(dy[j+1]) - npy_creal(dy[j])) * inv_dx);
                    npy_csetimag(&slope, (npy_cimag(dy[j+1]) - npy_cimag(dy[j])) * inv_dx);
                }
    
                // 进行线性插值计算，同时处理可能出现的 NaN
                npy_csetreal(&dres[i], npy_creal(slope)*(x_val - dx[j]) + npy_creal(dy[j]));
                if (NPY_UNLIKELY(npy_isnan(npy_creal(dres[i])))) {
                    npy_csetreal(&dres[i], npy_creal(slope)*(x_val - dx[j+1]) + npy_creal(dy[j+1]));
                    if (NPY_UNLIKELY(npy_isnan(npy_creal(dres[i]))) &&
                            npy_creal(dy[j]) == npy_creal(dy[j+1])) {
                        npy_csetreal(&dres[i], npy_creal(dy[j]));
                    }
                }
                npy_csetimag(&dres[i], npy_cimag(slope)*(x_val - dx[j]) + npy_cimag(dy[j]));
                if (NPY_UNLIKELY(npy_isnan(npy_cimag(dres[i])))) {
                    npy_csetimag(&dres[i], npy_cimag(slope)*(x_val - dx[j+1]) + npy_cimag(dy[j+1]));
                    if (NPY_UNLIKELY(npy_isnan(npy_cimag(dres[i]))) &&
                            npy_cimag(dy[j]) == npy_cimag(dy[j+1])) {
                        npy_csetimag(&dres[i], npy_cimag(dy[j]));
                    }
                }
            }
        }
    
        NPY_END_THREADS; // 结束线程安全操作
    }
    # 释放 slopes 数组占用的内存
    PyArray_free(slopes);
    
    # 递减引用计数，可能释放 afp 所引用对象的内存
    Py_DECREF(afp);
    
    # 递减引用计数，可能释放 axp 所引用对象的内存
    Py_DECREF(axp);
    
    # 递减引用计数，可能释放 ax 所引用对象的内存
    Py_DECREF(ax);
    
    # 返回一个 Python 对象，该对象是 af 的 NumPy 数组表示
    return PyArray_Return(af);
/* 
 * 清理和释放资源，然后返回 NULL，表示失败
 */
fail:
    Py_XDECREF(afp);
    Py_XDECREF(axp);
    Py_XDECREF(ax);
    Py_XDECREF(af);
    return NULL;
}

/*
 * 空序列错误信息，用于指示非整数索引的情况
 */
static const char *EMPTY_SEQUENCE_ERR_MSG = "indices must be integral: the provided " \
    "empty sequence was inferred as float. Wrap it with " \
    "'np.array(indices, dtype=np.intp)'";

/*
 * 非整数错误信息，仅允许整数索引
 */
static const char *NON_INTEGRAL_ERROR_MSG = "only int indices permitted";

/* 
 * 将 Python 对象 obj 转换为具有整数 dtype 的 ndarray，或者转换失败
 */
static PyArrayObject *
astype_anyint(PyObject *obj) {
    PyArrayObject *ret;

    if (!PyArray_Check(obj)) {
        /* 首选 int dtype */
        PyArray_Descr *dtype_guess = NULL;
        if (PyArray_DTypeFromObject(obj, NPY_MAXDIMS, &dtype_guess) < 0) {
            return NULL;
        }
        if (dtype_guess == NULL) {
            if (PySequence_Check(obj) && PySequence_Size(obj) == 0) {
                PyErr_SetString(PyExc_TypeError, EMPTY_SEQUENCE_ERR_MSG);
            }
            return NULL;
        }
        ret = (PyArrayObject*)PyArray_FromAny(obj, dtype_guess, 0, 0, 0, NULL);
        if (ret == NULL) {
            return NULL;
        }
    }
    else {
        ret = (PyArrayObject *)obj;
        Py_INCREF(ret);
    }

    if (!(PyArray_ISINTEGER(ret) || PyArray_ISBOOL(ret))) {
        /* 确保 dtype 是基于 int 的 */
        PyErr_SetString(PyExc_TypeError, NON_INTEGRAL_ERROR_MSG);
        Py_DECREF(ret);
        return NULL;
    }

    return ret;
}

/*
 * 将 Python 序列转换为 'count' 个 PyArrayObject 数组
 *
 * seq         - 输入的 Python 对象，通常是一个元组，但任何序列都可以工作。
 *               必须包含整数内容。
 * paramname   - 产生 'seq' 的参数名称。
 * count       - 应该有多少数组（如果不匹配则报错）。
 * op          - 数组的存放位置。
 */
static int int_sequence_to_arrays(PyObject *seq,
                              char *paramname,
                              int count,
                              PyArrayObject **op
                              )
{
    int i;

    if (!PySequence_Check(seq) || PySequence_Size(seq) != count) {
        PyErr_Format(PyExc_ValueError,
                "parameter %s must be a sequence of length %d",
                paramname, count);
        return -1;
    }

    for (i = 0; i < count; ++i) {
        PyObject *item = PySequence_GetItem(seq, i);
        if (item == NULL) {
            goto fail;
        }
        op[i] = astype_anyint(item);
        Py_DECREF(item);
        if (op[i] == NULL) {
            goto fail;
        }
    }

    return 0;

fail:
    while (--i >= 0) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }
    return -1;
}

/* 
 * ravel_multi_index 的内部循环
 */
static int
ravel_multi_index_loop(int ravel_ndim, npy_intp *ravel_dims,
                        npy_intp *ravel_strides,
                        npy_intp count,
                        NPY_CLIPMODE *modes,
                        char **coords, npy_intp *coords_strides)
{
    int i;
    char invalid;
    npy_intp j, m;

    /*
     * 检查是否存在零维的轴，除非没有需要处理的情况。
     * 空数组/形状根本无法进行索引。
     */
    if (count != 0) {
        // 遍历展平后的维度数组
        for (i = 0; i < ravel_ndim; ++i) {
            // 如果有任何一个维度为0，抛出数值错误异常
            if (ravel_dims[i] == 0) {
                PyErr_SetString(PyExc_ValueError,
                        "cannot unravel if shape has zero entries (is empty).");
                return NPY_FAIL;
            }
        }
    }

    // 允许线程进入临界区域
    NPY_BEGIN_ALLOW_THREADS;
    invalid = 0;
    // 循环处理每个索引
    while (count--) {
        npy_intp raveled = 0;
        // 遍历展平后的维度数组
        for (i = 0; i < ravel_ndim; ++i) {
            m = ravel_dims[i];
            // 获取当前坐标的索引值
            j = *(npy_intp *)coords[i];
            switch (modes[i]) {
                case NPY_RAISE:
                    // 如果索引超出了范围，设置无效标志并跳出循环
                    if (j < 0 || j >= m) {
                        invalid = 1;
                        goto end_while;
                    }
                    break;
                case NPY_WRAP:
                    // 对超出范围的索引进行循环包裹
                    if (j < 0) {
                        j += m;
                        if (j < 0) {
                            j = j % m;
                            if (j != 0) {
                                j += m;
                            }
                        }
                    }
                    else if (j >= m) {
                        j -= m;
                        if (j >= m) {
                            j = j % m;
                        }
                    }
                    break;
                case NPY_CLIP:
                    // 对超出范围的索引进行截断处理
                    if (j < 0) {
                        j = 0;
                    }
                    else if (j >= m) {
                        j = m - 1;
                    }
                    break;

            }
            // 计算展平后的索引值
            raveled += j * ravel_strides[i];

            coords[i] += coords_strides[i];
        }
        // 将展平后的索引值写入坐标数组
        *(npy_intp *)coords[ravel_ndim] = raveled;
        coords[ravel_ndim] += coords_strides[ravel_ndim];
    }
end_while:
    NPY_END_ALLOW_THREADS;
    // 结束线程安全区域
    if (invalid) {
        // 如果坐标数组中存在无效项，设置值错误异常并返回失败状态
        PyErr_SetString(PyExc_ValueError,
              "invalid entry in coordinates array");
        return NPY_FAIL;
    }
    // 返回成功状态
    return NPY_SUCCEED;
}

/* ravel_multi_index implementation - see add_newdocs.py */
NPY_NO_EXPORT PyObject *
arr_ravel_multi_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    int i;
    PyObject *mode0=NULL, *coords0=NULL;
    PyArrayObject *ret = NULL;
    PyArray_Dims dimensions={0,0};
    npy_intp s, ravel_strides[NPY_MAXDIMS];
    NPY_ORDER order = NPY_CORDER;
    NPY_CLIPMODE modes[NPY_MAXDIMS];

    PyArrayObject *op[NPY_MAXARGS];
    PyArray_Descr *dtype[NPY_MAXARGS];
    npy_uint32 op_flags[NPY_MAXARGS];

    NpyIter *iter = NULL;

    static char *kwlist[] = {"multi_index", "dims", "mode", "order", NULL};

    // 初始化操作数组和数据类型数组
    memset(op, 0, sizeof(op));
    dtype[0] = NULL;

    // 解析参数和关键字参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                        "OO&|OO&:ravel_multi_index", kwlist,
                     &coords0,
                     PyArray_IntpConverter, &dimensions,
                     &mode0,
                     PyArray_OrderConverter, &order)) {
        goto fail;
    }

    // 检查维度数量是否超过最大值
    if (dimensions.len+1 > NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError,
                    "too many dimensions passed to ravel_multi_index");
        goto fail;
    }

    // 转换并检查剪切模式序列
    if (!PyArray_ConvertClipmodeSequence(mode0, modes, dimensions.len)) {
       goto fail;
    }

    // 根据顺序计算展平步长
    switch (order) {
        case NPY_CORDER:
            s = 1;
            for (i = dimensions.len-1; i >= 0; --i) {
                ravel_strides[i] = s;
                if (npy_mul_sizes_with_overflow(&s, s, dimensions.ptr[i])) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid dims: array size defined by dims is larger "
                        "than the maximum possible size.");
                    goto fail;
                }
            }
            break;
        case NPY_FORTRANORDER:
            s = 1;
            for (i = 0; i < dimensions.len; ++i) {
                ravel_strides[i] = s;
                if (npy_mul_sizes_with_overflow(&s, s, dimensions.ptr[i])) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid dims: array size defined by dims is larger "
                        "than the maximum possible size.");
                    goto fail;
                }
            }
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    // 将多索引转换为操作数组
    if (int_sequence_to_arrays(coords0, "multi_index", dimensions.len, op) < 0) {
        goto fail;
    }

    // 设置操作数组的标志
    for (i = 0; i < dimensions.len; ++i) {
        op_flags[i] = NPY_ITER_READONLY|
                      NPY_ITER_ALIGNED;
    }
    # 将操作标志设置为写入、对齐和分配内存的组合
    op_flags[dimensions.len] = NPY_ITER_WRITEONLY|
                               NPY_ITER_ALIGNED|
                               NPY_ITER_ALLOCATE;
    # 使用整数类型创建第一个数据类型描述符
    dtype[0] = PyArray_DescrFromType(NPY_INTP);
    # 复制第一个数据类型描述符到所有维度中
    for (i = 1; i <= dimensions.len; ++i) {
        dtype[i] = dtype[0];
    }

    # 创建多迭代器对象，设置迭代器的特性
    iter = NpyIter_MultiNew(dimensions.len+1, op, NPY_ITER_BUFFERED|
                                                  NPY_ITER_EXTERNAL_LOOP|
                                                  NPY_ITER_ZEROSIZE_OK,
                                                  NPY_KEEPORDER,
                                                  NPY_SAME_KIND_CASTING,
                                                  op_flags, dtype);
    # 如果迭代器创建失败，则跳转到失败标签
    if (iter == NULL) {
        goto fail;
    }

    # 如果迭代器大小不为零，则执行迭代操作
    if (NpyIter_GetIterSize(iter) != 0) {
        # 获取迭代器的下一个迭代函数指针及相关指针
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strides;
        npy_intp *countptr;

        iternext = NpyIter_GetIterNext(iter, NULL);
        # 如果获取迭代函数失败，则跳转到失败标签
        if (iternext == NULL) {
            goto fail;
        }
        # 获取数据指针数组、步长数组和内循环大小指针
        dataptr = NpyIter_GetDataPtrArray(iter);
        strides = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        # 执行多维索引展开循环，如果失败则跳转到失败标签
        do {
            if (ravel_multi_index_loop(dimensions.len, dimensions.ptr,
                        ravel_strides, *countptr, modes,
                        dataptr, strides) != NPY_SUCCEED) {
                goto fail;
            }
        } while(iternext(iter));
    }

    # 获取操作数数组中的返回值
    ret = NpyIter_GetOperandArray(iter)[dimensions.len];
    # 增加返回值的引用计数
    Py_INCREF(ret);

    # 释放第一个数据类型描述符的引用
    Py_DECREF(dtype[0]);
    # 释放操作数数组中的每个操作数的引用
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    # 释放维度缓存对象
    npy_free_cache_dim_obj(dimensions);
    # 释放迭代器对象
    NpyIter_Deallocate(iter);
    # 返回返回值的 Python 对象表示
    return PyArray_Return(ret);
fail:
    // 释放 dtype[0] 指向的对象，减少其引用计数
    Py_XDECREF(dtype[0]);
    // 循环释放 op 数组中每个元素指向的对象，减少它们的引用计数
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    // 释放 dimensions 所占用的内存
    npy_free_cache_dim_obj(dimensions);
    // 释放 NpyIter 迭代器所占用的资源
    NpyIter_Deallocate(iter);
    // 返回 NULL，表示操作失败
    return NULL;
}


/*
 * Inner loop for unravel_index
 * order must be NPY_CORDER or NPY_FORTRANORDER
 */
static int
unravel_index_loop(int unravel_ndim, npy_intp const *unravel_dims,
                   npy_intp unravel_size, npy_intp count,
                   char *indices, npy_intp indices_stride,
                   npy_intp *coords, NPY_ORDER order)
{
    int i, idx;
    // 根据 order 设置起始索引位置和步长
    int idx_start = (order == NPY_CORDER) ? unravel_ndim - 1: 0;
    int idx_step = (order == NPY_CORDER) ? -1 : 1;
    char invalid = 0;
    npy_intp val = 0;

    NPY_BEGIN_ALLOW_THREADS;
    // 断言 order 必须是 NPY_CORDER 或 NPY_FORTRANORDER
    assert(order == NPY_CORDER || order == NPY_FORTRANORDER);
    while (count--) {
        // 从 indices 中读取当前的 val 值
        val = *(npy_intp *)indices;
        // 检查 val 是否在合法范围内
        if (val < 0 || val >= unravel_size) {
            invalid = 1;
            break;
        }
        idx = idx_start;
        for (i = 0; i < unravel_ndim; ++i) {
            /*
             * 使用一个局部变量可能启用单一的除法优化
             * 但是只有在 / 操作符在 % 操作符之前时才会生效
             */
            npy_intp tmp = val / unravel_dims[idx];
            // 计算当前坐标的值，并更新 coords 数组
            coords[idx] = val % unravel_dims[idx];
            // 更新 val 为下一个维度的坐标值
            val = tmp;
            // 根据步长更新 idx 索引
            idx += idx_step;
        }
        // 更新 coords 和 indices 的指针位置
        coords += unravel_ndim;
        indices += indices_stride;
    }
    NPY_END_ALLOW_THREADS;
    // 如果发现索引超出范围，抛出异常并返回失败标志
    if (invalid) {
        PyErr_Format(PyExc_ValueError,
            "index %" NPY_INTP_FMT " is out of bounds for array with size "
            "%" NPY_INTP_FMT,
            val, unravel_size
        );
        return NPY_FAIL;
    }
    // 操作成功，返回成功标志
    return NPY_SUCCEED;
}

/* unravel_index implementation - see add_newdocs.py */
NPY_NO_EXPORT PyObject *
arr_unravel_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *indices0 = NULL;
    PyObject *ret_tuple = NULL;
    PyArrayObject *ret_arr = NULL;
    PyArrayObject *indices = NULL;
    PyArray_Descr *dtype = NULL;
    PyArray_Dims dimensions = {0, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_intp unravel_size;

    NpyIter *iter = NULL;
    int i, ret_ndim;
    npy_intp ret_dims[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];

    static char *kwlist[] = {"indices", "shape", "order", NULL};

    // 解析输入参数，获取 indices、dimensions 和 order
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|O&:unravel_index",
                    kwlist,
                    &indices0,
                    PyArray_IntpConverter, &dimensions,
                    PyArray_OrderConverter, &order)) {
        // 解析失败，跳转到 fail 标签处
        goto fail;
    }

    // 计算 unravel_size，即 dimensions 中所有元素的乘积
    unravel_size = PyArray_OverflowMultiplyList(dimensions.ptr, dimensions.len);
    // 检查是否计算溢出
    if (unravel_size == -1) {
        // 如果溢出，抛出 ValueError 异常并跳转到 fail 标签处
        PyErr_SetString(PyExc_ValueError,
                        "dimensions are too large; arrays and shapes with "
                        "a total size greater than 'intp' are not supported.");
        goto fail;
    }
    indices = astype_anyint(indices0);
    # 将输入的indices0转换为任意整数类型的数组indices
    if (indices == NULL) {
        goto fail;
    }

    dtype = PyArray_DescrFromType(NPY_INTP);
    # 创建一个描述器，表示NPY_INTP类型的数组
    if (dtype == NULL) {
        goto fail;
    }

    iter = NpyIter_New(indices, NPY_ITER_READONLY|
                                NPY_ITER_ALIGNED|
                                NPY_ITER_BUFFERED|
                                NPY_ITER_ZEROSIZE_OK|
                                NPY_ITER_DONT_NEGATE_STRIDES|
                                NPY_ITER_MULTI_INDEX,
                                NPY_KEEPORDER, NPY_SAME_KIND_CASTING,
                                dtype);
    # 使用indices创建一个迭代器，设置迭代器的属性和类型
    if (iter == NULL) {
        goto fail;
    }

    /*
     * Create the return array with a layout compatible with the indices
     * and with a dimension added to the end for the multi-index
     */
    // 创建一个返回数组，其布局与indices兼容，并在末尾增加一个维度用于多索引
    ret_ndim = PyArray_NDIM(indices) + 1;
    if (NpyIter_GetShape(iter, ret_dims) != NPY_SUCCEED) {
        goto fail;
    }
    // 获取返回数组的维度，并设置最后一个维度为dimensions.len
    ret_dims[ret_ndim-1] = dimensions.len;
    if (NpyIter_CreateCompatibleStrides(iter,
                dimensions.len*sizeof(npy_intp), ret_strides) != NPY_SUCCEED) {
        goto fail;
    }
    // 创建与迭代器兼容的步幅，并设置最后一个维度的步幅为sizeof(npy_intp)

    /* Remove the multi-index and inner loop */
    // 移除多索引和内部循环
    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
        goto fail;
    }
    // 移除迭代器的多索引功能
    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
        goto fail;
    }
    // 启用外部循环功能

    ret_arr = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                            ret_ndim, ret_dims, ret_strides, NULL, 0, NULL);
    // 使用描述器创建一个新的PyArrayObject作为返回数组
    dtype = NULL;
    if (ret_arr == NULL) {
        goto fail;
    }

    if (order != NPY_CORDER && order != NPY_FORTRANORDER) {
        PyErr_SetString(PyExc_ValueError,
                        "only 'C' or 'F' order is permitted");
        goto fail;
    }
    // 检查order是否为NPY_CORDER或NPY_FORTRANORDER，否则设置错误并跳转到fail标签
    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strides;
        npy_intp *countptr, count;
        npy_intp *coordsptr = (npy_intp *)PyArray_DATA(ret_arr);

        iternext = NpyIter_GetIterNext(iter, NULL);
        // 获取迭代器的下一个函数指针
        if (iternext == NULL) {
            goto fail;
        }
        // 如果获取失败则跳转到fail标签

        dataptr = NpyIter_GetDataPtrArray(iter);
        // 获取迭代器中数据指针的数组
        strides = NpyIter_GetInnerStrideArray(iter);
        // 获取迭代器中内部步幅的数组
        countptr = NpyIter_GetInnerLoopSizePtr(iter);
        // 获取迭代器中内部循环大小的指针

        do {
            count = *countptr;
            // 获取当前内部循环的大小
            if (unravel_index_loop(dimensions.len, dimensions.ptr,
                                   unravel_size, count, *dataptr, *strides,
                                   coordsptr, order) != NPY_SUCCEED) {
                goto fail;
            }
            // 对当前内部循环进行展开索引操作，如果失败则跳转到fail标签
            coordsptr += count * dimensions.len;
            // 更新coordsptr，移动到下一个内部循环的开始位置
        } while (iternext(iter));
        // 使用iternext函数进行迭代，直到迭代结束
    }
    /*
     * 如果 dimensions.len 为 0 且 indices 的维度不为 0，
     * 表示对于零维数组没有索引意义上的“取唯一元素十次”操作，
     * 因此我们别无选择只能报错。（参见 gh-580）
     *
     * 在迭代完成后进行这个检查，这样可以为无效索引提供更好的错误消息。
     */
    if (dimensions.len == 0 && PyArray_NDIM(indices) != 0) {
        PyErr_SetString(PyExc_ValueError,
                "multiple indices are not supported for 0d arrays");
        goto fail;
    }

    /* 现在根据每个索引创建视图的元组 */
    ret_tuple = PyTuple_New(dimensions.len);
    if (ret_tuple == NULL) {
        goto fail;
    }
    for (i = 0; i < dimensions.len; ++i) {
        PyArrayObject *view;

        view = (PyArrayObject *)PyArray_NewFromDescrAndBase(
                &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
                ret_ndim - 1, ret_dims, ret_strides,
                PyArray_BYTES(ret_arr) + i*sizeof(npy_intp),
                NPY_ARRAY_WRITEABLE, NULL, (PyObject *)ret_arr);
        if (view == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(ret_tuple, i, PyArray_Return(view));
    }

    Py_DECREF(ret_arr);
    Py_XDECREF(indices);
    npy_free_cache_dim_obj(dimensions);
    NpyIter_Deallocate(iter);

    // 返回创建的视图元组
    return ret_tuple;
fail:
    // 释放 ret_tuple 所引用的 Python 对象，减少其引用计数
    Py_XDECREF(ret_tuple);
    // 释放 ret_arr 所引用的 Python 对象，减少其引用计数
    Py_XDECREF(ret_arr);
    // 释放 dtype 所引用的 Python 对象，减少其引用计数
    Py_XDECREF(dtype);
    // 释放 indices 所引用的 Python 对象，减少其引用计数
    Py_XDECREF(indices);
    // 释放之前分配的维度缓存对象
    npy_free_cache_dim_obj(dimensions);
    // 释放 NpyIter 对象及其内部资源
    NpyIter_Deallocate(iter);
    // 返回 NULL 指针，表示失败
    return NULL;
}

/* Can only be called if doc is currently NULL */
NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len_args)
{
    PyObject *obj;
    PyObject *str;
    const char *docstr;
    static char *msg = "already has a different docstring";

    /* Don't add docstrings */
    // 如果 Python 解释器版本支持静态优化并且已启用，直接返回 None
#if PY_VERSION_HEX > 0x030b0000
    if (npy_static_cdata.optimize > 1) {
#else
    if (Py_OptimizeFlag > 1) {
#endif
        Py_RETURN_NONE;
    }

    // 准备解析参数
    NPY_PREPARE_ARGPARSER;
    // 解析参数，期望两个参数：一个对象和一个字符串
    if (npy_parse_arguments("add_docstring", args, len_args, NULL,
            "", NULL, &obj,
            "", NULL, &str,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    // 确保传入的字符串参数是一个 Unicode 对象
    if (!PyUnicode_Check(str)) {
        PyErr_SetString(PyExc_TypeError,
                "argument docstring of add_docstring should be a str");
        return NULL;
    }

    // 将 Unicode 字符串转换为 UTF-8 格式的 C 字符串
    docstr = PyUnicode_AsUTF8(str);
    if (docstr == NULL) {
        return NULL;
    }

    // 定义宏 _ADDDOC，用于设置对象的文档字符串
#define _ADDDOC(doc, name)                                              \
        if (!(doc)) {                                                   \
            doc = docstr;                                               \
            Py_INCREF(str);  /* hold on to string (leaks reference) */  \
        }                                                               \
        else if (strcmp(doc, docstr) != 0) {                            \
            PyErr_Format(PyExc_RuntimeError, "%s method %s", name, msg); \
            return NULL;                                                \
        }

    // 根据对象类型设置相应的文档字符串
    if (Py_TYPE(obj) == &PyCFunction_Type) {
        // 如果是 C 函数对象，则设置其 ml_doc 字段为传入的文档字符串
        PyCFunctionObject *new = (PyCFunctionObject *)obj;
        _ADDDOC(new->m_ml->ml_doc, new->m_ml->ml_name);
    }
    else if (PyObject_TypeCheck(obj, &PyType_Type)) {
        // 如果是 Python 类型对象，则设置其 tp_doc 字段和 __doc__ 字典项为传入的文档字符串
        PyTypeObject *new = (PyTypeObject *)obj;
        _ADDDOC(new->tp_doc, new->tp_name);
        // 如果 tp_dict 不为空且 __doc__ 项为 None，则替换为传入的文档字符串
        if (new->tp_dict != NULL && PyDict_CheckExact(new->tp_dict) &&
                PyDict_GetItemString(new->tp_dict, "__doc__") == Py_None) {
            // 警告：修改 tp_dict 不一定安全！
            if (PyDict_SetItemString(new->tp_dict, "__doc__", str) < 0) {
                return NULL;
            }
        }
    }
    # 如果对象的类型是 PyMemberDescr_Type
    else if (Py_TYPE(obj) == &PyMemberDescr_Type) {
        # 将 obj 转换为 PyMemberDescrObject 类型
        PyMemberDescrObject *new = (PyMemberDescrObject *)obj;
        # 调用 _ADDDOC 函数，将 new 对象的成员变量文档和名称传入
        _ADDDOC(new->d_member->doc, new->d_member->name);
    }
    # 如果对象的类型是 PyGetSetDescr_Type
    else if (Py_TYPE(obj) == &PyGetSetDescr_Type) {
        # 将 obj 转换为 PyGetSetDescrObject 类型
        PyGetSetDescrObject *new = (PyGetSetDescrObject *)obj;
        # 调用 _ADDDOC 函数，将 new 对象的获取设置文档和名称传入
        _ADDDOC(new->d_getset->doc, new->d_getset->name);
    }
    # 如果对象的类型是 PyMethodDescr_Type
    else if (Py_TYPE(obj) == &PyMethodDescr_Type) {
        # 将 obj 转换为 PyMethodDescrObject 类型
        PyMethodDescrObject *new = (PyMethodDescrObject *)obj;
        # 调用 _ADDDOC 函数，将 new 对象的方法文档和名称传入
        _ADDDOC(new->d_method->ml_doc, new->d_method->ml_name);
    }
    # 如果对象不属于上述三种类型
    else {
        PyObject *doc_attr;

        # 获取对象的 "__doc__" 属性
        doc_attr = PyObject_GetAttrString(obj, "__doc__");
        # 如果获取成功且不为 None，并且对象的 __doc__ 属性与 str 不相等
        if (doc_attr != NULL && doc_attr != Py_None &&
                (PyUnicode_Compare(doc_attr, str) != 0)) {
            Py_DECREF(doc_attr);
            # 如果在比较时发生错误
            if (PyErr_Occurred()) {
                /* error during PyUnicode_Compare */
                return NULL;
            }
            # 抛出运行时错误，指明对象的信息
            PyErr_Format(PyExc_RuntimeError, "object %s", msg);
            return NULL;
        }
        Py_XDECREF(doc_attr);

        # 将对象的 "__doc__" 属性设置为 str
        if (PyObject_SetAttrString(obj, "__doc__", str) < 0) {
            # 如果设置失败，抛出类型错误
            PyErr_SetString(PyExc_TypeError,
                            "Cannot set a docstring for that object");
            return NULL;
        }
        # 返回 None
        Py_RETURN_NONE;
    }
/*
 * 返回一个 Python None 对象，表示函数执行完毕没有返回值。
 */
Py_RETURN_NONE;
}

/*
 * 此函数将输入数组中的布尔值打包到字节数组的位中。布尔真值按常规方式确定：0为假，其他值为真。
 */
static NPY_GCC_OPT_3 inline void
pack_inner(const char *inptr,
           npy_intp element_size,   /* 每个元素的大小，以字节为单位 */
           npy_intp n_in,           /* 输入数组中的元素数量 */
           npy_intp in_stride,      /* 输入数组的步幅 */
           char *outptr,
           npy_intp n_out,          /* 输出数组的元素数量 */
           npy_intp out_stride,     /* 输出数组的步幅 */
           PACK_ORDER order)        /* 打包顺序 */
{
    /*
     * 遍历 inptr 的元素。
     * 确定它是否为非零值。
     *   是：设置对应的位（并调整构建值）
     *   否：继续下一个元素
     * 每8个值，设置构建值并递增 outptr
     */
    npy_intp index = 0;
    int remain = n_in % 8;              /* 不均匀的位数 */

#if NPY_SIMD
    // 检查条件：输入步长为1、元素大小为1字节、输出数量大于2时进入条件块
    if (in_stride == 1 && element_size == 1 && n_out > 2) {
        // 创建全零的8位整数向量
        npyv_u8 v_zero = npyv_zero_u8();
        /* 不处理非完整的8字节余数 */
        // 计算有效输出数量，排除余数的影响
        npy_intp vn_out = n_out - (remain ? 1 : 0);
        // 设置向量步长为64位
        const int vstep = npyv_nlanes_u64;
        // 设置4倍向量步长
        const int vstepx4 = vstep * 4;
        // 检查输出指针是否按64位对齐
        const int isAligned = npy_is_aligned(outptr, sizeof(npy_uint64));
        // 调整有效输出数量，使其按向量步长对齐
        vn_out -= (vn_out & (vstep - 1));
        // 循环处理主体，以向量步长x4为单位处理数据
        for (; index <= vn_out - vstepx4; index += vstepx4, inptr += npyv_nlanes_u8 * 4) {
            // 加载4个8位整数向量
            npyv_u8 v0 = npyv_load_u8((const npy_uint8*)inptr);
            npyv_u8 v1 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 1);
            npyv_u8 v2 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 2);
            npyv_u8 v3 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 3);
            // 如果顺序为大端，反转每个8位整数向量
            if (order == PACK_ORDER_BIG) {
                v0 = npyv_rev64_u8(v0);
                v1 = npyv_rev64_u8(v1);
                v2 = npyv_rev64_u8(v2);
                v3 = npyv_rev64_u8(v3);
            }
            // 定义一个64位整数数组
            npy_uint64 bb[4];
            // 将每个8位整数向量转换为比特表示，并存储到数组中
            bb[0] = npyv_tobits_b8(npyv_cmpneq_u8(v0, v_zero));
            bb[1] = npyv_tobits_b8(npyv_cmpneq_u8(v1, v_zero));
            bb[2] = npyv_tobits_b8(npyv_cmpneq_u8(v2, v_zero));
            bb[3] = npyv_tobits_b8(npyv_cmpneq_u8(v3, v_zero));
            // 如果输出步长为1且满足对齐要求或者不需要对齐
            if(out_stride == 1 && 
                (!NPY_ALIGNMENT_REQUIRED || isAligned)) {
                // 强制类型转换为64位整数指针
                npy_uint64 *ptr64 = (npy_uint64*)outptr;
                // 根据 SIMD 宽度执行不同的位运算和存储操作
                #if NPY_SIMD_WIDTH == 16
                    npy_uint64 bcomp = bb[0] | (bb[1] << 16) | (bb[2] << 32) | (bb[3] << 48);
                    ptr64[0] = bcomp;
                #elif NPY_SIMD_WIDTH == 32
                    ptr64[0] = bb[0] | (bb[1] << 32);
                    ptr64[1] = bb[2] | (bb[3] << 32);
                #else
                    ptr64[0] = bb[0]; ptr64[1] = bb[1];
                    ptr64[2] = bb[2]; ptr64[3] = bb[3];
                #endif
                // 更新输出指针位置
                outptr += vstepx4;
            } else {
                // 如果输出步长不为1或者需要对齐，逐位复制数据到输出指针
                for(int i = 0; i < 4; i++) {
                    for (int j = 0; j < vstep; j++) {
                        memcpy(outptr, (char*)&bb[i] + j, 1);
                        outptr += out_stride;
                    }
                }
            }
        }
        // 处理剩余的向量元素，不足一组的部分
        for (; index < vn_out; index += vstep, inptr += npyv_nlanes_u8) {
            // 加载一个8位整数向量
            npyv_u8 va = npyv_load_u8((const npy_uint8*)inptr);
            // 如果顺序为大端，反转该8位整数向量
            if (order == PACK_ORDER_BIG) {
                va = npyv_rev64_u8(va);
            }
            // 将该8位整数向量转换为比特表示
            npy_uint64 bb = npyv_tobits_b8(npyv_cmpneq_u8(va, v_zero));
            // 逐位复制数据到输出指针
            for (int i = 0; i < vstep; ++i) {
                memcpy(outptr, (char*)&bb + i, 1);
                outptr += out_stride;
            }
        }
    }
#endif

    if (remain == 0) {                  /* assumes n_in > 0 */
        remain = 8;
    }
    /* 不重置索引，只处理上面代码块的剩余部分 */
    for (; index < n_out; index++) {
        unsigned char build = 0;
        int maxi = (index == n_out - 1) ? remain : 8;
        if (order == PACK_ORDER_BIG) {
            // 大端序
            for (int i = 0; i < maxi; i++) {
                build <<= 1;
                for (npy_intp j = 0; j < element_size; j++) {
                    build |= (inptr[j] != 0);
                }
                inptr += in_stride;
            }
            if (index == n_out - 1) {
                build <<= 8 - remain;
            }
        }
        else
        {
            // 小端序
            for (int i = 0; i < maxi; i++) {
                build >>= 1;
                for (npy_intp j = 0; j < element_size; j++) {
                    build |= (inptr[j] != 0) ? 128 : 0;
                }
                inptr += in_stride;
            }
            if (index == n_out - 1) {
                build >>= 8 - remain;
            }
        }
        *outptr = (char)build;
        outptr += out_stride;
    }
}

static PyObject *
pack_bits(PyObject *input, int axis, char order)
{
    PyArrayObject *inp;
    PyArrayObject *new = NULL;
    PyArrayObject *out = NULL;
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    NPY_BEGIN_THREADS_DEF;

    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        return NULL;
    }
    if (!PyArray_ISBOOL(inp) && !PyArray_ISINTEGER(inp)) {
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of integer or boolean data type");
        Py_DECREF(inp);
        goto fail;
    }

    new = (PyArrayObject *)PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(new) == 0) {
        char *optr, *iptr;

        out = (PyArrayObject *)PyArray_NewFromDescr(
                Py_TYPE(new), PyArray_DescrFromType(NPY_UBYTE),
                0, NULL, NULL, NULL,
                0, NULL);
        if (out == NULL) {
            goto fail;
        }
        optr = PyArray_DATA(out);
        iptr = PyArray_DATA(new);
        *optr = 0;
        for (i = 0; i < PyArray_ITEMSIZE(new); i++) {
            if (*iptr != 0) {
                *optr = 1;
                break;
            }
            iptr++;
        }
        goto finish;
    }


    /* 设置输出形状 */
    for (i = 0; i < PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    /*
     * 将轴的维度除以8
     * 8 -> 1, 9 -> 2, 16 -> 2, 17 -> 3 等等..
     */
    outdims[axis] = ((outdims[axis] - 1) >> 3) + 1;

    /* 创建输出数组 */
    out = (PyArrayObject *)PyArray_NewFromDescr(
            Py_TYPE(new), PyArray_DescrFromType(NPY_UBYTE),
            PyArray_NDIM(new), outdims, NULL, NULL,
            PyArray_ISFORTRAN(new), NULL);
    if (out == NULL) {
        goto fail;
    }
    # 设置迭代器以便在除了给定轴以外的所有维度上进行迭代
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)new, &axis);
    ot = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)out, &axis);
    # 如果迭代器初始化失败，则释放资源并跳转到错误处理部分
    if (it == NULL || ot == NULL) {
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }
    # 根据给定的字节序列创建一个枚举类型，'b'表示大端序，'l'表示小端序
    const PACK_ORDER ordere = order == 'b' ? PACK_ORDER_BIG : PACK_ORDER_LITTLE;
    # 多线程处理的起始点，根据输出数组在给定轴上的维度进行决定
    NPY_BEGIN_THREADS_THRESHOLDED(PyArray_DIM(out, axis));
    # 迭代处理数组中的每个元素
    while (PyArray_ITER_NOTDONE(it)) {
        # 调用内部函数，对数组中的数据进行打包处理
        pack_inner(PyArray_ITER_DATA(it), PyArray_ITEMSIZE(new),
                   PyArray_DIM(new, axis), PyArray_STRIDE(new, axis),
                   PyArray_ITER_DATA(ot), PyArray_DIM(out, axis),
                   PyArray_STRIDE(out, axis), ordere);
        # 移动到下一个元素
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    # 多线程处理结束
    NPY_END_THREADS;

    # 释放迭代器对象
    Py_DECREF(it);
    Py_DECREF(ot);
    // 释放新建对象的 Python 引用计数，避免内存泄漏
    Py_DECREF(new);
    // 返回输出对象的 PyObject 指针类型，完成函数执行
    return (PyObject *)out;

fail:
    // 出错处理：释放新建对象的 Python 引用计数
    Py_XDECREF(new);
    // 出错处理：释放输出对象的 Python 引用计数
    Py_XDECREF(out);
    // 出错处理：返回空指针表示函数执行失败
    return NULL;
}



static PyObject *
unpack_bits(PyObject *input, int axis, PyObject *count_obj, char order)
{
    PyArrayObject *inp;
    PyArrayObject *new = NULL;
    PyArrayObject *out = NULL;
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    npy_intp count, in_n, in_tail, out_pad, in_stride, out_stride;
    NPY_BEGIN_THREADS_DEF;

    // 从 Python 对象创建 PyArrayObject，转换为 NumPy 数组
    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        // 如果转换失败，返回空指针表示函数执行失败
        return NULL;
    }
    if (PyArray_TYPE(inp) != NPY_UBYTE) {
        // 如果输入数组不是无符号字节类型，设置错误信息并释放输入数组
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of unsigned byte data type");
        Py_DECREF(inp);
        // 转到出错处理标签
        goto fail;
    }

    // 检查轴的有效性，并返回新的 PyArrayObject
    new = (PyArrayObject *)PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        // 如果检查轴失败，返回空指针表示函数执行失败
        return NULL;
    }

    if (PyArray_NDIM(new) == 0) {
        // 处理0维数组，将其转换为1维数组
        PyArrayObject *temp;
        PyArray_Dims newdim = {NULL, 1};
        npy_intp shape = 1;

        newdim.ptr = &shape;
        // 创建新形状的数组，并释放原数组
        temp = (PyArrayObject *)PyArray_Newshape(new, &newdim, NPY_CORDER);
        Py_DECREF(new);
        if (temp == NULL) {
            // 如果创建新形状数组失败，返回空指针表示函数执行失败
            return NULL;
        }
        new = temp;
    }

    // 设置输出数组的形状
    for (i = 0; i < PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    // 将指定轴的维度乘以8
    outdims[axis] *= 8;
    if (count_obj != Py_None) {
        // 如果 count_obj 不是 None，将其转换为整数并检查
        count = PyArray_PyIntAsIntp(count_obj);
        if (error_converting(count)) {
            // 如果转换失败，转到出错处理标签
            goto fail;
        }
        if (count < 0) {
            // 如果 count 是负数，调整输出数组的维度
            outdims[axis] += count;
            if (outdims[axis] < 0) {
                // 如果调整后的维度小于0，设置错误信息并转到出错处理标签
                PyErr_Format(PyExc_ValueError,
                             "-count larger than number of elements");
                goto fail;
            }
        }
        else {
            // 如果 count 是非负数，设置输出数组的维度为 count
            outdims[axis] = count;
        }
    }

    // 创建输出数组
    out = (PyArrayObject *)PyArray_NewFromDescr(
            Py_TYPE(new), PyArray_DescrFromType(NPY_UBYTE),
            PyArray_NDIM(new), outdims, NULL, NULL,
            PyArray_ISFORTRAN(new), NULL);
    if (out == NULL) {
        // 如果创建输出数组失败，转到出错处理标签
        goto fail;
    }

    // 设置迭代器，用于遍历除指定轴以外的所有维度
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)new, &axis);
    ot = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)out, &axis);
    if (it == NULL || ot == NULL) {
        // 如果设置迭代器失败，释放迭代器并转到出错处理标签
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }

    // 计算输入数据的位数和填充要求
    count = PyArray_DIM(new, axis) * 8;
    if (outdims[axis] > count) {
        in_n = count / 8;
        in_tail = 0;
        out_pad = outdims[axis] - count;
    }
    else {
        in_n = outdims[axis] / 8;
        in_tail = outdims[axis] % 8;
        out_pad = 0;
    }

    // 计算输入和输出数组的步幅
    in_stride = PyArray_STRIDE(new, axis);
    out_stride = PyArray_STRIDE(out, axis);
    # 根据输出数组的大小设置线程阈值，用于多线程处理
    NPY_BEGIN_THREADS_THRESHOLDED(PyArray_Size((PyObject *)out) / 8);

    # 迭代器循环，直到迭代结束
    while (PyArray_ITER_NOTDONE(it)) {
        npy_intp index;
        # 获取输入迭代器的当前数据指针
        unsigned const char *inptr = PyArray_ITER_DATA(it);
        # 获取输出迭代器的当前数据指针
        char *outptr = PyArray_ITER_DATA(ot);

        # 如果输出步长为1
        if (out_stride == 1) {
            /* 对于单位步长，可以直接从查找表中复制数据 */
            if (order == 'b') {
                # 按字节顺序（大端）循环处理输入数据
                for (index = 0; index < in_n; index++) {
                    # 从静态数据结构中查找并获取64位整数，复制到输出指针位置
                    npy_uint64 v = npy_static_cdata.unpack_lookup_big[*inptr].uint64;
                    memcpy(outptr, &v, 8);
                    outptr += 8;
                    inptr += in_stride;
                }
            }
            else {
                # 按字节顺序（小端或者未指定顺序）循环处理输入数据
                for (index = 0; index < in_n; index++) {
                    npy_uint64 v = npy_static_cdata.unpack_lookup_big[*inptr].uint64;
                    # 如果顺序不是大端，进行字节交换
                    if (order != 'b') {
                        v = npy_bswap8(v);
                    }
                    memcpy(outptr, &v, 8);
                    outptr += 8;
                    inptr += in_stride;
                }
            }
            /* 清理尾部剩余部分 */
            if (in_tail) {
                npy_uint64 v = npy_static_cdata.unpack_lookup_big[*inptr].uint64;
                if (order != 'b') {
                    v = npy_bswap8(v);
                }
                memcpy(outptr, &v, in_tail);
            }
            /* 添加填充 */
            else if (out_pad) {
                memset(outptr, 0, out_pad);
            }
        }
        else {
            # 如果输出步长不为1
            if (order == 'b') {
                # 按字节顺序（大端）循环处理输入数据
                for (index = 0; index < in_n; index++) {
                    # 每个字节内的位处理
                    for (i = 0; i < 8; i++) {
                        *outptr = ((*inptr & (128 >> i)) != 0);
                        outptr += out_stride;
                    }
                    inptr += in_stride;
                }
                /* 清理尾部剩余部分 */
                for (i = 0; i < in_tail; i++) {
                    *outptr = ((*inptr & (128 >> i)) != 0);
                    outptr += out_stride;
                }
            }
            else {
                # 按字节顺序（小端或者未指定顺序）循环处理输入数据
                for (index = 0; index < in_n; index++) {
                    # 每个字节内的位处理
                    for (i = 0; i < 8; i++) {
                        *outptr = ((*inptr & (1 << i)) != 0);
                        outptr += out_stride;
                    }
                    inptr += in_stride;
                }
                /* 清理尾部剩余部分 */
                for (i = 0; i < in_tail; i++) {
                    *outptr = ((*inptr & (1 << i)) != 0);
                    outptr += out_stride;
                }
            }
            /* 添加填充 */
            for (index = 0; index < out_pad; index++) {
                *outptr = 0;
                outptr += out_stride;
            }
        }

        # 移动输入和输出迭代器到下一个位置
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    # 结束多线程处理
    NPY_END_THREADS;

    # 释放迭代器对象
    Py_DECREF(it);
    Py_DECREF(ot);

    # 释放新创建的数组对象
    Py_DECREF(new);

    # 返回输出数组对象
    return (PyObject *)out;
fail:
    // 减少 new 指针的引用计数，处理异常情况
    Py_XDECREF(new);
    // 减少 out 指针的引用计数，处理异常情况
    Py_XDECREF(out);
    // 返回空指针，表示函数执行失败
    return NULL;
}


NPY_NO_EXPORT PyObject *
io_pack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;                      // 输入的 Python 对象
    int axis = NPY_RAVEL_AXIS;          // 指定的轴，默认为展平
    static char *kwlist[] = {"in", "axis", "bitorder", NULL};  // 关键字参数列表
    char c = 'b';                       // 默认的字节顺序为 'b' (big-endian)
    const char * order_str = NULL;      // 指定的字节顺序字符串

    // 解析 Python 的参数元组和关键字参数
    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&s:pack" , kwlist,
                &obj, PyArray_AxisConverter, &axis, &order_str)) {
        // 解析失败时返回空指针
        return NULL;
    }

    // 如果指定了字节顺序字符串，则根据字符串设置字节顺序
    if (order_str != NULL) {
        if (strncmp(order_str, "little", 6) == 0)
            c = 'l';    // 如果字符串为 "little"，设置为小端序
        else if (strncmp(order_str, "big", 3) == 0)
            c = 'b';    // 如果字符串为 "big"，设置为大端序
        else {
            // 如果字符串既不是 "little" 也不是 "big"，抛出数值错误异常
            PyErr_SetString(PyExc_ValueError,
                    "'order' must be either 'little' or 'big'");
            return NULL;
        }
    }

    // 调用 pack_bits 函数进行打包操作，并返回其结果
    return pack_bits(obj, axis, c);
}


NPY_NO_EXPORT PyObject *
io_unpack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;                      // 输入的 Python 对象
    int axis = NPY_RAVEL_AXIS;          // 指定的轴，默认为展平
    PyObject *count = Py_None;          // 解包的位数
    static char *kwlist[] = {"in", "axis", "count", "bitorder", NULL};  // 关键字参数列表
    const char * c = NULL;              // 指定的字节顺序字符串

    // 解析 Python 的参数元组和关键字参数
    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&Os:unpack" , kwlist,
                &obj, PyArray_AxisConverter, &axis, &count, &c)) {
        // 解析失败时返回空指针
        return NULL;
    }

    // 如果没有指定字节顺序，则默认为 'b' (big-endian)
    if (c == NULL) {
        c = "b";
    }

    // 检查指定的字节顺序是否有效，必须以 'l' 或 'b' 开头
    if (c[0] != 'l' && c[0] != 'b') {
        // 如果不是有效的顺序，抛出数值错误异常
        PyErr_SetString(PyExc_ValueError,
                    "'order' must begin with 'l' or 'b'");
        return NULL;
    }

    // 调用 unpack_bits 函数进行解包操作，并返回其结果
    return unpack_bits(obj, axis, count, c[0]);
}
```