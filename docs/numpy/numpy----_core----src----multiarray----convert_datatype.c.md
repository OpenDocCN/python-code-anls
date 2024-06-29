# `.\numpy\numpy\_core\src\multiarray\convert_datatype.c`

```py
/*
 * 定义宏，指定使用的 NumPy API 版本
 * 禁用已弃用的 NumPy API
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

/*
 * 清除 PY_SSIZE_T_CLEAN 宏定义
 * 包含 Python.h 头文件，引入 Python 运行时支持
 * 引入 structmember.h，用于定义结构体成员访问
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

/*
 * 引入 NumPy 头文件，用于处理数组对象
 * 引入数组标量的头文件
 */
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

/*
 * 引入 NumPy 配置头文件
 * 引入低级分步循环头文件
 */
#include "npy_config.h"
#include "lowlevel_strided_loops.h"

/*
 * 引入 Python 兼容性头文件
 * 引入 NumPy 数学函数头文件
 */
#include "npy_pycompat.h"
#include "numpy/npy_math.h"

/*
 * 引入数组强制转换相关头文件
 * 引入类型转换表头文件
 * 引入通用工具函数头文件
 * 引入构造函数头文件
 * 引入描述符头文件
 * 引入数据类型元数据头文件
 */
#include "array_coercion.h"
#include "can_cast_table.h"
#include "common.h"
#include "ctors.h"
#include "descriptor.h"
#include "dtypemeta.h"

/*
 * 引入标量类型头文件
 * 引入映射头文件
 * 引入旧数据类型实现头文件
 * 引入字符串数据类型头文件
 */
#include "scalartypes.h"
#include "mapping.h"
#include "legacy_dtype_implementation.h"
#include "stringdtype/dtype.h"

/*
 * 引入抽象数据类型头文件
 * 引入数据类型转换头文件
 * 引入日期时间处理头文件
 * 引入日期时间字符串处理头文件
 * 引入数组方法头文件
 * 引入用户自定义类型头文件
 * 引入数据类型转移头文件
 * 引入数据类型遍历头文件
 * 引入数组对象头文件
 * 引入 NumPy 静态数据头文件
 * 引入多维数组模块头文件
 */
#include "abstractdtypes.h"
#include "convert_datatype.h"
#include "_datetime.h"
#include "datetime_strings.h"
#include "array_method.h"
#include "usertypes.h"
#include "dtype_transfer.h"
#include "dtype_traversal.h"
#include "arrayobject.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"

/*
 * 定义必要的字符串长度数组
 * 用于不同大小的无符号整数类型转换时的字符串长度需求
 */
/*
 * Required length of string when converting from unsigned integer type.
 * Array index is integer size in bytes.
 * - 3 chars needed for cast to max value of 255 or 127
 * - 5 chars needed for cast to max value of 65535 or 32767
 * - 10 chars needed for cast to max value of 4294967295 or 2147483647
 * - 20 chars needed for cast to max value of 18446744073709551615
 *   or 9223372036854775807
 */
NPY_NO_EXPORT npy_intp REQUIRED_STR_LEN[] = {0, 3, 5, 10, 10, 20, 20, 20, 20};

/*
 * 定义线程局部存储的 NumPy 促进状态变量，初始为使用传统的促进策略
 */
static NPY_TLS int npy_promotion_state = NPY_USE_LEGACY_PROMOTION;

/*
 * 获取当前 NumPy 促进状态的函数
 */
NPY_NO_EXPORT int
get_npy_promotion_state() {
    return npy_promotion_state;
}

/*
 * 设置 NumPy 促进状态的函数
 */
NPY_NO_EXPORT void
set_npy_promotion_state(int new_promotion_state) {
    npy_promotion_state = new_promotion_state;
}

/*
 * 获取本地上下文中是否应给出促进警告的函数
 */
NPY_NO_EXPORT int
npy_give_promotion_warnings(void)
{
    PyObject *val;

    /*
     * 缓存导入 "numpy._core._ufunc_config" 模块的 NO_NEP50_WARNING 属性
     */
    npy_cache_import(
            "numpy._core._ufunc_config", "NO_NEP50_WARNING",
            &npy_thread_unsafe_state.NO_NEP50_WARNING);

    /*
     * 如果无法导入 NO_NEP50_WARNING 属性，则写入不可提升的错误信息，并返回 1
     */
    if (npy_thread_unsafe_state.NO_NEP50_WARNING == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 1;
    }

    /*
     * 获取 NO_NEP50_WARNING 属性的值
     */
    if (PyContextVar_Get(npy_thread_unsafe_state.NO_NEP50_WARNING,
                         Py_False, &val) < 0) {
        /*
         * 如果发生错误，则写入不可提升的错误信息，并返回 1
         */
        PyErr_WriteUnraisable(NULL);
        return 1;
    }
    Py_DECREF(val);
    
    /*
     * 只有在 no-warnings 上下文为 false 时，才给出警告
     */
    return val == Py_False;
}

/*
 * 获取 NumPy 促进状态的 Python 对象表示
 */
NPY_NO_EXPORT PyObject *
npy__get_promotion_state(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(arg)) {
    int promotion_state = get_npy_promotion_state();
    /*
     * 如果当前状态为 NPY_USE_WEAK_PROMOTION，则返回字符串 "weak"
     */
    if (promotion_state == NPY_USE_WEAK_PROMOTION) {
        return PyUnicode_FromString("weak");
    }

    /*
     * 返回空对象，表示没有特定的促进状态
     */
    return Py_None;
}
    else if (promotion_state == NPY_USE_WEAK_PROMOTION_AND_WARN) {
        // 如果 promotion_state 等于 NPY_USE_WEAK_PROMOTION_AND_WARN，则返回字符串 "weak_and_warn"
        return PyUnicode_FromString("weak_and_warn");
    }
    else if (promotion_state == NPY_USE_LEGACY_PROMOTION) {
        // 如果 promotion_state 等于 NPY_USE_LEGACY_PROMOTION，则返回字符串 "legacy"
        return PyUnicode_FromString("legacy");
    }
    // 如果 promotion_state 不是上述两种状态，则抛出系统错误异常，错误消息为 "invalid promotion state!"
    PyErr_SetString(PyExc_SystemError, "invalid promotion state!");
    // 返回空指针，表示函数执行失败
    return NULL;
/**
 * 设置 NumPy 的类型提升状态。
 *
 * @param mod 空指针，未使用，仅为兼容性考虑。
 * @param arg 用于设置提升状态的参数，必须为字符串类型。
 * @returns None 或者 NULL，如果参数类型不正确则返回错误信息。
 */
NPY_NO_EXPORT PyObject *
npy__set_promotion_state(PyObject *NPY_UNUSED(mod), PyObject *arg)
{
    // 检查参数是否为 Unicode 字符串类型
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError,
                "_set_promotion_state() argument or NPY_PROMOTION_STATE "
                "must be a string.");
        return NULL;
    }
    int new_promotion_state;
    // 根据参数字符串值设置新的提升状态
    if (PyUnicode_CompareWithASCIIString(arg, "weak") == 0) {
        new_promotion_state = NPY_USE_WEAK_PROMOTION;
    }
    else if (PyUnicode_CompareWithASCIIString(arg, "weak_and_warn") == 0) {
        new_promotion_state = NPY_USE_WEAK_PROMOTION_AND_WARN;
    }
    else if (PyUnicode_CompareWithASCIIString(arg, "legacy") == 0) {
        new_promotion_state = NPY_USE_LEGACY_PROMOTION;
    }
    else {
        // 如果参数不符合预期的值，返回类型错误，并指出预期的值范围
        PyErr_Format(PyExc_TypeError,
                "_set_promotion_state() argument or NPY_PROMOTION_STATE must be "
                "'weak', 'legacy', or 'weak_and_warn' but got '%.100S'", arg);
        return NULL;
    }
    // 调用设置 NumPy 提升状态的函数
    set_npy_promotion_state(new_promotion_state);
    // 返回 None 表示成功执行
    Py_RETURN_NONE;
}

/**
 * 获取从一个数据类型到另一个数据类型的类型转换实现。
 *
 * @param from 起始数据类型元信息
 * @param to 目标数据类型元信息
 * @returns 如果找到对应的转换实现，则返回对应的对象；如果找不到或者发生错误，则返回 None 或 NULL，并设置错误状态。
 */
NPY_NO_EXPORT PyObject *
PyArray_GetCastingImpl(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to)
{
    PyObject *res;
    // 如果起始数据类型与目标数据类型相同，则返回其内部的转换实现
    if (from == to) {
        res = (PyObject *)NPY_DT_SLOTS(from)->within_dtype_castingimpl;
    }
    else {
        // 否则从起始数据类型的转换实现字典中获取目标数据类型的转换实现
        res = PyDict_GetItemWithError(NPY_DT_SLOTS(from)->castingimpls, (PyObject *)to);
    }
    // 如果成功获取到结果或者出现了错误，则增加引用计数并返回结果
    if (res != NULL || PyErr_Occurred()) {
        Py_XINCREF(res);
        return res;
    }
    /*
     * 下面的代码根据任意类型转换为和从对象或结构化（void）数据类型出发，
     * 动态地添加转换。
     */
    // 如果起始数据类型为 NPY_OBJECT，则获取从对象到通用数据类型的转换实现
    if (from->type_num == NPY_OBJECT) {
        res = PyArray_GetObjectToGenericCastingImpl();
    }
    // 如果目标数据类型为 NPY_OBJECT，则获取从通用数据类型到对象的转换实现
    else if (to->type_num == NPY_OBJECT) {
        res = PyArray_GetGenericToObjectCastingImpl();
    }
    // 如果起始数据类型为 NPY_VOID，则获取从 void 数据类型到通用数据类型的转换实现
    else if (from->type_num == NPY_VOID) {
        res = PyArray_GetVoidToGenericCastingImpl();
    }
    // 如果目标数据类型为 NPY_VOID，则获取从通用数据类型到 void 数据类型的转换实现
    else if (to->type_num == NPY_VOID) {
        res = PyArray_GetGenericToVoidCastingImpl();
    }
    /*
     * 拒绝非旧版（legacy）数据类型。它们需要使用新的 API 添加转换，
     * 这样做可以向起始描述符的转换实现字典中添加转换。
     */
    // 如果起始或目标数据类型不是旧版数据类型，则返回 None
    else if (!NPY_DT_is_legacy(from) || !NPY_DT_is_legacy(to)) {
        Py_RETURN_NONE;
    }
    // 如果起始和目标数据类型在旧版数据类型范围内，则报告运行时错误
    else if (from->type_num < NPY_NTYPES_LEGACY && to->type_num < NPY_NTYPES_LEGACY) {
        PyErr_Format(PyExc_RuntimeError,
                "builtin cast from %S to %S not found, this should not "
                "be possible.", from, to);
        return NULL;
    }
    // 增加引用计数并返回最终的结果对象
    Py_XINCREF(res);
    return res;
}
    else {
        // 如果 from 和 to 不相等，说明需要注册一个类型转换函数
        if (from != to) {
            /* A cast function must have been registered */
            // 获取注册的类型转换函数
            PyArray_VectorUnaryFunc *castfunc = PyArray_GetCastFunc(
                    from->singleton, to->type_num);
            // 如果找不到对应的类型转换函数，则进行处理
            if (castfunc == NULL) {
                PyErr_Clear();
                /* Remember that this cast is not possible */
                // 将此转换标记为不可行，并存入 castingimpls 字典中
                if (PyDict_SetItem(NPY_DT_SLOTS(from)->castingimpls,
                            (PyObject *) to, Py_None) < 0) {
                    return NULL;
                }
                // 返回 Py_None 表示无法进行转换
                Py_RETURN_NONE;
            }
        }

        /* PyArray_AddLegacyWrapping_CastingImpl find the correct casting level: */
        /*
         * TODO: Possibly move this to the cast registration time. But if we do
         *       that, we have to also update the cast when the casting safety
         *       is registered.
         */
        // 添加一个遗留包装的转换实现，查找正确的转换级别
        if (PyArray_AddLegacyWrapping_CastingImpl(from, to, -1) < 0) {
            return NULL;
        }
        // 返回 from 到 to 的转换实现
        return PyArray_GetCastingImpl(from, to);
    }

    // 如果 res 为 NULL，则返回 NULL
    if (res == NULL) {
        return NULL;
    }
    // 如果 from 和 to 相等，则抛出运行时错误
    if (from == to) {
        PyErr_Format(PyExc_RuntimeError,
                "Internal NumPy error, within-DType cast missing for %S!", from);
        // 释放 res 并返回 NULL
        Py_DECREF(res);
        return NULL;
    }
    // 将转换实现 res 存入 from 的 castingimpls 字典中
    if (PyDict_SetItem(NPY_DT_SLOTS(from)->castingimpls,
                (PyObject *)to, res) < 0) {
        Py_DECREF(res);
        return NULL;
    }
    // 返回转换实现 res
    return res;
/**
 * Fetch the (bound) casting implementation from one DType to another.
 *
 * @params from The source DTypeMeta object.
 * @params to The destination DTypeMeta object.
 *
 * @returns A bound casting implementation or None (or NULL for error).
 */
static PyObject *
PyArray_GetBoundCastingImpl(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to)
{
    // Fetch the casting implementation from 'from' to 'to'
    PyObject *method = PyArray_GetCastingImpl(from, to);
    if (method == NULL || method == Py_None) {
        return method;
    }

    /* TODO: Create better way to wrap method into bound method */
    // Allocate memory for PyBoundArrayMethodObject
    PyBoundArrayMethodObject *res;
    res = PyObject_New(PyBoundArrayMethodObject, &PyBoundArrayMethod_Type);
    if (res == NULL) {
        return NULL;
    }
    res->method = (PyArrayMethodObject *)method;

    // Allocate memory for dtypes array and populate it
    res->dtypes = PyMem_Malloc(2 * sizeof(PyArray_DTypeMeta *));
    if (res->dtypes == NULL) {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(from);
    res->dtypes[0] = from;
    Py_INCREF(to);
    res->dtypes[1] = to;

    return (PyObject *)res;
}


NPY_NO_EXPORT PyObject *
_get_castingimpl(PyObject *NPY_UNUSED(module), PyObject *args)
{
    PyArray_DTypeMeta *from, *to;
    if (!PyArg_ParseTuple(args, "O!O!:_get_castingimpl",
            &PyArrayDTypeMeta_Type, &from, &PyArrayDTypeMeta_Type, &to)) {
        return NULL;
    }
    return PyArray_GetBoundCastingImpl(from, to);
}


/**
 * Find the minimal cast safety level given two cast-levels as input.
 * Supports the NPY_CAST_IS_VIEW check, and should be preferred to allow
 * extending cast-levels if necessary.
 * It is not valid for one of the arguments to be -1 to indicate an error.
 *
 * @param casting1 The first cast safety level.
 * @param casting2 The second cast safety level.
 * @return The minimal casting error (can be -1).
 */
NPY_NO_EXPORT NPY_CASTING
PyArray_MinCastSafety(NPY_CASTING casting1, NPY_CASTING casting2)
{
    if (casting1 < 0 || casting2 < 0) {
        return -1;
    }
    // Determine the minimal cast safety level
    if (casting1 > casting2) {
        return casting1;
    }
    return casting2;
}


/*NUMPY_API
 * For backward compatibility
 *
 * Cast an array using typecode structure.
 * steals reference to dtype --- cannot be NULL
 *
 * This function always makes a copy of arr, even if the dtype
 * doesn't change.
 */
NPY_NO_EXPORT PyObject *
PyArray_CastToType(PyArrayObject *arr, PyArray_Descr *dtype, int is_f_order)
{
    PyObject *out;

    if (dtype == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "dtype is NULL in PyArray_CastToType");
        return NULL;
    }

    // Adapt the descriptor to match the array's dtype
    Py_SETREF(dtype, PyArray_AdaptDescriptorToArray(arr, NULL, dtype));
    if (dtype == NULL) {
        return NULL;
    }

    // Create a new array from the given dtype
    out = PyArray_NewFromDescr(Py_TYPE(arr), dtype,
                               PyArray_NDIM(arr),
                               PyArray_DIMS(arr),
                               NULL, NULL,
                               is_f_order,
                               (PyObject *)arr);

    if (out == NULL) {
        return NULL;
    }

    /* ... (remaining code omitted as it exceeds the requested scope) */
}
    # 如果使用 PyArray_CopyInto 将 arr 复制到 out 中遇到错误（返回值小于 0）
    if (PyArray_CopyInto((PyArrayObject *)out, arr) < 0) {
        # 释放 out 对象的 Python 引用
        Py_DECREF(out);
        # 返回 NULL 表示复制失败
        return NULL;
    }
    
    # 返回成功复制后的 out 对象
    return out;
/*
 * Fetches the legacy cast function. Warning, this only makes sense for legacy
 * dtypes.  Even most NumPy ones do NOT implement these anymore and the use
 * should be fully phased out.
 * The sole real purpose is supporting legacy style user dtypes.
 */
NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num)
{
    PyArray_VectorUnaryFunc *castfunc = NULL;

    // 如果 type_num 小于 NPY_NTYPES_ABI_COMPATIBLE，则使用 descr 获取的类型函数进行转换
    if (type_num < NPY_NTYPES_ABI_COMPATIBLE) {
        castfunc = PyDataType_GetArrFuncs(descr)->cast[type_num];
    }
    else {
        // 否则尝试从 castdict 中获取转换函数
        PyObject *obj = PyDataType_GetArrFuncs(descr)->castdict;
        if (obj && PyDict_Check(obj)) {
            PyObject *key;
            PyObject *cobj;

            // 使用 type_num 作为键从 castdict 中获取转换函数
            key = PyLong_FromLong(type_num);
            cobj = PyDict_GetItem(obj, key);
            Py_DECREF(key);

            // 如果获取到的对象是 PyCapsule 类型，则提取其中的指针作为转换函数
            if (cobj && PyCapsule_CheckExact(cobj)) {
                castfunc = PyCapsule_GetPointer(cobj, NULL);
                if (castfunc == NULL) {
                    return NULL;
                }
            }
        }
    }

    // 如果转换的是复数到实数且丢弃了虚部，则发出警告
    if (PyTypeNum_ISCOMPLEX(descr->type_num) &&
            !PyTypeNum_ISCOMPLEX(type_num) &&
            PyTypeNum_ISNUMBER(type_num) &&
            !PyTypeNum_ISBOOL(type_num)) {
        int ret = PyErr_WarnEx(npy_static_pydata.ComplexWarning,
                "Casting complex values to real discards "
                "the imaginary part", 1);
        if (ret < 0) {
            return NULL;
        }
    }

    // 如果成功获取到转换函数，则返回该函数
    if (castfunc) {
        return castfunc;
    }

    // 否则，报错：没有可用的转换函数
    PyErr_SetString(PyExc_ValueError,
            "No cast function available.");
    return NULL;
}

/*
 * Retrieves the casting safety level from a casting implementation method.
 * It resolves descriptors and checks for compatibility between the input
 * and output descriptors.
 */
static NPY_CASTING
_get_cast_safety_from_castingimpl(PyArrayMethodObject *castingimpl,
        PyArray_DTypeMeta *dtypes[2], PyArray_Descr *from, PyArray_Descr *to,
        npy_intp *view_offset)
{
    PyArray_Descr *descrs[2] = {from, to};
    PyArray_Descr *out_descrs[2];

    // 初始化 view_offset
    *view_offset = NPY_MIN_INTP;

    // 解析描述符并获取转换的安全级别
    NPY_CASTING casting = castingimpl->resolve_descriptors(
            castingimpl, dtypes, descrs, out_descrs, view_offset);

    // 如果解析失败，返回错误
    if (casting < 0) {
        return -1;
    }

    // 检查返回的描述符是否匹配，需要进行第二次检查
    if (out_descrs[0] != descrs[0]) {
        npy_intp from_offset = NPY_MIN_INTP;

        // 获取从 from 到 out_descrs[0] 的转换信息
        NPY_CASTING from_casting = PyArray_GetCastInfo(
                descrs[0], out_descrs[0], NULL, &from_offset);

        // 计算两次转换的最小安全级别
        casting = PyArray_MinCastSafety(casting, from_casting);

        // 如果 view_offset 不一致，说明多步转换不能是视图
        if (from_offset != *view_offset) {
            *view_offset = NPY_MIN_INTP;
        }

        // 如果转换失败，结束
        if (casting < 0) {
            goto finish;
        }
    }
    # 检查输入描述符和输出描述符的第二个元素是否不为空且不相同
    if (descrs[1] != NULL && out_descrs[1] != descrs[1]) {
        # 设置起始偏移量为最小整数值
        npy_intp from_offset = NPY_MIN_INTP;
        # 获取从输入描述符到输出描述符的转换信息，并更新起始偏移量
        NPY_CASTING from_casting = PyArray_GetCastInfo(
                descrs[1], out_descrs[1], NULL, &from_offset);
        # 确定安全的类型转换级别
        casting = PyArray_MinCastSafety(casting, from_casting);
        # 如果起始偏移量与视图偏移量不同，则视为多步转换不是视图
        if (from_offset != *view_offset) {
            /* `view_offset` differs: The multi-step cast cannot be a view. */
            *view_offset = NPY_MIN_INTP;
        }
        # 如果转换级别小于 0，则跳转至完成标签
        if (casting < 0) {
            goto finish;
        }
    }

  finish:
    # 释放输出描述符的第一个和第二个元素的引用计数
    Py_DECREF(out_descrs[0]);
    Py_DECREF(out_descrs[1]);
    /*
     * 检查较为安全的非标准返回情况。以下两种情况应该不会发生：
     * 1. 无需转换必须意味着视图偏移量为 0，除非数据类型定义了最终化函数，
     *    这意味着它在描述符上存储数据。
     * 2. 等效转换 + 0 视图偏移量通常是“无”转换的定义。然而，改变字段的顺序
     *    也可能创建不等效但是视图的描述符。
     * 注意，不安全的转换可能具有视图偏移量。例如，原则上，将 `<i8` 转换为 `<i4`
     * 是一个带有 0 偏移量的转换。
     */
    if ((*view_offset != 0 &&
         NPY_DT_SLOTS(NPY_DTYPE(from))->finalize_descr == NULL)) {
        assert(casting != NPY_NO_CASTING);
    }
    else {
        assert(casting != NPY_EQUIV_CASTING
               || (PyDataType_HASFIELDS(from) && PyDataType_HASFIELDS(to)));
    }
    # 返回最终确定的转换级别
    return casting;
/**
 * Given two dtype instances, find the correct casting safety.
 *
 * Note that in many cases, it may be preferable to fetch the casting
 * implementations fully to have them available for doing the actual cast
 * later.
 *
 * @param from The descriptor to cast from
 * @param to The descriptor to cast to (may be NULL)
 * @param to_dtype If `to` is NULL, must pass the to_dtype (otherwise this
 *        is ignored).
 * @param[out] view_offset Pointer to store the view offset
 * @return NPY_CASTING or -1 on error or if the cast is not possible.
 */
NPY_NO_EXPORT NPY_CASTING
PyArray_GetCastInfo(
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype,
        npy_intp *view_offset)
{
    // If `to` is not NULL, assign `to_dtype` to the dtype of `to`
    if (to != NULL) {
        to_dtype = NPY_DTYPE(to);
    }
    // Fetch the casting implementation method for casting from `from` to `to_dtype`
    PyObject *meth = PyArray_GetCastingImpl(NPY_DTYPE(from), to_dtype);
    // Return -1 if no valid casting method found
    if (meth == NULL) {
        return -1;
    }
    // Return -1 if the casting method is Py_None (indicating no valid cast)
    if (meth == Py_None) {
        Py_DECREF(Py_None);
        return -1;
    }

    // Cast the PyObject `meth` to PyArrayMethodObject
    PyArrayMethodObject *castingimpl = (PyArrayMethodObject *)meth;
    // Array of dtypes involved in the cast
    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(from), to_dtype};
    // Determine the casting safety using `_get_cast_safety_from_castingimpl` method
    NPY_CASTING casting = _get_cast_safety_from_castingimpl(castingimpl,
            dtypes, from, to, view_offset);
    // Decrement the reference count of `meth`
    Py_DECREF(meth);

    // Return the determined casting safety
    return casting;
}


/**
 * Check whether a cast is safe, see also `PyArray_GetCastInfo` for
 * a similar function.  Unlike GetCastInfo, this function checks the
 * `castingimpl->casting` when available.  This allows for two things:
 *
 * 1. It avoids  calling `resolve_descriptors` in some cases.
 * 2. Strings need to discover the length, but in some cases we know that the
 *    cast is valid (assuming the string length is discovered first).
 *
 * The latter means that a `can_cast` could return True, but the cast fail
 * because the parametric type cannot guess the correct output descriptor.
 * (I.e. if `object_arr.astype("S")` did _not_ inspect the objects, and the
 * user would have to guess the string length.)
 *
 * @param casting the requested casting safety.
 * @param from The descriptor to cast from
 * @param to The descriptor to cast to (may be NULL)
 * @param to_dtype If `to` is NULL, must pass the to_dtype (otherwise this
 *        is ignored).
 * @return 0 for an invalid cast, 1 for a valid and -1 for an error.
 */
NPY_NO_EXPORT int
PyArray_CheckCastSafety(NPY_CASTING casting,
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype)
{
    // If `to` is not NULL, assign `to_dtype` to the dtype of `to`
    if (to != NULL) {
        to_dtype = NPY_DTYPE(to);
    }
    // Fetch the casting implementation method for casting from `from` to `to_dtype`
    PyObject *meth = PyArray_GetCastingImpl(NPY_DTYPE(from), to_dtype);
    // Return -1 if no valid casting method found
    if (meth == NULL) {
        return -1;
    }
    // Return -1 if the casting method is Py_None (indicating no valid cast)
    if (meth == Py_None) {
        Py_DECREF(Py_None);
        return -1;
    }
    // Cast the PyObject `meth` to PyArrayMethodObject
    PyArrayMethodObject *castingimpl = (PyArrayMethodObject *)meth;

    // Check if the casting requested (`casting`) is at least as safe as `castingimpl->casting`
    if (PyArray_MinCastSafety(castingimpl->casting, casting) == casting) {
        /* No need to check using `castingimpl.resolve_descriptors()` */
        Py_DECREF(meth);
        return 1;
    }

    // Array of dtypes involved in the cast
    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(from), to_dtype};
    // Temporary variable to store view offset
    npy_intp view_offset;
    # 调用函数 _get_cast_safety_from_castingimpl 获取一个安全的类型转换值
    NPY_CASTING safety = _get_cast_safety_from_castingimpl(castingimpl,
            dtypes, from, to, &view_offset);
    # 减少 Python 对象的引用计数，释放内存
    Py_DECREF(meth);
    # 如果安全类型转换值小于 0，则返回 -1，表示类型转换不安全
    /* If casting is the smaller (or equal) safety we match */
    if (safety < 0) {
        return -1;
    }
    # 检查给定的类型转换值是否与最小安全转换值相等，返回比较结果
    return PyArray_MinCastSafety(safety, casting) == casting;
/*NUMPY_API
 *Check the type coercion rules.
 */
NPY_NO_EXPORT int
PyArray_CanCastSafely(int fromtype, int totype)
{
    /* Identity */
    // 如果源类型和目标类型相同，直接返回可以安全转换
    if (fromtype == totype) {
        return 1;
    }
    /*
     * As a micro-optimization, keep the cast table around.  This can probably
     * be removed as soon as the ufunc loop lookup is modified (presumably
     * before the 1.21 release).  It does no harm, but the main user of this
     * function is the ufunc-loop lookup calling it until a loop matches!
     *
     * (The table extends further, but is not strictly correct for void).
     * TODO: Check this!
     */
    // 如果类型在安全转换表的范围内，返回预先计算的安全转换结果
    if ((unsigned int)fromtype <= NPY_CLONGDOUBLE &&
            (unsigned int)totype <= NPY_CLONGDOUBLE) {
        return _npy_can_cast_safely_table[fromtype][totype];
    }

    // 获取源类型和目标类型的 dtype 元数据
    PyArray_DTypeMeta *from = PyArray_DTypeFromTypeNum(fromtype);
    if (from == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    PyArray_DTypeMeta *to = PyArray_DTypeFromTypeNum(totype);
    if (to == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    // 获取从源类型到目标类型的转换实现
    PyObject *castingimpl = PyArray_GetCastingImpl(from, to);
    Py_DECREF(from);
    Py_DECREF(to);

    if (castingimpl == NULL) {
        PyErr_WriteUnraisable(NULL);
        return 0;
    }
    else if (castingimpl == Py_None) {
        // 如果没有找到转换实现，返回不能安全转换
        Py_DECREF(Py_None);
        return 0;
    }
    // 获取转换的安全性级别
    NPY_CASTING safety = ((PyArrayMethodObject *)castingimpl)->casting;
    // 检查转换是否至少达到指定的安全级别
    int res = PyArray_MinCastSafety(safety, NPY_SAFE_CASTING) == NPY_SAFE_CASTING;
    Py_DECREF(castingimpl);
    return res;
}



/*NUMPY_API
 * leaves reference count alone --- cannot be NULL
 *
 * PyArray_CanCastTypeTo is equivalent to this, but adds a 'casting'
 * parameter.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastTo(PyArray_Descr *from, PyArray_Descr *to)
{
    // 直接调用 PyArray_CanCastTypeTo，并使用 NPY_SAFE_CASTING 安全级别
    return PyArray_CanCastTypeTo(from, to, NPY_SAFE_CASTING);
}


/*
 * This function returns true if the two types can be safely cast at
 * *minimum_safety* casting level. Sets the *view_offset* if that is set
 * for the cast. If ignore_error is set, the error indicator is cleared
 * if there are any errors in cast setup and returns false, otherwise
 * the error indicator is left set and returns -1.
 */
NPY_NO_EXPORT npy_intp
PyArray_SafeCast(PyArray_Descr *type1, PyArray_Descr *type2,
                 npy_intp* view_offset, NPY_CASTING minimum_safety,
                 npy_intp ignore_error)
{
    // 如果两个类型相同，直接返回可以安全转换，且视图偏移为 0
    if (type1 == type2) {
        *view_offset = 0;
        return 1;
    }

    // 获取从 type1 到 type2 的转换安全级别
    NPY_CASTING safety = PyArray_GetCastInfo(type1, type2, NULL, view_offset);
    if (safety < 0) {
        // 如果设置了忽略错误，清除错误并返回不安全转换；否则返回错误码
        if (ignore_error) {
            PyErr_Clear();
            return 0;
        }
        return -1;
    }
    // 检查转换安全级别是否至少达到指定的最小安全级别
    return PyArray_MinCastSafety(safety, minimum_safety) == minimum_safety;
}


/* Provides an ordering for the dtype 'kind' character codes */
NPY_NO_EXPORT int
dtype_kind_to_ordering(char kind)
{
    switch (kind) {
        /* Boolean kind */
        case 'b':
            // 返回布尔类型的标识
            return 0;
        /* Unsigned int kind */
        case 'u':
            // 返回无符号整数类型的标识
            return 1;
        /* Signed int kind */
        case 'i':
            // 返回有符号整数类型的标识
            return 2;
        /* Float kind */
        case 'f':
            // 返回浮点数类型的标识
            return 4;
        /* Complex kind */
        case 'c':
            // 返回复数类型的标识
            return 5;
        /* String kind */
        case 'S':
        case 'a':
            // 返回字符串类型的标识
            return 6;
        /* Unicode kind */
        case 'U':
            // 返回Unicode字符串类型的标识
            return 7;
        /* Void kind */
        case 'V':
            // 返回Void类型的标识
            return 8;
        /* Object kind */
        case 'O':
            // 返回对象类型的标识
            return 9;
        /*
         * Anything else, like datetime, is special cased to
         * not fit in this hierarchy
         */
        default:
            // 如果不属于上述任何一种类型，则返回-1，表示未知类型
            return -1;
    }
/* Converts a type number from unsigned to signed */
static int
type_num_unsigned_to_signed(int type_num)
{
    // 根据给定的类型数将无符号类型转换为有符号类型
    switch (type_num) {
        case NPY_UBYTE:
            return NPY_BYTE;
        case NPY_USHORT:
            return NPY_SHORT;
        case NPY_UINT:
            return NPY_INT;
        case NPY_ULONG:
            return NPY_LONG;
        case NPY_ULONGLONG:
            return NPY_LONGLONG;
        default:
            return type_num;  // 如果没有匹配的类型，直接返回原始类型数
    }
}


/*NUMPY_API
 * Returns true if data of type 'from' may be cast to data of type
 * 'to' according to the rule 'casting'.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting)
{
    PyArray_DTypeMeta *to_dtype = NPY_DTYPE(to);

    /*
     * NOTE: This code supports U and S, this is identical to the code
     *       in `ctors.c` which does not allow these dtypes to be attached
     *       to an array. Unlike the code for `np.array(..., dtype=)`
     *       which uses `PyArray_ExtractDTypeAndDescriptor` it rejects "m8"
     *       as a flexible dtype instance representing a DType.
     */
    /*
     * TODO: We should grow support for `np.can_cast("d", "S")` being
     *       different from `np.can_cast("d", "S0")` here, at least for
     *       the python side API.
     *       The `to = NULL` branch, which considers "S0" to be "flexible"
     *       should probably be deprecated.
     *       (This logic is duplicated in `PyArray_CanCastArrayTo`)
     */
    if (PyDataType_ISUNSIZED(to) && PyDataType_SUBARRAY(to) == NULL) {
        to = NULL;  /* consider mainly S0 and U0 as S and U */
    }

    // 检查是否可以安全地将类型 'from' 转换为类型 'to'，根据给定的转换规则 'casting'
    int is_valid = PyArray_CheckCastSafety(casting, from, to, to_dtype);
    /* Clear any errors and consider this unsafe (should likely be changed) */
    if (is_valid < 0) {
        PyErr_Clear();  // 清除任何错误状态
        return 0;       // 如果转换不安全或出错，返回假
    }
    return is_valid;    // 返回转换是否有效的结果
}


/* CanCastArrayTo needs this function */
static int min_scalar_type_num(char *valueptr, int type_num,
                                            int *is_small_unsigned);


/*
 * NOTE: This function uses value based casting logic for scalars. It will
 *       require updates when we phase out value-based-casting.
 */
NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data,
                    PyArray_Descr *to, NPY_CASTING casting)
{
    /*
     * If the two dtypes are actually references to the same object
     * or if casting type is forced unsafe then always OK.
     *
     * TODO: Assuming that unsafe casting always works is not actually correct
     */
    if (scal_type == to || casting == NPY_UNSAFE_CASTING ) {
        return 1;   // 如果两个数据类型相同或者转换类型为不安全，直接返回真
    }

    // 检查是否可以安全地将标量 'scal_type' 转换为类型 'to'，根据给定的转换规则 'casting'
    int valid = PyArray_CheckCastSafety(casting, scal_type, to, NPY_DTYPE(to));
    if (valid == 1) {
        /* This is definitely a valid cast. */
        return 1;   // 如果转换有效，返回真
    }
    if (valid < 0) {
        /* Probably must return 0, but just keep trying for now. */
        PyErr_Clear();  // 清除任何错误状态
    }
    /*
     * 如果标量不是数字，那么无法进行基于值的类型转换，
     * 因此我们不能尝试执行这种转换。
     * （虽然可能有其他快速检查方法，但可能是不必要的。）
     */
    if (!PyTypeNum_ISNUMBER(scal_type->type_num)) {
        // 如果标量类型不是数字类型，直接返回失败（0）
        return 0;
    }
    
    /*
     * 现在我们需要检查基于值的类型转换。
     */
    PyArray_Descr *dtype;
    int is_small_unsigned = 0, type_num;
    /* 一个对齐的内存缓冲区，足够容纳任何内置数值类型 */
    npy_longlong value[4];
    
    // 检查字节序是否需要交换，并使用相应的函数处理数据
    int swap = !PyArray_ISNBO(scal_type->byteorder);
    PyDataType_GetArrFuncs(scal_type)->copyswap(&value, scal_data, swap, NULL);
    
    // 确定最小的标量数据类型编号，用于表示给定数据
    type_num = min_scalar_type_num((char *)&value, scal_type->type_num,
                                    &is_small_unsigned);
    
    /*
     * 如果我们得到了一个小的无符号标量，并且目标类型（'to' 类型）
     * 不是无符号的，则将其转换为有符号类型，以便更合适地进行类型转换。
     */
    if (is_small_unsigned && !(PyTypeNum_ISUNSIGNED(to->type_num))) {
        // 如果是小的无符号整数且目标类型不是无符号类型，转换为有符号类型
        type_num = type_num_unsigned_to_signed(type_num);
    }
    
    // 根据类型编号获取数据类型描述符
    dtype = PyArray_DescrFromType(type_num);
    if (dtype == NULL) {
        // 如果获取数据类型描述符失败，返回失败（0）
        return 0;
    }
#if 0
    // 打印调试信息，显示最小标量类型转换的源数据类型
    printf("min scalar cast ");
    PyObject_Print(dtype, stdout, 0);
    // 打印调试信息，显示目标数据类型
    printf(" to ");
    PyObject_Print(to, stdout, 0);
    printf("\n");
#endif
    // 调用NumPy函数，检查是否可以将dtype转换为to指定的数据类型
    npy_bool ret = PyArray_CanCastTypeTo(dtype, to, casting);
    // 减少dtype对象的引用计数
    Py_DECREF(dtype);
    // 返回转换结果
    return ret;
}


NPY_NO_EXPORT npy_bool
can_cast_pyscalar_scalar_to(
        int flags, PyArray_Descr *to, NPY_CASTING casting)
{
    /*
     * This function only works reliably for legacy (NumPy dtypes).
     * If we end up here for a non-legacy DType, it is a bug.
     */
    // 断言to是否为NumPy的传统数据类型
    assert(NPY_DT_is_legacy(NPY_DTYPE(to)));

    /*
     * Quickly check for the typical numeric cases, where the casting rules
     * can be hardcoded fairly easily.
     */
    // 如果to是复数类型，可以进行转换
    if (PyDataType_ISCOMPLEX(to)) {
        return 1;
    }
    // 如果to是浮点数类型
    else if (PyDataType_ISFLOAT(to)) {
        // 如果标志中包含NPY_ARRAY_WAS_PYTHON_COMPLEX，只允许不安全转换
        if (flags & NPY_ARRAY_WAS_PYTHON_COMPLEX) {
            return casting == NPY_UNSAFE_CASTING;
        }
        return 1;
    }
    // 如果to是整数类型
    else if (PyDataType_ISINTEGER(to)) {
        // 如果标志中不包含NPY_ARRAY_WAS_PYTHON_INT，只允许不安全转换
        if (!(flags & NPY_ARRAY_WAS_PYTHON_INT)) {
            return casting == NPY_UNSAFE_CASTING;
        }
        return 1;
    }

    /*
     * For all other cases we use the default dtype.
     */
    // 根据标志位选择默认的数据类型
    PyArray_Descr *from;
    if (flags & NPY_ARRAY_WAS_PYTHON_INT) {
        from = PyArray_DescrFromType(NPY_LONG);
    }
    else if (flags & NPY_ARRAY_WAS_PYTHON_FLOAT) {
        from = PyArray_DescrFromType(NPY_DOUBLE);
    }
    else {
        from = PyArray_DescrFromType(NPY_CDOUBLE);
    }
    // 调用NumPy函数，检查是否可以将from转换为to指定的数据类型
    int res = PyArray_CanCastTypeTo(from, to, casting);
    // 减少from对象的引用计数
    Py_DECREF(from);
    // 返回转换结果
    return res;
}

/*NUMPY_API
 * Returns 1 if the array object may be cast to the given data type using
 * the casting rule, 0 otherwise.  This differs from PyArray_CanCastTo in
 * that it handles scalar arrays (0 dimensions) specially, by checking
 * their value.
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastArrayTo(PyArrayObject *arr, PyArray_Descr *to,
                        NPY_CASTING casting)
{
    // 获取数组arr的数据类型描述符
    PyArray_Descr *from = PyArray_DESCR(arr);
    // 获取to的数据类型元信息
    PyArray_DTypeMeta *to_dtype = NPY_DTYPE(to);

    /* NOTE, TODO: The same logic as `PyArray_CanCastTypeTo`: */
    // 如果to是无大小的数据类型且不是子数组，则将to置为NULL
    if (PyDataType_ISUNSIZED(to) && PyDataType_SUBARRAY(to) == NULL) {
        to = NULL;
    }

    // 如果使用传统的类型提升策略
    if (get_npy_promotion_state() == NPY_USE_LEGACY_PROMOTION) {
        /*
         * If it's a scalar, check the value.  (This only currently matters for
         * numeric types and for `to == NULL` it can't be numeric.)
         */
        // 如果数组是标量且没有字段，并且to不为NULL，则调用函数检查值是否可以转换
        if (PyArray_NDIM(arr) == 0 && !PyArray_HASFIELDS(arr) && to != NULL) {
            return can_cast_scalar_to(from, PyArray_DATA(arr), to, casting);
        }
    }
    /*
     * 否则，使用标准规则（与 `PyArray_CanCastTypeTo` 相同）
     * 判断是否可以安全地进行类型转换，返回一个整数表示结果
     */
    int is_valid = PyArray_CheckCastSafety(casting, from, to, to_dtype);
    /*
     * 清除任何错误并将此判断为不安全（应该可能被修改）
     * 如果判断为负值，则清除错误并返回0
     */
    if (is_valid < 0) {
        PyErr_Clear();
        return 0;
    }
    // 返回类型转换是否安全的结果
    return is_valid;
/**
 * 返回表示给定转换类型的字符串表示的 C 字符串指针。
 *
 * @param casting 转换类型枚举值
 * @return 表示转换类型的字符串常量
 */
NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "'no'";
        case NPY_EQUIV_CASTING:
            return "'equiv'";
        case NPY_SAFE_CASTING:
            return "'safe'";
        case NPY_SAME_KIND_CASTING:
            return "'same_kind'";
        case NPY_UNSAFE_CASTING:
            return "'unsafe'";
        default:
            return "<unknown>";
    }
}

/**
 * 设置当转换不可能时的有用错误消息的辅助函数。
 *
 * @param src_dtype 源数据类型描述符
 * @param dst_dtype 目标数据类型描述符
 * @param casting 转换类型
 * @param scalar 是否为标量转换
 */
NPY_NO_EXPORT void
npy_set_invalid_cast_error(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar)
{
    char *msg;

    if (!scalar) {
        msg = "Cannot cast array data from %R to %R according to the rule %s";
    }
    else {
        msg = "Cannot cast scalar from %R to %R according to the rule %s";
    }
    PyErr_Format(PyExc_TypeError,
            msg, src_dtype, dst_dtype, npy_casting_to_string(casting));
}

/**
 * 检查是否可以将数组标量转换。
 *
 * TODO: 对于 NumPy 2.0，添加一个 NPY_CASTING 参数。
 *
 * @param from 源数据类型对象
 * @param to 目标数据类型对象
 * @return 如果可以安全转换则返回 NPY_TRUE，否则返回 NPY_FALSE
 */
NPY_NO_EXPORT npy_bool
PyArray_CanCastScalar(PyTypeObject *from, PyTypeObject *to)
{
    int fromtype;
    int totype;

    fromtype = _typenum_fromtypeobj((PyObject *)from, 0);
    totype = _typenum_fromtypeobj((PyObject *)to, 0);
    if (fromtype == NPY_NOTYPE || totype == NPY_NOTYPE) {
        return NPY_FALSE;
    }
    return (npy_bool) PyArray_CanCastSafely(fromtype, totype);
}

/**
 * 内部升级类型的函数，特别处理能够适配同等大小有符号整数的无符号整数。
 *
 * @param type1 第一个数据类型描述符
 * @param type2 第二个数据类型描述符
 * @param is_small_unsigned1 第一个类型是否为小的无符号整数
 * @param is_small_unsigned2 第二个类型是否为小的无符号整数
 * @return 升级后的数据类型描述符
 */
static PyArray_Descr *
promote_types(PyArray_Descr *type1, PyArray_Descr *type2,
                        int is_small_unsigned1, int is_small_unsigned2)
{
    if (is_small_unsigned1) {
        int type_num1 = type1->type_num;
        int type_num2 = type2->type_num;
        int ret_type_num;

        if (type_num2 < NPY_NTYPES_LEGACY && !(PyTypeNum_ISBOOL(type_num2) ||
                                        PyTypeNum_ISUNSIGNED(type_num2))) {
            /* 转换为等效大小的有符号整数 */
            type_num1 = type_num_unsigned_to_signed(type_num1);

            ret_type_num = _npy_type_promotion_table[type_num1][type_num2];
            /* 表格不处理字符串/Unicode/void，请检查结果 */
            if (ret_type_num >= 0) {
                return PyArray_DescrFromType(ret_type_num);
            }
        }

        return PyArray_PromoteTypes(type1, type2);
    }
}
    # 如果其中一个类型是小的无符号整数类型，则执行以下逻辑
    else if (is_small_unsigned2) {
        # 获取 type1 和 type2 的类型编号
        int type_num1 = type1->type_num;
        int type_num2 = type2->type_num;
        int ret_type_num;

        # 如果 type1 的类型编号小于 NPY_NTYPES_LEGACY，并且不是布尔类型或无符号整数类型
        if (type_num1 < NPY_NTYPES_LEGACY && !(PyTypeNum_ISBOOL(type_num1) ||
                                        PyTypeNum_ISUNSIGNED(type_num1))) {
            /* 将 type2 转换为相同大小的有符号整数类型 */
            type_num2 = type_num_unsigned_to_signed(type_num2);

            # 从 _npy_type_promotion_table 中获取类型提升后的类型编号
            ret_type_num = _npy_type_promotion_table[type_num1][type_num2];
            /* 表格不处理字符串/Unicode/void 类型，需要检查结果 */
            if (ret_type_num >= 0) {
                # 根据 ret_type_num 创建一个 PyArray_Descr 结构
                return PyArray_DescrFromType(ret_type_num);
            }
        }

        # 如果上述条件不满足，则执行类型提升操作，返回 PyArray_PromoteTypes 的结果
        return PyArray_PromoteTypes(type1, type2);
    }
    else {
        # 如果不满足 is_small_unsigned2 的条件，则执行类型提升操作，返回 PyArray_PromoteTypes 的结果
        return PyArray_PromoteTypes(type1, type2);
    }
/**
 * This function adapts a given dtype instance (`descr`) to match or cast it to
 * the specified `given_DType`. It attempts to find an appropriate descriptor
 * for `given_DType` and returns an instance of `PyArray_Descr` that represents
 * this adaptation.
 *
 * If `descr` already matches `given_DType`, it returns `descr` itself without
 * modification. If `given_DType` is not parametric, it returns the default
 * descriptor for that type. If `descr` can be directly type-checked against
 * `given_DType`, it also returns `descr`.
 *
 * If none of these conditions are met, it tries to resolve the adaptation using
 * a suitable casting implementation (`PyArray_GetCastingImpl`). If successful,
 * it returns the adapted descriptor; otherwise, it raises a `TypeError`.
 *
 * @param descr The dtype instance to adapt or cast.
 * @param given_DType The target DType class to which `descr` should be adapted.
 * @returns An instance of `PyArray_Descr` representing `given_DType` or NULL
 *          on error.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType)
{
    if (NPY_DTYPE(descr) == given_DType) {
        Py_INCREF(descr);
        return descr;
    }
    if (!NPY_DT_is_parametric(given_DType)) {
        /*
         * If `given_DType` is not parametric, return the default descriptor
         * for that type.
         */
        return NPY_DT_CALL_default_descr(given_DType);
    }
    if (PyObject_TypeCheck((PyObject *)descr, (PyTypeObject *)given_DType)) {
        /*
         * If `descr` matches `given_DType` via type checking, return `descr`.
         */
        Py_INCREF(descr);
        return descr;
    }

    // Attempt to find a suitable casting implementation
    PyObject *tmp = PyArray_GetCastingImpl(NPY_DTYPE(descr), given_DType);
    if (tmp == NULL || tmp == Py_None) {
        Py_XDECREF(tmp);
        goto error;
    }

    // Prepare arrays and descriptors for casting resolution
    PyArray_DTypeMeta *dtypes[2] = {NPY_DTYPE(descr), given_DType};
    PyArray_Descr *given_descrs[2] = {descr, NULL};
    PyArray_Descr *loop_descrs[2];

    // Resolve the casting method and descriptors
    PyArrayMethodObject *meth = (PyArrayMethodObject *)tmp;
    npy_intp view_offset = NPY_MIN_INTP;
    NPY_CASTING casting = meth->resolve_descriptors(
            meth, dtypes, given_descrs, loop_descrs, &view_offset);
    Py_DECREF(tmp);
    if (casting < 0) {
        goto error;
    }
    
    // Return the adapted descriptor
    Py_DECREF(loop_descrs[0]);
    return loop_descrs[1];

  error:;  /* (; due to compiler limitations) */
    // Handle errors and raise a TypeError
    PyObject *err_type = NULL, *err_value = NULL, *err_traceback = NULL;
    PyErr_Fetch(&err_type, &err_value, &err_traceback);
    PyErr_Format(PyExc_TypeError,
            "cannot cast dtype %S to %S.", descr, given_DType);
    npy_PyErr_ChainExceptionsCause(err_type, err_value, err_traceback);
    return NULL;
}


/**
 * This function assists in finding the target descriptor for multiple arrays,
 * given an input one that may represent a DType class (e.g., "U" or "S").
 * It is used with arrays, especially in functions like `concatenate`.
 *
 * Unlike `np.array(...)` or `arr.astype()`, this function does not inspect
 * the content of the arrays. Therefore, object arrays can only be cast to
 * strings if a fixed width is provided, and similarly for string -> generic
 * datetime conversions.
 *
 * As this function utilizes `PyArray_ExtractDTypeAndDescriptor`, it is planned
 * to refactor this step to an earlier part of the process.
 *
 * @returns An instance of `PyArray_Descr` representing the target descriptor
 *          or NULL on error.
 */
NPY_NO_EXPORT PyArray_Descr *
/*
 * 查找连接描述符函数，用于确定多个数组的连接结果的数据描述符。
 * 
 * 参数：
 *   - n: 数组的数量
 *   - arrays: 数组对象的指针数组
 *   - requested_dtype: 请求的数据类型描述符
 * 
 * 返回值：
 *   - 返回连接后的结果的数据类型描述符，如果出错返回 NULL
 */
PyArray_FindConcatenationDescriptor(
        npy_intp n, PyArrayObject **arrays, PyArray_Descr *requested_dtype)
{
    if (requested_dtype == NULL) {
        // 如果请求的数据类型描述符为 NULL，则调用 PyArray_ResultType 返回推断的结果类型。
        return PyArray_ResultType(n, arrays, 0, NULL);
    }

    PyArray_DTypeMeta *common_dtype;
    PyArray_Descr *result = NULL;
    if (PyArray_ExtractDTypeAndDescriptor(
            requested_dtype, &result, &common_dtype) < 0) {
        // 从请求的数据类型描述符中提取类型和描述符，如果失败则返回 NULL。
        return NULL;
    }
    if (result != NULL) {
        if (PyDataType_SUBARRAY(result) != NULL) {
            // 如果结果描述符是子数组描述符，则抛出类型错误异常。
            PyErr_Format(PyExc_TypeError,
                    "The dtype `%R` is not a valid dtype for concatenation "
                    "since it is a subarray dtype (the subarray dimensions "
                    "would be added as array dimensions).", result);
            Py_SETREF(result, NULL);
        }
        goto finish;
    }
    assert(n > 0);  /* concatenate requires at least one array input. */

    /*
     * 注意：此段代码复制了 `PyArray_CastToDTypeAndPromoteDescriptors` 的逻辑，
     *       用于处理数组的类型转换和描述符的提升。
     */
    PyArray_Descr *descr = PyArray_DESCR(arrays[0]);
    // 将第一个数组的描述符转换为公共数据类型描述符。
    result = PyArray_CastDescrToDType(descr, common_dtype);
    if (result == NULL || n == 1) {
        goto finish;
    }
    for (npy_intp i = 1; i < n; i++) {
        descr = PyArray_DESCR(arrays[i]);
        // 将当前数组的描述符转换为公共数据类型描述符。
        PyArray_Descr *curr = PyArray_CastDescrToDType(descr, common_dtype);
        if (curr == NULL) {
            Py_SETREF(result, NULL);
            goto finish;
        }
        // 使用公共数据类型描述符的 common_instance 方法来推导结果描述符。
        Py_SETREF(result, NPY_DT_SLOTS(common_dtype)->common_instance(result, curr));
        Py_DECREF(curr);
        if (result == NULL) {
            goto finish;
        }
    }

  finish:
    // 释放公共数据类型的引用。
    Py_DECREF(common_dtype);
    return result;
}
    // 如果 type1 为空指针，则释放 common_dtype 对象并返回空指针
    if (type1 == NULL) {
        Py_DECREF(common_dtype);
        return NULL;
    }
    // 将 type2 转换为与 common_dtype 兼容的数据类型对象
    type2 = PyArray_CastDescrToDType(type2, common_dtype);
    // 如果 type2 为空指针，则释放 type1、common_dtype 对象并返回空指针
    if (type2 == NULL) {
        Py_DECREF(type1);
        Py_DECREF(common_dtype);
        return NULL;
    }

    /*
     * 找到两个输入类型的公共实例
     * 注意：公共实例保留元数据（通常是一个输入的元数据）
     */
    // 使用 common_dtype 的数据类型插槽调用 common_instance 方法获取结果
    res = NPY_DT_SLOTS(common_dtype)->common_instance(type1, type2);
    // 释放 type1、type2、common_dtype 对象的引用
    Py_DECREF(type1);
    Py_DECREF(type2);
    Py_DECREF(common_dtype);
    // 返回结果对象 res
    return res;
/*
 * } 结束了一个代码块，可能对应于某个函数或条件语句的结束。
 */

/*
 * 根据输入类型，产生能容纳所有输入类型的最小尺寸和最低种类类型。
 *
 * 大致相当于 functools.reduce(PyArray_PromoteTypes, types)，但使用了更复杂的成对方法。
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_PromoteTypeSequence(PyArray_Descr **types, npy_intp ntypes)
{
    // 如果没有输入类型，则设置类型错误并返回空
    if (ntypes == 0) {
        PyErr_SetString(PyExc_TypeError, "at least one type needed to promote");
        return NULL;
    }
    // 返回通过输入类型计算得出的结果类型
    return PyArray_ResultType(0, NULL, ntypes, types);
}

/*
 * 注意：虽然这不太可能成为性能问题，但如果是，可以回退到简单的正/负检查，
 *      就像以前的系统一样。
 *
 * is_small_unsigned 输出标志指示它是否为无符号整数，并且能适合于相同位大小的有符号整数。
 */
static int min_scalar_type_num(char *valueptr, int type_num,
                               int *is_small_unsigned)
{
    switch (type_num) {
        case NPY_BOOL: {
            // 布尔类型直接返回
            return NPY_BOOL;
        }
        case NPY_UBYTE: {
            npy_ubyte value = *(npy_ubyte *)valueptr;
            // 如果值小于等于最大的有符号字节，则标记为小的无符号整数
            if (value <= NPY_MAX_BYTE) {
                *is_small_unsigned = 1;
            }
            return NPY_UBYTE;
        }
        case NPY_BYTE: {
            npy_byte value = *(npy_byte *)valueptr;
            // 如果值大于等于0，则标记为小的无符号整数并返回无符号字节类型
            if (value >= 0) {
                *is_small_unsigned = 1;
                return NPY_UBYTE;
            }
            break;
        }
        case NPY_USHORT: {
            npy_ushort value = *(npy_ushort *)valueptr;
            // 如果值小于等于最大的无符号字节，则标记为小的无符号整数并返回无符号字节类型
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            // 如果值小于等于最大的短整数，则标记为小的无符号整数
            if (value <= NPY_MAX_SHORT) {
                *is_small_unsigned = 1;
            }
            break;
        }
        case NPY_SHORT: {
            npy_short value = *(npy_short *)valueptr;
            // 如果值大于等于0，则递归调用以获取更小的无符号整数类型
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_USHORT, is_small_unsigned);
            }
            // 如果值大于等于最小的字节，则返回字节类型
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_ULONG:
#endif
        case NPY_UINT: {
            npy_uint value = *(npy_uint *)valueptr;
            // 如果值小于等于最大的无符号字节，则标记为小的无符号整数并返回无符号字节类型
            if (value <= NPY_MAX_UBYTE) {
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            // 如果值小于等于最大的无符号短整数，则标记为小的无符号整数并返回无符号短整数类型
            else if (value <= NPY_MAX_USHORT) {
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            // 如果值小于等于最大的整数，则标记为小的无符号整数
            if (value <= NPY_MAX_INT) {
                *is_small_unsigned = 1;
            }
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_INT
        case NPY_LONG:
#endif
        // 处理 NPY_INT 类型的情况
        case NPY_INT: {
            // 获取 npy_int 类型的值
            npy_int value = *(npy_int *)valueptr;
            // 如果值大于等于 0
            if (value >= 0) {
                // 调用 min_scalar_type_num 函数，返回适合的无符号整数类型
                return min_scalar_type_num(valueptr, NPY_UINT, is_small_unsigned);
            }
            // 如果值小于 0，则根据大小返回相应的有符号整数类型
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            break;
        }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
        // 处理 NPY_ULONG 类型的情况
        case NPY_ULONG: {
            // 获取 npy_ulong 类型的值
            npy_ulong value = *(npy_ulong *)valueptr;
            // 如果值在 NPY_MAX_UBYTE 范围内
            if (value <= NPY_MAX_UBYTE) {
                // 如果值在 NPY_MAX_BYTE 范围内
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            // 如果值在 NPY_MAX_USHORT 范围内
            else if (value <= NPY_MAX_USHORT) {
                // 如果值在 NPY_MAX_SHORT 范围内
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            // 如果值在 NPY_MAX_UINT 范围内
            else if (value <= NPY_MAX_UINT) {
                // 如果值在 NPY_MAX_INT 范围内
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }

            // 如果值在 NPY_MAX_LONG 范围内
            if (value <= NPY_MAX_LONG) {
                *is_small_unsigned = 1;
            }
            break;
        }
        // 处理 NPY_LONG 类型的情况
        case NPY_LONG: {
            // 获取 npy_long 类型的值
            npy_long value = *(npy_long *)valueptr;
            // 如果值大于等于 0
            if (value >= 0) {
                // 调用 min_scalar_type_num 函数，返回适合的无符号整数类型
                return min_scalar_type_num(valueptr, NPY_ULONG, is_small_unsigned);
            }
            // 如果值小于 0，则根据大小返回相应的有符号整数类型
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
            break;
        }
#endif
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        // 在 NPY_SIZEOF_LONG 等于 NPY_SIZEOF_LONGLONG 的情况下处理 NPY_ULONG 类型
        case NPY_ULONG:
#endif
        // 处理 NPY_ULONGLONG 类型的情况
        case NPY_ULONGLONG: {
            // 获取 npy_ulonglong 类型的值
            npy_ulonglong value = *(npy_ulonglong *)valueptr;
            // 如果值在 NPY_MAX_UBYTE 范围内
            if (value <= NPY_MAX_UBYTE) {
                // 如果值在 NPY_MAX_BYTE 范围内
                if (value <= NPY_MAX_BYTE) {
                    *is_small_unsigned = 1;
                }
                return NPY_UBYTE;
            }
            // 如果值在 NPY_MAX_USHORT 范围内
            else if (value <= NPY_MAX_USHORT) {
                // 如果值在 NPY_MAX_SHORT 范围内
                if (value <= NPY_MAX_SHORT) {
                    *is_small_unsigned = 1;
                }
                return NPY_USHORT;
            }
            // 如果值在 NPY_MAX_UINT 范围内
            else if (value <= NPY_MAX_UINT) {
                // 如果值在 NPY_MAX_INT 范围内
                if (value <= NPY_MAX_INT) {
                    *is_small_unsigned = 1;
                }
                return NPY_UINT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            // 如果值在 NPY_MAX_ULONG 范围内
            else if (value <= NPY_MAX_ULONG) {
                // 如果值在 NPY_MAX_LONG 范围内
                if (value <= NPY_MAX_LONG) {
                    *is_small_unsigned = 1;
                }
                return NPY_ULONG;
            }
#endif
#endif

            // 如果 value 小于等于 NPY_MAX_LONGLONG，则标记为小型无符号数
            if (value <= NPY_MAX_LONGLONG) {
                *is_small_unsigned = 1;
            }
            // 结束 switch 语句块
            break;
        }
#if NPY_SIZEOF_LONG == NPY_SIZEOF_LONGLONG
        case NPY_LONG:
#endif
        // 长长整型情况
        case NPY_LONGLONG: {
            // 获取 value 的值
            npy_longlong value = *(npy_longlong *)valueptr;
            // 如果 value 大于等于 0，则调用 min_scalar_type_num 函数并返回结果
            if (value >= 0) {
                return min_scalar_type_num(valueptr, NPY_ULONGLONG, is_small_unsigned);
            }
            // 如果 value 在 NPY_MIN_BYTE 和 NPY_MIN_SHORT 之间，则返回 NPY_BYTE
            else if (value >= NPY_MIN_BYTE) {
                return NPY_BYTE;
            }
            // 如果 value 在 NPY_MIN_SHORT 和 NPY_MIN_INT 之间，则返回 NPY_SHORT
            else if (value >= NPY_MIN_SHORT) {
                return NPY_SHORT;
            }
            // 如果 value 在 NPY_MIN_INT 和 NPY_MIN_LONG 之间，则返回 NPY_INT
            else if (value >= NPY_MIN_INT) {
                return NPY_INT;
            }
#if NPY_SIZEOF_LONG != NPY_SIZEOF_INT && NPY_SIZEOF_LONG != NPY_SIZEOF_LONGLONG
            // 如果 value 大于等于 NPY_MIN_LONG，则返回 NPY_LONG
            else if (value >= NPY_MIN_LONG) {
                return NPY_LONG;
            }
    }

    // 返回 type_num 变量值
    return type_num;
}


NPY_NO_EXPORT PyArray_Descr *
PyArray_MinScalarType_internal(PyArrayObject *arr, int *is_small_unsigned)
{
    // 获取 arr 的数据类型描述符
    PyArray_Descr *dtype = PyArray_DESCR(arr);
    // 将 is_small_unsigned 标记为 0
    *is_small_unsigned = 0;
    /*
     * 如果数组不是数值标量，直接返回数组的数据类型描述符。
     */
    if (PyArray_NDIM(arr) > 0 || !PyTypeNum_ISNUMBER(dtype->type_num)) {
        Py_INCREF(dtype);
        return dtype;
    }
    else {
        // 获取数组的字节数据的起始位置
        char *data = PyArray_BYTES(arr);
        // 判断是否需要进行字节序交换
        int swap = !PyArray_ISNBO(dtype->byteorder);
        /* An aligned memory buffer large enough to hold any type */
        // 声明一个能够存放任何类型数据的对齐内存缓冲区
        npy_longlong value[4];
        // 使用 dtype 的数组函数进行数据的拷贝和交换
        PyDataType_GetArrFuncs(dtype)->copyswap(&value, data, swap, NULL);

        // 调用 min_scalar_type_num 函数，并返回其返回值转换成数据类型描述符
        return PyArray_DescrFromType(
                        min_scalar_type_num((char *)&value,
                                dtype->type_num, is_small_unsigned));

    }
}

/*NUMPY_API
 * 如果 arr 是标量（维度为 0）并且具有内置数值数据类型，
 * 查找能够表示其数据的最小类型大小/种类。否则，返回数组的数据类型。
 *
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_MinScalarType(PyArrayObject *arr)
{
    // 标记是否为小型无符号数
    int is_small_unsigned;
    // 调用 PyArray_MinScalarType_internal 函数，并返回其结果
    return PyArray_MinScalarType_internal(arr, &is_small_unsigned);
}

/*
 * 提供用于 dtype 'kind' 字符代码的排序，以帮助确定何时使用 min_scalar_type 函数。
 * 将 'kind' 分为布尔、整数、浮点和其他类型。
 */
static int
dtype_kind_to_simplified_ordering(char kind)
{
    // 根据 dtype 'kind' 字符进行分类和排序
    switch (kind) {
        // 布尔类型
        case 'b':
            return 0;
        // 无符号整数类型
        case 'u':
        // 有符号整数类型
        case 'i':
            return 1;
        // 浮点类型
        case 'f':
        // 复数类型
        case 'c':
            return 2;
        // 其他类型
        default:
            return 3;
    }
}
/*
 * 确定是否存在标量和数组/数据类型的混合。
 * 如果存在混合情况，标量应该被处理为能够容纳其值的最小类型，
 * 当标量的最大“类别”超过数组/数据类型的最大“类别”时。
 * 如果标量的类别低于或等于数组的类别，它们可以降级到其类别内的更低类型
 * （根据标量强制转换规则安全地转换为的最低类型）。
 *
 * 如果涉及任何新样式的数据类型（非传统），始终返回0。
 */
NPY_NO_EXPORT int
should_use_min_scalar(npy_intp narrs, PyArrayObject **arr,
                      npy_intp ndtypes, PyArray_Descr **dtypes)
{
    int use_min_scalar = 0;

    if (narrs > 0) {
        int all_scalars;
        int max_scalar_kind = -1;
        int max_array_kind = -1;

        all_scalars = (ndtypes > 0) ? 0 : 1;

        /* 计算最大“类别”，并确定是否所有元素都是标量 */
        for (npy_intp i = 0; i < narrs; ++i) {
            // 检查是否为传统类型，若非传统类型则直接返回0
            if (!NPY_DT_is_legacy(NPY_DTYPE(PyArray_DESCR(arr[i])))) {
                return 0;
            }
            // 如果数组的维度为0，获取简化后的类别并更新最大标量类别
            if (PyArray_NDIM(arr[i]) == 0) {
                int kind = dtype_kind_to_simplified_ordering(
                                    PyArray_DESCR(arr[i])->kind);
                if (kind > max_scalar_kind) {
                    max_scalar_kind = kind;
                }
            }
            else {
                // 否则获取简化后的类别并更新最大数组类别，同时设置标量标记为否
                int kind = dtype_kind_to_simplified_ordering(
                                    PyArray_DESCR(arr[i])->kind);
                if (kind > max_array_kind) {
                    max_array_kind = kind;
                }
                all_scalars = 0;
            }
        }
        /*
         * 如果最大标量类别大于等于最大数组类别，
         * 继续计算最大数组类别
         */
        for (npy_intp i = 0; i < ndtypes; ++i) {
            // 检查是否为传统类型，若非传统类型则直接返回0
            if (!NPY_DT_is_legacy(NPY_DTYPE(dtypes[i]))) {
                return 0;
            }
            // 获取简化后的类别并更新最大数组类别
            int kind = dtype_kind_to_simplified_ordering(dtypes[i]->kind);
            if (kind > max_array_kind) {
                max_array_kind = kind;
            }
        }

        /* 标识是否需要使用 min_scalar_type 函数 */
        if (!all_scalars && max_array_kind >= max_scalar_kind) {
            use_min_scalar = 1;
        }
    }
    return use_min_scalar;
}


NPY_NO_EXPORT int
should_use_min_scalar_weak_literals(int narrs, PyArrayObject **arr) {
    int all_scalars = 1;
    int max_scalar_kind = -1;
    int max_array_kind = -1;
    // 遍历数组 `arr` 中的每一个元素
    for (int i = 0; i < narrs; i++) {
        // 检查当前数组元素是否标记为 Python 整数
        if (PyArray_FLAGS(arr[i]) & NPY_ARRAY_WAS_PYTHON_INT) {
            /* 如果是 Python 整数，相当于 'u'，获取其简化的类型顺序 */
            int new = dtype_kind_to_simplified_ordering('u');
            // 如果新的类型顺序大于当前的最大标量类型顺序，更新最大标量类型顺序
            if (new > max_scalar_kind) {
                max_scalar_kind = new;
            }
        }
        /* 对于新逻辑，只有复数和非复数有意义: */
        else if (PyArray_FLAGS(arr[i]) & NPY_ARRAY_WAS_PYTHON_FLOAT) {
            // 如果数组元素标记为 Python 浮点数，设置最大标量类型顺序为 'f'
            max_scalar_kind = dtype_kind_to_simplified_ordering('f');
        }
        else if (PyArray_FLAGS(arr[i]) & NPY_ARRAY_WAS_PYTHON_COMPLEX) {
            // 如果数组元素标记为 Python 复数，设置最大标量类型顺序为 'f'
            max_scalar_kind = dtype_kind_to_simplified_ordering('f');
        }
        else {
            // 如果以上条件都不满足，说明数组元素不全是标量
            all_scalars = 0;
            // 获取数组元素描述符的类型种类，然后转换为简化的类型顺序
            int kind = dtype_kind_to_simplified_ordering(
                    PyArray_DESCR(arr[i])->kind);
            // 如果新的类型顺序大于当前的最大数组类型顺序，更新最大数组类型顺序
            if (kind > max_array_kind) {
                max_array_kind = kind;
            }
        }
    }
    // 如果不是所有元素都是标量，并且最大数组类型顺序大于等于最大标量类型顺序，返回1
    if (!all_scalars && max_array_kind >= max_scalar_kind) {
        return 1;
    }

    // 否则返回0
    return 0;
/*NUMPY_API
 *
 * Produces the result type of a bunch of inputs, using the same rules
 * as `np.result_type`.
 *
 * NOTE: This function is expected to through a transitional period or
 *       change behaviour.  DTypes should always be strictly enforced for
 *       0-D arrays, while "weak DTypes" will be used to represent Python
 *       integers, floats, and complex in all cases.
 *       (Within this function, these are currently flagged on the array
 *       object to work through `np.result_type`, this may change.)
 *
 *       Until a time where this transition is complete, we probably cannot
 *       add new "weak DTypes" or allow users to create their own.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_ResultType(
        npy_intp narrs, PyArrayObject *arrs[],
        npy_intp ndtypes, PyArray_Descr *descrs[])
{
    PyArray_Descr *result = NULL;

    if (narrs + ndtypes <= 1) {
        /* If the input is a single value, skip promotion. */
        if (narrs == 1) {
            result = PyArray_DTYPE(arrs[0]); // 获取单个数组的数据类型描述符
        }
        else if (ndtypes == 1) {
            result = descrs[0]; // 获取单个数据类型描述符
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "no arrays or types available to calculate result type");
            return NULL; // 如果没有数组或类型可用来计算结果类型，则返回 NULL
        }
        return NPY_DT_CALL_ensure_canonical(result); // 确保结果数据类型是规范化的并返回
    }

    void **info_on_heap = NULL;
    void *_info_on_stack[NPY_MAXARGS * 2];
    PyArray_DTypeMeta **all_DTypes;
    PyArray_Descr **all_descriptors;

    if (narrs + ndtypes > NPY_MAXARGS) {
        info_on_heap = PyMem_Malloc(2 * (narrs+ndtypes) * sizeof(PyObject *));
        if (info_on_heap == NULL) {
            PyErr_NoMemory(); // 内存分配失败时报错
            return NULL; // 返回 NULL
        }
        all_DTypes = (PyArray_DTypeMeta **)info_on_heap;
        all_descriptors = (PyArray_Descr **)(info_on_heap + narrs + ndtypes);
    }
    else {
        all_DTypes = (PyArray_DTypeMeta **)_info_on_stack;
        all_descriptors = (PyArray_Descr **)(_info_on_stack + narrs + ndtypes);
    }

    /* Copy all dtypes into a single array defining non-value-based behaviour */
    for (npy_intp i=0; i < ndtypes; i++) {
        all_DTypes[i] = NPY_DTYPE(descrs[i]); // 将输入的数据类型描述符复制到一个数组中
        Py_INCREF(all_DTypes[i]); // 增加对数据类型对象的引用计数
        all_descriptors[i] = descrs[i]; // 复制数据类型描述符
    }

    int at_least_one_scalar = 0;
    int all_pyscalar = ndtypes == 0;
    for (npy_intp i=0, i_all=ndtypes; i < narrs; i++, i_all++) {
        /* 遍历数组列表和描述符列表 */
        /* i 是当前数组的索引，i_all 是在所有描述符中的索引 */

        /* 如果当前数组是标量（0维数组），设置标志位 */
        if (PyArray_NDIM(arrs[i]) == 0) {
            at_least_one_scalar = 1;
        }

        /*
         * 如果原始数组是 Python 标量/字面值，则下面只使用对应的抽象数据类型（不使用描述符）。
         * 否则，传播描述符。
         */
        all_descriptors[i_all] = NULL;  /* 对于 Python 标量，没有描述符 */
        /* 检查数组的标志位，判断是否为 Python 的整数类型 */
        if (PyArray_FLAGS(arrs[i]) & NPY_ARRAY_WAS_PYTHON_INT) {
            /* 在这里即使是大整数也可能是对象数据类型 */
            all_DTypes[i_all] = &PyArray_PyLongDType;
            /* 如果数组类型不是长整型，则不能避免使用旧的路径 */
            if (PyArray_TYPE(arrs[i]) != NPY_LONG) {
                all_pyscalar = 0;
            }
        }
        /* 检查数组的标志位，判断是否为 Python 的浮点数类型 */
        else if (PyArray_FLAGS(arrs[i]) & NPY_ARRAY_WAS_PYTHON_FLOAT) {
            all_DTypes[i_all] = &PyArray_PyFloatDType;
        }
        /* 检查数组的标志位，判断是否为 Python 的复数类型 */
        else if (PyArray_FLAGS(arrs[i]) & NPY_ARRAY_WAS_PYTHON_COMPLEX) {
            all_DTypes[i_all] = &PyArray_PyComplexDType;
        }
        /* 否则，使用数组的描述符和数据类型 */
        else {
            all_descriptors[i_all] = PyArray_DTYPE(arrs[i]);
            all_DTypes[i_all] = NPY_DTYPE(all_descriptors[i_all]);
            all_pyscalar = 0;
        }
        /* 增加所有数据类型的引用计数 */
        Py_INCREF(all_DTypes[i_all]);
    }

    /* 提升数据类型序列，以找到一个公共的数据类型 */
    PyArray_DTypeMeta *common_dtype = PyArray_PromoteDTypeSequence(
            narrs+ndtypes, all_DTypes);
    /* 减少所有数据类型的引用计数 */
    for (npy_intp i=0; i < narrs+ndtypes; i++) {
        Py_DECREF(all_DTypes[i]);
    }
    /* 如果没有找到公共数据类型，则跳转到错误处理 */
    if (common_dtype == NULL) {
        goto error;
    }

    /* 如果公共数据类型是抽象的，使用默认描述符来定义一个默认值 */
    if (NPY_DT_is_abstract(common_dtype)) {
        /* 调用默认描述符函数来获取默认描述符 */
        PyArray_Descr *tmp_descr = NPY_DT_CALL_default_descr(common_dtype);
        /* 如果获取失败，则跳转到错误处理 */
        if (tmp_descr == NULL) {
            goto error;
        }
        /* 增加默认描述符的引用计数 */
        Py_INCREF(NPY_DTYPE(tmp_descr));
        /* 设置公共数据类型为默认描述符 */
        Py_SETREF(common_dtype, NPY_DTYPE(tmp_descr));
        /* 减少临时描述符的引用计数 */
        Py_DECREF(tmp_descr);
    }

    /*
     * 注意：此段代码与 PyArray_CastToDTypeAndPromoteDescriptors 重复，
     *       但支持对抽象值的特殊处理。
     */
    /* 如果公共数据类型是参数化的 */
    if (NPY_DT_is_parametric(common_dtype)) {
        /* 遍历所有描述符，将其转换为公共数据类型 */
        for (npy_intp i = 0; i < ndtypes+narrs; i++) {
            /* 如果描述符为空，则原始为 Python 标量/字面值，跳过 */
            if (all_descriptors[i] == NULL) {
                continue;
            }
            /* 将当前描述符转换为公共数据类型 */
            PyArray_Descr *curr = PyArray_CastDescrToDType(
                    all_descriptors[i], common_dtype);
            /* 如果转换失败，则跳转到错误处理 */
            if (curr == NULL) {
                goto error;
            }
            /* 如果结果为 NULL，则跳转到错误处理 */
            if (result == NULL) {
                result = curr;
                continue;
            }
            /* 更新结果为公共实例 */
            Py_SETREF(result, NPY_DT_SLOTS(common_dtype)->common_instance(result, curr));
            /* 减少当前描述符的引用计数 */
            Py_DECREF(curr);
            /* 如果结果为 NULL，则跳转到错误处理 */
            if (result == NULL) {
                goto error;
            }
        }
    }
    # 如果结果指针为 NULL，则说明结果尚未设置
    if (result == NULL) {
        /*
         * 如果 DType 不是参数化的，或者全部是弱标量，
         * 可能还未设置结果。
         */
        // 根据通用 DType 调用默认描述符生成结果
        result = NPY_DT_CALL_default_descr(common_dtype);
        // 如果生成结果失败，则跳转到错误处理标签
        if (result == NULL) {
            goto error;
        }
    }

    /*
     * 不幸的是，当涉及 0-D “标量”数组且混合时，我们可能需要使用基于值的逻辑。
     * `PyArray_CheckLegacyResultType` 可能会根据 `npy_legacy_promotion` 的当前值而行事：
     * 1. 它什么也不做（使用“新”行为）
     * 2. 它什么也不做，但在结果不同的情况下发出警告。
     * 3. 它基于传统的值逻辑替换结果。
     */
    // 如果至少有一个标量并且不是全部都是 Python 标量，并且结果的类型编号小于 NPY_NTYPES_LEGACY
    if (at_least_one_scalar && !all_pyscalar && result->type_num < NPY_NTYPES_LEGACY) {
        // 检查是否需要使用传统的结果类型推断逻辑
        if (PyArray_CheckLegacyResultType(
                &result, narrs, arrs, ndtypes, descrs) < 0) {
            // 如果检查失败，释放通用 DType 和结果，并返回 NULL
            Py_DECREF(common_dtype);
            Py_DECREF(result);
            return NULL;
        }
    }

    // 释放通用 DType 对象
    Py_DECREF(common_dtype);
    // 释放堆上的信息
    PyMem_Free(info_on_heap);
    // 返回结果指针
    return result;

  error:
    // 清理错误情况下的结果和通用 DType 对象
    Py_XDECREF(result);
    Py_XDECREF(common_dtype);
    // 释放堆上的信息
    PyMem_Free(info_on_heap);
    // 返回 NULL 指针，表示错误状态
    return NULL;
/*
 * Produces the result type of a bunch of inputs, using the UFunc
 * type promotion rules. Use this function when you have a set of
 * input arrays, and need to determine an output array dtype.
 *
 * If all the inputs are scalars (have 0 dimensions) or the maximum "kind"
 * of the scalars is greater than the maximum "kind" of the arrays, does
 * a regular type promotion.
 *
 * Otherwise, does a type promotion on the MinScalarType
 * of all the inputs.  Data types passed directly are treated as array
 * types.
 */
NPY_NO_EXPORT int
PyArray_CheckLegacyResultType(
        PyArray_Descr **new_result,
        npy_intp narrs, PyArrayObject **arr,
        npy_intp ndtypes, PyArray_Descr **dtypes)
{
    PyArray_Descr *ret = NULL;
    int promotion_state = get_npy_promotion_state();

    // 如果处于弱类型提升状态，则直接返回
    if (promotion_state == NPY_USE_WEAK_PROMOTION) {
        return 0;
    }

    // 如果处于弱类型提升且警告状态，并且未设置提升警告，直接返回
    if (promotion_state == NPY_USE_WEAK_PROMOTION_AND_WARN
            && !npy_give_promotion_warnings()) {
        return 0;
    }

    npy_intp i;

    /* If there's just one type, results must match */
    // 如果只有一个类型，则结果必须匹配，直接返回
    if (narrs + ndtypes == 1) {
        return 0;
    }

    // 根据输入情况决定是否使用最小标量类型
    int use_min_scalar = should_use_min_scalar(narrs, arr, ndtypes, dtypes);

    /* Loop through all the types, promoting them */
    // 遍历所有类型，进行类型提升
    if (!use_min_scalar) {

        /* Build a single array of all the dtypes */
        // 创建包含所有数据类型的数组
        PyArray_Descr **all_dtypes = PyArray_malloc(
            sizeof(*all_dtypes) * (narrs + ndtypes));
        if (all_dtypes == NULL) {
            PyErr_NoMemory();  // 内存分配失败时抛出异常
            return -1;
        }
        for (i = 0; i < narrs; ++i) {
            all_dtypes[i] = PyArray_DESCR(arr[i]);  // 获取数组对象的数据类型描述符
        }
        for (i = 0; i < ndtypes; ++i) {
            all_dtypes[narrs + i] = dtypes[i];  // 复制直接传递的数据类型
        }
        // 调用函数进行数据类型序列的提升
        ret = PyArray_PromoteTypeSequence(all_dtypes, narrs + ndtypes);
        PyArray_free(all_dtypes);  // 释放内存
    }
    else {
        // 初始化一个标志变量，用于检查是否存在小型无符号整数类型
        int ret_is_small_unsigned = 0;

        // 遍历输入的数组
        for (i = 0; i < narrs; ++i) {
            // 临时变量，用于存储是否存在小型无符号整数类型
            int tmp_is_small_unsigned;
            // 获取数组 arr[i] 的最小标量类型，并返回其描述符
            PyArray_Descr *tmp = PyArray_MinScalarType_internal(
                arr[i], &tmp_is_small_unsigned);
            // 如果获取失败，则释放已有结果并返回错误
            if (tmp == NULL) {
                Py_XDECREF(ret);
                return -1;
            }
            /* 将获取到的最小标量类型与已有类型合并 */
            if (ret == NULL) {
                ret = tmp;
                ret_is_small_unsigned = tmp_is_small_unsigned;
            }
            else {
                // 合并两个类型，并返回结果的描述符
                PyArray_Descr *tmpret = promote_types(
                    tmp, ret, tmp_is_small_unsigned, ret_is_small_unsigned);
                Py_DECREF(tmp);
                Py_DECREF(ret);
                ret = tmpret;
                // 如果合并失败，则返回错误
                if (ret == NULL) {
                    return -1;
                }

                // 更新合并后的类型是否为小型无符号整数类型的标志
                ret_is_small_unsigned = tmp_is_small_unsigned &&
                                        ret_is_small_unsigned;
            }
        }

        // 遍历输入的描述符数组
        for (i = 0; i < ndtypes; ++i) {
            // 获取描述符数组中的描述符
            PyArray_Descr *tmp = dtypes[i];
            /* 将获取到的描述符与已有类型合并 */
            if (ret == NULL) {
                ret = tmp;
                Py_INCREF(ret);
            }
            else {
                // 合并两个类型，并返回结果的描述符
                PyArray_Descr *tmpret = promote_types(
                    tmp, ret, 0, ret_is_small_unsigned);
                Py_DECREF(ret);
                ret = tmpret;
                // 如果合并失败，则返回错误
                if (ret == NULL) {
                    return -1;
                }
            }
        }
        /* 上述循环均未执行 */
        // 如果没有合并任何类型，则设置类型错误异常
        if (ret == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "no arrays or types available to calculate result type");
        }
    }

    // 如果结果为空，则返回错误
    if (ret == NULL) {
        return -1;
    }

    // 检查新结果类型与旧结果类型是否等效
    int unchanged_result = PyArray_EquivTypes(*new_result, ret);
    // 如果等效，则释放结果并返回成功
    if (unchanged_result) {
        Py_DECREF(ret);
        return 0;
    }

    // 根据提升状态进行处理
    if (promotion_state == NPY_USE_LEGACY_PROMOTION) {
        // 使用传统的提升方式设置新结果
        Py_SETREF(*new_result, ret);
        return 0;
    }

    // 断言使用弱提升并发出警告
    assert(promotion_state == NPY_USE_WEAK_PROMOTION_AND_WARN);
    // 如果警告失败，则释放结果并返回错误
    if (PyErr_WarnFormat(PyExc_UserWarning, 1,
            "result dtype changed due to the removal of value-based "
            "promotion from NumPy. Changed from %S to %S.",
            ret, *new_result) < 0) {
        Py_DECREF(ret);
        return -1;
    }
    // 释放结果并返回成功
    Py_DECREF(ret);
    return 0;
/**
 * Promotion of descriptors (of arbitrary DType) to their correctly
 * promoted instances of the given DType.
 * I.e. the given DType could be a string, which then finds the correct
 * string length, given all `descrs`.
 *
 * @param ndescr number of descriptors to cast and find the common instance.
 *        At least one must be passed in.
 * @param descrs The descriptors to work with.
 * @param DType The DType of the desired output descriptor.
 */
NPY_NO_EXPORT PyArray_Descr *
PyArray_CastToDTypeAndPromoteDescriptors(
        npy_intp ndescr, PyArray_Descr *descrs[], PyArray_DTypeMeta *DType)
{
    assert(ndescr > 0);

    // Cast the first descriptor to the desired DType
    PyArray_Descr *result = PyArray_CastDescrToDType(descrs[0], DType);
    if (result == NULL || ndescr == 1) {
        return result;
    }
    // If DType is not parametric, use the default descriptor
    if (!NPY_DT_is_parametric(DType)) {
        /* Note that this "fast" path loses all metadata */
        Py_DECREF(result);
        return NPY_DT_CALL_default_descr(DType);
    }

    // Iterate over the rest of the descriptors and find the common instance
    for (npy_intp i = 1; i < ndescr; i++) {
        PyArray_Descr *curr = PyArray_CastDescrToDType(descrs[i], DType);
        if (curr == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        // Find the common instance of the descriptors
        Py_SETREF(result, NPY_DT_SLOTS(DType)->common_instance(result, curr));
        Py_DECREF(curr);
        if (result == NULL) {
            return NULL;
        }
    }
    return result;
}


/**
 * Is the typenum valid?
 *
 * @param type The type number to check validity.
 * @return 1 if valid, 0 otherwise.
 */
NPY_NO_EXPORT int
PyArray_ValidType(int type)
{
    PyArray_Descr *descr;
    int res=NPY_TRUE;

    // Get the descriptor from the type number
    descr = PyArray_DescrFromType(type);
    if (descr == NULL) {
        res = NPY_FALSE;
    }
    Py_DECREF(descr);
    return res;
}

/**
 * Check if the object record descriptor is supported.
 *
 * @param descr The descriptor to check.
 * @return 0 if supported, -1 otherwise.
 */
static int
_check_object_rec(PyArray_Descr *descr)
{
    // Check if the descriptor has fields and requires reference check
    if (PyDataType_HASFIELDS(descr) && PyDataType_REFCHK(descr)) {
        PyErr_SetString(PyExc_TypeError, "Not supported for this data-type.");
        return -1;
    }
    return 0;
}

/**
 * Get pointer to zero of correct type for array.
 *
 * @param arr The array object to get the zero pointer for.
 * @return Pointer to zero value of correct type, or NULL on error.
 */
NPY_NO_EXPORT char *
PyArray_Zero(PyArrayObject *arr)
{
    char *zeroval;
    int ret, storeflags;

    // Check if object record descriptor is supported
    if (_check_object_rec(PyArray_DESCR(arr)) < 0) {
        return NULL;
    }
    // Allocate memory for zero value based on array item size
    zeroval = PyDataMem_NEW(PyArray_ITEMSIZE(arr));
    if (zeroval == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }
    # 检查数组是否包含对象类型的元素
    if (PyArray_ISOBJECT(arr)) {
        # 如果是对象数组，需要注意以下问题：
        # XXX 这里存在潜在风险，调用者可能不知道 zeroval 实际上是一个静态 PyObject*
        # 最好的情况是他们只会按原样使用它，但如果他们简单地将其 memcpy 到一个 ndarray 中而不使用 setitem()，
        # 可能会导致引用计数错误。
        memcpy(zeroval, &npy_static_pydata.zero_obj, sizeof(PyObject *));
        # 返回复制的 zeroval
        return zeroval;
    }
    # 存储数组的原始标志位
    storeflags = PyArray_FLAGS(arr);
    # 启用数组的行为标志位
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_BEHAVED);
    # 将 npy_static_pydata.zero_obj 设置为数组 arr 的零元素
    ret = PyArray_SETITEM(arr, zeroval, npy_static_pydata.zero_obj);
    # 恢复数组原始的标志位
    ((PyArrayObject_fields *)arr)->flags = storeflags;
    # 如果设置元素操作失败
    if (ret < 0) {
        # 释放 zeroval 的内存
        PyDataMem_FREE(zeroval);
        # 返回空指针
        return NULL;
    }
    # 返回设置后的 zeroval
    return zeroval;
/*NUMPY_API
 * Return the typecode of the array a Python object would be converted to
 *
 * Returns the type number the result should have, or NPY_NOTYPE on error.
 */
NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type)
{
    PyArray_Descr *dtype = NULL;
    int ret;

    if (minimum_type != NPY_NOTYPE && minimum_type >= 0) {
        // 根据最小类型获取描述符
        dtype = PyArray_DescrFromType(minimum_type);
        if (dtype == NULL) {
            return NPY_NOTYPE;
        }
    }
    // 根据对象获取数据类型描述符
    if (PyArray_DTypeFromObject(op, NPY_MAXDIMS, &dtype) < 0) {
        return NPY_NOTYPE;
    }

    if (dtype == NULL) {
        // 如果描述符为空，默认返回默认类型
        ret = NPY_DEFAULT_TYPE;
    }
    else if (!NPY_DT_is_legacy(NPY_DTYPE(dtype))) {
        /*
         * TODO: If we keep all type number style API working, by defining
         *       type numbers always. We may be able to allow this again.
         */
        // 如果描述符不是旧式 NumPy dtypes 或者用户定义的 dtypes，抛出类型错误
        PyErr_Format(PyExc_TypeError,
                "This function currently only supports native NumPy dtypes "
                "and old-style user dtypes, but the dtype was %S.\n"
                "(The function may need to be updated to support arbitrary"
                "user dtypes.)",
                dtype);
        ret = NPY_NOTYPE;
    }
    else {
        // 返回描述符的类型编号
        ret = dtype->type_num;
    }

    Py_XDECREF(dtype);

    return ret;
}

/* Raises error when len(op) == 0 */

/*NUMPY_API
 *
 * This function is only used in one place within NumPy and should
 * generally be avoided. It is provided mainly for backward compatibility.
 *
 * The user of the function has to free the returned array with PyDataMem_FREE.
 */
NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn)
{
    int i, n;
    PyArray_Descr *common_descr = NULL;
    PyArrayObject **mps = NULL;

    // 获取序列对象的长度
    *retn = n = PySequence_Length(op);
    # 如果 n 等于 0，则设置一个 ValueError 异常
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "0-length sequence.");
    }
    # 检查是否有异常发生，如果有，则设置返回值为 0 并返回 NULL
    if (PyErr_Occurred()) {
        *retn = 0;
        return NULL;
    }
    # 分配内存以存储 n 个 PyArrayObject 指针，并检查分配是否成功
    mps = (PyArrayObject **)PyDataMem_NEW(n*sizeof(PyArrayObject *));
    if (mps == NULL) {
        *retn = 0;
        # 分配内存失败，返回内存不足的异常对象
        return (void*)PyErr_NoMemory();
    }

    # 如果 op 是一个数组对象，将其转换为 PyArrayObject 数组
    if (PyArray_Check(op)) {
        for (i = 0; i < n; i++) {
            # 将 op 数组中的每个元素转换为 PyArrayObject，并存储在 mps 数组中
            mps[i] = (PyArrayObject *) array_item_asarray((PyArrayObject *)op, i);
        }
        # 如果 op 不是 C 连续数组，则复制每个数组对象以确保其是 C 连续的
        if (!PyArray_ISCARRAY((PyArrayObject *)op)) {
            for (i = 0; i < n; i++) {
                PyObject *obj;
                # 创建一个以 C 顺序排列的 mps[i] 的副本，并释放原来的 mps[i]
                obj = PyArray_NewCopy(mps[i], NPY_CORDER);
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)obj;
            }
        }
        # 返回 PyArrayObject 数组 mps
        return mps;
    }

    # 初始化 mps 数组的所有元素为 NULL
    for (i = 0; i < n; i++) {
        mps[i] = NULL;
    }

    # 将 op 中的每个元素转换为 PyArrayObject 并存储在 mps 数组中
    for (i = 0; i < n; i++) {
        /* Convert everything to an array, this could be optimized away */
        PyObject *tmp = PySequence_GetItem(op, i);
        if (tmp == NULL) {
            goto fail;
        }

        mps[i] = (PyArrayObject *)PyArray_FROM_O(tmp);
        if (mps[i] == NULL) {
            Py_DECREF(tmp);
            goto fail;
        }
        # 标记 tmp 为临时数组，如果是标量则用 mps[i] 替换 tmp
        npy_mark_tmp_array_if_pyscalar(tmp, mps[i], NULL);
        Py_DECREF(tmp);
    }

    # 计算所有 mps 数组元素的结果类型并存储在 common_descr 中
    common_descr = PyArray_ResultType(n, mps, 0, NULL);
    if (common_descr == NULL) {
        goto fail;
    }

    # 确保所有数组都是连续的并且具有正确的数据类型
    for (i = 0; i < n; i++) {
        int flags = NPY_ARRAY_CARRAY;
        PyArrayObject *tmp = mps[i];

        Py_INCREF(common_descr);
        # 使用 common_descr 和 flags 创建一个新的 PyArrayObject
        mps[i] = (PyArrayObject *)PyArray_FromArray(tmp, common_descr, flags);
        Py_DECREF(tmp);
        if (mps[i] == NULL) {
            goto fail;
        }
    }
    Py_DECREF(common_descr);
    # 返回成功创建的 PyArrayObject 数组 mps
    return mps;

 fail:
    # 处理失败的情况：释放资源并返回 NULL
    Py_XDECREF(common_descr);
    *retn = 0;
    for (i = 0; i < n; i++) {
        Py_XDECREF(mps[i]);
    }
    PyDataMem_FREE(mps);
    return NULL;
/**
 * Private function to add a casting implementation by unwrapping a bound
 * array method.
 *
 * @param meth The PyBoundArrayMethodObject representing the bound array method.
 * @return 0 on success -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AddCastingImplementation(PyBoundArrayMethodObject *meth)
{
    // Check if the number of input and output arguments is not equal to 1
    if (meth->method->nin != 1 || meth->method->nout != 1) {
        PyErr_SetString(PyExc_TypeError,
                "A cast must have one input and one output.");
        return -1;
    }
    // Check if the input and output dtypes are identical
    if (meth->dtypes[0] == meth->dtypes[1]) {
        /*
         * The method casting between instances of the same dtype is special,
         * since it is common, it is stored explicitly (currently) and must
         * obey additional constraints to ensure convenient casting.
         */
        // Check if the method supports unaligned data
        if (!(meth->method->flags & NPY_METH_SUPPORTS_UNALIGNED)) {
            PyErr_Format(PyExc_TypeError,
                    "A cast where input and output DType (class) are identical "
                    "must currently support unaligned data. (method: %s)",
                    meth->method->name);
            return -1;
        }
        // Check if a casting implementation already exists
        if (NPY_DT_SLOTS(meth->dtypes[0])->within_dtype_castingimpl != NULL) {
            PyErr_Format(PyExc_RuntimeError,
                    "A cast was already added for %S -> %S. (method: %s)",
                    meth->dtypes[0], meth->dtypes[1], meth->method->name);
            return -1;
        }
        // Increment the reference count of the method and set it as the casting implementation
        Py_INCREF(meth->method);
        NPY_DT_SLOTS(meth->dtypes[0])->within_dtype_castingimpl = meth->method;

        return 0;
    }
    // Check if a casting implementation already exists for the given dtypes
    if (PyDict_Contains(NPY_DT_SLOTS(meth->dtypes[0])->castingimpls,
            (PyObject *)meth->dtypes[1])) {
        PyErr_Format(PyExc_RuntimeError,
                "A cast was already added for %S -> %S. (method: %s)",
                meth->dtypes[0], meth->dtypes[1], meth->method->name);
        return -1;
    }
    // Set the casting method for the given dtypes
    if (PyDict_SetItem(NPY_DT_SLOTS(meth->dtypes[0])->castingimpls,
            (PyObject *)meth->dtypes[1], (PyObject *)meth->method) < 0) {
        return -1;
    }
    return 0;
}

/**
 * Add a new casting implementation using a PyArrayMethod_Spec.
 *
 * @param spec The PyArrayMethod_Spec defining the method specification.
 * @param private If private, allow slots not publicly exposed.
 * @return 0 on success -1 on failure
 */
NPY_NO_EXPORT int
PyArray_AddCastingImplementation_FromSpec(PyArrayMethod_Spec *spec, int private)
{
    /* Create a bound method, unbind and store it */
    // Create a PyBoundArrayMethodObject from the given specification
    PyBoundArrayMethodObject *meth = PyArrayMethod_FromSpec_int(spec, private);
    if (meth == NULL) {
        return -1;
    }
    // Add the casting implementation using the created method object
    int res = PyArray_AddCastingImplementation(meth);
    Py_DECREF(meth);
    if (res < 0) {
        return -1;
    }
    return 0;
}


NPY_NO_EXPORT NPY_CASTING
legacy_same_dtype_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    // Increment the reference count of the given descriptor
    Py_INCREF(given_descrs[0]);
    // Set the loop descriptor to the given descriptor
    loop_descrs[0] = given_descrs[0];
    # 检查给定描述符数组的第二个元素是否为空
    if (given_descrs[1] == NULL) {
        # 如果为空，通过调用函数确保第一个描述符的规范性
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(loop_descrs[0]);
        # 如果第一个描述符的规范性未能保证，则释放第一个描述符并返回错误
        if (loop_descrs[1] == NULL) {
            Py_DECREF(loop_descrs[0]);
            return -1;
        }
    }
    else {
        # 如果第二个描述符不为空，则增加其引用计数，并将其赋值给第二个循环描述符
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    # 断言，这个函数仅适用于非灵活的传统数据类型：
    assert(loop_descrs[0]->elsize == loop_descrs[1]->elsize);

    """
     传统数据类型（除了日期时间）只有字节顺序和元素大小作为存储参数。
    """

    # 如果循环描述符0和1的字节顺序相同，则设置视图偏移为0并返回NPY_NO_CASTING
    if (PyDataType_ISNOTSWAPPED(loop_descrs[0]) ==
                PyDataType_ISNOTSWAPPED(loop_descrs[1])) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }
    # 如果循环描述符0和1的字节顺序不同，则返回NPY_EQUIV_CASTING
    return NPY_EQUIV_CASTING;
}

/*
 * 获取跨度循环的遗留类型强制转换函数
 */
NPY_NO_EXPORT int
legacy_cast_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 获取描述符数组
    PyArray_Descr *const *descrs = context->descriptors;
    int out_needs_api = 0;

    // 设置方法的运行时标志
    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;

    // 获取包装的遗留类型转换函数
    if (get_wrapped_legacy_cast_function(
            aligned, strides[0], strides[1], descrs[0], descrs[1],
            move_references, out_loop, out_transferdata, &out_needs_api, 0) < 0) {
        return -1;
    }
    // 如果不需要 API，清除相应标志
    if (!out_needs_api) {
        *flags &= ~NPY_METH_REQUIRES_PYAPI;
    }
    return 0;
}

/*
 * 简单的数据类型解析器，用于两个不同（非参数化）遗留数据类型之间的强制转换
 */
NPY_NO_EXPORT NPY_CASTING
simple_cast_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    // 断言数据类型是否为遗留类型
    assert(NPY_DT_is_legacy(dtypes[0]) && NPY_DT_is_legacy(dtypes[1]));

    // 确保第一个描述符是规范的
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return -1;
    }

    // 如果给定第二个描述符，确保其是规范的
    if (given_descrs[1] != NULL) {
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
        if (loop_descrs[1] == NULL) {
            Py_DECREF(loop_descrs[0]);
            return -1;
        }
    }
    else {
        // 否则使用默认描述符
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
    }

    // 如果指定了转换类型，返回该类型
    if (self->casting != NPY_NO_CASTING) {
        return self->casting;
    }

    // 如果两个描述符的字节序相同，设置视图偏移为0，返回无需转换
    if (PyDataType_ISNOTSWAPPED(loop_descrs[0]) ==
            PyDataType_ISNOTSWAPPED(loop_descrs[1])) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }

    // 否则返回等效转换
    return NPY_EQUIV_CASTING;
}

/*
 * 获取字节交换循环的函数
 */
NPY_NO_EXPORT int
get_byteswap_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 获取描述符数组
    PyArray_Descr *const *descrs = context->descriptors;

    // 断言两个描述符的类型和元素大小相同
    assert(descrs[0]->kind == descrs[1]->kind);
    assert(descrs[0]->elsize == descrs[1]->elsize);
    int itemsize = descrs[0]->elsize;

    // 设置方法的标志为不引发浮点错误
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_transferdata = NULL;

    // 如果是复数类型，存在复数对齐问题，因此设置 aligned 为 0 可能会导致性能下降
    if (descrs[0]->kind == 'c') {
        aligned = 0;
    }

    // 如果两个描述符的字节序相同，获取相应的循环函数
    if (PyDataType_ISNOTSWAPPED(descrs[0]) ==
            PyDataType_ISNOTSWAPPED(descrs[1])) {
        *out_loop = PyArray_GetStridedCopyFn(
                aligned, strides[0], strides[1], itemsize);
    }
}
    # 如果第一个描述符不是复数类型，则执行以下操作
    else if (!PyTypeNum_ISCOMPLEX(descrs[0]->type_num)) {
        # 调用 PyArray_GetStridedCopySwapFn 函数获取适合的循环函数指针
        *out_loop = PyArray_GetStridedCopySwapFn(
                aligned, strides[0], strides[1], itemsize);
    }
    # 否则，执行以下操作
    else {
        # 调用 PyArray_GetStridedCopySwapPairFn 函数获取适合的循环函数指针
        *out_loop = PyArray_GetStridedCopySwapPairFn(
                aligned, strides[0], strides[1], itemsize);
    }
    # 如果 out_loop 为 NULL，则返回 -1 表示出错
    if (*out_loop == NULL) {
        return -1;
    }
    # 返回 0 表示成功执行
    return 0;
}

NPY_NO_EXPORT int
complex_to_noncomplex_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 发出警告，如果警告失败则返回 -1
    int ret = PyErr_WarnEx(npy_static_pydata.ComplexWarning,
            "Casting complex values to real discards "
            "the imaginary part", 1);
    if (ret < 0) {
        return -1;
    }
    // 调用默认的 strided loop 获取函数
    return npy_default_get_strided_loop(
            context, aligned, move_references, strides,
            out_loop, out_transferdata, flags);
}


static int
add_numeric_cast(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to)
{
    // 定义一个 slots 数组，长度为 7
    PyType_Slot slots[7];
    // 创建一个包含两个 DTypeMeta 元数据的数组
    PyArray_DTypeMeta *dtypes[2] = {from, to};
    // 定义一个 PyArrayMethod_Spec 结构体
    PyArrayMethod_Spec spec = {
            .name = "numeric_cast",
            .nin = 1,
            .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    // 获取 from 和 to 的元素大小
    npy_intp from_itemsize = from->singleton->elsize;
    npy_intp to_itemsize = to->singleton->elsize;

    // 设置 slots 中的不同方法及其函数指针
    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &simple_cast_resolve_descriptors;
    /* Fetch the optimized loops (2<<10 is a non-contiguous stride) */
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = PyArray_GetStridedNumericCastFn(
            1, 2<<10, 2<<10, from->type_num, to->type_num);
    slots[2].slot = NPY_METH_contiguous_loop;
    slots[2].pfunc = PyArray_GetStridedNumericCastFn(
            1, from_itemsize, to_itemsize, from->type_num, to->type_num);
    slots[3].slot = NPY_METH_unaligned_strided_loop;
    slots[3].pfunc = PyArray_GetStridedNumericCastFn(
            0, 2<<10, 2<<10, from->type_num, to->type_num);
    slots[4].slot = NPY_METH_unaligned_contiguous_loop;
    slots[4].pfunc = PyArray_GetStridedNumericCastFn(
            0, from_itemsize, to_itemsize, from->type_num, to->type_num);
    
    // 如果 from 是复数类型，to 不是复数且不是布尔类型，则发出 ComplexWarning
    if (PyTypeNum_ISCOMPLEX(from->type_num) &&
            !PyTypeNum_ISCOMPLEX(to->type_num) &&
            !PyTypeNum_ISBOOL(to->type_num)) {
        // 设置 get_loop 方法及其函数指针为 complex_to_noncomplex_get_loop
        slots[5].slot = NPY_METH_get_loop;
        slots[5].pfunc = &complex_to_noncomplex_get_loop;
        slots[6].slot = 0;
        slots[6].pfunc = NULL;
    }
    else {
        // 否则将 get_loop 方法设为空
        slots[5].slot = 0;
        slots[5].pfunc = NULL;
    }

    // 断言确保非空的函数指针
    assert(slots[1].pfunc && slots[2].pfunc && slots[3].pfunc && slots[4].pfunc);

    // 查找正确的转换级别，并特殊处理无需转换的情况
    # 检查第一个和第二个数据类型的单例类型是否相同，并且元素大小是否相等
    if (dtypes[0]->singleton->kind == dtypes[1]->singleton->kind
            && from_itemsize == to_itemsize) {
        # 设置转换规范为等效转换
        spec.casting = NPY_EQUIV_CASTING;

        /* 当没有类型转换时（C类型等效），使用字节交换循环 */
        # 设置第一个插槽为解析描述符的方法，并指定其函数为 legacy_same_dtype_resolve_descriptors
        slots[0].slot = NPY_METH_resolve_descriptors;
        slots[0].pfunc = &legacy_same_dtype_resolve_descriptors;
        # 设置第二个插槽为获取循环的方法，并指定其函数为 get_byteswap_loop
        slots[1].slot = NPY_METH_get_loop;
        slots[1].pfunc = &get_byteswap_loop;
        # 第三个插槽为空，函数指针为 NULL
        slots[2].slot = 0;
        slots[2].pfunc = NULL;

        # 设置规范名称为 "numeric_copy_or_byteswap"，并设置标志为不产生浮点数错误
        spec.name = "numeric_copy_or_byteswap";
        spec.flags |= NPY_METH_NO_FLOATINGPOINT_ERRORS;
    }
    else if (_npy_can_cast_safely_table[from->type_num][to->type_num]) {
        # 如果可以安全地从 from 类型转换到 to 类型，设置转换规范为安全转换
        spec.casting = NPY_SAFE_CASTING;
    }
    else if (dtype_kind_to_ordering(dtypes[0]->singleton->kind) <=
             dtype_kind_to_ordering(dtypes[1]->singleton->kind)) {
        # 如果第一个数据类型的种类排序值小于等于第二个，设置转换规范为相同种类转换
        spec.casting = NPY_SAME_KIND_CASTING;
    }
    else {
        # 否则，设置转换规范为不安全转换
        spec.casting = NPY_UNSAFE_CASTING;
    }

    /* 创建一个绑定方法，解绑并存储它 */
    # 使用规范对象创建一个从规范到 PyArray_AddCastingImplementation_FromSpec 的转换实现，并返回
    return PyArray_AddCastingImplementation_FromSpec(&spec, 1);
}
/*
 * This registers the castingimpl for all casts between numeric types.
 * Eventually, this function should likely be defined as part of a .c.src
 * file to remove `PyArray_GetStridedNumericCastFn` entirely.
 */
static int
PyArray_InitializeNumericCasts(void)
{
    // 遍历所有旧版数据类型
    for (int from = 0; from < NPY_NTYPES_LEGACY; from++) {
        // 如果不是数字类型且不是布尔类型，跳过
        if (!PyTypeNum_ISNUMBER(from) && from != NPY_BOOL) {
            continue;
        }
        // 根据旧版数据类型获取数据类型元数据
        PyArray_DTypeMeta *from_dt = PyArray_DTypeFromTypeNum(from);

        // 再次遍历所有旧版数据类型
        for (int to = 0; to < NPY_NTYPES_LEGACY; to++) {
            // 如果不是数字类型且不是布尔类型，跳过
            if (!PyTypeNum_ISNUMBER(to) && to != NPY_BOOL) {
                continue;
            }
            // 根据旧版数据类型获取数据类型元数据
            PyArray_DTypeMeta *to_dt = PyArray_DTypeFromTypeNum(to);
            // 添加从一个数据类型到另一个数据类型的数值转换
            int res = add_numeric_cast(from_dt, to_dt);
            // 减少目标数据类型的引用计数
            Py_DECREF(to_dt);
            // 如果添加转换失败，减少源数据类型的引用计数并返回错误
            if (res < 0) {
                Py_DECREF(from_dt);
                return -1;
            }
        }
        // 减少源数据类型的引用计数
        Py_DECREF(from_dt);
    }
    // 成功注册所有数值类型转换，返回 0
    return 0;
}


static int
cast_to_string_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * NOTE: The following code used to be part of PyArray_AdaptFlexibleDType
     *
     * Get a string-size estimate of the input. These
     * are generally the size needed, rounded up to
     * a multiple of eight.
     */
    // 初始化字符串转换过程中的描述符解析
    npy_intp size = -1;
    switch (given_descrs[0]->type_num) {
        case NPY_BOOL:
        case NPY_UBYTE:
        case NPY_BYTE:
        case NPY_USHORT:
        case NPY_SHORT:
        case NPY_UINT:
        case NPY_INT:
        case NPY_ULONG:
        case NPY_LONG:
        case NPY_ULONGLONG:
        case NPY_LONGLONG:
            // 确保元素大小在 1 到 8 之间
            assert(given_descrs[0]->elsize <= 8);
            assert(given_descrs[0]->elsize > 0);
            if (given_descrs[0]->kind == 'b') {
                /* 5 chars needed for cast to 'True' or 'False' */
                // 对于布尔类型，字符串长度需为 5（"True" 或 "False"）
                size = 5;
            }
            else if (given_descrs[0]->kind == 'u') {
                // 对于无符号整数类型，字符串长度根据给定的元素大小确定
                size = REQUIRED_STR_LEN[given_descrs[0]->elsize];
            }
            else if (given_descrs[0]->kind == 'i') {
                /* Add character for sign symbol */
                // 对于有符号整数类型，字符串长度为元素大小加上符号位的长度
                size = REQUIRED_STR_LEN[given_descrs[0]->elsize] + 1;
            }
            break;
        case NPY_HALF:
        case NPY_FLOAT:
        case NPY_DOUBLE:
            // 对于浮点数类型，固定字符串长度为 32
            size = 32;
            break;
        case NPY_LONGDOUBLE:
            // 对于长双精度浮点数类型，固定字符串长度为 48
            size = 48;
            break;
        case NPY_CFLOAT:
        case NPY_CDOUBLE:
            // 对于复数浮点数类型，字符串长度为实部和虚部各为 32 的两倍
            size = 2 * 32;
            break;
        case NPY_CLONGDOUBLE:
            // 对于复数长双精度浮点数类型，字符串长度为实部和虚部各为 48 的两倍
            size = 2 * 48;
            break;
        case NPY_STRING:
        case NPY_VOID:
            // 对于字符串或 void 类型，字符串长度等于元素大小
            size = given_descrs[0]->elsize;
            break;
        case NPY_UNICODE:
            // 对于 Unicode 类型，字符串长度为元素大小除以 4
            size = given_descrs[0]->elsize / 4;
            break;
        default:
            // 若请求了不可能的字符串路径转换，则设置异常并返回 -1
            PyErr_SetString(PyExc_SystemError,
                    "Impossible cast to string path requested.");
            return -1;
    }
    if (dtypes[1]->type_num == NPY_UNICODE) {
        // 若第二个数据类型为 Unicode，则字符串长度需乘以 4
        size *= 4;
    }

    if (given_descrs[1] == NULL) {
        // 若第二个描述符为空，创建一个新的描述符，指定元素大小为计算出的字符串长度
        loop_descrs[1] = PyArray_DescrNewFromType(dtypes[1]->type_num);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
        loop_descrs[1]->elsize = size;
    }
    else {
        /* The legacy loop can handle mismatching itemsizes */
        // 使用遗留循环处理不匹配的元素大小
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }

    /* Set the input one as well (late for easier error management) */
    // 设置第一个描述符为规范化后的给定描述符，以便更容易管理错误
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return -1;
    }

    if (self->casting == NPY_UNSAFE_CASTING) {
        assert(dtypes[0]->type_num == NPY_UNICODE &&
               dtypes[1]->type_num == NPY_STRING);
        // 若转换模式为不安全转换，且第一个数据类型为 Unicode，第二个为字符串，则返回不安全转换标志
        return NPY_UNSAFE_CASTING;
    }

    if (loop_descrs[1]->elsize >= size) {
        // 若第二个描述符的元素大小大于等于计算出的字符串长度，则返回安全转换标志
        return NPY_SAFE_CASTING;
    }
    // 否则返回相同类型的转换标志
    return NPY_SAME_KIND_CASTING;
# 结束函数 add_other_to_and_from_string_cast 的定义

static int
add_other_to_and_from_string_cast(
        PyArray_DTypeMeta *string, PyArray_DTypeMeta *other)
{
    # 如果 string 和 other 相同，无需进行任何转换
    if (string == other) {
        return 0;
    }

    /* Casting from string, is always a simple legacy-style cast */
    # 如果 other 的类型不是 NPY_STRING 也不是 NPY_UNICODE，进行非安全转换
    if (other->type_num != NPY_STRING && other->type_num != NPY_UNICODE) {
        # 尝试添加一个遗留风格的转换实现
        if (PyArray_AddLegacyWrapping_CastingImpl(
                string, other, NPY_UNSAFE_CASTING) < 0) {
            return -1;
        }
    }
    /*
     * Casting to strings, is almost the same, but requires a custom resolver
     * to define the correct string length. Right now we use a generic function
     * for this.
     */
    # 设置一个包含 other 和 string 的数组，用于定义类型方法的插槽
    PyArray_DTypeMeta *dtypes[2] = {other, string};
    PyType_Slot slots[] = {
            {NPY_METH_get_loop, &legacy_cast_get_strided_loop},
            {NPY_METH_resolve_descriptors, &cast_to_string_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
        .name = "legacy_cast_to_string",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS,
        .dtypes = dtypes,
        .slots = slots,
    };
    /* Almost everything can be same-kind cast to string (except unicode) */
    # 如果 other 的类型不是 NPY_UNICODE，采用相同类型转换；否则采用非安全转换
    if (other->type_num != NPY_UNICODE) {
        spec.casting = NPY_SAME_KIND_CASTING;  /* same-kind if too short */
    }
    else {
        spec.casting = NPY_UNSAFE_CASTING;
    }

    # 将转换规范添加到转换实现中
    return PyArray_AddCastingImplementation_FromSpec(&spec, 1);
}


NPY_NO_EXPORT NPY_CASTING
string_to_string_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    # 增加给定描述符的引用计数，并将其分配给循环描述符
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    # 如果给定描述符的第二个元素为空，确保循环描述符能够规范化
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = NPY_DT_CALL_ensure_canonical(loop_descrs[0]);
        # 如果规范化失败，返回错误标志
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }
    else {
        # 增加给定描述符的引用计数，并将其分配给循环描述符
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    # 如果新字符串的长度大于旧字符串的长度，进行安全转换
    if (loop_descrs[0]->elsize < loop_descrs[1]->elsize) {
        /* New string is longer: safe but cannot be a view */
        return NPY_SAFE_CASTING;
    }
    else {
        # 如果字节顺序匹配，可以视为视图，进行相同类型转换
        int not_swapped = (PyDataType_ISNOTSWAPPED(loop_descrs[0])
                           == PyDataType_ISNOTSWAPPED(loop_descrs[1]));
        if (not_swapped) {
            *view_offset = 0;
        }

        # 如果新字符串的长度小于旧字符串的长度，进行相同类型转换
        if (loop_descrs[0]->elsize > loop_descrs[1]->elsize) {
            return NPY_SAME_KIND_CASTING;
        }
        /* The strings have the same length: */
        # 如果字节顺序匹配，进行无需转换；否则进行等效转换
        if (not_swapped) {
            return NPY_NO_CASTING;
        }
        else {
            return NPY_EQUIV_CASTING;
        }
    }
}


NPY_NO_EXPORT int
/*
 * Function: string_to_string_get_loop
 * -------------------------------------------------------------
 * This function sets up a strided loop for string operations based on provided context and descriptors.
 * It determines if Unicode swapping is necessary, initializes flags, and retrieves strided copy functions
 * for zero-padding. Returns -1 on failure and 0 on success.
 *
 * Parameters:
 *     context: Pointer to PyArrayMethod_Context containing method context and descriptors
 *     aligned: Flag indicating alignment status of data
 *     strides: Pointer to array of stride values
 *     out_loop: Output parameter for the strided loop function pointer
 *     out_transferdata: Output parameter for auxiliary data associated with the strided loop
 *     flags: Pointer to NPY_ARRAYMETHOD_FLAGS for storing method flags
 *
 * Returns:
 *     Returns -1 if PyArray_GetStridedZeroPadCopyFn fails, otherwise returns 0.
 */
string_to_string_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    int unicode_swap = 0;
    PyArray_Descr *const *descrs = context->descriptors;

    // Assert that descriptors of the two operands are of the same data type
    assert(NPY_DTYPE(descrs[0]) == NPY_DTYPE(descrs[1]));

    // Set flags based on runtime flags of the method context
    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;

    // Check if the data type is NPY_UNICODE and determine if Unicode swapping is required
    if (descrs[0]->type_num == NPY_UNICODE) {
        if (PyDataType_ISNOTSWAPPED(descrs[0]) !=
                PyDataType_ISNOTSWAPPED(descrs[1])) {
            unicode_swap = 1;
        }
    }

    // Retrieve the strided zero-padding copy function
    if (PyArray_GetStridedZeroPadCopyFn(
            aligned, unicode_swap, strides[0], strides[1],
            descrs[0]->elsize, descrs[1]->elsize,
            out_loop, out_transferdata) == NPY_FAIL) {
        return -1;  // Return -1 on failure
    }
    return 0;  // Return 0 on success
}

/*
 * Function: PyArray_InitializeStringCasts
 * -------------------------------------------------------------
 * Initializes string casts for NumPy array types, excluding legacy types such as datetime and object.
 * Adds legacy and specialized casts for string<->string and unicode<->unicode conversions.
 *
 * Returns:
 *     Returns -1 on failure, otherwise returns success status.
 */
static int
PyArray_InitializeStringCasts(void)
{
    int result = -1;
    PyArray_DTypeMeta *string = &PyArray_BytesDType;
    PyArray_DTypeMeta *unicode = &PyArray_UnicodeDType;
    PyArray_DTypeMeta *other_dt = NULL;

    /* Add most casts as legacy ones */
    for (int other = 0; other < NPY_NTYPES_LEGACY; other++) {
        // Skip certain types like datetime, void, and object
        if (PyTypeNum_ISDATETIME(other) || other == NPY_VOID ||
                other == NPY_OBJECT) {
            continue;
        }
        // Get data type meta information for the current type number
        other_dt = PyArray_DTypeFromTypeNum(other);

        // Add string <-> other_dt and unicode <-> other_dt casts
        if (add_other_to_and_from_string_cast(string, other_dt) < 0) {
            goto finish;  // Jump to finish label on failure
        }
        if (add_other_to_and_from_string_cast(unicode, other_dt) < 0) {
            goto finish;  // Jump to finish label on failure
        }

        Py_SETREF(other_dt, NULL);  // Clear reference to other_dt
    }

    /* string<->string and unicode<->unicode have their own specialized casts */
    PyArray_DTypeMeta *dtypes[2];
    PyType_Slot slots[] = {
            {NPY_METH_get_loop, &string_to_string_get_loop},
            {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = "string_to_string_cast",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = (NPY_METH_REQUIRES_PYAPI |
                      NPY_METH_NO_FLOATINGPOINT_ERRORS |
                      NPY_METH_SUPPORTS_UNALIGNED),
            .dtypes = dtypes,
            .slots = slots,
    };

    dtypes[0] = string;
    dtypes[1] = string;

    // Add casting implementation from the provided spec
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        goto finish;  // Jump to finish label on failure
    }

finish:
    return result;  // Return result (-1 on failure by default)
}
    # 设置第一个元素的数据类型为 unicode
    dtypes[0] = unicode;
    # 设置第二个元素的数据类型为 unicode
    dtypes[1] = unicode;
    # 尝试添加一个从给定规范(spec)创建的类型转换实现到数组中
    if (PyArray_AddCastingImplementation_FromSpec(&spec, 1) < 0) {
        # 如果添加失败，则跳转到完成标签
        goto finish;
    }

    # 将结果设置为 0，表示成功
    result = 0;
  finish:
    # 清理并释放 other_dt 对象
    Py_XDECREF(other_dt);
    # 返回操作结果
    return result;
/*
 * Small helper function to handle the case of `arr.astype(dtype="V")`.
 * When the output descriptor is not passed, we always use `V<itemsize>`
 * of the other dtype.
 */
static NPY_CASTING
cast_to_void_dtype_class(
        PyArray_Descr *const *given_descrs, PyArray_Descr **loop_descrs,
        npy_intp *view_offset)
{
    /* `dtype="V"` means unstructured currently (compare final path) */
    // 将 loop_descrs[1] 设为 NPY_VOID 类型的新描述符
    loop_descrs[1] = PyArray_DescrNewFromType(NPY_VOID);
    if (loop_descrs[1] == NULL) {
        return -1;
    }
    // 设置新描述符的元素大小为给定描述符的元素大小
    loop_descrs[1]->elsize = given_descrs[0]->elsize;
    // 增加给定描述符的引用计数
    Py_INCREF(given_descrs[0]);
    // 将 loop_descrs[0] 设为给定的第一个描述符
    loop_descrs[0] = given_descrs[0];

    // 将视图偏移设置为 0
    *view_offset = 0;
    // 如果给定的第一个描述符是 NPY_VOID 类型且不是子数组，且第二个描述符没有字段名
    if (loop_descrs[0]->type_num == NPY_VOID &&
            PyDataType_SUBARRAY(loop_descrs[0]) == NULL &&
            PyDataType_NAMES(loop_descrs[1]) == NULL) {
        return NPY_NO_CASTING;  // 没有转换
    }
    return NPY_SAFE_CASTING;  // 安全转换
}


static NPY_CASTING
nonstructured_to_structured_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    NPY_CASTING casting;

    // 如果给定的第二个描述符为空，则调用 cast_to_void_dtype_class 处理
    if (given_descrs[1] == NULL) {
        return cast_to_void_dtype_class(given_descrs, loop_descrs, view_offset);
    }

    // 将 from_descr 设为给定的第一个描述符，to_descr 设为给定的第二个描述符
    PyArray_Descr *from_descr = given_descrs[0];
    _PyArray_LegacyDescr *to_descr = (_PyArray_LegacyDescr *)given_descrs[1];

    // 如果 to_descr 有子数组
    if (to_descr->subarray != NULL) {
        /*
         * We currently consider this at most a safe cast. It would be
         * possible to allow a view if the field has exactly one element.
         */
        // 当前我们认为这是最安全的转换。如果字段恰好有一个元素，则可以允许视图
        casting = NPY_SAFE_CASTING;
        npy_intp sub_view_offset = NPY_MIN_INTP;
        // 子数组的数据类型
        NPY_CASTING base_casting = PyArray_GetCastInfo(
                from_descr, to_descr->subarray->base, NULL,
                &sub_view_offset);
        if (base_casting < 0) {
            return -1;
        }
        // 如果子数组的元素大小等于子数组的基本元素大小
        if (to_descr->elsize == to_descr->subarray->base->elsize) {
            /* A single field, view is OK if sub-view is */
            // 单个字段，如果子视图是 OK 的，则视图偏移为子视图偏移
            *view_offset = sub_view_offset;
        }
        // 取安全转换的最小值
        casting = PyArray_MinCastSafety(casting, base_casting);
    }
    else if (to_descr->names != NULL) {
        /* 如果目标描述符具有字段名，表示结构化数据类型 */
        if (PyTuple_Size(to_descr->names) == 0) {
            /* 如果字段名元组为空，保留当前行为，但可能需要修改 */
            casting = NPY_UNSAFE_CASTING;
        }
        else {
            /* 考虑最不安全的转换方式（但这可能会改变） */
            casting = NPY_UNSAFE_CASTING;

            Py_ssize_t pos = 0;
            PyObject *key, *tuple;
            while (PyDict_Next(to_descr->fields, &pos, &key, &tuple)) {
                /* 获取字段描述符 */
                PyArray_Descr *field_descr = (PyArray_Descr *)PyTuple_GET_ITEM(tuple, 0);
                /* 初始化字段视图偏移量 */
                npy_intp field_view_off = NPY_MIN_INTP;
                /* 获取字段之间的转换信息 */
                NPY_CASTING field_casting = PyArray_GetCastInfo(
                        from_descr, field_descr, NULL, &field_view_off);
                /* 更新整体的最安全转换方式 */
                casting = PyArray_MinCastSafety(casting, field_casting);
                /* 如果转换不安全，返回错误 */
                if (casting < 0) {
                    return -1;
                }
                /* 如果存在视图偏移量 */
                if (field_view_off != NPY_MIN_INTP) {
                    /* 获取目标偏移量 */
                    npy_intp to_off = PyLong_AsSsize_t(PyTuple_GET_ITEM(tuple, 1));
                    /* 如果转换偏移量失败，返回错误 */
                    if (error_converting(to_off)) {
                        return -1;
                    }
                    /* 计算视图偏移量 */
                    *view_offset = field_view_off - to_off;
                }
            }
            /* 如果字段名元组长度不为1或视图偏移量小于0 */
            if (PyTuple_Size(to_descr->names) != 1 || *view_offset < 0) {
                /*
                 * 假设当存在多个字段时无法创建视图。
                 * （字段可能重叠，但这看起来很奇怪...）
                 */
                *view_offset = NPY_MIN_INTP;
            }
        }
    }
    else {
        /* 简单的 void 类型，类似于“视图” */
        if (from_descr->elsize == to_descr->elsize &&
                !PyDataType_REFCHK(from_descr)) {
            /*
             * 简单视图，目前被认为是“安全”的（引用检查可能不是必要的，
             * 但更具未来的兼容性）
             */
            *view_offset = 0;
            casting = NPY_SAFE_CASTING;
        }
        else if (from_descr->elsize <= to_descr->elsize) {
            casting = NPY_SAFE_CASTING;
        }
        else {
            casting = NPY_UNSAFE_CASTING;
            /* 新的元素大小较小，因此视图是可以接受的（目前不接受引用） */
            if (!PyDataType_REFCHK(from_descr)) {
                *view_offset = 0;
            }
        }
    }

    /* Void 类型总是进行完整的转换 */
    Py_INCREF(from_descr);
    loop_descrs[0] = from_descr;
    Py_INCREF(to_descr);
    loop_descrs[1] = (PyArray_Descr *)to_descr;

    /* 返回最终确定的转换方式 */
    return casting;
}


int give_bad_field_error(PyObject *key)
{
    // 如果没有其他异常发生，设置一个运行时错误，指出无效或丢失的字段，可能是 NumPy 的 bug
    if (!PyErr_Occurred()) {
        PyErr_Format(PyExc_RuntimeError,
                "Invalid or missing field %R, this should be impossible "
                "and indicates a NumPy bug.", key);
    }
    // 返回错误代码 -1
    return -1;
}


static int
nonstructured_to_structured_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 如果第二个描述符是结构化数据类型
    if (PyDataType_NAMES(context->descriptors[1]) != NULL) {
        // 调用获取字段转换函数，如果失败则返回错误代码 -1
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    // 如果第二个描述符是子数组类型
    else if (PyDataType_SUBARRAY(context->descriptors[1]) != NULL) {
        // 调用获取子数组转换函数，如果失败则返回错误代码 -1
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    // 如果以上两种情况都不满足
    else {
        /*
         * TODO: This could be a simple zero padded cast, adding a decref
         *       in case of `move_references`. But for now use legacy casts
         *       (which is the behaviour at least up to 1.20).
         */
        // 选择使用旧的类型转换函数，如果失败则返回错误代码 -1
        int needs_api = 0;
        if (get_wrapped_legacy_cast_function(
                1, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api, 1) < 0) {
            return -1;
        }
        // 根据需要设置标志，表示是否需要 Python API
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    // 成功执行，返回 0 表示无错误
    return 0;
}

static PyObject *
PyArray_GetGenericToVoidCastingImpl(void)
{
    // 增加对通用到空类型的引用计数并返回
    Py_INCREF(npy_static_pydata.GenericToVoidMethod);
    return npy_static_pydata.GenericToVoidMethod;
}


static NPY_CASTING
structured_to_nonstructured_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    PyArray_Descr *base_descr;
    /* The structured part may allow a view (and have its own offset): */
    // 结构化部分可能允许视图（并具有自己的偏移量）：
    npy_intp struct_view_offset = NPY_MIN_INTP;

    // 如果给定的第一个描述符是子数组类型
    if (PyDataType_SUBARRAY(given_descrs[0]) != NULL) {
        base_descr = PyDataType_SUBARRAY(given_descrs[0])->base;
        // 如果子数组的大小等于其基本类型的大小，则可能存在视图
        if (given_descrs[0]->elsize == PyDataType_SUBARRAY(given_descrs[0])->base->elsize) {
            struct_view_offset = 0;
        }
    }
    /*
     * 如果第一个描述符的 PyDataType_NAMES 不为空，则执行以下操作
     * 这表示正在尝试进行结构化的数据类型转换
     */
    else if (PyDataType_NAMES(given_descrs[0]) != NULL) {
        /*
         * 确保只允许对单个字段进行类型转换
         */
        if (PyTuple_Size(PyDataType_NAMES(given_descrs[0])) != 1) {
            /* 只允许转换单个字段 */
            return -1;
        }
        // 获取第一个字段的名称作为关键字
        PyObject *key = PyTuple_GetItem(PyDataType_NAMES(given_descrs[0]), 0);
        // 在字段字典中查找关键字对应的元组
        PyObject *base_tup = PyDict_GetItem(PyDataType_FIELDS(given_descrs[0]), key);
        // 获取基础描述符和视图偏移量
        base_descr = (PyArray_Descr *)PyTuple_GET_ITEM(base_tup, 0);
        struct_view_offset = PyLong_AsSsize_t(PyTuple_GET_ITEM(base_tup, 1));
        // 如果转换视图偏移量出错，则返回错误
        if (error_converting(struct_view_offset)) {
            return -1;
        }
    }
    else {
        /*
         * 非结构化的 void 类型被视为不安全的转换，并定义了后向兼容的行为，
         * 此时它们使用 getitem/setitem 返回到传统的行为。
         */
        base_descr = NULL;
        struct_view_offset = 0;
    }

    /*
     * 转换始终被视为不安全，因此 PyArray_GetCastInfo 的结果目前仅关注视图偏移量。
     */
    npy_intp base_view_offset = NPY_MIN_INTP;
    // 如果基础描述符不为空，并且转换失败，则返回错误
    if (base_descr != NULL && PyArray_GetCastInfo(
            base_descr, given_descrs[1], dtypes[1], &base_view_offset) < 0) {
        return -1;
    }
    // 如果基础视图偏移量和结构视图偏移量均不是最小整数值，则计算视图偏移量
    if (base_view_offset != NPY_MIN_INTP
            && struct_view_offset != NPY_MIN_INTP) {
        *view_offset = base_view_offset + struct_view_offset;
    }

    /* Void 类型总是进行完整的转换。*/
    if (given_descrs[1] == NULL) {
        // 获取默认的描述符用于循环
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
        // 如果获取失败，则返回错误
        if (loop_descrs[1] == NULL) {
            return -1;
        }
        /*
         * 对于字符串特殊处理，这在实际上只对空数组有效，可能应该对所有参数化的数据类型
         * 抛出异常。
         */
        if (dtypes[1]->type_num == NPY_STRING) {
            loop_descrs[1]->elsize = given_descrs[0]->elsize;
        }
        else if (dtypes[1]->type_num == NPY_UNICODE) {
            loop_descrs[1]->elsize = given_descrs[0]->elsize * 4;
        }
    }
    else {
        // 增加给定描述符的引用计数，用于循环
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    // 增加给定描述符的引用计数，用于循环
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // 返回不安全转换标记
    return NPY_UNSAFE_CASTING;
}

static int
structured_to_nonstructured_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    // 检查第一个描述符的类型是否为结构体
    if (PyDataType_NAMES(context->descriptors[0]) != NULL) {
        // 如果是结构体，获取结构体字段之间的转换函数
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    // 检查第一个描述符的类型是否为子数组
    else if (PyDataType_SUBARRAY(context->descriptors[0]) != NULL) {
        // 如果是子数组，获取子数组之间的转换函数
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;
        }
    }
    else {
        /*
         * 一般情况下，这个分支通过对标量的遗留行为定义，很可能不应该被允许。
         */
        int needs_api = 0;
        // 获取包装的遗留类型转换函数
        if (get_wrapped_legacy_cast_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                &needs_api, 1) < 0) {
            return -1;
        }
        // 根据需要API的情况设置标志位
        *flags = needs_api ? NPY_METH_REQUIRES_PYAPI : 0;
    }
    return 0;
}


static PyObject *
PyArray_GetVoidToGenericCastingImpl(void)
{
    // 增加空指针到通用方法的引用计数，并返回其引用
    Py_INCREF(npy_static_pydata.VoidToGenericMethod);
    return npy_static_pydata.VoidToGenericMethod;
}


/*
 * 查找正确的字段转换安全性。参见下面的TODO注释，在1.20版（及以后）中，应基于字段名称而不是字段顺序来确定。
 *
 * 注意：理论上可以在dtype上缓存所有字段转换实现，以避免重复工作。
 */
static NPY_CASTING
can_cast_fields_safety(
        PyArray_Descr *from, PyArray_Descr *to, npy_intp *view_offset)
{
    Py_ssize_t field_count = PyTuple_Size(PyDataType_NAMES(from));
    // 如果源dtype和目标dtype的字段数不同，则返回不允许转换
    if (field_count != PyTuple_Size(PyDataType_NAMES(to))) {
        return -1;
    }

    NPY_CASTING casting = NPY_NO_CASTING;
    *view_offset = 0;  // 如果没有字段，则视图是允许的。
    for (Py_ssize_t i = 0; i < field_count; i++) {
        // 初始化字段视图偏移量为最小整数值
        npy_intp field_view_off = NPY_MIN_INTP;
        // 获取源数据类型的字段名元组的第i项作为键值
        PyObject *from_key = PyTuple_GET_ITEM(PyDataType_NAMES(from), i);
        // 在源数据类型的字段字典中查找对应键值的条目
        PyObject *from_tup = PyDict_GetItemWithError(PyDataType_FIELDS(from), from_key);
        // 如果查找失败，返回相应的字段错误
        if (from_tup == NULL) {
            return give_bad_field_error(from_key);
        }
        // 获取源数据类型的字段描述符
        PyArray_Descr *from_base = (PyArray_Descr *) PyTuple_GET_ITEM(from_tup, 0);

        /* 检查字段名是否匹配 */
        // 获取目标数据类型的字段名元组的第i项作为键值
        PyObject *to_key = PyTuple_GET_ITEM(PyDataType_NAMES(to), i);
        // 在目标数据类型的字段字典中查找对应键值的条目
        PyObject *to_tup = PyDict_GetItem(PyDataType_FIELDS(to), to_key);
        // 如果查找失败，返回相应的字段错误
        if (to_tup == NULL) {
            return give_bad_field_error(from_key);
        }
        // 获取目标数据类型的字段描述符
        PyArray_Descr *to_base = (PyArray_Descr *) PyTuple_GET_ITEM(to_tup, 0);

        // 比较源字段名和目标字段名
        int cmp = PyUnicode_Compare(from_key, to_key);
        // 如果比较出错，返回-1
        if (error_converting(cmp)) {
            return -1;
        }
        // 如果字段名不相等，考虑最多为安全类型转换
        if (cmp != 0) {
            /* 字段名不匹配，考虑至多安全转换 */
            casting = PyArray_MinCastSafety(casting, NPY_SAFE_CASTING);
        }

        /* 同样检查标题（仅将不匹配视为安全） */
        // 设置源标题和目标标题
        PyObject *from_title = from_key;
        PyObject *to_title = to_key;
        // 如果源元组的大小大于2，则使用第三项作为标题
        if (PyTuple_GET_SIZE(from_tup) > 2) {
            from_title = PyTuple_GET_ITEM(from_tup, 2);
        }
        // 如果目标元组的大小大于2，则使用第三项作为标题
        if (PyTuple_GET_SIZE(to_tup) > 2) {
            to_title = PyTuple_GET_ITEM(to_tup, 2);
        }
        // 比较源标题和目标标题
        cmp = PyObject_RichCompareBool(from_title, to_title, Py_EQ);
        // 如果比较出错，返回-1
        if (error_converting(cmp)) {
            return -1;
        }
        // 如果标题不相等，考虑最多为安全类型转换
        if (!cmp) {
            casting = PyArray_MinCastSafety(casting, NPY_SAFE_CASTING);
        }

        // 获取字段转换的信息和视图偏移量
        NPY_CASTING field_casting = PyArray_GetCastInfo(
                from_base, to_base, NULL, &field_view_off);
        // 如果获取转换信息失败，返回-1
        if (field_casting < 0) {
            return -1;
        }
        // 考虑最多为安全类型转换
        casting = PyArray_MinCastSafety(casting, field_casting);

        /* 根据字段偏移量调整“视图偏移量”： */
        // 如果字段视图偏移量不是最小整数值
        if (field_view_off != NPY_MIN_INTP) {
            // 获取目标元组的第二项作为目标偏移量
            npy_intp to_off = PyLong_AsSsize_t(PyTuple_GET_ITEM(to_tup, 1));
            // 如果转换偏移量出错，返回-1
            if (error_converting(to_off)) {
                return -1;
            }
            // 获取源元组的第二项作为源偏移量
            npy_intp from_off = PyLong_AsSsize_t(PyTuple_GET_ITEM(from_tup, 1));
            // 如果转换偏移量出错，返回-1
            if (error_converting(from_off)) {
                return -1;
            }
            // 根据字段偏移量调整视图偏移量
            field_view_off = field_view_off - to_off + from_off;
        }

        /*
         * 如果只有一个字段，使用其字段偏移量。
         * 否则，如果视图偏移量匹配，则传播它，并将其设置为“无效”。
         */
        // 如果是第一个字段，将其字段偏移量赋给视图偏移量
        if (i == 0) {
            *view_offset = field_view_off;
        }
        // 否则，如果视图偏移量不等于字段偏移量，将视图偏移量设置为最小整数值
        else if (*view_offset != field_view_off) {
            *view_offset = NPY_MIN_INTP;
        }
    }
    # 如果视图偏移量不为零或者源数据类型的大小与目标数据类型的大小不相等
    if (*view_offset != 0 || from->elsize != to->elsize) {
        # 需要进行类型转换，不能认为是“无”转换
        casting = PyArray_MinCastSafety(casting, NPY_EQUIV_CASTING);
    }

    # 新的数据类型可能由于填充而使得访问超出旧数据类型的范围：
    # 如果视图偏移量为负数
    if (*view_offset < 0) {
        # 负偏移量会导致在原始数据类型之前间接访问
        *view_offset = NPY_MIN_INTP;
    }
    # 如果源数据类型的大小小于目标数据类型的大小加上视图偏移量
    if (from->elsize < to->elsize + *view_offset) {
        # 新数据类型的访问超出了原始数据类型的范围
        *view_offset = NPY_MIN_INTP;
    }

    # 返回转换策略
    return casting;
    }



    static NPY_CASTING
    void_to_void_resolve_descriptors(
            PyArrayMethodObject *self,
            PyArray_DTypeMeta *const dtypes[2],
            PyArray_Descr *const given_descrs[2],
            PyArray_Descr *loop_descrs[2],
            npy_intp *view_offset)
    {
        NPY_CASTING casting;

        if (given_descrs[1] == NULL) {
            /* 如果第二个给定的描述符为空，则调用特定函数进行转换 */
            return cast_to_void_dtype_class(given_descrs, loop_descrs, view_offset);
        }

        if (PyDataType_NAMES(given_descrs[0]) != NULL && PyDataType_NAMES(given_descrs[1]) != NULL) {
            /* 如果两个描述符都是结构化的，需要检查字段 */
            casting = can_cast_fields_safety(
                    given_descrs[0], given_descrs[1], view_offset);
            if (casting < 0) {
                return -1;
            }
        }
        else if (PyDataType_NAMES(given_descrs[0]) != NULL) {
            /* 如果第一个描述符是结构化的，而第二个不是，则调用相应函数处理 */
            return structured_to_nonstructured_resolve_descriptors(
                    self, dtypes, given_descrs, loop_descrs, view_offset);
        }
        else if (PyDataType_NAMES(given_descrs[1]) != NULL) {
            /* 如果第二个描述符是结构化的，而第一个不是，则调用相应函数处理 */
            return nonstructured_to_structured_resolve_descriptors(
                    self, dtypes, given_descrs, loop_descrs, view_offset);
        }
        else if (PyDataType_SUBARRAY(given_descrs[0]) == NULL &&
                    PyDataType_SUBARRAY(given_descrs[1]) == NULL) {
            /* 如果两个描述符都是简单的空类型 */
            if (given_descrs[0]->elsize == given_descrs[1]->elsize) {
                casting = NPY_NO_CASTING;
                *view_offset = 0;
            }
            else if (given_descrs[0]->elsize < given_descrs[1]->elsize) {
                casting = NPY_SAFE_CASTING;
            }
            else {
                casting = NPY_SAME_KIND_CASTING;
                *view_offset = 0;
            }
        }
    else {
        /*
         * 此时，其中一个数据类型必须是子数组数据类型，另一个肯定不是结构化数据类型。
         */
        // 从给定的描述符中获取第一个子数组描述符
        PyArray_ArrayDescr *from_sub = PyDataType_SUBARRAY(given_descrs[0]);
        // 从给定的描述符中获取第二个子数组描述符
        PyArray_ArrayDescr *to_sub = PyDataType_SUBARRAY(given_descrs[1]);
        // 断言至少有一个是子数组描述符
        assert(from_sub || to_sub);

        /* 如果形状不匹配，则至多是一个不安全的强制转换 */
        casting = NPY_UNSAFE_CASTING;
        
        /*
         * 有两种情况可以使用视图：
         * 1. 形状和元素大小匹配，因此任何视图偏移都适用于子数组的每个元素。
         *    （实际上，这可能意味着`view_offset`将为0）
         * 2. 存在仅一个元素，并且子数组没有影响
         *    （可以通过检查基本的元素大小是否匹配来测试）
         */
        npy_bool subarray_layout_supports_view = NPY_FALSE;
        if (from_sub && to_sub) {
            // 比较两个子数组的形状是否相等
            int res = PyObject_RichCompareBool(from_sub->shape, to_sub->shape, Py_EQ);
            if (res < 0) {
                return -1;
            }
            else if (res) {
                /* 两者都是子数组且形状匹配，可能不需要转换 */
                casting = NPY_NO_CASTING;
                /* 如果有一个元素或元素大小匹配，可能是视图 */
                if (from_sub->base->elsize == to_sub->base->elsize
                        || given_descrs[0]->elsize == from_sub->base->elsize) {
                    subarray_layout_supports_view = NPY_TRUE;
                }
            }
        }
        else if (from_sub) {
            /* 如果“from”只有一个元素，则可能使用视图 */
            if (given_descrs[0]->elsize == from_sub->base->elsize) {
                subarray_layout_supports_view = NPY_TRUE;
            }
        }
        else {
            /* 如果“to”只有一个元素，则可能使用视图 */
            if (given_descrs[1]->elsize == to_sub->base->elsize) {
                subarray_layout_supports_view = NPY_TRUE;
            }
        }

        // 获取“from”和“to”的基础描述符
        PyArray_Descr *from_base = (from_sub == NULL) ? given_descrs[0] : from_sub->base;
        PyArray_Descr *to_base = (to_sub == NULL) ? given_descrs[1] : to_sub->base;
        
        /* 为字段转换获取一个偏移量 */
        NPY_CASTING field_casting = PyArray_GetCastInfo(
                from_base, to_base, NULL, view_offset);
        
        // 如果不支持子数组布局的视图，则将视图偏移设置为最小整数值
        if (!subarray_layout_supports_view) {
            *view_offset = NPY_MIN_INTP;
        }
        
        // 如果获取字段转换失败，则返回-1
        if (field_casting < 0) {
            return -1;
        }
        
        // 计算强制转换类型
        casting = PyArray_MinCastSafety(casting, field_casting);
    }

    /* Void 数据类型始终执行完全转换。*/
    // 增加“from”描述符的引用计数，并将其设置为循环描述符的第一个
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    // 增加“to”描述符的引用计数，并将其设置为循环描述符的第二个
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    // 返回计算得到的强制转换类型
    return casting;
}

NPY_NO_EXPORT int
void_to_void_get_loop(
        PyArrayMethod_Context *context,  // PyArray 方法的执行上下文
        int aligned, int move_references,  // 对齐标志和移动引用标志
        const npy_intp *strides,  // 数组步幅
        PyArrayMethod_StridedLoop **out_loop,  // 输出的循环函数指针
        NpyAuxData **out_transferdata,  // 输出的传输数据指针
        NPY_ARRAYMETHOD_FLAGS *flags)  // 数组方法标志指针
{
    if (PyDataType_NAMES(context->descriptors[0]) != NULL ||
            PyDataType_NAMES(context->descriptors[1]) != NULL) {
        // 如果数据类型有名称，则调用字段传输函数
        if (get_fields_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;  // 失败返回-1
        }
    }
    else if (PyDataType_SUBARRAY(context->descriptors[0]) != NULL ||
             PyDataType_SUBARRAY(context->descriptors[1]) != NULL) {
        // 如果是子数组数据类型，则调用子数组传输函数
        if (get_subarray_transfer_function(
                aligned, strides[0], strides[1],
                context->descriptors[0], context->descriptors[1],
                move_references, out_loop, out_transferdata,
                flags) == NPY_FAIL) {
            return -1;  // 失败返回-1
        }
    }
    else {
        /*
         * 这是两个字节的类似字符串的复制（如果需要则进行零填充）
         */
        // 如果既不是有名称的数据类型也不是子数组数据类型，则进行零填充复制
        if (PyArray_GetStridedZeroPadCopyFn(
                0, 0, strides[0], strides[1],
                context->descriptors[0]->elsize, context->descriptors[1]->elsize,
                out_loop, out_transferdata) == NPY_FAIL) {
            return -1;  // 失败返回-1
        }
        *flags = PyArrayMethod_MINIMAL_FLAGS;  // 设置最小的数组方法标志
    }
    return 0;  // 成功返回0
}


/*
 * 这个函数初始化了从 void 到 void 的类型转换。Void 类型包括结构化数据类型，
 * 这意味着它们可以从任何其他数据类型转换到和从任何其他数据类型转换出来，
 * 在这个意义上它们是特殊的（类似于 Object 类型）。
 */
static int
PyArray_InitializeVoidToVoidCast(void)
{
    PyArray_DTypeMeta *Void = &PyArray_VoidDType;  // 获取 Void 数据类型元数据
    PyArray_DTypeMeta *dtypes[2] = {Void, Void};  // 定义两个 Void 类型的数组
    PyType_Slot slots[] = {
            {NPY_METH_get_loop, &void_to_void_get_loop},  // 获取循环函数的插槽
            {NPY_METH_resolve_descriptors, &void_to_void_resolve_descriptors},  // 解析描述符的插槽
            {0, NULL}};  // 结束插槽列表的标志
    PyArrayMethod_Spec spec = {
            .name = "void_to_void_cast",  // 方法名称
            .nin = 1,  // 输入参数数量
            .nout = 1,  // 输出参数数量
            .casting = -1,  /* may not cast at all */  // 可能不会进行任何类型转换
            .flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_SUPPORTS_UNALIGNED,  // 方法支持的标志
            .dtypes = dtypes,  // 数据类型数组
            .slots = slots,  // 插槽数组
    };

    int res = PyArray_AddCastingImplementation_FromSpec(&spec, 1);  // 添加从规格中获取的转换实现
    return res;  // 返回结果
}


/*
 * 实现从对象到任何类型的转换。从对象转换可能需要检查所有数组元素（对于参数化数据类型），
 * 因此如果输出数据类型未提供，则解析器将拒绝所有参数化数据类型。
 */
static NPY_CASTING
/*
 * Resolve descriptors for object-to-any casting method.
 */
NPY_CASTING object_to_any_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        /*
         * This should not really be called, since object -> parametric casts
         * require inspecting the object array. Allow legacy ones, the path
         * here is that e.g. "M8" input is considered to be the DType class,
         * and by allowing it here, we go back to the "M8" instance.
         *
         * StringDType is excluded since using the parameters of that dtype
         * requires creating an instance explicitly
         */
        if (NPY_DT_is_parametric(dtypes[1]) && dtypes[1] != &PyArray_StringDType) {
            PyErr_Format(PyExc_TypeError,
                    "casting from object to the parametric DType %S requires "
                    "the specified output dtype instance. "
                    "This may be a NumPy issue, since the correct instance "
                    "should be discovered automatically, however.", dtypes[1]);
            return -1;
        }
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    return NPY_UNSAFE_CASTING;
}


/*
 * Returns the method for object-to-generic casting.
 */
static PyObject *
PyArray_GetObjectToGenericCastingImpl(void)
{
    Py_INCREF(npy_static_pydata.ObjectToGenericMethod);
    return npy_static_pydata.ObjectToGenericMethod;
}


/*
 * Resolve descriptors for any-to-object casting method.
 */
static NPY_CASTING
any_to_object_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
        if (loop_descrs[1] == NULL) {
            return -1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    return NPY_SAFE_CASTING;
}


/*
 * Returns the method for generic-to-object casting.
 */
static PyObject *
PyArray_GetGenericToObjectCastingImpl(void)
{
    Py_INCREF(npy_static_pydata.GenericToObjectMethod);
    return npy_static_pydata.GenericToObjectMethod;
}


/*
 * Placeholder for a function related to casts within the object dtype,
 * indicating it might remain unimplemented.
 */
static int
// 设置方法标志，要求使用 Python API 并且不会出现浮点错误
*flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_NO_FLOATINGPOINT_ERRORS;

// 如果需要移动引用，选择移动引用的循环方法和传输数据为空
if (move_references) {
    *out_loop = &_strided_to_strided_move_references;
    *out_transferdata = NULL;
} else {
    // 否则选择复制引用的循环方法和传输数据为空
    *out_loop = &_strided_to_strided_copy_references;
    *out_transferdata = NULL;
}

// 返回成功状态
return 0;
}

// 初始化对象到对象的类型转换方法
static int
PyArray_InitializeObjectToObjectCast(void)
{
    // 设置对象数据类型元信息
    PyArray_DTypeMeta *Object = &PyArray_ObjectDType;
    PyArray_DTypeMeta *dtypes[2] = {Object, Object};
    // 定义类型插槽，包括获取循环方法和结束符
    PyType_Slot slots[] = {
        {NPY_METH_get_loop, &object_to_object_get_loop},
        {0, NULL}};
    // 定义方法规范
    PyArrayMethod_Spec spec = {
        .name = "object_to_object_cast",
        .nin = 1,
        .nout = 1,
        .casting = NPY_NO_CASTING,
        .flags = NPY_METH_REQUIRES_PYAPI | NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = dtypes,
        .slots = slots,
    };

    // 添加从规范中获取的类型转换实现
    int res = PyArray_AddCastingImplementation_FromSpec(&spec, 1);
    // 返回添加结果
    return res;
}

// 初始化空和对象的全局变量
static int
initialize_void_and_object_globals(void) {
    // 创建一个新的数组方法对象
    PyArrayMethodObject *method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    // 如果内存分配失败，报错并返回
    if (method == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    // 设置方法的名称和标志，支持非对齐访问并需要 Python API
    method->name = "void_to_any_cast";
    method->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method->casting = -1;
    method->resolve_descriptors = &structured_to_nonstructured_resolve_descriptors;
    method->get_strided_loop = &structured_to_nonstructured_get_loop;
    method->nin = 1;
    method->nout = 1;
    // 将方法赋给全局变量
    npy_static_pydata.VoidToGenericMethod = (PyObject *)method;

    // 重复上述步骤，设置另一个类型转换方法
    method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    if (method == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    method->name = "any_to_void_cast";
    method->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method->casting = -1;
    method->resolve_descriptors = &nonstructured_to_structured_resolve_descriptors;
    method->get_strided_loop = &nonstructured_to_structured_get_loop;
    method->nin = 1;
    method->nout = 1;
    npy_static_pydata.GenericToVoidMethod = (PyObject *)method;

    // 重复上述步骤，设置另一个类型转换方法
    method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    if (method == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    method->nin = 1;
    method->nout = 1;
    method->name = "object_to_any_cast";
    method->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;
    method->casting = NPY_UNSAFE_CASTING;
    method->resolve_descriptors = &object_to_any_resolve_descriptors;
    method->get_strided_loop = &object_to_any_get_loop;
    // 将 PyObject 指针赋值给 npy_static_pydata.ObjectToGenericMethod
    npy_static_pydata.ObjectToGenericMethod = (PyObject *)method;

    // 使用 PyArrayMethod_Type 类型创建一个新的 PyArrayMethodObject 对象，并将其赋给 method 指针
    method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    // 检查内存分配是否成功，如果失败则设置内存错误并返回 -1
    if (method == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    // 设置 method 对象的属性
    method->nin = 1;  // 输入参数数量为 1
    method->nout = 1;  // 输出参数数量为 1
    method->name = "any_to_object_cast";  // 方法名为 "any_to_object_cast"
    method->flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI;  // 设置标志位
    method->casting = NPY_SAFE_CASTING;  // 设置类型转换规则为安全转换
    method->resolve_descriptors = &any_to_object_resolve_descriptors;  // 设置解析描述符的函数指针
    method->get_strided_loop = &any_to_object_get_loop;  // 设置获取循环函数的指针

    // 将 PyObject 指针赋值给 npy_static_pydata.GenericToObjectMethod
    npy_static_pydata.GenericToObjectMethod = (PyObject *)method;

    // 返回成功标志 0
    return 0;
}



NPY_NO_EXPORT int
PyArray_InitializeCasts()
{
    // 初始化数值类型到数值类型的转换
    if (PyArray_InitializeNumericCasts() < 0) {
        return -1;
    }
    // 初始化字符串到数值类型的转换
    if (PyArray_InitializeStringCasts() < 0) {
        return -1;
    }
    // 初始化空类型到空类型的转换
    if (PyArray_InitializeVoidToVoidCast() < 0) {
        return -1;
    }
    // 初始化对象类型到对象类型的转换
    if (PyArray_InitializeObjectToObjectCast() < 0) {
        return -1;
    }
    /* Datetime casts are defined in datetime.c */
    // 初始化日期时间类型的转换（在 datetime.c 文件中定义）
    if (PyArray_InitializeDatetimeCasts() < 0) {
        return -1;
    }

    // 初始化空类型和对象类型的全局变量
    if (initialize_void_and_object_globals() < 0) {
        return -1;
    }

    // 返回成功状态
    return 0;
}
```