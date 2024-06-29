# `.\numpy\numpy\_core\src\umath\ufunc_type_resolution.c`

```
/*
 * NOTE: The type resolution defined in this file is considered legacy.
 *
 * The new mechanism separates type resolution and promotion into two
 * distinct steps, as per NEP 43.
 * Further, the functions in this file rely on the operands rather than
 * only the DTypes/descriptors.  They are still called and at this point
 * vital (NumPy ~1.21), but should hopefully become largely irrelevant very
 * quickly.
 *
 * At that point, this file should be deletable in its entirety.
 *
 *
 * This file implements type resolution for NumPy element-wise ufuncs.
 * This mechanism is still backwards-compatible with the pre-existing
 * legacy mechanism, so performs much slower than is necessary.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// printif debug tracing
#ifndef NPY_UF_DBG_TRACING
    #define NPY_UF_DBG_TRACING 0
#endif

#include "npy_config.h"

#include "numpy/npy_common.h"
#include "npy_import.h"

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "ufunc_type_resolution.h"
#include "ufunc_object.h"
#include "common.h"
#include "convert_datatype.h"
#include "dtypemeta.h"

#include "mem_overlap.h"
#if defined(HAVE_CBLAS)
#include "cblasfuncs.h"
#endif

#include <stdbool.h>
#include <arrayobject.h>

/**
 * Converts the NumPy casting enum `casting` to a Python string object.
 * Returns a Python object representing the casting type.
 */
static PyObject *
npy_casting_to_py_object(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return PyUnicode_FromString("no");
        case NPY_EQUIV_CASTING:
            return PyUnicode_FromString("equiv");
        case NPY_SAFE_CASTING:
            return PyUnicode_FromString("safe");
        case NPY_SAME_KIND_CASTING:
            return PyUnicode_FromString("same_kind");
        case NPY_UNSAFE_CASTING:
            return PyUnicode_FromString("unsafe");
        default:
            return PyLong_FromLong(casting);
    }
}


/**
 * Helper function to raise a binary type resolution error for ufuncs.
 * This function raises an exception with detailed information on ufunc,
 * operands[0], and operands[1].
 * Returns -1 to indicate that an exception was raised.
 */
static int
raise_binary_type_reso_error(PyUFuncObject *ufunc, PyArrayObject **operands) {
    PyObject *exc_value;

    /* Produce an error object with ufunc, operand[0], and operand[1] descriptors */
    exc_value = Py_BuildValue(
        "O(OO)", ufunc,
        (PyObject *)PyArray_DESCR(operands[0]),
        (PyObject *)PyArray_DESCR(operands[1])
    );
    if (exc_value == NULL){
        return -1;
    }
    PyErr_SetObject(
            npy_static_pydata._UFuncBinaryResolutionError, exc_value);
    Py_DECREF(exc_value);

    return -1;
}

/**
 * Helper function to raise UFuncNoLoopError exception for ufuncs.
 * This function raises an exception with detailed information on ufunc and dtypes.
 * Returns -1 to indicate that an exception was raised.
 */
NPY_NO_EXPORT int
raise_no_loop_found_error(
        PyUFuncObject *ufunc, PyObject **dtypes)
{
    PyObject *dtypes_tup = PyArray_TupleFromItems(ufunc->nargs, dtypes, 1);
    if (dtypes_tup == NULL) {
        return -1;
    }
    /* 创建一个错误对象 */
    PyObject *exc_value = PyTuple_Pack(2, ufunc, dtypes_tup);
    // 使用 PyTuple_Pack 函数创建一个包含 ufunc 和 dtypes_tup 的元组作为错误对象
    Py_DECREF(dtypes_tup);  // 减少 dtypes_tup 的引用计数

    // 检查错误对象是否创建成功
    if (exc_value == NULL) {
        return -1;  // 如果创建失败，返回 -1 表示错误
    }

    // 设置异常对象为 _UFuncNoLoopError，并传入之前创建的错误对象
    PyErr_SetObject(npy_static_pydata._UFuncNoLoopError, exc_value);

    Py_DECREF(exc_value);  // 减少错误对象的引用计数

    return -1;  // 返回 -1 表示函数执行失败
/* 
 * 静态函数：raise_casting_error
 * 抛出类型转换错误异常，用于输入或输出类型转换失败时调用
 * 返回：-1 表示异常已经抛出
 */
static int
raise_casting_error(
        PyObject *exc_type,
        PyUFuncObject *ufunc,
        NPY_CASTING casting,
        PyArray_Descr *from,
        PyArray_Descr *to,
        npy_intp i)
{
    PyObject *exc_value;
    PyObject *casting_value;

    // 将 NumPy 的类型转换枚举值转换为 Python 对象
    casting_value = npy_casting_to_py_object(casting);
    if (casting_value == NULL) {
        return -1;
    }

    // 构建异常对象，包含 ufunc 对象、转换方式、源类型、目标类型和索引
    exc_value = Py_BuildValue(
        "ONOOi",
        ufunc,
        casting_value,
        (PyObject *)from,
        (PyObject *)to,
        i
    );
    if (exc_value == NULL){
        return -1;
    }
    // 设置异常类型和值
    PyErr_SetObject(exc_type, exc_value);
    Py_DECREF(exc_value);

    return -1;
}

/* 
 * 静态函数：raise_input_casting_error
 * 抛出输入类型转换错误异常，始终返回 -1 表示异常已抛出
 */
static int
raise_input_casting_error(
        PyUFuncObject *ufunc,
        NPY_CASTING casting,
        PyArray_Descr *from,
        PyArray_Descr *to,
        npy_intp i)
{
    // 调用 raise_casting_error 函数，抛出输入类型转换错误异常
    return raise_casting_error(npy_static_pydata._UFuncInputCastingError,
                               ufunc, casting, from, to, i);
}

/* 
 * 静态函数：raise_output_casting_error
 * 抛出输出类型转换错误异常，始终返回 -1 表示异常已抛出
 */
static int
raise_output_casting_error(
        PyUFuncObject *ufunc,
        NPY_CASTING casting,
        PyArray_Descr *from,
        PyArray_Descr *to,
        npy_intp i)
{
    // 调用 raise_casting_error 函数，抛出输出类型转换错误异常
    return raise_casting_error(npy_static_pydata._UFuncOutputCastingError,
                               ufunc, casting, from, to, i);
}

/* 
 * UFUNC_API
 * 
 * 函数：PyUFunc_ValidateCasting
 * 验证输入和输出操作数能否按指定的转换方式转换为指定的数据类型
 * 返回：成功返回 0，验证失败返回 -1 并抛出异常
 */
NPY_NO_EXPORT int
PyUFunc_ValidateCasting(PyUFuncObject *ufunc,
                            NPY_CASTING casting,
                            PyArrayObject **operands,
                            PyArray_Descr *const *dtypes)
{
    int i, nin = ufunc->nin, nop = nin + ufunc->nout;

    // 遍历所有操作数（输入和输出）
    for (i = 0; i < nop; ++i) {
        if (i < nin) {
            // 对于输入操作数，验证其能否按指定的转换方式转换为指定数据类型
            if (!PyArray_CanCastArrayTo(operands[i], dtypes[i], casting)) {
                // 如果无法转换，抛出输入类型转换错误异常
                return raise_input_casting_error(
                    ufunc, casting, PyArray_DESCR(operands[i]), dtypes[i], i);
            }
        } else if (operands[i] != NULL) {
            // 对于输出操作数，验证指定数据类型能否按指定的转换方式转换为其类型
            if (!PyArray_CanCastTypeTo(dtypes[i],
                                    PyArray_DESCR(operands[i]), casting)) {
                // 如果无法转换，抛出输出类型转换错误异常
                return raise_output_casting_error(
                    ufunc, casting, dtypes[i], PyArray_DESCR(operands[i]), i);
            }
        }
    }

    return 0;
}

/* 
 * 函数：PyUFunc_ValidateOutCasting
 * 验证输出操作数能否按指定的转换方式转换为指定的数据类型
 * 返回：成功返回 0，验证失败返回 -1 并抛出异常
 */
NPY_NO_EXPORT int
PyUFunc_ValidateOutCasting(PyUFuncObject *ufunc,
        NPY_CASTING casting, PyArrayObject **operands, PyArray_Descr **dtypes)
{
    int i, nin = ufunc->nin, nop = nin + ufunc->nout;
    
    // 遍历所有输出操作数
    for (i = 0; i < nop; ++i) {
        if (operands[i] != NULL) {
            // 验证输出操作数能否按指定的转换方式转换为指定数据类型
            if (!PyArray_CanCastTypeTo(dtypes[i],
                                    PyArray_DESCR(operands[i]), casting)) {
                // 如果无法转换，抛出输出类型转换错误异常
                return raise_output_casting_error(
                    ufunc, casting, dtypes[i], PyArray_DESCR(operands[i]), i);
            }
        }
    }
    
    return 0;
}
    # 遍历从 nin 到 nop 之间的索引值，其中 nin 和 nop 可能是变量或常量
    for (i = nin; i < nop; ++i) {
        # 如果操作数列表中的当前索引 i 处的操作数为空，则跳过当前循环，继续下一个操作数
        if (operands[i] == NULL) {
            continue;
        }
        # 检查是否可以将当前操作数的数据类型转换为指定的目标数据类型 dtypes[i]
        if (!PyArray_CanCastTypeTo(dtypes[i],
                PyArray_DESCR(operands[i]), casting)) {
            # 如果无法转换，则调用函数 raise_output_casting_error 抛出类型转换错误，并返回错误码
            return raise_output_casting_error(
                    ufunc, casting, dtypes[i], PyArray_DESCR(operands[i]), i);
        }
    }
    # 循环结束后，如果没有发生类型转换错误，则返回 0 表示成功
    return 0;
/*UFUNC_API
 *
 * This function applies the default type resolution rules
 * for the provided ufunc.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_DefaultTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int i, nop = ufunc->nin + ufunc->nout;
    int retval = 0, any_object = 0;
    NPY_CASTING input_casting;

    // 遍历操作数，检查是否有对象数组
    for (i = 0; i < nop; ++i) {
        if (operands[i] != NULL &&
                PyTypeNum_ISOBJECT(PyArray_DESCR(operands[i])->type_num)) {
            any_object = 1;
            break;
        }
    }

    /*
     * Decide the casting rules for inputs and outputs.  We want
     * NPY_SAFE_CASTING or stricter, so that the loop selection code
     * doesn't choose an integer loop for float inputs, or a float32
     * loop for float64 inputs.
     */
    // 根据传入的 casting 参数确定输入和输出的转换规则
    input_casting = (casting > NPY_SAFE_CASTING) ? NPY_SAFE_CASTING : casting;

    if (type_tup == NULL) {
        /* Find the best ufunc inner loop, and fill in the dtypes */
        // 如果 type_tup 为 NULL，则使用线性搜索来确定最佳的内部循环并填充 dtypes
        retval = linear_search_type_resolver(ufunc, operands,
                        input_casting, casting, any_object,
                        out_dtypes);
    } else {
        /* Find the specified ufunc inner loop, and fill in the dtypes */
        // 如果指定了 type_tup，则使用类型元组解析器来确定指定的内部循环并填充 dtypes
        retval = type_tuple_type_resolver(ufunc, type_tup,
                        operands, input_casting, casting, any_object, out_dtypes);
    }

    return retval;
}

/*
 * This function applies special type resolution rules for the case
 * where all the functions have the pattern XX->bool, using
 * PyArray_ResultType instead of a linear search to get the best
 * loop.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_SimpleBinaryComparisonTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int i, type_num1, type_num2;
    const char *ufunc_name = ufunc_get_name_cstr(ufunc);

    // 如果输入不符合二进制比较类型解析的条件，抛出运行时错误
    if (ufunc->nin != 2 || ufunc->nout != 1) {
        PyErr_Format(PyExc_RuntimeError, "ufunc %s is configured "
                "to use binary comparison type resolution but has "
                "the wrong number of inputs or outputs",
                ufunc_name);
        return -1;
    }

    /*
     * Use the default type resolution if there's a custom data type
     * or object arrays.
     */
    // 获取操作数的类型编号
    type_num1 = PyArray_DESCR(operands[0])->type_num;
    type_num2 = PyArray_DESCR(operands[1])->type_num;
    if (type_num1 >= NPY_NTYPES_LEGACY || type_num2 >= NPY_NTYPES_LEGACY ||
            type_num1 == NPY_OBJECT || type_num2 == NPY_OBJECT) {
        // 如果操作数的类型编号超过了旧版 NumPy 类型的数量，或者其中有任何一个是对象类型，使用默认的类型解析器处理
        return PyUFunc_DefaultTypeResolver(ufunc, casting, operands,
                type_tup, out_dtypes);
    }

    if (type_tup == NULL) {
        // 如果类型元组为空
        if (PyArray_ISDATETIME(operands[0])
                && PyArray_ISDATETIME(operands[1])
                && type_num1 != type_num2) {
            /*
             * 拒绝混合的日期时间和时间增量，这总是会失败，因为类型转换会失败（除非使用 `casting="unsafe"`）。
             * 这是必需的，以确保 `==` 和 `!=` 能够正确检测并返回结果数组的 False/True。
             */
            return raise_binary_type_reso_error(ufunc, operands);
        }
        /*
         * 这个检查是为了避免可能出现的 FutureWarning，ResultType 在数值->字符串提升时会给出警告。
         * （我们从未支持灵活的 dtype。）
         */
        else if (!PyArray_ISFLEXIBLE(operands[0]) &&
                !PyArray_ISFLEXIBLE(operands[1])) {
            // 如果操作数都不是灵活的 dtype
            out_dtypes[0] = PyArray_ResultType(2, operands, 0, NULL);
            if (out_dtypes[0] == NULL) {
                // 如果获取结果类型失败，返回 -1
                return -1;
            }
            if (PyArray_ISINTEGER(operands[0])
                    && PyArray_ISINTEGER(operands[1])
                    && !PyDataType_ISINTEGER(out_dtypes[0])) {
                /*
                 * NumPy 的提升允许无符号整数和有符号整数相加得到浮点数，避免这种情况
                 * （输入必须是有符号和无符号混合的情况）
                 */
                if (PyArray_ISSIGNED(operands[0])) {
                    // 如果第一个操作数是有符号整数
                    Py_SETREF(out_dtypes[0], PyArray_DescrFromType(NPY_LONGLONG));
                    out_dtypes[1] = PyArray_DescrFromType(NPY_ULONGLONG);
                    Py_INCREF(out_dtypes[1]);
                }
                else {
                    // 如果第二个操作数是有符号整数
                    Py_SETREF(out_dtypes[0], PyArray_DescrFromType(NPY_ULONGLONG));
                    out_dtypes[1] = PyArray_DescrFromType(NPY_LONGLONG);
                    Py_INCREF(out_dtypes[1]);
                }
            }
            else {
                // 否则，输出类型相同
                out_dtypes[1] = out_dtypes[0];
                Py_INCREF(out_dtypes[1]);
            }
        }
        else {
            // 否则，不做任何处理，继续使用操作数的描述符
            /* Not doing anything will lead to a loop no found error. */
            out_dtypes[0] = PyArray_DESCR(operands[0]);
            Py_INCREF(out_dtypes[0]);
            out_dtypes[1] = PyArray_DESCR(operands[1]);
            Py_INCREF(out_dtypes[1]);
        }
    }
    else {
        // 如果类型元组不为空，通常是失败的情况，让默认版本处理
        /* Usually a failure, but let the default version handle it */
        return PyUFunc_DefaultTypeResolver(ufunc, casting,
                operands, type_tup, out_dtypes);
    }

    /* 输出类型始终是布尔类型（内置类型不会失败） */
    out_dtypes[2] = PyArray_DescrFromType(NPY_BOOL);
    /* 检查根据转换规则进行验证 */
    if (PyUFunc_ValidateCasting(ufunc, casting, operands, out_dtypes) < 0) {
        /* 如果验证失败，释放输出数据类型对象的引用并置空 */
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        // 返回错误标志
        return -1;
    }

    // 返回成功标志
    return 0;
/*
 * PyUFunc_NegativeTypeResolver函数用于解析一元负号操作的数据类型。
 * 它调用PyUFunc_SimpleUniformOperationTypeResolver函数来进行统一操作类型的解析。
 * 参数解释：
 * ufunc: 指向PyUFuncObject结构的指针，表示当前正在处理的ufunc对象。
 * casting: 表示操作的转换级别。
 * operands: 指向PyArrayObject指针数组的指针，表示操作数数组。
 * type_tup: 表示一个Python元组对象，包含用户指定的类型信息。
 * out_dtypes: 指向PyArray_Descr指针数组的指针，用于存储输出的数据类型。
 */
NPY_NO_EXPORT int
PyUFunc_NegativeTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArrayObject **operands,
                             PyObject *type_tup,
                             PyArray_Descr **out_dtypes)
{
    int ret;
    // 调用PyUFunc_SimpleUniformOperationTypeResolver进行类型解析
    ret = PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting, operands,
                                                   type_tup, out_dtypes);
    if (ret < 0) {
        return ret;
    }

    /* The type resolver would have upcast already */
    // 如果输出的第一个数据类型是布尔类型，抛出类型错误异常
    if (out_dtypes[0]->type_num == NPY_BOOL) {
        PyErr_Format(PyExc_TypeError,
            "The numpy boolean negative, the `-` operator, is not supported, "
            "use the `~` operator or the logical_not function instead.");
        return -1;
    }

    return ret;
}

/*
 * PyUFunc_OnesLikeTypeResolver函数提供了对ones_like函数的类型解析。
 * 当ones_like函数作为ufunc使用时，它总是强制使用UNSAFE转换级别进行类型解析。
 * 参数解释：
 * ufunc: 指向PyUFuncObject结构的指针，表示当前正在处理的ufunc对象。
 * casting: 表示操作的转换级别（此处未使用）。
 * operands: 指向PyArrayObject指针数组的指针，表示操作数数组。
 * type_tup: 表示一个Python元组对象，包含用户指定的类型信息。
 * out_dtypes: 指向PyArray_Descr指针数组的指针，用于存储输出的数据类型。
 */
NPY_NO_EXPORT int
PyUFunc_OnesLikeTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING NPY_UNUSED(casting),
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    // 调用PyUFunc_SimpleUniformOperationTypeResolver进行类型解析，强制使用UNSAFE转换级别
    return PyUFunc_SimpleUniformOperationTypeResolver(ufunc,
                        NPY_UNSAFE_CASTING,
                        operands, type_tup, out_dtypes);
}

/*
 * PyUFunc_SimpleUniformOperationTypeResolver函数用于处理所有输入类型相同的ufunc操作。
 * 它利用PyArray_ResultType而不是线性搜索来获取最佳的循环。
 * 返回0表示成功，返回-1表示失败。
 * 参数解释：
 * ufunc: 指向PyUFuncObject结构的指针，表示当前正在处理的ufunc对象。
 * casting: 表示操作的转换级别。
 * operands: 指向PyArrayObject指针数组的指针，表示操作数数组。
 * type_tup: 表示一个Python元组对象，包含用户指定的类型信息。
 * out_dtypes: 指向PyArray_Descr指针数组的指针，用于存储输出的数据类型。
 */
NPY_NO_EXPORT int
PyUFunc_SimpleUniformOperationTypeResolver(
        PyUFuncObject *ufunc,
        NPY_CASTING casting,
        PyArrayObject **operands,
        PyObject *type_tup,
        PyArray_Descr **out_dtypes)
{
    // 获取当前ufunc的名称
    const char *ufunc_name = ufunc_get_name_cstr(ufunc);

    // 检查ufunc的输入数是否小于1，如果是则抛出运行时错误
    if (ufunc->nin < 1) {
        PyErr_Format(PyExc_RuntimeError, "ufunc %s is configured "
                "to use uniform operation type resolution but has "
                "no inputs",
                ufunc_name);
        return -1;
    }
    int nop = ufunc->nin + ufunc->nout;

    /*
     * 判断是否存在自定义数据类型或对象数组
     */
    bool has_custom_or_object = false;
    for (int iop = 0; iop < ufunc->nin; iop++) {
        int type_num = PyArray_DESCR(operands[iop])->type_num;
        // 如果类型号大于等于NPY_NTYPES_LEGACY或者等于NPY_OBJECT，表示存在自定义数据类型或对象数组
        if (type_num >= NPY_NTYPES_LEGACY || type_num == NPY_OBJECT) {
            has_custom_or_object = true;
            break;
        }
    }

    // 如果存在自定义数据类型或对象数组，则调用默认的类型解析器
    if (has_custom_or_object) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting, operands,
                type_tup, out_dtypes);
    }
    # 如果 type_tup 是 NULL，即类型元组为空
    if (type_tup == NULL) {
        # PyArray_ResultType 在 nin == 1 时忘记强制字节顺序
        if (ufunc->nin == 1){
            # 获取第一个操作数的描述符，并确保规范化其字节顺序
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
        }
        else {
            # 初始化循环变量 iop
            int iop;
            npy_bool has_flexible = 0;
            npy_bool has_object = 0;
            # 检查每个操作数的类型
            for (iop = 0; iop < ufunc->nin; iop++) {
                # 如果操作数是对象类型数组
                if (PyArray_ISOBJECT(operands[iop])) {
                    has_object = 1;
                }
                # 如果操作数是灵活类型数组
                if (PyArray_ISFLEXIBLE(operands[iop])) {
                    has_flexible = 1;
                }
            }
            # 如果有灵活类型但没有对象类型
            if (NPY_UNLIKELY(has_flexible && !has_object)) {
                /*
                 * NumPy 1.20 弃用提醒，2020-12。
                 * 此检查是为了避免 ResultType 在数字->字符串提升时产生的 FutureWarning。
                 * （我们从不支持这里的灵活 dtype。）
                 */
                # 将每个操作数的描述符复制给输出类型数组
                for (iop = 0; iop < ufunc->nin; iop++) {
                    out_dtypes[iop] = PyArray_DESCR(operands[iop]);
                    Py_INCREF(out_dtypes[iop]);
                }
                # 抛出找不到合适循环的错误
                raise_no_loop_found_error(ufunc, (PyObject **)out_dtypes);
                # 释放之前增加的引用计数
                for (iop = 0; iop < ufunc->nin; iop++) {
                    Py_DECREF(out_dtypes[iop]);
                    out_dtypes[iop] = NULL;
                }
                # 返回错误码
                return -1;
            }
            # 使用 PyArray_ResultType 计算输出的数据类型
            out_dtypes[0] = PyArray_ResultType(ufunc->nin, operands, 0, NULL);
        }
        # 如果输出类型为空，返回错误码
        if (out_dtypes[0] == NULL) {
            return -1;
        }
    }
    else {
        /*
         * 这是一个快速路径，因为所有描述符都将是相同的，主要是当只传递了单个描述符时
         * （这将设置元组中的输出描述符），就没有必要检查所有循环。
         * 注意，这也允许（None, None, float64）解析为（float64, float64, float64），
         * 即使输入不匹配，修复签名的输出部分可以修复所有这些情况。
         * 这是支持 `nextafter(1., inf, dtype=float32)` 所必需的，这里很“清楚”我们想要将 1. 和 inf 转换为 float32。
         */
        PyArray_Descr *descr = NULL;
        if (PyTuple_CheckExact(type_tup) &&
                PyTuple_GET_SIZE(type_tup) == nop) {
            for (int i = 0; i < nop; i++) {
                PyObject *item = PyTuple_GET_ITEM(type_tup, i);
                if (item == Py_None) {
                    if (i < ufunc->nin) {
                        continue;
                    }
                    /* 所有输出必须被设置（这可能会放宽） */
                    descr = NULL;
                    break;
                }
                if (!PyArray_DescrCheck(item)) {
                    /* 推迟到默认解析器（将在那里引发错误） */
                    descr = NULL;
                    break;
                }
                if (descr != NULL && descr != (PyArray_Descr *)item) {
                    /* 描述符不匹配：尝试使用默认值（可能是错误） */
                    descr = NULL;
                    break;
                }
                descr = (PyArray_Descr *)item;
            }
        }
        if (descr == NULL) {
            /* 在所有坏/不太可能的情况下，使用默认类型解析器： */
            return PyUFunc_DefaultTypeResolver(ufunc, casting,
                    operands, type_tup, out_dtypes);
        }
        else if (descr->type_num == PyArray_DESCR(operands[0])->type_num) {
            /* 如果匹配，则优先使用输入描述符（保留元数据） */
            descr = PyArray_DESCR(operands[0]);
        }
        out_dtypes[0] = NPY_DT_CALL_ensure_canonical(descr);
    }

    /* 所有类型相同 - 将第一个复制到其余部分 */
    for (int iop = 1; iop < nop; iop++) {
        out_dtypes[iop] = out_dtypes[0];
        Py_INCREF(out_dtypes[iop]);
    }

    /* 根据类型转换规则进行检查 */
    if (PyUFunc_ValidateCasting(ufunc, casting, operands, out_dtypes) < 0) {
        for (int iop = 0; iop < nop; iop++) {
            Py_DECREF(out_dtypes[iop]);
            out_dtypes[iop] = NULL;
        }
        return -1;
    }

    return 0;
/*
 * This function applies special type resolution rules for the absolute
 * ufunc. This ufunc converts complex -> float, so isn't covered
 * by the simple unary type resolution.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_AbsoluteTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    /* Use the default for complex types, to find the loop producing float */
    // 如果操作数的数据类型是复数类型，使用默认的类型解析器来确定生成浮点数的循环
    if (PyTypeNum_ISCOMPLEX(PyArray_DESCR(operands[0])->type_num)) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting, operands,
                    type_tup, out_dtypes);
    }
    else {
        // 对于非复数类型，使用简单统一操作类型解析器
        return PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                    operands, type_tup, out_dtypes);
    }
}

/*
 * This function applies special type resolution rules for the isnat
 * ufunc. This ufunc converts datetime/timedelta -> bool, and is not covered
 * by the simple unary type resolution.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyUFunc_IsNaTTypeResolver(PyUFuncObject *ufunc,
                          NPY_CASTING casting,
                          PyArrayObject **operands,
                          PyObject *type_tup,
                          PyArray_Descr **out_dtypes)
{
    // 如果操作数的数据类型不是日期时间类型，设置类型错误并返回 -1
    if (!PyTypeNum_ISDATETIME(PyArray_DESCR(operands[0])->type_num)) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc 'isnat' is only defined for np.datetime64 and np.timedelta64.");
        return -1;
    }

    // 为输出的数据类型设置日期时间的规范化描述符和布尔型描述符
    out_dtypes[0] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(operands[0]));
    out_dtypes[1] = PyArray_DescrFromType(NPY_BOOL);

    return 0;
}


NPY_NO_EXPORT int
PyUFunc_IsFiniteTypeResolver(PyUFuncObject *ufunc,
                          NPY_CASTING casting,
                          PyArrayObject **operands,
                          PyObject *type_tup,
                          PyArray_Descr **out_dtypes)
{
    // 如果操作数的数据类型不是日期时间类型，使用默认的类型解析器
    if (!PyTypeNum_ISDATETIME(PyArray_DESCR(operands[0])->type_num)) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting, operands,
                                    type_tup, out_dtypes);
    }

    // 为输出的数据类型设置日期时间的规范化描述符和布尔型描述符
    out_dtypes[0] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(operands[0]));
    out_dtypes[1] = PyArray_DescrFromType(NPY_BOOL);

    return 0;
}


/*
 * Creates a new NPY_TIMEDELTA dtype, copying the datetime metadata
 * from the given dtype.
 *
 * NOTE: This function is copied from datetime.c in multiarray,
 *       because umath and multiarray are not linked together.
 */
static PyArray_Descr *
timedelta_dtype_with_copied_meta(PyArray_Descr *dtype)
{
    PyArray_Descr *ret;
    PyArray_DatetimeMetaData *dst, *src;
    PyArray_DatetimeDTypeMetaData *dst_dtmd, *src_dtmd;

    // 创建一个新的时间增量类型描述符，并从给定的类型描述符复制日期时间元数据
    ret = PyArray_DescrNewFromType(NPY_TIMEDELTA);
    if (ret == NULL) {
        return NULL;
    }
    // 将输入数据类型的日期时间元数据指针转换为旧式描述符的指针，然后获取其C风格的元数据结构体
    src_dtmd = (PyArray_DatetimeDTypeMetaData *)((_PyArray_LegacyDescr *)dtype)->c_metadata;
    // 将返回数据类型的日期时间元数据指针转换为旧式描述符的指针，然后获取其C风格的元数据结构体
    dst_dtmd = (PyArray_DatetimeDTypeMetaData *)((_PyArray_LegacyDescr *)ret)->c_metadata;
    // 获取源数据类型的日期时间元数据的元数据结构体指针
    src = &(src_dtmd->meta);
    // 获取目标数据类型的日期时间元数据的元数据结构体指针
    dst = &(dst_dtmd->meta);

    // 将源元数据结构体的内容复制到目标元数据结构体
    *dst = *src;

    // 返回处理后的数据类型描述符
    return ret;
/*
 * This function applies the type resolution rules for addition.
 * In particular, there are special cases for string and unicode types,
 * as well as specific cases involving datetime types:
 *    m8[<A>] + m8[<B>] => m8[gcd(<A>,<B>)] + m8[gcd(<A>,<B>)]
 *    m8[<A>] + int     => m8[<A>] + m8[<A>]
 *    int     + m8[<A>] => m8[<A>] + m8[<A>]
 *    M8[<A>] + int     => M8[<A>] + m8[<A>]
 *    int     + M8[<A>] => m8[<A>] + M8[<A>]
 *    M8[<A>] + m8[<B>] => M8[gcd(<A>,<B>)] + m8[gcd(<A>,<B>)]
 *    m8[<A>] + M8[<B>] => m8[gcd(<A>,<B>)] + M8[gcd(<A>,<B>)]
 * TODO: Non-linear time unit cases require highly specialized loops
 *    M8[<A>] + m8[Y|M|B]
 *    m8[Y|M|B] + M8[<A>]
 */
NPY_NO_EXPORT int
PyUFunc_AdditionTypeResolver(PyUFuncObject *ufunc,
                             NPY_CASTING casting,
                             PyArrayObject **operands,
                             PyObject *type_tup,
                             PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    // Get the type number of the first and second operands
    type_num1 = PyArray_DESCR(operands[0])->type_num;
    type_num2 = PyArray_DESCR(operands[1])->type_num;

    /* Use the default resolver when neither operand involves datetime,
     * timedelta, string, or unicode types */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)
        && !(PyTypeNum_ISSTRING(type_num1) && PyTypeNum_ISSTRING(type_num2))) {
        return PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                    operands, type_tup, out_dtypes);
    }

    // Handle cases where both operands are of string or unicode types
    if ((type_num1 == NPY_STRING && type_num2 == NPY_STRING)
            || (type_num1 == NPY_UNICODE && type_num2 == NPY_UNICODE)) {
        // Set the output dtypes to match the input dtypes
        // This is required to ensure compatibility with the loop implementation
        out_dtypes[0] = PyArray_DescrFromType(type_num1);
        out_dtypes[1] = out_dtypes[0];
        Py_INCREF(out_dtypes[1]);  // Increment reference count for the dtype
        out_dtypes[2] = out_dtypes[0];
        Py_INCREF(out_dtypes[2]);  // Increment reference count for the dtype
        // The function does not return yet; it proceeds with more specific cases
        // involving datetime and timedelta types
    } else if (type_num1 == NPY_TIMEDELTA) {
        /* 若 type_num1 是 NPY_TIMEDELTA 类型 */

        /* m8[<A>] + m8[<B>] => m8[gcd(<A>,<B>)] + m8[gcd(<A>,<B>)] */
        if (type_num2 == NPY_TIMEDELTA) {
            // 若 type_num2 也是 NPY_TIMEDELTA 类型，则需要计算类型的提升
            out_dtypes[0] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            if (out_dtypes[0] == NULL) {
                return -1;  // 处理错误情况
            }
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);
        }
        /* m8[<A>] + M8[<B>] => m8[gcd(<A>,<B>)] + M8[gcd(<A>,<B>)] */
        else if (type_num2 == NPY_DATETIME) {
            // 若 type_num2 是 NPY_DATETIME 类型，则需要特殊处理
            out_dtypes[1] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            if (out_dtypes[1] == NULL) {
                return -1;  // 处理错误情况
            }
            /* 创建一个新的 NPY_TIMEDELTA，并复制 datetime 的元数据 */
            out_dtypes[0] = timedelta_dtype_with_copied_meta(out_dtypes[1]);
            if (out_dtypes[0] == NULL) {
                Py_DECREF(out_dtypes[1]);
                out_dtypes[1] = NULL;
                return -1;  // 处理错误情况
            }
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[2]);
        }
        /* m8[<A>] + int => m8[<A>] + m8[<A>] */
        else if (PyTypeNum_ISINTEGER(type_num2) ||
                                    PyTypeNum_ISBOOL(type_num2)) {
            // 若 type_num2 是整数或布尔型，则需要确保类型为 NPY_TIMEDELTA
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
            if (out_dtypes[0] == NULL) {
                return -1;  // 处理错误情况
            }
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_TIMEDELTA;  // 强制将 type_num2 设为 NPY_TIMEDELTA
        }
        else {
            return raise_binary_type_reso_error(ufunc, operands);
            // 若不满足上述情况，则抛出类型解析错误
        }
    }
    else if (type_num1 == NPY_DATETIME) {
        /* 如果 type_num1 是 NPY_DATETIME 类型 */

        /* M8[<A>] + m8[<B>] => M8[gcd(<A>,<B>)] + m8[gcd(<A>,<B>)] */
        /* 如果 type_num2 也是 NPY_TIMEDELTA 类型，则处理如下 */
        if (type_num2 == NPY_TIMEDELTA) {
            /* 推断操作数的输出类型 */
            out_dtypes[0] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* 创建一个新的 NPY_TIMEDELTA 类型，并复制 datetime 的元数据 */
            out_dtypes[1] = timedelta_dtype_with_copied_meta(out_dtypes[0]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);
        }
        /* 如果 type_num2 是整数类型或布尔类型 */
        else if (PyTypeNum_ISINTEGER(type_num2) ||
                    PyTypeNum_ISBOOL(type_num2)) {
            /* 确保第一个操作数的规范类型 */
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* 创建一个新的 NPY_TIMEDELTA 类型，并复制 type1 的元数据 */
            out_dtypes[1] = timedelta_dtype_with_copied_meta(
                                            PyArray_DESCR(operands[0]));
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            /* 将 type_num2 设为 NPY_TIMEDELTA */
            type_num2 = NPY_TIMEDELTA;
        }
        else {
            /* 如果 type_num2 不是预期的类型，则触发二进制类型解析错误 */
            return raise_binary_type_reso_error(ufunc, operands);
        }
    }
    else if (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) {
        /* 如果 type_num1 是整数或布尔类型 */

        /* int + m8[<A>] => m8[<A>] + m8[<A>] */
        /* 如果 type_num2 是 NPY_TIMEDELTA */
        if (type_num2 == NPY_TIMEDELTA) {
            /* 获取操作数 operands[1] 的描述符，并确保它是规范的数据类型 */
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[1]));
            /* 如果无法确保规范数据类型，则返回错误 */
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* 复制数据类型到其他两个输出数据类型 */
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            /* 将 type_num1 设置为 NPY_TIMEDELTA */
            type_num1 = NPY_TIMEDELTA;
        }
        /* 如果 type_num2 是 NPY_DATETIME */
        else if (type_num2 == NPY_DATETIME) {
            /* 创建一个带有复制元数据的新 NPY_TIMEDELTA 数据类型 */
            out_dtypes[0] = timedelta_dtype_with_copied_meta(
                                            PyArray_DESCR(operands[1]));
            /* 如果创建失败，则返回错误 */
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            /* 确保 operands[1] 的描述符是规范的数据类型 */
            out_dtypes[1] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[1]));
            /* 如果无法确保规范数据类型，则释放已分配的内存并返回错误 */
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            /* 复制数据类型到第三个输出数据类型 */
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[2]);

            /* 将 type_num1 设置为 NPY_TIMEDELTA */
            type_num1 = NPY_TIMEDELTA;
        }
        /* 如果 type_num2 不是 NPY_TIMEDELTA 或 NPY_DATETIME，则返回二元类型解析错误 */
        else {
            return raise_binary_type_reso_error(ufunc, operands);
        }
    }
    /* 如果 type_num1 不是整数或布尔类型，则返回二元类型解析错误 */
    else {
        return raise_binary_type_reso_error(ufunc, operands);
    }

    /* 检查是否符合类型转换规则 */
    /* 如果不符合类型转换规则，则释放已分配的内存并返回错误 */
    if (PyUFunc_ValidateCasting(ufunc, casting, operands, out_dtypes) < 0) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    /* 操作成功完成，返回成功状态 */
    return 0;
/*
 * This function applies the type resolution rules for subtraction.
 * In particular, there are a number of special cases with datetime:
 *    m8[<A>] - m8[<B>] => m8[gcd(<A>,<B>)] - m8[gcd(<A>,<B>)]
 *    m8[<A>] - int     => m8[<A>] - m8[<A>]
 *    int     - m8[<A>] => m8[<A>] - m8[<A>]
 *    M8[<A>] - int     => M8[<A>] - m8[<A>]
 *    M8[<A>] - m8[<B>] => M8[gcd(<A>,<B>)] - m8[gcd(<A>,<B>)]
 * TODO: Non-linear time unit cases require highly special-cased loops
 *    M8[<A>] - m8[Y|M|B]
 */
NPY_NO_EXPORT int
PyUFunc_SubtractionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    type_num1 = PyArray_DESCR(operands[0])->type_num;  // 获取第一个操作数的数据类型编号
    type_num2 = PyArray_DESCR(operands[1])->type_num;  // 获取第二个操作数的数据类型编号

    /* Use the default when datetime and timedelta are not involved */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        int ret;
        // 如果操作数不涉及 datetime 或 timedelta，则使用默认的类型解析器
        ret = PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                                                operands, type_tup, out_dtypes);
        if (ret < 0) {
            return ret;
        }

        /* The type resolver would have upcast already */
        // 类型解析器应该已经完成了类型提升
        if (out_dtypes[0]->type_num == NPY_BOOL) {
            PyErr_Format(PyExc_TypeError,
                "numpy boolean subtract, the `-` operator, is not supported, "
                "use the bitwise_xor, the `^` operator, or the logical_xor "
                "function instead.");
            return -1;
        }
        return ret;
    }

    if (type_num1 == NPY_TIMEDELTA) {
        /* m8[<A>] - m8[<B>] => m8[gcd(<A>,<B>)] - m8[gcd(<A>,<B>)] */
        // 如果第一个操作数是 timedelta
        if (type_num2 == NPY_TIMEDELTA) {
            // 并且第二个操作数也是 timedelta，则结果类型为两者类型的最小公倍数
            out_dtypes[0] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            // 复制给其他输出类型
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);
        }
        /* m8[<A>] - int => m8[<A>] - m8[<A>] */
        else if (PyTypeNum_ISINTEGER(type_num2) ||
                                        PyTypeNum_ISBOOL(type_num2)) {
            // 如果第二个操作数是整数或布尔类型，则结果类型保持为 timedelta
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            // 复制给其他输出类型
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_TIMEDELTA;  // 将第二个操作数类型设置为 timedelta
        }
        else {
            return raise_binary_type_reso_error(ufunc, operands);  // 抛出二进制类型解析错误
        }
    }
    else if (type_num1 == NPY_DATETIME) {
        /* 如果第一个操作数是日期时间类型 M8[<A>] */
        
        /* M8[<A>] - m8[<B>] => M8[gcd(<A>,<B>)] - m8[gcd(<A>,<B>)] */
        if (type_num2 == NPY_TIMEDELTA) {
            /* 如果第二个操作数是时间增量类型 */
            
            // 推断出输出类型，基于操作数的类型
            out_dtypes[0] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            if (out_dtypes[0] == NULL) {
                return -1;  // 返回错误代码
            }
            
            /* 创建一个新的 NPY_TIMEDELTA 类型，并复制日期时间的元数据 */
            out_dtypes[1] = timedelta_dtype_with_copied_meta(out_dtypes[0]);
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;  // 返回错误代码
            }
            
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);  // 增加引用计数，防止释放

        }
        /* M8[<A>] - int => M8[<A>] - m8[<A>] */
        else if (PyTypeNum_ISINTEGER(type_num2) ||
                    PyTypeNum_ISBOOL(type_num2)) {
            /* 如果第二个操作数是整数或布尔类型 */

            // 确保规范化操作数1的描述符
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
            if (out_dtypes[0] == NULL) {
                return -1;  // 返回错误代码
            }
            
            /* 创建一个新的 NPY_TIMEDELTA 类型，并复制类型1的元数据 */
            out_dtypes[1] = timedelta_dtype_with_copied_meta(
                                            PyArray_DESCR(operands[0]));
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;  // 返回错误代码
            }
            
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);  // 增加引用计数，防止释放

            type_num2 = NPY_TIMEDELTA;  // 设置第二个操作数为时间增量类型
        }
        /* M8[<A>] - M8[<B>] => M8[gcd(<A>,<B>)] - M8[gcd(<A>,<B>)] */
        else if (type_num2 == NPY_DATETIME) {
            /* 如果第二个操作数也是日期时间类型 M8[<B>] */

            // 推断出输出类型，基于操作数的类型
            out_dtypes[0] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            if (out_dtypes[0] == NULL) {
                return -1;  // 返回错误代码
            }
            
            /* 创建一个新的 NPY_TIMEDELTA 类型，并复制类型1的元数据 */
            out_dtypes[2] = timedelta_dtype_with_copied_meta(out_dtypes[0]);
            if (out_dtypes[2] == NULL) {
                Py_DECREF(out_dtypes[0]);
                return -1;  // 返回错误代码
            }
            
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);  // 增加引用计数，防止释放

        }
        else {
            return raise_binary_type_reso_error(ufunc, operands);
            // 如果类型不匹配，则触发二进制类型解析错误
        }
    }
    else if (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) {
        # 如果 type_num1 是整数或布尔类型
        /* int - m8[<A>] => m8[<A>] - m8[<A>] */
        # 执行 int - m8[<A>] => m8[<A>] - m8[<A>] 的操作，这是一种特定的数学操作符重载形式
        if (type_num2 == NPY_TIMEDELTA) {
            # 如果 type_num2 是 NPY_TIMEDELTA 类型
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[1]));
            # 确保操作数 operands[1] 的描述符是规范化的，并赋给输出类型数组的第一个元素
            if (out_dtypes[0] == NULL) {
                # 如果未能获取描述符则返回错误
                return -1;
            }
            out_dtypes[1] = out_dtypes[0];
            # 输出类型数组的第二个元素等于第一个元素
            Py_INCREF(out_dtypes[1]);
            # 增加输出类型数组的第二个元素的引用计数
            out_dtypes[2] = out_dtypes[0];
            # 输出类型数组的第三个元素等于第一个元素
            Py_INCREF(out_dtypes[2]);

            type_num1 = NPY_TIMEDELTA;
            # 将 type_num1 设为 NPY_TIMEDELTA 类型
        }
        else {
            # 否则，如果 type_num2 不是 NPY_TIMEDELTA 类型
            return raise_binary_type_reso_error(ufunc, operands);
            # 调用函数 raise_binary_type_reso_error 处理二进制操作类型冲突的错误，并返回错误代码
        }
    }
    else {
        # 如果 type_num1 不是整数或布尔类型
        return raise_binary_type_reso_error(ufunc, operands);
        # 同样调用函数 raise_binary_type_reso_error 处理二进制操作类型冲突的错误，并返回错误代码
    }

    /* Check against the casting rules */
    # 检查是否符合类型转换规则
    if (PyUFunc_ValidateCasting(ufunc, casting, operands, out_dtypes) < 0) {
        # 如果类型转换验证失败
        for (i = 0; i < 3; ++i) {
            # 遍历输出类型数组的前三个元素
            Py_DECREF(out_dtypes[i]);
            # 逐个减少输出类型数组元素的引用计数
            out_dtypes[i] = NULL;
            # 将输出类型数组元素置为 NULL
        }
        return -1;
        # 返回错误代码
    }

    return 0;
    # 执行成功，返回 0
/*
 * This function applies the type resolution rules for multiplication.
 * In particular, there are a number of special cases with datetime:
 *    int## * m8[<A>] => int64 * m8[<A>]
 *    m8[<A>] * int## => m8[<A>] * int64
 *    float## * m8[<A>] => float64 * m8[<A>]
 *    m8[<A>] * float## => m8[<A>] * float64
 */
NPY_NO_EXPORT int
PyUFunc_MultiplicationTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    // 获取第一个操作数的数据类型编号
    type_num1 = PyArray_DESCR(operands[0])->type_num;
    // 获取第二个操作数的数据类型编号
    type_num2 = PyArray_DESCR(operands[1])->type_num;

    /* 当涉及到 datetime 和 timedelta 时使用默认规则 */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)
        && !((PyTypeNum_ISSTRING(type_num1) && PyTypeNum_ISINTEGER(type_num2))
             || (PyTypeNum_ISINTEGER(type_num1) && PyTypeNum_ISSTRING(type_num2)))) {
        // 调用默认的类型解析器，用于非特殊情况
        return PyUFunc_SimpleUniformOperationTypeResolver(ufunc, casting,
                    operands, type_tup, out_dtypes);
    }

    // 当其中一个操作数是字符串时
    if (PyTypeNum_ISSTRING(type_num1) || PyTypeNum_ISSTRING(type_num2)) {
        // 如果第一个操作数是字符串
        if (PyTypeNum_ISSTRING(type_num1)) {
            // 获取第一个操作数的规范描述符，并确保它是规范的
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(operands[0]));
            if (out_dtypes[0] == NULL) {
                return -1;
            }

            // 创建一个新的 int64 类型描述符
            out_dtypes[1] = PyArray_DescrNewFromType(NPY_INT64);
            if (out_dtypes[1] == NULL) {
                return -1;
            }

            // 对第一个操作数的描述符进行增加引用计数，因为它在 out_dtypes[2] 中被重复使用
            // 这里只关注类型而非实际大小
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[0]);
        }
        // 如果第二个操作数是字符串
        else {
            // 创建一个新的 int64 类型描述符
            out_dtypes[0] = PyArray_DescrNewFromType(NPY_INT64);
            if (out_dtypes[0] == NULL) {
                return -1;
            }

            // 获取第二个操作数的规范描述符，并确保它是规范的
            out_dtypes[1] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(operands[1]));
            if (out_dtypes[1] == NULL) {
                return -1;
            }

            // 对第二个操作数的描述符进行增加引用计数，因为它在 out_dtypes[2] 中被重复使用
            // 这里只关注类型而非实际大小
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[1]);
        }
    }
    else if (type_num1 == NPY_TIMEDELTA) {
        /* 如果第一个操作数的类型是时间差类型（NPY_TIMEDELTA） */

        /* m8[<A>] * int## => m8[<A>] * int64 */
        if (PyTypeNum_ISINTEGER(type_num2) || PyTypeNum_ISBOOL(type_num2)) {
            /* 如果第二个操作数的类型是整数或布尔类型 */
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
            /* 确保第一个操作数的数据类型是规范的 */
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = PyArray_DescrNewFromType(NPY_LONGLONG);
            /* 创建一个新的数据类型描述符，表示int64类型 */
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_LONGLONG;
            /* 更新第二个操作数的类型为int64 */
        }
        /* m8[<A>] * float## => m8[<A>] * float64 */
        else if (PyTypeNum_ISFLOAT(type_num2)) {
            /* 如果第二个操作数的类型是浮点数类型 */
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
            /* 确保第一个操作数的数据类型是规范的 */
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = PyArray_DescrNewFromType(NPY_DOUBLE);
            /* 创建一个新的数据类型描述符，表示float64类型 */
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            type_num2 = NPY_DOUBLE;
            /* 更新第二个操作数的类型为float64 */
        }
        else {
            return raise_binary_type_reso_error(ufunc, operands);
            /* 如果第二个操作数的类型不是整数、布尔或浮点数类型，则引发类型解析错误 */
        }
    }
    else if (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) {
        /* 如果第一个操作数的类型是整数或布尔类型 */

        /* int## * m8[<A>] => int64 * m8[<A>] */
        if (type_num2 == NPY_TIMEDELTA) {
            /* 如果第二个操作数的类型是时间差类型（NPY_TIMEDELTA） */
            out_dtypes[0] = PyArray_DescrNewFromType(NPY_LONGLONG);
            /* 创建一个新的数据类型描述符，表示int64类型 */
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            out_dtypes[1] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[1]));
            /* 确保第二个操作数的数据类型是规范的 */
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[2]);

            type_num1 = NPY_LONGLONG;
            /* 更新第一个操作数的类型为int64 */
        }
        else {
            return raise_binary_type_reso_error(ufunc, operands);
            /* 如果第二个操作数的类型不是时间差类型，则引发类型解析错误 */
        }
    }
    # 如果第一个操作数的类型是浮点数（float##），则执行以下操作
    else if (PyTypeNum_ISFLOAT(type_num1)) {
        /* float## * m8[<A>] => float64 * m8[<A>] */
        # 如果第二个操作数的类型是 NPY_TIMEDELTA
        if (type_num2 == NPY_TIMEDELTA) {
            # 设置输出数据类型为双精度浮点数（NPY_DOUBLE）
            out_dtypes[0] = PyArray_DescrNewFromType(NPY_DOUBLE);
            # 检查是否成功创建输出数据类型，如果失败则返回错误
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            # 确保第二个操作数的数据类型是规范的
            out_dtypes[1] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[1]));
            # 检查第二个输出数据类型是否有效，如果无效则清理资源并返回错误
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            # 第三个输出数据类型与第二个相同
            out_dtypes[2] = out_dtypes[1];
            Py_INCREF(out_dtypes[2]);

            # 将第一个操作数的类型设置为双精度浮点数
            type_num1 = NPY_DOUBLE;
        }
        else {
            # 如果第二个操作数的类型不是 NPY_TIMEDELTA，则返回二元类型解析错误
            return raise_binary_type_reso_error(ufunc, operands);
        }
    }
    else {
        # 如果第一个操作数的类型不是浮点数，则返回二元类型解析错误
        return raise_binary_type_reso_error(ufunc, operands);
    }

    /* Check against the casting rules */
    # 根据转换规则检查数据类型转换是否有效
    if (PyUFunc_ValidateCasting(ufunc, casting, operands, out_dtypes) < 0) {
        # 如果转换无效，清理输出数据类型并返回错误
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    # 返回操作成功
    return 0;
/*
 * This function applies the type resolution rules for division.
 * In particular, there are a number of special cases with datetime:
 *    m8[<A>] / m8[<B>] to  m8[gcd(<A>,<B>)] / m8[gcd(<A>,<B>)]  -> float64
 *    m8[<A>] / int##   to m8[<A>] / int64 -> m8[<A>]
 *    m8[<A>] / float## to m8[<A>] / float64 -> m8[<A>]
 */
NPY_NO_EXPORT int
PyUFunc_DivisionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    type_num1 = PyArray_DESCR(operands[0])->type_num;  // 获取第一个操作数的数据类型编号
    type_num2 = PyArray_DESCR(operands[1])->type_num;  // 获取第二个操作数的数据类型编号

    /* Use the default when datetime and timedelta are not involved */
    // 如果操作数不涉及 datetime 和 timedelta 类型，则使用默认类型解析器
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        return PyUFunc_DefaultTypeResolver(ufunc, casting, operands,
                    type_tup, out_dtypes);  // 调用默认类型解析器并返回结果
    }
    
    // 如果涉及 datetime 或 timedelta 类型，则继续下面的类型解析规则处理
    # 如果第一个操作数的类型为 NPY_TIMEDELTA
    if (type_num1 == NPY_TIMEDELTA) {
        """
         * m8[<A>] / m8[<B>] to
         * m8[gcd(<A>,<B>)] / m8[gcd(<A>,<B>)]  -> float64
         """
        # 如果第二个操作数的类型也为 NPY_TIMEDELTA
        if (type_num2 == NPY_TIMEDELTA) {
            # 计算并设置输出的数据类型，根据两个操作数的类型进行提升
            out_dtypes[0] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            # 如果提升类型失败，则返回错误
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            # 复制第一个输出类型到第二个输出类型
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);

            """
             * TODO: split function into truediv and floordiv resolvers
             """
            # 如果 ufunc 的名称是 "floor_divide"，设置第三个输出类型为 NPY_LONGLONG
            if (strcmp(ufunc->name, "floor_divide") == 0) {
                out_dtypes[2] = PyArray_DescrFromType(NPY_LONGLONG);
            }
            else {
                out_dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);
            }
            # 如果获取第三个输出类型失败，则清理之前分配的内存，并返回错误
            if (out_dtypes[2] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                Py_DECREF(out_dtypes[1]);
                out_dtypes[1] = NULL;
                return -1;
            }
        }
        # 如果第二个操作数是整数类型
        /* m8[<A>] / int## => m8[<A>] / int64 */
        else if (PyTypeNum_ISINTEGER(type_num2)) {
            # 确保第一个输出类型为规范类型
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
            # 如果获取失败，则返回错误
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            # 设置第二个输出类型为 NPY_LONGLONG
            out_dtypes[1] = PyArray_DescrFromType(NPY_LONGLONG);
            # 如果获取失败，则清理之前分配的内存，并返回错误
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            # 复制第一个输出类型到第三个输出类型
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            # 更新第二个操作数的类型为 NPY_LONGLONG
            type_num2 = NPY_LONGLONG;
        }
        # 如果第二个操作数是浮点数类型
        /* m8[<A>] / float## => m8[<A>] / float64 */
        else if (PyTypeNum_ISFLOAT(type_num2)) {
            # 确保第一个输出类型为规范类型
            out_dtypes[0] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(operands[0]));
            # 如果获取失败，则返回错误
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            # 设置第二个输出类型为 NPY_DOUBLE
            out_dtypes[1] = PyArray_DescrNewFromType(NPY_DOUBLE);
            # 如果获取失败，则清理之前分配的内存，并返回错误
            if (out_dtypes[1] == NULL) {
                Py_DECREF(out_dtypes[0]);
                out_dtypes[0] = NULL;
                return -1;
            }
            # 复制第一个输出类型到第三个输出类型
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);

            # 更新第二个操作数的类型为 NPY_DOUBLE
            type_num2 = NPY_DOUBLE;
        }
        # 如果第二个操作数类型不符合上述情况，则返回二元类型解析错误
        else {
            return raise_binary_type_reso_error(ufunc, operands);
        }
    }
    # 如果第一个操作数的类型不是 NPY_TIMEDELTA，则返回二元类型解析错误
    else {
        return raise_binary_type_reso_error(ufunc, operands);
    }

    # 检查是否满足类型转换规则
    if (PyUFunc_ValidateCasting(ufunc, casting, operands, out_dtypes) < 0) {
        # 如果不满足，则清理所有输出类型并返回错误
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    # 操作成功完成，返回 0 表示成功
    return 0;
# 定义一个非导出的函数 PyUFunc_RemainderTypeResolver，用于解析 PyUFuncObject 结构的剩余类型
NPY_NO_EXPORT int
PyUFunc_RemainderTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    # 获取第一个操作数的数据类型编号
    type_num1 = PyArray_DESCR(operands[0])->type_num;
    # 获取第二个操作数的数据类型编号
    type_num2 = PyArray_DESCR(operands[1])->type_num;

    /* 当没有涉及到 datetime 和 timedelta 时使用默认处理 */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        # 调用 PyUFunc_DefaultTypeResolver 处理默认类型解析
        return PyUFunc_DefaultTypeResolver(ufunc, casting, operands,
                    type_tup, out_dtypes);
    }
    # 当第一个操作数的类型是 NPY_TIMEDELTA 时
    if (type_num1 == NPY_TIMEDELTA) {
        # 当第二个操作数的类型也是 NPY_TIMEDELTA 时
        if (type_num2 == NPY_TIMEDELTA) {
            # 提升第一个操作数和第二个操作数的类型为共同的最小公倍数
            out_dtypes[0] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            # 如果无法提升类型，则返回错误
            if (out_dtypes[0] == NULL) {
                return -1;
            }
            # 复制结果类型到输出类型数组的其他位置
            out_dtypes[1] = out_dtypes[0];
            Py_INCREF(out_dtypes[1]);
            out_dtypes[2] = out_dtypes[0];
            Py_INCREF(out_dtypes[2]);
        }
        else {
            # 如果第二个操作数不是 NPY_TIMEDELTA 类型，则返回二进制类型解析错误
            return raise_binary_type_reso_error(ufunc, operands);
        }
    }
    else {
        # 如果第一个操作数不是 NPY_TIMEDELTA 类型，则返回二进制类型解析错误
        return raise_binary_type_reso_error(ufunc, operands);
    }

    /* 根据转换规则验证操作是否符合 */
    if (PyUFunc_ValidateCasting(ufunc, casting, operands, out_dtypes) < 0) {
        # 如果验证失败，则清除输出类型数组，并返回错误
        for (i = 0; i < 3; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    # 所有处理完成，返回成功状态
    return 0;
}

# 定义一个非导出的函数 PyUFunc_TrueDivisionTypeResolver，用于解析 PyUFuncObject 结构的真除类型
NPY_NO_EXPORT int
PyUFunc_TrueDivisionTypeResolver(PyUFuncObject *ufunc,
                                 NPY_CASTING casting,
                                 PyArrayObject **operands,
                                 PyObject *type_tup,
                                 PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;

    # 获取第一个操作数的数据类型编号
    type_num1 = PyArray_DESCR(operands[0])->type_num;
    # 获取第二个操作数的数据类型编号
    type_num2 = PyArray_DESCR(operands[1])->type_num;

    # 当 type_tup 为 NULL 且第一个和第二个操作数都是整数类型或布尔类型时
    if (type_tup == NULL &&
            (PyTypeNum_ISINTEGER(type_num1) || PyTypeNum_ISBOOL(type_num1)) &&
            (PyTypeNum_ISINTEGER(type_num2) || PyTypeNum_ISBOOL(type_num2))) {
        # 使用默认的真除类型解析处理
        return PyUFunc_DefaultTypeResolver(
                ufunc, casting, operands,
                npy_static_pydata.default_truediv_type_tup, out_dtypes);
    }
    # 其他情况使用 PyUFunc_DivisionTypeResolver 处理类型解析
    return PyUFunc_DivisionTypeResolver(ufunc, casting, operands,
                                        type_tup, out_dtypes);
}

# 定义一个静态整型变量
static int
/* 查找用户定义的内循环函数来执行指定的通用函数对象 */

NPY_NO_EXPORT int
PyUFunc_DefaultLegacyInnerLoopSelector(PyUFuncObject *ufunc,
                                PyArray_Descr *const *dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata,
                                int *out_needs_api)
{
    int nargs = ufunc->nargs;  // 获取通用函数对象的参数数量
    const char *types;  // 定义一个字符指针变量 types
    int i, j;

    /*
     * 如果存在用户定义的循环，首先搜索它们。
     * TODO: 需要一个循环选择加速结构，比如哈希表。
     */
    if (ufunc->userloops) {
        switch (find_userloop(ufunc, dtypes,
                    out_innerloop, out_innerloopdata)) {
            /* 错误 */
            case -1:
                return -1;
            /* 找到一个循环 */
            case 1:
                return 0;
        }
    }

    types = ufunc->types;
    // 遍历ufunc对象中的每个类型
    for (i = 0; i < ufunc->ntypes; ++i) {
        // 将类型复制到一个整数数组以便进行匹配
        for (j = 0; j < nargs; ++j) {
            // 检查当前参数的类型是否与目标类型匹配
            if (types[j] != dtypes[j]->type_num) {
                break;
            }
        }
        // 如果所有参数的类型都匹配
        if (j == nargs) {
            // 设置内部循环函数指针
            *out_innerloop = ufunc->functions[i];
            // 设置内部循环数据指针，如果数据为空则设为NULL
            *out_innerloopdata = (ufunc->data == NULL) ? NULL : ufunc->data[i];
            // 返回成功标志
            return 0;
        }

        // 移动到下一组类型
        types += nargs;
    }

    // 如果找不到匹配的内部循环函数，则返回找不到循环错误
    return raise_no_loop_found_error(ufunc, (PyObject **)dtypes);
    # 定义一个静态函数，用于执行通用函数的循环匹配过程
    static int
    ufunc_loop_matches(PyUFuncObject *self,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING output_casting,
                        int any_object,
                        int use_min_scalar,
                        int *types, PyArray_Descr **dtypes,
                        int *out_no_castable_output,
                        char *out_err_src_typecode,
                        char *out_err_dst_typecode)
    {
        # 声明变量 i 用于循环遍历输入和输出操作数，nin 表示输入数量，nop 表示总操作数（输入加输出）
        npy_intp i, nin = self->nin, nop = nin + self->nout;

        /*
         * 首先检查所有输入是否可以安全地转换为此函数所需的类型
         */
        for (i = 0; i < nin; ++i) {
            PyArray_Descr *tmp;

            /*
             * 如果没有输入是对象，并且存在多个循环，不允许转换为对象。
             * 这主要出于性能考虑。除非只有一个对象参数的内部循环构建的自定义ufunc，
             * 否则只实现支持的类型。尝试在浮点参数上使用逻辑或的对象版本似乎不正确。
             */
            if (types[i] == NPY_OBJECT && !any_object && self->ntypes > 1) {
                return 0;
            }
            if (types[i] == NPY_NOTYPE) {
                continue;  /* 通过显式指定匹配 */
            }

            /*
             * 如果类型编号为 NPY_VOID 并且传入了结构 dtypes，则使用结构 dtype 对象。
             * 否则，从类型编号创建新的 dtype 对象。
             */
            if (types[i] == NPY_VOID && dtypes != NULL) {
                tmp = dtypes[i];
                Py_INCREF(tmp);
            }
            else {
                tmp = PyArray_DescrFromType(types[i]);
            }
            if (tmp == NULL) {
                return -1;
            }

            # 如果启用了调试跟踪，则打印详细的类型检查信息
#if NPY_UF_DBG_TRACING
            printf("Checking type for op %d, type %d: ", (int)i, (int)types[i]);
            PyObject_Print((PyObject *)tmp, stdout, 0);
            printf(", operand type: ");
            PyObject_Print((PyObject *)PyArray_DESCR(op[i]), stdout, 0);
            printf("\n");
#endif
            /*
             * 如果所有输入都是标量，则使用常规提升规则，而不是特殊的值检查规则。
             */
            if (!use_min_scalar) {
                if (!PyArray_CanCastTypeTo(PyArray_DESCR(op[i]), tmp,
                                                        input_casting)) {
                    Py_DECREF(tmp);
                    return 0;
                }
            }
            else {
                if (!PyArray_CanCastArrayTo(op[i], tmp, input_casting)) {
                    Py_DECREF(tmp);
                    return 0;
                }
            }
            Py_DECREF(tmp);
        }

        /*
         * 如果所有输入都符合要求，则检查是否能够将结果转换为输出类型。
         */
    for (i = nin; i < nop; ++i) {
        # 循环遍历从 nin 到 nop 的索引范围
        if (types[i] == NPY_NOTYPE) {
            # 如果 types[i] 等于 NPY_NOTYPE，则跳过当前循环，继续下一个迭代
            continue;  /* Matched by being explicitly specified. */
        }
        if (op[i] != NULL) {
            # 如果 op[i] 非空
            # 根据 types[i] 创建一个新的 PyArray_Descr 对象
            PyArray_Descr *tmp = PyArray_DescrFromType(types[i]);
            # 如果创建失败，则返回 -1
            if (tmp == NULL) {
                return -1;
            }
            # 检查是否可以将 tmp 类型转换为 op[i] 的类型，并且符合输出转换规则 output_casting
            if (!PyArray_CanCastTypeTo(tmp, PyArray_DESCR(op[i]),
                                                        output_casting)) {
                # 如果无法转换
                if (!(*out_no_castable_output)) {
                    # 如果没有设置 *out_no_castable_output 标志，则设置为 1
                    *out_no_castable_output = 1;
                    # 设置 *out_err_src_typecode 为 tmp 的类型码
                    *out_err_src_typecode = tmp->type;
                    # 设置 *out_err_dst_typecode 为 op[i] 的类型码
                    *out_err_dst_typecode = PyArray_DESCR(op[i])->type;
                }
                # 释放 tmp 对象
                Py_DECREF(tmp);
                # 返回 0 表示无法转换
                return 0;
            }
            # 释放 tmp 对象
            Py_DECREF(tmp);
        }
    }
    # 如果循环完成，则返回 1 表示所有操作成功
    return 1;
}

static int
set_ufunc_loop_data_types(PyUFuncObject *self, PyArrayObject **op,
                    PyArray_Descr **out_dtypes,
                    int *type_nums, PyArray_Descr **dtypes)
{
    int i, nin = self->nin, nop = nin + self->nout;

    /*
     * 填充 dtypes 数组。
     * 对于输出变量，
     * 还要搜索输入变量以找到匹配的 type_num 来复制，
     * 而不是创建一个新的，类似于保留元数据。
     **/
    for (i = 0; i < nop; ++i) {
        if (dtypes != NULL) {
            out_dtypes[i] = dtypes[i];
            Py_XINCREF(out_dtypes[i]);
        /*
         * 如果 type_num 匹配，则从 'op' 复制 dtype，
         * 以保留元数据。
         */
        }
        else if (op[i] != NULL &&
                 PyArray_DESCR(op[i])->type_num == type_nums[i]) {
            out_dtypes[i] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(op[i]));
        /*
         * 对于输出变量，如果 type_num 匹配，则从 op[0] 复制 dtype，
         * 类似地保留元数据。
         */
        }
        else if (i >= nin && op[0] != NULL &&
                            PyArray_DESCR(op[0])->type_num == type_nums[i]) {
            out_dtypes[i] = NPY_DT_CALL_ensure_canonical(
                    PyArray_DESCR(op[0]));
        /* 否则根据 type_nums[i] 创建一个普通的 descr */
        }
        else {
            out_dtypes[i] = PyArray_DescrFromType(type_nums[i]);
        }

        if (out_dtypes[i] == NULL) {
            goto fail;
        }
    }

    return 0;

fail:
    while (--i >= 0) {
        Py_DECREF(out_dtypes[i]);
        out_dtypes[i] = NULL;
    }
    return -1;
}

/*
 * 在参数和循环中进行搜索
 */
static int
linear_search_userloop_type_resolver(PyUFuncObject *self,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING output_casting,
                        int any_object,
                        int use_min_scalar,
                        PyArray_Descr **out_dtype,
                        int *out_no_castable_output,
                        char *out_err_src_typecode,
                        char *out_err_dst_typecode)
{
    npy_intp i, nop = self->nin + self->nout;

    /* 用于尝试避免重复相同的用户定义循环搜索 */
    int last_userdef = -1;
    for (i = 0; i < nop; ++i) {
        int type_num;

        /* 检查是否还有要检查的 ufunc 参数 */
        if (op[i] == NULL) {
            break;
        }

        // 获取当前操作数 op[i] 的数据类型编号
        type_num = PyArray_DESCR(op[i])->type_num;
        // 如果当前数据类型编号不是上一个用户定义的类型，并且是用户定义类型或者是 NPY_VOID 类型
        if (type_num != last_userdef &&
                (PyTypeNum_ISUSERDEF(type_num) || type_num == NPY_VOID)) {
            PyObject *key, *obj;

            // 更新上一个用户定义类型为当前类型编号
            last_userdef = type_num;

            // 创建一个 Python 整数对象 key，表示当前类型编号
            key = PyLong_FromLong(type_num);
            // 如果创建 key 失败，则返回错误
            if (key == NULL) {
                return -1;
            }
            // 在 self->userloops 字典中查找 key 对应的值
            obj = PyDict_GetItemWithError(self->userloops, key);
            // 减少 key 的引用计数
            Py_DECREF(key);
            // 如果在查找过程中发生错误，则返回 -1
            if (obj == NULL && PyErr_Occurred()) {
                return -1;
            }
            // 如果在字典中没有找到对应的值，继续下一次循环
            else if (obj == NULL) {
                continue;
            }
            // 从 obj 中获取 PyUFunc_Loop1d 结构体指针 funcdata
            PyUFunc_Loop1d *funcdata = PyCapsule_GetPointer(obj, NULL);
            // 如果获取 funcdata 失败，则返回 -1
            if (funcdata == NULL) {
                return -1;
            }
            // 遍历 funcdata 链表，查找匹配的循环函数数据
            for (; funcdata != NULL; funcdata = funcdata->next) {
                int *types = funcdata->arg_types;
                // 根据一些条件判断是否匹配当前的循环函数
                switch (ufunc_loop_matches(self, op,
                            input_casting, output_casting,
                            any_object, use_min_scalar,
                            types, funcdata->arg_dtypes,
                            out_no_castable_output, out_err_src_typecode,
                            out_err_dst_typecode)) {
                    // 发生错误
                    case -1:
                        return -1;
                    // 找到匹配的循环函数
                    case 1:
                        // 设置 ufunc 循环的数据类型
                        set_ufunc_loop_data_types(self, op, out_dtype, types, funcdata->arg_dtypes);
                        return 1;
                }
            }
        }
    }

    /* 没有找到匹配的循环函数 */
    return 0;
}

/*
 * 这是一个静态函数，用于在给定的PyUFuncObject上执行类型元组解析器的搜索。
 * 它根据传入的参数和循环进行搜索。
 */
static int
type_tuple_userloop_type_resolver(PyUFuncObject *self,
                        int n_specified,
                        int *specified_types,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING casting,
                        int any_object,
                        int use_min_scalar,
                        PyArray_Descr **out_dtype)
{
    int i, j, nin = self->nin, nop = nin + self->nout;
    assert(n_specified == nop);
    int types[NPY_MAXARGS];

    /* 用于尝试避免重复搜索相同的用户定义循环 */
    int last_userdef = -1;

    int no_castable_output = 0;
    char err_src_typecode = '-', err_dst_typecode = '-';

    }

    /* 没有找到匹配 */
    return 0;
}


/*
 * 执行ufunc的最佳内部循环的线性搜索。
 *
 * 注意，如果返回错误，调用者必须释放out_dtype中的非零引用。
 * 这个函数本身不负责清理。
 */
NPY_NO_EXPORT int
linear_search_type_resolver(PyUFuncObject *self,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING output_casting,
                        int any_object,
                        PyArray_Descr **out_dtype)
{
    npy_intp i, j, nin = self->nin, nop = nin + self->nout;
    int types[NPY_MAXARGS];
    const char *ufunc_name;
    int no_castable_output = 0;

    /* 用于在强制错误时生成更好的错误消息 */
    char err_dst_typecode = '-', err_src_typecode = '-';

    ufunc_name = ufunc_get_name_cstr(self);

    int promotion_state = get_npy_promotion_state();

    assert(promotion_state != NPY_USE_WEAK_PROMOTION_AND_WARN);
    /* 对于Python int/float/complex，总是使用新的提升 */
    int use_min_scalar;
    if (promotion_state == NPY_USE_LEGACY_PROMOTION) {
        use_min_scalar = should_use_min_scalar(nin, op, 0, NULL);
    }
    else {
        use_min_scalar = should_use_min_scalar_weak_literals(nin, op);
    }

    /* 如果ufunc具有用户定义的循环，则搜索它们 */
    if (self->userloops) {
        switch (linear_search_userloop_type_resolver(self, op,
                                input_casting, output_casting,
                                any_object, use_min_scalar, out_dtype,
                                &no_castable_output, &err_src_typecode,
                                &err_dst_typecode)) {
            /* 错误 */
            case -1:
                return -1;
            /* 找到了一个循环 */
            case 1:
                return 0;
        }
    }
    /*
     * 确定 UFunc 循环。一般情况下，这可能会更快，更好的实现方式可能是让 ufunc
     * 提供一个函数，返回结果类型和内部循环函数。
     *
     * 对于遵循最典型模式的函数，可以提供默认的快速机制，当所有函数的签名为 "xx...x -> x"
     * 对于某个内置数据类型 x 时，按如下方式操作：
     *  - 使用 PyArray_ResultType 获取输出类型
     *  - 根据输出类型编号在表中查找内部循环
     *
     * 在前面代码中找到循环的方法似乎不一致（如 np.add 生成的强制转换表格中的某些不对称性）。
     */
    no_castable_output = 0;  // 初始化没有可转换输出的标志为 0
    for (i = 0; i < self->ntypes; ++i) {  // 遍历每种类型
        const char *orig_types = self->types + i*self->nargs;  // 指向原始类型的指针

        /* 将类型复制到一个整数数组以进行匹配 */
        for (j = 0; j < nop; ++j) {
            types[j] = orig_types[j];  // 复制类型到匹配数组
        }

        switch (ufunc_loop_matches(self, op,
                    input_casting, output_casting,
                    any_object, use_min_scalar,
                    types, NULL,
                    &no_castable_output, &err_src_typecode,
                    &err_dst_typecode)) {
            /* 出错 */
            case -1:
                return -1;  // 返回错误代码
            /* 找到匹配 */
            case 1:
                set_ufunc_loop_data_types(self, op, out_dtype, types, NULL);  // 设置 UFunc 循环的数据类型
                return 0;  // 返回成功代码
        }
    }

    /* 如果找不到函数，抛出错误 */
    if (no_castable_output) {
        PyErr_Format(PyExc_TypeError,
                "ufunc '%s' 输出（类型码 '%c'）无法强制转换为提供的输出参数 "
                "（类型码 '%c'），根据强制转换规则 '%s'",
                ufunc_name, err_src_typecode, err_dst_typecode,
                npy_casting_to_string(output_casting));
    }
    else {
        /*
         * TODO: 如果强制转换规则是 same_kind 或 unsafe，应该再次尝试，并更宽松地查找函数。
         */
        PyErr_Format(PyExc_TypeError,
                "ufunc '%s' 不支持输入类型，并且根据强制转换规则 '%s' 无法安全地将输入强制转换为任何支持的类型",
                ufunc_name,
                npy_casting_to_string(input_casting));
    }

    return -1;  // 返回错误代码
static int
type_tuple_type_resolver_core(PyUFuncObject *self,
        PyArrayObject **op,
        NPY_CASTING input_casting, NPY_CASTING casting,
        int specified_types[],
        int any_object,
        int no_castable_output, int use_min_scalar,
        PyArray_Descr **out_dtype)
{
    int i, j;
    int nop = self->nargs;
    int types[NPY_MAXARGS];

    /* For making a better error message on coercion error */
    // 定义错误目标和源类型码的字符变量，用于更好的错误消息
    char err_dst_typecode = '-', err_src_typecode = '-';

    /* If the ufunc has userloops, search for them. */
    // 如果ufunc有用户自定义循环，进行搜索
    if (self->userloops) {
        switch (type_tuple_userloop_type_resolver(self,
                nop, specified_types,
                op, input_casting, casting,
                any_object, use_min_scalar,
                out_dtype)) {
            /* Error */
            // 错误情况
            case -1:
                return -1;
            /* Found matching loop */
            // 找到匹配的循环
            case 1:
                return 0;
        }
    }

    for (i = 0; i < self->ntypes; ++i) {
        const char *orig_types = self->types + i*self->nargs;

        /*
         * Check specified types and copy into an int array for matching
         * (Mostly duplicated in `type_tuple_userloop_type_resolver`)
         */
        // 检查指定的类型并复制到一个int数组中进行匹配
        for (j = 0; j < nop; ++j) {
            if (specified_types[j] == NPY_NOTYPE) {
                types[j] = orig_types[j];
                continue;
            }
            if (orig_types[j] != specified_types[j]) {
                break;
            }
            /* indicate that we do not have to check this type anymore. */
            // 表示我们不需要再检查这种类型了
            types[j] = NPY_NOTYPE;
        }
        if (j < nop) {
            /* no match */
            // 没有匹配项
            continue;
        }

        switch (ufunc_loop_matches(self, op,
                input_casting, casting,
                any_object, use_min_scalar,
                types, NULL,
                &no_castable_output, &err_src_typecode,
                &err_dst_typecode)) {
            case -1:
                /* Error */
                // 错误情况
                return -1;
            case 0:
                /* Cannot cast inputs */
                // 无法转换输入
                continue;
            case 1:
                /* Success, fill also the NPY_NOTYPE (cast from char to int) */
                // 成功，还填充NPY_NOTYPE（从char到int的转换）
                for (j = 0; j < nop; j++) {
                    types[j] = orig_types[j];
                }
                set_ufunc_loop_data_types(self, op, out_dtype, types, NULL);
                /* In principle, we only need to validate the NPY_NOTYPE ones */
                // 原则上，我们只需要验证NPY_NOTYPE的情况
                if (PyUFunc_ValidateCasting(self, casting, op, out_dtype) < 0) {
                    for (j = 0; j < self->nargs; j++) {
                        Py_DECREF(out_dtype[j]);
                        out_dtype[j] = NULL;
                    }
                    return -1;
                }
                return 0;
        }
    }
    return -2;
}
/*
 * 在给定的 `type_tup` 中执行 ufunc 的线性搜索。
 * 如果返回错误，调用者必须释放 `out_dtype` 中的非零引用。该函数不负责清理工作。
 */
NPY_NO_EXPORT int
type_tuple_type_resolver(PyUFuncObject *self,
                        PyObject *type_tup,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING casting,
                        int any_object,
                        PyArray_Descr **out_dtype)
{
    // 获取输入参数个数 nin 和总参数个数 nop
    int nin = self->nin, nop = nin + self->nout;
    // 初始化指定类型数组
    int specified_types[NPY_MAXARGS];
    // 获取 ufunc 名称
    const char *ufunc_name;
    ufunc_name = ufunc_get_name_cstr(self);

    // 获取 Numpy 推广状态
    int promotion_state = get_npy_promotion_state();
    // 断言推广状态不为 NPY_USE_WEAK_PROMOTION_AND_WARN
    assert(promotion_state != NPY_USE_WEAK_PROMOTION_AND_WARN);

    // 始终使用新的推广方式以确保兼容 Python 的 int/float/complex 类型
    int use_min_scalar;
    if (promotion_state == NPY_USE_LEGACY_PROMOTION) {
        use_min_scalar = should_use_min_scalar(nin, op, 0, NULL);
    }
    else {
        use_min_scalar = should_use_min_scalar_weak_literals(nin, op);
    }

    // 从元组或字符串中填充指定的类型
    const char *bad_type_tup_msg = (
            "Only NumPy must call `ufunc->type_resolver()` explicitly. "
            "NumPy ensures that a type-tuple is normalized now to be a tuple "
            "only containing None or descriptors.  If anything else is passed "
            "(you are seeing this message), the `type_resolver()` was called "
            "directly by a third party. "
            "This is unexpected, please inform the NumPy developers about it. "
            "Also note that `type_resolver` will be phased out, since it must "
            "be replaced.");

    if (PyTuple_CheckExact(type_tup)) {
        Py_ssize_t n = PyTuple_GET_SIZE(type_tup);
        if (n != nop) {
            PyErr_SetString(PyExc_RuntimeError, bad_type_tup_msg);
            return -1;
        }
        for (int i = 0; i < nop; ++i) {
            PyObject *item = PyTuple_GET_ITEM(type_tup, i);
            if (item == Py_None) {
                specified_types[i] = NPY_NOTYPE;
            }
            else {
                if (!PyArray_DescrCheck(item)) {
                    PyErr_SetString(PyExc_RuntimeError, bad_type_tup_msg);
                    return -1;
                }
                specified_types[i] = ((PyArray_Descr *)item)->type_num;
            }
        }
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, bad_type_tup_msg);
        return -1;
    }

    // 调用核心的类型解析器函数
    int res = type_tuple_type_resolver_core(self,
            op, input_casting, casting, specified_types, any_object,
            no_castable_output, use_min_scalar, out_dtype);

    // 如果结果不为 -2，直接返回结果
    if (res != -2) {
        return res;
    }
    /*
     * 如果用户传递了 `dtype=dtype`，它会被转换为 `signature=(None,)*nin + (dtype,)*nout`。
     * 如果签名完全匹配（可以放宽，但对于向后兼容性来说不是必须的），
     * 我们也尝试 `signature=(dtype,)*(nin+nout)`。
     * 由于 reduction 传递中使用 `(dtype, None, dtype)`，我们将所有未指定的 dtype 替换为同类输出类型。
     * 注意，这可能（通常会）导致不安全的类型转换。通常情况下会拒绝这样的转换（但目前不适用于 reductions）。
     * 这曾经是 `dtype=dtype` 的主要含义，但一些调用打破了这种期望，改变它允许将来对像 `np.ldexp` 这样的 ufuncs 有用，
     * 同时也在早期将其规范化为一个 `signature`。
     */
    int homogeneous_type = NPY_NOTYPE;
    if (self->nout > 0) {
        homogeneous_type = specified_types[nin];
        for (int i = nin+1; i < nop; i++) {
            if (specified_types[i] != homogeneous_type) {
                homogeneous_type = NPY_NOTYPE;
                break;
            }
        }
    }
    if (homogeneous_type != NPY_NOTYPE) {
        for (int i = 0; i < nin; i++) {
            if (specified_types[i] != NPY_NOTYPE) {
                /* 永远不要替换已指定的类型！ */
                continue;
            }
            specified_types[i] = homogeneous_type;
        }

        /* 使用同类指定的类型再次尝试。 */
        res = type_tuple_type_resolver_core(self,
                op, input_casting, casting, specified_types, any_object,
                no_castable_output, use_min_scalar, out_dtype);

        if (res != -2) {
            return res;
        }
    }

    /* 如果找不到匹配指定签名和转换方式的循环函数，则抛出错误 */
    PyErr_Format(PyExc_TypeError,
            "No loop matching the specified signature and casting "
            "was found for ufunc %s", ufunc_name);

    return -1;
NPY_NO_EXPORT int
PyUFunc_DivmodTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes)
{
    int type_num1, type_num2;
    int i;

    // 获取第一个操作数的类型编号
    type_num1 = PyArray_DESCR(operands[0])->type_num;
    // 获取第二个操作数的类型编号
    type_num2 = PyArray_DESCR(operands[1])->type_num;

    /* 当涉及到 datetime 和 timedelta 类型时使用默认解析器 */
    if (!PyTypeNum_ISDATETIME(type_num1) && !PyTypeNum_ISDATETIME(type_num2)) {
        // 返回默认类型解析器的结果
        return PyUFunc_DefaultTypeResolver(ufunc, casting, operands,
                    type_tup, out_dtypes);
    }
    // 若第一个类型是 timedelta
    if (type_num1 == NPY_TIMEDELTA) {
        // 若第二个类型也是 timedelta
        if (type_num2 == NPY_TIMEDELTA) {
            // 选择并推广操作数的类型以便匹配
            out_dtypes[0] = PyArray_PromoteTypes(PyArray_DESCR(operands[0]),
                                                PyArray_DESCR(operands[1]));
            // 第二个返回类型等同于第一个
            out_dtypes[1] = out_dtypes[0];
            // 增加第二个返回类型的引用计数
            Py_INCREF(out_dtypes[1]);
            // 第三个返回类型是 NPY_LONGLONG
            out_dtypes[2] = PyArray_DescrFromType(NPY_LONGLONG);
            // 第四个返回类型等同于第一个
            out_dtypes[3] = out_dtypes[0];
            // 增加第四个返回类型的引用计数
            Py_INCREF(out_dtypes[3]);
        }
        else {
            // 若第二个类型不是 timedelta，则引发二进制类型解析错误
            return raise_binary_type_reso_error(ufunc, operands);
        }
    }
    else {
        // 若第一个类型不是 timedelta，则引发二进制类型解析错误
        return raise_binary_type_reso_error(ufunc, operands);
    }

    /* 根据转换规则检查 */
    if (PyUFunc_ValidateCasting(ufunc, casting, operands, out_dtypes) < 0) {
        // 若转换无效，则释放所有返回类型并返回错误
        for (i = 0; i < 4; ++i) {
            Py_DECREF(out_dtypes[i]);
            out_dtypes[i] = NULL;
        }
        return -1;
    }

    // 返回成功
    return 0;
}
```