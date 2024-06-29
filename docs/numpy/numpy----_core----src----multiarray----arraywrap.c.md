# `.\numpy\numpy\_core\src\multiarray\arraywrap.c`

```
/*
 * Definitions for dealing with array-wrap or array-prepare.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "get_attr_string.h"

#include "arraywrap.h"
#include "npy_static_data.h"


/*
 * Find the array wrap or array prepare method that applies to the inputs.
 * outputs should NOT be passed, as they are considered individually while
 * applying the wrapping.
 *
 * @param nin number of inputs
 * @param inputs Original input objects
 * @param out_wrap Set to the python callable or None (on success).
 * @param out_wrap_type Set to the type belonging to the wrapper.
 */
NPY_NO_EXPORT int
npy_find_array_wrap(
        int nin, PyObject *const *inputs,
        PyObject **out_wrap, PyObject **out_wrap_type)
{
    PyObject *wrap = NULL;
    PyObject *wrap_type = NULL;

    double priority = 0;  /* silence uninitialized warning */

    /*
     * Iterate through all inputs taking the first one with an __array_wrap__
     * and replace it if a later one has a higher priority.
     * (Currently even priority=-inf can be picked if it is the only argument.)
     */
    for (int i = 0; i < nin; i++) {
        PyObject *obj = inputs[i];
        if (PyArray_CheckExact(obj)) {
            if (wrap == NULL || priority < NPY_PRIORITY) {
                Py_INCREF(Py_None);
                Py_XSETREF(wrap, Py_None);
                priority = 0;
            }
        }
        else if (PyArray_IsAnyScalar(obj)) {
            if (wrap == NULL || priority < NPY_SCALAR_PRIORITY) {
                Py_INCREF(Py_None);
                Py_XSETREF(wrap, Py_None);
                priority = NPY_SCALAR_PRIORITY;
            }
        }
        else {
            PyObject *new_wrap = PyArray_LookupSpecial_OnInstance(obj, npy_interned_str.array_wrap);
            if (new_wrap == NULL) {
                if (PyErr_Occurred()) {
                    goto fail;
                }
                continue;
            }
            double curr_priority = PyArray_GetPriority(obj, 0);
            if (wrap == NULL || priority < curr_priority
                    /* Prefer subclasses `__array_wrap__`: */
                    || (curr_priority == 0 && wrap == Py_None)) {
                Py_XSETREF(wrap, new_wrap);
                Py_INCREF(Py_TYPE(obj));
                Py_XSETREF(wrap_type, (PyObject *)Py_TYPE(obj));
                priority = curr_priority;
            }
            else {
                Py_DECREF(new_wrap);
            }
        }
    }

    if (wrap == NULL) {
        Py_INCREF(Py_None);
        wrap = Py_None;
    }
    if (wrap_type == NULL) {
        Py_INCREF(&PyArray_Type);
        wrap_type = (PyObject *)&PyArray_Type;
    }

    *out_wrap = wrap;
    *out_wrap_type = wrap_type;

    return 0;

  fail:
    Py_XDECREF(wrap);
    Py_XDECREF(wrap_type);
    return -1;
}
/* 获取用于传递给 __array_wrap__ 方法中 context 参数的参数元组。
 *
 * 仅当至少一个参数不为 None 时，才会传递输出参数。
 */
static PyObject *
_get_wrap_prepare_args(NpyUFuncContext *context) {
    // 如果 context 中的 out 为 NULL，增加输入参数的引用计数并返回
    if (context->out == NULL) {
        Py_INCREF(context->in);
        return context->in;
    }
    else {
        // 否则将输入参数和输出参数连接起来并返回
        return PySequence_Concat(context->in, context->out);
    }
}


/*
 * 对结果数组应用数组包装。
 *
 * @param obj 需要包装的对象（应该是一个数组）。
 * @param original_out NULL/None（两者都有效）或者其包装方法始终被使用/优先的对象。
 *        命名方式是因为对于 `out=` 参数，我们总是触发其自己的包装方法。
 * @param wrap 要调用的数组包装函数。
 * @param wrap_type 包装函数所属的类型，在匹配时可以进行快速包装。
 * @param context ufunc 的上下文或者 NULL，在普通 ufunc 调用中使用。
 * @param return_scalar 是否优先返回标量。当传递 `original_out` 时忽略，因为 `out` 参数从不是标量。
 * @param force_wrap 如果为 True，我们总是调用包装（对于子类），因为 ufunc 可能已经改变了内容。
 */
NPY_NO_EXPORT PyObject *
npy_apply_wrap(
        PyObject *obj, PyObject *original_out,
        PyObject *wrap, PyObject *wrap_type,
        NpyUFuncContext *context, npy_bool return_scalar, npy_bool force_wrap)
{
    PyObject *res = NULL;
    PyObject *new_wrap = NULL;
    PyArrayObject *arr = NULL;
    PyObject *err_type, *err_value, *traceback;

    /* 如果提供了 original_out，并且不是 NULL，则优先使用实际的 out 对象进行包装： */
    if (original_out != NULL && original_out != Py_None) {
        /* 
         * 如果传递了原始输出对象，则包装不应更改它。特别是，将其转换为标量没有意义。
         * 因此，替换传入的 wrap 和 wrap_type。
         */
        return_scalar = NPY_FALSE;

        if (PyArray_CheckExact(original_out)) {
            /* 替换传入的 wrap/wrap_type（借用的引用）为默认值。 */
            wrap = Py_None;
            wrap_type = (PyObject *)&PyArray_Type;
        }
        else {
            /* 替换传入的 wrap/wrap_type（借用的引用）为 new_wrap/type。 */
            new_wrap = PyArray_LookupSpecial_OnInstance(
                    original_out, npy_interned_str.array_wrap);
            if (new_wrap != NULL) {
                wrap = new_wrap;
                wrap_type = (PyObject *)Py_TYPE(original_out);
            }
            else if (PyErr_Occurred()) {
                return NULL;
            }
        }
    }
    /*
     * 如果结果与包装类型相同（并且没有 original_out，当我们应该包装 `self` 时）
     * 我们可以跳过包装，除非我们需要标量返回。
     */
    /*
     * 如果不返回标量并且不强制包装，并且对象的类型与wrap_type相同，
     * 则直接返回对象，增加其引用计数。
     */
    if (!return_scalar && !force_wrap
            && (PyObject *)Py_TYPE(obj) == wrap_type) {
        Py_XDECREF(new_wrap);  // 释放 new_wrap 的引用
        Py_INCREF(obj);  // 增加对象的引用计数
        return obj;  // 返回对象
    }

    /*
     * 如果 wrap 是 Py_None，则释放 new_wrap 的引用，增加对象的引用计数。
     * 如果 return_scalar 为真，则使用 PyArray_Return 转换为标量。
     */
    if (wrap == Py_None) {
        Py_XDECREF(new_wrap);  // 释放 new_wrap 的引用
        Py_INCREF(obj);  // 增加对象的引用计数
        if (return_scalar) {
            /* 
             * 当需要时使用 PyArray_Return 将对象转换为标量
             * （PyArray_Return 实际上检查非数组情况）。
             */
            return PyArray_Return((PyArrayObject *)obj);
        }
        else {
            return obj;  // 返回对象
        }
    }

    /*
     * 需要调用 array-wrap 函数。在某些分支中，输入可能是非数组。
     * （尽管我们应该尝试逐步淘汰所有这些分支！）
     */
    PyObject *py_context = NULL;
    if (context == NULL) {
        Py_INCREF(Py_None);  // 增加对 Py_None 的引用计数
        py_context = Py_None;  // 将 py_context 设置为 Py_None
    }
    else {
        /* 使用适当的上下文调用方法 */
        PyObject *args_tup = _get_wrap_prepare_args(context);  // 获取准备参数的元组
        if (args_tup == NULL) {
            goto finish;  // 如果获取参数元组失败，则跳转到 finish 标签
        }
        py_context = Py_BuildValue("OOi",
                context->ufunc, args_tup, context->out_i);  // 创建 py_context
        Py_DECREF(args_tup);  // 释放参数元组的引用
        if (py_context == NULL) {
            goto finish;  // 如果创建 py_context 失败，则跳转到 finish 标签
        }
    }

    if (PyArray_Check(obj)) {
        Py_INCREF(obj);  // 增加对象的引用计数
        arr = (PyArrayObject *)obj;  // 将 obj 转换为 PyArrayObject 类型
    }
    else {
        /*
         * TODO: 理想情况下，我们永远不会进入此分支！
         * 但是当我们从 Python 使用时，NumPy 将 0 维数组转换为标量，
         * 这意味着在 Python 中将其转换回数组可能不需要包装，这可能会出现问题。
         */
        arr = (PyArrayObject *)PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);  // 尝试将 obj 转换为 PyArrayObject
        if (arr == NULL) {
            goto finish;  // 如果转换失败，则跳转到 finish 标签
        }
    }

    res = PyObject_CallFunctionObjArgs(
            wrap, arr, py_context,
            (return_scalar && PyArray_NDIM(arr) == 0) ? Py_True : Py_False,
            NULL);  // 调用 wrap 函数
    if (res != NULL) {
        goto finish;  // 如果调用成功，则跳转到 finish 标签
    }
    else if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
        goto finish;  // 如果异常不是 TypeError，则跳转到 finish 标签
    }

    /*
     * 在不传递 return_scalar 参数的情况下重试。如果成功，则发出 DeprecationWarning。
     * 当 context 为 None 时，无需尝试这样做。
     */
    if (py_context != Py_None) {
        PyErr_Fetch(&err_type, &err_value, &traceback);  // 获取当前的错误信息
        res = PyObject_CallFunctionObjArgs(wrap, arr, py_context, NULL);  // 再次调用 wrap 函数
        if (res != NULL) {
            goto deprecation_warning;  // 如果调用成功，则跳转到 deprecation_warning 标签
        }
        Py_DECREF(err_type);  // 释放错误类型的引用
        Py_XDECREF(err_value);  // 释放错误值的引用
        Py_XDECREF(traceback);  // 释放 traceback 的引用
        if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
            goto finish;  // 如果异常不是 TypeError，则跳转到 finish 标签
        }
    }

    /* 
     * 在不传递 context 和 return_scalar 参数的情况下重试。
     * 如果成功，则发出 DeprecationWarning。
     */
    PyErr_Fetch(&err_type, &err_value, &traceback);  // 获取当前的错误信息
    res = PyObject_CallFunctionObjArgs(wrap, arr, NULL);  // 再次调用 wrap 函数
    # 如果 res 为 NULL，说明发生了错误，需要清理错误状态并跳转到结束标签
    if (res == NULL) {
        // 清理错误类型对象的引用计数
        Py_DECREF(err_type);
        // 清理错误值对象的引用计数
        Py_XDECREF(err_value);
        // 清理 traceback 对象的引用计数
        Py_XDECREF(traceback);
        // 跳转到函数结束标签
        goto finish;
    }

  deprecation_warning:
    /* 如果程序执行到这里，说明原始的错误仍然被保留。 */
    /* 在2024年1月17日被弃用，NumPy 2.0 */
    // 发出弃用警告，警告未来版本中 __array_wrap__ 必须接受 context 和 return_scalar 参数（位置参数）。
    // （在NumPy 2.0版本弃用）
    if (DEPRECATE(
            "__array_wrap__ must accept context and return_scalar arguments "
            "(positionally) in the future. (Deprecated NumPy 2.0)") < 0) {
        // 将当前错误链接到先前的错误链中
        npy_PyErr_ChainExceptionsCause(err_type, err_value, traceback);
        // 清理结果对象的引用计数
        Py_CLEAR(res);
    }
    else {
        // 清理错误类型对象的引用计数
        Py_DECREF(err_type);
        // 清理错误值对象的引用计数
        Py_XDECREF(err_value);
        // 清理 traceback 对象的引用计数
        Py_XDECREF(traceback);
    }

  finish:
    // 清理 py_context 对象的引用计数
    Py_XDECREF(py_context);
    // 清理 arr 对象的引用计数
    Py_XDECREF(arr);
    // 清理 new_wrap 对象的引用计数
    Py_XDECREF(new_wrap);
    // 返回结果对象
    return res;
/*
 * 调用 arr_of_subclass 的 __array_wrap__(towrap) 方法，
 * 使 'towrap' 具有与 'arr_of_subclass' 相同的 ndarray 子类。
 * `towrap` 应为一个基类 ndarray。
 */
NPY_NO_EXPORT PyObject *
npy_apply_wrap_simple(PyArrayObject *arr_of_subclass, PyArrayObject *towrap)
{
    /*
     * 当只有单个其他数组时，与 apply-wrap 相同，
     * 我们可以使用 `original_out`，而不必担心传递一个有用的 wrap 对象。
     */
    return npy_apply_wrap(
            (PyObject *)towrap, (PyObject *)arr_of_subclass, Py_None, NULL,
            NULL, NPY_FALSE, NPY_TRUE);
}
```