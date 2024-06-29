# `.\numpy\numpy\_core\src\umath\override.c`

```py
/*
 * 定义 NPY_NO_DEPRECATED_API 为 NPY_API_VERSION，禁用过时 API
 * 定义 NO_IMPORT_ARRAY，禁止导入数组
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY

#include "numpy/ndarraytypes.h"    // 导入 NumPy 的 ndarray 类型定义
#include "numpy/ufuncobject.h"     // 导入 NumPy 的 ufunc 对象定义
#include "npy_import.h"            // 导入 NumPy 的导入函数
#include "npy_static_data.h"       // 导入 NumPy 的静态数据
#include "multiarraymodule.h"      // 导入 NumPy 的多维数组模块
#include "npy_pycompat.h"          // 导入 NumPy 的 Python 兼容性函数
#include "override.h"              // 导入 NumPy 的覆盖功能
#include "ufunc_override.h"        // 导入 NumPy 的ufunc覆盖功能

/*
 * 对于每个位置参数和可能的 "out" 关键字参数，查找标准 ufunc 行为的覆盖，
 * 即非默认的 __array_ufunc__ 方法。
 *
 * 返回覆盖数量，设置 PyObject 数组中的相应对象到 ``with_override``，
 * 并将对应的 __array_ufunc__ 方法设置到 ``methods`` 中（都使用新引用）。
 *
 * 对于给定类别的第一个覆盖只返回一次。
 *
 * 失败时返回 -1。
 */
static int
get_array_ufunc_overrides(PyObject *in_args, PyObject *out_args, PyObject *wheremask_obj,
                          PyObject **with_override, PyObject **methods)
{
    int i;
    int num_override_args = 0;
    int narg, nout, nwhere;

    narg = (int)PyTuple_GET_SIZE(in_args);  // 获取输入参数元组的大小
    /* out_args 可以为 NULL： */
    nout = (out_args != NULL) ? (int)PyTuple_GET_SIZE(out_args) : 0;  // 获取输出参数元组的大小
    nwhere = (wheremask_obj != NULL) ? 1 : 0;  // 检查是否有 wheremask_obj

    for (i = 0; i < narg + nout + nwhere; ++i) {
        PyObject *obj;
        int j;
        int new_class = 1;

        if (i < narg) {
            obj = PyTuple_GET_ITEM(in_args, i);  // 获取输入参数元组中的对象
        }
        else if (i < narg + nout){
            obj = PyTuple_GET_ITEM(out_args, i - narg);  // 获取输出参数元组中的对象
        }
        else {
            obj = wheremask_obj;  // 获取 wheremask_obj 对象
        }
        /*
         * 是否之前见过这个类？如果是，则忽略。
         */
        for (j = 0; j < num_override_args; j++) {
            new_class = (Py_TYPE(obj) != Py_TYPE(with_override[j]));  // 检查是否为新类
            if (!new_class) {
                break;
            }
        }
        if (new_class) {
            /*
             * 现在查看对象是否提供 __array_ufunc__ 方法。但是，我们应该
             * 忽略基本的 ndarray.__ufunc__，因此我们跳过任何 ndarray 以及
             * 未覆盖 __array_ufunc__ 的任何 ndarray 子类实例。
             */
            PyObject *method = PyUFuncOverride_GetNonDefaultArrayUfunc(obj);  // 获取非默认的 __array_ufunc__ 方法
            if (method == NULL) {
                continue;
            }
            if (method == Py_None) {
                PyErr_Format(PyExc_TypeError,
                             "operand '%.200s' does not support ufuncs "
                             "(__array_ufunc__=None)",
                             obj->ob_type->tp_name);  // 报告错误，操作数不支持 ufuncs
                Py_DECREF(method);
                goto fail;
            }
            Py_INCREF(obj);
            with_override[num_override_args] = obj;  // 设置覆盖对象
            methods[num_override_args] = method;    // 设置方法
            ++num_override_args;
        }
    }
    return num_override_args;

fail:
    for (i = 0; i < num_override_args; i++) {
        Py_DECREF(with_override[i]);
        Py_DECREF(methods[i]);
    }
    # 返回整数 -1，通常用于表示函数执行失败或未找到期望的结果
    return -1;
/*
 * Build a dictionary from the keyword arguments, but replace out with the
 * normalized version (and always pass it even if it was passed by position).
 */
static int
initialize_normal_kwds(PyObject *out_args,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject *normal_kwds)
{
    // 如果传入了关键字参数的名称元组 kwnames
    if (kwnames != NULL) {
        // 遍历 kwnames 中的每个元素
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); i++) {
            // 将 args[i + len_args] 作为值，以 PyTuple_GET_ITEM(kwnames, i) 作为键，存入 normal_kwds 字典中
            if (PyDict_SetItem(normal_kwds,
                    PyTuple_GET_ITEM(kwnames, i), args[i + len_args]) < 0) {
                return -1;  // 如果设置失败，返回 -1
            }
        }
    }

    // 如果传入了 out_args
    if (out_args != NULL) {
        /* Replace `out` argument with the normalized version */
        // 将 npy_interned_str.out 作为键，out_args 作为值，存入 normal_kwds 字典中
        int res = PyDict_SetItem(normal_kwds, npy_interned_str.out, out_args);
        if (res < 0) {
            return -1;  // 如果设置失败，返回 -1
        }
    }
    else {
        /* Ensure that `out` is not present. */
        // 确保 normal_kwds 字典中没有名为 npy_interned_str.out 的键
        int res = PyDict_Contains(normal_kwds, npy_interned_str.out);
        if (res < 0) {
            return -1;  // 如果操作失败，返回 -1
        }
        if (res) {
            // 如果存在 npy_interned_str.out 这个键，则从 normal_kwds 字典中删除它
            return PyDict_DelItem(normal_kwds, npy_interned_str.out);
        }
    }
    return 0;  // 函数执行成功，返回 0
}

/*
 * ufunc() and ufunc.outer() accept 'sig' or 'signature'.  We guarantee
 * that it is passed as 'signature' by renaming 'sig' if present.
 * Note that we have already validated that only one of them was passed
 * before checking for overrides.
 */
static int
normalize_signature_keyword(PyObject *normal_kwds)
{
    /* If the keywords include `sig` rename to `signature`. */
    // 如果 normal_kwds 字典中包含键为 "sig"
    PyObject* obj = NULL;
    int result = PyDict_GetItemStringRef(normal_kwds, "sig", &obj);
    if (result == -1) {
        return -1;  // 如果操作失败，返回 -1
    }
    if (result == 1) {
        // 如果找到了键为 "sig" 的项
        if (PyDict_SetItemString(normal_kwds, "signature", obj) < 0) {
            Py_DECREF(obj);  // 设置失败时释放 obj 对象
            return -1;  // 返回 -1 表示失败
        }
        Py_DECREF(obj);  // 释放 obj 对象
        // 成功将 "sig" 改名为 "signature"，现在从 normal_kwds 中删除 "sig"
        if (PyDict_DelItemString(normal_kwds, "sig") < 0) {
            return -1;  // 删除失败，返回 -1
        }
    }
    return 0;  // 函数执行成功，返回 0
}

static int
copy_positional_args_to_kwargs(const char **keywords,
        PyObject *const *args, Py_ssize_t len_args,
        PyObject *normal_kwds)
{
    // 遍历传入的位置参数和对应的关键字数组
    for (Py_ssize_t i = 0; i < len_args; i++) {
        // 如果关键字为 NULL，说明是输入或输出且未在此处设置
        if (keywords[i] == NULL) {
            continue;  // 跳过此次循环
        }
        // 对于 reduce 函数来说，只有 5 个关键字参数是相关的
        if (NPY_UNLIKELY(i == 5)) {
            /*
             * This is only relevant for reduce, which is the only one with
             * 5 keyword arguments.
             */
            assert(strcmp(keywords[i], "initial") == 0);  // 确保关键字为 "initial"
            if (args[i] == npy_static_pydata._NoValue) {
                continue;  // 如果是特殊值 _NoValue，继续下一轮循环
            }
        }

        // 将 args[i] 作为值，keywords[i] 作为键，存入 normal_kwds 字典中
        int res = PyDict_SetItemString(normal_kwds, keywords[i], args[i]);
        if (res < 0) {
            return -1;  // 如果设置失败，返回 -1
        }
    }
    return 0;  // 函数执行成功，返回 0
}
/*
 * Check a set of args for the `__array_ufunc__` method.  If more than one of
 * the input arguments implements `__array_ufunc__`, they are tried in the
 * order: subclasses before superclasses, otherwise left to right. The first
 * (non-None) routine returning something other than `NotImplemented`
 * determines the result. If all of the `__array_ufunc__` operations return
 * `NotImplemented` (or are None), a `TypeError` is raised.
 *
 * Returns 0 on success and 1 on exception. On success, *result contains the
 * result of the operation, if any. If *result is NULL, there is no override.
 */
NPY_NO_EXPORT int
PyUFunc_CheckOverride(PyUFuncObject *ufunc, char *method,
        PyObject *in_args, PyObject *out_args, PyObject *wheremask_obj,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject **result)
{
    int status;

    int num_override_args;
    PyObject *with_override[NPY_MAXARGS];
    PyObject *array_ufunc_methods[NPY_MAXARGS];

    PyObject *method_name = NULL;
    PyObject *normal_kwds = NULL;

    PyObject *override_args = NULL;

    /*
     * Check inputs for overrides
     */
    num_override_args = get_array_ufunc_overrides(
           in_args, out_args, wheremask_obj, with_override, array_ufunc_methods);
    if (num_override_args == -1) {
        goto fail;
    }
    /* No overrides, bail out.*/
    if (num_override_args == 0) {
        *result = NULL;
        return 0;
    }

    /*
     * Normalize ufunc arguments, note that any input and output arguments
     * have already been stored in `in_args` and `out_args`.
     */
    normal_kwds = PyDict_New();
    if (normal_kwds == NULL) {
        goto fail;
    }
    if (initialize_normal_kwds(out_args,
            args, len_args, kwnames, normal_kwds) < 0) {
        goto fail;
    }

    /*
     * Reduce-like methods can pass keyword arguments also by position,
     * in which case the additional positional arguments have to be copied
     * into the keyword argument dictionary. The `__call__` and `__outer__`
     * method have to normalize sig and signature.
     */

    /* ufunc.__call__ */
    if (strcmp(method, "__call__") == 0) {
        status = normalize_signature_keyword(normal_kwds);
    }
    /* ufunc.reduce */
    else if (strcmp(method, "reduce") == 0) {
        static const char *keywords[] = {
                NULL, "axis", "dtype", NULL, "keepdims",
                "initial", "where"};
        status = copy_positional_args_to_kwargs(keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.accumulate */
    else if (strcmp(method, "accumulate") == 0) {
        static const char *keywords[] = {
                NULL, "axis", "dtype", NULL};
        status = copy_positional_args_to_kwargs(keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.reduceat */


注释：
    # 如果方法名与 "reduceat" 相同，则执行以下操作
    else if (strcmp(method, "reduceat") == 0) {
        # 关键字参数列表，仅包含 "axis" 和 "dtype"，其余为 NULL
        static const char *keywords[] = {
                NULL, NULL, "axis", "dtype", NULL};
        # 将位置参数转换为关键字参数
        status = copy_positional_args_to_kwargs(keywords,
                args, len_args, normal_kwds);
    }
    /* ufunc.outer (与调用相同) */
    else if (strcmp(method, "outer") == 0) {
        # 标准化签名中的关键字参数
        status = normalize_signature_keyword(normal_kwds);
    }
    /* ufunc.at */
    else if (strcmp(method, "at") == 0) {
        # 状态设置为 0
        status = 0;
    }
    /* 未知的方法 */
    else {
        # 抛出类型错误异常，指明未知的 ufunc 方法
        PyErr_Format(PyExc_TypeError,
                     "Internal Numpy error: unknown ufunc method '%s' in call "
                     "to PyUFunc_CheckOverride", method);
        # 状态设置为 -1，表示出错
        status = -1;
    }
    # 如果状态不为 0，则跳转到失败处理部分
    if (status != 0) {
        goto fail;
    }

    # 从方法名字符串创建 PyUnicode 对象
    method_name = PyUnicode_FromString(method);
    # 如果创建失败，则跳转到失败处理部分
    if (method_name == NULL) {
        goto fail;
    }

    # 获取输入参数元组的长度
    int len = (int)PyTuple_GET_SIZE(in_args);

    /* 按正确的顺序调用 __array_ufunc__ 函数 */
    }
    # 状态设置为 0
    status = 0;
    /* 找到覆盖，返回它 */
    goto cleanup;
fail:
    # 设置状态为 -1，表示操作失败
    status = -1;
cleanup:
    # 清理操作开始，循环处理所有的覆盖参数
    for (int i = 0; i < num_override_args; i++) {
        // 释放 Python 对象的引用，防止内存泄漏
        Py_XDECREF(with_override[i]);
        // 释放 Python 对象的引用，防止内存泄漏
        Py_XDECREF(array_ufunc_methods[i]);
    }
    // 释放 Python 对象的引用，防止内存泄漏
    Py_XDECREF(method_name);
    // 释放 Python 对象的引用，防止内存泄漏
    Py_XDECREF(normal_kwds);
    // 返回函数执行的状态值
    return status;
}
```