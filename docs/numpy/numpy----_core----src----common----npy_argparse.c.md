# `.\numpy\numpy\_core\src\common\npy_argparse.c`

```
/**
 * Define NPY_NO_DEPRECATED_API to use the latest NumPy API version.
 * This prevents the use of deprecated API features.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/**
 * Define _MULTIARRAYMODULE to specify the multi-array module.
 * This is used to indicate the module being compiled.
 */
#define _MULTIARRAYMODULE

/**
 * Define PY_SSIZE_T_CLEAN to use Py_ssize_t API for Python C API functions.
 * This ensures compatibility with Python's size type, which may vary across
 * different versions and configurations.
 */
#define PY_SSIZE_T_CLEAN

/**
 * Include Python.h to gain access to Python C API functions and definitions.
 * This header file provides essential macros, types, and function declarations
 * for extending Python with C or C++ code.
 */
#include <Python.h>

/**
 * Include ndarraytypes.h to access NumPy's array and dtype definitions.
 * This header file defines data structures and macros necessary for working
 * with NumPy arrays and data types in C extension modules.
 */
#include "numpy/ndarraytypes.h"

/**
 * Include npy_2_compat.h for backward compatibility with older NumPy versions.
 * This ensures that the extension module remains compatible with NumPy's API
 * across different versions of the library.
 */
#include "numpy/npy_2_compat.h"

/**
 * Include npy_argparse.h to utilize argument parsing utilities provided by NumPy.
 * This header file provides functions and macros for parsing and validating
 * function arguments passed from Python to C extension modules.
 */
#include "npy_argparse.h"

/**
 * Include npy_import.h for functions related to importing NumPy in C extension modules.
 * This header file includes functions and macros that facilitate importing NumPy
 * and ensuring compatibility across different configurations.
 */
#include "npy_import.h"

/**
 * Include arrayfunction_override.h for overriding array functions in NumPy.
 * This header file contains declarations and macros that enable overriding or
 * extending built-in NumPy array functions with custom implementations.
 */
#include "arrayfunction_override.h"

/**
 * Small wrapper converting Python integer to C int using PyLong_AsLong function.
 *
 * This function handles conversion of Python integers to C int, checking for
 * overflow conditions and type errors.
 *
 * @param obj The Python object to convert (should be an integer)
 * @param value Pointer to the output C int value
 * @returns NPY_SUCCEED on success, NPY_FAIL on failure
 */
NPY_NO_EXPORT int
PyArray_PythonPyIntFromInt(PyObject *obj, int *value)
{
    /* Python's behavior is to check explicitly for float types */
    if (NPY_UNLIKELY(PyFloat_Check(obj))) {
        PyErr_SetString(PyExc_TypeError,
                        "integer argument expected, got float");
        return NPY_FAIL;
    }

    long result = PyLong_AsLong(obj);
    if (NPY_UNLIKELY((result == -1) && PyErr_Occurred())) {
        return NPY_FAIL;
    }
    if (NPY_UNLIKELY((result > INT_MAX) || (result < INT_MIN))) {
        PyErr_SetString(PyExc_OverflowError,
                        "Python int too large to convert to C int");
        return NPY_FAIL;
    }
    else {
        *value = (int)result;
        return NPY_SUCCEED;
    }
}

/**
 * Type definition for a function pointer to convert a Python object to a C type.
 */
typedef int convert(PyObject *, void *);

/**
 * Internal function to initialize keyword argument parsing for NumPy functions.
 *
 * This function performs several tasks:
 * 1. Checks input consistency to detect coding errors, such as missing | after optional parameters.
 * 2. Determines the number of positional-only arguments, total arguments, required arguments,
 *    and keyword arguments.
 * 3. Interns all keyword argument strings to optimize parsing performance by using
 *    identity-based comparisons and reducing string creation overhead.
 *
 * @param funcname Name of the function being parsed, used mainly for error reporting.
 * @param cache A cache object stored statically within the parsing function.
 * @param va_orig Argument list passed to npy_parse_arguments.
 * @return 0 on success, -1 on failure
 */
static int
initialize_keywords(const char *funcname,
                    _NpyArgParserCache *cache, va_list va_orig) {
    va_list va;
    int nargs = 0;              // Total number of arguments
    int nkwargs = 0;            // Number of keyword arguments
    int npositional_only = 0;   // Number of positional-only arguments
    int nrequired = 0;          // Number of required arguments
    int npositional = 0;        // Number of positional arguments
    char state = '\0';          // State variable for argument parsing

    va_copy(va, va_orig);
    while (1) {
        /* Count length first: */
        // 从可变参数中依次取出参数：name（字符串指针）、converter（转换器指针）、data（数据指针）
        char *name = va_arg(va, char *);
        convert *converter = va_arg(va, convert *);
        void *data = va_arg(va, void *);

        /* Check if this is the sentinel, only converter may be NULL */
        // 检查是否为哨兵值（结束标志），只有 converter 可能为 NULL
        if ((name == NULL) && (converter == NULL) && (data == NULL)) {
            break;
        }

        // 如果 name 为 NULL，则抛出异常并返回 -1
        if (name == NULL) {
            PyErr_Format(PyExc_SystemError,
                    "NumPy internal error: name is NULL in %s() at "
                    "argument %d.", funcname, nargs);
            va_end(va);
            return -1;
        }
        // 如果 data 为 NULL，则抛出异常并返回 -1
        if (data == NULL) {
            PyErr_Format(PyExc_SystemError,
                    "NumPy internal error: data is NULL in %s() at "
                    "argument %d.", funcname, nargs);
            va_end(va);
            return -1;
        }

        // 参数计数增加
        nargs += 1;
        // 如果参数名以 '|' 开头
        if (*name == '|') {
            // 如果当前状态为 '$'，则抛出异常并返回 -1
            if (state == '$') {
                PyErr_Format(PyExc_SystemError,
                        "NumPy internal error: positional argument `|` "
                        "after keyword only `$` one to %s() at argument %d.",
                        funcname, nargs);
                va_end(va);
                return -1;
            }
            // 更新状态为 '|'
            state = '|';
            // 将 name 指针向前移动到实际的参数名位置
            name++;  /* advance to actual name. */
            // 增加位置参数计数
            npositional += 1;
        }
        // 如果参数名以 '$' 开头
        else if (*name == '$') {
            // 更新状态为 '$'
            state = '$';
            // 将 name 指针向前移动到实际的参数名位置
            name++;  /* advance to actual name. */
        }
        // 如果参数名不以 '|' 或 '$' 开头
        else {
            // 如果状态不是初始状态 '\0'，则抛出异常并返回 -1
            if (state != '\0') {
                PyErr_Format(PyExc_SystemError,
                        "NumPy internal error: non-required argument after "
                        "required | or $ one to %s() at argument %d.",
                        funcname, nargs);
                va_end(va);
                return -1;
            }

            // 必需参数计数增加，位置参数计数增加
            nrequired += 1;
            npositional += 1;
        }

        // 如果参数名为空字符串 '\0'
        if (*name == '\0') {
            // 增加位置参数且只能作为位置参数的计数
            npositional_only += 1;
            // 如果状态为 '$' 或者非关键字参数的数量与位置参数数量不一致，则抛出异常并返回 -1
            if (state == '$' || npositional_only != npositional) {
                PyErr_Format(PyExc_SystemError,
                        "NumPy internal error: non-kwarg marked with $ "
                        "to %s() at argument %d or positional only following "
                        "kwarg.", funcname, nargs);
                va_end(va);
                return -1;
            }
        }
        // 如果参数名不为空字符串
        else {
            // 关键字参数计数增加
            nkwargs += 1;
        }
    }
    va_end(va);

    // 如果位置参数计数为 -1，则将其设置为 nargs
    if (npositional == -1) {
        npositional = nargs;
    }

    // 如果参数数量超过 _NPY_MAX_KWARGS，抛出异常并返回 -1
    if (nargs > _NPY_MAX_KWARGS) {
        PyErr_Format(PyExc_SystemError,
                "NumPy internal error: function %s() has %d arguments, but "
                "the maximum is currently limited to %d for easier parsing; "
                "it can be increased by modifying `_NPY_MAX_KWARGS`.",
                funcname, nargs, _NPY_MAX_KWARGS);
        return -1;
    }
    /*
     * 设置缓存对象的参数信息，用于后续的处理。
     */
    cache->nargs = nargs;
    cache->npositional_only = npositional_only;
    cache->npositional = npositional;
    cache->nrequired = nrequired;
    
    /* 
     * 将 kw_strings 数组全部置为 NULL，以便后续更容易进行清理（并且保证 NULL 结尾）。
     */
    memset(cache->kw_strings, 0, sizeof(PyObject *) * (nkwargs + 1));
    
    /*
     * 使用 va_orig 复制一个可变参数列表，以便后续操作。
     */
    va_copy(va, va_orig);
    for (int i = 0; i < nargs; i++) {
        /* 
         * 遍历非关键字参数，这些参数不需要额外的设置。
         */
        char *name = va_arg(va, char *);
        va_arg(va, convert *);
        va_arg(va, void *);
    
        if (*name == '|' || *name == '$') {
            name++;  /* 忽略 | 和 $ 符号 */
        }
        if (i >= npositional_only) {
            int i_kwarg = i - npositional_only;
            /*
             * 如果当前参数是关键字参数，则将其字符串名转换为 Python Unicode 对象并存储在缓存的 kw_strings 数组中。
             * 如果转换失败，则清理资源并跳转到错误处理标签。
             */
            cache->kw_strings[i_kwarg] = PyUnicode_InternFromString(name);
            if (cache->kw_strings[i_kwarg] == NULL) {
                va_end(va);
                goto error;
            }
        }
    }
    
    /*
     * 结束可变参数的处理。
     */
    va_end(va);
    return 0;
/**
 * 用于处理参数解析的通用辅助函数
 *
 * 查看宏版本以获取如何使用此函数的示例模式。
 *
 * @param funcname 函数名字符串
 * @param cache 参数解析器缓存对象
 * @param args 传递给 Python 的参数（METH_FASTCALL）
 * @param len_args 参数数组的长度
 * @param kwnames 关键字参数的名称
 * @param ... 参数列表（参见宏版本），以 NULL, NULL, NULL 结尾：名称，转换器，值
 * @return 成功返回 0，失败返回 -1
 */
NPY_NO_EXPORT int
_npy_parse_arguments(const char *funcname,
         _NpyArgParserCache *cache,
         PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
         ...)
{
    // 如果 npositional 未初始化
    if (NPY_UNLIKELY(cache->npositional == -1)) {
        va_list va;
        va_start(va, kwnames);

        // 初始化关键字参数
        int res = initialize_keywords(funcname, cache, va);
        va_end(va);
        if (res < 0) {
            return -1;
        }
    }

    // 如果传入参数个数大于需要的位置参数个数
    if (NPY_UNLIKELY(len_args > cache->npositional)) {
        // 抛出位置参数个数不正确的错误
        return raise_incorrect_number_of_positional_args(
                funcname, cache, len_args);
    }

    /* NOTE: Could remove the limit but too many kwargs are slow anyway. */
    // 所有参数的数组
    PyObject *all_arguments[NPY_MAXARGS];

    // 将传入的位置参数放入 all_arguments 数组
    for (Py_ssize_t i = 0; i < len_args; i++) {
        all_arguments[i] = args[i];
    }

    /* Without kwargs, do not iterate all converters. */
    // 最大参数个数为传入位置参数个数
    int max_nargs = (int)len_args;
    Py_ssize_t len_kwargs = 0;

    // 如果有关键字参数，首先处理它们
    // 如果关键字参数列表不为空
    if (NPY_LIKELY(kwnames != NULL)) {
        // 获取关键字参数的个数
        len_kwargs = PyTuple_GET_SIZE(kwnames);
        // 获取缓存中的最大参数个数
        max_nargs = cache->nargs;

        // 将额外的位置参数初始化为NULL
        for (int i = len_args; i < cache->nargs; i++) {
            all_arguments[i] = NULL;
        }

        // 遍历关键字参数列表
        for (Py_ssize_t i = 0; i < len_kwargs; i++) {
            // 获取关键字参数的键和对应的值
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            PyObject *value = args[i + len_args];
            PyObject *const *name;

            /* 超快速路径，检查对象的身份是否相同: */
            // 遍历缓存中的关键字字符串列表，查找是否有身份相同的关键字对象
            for (name = cache->kw_strings; *name != NULL; name++) {
                if (*name == key) {
                    break;
                }
            }
            // 如果没有找到身份相同的关键字对象
            if (NPY_UNLIKELY(*name == NULL)) {
                /* 慢速回退，如果由于某些原因身份检查失败 */
                // 再次遍历缓存中的关键字字符串列表，进行对象的相等性比较
                for (name = cache->kw_strings; *name != NULL; name++) {
                    int eq = PyObject_RichCompareBool(*name, key, Py_EQ);
                    if (eq == -1) {
                        return -1;
                    }
                    else if (eq) {
                        break;
                    }
                }
                // 如果还是没有找到匹配的关键字对象
                if (NPY_UNLIKELY(*name == NULL)) {
                    /* 无效的关键字参数。 */
                    PyErr_Format(PyExc_TypeError,
                            "%s() got an unexpected keyword argument '%S'",
                            funcname, key);
                    return -1;
                }
            }

            // 计算参数在函数参数列表中的位置
            Py_ssize_t param_pos = (
                    (name - cache->kw_strings) + cache->npositional_only);

            /* 可能会有相同位置的参数 */
            // 如果该位置已经存在参数对象，则报错
            if (NPY_UNLIKELY(all_arguments[param_pos] != NULL)) {
                PyErr_Format(PyExc_TypeError,
                        "argument for %s() given by name ('%S') and position "
                        "(position %zd)", funcname, key, param_pos);
                return -1;
            }

            // 将参数值存储到参数列表中对应的位置
            all_arguments[param_pos] = value;
        }
    }

    /*
     * 这时候 `all_arguments` 中要么是 NULL 要么是对象
     * 参数和关键字参数的总数不会超过函数声明的最大参数个数，否则上面的逻辑会检测到错误。
     */
    // 断言：位置参数个数加上关键字参数个数不会超过缓存中声明的最大参数个数
    assert(len_args + len_kwargs <= cache->nargs);

    /* 现在 `all_arguments` 包含的要么是NULL要么是实际的对象 */
    // 初始化可变参数列表
    va_list va;
    va_start(va, kwnames);
    // 遍历可变参数列表，处理每个参数的转换
    for (int i = 0; i < max_nargs; i++) {
        // 跳过当前可变参数列表中的下一个参数
        va_arg(va, char *);
        // 获取下一个参数作为转换器的指针
        convert *converter = va_arg(va, convert *);
        // 获取下一个参数作为需要填充数据的指针
        void *data = va_arg(va, void *);

        // 如果当前参数为空，则继续处理下一个参数
        if (all_arguments[i] == NULL) {
            continue;
        }

        // 定义变量 res 来存储转换结果
        int res;
        // 如果转换器为空，则直接将当前参数赋值给数据指针
        if (converter == NULL) {
            *((PyObject **) data) = all_arguments[i];
            continue;
        }
        // 使用转换器将当前参数转换为目标数据，并获取转换结果
        res = converter(all_arguments[i], data);

        // 根据转换结果判断下一步动作
        // 如果转换成功，继续处理下一个参数
        if (NPY_UNLIKELY(res == NPY_SUCCEED)) {
            continue;
        }
        // 如果转换失败，跳转到转换失败的处理标签
        else if (NPY_UNLIKELY(res == NPY_FAIL)) {
            /* It is usually the users responsibility to clean up. */
            goto converting_failed;
        }
        // 如果需要支持清理操作，给出错误信息并跳转到转换失败的处理标签
        else if (NPY_UNLIKELY(res == Py_CLEANUP_SUPPORTED)) {
            /* TODO: Implementing cleanup if/when needed should not be hard */
            PyErr_Format(PyExc_SystemError,
                    "converter cleanup of parameter %d to %s() not supported.",
                    i, funcname);
            goto converting_failed;
        }
        // 如果出现了意料之外的结果，触发断言错误
        assert(0);
    }

    // 检查是否传递了足够的必需参数
    // 通常情况下，必需参数不作为关键字参数传递
    if (NPY_UNLIKELY(len_args < cache->nrequired)) {
        // 如果最大参数个数小于必需参数个数，抛出缺少参数的异常
        if (NPY_UNLIKELY(max_nargs < cache->nrequired)) {
            raise_missing_argument(funcname, cache, max_nargs);
            goto converting_failed;
        }
        // 遍历检查每个必需参数，如果为空，抛出缺少参数的异常
        for (int i = 0; i < cache->nrequired; i++) {
            if (NPY_UNLIKELY(all_arguments[i] == NULL)) {
                raise_missing_argument(funcname, cache, i);
                goto converting_failed;
            }
        }
    }

    // 结束可变参数处理
    va_end(va);
    // 返回成功状态
    return 0;

converting_failed:
    // 处理转换失败的标签，返回失败状态
    return -1;
converting_failed:
    // 结束可变参数列表的使用
    va_end(va);
    // 返回错误码 -1，表示转换失败
    return -1;
}
```