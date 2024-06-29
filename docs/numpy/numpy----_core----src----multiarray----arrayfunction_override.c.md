# `.\numpy\numpy\_core\src\multiarray\arrayfunction_override.c`

```
/*
 * Define NPY_NO_DEPRECATED_API to use the latest NumPy API version.
 * This helps in avoiding deprecated features.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * Define _MULTIARRAYMODULE to include definitions specific to the multiarray module.
 * This is necessary for integrating with NumPy's multiarray functionalities.
 */
#define _MULTIARRAYMODULE

#include <Python.h>
#include "structmember.h"

#include "numpy/ndarraytypes.h"
#include "get_attr_string.h"
#include "npy_import.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"

#include "arrayfunction_override.h"

/*
 * Get an object's __array_function__ method in the fastest way possible.
 * Never raises an exception. Returns NULL if the method doesn't exist.
 */
static PyObject *
get_array_function(PyObject *obj)
{
    /* Fast return for ndarray */
    if (PyArray_CheckExact(obj)) {
        Py_INCREF(npy_static_pydata.ndarray_array_function);
        return npy_static_pydata.ndarray_array_function;
    }

    /*
     * Lookup the __array_function__ attribute for the given object.
     * If not found, return NULL. Clear any previous errors if they occurred.
     */
    PyObject *array_function = PyArray_LookupSpecial(obj, npy_interned_str.array_function);
    if (array_function == NULL && PyErr_Occurred()) {
        PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
    }

    return array_function;
}


/*
 * Like list.insert(), but for C arrays of PyObject*. Skips error checking.
 */
static void
pyobject_array_insert(PyObject **array, int length, int index, PyObject *item)
{
    /*
     * Insert an item into a C array of PyObject* at the specified index.
     * Shift subsequent elements to the right to make space for the new item.
     */
    for (int j = length; j > index; j--) {
        array[j] = array[j - 1];
    }
    array[index] = item;
}


/*
 * Collects arguments with __array_function__ and their corresponding methods
 * in the order in which they should be tried (i.e., skipping redundant types).
 * `relevant_args` is expected to have been produced by PySequence_Fast.
 * Returns the number of arguments, or -1 on failure.
 */
static int
get_implementing_args_and_methods(PyObject *relevant_args,
                                  PyObject **implementing_args,
                                  PyObject **methods)
{
    int num_implementing_args = 0;

    /*
     * Extract the array of items from `relevant_args` and determine its length.
     */
    PyObject **items = PySequence_Fast_ITEMS(relevant_args);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(relevant_args);
    // 遍历 items 数组中的每个元素
    for (Py_ssize_t i = 0; i < length; i++) {
        // 默认将当前元素视为新类型
        int new_class = 1;
        // 获取当前元素作为 argument
        PyObject *argument = items[i];

        /* 我们之前见过这种类型吗？ */
        // 遍历已知的实现参数列表
        for (int j = 0; j < num_implementing_args; j++) {
            // 检查当前 argument 是否与某个已知实现参数具有相同的类型
            if (Py_TYPE(argument) == Py_TYPE(implementing_args[j])) {
                // 如果找到相同类型的实现参数，标记为不是新类型
                new_class = 0;
                break;
            }
        }
        // 如果是新类型
        if (new_class) {
            // 获取 argument 的数组函数方法
            PyObject *method = get_array_function(argument);

            // 如果成功获取方法
            if (method != NULL) {
                int arg_index;

                // 检查是否超过了最大的参数数量限制
                if (num_implementing_args >= NPY_MAXARGS) {
                    PyErr_Format(
                        PyExc_TypeError,
                        "maximum number (%d) of distinct argument types " \
                        "implementing __array_function__ exceeded",
                        NPY_MAXARGS);
                    Py_DECREF(method);
                    // 失败时跳转到 fail 标签处
                    goto fail;
                }

                /* "subclasses before superclasses, otherwise left to right" */
                // 确定当前 argument 应该插入到 implementing_args 数组的位置
                arg_index = num_implementing_args;
                for (int j = 0; j < num_implementing_args; j++) {
                    PyObject *other_type;
                    other_type = (PyObject *)Py_TYPE(implementing_args[j]);
                    // 检查当前 argument 是否是已知实现参数 j 的实例
                    if (PyObject_IsInstance(argument, other_type)) {
                        arg_index = j;
                        break;
                    }
                }
                // 增加 argument 的引用计数并插入到 implementing_args 数组中
                Py_INCREF(argument);
                pyobject_array_insert(implementing_args, num_implementing_args,
                                      arg_index, argument);
                // 同样插入方法到 methods 数组中
                pyobject_array_insert(methods, num_implementing_args,
                                      arg_index, method);
                // 增加实现参数的数量
                ++num_implementing_args;
            }
        }
    }
    // 返回找到的实现参数的数量
    return num_implementing_args;
fail:
    for (int j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(methods[j]);
    }
    // 释放所有引用计数，避免内存泄漏
    return -1;
}


/*
 * Is this object ndarray.__array_function__?
 */
static int
is_default_array_function(PyObject *obj)
{
    // 检查对象是否为 ndarray.__array_function__
    return obj == npy_static_pydata.ndarray_array_function;
}


/*
 * Core implementation of ndarray.__array_function__. This is exposed
 * separately so we can avoid the overhead of a Python method call from
 * within `implement_array_function`.
 */
NPY_NO_EXPORT PyObject *
array_function_method_impl(PyObject *func, PyObject *types, PyObject *args,
                           PyObject *kwargs)
{
    PyObject **items = PySequence_Fast_ITEMS(types);
    Py_ssize_t length = PySequence_Fast_GET_SIZE(types);

    // 检查每个类型是否为 PyArray_Type 的子类
    for (Py_ssize_t j = 0; j < length; j++) {
        int is_subclass = PyObject_IsSubclass(
            items[j], (PyObject *)&PyArray_Type);
        if (is_subclass == -1) {
            return NULL;
        }
        if (!is_subclass) {
            // 如果不是，则返回 Py_NotImplemented
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
    }

    // 获取 func 对象的 implementation 属性
    PyObject *implementation = PyObject_GetAttr(func, npy_interned_str.implementation);
    if (implementation == NULL) {
        return NULL;
    }
    // 调用 implementation 对象，并传递 args 和 kwargs
    PyObject *result = PyObject_Call(implementation, args, kwargs);
    Py_DECREF(implementation);
    return result;
}


/*
 * Calls __array_function__ on the provided argument, with a fast-path for
 * ndarray.
 */
static PyObject *
call_array_function(PyObject* argument, PyObject* method,
                    PyObject* public_api, PyObject* types,
                    PyObject* args, PyObject* kwargs)
{
    // 如果 method 是 ndarray 的默认 array_function，则调用 array_function_method_impl
    if (is_default_array_function(method)) {
        return array_function_method_impl(public_api, types, args, kwargs);
    }
    else {
        // 否则，调用 method 对象，并传递 argument, public_api, types, args, kwargs
        return PyObject_CallFunctionObjArgs(
            method, argument, public_api, types, args, kwargs, NULL);
    }
}



/*
 * Helper to convert from vectorcall convention, since the protocol requires
 * args and kwargs to be passed as tuple and dict explicitly.
 * We always pass a dict, so always returns it.
 */
static int
get_args_and_kwargs(
        PyObject *const *fast_args, Py_ssize_t len_args, PyObject *kwnames,
        PyObject **out_args, PyObject **out_kwargs)
{
    // 转换为 vectorcall 约定，显式传递 args 和 kwargs
    len_args = PyVectorcall_NARGS(len_args);
    PyObject *args = PyTuple_New(len_args);
    PyObject *kwargs = NULL;

    if (args == NULL) {
        return -1;
    }
    // 将 fast_args 转移到 args 元组中
    for (Py_ssize_t i = 0; i < len_args; i++) {
        Py_INCREF(fast_args[i]);
        PyTuple_SET_ITEM(args, i, fast_args[i]);
    }
    // 创建一个空的 kwargs 字典
    kwargs = PyDict_New();
    if (kwargs == NULL) {
        Py_DECREF(args);
        return -1;
    }
    # 如果关键字参数名列表不为空
    if (kwnames != NULL) {
        # 获取关键字参数名列表的长度
        Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
        # 遍历关键字参数名列表
        for (Py_ssize_t i = 0; i < nkwargs; i++) {
            # 获取第 i 个关键字参数名
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            # 获取对应的快速参数中的值（即可变参数中的值）
            PyObject *value = fast_args[i+len_args];
            # 将关键字参数名和对应值设置到 kwargs 字典中
            if (PyDict_SetItem(kwargs, key, value) < 0) {
                # 如果设置失败，释放内存并返回错误
                Py_DECREF(args);
                Py_DECREF(kwargs);
                return -1;
            }
        }
    }
    # 将构建好的 args 和 kwargs 分别传递出去
    *out_args = args;
    *out_kwargs = kwargs;
    # 返回成功状态
    return 0;
    /* 设置没有匹配类型的错误，抛出 TypeError 异常 */
    npy_cache_import("numpy._core._internal",
                     "array_function_errmsg_formatter",
                     &npy_thread_unsafe_state.array_function_errmsg_formatter);
    if (npy_thread_unsafe_state.array_function_errmsg_formatter != NULL) {
        // 调用 array_function_errmsg_formatter 函数，生成错误信息对象
        PyObject *errmsg = PyObject_CallFunctionObjArgs(
                npy_thread_unsafe_state.array_function_errmsg_formatter,
                public_api, types, NULL);
        if (errmsg != NULL) {
            // 将错误信息设置为 TypeError 异常的对象
            PyErr_SetObject(PyExc_TypeError, errmsg);
            Py_DECREF(errmsg);
        }
    }
}

/*
 * 实现了 __array_function__ 协议，用于 C 数组创建函数。
 * 在 NEP-18 的基础上添加，旨在以最小的分发开销实现 NEP-35。
 *
 * 调用者必须确保 `like != Py_None` 或 `like == NULL`。
 */
NPY_NO_EXPORT PyObject *
array_implement_c_array_function_creation(
    const char *function_name, PyObject *like,
    PyObject *args, PyObject *kwargs,
    PyObject *const *fast_args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *dispatch_types = NULL;
    PyObject *numpy_module = NULL;
    PyObject *public_api = NULL;
    PyObject *result = NULL;

    /* 如果 `like` 没有实现 `__array_function__`，抛出 `TypeError` */
    PyObject *method = get_array_function(like);
    if (method == NULL) {
        return PyErr_Format(PyExc_TypeError,
                "The `like` argument must be an array-like that "
                "implements the `__array_function__` protocol.");
    }
    if (is_default_array_function(method)) {
        /*
         * 返回 Py_NotImplemented 的借用引用，以将处理权返回给原始函数。
         */
        Py_DECREF(method);
        return Py_NotImplemented;
    }

    /* 需要为 `__array_function__` 准备 args 和 kwargs（在不使用时）。 */
    if (fast_args != NULL) {
        assert(args == NULL);
        assert(kwargs == NULL);
        if (get_args_and_kwargs(
                fast_args, len_args, kwnames, &args, &kwargs) < 0) {
            goto finish;
        }
    }
    else {
        Py_INCREF(args);
        Py_INCREF(kwargs);
    }

    dispatch_types = PyTuple_Pack(1, Py_TYPE(like));
    if (dispatch_types == NULL) {
        goto finish;
    }

    /* 在关键字参数中必须包含 like 参数，移除它 */
    if (PyDict_DelItem(kwargs, npy_interned_str.like) < 0) {
        goto finish;
    }

    /* 获取实际的符号（目前的长途方法） */
    numpy_module = PyImport_Import(npy_interned_str.numpy);
    if (numpy_module == NULL) {
        goto finish;
    }

    /* 获取 numpy 模块中的函数名对应的公共 API */
    public_api = PyObject_GetAttrString(numpy_module, function_name);
    Py_DECREF(numpy_module);
    if (public_api == NULL) {
        goto finish;
    }
    # 检查 public_api 是否为可调用对象，如果不是则抛出运行时错误
    if (!PyCallable_Check(public_api)) {
        # 格式化错误信息，指出 numpy.function_name 不可调用
        PyErr_Format(PyExc_RuntimeError,
                "numpy.%s is not callable.", function_name);
        # 跳转到完成处理的标签位置
        goto finish;
    }

    # 调用 call_array_function 函数，传递 like, method, public_api, dispatch_types, args, kwargs 参数
    result = call_array_function(like, method,
            public_api, dispatch_types, args, kwargs);

    # 如果 call_array_function 返回 Py_NotImplemented，应该不会发生，但如果发生则处理
    if (result == Py_NotImplemented) {
        /* 这实际上不应该发生，因为只有一种类型，但是... */
        # 释放 result 对象的引用计数
        Py_DECREF(result);
        # 将 result 置为 NULL
        result = NULL;
        # 设置没有匹配类型错误，传递 public_api 和 dispatch_types 参数
        set_no_matching_types_error(public_api, dispatch_types);
    }

  finish:
    # 释放 method 对象的引用计数
    Py_DECREF(method);
    # 释放 args 对象的引用计数，如果为空则无影响
    Py_XDECREF(args);
    # 释放 kwargs 对象的引用计数，如果为空则无影响
    Py_XDECREF(kwargs);
    # 释放 dispatch_types 对象的引用计数，如果为空则无影响
    Py_XDECREF(dispatch_types);
    # 释放 public_api 对象的引用计数，如果为空则无影响
    Py_XDECREF(public_api);
    # 返回 result 对象
    return result;
/*
 * Python wrapper for get_implementing_args_and_methods, for testing purposes.
 */
# Python函数array__get_implementing_args的C扩展实现
NPY_NO_EXPORT PyObject *
array__get_implementing_args(
    PyObject *NPY_UNUSED(dummy), PyObject *positional_args)
{
    PyObject *relevant_args;    // 相关参数对象
    PyObject *implementing_args[NPY_MAXARGS];   // 实现参数数组
    PyObject *array_function_methods[NPY_MAXARGS];   // 数组函数方法数组
    PyObject *result = NULL;    // 结果对象，默认为NULL

    // 解析传入的参数元组，获取relevant_args对象
    if (!PyArg_ParseTuple(positional_args, "O:array__get_implementing_args",
                          &relevant_args)) {
        return NULL;
    }

    // 快速转换relevant_args为一个Python序列，用于迭代
    relevant_args = PySequence_Fast(
        relevant_args,
        "dispatcher for __array_function__ did not return an iterable");
    if (relevant_args == NULL) {
        return NULL;
    }

    // 获取相关参数和数组函数方法
    int num_implementing_args = get_implementing_args_and_methods(
        relevant_args, implementing_args, array_function_methods);
    if (num_implementing_args == -1) {
        goto cleanup;
    }

    /* create a Python object for implementing_args */
    // 创建一个Python列表对象，用于存放实现参数
    result = PyList_New(num_implementing_args);
    if (result == NULL) {
        goto cleanup;
    }
    // 将实现参数复制到结果列表中
    for (int j = 0; j < num_implementing_args; j++) {
        PyObject *argument = implementing_args[j];
        Py_INCREF(argument);
        PyList_SET_ITEM(result, j, argument);
    }

cleanup:
    // 清理工作：减少实现参数和数组函数方法的引用计数，释放relevant_args
    for (int j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(array_function_methods[j]);
    }
    Py_DECREF(relevant_args);
    return result;   // 返回结果对象
}


typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;  // 向量调用函数
    PyObject *dict; // 字典对象
    PyObject *relevant_arg_func;    // 相关参数函数
    PyObject *default_impl; // 默认实现
    /* The following fields are used to clean up TypeError messages only: */
    PyObject *dispatcher_name;  // 分发器名称
    PyObject *public_name;  // 公共名称
} PyArray_ArrayFunctionDispatcherObject;


static void
dispatcher_dealloc(PyArray_ArrayFunctionDispatcherObject *self)
{
    // 释放PyArray_ArrayFunctionDispatcherObject对象
    Py_CLEAR(self->relevant_arg_func);
    Py_CLEAR(self->default_impl);
    Py_CLEAR(self->dict);
    Py_CLEAR(self->dispatcher_name);
    Py_CLEAR(self->public_name);
    PyObject_FREE(self);    // 释放内存
}


static void
fix_name_if_typeerror(PyArray_ArrayFunctionDispatcherObject *self)
{
    // 如果当前异常不是TypeError，则直接返回
    if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
        return;
    }

    PyObject *exc, *val, *tb, *message; // 异常对象、值、回溯和消息对象
    PyErr_Fetch(&exc, &val, &tb);   // 获取当前异常信息
    if (!PyUnicode_CheckExact(val)) {
        /*
         * 如果 val 不是 PyUnicode 对象，我们期望错误未规范化，
         * 但可能并非总是如此，因此如果它不是字符串，则规范化并获取 args[0]。
         */
        PyErr_NormalizeException(&exc, &val, &tb);

        // 获取异常对象的 "args" 属性，应该返回一个元组
        PyObject *args = PyObject_GetAttrString(val, "args");
        if (args == NULL || !PyTuple_CheckExact(args)
                || PyTuple_GET_SIZE(args) != 1) {
            Py_XDECREF(args);
            // 如果获取失败或者 args 不是一个单元素元组，则恢复原始错误状态
            goto restore_error;
        }
        // 获取元组 args 中的第一个元素作为错误消息
        message = PyTuple_GET_ITEM(args, 0);
        Py_INCREF(message);
        Py_DECREF(args);
        // 如果消息不是字符串类型，则恢复原始错误状态
        if (!PyUnicode_CheckExact(message)) {
            Py_DECREF(message);
            goto restore_error;
        }
    }
    else {
        // 如果 val 是 PyUnicode 对象，则直接增加其引用计数
        Py_INCREF(val);
        message = val;
    }

    // 比较 message 和 self->dispatcher_name 的尾部，期望匹配
    Py_ssize_t cmp = PyUnicode_Tailmatch(
            message, self->dispatcher_name, 0, -1, -1);
    // 如果不匹配或者出错，则恢复原始错误状态
    if (cmp <= 0) {
        Py_DECREF(message);
        goto restore_error;
    }
    // 将 message 中的 self->dispatcher_name 替换为 self->public_name
    Py_SETREF(message, PyUnicode_Replace(
            message, self->dispatcher_name, self->public_name, 1));
    // 如果替换失败，则恢复原始错误状态
    if (message == NULL) {
        goto restore_error;
    }
    // 设置异常类型为 TypeError，消息为 message
    PyErr_SetObject(PyExc_TypeError, message);
    Py_DECREF(exc);
    Py_XDECREF(val);
    Py_XDECREF(tb);
    Py_DECREF(message);
    return;

  restore_error:
    /* 替换未成功，因此恢复原始错误 */
    PyErr_Restore(exc, val, tb);
    # 定义 Python C 扩展模块中的一个函数，用于分派调用向量化数组函数的操作
    static PyObject *
    dispatcher_vectorcall(PyArray_ArrayFunctionDispatcherObject *self,
            PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
    {
        PyObject *result = NULL;
        PyObject *types = NULL;
        PyObject *relevant_args = NULL;

        PyObject *public_api;

        # 如果定义了 relevant_arg_func 函数，则使用 self 作为公共 API
        if (self->relevant_arg_func != NULL) {
            public_api = (PyObject *)self;

            # 调用 relevant_arg_func 函数，获取返回的参数列表
            relevant_args = PyObject_Vectorcall(
                    self->relevant_arg_func, args, len_args, kwnames);
            if (relevant_args == NULL) {
                # 如果获取参数列表失败，修复错误名称并返回空值
                fix_name_if_typeerror(self);
                return NULL;
            }
            # 将 relevant_args 转换为快速序列对象
            Py_SETREF(relevant_args, PySequence_Fast(relevant_args,
                    "dispatcher for __array_function__ did not return an iterable"));
            if (relevant_args == NULL) {
                return NULL;
            }

            # 获取实现参数和方法，并存储到相应数组中
            num_implementing_args = get_implementing_args_and_methods(
                    relevant_args, implementing_args, array_function_methods);
            if (num_implementing_args < 0) {
                Py_DECREF(relevant_args);
                return NULL;
            }
        }
        else {
            # 对于像 from Python 中的 like= 分派，公共符号是默认实现
            public_api = self->default_impl;

            # 如果没有传入参数，报错并返回空值
            if (PyVectorcall_NARGS(len_args) == 0) {
                PyErr_Format(PyExc_TypeError,
                        "`like` argument dispatching, but first argument is not "
                        "positional in call to %S.", self->default_impl);
                return NULL;
            }

            # 获取第一个参数的数组函数方法
            array_function_methods[0] = get_array_function(args[0]);
            if (array_function_methods[0] == NULL) {
                return PyErr_Format(PyExc_TypeError,
                        "The `like` argument must be an array-like that "
                        "implements the `__array_function__` protocol.");
            }
            num_implementing_args = 1;
            implementing_args[0] = args[0];
            Py_INCREF(implementing_args[0]);

            # 不传递 like 参数
            len_args = PyVectorcall_NARGS(len_args) - 1;
            len_args |= PY_VECTORCALL_ARGUMENTS_OFFSET;
            args++;
        }

        # 处理没有覆盖情况的典型情况，检查是否有重载函数
        int any_overrides = 0;
    // 遍历实现参数数组，检查是否有非默认数组函数
    for (int j = 0; j < num_implementing_args; j++) {
        if (!is_default_array_function(array_function_methods[j])) {
            // 如果有任何一个函数不是默认的数组函数，设置标志为真并跳出循环
            any_overrides = 1;
            break;
        }
    }
    // 如果没有函数重写，默认调用实现
    if (!any_overrides) {
        /* 直接调用实际的实现函数。 */
        result = PyObject_Vectorcall(self->default_impl, args, len_args, kwnames);
        // 跳转到清理操作
        goto cleanup;
    }

    /* 找到参数和关键字参数作为元组和字典，因为我们要传递它们出去： */
    if (get_args_and_kwargs(
            args, len_args, kwnames, &packed_args, &packed_kwargs) < 0) {
        // 如果获取参数和关键字参数失败，跳转到清理操作
        goto cleanup;
    }

    /*
     * 为类型创建一个 Python 对象。
     * 我们使用元组，因为它是创建速度最快的 Python 集合，
     * 并且额外的好处是它是不可变的。
     */
    types = PyTuple_New(num_implementing_args);
    if (types == NULL) {
        // 如果创建元组失败，跳转到清理操作
        goto cleanup;
    }
    // 填充元组，将每个实现参数的类型添加到元组中
    for (int j = 0; j < num_implementing_args; j++) {
        PyObject *arg_type = (PyObject *)Py_TYPE(implementing_args[j]);
        Py_INCREF(arg_type);
        PyTuple_SET_ITEM(types, j, arg_type);
    }

    /* 调用 __array_function__ 方法 */
    for (int j = 0; j < num_implementing_args; j++) {
        PyObject *argument = implementing_args[j];
        PyObject *method = array_function_methods[j];

        result = call_array_function(
                argument, method, public_api, types,
                packed_args, packed_kwargs);

        if (result == Py_NotImplemented) {
            /* 尝试下一个方法 */
            Py_DECREF(result);
            result = NULL;
        }
        else {
            /* 获得了有效的结果，或者引发了异常。 */
            goto cleanup;
        }
    }

    // 设置没有匹配类型的错误
    set_no_matching_types_error(public_api, types);
cleanup:
    # 清理阶段，释放实现参数和数组方法数组中的引用计数
    for (int j = 0; j < num_implementing_args; j++) {
        Py_DECREF(implementing_args[j]);
        Py_DECREF(array_function_methods[j]);
    }
    // 释放打包参数和打包关键字参数的引用计数
    Py_XDECREF(packed_args);
    Py_XDECREF(packed_kwargs);
    // 释放类型对象的引用计数
    Py_XDECREF(types);
    // 释放相关参数的引用计数
    Py_XDECREF(relevant_args);
    // 返回计算结果
    return result;
}


static PyObject *
dispatcher_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwargs)
{
    PyArray_ArrayFunctionDispatcherObject *self;

    // 创建一个新的对象，分配内存并初始化为指定类型
    self = PyObject_New(
            PyArray_ArrayFunctionDispatcherObject,
            &PyArrayFunctionDispatcher_Type);
    if (self == NULL) {
        // 内存分配失败，返回内存错误异常
        return PyErr_NoMemory();
    }

    // 定义关键字参数列表
    char *kwlist[] = {"", "", NULL};
    // 解析输入参数并设置实例属性
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO:_ArrayFunctionDispatcher", kwlist,
            &self->relevant_arg_func, &self->default_impl)) {
        // 解析失败时释放对象并返回空
        Py_DECREF(self);
        return NULL;
    }

    // 设置向量调用函数
    self->vectorcall = (vectorcallfunc)dispatcher_vectorcall;
    // 增加默认实现的引用计数
    Py_INCREF(self->default_impl);
    // 初始化字典为空
    self->dict = NULL;
    self->dispatcher_name = NULL;
    self->public_name = NULL;

    // 如果相关参数函数是 Py_None，则清除引用
    if (self->relevant_arg_func == Py_None) {
        /* NULL in the relevant arg function means we use `like=` */
        Py_CLEAR(self->relevant_arg_func);
    }
    else {
        // 增加相关参数函数的引用计数，并获取其限定名称
        Py_INCREF(self->relevant_arg_func);
        self->dispatcher_name = PyObject_GetAttrString(
            self->relevant_arg_func, "__qualname__");
        if (self->dispatcher_name == NULL) {
            // 获取失败时释放对象并返回空
            Py_DECREF(self);
            return NULL;
        }
        // 获取默认实现的限定名称
        self->public_name = PyObject_GetAttrString(
            self->default_impl, "__qualname__");
        if (self->public_name == NULL) {
            // 获取失败时释放对象并返回空
            Py_DECREF(self);
            return NULL;
        }
    }

    // 创建新的空字典对象
    self->dict = PyDict_New();
    if (self->dict == NULL) {
        // 字典创建失败时释放对象并返回空
        Py_DECREF(self);
        return NULL;
    }
    // 返回创建的对象实例
    return (PyObject *)self;
}


static PyObject *
dispatcher_str(PyArray_ArrayFunctionDispatcherObject *self)
{
    // 返回默认实现对象的字符串表示形式
    return PyObject_Str(self->default_impl);
}


static PyObject *
dispatcher_repr(PyObject *self)
{
    // 获取对象的名称属性
    PyObject *name = PyObject_GetAttrString(self, "__name__");
    if (name == NULL) {
        // 获取失败时返回空
        return NULL;
    }
    // 格式化字符串表示为函数形式
    return PyUnicode_FromFormat("<function %S at %p>", name, self);
}


static PyObject *
func_dispatcher___get__(PyObject *self, PyObject *obj, PyObject *cls)
{
    if (obj == NULL) {
        // 如果对象为空，表现为静态方法，无需绑定，增加自身引用计数并返回自身
        Py_INCREF(self);
        return self;
    }
    // 创建一个新的方法对象并返回
    return PyMethod_New(self, obj);
}


static PyObject *
dispatcher_get_implementation(
        PyArray_ArrayFunctionDispatcherObject *self, void *NPY_UNUSED(closure))
{
    // 增加默认实现对象的引用计数并返回
    Py_INCREF(self->default_impl);
    return self->default_impl;
}


static PyObject *
dispatcher_reduce(PyObject *self, PyObject *NPY_UNUSED(args))
{
    // 在此处实现对象的减少协议
    // (此处需补充完整该函数的代码和注释)
}
    # 获取对象的 "__qualname__" 属性，并返回其值
    return PyObject_GetAttrString(self, "__qualname__");
# 定义一个静态的 PyMethodDef 结构体数组，用于描述函数调度器对象的方法
static struct PyMethodDef func_dispatcher_methods[] = {
    # "__reduce__" 方法，指定为 dispatcher_reduce 函数处理，不带参数，无额外信息
    {"__reduce__",
        (PyCFunction)dispatcher_reduce, METH_NOARGS, NULL},
    # 结束标志，表示方法列表的结束
    {NULL, NULL, 0, NULL}
};

# 定义一个静态的 PyGetSetDef 结构体数组，用于描述函数调度器对象的属性获取和设置
static struct PyGetSetDef func_dispatcher_getset[] = {
    # "__dict__" 属性，使用 PyObject_GenericGetDict 获取，不带额外信息
    {"__dict__", &PyObject_GenericGetDict, 0, NULL, 0},
    # "_implementation" 属性，使用 dispatcher_get_implementation 函数获取，不带额外信息
    {"_implementation", (getter)&dispatcher_get_implementation, 0, NULL, 0},
    # 结束标志，表示属性列表的结束
    {0, 0, 0, 0, 0}
};

# 定义一个非导出的 PyTypeObject 结构体，描述数组函数调度器对象的类型信息
NPY_NO_EXPORT PyTypeObject PyArrayFunctionDispatcher_Type = {
     # 初始化头部信息，没有基类，大小为 0
     PyVarObject_HEAD_INIT(NULL, 0)
     # 类型名称为 "numpy._ArrayFunctionDispatcher"
     .tp_name = "numpy._ArrayFunctionDispatcher",
     # 类型的基本大小，为 PyArray_ArrayFunctionDispatcherObject 结构体的大小
     .tp_basicsize = sizeof(PyArray_ArrayFunctionDispatcherObject),
     # tp_dictoffset 偏移量，指向对象中字典的位置，用于快速访问 __dict__ 属性
     .tp_dictoffset = offsetof(PyArray_ArrayFunctionDispatcherObject, dict),
     # 析构函数，释放对象时调用 dispatcher_dealloc 函数
     .tp_dealloc = (destructor)dispatcher_dealloc,
     # 创建新对象的函数，调用 dispatcher_new 函数
     .tp_new = (newfunc)dispatcher_new,
     # 字符串表示函数，调用 dispatcher_str 函数
     .tp_str = (reprfunc)dispatcher_str,
     # repr 表示函数，调用 dispatcher_repr 函数
     .tp_repr = (reprfunc)dispatcher_repr,
     # 类型标志，包括默认标志、支持向量调用、方法描述符
     .tp_flags = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL
                  | Py_TPFLAGS_METHOD_DESCRIPTOR),
     # 方法定义数组，指向 func_dispatcher_methods
     .tp_methods = func_dispatcher_methods,
     # 属性获取和设置数组，指向 func_dispatcher_getset
     .tp_getset = func_dispatcher_getset,
     # 描述符获取函数，调用 func_dispatcher___get__ 函数
     .tp_descr_get = func_dispatcher___get__,
     # 调用函数，使用 PyVectorcall_Call 处理
     .tp_call = &PyVectorcall_Call,
     # 向量调用偏移量，指向 vectorcall 字段的位置
     .tp_vectorcall_offset = offsetof(PyArray_ArrayFunctionDispatcherObject, vectorcall),
};
```