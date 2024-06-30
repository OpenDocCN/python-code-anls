# `D:\src\scipysrc\scipy\scipy\_lib\src\ccallback.h`

```
/*
 * ccallback
 *
 * Callback function interface, supporting
 *
 * (1) pure Python functions
 * (2) plain C functions wrapped in PyCapsules (with Cython CAPI style signatures)
 * (3) ctypes function pointers
 * (4) cffi function pointers
 *
 * This is done avoiding magic or code generation, so you need to write some
 * boilerplate code manually.
 *
 * For an example see `scipy/_lib/src/_test_ccallback.c`.
 */


#ifndef CCALLBACK_H_
#define CCALLBACK_H_


#include <Python.h>
#include <setjmp.h>

/* Default behavior */
#define CCALLBACK_DEFAULTS 0x0
/* Whether calling ccallback_obtain is enabled */
#define CCALLBACK_OBTAIN   0x1
/* Deal with also other input objects than LowLevelCallable.
 * Useful for maintaining legacy behavior.
 */
#define CCALLBACK_PARSE    0x2


typedef struct ccallback ccallback_t;
typedef struct ccallback_signature ccallback_signature_t;

struct ccallback_signature {
    /* Function signature as a Cython/cffi-like prototype string */
    char *signature;
    /* Value that can be used for any purpose */
    int value;
};

struct ccallback {
    /* Pointer to a C function to call. NULL if none. */
    void *c_function;
    /* Pointer to a Python function to call (refcount is owned). NULL if none. */
    PyObject *py_function;
    /* Additional data pointer provided by the user. */
    void *user_data;
    /* Function signature selected */
    ccallback_signature_t *signature;
    /* setjmp buffer to jump to on error */
    jmp_buf error_buf;
    /* Previous callback, for TLS reentrancy */
    ccallback_t *prev_callback;

    /* Unused variables that can be used by the thunk etc. code for any purpose */
    long info;
    void *info_p;
};


/*
 * Thread-local storage
 */

#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 4)))

static __thread ccallback_t *_active_ccallback = NULL;

/* Get the current thread's local ccallback_t structure */
static void *ccallback__get_thread_local(void)
{
    return (void *)_active_ccallback;
}

/* Set the current thread's local ccallback_t structure */
static int ccallback__set_thread_local(void *value)
{
    _active_ccallback = value;
    return 0;
}

/*
 * Obtain a pointer to the current ccallback_t structure.
 */
static ccallback_t *ccallback_obtain(void)
{
    return (ccallback_t *)ccallback__get_thread_local();
}

#elif defined(_MSC_VER)

static __declspec(thread) ccallback_t *_active_ccallback = NULL;

/* Get the current thread's local ccallback_t structure */
static void *ccallback__get_thread_local(void)
{
    return (void *)_active_ccallback;
}

/* Set the current thread's local ccallback_t structure */
static int ccallback__set_thread_local(void *value)
{
    _active_ccallback = value;
    return 0;
}

/*
 * Obtain a pointer to the current ccallback_t structure.
 */
static ccallback_t *ccallback_obtain(void)
{
    return (ccallback_t *)ccallback__get_thread_local();
}

#else

/* Fallback implementation with Python thread API */

/* Get the current thread's local ccallback_t structure using Python thread API */
static void *ccallback__get_thread_local(void)
{
    PyObject *local_dict, *capsule;
    void *callback_ptr;

    local_dict = PyThreadState_GetDict();
    if (local_dict == NULL) {
        /* Fatal error if failed to get local thread state */
        Py_FatalError("scipy/ccallback: failed to get local thread state");
    // 从本地字典中获取名称为 "__scipy_ccallback" 的项，并赋值给变量 capsule
    capsule = PyDict_GetItemString(local_dict, "__scipy_ccallback");
    // 如果 capsule 为 NULL，返回 NULL
    if (capsule == NULL) {
        return NULL;
    }

    // 从 capsule 中获取指针，并赋值给 callback_ptr
    callback_ptr = PyCapsule_GetPointer(capsule, NULL);
    // 如果 callback_ptr 为 NULL，则发生致命错误并打印错误信息
    if (callback_ptr == NULL) {
        Py_FatalError("scipy/ccallback: invalid callback state");
    }

    // 返回 callback_ptr，即回调函数的指针
    return callback_ptr;
}

static int ccallback__set_thread_local(void *value)
{
    // 获取当前线程状态的局部字典
    PyObject *local_dict;
    local_dict = PyThreadState_GetDict();
    if (local_dict == NULL) {
        // 如果获取失败，致命错误
        Py_FatalError("scipy/ccallback: failed to get local thread state");
    }

    if (value == NULL) {
        // 如果值为NULL，尝试从局部字典中删除特定键名
        return PyDict_DelItemString(local_dict, "__scipy_ccallback");
    }
    else {
        // 如果值非NULL，创建一个PyCapsule对象并将其存储到局部字典中
        PyObject *capsule;
        int ret;

        capsule = PyCapsule_New(value, NULL, NULL);
        if (capsule == NULL) {
            return -1;
        }
        ret = PyDict_SetItemString(local_dict, "__scipy_ccallback", capsule);
        Py_DECREF(capsule);
        return ret;
    }
}

/*
 * Obtain a pointer to the current ccallback_t structure.
 */
static ccallback_t *ccallback_obtain(void)
{
    // 获取全局解释器锁状态
    PyGILState_STATE state;
    ccallback_t *callback_ptr;

    state = PyGILState_Ensure();

    // 获取当前线程的ccallback_t结构指针
    callback_ptr = (ccallback_t *)ccallback__get_thread_local();
    if (callback_ptr == NULL) {
        // 如果获取失败，致命错误
        Py_FatalError("scipy/ccallback: failed to get thread local state");
    }

    // 释放全局解释器锁状态
    PyGILState_Release(state);

    return callback_ptr;
}

#endif


/*
 * Set Python error status indicating a signature mismatch.
 *
 * Parameters
 * ----------
 * signatures
 *     NULL terminated list of allowed signatures.
 * capsule_signature
 *     The mismatcing signature from user-provided PyCapsule.
 */
static void ccallback__err_invalid_signature(ccallback_signature_t *signatures,
                                             const char *capsule_signature)
{
    // 创建一个空的Python列表对象
    PyObject *sig_list = NULL;
    ccallback_signature_t *sig;

    sig_list = PyList_New(0);
    if (sig_list == NULL) {
        return;
    }

    if (capsule_signature == NULL) {
        capsule_signature = "NULL";
    }

    // 遍历允许的签名列表，将签名字符串添加到sig_list中
    for (sig = signatures; sig->signature != NULL; ++sig) {
        PyObject *str;
        int ret;

        str = PyUnicode_FromString(sig->signature);
        if (str == NULL) {
            goto fail;
        }

        ret = PyList_Append(sig_list, str);
        Py_DECREF(str);
        if (ret == -1) {
            goto fail;
        }
    }

    // 设置Python异常信息，指示签名不匹配的错误
    PyErr_Format(PyExc_ValueError,
                 "Invalid scipy.LowLevelCallable signature \"%s\". Expected one of: %R",
                 capsule_signature, sig_list);

fail:
    Py_XDECREF(sig_list);
    return;
}


/*
 * Set up callback.
 *
 * Parameters
 * ----------
 * callback : ccallback_t
 *     Callback structure to initialize.
 * signatures : ccallback_signature_t *
 *     Pointer to a NULL-terminated array of C function signatures.
 *     The list of signatures should always contain a signature defined in
 *     terms of C basic data types only.
 * callback_obj : PyObject
 *     Object provided by the user. Usually, LowLevelCallback object, or a
 *     Python callable.
 * flags : int
 *     Bitmask of CCALLBACK_* flags.
 *
 * Returns
 * -------
 * success : int
 *     0 if success, != 0 on failure (an appropriate Python exception is set).
 *
 */
static int ccallback_prepare(ccallback_t *callback, ccallback_signature_t *signatures,
                             PyObject *callback_obj, int flags)
{
    static PyTypeObject *lowlevelcallable_type = NULL;
    PyObject *callback_obj2 = NULL;
    PyObject *capsule = NULL;

    // 检查是否首次调用，若是，则导入必要的模块和类型
    if (lowlevelcallable_type == NULL) {
        PyObject *module;

        // 尝试导入 "scipy._lib._ccallback" 模块
        module = PyImport_ImportModule("scipy._lib._ccallback");
        if (module == NULL) {
            // 导入失败，跳转到错误处理
            goto error;
        }

        // 获取 LowLevelCallable 类型对象
        lowlevelcallable_type = (PyTypeObject *)PyObject_GetAttrString(module, "LowLevelCallable");
        Py_DECREF(module);
        if (lowlevelcallable_type == NULL) {
            // 获取类型对象失败，跳转到错误处理
            goto error;
        }
    }

    // 如果需要解析回调函数，并且传入的对象不是 LowLevelCallable 类型，则进行解析
    if ((flags & CCALLBACK_PARSE) && !PyObject_TypeCheck(callback_obj, lowlevelcallable_type)) {
        /* 解析回调函数 */
        callback_obj2 = PyObject_CallMethod((PyObject *)lowlevelcallable_type,
                                            "_parse_callback", "O", callback_obj);
        if (callback_obj2 == NULL) {
            // 解析失败，跳转到错误处理
            goto error;
        }

        // 使用解析后的回调对象继续操作
        callback_obj = callback_obj2;

        // 如果回调对象是 PyCapsule 类型，则直接使用它作为 capsule
        if (PyCapsule_CheckExact(callback_obj)) {
            capsule = callback_obj;
        }
    }

    // 如果回调对象是可调用的 Python 函数
    if (PyCallable_Check(callback_obj)) {
        /* Python 可调用函数 */
        callback->py_function = callback_obj;
        Py_INCREF(callback->py_function);
        callback->c_function = NULL;
        callback->user_data = NULL;
        callback->signature = NULL;
    }
    // 否则，如果是 PyCapsule 或者 LowLevelCallable 类型中的 PyCapsule
    else if (capsule != NULL ||
             (PyObject_TypeCheck(callback_obj, lowlevelcallable_type) &&
              PyCapsule_CheckExact(PyTuple_GET_ITEM(callback_obj, 0)))) {
        /* LowLevelCallable 中的 PyCapsule（或上述解析结果） */
        void *ptr, *user_data;
        ccallback_signature_t *sig;
        const char *name;

        // 如果 capsule 为 NULL，则从 callback_obj 中获取 PyCapsule
        if (capsule == NULL) {
            capsule = PyTuple_GET_ITEM(callback_obj, 0);
        }

        // 获取 PyCapsule 的名称
        name = PyCapsule_GetName(capsule);
        if (PyErr_Occurred()) {
            // 获取名称出错，跳转到错误处理
            goto error;
        }

        // 在签名列表中查找匹配的签名
        for (sig = signatures; sig->signature != NULL; ++sig) {
            if (name && strcmp(name, sig->signature) == 0) {
                break;
            }
        }

        // 如果未找到匹配的签名，报告无效签名错误
        if (sig->signature == NULL) {
            ccallback__err_invalid_signature(signatures, name);
            goto error;
        }

        // 获取 PyCapsule 的指针和上下文数据
        ptr = PyCapsule_GetPointer(capsule, sig->signature);
        if (ptr == NULL) {
            // 获取指针失败，报告错误
            PyErr_SetString(PyExc_ValueError, "PyCapsule_GetPointer failed");
            goto error;
        }

        user_data = PyCapsule_GetContext(capsule);
        if (PyErr_Occurred()) {
            // 获取上下文数据失败，跳转到错误处理
            goto error;
        }

        // 设置回调结构体的相关字段
        callback->py_function = NULL;
        callback->c_function = ptr;
        callback->user_data = user_data;
        callback->signature = sig;
    }
    else {
        // 其他情况下，传入的回调对象无效，报告错误
        PyErr_SetString(PyExc_ValueError, "invalid callable given");
        goto error;
    }
    # 如果 flags 中包含 CCALLBACK_OBTAIN 标志位
    if (flags & CCALLBACK_OBTAIN) {
        # 将当前线程的先前回调保存到当前回调的上一个回调中
        callback->prev_callback = ccallback__get_thread_local();
        # 尝试将当前回调设置为线程局部变量，若失败则跳转到错误处理部分
        if (ccallback__set_thread_local((void *)callback) != 0) {
            goto error;
        }
    }
    else {
        # 如果 flags 中不包含 CCALLBACK_OBTAIN 标志位，则将 prev_callback 设置为 NULL
        callback->prev_callback = NULL;
    }

    # 释放 callback_obj2 所指向的 Python 对象，处理它的引用计数
    Py_XDECREF(callback_obj2);
    # 返回成功状态码 0
    return 0;
/*
 * Decrease the reference count of callback_obj2 and return -1 to indicate error.
 */
Py_XDECREF(callback_obj2);
return -1;
}


/*
 * Tear down callback.
 *
 * Parameters
 * ----------
 * callback : ccallback_t
 *     A callback structure, previously initialized by ccallback_prepare
 *
 */
static int ccallback_release(ccallback_t *callback)
{
    /*
     * Decrease the reference count of the Python function object pointed to by callback->py_function.
     * Set both c_function and py_function pointers in callback structure to NULL.
     */
    Py_XDECREF(callback->py_function);
    callback->c_function = NULL;
    callback->py_function = NULL;

    /*
     * If there exists a previous callback in the chain (callback->prev_callback is not NULL),
     * set the thread-local state to that of the previous callback.
     * Return -1 if setting the thread-local state fails.
     */
    if (callback->prev_callback != NULL) {
        if (ccallback__set_thread_local(callback->prev_callback) != 0) {
            return -1;
        }
    }
    callback->prev_callback = NULL;

    /*
     * Return 0 to indicate successful teardown of the callback.
     */
    return 0;
}

#endif /* CCALLBACK_H_ */
```