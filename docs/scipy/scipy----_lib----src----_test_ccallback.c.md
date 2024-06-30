# `D:\src\scipysrc\scipy\scipy\_lib\src\_test_ccallback.c`

```
/*
 * Test code for ccallback.h
 *
 * This also is an internal "best-practices" code example on how to write
 * low-level callback code. (In the examples below, it is assumed the semantics
 * and signatures of the callbacks in test_call_* are fixed by some 3rd party
 * library e.g., implemented in FORTRAN, and they are not necessarily the optimal
 * way.)
 *
 * The general structure of callbacks is the following:
 *
 * - entry point function, callable from Python, calls 3rd party library code
 *   (test_call_*)
 * - 3rd party library code, calls the callback (included as a part of test_call_*)
 * - callback thunk, has the signature expected by the 3rd party library, and
 *   translates the callback call to a Python function or user-provided
 *   low-level function call (thunk_*).
 *
 * The *thunk_simple* function shows how to write a callback thunk that
 * dispatches to user-provided code (written in Python, C, Cython etc.).
 *
 * The *call_simple* function shows how to setup and teardown the ccallback_t
 * data structure in the entry point.
 *
 * The *call_nodata* and *thunk_nodata* show what to do when you need a
 * callback function for some code where it's not possible to pass along a
 * custom data pointer.
 *
 * The *call_nonlocal* and *thunk_nonlocal* show how to use setjmp/longjmp
 * to obtain a nonlocal return on error conditions, in cases where there's no
 * mechanism to interrupt computation. Note that this is the last-resort option,
 * and only safe if there is no memory allocation between setjmp/longjmp (or you
 * need to add additional cleanup yourself).
 *
 */

#include <Python.h>
#include <setjmp.h>

#include "ccallback.h"


#define ERROR_VALUE 2


/*
 * Example 3rd party library code, to be interfaced with
 */

static double library_call_simple(double value, int *error_flag, double (*callback)(double, int*, void*),
                                  void *data)
{
    *error_flag = 0;
    return callback(value, error_flag, data);
}


static double library_call_nodata(double value, int *error_flag, double (*callback)(double, int*))
{
    *error_flag = 0;
    return callback(value, error_flag);
}


static double library_call_nonlocal(double value, double (*callback)(double))
{
    return callback(value);
}



/*
 * Callback thunks for the different cases
 */

static double test_thunk_simple(double a, int *error_flag, void *data)
{
    // 将数据指针解释为 ccallback_t 类型
    ccallback_t *callback = (ccallback_t*)data;
    // 定义结果和错误标志
    double result = 0;
    int error = 0;
    # 检查回调函数是否存在
    if (callback->py_function) {
        # 在调用Python函数之前，确保获取全局解释器锁
        PyGILState_STATE state = PyGILState_Ensure();
        PyObject *res, *res2;

        # 调用Python回调函数，传递参数a作为浮点数
        res = PyObject_CallFunction(callback->py_function, "d", a);

        # 检查函数调用是否成功
        if (res == NULL) {
            error = 1;  # 如果函数调用失败，设置错误标志
        }
        else {
            # 尝试将返回值转换为Python浮点数对象
            res2 = PyNumber_Float(res);
            if (res2 == NULL) {
                error = 1;  # 如果转换失败，设置错误标志
            }
            else {
                # 将Python浮点数对象转换为C的double类型
                result = PyFloat_AsDouble(res2);
                # 检查转换过程中是否有Python异常
                if (PyErr_Occurred()) {
                    error = 1;  # 如果有异常，设置错误标志
                }
                Py_DECREF(res2);  # 释放Python浮点数对象的引用
            }
            Py_DECREF(res);  # 释放Python函数调用返回的对象引用
        }

        # 释放全局解释器锁
        PyGILState_Release(state);
    }
    else {
        # 如果没有Python回调函数，根据函数签名调用对应的C函数
        if (callback->signature->value == 0) {
            # 调用C函数，传递参数a，返回结果存储在result中
            result = ((double(*)(double, int *, void *))callback->c_function)(
                a, &error, callback->user_data);
        }
        else {
            # 调用C函数，传递参数a和0.0，返回结果存储在result中
            result = ((double(*)(double, double, int *, void *))callback->c_function)(
                a, 0.0, &error, callback->user_data);
        }
    }

    # 如果在以上过程中有任何错误，设置错误标志
    if (error) {
        *error_flag = 1;
    }

    # 返回计算结果，无论是来自Python回调函数还是C函数
    return result;
}


static double test_thunk_nodata(double a, int *error_flag)
{
    // 获取回调对象
    ccallback_t *callback = ccallback_obtain();
    // 调用简单的 thunk 函数，并返回结果
    return test_thunk_simple(a, error_flag, (void *)callback);
}


static double test_thunk_nonlocal(double a)
{
    // 获取回调对象
    ccallback_t *callback = ccallback_obtain();
    double result;
    int error_flag = 0;

    // 调用简单的 thunk 函数，处理错误标志并返回结果
    result = test_thunk_simple(a, &error_flag, (void *)callback);

    // 如果有错误标志，跳转到错误处理代码
    if (error_flag) {
        longjmp(callback->error_buf, 1);
    }

    // 返回结果
    return result;
}


/*
 * Caller entry point functions
 */

static ccallback_signature_t signatures[] = {
    {"double (double, int *, void *)", 0},
    {"double (double, double, int *, void *)", 1},
#if NPY_SIZEOF_INT == NPY_SIZEOF_SHORT
    {"double (double, short *, void *)", 0},
    {"double (double, double, short *, void *)", 1},
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    {"double (double, long *, void *)", 0},
    {"double (double, double, long *, void *)", 1},
#endif
    {NULL}
};

static PyObject *test_call_simple(PyObject *obj, PyObject *args)
{
    PyObject *callback_obj;
    double value, result;
    ccallback_t callback;
    int error_flag;
    int ret;

    // 解析 Python 元组参数
    if (!PyArg_ParseTuple(args, "Od", &callback_obj, &value)) {
        return NULL;
    }

    // 准备回调函数并检查返回状态
    ret = ccallback_prepare(&callback, signatures, callback_obj, CCALLBACK_DEFAULTS);
    if (ret != 0) {
        return NULL;
    }

    /* 调用第三方库代码 */
    Py_BEGIN_ALLOW_THREADS
    result = library_call_simple(value, &error_flag, test_thunk_simple, (void *)&callback);
    Py_END_ALLOW_THREADS

    // 释放回调资源
    ccallback_release(&callback);

    // 如果有错误标志，返回 NULL；否则返回浮点数对象
    if (error_flag) {
        return NULL;
    }
    else {
        return PyFloat_FromDouble(result);
    }
}


static PyObject *test_call_nodata(PyObject *obj, PyObject *args)
{
    PyObject *callback_obj;
    double value, result;
    ccallback_t callback;
    int ret;
    int error_flag;

    // 解析 Python 元组参数
    if (!PyArg_ParseTuple(args, "Od", &callback_obj, &value)) {
        return NULL;
    }

    // 准备回调函数并检查返回状态
    ret = ccallback_prepare(&callback, signatures, callback_obj, CCALLBACK_OBTAIN);
    if (ret != 0) {
        return NULL;
    }

    /* 调用第三方库代码 */
    Py_BEGIN_ALLOW_THREADS
    result = library_call_nodata(value, &error_flag, test_thunk_nodata);
    Py_END_ALLOW_THREADS

    // 释放回调资源
    ccallback_release(&callback);

    // 如果有错误标志，返回 NULL；否则返回浮点数对象
    if (error_flag) {
        return NULL;
    }
    else {
        return PyFloat_FromDouble(result);
    }
}


static PyObject *test_call_nonlocal(PyObject *obj, PyObject *args)
{
    PyObject *callback_obj;
    double value, result;
    int ret;
    ccallback_t callback;
    PyThreadState *_save = NULL;

    // 解析 Python 元组参数
    if (!PyArg_ParseTuple(args, "Od", &callback_obj, &value)) {
        return NULL;
    }

    // 准备回调函数并检查返回状态
    ret = ccallback_prepare(&callback, signatures, callback_obj, CCALLBACK_OBTAIN);
    if (ret != 0) {
        /* 立即返回错误 */
        return NULL;
    }

    /* 非本地返回 */
    _save = PyEval_SaveThread();
    if (setjmp(callback.error_buf) != 0) {
        /* 设置长跳转点，用于非局部错误返回 */
        PyEval_RestoreThread(_save);
        // 恢复 Python 线程状态
        ccallback_release(&callback);
        // 释放 C 回调函数资源
        return NULL;
    }

    /* 调用第三方库代码 */
    result = library_call_nonlocal(value, test_thunk_nonlocal);

    // 恢复 Python 线程状态
    PyEval_RestoreThread(_save);

    // 释放 C 回调函数资源
    ccallback_release(&callback);

    // 返回 Python 浮点数对象，表示结果
    return PyFloat_FromDouble(result);
}

/*
 * Functions for testing the PyCapsule interface
 */

// test_plus1_signature defines the signature of the test_plus1_callback function
static char *test_plus1_signature = "double (double, int *, void *)";

// test_plus1_callback function definition
static double test_plus1_callback(double a, int *error_flag, void *user_data)
{
    // Check if 'a' is equal to ERROR_VALUE
    if (a == ERROR_VALUE) {
        // Ensure the Python Global Interpreter Lock (GIL) state
        PyGILState_STATE state = PyGILState_Ensure();
        // Set the error flag to 1
        *error_flag = 1;
        // Set a ValueError exception with a specific message
        PyErr_SetString(PyExc_ValueError, "ERROR_VALUE encountered!");
        // Release the Python GIL state
        PyGILState_Release(state);
        return 0;
    }

    // Check if user_data is NULL, return a + 1 if true
    if (user_data == NULL) {
        return a + 1;
    }
    // Otherwise, return a + *(double *)user_data
    else {
        return a + *(double *)user_data;
    }
}

// Function to create a PyCapsule for test_plus1_callback
static PyObject *test_get_plus1_capsule(PyObject *obj, PyObject *args)
{
    // Parse the input arguments tuple; if parsing fails, return NULL
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    // Create and return a PyCapsule encapsulating test_plus1_callback
    return PyCapsule_New((void *)test_plus1_callback, test_plus1_signature, NULL);
}

// test_plus1b_signature defines the signature of the test_plus1b_callback function
static char *test_plus1b_signature = "double (double, double, int *, void *)";

// test_plus1b_callback function definition
static double test_plus1b_callback(double a, double b, int *error_flag, void *user_data)
{
    // Call test_plus1_callback with arguments a, error_flag, user_data, and add b
    return test_plus1_callback(a, error_flag, user_data) + b;
}

// Function to create a PyCapsule for test_plus1b_callback
static PyObject *test_get_plus1b_capsule(PyObject *obj, PyObject *args)
{
    // Parse the input arguments tuple; if parsing fails, return NULL
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    // Create and return a PyCapsule encapsulating test_plus1b_callback
    return PyCapsule_New((void *)test_plus1b_callback, test_plus1b_signature, NULL);
}

// test_plus1bc_signature defines the signature of the test_plus1bc_callback function
static char *test_plus1bc_signature = "double (double, double, double, int *, void *)";

// test_plus1bc_callback function definition
static double test_plus1bc_callback(double a, double b, double c, int *error_flag, void *user_data)
{
    // Call test_plus1_callback with arguments a, error_flag, user_data, and add b and c
    return test_plus1_callback(a, error_flag, user_data) + b + c;
}

// Function to create a PyCapsule for test_plus1bc_callback
static PyObject *test_get_plus1bc_capsule(PyObject *obj, PyObject *args)
{
    // Parse the input arguments tuple; if parsing fails, return NULL
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    // Create and return a PyCapsule encapsulating test_plus1bc_callback
    return PyCapsule_New((void *)test_plus1bc_callback, test_plus1bc_signature, NULL);
}

// Function to free memory allocated for a capsule's data
static void data_capsule_destructor(PyObject *capsule)
{
    // Retrieve the pointer stored in the capsule
    void *data;
    data = PyCapsule_GetPointer(capsule, NULL);
    // Free the allocated memory
    free(data);
}

// Function to create a PyCapsule encapsulating allocated data
static PyObject *test_get_data_capsule(PyObject *obj, PyObject *args)
{
    // Declare a pointer for data
    double *data;

    // Parse the input arguments tuple; if parsing fails, return NULL
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }

    // Allocate memory for data
    data = (double *)malloc(sizeof(double));
    // Check if memory allocation failed; if so, return memory error exception
    if (data == NULL) {
        return PyErr_NoMemory();
    }

    // Assign a value to the allocated data
    *data = 2.0;

    // Create and return a PyCapsule encapsulating data with a custom destructor
    return PyCapsule_New((void *)data, NULL, data_capsule_destructor);
}

/*
 * Initialize the module
 */

// Array of method definitions for the Python module
static PyMethodDef test_ccallback_methods[] = {
    {"test_call_simple", (PyCFunction)test_call_simple, METH_VARARGS, ""},
    {"test_call_nodata", (PyCFunction)test_call_nodata, METH_VARARGS, ""},
    {"test_call_nonlocal", (PyCFunction)test_call_nonlocal, METH_VARARGS, ""},
    {"test_get_plus1_capsule", (PyCFunction)test_get_plus1_capsule, METH_VARARGS, ""},
    {"test_get_plus1b_capsule", (PyCFunction)test_get_plus1b_capsule, METH_VARARGS, ""},
    {"test_get_plus1bc_capsule", (PyCFunction)test_get_plus1bc_capsule, METH_VARARGS, ""},
    {"test_get_data_capsule", (PyCFunction)test_get_data_capsule, METH_VARARGS, ""},
};
    {"test_get_data_capsule", (PyCFunction)test_get_data_capsule, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}



    # 定义一个名称为 "test_get_data_capsule" 的函数
    {"test_get_data_capsule", 
     # 将 C 函数 test_get_data_capsule 转换为 Python C 函数对象
     (PyCFunction)test_get_data_capsule, 
     # 声明该函数接受位置参数并使用 METH_VARARGS 标志
     METH_VARARGS, 
     # 空字符串，用于描述函数的参数和返回值，这里是空的
     ""},
    # 表示该结构体数组的末尾，标志着函数列表的结束
    {NULL, NULL, 0, NULL}


这段代码是用于在 Python 的 C 扩展模块中注册函数的。每个元素对应一个函数的相关信息，如函数名、函数指针、参数传递方式和描述信息。
};

`
# 定义结构体 PyModuleDef 的静态实例 test_ccallback_module
static struct PyModuleDef test_ccallback_module = {
    PyModuleDef_HEAD_INIT,  // 使用 PyModuleDef_HEAD_INIT 进行结构体初始化
    "_test_ccallback",      // 模块名为 "_test_ccallback"
    NULL,                   // 模块文档字符串为空
    -1,                     // 模块状态为 -1 (表示任何错误)
    test_ccallback_methods, // 指定模块方法的数组 test_ccallback_methods
    NULL,                   // 模块的全局状态对象为空
    NULL,                   // 模块的 slot 函数为空
    NULL,                   // 模块的清理函数为空
    NULL                    // 模块的无法解析加载器方法为空
};

# PyInit__test_ccallback 函数，用于创建并返回一个 Python 模块对象
PyMODINIT_FUNC
PyInit__test_ccallback(void)
{
    return PyModule_Create(&test_ccallback_module);  // 使用 test_ccallback_module 创建 Python 模块对象
}
```