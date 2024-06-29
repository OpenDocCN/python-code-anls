# `.\numpy\numpy\_core\src\umath\extobj.c`

```
/* 定义宏，指定使用的 NumPy API 版本 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
/* 定义宏，标识当前编译环境为多维数组模块 */
#define _MULTIARRAYMODULE
/* 定义宏，标识当前编译环境为数学函数模块 */
#define _UMATHMODULE

/* 清除 PY_SSIZE_T 类型的旧定义 */
#define PY_SSIZE_T_CLEAN
/* 包含 Python 核心头文件 */
#include <Python.h>

/* 包含 NumPy 的配置文件 */
#include "npy_config.h"

/* 包含 NumPy 的命令行参数解析头文件 */
#include "npy_argparse.h"

/* 包含 NumPy 的类型转换工具函数 */
#include "conversion_utils.h"

/* 包含外部对象的头文件 */
#include "extobj.h"
/* 包含 NumPy 的通用函数对象头文件 */
#include "numpy/ufuncobject.h"

/* 包含通用的工具函数和宏定义 */
#include "common.h"

/* 定义错误处理的策略：忽略除零错误，其他错误警告 */
#define UFUNC_ERR_IGNORE 0
#define UFUNC_ERR_WARN   1
#define UFUNC_ERR_RAISE  2
#define UFUNC_ERR_CALL   3
#define UFUNC_ERR_PRINT  4
#define UFUNC_ERR_LOG    5

/* 定义错误掩码的位移和掩码值 */
#define UFUNC_MASK_DIVIDEBYZERO 0x07
#define UFUNC_MASK_OVERFLOW (0x07 << UFUNC_SHIFT_OVERFLOW)
#define UFUNC_MASK_UNDERFLOW (0x07 << UFUNC_SHIFT_UNDERFLOW)
#define UFUNC_MASK_INVALID (0x07 << UFUNC_SHIFT_INVALID)

/* 定义错误类型的位移值 */
#define UFUNC_SHIFT_DIVIDEBYZERO 0
#define UFUNC_SHIFT_OVERFLOW     3
#define UFUNC_SHIFT_UNDERFLOW    6
#define UFUNC_SHIFT_INVALID      9

/* 默认的用户错误处理模式：忽略下溢出错误，其他警告 */
#define UFUNC_ERR_DEFAULT                               \
        (UFUNC_ERR_WARN << UFUNC_SHIFT_DIVIDEBYZERO) +  \
        (UFUNC_ERR_WARN << UFUNC_SHIFT_OVERFLOW) +      \
        (UFUNC_ERR_WARN << UFUNC_SHIFT_INVALID)

/* 定义错误处理函数，根据错误类型和处理方法处理异常 */
static int
_error_handler(const char *name, int method, PyObject *pyfunc, char *errtype,
               int retstatus);

/* 定义宏，用于处理浮点数异常 */
#define HANDLEIT(NAME, str) {if (retstatus & NPY_FPE_##NAME) {          \
            handle = errmask & UFUNC_MASK_##NAME;                       \
            if (handle &&                                               \
                _error_handler(name, handle >> UFUNC_SHIFT_##NAME,      \
                               pyfunc, str, retstatus) < 0)      \
                return -1;                                              \
        }}

/* 处理浮点数异常的函数 */
static int
PyUFunc_handlefperr(
        const char *name, int errmask, PyObject *pyfunc, int retstatus)
{
    int handle;
    /* 如果有错误掩码和异常状态 */
    if (errmask && retstatus) {
        /* 处理除零错误 */
        HANDLEIT(DIVIDEBYZERO, "divide by zero");
        /* 处理溢出错误 */
        HANDLEIT(OVERFLOW, "overflow");
        /* 处理下溢出错误 */
        HANDLEIT(UNDERFLOW, "underflow");
        /* 处理无效数值错误 */
        HANDLEIT(INVALID, "invalid value");
    }
    return 0;
}

/* 取消 HANDLEIT 宏的定义 */
#undef HANDLEIT

/* 定义外部对象的析构函数，释放资源 */
static void
extobj_capsule_destructor(PyObject *capsule)
{
    /* 从 Capsule 中获取扩展对象指针 */
    npy_extobj *extobj = PyCapsule_GetPointer(capsule, "numpy.ufunc.extobj");
    /* 清理扩展对象 */
    npy_extobj_clear(extobj);
    /* 释放内存 */
    PyMem_FREE(extobj);
}

/* 创建扩展对象的 Capsule 对象 */
static PyObject *
make_extobj_capsule(npy_intp bufsize, int errmask, PyObject *pyfunc)
{
    /* 分配扩展对象的内存 */
    npy_extobj *extobj = PyMem_Malloc(sizeof(npy_extobj));
    /* 如果内存分配失败 */
    if (extobj == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    /* 设置扩展对象的属性 */
    extobj->bufsize = bufsize;
    extobj->errmask = errmask;
    Py_XINCREF(pyfunc);
    extobj->pyfunc = pyfunc;

    /* 创建并返回 Capsule 对象 */
    PyObject *capsule = PyCapsule_New(
            extobj, "numpy.ufunc.extobj",
            (destructor)&extobj_capsule_destructor);
    /* 如果创建 Capsule 失败 */
    if (capsule == NULL) {
        /* 清理扩展对象 */
        npy_extobj_clear(extobj);
        /* 释放内存 */
        PyMem_Free(extobj);
        return NULL;
    }
    return capsule;
}
/*
 * Fetch the current error/extobj state and fill it into `npy_extobj *extobj`.
 * On success, the filled `extobj` must be cleared using `npy_extobj_clear`.
 * Returns -1 on failure and 0 on success.
 */
static int
fetch_curr_extobj_state(npy_extobj *extobj)
{
    // 获取全局的 extobj 对象封装，使用默认的 extobj capsule
    PyObject *capsule;
    if (PyContextVar_Get(
            npy_static_pydata.npy_extobj_contextvar,
            npy_static_pydata.default_extobj_capsule, &capsule) < 0) {
        return -1; // 获取失败，返回错误状态
    }
    // 从 capsule 中获取指向 npy_extobj 结构体的指针
    npy_extobj *obj = PyCapsule_GetPointer(capsule, "numpy.ufunc.extobj");
    if (obj == NULL) {
        Py_DECREF(capsule);
        return -1; // 获取指针失败，返回错误状态
    }

    // 将获取到的 extobj 数据填充到传入的 extobj 结构体中
    extobj->bufsize = obj->bufsize;
    extobj->errmask = obj->errmask;
    extobj->pyfunc = obj->pyfunc;
    Py_INCREF(extobj->pyfunc); // 增加 Python 函数对象的引用计数，避免被释放

    Py_DECREF(capsule); // 释放 capsule 对象的引用
    return 0; // 返回成功状态
}


NPY_NO_EXPORT int
init_extobj(void)
{
    // 创建默认的 extobj capsule
    npy_static_pydata.default_extobj_capsule = make_extobj_capsule(
            NPY_BUFSIZE, UFUNC_ERR_DEFAULT, Py_None);
    if (npy_static_pydata.default_extobj_capsule == NULL) {
        return -1; // 创建失败，返回错误状态
    }
    // 创建并初始化 extobj 的上下文变量
    npy_static_pydata.npy_extobj_contextvar = PyContextVar_New(
            "numpy.ufunc.extobj", npy_static_pydata.default_extobj_capsule);
    if (npy_static_pydata.npy_extobj_contextvar == NULL) {
        Py_CLEAR(npy_static_pydata.default_extobj_capsule);
        return -1; // 创建失败，释放之前分配的 capsule 并返回错误状态
    }
    return 0; // 返回成功状态
}


/*
 * Parsing helper for extobj_seterrobj to extract the modes
 * "ignore", "raise", etc.
 */
static int
errmodeconverter(PyObject *obj, int *mode)
{
    if (obj == Py_None) {
        return 1; // 如果输入是 Py_None，返回成功状态，并将 mode 设置为 1
    }
    int i = 0;
    // 遍历预定义的错误模式字符串，与输入对象进行比较
    for (; i <= UFUNC_ERR_LOG; i++) {
        // 使用 PyObject_RichCompareBool 检查是否匹配错误模式字符串
        int eq = PyObject_RichCompareBool(
                obj, npy_interned_str.errmode_strings[i], Py_EQ);
        if (eq == -1) {
            return 0; // 比较失败，返回错误状态
        }
        else if (eq) {
            break; // 找到匹配的错误模式，跳出循环
        }
    }
    // 如果未找到匹配的错误模式，抛出 ValueError 异常
    if (i > UFUNC_ERR_LOG) {
        PyErr_Format(PyExc_ValueError, "invalid error mode %.100R", obj);
        return 0; // 返回错误状态
    }

    *mode = i; // 将匹配的错误模式索引赋值给 mode
    return 1; // 返回成功状态
 }


/*
 * This function is currently exposed as `umath._seterrobj()`, it is private
 * and returns a capsule representing the errstate.  This capsule is then
 * assigned to the `_extobj_contextvar` in Python.
 */
NPY_NO_EXPORT PyObject *
extobj_make_extobj(PyObject *NPY_UNUSED(mod),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    // 初始化错误模式变量
    int all_mode = -1;
    int divide_mode = -1;
    int over_mode = -1;
    int under_mode = -1;
    int invalid_mode = -1;
    npy_intp bufsize = -1;
    PyObject *pyfunc = NULL;

    // 解析传入的参数和关键字参数
    NPY_PREPARE_ARGPARSER;

    // 函数体未提供完整，请参阅原始文档或源代码了解更多细节
}
    # 调用函数 npy_parse_arguments，解析传入的参数并设置错误处理模式和缓冲区大小
    if (npy_parse_arguments("_seterrobj", args, len_args, kwnames,
            "$all", &errmodeconverter, &all_mode,
            "$divide", &errmodeconverter, &divide_mode,
            "$over", &errmodeconverter, &over_mode,
            "$under", &errmodeconverter, &under_mode,
            "$invalid", &errmodeconverter, &invalid_mode,
            "$bufsize", &PyArray_IntpFromPyIntConverter, &bufsize,
            "$call", NULL, &pyfunc,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    /* 检查新的缓冲区大小是否有效（负数表示不改变） */
    if (bufsize >= 0) {
        if (bufsize > 10e6) {
            PyErr_Format(PyExc_ValueError,
                    "Buffer size, %" NPY_INTP_FMT ", is too big",
                    bufsize);
            return NULL;
        }
        if (bufsize < 5) {
            PyErr_Format(PyExc_ValueError,
                    "Buffer size, %" NPY_INTP_FMT ", is too small",
                    bufsize);
            return NULL;
        }
        if (bufsize % 16 != 0) {
            PyErr_Format(PyExc_ValueError,
                    "Buffer size, %" NPY_INTP_FMT ", is not a multiple of 16",
                    bufsize);
            return NULL;
        }
    }

    /* 验证 pyfunc 是否为 None、可调用对象，或者具有可调用的 write 方法 */
    if (pyfunc != NULL && pyfunc != Py_None && !PyCallable_Check(pyfunc)) {
        PyObject *temp;
        temp = PyObject_GetAttrString(pyfunc, "write");
        if (temp == NULL || !PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError,
                            "python object must be callable or have "
                            "a callable write method");
            Py_XDECREF(temp);
            return NULL;
        }
        Py_DECREF(temp);
    }

    /* 获取当前的 extobj 状态，如果获取失败则返回 NULL */
    npy_extobj extobj;
    if (fetch_curr_extobj_state(&extobj) < 0) {
        return NULL;
    }

    /* 根据 all_mode 设置默认错误处理模式 */
    if (all_mode != -1) {
        /* 如果传入了 all_mode，则用它来设置未明确指定的其他模式 */
        divide_mode = divide_mode == -1 ? all_mode : divide_mode;
        over_mode = over_mode == -1 ? all_mode : over_mode;
        under_mode = under_mode == -1 ? all_mode : under_mode;
        invalid_mode = invalid_mode == -1 ? all_mode : invalid_mode;
    }
    /* 根据 divide_mode 设置 extobj 的错误掩码 */
    if (divide_mode != -1) {
        extobj.errmask &= ~UFUNC_MASK_DIVIDEBYZERO;
        extobj.errmask |= divide_mode << UFUNC_SHIFT_DIVIDEBYZERO;
    }
    /* 根据 over_mode 设置 extobj 的错误掩码 */
    if (over_mode != -1) {
        extobj.errmask &= ~UFUNC_MASK_OVERFLOW;
        extobj.errmask |= over_mode << UFUNC_SHIFT_OVERFLOW;
    }
    /* 根据 under_mode 设置 extobj 的错误掩码 */
    if (under_mode != -1) {
        extobj.errmask &= ~UFUNC_MASK_UNDERFLOW;
        extobj.errmask |= under_mode << UFUNC_SHIFT_UNDERFLOW;
    }
    /* 根据 invalid_mode 设置 extobj 的错误掩码 */
    if (invalid_mode != -1) {
        extobj.errmask &= ~UFUNC_MASK_INVALID;
        extobj.errmask |= invalid_mode << UFUNC_SHIFT_INVALID;
    }

    /* 如果缓冲区大小大于 0，则设置 extobj 的缓冲区大小 */
    if (bufsize > 0) {
        extobj.bufsize = bufsize;
    }
    // 检查传入的 pyfunc 是否为非空指针
    if (pyfunc != NULL) {
        // 增加 pyfunc 的引用计数，防止其被释放
        Py_INCREF(pyfunc);
        // 将传入的 pyfunc 设置为 extobj 结构体中的 pyfunc 成员
        Py_SETREF(extobj.pyfunc, pyfunc);
    }
    // 使用 extobj 结构体中的 bufsize、errmask 和 pyfunc 成员创建一个新的 Capsule 对象
    PyObject *capsule = make_extobj_capsule(
            extobj.bufsize, extobj.errmask, extobj.pyfunc);
    // 清空 extobj 结构体的内容，准备返回 Capsule 对象
    npy_extobj_clear(&extobj);
    // 返回创建的 Capsule 对象
    return capsule;
/*
 * For inspection purposes, allow fetching a dictionary representing the
 * current extobj/errobj.
 */
NPY_NO_EXPORT PyObject *
extobj_get_extobj_dict(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(noarg))
{
    PyObject *result = NULL, *bufsize_obj = NULL;
    npy_extobj extobj;
    int mode;

    // 获取当前的 extobj 状态，如果获取失败则跳转到失败处理标签
    if (fetch_curr_extobj_state(&extobj) < 0) {
        goto fail;
    }

    // 创建一个新的空字典对象用于存储结果
    result = PyDict_New();
    if (result == NULL) {
        goto fail;
    }

    /* 设置所有的错误模式：*/

    // 设置除零错误处理模式
    mode = (extobj.errmask & UFUNC_MASK_DIVIDEBYZERO) >> UFUNC_SHIFT_DIVIDEBYZERO;
    if (PyDict_SetItemString(result, "divide",
                             npy_interned_str.errmode_strings[mode]) < 0) {
        goto fail;
    }

    // 设置溢出错误处理模式
    mode = (extobj.errmask & UFUNC_MASK_OVERFLOW) >> UFUNC_SHIFT_OVERFLOW;
    if (PyDict_SetItemString(result, "over",
                             npy_interned_str.errmode_strings[mode]) < 0) {
        goto fail;
    }

    // 设置下溢错误处理模式
    mode = (extobj.errmask & UFUNC_MASK_UNDERFLOW) >> UFUNC_SHIFT_UNDERFLOW;
    if (PyDict_SetItemString(result, "under",
                             npy_interned_str.errmode_strings[mode]) < 0) {
        goto fail;
    }

    // 设置无效操作错误处理模式
    mode = (extobj.errmask & UFUNC_MASK_INVALID) >> UFUNC_SHIFT_INVALID;
    if (PyDict_SetItemString(result, "invalid",
                             npy_interned_str.errmode_strings[mode]) < 0) {
        goto fail;
    }

    /* 设置可调用对象：*/

    // 将 extobj.pyfunc 设置为 "call" 字段的值
    if (PyDict_SetItemString(result, "call", extobj.pyfunc) < 0) {
        goto fail;
    }

    /* 设置 bufsize 字段：*/

    // 将 extobj.bufsize 转换为 Python 的长整型对象
    bufsize_obj = PyLong_FromSsize_t(extobj.bufsize);
    if (bufsize_obj == NULL) {
        goto fail;
    }
    // 将 bufsize_obj 设置为 "bufsize" 字段的值
    if (PyDict_SetItemString(result, "bufsize", bufsize_obj) < 0) {
        goto fail;
    }
    Py_DECREF(bufsize_obj);

    // 清理并释放 extobj 结构体
    npy_extobj_clear(&extobj);

    // 返回结果字典
    return result;

  fail:
    // 失败时释放申请的资源并返回 NULL
    Py_XDECREF(result);
    Py_XDECREF(bufsize_obj);
    npy_extobj_clear(&extobj);
    return NULL;
}
    # 如果发生错误但只是发出警告，使用 PyErr_Warn 发出 RuntimeWarning 警告
    case UFUNC_ERR_WARN:
        # 根据错误类型和函数名格式化错误消息
        PyOS_snprintf(msg, sizeof(msg), "%s encountered in %s", errtype, name);
        # 发出 RuntimeWarning 警告，如果发生错误返回负值则跳转到 fail 标签处
        if (PyErr_Warn(PyExc_RuntimeWarning, msg) < 0) {
            goto fail;
        }
        break;
        
    # 如果发生错误需要抛出异常，使用 PyErr_Format 抛出 FloatingPointError 异常
    case UFUNC_ERR_RAISE:
        PyErr_Format(PyExc_FloatingPointError, "%s encountered in %s",
                errtype, name);
        goto fail;
        
    # 如果发生错误需要调用 Python 回调函数，处理 UFUNC_ERR_CALL 情况
    case UFUNC_ERR_CALL:
        # 如果 pyfunc 是 None，则抛出 NameError 异常
        if (pyfunc == Py_None) {
            PyErr_Format(PyExc_NameError,
                    "python callback specified for %s (in " \
                    " %s) but no function found.",
                    errtype, name);
            goto fail;
        }
        # 构建参数 args，包含错误类型和返回状态的 Python 对象
        args = Py_BuildValue("NN", PyUnicode_FromString(errtype),
                PyLong_FromLong((long) retstatus));
        # 如果构建参数失败，则跳转到 fail 标签处
        if (args == NULL) {
            goto fail;
        }
        # 调用 pyfunc 函数，并传入参数 args，处理返回结果
        ret = PyObject_CallObject(pyfunc, args);
        Py_DECREF(args);
        # 如果调用失败，则跳转到 fail 标签处
        if (ret == NULL) {
            goto fail;
        }
        Py_DECREF(ret);
        break;
        
    # 如果发生错误需要记录日志，处理 UFUNC_ERR_LOG 情况
    case UFUNC_ERR_LOG:
        # 如果 pyfunc 是 None，则抛出 NameError 异常
        if (pyfunc == Py_None) {
            PyErr_Format(PyExc_NameError,
                    "log specified for %s (in %s) but no " \
                    "object with write method found.",
                    errtype, name);
            goto fail;
        }
        # 根据错误类型和函数名格式化警告消息
        PyOS_snprintf(msg, sizeof(msg),
                "Warning: %s encountered in %s\n", errtype, name);
        # 调用 pyfunc 对象的 write 方法，写入警告消息
        ret = PyObject_CallMethod(pyfunc, "write", "s", msg);
        # 如果调用失败，则跳转到 fail 标签处
        if (ret == NULL) {
            goto fail;
        }
        Py_DECREF(ret);
        break;
    }
    # 禁用 C API
    NPY_DISABLE_C_API;
    # 返回 0 表示处理完成
    return 0;
/*
 * Disable the C API and return -1 to indicate failure.
 */
fail:
    NPY_DISABLE_C_API;
    return -1;
}


/*
 * Extracts some values from the global pyvals tuple.
 * all destinations may be NULL, in which case they are not retrieved
 * ref - should hold the global tuple
 * name - is the name of the ufunc (ufuncobj->name)
 *
 * bufsize - receives the buffer size to use
 * errmask - receives the bitmask for error handling
 * pyfunc - receives the python object to call with the error,
 *          if an error handling method is 'call'
 */
static int
_extract_pyvals(int *bufsize, int *errmask, PyObject **pyfunc)
{
    npy_extobj extobj;
    // Fetches the current state of the extended objects (extobj)
    if (fetch_curr_extobj_state(&extobj) < 0) {
        return -1;
    }

    // Retrieve bufsize if not NULL
    if (bufsize != NULL) {
        *bufsize = extobj.bufsize;
    }

    // Retrieve errmask if not NULL
    if (errmask != NULL) {
        *errmask = extobj.errmask;
    }

    // Retrieve pyfunc if not NULL and increment its reference count
    if (pyfunc != NULL) {
        *pyfunc = extobj.pyfunc;
        Py_INCREF(*pyfunc);
    }
    // Clears the extobj structure
    npy_extobj_clear(&extobj);
    return 0;
}

/*UFUNC_API
 * Signal a floating point error respecting the error signaling setting in
 * the NumPy errstate. Takes the name of the operation to use in the error
 * message and an integer flag that is one of NPY_FPE_DIVIDEBYZERO,
 * NPY_FPE_OVERFLOW, NPY_FPE_UNDERFLOW, NPY_FPE_INVALID to indicate
 * which errors to check for.
 *
 * Returns -1 on failure (an error was raised) and 0 on success.
 */
NPY_NO_EXPORT int
PyUFunc_GiveFloatingpointErrors(const char *name, int fpe_errors)
{
    int bufsize, errmask;
    PyObject *pyfunc = NULL;

    // Extracts bufsize, errmask, and pyfunc from the global pyvals tuple
    if (_extract_pyvals(&bufsize, &errmask, &pyfunc) < 0) {
        Py_XDECREF(pyfunc);
        return -1;
    }
    // Handles floating point errors using extracted parameters
    if (PyUFunc_handlefperr(name, errmask, pyfunc, fpe_errors)) {
        Py_XDECREF(pyfunc);
        return -1;
    }
    Py_XDECREF(pyfunc);
    return 0;
}


/*
 * check the floating point status
 *  - errmask: mask of status to check
 *  - extobj: ufunc pyvals object
 *            may be null, in which case the thread global one is fetched
 *  - ufunc_name: name of ufunc
 */
NPY_NO_EXPORT int
_check_ufunc_fperr(int errmask, const char *ufunc_name) {
    int fperr;
    PyObject *pyfunc = NULL;
    int ret;

    // If errmask is 0, no error checking is needed
    if (!errmask) {
        return 0;
    }
    // Fetches the floating point status using the ufunc name
    fperr = npy_get_floatstatus_barrier((char*)ufunc_name);
    // If no floating point errors occurred, return success
    if (!fperr) {
        return 0;
    }

    // Retrieves pyvals tuple and handles floating point errors
    if (_extract_pyvals(NULL, NULL, &pyfunc) < 0) {
        Py_XDECREF(pyfunc);
        return -1;
    }

    ret = PyUFunc_handlefperr(ufunc_name, errmask, pyfunc, fperr);
    Py_XDECREF(pyfunc);

    return ret;
}


NPY_NO_EXPORT int
_get_bufsize_errmask(int *buffersize, int *errormask)
{
    // Retrieves bufsize and errmask using _extract_pyvals function
    return _extract_pyvals(buffersize, errormask, NULL);
}
```