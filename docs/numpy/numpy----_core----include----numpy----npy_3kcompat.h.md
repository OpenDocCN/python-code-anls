# `.\numpy\numpy\_core\include\numpy\npy_3kcompat.h`

```py
/*
 * This is a convenience header file providing compatibility utilities
 * for supporting different minor versions of Python 3.
 * It was originally used to support the transition from Python 2,
 * hence the "3k" naming.
 *
 * If you want to use this for your own projects, it's recommended to make a
 * copy of it. Although the stuff below is unlikely to change, we don't provide
 * strong backwards compatibility guarantees at the moment.
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_3KCOMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_3KCOMPAT_H_

#include <Python.h>
#include <stdio.h>

#ifndef NPY_PY3K
#define NPY_PY3K 1
#endif

#include "numpy/npy_common.h"
#include "numpy/ndarrayobject.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * PyInt -> PyLong
 */

/*
 * This function converts a Python object to a C int, handling overflow cases.
 * It mimics _PyLong_AsInt from Python's limited API, included for compatibility.
 */
static inline int
Npy__PyLong_AsInt(PyObject *obj)
{
    int overflow;
    long result = PyLong_AsLongAndOverflow(obj, &overflow);

    /* INT_MAX and INT_MIN are defined in Python.h */
    if (overflow || result > INT_MAX || result < INT_MIN) {
        /* Sets an OverflowError if the Python int is too large for a C int */
        PyErr_SetString(PyExc_OverflowError,
                        "Python int too large to convert to C int");
        return -1;
    }
    return (int)result;
}


#if defined(NPY_PY3K)
/* Check if the given object is a PyInt (in Python 2) */
static inline int PyInt_Check(PyObject *op) {
    int overflow = 0;
    if (!PyLong_Check(op)) {
        return 0;
    }
    PyLong_AsLongAndOverflow(op, &overflow);
    return (overflow == 0);
}

/* Macros for compatibility with Python 3 */
#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AsLong
#define PyInt_AsSsize_t PyLong_AsSsize_t
#define PyNumber_Int PyNumber_Long

/* NOTE:
 *
 * Since the PyLong type is very different from the fixed-range PyInt,
 * we don't define PyInt_Type -> PyLong_Type.
 */
#endif /* NPY_PY3K */

/* Py3 changes PySlice_GetIndicesEx' first argument's type to PyObject* */
#ifdef NPY_PY3K
#  define NpySlice_GetIndicesEx PySlice_GetIndicesEx
#else
#  define NpySlice_GetIndicesEx(op, nop, start, end, step, slicelength) \
    PySlice_GetIndicesEx((PySliceObject *)op, nop, start, end, step, slicelength)
#endif

#if PY_VERSION_HEX < 0x030900a4
    /* Introduced in https://github.com/python/cpython/commit/d2ec81a8c99796b51fb8c49b77a7fe369863226f */
    #define Py_SET_TYPE(obj, type) ((Py_TYPE(obj) = (type)), (void)0)
    /* Introduced in https://github.com/python/cpython/commit/b10dc3e7a11fcdb97e285882eba6da92594f90f9 */
    #define Py_SET_SIZE(obj, size) ((Py_SIZE(obj) = (size)), (void)0)
    /* Introduced in https://github.com/python/cpython/commit/c86a11221df7e37da389f9c6ce6e47ea22dc44ff */
#endif

#endif /* NUMPY_CORE_INCLUDE_NUMPY_NPY_3KCOMPAT_H_ */
    #define Py_SET_REFCNT(obj, refcnt) ((Py_REFCNT(obj) = (refcnt)), (void)0)
#endif

#define Npy_EnterRecursiveCall(x) Py_EnterRecursiveCall(x)

/*
 * 定义宏：将 PyString 相关宏重定向到 PyBytes 相关宏
 */

#if defined(NPY_PY3K)

// 在 Python 3 中，将 PyString 相关宏重定向到 PyBytes 相关宏
#define PyString_Type PyBytes_Type
#define PyString_Check PyBytes_Check
#define PyStringObject PyBytesObject
#define PyString_FromString PyBytes_FromString
#define PyString_FromStringAndSize PyBytes_FromStringAndSize
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_AsStringAndSize PyBytes_AsStringAndSize
#define PyString_FromFormat PyBytes_FromFormat
#define PyString_Concat PyBytes_Concat
#define PyString_ConcatAndDel PyBytes_ConcatAndDel
#define PyString_AsString PyBytes_AsString
#define PyString_GET_SIZE PyBytes_GET_SIZE
#define PyString_Size PyBytes_Size

// 同时，将 PyUnicode 相关宏也定义为 PyString 相关宏
#define PyUString_Type PyUnicode_Type
#define PyUString_Check PyUnicode_Check
#define PyUStringObject PyUnicodeObject
#define PyUString_FromString PyUnicode_FromString
#define PyUString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyUString_FromFormat PyUnicode_FromFormat
#define PyUString_Concat PyUnicode_Concat2
#define PyUString_ConcatAndDel PyUnicode_ConcatAndDel
#define PyUString_GET_SIZE PyUnicode_GET_SIZE
#define PyUString_Size PyUnicode_Size
#define PyUString_InternFromString PyUnicode_InternFromString
#define PyUString_Format PyUnicode_Format

// 定义宏：检查对象是否为基本字符串对象（PyString 或 PyBytes）
#define PyBaseString_Check(obj) (PyUnicode_Check(obj))

#else

// 在 Python 2 中，将 PyBytes 相关宏重定向到 PyString 相关宏
#define PyBytes_Type PyString_Type
#define PyBytes_Check PyString_Check
#define PyBytesObject PyStringObject
#define PyBytes_FromString PyString_FromString
#define PyBytes_FromStringAndSize PyString_FromStringAndSize
#define PyBytes_AS_STRING PyString_AS_STRING
#define PyBytes_AsStringAndSize PyString_AsStringAndSize
#define PyBytes_FromFormat PyString_FromFormat
#define PyBytes_Concat PyString_Concat
#define PyBytes_ConcatAndDel PyString_ConcatAndDel
#define PyBytes_AsString PyString_AsString
#define PyBytes_GET_SIZE PyString_GET_SIZE
#define PyBytes_Size PyString_Size

// 同时，将 PyUnicode 相关宏也定义为 PyString 相关宏
#define PyUString_Type PyString_Type
#define PyUString_Check PyString_Check
#define PyUStringObject PyStringObject
#define PyUString_FromString PyString_FromString
#define PyUString_FromStringAndSize PyString_FromStringAndSize
#define PyUString_FromFormat PyString_FromFormat
#define PyUString_Concat PyString_Concat
#define PyUString_ConcatAndDel PyString_ConcatAndDel
#define PyUString_GET_SIZE PyString_GET_SIZE
#define PyUString_Size PyString_Size
#define PyUString_InternFromString PyString_InternFromString
#define PyUString_Format PyString_Format

// 定义宏：检查对象是否为基本字符串对象（PyString）
#define PyBaseString_Check(obj) (PyBytes_Check(obj) || PyUnicode_Check(obj))

#endif /* NPY_PY3K */

/*
 * 定义宏：保护 CRT 调用，防止因为传递无效参数而导致的程序立即终止
 * 参见：https://bugs.python.org/issue23524
 */
#if defined _MSC_VER && _MSC_VER >= 1900

#include <stdlib.h>

extern _invalid_parameter_handler _Py_silent_invalid_parameter_handler;
// 定义宏：进入抑制无效参数处理器的范围
#define NPY_BEGIN_SUPPRESS_IPH { _invalid_parameter_handler _Py_old_handler = \
    # 设置线程本地的无效参数处理函数为静默模式的无效参数处理函数
    _set_thread_local_invalid_parameter_handler(_Py_silent_invalid_parameter_handler);
/*
 * 定义 _WIN32 环境下的宏 NPY_BEGIN_SUPPRESS_IPH 和 NPY_END_SUPPRESS_IPH，
 * 这些宏用于暂时禁用 Windows 平台的无效参数处理程序
 */
#ifdef _WIN32
#define NPY_BEGIN_SUPPRESS_IPH \
    _set_thread_local_invalid_parameter_handler(_Py_old_handler);
#define NPY_END_SUPPRESS_IPH
#else
#define NPY_BEGIN_SUPPRESS_IPH
#define NPY_END_SUPPRESS_IPH
#endif /* _MSC_VER >= 1900 */

/*
 * PyUnicode_ConcatAndDel 和 PyUnicode_Concat2 函数
 */

/*
 * 将 right 参数连接到 left 参数指向的 Unicode 对象，并释放 right
 */
static inline void
PyUnicode_ConcatAndDel(PyObject **left, PyObject *right)
{
    Py_SETREF(*left, PyUnicode_Concat(*left, right));
    Py_DECREF(right);
}

/*
 * 将 right 参数连接到 left 参数指向的 Unicode 对象
 */
static inline void
PyUnicode_Concat2(PyObject **left, PyObject *right)
{
    Py_SETREF(*left, PyUnicode_Concat(*left, right));
}

/*
 * PyFile_* 兼容性
 */

/*
 * 获取表示 Python 对象所代表的文件的 FILE* 句柄
 */
static inline FILE*
npy_PyFile_Dup2(PyObject *file, char *mode, npy_off_t *orig_pos)
{
    int fd, fd2, unbuf;
    Py_ssize_t fd2_tmp;
    PyObject *ret, *os, *io, *io_raw;
    npy_off_t pos;
    FILE *handle;

    /* 对于 Python 2 的 PyFileObject，使用 PyFile_AsFile 返回 FILE* 句柄 */
#if !defined(NPY_PY3K)
    if (PyFile_Check(file)) {
        return PyFile_AsFile(file);
    }
#endif

    /* 先刷新以确保数据按正确顺序写入文件 */
    ret = PyObject_CallMethod(file, "flush", "");
    if (ret == NULL) {
        return NULL;
    }
    Py_DECREF(ret);
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        return NULL;
    }

    /*
     * 需要 dup 句柄，因为最终要调用 fclose
     */
    os = PyImport_ImportModule("os");
    if (os == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(os, "dup", "i", fd);
    Py_DECREF(os);
    if (ret == NULL) {
        return NULL;
    }
    fd2_tmp = PyNumber_AsSsize_t(ret, PyExc_IOError);
    Py_DECREF(ret);
    if (fd2_tmp == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (fd2_tmp < INT_MIN || fd2_tmp > INT_MAX) {
        PyErr_SetString(PyExc_IOError,
                        "从 os.dup() 获取 int 失败");
        return NULL;
    }
    fd2 = (int)fd2_tmp;

    /* 转换为 FILE* 句柄 */
#ifdef _WIN32
    NPY_BEGIN_SUPPRESS_IPH
    handle = _fdopen(fd2, mode);
    NPY_END_SUPPRESS_IPH
#else
    handle = fdopen(fd2, mode);
#endif
    if (handle == NULL) {
        PyErr_SetString(PyExc_IOError,
                        "从 Python 文件对象获取 FILE* 句柄失败。如果是在构建 NumPy 时出现问题，可能是因为链接了错误的调试/发布运行时库");
        return NULL;
    }

    /* 记录原始的文件句柄位置 */
    *orig_pos = npy_ftell(handle);
}
    # 如果原始位置是 -1，则说明需要确定文件当前的位置信息

    if (*orig_pos == -1) {
        /* 导入 io 模块以确定是否使用了缓冲 */
        io = PyImport_ImportModule("io");
        if (io == NULL) {
            // 如果导入失败，则关闭文件句柄并返回空指针
            fclose(handle);
            return NULL;
        }
        /* 文件对象实例的 RawIOBase 是无缓冲的 */
        io_raw = PyObject_GetAttrString(io, "RawIOBase");
        Py_DECREF(io);
        if (io_raw == NULL) {
            // 如果获取 RawIOBase 失败，则关闭文件句柄并返回空指针
            fclose(handle);
            return NULL;
        }
        // 检查文件是否是无缓冲的
        unbuf = PyObject_IsInstance(file, io_raw);
        Py_DECREF(io_raw);
        if (unbuf == 1) {
            // 如果文件是无缓冲的，则直接返回文件句柄
            return handle;
        }
        else {
            // 如果文件不是无缓冲的，设置异常并关闭文件句柄后返回空指针
            PyErr_SetString(PyExc_IOError, "obtaining file position failed");
            fclose(handle);
            return NULL;
        }
    }

    // 将原始句柄定位到 Python 端的位置
    ret = PyObject_CallMethod(file, "tell", "");
    if (ret == NULL) {
        // 如果获取位置信息失败，则关闭文件句柄并返回空指针
        fclose(handle);
        return NULL;
    }
    pos = PyLong_AsLongLong(ret);
    Py_DECREF(ret);
    if (PyErr_Occurred()) {
        // 如果发生异常，则关闭文件句柄并返回空指针
        fclose(handle);
        return NULL;
    }
    // 使用 npy_fseek 将文件定位到指定位置
    if (npy_fseek(handle, pos, SEEK_SET) == -1) {
        // 如果定位失败，则设置异常并关闭文件句柄后返回空指针
        PyErr_SetString(PyExc_IOError, "seeking file failed");
        fclose(handle);
        return NULL;
    }
    // 返回文件句柄
    return handle;
/*
 * Close the dup-ed file handle, and seek the Python one to the current position
 */
static inline int
npy_PyFile_DupClose2(PyObject *file, FILE* handle, npy_off_t orig_pos)
{
    int fd, unbuf;
    PyObject *ret, *io, *io_raw;
    npy_off_t position;

    /* For Python 2 PyFileObject, do nothing */
#if !defined(NPY_PY3K)
    if (PyFile_Check(file)) {
        return 0;
    }
#endif

    position = npy_ftell(handle);

    /* Close the FILE* handle */
    fclose(handle);

    /*
     * Restore original file handle position, in order to not confuse
     * Python-side data structures
     */
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        return -1;
    }

    if (npy_lseek(fd, orig_pos, SEEK_SET) == -1) {

        /* The io module is needed to determine if buffering is used */
        io = PyImport_ImportModule("io");
        if (io == NULL) {
            return -1;
        }
        /* File object instances of RawIOBase are unbuffered */
        io_raw = PyObject_GetAttrString(io, "RawIOBase");
        Py_DECREF(io);
        if (io_raw == NULL) {
            return -1;
        }
        unbuf = PyObject_IsInstance(file, io_raw);
        Py_DECREF(io_raw);
        if (unbuf == 1) {
            /* Succeed if the IO is unbuffered */
            return 0;
        }
        else {
            PyErr_SetString(PyExc_IOError, "seeking file failed");
            return -1;
        }
    }

    if (position == -1) {
        PyErr_SetString(PyExc_IOError, "obtaining file position failed");
        return -1;
    }

    /* Seek Python-side handle to the FILE* handle position */
    ret = PyObject_CallMethod(file, "seek", NPY_OFF_T_PYFMT "i", position, 0);
    if (ret == NULL) {
        return -1;
    }
    Py_DECREF(ret);
    return 0;
}

static inline int
npy_PyFile_Check(PyObject *file)
{
    int fd;
    /* For Python 2, check if it is a PyFileObject */
#if !defined(NPY_PY3K)
    if (PyFile_Check(file)) {
        return 1;
    }
#endif
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

static inline PyObject*
npy_PyFile_OpenFile(PyObject *filename, const char *mode)
{
    PyObject *open;
    open = PyDict_GetItemString(PyEval_GetBuiltins(), "open");
    if (open == NULL) {
        return NULL;
    }
    return PyObject_CallFunction(open, "Os", filename, mode);
}

static inline int
npy_PyFile_CloseFile(PyObject *file)
{
    PyObject *ret;

    ret = PyObject_CallMethod(file, "close", NULL);
    if (ret == NULL) {
        return -1;
    }
    Py_DECREF(ret);
    return 0;
}


/* This is a copy of _PyErr_ChainExceptions
 */
static inline void
npy_PyErr_ChainExceptions(PyObject *exc, PyObject *val, PyObject *tb)
{
    if (exc == NULL)
        return;
}
    # 检查是否有异常发生
    if (PyErr_Occurred()) {
        /* 只有 Python 3 支持这个功能 */
        #ifdef NPY_PY3K
            // 声明用于存储第二个异常的变量
            PyObject *exc2, *val2, *tb2;
            // 获取并清除第二个异常
            PyErr_Fetch(&exc2, &val2, &tb2);
            // 规范化当前异常
            PyErr_NormalizeException(&exc, &val, &tb);
            // 如果有 traceback 信息，则将其设置给当前异常
            if (tb != NULL) {
                PyException_SetTraceback(val, tb);
                // 释放 traceback 对象的引用计数
                Py_DECREF(tb);
            }
            // 释放当前异常对象的引用计数
            Py_DECREF(exc);
            // 规范化第二个异常
            PyErr_NormalizeException(&exc2, &val2, &tb2);
            // 将第二个异常设置为当前异常的上下文
            PyException_SetContext(val2, val);
            // 恢复第二个异常
            PyErr_Restore(exc2, val2, tb2);
        #endif
    }
    else {
        // 恢复之前捕获的异常
        PyErr_Restore(exc, val, tb);
    }
/* This is a copy of _PyErr_ChainExceptions, with:
 *  - a minimal implementation for python 2
 *  - __cause__ used instead of __context__
 */
/* 这是 _PyErr_ChainExceptions 的副本，包括：
 *  - 用于 Python 2 的最小实现
 *  - 使用 __cause__ 而不是 __context__
 */
static inline void
npy_PyErr_ChainExceptionsCause(PyObject *exc, PyObject *val, PyObject *tb)
{
    if (exc == NULL)
        return;
    // 检查当前是否有异常发生
    if (PyErr_Occurred()) {
        /* only py3 supports this anyway */
        // 只有 Python 3 支持这种操作
        #ifdef NPY_PY3K
            PyObject *exc2, *val2, *tb2;
            // 获取当前的异常信息
            PyErr_Fetch(&exc2, &val2, &tb2);
            // 规范化异常对象，确保是标准的异常对象
            PyErr_NormalizeException(&exc, &val, &tb);
            // 如果 traceback 不为空，则设置到异常值中，并释放 traceback 对象
            if (tb != NULL) {
                PyException_SetTraceback(val, tb);
                Py_DECREF(tb);
            }
            // 释放旧的异常对象
            Py_DECREF(exc);
            // 再次规范化异常对象，确保是标准的异常对象
            PyErr_NormalizeException(&exc2, &val2, &tb2);
            // 将新异常对象作为原因设置给旧的异常对象
            PyException_SetCause(val2, val);
            // 恢复之前的异常状态
            PyErr_Restore(exc2, val2, tb2);
        #endif
    }
    else {
        // 恢复之前的异常状态
        PyErr_Restore(exc, val, tb);
    }
}

/*
 * PyObject_Cmp
 */
/* PyObject_Cmp 函数 */
#if defined(NPY_PY3K)
static inline int
PyObject_Cmp(PyObject *i1, PyObject *i2, int *cmp)
{
    int v;
    // 比较 i1 < i2
    v = PyObject_RichCompareBool(i1, i2, Py_LT);
    if (v == 1) {
        *cmp = -1;
        return 1;
    }
    else if (v == -1) {
        return -1;
    }

    // 比较 i1 > i2
    v = PyObject_RichCompareBool(i1, i2, Py_GT);
    if (v == 1) {
        *cmp = 1;
        return 1;
    }
    else if (v == -1) {
        return -1;
    }

    // 比较 i1 == i2
    v = PyObject_RichCompareBool(i1, i2, Py_EQ);
    if (v == 1) {
        *cmp = 0;
        return 1;
    }
    else {
        *cmp = 0;
        return -1;
    }
}
#endif

/*
 * PyCObject functions adapted to PyCapsules.
 *
 * The main job here is to get rid of the improved error handling
 * of PyCapsules. It's a shame...
 */
/* 适应于 PyCapsules 的 PyCObject 函数 */

static inline PyObject *
NpyCapsule_FromVoidPtr(void *ptr, void (*dtor)(PyObject *))
{
    // 创建一个 PyCapsule 对象，封装指针 ptr，使用 dtor 作为析构函数
    PyObject *ret = PyCapsule_New(ptr, NULL, dtor);
    // 如果创建失败，则清除当前的异常状态
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

static inline PyObject *
NpyCapsule_FromVoidPtrAndDesc(void *ptr, void* context, void (*dtor)(PyObject *))
{
    // 创建一个 PyCapsule 对象，封装指针 ptr，使用 dtor 作为析构函数
    PyObject *ret = NpyCapsule_FromVoidPtr(ptr, dtor);
    // 如果创建成功且设置 context 失败，则清除当前的异常状态并释放对象
    if (ret != NULL && PyCapsule_SetContext(ret, context) != 0) {
        PyErr_Clear();
        Py_DECREF(ret);
        ret = NULL;
    }
    return ret;
}

static inline void *
NpyCapsule_AsVoidPtr(PyObject *obj)
{
    // 从 PyCapsule 对象中获取指针
    void *ret = PyCapsule_GetPointer(obj, NULL);
    // 如果获取失败，则清除当前的异常状态
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

static inline void *
NpyCapsule_GetDesc(PyObject *obj)
{
    // 获取 PyCapsule 对象的 context
    return PyCapsule_GetContext(obj);
}

static inline int
NpyCapsule_Check(PyObject *ptr)
{
    // 检查对象是否为 PyCapsule 类型
    return PyCapsule_CheckExact(ptr);
}

#ifdef __cplusplus
}
#endif


#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_3KCOMPAT_H_ */
```