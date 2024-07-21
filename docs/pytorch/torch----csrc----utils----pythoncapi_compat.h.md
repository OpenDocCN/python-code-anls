# `.\pytorch\torch\csrc\utils\pythoncapi_compat.h`

```py
/*
// Header file providing new C API functions to old Python versions.
//
// File distributed under the Zero Clause BSD (0BSD) license.
// Copyright Contributors to the pythoncapi_compat project.
//
// Homepage:
// https://github.com/python/pythoncapi_compat
//
// Latest version:
// https://raw.githubusercontent.com/python/pythoncapi_compat/master/pythoncapi_compat.h
//
// SPDX-License-Identifier: 0BSD
*/

#ifndef PYTHONCAPI_COMPAT
#define PYTHONCAPI_COMPAT

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include "frameobject.h"          // PyFrameObject, PyFrame_GetBack()


// Compatibility with Visual Studio 2013 and older which don't support
// the inline keyword in C (only in C++): use __inline instead.
#if (defined(_MSC_VER) && _MSC_VER < 1900 \
     && !defined(__cplusplus) && !defined(inline))
#  define PYCAPI_COMPAT_STATIC_INLINE(TYPE) static __inline TYPE
#else
#  define PYCAPI_COMPAT_STATIC_INLINE(TYPE) static inline TYPE
#endif


#ifndef _Py_CAST
#  define _Py_CAST(type, expr) ((type)(expr))
#endif

// On C++11 and newer, _Py_NULL is defined as nullptr on C++11,
// otherwise it is defined as NULL.
#ifndef _Py_NULL
#  if defined(__cplusplus) && __cplusplus >= 201103
#    define _Py_NULL nullptr
#  else
#    define _Py_NULL NULL
#  endif
#endif

// Cast argument to PyObject* type.
#ifndef _PyObject_CAST
#  define _PyObject_CAST(op) _Py_CAST(PyObject*, op)
#endif


// bpo-42262 added Py_NewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_NewRef)
// Define inline function _Py_NewRef for older Python versions
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
_Py_NewRef(PyObject *obj)
{
    Py_INCREF(obj);  // Increment reference count of obj
    return obj;      // Return obj with increased reference count
}
#define Py_NewRef(obj) _Py_NewRef(_PyObject_CAST(obj))  // Define Py_NewRef macro
#endif


// bpo-42262 added Py_XNewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_XNewRef)
// Define inline function _Py_XNewRef for older Python versions
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
_Py_XNewRef(PyObject *obj)
{
    Py_XINCREF(obj);  // Increment (or handle null) reference count of obj
    return obj;        // Return obj with incremented reference count
}
#define Py_XNewRef(obj) _Py_XNewRef(_PyObject_CAST(obj))  // Define Py_XNewRef macro
#endif


// bpo-39573 added Py_SET_REFCNT() to Python 3.9.0a4
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_REFCNT)
// Define inline function _Py_SET_REFCNT for older Python versions
PYCAPI_COMPAT_STATIC_INLINE(void)
_Py_SET_REFCNT(PyObject *ob, Py_ssize_t refcnt)
{
    ob->ob_refcnt = refcnt;  // Set the reference count of ob to refcnt
}
#define Py_SET_REFCNT(ob, refcnt) _Py_SET_REFCNT(_PyObject_CAST(ob), refcnt)  // Define Py_SET_REFCNT macro
#endif


// Py_SETREF() and Py_XSETREF() were added to Python 3.5.2.
// It is excluded from the limited C API.
#if (PY_VERSION_HEX < 0x03050200 && !defined(Py_SETREF)) && !defined(Py_LIMITED_API)
// Define Py_SETREF macro for older Python versions
#define Py_SETREF(dst, src)                                     \
    do {                                                        \
        PyObject **_tmp_dst_ptr = _Py_CAST(PyObject**, &(dst)); \
        PyObject *_tmp_dst = (*_tmp_dst_ptr);                   \
        *_tmp_dst_ptr = _PyObject_CAST(src);                    \
        Py_DECREF(_tmp_dst);                                    \
    } while (0)

#define Py_XSETREF(dst, src)                                    \
    // Define Py_XSETREF macro (currently empty) for older Python versions
#endif
    # 使用宏来实现安全地将源对象转换并赋值给目标对象的操作
    do {                                                        \
        # 定义临时指针，用于存储目标对象的地址
        PyObject **_tmp_dst_ptr = _Py_CAST(PyObject**, &(dst)); \
        # 定义临时变量，用于存储目标对象的值
        PyObject *_tmp_dst = (*_tmp_dst_ptr);                   \
        # 将源对象转换为目标对象的类型，并赋值给目标对象
        *_tmp_dst_ptr = _PyObject_CAST(src);                    \
        # 减少临时变量的引用计数，释放临时变量占用的资源
        Py_XDECREF(_tmp_dst);                                   \
    } while (0)
// 如果 Python 版本低于 3.10.0b1，并且未定义 Py_Is 宏，则定义 Py_Is 宏
#if PY_VERSION_HEX < 0x030A00B1 && !defined(Py_Is)
#  define Py_Is(x, y) ((x) == (y))
#endif

// 如果 Python 版本低于 3.10.0b1，并且未定义 Py_IsNone 宏，则定义 Py_IsNone 宏
#if PY_VERSION_HEX < 0x030A00B1 && !defined(Py_IsNone)
#  define Py_IsNone(x) Py_Is(x, Py_None)
#endif

// 如果 Python 版本低于 3.10.0b1，并且未定义 Py_IsTrue 宏，则定义 Py_IsTrue 宏
#if PY_VERSION_HEX < 0x030A00B1 && !defined(Py_IsTrue)
#  define Py_IsTrue(x) Py_Is(x, Py_True)
#endif

// 如果 Python 版本低于 3.10.0b1，并且未定义 Py_IsFalse 宏，则定义 Py_IsFalse 宏
#if PY_VERSION_HEX < 0x030A00B1 && !defined(Py_IsFalse)
#  define Py_IsFalse(x) Py_Is(x, Py_False)
#endif


// 如果 Python 版本低于 3.9.0a4，并且未定义 Py_SET_TYPE 宏，则定义 Py_SET_TYPE 宏
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
PYCAPI_COMPAT_STATIC_INLINE(void)
_Py_SET_TYPE(PyObject *ob, PyTypeObject *type)
{
    ob->ob_type = type;
}
#define Py_SET_TYPE(ob, type) _Py_SET_TYPE(_PyObject_CAST(ob), type)
#endif


// 如果 Python 版本低于 3.9.0a4，并且未定义 Py_SET_SIZE 宏，则定义 Py_SET_SIZE 宏
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_SIZE)
PYCAPI_COMPAT_STATIC_INLINE(void)
_Py_SET_SIZE(PyVarObject *ob, Py_ssize_t size)
{
    ob->ob_size = size;
}
#define Py_SET_SIZE(ob, size) _Py_SET_SIZE((PyVarObject*)(ob), size)
#endif


// 如果 Python 版本低于 3.9.0b1 或者定义了 PYPY_VERSION，则定义 PyFrame_GetCode 函数
#if PY_VERSION_HEX < 0x030900B1 || defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyCodeObject*)
PyFrame_GetCode(PyFrameObject *frame)
{
    assert(frame != _Py_NULL);
    assert(frame->f_code != _Py_NULL);
    return _Py_CAST(PyCodeObject*, Py_NewRef(frame->f_code));
}
#endif

// 定义 PyFrame_GetCodeBorrow 函数，从 PyFrame_GetCode 函数中获取代码对象并减少其引用计数后返回
PYCAPI_COMPAT_STATIC_INLINE(PyCodeObject*)
_PyFrame_GetCodeBorrow(PyFrameObject *frame)
{
    PyCodeObject *code = PyFrame_GetCode(frame);
    Py_DECREF(code);
    return code;
}


// 如果 Python 版本低于 3.9.0b1 并且未定义 PYPY_VERSION，则定义 PyFrame_GetBack 函数
#if PY_VERSION_HEX < 0x030900B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
PyFrame_GetBack(PyFrameObject *frame)
{
    assert(frame != _Py_NULL);
    return _Py_CAST(PyFrameObject*, Py_XNewRef(frame->f_back));
}
#endif

// 如果未定义 PYPY_VERSION，则定义 PyFrame_GetBackBorrow 函数，从 PyFrame_GetBack 函数中获取后向帧对象并减少其引用计数后返回
#if !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
_PyFrame_GetBackBorrow(PyFrameObject *frame)
{
    PyFrameObject *back = PyFrame_GetBack(frame);
    Py_XDECREF(back);
    return back;
}
#endif


// 如果 Python 版本低于 3.11.0a7 并且未定义 PYPY_VERSION，则定义 PyFrame_GetLocals 函数
#if PY_VERSION_HEX < 0x030B00A7 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetLocals(PyFrameObject *frame)
{
#if PY_VERSION_HEX >= 0x030400B1
    if (PyFrame_FastToLocalsWithError(frame) < 0) {
        return NULL;
    }
#else
    PyFrame_FastToLocals(frame);
#endif
    return Py_NewRef(frame->f_locals);
}
#endif


// 如果 Python 版本低于 3.11.0a7 并且未定义 PYPY_VERSION，则定义 PyFrame_GetGlobals 函数
#if PY_VERSION_HEX < 0x030B00A7 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetGlobals(PyFrameObject *frame)
{
    return Py_NewRef(frame->f_globals);
}
#endif


// 如果 Python 版本低于 3.11.0a7 并且未定义 PYPY_VERSION，则定义 PyFrame_GetBuiltins 函数
#if PY_VERSION_HEX < 0x030B00A7 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetBuiltins(PyFrameObject *frame)
{
    # 返回当前 Python 执行帧的内置变量字典的新引用
    return Py_NewRef(frame->f_builtins);
// 如果 Python 版本小于 3.9.0a5 或者是 PyPy，定义 PYCAPI_COMPAT_STATIC_INLINE 宏，返回当前线程状态的解释器指针
#if PY_VERSION_HEX < 0x030900A5 || defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyInterpreterState*)
PyInterpreterState_Get(void)
{
    PyThreadState *tstate;
    PyInterpreterState *interp;

    // 获取当前线程状态
    tstate = PyThreadState_GET();
    // 断言当前线程状态不为空
    assert(tstate != _Py_NULL);
    // 返回当前线程状态的解释器
    return tstate->interp;
}
#endif


// 如果 Python 版本小于 3.9.0b1 且不是 PyPy，定义 PYCAPI_COMPAT_STATIC_INLINE 宏，返回当前线程状态的帧对象指针
#if PY_VERSION_HEX < 0x030900B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
PyThreadState_GetFrame(PyThreadState *tstate)
{
    // 断言当前线程状态不为空
    assert(tstate != _Py_NULL);
    // 使用 Py_XNewRef 返回当前线程状态的帧对象的新引用
    return _Py_CAST(PyFrameObject *, Py_XNewRef(tstate->frame));
}
#endif


// 如果不是 PyPy 版本，定义 PYCAPI_COMPAT_STATIC_INLINE 宏，返回当前线程状态的帧对象指针
#if !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyFrameObject*)
_PyThreadState_GetFrameBorrow(PyThreadState *tstate)
{
    // 调用 PyThreadState_GetFrame 获取当前线程状态的帧对象
    PyFrameObject *frame = PyThreadState_GetFrame(tstate);
    // 释放帧对象的引用
    Py_XDECREF(frame);
    // 返回帧对象
    return frame;
}
#endif


// 如果 Python 版本小于 3.9.0a5 或者是 PyPy，定义 PYCAPI_COMPAT_STATIC_INLINE 宏，返回给定线程状态的解释器指针
#if PY_VERSION_HEX < 0x030900A5 || defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyInterpreterState *)
PyThreadState_GetInterpreter(PyThreadState *tstate)
{
    // 断言给定线程状态不为空
    assert(tstate != _Py_NULL);
    // 返回给定线程状态的解释器
    return tstate->interp;
}
#endif


// 如果 Python 版本小于 3.12.0a2 且不是 PyPy，定义 PYCAPI_COMPAT_STATIC_INLINE 宏，返回给定帧对象和变量名对应的对象指针
#if PY_VERSION_HEX < 0x030C00A2 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyFrame_GetVarString(PyFrameObject *frame, const char *name)
{
    PyObject *name_obj, *value;
    
    // 根据变量名创建 Unicode 对象
    name_obj = PyUnicode_FromString(name);
    if (name_obj == NULL) {
        return NULL;
    }
    // 调用 PyFrame_GetVar 获取给定帧对象和变量名对应的值对象
    value = PyFrame_GetVar(frame, name_obj);
    // 释放变量名对象的引用
    Py_DECREF(name_obj);
    // 返回值对象
    return value;
}
#endif


// 如果 Python 版本小于 3.11.0b1 且不是 PyPy，定义 PYCAPI_COMPAT_STATIC_INLINE 宏，返回给定帧对象的 f_lasti 属性值
#if PY_VERSION_HEX < 0x030B00B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyFrame_GetLasti(PyFrameObject *frame)
{
    // 如果 f_lasti 小于 0，则返回 -1
    if (frame->f_lasti < 0) {
        return -1;
    }
    // 如果 Python 版本大于等于 3.10.0a7，f_lasti 是指令偏移量，返回 f_lasti 的两倍
    return frame->f_lasti * 2;
    // 否则返回 f_lasti 的原始值
}
#endif
    # 如果线程状态 tstate 为 _Py_NULL，即 NULL 指针，发出致命错误信息并终止程序
    if (tstate == _Py_NULL) {
        Py_FatalError("GIL released (tstate is NULL)");
    }
    # 从线程状态 tstate 中获取解释器 interp
    interp = tstate->interp;
    # 如果解释器 interp 为 _Py_NULL，即 NULL 指针，发出致命错误信息并终止程序
    if (interp == _Py_NULL) {
        Py_FatalError("no current interpreter");
    }
    # 返回获取到的解释器 interp
    return interp;
// 如果 Python 版本大于等于 3.7.0a6 且小于 3.9.0a6，并且不是 PyPy 版本
#if 0x030700A1 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x030900A6 && !defined(PYPY_VERSION)
// 定义静态内联函数 PyThreadState_GetID，返回给定线程状态的 ID
PYCAPI_COMPAT_STATIC_INLINE(uint64_t)
PyThreadState_GetID(PyThreadState *tstate)
{
    // 断言给定的线程状态非空
    assert(tstate != _Py_NULL);
    // 返回线程状态的 ID
    return tstate->id;
}
#endif

// 如果 Python 版本小于 3.11.0a2，并且不是 PyPy 版本
#if PY_VERSION_HEX < 0x030B00A2 && !defined(PYPY_VERSION)
// 定义静态内联函数 PyThreadState_EnterTracing，开启给定线程状态的跟踪
PYCAPI_COMPAT_STATIC_INLINE(void)
PyThreadState_EnterTracing(PyThreadState *tstate)
{
    // 增加跟踪计数
    tstate->tracing++;
    // 根据 Python 版本设置使用跟踪的标志位
#if PY_VERSION_HEX >= 0x030A00A1
    tstate->cframe->use_tracing = 0;
#else
    tstate->use_tracing = 0;
#endif
}
#endif

// 如果 Python 版本小于 3.11.0a2，并且不是 PyPy 版本
#if PY_VERSION_HEX < 0x030B00A2 && !defined(PYPY_VERSION)
// 定义静态内联函数 PyThreadState_LeaveTracing，关闭给定线程状态的跟踪
PYCAPI_COMPAT_STATIC_INLINE(void)
PyThreadState_LeaveTracing(PyThreadState *tstate)
{
    // 检查是否有设置跟踪函数或者性能分析函数
    int use_tracing = (tstate->c_tracefunc != _Py_NULL
                       || tstate->c_profilefunc != _Py_NULL);
    // 减少跟踪计数
    tstate->tracing--;
    // 根据 Python 版本设置使用跟踪的标志位
#if PY_VERSION_HEX >= 0x030A00A1
    tstate->cframe->use_tracing = use_tracing;
#else
    tstate->use_tracing = use_tracing;
#endif
}
#endif

// 如果没有定义 PyObject_CallNoArgs，并且 Python 版本小于 3.9.0a1
#if !defined(PyObject_CallNoArgs) && PY_VERSION_HEX < 0x030900A1
// 定义静态内联函数 PyObject_CallNoArgs，调用无参数的 Python 对象方法
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyObject_CallNoArgs(PyObject *func)
{
    return PyObject_CallFunctionObjArgs(func, NULL);
}

// 定义静态内联函数 PyObject_CallMethodNoArgs，调用对象的无参数方法
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyObject_CallMethodNoArgs(PyObject *obj, PyObject *name)
{
    return PyObject_CallMethodObjArgs(obj, name, NULL);
}
#endif

// 如果没有定义 PyObject_CallOneArg，并且 Python 版本小于 3.9.0a4
#if !defined(PyObject_CallOneArg) && PY_VERSION_HEX < 0x030900A4
// 定义静态内联函数 PyObject_CallOneArg，调用带一个参数的 Python 对象方法
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyObject_CallOneArg(PyObject *func, PyObject *arg)
{
    return PyObject_CallFunctionObjArgs(func, arg, NULL);
}

// 定义静态内联函数 PyObject_CallMethodOneArg，调用对象的带一个参数的方法
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyObject_CallMethodOneArg(PyObject *obj, PyObject *name, PyObject *arg)
{
    return PyObject_CallMethodObjArgs(obj, name, arg, NULL);
}
#endif

// 如果 Python 版本小于 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3
// 定义静态内联函数 PyModule_AddObjectRef，在模块中添加对象引用
PYCAPI_COMPAT_STATIC_INLINE(int)
PyModule_AddObjectRef(PyObject *module, const char *name, PyObject *value)
{
    int res;
    // 增加对象引用计数
    Py_XINCREF(value);
    // 在模块中添加对象，并获取返回值
    res = PyModule_AddObject(module, name, value);
    // 如果添加失败，释放对象引用
    if (res < 0) {
        Py_XDECREF(value);
    }
    // 返回操作结果
    return res;
}
#endif

// 如果 Python 版本小于 3.9.0a5
#if PY_VERSION_HEX < 0x030900A5
// 定义静态内联函数 PyModule_AddType，在模块中添加类型对象
PYCAPI_COMPAT_STATIC_INLINE(int)
PyModule_AddType(PyObject *module, PyTypeObject *type)
{
    const char *name, *dot;

    // 如果类型对象准备失败，则返回错误
    if (PyType_Ready(type) < 0) {
        return -1;
    }

    // 内联函数 _PyType_Name()，获取类型对象的名称
    name = type->tp_name;
    // 断言类型名称非空
    assert(name != _Py_NULL);
    // 查找类型名称中的最后一个 '.'
    dot = strrchr(name, '.');
    # 如果 dot 不为空指针，则将 name 设置为 dot 指针后的位置（即名称中的点后面部分）
    if (dot != _Py_NULL) {
        name = dot + 1;
    }
    
    # 将 type 强制转换为 PyObject 类型，并将其作为 module 模块的对象引用添加到 name 对应的属性中
    return PyModule_AddObjectRef(module, name, _PyObject_CAST(type));
// 如果 Python 版本小于 3.9.0a6 且不是 PyPy，定义 PyObject_GC_IsTracked 函数
// bpo-40241 在 Python 3.9.0a6 中添加了 PyObject_GC_IsTracked()
#if PY_VERSION_HEX < 0x030900A6 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyObject_GC_IsTracked(PyObject* obj)
{
    // 检查对象是否启用了垃圾回收，并且是否被跟踪
    return (PyObject_IS_GC(obj) && _PyObject_GC_IS_TRACKED(obj));
}
#endif

// 如果 Python 版本小于 3.9.0a6 且大于等于 3.4.0 且不是 PyPy，定义 PyObject_GC_IsFinalized 函数
// bpo-40241 在 Python 3.9.0a6 中添加了 PyObject_GC_IsFinalized()
// bpo-18112 在 Python 3.4.0 final 中添加了 _PyGCHead_FINALIZED()
#if PY_VERSION_HEX < 0x030900A6 && PY_VERSION_HEX >= 0x030400F0 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyObject_GC_IsFinalized(PyObject *obj)
{
    // 将对象转换为 PyGC_Head，然后检查对象是否启用了垃圾回收并已经被终结
    PyGC_Head *gc = _Py_CAST(PyGC_Head*, obj) - 1;
    return (PyObject_IS_GC(obj) && _PyGCHead_FINALIZED(gc));
}
#endif

// 如果 Python 版本小于 3.9.0a4 且未定义 Py_IS_TYPE 宏，定义 _Py_IS_TYPE 函数及其宏
// bpo-39573 在 Python 3.9.0a4 中添加了 Py_IS_TYPE()
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_IS_TYPE)
PYCAPI_COMPAT_STATIC_INLINE(int)
_Py_IS_TYPE(PyObject *ob, PyTypeObject *type) {
    // 检查对象的类型是否与指定类型相匹配
    return Py_TYPE(ob) == type;
}
// 定义 Py_IS_TYPE 宏来调用 _Py_IS_TYPE 函数
#define Py_IS_TYPE(ob, type) _Py_IS_TYPE(_PyObject_CAST(ob), type)
#endif

// 如果 Python 版本在 3.6.0b1 到 3.11a1 之间且不是 PyPy，定义 PyFloat_Pack2 和 PyFloat_Unpack2 函数
// bpo-46906 在 Python 3.11a7 中添加了 PyFloat_Pack2 和 PyFloat_Unpack2()
// bpo-11734 在 Python 3.6.0b1 中添加了 _PyFloat_Pack2 和 _PyFloat_Unpack2()
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyFloat_Pack2(double x, char *p, int le)
{ return _PyFloat_Pack2(x, (unsigned char*)p, le); }

PYCAPI_COMPAT_STATIC_INLINE(double)
PyFloat_Unpack2(const char *p, int le)
{ return _PyFloat_Unpack2((const unsigned char *)p, le); }
#endif

// 如果 Python 版本小于等于 3.11a1 且不是 PyPy，定义 PyFloat_Pack4、PyFloat_Pack8、PyFloat_Unpack4 和 PyFloat_Unpack8 函数
// bpo-46906 在 Python 3.11a7 中添加了 PyFloat_Pack4、PyFloat_Pack8、PyFloat_Unpack4 和 PyFloat_Unpack8
// Python 3.11a2 将这些函数移到了内部 C API，3.11a2 到 3.11a6 版本不受支持
#if PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(int)
PyFloat_Pack4(double x, char *p, int le)
{ return _PyFloat_Pack4(x, (unsigned char*)p, le); }

PYCAPI_COMPAT_STATIC_INLINE(int)
PyFloat_Pack8(double x, char *p, int le)
{ return _PyFloat_Pack8(x, (unsigned char*)p, le); }

PYCAPI_COMPAT_STATIC_INLINE(double)
PyFloat_Unpack4(const char *p, int le)
{ return _PyFloat_Unpack4((const unsigned char *)p, le); }

PYCAPI_COMPAT_STATIC_INLINE(double)
PyFloat_Unpack8(const char *p, int le)
{ return _PyFloat_Unpack8((const unsigned char *)p, le); }
#endif

// 如果 Python 版本小于 3.11.0b1 且不是 PyPy，定义 PyCode_GetCode 函数
// gh-92154 在 Python 3.11.0b1 中添加了 PyCode_GetCode()
#if PY_VERSION_HEX < 0x030B00B1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyCode_GetCode(PyCodeObject *code)
{
    // 返回代码对象的代码部分的新引用
    return Py_NewRef(code->co_code);
}
#endif

// 如果 Python 版本小于 3.11.0rc1 且不是 PyPy，定义 PyCode_GetVarnames 函数
// gh-95008 在 Python 3.11.0rc1 中添加了 PyCode_GetVarnames()
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
// PyCode_GetVarnames 函数的实现，返回给定 PyCodeObject 结构体中 co_varnames 成员的引用
PyCode_GetVarnames(PyCodeObject *code)
{
    return Py_NewRef(code->co_varnames);
}
#endif

// Python 3.11.0rc1 中添加了 PyCode_GetFreevars() 函数，用于返回给定 PyCodeObject 结构体中 co_freevars 成员的引用
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyCode_GetFreevars(PyCodeObject *code)
{
    return Py_NewRef(code->co_freevars);
}
#endif

// Python 3.11.0rc1 中添加了 PyCode_GetCellvars() 函数，用于返回给定 PyCodeObject 结构体中 co_cellvars 成员的引用
#if PY_VERSION_HEX < 0x030B00C1 && !defined(PYPY_VERSION)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyCode_GetCellvars(PyCodeObject *code)
{
    return Py_NewRef(code->co_cellvars);
}
#endif

// 在 Python 3.4.0b2 中引入了 Py_UNUSED() 宏，用于标记未使用的函数参数
#if PY_VERSION_HEX < 0x030400B2 && !defined(Py_UNUSED)
#  if defined(__GNUC__) || defined(__clang__)
#    define Py_UNUSED(name) _unused_ ## name __attribute__((unused))
#  else
#    define Py_UNUSED(name) _unused_ ## name
#  endif
#endif

// Python 3.13.0a1 中添加了 PyImport_AddModuleRef() 函数，用于添加一个模块的引用并返回其 PyObject 对象的新引用
#if PY_VERSION_HEX < 0x030D00A0
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyImport_AddModuleRef(const char *name)
{
    return Py_XNewRef(PyImport_AddModule(name));
}
#endif

// Python 3.13.0a1 中添加了 PyWeakref_GetRef() 函数，用于获取弱引用指向的对象，并返回其新引用
#if PY_VERSION_HEX < 0x030D0000
PYCAPI_COMPAT_STATIC_INLINE(int)
PyWeakref_GetRef(PyObject *ref, PyObject **pobj)
{
    PyObject *obj;
    if (ref != NULL && !PyWeakref_Check(ref)) {
        *pobj = NULL;
        PyErr_SetString(PyExc_TypeError, "expected a weakref");
        return -1;
    }
    obj = PyWeakref_GetObject(ref);
    if (obj == NULL) {
        // 如果 ref 为 NULL 则抛出 SystemError
        *pobj = NULL;
        return -1;
    }
    if (obj == Py_None) {
        *pobj = NULL;
        return 0;
    }
    // 返回 obj 的新引用
    *pobj = Py_NewRef(obj);
    return (*pobj != NULL);
}
#endif

// 在 Python 3.8b1 中引入了 PY_VECTORCALL_ARGUMENTS_OFFSET 宏，用于获取向量调用参数的偏移量
#ifndef PY_VECTORCALL_ARGUMENTS_OFFSET
#  define PY_VECTORCALL_ARGUMENTS_OFFSET (_Py_CAST(size_t, 1) << (8 * sizeof(size_t) - 1))
#endif

// Python 3.8b1 中添加了 PyVectorcall_NARGS() 函数，用于从向量调用的参数数目中提取真实参数数目
#if PY_VERSION_HEX < 0x030800B1
static inline Py_ssize_t
PyVectorcall_NARGS(size_t n)
{
    return n & ~PY_VECTORCALL_ARGUMENTS_OFFSET;
}
#endif

// Python 3.9.0a4 中添加了 PyObject_Vectorcall() 函数，用于调用对象的向量调用函数
#if PY_VERSION_HEX < 0x030900A4
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
PyObject_Vectorcall(PyObject *callable, PyObject *const *args,
                     size_t nargsf, PyObject *kwnames)
{
#if PY_VERSION_HEX >= 0x030800B1 && !defined(PYPY_VERSION)
    // 在 Python 3.8.0b1 中添加了 _PyObject_Vectorcall() 函数，用于真正执行向量调用
    return _PyObject_Vectorcall(callable, args, nargsf, kwnames);
#else
    PyObject *posargs = NULL, *kwargs = NULL;
    PyObject *res;
    Py_ssize_t nposargs, nkwargs, i;

    if (nargsf != 0 && args == NULL) {
        PyErr_BadInternalCall();
        goto error;
    }
    if (kwnames != NULL && !PyTuple_Check(kwnames)) {
        PyErr_BadInternalCall();
        goto error;
    }

    // 解析向量调用的参数数目，计算出位置参数的个数
    nposargs = (Py_ssize_t)PyVectorcall_NARGS(nargsf);
    # 如果传入了关键字参数列表 kwnames，则获取其长度，否则设置为 0
    if (kwnames) {
        nkwargs = PyTuple_GET_SIZE(kwnames);
    }
    else {
        nkwargs = 0;
    }

    # 创建一个新的元组对象，用于存储位置参数
    posargs = PyTuple_New(nposargs);
    if (posargs == NULL) {
        goto error;
    }
    # 如果有位置参数，则逐个复制并增加引用计数
    if (nposargs) {
        for (i=0; i < nposargs; i++) {
            PyTuple_SET_ITEM(posargs, i, Py_NewRef(*args));
            args++;
        }
    }

    # 如果有关键字参数，则创建一个新的字典对象来存储关键字参数
    if (nkwargs) {
        kwargs = PyDict_New();
        if (kwargs == NULL) {
            goto error;
        }

        # 遍历关键字参数列表，将参数名和参数值对应存储到 kwargs 字典中
        for (i = 0; i < nkwargs; i++) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            PyObject *value = *args;
            args++;
            if (PyDict_SetItem(kwargs, key, value) < 0) {
                goto error;
            }
        }
    }
    else {
        kwargs = NULL;
    }

    # 调用给定的可调用对象 callable，传入位置参数和关键字参数，并获取返回值
    res = PyObject_Call(callable, posargs, kwargs);
    # 释放位置参数对象的引用
    Py_DECREF(posargs);
    # 释放关键字参数对象的引用，使用 Py_XDECREF 处理 NULL 指针
    Py_XDECREF(kwargs);
    # 返回调用结果对象
    return res;
error:
    // 递减 Python 对象的引用计数，释放对传递给函数的位置参数的引用
    Py_DECREF(posargs);
    // 释放对传递给函数的关键字参数的引用，即使它可能为 NULL
    Py_XDECREF(kwargs);
    // 返回 NULL 表示函数执行过程中发生错误
    return NULL;
#endif
}
#endif

#ifdef __cplusplus
}
#endif
#endif  // PYTHONCAPI_COMPAT
```