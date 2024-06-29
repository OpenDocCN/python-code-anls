# `.\numpy\numpy\_core\src\umath\dispatching.h`

```py
#ifndef _NPY_DISPATCHING_H
#define _NPY_DISPATCHING_H

// 如果 _NPY_DISPATCHING_H 宏未定义，则定义它，用于防止头文件重复包含


#define _UMATHMODULE

// 定义 _UMATHMODULE 宏，用于标记当前为 ufunc 模块


#include <numpy/ufuncobject.h>
#include "array_method.h"

// 引入必要的头文件：<numpy/ufuncobject.h> 是 NumPy 中 ufunc 对象的头文件，
// "array_method.h" 是本地头文件，可能包含了一些数组方法的声明和定义


#ifdef __cplusplus
extern "C" {
#endif

// 如果是 C++ 编译环境，则声明下面的内容是用 C 语言编写的


NPY_NO_EXPORT int
PyUFunc_AddLoop(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate);

// 声明一个不导出的函数 PyUFunc_AddLoop，用于向给定的 ufunc 添加一个循环


NPY_NO_EXPORT int
PyUFunc_AddLoopFromSpec_int(PyObject *ufunc, PyArrayMethod_Spec *spec, int priv);

// 声明一个不导出的函数 PyUFunc_AddLoopFromSpec_int，从给定的规范 spec 中添加一个循环到 ufunc


NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool force_legacy_promotion,
        npy_bool allow_legacy_promotion,
        npy_bool promote_pyscalars,
        npy_bool ensure_reduce_compatible);

// 声明一个不导出的函数 promote_and_get_ufuncimpl，用于推广并获取 ufunc 的实现对象


NPY_NO_EXPORT PyObject *
add_and_return_legacy_wrapping_ufunc_loop(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *operation_dtypes[], int ignore_duplicate);

// 声明一个不导出的函数 add_and_return_legacy_wrapping_ufunc_loop，用于添加并返回具有遗留包装的 ufunc 循环


NPY_NO_EXPORT int
default_ufunc_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

// 声明一个不导出的函数 default_ufunc_promoter，用于设置默认的 ufunc 推广策略


NPY_NO_EXPORT int
object_only_ufunc_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

// 声明一个不导出的函数 object_only_ufunc_promoter，用于设置仅对象类型的 ufunc 推广策略


NPY_NO_EXPORT int
install_logical_ufunc_promoter(PyObject *ufunc);

// 声明一个不导出的函数 install_logical_ufunc_promoter，用于安装逻辑运算 ufunc 的推广策略


#ifdef __cplusplus
}
#endif

// 结束 C 语言函数声明的部分，如果是在 C++ 编译环境下，取消 C 风格的函数名修饰


#endif  /*_NPY_DISPATCHING_H */

// 结束 _NPY_DISPATCHING_H 头文件的条件编译部分
```