# `.\numpy\numpy\_core\src\umath\legacy_array_method.h`

```py
#ifndef _NPY_LEGACY_ARRAY_METHOD_H
#define _NPY_LEGACY_ARRAY_METHOD_H

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "array_method.h"

#ifdef __cplusplus
extern "C" {
#endif

// 定义了一个新的类型 PyArrayMethodObject，用于表示数组方法对象
NPY_NO_EXPORT PyArrayMethodObject *
PyArray_NewLegacyWrappingArrayMethod(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *signature[]);

/*
 * 下面两个符号在头文件中定义，以便其他地方可以使用它们来探测特殊情况
 * （或者一个 ArrayMethod 是否为 "legacy" 类型）。
 */

// 获取被包装的 legacy ufunc 循环函数的信息，并返回相关参数
NPY_NO_EXPORT int
get_wrapped_legacy_ufunc_loop(PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

// 解析 legacy 类型数组方法的描述符，返回最终的数据类型和形状信息
NPY_NO_EXPORT NPY_CASTING
wrapped_legacy_resolve_descriptors(PyArrayMethodObject *,
        PyArray_DTypeMeta *const *, PyArray_Descr *const *, PyArray_Descr **, npy_intp *);

#ifdef __cplusplus
}
#endif

#endif  /*_NPY_LEGACY_ARRAY_METHOD_H */
```