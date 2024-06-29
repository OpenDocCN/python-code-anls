# `.\numpy\numpy\_core\src\multiarray\convert_datatype.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_

#include "array_method.h"  // 包含数组方法的头文件

#ifdef __cplusplus
extern "C" {
#endif

extern NPY_NO_EXPORT npy_intp REQUIRED_STR_LEN[];  // 声明一个外部的 npy_intp 数组 REQUIRED_STR_LEN

#define NPY_USE_LEGACY_PROMOTION 0  // 定义宏 NPY_USE_LEGACY_PROMOTION 并赋值为 0
#define NPY_USE_WEAK_PROMOTION 1    // 定义宏 NPY_USE_WEAK_PROMOTION 并赋值为 1
#define NPY_USE_WEAK_PROMOTION_AND_WARN 2  // 定义宏 NPY_USE_WEAK_PROMOTION_AND_WARN 并赋值为 2

NPY_NO_EXPORT int
npy_give_promotion_warnings(void);  // 声明一个不导出的函数 npy_give_promotion_warnings，返回 int

NPY_NO_EXPORT PyObject *
npy__get_promotion_state(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(arg));  // 声明一个不导出的函数 npy__get_promotion_state，返回 PyObject*

NPY_NO_EXPORT PyObject *
npy__set_promotion_state(PyObject *NPY_UNUSED(mod), PyObject *arg);  // 声明一个不导出的函数 npy__set_promotion_state，返回 PyObject*

NPY_NO_EXPORT PyObject *
PyArray_GetCastingImpl(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to);  // 声明一个不导出的函数 PyArray_GetCastingImpl，返回 PyObject*

NPY_NO_EXPORT PyObject *
_get_castingimpl(PyObject *NPY_UNUSED(module), PyObject *args);  // 声明一个不导出的函数 _get_castingimpl，返回 PyObject*

NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num);  // 声明一个不导出的函数 PyArray_GetCastFunc，返回 PyArray_VectorUnaryFunc*

NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type);  // 声明一个不导出的函数 PyArray_ObjectType，返回 int

NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn);  // 声明一个不导出的函数 PyArray_ConvertToCommonType，返回 PyArrayObject**

NPY_NO_EXPORT PyArray_Descr *
PyArray_CastToDTypeAndPromoteDescriptors(
        npy_intp ndescr, PyArray_Descr *descrs[], PyArray_DTypeMeta *DType);  // 声明一个不导出的函数 PyArray_CastToDTypeAndPromoteDescriptors，返回 PyArray_Descr*

NPY_NO_EXPORT int
PyArray_CheckLegacyResultType(
        PyArray_Descr **new_result,
        npy_intp narrs, PyArrayObject **arr,
        npy_intp ndtypes, PyArray_Descr **dtypes);  // 声明一个不导出的函数 PyArray_CheckLegacyResultType，返回 int

NPY_NO_EXPORT int
PyArray_ValidType(int type);  // 声明一个不导出的函数 PyArray_ValidType，返回 int

NPY_NO_EXPORT int
dtype_kind_to_ordering(char kind);  // 声明一个不导出的函数 dtype_kind_to_ordering，返回 int

/* Used by PyArray_CanCastArrayTo and in the legacy ufunc type resolution */
NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data,
                   PyArray_Descr *to, NPY_CASTING casting);  // 声明一个不导出的函数 can_cast_scalar_to，返回 npy_bool

NPY_NO_EXPORT npy_bool
can_cast_pyscalar_scalar_to(
        int flags, PyArray_Descr *to, NPY_CASTING casting);  // 声明一个不导出的函数 can_cast_pyscalar_scalar_to，返回 npy_bool

NPY_NO_EXPORT int
should_use_min_scalar(npy_intp narrs, PyArrayObject **arr,
                      npy_intp ndtypes, PyArray_Descr **dtypes);  // 声明一个不导出的函数 should_use_min_scalar，返回 int

NPY_NO_EXPORT int
should_use_min_scalar_weak_literals(int narrs, PyArrayObject **arr);  // 声明一个不导出的函数 should_use_min_scalar_weak_literals，返回 int

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);  // 声明一个不导出的函数 npy_casting_to_string，返回 const char*

NPY_NO_EXPORT void
npy_set_invalid_cast_error(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar);  // 声明一个不导出的函数 npy_set_invalid_cast_error，返回 void

NPY_NO_EXPORT PyArray_Descr *
PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType);  // 声明一个不导出的函数 PyArray_CastDescrToDType，返回 PyArray_Descr*

NPY_NO_EXPORT PyArray_Descr *
PyArray_FindConcatenationDescriptor(
        npy_intp n, PyArrayObject **arrays, PyArray_Descr *requested_dtype);  // 声明一个不导出的函数 PyArray_FindConcatenationDescriptor，返回 PyArray_Descr*

NPY_NO_EXPORT int
PyArray_AddCastingImplementation(PyBoundArrayMethodObject *meth);  // 声明一个不导出的函数 PyArray_AddCastingImplementation，返回 int

NPY_NO_EXPORT int
PyArray_AddCastingImplementation_FromSpec(PyArrayMethod_Spec *spec, int private_);  // 声明一个不导出的函数 PyArray_AddCastingImplementation_FromSpec，返回 int

NPY_NO_EXPORT NPY_CASTING
PyArray_MinCastSafety(NPY_CASTING casting1, NPY_CASTING casting2);  // 声明一个不导出的函数 PyArray_MinCastSafety，返回 NPY_CASTING

NPY_NO_EXPORT NPY_CASTING
PyArray_GetCastInfo(
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype,
        npy_intp *view_offset);  // 声明一个不导出的函数 PyArray_GetCastInfo，返回 NPY_CASTING

NPY_NO_EXPORT npy_intp
# 安全地将一个数组从一个数据类型转换为另一个数据类型，返回转换后的结果数组
PyArray_SafeCast(PyArray_Descr *type1, PyArray_Descr *type2,
                 npy_intp* view_offset, NPY_CASTING minimum_safety,
                 npy_intp ignore_errors);

# 检查在给定的转换方式下，是否可以安全地将一个数据类型从一种类型转换为另一种类型，返回整数结果
NPY_NO_EXPORT int
PyArray_CheckCastSafety(NPY_CASTING casting,
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype);

# 在处理遗留代码时，解析两个描述符以确定它们是否具有相同的数据类型，返回转换的安全性级别
NPY_NO_EXPORT NPY_CASTING
legacy_same_dtype_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset);

# 获取一个支持遗留转换的循环函数，返回整数结果表示成功与否
NPY_NO_EXPORT int
legacy_cast_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

# 在简单的类型转换中，解析两个描述符以确定它们是否可以安全地转换，返回转换的安全性级别
NPY_NO_EXPORT NPY_CASTING
simple_cast_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const input_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset);

# 初始化数组对象的类型转换机制，返回整数结果表示成功与否
NPY_NO_EXPORT int
PyArray_InitializeCasts(void);

# 获取当前 NumPy 类型提升的状态，返回整数表示当前状态
NPY_NO_EXPORT int
get_npy_promotion_state();

# 设置新的 NumPy 类型提升状态，无返回值
NPY_NO_EXPORT void
set_npy_promotion_state(int new_promotion_state);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_ */
```