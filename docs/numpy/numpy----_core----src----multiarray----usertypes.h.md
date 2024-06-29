# `.\numpy\numpy\_core\src\multiarray\usertypes.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_USERTYPES_H_
#define NUMPY_CORE_SRC_MULTIARRAY_USERTYPES_H_

#include "array_method.h"  // 包含另一个头文件 "array_method.h"

extern NPY_NO_EXPORT _PyArray_LegacyDescr **userdescrs;  // 声明一个名为 userdescrs 的外部全局变量，类型为 _PyArray_LegacyDescr**，在其他文件中可见

// 初始化数组函数指针数组结构体
NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f);

// 注册能够进行类型转换的方法
NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar);

// 注册数据类型描述符
NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_DescrProto *descr);

// 注册类型转换函数
NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc);

// 用于确定两个用户定义数据类型的公共数据类型
NPY_NO_EXPORT PyArray_DTypeMeta *
legacy_userdtype_common_dtype_function(
        PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other);

// 添加旧版包装的转换实现
NPY_NO_EXPORT int
PyArray_AddLegacyWrapping_CastingImpl(
        PyArray_DTypeMeta *from, PyArray_DTypeMeta *to, NPY_CASTING casting);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_USERTYPES_H_ */
```