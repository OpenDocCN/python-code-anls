# `.\numpy\numpy\_core\src\common\umathmodule.h`

```
#ifndef NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_
#define NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_

// 包含以下头文件，这些头文件定义了ufunc对象、ufunc类型解析和Python侧的扩展对象的设置/获取
#include "ufunc_object.h"
#include "ufunc_type_resolution.h"
#include "extobj.h"  /* for the python side extobj set/get */

// NPY_NO_EXPORT表示这个函数不会被导出到模块外部
NPY_NO_EXPORT PyObject *
get_sfloat_dtype(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(args));

// 添加新的文档字符串到ufunc对象中
PyObject * add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args);

// 根据给定的Python函数创建一个ufunc对象
PyObject * ufunc_frompyfunc(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *NPY_UNUSED(kwds));

// 初始化umath模块，返回一个整数作为初始化的结果
int initumath(PyObject *m);

#endif  /* NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_ */
```