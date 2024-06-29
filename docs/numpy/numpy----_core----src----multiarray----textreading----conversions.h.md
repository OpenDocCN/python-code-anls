# `.\numpy\numpy\_core\src\multiarray\textreading\conversions.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_CONVERSIONS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_CONVERSIONS_H_

// 包含标准布尔类型的头文件
#include <stdbool.h>

// 定义宏，指定不使用废弃的 API 版本
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义宏，标记当前为多维数组模块
#define _MULTIARRAYMODULE
// 包含 NumPy 的数组对象头文件
#include "numpy/arrayobject.h"

// 包含文本解析器配置文件的头文件
#include "textreading/parser_config.h"

// 声明不导出的函数，用于将字符串转换为布尔类型
NPY_NO_EXPORT int
npy_to_bool(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

// 声明不导出的函数，用于将字符串转换为单精度浮点数类型
NPY_NO_EXPORT int
npy_to_float(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

// 声明不导出的函数，用于将字符串转换为双精度浮点数类型
NPY_NO_EXPORT int
npy_to_double(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

// 声明不导出的函数，用于将字符串转换为单精度复数类型
NPY_NO_EXPORT int
npy_to_cfloat(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

// 声明不导出的函数，用于将字符串转换为双精度复数类型
NPY_NO_EXPORT int
npy_to_cdouble(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

// 声明不导出的函数，用于将字符串转换为字符串类型
NPY_NO_EXPORT int
npy_to_string(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused);

// 声明不导出的函数，用于将字符串转换为 Unicode 类型
NPY_NO_EXPORT int
npy_to_unicode(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused);

// 声明不导出的函数，用于通过自定义转换器将字符串转换为通用类型
NPY_NO_EXPORT int
npy_to_generic_with_converter(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused, PyObject *func);

// 声明不导出的函数，用于将字符串通过默认转换器转换为通用类型
NPY_NO_EXPORT int
npy_to_generic(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_CONVERSIONS_H_ */


注释：
```