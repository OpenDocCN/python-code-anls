# `.\numpy\numpy\_core\src\common\npy_longdouble.h`

```py
#ifndef NUMPY_CORE_SRC_COMMON_NPY_LONGDOUBLE_H_
#define NUMPY_CORE_SRC_COMMON_NPY_LONGDOUBLE_H_

#include "npy_config.h"  // 包含 numpy 配置信息的头文件
#include "numpy/ndarraytypes.h"  // 包含 numpy 数组类型相关的头文件

/* Convert a npy_longdouble to a python `long` integer.
 *
 * Results are rounded towards zero.
 *
 * This performs the same task as PyLong_FromDouble, but for long doubles
 * which have a greater range.
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_longdouble_to_PyLong(npy_longdouble ldval);  // 将 npy_longdouble 类型转换为 Python 的 long 整数对象

/* Convert a python `long` integer to a npy_longdouble
 *
 * This performs the same task as PyLong_AsDouble, but for long doubles
 * which have a greater range.
 *
 * Returns -1 if an error occurs.
 */
NPY_VISIBILITY_HIDDEN npy_longdouble
npy_longdouble_from_PyLong(PyObject *long_obj);  // 将 Python 的 long 整数对象转换为 npy_longdouble 类型

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_LONGDOUBLE_H_ */
```