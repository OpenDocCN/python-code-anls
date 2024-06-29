# `.\numpy\numpy\_core\src\multiarray\temp_elide.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_
// 如果未定义 NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_ 宏，则执行以下内容
#define NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_
// 定义 NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_ 宏

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
// 定义 NPY_NO_DEPRECATED_API 宏为当前 NPY_API_VERSION，表示不使用已弃用的 API
#define _MULTIARRAYMODULE
// 定义 _MULTIARRAYMODULE 宏，用于某些条件编译

#include <numpy/ndarraytypes.h>
// 包含 numpy/ndarraytypes.h 头文件，提供对 NumPy 数组类型的支持

NPY_NO_EXPORT int
// NPY_NO_EXPORT 指示编译器不导出此函数符号
can_elide_temp_unary(PyArrayObject * m1);
// 函数声明：判断是否可以省略一元操作的临时变量

NPY_NO_EXPORT int
// NPY_NO_EXPORT 指示编译器不导出此函数符号
try_binary_elide(PyObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative);
// 函数声明：尝试省略二元操作的临时变量，支持就地操作和交换性操作

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_ */
// 结束条件编译指令，确保头文件内容只在未定义 NUMPY_CORE_SRC_MULTIARRAY_TEMP_ELIDE_H_ 宏时有效
```