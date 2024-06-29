# `.\numpy\numpy\_core\src\common\cblasfuncs.h`

```py
#ifndef NUMPY_CORE_SRC_COMMON_CBLASFUNCS_H_
#define NUMPY_CORE_SRC_COMMON_CBLASFUNCS_H_

// 定义条件编译宏，用于避免重复包含该头文件
NPY_NO_EXPORT PyObject *
// 函数声明：计算矩阵乘积的CBLAS函数
cblas_matrixproduct(int, PyArrayObject *, PyArrayObject *, PyArrayObject *);

#endif  /* NUMPY_CORE_SRC_COMMON_CBLASFUNCS_H_ */
```