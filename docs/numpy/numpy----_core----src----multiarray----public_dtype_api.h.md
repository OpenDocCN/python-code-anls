# `.\numpy\numpy\_core\src\multiarray\public_dtype_api.h`

```py
/*
 * This file exports the private function that exposes the DType API
 *
 * This file is a stub, all important definitions are in the code file.
 */

#ifndef NUMPY_CORE_SRC_MULTIARRAY_EXPERIMENTAL_PUBLIC_DTYPE_API_H_
#define NUMPY_CORE_SRC_MULTIARRAY_EXPERIMENTAL_PUBLIC_DTYPE_API_H_

// 声明一个不导出的函数 _fill_dtype_api，用于暴露 DType API
NPY_NO_EXPORT void
_fill_dtype_api(void *numpy_api_table[]);

// 结束条件预处理指令，确保头文件只被包含一次
#endif  /* NUMPY_CORE_SRC_MULTIARRAY_EXPERIMENTAL_PUBLIC_DTYPE_API_H_ */
```