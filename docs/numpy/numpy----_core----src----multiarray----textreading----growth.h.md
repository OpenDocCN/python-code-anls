# `.\numpy\numpy\_core\src\multiarray\textreading\growth.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_GROWTH_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_GROWTH_H_

#ifdef __cplusplus
extern "C" {
#endif

// NPY_NO_EXPORT 宏定义用于标记函数或变量不会被动态链接到共享库中
// grow_size_and_multiply 函数用于增长数组大小并计算乘积，返回结果为增长后的大小
NPY_NO_EXPORT npy_intp
grow_size_and_multiply(npy_intp *size, npy_intp min_grow, npy_intp itemsize);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_GROWTH_H_ */
```