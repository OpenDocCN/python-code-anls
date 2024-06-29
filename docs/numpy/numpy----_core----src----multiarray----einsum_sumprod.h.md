# `.\numpy\numpy\_core\src\multiarray\einsum_sumprod.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY_EINSUM_SUMPROD_H_
#define NUMPY_CORE_SRC_MULTIARRAY_EINSUM_SUMPROD_H_

#include <numpy/npy_common.h>

// 定义函数指针类型 sum_of_products_fn，用于表示一个函数指针，该函数接受四个参数：
//   - int，参数个数
//   - char **，参数数组
//   - npy_intp const*，整数数组，常量指针
//   - npy_intp，整数
typedef void (*sum_of_products_fn)(int, char **, npy_intp const*, npy_intp);

// 声明一个隐藏（visibility hidden）的函数 get_sum_of_products_function，返回类型为 sum_of_products_fn，
// 接受如下参数：
//   - int，操作数个数
//   - int，类型编号
//   - npy_intp，项大小
//   - npy_intp const*，固定步长的整数数组，常量指针
NPY_VISIBILITY_HIDDEN sum_of_products_fn
get_sum_of_products_function(int nop, int type_num,
                             npy_intp itemsize, npy_intp const *fixed_strides);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_EINSUM_SUMPROD_H_ */
```