# `.\pytorch\aten\src\ATen\native\cuda\Sorting.h`

```py
#pragma once
#include <cstdint>  // 包含标准整数类型的头文件

namespace at {
class TensorBase;  // 声明 TensorBase 类

}

namespace at {
namespace native {

void launch_kthvalue_kernel(
    const TensorBase &values, const TensorBase &indices,  // 启动第k个值计算的核心函数，接受值和索引张量作为参数
    const TensorBase &self, int64_t dim, int64_t k);  // 启动第k个值计算的核心函数，接受自身张量和维度索引作为参数

void launch_median_kernel(
    const TensorBase &vals, const TensorBase &inds,  // 启动中位数计算的核心函数，接受值和索引张量作为参数
    const TensorBase &in, int64_t dim, bool ignore_nan);  // 启动中位数计算的核心函数，接受输入张量、维度索引和是否忽略NaN值作为参数

}}  // namespace at::native
```