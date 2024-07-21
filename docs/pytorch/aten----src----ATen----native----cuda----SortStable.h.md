# `.\pytorch\aten\src\ATen\native\cuda\SortStable.h`

```py
#pragma once
// 使用 #pragma once 防止头文件被多次包含

#include <ATen/core/TensorBase.h>
// 包含 ATen 库的 TensorBase 类的头文件

#include <cstdint>
// 包含标准 C++ 头文件，定义了标准整数类型的宏

namespace at {
namespace native {

// 在 self 张量中稳定排序其值，并将 indices 设置为从 values 到 self 的反向排列
// 输出张量必须预先分配并且是连续的。
void launch_stable_sort_kernel(
    const TensorBase& self,        // 输入参数：要排序的张量 self
    int64_t dim,                   // 输入参数：排序的维度 dim
    bool descending,               // 输入参数：是否降序排序的标志
    const TensorBase& values,      // 输入参数：用于排序的值
    const TensorBase& indices);    // 输入参数：反向排列的索引

} // namespace native
} // namespace at


这段代码是 C++ 的头文件声明部分，定义了一个命名空间 `at::native` 中的函数 `launch_stable_sort_kernel`，用于在给定的张量中进行稳定排序，并返回排序后的值和反向排列的索引。
```