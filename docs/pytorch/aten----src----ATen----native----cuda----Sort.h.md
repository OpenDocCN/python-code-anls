# `.\pytorch\aten\src\ATen\native\cuda\Sort.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <cstdint>
// 包含标准整数类型的头文件

#include <ATen/core/TensorBase.h>
// 包含 ATen 库中的 TensorBase 类的头文件

#include <ATen/native/cuda/SortStable.h>
// 包含 ATen 库中 CUDA 相关的稳定排序函数的头文件

namespace at {
namespace native {

inline bool should_use_small_sort(const TensorBase &self, int64_t dim) {
  // 内联函数，判断是否应该使用小规模排序的标准
  return self.size(dim) <= 4096;
}

void sortKeyValueInplace(
    const TensorBase &key, const TensorBase &value, int dim,
    bool descending, bool stable=false);
// 排序关键-值对的函数声明，支持在指定维度上进行排序，可选择降序和稳定排序

}}  // namespace at::native
// 命名空间结束声明
```