# `.\pytorch\aten\src\ATen\CachedTensorUtils.h`

```py
#pragma once
// 指令，确保头文件只被包含一次

#include <ATen/ATen.h>
// 包含 ATen 库的头文件

namespace at::caching {

// 命名空间开始：ATen 库中的缓存相关功能

// 某些系统（目前仅限 cudagraphs）将持久化一个静态张量输出，其 TensorImpl 在迭代中不会改变。
// 对于这些张量，缓存 dtype 转换是无效的。此外，这些缓存张量会有额外的引用计数，阻止缓冲区就地替换和张量唯一性的其他检查。
// 如果我们不使用这些系统，enabled 标志将为 false，并且我们将避免进行哈希查找。
TORCH_API bool is_cached_tensor(const at::Tensor& t);
// 函数声明：检查给定张量是否被缓存

TORCH_API void add_cached_tensor(const at::Tensor& t);
// 函数声明：向缓存中添加给定张量

TORCH_API void remove_cached_tensor(const at::Tensor& t);
// 函数声明：从缓存中移除给定张量

TORCH_API void set_cached_tensors_enabled(bool enable);
// 函数声明：设置是否启用缓存张量的标志

// 对于梯度缓冲区窃取，我们将调整 cudagraphs 持久化的张量的使用计数，就像我们需要调整带有钩子的张量的引用计数一样。
TORCH_API size_t adjusted_use_count(const at::Tensor& t);
// 函数声明：调整给定张量的使用计数

} // namespace at::caching
// 命名空间结束：ATen 库中的缓存相关功能
```