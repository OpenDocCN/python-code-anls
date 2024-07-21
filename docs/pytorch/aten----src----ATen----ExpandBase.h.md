# `.\pytorch\aten\src\ATen\ExpandBase.h`

```py
#include <ATen/core/TensorBase.h>

// Broadcasting utilities for working with TensorBase

// ATen 命名空间，包含了内部实用函数和类
namespace at {

// internal 命名空间，包含了 TensorBase 类的内部实用函数
namespace internal {

// 使用 TensorBase 类的引用 self 和 size 数组，执行扩展操作的慢速路径
TORCH_API TensorBase expand_slow_path(const TensorBase& self, IntArrayRef size);

} // namespace internal

// 在给定的 TensorBase 对象 self 上扩展到指定的 size
inline c10::MaybeOwned<TensorBase> expand_size(
    const TensorBase& self,
    IntArrayRef size) {
  // 如果指定的 size 与 self 的尺寸相同，则直接返回对 self 的引用
  if (size.equals(self.sizes())) {
    return c10::MaybeOwned<TensorBase>::borrowed(self);
  }
  // 否则，执行慢速扩展路径，并返回一个拥有新 TensorBase 对象的 MaybeOwned 包装
  return c10::MaybeOwned<TensorBase>::owned(
      at::internal::expand_slow_path(self, size));
}

// 禁止在 TensorBase&& 上执行 expand_size 操作
c10::MaybeOwned<TensorBase> expand_size(TensorBase&& self, IntArrayRef size) =
    delete;

// 在给定的 tensor 和 to_expand 上执行原地扩展
inline c10::MaybeOwned<TensorBase> expand_inplace(
    const TensorBase& tensor,
    const TensorBase& to_expand) {
  // 使用 tensor 的尺寸扩展 to_expand
  return expand_size(to_expand, tensor.sizes());
}

// 禁止在 TensorBase&& 上执行 expand_inplace 操作
c10::MaybeOwned<TensorBase> expand_inplace(
    const TensorBase& tensor,
    TensorBase&& to_expand) = delete;

} // namespace at


这段代码定义了一些与 TensorBase 类型相关的广播（Broadcasting）实用程序函数。在 C++ 中，这些函数用于处理张量（tensor）对象的扩展操作，以便适应不同的尺寸需求。
```