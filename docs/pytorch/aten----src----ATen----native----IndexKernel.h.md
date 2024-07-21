# `.\pytorch\aten\src\ATen\native\IndexKernel.h`

```
#pragma once
// 防止头文件被多次包含，保证头文件内容只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库中的 DispatchStub.h 头文件，用于声明调度相关的功能

#include <c10/util/ArrayRef.h>
// 包含 c10 库中的 ArrayRef.h 头文件，用于定义 ArrayRef 类型和相关工具函数

namespace at {
class Tensor;
class TensorBase;
struct TensorIterator;
struct TensorIteratorBase;
}
// 命名空间 at 中声明 Tensor、TensorBase、TensorIterator 和 TensorIteratorBase 类或结构体

namespace c10 {
class Scalar;
}
// 命名空间 c10 中声明 Scalar 类

namespace at::native {

using index_fn = void(*)(TensorIteratorBase &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
// 定义 index_fn 为函数指针类型，接受 TensorIteratorBase 引用、IntArrayRef indexed_sizes 和 indexed_strides 作为参数

using index_fill_fn = void(*)(TensorIterator & iter, int64_t dim, int64_t self_dim_size, int64_t self_dim_stride, const Scalar& source);
// 定义 index_fill_fn 为函数指针类型，接受 TensorIterator 引用、dim、self_dim_size、self_dim_stride 和 Scalar 引用 source 作为参数

using index_copy_fn = void(*)(TensorIterator & iter, int64_t dim, int64_t self_dim_size, int64_t self_dim_stride);
// 定义 index_copy_fn 为函数指针类型，接受 TensorIterator 引用、dim、self_dim_size 和 self_dim_stride 作为参数

using index_put_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides, bool accumulate);
// 定义 index_put_fn 为函数指针类型，接受 TensorIterator 引用、IntArrayRef indexed_sizes、indexed_strides 和布尔值 accumulate 作为参数

using put_fn = void(*)(TensorIterator & iter, const TensorBase& self, const bool accumulate);
// 定义 put_fn 为函数指针类型，接受 TensorIterator 引用、TensorBase 引用 self 和布尔值 accumulate 作为参数

using take_fn = void(*)(TensorIterator & iter, const TensorBase& input);
// 定义 take_fn 为函数指针类型，接受 TensorIterator 引用和 TensorBase 引用 input 作为参数

using flip_fn = void(*)(TensorIterator &, const bool);
// 定义 flip_fn 为函数指针类型，接受 TensorIterator 引用和布尔值作为参数

using masked_fill_fn = void(*)(TensorIterator &, const Scalar& scalar);
// 定义 masked_fill_fn 为函数指针类型，接受 TensorIterator 引用和 Scalar 引用 scalar 作为参数

using masked_select_fn = void(*)(TensorIterator &, int64_t orig_stride);
// 定义 masked_select_fn 为函数指针类型，接受 TensorIterator 引用和 orig_stride 作为参数

using masked_scatter_fn = void(*)(TensorIterator &, const TensorBase &);
// 定义 masked_scatter_fn 为函数指针类型，接受 TensorIterator 引用和 TensorBase 引用作为参数

DECLARE_DISPATCH(index_fn, index_stub);
// 声明 index_stub 函数，使用 index_fn 类型作为调度分发器

DECLARE_DISPATCH(index_fill_fn, index_fill_stub);
// 声明 index_fill_stub 函数，使用 index_fill_fn 类型作为调度分发器

DECLARE_DISPATCH(index_copy_fn, index_copy_stub);
// 声明 index_copy_stub 函数，使用 index_copy_fn 类型作为调度分发器

DECLARE_DISPATCH(index_put_fn, index_put_stub);
// 声明 index_put_stub 函数，使用 index_put_fn 类型作为调度分发器

DECLARE_DISPATCH(put_fn, put_stub);
// 声明 put_stub 函数，使用 put_fn 类型作为调度分发器

DECLARE_DISPATCH(take_fn, take_stub);
// 声明 take_stub 函数，使用 take_fn 类型作为调度分发器

DECLARE_DISPATCH(flip_fn, flip_stub);
// 声明 flip_stub 函数，使用 flip_fn 类型作为调度分发器

DECLARE_DISPATCH(masked_fill_fn, masked_fill_stub);
// 声明 masked_fill_stub 函数，使用 masked_fill_fn 类型作为调度分发器

DECLARE_DISPATCH(masked_select_fn, masked_select_serial_stub);
// 声明 masked_select_serial_stub 函数，使用 masked_select_fn 类型作为调度分发器

DECLARE_DISPATCH(masked_select_fn, masked_select_stub);
// 声明 masked_select_stub 函数，使用 masked_select_fn 类型作为调度分发器

DECLARE_DISPATCH(masked_scatter_fn, masked_scatter_stub);
// 声明 masked_scatter_stub 函数，使用 masked_scatter_fn 类型作为调度分发器

} // namespace at::native
// 结束命名空间 at::native
```