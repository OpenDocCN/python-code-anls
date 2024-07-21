# `.\pytorch\aten\src\ATen\native\sparse\SparseStubs.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/native/DispatchStub.h>
// 包含 ATen 库的 DispatchStub.h 头文件，用于分发机制

#include <c10/util/ArrayRef.h>
// 包含 c10 库的 ArrayRef.h 头文件，用于 ArrayRef 类型的支持

#include <c10/util/Optional.h>
// 包含 c10 库的 Optional.h 头文件，用于 Optional 类型的支持

namespace at {
// at 命名空间开始

class Tensor;
// 前置声明 Tensor 类

namespace native {
// native 命名空间开始

using mul_sparse_sparse_out_fn = void (*)(Tensor& res, const Tensor& x, const Tensor& y);
// 定义 mul_sparse_sparse_out_fn 类型别名，表示一个函数指针类型，接受三个 Tensor 参数并返回 void

DECLARE_DISPATCH(mul_sparse_sparse_out_fn, mul_sparse_sparse_out_stub);
// 声明 mul_sparse_sparse_out_stub 函数，通过分发机制调用 mul_sparse_sparse_out_fn 类型的函数

using sparse_mask_intersection_out_fn = void (*)(Tensor& res, const Tensor& x, const Tensor& y, const std::optional<Tensor>& x_hash_opt);
// 定义 sparse_mask_intersection_out_fn 类型别名，表示一个函数指针类型，接受四个参数（三个 Tensor 和一个 std::optional<Tensor>）并返回 void

DECLARE_DISPATCH(sparse_mask_intersection_out_fn, sparse_mask_intersection_out_stub);
// 声明 sparse_mask_intersection_out_stub 函数，通过分发机制调用 sparse_mask_intersection_out_fn 类型的函数

using sparse_mask_projection_out_fn = void (*)(Tensor& res, const Tensor& x, const Tensor& y, const std::optional<Tensor>& x_hash_opt, bool accumulate_matches);
// 定义 sparse_mask_projection_out_fn 类型别名，表示一个函数指针类型，接受五个参数（三个 Tensor、一个 std::optional<Tensor> 和一个 bool）并返回 void

DECLARE_DISPATCH(sparse_mask_projection_out_fn, sparse_mask_projection_out_stub);
// 声明 sparse_mask_projection_out_stub 函数，通过分发机制调用 sparse_mask_projection_out_fn 类型的函数

using flatten_indices_fn = Tensor (*)(const Tensor& indices, IntArrayRef size);
// 定义 flatten_indices_fn 类型别名，表示一个函数指针类型，接受两个参数（一个 Tensor 和一个 IntArrayRef）并返回 Tensor

DECLARE_DISPATCH(flatten_indices_fn, flatten_indices_stub);
// 声明 flatten_indices_stub 函数，通过分发机制调用 flatten_indices_fn 类型的函数

} // namespace native
} // namespace at
// native 和 at 命名空间结束
```