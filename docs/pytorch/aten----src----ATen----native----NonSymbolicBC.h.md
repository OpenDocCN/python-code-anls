# `.\pytorch\aten\src\ATen\native\NonSymbolicBC.h`

```
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <ATen/core/Tensor.h>
// 引入 ATen 库中的 Tensor 类头文件
#include <c10/util/irange.h>
// 引入 c10 库中的 irange 函数
#include <ATen/core/IListRef.h>
// 引入 ATen 库中的 IListRef 类头文件

namespace at::native {
// 进入 at::native 命名空间

// This file contains non-symbolic signatures for ops that we have sym-intified the signature of.
// However, in certain cases (such as static runtime), we call the native versions of the ops directly.
// In those cases, we will duplicate the signature here with non-symbolic ints, and also duplicate the C++ implementation.

// 声明 reshape 函数，接受一个 Tensor 类型的引用 self 和一个 proposed_shape 参数，返回一个 Tensor 类型对象
TORCH_API at::Tensor reshape(const at::Tensor& self, at::IntArrayRef proposed_shape);

// 声明 narrow 函数，接受一个 Tensor 类型的引用 self，以及 int64_t 类型的 dim、start 和 length 参数，返回一个 Tensor 类型对象
TORCH_API at::Tensor narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length);

// 声明 _sparse_coo_tensor_unsafe 函数，接受两个 Tensor 类型的引用 indices 和 values，以及一个 IntArrayRef 类型的 size 参数，
// 可选的 ScalarType、Layout、Device、pin_memory、is_coalesced 参数，返回一个 Tensor 类型对象
TORCH_API at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size,
                                               std::optional<at::ScalarType> dtype=c10::nullopt,
                                               std::optional<at::Layout> layout=c10::nullopt,
                                               std::optional<at::Device> device=c10::nullopt,
                                               std::optional<bool> pin_memory=c10::nullopt,
                                               std::optional<bool> is_coalesced=c10::nullopt);

// 声明 nll_loss 函数，接受三个 Tensor 类型的引用 self、target 和 weight_opt，以及两个 int64_t 类型的参数 reduction 和 ignore_index，
// 返回一个 Tensor 类型对象
TORCH_API at::Tensor nll_loss(const at::Tensor & self, const at::Tensor & target, const std::optional<at::Tensor>& weight_opt,
                              int64_t reduction, int64_t ignore_index);

// 声明 nll_loss2d 函数，接受三个 Tensor 类型的引用 self、target 和 weight_opt，以及两个 int64_t 类型的参数 reduction 和 ignore_index，
// 返回一个 Tensor 类型对象
TORCH_API at::Tensor nll_loss2d(const at::Tensor & self, const at::Tensor & target, const std::optional<at::Tensor>& weight_opt,
                                int64_t reduction, int64_t ignore_index);

// The below ops don't get a duplicated C++ implementation.
// They are backward ops, which make them very unlikely to be called directly
// by external code (at::native::trace_backward).
// They get their own declaration for BC purposes however.

// 声明 _embedding_bag_backward 函数，接受多个 Tensor 类型的引用 grad、indices、offsets、offset2bag、bag_size 和 maximum_indices，
// 以及多个 int64_t 类型的参数 num_weights、mode 和 padding_idx，bool 类型的参数 scale_grad_by_freq 和 sparse，可选的 Tensor 类型参数 per_sample_weights，
// 返回一个 Tensor 类型对象
TORCH_API at::Tensor _embedding_bag_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets,
                                             const at::Tensor & offset2bag, const at::Tensor & bag_size,
                                             const at::Tensor & maximum_indices, int64_t num_weights,
                                             bool scale_grad_by_freq, int64_t mode, bool sparse,
                                             const std::optional<at::Tensor> & per_sample_weights,
                                             int64_t padding_idx=-1);

// 声明 _embedding_bag_sparse_backward 函数，接受多个 Tensor 类型的引用 grad、indices、offsets、offset2bag 和 bag_size，
// 以及多个 int64_t 类型的参数 num_weights、mode 和 padding_idx，bool 类型的参数 scale_grad_by_freq，可选的 Tensor 类型参数 per_sample_weights，
// 返回一个 Tensor 类型对象
TORCH_API at::Tensor _embedding_bag_sparse_backward(const at::Tensor & grad, const at::Tensor & indices,
                                                    const at::Tensor & offsets, const at::Tensor & offset2bag,
                                                    const at::Tensor & bag_size, int64_t num_weights,
                                                    bool scale_grad_by_freq, int64_t mode,
                                                    const std::optional<at::Tensor> & per_sample_weights,
                                                    int64_t padding_idx=-1);

// 声明 value_selecting_reduction_backward 函数，接受一个 Tensor 类型的引用 grad，一个 int64_t 类型的 dim 参数，
// 一个 Tensor 类型的引用 indices，一个 IntArrayRef 类型的 sizes 参数，一个 bool 类型的 keepdim 参数，返回一个 Tensor 类型对象
TORCH_API at::Tensor value_selecting_reduction_backward(const at::Tensor & grad, int64_t dim,
                                                        const at::Tensor & indices, at::IntArrayRef sizes,
                                                        bool keepdim);

// 声明 trace_backward 函数，接受一个 Tensor 类型的引用 grad，一个 IntArrayRef 类型的 sizes 参数，返回一个 Tensor 类型对象
TORCH_API at::Tensor trace_backward(const at::Tensor & grad, at::IntArrayRef sizes);

// 声明 index_select_backward 函数，接受一个 Tensor 类型的引用 grad，一个 IntArrayRef 类型的 self_sizes 参数，
// 一个 int64_t 类型的 dim 参数，一个 Tensor 类型的引用 index，返回一个 Tensor 类型对象
TORCH_API at::Tensor index_select_backward(const at::Tensor & grad, at::IntArrayRef self_sizes,
                                           int64_t dim, const at::Tensor & index);

// 声明 select 函数，接受一个 Tensor 类型的引用 self，一个 int64_t 类型的 dim 和一个 int64_t 类型的 index 参数，返回一个 Tensor 类型对象
TORCH_API at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index);

// 声明 tensor_split 函数，接受一个 Tensor 类型的引用 self，一个 IntArrayRef 类型的 indices 和一个 int64_t 类型的 dim 参数，
// 返回一个 std::vector<Tensor> 类型对象
TORCH_API std::vector<Tensor> tensor_split(const Tensor& self, IntArrayRef indices, int64_t dim);

} // namespace at::native
// 结束 at::native 命名空间
```