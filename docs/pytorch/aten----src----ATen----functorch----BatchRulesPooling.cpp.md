# `.\pytorch\aten\src\ATen\functorch\BatchRulesPooling.cpp`

```py
// 包含 ATen 库中所需的头文件
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

// 定义命名空间 at::functorch
namespace at::functorch {

// 定义模板函数 max_pool_with_indices_batch_rule_helper，返回一个包含四个元素的元组
template <typename Func>
std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
max_pool_with_indices_batch_rule_helper(
  const Tensor& self, optional<int64_t> self_bdim,
  IntArrayRef kernel_size, IntArrayRef stride,
  IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, int64_t n, Func pooling_fn) {

  // 计算没有批维度的张量的逻辑秩
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  // 断言逻辑秩为 n+1 或 n+2
  TORCH_INTERNAL_ASSERT(logical_rank == n + 1 || logical_rank == n + 2);

  // 如果逻辑秩为 n+1，将批维度移到最前面，然后调用池化函数
  if (logical_rank == n + 1) {
    auto self_ = moveBatchDimToFront(self, self_bdim);
    auto result = pooling_fn(
        self_, kernel_size, stride, padding, dilation, ceil_mode);
    return std::make_tuple(std::move(std::get<0>(result)), 0, std::move(std::get<1>(result)), 0);
  }

  // 如果逻辑秩为 n+2，将批维度的大小保存下来，将批维度移到最前面，然后调用池化函数
  auto bdim_size = self.size(*self_bdim);
  auto self_ = reshape_dim_into(*self_bdim, 0, self);
  auto result = pooling_fn(
      self_, kernel_size, stride, padding, dilation, ceil_mode);
  // 将结果张量的批维度恢复，返回结果
  return std::make_tuple(
      reshape_dim_outof(0, bdim_size, std::get<0>(result)), 0,
      reshape_dim_outof(0, bdim_size, std::get<1>(result)), 0);
}

// 定义静态函数 max_pool3d_with_indices_batch_rule，调用 max_pool_with_indices_batch_rule_helper 处理 3 维池化
static std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
max_pool3d_with_indices_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    return max_pool_with_indices_batch_rule_helper(self, self_bdim, kernel_size, stride, padding, dilation, ceil_mode, 3, at::max_pool3d_with_indices);
}

// 定义静态函数 max_pool2d_with_indices_batch_rule，调用 max_pool_with_indices_batch_rule_helper 处理 2 维池化
static std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
max_pool2d_with_indices_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    return max_pool_with_indices_batch_rule_helper(self, self_bdim, kernel_size, stride, padding, dilation, ceil_mode, 2, at::max_pool2d_with_indices);
}

} // namespace at::functorch
// 实现 Torch 库中 aten 命名空间的函数 FuncTorchBatched
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 声明 _adaptive_avg_pool2d 存在于 aten 命名空间的函数
  EXISTING_BDIM(_adaptive_avg_pool2d);
  // 声明 _adaptive_avg_pool2d_backward 存在于 aten 命名空间的函数，接受任意数量的参数
  EXISTING_BDIM_ALL_BOXED(_adaptive_avg_pool2d_backward);
  // 声明 _adaptive_avg_pool3d 存在于 aten 命名空间的函数
  EXISTING_BDIM(_adaptive_avg_pool3d);
  // 声明 _adaptive_avg_pool3d_backward 存在于 aten 命名空间的函数，接受任意数量的参数
  EXISTING_BDIM_ALL_BOXED(_adaptive_avg_pool3d_backward);
  // 声明 avg_pool2d 存在于 aten 命名空间的函数
  EXISTING_BDIM(avg_pool2d);
  // 声明 avg_pool3d 存在于 aten 命名空间的函数
  EXISTING_BDIM(avg_pool3d);
  // 声明 avg_pool2d_backward 存在于 aten 命名空间的函数，接受任意数量的参数
  EXISTING_BDIM_ALL_BOXED(avg_pool2d_backward);
  // 声明 avg_pool3d_backward 存在于 aten 命名空间的函数，接受任意数量的参数
  EXISTING_BDIM_ALL_BOXED(avg_pool3d_backward);
  // 声明 adaptive_max_pool2d 存在于 aten 命名空间的函数，接受任意数量的参数
  EXISTING_BDIM_ALL_BOXED(adaptive_max_pool2d);
  // 声明 adaptive_max_pool3d 存在于 aten 命名空间的函数，接受任意数量的参数
  EXISTING_BDIM_ALL_BOXED(adaptive_max_pool3d);
  // 声明所有张量（Tensor）具有可选维度，并且保证是连续的，参数个数为 3
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(3, adaptive_max_pool2d_backward, 2);
  // 声明所有张量（Tensor）具有可选维度，并且保证是连续的，参数个数为 4
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(4, adaptive_max_pool3d_backward, 2);
  // 支持 VMAP，指定 max_pool2d_with_indices 函数及其批处理规则
  VMAP_SUPPORT(max_pool2d_with_indices, max_pool2d_with_indices_batch_rule);
  // 支持 VMAP，指定 max_pool3d_with_indices 函数及其批处理规则
  VMAP_SUPPORT(max_pool3d_with_indices, max_pool3d_with_indices_batch_rule);
  // 声明所有张量（Tensor）具有可选维度，并且保证是连续的，参数个数为 3
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(3, max_pool2d_with_indices_backward, 2);
  // 声明所有张量（Tensor）具有可选维度，并且保证是连续的，参数个数为 4
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(4, max_pool3d_with_indices_backward, 2);
}
// 结束命名空间 at::functorch
} // namespace at::functorch
```