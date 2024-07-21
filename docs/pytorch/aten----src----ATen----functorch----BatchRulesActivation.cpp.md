# `.\pytorch\aten\src\ATen\functorch\BatchRulesActivation.cpp`

```
// 包含标准库头文件，用于实现 C++ 程序
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>

// 函数库的命名空间，定义了批处理规则的支持函数
namespace at::functorch {

// 定义 glu 函数的批处理规则
static std::tuple<Tensor,optional<int64_t>>
glu_batch_rule(const Tensor& self, optional<int64_t> self_bdim, int64_t dim) {
  // 检查输入张量的维度是否大于 1，因为 glu 不支持 0 维张量
  TORCH_CHECK(self.dim() > 1, "glu does not support 0-dimensional tensors");

  // 获取没有批次维度的张量秩
  const auto rank = rankWithoutBatchDim(self, self_bdim);
  // 确定维度值，可能会调整维度索引
  const auto dim_ = maybe_wrap_dim(dim, rank) + 1;

  // 将批次维度移到张量的最前面
  const auto self_ = moveBatchDimToFront(self, self_bdim);

  // 调用 ATen 库的 glu 函数进行操作
  const auto res = at::glu(self_, dim_);
  // 返回 glu 操作后的结果张量和标志值 0
  return std::make_tuple(res, 0);
}

// 定义 glu 反向传播函数的批处理规则
static std::tuple<Tensor,optional<int64_t>> glu_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim) {
  // 如果存在自变量的批次维度，检查输入张量的维度是否大于 1，因为 glu 不支持 0 维张量
  if (self_bdim) {
    TORCH_CHECK(self.dim() > 1, "glu does not support 0-dimensional tensors");
  }

  // 获取没有批次维度的张量秩
  const auto rank = rankWithoutBatchDim(self, self_bdim);
  // 确定维度值，可能会调整维度索引
  const auto dim_ = maybe_wrap_dim(dim, rank) + 1;

  // 获取输入张量和梯度输出张量的批次大小
  const auto batch_size = get_bdim_size2(grad_output, grad_output_bdim, self, self_bdim);
  // 确保梯度输出张量和自变量张量有批次维度，如果没有，则根据批次大小进行调整
  const auto grad_output_ = ensure_has_bdim(moveBatchDimToFront(grad_output, grad_output_bdim), grad_output_bdim.has_value(), batch_size);
  const auto self_ = ensure_has_bdim(moveBatchDimToFront(self, self_bdim), self_bdim.has_value(), batch_size);

  // 调用 ATen 库的 glu 反向传播函数进行操作
  const auto res = at::glu_backward(grad_output_, self_, dim_);
  // 返回 glu 反向传播操作后的结果张量和标志值 0
  return std::make_tuple(res, 0);
}

// 在 ATen 库中注册批处理支持规则的实现
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 启用 glu 反向传播函数的批处理支持规则
  VMAP_SUPPORT(glu_backward, glu_backward_batch_rule);
  // 启用 glu 函数的批处理支持规则
  VMAP_SUPPORT(glu, glu_batch_rule);
}

} // namespace at::functorch
```