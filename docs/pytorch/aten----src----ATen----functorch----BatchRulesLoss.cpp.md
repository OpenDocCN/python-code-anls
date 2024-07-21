# `.\pytorch\aten\src\ATen\functorch\BatchRulesLoss.cpp`

```
// 包含头文件：BatchRulesHelper.h, PlumbingHelper.h, BatchedFallback.h, Dispatcher.h
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

// 进入 funtorch 命名空间
namespace at::functorch {

// 定义静态函数：将张量展平（除批次维外），并将批次维（如果存在）移动到前面
static at::Tensor flatten_logical(const Tensor& tensor, optional<int64_t> bdim) {
  // 如果指定了批次维
  if (bdim.has_value()) {
    // 调用 moveBatchDimToFront 函数将批次维移动到前面
    auto result = moveBatchDimToFront(tensor, bdim);
    // 如果结果张量的维度大于1，则按第1维展平
    if (result.dim() > 1) {
      return result.flatten(1);
    } else {
      return result;
    }
  } else {
    // 否则，直接展平整个张量
    return tensor.flatten();
  }
}

// 用于多个损失函数的辅助函数模板
template <typename Func>
static std::tuple<at::Tensor,optional<int64_t>>
loss_batch_rule_helper(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction,
          Func loss_fn) {
  // 将输入张量展平
  auto self_ = flatten_logical(self, self_bdim);
  auto target_ = flatten_logical(target, target_bdim);
  // 调用指定的损失函数计算损失值
  auto result = loss_fn(self_, target_, Reduction::None);
  // 如果结果的维度为1，则返回结果和0
  if (result.dim() == 1) {
    return std::make_tuple(result, 0);
  } else if (reduction == Reduction::None) {
    // 如果没有减少维度，则重塑结果以匹配原始输入的批次维
    DimVector end_shape;
    const auto batched_elem = self_bdim.has_value() ?
        moveBatchDimToFront(self, self_bdim) : moveBatchDimToFront(target, target_bdim);
    return std::make_tuple(result.reshape(batched_elem.sizes()), 0);
  } else if (reduction == Reduction::Sum) {
    // 如果减少维度为求和，则在最后一个维度上进行求和
    return std::make_tuple(result.sum(-1), 0);
  } else if (reduction == Reduction::Mean) {
    // 如果减少维度为平均值，则在最后一个维度上进行平均
    return std::make_tuple(result.mean(-1), 0);
  }
  // 如果未匹配到任何减少维度选项，则抛出内部断言错误
  TORCH_INTERNAL_ASSERT(false);
};

// 均方误差损失函数的批处理规则
static std::tuple<at::Tensor,optional<int64_t>>
mse_loss_batch_rule(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction) {
  // 调用 loss_batch_rule_helper 函数，使用 mse_loss 计算损失值
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::mse_loss(self, target, reduction);
                                });
};

// Huber 损失函数的批处理规则
static std::tuple<at::Tensor,optional<int64_t>>
huber_loss_batch_rule(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction, double delta) {
  // 调用 loss_batch_rule_helper 函数，使用 huber_loss 计算损失值
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [delta](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::huber_loss(self, target, reduction, delta);
                                });
};

// 这里留有空白行，表示代码尚未完整
// 定义一个函数，计算 Smooth L1 损失的批处理规则
smooth_l1_loss_batch_rule(const at::Tensor& self, optional<int64_t> self_bdim, const at::Tensor& target,
          optional<int64_t> target_bdim, int64_t reduction, double beta) {
  // 调用辅助函数，根据给定的参数计算 Smooth L1 损失
  return loss_batch_rule_helper(self, self_bdim, target, target_bdim,
                                reduction, [beta](const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
                                  return at::smooth_l1_loss(self, target, reduction, beta);
                                });
};

// 静态函数，应用损失的减少操作
static Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
  // 根据指定的 reduction 类型进行损失的降维处理
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  // 默认情况下返回未降维的损失结果
  return unreduced;
}

// 静态函数，处理二元交叉熵的底层实现
static Tensor binary_cross_entropy_plumbing(
    const Tensor& self, const Tensor& target,
    const optional<Tensor>& weight, int64_t reduction) {
  // 获取当前可能的动态层，并检查其是否适用于批处理
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "binary_cross_entropy_plumbing");
  int64_t cur_level = maybe_layer->layerId();

  // 如果输入的 self、target 和 weight 都未在当前层进行批处理，则直接调用二元交叉熵函数
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(target, cur_level)
      && !isBatchedAtLevel(weight, cur_level)) {
    // 临时禁用批处理的分发键，调用 PyTorch 的二元交叉熵函数
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::binary_cross_entropy(self, target, weight, reduction);
  }

  // 在当前层级上解包 self 和 target 张量
  auto [self_value, self_bdim] = unwrapTensorAtLevel(self, cur_level);
  auto [target_value, target_bdim] = unwrapTensorAtLevel(target, cur_level);

  Tensor result;
  // 如果 self 或 target 具有批处理维度
  if (self_bdim || target_bdim) {
    // 临时禁用批处理的分发键，计算批处理维度的大小并重新排序张量
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    const auto bdim_size = get_bdim_size2(self_value, self_bdim, target_value, target_bdim);
    auto self_ = moveBatchDimToFront(self_value, self_bdim);
    auto target_ = moveBatchDimToFront(target_value, target_bdim);
    self_ = ensure_has_bdim(self_, self_bdim.has_value(), bdim_size);
    target_ = ensure_has_bdim(target_, target_bdim.has_value(), bdim_size);
    // 调用不降维的二元交叉熵函数，并将结果重新包装为批处理形式
    result = at::binary_cross_entropy(self_, target_, nullopt, Reduction::None);
    result = makeBatched(result, 0, cur_level);
  } else {
    // 临时禁用批处理的分发键，调用不降维的二元交叉熵函数
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    result = at::binary_cross_entropy(self_value, target_value, nullopt, Reduction::None);
  }
  // 如果 weight 存在且已定义，则将结果乘以权重
  if (weight.has_value() && weight->defined()) {
    result = result * weight.value();
  }
  // 应用指定的损失减少方式，并返回处理后的结果
  return apply_loss_reduction(result, reduction);
}

// 静态函数，处理二元交叉熵的反向传播的底层实现
static Tensor binary_cross_entropy_backward_plumbing(
    const Tensor& grad, const Tensor& input, const Tensor& target,
    const std::optional<Tensor>& weight_opt, int64_t reduction) {
  // 获取当前可能的动态层，并检查其是否适用于批处理
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "binary_cross_entropy_backward_plumbing");
  int64_t cur_level = maybe_layer->layerId();

  // 如果 grad、input、target 和 weight_opt 中任何一个在当前层级进行了批处理，则禁用批处理的分发键
  if (!areAnyBatchedAtLevel({grad, input, target, weight_opt}, cur_level)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 在不考虑批处理的情况下调用 PyTorch 的二元交叉熵反向传播函数

    return at::binary_cross_entropy_backward(grad, input, target, weight_opt, reduction);
  }

  // 如果存在批处理，则在当前层级上解包张量
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(grad, cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [target_value, target_bdim] = unwrapTensorAtLevel(target, cur_level);

  // 临时禁用批处理的分发键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);

  // 在保持批处理维度的前提下，调用不降维的二元交叉熵反向传播函数
  auto result = at::binary_cross_entropy_backward(grad_value, input_value, target_value, weight_opt, Reduction::None);
  if (grad_bdim || input_bdim || target_bdim) {
    result = moveBatchDimToFront(result, grad_bdim);
  }
  // 应用指定的损失减少方式，并返回处理后的结果
  return apply_loss_reduction(result, reduction);
}
    // 调用 ATen 库中的二元交叉熵反向传播函数，计算输入梯度
    return at::binary_cross_entropy_backward(grad, input, target, weight_opt, reduction);
  }

  // 解包输入、目标和梯度张量，并根据当前级别展开或移动批次维度
  auto [grad_value, grad_bdim] = unwrapTensorAtLevel(
      reduction == Reduction::None ? grad : grad.expand_as(input), cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [target_value, target_bdim] = unwrapTensorAtLevel(target, cur_level);

  // 定义梯度输入张量
  Tensor grad_input;

  // 如果存在批次维度，则在 FuncTorchBatched 调度键下执行操作
  if (grad_bdim || input_bdim || target_bdim) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);

    // 获取批次维度大小
    const auto bdim_size = get_bdim_size3(
        grad_value, grad_bdim, input_value, input_bdim, target_value, target_bdim);

    // 将批次维度移至张量的前面
    auto grad_ = moveBatchDimToFront(grad_value, grad_bdim);
    auto input_ = moveBatchDimToFront(input_value, input_bdim);
    auto target_ = moveBatchDimToFront(target_value, target_bdim);

    // 确保张量具有批次维度
    grad_ = ensure_has_bdim(grad_, grad_bdim.has_value(), bdim_size);
    input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size);
    target_ = ensure_has_bdim(target_, target_bdim.has_value(), bdim_size);

    // 计算不带权重的二元交叉熵反向传播
    grad_input = at::binary_cross_entropy_backward(
        grad_, input_, target_, nullopt, Reduction::None);

    // 在指定级别上重新构建批次维度
    grad_input = makeBatched(grad_input, 0, cur_level);
  } else {
    // 否则，不移动批次维度，直接计算二元交叉熵反向传播
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    grad_input = at::binary_cross_entropy_backward(
        grad_value, input_value, target_value, nullopt, Reduction::None);
  }

  // 如果权重选项有值且已定义，则乘以权重
  if (weight_opt.has_value() && weight_opt->defined()) {
    grad_input = grad_input * weight_opt.value();
  }

  // 如果减少方式为平均，则将梯度除以输入元素数量
  if (reduction == Reduction::Mean) {
    grad_input.div_(input.numel());
  }

  // 返回计算得到的梯度输入张量
  return grad_input;
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 定义 ATen 库函数的批处理版本
  VMAP_SUPPORT(mse_loss, mse_loss_batch_rule);
  // 启用 mse_loss 的批处理规则
  VMAP_SUPPORT(huber_loss, huber_loss_batch_rule);
  // 启用 huber_loss 的批处理规则
  VMAP_SUPPORT(smooth_l1_loss, smooth_l1_loss_batch_rule);
  // 启用 smooth_l1_loss 的批处理规则
  m.impl("binary_cross_entropy", binary_cross_entropy_plumbing);
  // 实现二进制交叉熵函数的底层功能
  m.impl("binary_cross_entropy_backward", binary_cross_entropy_backward_plumbing);
  // 实现二进制交叉熵函数的反向传播的底层功能
}

} // namespace at::functorch
```