# `.\pytorch\aten\src\ATen\functorch\PyTorchOperatorHacks.cpp`

```
// 包含头文件 DynamicLayer.h，声明了 functorch 的动态层功能
#include <ATen/functorch/DynamicLayer.h>
// 包含 torch 库的头文件
#include <torch/library.h>
// 包含 ATen 头文件，提供张量操作的核心功能
#include <ATen/ATen.h>
// 包含 WrapDimUtils.h 头文件，提供对张量维度的包装操作
#include <ATen/WrapDimUtils.h>
// 包含 TensorWrapper.h 头文件，定义了张量的包装器
#include <ATen/functorch/TensorWrapper.h>
// 包含 BatchedTensorImpl.h 头文件，定义了批次张量的实现
#include <ATen/functorch/BatchedTensorImpl.h>
// 再次包含 ATen 头文件，用于张量操作
#include <ATen/ATen.h>
// 包含 Dispatch.h 头文件，定义了 ATen 的分发机制
#include <ATen/Dispatch.h>
// 包含 c10/util/irange.h 头文件，提供了迭代范围的实用工具
#include <c10/util/irange.h>
// 包含 NamedTensorUtils.h 头文件，提供了对命名张量的支持
#include <ATen/NamedTensorUtils.h>
// 包含 LinearAlgebraUtils.h 头文件，提供线性代数相关的实用函数
#include <ATen/native/LinearAlgebraUtils.h>
// 包含 xnnpack/Engine.h 头文件，引擎实现了 XNNPACK 的功能
#include <ATen/native/xnnpack/Engine.h>

namespace at::functorch {

// NOTE: [functorch's PyTorch Operator Hacks]
//
// This file contains hacks for composite PyTorch operators that are problematic.
// For example, the composite op might have in-place operations,
// or call data_ptr. We have some idea of how to fix these things in the long term
// e.g., upstream the changes to PyTorch.
//
// TODO: all of these should be fixed in a more blessed way. In particular,
// it is bad if any of these go out-of-sync with the implementations in
// pytorch/pytorch.

// TODO: upstream into core

// 匿名命名空间，用于定义一些内部实用函数
namespace {
// 使用索引选择操作的反向传播的hack实现
Tensor index_select_backward_hack(const Tensor& grad, IntArrayRef self_sizes, int64_t dim, const Tensor& index) {
  return at::zeros(self_sizes, grad.options()).index_add(dim, index, grad);
}

// 线性操作的hack实现，用于性能优化
Tensor linear_hack(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 如果 bias_opt 有值，则使用该值作为 bias，否则创建一个新的空张量作为 bias
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(std::in_place);

  // 如果输入张量是 mkldnn 类型，则调用 mkldnn_linear 函数
  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, *bias);
  }
  // 如果在移动设备上，并且可以使用 xnnpack 加速线性操作，则调用 xnnpack::linear 函数
#if defined(C10_MOBILE)
  if (at::native::xnnpack::use_linear(input, weight, *bias)) {
    return at::native::xnnpack::linear(input, weight, *bias);
  }
#endif
  // 如果输入张量维度为 2，并且 bias 已定义，则使用融合的 addmm 操作
  if (input.dim() == 2 && bias->defined()) {
    return at::addmm(*bias, input, weight.t());
  }
  // 如果输入张量维度为 3，并且 bias 已定义，并且输入张量是连续的，则使用融合的 addmm 操作
  if (input.dim() == 3 && bias->defined() && input.is_contiguous()) {
    // 对连续的 3D 输入使用融合路径
    const auto input_sizes = input.sizes();
    const auto result = at::addmm(*bias, input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}), weight.t());
    return result.view({input_sizes[0], input_sizes[1], result.size(1)});
  }
  // 默认情况下使用 matmul 进行矩阵乘法操作
  auto output = at::matmul(input, weight.t());
  // 如果 bias 已定义，则根据当前动态层情况进行 add 或 add_ 操作
  if (bias->defined()) {
    const auto& stack = getDynamicLayerStack();
    // 检查是否存在任何 Vmap 层，若存在则使用 add，否则使用 add_
    bool any_vmap_layers = std::any_of(
        stack.begin(), stack.end(),
        [](const DynamicLayer& dl){ return dl.key() == TransformType::Vmap; });
    if (any_vmap_layers) {
      return output.add(*bias);
    }
    return output.add_(*bias);
  }
  // 返回最终的输出张量
  return output;
}

// 应用损失函数减少的函数，根据指定的 reduction 类型进行操作
static inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  // 默认情况下返回未减少的张量
  return unreduced;
}
// 定义一个名为 `binary_cross_entropy_with_logits_hack` 的函数，接受输入张量 `input`、目标张量 `target`、可选的权重张量 `weight_opt`、可选的正权重张量 `pos_weight_opt` 和减少方式 `reduction`
Tensor binary_cross_entropy_with_logits_hack(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt,
    int64_t reduction) {
  
  // 见注释：[Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量 `weight_opt` 中借用张量，转为 `weight_maybe_owned`
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 使用 `c10::value_or_else` 获取 `pos_weight_opt` 的值，如果为空则返回空张量
  const Tensor& pos_weight = c10::value_or_else(pos_weight_opt, [] {return Tensor();});

  // 定义损失张量
  Tensor loss;

  // 计算 `input` 的负值后取非负的张量 `max_val`
  auto max_val = (-input).clamp_min(0);

  // 如果 `pos_weight` 定义了
  if (pos_weight.defined()) {
    // 需要广播 `pos_weight`，因此 `mul(target)` 不是原地操作
    auto log_weight = (pos_weight - 1).mul(target).add_(1);
    // 计算损失函数
    loss = (1 - target).mul(input).add(log_weight.mul(((-max_val).exp_().add((-input - max_val).exp_())).log_().add_(max_val)));
  } else {
    // 如果 `pos_weight` 未定义，直接计算损失函数
    loss = (1 - target).mul(input).add_(max_val).add_((-max_val).exp_().add((-input -max_val).exp_()).log_());
  }

  // 如果 `weight` 定义了，则将损失乘以 `weight`
  if (weight.defined()) {
    loss = loss * weight;
  }

  // 应用损失减少操作并返回结果
  return apply_loss_reduction(loss, reduction);
}

// 定义一个名为 `trace_backward_decomp` 的函数，接受梯度张量 `grad` 和大小信息 `sizes`
Tensor trace_backward_decomp(const Tensor& grad, IntArrayRef sizes) {
  // 如果 `sizes` 的大小不等于 2，则抛出运行时错误
  if (sizes.size() != 2) {
    throw std::runtime_error("expected matrix input");
  }

  // 创建大小为 `sizes[0] * sizes[1]` 的零张量 `grad_input`
  auto grad_input = at::zeros(sizes[0] * sizes[1], grad.options());

  // 创建索引张量 `indices`，步长为 `sizes[1] + 1`
  auto indices = at::arange(0, grad_input.numel(), sizes[1] + 1, grad.options().dtype(at::kLong));

  // 使用 `index_put` 替代尚未支持的 `index_fill_`，将 `grad` 填充到 `grad_input` 中指定的位置
  grad_input = grad_input.index_put({indices}, grad);

  // 调整 `grad_input` 的视图大小为 `sizes`，并返回结果
  return grad_input.view(sizes);
}

// 声明 `dropout_hack` 命名空间
// TODO: 在 pytorch/pytorch 中进行以下更改
namespace dropout_hack {

// 嵌套命名空间
namespace {

// 使用模板定义 `Ctype`，根据 `inplace` 决定返回类型是 `Tensor&` 还是 `Tensor`
template<bool inplace>
using Ctype = std::conditional_t<inplace, Tensor&, Tensor>;

// 静态函数，生成与 `input` 大小相同的特征噪声张量
static Tensor make_feature_noise(const Tensor& input) {
  // 获取 `input` 的大小信息
  auto input_sizes = input.sizes();
  // 检查 `input` 的维度至少为 2
  TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");

  // 构建大小向量，与 `input_sizes` 共享第一和第二维度，其它维度为 1
  std::vector<int64_t> sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for (C10_UNUSED const auto i : c10::irange(2, input.dim())) {
    sizes.push_back(1);
  }

  // NB: THIS WAS CHANGED FROM THE ORIGINAL
  // 创建一个空张量，与 `input` 具有相同的选项和指定大小
  return at::empty(sizes, input.options());
}

// 检查是否可以接受融合的核心函数，要求 `input` 是 CUDA 或 XPU 或 Lazy，并且 `p` 在 (0, 1) 之间，且元素个数大于 0
static bool is_fused_kernel_acceptable(const Tensor& input, double p) {
  return (input.is_cuda() || input.is_xpu() || input.is_lazy()) && p > 0 && p < 1 && input.numel() > 0;
}

// 模板函数，根据 `inplace` 的值选择合适的乘法重载版本，返回 `Tensor&`
// NB: sure, we could have used different overloads here, but I would feel insecure
// knowing that this dispatch depends only on the constness of the references
template<bool inplace>
Tensor& multiply(Tensor& input, const Tensor& noise) {
  // 静态断言，如果 `inplace` 不为真，则触发错误信息
  static_assert(inplace, "Wrong multiply overload triggered in Dropout.cpp");
  // 返回原地乘法操作的结果
  return input.mul_(noise);
}

// 模板函数，根据 `inplace` 的值选择合适的乘法重载版本，返回 `Tensor`
template<bool inplace>
Tensor multiply(const Tensor& input, const Tensor& noise) {
  // 静态断言，如果 `inplace` 为真，则触发错误信息
  static_assert(!inplace, "Wrong multiply overload triggered in Dropout.cpp");
  // 返回非原地乘法操作的结果
  return input.mul(noise);
}
// 定义一个模板函数，实现特定类型的dropout操作
template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {
  // 检查dropout概率p的有效性，必须在0到1之间
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  
  // 如果概率p为0，或者不处于训练状态，或者输入为空，则直接返回输入数据
  if (p == 0 || !train || input.numel() == 0) {
    return input;
  }

  // 如果概率p为1，返回与input同样大小但值全为0的张量
  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // 用于alpha_dropout时使用的变量

  // 创建一个Tensor变量noise，用于存储dropout噪声
  Tensor noise;
  if (feature_dropout) {
    // 如果进行特征级dropout，则生成特征级噪声
    auto empty = make_feature_noise(input);
    noise = at::bernoulli(empty, 1 - p);
  } else {
    // 否则生成与input同样大小的空张量，并生成噪声
    auto empty = at::empty({}, input.options()).expand(input.sizes());
    noise = at::bernoulli(empty, 1 - p);
  }

  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    // 计算alpha_dropout的参数a
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    // 计算b，并对噪声进行调整
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);
  } else {
    // 对噪声进行调整，用于普通的dropout
    noise.div_(1 - p);
  }

  // 根据alpha_dropout的情况，对输入数据进行dropout操作并返回
  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  } else {
    return multiply<inplace>(input, noise).add_(b);
  }
}

// 定义一个宏，简化不同dropout操作的函数模板的声明
#define ALIAS_SPECIALIZATION(ALIAS_NAME, IS_FEATURE, IS_ALPHA)                      \
template <bool inplace, typename... Args>                                           \
Ctype<inplace> ALIAS_NAME(Args&&... args) {                                         \
  return _dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(std::forward<Args>(args)...); \
}

// 为不同的dropout操作定义具体函数的模板声明
ALIAS_SPECIALIZATION(_dropout,               false, false)
ALIAS_SPECIALIZATION(_feature_dropout,       true,  false)
ALIAS_SPECIALIZATION(_alpha_dropout,         false, true )
ALIAS_SPECIALIZATION(_feature_alpha_dropout, true,  true )

// 定义一个静态函数，实现dropout操作，并支持名字推断
static Tensor dropout(const Tensor& input, double p, bool train) {
  // 使用lambda表达式定义函数，检查是否可以使用融合内核进行dropout操作
  auto result = [&]() {
    NoNamesGuard guard;
    if (train && is_fused_kernel_acceptable(input, p)) {
      return std::get<0>(at::native_dropout(input, p, train));
    }
    return _dropout<false>(input, p, train);
  }();
  // 推断输出张量的命名属性，并返回结果
  namedinference::propagate_names(result, input);
  return result;
}

// 实现在原位操作下的dropout，并返回结果
Tensor& dropout_(Tensor& input, double p, bool train) {
  return _dropout<true>(input, p, train);
}

// 实现特征级dropout操作，并返回结果
Tensor feature_dropout(const Tensor& input, double p, bool train) {
  return _feature_dropout<false>(input, p, train);
}

// 实现在原位操作下的特征级dropout，并返回结果
Tensor& feature_dropout_(Tensor& input, double p, bool train) {
  return _feature_dropout<true>(input, p, train);
}

// 实现alpha_dropout操作，并返回结果
Tensor alpha_dropout(const Tensor& input, double p, bool train) {
  return _alpha_dropout<false>(input, p, train);
}

// 实现在原位操作下的alpha_dropout，并返回结果
Tensor& alpha_dropout_(Tensor& input, double p, bool train) {
  return _alpha_dropout<true>(input, p, train);
}

// 实现特征级alpha_dropout操作，并返回结果
Tensor feature_alpha_dropout(const Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<false>(input, p, train);
}

// 实现在原位操作下的特征级alpha_dropout，并返回结果
Tensor& feature_alpha_dropout_(Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<true>(input, p, train);
}
// 实现 TORCH_LIBRARY_IMPL 宏，用于定义 Torch 库中 aten 命名空间的函数
TORCH_LIBRARY_IMPL(aten, FuncTorchDynamicLayerFrontMode, m) {
    // 注册 index_select_backward 函数的实现为 index_select_backward_hack
    m.impl("index_select_backward", index_select_backward_hack);
    // 注册 linear 函数的实现为 linear_hack
    m.impl("linear", linear_hack);
    // 注册 binary_cross_entropy_with_logits 函数的实现为 binary_cross_entropy_with_logits_hack
    m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_hack);
    // 注册 trace_backward 函数的实现为 trace_backward_decomp
    m.impl("trace_backward", trace_backward_decomp);

    // 注册 dropout 函数的实现为 dropout_hack 命名空间中的 dropout 函数
    m.impl("dropout", dropout_hack::dropout);
    // 注册 feature_dropout 函数的实现为 dropout_hack 命名空间中的 feature_dropout 函数
    m.impl("feature_dropout", dropout_hack::feature_dropout);
    // 注册 alpha_dropout 函数的实现为 dropout_hack 命名空间中的 alpha_dropout 函数
    m.impl("alpha_dropout", dropout_hack::alpha_dropout);
    // 注册 feature_alpha_dropout 函数的实现为 dropout_hack 命名空间中的 feature_alpha_dropout 函数
    m.impl("feature_alpha_dropout", dropout_hack::feature_alpha_dropout);

    // 注册 dropout_ 函数的实现为 dropout_hack 命名空间中的 dropout_ 函数
    m.impl("dropout_", dropout_hack::dropout_);
    // 注册 feature_dropout_ 函数的实现为 dropout_hack 命名空间中的 feature_dropout_ 函数
    m.impl("feature_dropout_", dropout_hack::feature_dropout_);
    // 注册 alpha_dropout_ 函数的实现为 dropout_hack 命名空间中的 alpha_dropout_ 函数
    m.impl("alpha_dropout_", dropout_hack::alpha_dropout_);
    // 注册 feature_alpha_dropout_ 函数的实现为 dropout_hack 命名空间中的 feature_alpha_dropout_ 函数
    m.impl("feature_alpha_dropout_", dropout_hack::feature_alpha_dropout_);
}

// 结束定义命名空间 at::functorch
} // namespace at::functorch
```