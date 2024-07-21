# `.\pytorch\aten\src\ATen\native\Loss.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义预处理指令，限定仅包含方法操作符

#include <ATen/core/Tensor.h>
#include <ATen/core/Reduction.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/Exception.h>
#include <ATen/TensorSubclassLikeUtils.h>
// 包含各种头文件，定义了各种张量运算和操作的相关函数和结构体

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/binary_cross_entropy_backward_native.h>
#include <ATen/ops/binary_cross_entropy_native.h>
#include <ATen/ops/binary_cross_entropy_with_logits_native.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/cosine_embedding_loss_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/hinge_embedding_loss_native.h>
#include <ATen/ops/huber_loss_backward.h>
#include <ATen/ops/huber_loss_backward_native.h>
#include <ATen/ops/huber_loss_native.h>
#include <ATen/ops/kl_div_native.h>
#include <ATen/ops/l1_loss_native.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log_sigmoid.h>
#include <ATen/ops/margin_ranking_loss_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/min.h>
#include <ATen/ops/mse_loss_backward.h>
#include <ATen/ops/mse_loss_backward_native.h>
#include <ATen/ops/mse_loss_meta.h>
#include <ATen/ops/mse_loss_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/pairwise_distance.h>
#include <ATen/ops/poisson_nll_loss_native.h>
#include <ATen/ops/smooth_l1_loss_backward.h>
#include <ATen/ops/smooth_l1_loss_backward_native.h>
#include <ATen/ops/smooth_l1_loss_meta.h>
#include <ATen/ops/smooth_l1_loss_native.h>
#include <ATen/ops/soft_margin_loss.h>
#include <ATen/ops/soft_margin_loss_backward.h>
#include <ATen/ops/soft_margin_loss_backward_native.h>
#include <ATen/ops/soft_margin_loss_native.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/triplet_margin_loss_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/zeros_like.h>
#endif
// 根据条件包含不同的头文件，根据预处理变量决定是否包含各种操作的函数头文件

constexpr float EPSILON = 1e-12;
// 定义浮点常量 EPSILON，并设置为 1e-12

namespace {
  static inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
    // 根据指定的 reduction 类型对未减少的张量进行减少操作
    if (reduction == at::Reduction::Mean) {
      return unreduced.mean();
    } else if (reduction == at::Reduction::Sum) {
      return unreduced.sum();
    }
    return unreduced;
  }
}
// 定义匿名命名空间和静态内联函数 apply_loss_reduction，用于处理张量的减少操作

namespace at::meta {

TORCH_META_FUNC(smooth_l1_loss)
(const Tensor& input, const Tensor& target, const int64_t reduction, double beta) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  // 检查 beta 是否为非负数，否则抛出异常

  // TODO: Reduce this extra TensorIterator construction for Reduction::Mean & Sum.
  // We do another TensorIterator construction in the IMPL for the two cases.
  // 待办事项：减少 Reduction::Mean 和 Sum 的额外 TensorIterator 构造
  // 在 IMPL 中，我们为这两种情况进行了另一个 TensorIterator 的构造。

  build_borrowing_binary_op(maybe_get_output(), input, target);
  // 使用 build_borrowing_binary_op 函数构建二元操作，可能获取输出

  if (reduction == Reduction::None) {
    // 如果 reduction 类型为 None
    return;
  }



    # 如果条件不满足，直接返回，结束函数执行
    return;
  }



  TORCH_INTERNAL_ASSERT(reduction == Reduction::Mean || reduction == Reduction::Sum);
  maybe_get_output().resize_({});



  # 使用内部断言确保 reduction 只能是 Reduction::Mean 或 Reduction::Sum 中的一个
  TORCH_INTERNAL_ASSERT(reduction == Reduction::Mean || reduction == Reduction::Sum);
  # 调用 maybe_get_output() 方法获取输出对象，然后调用 resize_({}) 方法对其进行重新调整大小，可能是将其置为空
  maybe_get_output().resize_({});
}

// 定义一个 TORCH_META_FUNC 宏，注册 mse_loss 函数的元信息
TORCH_META_FUNC(mse_loss)
// 定义 mse_loss 函数，接受输入张量 input、目标张量 target 和缩减模式 reduction
(const Tensor& input, const Tensor& target, const int64_t reduction) {
  // 调用 build_borrowing_binary_op 函数，构建用于借用二元操作的对象，将输出指向 maybe_get_output()，操作数为 input 和 target
  build_borrowing_binary_op(maybe_get_output(), input, target);
  // 如果 reduction 为 Reduction::None，直接返回
  if (reduction == Reduction::None) {
    return;
  }

  // 确保 reduction 是 Reduction::Mean 或 Reduction::Sum
  TORCH_INTERNAL_ASSERT(reduction == Reduction::Mean || reduction == Reduction::Sum);
  // 重置 maybe_get_output() 的大小为空
  maybe_get_output().resize_({});
}

} // namespace at::meta

namespace at::native {

// 定义 smooth_l1_stub 函数的分发器
DEFINE_DISPATCH(smooth_l1_stub);
// 定义 smooth_l1_backward_stub 函数的分发器
DEFINE_DISPATCH(smooth_l1_backward_stub);
// 定义 huber_stub 函数的分发器
DEFINE_DISPATCH(huber_stub);
// 定义 huber_backward_stub 函数的分发器
DEFINE_DISPATCH(huber_backward_stub);
// 定义 mse_stub 函数的分发器
DEFINE_DISPATCH(mse_stub);
// 定义 mse_backward_stub 函数的分发器
DEFINE_DISPATCH(mse_backward_stub);

// 实现 smooth_l1_loss_out 函数，输出结果到 result 张量
TORCH_IMPL_FUNC(smooth_l1_loss_out)
(const Tensor& input, const Tensor& target, int64_t reduction, double beta, const Tensor& result) {
  // 如果 reduction 不是 Reduction::None
  if (reduction != Reduction::None) {
    // 创建 loss 张量
    Tensor loss;
    // 借用二元操作的迭代器，操作数为 input 和 target，结果存储在 loss 中
    auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
    // 调用 smooth_l1_stub 函数，根据设备类型和迭代器执行 smooth L1 损失计算，使用参数 beta
    smooth_l1_stub(iter.device_type(), iter, beta);
    // 如果 reduction 是 Reduction::Mean
    if (reduction == Reduction::Mean) {
      // 计算 iter 的输出的平均值，结果存储在 result 中
      at::mean_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    } else {
      // 计算 iter 的输出的总和，结果存储在 result 中
      at::sum_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    }
  } else {
    // 如果 reduction 是 Reduction::None，直接调用 smooth_l1_stub 函数，对当前对象执行 smooth L1 损失计算
    smooth_l1_stub(device_type(), *this, beta);
  }
}

// 实现 mse_loss_out 函数，输出结果到 result 张量
TORCH_IMPL_FUNC(mse_loss_out)
(const Tensor& input, const Tensor& target, int64_t reduction, const Tensor& result) {
  // 如果 reduction 不是 Reduction::None
  if (reduction != Reduction::None) {
    // 创建 loss 张量
    Tensor loss;
    // 借用二元操作的迭代器，操作数为 input 和 target，结果存储在 loss 中
    auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
    // 调用 mse_stub 函数，根据设备类型和迭代器执行均方误差损失计算
    mse_stub(iter.device_type(), iter);
    // 如果 reduction 是 Reduction::Mean
    if (reduction == Reduction::Mean) {
      // 计算 iter 的输出的平均值，结果存储在 result 中
      at::mean_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    } else {
      // 计算 iter 的输出的总和，结果存储在 result 中
      at::sum_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    }
  } else {
    // 如果 reduction 是 Reduction::None，直接调用 mse_stub 函数，对当前对象执行均方误差损失计算
    mse_stub(device_type(), *this);
  }
}

// 实现 cosine_embedding_loss 函数，计算余弦嵌入损失
Tensor cosine_embedding_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, int64_t reduction) {
  // 获取目标张量的维度
  auto targ_dim = target.dim();
  // 检查目标张量是否为0维或1维，不支持多目标
  TORCH_CHECK(
      targ_dim == 1 || targ_dim == 0,
      "0D or 1D target tensor expected, multi-target not supported");
  // 如果目标张量是1维
  if (targ_dim == 1) {
    // 检查 input1 和 input2 张量是否为2维
    TORCH_CHECK(
        input1.dim() == 2 && input2.dim() == 2,
        "1D target tensor expects 2D input tensors, but found inputs with sizes ",
        input1.sizes(),
        " and ",
        input2.sizes(),
        ".");
  } else {
    // 检查输入张量的维度是否为1，如果不是则抛出错误信息
    TORCH_CHECK(
        input1.dim() == 1 && input2.dim() == 1,
        "0D target tensor expects 1D input tensors, but found inputs with sizes ",
        input1.sizes(),
        " and ",
        input2.sizes(),
        ".");

  }

  // 计算输入张量按指定维度 targ_dim 的乘积和
  auto prod_sum = (input1 * input2).sum(targ_dim);

  // 计算输入张量按指定维度 targ_dim 的平方和并加上 EPSILON，防止除零错误
  auto mag_square1 = (input1 * input1).sum(targ_dim) + EPSILON;
  auto mag_square2 = (input2 * input2).sum(targ_dim) + EPSILON;

  // 计算分母，为 mag_square1 和 mag_square2 的平方根
  auto denom = (mag_square1 * mag_square2).sqrt_();

  // 计算余弦相似度
  auto cos = prod_sum / denom;

  // 创建一个与 cos 相同形状的全零张量
  auto zeros = at::zeros_like(cos, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // 计算正向损失，即 1 - cos
  auto pos = 1 - cos;

  // 计算负向损失，使用 clamp_min_ 函数确保不小于0
  auto neg = (cos - margin).clamp_min_(0);

  // 根据目标张量的值（1 或 -1），选择输出 pos 或 neg，其余部分填充为 zeros
  auto output_pos = at::where(target == 1, pos, zeros);
  auto output_neg = at::where(target == -1, neg, zeros);

  // 最终输出为正向损失和负向损失的和
  auto output = output_pos + output_neg;

  // 应用损失函数的降维操作（如平均或求和）
  return apply_loss_reduction(output, reduction);
// 计算 Hinge Embedding Loss 的函数，输入为张量 self、target，以及 margin 和 reduction 参数
Tensor hinge_embedding_loss(const Tensor& self, const Tensor& target, double margin, int64_t reduction) {
  // 创建一个与 self 相同形状的全零张量
  auto zeros = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 计算 margin 和 self 之间的差异
  auto margin_diff = (margin - self);
  // 根据 Composite Compliance 的要求，在 Forward AD 中，
  // 如果 margin_diff 是 CCT 但其切线不是，则使用 inplace clamp_min 会导致问题，
  // 因此根据条件选择 clamp_min 或 clamp_min_ 方法进行处理
  auto margin_clamp = (margin_diff._fw_grad(/*level*/ 0).defined() &&
                       isTensorSubclassLike(margin_diff))
      ? margin_diff.clamp_min(0)
      : margin_diff.clamp_min_(0);
  // 根据 target 的值，选择填充 output_margin 的方式
  auto output_margin = at::where(target != 1, margin_clamp, zeros);
  // 根据 target 的值，选择填充 output_self 的方式
  auto output_self = at::where(target != -1, self, zeros);
  // 将 output_margin 和 output_self 相加得到最终的输出张量
  auto output = output_margin + output_self;
  // 应用指定的损失函数减少方法，并返回结果
  return apply_loss_reduction(output, reduction);
}

// 计算 Triplet Margin Loss 的函数，输入为 anchor、positive、negative 三个张量，以及 margin、p、eps、swap 和 reduction 参数
Tensor triplet_margin_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative, double margin,
                           double p, double eps, bool swap, int64_t reduction) {
  // 检查 anchor、positive 和 negative 的维度是否一致
  auto a_dim = anchor.dim();
  auto p_dim = positive.dim();
  auto n_dim = negative.dim();
  TORCH_CHECK(
      a_dim == p_dim && p_dim == n_dim,
      "The anchor, positive, and negative tensors are expected to have "
      "the same number of dimensions, but got: anchor ", a_dim, "D, "
      "positive ", p_dim, "D, and negative ", n_dim, "D inputs")

  // 计算 anchor 和 positive 之间的距离
  auto dist_pos = at::pairwise_distance(anchor, positive, p, eps);
  // 计算 anchor 和 negative 之间的距离
  auto dist_neg = at::pairwise_distance(anchor, negative, p, eps);
  
  // 根据 swap 参数，决定是否进行距离交换处理
  if (swap) {
    auto dist_swap = at::pairwise_distance(positive, negative, p, eps);
    // 在 swap 为真时，选择 dist_neg 和 dist_swap 中较小的距离作为 dist_neg 的值
    dist_neg = at::min(dist_neg, dist_swap);
  }
  
  // 计算 Triplet Margin Loss 的输出张量，应用 clamp_min 函数进行截断
  auto output = at::clamp_min(margin + dist_pos - dist_neg, 0);
  // 应用指定的损失函数减少方法，并返回结果
  return apply_loss_reduction(output, reduction);
}

// 计算 Margin Ranking Loss 的函数，输入为 input1、input2、target 三个张量，以及 margin 和 reduction 参数
Tensor margin_ranking_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, int64_t reduction) {
  // 计算未经截断的输出张量 unclamped_output
  auto unclamped_output = (-target * (input1 - input2) + margin);
  
  // 根据 Composite Compliance 的要求，在 Forward AD 中，
  // 如果 unclamped_output 是 CCT 但其切线不是，则使用 inplace clamp_min 会导致问题，
  // 因此根据条件选择 clamp_min 或 clamp_min_ 方法进行处理
  auto output = (unclamped_output._fw_grad(/*level*/ 0).defined() &&
                 isTensorSubclassLike(unclamped_output))
      ? unclamped_output.clamp_min(0)
      : unclamped_output.clamp_min_(0);
  
  // 应用指定的损失函数减少方法，并返回结果
  return apply_loss_reduction(output, reduction);
}
Tensor kl_div(const Tensor& input, const Tensor& target, int64_t reduction, bool log_target) {
  // 检查输入张量是否为复数类型，不支持复数输入
  TORCH_CHECK(!input.is_complex() && !target.is_complex(),
              "kl_div: Complex inputs not supported.");
  // 检查输入张量是否为整数类型（包括布尔类型），不支持整数输入
  TORCH_CHECK(!at::isIntegralType(input.scalar_type(), /*include_bool*/true) &&
              !at::isIntegralType(target.scalar_type(), /*include_bool*/true),
              "kl_div: Integral inputs not supported.");
  Tensor output;
  // 根据 log_target 参数选择不同的计算方式
  if (log_target) {
    // 如果 log_target 为 true，计算 KL 散度的输出
    output = at::exp(target) * (target - input);
  } else {
    // 如果 log_target 为 false，计算 KL 散度的输出
    output = at::xlogy(target, target) - target * input;
  }
  // 应用损失函数的减少操作，并返回结果
  return apply_loss_reduction(output, reduction);
}

Tensor binary_cross_entropy_cpu(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中获取权重数据
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 创建一个与输入张量 input 类型和大小相同的空张量 loss
  Tensor loss = at::empty_like(input);
  // 调用 binary_cross_entropy_out_cpu 函数计算二元交叉熵
  return at::native::binary_cross_entropy_out_cpu(
      input, target, weight, reduction, loss);
}

Tensor& binary_cross_entropy_out_cpu(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, Tensor& loss) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从可选的权重张量中获取权重数据
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 对损失张量 loss 进行压缩处理
  Tensor loss_squeezed = at::squeeze(loss);

  // 配置张量迭代器，指定输出、输入张量及操作
  auto iter = TensorIteratorConfig()
    .add_output(loss_squeezed)
    .add_owned_const_input(at::squeeze(input))
    .add_owned_const_input(at::squeeze(target))
    .build();

  // 根据输入张量的浮点类型执行 CPU 内核操作
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      loss.scalar_type(),
      "binary_cross_entropy",
      [&] {
        at::native::cpu_kernel(
            iter, [](scalar_t input_val, scalar_t target_val) {
              // 检查输入张量元素是否在 [0, 1] 范围内
              TORCH_CHECK(
                  (input_val >= 0) && (input_val <= 1),
                  "all elements of input should be between 0 and 1");
              TORCH_CHECK(
                  (target_val >= 0) && (target_val <= 1),
                  "all elements of target should be between 0 and 1");

              // 二元交叉熵张量由以下方程定义：
              // L = -w (y ln(x) + (1-y) ln(1-x))
              return (target_val - scalar_t(1)) *
                  std::max(scalar_t(std::log1p(-input_val)), scalar_t(-100)) -
                  target_val *
                  std::max(scalar_t(std::log(input_val)), scalar_t(-100));
            });
      });

  // 如果定义了权重张量，将损失张量乘以权重
  if (weight.defined()) {
      loss.mul_(weight);
  }
  // 如果 reduction 参数不等于 None，则应用损失函数的减少操作
  if (reduction != at::Reduction::None) {
      Tensor loss_reduced = apply_loss_reduction(loss, reduction);
      loss.resize_as_(loss_reduced).copy_(loss_reduced);
  }
  // 返回损失张量的引用
  return loss;
}
// 计算二进制交叉熵损失函数在 CPU 上的反向传播，返回梯度输入张量
Tensor binary_cross_entropy_backward_cpu(const Tensor& grad, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction) {
  // 参考注释：用于处理可选张量的包装器去除
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 创建一个与输入张量相同形状的空张量 grad_input
  Tensor grad_input = at::empty_like(input);
  // 调用 binary_cross_entropy_backward_out_cpu 函数计算梯度并存储在 grad_input 中
  return at::native::binary_cross_entropy_backward_out_cpu(
      grad, input, target, weight, reduction, grad_input);
}

// 计算二进制交叉熵损失函数在 CPU 上的反向传播，结果存储在 grad_input 中
Tensor& binary_cross_entropy_backward_out_cpu(const Tensor& grad, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, Tensor& grad_input) {
  // 参考注释：用于处理可选张量的包装器去除
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 对 grad_input 进行挤压操作，去除维度为1的轴
  Tensor grad_input_squeezed = at::squeeze(grad_input);

  // 配置张量迭代器，以便对多个张量进行迭代处理
  auto iter = TensorIteratorConfig()
    .add_output(grad_input_squeezed)  // 输出为 grad_input_squeezed
    .add_owned_const_input(at::squeeze(grad))  // 添加梯度张量的挤压版本作为输入
    .add_owned_const_input(at::squeeze(input))  // 添加输入张量的挤压版本作为输入
    .add_owned_const_input(at::squeeze(target))  // 添加目标张量的挤压版本作为输入
    .build();

  // 根据输入张量的数据类型选择不同的处理函数，用于计算梯度
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      grad_input.scalar_type(),
      "binary_cross_entropy_backward",
      [&] {
        at::native::cpu_kernel(
            iter,
            [](scalar_t grad_val, scalar_t input_val, scalar_t target_val) {
              // 梯度是 BCELoss 对 x 的偏导数
              // d(L)/d(x) = -w (y - x) / (x - x^2)
              return grad_val * (input_val - target_val) /
                  (scalar_t(std::max(
                      (scalar_t(1) - input_val) * input_val,
                      scalar_t(EPSILON))));
            });
      });

  // 如果定义了权重张量 weight，则将 grad_input 乘以权重
  if (weight.defined()) {
      grad_input.mul_(weight);
  }
  // 如果 reduction 为 Reduction::Mean，则将 grad_input 除以输入张量的元素数
  if (reduction == at::Reduction::Mean) {
      grad_input.div_(input.numel());
  }
  // 返回更新后的 grad_input 张量
  return grad_input;
}
// 计算带有 logits 的二元交叉熵损失函数
Tensor binary_cross_entropy_with_logits(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& pos_weight_opt, int64_t reduction) {
  // 查看 [注意: 用于可选张量的 hacky 包装移除]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  // 获取权重张量
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> pos_weight_maybe_owned = at::borrow_from_optional_tensor(pos_weight_opt);
  // 获取正样本权重张量
  const Tensor& pos_weight = *pos_weight_maybe_owned;

  // 计算 input 的 log sigmoid
  auto log_sigmoid_input = at::log_sigmoid(input);

  // 如果定义了 pos_weight
  if (pos_weight.defined()) {
      // 需要广播 pos_weight，因此 mul(target) 不是原位操作
      auto log_weight = (pos_weight - 1).mul(target).add_(1);
      log_sigmoid_input.mul_(log_weight);
  }

  // 计算损失
  Tensor loss = (1 - target).mul_(input).sub_(log_sigmoid_input);

  // 如果定义了 weight
  if (weight.defined()) {
      loss.mul_(weight);
  }

  // 应用损失函数的缩减（如 mean 或 sum）
  return apply_loss_reduction(loss, reduction);
}

// 计算泊松分布的负对数似然损失函数
Tensor poisson_nll_loss(const Tensor& input, const Tensor& target, const bool log_input, const bool full, const double eps, const int64_t reduction)
{
    Tensor loss;
    if (log_input) {
        // 如果 log_input 为真，使用指数函数
        loss = at::exp(input) - target * input;
    } else {
        // 否则，使用对数函数
        loss = input - target * at::log(input + eps);
    }

    // 如果 full 为真
    if (full) {
        // 计算斯特林项
        auto stirling_term = target * at::log(target) - target + 0.5 * at::log(2 * c10::pi<double> * target);
        // 对小于等于 1 的值进行屏蔽填充
        loss += stirling_term.masked_fill(target <= 1, 0);
    }

    // 应用损失函数的缩减（如 mean 或 sum）
    return apply_loss_reduction(loss, reduction);
}

// 计算 soft margin 损失函数的反向传播
Tensor& soft_margin_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, Tensor& grad_input) {
  // 根据缩减类型确定归一化常数
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto z = at::exp(-target * input);
  // 原位版本的操作: grad_input = -norm * target * z / (1. + z) * grad_output;
  at::mul_out(grad_input, target, z).mul_(-norm);
  z.add_(1);
  grad_input.div_(z).mul_(grad_output);
  return grad_input;
}

// soft margin 损失函数的反向传播
Tensor soft_margin_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  auto grad_input = at::empty({0}, input.options());
  // 调用原位版本的反向传播函数
  at::soft_margin_loss_backward_out(grad_input, grad_output, input, target, reduction);
  return grad_input;
}

// 计算 soft margin 损失函数的输出
Tensor& soft_margin_loss_out(const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output) {
  // 计算原位版本的函数: output = at::log1p(at::exp(-input * target));
  at::neg_out(output, input).mul_(target).exp_().log1p_();
  // 如果不是无缩减模式，则应用损失函数的缩减
  if (reduction != Reduction::None) {
    auto tmp = apply_loss_reduction(output, reduction);
    output.resize_({});
    output.copy_(tmp);
  }
  return output;
}

// soft margin 损失函数
Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  auto output = at::empty({0}, input.options());
  // 调用 soft_margin_loss_out 函数计算 soft margin 损失
  at::soft_margin_loss_out(output, input, target, reduction);
  return output;
}
// 计算 Smooth L1 损失函数对输入的梯度，并存储到给定的 grad_input 中
Tensor& smooth_l1_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double beta, Tensor& grad_input) {
  // 根据 reduction 类型确定归一化系数
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  // 创建 Tensor 迭代器配置对象，配置输出、输入常量、以及类型转换等参数
  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(input)
    .add_const_input(target)
    .add_const_input(grad_output)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .build();
  // 调用 Smooth L1 损失函数的反向传播方法
  smooth_l1_backward_stub(iter.device_type(), iter, norm, beta);
  return grad_input;
}

// 计算 Smooth L1 损失函数对输入的梯度，并返回结果
Tensor smooth_l1_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double beta) {
  // 创建一个和 input 同样大小的全零 Tensor 作为 grad_input
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用 smooth_l1_loss_backward_out 函数计算梯度
  return at::smooth_l1_loss_backward_out(grad_input, grad_output, input, target, reduction, beta);
}

// 计算 Huber 损失函数
Tensor huber_loss(const Tensor& input, const Tensor& target, int64_t reduction, double delta) {
  // 检查 delta 是否大于 0
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
  // 创建一个和 input 相同大小的空 Tensor 用于存储损失值
  Tensor loss = at::empty_like(input);
  // 创建 Tensor 迭代器来执行 Huber 损失计算
  auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
  huber_stub(iter.device_type(), iter, delta);
  // 应用指定的损失函数减少方式，并返回结果
  return apply_loss_reduction(loss, reduction);
}

// 计算 Huber 损失函数，并将结果存储到指定的 result 中
Tensor& huber_loss_out(const Tensor& input, const Tensor& target, int64_t reduction, double delta, Tensor& result) {
  // 检查 delta 是否大于 0
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
  // 创建 Tensor 迭代器来执行 Huber 损失计算
  auto iter = TensorIterator::borrowing_binary_op(result, input, target);
  huber_stub(iter.device_type(), iter, delta);
  // 如果 reduction 不为 None，则应用损失函数减少方式并更新 result
  if (reduction != Reduction::None) {
    auto reduced = apply_loss_reduction(result, reduction);
    result.resize_({});  // 将 result 重置为标量
    result.copy_(reduced);  // 将减少后的结果复制到 result 中
  }
  return result;
}

// 计算 Huber 损失函数对输入的梯度
Tensor huber_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double delta) {
  // 创建一个和 input 同样大小的全零 Tensor 作为 grad_input
  auto grad_input = at::zeros_like(input, MemoryFormat::Contiguous);
  // 调用 huber_loss_backward_out 函数计算梯度
  return at::huber_loss_backward_out(grad_input, grad_output, input, target, reduction, delta);
}

// 计算 Huber 损失函数对输入的梯度，并存储到给定的 grad_input 中
Tensor& huber_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double delta, Tensor& grad_input) {
  // 根据 reduction 类型确定归一化系数
  auto norm = (reduction == Reduction::Mean) ? (1. / input.numel()) : 1.;
  // 创建 Tensor 迭代器配置对象，配置输出、输入常量、以及类型转换等参数
  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(input)
    .add_const_input(target)
    .add_const_input(grad_output)
    .build();
  // 调用 Huber 损失函数的反向传播方法
  huber_backward_stub(iter.device_type(), iter, norm, delta);
  return grad_input;
}

// 计算 MSE 损失函数对输入的梯度，并返回结果
Tensor mse_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  // 创建一个和 input 同样大小的全零 Tensor 作为 grad_input
  Tensor grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 调用 mse_loss_backward_out 函数计算梯度
  return at::mse_loss_backward_out(grad_input, grad_output, input, target, reduction);
}
    const Tensor& input, const Tensor& target, int64_t reduction, Tensor& grad_input) {

参数说明：接收四个参数，分别是输入张量 `input`，目标张量 `target`，缩减方式 `reduction`，以及梯度输入张量 `grad_input`。


  auto norm = reduction == Reduction::Mean ? 2. / input.numel() : 2.;

计算规范化因子 `norm`，根据缩减方式 `reduction` 的不同，如果是平均缩减方式 (`Reduction::Mean`)，则规范化因子为 `2. / input.numel()`，否则为 `2.0`。


  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(input)
    .add_const_input(target)
    .add_const_input(grad_output)
    .build();

配置张量迭代器 `iter`，用于对输入张量 `input`、目标张量 `target`、梯度输出张量 `grad_output` 进行迭代计算，并将结果写入梯度输入张量 `grad_input`。


  mse_backward_stub(iter.device_type(), iter, norm);

调用均方误差反向传播的底层函数 `mse_backward_stub`，传递设备类型、迭代器 `iter` 和规范化因子 `norm`，执行梯度计算。


  return grad_input;

返回计算得到的梯度输入张量 `grad_input`。
}

Tensor l1_loss(const Tensor& input, const Tensor& target, int64_t reduction) {
    // 计算输入张量与目标张量的绝对差值，然后应用指定的减少（reduction）操作
    return apply_loss_reduction((input - target).abs(), reduction);
}
}  // namespace at::native
```