# `.\pytorch\aten\src\ATen\functorch\BatchRulesConvolution.cpp`

```py
// 包含头文件：BatchRulesHelper.h，PlumbingHelper.h，Dispatcher.h，这些文件提供了所需的函数和类
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/core/dispatch/Dispatcher.h>

// 定义命名空间 at::functorch，用于组织函数和类，避免命名冲突
namespace at::functorch {

// convolution_batch_rule 函数的声明和文档说明，这是从 JAX 转换并经过修改的卷积批处理规则
// 参考链接：https://github.com/google/jax/blob/master/jax/_src/lax/lax.py#L3143
static std::tuple<Tensor,optional<int64_t>>
convolution_batch_rule(const Tensor& lhs, optional<int64_t> lhs_bdim, const Tensor& rhs, optional<int64_t> rhs_bdim, const optional<Tensor>& bias, optional<int64_t> bias_bdim, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups) {
  
  // 创建 lhs_spec，rhs_spec，out_spec 作为维度向量，根据 stride，padding，dilation 配置大小
  DimVector lhs_spec(stride.size() + 2);
  std::iota(lhs_spec.begin(), lhs_spec.end(), 0);
  DimVector rhs_spec = lhs_spec;
  DimVector out_spec = lhs_spec;
  
  // 如果是转置卷积，修改 rhs_spec 的维度顺序
  if (transposed) {
    rhs_spec[0] = 1;
    rhs_spec[1] = 0;
  }

  // 处理批次的 bias 或 weight 的分离计算
  optional<Tensor> unbatched_bias;
  bool separate_bias = false;
  if ((rhs_bdim && bias && bias->defined()) || bias_bdim) {
    TORCH_INTERNAL_ASSERT(bias.has_value());
    TORCH_INTERNAL_ASSERT(bias->defined());
    unbatched_bias = nullopt;
    separate_bias = true;
  } else {
    unbatched_bias = bias;
    separate_bias = false;
  }

  std::tuple<Tensor, optional<int64_t>> result;

  // 根据是否有左边维度和右边维度来调整输入，并执行卷积操作
  if (lhs_bdim && !rhs_bdim) {
    auto new_x = reshape_dim_into(*lhs_bdim, lhs_spec[0], lhs);
    auto out = at::convolution_symint(new_x, rhs, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
    out = reshape_dim_outof_symint(out_spec[0], lhs.sizes()[*lhs_bdim], out);
    result = std::make_tuple(out, out_spec[0]);
  } else if (!lhs_bdim && rhs_bdim) {
    if (groups == 1) {
      auto new_w = reshape_dim_into(*rhs_bdim, rhs_spec[0], rhs);
      auto out = at::convolution_symint(lhs, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
      out = reshape_dim_outof_symint(out_spec[1], rhs.size(*rhs_bdim), out);
      result = std::make_tuple(out, out_spec[1]);
      // 如果 groups 不等于 1，执行一些其他操作，但代码截断在此


这段代码是一个 C++ 函数的声明和定义，实现了卷积操作的批处理规则，主要处理了输入张量的维度调整和卷积计算。
    } else {
      if (transposed) {
        // conv_transpose with groups is normally NIHW, IOHW -> N(GO)HW
        // With RHS batched, we do the following:
        // NIHW, BIOHW -> NIHW, I(BO)HW -> N(GBO)HW -> BN(GO)HW
        // NB: the following isn't written using rhs_spec
        // (PyTorch convs have a fixed dimension order)

        // BIOHW -> I(BO)HW
        auto new_w = reshape_dim_into(*rhs_bdim, 1, rhs);
        // NIHW, I(BO)HW -> N(GBO)HW
        auto out = at::convolution_symint(lhs, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
        // N(GBO)HW -> NG(BO)HW
        out = reshape_dim_outof_symint(1, groups, out);
        // NG(BO)HW -> NGBOHW
        out = reshape_dim_outof_symint(2, rhs.size(*rhs_bdim), out);
        // NGBOHW -> NB(GO)HW
        out = reshape_dim_into(1, 2, out);
        result = std::make_tuple(out, 1);
      } else {
        // conv with groups is normally N(GI)HW, (GO)IHW -> N(GO)HW
        // With RHS batched, we do the following:
        // N(GI)HW, B(GO)IHW -> N(GI)HW, (GBO)IHW -> N(GBO)HW -> BN(GO)HW
        // NB: the following isn't written using rhs_spec
        // (PyTorch convs have a fixed dimension order)

        // B(GO)IHW -> BGOIHW
        auto new_w = reshape_dim_outof_symint(0 + (*rhs_bdim == 0), groups, rhs);
        // BGOIHW -> G(BO)IHW
        new_w = reshape_dim_into(*rhs_bdim + (*rhs_bdim > 0), 1, new_w);
        // G(BO)IHW -> (GBO)IHW
        new_w = reshape_dim_into(0, 0, new_w);
        // N(GI)HW, (GBO)IHW -> N(GBO)HW
        auto out = at::convolution_symint(lhs, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
        // N(GBO)HW -> NG(BO)HW
        out = reshape_dim_outof_symint(1, groups, out);
        // NG(BO)HW -> NGBOHW
        out = reshape_dim_outof_symint(2, rhs.size(*rhs_bdim), out);
        // NGBOHW -> NB(GO)HW
        out = reshape_dim_into(1, 2, out);
        result = std::make_tuple(out, 1);
      }
    }
  } else if (lhs_bdim && rhs_bdim) {
    // Handle the case when both lhs and rhs have batch dimensions
    auto new_x = reshape_dim_into(*lhs_bdim, lhs_spec[1], lhs);
    groups *= lhs.sizes()[*lhs_bdim];
    auto dim_with_groups = transposed ? 1 : 0;
    auto new_w = reshape_dim_into(*rhs_bdim, rhs_spec[dim_with_groups], rhs);
    auto out = at::convolution_symint(new_x, new_w, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups);
    out = reshape_dim_outof_symint(out_spec[1], lhs.sizes()[*lhs_bdim], out);
    result = std::make_tuple(out, out_spec[1]);
  } else {
    // Handle the case when neither lhs nor rhs has a batch dimension
    result = std::make_tuple(at::convolution_symint(lhs, rhs, unbatched_bias, stride, padding, dilation, transposed, output_padding, groups), nullopt);
  }
  if (separate_bias) {
    // If separate_bias is true, move the batch dimension to the front for both tensors A and B
    auto A = std::get<0>(result);
    auto A_batch_dim = std::get<1>(result);
    auto B = *bias;
    auto B_batch_dim = bias_bdim;
    A = moveBatchDimToFront(A, A_batch_dim);
    B = moveBatchDimToFront(B, B_batch_dim);
    # 循环迭代直到倒数第二个元素，对张量 B 进行尺寸扩展（unsqueeze）
    for (size_t i = 0; i < out_spec.size() - 2; i++) {
      B = B.unsqueeze(-1);
    }
    # 使用 maybePadToLogicalRank 函数对张量 B 进行可能的填充操作，
    # 以使其达到与张量 A 相同的逻辑秩（rankWithoutBatchDim 返回去除批次维度后的秩）
    B = maybePadToLogicalRank(B, B_batch_dim, rankWithoutBatchDim(A, A_batch_dim));

    # 返回一个元组，包含 A 和 B 相加的结果以及数字 0
    return std::make_tuple(at::add(A, B), 0);
  } else {
    # 如果条件不满足，则直接返回 result
    return result;
  }
}

// 定义静态函数 `_convolution_decomp`，用于执行卷积操作的分解
static Tensor _convolution_decomp(
    // 输入张量、权重张量及可选的偏置张量
    const Tensor& input_r, const Tensor& weight_r, const std::optional<Tensor>& bias_r_opt,
    // 步长、填充、膨胀参数
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    // 是否转置、输出填充参数、分组数
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    // 是否进行基准测试、确定性、是否启用cuDNN、是否允许使用TF32
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  // 忽略这些参数。如果用户正常调用此函数，则应该无需关心这些参数。
  (void) benchmark;
  (void) deterministic;
  (void) cudnn_enabled;
  (void) allow_tf32;
  // 调用ATen库的卷积函数，返回卷积计算结果张量
  return at::convolution(
      input_r, weight_r, bias_r_opt, stride_, padding_, dilation_, transposed_, output_padding_, groups_);
}

// TODO: 在确认性能后删除以下内容
// bool first_dim_has_size_1(const Tensor& value, int64_t bdim) {
//   if (bdim == 0) {
//     return value.size(1) == 1;
//   }
//   return value.size(0) == 1;
// }
//
// std::tuple<Tensor,int64_t,Tensor,int64_t> cudnn_conv_per_sample_grad_rule(
//     const Tensor& self, optional<int64_t> self_bdim,
//     const Tensor& grad_output, optional<int64_t> grad_output_bdim,
//     const Tensor& weight, optional<int64_t> weight_bdim,
//     IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark,
//     bool deterministic, bool allow_tf32, std::array<bool, 2> output_mask) {
//   TORCH_INTERNAL_ASSERT(self_bdim && grad_output_bdim && !weight_bdim);
//   // TODO: No clue if this works if the first non-batch dim isn't size 1
//   TORCH_INTERNAL_ASSERT(first_dim_has_size_1(self, *self_bdim));
//   TORCH_INTERNAL_ASSERT(self.dim() == 5);
//
//   auto bdim_size = self.size(*self_bdim);
//   auto self_ = reshape_dim_into(*self_bdim, 0, self);
//   auto in_channels = self_.size(1);
//   auto grad_output_ = reshape_dim_into(*grad_output_bdim, 0, grad_output);
//
//   auto grad_self = at::cudnn_convolution_backward_input(
//       self_.sizes(), grad_output_, weight,
//       padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
//   grad_self = reshape_dim_outof(0, bdim_size, grad_self);
//
//   // Copied from https://github.com/pytorch/opacus/blob/master/opacus/grad_sample/conv.py
//   auto A = at::im2col(self_, {weight.size(2), weight.size(3)}, dilation, padding, stride);
//   auto B = grad_output_.reshape({bdim_size, -1, A.size(-1)});
//   auto grad_sample = at::einsum("noq,npq->nop", {B, A});
//   grad_sample = grad_sample.view({
//       bdim_size, groups, -1, groups, in_channels / groups,
//       weight.size(2) * weight.size(3) });
//   grad_sample = at::einsum("ngrg...->ngr...", {grad_sample});
//   grad_sample = grad_sample.reshape(
//       {bdim_size, weight.size(0), weight.size(1), weight.size(2), weight.size(3)});
//
//   return std::make_tuple(grad_self, 0, grad_sample, 0);
// }
//
// 定义函数 cudnn_convolution_backward_plumbing，计算 CUDNN 卷积反向传播
static std::tuple<Tensor, Tensor> cudnn_convolution_backward_plumbing(
    const Tensor & self,                   // 输入张量 self
    const Tensor & grad_output,            // 梯度输出张量 grad_output
    const Tensor & weight,                 // 权重张量 weight
    IntArrayRef padding,                   // 填充数组 padding
    IntArrayRef stride,                    // 步幅数组 stride
    IntArrayRef dilation,                  // 扩张数组 dilation
    int64_t groups,                        // 组数
    bool benchmark,                        // 是否基准测试
    bool deterministic,                    // 是否确定性操作
    bool allow_tf32,                       // 是否允许 TF32 模式
    std::array<bool, 2> output_mask) {     // 输出掩码数组
  auto maybe_layer = maybeCurrentDynamicLayer();    // 获取当前动态图层
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());  // 断言当前图层有效
  int64_t cur_level = maybe_layer->layerId();       // 获取当前图层 ID

  Tensor self_value;                        // 定义 self_value 张量
  optional<int64_t> self_bdim;              // self 的批次维度（如果有）
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);  // 解封 self 张量在当前层级下的值和批次维度
  Tensor grad_output_value;                 // 定义 grad_output_value 张量
  optional<int64_t> grad_output_bdim;       // grad_output 的批次维度（如果有）
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);  // 解封 grad_output 张量在当前层级下的值和批次维度
  Tensor weight_value;                      // 定义 weight_value 张量
  optional<int64_t> weight_bdim;            // weight 的批次维度（如果有）
  std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);  // 解封 weight 张量在当前层级下的值和批次维度

  // 如果 self 的批次维度存在且 self_value 的维度为 5，并且 self_value 的第一个维度大小为 1，且 grad_output 的批次维度存在且 weight 的批次维度不存在
  if (self_bdim.has_value() && self_value.dim() == 5 && first_dim_has_size_1(self_value, *self_bdim) && grad_output_bdim.has_value() && !weight_bdim.has_value()) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);  // 排除 FuncTorchBatched 调度键的保护
    auto result = cudnn_conv_per_sample_grad_rule(  // 使用 cudnn_conv_per_sample_grad_rule 计算
        self_value, self_bdim,
        grad_output_value, grad_output_bdim,
        weight_value, weight_bdim,
        padding, stride, dilation, groups,
        benchmark, deterministic, allow_tf32, output_mask);  // 参数传递给 cudnn_conv_per_sample_grad_rule 计算结果
    return std::make_tuple(   // 返回批量化后的结果元组
        makeBatched(std::get<0>(result), std::get<1>(result), cur_level),  // 对结果进行批量化处理
        makeBatched(std::get<2>(result), std::get<3>(result), cur_level));  // 对结果进行批量化处理
  }

  static auto op = c10::Dispatcher::singleton()   // 获取静态的分发器
    .findSchemaOrThrow("aten::cudnn_convolution_backward", "");  // 查找或抛出 "aten::cudnn_convolution_backward" 模式的架构
  return slow_fallback<Tensor, Tensor>(op, { self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask });  // 返回缓慢的回退函数的结果
}

// 定义函数 compute_grad_bias，计算梯度偏置
static Tensor compute_grad_bias(
    const Tensor& grad_output_,            // 输入梯度输出张量 grad_output_
    std::array<bool, 3> output_mask) {     // 输出掩码数组
  if (!output_mask[2]) {                   // 如果输出掩码数组中的第三个元素为假
    return Tensor();                      // 返回空张量
  }
  DimVector reduce_dims;                    // 定义维度向量 reduce_dims
  reduce_dims.resize(grad_output_.dim() - 1);  // 调整 reduce_dims 的大小为 grad_output_ 的维度数减去 1
  reduce_dims[0] = 0;                       // 设置 reduce_dims 的第一个元素为 0
  std::iota(reduce_dims.begin() + 1, reduce_dims.end(), 2);  // 从 2 开始为 reduce_dims 的剩余元素填充增量值
  return grad_output_.sum(reduce_dims);     // 返回在 reduce_dims 上对 grad_output_ 进行求和的结果
}

// 定义函数 make_dummy，创建虚拟张量
static Tensor make_dummy(
    const Tensor& tensor,                  // 输入张量 tensor
    optional<int64_t> tensor_bdim,         // tensor 的批次维度（如果有）
    int64_t dim,                           // 维度 dim
    int64_t batch_size) {                  // 批量大小
  auto tensor_ = tensor_bdim ? tensor.select(*tensor_bdim, 0) : tensor;  // 如果 tensor_bdim 存在，则选择 tensor_bdim 维度的第一个切片为 tensor_
  auto orig_size = tensor_.size(dim);      // 获取 tensor_ 在维度 dim 上的原始大小
  tensor_ = tensor_.slice(dim, 0, 1);      // 在维度 dim 上对 tensor_ 进行切片，取第 0 到 1 的部分

  DimVector expand_shape(tensor_.sizes().begin(), tensor_.sizes().end());  // 定义扩展形状，从 tensor_.sizes().begin() 到 tensor_.sizes().end() 的张量
  expand_shape[dim] = batch_size * orig_size;  // 在维度 dim 上设置扩展形状的大小为 batch_size 乘以 orig_size

  return tensor_.new_empty({}).expand(expand_shape);  // 返回在扩展形状上进行扩展的 tensor_ 的新空张量
}

// 定义函数 convolution_backward_input_batch_rule，卷积反向传播输入批处理规则
static std::tuple<Tensor, optional<int64_t>>
convolution_backward_input_batch_rule(
    const Tensor& grad_output,             // 梯度输出张量 grad_output
    optional<int64_t> grad_output_bdim,    // grad_output 的批次维度（如果有）
    const Tensor& input,                   // 输入张量 input
    optional<int64_t> input_bdim,          // input 的批次维度（如果有）
    const Tensor& weight, optional<int64_t> weight_bdim,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, const c10::SymInt& groups) {
  // 定义一个布尔型的固定大小数组，表示是否对应的维度是批量维度
  const std::array<bool, 3> mask = {true, false, false};
  // 如果存在梯度输出维度和权重维度
  if (grad_output_bdim && weight_bdim) {
    // 普通情况下和转置情况下的注释
    // BNO, BOI -> N(BO), (BO)I -> N(BI)
    // transposed: BNO, BIO -> N(BO), (BI)O -> N(BI)
    // 计算批量大小
    const auto batch_size = weight.size(*weight_bdim);
    // 将梯度输出张量按指定维度重塑
    const auto grad_output_ = reshape_dim_into(*grad_output_bdim, 1, grad_output);
    // 将权重张量按指定维度重塑
    const auto weight_ = reshape_dim_into(*weight_bdim, 0, weight);
    // 创建虚拟输入
    auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);
    // 执行符号整数卷积的反向传播
    const auto result = at::convolution_backward_symint(
        grad_output_, dummy_input, weight_, nullopt, stride, padding,
        dilation, transposed, output_padding, groups * batch_size, mask);
    // 将梯度输入张量按指定维度重塑
    const auto grad_input = reshape_dim_outof(1, batch_size, std::get<0>(result));
    // 返回梯度输入和标记值1的元组
    return std::make_tuple(grad_input, 1);
  } else if (grad_output_bdim && !weight_bdim) {
    // BNO, OI -> (BN)O, OI -> (BN)I
    // transposed is the same.
    // 计算批量大小
    const auto batch_size = grad_output.size(*grad_output_bdim);
    // 将梯度输出张量按指定维度重塑
    const auto grad_output_ = reshape_dim_into(*grad_output_bdim, 0, grad_output);
    // 创建虚拟输入
    auto dummy_input = make_dummy(input, input_bdim, 0, batch_size);
    // 执行符号整数卷积的反向传播
    const auto result = at::convolution_backward_symint(
        grad_output_, dummy_input, weight, nullopt, stride, padding,
        dilation, transposed, output_padding, groups, mask);
    // 将梯度输入张量按指定维度重塑
    const auto grad_input = reshape_dim_outof(0, batch_size, std::get<0>(result));
    // 返回梯度输入和标记值0的元组
    return std::make_tuple(grad_input, 0);
  } else if (!grad_output_bdim && weight_bdim) {
    // 计算批量大小
    const auto batch_size = weight.size(*weight_bdim);
    // 如果分组数为1
    if (groups == 1) {
      // 普通情况下和转置情况下的注释
      // regular: NO, BOI -> NO, O(BI) -> N(BI)
      // transposed: NO, BIO -> NO, (BI)O -> N(BI)
      // 确定输入通道维度的位置
      const auto in_ch_dim = transposed ? 0 : 1;
      // 将权重张量按指定维度重塑
      const auto weight_ = reshape_dim_into(*weight_bdim, in_ch_dim, weight);
      // 创建虚拟输入
      auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);
      // 执行符号整数卷积的反向传播
      const auto result = at::convolution_backward_symint(
          grad_output, dummy_input, weight_, nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask);
      // 将梯度输入张量按指定维度重塑
      const auto grad_input = reshape_dim_outof(1, batch_size, std::get<0>(result));
      // 返回梯度输入和标记值1的元组
      return std::make_tuple(grad_input, 1);
    }
    // 声明一个张量变量用于梯度输入
    Tensor grad_input;
    // 如果不是转置操作
    if (!transposed) {
      // N(GO), B(GO)I -> N(GO), (GO)(BI) -> N(GBI)
      // 将权重张量按指定维度重塑
      const auto weight_ = reshape_dim_into(*weight_bdim, 1, weight);
      // 创建虚拟输入
      auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);
      // 执行符号整数卷积的反向传播
      const auto result = at::convolution_backward_symint(
          grad_output, dummy_input, weight_, nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask);
      // 获取结果中的梯度输入
      grad_input = std::get<0>(result); // N(GBI)
      // 返回梯度输入和标记值1的元组
      return std::make_tuple(grad_input, 1);
    }
    } else {
      // N(GO), B(GI)O -> N(GO), (GBI)O -> N(GBI)
      // 将批量维度移动到前面，例如将 B(GI)O 转换为 GBIO
      auto weight_ = moveBatchDimToFront(weight, weight_bdim);
      // 将维度 1 和 groups 插入到 weight_ 的形状中，例如将 BGIO 转换为 N(GBI)
      weight_ = reshape_dim_outof_symint(1, groups, weight_);
      // 转置 weight_ 的维度 0 和 1，例如将 GBIO 转换为 GBIO
      weight_ = weight_.transpose(0, 1);
      // 将 weight_ 的前两个维度展平，例如将 (GBI)O 转换为 (GBI)O
      weight_ = weight_.flatten(0, 2);
      // 创建输入的虚拟数据，例如 make_dummy(input, input_bdim, 1, batch_size)
      const auto dummy_input = make_dummy(input, input_bdim, 1, batch_size);
      // 执行反向对称整数卷积操作，例如 at::convolution_backward_symint(...)
      const auto result = at::convolution_backward_symint(
          grad_output, dummy_input, weight_, nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask);
      // 获取卷积反向操作的梯度输入结果，例如 std::get<0>(result)
      grad_input = std::get<0>(result); // N(GBI)
    }
    // N(GBI) -> NG(BI) -> NGBI -> NBGI -> NB(GI)
    // 将维度 1 和 groups 插入到 grad_input 的形状中，例如 NG(BI)
    grad_input = reshape_dim_outof_symint(1, groups, grad_input);
    // 将维度 2 和 batch_size 插入到 grad_input 的形状中，例如 NGBI
    grad_input = reshape_dim_outof_symint(2, batch_size, grad_input);
    // 转置 grad_input 的维度 1 和 2，例如 NBGI
    grad_input = grad_input.transpose(1, 2);
    // 将 grad_input 的维度 2 划分成两个，例如 NB(GI)
    grad_input = reshape_dim_into(2, 2, grad_input);
    // 返回梯度输入和一个值为 1 的元组
    return std::make_tuple(grad_input, 1);
  } else {
    // 如果 input_bdim 为真，则执行以下代码块
    TORCH_INTERNAL_ASSERT(input_bdim);
    // 创建输入的虚拟数据，例如 make_dummy(input, input_bdim, 0, 1)
    const auto dummy_input = make_dummy(input, input_bdim, 0, 1);
    // 执行反向对称整数卷积操作，例如 at::convolution_backward_symint(...)
    const auto result = at::convolution_backward_symint(
        grad_output, dummy_input, weight, nullopt, stride, padding,
        dilation, transposed, output_padding, groups, mask);
    // 返回梯度输入和空的可选值元组
    return std::make_tuple(std::get<0>(result), nullopt);
  }
}
static std::tuple<Tensor,optional<int64_t>>
convolution_backward_weight_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& weight, optional<int64_t> weight_bdim,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, const c10::SymInt& groups) {
  // 定义用于确定输入、输出和权重批处理维度的掩码
  const std::array<bool, 3> mask = {false, true, false};
  // 如果梯度输出维度和输入维度均存在
  if (grad_output_bdim && input_bdim) {
    // 计算输入的批处理大小
    const auto batch_size = input.size(*input_bdim);
    // 将梯度输出张量重塑，将其批处理维度设置为1
    const auto grad_output_ = reshape_dim_into(*grad_output_bdim, 1, grad_output);
    // 将输入张量重塑，将其批处理维度设置为1
    const auto input_ = reshape_dim_into(*input_bdim, 1, input);
    // 创建虚拟权重，将权重的批处理维度设置为0，大小为批处理大小
    const auto dummy_weight = make_dummy(weight, weight_bdim, 0, batch_size);
    // 调用对称整数的卷积反向传播函数
    const auto result = at::convolution_backward_symint(
        grad_output_, input_, dummy_weight, nullopt, stride, padding,
        dilation, transposed, output_padding, groups * batch_size, mask);
    // 获取梯度权重并重塑，将批处理维度设置为0
    auto grad_weight = std::get<1>(result);
    grad_weight = reshape_dim_outof_symint(0, batch_size, grad_weight);
    // 返回梯度权重和输出通道维度0的元组
    return std::make_tuple(grad_weight, 0);
  } else if (grad_output_bdim && !input_bdim) {
    // 获取梯度输出张量的批处理大小
    const auto batch_size = grad_output.size(*grad_output_bdim);
    // 如果组数为1
    if (groups == 1) {
      // 根据是否转置选择不同的维度重塑方式
      // regular: BNO, NI -> N(BO), NI -> (BO)I
      // transposed: BNO, NI -> N(BO), NI -> I(BO)
      const auto grad_output_ = reshape_dim_into(*grad_output_bdim, 1, grad_output);
      const auto out_ch_dim = transposed ? 1 : 0;
      // 创建虚拟权重，将权重的批处理维度设置为输出通道维度，大小为批处理大小
      const auto dummy_weight = make_dummy(weight, weight_bdim, out_ch_dim, batch_size);
      // 调用对称整数的卷积反向传播函数
      const auto result = at::convolution_backward_symint(
          grad_output_, input, dummy_weight, nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask);
      // 获取梯度权重并重塑，将批处理维度设置为输出通道维度
      auto grad_weight = std::get<1>(result);
      grad_weight = reshape_dim_outof_symint(out_ch_dim, batch_size, grad_weight);
      // 返回梯度权重和输出通道维度的元组
      return std::make_tuple(grad_weight, out_ch_dim);
    } else {
      // 如果 grad_output_bdim 为假且 input_bdim 为真，则执行以下逻辑
      auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim); // 将批量维度移至前面，BN(GO)
      grad_output_ = reshape_dim_outof_symint(2, groups, grad_output_);              // 重塑符号整数中的维度，BNGO
      grad_output_ = grad_output_.movedim(0, 2);                              // 移动维度，NGBO
      grad_output_ = grad_output_.flatten(1, 3);                              // 展平维度，N(GBO)
      if (!transposed) {
        // 如果未转置，则执行以下逻辑
        // BN(GO), N(GI) -> N(GBO), N(GI) -> (GBO)I
        const auto dummy_weight = make_dummy(weight, weight_bdim, 0, batch_size); // 创建虚拟权重
        const auto result = at::convolution_backward_symint(
            grad_output_, input, dummy_weight, nullopt, stride, padding,
            dilation, transposed, output_padding, groups, mask); // 对称整数卷积反向传播
        auto grad_weight = std::get<1>(result); // 获取梯度权重
        grad_weight = grad_weight.unflatten_symint(0, { groups, batch_size, -1 }); // 反展平符号整数维度，GBOI
        grad_weight = grad_weight.transpose(0, 1);                          // 转置维度，BGOI
        grad_weight = grad_weight.flatten(1, 2);                            // 展平维度，B(GO)I
        return std::make_tuple(grad_weight, 0); // 返回梯度权重和标志0
      } else {
        // 如果已转置，则执行以下逻辑
        // BN(GO), N(GI) -> N(GBO), N(GI) -> (GI)(BO)
        const auto dummy_weight = make_dummy(weight, weight_bdim, 1, batch_size); // 创建虚拟权重
        const auto result = at::convolution_backward_symint(
            grad_output_, input, dummy_weight, nullopt, stride, padding,
            dilation, transposed, output_padding, groups, mask); // 对称整数卷积反向传播
        auto grad_weight = std::get<1>(result); // 获取梯度权重
        grad_weight = reshape_dim_outof_symint(1, batch_size, grad_weight); // 重塑符号整数中的维度，(GI)(BO)
        return std::make_tuple(grad_weight, 1); // 返回梯度权重和标志1
      }
    }
  } else if (!grad_output_bdim && input_bdim) {
    // 如果 grad_output_bdim 为假且 input_bdim 为真，则执行以下逻辑
    const auto batch_size = input.size(*input_bdim); // 获取批量大小
    if (groups == 1) {
      // 如果组数为1，则执行以下逻辑
      // regular: NO, BNI -> NO, N(BI) -> O(BI)
      // transposed: NO, BNI -> NO, N(BI) -> (BI)O
      const auto input_ = reshape_dim_into(*input_bdim, 1, input); // 重塑维度为1的输入
      const auto in_ch_dim = transposed ? 0 : 1; // 如果转置则选择0，否则选择1作为输入通道维度
      const auto dummy_weight = make_dummy(weight, weight_bdim, in_ch_dim, batch_size); // 创建虚拟权重
      const auto result = at::convolution_backward_symint(
          grad_output, input_, dummy_weight, nullopt, stride, padding,
          dilation, transposed, output_padding, groups, mask); // 对称整数卷积反向传播
      auto grad_weight = std::get<1>(result); // 获取梯度权重
      grad_weight = reshape_dim_outof_symint(in_ch_dim, batch_size, grad_weight); // 重塑符号整数中的维度
      return std::make_tuple(grad_weight, in_ch_dim); // 返回梯度权重和输入通道维度
    } else {
      auto input_ = moveBatchDimToFront(input, input_bdim); // 将输入张量的批量维度移到最前面，操作表示为 BN(GI)
      input_ = reshape_dim_outof_symint(2, groups, input_);        // 将输入张量重新形状化，添加维度 2，操作表示为 BNGI
      input_ = input_.movedim(0, 2);                        // 将输入张量的维度 0 移动到维度 2，操作表示为 NGBI
      input_ = input_.flatten(1, 3);                        // 将输入张量在维度 1 到 3 之间展平，操作表示为 N(GBI)
      if (!transposed) {
        // 对于非转置情况: 输入为 N(GO), BN(GI) -> N(GO), N(GBI) -> (GO)(BI)
        const auto dummy_weight = make_dummy(weight, weight_bdim, 1, batch_size);
        const auto result = at::convolution_backward_symint(
            grad_output, input_, dummy_weight, nullopt, stride, padding,
            dilation, transposed, output_padding, groups, mask);
        auto grad_weight = std::get<1>(result);
        grad_weight = reshape_dim_outof_symint(1, batch_size, grad_weight); // 将梯度权重张量重新形状化，添加维度 1，操作表示为 GO
        return std::make_tuple(grad_weight, 1); // 返回梯度权重和标志 1
      } else {
        // 对于转置情况: 输入为 N(GO), BN(GI) -> N(GO), N(GBI) -> (GBI)O
        const auto dummy_weight = make_dummy(weight, weight_bdim, 0, batch_size);
        const auto result = at::convolution_backward_symint(
            grad_output, input_, dummy_weight, nullopt, stride, padding,
            dilation, transposed, output_padding, groups, mask);
        auto grad_weight = std::get<1>(result);
        grad_weight = grad_weight.unflatten_symint(0, { groups, batch_size, -1 }); // 将梯度权重张量取消展平，操作表示为 GBIO
        grad_weight = grad_weight.transpose(0, 1);                          // 转置梯度权重张量，操作表示为 BGIO
        grad_weight = grad_weight.flatten(1, 2);                            // 展平梯度权重张量在维度 1 到 2 之间，操作表示为 B(GI)O
        return std::make_tuple(grad_weight, 0); // 返回梯度权重和标志 0
      }
    }
  } else {
    TORCH_INTERNAL_ASSERT(weight_bdim);
    const auto dummy_weight = make_dummy(weight, weight_bdim, 0, 1);
    const auto result = at::convolution_backward_symint(
        grad_output, input, dummy_weight, nullopt, stride, padding,
        dilation, transposed, output_padding, groups, mask);
    return std::make_tuple(std::get<1>(result), nullopt); // 返回梯度权重和空指针选项
  }
# 函数用于卷积反向传播的辅助功能，处理梯度、输入和权重
static std::tuple<Tensor,Tensor,Tensor> convolution_backward_plumbing(
    const Tensor& grad_output_, const Tensor& input_, const Tensor& weight_,
    const c10::OptionalArrayRef<SymInt> bias_sizes_opt,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups, std::array<bool, 3> output_mask) {
  # 获取当前动态图层的可能性
  const auto maybe_layer = maybeCurrentDynamicLayer();
  # 检查是否存在动态图层，并标记为“卷积反向传播”
  vmap_check_escaped(maybe_layer, "convolution_backward_plumbing");
  # 获取当前层级的层标识符
  int64_t cur_level = maybe_layer->layerId();

  # 如果在当前层级不存在任何批次张量，则进入此分支
  if (!areAnyBatchedAtLevel({grad_output_, input_, weight_}, cur_level)){
    # 使用排除分发键保护功能
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    # 调用符号整数卷积反向传播函数，并返回结果
    return at::convolution_backward_symint(
        grad_output_, input_, weight_, bias_sizes_opt, stride, padding,
        dilation, transposed, output_padding, groups, output_mask);
  }

  # 解开在当前层级的梯度输出、输入和权重张量
  auto [grad_output, grad_output_bdim] = unwrapTensorAtLevel(grad_output_, cur_level);
  auto [input, input_bdim] = unwrapTensorAtLevel(input_, cur_level);
  auto [weight, weight_bdim] = unwrapTensorAtLevel(weight_, cur_level);

  # 计算梯度偏置
  const auto grad_bias = compute_grad_bias(grad_output_, output_mask);
  # 禁用输出掩码的第三个元素
  output_mask[2] = false;

  // TODO: 有传言称，在许多情况下，展开 + 矩阵乘法比组卷积更快。我们应该对一些常见情况进行基准测试，并在必要时用展开 + 矩阵乘法替换一些东西。

  // 符号表示：
  // B - 批次维度
  // G - 组（有时会省略，因为它无关紧要）
  // NO - grad_output
  // NI - input
  // OI - weight
  // "(BO)I" - 我们实际上不关心此张量的值，我们只需创建一个具有正确形状的张量，并祈祷实现足够智能，不会对其执行任何操作。

  // BNO, BNI, BOI
  // 即模型集成的一种情况
  if (grad_output_bdim && input_bdim && weight_bdim) {
    # 使用排除分发键保护功能
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    # 在第一个维度上重塑梯度输出张量
    grad_output = reshape_dim_into(*grad_output_bdim, 1, grad_output);

    // BNO, BNI, BOI -> N(BO), N(BI), (BO)I
    const auto batch_size = weight.size(*weight_bdim);
    # 在第一个维度上重塑输入张量
    input = reshape_dim_into(*input_bdim, 1, input);
    # 在零维上重塑权重张量
    weight = reshape_dim_into(*weight_bdim, 0, weight);
    # 调用符号整数卷积反向传播函数，并返回结果
    const auto result = at::convolution_backward_symint(
        grad_output, input, weight, nullopt, stride, padding, dilation,
        transposed, output_padding, batch_size * groups, output_mask);
    // N(BI), (BO)I -> NBI, BOI
    # 如果输出掩码的第一个元素为真，则将输出形状重新调整为1
    const auto grad_input = output_mask[0] ?
      reshape_dim_outof(1, batch_size, std::get<0>(result)) : Tensor();
    # 如果输出掩码的第二个元素为真，则将输出形状重新调整为0
    const auto grad_weight = output_mask[1] ?
      reshape_dim_outof(0, batch_size, std::get<1>(result)) : Tensor();
    // 返回一个包含三个元素的 tuple：
    //   - 如果 output_mask[0] 为真，则将 grad_input 使用 makeBatched 处理成批量形式，否则直接返回 grad_input
    //   - 如果 output_mask[1] 为真，则将 grad_weight 使用 makeBatched 处理成批量形式，否则直接返回 grad_weight
    //   - 直接返回 grad_bias
    return std::make_tuple(
        output_mask[0] ? makeBatched(grad_input, 1, cur_level) : grad_input,
        output_mask[1] ? makeBatched(grad_weight, 0, cur_level) : grad_weight,
        grad_bias);
    }
    
    // 初始化 grad_input 变量
    Tensor grad_input;
    // 如果 output_mask[0] 为真，则执行以下代码块
    if (output_mask[0]) {
        // 创建一个 ExcludeDispatchKeyGuard 对象，排除 DispatchKey::FuncTorchBatched 键
        c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
        // 调用 convolution_backward_input_batch_rule 函数计算梯度
        const auto result = convolution_backward_input_batch_rule(
            grad_output, grad_output_bdim,
            input, input_bdim,
            weight, weight_bdim,
            stride, padding, dilation, transposed, output_padding, groups);
        // 使用 makeBatched 处理计算结果并赋值给 grad_input
        grad_input = makeBatched(std::get<0>(result), std::get<1>(result), cur_level);
    }
    
    // 初始化 grad_weight 变量
    Tensor grad_weight;
    // 如果 output_mask[1] 为真，则执行以下代码块
    if (output_mask[1]) {
        // 创建一个 ExcludeDispatchKeyGuard 对象，排除 DispatchKey::FuncTorchBatched 键
        c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
        // 调用 convolution_backward_weight_batch_rule 函数计算梯度
        const auto result = convolution_backward_weight_batch_rule(
            grad_output, grad_output_bdim,
            input, input_bdim,
            weight, weight_bdim,
            stride, padding, dilation, transposed, output_padding, groups);
        // 使用 makeBatched 处理计算结果并赋值给 grad_weight
        grad_weight = makeBatched(std::get<0>(result), std::get<1>(result), cur_level);
    }
    
    // 返回一个包含三个元素的 tuple：
    //   - grad_input：经过条件处理后的梯度
    //   - grad_weight：经过条件处理后的权重梯度
    //   - grad_bias：未经处理的偏置梯度
    return std::make_tuple(grad_input, grad_weight, grad_bias);
    
    // 下面的注释表明有人可能会发现批处理规则存在问题，因此留下了以下备用方案。
    // static auto op = c10::Dispatcher::singleton()
    //   .findSchemaOrThrow("aten::convolution_backward", "");
    // auto result = slow_fallback<Tensor,Tensor,Tensor>(op, {
    //   grad_output_, input_, weight_, bias_sizes_opt,
    //   stride, padding, dilation, transposed, output_padding, groups, output_mask
    // });
    // return std::make_tuple(grad_input, std::get<1>(result), grad_bias);
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 在 torch 的 aten 库中注册 FuncTorchBatched 实现
  VMAP_SUPPORT(convolution, convolution_batch_rule);
  // 启用批处理支持，指定 convolution 函数使用 convolution_batch_rule 规则
  m.impl("_convolution", _convolution_decomp);
  // 在模块 m 中实现 "_convolution" 函数，使用 _convolution_decomp 实现
  m.impl("convolution_backward", convolution_backward_plumbing);
  // 在模块 m 中实现 "convolution_backward" 函数，使用 convolution_backward_plumbing 实现
}

} // namespace at;:functorch
// 结束 at 命名空间下的 functorch 子命名空间定义
```