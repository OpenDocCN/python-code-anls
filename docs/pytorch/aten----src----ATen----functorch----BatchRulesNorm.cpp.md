# `.\pytorch\aten\src\ATen\functorch\BatchRulesNorm.cpp`

```py
// 包含头文件，用于处理批处理规则
#include <ATen/functorch/BatchRulesHelper.h>
// 包含头文件，用于处理管道辅助功能
#include <ATen/functorch/PlumbingHelper.h>
// 包含头文件，用于处理批处理回退操作
#include <ATen/functorch/BatchedFallback.h>
// 包含头文件，用于调度器
#include <ATen/core/dispatch/Dispatcher.h>

// 命名空间：at::functorch
namespace at::functorch {

// 检查张量是否为空
static bool is_empty_tensor(const Tensor& tensor) {
  // 获取张量形状
  const auto shape = tensor.sizes();
  // 返回形状为 [0] 的张量是否为空
  return shape.size() == 1 && shape[0] == 0;
}

// 计算统计批处理维度
static optional<int64_t> compute_stat_bdim(
    optional<int64_t> input_bdim,
    const Tensor& stat) {
  // 特殊情况：mean、rstd 可能形状为 (0,)，可能是 PyTorch 的 bug
  // 在这种情况下，不返回批处理张量
  if (input_bdim.has_value() && !is_empty_tensor(stat)) {
    // 如果输入批处理维度有值且统计张量不为空，返回0作为批处理维度
    return 0;
  }
  // 否则返回空值
  return nullopt;
}

// 右侧填充张量
static Tensor padRight(const Tensor& tensor, optional<int64_t> has_bdim, int64_t logical_rank) {
  // 注意：如果存在批处理维度，假设其为第一维度
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, has_bdim);
  // 如果张量的逻辑秩大于等于目标逻辑秩，则直接返回张量
  if (tensor_logical_rank >= logical_rank) {
    return tensor;
  }
  // 否则，构造新的大小向量，填充为1直至达到目标逻辑秩
  VmapDimVector new_sizes(tensor.sizes().begin(), tensor.sizes().end());
  for (int64_t i = 0; i < logical_rank - tensor_logical_rank; i++) {
    new_sizes.push_back(1);
  }
  // 返回调整后的张量
  return tensor.view(new_sizes);
}

// 批量归一化批处理规则模板
template<typename F, F Func>
std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>,Tensor,optional<int64_t>>
batch_norm_batch_rule(
    const Tensor& input, optional<int64_t> input_bdim,
    const std::optional<Tensor>& weight_opt, optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, optional<int64_t> bias_bdim,
    const std::optional<Tensor>& running_mean_opt, optional<int64_t> running_mean_bdim,
    const std::optional<Tensor>& running_var_opt, optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps) {
  // 将权重转换为常量引用
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  // 将偏置转换为常量引用
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  // 将运行时均值转换为常量引用
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const auto& running_mean = *running_mean_maybe_owned;
  // 将运行时方差转换为常量引用
  c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var_opt);
  const auto& running_var = *running_var_maybe_owned;
  // 检查是否需要进行批标准化计算
  TORCH_CHECK(!training || (!input_bdim || ((!running_mean.defined() || running_mean_bdim) && (!running_var.defined() || running_var_bdim))),
      "Batch norm got a batched tensor as input while the running_mean or running_var, which will be updated in place, ",
      "were not batched.\nIf you are using a module and do not need eval mode, please set `track_running_stats` to be False.",
      "If you are using a prebuilt module and do not need eval mode, please see the functorch website for resources on ",
      "how to patch your module to work with vmap");
  // 初始化可选的批次维度大小
  std::optional<int64_t> bdim_size;
  // 初始化结果张量
  Tensor result0;
  // 初始化均值张量
  Tensor mean;
  // 初始化反标准化标准差张量
  Tensor rstd;
  // 如果输入不包含批次维度且运行时均值及方差也不包含批次维度
  if (!input_bdim && !running_mean_bdim && !running_var_bdim) {
    // 创建一个与输入相同大小的全一张量作为虚拟权重
    const auto dummy_weight = at::ones(input.size(1), input.options());  // cudnn and miopen require a weight
    // 创建一个与输入相同大小的全零张量作为虚拟偏置
    const auto dummy_bias = at::zeros(input.size(1), input.options());   // without this, get "strides() called on undefined Tensor" on cuda
    // 调用函数 Func 进行计算，传入虚拟权重和偏置
    const auto result = Func(input, dummy_weight, dummy_bias, running_mean_opt, running_var_opt, training, momentum, eps);
    // 对结果的第一个张量进行转置，调整维度顺序为 [C, B, *]
    result0 = std::get<0>(result).transpose(0, 1);          // [C, B, *]
    // 获取计算得到的均值张量
    mean = std::get<1>(result);
    // 获取计算得到的反标准化标准差张量
    rstd = std::get<2>(result);
  } else {
    // 获取批次维度的大小
    bdim_size = get_bdim_size3(input, input_bdim, running_mean, running_mean_bdim, running_var, running_var_bdim);
    // 将批次维度移动到张量的最前面
    auto input_ = moveBatchDimToFront(input, input_bdim);
    // 确保输入张量具有指定的批次维度和大小
    input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size.value());
    // 将维度 0 重新塑形为 1（通道维度）
    input_ = reshape_dim_into(0, /*channels dim*/1, input_);

    // 初始化可选的运行时均值和方差张量
    std::optional<Tensor> running_mean_;
    std::optional<Tensor> running_var_;
    // 如果运行时均值已定义，则将其批次维度移动到最前面，并确保具有指定的批次维度和大小，并使其连续
    if (running_mean.defined()) {
      running_mean_ = moveBatchDimToFront(running_mean, running_mean_bdim);
      running_mean_ = ensure_has_bdim(*running_mean_, running_mean_bdim.has_value(), bdim_size.value());
      running_mean_ = reshape_dim_into(0, 0, *running_mean_).contiguous();
    }
    // 如果运行时方差已定义，则将其批次维度移动到最前面，并确保具有指定的批次维度和大小，并使其连续
    if (running_var.defined()) {
      running_var_ = moveBatchDimToFront(running_var, running_var_bdim);
      running_var_ = ensure_has_bdim(*running_var_, running_var_bdim.has_value(), bdim_size.value());
      running_var_ = reshape_dim_into(0, 0, *running_var_).contiguous();
    }

    // 创建一个与 input_ 大小相同的全一张量作为虚拟权重
    const auto dummy_weight = at::ones(input_.size(1), input_.options());  // cudnn and miopen require a weight
    // 创建一个与 input_ 同样大小的零张量作为 dummy_bias，用于避免在 CUDA 上出现 "strides() called on undefined Tensor" 错误
    const auto dummy_bias = at::zeros(input_.size(1), input_.options());
    
    // 调用 Func 函数处理 input_，dummy_weight，dummy_bias，running_mean_，running_var_ 等参数，返回一个元组 result
    const auto result = Func(input_, dummy_weight, dummy_bias, running_mean_, running_var_, training, momentum, eps);
    
    // 从 result 中获取第一个元素，并将其转置，结果存储在 result0 中，形状为 [(B0, C), B, *]
    result0 = std::get<0>(result).transpose(0, 1);
    
    // 将 result0 进行维度重塑，将第 0 维度（通常是 batch 维度）拉到最前面，形状变为 [B0, C, B, *]
    result0 = reshape_dim_outof(0, bdim_size.value(), result0);
    
    // 从 result 中获取第二个元素，赋值给 mean，并将其进行维度重塑，形状变为 [B0, C]
    mean = std::get<1>(result);
    mean = reshape_dim_outof(0, bdim_size.value(), mean);
    
    // 从 result 中获取第三个元素，赋值给 rstd，并将其进行维度重塑，形状变为 [B0, C]
    rstd = std::get<2>(result);
    rstd = reshape_dim_outof(0, bdim_size.value(), rstd);
  }

  // 计算统计数据的 batch 维度，使用 compute_stat_bdim 函数
  const auto stats_bdim = compute_stat_bdim(bdim_size, mean);
  
  // 如果 weight 已定义，则进行以下操作
  if (weight.defined()) {
    // 计算 input 的逻辑秩（排除 batch 维度），使用 rankWithoutBatchDim 函数
    const auto input_logical_rank = rankWithoutBatchDim(input, input_bdim);
    
    // 将 weight 的 batch 维度移动到最前面，使用 moveBatchDimToFront 函数
    auto weight_ = moveBatchDimToFront(weight, weight_bdim);
    
    // 在 weight_ 的右侧填充维度，使其与 input 的逻辑秩相匹配，使用 padRight 函数
    weight_ = padRight(weight_, weight_bdim, input_logical_rank);
    
    // 将 result0 乘以 weight_
    result0 = result0 * weight_;
  }
  
  // 如果 bias 已定义，则进行以下操作
  if (bias.defined()) {
    // 计算 result0 的逻辑秩，考虑到是否有 bdim_size 或 weight_bdim，使用 rankWithoutBatchDim 和 optional 值
    const auto result_logical_rank = rankWithoutBatchDim(
        result0,
        bdim_size.has_value() || weight_bdim.has_value() ? optional<int64_t>(0) : optional<int64_t>(nullopt));
    
    // 将 bias 的 batch 维度移动到最前面，使用 moveBatchDimToFront 函数
    auto bias_ = moveBatchDimToFront(bias, bias_bdim);
    
    // 在 bias_ 的右侧填充维度，使其与 result0 的逻辑秩相匹配，使用 padRight 函数
    bias_ = padRight(bias_, bias_bdim, result_logical_rank);
    
    // 将 result0 加上 bias_
    result0 = result0 + bias_;
  }
  
  // 将 result0 进行最后一次转置，将第 1 和第 2 维度交换位置，形状变为 [B0, B, C, *]
  result0 = result0.transpose(1, 2);
  
  // 返回一个包含 result0, 0, mean, stats_bdim, rstd, stats_bdim 的元组
  return std::make_tuple(result0, 0, mean, stats_bdim, rstd, stats_bdim);
  // 模板函数定义，接受一个函数指针作为模板参数，用于批量归一化反向传播计算，无权重和偏置
  template<typename F, F Func>
  // 返回类型为包含梯度输出张量和可选整数的元组
  std::tuple<at::Tensor,optional<int64_t>> batch_norm_backward_no_weight_bias_batch_rule(
      // 梯度输出张量及其批量维度（可选）
      const at::Tensor & grad_out, optional<int64_t> grad_out_bdim,
      // 输入张量及其批量维度（可选）
      const at::Tensor & input, optional<int64_t> input_bdim,
      // 运行时均值张量（可选）及其批量维度（可选）
      const std::optional<at::Tensor> & running_mean_opt, optional<int64_t> running_mean_bdim,
      // 运行时方差张量（可选）及其批量维度（可选）
      const std::optional<at::Tensor> & running_var_opt, optional<int64_t> running_var_bdim,
      // 均值张量及其批量维度
      const at::Tensor & mean, optional<int64_t> mean_bdim,
      // 反标准差张量及其批量维度
      const at::Tensor & rstd, optional<int64_t> rstd_bdim,
      // 训练标志和小数点位置修订
      bool training, double eps) {
    // 从运行时均值的可选张量借用所有权，并确保它不为空
    c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
    const Tensor& running_mean = *running_mean_maybe_owned;
    // 从运行时方差的可选张量借用所有权，并确保它不为空
    c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var_opt);
    const Tensor& running_var = *running_var_maybe_owned;

    // 如果梯度输出和输入的批量维度均未定义，且运行时均值和方差的批量维度也未定义
    if (!grad_out_bdim.has_value() && !input_bdim.has_value() && !running_mean_bdim.has_value() && !running_var_bdim.has_value()) {
      // 断言均值和反标准差的批量维度未定义
      TORCH_INTERNAL_ASSERT(!mean_bdim);
      TORCH_INTERNAL_ASSERT(!rstd_bdim);
      // 创建一个全为1的权重张量，形状与输入的第二个维度大小相同
      const auto dummy_weight = at::ones(input.size(1), input.options());
      // 调用模板函数 Func，执行批量归一化反向传播计算
      const auto result = Func(
          grad_out, input, dummy_weight, running_mean_opt, running_var_opt, mean, rstd, training, eps, {true, false, false});
      // 返回结果，包括计算得到的梯度输出张量和空的整数值
      return std::make_tuple(std::get<0>(result), nullopt);
    }

    // 将梯度输出、输入、均值和反标准差的批量维度移至最前
    auto grad_out_ = moveBatchDimToFront(grad_out, grad_out_bdim);
    auto input_ = moveBatchDimToFront(input, input_bdim);
    auto mean_ = moveBatchDimToFront(mean, mean_bdim);
    auto rstd_ = moveBatchDimToFront(rstd, rstd_bdim);

    // 获取所有输入张量的批量维度大小
    const auto bdim_size = get_bdim_size4(grad_out, grad_out_bdim, input, input_bdim, running_mean, running_mean_bdim, running_var, running_var_bdim);
    // 确保梯度输出、输入、均值和反标准差张量都具有批量维度
    grad_out_ = ensure_has_bdim(grad_out_, grad_out_bdim.has_value(), bdim_size);
    input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size);
    mean_ = ensure_has_bdim(mean_, mean_bdim.has_value(), bdim_size);
    rstd_ = ensure_has_bdim(rstd_, rstd_bdim.has_value(), bdim_size);

    // 可选的运行时均值和方差张量
    optional<Tensor> running_mean_;
    optional<Tensor> running_var_;
    // 如果运行时均值已定义，将其批量维度移至最前并确保具有批量维度，同时进行内存连续性操作
    if (running_mean.defined()) {
      running_mean_ = moveBatchDimToFront(running_mean, running_mean_bdim);
      running_mean_ = ensure_has_bdim(*running_mean_, running_mean_bdim.has_value(), bdim_size);
      running_mean_ = reshape_dim_into(0, 0, *running_mean_).contiguous();
    }
    // 如果运行时方差已定义，将其批量维度移至最前并确保具有批量维度
    if (running_var.defined()) {
      running_var_ = moveBatchDimToFront(running_var, running_var_bdim);
      running_var_ = ensure_has_bdim(*running_var_, running_var_bdim.has_value(), bdim_size);
      // 返回按行的内存操作
      running_var_ = reshape_dim_into(0, 0, *running_var_).contiguous();
    }
    // 调用函数 reshape_dim_into，将 running_var_ 在第一个维度（0）和第二个维度（0）之后的所有维度扩展并转换为连续存储
    running_var_ = reshape_dim_into(0, 0, *running_var_).contiguous();
  }

  // 调用函数 reshape_dim_into，将 input_ 在第一个维度（0）和第二个维度（channels dim，1）之后的所有维度扩展
  input_ = reshape_dim_into(0, /*channels dim*/1, input_);

  // 断言 mean_ 的维度为 2
  TORCH_INTERNAL_ASSERT(mean_.dim() == 2);

  // 断言 rstd_ 的维度为 2
  TORCH_INTERNAL_ASSERT(rstd_.dim() == 2);

  // 调用函数 reshape_dim_into，将 mean_ 在第一个维度（0）和第二个维度（0）之后的所有维度扩展
  mean_ = reshape_dim_into(0, 0, mean_);

  // 调用函数 reshape_dim_into，将 rstd_ 在第一个维度（0）和第二个维度（0）之后的所有维度扩展
  rstd_ = reshape_dim_into(0, 0, rstd_);

  // 将 grad_out_ 的维度交换第 0 和第 1 维度，并将其余维度展平，结果存储在 grad_out_ 中
  grad_out_ = grad_out_.transpose(0, 1).flatten(1, 2); // [B0, B, C, *] -> [B, (B0, C), *]

  // 创建一个包含全为 1 的权重 tensor，与 input_ 的第 1 维度大小相同
  const auto dummy_weight = at::ones(input_.size(1), input_.options());

  // 调用 native_batch_norm_backward 函数，计算批量归一化的反向传播结果
  auto result = at::native_batch_norm_backward(
      grad_out_.contiguous(),
      input_.contiguous(),
      dummy_weight,
      running_mean_,  // 如果给定了 tensor，则调用 contiguous
      running_var_,   // 如果给定了 tensor，则调用 contiguous
      mean_.contiguous(),
      rstd_.contiguous(),
      training, eps, {true, false, false});

  // 从 result 中获取第一个元素，将其在第 1 个维度（1）上展开为 bdim_size 的大小
  auto result0 = std::get<0>(result); // [B, B0, C, *]

  // 将 result0 的维度交换第 0 和第 1 维度，得到 [B0, B, C, *] 的形式
  result0 = reshape_dim_outof(1, bdim_size, result0);

  // 将 result0 的维度交换第 0 和第 1 维度，得到 [B0, B, C, *] 的形式
  result0 = result0.transpose(0, 1); // [B0, B, C, *]

  // 返回一个包含 result0 和整数 0 的 tuple
  return std::make_tuple(result0, 0);
// 结束函数模板的定义
template<typename F, F Func>
std::tuple<at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_plumbing(
    // 输入参数：梯度输出，输入张量，权重（可选），运行均值（可选），运行方差（可选），保存均值（可选），保存反标准差（可选），训练标志，epsilon值，输出掩码
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const std::optional<at::Tensor> & weight_opt,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    bool training,
    double eps,
    std::array<bool,3> output_mask) {
  // 查看 [Note: hacky wrapper removal for optional tensor] 注释
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  // 获取权重张量
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  // 获取运行均值张量
  const Tensor& running_mean = *running_mean_maybe_owned;
  c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var_opt);
  // 获取运行方差张量
  const Tensor& running_var = *running_var_maybe_owned;
  // 注意：为什么这些是可选的不太清楚...这些在前向传播中是必需的
  // 获取保存的均值张量
  const Tensor& save_mean = *save_mean_opt;
  // 获取保存的反标准差张量
  const Tensor& save_rstd = *save_rstd_opt;
  // 内部断言，确保保存的均值张量已定义
  TORCH_INTERNAL_ASSERT(save_mean.defined());
  // 内部断言，确保保存的反标准差张量已定义
  TORCH_INTERNAL_ASSERT(save_rstd.defined());

  // 管道处理
  // 获取当前动态层（如果存在）
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查当前动态层是否逃逸，以及逃逸的检查字符串
  vmap_check_escaped(maybe_layer, "batch_norm_backward_plumbing");
  // 获取当前层级的层ID
  int64_t cur_level = maybe_layer->layerId();

  // 获取梯度输出在当前层级的值和维度信息
  auto [grad_out_value, grad_out_bdim] = unwrapTensorAtLevel(grad_out, cur_level);
  // 获取输入张量在当前层级的值和维度信息
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  // 声明均值张量
  Tensor mean_value;
  // 声明权重值（可选）
  optional<Tensor> weight_value;
  // 声明权重维度（可选）
  optional<int64_t> weight_bdim;
  // 如果权重已定义，则获取权重值和维度
  if (weight.defined()) {
    std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  }
  // 声明运行均值值（可选）
  optional<Tensor> running_mean_value;
  // 声明运行均值维度（可选）
  optional<int64_t> running_mean_bdim;
  // 如果运行均值已定义，则获取运行均值值和维度
  if (running_mean.defined()) {
    std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean, cur_level);
  }
  // 声明运行方差值（可选）
  optional<Tensor> running_var_value;
  // 声明运行方差维度（可选）
  optional<int64_t> running_var_bdim;
  // 如果运行方差已定义，则获取运行方差值和维度
  if (running_var.defined()) {
    std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var, cur_level);
  }
  // 获取保存的均值值和维度
  auto [save_mean_value, save_mean_bdim] = unwrapTensorAtLevel(save_mean, cur_level);
  // 获取保存的反标准差值和维度
  auto [save_rstd_value, save_rstd_bdim] = unwrapTensorAtLevel(save_rstd, cur_level);

  // 结果
  // 声明梯度偏置张量
  Tensor grad_bias;
  // 声明梯度权重张量
  Tensor grad_weight;
  // 声明梯度输入张量
  Tensor grad_input;

  // 内部断言：确保梯度输出张量的维度大于1，因为批归一化不能对1维张量进行操作，输出至少是2维的
  TORCH_INTERNAL_ASSERT(grad_out_value.dim() > 1);
  // 如果输出掩码中对应位置为true，计算梯度偏置张量
  if (output_mask[2]) {
    grad_bias = grad_out.transpose(0, 1).sum(range(1, grad_out.dim()));
  }
  // 如果输出掩码中对应位置为true，并且权重值存在
  if (output_mask[1] && weight_value.has_value()) {
    // 注意：输出没有保存...
    // 如果是训练模式，使用保存的均值；否则使用运行时均值
    auto mean = training ? save_mean : running_mean;
    // 如果是训练模式，使用保存的反标准差；否则使用运行时的标准差倒数
    auto var = training ? save_rstd : (1 / at::sqrt(running_var + eps));
    // 计算归一化的输入值：将输入转置后减去均值并乘以方差的扩展值
    const auto normalized_input = (input.transpose(0, 1) - padRight(mean, nullopt, input.dim())) * padRight(var, nullopt, input.dim());

    // 扩展梯度权重：归一化输入乘以梯度输出的转置
    const auto expanded_grad_weight = normalized_input * grad_out.transpose(0, 1);

    // 计算梯度权重：对扩展梯度权重在第1到grad_out维度上求和
    grad_weight = expanded_grad_weight.sum(range(1, grad_out.dim()));
  }
  
  // 如果输出掩码的第一个元素为真
  if (output_mask[0]) {
    // 计算归一化后的输入梯度：如果权重已定义，则梯度输出的转置乘以权重，否则直接使用梯度输出的转置
    const auto grad_normalized_input = weight.defined() ?
      grad_out.transpose(0, 1) * padRight(weight, nullopt, grad_out.dim()) : grad_out.transpose(0, 1);           // [B0, C, B, *]
    
    // 拆解归一化后的输入梯度值和维度
    auto [grad_normalized_input_value, grad_normalized_input_bdim] =
        unwrapTensorAtLevel(grad_normalized_input.transpose(0, 1), cur_level);       // [B0, B, C, *]

    // 临时排除 DispatchKey 的保护区域
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);

    // 使用 batch_norm_backward_no_weight_bias_batch_rule 函数计算反向传播结果
    const auto results = batch_norm_backward_no_weight_bias_batch_rule<F, Func>(
        grad_normalized_input_value, grad_normalized_input_bdim,
        input_value, input_bdim,
        running_mean_value, running_mean_bdim,
        running_var_value, running_var_bdim,
        save_mean_value, save_mean_bdim,
        save_rstd_value, save_rstd_bdim,
        training, eps);

    // 生成批处理后的梯度输入
    grad_input = makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
  }

  // 返回梯度输入、梯度权重和梯度偏置的元组
  return std::make_tuple(grad_input, grad_weight, grad_bias);
static std::tuple<Tensor,Tensor,Tensor> native_group_norm_plumbing(
    // 定义 native_group_norm_plumbing 函数，接受多个参数：
    const Tensor & input,                      // 输入张量 input
    const std::optional<Tensor> & weight_opt,  // 可选参数，权重张量的可能性
    const std::optional<Tensor> & bias_opt,    // 可选参数，偏置张量的可能性
    int64_t N,                                 // 整数参数 N，表示批次大小
    int64_t C,                                 // 整数参数 C，表示通道数
    int64_t HxW,                               // 整数参数 HxW，表示高度乘宽度
    int64_t group,                             // 整数参数 group，表示组数
    double eps) {                              // 双精度参数 eps，表示 epsilon 值
  // See [Note: hacky wrapper removal for optional tensor]
  // 参见 [Note: hacky wrapper removal for optional tensor]

  // 从可选的权重张量中借用可能拥有的张量
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  // 引用借用的权重张量
  const Tensor& weight = *weight_maybe_owned;
  // 从可选的偏置张量中借用可能拥有的张量
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  // 引用借用的偏置张量
  const Tensor& bias = *bias_maybe_owned;

  // 获取当前可能的动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查是否有批处理在当前层中逃逸
  vmap_check_escaped(maybe_layer, "native_group_norm_plumbing");
  // 获取当前层的层级 ID
  int64_t cur_level = maybe_layer->layerId();

  // 如果在当前层级没有任何批处理的张量
  if (!areAnyBatchedAtLevel({input, weight_opt, bias_opt}, cur_level)) {
    // 临时排除 DispatchKey::FuncTorchBatched 调度键
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用原生的组归一化函数 native_group_norm
    return at::native_group_norm(input, weight_opt, bias_opt, N, C, HxW, group, eps);
  }

  // 解开当前层级的张量 input_value 和 input_bdim
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);

  Tensor result0;
  Tensor mean;
  Tensor rstd;
  // 如果存在 input_bdim
  if (input_bdim) {
    // 将 *input_bdim 重新整形到 input_value 的第 0 维
    const auto input_ = reshape_dim_into(*input_bdim, 0, input_value);
    // 获取 input_value 在 *input_bdim 维度的大小
    const auto bdim_size = input_value.size(*input_bdim);

    // 临时排除 DispatchKey::FuncTorchBatched 调度键
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用原生的组归一化函数 native_group_norm，传入重新整形后的 input_
    const auto result = at::native_group_norm(input_, nullopt, nullopt, N * bdim_size, C, HxW, group, eps);
    // 生成批处理后的 result0，将第 0 维重新整形到 bdim_size
    result0 = makeBatched(reshape_dim_outof(0, bdim_size, std::get<0>(result)), 0, cur_level);
    // 生成批处理后的 mean，将第 0 维重新整形到 bdim_size
    mean = makeBatched(reshape_dim_outof(0, bdim_size, std::get<1>(result)), 0, cur_level);
    // 生成批处理后的 rstd，将第 0 维重新整形到 bdim_size
    rstd = makeBatched(reshape_dim_outof(0, bdim_size, std::get<2>(result)), 0, cur_level);
  } else {
    // 临时排除 DispatchKey::FuncTorchBatched 调度键
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用原生的组归一化函数 native_group_norm，传入 input_value
    const auto result = at::native_group_norm(input_value, nullopt, nullopt, N, C, HxW, group, eps);
    // 获取 result 的第 0 维度
    result0 = std::get<0>(result);
    // 获取 result 的均值（mean）
    mean = std::get<1>(result);
    // 获取 result 的反标准差（rstd）
    rstd = std::get<2>(result);
  }

  // 如果定义了权重张量
  if (weight.defined()) {
    // 将权重张量右侧填充到与 result0 的维度减 1 相等
    const auto padded_weight = padRight(weight, nullopt, result0.dim() - 1);
    // 将 result0 乘以填充后的权重张量
    result0 = result0 * padded_weight;
  }

  // 如果定义了偏置张量
  if (bias.defined()) {
    // 将偏置张量右侧填充到与 result0 的维度减 1 相等
    const auto padded_bias = padRight(bias, nullopt, result0.dim() - 1);
    // 将 result0 加上填充后的偏置张量
    result0 = result0 + padded_bias;
  }

  // 返回结果元组，包括 result0、mean 和 rstd
  return std::make_tuple(result0, mean, rstd);
}
    int64_t N, int64_t C, int64_t HxW, int64_t group) {
  // 将梯度张量和输入张量移动批次维度到最前面
  auto grad_out_ = moveBatchDimToFront(grad_out, grad_out_bdim);
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto mean_ = moveBatchDimToFront(mean, mean_bdim);
  auto rstd_ = moveBatchDimToFront(rstd, rstd_bdim);

  // 计算批次维度大小
  const auto bdim_size = get_bdim_size2(grad_out, grad_out_bdim, input, input_bdim);
  // 确保梯度张量和输入张量都有批次维度
  grad_out_ = ensure_has_bdim(grad_out, grad_out_bdim.has_value(), bdim_size);
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size);
  mean_ = ensure_has_bdim(mean_, mean_bdim.has_value(), bdim_size);
  rstd_ = ensure_has_bdim(rstd_, rstd_bdim.has_value(), bdim_size);

  // 将张量重新整形，调整维度结构
  grad_out_ = reshape_dim_into(0, 0, grad_out_); // [B0 * N, C, *]
  input_ = reshape_dim_into(0, 0, input_);       // [B0 * N, C, *]
  mean_ = reshape_dim_into(0, 0, mean_);         // [B0 * N, G]
  rstd_ = reshape_dim_into(0, 0, rstd_);         // [B0 * N, G]

  // 执行本地组归一化的反向传播
  const auto result = native_group_norm_backward(
      grad_out_.contiguous(),
      input_.contiguous(),
      mean_.contiguous(),
      rstd_.contiguous(),
      nullopt, N * bdim_size, C, HxW, group, {true, false, false});
  auto result0 = std::get<0>(result);
  // 将结果重新整形，还原批次维度
  result0 = reshape_dim_outof(0, bdim_size, result0);
  // 返回包含结果和零的元组
  return std::make_tuple(result0, 0);
}
// static函数定义，用于实现Group Normalization的反向传播，返回三个Tensor作为结果
static std::tuple<Tensor,Tensor,Tensor> native_group_norm_backward_plumbing(
  // 输入参数：梯度输出grad_out，输入input，均值mean，标准差倒数rstd，可选权重weight_opt，
  // 其中N表示批量大小，C表示通道数，HxW表示高度乘以宽度，group表示分组数，output_mask表示输出掩码
  const Tensor & grad_out, const Tensor & input, const Tensor & mean,
  const Tensor & rstd, const std::optional<Tensor> & weight_opt,
  int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool,3> output_mask
) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 处理可选权重Tensor，获取其实际的Tensor引用
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // plumbing
  // 获取当前动态图层，用于跟踪和检查动态图层的变化
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查当前动态图层是否已经释放，如果已释放则输出警告信息
  vmap_check_escaped(maybe_layer, "native_group_norm_backward_plumbing");
  // 获取当前层级的层ID
  int64_t cur_level = maybe_layer->layerId();

  // 检查是否有任何张量在当前层级上批处理
  if (!areAnyBatchedAtLevel({grad_out, input, mean, rstd, weight_opt}, cur_level)) {
    // 排除FuncTorchBatched调度键，确保不进行批处理操作
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用原生的Group Normalization反向传播函数
    return at::native_group_norm_backward(grad_out, input, mean, rstd, weight_opt, N, C, HxW, group, output_mask);
  }

  // 解包输入张量和对应的批处理维度
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  optional<int64_t> weight_bdim;
  if (weight.defined()){
    std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  }
  auto [mean_value, mean_bdim] = unwrapTensorAtLevel(mean, cur_level);
  auto [rstd_value, rstd_bdim] = unwrapTensorAtLevel(rstd, cur_level);

  // 定义结果张量
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  // 断言梯度输出的维度大于1，因为Group Norm不能处理1维张量，输出至少是2维
  TORCH_INTERNAL_ASSERT(grad_out.dim() > 1);

  // 如果输出掩码中包含第3个位置，则计算grad_bias
  if (output_mask[2]) {
    grad_bias = grad_out.transpose(0, 1).sum(range(1, grad_out.dim()));
  }

  // 如果输出掩码中包含第2个位置且权重已定义，则计算grad_weight
  if (output_mask[1] && weight.defined()) {
    // 重塑输入张量以便按组进行归一化
    const auto reshaped_input = reshape_dim_outof(1, group, input);
    // 计算归一化后的输入
    const auto normalized_input = (reshaped_input - padRight(mean, nullopt, reshaped_input.dim())) * padRight(rstd, nullopt, reshaped_input.dim());
    // 扩展grad_weight并计算其总和
    const auto expanded_grad_weight = reshape_dim_into(1, 1, normalized_input) * grad_out;
    grad_weight = expanded_grad_weight.transpose(0, 1).sum(range(1, expanded_grad_weight.dim()));
  }

  // 如果输出掩码中包含第1个位置，则计算grad_input
  if (output_mask[0]) {
    // 计算归一化输入的梯度
    const auto grad_normalized_input = weight.defined() ?
      grad_out * padRight(weight, nullopt, grad_out.dim() - 1) : grad_out;
    auto [grad_normalized_input_value, grad_normalized_input_bdim] =
        unwrapTensorAtLevel(grad_normalized_input, cur_level);

    // 排除FuncTorchBatched调度键，确保不进行批处理操作
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    // 调用无权重和偏置的Group Norm反向传播批处理规则
    const auto res = group_norm_backward_no_weight_bias_batch_rule(
        grad_normalized_input_value, grad_normalized_input_bdim,
        input_value, input_bdim,
        mean_value, mean_bdim,
        rstd_value, rstd_bdim,
        N, C, HxW, group
    );
    // 在当前层级上创建批处理张量
    grad_input = makeBatched(std::get<0>(res), std::get<1>(res), cur_level);
  }
  // 返回结果元组：grad_input, grad_weight, grad_bias
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}
    // 检查张量是否未定义，如果是则返回 true
    if (!tensor.defined()) {
        return true;
    }
    // 检查张量的维度是否与规范化的形状数组的大小相匹配，如果不匹配则返回 false
    if (rankWithoutBatchDim(tensor, tensor_bdim) != (int64_t) normalized_shape.size()) {
        return false;
    }
    // 获取张量的形状
    const auto tensor_shape = tensor.sizes();
    // 遍历规范化的形状数组
    for (const auto i : c10::irange(normalized_shape.size())) {
        auto j = i;
        // 如果存在张量批次维度，并且当前索引 i 大于等于批次维度的值，将 j 值增加 1
        if (tensor_bdim.has_value() && (int64_t)i >= tensor_bdim.value()) {
            j = j + 1;
        }
        // 检查规范化的形状数组中的元素是否与张量的对应维度大小相等，如果不相等则返回 false
        if (normalized_shape[i] != tensor_shape[j]) {
            return false;
        }
    }
    // 如果所有检查通过，则返回 true
    return true;
}

// 定义一个内联函数，用于检查张量形状是否与指定的标准化形状相同
C10_ALWAYS_INLINE void check_same_shape(
    const Tensor& tensor, optional<int64_t> tensor_bdim,
    c10::SymIntArrayRef normalized_shape, const std::string& name) {
  // 使用 TORCH_CHECK 断言确保张量 tensor 与标准化形状 normalized_shape 具有相同的形状
  TORCH_CHECK(has_same_shape(tensor, tensor_bdim, normalized_shape),
      "Expected ", name, " to be of same shape as normalized_shape, but got ",
      name, " of shape ",
      tensor.sizes(),
      " and normalized_shape = ",
      normalized_shape);
}

// 辅助函数，用于检查 LayerNorm 操作的输入参数是否合法
C10_ALWAYS_INLINE void _check_layer_norm_inputs(
    SymIntArrayRef normalized_shape,
    const Tensor& weight, optional<int64_t> weight_bdim,
    const Tensor& bias, optional<int64_t> bias_bdim) {

  // 获取标准化形状的维度数
  const auto normalized_ndim = normalized_shape.size();
  // 使用 TORCH_CHECK 断言确保标准化形状至少为一维，即至少包含一个元素
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  // 检查权重和偏置的形状是否与标准化形状一致
  check_same_shape(weight, weight_bdim, normalized_shape, "weight");
  check_same_shape(bias, bias_bdim, normalized_shape, "weight");
}

// 定义静态函数，实现原生 LayerNorm 操作的批处理规则
static std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>,Tensor,optional<int64_t>>
native_layer_norm_batch_rule(
    const Tensor& input, optional<int64_t> input_bdim,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt, optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, optional<int64_t> bias_bdim,
    double eps) {
  // 将批处理维度移动到张量前部
  auto input_ = moveBatchDimToFront(input, input_bdim);
  // 如果没有权重和偏置的批处理维度
  if (!weight_bdim && !bias_bdim) {
    // 执行标准化 LayerNorm 操作，并获取结果的均值和逆标准差
    const auto result = at::native_layer_norm_symint(input_, normalized_shape, weight_opt, bias_opt, eps);
    const auto mean = std::get<1>(result);
    const auto rstd = std::get<2>(result);
    // 计算统计信息的批处理维度
    const auto stats_bdim = compute_stat_bdim(input_bdim, mean);
    // 返回操作结果的元组
    return std::make_tuple(std::get<0>(result), 0, mean, stats_bdim, rstd, stats_bdim);
  }

  // 从可选的权重张量获取权重，并检查输入参数的合法性
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  // 从可选的偏置张量获取偏置，并检查输入参数的合法性
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  // 检查 LayerNorm 操作的输入参数是否合法
  _check_layer_norm_inputs(normalized_shape, weight, weight_bdim, bias, bias_bdim);

  // 获取输入张量的逻辑秩（去除批处理维度后的秩）
  const auto input_logical_rank = rankWithoutBatchDim(input, input_bdim);
  // 执行标准化 LayerNorm 操作，并获取结果的元组
  const auto result = at::native_layer_norm_symint(input_, normalized_shape, nullopt, nullopt, eps);
  auto result0 = std::get<0>(result);
  const auto mean = std::get<1>(result);
  const auto rstd = std::get<2>(result);
  // 计算统计信息的批处理维度
  const auto stats_bdim = compute_stat_bdim(input_bdim, mean);

  // 如果定义了权重张量，则将其移动到前部并根据逻辑秩进行可能的填充
  if (weight.defined()) {
    auto weight_ = moveBatchDimToFront(weight, weight_bdim);
    weight_ = maybePadToLogicalRank(weight_, /*has_bdim*/weight_bdim, input_logical_rank);
    result0 = result0 * weight_;
  }
  // 如果定义了偏置张量，则将其应用到操作结果中
  if (bias.defined()) {

    // 将偏置张量移动到前部
    auto bias_ = moveBatchDimToFront(bias, bias_bdim);
    // 将偏置张量应用到操作结果中
    result0 = result0 + bias_;
  }

  // 返回操作结果的元组
  return std::make_tuple(result0, 0, mean, stats_bdim, rstd, stats_bdim);
}
    // 调用rankWithoutBatchDim函数计算结果的逻辑排名，传入result0作为参数
    const auto result_logical_rank = rankWithoutBatchDim(
        result0,
        // 根据条件设置是否传入optional值，如果input_bdim或weight_bdim有值则传入0，否则传入nullopt
        input_bdim.has_value() || weight_bdim.has_value() ? optional<int64_t>(0) : optional<int64_t>(nullopt));
    
    // 将bias向量的批次维度移动到前面，返回移动后的新bias_
    auto bias_ = moveBatchDimToFront(bias, bias_bdim);
    
    // 可能根据逻辑排名对bias_进行填充，如果bias_有批次维度(bias_bdim)，则填充到result_logical_rank的长度
    bias_ = maybePadToLogicalRank(bias_, /*has_bdim*/bias_bdim, result_logical_rank);
    
    // 将result0与bias_相加，更新result0的值
    result0 = result0 + bias_;
  }
  
  // 返回一个包含多个元素的tuple，分别是result0, 0, mean, stats_bdim, rstd, stats_bdim
  return std::make_tuple(result0, 0, mean, stats_bdim, rstd, stats_bdim);
// 定义静态函数，实现无权重和偏置的 Layer Norm 反向传播批处理规则
static std::tuple<at::Tensor, optional<int64_t>> native_layer_norm_backward_no_weight_bias_batch_rule(
    const at::Tensor & grad_out,  // 输入：梯度输出张量
    optional<int64_t> grad_out_bdim,  // 输入：梯度输出的批处理维度（可选）
    const at::Tensor & input,  // 输入：输入张量
    optional<int64_t> input_bdim,  // 输入：输入的批处理维度（可选）
    at::IntArrayRef normalized_shape,  // 输入：归一化形状
    const at::Tensor & mean,  // 输入：均值张量
    optional<int64_t> mean_bdim,  // 输入：均值的批处理维度（可选）
    const at::Tensor & rstd,  // 输入：反标准差张量
    optional<int64_t> rstd_bdim) {  // 输入：反标准差的批处理维度（可选）

  // 检查是否所有的批处理维度均未指定
  if (!grad_out_bdim.has_value() && !input_bdim.has_value() &&
      !mean_bdim.has_value() && !rstd_bdim.has_value()) {
    // 如果是，则调用 native_layer_norm_backward 函数进行反向传播
    const auto result = at::native_layer_norm_backward(
        grad_out, input, normalized_shape, mean, rstd, nullopt, nullopt, {true, false, false});
    return std::make_tuple(std::get<0>(result), nullopt);
  }

  // 将具有指定批处理维度的张量移到批处理维度的最前面
  auto grad_out_ = moveBatchDimToFront(grad_out, grad_out_bdim);
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto mean_ = moveBatchDimToFront(mean, mean_bdim);
  auto rstd_ = moveBatchDimToFront(rstd, rstd_bdim);

  // 确保 grad_out 和 input 张量具有批处理维度
  const auto bdim_size = get_bdim_size2(grad_out, grad_out_bdim, input, input_bdim);
  grad_out_ = ensure_has_bdim(grad_out_, grad_out_bdim.has_value(), bdim_size);
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size);
  mean_ = ensure_has_bdim(mean_, mean_bdim.has_value(), bdim_size);
  rstd_ = ensure_has_bdim(rstd_, rstd_bdim.has_value(), bdim_size);

  // 调用 native_layer_norm_backward 函数进行反向传播
  auto result = at::native_layer_norm_backward(
      grad_out_.contiguous(),
      input_.contiguous(),
      normalized_shape,
      mean_.contiguous(),
      rstd_.contiguous(),
      nullopt, nullopt, {true, false, false});

  // 返回结果，包括梯度输出和一个空的可选值作为第二个元素
  return std::make_tuple(std::get<0>(result), 0);
}

// 定义静态函数，实现 Layer Norm 反向传播的管道处理
static std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward_plumbing(
    const at::Tensor & grad_out,  // 输入：梯度输出张量
    const at::Tensor & input,  // 输入：输入张量
    at::IntArrayRef normalized_shape,  // 输入：归一化形状
    const at::Tensor & mean,  // 输入：均值张量
    const at::Tensor & rstd,  // 输入：反标准差张量
    const std::optional<at::Tensor> & weight_opt,  // 输入：权重张量（可选）
    const std::optional<at::Tensor> & bias_opt,  // 输入：偏置张量（可选）
    std::array<bool, 3> output_mask) {  // 输入：输出掩码数组

  // 移动权重和偏置的可能所有权到 Tensor 类型的常量引用
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 获取当前动态层的可能图层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查是否逃逸的 vmap
  vmap_check_escaped(maybe_layer, "native_layer_norm_backward_plumbing");
  // 获取当前层 ID
  int64_t cur_level = maybe_layer->layerId();
  // 检查是否在当前层有批处理的张量
  if (!areAnyBatchedAtLevel({grad_out, input, mean, rstd, weight_opt, bias_opt}, cur_level)) {
    // 在 FuncTorchBatched 的调度键下排除调度的键保护
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  return at::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd,
        weight_opt, bias_opt, output_mask);

# 调用PyTorch的本地Layer Norm反向传播函数，计算梯度

  auto [grad_out_value, grad_out_bdim] = unwrapTensorAtLevel(grad_out, cur_level);

  // 解包梯度张量 `grad_out` 到值和维度信息，以当前级别为基准

  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);

  // 解包输入张量 `input` 到值和维度信息，以当前级别为基准

  auto [mean_value, mean_bdim] = unwrapTensorAtLevel(mean, cur_level);

  // 解包均值张量 `mean` 到值和维度信息，以当前级别为基准

  auto [rstd_value, rstd_bdim] = unwrapTensorAtLevel(rstd, cur_level);

  // 解包标准差张量 `rstd` 到值和维度信息，以当前级别为基准

  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight.defined()) {
    std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  }

  // 解包权重张量 `weight` 到值和维度信息，以当前级别为基准，如果定义了的话

  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias.defined()) {
    std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias, cur_level);
  }

  // 解包偏置张量 `bias` 到值和维度信息，以当前级别为基准，如果定义了的话

  // results
  Tensor grad_bias;
  Tensor grad_weight;
  Tensor grad_input;

  // 定义梯度偏置 `grad_bias`，梯度权重 `grad_weight`，和梯度输入 `grad_input` 张量

  if (output_mask[2] && bias_value.has_value()) {
    const auto num_front_dims_to_reduce = grad_out.dim() - normalized_shape.size();
    if (num_front_dims_to_reduce == 0) {
      grad_bias = grad_out;
    } else {
      grad_bias = grad_out.sum(range(0, static_cast<int64_t>(num_front_dims_to_reduce)));
    }
  }

  // 如果 `output_mask` 的第三位为真，并且偏置张量 `bias_value` 有值，则计算梯度偏置 `grad_bias`

  if (output_mask[1] && weight_value.has_value()) {
    // NB: output isn't saved...
    const auto normalized_input = (input - mean) * rstd;
    const auto expanded_grad_weight = normalized_input * grad_out;
    const auto num_front_dims_to_reduce =
        expanded_grad_weight.dim() - normalized_shape.size();
    if (num_front_dims_to_reduce == 0) {
      grad_weight = expanded_grad_weight;
    } else {
      grad_weight = expanded_grad_weight.sum(range(0, static_cast<int64_t>(num_front_dims_to_reduce)));
    }
  }

  // 如果 `output_mask` 的第二位为真，并且权重张量 `weight_value` 有值，则计算梯度权重 `grad_weight`

  if (output_mask[0]) {
    const auto grad_normalized_input = weight.defined() ?
      grad_out * weight : grad_out;
    auto [grad_normalized_input_value, grad_normalized_input_bdim] =
        unwrapTensorAtLevel(grad_normalized_input, cur_level);

  // 如果 `output_mask` 的第一位为真，则计算归一化梯度输入 `grad_normalized_input`

    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    const auto results = native_layer_norm_backward_no_weight_bias_batch_rule(
        grad_normalized_input_value, grad_normalized_input_bdim,
        input_value, input_bdim,
        normalized_shape,
        mean_value, mean_bdim,
        rstd_value, rstd_bdim);
    grad_input = makeBatched(std::get<0>(results), std::get<1>(results), cur_level);

  // 调用不带权重和偏置的本地Layer Norm批处理规则的反向传播，计算梯度输入 `grad_input`

  return std::make_tuple(grad_input, grad_weight, grad_bias);

  // 返回梯度输入 `grad_input`，梯度权重 `grad_weight`，和梯度偏置 `grad_bias` 的元组
}

template <typename F, F Func>
struct NativeBatchNormBatchRuleHelper {
  // 定义结构体，用于帮助应用本地批归一化批处理规则
  static std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>,Tensor,optional<int64_t>> apply(
    // 应用函数，接受输入张量、输入批维度、权重张量（可选）、权重批维度、偏置张量（可选）、偏置批维度、
    // 运行均值张量（可选）、运行均值批维度、运行方差张量（可选）、运行方差批维度、训练标志、动量、eps
    const Tensor& input, optional<int64_t> input_bdim,
    const std::optional<Tensor>& weight_opt, optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, optional<int64_t> bias_bdim,
    const std::optional<Tensor>& running_mean_opt, optional<int64_t> running_mean_bdim,
    const std::optional<Tensor>& running_var_opt, optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps) {
    // 调用 batch_norm_batch_rule 函数，应用本地批归一化批处理规则，返回结果元组
    return batch_norm_batch_rule<F, Func>(
        input, input_bdim, weight_opt, weight_bdim, bias_opt, bias_bdim,
        running_mean_opt, running_mean_bdim, running_var_opt, running_var_bdim, training, momentum, eps);
  }
};

template <typename F, F Func>
struct CudnnBatchNormBatchRuleHelper {
  // 定义结构体，用于帮助应用 Cudnn 批归一化批处理规则
  static std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>,Tensor,optional<int64_t>,Tensor,optional<int64_t>> apply(
    // 应用函数，接受输入张量、输入批维度、权重张量、权重批维度（可选）、偏置张量（可选）、偏置批维度、
    // 运行均值张量（可选）、运行均值批维度、运行方差张量（可选）、运行方差批维度、训练标志、动量、eps
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& weight_opt, optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, optional<int64_t> bias_bdim,
    const std::optional<Tensor>& running_mean_opt, optional<int64_t> running_mean_bdim,
    const std::optional<Tensor>& running_var_opt, optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps) {
    // 使用 input 的选项创建一个空张量，类型为 kByte，命名为 reserve，通常在实验中由 cuda 设置为空
    auto reserve = at::empty({0}, input.options().dtype(kByte));
    // 调用 batch_norm_batch_rule 函数，应用 Cudnn 批归一化批处理规则，返回结果元组
    auto res = batch_norm_batch_rule<F, Func>(
        input, input_bdim, weight_opt, weight_bdim, bias_opt, bias_bdim,
        running_mean_opt, running_mean_bdim, running_var_opt, running_var_bdim, training, momentum, eps);
    // 将 reserve 和空的 nullopt 添加到结果元组的末尾，并返回
    return std::tuple_cat(res, std::make_tuple(reserve, nullopt));
  }
};

template <typename F, F Func>
struct MiopenBatchNormBatchRuleHelper {
  // 定义结构体，用于帮助应用 Miopen 批归一化批处理规则
  static std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>,Tensor,optional<int64_t>> apply(
    // 应用函数，接受输入张量、输入批维度、权重张量、权重批维度（可选）、偏置张量（可选）、偏置批维度、
    // 运行均值张量（可选）、运行均值批维度、运行方差张量（可选）、运行方差批维度、训练标志、动量、eps
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& weight_opt, optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, optional<int64_t> bias_bdim,
    const std::optional<Tensor>& running_mean_opt, optional<int64_t> running_mean_bdim,
    const std::optional<Tensor>& running_var_opt, optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps) {
    // 调用 batch_norm_batch_rule 函数，应用 Miopen 批归一化批处理规则，返回结果元组
    return batch_norm_batch_rule<F, Func>(
        input, input_bdim, weight_opt, weight_bdim, bias_opt, bias_bdim,
        running_mean_opt, running_mean_bdim, running_var_opt, running_var_bdim, training, momentum, eps);
  }
};

// 定义宏，展开为调用 NativeBatchNormBatchRuleHelper 的 apply 函数
#define NATIVE_BATCH_NORM_BATCH_RULE(fn) SINGLE_ARG(\
    NativeBatchNormBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)
#define CUDNN_BATCH_NORM_BATCH_RULE(fn) SINGLE_ARG(\
   CudnnBatchNormBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

#define MIOPEN_BATCH_NORM_BATCH_RULE(fn) SINGLE_ARG(\
    MiopenBatchNormBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

template <typename F, F Func>
struct NativeBatchNormBackwardBatchRuleHelper {
  static std::tuple<Tensor,Tensor,Tensor> apply(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const std::optional<at::Tensor> & weight_opt,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    bool training,
    double eps,
    std::array<bool,3> output_mask) {

    auto maybe_layer = maybeCurrentDynamicLayer();
    vmap_check_escaped(maybe_layer, "NativeBatchNormBackwardBatchRuleHelper.apply");
    int64_t cur_level = maybe_layer->layerId();

    // 检查是否有任何张量在当前层级上进行了批处理
    if (!areAnyBatchedAtLevel({grad_out, input, weight_opt, running_mean_opt,
          running_var_opt, save_mean_opt, save_rstd_opt}, cur_level)) {
      // 临时排除 DispatchKey::FuncTorchBatched 分发键
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      // 调用原生的批处理标准化反向传播函数
      return at::native_batch_norm_backward(grad_out, input, weight_opt,
          running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt,
          training, eps, output_mask);
    }

    // 使用通用的批处理反向传播管道函数
    return batch_norm_backward_plumbing<F, Func>(
        grad_out, input, weight_opt, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, training, eps, output_mask);
  }
};

template <typename F, F Func>
struct CudnnBatchNormBackwardBatchRuleHelper {
  static std::tuple<Tensor,Tensor,Tensor> apply(
    const at::Tensor & input,
    const at::Tensor & grad_out,
    const at::Tensor & weight,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    double eps,
    const at::Tensor & reserve) {

    auto maybe_layer = maybeCurrentDynamicLayer();
    vmap_check_escaped(maybe_layer, "CudnnBatchNormBackwardBatchRuleHelper.apply");
    int64_t cur_level = maybe_layer->layerId();

    // 检查是否有任何张量在当前层级上进行了批处理
    if (!areAnyBatchedAtLevel({input, grad_out, weight, running_mean_opt,
          running_var_opt, save_mean_opt, save_rstd_opt, reserve}, cur_level)) {
      // 临时排除 DispatchKey::FuncTorchBatched 分发键
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      // 调用 CuDNN 的批处理标准化反向传播函数
      return at::cudnn_batch_norm_backward(input, grad_out, weight,
          running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, eps, reserve);
    }

    // 使用通用的批处理反向传播管道函数
    return batch_norm_backward_plumbing<F, Func>(
        grad_out, input, weight, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, true, eps, {true, true, true});
  }
};


注释：

// 定义 CUDNN 批处理规则宏，调用 CudnnBatchNormBatchRuleHelper 的 apply 方法
#define CUDNN_BATCH_NORM_BATCH_RULE(fn) SINGLE_ARG(\
   CudnnBatchNormBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

// 定义 MIOPEN 批处理规则宏，调用 MiopenBatchNormBatchRuleHelper 的 apply 方法
#define MIOPEN_BATCH_NORM_BATCH_RULE(fn) SINGLE_ARG(\
    MiopenBatchNormBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

// NativeBatchNormBackwardBatchRuleHelper 结构体模板的定义
template <typename F, F Func>
struct NativeBatchNormBackwardBatchRuleHelper {
  // apply 方法，用于原生批处理标准化反向传播
  static std::tuple<Tensor,Tensor,Tensor> apply(
    const at::Tensor & grad_out,  // 梯度输出张量
    const at::Tensor & input,  // 输入张量
    const std::optional<at::Tensor> & weight_opt,  // 权重的可选张量
    const std::optional<at::Tensor> & running_mean_opt,  // 运行时均值的可选张量
    const std::optional<at::Tensor> & running_var_opt,  // 运行时方差的可选张量
    const std::optional<at::Tensor> & save_mean_opt,  // 保存均值的可选张量
    const std::optional<at::Tensor> & save_rstd_opt,  // 保存反标准化系数的可选张量
    bool training,  // 是否训练模式
    double eps,  // epsilon 参数
    std::array<bool,3> output_mask) {  // 输出掩码数组

    auto maybe_layer = maybeCurrentDynamicLayer();  // 获取当前动态层
    vmap_check_escaped(maybe_layer, "NativeBatchNormBackwardBatchRuleHelper.apply");  // 检查动态层是否存在

    int64_t cur_level = maybe_layer->layerId();  // 获取当前层级 ID

    // 如果没有任何张量在当前层级上进行了批处理
    if (!areAnyBatchedAtLevel({grad_out, input, weight_opt, running_mean_opt,
          running_var_opt, save_mean_opt, save_rstd_opt}, cur_level)) {
      // 临时排除 DispatchKey::FuncTorchBatched 分发键
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      // 调用原生的批处理标准化反向传播函数
      return at::native_batch_norm_backward(grad_out, input, weight_opt,
          running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt,
          training, eps, output_mask);
    }

    // 使用通用的批处理反向传播管道函数
    return batch_norm_backward_plumbing<F, Func>(
        grad_out, input, weight_opt, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, training, eps, output_mask);
  }
};

// CudnnBatchNormBackwardBatchRuleHelper 结构体模板的定义
template <typename F, F Func>
struct CudnnBatchNormBackwardBatchRuleHelper {
  // apply 方法，用于 CuDNN 批处理标准化反向传播
  static std::tuple<Tensor,Tensor,Tensor> apply(
    const at::Tensor & input,  // 输入张量
    const at::Tensor & grad_out,  // 梯度输出张量
    const at::Tensor & weight,  // 权重张量
    const std::optional<at::Tensor> & running_mean_opt,  // 运行时均值的可选张量
    const std::optional<at::Tensor> & running_var_opt,  // 运行时方差的可选张量
    const std::optional<at::Tensor> & save_mean_opt,  // 保存均值的可选张量
    const std::optional<at::Tensor> & save_rstd_opt,  // 保存反标准化系数的可选张量
    double eps,  // epsilon 参数
    const at::Tensor & reserve) {  // 保留张量

    auto maybe_layer = maybeCurrentDynamicLayer();  // 获取当前动态层
    vmap_check_escaped(maybe_layer, "CudnnBatchNormBackwardBatchRuleHelper.apply");  // 检查动态层是否存在

    int64_t cur_level = maybe_layer->layerId();  // 获取当前层级 ID

    // 如果没有任何张量在当前层级上进行了批
// 定义 MiopenBatchNormBackwardBatchRuleHelper 结构体，用于处理 Miopen 批量归一化的反向传播规则
struct MiopenBatchNormBackwardBatchRuleHelper {
  // 定义静态函数 apply，接收多个参数并返回三个张量的元组
  static std::tuple<Tensor,Tensor,Tensor> apply(
    const at::Tensor & input, // 输入张量
    const at::Tensor & grad_out, // 梯度输出张量
    const at::Tensor & weight, // 权重张量
    const std::optional<at::Tensor> & running_mean_opt, // 可选的运行均值张量
    const std::optional<at::Tensor> & running_var_opt, // 可选的运行方差张量
    const std::optional<at::Tensor> & save_mean_opt, // 可选的保存均值张量
    const std::optional<at::Tensor> & save_rstd_opt, // 可选的保存反标准差张量
    double eps) { // epsilon 参数

    auto maybe_layer = maybeCurrentDynamicLayer(); // 获取当前可能的动态层
    vmap_check_escaped(maybe_layer, "MiopenBatchNormBackwardBatchRuleHelper.apply"); // 检查动态映射是否逃逸

    int64_t cur_level = maybe_layer->layerId(); // 获取当前层级 ID

    // 如果没有任何张量在当前层级进行批处理
    if (!areAnyBatchedAtLevel({input, grad_out, weight, running_mean_opt,
          running_var_opt, save_mean_opt, save_rstd_opt}, cur_level)) {
      // 临时排除 DispatchKey::FuncTorchBatched 调度键
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      // 调用 miopen 批量归一化的反向传播函数，并返回结果
      return at::miopen_batch_norm_backward(input, grad_out, weight,
          running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, eps);
    }

    // 调用批处理归一化反向传播的管道函数，并返回结果
    return batch_norm_backward_plumbing<F, Func>(
        grad_out, input, weight, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, true, eps, {true, true, true});
  }
};

// 定义宏，用于生成原生批量归一化的反向传播规则
#define NATIVE_BATCH_NORM_BACKWARD_BATCH_RULE(fn) SINGLE_ARG(\
    NativeBatchNormBackwardBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

// 定义宏，用于生成 CUDNN 批量归一化的反向传播规则
#define CUDNN_BATCH_NORM_BACKWARD_BATCH_RULE(fn) SINGLE_ARG(\
   CudnnBatchNormBackwardBatchRuleHelper<\
      decltype(&fn),\
      &fn>::apply)

// 定义宏，用于生成 MIOPEN 批量归一化的反向传播规则
#define MIOPEN_BATCH_NORM_BACKWARD_BATCH_RULE(fn) SINGLE_ARG(\
    MiopenBatchNormBackwardBatchRuleHelper<\
      decltype(&fn),\
      &fn>::apply)

// 定义 CUDNN 批量归一化反向传播的包装函数
static std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm_backward_wrapper(
    const at::Tensor & grad_out, // 梯度输出张量
    const at::Tensor & input, // 输入张量
    const at::Tensor& weight_opt, // 权重张量（可选）
    const std::optional<at::Tensor> & running_mean_opt, // 可选的运行均值张量
    const std::optional<at::Tensor> & running_var_opt, // 可选的运行方差张量
    const std::optional<at::Tensor> & save_mean_opt, // 可选的保存均值张量
    const std::optional<at::Tensor> & save_rstd_opt, // 可选的保存反标准差张量
    bool training, // 是否训练标志
    double eps, // epsilon 参数
    std::array<bool,3> output_mask) { // 输出掩码数组
    TORCH_INTERNAL_ASSERT(!training); // 由 batch_norm_impl 确保不在训练状态

    // 创建空的 reserve 张量，使用输入张量的选项和字节类型
    auto reserve = at::empty({0}, input.options().dtype(kByte));
    // 调用 cudnn 批量归一化反向传播函数，并返回结果
    return at::cudnn_batch_norm_backward(input, grad_out, weight_opt, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, eps, reserve);
  }

// 定义 MIOPEN 批量归一化反向传播的包装函数
static std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm_backward_wrapper(
    const at::Tensor & grad_out, // 梯度输出张量
    const at::Tensor & input, // 输入张量
    const at::Tensor& weight_opt, // 权重张量（可选）
    const std::optional<at::Tensor> & running_mean_opt, // 可选的运行均值张量
    const std::optional<at::Tensor> & running_var_opt, // 可选的运行方差张量
    const std::optional<at::Tensor> & save_mean_opt, // 可选的保存均值张量
    const std::optional<at::Tensor> & save_rstd_opt, // 可选的保存反标准差张量
    bool training, // 是否训练标志
    double eps, // epsilon 参数
    std::array<bool,3> output_mask) { // 输出掩码数组
    TORCH_INTERNAL_ASSERT(!training); // 由 batch_norm_impl 确保不在训练状态
    // 调用 ATen（PyTorch C++ 前端）的 miopen_batch_norm_backward 函数
    // 该函数用于计算 MiOpen（MIOpen 是 AMD 开发的深度学习库）批归一化层的反向传播
    // 参数包括输入张量 input、梯度张量 grad_out、权重优化选项 weight_opt、
    // 运行时均值 running_mean_opt、运行时方差 running_var_opt、
    // 保存的均值 save_mean_opt、保存的反标准差 save_rstd_opt 和 epsilon eps
    return at::miopen_batch_norm_backward(input, grad_out, weight_opt, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, eps);
}
// 定义一个静态函数 _native_batch_norm_legit_batch，执行带有运行统计数据的批量原生批标准化操作
static std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit_batch(
  const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
  Tensor& running_mean, Tensor& running_var, bool train, double momentum, double eps) {
    // 调用原生批标准化函数 native_batch_norm，返回批量的标准化结果
    return at::native_batch_norm(self, weight_opt, bias_opt, running_mean, running_var, train, momentum, eps);
}

// 定义一个静态函数 _native_batch_norm_legit_no_stats_batch，执行不带运行统计数据的批量原生批标准化操作
static std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit_no_stats_batch(
  const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
  bool train, double momentum, double eps) {
    // 调用原生批标准化函数 native_batch_norm，返回批量的标准化结果，但不使用运行统计数据
    return at::native_batch_norm(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, eps);
}

// 实现 TORCH 库的批处理规则，支持原生批标准化及其后向传播
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 为原生批标准化函数添加 vmap 支持，并使用 NATIVE_BATCH_NORM_BATCH_RULE 宏规则
  VMAP_SUPPORT(native_batch_norm, NATIVE_BATCH_NORM_BATCH_RULE(native_batch_norm));
  // 为 cuDNN 的批标准化函数添加 vmap 支持，并使用 CUDNN_BATCH_NORM_BATCH_RULE 宏规则
  VMAP_SUPPORT(cudnn_batch_norm, CUDNN_BATCH_NORM_BATCH_RULE(cudnn_batch_norm));
  // 为 MIOpen 的批标准化函数添加 vmap 支持，并使用 MIOPEN_BATCH_NORM_BATCH_RULE 宏规则
  VMAP_SUPPORT(miopen_batch_norm, MIOPEN_BATCH_NORM_BATCH_RULE(miopen_batch_norm));
  // 将 _native_batch_norm_legit 函数实现注册到 TORCH 库，命名为 "_native_batch_norm_legit"
  m.impl("_native_batch_norm_legit", _native_batch_norm_legit_batch);
  // 将 _native_batch_norm_legit_no_stats_batch 函数实现注册到 TORCH 库，命名为 "_native_batch_norm_legit.no_stats"
  m.impl("_native_batch_norm_legit.no_stats", _native_batch_norm_legit_no_stats_batch);
  // 将 native_batch_norm_backward 函数的批处理规则实现注册到 TORCH 库
  m.impl("native_batch_norm_backward", NATIVE_BATCH_NORM_BACKWARD_BATCH_RULE(native_batch_norm_backward));
  // 将 cudnn_batch_norm_backward 函数的批处理规则实现注册到 TORCH 库
  m.impl("cudnn_batch_norm_backward", CUDNN_BATCH_NORM_BACKWARD_BATCH_RULE(at::functorch::cudnn_batch_norm_backward_wrapper));
  // 将 miopen_batch_norm_backward 函数的批处理规则实现注册到 TORCH 库
  m.impl("miopen_batch_norm_backward", MIOPEN_BATCH_NORM_BACKWARD_BATCH_RULE(at::functorch::miopen_batch_norm_backward_wrapper));
  // 将 native_group_norm 函数实现注册到 TORCH 库
  m.impl("native_group_norm", native_group_norm_plumbing);
  // 将 native_group_norm_backward 函数实现注册到 TORCH 库
  m.impl("native_group_norm_backward", native_group_norm_backward_plumbing);
  // 为原生层标准化函数添加 vmap 支持，并使用 native_layer_norm_batch_rule 宏规则
  VMAP_SUPPORT(native_layer_norm, native_layer_norm_batch_rule);
  // 将 native_layer_norm_backward 函数实现注册到 TORCH 库
  m.impl("native_layer_norm_backward", native_layer_norm_backward_plumbing);
}

} // namespace at::functorch
```