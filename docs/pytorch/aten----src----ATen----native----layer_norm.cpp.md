# `.\pytorch\aten\src\ATen\native\layer_norm.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/layer_norm.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/layer_norm_native.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#include <array>
#include <tuple>
#include <vector>

namespace at::native {

// 定义静态函数，实现基于给定参数的 layer normalization，并输出结果到 out、mean 和 rstd
static void layer_norm_with_mean_rstd_out(
    at::Tensor& out,
    at::Tensor& mean,
    at::Tensor& rstd,
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N) {
  // 调用 LayerNormKernel 函数，在 CPU 上执行 layer normalization 的核心计算
  LayerNormKernel(kCPU, input, gamma, beta, M, N, eps, &out, &mean, &rstd);
  // 获取输入张量的形状
  const auto input_shape = input.sizes();
  // 计算规范化的轴
  const size_t axis = input.dim() - normalized_shape.size();

  // 创建一个向量用于存储统计信息的形状
  DimVector stat_shape;
  // 对于轴之前的维度，保持与输入相同的形状
  for (const auto idx : c10::irange(axis)) {
    stat_shape.emplace_back(input_shape[idx]);
  }
  // 对于轴之后的维度，设置为1，用于 broadcasting
  for (const auto idx C10_UNUSED : c10::irange(axis, input.dim())) {
    stat_shape.emplace_back(1);
  }

  // 将 mean 和 rstd 张量重塑为统计信息的形状
  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
}

// 定义函数，在 CPU 上执行 layer normalization，并输出结果到 out
void layer_norm_cpu_out(
    at::Tensor& out,
    const at::Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N) {
  // 若 M 小于等于 0，直接返回
  if (M <= 0) {
    return;
  }
  // 调用 LayerNormKernel 函数，在 CPU 上执行 layer normalization 的核心计算，mean 和 rstd 参数为 nullptr
  LayerNormKernel(kCPU, input, gamma, beta, M, N, eps, &out, /*mean=*/nullptr, /*rstd=*/nullptr);
}

// 定义函数，执行 CPU 上的 layer normalization，并返回输出张量 out、mean 和 rstd 的元组
std::tuple<Tensor, Tensor, Tensor> layer_norm_cpu(
    const Tensor& input,
    IntArrayRef normalized_shape, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  // 使用 borrow_from_optional_tensor 函数处理可选的权重和偏置张量
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 检查输入、权重和偏置是否混合数据类型
  bool mixed_type = is_mixed_type(input, weight, bias);
  // 若混合类型为 true，则执行特定的操作
  if (mixed_type) {
    // 调用一个检查函数，用于处理可能存在的混合数据类型的输入、权重和偏置
    check_mixed_data_type(input, weight, bias);
  }

  // 检查和标准化 Layer Normalization 的输入参数，返回包含 M 和 N 的元组
  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;  // 提取元组中的 M 值
  auto N = M_N.second;  // 提取元组中的 N 值
  auto X = input.expect_contiguous();  // 获取连续存储的输入 Tensor
  auto gamma = weight.expect_contiguous();  // 获取连续存储的权重 Tensor
  auto beta = bias.expect_contiguous();  // 获取连续存储的偏置 Tensor

  // 创建一个形状与 X 相同的空 Tensor Y，使用相同的内存布局和设备类型
  Tensor Y = at::native::empty_like(
      *X,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  const auto dtype = param_scalar_type(input, mixed_type);  // 根据输入和混合类型获取数据类型
  // 创建一个形状为 {M} 的空 Tensor mean，使用与 X 相同的数据类型
  Tensor mean = at::empty({M}, X->options().dtype(dtype));
  // 创建一个形状为 {M} 的空 Tensor rstd，使用与 X 相同的数据类型
  Tensor rstd = at::empty({M}, X->options().dtype(dtype));

  // 调用 Layer Normalization 函数，计算输出 Y、均值 mean 和反标准差 rstd
  layer_norm_with_mean_rstd_out(Y, mean, rstd, *X, normalized_shape, *gamma, *beta, eps, M, N);
  // 返回一个包含 Y、mean 和 rstd 的元组，移动这些 Tensor 的所有权
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(
    const Tensor& dY,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    std::array<bool, 3> grad_input_mask) {
  // 从可选的权重张量中获取权重，如果未提供则创建一个空张量
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  // 从可选的偏置张量中获取偏置，如果未提供则创建一个空张量
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 检查和标准化输入，权重和偏置的维度和形状
  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  // 如果需要计算输入梯度（grad_input_mask[0] 为真），则创建一个与输入张量 X 相同形状的空张量 dX
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        *X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  // 如果需要计算 gamma 的梯度（grad_input_mask[1] 为真），则根据 M 的值创建相应的空或零张量 dgamma
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous)
                   : at::native::zeros_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous);
  }
  // 如果需要计算 beta 的梯度（grad_input_mask[2] 为真），则根据 M 的值创建相应的空或零张量 dbeta
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous)
                  : at::native::zeros_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous);
  }
  // 如果 M 大于 0，则调用 LayerNormBackwardKernel 函数计算梯度
  if (M > 0) {
    LayerNormBackwardKernel(
        kCPU, dY, *X, mean, rstd, *gamma, M, N, &dX, &dgamma, &dbeta);
  }
  // 返回计算得到的梯度张量作为元组
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}
    // 定义函数 `native_layer_norm`，接受输入张量 `input`，标准化形状 `normalized_shape`，可选的权重张量 `weight_opt` 和偏置张量 `bias_opt`
    // `eps` 是标准化操作中的 epsilon 参数
    // `bool` 参数 `cudnn_enable` 已弃用，不再使用
    c10::SymIntArrayRef normalized_shape, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
      // 查看注释: [Note: hacky wrapper removal for optional tensor]
    
      // 从可选的权重张量 `weight_opt` 中获取可能拥有的张量，转为可持有对象
      c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
      // 引用权重张量 `weight`
      const Tensor& weight = *weight_maybe_owned;
    
      // 从可选的偏置张量 `bias_opt` 中获取可能拥有的张量，转为可持有对象
      c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
      // 引用偏置张量 `bias`
      const Tensor& bias = *bias_maybe_owned;
    
      // 调用 `at::native_layer_norm_symint` 函数，对输入张量 `input` 进行符号整数标准化操作，
      // 返回值包含在元组的第一个元素中
      return std::get<0>(at::native_layer_norm_symint(input, normalized_shape, weight, bias, eps));
    }
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);

// 从pytorch/xla库移植而来，实现了Layer Normalization算法
std::tuple<Tensor, Tensor, Tensor> math_native_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    double eps) {
  // [注意：用于处理可选张量的包装器]
  // 从可选的weight张量中获取MaybeOwned对象，如果不存在，则创建一个空Tensor
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  // 从可选的bias张量中获取MaybeOwned对象，如果不存在，则创建一个空Tensor
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // 检查并获取Layer Norm的输入参数
  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  // 确保输入张量是连续的
  auto X = input.expect_contiguous();
  // 确保weight张量是连续的
  auto gamma = weight.expect_contiguous();

  // 获取输入张量的形状和维度信息
  auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  // 计算Layer Norm的操作轴
  const int axis = input_ndim - normalized_ndim;

  // 处理输入张量为空的情况：返回空张量
  if (input.numel() == 0) {
    // 根据输入张量的数据类型创建相同形状的空张量
    auto result_type = c10::promoteTypes(input.scalar_type(), kFloat);
    return std::make_tuple(
      at::empty_like(input),
      at::empty_like(input, c10::TensorOptions().dtype(result_type)),
      at::empty_like(input, c10::TensorOptions().dtype(result_type))
    );
  }
  // 将输入张量重塑为形状为{1, M, -1}的张量
  at::Tensor input_reshaped = input.reshape({1, M, -1});
  // 使用native_batch_norm函数执行Batch Normalization操作
  auto outputs = at::native_batch_norm(
      input_reshaped, /*weight=*/{}, /*bias=*/{}, /*running_mean=*/{},
      /*running_var=*/{}, /*training=*/true, /*momentum=*/0, eps);
  // 获取Batch Normalization的输出张量
  at::Tensor out = std::get<0>(outputs);
  // 将输出张量重新视图为原始输入张量的形状
  out = out.view(input_shape);
  // 如果定义了weight和bias张量，则执行out = bias + out * weight的运算
  if (weight.defined() && bias.defined()) {
    out = bias.addcmul(out, weight, 1);
  } else if (weight.defined()) {
    // 如果只定义了weight张量，则执行out = out * weight的运算
    out = out.mul(weight);
  } else if (bias.defined()) {
    // 如果只定义了bias张量，则执行out = out + bias的运算
    out = out.add(bias);
  }
  // 获取Batch Normalization的mean和rstd张量
  at::Tensor mean = std::get<1>(outputs);
  at::Tensor rstd = std::get<2>(outputs);
  // 创建统计形状向量
  std::vector<int64_t> stat_shape;
  // 遍历轴范围内的索引，将其添加到统计形状向量中
  for (const auto idx : c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  // 对于超过轴范围的维度索引，将1添加到统计形状向量中
  for (const auto idx C10_UNUSED : c10::irange(axis, input.dim())) {
    stat_shape.push_back(1);
  }
  // 将mean张量视图重新形状为统计形状向量的形状
  mean = mean.view(stat_shape);
  // 将rstd张量视图重新形状为统计形状向量的形状
  rstd = rstd.view(stat_shape);
  // 返回Layer Normalization的输出张量、mean张量和rstd张量
  return std::make_tuple(out, mean, rstd);
}

Tensor rms_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  // 接受一个可选的双精度浮点数 eps 作为参数

  // 使用 at::borrow_from_optional_tensor 将可选的权重张量转换为不可变的常量引用
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // 创建一个空的可选张量 bias_opt，并将其转换为不可变的常量引用
  auto bias_opt = at::optional<Tensor>();
  const Tensor& bias = *at::borrow_from_optional_tensor(bias_opt);

  // 调用 _check_layer_norm_inputs 函数检查输入参数，并忽略返回值
  (void) _check_layer_norm_inputs(input, normalized_shape, weight, bias);

  // 创建一个整数向量 dims_to_reduce，用于保存要减少的维度
  std::vector<int64_t> dims_to_reduce;
  for (const auto i : c10::irange(normalized_shape.size())) {
    // 将每个维度的索引添加到 dims_to_reduce 中
    dims_to_reduce.push_back(input.dim() - i - 1);
  }
  // 创建 IntArrayRef 对象 dims_to_reduce_ref，用于引用 dims_to_reduce 的内容
  IntArrayRef dims_to_reduce_ref = IntArrayRef(dims_to_reduce);

  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2 宏来分发处理不同的数值类型
  auto result = AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "rms_norm",
        [&] {
    // 定义一个名为 scalar_t 的类型别名，表示当前数值类型
    scalar_t eps_val;
    // 如果 eps 没有值，则使用数值类型的最小值作为 eps_val
    if (!eps.has_value()) {
      eps_val = std::numeric_limits<at::scalar_value_type<scalar_t>::type>::epsilon();
    } else {
      // 否则，使用 eps 的值作为 eps_val
      eps_val = eps.value();
    }

    // 计算输入张量的 RMS 标准化结果
    auto result = input.mul(at::rsqrt(at::pow(input, 2).mean(dims_to_reduce_ref, /*keep_dim=*/true).add_(eps_val)));

    // 如果 weight_opt 有值，则将结果乘以权重张量
    if (weight_opt.has_value()) {
      result = result.mul(weight_opt.value());
    }

    // 返回计算得到的结果
    return result;
  });

  // 返回处理后的结果
  return result;
}
} // namespace at::native
```