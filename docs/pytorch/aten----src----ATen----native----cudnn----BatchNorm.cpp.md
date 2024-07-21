# `.\pytorch\aten\src\ATen\native\cudnn\BatchNorm.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>

#ifdef __HIP_PLATFORM_AMD__
#include <ATen/native/cudnn/hip/BatchNorm.h>
#else
#include <ATen/native/cudnn/BatchNorm.h>
#endif

#if !AT_CUDNN_ENABLED()

namespace at {
namespace native {

// 如果 ATen 没有启用 cuDNN 支持，则定义以下函数并报错
std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double exponential_average_factor,
    double epsilon) {
  AT_ERROR("cudnn_batch_norm: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt,
    const std::optional<Tensor>& save_var_opt,
    double epsilon,
    const Tensor& reservedSpace) {
  AT_ERROR("cudnn_batch_norm_backward: ATen not compiled with cuDNN support");
}

size_t _get_cudnn_batch_norm_reserve_space_size(
    const Tensor& input_t,
    bool training) {
  AT_ERROR(
      "_get_cudnn_batch_norm_reserve_space_size: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED

#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cudnn_batch_norm_backward_native.h>
#include <ATen/ops/cudnn_batch_norm_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at {
namespace native {

namespace {

// 定义一个辅助函数，用于扩展张量的尺寸
Tensor expandScale(const Tensor& t, int64_t dim) {
  // 创建一个向量表示新的张量尺寸
  std::vector<int64_t> size{1, t.numel()};
  // 将尺寸扩展到指定维度 dim
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  // 返回视图，即具有新尺寸的张量
  return t.view(size);
}

// 根据训练状态、内存格式和维度，获取 cuDNN 批归一化的模式
cudnnBatchNormMode_t getCudnnBatchNormMode(
    bool training,
    at::MemoryFormat memory_format,
    int64_t dim) {
  // 如果维度是2，则返回逐激活的批归一化模式
  if (dim == 2) {
    return CUDNN_BATCHNORM_PER_ACTIVATION;
  } else if (training && memory_format == at::MemoryFormat::ChannelsLast) {
    // 如果在训练且内存格式为ChannelsLast，则返回空间持久化的批归一化模式
    return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  } else if (training && memory_format == at::MemoryFormat::ChannelsLast3d) {
    // 如果在训练且内存格式为ChannelsLast3d，则返回空间持久化的批归一化模式
    return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  } else {
    // 否则，回退到普通的空间批归一化模式
    // TODO: 在 CuDNN 7 中引入了 CUDNN_BATCHNORM_SPATIAL_PERSISTENT 模式，用于性能优化，
    // 但对于某些卷积模型（如 ResNeXt-101 和视频 R(2+1)D），会导致精度损失。因此我们将回退到普通的 CUDNN_BATCHNORM_SPATIAL 模式。
    return CUDNN_BATCHNORM_SPATIAL;
  }
}

} // namespace
// 获取 CUDNN 批归一化操作所需的保留空间大小
size_t _get_cudnn_batch_norm_reserve_space_size(
    const Tensor& input_t,   // 输入张量
    bool training) {         // 训练模式标志

  size_t reserve_size;  // 保留空间大小变量
  TensorArg input{input_t, "input", 1};  // 输入张量参数
  TensorDescriptor idesc{*input, 4};     // 输入张量描述符
  auto handle = getCudnnHandle();        // 获取 CUDNN 句柄
  cudnnBatchNormMode_t mode = getCudnnBatchNormMode(
      training, input->suggest_memory_format(), input->dim());  // 获取 CUDNN 批归一化模式
  auto op = CUDNN_BATCHNORM_OPS_BN;       // CUDNN 批归一化操作
  AT_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      handle, mode, op, nullptr, idesc.desc(), &reserve_size));  // 获取训练阶段的保留空间大小
  return reserve_size;  // 返回保留空间大小
}

// 执行 CUDNN 批归一化操作，返回归一化后的张量及相关信息
std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input_t,                      // 输入张量
    const Tensor& weight_t,                     // 权重张量
    const std::optional<Tensor>& bias_t_opt,    // 可选偏置张量
    const std::optional<Tensor>& running_mean_t_opt,   // 可选运行时均值张量
    const std::optional<Tensor>& running_var_t_opt,    // 可选运行时方差张量
    bool training,                              // 训练模式标志
    double exponential_average_factor,          // 指数加权平均因子
    double epsilon) {                           // 归一化过程中的 epsilon 参数

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned =
      at::borrow_from_optional_tensor(bias_t_opt);  // 从可选的偏置张量中获取 MaybeOwned 张量
  const Tensor& bias_t = *bias_t_maybe_owned;       // 获取最终的偏置张量引用

  const Tensor& running_mean_t =
      c10::value_or_else(running_mean_t_opt, [] { return Tensor(); });  // 获取运行时均值张量或空张量
  const Tensor& running_var_t =
      c10::value_or_else(running_var_t_opt, [] { return Tensor(); });  // 获取运行时方差张量或空张量

  // 定义输入张量及相关参数
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2},
      bias{bias_t, "bias", 3}, running_mean{running_mean_t, "running_mean", 4},
      running_var{running_var_t, "running_var", 5};
  CheckedFrom c = "cudnn_batch_norm";  // 错误检查来源

  // 检查所有必须定义的张量
  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});  // 如果不是训练模式，检查运行时均值和方差是否定义
  }
  // 检查所有张量在相同的 GPU 上
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);  // 如果输入是半精度，检查权重张量是否为单精度
  } else {
    checkAllSameType(c, {input, weight});  // 否则检查输入和权重张量的数据类型相同
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});  // 检查权重、偏置、运行时均值和方差的数据类型相同
  // 检查所有张量是否是连续的
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  // TODO: TensorArg check should start handle memory format
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));  // 检查输入张量是否按照推荐的内存格式连续

  // 检查输入张量的维度范围
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);  // 获取输入张量的特征数量

  // 检查权重、偏置、运行时均值和方差张量的元素数量是否一致
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
  }
}

cudnnBatchNormMode_t mode = getCudnnBatchNormMode(
    training, input->suggest_memory_format(), input->dim());

auto output_t =
    at::empty_like(*input, input->options(), input->suggest_memory_format());

TensorArg output{output_t, "output", 0};

auto handle = getCudnnHandle();
auto dataType = getCudnnDataType(*input);
TensorDescriptor idesc{*input, 4}; // input descriptor
TensorDescriptor wdesc{
    expandScale(*weight, input->dim()),
    4}; // descriptor for weight, bias, running_mean, etc.

Constant one(dataType, 1);
Constant zero(dataType, 0);
Tensor save_mean, save_var;

Tensor reserve;

if (training) {
  // Calculate the number of features and initialize save_mean and save_var tensors
  int64_t num_features = input_t.size(1);
  save_mean = at::empty({num_features}, weight_t.options());
  save_var = at::empty({num_features}, weight_t.options());

  auto op = CUDNN_BATCHNORM_OPS_BN;
  size_t workspace_size;
  // Query workspace size required for batch normalization forward training
  AT_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle,
      mode,
      op,
      idesc.desc(),
      idesc.desc(),
      idesc.desc(),
      wdesc.desc(),
      nullptr,
      &workspace_size));
  // Allocate workspace tensor
  Tensor workspace = at::empty(workspace_size, input->options().dtype(kByte));

  // Query reserve space size and allocate reserve tensor
  size_t reserve_size =
      _get_cudnn_batch_norm_reserve_space_size(input_t, true /* training */);
  reserve = at::empty(reserve_size, input->options().dtype(kByte));

  // Perform batch normalization forward training
  AT_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
      handle,
      mode,
      op,
      &one,
      &zero,
      idesc.desc(),
      input->const_data_ptr(),
      nullptr, // z descriptor for BN-Add-Relu
      nullptr, // z for BN-Add-ReLU
      idesc.desc(),
      output->data_ptr(),
      wdesc.desc(),
      weight->const_data_ptr(),
      bias->const_data_ptr(),
      exponential_average_factor,
      at::maybe_data_ptr(running_mean),
      at::maybe_data_ptr(running_var),
      epsilon,
      save_mean.mutable_data_ptr(),
      save_var.mutable_data_ptr(),
      nullptr,
      workspace.data_ptr(),
      workspace_size,
      reserve.mutable_data_ptr(),
      reserve_size));
} else {
  // Allocate an empty reserve tensor
  reserve = at::empty({0}, input->options().dtype(kByte));
  // Initialize save_mean and save_var tensors as empty tensors
  // This keeps a consistent output with native_batch_norm
  save_mean = at::empty({0}, weight_t.options());
  save_var = at::empty({0}, weight_t.options());
    // 调用 cuDNN 库进行批量归一化的推理过程
    AT_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        handle,                     // cuDNN 批量归一化操作的句柄
        mode,                       // 批量归一化的模式（如训练模式或推理模式）
        &one,                       // 归一化操作的缩放因子的指针（1.0）
        &zero,                      // 归一化操作的偏移因子的指针（0.0）
        idesc.desc(),               // 输入数据的描述符
        input->const_data_ptr(),    // 输入数据的指针
        idesc.desc(),               // 输出数据的描述符（与输入描述符相同）
        output->data_ptr(),         // 输出数据的指针
        wdesc.desc(),               // 权重数据的描述符
        weight->const_data_ptr(),   // 权重数据的指针
        bias->const_data_ptr(),     // 偏置数据的指针
        running_mean->const_data_ptr(),  // 移动平均的均值数据的指针
        running_var->const_data_ptr(),   // 移动平均的方差数据的指针
        epsilon));                  // 一个小的常数，用于防止除以零的情况

  }

  // save_mean 和 save_var 可能未定义
  // 如果这会导致问题，我们可以将它们初始化为空张量，保证它们是正确类型的
  // 返回由四个张量组成的元组，分别是输出张量、保存的均值、保存的方差和保留值
  return std::tuple<Tensor, Tensor, Tensor, Tensor>{
      output_t, save_mean, save_var, reserve};
}

// 注意事项：CuDNN 只在训练模式下实现批归一化的反向算法（评估模式下的批归一化有不同的算法），
// 因此此函数不接受 'training' 参数。

// 定义函数 cudnn_batch_norm_backward，接收多个输入张量和参数
std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input_t,                     // 输入张量
    const Tensor& grad_output_t,               // 梯度输出张量
    const Tensor& weight_t,                    // 权重张量
    const std::optional<Tensor>& running_mean_opt,    // 运行时均值（可选）
    const std::optional<Tensor>& running_var_opt,     // 运行时方差（可选）
    const std::optional<Tensor>& save_mean_t_opt,      // 保存的均值（可选）
    const std::optional<Tensor>& save_var_t_opt,       // 保存的方差（可选）
    double epsilon,                            // epsilon 参数
    const Tensor& reserveSpace) {              // 保留空间张量

  // 使用 value_or_else 确保 save_mean_t_opt 和 save_var_t_opt 不为空
  const Tensor& save_mean_t =
      c10::value_or_else(save_mean_t_opt, [] { return Tensor(); });
  const Tensor& save_var_t =
      c10::value_or_else(save_var_t_opt, [] { return Tensor(); });

  // TODO: 是否值得调用 contiguous，或者应该根据给定的格式进行操作？

  // 将梯度输出张量进行连续化处理，使用输入建议的内存格式
  auto grad_output_contig =
      grad_output_t.contiguous(input_t.suggest_memory_format());
  // 定义 TensorArg 对象，用于后续的检查和处理
  TensorArg input{input_t, "input", 1},
      grad_output{grad_output_contig, "grad_output", 2},
      weight{weight_t, "weight", 3}, save_mean{save_mean_t, "save_mean", 4},
      save_var{save_var_t, "save_var", 5},
      reserve{reserveSpace, "reserve_space", 6};
  CheckedFrom c = "cudnn_batch_norm_backward";

  // 检查所有张量是否已定义
  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  // 检查所有张量是否在同一 GPU 上
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  // 如果输入张量的标量类型是 Half，则权重张量的标量类型应为 Float
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    // 否则，检查输入张量和权重张量的标量类型是否相同
    checkAllSameType(c, {input, weight});
  }
  // 检查输入张量和梯度输出张量的标量类型是否相同
  checkAllSameType(c, {input, grad_output});
  // 检查权重、保存的均值和保存的方差张量的标量类型是否相同
  checkAllSameType(c, {weight, save_mean, save_var});
  // TODO: 是否要求权重张量是连续的？
  checkAllContiguous(c, {save_mean, save_var});
  // TODO: TensorArg 检查应该从处理内存格式开始
  // 检查输入张量和梯度输出张量是否连续化，使用输入建议的内存格式
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  TORCH_CHECK(grad_output->is_contiguous(input->suggest_memory_format()));
  // 检查输入张量的维度范围是否在 [2, 6) 内
  checkDimRange(c, input, 2, 6 /* exclusive */);
  // 检查输入张量和梯度输出张量的大小是否相同
  checkSameSize(c, input, grad_output);
  // 获取输入张量的特征数量（第二维的大小）
  auto num_features = input->size(1);
  // 遍历权重、保存的均值和保存的方差张量
  for (auto t : {weight, save_mean, save_var}) {
  // 调用自定义函数 checkNumel 检查 c, t, num_features 的匹配性
  checkNumel(c, t, num_features);
}

// 获取用于训练的 CuDNN 批归一化模式，并根据输入推荐的内存格式确定
cudnnBatchNormMode_t mode = getCudnnBatchNormMode(
    true, // training
    input->suggest_memory_format(),
    input->dim());

// 创建与输入张量相同大小和内存格式的梯度输入张量
auto grad_input_t = at::empty(
    input->sizes(), input->options(), input->suggest_memory_format());
// 创建与权重张量相同大小和选项的梯度权重张量
auto grad_weight_t = at::empty(weight->sizes(), weight->options());
// 创建与权重张量相同大小和选项的梯度偏置张量
auto grad_bias_t = at::empty(weight->sizes(), weight->options());

// 获取 CuDNN 句柄
auto handle = getCudnnHandle();
// 获取输入张量的数据类型
auto dataType = getCudnnDataType(*input);

// 创建输入和梯度输出的张量描述符
TensorDescriptor idesc{*input, 4}; // input, grad_output descriptor
TensorDescriptor odesc{*grad_output, 4}; // input, grad_output descriptor
// 创建权重、保存均值等张量的描述符
TensorDescriptor wdesc{
    expandScale(*weight, input->dim()),
    4}; // descriptor for weight, save_mean, etc.

// 创建常量对象 one 和 zero，分别表示值 1 和 0，类型为 dataType
Constant one(dataType, 1);
Constant zero(dataType, 0);

// 指定操作为批归一化操作
auto op = CUDNN_BATCHNORM_OPS_BN;

size_t workspace_size;
// 获取执行批归一化反向传播需要的工作空间大小
AT_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    handle,
    mode,
    op,
    idesc.desc(),
    idesc.desc(),
    idesc.desc(),
    nullptr,
    odesc.desc(),
    wdesc.desc(),
    nullptr,
    &workspace_size));
// 创建相应大小的工作空间张量
Tensor workspace = at::empty(workspace_size, input->options().dtype(kByte));

// 执行批归一化反向传播操作
AT_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
    handle,
    mode,
    op,
    &one,
    &zero,
    &one,
    &zero,
    idesc.desc(),
    input->const_data_ptr(),
    nullptr,
    nullptr,
    odesc.desc(),
    grad_output->const_data_ptr(),
    nullptr,
    nullptr,
    idesc.desc(),
    grad_input_t.data_ptr(),
    wdesc.desc(),
    weight->const_data_ptr(),
    nullptr,
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->const_data_ptr(),
    save_var->const_data_ptr(),
    nullptr,
    workspace.data_ptr(),
    workspace_size,
    reserve->data_ptr(),
    reserve->numel()));

// 返回包含梯度输入、梯度权重和梯度偏置的元组
return std::tuple<Tensor, Tensor, Tensor>{
    grad_input_t, grad_weight_t, grad_bias_t};
}

} // namespace native
} // namespace at

#endif
```