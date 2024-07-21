# `.\pytorch\aten\src\ATen\native\mkldnn\Normalization.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
// 引入 ATen 核心 Tensor 头文件

#include <ATen/Config.h>
// 引入 ATen 配置头文件

#include <tuple>
// 引入 C++ 标准库中的 tuple 头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_batch_norm_with_update_native.h>
#include <ATen/ops/batch_norm_backward_native.h>
#include <ATen/ops/_native_batch_norm_legit_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/native_batch_norm_backward_native.h>
#include <ATen/ops/native_batch_norm_native.h>
#endif
// 根据编译设置，选择性地引入 ATen 的不同头文件

#include <ATen/native/mkldnn/Utils.h>
// 引入 ATen MKLDNN 工具函数头文件

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "mkldnn_batch_norm: ATen not compiled with MKLDNN support");
  // 如果未启用 MKLDNN，抛出错误信息
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& grad_output,
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  TORCH_CHECK(false, "mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
  // 如果未启用 MKLDNN，抛出错误信息
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace) {
  TORCH_CHECK(false, "mkldnn_layer_norm_last_index_weight_bias_f32: ATen not compiled with MKLDNN support");
  // 如果未启用 MKLDNN，抛出错误信息
}

std::tuple<Tensor, Tensor, Tensor> _mkldnn_batch_norm_legit(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, Tensor& running_mean, Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "_mkldnn_batch_norm_legit: ATen not compiled with MKLDNN support");
  // 如果未启用 MKLDNN，抛出错误信息
}


std::tuple<Tensor, Tensor, Tensor> _mkldnn_batch_norm_legit_no_stats(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "_mkldnn_batch_norm_legit_no_stats: ATen not compiled with MKLDNN support");
  // 如果未启用 MKLDNN，抛出错误信息
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_mkldnn(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps) {
  TORCH_CHECK(false, "_batch_norm_with_update_mkldnn: ATen not compiled with MKLDNN support");
  // 如果未启用 MKLDNN，抛出错误信息
}

std::tuple<Tensor, Tensor, Tensor> _new_batch_norm_backward_mkldnn(
    const Tensor& grad_out, const Tensor& input, const Tensor& mean, const Tensor& invstd,
    const Tensor& weight, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps, const std::array<bool,3>& grad_input_mask) {
  TORCH_CHECK(false, "_new_batch_norm_backward_mkldnn: ATen not compiled with MKLDNN support");
  // 如果未启用 MKLDNN，抛出错误信息
}

} // namespace native
} // namespace at

#endif  // !AT_MKLDNN_ENABLED()
// 结束条件编译指令，检查是否启用 MKLDNN
    // 检查是否编译了 MKLDNN 支持，如果没有则抛出错误
    TORCH_CHECK(false, "_new_batch_norm_backward_mkldnn: ATen not compiled with MKLDNN support");
#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/layer_norm.h>
#include <ideep/abstract_types.hpp>

namespace at {
namespace native {

// 定义函数 mkldnn_layer_norm_last_index_weight_bias_f32，接受输入张量 input、归一化形状 normalized_shape、权重 weight、偏置 bias、epsilon 参数 eps 和 inplace 标志
std::tuple<Tensor, Tensor, Tensor> mkldnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace) {

  // 内部断言，确保 normalized_shape 的维度为 1，只接受最后一个维度的形状
  TORCH_INTERNAL_ASSERT(normalized_shape.size() == 1, "only accept shapes with the last dimension");
  // 内部断言，确保 input 张量的标量类型为 kFloat
  TORCH_INTERNAL_ASSERT(input.scalar_type() == at::kFloat);
  // 调用 _check_layer_norm_inputs 函数，验证并获取归一化参数 M_N
  auto M_N = at::native::_check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;

  // 创建 mean 和 rstd 张量，使用 MKLDNN 引擎创建空的张量
  auto mean = empty_mkldnn(
        {M},
        input.scalar_type(),
        input.options().layout_opt(),
        input.options().device_opt(),
        input.options().pinned_memory_opt());
  auto rstd = empty_mkldnn(
        {M},
        input.scalar_type(),
        input.options().layout_opt(),
        input.options().device_opt(),
        input.options().pinned_memory_opt());

  // 获取 mean 和 rstd 的 ideep::tensor 包装器
  auto mean_it = at::native::itensor_from_mkldnn(mean);
  auto rstd_it = at::native::itensor_from_mkldnn(rstd);

  // 获取 input、weight、bias 的 ideep::tensor 包装器
  auto input_it = at::native::itensor_from_mkldnn(input);
  auto weight_it = at::native::itensor_from_mkldnn(weight);
  auto bias_it = at::native::itensor_from_mkldnn(bias);

  // 如果 inplace 为 true，则直接使用 input_it；否则，创建一个 ideep::tensor 对象 out_it
  auto out_it = inplace ? input_it : ideep::tensor(input_it.get_desc());

  // 调用 ideep::layer_normalization_forward::compute 函数进行层归一化的前向计算
  ideep::layer_normalization_forward::compute(input_it, weight_it, bias_it, out_it, mean_it, rstd_it, static_cast<float>(eps));

  // 根据计算结果创建新的 Torch 张量 dst，封装 ideep::tensor 对象 out_it
  auto dst = at::native::new_with_itensor_mkldnn(
      std::move(out_it),
      optTypeMetaToScalarType(input.options().dtype_opt()),
      input.options().device_opt());

  // 返回结果，包括 dst、mean 和 rstd 这三个张量
  return std::make_tuple(dst, mean, rstd);
}


// 定义函数 mkldnn_batch_norm，接受输入张量 input 和可选的权重、偏置、运行时均值、运行时方差、训练标志 train、动量 momentum、epsilon 参数 eps
std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  // 从权重可选张量获取权重，从偏置可选张量获取偏置、从运行时均值可选张量获取运行时均值、从运行时方差可选张量获取运行时方差
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  // 如果输入张量的标量类型为 ScalarType::BFloat16
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_batch_norm: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");


    // 检查是否支持 mkldnn 的 bf16 路径，需要 CPU 支持 avx512bw、avx512vl 和 avx512dq



  }
  TORCH_CHECK(weight.defined() && bias.defined(),
             "mkldnn_batch_norm: currently mkldnn only support affine model");


  // 检查权重和偏置是否已定义，当前 mkldnn 仅支持仿射模型



  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor w = itensor_from_tensor(weight);
  ideep::tensor b = itensor_from_tensor(bias);
  bool use_running_stat = (running_mean.defined() && running_var.defined());


  // 转换输入、权重和偏置为 mkldnn 的 tensor 表示
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor w = itensor_from_tensor(weight);
  ideep::tensor b = itensor_from_tensor(bias);
  // 检查是否使用运行统计信息
  bool use_running_stat = (running_mean.defined() && running_var.defined());



  ideep::tensor y;


  // 定义 mkldnn 的输出 tensor y



  if (train) {
    // TODO: enable 3d batchnorm.
    TORCH_CHECK(input.dim() == 4,
        "mkldnn_batch_norm: currently mkldnn training only support 2d batchnorm");


    // 如果是训练模式，则检查输入维度是否为 4
    // TODO: 启用 3D 批归一化
    TORCH_CHECK(input.dim() == 4,
        "mkldnn_batch_norm: currently mkldnn training only support 2d batchnorm");



    ideep::tensor saved_mean;
    ideep::tensor saved_var;
    ideep::batch_normalization_forward_training::compute(
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        x, w, b, y, saved_mean, saved_var, momentum, eps);


    // 定义保存均值和方差的 tensor
    ideep::tensor saved_mean;
    ideep::tensor saved_var;
    // 执行 mkldnn 的批归一化前向训练计算
    ideep::batch_normalization_forward_training::compute(
        x, w, b, y, saved_mean, saved_var, momentum, eps);



    if (use_running_stat) {
      auto len = x.get_nelems() / w.get_nelems(); // n*h*w
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      const std::vector<float> scales_mean{static_cast<float>(1 - momentum),
                                           static_cast<float>(momentum)};
      const std::vector<float> scales_var{static_cast<float>(1 - momentum),
                                          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
                                          static_cast<float>(momentum * len / (len - 1))};
      ideep::sum::compute(scales_mean, {m, saved_mean}, m);
      ideep::sum::compute(scales_var, {v, saved_var}, v);
    }


    // 如果使用运行统计信息，则更新均值和方差
    if (use_running_stat) {
      auto len = x.get_nelems() / w.get_nelems(); // 计算元素数量 n*h*w
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      const std::vector<float> scales_mean{static_cast<float>(1 - momentum),
                                           static_cast<float>(momentum)};
      const std::vector<float> scales_var{static_cast<float>(1 - momentum),
                                          static_cast<float>(momentum * len / (len - 1))};
      // 更新均值和方差
      ideep::sum::compute(scales_mean, {m, saved_mean}, m);
      ideep::sum::compute(scales_var, {v, saved_var}, v);
    }



    return std::make_tuple(
         new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt()),
         new_with_itensor_mkldnn(std::move(saved_mean), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()),
         new_with_itensor_mkldnn(std::move(saved_var), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()));


    // 返回 mkldnn tensor 对象的元组
    return std::make_tuple(
         new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt()),
         new_with_itensor_mkldnn(std::move(saved_mean), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()),
         new_with_itensor_mkldnn(std::move(saved_var), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()));



  } else {
    TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
        "mkldnn_batch_norm: currently mkldnn inference only support 2d and 3d batchnorm");


    // 如果不是训练模式，则检查输入维度是否为 4 或 5
    TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
        "mkldnn_batch_norm: currently mkldnn inference only support 2d and 3d batchnorm");



    if (use_running_stat) {
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      ideep::batch_normalization_forward_inference::compute(
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          x, m, v, w, b, y, eps);
    } else {
      // TODO: keep running estimates.
      TORCH_CHECK(false, "mkldnn_batch_norm: mkldnn inference is not keep running estimates.");
    }


    // 如果使用运行统计信息，则执行 mkldnn 推断时的批归一化计算
    if (use_running_stat) {
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      ideep::batch_normalization_forward_inference::compute(
          x, m, v, w, b, y, eps);
    } else {
      // 否则，输出错误信息，暂不支持保留运行估计
      TORCH_CHECK(false, "mkldnn_batch_norm: mkldnn inference is not keep running estimates.");
    }
    # 返回一个包含三个元素的 std::tuple
    return std::make_tuple(
        # 使用 new_with_itensor_mkldnn 创建新的 ideep::tensor，并移动 y 到新的 ideep::tensor
        new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()),
        # 使用 new_with_itensor_mkldnn 创建一个空的 ideep::tensor，使用 weight 的数据类型和设备选项
        new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()),
        # 使用 new_with_itensor_mkldnn 创建一个空的 ideep::tensor，使用 weight 的数据类型和设备选项
        new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()));
    }
}

# 使用 MKL-DNN 执行带更新的批量归一化操作
std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_mkldnn(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps) {
  # 定义输出张量、保存均值和保存方差张量
  Tensor output, save_mean, save_var;
  # 调用 MKL-DNN 提供的批量归一化函数，获取输出、保存的均值和方差张量
  std::tie(output, save_mean, save_var) =
    mkldnn_batch_norm(input, weight_opt, bias_opt, running_mean, running_var, /*train*/true, momentum, eps);
  # 创建空的 MKL-DNN 张量作为保留张量
  Tensor reserve = empty_mkldnn({0}, input.scalar_type());
  # 返回包含输出、保存均值、保存方差和保留张量的元组
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, save_mean, save_var, reserve);
}

# 使用 MKL-DNN 执行批量归一化操作，返回输出、保存的均值和方差张量的元组
std::tuple<Tensor, Tensor, Tensor> _mkldnn_batch_norm_legit(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, Tensor& running_mean, Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  return mkldnn_batch_norm(input, weight_opt, bias_opt, running_mean, running_var, train, momentum, eps);
}

# 使用 MKL-DNN 执行批量归一化操作，不包括统计信息，返回输出、保存的均值和方差张量的元组
std::tuple<Tensor, Tensor, Tensor> _mkldnn_batch_norm_legit_no_stats(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    bool train,
    double momentum,
    double eps) {
  return mkldnn_batch_norm(input, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, eps);
}

# 使用 MKL-DNN 执行批量归一化的反向传播操作，返回梯度输入、梯度权重和梯度偏置张量的元组
std::tuple<Tensor, Tensor, Tensor> _new_batch_norm_backward_mkldnn(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    bool update, double eps, std::array<bool,3> grad_input_mask, const Tensor& reserve) {
  return mkldnn_batch_norm_backward(grad_output, input, weight, running_mean_opt, running_var_opt, save_mean_opt, save_var_opt, update, eps, grad_input_mask);
}

# 使用 MKL-DNN 执行批量归一化的反向传播操作，返回梯度输入、梯度权重和梯度偏置张量的元组
std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(const Tensor& grad_output,
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_invstd_opt,
    bool train,
    double eps,
    // 使用传入的权重参数（可能是一个可选的张量），获取对应的 MaybeOwned<Tensor> 对象
    c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
    // 将 MaybeOwned<Tensor> 转换为常量引用 Tensor 对象，表示权重张量
    const Tensor& weight = *weight_maybe_owned;
    // 从可选的 save_mean_opt 参数中获取保存的均值张量，如果不存在则返回一个空张量
    const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});
    // 从可选的 save_invstd_opt 参数中获取保存的标准差倒数张量，如果不存在则返回一个空张量
    const Tensor& save_invstd = c10::value_or_else(save_invstd_opt, [] {return Tensor();});
    
    // 检查是否处于训练模式，因为 mkldnn_batch_norm_backward 目前仅支持训练模式
    TORCH_CHECK(train, "mkldnn_batch_norm_backward: currently mkldnn only support train model");
    
    // 从 mkldnn 格式的输出张量获取 grad_output 对应的 ideep::tensor
    ideep::tensor& grady = itensor_from_mkldnn(grad_output);
    // 从 mkldnn 格式的输入张量获取 input 对应的 ideep::tensor
    ideep::tensor& x = itensor_from_mkldnn(input);
    // 将权重张量转换为 ideep::tensor 格式
    ideep::tensor w = itensor_from_tensor(weight);
    // 从 mkldnn 格式的保存均值张量获取 save_mean 对应的 ideep::tensor
    ideep::tensor& m = itensor_from_mkldnn(save_mean);
    // 从 mkldnn 格式的保存标准差倒数张量获取 save_invstd 对应的 ideep::tensor
    ideep::tensor& v = itensor_from_mkldnn(save_invstd);
    
    // 初始化用于存储梯度的 ideep::tensor 对象
    ideep::tensor gradx, gradw, gradb;
    // 调用批标准化反向传播的计算函数，计算梯度
    ideep::batch_normalization_backward::compute(
        // 忽略类型转换警告，传入参数顺序为 x, m, v, grady, w，计算结果存储在 gradx, gradw, gradb 中
        x, m, v, grady, w, gradx, gradw, gradb, eps);
    
    // 返回三个张量的元组，这些张量分别表示对输入、权重和偏置的梯度
    return std::make_tuple(
        // 将 mkldnn 格式的 gradx 转换为 Torch 的 Tensor，并使用输入的数据类型和设备类型创建新张量
        new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()),
        // 将 mkldnn 格式的 gradw 转换为 Torch 的 Tensor，并使用权重的数据类型和设备类型创建新张量
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw),
                                                optTypeMetaToScalarType(weight.options().dtype_opt()),
                                                weight.options().device_opt())),
        // 将 mkldnn 格式的 gradb 转换为 Torch 的 Tensor，并使用权重的数据类型和设备类型创建新张量
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb),
                                                optTypeMetaToScalarType(weight.options().dtype_opt()),
                                                weight.options().device_opt())));
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED


注释：


} // 关闭 at 命名空间

} // 关闭 native 命名空间

} // 关闭 at 命名空间的条件编译结束指令

#endif // 如果 MKLDNN 被启用，则结束条件编译
```