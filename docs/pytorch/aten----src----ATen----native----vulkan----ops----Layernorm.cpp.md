# `.\pytorch\aten\src\ATen\native\vulkan\ops\Layernorm.cpp`

```
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <ATen/Context.h>
#include <c10/util/irange.h>

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/native_layer_norm.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

// 定义 LayernormPackedContext 类的构造函数，接受权重、偏置和 epsilon 参数
LayernormPackedContext::LayernormPackedContext(
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    double eps)
    : unpacked_{c10::AnyType::get()} {
  packed_.reserve(ListArgs::kNumArgs);

  // 断言必须提供权重张量，否则抛出错误
  TORCH_CHECK(weight, "Weight must be provided!");
  // 将权重 Vulkan 张量添加到 packed_ 中
  packed_.emplace_back(weight->vulkan());
  // 断言必须提供偏置张量，否则抛出错误
  TORCH_CHECK(bias, "Bias must be provided!");
  // 将偏置 Vulkan 张量添加到 packed_ 中
  packed_.emplace_back(bias->vulkan());
  // 将 epsilon 值添加到 packed_ 中
  packed_.emplace_back(eps);

  // 如果全局上下文不释放预打包时的权重和偏置，将它们也添加到 unpacked_ 中
  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(ListArgs::kNumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(bias);
    unpacked_.emplace_back(eps);
  }
}

// 静态方法，根据未打包的参数列表创建 LayernormPackedContext 对象
LayernormPackedContext LayernormPackedContext::pack(
    c10::impl::GenericList unpacked) {
  return LayernormPackedContext(
      get_optional_tensor(unpacked, ListArgs::kWeight),
      get_optional_tensor(unpacked, ListArgs::kBias),
      unpacked.get(ListArgs::kEps).toDouble());
}

// 创建并返回 LayernormPackedContext 的智能指针，包装调用构造函数的过程
c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps) {
  return c10::make_intrusive<LayernormPackedContext>(
      LayernormPackedContext(weight, bias, eps));
}

// 运行 LayernormPackedContext 上下文，应用 layer norm 操作并返回结果张量
Tensor run_layernorm_context(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context) {
  // 如果输入张量是 Vulkan 张量，则直接使用；否则转换为 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  // 从 layernorm_context 获取权重、偏置和 epsilon 值
  const std::optional<Tensor>& weight_opt =
      layernorm_context->get_val(LayernormPackedContext::ListArgs::kWeight)
          .toTensor();
  const std::optional<Tensor>& bias_opt =
      layernorm_context->get_val(LayernormPackedContext::ListArgs::kBias)
          .toTensor();
  const float eps = api::utils::safe_downcast<float>(
      layernorm_context->get_val(LayernormPackedContext::ListArgs::kEps)
          .toDouble());

  // 调用 native_layer_norm 函数进行 layer norm 操作，返回一个张量元组
  // 元组包含三个张量：layer_norm 结果、均值和 1/sqrt(var+eps)，这里只需要第一个张量
  std::tuple<Tensor, Tensor, Tensor> native_layer_norm_output =
      at::native_layer_norm(input, normalized_shape, weight_opt, bias_opt, eps);
  // 返回 layer_norm 结果张量
  return std::get<0>(native_layer_norm_output);
}

// 对外公开的 layer_norm 函数，调用 run_layernorm_context 进行 layer norm 操作
Tensor layer_norm(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps,
    # 定义一个布尔类型参数（cudnn_enable），该参数已被弃用
    bool /* cudnn_enable, deprecated */) {
  # 调用 run_layernorm_context 函数，并返回其结果
  return run_layernorm_context(
      # 输入参数（input_arg）作为函数的第一个参数
      input_arg,
      # 归一化的形状参数（normalized_shape）作为函数的第二个参数
      normalized_shape,
      # 创建一个 LayernormPackedContext 对象，并传入以下参数：
      c10::make_intrusive<LayernormPackedContext>(
          # LayernormPackedContext 的权重参数（weight_opt）
          LayernormPackedContext(weight_opt, 
          # LayernormPackedContext 的偏置参数（bias_opt）
                                bias_opt, 
          # LayernormPackedContext 的 ε 参数（eps）
                                eps)));
} // 结束 at 命名空间

#ifdef USE_VULKAN_API
    // 如果定义了 USE_VULKAN_API 宏，则执行以下内容

    // 实现 Torch 库中 aten 命名空间下 layer_norm 函数，使用 Vulkan API
    TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
        m.impl(TORCH_SELECTIVE_NAME("aten::layer_norm"), TORCH_FN(layer_norm));
    }
#endif /* USE_VULKAN_API */

} // 结束 ops 命名空间
} // 结束 vulkan 命名空间
} // 结束 native 命名空间
} // 结束 at 命名空间
```