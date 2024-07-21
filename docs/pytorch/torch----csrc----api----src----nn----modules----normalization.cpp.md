# `.\pytorch\torch\csrc\api\src\nn\modules\normalization.cpp`

```
// 包含 Torch 库中的归一化模块头文件
#include <torch/nn/modules/normalization.h>

// 包含 Torch CUDA 相关的头文件
#include <torch/cuda.h>
// 包含 Torch 初始化相关的头文件
#include <torch/nn/init.h>
// 包含 Torch 实用工具的头文件
#include <torch/utils.h>

// 包含输出流相关的头文件
#include <ostream>
// 包含实用工具函数的头文件
#include <utility>

// 将 torch::nn::functional 命名空间重命名为 F，简化使用
namespace F = torch::nn::functional;

// 定义 torch::nn 命名空间
namespace torch {
namespace nn {

// LayerNormImpl 类的构造函数实现
LayerNormImpl::LayerNormImpl(LayerNormOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset() 函数进行初始化
  reset();
}

// LayerNormImpl 类的 reset() 方法实现
void LayerNormImpl::reset() {
  // 如果启用了逐元素仿射变换
  if (options.elementwise_affine()) {
    // 注册 weight 参数，并用空的张量进行初始化
    weight = register_parameter("weight", torch::empty(options.normalized_shape()));
    // 注册 bias 参数，并用空的张量进行初始化
    bias = register_parameter("bias", torch::empty(options.normalized_shape()));
  } else {
    // 否则，不启用梯度的张量进行初始化
    weight = register_parameter("weight", torch::Tensor(), /*requires_grad=*/false);
    bias = register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);
  }
  // 调用 reset_parameters() 方法，进一步初始化参数
  reset_parameters();
}

// LayerNormImpl 类的 reset_parameters() 方法实现
void LayerNormImpl::reset_parameters() {
  // 如果启用了逐元素仿射变换
  if (options.elementwise_affine()) {
    // 使用 ones_() 初始化 weight 参数为全 1 张量
    torch::nn::init::ones_(weight);
    // 使用 zeros_() 初始化 bias 参数为全 0 张量
    torch::nn::init::zeros_(bias);
  }
}

// LayerNormImpl 类的 pretty_print() 方法实现
void LayerNormImpl::pretty_print(std::ostream& stream) const {
  // 输出 LayerNorm 的信息，包括 normalized_shape、eps 和 elementwise_affine
  stream << std::boolalpha << "torch::nn::LayerNorm("
         << torch::IntArrayRef(options.normalized_shape())
         << ", eps=" << options.eps()
         << ", elementwise_affine=" << options.elementwise_affine() << ")";
}

// LayerNormImpl 类的 forward() 方法实现
torch::Tensor LayerNormImpl::forward(const Tensor& input) {
  // 调用 F::detail::layer_norm() 函数执行 LayerNorm 操作
  return F::detail::layer_norm(
      input, options.normalized_shape(), weight, bias, options.eps());
}

// ============================================================================

// LocalResponseNormImpl 类的构造函数实现
LocalResponseNormImpl::LocalResponseNormImpl(
    const LocalResponseNormOptions& options_)
    : options(options_) {}

// LocalResponseNormImpl 类的 forward() 方法实现
torch::Tensor LocalResponseNormImpl::forward(const Tensor& input) {
  // 调用 F::detail::local_response_norm() 函数执行本地响应归一化操作
  return F::detail::local_response_norm(
      input, options.size(), options.alpha(), options.beta(), options.k());
}

// LocalResponseNormImpl 类的 reset() 方法实现
void LocalResponseNormImpl::reset() {}

// LocalResponseNormImpl 类的 pretty_print() 方法实现
void LocalResponseNormImpl::pretty_print(std::ostream& stream) const {
  // 输出 LocalResponseNorm 的信息，包括 size、alpha、beta 和 k
  stream << std::boolalpha << "torch::nn::LocalResponseNorm(" << options.size()
         << ", alpha=" << options.alpha() << ", beta=" << options.beta()
         << ", k=" << options.k() << ")";
}

// ============================================================================

// CrossMapLRN2dImpl 类的 reset() 方法实现
void CrossMapLRN2dImpl::reset() {}

// CrossMapLRN2dImpl 类的 pretty_print() 方法实现
void CrossMapLRN2dImpl::pretty_print(std::ostream& stream) const {
  // 输出 CrossMapLRN2d 的信息，包括 size、alpha、beta 和 k
  stream << std::boolalpha << "torch::nn::CrossMapLRN2d(" << options.size()
         << ", alpha=" << options.alpha() << ", beta=" << options.beta()
         << ", k=" << options.k() << ")";
}

// CrossMapLRN2dImpl 类的 forward() 方法实现
torch::Tensor CrossMapLRN2dImpl::forward(const torch::Tensor& input) {
  // 调用 functions::CrossMapLRN2d::apply() 执行跨通道局部响应归一化操作
  return functions::CrossMapLRN2d::apply(input, options);
}

// ============================================================================

// GroupNormImpl 类的构造函数实现
GroupNormImpl::GroupNormImpl(const GroupNormOptions& options_)
    : options(options_) { // NOLINT(modernize-pass-by-value)
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset() 方法进行初始化
  reset();
}
void GroupNormImpl::reset() {
  // 如果 GroupNorm 层支持仿射变换（affine），则注册权重和偏置参数
  if (options.affine()) {
    weight = register_parameter("weight", torch::empty(options.num_channels()));
    bias = register_parameter("bias", torch::empty(options.num_channels()));
  } else {
    // 如果不支持仿射变换，则注册空的权重和偏置参数，并且不要求梯度计算
    weight =
        register_parameter("weight", torch::Tensor(), /*requires_grad=*/false);
    bias = register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);
  }
  // 重置参数的初始化
  reset_parameters();
}

void GroupNormImpl::reset_parameters() {
  // 如果支持仿射变换，则初始化权重为全1，偏置为全0
  if (options.affine()) {
    torch::nn::init::ones_(weight);
    torch::nn::init::zeros_(bias);
  }
}

torch::Tensor GroupNormImpl::forward(const Tensor& input) {
  // 调用 F::detail::group_norm 函数进行 GroupNorm 操作
  return F::detail::group_norm(
      input, options.num_groups(), weight, bias, options.eps());
}

void GroupNormImpl::pretty_print(std::ostream& stream) const {
  // 打印 GroupNorm 层的信息，包括 num_groups、num_channels、eps 和 affine
  stream << std::boolalpha << "torch::nn::GroupNorm(" << options.num_groups()
         << ", " << options.num_channels() << ", eps=" << options.eps()
         << ", affine=" << options.affine() << ")";
}

} // namespace nn
} // namespace torch
```