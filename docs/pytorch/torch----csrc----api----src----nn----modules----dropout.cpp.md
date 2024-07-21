# `.\pytorch\torch\csrc\api\src\nn\modules\dropout.cpp`

```py
// 引入 Torch 库中的 dropout 功能头文件
#include <torch/nn/functional/dropout.h>
// 引入 Torch 库中的 dropout 模块头文件
#include <torch/nn/modules/dropout.h>

// 引入 Torch 的基本类型定义
#include <torch/types.h>

// 引入 C10 库中的异常处理头文件
#include <c10/util/Exception.h>

// 引入标准库头文件
#include <cstddef>
#include <ostream>
#include <vector>

// Torch 命名空间，使用 F 作为 nn::functional 的别名
namespace F = torch::nn::functional;

// Torch 的 nn 命名空间
namespace torch {
namespace nn {

// DropoutImpl 类的 forward 方法实现
Tensor DropoutImpl::forward(Tensor input) {
  // 调用 nn::functional 命名空间中 detail 命名空间的 dropout 函数，
  // 传入输入张量 input，dropout 概率为 options.p()，当前是否为训练模式以及是否原地操作的选项
  return F::detail::dropout(
      input, options.p(), is_training(), options.inplace());
}

// DropoutImpl 类的 pretty_print 方法实现
void DropoutImpl::pretty_print(std::ostream& stream) const {
  // 将类的参数选项以可读格式打印到输出流中
  stream << std::boolalpha << "torch::nn::Dropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

// Dropout2dImpl 类的 forward 方法实现
Tensor Dropout2dImpl::forward(Tensor input) {
  // 调用 nn::functional 命名空间中 detail 命名空间的 dropout2d 函数，
  // 传入输入张量 input，dropout 概率为 options.p()，当前是否为训练模式以及是否原地操作的选项
  return F::detail::dropout2d(
      input, options.p(), is_training(), options.inplace());
}

// Dropout2dImpl 类的 pretty_print 方法实现
void Dropout2dImpl::pretty_print(std::ostream& stream) const {
  // 将类的参数选项以可读格式打印到输出流中
  stream << std::boolalpha << "torch::nn::Dropout2d(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

// Dropout3dImpl 类的 forward 方法实现
Tensor Dropout3dImpl::forward(Tensor input) {
  // 调用 nn::functional 命名空间中 detail 命名空间的 dropout3d 函数，
  // 传入输入张量 input，dropout 概率为 options.p()，当前是否为训练模式以及是否原地操作的选项
  return F::detail::dropout3d(
      input, options.p(), is_training(), options.inplace());
}

// Dropout3dImpl 类的 pretty_print 方法实现
void Dropout3dImpl::pretty_print(std::ostream& stream) const {
  // 将类的参数选项以可读格式打印到输出流中
  stream << std::boolalpha << "torch::nn::Dropout3d(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

// AlphaDropoutImpl 类的 forward 方法实现
Tensor AlphaDropoutImpl::forward(const Tensor& input) {
  // 调用 nn::functional 命名空间中 detail 命名空间的 alpha_dropout 函数，
  // 传入输入张量 input，dropout 概率为 options.p()，当前是否为训练模式，不进行原地操作
  return F::detail::alpha_dropout(
      input, options.p(), is_training(), /*inplace=*/false);
}

// AlphaDropoutImpl 类的 pretty_print 方法实现
void AlphaDropoutImpl::pretty_print(std::ostream& stream) const {
  // 将类的参数选项以可读格式打印到输出流中
  stream << std::boolalpha << "torch::nn::AlphaDropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

// FeatureAlphaDropoutImpl 类的 forward 方法实现
Tensor FeatureAlphaDropoutImpl::forward(const Tensor& input) {
  // 调用 nn::functional 命名空间中 detail 命名空间的 feature_alpha_dropout 函数，
  // 传入输入张量 input，dropout 概率为 options.p()，当前是否为训练模式，不进行原地操作
  return F::detail::feature_alpha_dropout(
      input, options.p(), is_training(), /*inplace=*/false);
}

// FeatureAlphaDropoutImpl 类的 pretty_print 方法实现
void FeatureAlphaDropoutImpl::pretty_print(std::ostream& stream) const {
  // 将类的参数选项以可读格式打印到输出流中
  stream << std::boolalpha << "torch::nn::FeatureAlphaDropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

} // namespace nn
} // namespace torch
```