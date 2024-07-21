# `.\pytorch\torch\csrc\api\src\nn\modules\fold.cpp`

```
// 包含 Torch 库中的折叠相关头文件
#include <torch/nn/modules/fold.h>

// 包含 Torch 库中的展开数组、类型和实用工具的头文件
#include <torch/expanding_array.h>
#include <torch/types.h>
#include <torch/utils.h>

// 定义命名空间 F 为 torch::nn::functional 命名空间的别名
namespace F = torch::nn::functional;

// 定义命名空间 torch::nn
namespace torch {
namespace nn {

// 实现 Fold 类的构造函数，接受 FoldOptions 类型的参数 options_
FoldImpl::FoldImpl(const FoldOptions& options_) : options(options_) {}

// 重置 Fold 类的状态，实际上是空实现
void FoldImpl::reset() {}

// 打印 Fold 类的信息到给定流对象 stream 中
void FoldImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Fold(output_size=" << options.output_size()
         << ", kernel_size=" << options.kernel_size()
         << ", dilation=" << options.dilation()
         << ", padding=" << options.padding() << ", stride=" << options.stride()
         << ")";
}

// Fold 类的前向传播函数，调用 F::detail::fold 函数实现数据的折叠操作
Tensor FoldImpl::forward(const Tensor& input) {
  return F::detail::fold(
      input,
      options.output_size(),
      options.kernel_size(),
      options.dilation(),
      options.padding(),
      options.stride());
}

// ============================================================================

// 实现 Unfold 类的构造函数，接受 UnfoldOptions 类型的参数 options_
UnfoldImpl::UnfoldImpl(const UnfoldOptions& options_) : options(options_) {}

// 重置 Unfold 类的状态，实际上是空实现
void UnfoldImpl::reset() {}

// 打印 Unfold 类的信息到给定流对象 stream 中
void UnfoldImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Unfold(kernel_size=" << options.kernel_size()
         << ", dilation=" << options.dilation()
         << ", padding=" << options.padding() << ", stride=" << options.stride()
         << ")";
}

// Unfold 类的前向传播函数，调用 F::detail::unfold 函数实现数据的展开操作
Tensor UnfoldImpl::forward(const Tensor& input) {
  return F::detail::unfold(
      input,
      options.kernel_size(),
      options.dilation(),
      options.padding(),
      options.stride());
}

} // namespace nn
} // namespace torch
```