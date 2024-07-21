# `.\pytorch\torch\csrc\api\src\nn\modules\distance.cpp`

```
// 包含 torch 库中的 distance.h 头文件，提供了距离计算的相关功能
#include <torch/nn/modules/distance.h>

// 定义命名空间 F 为 torch::nn::functional，简化函数调用
namespace F = torch::nn::functional;

// 定义命名空间 torch::nn，实现神经网络相关功能
namespace torch {
namespace nn {

// 实现 CosineSimilarity 类的构造函数
CosineSimilarityImpl::CosineSimilarityImpl(
    const CosineSimilarityOptions& options_)
    : options(options_) {}

// 重置 CosineSimilarity 类的状态，这里为空实现
void CosineSimilarityImpl::reset() {}

// 打印 CosineSimilarity 类的信息到输出流 stream 中
void CosineSimilarityImpl::pretty_print(std::ostream& stream) const {
  // 使用 std::boolalpha 将布尔值转换为文本形式输出
  stream << std::boolalpha << "torch::nn::CosineSimilarity"
         << "(dim=" << options.dim() << ", eps=" << options.eps() << ")";
}

// 实现 CosineSimilarity 类的前向传播方法
Tensor CosineSimilarityImpl::forward(const Tensor& x1, const Tensor& x2) {
  // 调用 torch::nn::functional 命名空间中的 detail::cosine_similarity 函数
  // 计算 x1 和 x2 的余弦相似度，使用 options 中指定的维度和 eps 值
  return F::detail::cosine_similarity(x1, x2, options.dim(), options.eps());
}

// ============================================================================

// 实现 PairwiseDistance 类的构造函数
PairwiseDistanceImpl::PairwiseDistanceImpl(
    const PairwiseDistanceOptions& options_)
    : options(options_) {}

// 重置 PairwiseDistance 类的状态，这里为空实现
void PairwiseDistanceImpl::reset() {}

// 打印 PairwiseDistance 类的信息到输出流 stream 中
void PairwiseDistanceImpl::pretty_print(std::ostream& stream) const {
  // 使用 std::boolalpha 将布尔值转换为文本形式输出
  stream << std::boolalpha << "torch::nn::PairwiseDistance"
         << "(p=" << options.p() << ", eps=" << options.eps()
         << ", keepdim=" << options.keepdim() << ")";
}

// 实现 PairwiseDistance 类的前向传播方法
Tensor PairwiseDistanceImpl::forward(const Tensor& x1, const Tensor& x2) {
  // 调用 torch::nn::functional 命名空间中的 detail::pairwise_distance 函数
  // 计算 x1 和 x2 的成对距离，使用 options 中指定的 p 值、eps 值和 keepdim 标志
  return F::detail::pairwise_distance(
      x1, x2, options.p(), options.eps(), options.keepdim());
}

} // namespace nn
} // namespace torch
```