# `.\pytorch\torch\csrc\api\include\torch\nn\modules\distance.h`

```
#pragma once

#include <torch/nn/cloneable.h>  // 引入克隆相关的头文件
#include <torch/nn/functional/distance.h>  // 引入距离函数相关的头文件
#include <torch/nn/options/distance.h>  // 引入距离选项相关的头文件
#include <torch/nn/pimpl.h>  // 引入私有实现相关的头文件
#include <torch/types.h>  // 引入类型相关的头文件

#include <torch/csrc/Export.h>  // 引入导出相关的头文件

namespace torch {
namespace nn {

/// 返回输入张量 input1 和 input2 沿指定维度 dim 的余弦相似度。
/// 详见 https://pytorch.org/docs/main/nn.html#torch.nn.CosineSimilarity 了解此模块的确切行为。
///
/// 参见 `torch::nn::CosineSimilarityOptions` 类的文档，了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// CosineSimilarity model(CosineSimilarityOptions().dim(0).eps(0.5));
/// ```
class TORCH_API CosineSimilarityImpl : public Cloneable<CosineSimilarityImpl> {
 public:
  explicit CosineSimilarityImpl(const CosineSimilarityOptions& options_ = {});  // 显式构造函数

  void reset() override;  // 重置函数的重写

  /// 在给定流中漂亮打印 `CosineSimilarity` 模块。
  void pretty_print(std::ostream& stream) const override;  // 漂亮打印函数的重写

  Tensor forward(const Tensor& input1, const Tensor& input2);  // 前向传播函数

  /// 构造此 `Module` 时使用的选项。
  CosineSimilarityOptions options;  // 构造选项
};

/// `CosineSimilarityImpl` 的 `ModuleHolder` 子类。
/// 参见 `CosineSimilarityImpl` 类的文档，了解它提供的方法，以及如何使用 `CosineSimilarity` 和 `torch::nn::CosineSimilarityOptions`。
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(CosineSimilarity);  // Torch 模块

// ============================================================================

/// 返回使用 p-范数计算的向量 v1 和 v2 之间的批次间成对距离。
/// 详见 https://pytorch.org/docs/main/nn.html#torch.nn.PairwiseDistance 了解此模块的确切行为。
///
/// 参见 `torch::nn::PairwiseDistanceOptions` 类的文档，了解此模块支持的构造函数参数。
///
/// 示例：
/// ```
/// PairwiseDistance model(PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true));
/// ```
class TORCH_API PairwiseDistanceImpl : public Cloneable<PairwiseDistanceImpl> {
 public:
  explicit PairwiseDistanceImpl(const PairwiseDistanceOptions& options_ = {});  // 显式构造函数

  void reset() override;  // 重置函数的重写

  /// 在给定流中漂亮打印 `PairwiseDistance` 模块。
  void pretty_print(std::ostream& stream) const override;  // 漂亮打印函数的重写

  Tensor forward(const Tensor& input1, const Tensor& input2);  // 前向传播函数

  /// 构造此 `Module` 时使用的选项。
  PairwiseDistanceOptions options;  // 构造选项
};

/// `PairwiseDistanceImpl` 的 `ModuleHolder` 子类。
/// 参见 `PairwiseDistanceImpl` 类的文档，了解它提供的方法，以及如何使用 `PairwiseDistance` 和 `torch::nn::PairwiseDistanceOptions`。
/// 参见 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(PairwiseDistance);  // Torch 模块
/// 定义一个名为 `PairwiseDistance` 的宏，用于创建 PyTorch 模块的存储语义。
TORCH_MODULE(PairwiseDistance);
/// 结束 nn 命名空间
} // namespace nn
/// 结束 torch 命名空间
} // namespace torch
```