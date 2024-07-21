# `.\pytorch\torch\csrc\api\include\torch\nn\modules\adaptive.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/nn/cloneable.h>
// 引入 Torch 的 nn 模块中的 cloneable 头文件，支持模型克隆功能

#include <torch/nn/functional/activation.h>
// 引入 Torch 的 nn 模块中的 functional/activation 头文件，包含激活函数相关功能

#include <torch/nn/module.h>
// 引入 Torch 的 nn 模块中的 module 头文件，定义神经网络模块的基本接口

#include <torch/nn/modules/container/modulelist.h>
// 引入 Torch 的 nn 模块中的 container/modulelist 头文件，包含模块列表容器的定义

#include <torch/nn/modules/container/sequential.h>
// 引入 Torch 的 nn 模块中的 container/sequential 头文件，包含顺序容器的定义

#include <torch/nn/modules/linear.h>
// 引入 Torch 的 nn 模块中的 linear 头文件，包含线性层的定义

#include <torch/nn/options/adaptive.h>
// 引入 Torch 的 nn 模块中的 options/adaptive 头文件，包含自适应模块的选项

namespace torch {
namespace nn {

/// The output of a single invocation of an AdaptiveLogSoftmaxWithLoss
/// module's `forward()` method.
/// AdaptiveLogSoftmaxWithLoss 模块的 forward() 方法的单次调用输出。
struct TORCH_API ASMoutput {
  ASMoutput(Tensor output_, double loss_);

  /// Tensor containing computed target log probabilities for each example
  /// 包含每个样本的目标对数概率的张量
  Tensor output;

  /// Scalar representing the computed negative log likelihood loss
  /// 表示计算的负对数似然损失的标量
  double loss;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveLogSoftmaxWithLoss
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Efficient softmax approximation as described in
/// `Efficient softmax approximation for GPUs`_ by Edouard Grave, Armand Joulin,
/// Moustapha Cissé, David Grangier, and Hervé Jégou.
/// See
/// https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveLogSoftmaxWithLoss
/// to learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::AdaptiveLogSoftmaxWithLossOptions`
/// class to learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
/// {4, 8}).div_value(2.).head_bias(true));
/// ```
/// 高效的 softmax 近似方法，详见 Edouard Grave 等人的论文《Efficient softmax approximation for GPUs》。
/// 请参阅 https://pytorch.org/docs/main/nn.html#torch.nn.AdaptiveLogSoftmaxWithLoss 了解模块的确切行为。
/// 查看 `torch::nn::AdaptiveLogSoftmaxWithLossOptions` 类的文档，了解此模块支持的构造参数。
///
/// 示例：
/// ```
/// AdaptiveLogSoftmaxWithLoss model(AdaptiveLogSoftmaxWithLossOptions(8, 10,
/// {4, 8}).div_value(2.).head_bias(true));
/// ```
class TORCH_API AdaptiveLogSoftmaxWithLossImpl
    /// `AdaptiveLogSoftmaxWithLossImpl` 类定义，公开继承自 `Cloneable<AdaptiveLogSoftmaxWithLossImpl>` 接口
    /// 该类用于实现自适应的带损失的 LogSoftmax 模块
    
    public:
      /// 构造函数，接受输入特征数 `in_features`，类别数 `n_classes` 和分段信息 `cutoffs`
      /// 使用委托构造函数初始化 `AdaptiveLogSoftmaxWithLossImpl` 对象
      AdaptiveLogSoftmaxWithLossImpl(
          int64_t in_features,
          int64_t n_classes,
          std::vector<int64_t> cutoffs)
          : AdaptiveLogSoftmaxWithLossImpl(AdaptiveLogSoftmaxWithLossOptions(
                in_features,
                n_classes,
                cutoffs)) {}
    
      /// 显式构造函数，接受 `AdaptiveLogSoftmaxWithLossOptions` 对象作为参数
      explicit AdaptiveLogSoftmaxWithLossImpl(
          AdaptiveLogSoftmaxWithLossOptions options_);
    
      /// 前向传播函数，计算给定输入 `input` 和目标 `target` 的输出
      ASMoutput forward(const Tensor& input, const Tensor& target);
    
      /// 重置函数，重写自基类 `Module` 的虚函数 `reset()`
      void reset() override;
    
      /// 重置模型参数的函数
      void reset_parameters();
    
      /// 将模块的信息以美观的形式打印到给定的流 `stream` 中
      void pretty_print(std::ostream& stream) const override;
    
      /// 给定输入张量 `input` 和 `head` 的输出，计算完整分布的对数概率
      Tensor _get_full_log_prob(const Tensor& input, const Tensor& head_output);
    
      /// 计算所有 `n_classes` 的对数概率
      Tensor log_prob(const Tensor& input);
    
      /// 预测函数，等效于 `log_prob(input).argmax(1)`，但在某些情况下更高效
      Tensor predict(const Tensor& input);
    
      /// 构造模块时使用的选项
      AdaptiveLogSoftmaxWithLossOptions options;
    
      /// 用于将目标分配到其桶中的分段信息，应为升序排序的整数序列
      std::vector<int64_t> cutoffs;
    
      /// 短列表的大小
      int64_t shortlist_size;
    
      /// 群集的数量
      int64_t n_clusters;
    
      /// 头部分类器的输出大小
      int64_t head_size;
    
      /// 头部线性层，默认为 `nullptr`
      Linear head = nullptr;
    
      /// 尾部模块列表
      ModuleList tail;
/// 结束 `torch::nn` 命名空间
};

/// `AdaptiveLogSoftmaxWithLossImpl` 的 `ModuleHolder` 子类。
/// 查看 `AdaptiveLogSoftmaxWithLossImpl` 类的文档以了解其提供的方法，
/// 以及如何使用 `AdaptiveLogSoftmaxWithLoss` 与 `torch::nn::AdaptiveLogSoftmaxWithLossOptions` 的示例。
/// 参阅 `ModuleHolder` 的文档以了解 PyTorch 的模块存储语义。
TORCH_MODULE(AdaptiveLogSoftmaxWithLoss);

/// 结束 `nn` 命名空间
} // namespace nn

/// 结束 `torch` 命名空间
} // namespace torch
```