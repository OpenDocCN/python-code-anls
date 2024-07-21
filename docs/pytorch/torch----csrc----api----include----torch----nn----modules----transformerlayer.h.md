# `.\pytorch\torch\csrc\api\include\torch\nn\modules\transformerlayer.h`

```
#pragma once

#include <torch/nn/cloneable.h>  // 包含 Torch 库中克隆相关的头文件
#include <torch/nn/module.h>     // 包含 Torch 库中模块相关的头文件
#include <torch/nn/modules/activation.h>   // 包含 Torch 库中激活函数模块的头文件
#include <torch/nn/modules/common.h>       // 包含 Torch 库中通用模块的头文件
#include <torch/nn/modules/dropout.h>      // 包含 Torch 库中 dropout 模块的头文件
#include <torch/nn/modules/linear.h>       // 包含 Torch 库中线性层模块的头文件
#include <torch/nn/modules/normalization.h>    // 包含 Torch 库中归一化模块的头文件
#include <torch/nn/options/transformerlayer.h>  // 包含 Torch 库中 transformer 层选项的头文件
#include <torch/nn/pimpl.h>   // 包含 Torch 库中私有实现的头文件

#include <torch/types.h>     // 包含 Torch 库中类型定义的头文件

#include <ostream>   // 包含标准输出流的头文件

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerEncoderLayer
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerEncoderLayer module.
/// See
/// https://pytorch.org/docs/main/generated/torch.nn.TransformerEncoderLayer.html
/// to learn abouut the exact behavior of this encoder layer model
///
/// See the documentation for `torch::nn::TransformerEncoderLayer` class to
/// learn what constructor arguments are supported for this encoder layer model
///
/// Example:
/// ```
/// TransformerEncoderLayer encoderLayer(TransformerEncoderLayerOptions(512,
/// 8).dropout(0.1));
/// ```
class TORCH_API TransformerEncoderLayerImpl
    : public Cloneable<TransformerEncoderLayerImpl> {
 public:
  /// 构造函数，使用给定的维度和头数初始化 TransformerEncoderLayerImpl
  TransformerEncoderLayerImpl(int64_t d_model, int64_t nhead)
      : TransformerEncoderLayerImpl(
            TransformerEncoderLayerOptions(d_model, nhead)) {}
  
  /// 显式构造函数，使用给定的选项初始化 TransformerEncoderLayerImpl
  explicit TransformerEncoderLayerImpl(TransformerEncoderLayerOptions options_);

  /// 前向传播函数，接受输入张量 src、src_mask 和 src_key_padding_mask
  Tensor forward(
      const Tensor& src,
      const Tensor& src_mask = {},
      const Tensor& src_key_padding_mask = {});

  /// 重置函数，重置模型的状态
  void reset() override;

  /// 重置参数函数，初始化模型的参数
  void reset_parameters();

 protected:
  /// 定义默认参数，第一个参数是 src_mask，默认为空张量；第二个参数是 src_key_padding_mask，默认为空张量
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})

 public:
  /// 存储该 `TransformerEncoderLayer` 的选项
  TransformerEncoderLayerOptions options;

  /// 自注意力模块
  MultiheadAttention self_attn = nullptr;

  /// 第一个线性层
  Linear linear1 = nullptr;

  /// dropout 层
  Dropout dropout = nullptr;

  /// 第二个线性层
  Linear linear2 = nullptr;

  /// 第一个归一化层
  LayerNorm norm1 = nullptr;

  /// 第二个归一化层
  LayerNorm norm2 = nullptr;

  /// 第一个 dropout 层
  Dropout dropout1 = nullptr;

  /// 第二个 dropout 层
  Dropout dropout2 = nullptr;
};

/// `TransformerEncoderLayerImpl` 的 `ModuleHolder` 子类。
/// 查看 `TransformerEncoderLayerImpl` 类的文档以了解其提供的方法，以及如何使用 `torch::nn::TransformerEncoderLayerOptions` 与 `TransformerEncoderLayer`。
/// 查看 `ModuleHolder` 的文档以了解 PyTorch 的模块存储语义。
TORCH_MODULE(TransformerEncoderLayer);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerDecoderLayer
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerDecoderLayer 由自注意力、多头注意力和前馈网络组成。
/// 标准的解码器层基于论文
/// "Attention Is All You Need". Ashish Vaswani, Noam Shazeer, Niki Parmar,
/// Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia
/// Polosukhin. 2017. Attention is all you need. In Advances in Neural
/// Information Processing Systems, pages 6000-6010. This references the seminal
/// paper introducing the Transformer model, a key architecture in modern
/// natural language processing and machine translation.
///
/// Users may modify or implement in a different way during application. See
/// https://pytorch.org/docs/main/nn.html#transformer-layers to learn about
/// the exact behavior of this module. This URL directs users to PyTorch's
/// official documentation for detailed information on Transformer layers.
///
/// See the documentation for `torch::nn::TransformerDecoderLayerOptions` class
/// to learn what constructor arguments are supported for this module. This
/// refers to the specific class and its options in PyTorch's C++ API for
/// configuring Transformer decoder layers.
///
/// Example:
/// ```
/// TransformerDecoderLayer model(TransformerDecoderLayerOptions(512,
/// 8).dropout(0.2));
/// ```
/// This example demonstrates how to instantiate a Transformer decoder layer
/// with specified options (e.g., hidden size of 512, number of attention heads
/// as 8, and dropout rate of 0.2).
class TORCH_API TransformerDecoderLayerImpl


注释部分详细解释了代码段中的引用、文档链接以及示例用法，帮助用户理解和使用TransformerDecoderLayerImpl类。
    /// TransformerDecoderLayerImpl 类的实现，继承自 Cloneable<TransformerDecoderLayerImpl>，表示可克隆的 Decoder 层
    /// 公共部分
    public:
      /// 构造函数，初始化 TransformerDecoderLayerImpl 对象
      TransformerDecoderLayerImpl(int64_t d_model, int64_t nhead)
          : TransformerDecoderLayerImpl(
                TransformerDecoderLayerOptions(d_model, nhead)) {}
    
      /// 显式构造函数，接受 TransformerDecoderLayerOptions 对象作为参数
      explicit TransformerDecoderLayerImpl(TransformerDecoderLayerOptions options_);
    
      /// 重置函数，覆盖基类中的虚函数
      void reset() override;
    
      /// 重置参数的函数
      void reset_parameters();
    
      /// 通过解码器层传递输入（以及掩码）
      /// Args:
      ///       tgt: 解码器层的输入序列（必需）
      ///       memory: 编码器最后一层的序列（必需）
      ///       tgt_mask: tgt 序列的掩码（可选）
      ///       memory_mask: memory 序列的掩码（可选）
      ///       tgt_key_padding_mask: tgt 批次键的掩码（可选）
      ///       memory_key_padding_mask: memory 批次键的掩码（可选）
      Tensor forward(
          Tensor tgt,
          const Tensor& memory,
          const Tensor& tgt_mask = {},
          const Tensor& memory_mask = {},
          const Tensor& tgt_key_padding_mask = {},
          const Tensor& memory_key_padding_mask = {});
    
      /// 配置此模块的选项
      TransformerDecoderLayerOptions options;
    
      /// 自注意力
      MultiheadAttention self_attn{nullptr};
    
      /// 自注意力后的 dropout
      Dropout dropout1{nullptr};
    
      /// 自注意力后的归一化
      LayerNorm norm1{nullptr};
    
      /// 多头注意力
      MultiheadAttention multihead_attn{nullptr};
    
      /// 多头注意力后的 dropout
      Dropout dropout2{nullptr};
    
      /// 多头注意力后的归一化
      LayerNorm norm2{nullptr};
    
      /// 前向传播的第一个线性层
      Linear linear1{nullptr};
    
      /// 前向传播的 dropout 层
      Dropout dropout{nullptr};
    
      /// 前向传播的第二个线性层
      Linear linear2{nullptr};
    
      /// 前向传播后的 dropout
      Dropout dropout3{nullptr};
    
      /// 前向传播后的归一化
      LayerNorm norm3{nullptr};
    
    protected:
      /// 前向传播的默认参数
      FORWARD_HAS_DEFAULT_ARGS(
          {2, AnyValue(Tensor())},
          {3, AnyValue(Tensor())},
          {4, AnyValue(Tensor())},
          {5, AnyValue(Tensor())})
    
      /// 根据配置应用激活函数
      Tensor activation(const Tensor& input);
};

/// 结束 `torch` 命名空间

/// `TransformerDecoderLayerImpl` 的 `ModuleHolder` 子类。
/// 参考 `TransformerDecoderLayerImpl` 类的文档，了解它提供的方法，
/// 以及如何使用 `TransformerDecoderLayer` 和 `torch::nn::TransformerDecoderLayerOptions` 的示例。
/// 查看 `ModuleHolder` 的文档，了解 PyTorch 的模块存储语义。
TORCH_MODULE(TransformerDecoderLayer);

} // namespace nn
} // namespace torch
```