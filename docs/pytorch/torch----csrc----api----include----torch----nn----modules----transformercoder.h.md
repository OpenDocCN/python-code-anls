# `.\pytorch\torch\csrc\api\include\torch\nn\modules\transformercoder.h`

```
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/options/transformercoder.h>
#include <torch/nn/pimpl.h>

#include <torch/types.h>

#include <ostream>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerEncoder
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerEncoder module.
/// See
/// https://pytorch.org/docs/main/generated/torch.nn.TransformerEncoder.html
/// to learn abouut the exact behavior of this encoder layer module.
///
/// See the documentation for `torch::nn::TransformerEncoder` class to learn
/// what constructor arguments are supported for this encoder module.
///
/// Example:
/// ```
/// TransformerEncoderLayer encoderLayer(TransformerEncoderLayerOptions(512,
/// 8).dropout(0.1)); TransformerEncoder
/// encoder(TransformerEncoderOptions(encoderLayer,
/// 6).norm(LayerNorm(LayerNormOptions({2}))));
/// ```
class TORCH_API TransformerEncoderImpl
    : public Cloneable<TransformerEncoderImpl> {
 public:
  /// Constructor for `TransformerEncoderImpl`.
  ///
  /// \param encoder_layer The encoder layer to be used in this encoder stack.
  /// \param num_layers Number of encoder layers in the stack.
  TransformerEncoderImpl(
      TransformerEncoderLayer encoder_layer,
      int64_t num_layers)
      : TransformerEncoderImpl(
            TransformerEncoderOptions(encoder_layer, num_layers)) {}

  /// Explicit constructor for `TransformerEncoderImpl`.
  ///
  /// \param options_ Options object specifying configuration for the encoder.
  explicit TransformerEncoderImpl(TransformerEncoderOptions options_);

  /// Forward pass of the transformer encoder.
  ///
  /// \param src Source tensor to be encoded.
  /// \param src_mask Optional mask tensor for the source sequence.
  /// \param src_key_padding_mask Optional tensor indicating elements to be masked in the source.
  ///
  /// \returns Encoded tensor from the transformer encoder.
  Tensor forward(
      const Tensor& src,
      const Tensor& src_mask = {},
      const Tensor& src_key_padding_mask = {});

  /// Reset method for resetting internal states of the encoder.
  void reset() override;

  /// Method for resetting parameters of the encoder layers.
  void reset_parameters();

 protected:
  /// Define default arguments for the forward method.
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})

 public:
  /// Options with which this `TransformerEncoder` was constructed.
  TransformerEncoderOptions options;

  /// Module list that contains all the encoder layers.
  ModuleList layers = nullptr;

  /// Optional normalization module.
  AnyModule norm;
};

/// A `ModuleHolder` subclass for `TransformerEncoderImpl`.
/// See the documentation for `TransformerEncoderImpl` class to learn what
/// methods it provides, and examples of how to use `TransformerEncoder` with
/// `torch::nn::TransformerEncoderOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(TransformerEncoder);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerDecoder
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerDecoder is a stack of N decoder layers.
/// See
/// https://pytorch.org/docs/main/generated/torch.nn.TransformerDecoder.html
/// to learn abouut the exact behavior of this decoder module
///
/// See the documentation for `torch::nn::TransformerDecoderOptions` class to
/// learn what constructor arguments are supported for this decoder module
///
/// Example:
/// ```
/// TransformerDecoderLayer decoder_layer(TransformerDecoderLayerOptions(512,
/// 8).dropout(0.1)); TransformerDecoder
/// transformer_decoder(TransformerDecoderOptions(decoder_layer,
/// 6).norm(LayerNorm(LayerNormOptions({2}))));
/// ```
class TORCH_API TransformerDecoderImpl
    : public Cloneable<TransformerDecoderImpl> {
 public:
  // (Incomplete, continuation of the code not provided by user request)


注释：
/// 6).norm(LayerNorm(LayerNormOptions({2})))); const auto memory =
/// torch::rand({10, 32, 512}); const auto tgt = torch::rand({20, 32, 512});
/// auto out = transformer_decoder(tgt, memory);
/// ```
class TORCH_API TransformerDecoderImpl
    : public Cloneable<TransformerDecoderImpl> {
 public:
  /// 构造函数，初始化 TransformerDecoderImpl 对象。
  TransformerDecoderImpl(
      TransformerDecoderLayer decoder_layer,
      int64_t num_layers)
      : TransformerDecoderImpl(
            TransformerDecoderOptions(decoder_layer, num_layers)) {}

  /// 通过选项对象初始化 TransformerDecoderImpl 对象。
  explicit TransformerDecoderImpl(TransformerDecoderOptions options_);

  /// 重置模型状态。
  void reset() override;

  /// 重置模型参数。
  void reset_parameters();

  /// 通过解码器层依次处理输入（及掩码）。
  /// Args:
  ///       tgt: 解码器层的输入序列（必需）。
  ///       memory: 编码器最后一层的输出序列（必需）。
  ///       tgt_mask: 目标序列的掩码（可选）。
  ///       memory_mask: 编码器输出序列的掩码（可选）。
  ///       tgt_key_padding_mask: 每批次目标键的掩码（可选）。
  ///       memory_key_padding_mask: 每批次编码器键的掩码（可选）。
  Tensor forward(
      const Tensor& tgt,
      const Tensor& memory,
      const Tensor& tgt_mask = {},
      const Tensor& memory_mask = {},
      const Tensor& tgt_key_padding_mask = {},
      const Tensor& memory_key_padding_mask = {});

  /// 用于配置该模块的选项。
  TransformerDecoderOptions options;

  /// 克隆的解码器层的模块列表。
  ModuleList layers{nullptr};

  /// 可选的层归一化模块。
  AnyModule norm;

 protected:
  /// 默认参数的前向传播函数。
  FORWARD_HAS_DEFAULT_ARGS(
      {2, AnyValue(Tensor())},
      {3, AnyValue(Tensor())},
      {4, AnyValue(Tensor())},
      {5, AnyValue(Tensor())})
};

/// 用于持有 `TransformerDecoderImpl` 的 `ModuleHolder` 子类。
/// 参阅 `TransformerDecoderImpl` 类的文档以了解它提供的方法，以及使用 `torch::nn::TransformerDecoderOptions` 与 `TransformerDecoder` 的示例。
/// 参阅 `ModuleHolder` 的文档以了解 PyTorch 的模块存储语义。
TORCH_MODULE(TransformerDecoder);

} // namespace nn
} // namespace torch
```