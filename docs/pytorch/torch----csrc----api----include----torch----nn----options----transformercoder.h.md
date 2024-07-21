# `.\pytorch\torch\csrc\api\include\torch\nn\options\transformercoder.h`

```py
#pragma once
// 一次性引用 torch 库中所需的头文件
#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

// 引用定义在 torch.nn 命名空间下的各种模块
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/transformerlayer.h>

// 定义 torch 命名空间下的 nn 命名空间
namespace torch {
namespace nn {

/// Options for the `TransformerEncoder` module.
///
/// Example:
/// ```
/// TransformerEncoderLayer encoderLayer(TransformerEncoderLayerOptions(512,
/// 8).dropout(0.1)); auto options = TransformerEncoderOptions(encoderLayer,
/// 6).norm(LayerNorm(LayerNormOptions({2})));
/// ```py
struct TORCH_API TransformerEncoderOptions {
  // 构造函数，浅复制 encoder_layer，保留其中所有数据
  TransformerEncoderOptions(
      TransformerEncoderLayer encoder_layer,
      int64_t num_layers);
  // 构造函数，基于传入的 encoder_layer_options 创建新的 TransformerEncoderLayer 对象
  TransformerEncoderOptions(
      const TransformerEncoderLayerOptions& encoder_layer_options,
      int64_t num_layers);

  /// transformer 编码器层
  TORCH_ARG(TransformerEncoderLayer, encoder_layer) = nullptr;

  /// 编码器层数
  TORCH_ARG(int64_t, num_layers);

  /// 标准化模块
  TORCH_ARG(AnyModule, norm);
};

/// Options for the `TransformerDecoder` module.
///
/// Example:
/// ```
/// TransformerDecoderLayer decoder_layer(TransformerDecoderLayerOptions(512,
/// 8).dropout(0.1)); auto options = TransformerDecoderOptions(decoder_layer,
/// 6)norm(LayerNorm(LayerNormOptions({2}))); TransformerDecoder
/// transformer_decoder(options);
/// ```py
struct TORCH_API TransformerDecoderOptions {
  // 构造函数，保留传入的 decoder_layer 的引用，保留其中所有数据
  TransformerDecoderOptions(
      TransformerDecoderLayer decoder_layer,
      int64_t num_layers);
  // 构造函数，基于传入的 decoder_layer_options 创建新的 TransformerDecoderLayer 对象
  TransformerDecoderOptions(
      const TransformerDecoderLayerOptions& decoder_layer_options,
      int64_t num_layers);

  /// 要克隆的解码器层
  TORCH_ARG(TransformerDecoderLayer, decoder_layer) = nullptr;

  /// 解码器层数
  TORCH_ARG(int64_t, num_layers);

  /// 标准化模块
  TORCH_ARG(AnyModule, norm);
};

} // namespace nn
} // namespace torch
```