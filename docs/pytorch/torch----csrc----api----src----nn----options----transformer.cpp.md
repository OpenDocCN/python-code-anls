# `.\pytorch\torch\csrc\api\src\nn\options\transformer.cpp`

```
// 引入 Transformer 编码器选项头文件
#include <torch/nn/options/transformer.h>
// 引入 Transformer 编码器层选项头文件
#include <torch/nn/options/transformercoder.h>
// 引入 Transformer 编码器层头文件
#include <torch/nn/options/transformerlayer.h>

// 定义 torch 命名空间下的 nn 命名空间
namespace torch {
namespace nn {

// TransformerEncoderLayerOptions 类的构造函数，初始化 d_model 和 nhead 成员变量
TransformerEncoderLayerOptions::TransformerEncoderLayerOptions(
    int64_t d_model,
    int64_t nhead)
    : d_model_(d_model), nhead_(nhead) {}

// TransformerDecoderLayerOptions 类的构造函数，初始化 d_model 和 nhead 成员变量
TransformerDecoderLayerOptions::TransformerDecoderLayerOptions(
    int64_t d_model,
    int64_t nhead)
    : d_model_(d_model), nhead_(nhead) {}

// TransformerEncoderOptions 类的构造函数，初始化 encoder_layer_ 和 num_layers 成员变量
TransformerEncoderOptions::TransformerEncoderOptions(
    TransformerEncoderLayer encoder_layer,
    int64_t num_layers)
    : encoder_layer_(std::move(encoder_layer)), num_layers_(num_layers) {}

// TransformerEncoderOptions 类的构造函数，初始化 encoder_layer_ 和 num_layers 成员变量
// 使用 TransformerEncoderLayerOptions 作为参数
TransformerEncoderOptions::TransformerEncoderOptions(
    const TransformerEncoderLayerOptions& encoder_layer_options,
    int64_t num_layers)
    : encoder_layer_(encoder_layer_options), num_layers_(num_layers) {}

// TransformerDecoderOptions 类的构造函数，初始化 decoder_layer_ 和 num_layers 成员变量
TransformerDecoderOptions::TransformerDecoderOptions(
    TransformerDecoderLayer decoder_layer,
    int64_t num_layers)
    : decoder_layer_(std::move(decoder_layer)), num_layers_(num_layers) {}

// TransformerDecoderOptions 类的构造函数，初始化 decoder_layer_ 和 num_layers 成员变量
// 使用 TransformerDecoderLayerOptions 作为参数
TransformerDecoderOptions::TransformerDecoderOptions(
    const TransformerDecoderLayerOptions& decoder_layer_options,
    int64_t num_layers)
    : decoder_layer_(decoder_layer_options), num_layers_(num_layers) {}

// TransformerOptions 类的构造函数，初始化 d_model_ 和 nhead_ 成员变量
TransformerOptions::TransformerOptions(int64_t d_model, int64_t nhead)
    : d_model_(d_model), nhead_(nhead) {}

// TransformerOptions 类的构造函数，初始化 d_model_、nhead_、num_encoder_layers_ 和 num_decoder_layers_ 成员变量
TransformerOptions::TransformerOptions(
    int64_t d_model,
    int64_t nhead,
    int64_t num_encoder_layers,
    int64_t num_decoder_layers)
    : d_model_(d_model),
      nhead_(nhead),
      num_encoder_layers_(num_encoder_layers),
      num_decoder_layers_(num_decoder_layers) {}

} // namespace nn
} // namespace torch
```