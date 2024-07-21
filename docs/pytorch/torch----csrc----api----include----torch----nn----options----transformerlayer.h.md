# `.\pytorch\torch\csrc\api\include\torch\nn\options\transformerlayer.h`

```
#pragma once

#include <torch/arg.h>  // 包含了 torch 库中的参数处理模块
#include <torch/csrc/Export.h>  // 包含了 torch 库中的导出相关定义
#include <torch/enum.h>  // 包含了 torch 库中的枚举类型定义
#include <torch/types.h>  // 包含了 torch 库中的数据类型定义

namespace torch {
namespace nn {

using activation_t = std::variant<
    enumtype::kReLU,
    enumtype::kGELU,
    std::function<Tensor(const Tensor&)>>;  // 定义了 activation_t 类型，可以是 kReLU 枚举、kGELU 枚举或者一个接受 Tensor 并返回 Tensor 的函数

/// Options for the `TransformerEncoderLayer`
///
/// Example:
/// ```
/// auto options = TransformerEncoderLayer(512, 8).dropout(0.2);
/// ```
struct TORCH_API TransformerEncoderLayerOptions {
  /* implicit */ TransformerEncoderLayerOptions(int64_t d_model, int64_t nhead);  // 构造函数声明，参数为输入特征数 d_model 和头数 nhead

  /// the number of expected features in the input
  TORCH_ARG(int64_t, d_model);  // 输入特征数 d_model

  /// the number of heads in the multiheadattention models
  TORCH_ARG(int64_t, nhead);  // 头数 nhead

  /// the dimension of the feedforward network model, default is 2048
  TORCH_ARG(int64_t, dim_feedforward) = 2048;  // 前馈网络模型的维度，默认为 2048

  /// the dropout value, default is 0.1
  TORCH_ARG(double, dropout) = 0.1;  // dropout 的值，默认为 0.1

  /// the activation function of intermediate layer, can be ``torch::kReLU``,
  /// ``torch::GELU``, or a unary callable. Default: ``torch::kReLU``
  TORCH_ARG(activation_t, activation) = torch::kReLU;  // 中间层的激活函数，可以是 kReLU、kGELU 或者一个一元可调用函数，默认为 kReLU
};

// ============================================================================

/// Options for the `TransformerDecoderLayer` module.
///
/// Example:
/// ```
/// TransformerDecoderLayer model(TransformerDecoderLayerOptions(512,
/// 8).dropout(0.2));
/// ```
struct TORCH_API TransformerDecoderLayerOptions {
  TransformerDecoderLayerOptions(int64_t d_model, int64_t nhead);  // 构造函数声明，参数为输入特征数 d_model 和头数 nhead

  /// number of expected features in the input
  TORCH_ARG(int64_t, d_model);  // 输入特征数 d_model

  /// number of heads in the multiheadattention models
  TORCH_ARG(int64_t, nhead);  // 头数 nhead

  /// dimension of the feedforward network model. Default: 2048
  TORCH_ARG(int64_t, dim_feedforward) = 2048;  // 前馈网络模型的维度，默认为 2048

  /// dropout value. Default: 1
  TORCH_ARG(double, dropout) = 0.1;  // dropout 的值，默认为 0.1

  /// activation function of intermediate layer, can be ``torch::kGELU``,
  /// ``torch::kReLU``, or a unary callable. Default: ``torch::kReLU``
  TORCH_ARG(activation_t, activation) = torch::kReLU;  // 中间层的激活函数，可以是 kGELU、kReLU 或者一个一元可调用函数，默认为 kReLU
};

} // namespace nn
} // namespace torch
```