# `.\pytorch\torch\csrc\api\include\torch\nn\options\rnn.h`

```py
#pragma once

#include <torch/arg.h>  // 包含 torch 库中的参数定义
#include <torch/csrc/Export.h>  // 包含 torch 库中的导出定义
#include <torch/enum.h>  // 包含 torch 库中的枚举类型定义
#include <torch/types.h>  // 包含 torch 库中的类型定义

namespace torch {
namespace nn {

namespace detail {

/// RNN、LSTM 和 GRU 模块的通用选项。
struct TORCH_API RNNOptionsBase {
  typedef std::variant<
      enumtype::kLSTM,  // 支持的 RNN 模型类型：LSTM
      enumtype::kGRU,   // 支持的 RNN 模型类型：GRU
      enumtype::kRNN_TANH,  // 支持的 RNN 模型类型：Tanh 激活的 RNN
      enumtype::kRNN_RELU>  // 支持的 RNN 模型类型：ReLU 激活的 RNN
      rnn_options_base_mode_t;  // RNN 模型类型的变体

  RNNOptionsBase(
      rnn_options_base_mode_t mode,  // RNN 模型的类型
      int64_t input_size,  // 输入序列 `x` 的特征数
      int64_t hidden_size);  // 隐藏状态 `h` 的特征数

  TORCH_ARG(rnn_options_base_mode_t, mode);  // RNN 模型的类型参数
  /// 输入序列 `x` 中单个样本的特征数
  TORCH_ARG(int64_t, input_size);
  /// 隐藏状态 `h` 的特征数
  TORCH_ARG(int64_t, hidden_size);
  /// 要使用的循环层（单元）的数量
  TORCH_ARG(int64_t, num_layers) = 1;  // 默认为 1 层
  /// 是否对所有线性操作添加偏置项
  TORCH_ARG(bool, bias) = true;  // 默认为添加偏置项
  /// 如果为 true，则输入序列应提供为 `(batch, sequence, features)`；如果为 false（默认），则期望布局为 `(sequence, batch, features)`
  TORCH_ARG(bool, batch_first) = false;  // 默认为 false
  /// 如果非零，则对每个 RNN 层的输出添加给定概率的 dropout，最后一层除外
  TORCH_ARG(double, dropout) = 0.0;  // 默认为 0.0，即不使用 dropout
  /// 是否使 RNN 双向
  TORCH_ARG(bool, bidirectional) = false;  // 默认为单向
  /// 单元投影维度。如果为 0，则不添加投影。仅可用于 LSTM
  TORCH_ARG(int64_t, proj_size) = 0;  // 默认为 0，不添加投影
};

} // namespace detail

/// `RNN` 模块的选项。
///
/// 示例：
/// ```
/// RNN model(RNNOptions(128, 64).num_layers(3).dropout(0.2).nonlinearity(torch::kTanh));
/// ```py
/// Struct defining options for the RNN module.
struct TORCH_API RNNOptions {
  /// Type definition for the nonlinearity options, which can be either Tanh or ReLU.
  typedef std::variant<enumtype::kTanh, enumtype::kReLU> nonlinearity_t;

  /// Constructor initializing input size and hidden size.
  RNNOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// Number of recurrent layers. E.g., setting ``num_layers=2``
  /// would mean stacking two RNNs together to form a `stacked RNN`,
  /// with the second RNN taking in outputs of the first RNN and
  /// computing the final results. Default: 1
  TORCH_ARG(int64_t, num_layers) = 1;
  /// The non-linearity to use. Can be either ``torch::kTanh`` or
  /// ``torch::kReLU``. Default: ``torch::kTanh``
  TORCH_ARG(nonlinearity_t, nonlinearity) = torch::kTanh;
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
  /// If ``true``, then the input and output tensors are provided
  /// as `(batch, seq, feature)`. Default: ``false``
  TORCH_ARG(bool, batch_first) = false;
  /// If non-zero, introduces a `Dropout` layer on the outputs of each
  /// RNN layer except the last layer, with dropout probability equal to
  /// `dropout`. Default: 0
  TORCH_ARG(double, dropout) = 0.0;
  /// If ``true``, becomes a bidirectional RNN. Default: ``false``
  TORCH_ARG(bool, bidirectional) = false;
};

/// Struct defining options for the LSTM module.
///
/// Example:
/// ```
/// LSTM model(LSTMOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```py
struct TORCH_API LSTMOptions {
  /// Constructor initializing input size and hidden size.
  LSTMOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// Number of recurrent layers. E.g., setting ``num_layers=2``
  /// would mean stacking two LSTMs together to form a `stacked LSTM`,
  /// with the second LSTM taking in outputs of the first LSTM and
  /// computing the final results. Default: 1
  TORCH_ARG(int64_t, num_layers) = 1;
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
  /// If ``true``, then the input and output tensors are provided
  /// as (batch, seq, feature). Default: ``false``
  TORCH_ARG(bool, batch_first) = false;
  /// If non-zero, introduces a `Dropout` layer on the outputs of each
  /// LSTM layer except the last layer, with dropout probability equal to
  /// `dropout`. Default: 0
  TORCH_ARG(double, dropout) = 0.0;
  /// If ``true``, becomes a bidirectional LSTM. Default: ``false``
  TORCH_ARG(bool, bidirectional) = false;
  /// Cell projection dimension. If 0, projections are not added
  TORCH_ARG(int64_t, proj_size) = 0;
};

/// Struct defining options for the GRU module.
///
/// Example:
/// ```
/// GRU model(GRUOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```py
/// 结构体 `GRUOptions`，定义了 GRU 模型的参数选项
struct TORCH_API GRUOptions {
  /// 构造函数，初始化输入特征大小和隐藏状态特征大小
  GRUOptions(int64_t input_size, int64_t hidden_size);

  /// 输入 `x` 中预期的特征数量
  TORCH_ARG(int64_t, input_size);
  /// 隐藏状态 `h` 中的特征数量
  TORCH_ARG(int64_t, hidden_size);
  /// 循环层的数量。例如，设置 `num_layers=2` 将会堆叠两个 GRU 层，
  /// 形成一个堆叠的 GRU，第二个 GRU 接收第一个 GRU 的输出并计算最终结果。默认值为 1
  TORCH_ARG(int64_t, num_layers) = 1;
  /// 如果为 `false`，则该层不使用偏置权重 `b_ih` 和 `b_hh`。默认为 `true`
  TORCH_ARG(bool, bias) = true;
  /// 如果为 `true`，则输入和输出张量按 (batch, seq, feature) 提供。默认为 `false`
  TORCH_ARG(bool, batch_first) = false;
  /// 如果非零，则在每个 GRU 层（最后一层除外）的输出上引入一个 `Dropout` 层，
  /// 丢弃概率等于 `dropout`。默认为 0
  TORCH_ARG(double, dropout) = 0.0;
  /// 如果为 `true`，则成为一个双向 GRU。默认为 `false`
  TORCH_ARG(bool, bidirectional) = false;
};

namespace detail {

/// RNNCell、LSTMCell 和 GRUCell 模块的通用选项
struct TORCH_API RNNCellOptionsBase {
  /// 构造函数，初始化输入大小、隐藏状态大小、是否使用偏置、块的数量
  RNNCellOptionsBase(
      int64_t input_size,
      int64_t hidden_size,
      bool bias,
      int64_t num_chunks);
  TORCH_ARG(int64_t, input_size);  // 输入大小
  TORCH_ARG(int64_t, hidden_size); // 隐藏状态大小
  TORCH_ARG(bool, bias);           // 是否使用偏置
  TORCH_ARG(int64_t, num_chunks);  // 块的数量
};

} // namespace detail

/// `RNNCell` 模块的选项
///
/// 示例：
/// ```
/// RNNCell model(RNNCellOptions(20, 10).bias(false).nonlinearity(torch::kReLU));
/// ```py
struct TORCH_API RNNCellOptions {
  typedef std::variant<enumtype::kTanh, enumtype::kReLU> nonlinearity_t;

  /// 构造函数，初始化输入大小和隐藏状态大小
  RNNCellOptions(int64_t input_size, int64_t hidden_size);

  /// 输入 `x` 中预期的特征数量
  TORCH_ARG(int64_t, input_size); // 输入大小
  /// 隐藏状态 `h` 中的特征数量
  TORCH_ARG(int64_t, hidden_size); // 隐藏状态大小
  /// 如果为 `false`，则该层不使用偏置权重 `b_ih` 和 `b_hh`。默认为 `true`
  TORCH_ARG(bool, bias) = true;
  /// 要使用的非线性函数。可以是 `torch::kTanh` 或 `torch::kReLU`。默认为 `torch::kTanh`
  TORCH_ARG(nonlinearity_t, nonlinearity) = torch::kTanh;
};

/// `LSTMCell` 模块的选项
///
/// 示例：
/// ```
/// LSTMCell model(LSTMCellOptions(20, 10).bias(false));
/// ```py
struct TORCH_API LSTMCellOptions {
  /// 构造函数，初始化输入大小和隐藏状态大小
  LSTMCellOptions(int64_t input_size, int64_t hidden_size);

  /// 输入 `x` 中预期的特征数量
  TORCH_ARG(int64_t, input_size); // 输入大小
  /// 隐藏状态 `h` 中的特征数量
  TORCH_ARG(int64_t, hidden_size); // 隐藏状态大小
  /// 如果为 `false`，则该层不使用偏置权重 `b_ih` 和 `b_hh`。默认为 `true`
  TORCH_ARG(bool, bias) = true;
};

/// `GRUCell` 模块的选项
///
/// 示例：
/// ```
/// GRUCell model(GRUCellOptions(20, 10).bias(false));
/// ```py
struct TORCH_API GRUCellOptions {
  /// 构造函数，初始化输入大小和隐藏状态大小
  GRUCellOptions(int64_t input_size, int64_t hidden_size);

  /// 输入 `x` 中预期的特征数量
  TORCH_ARG(int64_t, input_size); // 输入大小
  /// 隐藏状态 `h` 中的特征数量
  TORCH_ARG(int64_t, hidden_size); // 隐藏状态大小
  /// 如果为 `false`，则该层不使用偏置权重 `b_ih` 和 `b_hh`。默认为 `true`
  TORCH_ARG(bool, bias) = true;
};
/// 定义了一个名为 `GRUCellOptions` 的结构体，用于配置 GRU 单元的选项
/// ```
struct TORCH_API GRUCellOptions {
  /// 构造函数，初始化 GRU 单元选项
  GRUCellOptions(int64_t input_size, int64_t hidden_size);

  /// 输入 `x` 中预期的特征数量
  TORCH_ARG(int64_t, input_size);
  /// 隐藏状态 `h` 中的特征数量
  TORCH_ARG(int64_t, hidden_size);
  /// 如果为 ``false``，则该层不使用偏置权重 `b_ih` 和 `b_hh`
  /// 默认为 ``true``
  TORCH_ARG(bool, bias) = true;
};

} // namespace nn
} // namespace torch
```