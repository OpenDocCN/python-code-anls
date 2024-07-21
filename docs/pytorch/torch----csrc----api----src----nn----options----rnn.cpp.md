# `.\pytorch\torch\csrc\api\src\nn\options\rnn.cpp`

```
// 包含 Torch 库中 RNN 相关选项的头文件
#include <torch/nn/options/rnn.h>

// 定义 torch 命名空间内的 nn 命名空间
namespace torch {
namespace nn {

// 定义 nn 命名空间内的 detail 命名空间
namespace detail {

// RNNOptionsBase 类的构造函数实现
RNNOptionsBase::RNNOptionsBase(
    rnn_options_base_mode_t mode,       // RNN 模型的模式
    int64_t input_size,                 // 输入大小
    int64_t hidden_size)                // 隐藏状态大小
    : mode_(mode), input_size_(input_size), hidden_size_(hidden_size) {}

} // namespace detail

// RNNOptions 类的构造函数实现
RNNOptions::RNNOptions(int64_t input_size,  // 输入大小
                       int64_t hidden_size) // 隐藏状态大小
    : input_size_(input_size), hidden_size_(hidden_size) {}

// LSTMOptions 类的构造函数实现
LSTMOptions::LSTMOptions(int64_t input_size,   // 输入大小
                         int64_t hidden_size)  // 隐藏状态大小
    : input_size_(input_size), hidden_size_(hidden_size) {}

// GRUOptions 类的构造函数实现
GRUOptions::GRUOptions(int64_t input_size,    // 输入大小
                       int64_t hidden_size)   // 隐藏状态大小
    : input_size_(input_size), hidden_size_(hidden_size) {}

namespace detail {

// RNNCellOptionsBase 类的构造函数实现
RNNCellOptionsBase::RNNCellOptionsBase(
    int64_t input_size,     // 输入大小
    int64_t hidden_size,    // 隐藏状态大小
    bool bias,              // 是否使用偏置
    int64_t num_chunks)     // 分块数目
    : input_size_(input_size),
      hidden_size_(hidden_size),
      bias_(bias),
      num_chunks_(num_chunks) {}

} // namespace detail

// RNNCellOptions 类的构造函数实现
RNNCellOptions::RNNCellOptions(int64_t input_size,    // 输入大小
                               int64_t hidden_size)   // 隐藏状态大小
    : input_size_(input_size), hidden_size_(hidden_size) {}

// LSTMCellOptions 类的构造函数实现
LSTMCellOptions::LSTMCellOptions(int64_t input_size,   // 输入大小
                                 int64_t hidden_size)  // 隐藏状态大小
    : input_size_(input_size), hidden_size_(hidden_size) {}

// GRUCellOptions 类的构造函数实现
GRUCellOptions::GRUCellOptions(int64_t input_size,    // 输入大小
                               int64_t hidden_size)   // 隐藏状态大小
    : input_size_(input_size), hidden_size_(hidden_size) {}

} // namespace nn
} // namespace torch
```