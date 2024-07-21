# `.\pytorch\torch\csrc\api\src\nn\options\activation.cpp`

```
// 包含 torch 库中的激活函数选项头文件
#include <torch/nn/options/activation.h>

// 定义 torch 命名空间内的 nn 命名空间
namespace torch {
namespace nn {

// SELU 激活函数的选项类构造函数，设置是否原地操作的标志
SELUOptions::SELUOptions(bool inplace) : inplace_(inplace) {}

// GLU 激活函数的选项类构造函数，设置切分维度
GLUOptions::GLUOptions(int64_t dim) : dim_(dim) {}

// Hardshrink 激活函数的选项类构造函数，设置阈值 lambda
HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

// Softmax 操作的选项类构造函数，设置操作维度
SoftmaxOptions::SoftmaxOptions(int64_t dim) : dim_(dim) {}

// Softmin 操作的选项类构造函数，设置操作维度
SoftminOptions::SoftminOptions(int64_t dim) : dim_(dim) {}

// LogSoftmax 操作的选项类构造函数，设置操作维度
LogSoftmaxOptions::LogSoftmaxOptions(int64_t dim) : dim_(dim) {}

// ReLU 激活函数的选项类构造函数，设置是否原地操作的标志
ReLUOptions::ReLUOptions(bool inplace) : inplace_(inplace) {}

// ReLU6 激活函数的选项类构造函数，设置是否原地操作的标志
ReLU6Options::ReLU6Options(bool inplace) : inplace_(inplace) {}

// Softshrink 激活函数的选项类构造函数，设置阈值 lambda
SoftshrinkOptions::SoftshrinkOptions(double lambda) : lambda_(lambda) {}

// 多头注意力机制的选项类构造函数，设置嵌入维度和注意力头数
MultiheadAttentionOptions::MultiheadAttentionOptions(
    int64_t embed_dim,
    int64_t num_heads)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      kdim_(embed_dim),
      vdim_(embed_dim) {}

// torch.nn.functional 命名空间内的 Softmax 函数操作选项类构造函数，设置操作维度
namespace functional {

SoftmaxFuncOptions::SoftmaxFuncOptions(int64_t dim) : dim_(dim) {}

// torch.nn.functional 命名空间内的 Softmin 函数操作选项类构造函数，设置操作维度
SoftminFuncOptions::SoftminFuncOptions(int64_t dim) : dim_(dim) {}

// torch.nn.functional 命名空间内的 LogSoftmax 函数操作选项类构造函数，设置操作维度
LogSoftmaxFuncOptions::LogSoftmaxFuncOptions(int64_t dim) : dim_(dim) {}

// torch.nn.functional 命名空间内的多头注意力机制前向函数操作选项类构造函数，设置各种权重和偏置
MultiheadAttentionForwardFuncOptions::MultiheadAttentionForwardFuncOptions(
    int64_t embed_dim_to_check,
    int64_t num_heads,
    Tensor in_proj_weight,
    Tensor in_proj_bias,
    Tensor bias_k,
    Tensor bias_v,
    bool add_zero_attn,
    double dropout_p,
    Tensor out_proj_weight,
    Tensor out_proj_bias)
    : embed_dim_to_check_(embed_dim_to_check),
      num_heads_(num_heads),
      in_proj_weight_(std::move(in_proj_weight)),
      in_proj_bias_(std::move(in_proj_bias)),
      bias_k_(std::move(bias_k)),
      bias_v_(std::move(bias_v)),
      add_zero_attn_(add_zero_attn),
      dropout_p_(dropout_p),
      out_proj_weight_(std::move(out_proj_weight)),
      out_proj_bias_(std::move(out_proj_bias)) {}

} // namespace functional
} // namespace nn
} // namespace torch
```