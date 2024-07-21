# `.\pytorch\torch\csrc\api\src\nn\modules\embedding.cpp`

```
// 包含了 Torch 库中的 Embedding 头文件
#include <torch/nn/modules/embedding.h>

// 包含了 Torch 库中的初始化、类型和实用函数的头文件
#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>

// 包含了标准库的头文件
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

// 使用命名空间简化代码，将 torch::nn::functional 命名空间别名为 F
namespace F = torch::nn::functional;

// torch::nn 命名空间，包含了神经网络相关的类和函数
namespace torch {
namespace nn {

// EmbeddingImpl 类的构造函数实现
EmbeddingImpl::EmbeddingImpl(EmbeddingOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset() 函数进行初始化
  reset();
}

// EmbeddingImpl 类的 reset() 函数实现
void EmbeddingImpl::reset() {
  // 如果设置了 padding_idx
  if (options.padding_idx() != c10::nullopt) {
    // 如果 padding_idx 大于 0，检查其是否在 num_embeddings 内
    if (*options.padding_idx() > 0) {
      TORCH_CHECK(
          *options.padding_idx() < options.num_embeddings(),
          "Padding_idx must be within num_embeddings");
    } else if (*options.padding_idx() < 0) {
      // 如果 padding_idx 小于 0，检查其是否在 -num_embeddings 内
      TORCH_CHECK(
          *options.padding_idx() >= -options.num_embeddings(),
          "Padding_idx must be within num_embedding");
      // 将负数的 padding_idx 转换为对应的正数索引
      options.padding_idx(options.num_embeddings() + *options.padding_idx());
    }
  }

  // 如果权重参数 _weight 未定义
  if (!options._weight().defined()) {
    // 创建一个空的权重张量，并注册为模型的参数
    weight = register_parameter(
        "weight",
        torch::empty({options.num_embeddings(), options.embedding_dim()}));
    // 初始化权重参数
    reset_parameters();
  } else {
    // 如果已经定义了 _weight，检查其形状是否和 num_embeddings、embedding_dim 匹配
    TORCH_CHECK(
        options._weight().sizes() ==
            torch::IntArrayRef(
                {options.num_embeddings(), options.embedding_dim()}),
        "Shape of _weight does not match num_embeddings and embedding_dim");
    // 将 _weight 注册为模型的参数
    weight = register_parameter("weight", options._weight());
  }
}

// EmbeddingImpl 类的 reset_parameters() 函数实现
void EmbeddingImpl::reset_parameters() {
  // 使用正态分布初始化权重参数
  torch::nn::init::normal_(weight);
  // 如果设置了 padding_idx，将对应位置的权重置为零
  if (options.padding_idx() != c10::nullopt) {
    torch::NoGradGuard no_grad;
    weight[*options.padding_idx()].fill_(0);
  }
}

// EmbeddingImpl 类的 pretty_print() 函数实现
void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  // 输出 Embedding 的相关信息到流中，用于打印模型信息
  stream << "torch::nn::Embedding(num_embeddings=" << options.num_embeddings()
         << ", embedding_dim=" << options.embedding_dim();
  // 如果设置了 padding_idx，将其信息也输出
  if (options.padding_idx() != c10::nullopt) {
    stream << ", padding_idx=" << *options.padding_idx();
  }
  // 如果设置了 max_norm，将其信息也输出
  if (options.max_norm() != c10::nullopt) {
    stream << ", max_norm=" << *options.max_norm();
  }
  // 如果设置了 norm_type，将其信息也输出
  if (options.norm_type() != 2) {
    stream << ", norm_type=" << options.norm_type();
  }
  // 如果设置了 scale_grad_by_freq，将其信息也输出
  if (options.scale_grad_by_freq()) {
    stream << ", scale_grad_by_freq=" << std::boolalpha
           << options.scale_grad_by_freq();
  }
  // 如果设置了 sparse，将其信息也输出
  if (options.sparse()) {
    stream << ", sparse=" << std::boolalpha << options.sparse();
  }
  // 输出结束符号 ")"
  stream << ")";
}

// EmbeddingImpl 类的 forward() 函数实现
torch::Tensor EmbeddingImpl::forward(const Tensor& input) {
  // 调用 functional 命名空间中的 embedding 函数进行前向传播
  return F::detail::embedding(
      input,
      weight,
      options.padding_idx(),
      options.max_norm(),
      options.norm_type(),
      options.scale_grad_by_freq(),
      options.sparse());
}

// EmbeddingBagImpl 类的构造函数实现
EmbeddingBagImpl::EmbeddingBagImpl(EmbeddingBagOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset() 函数进行初始化
  reset();
}

// EmbeddingBagImpl 类的 reset() 函数实现
void EmbeddingBagImpl::reset() {
  // 如果设置了 padding_idx
  if (options.padding_idx().has_value()) {
    // 获取填充索引，从选项中获取填充索引的数值
    auto padding_idx = options.padding_idx().value();
    // 如果填充索引大于0，则进行以下检查
    if (padding_idx > 0) {
      // 确保填充索引小于词嵌入的数量，否则抛出错误信息
      TORCH_CHECK(
          padding_idx < options.num_embeddings(),
          "Padding_idx must be within num_embeddings");
    } else if (padding_idx < 0) {
      // 如果填充索引小于0，则进行以下检查
      TORCH_CHECK(
          padding_idx >= -options.num_embeddings(),
          "Padding_idx must be within num_embedding");
      // 将填充索引调整为 num_embeddings + padding_idx
      options.padding_idx(options.num_embeddings() + padding_idx);
    }
  }
  // 如果权重参数未定义
  if (!options._weight().defined()) {
    // 创建一个空的权重张量，形状为 num_embeddings x embedding_dim
    weight = register_parameter(
        "weight",
        torch::empty({options.num_embeddings(), options.embedding_dim()}));
    // 重置权重的初始值
    reset_parameters();
  } else {
    // 如果权重参数已定义，则进行以下检查
    TORCH_CHECK(
        // 确保权重参数的形状与 num_embeddings 和 embedding_dim 匹配
        options._weight().sizes() ==
            torch::IntArrayRef(
                {options.num_embeddings(), options.embedding_dim()}),
        "Shape of weight does not match num_embeddings and embedding_dim");
    // 注册权重参数，使用给定的权重张量
    weight = register_parameter("weight", options._weight());
  }
// 重置 EmbeddingBag 的参数。如果指定了填充索引，将填充索引对应的权重张量清零。
void EmbeddingBagImpl::reset_parameters() {
  // 如果指定了填充索引
  if (options.padding_idx().has_value()) {
    // 使用 NoGradGuard 禁止梯度计算
    torch::NoGradGuard no_grad;
    // 将填充索引对应的权重张量填充为零
    weight[options.padding_idx().value()].fill_(0);
  }
  // 对权重张量进行正态分布初始化
  torch::nn::init::normal_(weight);
}

// 前向传播函数，实现 EmbeddingBag 的功能
torch::Tensor EmbeddingBagImpl::forward(
    const Tensor& input,
    const Tensor& offsets,
    const Tensor& per_sample_weights) {
  // 调用 F::detail::embedding_bag 函数进行 EmbeddingBag 操作
  return F::detail::embedding_bag(
      input,
      weight,
      offsets,
      options.max_norm(),
      options.norm_type(),
      options.scale_grad_by_freq(),
      options.mode(),
      options.sparse(),
      per_sample_weights,
      options.include_last_offset(),
      options.padding_idx());
}

// 打印 EmbeddingBag 的信息到输出流中
void EmbeddingBagImpl::pretty_print(std::ostream& stream) const {
  // 输出 EmbeddingBag 的基本信息：num_embeddings 和 embedding_dim
  stream << "torch::nn::EmbeddingBag(num_embeddings="
         << options.num_embeddings()
         << ", embedding_dim=" << options.embedding_dim();
  // 如果设置了 max_norm 参数，则输出 max_norm 的值
  if (options.max_norm() != c10::nullopt) {
    stream << ", max_norm=" << *options.max_norm();
  }
  // 如果设置了 norm_type 参数且不等于默认值 2，则输出 norm_type 的值
  if (options.norm_type() != 2) {
    stream << ", norm_type=" << options.norm_type();
  }
  // 如果设置了 scale_grad_by_freq 参数，则输出 scale_grad_by_freq 的值
  if (options.scale_grad_by_freq()) {
    stream << ", scale_grad_by_freq=" << std::boolalpha
           << options.scale_grad_by_freq();
  }
  // 如果设置了 sparse 参数，则输出 sparse 的值
  if (options.sparse()) {
    stream << ", sparse=" << std::boolalpha << options.sparse();
  }
  // 如果设置了 mode 参数且不是默认值 kMean，则输出 mode 的名称
  if (!std::get_if<enumtype::kMean>(&options.mode())) {
    stream << ", mode=" << torch::enumtype::get_enum_name(options.mode());
  }
  // 如果设置了 include_last_offset 参数，则输出 include_last_offset 的值
  if (options.include_last_offset()) {
    stream << ", include_last_offset=" << std::boolalpha
           << options.include_last_offset();
  }
  // 如果指定了填充索引，则输出 padding_idx 的值
  if (options.padding_idx().has_value()) {
    stream << ", padding_idx=" << options.padding_idx().value();
  }
  // 输出完成，关闭流
  stream << ")";
}
} // namespace nn
} // namespace torch
```