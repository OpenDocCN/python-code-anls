# `.\pytorch\torch\csrc\api\src\nn\modules\transformer.cpp`

```
// 引入所需的头文件
#include <c10/util/irange.h>
#include <torch/nn/init.h>
#include <torch/nn/modules/transformer.h>
#include <torch/nn/modules/transformercoder.h>
#include <torch/nn/modules/transformerlayer.h>
#include <limits>

namespace F = torch::nn::functional;  // 命名空间别名

namespace torch {
namespace nn {

// ========================TransformerEncoderLayerImpl=========================

// TransformerEncoderLayerImpl 构造函数，接受 TransformerEncoderLayerOptions 作为参数
TransformerEncoderLayerImpl::TransformerEncoderLayerImpl(
    TransformerEncoderLayerOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();  // 调用 reset() 初始化模型
}

// reset() 方法用于初始化模型，调用 register_module() 注册各个模块
void TransformerEncoderLayerImpl::reset() {
  // NOTE: reset() is for initializing the model only, calling reset() after the
  // model is created will throw exceptionss. Call reset_parameter() if the
  // created model needs a reset

  // 注册多头注意力机制 self_attn
  self_attn = this->register_module(
      "self_attn",
      MultiheadAttention(
          MultiheadAttentionOptions(options.d_model(), options.nhead())
              .dropout(options.dropout())));

  // 注册第一个线性层 linear1
  linear1 = this->register_module(
      "linear1", Linear(options.d_model(), options.dim_feedforward()));
  dropout = this->register_module("dropout", Dropout(options.dropout()));  // 注册 dropout
  // 注册第二个线性层 linear2
  linear2 = this->register_module(
      "linear2", Linear(options.dim_feedforward(), options.d_model()));

  // 注册 LayerNorm 层 norm1 和 norm2
  norm1 = this->register_module(
      "norm1", LayerNorm(LayerNormOptions({options.d_model()})));
  norm2 = this->register_module(
      "norm2", LayerNorm(LayerNormOptions({options.d_model()})));

  // 注册 dropout1 和 dropout2
  dropout1 = this->register_module("dropout1", Dropout(options.dropout()));
  dropout2 = this->register_module("dropout2", Dropout(options.dropout()));
}

// reset_parameters() 方法用于重置模型的参数
void TransformerEncoderLayerImpl::reset_parameters() {
  // TODO xinyu: standardrize reset_parameters virtual funcs
  self_attn->_reset_parameters();  // 重置 self_attn 的参数

  linear1->reset_parameters();  // 重置 linear1 的参数
  // dropout->reset_parameters();  // dropout 的参数重置，已注释掉

  linear2->reset_parameters();  // 重置 linear2 的参数

  norm1->reset_parameters();  // 重置 norm1 的参数
  norm2->reset_parameters();  // 重置 norm2 的参数

  // dropout1->reset_parameters();  // dropout1 的参数重置，已注释掉
  // dropout2->reset_parameters();  // dropout2 的参数重置，已注释掉
}

// forward 方法定义了前向传播过程
Tensor TransformerEncoderLayerImpl::forward(
    const Tensor& src,
    const Tensor& src_mask,
    const Tensor& src_key_padding_mask) {
  // multihead attention
  // 使用 self_attn 进行多头注意力机制计算
  Tensor src2 = std::get<0>(self_attn(
      src, src, src, src_key_padding_mask, /*need_weights=*/true, src_mask));
  
  // add & norm 步骤，将注意力机制的输出加和并进行层归一化
  Tensor ret = norm1(src + dropout1(src2));

  // feedforward 步骤，根据激活函数选项选择不同的激活函数计算
  if (std::holds_alternative<enumtype::kGELU>(options.activation())) {
    src2 = linear2(dropout(F::gelu(linear1(ret))));
  } else if (std::holds_alternative<enumtype::kReLU>(options.activation())) {
    src2 = linear2(dropout(F::relu(linear1(ret))));
  } else if (std::holds_alternative<std::function<Tensor(const Tensor&)>>(
                 options.activation())) {
    auto callable_activation =
        *std::get_if<std::function<Tensor(const Tensor&)>>(
            &options.activation());
    src2 = linear2(dropout(callable_activation(linear1(ret))));
  } else {
    // Handle unknown activation function
    // 处理未知的激活函数选项
    // Normally, an error message or default behavior should be implemented here
  }
    TORCH_CHECK(false, "activation should be kGELU, kReLU, or a callable");


    // 使用 TORCH_CHECK 宏来检查条件，如果条件为 false，则抛出错误信息
    TORCH_CHECK(false, "activation should be kGELU, kReLU, or a callable");



  // add & norm
  return norm2(ret + dropout2(src2));


  // 对 ret 和 dropout2(src2) 的结果进行加法操作，并将结果传递给 norm2 进行归一化处理
  // 然后返回 norm2 处理后的结果
  return norm2(ret + dropout2(src2));
// ========================TransformerDecoderLayerImpl=========================

// 构造函数，初始化 TransformerDecoderLayerImpl 对象
TransformerDecoderLayerImpl::TransformerDecoderLayerImpl(
    TransformerDecoderLayerOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset() 进行初始化
  reset();
}

// reset() 方法用于初始化模型，在模型创建后调用将会抛出异常，如果需要重置已创建的模型，应调用 reset_parameters()
void TransformerDecoderLayerImpl::reset() {
  // NOTE: reset() is for initializing the model only, calling reset() after the
  // model is created will cause throwing exceptions. Call reset_parameter() if
  // the created model needs a reset.

  // 初始化自注意力机制（self attention）
  self_attn = this->register_module(
      "self_attn",
      MultiheadAttention(
          MultiheadAttentionOptions(options.d_model(), options.nhead())
              .dropout(options.dropout())));

  // 初始化多头注意力机制（multihead attention）
  multihead_attn = this->register_module(
      "multihead_attn",
      MultiheadAttention(
          MultiheadAttentionOptions(options.d_model(), options.nhead())
              .dropout(options.dropout())));

  // 初始化前向传播的第一个线性层（Feed forward first linear layer）
  linear1 = this->register_module(
      "linear1", Linear(options.d_model(), options.dim_feedforward()));
  // 初始化前向传播的 dropout 层
  dropout = this->register_module("dropout", Dropout(options.dropout()));
  // 初始化前向传播的第二个线性层（Feed forward second linear layer）
  linear2 = this->register_module(
      "linear2", Linear(options.dim_feedforward(), options.d_model()));

  // 初始化自注意力后的归一化层（Normalization, post self attention）
  norm1 = this->register_module(
      "norm1", LayerNorm(LayerNormOptions({options.d_model()})));
  // 初始化多头注意力后的归一化层（Normalization, post multi-headed attention）
  norm2 = this->register_module(
      "norm2", LayerNorm(LayerNormOptions({options.d_model()})));
  // 初始化前向传播后的归一化层（Normalization, post feed forward）
  norm3 = this->register_module(
      "norm3", LayerNorm(LayerNormOptions({options.d_model()})));

  // 初始化自注意力后的 dropout 层（Dropout, post self attention）
  dropout1 = this->register_module("dropout1", Dropout(options.dropout()));
  // 初始化多头注意力后的 dropout 层（Dropout, post multi-headed attention）
  dropout2 = this->register_module("dropout2", Dropout(options.dropout()));
  // 初始化前向传播后的 dropout 层（Dropout, post feed forward）
  dropout3 = this->register_module("dropout3", Dropout(options.dropout()));
}

// reset_parameters() 方法用于重置模型参数
void TransformerDecoderLayerImpl::reset_parameters() {
  // TODO xinyu: standardrize reset_parameters virtual funcs
  // 调用各子模块的 reset_parameters() 方法以重置其参数
  self_attn->_reset_parameters();
  multihead_attn->_reset_parameters();

  linear1->reset_parameters();
  // dropout->reset_paramteres();
  linear2->reset_parameters();

  norm1->reset_parameters();
  norm2->reset_parameters();
  norm3->reset_parameters();
  // dropout1->reset_parameters();
  // dropout2->reset_parameters();
  // dropout3->reset_paramteres();
}

/// 将输入（和掩码）通过解码器层进行前向传播处理。
Tensor TransformerDecoderLayerImpl::forward(
    Tensor tgt,
    const Tensor& memory,
    const Tensor& tgt_mask,
    const Tensor& memory_mask,
    // 对目标序列进行自注意力机制计算
    Tensor tgt2 = std::get<0>(self_attn(
        tgt, // 查询（query）
        tgt, // 键（key）
        tgt, // 值（value）
        tgt_key_padding_mask, // 目标序列的填充遮罩
        false, // 不需要计算注意力权重
        tgt_mask) // 注意力机制遮罩（attn_mask）
    );
    
    // 应用残差连接和 dropout 到目标序列
    tgt = tgt + dropout1(tgt2);
    tgt = norm1(tgt);
    
    // 对目标序列和记忆序列进行多头注意力机制计算
    tgt2 = std::get<0>(multihead_attn(
        tgt, // 查询（query）
        memory, // 键（key，即记忆序列）
        memory, // 值（value，即记忆序列）
        memory_key_padding_mask, // 记忆序列的填充遮罩
        false, // 不需要计算注意力权重
        memory_mask) // 注意力机制遮罩（attn_mask）
    );
    
    // 再次应用残差连接和 dropout 到目标序列
    tgt = tgt + dropout2(tgt2);
    tgt = norm2(tgt);
    
    // 前馈神经网络部分：线性变换、激活函数、残差连接和 dropout
    tgt2 = linear2(dropout(activation(linear1(tgt))));
    tgt = tgt + dropout3(tgt2);
    tgt = norm3(tgt);
    
    // 返回处理后的目标序列
    return tgt;
}

// TransformerDecoderLayerImpl 类的 activation 方法，根据 options 中的激活函数类型对输入进行激活处理
Tensor TransformerDecoderLayerImpl::activation(const Tensor& input) {
  // 激活函数为 GELU，则调用 F::gelu 函数
  if (std::holds_alternative<enumtype::kGELU>(options.activation())) {
    return F::gelu(input);
  } 
  // 激活函数为 ReLU，则调用 F::relu 函数
  else if (std::holds_alternative<enumtype::kReLU>(options.activation())) {
    return F::relu(input);
  } 
  // 如果激活函数为用户自定义的 callable 对象，则调用该对象对输入进行处理
  else if (std::holds_alternative<std::function<Tensor(const Tensor&)>>(
                 options.activation())) {
    auto callable_activation =
        *std::get_if<std::function<Tensor(const Tensor&)>>(
            &options.activation());
    return callable_activation(input);
  } 
  // 如果激活函数类型未知，则抛出错误信息
  else {
    TORCH_CHECK(false, "activation should be kGELU, kReLU, or a callable");
  }
}

// ========================TransformerEncoderImpl=========================
// TransformerEncoderImpl 类的构造函数，初始化 options，并调用 reset 方法
TransformerEncoderImpl::TransformerEncoderImpl(
    TransformerEncoderOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

// TransformerEncoderImpl 类的 reset 方法，初始化 layers 和 norm
void TransformerEncoderImpl::reset() {
  layers = this->register_module("layers", ModuleList());
  // 根据 options 中指定的层数，复制 encoder_layer 到 layers 中
  for (const auto i : c10::irange(options.num_layers())) {
    (void)i; // 抑制未使用变量警告
    layers->push_back(options.encoder_layer()->clone());
  }

  // 如果 options 中的 norm 不为空，则克隆并注册为模块
  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

// TransformerEncoderImpl 类的 reset_parameters 方法，重置每个 encoder layer 的参数，并更新 norm
void TransformerEncoderImpl::reset_parameters() {
  TORCH_CHECK(
      layers->size() == static_cast<size_t>(options.num_layers()),
      "TransformerEncoder should have",
      options.num_layers(),
      " encoder layers, but got ",
      layers->size());

  size_t num_layers = layers->size();
  // 遍历每个 encoder layer，调用其 reset_parameters 方法
  for (const auto i : c10::irange(num_layers)) {
    layers->at<TransformerEncoderLayerImpl>(i).reset_parameters();
  }
  // 替换 norm 模块或清空，以便在重置参数时可以添加或删除 normalization 模块
  if (!norm.is_empty()) {
    this->unregister_module("norm");
    norm = AnyModule();
  }
  // 如果 options 中的 norm 不为空，则更新 norm
  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

// TransformerEncoderImpl 类的 forward 方法，执行整个编码器的前向传播过程
Tensor TransformerEncoderImpl::forward(
    const Tensor& src,
    const Tensor& src_mask,
    const Tensor& src_key_padding_mask) {
  size_t num_layers = layers->size();
  Tensor output;
  if (num_layers > 0) {
    // 对第一个 encoder layer 执行前向传播
    output = layers->at<TransformerEncoderLayerImpl>(0).forward(
        src, src_mask, src_key_padding_mask);
  }
  // 对剩余的 encoder layers 执行前向传播
  for (const auto i : c10::irange(1, num_layers)) {
    output = layers->at<TransformerEncoderLayerImpl>(i).forward(
        output, src_mask, src_key_padding_mask);
  }

  // 如果 norm 不为空，则对输出进行归一化处理
  if (!norm.is_empty()) {
    output = norm.forward<Tensor>(num_layers == 0 ? src : output);
  }
  return output;
}

// ========================TransformerDecoderImpl=========================
// TransformerDecoderImpl 类的构造函数，初始化 options
TransformerDecoderImpl::TransformerDecoderImpl(
    TransformerDecoderOptions options_)
    : options(std::move(options_)) {
  // 使用移动语义将 options_ 移动到成员变量 options 中
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 执行 reset() 函数来初始化对象的状态
  reset();
}

// 重置 TransformerDecoderImpl 对象的状态
void TransformerDecoderImpl::reset() {
  // 初始化 layers 为一个空的 ModuleList
  layers = this->register_module("layers", ModuleList());
  // 遍历每一层解码器，将复制的解码器层对象添加到 layers 中
  for (const auto i : c10::irange(options.num_layers())) {
    (void)i; // 抑制未使用变量警告
    layers->push_back(options.decoder_layer()->clone());
  }

  // 如果规范化模块不为空，将其克隆并注册到当前对象中
  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

// 重置 TransformerDecoderImpl 对象的参数
void TransformerDecoderImpl::reset_parameters() {
  // 检查 layers 的大小与指定的解码器层数是否相等
  TORCH_CHECK(
      layers->size() == static_cast<size_t>(options.num_layers()),
      "TransformerDecoder should have",
      options.num_layers(),
      " decoder layers, but got ",
      layers->size());

  // 获取 layers 的层数
  size_t num_layers = layers->size();
  // 遍历每一层解码器，重置其参数
  for (const auto i : c10::irange(num_layers)) {
    layers->at<TransformerDecoderLayerImpl>(i).reset_parameters();
  }

  // a. 没有办法确定 AnyModule 中的模块是否有 reset_parameters 方法，因此替换而不是 b. 允许用户在重置参数时添加或删除规范化模块
  // 如果规范化模块不为空，取消注册当前的规范化模块
  if (!norm.is_empty()) {
    this->unregister_module("norm");
    // 将 norm 置为空
    norm = AnyModule();
  }
  // 如果配置中的规范化模块不为空，克隆并注册它到当前对象中
  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

// TransformerDecoderImpl 对象的前向传播函数
Tensor TransformerDecoderImpl::forward(
    const Tensor& tgt,
    const Tensor& memory,
    const Tensor& tgt_mask,
    const Tensor& memory_mask,
    const Tensor& tgt_key_padding_mask,
    const Tensor& memory_key_padding_mask) {
  // 获取 layers 的层数
  size_t num_layers = layers->size();
  // 初始化输出 Tensor
  Tensor output;
  // 如果有解码器层
  if (num_layers > 0) {
    // 对第一层解码器进行前向传播
    output = layers->at<TransformerDecoderLayerImpl>(0).forward(
        tgt,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask);
  }
  // 对剩余解码器层进行前向传播
  for (const auto i : c10::irange(1, num_layers)) {
    output = layers->at<TransformerDecoderLayerImpl>(i).forward(
        output,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask);
  }

  // 如果规范化模块不为空，应用规范化操作
  if (!norm.is_empty()) {
    output = norm.forward<Tensor>(num_layers == 0 ? tgt : output);
  }

  // 返回最终的输出 Tensor
  return output;
}

// TransformerImpl 类的构造函数，初始化 Transformer 对象
TransformerImpl::TransformerImpl(TransformerOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset 函数，设置 Transformer 对象的初始状态
  reset();
}

// 重置 TransformerImpl 对象的状态
void TransformerImpl::reset() {
  // 设置编码器的初始化
  if (options.custom_encoder().is_empty()) {
    // 如果没有自定义编码器，创建默认的 TransformerEncoder 对象
    LayerNorm norm(LayerNormOptions({options.d_model()}));
    TransformerEncoder trans_encoder(
        TransformerEncoderOptions(
            TransformerEncoderLayerOptions(options.d_model(), options.nhead())
                .dim_feedforward(options.dim_feedforward())
                .dropout(options.dropout())
                .activation(options.activation()),
            options.num_encoder_layers())
            .norm(AnyModule(norm)));

    // 将创建的编码器注册为 AnyModule 对象，并赋值给当前对象的 encoder 成员变量
    this->encoder = AnyModule(trans_encoder);
  } else {
    // 如果提供了自定义编码器选项，则使用其克隆来设置编码器
    this->encoder = options.custom_encoder().clone();
  }
  // 将编码器注册为模块
  this->register_module("encoder", this->encoder.ptr());

  // 设置解码器
  if (options.custom_decoder().is_empty()) {
    // 如果没有提供自定义解码器，则创建一个新的 TransformerDecoder 对象
    // 初始化一个 LayerNorm 对象和 TransformerDecoder 对象
    LayerNorm norm(LayerNormOptions({options.d_model()}));
    TransformerDecoder trans_decoder(
        TransformerDecoderOptions(
            // 设置 TransformerDecoderLayer 的选项
            TransformerDecoderLayerOptions(options.d_model(), options.nhead())
                .dim_feedforward(options.dim_feedforward())
                .dropout(options.dropout())
                .activation(options.activation()),
            options.num_decoder_layers())
            .norm(AnyModule(norm)));

    // 将新创建的 TransformerDecoder 对象包装为 AnyModule，并设置为解码器
    this->decoder = AnyModule(trans_decoder);
  } else {
    // 如果提供了自定义解码器选项，则使用其克隆来设置解码器
    this->decoder = options.custom_decoder().clone();
  }
  // 将解码器注册为模块
  this->register_module("decoder", this->decoder.ptr());

  // 调用重置参数的方法
  reset_parameters();
}

void TransformerImpl::reset_parameters() {
  // 获取当前模型的所有参数
  auto parameters = this->parameters();
  // 遍历模型的每个参数
  for (auto& param : parameters) {
    // 如果参数的维度大于1，使用 Xavier 均匀初始化
    if (param.dim() > 1) {
      torch::nn::init::xavier_uniform_(param);
    }
  }
}

Tensor TransformerImpl::forward(
    const Tensor& src,
    const Tensor& tgt,
    const Tensor& src_mask,
    const Tensor& tgt_mask,
    const Tensor& memory_mask,
    const Tensor& src_key_padding_mask,
    const Tensor& tgt_key_padding_mask,
    const Tensor& memory_key_padding_mask) {
  // 检查输入张量的维度
  TORCH_CHECK(
      src.dim() == 3 && tgt.dim() == 3,
      "src and tgt should have 3 dimensions, but got ",
      src.dim(),
      " and ",
      tgt.dim());

  // 检查输入张量的批大小是否相同
  TORCH_CHECK(
      src.size(1) == tgt.size(1),
      "src and tgt should have equal batch size (at dim 1), but got ",
      src.size(1),
      " and ",
      tgt.size(1));

  // 检查输入张量的特征维度是否与模型要求一致
  TORCH_CHECK(
      src.size(2) == options.d_model() && tgt.size(2) == options.d_model(),
      "src and tgt should have same feature size as d_model (at dim 2), but got ",
      src.size(2),
      " and ",
      tgt.size(2),
      " while d_model is ",
      options.d_model());

  // 使用编码器处理源数据
  Tensor memory =
      this->encoder.forward<Tensor>(src, src_mask, src_key_padding_mask);
  // 使用解码器处理目标数据
  Tensor output = this->decoder.forward<Tensor>(
      tgt,
      memory,
      tgt_mask,
      memory_mask,
      tgt_key_padding_mask,
      memory_key_padding_mask);

  // 返回解码器的输出作为模型的前向传播结果
  return output;
}

Tensor TransformerImpl::generate_square_subsequent_mask(int64_t sz) {
  // 确保输入大小为非负数，以生成有效的方形次序掩码
  TORCH_CHECK(
      sz >= 0,
      "Input size must be non-negative to generate a valid square subsequent mask, but got ",
      sz);

  // 检查当前平台是否支持 IEEE754 标准，因为非 IEEE754 平台上可能不支持 -inf
  if (std::numeric_limits<float>::is_iec559) {
    // 如果支持 IEEE754，使用负无穷大作为掩码填充值
    return torch::triu(
        torch::full({sz, sz}, -std::numeric_limits<float>::infinity()), 1);
  }
  // 如果不支持 IEEE754，使用当前平台支持的最小浮点数
  else {
    TORCH_WARN_ONCE(
        "IEEE754 is not supported on this platform, generate_square_subsequent_mask will fill "
        "the mask with smallest float number on this platform instead of -inf");
    return torch::triu(
        torch::full({sz, sz}, std::numeric_limits<float>::lowest()), 1);
  }
}

} // namespace nn
} // namespace torch
```