# `.\pytorch\torch\csrc\api\src\nn\modules\activation.cpp`

```
// 使用 torch 库中的模块和函数来定义激活函数的行为
#include <torch/nn/functional/activation.h>
#include <torch/nn/init.h>
#include <torch/nn/modules/activation.h>

// 命名空间别名，简化使用 torch::nn::functional 的代码
namespace F = torch::nn::functional;

// 定义 torch::nn 命名空间
namespace torch {
namespace nn {

// ELUOptions 和 ELUImpl 类的构造函数实现
ELUImpl::ELUImpl(const ELUOptions& options_) : options(options_) {}

// ELUImpl 类的前向传播函数实现，调用 torch::nn::functional::detail::elu 函数
Tensor ELUImpl::forward(Tensor input) {
  return F::detail::elu(input, options.alpha(), options.inplace());
}

// ELUImpl 类的 reset 函数，空实现
void ELUImpl::reset() {}

// ELUImpl 类的 pretty_print 函数实现，输出对象的可读字符串描述
void ELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ELU(alpha=" << options.alpha();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

// SELUOptions 和 SELUImpl 类的构造函数实现
SELUImpl::SELUImpl(const SELUOptions& options_) : options(options_) {}

// SELUImpl 类的前向传播函数实现，调用 torch::nn::functional::detail::selu 函数
Tensor SELUImpl::forward(Tensor input) {
  return F::detail::selu(input, options.inplace());
}

// SELUImpl 类的 reset 函数，空实现
void SELUImpl::reset() {}

// SELUImpl 类的 pretty_print 函数实现，输出对象的可读字符串描述
void SELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SELU(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

// HardshrinkOptions 和 HardshrinkImpl 类的构造函数实现
HardshrinkImpl::HardshrinkImpl(const HardshrinkOptions& options_)
    : options(options_) {}

// HardshrinkImpl 类的前向传播函数实现，调用 torch::nn::functional::detail::hardshrink 函数
Tensor HardshrinkImpl::forward(const Tensor& input) {
  return F::detail::hardshrink(input, options.lambda());
}

// HardshrinkImpl 类的 reset 函数，空实现
void HardshrinkImpl::reset() {}

// HardshrinkImpl 类的 pretty_print 函数实现，输出对象的可读字符串描述
void HardshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Hardshrink(" << options.lambda()
         << ")";
}

// ============================================================================

// HardtanhOptions 和 HardtanhImpl 类的构造函数实现
HardtanhImpl::HardtanhImpl(const HardtanhOptions& options_)
    : options(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

// HardtanhImpl 类的前向传播函数实现，调用 torch::nn::functional::detail::hardtanh 函数
Tensor HardtanhImpl::forward(Tensor input) {
  return F::detail::hardtanh(
      input, options.min_val(), options.max_val(), options.inplace());
}

// HardtanhImpl 类的 reset 函数，检查 HardtanhOptions 的有效性
void HardtanhImpl::reset() {
  TORCH_CHECK(
      options.max_val() > options.min_val(),
      "max_val must be greater than min_val");
}

// HardtanhImpl 类的 pretty_print 函数实现，输出对象的可读字符串描述
void HardtanhImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Hardtanh(min_val=" << options.min_val()
         << ", max_val=" << options.max_val();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

// LeakyReLUOptions 和 LeakyReLUImpl 类的构造函数实现
LeakyReLUImpl::LeakyReLUImpl(const LeakyReLUOptions& options_)
    : options(options_) {}

// LeakyReLUImpl 类的前向传播函数实现，调用 torch::nn::functional::detail::leaky_relu 函数
Tensor LeakyReLUImpl::forward(Tensor input) {
  return F::detail::leaky_relu(
      input, options.negative_slope(), options.inplace());
}

// LeakyReLUImpl 类的 reset 函数，空实现
void LeakyReLUImpl::reset() {}

// LeakyReLUImpl 类的 pretty_print 函数实现，输出对象的可读字符串描述
void LeakyReLUImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::LeakyReLU(negative_slope=" << options.negative_slope();
  if (options.inplace()) {
    // 如果 inplace 选项为真，则添加在可读字符串中
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// 结束 torch::nn 命名空间定义
} // namespace nn
} // namespace torch
    // 将布尔值以人类可读的形式输出到流中，std::boolalpha 设置流的格式为显示 true/false 而不是 1/0
    stream << std::boolalpha << ", inplace=" << options.inplace();
    // 输出字符串 ")"" 到流中，完成函数调用的结尾
    stream << ")";
}

// ============================================================================

// LogSigmoidImpl 类的 forward 方法实现
Tensor LogSigmoidImpl::forward(const Tensor& input) {
  // 调用 torch::nn::functional 的 logsigmoid 函数计算输入张量的 log-sigmoid
  return F::logsigmoid(input);
}

// 重置 LogSigmoidImpl 类的状态
void LogSigmoidImpl::reset() {}

// 将 LogSigmoidImpl 类的信息输出到流中
void LogSigmoidImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::LogSigmoid()";
}

// ============================================================================

// SoftmaxImpl 类的构造函数，根据给定选项初始化
SoftmaxImpl::SoftmaxImpl(const SoftmaxOptions& options_) : options(options_) {}

// 重置 SoftmaxImpl 类的状态
void SoftmaxImpl::reset() {}

// 将 SoftmaxImpl 类的信息输出到流中
void SoftmaxImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmax(dim=" << options.dim() << ")";
}

// SoftmaxImpl 类的 forward 方法实现，使用 F::detail::softmax 计算 softmax
Tensor SoftmaxImpl::forward(const Tensor& input) {
  return F::detail::softmax(input, options.dim(), c10::nullopt);
}

// ============================================================================

// SoftminImpl 类的构造函数，根据给定选项初始化
SoftminImpl::SoftminImpl(const SoftminOptions& options_) : options(options_) {}

// 重置 SoftminImpl 类的状态
void SoftminImpl::reset() {}

// 将 SoftminImpl 类的信息输出到流中
void SoftminImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmin(dim=" << options.dim() << ")";
}

// SoftminImpl 类的 forward 方法实现，使用 F::detail::softmin 计算 softmin
Tensor SoftminImpl::forward(const Tensor& input) {
  return F::detail::softmin(input, options.dim(), c10::nullopt);
}

// ============================================================================

// LogSoftmaxImpl 类的构造函数，根据给定选项初始化
LogSoftmaxImpl::LogSoftmaxImpl(const LogSoftmaxOptions& options_)
    : options(options_) {}

// 重置 LogSoftmaxImpl 类的状态
void LogSoftmaxImpl::reset() {}

// 将 LogSoftmaxImpl 类的信息输出到流中
void LogSoftmaxImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::LogSoftmax(dim=" << options.dim() << ")";
}

// LogSoftmaxImpl 类的 forward 方法实现，使用 F::detail::log_softmax 计算 log-softmax
Tensor LogSoftmaxImpl::forward(const Tensor& input) {
  return F::detail::log_softmax(input, options.dim(), c10::nullopt);
}

// ============================================================================

// Softmax2dImpl 类的构造函数，根据给定选项初始化
Softmax2dImpl::Softmax2dImpl(const SoftmaxOptions& options_) : options(options_) {}

// 重置 Softmax2dImpl 类的状态
void Softmax2dImpl::reset() {}

// 将 Softmax2dImpl 类的信息输出到流中
void Softmax2dImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmax2d()";
}

// Softmax2dImpl 类的 forward 方法实现，检查输入张量维度是否符合要求，然后使用 F::detail::softmax 计算 softmax
Tensor Softmax2dImpl::forward(const Tensor& input) {
  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 3,
      "Softmax2d requires a 3D or 4D tensor as input");
  return F::detail::softmax(input, /*dim=*/-3, c10::nullopt);
}

// ============================================================================

// PReLUImpl 类的构造函数，根据给定选项初始化，同时执行重置操作
PReLUImpl::PReLUImpl(const PReLUOptions& options_) : options(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset(); // 调用 reset 方法初始化权重参数
}

// PReLUImpl 类的 forward 方法实现，使用 F::prelu 函数计算 PReLU 激活
Tensor PReLUImpl::forward(const Tensor& input) {
  return F::prelu(input, weight); // 使用类成员变量 weight 进行计算
}

// 重置 PReLUImpl 类的状态，注册并初始化权重参数
void PReLUImpl::reset() {
  weight = register_parameter(
      "weight", torch::full(options.num_parameters(), options.init())); // 注册名为 "weight" 的参数，并用给定初始值初始化
}

// 将 PReLUImpl 类的信息输出到流中
void PReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PReLU(num_parameters=" << options.num_parameters()
         << ")";
}

// ============================================================================

// ReLUImpl 类的构造函数，根据给定选项初始化
ReLUImpl::ReLUImpl(const ReLUOptions& options_) : options(options_) {}

// ReLUImpl 类的 forward 方法实现，使用 F::detail::relu 计算 ReLU 激活
Tensor ReLUImpl::forward(Tensor input) {
  return F::detail::relu(input, options.inplace());
}

// 重置 ReLUImpl 类的状态
void ReLUImpl::reset() {}
void ReLUImpl::pretty_print(std::ostream& stream) const {
  // 打印 ReLU 层的描述信息，包括是否原地操作
  stream << "torch::nn::ReLU(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

ReLU6Impl::ReLU6Impl(const ReLU6Options& options_) : options(options_) {}

Tensor ReLU6Impl::forward(Tensor input) {
  // 调用 F::detail::relu6 函数进行 ReLU6 激活函数操作，传递是否原地操作参数
  return F::detail::relu6(input, options.inplace());
}

void ReLU6Impl::reset() {}

void ReLU6Impl::pretty_print(std::ostream& stream) const {
  // 打印 ReLU6 层的描述信息，包括是否原地操作
  stream << "torch::nn::ReLU6(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

RReLUImpl::RReLUImpl(const RReLUOptions& options_) : options(options_) {}

Tensor RReLUImpl::forward(Tensor input) {
  // 调用 F::detail::rrelu 函数进行 RReLU 激活函数操作，传递 lower、upper、训练状态和是否原地操作参数
  return F::detail::rrelu(
      input,
      options.lower(),
      options.upper(),
      is_training(),
      options.inplace());
}

void RReLUImpl::reset() {}

void RReLUImpl::pretty_print(std::ostream& stream) const {
  // 打印 RReLU 层的描述信息，包括 lower、upper 和是否原地操作
  stream << "torch::nn::RReLU(lower=" << options.lower()
         << ", upper=" << options.upper();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

CELUImpl::CELUImpl(const CELUOptions& options_) : options(options_) {}

Tensor CELUImpl::forward(Tensor input) {
  // 调用 F::detail::celu 函数进行 CELU 激活函数操作，传递 alpha 和是否原地操作参数
  return F::detail::celu(input, options.alpha(), options.inplace());
}

void CELUImpl::reset() {}

void CELUImpl::pretty_print(std::ostream& stream) const {
  // 打印 CELU 层的描述信息，包括 alpha 和是否原地操作
  stream << "torch::nn::CELU(alpha=" << options.alpha();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

GLUImpl::GLUImpl(const GLUOptions& options_) : options(options_) {}

Tensor GLUImpl::forward(const Tensor& input) {
  // 调用 F::detail::glu 函数进行 GLU 操作，传递 dim 参数
  return F::detail::glu(input, options.dim());
}

void GLUImpl::reset() {}

void GLUImpl::pretty_print(std::ostream& stream) const {
  // 打印 GLU 层的描述信息，包括 dim 参数
  stream << "torch::nn::GLU(dim=" << options.dim() << ")";
}

// ============================================================================

GELUImpl::GELUImpl(GELUOptions options_) : options(std::move(options_)) {}

Tensor GELUImpl::forward(const Tensor& input) {
  // 调用 F::detail::gelu 函数进行 GELU 激活函数操作，传递是否使用近似计算参数
  return F::detail::gelu(input, options.approximate());
}

void GELUImpl::reset() {}

void GELUImpl::pretty_print(std::ostream& stream) const {
  // 打印 GELU 层的描述信息，这里没有额外的参数
  stream << "torch::nn::GELU()";
}

// ============================================================================

Tensor SiLUImpl::forward(const Tensor& input) {
  // 调用 F::silu 函数进行 SiLU 激活函数操作
  return F::silu(input);
}

void SiLUImpl::reset() {}

void SiLUImpl::pretty_print(std::ostream& stream) const {
  // 打印 SiLU 层的描述信息，这里没有额外的参数
  stream << "torch::nn::SiLU()";
}

// ============================================================================
// 返回输入张量经过 Mish 激活函数处理后的结果张量
Tensor MishImpl::forward(const Tensor& input) {
  return F::mish(input);
}

// 重置 Mish 激活函数的状态，这里未做任何操作
void MishImpl::reset() {}

// 将 Mish 激活函数的信息打印到输出流中
void MishImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Mish()";
}

// ============================================================================

// 返回输入张量经过 Sigmoid 激活函数处理后的结果张量
Tensor SigmoidImpl::forward(const Tensor& input) {
  return torch::sigmoid(input);
}

// 重置 Sigmoid 激活函数的状态，这里未做任何操作
void SigmoidImpl::reset() {}

// 将 Sigmoid 激活函数的信息打印到输出流中
void SigmoidImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Sigmoid()";
}

// ============================================================================

// Softplus 激活函数的构造函数，设置其选项
SoftplusImpl::SoftplusImpl(const SoftplusOptions& options_)
    : options(options_) {}

// 返回输入张量经过 Softplus 激活函数处理后的结果张量
Tensor SoftplusImpl::forward(const Tensor& input) {
  return F::detail::softplus(input, options.beta(), options.threshold());
}

// 重置 Softplus 激活函数的状态，这里未做任何操作
void SoftplusImpl::reset() {}

// 将 Softplus 激活函数的信息打印到输出流中，包括其选项 beta 和 threshold
void SoftplusImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softplus(beta=" << options.beta()
         << ", threshold=" << options.threshold() << ")";
}

// ============================================================================

// Softshrink 激活函数的构造函数，设置其选项
SoftshrinkImpl::SoftshrinkImpl(const SoftshrinkOptions& options_)
    : options(options_) {}

// 返回输入张量经过 Softshrink 激活函数处理后的结果张量
Tensor SoftshrinkImpl::forward(const Tensor& input) {
  return F::detail::softshrink(input, options.lambda());
}

// 重置 Softshrink 激活函数的状态，这里未做任何操作
void SoftshrinkImpl::reset() {}

// 将 Softshrink 激活函数的信息打印到输出流中，包括其选项 lambda
void SoftshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softshrink(" << options.lambda() << ")";
}

// ============================================================================

// 返回输入张量经过 Softsign 激活函数处理后的结果张量
Tensor SoftsignImpl::forward(const Tensor& input) {
  return F::softsign(input);
}

// 重置 Softsign 激活函数的状态，这里未做任何操作
void SoftsignImpl::reset() {}

// 将 Softsign 激活函数的信息打印到输出流中
void SoftsignImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softsign()";
}

// ============================================================================

// 返回输入张量经过 Tanh 激活函数处理后的结果张量
Tensor TanhImpl::forward(const Tensor& input) {
  return torch::tanh(input);
}

// 重置 Tanh 激活函数的状态，这里未做任何操作
void TanhImpl::reset() {}

// 将 Tanh 激活函数的信息打印到输出流中
void TanhImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Tanh()";
}

// ============================================================================

// 返回输入张量经过 Tanhshrink 激活函数处理后的结果张量
Tensor TanhshrinkImpl::forward(const Tensor& input) {
  return F::tanhshrink(input);
}

// 重置 Tanhshrink 激活函数的状态，这里未做任何操作
void TanhshrinkImpl::reset() {}

// 将 Tanhshrink 激活函数的信息打印到输出流中
void TanhshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Tanhshrink()";
}

// ============================================================================

// Threshold 激活函数的构造函数，设置其选项
ThresholdImpl::ThresholdImpl(const ThresholdOptions& options_)
    : options(options_) {}

// 返回输入张量经过 Threshold 激活函数处理后的结果张量
Tensor ThresholdImpl::forward(Tensor input) {
  return F::detail::threshold(
      input, options.threshold(), options.value(), options.inplace());
}

// 重置 Threshold 激活函数的状态，这里未做任何操作
void ThresholdImpl::reset() {}

// 将 Threshold 激活函数的信息打印到输出流中，包括其选项 threshold 和 value，以及是否 inplace
void ThresholdImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Threshold(threshold=" << options.threshold()
         << ", value=" << options.value();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}
// ============================================================================
MultiheadAttentionImpl::MultiheadAttentionImpl(
    const MultiheadAttentionOptions& options_)
    : Cloneable("torch::nn::MultiheadAttention"), options(options_) {
  // 构造函数，初始化 MultiheadAttentionImpl 对象
  // 使用 options_ 参数设置选项，调用基类 Cloneable 的构造函数
  // 将选项保存到成员变量 options 中
  reset();  // 调用 reset 方法进行初始化
}

std::tuple<Tensor, Tensor> MultiheadAttentionImpl::forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& key_padding_mask,
    bool need_weights,
    const Tensor& attn_mask,
    bool average_attn_weights) {
  if (!_qkv_same_embed_dim) {
    // 如果输入的 query、key、value 的嵌入维度不同，则调用 F::multi_head_attention_forward 方法
    return F::multi_head_attention_forward(
        query,
        key,
        value,
        F::MultiheadAttentionForwardFuncOptions(
            /*embed_dim=*/options.embed_dim(),
            /*num_heads=*/options.num_heads(),
            /*in_proj_weight=*/in_proj_weight,
            /*in_proj_bias=*/in_proj_bias,
            /*bias_k=*/bias_k,
            /*bias_v=*/bias_v,
            /*add_zero_attn=*/options.add_zero_attn(),
            /*dropout_p=*/options.dropout(),
            /*out_proj_weight=*/out_proj->weight,
            /*out_proj_bias=*/out_proj->bias)
            .training(is_training())  // 设置是否处于训练模式
            .key_padding_mask(key_padding_mask)  // 设置 key_padding_mask
            .need_weights(need_weights)  // 设置是否需要权重
            .attn_mask(attn_mask)  // 设置注意力掩码
            .use_separate_proj_weight(true)  // 使用独立的投影权重
            .q_proj_weight(q_proj_weight)  // 设置查询投影权重
            .k_proj_weight(k_proj_weight)  // 设置键投影权重
            .v_proj_weight(v_proj_weight)  // 设置值投影权重
            .average_attn_weights(average_attn_weights));  // 设置是否平均注意力权重
  } else {
    // 如果输入的 query、key、value 的嵌入维度相同，则调用 F::multi_head_attention_forward 方法
    return F::multi_head_attention_forward(
        query,
        key,
        value,
        F::MultiheadAttentionForwardFuncOptions(
            /*embed_dim=*/options.embed_dim(),
            /*num_heads=*/options.num_heads(),
            /*in_proj_weight=*/in_proj_weight,
            /*in_proj_bias=*/in_proj_bias,
            /*bias_k=*/bias_k,
            /*bias_v=*/bias_v,
            /*add_zero_attn=*/options.add_zero_attn(),
            /*dropout_p=*/options.dropout(),
            /*out_proj_weight=*/out_proj->weight,
            /*out_proj_bias=*/out_proj->bias)
            .training(is_training())  // 设置是否处于训练模式
            .key_padding_mask(key_padding_mask)  // 设置 key_padding_mask
            .need_weights(need_weights)  // 设置是否需要权重
            .attn_mask(attn_mask)  // 设置注意力掩码
            .average_attn_weights(average_attn_weights));  // 设置是否平均注意力权重
  }
}

void MultiheadAttentionImpl::reset() {
  // 重置 MultiheadAttentionImpl 对象的状态
  _qkv_same_embed_dim = options.kdim() == options.embed_dim() &&
      options.vdim() == options.embed_dim();  // 检查查询、键、值的嵌入维度是否相同
  head_dim = options.embed_dim() / options.num_heads();  // 计算每个头的维度
  TORCH_CHECK(
      head_dim * options.num_heads() == options.embed_dim(),
      "embed_dim must be divisible by num_heads");  // 检查 embed_dim 是否可以被 num_heads 整除
  if (!_qkv_same_embed_dim) {
    // 如果查询、键、值的嵌入维度不同，则注册查询投影权重参数
    q_proj_weight = register_parameter(
        "q_proj_weight",
        torch::empty({options.embed_dim(), options.embed_dim()}));
    // 注册权重参数 k_proj_weight，维度为 (embed_dim, kdim)，用于查询投影
    k_proj_weight = register_parameter(
        "k_proj_weight", torch::empty({options.embed_dim(), options.kdim()}));
    
    // 注册权重参数 v_proj_weight，维度为 (embed_dim, vdim)，用于数值投影
    v_proj_weight = register_parameter(
        "v_proj_weight", torch::empty({options.embed_dim(), options.vdim()}));
    
    // 注册不需要梯度的参数 in_proj_weight，维度为 (3 * embed_dim, embed_dim)，在需要时使用
    register_parameter("in_proj_weight", {}, /*requires_grad=*/false);
  } else {
    // 如果不需要分离的投影，注册参数 in_proj_weight
    in_proj_weight = register_parameter(
        "in_proj_weight",
        torch::empty({3 * options.embed_dim(), options.embed_dim()}));
    
    // 注册不需要梯度的参数 q_proj_weight，用于查询投影
    register_parameter("q_proj_weight", {}, /*requires_grad=*/false);
    
    // 注册不需要梯度的参数 k_proj_weight，用于键的投影
    register_parameter("k_proj_weight", {}, /*requires_grad=*/false);
    
    // 注册不需要梯度的参数 v_proj_weight，用于值的投影
    register_parameter("v_proj_weight", {}, /*requires_grad=*/false);
  }
  
  // 如果需要偏置项
  if (options.bias()) {
    // 注册带有偏置项的参数 in_proj_bias，维度为 (3 * embed_dim)
    in_proj_bias = register_parameter(
        "in_proj_bias", torch::empty(3 * options.embed_dim()));
  } else {
    // 否则注册不需要梯度的参数 in_proj_bias
    register_parameter("in_proj_bias", {}, /*requires_grad=*/false);
  }
  
  // 注册输出投影层 out_proj，是一个线性层，输入输出维度均为 embed_dim
  out_proj = register_module(
      "out_proj",
      Linear(LinearOptions(options.embed_dim(), options.embed_dim())
                 .bias(options.bias())));
  
  // 如果需要添加偏置项到键值中
  if (options.add_bias_kv()) {
    // 注册 bias_k 参数，维度为 (1, 1, embed_dim)，用于键的偏置
    bias_k =
        register_parameter("bias_k", torch::empty({1, 1, options.embed_dim()}));
    
    // 注册 bias_v 参数，维度为 (1, 1, embed_dim)，用于值的偏置
    bias_v =
        register_parameter("bias_v", torch::empty({1, 1, options.embed_dim()}));
  } else {
    // 否则重置 bias_k 和 bias_v 参数
    bias_k.reset();
    bias_v.reset();
  }
  
  // 调用 _reset_parameters 方法，用于初始化或重置模型参数
  _reset_parameters();
}

void MultiheadAttentionImpl::_reset_parameters() {
  // 命名空间别名引用，初始化 torch::nn::init 命名空间下的函数和变量
  using namespace torch::nn::init;

  // 如果查询、键、值的嵌入维度相同，则对输入投影权重进行均匀分布的 Xavier 初始化
  if (_qkv_same_embed_dim) {
    xavier_uniform_(in_proj_weight);
  } else {
    // 否则分别对查询、键、值的投影权重进行均匀分布的 Xavier 初始化
    xavier_uniform_(q_proj_weight);
    xavier_uniform_(k_proj_weight);
    xavier_uniform_(v_proj_weight);
  }

  // 如果输入投影偏置已定义，则将其初始化为常数 0
  if (in_proj_bias.defined()) {
    constant_(in_proj_bias, 0.);
    // 同时将输出投影层的偏置初始化为常数 0
    constant_(out_proj->bias, 0.);
  }

  // 如果存在键的偏置项，则对其进行 Xavier 正态分布初始化
  if (bias_k.defined()) {
    xavier_normal_(bias_k);
  }

  // 如果存在值的偏置项，则对其进行 Xavier 正态分布初始化
  if (bias_v.defined()) {
    xavier_normal_(bias_v);
  }
}

} // namespace nn
} // namespace torch
```