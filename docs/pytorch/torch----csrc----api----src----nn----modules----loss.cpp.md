# `.\pytorch\torch\csrc\api\src\nn\modules\loss.cpp`

```py
// 包含 Torch 库中的损失函数模块头文件
#include <torch/nn/modules/loss.h>

// 定义命名空间 F 为 torch::nn::functional
namespace F = torch::nn::functional;

// 定义命名空间 torch::nn
namespace torch {
namespace nn {

// L1 损失的实现类构造函数，初始化选项
L1LossImpl::L1LossImpl(L1LossOptions options_) : options(std::move(options_)) {}

// 重置函数，未实现内容
void L1LossImpl::reset() {}

// 打印漂亮的字符串表示，指示 L1 损失
void L1LossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::L1Loss()";
}

// 前向传播函数，调用 torch::nn::functional 的 L1 损失计算函数
Tensor L1LossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::l1_loss(input, target, options.reduction());
}

// ============================================================================

// KL 散度损失的实现类构造函数，初始化选项
KLDivLossImpl::KLDivLossImpl(KLDivLossOptions options_)
    : options(std::move(options_)) {}

// 重置函数，未实现内容
void KLDivLossImpl::reset() {}

// 打印漂亮的字符串表示，指示 KL 散度损失
void KLDivLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::KLDivLoss()";
}

// 前向传播函数，调用 torch::nn::functional 的 KL 散度计算函数
Tensor KLDivLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::kl_div(
      input, target, options.reduction(), options.log_target());
}

// ============================================================================

// 均方误差损失的实现类构造函数，初始化选项
MSELossImpl::MSELossImpl(MSELossOptions options_)
    : options(std::move(options_)) {}

// 重置函数，未实现内容
void MSELossImpl::reset() {}

// 打印漂亮的字符串表示，指示均方误差损失
void MSELossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MSELoss()";
}

// 前向传播函数，调用 torch::nn::functional 的均方误差计算函数
Tensor MSELossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::mse_loss(input, target, options.reduction());
}

// ============================================================================

// 二元交叉熵损失的实现类构造函数，初始化选项
BCELossImpl::BCELossImpl(BCELossOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

// 重置函数，注册权重缓冲区
void BCELossImpl::reset() {
  register_buffer("weight", options.weight());
}

// 打印漂亮的字符串表示，指示二元交叉熵损失
void BCELossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::BCELoss()";
}

// 前向传播函数，调用 torch::nn::functional 的二元交叉熵计算函数
Tensor BCELossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::binary_cross_entropy(
      input, target, options.weight(), options.reduction());
}

// ============================================================================

// 合页损失的实现类构造函数，初始化选项
HingeEmbeddingLossImpl::HingeEmbeddingLossImpl(
    HingeEmbeddingLossOptions options_)
    : options(std::move(options_)) {}

// 重置函数，未实现内容
void HingeEmbeddingLossImpl::reset() {}

// 打印漂亮的字符串表示，指示合页损失，显示边界参数
void HingeEmbeddingLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::HingeEmbeddingLoss(margin=" << options.margin() << ")";
}

// 前向传播函数，调用 torch::nn::functional 的合页损失计算函数
Tensor HingeEmbeddingLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::detail::hinge_embedding_loss(
      input, target, options.margin(), options.reduction());
}

// ============================================================================

// 多类别边缘损失的实现类构造函数，初始化选项
MultiMarginLossImpl::MultiMarginLossImpl(MultiMarginLossOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}
// 重置损失函数的状态，检查选项中的参数 p 只支持 1 或 2
void MultiMarginLossImpl::reset() {
  TORCH_CHECK(
      (options.p() == 1) || (options.p() == 2),
      "only p == 1 and p == 2 supported");
  // 检查权重是否定义且是一维的
  TORCH_CHECK(!options.weight().defined() || options.weight().dim() == 1);

  // 注册缓冲区 "weight"，使用 options 中的权重
  register_buffer("weight", options.weight());
}

// 将损失函数的信息输出到流中，包括参数 p、margin、weight 和 reduction
void MultiMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiMarginLoss(p=" << options.p()
         << ", margin=" << options.margin() << ", weight=" << options.weight()
         << ", reduction=" << enumtype::get_enum_name(options.reduction())
         << ")";
}

// 计算前向传播，使用 F::detail::multi_margin_loss 函数计算多分类边界损失
Tensor MultiMarginLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::multi_margin_loss(
      input,
      target,
      options.p(),
      options.margin(),
      options.weight(),
      options.reduction());
}

// ============================================================================

// 初始化余弦嵌入损失的实现，保存选项到成员变量 options
CosineEmbeddingLossImpl::CosineEmbeddingLossImpl(
    CosineEmbeddingLossOptions options_)
    : options(std::move(options_)) {}

// 重置余弦嵌入损失的状态，目前无操作
void CosineEmbeddingLossImpl::reset() {}

// 将余弦嵌入损失的信息输出到流中，包括 margin 参数
void CosineEmbeddingLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CosineEmbeddingLoss(margin=" << options.margin() << ")";
}

// 计算前向传播，使用 F::detail::cosine_embedding_loss 函数计算余弦嵌入损失
Tensor CosineEmbeddingLossImpl::forward(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target) {
  return F::detail::cosine_embedding_loss(
      input1, input2, target, options.margin(), options.reduction());
}
// ============================================================================

// 初始化多标签软最小化损失的实现，保存选项到成员变量 options
MultiLabelSoftMarginLossImpl::MultiLabelSoftMarginLossImpl(
    torch::nn::MultiLabelSoftMarginLossOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

// 将多标签软最小化损失的信息输出到流中
void MultiLabelSoftMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiLabelSoftMarginLoss()";
}

// 重置多标签软最小化损失的状态，注册权重缓冲区
void MultiLabelSoftMarginLossImpl::reset() {
  register_buffer("weight", options.weight());
}

// 计算前向传播，使用 F::detail::multilabel_soft_margin_loss 函数计算多标签软最小化损失
Tensor MultiLabelSoftMarginLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::detail::multilabel_soft_margin_loss(
      input, target, options.weight(), options.reduction());
}

// ============================================================================

// 初始化三元组边界损失的实现，保存选项到成员变量 options
TripletMarginLossImpl::TripletMarginLossImpl(TripletMarginLossOptions options_)
    : options(std::move(options_)) {}

// 重置三元组边界损失的状态，目前无操作
void TripletMarginLossImpl::reset() {}

// 将三元组边界损失的信息输出到流中，包括 margin、p、eps 和 swap 参数
void TripletMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::TripletMarginLoss(margin=" << options.margin()
         << ", p=" << options.p() << ", eps=" << options.eps() << std::boolalpha
         << ", swap=" << options.swap() << ")";
}

// 计算前向传播，未完整展示的部分
Tensor TripletMarginLossImpl::forward(
    const Tensor& anchor,
    const Tensor& positive,
    # 定义一个函数，计算三元组间隔损失
    def triplet_margin_loss(
        # 第一个参数，表示锚点样本的张量
        const Tensor& anchor,
        # 第二个参数，表示正样本的张量
        const Tensor& positive,
        # 第三个参数，表示负样本的张量
        const Tensor& negative) {
      # 调用底层函数 detail::triplet_margin_loss 计算损失
      return F::detail::triplet_margin_loss(
          anchor,
          positive,
          negative,
          # 使用 options 对象中设置的间隔值
          options.margin(),
          # 使用 options 对象中设置的 p 值
          options.p(),
          # 使用 options 对象中设置的 eps 值
          options.eps(),
          # 使用 options 对象中设置的 swap 值
          options.swap(),
          # 使用 options 对象中设置的 reduction 类型
          options.reduction());
    }
// ============================================================================

// 实现 Triplet Margin with Distance Loss 的类 TripletMarginWithDistanceLossImpl
TripletMarginWithDistanceLossImpl::TripletMarginWithDistanceLossImpl(
    TripletMarginWithDistanceLossOptions options_)
    : options(std::move(options_)) {}

// 重置操作，暂无具体实现
void TripletMarginWithDistanceLossImpl::reset() {}

// 将 Triplet Margin with Distance Loss 的信息输出到流中
void TripletMarginWithDistanceLossImpl::pretty_print(
    std::ostream& stream) const {
  stream << "torch::nn::TripletMarginWithDistanceLoss(margin="
         << options.margin() << std::boolalpha << ", swap=" << options.swap()
         << ")";
}

// 执行前向传播，计算 Triplet Margin with Distance Loss
Tensor TripletMarginWithDistanceLossImpl::forward(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative) {
  return F::detail::triplet_margin_with_distance_loss(
      anchor,
      positive,
      negative,
      options.distance_function(),
      options.margin(),
      options.swap(),
      options.reduction());
}

// ============================================================================

// 实现 MultiLabel Margin Loss 的类 MultiLabelMarginLossImpl
MultiLabelMarginLossImpl::MultiLabelMarginLossImpl(
    torch::nn::MultiLabelMarginLossOptions options_)
    : options(std::move(options_)) {}

// 重置操作，暂无具体实现
void MultiLabelMarginLossImpl::reset() {}

// 将 MultiLabel Margin Loss 的信息输出到流中
void MultiLabelMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiLabelMarginLoss()";
}

// 执行前向传播，计算 MultiLabel Margin Loss
Tensor MultiLabelMarginLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::detail::multilabel_margin_loss(input, target, options.reduction());
}

// ============================================================================

// 实现 Soft Margin Loss 的类 SoftMarginLossImpl
SoftMarginLossImpl::SoftMarginLossImpl(
    torch::nn::SoftMarginLossOptions options_)
    : options(std::move(options_)) {}

// 重置操作，暂无具体实现
void SoftMarginLossImpl::reset() {}

// 将 Soft Margin Loss 的信息输出到流中
void SoftMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SoftMarginLoss()";
}

// 执行前向传播，计算 Soft Margin Loss
Tensor SoftMarginLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::soft_margin_loss(input, target, options.reduction());
}

// ============================================================================

// 实现 Smooth L1 Loss 的类 SmoothL1LossImpl
SmoothL1LossImpl::SmoothL1LossImpl(torch::nn::SmoothL1LossOptions options_)
    : options(std::move(options_)) {}

// 重置操作，暂无具体实现
void SmoothL1LossImpl::reset() {}

// 将 Smooth L1 Loss 的信息输出到流中
void SmoothL1LossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SmoothL1Loss";
}

// 执行前向传播，计算 Smooth L1 Loss
Tensor SmoothL1LossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::smooth_l1_loss(
      input, target, options.reduction(), options.beta());
}

// ============================================================================

// 实现 Huber Loss 的类 HuberLossImpl
HuberLossImpl::HuberLossImpl(torch::nn::HuberLossOptions options_)
    : options(std::move(options_)) {}

// 重置操作，暂无具体实现
void HuberLossImpl::reset() {}

// 将 Huber Loss 的信息输出到流中
void HuberLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::HuberLoss";
}

// 执行前向传播，计算 Huber Loss
Tensor HuberLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::huber_loss(
      input, target, options.reduction(), options.delta());
}

// ============================================================================
// ============================================================================
// CTCLossImpl 类的构造函数，初始化 options 成员变量
CTCLossImpl::CTCLossImpl(CTCLossOptions options_)
    : options(std::move(options_)) {}

// 重置函数，未实现具体功能
void CTCLossImpl::reset() {}

// 将 CTCLossImpl 对象信息输出到流中
void CTCLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CTCLoss()";
}

// CTCLoss 的前向传播函数，调用 F::detail::ctc_loss 执行 CTC 损失计算
Tensor CTCLossImpl::forward(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths) {
  return F::detail::ctc_loss(
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      options.blank(),
      options.reduction(),
      options.zero_infinity());
}
// ============================================================================

// ============================================================================
// PoissonNLLLossImpl 类的构造函数，初始化 options 成员变量
PoissonNLLLossImpl::PoissonNLLLossImpl(PoissonNLLLossOptions options_)
    : options(std::move(options_)) {}

// 重置函数，未实现具体功能
void PoissonNLLLossImpl::reset() {}

// 将 PoissonNLLLossImpl 对象信息输出到流中
void PoissonNLLLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PoissonNLLLoss()";
}

// PoissonNLLLoss 的前向传播函数，调用 F::detail::poisson_nll_loss 执行 Poisson Negative Log-Likelihood 损失计算
Tensor PoissonNLLLossImpl::forward(
    const Tensor& log_input,
    const Tensor& target) {
  return F::detail::poisson_nll_loss(
      log_input,
      target,
      options.log_input(),
      options.full(),
      options.eps(),
      options.reduction());
}
// ============================================================================

// ============================================================================
// MarginRankingLossImpl 类的构造函数，初始化 options 成员变量
MarginRankingLossImpl::MarginRankingLossImpl(MarginRankingLossOptions options_)
    : options(std::move(options_)) {}

// 重置函数，未实现具体功能
void MarginRankingLossImpl::reset() {}

// 将 MarginRankingLossImpl 对象信息输出到流中
void MarginRankingLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MarginRankingLoss()";
}

// MarginRankingLoss 的前向传播函数，调用 F::detail::margin_ranking_loss 执行 Margin Ranking Loss 损失计算
Tensor MarginRankingLossImpl::forward(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target) {
  return F::detail::margin_ranking_loss(
      input1, input2, target, options.margin(), options.reduction());
}
// ============================================================================

// ============================================================================
// NLLLossImpl 类的构造函数，初始化 options 成员变量，并调用 reset 函数
NLLLossImpl::NLLLossImpl(NLLLossOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

// NLLLossImpl 类的 reset 函数，将权重信息注册到缓冲区
void NLLLossImpl::reset() {
  weight = register_buffer("weight", options.weight());
}

// 将 NLLLossImpl 对象信息输出到流中
void NLLLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::NLLLoss()";
}

// NLLLoss 的前向传播函数，调用 F::detail::nll_loss 执行 Negative Log-Likelihood 损失计算
Tensor NLLLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::nll_loss(
      input, target, weight, options.ignore_index(), options.reduction());
}
// ============================================================================

// ============================================================================
// CrossEntropyLossImpl 类的构造函数，初始化 options 成员变量，并调用 reset 函数
CrossEntropyLossImpl::CrossEntropyLossImpl(CrossEntropyLossOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

// CrossEntropyLossImpl 类的 reset 函数，将权重信息注册到缓冲区
void CrossEntropyLossImpl::reset() {
  weight = register_buffer("weight", options.weight());
}

// 将 CrossEntropyLossImpl 对象信息输出到流中
void CrossEntropyLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CrossEntropyLoss()";
}

// CrossEntropyLoss 的前向传播函数，未完全实现
Tensor CrossEntropyLossImpl::forward(
    // 定义一个函数，计算交叉熵损失
    const Tensor& input,  // 输入张量，包含模型的预测结果
    const Tensor& target) {  // 目标张量，包含真实的标签值
  return F::detail::cross_entropy(  // 调用细节命名空间下的交叉熵函数
      input,  // 将输入张量传递给交叉熵函数
      target,  // 将目标张量传递给交叉熵函数
      weight,  // 权重张量，用于加权损失计算（假如有的话）
      options.ignore_index(),  // 忽略指定的索引，如果目标值等于该索引，则损失为0
      options.reduction(),  // 损失函数的缩减方式（如求和、求平均等）
      options.label_smoothing());  // 标签平滑的参数，用于减少过拟合和提升泛化能力
}

// ============================================================================

// 使用给定的选项初始化 BCEWithLogitsLossImpl 类
BCEWithLogitsLossImpl::BCEWithLogitsLossImpl(BCEWithLogitsLossOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  // 调用 reset 方法进行初始化
  reset();
}

// 重置 BCEWithLogitsLossImpl 对象
void BCEWithLogitsLossImpl::reset() {
  // 根据选项中的权重参数注册权重缓冲区
  weight = register_buffer("weight", options.weight());
  // 根据选项中的正权重参数注册正权重缓冲区
  pos_weight = register_buffer("pos_weight", options.pos_weight());
}

// 将 BCEWithLogitsLossImpl 对象的信息打印到给定的流中
void BCEWithLogitsLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::BCEWithLogitsLoss()";
}

// 计算并返回二进制交叉熵损失的前向传播结果
Tensor BCEWithLogitsLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::detail::binary_cross_entropy_with_logits(
      input,
      target,
      options.weight(),
      options.reduction(),
      options.pos_weight());
}

} // namespace nn
} // namespace torch
```