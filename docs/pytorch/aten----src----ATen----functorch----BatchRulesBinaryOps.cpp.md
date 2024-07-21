# `.\pytorch\aten\src\ATen\functorch\BatchRulesBinaryOps.cpp`

```
// 包含头文件：定义了批处理规则助手的实现所需的所有必要头文件
#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <utility>

// 进入functorch命名空间
namespace at::functorch {

// 定义模板函数_binary_pointwise_batch_rule，根据给定的二元点逐元素操作函数和额外参数，返回结果张量和可选的批处理维度
template <typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor,optional<int64_t>> _binary_pointwise_batch_rule(
    const Tensor& tensor, optional<int64_t> tensor_batch_dim,
    const Tensor& other, optional<int64_t> other_batch_dim,
    ExtraArgs... extra_args) {

  // 调用_binary_pointwise_helper函数处理输入张量和批处理维度，得到处理后的张量
  auto tensor_other = _binary_pointwise_helper(
      tensor, tensor_batch_dim, other, other_batch_dim);
  // 分别获取处理后的张量
  auto tensor_ = std::get<0>(tensor_other);
  auto other_ = std::get<1>(tensor_other);

  // 调用给定的二元点逐元素操作函数Func，计算结果
  auto result = Func(tensor_, other_, std::forward<ExtraArgs>(extra_args)...);
  // 返回结果张量和0作为默认的批处理维度
  return std::make_tuple(result, 0);
}

// 定义模板结构BinaryPointwiseBatchRuleHelper，用于生成二元点逐元素操作的批处理规则助手
template <typename F, F Func, typename T1, typename T2, typename... T>
struct BinaryPointwiseBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  // 静态成员函数apply，调用_binary_pointwise_batch_rule生成批处理规则结果
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& tensor, optional<int64_t> tensor_batch_dim,
      const Tensor& other, optional<int64_t> other_batch_dim,
      T... extra_args) {
    return _binary_pointwise_batch_rule<F, Func, T...>(
        tensor, tensor_batch_dim, other, other_batch_dim,
        std::forward<T>(extra_args)...);
  }
};

// 宏定义BINARY_POINTWISE_BATCH_RULE(fn)，用于简化调用BinaryPointwiseBatchRuleHelper的单参数版本
#define BINARY_POINTWISE_BATCH_RULE(fn) SINGLE_ARG(\
    BinaryPointwiseBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

// 定义模板结构BinaryRandomPointwiseBatchRuleHelper，用于生成随机的二元点逐元素操作的批处理规则助手
template <typename F, F Func, typename T1, typename T2, typename... T>
struct BinaryRandomPointwiseBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  // 静态成员函数apply，进行随机二元点逐元素操作，并在FuncTorchVmapMode下排除调度键，返回处理后的张量
  static Tensor apply(const Tensor& tensor, const Tensor& other, T... extra_args) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchVmapMode);
    // 获取当前动态层的信息，包括层ID和随机性类型
    auto maybe_layer = maybeCurrentDynamicLayer();
    auto cur_level = maybe_layer->layerId();
    RandomnessType randomness = maybe_layer->randomness();

    // 解包tensor和other张量在当前层级的值和批处理维度
    auto [tensor_value, tensor_bdim] = unwrapTensorAtLevel(tensor, cur_level);
    auto [other_value, other_bdim] = unwrapTensorAtLevel(other, cur_level);

    // 检查随机性，确保随机性与张量的批处理维度一致
    check_randomness(randomness, (tensor_bdim || other_bdim));
    // 如果随机性是不同且张量和其他张量都未分批
    if (randomness == RandomnessType::Different && !tensor_bdim && !other_bdim) {
      // 获取张量值的形状
      auto shape = tensor_value.sizes();
      // 创建一个包含可能的层批大小的符号维度向量
      VmapSymDimVector shapeVec(1, maybe_layer->batchSize());
      // 预留足够的空间以容纳形状的大小加一
      shapeVec.reserve(shape.size() + 1);
      // 将形状的元素插入到符号维度向量的末尾
      shapeVec.insert(shapeVec.end(), shape.begin(), shape.end());

      // 根据二进制批处理规则扩展张量值，这里假设至少有一个输入是批处理的
      tensor_value = tensor_value.expand_symint(shapeVec);
      // 将张量批次维度设置为零
      tensor_bdim = 0;
    } else if (randomness == RandomnessType::Same && !tensor_bdim && !other_bdim) {
      // 避免不必要的检查和批处理规则，假设输出是批处理的情况下返回函数调用结果
      return Func(tensor_value, other_value, std::forward<T>(extra_args)...);
    }
    // 调用二元逐点批处理规则，传递张量值、张量批次维度、其他值、其他批次维度以及额外参数
    auto res = _binary_pointwise_batch_rule<F, Func, T...>(
      tensor_value, tensor_bdim, other_value, other_bdim,
      std::forward<T>(extra_args)...);
    // 使用结果的第一个和第二个元素以及当前级别创建批处理
    return makeBatched(std::get<0>(res), std::get<1>(res), cur_level);
  }
};

// 定义宏，将二进制随机逐点批处理规则转换为调用
#define BINARY_RANDOM_POINTWISE_BATCH_RULE(fn) SINGLE_ARG(\
    BinaryRandomPointwiseBatchRuleHelper<\
      decltype(&fn),\
      &fn,\
      c10::guts::function_traits<decltype(fn)>::parameter_types>::apply)

// 二进制逐点就地批处理规则函数模板
template <typename M, M Meth, typename... ExtraArgs>
void binary_pointwise_inplace_batch_rule(
    Tensor& tensor, optional<int64_t> tensor_batch_dim,
    const Tensor& other, optional<int64_t> other_batch_dim,
    ExtraArgs... extra_args) {
  // 如果张量没有批次维度，但是其他张量有批次维度，则抛出不兼容的就地算术错误
  if (!tensor_batch_dim && other_batch_dim) {
    vmapIncompatibleInplaceError("inplace arithmetic");
  }

  // 计算最大逻辑秩
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  // 将批次维度移动到张量的最前面
  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // 如果维度不对齐，则进行填充以匹配最大逻辑秩
  // 例如：Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  // 调用成员函数指针 Meth 所指示的函数
  (tensor_.*Meth)(other_, std::forward<ExtraArgs>(extra_args)...);
}

// 比较逐点批处理规则函数模板
template <typename F, F Func>
std::tuple<Tensor,optional<int64_t>> comparison_pointwise_batch_rule(
    const Tensor& tensor, optional<int64_t> tensor_batch_dim,
    const Tensor& other, optional<int64_t> other_batch_dim) {
  // 计算最大逻辑秩
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, tensor_batch_dim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_batch_dim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  // 将批次维度移动到张量的最前面
  auto tensor_ = moveBatchDimToFront(tensor, tensor_batch_dim);
  auto other_ = moveBatchDimToFront(other, other_batch_dim);

  // 如果维度不对齐，则进行填充以匹配最大逻辑秩
  // 例如：Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  tensor_ = maybePadToLogicalRank(tensor_, tensor_batch_dim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, other_batch_dim, max_logical_rank);

  // 调用模板参数 Func 所指示的函数，并返回结果
  auto result = Func(tensor_, other_);
  return std::make_tuple( std::move(result), 0 );
}

// where_self_batch_rule 函数的声明
static std::tuple<Tensor,optional<int64_t>> where_self_batch_rule(
    const Tensor& condition, optional<int64_t> condition_bdim,
    const Tensor& input, optional<int64_t> input_bdim);
    // 获取条件张量的逻辑排名（即不考虑批量维度的秩）
    auto condition_logical_rank = rankWithoutBatchDim(condition, condition_bdim);
    // 获取 self 张量的逻辑排名（即不考虑批量维度的秩）
    auto tensor_logical_rank = rankWithoutBatchDim(self, self_bdim);
    // 获取 other 张量的逻辑排名（即不考虑批量维度的秩）
    auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
    // 计算出三者之间的最大逻辑排名
    auto max_logical_rank = std::max({tensor_logical_rank, other_logical_rank, condition_logical_rank});
    
    // 将条件张量的批量维度移动到最前面，并返回移动后的张量
    auto condition_ = moveBatchDimToFront(condition, condition_bdim);
    // 将 self 张量的批量维度移动到最前面，并返回移动后的张量
    auto self_ = moveBatchDimToFront(self, self_bdim);
    // 将 other 张量的批量维度移动到最前面，并返回移动后的张量
    auto other_ = moveBatchDimToFront(other, other_bdim);
    
    // 可能将条件张量填充到最大逻辑排名，返回填充后的条件张量
    condition_ = maybePadToLogicalRank(condition_, condition_bdim, max_logical_rank);
    // 可能将 self 张量填充到最大逻辑排名，返回填充后的 self 张量
    self_ = maybePadToLogicalRank(self_, self_bdim, max_logical_rank);
    // 可能将 other 张量填充到最大逻辑排名，返回填充后的 other 张量
    other_ = maybePadToLogicalRank(other_, other_bdim, max_logical_rank);
    
    // 使用 where 操作根据条件张量进行条件选择，返回选择后的张量和 0
    return std::make_tuple(at::where(condition_, self_, other_), 0);
// 定义静态函数，用于计算 Gelu 函数的反向传播规则
static std::tuple<Tensor, optional<int64_t>> gelu_backward_batch_rule(
    // 输入参数：梯度输出张量、梯度输出的批维度、输入张量、输入张量的批维度、近似计算标志
    const Tensor& grad_out, optional<int64_t> grad_out_bdim, const Tensor& input, optional<int64_t> input_bdim,
    c10::string_view approximate) {

  // 重复 _binary_pointwise_batch_rule 中的预处理步骤
  const auto tensor_other = _binary_pointwise_helper(grad_out, grad_out_bdim, input, input_bdim);
  auto grad_out_ = std::get<0>(tensor_other);
  auto input_ = std::get<1>(tensor_other);

  // Gelu 反向传播不支持广播，因此我们需要确保所有输入都有批维度
  const auto batch_size = get_bdim_size2(grad_out, grad_out_bdim, input, input_bdim);
  grad_out_ = ensure_has_bdim(grad_out_, grad_out_bdim.has_value(), batch_size);
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), batch_size);

  // 调用 ATen 库中的 gelu_backward 函数计算 Gelu 函数的反向传播结果
  return std::make_tuple(at::gelu_backward(grad_out_, input_, approximate), 0);
}

// 定义静态函数，用于执行 masked_select 函数的批规则
static std::tuple<Tensor, optional<int64_t>> masked_select_batch_rule(
    // 输入参数：自身张量、自身张量的批维度、掩码张量、掩码张量的批维度
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& mask, optional<int64_t> mask_bdim) {
  // 检查是否存在掩码张量的批维度，若存在则抛出错误信息
  TORCH_CHECK(!mask_bdim.has_value(),
      "vmap: Attempted to vmap over `mask` in torch.masked_select(self, mask) ",
      "We cannot support this because for each batch this would return a ",
      "differently shaped Tensor. "
      "Please voice your support in https://github.com/pytorch/functorch/issues/256");
  
  // 将自身张量中的批维度移至最前端
  auto self_ = moveBatchDimToFront(self, self_bdim);
  // 获取批大小
  const auto batch_size = self_.size(0);
  // 计算自身张量的逻辑秩（排除批维度后的秩）
  const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  // 计算最大逻辑秩，取自身张量的逻辑秩与掩码张量的维度之间的最大值
  const auto max_logical_rank = std::max(self_logical_rank, mask.dim());
  // 可能将自身张量填充到逻辑秩的最大值
  self_ = maybePadToLogicalRank(self_, 0, max_logical_rank);

  // 调用 ATen 库中的 masked_select 函数，获取结果并将其视图变形为2D张量
  const auto result = at::masked_select(self_, mask).view({ batch_size, -1 });
  return std::make_tuple(result, 0);
}

// 定义静态函数，用于执行 masked_select 函数的反向传播批规则
static std::tuple<Tensor, optional<int64_t>> masked_select_backward_batch_rule(
    // 输入参数：梯度张量、梯度张量的批维度、自身张量、自身张量的批维度
    const Tensor& grad, optional<int64_t> grad_bdim,
    const Tensor& self, optional<int64_t> self_bdim,
    // 继续函数定义，略去部分...
    // 检查是否存在 mask_bdim 的值，如果存在则抛出错误，因为无法支持在每个批次上对 mask 进行 vmap 操作
    TORCH_CHECK(!mask_bdim.has_value(),
        "vmap: Attempted to vmap over `mask` in torch.masked_select_backward(grad, self, mask) ",
        "We cannot support this because for each batch this would return a ",
        "differently shaped Tensor. "
        "Please voice your support in https://github.com/pytorch/functorch/issues/256");
    
    // 将 self 张量的批次维度移动到最前面
    auto self_ = moveBatchDimToFront(self, self_bdim);
    
    // 将 grad 张量的批次维度移动到最前面
    auto grad_ = moveBatchDimToFront(grad, grad_bdim);
    
    // 计算排除批次维度后的 self 张量的逻辑秩（维度）
    const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
    
    // 计算 self 张量和 mask 张量的逻辑秩的最大值
    const auto max_logical_rank = std::max(self_logical_rank, mask.dim());
    
    // 可能对 self_ 进行填充，使其达到指定的逻辑秩
    self_ = maybePadToLogicalRank(self_, self_bdim, max_logical_rank);
    
    // 获取 grad 张量在指定维度上的批次大小
    const auto batch_size = get_bdim_size2(grad, grad_bdim, self, self_bdim);
    
    // 确保 self_ 张量包含指定的批次维度，如果 self_bdim 存在，则批次大小为 batch_size
    self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
    
    // 确保 grad_ 张量包含指定的批次维度，如果 grad_bdim 存在，则批次大小为 batch_size
    grad_ = ensure_has_bdim(grad_, grad_bdim.has_value(), batch_size);
    
    // 对 grad_ 和 self_.contiguous() 进行 masked_select 操作，并返回结果以及固定的零值
    const auto result = at::masked_select_backward(grad_, self_.contiguous(), mask);
    return std::make_tuple(result, 0);
}

// 批处理规则函数，计算 CDist 函数的反向传播梯度
static std::tuple<Tensor, optional<int64_t>> cdist_backward_batch_rule(
    const Tensor& grad, optional<int64_t> grad_bdim,
    const Tensor& x1, optional<int64_t> x1_bdim,
    const Tensor& x2, optional<int64_t> x2_bdim,
    const double p,
    const Tensor& cdist, optional<int64_t> cdist_bdim) {

  auto x1_ = x1;
  if (cdist_bdim && !x1_bdim) {
    // 如果 cdist 存在批次维度但 x1 没有，则确保 x1 也有批次维度
    // 否则会导致 RuntimeError，例如：Function CdistBackward0 returned an invalid gradient at index 1 - got [5] but expected shape compatible with [4, 5]
    auto bs = cdist.size(*cdist_bdim);
    x1_ = ensure_has_bdim(x1, false, bs);  // 确保 x1 有批次维度
    x1_ = x1_.contiguous();  // 使 x1 连续存储
    x1_bdim = 0;
  }

  // 需要对 x1 和 x2 进行与前向传播相同的预处理
  auto x12 = _binary_pointwise_helper(x1_, x1_bdim, x2, x2_bdim);
  x1_ = std::get<0>(x12);
  auto x2_ = std::get<1>(x12);

  auto grad_ = moveBatchDimToFront(grad, grad_bdim);
  if ((x1_bdim || x2_bdim) && !grad_bdim) {
    // 如果 x1 或 x2 其中之一有批次维度但 grad 没有，则确保 grad 也有批次维度
    // 可能存在对步长的假设
    // 否则 grad 输入可能包含无效值，如 -7.0816e+29, 7.0816e+29
    auto bs = get_bdim_size2(x1_, 0, x2_, 0);
    grad_ = ensure_has_bdim(grad_, grad_bdim.has_value(), bs);  // 确保 grad 有批次维度
    grad_ = grad_.contiguous();  // 使 grad 连续存储
  }

  // 调用 _cdist_backward 函数计算梯度传播结果
  auto out = at::_cdist_backward(grad_, x1_, x2_, p, cdist);

  optional<int64_t> out_bdim = nullopt;
  if (x1_bdim || x2_bdim) {
    out_bdim = 0;  // 如果 x1 或 x2 其中之一有批次维度，则输出结果有批次维度
  }

  return std::make_tuple(out, out_bdim);  // 返回结果和可能的批次维度
}

// 批处理规则函数，填充 Tensor 的元素
static void fill__Tensor_batch_rule(
    Tensor& self,
    optional<int64_t> self_bdim,
    const Tensor& other,
    optional<int64_t> other_bdim) {
  if (!other_bdim.has_value()) {
    // 优化：使用 fill_ 比其他路径更快，该路径包括重新形状和复制
    self.fill_(other);
    return;
  }
  if (!self_bdim && other_bdim) {
    // 如果 self 没有批次维度但 other 有，则抛出不兼容错误
    vmapIncompatibleInplaceError("fill_");
  }
  // 对 self 和 other 应用二进制点逐元素辅助函数
  auto self_and_other = _binary_pointwise_helper(
      self, self_bdim, other, other_bdim, /*do_type_promotion*/false);
  // 将 self 的数据复制为 other 的数据
  std::get<0>(self_and_other).copy_(std::get<1>(self_and_other));
}

// 批处理规则函数，计算 LogSigmoid 函数的反向传播梯度
static std::tuple<Tensor, optional<int64_t>> log_sigmoid_backward_batch_rule(
  Tensor& grad, optional<int64_t> grad_bdim,
  Tensor& self, optional<int64_t> self_bdim,
  Tensor& buffer, optional<int64_t> buffer_bdim) {
  // 注意：此处模拟了 handle_pointwise_ops，但忽略了最后一个参数 buffer
  // 当输入任意一个张量位于 cuda 上时，忽略 buffer，因为在 cuda 上，buffer 始终是逻辑秩 1 的虚拟张量
  // 当其余输入是标量时，这将成为一个问题
  int64_t out_logical_rank = std::max(rankWithoutBatchDim(grad, grad_bdim), rankWithoutBatchDim(self, self_bdim));
  if (!grad.is_cuda() && !self.is_cuda() && !buffer.is_cuda()) {
    # 计算输出的逻辑秩，取当前的 out_logical_rank 和 buffer 无批维度的秩的较大值
    out_logical_rank = std::max(out_logical_rank, rankWithoutBatchDim(buffer, buffer_bdim));
  }
  # 将梯度张量 grad 移动批维度到最前面，并可能对其进行填充以达到输出逻辑秩
  Tensor out_grad = maybePadToLogicalRank(moveBatchDimToFront(grad, grad_bdim), grad_bdim, out_logical_rank);
  # 将输入张量 self 移动批维度到最前面，并可能对其进行填充以达到输出逻辑秩
  Tensor out_self = maybePadToLogicalRank(moveBatchDimToFront(self, self_bdim), self_bdim, out_logical_rank);
  # 将缓冲张量 buffer 移动批维度到最前面，并可能对其进行填充以达到输出逻辑秩
  Tensor out_buffer = maybePadToLogicalRank(moveBatchDimToFront(buffer, buffer_bdim), buffer_bdim, out_logical_rank);
  # 返回计算后的结果，其中包括 log sigmoid 函数的反向传播结果和一个附加的标志（这里为 0）
  return std::make_tuple(at::log_sigmoid_backward(out_grad, out_self, out_buffer), 0);
}

static Tensor binomial_wrapper(const Tensor& count, const Tensor& prob, std::optional<Generator> gen) {
  // 调用PyTorch的binomial函数，生成服从二项分布的张量
  return at::binomial(count, prob.contiguous(), std::move(gen)); // Bug in PyTorch, prob shouldn't need to be contiguous
}

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  // 定义宏，用于批处理规则的二元随机点对点操作
  #define BINARY_RANDOM_POINTWISE(op) \
    m.impl(#op, BINARY_RANDOM_POINTWISE_BATCH_RULE(ATEN_FN(op)));
  // 定义宏，用于批处理规则的二元随机点对点操作，带有重载
  #define BINARY_RANDOM_POINTWISE2(op, overload) \
    m.impl(#op"."#overload, BINARY_RANDOM_POINTWISE_BATCH_RULE(ATEN_FN2(op, overload)));

  // 实现normal函数的批处理规则
  BINARY_RANDOM_POINTWISE2(normal, Tensor_Tensor);
  // 实现binomial函数的批处理规则，调用functorch中的包装器
  m.impl("binomial", BINARY_RANDOM_POINTWISE_BATCH_RULE(at::functorch::binomial_wrapper));
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 定义宏，用于批处理规则的二元点对点操作，带有两个张量参数
#define BINARY_POINTWISE2(op, overload) \
  VMAP_SUPPORT2(op, overload, BINARY_POINTWISE_BATCH_RULE(ATEN_FN2(op, overload)));
  // 定义宏，用于批处理规则的二元点对点操作，带有一个张量参数
#define BINARY_POINTWISE(op) \
  VMAP_SUPPORT(op, BINARY_POINTWISE_BATCH_RULE(ATEN_FN(op)));
  // 定义宏，用于批处理规则的一元点对点操作，带有两个张量参数
#define UNARY_POINTWISE2(op, overload) \
  VMAP_SUPPORT2(op, overload, BASIC_UNARY_BATCH_RULE(ATEN_FN2(op, overload)));
  // 定义宏，用于批处理规则的一元点对点操作，带有一个张量参数
#define UNARY_POINTWISE(op) \
  VMAP_SUPPORT(op, BASIC_UNARY_BATCH_RULE(ATEN_FN(op)));
  // 定义宏，用于批处理规则的一元点对标量操作，带有一个张量参数和一个标量参数
#define UNARY_SCALAR_POINTWISE2(op, overload) \
  VMAP_SUPPORT(op, overload, SCALAR_UNARY_BATCH_RULE(ATEN_FN2(op, overload)));

  // 定义宏，用于三种组合的二元点对标量操作：两个张量，一个张量和一个标量，一个标量和一个张量
#define BINARY_SCALAR_2(op, tensor_tensor, tensor_scalar) \
  BINARY_POINTWISE2(op, tensor_tensor);\
  UNARY_POINTWISE2(op, tensor_scalar);

  // 定义宏，用于三种组合的二元点对标量操作：两个张量，一个张量和一个标量，一个标量和一个张量
#define BINARY_SCALAR_3(op, tensor_tensor, tensor_scalar, scalar_tensor) \
  BINARY_POINTWISE2(op, tensor_tensor);\
  UNARY_POINTWISE2(op, tensor_scalar);\
  POINTWISE_BOXED(op.scalar_tensor);

  // 定义宏，用于比较操作的点对点操作
#define COMPARISON_POINTWISE(op) \
  VMAP_SUPPORT2(op, Tensor, \
      SINGLE_ARG(comparison_pointwise_batch_rule<decltype(&ATEN_FN2(op, Tensor)), &at::op>)); \
  UNARY_POINTWISE2(op, Scalar)

  // 实现相等比较的批处理规则
  COMPARISON_POINTWISE(eq);
  // 实现大于比较的批处理规则
  COMPARISON_POINTWISE(gt);
  // 实现大于等于比较的批处理规则
  COMPARISON_POINTWISE(ge);
  // 实现小于等于比较的批处理规则
  COMPARISON_POINTWISE(le);
  // 实现小于比较的批处理规则
  COMPARISON_POINTWISE(lt);
  // 实现不等比较的批处理规则
  COMPARISON_POINTWISE(ne);

#undef COMPARISON_POINTWISE
#undef BINARY_POINTWISE2
#undef BINARY_POINTWISE
#undef UNARY_POINTWISE2
#undef UNARY_POINTWISE
#undef UNARY_SCALAR_POINTWISE2
#undef BINARY_SCALAR_3

  // 定义宏，用于逻辑比较操作的点对点操作
#define LOGICAL_COMPARISON_POINTWISE(op) \
  VMAP_SUPPORT(op, \
      SINGLE_ARG(comparison_pointwise_batch_rule<decltype(&ATEN_FN(op)), &ATEN_FN(op)>)); \
  VMAP_SUPPORT(op ## _, \
      SINGLE_ARG(binary_pointwise_inplace_batch_rule<TensorInplaceT, &Tensor:: op ## _ >));

  // 实现逻辑与的批处理规则
  LOGICAL_COMPARISON_POINTWISE(logical_and);
  // 实现逻辑或的批处理规则
  LOGICAL_COMPARISON_POINTWISE(logical_or);
  // 实现逻辑异或的批处理规则
  LOGICAL_COMPARISON_POINTWISE(logical_xor);

#undef SINGLE_ARG
#undef LOGICAL_COMPARISON_POINTWISE
  // 实现masked_select函数的批处理规则
  VMAP_SUPPORT(masked_select, masked_select_batch_rule);
  // 实现masked_select_backward函数的批处理规则
  VMAP_SUPPORT(masked_select_backward, masked_select_backward_batch_rule);

  // 实现fill_函数的批处理规则
  VMAP_SUPPORT2(fill_, Tensor, fill__Tensor_batch_rule);
}

} // namespace at::functorch
```