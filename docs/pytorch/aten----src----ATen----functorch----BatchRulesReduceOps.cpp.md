# `.\pytorch\aten\src\ATen\functorch\BatchRulesReduceOps.cpp`

```
// 版权声明，版权归 Facebook, Inc. 及其关联公司所有。
// 保留所有权利。
//
// 此源代码使用 BSD 风格许可证授权，许可证文件位于源目录的 LICENSE 文件中。

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <utility>

// at::functorch 命名空间
namespace at::functorch {

// 检查维度是否在标量张量上是允许的
static bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

// 对 sum 操作进行分解，支持指定数据类型
static Tensor sum_decomp(
    const Tensor& self, optional<ScalarType> dtype) {
  return at::sum(self, range(0, self.dim()), false, dtype);
}

// 对 _is_all_true 操作的批处理规则
static std::tuple<Tensor, optional<int64_t>> _is_all_true_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim) {
  return std::make_tuple(at::_is_all_true(self), nullopt);
}

// 对 _is_any_true 操作的批处理规则
static std::tuple<Tensor, optional<int64_t>> _is_any_true_batch_rule(
     const Tensor& self, optional<int64_t> self_bdim) {
   return std::make_tuple(at::_is_any_true(self), nullopt);
 }

// 对 mean 操作进行分解，支持指定数据类型
static Tensor mean_decomp(
    const Tensor& self, optional<ScalarType> dtype) {
  return at::mean(self, range(0, self.dim()), false, dtype);
}

// 对 prod 操作进行分解，支持指定数据类型
static Tensor prod_decomp(
    const Tensor& self, optional<ScalarType> dtype) {
  return at::prod(self.flatten(), 0, false, dtype);
}

// 对 max 操作进行分解
static Tensor max_decomp(
    const Tensor& self) {
  return std::get<0>(at::max(self.flatten(), 0, false));
}

// 对 min 操作进行分解
static Tensor min_decomp(
    const Tensor& self) {
  return std::get<0>(at::min(self.flatten(), 0, false));
}

// 对 norm 操作进行分解，使用标量进行规范化
static Tensor norm_scalar_decomp(
    const Tensor& self, const Scalar& p) {
  return at::norm(self, p, range(0, self.dim()), false);
}

// 对 nanmedian 操作进行分解
static Tensor nanmedian_decomp(
    const Tensor& self) {
  return std::get<0>(at::nanmedian(self.flatten(), 0, false));
}

// 对 median 操作进行分解
static Tensor median_decomp(
    const Tensor& self) {
  return std::get<0>(at::median(self.flatten(), 0, false));
}

// 对 all 操作进行分解
static Tensor all_decomp(const Tensor& self) {
  return at::all(self.flatten(), 0, false);
}

// 对 any 操作进行分解
static Tensor any_decomp(const Tensor& self) {
  return at::any(self.flatten(), 0, false);
}

// 约简情况的枚举类型
enum class ReductionCase:uint8_t { DimArray, Dim };

// 定义 keepdim 情况的宏常量
static constexpr int KEEPDIM_CASE_FALSE = 0;
static constexpr int KEEPDIM_CASE_TRUE = 1;
static constexpr int KEEPDIM_CASE_VARIABLE = 2;

// dim_arg_pos 允许我们指定 dim/dim 数组参数的位置
//
// NOTE: [keepdim cases]
// 此操作符可能：
// - 有一个 keepdim 参数（KeepdimCase.Variable）
//   在这种情况下，`maybe_keepdim_arg_pos` 指定了 keepdim 参数的位置。
//   示例：sum(tensor, dim, keepdim)
// - 总是进行约简，没有 keepdim 参数（KeepdimCase.False）
//   即，输出张量的秩小于输入张量的秩。
// - always does a reduction with keepdim=True semantics (KeepdimCase.True)
//   That is, the rank of the output tensor is always the same as that of the input.
//   examples: log_softmax(tensor, dim), cumsum(tensor, dim)
template<
  int dim_arg_pos,
  int keepdim_case,
  // optional cannot be used in a template, otherwise we would use it here.
  int maybe_keepdim_arg_pos
>
void boxed_reduction_batch_rule(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 获取操作符的模式
  const auto& schema = op.schema();
  // 获取返回值的数量
  const auto num_returns = schema.returns().size();
  // 获取参数的数量
  const auto num_arguments = schema.arguments().size();

  // 用于排除 DispatchKey::FuncTorchBatched 的调度键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 获取当前动态层的可能性
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查动态层是否逃逸，并标记相关信息
  vmap_check_escaped(maybe_layer, "boxed_reduction_batch_rule");
  // 获取当前层级的 ID
  int64_t cur_level = maybe_layer->layerId();

  // 从栈中获取原始参数
  auto orig_arguments = torch::jit::last(*stack, num_arguments);
  // 如果原始参数中没有任何一个参与到当前层级中，则调用原始操作符并返回
  if (std::none_of(orig_arguments.begin(), orig_arguments.end(), ivalueParticipatesInCurrentLevel)) {
    c10::impl::ExcludeDispatchKeyGuard guard_2(DispatchKey::FuncTorchBatched);
    op.callBoxed(stack);
    return;
  }

  // 从栈中弹出参数
  auto arguments = torch::jit::pop(*stack, num_arguments);

  // 断言第一个参数是一个张量
  TORCH_INTERNAL_ASSERT(arguments[0].isTensor());
  // 解封第一个张量，并返回解封后的张量以及批处理维度
  auto [self, self_bdim] = unwrapTensorAtLevel(arguments[0].toTensor(), cur_level);

  // 将批处理维度移动到张量的最前面
  self = moveBatchDimToFront(self, self_bdim);

  // 计算没有批处理维度的逻辑维度
  auto logical_dim = rankWithoutBatchDim(self, self_bdim);
  std::vector<int64_t> dims;
  ReductionCase reduction_case{};
  // 根据不同的参数类型，确定减少维度的情况
  if (arguments[dim_arg_pos].isIntList()) {
    reduction_case = ReductionCase::DimArray;
    // 如果参数是整数列表，则将其转换为向量
    dims = arguments[dim_arg_pos].toIntList().vec();
    if (dims.empty()) {
      // 如果为空，则创建一个从 0 到逻辑维度减 1 的范围
      auto all_dims = range(0, std::max((int64_t)1, logical_dim));
      dims = std::vector<int64_t>(all_dims.begin(), all_dims.end());
    }
  } else if (arguments[dim_arg_pos].isInt()) {
    reduction_case = ReductionCase::Dim;
    // 如果参数是单个整数，则直接作为维度
    dims = {arguments[dim_arg_pos].toInt()};
  } else if (arguments[dim_arg_pos].isNone())  {
    auto param_type = schema.arguments()[dim_arg_pos].type()->expect<OptionalType>()->getElementType();
    if (param_type->kind() == IntType::Kind) {
      reduction_case = ReductionCase::Dim;
      // 如果张量的维度大于 1，则将其展平为一维
      if (self.dim() > 1) {
        self = self.flatten(1);
      }
      // 将维度设为 {0}
      dims = {0};
    } else if (param_type->kind() == ListType::Kind) {
      reduction_case = ReductionCase::DimArray;
      // 如果逻辑维度为 0，则设定维度为 {0}，否则为 0 到 self.dim() - 1 的范围
      if (logical_dim == 0) {
        dims = {0};
      } else {
        auto all_dims = range(0, self.dim() - 1);
        dims = std::vector<int64_t>(all_dims.begin(), all_dims.end());
      }
    } else {
      // 如果遇到意外的数据类型，则抛出错误
      TORCH_INTERNAL_ASSERT(false, "Unexpected dtype found at dims");
    }
  } else{
    // 如果遇到意外的数据类型，则抛出错误
    TORCH_INTERNAL_ASSERT(false, "Unexpected dtype found at dims");
  }

  // 创建新维度的向量
  VmapDimVector new_dims;
  new_dims.reserve(dims.size());
  for (auto dim: dims) {
    // 将处理后的物理维度推入 new_dims 中
    new_dims.push_back(getPhysicalDim(self, self_bdim.has_value(), dim));
  }
  // 判断是否为标量情况
  bool is_scalar_case = logical_dim == 0 && dims.size() == 1 && is_allowed_dim_on_scalar_tensor(dims[0]);
  // 可选布尔值，用于确定是否保持维度
  std::optional<bool> maybe_keepdim;
  if (is_scalar_case) {
    // NOTE: [boxed_reduction_batch_rule scalar tensor handling]
    // PyTorch 中的减少操作存在一种特例，允许在张量形状为 [] 时使用 dim=0 和 dim=-1。
    //
    // 这种情况可能出现在如下代码中：
    // vmap(lambda x: x.sum(0))(torch.tensor([10.])),
    //
    // 为了处理这种特例，我们在张量上插入一个维度，
    // 使用 dim=1 进行操作，然后处理输出张量。
    // 有两种情况：
    // - keepdim = True
    //     插入维度    操作      压缩维度
    //   [B] -> [B, 1] -> [B, 1] -> [B]
    // - keepdim = False
    //     插入维度    操作     无需压缩
    //   [B] -> [B, 1] -> [B]
    // 如果 keepdim 为 True，则需要压缩大小为 1 的维度。

    // 确定 keepdim 的值
    switch (keepdim_case) {
      case KEEPDIM_CASE_FALSE:
        maybe_keepdim = false;
        break;
      case KEEPDIM_CASE_TRUE:
        maybe_keepdim = true;
        break;
      case KEEPDIM_CASE_VARIABLE:
        TORCH_INTERNAL_ASSERT(maybe_keepdim_arg_pos >= 0);
        maybe_keepdim = arguments[maybe_keepdim_arg_pos].toBool();
        break;
    }
    // 在 self 张量上插入一个维度
    self = self.unsqueeze(-1);
    // 更新 new_dims 为包含一个元素的向量 [1]
    new_dims = {1};
  }
  // 将处理后的 self 张量移动到 arguments[0]
  arguments[0] = std::move(self);
  // 根据不同的减少情况，更新 arguments[dim_arg_pos]
  if (reduction_case == ReductionCase::DimArray) {
    arguments[dim_arg_pos] = std::vector<int64_t>(new_dims.begin(), new_dims.end());
  } else if (reduction_case == ReductionCase::Dim) {
    arguments[dim_arg_pos] = new_dims[0];
  }
  // 将 arguments 数组中的所有元素推入 Torch 的运算栈中
  for (const auto arg_idx : c10::irange(0, num_arguments)) {
    torch::jit::push(stack, arguments[arg_idx]);
  }
  // 调用 op 的 boxed 方法执行操作
  op.callBoxed(stack);

  // 从运算栈中弹出返回值
  const auto returns = torch::jit::pop(*stack, num_returns);
  // 遍历返回值列表
  for (const auto& ret : returns) {
    if (ret.isTensor()) {
      auto res = ret.toTensor();
      // 见 NOTE: [boxed_reduction_batch_rule scalar tensor handling]
      // 如果是标量情况且 keepdim 为 true，则进行维度压缩操作
      if (is_scalar_case && maybe_keepdim.value()) {
        // 如果 dim 的形状不为 1，则 squeeze(-1) 不会产生任何效果。
        // 为了安全起见，在这里进行内部断言。
        TORCH_INTERNAL_ASSERT(res.size(-1) == 1);
        res = res.squeeze(-1);
      }
      // 将处理后的张量 res 推入 Torch 运算栈中
      torch::jit::push(stack, makeBatched(res, 0, cur_level));
    } else {
      // 如果返回值不是张量，则抛出错误，因为该 boxed 批处理规则不支持返回非张量值
      TORCH_INTERNAL_ASSERT(false, "This boxed batching rule does not currently support ops that return non-tensor values");
    }
  }
}

// 结束函数定义的右花括号

// Skipping all/any since they don't have opinfo tests right now :P
// 由于当前没有 opinfo 测试，跳过所有/任何内容

static Tensor dist_decomp(const Tensor& self, const Tensor& other, const Scalar& p) {
  // 计算张量 self 和 other 之间的 p 范数
  return at::norm((self - other), p);
}

static std::tuple<Tensor, Tensor> expand_bdims(
    const Tensor& a, bool a_has_bdim,
    const Tensor& b, bool b_has_bdim) {
  // 定义一个临时张量 flagpole
  Tensor flagpole;
  if (a_has_bdim) {
    // 如果 a 具有批量维度，将 flagpole 设置为 a
    flagpole = a;
  } else if (b_has_bdim) {
    // 如果 b 具有批量维度，将 flagpole 设置为 b
    flagpole = b;
  } else {
    // 如果既非 a 也非 b 具有批量维度，则断言错误
    TORCH_INTERNAL_ASSERT(false);
  }
  // 返回扩展后的张量 a 和 b，如果它们没有批量维度，则根据 flagpole 进行扩展
  return std::make_tuple(
      a_has_bdim ? a : a.expand_as(flagpole),
      b_has_bdim ? b : b.expand_as(flagpole));
}

static std::tuple<Tensor,optional<int64_t>> _softmax_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& output, optional<int64_t> output_bdim,
    int64_t dim,
    ScalarType input_dtype) {
  // softmax_backward 的分解是 y * gy - y * (y * gy).sum(dim, keepdim=True)
  // 注意：CUDA 内核处理步幅，因此我们只需扩展所有张量并结束。CPU 内核性能可能不佳，但这可能并不重要。
  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  auto output_ = moveBatchDimToFront(output, output_bdim);

  // 扩展所有张量的额外维度
  std::tie(grad_output_, output_) = expand_bdims(
      grad_output_, grad_output_bdim.has_value(),
      output_, output_bdim.has_value());

  // 标量张量情况。当发生这种情况时，softmax 变成了恒等映射。
  // 我不知道为什么输出是零，但这是 softmax 告诉我的…
  if (output_.dim() == 1 && (dim == 0 || dim == -1)) {
    // 返回与 grad_output_ 相同形状的零张量和零作为批量维度
    return std::make_tuple(at::zeros_like(grad_output_), 0);
  }

  // 获取物理维度
  dim = getPhysicalDim(output_, /*has_batch_dim*/true, dim);

  // 不确定为什么 output_ 需要标记为 .contiguous()。PyTorch 中可能有些变化（softmax 输出可能总是连续的）
  // 返回 softmax 反向传播数据的结果
  return std::make_tuple(at::_softmax_backward_data(grad_output_, output_.contiguous(), dim, input_dtype), 0);
}

static std::tuple<Tensor,optional<int64_t>> _log_softmax_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& output, optional<int64_t> output_bdim,
    int64_t dim,
    // 根据给定的参数，计算 log_softmax 操作的反向传播
    auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
    // 将批处理维度移动到张量的最前面，以便后续操作
    auto output_ = moveBatchDimToFront(output, output_bdim);
    
    // 将所有张量都扩展到具有相同形状的额外维度
    std::tie(grad_output_, output_) = expand_bdims(
        grad_output_, grad_output_bdim.has_value(),
        output_, output_bdim.has_value());
    
    // 处理标量张量的特殊情况，当输出维度为1且维度dim为0或-1时，返回全零张量
    if (output_.dim() == 1 && (dim == 0 || dim == -1)) {
      return std::make_tuple(at::zeros_like(grad_output_), 0);
    }
    
    // 确定物理维度以供后续计算，考虑批处理维度的存在
    dim = getPhysicalDim(output_, /*has_batch_dim*/true, dim);
    
    // 调用 PyTorch 库函数计算 log_softmax 操作的反向传播数据
    return std::make_tuple(at::_log_softmax_backward_data(grad_output_, output_, dim, input_dtype), 0);
}

// 定义静态函数 searchsorted_batch_rule，用于在给定参数下执行搜索排序的规则
static std::tuple<Tensor, optional<int64_t>> searchsorted_batch_rule(
    const Tensor& sorted_sequence,  // 排序后的序列张量
    optional<int64_t> sorted_sequence_bdim,  // 排序后序列的批次维度（可选）
    const Tensor& self,  // 待搜索的张量
    optional<int64_t> self_bdim,  // 待搜索张量的批次维度（可选）
    bool out_int32,  // 输出是否为 int32 类型
    bool right,  // 是否使用右侧查找
    std::optional<c10::string_view> side,  // 搜索排序的侧面参数（可选）
    const std::optional<Tensor>& sorter,  // 排序器（可选）
    std::optional<int64_t> sorter_bdim) {  // 排序器的批次维度（可选）

  // 计算排序后序列的逻辑排名（排除批次维度）
  auto buckets_logical_rank = rankWithoutBatchDim(sorted_sequence, sorted_sequence_bdim);
  // 计算待搜索张量的逻辑排名（排除批次维度）
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);

  // 预处理排序器和排序后的序列
  // 如果两者都存在，并且只有一个有批次维度，则确保两者都有批次维度
  auto buckets = moveBatchDimToFront(sorted_sequence, sorted_sequence_bdim);
  optional<int64_t> buckets_bdim;
  if (sorted_sequence_bdim.has_value()) {
    buckets_bdim = 0;
  }

  optional<Tensor> sorter_;
  if (sorter.has_value() && sorter->defined()) {
    auto sorter__ = moveBatchDimToFront(*sorter, sorter_bdim);
    if (sorted_sequence_bdim.has_value() != sorter_bdim.has_value()) {
      auto bdim_size = get_bdim_size2(
          sorted_sequence, sorted_sequence_bdim,
          sorter.value(), sorter_bdim);
      sorter__ = ensure_has_bdim(sorter__, sorter_bdim.has_value(), bdim_size);
      buckets = ensure_has_bdim(buckets, sorted_sequence_bdim.has_value(), bdim_size);
      buckets_bdim = 0;
    }
    sorter_ = sorter__;
  }

  // 两种情况：buckets_logical_rank 大于 1 或等于 1
  // searchsorted 基本上是两种语义不同的操作符结合在一起
  if (buckets_logical_rank > 1) {
    // B<...>D, B<...>V -> 没有变化
    if (buckets_bdim.has_value() && self_bdim.has_value()) {
      auto self_ = moveBatchDimToFront(self, self_bdim);
      auto result = at::searchsorted(buckets, self_, out_int32, right, std::move(side), sorter_);
      return std::make_tuple(std::move(result), 0);
    }
    // B<...>D, <...>V -> B<...>D, B<...>V
    if (buckets_bdim.has_value() && !self_bdim.has_value()) {
      auto self_ = moveBatchDimToFront(self, self_bdim);
      self_ = ensure_has_bdim(self_, self_bdim.has_value(), buckets.size(0));
      auto result = at::searchsorted(buckets, self_, out_int32, right, std::move(side), sorter_);
      return std::make_tuple(std::move(result), 0);
    }
    // <...>D, B<...>V -> <...>D, <...>(BV)
    if (!buckets_bdim.has_value() && self_bdim.has_value()) {
      auto bdim_size = self.size(*self_bdim);
      auto self_ = reshape_dim_into(*self_bdim, -1, self);
      auto result = at::searchsorted(buckets, self_, out_int32, right, std::move(side), sorter_);
      result = reshape_dim_outof(-1, bdim_size, result);
      return std::make_tuple(result, result.dim() - 2);
    }
    TORCH_INTERNAL_ASSERT(false);  // 断言，表示不应该运行到这里的情况
  }
  // buckets_logical_rank == 1 的情况。
  // BD, B* -> BD, B flat(*)
  if (buckets_bdim.has_value() && self_bdim.has_value()) {
    // 将批量维度移到最前面，处理自身和自身的维度
    auto self_ = moveBatchDimToFront(self, self_bdim);
    // 根据逻辑排名将自身视图化，可能是在最后添加维度或展平
    auto self_view_ = self_logical_rank == 0 ? self_.unsqueeze(-1) : self_.flatten(1);
    // 在 buckets 中搜索 self_view_ 的位置索引，返回结果
    auto result = at::searchsorted(buckets, self_view_, out_int32, right, std::move(side), sorter_);
    // 如果逻辑排名为 0，则去除结果的最后一个维度；否则按照 self_ 的尺寸重新视图化结果
    result = self_logical_rank == 0 ? result.squeeze(-1) : result.view(self_.sizes());
    // 返回移动后的结果和零
    return std::make_tuple(std::move(result), 0);
  }
  // BD, * -> BD, flat(*) -> BD, B flat(*)
  // 如果 buckets 的维度有值且 self 的维度没有值
  if (buckets_bdim.has_value() && !self_bdim.has_value()) {
    // 获取 buckets 的指定维度大小，并确保 self 具有此维度大小
    auto bdim_size = buckets.size(*buckets_bdim);
    auto self_ = ensure_has_bdim(self, false, bdim_size);
    // 根据逻辑排名将 self 视图化，可能是在最后添加维度或展平
    auto self_view_ = self_logical_rank == 0 ? self_.unsqueeze(-1) : self_.flatten(1);
    // 在 buckets 中搜索 self_view_ 的位置索引，返回结果
    auto result = at::searchsorted(buckets, self_view_, out_int32, right, std::move(side), sorter_);
    // 如果逻辑排名为 0，则去除结果的最后一个维度；否则按照 self_ 的尺寸重新视图化结果
    result = self_logical_rank == 0 ? result.squeeze(-1) : result.view(self_.sizes());
    // 返回移动后的结果和零
    return std::make_tuple(std::move(result), 0);
  }
  // D, B* -> no change
  // 如果 buckets 的维度没有值且 self 的维度有值
  if (!buckets_bdim.has_value() && self_bdim.has_value()) {
    // 在 buckets 中搜索 self 的位置索引，返回结果
    auto result = at::searchsorted(buckets, self, out_int32, right, std::move(side), sorter_);
    // 返回结果和 self 的维度
    return std::make_tuple(std::move(result), self_bdim);
  }
  // 如果以上条件都不满足，抛出内部断言错误
  TORCH_INTERNAL_ASSERT(false);
}

// 定义静态函数 bucketize_decomp_Tensor，用于在给定边界上进行分桶操作
static Tensor bucketize_decomp_Tensor(
    const Tensor& self,         // 输入张量 self
    const Tensor& boundaries,   // 分桶边界张量 boundaries
    bool out_int32,             // 是否输出为 int32 类型的标志
    bool right) {               // 是否使用右侧闭合的标志

  // 检查边界张量的维度是否为 1，否则抛出错误
  TORCH_CHECK(boundaries.dim() == 1, "bucketize: boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  
  // 调用 ATen 库中的 searchsorted 函数进行搜索和分桶操作，返回结果张量
  return at::searchsorted(boundaries, self, out_int32, right, nullopt, nullopt);
}

// 定义静态函数 bucketize_decomp_Scalar，用于在给定边界上进行分桶操作（针对标量输入）
static Tensor bucketize_decomp_Scalar(
    const Scalar& self,         // 输入标量 self
    const Tensor& boundaries,   // 分桶边界张量 boundaries
    bool out_int32,             // 是否输出为 int32 类型的标志
    bool right) {               // 是否使用右侧闭合的标志

  // 检查边界张量的维度是否为 1，否则抛出错误
  TORCH_CHECK(boundaries.dim() == 1, "bucketize: boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  
  // 调用 ATen 库中的 searchsorted 函数进行搜索和分桶操作，返回结果张量
  return at::searchsorted(boundaries, self, out_int32, right, nullopt, nullopt);
}

// 定义宏 REDUCTION_BOXED_ARGS，用于封装带有不同参数的 reduction 函数
// - op: 操作符名称
// - dim_pos: dim 参数的位置
// - keepdim_case: keepdim 参数的情况（True, False, Variable）
// - maybe_keepdim_pos: 可能的 keepdim 参数位置，如果不存在则忽略
#define REDUCTION_BOXED_ARGS(op, dim_pos, keepdim_case, maybe_keepdim_pos) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction< \
      SINGLE_ARG(boxed_reduction_batch_rule<dim_pos, keepdim_case, maybe_keepdim_pos>)>());

// 提供方便使用的宏 REDUCTION_WITH_KEEPDIM_ARG，适用于大多数带有 keepdim 参数的操作符
#define REDUCTION_WITH_KEEPDIM_ARG(op) \
  REDUCTION_BOXED_ARGS(op, 1, KEEPDIM_CASE_VARIABLE, 2)

// 提供方便使用的宏 REDUCTION_NO_KEEPDIM_ARG，适用于大多数不带有 keepdim 参数的操作符
#define REDUCTION_NO_KEEPDIM_ARG(op) \
  REDUCTION_BOXED_ARGS(op, 1, KEEPDIM_CASE_TRUE, -1)
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 声明和注册在 aten 命名空间下的批处理函数 FuncTorchBatched

  // 启用对 searchsorted 函数在 Tensor 上的批处理支持，并使用 searchsorted_batch_rule
  VMAP_SUPPORT2(searchsorted, Tensor, searchsorted_batch_rule);
  
  // 注册 _fft_r2c 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(_fft_r2c);
  // 注册 _fft_c2r 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(_fft_c2r);
  // 注册 _fft_c2c 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(_fft_c2c);
  
  // 注册 amax 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(amax);
  // 注册 amin 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(amin);
  // 注册 aminmax 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(aminmax);
  
  // 在 m 对象中注册 "all" 函数，使用 all_decomp 作为实现
  m.impl("all", all_decomp);
  // 注册 all.dim 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(all.dim);
  // 注册 all.dims 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(all.dims);
  
  // 在 m 对象中注册 "any" 函数，使用 any_decomp 作为实现
  m.impl("any", any_decomp);
  // 注册 any.dim 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(any.dim);
  // 注册 any.dims 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(any.dims);
  
  // 注册 argmax 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(argmax);
  // 注册 argmin 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(argmin);
  
  // 在 m 对象中注册 "bucketize.Tensor" 函数，使用 bucketize_decomp_Tensor 作为实现
  m.impl("bucketize.Tensor", bucketize_decomp_Tensor);
  // 在 m 对象中注册 "bucketize.Scalar" 函数，使用 bucketize_decomp_Scalar 作为实现
  m.impl("bucketize.Scalar", bucketize_decomp_Scalar);
  
  // 注册 count_nonzero.dim_IntList 函数，1 表示不保留维度，KEEPMODE_CASE_FALSE 表示不保留 keepdim 参数，-1 表示变量参数
  REDUCTION_BOXED_ARGS(count_nonzero.dim_IntList, 1, KEEPDIM_CASE_FALSE, -1);
  
  // 注册 cummax 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(cummax);
  // 注册 cummin 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(cummin);
  // 注册 cumprod 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(cumprod);
  // 注册 cumsum 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(cumsum);
  
  // 在 m 对象中注册 "dist" 函数，使用 dist_decomp 作为实现
  m.impl("dist", dist_decomp);
  
  // 注册 kthvalue 函数，2 表示不保留维度，KEEPMODE_CASE_VARIABLE 表示保留变量 keepdim 参数，3 表示第三个参数是变量
  REDUCTION_BOXED_ARGS(kthvalue, 2, KEEPDIM_CASE_VARIABLE, 3);
  
  // 注册 linalg_vector_norm 函数，2 表示不保留维度，KEEPMODE_CASE_VARIABLE 表示保留变量 keepdim 参数，3 表示第三个参数是变量
  REDUCTION_BOXED_ARGS(linalg_vector_norm, 2, KEEPDIM_CASE_VARIABLE, 3);
  
  // 注册 logcumsumexp 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(logcumsumexp);
  // 注册 logsumexp 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(logsumexp);
  
  // 在 m 对象中注册 "max" 函数，使用 max_decomp 作为实现
  m.impl("max", max_decomp);
  // 注册 max.dim 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(max.dim);
  
  // 在 m 对象中注册 "mean" 函数，使用 mean_decomp 作为实现
  m.impl("mean", mean_decomp);
  // 注册 mean.dim 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(mean.dim);
  
  // 在 m 对象中注册 "median" 函数，使用 median_decomp 作为实现
  m.impl("median", median_decomp);
  // 注册 median.dim 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(median.dim);
  
  // 在 m 对象中注册 "min" 函数，使用 min_decomp 作为实现
  m.impl("min", min_decomp);
  // 注册 min.dim 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(min.dim);
  
  // 注册 mode 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(mode);
  
  // 在 m 对象中注册 "nanmedian" 函数，使用 nanmedian_decomp 作为实现
  m.impl("nanmedian", nanmedian_decomp);
  // 注册 nanmedian.dim 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(nanmedian.dim);
  
  // 注册 nansum 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(nansum);
  
  // 在 m 对象中注册 "norm.Scalar" 函数，使用 norm_scalar_decomp 作为实现
  m.impl("norm.Scalar", norm_scalar_decomp);
  // 注册 norm.ScalarOpt_dim 函数，2 表示不保留维度，KEEPMODE_CASE_VARIABLE 表示保留变量 keepdim 参数，3 表示第三个参数是变量
  REDUCTION_BOXED_ARGS(norm.ScalarOpt_dim, 2, KEEPDIM_CASE_VARIABLE, 3);
  
  // 在 m 对象中注册 "prod" 函数，使用 prod_decomp 作为实现
  m.impl("prod", prod_decomp);
  // 注册 prod.dim_int 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(prod.dim_int);
  
  // 注册 std.correction 函数，1 表示不保留维度，KEEPMODE_CASE_VARIABLE 表示保留变量 keepdim 参数，3 表示第三个参数是变量
  REDUCTION_BOXED_ARGS(std.correction, 1, KEEPDIM_CASE_VARIABLE, 3);
  
  // 注册 _softmax 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(_softmax);
  // 注册 sort 函数，不保留维度参数
  REDUCTION_NO_KEEPDIM_ARG(sort);
  // 注册 sort.stable 函数，2 表示不保留维度，KEEPMODE_CASE_TRUE 表示保留 keepdim 参数，-1 表示变量参数
  REDUCTION_BOXED_ARGS(sort.stable, 2, KEEPDIM_CASE_TRUE, -1);
  
  // 注册 std_mean.correction 函数，1 表示不保留维度，KEEPMODE_CASE_VARIABLE 表示保留变量 keepdim 参数，3 表示第三个参数是变量
  REDUCTION_BOXED_ARGS(std_mean.correction, 1, KEEPDIM_CASE_VARIABLE, 3);
  
  // 在 m 对象中注册 "sum" 函数，使用 sum_decomp 作为实现
  m.impl("sum", sum_decomp);
  // 注册 sum.dim_IntList 函数，保留维度参数
  REDUCTION_WITH_KEEPDIM_ARG(sum.dim_IntList);
  
  // 注册 topk 函数，2 表示不保留维度，KEEPMODE_CASE_TRUE 表示保留 keepdim 参数，-1 表示变量参数
  REDUCTION_BOXED_ARGS(topk,
```