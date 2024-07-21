# `.\pytorch\aten\src\ATen\functorch\BatchRulesModules.cpp`

```py
// 包含 ATen 函数的头文件，用于张量操作
// 包含 Functorch 库的辅助函数和帮助类
// 包含调度器的头文件，用于分发函数调用

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/core/dispatch/Dispatcher.h>

// 引入标准库的工具
#include <utility>

// 进入 functorch 命名空间，ATen 的扩展
namespace at::functorch {

// 定义一个静态函数，根据输入的索引、批次大小和嵌入数返回一个步骤张量
static Tensor getStepTensor(const Tensor& indices, const c10::SymInt& bdim_size, const c10::SymInt& num_embeddings) {
  // 创建一个视图形状的向量，所有维度除第一个外都为1
  c10::SymDimVector view_shape(indices.dim(), 1);
  view_shape[0] = bdim_size;
  // 使用 arange 函数生成一个张量范围，用于索引
  auto range = at::arange(0, bdim_size * num_embeddings, num_embeddings, indices.options());
  // 返回重塑后的步骤张量
  return range.view_symint(view_shape);
}

// 定义嵌入操作的批处理规则，处理不同维度的嵌入操作
static std::tuple<Tensor,optional<int64_t>> embedding_batch_rule(
    const Tensor& weight, optional<int64_t> weight_bdim,
    const Tensor& indices, optional<int64_t> indices_bdim,
    c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  // 如果权重维度为空而索引维度不为空，执行以下操作
  if (!weight_bdim && indices_bdim) {
    // B*, ED -> B*D
    auto result = at::embedding_symint(weight, indices, std::move(padding_idx), scale_grad_by_freq, sparse);
    return std::make_tuple(std::move(result), indices_bdim);
  }
  // 如果权重维度不为空而索引维度为空，执行以下操作
  else if (weight_bdim && !indices_bdim) {
    // *, BED -> *, E(BD) -> *(BD) -> *BD
    // 获取批次大小并将权重张量的维度重塑为批次与嵌入维度
    const auto batch_size = weight.size(*weight_bdim);
    const auto weight_ = reshape_dim_into(*weight_bdim, /*embedding_dim*/1, weight);
    auto result = at::embedding_symint(weight_, indices, std::move(padding_idx), scale_grad_by_freq, sparse);
    // 将结果重塑为输出的维度
    result = reshape_dim_outof(-1, batch_size, result);
    return std::make_tuple(result, result.dim() - 2);
  }
  // 如果权重和索引维度均不为空，执行以下操作
  TORCH_INTERNAL_ASSERT(weight_bdim && indices_bdim);
  // B*, BED -> B*, (BE)D -> B*D
  // 需要额外的操作：将步骤添加到索引中
  const auto batch_size = weight.size(*weight_bdim);
  const auto num_embeddings = weight.size((*weight_bdim == 0) ? 1 : 0);
  const auto weight_ = reshape_dim_into(*weight_bdim, 0, weight);
  auto indices_ = moveBatchDimToFront(indices, indices_bdim);

  // 获取步骤张量并添加到索引上
  const auto range = getStepTensor(indices, batch_size, num_embeddings);
  indices_ = indices_ + range;
  // 执行嵌入操作并返回结果
  auto result = at::embedding_symint(weight_, indices_, std::move(padding_idx), scale_grad_by_freq, sparse);
  return std::make_tuple(std::move(result), 0);
}

// 定义稠密嵌入反向传播的批处理规则
static std::tuple<Tensor,optional<int64_t>>
embedding_dense_backward_batch_rule(
    const Tensor& grad_, optional<int64_t> grad_bdim,
    const Tensor& indices_, optional<int64_t> indices_bdim,
    c10::SymInt num_weights, c10::SymInt padding_idx, bool scale_grad_by_freq) {
  // 复制输入张量以进行修改
  Tensor grad = grad_;
  Tensor indices = indices_;
  // 如果索引维度为空而梯度维度不为空，执行以下操作
  if (!indices_bdim && grad_bdim) {
    // 获取梯度维度大小并将梯度张量的维度重塑为-1
    const auto bdim_size = grad.sym_size(*grad_bdim);
    grad = reshape_dim_into(*grad_bdim, -1, grad);
    auto result = at::embedding_dense_backward_symint(
        grad, indices, std::move(num_weights), std::move(padding_idx), scale_grad_by_freq);
    // 调用 reshape_dim_outof_symint 函数重新整形 result 变量的维度，使得第一个维度为 1
    result = reshape_dim_outof_symint(1, bdim_size, result);
    // 返回一个包含 result 和整数 1 的元组
    return std::make_tuple(std::move(result), 1);
  }
  // 计算 indices 的 bdim_size（批次维度大小）
  const auto bdim_size = indices.size(*indices_bdim);
  // 将 indices 的批次维度移动到最前面
  indices = moveBatchDimToFront(indices, indices_bdim);
  // 将 grad 的批次维度移动到最前面
  grad = moveBatchDimToFront(grad, grad_bdim);
  // 确保 grad 张量有批次维度，如果 grad_bdim 有值，则使用 bdim_size
  grad = ensure_has_bdim(grad, grad_bdim.has_value(), bdim_size);
  // 获取用于步长的张量 range
  const auto range = getStepTensor(indices, bdim_size, num_weights);
  // 调用 at::embedding_dense_backward_symint 函数进行反向传播计算
  auto result = at::embedding_dense_backward_symint(
      grad, indices + range, num_weights * bdim_size, -1, scale_grad_by_freq);
  // 调用 reshape_dim_outof 函数重新整形 result 张量的维度，使得第一个维度为 0
  result = reshape_dim_outof(0, bdim_size, result);
  // 填充填充值。无法在 embedding_dense_backward 调用中完成，因为需要填充多行！
  if (padding_idx >= 0) {
    // 选择 result 张量中的 padding_idx 行，并将其填充为 0
    result.select_symint(1, std::move(padding_idx)).fill_(0);
  }
  // 返回一个包含 result 和整数 0 的元组
  return std::make_tuple(std::move(result), 0);
/**
 * grid sample batch rule breaks down into 3 cases:
 *   case 1 (input is batched, grid is not):
 *     batch input along first dimension, unpack along first dimension
 *     2d:
 *       input: N(BC)H_{in}W_{in}, grid: NH_{out}W_{out}2
 *       output: N(BC)H_{out}W_{out}
 *     3d:
 *       input: N(BC)D_{in}H_{in}W_{in}, grid: ND_{out}H_{out}W_{out}3
 *       output: N(BC)D_{out}H_{out}W_{out}
 *   case 2 (input is not batched, grid is batched):
 *     batch grid along second dimension, unpack along second dimension
 *     2d:
 *       input: NCH_{in}W_{in}, grid: N(BH_{out})W_{out}2
 *       output: NC(BH_{out})W_{out}
 *     3d:
 *       input: NCD_{in}H_{in}W_{in}, grid: N(BD_{out})H_{out}W_{out}3
 *       output: NC(BD_{out})H_{out}W_{out}
 *   case 3 (input and grid are both batched):
 *     batch grid and input along 0th dimension, unpack along 0th dimension
 *     2d:
 *       input: (BN)CH_{in}W_{in}, grid: (BN)H_{out}W_{out}2
 *       output: (BN)CH_{out}W_{out}
 *     3d:
 *       input: (BN)CD_{in}H_{in}W_{in}, grid: (BN)D_{out}H_{out}W_{out}3
 *       output: (BN)CD_{out}H_{out}W_{out}
 */
template<typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor, optional<int64_t>>
grid_sample_batch_rule(const Tensor& input, optional<int64_t> input_bdim, const Tensor& grid, optional<int64_t> grid_bdim, ExtraArgs... extra_args) {
  std::tuple<Tensor, optional<int64_t>> result;
  
  // Case 1: Input is batched, grid is not batched
  if (input_bdim && !grid_bdim) {
    // Reshape input tensor to insert batch dimension
    auto new_input = reshape_dim_into(*input_bdim, 1, input);
    // Apply the function with the reshaped input and grid
    auto out = Func(new_input, grid, std::forward<ExtraArgs>(extra_args)...);
    // Reshape the output to remove the added batch dimension
    out = reshape_dim_outof(1, input.sizes()[*input_bdim], out);
    // Store result in tuple with corresponding case identifier
    result = std::make_tuple(std::move(out), 1);
  
  // Case 2: Input is not batched, grid is batched
  } else if (!input_bdim && grid_bdim) {
    // Reshape grid tensor to insert batch dimension
    auto new_grid = reshape_dim_into(*grid_bdim, 1, grid);
    // Apply the function with the original input and reshaped grid
    auto out = Func(input, new_grid, std::forward<ExtraArgs>(extra_args)...);
    // Reshape the output to adjust for the grid batch dimension
    out = reshape_dim_outof(2, grid.sizes()[*grid_bdim], out);
    // Store result in tuple with corresponding case identifier
    result = std::make_tuple(std::move(out), 2);
  
  // Case 3: Both input and grid are batched
  } else if (input_bdim && grid_bdim) {
    // Reshape input and grid tensors to insert batch dimension
    auto new_input = reshape_dim_into(*input_bdim, 0, input);
    auto new_grid = reshape_dim_into(*grid_bdim, 0, grid);
    // Apply the function with both tensors reshaped
    auto out = Func(new_input, new_grid, std::forward<ExtraArgs>(extra_args)...);
    // Reshape the output to adjust for the input batch dimension
    out = reshape_dim_outof(0, input.sizes()[*grid_bdim], out);
    // Store result in tuple with corresponding case identifier
    result = std::make_tuple(std::move(out), 0);
  
  // Case 4: Neither input nor grid is batched
  } else {
    // Apply the function directly with the original input and grid
    result = std::make_tuple(Func(input, grid, std::forward<ExtraArgs>(extra_args)...), nullopt);
  }
  
  // Return the resulting tuple
  return result;
}
  // 获取输入的梯度输出张量的批大小
  auto batch_size = get_bdim_size3(
      grad_output, grad_output_bdim, input, input_bdim, grid, grid_bdim);

  // 将梯度输出张量的批维度移至最前端
  auto grad_output_ = moveBatchDimToFront(grad_output, grad_output_bdim);
  // 确保梯度输出张量具有批维度，如果未指定批维度，则使用批大小
  grad_output_ = ensure_has_bdim(grad_output_, grad_output_bdim.has_value(), batch_size);
  // 重塑梯度输出张量的维度为期望的形状
  grad_output_ = reshape_dim_into(0, 0, grad_output_);

  // 将输入张量的批维度移至最前端
  auto input_ = moveBatchDimToFront(input, input_bdim);
  // 确保输入张量具有批维度，如果未指定批维度，则使用批大小
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), batch_size);
  // 重塑输入张量的维度为期望的形状
  input_ = reshape_dim_into(0, 0, input_);

  // 将网格张量的批维度移至最前端
  auto grid_ = moveBatchDimToFront(grid, grid_bdim);
  // 确保网格张量具有批维度，如果未指定批维度，则使用批大小
  grid_ = ensure_has_bdim(grid_, grid_bdim.has_value(), batch_size);
  // 重塑网格张量的维度为期望的形状
  grid_ = reshape_dim_into(0, 0, grid_);

  // 返回移动后的张量以及计算得到的批大小的元组
  return std::make_tuple(std::move(grad_output_), std::move(input_), std::move(grid_), batch_size);
}

// grid_sample_backward_helper_out 函数的定义，用于处理 grid_sample_backward_batch_rule 和 cudnn_grid_sample_backward_batch_rule 函数的输出
static std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>>
grid_sample_backward_helper_out(
    const std::tuple<Tensor, Tensor> & bw_out,
    optional<int64_t> grad_input_out_bdim,
    optional<int64_t> grad_grid_out_bdim,
    int64_t bdim_size) {
  auto grad_input = std::get<0>(bw_out); // 获取 bw_out 中的梯度输入张量
  auto grad_grid = std::get<1>(bw_out);  // 获取 bw_out 中的梯度网格张量
  grad_input = reshape_dim_outof(*grad_input_out_bdim, bdim_size, grad_input); // 重新整形梯度输入张量的特定维度
  grad_grid = reshape_dim_outof(*grad_grid_out_bdim, bdim_size, grad_grid);    // 重新整形梯度网格张量的特定维度
  auto result = std::make_tuple(grad_input, grad_input_out_bdim, grad_grid, grad_grid_out_bdim); // 构造结果元组
  return result;  // 返回结果元组
}

// grid_sample_backward_batch_rule 模板函数，用于执行梯度回传规则
template<typename F, F Func, typename... ExtraArgs>
std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>>
grid_sample_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& grid, optional<int64_t> grid_bdim,
    ExtraArgs... extra_args) {

  auto new_bw_input = grid_sample_backward_helper_in(
      grad_output, grad_output_bdim, input, input_bdim, grid, grid_bdim); // 调用辅助函数获取新的梯度输入

  auto new_grad_output = std::get<0>(new_bw_input);  // 获取新的梯度输出
  auto new_input = std::get<1>(new_bw_input);         // 获取新的输入张量
  auto new_grid = std::get<2>(new_bw_input);          // 获取新的网格张量
  int64_t batch_size = std::get<3>(new_bw_input);     // 获取批量大小

  auto bw_out = Func(new_grad_output, new_input, new_grid, std::forward<ExtraArgs>(extra_args)...); // 调用传入的函数处理新的输入输出

  return grid_sample_backward_helper_out(bw_out, 0, 0, batch_size); // 调用辅助函数处理输出结果
}

// cudnn_grid_sample_backward_batch_rule 模板函数，用于执行 cudnn 加速的梯度回传规则
template<typename F, F Func>
std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>>
cudnn_grid_sample_backward_batch_rule(
    const Tensor& input, optional<int64_t> input_bdim,
    const Tensor& grid, optional<int64_t> grid_bdim,
    const Tensor& grad_output, optional<int64_t> grad_output_bdim) {

  auto new_bw_input = grid_sample_backward_helper_in(
      grad_output, grad_output_bdim, input, input_bdim, grid, grid_bdim); // 调用辅助函数获取新的梯度输入

  auto new_grad_output = std::get<0>(new_bw_input);  // 获取新的梯度输出
  auto new_input = std::get<1>(new_bw_input);         // 获取新的输入张量
  auto new_grid = std::get<2>(new_bw_input);          // 获取新的网格张量
  int64_t bdim_size = std::get<3>(new_bw_input);      // 获取批量大小

  auto bw_out = Func(new_input, new_grid, new_grad_output); // 调用传入的函数处理新的输入输出

  return grid_sample_backward_helper_out(bw_out, 0, 0, bdim_size); // 调用辅助函数处理输出结果
}

// one_hot_decomposition_hack 函数的定义，用于在索引张量上执行 one-hot 编码的处理
// 该函数用于特殊情况，如索引张量为空时的处理
static Tensor one_hot_decomposition_hack(const Tensor &self, int64_t num_classes) {
    TORCH_CHECK(self.dtype() == kLong, "one_hot is only applicable to index tensor."); // 检查张量类型是否为长整型
    auto shape = self.sym_sizes().vec(); // 获取张量的形状

    // 空张量可以转换为 one-hot 表示，
    // 但是无法进行形状推断。
    if (self.sym_numel() == 0) { // 如果张量元素数为0
        if (num_classes <= 0) { // 如果类别数小于等于0，报错
            AT_ERROR("Can not infer total number of classes from empty tensor.");
        } else { // 否则根据给定的类别数生成空的 one-hot 张量
            shape.push_back(num_classes);
            return at::empty_symint(shape, self.options());
        }
    }
    # 检查 num_classes 是否大于 0，如果不是则抛出错误信息
    TORCH_CHECK(num_classes > 0, "When vmap-ing torch.nn.functional.one_hot, please "
        "provide an explicit positive num_classes argument.");

    // Disabling all of the following checks. This is OK because scatter has checks too.
    // Maybe one_hot should be a primitive wrt autograd so we don't have to deal with this.
    // // non-empty tensor
    // if (self.device().type() != at::kCUDA) {
    //   //for cuda, rely on device assert thrown by scatter
    //   TORCH_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");
    // }
    // if (self.device().type() != at::kCUDA) {
    //   //rely on device asserts from scatter to avoid sync here
    //   TORCH_CHECK(num_classes > self.max().item().toLong(), "Class values must be smaller than num_classes.");
    // }

    # 将 num_classes 添加到 shape 的末尾，用于构造全零的 Tensor
    shape.push_back(num_classes);
    # 创建一个形状为 shape 的全零 Tensor，使用与 self 相同的选项
    Tensor ret = at::zeros_symint(shape, self.options());
    # 使用 scatter 方法在 ret 上进行操作，将 self 扩展为列向量，填充 1
    return ret.scatter(-1, self.unsqueeze(-1), 1);
// 结构模板：UpsampleBackwardBatchRuleHelper，用于处理上采样操作的反向传播规则
template <typename A, A a, typename C>
struct UpsampleBackwardBatchRuleHelper;

// 结构模板特化：UpsampleBackwardBatchRuleHelper，用于具体类型列表<T1, T2, T...>的情况
template <typename F, F Func, typename A, typename B, typename C, typename... T>
struct UpsampleBackwardBatchRuleHelper<F, Func, typelist<A, B, C, T...>> {
  // 静态函数：apply，处理反向传播操作
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& grad_output, optional<int64_t> grad_output_bdim,
      c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size,
      T... extra_args) {
    // 将 grad_output 根据 grad_output_bdim 重塑到第0维
    auto grad_output_ = reshape_dim_into(*grad_output_bdim, 0, grad_output);
    // 内部断言：确保 input_size 不为空
    TORCH_INTERNAL_ASSERT(!input_size.empty());

    // 物理尺寸的输入大小：将 input_size 转换为物理尺寸
    c10::SymDimVector physical_input_size(input_size.begin(), input_size.end());
    physical_input_size[0] = grad_output_.sym_sizes()[0];

    // 调用 Func 函数处理上采样的反向传播操作
    auto out = Func(
        grad_output_,
        output_size,
        physical_input_size,
        std::forward<T>(extra_args)...);
    // 返回结果元组，将 out 按照第0维的符号大小重新塑形
    return std::make_tuple(reshape_dim_outof_symint(0, grad_output.sym_sizes()[*grad_output_bdim], out), 0);
  }

};

// 结构模板：GridSampleBatchRuleHelper，用于处理网格采样的批处理规则
template <typename A, A a, typename C>
struct GridSampleBatchRuleHelper;

// 结构模板特化：GridSampleBatchRuleHelper，用于具体类型列表<T1, T2, T...>的情况
template <typename F, F Func, typename T1, typename T2, typename... T>
struct GridSampleBatchRuleHelper<F, Func, typelist<T1, T2, T...>> {
  // 静态函数：apply，处理网格采样的批处理规则
  static std::tuple<Tensor,optional<int64_t>> apply(
      const Tensor& input, optional<int64_t> input_batch_dim,
      const Tensor& grid, optional<int64_t> grid_batch_dim,
      T... extra_args) {
    // 调用 grid_sample_batch_rule 函数处理网格采样的批处理规则
    return grid_sample_batch_rule<F, Func, T...>(
        input, input_batch_dim, grid, grid_batch_dim, std::forward<T>(extra_args)...);
  }
};

// 结构模板：GridSampleBackwardBatchRuleHelper，用于处理网格采样的反向传播批处理规则
template <typename A, A a, typename C>
struct GridSampleBackwardBatchRuleHelper;

// 结构模板特化：GridSampleBackwardBatchRuleHelper，用于具体类型列表<T1, T2, T3, T...>的情况
template <typename F, F Func, typename T1, typename T2, typename T3, typename... T>
struct GridSampleBackwardBatchRuleHelper<F, Func, typelist<T1, T2, T3, T...>> {
  // 静态函数：apply，处理网格采样的反向传播批处理规则
  static std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>> apply(
      const Tensor& grad_output, optional<int64_t> grad_output_batch_dim,
      const Tensor& input, optional<int64_t> input_batch_dim,
      const Tensor& grid, optional<int64_t> grid_batch_dim,
      T... extra_args) {
    // 调用 grid_sample_backward_batch_rule 函数处理网格采样的反向传播批处理规则
    return grid_sample_backward_batch_rule<F, Func, T...>(
        grad_output, grad_output_batch_dim,
        input, input_batch_dim,
        grid, grid_batch_dim,
        std::forward<T>(extra_args)...);
  }
};

// 结构模板：CudnnGridSampleBackwardBatchRuleHelper，用于处理 CUDNN 网格采样的反向传播批处理规则
template <typename F, F Func>
struct CudnnGridSampleBackwardBatchRuleHelper {
  // 静态函数：apply，处理 CUDNN 网格采样的反向传播批处理规则
  static std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>> apply(
      const Tensor& input, optional<int64_t> input_batch_dim,
      const Tensor& grid, optional<int64_t> grid_batch_dim,
      const Tensor& grad_output, optional<int64_t> grad_output_batch_dim) {
    // 调用 cudnn_grid_sample_backward_batch_rule 函数处理 CUDNN 网格采样的反向传播批处理规则
    return cudnn_grid_sample_backward_batch_rule<F, Func>(
        input, input_batch_dim,
        grid, grid_batch_dim,
        grad_output, grad_output_batch_dim
    );
  }
};

// 宏定义：GRID_SAMPLE_BATCH_RULE(fn)，用于生成单参数的宏定义
#define GRID_SAMPLE_BATCH_RULE(fn) SINGLE_ARG(\
    // 使用 GridSampleBatchRuleHelper 模板辅助类的 apply 方法，该方法接受以下参数：
    //   - decltype(&ATEN_FN(fn)): 函数指针的类型，指向 ATEN_FN(fn) 函数
    //   - &ATEN_FN(fn): 实际的函数指针，指向 fn 函数
    //   - c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types: 函数参数的类型，由函数特征工具提取
    GridSampleBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn),\
      c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types>::apply)
#define GRID_SAMPLE_BW_BATCH_RULE(fn) SINGLE_ARG(\
    GridSampleBackwardBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn),\
      c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types>::apply)

定义宏 `GRID_SAMPLE_BW_BATCH_RULE(fn)`，它将函数 `fn` 的反向批处理规则包装成一个参数的单一参数。


#define CUDNN_GRID_SAMPLE_BW_BATCH_RULE(fn)\
    CudnnGridSampleBackwardBatchRuleHelper<decltype(&ATEN_FN(fn)), &ATEN_FN(fn)>::apply

定义宏 `CUDNN_GRID_SAMPLE_BW_BATCH_RULE(fn)`，它使用 CUDNN 的库将函数 `fn` 的反向批处理规则应用于网格采样。


#define UPSAMPLE_BACKWARD(op) VMAP_SUPPORT(op, SINGLE_ARG(\
    UpsampleBackwardBatchRuleHelper<\
      decltype(&ATEN_FN(op)),\
      &ATEN_FN(op),\
      c10::guts::function_traits<decltype(ATEN_FN(op))>::parameter_types>::apply))

定义宏 `UPSAMPLE_BACKWARD(op)`，它通过支持变量映射（VMAP_SUPPORT）来应用函数 `op` 的反向批处理规则，用于上采样操作。


TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  EXISTING_BDIM(im2col);
  EXISTING_BDIM(col2im);

  VMAP_SUPPORT(embedding, embedding_batch_rule);
  VMAP_SUPPORT(embedding_dense_backward, embedding_dense_backward_batch_rule);

  VMAP_SUPPORT(grid_sampler_2d, GRID_SAMPLE_BATCH_RULE(grid_sampler));
  VMAP_SUPPORT(grid_sampler_2d_backward, GRID_SAMPLE_BW_BATCH_RULE(grid_sampler_2d_backward));

  VMAP_SUPPORT(grid_sampler_3d, GRID_SAMPLE_BATCH_RULE(grid_sampler));
  VMAP_SUPPORT(grid_sampler_3d_backward, GRID_SAMPLE_BW_BATCH_RULE(grid_sampler_3d_backward));
  VMAP_SUPPORT(cudnn_grid_sampler_backward, CUDNN_GRID_SAMPLE_BW_BATCH_RULE(cudnn_grid_sampler_backward));

  VMAP_SUPPORT(cudnn_grid_sampler, GRID_SAMPLE_BATCH_RULE(cudnn_grid_sampler));

  EXISTING_BDIM(pixel_shuffle);
  EXISTING_BDIM(pixel_unshuffle);

  VARIADIC_BDIMS(constant_pad_nd);
  EXISTING_BDIM(reflection_pad1d);
  EXISTING_BDIM(reflection_pad2d);
  EXISTING_BDIM(reflection_pad3d);
  EXISTING_BDIM(replication_pad1d);
  EXISTING_BDIM(replication_pad2d);
  EXISTING_BDIM(replication_pad3d);

  EXISTING_BDIM_ALL_BOXED(replication_pad1d_backward);
  EXISTING_BDIM_ALL_BOXED(replication_pad2d_backward);
  EXISTING_BDIM_ALL_BOXED(replication_pad3d_backward);

  EXISTING_BDIM_ALL_BOXED(reflection_pad1d_backward);
  EXISTING_BDIM_ALL_BOXED(reflection_pad2d_backward);
  EXISTING_BDIM_ALL_BOXED(reflection_pad3d_backward);

  EXISTING_BDIM(upsample_bicubic2d);
  EXISTING_BDIM(upsample_bilinear2d);
  EXISTING_BDIM(upsample_linear1d);
  EXISTING_BDIM(upsample_nearest1d);
  EXISTING_BDIM(upsample_nearest2d);
  EXISTING_BDIM(upsample_nearest3d);
  EXISTING_BDIM(upsample_trilinear3d);
  EXISTING_BDIM(_upsample_bilinear2d_aa);
  EXISTING_BDIM(_upsample_bicubic2d_aa);

  UPSAMPLE_BACKWARD(upsample_bicubic2d_backward);
  UPSAMPLE_BACKWARD(upsample_bilinear2d_backward);
  UPSAMPLE_BACKWARD(upsample_linear1d_backward);
  UPSAMPLE_BACKWARD(upsample_nearest1d_backward);
  UPSAMPLE_BACKWARD(upsample_nearest2d_backward);
  UPSAMPLE_BACKWARD(upsample_nearest3d_backward);
  UPSAMPLE_BACKWARD(upsample_trilinear3d_backward);
  UPSAMPLE_BACKWARD(_upsample_bilinear2d_aa_backward);
  UPSAMPLE_BACKWARD(_upsample_bicubic2d_aa_backward);

  m.impl("one_hot", one_hot_decomposition_hack);
}

TORCH 库实现中的代码块，注册了各种功能以支持批处理和变量映射操作。具体包括注册现有的操作 `EXISTING_BDIM` 和 `EXISTING_BDIM_ALL_BOXED`，以及使用宏定义的批处理规则和反向规则函数。
```