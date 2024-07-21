# `.\pytorch\aten\src\ATen\functorch\BatchRulesLinearAlgebra.cpp`

```
// 声明命名空间 at::functorch 内部的类型别名
// 一个输出的类型别名，包含一个张量和一个可选的整数
typedef std::tuple<Tensor, optional<int64_t>> oneOutput;
// 两个输出的类型别名，包含两个张量和各自的可选整数
typedef std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>> twoOutputs;
// 三个输出的类型别名，包含三个张量和各自的可选整数
typedef std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>, Tensor, optional<int64_t>> threeOutputs;
// 四个输出的类型别名，包含四个张量和各自的可选整数
typedef std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>, Tensor, optional<int64_t>, Tensor, optional<int64_t>> fourOutputs;

// 声明一个匿名命名空间，用于定义 matmul-like 运算的批处理规则
namespace {

// 注释 [Batching rules for matmul-like operators]
// at::matmul 不会对参数进行“反展开”以提高性能（也许应该这样做）。
// 在 matmul-like 运算（如 dot、mv、mm）的批处理规则中，我们应该小心，不要将不必要的维度展开。
// 即，如果两个参数中只有一个是 BatchedTensor，则尽量不要将批次维度展开到另一个参数上。

// 定义 dot 运算的批处理规则，接受两个张量和各自的可选批次维度作为参数
std::tuple<Tensor, optional<int64_t>> dot_batch_rule(const Tensor& A, optional<int64_t> A_bdim, const Tensor& B, optional<int64_t> B_bdim) {
  // 检查张量 A 和 B 的维度与可选批次维度是否符合 dot 运算的要求
  TORCH_CHECK(A.dim() - A_bdim.has_value() == 1 && B.dim() - B_bdim.has_value() == 1, "Got wrong shapes for dot");
  // 将具有批次维度的张量 A 和 B 移动到最前面
  auto A_ = moveBatchDimToFront(A, A_bdim);
  auto B_ = moveBatchDimToFront(B, B_bdim);
  // 如果 A_bdim 和 B_bdim 均存在，则进行批处理后的 matmul 操作
  if (A_bdim && B_bdim) {
    return std::make_tuple(at::matmul(A_.unsqueeze(-2), B_.unsqueeze(-1)).squeeze(-1).squeeze(-1), 0);
  } else {
    // 否则，进行普通的 matmul 操作
    return std::make_tuple(at::matmul(A_, B_.t()), 0);
  }
}

// 定义 vdot 运算的分解函数，接受两个张量作为参数
Tensor vdot_decomp(const Tensor& A, const Tensor& B) {
  // 如果张量 A 是复数类型，使用其共轭；否则使用原始张量 A 进行 dot 运算
  return at::dot(A.is_complex() ? A.conj() : A, B);
}

// 定义 tv 运算的批处理规则，接受两个张量和各自的可选批次维度作为参数
static std::tuple<Tensor, optional<int64_t>> tv_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  // 如果 self_bdim 和 other_bdim 均存在
  if (self_bdim && other_bdim) {
    // 对 self 进行维度调整，将批次维度移到倒数第三个位置
    auto self_ = at::movedim(self, *self_bdim, -3);
    // 将 other 的批次维度移到最前面
    auto other_ = moveBatchDimToFront(other, other_bdim);
    // 对 other 进行维度扩展
    other_ = other_.unsqueeze(-1);
    // 执行 matmul 运算并进行维度压缩
    auto result = at::matmul(self_, other_).squeeze(-1);
    auto result_bdim = result.dim() - 2;
    return std::make_tuple( std::move(result), result_bdim );
  }
  // 如果只有 self_bdim 存在
  else if (self_bdim && !other_bdim) {
    // 将 self 的批次维度移到最前面，然后与 other 执行 matmul 运算
    auto self_ = moveBatchDimToFront(self, self_bdim);
    return std::make_tuple( at::matmul(self_, other), 0 );
  }
  // 如果只有 other_bdim 存在
  else if (!self_bdim && other_bdim) {
    // 将 other 的批次维度移到最后一个位置，然后与 self 执行 matmul 运算
    auto other_ = at::movedim(other, *other_bdim, -1);
    auto result = at::matmul(self, other_);
    return std::make_tuple( std::move(result), 1 );
  }
  // 如果以上情况均不满足，抛出内部断言错误
  TORCH_INTERNAL_ASSERT(false, "can't get here");
}

} // end namespace
// 定义静态函数 mv_batch_rule，处理矩阵向量乘法的批处理规则
static std::tuple<Tensor, optional<int64_t>> mv_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  
  // 获取去除批处理维度后的逻辑维度
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
  
  // 检查矩阵向量乘法的维度是否符合预期
  TORCH_CHECK(self_logical_rank == 2 && other_logical_rank == 1,
      "Shape mismatch: ",
      "Got incorrect dims for mv(a, b). a has dim ", self_logical_rank,
      "and b has dim ", other_logical_rank,
      "but expected them to have dim 2 and dim 1");
  
  // 调用 tv_batch_rule 处理矩阵向量乘法的批处理规则
  return tv_batch_rule(self, self_bdim, other, other_bdim);
}

// 定义静态函数 mm_batch_rule，处理矩阵乘法的批处理规则
static std::tuple<Tensor, optional<int64_t>> mm_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  
  // 获取去除批处理维度后的逻辑维度
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
  
  // 检查矩阵乘法的维度是否符合预期
  TORCH_CHECK(self_logical_rank == 2 && other_logical_rank == 2,
      "Shape mismatch: Got incorrect dims for mm(a, b). "
      "a has dim ", self_logical_rank,
      "and b has dim ", other_logical_rank,
      "but expected them to have dim 2 and dim 2");
  
  // 将批处理维度移到矩阵的最前面，处理矩阵乘法的批处理规则
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(other, other_bdim);
  
  // 调用 matmul 函数进行矩阵乘法，并返回结果
  return std::make_tuple( at::matmul(self_, other_), 0 );
}

// 定义静态函数 bmm_batch_rule，处理批量矩阵乘法的批处理规则
static std::tuple<Tensor, optional<int64_t>> bmm_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  
  // 获取去除批处理维度后的逻辑维度
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
  
  // 检查批量矩阵乘法的维度是否符合预期
  TORCH_CHECK(self_logical_rank == 3 && other_logical_rank == 3,
      "Shape mismatch: Got incorrect dims for bmm(a, b). "
      "a has dim ", self_logical_rank,
      "and b has dim ", other_logical_rank,
      "but expected them to have dim 3 and dim 3");
  
  // 将批处理维度移到矩阵的最前面，处理批量矩阵乘法的批处理规则
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(other, other_bdim);
  
  // 调用 matmul 函数进行批量矩阵乘法，并返回结果
  return std::make_tuple( at::matmul(self_, other_), 0 );
}

// 对于 addmv 函数的分解实现，不支持批处理，因此对其进行逐元素操作
Tensor addmv_decomp(
  const Tensor& input, const Tensor& mat, const Tensor& vec, const Scalar& beta, const Scalar& alpha) {
  
  // 计算 mv 函数的结果
  Tensor out = at::mv(mat, vec);
  
  // 若 alpha 不等于 1，则对结果进行标量乘法
  if (!alpha.equal(1)) {
    out = alpha * out;
  }
  
  // 若 beta 不等于 0，则将 input 乘以 beta 并加到结果上
  if (!beta.equal(0)) {
    out = beta * input + out;
  }
  
  // 返回最终结果
  return out;
}

// 对于 addbmm 函数的分解实现，对批量矩阵乘法进行分解操作
Tensor addbmm_decomp(
  const Tensor& input, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  
  // 计算 bmm 函数的结果，并在 dim=0 上求和
  Tensor out = at::bmm(batch1, batch2).sum(0);
  
  // 若 alpha 不等于 1，则对结果进行标量乘法
  if (!alpha.equal(1)) {
    out = alpha * out;
  }
  
  // 若 beta 不等于 0，则将 input 乘以 beta 并加到结果上
  if (!beta.equal(0)) {
    out = beta * input + out;
  }
  
  // 返回最终结果
  return out;
}

// 对于 baddbmm 函数的分解实现，对批量矩阵乘法进行分解操作
Tensor baddbmm_decomp(
  const Tensor& input, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  
  // 计算 bmm 函数的结果
  Tensor out = at::bmm(batch1, batch2);
  
  // 若 alpha 不等于 1，则对结果进行标量乘法
  if (!alpha.equal(1)) {
    out = alpha * out;
  }
  
  // 若 beta 不等于 0，则将 input 乘以 beta 并加到结果上
  if (!beta.equal(0)) {
    out = beta * input + out;
  }
  
  // 返回最终结果
  return out;
}
    # 将输入 input 乘以 beta，并加上当前的 out 值，然后将结果赋给 out
    out = beta * input + out;
  }
  # 返回最终计算得到的 out 值
  return out;
}

// 函数：将矩阵相乘和加法分解，使用 beta 和 alpha 对结果进行加权
Tensor addmm_decomp(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  // 以下分解可能不是很快...
  return at::add(self * beta, at::mm(mat1, mat2), alpha);
}

// 函数：检查批处理规则中的错误，并将批处理维度移动到前面
void _linalg_check_errors_batch_rule(const Tensor& info, optional<int64_t> info_bdim, c10::string_view api_name, bool is_matrix) {
  auto info_ = moveBatchDimToFront(info, info_bdim);
  // 如果不是矩阵，则意味着这是一批矩阵
  at::_linalg_check_errors(info_, api_name, false);
}

// 函数：计算 Householder 乘积的批处理规则
std::tuple<Tensor, std::optional<int64_t>>
householder_product_batch_rule(const Tensor &input, std::optional<int64_t> input_bdim,
                               const Tensor &tau, std::optional<int64_t> tau_bdim)
{
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto tau_ = moveBatchDimToFront(tau, tau_bdim);

  auto batch_size = get_bdim_size2(input, input_bdim, tau, tau_bdim);

  input_ = ensure_has_bdim(input_, input_bdim.has_value(), batch_size);
  tau_ = ensure_has_bdim(tau_, tau_bdim.has_value(), batch_size);
  return std::make_tuple(at::linalg_householder_product(input_, tau_), 0);
}

// 结构体模板：用于检查矩阵的一元运算规则
template <char const *op_name, typename A, A a, typename C>
struct LinalgCheckMatrixUnaryRuleHelper;

// 结构体模板：用于检查矩阵的一元运算规则的辅助类
template <char const *op_name, typename F, F Func, typename A, typename... T>
struct LinalgCheckMatrixUnaryRuleHelper<op_name, F, Func, typelist<A, T...>> {
  // 函数：检查和重新整形输入张量，并将批处理维度移动到前面
  static inline Tensor check_and_reshape_input(const Tensor& tensor, optional<int64_t> batch_dim) {
    TORCH_CHECK(rankWithoutBatchDim(tensor, batch_dim) >= 2, op_name, ": 输入张量 A 必须至少有 2 维.");
    return moveBatchDimToFront(tensor, batch_dim);
  }

  // 函数：应用一元运算到张量，返回一个输出
  static oneOutput apply_one(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    const auto tensor_ = check_and_reshape_input(tensor, batch_dim);
    return std::make_tuple(Func(tensor_, std::forward<T>(extra_args)...), 0);
  }

  // 函数：应用一元运算到张量，返回两个输出
  static twoOutputs apply_two(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    const auto tensor_ = check_and_reshape_input(tensor, batch_dim);
    const auto res = Func(tensor_, std::forward<T>(extra_args)...);
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0);
  }

  // 函数：应用一元运算到张量，返回三个输出
  static threeOutputs apply_three(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    const auto tensor_ = check_and_reshape_input(tensor, batch_dim);
    const auto res = Func(tensor_, std::forward<T>(extra_args)...);
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0);
  }

  // 函数：应用一元运算到张量，返回四个输出
  static fourOutputs apply_four(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    const auto tensor_ = check_and_reshape_input(tensor, batch_dim);
    const auto res = Func(tensor_, std::forward<T>(extra_args)...);
    // 返回一个元组，包含四个元素，每个元素是原始元组 res 中对应位置的第一个、第三个、第五个和第七个元素，
    // 其余元素均为整数 0。这种操作实现了从 res 元组提取部分元素并填充零值。
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0, std::get<3>(res), 0);
}
};

template <char const *op_name, typename A, A a, typename C>
struct LinalgCheckMatrixBinaryRuleHelper;

// 模板特化，用于矩阵二进制操作规则的辅助结构体
template <char const *op_name, typename F, F Func, typename A, typename B, typename... T>
struct LinalgCheckMatrixBinaryRuleHelper<op_name, F, Func, typelist<A, B, T...>> {
  
  // 检查输入并重新整形输入，返回两个张量的元组
  static inline std::tuple<Tensor, Tensor> check_inputs_and_reshape_inputs(
      const Tensor& first, optional<int64_t> first_bdim,
      const Tensor& second, optional<int64_t> second_bdim) {
    TORCH_CHECK(rankWithoutBatchDim(first, first_bdim) >= 2,
                op_name, ": 输入张量 A 必须至少有 2 维.");
    TORCH_CHECK(rankWithoutBatchDim(second, second_bdim) >= 2,
                op_name, ": 输入张量 B 必须至少有 2 维.");
    return _binary_pointwise_helper(first, first_bdim, second, second_bdim, false);
  }

  // 应用单个输出
  static oneOutput apply_one(
      const Tensor& first, optional<int64_t> first_bdim,
      const Tensor& second, optional<int64_t> second_bdim,
      T... extra_args) {
    const auto tensor_other = check_inputs_and_reshape_inputs(first, first_bdim, second, second_bdim);
    const auto tensor_ = std::get<0>(tensor_other);
    const auto other_ = std::get<1>(tensor_other);
    return std::make_tuple(Func(tensor_, other_, std::forward<T>(extra_args)...), 0);
  }

  // 应用双输出
  static twoOutputs apply_two(
      const Tensor& first, optional<int64_t> first_bdim,
      const Tensor& second, optional<int64_t> second_bdim,
      T... extra_args) {
    const auto tensor_other = check_inputs_and_reshape_inputs(first, first_bdim, second, second_bdim);
    const auto tensor_ = std::get<0>(tensor_other);
    const auto other_ = std::get<1>(tensor_other);
    const auto res = Func(tensor_, other_, std::forward<T>(extra_args)...);
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0);
  }
};

// 检查张量是否至少具有期望的秩度
static void expect_at_least_rank(
    const Tensor& tensor,
    optional<int64_t> tensor_bdim,
    int64_t expected_rank,
    const char* name) {
  auto rank = rankWithoutBatchDim(tensor, tensor_bdim);
  TORCH_CHECK(rank >= expected_rank,
      name, " 应至少有 ", expected_rank, " 维，但实际有 ",
      rank, " 维。");
}

// LU 分解解包批处理规则
threeOutputs linalg_lu_unpack_batch_rule(
    const Tensor& LU, optional<int64_t> LU_bdim,
    const Tensor& pivots, optional<int64_t> pivots_bdim,
    bool unpack_data, bool unpack_pivots) {
  auto LU_ = moveBatchDimToFront(LU, LU_bdim);
  auto pivots_ = moveBatchDimToFront(pivots, pivots_bdim);

  // LU 和 pivots 的前 {N-2}（对于 LU），{N-1}（对于 pivots）维必须匹配
  // 因此，如果其中一个正在进行 vmap，我们必须扩展该维度。
  if (LU_bdim.has_value() != pivots_bdim.has_value()) {
    auto bdim_size = get_bdim_size2(LU, LU_bdim, pivots, pivots_bdim);
    LU_ = ensure_has_bdim(LU_, LU_bdim.has_value(), bdim_size);
    pivots_ = ensure_has_bdim(pivots_, pivots_bdim.has_value(), bdim_size);
    pivots_bdim = 0;
    LU_bdim = 0;
  }

  const auto res = at::lu_unpack(LU_, pivots_, unpack_data, unpack_pivots);
  return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0);


    // 将 LU_bdim 设为 0，这里的 LU_bdim 是一个变量或者状态的设定
    LU_bdim = 0;
  }
  
  // 调用 at::lu_unpack 函数解包 LU 分解结果
  const auto res = at::lu_unpack(LU_, pivots_, unpack_data, unpack_pivots);
  
  // 返回一个包含解包结果的元组，其中每个解包结果后面跟着 0
  return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0);
}

// linalg_lu_solve_batch_rule函数实现了批处理的LU求解规则
oneOutput linalg_lu_solve_batch_rule(
    const Tensor& LU, optional<int64_t> LU_bdim,
    const Tensor& pivots, optional<int64_t> pivots_bdim,
    const Tensor& B, optional<int64_t> B_bdim,
    bool left, bool adjoint) {
  
  // 最小的LU张量秩为2
  const auto LU_min_rank = 2;
  // 最小的pivots张量秩为1
  const auto pivots_min_rank = 1;
  // 最小的B张量秩为2
  const auto B_min_rank = 2;

  // 对LU张量的秩进行检查
  expect_at_least_rank(LU, LU_bdim, LU_min_rank, "LU");
  // 对pivots张量的秩进行检查
  expect_at_least_rank(pivots, pivots_bdim, pivots_min_rank, "pivots");
  // 对B张量的秩进行检查
  expect_at_least_rank(B, B_bdim, B_min_rank, "B");

  // 将批处理维度移到张量的最前面
  auto LU_ = moveBatchDimToFront(LU, LU_bdim);
  auto pivots_ = moveBatchDimToFront(pivots, pivots_bdim);
  auto B_ = moveBatchDimToFront(B, B_bdim);

  // 如果LU和pivots的批处理维度不同，则进行维度的扩展
  if (LU_bdim.has_value() ^ pivots_bdim.has_value()) {
    auto bdim_size = get_bdim_size2(LU, LU_bdim, pivots, pivots_bdim);
    LU_ = ensure_has_bdim(LU_, LU_bdim.has_value(), bdim_size);
    pivots_ = ensure_has_bdim(pivots_, pivots_bdim.has_value(), bdim_size);
    pivots_bdim = 0;
    LU_bdim = 0;
  }

  // 现在，LU、pivots和B的第一个维度可以进行广播
  // 后续的逻辑处理这一点
  const auto LU_num_batch_dims = rankWithoutBatchDim(LU_, LU_bdim) - LU_min_rank;
  const auto pivots_num_batch_dims = rankWithoutBatchDim(pivots_, pivots_bdim) - pivots_min_rank;
  const auto B_num_batch_dims = rankWithoutBatchDim(B_, B_bdim) - B_min_rank;
  const auto max_num_batch_dims = std::max(std::max(LU_num_batch_dims, pivots_num_batch_dims), B_num_batch_dims);

  LU_ = maybePadToLogicalRank(LU_, LU_bdim, max_num_batch_dims + LU_min_rank);
  pivots_ = maybePadToLogicalRank(pivots_, pivots_bdim, max_num_batch_dims + pivots_min_rank);
  B_ = maybePadToLogicalRank(B_, B_bdim, max_num_batch_dims + B_min_rank);

  // 调用ATen的linalg_lu_solve函数求解LU分解
  const auto result = at::linalg_lu_solve(LU_, pivots_, B_, left, adjoint);
  // 返回结果和标记0
  return std::make_tuple(result, 0);
}

// cholesky_solve_batch_rule函数实现了批处理的Cholesky求解规则
oneOutput cholesky_solve_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim,
    const Tensor& A, std::optional<int64_t> A_bdim,
    bool upper) {
  
  // 检查self张量是否至少有两个维度
  TORCH_CHECK(rankWithoutBatchDim(self, self_bdim) >= 2,
           "b should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  // 检查A张量是否至少有两个维度
  TORCH_CHECK(rankWithoutBatchDim(A, A_bdim) >= 2,
           "u should have at least 2 dimensions, but has ", A.dim(), " dimensions instead");

  // 对self和A张量进行类型无关的二元点对点帮助函数计算
  const auto tensor_other = _binary_pointwise_helper(self, self_bdim, A, A_bdim, /*do_type_promotion=*/false);
  const auto tensor_ = std::get<0>(tensor_other);
  const auto other_ = std::get<1>(tensor_other);

  // 调用ATen的cholesky_solve函数求解Cholesky分解
  return std::make_tuple(at::cholesky_solve(tensor_, other_, upper), 0);
}

// linalg_lu_factor_ex_batch_rule函数的开头，尚未完全展示
threeOutputs linalg_lu_factor_ex_batch_rule(
    // 对输入的张量 A 进行 LU 分解，其中 A_bdim 是批处理维度的可选参数，pivot 表示是否使用部分选主元策略，check_errors 表示是否检查错误
    TORCH_CHECK(rankWithoutBatchDim(A, A_bdim) >= 2, "torch.lu_factor_ex: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");
    // 将批处理维度移到张量 A 的最前面，并返回重新排列后的张量 A_
    const auto A_ = moveBatchDimToFront(A, A_bdim);
    // 调用 PyTorch 提供的 linalg_lu_factor_ex 函数进行 LU 分解，res 包含 LU 分解结果的三元组
    const auto res = at::linalg_lu_factor_ex(A_, pivot, check_errors);
    // 返回一个包含 LU 分解结果的元组，其中每个分量都用 0 表示第三个输出参数为空
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0);
}

// 定义一个函数，实现矩阵指数的批处理规则
oneOutput matrix_exp_batch_rule(const Tensor& self, std::optional<int64_t> self_bdim) {
  // 检查输入张量 self 的非批处理维度是否至少为2，否则抛出错误信息
  TORCH_CHECK(rankWithoutBatchDim(self, self_bdim) >= 2, "torch.matrix_exp: The input tensor A must have at least 2 dimensions.");
  // 将批处理维度移到前面并确保连续性
  const auto self_ = moveBatchDimToFront(self, self_bdim).contiguous();  // 似乎存在 bug
  // 返回矩阵指数操作的结果和一个标志
  return std::make_tuple(at::matrix_exp(self_), 0);
}

// 定义一个函数，实现批量解决线性方程组的规则
fourOutputs solve_ex_batch_rule(
    const Tensor& A, optional<int64_t> A_bdim,
    const Tensor& B, optional<int64_t> B_bdim,
    bool left, bool check_errors) {
  // 获取批次大小
  auto batch_size = get_bdim_size2(A, A_bdim, B, B_bdim);
  // 获取 A 和 B 的逻辑秩（排除批处理维度）
  const auto A_logical_rank = rankWithoutBatchDim(A, A_bdim);
  const auto B_logical_rank = rankWithoutBatchDim(B, B_bdim);
  // 获取最大的逻辑秩
  const auto max_logical_rank = std::max(A_logical_rank, B_logical_rank);

  // 检查 A 的逻辑秩是否至少为 2，否则抛出错误信息
  TORCH_CHECK(A_logical_rank >= 2,
            "linalg.solve: The input tensor A must have at least 2 dimensions.");

  // 根据 A 和 B 的逻辑秩更新 b_logical_rank
  auto b_logical_rank = max_logical_rank;
  if (A_logical_rank > B_logical_rank) {  // 向量情况：B 是向量或批处理向量
    // 不准确，但匹配 linalg 的错误消息
    TORCH_CHECK(B_logical_rank >= 1, "linalg.solve: The input tensor B must have at least 2 dimensions.");
    b_logical_rank = max_logical_rank - 1;
  } else {  // 矩阵情况：A 和 B 都是矩阵或矩阵批处理
    TORCH_CHECK(B_logical_rank >= 2, "linalg.solve: The input tensor B must have at least 2 dimensions.");
  }

  // 将批处理维度移到前面，并根据逻辑秩进行可能的填充
  auto A_ = moveBatchDimToFront(A, A_bdim);
  auto B_ = moveBatchDimToFront(B, B_bdim);
  A_ = maybePadToLogicalRank(A_, A_bdim, max_logical_rank);
  B_ = maybePadToLogicalRank(B_, B_bdim, b_logical_rank);

  // 确保 A 和 B 具有批处理维度
  A_ = ensure_has_bdim(A_, A_bdim.has_value(), batch_size);
  B_ = ensure_has_bdim(B_, B_bdim.has_value(), batch_size);

  // 注意 [ solve_ex Batch Rule Contiguity ]
  // A 决定 linalg_solve 是否采用优化路径。我们需要对 A_ 进行检查以匹配对 A 执行的批处理张量的检查，因为它可能已被 autograd 保存（特别是由 jvp 保存），autograd 的行为取决于是否采用了优化路径
  const auto batched_A_was_contiguous = A_bdim.has_value() ? at::select(A, *A_bdim, 0).is_contiguous() : A.is_contiguous();
  if (batched_A_was_contiguous && !A.is_complex()) {
    A_ = A_.contiguous();
  }
  // 调用 _linalg_solve_ex 函数求解线性方程组
  const auto res = _linalg_solve_ex(A_, B_, left, check_errors);
  // 返回结果元组，其中每个解的前面都有一个占位符 0
  return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0, std::get<3>(res), 0);
}

// 定义一个函数，实现批处理交叉操作的规则
oneOutput cross_batch_rule(const Tensor& self, std::optional<int64_t> self_bdim,
                           const Tensor& other, std::optional<int64_t> other_bdim, const int64_t dim) {
  // 检查去除批处理维度后 self 和 other 的秩是否相同
  TORCH_CHECK(rankWithoutBatchDim(self, self_bdim) == rankWithoutBatchDim(other, other_bdim),
    // 抛出异常，指示 linalg.cross 函数输入的张量必须具有相同的维度数量。
    "linalg.cross: inputs must have the same number of dimensions."
  );

  // 获取第二个张量的批次大小，该批次大小由 get_bdim_size2 函数确定。
  const auto batch_size = get_bdim_size2(self, self_bdim, other, other_bdim);

  // 使用 _binary_pointwise_helper 辅助函数处理两个张量，不进行原位操作。
  const auto self_other_bundled = _binary_pointwise_helper(self, self_bdim, other, other_bdim, false);

  // 确保第一个张量具有批次维度，根据情况在 self_ 中包含批次维度。
  const auto self_ = ensure_has_bdim(std::get<0>(self_other_bundled), self_bdim.has_value(), batch_size);
  
  // 确保第二个张量具有批次维度，根据情况在 other_ 中包含批次维度。
  const auto other_ = ensure_has_bdim(std::get<1>(self_other_bundled), other_bdim.has_value(), batch_size);

  // 获取物理维度，基于 self_ 的情况，考虑是否反转维度。
  const auto dim_ = getPhysicalDim(self_, true, dim);

  // 返回一个元组，包含 linalg_cross 函数的结果和一个标志值 0。
  return std::make_tuple(linalg_cross(self_, other_, dim_), 0);
// 如果张量 t 的维度为 1 且大小为 0，返回空的 std::optional<int64_t>
std::optional<int64_t> batch_dim_if_not_empty(const Tensor& t) {
  if (t.dim() == 1 && t.size(0) == 0) {
    return std::optional<int64_t>();
  }
  // 否则返回包含值 0 的 std::optional<int64_t>
  return std::optional<int64_t>(0);
}

// linalg_lstsq 的批处理规则实现
fourOutputs linalg_lstsq_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim, const Tensor& b, std::optional<int64_t> b_bdim,
    std::optional<double> rcond, std::optional<c10::string_view> driver) {
  // 检查 self 张量去除批处理维度后的秩至少为 2
  TORCH_CHECK(rankWithoutBatchDim(self, self_bdim) >= 2, "torch.linalg.lstsq: input must have at least 2 dimensions.");
  // 检查 b 张量去除批处理维度后的秩至少为 1
  TORCH_CHECK(rankWithoutBatchDim(b, b_bdim) >= 1, "torch.linalg.lstsq: other must have at least 1 dimension.");

  // 获取批处理大小
  const auto batch_size = get_bdim_size2(self, self_bdim, b, b_bdim);
  // 对 self 和 b 进行点对点操作，不进行类型提升
  const auto tensor_other = _binary_pointwise_helper(self, self_bdim, b, b_bdim, /*do_type_promotion=*/false);

  // 由于向量情况不明确，lstsq 可以广播 [1, 2] -> [batch_size, 2] 但不可以 [2] -> [batch_size, 2]
  // 因此，如果没有批处理维度，可以展开，否则确保具有批处理维度
  const auto self_ = ensure_has_bdim(std::get<0>(tensor_other), self_bdim.has_value(), batch_size);
  const auto b_ = ensure_has_bdim(std::get<1>(tensor_other), b_bdim.has_value(), batch_size);

  // 调用 at::linalg_lstsq 函数
  auto [res, res_1, res_2, res_3] = at::linalg_lstsq(self_, b_, rcond, driver);

  // 除了第 0 个输出以外的其他输出有时是计算的。当它们未计算时，它们是没有批处理维度的空张量
  // 计算 res_1、res_2、res_3 的批处理维度（如果它们非空）
  const auto res_1_bdim = batch_dim_if_not_empty(res_1);
  const auto res_2_bdim = batch_dim_if_not_empty(res_2);
  const auto res_3_bdim = batch_dim_if_not_empty(res_3);
  
  // 返回结果元组
  return std::make_tuple(res, 0, res_1, res_1_bdim, res_2, res_2_bdim, res_3, res_3_bdim);
}

// atol_rtol_tensor_batch_rule 函数模板的实现
template<typename F>
std::tuple<Tensor, std::optional<int64_t>>
atol_rtol_tensor_batch_rule(
    F Func, const Tensor& input, optional<int64_t> input_bdim,
    const optional<Tensor>& atol, const optional<int64_t> atol_bdim,
    // 继续函数参数列表...
    // 计算输入张量的逻辑秩（去除批量维度），用于确定操作的输入维度
    auto input_logical_rank = rankWithoutBatchDim(input, input_bdim);

    // 断言输入张量的逻辑秩至少为2，因为操作要求至少是二维的矩阵
    TORCH_CHECK(input_logical_rank >= 2,
            op_name, ": The input tensor input must have at least 2 dimensions.");

    // 计算需要广播到输入批量维度数的 atol 和 rtol 的逻辑维度
    // 这里的输入批量维度数是 input 的维度数减去2（因为 input 表示一批矩阵，2维用于矩阵维度）
    const auto input_logical_num_bdims = input_logical_rank - 2;
    const int64_t atol_logical_num_bdims = atol.has_value() ? rankWithoutBatchDim(*atol, atol_bdim) : 0;
    const int64_t rtol_logical_num_bdims = rtol.has_value() ? rankWithoutBatchDim(*rtol, rtol_bdim) : 0;
    // 计算三者中的最大逻辑维度数，以确定需要扩展的维度数
    const auto max_logical_bdims = std::max({input_logical_num_bdims, atol_logical_num_bdims, rtol_logical_num_bdims});

    // 将输入张量的批量维度移到前面，以便后续操作
    auto input_ = moveBatchDimToFront(input, input_bdim);
    // 如果有提供 atol，则也将其批量维度移到前面
    auto atol_ = atol.has_value() ? moveBatchDimToFront(*atol, atol_bdim) : atol;
    // 如果有提供 rtol，则也将其批量维度移到前面
    auto rtol_ = rtol.has_value() ? moveBatchDimToFront(*rtol, rtol_bdim) : rtol;

    // 对所有输入进行填充，使其具有相同数量的（非 vmap）批量维度
    // 这是为了保证操作的输入在批量维度上的一致性
    input_ = maybePadToLogicalRank(input_, input_bdim, max_logical_bdims + 2);
    atol_ = atol_.has_value() ? maybePadToLogicalRank(*atol_, atol_bdim, max_logical_bdims) : atol_;
    rtol_ = rtol_.has_value() ? maybePadToLogicalRank(*rtol_, rtol_bdim, max_logical_bdims) : rtol_;

    // 返回一个包含处理后输入的元组，以及一个默认的成功状态
    return std::make_tuple(Func(input_, atol_, rtol_, hermitian), 0);
}

// 定义 pinv_batch_rule 函数，接受多个参数并返回一个 tuple
static std::tuple<Tensor, std::optional<int64_t>>
pinv_batch_rule(
    const Tensor& input, std::optional<int64_t> input_bdim, const optional<Tensor>& atol,
    const std::optional<int64_t> atol_bdim, const optional<Tensor>& rtol,
    const std::optional<int64_t> rtol_bdim, bool hermitian) {
  // 调用 atol_rtol_tensor_batch_rule 函数，传递 linalg_pinv 作为参数之一，返回其结果
  return atol_rtol_tensor_batch_rule(ATEN_FN2(linalg_pinv, atol_rtol_tensor), input, input_bdim, atol, atol_bdim, rtol, rtol_bdim, hermitian, "linalg.pinv");
}
}

// 定义宏 LINALG_CHECK_MATRIX_UNARY_BATCH_RULE，用于生成检查矩阵一元批处理规则的辅助代码
#define LINALG_CHECK_MATRIX_UNARY_BATCH_RULE(fn, num_out) SINGLE_ARG(\
  LinalgCheckMatrixUnaryRuleHelper<\
    func_string_##fn,\
    decltype(&ATEN_FN(fn)),\
    &ATEN_FN(fn),\
    c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types>::apply_##num_out)

// 定义宏 LINALG_CHECK_MATRIX_UNARY_BATCH_RULE2，用于生成检查矩阵一元批处理规则的辅助代码（针对二元运算符重载）
#define LINALG_CHECK_MATRIX_UNARY_BATCH_RULE2(fn, overload, num_out) SINGLE_ARG(\
  LinalgCheckMatrixUnaryRuleHelper<\
    func_string_##fn_##overload,\
    decltype(&ATEN_FN2(fn, overload)),\
    &ATEN_FN2(fn, overload),\
    c10::guts::function_traits<decltype(ATEN_FN2(fn, overload))>::parameter_types>::apply_##num_out)

// 定义宏 LINALG_CHECK_MATRIX_BINARY_BATCH_RULE，用于生成检查矩阵二元批处理规则的辅助代码
#define LINALG_CHECK_MATRIX_BINARY_BATCH_RULE(fn, num_out) SINGLE_ARG(\
  LinalgCheckMatrixBinaryRuleHelper<\
    func_string_##fn,\
    decltype(&ATEN_FN(fn)),\
    &ATEN_FN(fn),\
    c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types>::apply_##num_out)

// 定义字符串常量，用于存储函数名称，这些常量将作为模板参数使用
// C++ 不允许我们直接使用字符串字面量作为模板参数，所以必须先声明它们为常量
// 这些宏定义主要用于跨平台的兼容性和编译器特定的处理

#if defined(_MSC_VER)
#define LINALG_STRING_CONST(fn, op_name) \
  const char func_string_##fn[] = #op_name;\

#define LINALG_STRING_CONST2(fn, overload, op_name) \
  const char func_string_##fn_##overload[] = #op_name;\

#else
#define LINALG_STRING_CONST(fn, op_name) \
  constexpr const char func_string_##fn[] = #op_name;\

#define LINALG_STRING_CONST2(fn, overload, op_name) \
  constexpr const char func_string_##fn_##overload[] = #op_name;\

#endif

// 定义宏 LINALG_CHECK_MATRIX_UNARY_ONE_OUT，生成检查矩阵一元操作的辅助代码，支持单一输出
#define LINALG_CHECK_MATRIX_UNARY_ONE_OUT(fn, op_name) \
  LINALG_STRING_CONST(fn, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT(fn, LINALG_CHECK_MATRIX_UNARY_BATCH_RULE(fn, one));\
  }

// 定义宏 LINALG_CHECK_MATRIX_UNARY_ONE_OUT2，生成检查矩阵一元操作的辅助代码，支持单一输出（针对二元运算符重载）
#define LINALG_CHECK_MATRIX_UNARY_ONE_OUT2(fn, overload, op_name) \
  LINALG_STRING_CONST2(fn, overload, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT2(fn, overload, LINALG_CHECK_MATRIX_UNARY_BATCH_RULE2(fn, overload, one));\
  }

// 定义宏 LINALG_CHECK_MATRIX_UNARY_TWO_OUT，生成检查矩阵一元操作的辅助代码，支持双输出
#define LINALG_CHECK_MATRIX_UNARY_TWO_OUT(fn, op_name) \
  LINALG_STRING_CONST(fn, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT(fn, LINALG_CHECK_MATRIX_UNARY_BATCH_RULE(fn, two));\
  }
#define LINALG_CHECK_MATRIX_UNARY_THREE_OUT(fn, op_name) \
  // 定义宏，用于检查矩阵一元操作（三个输出），并定义操作名称常量
  LINALG_STRING_CONST(fn, op_name);\
  // 在aten命名空间下实现TORCH库函数，支持批处理化的矩阵操作fn，使用三个输出的批处理规则

#define LINALG_CHECK_MATRIX_UNARY_FOUR_OUT(fn, op_name) \
  // 定义宏，用于检查矩阵一元操作（四个输出），并定义操作名称常量
  LINALG_STRING_CONST(fn, op_name);\
  // 在aten命名空间下实现TORCH库函数，支持批处理化的矩阵操作fn，使用四个输出的批处理规则

#define LINALG_CHECK_MATRIX_BINARY_ONE_OUT(fn, op_name) \
  // 定义宏，用于检查矩阵二元操作（一个输出），并定义操作名称常量
  LINALG_STRING_CONST(fn, op_name);\
  // 在aten命名空间下实现TORCH库函数，支持批处理化的矩阵操作fn，使用一个输出的批处理规则

#define LINALG_CHECK_MATRIX_BINARY_TWO_OUT(fn, op_name) \
  // 定义宏，用于检查矩阵二元操作（两个输出），并定义操作名称常量
  LINALG_STRING_CONST(fn, op_name);\
  // 在aten命名空间下实现TORCH库函数，支持批处理化的矩阵操作fn，使用两个输出的批处理规则

// 这些宏定义必须在外部进行，字符串常量必须在宏外声明以便作为模板参数使用
// NOLINTBEGIN(*array*)
// 调用宏定义，检查并定义 cholesky 操作，使用一元操作（一个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_ONE_OUT(cholesky, cholesky);
// 调用宏定义，检查并定义 cholesky_inverse 操作，使用一元操作（一个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_ONE_OUT(cholesky_inverse, cholesky_inverse);
// 调用宏定义，检查并定义 linalg_cholesky_ex 操作，使用二元操作（两个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_cholesky_ex, linalg.cholesky);
// 调用宏定义，检查并定义 linalg_eig 操作，使用二元操作（两个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_eig, linalg.eig);
// 调用宏定义，检查并定义 linalg_inv_ex 操作，使用二元操作（两个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_inv_ex, linalg.inv_ex);
// 调用宏定义，检查并定义 linalg_ldl_factor_ex 操作，使用一元操作（三个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_THREE_OUT(linalg_ldl_factor_ex, torch.linalg.ldl_factor_ex);
// 调用宏定义，检查并定义 linalg_qr 操作，使用二元操作（两个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_qr, linalg.qr);
// 调用宏定义，检查并定义 linalg_slogdet 操作，使用二元操作（两个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_slogdet, linalg.slogdet);
// 调用宏定义，检查并定义 linalg_solve_triangular 操作，使用二元操作（一个输出）的批处理规则
LINALG_CHECK_MATRIX_BINARY_ONE_OUT(linalg_solve_triangular, linalg.solve_triangular);

// 调用宏定义，检查并定义 geqrf 操作，使用二元操作（两个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(geqrf, geqrf);
// 调用宏定义，检查并定义 triangular_solve 操作，使用二元操作（两个输出）的批处理规则
LINALG_CHECK_MATRIX_BINARY_TWO_OUT(triangular_solve, triangular_solve);
// 调用宏定义，检查并定义 _linalg_det 操作，使用一元操作（三个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_THREE_OUT(_linalg_det, linalg.det);
// 调用宏定义，检查并定义 _linalg_eigh 操作，使用二元操作（两个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(_linalg_eigh, linalg.eigh);
// 调用宏定义，检查并定义 _linalg_slogdet 操作，使用一元操作（四个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_FOUR_OUT(_linalg_slogdet, linalg.slogdet);
// 调用宏定义，检查并定义 _linalg_svd 操作，使用一元操作（三个输出）的批处理规则
LINALG_CHECK_MATRIX_UNARY_THREE_OUT(_linalg_svd, linalg.svd);
// NOLINTEND(*array*)
// 实现 Torch 的 aten 库中的函数批处理机制
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
    // 支持批处理的 bmm 函数，使用 bmm_batch_rule 规则
    VMAP_SUPPORT(bmm, bmm_batch_rule);
    // 实现 addmv 函数的分解
    m.impl("addmv", addmv_decomp);
    // 实现 addmm 函数的分解
    m.impl("addmm", addmm_decomp);
    // 实现 addbmm 函数的分解
    m.impl("addbmm", addbmm_decomp);
    // 实现 baddbmm 函数的分解
    m.impl("baddbmm", baddbmm_decomp);
    // 支持批处理的 dot 函数，使用 dot_batch_rule 规则
    VMAP_SUPPORT(dot, dot_batch_rule);
    // 支持批处理的 mv 函数，使用 mv_batch_rule 规则
    VMAP_SUPPORT(mv, mv_batch_rule);
    // 支持批处理的 mm 函数，使用 mm_batch_rule 规则
    VMAP_SUPPORT(mm, mm_batch_rule);
    // 支持批处理的 lu_unpack 函数，使用 linalg_lu_unpack_batch_rule 规则
    VMAP_SUPPORT(lu_unpack, linalg_lu_unpack_batch_rule);
    // 支持批处理的 linalg_lu_solve 函数，使用 linalg_lu_solve_batch_rule 规则
    VMAP_SUPPORT(linalg_lu_solve, linalg_lu_solve_batch_rule);
    // 支持批处理的 linalg_householder_product 函数，使用 householder_product_batch_rule 规则
    VMAP_SUPPORT(linalg_householder_product, householder_product_batch_rule);
    // 支持批处理的 cholesky_solve 函数，使用 cholesky_solve_batch_rule 规则（自定义维度错误）
    VMAP_SUPPORT(cholesky_solve, cholesky_solve_batch_rule);  // custom dim error
    // 支持批处理的 linalg_lstsq 函数，使用 linalg_lstsq_batch_rule 规则（自定义错误和有时返回空）
    VMAP_SUPPORT(linalg_lstsq, linalg_lstsq_batch_rule);  // custom errors and sometimes empty return
    // 支持批处理的 linalg_lu_factor_ex 函数，使用 linalg_lu_factor_ex_batch_rule 规则
    VMAP_SUPPORT(linalg_lu_factor_ex, linalg_lu_factor_ex_batch_rule);
    // 支持批处理的 linalg_matrix_exp 函数，使用 matrix_exp_batch_rule 规则
    VMAP_SUPPORT(linalg_matrix_exp, matrix_exp_batch_rule);
    // 支持批处理的 _linalg_solve_ex 函数，使用 solve_ex_batch_rule 规则
    VMAP_SUPPORT(_linalg_solve_ex, solve_ex_batch_rule);
    // 支持批处理的 linalg_cross 函数，使用 cross_batch_rule 规则
    VMAP_SUPPORT(linalg_cross, cross_batch_rule);
    // 使用 atol_rtol_tensor 参数支持批处理的 linalg_pinv 函数，使用 pinv_batch_rule 规则
    VMAP_SUPPORT2(linalg_pinv, atol_rtol_tensor, pinv_batch_rule);

    // 支持批处理的 _linalg_check_errors 函数，使用 _linalg_check_errors_batch_rule 规则
    VMAP_SUPPORT(_linalg_check_errors, _linalg_check_errors_batch_rule);

    // 实现 vdot 函数的分解
    m.impl("vdot", vdot_decomp);
}
} // namespace at::functorch
```