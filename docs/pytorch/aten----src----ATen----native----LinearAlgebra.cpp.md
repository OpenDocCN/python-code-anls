# `.\pytorch\aten\src\ATen\native\LinearAlgebra.cpp`

```py
// 定义宏，仅包含每个操作符的方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库的头文件
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/int_mm_kernel.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/Matmul.h>
#include <c10/core/GradMode.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <variant>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含通用操作函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则包含各个操作的具体实现头文件
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_compute_linear_combination_native.h>
#include <ATen/ops/_convert_weight_to_int4pack_native.h>
#include <ATen/ops/_int_mm_native.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_det.h>
#include <ATen/ops/_linalg_det_native.h>
#include <ATen/ops/_linalg_slogdet.h>
#include <ATen/ops/_linalg_slogdet_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/_weight_int4pack_mm_native.h>
#include <ATen/ops/_weight_int8pack_mm_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addr.h>
#include <ATen/ops/addr_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/chain_matmul_native.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/det_native.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diff.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/frobenius_norm_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/full.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/ger_native.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/inner_native.h>
#include <ATen/ops/is_complex_native.h>
#include <ATen/ops/is_floating_point_native.h>
#include <ATen/ops/kron_native.h>
#include <ATen/ops/linalg_cond.h>
#include <ATen/ops/linalg_cond_native.h>
#include <ATen/ops/linalg_det.h>
#include <ATen/ops/linalg_det_native.h>
#include <ATen/ops/linalg_diagonal_native.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_matmul_native.h>
#endif
// 包含 ATen 库中的线性代数操作头文件
#include <ATen/ops/linalg_matrix_exp.h>
#include <ATen/ops/linalg_matrix_exp_native.h>
#include <ATen/ops/linalg_matrix_norm.h>
#include <ATen/ops/linalg_matrix_norm_native.h>
#include <ATen/ops/linalg_matrix_power_native.h>
#include <ATen/ops/linalg_matrix_rank.h>
#include <ATen/ops/linalg_matrix_rank_native.h>
#include <ATen/ops/linalg_multi_dot_native.h>
#include <ATen/ops/linalg_norm.h>
#include <ATen/ops/linalg_norm_native.h>
#include <ATen/ops/linalg_pinv.h>
#include <ATen/ops/linalg_pinv_native.h>
#include <ATen/ops/linalg_slogdet.h>
#include <ATen/ops/linalg_slogdet_native.h>
#include <ATen/ops/linalg_solve.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_tensorinv.h>
#include <ATen/ops/linalg_tensorinv_native.h>
#include <ATen/ops/linalg_tensorsolve.h>
#include <ATen/ops/linalg_tensorsolve_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/linalg_vector_norm_native.h>

// 包含 ATen 库中的数学操作头文件
#include <ATen/ops/log2.h>
#include <ATen/ops/logdet_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/matrix_exp_backward_native.h>
#include <ATen/ops/matrix_exp_native.h>
#include <ATen/ops/matrix_power_native.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mv.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/nuclear_norm_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/outer.h>
#include <ATen/ops/outer_native.h>
#include <ATen/ops/pinverse_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/real.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/slogdet_native.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tensordot.h>
#include <ATen/ops/unique_consecutive.h>
#include <ATen/ops/vdot_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>

// 包含标准库头文件
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

// 仅在非 s390x 和 powerpc 架构下包含 cpuinfo 头文件
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

// 定义 ATen 命名空间
namespace at {

// 定义 detail 命名空间
namespace detail {

// 声明静态函数 check_linalg_norm_dtype，用于检查线性代数操作的数据类型
static void check_linalg_norm_dtype(optional<ScalarType> opt_dtype, ScalarType self_dtype, const char* const name) {
    // 检查是否存在可选的数据类型
    if (opt_dtype.has_value()) {
      // 如果存在可选的数据类型，获取其数值
      auto dtype = opt_dtype.value();
      // 检查数据类型是否为浮点型或复数型
      TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype), name, ": dtype should"
          " be floating point or complex, but got ", dtype);
      // 检查输入张量的数据类型是否与指定的数据类型相匹配（实部或虚部）
      TORCH_CHECK(isComplexType(self_dtype) == isComplexType(dtype),
          name, ": dtype should be ", isComplexType(self_dtype) ? "complex" : "real",
          " for ", isComplexType(self_dtype) ? "complex" : "real", " inputs, but got ", dtype);
      // 检查是否可以将输入张量的数据类型提升到指定的数据类型，而不会造成缩小
      TORCH_CHECK(promoteTypes(self_dtype, dtype) == dtype,
          name, ": the dtype of the input ", "(", self_dtype, ") should be convertible ",
          "without narrowing to the specified dtype (", dtype, ")");
    }
}

namespace meta {

#define ADDMM_META() \
  // 检查 self 和 mat2 的数据类型是否相同
  TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "self and mat2 must have the same dtype, but got ", self.scalar_type(), " and ", mat2.scalar_type()); \
  // 检查 mat1 和 mat2 的数据类型是否相同
  TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must have the same dtype, but got ", mat1.scalar_type(), " and ", mat2.scalar_type()); \
  // 检查 mat1 是否为二维张量
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor"); \
  // 检查 mat2 是否为二维张量
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor"); \
  // 检查 mat1 和 mat2 的形状是否可以相乘
  TORCH_CHECK( \
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (", \
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")"); \
 \
  // 推断输出张量的命名属性
  auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self); \
  // 设置输出张量的原始步幅和形状
  set_output_raw_strided(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, mat1.options(), names);

TORCH_META_FUNC(addmm)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  // 调用 ADDMM_META 宏
  ADDMM_META();
}

TORCH_META_FUNC(_addmm_activation)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu) {
  // 调用 ADDMM_META 宏
  ADDMM_META();
}

TORCH_META_FUNC(mm)(const Tensor & self, const Tensor & mat2) {
  // 检查 self 是否为二维张量
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  // 检查 mat2 是否为二维张量
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  // 检查 self 和 mat2 的形状是否可以相乘
  TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0], "x", self.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  // 计算矩阵乘积的输出张量的命名属性
  auto names = at::namedinference::compute_matmul_outnames(self, mat2);
  // 设置输出张量的原始步幅和形状
  set_output_raw_strided(0, {self.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
}

TORCH_META_FUNC(linalg_vector_norm)(const Tensor& self, const Scalar& scalar_ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  // 检查输入张量是否为浮点数或复数类型
  at::native::checkFloatingOrComplex(self, "linalg.vector_norm");

  auto dim = opt_dim.value_or(IntArrayRef{});
  // 将标量 ord 转换为双精度浮点数
  auto ord = scalar_ord.toDouble();

  // 如果张量为空且 ord 小于 0 或为无穷大
  //   - 无法在整个张量上进行缩减
  //   - 无法在空维度上进行缩减
  if (self.numel() == 0 && (ord < 0. || ord == INFINITY)) {
    // dim=None 或 dim=() 表示在整个张量上进行缩减
    TORCH_CHECK(opt_dim.has_value() && !opt_dim->empty(),
      "linalg.vector_norm cannot compute the ", scalar_ord, " norm on an empty ",
      "tensor because the operation does not have an identity");
    // 遍历给定的维度列表 `dim`
    for (auto dim_num : dim) {
      // 使用 TORCH_CHECK 确保当前维度 `dim_num` 在 `self` 张量中不为空
      // 如果为空，输出错误信息，指出无法计算指定 `scalar_ord` 范数在空维度上的操作
      // TORCH_CHECK 用于断言条件是否满足，否则会抛出错误信息
      TORCH_CHECK(self.size(dim_num) != 0,
        "linalg.vector_norm cannot compute the ", scalar_ord, " norm on the dimension ", dim_num ,
        "because this dimension is empty and the operation does not have an identity");
    }
  }

  // 检查并确保张量的数据类型兼容于 linalg.vector_norm 操作所需的数据类型
  at::detail::check_linalg_norm_dtype(opt_dtype, self.scalar_type(), "linalg.vector_norm");

  // 根据维度列表 `dim` 和当前张量的维度，生成一个表示维度的掩码 `mask`
  auto mask = at::native::make_dim_mask(dim, self.dim());
  // 根据掩码 `mask` 和 `keepdim` 参数，计算操作后的张量形状 `shape`
  auto shape = at::native::shape_from_dim_mask(self, std::move(mask), keepdim);
  // 根据操作的数据类型选择合适的选项 `options`，保持输出张量的数据类型一致性
  auto options = self.options()
                     .dtype(toRealValueType(opt_dtype.value_or(self.scalar_type())));

  // 设置输出张量的原始连续内存布局，指定其形状为 `shape`，不添加步长信息 `{}`，使用计算得到的选项 `options`
  set_output_raw_strided(0, shape, {}, options);
}

TORCH_META_FUNC(_linalg_det)(const Tensor& A) {
  // 检查输入张量 A 是否为方阵
  at::native::squareCheckInputs(A, "linalg.det");
  // 检查输入张量 A 是否为浮点数或复数类型
  at::native::checkFloatingOrComplex(A, "linalg.det");

  // 获取输入张量 A 的形状
  auto shape = A.sizes();
  auto ndim = shape.size();

  // 设置输出张量 0 的形状为去掉最后两个维度的形状，并且保持连续性
  set_output_contiguous(0, shape.slice(0, ndim - 2), A.options());

  // 计算批次矩阵的 LU 分解后的连续步长
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  // 设置输出张量 1 的形状和步长，并且保持张量连续性
  set_output_strided(1, shape, LU_strides, A.options());

  // 设置输出张量 2 的形状为去掉最后一个维度的形状，并且保持连续性，数据类型为整数
  set_output_contiguous(2, shape.slice(0, ndim - 1), A.options().dtype(kInt));
}

TORCH_META_FUNC(_linalg_slogdet)(const Tensor& A) {
  // 检查输入张量 A 是否为方阵
  at::native::squareCheckInputs(A, "linalg.slogdet");
  // 检查输入张量 A 是否为浮点数或复数类型，要求高精度计算
  at::native::checkFloatingOrComplex(A, "linalg.slogdet", /*low_precision*/false);

  // 获取输入张量 A 的形状
  auto shape = A.sizes();
  auto ndim = shape.size();

  // 计算输出张量的形状为去掉最后两个维度的形状
  auto shape_outputs = shape.slice(0, ndim - 2);

  // 设置输出张量 0 的形状为 shape_outputs，并且保持连续性
  set_output_contiguous(0, shape_outputs, A.options());

  // 设置输出张量 1 的形状为 shape_outputs，数据类型为 A 的实数值类型，保持连续性
  set_output_contiguous(1, shape_outputs, A.options().dtype(toRealValueType(A.scalar_type())));

  // 计算批次矩阵的 LU 分解后的连续步长
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  // 设置输出张量 2 的形状和步长，并且保持张量连续性
  set_output_strided(2, shape, LU_strides, A.options());

  // 设置输出张量 3 的形状为去掉最后一个维度的形状，数据类型为整数，保持连续性
  set_output_contiguous(3, shape.slice(0, ndim - 1), A.options().dtype(kInt));
}

template <typename Meta>
void common_checks_baddbmm_bmm(Meta& meta, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm, const std::optional<Tensor>& self_baddbmm = nullopt) {
  // 检查 batch1 张量是否为三维张量
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  // 检查 batch2 张量是否为三维张量
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  // 获取 batch1 和 batch2 张量的大小
  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  // 提取 batch1 张量的相关维度大小
  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  // 提取 batch2 张量的相关维度大小
  int64_t res_cols = batch2_sizes[2];

  // 构建输出张量的大小为 [bs, res_rows, res_cols]
  std::vector<int64_t> output_size {bs, res_rows, res_cols};

  // 检查 batch2 张量的前两个维度是否与 batch1 张量匹配
  TORCH_CHECK(batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size,
              "Expected size for first two dimensions of batch2 tensor to be: [",
              bs, ", ", contraction_size, "] but got: [", batch2_sizes[0], ", ", batch2_sizes[1], "].");

  // 获取 meta 对象中的输出张量，并设置其形状为 output_size，保持张量原始步长
  auto& result = meta.maybe_get_output(0);
  meta.set_output_raw_strided(0, output_size, {}, batch2.options());

  // 检查 result 张量的形状是否与 output_size 匹配，如果不匹配则报错
  const auto result_sizes = result.sizes();
  TORCH_CHECK(result_sizes == output_size,
              "Expected an output tensor with shape [", output_size, "] but got shape ", result_sizes);

  // 初始化输出张量的维度名称列表
  std::vector<Dimname> outnames = {};
  // 如果不是标准矩阵乘法，则执行以下逻辑
  if (!is_bmm) {
    # 检查是否存在有效的 self_baddbmm 值
    if (self_baddbmm.has_value()) {
      # 获取 self_baddbmm 的引用赋值给 self
      const auto& self = self_baddbmm.value();
      # 如果 beta 转为复数形式不等于 0.0，则将 self 的内容复制给 result
      if (beta.toComplexDouble() != 0.0) result.copy_(self);
      # 检查 self 的维度是否为 3，否则抛出异常
      TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
      # 获取 self 的尺寸大小
      const auto self_sizes = self.sizes();
      # 检查 self 的尺寸是否与输出尺寸 output_size 相同，否则抛出异常
      TORCH_CHECK(self_sizes == output_size,
                  "Expected an input tensor shape with shape ", output_size, " but got shape: ", self_sizes);
      # 根据 result, batch1, batch2, self 计算 baddbmm 操作后的输出命名信息
      outnames = namedinference::compute_baddbmm_outnames(result, batch1, batch2, self);
    }
  } else {
    # 根据 result, batch1, batch2 计算 bmm 操作后的输出命名信息
    outnames = namedinference::compute_bmm_outnames(result, batch1, batch2);
  }

  # 如果 outnames 非空，则将其命名信息传播给 result
  namedinference::propagate_names_if_nonempty(
    result,
    outnames
  );
}

// TORCH_META_FUNC(bmm) 的实现，计算两个张量的批量矩阵乘积
TORCH_META_FUNC(bmm)(const Tensor& self, const Tensor& mat2) {
    // 执行通用检查，确保输入张量合法性
    common_checks_baddbmm_bmm(*this, self, mat2, Scalar(0.0), Scalar(1.0), true);
}

// TORCH_META_FUNC(baddbmm) 的实现，计算三个张量的批量加权矩阵乘积
TORCH_META_FUNC(baddbmm)(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
    // 扩展 self 到与 batch1 和 batch2 相同的尺寸
    auto self_ = expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
    // 检查输入张量的数据类型必须一致
    TORCH_CHECK(self.dtype() == batch1.dtype(), "Input dtypes must be the same, got: input ", self.dtype(), ", batch1: ", batch1.dtype(), ", batch2: ", batch2.dtype());
    // 执行通用检查，确保输入张量合法性
    common_checks_baddbmm_bmm(*this, batch1, batch2, beta, alpha, false, *self_);
}

} // namespace meta

namespace native {

// 定义 addr_stub 的调度分发函数

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.det ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 计算置换矩阵 P 的行列式
// 如果是偶置换，则 det(P) = 1；如果是奇置换，则 det(P) = -1
static Tensor lu_det_P(const Tensor& pivots) {
    return (at::arange(1, pivots.size(-1) + 1, pivots.options()) != pivots)
        .sum(-1, /*keepdim=*/false, /*dtype=*/at::kLong)
        .fmod_(2)
        .mul_(-2)
        .add_(1);
}

// 在后向传播中使用的 LU 分解的辅助函数
TORCH_IMPL_FUNC(_linalg_det_out)(const Tensor& A, const Tensor& result, const Tensor& LU, const Tensor& pivots) {
    // info 是一个辅助张量
    auto info = at::empty({0}, A.options().dtype(kInt));
    // 优化：lu_factor_ex 要求输入是 F-contig，否则会复制
    // 如果 A 是连续的且非复数矩阵，则使用其转置，因为 det(A^T) = det(A)
    at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU), const_cast<Tensor&>(pivots), const_cast<Tensor&>(info), A.is_contiguous() && !A.is_complex() ? A.mH() : A);

    // det = det_P * prod(diag(LU))
    at::mul_out(const_cast<Tensor&>(result), lu_det_P(pivots), at::prod(LU.diagonal(0, -2 ,-1), /*dim=*/-1));
}

// 计算矩阵的行列式
Tensor linalg_det(const Tensor& A) {
    return std::get<0>(at::_linalg_det(A));
}

// 计算矩阵的行列式，并将结果写入 result 张量
Tensor& linalg_det_out(const Tensor& A, Tensor& result) {
    auto LU = at::empty({0}, A.options());
    auto pivots = at::empty({0}, A.options().dtype(kInt));
    at::_linalg_det_out(result, LU, pivots, A);
    return result;
}

// torch.det 的别名，等同于 torch.linalg.det
Tensor det(const Tensor& self) {
    return at::linalg_det(self);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.slogdet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 在后向传播中使用的 LU 分解的辅助函数
// 定义函数 _linalg_slogdet_out，计算矩阵的行列式相关信息并输出到指定张量
TORCH_IMPL_FUNC(_linalg_slogdet_out)(const Tensor& A, const Tensor& sign, const Tensor& logabsdet, const Tensor& LU, const Tensor& pivots) {
  // 创建一个空的辅助张量 info，用于后续操作
  auto info = at::empty({0}, A.options().dtype(kInt));
  // 优化：lu_factor_ex 要求输入必须是 F-contiguous，否则会进行复制
  // 如果 A 是连续的且不是复数类型，则使用其转置来进行 LU 分解，因为 det(A^T) = det(A)
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU), const_cast<Tensor&>(pivots), const_cast<Tensor&>(info), A.is_contiguous() && !A.is_complex() ? A.mH() : A);

  // 提取 LU 分解结果的主对角线元素
  auto diag_U = LU.diagonal(0, -2, -1);
  // 计算行列式的符号部分
  at::mul_out(const_cast<Tensor&>(sign), diag_U.sgn().prod(-1), lu_det_P(pivots));

  // 计算行列式的绝对值的自然对数部分
  at::sum_out(const_cast<Tensor&>(logabsdet), diag_U.abs().log_(), -1);
}

// 计算矩阵 A 的行列式的符号和绝对值的自然对数，并返回结果
std::tuple<Tensor, Tensor> linalg_slogdet(const Tensor& A) {
  // 调用 _linalg_slogdet 函数获取结果
  auto out = at::_linalg_slogdet(A);
  // 将结果打包成元组并返回
  return std::make_tuple(std::move(std::get<0>(out)), std::move(std::get<1>(out)));
}

// 在给定输出张量中计算矩阵 A 的行列式的符号和绝对值的自然对数
std::tuple<Tensor&, Tensor&> linalg_slogdet_out(const Tensor& A, Tensor& sign, Tensor& logabsdet) {
  // 创建空张量 LU 和 pivots
  auto LU = at::empty({0}, A.options());
  auto pivots = at::empty({0}, A.options().dtype(kInt));
  // 调用 _linalg_slogdet_out 函数计算结果并存储到输出张量中
  at::_linalg_slogdet_out(sign, logabsdet, LU, pivots, A);
  // 返回结果张量的引用
  return std::tie(sign, logabsdet);
}

// 别名函数，调用 linalg_slogdet 函数并返回其结果
std::tuple<Tensor, Tensor> slogdet(const Tensor& A) {
  return at::linalg_slogdet(A);
}

// 在给定输出张量中计算矩阵 A 的行列式的符号和绝对值的自然对数
std::tuple<Tensor&, Tensor&> slogdet_out(const Tensor& A, Tensor& sign, Tensor& logabsdet) {
  return at::linalg_slogdet_out(sign, logabsdet, A);
}

// 计算矩阵 A 的对数行列式
Tensor logdet(const Tensor& A) {
  // 检查输入是否为方阵
  squareCheckInputs(A, "logdet");
  // 检查输入张量类型是否为浮点数或复数
  checkFloatingOrComplex(A, "logdet", /*low_precision*/false);
  // 调用 linalg_slogdet 函数计算行列式的符号和绝对值的自然对数
  auto [sign, logabsdet] = at::linalg_slogdet(A);

  // 如果 A 是复数类型，则返回符号的自然对数加上绝对值的自然对数
  if (A.is_complex()) {
    return sign.log() + logabsdet;
  } else {
    // 如果 A 是实数类型，则在符号为 -1 时返回 NaN，否则返回绝对值的自然对数
    return at::where(sign == -1., NAN, logabsdet);
  }
}

// 命名空间中定义的私有函数，用于从输入中提取可选的 atol 和 rtol 张量
std::tuple<Tensor, Tensor> get_atol_rtol(
    const Tensor& input,
    const optional<Tensor>& atol_opt,
    const optional<Tensor>& rtol_opt,
    const c10::string_view function_name) {
  // 获取输入张量的选项
  auto options = input.options();
  // 如果输入张量在 Metal 或 MPS 设备上，则设置其数据类型为 Float
  if (input.device().type() == kMetal || input.device().type() == kMPS) {
    options = options.dtype(ScalarType::Float);
  } else {
    // 否则，设置数据类型为 Double
    options = options.dtype(ScalarType::Double);
  }
  // 如果提供了 atol 参数，则使用提供的值，否则创建一个值为零的张量
  auto atol = atol_opt.has_value() ? atol_opt.value() : at::zeros({}, options);
  // 检查 atol 张量是否为复数类型
  checkNotComplexTolerance(atol, function_name, "atol");
  // 创建 rtol 张量
  Tensor rtol;
  // 如果提供了 rtol 参数，则使用提供的值，否则创建一个值为 eps*max(rows, cols) 的默认张量
  if (rtol_opt.has_value()) {
    rtol = rtol_opt.value();
    // 检查 rtol 张量是否为复数类型
    checkNotComplexTolerance(rtol, function_name, "rtol");
  } else {
    // 否则，创建一个值为零的张量

    rtol = at::zeros({}, options);
  }

  // 返回提取到的 atol 和 rtol 张量
  return std::make_tuple(atol, rtol);
}
    // 将输入的标量类型转换为实数类型
    ScalarType real_dtype = toRealValueType(input.scalar_type());

    // 计算默认的相对容差，默认容差值为 epsilon * 输入张量的最大对称大小
    auto default_rtol = at::full({}, _get_epsilon(real_dtype) * std::max(input.sym_size(-1), input.sym_size(-2)), options);

    // 如果指定了绝对容差的值，则根据其值来选择返回零张量或默认相对容差张量
    rtol = atol_opt.has_value()
           ? at::where(atol_opt.value() > 0, at::zeros({}, options), default_rtol)
           : std::move(default_rtol);
  }

  // 返回计算得到的绝对容差和相对容差作为元组
  return std::make_tuple(atol, rtol);
// 匿名命名空间结束，用于限定非导出的函数和变量的作用域
} // anonymous namespace

// 计算矩阵的伪逆，支持指定公差和相对公差，选择使用奇异值分解或特征值分解
Tensor linalg_pinv(
    const Tensor& input,               // 输入张量
    const optional<Tensor>& atol_opt,  // 绝对公差的可选张量
    const optional<Tensor>& rtol_opt,  // 相对公差的可选张量
    bool hermitian) {                  // 是否厄米特矩阵

  // FIXME: 当我们有一个好的最小二乘解时，应将此函数调度为
  // `torch.lstsq(A, torch.eye(A.shape[-1]), atol=atol, rtol=rtol)`
  // 使用支持奇异输入的驱动程序
  NoTF32Guard disable_tf32;  // 禁用TF32计算

  ScalarType t = input.scalar_type();  // 获取输入张量的标量类型
  // 检查输入张量的类型和维度是否满足要求
  TORCH_CHECK((t == ScalarType::Double || t == ScalarType::Float || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble)
              && input.dim() >= 2,
              "linalg.pinv(", t, "{", input.sizes(), "}): 期望至少两个维度的 float、double、cfloat 或 cdouble 类型的张量");

  // 获取绝对公差和相对公差的张量
  auto [atol, rtol] = get_atol_rtol(input, atol_opt, rtol_opt);

  if (input.sym_numel() == 0) {
    // 对于零元素的张量，由于下面的实现使用的操作无法工作
    // 因此对于 'input.numel() == 0' 情况，我们需要提前返回
    // TODO: 当 torch/xla 可以使用 at::linalg_svd 替换 input.svd 时，进行替换
    auto [U, S, V] = input.svd();
    return at::matmul(V * S.reciprocal().unsqueeze(-2), U.mH());
  }

  // 如果不是厄米特矩阵，则使用奇异值分解，否则使用特征值分解
  if (!hermitian) {
    // TODO: 使用 linalg_svd 替换 input.svd
    // 使用 linalg_svd 会破坏 pytorch/xla，参见 https://github.com/pytorch/xla/issues/2755
    auto [U, S, V] = input.svd();
    Tensor max_val = at::narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);  // 奇异值按降序排序
    Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_val);
    // 计算伪逆矩阵 V @ diag(S_pseudoinv) @ U.conj().T
    Tensor S_pseudoinv = at::where(S > tol, S.reciprocal(), at::zeros({}, S.options())).to(input.dtype());
    return at::matmul(V * S_pseudoinv.unsqueeze(-2), U.mH());
  } else {
    auto [S, U] = at::linalg_eigh(input);
    // 计算 Hermitian 矩阵的奇异值的绝对值
    Tensor S_abs = S.abs();
    // 将奇异值按升序排序，从负值开始，需要获取绝对值的最大值
    Tensor max_val = S_abs.amax(/*dim=*/-1, /*keepdim=*/true);
    // 计算公差，取最大值
    Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_val);
    // 计算广义逆矩阵 S_pseudoinv，当奇异值的绝对值大于公差时取其倒数，否则置零
    Tensor S_pseudoinv = at::where(S_abs > tol, S.reciprocal(), at::zeros({}, S.options())).to(input.dtype());
    // 计算 U @ diag(S_pseudoinv) @ U.conj().T，这是伪逆的计算方式
    return at::matmul(U * S_pseudoinv.unsqueeze(-2), U.mH());
}
}

// 定义一个函数 linalg_pinv，接受输入张量 input，可选参数 atol、rtol，以及布尔值 hermitian
Tensor linalg_pinv(const Tensor& input, optional<double> atol, optional<double> rtol, bool hermitian) {
  // 调用辅助函数 get_atol_rtol 获取相对容差参数 atol_tensor 和 rtol_tensor
  auto [atol_tensor, rtol_tensor] = get_atol_rtol(input, atol, rtol);
  // 调用 ATen 库函数 at::linalg_pinv，传入 input、atol_tensor、rtol_tensor 和 hermitian 参数
  return at::linalg_pinv(input, atol_tensor, rtol_tensor, hermitian);
}

// 定义一个函数 linalg_pinv，接受输入张量 input 和张量 rcond，以及布尔值 hermitian
Tensor linalg_pinv(const Tensor& input, const Tensor& rcond, bool hermitian) {
  // 为了与 NumPy 兼容性，rcond 参数被用作相对容差
  checkNotComplexTolerance(rcond, "torch.linalg.pinv", "rcond");
  // 获取输入张量的选项
  auto options = input.options();
  // 如果输入张量的设备类型为 kMetal 或 kMPS，则将选项设定为 Float 类型
  if (input.device().type() == kMetal || input.device().type() == kMPS) {
    options = options.dtype(ScalarType::Float);
  } else {
    // 否则将选项设定为 Double 类型
    options = options.dtype(ScalarType::Double);
  }
  // 调用 ATen 库函数 at::linalg_pinv，传入 input、at::zeros 创建的零张量、rcond 和 hermitian 参数
  return at::linalg_pinv(input, at::zeros({}, options), rcond, hermitian);
}

// 定义一个函数 linalg_pinv，接受输入张量 input 和 double 类型的 rcond，以及布尔值 hermitian
Tensor linalg_pinv(const Tensor& input, double rcond, bool hermitian) {
  // 为了与 NumPy 兼容性，rcond 参数被用作相对容差
  return at::linalg_pinv(input, 0.0, rcond, hermitian);
}

// 定义一个函数 linalg_pinv_out，实现对输出张量 result 的直接使用避免拷贝，并使用已分配的存储空间
Tensor& linalg_pinv_out(
    const Tensor& input,
    const optional<Tensor>& atol,
    const optional<Tensor>& rtol,
    bool hermitian,
    Tensor& result) {
  // 检查结果张量 result 和输入张量 input 的设备是否相同
  checkSameDevice("linalg.pinv", result, input);
  // 检查结果张量 result 和输入张量 input 的类型是否兼容
  checkLinalgCompatibleDtype("linalg.pinv", result, input);
  // 调用 ATen 库函数 at::linalg_pinv，传入 input、atol、rtol 和 hermitian 参数，将结果存储在 result_tmp 中
  Tensor result_tmp = at::linalg_pinv(input, atol, rtol, hermitian);
  // 调整输出张量 result 的大小以匹配 result_tmp 的大小
  at::native::resize_output(result, result_tmp.sizes());
  // 将 result_tmp 的数据复制到 result 中
  result.copy_(result_tmp);
  // 返回更新后的结果张量 result
  return result;
}

// 定义一个函数 linalg_pinv_out，实现对输出张量 result 的直接使用避免拷贝，并使用已分配的存储空间
Tensor& linalg_pinv_out(
    const Tensor& input,
    optional<double> atol,
    optional<double> rtol,
    bool hermitian,
    Tensor& result) {
  // 检查结果张量 result 和输入张量 input 的设备是否相同
  checkSameDevice("linalg.pinv", result, input);
  // 检查结果张量 result 和输入张量 input 的类型是否兼容
  checkLinalgCompatibleDtype("linalg.pinv", result, input);
  // 调用 ATen 库函数 at::linalg_pinv，传入 input、atol、rtol 和 hermitian 参数，将结果存储在 result_tmp 中
  Tensor result_tmp = at::linalg_pinv(input, atol, rtol, hermitian);
  // 调整输出张量 result 的大小以匹配 result_tmp 的大小
  at::native::resize_output(result, result_tmp.sizes());
  // 将 result_tmp 的数据复制到 result 中
  result.copy_(result_tmp);
  // 返回更新后的结果张量 result
  return result;
}

// 定义一个函数 linalg_pinv_out，实现对输出张量 result 的直接使用避免拷贝，并使用已分配的存储空间
Tensor& linalg_pinv_out(const Tensor& input, const Tensor& rcond, bool hermitian, Tensor& result) {
  // 检查结果张量 result 和输入张量 input 的设备是否相同
  checkSameDevice("linalg.pinv", result, input);
  // 检查结果张量 result 和输入张量 input 的类型是否兼容
  checkLinalgCompatibleDtype("linalg.pinv", result, input);
  // 调用 ATen 库函数 at::linalg_pinv，传入 input、rcond 和 hermitian 参数，将结果存储在 result_tmp 中
  Tensor result_tmp = at::linalg_pinv(input, rcond, hermitian);
  // 调整输出张量 result 的大小以匹配 result_tmp 的大小
  at::native::resize_output(result, result_tmp.sizes());
  // 将 result_tmp 的数据复制到 result 中
  result.copy_(result_tmp);
  // 返回更新后的结果张量 result
  return result;
}

// 定义一个函数 linalg_pinv_out，实现对输出张量 result 的直接使用避免拷贝，并使用已分配的存储空间
Tensor& linalg_pinv_out(const Tensor& input, double rcond, bool hermitian, Tensor& result) {
  // 创建一个与 rcond 值相同的张量 rcond_tensor，使用输入张量 input 的选项和 Double 类型
  Tensor rcond_tensor = at::full({}, rcond, input.options().dtype(ScalarType::Double));
  // 调用 ATen 库函数 at::linalg_pinv_out，传入 result、input、rcond_tensor 和 hermitian 参数
  return at::linalg_pinv_out(result, input, rcond_tensor, hermitian);
}

// 定义一个函数 pinverse，实现对输入张量 self 的伪逆计算，使用 double 类型的 rcond 参数
Tensor pinverse(const Tensor& self, double rcond) {
  // 调用 linalg_pinv 函数计算 self 的伪逆，hermitian 参数设为 false
  return at::linalg_pinv(self, rcond, /*hermitian=*/false);
}

// matrix_power 的实现在此命名空间内进行
namespace {
/**
 * @brief Raises the input matrix to the given power n
 *
 * If the exponent n is negative, the inverse of the input
 * matrix will be raised to power abs(n).
 *
 * @param self (batched) square matrix to raise to power n
 * @param n exponent to raise matrix (or matrices in batch) to
 * @param _out optional tensor to write the output to
 * @return Tensor input matrix raised to power n
 */
Tensor linalg_matrix_power_impl(
    const Tensor& self,
    int64_t n,
    std::optional<Tensor> _out) {
  NoTF32Guard disable_tf32;
  auto out = _out.value_or(Tensor());

  // Check if the input is a square matrix or batched square matrices
  squareCheckInputs(self, "linalg.matrix_power");

  // Resize the output tensor if _out is provided to match the shape of self
  if (_out.has_value()) {
    checkSameDevice("matrix_power", out, self);
    checkLinalgCompatibleDtype("matrix_power", out, self);
    at::native::resize_output_symint(out, self.sym_sizes());
  }

  // For n=0 we return the identity matrix of the same shape as input.
  if (n == 0) {
    if (!_out.has_value()) {
      // Clone input to include result in the autograd graph
      out = self.clone(at::MemoryFormat::Contiguous);
    }
    // Return the identity matrix of appropriate size
    return out.copy_(at::eye_symint(self.sym_size(-2), self.options()));
  }
  // Return the input matrix itself for n=1
  if (n == 1) {
    return _out.has_value() ? out.copy_(self)
                            : self.clone(at::MemoryFormat::Contiguous);
  }
  // Return the inverse of the input matrix for n=-1
  if (n == -1) {
    return _out.has_value() ? at::linalg_inv_out(out, self)
                            : at::linalg_inv(self);
  }

  // Choose whether to use the input matrix or its inverse based on n
  auto a = n < 0 ? at::linalg_inv(self) : self;
  n = std::abs(n);

  // Fast paths for small powers (2 and 3)
  if (n == 2) {
    return _out.has_value() ? at::matmul_out(out, a, a) : at::matmul(a, a);
  }
  if (n == 3) {
    return _out.has_value() ? at::matmul_out(out, at::matmul(a, a), a)
                            : at::matmul(at::matmul(a, a), a);
  }

  // Binary decomposition of n to reduce number of matrix multiplications
  Tensor z, result;
  while (n > 0) {
    const auto bit = n % 2;
    n = n / 2;
    // Square the matrix z or initialize it with a
    z = z.defined() ? at::matmul(z, z) : a;
    if (bit == 1) {
      if (_out.has_value() && n <= 0) {
        // Use out version for the last multiplication
        return result.defined() ? at::matmul_out(out, result, z) : out.copy_(z);
      }
      // Multiply result by z
      result = result.defined() ? at::matmul(result, z) : z;
    }
  }

  return result;
}

} // namespace

/**
 * @brief Compute the matrix power and store the result in the output tensor.
 *
 * @param self (batched) square matrix to raise to power n
 * @param n exponent to raise matrix (or matrices in batch) to
 * @param result output tensor to store the result
 * @return Tensor& reference to the output tensor with the result
 */
Tensor& linalg_matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  linalg_matrix_power_impl(self, n, result);
  return result;
}

/**
 * @brief Compute the matrix power.
 *
 * @param self (batched) square matrix to raise to power n
 * @param n exponent to raise matrix (or matrices in batch) to
 * @return Tensor input matrix raised to power n
 */
Tensor linalg_matrix_power(const Tensor& self, int64_t n) {
  return linalg_matrix_power_impl(self, n, c10::nullopt);
}
// 使用自定义矩阵幂函数 `linalg_matrix_power_out` 计算输入张量 `self` 的 n 次幂，并将结果保存到给定的张量 `result` 中
Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return at::native::linalg_matrix_power_out(self, n, result);
}

// 使用自定义矩阵幂函数 `linalg_matrix_power` 计算输入张量 `self` 的 n 次幂，并返回结果张量
Tensor matrix_power(const Tensor& self, int64_t n) {
  return at::native::linalg_matrix_power(self, n);
}

namespace {

// 计算输入张量 `input` 的秩，并将结果保存在给定的张量 `result` 中
// `hermitian` 控制使用 SVD 还是特征分解计算奇异值
// `atol_opt` 和 `rtol_opt` 是绝对和相对容差
Tensor& matrix_rank_impl(
    const Tensor& input,
    const optional<Tensor>& atol_opt,
    const optional<Tensor>& rtol_opt,
    bool hermitian,
    Tensor& result) {
  // 获取容差值 `atol` 和 `rtol`
  auto [atol, rtol] = get_atol_rtol(input, atol_opt, rtol_opt, "torch.linalg.matrix_rank");

  // 检查 `result` 和 `input` 张量的设备是否相同
  checkSameDevice("torch.linalg.matrix_rank", result, input);
  checkSameDevice("torch.linalg.matrix_rank", atol, input, "atol");
  checkSameDevice("torch.linalg.matrix_rank", rtol, input, "rtol");

  // 检查 `result` 张量的数据类型是否兼容
  ScalarType output_type = ScalarType::Long;
  checkLinalgCompatibleDtype("torch.linalg.matrix_rank", result.scalar_type(), output_type);

  // 检查容差值不能是复数
  checkNotComplexTolerance(atol, "torch.linalg.matrix_rank", "atol");
  checkNotComplexTolerance(rtol, "torch.linalg.matrix_rank", "rtol");

  // 如果输入张量元素数量为 0，直接将结果张量 `result` 填充为 0，并返回
  if (input.sym_numel() == 0) {
    result.fill_(0);
    return result;
  }

  // 根据是否 `hermitian`，选择使用 SVD 或特征分解计算奇异值或绝对特征值
  Tensor S, max_S;
  if (!hermitian) {
    S = at::linalg_svdvals(input);
    // 奇异值按降序排列，取最大值
    max_S = at::narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);
  } else {
    S = at::linalg_eigvalsh(input);
    S = S.abs();
    // 特征值按升序排列，从负值开始，取绝对值的最大值
    max_S = S.amax(/*dim=*/-1, /*keepdim=*/true);
  }

  // 计算容差阈值 `tol`
  Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_S);

  // 如果输入张量是 Tensor 的子类，直接返回大于容差阈值的个数
  if (isTensorSubclassLike(input)) {
     result = at::sum(S > tol, /*dim=*/-1);
     return result;
  }

  // 使用 `sum_out` 函数将大于容差阈值的个数保存到结果张量 `result` 中，并返回
  result = at::sum_out(result, S > tol, /*dim=*/-1);
  return result;
}

// 返回一个与输入张量 `input` 形状相同的结果张量，用于保存 `matrix_rank_impl` 函数的结果
Tensor get_matrix_rank_result_tensor(const Tensor& input) {
  // 检查输入张量 `input` 是否是矩阵或批量矩阵
  checkIsMatrix(input, "torch.linalg.matrix_rank", "input");

  // 为了遵循复合兼容性，分配正确形状的 `result` 张量，避免在 `out` 变体中重新调整大小
  // 参见 `NOTE [matrix rank output shape]`
  auto result_shape =
      SymIntArrayRef(input.sym_sizes().cbegin(), input.sym_sizes().cend() - 2);
  Tensor result =
      at::empty_symint(result_shape, input.options().dtype(ScalarType::Long));

  return result;
}

}  // 匿名命名空间

// 使用 `linalg_matrix_rank_out` 函数计算输入张量的秩，并将结果保存到给定的 `result` 张量中
    // 接受一个常量引用的Tensor作为输入
    const Tensor& input,
    // 可选的绝对容差Tensor，用于数值比较
    const optional<Tensor>& atol_opt,
    // 可选的相对容差Tensor，用于数值比较
    const optional<Tensor>& rtol_opt,
    // 指示是否处理共轭转置的布尔值
    bool hermitian,
    // 输出结果的Tensor引用
    Tensor& result) {
      // 检查输入是否为矩阵或矩阵批处理
      checkIsMatrix(input, "torch.linalg.matrix_rank", "input");
      // 获取结果Tensor的形状，排除最后两个维度
      auto result_shape =
        IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2);
      // 调整输出Tensor的形状以匹配结果Shape
      at::native::resize_output(result, result_shape);
      // 调用具体的矩阵秩计算实现函数，传入输入Tensor及相应的容差参数
      return matrix_rank_impl(input, atol_opt, rtol_opt, hermitian, result);
    }
// 输出参数版本的 linalg_matrix_rank_out 函数，计算输入张量的矩阵秩并将结果存储在提供的输出张量中
Tensor& linalg_matrix_rank_out(const Tensor& input, optional<double> atol, optional<double> rtol, bool hermitian, Tensor& result) {
  // 获取指定的绝对误差和相对误差的张量
  auto [atol_tensor, rtol_tensor] = get_atol_rtol(input, atol, rtol);
  // 调用 linalg_matrix_rank_out 函数计算矩阵秩，并将结果存储在输出张量中
  result = linalg_matrix_rank_out(input, atol_tensor, rtol_tensor, hermitian, result);
  // 返回存储了计算结果的输出张量
  return result;
}

// linalg_matrix_rank 函数的重载版本，计算输入张量的矩阵秩
Tensor linalg_matrix_rank(const Tensor& input, const optional<Tensor>& atol, const optional<Tensor>& rtol, bool hermitian) {
  // 获取存储结果的张量
  auto result = get_matrix_rank_result_tensor(input);
  // 调用 matrix_rank_impl 函数计算矩阵秩
  return matrix_rank_impl(input, atol, rtol, hermitian, result);
}

// linalg_matrix_rank 函数的重载版本，计算输入张量的矩阵秩
Tensor linalg_matrix_rank(const Tensor& input, optional<double> atol, optional<double> rtol, bool hermitian) {
  // 获取存储结果的张量
  auto result = get_matrix_rank_result_tensor(input);
  // 获取指定的绝对误差和相对误差的张量
  auto [atol_tensor, rtol_tensor] = get_atol_rtol(input, atol, rtol);
  // 调用 matrix_rank_impl 函数计算矩阵秩
  return matrix_rank_impl(input, atol_tensor, rtol_tensor, hermitian, result);
}

// 输出参数版本的 linalg_matrix_rank_out 函数，计算输入张量的矩阵秩并将结果存储在提供的输出张量中
Tensor& linalg_matrix_rank_out(const Tensor& input, const Tensor& tol, bool hermitian, Tensor& result) {
  // 根据提供的 tol 参数，创建相对误差的张量 rtol，以保持与 NumPy 兼容性
  Tensor rtol = at::zeros({}, tol.options());
  // 调用 at::linalg_matrix_rank_outf 函数计算矩阵秩，并将结果存储在输出张量中
  result = at::linalg_matrix_rank_outf(input, tol, rtol, hermitian, result);
  // 返回存储了计算结果的输出张量
  return result;
}

// 输出参数版本的 linalg_matrix_rank_out 函数，计算输入张量的矩阵秩并将结果存储在提供的输出张量中
Tensor& linalg_matrix_rank_out(const Tensor& input, double tol, bool hermitian, Tensor& result) {
  // 根据提供的 tol 参数，创建相对误差的张量 rtol，以保持与 NumPy 兼容性
  result = at::linalg_matrix_rank_outf(input, tol, 0.0, hermitian, result);
  // 返回存储了计算结果的输出张量
  return result;
}

// linalg_matrix_rank 函数的重载版本，计算输入张量的矩阵秩
Tensor linalg_matrix_rank(const Tensor& input, const Tensor& tol, bool hermitian) {
  // 获取存储结果的张量
  auto result = get_matrix_rank_result_tensor(input);
  // 创建相对误差的张量 rtol，其值为0，以保持与 NumPy 兼容性
  return matrix_rank_impl(input, tol, at::zeros({}, tol.options()), hermitian, result);
}

// linalg_matrix_rank 函数的重载版本，计算输入张量的矩阵秩
Tensor linalg_matrix_rank(const Tensor& input, double tol, bool hermitian) {
  // 获取存储结果的张量
  auto result = get_matrix_rank_result_tensor(input);
  // 获取指定的绝对误差和相对误差的张量
  auto [atol_tensor, rtol_tensor] = get_atol_rtol(input, tol, 0.0);
  // 调用 matrix_rank_impl 函数计算矩阵秩
  return matrix_rank_impl(input, atol_tensor, rtol_tensor, hermitian, result);
}
/**
 * @brief Computes the optimal matrix chain multiplication order
 *
 * Follows the dynamic programming algorithm from Cormen et al.,
 * "Introduction to Algorithms, Third Edition", Chapter 15.2,
 * p. 370-378. Note that the book uses 1-based indexing.
 *
 * The cost of multiplying two matrices with sizes p x q and q x r
 * is defined here as p * q * r. The optimal multiplication order
 * is the one that minimizes the total cost.
 *
 * @param tensors list of 2D tensors
 * @return a 2D vector s used by #matrix_chain_multiplication to construct
 *         the optimal matrix multiplication order. The optimal multiplication
 *         order for multiplying tensors i...j is to multiply tensors i...s[i, j]
 *         and tensors (s[i, j] + 1)...j first and then the result of that.
 */
std::vector<std::vector<int64_t>> matrix_chain_order(TensorList tensors) {
  // Number of tensors
  const size_t n = tensors.size();

  // Array to hold dimensions of tensors
  std::vector<int64_t> p(n + 1);
  // Extract dimensions p[i] from tensors[i]
  for (const auto i : c10::irange(n)) {
    p[i] = tensors[i].size(0);
  }
  // Last dimension p[n] from tensors[n-1]
  p[n] = tensors[n - 1].size(1);

  // Matrix to store minimum cost of multiplying tensors i...j
  std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));

  // Matrix to store optimal split point k for multiplying tensors i...j
  std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n));

  // Compute the optimal multiplication order
  for (const auto l : c10::irange(1, n)) {
    for (const auto i : c10::irange(n - l)) {
      const auto j = i + l;
      m[i][j] = std::numeric_limits<int64_t>::max();
      for (const auto k : c10::irange(i, j)) {
        // Calculate cost q for multiplying matrices i...k and k+1...j
        const auto q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
        // Update minimum cost and optimal split point
        if (q < m[i][j]) {
          m[i][j] = q;
          s[i][j] = k;
        }
      }
    }
  }

  return s;
}

/**
 * @brief Recursively multiplies the tensors i...j using the given order
 *
 * @param tensors matrices to multiply together
 * @param order optimal chain multiplication order from #matrix_chain_order
 * @param i index of first tensor to be multiplied
 * @param j index of last tensor to be multiplied
 * @return Tensor result of multiplying tensors[i...j] together.
 */
Tensor matrix_chain_multiplication(
    TensorList tensors,
    const std::vector<std::vector<int64_t>>& order,
    int64_t i,
    int64_t j) {
  // Base case: if only one tensor, return it
  if (i == j) {
    return tensors[i];
  }
  // Recursively multiply tensors using the optimal order
  return at::mm(
      matrix_chain_multiplication(tensors, order, i, order[i][j]),
      matrix_chain_multiplication(tensors, order, order[i][j] + 1, j));
}

// Implements torch.linalg.multi_dot
// 定义函数 multi_dot_impl，接受一个 TensorList 和一个可选的输出 Tensor，并返回一个 Tensor
Tensor multi_dot_impl(TensorList _tensors, std::optional<Tensor> _out) {
  // 获取输入 Tensor 的数量 n
  const size_t n = _tensors.size();
  // 检查输入 Tensor 数量是否至少为 2，否则抛出错误信息
  TORCH_CHECK(n >= 2, "multi_dot(): expected at least 2 tensors but got ", n);

  // 定义存储输出形状的向量
  std::vector<int64_t> out_shape;
  // 定义一个 Tensor 的向量 tensors，大小为 n
  std::vector<Tensor> tensors(n);

  // 如果第一个 Tensor 是 1 维的，将其视为行向量 (1, n)
  if (_tensors[0].dim() == 1) {
    tensors[0] = _tensors[0].unsqueeze(0);
  } else if (_tensors[0].dim() == 2) {
    // 如果第一个 Tensor 是 2 维的，直接使用
    tensors[0] = _tensors[0];
    // 记录输出形状的第一个维度大小
    out_shape.emplace_back(tensors[0].size(0));
  } else {
    // 如果第一个 Tensor 的维度既不是 1 维也不是 2 维，抛出错误信息
    TORCH_CHECK(
        false,
        "multi_dot(): the first tensor must be 1D or 2D but got ",
        _tensors[0].dim(),
        "D");
  }

  // 如果最后一个 Tensor 是 1 维的，将其视为列向量 (n, 1)
  if (_tensors[n - 1].dim() == 1) {
    tensors[n - 1] = _tensors[n - 1].unsqueeze(-1);
  } else if (_tensors[n - 1].dim() == 2) {
    // 如果最后一个 Tensor 是 2 维的，直接使用
    tensors[n - 1] = _tensors[n - 1];
    // 记录输出形状的第二个维度大小
    out_shape.emplace_back(tensors[n - 1].size(1));
  } else {
    // 如果最后一个 Tensor 的维度既不是 1 维也不是 2 维，抛出错误信息
    TORCH_CHECK(
        false,
        "multi_dot(): the last tensor must be 1D or 2D but got ",
        _tensors[n - 1].dim(),
        "D");
  }

  // 确保中间的 Tensor 都是 2 维的
  for (const auto i : c10::irange(1, n - 1)) {
    TORCH_CHECK(
        _tensors[i].dim() == 2,
        "multi_dot(): tensor ",
        i,
        " must be 2D but got ",
        _tensors[i].dim(),
        "D");
    // 将中间的 Tensor 直接使用
    tensors[i] = _tensors[i];
  }

  // 确保所有的 Tensor 具有相同的设备和数据类型，并检查它们的形状是否可以相乘
  const auto dtype = tensors[0].dtype();
  const auto device = tensors[0].device();
  for (const auto i : c10::irange(1, n)) {
    TORCH_CHECK(
        tensors[i].dtype() == dtype,
        "multi_dot(): all tensors must have be the same dtype but tensor 0 is ",
        dtype,
        " and tensor ",
        i,
        " ",
        tensors[i].dtype());
    TORCH_CHECK(
        tensors[i].device() == device,
        "multi_dot(): all tensors must be on the same device but tensor 0 is on ",
        device,
        " and tensor ",
        i,
        " on ",
        tensors[i].device());
    TORCH_CHECK(
        tensors[i - 1].size(-1) == tensors[i].size(0),
        "multi_dot(): tensors ",
        i - 1,
        " and ",
        i,
        " with shapes ",
        _tensors[i - 1].sizes(),
        " and ",
        _tensors[i].sizes(),
        " cannot be multiplied")
  }

  // 定义结果 Tensor
  Tensor result;

  // 如果提供了输出 Tensor _out
  if (_out.has_value()) {
    auto out = *_out;
    // 检查输出 Tensor 的数据类型是否与输入 Tensor 的相同
    TORCH_CHECK(
        dtype == out.dtype(),
        "multi_dot(): expected out tensor to have dtype ",
        dtype,
        " but got ",
        out.dtype());
    // 检查输出 Tensor 的设备是否与输入 Tensor 的相同
    TORCH_CHECK(
        device == out.device(),
        "multi_dot(): expected out tensor to be on device ",
        device,
        " but got ",
        out.device());

    // 如果第一个和最后一个 Tensor 的形状分别为 (a, b) 和 (b, c)，则输出的形状为 (a, c)。
    // 如果第一个或最后一个 Tensor 是 1 维的，则...
    // 调整输出张量的大小以匹配指定的形状
    at::native::resize_output(out, out_shape);

    // 将输出视图为二维张量，简化计算
    result = out.view({tensors[0].size(0), tensors.back().size(-1)});
  }

  // 下面的 resize_ 和 view 调用是为了确保输出形状尊重作为二维视图的第一个和最后一个张量的原始维度

  if (tensors.size() == 2) {
    // 如果张量数量为2，则执行矩阵乘法，并返回结果
    return _out.has_value() ? at::mm_out(result, tensors[0], tensors[1])
                         : at::mm(tensors[0], tensors[1]).view(out_shape);
  }

  // 为什么针对3个矩阵进行单独的实现？
  // 直接处理三个矩阵的逻辑要比多次比较和更多算术操作更快
  if (tensors.size() == 3) {
    const auto a = tensors[0].size(0);
    const auto b = tensors[1].size(0);
    const auto c = tensors[2].size(0);
    const auto d = tensors[2].size(1);

    // 这些矩阵的大小分别为 (a x b), (b x c), (c x d)
    // cost_1 是将 (a x b) 和 (b x c) 进行括号化，然后合并 (c x d) 的成本
    // cost_2 是将 (b x c) 和 (c x d) 进行括号化，然后合并 (a x b) 的成本
    const auto cost_1 = (a * c) * (b + d);
    const auto cost_2 = (b * d) * (a + c);

    if (cost_1 > cost_2) {
      // 根据计算成本选择更有效的矩阵乘法顺序，并返回结果
      return _out.has_value()
          ? at::mm_out(result, tensors[0], at::mm(tensors[1], tensors[2]))
          : at::mm(tensors[0], at::mm(tensors[1], tensors[2])).view(out_shape);
    } else {
      return _out.has_value()
          ? at::mm_out(result, at::mm(tensors[0], tensors[1]), tensors[2])
          : at::mm(at::mm(tensors[0], tensors[1]), tensors[2]).view(out_shape);
    }
  }

  // 多个矩阵乘法的算法
  const auto order = matrix_chain_order(tensors);
  const int64_t i = 0;
  const int64_t j = n - 1;

  if (_out.has_value()) {
    // 在这里手动实现第一个递归层，以便可以在最终乘法中使用 mm_out
    return at::mm_out(
        result,
        matrix_chain_multiplication(tensors, order, i, order[i][j]),
        matrix_chain_multiplication(tensors, order, order[i][j] + 1, j));
  }
  // 返回多个矩阵乘法的结果，并调整输出形状
  return matrix_chain_multiplication(tensors, order, i, j).view(out_shape);
}

} // namespace



Tensor linalg_multi_dot(TensorList tensors) {
  // 调用 multi_dot_impl 函数，并返回其结果
  return multi_dot_impl(tensors, c10::nullopt);
}



Tensor& linalg_multi_dot_out(TensorList tensors, Tensor& result) {
  // 调用 multi_dot_impl 函数，将结果存储到 result 中，并返回 result 引用
  multi_dot_impl(tensors, result);
  return result;
}



Tensor chain_matmul(TensorList matrices) {
  // 发出一次性警告消息，指出 chain_matmul 函数即将被移除，推荐使用 torch.linalg.multi_dot
  TORCH_WARN_ONCE(
      "torch.chain_matmul is deprecated and will be removed in a future PyTorch release. ",
      "Use torch.linalg.multi_dot instead, which accepts a list of two or more tensors rather than ",
      "multiple parameters."
  );
  // 检查所有矩阵的维度是否相同，都应该是二维的
  checkAllSameDim(matrices, 2);

  // 检查矩阵列表不能为空
  TORCH_CHECK(
      !matrices.empty(), "chain_matmul(): Expected one or more matrices");

  // 如果只有一个矩阵，返回其克隆副本
  if (matrices.size() == 1) {
    return matrices[0].clone();
  }

  // 否则调用 at::native::linalg_multi_dot 函数进行矩阵链乘操作并返回结果
  return at::native::linalg_multi_dot(matrices);
}



Tensor& chain_matmul_out(TensorList matrices, Tensor& result) {
  // 发出一次性警告消息，指出 chain_matmul 函数即将被移除，推荐使用 torch.linalg.multi_dot
  TORCH_WARN_ONCE(
      "torch.chain_matmul is deprecated and will be removed in a future PyTorch release. ",
      "Use torch.linalg.multi_dot instead, which accepts a list of two or more tensors rather than ",
      "multiple parameters."
  );
  // 检查所有矩阵的维度是否相同，都应该是二维的
  checkAllSameDim(matrices, 2);

  // 检查矩阵列表不能为空
  TORCH_CHECK(
      !matrices.empty(), "chain_matmul(): Expected one or more matrices");

  // 如果只有一个矩阵，调整输出 result 的大小并复制该矩阵的内容到 result 中
  if (matrices.size() == 1) {
    at::native::resize_output(result, matrices[0].sizes());
    return result.copy_(matrices[0]);
  }

  // 否则调用 at::native::linalg_multi_dot_out 函数进行矩阵链乘操作并返回结果
  return at::native::linalg_multi_dot_out(matrices, result);
}



static void check_1d(const Tensor& t, const char* arg, const char* fn) {
  // 检查张量 t 是否为一维，如果不是则抛出错误
  TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
}



static void check_addr_scalar(const ScalarType dtype,
                              const Scalar& scalar,
                              const std::string& scalar_name) {
  // 检查标量类型和值是否符合要求，根据类型的不同进行不同的检查
  TORCH_CHECK(
    !scalar.isBoolean() || dtype == ScalarType::Bool,
    "Boolean ", scalar_name, " only supported for Boolean results.");
  TORCH_CHECK(
    isFloatingType(dtype) || isComplexType(dtype) || scalar.isIntegral(true),
    "For integral input tensors, "
    "argument ", scalar_name ," must not be a floating point number.");
}



static TensorIterator build_addr_iter(Tensor& result,
                                      const Tensor& self,
                                      const Tensor& vec1,
                                      const Tensor& vec2) {
  // 检查 vec1 和 vec2 是否为一维张量
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");

  // 获取 vec1 和 vec2 的大小，并对 self 进行扩展或直接使用，确保其满足 addr 函数的要求
  const auto vec1_size0 = vec1.sizes()[0];
  const auto vec2_size0 = vec2.sizes()[0];
  auto self_ = &result == &self
    ? c10::MaybeOwned<Tensor>::borrowed(self)
    : expand_size(self, {vec1_size0, vec2_size0}, "addr");
  
  // 检查 self 是否为二维张量，并且其大小与 vec1 和 vec2 匹配
  TORCH_CHECK(
    self_->dim() == 2,
    "2D tensor expected, got ", self_->dim(), "D tensor for input"
  );
  TORCH_CHECK(
    self_->sizes()[0] == vec1_size0 && self_->sizes()[1] == vec2_size0,
    "size mismatch, input: ", self_->sizes(),
    ", v1: ", vec1.sizes(),
    ", v2: ", vec2.sizes()
  );

  // 配置并返回一个张量迭代器，用于执行张量计算操作
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .add_output(result)
    # 将 result 添加为操作的输出

    .add_owned_const_input(*self_)
    # 将 self_ 的内容作为操作的拥有常量输入

    .add_owned_const_input(vec1.reshape({vec1_size0, 1}))
    # 将 vec1 根据指定的形状进行重塑，并将重塑后的结果作为操作的拥有常量输入

    .add_const_input(vec2)
    # 将 vec2 作为操作的常量输入

    .allow_cpu_scalars(true)
    # 允许操作使用 CPU 标量

    .promote_inputs_to_common_dtype(true)
    # 将操作的输入提升为公共数据类型

    .cast_common_dtype_to_outputs(true)
    # 将公共数据类型转换为操作的输出类型

    .enforce_safe_casting_to_output(true)
    # 强制安全地将类型转换应用到输出

    .build();
    # 构建操作

  return iter;
    # 返回迭代器对象 iter
}

// 定义 addr 函数，用于执行向量化外积运算并添加到给定的 self 张量
Tensor addr(const Tensor& self,
            const Tensor& vec1, const Tensor& vec2,
            const Scalar& beta, const Scalar& alpha) {
  // 创建一个新的张量 result 用于存储结果
  Tensor result;
  // 构建 addr 迭代器，用于处理 self、vec1 和 vec2 的外积操作
  auto iter = build_addr_iter(result, self, vec1, vec2);

  // 检查 beta 和 alpha 是否为标量，并验证类型
  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  // 调用底层的 addr_stub 函数执行具体的外积操作
  addr_stub(iter.device_type(), iter, beta, alpha);
  // 返回外积结果的输出张量
  return iter.output();
}

// 定义 addr_ 函数，直接修改 self 张量为 addr 运算的结果
Tensor& addr_(Tensor& self,
              const Tensor& vec1, const Tensor& vec2,
              const Scalar& beta, const Scalar& alpha) {
  // 调用 at::addr_out 函数，将结果存储到 self 张量中
  return at::addr_out(self, self, vec1, vec2, beta, alpha);
}

// 定义 addr_out 函数，执行外积运算并将结果存储在给定的 result 张量中
Tensor& addr_out(const Tensor& self,
                 const Tensor& vec1, const Tensor& vec2,
                 const Scalar& beta, const Scalar& alpha, Tensor &result) {
  // 构建 addr 迭代器，用于处理 self、vec1 和 vec2 的外积操作
  auto iter = build_addr_iter(result, self, vec1, vec2);

  // 检查 beta 和 alpha 是否为标量，并验证类型
  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  // 调用底层的 addr_stub 函数执行具体的外积操作
  addr_stub(iter.device_type(), iter, beta, alpha);
  // 返回存储结果的 result 张量的引用
  return result;
}

// math_addr 和 math_addr_out 函数支持 CPU、CUDA 以外的后端，例如 XLA
// 它们是使用现有操作的组合来实现的

Tensor math_addr(const Tensor& self,
                 const Tensor& vec1, const Tensor& vec2,
                 const Scalar& beta, const Scalar& alpha) {
  // 当 beta == 0 时，忽略 self 中的值，确保 nans 和 infs 不传播
  Tensor out;
  if (beta.toComplexDouble() == 0.0) {
    if (alpha.toComplexDouble() == 1.0) {
      // 直接计算 vec1 和 vec2 的外积
      out = at::outer(vec1, vec2);
    } else {
      // 计算 alpha 乘以 vec1 和 vec2 的外积
      out = alpha * at::outer(vec1, vec2);
    }
  } else if (beta.toComplexDouble() == 1.0) {
    if (alpha.toComplexDouble() == 1.0) {
      // 将 self 与 vec1 和 vec2 的外积相加
      out = self + at::outer(vec1, vec2);
    } else {
      // 将 self 与 alpha 乘以 vec1 和 vec2 的外积相加
      out = self + alpha * at::outer(vec1, vec2);
    }
  } else if (alpha.toComplexDouble() == 1.0) {
    // 将 beta 乘以 self 后与 vec1 和 vec2 的外积相加
    out = beta * self + at::outer(vec1, vec2);
  } else {
    // 将 beta 和 alpha 各自乘以 self 和 vec1、vec2 的外积后相加
    out = beta * self + alpha * at::outer(vec1, vec2);
  }
  // 推断结果类型并将结果转换为相应的张量类型
  auto result_type = c10::promoteTypes(c10::promoteTypes(self.scalar_type(), vec1.scalar_type()), vec2.scalar_type());
  return out.to(c10::TensorOptions().dtype(result_type));
}

// math_addr_out 函数，执行 math_addr 并将结果存储在给定的 result 张量中
Tensor& math_addr_out(const Tensor& self,
                      const Tensor& vec1, const Tensor& vec2,
                      const Scalar& beta, const Scalar& alpha, Tensor &result) {
  // 调用 at::addr 函数计算结果
  auto addr_result = at::addr(self, vec1, vec2, beta, alpha);

  // 验证安全转换
  const auto result_dtype = addr_result.scalar_type();
  TORCH_CHECK(canCast(result_dtype, result.scalar_type()),
              "result type ", result_dtype,
              " can't be cast to the desired output type ", result.scalar_type());

  // 调整输出张量的大小并复制结果
  at::native::resize_output(result, addr_result.sizes().vec());
  result.copy_(addr_result);
  // 返回存储结果的 result 张量的引用
  return result;
}

// torch.ger 是 torch.outer 的别名
Tensor& ger_out(const Tensor& self, const Tensor& vec2, Tensor &result) {
  // 发出警告消息，提示 torch.ger 将在未来的 PyTorch 版本中移除，建议使用 torch.outer 代替
  TORCH_WARN("torch.ger is deprecated and will be removed in a future PyTorch release. "
             "Use torch.outer instead.");
  // 调用 outer_out 函数计算 self 和 vec2 的外积，并将结果存入 result 中
  return at::outer_out(result, self, vec2);
}

Tensor ger(const Tensor& self, const Tensor& vec2) {
  // 返回 self 和 vec2 的外积
  return self.outer(vec2);
}

Tensor& inner_out(const Tensor& self, const Tensor& other, Tensor& out) {
  // 检查输出张量 out、self 和 other 的设备类型是否一致
  checkDeviceType("inner()", {out, self, other}, self.device().type());

  // 如果 self 或 other 是标量，则直接相乘并存入 out 中
  if (self.dim() == 0 || other.dim() == 0) {
    at::mul_out(out, self, other);
    return out;
  }

  // 检查 self 和 other 的最后一个维度是否匹配
  TORCH_CHECK(
      self.size(-1) == other.size(-1),
      "inner() the last dimension must match on both input tensors but got shapes ",
      self.sizes(),
      " and ",
      other.sizes());

  // 计算 self 和 other 的张量内积，并将结果存入 out 中
  at::tensordot_out(out, self, other, -1, -1);
  return out;
}

Tensor inner(const Tensor& self, const Tensor& other) {
  // 检查 self 和 other 的设备类型是否一致
  checkDeviceType("inner()", {self, other}, self.device().type());

  // 如果 self 或 other 是标量，则直接相乘并返回结果
  if (self.dim() == 0 || other.dim() == 0) {
    return self * other;
  }

  // 检查 self 和 other 的最后一个维度是否匹配
  TORCH_CHECK(
      self.sym_size(-1) == other.sym_size(-1),
      "inner() the last dimension must match on both input tensors but got shapes ",
      self.sym_sizes(),
      " and ",
      other.sym_sizes());

  // 计算 self 和 other 的张量内积并返回结果
  return at::tensordot(self, other, -1, -1);
}

Tensor& outer_out(const Tensor& self, const Tensor& vec2, Tensor &result) {
  // 检查 self 和 vec2 是否为一维张量，如果不是则抛出错误
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");

  // 使用 mul_out 函数计算 self 和 vec2 的外积，并将结果存入 result 中
  at::mul_out(result, self.reshape({self.size(0), 1}), vec2);
  return result;
}

Tensor outer(const Tensor& self, const Tensor& vec2) {
  // 检查 self 和 vec2 是否为一维张量，如果不是则抛出错误
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");

  // 返回 self 和 vec2 的外积
  return self.reshape_symint({self.sym_size(0), 1}) * vec2;
}


#if !defined(C10_MOBILE)
#define _AT_DISPATCH_ADDMM_TYPES(TYPE, NAME, ...)                                               \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(                                                 \
            kBFloat16, kHalf, kFloat8_e5m2, kFloat8_e4m3fn, kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, \
            TYPE, NAME, __VA_ARGS__)
#else
// Include half dtype in ADDMM. Used to build ExecuTorch in xplat.
#if defined(C10_MOBILE_HALF)
#define _AT_DISPATCH_ADDMM_TYPES(TYPE, NAME, ...)        \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, \
            TYPE, NAME, __VA_ARGS__)
#else
#define _AT_DISPATCH_ADDMM_TYPES(TYPE, NAME, ...)        \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, \
            TYPE, NAME, __VA_ARGS__)
#endif
#endif


static inline int64_t get_mkldnn_matmul_min_dim() {
  static auto value = [&] {
    const int64_t default_min_dim = [&] {
      // 定义一个 lambda 表达式，捕获外部所有变量（&）
      // 这个 lambda 表达式用于计算 MKLDNN 的最小维度需求，基于实验得出
      // 默认情况下，它仅在 Neoverse V1 上启用
#if !defined(__s390x__)  && !defined(__powerpc__)
      // 如果不是 s390x 和 powerpc 架构
      if (cpuinfo_initialize() && cpuinfo_get_uarchs_count() == 1 && cpuinfo_get_uarch(0)->uarch == cpuinfo_uarch_neoverse_v1) {
        // 初始化 CPU 信息并且仅有一个微架构，且该微架构是 Neoverse V1
        return 8;
      }
#endif
      // 返回默认值 0
      return 0;
    }();
    // 获取环境变量 TORCH_MKLDNN_MATMUL_MIN_DIM 的值
    const char* ptr = std::getenv("TORCH_MKLDNN_MATMUL_MIN_DIM");
    // 如果环境变量存在，将其转换为整数返回；否则返回默认值
    return ptr != nullptr ? std::atoi(ptr) : default_min_dim;
  }();
  // 返回嵌套的 lambda 函数计算的值
  return value;
}


static inline int64_t get_mkldnn_matmul_min_size() {
  // 静态变量 value 包含一个 lambda 函数
  static auto value = [&] {
    // lambda 函数定义了一个内部 lambda 函数，计算默认的最小尺寸
    const int64_t default_min_size = [&] {
      // Minimum size requirement for MKLDNN; derived based on experiments.
      // By default, it's only enabled on Neoverse V1.
#if !defined(__s390x__)  && !defined(__powerpc__)
      // 如果不是 s390x 和 powerpc 架构
      if (cpuinfo_initialize() && cpuinfo_get_uarchs_count() == 1 && cpuinfo_get_uarch(0)->uarch == cpuinfo_uarch_neoverse_v1) {
        // 初始化 CPU 信息并且仅有一个微架构，且该微架构是 Neoverse V1
        return 8 * 1024;
      }
#endif
      // 返回默认值 0
      return 0;
    }();
    // 获取环境变量 TORCH_MKLDNN_MATMUL_MIN_SIZE 的值
    const char* ptr = std::getenv("TORCH_MKLDNN_MATMUL_MIN_SIZE");
    // 如果环境变量存在，将其转换为整数返回；否则返回默认的最小尺寸
    return ptr != nullptr ? std::atoi(ptr) : default_min_size;
  }();
  // 返回嵌套的 lambda 函数计算的值
  return value;
}


static inline bool apply_mkldnn_matmul_heur(int64_t m, int64_t k, int64_t n) {
  // 获取最小维度和最小尺寸
  const int64_t min_dim = get_mkldnn_matmul_min_dim();
  const int64_t min_size = get_mkldnn_matmul_min_size();
  // 应用 MKLDNN 矩阵乘法的启发式规则
  return at::globalContext().userEnabledMkldnn() && m > min_dim && k > min_dim && n > min_dim && m * k * n > min_size;
}


static void addmm_impl_cpu_(
    Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {
  // 检查输入张量的维度是否为 2
  TORCH_INTERNAL_ASSERT(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);

  // 检查 m1 和 m2 的数据类型是否相同
  TORCH_CHECK(
    m1.dtype() == m2.dtype(),
    "expected m1 and m2 to have the same dtype, but got: ", m1.dtype(), " != ", m2.dtype()
  )
  // 通过数组访问速度更快的方式获取张量的大小和步长信息
  const auto self_sizes = self.sizes();
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  // 检查输入张量的形状是否与矩阵乘法兼容
  TORCH_CHECK(
      self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
      "input shape is incompatible with matrix multiplication (",
      m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
      self_sizes[0], "x", self_sizes[1], ")");

  // 调整输出张量的大小为 self 的大小
  at::native::resize_output(result, self_sizes);
  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  // 如果结果张量的元素个数为 0，直接返回
  if (result.numel() == 0) {
    return;
  }

  // 如果 m1 的第二维度为 0
  if (m1_sizes[1] == 0) {
    // 如果 beta 是复数 0，将结果张量置零
    if (beta.toComplexDouble() == 0.0) {
      result.zero_();
    } else {
      // 否则，如果结果张量与 self 不是同一个张量，将 self 复制到 result，然后乘以 beta
      if (!self.is_same(result)) {
        result.copy_(self);
      }
      result.mul_(beta);
    }
    return;
  }

  // 如果 beta 不是复数 0，且结果张量与 self 不是同一个张量
  if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }

  bool transpose_c = false;
  Tensor c;

  // Cast result as matrix a
  if (result_strides[0] == 1 &&
      (result_sizes[1] == 1 || result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
    transpose_c = false;
    c = result.resolve_conj();
  } else if (result_strides[1] == 1 &&
             (result_sizes[0] == 1 || result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);
    std::swap(m1_sizes, m2_sizes);
    std::swap(m1_strides, m2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    // make c FORTRAN contiguous
    c = result.resolve_conj().transpose(0, 1).contiguous().transpose_(0, 1);
  }

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = m1_sizes[transpose_c ? 0 : 1];

  // Cast m1 as matrix a
  bool transpose_a = false;
  Tensor a;
  /* Need lda >= max(1, (transpose_a ? k : m)) */
  if (m1_strides[transpose_c ? 1 : 0] == 1 &&
      m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = m1.resolve_conj();
  } else if (m1_strides[transpose_c ? 0 : 1] == 1 &&
             m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = m1;
  } else {
    transpose_a = !transpose_c;
    a = m1.clone(at::MemoryFormat::Contiguous);
  }

  // Cast m2 as matrix b
  bool transpose_b = false;
  Tensor b;
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (m2_strides[transpose_c ? 1 : 0] == 1 &&
      m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = m2.resolve_conj();
  } else if (m2_strides[transpose_c ? 0 : 1] == 1 &&
             m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = m2;
  } else {
    transpose_b = !transpose_c;
    b = m2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
  const int64_t ldc = c.strides()[transpose_c ? 0 : 1];

  // Always ensure the conjugation for c is resolved since there's no way to specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  bool dispatched = false;


注释：


    result.copy_(self);
  }  // 结束语句块，将 self 的值复制给 result

  bool transpose_c = false;
  Tensor c;

  // 将 result 强制转换为矩阵 a
  if (result_strides[0] == 1 &&
      (result_sizes[1] == 1 || result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
    transpose_c = false;
    c = result.resolve_conj();  // 解析并返回 result 的共轭
  } else if (result_strides[1] == 1 &&
             (result_sizes[0] == 1 || result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);  // 交换 m1 和 m2 的值
    std::swap(m1_sizes, m2_sizes);  // 交换 m1_sizes 和 m2_sizes 的值
    std::swap(m1_strides, m2_strides);  // 交换 m1_strides 和 m2_strides 的值
    transpose_c = true;
    c = result.resolve_conj();  // 解析并返回 result 的共轭
  } else {
    transpose_c = false;
    // 使 c 成为 FORTRAN 连续数组
    c = result.resolve_conj().transpose(0, 1).contiguous().transpose_(0, 1);
  }

  const int64_t m = result_sizes[transpose_c ? 1 : 0];  // 根据 transpose_c 决定 m 的值
  const int64_t n = result_sizes[transpose_c ? 0 : 1];  // 根据 transpose_c 决定 n 的值
  const int64_t k = m1_sizes[transpose_c ? 0 : 1];  // 根据 transpose_c 决定 k 的值

  // 将 m1 强制转换为矩阵 a
  bool transpose_a = false;
  Tensor a;
  /* 需要 lda >= max(1, (transpose_a ? k : m)) */
  if (m1_strides[transpose_c ? 1 : 0] == 1 &&
      m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = m1.resolve_conj();  // 解析并返回 m1 的共轭
  } else if (m1_strides[transpose_c ? 0 : 1] == 1 &&
             m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = m1;
  } else {
    transpose_a = !transpose_c;
    a = m1.clone(at::MemoryFormat::Contiguous);  // 克隆 m1 为连续内存格式的 Tensor
  }

  // 将 m2 强制转换为矩阵 b
  bool transpose_b = false;
  Tensor b;
  /* 需要 ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (m2_strides[transpose_c ? 1 : 0] == 1 &&
      m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = m2.resolve_conj();  // 解析并返回 m2 的共轭
  } else if (m2_strides[transpose_c ? 0 : 1] == 1 &&
             m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = m2;
  } else {
    transpose_b = !transpose_c;
    b = m2.clone(at::MemoryFormat::Contiguous);  // 克隆 m2 为连续内存格式的 Tensor
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];  // 根据 transpose_a 和 transpose_c 确定 lda
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];  // 根据 transpose_b 和 transpose_c 确定 ldb
  const int64_t ldc = c.strides()[transpose_c ? 0 : 1];  // 根据 transpose_c 确定 ldc

  // 确保在 gemm 调用中 c 的共轭已解析，因为无法在 gemm 调用中指定 c 的共轭
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  bool dispatched = false;
#if defined(__aarch64__) && AT_MKLDNN_ACL_ENABLED()
  // 如果在AArch64架构上且启用了MKL-DNN ACL（Arm Compute Library），并且要求C矩阵转置，
  // 则通过调用MKL-DNN矩阵乘法原语，使用RHS*LHS的顺序执行更快。这会调用Arm Compute Library (ACL)的GEMM核心，
  // 并且额外支持使用BF16指令运行核心。
  if (transpose_c) {
    // 应用启发式方法决定是否应用MKL-DNN矩阵乘法优化
    bool apply_heur = apply_mkldnn_matmul_heur(b.sizes()[0], b.sizes()[1], a.sizes()[1]);
    // 如果满足优化条件：A矩阵需要转置，B矩阵不需要转置，结果类型为单精度浮点数
    if (apply_heur && transpose_a && !transpose_b && result.scalar_type() == at::ScalarType::Float) {
      try {
        // 调用MKL-DNN矩阵乘法函数
        mkldnn_matmul(b, a, c, beta.to<float>(), alpha.to<float>());
        // 已经通过ACL GEMM调度了单精度浮点数的计算，因此无需再调度BLAS GEMM
        dispatched = true;
      } catch (const std::exception& e) {
        // MKL-DNN矩阵乘法失败时，切换到BLAS GEMM并输出警告信息
        TORCH_WARN("mkldnn_matmul failed, switching to BLAS gemm:", e.what());
        at::globalContext().setUserEnabledMkldnn(false);
      }
    }
  }
#endif

// 如果没有通过MKL-DNN或者MKL-DNN调度失败，则执行以下的BLAS例程
if(!dispatched) {
  // 应用BLAS例程进行矩阵乘法
  _AT_DISPATCH_ADDMM_TYPES(result.scalar_type(), "addmm_impl_cpu_", [&]{
        using opmath_t = at::opmath_type<scalar_t>;
        at::native::cpublas::gemm(
            // 根据A矩阵是否需要转置来选择转置类型
            transpose_a ? a.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
            // 根据B矩阵是否需要转置来选择转置类型
            transpose_b ? b.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
            m, n, k,
            alpha.to<opmath_t>(),
            // A矩阵的数据指针和leading dimension
            a.const_data_ptr<scalar_t>(), lda,
            // B矩阵的数据指针和leading dimension
            b.const_data_ptr<scalar_t>(), ldb,
            beta.to<opmath_t>(),
            // C矩阵的可变数据指针和leading dimension
            c.mutable_data_ptr<scalar_t>(), ldc);
      });
}

// 如果结果张量与目标结果不同，则将计算的结果复制到目标结果张量
if (!c.is_same(result)) {
  result.copy_(c);
}
}

// 实现针对多批次3D张量的bmm（batch matrix multiplication）
static void addbmm_impl_(
    Tensor &result, const Tensor &self, const Tensor &batch1, const Tensor &batch2, const Scalar& beta, const Scalar& alpha) {
  // 检查batch1必须是一个3D张量
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  // 检查batch2必须是一个3D张量
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  // 检查batch1和batch2的批次数必须相同
  TORCH_CHECK(batch1.size(0) == batch2.size(0),
      "batch1 and batch2 must have same number of batches, got ",
      batch1.size(0), " and ", batch2.size(0));
  // 检查bmm操作的矩阵尺寸兼容性
  TORCH_CHECK(batch1.size(2) == batch2.size(1),
      "Incompatible matrix sizes for bmm (",
      batch1.size(1), "x", batch1.size(2), " and ",
      batch2.size(1), "x", batch2.size(2), ")");

  // 获取矩阵乘法的输出尺寸
  const int64_t dim1 = batch1.size(1);
  const int64_t dim2 = batch2.size(2);
  // 检查self张量的形状必须与矩阵乘法输出一致
  TORCH_CHECK(self.size(0) == dim1 && self.size(1) == dim2,
      "self tensor does not match matmul output shape");

  // 重置输出结果张量的尺寸为与self张量相同
  result.resize_as_(self);

  // 如果beta不为零且结果张量与self不是同一个张量，则将self张量的内容复制到结果张量
  if (beta.to<c10::complex<double>>() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }

  // 获取批次数目
  const int64_t num_batches = batch1.size(0);

  // 如果没有批次，则根据beta的值对结果张量进行相应的处理
  if (num_batches == 0) {
    if (beta.to<c10::complex<double>>() != 0.0) {
      result.mul_(beta);
    } else {
      result.zero_();
    }
    return;
  }

  // 使用 beta 的当前值初始化 adjusted_beta
  auto adjusted_beta(beta);

  // 遍历 num_batches 次数的循环
  for (const auto batch : c10::irange(num_batches)) {
    // 将 batch1[batch] 和 batch2[batch] 的矩阵乘积添加到 result 中
    // 使用 adjusted_beta 和 alpha 进行调整
    result.addmm_(batch1[batch], batch2[batch], adjusted_beta, alpha);

    // 将 adjusted_beta 设置为 1，表示累积输出一次
    adjusted_beta = 1; // accumulate output once
  }
}

Tensor& addbmm_out(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  // 根据 self 张量的大小扩展 batch1 和 batch2，以匹配张量的最后两个维度，并生成新的张量 b_self
  auto b_self = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out");
  {
    // 禁用命名传播，确保在执行 addbmm_impl_ 操作时不影响张量的命名
    at::NoNamesGuard guard;
    // 调用底层的 addbmm_impl_ 函数，执行批量矩阵乘法，并将结果存储在 result 张量中
    addbmm_impl_(result, *b_self, batch1, batch2, beta, alpha);
  }
  // 根据 batch1, batch2, self 张量的命名信息，推广名称以供后续操作使用
  auto names = at::namedinference::propagate_names_for_addmm(batch1, batch2, self);
  // 如果推广的名称信息不为空，则在 result 张量上应用命名传播
  at::namedinference::propagate_names_if_nonempty(result, names);
  // 返回存储了结果的 result 张量
  return result;
}

Tensor &addbmm_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  // 调用 addbmm_out 函数，将结果存储在 self 张量中，并返回修改后的 self 张量
  return native::addbmm_out(self, batch1, batch2, beta, alpha, self);
}

Tensor addbmm(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  // 创建一个空张量 result，使用与 self 张量相同的选项
  Tensor result = at::empty({0}, self.options());
  // 调用 addbmm_out 函数，将结果存储在 result 张量中，并返回 result 张量
  return native::addbmm_out(self, batch1, batch2, beta, alpha, result);
}

TORCH_IMPL_FUNC(addmm_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor &result) {
  // 根据 self 张量的大小扩展 mat1 和 mat2，以匹配张量的第一个和第三个维度，并生成新的张量 b_self
  auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out");
  {
    // 禁用命名传播，确保在执行 addmm_impl_cpu_ 操作时不影响张量的命名
    at::NoNamesGuard guard;
    // 调用底层的 addmm_impl_cpu_ 函数，执行矩阵乘法，并将结果存储在 result 张量中
    addmm_impl_cpu_(const_cast<Tensor&>(result), *b_self, mat1, mat2, beta, alpha);
  }
}

TORCH_IMPL_FUNC(addmm_activation_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu, const Tensor &result) {
  // 根据 self 张量的大小扩展 mat1 和 mat2，以匹配张量的第一个和第三个维度，并生成新的张量 b_self
  auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out");
  {
    // 禁用命名传播，确保在执行 addmm_impl_cpu_ 操作时不影响张量的命名
    at::NoNamesGuard guard;
    // 调用底层的 addmm_impl_cpu_ 函数，执行矩阵乘法，并将结果存储在 result 张量中
    addmm_impl_cpu_(const_cast<Tensor&>(result), *b_self, mat1, mat2, beta, alpha);
    // 根据 use_gelu 的值选择性地应用 GELU 或 ReLU 激活函数到结果张量
    if (use_gelu) {
      at::gelu_(const_cast<Tensor&>(result));
    } else {
      at::relu_(const_cast<Tensor&>(result));
    }
  }
}

TORCH_IMPL_FUNC(mm_out_cpu)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  {
    // 禁用命名传播，确保在执行 addmm_impl_cpu_ 操作时不影响张量的命名
    at::NoNamesGuard guard;
    // 调用底层的 addmm_impl_cpu_ 函数，执行矩阵乘法，并将结果存储在 result 张量中
    addmm_impl_cpu_(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
  }
}

template <typename scalar_t, bool is_bmm>
// 定义一个内联函数，执行 CPU 上的 baddbmm 操作
inline void baddbmm_cpu_kernel(const Tensor& result, const Tensor& self, const Tensor& mat2, const Scalar& beta_, const Scalar& alpha_) {
  // 获取结果张量的尺寸信息
  int64_t bs = result.size(0);  // batch size
  int64_t is = result.size(1);  // 行数
  int64_t js = result.size(2);  // 列数
  int64_t ks = self.size(2);    // self 张量的第三维大小

  // 定义 opmath_t 类型，并从 alpha_ 和 beta_ 中获取数值转换为 opmath_t 类型
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t alpha = alpha_.to<opmath_t>();
  opmath_t beta = beta_.to<opmath_t>();

  // 分别获取 result、self 和 mat2 张量的访问器
  auto r0 = result.accessor<scalar_t, 3>();
  auto s0 = self.accessor<const scalar_t, 3>();
  auto m0 = mat2.accessor<const scalar_t, 3>();

  // 计算并设定任务粒度
  int64_t grain_size = std::max(internal::GRAIN_SIZE / (is * js * ks), (int64_t)1);
  // 使用并行方式执行循环，对每个 batch 中的数据进行计算
  using opmath_t = at::opmath_type<scalar_t>;
  parallel_for(0, bs, grain_size, [&](int64_t b_begin, int64_t b_end) {
      for (const auto b : c10::irange(b_begin, b_end)) {
        auto r1 = r0[b];
        auto s1 = s0[b];
        auto m1 = m0[b];
        // 遍历每个 batch 中的行和列
        for (const auto i : c10::irange(is)) {
          auto r2 = r1[i];
          auto s2 = s1[i];
          // 遍历每个 batch 中的列
          for (const auto j : c10::irange(js)) {
            opmath_t acc_value = 0;  // 初始化累加器为 0
            // 遍历 self 张量的第三维
            for (const auto k : c10::irange(ks)) {
              // 计算 acc_value，累加 self 和 mat2 张量的乘积
              acc_value += static_cast<opmath_t>(s2[k]) *
                  static_cast<opmath_t>(m1[k][j]);
            }
            // 根据 is_bmm 变量选择如何更新 r2[j] 的值
            if (is_bmm) {
              r2[j] = acc_value;  // 如果 is_bmm 为真，则直接赋值 acc_value
            } else {
              // 当 beta == 0 时，r2[j] 的值将被忽略，尤其是对于 NaN 值
              if (beta == opmath_t{0}) {
                r2[j] = alpha * acc_value;  // 如果 beta 为 0，则只使用 alpha 和 acc_value 计算新值
              } else {
                // 否则，使用 beta 和 alpha 加权更新 r2[j] 的值
                r2[j] = static_cast<opmath_t>(r2[j]) * beta + alpha * acc_value;
              }
            }
          }
        }
      }
    });
}

// 定义静态函数 baddbmm_with_gemm_，使用 GEMM 算法执行 baddbmm 操作
static void baddbmm_with_gemm_(const Tensor &result, const Tensor &mat1, const Tensor &mat2, const Scalar &beta_, const Scalar &alpha_) {
  // 断言结果张量是连续的
  TORCH_INTERNAL_ASSERT(result.is_contiguous());

  // 获取各张量的尺寸和步长信息
  const auto result_sizes = result.sizes();
  const auto result_strides = result.strides();
  const auto mat1_strides = mat1.strides();
  const auto mat2_strides = mat2.strides();
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  // 定义一个 lambda 函数，用于判断张量是否转置
  auto is_transposed = [](const c10::IntArrayRef& strides, const c10::IntArrayRef& sizes) {
    return strides[1] == 1 && strides[2] >= sizes[1];
  };

  // gemm 函数期望 Fortran 顺序的矩阵，因此需要交换参数顺序来实现转置
  const auto transpose_a = is_transposed(mat2_strides, mat2_sizes);
  const auto transpose_b = is_transposed(mat1_strides, mat1_sizes);

  // 获取矩阵的尺寸信息
  const int64_t batch_size = mat1_sizes[0];
  const int64_t m = result_sizes[2];
  const int64_t n = result_sizes[1];
  const int64_t k = mat2_sizes[1];

  // 获取矩阵的步长信息
  const int64_t lda = mat2_strides[transpose_a ? 2 : 1];
  const int64_t ldb = mat1_strides[transpose_b ? 2 : 1];
  const int64_t ldc = result_strides[1];

  // 使用宏处理浮点数和复数类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "baddbmm_with_gemm", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    # 将alpha_转换为opmath_t类型，并赋值给alpha常量
    const auto alpha = alpha_.to<opmath_t>();
    # 将beta_转换为opmath_t类型，并赋值给beta常量
    const auto beta = beta_.to<opmath_t>();
    # 调用ATen库中的cpublas模块，执行批量矩阵乘法，带有步长信息
    at::native::cpublas::gemm_batched_with_stride(
        # 根据transpose_a的布尔值决定是否转置第一个矩阵
        transpose_a ? TransposeType::Transpose : TransposeType::NoTranspose,
        # 根据transpose_b的布尔值决定是否转置第二个矩阵
        transpose_b ? TransposeType::Transpose : TransposeType::NoTranspose,
        # 批量大小
        batch_size,
        # 第一个矩阵的行数（或转置后的列数）
        m,
        # 第二个矩阵的列数（或转置后的行数）
        n,
        # 公共维度
        k,
        # 系数alpha，用于乘以矩阵乘积的第一个输入矩阵
        alpha,
        # 第二个矩阵的数据指针，以scalar_t类型解释
        mat2.const_data_ptr<scalar_t>(),
        # 第二个矩阵的列偏移
        lda,
        # mat2的步长信息
        mat2_strides[0],
        # 第一个矩阵的数据指针，以scalar_t类型解释
        mat1.const_data_ptr<scalar_t>(),
        # 第一个矩阵的列偏移
        ldb,
        # mat1的步长信息
        mat1_strides[0],
        # 系数beta，用于乘以结果矩阵
        beta,
        # 结果矩阵的数据指针，以scalar_t类型解释
        result.data_ptr<scalar_t>(),
        # 结果矩阵的列偏移
        ldc,
        # result的步长信息
        result_strides[0]
    );
}

// 尝试对 bmm/baddbmm 应用一些优化策略：
// - 当操作数尺寸较小时，使用OMP在批处理维度上并行化计算，并采用简单的矩阵乘法。
// - 当操作数尺寸超过阈值时，并且编译时使用了MKL，使用MKL的批量矩阵乘法（batch gemm）。
// - 否则，使用一系列的矩阵乘法操作。
// 第一个阈值为400的优化还未经过充分基准测试，可能还有进一步优化的空间，这可能取决于CPU的特性、MKL与非MKL的差异等，
// 但这似乎是一个起点。

static inline void bmm_out_or_baddbmm_(const Tensor& self_or_result_, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm_out) {
  // is_bmm_out: 如果是 bmm_out 则为 true，如果是 baddbmm_ 则为 false
  // self_or_result 是 baddbmm_ 中的 "self"，是 bmm_out 中的 "result"
  Tensor& self_or_result = const_cast<Tensor&>(self_or_result_);

  // 获取 batch1 和 batch2 的尺寸
  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  // 获取批处理维度（batch size）、收缩维度（contraction size）、结果矩阵的行数和列数
  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];

  // 处理一些特殊情况，这些情况下 BLAS 可能无法正常工作
  if (self_or_result.numel() == 0) {
    return;  // 结果张量为空，直接返回
  } else if (contraction_size == 0) {
    if (is_bmm_out || (beta.to<c10::complex<double>>() == 0.0)) {
      self_or_result.zero_();  // 如果是 bmm_out 或者 beta 是零，则将结果张量置零
      return;
    } else {
      self_or_result.mul_(beta);  // 否则，将结果张量乘以 beta
      return;
    }
  }

  // 检查是否应用启发式策略
  auto batch_items_contiguous_or_transposed = [&](const Tensor& t) {
    const auto sizes = t.sizes();
    const auto strides = t.strides();
    // 如果维度的步长为1且大小为1，或者步长大于等于维度的大小，则认为是连续或者转置的
    return (strides[2] == 1 && (sizes[1] == 1 || strides[1] >= sizes[2])) ||
        (strides[1] == 1 && (sizes[2] == 1 || strides[2] >= sizes[1]));
  };

  // 如果应用了 MKL 并且可以使用 MKL 的矩阵乘法
  bool apply_heur = apply_mkldnn_matmul_heur(batch1.sizes()[1], batch1.sizes()[2], batch2.sizes()[2]);
  if (apply_heur && use_mkldnn_matmul(batch1, batch2, self_or_result)) {
    try {
      mkldnn_matmul(batch1, batch2, self_or_result, beta.to<float>(), alpha.to<float>());
      return;  // 使用 MKL 执行成功，直接返回
    } catch (const std::exception& e) {
      TORCH_WARN("mkldnn_matmul failed, switching to baddbmm:", e.what());
      at::globalContext().setUserEnabledMkldnn(false);  // MKL 执行失败，禁用 MKL
    }
  }

  // 如果收缩维度乘以结果矩阵的行列数小于400，则使用 baddbmm 或者 bmm 函数
  if (contraction_size * res_rows * res_cols < 400) {
    if (is_bmm_out) {
      // 根据数据类型调度执行 bmm 函数的 CPU 内核
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, batch1.scalar_type(), "bmm", [&] {
          baddbmm_cpu_kernel<scalar_t, true>(self_or_result, batch1, batch2, beta, alpha);
        });
    } else {
      // 根据数据类型调度执行 baddbmm 函数的 CPU 内核
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, batch1.scalar_type(), "baddbmm", [&] {
          baddbmm_cpu_kernel<scalar_t, false>(self_or_result, batch1, batch2, beta, alpha);
        });
    }
  } else if (at::hasMKL() && ((
            self_or_result.scalar_type() != kBFloat16 &&
            self_or_result.scalar_type() != kHalf &&
            at::native::is_floating_point(self_or_result)) ||
            at::native::is_complex(self_or_result))
            && batch_items_contiguous_or_transposed(batch1)
            && batch_items_contiguous_or_transposed(batch2)
            && self_or_result.is_contiguous()) {
    # 如果当前环境支持 MKL 并且满足以下条件：
    #   - self_or_result 的数据类型不是 kBFloat16 和 kHalf，并且是浮点数类型
    #   - 或者 self_or_result 是复数类型
    #   - batch1 和 batch2 中的批处理项目是连续或转置的
    #   - self_or_result 是连续的
    # 则调用 baddbmm_with_gemm_ 函数进行矩阵乘法运算
    baddbmm_with_gemm_(self_or_result, batch1, batch2, beta, alpha);
  } else { // split along batch dimension
#ifdef C10_MOBILE
    /*
     * 当推断模式启用时才使用多线程，因为各种线程本地状态在 at::parallel_for 中
     * 无法适当地传播。例如 RecordFunction 相关的状态，dispatchKeySet。
     * 主要问题在于，如果我们在状态未正确传播的情况下使用 at::parallel_for，
     * 分发机制在主线程和其他线程上可能表现不同，导致未定义的行为。
     * 因此，建议不要在需要通过分发器进行操作的 lambda 函数中使用 at::parallel_for。
     * 目前我们通过推断模式保护来解决这个问题，以提升性能。
     * 长期来看，可能需要一个专门的 API 来显式调用传播的 TLS（线程本地存储）。
     * 还需注意，此功能仅在移动平台启用，因为非移动版本的 BLAS 实现已经支持多线程。
     */
    // 进行了以下基准测试：
    // bmm_test：在 benchmarks/operator_benchmarks/pt/bmm_test.py 下的操作基准测试
    // 在 Samsung S8U 上针对不同矩阵大小运行了这些基准测试
    const bool enable_multithreaded_bmm = c10::InferenceMode::is_enabled() &&
        bs >= 4 && res_rows >= 4 && res_cols >= 16 && contraction_size >= 16;
#else
    const bool enable_multithreaded_bmm{false};
#endif

if (is_bmm_out) {
    if (enable_multithreaded_bmm) {
        // 多线程执行 BMM 计算
        auto bmm_out_fn = [&](uint64_t start, uint64_t end) {
            c10::InferenceMode guard;
            for (const auto b : c10::irange(start, end)) {
                auto r = self_or_result.select(0, b);
                // 调用 CPU 上的 addmm 实现
                addmm_impl_cpu_(
                    r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
            }
        };
        // 如果是 COW，需要实现数据的共享，因为在 parallel_for 过程中无法这样做
        self_or_result.mutable_data_ptr();
        at::parallel_for(0, bs, 1, bmm_out_fn);
    } else {
        // 单线程执行 BMM 计算
        for (const auto b : c10::irange(bs)) {
            auto r = self_or_result.select(0, b);
            // 调用 CPU 上的 addmm 实现
            addmm_impl_cpu_(r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
        }
    }
} else {
    if (enable_multithreaded_bmm) {
        // 多线程执行 BMM 计算
        auto bmm_fn = [&](uint64_t start, uint64_t end) {
            c10::InferenceMode guard;
            for (const auto b : c10::irange(start, end)) {
                self_or_result.select(0, b).addmm_(
                    batch1.select(0, b), batch2.select(0, b), beta, alpha);
            }
        };
        // 如果是 COW，需要实现数据的共享，因为在 parallel_for 过程中无法这样做
        self_or_result.mutable_data_ptr();
        at::parallel_for(0, bs, 1, bmm_fn);
    } else {
        // 单线程执行 BMM 计算
        for (const auto b : c10::irange(bs)) {
            self_or_result.select(0, b).addmm_(
                batch1.select(0, b), batch2.select(0, b), beta, alpha);
        }
    }
}



static void conjugate_mutable_input_if_needed(const Tensor& self, bool conjugate) {
    // 如果需要共轭变换输入张量
    if (conjugate) {
        // 调用张量的共轭物理操作
        self.conj_physical_();
    }
}
# 定义 Torch 实现函数 baddbmm_out_cpu，用于计算批次乘积加和
(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
    # 检查结果张量是否共轭
    bool self_is_conj = result.is_conj();
    # 如果需要，根据共轭性对结果张量进行变换
    conjugate_mutable_input_if_needed(result, self_is_conj);
    # 调用 bmm_out_or_baddbmm_ 函数计算批次乘积加和，不执行批量矩阵乘法
    bmm_out_or_baddbmm_(result, batch1.resolve_conj(), batch2.resolve_conj(), beta, alpha, false);
    # 恢复结果张量的共轭状态，如果在变换前是共轭的话
    conjugate_mutable_input_if_needed(result, self_is_conj);
}

# 定义 Torch 实现函数 bmm_out_cpu，用于计算批次矩阵乘法的结果
(const Tensor & batch1, const Tensor & batch2, const Tensor & result) {
    {
    # 不使用名称的保护装置
    NoNamesGuard guard;
    # 检查结果张量是否共轭
    bool result_is_conj = result.is_conj();
    # 如果需要，根据共轭性对结果张量进行变换
    conjugate_mutable_input_if_needed(result, result_is_conj);
    # 调用 bmm_out_or_baddbmm_ 函数计算批次矩阵乘法的结果
    bmm_out_or_baddbmm_(result, batch1.resolve_conj(), batch2.resolve_conj(), Scalar(0.0), Scalar(1.0), true);
    # 恢复结果张量的共轭状态，如果在变换前是共轭的话
    conjugate_mutable_input_if_needed(result, result_is_conj);
    }
}

# 定义 Torch 实现函数 dot_out，用于计算两个张量的点积并存储在结果张量中
Tensor& dot_out(const Tensor& self, const Tensor& other, Tensor& result) {
  # 获取输出、输入1和输入2张量的设备信息
  auto output_device = result.device();
  auto input1_device = self.device();
  auto input2_device = other.device();
  # 检查输入和输出张量是否在相同的设备上
  TORCH_CHECK(
    (output_device == input1_device) && (input1_device == input2_device),
    "dot: Expected the output and input tensors to be on the "
    "same device, but got the output tensor on ", output_device,
    ", the 'input' tensor on ", input1_device, ", and the 'other' tensor on ", input2_device);
  # 调整输出张量的大小，确保它与期望的点积结果兼容
  at::native::resize_output(result, {});
  # 检查结果张量的数据类型是否与输入张量的数据类型匹配
  TORCH_CHECK(result.scalar_type() == self.scalar_type(),
           "result dtype ", result.scalar_type(), " does not match input dtype ", self.scalar_type());
  # 使用点积计算填充结果张量
  return result.fill_(self.dot(other));
}

# 定义 Torch 实现函数 vdot_out，用于计算两个张量的共轭点积并存储在结果张量中
Tensor& vdot_out(const Tensor& self, const Tensor& other, Tensor& result) {
  # 获取输出、输入1和输入2张量的设备信息
  auto output_device = result.device();
  auto input1_device = self.device();
  auto input2_device = other.device();
  # 检查输入和输出张量是否在相同的设备上
  TORCH_CHECK(
    (output_device == input1_device) && (input1_device == input2_device),
    "vdot: Expected the output and input tensors to be on the "
    "same device, but got the output tensor on ", output_device,
    ", the 'input' tensor on ", input1_device, ", and the 'other' tensor on ", input2_device);
  # 调整输出张量的大小，确保它与期望的共轭点积结果兼容
  at::native::resize_output(result, {});
  # 检查结果张量的数据类型是否与输入张量的数据类型匹配
  TORCH_CHECK(result.scalar_type() == self.scalar_type(),
           "result dtype ", result.scalar_type(), " does not match input dtype ", self.scalar_type());
  # 使用共轭点积计算填充结果张量
  return result.fill_(self.vdot(other));
}
// 检查是否应该将两个张量折叠成矩阵乘积而不是批量矩阵乘法
static bool should_fold(const Tensor& tensor1, const Tensor& tensor2, bool has_out) {
  // 检查我们是否可以将较大的张量折叠成矩阵，并分派到 mm 或 mv，而不是 bmm。我们希望确保在不额外复制的情况下完成
  const auto tensor1_larger = tensor1.dim() >= tensor2.dim();

  // 对张量进行排序。如果 tensor1_larger 为 true，则 t1 将是较大的张量；否则使用 tensor2 的转置作为 t1
  const auto t1 = tensor1_larger ? MaybeOwned<Tensor>::borrowed(tensor1)
                                 : MaybeOwned<Tensor>::owned(tensor2.mT());
  const int64_t dim_t1 = t1->dim();
  const auto dim_t2 = tensor1_larger ? tensor2.dim()
                                     : tensor1.dim();

  // 只有当 dim_t1 >= 3 且 dim_t2 <= 2 时才折叠
  if (!(dim_t1 >= 3 && dim_t2 <= 2)) {
    return false;
  }

  // 在这种情况下，为了避免在反向传播中创建不必要的大张量，我们确实会产生额外的复制
  bool t2_requires_grad = tensor1_larger ? tensor2.requires_grad() : tensor1.requires_grad();
  if (t2_requires_grad && !has_out) {
    // 我们本应检查 !at::GradMode::is_enabled()，但显然在某些情况下这会降低性能：
    // https://github.com/pytorch/pytorch/issues/118548#issuecomment-1916022394
    return true;
  }

  // 如果张量维度为 2，则不折叠
  if (tensor1.dim() == 2) {
    return false;
  }

  // 如果 t1 为空，则可以始终折叠
  if (t1->numel() == 0) {
    return true;
  }

  // 检查 t1->view(-1, t1->size(-1)) 是否不复制，只有当前 n-1 维度是连续的时候
  const auto t1_shape = t1->sizes();
  const auto t1_strides = t1->strides();
  for (auto i = int64_t{0}; i < dim_t1 - int64_t{2}; ++i) {
    if (t1_strides[i] != t1_strides[i+1] * t1_shape[i+1]) {
      return false;
    }
  }
  return true;
}
// 定义 _matmul_impl 函数，实现张量之间的矩阵乘法或向量点积操作，将结果存入 out 中
static Tensor _matmul_impl(
    Tensor& out,                      // 输出张量，存储计算结果
    const Tensor& tensor1,            // 第一个输入张量
    const Tensor& tensor2) {          // 第二个输入张量
  NoNamesGuard guard;                 // 临时对象，用于保护不使用命名张量

  // 获取输入张量的维度信息
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();

  // 检查输入张量的维度是否至少为1
  TORCH_CHECK(dim_tensor1 != 0 && dim_tensor2 != 0,
              "both arguments to matmul need to be at least 1D, but they are ",
              dim_tensor1, "D and ", dim_tensor2, "D");

  const bool has_out = out.defined();  // 检查输出张量是否已定义

  if (has_out) {
    // 检查是否有自动求导需要，并确保对于 matmul 函数，不支持自动求导
    TORCH_CHECK(!(tensor1.requires_grad() || tensor2.requires_grad() || out.requires_grad()) || !at::GradMode::is_enabled(),
      "matmul(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad."
    );
  }

  // 根据输入张量的维度执行不同的矩阵乘法或向量点积操作
  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    // 两个输入张量都是1维向量，进行点积运算
    return has_out ? at::dot_out(out, tensor1, tensor2) : tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    // tensor1 是2维矩阵，tensor2 是1维向量，执行矩阵和向量的乘法
    return has_out ? at::mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    // tensor1 是1维向量，tensor2 是2维矩阵，执行向量和矩阵的乘法
    return has_out ? at::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                   : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    // 两个输入张量都是2维矩阵，执行矩阵乘法
    return has_out ? at::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
  } else if (should_fold(tensor1, tensor2, has_out)) {
    // 根据 should_fold 函数判断是否需要折叠张量的维度进行优化操作

    // 通过折叠大张量的批处理维度来优化，使用矩阵乘法替代批处理矩阵乘法
    const auto transpose = dim_tensor2 > dim_tensor1;
    const auto t1 = transpose ? MaybeOwned<Tensor>::owned(tensor2.mT())
                              : MaybeOwned<Tensor>::borrowed(tensor1);
    // 根据 transpose 变量选择合适的 tensor1 或 tensor2 的视图，确保 t1 引用正确的 Tensor

    const auto t2 = !transpose ? MaybeOwned<Tensor>::borrowed(tensor2)
                               : dim_tensor1 == 2
                                   ? MaybeOwned<Tensor>::owned(tensor1.t())
                                   : MaybeOwned<Tensor>::borrowed(tensor1);
    // 根据 transpose 变量和 tensor1 的维度选择合适的 tensor2 的视图，确保 t2 引用正确的 Tensor

    // 不变条件：t1 的维度至少为 3，而 t2 的维度为 1 或 2
    // 并且 *t1 和 *t2 可以进行矩阵乘法运算

    // 为什么不使用 t1->view(-1, sizes_1.back())？
    // 如果最后一个维度为 0，使用 view(-1, 0) 将会导致模糊性，因此需要特殊处理
    // 这种情况可能发生在例如 [3, 5, 0] @ [0, 0] 的情况下
    const auto sizes_1 = t1->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);

    // 如果我们要与矩阵相乘，则重新调整 output_shape
    const auto t2_is_matrix = t2->dim() == 2;
    if (t2_is_matrix) {
      output_shape.push_back(t2->sizes()[1]);
    }
    // 这几乎总是一个视图操作
    // 如果 t2->requires_grad() 为 true，可能不是一个视图，请参见 should_fold 的解释
    const auto t1_folded = t1->reshape({folded_dim1, sizes_1.back()});

    if (!has_out) {
      if (t2_is_matrix) {
        const auto output = at::_unsafe_view(t1_folded.mm(*t2), output_shape);
        // 如果我们执行 2D @ 3D 运算且第一个张量需要梯度，这将进行复制
        // 详见 should_fold 的解释
        // 如果 mm_out 是可微的，我们可以在这里使用它，并传递具有正确步幅的结果，以避免这种不必要的复制
        return transpose ? output.mT().contiguous() : output;
      } else {
        return at::_unsafe_view(t1_folded.mv(*t2), output_shape);
      }
    } else {
      // 见 !has_out 分支的解释
      TORCH_INTERNAL_ASSERT(!(transpose && t2_is_matrix));

      // 调整输出到正确的形状
      at::native::resize_output(out, output_shape);

      // 然后将输出重塑为预期形状，并调用 mm/mv
      // 如果必要，进行转置
      auto reshaped_out = t2_is_matrix ? out.reshape({folded_dim1, t2->sizes().back()})
                                       : out.reshape({folded_dim1});
      if (t2_is_matrix) {
        at::mm_out(reshaped_out, t1_folded, *t2);
      } else {
        at::mv_out(reshaped_out, t1_folded, *t2);
      }
      if (!reshaped_out.is_alias_of(out)) {
        out.copy_(reshaped_out);
      }
      return out;
    }
  } else {
    // dim_tensor1 >= 3 || dim_tensor2 >= 3
    // 尽管 m1 和 m2 必须匹配以获得更好的错误消息，但我们分别跟踪 m1 和 m2
    const int64_t n = dim_tensor1 > 1 ? tensor1.sizes().cend()[-2] : 1LL;
    const int64_t m1 = tensor1.sizes().back();
    auto batch_tensor1 = tensor1.sizes().slice(0, std::max<int64_t>(dim_tensor1 - 2, 0LL));
    const int64_t m2 = dim_tensor2 > 1 ? tensor2.sizes().cend()[-2] : tensor2.sizes().front();
    const int64_t p = dim_tensor2 > 1 ? tensor2.sizes().back() : 1LL;
    const IntArrayRef batch_tensor2(tensor2.sizes().data(),
                                    std::max<int64_t>(dim_tensor2 - 2, 0LL));

    // 对梯度和 should_fold 函数采用相同的优化
    // 如果我们要进行广播，强制通过 should_fold 分支处理
    if (dim_tensor1 == 3 && dim_tensor2 == 3 && batch_tensor1[0] != batch_tensor2[0]) {
      // 如果 batch_tensor1 的第一个元素为 1，并且 tensor1 需要梯度或者类似于 tensor 子类
      if (batch_tensor1[0] == 1 && (tensor1.requires_grad() || isTensorSubclassLike(tensor1))) {
        // 返回通过挤压第一个维度后的 tensor1 与 tensor2 的矩阵乘积的实现结果
        return _matmul_impl(out, tensor1.squeeze(0), tensor2);
      }
      // 如果 batch_tensor2 的第一个元素为 1，并且 tensor2 需要梯度或者类似于 tensor 子类
      if (batch_tensor2[0] == 1 && (tensor2.requires_grad() || isTensorSubclassLike(tensor2))) {
        // 返回 tensor1 与通过挤压第一个维度后的 tensor2 的矩阵乘积的实现结果
        return _matmul_impl(out, tensor1, tensor2.squeeze(0));
      }
    }

    auto output_shape = infer_size_dimvector(batch_tensor1, batch_tensor2);
    const int64_t expand_batch_product = c10::multiply_integers(output_shape);

    // 展开批次维度
    const auto tensor1_expand_size = [&output_shape, n, m1]{ DimVector ret(output_shape);
                                                             ret.append({n, m1});
                                                             return ret; }();
    const auto tensor1_expanded = tensor1.expand(tensor1_expand_size)
                                         .reshape({expand_batch_product, n, m1});

    // 需要单独处理 dim_tensor2 == 1 的情况，因为广播不会将形状为 (n,) 的向量转换为形状为 (*, n, 1) 的批量矩阵
    auto vector_rhs = dim_tensor2 == 1;
    const auto tensor2_expand_size = [&output_shape, m2, p, vector_rhs]{
      DimVector ret(output_shape);
      if (vector_rhs) {
        ret.push_back(m2);
      } else {
        ret.append({m2, p});
      }
      return ret;
    }();
    auto tensor2_expanded = tensor2.expand(tensor2_expand_size);
    if (vector_rhs) {
      tensor2_expanded = tensor2_expanded.reshape({expand_batch_product, m2}).unsqueeze(2);
    } else {
      tensor2_expanded = tensor2_expanded.reshape({expand_batch_product, m2, p});
    }

    // 如果 dim_tensor1 大于 1，则在 output_shape 中添加 n
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    // 如果 dim_tensor2 大于 1，则在 output_shape 中添加 p
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    // 如果没有提供输出张量 out
    if (!has_out) {
      // 如果 tensor2 是右手边的向量
      if (vector_rhs) {
        // 返回非安全视图(tensor1_expanded 与 tensor2_expanded 的批量矩阵乘积).挤压(-1)，形状为 output_shape
        return at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded).squeeze(-1), output_shape);
      } else {
        // 返回非安全视图(tensor1_expanded 与 tensor2_expanded 的批量矩阵乘积)，形状为 output_shape
        return at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded), output_shape);
      }
    } else {
      // 调整输出 out 的大小为 output_shape
      at::native::resize_output(out, output_shape);
      auto reshaped_out = out.reshape({expand_batch_product, n, p});
      // 在 reshaped_out 上执行批量矩阵乘积，tensor1_expanded 与 tensor2_expanded
      at::bmm_out(reshaped_out, tensor1_expanded, tensor2_expanded);
      // 如果 tensor2 是右手边的向量，则挤压 -1 维度
      if (vector_rhs) {
        reshaped_out = reshaped_out.squeeze(-1);
      }
      // 如果 reshaped_out 不是 out 的别名，则将其拷贝为 out
      if (!reshaped_out.is_alias_of(out)) {
        out.copy_(reshaped_out.view_as(out));
      }
      // 返回 out
      return out;
    }
  }
}

// 函数：执行张量的矩阵乘法，返回结果张量
Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  // 计算可能的输出名称
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  // 定义结果张量和未使用的占位符
  at::Tensor result, unused;
  // 调用底层的矩阵乘法实现
  result = at::native::_matmul_impl(unused, tensor1, tensor2);
  // 如果可能的输出名称非空，则传播名称
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  // 返回计算得到的结果张量
  return result;
}

// 函数：执行张量的矩阵乘法，并将结果写入预先提供的结果张量中，返回结果张量的引用
Tensor& matmul_out(const Tensor & tensor1, const Tensor & tensor2, Tensor &result) {
  // 计算可能的输出名称
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  // 调用底层的矩阵乘法实现，将结果写入提供的结果张量中
  at::native::_matmul_impl(result, tensor1, tensor2);
  // 如果可能的输出名称非空，则传播名称
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  // 返回结果张量的引用
  return result;
}

// 函数：torch.linalg.matmul 的别名，实际调用 torch.matmul
Tensor linalg_matmul(const Tensor & tensor1, const Tensor & tensor2) {
  return at::matmul(tensor1, tensor2);
}

// 函数：执行张量的矩阵乘法，并将结果写入预先提供的结果张量中，返回结果张量的引用
Tensor& linalg_matmul_out(const Tensor & tensor1, const Tensor & tensor2, Tensor &result) {
  return at::matmul_out(result, tensor1, tensor2);
}

// 函数：torch.linalg.diagonal 的别名，使用默认参数 dim1=-2, dim2=-1 调用 torch.diagonal
Tensor linalg_diagonal(const Tensor& A, int64_t offset, int64_t dim1, int64_t dim2) {
  return A.diagonal(offset, dim1, dim2);
}

// 帮助方法：用于 matrix_exp 的矩阵指数计算
namespace {

template <typename scalar_t, int ROW, int COL>
using array2d = std::array<std::array<scalar_t, COL>, ROW>;

// 我们考虑 6 个泰勒展开的阶数
// 1, 2, 4, 8, 12, 18
constexpr int total_n_degs = 6;

// 函数：计算张量的 1 范数
Tensor operator_1_norm(const Tensor& tensor) {
  return std::get<0>(tensor.abs().sum(-2).max(-1));
}

// 函数：分配一个未初始化或零值的缓冲区，形状为 [n_copies, a.size()]
Tensor _allocate_buffer(const Tensor& a, int n_copies, bool is_zero = false) {
  auto res = at::empty(
    {n_copies, a.size(0), a.size(1), a.size(2)},
    a.options().memory_format(at::MemoryFormat::Contiguous)
  );

  if (is_zero) {
    res.zero_();
  }

  return res;
}

// 函数：填充用于计算不同阶数矩阵指数的矩阵缓冲区
void _fill_matrix_powers(Tensor& buffer, const Tensor& a, int num_matrices) {
  auto a_sizes_minus_last = a.sizes().vec();
  a_sizes_minus_last.pop_back();
  // 填充单位矩阵 I
  buffer.select(0, 0).copy_(
    at::diag_embed(
      at::ones({1}, buffer.options())
        .expand(a_sizes_minus_last)
    )
  );

  // 填充矩阵 a
  buffer.select(0, 1).copy_(a);

  // 填充矩阵 a^2
  if (2 <= num_matrices - 1) {
    auto view_out = buffer.select(0, 2);
    at::matmul_out(view_out, buffer.select(0, 1), buffer.select(0, 1));
  }

  // 填充矩阵 a^3
  if (3 <= num_matrices - 1) {
    auto view_out = buffer.select(0, 3);
    at::matmul_out(view_out, buffer.select(0, 1), buffer.select(0, 2));
  }


The code continues beyond this point but the provided snippet is properly commented according to the guidelines.
    // 调用矩阵乘法实现函数 `_matmul_impl`，计算并存储结果到 `view_out` 中
    _matmul_impl(
      view_out,
      buffer.select(0, 1),   // 选择缓冲区中第一个维度为 1 的张量，作为乘法的第一个操作数
      buffer.select(0, 2)    // 选择缓冲区中第一个维度为 2 的张量，作为乘法的第二个操作数
    );
  }

  // 计算并填充矩阵的六次幂 a^6
  if (4 <= num_matrices - 1) {  // 如果可以计算 a^6（至少有 5 个矩阵）
    // 获取用于存储 a^6 结果的视图 view_out
    auto view_out = buffer.select(0, 4);
    // 执行矩阵乘法 `_matmul_impl`，将 a^3 与自身相乘，结果存储在 view_out 中
    _matmul_impl(
      view_out,
      buffer.select(0, 3),   // 选择缓冲区中第一个维度为 3 的张量，作为乘法的第一个操作数
      buffer.select(0, 3)    // 再次选择缓冲区中第一个维度为 3 的张量，作为乘法的第二个操作数
    );
  }
// 返回一个内存张量，如果输入张量在 CUDA 设备上，则移动内存到与输入张量相同的设备上
inline Tensor _move_memory_if_cuda_input(
  const Tensor& mem,
  const Tensor& in
) {
  return (in.device().type() == at::kCUDA)
    ? mem.to(at::device_of(in).value())
    : mem;
}

// 将一个一维 blob 转换为大小为 [1, blob.size()] 的二维张量
// 要求 blob.device() == in.device()
// 设计用于与 _compute_linear_combination 一起使用
template <typename scalar_t>
inline Tensor _blob_to_Tensor(
  std::initializer_list<scalar_t> blob,
  const Tensor& in
) {
  // 明确将 blob 转换为 void*，因为 begin() 返回一个常量指针。
  // 假设 Blob 是一个一维数组，因此我们还插入一个虚假的维度，使结果可以直接用于 _compute_linear_combination
  auto tensor = at::from_blob((void*)blob.begin(), blob.size(),
    c10::toRealValueType(in.scalar_type())).unsqueeze(0);
  return _move_memory_if_cuda_input(tensor, in);
}

// 执行线性组合操作
template <typename scalar_t>
inline Tensor _linear_combination(
    const Tensor& t,
    std::initializer_list<scalar_t> blob) {
  // _blob_to_Tensor 将 blob 转换为一个二维张量，以便用于 _compute_linear_combination。
  // 如果这个张量的形状是 (1, *)，_compute_linear_combination 的结果将是形状 (1, *t.shape)，
  // 所以我们压缩维度(0)，以便对于任何 t，满足 t.dim() >= 1：t.dim() == _compute_linear_combination(t, ...).dim()。
  return at::native::_compute_linear_combination(
      t, _blob_to_Tensor<scalar_t>(blob, t))
    .squeeze(0);
}

// 计算 T1 = I + A，其中 I 是单位矩阵
Tensor compute_T1(const Tensor& A) {
  // 为 {I, A} 分配缓冲区
  auto As = _allocate_buffer(A, 2);
  // 填充 As 中的矩阵幂，直到幂次为 2
  _fill_matrix_powers(As, A, 2);
  return As.sum(0); // 返回所有矩阵的总和
}

// 计算 T2 = I + A + A^2 / 2，其中 I 是单位矩阵
Tensor compute_T2(const Tensor& A) {
  auto As = _allocate_buffer(A, 3); // 为 {I, A, A^2} 分配缓冲区
  _fill_matrix_powers(As, A, 3); // 填充 As 中的矩阵幂，直到幂次为 3
  As.select(0, 2).div_(2.0); // 对 A^2 进行除以 2 的操作
  return As.sum(0); // 返回所有矩阵的总和
}

// 计算 T4 = I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)，其中 I 是单位矩阵
template <typename scalar_t>
Tensor compute_T4(const Tensor& A) {
  auto As = _allocate_buffer(A, 4); // 为 {I, A, A^2, A^3} 分配缓冲区
  _fill_matrix_powers(As, A, 3); // 填充 As 中的矩阵幂，直到幂次为 3

  auto view_out = As.select(0, 3); // 获取 A^3 的视图
  _matmul_impl(
    view_out,
    As.select(0, 2), // 包含 A^2
    _linear_combination<scalar_t>(
      As.narrow(0, 0, 3), // {I, A, A^2} 的张量视图
      {1 / 2.0, 1 / 6.0, 1 / 24.0} // 线性组合的权重
    )
  );

  // 返回 T4 = I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
  return _linear_combination<scalar_t>(
    As, {1.0, 1.0, 0.0, 1.0} // {I, A, A^2, A^3} 的线性组合权重
  );
}
// 计算张量 A 的 T8 矩阵
Tensor compute_T8(const Tensor& A) {
  // 定义常量
  constexpr scalar_t sqrt_177 = 0.1330413469565007072504e+2;
  constexpr scalar_t x3 = 2. / 3.;
  constexpr scalar_t x1 = x3 * ((1. + sqrt_177) / 88.);
  constexpr scalar_t x2 = x3 * ((1. + sqrt_177) / 352.);
  constexpr scalar_t x4 = (-271. + 29. * sqrt_177) / (315. * x3);
  constexpr scalar_t x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
  constexpr scalar_t x6 = (-99. + 11. * sqrt_177) / (5040. * x3);
  constexpr scalar_t x7 = (89. - sqrt_177) / (5040. * x3);
  constexpr scalar_t y2 = (857. - 58. * sqrt_177) / 630.;

  // 分配 As 数组空间
  auto As = _allocate_buffer(A, 5);
  // 填充 As 数组中的矩阵幂次，包括 A^0, A^1, A^2
  _fill_matrix_powers(As, A, 3);

  // 输出 A4
  auto view_out = As.select(0, 3);
  // 计算 A4 = A^2 * (x1 * A + x2 * A^2)
  _matmul_impl(
    view_out,
    // As.select(0, 2) = A^2
    As.select(0, 2),
    _linear_combination<scalar_t>(
      // 从 As 中提取 {A, A^2}
      As.narrow(0, 1, 2),
      {x1, x2}
    )
  );

  // 输出 A8
  view_out = As.select(0, 4);
  // 计算 A8 = (x3 * A^2 + A4) * (x4 * I + x5 * A + x6 * A^2 + x7 * A4)
  _matmul_impl(
    view_out,
    // x3 * A^2 + A4
    _linear_combination<scalar_t>(
      As.narrow(0, 2, 2),
      {x3, 1.0}
    ),
    _linear_combination<scalar_t>(
      // 从 As 中提取 {I, A, A^2, A4}
      As.narrow(0, 0, 4),
      {x4, x5, x6, x7}
    )
  );

  // 返回 I + A + y2 * A^2 + A8
  return _linear_combination<scalar_t>(
    As, {1.0, 1.0, y2, 0.0, 1.0}
  );
}

// 计算张量 A 的 T12 矩阵
template <typename scalar_t>
Tensor compute_T12(const Tensor& A) {
  // 定义常量
  constexpr int num_prods = 4;
  // 定义矩阵 b 的系数
  array2d<scalar_t, num_prods, num_prods> b = {{
    { // 第一行
      9.0198e-16,
      0.46932117595418237389,
      -0.20099424927047284052,
      -0.04623946134063071740
    },
    { // 第二行
      5.31597895759871264183,
      1.19926790417132231573,
      0.01179296240992997031,
      0.01108844528519167989
    },
    { // 第三行
      0.18188869982170434744,
      0.05502798439925399070,
      0.09351590770535414968,
      0.00610700528898058230
    },
    { // 第四行
      -2.0861320e-13,
      -0.13181061013830184015,
      -0.02027855540589259079,
      -0.00675951846863086359
    }
  }};

  // 将 b 的系数转换为张量，并移到与 A 相同的设备上
  auto bs = at::from_blob(
    reinterpret_cast<void*>(&b),
    {num_prods, num_prods},
    {num_prods, 1},
    c10::toRealValueType(A.scalar_type())
  );
  bs = _move_memory_if_cuda_input(bs, A);

  // 分配 As 数组空间
  auto As = _allocate_buffer(A, num_prods);
  // 填充 As 数组中的矩阵幂次
  _fill_matrix_powers(As, A, num_prods);

  // 计算线性组合 Bs
  auto Bs = at::native::_compute_linear_combination(As, bs);

  // 输出 A6
  auto view_out = As.select(0, 0);
  // 计算 A6
  Bs.select(0, 2).add_(_matmul_impl(
    view_out,
    Bs.select(0, 3),
    Bs.select(0, 3)
  ));

  // 返回计算结果 Bs.select(0, 0) + A6
  return Bs.select(0, 0).add_(_matmul_impl(
    view_out,
    Bs.select(0, 1).add_(Bs.select(0, 2)),
    Bs.select(0, 2)
  ));
}

// 计算张量 A 的 T18 矩阵
template <typename scalar_t>
Tensor compute_T18(const Tensor& A) {
  // 定义常量
  constexpr int num_prods = 5;
  // 定义矩阵 b 的系数
  array2d<scalar_t, num_prods, num_prods> b = {{
    // 这里省略了矩阵 b 的系数定义，因为代码未完整给出
    {
      // 第一个系数向量
      0.,
      // 第二个系数向量
      -1.00365581030144618291e-01,
      // 第三个系数向量
      -8.02924648241156932449e-03,
      // 第四个系数向量
      -8.92138498045729985177e-04,
      // 第五个系数向量
      0.
    },
    {
      // 第一个系数向量
      0.,
      // 第二个系数向量
      3.97849749499645077844e-01,
      // 第三个系数向量
      1.36783778460411720168e+00,
      // 第四个系数向量
      4.98289622525382669416e-01,
      // 第五个系数向量
      -6.37898194594723280150e-04
    },
    {
      // 第一个系数向量
      -1.09676396052962061844e+01,
      // 第二个系数向量
      1.68015813878906206114e+00,
      // 第三个系数向量
      5.71779846478865511061e-02,
      // 第四个系数向量
      -6.98210122488052056106e-03,
      // 第五个系数向量
      3.34975017086070470649e-05
    },
    {
      // 第一个系数向量
      -9.04316832390810593223e-02,
      // 第二个系数向量
      -6.76404519071381882256e-02,
      // 第三个系数向量
      6.75961301770459654925e-02,
      // 第四个系数向量
      2.95552570429315521194e-02,
      // 第五个系数向量
      -1.39180257516060693404e-05
    },
    {
      // 第一个系数向量
      0.,
      // 第二个系数向量
      0.,
      // 第三个系数向量
      -9.23364619367118555360e-02,
      // 第四个系数向量
      -1.69364939002081722752e-02,
      // 第五个系数向量
      -1.40086798182036094347e-05
    }
    };
    
    // 将上述系数向量 `b` 聚合成一个张量，并将其移动到与矩阵 `A` 相同的设备上
    auto bs = at::from_blob(
      reinterpret_cast<void*>(&b),  // 使用 `b` 的内存创建张量
      {num_prods, num_prods},       // 张量的形状
      {num_prods, 1},               // 张量的步幅
      c10::toRealValueType(A.scalar_type())  // 确保与 `A` 的数据类型匹配
    );
    bs = _move_memory_if_cuda_input(bs, A);  // 如果输入的 `bs` 在 CUDA 设备上，则移动其内存到 `A` 所在设备
    
    auto As = _allocate_buffer(A, num_prods);  // 分配用于存储矩阵幂的缓冲区 `As`
    _fill_matrix_powers(As, A, num_prods);     // 计算矩阵 `A` 的幂，并存储在 `As` 中
    
    auto Bs = at::native::_compute_linear_combination(As, bs);  // 计算 `As` 和 `bs` 的线性组合
    
    // 临时缓冲区用于存储这个矩阵乘积
    auto view_out = As.select(0, 0);  // 选择 `As` 的第一个维度上的第一个张量，作为视图输出
    
    // 计算 `A^9`
    Bs.select(0, 3).add_(_matmul_impl(
      view_out,          // 矩阵乘法的左操作数
      Bs.select(0, 0),   // 矩阵乘法的右操作数之一
      Bs.select(0, 4)    // 矩阵乘法的右操作数之二
    ));
    
    // 返回 `A^9` 与另一个矩阵乘积的和
    return Bs.select(0, 1).add_(_matmul_impl(
      view_out,                            // 矩阵乘法的左操作数
      Bs.select(0, 2).add_(Bs.select(0, 3)),  // 矩阵乘法的右操作数之一
      Bs.select(0, 3)                      // 矩阵乘法的右操作数之二
    ));
}
// Scale
// 我们最终需要进行矩阵乘法以计算结果。
// 例如，如果 `norm` 等于 [27, 6, 6, 0.05]，那么我们将得到 `s` 等于 [4, 1, 1, 0]，
// 因此我们可以使用它来计算结果，逐个计算矩阵 `matrix[0]^(2^4)`，`matrix[1]^(2^1)` 和 `matrix[2]^(2^1)`，
// 逐个计算将会相当慢。
const auto s = (at::ceil(at::log2(norm / theta))).clamp(/*min=*/0);

// Calculate pow(2, -s)
const auto pow2s = at::pow(2, -s);

// Scale `a` by `pow2s`
const auto a_scaled = a * pow2s.view({-1, 1, 1});

// Compute matrix exponentiation scaled by `a_scaled`
auto mexp_scaled = at::native::compute_T18<scalar_t>(a_scaled);

// Sort:
// 考虑输入是方阵，所以如果我们先对 `matrix 0, 1, 2` 求幂次，
// 那么剩下的事情只需将 `matrix 0` 乘以 (2^4 - 1) 次，这给了我们批量计算矩阵乘法的机会。
// 首先需要做的是对张量 `s` 进行排序，这将有助于按范围进行矩阵乘法计算。
auto [sorted_s, sorted_s_inds] = at::sort(s, /*dim=*/0);
sorted_s = sorted_s.to(at::kLong);

// Unique consecutive:
// 然后调用 `unique_consecutive` 函数，将其用于拆分 `sorted_s`。
auto split_counts = std::get<2>(at::unique_consecutive(sorted_s, true, /*return_counts=*/true));

// Compute split edges:
// 我们还需要知道每个拆分的最后一个元素的索引，以便知道每个拆分矩阵需要进行乘法的次数。
auto split_edges = at::cumsum(split_counts, /*dim=*/0) - 1;
auto unique_s = sorted_s.index_select(0, split_edges).clamp(/*min=*/0);

// Compute multiplication times per split:
// 计算每个拆分需要进行乘法的次数。
auto mul_times = at::diff(unique_s, 1, -1, /*prepend=*/unique_s.new_zeros({1}));

// Square
// 计算 `section_values`，这是 `split_counts` 和 `mul_times` 的连接。
auto section_values = at::cat({split_counts, mul_times}, 0).to(at::kCPU);

// Internal assertion
// 进行内部断言，确保 `section_values` 是连续的。
TORCH_INTERNAL_ASSERT(section_values.is_contiguous());

// Determine section numel
// 计算 `section_values` 中元素的数量的一半。
const auto section_numel = section_values.numel() / 2;

// Get pointers to section counts and pointers
// 获取 `section_values` 的数据指针和指针数组。
auto scs = section_values. template data_ptr<int64_t>();
auto pts = &scs[section_numel];

// Batch matrix multiplication:
// 现在我们将批量进行矩阵乘法，使用上述示例：
// 1. 将所有矩阵乘以 0 (`mul_times[0]`) 次，然后使用 `slice` 函数获取剩余矩阵（`split_counts[0]`），
// 2. 将剩余矩阵乘以 1 次并使用 `slice` 函数获取 `acc[2:]`，
// 3. 将剩余矩阵乘以 3 次并使用 `slice` 函数获取 `acc[1:]`。
// 所有处理过的矩阵将存储在 `output_pieces` 中。
std::vector<Tensor> output_pieces;
auto acc = mexp_scaled.index_select(0, sorted_s_inds);
for (int64_t i = 0; i < section_numel; ++i) {
    // 对于当前的pts[i]值，执行pts[i]次循环
    for (int64_t j = 0; j < pts[i]; j++) {
      // 为了避免由at::matmul引起的AMP自动转换，创建一个与acc相同类型和大小的空张量acc_out
      auto acc_out = at::empty_like(acc);
      // 计算acc_out = acc @ acc（矩阵乘法），结果存储在acc_out中
      acc = at::matmul_out(acc_out, acc, acc);
    }
    // 将acc的切片（从0到scs[i]）加入到output_pieces中
    output_pieces.push_back(acc.slice(0, 0, scs[i]));
    // 更新acc为其切片（从0到scs[i]）
    acc = acc.slice(0, scs[i]);
  }

  // 将output_pieces中的张量按照维度0进行连接，形成最终的输出张量output
  auto output = at::cat(output_pieces, 0);
  // 使用sorted_s_inds对output进行按索引排序后返回结果
  return output.index_select(0, at::argsort(sorted_s_inds));
// 定义模板函数 `mexp_impl`，用于计算矩阵指数
template <typename scalar_t>
Tensor mexp_impl(
  const Tensor& a,  // 输入矩阵
  std::array<scalar_t, total_n_degs> thetas,  // 存储阈值的数组
  bool compute_highest_degree_approx = false  // 是否计算最高阶近似，默认为否
) {
  const auto norm = operator_1_norm(a);  // 计算输入矩阵的1-范数
  const auto batch_size = a.size(0);  // 获取矩阵的批处理大小

  if (batch_size > 1) {
    compute_highest_degree_approx = true;  // 若批处理大小大于1，则强制计算最高阶近似
  }

  if (!compute_highest_degree_approx) {
    // 为了避免由于矩阵包含NaN值而导致的未定义行为，将NaN值填充到`res`中，
    // 这样如果输入包含NaN值，则直接返回包含NaN的`res`
    auto res = at::full_like(a, std::numeric_limits<double>::quiet_NaN(), {},
                             at::MemoryFormat::Contiguous);

    // `norm_cpu` 用于根据矩阵的范数决定哪些张量需要哪种近似，
    // 决策在CPU上进行，当输入矩阵在CUDA上时，需要在设备之间移动数据，
    // 但仅需进行一次CPU-CUDA同步（而不是6次），整体性能更优（经过基准测试验证）。
    const auto norm_cpu = (a.device().type() == at::kCUDA)
      ? norm.to(at::kCPU) : norm;

    // 定义计算不同阶数近似的函数数组 `compute_Ts`
    constexpr std::array<
      Tensor(*)(const Tensor&),
      total_n_degs - 1>
    compute_Ts = {
      compute_T1, compute_T2, compute_T4<scalar_t>,
      compute_T8<scalar_t>, compute_T12<scalar_t>
    };

    // 遍历计算不同阶数的近似
    for (int i = 0; i < total_n_degs - 1; ++i) {
      auto norm_lower_bound = (i == 0) ? static_cast<scalar_t>(-1) : thetas[i - 1];
      auto norm_upper_bound = thetas[i];

      // 计算当前范数区间内的索引
      auto idx_curr_norm_interval = (
        (norm_lower_bound < norm_cpu) * (norm_cpu <= norm_upper_bound)
      ).nonzero().squeeze(-1);

      // 如果存在符合条件的索引
      if (idx_curr_norm_interval.numel()) {
        // 将索引移动到相应设备上
        auto idx_to_device = _move_memory_if_cuda_input(
          idx_curr_norm_interval, a
        );
        // 根据索引选择子矩阵并计算近似值，更新到`res`中
        auto sub_a = at::index_select(a, 0, idx_to_device);
        res.index_put_({idx_to_device}, compute_Ts[i](sub_a));
      }
    }

    // 计算大范数区间的索引
    auto idx_large_norm = (norm_cpu >= thetas[total_n_degs - 2])
      .nonzero().squeeze(-1);

    // 如果存在符合条件的索引
    if (idx_large_norm.numel()) {
      // 将索引移动到相应设备上
      auto idx_to_device = _move_memory_if_cuda_input(
        idx_large_norm, a
      );
      // 根据索引选择大范数子矩阵，计算其近似值，并更新到`res`中
      auto a_large_norm = at::index_select(a, 0, idx_to_device);
      auto large_norm_subset = at::index_select(norm, 0, idx_to_device);
      auto mexp_out = compute_T18_scale_square(
        a_large_norm,
        large_norm_subset,
        thetas[total_n_degs - 1]
      );
      res.index_put_({idx_large_norm}, mexp_out);
    }

    return res;  // 返回计算得到的矩阵指数近似结果
  }

  return compute_T18_scale_square(
    a, norm,
    thetas[total_n_degs - 1]
  );  // 计算最高阶近似的矩阵指数，并返回结果
}

// 矩阵指数实现
// 计算矩阵指数函数，对给定的批量方阵进行计算
// 实现基于以下论文：
// Bader, P.; Blanes, S.; Casas, F.
// Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
// Mathematics 2019, 7, 1174.
Tensor linalg_matrix_exp(const Tensor& a) {
  // 检查输入是否为方阵
  squareCheckInputs(a, "linalg.matrix_exp");
  // 检查输入是否为浮点型或复数型
  checkFloatingOrComplex(a, "linalg.matrix_exp");

  // 禁用 TF32
  NoTF32Guard disable_tf32;

  // 处理特殊情况
  const auto n = a.size(-1);
  if (n == 0) {
    return a.clone();  // 如果矩阵大小为0，直接返回克隆的输入
  } else if (n == 1) {
    return a.exp();  // 如果矩阵大小为1，直接返回指数函数的结果
  } else {
    return at::native::mexp(a);  // 对于其他情况，调用 mexp 函数进行计算
  }
}

// 别名函数，调用 linalg_matrix_exp 来计算矩阵指数
Tensor matrix_exp(const Tensor& a) {
  return at::linalg_matrix_exp(a);
}

// TODO 应该废弃，推荐使用 FunctionsManual.cpp 中的 linalg_matrix_exp_differential
// 对矩阵指数函数的反向传播，禁用 TF32 加速
Tensor matrix_exp_backward(const Tensor& self, const Tensor& grad) {
  NoTF32Guard disable_tf32;
  // 调用 backward_analytic_function_of_a_matrix 函数，传入自身张量和梯度张量，并定义 lambda 函数以计算矩阵指数
  return backward_analytic_function_of_a_matrix(
    self, grad,
    [](const Tensor& a) {
      return a.matrix_exp();  // 返回输入张量 a 的矩阵指数
    }
  );
}

TORCH_IMPL_FUNC(linalg_vector_norm_out)(const Tensor& self, const Scalar& scalar_ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype, const Tensor& result) {
  // 将大整数 ord 转换为 double，对于大于 10^53（或负数）的值，转换为 double 可能会引入误差，但在这里是可接受的
  auto ord = scalar_ord.toDouble();
  auto dim = opt_dim.value_or(IntArrayRef{});  // 获取或者使用空的维度数组
  auto size = self.sizes();  // 获取自身张量的尺寸
  auto ndim = self.dim();  // 获取自身张量的维度数

  auto opt_dim_ = dim.vec();  // 转换为 vector 形式的维度
  maybe_wrap_dims(opt_dim_, ndim);  // 可能调整维度

  using Int = IntArrayRef::value_type;  // 使用 Int 表示维度数组中的值类型
  std::vector<Int> all_dim(ndim);  // 创建包含所有维度的向量
  std::iota(all_dim.begin(), all_dim.end(), 0);  // 从 0 开始递增填充 all_dim 向量

  bool is_all_reduce = !opt_dim.has_value() || opt_dim.value().empty();  // 是否进行全维度缩减
  auto reduce_dim = is_all_reduce ? all_dim : opt_dim_;  // 确定需要缩减的维度

  bool is_reduce_over_1D_vector = true;  // 默认为对 1 维向量进行缩减
  for (auto i : reduce_dim) {  // 遍历缩减的维度
    if (size[i] != 1){  // 如果指定维度的大小不为 1
      is_reduce_over_1D_vector = false;  // 将标志位设为 false
      break;  // 跳出循环
    }
  }

  if (is_reduce_over_1D_vector) {  // 如果是对 1 维向量进行缩减
    Tensor self_;
    if (opt_dtype.has_value()) {  // 如果指定了输出类型
      self_ = self.to(*opt_dtype);  // 将自身张量转换为指定类型
    } else {
      self_ = self;  // 否则不变
    }
    if (ord != 0.0) {  // 如果 ord 不为 0
      keepdim ? at::abs_outf(self_, const_cast<Tensor&>(result)) : at::abs_outf(self_.squeeze(reduce_dim), const_cast<Tensor&>(result));  // 计算张量的绝对值
    } else {  // 如果 ord 为 0
      keepdim ? at::ne_outf(self_, 0, const_cast<Tensor&>(result)) : at::ne_outf(self_.squeeze(reduce_dim), 0, const_cast<Tensor&>(result));  // 计算张量与零不等的元素
    }
    return;  // 返回
  }

  // 不需要显式处理 opt_dtype，因为已经在 result 的数据类型中编码

  // https://github.com/pytorch/pytorch/issues/52648
  // 缩减操作总是使用 std::abs 计算绝对值。在此函数的反向传播中，我们需要定位作为最大值的索引。为此，
  // 我们使用 self.abs() == result 来定位最大元素的索引。现在，self.abs() 可能会分派到一个向量化实现，
  // 这与 std::abs(std::complex<T>) 的实现略有不同。因此，在反向传播中，为了能够计算正确的索引，我们
  // 需要在前向和反向中都使用 self.abs()
  Tensor self_;
  if (self.is_cpu() && self.is_complex() && std::abs(ord) == INFINITY) {  // 如果张量在 CPU 上且是复数且 ord 的绝对值为无穷大
    if (opt_dtype.has_value()) {  // 如果指定了输出类型
      self_ = self.to(*opt_dtype).abs();  // 将自身张量转换为指定类型并计算绝对值
    } else {
      self_ = self.abs();  // 否则计算自身的绝对值
    }
  } else {
    self_ = self;  // 否则不变
  }

  auto iter = make_reduction("vector_norm", const_cast<Tensor&>(result), self_, dim, keepdim, result.scalar_type());  // 创建归约迭代器
  norm_stub(iter.device_type(), iter, ord);  // 使用指定的设备类型和迭代器计算范数
}
// 检查矩阵 A 是否符合矩阵范数计算的要求
static void _linalg_matrix_norm_checks(const Tensor& A, std::vector<int64_t>& dim, optional<ScalarType> opt_dtype, bool low_precision) {
  // 检查 A 是否为矩阵（即二维张量）
  at::native::checkIsMatrix(A, "linalg.matrix_norm");
  // 检查 A 的元素类型是否为浮点数或复数
  at::native::checkFloatingOrComplex(A, "linalg.matrix_norm", /*low_precision*/low_precision);

  // 检查 dim 是否为长度为 2 的向量
  TORCH_CHECK(dim.size() == 2, "linalg.matrix_norm: dim must be a 2-tuple. Got ", dim);
  // 在封装 dim 之前，先检查并修正不规则的情况，例如 A.ndim = 2, dim = (1, -1)
  // 封装过程中 dim 将被原地修改
  maybe_wrap_dims(dim, A.dim());
  // 检查 dim 中的维度值不能相等
  TORCH_CHECK(dim[0] != dim[1], "linalg.matrix_norm: dims must be different. Got (", dim[0], ", ", dim[1], ")");

  // 检查 dtype 是否符合矩阵范数计算的要求
  at::detail::check_linalg_norm_dtype(opt_dtype, A.scalar_type(), "linalg.matrix_norm");
}

// 计算矩阵的范数
Tensor linalg_matrix_norm(
    const Tensor& A,
    const Scalar& scalar_ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  // 检查 ord 是否支持
  auto ord = scalar_ord.toDouble();
  auto abs_ord = std::abs(ord);
  TORCH_CHECK(abs_ord == 2. || abs_ord == 1. || abs_ord == INFINITY, "linalg.matrix_norm: Order ", ord, " not supported.");

  // 将 dim 转换为标准的 vector 形式
  auto dim_ = dim.vec();
  // 对 A、dim 和 dtype 进行检查
  _linalg_matrix_norm_checks(A, dim_, opt_dtype, /*low_precision*/abs_ord != 2.);

  // 定义一个函数，根据 ord 和 keepdim 来选择是返回最大值还是最小值
  auto max_min = [ord, keepdim](const Tensor& A, int64_t dim) { return ord > 0 ? A.amax(dim, keepdim) : A.amin(dim, keepdim); };

  // 如果 ord 是 2
  if (abs_ord == 2.) {
    // 创建维度置换，将相关维度移到最后
    auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], A.dim());

    // 如果指定了 opt_dtype，则将 A 转换为相应的数据类型
    auto A_ = opt_dtype.has_value() ? A.to(*opt_dtype) : A;
    // 执行 SVD 操作并计算奇异值
    auto result = max_min(at::linalg_svdvals(A_.permute(permutation)), -1);
    // 如果 keepdim 为 true，则恢复维度的原始顺序
    if (keepdim) {
      auto permutation_reverse = create_reverse_permutation(std::move(permutation));
      result = result.unsqueeze(-1).permute(permutation_reverse);
    }
    return result;
  } else {  // 如果 ord 是 1, -1, inf, -inf
    // 如果 ord 是 INFINITY，则对调 dim_[0] 和 dim_[1] 的顺序
    if (abs_ord == INFINITY) {
      std::swap(dim_[0], dim_[1]);
    }

    // 如果不保持维度，并且第一个维度的索引小于第二个维度的索引，则减少 dim_[1] 的值
    if (!keepdim && (dim_[0] < dim_[1])) {
      dim_[1]--;
    }
    // 对 A 求取指定维度上的向量范数
    return max_min(at::linalg_vector_norm(A, 1., {dim_[0]}, keepdim, opt_dtype), dim_[1]);
  }
}

// 在给定输出张量中计算矩阵的范数
Tensor& linalg_matrix_norm_out(
    const Tensor& A,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& result) {
  // 检查输出张量 result 和输入张量 A 是否在同一设备上
  checkSameDevice("linalg.matrix_norm", A, result);
  // 计算矩阵的范数并存储在 result 中
  auto out = at::linalg_matrix_norm(A, ord, dim, keepdim, opt_dtype);
  // 检查 result 的数据类型是否与输出的数据类型一致
  TORCH_CHECK(out.scalar_type() == result.scalar_type(),
              "linalg.matrix_norm expected out tensor dtype ", out.scalar_type(),
              " but got: ", result.scalar_type());
  // 调整 result 的大小以匹配输出张量 out 的大小
  at::native::resize_output(result, out.sizes());
  // 将计算得到的范数结果复制到 result 中
  result.copy_(out);
  // 返回更新后的 result
  return result;
}
    const Tensor& A,                         // A 是输入张量，是要进行矩阵范数计算的对象
    c10::string_view ord,                    // ord 是指定的范数类型，可以是 "fro" (Frobenius 范数) 或 "nuc" (核范数)
    IntArrayRef dim,                         // dim 是指定操作的维度
    bool keepdim,                            // keepdim 指示是否保持输出张量的维度
    optional<ScalarType> opt_dtype) {        // opt_dtype 是可选的输出数据类型

  // 首先检查 ord，因为它将在对 A 的数据类型进行检查时使用
  TORCH_CHECK(ord == "fro" || ord == "nuc", "linalg.matrix_norm: Order ", ord, " not supported.");

  auto dim_ = dim.vec();                    // 将 IntArrayRef dim 转换为 std::vector<int64_t> dim_

  // 对 A、dim 和 dtype 进行检查
  _linalg_matrix_norm_checks(A, dim_, opt_dtype, /*low_precision*/ord != "nuc");

  if (ord == "fro") {                       // 如果 ord 是 "fro"，计算 Frobenius 范数
    return at::linalg_vector_norm(A, 2, dim_, keepdim, opt_dtype);
  } else {                                  // 如果 ord 是 "nuc"，计算核范数
    auto A_ = opt_dtype.has_value() ? A.to(*opt_dtype) : A;  // 将 A 转换为指定的数据类型（如果有指定）

    // 将指定维度的轴移动到张量的末尾
    auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], A_.dim());
    // 计算 A 的奇异值并对其进行求和以计算核范数
    auto result = at::linalg_svdvals(A_.permute(permutation)).sum(-1, keepdim);
    
    if (keepdim) {
      // 如果 keepdim 为 true，逆转置换排列，使结果张量保持与输入张量相同的维度顺序
      auto permutation_reverse = create_reverse_permutation(std::move(permutation));
      result = result.unsqueeze(-1).permute(permutation_reverse);
    }
    return result;                          // 返回计算得到的范数结果
  }
}
}

// 定义一个函数 linalg_matrix_norm_out，计算矩阵范数并将结果存入给定的 result 引用参数中
Tensor& linalg_matrix_norm_out(
    // 输入参数 A：要计算范数的张量
    const Tensor& A,
    // 输入参数 ord：指定的范数类型，以字符串形式给出
    c10::string_view ord,
    // 输入参数 dim：指定的维度
    IntArrayRef dim,
    // 输入参数 keepdim：是否保持结果张量的维度
    bool keepdim,
    // 输入参数 opt_dtype：可选的输出数据类型
    optional<ScalarType> opt_dtype,
    // 输出参数 result：用于存储计算结果的张量
    Tensor& result) {
  // 检查输入张量 A 和输出张量 result 的设备是否相同
  checkSameDevice("linalg.matrix_norm", A, result);
  // 调用 at::linalg_matrix_norm 函数计算矩阵范数的结果 out
  auto out = at::linalg_matrix_norm(A, ord, dim, keepdim, opt_dtype);
  // 检查计算得到的结果张量 out 的数据类型是否与 result 相同
  TORCH_CHECK(out.scalar_type() == result.scalar_type(),
              "linalg.matrix_norm expected out tensor dtype ", out.scalar_type(),
              " but got: ", result.scalar_type());
  // 调整 result 张量的大小以匹配计算结果 out 的大小
  at::native::resize_output(result, out.sizes());
  // 将计算得到的范数结果 out 复制到 result 张量中
  result.copy_(out);
  // 返回存储结果的 result 引用
  return result;
}

// 数值或空的范数
Tensor linalg_norm(
    // 输入参数 X：要计算范数的张量
    const Tensor& X,
    // 可选参数 opt_ord：指定的范数类型，作为标量值
    const optional<Scalar>& opt_ord,
    // 可选参数 opt_dim：指定的维度
    OptionalIntArrayRef opt_dim,
    // 输入参数 keepdim：是否保持结果张量的维度
    bool keepdim,
    // 可选参数 opt_dtype：可选的输出数据类型
    optional<ScalarType> opt_dtype) {
  // 如果 opt_dim 被指定，则检查其长度为 1 或 2
  if (opt_dim.has_value()) {
    TORCH_CHECK(opt_dim->size() == 1 || opt_dim->size() == 2, "linalg.norm: If ",
              "dim is specified, it must be of length 1 or 2. Got ", *opt_dim);
  } else {
    // 如果 opt_ord 被指定，则检查输入张量 X 的维度是否为 1 或 2
    if (opt_ord.has_value()) {
      TORCH_CHECK(X.dim() == 1 || X.dim() == 2, "linalg.norm: If ",
                  "dim is not specified but ord is, the input must be 1D or 2D. Got ", X.dim(), "D.");
    }
  }

  // 如果 opt_ord 为 None，使用默认的 2-范数或者 Frobenius 范数，通过 vector_norm 函数处理
  if (opt_ord.has_value() &&
       ((opt_dim.has_value() && opt_dim->size() == 2) ||
        (!opt_dim.has_value() && X.dim() == 2))) {
    // 定义整数类型 Int，用于处理 dim 的值
    using Int = IntArrayRef::value_type;
    // 如果 opt_dim 被指定，则将其转换为标准的 vector 类型；否则使用默认的维度 {0, 1}
    auto dim = opt_dim.has_value() ? opt_dim.value().vec() : std::vector<Int>{0, 1};
    // 调用 at::linalg_matrix_norm 计算矩阵范数的结果，并返回
    return at::linalg_matrix_norm(X, *opt_ord, dim, keepdim, opt_dtype);
  } else {
    // 否则，将 opt_ord 转换为标量值 scalar_ord，并调用 at::linalg_vector_norm 函数计算向量范数
    auto scalar_ord = opt_ord.value_or(Scalar(2.));
    return at::linalg_vector_norm(X, scalar_ord, opt_dim, keepdim, opt_dtype);
  }
}

// 指定输出张量的范数计算
Tensor& linalg_norm_out(
    // 输入参数 X：要计算范数的张量
    const Tensor& X,
    // 可选参数 opt_ord：指定的范数类型，作为标量值
    const optional<Scalar>& opt_ord,
    // 可选参数 opt_dim：指定的维度
    OptionalIntArrayRef opt_dim,
    // 输入参数 keepdim：是否保持结果张量的维度
    bool keepdim,
    // 可选参数 opt_dtype：可选的输出数据类型
    optional<ScalarType> opt_dtype,
    // 输出参数 result：用于存储计算结果的张量
    Tensor& result) {
  // 检查输入张量 X 和输出张量 result 的设备是否相同
  checkSameDevice("linalg.norm", X, result);
  // 调用 at::linalg_norm 函数计算范数的结果 out
  auto out = at::linalg_norm(X, opt_ord, opt_dim, keepdim, opt_dtype);
  // 检查计算得到的结果张量 out 的数据类型是否与 result 相同
  TORCH_CHECK(out.scalar_type() == result.scalar_type(),
              "linalg.norm expected out tensor dtype ", out.scalar_type(),
              " but got: ", result.scalar_type());
  // 调整 result 张量的大小以匹配计算结果 out 的大小
  at::native::resize_output(result, out.sizes());
  // 将计算得到的范数结果 out 复制到 result 张量中
  result.copy_(out);
  // 返回存储结果的 result 引用
  return result;
}

// Frobenius 范数和核范数
Tensor linalg_norm(
    // 输入参数 X：要计算范数的张量
    const Tensor& X,
    // 输入参数 ord：指定的范数类型，以字符串形式给出
    c10::string_view ord,
    // 可选参数 opt_dim：指定的维度
    OptionalIntArrayRef opt_dim,
    // 输入参数 keepdim：是否保持结果张量的维度
    bool keepdim,
    // 可选参数 opt_dtype：可选的输出数据类型
    optional<ScalarType> opt_dtype) {
  // 如果 opt_dim 被指定，则检查其长度为 1 或 2
  if (opt_dim.has_value()) {
    TORCH_CHECK(opt_dim->size() == 1 || opt_dim->size() == 2, "linalg.norm: If ",
              "dim is specified, it mut be of length 1 or 2. Got ", *opt_dim);
  } else {
    // 否则，如果未指定 opt_dim，则检查输入张量 X 的维度是否为 1 或 2
    TORCH_CHECK(X.dim() == 1 || X.dim() == 2, "linalg.norm: If ",
                "dim is specified, it must be of length 1 or 2. Got ", X.dim(), "D.");
  }
    # 使用 TORCH_CHECK 函数检查输入张量 X 的维度是否为1或2，否则抛出异常信息
    TORCH_CHECK(X.dim() == 1 || X.dim() == 2, "linalg.norm: If ",
                "dim is not specified but ord is, the input must be 1D or 2D. Got ", X.dim(), "D.");
  }
  # 定义 Int 类型为 IntArrayRef::value_type
  using Int = IntArrayRef::value_type;
  # 根据是否提供了 opt_dim 的值来确定 dim 的取值，如果提供了则使用其值的向量表示，否则使用默认值 {0, 1}
  auto dim = opt_dim.has_value() ? opt_dim.value().vec() : std::vector<Int>{0, 1};
  # 调用 ATen 的 linalg_matrix_norm 函数计算输入张量 X 的矩阵范数
  return at::linalg_matrix_norm(X, ord, dim, keepdim, opt_dtype);
}

Tensor& linalg_norm_out(const Tensor& X, c10::string_view ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  // 检查输入张量和输出张量是否在同一设备上
  checkSameDevice("linalg.norm", X, result);
  // 计算 linalg.norm 操作的结果
  auto out = at::linalg_norm(X, ord, opt_dim, keepdim, opt_dtype);
  // 检查输出张量的数据类型是否符合预期
  TORCH_CHECK(out.scalar_type() == result.scalar_type(),
              "linalg.norm expected out tensor dtype ", out.scalar_type(),
              " but got: ", result.scalar_type());
  // 调整输出张量的大小以匹配计算结果
  at::native::resize_output(result, out.sizes());
  // 将计算结果复制到输出张量中
  result.copy_(out);
  // 返回输出张量
  return result;
}

////////////////////////////////////////////////////////////////////////////////
//                              Frobenius Norm                                //
////////////////////////////////////////////////////////////////////////////////

Tensor frobenius_norm(const Tensor& self, IntArrayRef dim, bool keepdim) {
  // 获取张量所在的设备
  auto device = self.device();
  // 如果张量布局为 Strided，并且设备为 CPU、CUDA 或 Meta
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    // 发出一次性警告，指出 frobenius_norm 已弃用，推荐使用 linalg.vector_norm 替代
    TORCH_WARN_ONCE(
      "at::frobenius_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.vector_norm(A, 2., dim, keepdim)` instead"
    );
  }
  // 检查维度数是否不超过2
  TORCH_CHECK(dim.size() <= 2,
              "Expected at most 2 dimensions, but got ", dim.size(), " dimensions instead.");
  // 调用 at::norm 函数计算 Frobenius 范数
  return at::norm(self, 2., dim, keepdim);
}

Tensor &frobenius_norm_out(const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    Tensor& result) {
  // 获取张量所在的设备
  auto device = self.device();
  // 如果张量布局为 Strided，并且设备为 CPU、CUDA 或 Meta
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    // 发出一次性警告，指出 frobenius_norm 已弃用，推荐使用 linalg.vector_norm 替代
    TORCH_WARN_ONCE(
      "at::frobenius_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.vector_norm(A, 2., dim, keepdim)` instead"
    );
  }
  // 检查维度数是否不超过2
  TORCH_CHECK(dim.size() <= 2,
              "Expected at most 2 dimensions, but got ", dim.size(), " dimensions instead.");
  // 调用 at::norm_out 函数计算 Frobenius 范数，并将结果存储到指定的输出张量中
  return at::norm_out(result, self, 2., dim, keepdim);
}

////////////////////////////////////////////////////////////////////////////////
//                                Nuclear Norm                                //
////////////////////////////////////////////////////////////////////////////////

Tensor nuclear_norm(const Tensor& self, bool keepdim) {
  // 调用 at::native::nuclear_norm 函数计算核范数
  return at::native::nuclear_norm(self, IntArrayRef({-2, -1}), keepdim);
}

Tensor &nuclear_norm_out(const Tensor& self, bool keepdim, Tensor& result) {
  // 获取张量所在的设备
  auto device = self.device();
  // 如果张量布局为 Strided，并且设备为 CPU、CUDA 或 Meta
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    // 发出一次性警告，指出 at::nuclear_norm 已被弃用，仅为了保持 JIT 兼容性而存在
    TORCH_WARN_ONCE(
      "at::nuclear_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.matrix_norm(A, 'nuc', dim, keepdim)` instead"
    );
  }
  // 使用 at::linalg_matrix_norm_out 计算 nuclear norm，并将结果存储在 result 中
  return at::linalg_matrix_norm_out(result, self, "nuc", IntArrayRef({-2, -1}), keepdim);
// This function computes the nuclear norm of a tensor along specified dimensions.
Tensor nuclear_norm(const Tensor& self, IntArrayRef dim, bool keepdim) {
  // Determine the device of the input tensor
  auto device = self.device();
  // Check if the tensor layout is strided and if it's on supported devices (CPU, CUDA, Meta)
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    // Issue a warning that nuclear_norm is deprecated and will be removed in the future
    TORCH_WARN_ONCE(
      "at::nuclear_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.matrix_norm(A, 'nuc', dim, keepdim)` instead"
    );
  }
  // Return the result of computing the nuclear norm using linalg_matrix_norm
  return at::linalg_matrix_norm(self, "nuc", dim, keepdim);
}

// This function computes the nuclear norm of a tensor and stores the result in the output tensor 'result'.
Tensor& nuclear_norm_out(const Tensor& self, IntArrayRef dim, bool keepdim, Tensor& result) {
  // Determine the device of the input tensor
  auto device = self.device();
  // Check if the tensor layout is strided and if it's on supported devices (CPU, CUDA, Meta)
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    // Issue a warning that nuclear_norm is deprecated and will be removed in the future
    TORCH_WARN_ONCE(
      "at::nuclear_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.matrix_norm(A, 'nuc', dim, keepdim)` instead"
    );
  }
  // Compute the nuclear norm of the input tensor and store the result in 'result'
  return at::linalg_matrix_norm_out(result, self, "nuc", dim, keepdim);
}

////////////////////////////////////////////////////////////////////////////////
//                              linalg.cond                                   //
////////////////////////////////////////////////////////////////////////////////

// This function computes the condition number of a tensor using matrix norms based on the 'ord' variant type.
static Tensor _linalg_cond_helper(const Tensor& self, std::variant<Scalar, c10::string_view> ord_variant) {
  Tensor inverse, info;
  // Compute the inverse and info using linalg_inv_ex
  std::tie(inverse, info) = at::linalg_inv_ex(self);
  // Expand dimensions of info tensor and replace values in inverse where info > 0 with INFINITY
  info.unsqueeze_(-1).unsqueeze_(-1);
  inverse.masked_fill_(info > 0, INFINITY);

  // Compute matrix norms based on the 'ord' variant type (e.g., 'fro' or 'nuc')
  return std::visit([&](auto&& ord) {
    // Compute matrix norm of the original tensor and its inverse
    Tensor norm_self = at::linalg_matrix_norm(self, ord);
    Tensor norm_inverse = at::linalg_matrix_norm(inverse, ord);
    // Compute the condition number as the product of the norms
    Tensor result = norm_self * norm_inverse;
    // Adjust for NumPy compatibility: replace NaN and INF values with appropriate values
    result.nan_to_num_(INFINITY, INFINITY, -INFINITY);
    return result;
  }, ord_variant);
}

// Return a tensor filled with zeros matching the shape of the input tensor except for the last two dimensions.
static Tensor _linalg_cond_empty_matrix(const Tensor& self, c10::ScalarType dtype) {
  // Determine the shape of the result tensor
  auto result_shape = IntArrayRef(self.sizes().cbegin(), self.sizes().cend()-2);
  // Create options for the result tensor matching the data type of the input tensor
  TensorOptions options = self.options().dtype(toRealValueType(self.scalar_type()));
  // Return a tensor filled with zeros of the specified shape and options
  return at::zeros(result_shape, options);
}

// Check the validity of the 'ord' variant type for computing the condition number.
static void _linalg_cond_check_ord(std::variant<Scalar, c10::string_view> ord_variant) {
  // Check if the variant type is Scalar (numeric ord) and validate the norm type
  if (ord_variant.index() == 0) {
    Scalar* ord = std::get_if<Scalar>(&ord_variant);
    double abs_ord = std::abs(ord->toDouble());
    // Ensure the norm type is one of the supported values (2.0, 1.0, INFINITY)
    TORCH_CHECK(abs_ord == 2.0 || abs_ord == 1.0 || abs_ord == INFINITY,
      "linalg.cond got an invalid norm type: ", ord->toDouble());
  }
  // Check if the variant type is c10::string_view (string ord) and validate the norm type
  else if (ord_variant.index() == 1) {
    c10::string_view* ord = std::get_if<c10::string_view>(&ord_variant);
    // Ensure the norm type is either 'fro' or 'nuc'
    TORCH_CHECK(*ord == "fro" || *ord == "nuc",
      "linalg.cond got an invalid norm type: ", *ord);
  }
  // If neither Scalar nor c10::string_view, an unsupported variant type is encountered
  else {
    // This case should not occur under normal usage and indicates an internal error
    TORCH_CHECK(false, "Invalid variant type for linalg.cond: ", ord_variant.index());
  }
}
    # 使用 TORCH_CHECK 宏来检查条件是否为 false，如果条件为 false，将发出错误消息
    TORCH_CHECK(false,
      "linalg.cond: something went wrong while checking the norm type");
    # 结束代码块
    }
// Numerical or None norms
Tensor linalg_cond(const Tensor& self, const optional<Scalar>& opt_ord) {
  // 检查输入张量的维度至少为2
  TORCH_CHECK(self.dim() >= 2, "linalg.cond: The input tensor must have at least 2 dimensions.");

  // 默认情况下使用2范数
  Scalar ord = opt_ord.has_value() ? opt_ord.value() : 2;

  // 将 ord 转换为标准变体类型
  std::variant<Scalar, c10::string_view> ord_variant = ord;
  _linalg_cond_check_ord(ord_variant);

  // NumPy 不定义0x0矩阵的条件数，对于这样的输入返回0.0
  if (self.sym_numel() == 0) {
    auto real_dtype = toRealValueType(typeMetaToScalarType(self.dtype()));
    return _linalg_cond_empty_matrix(self, real_dtype);
  }

  // 如果 ord == None 或 ord == ±2
  if (std::abs(ord.toDouble()) == 2.0) {
    // 计算奇异值
    auto singular_values = at::linalg_svdvals(self);
    // 奇异值按降序排列
    auto s_max = at::narrow(singular_values, /*dim=*/-1, /*start=*/0, /*length=*/1);
    auto s_min = at::narrow(singular_values, /*dim=*/-1, /*start=*/-1, /*length=*/1);
    Tensor result;
    if (ord.toDouble() == -2.0) {
      // 计算条件数
      result = s_min / s_max;
    } else {
      result = s_max / s_min;
    }
    // 压缩结果以符合 NumPy 的兼容性
    return result.squeeze(-1);
  }

  // 如果 ord == ±1 或 ord == ±inf
  if (ord.isFloatingPoint()) { // ord == ±1
    // 检查输入是否为方阵
    squareCheckInputs(self, ("linalg.cond(ord=" + std::to_string(ord.to<double>()) + ")").c_str());
  } else { // ord == ±inf
    // 检查输入是否为方阵
    squareCheckInputs(self, ("linalg.cond(ord=" + std::to_string(ord.to<int64_t>()) + ")").c_str());
  }
  // 调用辅助函数计算条件数
  return _linalg_cond_helper(self, std::move(ord_variant));
}

// 在输出张量 result 中计算 linalg_cond 的结果
Tensor& linalg_cond_out(const Tensor& self, const optional<Scalar>& opt_ord, Tensor& result) {
  // 检查结果张量和输入张量是否在相同设备上
  checkSameDevice("linalg.cond", result, self);
  // 确定结果张量的数据类型与输入张量兼容
  ScalarType real_dtype = toRealValueType(self.scalar_type());
  checkLinalgCompatibleDtype("linalg.cond", result.scalar_type(), real_dtype);

  // 调用 linalg_cond 计算条件数，并将结果复制到 result_tmp 中
  Tensor result_tmp = at::linalg_cond(self, opt_ord);
  // 调整输出张量的尺寸以匹配结果
  at::native::resize_output(result, result_tmp.sizes());
  // 将计算结果复制到输出张量 result 中
  result.copy_(result_tmp);
  return result;
}

// Frobenius or nuclear norms
Tensor linalg_cond(const Tensor& self, c10::string_view ord) {
  // 检查输入是否为方阵
  squareCheckInputs(self, ("linalg.cond(ord=" + std::string(ord) + ")").c_str());
  // 将 ord 转换为标准变体类型
  std::variant<Scalar, c10::string_view> ord_variant = ord;
  _linalg_cond_check_ord(ord_variant);

  // NumPy 不定义0x0矩阵的条件数，对于这样的输入返回0.0
  if (self.numel() == 0) {
    return _linalg_cond_empty_matrix(self, self.scalar_type());
  }

  // 如果 ord == "nuc"，计算核范数
  if (ord == "nuc") {
    // 计算奇异值
    auto singular_values = at::linalg_svdvals(self);
    // 直接使用核范数的数学定义计算结果
    return singular_values.sum(-1) * (singular_values.reciprocal().sum(-1));
  }

  // 调用辅助函数计算条件数
  return _linalg_cond_helper(self, std::move(ord_variant));
}
// 实现_linalg_cond_out变体，避免复制并直接使用已分配的存储空间
Tensor& linalg_cond_out(const Tensor& self, c10::string_view ord, Tensor& result) {
  // 检查结果张量与输入张量是否在同一设备上
  checkSameDevice("linalg.cond", result, self);
  // 转换输入张量的标量类型为实数类型
  ScalarType real_dtype = toRealValueType(self.scalar_type());
  // 检查结果张量的数据类型是否与输入张量的实数类型兼容
  checkLinalgCompatibleDtype("linalg.cond", result.scalar_type(), real_dtype);

  // 调用at::linalg_cond计算条件数的结果
  Tensor result_tmp = at::linalg_cond(self, ord);
  // 调整输出张量的大小以匹配计算结果
  at::native::resize_output(result, result_tmp.sizes());
  // 将计算结果复制到输出张量中
  result.copy_(result_tmp);
  // 返回结果张量的引用
  return result;
}

Tensor linalg_tensorinv(const Tensor& self, int64_t ind) {
  /*
  将问题简化为2D方阵求逆的问题。
  步骤1. 计算结果的形状和中间2D矩阵的形状。
  步骤2. 将`self`重塑为2D矩阵。
  步骤3. 求解2D矩阵的逆self.to_2D()
          没有快速方法可以确定矩阵是否可逆，
          所以在此阶段可能会由at::inverse引发错误。
          注意，对于CUDA，这会导致可能很慢的跨设备内存同步。
  步骤4. 重塑结果。
  */
  TORCH_CHECK(ind > 0, "Expected a strictly positive integer for 'ind', but got ", ind);

  // self[ind:] 的形状
  std::vector<c10::SymInt> shape_ind_end = self.sym_sizes().slice(ind).vec();
  // self[:ind] 的形状
  std::vector<c10::SymInt> shape_start_ind = self.sym_sizes().slice(0, ind).vec();

  // 计算 self[ind:] 的所有维度乘积
  c10::SymInt prod_ind_end = c10::multiply_integers(shape_ind_end.cbegin(), shape_ind_end.cend());
  // 计算 self[:ind] 的所有维度乘积
  c10::SymInt prod_start_ind = c10::multiply_integers(shape_start_ind.cbegin(), shape_start_ind.cend());

  // 检查是否可以将输入张量重塑为2D方阵
  TORCH_CHECK(prod_ind_end == prod_start_ind,
    "Expected self to satisfy the requirement prod(self.shape[ind:]) == prod(self.shape[:ind]), but got ",
    prod_ind_end, " != ", prod_start_ind);

  // 将 shape_ind_end 和 shape_start_ind 连接以形成结果的形状
  // self[ind:] + self[:ind]
  shape_ind_end.insert(shape_ind_end.cend(), shape_start_ind.cbegin(), shape_start_ind.cend());

  // 尝试求解重塑后的 self 的逆矩阵，如果失败则捕获错误
  auto [result, info] = at::linalg_inv_ex(self.reshape_symint({prod_ind_end, prod_ind_end}), /*check_errors=*/false);
  at::_linalg_check_errors(info, "inv", /*is_matrix*/true);

  // 返回重塑后的结果张量
  return result.reshape_symint(shape_ind_end);
}

// 实现_linalg_tensorinv_out变体，避免复制并直接使用已分配的存储空间
Tensor& linalg_tensorinv_out(const Tensor& self, int64_t ind, Tensor& result) {
  // 检查结果张量与输入张量是否在同一设备上
  checkSameDevice("tensorinv", result, self);
  // 检查结果张量的数据类型是否与输入张量兼容
  checkLinalgCompatibleDtype("tensorinv", result, self);

  // 调用at::linalg_tensorinv计算张量的逆
  Tensor result_tmp = at::linalg_tensorinv(self, ind);
  // 调整输出张量的大小以匹配计算结果
  at::native::resize_output(result, result_tmp.sizes());
  // 将计算结果复制到输出张量中
  result.copy_(result_tmp);
  // 返回结果张量的引用
  return result;
}
Tensor linalg_tensorsolve(const Tensor& self, const Tensor& other, OptionalIntArrayRef dims) {
  /*
  The idea is to reduce the problem to 2D matrix solve.
  Step 1. (optional) `self` is permuted with `dims` such that dimensions from `dims` are moved to the right.
  For example, if we have 4D input with the shape (1, 2, 3, 4) and dims=(0, 2),
  then the result of permutation would have the shape (2, 4, 1, 3).
  Step 2. reshape `self` to 2D matrix.
  Step 3. solve the matrix equation self.to_2D() @ result = other.to_1D()
  Step 4. reshape the result.
  */
  int64_t ndim = self.dim();
  Tensor self_ = self;

  // move dimensions of `self_` from `dims` to the end
  if (dims.has_value()) {
    DimVector dest_axes(dims.value().size());
    std::iota(dest_axes.begin(), dest_axes.end(), ndim - dest_axes.size());
    self_ = at::movedim(self_, dims.value(), dest_axes);
  }

  // result_shape is self_.sizes[-(an-other.dim):]
  // 获取结果张量的形状，以 self_ 和 other 的维度决定
  std::vector<c10::SymInt> result_shape = self_.sym_sizes().slice(other.dim(), ndim - other.dim()).vec();

  // 计算结果张量形状的乘积
  c10::SymInt result_product = c10::multiply_integers(result_shape.begin(), result_shape.end());
  c10::SymInt other_product = c10::multiply_integers(other.sym_sizes().begin(), other.sym_sizes().end());

  // 检查是否可以将 self 张量重塑为二维方阵
  TORCH_CHECK(result_product == other_product,
    "Expected self to satisfy the requirement prod(self.shape[other.ndim:]) == prod(self.shape[:other.ndim]), but got ",
    result_product, " != ", other_product);

  // 将 self_ 重塑为二维方阵
  self_ = self_.reshape_symint({result_product, result_product});

  // 通常情况下，`other` 将被展平，因为 at::linalg_solve 需要二维输入
  // 使用 linalg_solve 求解方程 self_ @ result = other
  Tensor result = at::linalg_solve(self_, other.flatten());

  // 将结果重塑为符合 result_shape 的形状
  return result.reshape_symint(result_shape);
}

Tensor& linalg_tensorsolve_out(const Tensor& self, const Tensor& other, OptionalIntArrayRef dims, Tensor& result) {
  // 检查输出张量 `result` 的设备是否与 `self` 相同
  checkSameDevice("tensorsolve", result, self);
  // 检查输出张量 `result` 的数据类型是否与 `self` 兼容
  checkLinalgCompatibleDtype("tensorsolve", result, self);

  // 调用 linalg_tensorsolve 获取解，并将结果存入 result_tmp
  Tensor result_tmp = at::linalg_tensorsolve(self, other, dims);

  // 调整输出张量 `result` 的大小以匹配 result_tmp 的形状
  at::native::resize_output(result, result_tmp.sizes());

  // 将 result_tmp 的内容复制到 result 中
  result.copy_(result_tmp);

  // 返回输出张量 result 的引用
  return result;
}

namespace {
struct KronImpl final {
public:
    explicit KronImpl(const Tensor& self, const Tensor& other) {
      // 计算两个张量的最大维度
      maxdim = std::max(self.dim(), other.dim());
      // 计算 self 需要填充的维度
      int64_t pad_self = maxdim - self.dim();
      // 计算 other 需要填充的维度
      int64_t pad_other = maxdim - other.dim();
      // 初始化 a_reshape 和 b_reshape 为两倍的 maxdim 大小
      a_reshape = c10::SmallVector<int64_t, 10>(2 * maxdim);
      b_reshape = c10::SmallVector<int64_t, 10>(2 * maxdim);
      // 初始化 result_reshape 为 maxdim 大小
      result_reshape = c10::SmallVector<int64_t, 10>(maxdim);
      // 遍历维度范围
      for (const auto i : c10::irange(maxdim)) {
        // 根据需要填充的维度设置 a_reshape 的每一对维度
        a_reshape[2 * i] = (i >= pad_self ? self.sizes()[i - pad_self] : 1);
        a_reshape[2 * i + 1] = 1;
        // 根据需要填充的维度设置 b_reshape 的每一对维度
        b_reshape[2 * i] = 1;
        b_reshape[2 * i + 1] = (i >= pad_other ? other.sizes()[i - pad_other] : 1);
        // 设置 result_reshape 的每个维度为 a_reshape 和 b_reshape 对应位置的乘积
        result_reshape[i] = a_reshape[2 * i] * b_reshape[2 * i + 1];
      }
      // 使用 _unsafe_view 创建 self_view 和 other_view
      self_view = at::_unsafe_view(self, a_reshape);
      other_view = at::_unsafe_view(other, b_reshape);
    }

    Tensor& kron_out(Tensor& result) const {
      // 检查 result 是否已定义
      TORCH_INTERNAL_ASSERT(result.defined(), "Cannot call kron_out with an undefined result tensor as the out argument. Please allocate a Tensor before calling kron_out with it.");

      // 创建 mul_shape 作为结果张量的形状
      c10::SmallVector<int64_t, 10> mul_shape(2 * maxdim);
      // 遍历维度范围
      for (const auto i : c10::irange(maxdim)) {
        // 设置 mul_shape 的每一对维度
        mul_shape[2 * i] = a_reshape[2 * i];
        mul_shape[2 * i + 1] = b_reshape[2 * i + 1];
      }
      // 调整 result 的形状为 result_reshape
      at::native::resize_output(result, result_reshape);
      // 使用 _unsafe_view 创建 result_mul
      auto result_mul = at::_unsafe_view(result, mul_shape);
      // 在 result_mul 上执行 self_view 和 other_view 的乘法操作
      at::mul_out(result_mul, self_view, other_view);

      // 返回结果张量 result
      return result;
    }

    Tensor kron() const {
      // 返回 self_view 和 other_view 的乘法结果，并使用 _unsafe_view 设置其形状为 result_reshape
      return at::_unsafe_view(at::mul(self_view, other_view), result_reshape);
    }
  private:
    // 最大维度
    int64_t maxdim;
    // self_view 和 other_view 分别表示输入张量的 _unsafe_view
    Tensor self_view;
    Tensor other_view;
    // 结果张量的形状
    c10::SmallVector<int64_t, 10> result_reshape;
    // 重塑后的 self 和 other 的形状
    c10::SmallVector<int64_t, 10> a_reshape;
    c10::SmallVector<int64_t, 10> b_reshape;
/*
  结束函数定义，返回空对象的封装
*/
};

/*
  结束函数定义，返回Tensors的封装
*/
}

/*
  计算两个张量之间的Kronecker积。
*/
Tensor& kron_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // 调用KronImpl类实现的kron_out方法，返回结果
  return KronImpl(self, other).kron_out(result);
}

/*
  计算两个张量之间的Kronecker积。
*/
Tensor kron(const Tensor& self, const Tensor& other) {
  // 调用KronImpl类实现的kron方法，返回结果
  return KronImpl(self, other).kron();
}

/*
  权重量化乘法的Gem命令
*/
DEFINE_DISPATCH(weight_to_int4pack_stub);

/*
  权重量化乘法的int4pack_mm_stub
*/
DEFINE_DISPATCH(int4pack_mm_stub);

/*
  权重量化乘法的int8pack_mm_stub
*/
DEFINE_DISPATCH(int8pack_mm_stub);

/*
  将输入的权重张量转换为int4pack的CPU形式
*/
Tensor _convert_weight_to_int4pack_cpu(
    const Tensor& in,
    int64_t innerKTiles) {

  TORCH_CHECK(in.dim() == 2,
      __func__, " : 期望权重是2维张量。");
  TORCH_CHECK(in.dtype() == at::kInt,
      __func__, " : 期望权重是kInt类型。");
  TORCH_CHECK(innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8,
      __func__, " : innerKTiles应为2、4或8，而得到的是 ", innerKTiles);

  auto weight = in.contiguous();
  auto N = weight.size(0);
  auto K = weight.size(1);

  // 为CPU创建虚拟形状。在动态操作符注册中，操作符需要每个设备有相同的输出形状。
  // 因此创建一个虚拟形状 {N / 8, K / (16 * innerKTiles), 32, innerKTiles / 2}
  constexpr int64_t kNTileSize = 8;
  constexpr int64_t kKTileSize = 16;
  auto nTiles = (N + kNTileSize - 1) / kNTileSize;

  TORCH_CHECK(N % 16 == 0,
      __func__, " : 期望N能被16整除");
  const int64_t kSuperKTileSize = kKTileSize * innerKTiles;
  TORCH_CHECK( K % kSuperKTileSize == 0,
      __func__, " : 期望K能被 ", kSuperKTileSize, " 整除");
  auto kSuperTiles = (K + kSuperKTileSize - 1) / kSuperKTileSize;

  // 创建一个空的张量，用于存储权重的int4pack形式
  auto weight_packed = at::empty(
      {nTiles, kSuperTiles, 32, innerKTiles / 2},
      at::TensorOptions().dtype(at::kInt));

  // 调用weight_to_int4pack_stub方法将权重打包为int4pack形式
  weight_to_int4pack_stub(kCPU, weight_packed, weight, N, K);
  return weight_packed;
}

/*
  对权重int4pack形式的矩阵乘法进行处理
*/
Tensor _weight_int4pack_mm_cpu(
    const Tensor& A,
    const Tensor& B,
    int64_t qGroupSize,
    # 定义函数，执行矩阵乘法操作
    const Tensor& qlinear(
        // 定义常量，指定矩阵块的大小为8
        const Tensor& A,
        const Tensor& B,
        int64_t qGroupSize,
        // 传入的张量，包含了缩放因子和零点偏移
        const Tensor& qScaleAndZeros) {
    
      // 获取矩阵 A 的行数 M、矩阵 B 的行数乘上常量 kNTileSize 的结果 N、矩阵 A 的列数 K
      auto M = A.size(0);
      auto N = B.size(0) * kNTileSize;
      auto K = A.size(1);
    
      // 检查矩阵 A 的数据类型，应为 kBFloat16、kHalf 或 kFloat 中的一种
      TORCH_CHECK(A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
          __func__, " : expect A to be either 32-bit or 16-bit float tensor.");
      // 检查矩阵 A 是否为连续存储
      TORCH_CHECK(A.is_contiguous(),
          __func__, " : expect A to be contiguous.");
      // 检查矩阵 A 是否为二维张量
      TORCH_CHECK(A.dim() == 2,
          __func__, " : expect A to be 2D tensor.");
    
      // 检查矩阵 B 的数据类型，应为 kInt 类型
      TORCH_CHECK(B.dtype() == kInt,
          __func__, " : expect B to be int32 tensor.");
      // 检查矩阵 B 是否为连续存储
      TORCH_CHECK(B.is_contiguous(),
          __func__, " : expect B to be contiguous.");
      // 检查矩阵 B 是否为四维张量
      TORCH_CHECK(B.dim() == 4,
          __func__, " : expect B to 4d tensor.");
    
      // 检查 qGroupSize 是否为 32、64、128 或 256 中的一个值
      TORCH_CHECK(qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128
          || qGroupSize == 256,
          __func__, ": expect qGroupSize to be 32, 64, 128 or 256, got ", qGroupSize);
    
      // 检查 qScaleAndZeros 的维度和尺寸是否符合预期，应为三维张量且第二维为 N，第三维为 2
      TORCH_CHECK(qScaleAndZeros.dim() == 3 && qScaleAndZeros.size(1) == N
          && qScaleAndZeros.size(2) == 2,
          __func__, ": expect qScaleAndZeros to be 3d tensor with sizes [:, ", N, ", 2]");
    
      // 创建一个空张量 C，尺寸为 M x N，使用与 A 相同的选项
      auto C = at::empty({M, N}, A.options());
      // 调用特定的 CPU 操作函数进行整数矩阵乘法
      int4pack_mm_stub(kCPU, C, A, B, qGroupSize, qScaleAndZeros, N, K);
    
      // 返回结果张量 C
      return C;
    }
// 定义一个名为 `_weight_int8pack_mm_cpu` 的函数，接受三个输入参数 A、B 和 scales，并返回一个 Tensor 类型的结果
Tensor _weight_int8pack_mm_cpu(
    const Tensor& A,                   // 输入参数 A，代表一个 Tensor 对象
    const Tensor& B,                   // 输入参数 B，代表一个 Tensor 对象
    const Tensor& scales) {            // 输入参数 scales，代表一个 Tensor 对象，用于缩放

  auto M = A.size(0);                  // 获取 A 的第一维度大小
  auto N = B.size(0);                  // 获取 B 的第一维度大小
  auto K = A.size(1);                  // 获取 A 的第二维度大小

  TORCH_CHECK(A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
      __func__, " : expect A to be either 32-bit or 16-bit float tensor.");  // 检查 A 的数据类型是否为指定类型之一
  TORCH_CHECK(A.is_contiguous(),
      __func__, " : expect A to be contiguous.");                            // 检查 A 是否是连续存储的
  TORCH_CHECK(A.dim() == 2,
      __func__, " : expect A to be 2D tensor.");                             // 检查 A 是否为二维张量

  TORCH_CHECK(B.dtype() == kChar,
      __func__, " : expect B to be int8 tensor.");                           // 检查 B 的数据类型是否为 int8
  TORCH_CHECK(B.is_contiguous(),
      __func__, " : expect B to be contiguous.");                            // 检查 B 是否是连续存储的
  TORCH_CHECK(B.size(1) == K,
      __func__, " : expect B.size(1) == ", K);                               // 检查 B 的第二维度大小是否等于 K

  TORCH_CHECK(scales.dim() == 1 && scales.size(0) == N,
      __func__, " : expect scales to be 1d tensor with size ", N);           // 检查 scales 是否为一维张量且大小为 N

  auto C = at::empty({M, N}, A.options());  // 创建一个与 A 具有相同类型的空 Tensor C，大小为 MxN
  int8pack_mm_stub(kCPU, C, A, B, scales);  // 调用 int8pack_mm_stub 函数，执行特定的矩阵乘法运算

  return C;                               // 返回计算结果 C
}

// 定义一个名为 `_int_mm_out_cpu` 的函数，接受三个输入参数 self、mat2 和 result，返回一个 Tensor 引用
Tensor& _int_mm_out_cpu(const Tensor& self, const Tensor& mat2, Tensor& result) {
  static constexpr c10::string_view func_name = "int_mm_out_cpu";  // 定义一个静态常量字符串 func_name

  TORCH_CHECK(self.dim() == 2, func_name, ": Expected self to be of dimension 2 but got ", self.dim());  // 检查 self 是否为二维张量
  TORCH_CHECK(mat2.dim() == 2, func_name, ": Expected mat2 to be of dimension 2 but got ", mat2.dim());  // 检查 mat2 是否为二维张量
  TORCH_CHECK(self.size(1) == mat2.size(0), func_name, ": self.size(1) needs to match mat2.size(0) but got ", self.size(1), " and ", mat2.size(0));  // 检查 self 的第二维度是否等于 mat2 的第一维度
  TORCH_CHECK(self.dtype() == at::kChar, func_name, ": Expected self dtype to be of type int8 but got ", self.dtype());  // 检查 self 的数据类型是否为 int8
  TORCH_CHECK(mat2.dtype() == at::kChar, func_name, ": Expected mat2 dtype to be of type int8 but got ", mat2.dtype());  // 检查 mat2 的数据类型是否为 int8
  TORCH_CHECK(result.dtype() == at::kInt, func_name, ": Expected result dtype to be of type kInt but got ", result.dtype());  // 检查 result 的数据类型是否为 int
  TORCH_CHECK(result.size(0) == self.size(0), func_name, ": Expected result.size(0) to be ", self.size(0), " but got ", result.size(0));  // 检查 result 的第一维度大小是否等于 self 的第一维度大小
  TORCH_CHECK(result.size(1) == mat2.size(1), func_name, ": Expected result.size(1) to be ", mat2.size(1), " but got ", result.size(1));  // 检查 result 的第二维度大小是否等于 mat2 的第二维度大小
  TORCH_CHECK(result.dim() == 2, func_name, ": Expected result to be of dimension 2 but got ", result.dim());  // 检查 result 是否为二维张量
  TORCH_CHECK(result.is_contiguous(), func_name, ": Expected result to be contiguous.");  // 检查 result 是否是连续存储的

  if (result.numel() == 0 || self.size(1) == 0) {  // 如果 result 的元素个数为 0 或者 self 的第二维度大小为 0
    return result.zero_();  // 将 result 中的元素全部置为 0
  }

  bool dispatched = false;  // 定义一个布尔变量 dispatched，初始化为 false
  if (at::globalContext().userEnabledMkldnn()) {  // 如果启用了 MKLDNN
    try {
      mkldnn_matmul_i8i8i32(self, mat2, result);  // 调用 MKLDNN 提供的整数矩阵乘法运算
      dispatched = true;  // 设置 dispatched 为 true，表示成功调度使用 MKLDNN
    } catch (const std::exception& e) {  // 捕获可能抛出的异常
      TORCH_WARN(func_name, " failed, switching to BLAS gemm: ", e.what());  // 发出警告，并切换到 BLAS gemm 运算
    }
  }
  if (!dispatched) {  // 如果未成功调度使用 MKLDNN
    auto a = reinterpret_cast<int8_t*>(self.data_ptr());     // 将 self 的数据指针转换为 int8 类型指针 a
    auto b = reinterpret_cast<int8_t*>(mat2.data_ptr());     // 将 mat2 的数据指针转换为 int8 类型指针 b
    auto c = reinterpret_cast<int32_t*>(result.data_ptr());  // 将 result 的数据指针转换为 int32 类型指针 c
    const int64_t m = result.size(0);                        // 获取 result 的第一维度大小 m
    const int64_t n = result.size(1);                        // 获取 result 的第二维度大小 n
    const int64_t k = self.size(1);                          // 获取 self 的第二维度大小 k
    // 计算矩阵 self 的步幅在第一个维度上的值
    const int64_t lda_0 = self.strides()[0];
    // 计算矩阵 self 的步幅在第二个维度上的值
    const int64_t lda_1 = self.strides()[1];
    // 计算矩阵 mat2 的步幅在第一个维度上的值
    const int64_t ldb_0 = mat2.strides()[0];
    // 计算矩阵 mat2 的步幅在第二个维度上的值
    const int64_t ldb_1 = mat2.strides()[1];
    // 计算结果矩阵 result 的步幅
    const int64_t ldc = result.strides()[0];

    // 使用并行化方法计算矩阵乘法
    parallel_for(0, m * n, 1, [&](int64_t start, int64_t end) {
      // 遍历计算范围内的每个元素
      for (const auto i : c10::irange(start, end)) {
        // 计算当前元素所在的行号和列号
        auto row = i / n;
        auto col = i % n;
        // 初始化结果矩阵中当前位置的值为 0
        c[row * ldc + col] = 0;
        // 遍历计算矩阵乘法的每一项
        for (const auto k : c10::irange(k)) {
          // 执行矩阵乘法的累加操作
          c[row * ldc + col] = c[row * ldc + col] +
              static_cast<int32_t>(a[row * lda_0 + k * lda_1]) *
                  static_cast<int32_t>(b[k * ldb_0 + col * ldb_1]);
        }
      }
    });
  }
  // 返回计算得到的结果矩阵 result
  return result;
} // 结束 namespace native

Tensor _int_mm_cpu(const Tensor& self, const Tensor& mat2) {
    // 创建一个新的 Tensor 对象作为结果，形状为 (self.size(0), mat2.size(1))，数据类型为整型
    Tensor result = at::empty({self.size(0), mat2.size(1)}, self.options().dtype(at::kInt));
    // 调用 _int_mm_out_cpu 函数，计算 self 和 mat2 的矩阵乘积，结果存储在 result 中
    return _int_mm_out_cpu(self, mat2, result);
}

} // 结束 namespace native
} // 结束 namespace at
```