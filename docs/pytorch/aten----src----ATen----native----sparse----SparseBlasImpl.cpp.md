# `.\pytorch\aten\src\ATen\native\sparse\SparseBlasImpl.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/mkl/Sparse.h>
#include <ATen/native/mkl/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlasImpl.h>
#include <ATen/SparseCsrTensorUtils.h>

// Required for checking whether Triton kernels are available
#include <ATen/core/dispatch/Dispatcher.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#else
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#endif

#if !AT_USE_MKL_SPARSE()
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#endif

// 这段代码定义了在实现稀疏张量的操作时使用的内部函数和变量

namespace at::native::sparse::impl {

// 实现部分的匿名命名空间开始
namespace {

// 函数：判断一个整数是否为2的幂次方
bool operands_support_triton_mm_kernel(const Tensor& compressed, const Tensor& strided) {
  const auto is_power_of_2 = [](int64_t v) -> bool {
    return !(v & (v - 1));
  };
  
  // 判断是否支持使用 Triton MM 内核的条件
  return AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(compressed.layout(), "operands_support_triton_mm_kernel", [&] { return false; },
     [&] {
       const auto blocksize = at::sparse_csr::getBlockSize(compressed);
       
       // 检查数据类型和块大小，以确定是否可以使用 Triton 内核
       return ((strided.scalar_type() == ScalarType::Half
                || strided.scalar_type() == ScalarType::BFloat16
                || strided.scalar_type() == ScalarType::Float)
               && compressed.scalar_type() == strided.scalar_type()
               && is_power_of_2(blocksize[0]) && is_power_of_2(blocksize[1])
               && (blocksize[0] >= 16) && (blocksize[1] >= 16)
               && strided.size(-1) % blocksize[0] == 0);
     });
}

} // 匿名命名空间结束

} // at::native::sparse::impl 命名空间结束
// 获取压缩矩阵的布局
const auto compressed_layout = compressed.layout();
// 将压缩矩阵的布局转换为字符串表示
const auto compressed_layout_str = at::sparse_csr::layoutToString(compressed_layout);

// 设备限制检查：要求所有输入张量在相同的设备上
TORCH_CHECK(compressed.device() == strided.device()
    && compressed.device() == result.device(),
    "spmm_out(): all input arguments are expected to be on the same device.");

// 布局限制检查：仅支持 CSR 和 BSR 格式的稀疏矩阵作为压缩矩阵的输入
TORCH_CHECK(compressed_layout == kSparseCsr || compressed_layout == kSparseBsr,
    "spmm(", compressed_layout_str, ", Strided): only Csr and Bsr formats are supported for the sparse argument.");

// 布局限制检查：要求输出结果张量的布局为 Strided
TORCH_CHECK(result.layout() == kStrided,
    "spmm_out(): out argument is expected to be strided.");

// 数据类型限制检查：要求压缩矩阵和 Strided 矩阵具有相同的数据类型
TORCH_CHECK(compressed.scalar_type() == strided.scalar_type(),
    "spmm(", compressed_layout_str, ", Strided): arguments expected to have the same dtype.");

// 维度限制检查：要求压缩矩阵是二维的
TORCH_CHECK(compressed.dim() == 2,
    "spmm(", compressed_layout_str, ", Strided): sparse arguments which are not 2D are not supported.");

// 维度限制检查：要求 Strided 矩阵至少是二维的
TORCH_CHECK(strided.dim() >= 2,
    "spmm(", compressed_layout_str, ", Strided): expects strided inputs to be at least 2D.");

// 矩阵乘积大小兼容性检查：要求 Strided 矩阵的倒数第二维大小等于压缩矩阵的第二维大小
const auto m = compressed.sizes()[0];
const auto k = compressed.sizes()[1];
const auto n = strided.size(-1);
TORCH_CHECK(strided.size(-2) == k,
    "spmm(", compressed_layout_str, "Strided): argument sizes are not compatible for matrix multiplication. ",
    "Got ", compressed_layout_str, ".sizes(-1) == ", k, " is not equal to ",
    "Strided.sizes(-2) == ", strided.size(-2), ".");

// 预期结果大小检查：检查输出结果张量的大小是否符合预期
auto result_expected_size = at::DimVector(strided.sizes().slice(0, strided.dim() - 2));
result_expected_size.push_back(m);
result_expected_size.push_back(n);
TORCH_CHECK(result.sizes() == result_expected_size,
    "spmm_out(): out argument has wrong size. ",
    "Expected (", result_expected_size, ") but got (", result.sizes(), ").");

auto values = compressed.values();

using Blocksize = std::array<int64_t, 2>;
// 在下面的注释中将其称为 (b0, b1)
Blocksize blocksize = {1, 1};
// 如果压缩矩阵的布局是 BSR，则根据其值张量的大小设置块大小
if (compressed_layout == kSparseBsr) {
  blocksize = {values.size(-2), values.size(-1)};
}

// 不支持 ROCM 的稳定支持（暂时）
#ifndef USE_ROCM

// 如果压缩矩阵和 Strided 矩阵支持 Triton 矩阵乘积内核
if (operands_support_triton_mm_kernel(compressed, strided)) {
  // 获取 Triton 内核的模式
  const auto triton_schema = c10::Dispatcher::singleton()
    .findSchema({"triton::_triton_bsr_dense_mm_out", ""});
  // 如果找到 Triton 内核
  if (triton_schema.has_value()) {
    // 获取 Triton 内核的函数指针，用于 SparseCsrCUDA 分发键
    const auto triton_kernel = triton_schema.value().typed<Tensor&(const Tensor&, const Tensor&, Tensor&)>();
    if (triton_kernel.hasKernelForDispatchKey(c10::DispatchKey::SparseCsrCUDA)) {
      // 调用 Triton 内核进行矩阵乘积运算
      return triton_kernel.call(compressed, strided, result);
    }
    } /* 否则，表示模式未定义或者键未被覆盖，
         因此跳过并执行下面的代码。 */
  }
#endif

// 定义一个 Lambda 函数 tile_tensor，用于将输入张量按照给定的块大小块化
// 如果压缩布局为稀疏 CSR，则简单地在最后两个维度上添加两个维度
const auto tile_tensor = [compressed_layout](
    const Tensor& t, Blocksize blocksize) -> Tensor {
  if (compressed_layout == kSparseCsr) {
    return t.unsqueeze(-1).unsqueeze_(-1);
  }
  else {
    // 计算在指定块大小下，张量 t 在倒数第二和倒数第一维度上的块数
    const auto size_neg_2_blocked = t.size(-2) / blocksize[0];
    const auto size_neg_1_blocked = t.size(-1) / blocksize[1];
    // 构造块化后的尺寸向量
    auto tiled_sizes = at::DimVector(t.sizes().slice(0, t.dim() - 2));
    tiled_sizes.push_back(size_neg_2_blocked);
    tiled_sizes.push_back(blocksize[0]);
    tiled_sizes.push_back(size_neg_1_blocked);
    tiled_sizes.push_back(blocksize[1]);
    // 将张量 t 重新形状为块化后的尺寸，并交换倒数第三和倒数第二维度
    return t.reshape(tiled_sizes).transpose(-3, -2);
  }
};

// 注意稀疏值的形状为 (..., b0, b1)，这意味着 strided 输入必须能够 "块化" 到 (..., b1, x)
// 其中 x >= 1，并且所有形状必须是块矩阵乘积兼容的
// 矩阵乘积的结果形状为 (..., b0, x)，因此结果也必须能够 "块化" 到 (..., b0, x)
//
// 这些观察导致以下限制：
// 1. strided.size(-2) 必须能被 b1 整除。
// 2. result.size(-2) 必须能被 b0 整除。
// 3. strided.size(-1) 和 result.size(-1) 必须能被 x 整除。
//
// 限制 1 和 2 是显然满足的。关于限制 3：
// 出于性能考虑，应选择尽可能大的 x，因为最后一个维度很可能是连续的。
// 因此，这个值恰好是 x = strided.size(-1)，因为 strided.size(-1) == result.size(-1)
//
// 查看上面的注释。这里我们的 x 就是 outer_blocksize。
const auto outer_blocksize = n;

// 定义 strided 输入的块大小
Blocksize strided_blocksize = {blocksize[1], outer_blocksize};
// 对 strided 输入进行块化处理
const auto strided_tiled = tile_tensor(strided, strided_blocksize);

// 左侧参数为 (..., b0, b1)，右侧为 (..., b1, x)。
// 这自然意味着结果应该能够 "块化" 到 (..., b0, x)。
// 定义结果的块大小
Blocksize result_blocksize = {blocksize[0], outer_blocksize};
// 对结果进行块化处理
auto result_tiled = tile_tensor(result, result_blocksize);

// 如果压缩布局为稀疏 CSR，则对 values 张量在最后两个维度上添加两个维度
if (compressed_layout == kSparseCsr) {
  values.unsqueeze_(-1).unsqueeze_(-1);
}

// 从压缩输入中获取压缩和普通索引
auto [compressed_indices, plain_indices] = at::sparse_csr::getCompressedPlainIndices(compressed);

// 选择与稀疏输入的块列交集的 strided 输入的块行
auto strided_tiled_selected_rows = strided_tiled.index_select(-4, plain_indices);

// 如果输出是半精度或 BF16，提升为 float 类型以提高精度
const auto mm_dtype = (result.scalar_type() == kHalf || result.scalar_type() == kBFloat16)
  // 根据 result 的数据类型确定 mm_dtype 或 kFloat
  ? kFloat : result.scalar_type();
  
  // 现在我们知道哪些块行与哪些块列相交，
  // 我们可以在块之间执行矩阵乘积。
  // 注意：当 result.scalar_type() == mm_dtype 时，.to 是一个无操作。
  const auto pairwise_block_mm = values.unsqueeze(-3).to(mm_dtype)
    .matmul(strided_tiled_selected_rows.to(mm_dtype));

  // 将块之间的矩阵乘积存储在 pairwise_block_mm 中，
  // 只需对共享相同行索引的所有块乘积进行求和。
  // 由于使用了高级索引方法进行了归约步骤，压缩索引应转换为 COO 格式。
  const auto compressed_indices_coo = at::_convert_indices_from_csr_to_coo(
      compressed_indices,
      plain_indices,
      compressed_indices.scalar_type() == kInt).select(0, 0);

  // 归约步骤。
  // 如果 result 不是半精度或 bfloat16，则一切都在原地完成。
  if (result.scalar_type() == mm_dtype) {
    // 清零并对共享相同行索引的块求和。
    result_tiled.zero_();
    result_tiled.index_add_(
        /*dim=*/-4,
        /*index=*/compressed_indices_coo,
        /*source=*/pairwise_block_mm);
  }
  // 否则累加到一个缓冲区，然后复制到 result 中。
  else {
    // 不需要清零，块之间的求和进入缓冲区，
    // 然后复制到 result 中。
    auto promoted_result_tiled = at::zeros(
        result_tiled.sizes(),
        result_tiled.options().dtype(mm_dtype));
    promoted_result_tiled.index_add_(
        /*dim=*/-4,
        /*index=*/compressed_indices_coo,
        /*source=*/pairwise_block_mm);
    result_tiled.copy_(promoted_result_tiled);
  }

  // 返回计算结果
  return result;
}

Tensor& _compressed_row_strided_addmm_out(
    const Tensor& self,  // 输入：自身张量
    const Tensor& mat1,  // 输入：矩阵1
    const Tensor& mat2,  // 输入：矩阵2
    const Scalar& beta,  // 输入：标量 beta
    const Scalar& alpha,  // 输入：标量 alpha
    Tensor& result) {  // 输出：结果张量引用

// No stable support for ROCM in Triton yet.
#ifndef USE_ROCM
  // 检查是否支持 Triton 内核并且 Triton 模式已启用
  if (operands_support_triton_mm_kernel(mat1, mat2)) {
    // 查找 Triton 内核的函数模式
    const auto triton_schema = c10::Dispatcher::singleton()
      .findSchema({"triton::_triton_bsr_dense_addmm_out", ""});
    if (triton_schema.has_value()) {
      // 获取 Triton 内核的函数指针
      const auto triton_kernel = triton_schema.value().typed<Tensor&(const Tensor&, const Tensor&, const Tensor&, const Scalar&, const Scalar&, Tensor&)>();
      // 检查是否为 SparseCsrCUDA 分发键提供了内核
      if (triton_kernel.hasKernelForDispatchKey(c10::DispatchKey::SparseCsrCUDA)) {
        try {
          // 调用 Triton 内核进行计算
          return triton_kernel.call(self, mat1, mat2, beta, alpha, result);
        } catch (std::runtime_error& e) {
          const std::string msg = e.what();
          // 捕获并处理运行时错误
          if (msg != std::string("Unable to cast NotImplemented to Tensor")) {
            throw std::runtime_error(msg);
          }
        } /* else triton_kernel 返回了 NotImplemented，继续
             使用下面的通用方法 */
      }
    } /* else 如果模式未定义或键未覆盖，则跳过并执行下面的代码。*/
  }
#endif

  auto alpha_val = alpha.toComplexDouble();
  auto beta_val = beta.toComplexDouble();
  // 如果结果不是与自身相同，则可以作为输出参数传递给 mm。
  if (!result.is_same(self)) {
    // 使用优化过的 mm 方法计算结果
    _compressed_row_strided_mm_out(mat1, mat2, result);
    // 处理 alpha
    if (alpha_val != 1.) {
      result.mul_(alpha);
    }
    // 处理 beta
    if (beta_val != 0.) {
      if (beta_val == 1.) {
        result.add_(self);
      } else {
        result.add_(self.mul(beta));
      }
    }
  }
  // 否则，需要为 mm 分配外部内存，如果 beta != 0。
  else {
    // 处理 beta
    if (beta_val != 0.) {
      if (beta_val != 1.) {
        result.mul_(beta);
      }
      // 创建一个与 result 类型相同的空张量 mm
      auto mm = at::empty_like(result);
      // 使用优化过的 mm 方法计算 mm
      _compressed_row_strided_mm_out(mat1, mat2, mm);
      // 处理 alpha
      if (alpha_val != 1.) {
        mm.mul_(alpha);
      }
      result.add_(mm);
    }
    else {
      // 使用优化过的 mm 方法计算结果
      _compressed_row_strided_mm_out(mat1, mat2, result);
      // 处理 alpha
      if (alpha_val != 1.) {
        result.mul_(alpha);
      }
    }
  }

  return result;  // 返回计算后的结果张量
}

namespace cpu {
#if !AT_USE_MKL_SPARSE()
namespace {
template<typename scalar_t, typename idx_t>
void addmv_sparse_csr(
    const scalar_t* mat_values,  // 矩阵值数组指针
    const idx_t* crow_index,  // 行索引数组指针
    const idx_t* col_index,  // 列索引数组指针
    const int64_t mat_rows,  // 矩阵行数
    const scalar_t* vec,  // 向量数组指针
    const size_t vec_stride,  // 向量步长
    const scalar_t alpha,  // 标量 alpha
    const scalar_t beta,  // 标量 beta
    scalar_t* result,  // 结果数组指针
    const size_t result_stride) {  // 结果步长

  // 使用并行处理对稀疏 CSR 矩阵向量乘法进行计算
  at::parallel_for(0, mat_rows, 0, [&](int64_t rstart, int64_t rend) {

    // 并行处理范围内的每一行
    for (auto r = rstart; r < rend; ++r) {
      // 计算行起始和结束位置
      auto start = crow_index[r];
      auto end = crow_index[r + 1];
      // 初始化结果为零
      scalar_t sum = 0;
      // 对当前行的每个非零元素执行加权累加
      for (auto j = start; j < end; ++j) {
        sum += mat_values[j] * vec[col_index[j] * vec_stride];
      }
      // 将 alpha 乘以总和，加上 beta 乘以当前结果
      result[r * result_stride] = alpha * sum + beta * result[r * result_stride];
    }
}
} // namespace anonymous
#endif // !AT_USE_MKL_SPARSE()
} // namespace cpu


这段代码包含了稀疏矩阵-向量乘法的实现，通过注释详细解释了每个函数和条件分支的作用和功能。
    // 对于每一行 row，从 rstart 到 rend 进行迭代
    for(const auto row: c10::irange(rstart, rend)) {
      // 初始化累加器 acc，并置为 0
      scalar_t acc(0);
      // 对于当前行中从 crow_index[row] 到 crow_index[row + 1] 的每一个索引 idx 进行迭代
      for(const auto idx: c10::irange(crow_index[row], crow_index[row + 1])) {
        // 将矩阵 mat_values 中的值与向量 vec 中的对应元素相乘，累加到 acc 中
        acc += mat_values[idx] * vec[col_index[idx] * vec_stride];
      }
      // 将累加器 acc 乘以 alpha，并加上 result[row * result_stride] 乘以 beta，存入 result 数组的对应位置
      result[row * result_stride] = acc * alpha + result[row * result_stride] * beta;
    }
  });
/*
  函数定义：稀疏矩阵-稠密向量乘法，计算 y <- alpha*op(A)*x + beta*y

  Args:
  * `mat` - 存储稀疏矩阵 A 的 Tensor，大小为 m x n。
  * `vec` - 存储大小为 n 的稠密向量 x 的 Tensor。
  * `result` - [in] 存储大小为 m 的稠密向量 y 的 Tensor。
               [out] 计算结果的 Tensor。

  注意：这里使用了 AT_USE_MKL_SPARSE 宏来决定稀疏矩阵的布局。
*/

#if !AT_USE_MKL_SPARSE()
// 检查稀疏矩阵的布局是否为 kSparseBsr 或 kSparseCsr，否则抛出异常
TORCH_CHECK(mat.layout() == kSparseBsr || mat.layout() == kSparseCsr, "Unexpected layout", mat.layout());

// 如果 beta 转换为复数后为 0，则执行以下代码块
if (beta.toComplexDouble() == 0.) {
    result.zero_();

# 将结果张量 `result` 的所有元素置零。


  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "addmv_out_sparse_csr_impl_reference", [&] {

# 使用模板元函数 `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES` 针对结果张量 `result` 的数据类型进行分发，处理浮点数和复数类型。Lambda 函数 `[&]` 捕获外部变量。


        if (mat.crow_indices().scalar_type() == kLong) {

# 检查稀疏矩阵 `mat` 的行索引是否为长整型（64位整数类型）。


          addmv_out_sparse_csr<scalar_t, int64_t>(mat, vec, beta, alpha, result);

# 调用模板函数 `addmv_out_sparse_csr` 处理稀疏矩阵向量乘法，使用类型 `scalar_t` 和 `int64_t` 进行计算。


        } else {

# 如果稀疏矩阵 `mat` 的行索引不是长整型，则执行以下代码块。


          addmv_out_sparse_csr<scalar_t, int32_t>(mat, vec, beta, alpha, result);
        }
      });

# 调用模板函数 `addmv_out_sparse_csr` 处理稀疏矩阵向量乘法，使用类型 `scalar_t` 和 `int32_t` 进行计算。Lambda 表达式 `&` 结束并执行闭包。
/*
  调用由 MKL 提供的稀疏矩阵向量加法操作，将结果写入给定的稀疏 CSR 张量。
  若未启用 MKL 支持，则抛出错误信息并终止程序。

  Args:
  * `mat` - CSR 存储的稀疏矩阵。
  * `vec` - 存储稀疏矩阵的向量。
  * `beta` - 矩阵乘法结果的缩放因子。
  * `alpha` - 矩阵和向量乘法的缩放因子。
  * `result` - [in] CSR 存储的稀疏矩阵。
               [out] 运算的结果。
*/
void addmv_out_sparse_csr(
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  sparse::impl::mkl::addmv_out_sparse_csr(mat, vec, beta, alpha, result);
#endif
}

/*
  计算两个稀疏矩阵的和，定义为：
  result <- mat1 + alpha * mat2

  Args:
  * `mat1` - CSR 存储的稀疏 m x n 矩阵。
  * `mat2` - CSR 存储的稀疏 m x n 矩阵。
  * `alpha` - 矩阵 mat2 的缩放因子。
  * `result` - [in] CSR 存储的稀疏 m x n 矩阵。
               [out] 运算的结果。
*/
void add_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling add on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  sparse::impl::mkl::add_out_sparse_csr(mat1, mat2, alpha, result);
#endif
}

/*
  解稀疏 CSR 矩阵的三角线性方程组，并将结果写入给定的稀疏 CSR 张量。

  Args:
  * `A` - CSR 存储的稀疏矩阵。
  * `B` - CSR 存储的稀疏矩阵。
  * `X` - [in] 存储稀疏矩阵的解。
          [out] 运算的结果。
  * `upper` - 是否使用上三角矩阵。
  * `transpose` - 是否使用矩阵的转置。
  * `unitriangular` - 是否使用单位上三角矩阵。
*/
void triangular_solve_out_sparse_csr(
    const Tensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling triangular_solve on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  // 断言输入的稀疏矩阵 A 的布局为 CSR 或 BSR
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.layout() == kSparseCsr || A.layout() == kSparseBsr);
  // 调用 MKL 提供的稀疏矩阵三角线性方程组求解操作
  sparse::impl::mkl::triangular_solve_out_sparse_csr(A, B, X, upper, transpose, unitriangular);
#endif
}
```