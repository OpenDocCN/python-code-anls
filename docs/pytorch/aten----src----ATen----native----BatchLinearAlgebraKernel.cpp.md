# `.\pytorch\aten\src\ATen\native\BatchLinearAlgebraKernel.cpp`

```
/*
Defines TORCH_ASSERT_ONLY_METHOD_OPERATORS to enable specific method operators.
Includes necessary headers for ATen library functionalities.
*/
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cpu/zmath.h>

#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif

// Defines the namespace for native ATen operations
namespace at::native {

// Anonymous namespace for local functions and constants
namespace {

/*
  Computes the Cholesky decomposition of matrices stored in `input`.
  This is an in-place routine and the content of 'input' is overwritten with the result.

  Args:
  * `input` - [in] Input tensor for the Cholesky decomposition
              [out] Cholesky decomposition result
  * `info` -  [out] Tensor filled with LAPACK error codes,
                    positive values indicate that the matrix is not positive definite.
  * `upper` - controls whether the upper (true) or lower (false) triangular portion of `input` is used

  For further details, please see the LAPACK documentation for POTRF.
*/
template <typename scalar_t>
void apply_cholesky(const Tensor& input, const Tensor& info, bool upper) {
  // Check if LAPACK support is available; if not, throw an error
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.cholesky on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // Determine whether to use upper or lower triangle based on the 'upper' flag
  char uplo = upper ? 'U' : 'L';
  auto input_data = input.data_ptr<scalar_t>();
  auto info_data = info.data_ptr<int>();
  auto input_matrix_stride = matrixStride(input);
  auto batch_size = batchCount(input);
  auto n = input.size(-2);
  auto lda = std::max<int64_t>(1, n);

  // Iterate over batches and perform Cholesky decomposition using LAPACK
  for (const auto i : c10::irange(batch_size)) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    int* info_working_ptr = &info_data[i];
    lapackCholesky<scalar_t>(uplo, n, input_working_ptr, lda, info_working_ptr);
  }
#endif
}

// This is a type dispatching helper function for 'apply_cholesky'
void cholesky_kernel(const Tensor& input, const Tensor& infos, bool upper) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "cholesky_cpu", [&]{
    apply_cholesky<scalar_t>(input, infos, upper);
  });
}

/*
Copies the lower (or upper) triangle of the square matrix to the other half and conjugates it.
This operation is performed in-place.
*/
template <typename scalar_t>
void apply_reflect_conj_tri_single(scalar_t* self, int64_t n, int64_t stride, bool upper) {
  std::function<void(int64_t, int64_t)> loop = [](int64_t, int64_t){};
  // Depending on 'upper', either reflect and conjugate the upper or lower triangle
  if (upper) {
    loop = [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        for (int64_t j = i + 1; j < n; j++) {
          self[i * stride + j] = conj_impl(self[j * stride + i]);
        }
      }
    };
  } else {
    // 定义一个 lambda 函数 `loop`，用于执行对称矩阵的转置操作
    loop = [&](int64_t start, int64_t end) {
      // 外层循环遍历矩阵的行范围 [start, end)
      for (const auto i : c10::irange(start, end)) {
        // 内层循环遍历行 `i` 之前的列范围 [0, i)
        for (const auto j : c10::irange(i)) {
          // 将矩阵中位置 `(i, j)` 处的元素赋值为位置 `(j, i)` 处元素的共轭
          self[i * stride + j] = conj_impl(self[j * stride + i]);
        }
      }
    };
  }
  // 当矩阵维度 `n` 小于 256 时，使用简单的串行循环执行转置操作，因为 OpenMP 的并行化开销较大
  if (n < 256) {
    loop(0, n);
  } else {
    // 当矩阵维度 `n` 较大时，使用 OpenMP 并行化执行转置操作
    at::parallel_for(0, n, 0, loop);
  }
}

/*
计算对称（Hermitian）正定矩阵的逆矩阵，使用Cholesky分解。
这是一个原地操作，'input'的内容会被覆盖。
'infos'是一个int Tensor，包含批量输入中每个矩阵的错误码。
更多信息请参考LAPACK的POTRI例程文档。
*/
template <typename scalar_t>
void apply_cholesky_inverse(Tensor& input, Tensor& infos, bool upper) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "cholesky_inverse: 编译时未找到LAPACK库");
#else
  // 根据upper参数设置Cholesky分解中的上/下三角形式
  char uplo = upper ? 'U' : 'L';

  // 获取input的数据指针和infos的数据指针
  auto input_data = input.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();

  // 计算input矩阵的步长
  auto input_matrix_stride = matrixStride(input);
  // 获取批量大小
  auto batch_size = batchCount(input);
  // 获取矩阵维度n
  auto n = input.size(-2);
  // 设置LAPACK函数中的lda参数
  auto lda = std::max<int64_t>(1, n);

  // 对于每个批量中的矩阵进行操作
  for (const auto i : c10::irange(batch_size)) {
    // 获取当前矩阵的工作指针
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    // 获取当前矩阵的info指针
    int* info_working_ptr = &infos_data[i];

    // 调用lapackCholeskyInverse函数进行Cholesky逆操作
    lapackCholeskyInverse<scalar_t>(uplo, n, input_working_ptr, lda, info_working_ptr);
    // LAPACK只会写入矩阵的上/下三角部分，保留另一侧不变
    // 对矩阵进行反射共轭三角转换
    apply_reflect_conj_tri_single<scalar_t>(input_working_ptr, n, lda, upper);
  }
#endif
}

// 这是'apply_cholesky_inverse'的类型分发辅助函数
Tensor& cholesky_inverse_kernel_impl(Tensor& result, Tensor& infos, bool upper) {
  // 此函数在原地计算逆矩阵
  // result应以列优先顺序排列，并包含要求逆的矩阵
  // result的内容会被'apply_cholesky_inverse'覆盖
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "cholesky_inverse_out_cpu", [&]{
    apply_cholesky_inverse<scalar_t>(result, infos, upper);
  });
  return result;
}

/*
  计算n-by-n矩阵'input'的特征值和特征向量。
  这是一个原地操作，'input'、'values'、'vectors'的内容会被覆盖。
  'infos'是一个int Tensor，包含批量输入中每个矩阵的错误码。
  更多信息请参考LAPACK的GEEV例程文档。
*/
template <typename scalar_t>
void apply_linalg_eig(Tensor& values, Tensor& vectors, Tensor& input, Tensor& infos, bool compute_eigenvectors) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "在CPU张量上调用torch.linalg.eig需要编译支持LAPACK的PyTorch版本。请使用已编译LAPACK支持的PyTorch。");

    TORCH_CHECK(false, "在CPU张量上调用torch.linalg.eig需要编译支持LAPACK的PyTorch版本。请使用已编译LAPACK支持的PyTorch。");
#endif
}
#else
  // 使用typename获取标量值类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  // 设置求解特征向量的标志，'V'表示计算右特征向量，'N'表示不计算左特征向量
  char jobvr = compute_eigenvectors ? 'V' : 'N';
  // 仅计算右特征向量，因此左特征向量的标志为'N'
  char jobvl = 'N';
  // 获取输入张量的最后一个维度大小
  auto n = input.size(-1);
  // 确定矩阵的行列数，最小为1
  auto lda = std::max<int64_t>(1, n);
  // 计算批次大小
  auto batch_size = batchCount(input);
  // 获取输入张量的矩阵步幅
  auto input_matrix_stride = matrixStride(input);
  // 获取值张量的步幅（一般是最后一个维度的大小）
  auto values_stride = values.size(-1);
  // 获取输入张量的数据指针
  auto input_data = input.data_ptr<scalar_t>();
  // 获取值张量的数据指针
  auto values_data = values.data_ptr<scalar_t>();
  // 获取infos张量的数据指针
  auto infos_data = infos.data_ptr<int>();
  // 如果需要计算右特征向量，则获取右特征向量张量的数据指针，否则设为nullptr
  auto rvectors_data = compute_eigenvectors ? vectors.data_ptr<scalar_t>() : nullptr;
  // 仅计算右特征向量，左特征向量的数据指针设为nullptr
  scalar_t* lvectors_data = nullptr;
  // 计算右特征向量的列数，若计算则为lda，否则为1
  int64_t ldvr = compute_eigenvectors ? lda : 1;
  // 左特征向量的列数为1，因为仅计算右特征向量
  int64_t ldvl = 1;

  // 初始化rwork和rwork_data用于存储实部数据（复数类型需用到）
  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (input.is_complex()) {
    // 将输入张量的数据类型转换为实数类型，以便存储实部数据
    ScalarType real_dtype = toRealValueType(input.scalar_type());
    // 创建rwork张量以存储实部数据，长度为lda*2
    rwork = at::empty({lda * 2}, input.options().dtype(real_dtype));
    // 获取rwork张量的数据指针
    rwork_data = rwork.mutable_data_ptr<value_t>();
  }

  // 调用lapackEig函数一次以获取work数据的最佳大小
  scalar_t work_query;
  lapackEig<scalar_t, value_t>(jobvl, jobvr, n, input_data, lda, values_data,
    lvectors_data, ldvl, rvectors_data, ldvr, &work_query, -1, rwork_data, &infos_data[0]);

  // 计算work数据的最佳大小
  int lwork = std::max<int>(1, static_cast<int>(real_impl<scalar_t, value_t>(work_query)));
  // 创建work张量以存储work数据，长度为lwork
  Tensor work = at::empty({lwork}, input.dtype());
  // 获取work张量的数据指针
  auto work_data = work.mutable_data_ptr<scalar_t>();

  // 遍历批次大小，对每个批次的输入数据进行特征值计算
  for (const auto i : c10::irange(batch_size)) {
    // 获取当前批次的输入数据指针
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    // 获取当前批次的值数据指针
    scalar_t* values_working_ptr = &values_data[i * values_stride];
    // 若计算右特征向量，则获取当前批次的右特征向量数据指针，否则设为nullptr
    scalar_t* rvectors_working_ptr = compute_eigenvectors ? &rvectors_data[i * input_matrix_stride] : nullptr;
    // 获取当前批次的infos数据指针
    int* info_working_ptr = &infos_data[i];
    // 调用lapackEig函数进行特征值计算
    lapackEig<scalar_t, value_t>(jobvl, jobvr, n, input_working_ptr, lda, values_working_ptr,
      lvectors_data, ldvl, rvectors_working_ptr, ldvr, work_data, lwork, rwork_data, info_working_ptr);
  }
#endif
}

// 这是'apply_linalg_eig'的类型分派辅助函数
void linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  // 此函数在原地计算非对称特征分解
  // 张量应该采用批量列主内存格式
  // eigenvalues、eigenvectors和infos的内容将被'apply_linalg_eig'重写

  // apply_linalg_eig在提供的输入矩阵上原地修改，因此需要创建副本
  Tensor input_working_copy = at::empty(input.mT().sizes(), input.options());
  // 转置输入副本，使其具有Fortran连续的内存布局
  input_working_copy.transpose_(-2, -1);
  // 复制原始输入到工作副本
  input_working_copy.copy_(input);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_eig_out_cpu", [&]{
    // 调用apply_linalg_eig函数进行特征值计算
    apply_linalg_eig<scalar_t>(eigenvalues, eigenvectors, input_working_copy, infos, compute_eigenvectors);
  });
}
/*
  Computes eigenvalues and eigenvectors of the input tensor 'vectors' using LAPACK library functions.
  The computation modifies 'vectors' in-place, overwriting its content with eigenvectors.
  Eigenvalues are stored in 'values', which should be pre-allocated as an empty array.
  'infos' stores status information for potential error checks.
  'upper' determines if the upper or lower triangular part of the input matrix is used.
  'compute_eigenvectors' specifies whether eigenvectors should be computed.
  No error checks are performed, assuming all inputs are valid.
*/
template <typename scalar_t>
void apply_lapack_eigh(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
#if !AT_BUILD_WITH_LAPACK()
  // Check if LAPACK support is available; if not, raise an error.
  TORCH_CHECK(
      false,
      "Calling torch.linalg.eigh or eigvalsh on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  // Determine whether to compute eigenvectors ('V') or eigenvalues only ('N').
  char uplo = upper ? 'U' : 'L';  // 'U' for upper triangle, 'L' for lower triangle.
  char jobz = compute_eigenvectors ? 'V' : 'N';  // 'V' for computing eigenvectors, 'N' for eigenvalues only.

  auto n = vectors.size(-1);  // Size of the last dimension of 'vectors'.
  auto lda = std::max<int64_t>(1, n);  // Leading dimension of 'vectors'.
  auto batch_size = batchCount(vectors);  // Number of matrices in the batch.

  auto vectors_stride = matrixStride(vectors);  // Stride between matrices in 'vectors'.
  auto values_stride = values.size(-1);  // Stride of the last dimension in 'values'.

  auto vectors_data = vectors.data_ptr<scalar_t>();  // Pointer to 'vectors' data.
  auto values_data = values.data_ptr<value_t>();  // Pointer to 'values' data.
  auto infos_data = infos.data_ptr<int>();  // Pointer to 'infos' data.

  // Using 'int' for work size is consistent with LAPACK interface, but future versions may use more specific types.
  int lwork = -1;  // Initial size query for work array.
  int lrwork = -1;  // Initial size query for rwork array (real workspace).
  int liwork = -1;  // Initial size query for iwork array (integer workspace).
  scalar_t lwork_query;  // Query result for work array size.
  value_t rwork_query;  // Query result for rwork array size.
  int iwork_query;  // Query result for iwork array size.

  // Call LAPACK function to get optimal work array sizes.
  lapackSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_data, lda, values_data,
    &lwork_query, lwork, &rwork_query, lrwork, &iwork_query, liwork, infos_data);

  // Adjust sizes based on query results.
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(lwork_query));
  Tensor work = at::empty({lwork}, vectors.options());  // Allocate work array.
  auto work_data = work.mutable_data_ptr<scalar_t>();  // Pointer to work array data.

  liwork = std::max<int>(1, iwork_query);
  Tensor iwork = at::empty({liwork}, vectors.options().dtype(at::kInt));  // Allocate iwork array.
  auto iwork_data = iwork.mutable_data_ptr<int>();  // Pointer to iwork array data.

  Tensor rwork;  // Real workspace tensor.
  value_t* rwork_data = nullptr;  // Pointer to rwork array data.
  if (vectors.is_complex()) {
    lrwork = std::max<int>(1, rwork_query);
    rwork = at::empty({lrwork}, values.options());  // Allocate rwork array for complex case.
    rwork_data = rwork.mutable_data_ptr<value_t>();  // Pointer to rwork array data.
  }

  // Iterate over each matrix in the batch and call LAPACK function for each.
  for (const auto i : c10::irange(batch_size)) {
    scalar_t* vectors_working_ptr = &vectors_data[i * vectors_stride];  // Pointer to current matrix in 'vectors'.
    value_t* values_working_ptr = &values_data[i * values_stride];  // Pointer to current 'values' entry.
    int* info_working_ptr = &infos_data[i];  // Pointer to current 'infos' entry.
    lapackSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_working_ptr, lda, values_working_ptr,
      work_data, lwork, rwork_data, lrwork, iwork_data, liwork, info_working_ptr);
    // 调用 LAPACK 库中的特征值求解函数 lapackSyevd
    // 该函数执行特征值分解操作，计算给定矩阵的特征值和（可选）特征向量

    // 如果 LAPACK 函数返回非零值，表示计算中出现了错误
    // 直接返回，因为之后的计算已经没有意义了
    if (*info_working_ptr != 0) {
      return;
    }
}
#endif
}

// 这是用于 'apply_lapack_eigh' 的类型分发辅助函数
void linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  // 此函数计算对称/埃尔米特特征分解
  // 原地张量应该以批次列主内存格式存在
  // eigenvalues、eigenvectors 和 infos 的内容将被 'apply_lapack_eigh' 覆盖
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      eigenvectors.scalar_type(), "linalg_eigh_cpu", [&] {
        // 调用 LAPACK 的特征值分解函数 apply_lapack_eigh
        apply_lapack_eigh<scalar_t>(
            eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
      });
}

/*
  geqrf 函数计算存储在 'input' 中的矩阵的 QR 分解。
  但是，它并不直接产生 Q 矩阵，而是产生一系列初等反射器，
  这些反射器可以稍后组合以构造 Q — 例如使用 orgqr 或 ormqr 函数。

  Args:
  * `input` - [in] QR 分解的输入张量
              [out] 包含 QR 分解结果的张量，其中包括：
              i)   R 矩阵的对角线及其以上的元素。
              ii)  隐式定义 Q 的反射器方向。
              此张量的下对角线将被重写为结果
  * `tau` - [out] 将包含隐式定义 Q 的反射器的大小

  更多细节，请参阅 LAPACK 文档中关于 GEQRF 的说明。
*/
template <typename scalar_t>
static void apply_geqrf(const Tensor& input, const Tensor& tau) {
#if !AT_BUILD_WITH_LAPACK()
  // 如果没有使用 LAPACK 构建，抛出错误
  TORCH_CHECK(
      false,
      "Calling torch.geqrf on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // 使用类型别名来获取 scalar_t 类型的值类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  // 获取输入张量和 tau 张量的数据指针
  auto input_data = input.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  // 计算输入矩阵的步长和 tau 张量的步长
  auto input_matrix_stride = matrixStride(input);
  auto tau_stride = tau.size(-1);
  // 获取批次大小、矩阵的行数 m 和列数 n
  auto batch_size = batchCount(input);
  auto m = input.size(-2);
  auto n = input.size(-1);
  // 确定 lda 的值，至少为 1
  auto lda = std::max<int>(1, m);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 定义 LAPACK 函数返回的信息变量
  int info;
  // 运行一次，以获取最优工作空间大小
  // 因为我们处理具有相同维度的矩阵批次，将此操作放在循环之外可以节省工作空间查询和分配的次数
  int lwork = -1;
  // 声明用于存储工作空间大小的标量 wkopt
  scalar_t wkopt;
  // 调用 LAPACK 的 GEQRF 函数获取最优工作空间大小
  lapackGeqrf<scalar_t>(m, n, input_data, lda, tau_data, &wkopt, lwork, &info);
  // 断言确保 LAPACK 操作成功执行
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);

  // 如果 lwork 小于 'n'，则会打印警告信息:
  // Intel MKL ERROR: Parameter 7 was incorrect on entry to SGEQRF.
  // 确定实际需要的工作空间大小 lwork
  lwork = std::max<int>(std::max<int>(1, n), real_impl<scalar_t, value_t>(wkopt));
  // 使用 ATen 的函数创建工作空间张量 work
  Tensor work = at::empty({lwork}, input.options());

  // 对每个批次中的矩阵执行 QR 分解
  for (const auto i : c10::irange(batch_size)) {
    // 计算当前批次的输入矩阵和 tau 张量的工作指针
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // 调用 LAPACK 的 GEQRF 函数计算实际的 QR 分解和 tau
    lapackGeqrf<scalar_t>(m, n, input_working_ptr, lda, tau_working_ptr, work.data_ptr<scalar_t>(), lwork, &info);

    // LAPACK 的 info 只会报告第 i 个参数错误，因此不需要每次都检查它
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// 这是 'apply_geqrf' 的类型分发辅助函数
void geqrf_kernel(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_cpu", [&]{
    // 调用模板函数 apply_geqrf 来执行 GEQRF 分解
    apply_geqrf<scalar_t>(input, tau);
  });
}

/*
  orgqr 函数允许从一系列初等反射器重构正交（或单位）矩阵 Q，
  此类反射器由 geqrf 函数生成。

  Args:
  * `self` - 包含初等反射器方向的张量（对角线以下），将被重写为结果
  * `tau` - 包含初等反射器大小的张量

  更多细节，请参阅 LAPACK 文档中的 ORGQR 和 UNGQR。
*/
template <typename scalar_t>
inline void apply_orgqr(Tensor& self, const Tensor& tau) {
#if !AT_BUILD_WITH_LAPACK()
  // 如果没有编译 LAPACK 支持，则抛出错误
  TORCH_CHECK(false, "Calling torch.orgqr on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // 一些 LAPACK 实现可能对空矩阵的处理不佳：
  // 工作空间查询可能返回 lwork 为 0，这是不允许的（要求 lwork >= 1）
  // 在这种情况下我们不需要做任何计算，可以提前返回
  if (self.numel() == 0) {
    return;
  }

  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto tau_data = tau.const_data_ptr<scalar_t>();
  auto self_matrix_stride = matrixStride(self);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(self);
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = tau.size(-1);
  auto lda = std::max<int64_t>(1, m);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;

  // LAPACK's requirement
  // 断言矩阵的维度关系满足 LAPACK 的要求
  TORCH_INTERNAL_ASSERT(m >= n);
  TORCH_INTERNAL_ASSERT(n >= k);

  // 运行一次以获取最优工作空间大小。
  // 因为我们处理的是具有相同维度的矩阵批次，将此操作放在循环外可以节省 (batch_size - 1) 次工作空间查询和相同结果的工作空间分配与释放操作。
  int lwork = -1;
  scalar_t wkopt;
  // 调用 LAPACK 的 orgqr 函数获取最优工作空间大小
  lapackOrgqr<scalar_t>(m, n, k, self_data, lda, const_cast<scalar_t*>(tau_data), &wkopt, lwork, &info);
  // 断言调试模式下返回值为零，表示操作成功
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  // 计算实际所需的工作空间大小
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  // 分配工作空间
  Tensor work = at::empty({lwork}, self.options());

  for (const auto i : c10::irange(batch_size)) {
    // 指向当前处理批次中 self 的指针
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    // 指向当前处理批次中 tau 的指针
    const scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // 计算实际的 Q 矩阵
    // 调用 LAPACK 的 orgqr 函数计算 Q 矩阵
    lapackOrgqr<scalar_t>(m, n, k, self_working_ptr, lda, const_cast<scalar_t*>(tau_working_ptr), work.data_ptr<scalar_t>(), lwork, &info);

    // LAPACK 的 orgqr 函数只在参数错误时返回 info 值
    // 因此这里只需断言调试模式下返回值为零，表示操作成功
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_orgqr'
Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
  // 根据结果张量的数据类型，调度不同类型的 'apply_orgqr' 函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "orgqr_cpu", [&]{
    apply_orgqr<scalar_t>(result, tau);
  });
  return result;
}

/*
  Solves a least squares problem. That is minimizing ||B - A X||.

  Input args:
  * 'input' - Tensor containing batches of m-by-n matrix A.
  * 'other' - Tensor containing batches of max(m, n)-by-nrhs matrix B.
  * 'cond' - relative tolerance for determining rank of A.
  * 'driver' - the name of the LAPACK driver that is used to compute the solution.
  Output args (modified in-place):
  * 'solution' - Tensor to store the solution matrix X.
  * 'residuals' - Tensor to store values of ||B - A X||.
  * 'rank' - Tensor to store the rank of A.
  * 'singular_values' - Tensor to store the singular values of A.
  * 'infos' - Tensor to store error codes of linear algebra math library.

  For further details, please see the LAPACK documentation for GELS/GELSY/GELSS/GELSD routines.
*/
template <typename scalar_t>
void apply_lstsq(const Tensor& A, Tensor& B, Tensor& rank, Tensor& singular_values, Tensor& infos, double rcond, LapackLstsqDriverType driver_type) {
#if !AT_BUILD_WITH_LAPACK()
  // 如果 PyTorch 未使用 LAPACK 编译，则抛出错误信息
  TORCH_CHECK(
      false,
      "Calling torch.linalg.lstsq on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  using driver_t = at::native::LapackLstsqDriverType;

  // 定义 LAPACK 解法函数指针和其对应的类型
  auto lapack_func = lapackLstsq<driver_t::Gelsd, scalar_t, value_t>;
  static auto driver_type_to_func
    = std::unordered_map<driver_t, decltype(lapack_func)>({
    {driver_t::Gels, lapackLstsq<driver_t::Gels, scalar_t, value_t>},
    {driver_t::Gelsy, lapackLstsq<driver_t::Gelsy, scalar_t, value_t>},
    {driver_t::Gelsd, lapackLstsq<driver_t::Gelsd, scalar_t, value_t>},
    {driver_t::Gelss, lapackLstsq<driver_t::Gelss, scalar_t, value_t>}
  });
  lapack_func = driver_type_to_func[driver_type];

  char trans = 'N';

  // 获取 A 和 B 的数据指针及其尺寸信息
  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto nrhs = B.size(-1);
  auto lda = std::max<int64_t>(1, m);
  auto ldb = std::max<int64_t>(1, std::max(m, n));
  auto infos_data = infos.data_ptr<int>();

  // 只有 'gels' 驱动器不计算秩
  int rank_32;
  int64_t* rank_data;
  int64_t* rank_working_ptr = nullptr;
  if (driver_t::Gels != driver_type) {
    rank_data = rank.data_ptr<int64_t>();
    rank_working_ptr = rank_data;
  }

  // 'gelsd' 和 'gelss' 都是基于 SVD 的算法，可以获取奇异值
  value_t* s_data;
  value_t* s_working_ptr = nullptr;
  int64_t s_stride;
  if (driver_t::Gelsd == driver_type || driver_t::Gelss == driver_type) {
    s_data = singular_values.data_ptr<value_t>();
    s_working_ptr = s_data;
    s_stride = singular_values.stride(-1);
  }
    s_stride = singular_values.size(-1);
  }

  // 'jpvt' workspace array is used only for 'gelsy' which uses QR factorization with column pivoting
  Tensor jpvt;
  int* jpvt_data = nullptr;
  if (driver_t::Gelsy == driver_type) {
    jpvt = at::empty({std::max<int64_t>(1, n)}, A.options().dtype(at::kInt));
    jpvt_data = jpvt.mutable_data_ptr<int>();
  }

  // Run once the driver, first to get the optimal workspace size
  int lwork = -1; // default value to decide the opt size for workspace arrays
  scalar_t work_opt;
  value_t rwork_opt;
  int iwork_opt;
  lapack_func(trans, m, n, nrhs,
    A_data, lda,
    B_data, ldb,
    &work_opt, lwork,
    infos_data,
    jpvt_data,
    static_cast<value_t>(rcond),
    &rank_32,
    &rwork_opt,
    s_working_ptr,
    &iwork_opt);

  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(work_opt));
  Tensor work = at::empty({lwork}, A.options());
  scalar_t* work_data = work.mutable_data_ptr<scalar_t>();

  // 'rwork' only used for complex inputs and 'gelsy', 'gelsd' and 'gelss' drivers
  Tensor rwork;
  value_t* rwork_data;
  if (A.is_complex() && driver_t::Gels != driver_type) {
    int64_t rwork_len;
    switch (driver_type) {
      case driver_t::Gelsy:
        rwork_len = std::max<int64_t>(1, 2 * n);
        break;
      case driver_t::Gelss:
        rwork_len = std::max<int64_t>(1, 5 * std::min(m, n));
        break;
      // case driver_t::Gelsd:
      default:
        rwork_len = std::max<int64_t>(1, rwork_opt);
    }
    rwork = at::empty({rwork_len}, A.options().dtype(c10::toRealValueType(A.scalar_type())));
    rwork_data = rwork.mutable_data_ptr<value_t>();
  }

  // 'iwork' workspace array is relevant only for 'gelsd'
  Tensor iwork;
  int* iwork_data;
  if (driver_t::Gelsd == driver_type) {
    iwork = at::empty({std::max<int>(1, iwork_opt)}, A.options().dtype(at::kInt));
    iwork_data = iwork.mutable_data_ptr<int>();
  }

  // Perform batched matrix operations using LAPACK functions
  at::native::batch_iterator_with_broadcasting<scalar_t>(A, B,
    [&](scalar_t* A_working_ptr, scalar_t* B_working_ptr, int64_t A_linear_batch_idx) {
      // Prepare pointers and data for each batch iteration
      rank_working_ptr = rank_working_ptr ? &rank_data[A_linear_batch_idx] : nullptr;
      s_working_ptr = s_working_ptr ? &s_data[A_linear_batch_idx * s_stride] : nullptr;
      int* infos_working_ptr = &infos_data[A_linear_batch_idx];

      // Call LAPACK function for the current batch iteration
      lapack_func(trans, m, n, nrhs,
        A_working_ptr, lda,
        B_working_ptr, ldb,
        work_data, lwork,
        infos_working_ptr,
        jpvt_data,
        static_cast<value_t>(rcond),
        &rank_32,
        rwork_data,
        s_working_ptr,
        iwork_data);

      // Convert LAPACK output to int64_t for 'rank' Tensor
      if (rank_working_ptr) {
        *rank_working_ptr = static_cast<int64_t>(rank_32);
      }
    }
  );
#endif
}

// This is a type and driver dispatching helper function for 'apply_lstsq'
void lstsq_kernel(const Tensor& a, Tensor& b, Tensor& rank, Tensor& singular_values, Tensor& infos, double rcond, std::string driver_name) {

  // 定义静态映射，将字符串驱动程序名称映射到对应的枚举类型
  static auto driver_string_to_type = std::unordered_map<c10::string_view, LapackLstsqDriverType>({
    {"gels", at::native::LapackLstsqDriverType::Gels},
    {"gelsy", at::native::LapackLstsqDriverType::Gelsy},
    {"gelsd", at::native::LapackLstsqDriverType::Gelsd},
    {"gelss", at::native::LapackLstsqDriverType::Gelss}
  });
  // 根据给定的驱动程序名称获取对应的枚举类型
  auto driver_type = driver_string_to_type[driver_name];

  // 根据输入张量的类型分发到特定类型的求解函数 apply_lstsq
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "linalg_lstsq_cpu", [&]{
    apply_lstsq<scalar_t>(a, b, rank, singular_values, infos, rcond, driver_type);
  });
}

/*
  The ormqr function multiplies Q with another matrix from a sequence of
  elementary reflectors, such as is produced by the geqrf function.

  Args:
  * `input`     - Tensor with elementary reflectors below the diagonal,
                  encoding the matrix Q.
  * `tau`       - Tensor containing the magnitudes of the elementary
                  reflectors.
  * `other`     - [in] Tensor containing the matrix to be multiplied.
                  [out] result of the matrix multiplication with Q.
  * `left`      - bool, determining whether `other` is left- or right-multiplied with Q.
  * `transpose` - bool, determining whether to transpose (or conjugate transpose) Q before multiplying.

  For further details, please see the LAPACK documentation.
*/
template <typename scalar_t>
void apply_ormqr(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
#if !AT_BUILD_WITH_LAPACK()
  // 如果未使用 LAPACK 编译选项，则抛出错误信息
  TORCH_CHECK(false, "Calling torch.ormqr on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // 定义类型别名，用于确定标量类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  // 根据参数 left 和 transpose 确定侧面标识和转置标译方式
  char side = left ? 'L' : 'R';
  char trans = transpose ? (input.is_complex() ? 'C' : 'T') : 'N';

  // 获取输入数据、tau 数据和其他数据的指针
  auto input_data = input.const_data_ptr<scalar_t>();
  auto tau_data = tau.const_data_ptr<scalar_t>();
  auto other_data = other.data_ptr<scalar_t>();

  // 计算输入矩阵、其他矩阵和 tau 向量的步幅
  auto input_matrix_stride = matrixStride(input);
  auto other_matrix_stride = matrixStride(other);
  auto tau_stride = tau.size(-1);

  // 获取批处理大小和矩阵尺寸信息
  auto batch_size = batchCount(input);
  auto m = other.size(-2);
  auto n = other.size(-1);
  auto k = tau.size(-1);

  // 确定 lda 和 ldc 的值
  auto lda = std::max<int64_t>(1, left ? m : n);
  auto ldc = std::max<int64_t>(1, m);

  // 初始化 LAPACK 操作的返回信息
  int info = 0;

  // 断言 LAPACK 操作的前提条件
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY((left ? m : n) >= k);

  // 查询工作空间张量的最优尺寸
  int lwork = -1;
  scalar_t wkopt;
  // 调用 LAPACK 中的 ormqr 函数，用于计算工作空间尺寸
  lapackOrmqr<scalar_t>(side, trans, m, n, k, const_cast<scalar_t*>(input_data), lda, const_cast<scalar_t*>(tau_data), other_data, ldc, &wkopt, lwork, &info);
  // 断言 LAPACK 操作的返回信息
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);

  // 确定工作空间的实际尺寸
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  // 创建工作空间张量
  Tensor work = at::empty({lwork}, input.options());

  // 对每个批次执行 ormqr 计算
  for (const auto i : c10::irange(batch_size)) {
    // 获取当前批次的输入、其他和 tau 的工作指针
    const scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* other_working_ptr = &other_data[i * other_matrix_stride];
    const scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // 调用 LAPACK 中的 ormqr 函数，计算实际结果
    lapackOrmqr<scalar_t>(
        side, trans, m, n, k,
        const_cast<scalar_t*>(input_working_ptr), lda,
        const_cast<scalar_t*>(tau_working_ptr),
        other_working_ptr, ldc,
        work.data_ptr<scalar_t>(), lwork, &info);

    // 断言 LAPACK 操作的返回信息
    // lapackOrmqr 函数仅在参数错误时报告 info 不为零
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_ormqr'
// 为 'apply_ormqr' 提供类型分发的辅助函数
void ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  // 在浮点和复数类型上调度 'apply_ormqr' 函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "ormqr_cpu", [&]{
    apply_ormqr<scalar_t>(input, tau, other, left, transpose);
  });
}

/*
Solves the matrix equation op(A) X = B
X and B are n-by-nrhs matrices, A is a unit, or non-unit, upper or lower triangular matrix
and op(A) is one of op(A) = A or op(A) = A^T or op(A) = A^H.
This is an in-place routine, content of 'B' is overwritten.
'upper' controls the portion of input matrix to consider in computations,
'transpose' chooses op(A)
'unitriangular' if true then the diagonal elements of A are assumed to be 1
and the actual diagonal values are not used.
*/
// 解决矩阵方程 op(A) X = B
// X 和 B 是 n × nrhs 矩阵，A 是单位或非单位、上三角或下三角矩阵
// op(A) 是 op(A) = A 或 op(A) = A^T 或 op(A) = A^H 中的一个
// 这是一个原地操作，'B' 的内容会被覆盖
// 'upper' 控制在计算中要考虑的输入矩阵部分
// 'transpose' 选择 op(A)
// 'unitriangular' 如果为 true，则假定 A 的对角线元素为 1，实际对角线值不使用
template<typename scalar_t>
void apply_triangular_solve(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
#if !AT_BUILD_WITH_BLAS()
// 如果未使用 BLAS 构建 PyTorch，抛出错误信息，要求使用支持 BLAS 的 PyTorch 版本
  TORCH_CHECK(
      false,
      "Calling torch.triangular_solve on a CPU tensor requires compiling ",
      "PyTorch with BLAS. Please use PyTorch built with BLAS support.");
#else
// 否则，根据输入参数确定解的特性：上三角还是下三角，单位三角形式还是一般形式，以及左乘还是右乘
  char uplo = upper ? 'U' : 'L';  // 指定矩阵的上三角或下三角
  char diag = unitriangular ? 'U' : 'N';  // 指定是否是单位三角形式
  char side = left ? 'L' : 'R';  // 指定是左乘还是右乘
  const char trans = to_blas(transpose);  // 转置操作对应的 BLAS 字符表示

  auto A_data = A.const_data_ptr<scalar_t>();  // 获取 A 的数据指针
  auto B_data = B.data_ptr<scalar_t>();  // 获取 B 的数据指针
  auto A_mat_stride = matrixStride(A);  // 获取 A 的矩阵步幅
  auto B_mat_stride = matrixStride(B);  // 获取 B 的矩阵步幅
  auto batch_size = batchCount(A);  // 获取批次数目
  // 当 left = True 时，允许传递矩形的 A 和 B
  auto m = left ? A.size(-1) : B.size(-2);  // 获取 m 维度的大小
  auto n = B.size(-1);  // 获取 n 维度的大小
  auto lda = std::max<int64_t>(1, A.size(-2));  // 获取 A 的列步幅
  auto ldb = std::max<int64_t>(1, B.size(-2));  // 获取 B 的列步幅

  // 对每个批次执行解算
  for (const auto i : c10::irange(batch_size)) {
    const scalar_t* A_working_ptr = &A_data[i * A_mat_stride];  // 当前批次下的 A 数据指针
    scalar_t* B_working_ptr = &B_data[i * B_mat_stride];  // 当前批次下的 B 数据指针
    // 调用 BLAS 中的三角解算函数
    blasTriangularSolve<scalar_t>(side, uplo, trans, diag, m, n, const_cast<scalar_t*>(A_working_ptr), lda, B_working_ptr, ldb);
  }
#endif
}

void triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cpu", [&]{
    // 调用模板函数 apply_triangular_solve 进行三角解算
    apply_triangular_solve<scalar_t>(A, B, left, upper, transpose, unitriangular);
  });
}

template <typename scalar_t>
void apply_ldl_factor(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
#if !AT_BUILD_WITH_LAPACK()
// 如果未使用 LAPACK 构建 PyTorch，抛出错误信息，要求使用支持 LAPACK 的 PyTorch 版本
  TORCH_CHECK(
      false,
      "Calling torch.linalg.ldl_factor on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
// 否则，执行 LDL 分解的计算
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(A) > 0);  // 断言批次数大于 0
  auto batch_size = batchCount(A);  // 获取批次数目
  auto n = A.size(-2);  // 获取矩阵 A 的倒数第二维度大小
  auto leading_dim = A.stride(-1);  // 获取 A 的主维度步幅
  auto uplo = upper ? 'U' : 'L';  // 指定矩阵的上三角或下三角

  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;  // 获取 A 的步幅，用于多维情况
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;  // 获取 pivots 的步幅，用于多维情况

  auto a_data = A.data_ptr<scalar_t>();  // 获取 A 的数据指针
  auto pivots_data = pivots.data_ptr<int>();  // 获取 pivots 的数据指针
  auto info_data = info.data_ptr<int>();  // 获取 info 的数据指针

  // 选择正确的 LDL 分解函数
  auto ldl_func =
      hermitian ? lapackLdlHermitian<scalar_t> : lapackLdlSymmetric<scalar_t>;

  scalar_t wkopt;
  // 计算所需工作空间大小
  ldl_func(uplo, n, a_data, leading_dim, pivots_data, &wkopt, -1, info_data);
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  int lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  // 创建工作空间 Tensor
  Tensor work = at::empty({lwork}, A.dtype());
  auto work_data = work.mutable_data_ptr<scalar_t>();

  // 对每个批次执行 LDL 分解
  for (const auto i : c10::irange(batch_size)) {
    scalar_t* a_working_ptr = &a_data[i * a_stride];  // 当前批次下的 A 数据指针
    auto* pivots_working_ptr = &pivots_data[i * pivots_stride];  // 当前批次下的 pivots 数据指针
    auto* info_working_ptr = &info_data[i];  // 当前批次下的 info 数据指针
    ldl_func(
        uplo,
        n,
        a_working_ptr,
        leading_dim,
        pivots_working_ptr,
        work_data,
        lwork,
        info_working_ptr);
  }



// 调用名为 ldl_func 的函数，传递以下参数：
// - uplo: 指定矩阵是上三角还是下三角的标志
// - n: 矩阵的维度
// - a_working_ptr: 指向矩阵数据的指针
// - leading_dim: 矩阵每列之间的跨度
// - pivots_working_ptr: 指向主元数据的指针
// - work_data: 执行计算过程中使用的工作区数据
// - lwork: 工作区数据的长度
// - info_working_ptr: 指向存储操作结果信息的指针
// 结束当前函数的定义
#endif
}

void ldl_factor_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  // 使用AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES宏，根据LD张量的数据类型选择适当的模板函数进行LDL分解
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_factor_kernel_cpu", [&] {
        // 调用模板函数apply_ldl_factor，对LD张量进行LDL分解操作
        apply_ldl_factor<scalar_t>(LD, pivots, info, upper, hermitian);
      });
}

template <typename scalar_t>
void apply_ldl_solve(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
#if !AT_BUILD_WITH_LAPACK()
  // 检查是否使用了AT_BUILD_WITH_LAPACK宏编译，如果未使用，则抛出错误
  TORCH_CHECK(
      false,
      "Calling torch.linalg.ldl_factor on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // 调试模式下，断言A张量的批次数量大于0
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(A) > 0);
  // 调试模式下，断言扩展后的pivots张量批次数量大于0
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(pivots.unsqueeze(-1)) > 0);
  auto batch_size = batchCount(B);
  auto n = A.size(-2);
  auto nrhs = B.size(-1);
  auto lda = A.stride(-1);
  auto ldb = B.stride(-1);
  auto uplo = upper ? 'U' : 'L';

  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  auto b_stride = B.dim() > 2 ? B.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  // 获取A张量数据指针和B张量数据指针
  auto a_data = A.const_data_ptr<scalar_t>();
  auto b_data = B.data_ptr<scalar_t>();
  // 将pivots张量转换为整型张量pivots_
  auto pivots_ = pivots.to(kInt);
  // 获取pivots_张量数据指针
  auto pivots_data = pivots_.const_data_ptr<int>();

  // 根据hermitian标志选择LDL解算函数
  auto ldl_solve_func = hermitian ? lapackLdlSolveHermitian<scalar_t>
                                  : lapackLdlSolveSymmetric<scalar_t>;

  int info = 0;
  // 对每个批次执行LDL解算
  for (const auto i : c10::irange(batch_size)) {
    // 获取当前批次的A张量和B张量的工作指针
    const scalar_t* a_working_ptr = &a_data[i * a_stride];
    scalar_t* b_working_ptr = &b_data[i * b_stride];
    // 获取当前批次的pivots_张量的工作指针
    const auto* pivots_working_ptr = &pivots_data[i * pivots_stride];
    // 调用LDL解算函数ldl_solve_func进行LDL求解
    ldl_solve_func(
        uplo,
        n,
        nrhs,
        const_cast<scalar_t*>(a_working_ptr),
        lda,
        const_cast<int*>(pivots_working_ptr),
        b_working_ptr,
        ldb,
        &info);
  }
  // 调试模式下，断言info为0，表示LDL求解过程中无错误
  TORCH_INTERNAL_ASSERT(info == 0);
#endif
}

void ldl_solve_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& result,
    bool upper,
    bool hermitian) {
  // 使用AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES宏，根据LD张量的数据类型选择适当的模板函数进行LDL求解
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_solve_kernel_cpu", [&] {
        // 调用模板函数apply_ldl_solve，对LD张量进行LDL求解操作
        apply_ldl_solve<scalar_t>(LD, pivots, result, upper, hermitian);
      });
}

/*
  Computes the LU decomposition of a m×n matrix or batch of matrices in 'input' tensor.
  This is an in-place routine, content of 'input', 'pivots', and 'infos' is overwritten.

  Args:
  * `input` - [in] the input matrix for LU decomposition
              [out] the LU decomposition
  * `pivots` - [out] the pivot indices
  * `infos` - [out] error codes, positive values indicate singular matrices
  * `compute_pivots` - should always be true (can be false only for CUDA)

  For further details, please see the LAPACK documentation for GETRF.
*/
template <typename scalar_t>
void apply_lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
#if !AT_BUILD_WITH_LAPACK()
  // 如果未使用 LAPACK 编译选项，抛出错误信息
  TORCH_CHECK(
      false,
      "Calling torch.linalg.lu_factor on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // 检查是否需要计算枢轴
  TORCH_CHECK(compute_pivots, "linalg.lu_factor: LU without pivoting is not implemented on the CPU");

  // 获取输入张量的数据指针和维度信息
  auto input_data = input.data_ptr<scalar_t>();
  // 获取枢轴数据指针
  auto pivots_data = pivots.data_ptr<int>();
  // 获取信息数据指针
  auto infos_data = infos.data_ptr<int>();
  // 计算输入矩阵的步幅
  auto input_matrix_stride = matrixStride(input);
  // 获取枢轴的步幅
  auto pivots_stride = pivots.size(-1);
  // 获取批处理大小
  auto batch_size = batchCount(input);
  // 获取矩阵的行数和列数
  auto m = input.size(-2);
  auto n = input.size(-1);
  // 计算主导维度，取 m 和 1 之间的较大值
  auto leading_dimension = std::max<int64_t>(1, m);

  // 定义并执行并行循环
  const auto loop = [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      // 指向当前批次的输入数据起始位置
      scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
      // 指向当前批次的枢轴数据起始位置
      int* pivots_working_ptr = &pivots_data[i * pivots_stride];
      // 指向当前批次的信息数据起始位置
      int* infos_working_ptr = &infos_data[i];
      // 调用 LAPACK 的 LU 分解函数
      lapackLu<scalar_t>(
          m,
          n,
          input_working_ptr,
          leading_dimension,
          pivots_working_ptr,
          infos_working_ptr);
    }
  };

  // 避免溢出，计算矩阵秩的浮点数表示
  float matrix_rank = float(std::min(m, n));
  // 设置每个线程的块大小，基于经验启发式方法
  int64_t chunk_size_per_thread = int64_t(
      std::min(1.0, 3200.0 / (matrix_rank * matrix_rank * matrix_rank)));
  // 计算并行执行的粒度
  int64_t grain_size = chunk_size_per_thread * at::get_num_threads();
  // 调用 PyTorch 的并行处理函数
  at::parallel_for(0, batch_size, grain_size, loop);
#endif
}

// 此函数用于类型分发，调用对应的 LU 分解函数
void lu_factor_kernel(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "lu_cpu", [&]{
    // 调用模板函数 apply_lu_factor 实现 LU 分解
    apply_lu_factor<scalar_t>(input, pivots, infos, compute_pivots);
  });
}

/*
  解方程 A X = B
  X 和 B 是 n×nrhs 的矩阵，A 使用 LU 分解表示。
  这是一个原地操作，输入的 `b` 内容将被覆盖。

  参数:
  * `b` -  [in] 右手边的矩阵 B
           [out] 解的矩阵 X
  * `lu` - [in] 矩阵 A 的 LU 分解 (参见 at::linalg_lu_factor)
  * `pivots` - [in] 枢轴索引 (参见 at::linalg_lu_factor)

  更多细节，请参阅 LAPACK 文档中的 GETRS。
*/
template <typename scalar_t>
void apply_lu_solve(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
#if !AT_BUILD_WITH_LAPACK()
  // 如果未使用 LAPACK 编译选项，抛出错误信息
  TORCH_CHECK(
      false,
      "Calling linalg.lu_solve on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // 获取 B 的数据指针，类型为 scalar_t
  auto b_data = B.data_ptr<scalar_t>();
  // 获取 LU 的常量数据指针，类型为 scalar_t
  auto lu_data = LU.const_data_ptr<scalar_t>();
  // 根据 transpose 转置情况设置 BLAS 的 transpose 标志
  const auto trans = to_blas(transpose);
  // 获取 pivots 的常量数据指针，类型为 int
  auto pivots_data = pivots.const_data_ptr<int>();
  // 获取 B 的矩阵步幅
  auto b_stride = matrixStride(B);
  // 获取 LU 的步幅，如果 LU 的维度大于 2，则取其指定维度的步幅，否则为 0
  auto lu_stride = LU.dim() > 2 ? LU.stride(-3) : 0;
  // 获取 pivots 的步幅，如果 pivots 的维度大于 1，则取其指定维度的步幅，否则为 0
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;
  // 获取批处理的大小
  auto batch_size = batchCount(B);

  // 获取 LU 的倒数第二维大小
  auto n = LU.size(-2);
  // 获取 B 的最后一维大小
  auto nrhs = B.size(-1);
  // 计算主维度的最大值
  auto leading_dimension = std::max<int64_t>(1, n);

  // 初始化 info 为 0
  int info = 0;

  // LU 和 pivots 可以广播到 B，构建帮助索引张量 lu_index 来线性索引 LU 和 pivots
  IntArrayRef lu_batch_shape(LU.sizes().data(), LU.dim() - 2);
  IntArrayRef b_batch_shape(B.sizes().data(), B.dim() - 2);
  BroadcastLinearIndices lu_index(
      batchCount(LU), lu_batch_shape, b_batch_shape);

  // 遍历每个批次
  for (const auto i : c10::irange(batch_size)) {
    // 计算 lu_index 的索引值
    int64_t lu_index_i = lu_index(i);
    // 获取当前批次的 B 数据指针起始位置
    scalar_t* b_working_ptr = &b_data[i * b_stride];
    // 获取当前批次的 LU 数据指针起始位置
    const scalar_t* lu_working_ptr = &lu_data[lu_index_i * lu_stride];
    // 获取当前批次的 pivots 数据指针起始位置
    const int* pivots_working_ptr = &pivots_data[lu_index_i * pivots_stride];

    // 调用 lapackLuSolve 函数解决方程
    lapackLuSolve<scalar_t>(trans, n, nrhs, const_cast<scalar_t*>(lu_working_ptr), leading_dimension, const_cast<int*>(pivots_working_ptr),
                            b_working_ptr, leading_dimension, &info);

    // lapackLuSolve 返回的 info 只有在参数错误时才会报错，所以不需要每次都检查
    // 使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 确保 info 为 0
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// 这是 'apply_lu_solve' 的类型调度辅助函数
void lu_solve_kernel(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // 对于 CPU 版本，确保 pivots 在正确范围内，以避免 Lapack 写入不相关的内存
  TORCH_CHECK(pivots.gt(0).all().item<bool>(),
              "Pivots given to lu_solve must all be greater or equal to 1. "
              "Did you properly pass the result of lu_factor?");
  TORCH_CHECK(pivots.le(LU.size(-2)).all().item<bool>(),
              "Pivots given to lu_solve must all be smaller or equal to LU.size(-2). "
              "Did you properly pass the result of lu_factor?");

  // 对 LU 的标量类型进行分发，调用 apply_lu_solve 函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "linalg.lu_solve_cpu", [&]{
    apply_lu_solve<scalar_t>(LU, pivots, B, trans);
  });
}

template <typename scalar_t>
static void apply_svd(const Tensor& A,
                      const bool full_matrices,
                      const bool compute_uv,
                      const Tensor& U,
                      const Tensor& S,
                      const Tensor& Vh,
                      const Tensor& info) {
#if !AT_BUILD_WITH_LAPACK()
  // 如果没有编译 LAPACK 库，则报错
  TORCH_CHECK(false, "svd: LAPACK library not found in compilation");
#else
  // 定义 value_t 类型，该类型为 scalar_t 的标量值类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  // 获取 A 张量的数据指针
  const auto A_data = A.data_ptr<scalar_t>();
  // 获取 U 张量的数据指针，如果 compute_uv 为 true；否则为 nullptr
  const auto U_data = compute_uv ? U.data_ptr<scalar_t>() : nullptr;
  // 获取 S 张量的数据指针，使用 value_t 类型
  const auto S_data = S.data_ptr<value_t>();
  // 获取 info 张量的数据指针
  const auto info_data = info.data_ptr<int>();
  // 获取 Vh 张量的数据指针，如果 compute_uv 为 true；否则为 nullptr
  const auto Vh_data = compute_uv ? Vh.data_ptr<scalar_t>() : nullptr;
  // 获取 A 张量的矩阵步幅
  const auto A_stride = matrixStride(A);
  // 获取 S 张量的最后一个维度大小
  const auto S_stride = S.size(-1);
  // 获取 U 张量的矩阵步幅，如果 compute_uv 为 true；否则为 1
  const auto U_stride = compute_uv ? matrixStride(U) : 1;
  // 获取 Vh 张量的矩阵步幅，如果 compute_uv 为 true；否则为 1
  const auto Vh_stride = compute_uv ? matrixStride(Vh) : 1;
  // 获取 A 张量的批次大小
  const auto batchsize = batchCount(A);
  // 设置 jobz 字符，根据 compute_uv 和 full_matrices 的值
  const char jobz = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';

  // 获取 A 张量的倒数第二个维度大小
  const auto m = A.size(-2);
  // 获取 A 张量的最后一个维度大小
  const auto n = A.size(-1);
  // 获取 A 张量的最后一个维度步幅
  const auto lda = A.stride(-1);
  // 获取 U 张量的最后一个维度步幅，如果 compute_uv 为 false；则为 1
  const auto ldu = compute_uv ? U.stride(-1) : 1;
  // 获取 Vh 张量的最后一个维度步幅，如果 compute_uv 为 false；则为 1
  const auto ldvh = compute_uv ? Vh.stride(-1) : 1;

  // 创建用于工作的整数数组 iwork，大小为 8 * min(m, n)
  auto iwork = std::vector<int>(8 * std::min(m, n));
  // 获取 iwork 数组的指针
  auto* const iwork_data = iwork.data();

  // rwork 用于复数分解，如果 A 张量为复数，则调整其大小
  auto rwork = std::vector<value_t>{};
  if (A.is_complex()) {
    rwork.resize(std::max(computeLRWorkDim(jobz, m, n), int64_t{1}));
  }
  // 获取 rwork 数组的指针
  auto* const rwork_data = rwork.data();

  // 查询 SVD 算法的最优 lwork 大小
  int lwork = -1;
  {
    // 声明 wkopt 变量
    scalar_t wkopt;
    // 调用 lapackSvd 函数查询 lwork 大小
    lapackSvd<scalar_t, value_t>(jobz, m, n, A_data, lda, S_data, U_data, ldu, Vh_data, ldvh, &wkopt, lwork, rwork_data, iwork_data, info_data);
    // 更新 lwork 为查询得到的最优大小
    lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  }
  // 创建用于工作的数组 work，大小为 lwork
  auto work = std::vector<scalar_t>(lwork);
  // 获取 work 数组的指针
  auto* const work_data = work.data();

  // 遍历批次中的每个元素
  for (const auto i : c10::irange(batchsize)) {
    // 获取当前批次中 A 张量的工作指针
    auto* const A_working_ptr = &A_data[i * A_stride];
    // 获取当前批次中 S 张量的工作指针
    auto* const S_working_ptr = &S_data[i * S_stride];
    // 获取当前批次中 U 张量的工作指针，如果 compute_uv 为 false，则为 nullptr
    auto* const U_working_ptr = compute_uv ? &U_data[i * U_stride] : nullptr;
    // 获取当前批次中 Vh 张量的工作指针，如果 compute_uv 为 false，则为 nullptr
    auto* const Vh_working_ptr = compute_uv ? &Vh_data[i * Vh_stride] : nullptr;

    // 调用 lapackSvd 函数进行奇异值分解计算
    lapackSvd<scalar_t, value_t>(jobz, m, n, A_working_ptr, lda,
                        S_working_ptr, U_working_ptr, ldu, Vh_working_ptr, ldvh, work_data, lwork, rwork_data, iwork_data, info_data + i);
  }
#endif
}
void unpack_pivots_cpu_kernel(TensorIterator& iter, const int64_t dim_size, const int64_t max_pivot) {
  // 如果迭代器元素数量为零或维度大小为零，则直接返回
  if (iter.numel() == 0 || dim_size == 0) {
    return;
  }
  // 定义 lambda 函数 loop，处理数据和步长，执行迭代
  auto loop = [&](char* const* const  data, const int64_t* const strides, const int64_t nelems) {
    // 获取置换数组指针和枢轴数组指针
    auto* perm_ptr = data[0];
    const auto* pivots_ptr = data[1];

    // 遍历元素范围内的每个元素
    for (C10_UNUSED const auto elem : c10::irange(nelems)) {
      // 将 perm_ptr 解释为 int64_t* 类型的数据
      const auto perm_data = reinterpret_cast<int64_t*>(perm_ptr);
      // 将 pivots_ptr 解释为 const int32_t* 类型的数据
      const auto pivots_data = reinterpret_cast<const int32_t*>(pivots_ptr);

      // 遍历维度大小
      for (const auto i : c10::irange(dim_size)) {
        // 计算新索引，注意减一因为 pivots_data 中的索引从1开始
        auto new_idx = pivots_data[i] - 1;
        // 检查新索引是否在有效范围内
        TORCH_CHECK(new_idx >= 0 && new_idx < max_pivot,
                    "pivots passed to lu_unpack must be between 1 and LU.size(-2) inclusive."
                    "Did you properly pass the result of lu_factor?");
        // 交换 perm_data[i] 和 perm_data[new_idx] 的值
        std::swap(
          perm_data[i],
          perm_data[new_idx]
        );
      }

      // 更新指针位置
      perm_ptr += strides[0];
      pivots_ptr += strides[1];
    }
  };

  // 使用迭代器的 for_each 方法执行 loop lambda 函数
  iter.for_each(loop);
}
} // 匿名命名空间结束

// 注册各种指令集下的 cholesky_stub 函数调用
REGISTER_ARCH_DISPATCH(cholesky_stub, DEFAULT, &cholesky_kernel);
REGISTER_AVX512_DISPATCH(cholesky_stub, &cholesky_kernel);
REGISTER_AVX2_DISPATCH(cholesky_stub, &cholesky_kernel);
REGISTER_VSX_DISPATCH(cholesky_stub, &cholesky_kernel);
REGISTER_ZVECTOR_DISPATCH(cholesky_stub, &cholesky_kernel);

// 注册各种指令集下的 cholesky_inverse_stub 函数调用
REGISTER_ARCH_DISPATCH(cholesky_inverse_stub, DEFAULT, &cholesky_inverse_kernel_impl);
REGISTER_AVX512_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);
REGISTER_AVX2_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);
REGISTER_VSX_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);
REGISTER_ZVECTOR_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);

// 注册各种指令集下的 linalg_eig_stub 函数调用
REGISTER_ARCH_DISPATCH(linalg_eig_stub, DEFAULT, &linalg_eig_kernel);
REGISTER_AVX512_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);
REGISTER_AVX2_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);
REGISTER_VSX_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);
REGISTER_ZVECTOR_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);

// 注册各种指令集下的 linalg_eigh_stub 函数调用
REGISTER_ARCH_DISPATCH(linalg_eigh_stub, DEFAULT, &linalg_eigh_kernel);
REGISTER_AVX512_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);
REGISTER_AVX2_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);
REGISTER_VSX_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);
REGISTER_ZVECTOR_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);

// 注册各种指令集下的 geqrf_stub 函数调用
REGISTER_ARCH_DISPATCH(geqrf_stub, DEFAULT, &geqrf_kernel);
REGISTER_AVX512_DISPATCH(geqrf_stub, &geqrf_kernel);
REGISTER_AVX2_DISPATCH(geqrf_stub, &geqrf_kernel);
REGISTER_VSX_DISPATCH(geqrf_stub, &geqrf_kernel);
REGISTER_ZVECTOR_DISPATCH(geqrf_stub, &geqrf_kernel);

// 注册各种指令集下的 orgqr_stub 函数调用
REGISTER_ARCH_DISPATCH(orgqr_stub, DEFAULT, &orgqr_kernel_impl);
REGISTER_AVX512_DISPATCH(orgqr_stub, &orgqr_kernel_impl);
REGISTER_AVX2_DISPATCH(orgqr_stub, &orgqr_kernel_impl);
# 注册 orgqr_stub 到 orgqr_kernel_impl 的分发机制
REGISTER_VSX_DISPATCH(orgqr_stub, &orgqr_kernel_impl);
# 注册 orgqr_stub 到 orgqr_kernel_impl 的分发机制

# 注册 ormqr_stub 到 ormqr_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(ormqr_stub, DEFAULT, &ormqr_kernel);
# 注册 ormqr_stub 到 ormqr_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(ormqr_stub, &ormqr_kernel);
# 注册 ormqr_stub 到 ormqr_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(ormqr_stub, &ormqr_kernel);
# 注册 ormqr_stub 到 ormqr_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(ormqr_stub, &ormqr_kernel);
# 注册 ormqr_stub 到 ormqr_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(ormqr_stub, &ormqr_kernel);

# 注册 lstsq_stub 到 lstsq_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(lstsq_stub, DEFAULT, &lstsq_kernel);
# 注册 lstsq_stub 到 lstsq_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(lstsq_stub, &lstsq_kernel);
# 注册 lstsq_stub 到 lstsq_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(lstsq_stub, &lstsq_kernel);
# 注册 lstsq_stub 到 lstsq_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(lstsq_stub, &lstsq_kernel);
# 注册 lstsq_stub 到 lstsq_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(lstsq_stub, &lstsq_kernel);

# 注册 triangular_solve_stub 到 triangular_solve_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(triangular_solve_stub, DEFAULT, &triangular_solve_kernel);
# 注册 triangular_solve_stub 到 triangular_solve_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);
# 注册 triangular_solve_stub 到 triangular_solve_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);
# 注册 triangular_solve_stub 到 triangular_solve_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);
# 注册 triangular_solve_stub 到 triangular_solve_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);

# 注册 lu_factor_stub 到 lu_factor_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(lu_factor_stub, DEFAULT, &lu_factor_kernel);
# 注册 lu_factor_stub 到 lu_factor_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(lu_factor_stub, &lu_factor_kernel);
# 注册 lu_factor_stub 到 lu_factor_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(lu_factor_stub, &lu_factor_kernel);
# 注册 lu_factor_stub 到 lu_factor_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(lu_factor_stub, &lu_factor_kernel);
# 注册 lu_factor_stub 到 lu_factor_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(lu_factor_stub, &lu_factor_kernel);

# 注册 ldl_factor_stub 到 ldl_factor_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(ldl_factor_stub, DEFAULT, &ldl_factor_kernel);
# 注册 ldl_factor_stub 到 ldl_factor_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(ldl_factor_stub, &ldl_factor_kernel);
# 注册 ldl_factor_stub 到 ldl_factor_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(ldl_factor_stub, &ldl_factor_kernel);
# 注册 ldl_factor_stub 到 ldl_factor_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(ldl_factor_stub, &ldl_factor_kernel);
# 注册 ldl_factor_stub 到 ldl_factor_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(ldl_factor_stub, &ldl_factor_kernel);

# 注册 ldl_solve_stub 到 ldl_solve_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(ldl_solve_stub, DEFAULT, &ldl_solve_kernel);
# 注册 ldl_solve_stub 到 ldl_solve_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(ldl_solve_stub, &ldl_solve_kernel);
# 注册 ldl_solve_stub 到 ldl_solve_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(ldl_solve_stub, &ldl_solve_kernel);
# 注册 ldl_solve_stub 到 ldl_solve_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(ldl_solve_stub, &ldl_solve_kernel);
# 注册 ldl_solve_stub 到 ldl_solve_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(ldl_solve_stub, &ldl_solve_kernel);

# 注册 lu_solve_stub 到 lu_solve_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(lu_solve_stub, DEFAULT, &lu_solve_kernel);
# 注册 lu_solve_stub 到 lu_solve_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(lu_solve_stub, &lu_solve_kernel);
# 注册 lu_solve_stub 到 lu_solve_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(lu_solve_stub, &lu_solve_kernel);
# 注册 lu_solve_stub 到 lu_solve_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(lu_solve_stub, &lu_solve_kernel);
# 注册 lu_solve_stub 到 lu_solve_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(lu_solve_stub, &lu_solve_kernel);

# 注册 svd_stub 到 svd_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(svd_stub, DEFAULT, &svd_kernel);
# 注册 svd_stub 到 svd_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(svd_stub, &svd_kernel);
# 注册 svd_stub 到 svd_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(svd_stub, &svd_kernel);
# 注册 svd_stub 到 svd_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(svd_stub, &svd_kernel);
# 注册 svd_stub 到 svd_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(svd_stub, &svd_kernel);

# 注册 unpack_pivots_stub 到 unpack_pivots_cpu_kernel 的架构分发，默认版本
REGISTER_ARCH_DISPATCH(unpack_pivots_stub, DEFAULT, &unpack_pivots_cpu_kernel);
# 注册 unpack_pivots_stub 到 unpack_pivots_cpu_kernel 的 AVX512 版本的分发机制
REGISTER_AVX512_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel);
# 注册 unpack_pivots_stub 到 unpack_pivots_cpu_kernel 的 AVX2 版本的分发机制
REGISTER_AVX2_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel);
# 注册 unpack_pivots_stub 到 unpack_pivots_cpu_kernel 的 VSX 版本的分发机制
REGISTER_VSX_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel);
# 注册 unpack_pivots_stub 到 unpack_pivots_cpu_kernel 的 ZVECTOR 版本的分发机制
REGISTER_ZVECTOR_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel);
```