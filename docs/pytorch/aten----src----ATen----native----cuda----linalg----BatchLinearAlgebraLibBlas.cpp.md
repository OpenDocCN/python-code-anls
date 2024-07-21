# `.\pytorch\aten\src\ATen\native\cuda\linalg\BatchLinearAlgebraLibBlas.cpp`

```
// Note [BatchLinearAlgebraLib split implementation files]
//
// There are two files that implement the interfaces found in
// BatchLinearAlgebraLib.h
// - BatchLinearAlgebraLib.cpp
// - BatchLinearAlgebraLibBlas.cpp (this file)
//
// In order to support the ROCm build target, the use of cublas and
// cusolver APIs needed to be split into separate source files to
// accommodate the hipify step of the ROCm build process.
//
// To create this current file, the original file
// BatchLinearAlgebraLib.cpp was copied to
// BatchLinearAlgebraLibBlas.cpp, then any functions that used cusolver
// APIs were removed. Similarly, in the original file
// BatchLinearAlgebraLib.cpp, any use of cublas APIs was removed.
// The net result is a split of the BatchLinearAlgebraLib
// implementation files. The original file BatchLinearAlgebraLib.cpp
// contains the full, original git history for both files.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/cuda/linalg/CUDASolver.h>
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// Helper function to convert TransposeType to cublasOperation_t
static cublasOperation_t to_cublas(TransposeType trans) {
  switch (trans) {
    case TransposeType::NoTranspose: return CUBLAS_OP_N;
    case TransposeType::Transpose: return CUBLAS_OP_T;
    case TransposeType::ConjTranspose: return CUBLAS_OP_C;
  }
  // Internal assertion for invalid transpose type
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

// Function template to get device pointers for batched cuBLAS/cuSOLVER routines
// 'input' must be a contiguous tensor
template <typename scalar_t>
static Tensor get_device_pointers(const Tensor& input) {
  auto input_data = input.const_data_ptr<scalar_t>();
  // Determine the stride between matrices in the batch
  int64_t input_mat_stride = matrixStride(input);

  // cuBLAS/cuSOLVER interfaces require batch_size as 'int'
  int batch_size = cuda_int_cast(batchCount(input), "batch_size");

  // If batch_size==0, then the range is empty
  // If input_mat_stride==0, then step is sizeof(scalar_t)
  return at::arange(
      /*start=*/reinterpret_cast<int64_t>(input_data),
      /*end=*/reinterpret_cast<int64_t>(input_data + batch_size * input_mat_stride),
      /*step=*/static_cast<int64_t>(std::max<int64_t>(input_mat_stride, 1) * sizeof(scalar_t)),
      input.options().dtype(at::kLong));
}

template <typename scalar_t>
// 对输入张量应用 cuBLAS 的 geqrf 批处理操作
void apply_geqrf_batched(const Tensor& input, const Tensor& tau) {
  // 获取批处理的大小
  auto batch_size = cuda_int_cast(batchCount(input), "batch_size");
  // 获取矩阵的行数 m
  auto m = cuda_int_cast(input.size(-2), "m");
  // 获取矩阵的列数 n
  auto n = cuda_int_cast(input.size(-1), "n");
  // 计算 lda，并确保至少为 1
  auto lda = std::max<int>(1, m);

  // cuBLAS 批处理 geqrf 要求输入为指向设备上单个矩阵的指针数组
  // 获取输入张量的设备指针数组
  Tensor input_ptr_array = get_device_pointers<scalar_t>(input);
  // 获取 tau 张量的设备指针数组，需要将其扩展为包含单个列的张量
  Tensor tau_ptr_array = get_device_pointers<scalar_t>(tau.unsqueeze(-1));
  // 将输入张量的指针数组数据转换为 scalar_t 类型的双重指针
  auto input_ptr_array_data = reinterpret_cast<scalar_t**>(input_ptr_array.data_ptr());
  // 将 tau 张量的指针数组数据转换为 scalar_t 类型的双重指针
  auto tau_ptr_array_data = reinterpret_cast<scalar_t**>(tau_ptr_array.data_ptr());

  // 用于存储信息的整型变量
  int info;
  // 获取当前 CUDA cuBLAS 句柄
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  // 调用 cuBLAS 的 geqrfBatched 函数执行批处理 QR 分解操作
  at::cuda::blas::geqrfBatched(handle, m, n, input_ptr_array_data, lda, tau_ptr_array_data, &info, batch_size);

  // info 变量只表示对 geqrfBatched 调用的参数是否正确
  // info 是一个主机变量，我们可以在不同步设备的情况下检查它
  TORCH_INTERNAL_ASSERT(info == 0);
}

// 对输入张量应用 cuBLAS 的 geqrf 批处理操作，使用指定的标量类型
void geqrf_batched_cublas(const Tensor& input, const Tensor& tau) {
  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏遍历所有浮点数和复数类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_batched_cuda", [&]{
    // 调用 apply_geqrf_batched 函数对指定的标量类型执行批处理 QR 分解操作
    apply_geqrf_batched<scalar_t>(input, tau);
  });
}

// 应用 cuBLAS 的 LU 分解批处理操作，使用指定的标量类型
template <typename scalar_t>
static void apply_lu_factor_batched_cublas(const Tensor& A, const Tensor& pivots, const Tensor& infos, bool get_pivots) {
  // 该函数仅适用于方阵
  TORCH_INTERNAL_ASSERT(A.size(-2) == A.size(-1));

  // 获取批处理的大小
  auto batch_size = cuda_int_cast(batchCount(A), "batch_size");
  // 获取矩阵的维度 n
  auto n = cuda_int_cast(A.size(-2), "n");
  // 计算 lda，并确保至少为 1
  auto lda = cuda_int_cast(std::max<int>(1, n), "lda");

  // 如果需要获取主元，则使用 pivots 张量的数据指针，否则设为 nullptr
  auto pivots_data = get_pivots ? pivots.data_ptr<int>() : nullptr;
  // 使用 infos 张量的数据指针
  auto infos_data = infos.data_ptr<int>();
  // 获取 A 张量的设备指针数组
  Tensor a_ptr_array = get_device_pointers<scalar_t>(A);
  // 将 A 张量的指针数组数据转换为 scalar_t 类型的双重指针
  auto a_ptr_array_data = reinterpret_cast<scalar_t**>(a_ptr_array.data_ptr());

  // 调用 cuBLAS 的 getrfBatched 函数执行批处理 LU 分解操作
  at::cuda::blas::getrfBatched(n, a_ptr_array_data, lda, pivots_data, infos_data, batch_size);
}

// 对输入张量应用 cuBLAS 的 LU 分解批处理操作，使用指定的标量类型
void lu_factor_batched_cublas(const Tensor& A, const Tensor& pivots, const Tensor& infos, bool get_pivots) {
  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏遍历所有浮点数和复数类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "lu_factor_cublas", [&]{
    // 调用 apply_lu_factor_batched_cublas 函数对指定的标量类型执行批处理 LU 分解操作
    apply_lu_factor_batched_cublas<scalar_t>(A, pivots, infos, get_pivots);
  });
}
// 使用 CUBLAS 库批处理解 LU 分解后的线性方程组 AX = B
static void apply_lu_solve_batched_cublas(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
  // 断言检查批处理数量必须相同
  TORCH_INTERNAL_ASSERT(batchCount(LU) == batchCount(B), "batch_size of LU and B must be the same");
  // 断言检查 LU 和 pivots 的批处理数量必须相同
  TORCH_INTERNAL_ASSERT(batchCount(LU) == batchCount(pivots.unsqueeze(-1)), "batch_size of LU and pivots must be the same");
  // 将 TransposeType 转换为 CUBLAS 格式
  const auto trans = to_cublas(transpose);

  // 获取 pivots 张量的数据指针
  auto pivots_data = pivots.const_data_ptr<int>();
  // 计算批处理大小
  auto batch_size = cuda_int_cast(batchCount(LU), "batch_size");;
  // 获取 LU 张量的行数（m 维度）
  auto m = cuda_int_cast(LU.size(-2), "m");
  // 获取 B 张量的列数（nrhs 维度）
  auto nrhs = cuda_int_cast(B.size(-1), "nrhs");
  // 计算 LU 张量的 lda 参数
  auto lda = cuda_int_cast(std::max<int>(1, m), "lda");
  // 初始化 info 变量
  int info = 0;

  // 获取 LU 和 B 张量的设备指针数组
  Tensor lu_ptr_array = get_device_pointers<scalar_t>(LU);
  Tensor b_ptr_array = get_device_pointers<scalar_t>(B);
  // 获取 LU 和 B 张量的设备指针数组数据指针
  auto lu_ptr_array_data = reinterpret_cast<const scalar_t* const*>(lu_ptr_array.const_data_ptr());
  auto b_ptr_array_data = reinterpret_cast<scalar_t**>(b_ptr_array.data_ptr());

  // 获取当前 CUDA 的 CUBLAS 句柄
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  // 调用 CUBLAS 批处理解线性方程组函数 getrsBatched
  at::cuda::blas::getrsBatched(handle, trans, m, nrhs, const_cast<scalar_t**>(lu_ptr_array_data),
    lda, const_cast<int*>(pivots_data), b_ptr_array_data, lda, &info, batch_size);
  // 断言调试模式下确保返回值 info 为 0
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
}

// 使用 CUBLAS 库解批处理 LU 分解后的线性方程组 AX = B
void lu_solve_batched_cublas(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏对浮点和复数类型进行分发
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_cublas", [&]{
    // 调用 apply_lu_solve_batched_cublas 函数来实现 LU 分解后的线性方程组解法
    apply_lu_solve_batched_cublas<scalar_t>(LU, pivots, B, trans);
  });
}

// 应用三角解算法，解解三角矩阵方程 AX = B
template <typename scalar_t>
static void apply_triangular_solve(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // 将 upper 转换为 CUBLAS 填充模式
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // 将 transpose 转换为 CUBLAS 转置类型
  const auto trans = to_cublas(transpose);
  // 将 left 转换为 CUBLAS 侧边模式
  cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  // 将 unitriangular 转换为 CUBLAS 对角类型
  cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  // 获取 A 和 B 张量的数据指针
  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  // 获取 A 和 B 张量的矩阵步幅
  auto A_mat_stride = matrixStride(A);
  auto B_mat_stride = matrixStride(B);
  // 获取批处理大小
  auto batch_size = batchCount(A);
  // 当 left 为 True 时，允许传递矩形 A 和 B
  auto m = cuda_int_cast(left ? A.size(-1) : B.size(-2), "m");
  auto n = cuda_int_cast(B.size(-1), "n");
  // 计算 A 和 B 张量的 lda 和 ldb 参数
  auto lda = std::max<int>(1, cuda_int_cast(A.size(-2), "lda"));
  auto ldb = std::max<int>(1, cuda_int_cast(B.size(-2), "ldb"));

  // 设置 alpha 值为 1
  auto alpha = scalar_t{1};

  // 循环处理每个批次的数据
  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    // 获取当前批次的 A 和 B 数据指针
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* B_working_ptr = &B_data[i * B_mat_stride];
    // 获取当前 CUDA 的 CUBLAS 句柄
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    // 调用 CUBLAS 三角矩阵求解函数 trsm 解决 AX = B
    at::cuda::blas::trsm(handle, side, uplo, trans, diag, m, n, &alpha, A_working_ptr, lda, B_working_ptr, ldb);
  }
}
void triangular_solve_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // 调用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏，根据 A 的数据类型分发任务
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cuda", [&]{
    // 调用模板函数 apply_triangular_solve，具体实现解三角矩阵方程的操作
    apply_triangular_solve<scalar_t>(A, B, left, upper, transpose, unitriangular);
  });
}

template <typename scalar_t>
static void apply_triangular_solve_batched(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // 根据 upper 参数选择 cuBLAS 的填充模式
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // 将 transpose 参数转换为 cuBLAS 的转置类型
  const auto trans = to_cublas(transpose);
  // 根据 left 参数选择 cuBLAS 的操作方向
  cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  // 根据 unitriangular 参数选择 cuBLAS 的对角类型
  cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  // 获取批处理大小
  auto batch_size = cuda_int_cast(batchCount(A), "batch_size");
  // 如果 left = True，选择 A 的列数作为 m，否则选择 B 的行数
  auto m = cuda_int_cast(left ? A.size(-1) : B.size(-2), "m");
  // 选择 B 的列数作为 n
  auto n = cuda_int_cast(B.size(-1), "n");
  // 计算 A 和 B 的 leading dimension
  auto lda = std::max<int>(1, cuda_int_cast(A.size(-2), "lda"));
  auto ldb = std::max<int>(1, cuda_int_cast(B.size(-2), "ldb"));

  // 设置 alpha 参数为标量 1
  auto alpha = scalar_t{1};

  // 准备 A 和 B 的指针数组，用于 cuBLAS 的批处理操作
  Tensor A_ptr_array = get_device_pointers<scalar_t>(A);
  Tensor B_ptr_array = get_device_pointers<scalar_t>(B);
  auto A_ptr_array_data = reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr());
  auto B_ptr_array_data = reinterpret_cast<scalar_t**>(B_ptr_array.data_ptr());

  // 获取当前 CUDA cuBLAS 句柄
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  // 调用 cuBLAS 的批处理解三角矩阵方程操作 trsmBatched
  at::cuda::blas::trsmBatched(handle, side, uplo, trans, diag, m, n, &alpha, A_ptr_array_data, lda, B_ptr_array_data, ldb, batch_size);
}

void triangular_solve_batched_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // 在 CUDA < 12.1 版本中存在的问题的解决方法
  // 当 B 的最后一个维度大小超过 max_batch_size 时，分割 B 并递归调用 triangular_solve_batched_cublas
#if defined(CUSOLVER_VERSION) && CUSOLVER_VERSION < 12100
  constexpr auto max_batch_size = 524280;
  if (B.size(-1) > max_batch_size) {
    auto n_chunks = (B.size(-1) + max_batch_size - 1) / max_batch_size; // ceildiv
    auto splits = B.split(n_chunks, /*dim=*/-1);
    for (const Tensor& b : splits) {
      triangular_solve_batched_cublas(A, b, left, upper, transpose, unitriangular);
    }
    return;
  }
#endif
  // 调用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏，根据 A 的数据类型分发任务
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cuda", [&]{
    // 调用模板函数 apply_triangular_solve_batched，具体实现批处理解三角矩阵方程的操作
    apply_triangular_solve_batched<scalar_t>(A, B, left, upper, transpose, unitriangular);
  });
}
// 将 apply_gels_batched 函数定义为内联函数，用于在编译时展开函数调用
inline void apply_gels_batched(const Tensor& A, Tensor& B, Tensor& infos) {
  // 设置转置选项为 CUBLAS_OP_N（不进行转置）
  auto trans = CUBLAS_OP_N;
  // 获取矩阵 A 的行数 m，并将其转换为 CUDA 所需的整数类型
  auto m = cuda_int_cast(A.size(-2), "m");
  // 获取矩阵 A 的列数 n，并将其转换为 CUDA 所需的整数类型
  auto n = cuda_int_cast(A.size(-1), "n");

  // 获取矩阵 B 的列数（右侧矩阵的列数）nrhs，并将其转换为 CUDA 所需的整数类型
  auto nrhs = cuda_int_cast(B.size(-1), "nrhs");
  // 对于 cuBLAS 版本低于 cuda11，nrhs 为 0 时会出现问题，因此在此处进行提前返回
  if (nrhs == 0) {
    return;
  }

  // 获取批处理的数量，并将其转换为 CUDA 所需的整数类型
  auto batch_size = cuda_int_cast(batchCount(B), "batch_size");
  // 计算矩阵 A 的列主元素 lda，并确保其最小值为 1
  auto lda = std::max<int>(1, m);
  // 计算矩阵 B 的列主元素 ldb，并确保其最小值为 1
  auto ldb = std::max<int>(1, m);

  // 检查矩阵维度是否满足 cuBLAS 的要求，即 m >= n
  TORCH_CHECK(
    m >= n,
    "torch.linalg.lstsq: only overdetermined systems (input.size(-2) >= input.size(-1)) are allowed on CUDA with cuBLAS backend.");

  // cuBLAS 文档要求：
  // Aarray[i] 矩阵不能重叠；否则，会出现未定义行为。
  // 显式广播 A 的批处理维度
  IntArrayRef A_batch_sizes(A.sizes().data(), A.dim() - 2);
  IntArrayRef B_batch_sizes(B.sizes().data(), B.dim() - 2);
  std::vector<int64_t> expand_batch_portion = at::infer_size(A_batch_sizes, B_batch_sizes);
  expand_batch_portion.insert(expand_batch_portion.end(), {A.size(-2), A.size(-1)});
  // 扩展 A 的维度以匹配 B 的维度
  Tensor A_expanded = A.expand({expand_batch_portion});
  // 复制 A 以保证每个批处理矩阵的列主元素顺序
  Tensor A_broadcasted = cloneBatchedColumnMajor(A_expanded);

  // cuBLAS 批处理 gels 要求输入是指向设备单独矩阵的设备指针数组
  // 获取 A_broadcasted 的设备指针数组
  Tensor A_ptr_array = get_device_pointers<scalar_t>(A_broadcasted);
  // 获取 B 的设备指针数组
  Tensor B_ptr_array = get_device_pointers<scalar_t>(B);
  // 将 A_ptr_array 转换为 scalar_t** 类型的数据
  auto A_ptr_array_data = reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr());
  // 将 B_ptr_array 转换为 scalar_t** 类型的数据
  auto B_ptr_array_data = reinterpret_cast<scalar_t**>(B_ptr_array.data_ptr());

  // 获取 infos 的数据指针，并将其类型设置为 int
  auto infos_data = infos.data_ptr<int>();
  // 获取当前 CUDA cuBLAS 句柄
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  int info;

  // 调用 cuBLAS 批处理 gels 函数
  at::cuda::blas::gelsBatched<scalar_t>(
    handle, trans, m, n, nrhs,
    A_ptr_array_data, lda,
    B_ptr_array_data, ldb,
    &info,
    infos_data,
    batch_size);

  // 负的 info 值表示 gelsBatched 调用中的一个参数无效
  TORCH_INTERNAL_ASSERT(info == 0);
}

// 这是 'apply_gels_batched' 的类型分发辅助函数
void gels_batched_cublas(const Tensor& a, Tensor& b, Tensor& infos) {
  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏对浮点数和复数类型进行分发
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "gels_batched_cublas", [&]{
    // 调用具体类型的 apply_gels_batched 函数
    apply_gels_batched<scalar_t>(a, b, infos);
  });
}

// 命名空间结束标记，结束 at::native 命名空间的定义
} // namespace at::native
```