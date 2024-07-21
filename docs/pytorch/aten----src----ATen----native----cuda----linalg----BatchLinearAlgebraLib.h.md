# `.\pytorch\aten\src\ATen\native\cuda\linalg\BatchLinearAlgebraLib.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <ATen/core/Tensor.h>
// 包含 ATen 库的 Tensor 头文件

#include <ATen/Context.h>
// 包含 ATen 库的 Context 头文件

#include <ATen/cuda/CUDAContext.h>
// 包含 ATen CUDA 上下文的头文件

#include <c10/cuda/CUDACachingAllocator.h>
// 包含 c10 CUDA 缓存分配器的头文件

#include <ATen/native/TransposeType.h>
// 包含 ATen native 模块的 TransposeType 头文件

#include <ATen/native/cuda/MiscUtils.h>
// 包含 ATen CUDA 的 MiscUtils 头文件

#if (defined(CUDART_VERSION) && defined(CUSOLVER_VERSION)) || defined(USE_ROCM)
#define USE_LINALG_SOLVER
#endif
// 如果定义了 CUDART_VERSION 和 CUSOLVER_VERSION 或者定义了 USE_ROCM，则定义 USE_LINALG_SOLVER

// cusolverDn<T>potrfBatched 可能在 cuda 11.3 之前的版本存在数值问题，
// 因此我们只在 cuda 版本 >= 11.3 时使用 cusolver potrf batched
#if CUSOLVER_VERSION >= 11101
  constexpr bool use_cusolver_potrf_batched_ = true;
#else
  constexpr bool use_cusolver_potrf_batched_ = false;
#endif
// 如果 CUSOLVER_VERSION >= 11101，则设置 use_cusolver_potrf_batched_ 为 true，否则为 false

// cusolverDn<T>syevjBatched 可能在 cuda 11.3.1 之前的版本存在数值问题，
// 因此我们只在 cuda 版本 >= 11.3.1 时使用 cusolver syevj batched
#if CUSOLVER_VERSION >= 11102
  constexpr bool use_cusolver_syevj_batched_ = true;
#else
  constexpr bool use_cusolver_syevj_batched_ = false;
#endif
// 如果 CUSOLVER_VERSION >= 11102，则设置 use_cusolver_syevj_batched_ 为 true，否则为 false

// 根据 cuSOLVER 文档：Jacobi 方法具有二次收敛性，因此精度与迭代次数无关。
// 为了保证一定的精度，用户只需配置容差。
// 当前 PyTorch 实现将 gesvdj 的容差设置为 C++ 数据类型的 epsilon，以获得最佳精度。
constexpr int cusolver_gesvdj_max_sweeps = 400;
// 设置 cusolver_gesvdj_max_sweeps 为 400，用于 Jacobi SVD 方法的最大迭代次数

namespace at {
namespace native {

void geqrf_batched_cublas(const Tensor& input, const Tensor& tau);
// 使用 cuBLAS 进行 geqrf 的批处理计算，对输入张量 input 进行 QR 分解

void triangular_solve_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular);
// 使用 cuBLAS 进行三角求解，解方程 A * X = B，其中 A 是系数矩阵，B 是结果张量

void triangular_solve_batched_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular);
// 使用 cuBLAS 进行批处理的三角求解，解方程组 A_i * X_i = B_i，其中 A_i 和 B_i 是批处理中的系数矩阵和结果张量

void gels_batched_cublas(const Tensor& a, Tensor& b, Tensor& infos);
// 使用 cuBLAS 进行 gels 的批处理计算，对多个线性方程组进行最小二乘拟合

void ldl_factor_cusolver(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian);
// 使用 cuSOLVER 进行 LDL 分解，计算对称正定矩阵的 LDL 分解

void ldl_solve_cusolver(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper);
// 使用 cuSOLVER 进行 LDL 分解的线性方程组求解

void lu_factor_batched_cublas(const Tensor& A, const Tensor& pivots, const Tensor& infos, bool get_pivots);
// 使用 cuBLAS 进行 LU 分解的批处理计算，对输入的批量矩阵 A 进行 LU 分解

void lu_solve_batched_cublas(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose);
// 使用 cuBLAS 进行批处理的 LU 分解的线性方程组求解，解方程组 LU_i * X_i = B_i

#if defined(USE_LINALG_SOLVER)

// 使用 cusolver 进行 SVD 计算的入口，支持 gesvdj 和 gesvdjBatched
void svd_cusolver(const Tensor& A, const bool full_matrices, const bool compute_uv,
  const std::optional<c10::string_view>& driver, const Tensor& U, const Tensor& S, const Tensor& V, const Tensor& info);
// 使用 cuSOLVER 进行 SVD 分解计算，计算输入张量 A 的奇异值分解，并可选返回 U, S, V

// 使用 cusolver 进行 Cholesky 分解计算的入口，支持 potrf 和 potrfBatched
void cholesky_helper_cusolver(const Tensor& input, bool upper, const Tensor& info);
// 使用 cuSOLVER 进行 Cholesky 分解计算，计算输入张量 input 的 Cholesky 分解，可选计算上/下三角形式
// 定义一个函数 _cholesky_solve_helper_cuda_cusolver，接受两个张量参数 self 和 A，以及一个布尔值 upper
Tensor _cholesky_solve_helper_cuda_cusolver(const Tensor& self, const Tensor& A, bool upper);

// 定义一个函数 cholesky_inverse_kernel_impl_cusolver，接受一个张量 result，一个张量 infos，以及一个布尔值 upper
Tensor& cholesky_inverse_kernel_impl_cusolver(Tensor &result, Tensor& infos, bool upper);

// 定义一个函数 geqrf_cusolver，接受一个张量 input 和一个张量 tau
void geqrf_cusolver(const Tensor& input, const Tensor& tau);

// 定义一个函数 ormqr_cusolver，接受三个张量 input、tau 和 other，以及两个布尔值 left 和 transpose
void ormqr_cusolver(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose);

// 定义一个函数 orgqr_helper_cusolver，接受一个张量 result 和一个张量 tau
Tensor& orgqr_helper_cusolver(Tensor& result, const Tensor& tau);

// 定义一个函数 linalg_eigh_cusolver，接受四个参数 eigenvalues、eigenvectors、infos 和两个布尔值 upper 和 compute_eigenvectors
void linalg_eigh_cusolver(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors);

// 定义一个函数 lu_solve_looped_cusolver，接受一个张量 LU、一个张量 pivots 和一个张量 B，以及一个枚举类型 TransposeType transpose
void lu_solve_looped_cusolver(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose);

// 定义一个函数 lu_factor_looped_cusolver，接受一个张量 self、一个张量 pivots、一个张量 infos 和一个布尔值 get_pivots
void lu_factor_looped_cusolver(const Tensor& self, const Tensor& pivots, const Tensor& infos, bool get_pivots);

// 结束 if 条件，这部分代码仅在 BUILD_LAZY_CUDA_LINALG 被定义时编译
#endif  // USE_LINALG_SOLVER

// 如果 BUILD_LAZY_CUDA_LINALG 被定义，则进入 cuda::detail 命名空间
#if defined(BUILD_LAZY_CUDA_LINALG)
namespace cuda { namespace detail {

// 定义一个结构体 LinalgDispatch，用于旧式调度
struct LinalgDispatch {
   // 定义一个指向函数的指针 cholesky_solve_helper，接受两个张量参数，返回一个张量
   Tensor (*cholesky_solve_helper)(const Tensor& self, const Tensor& A, bool upper);
};

// 声明函数 registerLinalgDispatch，接受一个 LinalgDispatch 结构体作为参数
C10_EXPORT void registerLinalgDispatch(const LinalgDispatch&);
}} // namespace cuda::detail
#endif

// 结束 cuda 命名空间和 at::native 命名空间
}}  // namespace at::native
```