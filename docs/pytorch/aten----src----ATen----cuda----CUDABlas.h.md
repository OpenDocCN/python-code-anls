# `.\pytorch\aten\src\ATen\cuda\CUDABlas.h`

```
#pragma once
/*
  提供CUDA BLAS功能的模板子集：

    gemm<Dtype>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
  ldc)

    gemv<Dtype>(transa, m, n, alpha, a, lda, x, incx, beta, y, incy)

    dot<Dtype>(n, x, incx, y, incy, result)

  其中Dtype可以是double、float、at::Half或at::BFloat16（ROCm，dot函数除外）。
  这些函数在at::cuda::blas命名空间中可用。
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/OpMathType.h>

namespace at::cuda::blas {

// RAII（资源获取即初始化）守卫，设置CuBLAS指针模式，并在守卫销毁时恢复到先前的值
class PointerModeGuard {
public:
  // 构造函数，设置指针模式为给定的mode
  PointerModeGuard(cublasHandle_t handle, cublasPointerMode_t mode) :
      handle(handle) {
    // 获取并记录当前的指针模式
    TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, &previous_mode));
    // 设置新的指针模式
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, mode));
  }

  // 析构函数，恢复到先前的指针模式
  ~PointerModeGuard() {
    cublasSetPointerMode(handle, previous_mode);
  }

private:
  cublasHandle_t handle;               // CuBLAS句柄
  cublasPointerMode_t previous_mode;   // 先前的指针模式
};

/* LEVEL 3 BLAS FUNCTIONS */

// 定义宏，用于指定gemm函数的参数类型
#define CUDABLAS_GEMM_ARGTYPES(Dtype)                                                       \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,  \
      const Dtype *a, int64_t lda, const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta,\
      Dtype *c, int64_t ldc

// 定义宏，用于展开gemm函数的参数
#define CUDABLAS_GEMM_ARGS(Dtype) transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc

// gemm函数模板，用于矩阵乘法
template <typename Dtype>
inline void gemm(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  // 静态断言，如果调用该函数，则输出错误信息，提示该函数未实现
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::gemm: not implemented");
}

// 实现特定数据类型（double）的gemm函数
template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double));

// 实现特定数据类型（float）的gemm函数
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float));

// 实现特定数据类型（c10::complex<double>）的gemm函数
template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>));

// 实现特定数据类型（c10::complex<float>）的gemm函数
template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>));

// 实现特定数据类型（at::Half）的gemm函数
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));

// 实现特定数据类型（at::BFloat16）的gemm函数
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));

// gemm_internal函数模板，用于内部实现的矩阵乘法
template <typename Dtype>
inline void gemm_internal(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  // 静态断言，如果调用该函数，则输出错误信息，提示该函数未实现
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::gemm_internal: not implemented");
}

// 实现特定数据类型（double）的gemm_internal函数
template <>
void gemm_internal<double>(CUDABLAS_GEMM_ARGTYPES(double));

// 实现特定数据类型（float）的gemm_internal函数
template <>
void gemm_internal<float>(CUDABLAS_GEMM_ARGTYPES(float));

// 实现特定数据类型（c10::complex<double>）的gemm_internal函数
template <>
void gemm_internal<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>));

// 实现特定数据类型（c10::complex<float>）的gemm_internal函数
template <>
void gemm_internal<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>));

// 实现特定数据类型（at::Half）的gemm_internal函数
template <>
void gemm_internal<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half));

// 实现特定数据类型（at::BFloat16）的gemm_internal函数
template <>
void gemm_internal<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16));

// 定义GEMMAndBiasActivationEpilogue枚举类型，用于GEMM函数的偏置和激活
enum GEMMAndBiasActivationEpilogue {
  None,   // 无激活函数
  RELU,   // ReLU激活函数
  GELU,   // GELU激活函数（仅支持CUDA 11.4及以上版本）
};

// 模板函数，实现带偏置和激活函数的矩阵乘法
template <typename Dtype>
void gemm_and_bias(
    // 是否对第一个矩阵进行转置的标志
    bool transpose_mat1,
    // 是否对第二个矩阵进行转置的标志
    bool transpose_mat2,
    // 矩阵乘法的维度：第一个矩阵的行数或第二个矩阵的列数
    int64_t m,
    // 矩阵乘法的维度：第一个矩阵的列数或第二个矩阵的行数
    int64_t n,
    // 矩阵乘法的维度：第一个矩阵的列数或第二个矩阵的列数
    int64_t k,
    // 乘法运算中的倍数因子
    at::opmath_type<Dtype> alpha_val,
    // 第一个矩阵的指针
    const Dtype* mat1_ptr,
    // 第一个矩阵的领导维度（leading dimension）
    int64_t mat1_ld,
    // 第二个矩阵的指针
    const Dtype* mat2_ptr,
    // 第二个矩阵的领导维度（leading dimension）
    int64_t mat2_ld,
    // 偏置项的指针（可选）
    const Dtype* bias,
    // 结果矩阵的指针
    Dtype* result_ptr,
    // 结果矩阵的领导维度（leading dimension）
    int64_t result_ld,
    // 激活函数和矩阵乘法后处理的选项，默认为无后处理
    GEMMAndBiasActivationEpilogue activation = GEMMAndBiasActivationEpilogue::None);
void int8_gemm(
    bool transpose_mat1,            // 是否转置矩阵1
    bool transpose_mat2,            // 是否转置矩阵2
    int64_t m,                      // 矩阵1的行数
    int64_t n,                      // 矩阵2的列数
    int64_t k,                      // 矩阵1的列数/矩阵2的行数
    const int8_t* mat1_ptr,         // 矩阵1数据指针
    int64_t mat1_ld,                // 矩阵1的leading dimension
    const int8_t* mat2_ptr,         // 矩阵2数据指针
    int64_t mat2_ld,                // 矩阵2的leading dimension
    int32_t* result_ptr,            // 结果数据指针
    int64_t result_ld);             // 结果矩阵的leading dimension

void scaled_gemm(
    char transa,                    // 矩阵A是否转置
    char transb,                    // 矩阵B是否转置
    int64_t m,                      // 矩阵A的行数
    int64_t n,                      // 矩阵B的列数
    int64_t k,                      // 矩阵A的列数/矩阵B的行数
    const void* mat1_ptr,           // 矩阵A数据指针
    const void* mat1_scale_ptr,     // 矩阵A的缩放因子指针
    int64_t mat1_ld,                // 矩阵A的leading dimension
    ScalarType mat1_dtype,          // 矩阵A的数据类型
    const void* mat2_ptr,           // 矩阵B数据指针
    const void* mat2_scale_ptr,     // 矩阵B的缩放因子指针
    int64_t mat2_ld,                // 矩阵B的leading dimension
    ScalarType mat2_dtype,          // 矩阵B的数据类型
    const void* bias_ptr,           // 偏置数据指针
    ScalarType bias_dtype,          // 偏置的数据类型
    void* result_ptr,               // 结果数据指针
    const void* result_scale_ptr,   // 结果的缩放因子指针
    int64_t result_ld,              // 结果矩阵的leading dimension
    ScalarType result_dtype,        // 结果的数据类型
    void* amax_ptr,                 // 用于存储最大值的指针
    bool use_fast_accum);           // 是否使用快速累积

#define CUDABLAS_BGEMM_ARGTYPES(Dtype)                                                        \
  char transa, char transb, int64_t m, int64_t n, int64_t k, at::opmath_type<Dtype> alpha,    \
      const Dtype *a, int64_t lda, int64_t stridea,                                           \
      const Dtype *b, int64_t ldb, int64_t strideb,                                           \
      at::opmath_type<Dtype> beta, Dtype *c, int64_t ldc, int64_t stridec, int64_t num_batches

#define CUDABLAS_BGEMM_ARGS(Dtype) \
  transa, transb, m, n, k, alpha, a, lda, stridea, b, ldb, strideb, beta, c, ldc, stridec, num_batches

template <typename Dtype>
inline void bgemm(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::bgemm: not implemented");
}

template <>
void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double));
template <>
void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float));
template <>
void bgemm<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>));
template <>
void bgemm<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>));
template <>
void bgemm<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half));
template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));

template <typename Dtype>
inline void bgemm_internal(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::bgemm_internal: not implemented");
}

template <>
void bgemm_internal<double>(CUDABLAS_BGEMM_ARGTYPES(double));
template <>
void bgemm_internal<float>(CUDABLAS_BGEMM_ARGTYPES(float));
template <>
void bgemm_internal<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>));
template <>
void bgemm_internal<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>));
template <>
void bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half));
template <>
void bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));

#define CUDABLAS_TRSM_ARGTYPES(Dtype)                                  \
  cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, const Dtype *A, int lda, Dtype *B, int ldb
// 定义模板函数 trsm，用于解决线性系统 Ax = B 的问题，目前未实现，会触发静态断言
template <typename Dtype>
inline void trsm(CUDABLAS_TRSM_ARGTYPES(Dtype)) {
  // 静态断言，当此函数被调用时，引发错误信息表明该函数未被实现
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::trsm: not implemented");
}

// 显式实例化模板函数 trsm<float> 的定义
template <>
TORCH_CUDA_CU_API void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float));

// 显式实例化模板函数 trsm<double> 的定义
template <>
TORCH_CUDA_CU_API void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double));

// 显式实例化模板函数 trsm<c10::complex<float>> 的定义
template <>
TORCH_CUDA_CU_API void trsm<c10::complex<float>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<float>));

// 显式实例化模板函数 trsm<c10::complex<double>> 的定义
template <>
TORCH_CUDA_CU_API void trsm<c10::complex<double>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<double>));

// 定义宏 CUDABLAS_TRSM_BATCHED_ARGTYPES，用于批量处理 trsmBatched 的参数类型
#define CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)                          \
  cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, \
      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, Dtype *A[], int lda, Dtype *B[], int ldb,    \
      int batchCount

// 定义模板函数 trsmBatched，用于批量解决线性系统的问题，目前未实现，会触发静态断言
template <typename Dtype>
inline void trsmBatched(CUDABLAS_TRSM_BATCHED_ARGTYPES(Dtype)) {
  // 静态断言，当此函数被调用时，引发错误信息表明该函数未被实现
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::trsmBatched: not implemented");
}

// 显式实例化模板函数 trsmBatched<float> 的定义
template <>
TORCH_CUDA_CU_API void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float));

// 显式实例化模板函数 trsmBatched<double> 的定义
template <>
TORCH_CUDA_CU_API void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double));

// 显式实例化模板函数 trsmBatched<c10::complex<float>> 的定义
template <>
TORCH_CUDA_CU_API void trsmBatched<c10::complex<float>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>));

// 显式实例化模板函数 trsmBatched<c10::complex<double>> 的定义
template <>
TORCH_CUDA_CU_API void trsmBatched<c10::complex<double>>(CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>));

// 定义宏 CUDABLAS_GEMV_ARGTYPES，用于描述 gemv 函数的参数类型
#define CUDABLAS_GEMV_ARGTYPES(Dtype)                                         \
  char trans, int64_t m, int64_t n, Dtype alpha, const Dtype *a, int64_t lda, \
      const Dtype *x, int64_t incx, Dtype beta, Dtype *y, int64_t incy

// 定义模板函数 gemv，用于实现矩阵向量乘法，目前未实现，会触发静态断言
template <typename Dtype>
inline void gemv(CUDABLAS_GEMV_ARGTYPES(Dtype)) {
  // 静态断言，当此函数被调用时，引发错误信息表明该函数未被实现
  static_assert(false&&sizeof(Dtype), "at::cuda::blas::gemv: not implemented");
}

// 显式实例化模板函数 gemv<double> 的定义
template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double));

// 显式实例化模板函数 gemv<float> 的定义
template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float));

// 显式实例化模板函数 gemv<c10::complex<double>> 的定义
template <>
void gemv<c10::complex<double>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<double>));

// 显式实例化模板函数 gemv<c10::complex<float>> 的定义
template <>
void gemv<c10::complex<float>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<float>));

// 显式实例化模板函数 gemv<at::Half> 的定义
template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half));

// 显式实例化模板函数 gemv<at::BFloat16> 的定义
template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16));

// 定义宏 CUDABLAS_DOT_ARGTYPES，用于描述 dot 函数的参数类型
#define CUDABLAS_DOT_ARGTYPES(Dtype)                                      \
  cublasHandle_t handle, int n, const Dtype *x, int incx, const Dtype *y, \
      int incy, Dtype *result

// 定义模板函数 dot，用于计算向量的内积，目前未实现，会触发静态断言
template <typename Dtype>
inline void dot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
  // 静态断言，当此函数被调用时，引发错误信息表明该函数未被实现
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::dot: not implemented");
}

// 显式实例化模板函数 dot<double> 的定义
template <>
void dot<double>(CUDABLAS_DOT_ARGTYPES(double));

// 显式实例化模板函数 dot<float> 的定义
template <>
void dot<float>(CUDABLAS_DOT_ARGTYPES(float));

// 显式实例化模板函数 dot<at::Half> 的定义
template <>
void dot<at::Half>(CUDABLAS_DOT_ARGTYPES(at::Half));

// 显式实例化模板函数 dot<at::BFloat16> 的定义
template <>
void dot<at::BFloat16>(CUDABLAS_DOT_ARGTYPES(at::BFloat16));
// 声明一个模板函数 dot，接受 c10::complex<double> 类型的参数，用于执行复数向量的点积运算
void dot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>));

// 显式实例化模板函数 dot，针对 c10::complex<float> 类型的参数，用于执行复数向量的点积运算
template <>
void dot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>));

// 声明一个模板函数 vdot，接受 Dtype 类型的参数，实现向量的点积运算
template <typename Dtype>
inline void vdot(CUDABLAS_DOT_ARGTYPES(Dtype)) {
  // 静态断言，当前类型不支持此函数，抛出编译时错误
  static_assert(false && sizeof(Dtype), "at::cuda::blas::vdot: not implemented");
}

// 显式实例化模板函数 vdot，针对 c10::complex<float> 类型的参数，实现复数向量的点积运算
template <>
void vdot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>));

// 显式实例化模板函数 vdot，针对 c10::complex<double> 类型的参数，实现复数向量的点积运算
template <>
void vdot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>));

// 定义一个宏，展开为模板函数的参数列表，用于 cublas 的 getrsBatched 函数
#define CUDABLAS_GETRS_ARGTYPES(Dtype)  \
  cublasHandle_t handle, cublasOperation_t trans, \
  int n, int nrhs, Dtype** dA_array, int lda, int* ipiv_array, \
  Dtype** dB_array, int ldb, int* info_array, int batchsize

// 声明一个模板函数 getrsBatched，接受 Dtype 类型的参数，用于批量求解线性方程组
template<class Dtype>
void getrsBatched(CUDABLAS_GETRS_ARGTYPES(Dtype)) {
  // 静态断言，当前类型不支持此函数，抛出编译时错误
  static_assert(false && sizeof(Dtype), "at::cuda::blas::getrsBatched: not implemented");
}

// 显式实例化模板函数 getrsBatched，针对 float 类型的参数，实现批量求解线性方程组
template<>
TORCH_CUDA_CU_API void getrsBatched<float>(CUDABLAS_GETRS_ARGTYPES(float));

// 显式实例化模板函数 getrsBatched，针对 double 类型的参数，实现批量求解线性方程组
template<>
TORCH_CUDA_CU_API void getrsBatched<double>(CUDABLAS_GETRS_ARGTYPES(double));

// 显式实例化模板函数 getrsBatched，针对 c10::complex<float> 类型的参数，实现批量求解线性方程组
template<>
TORCH_CUDA_CU_API void getrsBatched<c10::complex<float>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<float>));

// 显式实例化模板函数 getrsBatched，针对 c10::complex<double> 类型的参数，实现批量求解线性方程组
template<>
TORCH_CUDA_CU_API void getrsBatched<c10::complex<double>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<double>));

// 定义一个宏，展开为模板函数的参数列表，用于 cublas 的 geqrfBatched 函数
#define CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)                   \
  cublasHandle_t handle, int m, int n, Dtype **A_array, int lda, \
      Dtype **tau_array, int *info, int batchsize

// 声明一个模板函数 geqrfBatched，接受 Dtype 类型的参数，用于批量 QR 分解
template <class Dtype>
void geqrfBatched(CUDABLAS_GEQRF_BATCHED_ARGTYPES(Dtype)) {
  // 静态断言，当前类型不支持此函数，抛出编译时错误
  static_assert(false && sizeof(Dtype), "at::cuda::blas::geqrfBatched: not implemented");
}

// 显式实例化模板函数 geqrfBatched，针对 float 类型的参数，实现批量 QR 分解
template <>
TORCH_CUDA_CU_API void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float));

// 显式实例化模板函数 geqrfBatched，针对 double 类型的参数，实现批量 QR 分解
template <>
TORCH_CUDA_CU_API void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double));

// 显式实例化模板函数 geqrfBatched，针对 c10::complex<double> 类型的参数，实现批量 QR 分解
template <>
TORCH_CUDA_CU_API void geqrfBatched<c10::complex<double>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<double>));

// 显式实例化模板函数 geqrfBatched，针对 c10::complex<float> 类型的参数，实现批量 QR 分解
template <>
TORCH_CUDA_CU_API void geqrfBatched<c10::complex<float>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<float>));

// 定义一个宏，展开为模板函数的参数列表，用于 cublas 的 getrfBatched 函数
#define CUDABLAS_GETRF_ARGTYPES(Dtype)  \
  int n, Dtype** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize

// 声明一个模板函数 getrfBatched，接受 Dtype 类型的参数，用于批量 LU 分解
template<class Dtype>
void getrfBatched(CUDABLAS_GETRF_ARGTYPES(Dtype)) {
  // 抛出运行时错误，批量 LU 分解函数未实现
  TORCH_CHECK(false, "at::cuda::blas::getrfBatched: not implemented");
}

// 显式实例化模板函数 getrfBatched，针对 float 类型的参数，实现批量 LU 分解
template<>
TORCH_CUDA_CU_API void getrfBatched<float>(CUDABLAS_GETRF_ARGTYPES(float));

// 显式实例化模板函数 getrfBatched，针对 double 类型的参数，实现批量 LU 分解
template<>
TORCH_CUDA_CU_API void getrfBatched<double>(CUDABLAS_GETRF_ARGTYPES(double));

// 显式实例化模板函数 getrfBatched，针对 c10::complex<double> 类型的参数，实现批量 LU 分解
template<>
TORCH_CUDA_CU_API void getrfBatched<c10::complex<double>>(CUDABLAS_GETRF_ARGTYPES(c10::complex<double>));

// 显式实例化模板函数 getrfBatched，针对 c10::complex<float> 类型的参数，实现批量 LU 分解
template<>
TORCH_CUDA_CU_API void getrfBatched<c10::complex<float>>(CUDABLAS_GETRF_ARGTYPES(c10::complex<float>));
# 定义模板宏，用于生成不同数据类型的 gelsBatched 函数的参数列表
#define CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)  \
  cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, Dtype** dA_array, int ldda, Dtype** dC_array, int lddc, int* info, int *devInfoArray, int batchSize

# 定义 gelsBatched 函数模板，针对不同的 Dtype 类型，静态断言指出函数未实现
template <class Dtype>
void gelsBatched(CUDABLAS_GELS_BATCHED_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),"at::cuda::blas::gelsBatched: not implemented");
}

# 显式实例化 gelsBatched 函数模板，针对不同的 Dtype 类型
template<>
TORCH_CUDA_CU_API void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double));
template<>
TORCH_CUDA_CU_API void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float));
template<>
TORCH_CUDA_CU_API void gelsBatched<c10::complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<double>));
template<>
TORCH_CUDA_CU_API void gelsBatched<c10::complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<float>));

} // namespace at::cuda::blas
```