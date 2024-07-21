# `.\pytorch\aten\src\ATen\native\cuda\linalg\BatchLinearAlgebra.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含标准库头文件 <utility>
#include <utility>

// 包含 ATen 库的头文件
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

// 包含 C10 异常处理工具
#include <c10/util/Exception.h>

// 包含 ATen 线性代数和 CUDA 相关的实用工具和操作
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>
#include <ATen/native/cuda/linalg/MagmaUtils.h>
#include <ATen/native/cpu/zmath.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含更多的 ATen 操作头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_solve_triangular.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_linalg_check_errors.h>
#endif

// 如果 AT_MAGMA_ENABLED 宏定义为真，则包含 MAGMA 库相关的头文件
#if AT_MAGMA_ENABLED()
#include <magma_types.h>
#include <magma_v2.h>
#include <ATen/cuda/detail/CUDAHooks.h>

// 设置使用 MAGMA 库
const bool use_magma_ = true;

namespace {
// 匿名命名空间，用于定义 MAGMA 初始化器结构
struct MagmaInitializer {
  MagmaInitializer() {
    // 根据条件选择是否延迟初始化 CUDA 的 MAGMA 库
#if defined(BUILD_LAZY_CUDA_LINALG)
    magma_init();
#else
    ::at::cuda::detail::set_magma_init_fn([]{ magma_init(); });
#endif
  }
} initializer;
}  // namespace (anonymous)

// 定义 AT_MAGMA_VERSION 宏，用于检查 MAGMA 版本
#define AT_MAGMA_VERSION MAGMA_VERSION_MAJOR*100 + MAGMA_VERSION_MINOR*10 + MAGMA_VERSION_MICRO

// 检查 MAGMA 版本是否满足要求，不满足则抛出错误
#if MAGMA_VERSION_MINOR >= 10 || MAGMA_VERSION_MICRO >= 10
#error "MAGMA release minor or micro version >= 10, please correct AT_MAGMA_VERSION"
#endif

// 如果未启用 MAGMA，则设置 use_magma_ 为假
#else
const bool use_magma_ = false;

#endif

namespace at::native {
// 如果定义了 BUILD_LAZY_CUDA_LINALG，则所有与 PyTorch 运行时的注册都必须动态完成
#if defined(BUILD_LAZY_CUDA_LINALG)
// 所有与 PyTorch 运行时注册的操作必须动态完成
namespace lazy_linalg {
#endif

// 如果 MAGMA 可用，则定义使用 MAGMA 库的 LDL 分解函数模板
#if AT_MAGMA_ENABLED()
template <class scalar_t>
void magmaLdlHermitian(
    magma_uplo_t uplo,
    magma_int_t n,
    scalar_t* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  // 当前版本的 MAGMA 不支持 LDL 分解，抛出错误提示
  TORCH_CHECK(
      false,
      "LDL decomposition is not available.",
      "Please rebuild with MAGMA 2.5.4+.");
}

// 定义使用 MAGMA 库的 LU 分解函数模板
template<class scalar_t>
void magmaLu(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info);

// 定义使用 MAGMA 库的批量 LU 分解函数模板
template<class scalar_t>
void magmaLuBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);

// 定义使用 MAGMA 库的无主元 LU 分解函数模板
template<class scalar_t>
void magmaLuNoPiv(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    magma_int_t* info);



// 声明一个指向 magma_int_t 类型的指针参数 info 的函数
// 批量解决没有主元的 LU 分解。对每个矩阵执行 LU 分解，dA_array 是包含矩阵指针的数组，
// m 是每个矩阵的行数，n 是每个矩阵的列数，ldda 是每个矩阵的列偏移，
// info_array 是返回每个 LU 分解的信息的数组，batchsize 是批处理的矩阵数量，
// magma_queue 是用于执行操作的 MAGMA 队列。
template<class scalar_t>
void magmaLuNoPivBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

// 解决 Cholesky 分解后的线性方程组，dA 是 Cholesky 分解后的因子，dB 是右侧矩阵，
// uplo 指定是使用上三角分解还是下三角分解，n 是矩阵的阶数，nrhs 是右侧矩阵的列数，
// ldda 和 lddb 分别是矩阵 dA 和 dB 的列偏移，info 是返回操作信息的整数数组。
template<class scalar_t>
void magmaCholeskySolve(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t* dA, magma_int_t ldda,
    scalar_t* dB, magma_int_t lddb, magma_int_t* info);

// 批量解决 Cholesky 分解后的线性方程组，dA_array 是 Cholesky 分解后的因子指针数组，
// dB_array 是右侧矩阵指针数组，uplo 指定是使用上三角分解还是下三角分解，n 是矩阵的阶数，
// nrhs 是右侧矩阵的列数，ldda 和 lddb 分别是矩阵 dA 和 dB 的列偏移，
// info 是返回操作信息的整数，batchsize 是批处理的矩阵数量，magma_queue 是用于执行操作的 MAGMA 队列。
template<class scalar_t>
void magmaCholeskySolveBatched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue);

// 执行 Cholesky 分解，dA 是输入矩阵，uplo 指定是使用上三角分解还是下三角分解，
// n 是矩阵的阶数，ldda 是矩阵 dA 的列偏移，info 是返回操作信息的整数数组。
template<class scalar_t>
void magmaCholesky(
    magma_uplo_t uplo, magma_int_t n, scalar_t* dA,
    magma_int_t ldda, magma_int_t* info);

// 批量执行 Cholesky 分解，dA_array 是输入矩阵指针数组，uplo 指定是使用上三角分解还是下三角分解，
// n 是矩阵的阶数，ldda 是矩阵 dA 的列偏移，info_array 是返回每个 Cholesky 分解的信息的数组，
// batchsize 是批处理的矩阵数量，magma_queue 是用于执行操作的 MAGMA 队列。
template<class scalar_t>
void magmaCholeskyBatched(
    magma_uplo_t uplo, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

// 批量解决三角线性方程组，side 指定是左乘还是右乘，uplo 指定是使用上三角分解还是下三角分解，
// trans 指定是使用矩阵的转置还是共轭转置，diag 指定是否为单位对角线，m 和 n 是矩阵的行数和列数，
// dA_array 是三角矩阵的指针数组，ldda 是矩阵 dA 的列偏移，dB_array 是右侧矩阵的指针数组，
// lddb 是矩阵 dB 的列偏移，batchsize 是批处理的矩阵数量，magma_queue 是用于执行操作的 MAGMA 队列。
template<class scalar_t>
void magmaTriangularSolveBatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    scalar_t** dA_array, magma_int_t ldda, scalar_t** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);

// 计算 Geqrf 算法的最佳块大小，m 和 n 是矩阵的行数和列数。
template<class scalar_t>
inline magma_int_t magmaGeqrfOptimalBlocksize(magma_int_t m, magma_int_t n);

// 执行 Geqrf QR 分解，dA 是输入矩阵，m 和 n 是矩阵的行数和列数，
// ldda 是矩阵 dA 的列偏移，tau 是返回的 Householder 变换系数，dT 是临时存储变换数据的数组，
// info 是返回操作信息的整数数组，is_v2 是指定执行的变体。
template<class scalar_t>
void magmaGeqrf(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    scalar_t* tau, scalar_t* dT, magma_int_t* info, bool is_v2);

// 执行对称矩阵的特征值分解，dA 是输入矩阵，jobz 指定是否计算特征向量，
// uplo 指定是使用上三角分解还是下三角分解，n 是矩阵的阶数，w 是返回的特征值数组，
// wA 是返回的特征向量数组，ldwa 是矩阵 wA 的列偏移，work 是临时存储空间数组，
// lwork 是工作空间数组的大小，rwork 是实数工作空间数组，lrwork 是实数工作空间数组的大小，
// iwork 是整数工作空间数组，liwork 是整数工作空间数组的大小，info 是返回操作信息的整数数组。
template<class scalar_t, class value_t=scalar_t>
void magmaSyevd(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    value_t* w, scalar_t* wA, magma_int_t ldwa, scalar_t* work, magma_int_t lwork, value_t* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info);

// 执行一般实矩阵的特征值和特征向量计算，jobvl 和 jobvr 分别指定是否计算左特征向量和右特征向量，
// n 是矩阵的阶数，A 是输入矩阵，lda 是矩阵 A 的列偏移，w 是返回的特征值数组，
// VL 和 VR 是返回的左特征向量和右特征向量数组，ldvl 和 ldvr 分别是 VL 和 VR 的列偏移，
// work 是临时存储空间数组，lwork 是工作空间数组的大小，rwork 是实数工作空间数组，
// info 是返回操作信息的整数数组。
template<class scalar_t, class value_t=scalar_t>
void magmaEig(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, scalar_t *A, magma_int_t lda,
    scalar_t *w, scalar_t *VL, magma_int_t ldvl,
    scalar_t *VR, magma_int_t ldvr, scalar_t *work, magma_int_t lwork,
    value_t *rwork,
    magma_int_t *info);

// 执行 SVD 分解，jobz
    scalar_t** dB_array,   // scalar_t 类型的双指针数组，用于存储数据
    magma_int_t lddb,       // magma_int_t 类型的参数，表示数组的 leading dimension
    magma_int_t& info,      // magma_int_t 类型的引用参数，用于返回操作的信息
    magma_int_t batchsize,  // magma_int_t 类型的参数，表示批处理的大小
    const MAGMAQueue& magma_queue,  // MAGMAQueue 类型的常量引用，表示用于操作的队列
    magma_trans_t trans);   // magma_trans_t 类型的参数，表示操作的转置情况
template<class scalar_t>
void magmaGels(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    scalar_t* dA, magma_int_t ldda, scalar_t* dB, magma_int_t lddb,
    scalar_t* hwork, magma_int_t lwork, magma_int_t* info);


# magmaGels 模板函数定义，用于解线性方程组 Ax = B 中的最小二乘问题
template<class scalar_t>
void magmaGels(
    magma_trans_t trans,        // 转置选项，控制 A 是否转置
    magma_int_t m,              // A 矩阵的行数
    magma_int_t n,              // A 矩阵的列数
    magma_int_t nrhs,           // B 矩阵的列数，即右侧矩阵的列数
    scalar_t* dA,               // A 矩阵在 GPU 上的地址
    magma_int_t ldda,           // A 矩阵在 GPU 上的 leading dimension
    scalar_t* dB,               // B 矩阵在 GPU 上的地址
    magma_int_t lddb,           // B 矩阵在 GPU 上的 leading dimension
    scalar_t* hwork,            // CPU 上的工作空间数组
    magma_int_t lwork,          // 工作空间数组 hwork 的长度
    magma_int_t* info           // 输出参数，包含执行结果信息
);



#if AT_MAGMA_VERSION >= 254


# 如果 MAGMA 的版本号大于等于 254，执行以下模板特化函数

template <>
void magmaLdlHermitian<double>(
    magma_uplo_t uplo,          // 上/下三角矩阵选项
    magma_int_t n,              // 矩阵的阶数
    double* dA,                 // 双精度实数类型的矩阵 A 在 GPU 上的地址
    magma_int_t ldda,           // A 矩阵在 GPU 上的 leading dimension
    magma_int_t* ipiv,          // 存储主元信息的数组
    magma_int_t* info           // 输出参数，包含执行结果信息
) {
  MagmaStreamSyncGuard guard;   // 确保操作在正确的 CUDA 流上进行同步
  magma_dsytrf_gpu(uplo, n, dA, ldda, ipiv, info);   // 调用 MAGMA 函数进行双精度实数类型的 LDL 分解
  AT_CUDA_CHECK(cudaGetLastError());   // 检查 CUDA 操作是否出错
}



template <>
void magmaLdlHermitian<float>(
    magma_uplo_t uplo,          // 上/下三角矩阵选项
    magma_int_t n,              // 矩阵的阶数
    float* dA,                  // 单精度实数类型的矩阵 A 在 GPU 上的地址
    magma_int_t ldda,           // A 矩阵在 GPU 上的 leading dimension
    magma_int_t* ipiv,          // 存储主元信息的数组
    magma_int_t* info           // 输出参数，包含执行结果信息
) {
  MagmaStreamSyncGuard guard;   // 确保操作在正确的 CUDA 流上进行同步
  magma_ssytrf_gpu(uplo, n, dA, ldda, ipiv, info);   // 调用 MAGMA 函数进行单精度实数类型的 LDL 分解
  AT_CUDA_CHECK(cudaGetLastError());   // 检查 CUDA 操作是否出错
}



template <>
void magmaLdlHermitian<c10::complex<double>>(
    magma_uplo_t uplo,                  // 上/下三角矩阵选项
    magma_int_t n,                      // 矩阵的阶数
    c10::complex<double>* dA,           // 双精度复数类型的矩阵 A 在 GPU 上的地址
    magma_int_t ldda,                   // A 矩阵在 GPU 上的 leading dimension
    magma_int_t* ipiv,                  // 存储主元信息的数组
    magma_int_t* info                   // 输出参数，包含执行结果信息
) {
  MagmaStreamSyncGuard guard;           // 确保操作在正确的 CUDA 流上进行同步
  magma_zhetrf_gpu(                     // 调用 MAGMA 函数进行双精度复数类型的 Hermitian LDL 分解
      uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());   // 检查 CUDA 操作是否出错
}



template <>
void magmaLdlHermitian<c10::complex<float>>(
    magma_uplo_t uplo,                  // 上/下三角矩阵选项
    magma_int_t n,                      // 矩阵的阶数
    c10::complex<float>* dA,            // 单精度复数类型的矩阵 A 在 GPU 上的地址
    magma_int_t ldda,                   // A 矩阵在 GPU 上的 leading dimension
    magma_int_t* ipiv,                  // 存储主元信息的数组
    magma_int_t* info                   // 输出参数，包含执行结果信息
) {
  MagmaStreamSyncGuard guard;           // 确保操作在正确的 CUDA 流上进行同步
  magma_chetrf_gpu(                     // 调用 MAGMA 函数进行单精度复数类型的 Hermitian LDL 分解
      uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());   // 检查 CUDA 操作是否出错
}



#endif // AT_MAGMA_VERSION >= 254



template<>
void magmaLu<double>(
    magma_int_t m,                      // 矩阵 A 的行数
    magma_int_t n,                      // 矩阵 A 的列数
    double* dA,                         // 双精度实数类型的矩阵 A 在 GPU 上的地址
    magma_int_t ldda,                   // A 矩阵在 GPU 上的 leading dimension
    magma_int_t* ipiv,                  // 存储主元信息的数组
    magma_int_t* info                   // 输出参数，包含执行结果信息
) {
  MagmaStreamSyncGuard guard;           // 确保操作在正确的 CUDA 流上进行同步
  magma_dgetrf_gpu(m, n, dA, ldda, ipiv, info);   // 调用 MAGMA 函数进行双精度实数类型的 LU 分解
  AT_CUDA_CHECK(cudaGetLastError());   // 检查 CUDA 操作是否出错
}



template<>
void magmaLu<float>(
    magma_int_t m,                      // 矩阵 A 的行数
    magma_int_t n,                      // 矩阵 A 的列数
    float* dA,                          // 单精度实数类型的矩阵 A 在 GPU 上的地址
    magma_int_t ldda,                   // A 矩阵在 GPU 上的 leading dimension
    magma_int_t* ipiv,                  // 存储主元信息的数组
    magma_int_t* info                   // 输出参数，包含执行结果信息
) {
  MagmaStreamSyncGuard guard;           // 确保操作在正确的 CUDA 流上进行同步
  magma_sgetrf_gpu(m, n, dA, ldda, ipiv, info);   // 调用 MAGMA 函数进行单精度实数类型的 LU 分解
  AT_CUDA_CHECK(cudaGetLastError());   // 检查 CUDA 操作是否出错
}



template<>
void magmaLu<c10::complex<double>>(
    magma_int_t m,                          // 矩阵 A 的行数
    magma_int_t n,                          // 矩阵 A 的列数
    c10::complex<double>* dA,               // 双精度复数类型的矩阵 A 在 GPU 上的地址
    magma_int_t ldda,                       // A 矩阵在 GPU 上的 leading dimension
    magma_int_t* ipiv,                      // 存储主元信息的数组
    magma_int_t* info                       // 输出参数，包含执行结果信息
) {
  MagmaStreamSyncGuard guard;               // 确保操作在正确的 CUDA 流上进行同步
  magma_zgetrf
    // 使用 MAGMA 库中的批量 LU 分解函数 magma_dgetrf_batched 对输入的一批设备上的矩阵执行 LU 分解
    magma_dgetrf_batched(m,                     // 矩阵的行数
                         n,                     // 矩阵的列数
                         dA_array,              // 指向包含批次矩阵的设备内存数组的指针
                         ldda,                  // 每个矩阵的主设备存储区域的第一个维度大小
                         ipiv_array,            // 指向包含批次矩阵的主设备存储区域的指针
                         info_array,            // 指向包含每个矩阵的返回状态的设备指针
                         batchsize,             // 批处理中矩阵的数量
                         magma_queue.get_queue());// 与 MAGMAQueue 对象关联的 CUDA 队列
    // 检查 CUDA 函数调用期间是否发生错误
    AT_CUDA_CHECK(cudaGetLastError());
template<>
void magmaLuBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库的批量 LU 分解函数，对 float 类型的矩阵数组进行操作
  magma_sgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库的批量 LU 分解函数，对 complex<double> 类型的矩阵数组进行操作
  magma_zgetrf_batched(m, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库的批量 LU 分解函数，对 complex<float> 类型的矩阵数组进行操作
  magma_cgetrf_batched(m, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    magma_int_t* info) {
  // 同步 CUDA 流并调用 MAGMA 库的无主元 LU 分解函数，对 double 类型的矩阵进行操作
  MagmaStreamSyncGuard guard;
  magma_dgetrf_nopiv_gpu(m, n, dA, ldda, info);
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    magma_int_t* info) {
  // 同步 CUDA 流并调用 MAGMA 库的无主元 LU 分解函数，对 float 类型的矩阵进行操作
  MagmaStreamSyncGuard guard;
  magma_sgetrf_nopiv_gpu(m, n, dA, ldda, info);
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    magma_int_t* info) {
  // 同步 CUDA 流并调用 MAGMA 库的无主元 LU 分解函数，对 complex<double> 类型的矩阵进行操作
  MagmaStreamSyncGuard guard;
  magma_zgetrf_nopiv_gpu(m, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, info);
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    magma_int_t* info) {
  // 同步 CUDA 流并调用 MAGMA 库的无主元 LU 分解函数，对 complex<float> 类型的矩阵进行操作
  MagmaStreamSyncGuard guard;
  magma_cgetrf_nopiv_gpu(m, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, info);
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库的批量无主元 LU 分解函数，对 double 类型的矩阵数组进行操作
  magma_dgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库的批量无主元 LU 分解函数，对 float 类型的矩阵数组进行操作
  magma_sgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否产生错误
  AT_CUDA_CHECK(cudaGetLastError());
}
    // 调用 magma_sgetrf_nopiv_batched 函数执行批量的无主元 LU 分解操作
    // 参数说明：
    //   - m: 矩阵的行数
    //   - n: 矩阵的列数
    //   - dA_array: 指向设备内存中包含多个矩阵的数组的指针
    //   - ldda: 每个矩阵的 leading dimension (每列中相邻元素间的跨度)
    //   - info_array: 存储每个矩阵 LU 分解结果的信息数组
    //   - batchsize: 批处理中矩阵的数量
    //   - magma_queue.get_queue(): 返回与 MAGMAQueue 对象关联的 CUDA 队列的句柄
    magma_sgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
    
    // 检查上一个 CUDA 操作的错误状态
    AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库中的批量无主元 LU 分解函数，处理双精度复数类型的输入数据
  magma_zgetrf_nopiv_batched(m, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  // 检查 CUDA 函数执行是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库中的批量无主元 LU 分解函数，处理单精度复数类型的输入数据
  magma_cgetrf_nopiv_batched(m, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  // 检查 CUDA 函数执行是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda,
    double* dB, magma_int_t lddb, magma_int_t* info) {
  // 创建一个同步对象，确保 CUDA 流同步
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 库中的 Cholesky 解函数，处理双精度浮点数类型的输入数据
  magma_dpotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info);
  // 检查 CUDA 函数执行是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda,
    float* dB, magma_int_t lddb, magma_int_t* info) {
  // 创建一个同步对象，确保 CUDA 流同步
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 库中的 Cholesky 解函数，处理单精度浮点数类型的输入数据
  magma_spotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info);
  // 检查 CUDA 函数执行是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda,
    c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info) {
  // 创建一个同步对象，确保 CUDA 流同步
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 库中的 Cholesky 解函数，处理双精度复数类型的输入数据
  magma_zpotrs_gpu(uplo, n, nrhs,
    reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
    reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  // 检查 CUDA 函数执行是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda,
    c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info) {
  // 创建一个同步对象，确保 CUDA 流同步
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 库中的 Cholesky 解函数，处理单精度复数类型的输入数据
  magma_cpotrs_gpu(uplo, n, nrhs,
    reinterpret_cast<magmaFloatComplex*>(dA), ldda,
    reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  // 检查 CUDA 函数执行是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    double** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库中的批量 Cholesky 解函数，处理双精度浮点数类型的输入数据
  info = magma_dpotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  // 检查 CUDA 函数执行是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    float** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    // 调用 MAGMA 库的批量解线性方程组函数 magma_spotrs_batched，解决方程组 A_i * X_i = B_i，其中：
    // - uplo: 上三角或下三角部分的存储模式
    // - n: 每个线性方程组的阶数
    // - nrhs: 每个线性方程组的右侧向量数量
    // - dA_array: 包含 A_i 矩阵的指针数组
    // - ldda: A_i 矩阵在内存中的列跨度
    // - dB_array: 包含 B_i 向量的指针数组，存储解 X_i
    // - lddb: B_i 向量在内存中的列跨度
    // - batchsize: 线性方程组的批处理数量
    // - magma_queue.get_queue(): 获取用于异步操作的 MAGMA 队列
    info = magma_spotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());

    // 检查 CUDA 操作是否出错
    AT_CUDA_CHECK(cudaGetLastError());
template<>
void magmaCholeskySolveBatched<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<double>** dA_array, magma_int_t ldda,
    c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库函数 magma_zpotrs_batched 执行解线性方程组的批处理操作
  info = magma_zpotrs_batched(uplo, n, nrhs,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  // 检查并处理 CUDA 错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库函数 magma_cpotrs_batched 执行解线性方程组的批处理操作
  info = magma_cpotrs_batched(uplo, n, nrhs,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  // 检查并处理 CUDA 错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<double>(
    magma_uplo_t uplo, magma_int_t n, double* dA,
    magma_int_t ldda, magma_int_t* info) {
  // 创建 MagmaStreamSyncGuard 对象，确保操作在正确的 CUDA 流中进行
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 库函数 magma_dpotrf_gpu 执行双精度 Cholesky 分解
  magma_dpotrf_gpu(uplo, n, dA, ldda, info);
  // 检查并处理 CUDA 错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<float>(
    magma_uplo_t uplo, magma_int_t n, float* dA,
    magma_int_t ldda, magma_int_t* info) {
  // 创建 MagmaStreamSyncGuard 对象，确保操作在正确的 CUDA 流中进行
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 库函数 magma_spotrf_gpu 执行单精度 Cholesky 分解
  magma_spotrf_gpu(uplo, n, dA, ldda, info);
  // 检查并处理 CUDA 错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<double>* dA,
    magma_int_t ldda, magma_int_t* info) {
  // 创建 MagmaStreamSyncGuard 对象，确保操作在正确的 CUDA 流中进行
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 库函数 magma_zpotrf_gpu 执行复数双精度 Cholesky 分解
  magma_zpotrf_gpu(uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, info);
  // 检查并处理 CUDA 错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<float>* dA,
    magma_int_t ldda, magma_int_t* info) {
  // 创建 MagmaStreamSyncGuard 对象，确保操作在正确的 CUDA 流中进行
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 库函数 magma_cpotrf_gpu 执行复数单精度 Cholesky 分解
  magma_cpotrf_gpu(uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, info);
  // 检查并处理 CUDA 错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<double>(
    magma_uplo_t uplo, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库函数 magma_dpotrf_batched 执行双精度 Cholesky 批处理操作
  magma_dpotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  // 检查并处理 CUDA 错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<float>(
    magma_uplo_t uplo, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 库函数 magma_spotrf_batched 执行单精度 Cholesky 批处理操作
  magma_spotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  // 检查并处理 CUDA 错误
  AT_CUDA_CHECK(cudaGetLastError());
}
    # 调用 MAGMA 库中的批量 Cholesky 分解函数 magma_zpotrf_batched
    # 这里 magma_zpotrf_batched 是 MAGMA 库中进行批量 Cholesky 分解的函数
    # uplo: Cholesky 分解的上三角或下三角存储选项
    # n: 矩阵的阶数
    # reinterpret_cast<magmaDoubleComplex**>(dA_array): 将输入的复数矩阵数组转换为 MAGMA 库所需的类型
    # ldda: 矩阵 dA_array 中每个矩阵的 leading dimension
    # info_array: 输出数组，用于存储每个矩阵 Cholesky 分解的状态信息
    # batchsize: 批量操作中矩阵的数量
    # magma_queue.get_queue(): 获取 MAGMA 库中的队列，用于执行操作
    magma_zpotrf_batched(uplo, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());

    # 检查 CUDA 操作是否出错
    AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 的批量复数浮点数 Cholesky 分解函数
  magma_cpotrf_batched(uplo, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否成功
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<double>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    double** dA_array, magma_int_t ldda, double** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 的批量双精度浮点数三角求解函数
  magmablas_dtrsm_batched(side, uplo, trans, diag, m, n, 1, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否成功
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<float>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t ldda, float** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  // 调用 MAGMA 的批量单精度浮点数三角求解函数
  magmablas_strsm_batched(side, uplo, trans, diag, m, n, 1, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否成功
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<c10::complex<double>>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<double>** dA_array, magma_int_t ldda, c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  // 定义复数 alpha 为 {1, 0}
  magmaDoubleComplex alpha({1, 0});
  // 调用 MAGMA 的批量双精度复数三角求解函数
  magmablas_ztrsm_batched(side, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否成功
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<c10::complex<float>>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<float>** dA_array, magma_int_t ldda, c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  // 定义复数 alpha 为 {1, 0}
  magmaFloatComplex alpha({1, 0});
  // 调用 MAGMA 的批量单精度复数三角求解函数
  magmablas_ctrsm_batched(side, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否成功
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
inline magma_int_t magmaGeqrfOptimalBlocksize<double>(magma_int_t m, magma_int_t n) {
  // 返回双精度浮点数 GEQRF 算法的最优块大小
  return magma_get_dgeqrf_nb(m, n);
}

template<>
inline magma_int_t magmaGeqrfOptimalBlocksize<float>(magma_int_t m, magma_int_t n) {
  // 返回单精度浮点数 GEQRF 算法的最优块大小
  return magma_get_sgeqrf_nb(m, n);
}

template <>
inline magma_int_t magmaGeqrfOptimalBlocksize<c10::complex<double>>(magma_int_t m, magma_int_t n) {
    magma_int_t m,
    magma_int_t n) {

这部分代码定义了一个名为 `magma_get_zgeqrf_nb` 的函数，它接受两个参数 `m` 和 `n`，这两个参数的类型是 `magma_int_t`。


  return magma_get_zgeqrf_nb(m, n);

该行代码调用了名为 `magma_get_zgeqrf_nb` 的函数，并返回其结果。函数接受参数 `m` 和 `n`，然后返回一个值，该值被直接返回给调用者。
template <>
inline magma_int_t magmaGeqrfOptimalBlocksize<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n) {
  // 返回适用于 c10::complex<float> 类型的 magma_geqrf_nb 函数的最佳块大小
  return magma_get_cgeqrf_nb(m, n);
}

template<>
void magmaGeqrf<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    double* tau, double* dT, magma_int_t* info, bool is_v2) {
  MagmaStreamSyncGuard guard;
  // 如果不是版本 2，调用双精度双重矩阵 QR 分解的 GPU 实现
  if (!is_v2) {
    magma_dgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  } else {
    // 否则调用双精度双重矩阵 QR 分解的第二种 GPU 实现
    magma_dgeqrf2_gpu(m, n, dA, ldda, tau, info);
  }
  // 检查 CUDA 函数调用是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGeqrf<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    float* tau, float* dT, magma_int_t* info, bool is_v2) {
  MagmaStreamSyncGuard guard;
  // 如果不是版本 2，调用单精度矩阵 QR 分解的 GPU 实现
  if (!is_v2) {
    magma_sgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  } else {
    // 否则调用单精度矩阵 QR 分解的第二种 GPU 实现
    magma_sgeqrf2_gpu(m, n, dA, ldda, tau, info);
  }
  // 检查 CUDA 函数调用是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGeqrf<c10::complex<double>>(
    magma_int_t m,
    magma_int_t n,
    c10::complex<double>* dA,
    magma_int_t ldda,
    c10::complex<double>* tau,
    c10::complex<double>* dT,
    magma_int_t* info,
    bool is_v2) {
  MagmaStreamSyncGuard guard;
  // 如果不是版本 2，调用双精度复数矩阵 QR 分解的 GPU 实现
  if (!is_v2) {
    magma_zgeqrf_gpu(
        m,
        n,
        reinterpret_cast<magmaDoubleComplex*>(dA),
        ldda,
        reinterpret_cast<magmaDoubleComplex*>(tau),
        reinterpret_cast<magmaDoubleComplex*>(dT),
        info);
  } else {
    // 否则调用双精度复数矩阵 QR 分解的第二种 GPU 实现
    magma_zgeqrf2_gpu(
        m,
        n,
        reinterpret_cast<magmaDoubleComplex*>(dA),
        ldda,
        reinterpret_cast<magmaDoubleComplex*>(tau),
        info);
  }
  // 检查 CUDA 函数调用是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGeqrf<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n,
    c10::complex<float>* dA,
    magma_int_t ldda,
    c10::complex<float>* tau,
    c10::complex<float>* dT,
    magma_int_t* info,
    bool is_v2) {
  MagmaStreamSyncGuard guard;
  // 如果不是版本 2，调用单精度复数矩阵 QR 分解的 GPU 实现
  if (!is_v2) {
    magma_cgeqrf_gpu(
        m,
        n,
        reinterpret_cast<magmaFloatComplex*>(dA),
        ldda,
        reinterpret_cast<magmaFloatComplex*>(tau),
        reinterpret_cast<magmaFloatComplex*>(dT),
        info);
  } else {
    // 否则调用单精度复数矩阵 QR 分解的第二种 GPU 实现
    magma_cgeqrf2_gpu(
        m,
        n,
        reinterpret_cast<magmaFloatComplex*>(dA),
        ldda,
        reinterpret_cast<magmaFloatComplex*>(tau),
        info);
  }
  // 检查 CUDA 函数调用是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<double>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, double* dA, magma_int_t ldda,
    double* w, double* wA, magma_int_t ldwa, double* work, magma_int_t lwork, double* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  (void)rwork;  // 未使用
  (void)lrwork;  // 未使用
  MagmaStreamSyncGuard guard;
  // 调用双精度对称矩阵特征值分解的 GPU 实现
  magma_dsyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
  // 检查 CUDA 函数调用是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<float>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, float* dA, magma_int_t ldda,
    float* w, float* wA, magma_int_t ldwa, float* work, magma_int_t lwork, float* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  (void)rwork;  // 未使用
  (void)lrwork;  // 未使用
  MagmaStreamSyncGuard guard;
  // 调用单精度对称矩阵特征值分解的 GPU 实现
  magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
  // 检查 CUDA 函数调用是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}
    // 忽略未使用的参数 rwork 和 lrwork
    (void)rwork;  // unused
    (void)lrwork;  // unused
    
    // 同步 CUDA 流以确保前面的 GPU 操作完成
    MagmaStreamSyncGuard guard;

    // 调用 GPU 上的对称矩阵特征值计算函数 magma_ssyevd_gpu
    // jobz: 指示是否计算特征向量
    // uplo: 指示矩阵的存储格式，上三角或下三角
    // n: 矩阵的阶数
    // dA: 存储矩阵的 GPU 设备内存地址
    // ldda: 矩阵 dA 的 leading dimension
    // w: 存储计算得到的特征值的数组
    // wA: 存储计算得到的特征向量（如果计算了）的数组
    // ldwa: wA 数组的 leading dimension
    // work: 存储工作空间的数组
    // lwork: 工作空间数组 work 的长度
    // iwork: 存储整型工作空间的数组
    // liwork: 整型工作空间数组 iwork 的长度
    // info: 返回操作的状态信息，成功为 0
    magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);

    // 检查 CUDA 操作是否出错
    AT_CUDA_CHECK(cudaGetLastError());


这段代码主要是调用了一个在 GPU 上执行的对称矩阵特征值计算函数 `magma_ssyevd_gpu`，并且在调用之前使用了一些 CUDA 相关的函数和宏来确保操作的正确性和效率。
}

template<>
void magmaSyevd<c10::complex<double>, double>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    double* w, c10::complex<double>* wA, magma_int_t ldwa, c10::complex<double>* work, magma_int_t lwork, double* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  // 调用MAGMA库函数进行复数双精度Hermitian矩阵特征值计算
  magma_zheevd_gpu(
      jobz, uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, w, reinterpret_cast<magmaDoubleComplex*>(wA),
      ldwa, reinterpret_cast<magmaDoubleComplex*>(work), lwork, rwork, lrwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError()); // 检查CUDA操作是否有错误
}

template<>
void magmaSyevd<c10::complex<float>, float>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    float* w, c10::complex<float>* wA, magma_int_t ldwa, c10::complex<float>* work, magma_int_t lwork, float* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  // 调用MAGMA库函数进行复数单精度Hermitian矩阵特征值计算
  magma_cheevd_gpu(
      jobz, uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, w, reinterpret_cast<magmaFloatComplex*>(wA),
      ldwa, reinterpret_cast<magmaFloatComplex*>(work), lwork, rwork, lrwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError()); // 检查CUDA操作是否有错误
}

template<>
void magmaEig<double>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    double *A, magma_int_t lda,
    double *w,
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    double *work, magma_int_t lwork,
    double *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  // 调用MAGMA库函数进行双精度通用矩阵特征值计算
  // magma_dgeev要求将输出数组分为实部和虚部两部分：wr和wi
  double *wr = w;
  double *wi = w + n;
  (void)rwork; // 未使用的参数
  magma_dgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError()); // 检查CUDA操作是否有错误
}

template<>
void magmaEig<float>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    float *A, magma_int_t lda,
    float *w,
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  // 调用MAGMA库函数进行单精度通用矩阵特征值计算
  // magma_sgeev要求将输出数组分为实部和虚部两部分：wr和wi
  float *wr = w;
  float *wi = w + n;
  (void)rwork; // 未使用的参数
  magma_sgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError()); // 检查CUDA操作是否有错误
}

template<>
void magmaEig<c10::complex<double>, double>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    c10::complex<double> *A, magma_int_t lda,
    c10::complex<double> *w,
    c10::complex<double> *VL, magma_int_t ldvl,
    c10::complex<double> *VR, magma_int_t ldvr,
    c10::complex<double> *work, magma_int_t lwork,
    double *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  // 调用MAGMA库函数进行复数双精度通用矩阵特征值计算
  magma_zgeev(jobvl, jobvr, n, reinterpret_cast<magmaDoubleComplex*>(A), lda,
              reinterpret_cast<magmaDoubleComplex*>(w), reinterpret_cast<magmaDoubleComplex*>(VL), ldvl,
              reinterpret_cast<magmaDoubleComplex*>(VR), ldvr, reinterpret_cast<magmaDoubleComplex*>(work), lwork,
              rwork, info);
}
    // 使用 MagmaStreamSyncGuard 对象自动同步 CUDA 流
    MagmaStreamSyncGuard guard;
    // 调用 magma_zgeev 函数执行特征值求解操作
    // jobvl 和 jobvr 控制是否计算左特征向量和右特征向量
    // n 是矩阵的阶数
    // A 是输入矩阵，需以复数形式给出，由于 CUDA 所需，需要进行类型转换
    // lda 是矩阵 A 的列数
    // w 是输出的特征值数组，以复数形式给出
    // VL 和 VR 是输出的左右特征向量矩阵，以复数形式给出
    // ldvl 和 ldvr 分别是 VL 和 VR 矩阵的列数
    // work 是工作空间数组，用于内部计算，以复数形式给出
    // lwork 是工作空间数组的长度
    // rwork 是实数工作空间数组，用于内部计算
    // info 是输出的状态指示器，指示函数执行是否成功
    magma_zgeev(jobvl, jobvr, n,
                reinterpret_cast<magmaDoubleComplex*>(A), lda,
                reinterpret_cast<magmaDoubleComplex*>(w),
                reinterpret_cast<magmaDoubleComplex*>(VL), ldvl,
                reinterpret_cast<magmaDoubleComplex*>(VR), ldvr,
                reinterpret_cast<magmaDoubleComplex*>(work), lwork,
                rwork, info);
    // 检查 CUDA 操作是否有错误
    AT_CUDA_CHECK(cudaGetLastError());
template<>
void magmaEig<c10::complex<float>, float>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    c10::complex<float> *A, magma_int_t lda,
    c10::complex<float> *w,
    c10::complex<float> *VL, magma_int_t ldvl,
    c10::complex<float> *VR, magma_int_t ldvr,
    c10::complex<float> *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;  // 创建一个Magma流同步守卫对象，确保CUDA流同步
  // 调用magma库中的cgeev函数进行复杂矩阵的特征值计算
  magma_cgeev(jobvl, jobvr, n,
         reinterpret_cast<magmaFloatComplex*>(A), lda,
         reinterpret_cast<magmaFloatComplex*>(w),
         reinterpret_cast<magmaFloatComplex*>(VL), ldvl,
         reinterpret_cast<magmaFloatComplex*>(VR), ldvr,
         reinterpret_cast<magmaFloatComplex*>(work), lwork,
         rwork, info);
  AT_CUDA_CHECK(cudaGetLastError());  // 检查CUDA最后的错误状态
}

template<>
void magmaSvd<double>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, double* A,
    magma_int_t lda, double* s, double* U, magma_int_t ldu,
    double* Vh, magma_int_t ldvh, double* work, magma_int_t lwork,
    double *rwork, magma_int_t* iwork, magma_int_t* info) {
  (void)rwork; // unused  // 不使用rwork，即未使用的参数
  MagmaStreamSyncGuard guard;  // 创建一个Magma流同步守卫对象，确保CUDA流同步
  // 调用magma库中的dgesdd函数进行双精度矩阵的奇异值分解
  magma_dgesdd(jobz, m, n, A, lda, s, U, ldu, Vh, ldvh, work, lwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());  // 检查CUDA最后的错误状态
}

template<>
void magmaSvd<float>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, float* A,
    magma_int_t lda, float* s, float* U, magma_int_t ldu,
    float* Vh, magma_int_t ldvh, float* work, magma_int_t lwork,
    float* rwork, magma_int_t* iwork, magma_int_t* info) {
  (void)rwork; // unused  // 不使用rwork，即未使用的参数
  MagmaStreamSyncGuard guard;  // 创建一个Magma流同步守卫对象，确保CUDA流同步
  // 调用magma库中的sgesdd函数进行单精度矩阵的奇异值分解
  magma_sgesdd(jobz, m, n, A, lda, s, U, ldu, Vh, ldvh, work, lwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());  // 检查CUDA最后的错误状态
}

template<>
void magmaSvd<c10::complex<float>, float>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, c10::complex<float>* A,
    magma_int_t lda, float* s, c10::complex<float>* U, magma_int_t ldu,
    c10::complex<float>* Vh, magma_int_t ldvh, c10::complex<float>* work, magma_int_t lwork,
    float *rwork, magma_int_t* iwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;  // 创建一个Magma流同步守卫对象，确保CUDA流同步
  // 调用magma库中的cgesdd函数进行复杂单精度矩阵的奇异值分解
  magma_cgesdd(jobz, m, n, reinterpret_cast<magmaFloatComplex*>(A), lda, s,
                reinterpret_cast<magmaFloatComplex*>(U), ldu,
                reinterpret_cast<magmaFloatComplex*>(Vh), ldvh,
                reinterpret_cast<magmaFloatComplex*>(work), lwork,
                rwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());  // 检查CUDA最后的错误状态
}

template<>
void magmaSvd<c10::complex<double>, double>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, c10::complex<double>* A,
    magma_int_t lda, double* s, c10::complex<double>* U, magma_int_t ldu,
    c10::complex<double>* Vh, magma_int_t ldvh, c10::complex<double>* work, magma_int_t lwork,
    double *rwork, magma_int_t* iwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;  // 创建一个Magma流同步守卫对象，确保CUDA流同步
  // 调用magma库中的zgesdd函数进行复杂双精度矩阵的奇异值分解
  magma_zgesdd(jobz, m, n, reinterpret_cast<magmaDoubleComplex*>(A), lda, s,
                reinterpret_cast<magmaDoubleComplex*>(U), ldu,
                reinterpret_cast<magmaDoubleComplex*>(Vh), ldvh,
                reinterpret_cast<magmaDoubleComplex*>(work), lwork,
                rwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());  // 检查CUDA最后的错误状态
}
    // 创建一个 MagmaStreamSyncGuard 对象，用于同步 CUDA 流
    MagmaStreamSyncGuard guard;
    // 调用 magma_zgesdd 函数执行奇异值分解操作
    // jobz: 指示是否计算 U 和 Vh，m 和 n 分别是矩阵 A 的行数和列数
    // A: 待分解的复数矩阵的指针
    // lda: A 矩阵的 leading dimension
    // s: 奇异值的数组
    // U: 左奇异向量矩阵的指针
    // ldu: U 矩阵的 leading dimension
    // Vh: 右奇异向量矩阵的指针
    // ldvh: Vh 矩阵的 leading dimension
    // work: 工作空间的指针
    // lwork: 工作空间的长度
    // rwork: 实数工作数组的指针
    // iwork: 整数工作数组的指针
    // info: 返回执行状态信息的指针
    magma_zgesdd(jobz, m, n, reinterpret_cast<magmaDoubleComplex*>(A), lda, s,
                  reinterpret_cast<magmaDoubleComplex*>(U), ldu,
                  reinterpret_cast<magmaDoubleComplex*>(Vh), ldvh,
                  reinterpret_cast<magmaDoubleComplex*>(work), lwork,
                  rwork, iwork, info);
    // 检查并报告 CUDA 函数调用期间的错误状态
    AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<double>(
    magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda, magma_int_t* ipiv,
    double* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  // 同步 CUDA 流
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 函数执行双精度 LU 解算
  magma_dgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  // 检查 CUDA 操作是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<float>(
    magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda, magma_int_t* ipiv,
    float* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  // 同步 CUDA 流
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 函数执行单精度 LU 解算
  magma_sgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  // 检查 CUDA 操作是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda, magma_int_t* ipiv,
    c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  // 同步 CUDA 流
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 函数执行双精度复数 LU 解算
  magma_zgetrs_gpu(trans, n, nrhs, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  // 检查 CUDA 操作是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda, magma_int_t* ipiv,
    c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  // 同步 CUDA 流
  MagmaStreamSyncGuard guard;
  // 调用 MAGMA 函数执行单精度复数 LU 解算
  magma_cgetrs_gpu(trans, n, nrhs, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  // 检查 CUDA 操作是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    double** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
  // 调用 MAGMA 批量函数执行双精度 LU 解算
  info = magma_dgetrs_batched(trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    float** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
  // 调用 MAGMA 批量函数执行单精度 LU 解算
  info = magma_sgetrs_batched(trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchsize, magma_queue.get_queue());
  // 检查 CUDA 操作是否有错误
  AT_CUDA_CHECK(cudaGetLastError());
}


注释：
    # 调用 MAGMA 库中的批量解线性方程组函数 magma_zgetrs_batched
    # 参数 trans: 指定解的转置形式
    # 参数 n: 方程组的维度
    # 参数 nrhs: 右侧矩阵的列数
    # 参数 dA_array: 存储待解矩阵数组的指针数组，类型为 magmaDoubleComplex**
    # 参数 ldda: 待解矩阵在设备上的 leading dimension
    # 参数 dipiv_array: 存储 pivot 数组的指针数组
    # 参数 dB_array: 存储右侧矩阵数组的指针数组，类型为 magmaDoubleComplex**
    # 参数 lddb: 右侧矩阵在设备上的 leading dimension
    # 参数 batchsize: 批量处理的矩阵个数
    # 参数 magma_queue.get_queue(): 获取 MAGMA 队列对象用于 GPU 加速计算
    info = magma_zgetrs_batched(trans, n, nrhs, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue())
    # 检查 CUDA 操作是否发生错误
    AT_CUDA_CHECK(cudaGetLastError())
}

template<>
void magmaLuSolveBatched<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
 info = magma_cgetrs_batched(trans, n, nrhs, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
 AT_CUDA_CHECK(cudaGetLastError());
}


# 特化模板函数，解决复数浮点数类型的 LU 分解批处理求解问题
template<>
void magmaLuSolveBatched<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
 info = magma_cgetrs_batched(trans, n, nrhs, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
 AT_CUDA_CHECK(cudaGetLastError());
}


template<>
void magmaGels<float>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    float* dA, magma_int_t ldda, float* dB, magma_int_t lddb,
    float* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgels_gpu(trans, m, n, nrhs,
      dA, ldda, dB, lddb,
      hwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}


# 特化模板函数，解决单精度浮点数类型的 Gels 问题
template<>
void magmaGels<float>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    float* dA, magma_int_t ldda, float* dB, magma_int_t lddb,
    float* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgels_gpu(trans, m, n, nrhs,
      dA, ldda, dB, lddb,
      hwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}


template<>
void magmaGels<double>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    double* dA, magma_int_t ldda, double* dB, magma_int_t lddb,
    double* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgels_gpu(trans, m, n, nrhs,
      dA, ldda, dB, lddb,
      hwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}


# 特化模板函数，解决双精度浮点数类型的 Gels 问题
template<>
void magmaGels<double>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    double* dA, magma_int_t ldda, double* dB, magma_int_t lddb,
    double* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgels_gpu(trans, m, n, nrhs,
      dA, ldda, dB, lddb,
      hwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}


template<>
void magmaGels<c10::complex<float>>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    c10::complex<float>* dA, magma_int_t ldda, c10::complex<float>* dB, magma_int_t lddb,
    c10::complex<float>* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgels_gpu(trans, m, n, nrhs,
      reinterpret_cast<magmaFloatComplex*>(dA), ldda,
      reinterpret_cast<magmaFloatComplex*>(dB), lddb,
      reinterpret_cast<magmaFloatComplex*>(hwork), lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}


# 特化模板函数，解决复数单精度浮点数类型的 Gels 问题
template<>
void magmaGels<c10::complex<float>>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    c10::complex<float>* dA, magma_int_t ldda, c10::complex<float>* dB, magma_int_t lddb,
    c10::complex<float>* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgels_gpu(trans, m, n, nrhs,
      reinterpret_cast<magmaFloatComplex*>(dA), ldda,
      reinterpret_cast<magmaFloatComplex*>(dB), lddb,
      reinterpret_cast<magmaFloatComplex*>(hwork), lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}


template<>
void magmaGels<c10::complex<double>>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    c10::complex<double>* dA, magma_int_t ldda, c10::complex<double>* dB, magma_int_t lddb,
    c10::complex<double>* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgels_gpu(trans, m, n, nrhs,
      reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
      reinterpret_cast<magmaDoubleComplex*>(dB), lddb,
      reinterpret_cast<magmaDoubleComplex*>(hwork), lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}


# 特化模板函数，解决复数双精度浮点数类型的 Gels 问题
template<>
void magmaGels<c10::complex<double>>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    c10::complex<double>* dA, magma_int_t ldda, c10::complex<double>* dB, magma_int_t lddb,
    c10::complex<double>* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgels_gpu(trans, m, n, nrhs,
      reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
      reinterpret_cast<magmaDoubleComplex*>(dB), lddb,
      reinterpret_cast<magmaDoubleComplex*>(hwork), lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}


namespace {


# 匿名命名空间，用于定义仅在当前编译单元可见的符号和函数
namespace {
/*
  MAGMA can return errors both as a return value and in the info argument.
  The return value and info should always be identical.
  In general, the meaning is as given in this table.
  Predefined error codes are large negative numbers. Using the symbolic
  constants below is preferred, but the numeric values can be found in
  include/magma_types.h.

  Info                       |  Description
  -----------                |  -----------
  info = 0 (MAGMA_SUCCESS)   |  Successful exit
  info < 0, but small        |  For info = -i, the i-th argument had an illegal value
  info > 0                   |  Function-specific error such as singular matrix
  MAGMA_ERR_DEVICE_ALLOC     |  Could not allocate GPU device memory
  MAGMA_ERR_HOST_ALLOC       |  Could not allocate CPU host memory
  MAGMA_ERR_ILLEGAL_VALUE    |  An argument had an illegal value (deprecated; instead it should return -i to say the i-th argument was bad)
  MAGMA_ERR_INVALID_PTR      |  Can't free pointer
  MAGMA_ERR_NOT_IMPLEMENTED  |  Function or option not implemented
  MAGMA_ERR_NOT_SUPPORTED    |  Function or option not supported on the current architecture
*/
void checkMagmaInternalError(magma_int_t info, const std::string& magma_function_name) {
  // 如果 info > 0，说明出现特定于函数的错误，此时不进行任何操作
  TORCH_CHECK(info >= 0,
      "MAGMA error: ",
      magma_strerror(info),
      ", info = ", info,
      ", when calling ", magma_function_name);
}

magma_trans_t to_magma(TransposeType trans) {
  switch (trans) {
    case TransposeType::NoTranspose: return MagmaNoTrans;
    case TransposeType::Transpose: return MagmaTrans;
    case TransposeType::ConjTranspose: return MagmaConjTrans;
  }
  // 如果转置类型不在预期范围内，则报错
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
} // anonymous namespace
#endif // AT_MAGMA_ENABLED()

#define ALLOCATE_ARRAY(name, type, size) \
  auto storage_##name = pin_memory<type>(size); \
  name = static_cast<type*>(storage_##name.mutable_data());

namespace {

template <typename scalar_t>
void apply_ldl_factor_magma(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& info,
    bool upper) {
#if !AT_MAGMA_ENABLED()
  // 如果未启用 MAGMA 库，则抛出错误
  TORCH_CHECK(
      false,
      "torch.linalg.ldl_factor: MAGMA library not found in "
      "compilation. Please rebuild with MAGMA.");
#else
  // 获取张量 A 的批量大小
  auto batch_size = batchCount(A);
  // 将张量 A 的倒数第二维度大小转换为 magma_int_t 类型
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  // 将张量 A 的最后一维度步长转换为 magma_int_t 类型
  magma_int_t leading_dim = magma_int_cast(A.stride(-1), "A.stride(-1)");
  // 根据 upper 变量选择上三角还是下三角
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  // 计算张量 A 的第三维度的步长，如果维度小于等于2，则步长为0
  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  // 计算张量 pivots 的第二维度的步长，如果维度小于等于1，则步长为0
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  // 获取张量 A 的可变数据指针
  auto a_data = A.mutable_data_ptr<scalar_t>();
  // 创建与张量 pivots 相同形状的空张量，在 CPU 上分配内存并固定
  Tensor pivots_cpu =
      at::empty_like(pivots, pivots.options().device(kCPU).pinned_memory(true));
  // 获取张量 pivots_cpu 的可变数据指针
  auto pivots_data = pivots_cpu.mutable_data_ptr<magma_int_t>();
  // 创建与张量 info 相同形状的空张量，在 CPU 上分配内存并固定
  Tensor info_cpu =
      at::empty_like(info, info.options().device(kCPU).pinned_memory(true));
  // 获取张量 info_cpu 的可变数据指针
  auto info_data = info_cpu.mutable_data_ptr<magma_int_t>();

  // 遍历批量中的每个元素
  for (const auto i : c10::irange(batch_size)) {
    // 计算当前批次中张量 a_data 的起始指针
    scalar_t* a_working_ptr = &a_data[i * a_stride];
    // 计算当前批次中张量 pivots_data 的起始指针
    magma_int_t* pivots_working_ptr = &pivots_data[i * pivots_stride];
    // 计算当前批次中张量 info_data 的起始指针
    magma_int_t* info_working_ptr = &info_data[i];
    // 调用 magmaLdlHermitian 函数执行 LDL 分解
    magmaLdlHermitian<scalar_t>(
        uplo,
        n,
        a_working_ptr,
        leading_dim,
        pivots_working_ptr,
        info_working_ptr);
  }
  // 将 pivots_cpu 的数据复制回 pivots 张量
  pivots.copy_(pivots_cpu);
  // 将 info_cpu 的数据复制回 info 张量
  info.copy_(info_cpu);
#endif
}

void ldl_factor_magma(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  // 检查是否为复数张量且要求为 Hermitian 矩阵
  if (LD.is_complex()) {
    TORCH_CHECK(
        hermitian,
        "torch.linalg.ldl_factor: complex tensors with hermitian=False flag are not supported with MAGMA backend. ",
        "Currently preferred backend is ",
        at::globalContext().linalgPreferredBackend(),
        ", please set 'default' or 'cusolver' backend with torch.backends.cuda.preferred_linalg_library");
  }
  // 根据 LD 张量的数据类型分发到 apply_ldl_factor_magma 函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_factor_magma", [&] {
        apply_ldl_factor_magma<scalar_t>(LD, pivots, info, upper);
      });
}

void ldl_factor_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  // 获取全局上下文中首选的线性代数后端
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  // 根据不同的线性代数后端选择不同的分解函数
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
      return ldl_factor_cusolver(
          LD, pivots, info, upper, hermitian);
    case at::LinalgBackend::Magma:
      return ldl_factor_magma(LD, pivots, info, upper, hermitian);
    default:
    // 默认情况下，如果 Cusolver 可用，则使用 Cusolver，否则使用 Magma
    // 如果同时支持 Cusolver 和 Magma 2.5.4+，且要求复数输入且 Hermitian=true，则调用 Magma
#ifdef USE_LINALG_SOLVER
#if AT_MAGMA_ENABLED() && (AT_MAGMA_VERSION >= 254)
      if (LD.is_complex() && hermitian) {
        return ldl_factor_magma(
            LD, pivots, info, upper, hermitian);
      }
#endif
      return ldl_factor_cusolver(
          LD, pivots, info, upper, hermitian);
#else
      return ldl_factor_magma(LD, pivots, info, upper, hermitian);
#endif
  }
}

void ldl_solve_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  // 如果LD是复数类型的张量，则需要检查hermitian标志是否为假，因为在CUDA上不支持带有hermitian=True标志的复数张量。
  if (LD.is_complex()) {
    TORCH_CHECK(
        !hermitian,
        "torch.linalg.ldl_solve: complex tensors with hermitian=True flag are not supported on CUDA.");
  }

  // 调用ldl_solve_cusolver函数，使用CUSOLVER进行LDL分解求解
  ldl_solve_cusolver(LD, pivots, B, upper);
}
}

} // anonymous namespace

// 注册 CUDA 调度函数，将 ldl_factor_stub 映射到 ldl_factor_kernel
REGISTER_CUDA_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
// 注册 CUDA 调度函数，将 ldl_solve_stub 映射到 ldl_solve_kernel
REGISTER_CUDA_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 定义一个模板函数 apply_cholesky_solve，处理 Cholesky 分解求解的操作
template <typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, int64_t& info) {
#if !AT_MAGMA_ENABLED()
// 如果未启用 MAGMA 库，则抛出错误信息
AT_ERROR("cholesky_solve: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  // 根据 upper 参数设置 uplo 变量，用于指示上三角或下三角
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  // 获取 A 和 b 的数据指针
  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  // 获取 A 和 b 的维度信息
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t lda = std::max<magma_int_t>(1, n);
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  // 定义一个临时信息变量
  int info_tmp = 0;
  // 如果 b 是二维张量
  if (b.dim() == 2) {
    // 调用单个 Cholesky 求解函数
    magmaCholeskySolve<scalar_t>(uplo, n, nrhs, A_data, lda,
                                 b_data, lda, &info_tmp);
    // 将临时信息传递给 info 变量
    info = info_tmp;
  } else {
    // 获取 A 和 b 的矩阵步长
    auto A_mat_stride = matrixStride(A);
    auto b_mat_stride = matrixStride(b);
    // 获取批处理大小
    magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");

    scalar_t** A_array;
    scalar_t** b_array;

    // 分配 A_array 和 b_array 的内存空间
    ALLOCATE_ARRAY(A_array, scalar_t*, batch_size);
    ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

    // 设置创建的数组
    for (int64_t i = 0; i < batch_size; i++) {
      A_array[i] = &A_data[i * A_mat_stride];
      b_array[i] = &b_data[i * b_mat_stride];
    }

    // 在当前设备上创建 MAGMA 队列
    MAGMAQueue magma_queue(b.get_device());

    // 定义一个常量表示每批次的限制
    constexpr int64_t batch_limit = 65535;
    // 计算尽可能多的批次
    int64_t mini_batches = batch_size / batch_limit, mini_idx;
    for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
      // 当前批次的 A_array 和 b_array
      scalar_t** A_array_cur = &A_array[mini_idx];
      scalar_t** b_array_cur = &b_array[mini_idx];

      // 调用批处理 Cholesky 求解函数
      magmaCholeskySolveBatched<scalar_t>(
          uplo, n, nrhs, A_array_cur, lda, b_array_cur, lda,
          info_tmp, batch_limit, magma_queue);

      // 如果出现非零信息，则中断循环
      if (info_tmp != 0) {
        break;
      }
    }

    // 计算剩余的批次
    if (batch_size % batch_limit != 0 && info_tmp == 0) {
      // 调用剩余的批次 Cholesky 求解函数
      magmaCholeskySolveBatched<scalar_t>(
          uplo, n, nrhs, &A_array[mini_idx], lda, &b_array[mini_idx], lda,
          info_tmp, batch_size % batch_limit, magma_queue);
    }

    // 将临时信息传递给 info 变量
    info = info_tmp;
  }
#endif
}

// 定义 CUDA 版本的 cholesky_solve 辅助函数，调用上面的模板函数处理
Tensor _cholesky_solve_helper_cuda_magma(const Tensor& self, const Tensor& A, bool upper) {
  // 初始化 info 变量
  int64_t info = 0;
  // 复制输入张量 self 和 A，使用列主元素批处理拷贝
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  // 根据张量类型和操作名称分发到对应的 CUDA 实现
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_solve_cuda", [&]{
    # 使用 apply_cholesky_solve 函数对 self_working_copy 应用 Cholesky 分解求解方法
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, info);
    # 使用 TORCH_CHECK 来验证 Cholesky 分解求解过程中是否有无效参数，info 应为 0
    TORCH_CHECK(info == 0, "MAGMA cholesky_solve : invalid argument: ", -info);
    # 返回经 Cholesky 分解求解后的 self_working_copy 结果
    return self_working_copy;
// Todo: cusolverDn<T>potrsBatched only supports nrhs == 1 and does not have good performance.
//     Batched cholesky_solve is dispatched to magma.
// 定义 CUDA 下的 cholesky_solve 助手函数，用于解决 Cholesky 分解的求解问题
Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  // 获取全局上下文中的线性代数求解器首选后端
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    // 如果首选后端为 Cusolver，则调用 cusolver 版本的求解函数
    case at::LinalgBackend::Cusolver:
      return _cholesky_solve_helper_cuda_cusolver(self, A, upper);
    // 如果首选后端为 Magma，则调用 magma 版本的求解函数
    case at::LinalgBackend::Magma:
      return _cholesky_solve_helper_cuda_magma(self, A, upper);
    default:
      // 如果 batch 数为 1 或者不使用 Magma，则调用 cusolver 版本的求解函数
      if (batchCount(self) == 1 || !use_magma_) {
        return _cholesky_solve_helper_cuda_cusolver(self, A, upper);
      } else {
        // 否则调用 magma 版本的求解函数
        return _cholesky_solve_helper_cuda_magma(self, A, upper);
      }
  }
#else
  // 如果未定义 USE_LINALG_SOLVER 或者不在 CUDA 环境下，直接调用 magma 版本的求解函数
  return _cholesky_solve_helper_cuda_magma(self, A, upper);
#endif
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 模板函数，应用 Cholesky 分解到给定的 CUDA 张量上
template <typename scalar_t>
static void apply_cholesky(const Tensor& self, bool upper, const Tensor& info) {
#if !AT_MAGMA_ENABLED()
  // 如果未启用 MAGMA 支持，抛出错误信息
  TORCH_CHECK(
      false,
      "Calling torch.linalg.cholesky on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please use PyTorch built with MAGMA support.");
#else
  // 设置上三角或下三角
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  // 获取 self 张量的数据指针和大小
  auto self_data = self.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");
  auto lda = std::max<magma_int_t>(1, n);

  // 如果 self 张量为二维
  if (self.dim() == 2) {
    // magmaCholesky 需要在 CPU 上进行，所以这里定义一个 CPU 端的 info
    magma_int_t info_cpu = 0;
    // 调用 magmaCholesky 进行 Cholesky 分解
    magmaCholesky<scalar_t>(uplo, n, self_data, lda, &info_cpu);
    // 将结果填充到 info 张量中
    info.fill_(info_cpu);
  } else {
    // 对于高维度的 self 张量，确保 info 在 CUDA 上
    TORCH_INTERNAL_ASSERT(info.is_cuda());
    auto info_data = info.data_ptr<magma_int_t>();

    // magmaCholeskyBatched 只支持上三角，所以强制设置 uplo 为 MagmaLower
    uplo = MagmaLower;

    // 获取 self 张量的批处理数和矩阵步幅
    auto self_mat_stride = matrixStride(self);
    magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");

    scalar_t** self_array;

    // 为 self_array 分配内存
    ALLOCATE_ARRAY(self_array, scalar_t*, batch_size);

    // 设置 self_array 的数组
    for (int64_t i = 0; i < batch_size; i++) {
      self_array[i] = &self_data[i * self_mat_stride];
    }

    // 在当前设备上创建 MAGMAQueue 对象
    MAGMAQueue magma_queue(self.get_device());

    // 计算可能的最大批次数
    // 262140 是可以运行的最大矩阵批次大小，不违反最大内核配置
    // 对于复数输入，批处理限制为 65535（经验确定，参见 https://github.com/pytorch/pytorch/pull/47047#discussion_r516086923 了解更多信息）
    int64_t batch_limit = self.is_complex() ? 65535 : 262140;
    // 使用循环迭代处理批次数据，每次处理 batch_limit 个元素，直到处理完所有 batch_size 的元素
    for (int64_t mini_idx = 0; mini_idx < batch_size; mini_idx += batch_limit) {
      // 计算当前迭代中实际处理的元素数量，不能超过 batch_size
      int64_t nbatches = std::min(batch_limit, batch_size - mini_idx);
      // 获取当前要处理的 self_array 的子数组的指针
      scalar_t** self_array_cur = &self_array[mini_idx];
      // 获取当前要处理的 info_data 的子数组的指针
      magma_int_t* info_array_cur = &info_data[mini_idx];

      // 调用 magmaCholeskyBatched 函数进行批量的 Cholesky 分解计算
      magmaCholeskyBatched<scalar_t>(
        uplo, n, self_array_cur, lda, info_array_cur, nbatches, magma_queue);
    }
  }


这段代码的作用是使用循环来批量处理数据，每次处理 `batch_limit` 个元素，直到处理完所有 `batch_size` 的元素。
#endif
}

// 定义一个名为 cholesky_helper_magma 的静态函数，用于在 MAGMA 支持下执行 Cholesky 分解
// 参数 input: 输入的张量，表示需要进行 Cholesky 分解的对称正定矩阵
// 参数 upper: 布尔值，指示是否计算上三角矩阵的 Cholesky 分解
// 参数 info: 整数张量，用于存储每个矩阵的错误码
void cholesky_helper_magma(const Tensor& input, bool upper, const Tensor& info) {
  // 将结果张量初始化为输入张量
  Tensor result = input;
  
  // 如果输入张量的维度大于2
  if (input.dim() > 2) {
    // MAGMA 的批处理 Cholesky 操作存在一个 off-by-one 错误，可能导致 IMA
    // 根据 #cloneBatchedColumnMajor 函数的实现，在此基础上对输入进行了填充，利用 resize_as_ 方法保留了存储
    // 这样，如果 MAGMA 读取超出边界，仍然会是有效的用户内存
    result = at::empty(input.numel() + 1, input.options());
    result.resize_as_(input).transpose_(-2, -1);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.mT().is_contiguous());

    // 批处理 MAGMA 不支持 upper=true 的情况
    // 我们对输入进行转置和共轭操作作为一种解决方法
    result.copy_(upper ? input.mH() : input);
  }

  // 调度至对应的浮点数和复数类型，调用 apply_cholesky 函数进行 Cholesky 分解
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    input.scalar_type(), "cholesky_cuda", [&] {
      apply_cholesky<scalar_t>(result, upper, info);
    });

  // 如果输入张量的维度大于2
  if (input.dim() > 2) {
    // 如果 upper=true，则需要对结果张量进行转置和共轭操作
    // 因为 Cholesky 分解结果存储在下三角部分
    if (upper) {
      input.copy_(result.mH());
    } else {
      input.copy_(result);
    }
  }
}

// 定义一个名为 cholesky_kernel 的静态函数，用于根据不同后端选择执行 Cholesky 分解的实现
// 参数 input: 输入的张量，表示需要进行 Cholesky 分解的对称正定矩阵
// 参数 info: 整数张量，用于存储每个矩阵的错误码
// 参数 upper: 布尔值，指示是否计算上三角矩阵的 Cholesky 分解
static void cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  // 获取当前全局上下文的线性代数首选后端
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  
  // 根据首选后端选择执行对应的 Cholesky 分解辅助函数
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
      cholesky_helper_cusolver(input, upper, info);
      break;
    case at::LinalgBackend::Magma:
      cholesky_helper_magma(input, upper, info);
      break;
    default:
      // 如果输入张量的批量数为1，或者不使用 MAGMA 或者使用 cusolver_potrf_batched_
      // 则调用 cholesky_helper_cusolver 函数，否则调用 cholesky_helper_magma 函数
      if (batchCount(input) == 1 || !use_magma_ || use_cusolver_potrf_batched_) {
        cholesky_helper_cusolver(input, upper, info);
      } else {
        cholesky_helper_magma(input, upper, info);
      }
  }
#else
  // 如果未定义 USE_LINALG_SOLVER 或者定义了 USE_ROCM，则直接调用 cholesky_helper_magma 函数
  cholesky_helper_magma(input, upper, info);
#endif // USE_LINALG_SOLVER
}

// 注册 CUDA 的 cholesky_stub 函数，指定其对应的核函数为 cholesky_kernel
REGISTER_CUDA_DISPATCH(cholesky_stub, &cholesky_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
计算对称（Hermitian）正定矩阵 n × n 的逆矩阵，使用 Cholesky 分解器进行计算
这是一个原地运算，'input' 的内容会被覆盖
'infos' 是一个整数张量，包含批量输入中每个矩阵的错误码
MAGMA 要求 'infos' 存储在 CPU 内存中
更多信息请参见 MAGMA 的 POTRS 例程文档
*/
template <typename scalar_t>
static void apply_cholesky_inverse(Tensor& input, Tensor& infos, bool upper) {
#if !AT_MAGMA_ENABLED()
  // 如果未启用 MAGMA 编译，则输出错误信息
  TORCH_CHECK(false, "cholesky_inverse: MAGMA library not found in compilation. Please rebuild with MAGMA.");
#else
  // magmaCholeskyInverse (magma_dpotri_gpu) is slow because internally
  // it transfers data several times between GPU and CPU and calls lapack routine on CPU
  // using magmaCholeskySolveBatched is a lot faster
  // note that magmaCholeskySolve is also slow

  // 'input' is modified in-place we need to clone it and replace with a diagonal matrix
  // for apply_cholesky_solve
  // 克隆 'input' 并替换为对角矩阵，以供 apply_cholesky_solve 使用
  auto input_working_copy = cloneBatchedColumnMajor(input);

  // 'input' tensor has to be a batch of diagonal matrix
  // 将 'input' 张量填充为零
  input.fill_(0);
  // 将 'input' 张量的对角线元素填充为1
  input.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);

  Tensor result_u, input_u;
  if (input.dim() == 2) {
    // unsqueezing here so that the batched version is used
    // 在这里进行 unsqueeze 操作，以便使用批处理版本
    result_u = input.unsqueeze(0);
    input_u = input_working_copy.unsqueeze(0);
  } else {
    result_u = input;
    input_u = input_working_copy;
  }

  // magma's potrs_batched doesn't take matrix-wise array of ints as an 'info' argument
  // it returns a single 'magma_int_t'
  // if info = 0 the operation is successful, if info = -i, the i-th parameter had an illegal value.
  // magma 的 potrs_batched 不接受矩阵的整数数组作为 'info' 参数
  // 它返回一个单一的 'magma_int_t'
  // 如果 info = 0，则操作成功；如果 info = -i，则第 i 个参数具有非法值。
  int64_t info_tmp = 0;
  // 应用 cholesky_solve 算法对 result_u 进行求解，结果存储在 result_u 中，同时更新 info_tmp
  apply_cholesky_solve<scalar_t>(result_u, input_u, upper, info_tmp);
  // 将 infos 张量填充为 info_tmp 的值
  infos.fill_(info_tmp);
#endif
}

// This is a type dispatching helper function for 'apply_cholesky_inverse'
// 这是一个用于 'apply_cholesky_inverse' 的类型调度辅助函数
Tensor& cholesky_inverse_kernel_impl_magma(Tensor &result, Tensor& infos, bool upper) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "cholesky_inverse_out_cuda", [&]{
    // 调用 apply_cholesky_inverse 对 result 进行求逆操作，结果保存在 result 中，信息保存在 infos 中
    apply_cholesky_inverse<scalar_t>(result, infos, upper);
  });
  return result;
}

Tensor& cholesky_inverse_kernel_impl(Tensor &result, Tensor& infos, bool upper) {
  // This function calculates the inverse matrix in-place
  // result should be in column major order and contain matrices to invert
  // the content of result is overwritten by 'apply_cholesky_inverse'
  // 此函数就地计算逆矩阵
  // result 应以列主序排列，并包含要求逆的矩阵
  // result 的内容将被 'apply_cholesky_inverse' 覆盖
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
      // 使用 Cusolver 后端计算 cholesky_inverse
      return cholesky_inverse_kernel_impl_cusolver(result, infos, upper);
    case at::LinalgBackend::Magma:
      // 使用 Magma 后端计算 cholesky_inverse
      return cholesky_inverse_kernel_impl_magma(result, infos, upper);
    default:
      if (batchCount(result) == 1 ||
          !use_magma_) {
        // 如果 batchCount(result) 为 1 或者不使用 magma，则使用 Cusolver 后端计算 cholesky_inverse
        return cholesky_inverse_kernel_impl_cusolver(result, infos, upper);
      } else {
        // 否则使用 Magma 后端计算 cholesky_inverse
        return cholesky_inverse_kernel_impl_magma(result, infos, upper);
      }
  }
#else
  // 如果未定义 USE_LINALG_SOLVER 或者定义了 USE_ROCM，则使用 Magma 后端计算 cholesky_inverse
  return cholesky_inverse_kernel_impl_magma(result, infos, upper);
#endif
}

// Register CUDA dispatch for 'cholesky_inverse_stub' with 'cholesky_inverse_kernel_impl'
// 使用 'cholesky_inverse_kernel_impl' 注册 'cholesky_inverse_stub' 的 CUDA 分发

REGISTER_CUDA_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/*
  Computes the LU decomposition of a m×n matrix or batch of matrices in 'input' tensor.
  This is an in-place routine, content of 'input', 'pivots', and 'infos' is overwritten.
  This is a "looped" variant for calling single input MAGMA function on batched input.

  Args:
  * `input` - [in] the input matrix for LU decomposition
              [out] the LU decomposition
  * `pivots` - [out] the pivot indices
  * `infos` - [out] error codes, positive values indicate singular matrices
  * `compute_pivots` - controls whether LU is computed with or without pivoting

  For further details, please see the MAGMA documentation for magma_dgetrf_gpu.
*/
template <typename scalar_t>
static void apply_lu_factor_looped_magma(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
#if !AT_MAGMA_ENABLED()
  // This should never be thrown if the calling functions are correct.
  AT_ERROR("linalg.lu_factor: PyTorch was not compiled with MAGMA support.");
#else
  // magmaLu and magmaLuNoPiv require infos and pivots tensor to be on CPU
  // the data is later copied back to the appropriate output tensor
  Tensor infos_cpu = at::empty_like(infos, infos.options().device(kCPU).pinned_memory(true));

  // Retrieve pointers and sizes for input, infos, and batch processing
  auto input_data = input.data_ptr<scalar_t>();
  auto infos_data = infos_cpu.mutable_data_ptr<magma_int_t>();
  auto input_matrix_stride = matrixStride(input);
  auto pivots_stride = pivots.size(-1);
  auto batch_size = batchCount(input);
  magma_int_t m = magma_int_cast(input.size(-2), "m");
  magma_int_t n = magma_int_cast(input.size(-1), "n");
  auto leading_dimension = std::max<magma_int_t>(1, m);

  if (compute_pivots) {
    // Prepare CPU tensor for pivots if pivoting is required
    Tensor pivots_cpu = at::empty_like(pivots, pivots.options().device(kCPU).pinned_memory(true));
    auto pivots_data = pivots_cpu.mutable_data_ptr<magma_int_t>();

    // Perform LU decomposition with pivoting for each batch
    for (decltype(batch_size) i = 0; i < batch_size; i++) {
      scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
      int* pivots_working_ptr = &pivots_data[i * pivots_stride];
      int* infos_working_ptr = &infos_data[i];
      magmaLu<scalar_t>(m, n, input_working_ptr, leading_dimension, pivots_working_ptr, infos_working_ptr);
    }

    // Copy pivots results back to the original tensor
    pivots.copy_(pivots_cpu);
  } else {
    // Perform LU decomposition without pivoting for each batch
    for (decltype(batch_size) i = 0; i < batch_size; i++) {
      scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
      int* infos_working_ptr = &infos_data[i];
      magmaLuNoPiv<scalar_t>(m, n, input_working_ptr, leading_dimension, infos_working_ptr);
    }
  }

  // Copy infos results back to the original tensor
  infos.copy_(infos_cpu);
#endif
}
/*
  计算 'input' 张量中 m×n 矩阵或矩阵批次的 LU 分解。
  这是一个原地操作，'input'、'pivots' 和 'infos' 的内容会被覆盖。
  这是一个专门的批处理变体，预计对于小输入比“循环”版本更快。

  参数:
  * `input` - [输入] LU 分解的输入矩阵
             [输出] LU 分解后的结果
  * `pivots` - [输出] 枢轴索引
  * `infos` - [输出] 错误代码，正值表示奇异矩阵
  * `compute_pivots` - 控制是否使用枢轴计算 LU 分解

  更多细节，请参阅 MAGMA 文档中的 magma_dgetrf_batched。

*/
template <typename scalar_t>
static void apply_lu_factor_batched_magma(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
#if !AT_MAGMA_ENABLED()
  // 如果未启用 MAGMA 支持，则抛出错误
  TORCH_CHECK(
      false,
      "Calling linalg.lu_factor on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please rebuild with MAGMA.");
#else
  // 检查是否存在 MAGMA < 2.5.2 的版本中已知的 lu_factor_batched_magma bug，详见：
  // https://bitbucket.org/icl/magma/issues/13/getrf_batched-kernel-produces-nans-on
  std::tuple<magma_int_t, magma_int_t, magma_int_t> version;
  magma_version(&std::get<0>(version), &std::get<1>(version), &std::get<2>(version));
  const bool magma_batched_buggy = version < std::make_tuple<magma_int_t, magma_int_t, magma_int_t>(2, 5, 2);
  TORCH_CHECK(!magma_batched_buggy, "linalg.lu_factor has bugs on MAGMA < 2.5.2. Please update your MAGMA version to a newer one.");

  // 获取输入数据指针和 infos 数据指针
  auto input_data = input.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<magma_int_t>();
  // 计算输入矩阵的步幅
  auto input_matrix_stride = matrixStride(input);
  // 获取批次大小
  magma_int_t batch_size = magma_int_cast(batchCount(input), "batchCount");

  // 获取矩阵的行数和列数，并确保 leading_dimension 至少为 1
  magma_int_t m = magma_int_cast(input.size(-2), "m");
  magma_int_t n = magma_int_cast(input.size(-1), "n");
  auto leading_dimension = std::max<magma_int_t>(1, m);

  // 分配输入矩阵数组的空间
  scalar_t** input_array;
  ALLOCATE_ARRAY(input_array, scalar_t*, batch_size);

  // 设置指向矩阵的指针数组
  for (int64_t i = 0; i < batch_size; i++) {
    input_array[i] = &input_data[i * input_matrix_stride];
  }

  // 需要在并行运行 lu 测试时同步设备，参见 https://github.com/pytorch/pytorch/issues/82894 有关失败示例
  c10::cuda::device_synchronize();
  // 创建 MAGMA 队列
  MAGMAQueue magma_queue(input.get_device());

  if (compute_pivots) {
    // 如果需要计算 pivots
    auto pivots_data = pivots.data_ptr<magma_int_t>();
    auto pivots_stride = pivots.size(-1);
    // 将 pivots 填充为 1，以避免在 magma 内核中发生内存访问违规
    // magmaLuBatched 可能不会为其设置值，参见 https://github.com/pytorch/pytorch/pull/53064
    pivots.fill_(1);
    // 分配 pivots 数组的空间
    magma_int_t** pivots_array;
    ALLOCATE_ARRAY(pivots_array, magma_int_t*, batch_size);
    // 设置指向 pivots 的指针数组
    for (int64_t i = 0; i < batch_size; i++) {
      pivots_array[i] = &pivots_data[i * pivots_stride];
    }
    // 如果存在置换向量，则调用批量 LU 分解函数，带有置换
    magmaLuBatched<scalar_t>(m, n, input_array, leading_dimension, pivots_array, infos_data, batch_size, magma_queue);
  } else {
    // 如果不存在置换向量，则调用批量 LU 分解函数，不带置换
    magmaLuNoPivBatched<scalar_t>(m, n, input_array, leading_dimension, infos_data, batch_size, magma_queue);
  }

  // 阻塞 CPU 直到队列上的所有操作完成
  // 这个显式同步防止后续从不同队列调用的 magmaLuSolveBatched 函数产生垃圾结果
  magma_queue_sync(magma_queue.get_queue());
#endif
}

static void lu_factor_looped_magma(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  // 使用宏AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES，根据输入张量的数据类型进行派发
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "lu_factor_magma_looped", [&]{
    // 调用模板函数apply_lu_factor_looped_magma，对输入进行循环执行LU分解
    apply_lu_factor_looped_magma<scalar_t>(input, pivots, infos, compute_pivots);
  });
}

static void lu_factor_batched_magma(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  // 使用宏AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES，根据输入张量的数据类型进行派发
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "lu_factor_magma_batched", [&]{
    // 调用模板函数apply_lu_factor_batched_magma，对输入进行批处理LU分解
    apply_lu_factor_batched_magma<scalar_t>(input, pivots, infos, compute_pivots);
  });
}

static void lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  // 获取批次大小
  auto batch_size = batchCount(input);
  (void) batch_size; // 在某些构建中消除未使用的警告
  // 获取输入张量的维度大小
  auto m = input.size(-2);
  auto n = input.size(-1);

  // 定义lambda函数lu_factor_magma，根据批次大小选择调用不同的Magma库LU分解函数
  const auto lu_factor_magma = [batch_size](const Tensor& input, const Tensor& pivots, const Tensor& infos, const bool compute_pivots) {
    if (batch_size == 1) {
      lu_factor_looped_magma(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_magma(input, pivots, infos, compute_pivots);
    }
  };

  // 获取首选的线性代数后端
  const auto preferred_backend = at::globalContext().linalgPreferredBackend();
#ifdef USE_LINALG_SOLVER
  // 定义lambda函数lu_factor_cusolver，根据条件选择调用CUSOLVER库LU分解函数
  const auto lu_factor_cusolver = [batch_size, m, n](const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
    // 在 CUDA 10.2 中，当输入矩阵恰好是奇异矩阵时，lu_factor_looped_cusolver 无法完成计算
    // 返回的枢轴包含垃圾值。这会破坏 linalg.det
    // 现在，batched_cublas 不处理矩形矩阵，因此即使 m != n，仍然分派给 looped_cusolver。
#ifdef USE_ROCM
    constexpr bool looped_correct = true;
#else
    constexpr bool looped_correct = CUSOLVER_VERSION >= 11100;
#endif
    if (m != n || (looped_correct && (batch_size == 1 || m >= 512))) {
      lu_factor_looped_cusolver(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_cublas(input, pivots, infos, compute_pivots);
    }
  };

  // 如果首选的后端是 CUSOLVER，调用 lu_factor_cusolver 函数
  if (preferred_backend == at::LinalgBackend::Cusolver) {
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
  } else
#endif // ifdef USE_LINALG_SOLVER
  // 如果首选的后端是 Magma，调用 lu_factor_magma 函数
  if (preferred_backend == at::LinalgBackend::Magma) {
    lu_factor_magma(input, pivots, infos, compute_pivots);
  } else {  // 首选的后端是默认值
#ifdef USE_LINALG_SOLVER
#if AT_MAGMA_ENABLED()
    // 如果 magma batched 有 bug，则使用 cusolver
    // 否则，对于非方形矩阵，lu_factor 对其正常工作；对于方形矩阵，magma batched 是最快的
    // 否则（即对于方形矩阵），我们使用启发式方法在 cusolver 和 magma 之间选择
    // ROCm: 在 rocm 上，magma batched 有 bug。如果运行到这里，我们有 hipSOLVER，所以总是使用它，而不是 magma
#ifdef USE_ROCM
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
#else
    # 如果 m 等于 n 并且满足以下条件之一：
    # - batch_size 等于 1
    # - m 小于等于 16
    # - m 小于等于 128 并且 batch_size 小于等于 16
    if (m == n && (batch_size == 1 || m <= 16 || (m <= 128 && batch_size <= 16))) {
        # 调用 cusolver 库的 LU 分解函数，根据 compute_pivots 参数决定是否计算枢轴
        lu_factor_cusolver(input, pivots, infos, compute_pivots);
    } else {
        # 否则，调用 magma 库的批量 LU 分解函数，根据 compute_pivots 参数决定是否计算枢轴
        lu_factor_batched_magma(input, pivots, infos, compute_pivots);
    }
#else // USE_ROCM
#else // !AT_MAGMA_ENABLED
    // 如果未启用 MAGMA，则使用 cusolver 进行 LU 分解
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
#endif // AT_MAGMA_ENABLED
#else // !USE_LINALG_SOLVER
    // 如果未启用自定义线性代数求解器，则使用 MAGMA 进行 LU 分解
    lu_factor_magma(input, pivots, infos, compute_pivots);
#endif // USE_LINALG_SOLVER
  }

  // 返回简单的置换数组，从1开始（FORTRAN 索引）
  if (!compute_pivots) {
    // 计算最小的维度作为置换的上限（支持 FORTRAN 索引）
    auto k = std::min(input.size(-2), input.size(-1));
    // 创建从 1 到 k 的整数置换数组，使用输入张量的数据类型
    auto pivots_tmp = at::arange(1, k + 1, input.options().dtype(at::kInt));
    // 将生成的置换数组拷贝到输出的 pivots 张量中
    pivots.copy_(pivots_tmp);
  }
}

// 注册 CUDA 版本的 LU 分解函数
REGISTER_CUDA_DISPATCH(lu_factor_stub, &lu_factor);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
// 应用 MAGMA 批处理的三角求解函数
static void apply_triangular_solve_batched_magma(const Tensor& A, const Tensor& b, bool left, bool upper, TransposeType transpose, bool unitriangular) {
#if !AT_MAGMA_ENABLED()
// 如果未启用 MAGMA 库，则抛出错误
AT_ERROR("triangular_solve: MAGMA library not found in "
         "compilation. Please rebuild with MAGMA.");
#else
  // 设置 MAGMA 的上三角/下三角、转置、单位三角、左/右侧参数
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;
  magma_trans_t trans = to_magma(transpose);
  magma_diag_t diag = unitriangular ? MagmaUnit : MagmaNonUnit;
  magma_side_t side = left ? MagmaLeft : MagmaRight;

  // 获取 A 和 b 张量的数据指针
  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  // 当 left = True 时，允许传递矩形的 A 和 b 张量
  magma_int_t m = magma_int_cast(left ? A.size(-1) : b.size(-2), "m");
  magma_int_t n = magma_int_cast(b.size(-1), "n");
  // 如果维度为零，则设置为 1，以避免 MAGMA 调用错误
  magma_int_t lda = std::max<magma_int_t>(1, A.size(-2));
  magma_int_t ldb = std::max<magma_int_t>(1, b.size(-2));
  // 获取批次数并进行类型转换
  magma_int_t batch_size = magma_int_cast(batchCount(A), "batch_size");

  // 获取 A 和 b 张量的矩阵步长
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  // 分配 A 和 b 数据的指针数组内存
  scalar_t** A_array;
  scalar_t** b_array;

  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

  // 设置创建的指针数组
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
  }

  // 创建 MAGMA 队列对象
  MAGMAQueue magma_queue(b.get_device());

  // 定义最大的批次数限制
  constexpr int64_t batch_limit = 65535;
  // 计算尽可能多的 65535 的小批次数
  // "mini"-batches 的数量为 floor(batch_size / batch_limit)
  // 这些覆盖了 floor(batch_size / batch_limit) * batch_limit 的矩阵求解
  int64_t mini_batches = batch_size / batch_limit;
  int64_t mini_idx; // 在循环外声明，因为用于处理 batch_size % batch_limit != 0 的情况
  for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
    // 获取当前批次的 A 和 b 数据指针数组
    scalar_t** A_array_cur = &A_array[mini_idx];
    scalar_t** b_array_cur = &b_array[mini_idx];
  // 对批量的三角矩阵求解进行批处理，使用 MAGMA 库中的函数
  magmaTriangularSolveBatched<scalar_t>(
      side, uplo, trans, diag, m, n, A_array_cur,
      lda, b_array_cur, ldb, batch_limit, magma_queue);
}

// 计算剩余的部分，即 batch_size - floor(batch_size / batch_limit) * batch_limit
// 这等价于 batch_size % batch_limit
if (batch_size % batch_limit != 0) {
  // 对剩余部分的三角矩阵求解进行批处理，使用 MAGMA 库中的函数
  magmaTriangularSolveBatched<scalar_t>(
      side, uplo, trans, diag, m, n, &A_array[mini_idx],
      lda, &b_array[mini_idx], ldb, batch_size % batch_limit, magma_queue);
}
#endif
}

// 通过调用 apply_triangular_solve_batched_magma 实现对 CUDA Tensor A 和 B 进行批量三角解算
void triangular_solve_batched_magma(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cuda", [&]{
    apply_triangular_solve_batched_magma<scalar_t>(A, B, left, upper, transpose, unitriangular);
  });
}

// 实现对 CUDA Tensor A 和 B 进行三角解算的核心函数
void triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // 对于批次小于 8 且矩阵大小大于等于 64x64，使用 cuBLAS 的 for 循环比批处理版本更快
  if (batchCount(A) <= 8 && A.size(-1) >= 64) {
    triangular_solve_cublas(A, B, left, upper, transpose, unitriangular);
  } else {
#if !AT_MAGMA_ENABLED()
    // 如果未启用 MAGMA，使用 cuBLAS 进行批量三角解算
    triangular_solve_batched_cublas(A, B, left, upper, transpose, unitriangular);
#else
    // cuBLAS 批处理在小于等于 512x512 的情况下比 MAGMA 快
    if (A.size(-1) <= 512) {
      triangular_solve_batched_cublas(A, B, left, upper, transpose, unitriangular);
    } else {
      // 使用 MAGMA 进行批量三角解算
      triangular_solve_batched_magma(A, B, left, upper, transpose, unitriangular);
    }
#endif // AT_MAGMA_ENABLED()
  }
}

// 将 triangular_solve_kernel 函数注册为 CUDA 的分发函数
REGISTER_CUDA_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orgqr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 实现 orgqr 的核心函数
Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
  // TODO: 对于小于等于 32 的 tau 大小，可以使用 MAGMA 实现高效的批量 orgqr
  // 但在 Windows 上由于 MAGMA 内部的非法内存读取问题而失败
#ifdef USE_LINALG_SOLVER
  return orgqr_helper_cusolver(result, tau); // 使用 cusolver
#else
  // 若未编译使用 cuSOLVER 的 PyTorch，抛出错误信息
  TORCH_CHECK(false, "Calling torch.orgqr on a CUDA tensor requires compiling ",
    "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

// 将 orgqr_kernel_impl 函数注册为 CUDA 的分发函数
REGISTER_CUDA_DISPATCH(orgqr_stub, &orgqr_kernel_impl);

// 实现 ormqr 的核心函数
void ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
#ifdef USE_LINALG_SOLVER
  // 调用 cusolver 实现 ormqr
  ormqr_cusolver(input, tau, other, left, transpose);
#else
  // 若未编译使用 cuSOLVER 的 PyTorch，抛出错误信息
  TORCH_CHECK(false,
      "Calling torch.ormqr on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

// 将 ormqr_kernel 函数注册为 CUDA 的分发函数
REGISTER_CUDA_DISPATCH(ormqr_stub, &ormqr_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 为 scalar_t 定义 apply_geqrf 模板函数，用于执行 GEQRF 分解
template <typename scalar_t>
static void apply_geqrf(const Tensor& input, const Tensor& tau) {
#if !AT_MAGMA_ENABLED()
  // 若未启用 MAGMA，抛出错误信息
  TORCH_CHECK(
    false,
    "Calling torch.geqrf on a CUDA tensor requires compiling ",
    "PyTorch with MAGMA. Please use PyTorch built with MAGMA support.");
#else

  // 从输入张量中获取矩阵的行数 m 和列数 n
  magma_int_t m = magma_int_cast(input.size(-2), "m");
  magma_int_t n = magma_int_cast(input.size(-1), "n");

  // 获取输入张量的数据指针和相关的步长信息
  auto input_data = input.data_ptr<scalar_t>();
  auto input_matrix_stride = matrixStride(input);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(input);
  auto lda = std::max<int>(1, m);

  // magmaGeqrf 使用混合 CPU-GPU 算法计算初等反射器
  // geqrf2_gpu 驱动程序接受 CPU 上的张量作为初等反射器
  // 创建一个与 tau 大小相同的 CPU 张量，用于存储初等反射器
  Tensor tau_cpu = at::empty(tau.sizes(), tau.options().device(at::kCPU).pinned_memory(true));
  scalar_t* tau_data = tau_cpu.mutable_data_ptr<scalar_t>();
  scalar_t* work_data = nullptr; // geqrf2_gpu 不需要工作空间

  // 初始化 info 为 0，用于记录 MAGMA 函数的执行信息
  magma_int_t info = 0;
  for (int64_t i = 0; i < batch_size; i++) {
    // 获取当前批次的输入数据和 tau 数据的指针
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // 调用 MAGMA 的 geqrf2_gpu 函数计算 QR 分解和 tau
    // 此版本的 geqrf2_gpu 具有符合 LAPACK 的参数
    magmaGeqrf<scalar_t>(m, n, input_working_ptr, lda, tau_working_ptr, work_data, &info, /*is_v2=*/true);
    
    // 检查 MAGMA 函数是否执行出错
    checkMagmaInternalError(info, "geqrf");
  }
  // 将计算得到的 tau 数据拷贝回 GPU 张量 tau 中
  tau.copy_(tau_cpu, /*non_blocking=*/true);
#endif
}

// 'apply_geqrf' 的类型分发辅助函数
void geqrf_magma(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_magma", [&]{
    // 调用模板函数 apply_geqrf 处理具体类型的输入和 tau 张量
    apply_geqrf<scalar_t>(input, tau);
  });
}

// 执行 geqrf_kernel 的函数，根据宏 USE_LINALG_SOLVER 的定义不同选择使用不同的后端
void geqrf_kernel(const Tensor& input, const Tensor& tau) {
#ifdef USE_LINALG_SOLVER
  // 根据输入张量的大小和 batch 数量选择使用不同的后端
  auto geqrf_cusolver_backend = [](const Tensor& input, const Tensor& tau) {
      // 详细的性能评估见链接
      // 根据条件选择使用 geqrf_batched_cublas 或 geqrf_cusolver
      if (input.size(-2) <= 256 && batchCount(input) >= std::max<int64_t>(2, input.size(-2) / 16)) {
        return geqrf_batched_cublas(input, tau);
      } else {
        return geqrf_cusolver(input, tau);
      }
      return geqrf_batched_cublas(input, tau);
  };

  // 获取全局上下文中优先选择的线性代数后端
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
  // TODO 调查 MAGMA 后端是否仍然存在的 bug
  // 可能 geqrf 后跟 orgqr 对于 MAGMA 后端存在问题
  // geqrf_magma 当前使用 geqrf2_gpu
  //
  // 由于 MAGMA 的 bug，我们需要重新执行 ?geqrf_gpu:
  // - ?geqrf_gpu 可以通过 ?orgqr_gpu 快速计算 Q，但不能正确给出 R
  // - ?geqrf2_gpu 可以正确给出 R，但不能通过 ?orgqr_gpu 计算 Q
  // 更多细节请参考下面的链接：
  // http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=1015&p=2800&hilit=geqrf_gpu#p2800
    case at::LinalgBackend::Magma:
      return geqrf_magma(input, tau);
    case at::LinalgBackend::Cusolver:
    default:
      return geqrf_cusolver_backend(input, tau);
  }
#else
  // 默认情况下选择使用 MAGMA 后端执行 QR 分解
  return geqrf_magma(input, tau);
#endif
}
// 注册 CUDA 分发函数 geqrf_stub，并指定其对应的内核函数 geqrf_kernel
REGISTER_CUDA_DISPATCH(geqrf_stub, &geqrf_kernel);

// 定义模板函数 apply_magma_eigh，接受标量类型 scalar_t 的张量 values、vectors 和 infos 作为输入，
// 并根据参数 upper 和 compute_eigenvectors 执行 MAGMA 库中的特征值分解操作
template <typename scalar_t>
static void apply_magma_eigh(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
#if !AT_MAGMA_ENABLED()
  // 如果未启用 MAGMA 支持，则抛出错误信息，要求使用支持 MAGMA 的 PyTorch 版本
  TORCH_CHECK(
    false,
    "Calling torch.linalg.eigh/eigvalsh on a CUDA tensor requires compiling ",
    "PyTorch with MAGMA. Please use PyTorch built with MAGMA support.");
#else
  // 在调试模式下，确保 values 和 infos 张量在 CPU 上
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.device() == kCPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.device() == kCPU);

  // 使用 scalar_t 类型的实际数值类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  // 根据 upper 参数设定上三角或下三角的标志
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;
  // 根据 compute_eigenvectors 参数设定是否计算特征向量的标志
  magma_vec_t jobz = compute_eigenvectors ? MagmaVec : MagmaNoVec;

  // 获取向量张量 vectors 的最后一个维度大小作为矩阵的维度 n
  magma_int_t n = magma_int_cast(vectors.size(-1), "n");
  // 计算矩阵的领域 lda，并确保至少为 1
  auto lda = std::max<magma_int_t>(1, n);
  // 计算批次数目，即 vectors 的批次维度
  auto batch_size = batchCount(vectors);

  // 计算 vectors 和 values 张量在内存中的步长
  auto vectors_stride = matrixStride(vectors);
  auto values_stride = values.size(-1);

  // 获取 vectors、values 和 infos 张量的数据指针
  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<magma_int_t>();

  // 分配 lda*lda 大小的临时数组 wA，并用 scalar_t 类型的数据填充
  scalar_t* wA;
  ALLOCATE_ARRAY(wA, scalar_t, lda * lda);

  // 第一次调用，用于获取最佳工作空间大小
  magma_int_t lwork = -1;
  scalar_t wkopt;
  magma_int_t liwork = -1;
  magma_int_t iwkopt;
  magma_int_t lrwork = -1;
  value_t rwkopt;
  // 调用 MAGMA 库中的特征值分解函数 magmaSyevd，获取最佳工作空间大小
  magmaSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_data, lda, values_data,
    wA, lda, &wkopt, lwork, &rwkopt, lrwork, &iwkopt, liwork, infos_data);

  // 分配实际所需的工作空间大小
  scalar_t* work;
  magma_int_t* iwork;
  lwork = magma_int_cast(std::max<int64_t>(1, real_impl<scalar_t, value_t>(wkopt)), "work_size");
  liwork = magma_int_cast(std::max<int64_t>(1, iwkopt), "iwork_size");
  ALLOCATE_ARRAY(work, scalar_t, lwork);
  ALLOCATE_ARRAY(iwork, magma_int_t, liwork);

  // 如果 vectors 是复数类型，则需额外分配 rwork
  value_t* rwork = nullptr;
  c10::Storage storage_rwork;
  if (vectors.is_complex()) {
    lrwork = magma_int_cast(std::max<int64_t>(1, rwkopt), "rwork_size");
    // 分配并固定 rwork 存储
    storage_rwork = pin_memory<value_t>(lrwork);
    rwork = static_cast<value_t*>(storage_rwork.mutable_data());
  }

  // 循环处理每个批次中的矩阵
  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    // 获取当前批次的 vectors、values 和 infos 数据指针
    scalar_t* vectors_working_ptr = &vectors_data[i * vectors_stride];
    value_t* values_working_ptr = &values_data[i * values_stride];
    magma_int_t* info_working_ptr = &infos_data[i];
    // 再次调用 MAGMA 库中的特征值分解函数 magmaSyevd，执行实际的特征值分解
    magmaSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_working_ptr, lda, values_working_ptr,
      wA, lda, work, lwork, rwork, lrwork, iwork, liwork, info_working_ptr);
    // 线性代数函数当前的行为是，如果出现问题或输入不符合要求，会引发错误
    // 如果 info_working_ptr 指向的值不为零，则提前返回函数，因为后续的计算都将是无用的
    if (*info_working_ptr != 0) {
      return;
    }
#endif
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eigh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This is a type dispatch function for 'apply_magma_eigh'
// For small inputs result is computed on CPU
void linalg_eigh_magma(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  // MAGMA just calls LAPACK for eigenvectors.size(-1) <= 128
  // See https://bitbucket.org/icl/magma/src/e6fdca447bd402693e8b0b950a898b6879bbcc41/src/zheevd_gpu.cpp?at=master#lines-258
  // in addition lda is ignored breaking 0x0 inputs
  if (eigenvectors.size(-1) > 128) {
    // MAGMA requires eigenvalues and infos tensors to reside on CPU
    Tensor eigenvalues_cpu = eigenvalues.to(kCPU);
    Tensor infos_cpu = infos.to(kCPU);

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      eigenvectors.scalar_type(), "linalg_eigh_magma", [&] {
        // Invoke MAGMA routine for eigen decomposition
        apply_magma_eigh<scalar_t>(
            eigenvalues_cpu, eigenvectors, infos_cpu, upper, compute_eigenvectors);
      });

    // Transfer computed by MAGMA results from CPU to GPU
    eigenvalues.copy_(eigenvalues_cpu);
    infos.copy_(infos_cpu);
  } else { // eigenvectors.size(-1) <= 128
    // transfer to CPU, compute the result and copy back to GPU
    // this is faster than going through MAGMA that does the same
    Tensor eigenvalues_cpu = at::empty_like(eigenvalues, eigenvalues.options().device(kCPU));
    if (compute_eigenvectors) {
      // Compute eigenvalues and eigenvectors using LAPACK on CPU
      Tensor eigenvectors_cpu = at::empty_like(eigenvectors, eigenvectors.options().device(kCPU));
      at::linalg_eigh_out(eigenvalues_cpu, eigenvectors_cpu, eigenvectors.to(kCPU), upper ? "U" : "L");
      eigenvectors.copy_(eigenvectors_cpu);
    } else {
      // Compute only eigenvalues using LAPACK on CPU
      at::linalg_eigvalsh_out(eigenvalues_cpu, eigenvectors.to(kCPU), upper ? "U" : "L");
    }
    eigenvalues.copy_(eigenvalues_cpu);
  }
}

void linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
#if defined(USE_LINALG_SOLVER)
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Magma:
      // Call MAGMA-based eigen decomposition function
      linalg_eigh_magma(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
      break;
    case at::LinalgBackend::Cusolver:
    default:
      // Fall back to Cusolver-based eigen decomposition function
      linalg_eigh_cusolver(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  }
#else
  // Use MAGMA-based eigen decomposition function by default
  linalg_eigh_magma(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
#endif
}

// Register CUDA dispatch for linalg_eigh_stub to invoke linalg_eigh_kernel

REGISTER_CUDA_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
Computes the eigenvalues and eigenvectors of n-by-n matrix 'input'.
This is an in-place routine, content of 'input', 'values', 'vectors' is overwritten.
'infos' is an int Tensor containing error codes for each matrix in the batched input.
For more information see MAGMA's documentation for GEEV routine.
*/
template <typename scalar_t>
void apply_linalg_eig(Tensor& values, Tensor& vectors, Tensor& input, Tensor& infos, bool compute_eigenvectors) {
#if !AT_MAGMA_ENABLED()
// 如果未启用 MAGMA，抛出错误信息，要求重新编译 PyTorch 或将张量转移到 CPU 上再调用 torch.linalg.eig
TORCH_CHECK(false, "Calling torch.linalg.eig on a CUDA tensor requires compiling PyTorch with MAGMA. "
                   "Either transfer the tensor to the CPU before calling torch.linalg.eig or recompile with MAGMA.");
#else
  // 断言输入张量在 CPU 设备上
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == at::kCPU);
  // 断言特征值张量在 CPU 设备上
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.device() == at::kCPU);
  // 断言信息张量在 CPU 设备上
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.device() == at::kCPU);
  // 如果需要计算特征向量，断言特征向量张量在 CPU 设备上
  if (compute_eigenvectors) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.device() == at::kCPU);
  }

  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  // 根据是否需要计算特征向量设置 MAGMA 的计算选项
  magma_vec_t jobvr = compute_eigenvectors ? MagmaVec : MagmaNoVec;
  // 仅计算右特征向量
  magma_vec_t jobvl = MagmaNoVec;  // only right eigenvectors are computed
  // 获取输入张量的维度
  magma_int_t n = magma_int_cast(input.size(-1), "n");
  // 计算 lda 的值，确保不小于 1
  auto lda = std::max<magma_int_t>(1, n);
  // 计算批处理的数量
  auto batch_size = batchCount(input);
  // 获取输入张量的矩阵步长
  auto input_matrix_stride = matrixStride(input);
  // 获取特征值张量的步长
  auto values_stride = values.size(-1);
  // 获取输入张量数据指针
  auto input_data = input.data_ptr<scalar_t>();
  // 获取特征值张量数据指针
  auto values_data = values.data_ptr<scalar_t>();
  // 获取信息张量数据指针
  auto infos_data = infos.data_ptr<magma_int_t>();
  // 如果需要计算特征向量，获取右特征向量张量数据指针
  auto rvectors_data = compute_eigenvectors ? vectors.data_ptr<scalar_t>() : nullptr;
  // 仅计算右特征向量，左特征向量数据指针为空
  scalar_t* lvectors_data = nullptr;  // only right eigenvectors are computed
  // 计算右特征向量的 ldvr 值
  int64_t ldvr = compute_eigenvectors ? lda : 1;
  // 计算左特征向量的 ldvl 值
  int64_t ldvl = 1;

  // 声明用于复数类型的工作空间张量和数据指针
  Tensor rwork;
  value_t* rwork_data = nullptr;
  // 如果输入张量为复数类型，分配实数类型的工作空间并获取其数据指针
  if (input.is_complex()) {
    ScalarType real_dtype = toRealValueType(input.scalar_type());
    rwork = at::empty({lda * 2}, input.options().dtype(real_dtype));
    rwork_data = rwork.mutable_data_ptr<value_t>();
  }

  // 调用 magmaEig 函数以获取工作空间大小
  scalar_t work_query;
  magmaEig<scalar_t, value_t>(jobvl, jobvr, n, input_data, lda, values_data,
    lvectors_data, ldvl, rvectors_data, ldvr, &work_query, -1, rwork_data, &infos_data[0]);

  // 计算实际所需的工作空间大小
  magma_int_t lwork = std::max<magma_int_t>(1, static_cast<magma_int_t>(real_impl<scalar_t, value_t>(work_query)));
  // 分配工作空间张量
  Tensor work = at::empty({lwork}, input.dtype());
  // 获取工作空间数据指针
  auto work_data = work.mutable_data_ptr<scalar_t>();

  // 循环处理每个批次的输入张量
  for (auto i = decltype(batch_size){0}; i < batch_size; i++) {
    // 获取当前批次的输入张量指针
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    // 获取当前批次的特征值张量指针
    scalar_t* values_working_ptr = &values_data[i * values_stride];
    // 如果需要计算特征向量，获取当前批次的右特征向量张量指针
    scalar_t* rvectors_working_ptr = compute_eigenvectors ? &rvectors_data[i * input_matrix_stride] : nullptr;
    // 获取当前批次的信息张量指针
    int* info_working_ptr = &infos_data[i];
    // 调用 magmaEig 函数进行特征值分解计算
    magmaEig<scalar_t, value_t>(jobvl, jobvr, n, input_working_ptr, lda, values_working_ptr,
      lvectors_data, ldvl, rvectors_working_ptr, ldvr, work_data, lwork, rwork_data, info_working_ptr);
  }
#endif
}
// 定义函数 linalg_eig_kernel，用于执行非对称特征分解（eigendecomposition），在输入的张量上进行操作
void linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  // 此函数在批处理的列主内存格式中计算非对称特征分解
  // eigenvalues、eigenvectors 和 infos 的内容将被 apply_linalg_eig 覆盖

  // apply_linalg_eig 在原地修改提供的输入矩阵，因此我们需要一个副本
  // MAGMA 没有 GPU 接口用于特征分解，强制我们将 'input' 转移到 CPU
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_cuda());
  Tensor input_working_copy = at::empty(input.sizes(), input.options().device(kCPU));
  input_working_copy.transpose_(-2, -1);  // 将 input_working_copy 调整为 Fortran 连续的内存布局
  input_working_copy.copy_(input);

  // 在指定的数据类型上分发（dispatch）特征分解函数 apply_linalg_eig
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_eig_out_cuda", [&]{
    apply_linalg_eig<scalar_t>(eigenvalues, eigenvectors, input_working_copy, infos, compute_eigenvectors);
  });
}

// 注册 CUDA 分发（dispatch）函数 linalg_eig_kernel
REGISTER_CUDA_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 模板函数：apply_svd_magma
template<typename scalar_t>
static void apply_svd_magma(const Tensor& A,
                            const bool full_matrices,
                            const bool compute_uv,
                            const Tensor& U,
                            const Tensor& S,
                            const Tensor& Vh,
                            const Tensor& info) {
#if !AT_MAGMA_ENABLED()
// 如果未启用 MAGMA 库，抛出错误
AT_ERROR("linalg.svd: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  // 使用标量类型中的值类型
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  // 获取各张量的数据指针
  const auto A_data = A.data_ptr<scalar_t>();
  const auto U_data = compute_uv ? U.data_ptr<scalar_t>() : nullptr;
  const auto S_data = S.data_ptr<value_t>();
  const auto Vh_data = compute_uv ? Vh.data_ptr<scalar_t>() : nullptr;
  const auto info_data = info.data_ptr<magma_int_t>();
  // 获取矩阵的步长信息
  const auto A_stride = matrixStride(A);
  const auto U_stride = compute_uv ? matrixStride(U) : 0;
  const auto S_stride = S.size(-1);
  const auto Vh_stride = compute_uv ? matrixStride(Vh) : 0;
  const auto batchsize = batchCount(A);
  // 设置是否计算特征向量和 full_matrices 的标志
  const auto jobz = compute_uv ? (full_matrices ? MagmaAllVec : MagmaSomeVec) : MagmaNoVec;

  // 获取矩阵的维度信息
  const auto m = magma_int_cast(A.size(-2), "m");
  const auto n = magma_int_cast(A.size(-1), "n");
  const auto lda = magma_int_cast(A.strides().end()[-1], "lda");
  const auto ldu = compute_uv ? magma_int_cast(U.strides().end()[-1], "ldu") : magma_int_t{1};
  const auto ldvh = compute_uv ? magma_int_cast(Vh.strides().end()[-1], "ldvh") : magma_int_t{1};

  // 设置用于复数类型的实数工作数组
  c10::Storage storage_rwork;
  value_t* rwork = nullptr;
  if (A.is_complex()) {
    // 计算复数类型的实数工作数组的长度
    auto lrwork = computeLRWorkDim(compute_uv ? (full_matrices ? 'A' : 'S') : 'N', m, n);
    // 分配固定大小的 pin_memory 存储
    storage_rwork = pin_memory<value_t>(lrwork);
  rwork = static_cast<value_t*>(storage_rwork.mutable_data());

# 将可变数据存储 `storage_rwork` 转换为 `value_t` 类型的指针 `rwork`

magma_int_t* iwork;
ALLOCATE_ARRAY(iwork, magma_int_t, 8 * std::min(m, n));
# 分配大小为 `8 * std::min(m, n)` 的 `magma_int_t` 类型数组 `iwork`

// Query svd for the optimal lwork size
magma_int_t lwork = -1;
{
  scalar_t wkopt = 1; // 如果 MAGMA 未设置最佳工作空间的值，则将默认值设置为 1
  // 调用 magmaSvd 函数查询最优 lwork 大小
  magmaSvd<scalar_t, value_t>(jobz, m, n,
                              A_data, lda,
                              S_data,
                              compute_uv ? U_data : nullptr, ldu,
                              compute_uv ? Vh_data : nullptr, ldvh,
                              &wkopt, lwork, rwork, iwork, info_data);
  lwork = magma_int_cast(real_impl<scalar_t, value_t>(wkopt), "work_size");
}
scalar_t* work;
ALLOCATE_ARRAY(work, scalar_t, lwork);
# 分配大小为 `lwork` 的 `scalar_t` 类型数组 `work`

for (int64_t i = 0; i < batchsize; i++) {
  // 计算 S、U（可选）、Vh（可选）
  // 对每个批次中的 A_data 执行 SVD 分解，结果存储在 S_data、U_data 和 Vh_data 中
  magmaSvd<scalar_t, value_t>(jobz, m, n,
                              A_data + i * A_stride, lda,
                              S_data + i * S_stride,
                              compute_uv ? U_data + i * U_stride : nullptr, ldu,
                              compute_uv ? Vh_data + i * Vh_stride : nullptr, ldvh,
                              work, lwork, rwork, iwork,
                              info_data + i);
}
#endif
}

void svd_magma(const Tensor& A,
               const bool full_matrices,
               const bool compute_uv,
               const Tensor& U,
               const Tensor& S,
               const Tensor& Vh,
               const Tensor& info) {
  // A is on GPU and may not have the right strides.
  // We copy it into CPU with the correct strides and in pinned_memory as MAGMA moves things between CPU and GPU
  // 将张量 A 在 GPU 上，复制到 CPU 上，并且保证数据格式连续，使用固定内存存储，因为 MAGMA 会在 CPU 和 GPU 之间移动数据
  const auto A_ = A.mT()
                   .to(A.options()
                        .device(kCPU)
                        .memory_format(at::MemoryFormat::Contiguous)
                        .pinned_memory(true))
                   .mT();
  // U, S, Vh, info are the right size and strides, but are on GPU
  // We copy them into CPU in pinned_memory
  // U, S, Vh, info 张量尺寸和步长正确，但位于 GPU 上，我们将它们复制到 CPU 上的固定内存中
  const auto empty_like_cpu = [](const Tensor& t) {
    return at::empty_like(t, t.options().device(kCPU).pinned_memory(true));
  };
  auto U_ = compute_uv ? empty_like_cpu(U) : Tensor{};
  auto S_ = empty_like_cpu(S);
  auto Vh_ = compute_uv ? empty_like_cpu(Vh) : Tensor{};
  auto info_ = empty_like_cpu(info);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda", [&] {
    apply_svd_magma<scalar_t>(A_, full_matrices, compute_uv, U_, S_, Vh_, info_);
  });

  // Copy from CPU back to CUDA
  // We can do a non_blocking copy, as there is an unconditional check of the infos in
  // the calling function
  // 从 CPU 复制数据回到 CUDA
  // 可以进行非阻塞的复制，因为在调用函数中有对信息（infos）的无条件检查
  if (compute_uv) {
    U.copy_(U_, /*non_blocking*/true);
    Vh.copy_(Vh_, /*non_blocking*/true);
  }
  S.copy_(S_, /*non_blocking*/true);
  info.copy_(info, /*non_blocking*/true);
}

void svd_kernel(const Tensor& A,
                const bool full_matrices,
                const bool compute_uv,
                const std::optional<c10::string_view>& driver,
                const Tensor& U,
                const Tensor& S,
                const Tensor& Vh,
                const Tensor& info) {
#ifdef USE_LINALG_SOLVER
  // We always use cuSOLVER unless the user has specified they want to use MAGMA
  // 除非用户明确指定使用 MAGMA，否则我们总是使用 cuSOLVER
  bool use_magma = at::globalContext().linalgPreferredBackend() == at::LinalgBackend::Magma;
  if (use_magma) {
    svd_magma(A, full_matrices, compute_uv, U, S, Vh, info);
  } else {
    // svd_cusolver computes V rather than Vh, so we pass a view of Vh.mT
    // and then conjugate Vh in-place
    // svd_cusolver 计算 V 而不是 Vh，因此我们传递 Vh.mT 的视图，然后原地对 Vh 进行共轭
    svd_cusolver(A, full_matrices, compute_uv, driver, U, S, compute_uv ? Vh.mT() : Vh, info);
    if (compute_uv && Vh.is_complex()) {
      Vh._set_conj(!Vh.is_conj());
    }
  }
#else
  svd_magma(A, full_matrices, compute_uv, U, S, Vh, info);
#endif
}

REGISTER_CUDA_DISPATCH(svd_stub, &svd_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/*
  解决矩阵方程 A X = B
  X 和 B 是 n × nrhs 维度的矩阵，A 使用 LU 分解表示。
  这是一个原地操作的例程，B 的内容将被覆盖。
  这是一个“循环”变体，用于对批量输入调用单个输入的 MAGMA 函数。

  Args:
  * `LU` - [in] 矩阵 A 的 LU 分解（参见 at::linalg_lu_factor）
  * `pivots` - [in] 主元索引（参见 at::linalg_lu_factor）
  * `B` - [in] 右手边的矩阵 B
          [out] 解矩阵 X

  有关详细信息，请参阅 MAGMA 文档中的 magma_dgetrs_gpu。
*/
template <typename scalar_t>
static void apply_lu_solve_looped_magma(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "Calling linalg.lu_solve on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please rebuild with MAGMA.");
#else
  // 转换 transpose 到 MAGMA 格式
  auto trans = to_magma(transpose);

  // 获取 B 的数据指针
  auto b_data = B.data_ptr<scalar_t>();

  // 获取 LU 的数据指针
  auto lu_data = LU.data_ptr<scalar_t>();

  // MAGMA 要求 pivots 是一个 CPU 张量
  Tensor pivots_cpu = pivots.cpu();
  auto pivots_data = pivots_cpu.data_ptr<magma_int_t>();

  // 计算 B 和 LU 的步长
  auto b_stride = matrixStride(B);
  auto lu_stride = LU.dim() > 2 ? LU.stride(-3) : 0;

  // 计算 pivots 的步长和批次大小
  auto pivots_stride = pivots_cpu.dim() > 1 ? pivots_cpu.stride(-2) : 0;
  auto batch_size = batchCount(B);

  // 获取 n 和 nrhs 的值，并计算 leading_dimension
  magma_int_t n = magma_int_cast(LU.size(-2), "n");
  magma_int_t nrhs = magma_int_cast(B.size(-1), "nrhs");
  auto leading_dimension = std::max<magma_int_t>(1, n);

  // LU 和 pivots 张量可以广播到 B
  // 这里我们构造一个帮助索引张量来线性索引到 lu 和 pivots
  IntArrayRef lu_batch_shape(LU.sizes().data(), LU.dim() - 2);
  IntArrayRef b_batch_shape(B.sizes().data(), B.dim() - 2);
  BroadcastLinearIndices lu_index(
      batchCount(LU), lu_batch_shape, b_batch_shape);

  // 初始化 info 为 0
  int info = 0;

  // 遍历批次大小
  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    // 计算 lu_index 的值
    int64_t lu_index_i = lu_index(i);

    // 获取当前批次的 B 和 LU 数据指针
    scalar_t* b_working_ptr = &b_data[i * b_stride];
    scalar_t* lu_working_ptr = &lu_data[lu_index_i * lu_stride];

    // 获取当前批次的 pivots 数据指针
    int* pivots_working_ptr = &pivots_data[lu_index_i * pivots_stride];

    // 调用 MAGMA 提供的 LU 解算函数
    magmaLuSolve<scalar_t>(n, nrhs, lu_working_ptr, leading_dimension, pivots_working_ptr, b_working_ptr, leading_dimension, &info, trans);

    // 检查 MAGMA 函数调用是否成功
    // MAGMA 的 info 只在第 i 个参数错误时报告错误，所以我们不需要每次都检查它
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}
/*
  解决矩阵方程 A X = B
  这里 X 和 B 是 n × nrhs 的矩阵，A 使用 LU 分解表示。
  这是一个原地计算的例程，B 的内容将被覆盖。
  这是一个专门的批处理变体，预计在小输入时比“循环”版本更快。

  Args:
  * `LU` - [in] 矩阵 A 的 LU 分解 (参见 at::linalg_lu_factor)
  * `pivots` - [in] 主元索引 (参见 at::linalg_lu_factor)
  * `B` - [in] 右手边的矩阵 B
         [out] 解矩阵 X

  更多细节，请参阅 MAGMA 文档中的 magma_dgetrs_batched。
*/
template <typename scalar_t>
static void apply_lu_solve_batched_magma(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "Calling linalg.lu_solve on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please rebuild with MAGMA.");
#else
  TORCH_INTERNAL_ASSERT(batchCount(B) == batchCount(LU), "LU 和 B 的批次大小必须相同");
  TORCH_INTERNAL_ASSERT(batchCount(LU) == batchCount(pivots.unsqueeze(-1)), "LU 和 pivots 的批次大小必须相同");
  auto trans = to_magma(transpose);
  auto b_data = B.data_ptr<scalar_t>();
  auto lu_data = LU.data_ptr<scalar_t>();

  // 将 LU 和 B 的大小转换为 magma_int_t 类型
  magma_int_t n = magma_int_cast(LU.size(-2), "n");
  magma_int_t nrhs = magma_int_cast(B.size(-1), "nrhs");
  // 计算每行的主导维度，至少为1
  auto leading_dimension = std::max<magma_int_t>(1, n);

  auto pivots_data = pivots.data_ptr<magma_int_t>();

  // 获取 B, LU 和 pivots 的步幅
  auto b_stride = matrixStride(B);
  auto lu_stride = matrixStride(LU);
  auto pivots_stride = pivots.size(-1);
  // 将批次大小转换为 magma_int_t 类型
  magma_int_t batch_size = magma_int_cast(batchCount(B), "batchCount");

  // 分配内存给指针数组
  magma_int_t** pivots_array;
  scalar_t** lu_array;
  scalar_t** b_array;

  ALLOCATE_ARRAY(pivots_array, magma_int_t*, batch_size);
  ALLOCATE_ARRAY(lu_array, scalar_t*, batch_size);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

  // 填充指针数组
  for (int64_t i = 0; i < batch_size; i++) {
    pivots_array[i] = &pivots_data[i * pivots_stride];
    b_array[i] = &b_data[i * b_stride];
    lu_array[i] = &lu_data[i * lu_stride];
  }

  // 获取设备相关的 MAGMA 队列
  MAGMAQueue magma_queue(B.get_device());

  // 在批次为65535的情况下计算结果，这是 MAGMA 中允许的最大批次数
  constexpr int64_t batch_limit = 65535;

  for (int64_t mini_idx = 0; mini_idx < batch_size; mini_idx += batch_limit) {
    int64_t nbatches = std::min(batch_limit, batch_size - mini_idx);
    scalar_t** lu_array_cur = &lu_array[mini_idx];
    scalar_t** b_array_cur = &b_array[mini_idx];
    magma_int_t** pivots_array_cur = &pivots_array[mini_idx];

    int info;
    // 调用 MAGMA 中的批量 LU 解算例程
    magmaLuSolveBatched<scalar_t>(
        n, nrhs, lu_array_cur, leading_dimension,
        pivots_array_cur, b_array_cur, leading_dimension,
        info, nbatches, magma_queue, trans);

    // magmaLuSolveBatched 返回的 info 仅在参数错误时报告，因此不需要始终检查它
    // 使用 TORCH_INTERNAL_ASSERT_DEBUG_ONLY 宏来断言变量 info 的值必须为 0
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

static void lu_solve_batched_magma(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // 确保在使用 MAGMA 时，TransposeType 不是转置或共轭转置，因为存在BUG
  TORCH_INTERNAL_ASSERT(trans == TransposeType::NoTranspose);
  // 根据 LU 的数据类型调度到对应的函数模板 apply_lu_solve_batched_magma
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_batched_magma", [&]{
    apply_lu_solve_batched_magma<scalar_t>(LU, pivots, B, trans);
  });
}

static void lu_solve_looped_magma(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // 根据 LU 的数据类型调度到对应的函数模板 apply_lu_solve_looped_magma
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_looped_magma", [&]{
    apply_lu_solve_looped_magma<scalar_t>(LU, pivots, B, trans);
  });
}

c10::MaybeOwned<Tensor> maybe_expand_lu(const Tensor& B, const Tensor& LU) {
  // B 和 LU 具有相同的维数
  if (batchCount(B) != batchCount(LU)) {
    auto n = B.dim();
    auto expand_shape = DimVector(B.sizes().slice(0, n - 2));
    expand_shape.append({LU.size(-2), LU.size(-1)});
    // 扩展 LU 张量以匹配 B 张量的批次大小和最后两个维度的形状
    return c10::MaybeOwned<Tensor>::owned(
        cloneBatchedColumnMajor(LU.expand(expand_shape)));
  } else {
    // 直接使用 LU 张量，无需扩展
    return c10::MaybeOwned<Tensor>::borrowed(LU);
  }
}

c10::MaybeOwned<Tensor> maybe_expand_pivots(const Tensor& B, const Tensor& pivots) {
  // B 和 pivots 具有相同的维数
  if (batchCount(B) != batchCount(pivots.unsqueeze(-1))) {
    auto expand_shape = DimVector(B.sizes().slice(0, B.dim() - 2));
    expand_shape.push_back(pivots.size(-1));
    // 扩展 pivots 张量以匹配 B 张量的批次大小和最后一个维度的形状，并保证连续性
    return c10::MaybeOwned<Tensor>::owned(pivots.expand(expand_shape).contiguous());
  } else {
    // 直接使用 pivots 张量，无需扩展
    return c10::MaybeOwned<Tensor>::borrowed(pivots);
  }
}

static void lu_solve_kernel(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // Trivial case. Remove it once `torch.solve` is removed, as linalg.solve already shortcuts this case
  // 如果 B 张量中没有元素，则直接返回
  if (B.numel() == 0) {
    return;
  }

  auto b = batchCount(B);
  auto n = LU.size(-2);
  auto k = B.size(-1);
  // MAGMA 实现的 LU 解决方法不能处理最后一个维度大于 1024 的 b 张量
  // 参考：https://bitbucket.org/icl/magma/issues/19/dgesv_batched-dgetrs_batched-fails-for
  bool over_batched_magma_dim_limit = k > 1024;
  // 从测试中确定的启发式规则，参见：https://github.com/pytorch/pytorch/pull/72935

  // 计算 X = U^{-1}L^{-1}P^T B，通过三角形求解来减轻 MAGMA 中的BUG
  auto lu_solve_triangular = [n](const Tensor& LU, const Tensor& pivots, const Tensor& B, const TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    // LAPACK / cublas / 等返回一个奇怪的排列格式的排列
    // 在这里，我们将其转换为表示排列的向量，即一个（批次的）向量，使得 P(i) = j
    auto perm = at::arange(n, pivots_->options().dtype(kLong)).expand(pivots_->sizes()).contiguous();
    // 创建一个张量迭代器配置对象，用于配置张量操作的迭代器
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false) // 关闭内存重叠检查
      .check_all_same_dtype(false)  // 不检查所有张量的数据类型是否相同
      .resize_outputs(false)        // 禁止自动调整输出大小
      .declare_static_shape(pivots_->sizes(), /*squash_dim=*/pivots_->dim() - 1)  // 声明静态形状，压缩指定维度
      .add_output(perm)             // 将 perm 添加为输出张量
      .add_const_input(*pivots_)    // 将 pivots_ 添加为常量输入张量
      .build();                     // 构建迭代器配置对象

    // 使用指定迭代器配置调用 unpack_pivots_stub 函数
    unpack_pivots_stub(pivots_->device().type(), iter, n, n);

    // 如果不进行转置操作
    if (trans == TransposeType::NoTranspose) {
      // 获取逆排列 perm 的张量
      // 这是一种插入排序，相当于 perm = at::argsort(perm);
      // 但更可并行化且复杂度为 O(n)，利用了 perm 是一个排列的特性
      auto id_perm = at::arange(n, perm.options()).expand(perm.sizes());
      auto inv_perm = perm.scatter(-1, perm, id_perm);
      // 计算 B1 = P^T @ B （必须为 out-of-place，因为 B 是源和目标）
      auto B1 = B.scatter(-2, inv_perm.unsqueeze(-1).expand_as(B), B);
      // 解 L^{-1} @ B1，并将结果存储在 B 中
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, std::move(B1), /*upper=*/false, /*left=*/true, /*unitriangular=*/true);
      // 解 U^{-1} @ B，并将结果存储在 B 中
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, B, /*upper=*/true);
    } else {
      // 获取 LU 矩阵的共轭转置 LU_H
      auto LU_H = LU_->mH();
      // 解 U^{-H} @ B，并将结果存储在 B 中
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), LU_H, B, /*upper=*/false);
      // 解 L^{-H} @ B，并将结果存储在 B 中
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), LU_H, B, /*upper=*/true, /*left=*/true, /*unitriangular=*/true);
      // 计算 P @ B，并将结果存储在 B 中
      B.scatter_(-2, perm.unsqueeze(-1).expand_as(B), B.clone());
    }
  };
#ifdef USE_LINALG_SOLVER
  // 定义一个使用 CUBLAS 批量 LU 分解求解器的函数对象
  auto lu_solve_batched_cublas_fn = [](const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
    // 可能扩展 LU 分解和置换矩阵，使其与 B 的形状匹配
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    // 使用 CUBLAS 执行批量 LU 解算
    lu_solve_batched_cublas(*LU_, *pivots_, B, trans);
  };
#endif

// 定义一个使用 MAGMA 批量 LU 分解求解器的函数对象
auto lu_solve_batched_magma_fn = [](const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // 可能扩展 LU 分解和置换矩阵，使其与 B 的形状匹配
  auto LU_ = maybe_expand_lu(B, LU);
  auto pivots_ = maybe_expand_pivots(B, pivots);
  // 使用 MAGMA 执行批量 LU 解算
  lu_solve_batched_magma(*LU_, *pivots_, B, trans);
};

// 获取首选的线性代数后端
auto preferred_backend = at::globalContext().linalgPreferredBackend();
#ifdef USE_LINALG_SOLVER
// 如果首选后端是 Cusolver
if (preferred_backend == at::LinalgBackend::Cusolver) {
  // 根据特定条件选择使用循环调用 Cusolver 或者批量调用 CUBLAS 进行 LU 解算
  if (b <= 2 && n >= 64) {
    lu_solve_looped_cusolver(LU, pivots, B, trans);
  } else {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  }
  return;
} else
#endif // ifdef USE_LINALG_SOLVER
// 如果首选后端是 Magma
if (preferred_backend == at::LinalgBackend::Magma) {
  // 根据特定条件选择使用批量调用 MAGMA 或者循环调用 MAGMA 进行 LU 解算
  // 在某些情况下，批量调用 MAGMA 存在错误，因此使用循环调用 MAGMA
  if (!over_batched_magma_dim_limit && trans == TransposeType::NoTranspose) {
    lu_solve_batched_magma_fn(LU, pivots, B, trans);
  }
  else {
    lu_solve_looped_magma(LU, pivots, B, trans);
  }
  return;
}

// 启发式选择解算策略
// 根据不同的矩阵大小和条件选择最优的解算方法
#ifdef USE_LINALG_SOLVER
// 当 n == k 时，特殊情况处理 A^{-1}B，其中 B 是方阵
if (n == k) {
  // 如果 n 小于等于 16，则使用 CUBLAS 批量 LU 解算
  if (n <= 16) {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    return;
  }
  // 否则使用三角分解法解算
  lu_solve_triangular(LU, pivots, B, trans);
  return;
}

// 根据 n 的大小选择不同的解算策略
if (n <= 8) {
  // 对于较小的 n，根据条件选择使用 MAGMA 或者 CUBLAS 进行批量 LU 解算
  if (use_magma_ && !over_batched_magma_dim_limit && trans == TransposeType::NoTranspose && k >= 256) {
    lu_solve_batched_magma_fn(LU, pivots, B, trans);
  } else {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  }
} else if (n <= 64) {
  // 对于中等大小的 n，根据不同的条件选择使用 Cusolver 进行循环调用或者使用 CUBLAS 进行批量 LU 解算
  if (b <= 2 && (k <= 64 || trans != TransposeType::NoTranspose || n <= 32)) {
    lu_solve_looped_cusolver(LU, pivots, B, trans);
  } else if (k <= 8) {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  } else {
    lu_solve_triangular(LU, pivots, B, trans);
  }
} else if (n <= 128) {
  // 对于较大的 n，根据不同的条件选择使用 Cusolver 进行循环调用或者使用 CUBLAS 进行批量 LU 解算
  if (b <= 2 && k <= 2)  {
    lu_solve_looped_cusolver(LU, pivots, B, trans);
  } else if (k <= 2)  {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  } else {
    lu_solve_triangular(LU, pivots, B, trans);


注释：


    // 调用 lu_solve_triangular 函数，解决三角形方程组
    // 参数：
    // LU: 包含 LU 分解结果的矩阵
    // pivots: 包含 LU 分解中使用的枢轴信息
    // B: 需要求解的矩阵或向量
    // trans: 指定是否使用 LU 分解的转置
    lu_solve_triangular(LU, pivots, B, trans);


这段代码调用了一个名为 `lu_solve_triangular` 的函数，用于解决三角形方程组。注释提供了关于每个参数的说明，包括其作用和用途，以便理解函数调用的含义和用法。
} else { // n > 128
  lu_solve_triangular(LU, pivots, B, trans);
}
#else
  // No cublas or cusolver
  // lu_solve_triangular is almost always best
  lu_solve_triangular(LU, pivots, B, trans);
#endif // ifdef USE_LINALG_SOLVER
}



// 如果 n > 128，使用 LU 分解和回代求解线性方程组
} else { // n > 128
  lu_solve_triangular(LU, pivots, B, trans);
}
// 如果没有可用的 cublas 或 cusolver 库，通常最佳选择是使用 lu_solve_triangular 函数进行求解
#else
  // No cublas or cusolver
  // lu_solve_triangular is almost always best
  lu_solve_triangular(LU, pivots, B, trans);
#endif // ifdef USE_LINALG_SOLVER
}

// 注册 CUDA 分发函数 lu_solve_stub，并指向 lu_solve_kernel 函数
REGISTER_CUDA_DISPATCH(lu_solve_stub, &lu_solve_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lstsq ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 应用 GELS 算法解决线性最小二乘问题
template <typename scalar_t>
static void apply_gels(const Tensor& a, Tensor& b, Tensor& infos) {
#if !AT_MAGMA_ENABLED()
  // 如果没有编译 MAGMA 库，抛出错误提示重新编译使用 MAGMA 库
  TORCH_CHECK(false, "torch.linalg.lstsq: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto trans = MagmaNoTrans;
  auto m = magma_int_cast(a.size(-2), "m");
  auto n = magma_int_cast(a.size(-1), "n");

  // 检查是否为过定系统（即 m >= n）
  TORCH_CHECK(
    m >= n,
    "torch.linalg.lstsq: only overdetermined systems (input.size(-2) >= input.size(-1)) are allowed on CUDA");

  auto nrhs = magma_int_cast(b.size(-1), "nrhs");
  auto ldda = std::max<magma_int_t>(1, m);
  auto lddb = std::max<magma_int_t>(1, std::max(m, n));
  auto nb = magmaGeqrfOptimalBlocksize<scalar_t>(m, n);
  auto lwork = (m - n + nb) * (nrhs + nb) + nrhs * nb;
  // 分配工作空间
  Tensor hwork = at::empty({static_cast<int64_t>(lwork)}, a.scalar_type());
  auto* hwork_ptr = hwork.mutable_data_ptr<scalar_t>();

  // MAGMA 要求 infos 张量必须在 CPU 上
  infos = infos.to(at::kCPU);
  auto infos_data = infos.data_ptr<magma_int_t>();

  // 对输入的 batch 进行迭代处理
  batch_iterator_with_broadcasting<scalar_t>(a, b,
    [&](scalar_t* a_working_ptr, scalar_t* b_working_ptr,
      int64_t a_linear_batch_idx) {
      // 每个 batch 执行 GELS 算法
      magma_int_t* infos_working_ptr = &infos_data[a_linear_batch_idx];
      magmaGels<scalar_t>(trans, m, n, nrhs,
        a_working_ptr, ldda, b_working_ptr, lddb,
        hwork_ptr, lwork, infos_working_ptr);
    }
  );
#endif
}

// 使用 MAGMA 实现的 gels_magma 函数，调用 apply_gels 处理不同类型的输入
void gels_magma(const Tensor& a, Tensor& b, Tensor& infos) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "gels_magma", [&] {
    apply_gels<scalar_t>(a, b, infos);
  });
}

// 使用 QR 分解解决最小二乘问题的函数，参考 https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
void linalg_lstsq_gels(const Tensor& A, const Tensor& B, const Tensor& /*infos*/) {
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto mn = std::min(m, n);

  // 明确广播 A 的批处理维度
  IntArrayRef A_batch_sizes(A.sizes().data(), A.dim() - 2);
  IntArrayRef B_batch_sizes(B.sizes().data(), B.dim() - 2);
  std::vector<int64_t> expand_batch_portion = at::infer_size(A_batch_sizes, B_batch_sizes);

  auto tau_shape = A.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = mn;
  // 创建用于存储 tau 的张量
  Tensor tau = at::empty(tau_shape, A.options());

  if (m >= n) {
    // 步骤1：使用 geqrf 函数进行 QR 分解
    geqrf_kernel(A, tau);

    // 明确广播 A 的批处理维度
    // 将扩展部分的维度添加到 A_expand_batch 中，以便根据 geqrf 的结果对 A 进行扩展
    auto A_expand_batch = expand_batch_portion;
    A_expand_batch.insert(A_expand_batch.end(), {A.size(-2), A.size(-1)});
    // 使用 A_expand_batch 扩展 A 张量的维度
    Tensor A_expanded = A.expand({A_expand_batch});
    // 检查 A_expanded 是否是 Fortran 连续的张量
    bool is_fortran_contiguous = A_expanded.mT().is_contiguous();
    // 如果 A_expanded 是 Fortran 连续的，则直接使用它；否则克隆一个列主序批处理张量 A_broadcasted
    Tensor A_broadcasted = is_fortran_contiguous ? A_expanded : cloneBatchedColumnMajor(A_expanded);
    // 将扩展部分的维度添加到 tau_expand_batch 中
    auto tau_expand_batch = expand_batch_portion;
    tau_expand_batch.push_back(tau.size(-1));
    // 使用 tau_expand_batch 扩展 tau 张量的维度，并保证其是连续的
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 2: B <- Q^H B
    // 使用 ormqr_kernel 计算 Q 的共轭转置与 B 的乘积，此处 left=true, transpose=true
    ormqr_kernel(A_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/true);

    // Step 3: solve R X = B
    // 使用 triangular_solve_kernel 解线性方程组 R X = B，此处 left=true, upper=true, transpose=NoTranspose
    triangular_solve_kernel(
        A_broadcasted,
        B,
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/TransposeType::NoTranspose,
        /*unitriangular=*/false);
  } else { // underdetermined case
    // 对于欠定情况，使用 A 的共轭转置 Ah 进行处理
    Tensor Ah = cloneBatchedColumnMajor(A.mH());

    // Step 1: compute QR factorization of conjugate transpose of A using geqrf
    // 使用 geqrf_kernel 计算 A 共轭转置的 QR 分解
    geqrf_kernel(Ah, tau);

    // 显式广播 A 的批处理维度，避免对相同输入进行冗余计算
    auto A_expand_batch = expand_batch_portion;
    A_expand_batch.insert(A_expand_batch.end(), {Ah.size(-2), Ah.size(-1)});
    // 使用 A_expand_batch 扩展 Ah 张量的维度
    Tensor Ah_expanded = Ah.expand({A_expand_batch});
    // 检查 Ah_expanded 是否是 Fortran 连续的张量
    bool is_fortran_contiguous = Ah_expanded.mT().is_contiguous();
    // 如果 Ah_expanded 是 Fortran 连续的，则直接使用它；否则克隆一个列主序批处理张量 Ah_broadcasted
    Tensor Ah_broadcasted = is_fortran_contiguous ? Ah_expanded : cloneBatchedColumnMajor(Ah_expanded);

    // Step 2: R^H Z = B
    // 确定是否对 Ah_broadcasted 进行共轭转置，然后使用 triangular_solve_kernel 解线性方程组 R^H Z = B
    const auto trans = Ah_broadcasted.is_complex() ? TransposeType::ConjTranspose
                                                   : TransposeType::Transpose;
    triangular_solve_kernel(
        Ah_broadcasted,
        B,
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/trans,
        /*unitriangular=*/false);

    // 将 B 矩阵的第 m 行之后的部分置零，确保第 3 步的乘法运算正确进行
    B.narrow(-2, m, n - m).zero_();

    // 将扩展部分的维度添加到 tau_expand_batch 中
    auto tau_expand_batch = expand_batch_portion;
    tau_expand_batch.push_back(tau.size(-1));
    // 使用 tau_expand_batch 扩展 tau 张量的维度，并保证其是连续的
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 3: X <- Q Z
    // 使用 ormqr_kernel 计算 Q 与 Z 的乘积，此处 left=true, transpose=false
    ormqr_kernel(Ah_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/false);
  }
}

// 定义函数 gels_looped，用于处理线性方程组求解，支持多种求解后端
void gels_looped(const Tensor& a, Tensor& b, Tensor& infos) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  // 获取全局上下文中的线性代数求解器首选后端
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  // 根据首选后端选择相应的求解函数
  switch (preferred_backend) {
    case at::LinalgBackend::Magma:
      return gels_magma(a, b, infos); // 使用 MAGMA 求解
    case at::LinalgBackend::Cusolver:
    default:
      // linalg_lstsq_gels 是一个通用函数，基于 geqrf_stub、ormqr_stub 和 triangular_solve_stub 实现
      // 在 CUDA 输入时，通过 cuSOLVER 进行具体实现
      return linalg_lstsq_gels(a, b, infos); // 使用 cuSOLVER 求解
  }
#else
  return gels_magma(a, b, infos); // 默认使用 MAGMA 求解
#endif
}

// 定义函数 lstsq_kernel，实现最小二乘解法
void lstsq_kernel(const Tensor& a, Tensor& b, Tensor& /*rank*/, Tensor& /*singular_values*/, Tensor& infos, double /*rcond*/, std::string /*driver_name*/)  {
  auto m = a.size(-2); // 获取输入张量 a 的倒数第二维大小
  auto n = a.size(-1); // 获取输入张量 a 的最后一维大小

  // 处理欠定情况 (m < n)
  // MAGMA 或 cuBLAS 不支持此情况
  if (m < n) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
    linalg_lstsq_gels(a, b, infos); // 使用 linalg_lstsq_gels 求解
#else
    // 在 CUDA 平台上，只允许处理超定系统 (input.size(-2) >= input.size(-1))
    TORCH_CHECK(
        false,
        "torch.linalg.lstsq: only overdetermined systems (input.size(-2) >= input.size(-1)) are allowed on CUDA. ",
        "Please rebuild with cuSOLVER.");
#endif
  } else { // 处理超定情况 (m >= n)
#if !AT_ROCM_ENABLED()
    // 在 CUDA 平台上，根据性能需求选择批处理或循环处理
    if (m <= 256 && batchCount(b) >= std::max<int64_t>(2, m / 16)) {
      gels_batched_cublas(a, b, infos); // 使用 gels_batched_cublas 进行批处理求解
    } else {
      gels_looped(a, b, infos); // 使用 gels_looped 进行循环求解
    }
#else
    // 在 ROCm 平台上，只能使用 MAGMA 进行求解
    // 如果 MAGMA 不可用，将抛出错误
    gels_magma(a, b, infos); // 使用 MAGMA 求解
#endif // !AT_ROCM_ENABLED()
  }
}

// 注册 CUDA 分发，将 lstsq_kernel 函数与 lstsq_stub 关联
REGISTER_CUDA_DISPATCH(lstsq_stub, &lstsq_kernel);

// 如果构建时开启了延迟 CUDA 线性代数库编译，进行相关初始化
#if defined(BUILD_LAZY_CUDA_LINALG)
struct DispatchInitializer {
  DispatchInitializer() {
    cuda::detail::LinalgDispatch disp{_cholesky_solve_helper_cuda};
    cuda::detail::registerLinalgDispatch(disp);
  };
} initializer;

}  // namespace lazy_linalg
#endif
}  // namespace at::native

#undef ALLOCATE_ARRAY
```