# `.\pytorch\aten\src\ATen\native\cuda\linalg\CUDASolver.cpp`

```
#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cuda/linalg/CUDASolver.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Export.h>

#if defined(CUDART_VERSION) || defined(USE_ROCM)

// 声明命名空间，用于包含 CUDA 特定的线性代数求解函数
namespace at::cuda::solver {

// 模板特化：双精度浮点数矩阵的 LU 分解
template <>
void getrf<double>(
    cusolverDnHandle_t handle, int m, int n, double* dA, int ldda, int* ipiv, int* info) {
  // 计算所需的工作空间大小
  int lwork;
  TORCH_CUSOLVER_CHECK(
      cusolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  // 获取 CUDA 缓存分配器的引用
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配所需大小的内存空间，并将其包装为指针
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);
  // 执行双精度 LU 分解
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrf(
      handle, m, n, dA, ldda, static_cast<double*>(dataPtr.get()), ipiv, info));
}

// 模板特化：单精度浮点数矩阵的 LU 分解
template <>
void getrf<float>(
    cusolverDnHandle_t handle, int m, int n, float* dA, int ldda, int* ipiv, int* info) {
  // 计算所需的工作空间大小
  int lwork;
  TORCH_CUSOLVER_CHECK(
      cusolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  // 获取 CUDA 缓存分配器的引用
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配所需大小的内存空间，并将其包装为指针
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);
  // 执行单精度 LU 分解
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrf(
      handle, m, n, dA, ldda, static_cast<float*>(dataPtr.get()), ipiv, info));
}

// 模板特化：双精度复数矩阵的 LU 分解
template <>
void getrf<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int m,
    int n,
    c10::complex<double>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  // 计算所需的工作空间大小
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(
      handle, m, n, reinterpret_cast<cuDoubleComplex*>(dA), ldda, &lwork));
  // 获取 CUDA 缓存分配器的引用
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配所需大小的内存空间，并将其包装为指针
  auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex) * lwork);
  // 执行双精度复数 LU 分解
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrf(
      handle,
      m,
      n,
      reinterpret_cast<cuDoubleComplex*>(dA),
      ldda,
      static_cast<cuDoubleComplex*>(dataPtr.get()),
      ipiv,
      info));
}

// 模板特化：单精度复数矩阵的 LU 分解
template <>
void getrf<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int m,
    int n,
    c10::complex<float>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  // 计算所需的工作空间大小
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(
      handle, m, n, reinterpret_cast<cuComplex*>(dA), ldda, &lwork));
  // 获取 CUDA 缓存分配器的引用
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配所需大小的内存空间，并将其包装为指针
  auto dataPtr = allocator.allocate(sizeof(cuComplex) * lwork);
  // 执行单精度复数 LU 分解
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrf(
      handle,
      m,
      n,
      reinterpret_cast<cuComplex*>(dA),
      ldda,
      static_cast<cuComplex*>(dataPtr.get()),
      ipiv,
      info));
}

// 模板特化：双精度浮点数矩阵的 LU 分解的解算
template <>
void getrs<double>(
    cusolverDnHandle_t handle, int n, int nrhs, double* dA, int lda, int* ipiv, double* ret, int ldb, int* info, cublasOperation_t trans) {
  // 执行双精度浮点数 LU 分解的解算
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrs(
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

// 模板特化：单精度浮点数矩阵的 LU 分解的解算
template <>
void getrs<float>(
    cusolverDnHandle_t handle, int n, int nrhs, float* dA, int lda, int* ipiv, float* ret, int ldb, int* info, cublasOperation_t trans) {
  // 执行单精度浮点数 LU 分解的解算
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrs(
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

// 结束命名空间声明
} // namespace at::cuda::solver

#endif  // defined(CUDART_VERSION) || defined(USE_ROCM)


注释完成，详细解释了每个模板函数的作用、内存分配和具体调用的 CUDA 求解函数。
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));



    调用某个函数或方法，传入多个参数：handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info
    这些参数的具体含义和作用根据上下文和函数定义来理解
    该行代码可能用于执行某种数值计算、线性代数操作或者其他需要大量参数的操作
    这些参数可能代表矩阵、向量、选项等
    ```
template <>
void getrs<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int n,
    int nrhs,
    c10::complex<double>* dA,
    int lda,
    int* ipiv,
    c10::complex<double>* ret,
    int ldb,
    int* info,
    cublasOperation_t trans) {
  // 使用cusolverDnZgetrs函数求解复数双精度矩阵方程
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrs(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<cuDoubleComplex*>(dA),  // 转换dA为cuDoubleComplex指针
      lda,
      ipiv,
      reinterpret_cast<cuDoubleComplex*>(ret),  // 转换ret为cuDoubleComplex指针
      ldb,
      info));
}

template <>
void getrs<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int n,
    int nrhs,
    c10::complex<float>* dA,
    int lda,
    int* ipiv,
    c10::complex<float>* ret,
    int ldb,
    int* info,
    cublasOperation_t trans) {
  // 使用cusolverDnCgetrs函数求解复数单精度矩阵方程
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrs(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<cuComplex*>(dA),  // 转换dA为cuComplex指针
      lda,
      ipiv,
      reinterpret_cast<cuComplex*>(ret),  // 转换ret为cuComplex指针
      ldb,
      info));
}

template <>
void sytrf_bufferSize<double>(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(double)) {
  // 计算双精度实数对称矩阵LU分解所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork));
}

template <>
void sytrf_bufferSize<float>(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(float)) {
  // 计算单精度实数对称矩阵LU分解所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork));
}

template <>
void sytrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_SYTRF_BUFFER_ARGTYPES(c10::complex<double>)) {
  // 计算双精度复数对称矩阵LU分解所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnZsytrf_bufferSize(
      handle, n, reinterpret_cast<cuDoubleComplex*>(A), lda, lwork));
}

template <>
void sytrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_SYTRF_BUFFER_ARGTYPES(c10::complex<float>)) {
  // 计算单精度复数对称矩阵LU分解所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnCsytrf_bufferSize(
      handle, n, reinterpret_cast<cuComplex*>(A), lda, lwork));
}

template <>
void sytrf<double>(CUDASOLVER_SYTRF_ARGTYPES(double)) {
  // 对双精度实数对称矩阵进行LU分解
  TORCH_CUSOLVER_CHECK(
      cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
}

template <>
void sytrf<float>(CUDASOLVER_SYTRF_ARGTYPES(float)) {
  // 对单精度实数对称矩阵进行LU分解
  TORCH_CUSOLVER_CHECK(
      cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
}

template <>
void sytrf<c10::complex<double>>(
    CUDASOLVER_SYTRF_ARGTYPES(c10::complex<double>)) {
  // 对双精度复数对称矩阵进行LU分解
  TORCH_CUSOLVER_CHECK(cusolverDnZsytrf(
      handle,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),  // 转换A为cuDoubleComplex指针
      lda,
      ipiv,
      reinterpret_cast<cuDoubleComplex*>(work),  // 转换work为cuDoubleComplex指针
      lwork,
      devInfo));
}

template <>
void sytrf<c10::complex<float>>(
    CUDASOLVER_SYTRF_ARGTYPES(c10::complex<float>)) {
  // 对单精度复数对称矩阵进行LU分解
  TORCH_CUSOLVER_CHECK(cusolverDnCsytrf(
      handle,
      uplo,
      n,
      reinterpret_cast<cuComplex*>(A),  // 转换A为cuComplex指针
      lda,
      ipiv,
      reinterpret_cast<cuComplex*>(work),  // 转换work为cuComplex指针
      lwork,
      devInfo));
}

template<>
void gesvd_buffersize<float>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  // 计算单精度通用矩阵奇异值分解所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle, m, n, lwork));
}
// 模板特化，计算双精度实数类型的奇异值分解缓冲区大小
template<>
void gesvd_buffersize<double>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  // 调用 cuSOLVER 库函数以获取双精度实数类型的奇异值分解所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(handle, m, n, lwork));
}

// 模板特化，计算单精度复数类型的奇异值分解缓冲区大小
template<>
void gesvd_buffersize<c10::complex<float>>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  // 调用 cuSOLVER 库函数以获取单精度复数类型的奇异值分解所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvd_bufferSize(handle, m, n, lwork));
}

// 模板特化，计算双精度复数类型的奇异值分解缓冲区大小
template<>
void gesvd_buffersize<c10::complex<double>>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  // 调用 cuSOLVER 库函数以获取双精度复数类型的奇异值分解所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(handle, m, n, lwork));
}


// 模板特化，执行单精度实数类型的奇异值分解
template<>
void gesvd<float>(CUDASOLVER_GESVD_ARGTYPES(float, float)) {
  // 调用 cuSOLVER 库函数执行单精度实数类型的奇异值分解
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvd(
      handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info));
}

// 模板特化，执行双精度实数类型的奇异值分解
template<>
void gesvd<double>(CUDASOLVER_GESVD_ARGTYPES(double, double)) {
  // 调用 cuSOLVER 库函数执行双精度实数类型的奇异值分解
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvd(
      handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info));
}


// 模板特化，执行单精度复数类型的奇异值分解
template<>
void gesvd<c10::complex<float>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<float>, float)) {
  // 调用 cuSOLVER 库函数执行单精度复数类型的奇异值分解
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvd(
      handle, jobu, jobvt, m, n,
      reinterpret_cast<cuComplex*>(A),
      lda, S,
      reinterpret_cast<cuComplex*>(U),
      ldu,
      reinterpret_cast<cuComplex*>(VT),
      ldvt,
      reinterpret_cast<cuComplex*>(work),
      lwork, rwork, info
  ));
}

// 模板特化，执行双精度复数类型的奇异值分解
template<>
void gesvd<c10::complex<double>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<double>, double)) {
  // 调用 cuSOLVER 库函数执行双精度复数类型的奇异值分解
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvd(
      handle, jobu, jobvt, m, n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda, S,
      reinterpret_cast<cuDoubleComplex*>(U),
      ldu,
      reinterpret_cast<cuDoubleComplex*>(VT),
      ldvt,
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork, rwork, info
  ));
}


// 模板特化，计算单精度实数类型的奇异值分解缓冲区大小
template<>
void gesvdj_buffersize<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float *A, int lda, float *S,
    float *U, int ldu, float *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  // 调用 cuSOLVER 库函数以获取单精度实数类型的奇异值分解缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params));
}

// 模板特化，计算双精度实数类型的奇异值分解缓冲区大小
template<>
void gesvdj_buffersize<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double *A, int lda, double *S,
    double *U, int ldu, double *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  // 调用 cuSOLVER 库函数以获取双精度实数类型的奇异值分解缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params));
}

// 模板特化，计算单精度复数类型的奇异值分解缓冲区大小
template<>
void gesvdj_buffersize<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<float> *A, int lda, float *S,
    c10::complex<float> *U, int ldu, c10::complex<float> *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  // 调用 cuSOLVER 库函数以获取单精度复数类型的奇异值分解缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda,
    S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv, lwork, params));
}
// 定义模板特化函数，用于求解单个 float 类型的 SVD 问题
template<>
void gesvdj<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float *V, int ldv, float* work, int lwork, int *info, gesvdjInfo_t params
) {
  // 调用 CUDA cuSOLVER 库的单精度 SVD 求解函数 cusolverDnSgesvdj，返回结果通过参数传出
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params));
}

// 定义模板特化函数，用于求解单个 double 类型的 SVD 问题
template<>
void gesvdj<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double *V, int ldv, double* work, int lwork, int *info, gesvdjInfo_t params
) {
  // 调用 CUDA cuSOLVER 库的双精度 SVD 求解函数 cusolverDnDgesvdj，返回结果通过参数传出
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params));
}

// 定义模板特化函数，用于求解单个复数 float 类型的 SVD 问题
template<>
void gesvdj<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float> *V, int ldv, c10::complex<float>* work, int lwork, int *info, gesvdjInfo_t params
) {
  // 调用 CUDA cuSOLVER 库的单精度复数 SVD 求解函数 cusolverDnCgesvdj，返回结果通过参数传出
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv,
    reinterpret_cast<cuComplex*>(work),
    lwork, info, params));
}

// 定义模板特化函数，用于求解单个复数 double 类型的 SVD 问题
template<>
void gesvdj<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double> *V, int ldv, c10::complex<double>* work, int lwork, int *info, gesvdjInfo_t params
) {
  // 调用 CUDA cuSOLVER 库的双精度复数 SVD 求解函数 cusolverDnZgesvdj，返回结果通过参数传出
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv,
    reinterpret_cast<cuDoubleComplex*>(work),
    lwork, info, params));
}
// 模板特化：计算单个批次的 SVD（奇异值分解），处理单精度浮点数
template<>
void gesvdjBatched<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float* V, int ldv, int* info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  // 获取执行 SVD 所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

  // 获取 CUDA 内存分配器
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配缓冲区所需的内存
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);

  // 执行单精度 SVD 批处理
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdjBatched(
    handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<float*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

// 模板特化：计算单个批次的 SVD，处理双精度浮点数
template<>
void gesvdjBatched<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double* V, int ldv, int* info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  // 获取执行 SVD 所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

  // 获取 CUDA 内存分配器
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配缓冲区所需的内存
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);

  // 执行双精度 SVD 批处理
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdjBatched(
    handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<double*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

// 模板特化：计算单个批次的 SVD，处理单精度复数
template<>
void gesvdjBatched<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float>* V, int ldv, int* info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  // 获取执行 SVD 所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdjBatched_bufferSize(
    handle, jobz, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv, &lwork, params, batchSize));

  // 获取 CUDA 内存分配器
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配缓冲区所需的内存
  auto dataPtr = allocator.allocate(sizeof(cuComplex)*lwork);

  // 执行单精度复数 SVD 批处理
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdjBatched(
    handle, jobz, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv,
    static_cast<cuComplex*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

// 模板特化：计算单个批次的 SVD，处理双精度复数
template<>
void gesvdjBatched<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double>* V, int ldv, int* info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  // 获取执行 SVD 所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdjBatched_bufferSize(
    handle, jobz, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv, &lwork, params, batchSize));

  // 获取 CUDA 内存分配器
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配缓冲区所需的内存
  auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex)*lwork);

  // 执行双精度复数 SVD 批处理
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdjBatched(
    handle, jobz, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv,
    static_cast<cuDoubleComplex*>(dataPtr.get()),
    lwork, info, params, batchSize));
}
    ldu,
    # 参数 ldu：描述第一个输入矩阵 V 的列数
    reinterpret_cast<cuDoubleComplex*>(V),
    # 将 V 的指针重新解释为 cuDoubleComplex 类型的指针，用于指向数据
    ldv,
    # 参数 ldv：描述第二个输入矩阵的列数
    static_cast<cuDoubleComplex*>(dataPtr.get()),
    # 将 dataPtr 的智能指针转换为 cuDoubleComplex 类型的指针，指向批处理数据的内存块
    lwork, 
    # 参数 lwork：描述所需的工作区大小
    info, 
    # 用于输出 LAPACK 函数的状态信息
    params, 
    # 附加参数对象，用于传递额外的控制参数
    batchSize));
    # 执行 LAPACK 函数，处理大小为 batchSize 的批量数据
// ROCM does not implement gesdva yet
#ifdef CUDART_VERSION
// 模板特化：计算单精度浮点数版本的 gesvdaStridedBatched_buffersize 函数
template<>
void gesvdaStridedBatched_buffersize<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, float *A, int lda, long long int strideA,
    float *S, long long int strideS, float *U, int ldu, long long int strideU, float *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  // 调用 cuSOLVER 库中的 cusolverDnSgesvdaStridedBatched_bufferSize 函数获取缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, lwork, batchSize
  ));
}

// 模板特化：计算双精度浮点数版本的 gesvdaStridedBatched_buffersize 函数
template<>
void gesvdaStridedBatched_buffersize<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, double *A, int lda, long long int strideA,
    double *S, long long int strideS, double *U, int ldu, long long int strideU, double *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  // 调用 cuSOLVER 库中的 cusolverDnDgesvdaStridedBatched_bufferSize 函数获取缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, lwork, batchSize
  ));
}

// 模板特化：计算单精度复数版本的 gesvdaStridedBatched_buffersize 函数
template<>
void gesvdaStridedBatched_buffersize<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, c10::complex<float> *A, int lda, long long int strideA,
    float *S, long long int strideS, c10::complex<float> *U, int ldu, long long int strideU,
    c10::complex<float> *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  // 调用 cuSOLVER 库中的 cusolverDnCgesvdaStridedBatched_bufferSize 函数获取缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n,
    reinterpret_cast<cuComplex*>(A), // 将 c10::complex<float>* 转换为 cuComplex* 类型
    lda, strideA, S, strideS,
    reinterpret_cast<cuComplex*>(U), // 将 c10::complex<float>* 转换为 cuComplex* 类型
    ldu, strideU,
    reinterpret_cast<cuComplex*>(V), // 将 c10::complex<float>* 转换为 cuComplex* 类型
    ldv, strideV, lwork, batchSize
  ));
}

// 模板特化：计算双精度复数版本的 gesvdaStridedBatched_buffersize 函数
template<>
void gesvdaStridedBatched_buffersize<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, c10::complex<double> *A, int lda, long long int strideA,
    double *S, long long int strideS, c10::complex<double> *U, int ldu, long long int strideU,
    c10::complex<double> *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  // 调用 cuSOLVER 库中的 cusolverDnZgesvdaStridedBatched_bufferSize 函数获取缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n,
    reinterpret_cast<cuDoubleComplex*>(A), // 将 c10::complex<double>* 转换为 cuDoubleComplex* 类型
    lda, strideA, S, strideS,
    reinterpret_cast<cuDoubleComplex*>(U), // 将 c10::complex<double>* 转换为 cuDoubleComplex* 类型
    ldu, strideU,
    reinterpret_cast<cuDoubleComplex*>(V), // 将 c10::complex<double>* 转换为 cuDoubleComplex* 类型
    ldv, strideV, lwork, batchSize
  ));
}

// 模板特化：计算单精度浮点数版本的 gesvdaStridedBatched 函数
template<>
void gesvdaStridedBatched<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, float *A, int lda, long long int strideA,
    float *S, long long int strideS, float *U, int ldu, long long int strideU, float *V, int ldv, long long int strideV,
    float *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  // 调用 cuSOLVER 库中的 cusolverDnSgesvdaStridedBatched 函数执行 SVD 分解
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize
  ));
}
#endif
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize
));

# 调用一个函数或方法，传递了一系列参数，这些参数依次为：
# handle: 一个句柄或者对象，用于处理某个资源或操作
# jobz: 控制作业的标志或选项
# rank: 表示某个属性或矩阵的秩
# m: 一个整数参数，可能表示行数或者某个维度的大小
# n: 另一个整数参数，可能表示列数或者另一维度的大小
# A: 一个数组或矩阵对象
# lda: A 矩阵的行跨度（leading dimension）
# strideA: A 矩阵数据的步幅
# S: 另一个数组或矩阵对象
# strideS: S 矩阵数据的步幅
# U: 第三个数组或矩阵对象
# ldu: U 矩阵的行跨度（leading dimension）
# strideU: U 矩阵数据的步幅
# V: 第四个数组或矩阵对象
# ldv: V 矩阵的行跨度（leading dimension）
# strideV: V 矩阵数据的步幅
# work: 用于工作空间的数组或缓冲区
# lwork: 工作空间数组的长度或大小
# info: 用于返回状态或信息的整数变量或指针
# h_R_nrmF: 可能是某种规范化因子的变量或参数
# batchSize: 另一个整数参数，可能表示批处理大小或数量
template<>
void potrf<c10::complex<double>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, c10::complex<double>* A, int lda, c10::complex<double>* work, int lwork, int* info
) {
  // 调用 cuSOLVER 中的 cuSolverDnZpotrf 函数进行复数双精度 Cholesky 分解
  TORCH_CUSOLVER_CHECK(cusolverDnZpotrf(
    handle,
    uplo,
    n,
    reinterpret_cast<cuDoubleComplex*>(A),  // 将 A 转换为 cuDoubleComplex* 类型
    lda,
    reinterpret_cast<cuDoubleComplex*>(work),  // 将 work 转换为 cuDoubleComplex* 类型
    lwork,
    info));  // 存储执行状态信息的指针
}
    // 将指针 work 重新解释为 cuComplex 类型的指针，并作为参数传递给函数
    reinterpret_cast<cuComplex*>(work),
    // 参数 lwork 传递给函数，表示工作区的大小或其他需要的长度信息
    lwork,
    // 参数 info 传递给函数，用于返回操作执行的信息或状态码
    info));
template <>
void geqrf_bufferSize<c10::complex<float>>(
  cusolverDnHandle_t handle, int m, int n, c10::complex<float>* A, int lda, int* lwork
) {
  // 调用 cuSOLVER 函数计算复数单精度 GEQRF 操作所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(
      cusolverDnCgeqrf_bufferSize(handle, m, n, reinterpret_cast<cuComplex*>(A), lda, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<double>>(
  cusolverDnHandle_t handle, int m, int n, c10::complex<double>* A, int lda, int* lwork
) {
  // 调用 cuSOLVER 函数计算复数双精度 GEQRF 操作所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(
      cusolverDnZgeqrf_bufferSize(handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, lwork));
}
    // 定义 CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES 宏，其参数类型为 c10::complex<float>
    CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  // 使用 cusolverDnCgeqrf_bufferSize 函数获取 GEQRF 操作所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnCgeqrf_bufferSize(
      // cuSolver 句柄
      handle,
      // 矩阵的行数 m
      m,
      // 矩阵的列数 n
      n,
      // 将 A 强制转换为 cuComplex* 类型
      reinterpret_cast<cuComplex*>(A),
      // A 矩阵的 leading dimension
      lda,
      // 工作空间数组的大小
      lwork));
template <>
void geqrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSOLVER 库中的 cusolverDnZgeqrf_bufferSize 函数，用于计算 geqrf 操作所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, lwork));
}

template <>
void geqrf<float>(CUDASOLVER_GEQRF_ARGTYPES(float)) {
  // 调用 cuSOLVER 库中的 cusolverDnSgeqrf 函数，执行单精度浮点数矩阵的 GEQRF 分解
  TORCH_CUSOLVER_CHECK(
      cusolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
}

template <>
void geqrf<double>(CUDASOLVER_GEQRF_ARGTYPES(double)) {
  // 调用 cuSOLVER 库中的 cusolverDnDgeqrf 函数，执行双精度浮点数矩阵的 GEQRF 分解
  TORCH_CUSOLVER_CHECK(
      cusolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
}

template <>
void geqrf<c10::complex<float>>(
    CUDASOLVER_GEQRF_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSOLVER 库中的 cusolverDnCgeqrf 函数，执行单精度复数矩阵的 GEQRF 分解
  TORCH_CUSOLVER_CHECK(cusolverDnCgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<cuComplex*>(A),
      lda,
      reinterpret_cast<cuComplex*>(tau),
      reinterpret_cast<cuComplex*>(work),
      lwork,
      devInfo));
}

template <>
void geqrf<c10::complex<double>>(
    CUDASOLVER_GEQRF_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSOLVER 库中的 cusolverDnZgeqrf 函数，执行双精度复数矩阵的 GEQRF 分解
  TORCH_CUSOLVER_CHECK(cusolverDnZgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      reinterpret_cast<cuDoubleComplex*>(tau),
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork,
      devInfo));
}

template<>
void potrs<float>(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float *A, int lda, float *B, int ldb, int *devInfo
) {
  // 调用 cuSOLVER 库中的 cusolverDnSpotrs 函数，解线性方程组 Ax = B，其中 A 是单精度浮点数对称正定矩阵
  TORCH_CUSOLVER_CHECK(cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
}

template<>
void potrs<double>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo
) {
  // 调用 cuSOLVER 库中的 cusolverDnDpotrs 函数，解线性方程组 Ax = B，其中 A 是双精度浮点数对称正定矩阵
  TORCH_CUSOLVER_CHECK(cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
}

template<>
void potrs<c10::complex<float>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const c10::complex<float> *A, int lda, c10::complex<float> *B, int ldb, int *devInfo
) {
  // 调用 cuSOLVER 库中的 cusolverDnCpotrs 函数，解线性方程组 Ax = B，其中 A 是单精度复数对称正定矩阵
  TORCH_CUSOLVER_CHECK(cusolverDnCpotrs(
    handle, uplo, n, nrhs,
    reinterpret_cast<const cuComplex*>(A),
    lda,
    reinterpret_cast<cuComplex*>(B),
    ldb, devInfo));
}

template<>
void potrs<c10::complex<double>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const c10::complex<double> *A, int lda, c10::complex<double> *B, int ldb, int *devInfo
) {
  // 调用 cuSOLVER 库中的 cusolverDnZpotrs 函数，解线性方程组 Ax = B，其中 A 是双精度复数对称正定矩阵
  TORCH_CUSOLVER_CHECK(cusolverDnZpotrs(
    handle, uplo, n, nrhs,
    reinterpret_cast<const cuDoubleComplex*>(A),
    lda,
    reinterpret_cast<cuDoubleComplex*>(B),
    ldb, devInfo));
}

template<>
void potrsBatched<float>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, float *Aarray[], int lda, float *Barray[], int ldb, int *info, int batchSize
) {
  // 调用 cuSOLVER 库中的 cusolverDnSpotrsBatched 函数，解多个批次的线性方程组 Ax = B，其中 A 是单精度浮点数对称正定矩阵
  TORCH_CUSOLVER_CHECK(cusolverDnSpotrsBatched(handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
}

template<>
// 解决方程组的批量求解函数，处理 double 类型的数据
void potrsBatched<double>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, double *Aarray[], int lda, double *Barray[], int ldb, int *info, int batchSize
) {
  // 调用 cuSolver 库中的函数 cusolverDnDpotrsBatched 进行批量的 Cholesky 分解求解操作
  TORCH_CUSOLVER_CHECK(cusolverDnDpotrsBatched(handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
}

// 模板特化：解决方程组的批量求解函数，处理 c10::complex<float> 类型的数据
template<>
void potrsBatched<c10::complex<float>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, c10::complex<float> *Aarray[], int lda, c10::complex<float> *Barray[], int ldb, int *info, int batchSize
) {
  // 调用 cuSolver 库中的函数 cusolverDnCpotrsBatched 进行批量的 Cholesky 分解求解操作，处理复数 float 类型
  TORCH_CUSOLVER_CHECK(cusolverDnCpotrsBatched(
    handle, uplo, n, nrhs,
    reinterpret_cast<cuComplex**>(Aarray),
    lda,
    reinterpret_cast<cuComplex**>(Barray),
    ldb, info, batchSize));
}

// 模板特化：解决方程组的批量求解函数，处理 c10::complex<double> 类型的数据
template<>
void potrsBatched<c10::complex<double>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, c10::complex<double> *Aarray[], int lda, c10::complex<double> *Barray[], int ldb, int *info, int batchSize
) {
  // 调用 cuSolver 库中的函数 cusolverDnZpotrsBatched 进行批量的 Cholesky 分解求解操作，处理复数 double 类型
  TORCH_CUSOLVER_CHECK(cusolverDnZpotrsBatched(
    handle, uplo, n, nrhs,
    reinterpret_cast<cuDoubleComplex**>(Aarray),
    lda,
    reinterpret_cast<cuDoubleComplex**>(Barray),
    ldb, info, batchSize));
}

// 模板特化：计算 orgqr 函数需要的缓冲区大小，处理 float 类型的数据
template <>
void orgqr_buffersize<float>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    const float* A, int lda,
    const float* tau, int* lwork) {
  // 调用 cuSolver 库中的函数 cusolverDnSorgqr_bufferSize 计算 orgqr 函数所需的工作空间大小
  TORCH_CUSOLVER_CHECK(
      cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}

// 模板特化：计算 orgqr 函数需要的缓冲区大小，处理 double 类型的数据
template <>
void orgqr_buffersize<double>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    const double* A, int lda,
    const double* tau, int* lwork) {
  // 调用 cuSolver 库中的函数 cusolverDnDorgqr_bufferSize 计算 orgqr 函数所需的工作空间大小
  TORCH_CUSOLVER_CHECK(
      cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}

// 模板特化：计算 orgqr 函数需要的缓冲区大小，处理复数 float 类型的数据
template <>
void orgqr_buffersize<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    const c10::complex<float>* A, int lda,
    const c10::complex<float>* tau, int* lwork) {
  // 调用 cuSolver 库中的函数 cusolverDnCungqr_bufferSize 计算 orgqr 函数所需的工作空间大小，处理复数 float 类型
  TORCH_CUSOLVER_CHECK(cusolverDnCungqr_bufferSize(
      handle,
      m, n, k,
      reinterpret_cast<const cuComplex*>(A), lda,
      reinterpret_cast<const cuComplex*>(tau), lwork));
}

// 模板特化：计算 orgqr 函数需要的缓冲区大小，处理复数 double 类型的数据
template <>
void orgqr_buffersize<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    const c10::complex<double>* A, int lda,
    const c10::complex<double>* tau, int* lwork) {
  // 调用 cuSolver 库中的函数 cusolverDnZungqr_bufferSize 计算 orgqr 函数所需的工作空间大小，处理复数 double 类型
  TORCH_CUSOLVER_CHECK(cusolverDnZungqr_bufferSize(
      handle,
      m, n, k,
      reinterpret_cast<const cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau), lwork));
}

// 模板特化：计算 QR 分解的 orgqr 函数，处理 float 类型的数据
template <>
void orgqr<float>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    float* A, int lda,
    const float* tau,
    float* work, int lwork,
    int* devInfo) {
  // 调用 cuSolver 库中的函数 cusolverDnSorgqr 进行 QR 分解计算，处理 float 类型的数据
  TORCH_CUSOLVER_CHECK(
      cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

// 模板特化：计算 QR 分解的 orgqr 函数，处理 double 类型的数据
template <>
void orgqr<double>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    double* A, int lda,
    const double* tau,
    double* work, int lwork,
    int* devInfo) {
  // 调用 cuSolver 库中的函数 cusolverDnDorgqr 进行 QR 分解计算，处理 double 类型的数据
  TORCH_CUSOLVER_CHECK(
      cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}
    # 调用 cuSOLVER 库中的 dorgqr 函数，用于计算 QR 分解后的 Q 矩阵
    TORCH_CUSOLVER_CHECK(
        cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
template <>
void orgqr<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    c10::complex<float>* A, int lda,
    const c10::complex<float>* tau,
    c10::complex<float>* work, int lwork,
    int* devInfo) {
  // 调用 cuSolver 库中的 cusolverDnCungqr 函数来计算 Q 矩阵，针对单精度复数类型
  TORCH_CUSOLVER_CHECK(cusolverDnCungqr(
      handle,
      m, n, k,
      reinterpret_cast<cuComplex*>(A), lda,
      reinterpret_cast<const cuComplex*>(tau),
      reinterpret_cast<cuComplex*>(work), lwork,
      devInfo));
}

template <>
void orgqr<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    c10::complex<double>* A, int lda,
    const c10::complex<double>* tau,
    c10::complex<double>* work, int lwork,
    int* devInfo) {
  // 调用 cuSolver 库中的 cusolverDnZungqr 函数来计算 Q 矩阵，针对双精度复数类型
  TORCH_CUSOLVER_CHECK(cusolverDnZungqr(
      handle,
      m, n, k,
      reinterpret_cast<cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau),
      reinterpret_cast<cuDoubleComplex*>(work), lwork,
      devInfo));
}

template <>
void ormqr_bufferSize<float>(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(float)) {
  // 调用 cuSolver 库中的 cusolverDnSormqr_bufferSize 函数，获取单精度浮点数类型的 ormqr 函数所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(
      cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
}

template <>
void ormqr_bufferSize<double>(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(double)) {
  // 调用 cuSolver 库中的 cusolverDnDormqr_bufferSize 函数，获取双精度浮点数类型的 ormqr 函数所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(
      cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
}

template <>
void ormqr_bufferSize<c10::complex<float>>(
    CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSolver 库中的 cusolverDnCunmqr_bufferSize 函数，获取单精度复数类型的 ormqr 函数所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnCunmqr_bufferSize(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuComplex*>(A), lda,
      reinterpret_cast<const cuComplex*>(tau),
      reinterpret_cast<const cuComplex*>(C), ldc,
      lwork));
}

template <>
void ormqr_bufferSize<c10::complex<double>>(
    CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSolver 库中的 cusolverDnZunmqr_bufferSize 函数，获取双精度复数类型的 ormqr 函数所需的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnZunmqr_bufferSize(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau),
      reinterpret_cast<const cuDoubleComplex*>(C), ldc,
      lwork));
}

template <>
void ormqr<float>(CUDASOLVER_ORMQR_ARGTYPES(float)) {
  // 调用 cuSolver 库中的 cusolverDnSormqr 函数，执行单精度浮点数类型的 ormqr 运算
  TORCH_CUSOLVER_CHECK(
      cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
}

template <>
void ormqr<double>(CUDASOLVER_ORMQR_ARGTYPES(double)) {
  // 调用 cuSolver 库中的 cusolverDnDormqr 函数，执行双精度浮点数类型的 ormqr 运算
  TORCH_CUSOLVER_CHECK(
      cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
}

template <>
void ormqr<c10::complex<float>>(CUDASOLVER_ORMQR_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSolver 库中的 cusolverDnCunmqr 函数，执行单精度复数类型的 ormqr 运算
  TORCH_CUSOLVER_CHECK(cusolverDnCunmqr(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuComplex*>(A), lda,
      reinterpret_cast<const cuComplex*>(tau),
      reinterpret_cast<cuComplex*>(C), ldc,
      reinterpret_cast<cuComplex*>(work), lwork,
      devInfo));
}

template <>
void ormqr<c10::complex<double>>(CUDASOLVER_ORMQR_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSolver 库中的 cusolverDnZunmqr 函数，执行双精度复数类型的 ormqr 运算
  TORCH_CUSOLVER_CHECK(cusolverDnZunmqr(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau),
      reinterpret_cast<cuDoubleComplex*>(C), ldc,
      reinterpret_cast<cuDoubleComplex*>(work), lwork,
      devInfo));
}
// 实现了 cusolverDnZunmqr 函数，用于在 CUDA 中进行 QR 分解后的矩阵乘法操作
void ormqr<c10::complex<double>>(CUDASOLVER_ORMQR_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSolver 库中的 cusolverDnZunmqr 函数执行 QR 分解后的矩阵乘法操作
  TORCH_CUSOLVER_CHECK(cusolverDnZunmqr(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau),
      reinterpret_cast<cuDoubleComplex*>(C), ldc,
      reinterpret_cast<cuDoubleComplex*>(work), lwork,
      devInfo));
}

#ifdef USE_CUSOLVER_64_BIT

// 返回 float 类型的 CUDA 数据类型
template<> cudaDataType get_cusolver_datatype<float>() { return CUDA_R_32F; }
// 返回 double 类型的 CUDA 数据类型
template<> cudaDataType get_cusolver_datatype<double>() { return CUDA_R_64F; }
// 返回复数 float 类型的 CUDA 数据类型
template<> cudaDataType get_cusolver_datatype<c10::complex<float>>() { return CUDA_C_32F; }
// 返回复数 double 类型的 CUDA 数据类型
template<> cudaDataType get_cusolver_datatype<c10::complex<double>>() { return CUDA_C_64F; }

// 计算调用 cusolverDnXpotrf_bufferSize 函数所需的设备和主机端工作空间大小
void xpotrf_buffersize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType computeType, size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  // 调用 cusolverDnXpotrf_bufferSize 函数获取 Cholesky 分解所需的设备和主机端工作空间大小
  TORCH_CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
    handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost
  ));
}

// 执行 Cholesky 分解操作，将结果存储在 A 中
void xpotrf(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A,
    int64_t lda, cudaDataType computeType, void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost,
    int *info) {
  // 调用 cusolverDnXpotrf 函数执行 Cholesky 分解
  TORCH_CUSOLVER_CHECK(cusolverDnXpotrf(
    handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info
  ));
}
#endif // USE_CUSOLVER_64_BIT

// 获取 float 类型数据的特征值分解所需的工作空间大小
template <>
void syevd_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float* A,
    int lda,
    const float* W,
    int* lwork) {
  // 调用 cusolverDnSsyevd_bufferSize 函数获取 float 类型数据的特征值分解所需的工作空间大小
  TORCH_CUSOLVER_CHECK(
      cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork));
}

// 获取 double 类型数据的特征值分解所需的工作空间大小
template <>
void syevd_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double* A,
    int lda,
    const double* W,
    int* lwork) {
  // 调用 cusolverDnDsyevd_bufferSize 函数获取 double 类型数据的特征值分解所需的工作空间大小
  TORCH_CUSOLVER_CHECK(
      cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork));
}

// 获取复数 float 类型数据的特征值分解所需的工作空间大小
template <>
void syevd_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork) {
  // 调用 cusolverDnCheevd_bufferSize 函数获取复数 float 类型数据的特征值分解所需的工作空间大小
  TORCH_CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuComplex*>(A),
      lda,
      W,
      lwork));
}

// 获取复数 double 类型数据的特征值分解所需的工作空间大小
template <>
void syevd_bufferSize<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork) {
  // 调用 cusolverDnZheevd_bufferSize 函数获取复数 double 类型数据的特征值分解所需的工作空间大小
  TORCH_CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuDoubleComplex*>(A),
      lda,
      W,
      lwork));
}
    // 调用 cusolver 库中的 cusolverDnZheevd_bufferSize 函数，用于获取执行 Hermitian 特征值问题所需的缓冲区大小
    TORCH_CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
        // cuSolver 库的句柄，用于管理 cuSolver 库的状态和资源
        handle,
        // 指定作业模式，'N' 表示仅计算特征值，不计算特征向量
        jobz,
        // 指定存储矩阵 A 的上三角部分还是下三角部分，这里可能是 'U' 或 'L'
        uplo,
        // 矩阵 A 的阶数（维度）
        n,
        // A 矩阵的数据，以 cuDoubleComplex 形式给出，需要进行类型转换
        reinterpret_cast<const cuDoubleComplex*>(A),
        // A 矩阵的 leading dimension，即每列之间的跨度，通常等于矩阵的行数 n
        lda,
        // 用于存储计算结果的数组，包括特征值或特征向量的实部
        W,
        // 工作区大小的输出参数，cusolverDnZheevd_bufferSize 将返回所需的工作区大小
        lwork));
}

template <>
void syevd<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    float* work,
    int lwork,
    int* info) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info));
}


# 关于 float 类型的矩阵的特化实现，使用 cusolver 库中的 ssyevd 函数求解特征值问题

template <>
void syevd<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    double* work,
    int lwork,
    int* info) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info));
}


# 关于 double 类型的矩阵的特化实现，使用 cusolver 库中的 dsyevd 函数求解特征值问题

template <>
void syevd<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<float>* A,
    int lda,
    float* W,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnCheevd(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuComplex*>(work),
      lwork,
      info));
}


# 关于复数 float 类型的矩阵的特化实现，使用 cusolver 库中的 cheevd 函数求解特征值问题

template <>
void syevd<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<double>* A,
    int lda,
    double* W,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnZheevd(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork,
      info));
}


# 关于复数 double 类型的矩阵的特化实现，使用 cusolver 库中的 zheevd 函数求解特征值问题

template <>
void syevj_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnSsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params));
}


# 关于 float 类型矩阵的特化实现，计算调用 cusolver 库中 ssyevj_bufferSize 函数所需的工作空间大小

template <>
void syevj_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnDsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params));
}


# 关于 double 类型矩阵的特化实现，计算调用 cusolver 库中 dsyevj_bufferSize 函数所需的工作空间大小

template <>
void syevj_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnCheevj_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuComplex*>(A),
      lda,
      W,
      lwork,
      params));
}


# 关于复数 float 类型矩阵的特化实现，计算调用 cusolver 库中 cheevj_bufferSize 函数所需的工作空间大小

template <>
void syevj_bufferSize<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnZheevj_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuDoubleComplex*>(A),
      lda,
      W,
      lwork,
      params));
}


# 关于复数 double 类型矩阵的特化实现，计算调用 cusolver 库中 zheevj_bufferSize 函数所需的工作空间大小
    # 调用 cusolverDnZheevj_bufferSize 函数，获取运行所需的缓冲区大小
    TORCH_CUSOLVER_CHECK(
        cusolverDnZheevj_bufferSize(
            handle,  # cusolverDn 上下文句柄
            jobz,    # 指定计算特征值或特征向量 ('N' 仅计算特征值，'V' 计算特征值和特征向量)
            uplo,    # 指定数组 A 的存储布局 ('U' 上三角部分有效，'L' 下三角部分有效)
            n,       # 矩阵的阶数
            reinterpret_cast<const cuDoubleComplex*>(A),  # 输入矩阵 A，类型转换为 cuDoubleComplex*
            lda,     # 矩阵 A 的首维长度
            W,       # 存储计算得到的特征值的数组
            lwork,   # 提供工作空间数组的长度或返回所需长度
            params   # cusolverDn 的特征值求解参数对象
        )
    );
}



// 模板特化：单精度复数的批量计算特征值问题的缓冲区大小
template <>
void syevjBatched_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  // 调用 cuSolver 库计算单精度复数批量特征值问题的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnCheevjBatched_bufferSize(
      handle, jobz, uplo, n, reinterpret_cast<const cuComplex*>(A), lda, W, lwork, params, batchsize));
}



// 模板特化：双精度复数的批量计算特征值问题的缓冲区大小
template <>
void syevjBatched_bufferSize<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  // 调用 cuSolver 库计算双精度复数批量特征值问题的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(
      handle, jobz, uplo, n, reinterpret_cast<const cuDoubleComplex*>(A), lda, W, lwork, params, batchsize));
}


这段代码是用于 CUDA 的 cuSolver 库中的特化模板函数，用于处理不同数据类型的特征值求解问题。特化函数中调用了 cuSolver 提供的具体求解函数，根据数据类型的不同进行了函数名称和参数类型的适配。
    int batchsize) {


// 定义一个名为 batchsize 的整数参数，用于指定批处理的大小
TORCH_CUSOLVER_CHECK(
    // 调用 cusolverDnCheevjBatched_bufferSize 函数进行错误检查，并获取缓冲区大小
    cusolverDnCheevjBatched_bufferSize(
        handle,                                      // CUSOLVER 句柄，用于调用 CUDA 套件
        jobz,                                        // 求解模式参数：'N' 表示不求特征向量，'V' 表示求特征向量
        uplo,                                        // 矩阵存储方式参数：'U' 表示上三角部分有效，'L' 表示下三角部分有效
        n,                                           // 矩阵维度参数，指定矩阵的大小
        reinterpret_cast<const cuComplex*>(A),        // 矩阵 A 的数据，以 cuComplex 形式进行解释
        lda,                                         // 矩阵 A 的 leading dimension，即 A 的列数
        W,                                           // 存储特征值的数组
        lwork,                                       // 工作空间大小
        params,                                      // 额外参数数组，可为 nullptr
        batchsize                                    // 批处理的数量
    ));
#ifdef USE_CUSOLVER_64_BIT


// 如果定义了 USE_CUSOLVER_64_BIT 宏，则进入以下代码段
void xpotrs(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
  // 调用 cusolverDnXpotrs 函数来解线性方程组
  TORCH_CUSOLVER_CHECK(cusolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info));
}


template <>


// 模板特化，处理复数 double 类型的 syevjBatched 函数
void syevjBatched<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<double>* A,
    int lda,
    double* W,
    c10::complex<double>* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  // 转换 A 和 work 到 cuDoubleComplex 类型，调用 cusolverDnZheevjBatched 函数进行特征值计算
  TORCH_CUSOLVER_CHECK(cusolverDnZheevjBatched(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork,
      info,
      params,
      batchsize));
}
// 特化模板函数 xgeqrf_bufferSize<float>(...)，计算单精度浮点数的 GEQRF 操作所需的缓冲区大小
void xgeqrf_bufferSize<float>(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(float)) {
  // 调用 cuSOLVER 库函数 cusolverDnXgeqrf_bufferSize 获取缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
      handle,                     // cuSOLVER 句柄
      params,                     // GEQRF 参数结构体
      m,                          // 矩阵的行数
      n,                          // 矩阵的列数
      CUDA_R_32F,                 // 输入矩阵 A 的数据类型为单精度浮点数
      reinterpret_cast<const void*>(A),  // 输入矩阵 A 的指针
      lda,                        // 矩阵 A 的 leading dimension
      CUDA_R_32F,                 // TAU 的数据类型为单精度浮点数
      reinterpret_cast<const void*>(tau),  // TAU 的指针
      CUDA_R_32F,                 // 输入工作空间的数据类型为单精度浮点数
      workspaceInBytesOnDevice,   // 设备端工作空间所需大小（字节数）
      workspaceInBytesOnHost));   // 主机端工作空间所需大小（字节数）
}

// 特化模板函数 xgeqrf_bufferSize<double>(...)，计算双精度浮点数的 GEQRF 操作所需的缓冲区大小
template <>
void xgeqrf_bufferSize<double>(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(double)) {
  // 调用 cuSOLVER 库函数 cusolverDnXgeqrf_bufferSize 获取缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
      handle,                     // cuSOLVER 句柄
      params,                     // GEQRF 参数结构体
      m,                          // 矩阵的行数
      n,                          // 矩阵的列数
      CUDA_R_64F,                 // 输入矩阵 A 的数据类型为双精度浮点数
      reinterpret_cast<const void*>(A),  // 输入矩阵 A 的指针
      lda,                        // 矩阵 A 的 leading dimension
      CUDA_R_64F,                 // TAU 的数据类型为双精度浮点数
      reinterpret_cast<const void*>(tau),  // TAU 的指针
      CUDA_R_64F,                 // 输入工作空间的数据类型为双精度浮点数
      workspaceInBytesOnDevice,   // 设备端工作空间所需大小（字节数）
      workspaceInBytesOnHost));   // 主机端工作空间所需大小（字节数）
}

// 特化模板函数 xgeqrf_bufferSize<c10::complex<float>>(...)，计算单精度复数的 GEQRF 操作所需的缓冲区大小
template <>
void xgeqrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSOLVER 库函数 cusolverDnXgeqrf_bufferSize 获取缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
      handle,                     // cuSOLVER 句柄
      params,                     // GEQRF 参数结构体
      m,                          // 矩阵的行数
      n,                          // 矩阵的列数
      CUDA_C_32F,                 // 输入矩阵 A 的数据类型为单精度复数
      reinterpret_cast<const void*>(A),  // 输入矩阵 A 的指针
      lda,                        // 矩阵 A 的 leading dimension
      CUDA_C_32F,                 // TAU 的数据类型为单精度复数
      reinterpret_cast<const void*>(tau),  // TAU 的指针
      CUDA_C_32F,                 // 输入工作空间的数据类型为单精度复数
      workspaceInBytesOnDevice,   // 设备端工作空间所需大小（字节数）
      workspaceInBytesOnHost));   // 主机端工作空间所需大小（字节数）
}

// 特化模板函数 xgeqrf_bufferSize<c10::complex<double>>(...)，计算双精度复数的 GEQRF 操作所需的缓冲区大小
template <>
void xgeqrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  // 调用 cuSOLVER 库函数 cusolverDnXgeqrf_bufferSize 获取缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
      handle,                     // cuSOLVER 句柄
      params,                     // GEQRF 参数结构体
      m,                          // 矩阵的行数
      n,                          // 矩阵的列数
      CUDA_C_64F,                 // 输入矩阵 A 的数据类型为双精度复数
      reinterpret_cast<const void*>(A),  // 输入矩阵 A 的指针
      lda,                        // 矩阵 A 的 leading dimension
      CUDA_C_64F,                 // TAU 的数据类型为双精度复数
      reinterpret_cast<const void*>(tau),  // TAU 的指针
      CUDA_C_64F,                 // 输入工作空间的数据类型为双精度复数
      workspaceInBytesOnDevice,   // 设备端工作空间所需大小（字节数）
      workspaceInBytesOnHost));   // 主机端工作空间所需大小（字节数）
}

// 特化模板函数 xgeqrf<float>(...)，执行单精度浮点数的 GEQRF 操作
template <>
void xgeqrf<float>(CUDASOLVER_XGEQRF_ARGTYPES(float)) {
  // 调用 cuSOLVER 库函数 cusolverDnXgeqrf 执行 GEQRF 操作
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
      handle,                     // cuSOLVER 句柄
      params,                     // GEQRF 参数结构体
      m,                          // 矩阵的行数
      n,                          // 矩阵的列数
      CUDA_R_32F,                 // 输入矩阵 A 的数据类型为单精度浮点数
      reinterpret_cast<void*>(A), // 输入矩阵 A 的指针
      lda,                        // 矩阵 A 的 leading dimension
      CUDA_R_32F,                 // TAU 的数据类型为单精度浮点数
      reinterpret_cast<void*>(tau),  // TAU 的指针
      CUDA_R_32F,                 // 输入设备端工作空间的数据类型为单精度浮点数
      reinterpret_cast<void*>(bufferOnDevice),   // 设备端工作空间的指针
      workspaceInBytesOnDevice,   // 设备端工作空间所需大小（字节数）
      reinterpret_cast<void*>(bufferOnHost),     // 主机端工作空间的指针
      workspaceInBytesOnHost,     // 主机端工作空间所需大小（字节数）
      info));                     // GEQRF 操作的信息状态
}

// 特化模板函数 xgeqrf<double>(...)，执行双精度浮点数的 GEQRF 操作
template <>
void xgeqrf<double>(CUDASOLVER_XGEQRF_ARGTYPES(double)) {
  // 调用 cuSOLVER 库函数 cusolverDnXgeqrf 执行 GEQRF 操作
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
      handle,                     // cuSOLVER 句柄
      params,                     // GEQRF 参数结构体
      m,                          // 矩阵的行数
      n,                          // 矩阵的列数
      CUDA_R_64F,                 // 输入矩阵 A 的数据类型为双精度浮点数
void xgeqrf<c10::complex<float>>(CUDASOLVER_XGEQRF_ARGTYPES(c10::complex<float>)) {
  // 调用 cuSolver 库中的 xgeqrf 函数来执行 QR 分解
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
      handle,                     // cuSolver 句柄
      params,                     // cuSolver 参数
      m,                          // 矩阵 A 的行数
      n,                          // 矩阵 A 的列数
      CUDA_C_32F,                 // 数据类型：32 位浮点数
      reinterpret_cast<void*>(A), // 矩阵 A 的数据，强制类型转换为 void*
      lda,                        // 主存储顺序中 A 的领先维度
      CUDA_C_32F,                 // 数据类型：32 位浮点数
      reinterpret_cast<void*>(tau),   // 存储反射系数的数组 tau，强制类型转换为 void*
      CUDA_C_32F,                 // 数据类型：32 位浮点数
      reinterpret_cast<void*>(bufferOnDevice),  // 在设备上分配的工作空间，强制类型转换为 void*
      workspaceInBytesOnDevice,   // 设备上工作空间的字节数
      reinterpret_cast<void*>(bufferOnHost),    // 在主机上分配的工作空间，强制类型转换为 void*
      workspaceInBytesOnHost,     // 主机上工作空间的字节数
      info));                     // 返回状态信息
}

template <>
void xgeqrf<c10::complex<double>>(CUDASOLVER_XGEQRF_ARGTYPES(c10::complex<double>)) {
  // 特化模板，使用双精度复数执行 xgeqrf 函数
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
      handle,                     // cuSolver 句柄
      params,                     // cuSolver 参数
      m,                          // 矩阵 A 的行数
      n,                          // 矩阵 A 的列数
      CUDA_C_64F,                 // 数据类型：64 位双精度浮点数
      reinterpret_cast<void*>(A), // 矩阵 A 的数据，强制类型转换为 void*
      lda,                        // 主存储顺序中 A 的领先维度
      CUDA_C_64F,                 // 数据类型：64 位双精度浮点数
      reinterpret_cast<void*>(tau),   // 存储反射系数的数组 tau，强制类型转换为 void*
      CUDA_C_64F,                 // 数据类型：64 位双精度浮点数
      reinterpret_cast<void*>(bufferOnDevice),  // 在设备上分配的工作空间，强制类型转换为 void*
      workspaceInBytesOnDevice,   // 设备上工作空间的字节数
      reinterpret_cast<void*>(bufferOnHost),    // 在主机上分配的工作空间，强制类型转换为 void*
      workspaceInBytesOnHost,     // 主机上工作空间的字节数
      info));                     // 返回状态信息
}

template <>
void xsyevd_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const float* A,
    int64_t lda,
    const float* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  // 特化模板，获取浮点数类型的 xSYEVD 需要的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
      handle,                     // cuSolver 句柄
      params,                     // cuSolver 参数
      jobz,                       // 计算特征值还是特征向量
      uplo,                       // 上三角部分存储方式
      n,                          // 矩阵 A 的维度
      CUDA_R_32F,                 // A 的数据类型：32 位浮点数
      reinterpret_cast<const void*>(A),    // 矩阵 A 的数据，强制类型转换为 const void*
      lda,                        // 主存储顺序中 A 的领先维度
      CUDA_R_32F,                 // W 的数据类型：32 位浮点数
      reinterpret_cast<const void*>(W),    // 矩阵 W 的数据，强制类型转换为 const void*
      CUDA_R_32F,                 // 工作空间的数据类型：32 位浮点数
      workspaceInBytesOnDevice,   // 设备上工作空间的字节数
      workspaceInBytesOnHost));   // 主机上工作空间的字节数
}

template <>
void xsyevd_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const double* A,
    int64_t lda,
    const double* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  // 特化模板，获取双精度浮点数类型的 xSYEVD 需要的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
      handle,                     // cuSolver 句柄
      params,                     // cuSolver 参数
      jobz,                       // 计算特征值还是特征向量
      uplo,                       // 上三角部分存储方式
      n,                          // 矩阵 A 的维度
      CUDA_R_64F,                 // A 的数据类型：64 位双精度浮点数
      reinterpret_cast<const void*>(A),    // 矩阵 A 的数据，强制类型转换为 const void*
      lda,                        // 主存储顺序中 A 的领先维度
      CUDA_R_64F,                 // W 的数据类型：64 位双精度浮点数
      reinterpret_cast<const void*>(W),    // 矩阵 W 的数据，强制类型转换为 const void*
      CUDA_R_64F,                 // 工作空间的数据类型：64 位双精度浮点数
      workspaceInBytesOnDevice,   // 设备上工作空间的字节数
      workspaceInBytesOnHost));   // 主机上工作空间的字节数
}

template <>
void xsyevd_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const c10::complex<float>* A,
    int64_t lda,
    const float* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  // 特化模板，获取单精度复数类型的 xSYEVD 需要的缓冲区大小
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
      handle,                     // cuSolver 句柄
      params,                     // cuSolver 参数
      jobz,                       // 计算特征值还是特征向量
      uplo,                       // 上三角部分存储方式
      n,                          // 矩阵 A 的维度
      CUDA_C_32F,                 // A 的数据类型：32 位复数
      reinterpret_cast<const void*>(A),    // 矩阵 A 的数据，强制类型转换为 const void*
      lda,                        // 主存储顺序中 A 的领先维度
      CUDA_R_32F,                 // W 的数据类型：32 位浮点数
      reinterpret_cast<const void*>(W),    // 矩阵 W 的数据，强制类型转换为 const void*
      CUDA_C_32F,                 // 工作空间的数据类型：32 位复数
      workspaceInBytesOnDevice,   // 设备上工作空间的字节数
      workspaceInBytesOnHost));   // 主机上工作空间的字节数
}

template <>
void xsyevd_bufferSize<c10
    // 调用 cusolverDnXsyevd_bufferSize 函数获取执行 XSYEVD 计算所需的设备和主机工作空间的字节数大小
    TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
        handle,                           // cusolverDnHandle_t 类型的句柄，表示 cusolverDN 库的操作句柄
        params,                           // cusolverDnParams_t 类型的参数，用于 cusolverDN 计算的参数
        jobz,                             // cusolverEigMode_t 类型的枚举，指定计算特征值还是特征向量
        uplo,                             // cublasFillMode_t 类型的枚举，指定矩阵 A 的上三角部分存储方式
        n,                                // int64_t 类型，矩阵 A 的维度大小
        CUDA_C_64F,                       // 指定 A 矩阵的数据类型为复数双精度浮点数
        reinterpret_cast<const void*>(A), // A 矩阵的数据，以 void* 形式传递
        lda,                              // int64_t 类型，A 矩阵的 leading dimension（领先维度）
        CUDA_R_64F,                       // 指定 W 向量的数据类型为双精度浮点数
        reinterpret_cast<const void*>(W), // W 向量的数据，以 void* 形式传递
        CUDA_C_64F,                       // 指定计算过程中使用的复数双精度浮点数数据类型
        workspaceInBytesOnDevice,         // 指向设备工作空间大小的指针
        workspaceInBytesOnHost));         // 指向主机工作空间大小的指针
}



// 模板特化：对 float 类型进行求解对称矩阵特征值和特征向量的操作
template <>
void xsyevd<float>(
    cusolverDnHandle_t handle,                      // cuSolver DN 句柄
    cusolverDnParams_t params,                      // cuSolver DN 参数
    cusolverEigMode_t jobz,                         // 计算特征值或特征向量
    cublasFillMode_t uplo,                          // 矩阵 A 的存储方式（上三角或下三角）
    int64_t n,                                      // 矩阵 A 的维度
    float* A,                                       // 输入矩阵 A
    int64_t lda,                                    // 矩阵 A 的 leading dimension
    float* W,                                       // 存储特征值的数组
    float* bufferOnDevice,                          // 设备上的工作空间缓冲区
    size_t workspaceInBytesOnDevice,                // 设备上工作空间缓冲区的大小（字节）
    float* bufferOnHost,                            // 主机上的工作空间缓冲区
    size_t workspaceInBytesOnHost,                  // 主机上工作空间缓冲区的大小（字节）
    int* info) {                                    // 返回的状态信息
  // 调用 cuSolver 的特化版本 xsyevd 进行特征值和特征向量计算，并检查错误
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_R_32F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<void*>(W),
      CUDA_R_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

// 模板特化：对 double 类型进行求解对称矩阵特征值和特征向量的操作
template <>
void xsyevd<double>(
    cusolverDnHandle_t handle,                      // cuSolver DN 句柄
    cusolverDnParams_t params,                      // cuSolver DN 参数
    cusolverEigMode_t jobz,                         // 计算特征值或特征向量
    cublasFillMode_t uplo,                          // 矩阵 A 的存储方式（上三角或下三角）
    int64_t n,                                      // 矩阵 A 的维度
    double* A,                                      // 输入矩阵 A
    int64_t lda,                                    // 矩阵 A 的 leading dimension
    double* W,                                      // 存储特征值的数组
    double* bufferOnDevice,                         // 设备上的工作空间缓冲区
    size_t workspaceInBytesOnDevice,                // 设备上工作空间缓冲区的大小（字节）
    double* bufferOnHost,                           // 主机上的工作空间缓冲区
    size_t workspaceInBytesOnHost,                  // 主机上工作空间缓冲区的大小（字节）
    int* info) {                                    // 返回的状态信息
  // 调用 cuSolver 的特化版本 xsyevd 进行特征值和特征向量计算，并检查错误
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_R_64F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<void*>(W),
      CUDA_R_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

// 模板特化：对复数 c10::complex<float> 类型进行求解对称矩阵特征值和特征向量的操作
template <>
void xsyevd<c10::complex<float>, float>(
    cusolverDnHandle_t handle,                      // cuSolver DN 句柄
    cusolverDnParams_t params,                      // cuSolver DN 参数
    cusolverEigMode_t jobz,                         // 计算特征值或特征向量
    cublasFillMode_t uplo,                          // 矩阵 A 的存储方式（上三角或下三角）
    int64_t n,                                      // 矩阵 A 的维度
    c10::complex<float>* A,                         // 输入矩阵 A
    int64_t lda,                                    // 矩阵 A 的 leading dimension
    float* W,                                       // 存储特征值的数组
    c10::complex<float>* bufferOnDevice,            // 设备上的工作空间缓冲区
    size_t workspaceInBytesOnDevice,                // 设备上工作空间缓冲区的大小（字节）
    c10::complex<float>* bufferOnHost,              // 主机上的工作空间缓冲区
    size_t workspaceInBytesOnHost,                  // 主机上工作空间缓冲区的大小（字节）
    int* info) {                                    // 返回的状态信息
  // 调用 cuSolver 的特化版本 xsyevd 进行特征值和特征向量计算，并检查错误
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_C_32F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<void*>(W),
      CUDA_C_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

// 模板特化：对复数 c10::complex<double> 类型进行求解对称矩阵特征值和特征向量的操作
template <>
void xsyevd<c10::complex<double>, double>(
    cusolverDnHandle_t handle,                      // cuSolver DN 句柄
    cusolverDnParams_t params,                      // cuSolver DN 参数
    cusolverEigMode_t jobz,                         // 计算特征值或特征向量
    cublasFillMode_t uplo,                          // 矩阵 A 的存储方式（上三角或下三角）
    int64_t n,                                      // 矩阵 A 的维度
    c10::complex<double>* A,                        // 输入矩阵 A
    int64_t lda,                                    // 矩阵 A 的 leading dimension
    double* W,                                      // 存储特征值的数组
    c10::complex<double>* bufferOnDevice,           // 设备上的工作空间缓冲区
    size_t workspaceInBytesOnDevice,                // 设备上工作空间缓冲区的大小（字节）
    c10::complex<double>* bufferOnHost,             // 主机上的工作空间缓冲区
    size_t workspaceInBytesOnHost,                  // 主机上工作空间缓冲区的大小（字节）
    int* info) {                                    // 返回的状态信息
  // 调用 cuSolver 的特化版本 xsyevd 进行特征值和特征向量计算，并检查错误
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_C_64F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<void*>(W),
      CUDA_C_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}
    TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
        handle,
        params,
        jobz,
        uplo,
        n,
        CUDA_C_64F,
        reinterpret_cast<void*>(A),
        lda,
        CUDA_R_64F,
        reinterpret_cast<void*>(W),
        CUDA_C_64F,
        reinterpret_cast<void*>(bufferOnDevice),
        workspaceInBytesOnDevice,
        reinterpret_cast<void*>(bufferOnHost),
        workspaceInBytesOnHost,
        info));



// 调用 cuSOLVER 库中的 cusolverDnXsyevd 函数，执行对称特征值问题的求解
TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
    handle,                     // cuSOLVER 句柄，用于管理 cuSOLVER 库的状态和资源
    params,                     // cuSOLVER 求解器参数对象
    jobz,                       // 指定计算特征向量 ('V' or 'N')
    uplo,                       // 指定矩阵的存储格式 ('U' or 'L')
    n,                          // 矩阵的阶数
    CUDA_C_64F,                 // 矩阵 A 的数据类型 (复数双精度浮点数)
    reinterpret_cast<void*>(A), // 输入矩阵 A 的指针，强制类型转换为 void*
    lda,                        // 矩阵 A 的 leading dimension
    CUDA_R_64F,                 // 特征值的数据类型 (实数双精度浮点数)
    reinterpret_cast<void*>(W), // 输出特征值的指针，强制类型转换为 void*
    CUDA_C_64F,                 // 特征向量的数据类型 (复数双精度浮点数)
    reinterpret_cast<void*>(bufferOnDevice), // 设备端工作区的指针，强制类型转换为 void*
    workspaceInBytesOnDevice,   // 设备端工作区大小（字节）
    reinterpret_cast<void*>(bufferOnHost),   // 主机端工作区的指针，强制类型转换为 void*
    workspaceInBytesOnHost,     // 主机端工作区大小（字节）
    info));                     // 输出参数，返回函数执行状态信息
} // namespace at::cuda::solver
#endif // CUDART_VERSION
```