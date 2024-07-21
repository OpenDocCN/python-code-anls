# `.\pytorch\aten\src\ATen\native\BatchLinearAlgebra.cpp`

```
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper.h>
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <ATen/ops/_linalg_eigh.h>
#include <ATen/ops/_linalg_eigh_meta.h>
#include <ATen/ops/_linalg_eigh_native.h>
#include <ATen/ops/_linalg_eigvals.h>
#include <ATen/ops/_linalg_eigvals_native.h>
#include <ATen/ops/_linalg_solve_ex.h>
#include <ATen/ops/_linalg_solve_ex_meta.h>
#include <ATen/ops/_linalg_solve_ex_native.h>
#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_linalg_svd_meta.h>
#include <ATen/ops/_linalg_svd_native.h>
#include <ATen/ops/_lu_with_info_native.h>
#include <ATen/ops/all.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cholesky.h>
#include <ATen/ops/cholesky_inverse.h>
#include <ATen/ops/cholesky_inverse_native.h>
#include <ATen/ops/cholesky_native.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/cholesky_solve_native.h>
#include <ATen/ops/clone.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/geqrf.h>
#include <ATen/ops/geqrf_native.h>
#include <ATen/ops/inverse_native.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <ATen/ops/linalg_cholesky_ex_meta.h>
#include <ATen/ops/linalg_cholesky_ex_native.h>
#include <ATen/ops/linalg_cholesky_native.h>
#include <ATen/ops/linalg_eig.h>
#include <ATen/ops/linalg_eig_native.h>
#include <ATen/ops/linalg_eigh_native.h>
#include <ATen/ops/linalg_eigvals.h>
#include <ATen/ops/linalg_eigvals_native.h>
#include <ATen/ops/linalg_eigvalsh_native.h>
#include <ATen/ops/linalg_householder_product.h>
#include <ATen/ops/linalg_householder_product_native.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#include <ATen/ops/linalg_inv_native.h>
#include <ATen/ops/linalg_ldl_factor_ex.h>
#include <ATen/ops/linalg_ldl_factor_ex_meta.h>
#include <ATen/ops/linalg_ldl_factor_ex_native.h>
#include <ATen/ops/linalg_ldl_factor_native.h>
#include <ATen/ops/linalg_ldl_solve_meta.h>
#include <ATen/ops/linalg_ldl_solve_native.h>
#include <ATen/ops/linalg_lstsq.h>
#include <ATen/ops/linalg_lstsq_native.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_lu_factor_ex_meta.h>
#endif



#ifndef AT_PER_OPERATOR_HEADERS
如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含以下头文件：
#include <ATen/Functions.h>：包含通用的 ATen 函数声明和定义。
#include <ATen/NativeFunctions.h>：包含 ATen 的本地函数声明和定义。
#else
如果定义了 AT_PER_OPERATOR_HEADERS 宏，则包含以下头文件：
#include <ATen/ops/_cholesky_solve_helper.h> 至 <ATen/ops/linalg_lu_factor_ex_meta.h>：包含了各种具体操作的头文件，例如矩阵分解、求逆、特征值计算等。
#endif


这段代码根据宏定义选择性地包含了不同的ATen头文件，用于提供不同的操作和功能支持。
// 包含 ATen 库中的各种线性代数运算的头文件

#include <ATen/ops/linalg_lu_factor_ex_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_lu_meta.h>
#include <ATen/ops/linalg_lu_native.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_lu_solve_meta.h>
#include <ATen/ops/linalg_lu_solve_native.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_qr_meta.h>
#include <ATen/ops/linalg_qr_native.h>
#include <ATen/ops/linalg_solve_ex.h>
#include <ATen/ops/linalg_solve_ex_native.h>
#include <ATen/ops/linalg_solve_native.h>
#include <ATen/ops/linalg_solve_triangular_native.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/linalg_svd_native.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_svdvals_native.h>
#include <ATen/ops/linalg_vander_native.h>
#include <ATen/ops/linalg_vecdot_native.h>
#include <ATen/ops/lu_solve_native.h>
#include <ATen/ops/lu_unpack.h>
#include <ATen/ops/lu_unpack_meta.h>
#include <ATen/ops/lu_unpack_native.h>
#include <ATen/ops/orgqr_native.h>
#include <ATen/ops/ormqr_native.h>
#include <ATen/ops/qr_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/svd_native.h>
#include <ATen/ops/triangular_solve_meta.h>
#include <ATen/ops/triangular_solve_native.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/vdot.h>
#include <ATen/ops/zeros.h>

// 如果构建时启用了 LAPACK 支持，则需要注册以下 LAPACK 实现

#if AT_BUILD_WITH_LAPACK()

// LAPACK 中的 getrf 函数声明
extern "C" void zgetrf_(int *m, int *n, std::complex<double> *a, int *lda, int *ipiv, int *info);
extern "C" void cgetrf_(int *m, int *n, std::complex<float> *a, int *lda, int *ipiv, int *info);
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);

// LAPACK 中的 potrs 函数声明
extern "C" void zpotrs_(char *uplo, int *n, int *nrhs, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, int *info);
extern "C" void cpotrs_(char *uplo, int *n, int *nrhs, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb, int *info);
extern "C" void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern "C" void spotrs_(char *uplo, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);

// LAPACK 中的 potrf 函数声明
extern "C" void zpotrf_(char *uplo, int *n, std::complex<double> *a, int *lda, int *info);
extern "C" void cpotrf_(char *uplo, int *n, std::complex<float> *a, int *lda, int *info);
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);

// LAPACK 中的 potri 函数声明
extern "C" void zpotri_(char *uplo, int *n, std::complex<double> *a, int *lda, int *info);

#endif
// 定义调用 LAPACK 库中解决复数矩阵正定或对称正定线性方程组的函数声明

// 解决复数单精度矩阵的逆问题
extern "C" void cpotri_(char *uplo, int *n, std::complex<float> *a, int *lda, int *info);

// 解决双精度矩阵的逆问题
extern "C" void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);

// 解决单精度矩阵的逆问题
extern "C" void spotri_(char *uplo, int *n, float *a, int *lda, int *info);

// 解决双精度对称正定矩阵的 LU 分解问题
extern "C" void dsytrf_(
    char* uplo,
    int* n,
    double* a,
    int* lda,
    int* ipiv,
    double* work,
    int* lwork,
    int* info);

// 解决单精度对称正定矩阵的 LU 分解问题
extern "C" void ssytrf_(
    char* uplo,
    int* n,
    float* a,
    int* lda,
    int* ipiv,
    float* work,
    int* lwork,
    int* info);

// 解决双精度复数对称正定矩阵的 LU 分解问题
extern "C" void zsytrf_(
    char* uplo,
    int* n,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* work,
    int* lwork,
    int* info);

// 解决单精度复数对称正定矩阵的 LU 分解问题
extern "C" void csytrf_(
    char* uplo,
    int* n,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* work,
    int* lwork,
    int* info);

// 解决双精度复数 Hermitian 正定矩阵的 LU 分解问题
extern "C" void zhetrf_(
    char* uplo,
    int* n,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* work,
    int* lwork,
    int* info);

// 解决单精度复数 Hermitian 正定矩阵的 LU 分解问题
extern "C" void chetrf_(
    char* uplo,
    int* n,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* work,
    int* lwork,
    int* info);

// 解决双精度对称矩阵线性方程组的求解问题
extern "C" void dsytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    double* a,
    int* lda,
    int* ipiv,
    double* b,
    int* ldb,
    int* info);

// 解决单精度对称矩阵线性方程组的求解问题
extern "C" void ssytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    float* a,
    int* lda,
    int* ipiv,
    float* b,
    int* ldb,
    int* info);

// 解决双精度复数对称矩阵线性方程组的求解问题
extern "C" void zsytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* b,
    int* ldb,
    int* info);

// 解决单精度复数对称矩阵线性方程组的求解问题
extern "C" void csytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* b,
    int* ldb,
    int* info);

// 解决双精度复数 Hermitian 矩阵线性方程组的求解问题
extern "C" void zhetrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* b,
    int* ldb,
    int* info);

// 解决单精度复数 Hermitian 矩阵线性方程组的求解问题
extern "C" void chetrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* b,
    int* ldb,
    int* info);

// 解决双精度复数矩阵的 QR 分解问题
extern "C" void zgeqrf_(int *m, int *n, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);

// 解决单精度复数矩阵的 QR 分解问题
extern "C" void cgeqrf_(int *m, int *n, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);

// 解决双精度矩阵的 QR 分解问题
extern "C" void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);

// 解决单精度矩阵的 QR 分解问题
extern "C" void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

// orgqr
// 声明了外部函数 zungqr_，用于计算非紧凑形式的 QR 分解（复双精度版本）
extern "C" void zungqr_(int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);
// 声明了外部函数 cungqr_，用于计算非紧凑形式的 QR 分解（复单精度版本）
extern "C" void cungqr_(int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
// 声明了外部函数 dorgqr_，用于计算非紧凑形式的 QR 分解（双精度版本）
extern "C" void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
// 声明了外部函数 sorgqr_，用于计算非紧凑形式的 QR 分解（单精度版本）
extern "C" void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

// 声明了外部函数 zunmqr_，用于计算矩阵乘以 Q 或 Q^H 的结果（复双精度版本）
extern "C" void zunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *c, int *ldc, std::complex<double> *work, int *lwork, int *info);
// 声明了外部函数 cunmqr_，用于计算矩阵乘以 Q 或 Q^H 的结果（复单精度版本）
extern "C" void cunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *c, int *ldc, std::complex<float> *work, int *lwork, int *info);
// 声明了外部函数 dormqr_，用于计算矩阵乘以 Q 或 Q^T 的结果（双精度版本）
extern "C" void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);
// 声明了外部函数 sormqr_，用于计算矩阵乘以 Q 或 Q^T 的结果（单精度版本）
extern "C" void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *lda, float *tau, float *c, int *ldc, float *work, int *lwork, int *info);

// 声明了外部函数 zheevd_，用于计算复对称矩阵的所有特征值和（可选）特征向量（复双精度版本）
extern "C" void zheevd_(char *jobz, char *uplo, int *n, std::complex<double> *a, int *lda, double *w, std::complex<double> *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);
// 声明了外部函数 cheevd_，用于计算复对称矩阵的所有特征值和（可选）特征向量（复单精度版本）
extern "C" void cheevd_(char *jobz, char *uplo, int *n, std::complex<float> *a, int *lda, float *w, std::complex<float> *work, int *lwork, float *rwork, int *lrwork, int *iwork, int *liwork, int *info);
// 声明了外部函数 dsyevd_，用于计算实对称矩阵的所有特征值和（可选）特征向量（双精度版本）
extern "C" void dsyevd_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *iwork, int *liwork, int *info);
// 声明了外部函数 ssyevd_，用于计算实对称矩阵的所有特征值和（可选）特征向量（单精度版本）
extern "C" void ssyevd_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *iwork, int *liwork, int *info);

// 声明了外部函数 dgeev_，用于计算一般矩阵的特征值和（可选）特征向量（双精度版本）
extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double* vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
// 声明了外部函数 sgeev_，用于计算一般矩阵的特征值和（可选）特征向量（单精度版本）
extern "C" void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr, float *wi, float* vl, int *ldvl, float *vr, int *ldvr, float *work, int *lwork, int *info);
// 声明了外部函数 cgeev_，用于计算一般矩阵的特征值和（可选）特征向量（复单精度版本）
extern "C" void cgeev_(char *jobvl, char *jobvr, int *n,
             std::complex<float> *a, int *lda,
             std::complex<float> *w,
             std::complex<float> *vl, int *ldvl,
             std::complex<float> *vr, int *ldvr,
             std::complex<float> *work, int *lwork,
             float *rwork,
             int *info);
// 声明一个外部 C 函数 zgeev_，用于计算复数双精度矩阵的特征值和（可选）特征向量
extern "C" void zgeev_(char *jobvl, char *jobvr, int *n,
                       std::complex<double> *a, int *lda,
                       std::complex<double> *w,
                       std::complex<double> *vl, int *ldvl,
                       std::complex<double> *vr, int *ldvr,
                       std::complex<double> *work, int *lwork,
                       double *rwork,
                       int *info);

// 声明一个外部 C 函数 zgesdd_，用于计算复数双精度矩阵的奇异值分解
extern "C" void zgesdd_(char *jobz, int *m, int *n, std::complex<double> *a, int *lda,
                        double *s, std::complex<double> *u, int *ldu, std::complex<double> *vt, int *ldvt, std::complex<double> *work, int *lwork, double *rwork, int *iwork, int *info);

// 声明一个外部 C 函数 cgesdd_，用于计算复数单精度矩阵的奇异值分解
extern "C" void cgesdd_(char *jobz, int *m, int *n, std::complex<float> *a, int *lda,
                        float *s, std::complex<float> *u, int *ldu, std::complex<float> *vt, int *ldvt, std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);

// 声明一个外部 C 函数 dgesdd_，用于计算双精度矩阵的奇异值分解
extern "C" void dgesdd_(char *jobz, int *m, int *n, double *a, int *lda,
                        double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *iwork, int *info);

// 声明一个外部 C 函数 sgesdd_，用于计算单精度矩阵的奇异值分解
extern "C" void sgesdd_(char *jobz, int *m, int *n, float *a, int *lda,
                        float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info);

// 声明一个外部 C 函数 zgetrs_，用于解复数双精度线性方程组
extern "C" void zgetrs_(char *trans, int *n, int *nrhs, std::complex<double> *a, int *lda, int *ipiv, std::complex<double> *b, int *ldb, int *info);

// 声明一个外部 C 函数 cgetrs_，用于解复数单精度线性方程组
extern "C" void cgetrs_(char *trans, int *n, int *nrhs, std::complex<float> *a, int *lda, int *ipiv, std::complex<float> *b, int *ldb, int *info);

// 声明一个外部 C 函数 dgetrs_，用于解双精度线性方程组
extern "C" void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);

// 声明一个外部 C 函数 sgetrs_，用于解单精度线性方程组
extern "C" void sgetrs_(char *trans, int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);

// 声明一个外部 C 函数 zgels_，用于求解复数双精度最小二乘问题
extern "C" void zgels_(char *trans, int *m, int *n, int *nrhs,
                       std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
                       std::complex<double> *work, int *lwork, int *info);

// 声明一个外部 C 函数 cgels_，用于求解复数单精度最小二乘问题
extern "C" void cgels_(char *trans, int *m, int *n, int *nrhs,
                       std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
                       std::complex<float> *work, int *lwork, int *info);

// 声明一个外部 C 函数 dgels_，用于求解双精度最小二乘问题
extern "C" void dgels_(char *trans, int *m, int *n, int *nrhs,
                       double *a, int *lda, double *b, int *ldb,
                       double *work, int *lwork, int *info);

// 声明一个外部 C 函数 sgels_，用于求解单精度最小二乘问题
extern "C" void sgels_(char *trans, int *m, int *n, int *nrhs,
                       float *a, int *lda, float *b, int *ldb,
                       float *work, int *lwork, int *info);

// 声明一个外部 C 函数 zgelsd_，用于求解复数双精度最小二乘问题（加上奇异值分解）
extern "C" void zgelsd_(int *m, int *n, int *nrhs,
                        std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
                        double *s, double *rcond, int *rank,
                        std::complex<double> *work, int *lwork, double *rwork, int *iwork, int *info);

// 声明一个外部 C 函数 cgelsd_，用于求解复数单精度最小二乘问题（加上奇异值分解）
extern "C" void cgelsd_(int *m, int *n, int *nrhs,
                        std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
                        float *s, float *rcond, int *rank,
                        std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);
    std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);



// 声明一个函数指针，该函数接受以下参数：
// 1. std::complex<float> *work：指向 std::complex<float> 类型的指针，用于传递复数类型的工作空间
// 2. int *lwork：指向 int 类型的指针，用于传递工作空间大小的参数
// 3. float *rwork：指向 float 类型的指针，用于传递浮点数类型的工作空间
// 4. int *iwork：指向 int 类型的指针，用于传递整数类型的工作空间
// 5. int *info：指向 int 类型的指针，用于传递信息或错误码
// 函数返回类型和名称未提供，仅声明参数列表
// 外部C函数声明，使用 dgelsd_ 完成双精度浮点数的最小二乘问题求解
extern "C" void dgelsd_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *s, double *rcond, int *rank,
    double *work, int *lwork, int *iwork, int *info);

// 外部C函数声明，使用 sgelsd_ 完成单精度浮点数的最小二乘问题求解
extern "C" void sgelsd_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *s, float *rcond, int *rank,
    float *work, int *lwork, int *iwork, int *info);

// 外部C函数声明，使用 zgelsy_ 完成复双精度浮点数的最小二乘问题求解
extern "C" void zgelsy_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    int *jpvt, double *rcond, int *rank,
    std::complex<double> *work, int *lwork,
    double *rwork, int *info);

// 外部C函数声明，使用 cgelsy_ 完成复单精度浮点数的最小二乘问题求解
extern "C" void cgelsy_(int *m, int *n, int *nrhs,
    std::complex<float> * a, int *lda, std::complex<float> *b, int *ldb,
    int *jpvt, float *rcond, int *rank,
    std::complex<float> *work, int *lwork,
    float *rwork, int *info);

// 外部C函数声明，使用 dgelsy_ 完成双精度浮点数的最小二乘问题求解
extern "C" void dgelsy_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    int *jpvt, double *rcond, int *rank,
    double *work, int *lwork, int *info);

// 外部C函数声明，使用 sgelsy_ 完成单精度浮点数的最小二乘问题求解
extern "C" void sgelsy_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    int *jpvt, float *rcond, int *rank,
    float *work, int *lwork, int *info);

// 外部C函数声明，使用 zgelss_ 完成复双精度浮点数的最小二乘问题求解
extern "C" void zgelss_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    double *s, double *rcond, int *rank,
    std::complex<double> *work, int *lwork,
    double *rwork, int *info);

// 外部C函数声明，使用 cgelss_ 完成复单精度浮点数的最小二乘问题求解
extern "C" void cgelss_(int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    float *s, float *rcond, int *rank,
    std::complex<float> *work, int *lwork,
    float *rwork, int *info);

// 外部C函数声明，使用 dgelss_ 完成双精度浮点数的最小二乘问题求解
extern "C" void dgelss_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *s, double *rcond, int *rank,
    double *work, int *lwork, int *info);

// 外部C函数声明，使用 sgelss_ 完成单精度浮点数的最小二乘问题求解
extern "C" void sgelss_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *s, float *rcond, int *rank,
    float *work, int *lwork, int *info);

#endif

#if AT_BUILD_WITH_BLAS()
// 外部C函数声明，使用 ztrsm_ 完成复双精度浮点数的三角矩阵方程求解
extern "C" void ztrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<double> *alpha, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb);

// 外部C函数声明，使用 ctrsm_ 完成复单精度浮点数的三角矩阵方程求解
extern "C" void ctrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<float> *alpha, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb);

// 外部C函数声明，使用 dtrsm_ 完成双精度浮点数的三角矩阵方程求解
extern "C" void dtrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, double *alpha, double *a, int *lda, double *b, int *ldb);

// 外部C函数声明，使用 strsm_ 完成单精度浮点数的三角矩阵方程求解
extern "C" void strsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, float *alpha, float *a, int *lda, float *b, int *ldb);
#endif

namespace at::meta {

// 声明 Torch 元函数 linalg_ldl_factor_ex
TORCH_META_FUNC(linalg_ldl_factor_ex)
(const Tensor& self, bool hermitian, bool check_errors) {
  // 检查输入张量是否为方阵，并抛出适当的错误信息
  at::native::squareCheckInputs(self, "torch.linalg.ldl_factor_ex");
  // 检查输入张量的类型是否为浮点数或复数，并抛出适当的错误信息
  at::native::checkFloatingOrComplex(self, "torch.linalg.ldl_factor_ex");

  // 获取输入张量的形状信息
  auto shape = self.sizes();
  auto ndim = shape.size();

  // 优先选择列优先存储的步幅信息
  auto ld_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig=*/true);
  // 设置输出张量0的形状、步幅及数据类型选项，用于存储LD分解结果
  set_output_strided(0, shape, ld_strides, self.options(), {}); // LD

  // 设置输出张量1的形状为除最后一个维度外的所有维度，数据类型为整数
  set_output_contiguous(
      1, shape.slice(0, ndim - 1), self.options().dtype(ScalarType::Int)); // pivots

  // 设置输出张量2的形状为除最后两个维度外的所有维度，数据类型为整数
  set_output_contiguous(
      2, shape.slice(0, ndim - 2), self.options().dtype(ScalarType::Int)); // info
}

TORCH_META_FUNC(linalg_ldl_solve)
(const Tensor& LD,
 const Tensor& pivots,
 const Tensor& B,
 bool hermitian) {
  // 检查LD是否为方阵，并抛出适当的错误信息
  at::native::squareCheckInputs(LD, "torch.linalg.ldl_solve");
  // 检查LD的数据类型是否为浮点数或复数，并抛出适当的错误信息
  at::native::checkFloatingOrComplex(LD, "torch.linalg.ldl_solve");
  // 检查线性求解的输入是否符合要求，并抛出适当的错误信息
  at::native::linearSolveCheckInputs(B, LD, "torch.linalg.ldl_solve");
  // 检查B的维度是否至少为2，并抛出适当的错误信息
  TORCH_CHECK(
      B.dim() >= 2,
      "torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, but it has ",
      B.dim(),
      " dimensions instead");
  // 检查pivots的形状是否与LD的形状（除去最后一个维度）一致，并抛出适当的错误信息
  auto expected_pivots_shape = LD.sizes().slice(0, LD.dim() - 1);
  TORCH_CHECK(
      expected_pivots_shape.equals(pivots.sizes()),
      "torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, but got pivots with shape ",
      pivots.sizes(),
      " instead");
  // 检查pivots的数据类型是否为整数类型，并抛出适当的错误信息
  // LAPACK使用32位接口，而cuSOLVER使用64位接口处理整数
  TORCH_CHECK(
      at::isIntegralType(pivots.scalar_type(), /*includeBool=*/false),
      "torch.linalg.ldl_solve: Expected pivots to be integers. Got ",
      pivots.scalar_type());
  // 检查LD和B的数据类型是否一致，并抛出适当的错误信息
  TORCH_CHECK(
      LD.scalar_type() == B.scalar_type(),
      "torch.linalg.ldl_solve: ",
      "LD dtype",
      LD.scalar_type(),
      " does not match b dtype ",
      B.scalar_type());

  // 根据输入张量B和LD的形状，计算广播后的形状
  auto [B_broadcast_size, _] = at::native::_linalg_broadcast_batch_dims(B, LD);

  // 优先选择列优先存储的步幅信息
  auto result_strides = at::native::batched_matrix_contiguous_strides(B_broadcast_size, /*column_major=*/true);
  // 设置输出张量0的形状、步幅及数据类型选项，用于存储求解结果
  set_output_strided(0, B_broadcast_size, result_strides, B.options(), {});
}

TORCH_META_FUNC(triangular_solve)(const Tensor& self, const Tensor& A, bool upper, bool transpose, bool unitriangular) {
  // 检查self张量的维度是否至少为2，并抛出适当的错误信息
  TORCH_CHECK(self.dim() >= 2,
           "torch.triangular_solve: Expected b to have at least 2 dimensions, but it has ", self.dim(), " dimensions instead");
  // 检查A张量的维度是否至少为2，并抛出适当的错误信息
  TORCH_CHECK(A.dim() >= 2,
           "torch.triangular_solve: Expected A to have at least 2 dimensions, but it has ", A.dim(), " dimensions instead");

  // 检查线性求解的输入是否符合要求，并抛出适当的错误信息
  at::native::linearSolveCheckInputs(self, A, "triangular_solve");

  // 如果A张量的布局为Strided，则执行以下操作
  if (A.layout() == Layout::Strided) {
    // 根据self和A张量的形状，计算广播后的形状
    auto [self_broadcast_size, A_broadcast_size] = at::native::_linalg_broadcast_batch_dims(self, A);

    // 优先选择列优先存储的步幅信息，用于BLAS操作
    // 这里的输出张量0步幅的设置被省略了，原代码可能包含了该设置，但需要保留注释中提到的信息
    // 计算自身张量的连续步长，用于F顺序（列优先），并设置输出张量的步长和大小
    const auto solution_strides = at::native::batched_matrix_contiguous_strides(self_broadcast_size, /*f-contig=*/true);
    set_output_raw_strided(0, self_broadcast_size, solution_strides, self.options(), {});

    // 为了BLAS，生成A张量的列优先步长
    auto clone_A_strides = at::native::batched_matrix_contiguous_strides(A_broadcast_size, /*f_contig=*/true);
    set_output_raw_strided(1, A_broadcast_size, clone_A_strides, A.options(), {});
  } else if (A.layout() == Layout::SparseCsr || A.layout() == Layout::SparseBsr) {
    // 对于稀疏CSR或BSR布局，不进行广播操作，设置输出张量的大小为self的尺寸，步长为空
    set_output_raw_strided(0, self.sizes(), {}, self.options(), {}); // make row major strides for Sparse BLAS
    // 返回一个大小为0的张量
    set_output_raw_strided(1, {0}, {}, self.options(), {}); // return 0-sized tensor
  } else {
    // 如果张量布局既不是连续布局也不是稀疏布局，则抛出内部断言错误
    TORCH_INTERNAL_ASSERT(false, "triangular_solve: Got an unexpected layout.");
  }
}

// 定义 Torch 元函数 _linalg_solve_ex，处理线性代数求解问题
TORCH_META_FUNC(_linalg_solve_ex)(const Tensor& A,  // 输入张量 A
                                  const Tensor& B,  // 输入张量 B
                                  bool left,        // 是否是左侧矩阵求解
                                  bool check_errors) {  // 是否检查错误

  // 检查 A 的数据类型必须为浮点型或复数类型
  at::native::checkFloatingOrComplex(A, "linalg.solve");

  // 检查 A 和 B 的数据类型必须一致
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "linalg.solve: Expected A and B to have the same dtype, but found A of type ",
              A.scalar_type(), " and B of type ", B.scalar_type(), " instead");

  // NumPy 兼容性：支持两种类型的 'B' 张量：
  // - 1D 张量或批量的 1D 张量（向量情况）
  // - 2D 张量或批量的 2D 张量（矩阵情况）
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(A, B);
  auto B_ = vector_case ? B.unsqueeze(-1) : B;

  // 检查矩阵形状是否符合要求
  at::native::checkInputsSolver(A, B_, /*left=*/left, "linalg.solve");

  // 检查 B 是否可以广播到 A 的形状
  auto B_broad_shape = std::get<0>(at::native::_linalg_broadcast_batch_dims(B_, A));
  // 当 left=False 时，禁止将 B 广播为向量，因为在这种情况下 A.shape = (*, 1, 1)
  TORCH_CHECK(left || !vector_case, "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. In this case linalg.solve is equivalent to B / A.squeeze(-1)");
  auto result_shape = vector_case ? IntArrayRef(B_broad_shape.data(), B_broad_shape.size() - 1)
                                  : B_broad_shape;

  // 设置输出张量的步幅和形状，确保连续性
  auto result_strides = at::native::batched_matrix_contiguous_strides(result_shape, /*column_major=*/left);
  set_output_strided(0, result_shape, result_strides, B.options(), {});

  // 获取 A 的形状和维度数
  auto shape = A.sizes();
  auto ndim = shape.size();

  // 设置 LU 分解的输出张量步幅和形状，以确保连续性
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig=*/true);
  set_output_strided(1, shape, LU_strides, A.options(), {});

  // 设置输出张量的步幅和形状，确保连续性，用于存储主元信息
  set_output_contiguous(2, shape.slice(0, ndim - 1), A.options().dtype(kInt));

  // 设置输出张量的步幅和形状，确保连续性，用于存储信息
  set_output_contiguous(3, shape.slice(0, ndim - 2), A.options().dtype(kInt));
}

// 定义 Torch 元函数 linalg_inv_ex，处理矩阵求逆问题
TORCH_META_FUNC(linalg_inv_ex)(const Tensor& A,  // 输入张量 A
                                bool check_errors) {  // 是否检查错误
  // 检查输入张量 A 是否为方阵
  at::native::squareCheckInputs(A, "linalg.inv");

  // 检查 A 的数据类型必须为浮点型或复数类型，不允许低精度数据类型
  at::native::checkFloatingOrComplex(A, "linalg.inv", /*allow_low_precision_dtypes=*/false);

  // 获取输入张量 A 的形状
  auto shape = A.sizes();

  // 设置输出张量的步幅和形状，确保连续性
  auto result_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig=*/true);
  set_output_strided(0, shape, result_strides, A.options(), {});

  // 设置输出张量的形状，确保连续性，用于存储信息
  set_output_contiguous(
      1, shape.slice(0, shape.size() - 2), A.options().dtype(ScalarType::Int));  // info
}
// 定义名为 `linalg_lu_factor_ex` 的 Torch 元函数，接受一个张量 A 和两个布尔值 pivot 和 check_errors
TORCH_META_FUNC(linalg_lu_factor_ex)(const Tensor& A, bool pivot, bool check_errors) {
  // 检查张量 A 的维度是否至少为 2
  TORCH_CHECK(A.dim() >= 2, "torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");

  // 获取张量 A 的大小，并转为向量
  auto sizes = A.sizes().vec();
  // 获取 A 的倒数第二维和倒数第一维的大小
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];

  // 为 BLAS 设置列优先的步长
  auto LU_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
  // 设置第 0 号输出张量的大小、步长和数据类型
  set_output_strided(0, sizes, LU_strides, A.options(), {});

  // 将 sizes 改为存储主元的大小
  sizes.pop_back();
  sizes.back() = std::min(m, n);
  // 设置第 1 号输出张量的大小和数据类型
  set_output_contiguous(1, sizes, A.options().dtype(kInt), {});

  // 将 sizes 改为存储 info 的大小
  sizes.pop_back();
  // 设置第 2 号输出张量的大小和数据类型
  set_output_contiguous(2, sizes, A.options().dtype(kInt), {});
}

// 定义名为 `linalg_lu_solve` 的 Torch 元函数，接受三个张量 LU、pivots 和 B，以及两个布尔值 left 和 adjoint
TORCH_META_FUNC(linalg_lu_solve)(const Tensor& LU,
                                 const Tensor& pivots,
                                 const Tensor& B,
                                 bool left,
                                 bool adjoint) {
  // 检查 LU 和 B 的数据类型是否为浮点数或复数
  at::native::checkFloatingOrComplex(LU, "torch.linalg.lu_solve");
  // 检查 LU 和 B 的数据类型是否一致
  TORCH_CHECK(LU.scalar_type() == B.scalar_type(),
              "linalg.lu_solve: Expected LU and B to have the same dtype, but found LU of type ",
              LU.scalar_type(), " and B of type ", B.scalar_type(), " instead");
  // 检查 pivots 的数据类型是否为 torch.int32
  TORCH_CHECK(pivots.dtype() == at::kInt,
              "linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32");

  // 检查矩阵的形状
  at::native::squareCheckInputs(LU, "torch.linalg.lu_solve");
  at::native::checkInputsSolver(LU, B, left, "torch.linalg.lu_solve");
  
  // 检查每个批次的 LU 和 pivots 的维度是否匹配
  TORCH_CHECK(
      LU.size(-1) == pivots.size(-1),
      "linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix");

  // 检查每个批次的 LU 和 pivots 的前几维是否匹配
  TORCH_CHECK(
      LU.sizes().slice(0, LU.dim() - 1).equals(pivots.sizes()),
      "linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape ",
      pivots.sizes(), " instead");

  // 检查 B 是否可以广播到 A 的形状
  auto B_broadcast_size = std::get<0>(at::native::_linalg_broadcast_batch_dims(B, LU));
  // 根据 left 参数设置结果张量的步长
  auto result_strides = at::native::batched_matrix_contiguous_strides(B_broadcast_size, /*column_major=*/left);

  // 设置第 0 号输出张量的大小、步长和数据类型
  set_output_strided(0, B_broadcast_size, result_strides, B.options(), {});
}

// 定义名为 `linalg_cholesky_ex` 的 Torch 元函数，接受一个张量 A 和两个布尔值 upper 和 check_errors
TORCH_META_FUNC(linalg_cholesky_ex)(const Tensor& A,
                                    bool upper,
                                    bool check_errors) {
  // 检查输入矩阵 A 是否为方阵
  at::native::squareCheckInputs(A, "torch.linalg.cholesky");
  // 检查输入矩阵 A 的数据类型是否为浮点数或复数
  at::native::checkFloatingOrComplex(A, "torch.linalg.cholesky");

  // 获取输入张量 A 的大小
  auto A_shape = A.sizes();
  auto ndim = A_shape.size();

  // 获取 A 的批次矩阵的步长
  auto L_strides = at::native::batched_matrix_contiguous_strides(A_shape, /*f-contig*=*/true);
  // 设置第 0 号输出张量的大小、步长和数据类型
  set_output_strided(0, A_shape, L_strides, A.options(), {});

  // 设置第 1 号输出张量的大小和数据类型，为 info
  set_output_contiguous(1, A_shape.slice(0, ndim - 2), A.options().dtype(ScalarType::Int));
}
# 定义 Torch 元函数 linalg_qr，计算矩阵 A 的 QR 分解
TORCH_META_FUNC(linalg_qr)(const Tensor& A,
                           c10::string_view mode) {
  # 检查输入张量 A 是否为矩阵
  at::native::checkIsMatrix(A, "linalg.qr");
  # 检查输入张量 A 是否为浮点数或复数类型
  at::native::checkFloatingOrComplex(A, "linalg.qr");
  # 解析 QR 分解的模式，得到是否计算 Q 和是否使用简化模式的标志
  auto [compute_q, reduced_mode] = at::native::_parse_qr_mode(mode);

  # 获取张量 A 的形状信息
  auto A_shape = A.sizes().vec();
  const auto m = A_shape.cend()[-2];  // 获取矩阵行数 m
  const auto n = A_shape.cend()[-1];  // 获取矩阵列数 n
  const auto k = std::min(m, n);      // 计算矩阵的最小维度

  # 如果需要计算 Q 矩阵
  if (compute_q) {
    # 预备输出张量 Q 的形状，根据是否简化模式选择输出维度
    auto Q_shape = A_shape;
    Q_shape.end()[-1] = reduced_mode ? k : m;
    # 计算 Q 张量的步长信息，确保是批量矩阵并且是 F-contiguous 的
    auto Q_strides = at::native::batched_matrix_contiguous_strides(Q_shape, /*f-contig*=*/true);
    # 设置输出张量 0 的形状和步长信息为 Q 的信息，使用 A 的选项
    set_output_strided(0, Q_shape, Q_strides, A.options(), {});
  } else {
    # 如果不计算 Q，则设置输出张量 0 的形状为空，保持为原始的输出信息
    set_output_raw_strided(0, {0}, {}, A.options(), {});
  }

  // For readability
  # 为了提高可读性，创建临时变量保存 R 的形状信息
  auto R_shape = std::move(A_shape);
  R_shape.end()[-2] = (reduced_mode || !compute_q) ? k : m;
  # 计算 R 张量的步长信息，确保是批量矩阵并且是 F-contiguous 的
  auto R_strides = at::native::batched_matrix_contiguous_strides(R_shape, /*f-contig*=*/true);
  # 设置输出张量 1 的形状和步长信息为 R 的信息，使用 A 的选项
  set_output_strided(1, R_shape, R_strides, A.options(), {});
}


# 定义 Torch 元函数 _linalg_svd，计算矩阵 A 的奇异值分解（SVD）
TORCH_META_FUNC(_linalg_svd)(const Tensor& A,
                             bool full_matrices,
                             bool compute_uv,
                             std::optional<c10::string_view> driver) {
  # 检查输入张量 A 是否为矩阵
  at::native::checkIsMatrix(A, "linalg.svd");
  # 检查输入张量 A 是否为浮点数或复数类型
  at::native::checkFloatingOrComplex(A, "linalg.svd");

  # 获取张量 A 的形状信息
  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];  // 获取矩阵行数 m
  const auto n = sizes.cend()[-1];  // 获取矩阵列数 n
  const auto k = std::min(m, n);    // 计算矩阵的最小维度

  # 如果需要计算 U 和 V 矩阵
  if (compute_uv) {
    # 预备输出张量 U 的形状，根据是否完整模式选择输出维度
    sizes.back() = full_matrices ? m : k;
    # 计算 U 张量的步长信息，确保是批量矩阵并且是 F-contiguous 的
    auto U_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
    # 设置输出张量 0 的形状和步长信息为 U 的信息，使用 A 的选项
    set_output_strided(0, sizes, U_strides, A.options(), {});

    # 预备输出张量 Vh 的形状，根据是否完整模式选择输出维度
    sizes.end()[-2] = full_matrices ? n : k;
    sizes.end()[-1] = n;
    # 根据是否使用 cuSOLVER，计算 Vh 张量的步长信息
    const bool use_cusolver = at::native::svd_uses_cusolver(A);
    auto Vh_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/!use_cusolver);
    # 设置输出张量 2 的形状和步长信息为 Vh 的信息，使用 A 的选项
    set_output_strided(2, sizes, Vh_strides, A.options(), {});
  } else {
    # 如果不计算 U 和 V 矩阵，则设置输出张量 0 和 2 的形状为空，保持为原始的输出信息
    set_output_raw_strided(0, {0}, {}, A.options(), {});
    set_output_raw_strided(2, {0}, {}, A.options(), {});
  }

  # 预备输出张量 S 的形状，S 张量总是实数类型，即使 A 是复数类型
  sizes.pop_back();
  sizes.end()[-1] = k;
  # 设置输出张量 1 的形状和类型为 S 的信息，使用 A 的实数值类型
  set_output_contiguous(1, sizes, A.options().dtype(c10::toRealValueType(A.scalar_type())), {});
}

# 定义 Torch 元函数 lu_unpack，用于 LU 分解后的解包操作
TORCH_META_FUNC(lu_unpack)(const Tensor& LU, const Tensor& pivots, bool unpack_data, bool unpack_pivots) {
  # 检查 LU 张量的维度是否大于等于 2
  TORCH_CHECK(LU.dim() >= 2, "torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: ", LU.sizes(), " instead");
  # 如果需要解包 LU 分解的置换向量
  if (unpack_pivots) {
    // 检查 LU_pivots 张量的数据类型是否为 torch.int32，并抛出错误信息如果不符合预期
    TORCH_CHECK(pivots.scalar_type() == at::kInt,
        "torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\n"
        "Note: this function is intended to be used with the output produced by torch.linalg.lu_factor");
    
    // 获取 LU 张量的尺寸信息
    auto sizes = LU.sizes().vec();
    const auto m = sizes.cend()[-2];  // 获取倒数第二维的大小，即 m
    const auto n = sizes.cend()[-1];  // 获取最后一维的大小，即 n
    const auto k = std::min(m, n);    // 计算 m 和 n 的最小值，即 k
    
    // 设置 P 矩阵的尺寸为 (m, m)，如果不展开 pivot，则尺寸为零
    sizes.end()[-1] = m;
    
    // 根据 unpack_pivots 的值设置输出张量 0 的尺寸
    if (unpack_pivots) {
        set_output_raw_strided(0, sizes, {}, LU.options(), {});  // 展开 pivot，尺寸为 sizes
    } else {
        set_output_raw_strided(0, {0}, {}, LU.options(), {});   // 不展开 pivot，尺寸为 {0}
    }
    
    // 如果 unpack_data 为 true，则设置 L 和 U 矩阵的尺寸
    if (unpack_data) {
        // 设置 L 矩阵的尺寸为 (m, k)
        sizes.end()[-1] = k;
        set_output_raw_strided(1, sizes, {}, LU.options(), {});
    
        // 设置 U 矩阵的尺寸为 (k, n)
        sizes.end()[-2] = k;
        sizes.end()[-1] = n;
        set_output_raw_strided(2, sizes, {}, LU.options(), {});
    } else {
        // 如果不展开数据，则将输出张量 1 和 2 的尺寸设置为 {0}
        set_output_raw_strided(1, {0}, {}, LU.options(), {});
        set_output_raw_strided(2, {0}, {}, LU.options(), {});
    }
} // 结束命名空间 at::meta

namespace at::native {

#if AT_BUILD_WITH_LAPACK()
// 在 LAPACK 可用时定义批处理线性代数操作的每个批处理函数
// 求解 Cholesky 分解的结果
template<class scalar_t>
void lapackCholeskySolve(char uplo, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, int *info);

// 求解对称特征值问题
template<class scalar_t, class value_t=scalar_t>
void lapackSymeig(char jobz, char uplo, int n, scalar_t *a, int lda, value_t *w, scalar_t *work, int lwork, value_t *rwork, int *info);

// LAPACK 中特例化的 LU 分解实现
template<> void lapackLu<c10::complex<double>>(int m, int n, c10::complex<double> *a, int lda, int *ipiv, int *info) {
  zgetrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, info);
}

template<> void lapackLu<c10::complex<float>>(int m, int n, c10::complex<float> *a, int lda, int *ipiv, int *info) {
  cgetrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, info);
}

template<> void lapackLu<double>(int m, int n, double *a, int lda, int *ipiv, int *info) {
  dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackLu<float>(int m, int n, float *a, int lda, int *ipiv, int *info) {
  sgetrf_(&m, &n, a, &lda, ipiv, info);
}

// 在 LAPACK 中特例化 Cholesky 求解实现
template<> void lapackCholeskySolve<c10::complex<double>>(char uplo, int n, int nrhs, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb, int *info) {
  zpotrs_(&uplo, &n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}
template<> void lapackCholeskySolve<c10::complex<float>>(char uplo, int n, int nrhs, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb, int *info) {
  // 调用 LAPACK 函数 cpotrs_ 解决复数浮点类型的 Cholesky 线性方程组问题
  cpotrs_(&uplo, &n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

template<> void lapackCholeskySolve<double>(char uplo, int n, int nrhs, double *a, int lda, double *b, int ldb, int *info) {
  // 调用 LAPACK 函数 dpotrs_ 解决双精度浮点类型的 Cholesky 线性方程组问题
  dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholeskySolve<float>(char uplo, int n, int nrhs, float *a, int lda, float *b, int ldb, int *info) {
  // 调用 LAPACK 函数 spotrs_ 解决单精度浮点类型的 Cholesky 线性方程组问题
  spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholesky<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  // 调用 LAPACK 函数 zpotrf_ 进行复数双精度 Cholesky 分解
  zpotrf_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholesky<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  // 调用 LAPACK 函数 cpotrf_ 进行复数单精度 Cholesky 分解
  cpotrf_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholesky<double>(char uplo, int n, double *a, int lda, int *info) {
  // 调用 LAPACK 函数 dpotrf_ 进行双精度 Cholesky 分解
  dpotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholesky<float>(char uplo, int n, float *a, int lda, int *info) {
  // 调用 LAPACK 函数 spotrf_ 进行单精度 Cholesky 分解
  spotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  // 调用 LAPACK 函数 zpotri_ 进行复数双精度 Cholesky 逆矩阵计算
  zpotri_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  // 调用 LAPACK 函数 cpotri_ 进行复数单精度 Cholesky 逆矩阵计算
  cpotri_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<double>(char uplo, int n, double *a, int lda, int *info) {
  // 调用 LAPACK 函数 dpotri_ 进行双精度 Cholesky 逆矩阵计算
  dpotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<float>(char uplo, int n, float *a, int lda, int *info) {
  // 调用 LAPACK 函数 spotri_ 进行单精度 Cholesky 逆矩阵计算
  spotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackGeqrf<c10::complex<double>>(int m, int n, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *work, int lwork, int *info) {
  // 调用 LAPACK 函数 zgeqrf_ 进行复数双精度 GEQRF QR 分解
  zgeqrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackGeqrf<c10::complex<float>>(int m, int n, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *work, int lwork, int *info) {
  // 调用 LAPACK 函数 cgeqrf_ 进行复数单精度 GEQRF QR 分解
  cgeqrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackGeqrf<double>(int m, int n, double *a, int lda, double *tau, double *work, int lwork, int *info) {
  // 调用 LAPACK 函数 dgeqrf_ 进行双精度 GEQRF QR 分解
  dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}
// 以下是针对 LAPACK 中 QR 分解相关操作的模板特化函数，针对不同数据类型（float、double、复数）实现了不同的函数重载。

template<> void lapackGeqrf<float>(int m, int n, float *a, int lda, float *tau, float *work, int lwork, int *info) {
  // 调用 LAPACK 库中的单精度实数 GEQRF 函数进行 QR 分解
  sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrgqr<c10::complex<double>>(int m, int n, int k, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *work, int lwork, int *info) {
  // 调用 LAPACK 库中的双精度复数 ORGQR 函数进行 QR 分解的 Q 矩阵计算
  zungqr_(&m, &n, &k, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackOrgqr<c10::complex<float>>(int m, int n, int k, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *work, int lwork, int *info) {
  // 调用 LAPACK 库中的单精度复数 UNGQR 函数进行 QR 分解的 Q 矩阵计算
  cungqr_(&m, &n, &k, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackOrgqr<double>(int m, int n, int k, double *a, int lda, double *tau, double *work, int lwork, int *info) {
  // 调用 LAPACK 库中的双精度实数 ORGQR 函数进行 QR 分解的 Q 矩阵计算
  dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrgqr<float>(int m, int n, int k, float *a, int lda, float *tau, float *work, int lwork, int *info) {
  // 调用 LAPACK 库中的单精度实数 ORGQR 函数进行 QR 分解的 Q 矩阵计算
  sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrmqr<c10::complex<double>>(char side, char trans, int m, int n, int k, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *c, int ldc, c10::complex<double> *work, int lwork, int *info) {
  // 调用 LAPACK 库中的双精度复数 UNMQR 函数进行矩阵乘积计算
  zunmqr_(&side, &trans, &m, &n, &k, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(c), &ldc, reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackOrmqr<c10::complex<float>>(char side, char trans, int m, int n, int k, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *c, int ldc, c10::complex<float> *work, int lwork, int *info) {
  // 调用 LAPACK 库中的单精度复数 UNMQR 函数进行矩阵乘积计算
  cunmqr_(&side, &trans, &m, &n, &k, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(c), &ldc, reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackOrmqr<double>(char side, char trans, int m, int n, int k, double *a, int lda, double *tau, double *c, int ldc, double *work, int lwork, int *info) {
  // 调用 LAPACK 库中的双精度实数 ORMQR 函数进行矩阵乘积计算
  dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
}

template<> void lapackOrmqr<float>(char side, char trans, int m, int n, int k, float *a, int lda, float *tau, float *c, int ldc, float *work, int lwork, int *info) {
  // 调用 LAPACK 库中的单精度实数 ORMQR 函数进行矩阵乘积计算
  sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
}
template<> void lapackSyevd<c10::complex<double>, double>(char jobz, char uplo, int n, c10::complex<double> *a, int lda, double *w, c10::complex<double> *work, int lwork, double *rwork, int lrwork, int *iwork, int liwork, int *info) {
  // 调用 LAPACK 中的 zheevd 函数，计算复数双精度对称矩阵特征值和特征向量
  zheevd_(&jobz, &uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, w, reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template<> void lapackSyevd<c10::complex<float>, float>(char jobz, char uplo, int n, c10::complex<float> *a, int lda, float *w, c10::complex<float> *work, int lwork, float *rwork, int lrwork, int *iwork, int liwork, int *info) {
  // 调用 LAPACK 中的 cheevd 函数，计算复数单精度对称矩阵特征值和特征向量
  cheevd_(&jobz, &uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, w, reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template<> void lapackSyevd<double>(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork, double *rwork, int lrwork, int *iwork, int liwork, int *info) {
  // 调用 LAPACK 中的 dsyevd 函数，计算双精度对称矩阵特征值和特征向量
  (void)rwork;  // unused，未使用的参数 rwork
  (void)lrwork;  // unused，未使用的参数 lrwork
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template<> void lapackSyevd<float>(char jobz, char uplo, int n, float *a, int lda, float *w, float *work, int lwork, float *rwork, int lrwork, int
# 特化模板实现，用于求解复数双精度特征值问题的 LAPACK 函数封装

template<> void lapackEig<c10::complex<double>, double>(char jobvl, char jobvr, int n, c10::complex<double> *a, int lda, c10::complex<double> *w, c10::complex<double> *vl, int ldvl, c10::complex<double> *vr, int ldvr, c10::complex<double> *work, int lwork, double *rwork, int *info) {
  # 调用 LAPACK 中的 zgeev 函数，解特征值问题
  zgeev_(&jobvl, &jobvr, &n,
         reinterpret_cast<std::complex<double>*>(a), &lda,
         reinterpret_cast<std::complex<double>*>(w),
         reinterpret_cast<std::complex<double>*>(vl), &ldvl,
         reinterpret_cast<std::complex<double>*>(vr), &ldvr,
         reinterpret_cast<std::complex<double>*>(work), &lwork,
         rwork, info);
}

# 特化模板实现，用于求解复数单精度特征值问题的 LAPACK 函数封装
template<> void lapackEig<c10::complex<float>, float>(char jobvl, char jobvr, int n, c10::complex<float> *a, int lda, c10::complex<float> *w, c10::complex<float> *vl, int ldvl, c10::complex<float> *vr, int ldvr, c10::complex<float> *work, int lwork, float *rwork, int *info) {
  # 调用 LAPACK 中的 cgeev 函数，解特征值问题
  cgeev_(&jobvl, &jobvr, &n,
         reinterpret_cast<std::complex<float>*>(a), &lda,
         reinterpret_cast<std::complex<float>*>(w),
         reinterpret_cast<std::complex<float>*>(vl), &ldvl,
         reinterpret_cast<std::complex<float>*>(vr), &ldvr,
         reinterpret_cast<std::complex<float>*>(work), &lwork,
         rwork, info);
}

# 特化模板实现，用于求解复数双精度奇异值分解的 LAPACK 函数封装
template<> void lapackSvd<c10::complex<double>, double>(char jobz, int m, int n, c10::complex<double> *a, int lda,
                                  double *s, c10::complex<double> *u, int ldu, c10::complex<double> *vt, int ldvt, c10::complex<double> *work, int lwork, double *rwork, int *iwork, int *info) {
  # 调用 LAPACK 中的 zgesdd 函数，解奇异值分解问题
  zgesdd_(&jobz, &m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, s, reinterpret_cast<std::complex<double>*>(u), &ldu,
          reinterpret_cast<std::complex<double>*>(vt), &ldvt, reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, iwork, info);
}

# 特化模板实现，用于求解复数单精度奇异值分解的 LAPACK 函数封装
template<> void lapackSvd<c10::complex<float>, float>(char jobz, int m, int n, c10::complex<float> *a, int lda,
                                 float *s, c10::complex<float> *u, int ldu, c10::complex<float> *vt, int ldvt, c10::complex<float> *work, int lwork, float *rwork, int *iwork, int *info) {
  # 调用 LAPACK 中的 cgesdd 函数，解奇异值分解问题
  cgesdd_(&jobz, &m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, s, reinterpret_cast<std::complex<float>*>(u), &ldu,
          reinterpret_cast<std::complex<float>*>(vt), &ldvt, reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, iwork, info);
}

# 特化模板实现，用于求解双精度奇异值分解的 LAPACK 函数封装
template<> void lapackSvd<double>(char jobz, int m, int n, double *a, int lda,
                                  double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork, double *rwork, int *iwork, int *info) {
  # 调用 LAPACK 中的 dgesdd 函数，解奇异值分解问题
  dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
}
template<> void lapackSvd<float>(char jobz, int m, int n, float *a, int lda,
                                 float *s, float *u, int ldu, float *vt, int ldvt, float *work, int lwork, float *rwork, int *iwork, int *info) {
  // 调用 LAPACK 的单精度奇异值分解函数 sgesdd_
  sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
}

template <>
void lapackLdlSymmetric<double>(
    char uplo,
    int n,
    double* a,
    int lda,
    int* ipiv,
    double* work,
    int lwork,
    int* info) {
  // 调用 LAPACK 的双精度对称 LDL 分解函数 dsytrf_
  dsytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlSymmetric<float>(
    char uplo,
    int n,
    float* a,
    int lda,
    int* ipiv,
    float* work,
    int lwork,
    int* info) {
  // 调用 LAPACK 的单精度对称 LDL 分解函数 ssytrf_
  ssytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlSymmetric<c10::complex<double>>(
    char uplo,
    int n,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  // 调用 LAPACK 的双精度复数对称 LDL 分解函数 zsytrf_
  zsytrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlSymmetric<c10::complex<float>>(
    char uplo,
    int n,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  // 调用 LAPACK 的单精度复数对称 LDL 分解函数 csytrf_
  csytrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlHermitian<double>(
    char uplo,
    int n,
    double* a,
    int lda,
    int* ipiv,
    double* work,
    int lwork,
    int* info) {
  // 调用 LAPACK 的双精度 Hermitian LDL 分解函数 dsytrf_
  dsytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlHermitian<float>(
    char uplo,
    int n,
    float* a,
    int lda,
    int* ipiv,
    float* work,
    int lwork,
    int* info) {
  // 调用 LAPACK 的单精度 Hermitian LDL 分解函数 ssytrf_
  ssytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlHermitian<c10::complex<double>>(
    char uplo,
    int n,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  // 调用 LAPACK 的双精度复数 Hermitian LDL 分解函数 zhetrf_
  zhetrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlHermitian<c10::complex<float>>(
    char uplo,
    int n,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  // 调用 LAPACK 的单精度复数 Hermitian LDL 分解函数 chetrf_
  chetrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlSolveSymmetric<double>(
    char uplo,
    int n,
    int nrhs,
    double* a,
    int lda,
    int* ipiv,
    double* b,
    int ldb,
    int* info) {
  // 调用 LAPACK 的双精度对称 LDL 解法 dsytrs_
  dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}


注释：
    dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);



    调用名为 dsytrs_ 的 Fortran 函数来解线性方程组。以下是各参数的含义：
    - &uplo: 指向存储上/下三角信息的指针
    - &n: 矩阵的阶数
    - &nrhs: 右侧矩阵 b 的列数
    - a: 指向矩阵 A 的数组的指针
    - &lda: 主存储器中矩阵 A 的第一个维度（行数）
    - ipiv: 指向存储主元信息的指针
    - b: 指向右侧矩阵 b 的数组的指针
    - &ldb: 主存储器中右侧矩阵 b 的第一个维度（行数）
    - info: 返回执行情况的状态信息
template <>
void lapackLdlSolveSymmetric<float>(
    char uplo,
    int n,
    int nrhs,
    float* a,
    int lda,
    int* ipiv,
    float* b,
    int ldb,
    int* info) {
  // 调用 LAPACK 函数 ssytrs 解决对称矩阵线性方程组
  ssytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveSymmetric<c10::complex<double>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* b,
    int ldb,
    int* info) {
  // 调用 LAPACK 函数 zsytrs 解决对称复数矩阵线性方程组
  zsytrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveSymmetric<c10::complex<float>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* b,
    int ldb,
    int* info) {
  // 调用 LAPACK 函数 csytrs 解决对称复数矩阵线性方程组
  csytrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveHermitian<double>(
    char uplo,
    int n,
    int nrhs,
    double* a,
    int lda,
    int* ipiv,
    double* b,
    int ldb,
    int* info) {
  // 调用 LAPACK 函数 dsytrs 解决 Hermite 矩阵线性方程组
  dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveHermitian<float>(
    char uplo,
    int n,
    int nrhs,
    float* a,
    int lda,
    int* ipiv,
    float* b,
    int ldb,
    int* info) {
  // 调用 LAPACK 函数 ssytrs 解决 Hermite 矩阵线性方程组
  ssytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveHermitian<c10::complex<double>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* b,
    int ldb,
    int* info) {
  // 调用 LAPACK 函数 zhetrs 解决 Hermite 复数矩阵线性方程组
  zhetrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveHermitian<c10::complex<float>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* b,
    int ldb,
    int* info) {
  // 调用 LAPACK 函数 chetrs 解决 Hermite 复数矩阵线性方程组
  chetrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(b),
      &ldb,
      info);
}

template<> void lapackLuSolve<c10::complex<double>>(char trans, int n, int nrhs, c10::complex<double> *a, int lda, int *ipiv, c10::complex<double> *b, int ldb, int *info) {
  // 调用 LAPACK 函数 zgetrs 解决复数矩阵线性方程组
  zgetrs_(&trans, &n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}
// 对复数浮点类型的 LAPACK LU 解算器的特化模板，解决方程组 AX = B
template<> void lapackLuSolve<c10::complex<float>>(char trans, int n, int nrhs, c10::complex<float> *a, int lda, int *ipiv, c10::complex<float> *b, int ldb, int *info) {
  // 调用 LAPACK 的 cgetrs 函数进行解算
  cgetrs_(&trans, &n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

// 对双精度浮点类型的 LAPACK LU 解算器的特化模板，解决方程组 AX = B
template<> void lapackLuSolve<double>(char trans, int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int *info) {
  // 调用 LAPACK 的 dgetrs 函数进行解算
  dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

// 对单精度浮点类型的 LAPACK LU 解算器的特化模板，解决方程组 AX = B
template<> void lapackLuSolve<float>(char trans, int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int *info) {
  // 调用 LAPACK 的 sgetrs 函数进行解算
  sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

// 对复数双精度浮点类型的 LAPACK GELS 解算器的特化模板，解决最小二乘问题 AX = B
template<> void lapackGels<c10::complex<double>>(
    char trans, int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    c10::complex<double> *work, int lwork, int *info) {
  // 调用 LAPACK 的 zgels 函数进行解算
  zgels_(&trans, &m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

// 对复数单精度浮点类型的 LAPACK GELS 解算器的特化模板，解决最小二乘问题 AX = B
template<> void lapackGels<c10::complex<float>>(
    char trans, int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    c10::complex<float> *work, int lwork, int *info) {
  // 调用 LAPACK 的 cgels 函数进行解算
  cgels_(&trans, &m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

// 对双精度浮点类型的 LAPACK GELS 解算器的特化模板，解决最小二乘问题 AX = B
template<> void lapackGels<double>(
    char trans, int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *work, int lwork, int *info) {
  // 调用 LAPACK 的 dgels 函数进行解算
  dgels_(&trans, &m, &n, &nrhs,
      a, &lda, b, &ldb, work, &lwork, info);
}

// 对单精度浮点类型的 LAPACK GELS 解算器的特化模板，解决最小二乘问题 AX = B
template<> void lapackGels<float>(
    char trans, int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *work, int lwork, int *info) {
  // 调用 LAPACK 的 sgels 函数进行解算
  sgels_(&trans, &m, &n, &nrhs,
      a, &lda, b, &ldb, work, &lwork, info);
}

// 对复数双精度浮点类型的 LAPACK GELSD 解算器的特化模板，解决最小二乘问题 AX = B
template<> void lapackGelsd<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    double *s, double rcond, int *rank,
    c10::complex<double> *work, int lwork,
    double *rwork, int *iwork, int *info) {
  // 调用 LAPACK 的 zgelsd 函数进行解算
  zgelsd_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, iwork, info);
}

// 对复数单精度浮点类型的 LAPACK GELSD 解算器的特化模板，解决最小二乘问题 AX = B
template<> void lapackGelsd<c10::complex<float>, float>(
    int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    float *s, float rcond, int *rank,
    c10::complex<float> *work, int lwork,
    float *rwork, int *iwork, int *info) {
  // 调用 LAPACK 的 cgelsd 函数进行解算
  cgelsd_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<float>*>(work), &lwork,
      rwork, iwork, info);
}
    // 调用 cgelsd_ 函数进行最小二乘解问题的求解
    cgelsd_(&m, &n, &nrhs,
        // 将实部和虚部分开传递给 cgelsd_ 函数的复数参数
        reinterpret_cast<std::complex<float>*>(a), &lda,
        reinterpret_cast<std::complex<float>*>(b), &ldb,
        // 输出参数：奇异值数组
        s, &rcond, rank,
        // 用于存储工作空间的复数数组
        reinterpret_cast<std::complex<float>*>(work), &lwork,
        // 存储实数工作空间
        rwork, iwork, info);
template<> void lapackGelsd<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *s, double rcond, int *rank,
    double *work, int lwork,
    double *rwork, int *iwork, int *info) {
  // 调用 LAPACK 中的双精度矩阵最小二乘求解函数 dgelsd_
  dgelsd_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, iwork, info);
}

template<> void lapackGelsd<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *s, float rcond, int *rank,
    float *work, int lwork,
    float *rwork, int *iwork, int *info) {
  // 调用 LAPACK 中的单精度矩阵最小二乘求解函数 sgelsd_
  sgelsd_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, iwork, info);
}

template<> void lapackGelsy<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    int *jpvt, double rcond, int *rank,
    c10::complex<double> *work, int lwork, double *rwork, int *info) {
  // 调用 LAPACK 中的复数双精度矩阵最小二乘求解函数 zgelsy_
  zgelsy_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      jpvt, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelsy<c10::complex<float>, float>(
    int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    int *jpvt, float rcond, int *rank,
    c10::complex<float> *work, int lwork, float *rwork, int *info) {
  // 调用 LAPACK 中的复数单精度矩阵最小二乘求解函数 cgelsy_
  cgelsy_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      jpvt, &rcond, rank,
      reinterpret_cast<std::complex<float>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelsy<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    int *jpvt, double rcond, int *rank,
    double *work, int lwork, double *rwork, int *info) {
  // 调用 LAPACK 中的双精度矩阵最小二乘求解函数 dgelsy_
  dgelsy_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      jpvt, &rcond, rank,
      work, &lwork, info);
}

template<> void lapackGelsy<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    int *jpvt, float rcond, int *rank,
    float *work, int lwork, float *rwork, int *info) {
  // 调用 LAPACK 中的单精度矩阵最小二乘求解函数 sgelsy_
  sgelsy_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      jpvt, &rcond, rank,
      work, &lwork, info);
}

template<> void lapackGelss<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    double *s, double rcond, int *rank,
    c10::complex<double> *work, int lwork,
    double *rwork, int *info
    ) {
  // 调用 LAPACK 中的复数双精度矩阵最小二乘求解函数 zgelss_
  zgelss_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, info);
}
    float *s, float rcond, int *rank,
    c10::complex<float> *work, int lwork,
    float *rwork, int *info
    ) {


# 定义函数签名，声明参数及其类型
cgelss_(
    &m, &n, &nrhs,                            // 调用的线性方程求解函数的参数
    reinterpret_cast<std::complex<float>*>(a), &lda,  // 转换并传递矩阵 A 的复数形式及其引导维度
    reinterpret_cast<std::complex<float>*>(b), &ldb,  // 转换并传递矩阵 B 的复数形式及其引导维度
    s, &rcond, rank,                          // 传递奇异值、阈值及其排序后的数量
    reinterpret_cast<std::complex<float>*>(work), &lwork,  // 传递工作区数组的复数形式及其长度
    rwork, info                               // 传递实数工作区和返回的信息
);
// 定义了模板特化，用于双精度浮点数的最小二乘解法
template<> void lapackGelss<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *s, double rcond, int *rank,
    double *work, int lwork,
    double *rwork, int *info) {
  // 调用双精度双精度最小二乘解 LAPACK 函数 dgelss_
  dgelss_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, info);
}

// 定义了模板特化，用于单精度浮点数的最小二乘解法
template<> void lapackGelss<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *s, float rcond, int *rank,
    float *work, int lwork,
    float *rwork, int *info) {
  // 调用单精度最小二乘解 LAPACK 函数 sgelss_
  sgelss_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, info);
}
#endif

#if AT_BUILD_WITH_BLAS()
// 定义了模板特化，用于解决复数双精度矩阵的三角线性方程组
template<> void blasTriangularSolve<c10::complex<double>>(char side, char uplo, char trans, char diag, int n, int nrhs, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb) {
  // 设置复数值为 (1, 0)
  std::complex<double> one{1., 0.};
  // 调用 LAPACK 的复数双精度三角线性方程组求解函数 ztrsm_
  ztrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(b), &ldb);
}

// 定义了模板特化，用于解决复数单精度矩阵的三角线性方程组
template<> void blasTriangularSolve<c10::complex<float>>(char side, char uplo, char trans, char diag, int n, int nrhs, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb) {
  // 设置复数值为 (1.f, 0.f)
  std::complex<float> one{1.f, 0.f};
  // 调用 LAPACK 的复数单精度三角线性方程组求解函数 ctrsm_
  ctrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(b), &ldb);
}

// 定义了模板特化，用于解决双精度矩阵的三角线性方程组
template<> void blasTriangularSolve<double>(char side, char uplo, char trans, char diag, int n, int nrhs, double *a, int lda, double *b, int ldb) {
  // 设置标量值为 1.0
  auto one = 1.;
  // 调用 BLAS 的双精度三角线性方程组求解函数 dtrsm_
  dtrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, a, &lda, b, &ldb);
}

// 定义了模板特化，用于解决单精度矩阵的三角线性方程组
template<> void blasTriangularSolve<float>(char side, char uplo, char trans, char diag, int n, int nrhs, float *a, int lda, float *b, int ldb) {
  // 设置标量值为 1.0f
  auto one = 1.f;
  // 调用 BLAS 的单精度三角线性方程组求解函数 strsm_
  strsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, a, &lda, b, &ldb);
}
#endif

void _linalg_check_errors(
    const Tensor& infos,
    const c10::string_view api_name,
    bool is_matrix) {
  // 断言 infos 张量的数据类型是整型
  TORCH_INTERNAL_ASSERT(infos.scalar_type() == kInt);
  // 断言 infos 张量是连续的
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());
  // 如果 infos 是元数据，则直接返回
  if (infos.is_meta()) {
    return;
  }

  // 如果 infos 张量全为零，则提前返回，优化最常见的情况
  if (C10_LIKELY(!infos.any().item<bool>())) {
    return;
  }

  int32_t info;
  std::string batch_str;
  if (is_matrix) {
    info = infos.item<int>();
    // 对于矩阵，batch_str 不需要设置
  } else {
    // 查找第一个非零 info
    auto infos_cpu = infos.to(at::kCPU);
    auto ptr = infos_cpu.const_data_ptr<int32_t>();
    auto n = infos.numel();
    auto info_ptr = std::find_if(ptr, ptr + n, [](int32_t x) { return x != 0; });
    info = *info_ptr;
    batch_str = ": (Batch element " + std::to_string(std::distance(ptr, info_ptr)) + ")";
  }

  if (info < 0) {
    // 参考 LAPACK 3.10+ 改变了对于输入包含非有限值的 `info` 行为
    // 以前，它会返回 `info` > 0，但现在返回 `info` = -4
    // 如果 API 名称中包含 "svd"
    if (api_name.find("svd") != api_name.npos) {
      // 检查是否 info 等于 -4，如果是则抛出错误
      TORCH_CHECK_LINALG(info != -4, api_name, batch_str,
          ": The algorithm failed to converge because the input matrix contained non-finite values.");
    }
    // 如果前面的条件不满足，执行下面的逻辑
    else if (info > 0) {
      // 如果 API 名称中包含 "inv"
      if (api_name.find("inv") != api_name.npos) {
        // 抛出错误，说明矩阵不可逆
        TORCH_CHECK_LINALG(false, api_name, batch_str,
            ": The diagonal element ", info, " is zero, the inversion could not be completed because the input matrix is singular.");
      }
      // 如果 API 名称中包含 "solve"
      else if (api_name.find("solve") != api_name.npos) {
        // 抛出错误，说明矩阵不可解
        TORCH_CHECK_LINALG(false, api_name, batch_str,
            ": The solver failed because the input matrix is singular.");
      }
      // 如果 API 名称中包含 "cholesky"
      else if (api_name.find("cholesky") != api_name.npos) {
        // 抛出错误，说明矩阵不是正定的
        TORCH_CHECK_LINALG(false, api_name, batch_str,
            ": The factorization could not be completed because the input is not positive-definite (the leading minor of order ", info, " is not positive-definite).");
      }
      // 如果 API 名称中包含 "eig" 或者 "syevd"
      else if (api_name.find("eig") != api_name.npos || api_name.find("syevd") != api_name.npos) {
        // 抛出错误，说明特征值分解失败
        TORCH_CHECK_LINALG(false, api_name, batch_str,
            ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: ", info, ").");
      }
      // 如果 API 名称中包含 "lstsq"
      else if (api_name.find("lstsq") != api_name.npos) {
        // 抛出错误，说明最小二乘解无法计算
        TORCH_CHECK_LINALG(false, api_name, batch_str,
            ": The least squares solution could not be computed because the input matrix does not have full rank (error code: ", info, ").");
      }
      // 如果 API 名称中包含 "lu_factor"
      else if (api_name.find("lu_factor") != api_name.npos) {
        // 抛出错误，说明 LU 分解中的某个元素为零
        TORCH_CHECK(false, api_name, batch_str,
            ": U[", info, ",", info, "] is zero and using it on lu_solve would result in a division by zero. "
            "If you still want to perform the factorization, consider calling linalg.lu(A, pivot) or "
            "linalg.lu_factor_ex(A, pivot)");
      }
      // 如果以上条件都不满足，抛出未知错误
      else {
        TORCH_INTERNAL_ASSERT(false, api_name, ": Unknown error code: ", info, ".");
      }
    }
    // 如果 info 不大于 0，则抛出内部断言错误
    // 这种情况应该不会发生，因为 info 应该是非零的
    TORCH_INTERNAL_ASSERT(false);
}

// 如果输入需要前向或后向梯度，则需要走一个不同的（更慢）路径来确保梯度是可计算的。
// 这就是 `_may_require_fw_or_bw_grad` 函数的作用。
//
// 为什么这里需要检查 isTensorSubclassLike？
// 没有它，这个函数可能会导致复合兼容性问题，可能会在 functorch 中出现 bug，
// 其中一个不需要梯度的 Tensor 子类可能会包装一个需要梯度的 Tensor 子类。
static bool _may_require_fw_or_bw_grad(const Tensor& input) {
  return ((at::GradMode::is_enabled() && input.requires_grad())
          || input._fw_grad(/*level */ 0).defined()
          || isTensorSubclassLike(input));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.inv ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 实现 linalg_inv_ex_out 函数，用于计算矩阵的逆
TORCH_IMPL_FUNC(linalg_inv_ex_out)(const Tensor& A, bool check_errors, const Tensor& result, const Tensor& info) {
  // 将结果张量 result 填充为单位矩阵
  result.zero_();
  result.diagonal(0, -2, -1).fill_(1.);
  // 调用 linalg_solve_ex_out 函数，求解方程 A * X = result，其中 X 是未知数
  at::linalg_solve_ex_out(const_cast<Tensor&>(result), const_cast<Tensor&>(info), A, result, /*left*/true);
  // 如果需要检查错误，则调用 _linalg_check_errors 函数
  if (check_errors) {
    at::_linalg_check_errors(info, "linalg.inv_ex", A.dim() == 2);
  }
}

// 计算 A 的逆矩阵，并将结果存储在 result 中
Tensor& linalg_inv_out(const Tensor& A, Tensor& result) {
  auto info = at::empty({0}, A.options().dtype(kInt));
  at::linalg_inv_ex_out(result, info, A);
  // 检查并报告 linalg.inv 操作的错误
  at::_linalg_check_errors(info, "linalg.inv", A.dim() == 2);
  return result;
}

// 计算 A 的逆矩阵，并返回结果
Tensor linalg_inv(const Tensor& A) {
  auto [result, info] = at::linalg_inv_ex(A);
  // 检查并报告 linalg.inv 操作的错误
  at::_linalg_check_errors(info, "linalg.inv", A.dim() == 2);
  return result;
}

// 计算 A 的逆矩阵，并将结果存储在 result 中
Tensor& inverse_out(const Tensor& A, Tensor& result) {
  return at::linalg_inv_out(result, A);
}

// 计算 A 的逆矩阵，并返回结果
Tensor inverse(const Tensor& A) {
  return at::linalg_inv(A);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 应用 Cholesky 分解求解线性系统的函数模板
template<typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, Tensor& infos) {
#if !AT_BUILD_WITH_LAPACK()
  AT_ERROR("cholesky_solve: LAPACK library not found in compilation");
#else
  // 确定上三角还是下三角 Cholesky 分解
  char uplo = upper ? 'U' : 'L';

  // 获取张量的数据指针和相关信息
  auto A_data = A.const_data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);
  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto ldab = std::max<int64_t>(1, n);
  auto nrhs = b.size(-1);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;
  // 遍历批次中的每个线性系统
  for (const auto i : c10::irange(batch_size)) {
    const scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
    // 调用 LAPACK 提供的 Cholesky 分解求解函数
    lapackCholeskySolve<scalar_t>(uplo, n, nrhs, const_cast<scalar_t*>(A_working_ptr), ldab, b_working_ptr, ldab, &info);
    // 将求解过程中的信息存储到 infos 中
    infos_data[i] = info;
    // 如果 LAPACK 操作失败，则返回
    if (info != 0) {
      return;
    }
  }
#endif
}
// 使用 CPU 上的特定张量解 Cholesky 分解的辅助函数
Tensor _cholesky_solve_helper_cpu(const Tensor& self, const Tensor& A, bool upper) {
  // 复制 self 和 A 张量，按列主序进行批处理复制
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  // 创建用于记录批次错误信息的张量，全部初始化为零
  auto infos = at::zeros({batchCount(self)}, self.options().dtype(kInt));
  // 根据 self 的数据类型分发调用 apply_cholesky_solve 函数，解 Cholesky 分解
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_solve_cpu", [&]{
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, infos);
  });

  // 检查并报告 Cholesky 操作中的错误
  at::_linalg_check_errors(infos, "cholesky_solve_cpu", self.dim() == 2);
  // 返回处理后的 self_working_copy 结果张量
  return self_working_copy;
}

// 支持 self 和 A 张量的任意批次维度
Tensor cholesky_solve(const Tensor& self, const Tensor& A, bool upper) {
  // 检查 self 张量的维度是否至少为2
  TORCH_CHECK(self.dim() >= 2,
           "b should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  // 检查 A 张量的维度是否至少为2
  TORCH_CHECK(A.dim() >= 2,
           "u should have at least 2 dimensions, but has ", A.dim(), " dimensions instead");
  // 广播 self 和 A 张量的批次维度，以便进行 Cholesky 分解
  auto [self_broadcasted, A_broadcasted] = _linalg_broadcast_batch_dims(self, A, "cholesky_solve");
  // 调用 _cholesky_solve_helper 函数进行 Cholesky 分解，并返回结果
  return at::_cholesky_solve_helper(self_broadcasted, A_broadcasted, upper);
}

// 在输出张量 result 上执行 Cholesky 分解，结果存储在 result 中
Tensor& cholesky_solve_out(const Tensor& self, const Tensor& A, bool upper, Tensor& result) {
  // 检查 result 张量与 self 张量的设备是否相同
  checkSameDevice("cholesky_solve", result, self);
  // 检查 result 张量与 self 张量的数据类型是否兼容
  checkLinalgCompatibleDtype("cholesky_solve", result, self);
  // 调用 at::cholesky_solve 函数计算 Cholesky 分解，并将结果存储在 result_tmp 中
  Tensor result_tmp = at::cholesky_solve(self, A, upper);
  // 调整 result 的大小以匹配 result_tmp 的尺寸
  at::native::resize_output(result, result_tmp.sizes());
  // 将 result_tmp 的值复制到 result 中
  result.copy_(result_tmp);
  // 返回 result 张量
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 对给定的 self 张量执行 Cholesky 分解，返回上/下三角矩阵
Tensor cholesky(const Tensor &self, bool upper) {
   // 发出一次性警告，torch.cholesky 将在未来的 PyTorch 版本中被 torch.linalg.cholesky 取代
   TORCH_WARN_ONCE(
    "torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be ",
    "removed in a future PyTorch release.\n",
    "L = torch.cholesky(A)\n",
    "should be replaced with\n",
    "L = torch.linalg.cholesky(A)\n",
    "and\n"
    "U = torch.cholesky(A, upper=True)\n",
    "should be replaced with\n",
    "U = torch.linalg.cholesky(A).mH\n"
    "This transform will produce equivalent results for all valid (symmetric positive definite) inputs."
  );
  // 如果 self 张量元素数量为零，返回与 self 同样大小的空张量，使用遗留内存格式
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  // 检查 self 张量是否是方阵
  squareCheckInputs(self, "cholesky");

  // 复制 self 张量，并按列主序进行批处理复制
  auto raw_cholesky_output = cloneBatchedColumnMajor(self);
  // 计算信息张量的形状，去除最后两个维度
  auto info_shape = IntArrayRef(
      self.sizes().cbegin(), self.sizes().cend() - 2); // self.shape[:-2]
  // 创建一个与 self 形状相同的信息张量，用于记录错误信息
  auto info = at::empty({info_shape}, self.options().dtype(kInt));

  // 使用 cholesky_stub 函数执行 Cholesky 分解，填充 raw_cholesky_output
  cholesky_stub(self.device().type(), raw_cholesky_output, info, upper);

  // 检查并报告 Cholesky 操作中的错误
  at::_linalg_check_errors(info, "cholesky", self.dim() == 2);

  // 如果 upper 为 true，则返回 raw_cholesky_output 的上三角部分，否则返回下三角部分
  if (upper) {
    return raw_cholesky_output.triu_();
  } else {
    return raw_cholesky_output.tril_();
  }
}

// 在输出张量 result 上执行 Cholesky 分解，结果存储在 result 中
Tensor& cholesky_out(const Tensor &self, bool upper, Tensor &result) {
   // 发出一次性警告，torch.cholesky 将在未来的 PyTorch 版本中被 torch.linalg.cholesky 取代
   TORCH_WARN_ONCE(
    "torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be ",
    "removed in a future PyTorch release.\n",  
    # 提示信息：在未来的 PyTorch 版本中将被移除。

    "L = torch.cholesky(A)\n",  
    # 使用 torch.cholesky 对 A 进行 Cholesky 分解，返回下三角矩阵 L。

    "should be replaced with\n",  
    # 建议替换为以下代码。

    "L = torch.linalg.cholesky(A)\n",  
    # 使用 torch.linalg.cholesky 对 A 进行 Cholesky 分解，返回下三角矩阵 L。

    "and\n"  
    # 并且，

    "U = torch.cholesky(A, upper=True)\n",  
    # 使用 torch.cholesky 对 A 进行 Cholesky 分解，并返回上三角矩阵 U。

    "should be replaced with\n",  
    # 建议替换为以下代码。

    "U = torch.linalg.cholesky(A).mH\n"  
    # 使用 torch.linalg.cholesky 对 A 进行 Cholesky 分解，返回上三角矩阵 U 的共轭转置。

    "This transform will produce equivalent results for all valid (symmetric positive definite) inputs."
  );
  # 这个转换将对所有有效的（对称正定）输入产生等效的结果。
  checkSameDevice("cholesky", result, self);
  # 检查结果张量和当前张量 self 是否在相同的设备上。
  checkLinalgCompatibleDtype("cholesky", result, self);
  # 检查结果张量和当前张量 self 的数据类型是否兼容。
  Tensor result_tmp = at::cholesky(self, upper);
  # 对当前张量 self 进行 Cholesky 分解，结果存储在 result_tmp 中。
  at::native::resize_output(result, result_tmp.sizes());
  # 调整输出 result 的大小以匹配 result_tmp 的尺寸。
  result.copy_(result_tmp);
  # 将 result_tmp 的值复制到结果张量 result 中。
  return result;
  # 返回结果张量 result。
}

// 实现 Torch 的 linalg_cholesky_ex_out 函数，用于计算 Cholesky 分解的扩展版本，可以输出额外的信息
TORCH_IMPL_FUNC(linalg_cholesky_ex_out)(const Tensor& A,    // 输入张量 A
                                        bool upper,         // 是否计算上三角矩阵的 Cholesky 分解
                                        bool check_errors,  // 是否检查错误
                                        const Tensor& L,    // 输出的下三角矩阵 L
                                        const Tensor& info) {
  // 如果 L 张量没有元素，将 info 张量置零并直接返回
  if (L.numel() == 0) {
    info.zero_();
    return;
  }
  const auto cpu = A.device() == kCPU;  // 判断 A 是否在 CPU 上运行

  // 只有在 CPU 上时才能进行以下优化，因为在 MAGMA 上存在某些 bug 会导致失败
  if (cpu) {
    // 如果 upper 为 true，则计算 A 的上三角部分赋值给 L
    if (upper) {
      at::triu_out(const_cast<Tensor&>(L), A);
    } else {
      // 否则计算 A 的下三角部分赋值给 L
      at::tril_out(const_cast<Tensor&>(L), A);
    }
  } else {
    // 如果不在 CPU 上，则直接将 A 的数据拷贝给 L
    L.copy_(A);
  }

  // 调用 cholesky_stub 执行 Cholesky 分解，结果存储在 L 中
  cholesky_stub(L.device().type(), L, info, upper);

  // 如果不在 CPU 上，则根据 upper 的值调整 L 张量为上/下三角矩阵
  if (!cpu) {
    if (upper) {
      L.triu_();
    } else {
      L.tril_();
    }
  }

  // 如果需要检查错误，则调用 _linalg_check_errors 函数
  if (check_errors) {
    at::_linalg_check_errors(info, "linalg.cholesky_ex", A.dim() == 2);
  }
}

// 对外接口函数 linalg_cholesky，计算输入张量 A 的 Cholesky 分解，返回下三角矩阵 L
Tensor linalg_cholesky(const Tensor& A, bool upper) {
  // 调用 linalg_cholesky_ex 获取下三角矩阵 L 和相关信息
  auto [L, info] = at::linalg_cholesky_ex(A, upper, /*check_errors=*/false);
  // 检查是否有错误发生，如果有则抛出异常
  at::_linalg_check_errors(info, "linalg.cholesky", A.dim() == 2);
  // 返回下三角矩阵 L
  return L;
}

// 对外接口函数 linalg_cholesky_out，计算输入张量 A 的 Cholesky 分解，并将结果存储在输出张量 L 中
Tensor& linalg_cholesky_out(const Tensor& A, bool upper, Tensor& L) {
  // 创建一个与 A 张量类型相同的、元素个数为 0 的 info 张量
  auto info = at::empty({0}, A.options().dtype(kInt));
  // 调用 linalg_cholesky_ex_out 将 A 的 Cholesky 分解结果存储在输出张量 L 中
  at::linalg_cholesky_ex_out(L, info, A, upper, /*check_errors=*/false);
  // 检查是否有错误发生，如果有则抛出异常
  at::_linalg_check_errors(info, "linalg.cholesky", A.dim() == 2);
  // 返回输出张量 L
  return L;
}

// 定义 dispatch 函数 cholesky_inverse_stub，用于 Cholesky 逆操作的分发
DEFINE_DISPATCH(cholesky_inverse_stub);

// 内部函数 cholesky_inverse_out_info，用于在输出张量 result 和 info 中执行 Cholesky 逆操作
static Tensor& cholesky_inverse_out_info(Tensor& result,     // 输出的逆矩阵结果
                                         Tensor& infos,      // 信息张量
                                         const Tensor& input,  // 输入的矩阵
                                         bool upper) {        // 是否计算上三角矩阵的逆
  // 断言输入张量 input 至少是二维的，并且是方阵
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-1) == input.size(-2));

  // 断言输出张量 result 和输入张量 input 的类型和设备类型相同
  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  // infos 张量必须是整型类型，且在 CPU 上，并且元素个数为 batchCount(input) 和 1 之间的较大值
  TORCH_INTERNAL_ASSERT(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT(infos.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT(infos.numel() == std::max<int64_t>(1, batchCount(input)));

  // 如果 result 没有元素，可以修改它
  if (result.numel() == 0) {
    // 调整 result 的大小以匹配 input 的转置，并保持连续内存布局
    at::native::resize_as_(result, input.mT(), MemoryFormat::Contiguous);
    // 将 result 进行转置
    result.transpose_(-2, -1);
  }

  // result 张量必须是批次列主序（Fortran 连续）的
  TORCH_INTERNAL_ASSERT(result.mT().is_contiguous());
  // 断言 result 的大小和 input 相同
  TORCH_INTERNAL_ASSERT(result.sizes().equals(input.sizes()));

  // 执行 Cholesky 逆操作，结果存储在 result 中，需复制 input 数据到 result 中
  result.copy_(input);

  // infos 张量必须是连续的，并且全部填充为 0
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());
  infos.fill_(0);

  // 调用 cholesky_inverse_stub 执行 Cholesky 逆操作，结果存储在 result 中
  result = cholesky_inverse_stub(result.device().type(), result, infos, upper);
  // 返回输出张量 result
  return result;
}
// 计算 Cholesky 分解的逆矩阵，并将结果存储在给定的 result 引用中
Tensor& cholesky_inverse_out(const Tensor &input, bool upper, Tensor &result) {
  // 检查输入是否为方阵，并进行相应的检查
  squareCheckInputs(input, "cholesky_inverse");
  // 检查结果张量与输入张量是否在相同的设备上
  checkSameDevice("cholesky_inverse", result, input);
  // 检查结果张量与输入张量的数据类型是否兼容
  checkLinalgCompatibleDtype("cholesky_inverse", result, input);

  // MAGMA 要求 'infos' 必须在 CPU 内存中，因此我们现在只在 CPU 上创建 'infos'
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, input.options().dtype(kInt).device(kCPU));

  // 检查结果张量是否与输入张量具有相同的数据类型
  bool result_input_same_type = (result.scalar_type() == input.scalar_type());
  // 检查结果张量是否具有与输入张量相同的形状
  bool result_equal_expected_shape = result.sizes().equals(input.sizes());
  // 检查结果张量是否是批处理的列优先格式
  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.mT().is_contiguous();
  }

  // 如果结果张量非空且不是批处理的列优先格式，则需要复制数据
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  // 或者结果张量的数据类型与输入张量的不同
  copy_needed |= !result_input_same_type;
  // 或者结果张量的形状与预期不同
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape);
  // 如果需要复制数据，则需要分配一个临时张量
  if (copy_needed) {
    // 创建一个空的临时张量
    Tensor result_tmp = at::empty({0}, input.options());
    // 调用 cholesky_inverse_out_info 函数计算 Cholesky 逆矩阵，并将结果存储在临时张量中
    result_tmp = cholesky_inverse_out_info(result_tmp, infos, input, upper);
    // 调整输出张量 result 的大小以匹配临时张量的大小，并将数据复制到 result 中
    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
  } else {
    // 直接使用结果张量的内存空间来存储计算结果
    result = cholesky_inverse_out_info(result, infos, input, upper);
  }

  // 检查 LAPACK/MAGMA 的错误码
  at::_linalg_check_errors(infos, "cholesky_inverse", result.dim() == 2);
  // 返回计算结果的引用
  return result;
}

// 计算输入张量的 Cholesky 逆矩阵，并返回结果张量
Tensor cholesky_inverse(const Tensor &input, bool upper) {
  // 创建一个空的结果张量
  Tensor result = at::empty({0}, input.options());
  // 调用 cholesky_inverse_out 函数计算 Cholesky 逆矩阵，并将结果存储在 result 中
  result = at::cholesky_inverse_out(result, input, upper);
  // 返回计算结果的张量
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 在反向传播中使用的辅助函数，返回 LU 分解以便后向传播使用
// 定义了一个名为 TORCH_IMPL_FUNC 的宏，用于实现 torch 中的 linalg_solve_ex_out 函数
TORCH_IMPL_FUNC(_linalg_solve_ex_out)(const Tensor& A,    // 输入张量 A
                                      const Tensor& B,    // 输入张量 B
                                      bool left,          // 指示是否解左乘的布尔值
                                      bool check_errors,  // 指示是否检查错误的布尔值
                                      const Tensor& result,  // 输出结果张量
                                      const Tensor& LU,       // LU 分解结果张量
                                      const Tensor& pivots,   // LU 分解的枢轴张量
                                      const Tensor& info) {   // 包含操作信息的张量
  // 可能的优化：如果 A 是连续的，则计算 A 的转置的 LU 分解
  // 然后使用 adjoint=True 解 A^T X = B
  // 这样可以避免将 A 复制到 F-连续矩阵中的操作，节省了一次复制
  // 这种优化使得 functorch 的批处理规则变得复杂。参见 NOTE [ solve_ex Batch Rule Contiguity ]
  const bool use_A_T = A.is_contiguous() && !A.is_complex();
  
  // 调用 ATen 库的 linalg_lu_factor_ex_out 函数，进行 LU 分解的扩展操作
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU),    // 输出的 LU 分解结果张量
                              const_cast<Tensor&>(pivots),  // 输出的 LU 分解的枢轴张量
                              const_cast<Tensor&>(info),    // 输出的包含操作信息的张量
                              use_A_T ? A.mT() : A);        // 如果 use_A_T 为 true，则使用 A 的转置进行 LU 分解，否则使用 A 本身

  // 如果需要检查错误，则调用 ATen 库的 _linalg_check_errors 函数检查错误
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.solve_ex", A.dim() == 2);
  }

  // [numpy-compat] 处理右侧为向量的情况
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(LU, B);

  // 根据 vector_case 的值决定是否对结果 result 和输入 B 进行维度调整
  auto result_ = vector_case ? result.unsqueeze(-1) : result;
  auto B_ = vector_case ? B.unsqueeze(-1) : B;

  // 调用 ATen 库的 linalg_lu_solve_out 函数，求解 LU X = B_，并将结果存储在 result_ 中
  at::linalg_lu_solve_out(result_, LU, pivots, B_, left, /*adjoint*/use_A_T);
}

// 定义了一个返回类型为 tuple<Tensor&, Tensor&> 的函数 linalg_solve_ex_out
std::tuple<Tensor&, Tensor&> linalg_solve_ex_out(const Tensor& A,   // 输入张量 A
                                                 const Tensor& B,   // 输入张量 B
                                                 bool left,         // 指示是否解左乘的布尔值
                                                 bool check_errors, // 指示是否检查错误的布尔值
                                                 Tensor& result,    // 输出结果张量
                                                 Tensor& info) {    // 包含操作信息的张量
  // 创建空的 LU 和 pivots 张量，用于存储结果
  auto LU = B.new_empty({0});
  auto pivots = B.new_empty({0}, kInt);

  // 调用 _linalg_solve_ex_out 函数，进行求解操作，并返回结果 result 和 info 的引用
  at::_linalg_solve_ex_out(result, LU, pivots, info, A, B, left, check_errors);

  // 返回结果的 tuple，包含 result 和 info 的引用
  return std::tie(result, info);
}

// 实现 linalg_solve_ex 函数，作为 _linalg_solve_ex 的复合函数
std::tuple<Tensor, Tensor> linalg_solve_ex(const Tensor& A,   // 输入张量 A
                                           const Tensor& B,   // 输入张量 B
                                           bool left,         // 指示是否解左乘的布尔值
                                           bool check_errors) {  // 指示是否检查错误的布尔值
  // 调用 _linalg_solve_ex 函数，获取结果 result, LU, pivots, info
  auto [result, LU, pivots, info] = at::_linalg_solve_ex(A, B, left, check_errors);

  // 返回结果的 tuple，包含移动后的 result 和 info
  return std::make_tuple(std::move(result), std::move(info));
}

// 实现 linalg_solve_out 函数，作为 linalg_solve_ex_out 的一个简化接口
Tensor& linalg_solve_out(const Tensor& A,   // 输入张量 A
                         const Tensor& B,   // 输入张量 B
                         bool left,         // 指示是否解左乘的布尔值
                         Tensor& result) {  // 输出结果张量
  // 创建空的 info 张量，用于存储操作信息
  auto info = B.new_empty({0}, kInt);

  // 调用 linalg_solve_ex_out 函数，求解 A X = B，并将结果存储在 result 中
  at::linalg_solve_ex_out(result, info, A, B, left);

  // 检查操作是否出错，并抛出异常
  at::_linalg_check_errors(info, "torch.linalg.solve", A.dim() == 2);

  // 返回结果张量 result 的引用
  return result;
}
Tensor linalg_solve(const Tensor& A,
                    const Tensor& B,
                    bool left) {
  // 调用 torch 的 linalg_solve_ex 函数求解线性方程组 Ax = B
  auto [result, info] = at::linalg_solve_ex(A, B, left);
  // 检查求解过程中是否有错误，并报告错误信息
  at::_linalg_check_errors(info, "torch.linalg.solve", A.dim() == 2);
  // 返回求解结果
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_factor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(lu_factor_stub);

TORCH_IMPL_FUNC(linalg_lu_factor_ex_out)(const Tensor& A,
                                         bool pivot,
                                         bool check_errors,
                                         const Tensor& LU,
                                         const Tensor& pivots,
                                         const Tensor& info) {
  // 如果输入矩阵 A 的元素个数为 0，将 info 张量清零并返回
  if (A.numel() == 0) {
    info.zero_();
    return;
  }
  // 如果 LU 不是指向 A 的相同张量，则将 A 的内容复制到 LU
  if (!LU.is_same(A)) {
    LU.copy_(A);
  }

  // 调用 lu_factor_stub 函数，执行 LU 分解，并更新 LU, pivots 和 info
  lu_factor_stub(A.device().type(), LU, pivots, info, pivot);

  // 如果需要检查错误，则调用 _linalg_check_errors 函数，报告 LU 分解过程中的错误信息
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.lu_factor_ex", A.dim() == 2);
  }
}

std::tuple<Tensor&, Tensor&> linalg_lu_factor_out(const Tensor& A, bool pivot, Tensor& LU, Tensor& pivots) {
  // 创建一个空的 info 张量，用于存储错误信息
  auto info = at::empty({0}, A.options().dtype(kInt));
  // 调用 linalg_lu_factor_ex_out 函数进行 LU 分解，并不进行错误检查
  at::linalg_lu_factor_ex_out(LU, pivots, info, A, pivot, /*check_errors=*/false);
  // 如果需要检查错误，则调用 _linalg_check_errors 函数，报告 LU 分解过程中的错误信息
  at::_linalg_check_errors(info, "torch.linalg.lu_factor", A.dim() == 2);
  // 返回 LU 和 pivots 张量的引用
  return std::tie(LU, pivots);
}

std::tuple<Tensor, Tensor> linalg_lu_factor(const Tensor& A, bool pivot) {
  // 调用 linalg_lu_factor_ex 函数进行 LU 分解，并不进行错误检查
  auto [LU, pivots, info] = at::linalg_lu_factor_ex(A, pivot, /*check_errors=*/false);
  // 如果需要检查错误，则调用 _linalg_check_errors 函数，报告 LU 分解过程中的错误信息
  at::_linalg_check_errors(info, "torch.linalg.lu_factor", A.dim() == 2);
  // 返回 LU 和 pivots 的元组
  return std::make_tuple(std::move(LU), std::move(pivots));
}

// TODO Deprecate this function in favour of linalg_lu_factor_ex
std::tuple<Tensor, Tensor, Tensor> _lu_with_info(const Tensor& self, bool compute_pivots, bool) {
  // 发出警告提示，告知用户该函数已被 torch.linalg.lu_factor / torch.linalg.lu_factor_ex 函数取代
  TORCH_WARN_ONCE(
    "torch.lu is deprecated in favor of torch.linalg.lu_factor / torch.linalg.lu_factor_ex and will be ",
    "removed in a future PyTorch release.\n",
    "LU, pivots = torch.lu(A, compute_pivots)\n",
    "should be replaced with\n",
    "LU, pivots = torch.linalg.lu_factor(A, compute_pivots)\n",
    "and\n",
    "LU, pivots, info = torch.lu(A, compute_pivots, get_infos=True)\n",
    "should be replaced with\n",
    "LU, pivots, info = torch.linalg.lu_factor_ex(A, compute_pivots)"
  );
  // 调用 linalg_lu_factor_ex 函数进行 LU 分解，并不进行错误检查
  return at::linalg_lu_factor_ex(self, compute_pivots, false);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(unpack_pivots_stub);
// 实现 LU 分解的核心函数，将分解结果存储在给定的输出张量中
TORCH_IMPL_FUNC(linalg_lu_out)(const Tensor& A,
                               bool pivot,
                               const Tensor& P,
                               const Tensor& L,
                               const Tensor& U) {
  // 获取矩阵 A 的行数 m 和列数 n
  const auto m = A.sizes().end()[-2];
  const auto n = A.sizes().end()[-1];

  // A.shape[-2:] == (m, n)
  // P.shape[-2:] == (m, m)
  // L.shape[-2:] == (m, k)
  // U.shape[-2:] == (k, n)
  // 其中 k = min(m, n)，这里记录了各张量的维度信息

  // 使用 L 张量，因为其大小是正确的
  const bool use_L = m > n;
  // 创建空的整数张量 pivots 和 info
  auto pivots = at::empty({0}, A.options().dtype(kInt));
  auto info = at::empty({0}, A.options().dtype(kInt));
  // 调用 ATen 的 linalg_lu_factor_ex_out 函数进行 LU 分解
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(use_L ? L : U),
                              const_cast<Tensor&>(pivots),
                              const_cast<Tensor&>(info),
                              A,
                              pivot,
                              /*check_errors=*/false);
  // 调用 ATen 的 lu_unpack_out 函数，解包 LU 分解结果到 P、L、U 张量
  at::lu_unpack_out(const_cast<Tensor&>(P),
                    const_cast<Tensor&>(L),
                    const_cast<Tensor&>(U),
                    use_L ? L : U,
                    pivots,
                    /*unpack_lu=*/true,
                    /*unpack_pivots=*/pivot);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_unpack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 解包 LU 分解的结果到 P、L、U 张量
TORCH_IMPL_FUNC(lu_unpack_out)(const Tensor& LU,
                               const Tensor& pivots,
                               bool unpack_lu,
                               bool unpack_pivots,
                               const Tensor& P,
                               const Tensor& L,
                               const Tensor& U) {
  // 获取 LU 张量的行数 m 和列数 n
  const auto m = LU.sizes().end()[-2];
  const auto n = LU.sizes().end()[-1];

  // A.shape[-2:] == (m, n)
  // P.shape[-2:] == (m, m)
  // L.shape[-2:] == (m, k)
  // U.shape[-2:] == (k, n)
  // 其中 k = min(m, n)，这里记录了各张量的维度信息

  if (unpack_lu) {
    if (m > n || LU.is_same(L)) {
      // 注意 triu 和 tril 的顺序很重要，因为 LU.is_same(L) 可能为真
      // 如果 m > n 或 LU 与 L 相同，则提取 LU 的上三角部分到 U，下三角部分到 L
      at::triu_out(const_cast<Tensor&>(U), m == n ? LU : LU.narrow(-2, 0, n), 0);
      at::tril_out(const_cast<Tensor&>(L), LU, -1);
      // 将 L 的对角线元素设为 1
      L.diagonal(0, -2, -1).fill_(1.);
    } else {
      // 如果 m <= n 且 LU 与 U 相同，则提取 LU 的下三角部分到 L，上三角部分到 U
      at::tril_out(const_cast<Tensor&>(L), m == n ? LU : LU.narrow(-1, 0, m), -1);
      // 将 L 的对角线元素设为 1
      L.diagonal(0, -2, -1).fill_(1.);
      at::triu_out(const_cast<Tensor&>(U), LU, 0);
    }
  }
  if (unpack_pivots) {
    // lu_factor_ex 返回基于 1 的 int32 索引，对应于 pivots 中的排列
    // 我们需要将这些索引转换为 {0, ..., m-1} 的正确置换
    const auto perm_sizes = IntArrayRef(P.sizes().data(), P.dim() - 1);

    // 使用 identity permutation 填充 perm 张量（可能是批处理的）
    const auto perm = at::arange(m, pivots.options().memory_format(at::MemoryFormat::Contiguous).dtype(kLong))
                        .expand(perm_sizes)
                        .contiguous();
    // 创建一个 Tensor 迭代器配置对象，用于操作 perm 和 pivots 张量
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)  // 设置不检查内存重叠
      .check_all_same_dtype(false)   // 设置不检查所有张量是否具有相同的数据类型
      .resize_outputs(false)         // 设置不调整输出张量的大小
      .declare_static_shape(pivots.sizes(), /*squash_dim=*/pivots.dim() - 1)  // 声明静态形状，使用 pivots 张量的尺寸，但压缩掉一个维度
      .add_output(perm)              // 添加输出张量 perm
      .add_owned_const_input(pivots.contiguous())  // 添加拥有的常量输入张量 pivots 的连续版本
      .build();                      // 构建 Tensor 迭代器对象
    
    // 调用 unpack_pivots_stub 函数来解压缩 pivots 张量，生成置换 perm，处理最小的维度 m 和 n
    unpack_pivots_stub(pivots.device().type(), iter, std::min(m, n), m);
    
    // 将置换 perm 转换为置换矩阵 P
    P.zero_();  // 将 P 张量置零
    P.scatter_(-2, perm.unsqueeze(-2), 1.);  // 在 P 张量的倒数第二维度上进行散射操作，使用 perm 张量的展开版本，设置散射值为 1
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 定义 linalg_lu_solve 的分发函数
DEFINE_DISPATCH(lu_solve_stub);

// 实现 torch 中的 linalg_lu_solve_out 函数，用于求解线性方程组 LU X = B
TORCH_IMPL_FUNC(linalg_lu_solve_out)(const Tensor& LU,
                                     const Tensor& pivots,
                                     const Tensor& B,
                                     bool left,
                                     bool adjoint,
                                     const Tensor& result) {
  // 如果结果张量为空，直接返回
  if (result.numel() == 0) {
    return;
  }

  // 解 A^H X = B^H。然后返回 X^H
  if (!left) {
    // 如果不是左侧求解，则进行共轭转置，并修改 adjoint 标志
    adjoint = !adjoint;
    result.transpose_(-2, -1);
  }

  // 将 B（或其共轭转置）复制到结果中
  if (!result.is_same(B)) {
    result.copy_(left ? B : B.mH());
  }

  // 使 LU 和 pivots 张量保持 F-contiguous
  auto pivots_ = pivots.expect_contiguous();
  auto LU_ = at::native::borrow_else_clone(
      LU.mT().is_contiguous(), LU, LU, /*row_major=*/false);

  // 确定转置类型
  const auto trans = !adjoint ? TransposeType::NoTranspose :
                     LU.is_complex() ? TransposeType::ConjTranspose
                                     : TransposeType::Transpose;

  // 调用 lu_solve_stub 进行 LU 分解求解
  lu_solve_stub(LU_->device().type(), *LU_, *pivots_, result, trans);

  // 如果不是左侧求解，则在原地进行共轭转置
  if (!left) {
    result.transpose_(-2, -1);
    // 如果结果是复数类型，进行共轭设置
    if (result.is_complex()) {
      result._set_conj(!result.is_conj());
    }
  }
}

// 实现 torch 中的 lu_solve 函数，用于求解线性方程组 LU X = B
Tensor lu_solve(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots) {
  // 发出警告信息，lu_solve 已弃用，请使用 torch.linalg.lu_solve
  TORCH_WARN_ONCE(
    "torch.lu_solve is deprecated in favor of torch.linalg.lu_solve",
    "and will be removed in a future PyTorch release.\n",
    "Note that torch.linalg.lu_solve has its arguments reversed.\n",
    "X = torch.lu_solve(B, LU, pivots)\n",
    "should be replaced with\n",
    "X = torch.linalg.lu_solve(LU, pivots, B)"
  );
  // 调用 at::linalg_lu_solve 进行求解
  return at::linalg_lu_solve(LU_data, LU_pivots, self);
}

// 实现 torch 中的 lu_solve_out 函数，用于求解线性方程组 LU X = B，并将结果写入 result 张量
Tensor& lu_solve_out(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots, Tensor& result) {
  // 发出警告信息，lu_solve 已弃用，请使用 torch.linalg.lu_solve
  TORCH_WARN_ONCE(
    "torch.lu_solve is deprecated in favor of torch.linalg.lu_solve",
    "and will be removed in a future PyTorch release.\n",
    "Note that torch.linalg.lu_solve has its arguments reversed.\n",
    "X = torch.lu_solve(B, LU, pivots)\n",
    "should be replaced with\n",
    "X = torch.linalg.lu_solve(LU, pivots, B)"
  );
  // 调用 at::linalg_lu_solve_out 进行求解，并将结果写入 result 张量
  return at::linalg_lu_solve_out(result, LU_data, LU_pivots, self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 定义 triangular_solve 的分发函数
DEFINE_DISPATCH(triangular_solve_stub);

/*
解决矩阵方程 'input' @ 'result' = 'other' 中 'result' 的问题。
计算结果被就地保存在 'result' 张量中，
'clone_input' 将是 'input' 的一个副本，
'infos' 用于存储可能的错误检查信息，
'upper' 控制在计算中考虑的输入矩阵的部分，
'transpose' 如果为 true，则解决 'input.mT()' @ 'result' = 'other'，
'unitriangular' 如果为 true，则假定 'input' 的对角线元素为 1
*/
/*
static void triangular_solve_out_impl(
    const Tensor& result,
    const Tensor& clone_input,
    const Tensor& input,
    const Tensor& other,
    bool upper, bool transpose, bool unitriangular) {
  TORCH_WARN_ONCE(
    "torch.triangular_solve is deprecated in favor of torch.linalg.solve_triangular",
    "and will be removed in a future PyTorch release.\n",
    "torch.linalg.solve_triangular has its arguments reversed and does not return a copy of one of the inputs.\n",
    "X = torch.triangular_solve(B, A).solution\n",
    "should be replaced with\n",
    "X = torch.linalg.solve_triangular(A, B).");
  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of
  // the hierarchy of calls
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);  // 输入张量维度至少为2
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.size(-2) == input.size(-1));  // 输入张量在倒数第二和倒数第一维度上的大小需相等

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == other.device());  // 输入张量和其他张量需位于相同设备上
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == result.device());  // 输入张量和结果张量需位于相同设备上
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == clone_input.device());  // 输入张量和克隆输入张量需位于相同设备上

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == other.scalar_type());  // 输入张量和其他张量需具有相同的数据类型
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == result.scalar_type());  // 输入张量和结果张量需具有相同的数据类型
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == clone_input.scalar_type());  // 输入张量和克隆输入张量需具有相同的数据类型

  // if 'result' has no elements we can modify it
  // 如果结果张量 'result' 没有元素，可以修改它
  if (result.numel() == 0) {
    result.resize_(other.mT().sizes(), MemoryFormat::Contiguous);  // 重新调整 'result' 的大小为 'other' 的大小，并保持连续内存格式
    result.transpose_(-2, -1);  // 将 'result' 转置为 Fortran 连续的内存布局
  }

  // if 'clone_input' has no elements we can modify it
  // 如果克隆输入张量 'clone_input' 没有元素，可以修改它
  if (clone_input.numel() == 0) {
    clone_input.resize_(input.mT().sizes(), MemoryFormat::Contiguous);  // 重新调整 'clone_input' 的大小为 'input' 的大小，并保持连续内存格式
    clone_input.transpose_(-2, -1);  // 将 'clone_input' 转置为 Fortran 连续的内存布局
  }

  // 'result' and 'clone_input' must be in batched column major order (Fortran contiguous)
  // 'result' 和 'clone_input' 必须按批次列主序排列（Fortran 连续）
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.mT().is_contiguous());  // 检查 'result' 是否为连续的内存布局
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(clone_input.mT().is_contiguous());  // 检查 'clone_input' 是否为连续的内存布局

  // triangular_solve_stub performs calculations in-place
  // 'result' must be a copy of 'other'
  // 'clone_input' must be a copy of 'input'
  // triangular_solve_stub 在原地执行计算
  // 'result' 必须是 'other' 的副本
  // 'clone_input' 必须是 'input' 的副本
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(other.sizes()));  // 检查 'result' 和 'other' 的大小是否相同
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(clone_input.sizes().equals(input.sizes()));  // 检查 'clone_input' 和 'input' 的大小是否相同
  result.copy_(other);  // 将 'result' 复制为 'other'
  clone_input.copy_(input);  // 将 'clone_input' 复制为 'input'

  // Call the triangular_solve_stub function to perform the triangular solve operation
  // 调用 triangular_solve_stub 函数执行三角求解操作
  triangular_solve_stub(input.device().type(), clone_input, result, /*left=*/true, upper, transpose ? TransposeType::Transpose : TransposeType::NoTranspose, unitriangular);
}
*/
// 实现 TORCH_IMPL_FUNC 宏定义的 triangular_solve_out 函数，用于解三角线性方程组并输出结果
TORCH_IMPL_FUNC(triangular_solve_out)(const Tensor& self, const Tensor& A, bool upper, bool transpose, bool unitriangular, const Tensor& result, const Tensor& clone_A) {
  // 对 self 和 A 进行广播以匹配 batch 维度
  auto [self_broadcast, A_broadcast] = _linalg_broadcast_batch_dims(self, A, "triangular_solve");

  // 检查是否需要进行数据复制
  bool copy_needed = !result.transpose(-2, -1).is_contiguous();
  copy_needed |= !clone_A.transpose(-2, -1).is_contiguous();

  // 如果需要复制数据
  if (copy_needed) {
    // 创建临时的 result_tmp 和 clone_A_tmp 张量
    Tensor result_tmp = at::empty({0}, self.options());
    Tensor clone_A_tmp = at::empty({0}, A.options());

    // 调用 triangular_solve_out_impl 函数进行解三角线性方程组操作
    triangular_solve_out_impl(result_tmp, clone_A_tmp, A_broadcast, self_broadcast, upper, transpose, unitriangular);

    // 将计算结果拷贝回原始结果 result 和 clone_A
    result.copy_(result_tmp);
    clone_A.copy_(clone_A_tmp);
  } else {
    // 直接调用 triangular_solve_out_impl 函数进行解三角线性方程组操作
    triangular_solve_out_impl(result, clone_A, A_broadcast, self_broadcast, upper, transpose, unitriangular);
  }
}

// 定义 geqrf_stub 的分发调度器
DEFINE_DISPATCH(geqrf_stub);

// 辅助函数 geqrf_out_helper 用于在给定的输入张量上执行 QR 分解操作
static void geqrf_out_helper(const Tensor& input, const Tensor& QR, const Tensor& tau) {
  // 断言输入张量 input 至少为二维
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);

  // 断言输入张量 input, QR, tau 的数据类型和设备类型一致
  TORCH_INTERNAL_ASSERT(input.scalar_type() == QR.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == QR.device());

  TORCH_INTERNAL_ASSERT(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == tau.device());

  // 如果 QR 张量没有元素，则调整其大小并转置为 Fortran-contiguous 排列
  if (QR.numel() == 0) {
    QR.resize_as_(input.mT(), MemoryFormat::Contiguous);
    QR.transpose_(-2, -1); // 转置为 Fortran-contiguous 排列
  }

  // 计算预期的 batch 维度下 tau 张量的形状，并根据需要调整其大小
  auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2).vec(); // input.shape[:-2]
  expected_batch_tau_shape.push_back(std::min(input.size(-2), input.size(-1)));
  if (tau.numel() == 0) {
    tau.resize_(expected_batch_tau_shape);
  }

  // 断言 QR 张量必须是批处理列主序（Fortran-contiguous）排列
  TORCH_INTERNAL_ASSERT(QR.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT(QR.sizes().equals(input.sizes()));

  // 断言 tau 张量必须是连续的，并检查其形状是否符合预期的 batch 维度
  TORCH_INTERNAL_ASSERT(tau.is_contiguous());
  TORCH_INTERNAL_ASSERT(tau.sizes().equals(expected_batch_tau_shape));

  // 将 input 的数据拷贝到 QR 张量中
  QR.copy_(input);

  // 调用 geqrf_stub 函数执行 QR 分解操作
  geqrf_stub(input.device().type(), QR, tau);
}
// 返回两个引用类型的张量，其中 QR 和 tau 为输出张量，在 input 上进行 GEQRF 分解
std::tuple<Tensor&, Tensor&> geqrf_out(const Tensor& input, Tensor& QR, Tensor& tau) {
  // 检查输入张量维度是否至少为 2
  TORCH_CHECK(input.dim() >= 2, "torch.geqrf: input must have at least 2 dimensions.");

  // 检查并确保 QR 和 tau 张量与 input 在同一设备上
  checkSameDevice("torch.geqrf", QR, input, "a"); // 'a' is used in documentation and native_functions.yml
  checkSameDevice("torch.geqrf", tau, input, "tau");
  
  // 检查并确保 QR 和 tau 张量与 input 具有兼容的数据类型
  checkLinalgCompatibleDtype("torch.geqrf", QR, input, "a");
  checkLinalgCompatibleDtype("torch.geqrf", tau, input, "tau");

  // 检查 QR 是否与 input 具有相同的数据类型
  bool QR_input_same_type = (QR.scalar_type() == input.scalar_type());
  // 检查 tau 是否与 input 具有相同的数据类型
  bool tau_input_same_type = (tau.scalar_type() == input.scalar_type());
  // 检查 QR 是否具有预期的形状
  bool QR_equal_expected_shape = QR.sizes().equals(input.sizes());

  // 获取预期的批次维度形状，用于比较 tau 的形状是否与预期相同
  auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2).vec(); // input.shape[:-2]
  expected_batch_tau_shape.push_back(std::min(input.size(-2), input.size(-1)));
  // 检查 tau 是否具有预期的形状
  bool tau_equal_expected_shape = tau.sizes().equals(expected_batch_tau_shape);

  bool is_batched_column_major = false;
  // 如果 QR 的维度大于等于 2，则检查是否为批次列主格式
  if (QR.dim() >= 2) {
    is_batched_column_major = QR.mT().is_contiguous();
  }

  // 检查是否需要复制数据到临时张量
  bool copy_needed = (QR.numel() != 0 && !is_batched_column_major);
  copy_needed |= (QR.numel() != 0 && !QR_equal_expected_shape); // 或者 QR 不具有预期的形状
  copy_needed |= !QR_input_same_type;  // 或者 QR 与 input 的数据类型不同
  // 如果需要，还要分配临时张量

  copy_needed |= (tau.numel() != 0 && !tau.is_contiguous());
  copy_needed |= (tau.numel() != 0 && !tau_equal_expected_shape); // 或者 tau 不具有预期的形状
  copy_needed |= !tau_input_same_type;  // 或者 tau 与 input 的数据类型不同

  // 如果需要复制数据到临时张量
  if (copy_needed) {
    // 分配空的临时张量，与 input 具有相同的选项
    Tensor QR_tmp = at::empty({0}, input.options());
    Tensor tau_tmp = at::empty({0}, input.options());

    // 调用辅助函数 geqrf_out_helper 进行 GEQRF 分解
    geqrf_out_helper(input, QR_tmp, tau_tmp);

    // 调整输出张量 QR 和 tau 的大小，并复制临时张量中的数据
    at::native::resize_output(QR, QR_tmp.sizes());
    QR.copy_(QR_tmp);
    at::native::resize_output(tau, tau_tmp.sizes());
    tau.copy_(tau_tmp);
  } else {
    // 直接使用输出张量的存储进行计算
    geqrf_out_helper(input, QR, tau);
  }

  // 返回 QR 和 tau 张量的引用
  return std::tuple<Tensor&, Tensor&>(QR, tau);
}

// 对输入张量 input 进行 GEQRF 分解，返回 QR 和 tau 张量
std::tuple<Tensor, Tensor> geqrf(const Tensor& input) {
  // 分配空的输出张量 QR 和 tau，与 input 具有相同的选项
  Tensor QR = at::empty({0}, input.options());
  Tensor tau = at::empty({0}, input.options());
  // 调用 geqrf_out 函数执行 GEQRF 分解，并获取结果
  std::tie(QR, tau) = at::geqrf_outf(input, QR, tau);
  // 返回 QR 和 tau 张量的元组
  return std::make_tuple(std::move(QR), std::move(tau));
}
/*
  Computes the QR decomposition using GEQRF and ORGQR operations.
  This is an in-place function and Q, R tensors must have correct shape and be Fortran contiguous.

  Args:
  * `A` - [in] Input tensor for QR decomposition
  * `mode` - String specifying the mode of QR decomposition ('reduced' or 'complete')
  * `Q` - [out] Tensor containing the Q matrices of QR decomposition
  * `R` - [out] Tensor containing the R matrices of QR decomposition

  This function implements the LAPACK operations GEQRF and ORGQR for QR decomposition.
*/
TORCH_IMPL_FUNC(linalg_qr_out)(const Tensor& A,
                               c10::string_view mode,
                               const Tensor & Q,
                               const Tensor & R) {
  // Determine dimensions
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto k = std::min(m, n);
  auto [compute_q, reduced_mode] = at::native::_parse_qr_mode(mode);

  // Allocate workspace tensor for tau
  auto tau_shape = A.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = k;
  auto tau = A.new_empty(tau_shape);

  // Determine which tensor to use as QR based on shape and options
  Tensor QR;
  if (compute_q && Q.size(-1) == n) {
    // Use Q if dimensions match and compute_q is true
    QR = Q;
    QR.copy_(A);
  } else if (R.size(-2) == m) {
    // Otherwise, use R if dimensions match
    QR = R;
    QR.copy_(A);
  } else {
    // Otherwise, create a new tensor for QR decomposition
    QR = cloneBatchedColumnMajor(A);
  }

  // Perform GEQRF decomposition
  geqrf_stub(A.device().type(), QR, tau);

  // Split QR into Q (if compute_q is true) and R
  if (QR.is_alias_of(R)) {
    // Copy QR into Q and triangularize R
    if (compute_q) {
      // Adjust Q dimensions based on whether m < n or m >= n
      TORCH_INTERNAL_ASSERT(Q.size(-1) == m);
      if (m < n) {
        Q.copy_(QR.slice(-1, 0, m));
      } else {
        Q.slice(-1, 0, n).copy_(QR);
      }
    }
    R.triu_();
  } else {
    // Copy QR into R from Q or the auxiliary tensor
    at::triu_out(const_cast<Tensor&>(R), QR.slice(-2, 0, n));
  }

  // Perform ORGQR operation on Q if compute_q is true
  if (compute_q) {
    orgqr_stub(A.device().type(), const_cast<Tensor&>(Q), tau);
  }
}
    "The boolean parameter 'some' has been replaced with a string parameter 'mode'.\n",
    // 提示：布尔参数 'some' 已被替换为字符串参数 'mode'
    "Q, R = torch.qr(A, some)\n",
    // 调用 torch.qr 函数，使用参数 'some' 进行 QR 分解
    "should be replaced with\n",
    // 提示应当替换为以下形式
    "Q, R = torch.linalg.qr(A, 'reduced' if some else 'complete')"
    // 使用 torch.linalg.qr 函数，根据条件 'some' 选择 'reduced' 或 'complete' 模式
  );
  // 将条件 'some' 转换为字符串 'mode'，如果 'some' 为真则为 "reduced"，否则为 "complete"
  const char* mode = some ? "reduced" : "complete";
  // 调用 at::linalg_qr_out 函数，使用 'mode' 参数进行 QR 分解操作，并将结果保存在 Q 和 R 中
  return at::linalg_qr_out(Q, R, self, mode);
// 定义了一个名为 `orgqr_stub` 的分发函数
DEFINE_DISPATCH(orgqr_stub);

/*
  `householder_product_out_helper` 函数用于计算 Householder 乘积，重构正交（或单位）矩阵 Q，
  使用一系列下对角线的初等反射器（由 geqrf 函数生成）。

  Args:
  * `input` - 包含下对角线初等反射器方向的张量。
  * `tau` - 包含初等反射器的大小的张量。
  * `result` - 结果张量，将包含正交（或单位）矩阵 Q。

  更多细节，请参阅 LAPACK/MAGMA 文档。
*/
static Tensor& householder_product_out_helper(const Tensor& input, const Tensor& tau, Tensor& result) {
  // 断言输入张量至少有两个维度
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  // 断言输入张量的倒数第二维大于等于最后一维
  TORCH_INTERNAL_ASSERT(input.size(-2) >= input.size(-1));
  // 断言输入张量的最后一维大于等于 tau 张量的最后一维
  TORCH_INTERNAL_ASSERT(input.size(-1) >= tau.size(-1));

  // 断言输入张量和 tau 张量的数据类型一致
  TORCH_INTERNAL_ASSERT(input.scalar_type() == tau.scalar_type());
  // 断言输入张量和结果张量在相同设备上
  TORCH_INTERNAL_ASSERT(input.device() == tau.device());

  // 断言结果张量的数据类型和设备与输入张量一致
  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  // 如果结果张量没有元素，则可以修改它
  if (result.numel() == 0) {
    // 调整结果张量的大小与输入张量相同，并使用连续的内存格式
    at::native::resize_as_(result, input.mT(), MemoryFormat::Contiguous);
    // 将结果张量进行转置，倒数第二维和最后一维交换
    result.transpose_(-2, -1);
  }

  // 断言结果张量必须是批量列主序（Fortran 连续）
  TORCH_INTERNAL_ASSERT(result.mT().is_contiguous());
  // 断言结果张量的大小与输入张量相同
  TORCH_INTERNAL_ASSERT(result.sizes().equals(input.sizes()));

  // 断言 tau 张量必须是连续的
  Tensor tau_ = tau;
  if (!tau.is_contiguous()) {
    // 如果 tau 张量不连续，创建一个连续的副本
    tau_ = at::empty(tau.sizes(), tau.options(), MemoryFormat::Contiguous);
    tau_.copy_(tau);
  }

  // 将结果张量复制为输入张量，因为 orgqr_stub 函数会就地执行计算，结果必须是输入的副本
  result.copy_(input);

  // 调用 orgqr_stub 函数执行计算，并将结果存储在 result 中
  result = orgqr_stub(result.device().type(), result, tau_);
  return result;
}

// 对外暴露的函数，计算 Householder 乘积的接口，将结果存储在提供的 result 张量中
Tensor& linalg_householder_product_out(const Tensor& input, const Tensor& tau, Tensor& result) {
  // 检查输入张量至少有两个维度
  TORCH_CHECK(input.dim() >= 2, "torch.linalg.householder_product: input must have at least 2 dimensions.");
  // 检查输入张量倒数第二维大于等于最后一维
  TORCH_CHECK(
      input.size(-2) >= input.size(-1),
      "torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]");
  // 检查输入张量的最后一维大于等于 tau 张量的最后一维
  TORCH_CHECK(
      input.size(-1) >= tau.size(-1),
      "torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]");

  // 检查输入张量的维度比 tau 张量的维度少 1
  TORCH_CHECK(
      input.dim() - tau.dim() == 1,
      "torch.linalg.householder_product: Expected tau to have one dimension less than input, but got tau.ndim equal to ",
      tau.dim(),
      " and input.ndim is equal to ",
      input.dim());

  // 如果输入张量的维度大于 2，检查批处理形状是否符合预期
  if (input.dim() > 2) {
    auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2); // input.shape[:-2]
    auto actual_batch_tau_shape = IntArrayRef(tau.sizes().data(), tau.dim() - 1); // tau.shape[:-1]
    // 检查实际的批次 tau 的形状是否与预期的批次 tau 形状相等，如果不相等则抛出错误信息
    TORCH_CHECK(
        actual_batch_tau_shape.equals(expected_batch_tau_shape),
        "torch.linalg.householder_product: Expected batch dimensions of tau to be equal to input.shape[:-2], but got ",
        actual_batch_tau_shape);
    }
    
    // 检查 tau 的数据类型是否与 input 的数据类型相同，如果不相同则抛出错误信息
    TORCH_CHECK(
        tau.scalar_type() == input.scalar_type(),
        "torch.linalg.householder_product: tau dtype ",
        tau.scalar_type(),
        " does not match input dtype ",
        input.scalar_type());
    
    // 检查 tau 和 input 是否在相同的设备上，如果不在相同的设备上则抛出错误信息
    checkSameDevice("torch.linalg.householder_product", tau, input, "tau");
    checkSameDevice("torch.linalg.householder_product", result, input);
    
    // 检查 result 和 input 是否具有兼容的线性代数操作的数据类型，如果不兼容则抛出错误信息
    checkLinalgCompatibleDtype("torch.linalg.householder_product", result, input);
    
    // 如果 result 的元素数不为 0，则检查 result 的形状是否与 input 的形状相等，如果不相等则抛出错误信息
    // TODO: 当传递错误大小的 'result' 不被允许时，请取消下面的注释
    // if (result.numel() != 0) {
    //   // Resize messes up the strides, so let's not use at::native::resize_output
    //   TORCH_CHECK(result.sizes().equals(input.sizes()),
    //   "result shape ", result.sizes(), " does not match input shape ", input.sizes());
    // }
    
    // 检查 result 和 input 是否具有相同的数据类型
    bool result_input_same_type = (result.scalar_type() == input.scalar_type());
    
    // 检查 result 的形状是否与 input 的形状相等
    bool result_equal_expected_shape = result.sizes().equals(input.sizes());
    
    bool is_batched_column_major = false;
    if (result.dim() >= 2) {
      // 检查 result 是否按批次列主序连续存储
      is_batched_column_major = result.mT().is_contiguous();
    }
    
    // 如果 result 不是空且不是按批次列主序格式存储，需要复制操作
    bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
    // 或者 result 的数据类型与 input 的数据类型不同
    copy_needed |= !result_input_same_type;
    // 或者 result 的形状与 input 的形状不相等
    copy_needed |= (result.numel() != 0 && !result_equal_expected_shape);
    
    // 如果需要复制操作，则需要分配临时张量来进行计算
    if (copy_needed) {
      // 创建一个空的临时张量，使用 input 的选项（数据类型和设备）
      Tensor result_tmp = at::empty({0}, input.options());
      // 调用辅助函数进行计算，将结果存储到 result_tmp
      result_tmp = householder_product_out_helper(input, tau, result_tmp);
      // 调整 result 的输出大小以匹配 result_tmp 的大小
      at::native::resize_output(result, result_tmp.sizes());
      // 将 result_tmp 的值复制到 result 中
      result.copy_(result_tmp);
    } else {
      // 直接使用 result 的存储空间进行计算，调用辅助函数将结果存储到 result 中
      result = householder_product_out_helper(input, tau, result);
    }
    
    // 返回计算结果 result
    return result;
}

// 定义函数 linalg_householder_product，计算 Householder 乘积
Tensor linalg_householder_product(const Tensor& input, const Tensor& tau) {
  // 创建一个空的张量 result，与 input 具有相同的选项（如数据类型和设备）
  Tensor result = at::empty({0}, input.options());
  // 调用 linalg_householder_product_outf 函数，将结果存储到 result 中
  result = at::linalg_householder_product_outf(input, tau, result);
  // 返回计算结果的张量
  return result;
}

// 函数 orgqr_out 是 torch.orgqr 的别名，用于计算 Householder 乘积
// torch.linalg.householder_product 是推荐的新函数
Tensor& orgqr_out(const Tensor& input, const Tensor& tau, Tensor& result) {
  // 调用 linalg_householder_product_outf 函数，将结果存储到 result 中，并返回 result 引用
  return at::linalg_householder_product_outf(input, tau, result);
}

// 函数 orgqr 是 torch.linalg.householder_product 的别名，计算 Householder 乘积
Tensor orgqr(const Tensor& input, const Tensor& tau) {
  // 直接调用 linalg_householder_product 函数，返回计算结果的张量
  return at::linalg_householder_product(input, tau);
}

// 定义分发函数 ormqr_stub
DEFINE_DISPATCH(ormqr_stub);

// 辅助函数 ormqr_out_helper，执行 ormqr 操作
static void ormqr_out_helper(const Tensor& input, const Tensor& tau, const Tensor& other, const Tensor& result, bool left, bool transpose) {
  // 使用断言确保 input 和 other 张量的维度至少为 2
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other.dim() >= 2);

  // 使用断言确保 other 的倒数第二维或倒数第一维长度大于等于 tau 的最后一维长度
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other.size(left ? -2 : -1) >= tau.size(-1));
  // 使用断言确保 other 的倒数第二维（如果 left 为 true）或倒数第一维与 input 的倒数第二维长度相同
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other.size(left ? -2 : -1) == input.size(-2));

  // 使用断言确保 input 和 tau 的数据类型和设备类型相同
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == tau.device());

  // 使用断言确保 input 和 other 的数据类型和设备类型相同
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == other.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == other.device());

  // 使用断言确保 result 的数据类型和设备类型与 input 相同
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.device() == input.device());

  // 如果 result 没有元素，则可以修改它
  if (result.numel() == 0) {
    // 调整 result 的大小与 other 的 mT 大小相同，使用连续的内存格式
    at::native::resize_as_(result, other.mT(), MemoryFormat::Contiguous);
    // 将 result 在倒数第二维和倒数第一维进行转置
    result.transpose_(-2, -1);
  }

  // 使用断言确保 result 是批处理列主序（Fortran 连续）
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.mT().is_contiguous());
  // 使用断言确保 result 的尺寸与 other 相同
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(other.sizes()));

  // 确保 tau 张量是连续的，如果不是则复制成连续的
  Tensor tau_ = tau;
  if (!tau.is_contiguous()) {
    tau_ = at::empty(tau.sizes(), tau.options(), MemoryFormat::Contiguous);
    tau_.copy_(tau);
  }

  // 确保 input 张量是列主序（Fortran 连续），如果不是则复制成列主序
  Tensor input_ = input;
  if (!input.mT().is_contiguous()) {
    input_ = at::empty(input.mT().sizes(), input.options(), MemoryFormat::Contiguous);
    input_.transpose_(-2, -1);
    input_.copy_(input);
  }

  // 将 other 的内容复制到 result 中，用作 ormqr_stub 的计算结果的初始化
  result.copy_(other);

  // 调用 ormqr_stub 函数进行 inplace 计算，result 必须是 other 的副本
  ormqr_stub(result.device().type(), input_, tau_, result, left, transpose);
}
// 检查输入张量的维度是否至少为2，否则抛出错误信息
Tensor& ormqr_out(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose, Tensor& result) {
  TORCH_CHECK(input.dim() >= 2, "torch.ormqr: input must have at least 2 dimensions.");
  // 检查other张量的维度是否至少为2，否则抛出错误信息
  TORCH_CHECK(other.dim() >= 2, "torch.ormqr: other must have at least 2 dimensions.");

  // 根据left参数确定left_size_condition值，如果left为true，设为-2，否则设为-1
  int64_t left_size_condition = left ? -2 : -1;
  // 检查other张量在left_size_condition指定的维度上是否大于或等于tau张量的最后一个维度大小，否则抛出错误信息
  TORCH_CHECK(
      other.size(left_size_condition) >= tau.size(-1),
      "torch.ormqr: other.shape[",
      left_size_condition,
      "] must be greater than or equal to tau.shape[-1]");

  // 检查other张量在left_size_condition指定的维度上是否等于input张量倒数第二个维度的大小，否则抛出错误信息
  TORCH_CHECK(
      other.size(left_size_condition) == input.size(-2),
      "torch.ormqr: other.shape[",
      left_size_condition,
      "] must be equal to input.shape[-2]");

  // 检查tau张量的最后一个维度大小是否小于或等于input张量的最后一个维度大小，否则抛出错误信息
  TORCH_CHECK(
      tau.size(-1) <= input.size(-1),
      "torch.ormqr: tau.shape[-1] must be less than or equal to input.shape[-1]");

  // 检查input张量的维度是否比tau张量少1，否则抛出错误信息
  TORCH_CHECK(
      input.dim() - tau.dim() == 1,
      "torch.ormqr: ",
      "Expected tau to have one dimension less than input, but got tau.ndim equal to ",
      tau.dim(),
      " and input.ndim is equal to ",
      input.dim());
  // 检查input张量与other张量的维度是否相同，否则抛出错误信息
  TORCH_CHECK(
      input.dim() == other.dim(),
      "torch.ormqr: ",
      "Expected other to have the same number of dimensions as input, but got other.ndim equal to ",
      other.dim(),
      " and input.ndim is equal to ",
      input.dim());

  // 如果input张量的维度大于2
  if (input.dim() > 2) {
    // 获取input张量的前两个维度构成的预期批处理形状
    auto expected_batch_shape = IntArrayRef(input.sizes().data(), input.dim() - 2); // input.shape[:-2]
    // 获取tau张量的前N-1个维度构成的实际批处理形状
    auto actual_batch_tau_shape = IntArrayRef(tau.sizes().data(), tau.dim() - 1); // tau.shape[:-1]
    // 检查实际批处理形状是否与预期批处理形状相等，否则抛出错误信息
    TORCH_CHECK(
        actual_batch_tau_shape.equals(expected_batch_shape),
        "torch.ormqr: Expected batch dimensions of tau to be equal to input.shape[:-2], but got ",
        actual_batch_tau_shape);

    // 获取other张量的前两个维度构成的实际批处理形状
    auto actual_batch_other_shape = IntArrayRef(other.sizes().data(), other.dim() - 2); // other.shape[:-2]
    TORCH_CHECK(
        actual_batch_other_shape.equals(expected_batch_shape),
        "torch.ormqr: Expected batch dimensions of other to be equal to input.shape[:-2], but got ",
        actual_batch_other_shape);
  }


  // 检查实际的批次维度（actual_batch_other_shape）与期望的批次维度（expected_batch_shape）是否相等，如果不相等则抛出错误信息。
  TORCH_CHECK(
      tau.scalar_type() == input.scalar_type(),
      "torch.ormqr: Expected input and tau to have the same dtype, but input has dtype", input.scalar_type(),
      " and tau has dtype ", tau.scalar_type());


  // 检查输入（input）和 tau 张量的数据类型是否相同，如果不相同则抛出错误信息。
  TORCH_CHECK(
      other.scalar_type() == input.scalar_type(),
      "torch.ormqr: Expected input and other to have the same dtype, but input has dtype", input.scalar_type(),
      " and other has dtype ", other.scalar_type());


  // 检查输入（input）和 other 张量的数据类型是否相同，如果不相同则抛出错误信息。
  TORCH_CHECK(
      result.scalar_type() == input.scalar_type(),
      "torch.ormqr: Expected input and result to have the same dtype, but input has dtype", input.scalar_type(),
      " and result has dtype ", result.scalar_type());


  // 检查输入（input）和 result 张量的数据类型是否相同，如果不相同则抛出错误信息。
  checkSameDevice("torch.ormqr", tau, input, "tau");
  // 检查 tau 张量和 input 张量是否位于相同的设备上，否则抛出错误信息。
  checkSameDevice("torch.ormqr", other, input, "other");
  // 检查 other 张量和 input 张量是否位于相同的设备上，否则抛出错误信息。
  checkSameDevice("torch.ormqr", result, input);
  // 检查 result 张量和 input 张量是否位于相同的设备上，否则抛出错误信息。


  bool result_equal_expected_shape = result.sizes().equals(other.sizes());
  // 检查 result 张量的大小是否与 other 张量的大小相等，并将结果存储在 result_equal_expected_shape 中。

  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.mT().is_contiguous();
  }
  // 检查 result 张量的维度是否大于等于 2，如果是，则检查其是否按列主序（column major）存储，将结果存储在 is_batched_column_major 中。


  // 如果 result 张量不为空且不按批次列主序存储
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  // 或者 result 张量不为空且其大小与 other 张量的大小不相等，则需要复制数据
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape);
  // 如果需要复制数据，则需要分配一个临时张量
  // 否则，直接使用 result 的存储空间
  if (copy_needed) {
    // 分配一个空的临时张量，使用与 input 张量相同的选项
    Tensor result_tmp = at::empty({0}, input.options());
    // 调用 ormqr_out_helper 函数执行计算，结果存储在 result_tmp 中
    ormqr_out_helper(input, tau, other, result_tmp, left, transpose);
    // 调整 result 张量的大小以匹配 result_tmp 的大小
    at::native::resize_output(result, result_tmp.sizes());
    // 将 result_tmp 的数据复制到 result 张量中
    result.copy_(result_tmp);
  } else {
    // 直接使用 result 的存储空间执行 ormqr 计算
    ormqr_out_helper(input, tau, other, result, left, transpose);
  }


  // 返回经过计算后的 result 张量
  return result;
}

// 实现矩阵乘法的 ORMQR 函数，返回乘积结果张量
Tensor ormqr(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  // 创建一个空张量 result，与输入张量 input 具有相同的设备和数据类型
  Tensor result = at::empty({0}, input.options());
  // 调用 native 命名空间下的 ormqr_out 函数，计算乘积结果，并将结果存储在 result 中
  result = at::native::ormqr_out(input, tau, other, left, transpose, result);
  // 返回乘积结果张量 result
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eigh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 定义 linalg_eigh_stub 的分派函数
DEFINE_DISPATCH(linalg_eigh_stub);

/*
  计算张量 'input' 的特征值和特征向量。

  Args:
  * 'input' - 进行特征分解的输入张量
  * 'values' - 用于存储计算得到的特征值的张量
  * 'vectors' - 用于存储计算得到的特征向量的张量
  * 'infos' - 用于存储 LAPACK/MAGMA/cuSOLVER 错误代码的张量
  * 'compute_eigenvectors' - 控制是否计算特征向量
  * 'uplo' - 控制计算中输入矩阵的哪部分应被考虑，允许的值为 "u", "U", "l", "L"
    "u", "U" - 使用输入矩阵的上三角部分进行计算; "l", "L" - 使用下三角部分。
*/

// 实现 _linalg_eigh_out 函数，计算输入张量 A 的特征值和特征向量，并将结果存储在 L 和 V 中
TORCH_IMPL_FUNC(_linalg_eigh_out)(const Tensor& A,
                                  c10::string_view uplo,
                                  bool compute_v,
                                  const Tensor& L,
                                  const Tensor& V) {
  // 如果输入张量 A 中元素数为 0，则直接返回
  if (A.numel() == 0) {
    return;
  }

  // 将 uplo 字符串的首字母转换为大写
  auto uplo_uppercase = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo[0])));
  // 根据 uplo_uppercase 的值判断是否使用上三角部分进行计算
  bool upper = (uplo_uppercase == 'U');

  // 复制输入张量 A 到 V_，如果需要计算特征向量，则直接复制 A
  Tensor V_ = V;
  if (compute_v) {
    V_.copy_(A);
  } else {
    // 否则，需要使用 cloneBatchedColumnMajor 函数创建一个与 A 相同的张量 V_
    V_ = cloneBatchedColumnMajor(A);
  }

  // 创建一个与 A.dim() - 2 维度切片相同大小的零张量 info，数据类型为 kInt
  const auto info = at::zeros(A.sizes().slice(0, A.dim() - 2), A.options().dtype(kInt));
  // 调用 linalg_eigh_stub 分派函数，计算特征值和特征向量，并将结果存储在 L 和 V_ 中
  linalg_eigh_stub(A.device().type(), L, V_, info, upper, compute_v);

  // 检查并报告 linalg.eigh 函数中的错误，如果 A 是 2 维矩阵，则 is_matrix 设置为 true
  at::_linalg_check_errors(info, "linalg.eigh", /*is_matrix*/A.dim() == 2);
}

// 实现 linalg_eigh 函数，返回计算得到的特征值和特征向量元组
std::tuple<Tensor, Tensor> linalg_eigh(const Tensor& A, c10::string_view uplo) {
  // TODO (Good intro task) Implement linalg_eigh_ex_out
  // 调用 _linalg_eigh 函数，计算特征值和特征向量，并设置 compute_v 为 true
  return at::_linalg_eigh(A, uplo, /*compute_v*/true);
}

// 实现 linalg_eigh_out 函数，计算输入张量 A 的特征值和特征向量，并将结果存储在 L 和 V 中
std::tuple<Tensor&, Tensor&> linalg_eigh_out(const Tensor& A, c10::string_view uplo, Tensor& L, Tensor& V) {
  // 调用 _linalg_eigh_out 函数，计算特征值和特征向量，并将结果存储在 L 和 V 中，设置 compute_v 为 true
  return at::_linalg_eigh_out(L, V, A, uplo, /*compute_v=*/true);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// 该函数返回 LAPACK GEEV 的实值输出获取的复数特征向量
// 该函数也用于 MAGMA 路径，因为 MAGMA 的中间结果存储在 CPU 上
template <typename scalar_t>
// 实现函数 linalg_eig_make_complex_eigenvectors_impl，用于生成复特征向量
static void linalg_eig_make_complex_eigenvectors_impl(Tensor& result, const Tensor& complex_values, const Tensor& real_vectors) {
  // 从 GEEV 文档中得知：
  // 复共轭特征值成对连续出现，具有正虚部的特征值首先出现
  // 如果第 j 个特征值是实数，则 v(j) = VR(:,j)，即 VR 的第 j 列。
  // 如果第 j 和 (j+1) 个特征值构成一个复共轭对，则 v(j) = VR(:,j) + i*VR(:,j+1) 并且 v(j+1) = VR(:,j) - i*VR(:,j+1)。

  auto batch_size = batchCount(real_vectors);  // 计算批次大小
  auto n = real_vectors.size(-1);             // 获取特征向量的维度
  auto matrix_stride = matrixStride(real_vectors);  // 计算矩阵步长

  auto result_data = result.data_ptr<c10::complex<scalar_t>>();  // 获取结果张量的数据指针
  auto real_vectors_data = real_vectors.const_data_ptr<scalar_t>();  // 获取实部向量的数据指针
  auto values_data = complex_values.const_data_ptr<c10::complex<scalar_t>>();  // 获取复值的数据指针

  for (auto b = decltype(batch_size){0}; b < batch_size; b++) {  // 遍历批次
    const scalar_t* vecs = &real_vectors_data[b * matrix_stride];  // 获取当前批次的实部向量
    c10::complex<scalar_t>* res = &result_data[b * matrix_stride];  // 获取当前批次的结果向量
    const c10::complex<scalar_t>* vals = &values_data[b * n];       // 获取当前批次的复值

    for (auto j = decltype(n){0}; j < n; j++) {  // 遍历特征值
      if (vals[j].imag() == 0.0) {  // 如果特征值是实数
        for (auto i = decltype(n){0}; i < n; i++) {
          res[j * n + i] = c10::complex<scalar_t>(vecs[j * n + i], 0);  // 设置结果向量的实部
        }
      } else {  // 如果特征值是复数
        for (auto i = decltype(n){0}; i < n; i++) {
          res[j * n + i] = c10::complex<scalar_t>(vecs[j * n + i],  vecs[(j+1) * n + i]);      // 设置结果向量的实部和虚部
          res[(j+1) * n + i] = c10::complex<scalar_t>(vecs[j * n + i], -vecs[(j+1) * n + i]);  // 设置下一个结果向量的实部和虚部
        }
        j++;  // 跳过下一个特征值
      }
    }
  }
}

// 函数 linalg_eig_make_complex_eigenvectors 用于生成复特征向量，并进行相关的断言检查
static Tensor& linalg_eig_make_complex_eigenvectors(Tensor& complex_vectors, const Tensor& complex_values, const Tensor& real_vectors) {
  // 这些断言明确了 'linalg_eig_make_complex_eigenvectors_impl' 函数对张量的要求
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.device() == at::kCPU);  // 检查复向量的设备是 CPU
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.device() == at::kCPU);   // 检查复值的设备是 CPU
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.device() == at::kCPU);      // 检查实部向量的设备是 CPU

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.is_complex());   // 断言复向量是复数类型
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.is_complex());    // 断言复值是复数类型
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.is_floating_point());  // 断言实部向量是浮点类型

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.mT().is_contiguous());  // 断言复向量是连续的
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.is_contiguous());        // 断言复值是连续的
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.mT().is_contiguous());     // 断言实部向量是连续的

  // 使用模板分发调用 linalg_eig_make_complex_eigenvectors_impl 函数
  AT_DISPATCH_FLOATING_TYPES(real_vectors.scalar_type(), "linalg_eig_make_complex_vector", [&]{
    linalg_eig_make_complex_eigenvectors_impl<scalar_t>(complex_vectors, complex_values, real_vectors);
  });

  return complex_vectors;  // 返回生成的复特征向量
}

// 定义 linalg_eig_stub 的分发
DEFINE_DISPATCH(linalg_eig_stub);
// 返回类型为 std::tuple<Tensor&, Tensor&> 的函数 linalg_eig_out_info，接受输入 Tensor 和输出 Tensor values、vectors、infos，
// 以及一个布尔值 compute_eigenvectors，表示是否计算特征向量

// MAGMA 库对于 GEEV 过程没有 GPU 接口，要求输入必须在 CPU 上，因此在此处将所有中间张量创建在 CPU 上
auto options = input.options().device(at::kCPU);

// 下面的内部断言明确了实现中的假设
// 错误检查和实际错误消息在调用层次结构的更高级别上执行
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);  // 输入张量至少是二维的
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.size(-2) == input.size(-1));  // 输入张量的倒数第二维和最后一维大小相等

// 对于实值 'input'，特征值可以是实数或复数
TORCH_INTERNAL_ASSERT_DEBUG_ONLY((toComplexType(input.scalar_type()) == values.scalar_type()) || (input.scalar_type() == values.scalar_type()));
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.device() == at::kCPU);  // values 张量必须在 CPU 上

// 如果要计算特征向量，对于实值 'input'，特征向量可以是实数或复数
if (compute_eigenvectors) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY((toComplexType(input.scalar_type()) == vectors.scalar_type()) || (input.scalar_type() == vectors.scalar_type()));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.device() == at::kCPU);  // vectors 张量必须在 CPU 上
}

TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.scalar_type() == at::kInt);  // infos 张量必须是整数类型
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.device() == at::kCPU);  // infos 张量必须在 CPU 上
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.numel() == std::max<int64_t>(1, batchCount(input)));  // infos 张量的元素数量，至少为 1 或 batchCount(input) 的大小
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.is_contiguous());  // infos 张量必须是连续的

// 如果 vectors 张量没有元素且需要计算特征向量，则修改它的大小并转置，以使其在 Fortran 连续的内存布局中
if (vectors.numel() == 0 && compute_eigenvectors) {
  vectors.resize_(input.sizes(), MemoryFormat::Contiguous);
  vectors.transpose_(-2, -1);  // 将 vectors 张量转置，以使其具有 Fortran 连续的内存布局
}

// 如果 values 张量没有元素，则修改其大小为 values_shape，以保持连续的内存布局
auto values_shape = IntArrayRef(input.sizes().data(), input.dim()-1);  // input.shape[:-1]
if (values.numel() == 0) {
  values.resize_(values_shape, MemoryFormat::Contiguous);
}

// 如果计算特征向量，则 vectors 张量必须是批次列主序（Fortran 连续）
if (compute_eigenvectors) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.mT().is_contiguous());  // vectors 张量转置后必须是连续的
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.sizes().equals(input.sizes()));  // vectors 张量的大小必须与 input 张量的大小相同
}

// values 张量必须是连续的
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.sizes().equals(values_shape));  // values 张量的大小必须与 values_shape 相同

// 如果 input 是复数，则直接使用 values 张量，否则创建一个临时张量来保存实部和虚部，然后使用 at::complex_out
Tensor real_imag_values = values;

// 如果 input 是复数，则直接使用 vectors 张量，否则可能创建一个临时张量来保存实向量，然后使用 linalg_eig_make_complex_eigenvectors
Tensor maybe_complex_vectors = vectors;
if (!input.is_complex()) {
    // 定义用于存储输出实部和虚部的数组形状，前 n 个元素用于存储实部，后 n 个元素用于存储虚部
    auto real_imag_shape = IntArrayRef(input.sizes().data(), input.dim()-2).vec();  // input.shape[:-2]
    real_imag_shape.push_back(input.size(-1) * 2);
    real_imag_values = at::empty(real_imag_shape, options, MemoryFormat::Contiguous);

    // linalg_eig_stub 函数期望接收实数张量以存储特征向量
    // linalg_eig_stub 的输出需要稍后进行后处理，以生成复数特征向量
    // 仅当 'vectors' 是复数值时，才进行此后处理
    // 否则直接使用 'vectors' 的存储
    if (vectors.is_complex() && compute_eigenvectors) {
      maybe_complex_vectors = at::empty(input.sizes(), options, MemoryFormat::Contiguous);
      maybe_complex_vectors.transpose_(-2, -1);  // 使 'maybe_complex_vectors' 具有 Fortran 连续的内存布局
    }
  }

  // MAGMA 使用混合 CPU-GPU 算法，仅对大矩阵性能良好
  // 参见：https://github.com/pytorch/pytorch/pull/52491#issuecomment-795685687
  // 此处对小于 2048x2048 的矩阵调用 CPU 路径
  // 通常比调用 MAGMA 快得多
  if (input.size(-1) <= 2048) {
    linalg_eig_stub(at::kCPU, real_imag_values, maybe_complex_vectors, infos, input.to(kCPU), compute_eigenvectors);
  } else {
    linalg_eig_stub(input.device().type(), real_imag_values, maybe_complex_vectors, infos, input, compute_eigenvectors);
  }

  // 如果输入张量不是复数，需要进行一些后处理
  if (!input.is_complex()) {
    // 提取输出的实部和虚部
    auto real_values = real_imag_values.slice(/*dim=*/-1, /*start=*/0, /*end*/input.size(-1));
    auto imag_values = real_imag_values.slice(/*dim=*/-1, /*start=*/input.size(-1));

    // 如果虚部为零，无需进行任何操作
    bool is_zero_imag = at::all(imag_values == 0.0).item().toBool();
    if (is_zero_imag) {
      values.copy_(real_values);
      if (compute_eigenvectors) {
        vectors.copy_(maybe_complex_vectors);  // 对于 !vectors.is_complex()，因为 vectors.is_same(maybe_complex_vectors) == true，此操作无效
      }
      return std::tuple<Tensor&, Tensor&>(values, vectors);
    }

    // 如果 values 是复数类型，根据 real_values 和 imag_values 创建复数张量
    if (values.is_complex()) {
      values = at::complex_out(values, real_values, imag_values);
    } else {
      // 如果 values 不是复数类型，则抛出错误
      TORCH_CHECK(false, "torch.linalg.eig: imaginary part of eigenvalues is non-zero, can't safely cast eigenvalues to non-complex dtype.")
    }
    
    // 如果需要计算特征向量
    if (compute_eigenvectors) {
      // 如果 vectors 是复数类型，通过 linalg_eig_make_complex_eigenvectors 创建复数特征向量
      if (vectors.is_complex()) {
          vectors = linalg_eig_make_complex_eigenvectors(vectors, values, maybe_complex_vectors);
      } else {
        // 如果 vectors 不是复数类型，则抛出错误
        TORCH_CHECK(false, "torch.linalg.eig: imaginary part of eigenvectors is non-zero, can't safely cast eigenvectors to non-complex dtype.")
      }
    }
  }

  // 返回处理后的结果，包括特征值和特征向量
  return std::tuple<Tensor&, Tensor&>(values, vectors);
}

std::tuple<Tensor&, Tensor&> linalg_eig_out(const Tensor& input, Tensor& values, Tensor& vectors) {
  // 检查输入张量是否全部为有限值，即不包含无穷大或NaN
  TORCH_CHECK(input.isfinite().all().item<bool>(), "torch.linalg.eig: input tensor should not contain infs or NaNs.");
  // 检查输入张量是否为方阵
  squareCheckInputs(input, "linalg.eig");

  // 对于实值输入，输出始终是复数值
  // 检查值张量的数据类型是否与输入数据类型对应的复数类型匹配
  checkLinalgCompatibleDtype("torch.linalg.eig", values.scalar_type(), toComplexType(input.scalar_type()), "eigenvalues");
  // 检查向量张量的数据类型是否与输入数据类型对应的复数类型匹配
  checkLinalgCompatibleDtype("torch.linalg.eig", vectors.scalar_type(), toComplexType(input.scalar_type()), "eigenvectors");
  // 检查值张量和输入张量是否在相同的设备上
  checkSameDevice("torch.linalg.eig", values, input, "eigenvalues");
  // 检查向量张量和输入张量是否在相同的设备上
  checkSameDevice("torch.linalg.eig", vectors, input, "eigenvectors");

  // MAGMA 不支持 GPU 接口的 GEEV 程序，需要将输入张量置于 CPU 上
  auto options = input.options().device(at::kCPU);
  // 创建一个与批次数目相关的信息张量，初始化为零
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, options.dtype(kInt));

  // 如果结果张量不为空且不是批次列主要格式，需要分配一个临时张量
  bool is_batched_column_major = false;
  if (vectors.dim() >= 2) {
    // 检查向量张量是否为连续的转置形式
    is_batched_column_major = vectors.mT().is_contiguous();
  }

  // 检查值张量是否符合预期的数据类型
  bool values_expected_type = (values.scalar_type() == toComplexType(input.scalar_type()));
  // 检查向量张量是否符合预期的数据类型
  bool vectors_expected_type = (vectors.scalar_type() == toComplexType(input.scalar_type()));

  // 获取预期的值张量形状
  auto expected_values_shape = IntArrayRef(input.sizes().data(), input.dim()-1);  // input.shape[:-1]
  // 检查值张量的形状是否与预期形状相等
  bool values_equal_expected_shape = values.sizes().equals(expected_values_shape);
  // 检查向量张量的形状是否与输入张量的形状相等
  bool vectors_equal_expected_shape = vectors.sizes().equals(input.sizes());

  // 如果结果张量不为空且不是批次列主要格式，或者结果张量形状不符合预期
  bool values_tmp_needed = (values.numel() != 0 && !values.is_contiguous());
  // 如果结果张量不为空且不是批次列主要格式，或者结果张量形状不符合预期
  bool vectors_tmp_needed = (vectors.numel() != 0 && !is_batched_column_major);
  // 或者结果张量不符合预期的数据类型
  values_tmp_needed |= (values.numel() != 0 && !values_equal_expected_shape);
  // 或者结果张量不符合预期的数据类型
  vectors_tmp_needed |= (vectors.numel() != 0 && !vectors_equal_expected_shape);
  // 我们将分配一个临时张量并进行复制

  // 因为 MAGMA 的 GEEV 需要 CPU 输入和返回 CPU 输出
  // 位于 GPU 设备上的“out”张量不能直接使用
  values_tmp_needed |= values.is_cuda();
  // 位于 GPU 设备上的“out”张量不能直接使用
  vectors_tmp_needed |= vectors.is_cuda();

  // 确定临时张量的适当标量类型
  ScalarType values_type = input.scalar_type();
  ScalarType vectors_type = input.scalar_type();
  if (!input.is_complex()) {
    // 对于实值输入，可以有实值或复值输出
    ScalarType input_complex_dtype = toComplexType(input.scalar_type());
    values_type = values.is_complex() ? input_complex_dtype : values_type;
    // 如果 vectors 是复数类型，选择输入的复数数据类型作为 vectors_type，否则保持 vectors_type 不变
    vectors_type = vectors.is_complex() ? input_complex_dtype : vectors_type;
  }

  // 如果需要临时存储 values 和 vectors
  if (values_tmp_needed && vectors_tmp_needed) {
    // 创建空的 values_tmp 和 vectors_tmp 张量，使用指定的数据类型
    Tensor values_tmp = at::empty({0}, options.dtype(values_type));
    Tensor vectors_tmp = at::empty({0}, options.dtype(vectors_type));
    // 调用 linalg_eig_out_info 函数，填充 values_tmp 和 vectors_tmp，并返回结果
    std::tie(values_tmp, vectors_tmp) = linalg_eig_out_info(input, values_tmp, vectors_tmp, infos, true);
    // 调整输出张量 values 的大小，并复制数据
    at::native::resize_output(values, values_tmp.sizes());
    values.copy_(values_tmp);
    // 调整输出张量 vectors 的大小，并复制数据
    at::native::resize_output(vectors, vectors_tmp.sizes());
    vectors.copy_(vectors_tmp);
  } else if (!values_tmp_needed && vectors_tmp_needed) {
    // 如果不需要 values_tmp，直接使用 values 的存储空间
    Tensor vectors_tmp = at::empty({0}, options.dtype(vectors_type));
    // 调用 linalg_eig_out_info 函数，填充 vectors_tmp，并返回结果
    std::tie(values, vectors_tmp) = linalg_eig_out_info(input, values, vectors_tmp, infos, true);
    // 调整输出张量 vectors 的大小，并复制数据
    at::native::resize_output(vectors, vectors_tmp.sizes());
    vectors.copy_(vectors_tmp);
  } else if (values_tmp_needed && !vectors_tmp_needed) {
    // 如果不需要 vectors_tmp，直接使用 vectors 的存储空间
    Tensor values_tmp = at::empty({0}, options.dtype(values_type));
    // 调用 linalg_eig_out_info 函数，填充 values_tmp，并返回结果
    std::tie(values_tmp, vectors) = linalg_eig_out_info(input, values_tmp, vectors, infos, true);
    // 调整输出张量 values 的大小，并复制数据
    at::native::resize_output(values, values_tmp.sizes());
    values.copy_(values_tmp);
  } else {
    // 如果既不需要 values_tmp 也不需要 vectors_tmp，直接使用 values 和 vectors 的存储空间
    std::tie(values, vectors) = linalg_eig_out_info(input, values, vectors, infos, true);
  }

  // 检查 LAPACK/MAGMA 返回的错误码
  at::_linalg_check_errors(infos, "torch.linalg.eig", input.dim() == 2);
  // 返回 values 和 vectors 的引用
  return std::tuple<Tensor&, Tensor&>(values, vectors);
}

std::tuple<Tensor, Tensor> linalg_eig(const Tensor& input) {
  // 确定复数类型的标量类型
  ScalarType complex_dtype = toComplexType(input.scalar_type());
  // 创建空的张量用于存储特征值和特征向量，数据类型为复数类型
  Tensor values = at::empty({0}, input.options().dtype(complex_dtype));
  Tensor vectors = at::empty({0}, input.options().dtype(complex_dtype));

  // 调用ATen的linalg_eig_outf函数计算特征值和特征向量
  at::linalg_eig_outf(input, values, vectors);

  // 返回特征值和特征向量的元组
  return std::tuple<Tensor, Tensor>(values, vectors);
}

Tensor& linalg_eigvals_out(const Tensor& input, Tensor& values) {
  // 检查输入张量是否为方阵
  squareCheckInputs(input, "linalg.eigvals");

  // 对于实数输入，特征值始终是复数类型的，需进行类型兼容性检查
  checkLinalgCompatibleDtype("torch.linalg.eigvals", values.scalar_type(), toComplexType(input.scalar_type()), "eigenvalues");
  // 检查values张量与input张量在相同设备上
  checkSameDevice("torch.linalg.eigvals", values, input, "eigenvalues");

  // 使用CPU设备选项，因为MAGMA的GEEV例程不支持GPU接口
  auto options = input.options().device(at::kCPU);
  // 创建用于存储错误信息的infos张量
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, options.dtype(kInt));

  // 检查是否需要临时存储values张量
  bool values_expected_type = (values.scalar_type() == toComplexType(input.scalar_type()));
  auto expected_values_shape = IntArrayRef(input.sizes().data(), input.dim()-1);  // input.shape[:-1]
  bool values_equal_expected_shape = values.sizes().equals(expected_values_shape);
  bool values_tmp_needed = (values.numel() != 0 && !values.is_contiguous());
  values_tmp_needed |= (values.numel() != 0 && !values_equal_expected_shape);
  values_tmp_needed |= !values_expected_type;
  values_tmp_needed |= (!values.is_cpu());

  // 确定临时张量的数据类型
  ScalarType values_type = input.scalar_type();
  if (!input.is_complex()) {
    ScalarType input_complex_dtype = toComplexType(input.scalar_type());
    values_type = values.is_complex() ? input_complex_dtype : values_type;
  }

  Tensor vectors;
  if (values_tmp_needed) {
    // 创建临时张量values_tmp来存储特征值
    Tensor values_tmp = at::empty({0}, options.dtype(values_type));
    // 调用linalg_eig_out_info函数计算特征值，并不计算特征向量
    std::tie(values_tmp, std::ignore) = linalg_eig_out_info(input, values_tmp, vectors, infos, /*compute_eigenvectors=*/false);
    // 调整输出values的大小，并将values_tmp的值复制到values中
    at::native::resize_output(values, values_tmp.sizes());
    values.copy_(values_tmp);
  } else { // 直接使用values的存储空间
    // 调用linalg_eig_out_info函数计算特征值，并不计算特征向量
    std::tie(values, std::ignore) = linalg_eig_out_info(input, values, vectors, infos, /*compute_eigenvectors=*/false);
  }

  // 检查LAPACK/MAGMA的错误代码
  at::_linalg_check_errors(infos, "torch.linalg.eigvals", input.dim() == 2);
  // 返回计算得到的特征值
  return values;
}
// 定义函数 linalg_eigvals，计算输入张量的特征值
Tensor linalg_eigvals(const Tensor& input) {
  // 如果输入需要梯度，必须计算特征向量以使该函数可微分
  // 特征向量对用户不可见
  if (_may_require_fw_or_bw_grad(input)) {
    // 返回特征值的张量，仅包含特征值部分
    return std::get<0>(at::linalg_eig(input));
  }
  // 返回特征值的张量，仅包含特征值部分
  return at::_linalg_eigvals(input);
}

// 定义函数 _linalg_eigvals，计算输入张量的特征值
Tensor _linalg_eigvals(const Tensor& input) {
  // 将输入张量的数据类型转换为复数类型
  ScalarType complex_dtype = toComplexType(input.scalar_type());
  // 创建一个空张量，用于存储特征值，数据类型为复数类型
  Tensor values = at::empty({0}, input.options().dtype(complex_dtype));
  // 调用 linalg_eigvals_out 函数，计算并填充输入张量的特征值到 values 张量中
  linalg_eigvals_out(input, values);
  // 返回包含特征值的张量
  return values;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/* torch.svd, implemented in terms of torch.linalg.svd. There are two main
   differences:

    1. the 2nd parameter is bool some=True, which if effectively the opposite
       of full_matrices=True

    2. svd returns V, while linalg.svd returns Vh = V^H
*/

// 定义 _linalg_svd_out 函数，执行奇异值分解操作
DEFINE_DISPATCH(svd_stub);

TORCH_IMPL_FUNC(_linalg_svd_out)(const Tensor& A,
                                 const bool full_matrices,
                                 const bool compute_uv,
                                 std::optional<c10::string_view> driver,
                                 const Tensor & U,
                                 const Tensor & S,
                                 const Tensor & Vh) {
  // 如果张量 A 中元素数为 0
  if (A.numel() == 0) {
    // 当 A 的形状为 (3, 0) 且 full_matrices=True 时，确保 U 或 Vh 填充为恒等矩阵
    if (compute_uv && full_matrices) {
      // 如果 U 的元素数不为 0，则将其置零并填充为恒等矩阵
      if (U.numel() != 0) {
        U.zero_();
        U.diagonal(0, -2, -1).fill_(1.);
      }
      // 如果 Vh 的元素数不为 0，则将其置零并填充为恒等矩阵
      if (Vh.numel() != 0) {
        Vh.zero_();
        Vh.diagonal(0, -2, -1).fill_(1.);
      }
    }
    return;
  }

  // 根据 cuSOLVER 是否可用或驱动程序是否已定义，对不同情况进行检查
  const bool use_cusolver = at::native::svd_uses_cusolver(A);
  // 如果使用 cuSOLVER 或驱动程序已定义，则进行检查
  TORCH_CHECK(use_cusolver || !driver.has_value(),
  // 在 CUDA 输入和 cuSOLVER 后端上，`driver=` 关键字参数仅受支持。
  // 需要复制 A，因为其内容在计算 SVD 过程中会被销毁。
  // 现在，MAGMA 需要复制在 CPU 上，而 cuSOLVER 需要在 CUDA 上，因此我们将延迟
  // 将复制作为列主要矩阵传递给后端。
  const auto info = at::zeros(IntArrayRef(A.sizes().begin(), A.sizes().end() - 2), A.options().dtype(kInt));

  // 调用 SVD 的 stub 函数，根据设备类型调用不同的实现
  svd_stub(A.device().type(),
           A,
           full_matrices,
           compute_uv,
           driver,
           U, S, Vh, info);

  // TODO 这应该被移除，并将收敛性检查的代码从 svd_cusolver 移到这个函数中。
  // 然后我们应确保这个函数不会出错。
  at::_linalg_check_errors(info, "linalg.svd", /*is_matrix*/A.dim() == 2);
}

std::tuple<Tensor&, Tensor&, Tensor&>
linalg_svd_out(const Tensor& A,
               bool full_matrices,
               std::optional<c10::string_view> driver,
               Tensor & U,
               Tensor & S,
               Tensor & Vh) {
  // 此函数没有 _ex 变体，因为我们总是在内部检查错误以确保算法的收敛性。
  // 参见 https://github.com/pytorch/pytorch/issues/28293
  // 参见 https://github.com/pytorch/pytorch/issues/64237
  //
  // 我们必须将 linalg_svd 和 linalg_svdvals 都委托给 _linalg_svd（而不是将 linalg_svdvals 委托给 linalg_svd），
  // 因为：
  //   1. 我们不希望在 svd 中暴露 `compute_uv` 参数
  //   2. 我们希望在 svdvals 中利用 `compute_uv=False` 的优化
  // 唯一能够同时实现这两点并遵循组合规则的方法是分派到另一个函数。
  return at::_linalg_svd_out(U, S, Vh, A, full_matrices, /*compute_uv=*/true, driver);
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd(const Tensor& A, bool full_matrices,
    std::optional<c10::string_view> driver) {
  // 调用 _linalg_svd，返回 SVD 分解的结果元组
  return at::_linalg_svd(A, full_matrices, /*compute_uv=*/true, driver);
}

// 参见 linalg_svd 中的注释，解释为何该函数没有 _ex 变体
Tensor& linalg_svdvals_out(const Tensor& A, std::optional<c10::string_view> driver, Tensor & S) {
  // 创建虚拟的 U 和 Vh 张量
  auto U = at::empty({0}, A.options());
  auto Vh = at::empty({0}, A.options());
  // 调用 _linalg_svd_out 进行 SVD 分解，但不计算 U 和 Vh
  at::_linalg_svd_out(U, S, Vh, A, /*full_matrices=*/false, /*compute_uv=*/false, /*driver=*/driver);
  // 返回结果张量 S
  return S;
}

Tensor linalg_svdvals(const Tensor& A, std::optional<c10::string_view> driver) {
  // 调用 _linalg_svd 返回 SVD 分解结果的第二个元素，即奇异值张量 S
  return std::get<1>(at::_linalg_svd(A, /*full_matrices=*/false,
                     /*compute_uv=*/_may_require_fw_or_bw_grad(A),
                     /*driver=*/driver));
}

std::tuple<Tensor&, Tensor&, Tensor&> svd_out(const Tensor& self, bool some, bool compute_uv,
    Tensor& U, Tensor& S, Tensor& V) {

  if (compute_uv) {
    if (V.dim() >= 2) {
      V.transpose_(-2, -1);
    }
    // 调用 linalg_svd_out 进行 SVD 分解，根据参数确定是否计算 U 和 V
    at::linalg_svd_out(U, S, V, self, /*full_matrices=*/!some);
    V.transpose_(-2, -1);
    if (V.is_complex()) {
      // 如果 V 是复数类型，则进行共轭操作
      V.conj_physical_();
    }
  } else {
    // 检查输出张量的数据类型与输入张量的数据类型是否一致
    TORCH_CHECK(self.scalar_type() == U.scalar_type(),
    "torch.svd: Expected out tensor to have dtype ", self.scalar_type(), " but got ", U.scalar_type(), " instead");

    TORCH_CHECK(self.scalar_type() == V.scalar_type(),
    "torch.svd: Expected out tensor to have dtype ", self.scalar_type(), " but got ", V.scalar_type(), " instead");

    // 调用 linalg_svdvals_out 进行奇异值分解
    at::linalg_svdvals_out(S, self);
    // 如果 some == false，则返回大小为 (m, m), (n, n) 的 U 和 Vh 张量全为零
    const auto m = self.size(-2);
    const auto n = self.size(-1);
    auto sizes = self.sizes().vec();

    sizes.end()[-1] = m;
    // 调整 U 的尺寸并置零
    at::native::resize_output(U, sizes);
    U.zero_();

    sizes.end()[-2] = n;
    // 调整 V 的尺寸并置零
    sizes.end()[-1] = n;
    # 修改 sizes 容器的最后一个元素为 n
    at::native::resize_output(V, sizes);
    # 调用 PyTorch 的原生函数 resize_output 对张量 V 进行尺寸调整，使用 sizes 作为参数
    V.zero_();
    # 对张量 V 执行零初始化操作
  }

  return std::tie(U, S, V);
  # 返回一个 tuple，包含 U, S, V 三个变量的引用
}

std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& self, bool some, bool compute_uv) {
  // 当 torch.svd 被 torch.linalg.svd 替代且不再仅仅是文档时，取消下面的注释
  // torch/xla 阻止了在 at::linalg_pinv 代码中从 at::svd 切换到 at::linalg_svd
  // 参见 https://github.com/pytorch/xla/issues/2755
  // TORCH_WARN_ONCE(
  //     "torch.svd is deprecated in favor of torch.linalg.svd and will be ",
  //     "removed in a future PyTorch release.\n",
  //     "U, S, V = torch.svd(A, some=some, compute_uv=True) (default)\n",
  //     "should be replaced with\n",
  //     "U, S, Vh = torch.linalg.svd(A, full_matrices=not some)\n",
  //     "V = Vh.mH\n",
  //     "and\n",
  //     "_, S, _ = torch.svd(A, some=some, compute_uv=False)\n",
  //     "should be replaced with\n",
  //     "S = torch.linalg.svdvals(A)");
  // 检查输入张量的维度是否至少为 2，否则抛出异常
  TORCH_CHECK(self.dim() >= 2, "linalg.svd: input should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  // 初始化变量 U, S, Vh
  Tensor U, S, Vh;
  // 如果需要计算 U 和 Vh
  if (compute_uv) {
    // 调用 torch.linalg_svd 函数，根据参数 some 确定是否完整输出矩阵
    std::tie(U, S, Vh) = at::linalg_svd(self, /*full_matrices=*/!some);
  } else {
    // 否则，只计算奇异值 S
    S = at::linalg_svdvals(self);
    // 当 some == false 时，返回大小为 (m, m) 和 (n, n) 的零矩阵
    const auto m = self.size(-2);
    const auto n = self.size(-1);
    // 根据输入张量的尺寸创建大小相同的零矩阵 U 和 Vh
    auto sizes = self.sizes().vec();
    sizes.end()[-1] = m;
    U = at::zeros(sizes, self.options());
    sizes.end()[-2] = n;
    sizes.end()[-1] = n;
    Vh = at::zeros(sizes, self.options());
  }
  // 返回结果元组 (U, S, Vh.mH())
  return std::make_tuple(std::move(U), std::move(S), Vh.mH());
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lstsq ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(lstsq_stub);

/*
  解决最小二乘问题，即最小化 |B - A X| 的平方 Frobenius 范数。

  输入参数：
  * 'input' - 包含 m-by-n 矩阵 A 的批次的张量。
  * 'other' - 包含 max(m, n)-by-nrhs 矩阵 B 的批次的张量。
  * 'cond' - 确定 A 排名的相对容差。
  * 'driver' - 用于计算解的 LAPACK 驱动程序的名称。
  输出参数（原地修改）：
  * 'solution' - 用于存储解矩阵 X 的张量。
  * 'residuals' - 用于存储每列解的残差平方和的张量。
  * 'rank' - 用于存储 A 的秩的张量。
  * 'singular_values' - 用于存储 A 的奇异值的张量。
  * 'infos' - 用于存储线性代数数学库的错误代码的张量。

  更多细节，请参阅 LAPACK GELS/GELSY/GELSS/GELSD 例程的文档。
*/
static void linalg_lstsq_out_info(
    Tensor& solution,
    Tensor& residuals,
    Tensor& rank,
    Tensor& singular_values,
    Tensor& infos,
    const Tensor& input,
    const Tensor& other,
    double rcond,
  // 这些内部断言明确了实现中的假设
  // 实际的错误检查和错误信息在更高级别的调用层次上执行

  // 断言输入张量的维度至少为2
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  // 断言另一个张量的维度至少为1
  TORCH_INTERNAL_ASSERT(other.dim() >= 1);

  // 计算输入张量和另一个张量的维度差
  auto dim_diff = input.dim() - other.dim();
  // 断言维度差在0到1之间
  TORCH_INTERNAL_ASSERT(0 <= dim_diff && dim_diff <= 1);

  // 断言输入张量和另一个张量的标量类型相同
  TORCH_INTERNAL_ASSERT(input.scalar_type() == other.scalar_type());
  // 断言输入张量和另一个张量在相同设备上
  TORCH_INTERNAL_ASSERT(input.device() == other.device());

  // 断言解张量与输入张量的标量类型相同
  TORCH_INTERNAL_ASSERT(solution.scalar_type() == input.scalar_type());
  // 断言解张量与输入张量在相同设备上
  TORCH_INTERNAL_ASSERT(solution.device() == input.device());

  // 断言残差张量与输入张量在相同设备上
  TORCH_INTERNAL_ASSERT(residuals.device() == input.device());

  // 断言秩张量的标量类型为长整型
  TORCH_INTERNAL_ASSERT(rank.scalar_type() == at::kLong);
  // 断言秩张量与输入张量在相同设备上
  TORCH_INTERNAL_ASSERT(rank.device() == input.device());

  // 获取输入张量的实际数据类型
  auto real_dtype = toRealValueType(input.scalar_type());
  // 断言奇异值张量的数据类型与实际数据类型相同
  TORCH_INTERNAL_ASSERT(singular_values.scalar_type() == real_dtype);
  // 断言奇异值张量与输入张量在相同设备上
  TORCH_INTERNAL_ASSERT(singular_values.device() == input.device());

  // 断言信息张量的数据类型为整型
  TORCH_INTERNAL_ASSERT(infos.scalar_type() == at::kInt);
  // 断言信息张量与输入张量在相同设备上
  TORCH_INTERNAL_ASSERT(infos.device() == input.device());
  // 断言信息张量的元素数量为1或者批次数的最大值
  TORCH_INTERNAL_ASSERT(infos.numel() == std::max<int64_t>(1, batchCount(input)));
  // 断言信息张量是连续存储的
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());

  // 检查是否是向量求解情况，如果是，需要对 'other' 进行 unsqueeze 操作使其变为二维张量
  bool vector_case = linalg_solve_is_vector_rhs(input, other);
  // 断言输入张量的倒数第二个维度与 'other_2d' 的倒数第二个维度相同
  TORCH_INTERNAL_ASSERT(input.size(-2) == other_2d.size(-2));

  // 计算预期解张量的形状，用于广播批处理大小
  std::vector<int64_t> expected_solution_shape = broadcast_batch_size(input, other_2d, input.dim() - 2);
  // 实际返回的解的形状是 (*, n,) 或 (*, n, nrhs)，但 LAPACK 需要额外的维度存储原始残差
  // 所以预期的形状是 (*, max(m, n),) 或 (*, max(m, n), nrhs)
  auto m = input.size(-2);
  auto n = input.size(-1);
  auto nrhs = other.size(-1);
  expected_solution_shape.push_back(std::max(m, n));
  if (!vector_case) {
    expected_solution_shape.push_back(nrhs);
  }

  // 如果 'solution' 没有元素，可以修改它
  if (solution.numel() == 0) {
    // 如果是向量求解情况，重新调整解张量的形状并使用连续内存格式
    if (vector_case) {
      solution.resize_(expected_solution_shape, MemoryFormat::Contiguous);
    } else {
      // 如果不是向量求解情况，先转置形状，然后调整解张量的形状并转置
      auto shape_transposed = expected_solution_shape;
      std::swap(shape_transposed.end()[-1], shape_transposed.end()[-2]);
      solution.resize_(shape_transposed, MemoryFormat::Contiguous);
      solution.transpose_(-2, -1);
    }
  }

  // 如果 'solution' 不为空，它必须具有预期的形状
  TORCH_INTERNAL_ASSERT(solution.sizes().equals(expected_solution_shape));

  // 对于二维输入，'solution' 必须以批处理列主序（Fortran 连续）存储，对于一维输入，必须是 C 连续的
  if (vector_case) {
    TORCH_INTERNAL_ASSERT(solution.is_contiguous());
  } else {
    // 如果不是向量求解情况，继续注释的部分代码在下一个代码块中
  // 确保 solution 是连续的
  TORCH_INTERNAL_ASSERT(solution.mT().is_contiguous());
}

// 如果 'other' 是一维的，则在传递给 "apply_solve" 之前需要展开 'solution'
if (vector_case) {
  solution = solution.unsqueeze_(-1);
}

// _linalg_lstsq_helper_ 在原地执行计算，'solution' 必须是 'other_2d' 的一个拷贝
solution.narrow(-2, 0, other_2d.size(-2)).copy_(other_2d);

// 如果 'rank' 是空的，可能需要调整其大小
auto input_batch_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2);
if (rank.numel() == 0 && driver != "gels") { // gels 驱动程序不设置 'rank'
  rank.resize_(input_batch_shape, MemoryFormat::Contiguous);
}

// 如果 'rank' 是非空的，必须具有预期的形状并且是连续的
if (driver != "gels") {
  TORCH_INTERNAL_ASSERT(rank.sizes().equals(input_batch_shape));
  TORCH_INTERNAL_ASSERT(rank.is_contiguous());
}

// 如果 'singular_values' 是空的，可能需要调整其大小
auto singular_values_shape = input_batch_shape.vec();
singular_values_shape.push_back(std::min(m, n));
if (singular_values.numel() == 0 && (driver == "gelsd" || driver == "gelss")) {
  singular_values.resize_(singular_values_shape, MemoryFormat::Contiguous);
}

// 如果 'singular_values' 是非空的，必须具有预期的形状并且是连续的
if (driver == "gelsd" || driver == "gelss") {
  TORCH_INTERNAL_ASSERT(singular_values.sizes().equals(singular_values_shape));
  TORCH_INTERNAL_ASSERT(singular_values.is_contiguous());
}

// 'input' 在原地修改，因此需要一个按列主序的拷贝
auto input_working_copy = copyBatchedColumnMajor(input);

// 现在是实际调用，原地计算结果（apply_lstsq）
lstsq_stub(input.device().type(), input_working_copy, solution, rank, singular_values, infos, rcond, driver);

// 仅当 m > n 并且使用的驱动程序不是 gelsy 时才有残差可用
if (m > n && driver != "gelsy") {
  // 如果驱动程序是 gelss 或 gelsd，则只有在 rank == n 时才有残差可用
  bool compute_residuals = true;
  if (driver == "gelss" || driver == "gelsd") {
    if (input.dim() == 2) {
      compute_residuals = (rank.item().toInt() == n);
    } else {
      // 如果某些矩阵的秩小于 n，处理方式不明确
      // 目前只有当所有矩阵的秩都等于 n 时才计算残差
      // 这种行为可能会在将来改变
      // 参见 https://github.com/pytorch/pytorch/issues/56483
      compute_residuals = at::all(rank == n).item().toBool();
    }
  }
    // 如果需要计算残差
    if (compute_residuals) {
      // LAPACK 存储残差数据以供后续处理，存储在行 n:(m-n) 中
      auto raw_residuals = solution.narrow(/*dim=*/-2, /*start=*/n, /*length*/m - n);
      // 如果残差是复数
      if (raw_residuals.is_complex()) {
        // 对复数残差进行乘法操作（与其共轭相乘）
        raw_residuals.mul_(raw_residuals.conj());
        // 提取乘积的实部
        raw_residuals = at::real(raw_residuals);
      } else {
        // 对实数残差进行平方操作
        raw_residuals.pow_(2);
      }
      // 求和并将结果写入到 residuals 中
      at::sum_out(residuals, raw_residuals, /*dim=*/-2, /*keepdim=*/false, /*dtype*/real_dtype);
    }
  }
  // 从 solution 中创建视图，仅包含前 n 行
  auto solution_view = solution.narrow(/*dim=*/-2, /*start=*/0, /*length*/n);
  // 手动重新设置原始 solution 的存储方式
  solution.set_(solution.storage(), solution_view.storage_offset(), solution_view.sizes(), solution_view.strides());
  // 如果 m 等于 0，则将 solution 所有元素置为 0
  if (m == 0) {
    solution.zero_();
  }

  // 对于一维的 'other'，在 "apply_lstsq" 后需要挤压（去掉大小为 1 的维度）
  if (vector_case) {
    solution.squeeze_(-1);
  }
}



static std::string get_default_lstsq_driver(std::optional<c10::string_view> driver, const Tensor& input) {
  // 如果 `driver` 为空，根据输入张量类型设置默认驱动器名称：对于 CUDA 张量，设置为 "gels"，否则为 "gelsy"。
  std::string driver_str;
  // 检查用户提供的驱动器名称是否有效
  if (driver.has_value()) {
    driver_str = std::string(driver.value());
    // 将 `driver_str` 转换为小写
    std::transform(driver_str.begin(), driver_str.end(), driver_str.begin(),
      [](unsigned char c) { return std::tolower(c); });
    // 允许的驱动器集合
    static std::unordered_set<c10::string_view> allowed_drivers = {
      "gels", "gelsy", "gelsd", "gelss"
    };
    // 如果输入张量在 CPU 上
    if (input.device() == at::kCPU) {
      TORCH_CHECK(
        allowed_drivers.find(driver_str) != allowed_drivers.end(),
        "torch.linalg.lstsq: 参数 `driver` 应为 (gels, gelsy, gelsd, gelss) 中的一种"
      );
    } else { // 否则如果在 CUDA 上
      TORCH_CHECK(
        driver_str == "gels",
        "torch.linalg.lstsq: CUDA 上不支持除 `gels` 以外的其他 `driver`"
      );
    }
  } else {
    // 如果未提供驱动器名称，则根据输入张量类型设置默认值：在 CPU 上为 'gelsy'，在 CUDA 上为 'gels'
    driver_str = input.is_cuda() ? "gels" : "gelsy";
  }
  return driver_str;
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> linalg_lstsq_out(
    const Tensor& input,
    const Tensor& other,
    std::optional<double> rcond,
    std::optional<c10::string_view> driver,
    Tensor& solution,
    Tensor& residuals,
    Tensor& rank,
  // 检查输入张量的维度至少为2
  TORCH_CHECK(input.dim() >= 2, "torch.linalg.lstsq: input must have at least 2 dimensions.");
  // 检查other张量的维度至少为1
  TORCH_CHECK(other.dim() >= 1, "torch.linalg.lstsq: other must have at least 1 dimension.");
  // 检查input和other张量的数据类型必须相同
  TORCH_CHECK(
      input.scalar_type() == other.scalar_type(),
      "torch.linalg.lstsq: Expected input and other to have the same dtype, but got input's dtype ",
      input.scalar_type(),
      " and other's dtype ",
      other.scalar_type());

  // 计算输入张量和other张量的维度差异
  auto dim_diff = input.dim() - other.dim();
  // 检查维度差异需在0到1之间
  TORCH_CHECK(
      0 <= dim_diff && dim_diff <= 1,
      "torch.linalg.lstsq: input.dim() must be greater or equal to other.dim() and (input.dim() - other.dim()) <= 1");
  // 将other张量升维至二维，若有必要
  Tensor other_2d = dim_diff ? other.unsqueeze(-1) : other;
  // 检查input张量和other_2d张量在倒数第二维的大小是否一致
  TORCH_CHECK(
      input.size(-2) == other_2d.size(-2),
      dim_diff ? "torch.linalg.lstsq: input.size(-2) should match other.size(-1)"
               : "torch.linalg.lstsq: input.size(-2) should match other.size(-2)");

  // 检查所有相关张量与input张量在相同设备上
  checkSameDevice("torch.linalg.lstsq", other, input, "other");
  checkSameDevice("torch.linalg.lstsq", solution, input, "solution");
  checkSameDevice("torch.linalg.lstsq", residuals, input, "residuals");
  checkSameDevice("torch.linalg.lstsq", rank, input, "rank");
  checkSameDevice("torch.linalg.lstsq", singular_values, input, "singular_values");

  // 检查'solution'张量的数据类型与input张量相同
  checkLinalgCompatibleDtype("torch.linalg.lstsq", solution, input, "solution");

  // 检查'residuals'张量的数据类型为实数浮点型
  ScalarType real_dtype = c10::toRealValueType(input.scalar_type());
  checkLinalgCompatibleDtype("torch.linalg.lstsq", residuals.scalar_type(), real_dtype, "solution");

  // 检查'rank'张量的数据类型为整型
  // 实际的LAPACK调用使用int32_t类型，但为了与torch.linalg.matrix_rank输出数据类型一致，我们提升为int64_t
  ScalarType rank_expected_type = ScalarType::Long;
  checkLinalgCompatibleDtype("torch.linalg.lstsq", rank.scalar_type(), rank_expected_type, "rank");

  // 检查'singular_values'张量的数据类型为实数浮点型
  checkLinalgCompatibleDtype("torch.linalg.lstsq", singular_values.scalar_type(), real_dtype, "singular_values");

  // 获取默认的最小二乘解法驱动器名称
  std::string driver_name = get_default_lstsq_driver(driver, input);

  // 设置默认的rcond值
  double rcond_value = rcond.has_value()
    ? rcond.value()
  : _get_epsilon(c10::toRealValueType(input.scalar_type())) * std::max<int64_t>(input.size(-2), input.size(-1));
// 根据输入张量的数据类型计算一个小的浮点数 epsilon，用于数值稳定性
// epsilon 的大小是输入张量最后两个维度的大小的最大值乘以类型相关的常数

  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, input.options().dtype(kInt));
// 创建一个整数张量 infos，用于存储求解过程中的额外信息
// 张量的形状是 [1, max(1, batchCount(input))]，数据类型与输入张量相同

  // now check whether the provided output tensors can be used directly
// 现在检查提供的输出张量是否可以直接使用

  // Two types of 'other' tensors are supported:
  // - 1-dimensional (1D) tensor or batch of 1D tensors (vector case)
  // - 2-dimensional (2D) tensor or batch of 2D tensors (matrix case)
  // original torch.lstsq supported only the matrix case, while NumPy works for both cases
  // for the batched input we need to be able to distinguish them
  // auto expected_batched_rhs_shape = IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  // bool vector_case = other.dim() == 1 || (input.dim() - 1 == other.dim() && other.sizes().equals(expected_batched_rhs_shape));
// 支持两种类型的 'other' 张量：
// - 一维张量或一维张量的批次（向量情况）
// - 二维张量或二维张量的批次（矩阵情况）
// 原始的 torch.lstsq 仅支持矩阵情况，而 NumPy 支持这两种情况
// 对于批处理输入，我们需要能够区分它们

  bool vector_case = linalg_solve_is_vector_rhs(input, other);
// 判断 'other' 是否是向量类型的右手边（rhs），返回布尔值

  // provided output tensor can be used directly if:
  // 1. the shape matches the expected shape
  // 2. the dtype matches the expected dtype
  // 3. the tensor is contiguous
// 如果提供的输出张量满足以下条件，则可以直接使用：
// 1. 形状与期望形状匹配
// 2. 数据类型与期望数据类型匹配
// 3. 张量是连续的

  // Checks for the 'solution' tensor
// 对 'solution' 张量进行检查

  std::vector<int64_t> expected_solution_shape = broadcast_batch_size(input, other_2d, input.dim() - 2);
// 创建一个期望的 'solution' 张量形状，通过广播批次大小计算得出
// 返回的形状中的最后一个维度会是输入张量最后两个维度的最大值

  // the actual shape of the shape of the solution returned in (*, n,) or (*, n, nrhs)
  // but LAPACK requires extra dimensions so the expected shape is (*, max(m, n),) or (*, max(m, n), nrhs)
// 实际上返回的解的形状是 (*, n,) 或 (*, n, nrhs)
// 但 LAPACK 需要额外的维度，因此期望的形状是 (*, max(m, n),) 或 (*, max(m, n), nrhs)

  expected_solution_shape.push_back(std::max(input.size(-1), input.size(-2)));
// 将解的形状中最后一个维度扩展为输入张量最后两个维度的最大值

  if (!vector_case && other.dim() > 2) {
    expected_solution_shape.push_back(other.size(-1));
  }
// 如果 'other' 不是向量类型且维度大于2，则在形状中添加 'other' 的最后一个维度

  bool solution_equal_expected_shape = solution.sizes().equals(expected_solution_shape);
// 检查 'solution' 的形状是否与期望的形状相同
  bool solution_input_same_type = (solution.scalar_type() == input.scalar_type());
// 检查 'solution' 的数据类型是否与输入张量的数据类型相同

  bool is_solution_batched_column_major = false;
  if (vector_case) {
    is_solution_batched_column_major = solution.is_contiguous();
  } else if (!vector_case && solution.dim() >= 2) {
    is_solution_batched_column_major = solution.mT().is_contiguous();
  }
// 检查 'solution' 是否按列主序存储

  // 'residuals' is not checked here because at::sum_out(residuals, ...) does that
// 在这里不检查 'residuals'，因为 at::sum_out(residuals, ...) 已经处理了

  auto input_batch_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2);
// 获取输入张量的批次形状，不包括最后两个维度

  // Checks for the 'rank' tensor
// 对 'rank' 张量进行检查

  // rank is a scalar value for each matrix in the batch so
  // rank's expected shape is equal to input.shape[0:input.ndim-2]
// 'rank' 对于批量中的每个矩阵是一个标量值
// 因此，'rank' 的期望形状等于输入张量的前 (input.ndim-2) 维度的形状

  bool rank_equal_expected_shape = true;
  bool rank_equal_expected_type = true;
  bool rank_is_contiguous = true;
  if (driver_name != "gels") { // gels driver doesn't set 'rank'
    rank_equal_expected_shape = rank.sizes().equals(input_batch_shape);
    rank_equal_expected_type = (rank.scalar_type() == at::kLong);
// 如果驱动程序名称不是 "gels"，则进行以下检查：
// - 检查 'rank' 的形状是否与输入张量的批次形状相同
// - 检查 'rank' 的数据类型是否为 Long 类型
  // 检查 rank 是否是连续的
  rank_is_contiguous = rank.is_contiguous();

  // 检查 'singular_values' 张量
  // 目前只有使用 "gelsd" 和 "gelss" 驱动程序才会计算奇异值
  bool singular_values_equal_expected_shape = true;  // 奇异值张量形状是否符合预期
  bool singular_values_equal_expected_type = true;   // 奇异值张量数据类型是否符合预期
  bool singular_values_is_contiguous = true;         // 奇异值张量是否是连续的
  if (driver_name == "gelsd" || driver_name == "gelss") {
    auto singular_values_shape = input_batch_shape.vec();
    singular_values_shape.push_back(std::min(input.size(-1), input.size(-2)));
    singular_values_equal_expected_shape = singular_values.sizes().equals(singular_values_shape);  // 检查奇异值张量的形状是否符合预期
    singular_values_equal_expected_type = (singular_values.scalar_type() == real_dtype);  // 检查奇异值张量的数据类型是否符合预期
    singular_values_is_contiguous = singular_values.is_contiguous();  // 检查奇异值张量是否是连续的
  }

  // 如果需要复制，我们需要分配临时张量
  bool copy_needed = (solution.numel() != 0 && !is_solution_batched_column_major);  // 如果解不为空且不是批量列主格式，则需要复制
  copy_needed |= !solution_input_same_type;  // 或者解的数据类型与输入不同
  copy_needed |= (solution.numel() != 0 && !solution_equal_expected_shape);  // 或者解的形状不符合预期

  copy_needed |= !rank_equal_expected_type;  // 或者 rank 的数据类型不符合预期
  copy_needed |= (rank.numel() != 0 && !rank_equal_expected_shape);  // 或者 rank 的形状不符合预期
  copy_needed |= (rank.numel() != 0 && !rank_is_contiguous);  // 或者 rank 不是连续的

  copy_needed |= !singular_values_equal_expected_type;  // 或者奇异值的数据类型不符合预期
  copy_needed |= (singular_values.numel() != 0 && !singular_values_equal_expected_shape);  // 或者奇异值的形状不符合预期
  copy_needed |= (singular_values.numel() != 0 && !singular_values_is_contiguous);  // 或者奇异值不是连续的

  if (copy_needed) {  // 需要分配临时张量
    // 分配空的临时张量
    Tensor solution_tmp = at::empty({0}, input.options());
    Tensor residuals_tmp = at::empty({0}, input.options().dtype(real_dtype));
    Tensor rank_tmp = at::empty({0}, input.options().dtype(at::kLong));
    Tensor singular_values_tmp = at::empty({0}, input.options().dtype(real_dtype));

    // 调用 linalg_lstsq_out_info 函数填充临时张量
    linalg_lstsq_out_info(solution_tmp, residuals_tmp, rank_tmp, singular_values_tmp, infos, input, other, rcond_value, driver_name);

    // 调整输出张量的大小并复制数据
    at::native::resize_output(solution, solution_tmp.sizes());
    solution.copy_(solution_tmp);

    at::native::resize_output(residuals, residuals_tmp.sizes());
    residuals.copy_(residuals_tmp);

    at::native::resize_output(rank, rank_tmp.sizes());
    rank.copy_(rank_tmp);

    at::native::resize_output(singular_values, singular_values_tmp.sizes());
    singular_values.copy_(singular_values_tmp);
  } else {
    // 否则直接使用提供的输出存储
    linalg_lstsq_out_info(solution, residuals, rank, singular_values, infos, input, other, rcond_value, driver_name);
  }

  // 检查是否有线性代数错误，并确保 infos 张量元素数不超过 1
  at::_linalg_check_errors(infos, "torch.linalg.lstsq", infos.numel() <= 1);
  // 返回解、残差、秩和奇异值的元组引用
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(solution, residuals, rank, singular_values);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_lstsq(
    const Tensor& input, const Tensor& other,
    std::optional<double> rcond,
    std::optional<c10::string_view> driver) {
  // 创建空的张量来存储最小二乘解、残差、秩和奇异值
  Tensor solution = at::empty({0}, input.options());
  Tensor residuals = at::empty({0}, input.options().dtype(toRealValueType(input.scalar_type())));
  Tensor rank = at::empty({0}, input.options().dtype(at::kLong));
  Tensor singular_values = at::empty({0}, input.options().dtype(toRealValueType(input.scalar_type())));
  // 调用 linalg_lstsq_outf 函数计算最小二乘解、残差、秩和奇异值
  std::tie(solution, residuals, rank, singular_values) =
      at::linalg_lstsq_outf(input, other, rcond, driver, solution, residuals, rank, singular_values);
  // 返回结果的元组
  return std::make_tuple(std::move(solution), std::move(residuals), std::move(rank), std::move(singular_values));
}

DEFINE_DISPATCH(ldl_factor_stub);

TORCH_IMPL_FUNC(linalg_ldl_factor_ex_out)
(const Tensor& self,
 bool hermitian,
 bool check_errors,
 const Tensor& LD,
 const Tensor& pivots,
 const Tensor& info) {
  // 如果输入张量在批次维度上有0元素，则 LAPACK 工作区查询会崩溃
  if (self.numel() == 0) {
    // 将 info 张量置零并直接返回
    info.zero_();
    return;
  }

  // 决定不在 API 中包含 upper 标志
  // https://github.com/pytorch/pytorch/pull/69828#issuecomment-1015143819
  // 可以稍后重新考虑这个决定，从低级函数中完全删除 upper 或将其添加到公共 API 中
  bool upper = false;
  // 根据 upper 标志选择性地调用上三角或下三角矩阵输出函数
  if (upper) {
    at::triu_out(const_cast<Tensor&>(LD), self);
  } else {
    at::tril_out(const_cast<Tensor&>(LD), self);
  }

  // 调用 ldl_factor_stub 函数填充结果张量
  ldl_factor_stub(
      self.device().type(), LD, pivots, info, upper, hermitian);

  // 如果需要检查错误，则调用 _linalg_check_errors 函数
  if (check_errors) {
    at::_linalg_check_errors(
        info, "torch.linalg.ldl_factor_ex", self.dim() == 2);
  }
}

std::tuple<Tensor&, Tensor&> linalg_ldl_factor_out(
    const Tensor& self,
    bool hermitian,
    Tensor& LD,
    Tensor& pivots) {
  // 创建空的 info 张量，用于存储操作的状态信息
  auto info = at::empty({0}, self.options().dtype(kInt));
  // 将 check_errors 设置为 false，因为我们想使用 lu_factor 而不是 lu_factor_ex 来处理错误
  at::linalg_ldl_factor_ex_outf(
      self, hermitian, /*check_errors=*/false, LD, pivots, info);
  // 检查并处理操作的错误信息
  at::_linalg_check_errors(info, "torch.linalg.ldl_factor", self.dim() == 2);
  // 返回 LD 和 pivots 张量的元组
  return std::tie(LD, pivots);
}

std::tuple<Tensor, Tensor> linalg_ldl_factor(
    const Tensor& self,
    bool hermitian) {
  // 调用 linalg_ldl_factor_ex 函数计算 LDL 分解，并获取结果中的 LD 和 pivots 张量
  auto [LD, pivots, info] =
      at::linalg_ldl_factor_ex(self, hermitian, /*check_errors=*/false);
  // 检查并处理操作的错误信息
  at::_linalg_check_errors(info, "torch.linalg.ldl_factor", self.dim() == 2);
  // 返回 LD 和 pivots 张量的元组
  return std::make_tuple(std::move(LD), std::move(pivots));
}

DEFINE_DISPATCH(ldl_solve_stub);

TORCH_IMPL_FUNC(linalg_ldl_solve_out)
(const Tensor& LD,
 const Tensor& pivots,
 const Tensor& B,
 bool hermitian,
 const Tensor& result) {
  // 如果 LD 或 pivots 张量的元素个数为0，则直接返回，不执行操作
  if (LD.numel() == 0 || pivots.numel() == 0) {
    return;
  }

  // 期望获取连续的中心点（pivots），否则抛出异常
  auto pivots_ = pivots.expect_contiguous();

  // 根据 LD 的连续性情况，选择是否借用 LD 或者克隆 LD
  auto LD_ = at::native::borrow_else_clone(
      LD.mT().is_contiguous(), LD, LD, /*row_major=*/false);

  // 将 B 的数据复制到结果中
  result.copy_(B);

  // 断言调试模式下，结果的批次数与自身的批次数相等
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(result) == batchCount(result));

  // 调用 ldl_solve_stub 函数解决线性方程组
  ldl_solve_stub(
      B.device().type(), *LD_, *pivots_, result, false, hermitian);
/*
   解决矩阵方程 AX = B，其中 A 是一个三角矩阵。
   'left' 如果为 true，则解决 AX = B，如果为 false，则解决 XA = B。
   'upper' 控制在计算中考虑的输入矩阵部分。
   'unitriangular' 如果为 true，则假设对角线元素为1。
   'out' 包含结果的张量。如果 A == out，则会就地修改 A。
*/
Tensor& linalg_solve_triangular_out(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool left,
    bool unitriangular,
    Tensor& out
) {
  // 检查 A 和 B 是否为浮点数或复数类型
  checkFloatingOrComplex(A, "linalg.solve_triangular");
  // 检查 A 和 B 的数据类型是否相同
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "linalg.solve_triangular: Expected A and B to have the same dtype, but found A of type ",
              A.scalar_type(), " and B of type ", B.scalar_type(), " instead");
  // 检查 out 的数据类型与 A 是否相同
  TORCH_CHECK(out.scalar_type() == A.scalar_type(),
              "linalg.solve_triangular: Expected out of dtype ", A.scalar_type(),
              " but found ", out.scalar_type());
  // 检查 A 和 out 是否在相同的设备上
  checkSameDevice("linalg.solve_triangular", A, out);

  // 如果 upper 为 true，则解决上三角矩阵方程
  if (upper) {
    // 如果 left 为 true，则解决 AX = B
    if (left) {
      // 如果 unitriangular 为 true，则假设 A 的对角线元素为1，并解决 AX = B
      if (unitriangular) {
        return at::triangular_solve_out(out, B, A, /*upper=*/true, /*transpose=*/false, /*unitriangular=*/true);
      } else {
        return at::triangular_solve_out(out, B, A, /*upper=*/true, /*transpose=*/false);
      }
    }
    // 如果 left 为 false，则解决 XA = B
    else {
      if (unitriangular) {
        return at::triangular_solve_out(out, B, A, /*upper=*/true, /*transpose=*/true, /*unitriangular=*/true);
      } else {
        return at::triangular_solve_out(out, B, A, /*upper=*/true, /*transpose=*/true);
      }
    }
  }
  // 如果 upper 为 false，则解决下三角矩阵方程
  else {
    if (left) {
      if (unitriangular) {
        return at::triangular_solve_out(out, B, A, /*upper=*/false, /*transpose=*/false, /*unitriangular=*/true);
      } else {
        return at::triangular_solve_out(out, B, A, /*upper=*/false, /*transpose=*/false);
      }
    } else {
      if (unitriangular) {
        return at::triangular_solve_out(out, B, A, /*upper=*/false, /*transpose=*/true, /*unitriangular=*/true);
      } else {
        return at::triangular_solve_out(out, B, A, /*upper=*/false, /*transpose=*/true);
      }
    }
  }
}
    // 检查输入的张量 A, B, left 是否满足求解器的要求，使用字符串 "linalg.solve_triangular" 进行错误检查
    checkInputsSolver(A, B, left, "linalg.solve_triangular");
    // 将 B 和 A 进行批处理广播，返回广播后的张量 B_ 和 A_，并且不检查错误
    auto [B_, A_] = _linalg_broadcast_batch_dims(B, A, /*don't check errors*/nullptr);
    
    // 下面的算法旨在最小化复制和分配操作。伪代码如下：
    // 如果 out 的尺寸不正确：
    //   调整 out 的尺寸
    // 不变量：out 的尺寸正确
    // Tensor out_f; // 将传递给 FORTRAN 的张量
    // 如果 out 是 F-ready：
    //   out_f = out;
    // 否则：
    //   分配一个 F-ready 的 out_f
    // 如果 B 不等于 out_f：
    //   将 B 复制到 out_f 中
    // 不变量：out_f 是 F-ready 并且已经复制了 B
    // 如果 out_f 是 F-transpose 的：
    //   转置方程
    // 如果 out_f 是共轭的：
    //   共轭方程
    // 不变量：out_f 不是共轭的并且是 F-contig 的
    // Tensor A_f; // 将传递给 FORTRAN 的张量
    // 如果 A 是 F-ready：
    //   如果 A 是共轭的并且 A 不是转置的：
    //     需要在这种情况下克隆 A。参见 [Cloning A]
    //     将 A 克隆为 F-contig 的 A_f
    //   否则：
    //     A_f = A;
    // 否则：
    //   将 A 克隆为 F-contig 的 A_f
    // 不变量：out_f 是 F-contig 的并且 A_f 是 F-ready 的
    // 我们传递给 FORTRAN 的标志指示 A_f 是否被转置或共轭
    //
    // 在这里，如果需要，我们撤消 out_f 的共轭和转置操作
    //
    // 如果 out_f 不等于 out：
    //   将 out_f 复制到 out 中
    // 返回 out
    //
    // 注意：负数位的逻辑与共轭位的逻辑相同
    //
    // 注意：[Cloning A] 如果我们在算法开始时在需要时精心分配 B，它是可能的，这样我们在此处始终可以省略 A 的复制。
    // 通过这个技巧，算法在 A 和 B 都是 F-ready 并且不是 A.is_neg()（在实践中几乎总是如此）时，最多只会复制 A 或 B 的一个（从不是同时复制两者）。
    // 在大多数实际情况下，如果调用为 f(A, B, out=B)，它将不执行任何复制。
    
    const bool avoid_copy_A = A_.transpose(-2, -1).is_contiguous() && A_.is_conj();
    // 如果避免复制 A，则调整 out 的尺寸以匹配 B_.sizes()
    if (avoid_copy_A) {
      at::native::resize_output(out, B_.sizes());
    }
    else {
      // 使用 F-contig 的内存格式重新调整 out 的尺寸，如果需要的话进行检查
      if (resize_output_check(out, B_.sizes())) {
        out.resize_(B_.transpose(-2, -1).sizes(), MemoryFormat::Contiguous);
        out.transpose_(-2, -1);  // 使 'out' 具有 Fortran contiguous 的内存布局
      }
    }
    // 不变量：out 的尺寸正确，因此我们随后可以复制数据到其中
    
    Tensor out_f; // 将传递给 FORTRAN 的 out
    // 我们主要用 C10_LIKELY 作为文档，因为它有助于跟踪最有可能的路径
    if C10_LIKELY (is_row_or_column_contiguous(out)) {
      out_f = out;
      // 如果 out 不同于 B_，则将 B_ 复制到 out_f 中
      if C10_LIKELY (!out.is_same(B_)) {
        out_f.copy_(B_);
      }
    } else {
    if (avoid_copy_A) {
      // 如果避免复制 A，则根据 B_ 的内存格式克隆 out_f
      out_f = B_.clone(at::MemoryFormat::Contiguous);
    }
    else {
      // 否则调用函数 cloneBatchedColumnMajor 克隆 out_f
      out_f = cloneBatchedColumnMajor(B_);
    }
  }
  // 不变条件：out_f 已经准备好并且 B 已经复制到其中

  // out_f 经过 F 转置
  bool transpose_A = false;
  bool transpose_out_f = false;
  if (out_f.stride(-1) == 1) {
    // 如果 out_f 在最后一个维度的步幅为 1，则进行转置操作
    left = !left;  // 翻转 left 标志位
    transpose_A = true;  // 标记需要转置 A
    transpose_out_f = true;  // 标记 out_f 已经转置
    out_f.transpose_(-2 ,-1);  // 对 out_f 进行转置操作
  }

  // 如果 out_f 是共轭的，就无需共轭任何东西，因为 AX = conj(B) <=> conj(A)conj(X) = B
  // 并且 X = B 在算法执行后。我们只注明 A 后续会进行共轭操作
  // 解决方案将写入 out_f，因此它已经被共轭了
  // out_f 已经 F 转置

  Tensor A_f = std::move(A_);  // 准备进入 Fortran 的 A

  bool A_is_conj = A_f.is_conj() != out_f.is_conj();  // A 是否需要共轭
  bool A_is_neg = A_f.is_neg() != out_f.is_neg();  // A 是否需要取反
  bool A_is_f_contig = (A_f.stride(-1) == 1) == transpose_A;  // A 是否 F 连续
  if C10_UNLIKELY (!is_row_or_column_contiguous(A_f)) {
    // 首先在 A_f 上注明出自 out 的共轭 / 转置 / 取反，然后克隆结果张量以解决所有这些内存中的标志
    if (out_f.is_conj()) {
      // 如果 out_f 是共轭的，则将 A_f 共轭化
      A_f = A_f.conj();
    }
    A_is_conj = false;

    if (out_f.is_neg()) {
      // 如果 out_f 是取反的，则将 A_f 取反
      A_f = A_f._neg_view();
    }
    A_is_neg = false;

    // 选择与后续翻转 upper 一致的策略
    // 注意这与下面的取反和共轭应用相同的推理
    // 如果 B 具有取反、共轭或转置，则我们需要在内存中解决它
    A_f = transpose_A ? A_f.clone(at::MemoryFormat::Contiguous)
                      : cloneBatchedColumnMajor(A_f);
    A_is_f_contig = true;
  } else if C10_UNLIKELY (A_is_f_contig && A_is_conj) {
    if C10_UNLIKELY (A_f.is_neg() || out_f.is_neg()) {
      // 情况 A_is_neg（注意 B.is_neg() 当且仅当 out_f.is_same(B)）
      // -AX = -B => A(-X) = B。交换 A_f 的取反。在 X.is_same(B) 的情况下，X 没有变化，所以 X.is_neg() == true
      // -AX = B。我们在内存中解决取反
      // AX = -B => -A -X = B。我们在内存中解决 A 的取反
      // 由于 X.is_same(B)，我们已经知道 X.is_neg() == true

      // 我们使用视图进行取反，因为这将在下面的克隆中解决
      if (out_f.is_neg()) {
        // 如果 out_f 是取反的，则将 A_f 取反
        A_f = A_f._neg_view();
      }
      A_is_neg = false;
    }
    // 如果需要，我们解决转置，然后将 A_f 保持为 F 转置状态，
    // 因为 BLAS 可以处理 F 转置和共轭的情况
    A_f = at::clone(transpose_A ? A_f.mT() : A_f, at::MemoryFormat::Contiguous);
    A_is_f_contig = false;
    if (transpose_A) {
      // 如果需要转置 A，则翻转 upper 标志位
      upper = !upper;
    }
    // 由于在克隆中已经解决了 A 的共轭
    A_is_conj = out_f.is_conj();
  } else if C10_UNLIKELY (A_is_neg) {
    // 我们遵循与上述相同的逻辑，只是在这种情况下，我们需要在内存中执行取反操作
    if (out_f.is_neg()) {
      // 如果 out_f 是取反的，则将 A_f 取反
      A_f = -A_f;
    } else {
      // 否则，我们在 A_f 上解决取反
      A_f = A_f.resolve_neg();
    }
    A_is_neg = false;
    // 设置 A_is_neg 变量为 false，表示 A 不是负数

    // 因为我们已经在上面解决了 A 的共轭问题
    A_is_conj = out_f.is_conj();
  }
  // 不变量：out_f 是 F-连续的，A_f 是 F-准备好的
  // neg 已经解决了

  // 如果我们物理上将矩阵 F-转置，我们需要改变 upper 的奇偶性
  if (A_f.stride(-1) == 1) {
    upper = !upper;
  }

  // 调用三角求解的桩函数
  triangular_solve_stub(
    A_f.device().type(), A_f, out_f,
    /*left=*/left,
    /*upper=*/upper,
    /*transpose*/to_transpose_type(A_is_f_contig, A_is_conj),
    /*unitriangular=*/unitriangular);

  // 如果 transpose_out_f 为真，则对 out_f 进行转置操作
  if (transpose_out_f) {
    out_f.transpose_(-2, -1);
  }

  // 如果 out_f 和 out 不相同，则将 out_f 的内容复制到 out
  if (!out_f.is_same(out)) {
    out.copy_(out_f);
  }
  // 返回 out 变量
  return out;
}

// 定义一个函数 linalg_solve_triangular，用于解三角线性方程组
Tensor linalg_solve_triangular(
    const Tensor& A,  // 输入参数 A，表示要解的三角矩阵
    const Tensor& B,  // 输入参数 B，表示方程组右侧的向量或矩阵
    bool upper,       // 指定 A 是否为上三角矩阵
    bool left,        // 指定是否为左三角矩阵
    bool unitriangular) {  // 指定 A 是否为单位三角矩阵
  Tensor out = at::empty({0}, A.options());  // 创建一个空 Tensor 作为输出
  // 调用 linalg_solve_triangular_out 函数，将解的结果存入 out
  linalg_solve_triangular_out(A, B, upper, left, unitriangular, out);
  // 返回解的结果 Tensor out
  return out;
}

// 定义一个函数 linalg_vander_symint，用于生成 Vandermonde 矩阵
Tensor linalg_vander_symint(
    const Tensor& x,  // 输入参数 x，要生成 Vandermonde 矩阵的向量
    std::optional<c10::SymInt> N) {  // 可选参数 N，指定矩阵的阶数
  auto t = x.scalar_type();  // 获取输入 Tensor 的数据类型
  // 检查数据类型是否为浮点型、复数型或整型
  TORCH_CHECK(t == ScalarType::Float ||
              t == ScalarType::Double ||
              t == ScalarType::ComplexFloat ||
              t == ScalarType::ComplexDouble ||
              c10::isIntegralType(t, false),
              "linalg.vander supports floating point, complex, and integer tensors, but got ", t);
  
  const auto x_ = x.dim() == 0 ? x.unsqueeze(-1) : x;  // 如果 x 的维度为 0，则扩展为一维
  
  auto shape = x_.sym_sizes().vec();  // 获取 x_ 的符号尺寸作为初始形状
  const auto n = N.value_or(shape.back());  // 获取矩阵阶数 N，如果未提供则使用 x_ 的最后一个维度
  TORCH_CHECK(n > 1, "N must be greater than 1.");  // 检查阶数 N 必须大于 1

  // 在形状末尾添加一个维度，用于存储 0...n-1 的累积乘积
  shape.push_back(n - 1);
  // 执行累积乘积操作，生成 Vandermonde 矩阵
  auto result = at::cumprod(x_.unsqueeze(-1).expand_symint(shape), -1);

  // 添加一行全为 1 的矩阵作为结果的首行
  shape.back() = 1LL;
  auto ones =  result.new_ones_symint(shape);

  // 将全为 1 的矩阵和累积乘积矩阵拼接在一起，作为最终的 Vandermonde 矩阵返回
  return at::cat({std::move(ones), std::move(result)}, /*dim=*/ -1);
}
}  // namespace at::native
```