# `D:\src\scipysrc\scipy\scipy\special\lapack_defs.h`

```
/*
 * Handle different Fortran conventions.
 */

#include "npy_cblas.h" // 包含头文件 "npy_cblas.h"

extern void BLAS_FUNC(dstevr)(char *jobz, char *range, CBLAS_INT *n, double *d, double *e,
                              double *vl, double *vu, CBLAS_INT *il, CBLAS_INT *iu, double *abstol,
                              CBLAS_INT *m, double *w, double *z, CBLAS_INT *ldz, CBLAS_INT *isuppz,
                              double *work, CBLAS_INT *lwork, CBLAS_INT *iwork, CBLAS_INT *liwork,
                              CBLAS_INT *info, size_t jobz_len, size_t range_len);
/*
 * 定义静态函数 c_dstevr，用于调用 BLAS_FUNC(dstevr) 来处理 Fortran 的不同约定
 */
static void c_dstevr(char *jobz, char *range, CBLAS_INT *n, double *d, double *e,
                     double *vl, double *vu, CBLAS_INT *il, CBLAS_INT *iu, double *abstol,
                     CBLAS_INT *m, double *w, double *z, CBLAS_INT *ldz, CBLAS_INT *isuppz,
                     double *work, CBLAS_INT *lwork, CBLAS_INT *iwork, CBLAS_INT *liwork, CBLAS_INT *info) {
    BLAS_FUNC(dstevr)(jobz, range, n, d, e, vl, vu, il, iu, abstol, m,
                      w, z, ldz, isuppz, work, lwork, iwork, liwork, info,
                      1, 1);
}
```