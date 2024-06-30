# `D:\src\scipysrc\scipy\scipy\_build_utils\src\wrap_dummy_g77_abi.c`

```
/*
Some linear algebra libraries are built with a Fortran ABI that is incompatible
with the compiler used to build scipy (see gh-11812). This results in segfaults
when calling functions with complex-valued return types.

The wrappers in wrap_g77_abi.c ensure compatibility by calling either:
1. The ABI-independent CBLAS API (cdotc, cdotu, zdotc, zdotu)
2. Fortran functions without complex-valued args/return type (cladiv, zladiv)

When these wrappers are not necessary, THIS FILE provides wrappers that have
the same name ('w' prefix) and calling convention as those in wrap_g77_abi.c.

The choice of which wrapper file to compile with is handled at build time by
Meson (g77_abi_wrappers in scipy/meson.build).

On x86 machines, segfaults occur when Cython/F2PY-generated C code calls the
'w'-prefixed wrappers because the wrappers return C99 complex types while
Cython/F2PY use struct complex types (`{float r, i;}`).

Cython/F2PY code should instead call the 'wrp'-suffixed wrappers in this file,
passing a pointer to a variable in which to store the computed result. Unlike
return values, struct complex arguments work without segfaulting.
*/
#include "npy_cblas.h"  // Include CBLAS header
#include "fortran_defs.h"  // Include Fortran definitions header

#ifdef __cplusplus
extern "C" {
#endif

#include <complex.h>  // Include standard complex number header

/* MSVC uses non-standard names for complex types. Intel compilers do not. */
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
typedef _Dcomplex double_complex;  // Define double complex type for MSVC
typedef _Fcomplex float_complex;  // Define float complex type for MSVC
#else /* !defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)*/
typedef double _Complex double_complex;  // Define double complex type for other compilers
typedef float _Complex float_complex;  // Define float complex type for other compilers
#endif

// Declaration of 'cdotc' wrapper function for float complex dot product
float_complex BLAS_FUNC(cdotc)(CBLAS_INT *n, float_complex *cx, CBLAS_INT *incx, \
    float_complex *cy, CBLAS_INT *incy);

// Implementation of 'wcdotc' wrapper function for float complex dot product
float_complex F_FUNC(wcdotc,WCDOTC)(CBLAS_INT *n, float_complex *cx, \
        CBLAS_INT *incx, float_complex *cy, CBLAS_INT *incy){
    return BLAS_FUNC(cdotc)(n, cx, incx, cy, incy);  // Call the actual 'cdotc' function
}

// Declaration of 'zdotc' wrapper function for double complex dot product
double_complex BLAS_FUNC(zdotc)(CBLAS_INT *n, double_complex *zx, CBLAS_INT *incx, \
    double_complex *zy, CBLAS_INT *incy);

// Implementation of 'wzdotc' wrapper function for double complex dot product
double_complex F_FUNC(wzdotc,WZDOTC)(CBLAS_INT *n, double_complex *zx, \
        CBLAS_INT *incx, double_complex *zy, CBLAS_INT *incy){
    return BLAS_FUNC(zdotc)(n, zx, incx, zy, incy);  // Call the actual 'zdotc' function
}

// Declaration of 'cdotu' wrapper function for float complex dot product
float_complex BLAS_FUNC(cdotu)(CBLAS_INT *n, float_complex *cx, CBLAS_INT *incx, \
    float_complex *cy, CBLAS_INT *incy);

// Implementation of 'wcdotu' wrapper function for float complex dot product
float_complex F_FUNC(wcdotu,WCDOTU)(CBLAS_INT *n, float_complex *cx, \
        CBLAS_INT *incx, float_complex *cy, CBLAS_INT *incy){
    return BLAS_FUNC(cdotu)(n, cx, incx, cy, incy);  // Call the actual 'cdotu' function
}

// Declaration of 'zdotu' wrapper function for double complex dot product
double_complex BLAS_FUNC(zdotu)(CBLAS_INT *n, double_complex *zx, CBLAS_INT *incx, \
    double_complex *zy, CBLAS_INT *incy);

// Implementation of 'wzdotu' wrapper function for double complex dot product
double_complex F_FUNC(wzdotu,WZDOTU)(CBLAS_INT *n, double_complex *zx, \
        CBLAS_INT *incx, double_complex *zy, CBLAS_INT *incy){
    return BLAS_FUNC(zdotu)(n, zx, incx, zy, incy);  // Call the actual 'zdotu' function
}

// Declaration of 'cladiv' wrapper function for float complex division
float_complex BLAS_FUNC(cladiv)(float_complex *x, float_complex *y);
float_complex F_FUNC(wcladiv,WCLADIV)(float_complex *x, float_complex *y){
    // 调用 BLAS 库中的 cladiv 函数，计算两个复数 x 和 y 的商并返回结果
    return BLAS_FUNC(cladiv)(x, y);
}

double_complex BLAS_FUNC(zladiv)(double_complex *x, double_complex *y);
double_complex F_FUNC(wzladiv,WZLADIV)(double_complex *x, double_complex *y){
    // 调用 BLAS 库中的 zladiv 函数，计算两个双精度复数 x 和 y 的商并返回结果
    return BLAS_FUNC(zladiv)(x, y);
}

void F_FUNC(cdotcwrp,WCDOTCWRP)(float_complex *ret, CBLAS_INT *n, float_complex *cx, \
        CBLAS_INT *incx, float_complex *cy, CBLAS_INT *incy){
    // 调用 wcdotc 函数，计算复数向量 cx 和 cy 的共轭点乘，返回结果存储在 ret 中
    *ret = F_FUNC(wcdotc,WCDOTC)(n, cx, incx, cy, incy);
}

void F_FUNC(zdotcwrp,WZDOTCWRP)(double_complex *ret, CBLAS_INT *n, double_complex *zx, \
        CBLAS_INT *incx, double_complex *zy, CBLAS_INT *incy){
    // 调用 wzdotc 函数，计算双精度复数向量 zx 和 zy 的共轭点乘，返回结果存储在 ret 中
    *ret = F_FUNC(wzdotc,WZDOTC)(n, zx, incx, zy, incy);
}

void F_FUNC(cdotuwrp,CDOTUWRP)(float_complex *ret, CBLAS_INT *n, float_complex *cx, \
        CBLAS_INT *incx, float_complex *cy, CBLAS_INT *incy){
    // 调用 wcdotu 函数，计算复数向量 cx 和 cy 的非共轭点乘，返回结果存储在 ret 中
    *ret = F_FUNC(wcdotu,WCDOTU)(n, cx, incx, cy, incy);
}

void F_FUNC(zdotuwrp,ZDOTUWRP)(double_complex *ret, CBLAS_INT *n, double_complex *zx, \
        CBLAS_INT *incx, double_complex *zy, CBLAS_INT *incy){
    // 调用 wzdotu 函数，计算双精度复数向量 zx 和 zy 的非共轭点乘，返回结果存储在 ret 中
    *ret = F_FUNC(wzdotu,WZDOTU)(n, zx, incx, zy, incy);
}

void F_FUNC(cladivwrp,CLADIVWRP)(float_complex *ret, float_complex *x, float_complex *y){
    // 调用 wcladiv 函数，计算两个复数 x 和 y 的商并返回结果，存储在 ret 中
    *ret = F_FUNC(wcladiv,WCLADIV)(x, y);
}

void F_FUNC(zladivwrp,ZLADIVWRP)(double_complex *ret, double_complex *x, double_complex *y){
    // 调用 wzladiv 函数，计算两个双精度复数 x 和 y 的商并返回结果，存储在 ret 中
    *ret = F_FUNC(wzladiv,WZLADIV)(x, y);
}

#ifdef __cplusplus
}
#endif


这些注释详细说明了每个函数的作用和参数，确保了代码的可读性和理解性。
```