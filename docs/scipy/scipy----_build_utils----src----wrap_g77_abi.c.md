# `D:\src\scipysrc\scipy\scipy\_build_utils\src\wrap_g77_abi.c`

```
/*
一些线性代数库使用的Fortran ABI与构建scipy时使用的编译器不兼容（参见gh-11812）。这导致在调用返回复杂类型的函数时发生段错误。

本文件中的包装器确保兼容性，通过调用以下方式之一：
1. ABI独立的CBLAS API（cdotc、cdotu、zdotc、zdotu）
2. 没有复杂参数的Fortran函数（cladiv、zladiv）

当不需要这些包装器时，wrap_g77_dummy_abi.c提供了与此处相同名称（'w'前缀）和调用约定的包装器。

在构建时，Meson（scipy/meson.build中的g77_abi_wrappers）处理选择要编译的包装器文件。

在x86机器上，当Cython/F2PY生成的C代码调用带有'w'前缀的包装器时，会发生段错误，因为包装器返回C99复杂类型，而Cython/F2PY使用结构体复杂类型（`{float r, i;}`）。

Cython/F2PY代码应该调用本文件中以'wrp'结尾的包装器，将计算结果存储在变量的指针中。与返回值不同，结构体复杂参数可以正常工作，不会导致段错误。
*/

#include "npy_cblas.h"
#include "fortran_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <complex.h>
/* MSVC使用非标准的复杂类型名称。Intel编译器则不然。 */
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
typedef _Dcomplex double_complex;
typedef _Fcomplex float_complex;
#else /* !defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)*/
typedef double _Complex double_complex;
typedef float _Complex float_complex;
#endif

/* 定义了四个函数，分别用于调用CBLAS的复杂数乘积函数
   这些函数确保了正确的ABI调用约定，以避免在复杂类型返回时发生段错误 */

float_complex F_FUNC(wcdotc,WCDOTC)(CBLAS_INT *n, float_complex *cx, \
        CBLAS_INT *incx, float_complex *cy, CBLAS_INT *incy){
    float_complex ret;
    /* 调用cblas_cdotc_sub函数计算复数乘积，存储结果到ret中 */
    CBLAS_FUNC(cblas_cdotc_sub)(*n, cx, *incx, cy, *incy,&ret);
    return ret;
}

double_complex F_FUNC(wzdotc,WZDOTC)(CBLAS_INT *n, double_complex *zx, \
        CBLAS_INT *incx, double_complex *zy, CBLAS_INT *incy){
    double_complex ret;
    /* 调用cblas_zdotc_sub函数计算双精度复数乘积，存储结果到ret中 */
    CBLAS_FUNC(cblas_zdotc_sub)(*n, zx, *incx, zy, *incy,&ret);
    return ret;
}

float_complex F_FUNC(wcdotu,WCDOTU)(CBLAS_INT *n, float_complex *cx, \
        CBLAS_INT *incx, float_complex *cy, CBLAS_INT *incy){
    float_complex ret;
    /* 调用cblas_cdotu_sub函数计算复数点积，存储结果到ret中 */
    CBLAS_FUNC(cblas_cdotu_sub)(*n, cx, *incx, cy, *incy,&ret);
    return ret;
}

double_complex F_FUNC(wzdotu,WZDOTU)(CBLAS_INT *n, double_complex *zx, \
        CBLAS_INT *incx, double_complex *zy, CBLAS_INT *incy){
    double_complex ret;
    /* 调用cblas_zdotu_sub函数计算双精度复数点积，存储结果到ret中 */
    CBLAS_FUNC(cblas_zdotu_sub)(*n, zx, *incx, zy, *incy,&ret);
    return ret;
}

/* 定义了一个Fortran函数，用于计算单精度复数的除法 */
void BLAS_FUNC(sladiv)(float *xr, float *xi, float *yr, float *yi, \
    # 定义函数参数列表，指定参数类型为指向 float 的指针，用于返回实部和虚部
    float *retr, float *reti);
float_complex F_FUNC(wcladiv,WCLADIV)(float_complex *x, float_complex *y){
    float_complex ret;
    /* float_complex has the same memory layout as float[2], so we can
       cast and use pointer arithmetic to get the real and imaginary parts */
    // 调用 BLAS 中的 sladiv 函数，计算复数 x/y 的商，结果存入 ret 中
    BLAS_FUNC(sladiv)((float*)(x), (float*)(x)+1, \
        (float*)(y), (float*)(y)+1, \
        (float*)(&ret), (float*)(&ret)+1);
    return ret;
}

void BLAS_FUNC(dladiv)(double *xr, double *xi, double *yr, double *yi, \
    double *retr, double *reti);
double_complex F_FUNC(wzladiv,WZLADIV)(double_complex *x, double_complex *y){
    double_complex ret;
    /* double_complex has the same memory layout as double[2], so we can
       cast and use pointer arithmetic to get the real and imaginary parts */
    // 调用 BLAS 中的 dladiv 函数，计算双精度复数 x/y 的商，结果存入 ret 中
    BLAS_FUNC(dladiv)((double*)(x), (double*)(x)+1, \
        (double*)(y), (double*)(y)+1, \
        (double*)(&ret), (double*)(&ret)+1);
    return ret;
}

void F_FUNC(cdotcwrp,WCDOTCWRP)(float_complex *ret, CBLAS_INT *n, float_complex *cx, \
        CBLAS_INT *incx, float_complex *cy, CBLAS_INT *incy){
    // 调用 wcdotc 函数，计算复数向量 cx 和 cy 的共轭点积，结果存入 ret 中
    *ret = F_FUNC(wcdotc,WCDOTC)(n, cx, incx, cy, incy);
}

void F_FUNC(zdotcwrp,WZDOTCWRP)(double_complex *ret, CBLAS_INT *n, double_complex *zx, \
        CBLAS_INT *incx, double_complex *zy, CBLAS_INT *incy){
    // 调用 wzdotc 函数，计算双精度复数向量 zx 和 zy 的共轭点积，结果存入 ret 中
    *ret = F_FUNC(wzdotc,WZDOTC)(n, zx, incx, zy, incy);
}

void F_FUNC(cdotuwrp,CDOTUWRP)(float_complex *ret, CBLAS_INT *n, float_complex *cx, \
        CBLAS_INT *incx, float_complex *cy, CBLAS_INT *incy){
    // 调用 wcdotu 函数，计算复数向量 cx 和 cy 的点积，结果存入 ret 中
    *ret = F_FUNC(wcdotu,WCDOTU)(n, cx, incx, cy, incy);
}

void F_FUNC(zdotuwrp,ZDOTUWRP)(double_complex *ret, CBLAS_INT *n, double_complex *zx, \
        CBLAS_INT *incx, double_complex *zy, CBLAS_INT *incy){
    // 调用 wzdotu 函数，计算双精度复数向量 zx 和 zy 的点积，结果存入 ret 中
    *ret = F_FUNC(wzdotu,WZDOTU)(n, zx, incx, zy, incy);
}

void F_FUNC(cladivwrp,CLADIVWRP)(float_complex *ret, float_complex *x, float_complex *y){
    // 调用 wcladiv 函数，计算复数 x/y 的商，结果存入 ret 中
    *ret = F_FUNC(wcladiv,WCLADIV)(x, y);
}

void F_FUNC(zladivwrp,ZLADIVWRP)(double_complex *ret, double_complex *x, double_complex *y){
    // 调用 wzladiv 函数，计算双精度复数 x/y 的商，结果存入 ret 中
    *ret = F_FUNC(wzladiv,WZLADIV)(x, y);
}

#ifdef __cplusplus
}
#endif
```