# `D:\src\scipysrc\scipy\scipy\_build_utils\src\npy_cblas.h`

```
/*
 * This header provides numpy a consistent interface to CBLAS code. It is needed
 * because not all providers of cblas provide cblas.h. For instance, MKL provides
 * mkl_cblas.h and also typedefs the CBLAS_XXX enums.
 */
#ifndef _NPY_CBLAS_H_
#define _NPY_CBLAS_H_

#include <numpy/npy_common.h>
#include <stddef.h>

/* Allow the use in C++ code.  */
#ifdef __cplusplus
extern "C"
{
#endif

/*
 * Enumerated and derived types
 */

// Enum defining the matrix storage order: row-major or column-major
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};

// Enum defining transpose operations: no transpose, transpose, conjugate transpose
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

// Enum defining whether matrices are upper or lower triangular
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};

// Enum defining whether matrices are unit or non-unit triangular
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};

// Enum defining which side matrices are multiplied: left or right
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

#define CBLAS_INDEX size_t  /* this may vary between platforms */

#ifdef NO_APPEND_FORTRAN
#define BLAS_FORTRAN_SUFFIX
#else
#define BLAS_FORTRAN_SUFFIX _
#endif

// Define BLAS function suffix based on ACCELERATE_NEW_LAPACK macro
#ifdef ACCELERATE_NEW_LAPACK
#undef BLAS_FORTRAN_SUFFIX
#define BLAS_FORTRAN_SUFFIX
#define BLAS_SYMBOL_SUFFIX $NEWLAPACK
#endif

#ifndef BLAS_SYMBOL_PREFIX
#define BLAS_SYMBOL_PREFIX
#endif

#ifndef BLAS_SYMBOL_SUFFIX
#define BLAS_SYMBOL_SUFFIX
#endif

// Macros for concatenating and expanding BLAS function names
#define BLAS_FUNC_CONCAT(name,prefix,suffix,suffix2) prefix ## name ## suffix ## suffix2
#define BLAS_FUNC_EXPAND(name,prefix,suffix,suffix2) BLAS_FUNC_CONCAT(name,prefix,suffix,suffix2)

// Macro to generate the actual CBLAS function name
#define CBLAS_FUNC(name) BLAS_FUNC_EXPAND(name,BLAS_SYMBOL_PREFIX,,BLAS_SYMBOL_SUFFIX)

/*
 * Use either the OpenBLAS scheme with the `64_` suffix behind the Fortran
 * compiler symbol mangling, or the MKL scheme (and upcoming
 * reference-lapack#666) which does it the other way around and uses `_64`.
 */
#ifdef OPENBLAS_ILP64_NAMING_SCHEME
#define BLAS_FUNC(name) BLAS_FUNC_EXPAND(name,BLAS_SYMBOL_PREFIX,BLAS_FORTRAN_SUFFIX,BLAS_SYMBOL_SUFFIX)
#else
#define BLAS_FUNC(name) BLAS_FUNC_EXPAND(name,BLAS_SYMBOL_PREFIX,BLAS_SYMBOL_SUFFIX,BLAS_FORTRAN_SUFFIX)
#endif

#ifdef HAVE_BLAS_ILP64
#define CBLAS_INT npy_int64
#define CBLAS_INT_MAX NPY_MAX_INT64
#else
#define CBLAS_INT int
#define CBLAS_INT_MAX INT_MAX
#endif

#define BLASNAME(name) CBLAS_FUNC(name)
#define BLASINT CBLAS_INT

// Include base definitions for CBLAS from npy_cblas_base.h
#include "npy_cblas_base.h"

#undef BLASINT
#undef BLASNAME


/*
 * Convert NumPy stride to BLAS stride. Returns 0 if conversion cannot be done
 * (BLAS won't handle negative or zero strides the way we want).
 */
static inline CBLAS_INT
blas_stride(npy_intp stride, unsigned itemsize)
{
    /*
     * Should probably check pointer alignment also, but this may cause
     * problems if we require complex to be 16 byte aligned.
     */
    if (stride > 0 && (stride % itemsize) == 0) {
        stride /= itemsize;
        if (stride <= CBLAS_INT_MAX) {
            return stride;
        }
    }
    return 0;
}

/*
 * Define a chunksize for CBLAS.
 *
 * The chunksize is the greatest power of two less than CBLAS_INT_MAX.
 */
#if NPY_MAX_INTP > CBLAS_INT_MAX
# define NPY_CBLAS_CHUNK  (CBLAS_INT_MAX / 2 + 1)
#else
`
# define NPY_CBLAS_CHUNK  NPY_MAX_INTP
#endif



#ifdef __cplusplus
}
#endif



#endif  /* _NPY_CBLAS_H_ */


注释：


// 定义 NPY_CBLAS_CHUNK 为 NPY_MAX_INTP（这里假设 NPY_MAX_INTP 是另一个宏或常量），用于某种计算或配置
# define NPY_CBLAS_CHUNK  NPY_MAX_INTP
#endif

#ifdef __cplusplus
// 如果是 C++ 编译环境，则结束 extern "C" 块，确保 C++ 代码可以正确调用 C 语言代码
}
#endif

// 结束 _NPY_CBLAS_H_ 头文件的条件编译，确保头文件内容不会被重复包含
#endif  /* _NPY_CBLAS_H_ */


在这里，以上代码片段是一个头文件的结尾部分，主要用于条件编译和宏定义。这
```