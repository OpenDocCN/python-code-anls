# `.\numpy\numpy\_core\src\common\npy_cblas.h`

```
/*
 * This header provides numpy a consistent interface to CBLAS code. It is needed
 * because not all providers of cblas provide cblas.h. For instance, MKL provides
 * mkl_cblas.h and also typedefs the CBLAS_XXX enums.
 */
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CBLAS_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CBLAS_H_

#include <stddef.h>

/* Allow the use in C++ code.  */
#ifdef __cplusplus
extern "C"
{
#endif

/*
 * Enumerated and derived types
 */

// Define enum for row-major and column-major order
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
// Define enum for transpose operations
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
// Define enum for upper and lower triangular matrices
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
// Define enum for unit and non-unit diagonal matrices
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
// Define enum for left and right side matrices in operations
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

#define CBLAS_INDEX size_t  /* this may vary between platforms */

#ifdef ACCELERATE_NEW_LAPACK
    #if __MAC_OS_X_VERSION_MAX_ALLOWED < 130300
        // Check macOS version compatibility for Accelerate ILP64 support
        #ifdef HAVE_BLAS_ILP64
            #error "Accelerate ILP64 support is only available with macOS 13.3 SDK or later"
        #endif
    #else
        #define NO_APPEND_FORTRAN
        // Define BLAS symbol suffix based on ILP64 support
        #ifdef HAVE_BLAS_ILP64
            #define BLAS_SYMBOL_SUFFIX $NEWLAPACK$ILP64
        #else
            #define BLAS_SYMBOL_SUFFIX $NEWLAPACK
        #endif
    #endif
#endif

#ifdef NO_APPEND_FORTRAN
#define BLAS_FORTRAN_SUFFIX
#else
#define BLAS_FORTRAN_SUFFIX _
#endif

#ifndef BLAS_SYMBOL_PREFIX
#define BLAS_SYMBOL_PREFIX
#endif

#ifndef BLAS_SYMBOL_SUFFIX
#define BLAS_SYMBOL_SUFFIX
#endif

// Concatenate and expand BLAS function names based on symbol conventions
#define BLAS_FUNC_CONCAT(name,prefix,suffix,suffix2) prefix ## name ## suffix ## suffix2
#define BLAS_FUNC_EXPAND(name,prefix,suffix,suffix2) BLAS_FUNC_CONCAT(name,prefix,suffix,suffix2)

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

/*
 * Note that CBLAS doesn't include Fortran compiler symbol mangling, so ends up
 * being the same in both schemes
 */
#define CBLAS_FUNC(name) BLAS_FUNC_EXPAND(name,BLAS_SYMBOL_PREFIX,,BLAS_SYMBOL_SUFFIX)

#ifdef HAVE_BLAS_ILP64
#define CBLAS_INT npy_int64
#define CBLAS_INT_MAX NPY_MAX_INT64
#else
#define CBLAS_INT int
#define CBLAS_INT_MAX INT_MAX
#endif

#define BLASNAME(name) CBLAS_FUNC(name)
#define BLASINT CBLAS_INT

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
    # 如果步长大于0并且步长能整除每个项的字节数（即 itemsize），则执行以下操作
    if (stride > 0 && (stride % itemsize) == 0) {
        # 将步长除以每个项的字节数，以获取项数
        stride /= itemsize;
        # 如果转换后的步长小于等于 CBLAS_INT_MAX（整数类型的最大值）
        if (stride <= CBLAS_INT_MAX) {
            # 返回转换后的步长作为结果
            return stride;
        }
    }
    # 如果上述条件不满足，则返回0
    return 0;
}
/*
 * 定义 CBLAS 的块大小。
 *
 * 块大小是小于 CBLAS_INT_MAX 的最大二的幂。
 */
#if NPY_MAX_INTP > CBLAS_INT_MAX
# define NPY_CBLAS_CHUNK  (CBLAS_INT_MAX / 2 + 1)
#else
# define NPY_CBLAS_CHUNK  NPY_MAX_INTP
#endif

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CBLAS_H_ */
```