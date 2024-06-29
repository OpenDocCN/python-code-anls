# `.\numpy\numpy\linalg\umath_linalg.cpp`

```
/*
 *****************************************************************************
 **                            INCLUDES                                     **
 *****************************************************************************
 */

#define PY_SSIZE_T_CLEAN        // 清除 Python.h 中对 ssize_t 的定义，使用 Py_ssize_t 代替
#include <Python.h>             // 包含 Python C API 头文件

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"  // NumPy 数组对象 API 头文件
#include "numpy/ufuncobject.h"  // NumPy 通用函数对象 API 头文件
#include "numpy/npy_math.h"     // NumPy 数学函数头文件

#include "npy_config.h"         // NumPy 配置文件

#include "npy_cblas.h"          // NumPy 的 CBLAS 头文件

#include <cstddef>              // C++ 标准库头文件，定义了标准的大小类型
#include <cstdio>               // C 标准输入输出头文件
#include <cassert>              // C 断言头文件
#include <cmath>                // C 数学库头文件
#include <type_traits>          // C++ 类型特性头文件，用于类型检查和转换
#include <utility>              // C++ 实用工具头文件，定义了各种实用函数和类

static const char* umath_linalg_version_string = "0.1.5";  // 定义 umath_linalg 的版本字符串

/*
 ****************************************************************************
 *                        Debugging support                                 *
 ****************************************************************************
 */

#define _UMATH_LINALG_DEBUG 0   // 定义调试开关，0 表示关闭调试

#define TRACE_TXT(...) do { fprintf (stderr, __VA_ARGS__); } while (0)  // 定义输出文本调试信息的宏
#define STACK_TRACE do {} while (0)  // 定义空的栈跟踪宏
#define TRACE\
    do {                                        \
        fprintf (stderr,                        \
                 "%s:%d:%s\n",                  \
                 __FILE__,                      \
                 __LINE__,                      \
                 __FUNCTION__);                 \
        STACK_TRACE;                            \
    } while (0)  // 定义输出详细调试信息的宏，包括文件名、行号和函数名

#if _UMATH_LINALG_DEBUG
#if defined HAVE_EXECINFO_H
#include <execinfo.h>           // 如果定义了 HAVE_EXECINFO_H，包含错误信息处理头文件
#elif defined HAVE_LIBUNWIND_H
#include <libunwind.h>         // 如果定义了 HAVE_LIBUNWIND_H，包含 libunwind 头文件
#endif

void
dbg_stack_trace()  // 调试用的栈跟踪函数
{
    void *trace[32];
    size_t size;

    size = backtrace(trace, sizeof(trace)/sizeof(trace[0]));  // 获取当前栈信息
    backtrace_symbols_fd(trace, size, 1);  // 输出栈信息到文件描述符 1（标准错误）
}

#undef STACK_TRACE
#define STACK_TRACE do { dbg_stack_trace(); } while (0)  // 重新定义栈跟踪宏，调用 dbg_stack_trace 函数
#endif

/*
 *****************************************************************************
 *                    BLAS/LAPACK calling macros                             *
 *****************************************************************************
 */

#define FNAME(x) BLAS_FUNC(x)  // 定义用于 BLAS 函数名字转换的宏

typedef CBLAS_INT         fortran_int;  // 将 CBLAS_INT 定义为 fortran_int 类型

typedef struct { float r, i; } f2c_complex;  // 复数结构体定义，单精度
typedef struct { double r, i; } f2c_doublecomplex;  // 复数结构体定义，双精度
/* typedef long int (*L_fp)(); */  // 定义一个 long int 类型的函数指针

typedef float             fortran_real;  // 将 float 定义为 fortran_real 类型
typedef double            fortran_doublereal;  // 将 double 定义为 fortran_doublereal 类型
typedef f2c_complex       fortran_complex;  // 将 f2c_complex 定义为 fortran_complex 类型
typedef f2c_doublecomplex fortran_doublecomplex;  // 将 f2c_doublecomplex 定义为 fortran_doublecomplex 类型

extern "C" fortran_int
FNAME(sgeev)(char *jobvl, char *jobvr, fortran_int *n,
             float a[], fortran_int *lda, float wr[], float wi[],
             float vl[], fortran_int *ldvl, float vr[], fortran_int *ldvr,
             float work[], fortran_int lwork[],
             fortran_int *info);  // 声明 sgeev 函数，用于求解特征值问题

extern "C" fortran_int
FNAME(dgeev)(char *jobvl, char *jobvr, fortran_int *n,
             double a[], fortran_int *lda, double wr[], double wi[],
             double vl[], fortran_int *ldvl, double vr[], fortran_int *ldvr,
             double work[], fortran_int lwork[],
             fortran_int *info);
extern "C" fortran_int
FNAME(cgeev)(char *jobvl, char *jobvr, fortran_int *n,
             f2c_complex a[], fortran_int *lda,
             f2c_complex w[],
             f2c_complex vl[], fortran_int *ldvl,
             f2c_complex vr[], fortran_int *ldvr,
             f2c_complex work[], fortran_int *lwork,
             float rwork[],
             fortran_int *info);
extern "C" fortran_int
FNAME(zgeev)(char *jobvl, char *jobvr, fortran_int *n,
             f2c_doublecomplex a[], fortran_int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], fortran_int *ldvl,
             f2c_doublecomplex vr[], fortran_int *ldvr,
             f2c_doublecomplex work[], fortran_int *lwork,
             double rwork[],
             fortran_int *info);



# 调用 LAPACK 中的 DGEEV 函数，求解实双精度矩阵的特征值和特征向量
FNAME(dgeev)(char *jobvl, char *jobvr, fortran_int *n,
             double a[], fortran_int *lda, double wr[], double wi[],
             double vl[], fortran_int *ldvl, double vr[], fortran_int *ldvr,
             double work[], fortran_int lwork[],
             fortran_int *info);

# 声明一个 C 函数接口，用于调用 LAPACK 中的 CGEEV 函数，求解复单精度矩阵的特征值和特征向量
extern "C" fortran_int
FNAME(cgeev)(char *jobvl, char *jobvr, fortran_int *n,
             f2c_complex a[], fortran_int *lda,
             f2c_complex w[],
             f2c_complex vl[], fortran_int *ldvl,
             f2c_complex vr[], fortran_int *ldvr,
             f2c_complex work[], fortran_int *lwork,
             float rwork[],
             fortran_int *info);

# 声明一个 C 函数接口，用于调用 LAPACK 中的 ZGEEV 函数，求解复双精度矩阵的特征值和特征向量
extern "C" fortran_int
FNAME(zgeev)(char *jobvl, char *jobvr, fortran_int *n,
             f2c_doublecomplex a[], fortran_int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], fortran_int *ldvl,
             f2c_doublecomplex vr[], fortran_int *ldvr,
             f2c_doublecomplex work[], fortran_int *lwork,
             double rwork[],
             fortran_int *info);
FNAME(cgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              f2c_complex a[], fortran_int *lda,
              f2c_complex b[], fortran_int *ldb,
              float s[], float *rcond, fortran_int *rank,
              f2c_complex work[], fortran_int *lwork,
              float rwork[], fortran_int iwork[],
              fortran_int *info);


// 声明一个调用名为 `cgelsd` 的外部函数，该函数用于解决线性最小二乘问题，接受多个参数：
// - m: 矩阵 A 的行数
// - n: 矩阵 A 的列数
// - nrhs: 右侧矩阵 b 的列数
// - a: 复数类型的矩阵 A 的数据数组
// - lda: 矩阵 A 的行跨度
// - b: 复数类型的右侧矩阵 b 的数据数组
// - ldb: 右侧矩阵 b 的行跨度
// - s: 存储奇异值的浮点数数组
// - rcond: 控制奇异值截断的浮点数
// - rank: 返回的有效秩
// - work: 工作区域的复数类型数组
// - lwork: 工作区域的长度
// - rwork: 实数类型的工作区域数组
// - iwork: 整数类型的工作区域数组
// - info: 返回状态信息的整数指针
extern "C" fortran_int
FNAME(zgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex b[], fortran_int *ldb,
              double s[], double *rcond, fortran_int *rank,
              f2c_doublecomplex work[], fortran_int *lwork,
              double rwork[], fortran_int iwork[],
              fortran_int *info);


// 声明一个调用名为 `zgelsd` 的外部函数，功能与 `cgelsd` 类似，但参数中使用了双精度复数和双精度浮点数类型。
// 用于解决复数形式的线性最小二乘问题，接受的参数与 `cgelsd` 相同。
extern "C" fortran_int
FNAME(dgeqrf)(fortran_int *m, fortran_int *n, double a[], fortran_int *lda,
              double tau[], double work[],
              fortran_int *lwork, fortran_int *info);


// 声明一个调用名为 `dgeqrf` 的外部函数，该函数实现双精度实数类型的 QR 分解，接受多个参数：
// - m: 矩阵 A 的行数
// - n: 矩阵 A 的列数
// - a: 矩阵 A 的数据数组
// - lda: 矩阵 A 的行跨度
// - tau: 存储元素的反射系数的数组
// - work: 工作区域的双精度实数数组
// - lwork: 工作区域的长度
// - info: 返回状态信息的整数指针
extern "C" fortran_int
FNAME(zgeqrf)(fortran_int *m, fortran_int *n, f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex tau[], f2c_doublecomplex work[],
              fortran_int *lwork, fortran_int *info);


// 声明一个调用名为 `zgeqrf` 的外部函数，该函数实现双精度复数类型的 QR 分解，接受的参数与 `dgeqrf` 相同。
// 用于解决复数形式的 QR 分解问题。
extern "C" fortran_int
FNAME(dorgqr)(fortran_int *m, fortran_int *n, fortran_int *k, double a[], fortran_int *lda,
              double tau[], double work[],
              fortran_int *lwork, fortran_int *info);


// 声明一个调用名为 `dorgqr` 的外部函数，该函数用于计算双精度实数类型的 QR 分解后的 Q 矩阵，接受多个参数：
// - m: 矩阵 A 的行数
// - n: 矩阵 A 的列数
// - k: 维度 k
// - a: 存储 QR 分解后的矩阵 Q 的数据数组
// - lda: 矩阵 A 的行跨度
// - tau: 存储元素的反射系数的数组
// - work: 工作区域的双精度实数数组
// - lwork: 工作区域的长度
// - info: 返回状态信息的整数指针
extern "C" fortran_int
FNAME(zungqr)(fortran_int *m, fortran_int *n, fortran_int *k, f2c_doublecomplex a[],
              fortran_int *lda, f2c_doublecomplex tau[],
              f2c_doublecomplex work[], fortran_int *lwork, fortran_int *info);


// 声明一个调用名为 `zungqr` 的外部函数，该函数用于计算双精度复数类型的 QR 分解后的 Q 矩阵，接受的参数与 `dorgqr` 相同。
// 用于解决复数形式的 QR 分解问题。
extern "C" fortran_int
FNAME(sgesv)(fortran_int *n, fortran_int *nrhs,
             float a[], fortran_int *lda,
             fortran_int ipiv[],
             float b[], fortran_int *ldb,
             fortran_int *info);


// 声明一个调用名为 `sgesv` 的外部函数，该函数用于解决单精度实数类型的线性方程组，接受多个参数：
// - n: 矩阵 A 的行数和列数
// - nrhs: 右侧矩阵 b 的列数
// - a: 存储系数矩阵 A 的数据数组
// - lda: 矩阵 A 的行跨度
// - ipiv: 存储主元信息的整数数组
// - b: 存储右侧矩阵 b 的数据数组
// - ldb: 右侧矩阵 b 的行跨度
// - info: 返回状态信息的整数指针
extern "C" fortran_int
FNAME(dgesv)(fortran_int *n, fortran_int *nrhs,
             double a[], fortran_int *lda,
             fortran_int ipiv[],
             double b[], fortran_int *ldb,
             fortran_int *info);


// 声明一个调用名为 `dgesv` 的外部函数，该函数用于解决双精度实数类型的线性方程组，接受的参数与 `sgesv` 相同。
// 用于解决实数形式的线性方程组问题。
extern "C" fortran_int
FNAME(cgesv)(fortran_int *n, fortran_int *nrhs,
             f2c_complex a[], fortran_int *lda,
             fortran_int ipiv[],
             f2c_complex b[], fortran_int *ldb,
             fortran_int *info);


// 声明一个调用名为 `cgesv` 的外部函数，该函数用于解决单精度复数类型的线性方程组，接受的参数与 `sgesv` 相同。
// 用于解决复数形式的线性方程组问题。
extern "C" fortran_int
FNAME(zgesv)(fortran_int *n, fortran_int *nrhs,
             f2c_doublecomplex a[], fortran_int *lda,
             fortran_int ipiv[],
             f2c_doublecomplex b[], fortran_int *ldb,
             fortran_int *info);


// 声明一个调用名为 `zgesv` 的外部函数，该函数用
FNAME(cgetrf)(fortran_int *m, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);
extern "C" fortran_int
FNAME(zgetrf)(fortran_int *m, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);


# 调用 LAPACK 中的 cgetrf 函数，对复数矩阵进行 LU 分解
FNAME(cgetrf)(fortran_int *m, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);
# 声明在 C++ 环境下，调用 LAPACK 中的 zgetrf 函数，对双精度复数矩阵进行 LU 分解
extern "C" fortran_int
FNAME(zgetrf)(fortran_int *m, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);



extern "C" fortran_int
FNAME(spotrf)(char *uplo, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(dpotrf)(char *uplo, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(cpotrf)(char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(zpotrf)(char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int *info);


# 声明在 C++ 环境下，调用 LAPACK 中的 spotrf 函数，对实数对称正定矩阵进行 Cholesky 分解
extern "C" fortran_int
FNAME(spotrf)(char *uplo, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int *info);
# 声明在 C++ 环境下，调用 LAPACK 中的 dpotrf 函数，对双精度实数对称正定矩阵进行 Cholesky 分解
extern "C" fortran_int
FNAME(dpotrf)(char *uplo, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int *info);
# 声明在 C++ 环境下，调用 LAPACK 中的 cpotrf 函数，对复数埃尔米特正定矩阵进行 Cholesky 分解
extern "C" fortran_int
FNAME(cpotrf)(char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int *info);
# 声明在 C++ 环境下，调用 LAPACK 中的 zpotrf 函数，对双精度复数埃尔米特正定矩阵进行 Cholesky 分解
extern "C" fortran_int
FNAME(zpotrf)(char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int *info);



extern "C" fortran_int
FNAME(sgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              float a[], fortran_int *lda, float s[], float u[],
              fortran_int *ldu, float vt[], fortran_int *ldvt, float work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *info);
extern "C" fortran_int
FNAME(dgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              double a[], fortran_int *lda, double s[], double u[],
              fortran_int *ldu, double vt[], fortran_int *ldvt, double work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *info);
extern "C" fortran_int
FNAME(cgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              float s[], f2c_complex u[], fortran_int *ldu,
              f2c_complex vt[], fortran_int *ldvt,
              f2c_complex work[], fortran_int *lwork,
              float rwork[], fortran_int iwork[], fortran_int *info);
extern "C" fortran_int
FNAME(zgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              double s[], f2c_doublecomplex u[], fortran_int *ldu,
              f2c_doublecomplex vt[], fortran_int *ldvt,
              f2c_doublecomplex work[], fortran_int *lwork,
              double rwork[], fortran_int iwork[], fortran_int *info);


# 声明在 C++ 环境下，调用 LAPACK 中的 sgesdd 函数，对实数矩阵进行奇异值分解
extern "C" fortran_int
FNAME(sgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              float a[], fortran_int *lda, float s[], float u[],
              fortran_int *ldu, float vt[], fortran_int *ldvt, float work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *info);
# 声明在 C++ 环境下，调用 LAPACK 中的 dgesdd 函数，对双精度实数矩阵进行奇异值分解
extern "C" fortran_int
FNAME(dgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              double a[], fortran_int *lda, double s[], double u[],
              fortran_int *ldu, double vt[], fortran_int *ldvt, double work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *info);
# 声明在 C++ 环境下，调用 LAPACK 中的 cgesdd 函数，对复数矩阵进行奇异值分解
extern "C" fortran_int
FNAME(cgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              float s[], f2c_complex u[], fortran_int *ldu,
              f2c_complex vt[], fortran_int *ldvt,
              f2c_complex work[], fortran_int *lwork,
              float rwork[], fortran_int iwork[], fortran_int *info);
# 声明在 C++ 环境下，调用 LAPACK 中的 zgesdd 函数，对双精度复数矩阵进行奇异值分解
extern "C" fortran_int
FNAME(zgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              double s[], f2c_doublecomplex u[], fortran_int *ldu,
              f2c_doublecomplex vt[], fortran_int *ldvt,
              f2c_doublecomplex work[], fortran_int *lwork,
              double rwork[], fortran_int iwork[], fortran_int *info);



extern "C" fortran_int
FNAME(spotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              float a[], fortran_int *lda,
              float b[], fortran_int *ldb,
              fortran_int *info);
extern "C" fortran_int
FNAME(dpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              double a[], fortran_int *lda,
              double b[], fortran_int *ldb,
              fortran_int *info);
extern "C" fortran_int
FNAME(cpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              f2c_complex a[], fortran_int *lda,
              f2c_complex b[], fortran_int *ldb,
              fortran_int *info);
extern "C" fortran_int


# 声明在 C++ 环境下，调用 LAPACK 中的 spotrs 函数，求解实数对称正
FNAME(zpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex b[], fortran_int *ldb,
              fortran_int *info);
// 声明一个函数 FNAME(zpotrs)，用于求解复数域的线性方程组 A * X = B，其中 A 是复数矩阵，B 是复数向量，返回解 X

extern "C" fortran_int
FNAME(spotri)(char *uplo, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int *info);
// 声明一个 extern "C" 的函数 FNAME(spotri)，用于计算实数域的对称正定矩阵的逆

extern "C" fortran_int
FNAME(dpotri)(char *uplo, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int *info);
// 声明一个 extern "C" 的函数 FNAME(dpotri)，用于计算实数域的对称正定矩阵的逆

extern "C" fortran_int
FNAME(cpotri)(char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int *info);
// 声明一个 extern "C" 的函数 FNAME(cpotri)，用于计算复数域的对称正定矩阵的逆

extern "C" fortran_int
FNAME(zpotri)(char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int *info);
// 声明一个 extern "C" 的函数 FNAME(zpotri)，用于计算复数域的对称正定矩阵的逆

extern "C" fortran_int
FNAME(scopy)(fortran_int *n,
             float *sx, fortran_int *incx,
             float *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(scopy)，用于复制实数数组中的元素到另一个数组

extern "C" fortran_int
FNAME(dcopy)(fortran_int *n,
             double *sx, fortran_int *incx,
             double *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(dcopy)，用于复制双精度数组中的元素到另一个数组

extern "C" fortran_int
FNAME(ccopy)(fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(ccopy)，用于复制复数数组中的元素到另一个数组

extern "C" fortran_int
FNAME(zcopy)(fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(zcopy)，用于复制双精度复数数组中的元素到另一个数组

extern "C" float
FNAME(sdot)(fortran_int *n,
            float *sx, fortran_int *incx,
            float *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(sdot)，用于计算实数数组之间的内积

extern "C" double
FNAME(ddot)(fortran_int *n,
            double *sx, fortran_int *incx,
            double *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(ddot)，用于计算双精度数组之间的内积

extern "C" void
FNAME(cdotu)(f2c_complex *ret, fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(cdotu)，用于计算复数数组之间的未共轭内积

extern "C" void
FNAME(zdotu)(f2c_doublecomplex *ret, fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(zdotu)，用于计算双精度复数数组之间的未共轭内积

extern "C" void
FNAME(cdotc)(f2c_complex *ret, fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(cdotc)，用于计算复数数组之间的共轭内积

extern "C" void
FNAME(zdotc)(f2c_doublecomplex *ret, fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);
// 声明一个 extern "C" 的函数 FNAME(zdotc)，用于计算双精度复数数组之间的共轭内积

extern "C" fortran_int
FNAME(sgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             float *alpha,
             float *a, fortran_int *lda,
             float *b, fortran_int *ldb,
             float *beta,
             float *c, fortran_int *ldc);
// 声明一个 extern "C" 的函数 FNAME(sgemm)，用于进行实数矩阵的乘法运算 C = alpha * A * B + beta * C

extern "C" fortran_int
FNAME(dgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             double *alpha,
             double *a, fortran_int *lda,
             double *b, fortran_int *ldb,
             double *beta,
             double *c, fortran_int *ldc);
// 声明一个 extern "C" 的函数 FNAME(dgemm)，用于进行双精度实数矩阵的乘法运算 C = alpha * A * B + beta * C
FNAME(cgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             f2c_complex *alpha,
             f2c_complex *a, fortran_int *lda,
             f2c_complex *b, fortran_int *ldb,
             f2c_complex *beta,
             f2c_complex *c, fortran_int *ldc);
// 声明一个函数指针 FNAME(cgemm)，接受一系列参数，用于进行复杂的矩阵乘法计算

extern "C" fortran_int
FNAME(zgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             f2c_doublecomplex *alpha,
             f2c_doublecomplex *a, fortran_int *lda,
             f2c_doublecomplex *b, fortran_int *ldb,
             f2c_doublecomplex *beta,
             f2c_doublecomplex *c, fortran_int *ldc);
// 声明一个 extern "C" 的函数 FNAME(zgemm)，用于复杂的双精度复数矩阵乘法计算

#define LAPACK_T(FUNC)                                          \
    TRACE_TXT("Calling LAPACK ( " # FUNC " )\n");               \
    FNAME(FUNC)
// 宏定义 LAPACK_T(FUNC)，用于调用 LAPACK 函数 FUNC，并输出追踪信息到文本

#define BLAS(FUNC)                              \
    FNAME(FUNC)
// 宏定义 BLAS(FUNC)，用于调用 BLAS 函数 FUNC

#define LAPACK(FUNC)                            \
    FNAME(FUNC)
// 宏定义 LAPACK(FUNC)，用于调用 LAPACK 函数 FUNC

/*
 *****************************************************************************
 **                      Some handy functions                               **
 *****************************************************************************
 */

static inline int
get_fp_invalid_and_clear(void)
{
    int status;
    status = npy_clear_floatstatus_barrier((char*)&status);
    return !!(status & NPY_FPE_INVALID);
}
// 定义静态内联函数 get_fp_invalid_and_clear，用于获取并清除浮点数无效标志位的状态

static inline void
set_fp_invalid_or_clear(int error_occurred)
{
    if (error_occurred) {
        npy_set_floatstatus_invalid();
    }
    else {
        npy_clear_floatstatus_barrier((char*)&error_occurred);
    }
}
// 定义静态内联函数 set_fp_invalid_or_clear，根据错误发生状态设置或清除浮点数无效状态

/*
 *****************************************************************************
 **                      Some handy constants                               **
 *****************************************************************************
 */

#define UMATH_LINALG_MODULE_NAME "_umath_linalg"
// 宏定义 UMATH_LINALG_MODULE_NAME，表示数学线性代数模块的名称

template<typename T>
struct numeric_limits;

template<>
struct numeric_limits<float> {
static constexpr float one = 1.0f;
static constexpr float zero = 0.0f;
static constexpr float minus_one = -1.0f;
static const float ninf;
static const float nan;
};
constexpr float numeric_limits<float>::one;
constexpr float numeric_limits<float>::zero;
constexpr float numeric_limits<float>::minus_one;
const float numeric_limits<float>::ninf = -NPY_INFINITYF;
const float numeric_limits<float>::nan = NPY_NANF;
// 定义模板特化结构 numeric_limits<float>，提供 float 类型的常量和特殊值

template<>
struct numeric_limits<double> {
static constexpr double one = 1.0;
static constexpr double zero = 0.0;
static constexpr double minus_one = -1.0;
static const double ninf;
static const double nan;
};
constexpr double numeric_limits<double>::one;
constexpr double numeric_limits<double>::zero;
constexpr double numeric_limits<double>::minus_one;
const double numeric_limits<double>::ninf = -NPY_INFINITY;
const double numeric_limits<double>::nan = NPY_NAN;
// 定义模板特化结构 numeric_limits<double>，提供 double 类型的常量和特殊值

template<>
struct numeric_limits<npy_cfloat> {
static constexpr npy_cfloat one = {1.0f};
static constexpr npy_cfloat zero = {0.0f};
// 定义模板特化结构 numeric_limits<npy_cfloat>，提供 npy_cfloat 类型的常量 one 和 zero
// 定义复数类型的常量，表示负一
static constexpr npy_cfloat minus_one = {-1.0f};

// 定义复数类型的常量，表示负无穷和非数值
static const npy_cfloat ninf;
static const npy_cfloat nan;

// 定义 numeric_limits 结构体的特化版本，为 npy_cfloat 类型
template<>
struct numeric_limits<npy_cfloat> {
    // 定义复数类型的常量，表示一、零、负一
    static constexpr npy_cfloat one = {1.0f, 0.0f};
    static constexpr npy_cfloat zero = {0.0f, 0.0f};
    static constexpr npy_cfloat minus_one = {-1.0f, 0.0f};

    // 定义复数类型的常量，表示负无穷和非数值
    static const npy_cfloat ninf;
    static const npy_cfloat nan;
};

// 初始化 numeric_limits 结构体中 npy_cfloat 类型的负无穷常量
const npy_cfloat numeric_limits<npy_cfloat>::ninf = {-NPY_INFINITYF};
// 初始化 numeric_limits 结构体中 npy_cfloat 类型的非数值常量
const npy_cfloat numeric_limits<npy_cfloat>::nan = {NPY_NANF, NPY_NANF};

// 定义 numeric_limits 结构体的特化版本，为 f2c_complex 类型
template<>
struct numeric_limits<f2c_complex> {
    // 定义复数类型的常量，表示一、零、负一
    static constexpr f2c_complex one = {1.0f, 0.0f};
    static constexpr f2c_complex zero = {0.0f, 0.0f};
    static constexpr f2c_complex minus_one = {-1.0f, 0.0f};

    // 定义复数类型的常量，表示负无穷和非数值
    static const f2c_complex ninf;
    static const f2c_complex nan;
};

// 初始化 numeric_limits 结构体中 f2c_complex 类型的负无穷常量
const f2c_complex numeric_limits<f2c_complex>::ninf = {-NPY_INFINITYF, 0.0f};
// 初始化 numeric_limits 结构体中 f2c_complex 类型的非数值常量
const f2c_complex numeric_limits<f2c_complex>::nan = {NPY_NANF, NPY_NANF};

// 定义 numeric_limits 结构体的特化版本，为 npy_cdouble 类型
template<>
struct numeric_limits<npy_cdouble> {
    // 定义复数类型的常量，表示一、零、负一
    static constexpr npy_cdouble one = {1.0};
    static constexpr npy_cdouble zero = {0.0};
    static constexpr npy_cdouble minus_one = {-1.0};

    // 定义复数类型的常量，表示负无穷和非数值
    static const npy_cdouble ninf;
    static const npy_cdouble nan;
};

// 初始化 numeric_limits 结构体中 npy_cdouble 类型的负无穷常量
const npy_cdouble numeric_limits<npy_cdouble>::ninf = {-NPY_INFINITY};
// 初始化 numeric_limits 结构体中 npy_cdouble 类型的非数值常量
const npy_cdouble numeric_limits<npy_cdouble>::nan = {NPY_NAN, NPY_NAN};

// 定义 numeric_limits 结构体的特化版本，为 f2c_doublecomplex 类型
template<>
struct numeric_limits<f2c_doublecomplex> {
    // 定义复数类型的常量，表示一、零、负一
    static constexpr f2c_doublecomplex one = {1.0};
    static constexpr f2c_doublecomplex zero = {0.0};
    static constexpr f2c_doublecomplex minus_one = {-1.0};

    // 定义复数类型的常量，表示负无穷和非数值
    static const f2c_doublecomplex ninf;
    static const f2c_doublecomplex nan;
};

// 初始化 numeric_limits 结构体中 f2c_doublecomplex 类型的负无穷常量
const f2c_doublecomplex numeric_limits<f2c_doublecomplex>::ninf = {-NPY_INFINITY};
// 初始化 numeric_limits 结构体中 f2c_doublecomplex 类型的非数值常量
const f2c_doublecomplex numeric_limits<f2c_doublecomplex>::nan = {NPY_NAN, NPY_NAN};

/*
 *****************************************************************************
 **               Structs used for data rearrangement                       **
 *****************************************************************************
 */
/*
 * 这个结构体包含了如何在本地缓冲区中线性化矩阵的信息，以便它可以被 BLAS 函数使用。
 * 所有步长都以字节为单位指定，稍后在特定类型的函数中转换为元素。
 *
 * rows: 矩阵中的行数
 * columns: 矩阵中的列数
 * row_strides: 连续行之间的字节数
 * column_strides: 连续列之间的字节数
 * output_lead_dim: BLAS/LAPACK 中的前导维度（以元素为单位）
 */
struct linearize_data
{
  npy_intp rows;            // 矩阵行数
  npy_intp columns;         // 矩阵列数
  npy_intp row_strides;     // 连续行之间的字节数
  npy_intp column_strides;  // 连续列之间的字节数
  npy_intp output_lead_dim; // 输出的前导维度，以元素为单位
};

static inline
linearize_data init_linearize_data_ex(npy_intp rows,
                       npy_intp columns,
                       npy_intp row_strides,
                       npy_intp column_strides,
                       npy_intp output_lead_dim)
{
    return {rows, columns, row_strides, column_strides, output_lead_dim};
}

static inline
linearize_data init_linearize_data(npy_intp rows,
                    npy_intp columns,
                    npy_intp row_strides,
                    npy_intp column_strides)
{
    // 初始化 linearize_data 结构体，使用 columns 作为输出的前导维度
    return init_linearize_data_ex(
        rows, columns, row_strides, column_strides, columns);
}

#if _UMATH_LINALG_DEBUG
static inline void
dump_ufunc_object(PyUFuncObject* ufunc)
{
    TRACE_TXT("\n\n%s '%s' (%d input(s), %d output(s), %d specialization(s).\n",
              ufunc->core_enabled? "generalized ufunc" : "scalar ufunc",
              ufunc->name, ufunc->nin, ufunc->nout, ufunc->ntypes);
    if (ufunc->core_enabled) {
        int arg;
        int dim;
        TRACE_TXT("\t%s (%d dimension(s) detected).\n",
                  ufunc->core_signature, ufunc->core_num_dim_ix);

        for (arg = 0; arg < ufunc->nargs; arg++){
            int * arg_dim_ix = ufunc->core_dim_ixs + ufunc->core_offsets[arg];
            TRACE_TXT("\t\targ %d (%s) has %d dimension(s): (",
                      arg, arg < ufunc->nin? "INPUT" : "OUTPUT",
                      ufunc->core_num_dims[arg]);
            for (dim = 0; dim < ufunc->core_num_dims[arg]; dim ++) {
                TRACE_TXT(" %d", arg_dim_ix[dim]);
            }
            TRACE_TXT(" )\n");
        }
    }
}

static inline void
dump_linearize_data(const char* name, const linearize_data* params)
{
    // 打印 linearize_data 结构体的详细信息，包括名称、行数、列数、行步长和列步长
    TRACE_TXT("\n\t%s rows: %zd columns: %zd"\
              "\n\t\trow_strides: %td column_strides: %td"\
              "\n", name, params->rows, params->columns,
              params->row_strides, params->column_strides);
}

static inline void
print(npy_float s)
{
    // 打印单精度浮点数 s
    TRACE_TXT(" %8.4f", s);
}
static inline void
print(npy_double d)
{
    // 打印双精度浮点数 d
    TRACE_TXT(" %10.6f", d);
}
static inline void
print(npy_cfloat c)
{
    // 打印复数结构 npy_cfloat c 的实部和虚部
    float* c_parts = (float*)&c;
    TRACE_TXT("(%8.4f, %8.4fj)", c_parts[0], c_parts[1]);
}
static inline void
print(npy_cdouble z)
{
    // 打印双精度复数结构 npy_cdouble z 的实部和虚部
    double* z_parts = (double*)&z;
    TRACE_TXT("(%8.4f, %8.4fj)", z_parts[0], z_parts[1]);
}
/*
 *****************************************************************************
 **                            Basics                                       **
 *****************************************************************************
 */

// 定义一个静态内联函数，返回两个fortran_int类型数中较小的一个
static inline fortran_int
fortran_int_min(fortran_int x, fortran_int y) {
    return x < y ? x : y;
}

// 定义一个静态内联函数，返回两个fortran_int类型数中较大的一个
static inline fortran_int
fortran_int_max(fortran_int x, fortran_int y) {
    return x > y ? x : y;
}

// 定义宏INIT_OUTER_LOOP_1，初始化外部循环的第一级
#define INIT_OUTER_LOOP_1 \
    npy_intp dN = *dimensions++;\
    npy_intp N_;\
    npy_intp s0 = *steps++;

// 定义宏INIT_OUTER_LOOP_2，初始化外部循环的第二级，继承INIT_OUTER_LOOP_1
#define INIT_OUTER_LOOP_2 \
    INIT_OUTER_LOOP_1\
    npy_intp s1 = *steps++;

// 定义宏INIT_OUTER_LOOP_3，初始化外部循环的第三级，继承INIT_OUTER_LOOP_2
#define INIT_OUTER_LOOP_3 \
    INIT_OUTER_LOOP_2\
    npy_intp s2 = *steps++;

// 定义宏INIT_OUTER_LOOP_4，初始化外部循环的第四级，继承INIT_OUTER_LOOP_3
#define INIT_OUTER_LOOP_4 \
    INIT_OUTER_LOOP_3\
    npy_intp s3 = *steps++;

// 定义宏INIT_OUTER_LOOP_5，初始化外部循环的第五级，继承INIT_OUTER_LOOP_4
#define INIT_OUTER_LOOP_5 \
    INIT_OUTER_LOOP_4\
    npy_intp s4 = *steps++;

// 定义宏INIT_OUTER_LOOP_6，初始化外部循环的第六级，继承INIT_OUTER_LOOP_5
#define INIT_OUTER_LOOP_6  \
    INIT_OUTER_LOOP_5\
    npy_intp s5 = *steps++;

// 定义宏INIT_OUTER_LOOP_7，初始化外部循环的第七级，继承INIT_OUTER_LOOP_6
#define INIT_OUTER_LOOP_7  \
    INIT_OUTER_LOOP_6\
    npy_intp s6 = *steps++;

// 定义宏BEGIN_OUTER_LOOP_2，开始外部循环的第二级
#define BEGIN_OUTER_LOOP_2 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1) {

// 定义宏BEGIN_OUTER_LOOP_3，开始外部循环的第三级
#define BEGIN_OUTER_LOOP_3 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2) {

// 定义宏BEGIN_OUTER_LOOP_4，开始外部循环的第四级
#define BEGIN_OUTER_LOOP_4 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3) {

// 定义宏BEGIN_OUTER_LOOP_5，开始外部循环的第五级
#define BEGIN_OUTER_LOOP_5 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4) {

// 定义宏BEGIN_OUTER_LOOP_6，开始外部循环的第六级
#define BEGIN_OUTER_LOOP_6 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4,\
             args[5] += s5) {

// 定义宏BEGIN_OUTER_LOOP_7，开始外部循环的第七级
#define BEGIN_OUTER_LOOP_7 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4,\
             args[5] += s5,\
             args[6] += s6) {

// 定义宏END_OUTER_LOOP，结束外部循环
#define END_OUTER_LOOP  }

// 定义静态内联函数，更新指针数组中的指针，使其增加偏移量offsets中对应的值
static inline void
update_pointers(npy_uint8** bases, ptrdiff_t* offsets, size_t count)
{
    size_t i;
    // 遍历指针数组，更新每个指针的偏移量
    for (i = 0; i < count; ++i) {
        bases[i] += offsets[i];
    }
}
/*
 *****************************************************************************
 **                             DISPATCHER FUNCS                            **
 *****************************************************************************
 */

// 定义一个静态函数copy，用于复制浮点数数组（单精度）
static fortran_int copy(fortran_int *n,
        float *sx, fortran_int *incx,
        float *sy, fortran_int *incy) { return FNAME(scopy)(n, sx, incx,
            sy, incy);
}

// 定义一个静态函数copy，用于复制浮点数数组（双精度）
static fortran_int copy(fortran_int *n,
        double *sx, fortran_int *incx,
        double *sy, fortran_int *incy) { return FNAME(dcopy)(n, sx, incx,
            sy, incy);
}

// 定义一个静态函数copy，用于复制复数数组（单精度复数）
static fortran_int copy(fortran_int *n,
        f2c_complex *sx, fortran_int *incx,
        f2c_complex *sy, fortran_int *incy) { return FNAME(ccopy)(n, sx, incx,
            sy, incy);
}

// 定义一个静态函数copy，用于复制复数数组（双精度复数）
static fortran_int copy(fortran_int *n,
        f2c_doublecomplex *sx, fortran_int *incx,
        f2c_doublecomplex *sy, fortran_int *incy) { return FNAME(zcopy)(n, sx, incx,
            sy, incy);
}

// 定义一个静态函数getrf，用于调用 LAPACK 库中的单精度 LU 分解函数
static fortran_int getrf(fortran_int *m, fortran_int *n, float a[], fortran_int
*lda, fortran_int ipiv[], fortran_int *info) {
 return LAPACK(sgetrf)(m, n, a, lda, ipiv, info);
}

// 定义一个静态函数getrf，用于调用 LAPACK 库中的双精度 LU 分解函数
static fortran_int getrf(fortran_int *m, fortran_int *n, double a[], fortran_int
*lda, fortran_int ipiv[], fortran_int *info) {
 return LAPACK(dgetrf)(m, n, a, lda, ipiv, info);
}

// 定义一个静态函数getrf，用于调用 LAPACK 库中的单精度复数 LU 分解函数
static fortran_int getrf(fortran_int *m, fortran_int *n, f2c_complex a[], fortran_int
*lda, fortran_int ipiv[], fortran_int *info) {
 return LAPACK(cgetrf)(m, n, a, lda, ipiv, info);
}

// 定义一个静态函数getrf，用于调用 LAPACK 库中的双精度复数 LU 分解函数
static fortran_int getrf(fortran_int *m, fortran_int *n, f2c_doublecomplex a[], fortran_int
*lda, fortran_int ipiv[], fortran_int *info) {
 return LAPACK(zgetrf)(m, n, a, lda, ipiv, info);
}

/*
 *****************************************************************************
 **                             HELPER FUNCS                                **
 *****************************************************************************
 */

// 定义模板结构fortran_type，用于确定 C 类型对应的 Fortran 类型
template<typename T>
struct fortran_type {
    using type = T;
};

// 部分特化，将numpy单精度复数类型映射到双精度复数类型
template<> struct fortran_type<npy_cfloat> { using type = f2c_complex;};
// 部分特化，将numpy双精度复数类型映射到双精度复数类型
template<> struct fortran_type<npy_cdouble> { using type = f2c_doublecomplex;};
// 使用类型别名，简化使用
template<typename T>
using fortran_type_t = typename fortran_type<T>::type;

// 定义模板结构basetype，用于确定 C 类型对应的基本类型
template<typename T>
struct basetype {
    using type = T;
};
// 部分特化，将numpy单精度复数类型映射到单精度浮点数
template<> struct basetype<npy_cfloat> { using type = npy_float;};
// 部分特化，将numpy双精度复数类型映射到双精度浮点数
template<> struct basetype<npy_cdouble> { using type = npy_double;};
// 部分特化，将单精度复数类型映射到Fortran实数类型
template<> struct basetype<f2c_complex> { using type = fortran_real;};
// 部分特化，将双精度复数类型映射到Fortran双精度实数类型
template<> struct basetype<f2c_doublecomplex> { using type = fortran_doublereal;};
// 使用类型别名，简化使用
template<typename T>
using basetype_t = typename basetype<T>::type;

// 定义标记结构体scalar_trait，表示标量类型
struct scalar_trait {};
// 定义标记结构体complex_trait，表示复数类型
struct complex_trait {};
// 使用模板条件语句，根据类型大小判断是标量还是复数
template<typename typ>
using dispatch_scalar = typename std::conditional<sizeof(basetype_t<typ>) == sizeof(typ), scalar_trait, complex_trait>::type;

             /* rearranging of 2D matrices using blas */

// 定义模板函数dispatch_scalar，用于重新排列二维矩阵，使用BLAS库
template<typename typ>
static inline void *
// 将矩阵展平为一维数组，返回展平后的目标数组指针
linearize_matrix(typ *dst,
                        typ *src,
                        const linearize_data* data)
{
    // 使用模板定义Fortran类型ftyp为typ类型对应的Fortran类型
    using ftyp = fortran_type_t<typ>;
    // 检查目标数组是否非空
    if (dst) {
        int i, j;
        // rv指向目标数组的起始地址
        typ* rv = dst;
        // 将列数转换为Fortran整型
        fortran_int columns = (fortran_int)data->columns;
        // 将列步长转换为Fortran整型（以typ类型为单位）
        fortran_int column_strides =
            (fortran_int)(data->column_strides/sizeof(typ));
        // Fortran中常用的步长为1
        fortran_int one = 1;
        // 遍历矩阵的每一行
        for (i = 0; i < data->rows; i++) {
            // 如果列步长大于0，调用copy函数复制数据
            if (column_strides > 0) {
                copy(&columns,
                              (ftyp*)src, &column_strides,
                              (ftyp*)dst, &one);
            }
            // 如果列步长小于0，从矩阵末尾开始复制数据
            else if (column_strides < 0) {
                copy(&columns,
                              ((ftyp*)src + (columns-1)*column_strides),
                              &column_strides,
                              (ftyp*)dst, &one);
            }
            // 如果列步长为0，手动复制每一列数据
            else {
                /*
                 * 零步长在某些BLAS实现（如OSX Accelerate）中具有未定义的行为，因此手动处理
                 */
                for (j = 0; j < columns; ++j) {
                    memcpy(dst + j, src, sizeof(typ));
                }
            }
            // 更新源数据指针到下一行起始位置
            src += data->row_strides/sizeof(typ);
            // 更新目标数据指针到下一行起始位置
            dst += data->output_lead_dim;
        }
        // 返回目标数组的起始地址
        return rv;
    } else {
        // 如果目标数组为空，直接返回源数据的起始地址
        return src;
    }
}

// 将一维数组反展平为矩阵，返回反展平后的源数组指针
template<typename typ>
static inline void *
delinearize_matrix(typ *dst,
                          typ *src,
                          const linearize_data* data)
{
    // 使用模板定义Fortran类型ftyp为typ类型对应的Fortran类型
    using ftyp = fortran_type_t<typ>;

    // 检查源数组是否非空
    if (src) {
        int i;
        // rv指向源数组的起始地址
        typ *rv = src;
        // 将列数转换为Fortran整型
        fortran_int columns = (fortran_int)data->columns;
        // 将列步长转换为Fortran整型（以typ类型为单位）
        fortran_int column_strides =
            (fortran_int)(data->column_strides/sizeof(typ));
        // Fortran中常用的步长为1
        fortran_int one = 1;
        // 遍历矩阵的每一行
        for (i = 0; i < data->rows; i++) {
            // 如果列步长大于0，调用copy函数反复制数据
            if (column_strides > 0) {
                copy(&columns,
                              (ftyp*)src, &one,
                              (ftyp*)dst, &column_strides);
            }
            // 如果列步长小于0，从矩阵末尾开始反复制数据
            else if (column_strides < 0) {
                copy(&columns,
                              (ftyp*)src, &one,
                              ((ftyp*)dst + (columns-1)*column_strides),
                              &column_strides);
            }
            // 如果列步长为0，手动反复制每一列数据
            else {
                /*
                 * 零步长在某些BLAS实现（如OSX Accelerate）中具有未定义的行为，因此手动处理
                 */
                if (columns > 0) {
                    memcpy(dst,
                           src + (columns-1),
                           sizeof(typ));
                }
            }
            // 更新源数据指针到下一行起始位置
            src += data->output_lead_dim;
            // 更新目标数据指针到下一行起始位置
            dst += data->row_strides/sizeof(typ);
        }

        // 返回源数组的起始地址
        return rv;
    } else {
        // 如果源数组为空，直接返回源数组的起始地址
        return src;
    }
}

template<typename typ>
static inline void
nan_matrix(typ *dst, const linearize_data* data)
{
    # 定义整型变量 i 和 j，用于循环索引
    int i, j;
    # 外层循环，遍历数据结构体 data 的行数
    for (i = 0; i < data->rows; i++) {
        # 指针 cp 指向目标数组 dst 的起始位置
        typ *cp = dst;
        # 计算列步幅 cs，将列步幅除以 typ 类型的大小得到实际步幅
        ptrdiff_t cs = data->column_strides/sizeof(typ);
        # 内层循环，遍历数据结构体 data 的列数
        for (j = 0; j < data->columns; ++j) {
            # 将当前指针 cp 指向的位置设为 typ 类型的 NaN 值
            *cp = numeric_limits<typ>::nan;
            # 将指针 cp 按列步幅 cs 向后移动
            cp += cs;
        }
        # 将目标数组 dst 按行步幅除以 typ 类型的大小向后移动
        dst += data->row_strides/sizeof(typ);
    }
}

template<typename typ>
static inline void
zero_matrix(typ *dst, const linearize_data* data)
{
    int i, j;
    // 遍历矩阵的每一行
    for (i = 0; i < data->rows; i++) {
        typ *cp = dst;
        // 计算列步长（单位为 typ 类型大小）
        ptrdiff_t cs = data->column_strides/sizeof(typ);
        // 遍历矩阵的每一列
        for (j = 0; j < data->columns; ++j) {
            // 将目标矩阵位置清零
            *cp = numeric_limits<typ>::zero;
            cp += cs; // 移动到下一列的位置
        }
        dst += data->row_strides/sizeof(typ); // 移动到下一行的位置
    }
}

               /* identity square matrix generation */
template<typename typ>
static inline void
identity_matrix(typ *matrix, size_t n)
{
    size_t i;
    // 将矩阵初始化为 n×n 的零矩阵
    memset((void *)matrix, 0, n*n*sizeof(typ));

    // 遍历矩阵的对角线元素
    for (i = 0; i < n; ++i)
    {
        // 设置对角线上的元素为 1
        *matrix = numeric_limits<typ>::one;
        matrix += n+1; // 移动到下一个对角线元素的位置
    }
}

/* -------------------------------------------------------------------------- */
                          /* Determinants */

// 返回单精度浮点数的自然对数
static npy_float npylog(npy_float f) { return npy_logf(f);}
// 返回双精度浮点数的自然对数
static npy_double npylog(npy_double d) { return npy_log(d);}
// 返回单精度浮点数的自然指数
static npy_float npyexp(npy_float f) { return npy_expf(f);}
// 返回双精度浮点数的自然指数
static npy_double npyexp(npy_double d) { return npy_exp(d);}

template<typename typ>
static inline void
slogdet_from_factored_diagonal(typ* src,
                                      fortran_int m,
                                      typ *sign,
                                      typ *logdet)
{
    typ acc_sign = *sign;
    typ acc_logdet = numeric_limits<typ>::zero;
    int i;
    // 遍历对角线因子的数组
    for (i = 0; i < m; i++) {
        typ abs_element = *src;
        // 如果元素小于零，调整符号并取绝对值
        if (abs_element < numeric_limits<typ>::zero) {
            acc_sign = -acc_sign;
            abs_element = -abs_element;
        }

        // 累加元素的自然对数
        acc_logdet += npylog(abs_element);
        src += m+1; // 移动到下一个对角线元素的位置
    }

    // 存储计算出的行列式的符号和对数绝对值
    *sign = acc_sign;
    *logdet = acc_logdet;
}

template<typename typ>
static inline typ
det_from_slogdet(typ sign, typ logdet)
{
    // 计算行列式的值
    typ result = sign * npyexp(logdet);
    return result;
}


npy_float npyabs(npy_cfloat z) { return npy_cabsf(z);}
npy_double npyabs(npy_cdouble z) { return npy_cabs(z);}

// 返回复数的实部（单精度浮点数）
inline float RE(npy_cfloat *c) { return npy_crealf(*c); }
// 返回复数的实部（双精度浮点数）
inline double RE(npy_cdouble *c) { return npy_creal(*c); }
#if NPY_SIZEOF_COMPLEX_LONGDOUBLE != NPY_SIZEOF_COMPLEX_DOUBLE
// 返回复数的实部（长双精度浮点数）
inline longdouble_t RE(npy_clongdouble *c) { return npy_creall(*c); }
#endif

// 返回复数的虚部（单精度浮点数）
inline float IM(npy_cfloat *c) { return npy_cimagf(*c); }
// 返回复数的虚部（双精度浮点数）
inline double IM(npy_cdouble *c) { return npy_cimag(*c); }
#if NPY_SIZEOF_COMPLEX_LONGDOUBLE != NPY_SIZEOF_COMPLEX_DOUBLE
// 返回复数的虚部（长双精度浮点数）
inline longdouble_t IM(npy_clongdouble *c) { return npy_cimagl(*c); }
#endif

// 设置复数的实部（单精度浮点数）
inline void SETRE(npy_cfloat *c, float real) { npy_csetrealf(c, real); }
// 设置复数的实部（双精度浮点数）
inline void SETRE(npy_cdouble *c, double real) { npy_csetreal(c, real); }
#if NPY_SIZEOF_COMPLEX_LONGDOUBLE != NPY_SIZEOF_COMPLEX_DOUBLE
// 设置复数的实部（长双精度浮点数）
inline void SETRE(npy_clongdouble *c, double real) { npy_csetreall(c, real); }
#endif

// 设置复数的虚部（单精度浮点数）
inline void SETIM(npy_cfloat *c, float real) { npy_csetimagf(c, real); }
/* 定义一个宏 SETIM，用于设置复数结构中的虚部 */
inline void SETIM(npy_cdouble *c, double real) { npy_csetimag(c, real); }
#if NPY_SIZEOF_COMPLEX_LONGDOUBLE != NPY_SIZEOF_COMPLEX_DOUBLE
/* 如果长双精度复数和双精度复数大小不同，定义另一个 SETIM 宏 */
inline void SETIM(npy_clongdouble *c, double real) { npy_csetimagl(c, real); }
#endif

/* 定义一个模板函数 mult，用于复数乘法运算 */
template<typename typ>
static inline typ
mult(typ op1, typ op2)
{
    typ rv;  // 定义结果变量

    // 计算复数乘法的实部和虚部
    SETRE(&rv, RE(&op1)*RE(&op2) - IM(&op1)*IM(&op2));
    SETIM(&rv, RE(&op1)*IM(&op2) + IM(&op1)*RE(&op2));

    return rv;  // 返回乘积结果
}

/* 定义一个模板函数 slogdet_from_factored_diagonal，用于计算对角线因式分解的 slogdet */
template<typename typ, typename basetyp>
static inline void
slogdet_from_factored_diagonal(typ* src,
                                      fortran_int m,
                                      typ *sign,
                                      basetyp *logdet)
{
    int i;
    typ sign_acc = *sign;  // 初始化累计的符号
    basetyp logdet_acc = numeric_limits<basetyp>::zero;  // 初始化累计的对数行列式值

    for (i = 0; i < m; i++)
    {
        basetyp abs_element = npyabs(*src);  // 计算当前元素的绝对值
        typ sign_element;
        // 计算当前元素的复数符号
        SETRE(&sign_element, RE(src) / abs_element);
        SETIM(&sign_element, IM(src) / abs_element);

        sign_acc = mult(sign_acc, sign_element);  // 累乘符号元素
        logdet_acc += npylog(abs_element);  // 累加对数行列式值
        src += m + 1;  // 移动到下一个对角线元素
    }

    *sign = sign_acc;  // 更新总符号
    *logdet = logdet_acc;  // 更新总对数行列式值
}

/* 定义一个模板函数 det_from_slogdet，用于根据 slogdet 计算行列式值 */
template<typename typ, typename basetyp>
static inline typ
det_from_slogdet(typ sign, basetyp logdet)
{
    typ tmp;
    SETRE(&tmp, npyexp(logdet));  // 计算指数化的对数行列式值的实部
    SETIM(&tmp, numeric_limits<basetyp>::zero);  // 设置虚部为零
    return mult(sign, tmp);  // 返回最终行列式值
}


/* 在 linalg 包中，使用 LAPACK 计算 LU 分解得到行列式 */
template<typename typ, typename basetyp>
static inline void
slogdet_single_element(fortran_int m,
                              typ* src,
                              fortran_int* pivots,
                              typ *sign,
                              basetyp *logdet)
{
using ftyp = fortran_type_t<typ>;  // 定义 Fortran 类型别名

    fortran_int info = 0;
    fortran_int lda = fortran_int_max(m, 1);  // 计算 lda，确保大于等于 m 的最小整数
    int i;
    /* 注意：原地操作 */

    getrf(&m, &m, (ftyp*)src, &lda, pivots, &info);  // 在 src 上进行 LU 分解

    if (info == 0) {
        int change_sign = 0;
        /* 注意：Fortran 使用基于 1 的索引 */

        // 计算置换过程中改变符号的次数
        for (i = 0; i < m; i++)
        {
            change_sign += (pivots[i] != (i+1));
        }

        // 根据符号改变次数设置整体符号
        *sign = (change_sign % 2)?numeric_limits<typ>::minus_one:numeric_limits<typ>::one;

        // 计算对角线因式分解的 slogdet
        slogdet_from_factored_diagonal(src, m, sign, logdet);
    } else {
        /*
          如果 getrf 失败，使用 0 作为符号和 -inf 作为 logdet
        */
        *sign = numeric_limits<typ>::zero;
        *logdet = numeric_limits<basetyp>::ninf;
    }
}

/* 定义 slogdet 函数，计算行列式的符号和对数行列式值 */
template<typename typ, typename basetyp>
static void
slogdet(char **args,
               npy_intp const *dimensions,
               npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    fortran_int m;
    char *tmp_buff = NULL;
    size_t matrix_size;
    size_t pivot_size;
    size_t safe_m;
    /* notes:
     *   matrix will need to be copied always, as factorization in lapack is
     *          made inplace
     *   matrix will need to be in column-major order, as expected by lapack
     *          code (fortran)
     *   always a square matrix
     *   need to allocate memory for both, matrix_buffer and pivot buffer
     */
    /* 初始化外部循环 3 */
    INIT_OUTER_LOOP_3
    /* 将 dimensions[0] 赋给 m，并确保 m 是 size_t 类型 */
    m = (fortran_int) dimensions[0];
    /* 避免空的 malloc（缓冲区可能未使用），并确保 safe_m 是非零的 */
    safe_m = m != 0 ? m : 1;
    /* 计算矩阵和 pivot 缓冲区的总大小 */
    matrix_size = safe_m * safe_m * sizeof(typ);
    pivot_size = safe_m * sizeof(fortran_int);
    /* 分配足够的内存给 tmp_buff，包括矩阵和 pivot 缓冲区 */
    tmp_buff = (char *)malloc(matrix_size + pivot_size);

    if (tmp_buff) {
        /* 为了按照 FORTRAN 的顺序获取矩阵，交换了这些步骤 */
        /* 初始化线性化数据结构 */
        linearize_data lin_data = init_linearize_data(m, m, steps[1], steps[0]);
        /* 开始外部循环 3 */
        BEGIN_OUTER_LOOP_3
            /* 将 args[0] 的数据线性化为 tmp_buff */
            linearize_matrix((typ*)tmp_buff, (typ*)args[0], &lin_data);
            /* 计算矩阵的 slogdet（行列式的对数） */
            slogdet_single_element(m,
                                   (typ*)tmp_buff,
                                   (fortran_int*)(tmp_buff+matrix_size),
                                   (typ*)args[1],
                                   (basetyp*)args[2]);
        /* 结束外部循环 */
        END_OUTER_LOOP

        /* 释放 tmp_buff 的内存 */
        free(tmp_buff);
    }
    else {
        /* 如果分配内存失败，则设置内存错误并禁用 C API */
        /* TODO: 需要使用新的 ufunc API 来指示错误返回 */
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
}

template<typename typ, typename basetyp>
static void
det(char **args,
    npy_intp const *dimensions,
    npy_intp const *steps,
    void *NPY_UNUSED(func))
{
    fortran_int m;
    char *tmp_buff;
    size_t matrix_size;
    size_t pivot_size;
    size_t safe_m;
    /* notes:
     *   matrix will need to be copied always, as factorization in lapack is
     *       made inplace
     *   matrix will need to be in column-major order, as expected by lapack
     *       code (fortran)
     *   always a square matrix
     *   need to allocate memory for both, matrix_buffer and pivot buffer
     */
    INIT_OUTER_LOOP_2
    // 从dimensions数组中获取矩阵的维度并转换为fortran_int类型
    m = (fortran_int) dimensions[0];
    /* 避免空的malloc（缓冲区可能未使用），确保m是`size_t` */
    safe_m = m != 0 ? m : 1;
    // 计算需要分配的内存空间大小，用于存储矩阵数据和主元数据
    matrix_size = safe_m * safe_m * sizeof(typ);
    pivot_size = safe_m * sizeof(fortran_int);
    // 分配内存空间，用于临时缓冲区，存储矩阵和主元数据
    tmp_buff = (char *)malloc(matrix_size + pivot_size);

    if (tmp_buff) {
        /* swapped steps to get matrix in FORTRAN order */
        // 初始化线性化数据结构，用于将数据转换为FORTRAN顺序的矩阵
        linearize_data lin_data = init_linearize_data(m, m, steps[1], steps[0]);

        typ sign;
        basetyp logdet;

        BEGIN_OUTER_LOOP_2
            // 线性化输入数据，使其符合FORTRAN顺序
            linearize_matrix((typ*)tmp_buff, (typ*)args[0], &lin_data);
            // 计算矩阵的行列式的符号和对数行列式
            slogdet_single_element(m,
                                   (typ*)tmp_buff,
                                   (fortran_int*)(tmp_buff + matrix_size),
                                   &sign,
                                   &logdet);
            // 将计算得到的行列式结果存储在输出参数中
            *(typ *)args[1] = det_from_slogdet(sign, logdet);
        END_OUTER_LOOP

        free(tmp_buff);
    }
    else {
        /* TODO: Requires use of new ufunc API to indicate error return */
        // 如果内存分配失败，则发出内存错误异常
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
}


/* -------------------------------------------------------------------------- */
                          /* Eigh family */

template<typename typ>
struct EIGH_PARAMS_t {
    typ *A;     /* matrix */
    basetype_t<typ> *W;     /* eigenvalue vector */
    typ *WORK;  /* main work buffer */
    basetype_t<typ> *RWORK; /* secondary work buffer (for complex versions) */
    fortran_int *IWORK;
    fortran_int N;
    fortran_int LWORK;
    fortran_int LRWORK;
    fortran_int LIWORK;
    char JOBZ;
    char UPLO;
    fortran_int LDA;
} ;

static inline fortran_int
call_evd(EIGH_PARAMS_t<npy_float> *params)
{
    fortran_int rv;
    // 调用LAPACK库中的ssyevd函数进行对称矩阵特征值分解
    LAPACK(ssyevd)(&params->JOBZ, &params->UPLO, &params->N,
                   params->A, &params->LDA, params->W,
                   params->WORK, &params->LWORK,
                   params->IWORK, &params->LIWORK,
                   &rv);
    return rv;
}
static inline fortran_int
call_evd(EIGH_PARAMS_t<npy_double> *params)
{
    fortran_int rv;
    // 调用LAPACK库中的dsyevd函数进行对称矩阵特征值分解
    LAPACK(dsyevd)(&params->JOBZ, &params->UPLO, &params->N,
                   params->A, &params->LDA, params->W,
                   params->WORK, &params->LWORK,
                   params->IWORK, &params->LIWORK,
                   &rv);
    return rv;
}
    // 调用 LAPACK 库中的 dsyevd 函数来计算对称矩阵的特征值和（可选）特征向量
    // 参数说明：
    // - params->JOBZ: 用于指定是否计算特征向量的标志
    // - params->UPLO: 指定矩阵存储的上/下三角部分
    // - params->N: 矩阵的阶数（维度）
    // - params->A: 指向存储矩阵数据的数组的指针
    // - &params->LDA: 指定矩阵 A 的第一个维度（列数）
    // - params->W: 用于存储计算得到的特征值的数组
    // - params->WORK: 提供给 LAPACK 函数用于工作空间的数组
    // - &params->LWORK: 指定工作空间数组的长度
    // - params->IWORK: 提供给 LAPACK 函数用于整数工作空间的数组
    // - &params->LIWORK: 指定整数工作空间数组的长度
    // - &rv: 返回值，指示函数调用的执行情况（通常是一个整数）

    // 返回 rv 变量的值作为函数的返回值
    return rv;
/*
 * Initialize the parameters to use in for the lapack function _syevd
 * Handles buffer allocation
 */
template<typename typ>
static inline int
init_evd(EIGH_PARAMS_t<typ>* params, char JOBZ, char UPLO,
                   fortran_int N, scalar_trait)
{
    npy_uint8 *mem_buff = NULL;  // 声明指向无符号字节的指针，用于内存缓冲区
    npy_uint8 *mem_buff2 = NULL;  // 声明指向无符号字节的指针，用于第二个内存缓冲区
    fortran_int lwork;  // 定义用于工作区大小的变量
    fortran_int liwork;  // 定义用于整数工作区大小的变量
    npy_uint8 *a, *w, *work, *iwork;  // 声明指向无符号字节的指针，用于矩阵A，特征值W，工作区WORK和整数工作区IWORK
    size_t safe_N = N;  // 安全的N值，用于分配内存大小
    size_t alloc_size = safe_N * (safe_N + 1) * sizeof(typ);  // 计算分配的内存大小
    fortran_int lda = fortran_int_max(N, 1);  // 计算LDA，确保大于等于N

    mem_buff = (npy_uint8 *)malloc(alloc_size);  // 分配内存缓冲区

    if (!mem_buff) {  // 检查内存分配是否成功
        goto error;  // 如果分配失败，跳转到错误处理
    }
    a = mem_buff;  // 设置矩阵A的指针
    w = mem_buff + safe_N * safe_N * sizeof(typ);  // 设置特征值W的指针

    params->A = (typ*)a;  // 将分配的内存地址赋给参数结构体中的矩阵A
    params->W = (typ*)w;  // 将分配的内存地址赋给参数结构体中的特征值W
    params->RWORK = NULL; /* unused */  // 参数结构体中的RWORK未使用
    params->N = N;  // 设置参数结构体中的N
    params->LRWORK = 0; /* unused */  // 参数结构体中的LRWORK未使用
    params->JOBZ = JOBZ;  // 设置参数结构体中的JOBZ
    params->UPLO = UPLO;  // 设置参数结构体中的UPLO
    params->LDA = lda;  // 设置参数结构体中的LDA

    /* Work size query */
    {
        typ query_work_size;  // 用于存储工作区大小的变量
        fortran_int query_iwork_size;  // 用于存储整数工作区大小的变量

        params->LWORK = -1;  // 设置LWORK为-1进行工作区大小查询
        params->LIWORK = -1;  // 设置LIWORK为-1进行整数工作区大小查询
        params->WORK = &query_work_size;  // 设置WORK指向query_work_size变量的地址
        params->IWORK = &query_iwork_size;  // 设置IWORK指向query_iwork_size变量的地址

        if (call_evd(params) != 0) {  // 调用evd函数进行工作区大小查询，并检查返回值
            goto error;  // 如果查询失败，跳转到错误处理
        }

        lwork = (fortran_int)query_work_size;  // 将查询得到的工作区大小赋给lwork变量
        liwork = query_iwork_size;  // 将查询得到的整数工作区大小赋给liwork变量
    }

    mem_buff2 = (npy_uint8 *)malloc(lwork*sizeof(typ) + liwork*sizeof(fortran_int));  // 根据查询得到的工作区大小和整数工作区大小分配第二个内存缓冲区
    if (!mem_buff2) {  // 检查第二个内存分配是否成功
        goto error;  // 如果分配失败，跳转到错误处理
    }

    work = mem_buff2;  // 设置工作区WORK的指针
    iwork = mem_buff2 + lwork*sizeof(typ);  // 设置整数工作区IWORK的指针

    params->LWORK = lwork;  // 设置参数结构体中的LWORK
    params->WORK = (typ*)work;  // 将分配的工作区内存地址赋给参数结构体中的WORK
    params->LIWORK = liwork;  // 设置参数结构体中的LIWORK
    params->IWORK = (fortran_int*)iwork;  // 将分配的整数工作区内存地址赋给参数结构体中的IWORK

    return 1;  // 初始化成功，返回1

 error:
    /* something failed */
    memset(params, 0, sizeof(*params));  // 初始化参数结构体为0
    free(mem_buff2);  // 释放第二个内存缓冲区
    free(mem_buff);  // 释放第一个内存缓冲区

    return 0;  // 返回初始化失败
}


static inline fortran_int
call_evd(EIGH_PARAMS_t<npy_cfloat> *params)
{
    fortran_int rv;  // 定义返回值变量
    LAPACK(cheevd)(&params->JOBZ, &params->UPLO, &params->N,
                          (fortran_type_t<npy_cfloat>*)params->A, &params->LDA, params->W,
                          (fortran_type_t<npy_cfloat>*)params->WORK, &params->LWORK,
                          params->RWORK, &params->LRWORK,
                          params->IWORK, &params->LIWORK,
                          &rv);  // 调用LAPACK库中的cheevd函数进行复数浮点数的特征值计算
    return rv;  // 返回函数调用的结果值
}

static inline fortran_int
call_evd(EIGH_PARAMS_t<npy_cdouble> *params)
{
    fortran_int rv;  // 定义返回值变量
    LAPACK(zheevd)(&params->JOBZ, &params->UPLO, &params->N,
                          (fortran_type_t<npy_cdouble>*)params->A, &params->LDA, params->W,
                          (fortran_type_t<npy_cdouble>*)params->WORK, &params->LWORK,
                          params->RWORK, &params->LRWORK,
                          params->IWORK, &params->LIWORK,
                          &rv);  // 调用LAPACK库中的zheevd函数进行复数双精度浮点数的特征值计算
    return rv;  // 返回函数调用的结果值
}

template<typename typ>
static inline int
init_evd(EIGH_PARAMS_t<typ> *params,
                   char JOBZ,
                   char UPLO,
                   fortran_int N, complex_trait)
{
    # 使用 `basetype_t` 模板生成 `typ` 类型的基本类型
    using basetyp = basetype_t<typ>;
using ftyp = fortran_type_t<typ>;
using fbasetyp = fortran_type_t<basetyp>;
// 定义类型别名，ftyp代表typ的Fortran类型，fbasetyp代表basetyp的Fortran类型

npy_uint8 *mem_buff = NULL;
npy_uint8 *mem_buff2 = NULL;
fortran_int lwork;
fortran_int lrwork;
fortran_int liwork;
// 声明指针和整型变量

npy_uint8 *a, *w, *work, *rwork, *iwork;
// 声明指针变量用于存储内存分配后的位置

size_t safe_N = N;
fortran_int lda = fortran_int_max(N, 1);
// 初始化safe_N为N，计算lda作为N和1中较大的值

mem_buff = (npy_uint8 *)malloc(safe_N * safe_N * sizeof(typ) +
                  safe_N * sizeof(basetyp));
// 分配内存给mem_buff，大小为safe_N * safe_N * sizeof(typ) + safe_N * sizeof(basetyp)
if (!mem_buff) {
    goto error;
}
// 如果内存分配失败，跳转到error标签

a = mem_buff;
w = mem_buff + safe_N * safe_N * sizeof(typ);
// 设置a和w指向内存分配后的位置

params->A = (typ*)a;
params->W = (basetyp*)w;
params->N = N;
params->JOBZ = JOBZ;
params->UPLO = UPLO;
params->LDA = lda;
// 设置params结构体的成员变量为分配后的内存地址和其他参数值

/* Work size query */
{
    ftyp query_work_size;
    fbasetyp query_rwork_size;
    fortran_int query_iwork_size;

    params->LWORK = -1;
    params->LRWORK = -1;
    params->LIWORK = -1;
    params->WORK = (typ*)&query_work_size;
    params->RWORK = (basetyp*)&query_rwork_size;
    params->IWORK = &query_iwork_size;

    if (call_evd(params) != 0) {
        goto error;
    }

    lwork = (fortran_int)*(fbasetyp*)&query_work_size;
    lrwork = (fortran_int)query_rwork_size;
    liwork = query_iwork_size;
}
// 执行工作大小查询，并将结果存储在query_work_size、query_rwork_size和query_iwork_size中

mem_buff2 = (npy_uint8 *)malloc(lwork*sizeof(typ) +
                   lrwork*sizeof(basetyp) +
                   liwork*sizeof(fortran_int));
// 分配内存给mem_buff2，大小为lwork*sizeof(typ) + lrwork*sizeof(basetyp) + liwork*sizeof(fortran_int)
if (!mem_buff2) {
    goto error;
}
// 如果内存分配失败，跳转到error标签

work = mem_buff2;
rwork = work + lwork*sizeof(typ);
iwork = rwork + lrwork*sizeof(basetyp);
// 设置work、rwork和iwork指向内存分配后的位置

params->WORK = (typ*)work;
params->RWORK = (basetyp*)rwork;
params->IWORK = (fortran_int*)iwork;
params->LWORK = lwork;
params->LRWORK = lrwork;
params->LIWORK = liwork;
// 更新params结构体的成员变量为新分配的内存地址和大小

return 1;

/* something failed */
error:
memset(params, 0, sizeof(*params));
free(mem_buff2);
free(mem_buff);
// 在出错时释放内存和清零params结构体

return 0;
}

/*
 * (M, M)->(M,)(M, M)
 * dimensions[1] -> M
 * args[0] -> A[in]
 * args[1] -> W
 * args[2] -> A[out]
 */

template<typename typ>
static inline void
release_evd(EIGH_PARAMS_t<typ> *params)
{
    /* allocated memory in A and WORK */
    free(params->A);
    free(params->WORK);
    memset(params, 0, sizeof(*params));
}
// 释放params结构体中A和WORK成员变量所分配的内存，并清零params结构体本身

template<typename typ>
static inline void
eigh_wrapper(char JOBZ,
                char UPLO,
                char**args,
                npy_intp const *dimensions,
                npy_intp const *steps)
{
    using basetyp = basetype_t<typ>;
    ptrdiff_t outer_steps[3];
    size_t iter;
    size_t outer_dim = *dimensions++;
    size_t op_count = (JOBZ=='N')?2:3;
    EIGH_PARAMS_t<typ> eigh_params;
    int error_occurred = get_fp_invalid_and_clear();

    for (iter = 0; iter < op_count; ++iter) {
        outer_steps[iter] = (ptrdiff_t) steps[iter];
    }
    steps += op_count;
    // 初始化函数参数和局部变量
    # 如果初始化成功，则执行以下代码块，处理特征值分解
    if (init_evd(&eigh_params,
                           JOBZ,
                           UPLO,
                           (fortran_int)dimensions[0], dispatch_scalar<typ>())) {
        # 初始化将要处理的输入矩阵数据的线性化描述
        linearize_data matrix_in_ld = init_linearize_data(eigh_params.N, eigh_params.N, steps[1], steps[0]);
        # 初始化存储特征值输出数据的线性化描述
        linearize_data eigenvalues_out_ld = init_linearize_data(1, eigh_params.N, 0, steps[2]);
        # 初始化存储特征向量输出数据的线性化描述，用于消除未初始化的警告
        linearize_data eigenvectors_out_ld  = {}; /* silence uninitialized warning */
        # 如果任务要求计算特征向量，则初始化存储特征向量输出数据的线性化描述
        if ('V' == eigh_params.JOBZ) {
            eigenvectors_out_ld = init_linearize_data(eigh_params.N, eigh_params.N, steps[4], steps[3]);
        }

        # 外部循环，迭代处理每个矩阵或向量的操作
        for (iter = 0; iter < outer_dim; ++iter) {
            int not_ok;
            # 将输入矩阵数据复制到线性化描述中
            linearize_matrix((typ*)eigh_params.A, (typ*)args[0], &matrix_in_ld);
            # 调用特征值分解函数，检查是否成功
            not_ok = call_evd(&eigh_params);
            if (!not_ok) {
                # 如果特征值分解成功，则将结果复制输出
                delinearize_matrix((basetyp*)args[1],
                                              (basetyp*)eigh_params.W,
                                              &eigenvalues_out_ld);

                # 如果任务要求计算特征向量，则将特征向量复制输出
                if ('V' == eigh_params.JOBZ) {
                    delinearize_matrix((typ*)args[2],
                                              (typ*)eigh_params.A,
                                              &eigenvectors_out_ld);
                }
            } else {
                # 如果特征值分解失败，则将输出结果设置为 NaN
                error_occurred = 1;
                nan_matrix((basetyp*)args[1], &eigenvalues_out_ld);
                if ('V' == eigh_params.JOBZ) {
                    nan_matrix((typ*)args[2], &eigenvectors_out_ld);
                }
            }
            # 更新指针，准备处理下一个矩阵或向量
            update_pointers((npy_uint8**)args, outer_steps, op_count);
        }

        # 释放特征值分解相关的内存资源
        release_evd(&eigh_params);
    }

    # 在特征值分解完成后，根据错误标志设置浮点数无效或清除状态
    set_fp_invalid_or_clear(error_occurred);
/* -------------------------------------------------------------------------- */
/* Solve family (includes inv) */

template<typename typ>
struct GESV_PARAMS_t
{
    typ *A; /* A is (N, N) of base type */
    typ *B; /* B is (N, NRHS) of base type */
    fortran_int * IPIV; /* IPIV is (N) */

    fortran_int N; /* Number of rows and columns in matrix A */
    fortran_int NRHS; /* Number of right-hand side vectors in matrix B */
    fortran_int LDA; /* Leading dimension of matrix A */
    fortran_int LDB; /* Leading dimension of matrix B */
};

/* Calls LAPACK function sgesv for single precision real numbers */
static inline fortran_int
call_gesv(GESV_PARAMS_t<fortran_real> *params)
{
    fortran_int rv;
    LAPACK(sgesv)(&params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->IPIV,
                          params->B, &params->LDB,
                          &rv);
    return rv;
}

/* Calls LAPACK function dgesv for double precision real numbers */
static inline fortran_int
call_gesv(GESV_PARAMS_t<fortran_doublereal> *params)
{
    fortran_int rv;
    LAPACK(dgesv)(&params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->IPIV,
                          params->B, &params->LDB,
                          &rv);
    return rv;
}

/* Calls LAPACK function cgesv for single precision complex numbers */
static inline fortran_int
call_gesv(GESV_PARAMS_t<fortran_complex> *params)
{
    fortran_int rv;
    LAPACK(cgesv)(&params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->IPIV,
                          params->B, &params->LDB,
                          &rv);
    return rv;
}

/* Calls LAPACK function zgesv for double precision complex numbers */
static inline fortran_int
call_gesv(GESV_PARAMS_t<fortran_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zgesv)(&params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->IPIV,
                          params->B, &params->LDB,
                          &rv);
    return rv;
}

/*
 * Initialize the parameters to use in for the LAPACK function _heev
 * Handles buffer allocation
 */
template<typename ftyp>
static inline int
init_gesv(GESV_PARAMS_t<ftyp> *params, fortran_int N, fortran_int NRHS)
{
    npy_uint8 *mem_buff = NULL;
    # 声明指向内存区域的指针变量 a, b, ipiv，分别用于存储矩阵 A、向量 B 和整型数组 IPIV 的起始地址
    npy_uint8 *a, *b, *ipiv;
    
    # 定义安全的矩阵和向量维度 safe_N 和 safe_NRHS，确保它们不小于 N 和 NRHS
    size_t safe_N = N;
    size_t safe_NRHS = NRHS;
    
    # 计算矩阵 A 的列数 ld，至少为 N 和 1 中的最大值
    fortran_int ld = fortran_int_max(N, 1);
    
    # 分配内存并初始化内存缓冲区 mem_buff，用于存储矩阵 A、向量 B 和整型数组 IPIV 的数据
    mem_buff = (npy_uint8 *)malloc(safe_N * safe_N * sizeof(ftyp) +
                      safe_N * safe_NRHS*sizeof(ftyp) +
                      safe_N * sizeof(fortran_int));
    
    # 检查内存分配是否成功，如果失败跳转到 error 标签处
    if (!mem_buff) {
        goto error;
    }
    
    # 将内存缓冲区分配给变量 a, b, ipiv，分别对应矩阵 A、向量 B 和整型数组 IPIV
    a = mem_buff;
    b = a + safe_N * safe_N * sizeof(ftyp);
    ipiv = b + safe_N * safe_NRHS * sizeof(ftyp);
    
    # 将分配的内存地址赋值给参数结构体 params 中对应的成员变量
    params->A = (ftyp*)a;
    params->B = (ftyp*)b;
    params->IPIV = (fortran_int*)ipiv;
    params->N = N;
    params->NRHS = NRHS;
    params->LDA = ld;
    params->LDB = ld;
    
    # 成功初始化参数后返回 1
    return 1;
    
    # 如果内存分配失败，则释放 mem_buff 所占用的内存
    error:
    free(mem_buff);
    # 将 params 结构体的内容清零，即初始化为默认值
    memset(params, 0, sizeof(*params));
    
    # 返回 0 表示初始化失败
    return 0;
}

template<typename ftyp>
static inline void
release_gesv(GESV_PARAMS_t<ftyp> *params)
{
    /* 释放 params 结构体中的 A 成员所指向的内存块 */
    free(params->A);
    /* 将 params 结构体清零 */
    memset(params, 0, sizeof(*params));
}

template<typename typ>
static void
solve(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    /* 定义 ftyp 为 typ 对应的 Fortran 类型 */
    using ftyp = fortran_type_t<typ>;
    /* 定义 GESV_PARAMS_t 结构体实例 params */
    GESV_PARAMS_t<ftyp> params;
    /* 定义 fortran_int 类型变量 n 和 nrhs */
    fortran_int n, nrhs;
    /* 获取并清除浮点无效标志 */
    int error_occurred = get_fp_invalid_and_clear();
    /* 初始化外层循环控制 */
    INIT_OUTER_LOOP_3

    /* 将 dimensions 数组中的第一个元素转换为 fortran_int 类型赋值给 n */
    n = (fortran_int)dimensions[0];
    /* 将 dimensions 数组中的第二个元素转换为 fortran_int 类型赋值给 nrhs */
    nrhs = (fortran_int)dimensions[1];
    /* 如果初始化 gesv 函数返回非零值，执行以下代码块 */
    if (init_gesv(&params, n, nrhs)) {
        /* 初始化 linearize_data 结构体 a_in、b_in、r_out */
        linearize_data a_in = init_linearize_data(n, n, steps[1], steps[0]);
        linearize_data b_in = init_linearize_data(nrhs, n, steps[3], steps[2]);
        linearize_data r_out = init_linearize_data(nrhs, n, steps[5], steps[4]);

        /* 开始外层循环 */
        BEGIN_OUTER_LOOP_3
            int not_ok;
            /* 将 args[0] 的数据线性化到 params.A 中 */
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            /* 将 args[1] 的数据线性化到 params.B 中 */
            linearize_matrix((typ*)params.B, (typ*)args[1], &b_in);
            /* 调用 gesv 函数并将返回值赋给 not_ok */
            not_ok = call_gesv(&params);
            /* 如果 gesv 函数返回正常，将 params.B 的数据反线性化到 args[2] 中 */
            if (!not_ok) {
                delinearize_matrix((typ*)args[2], (typ*)params.B, &r_out);
            } else {
                /* 如果 gesv 函数返回异常，将 args[2] 的数据设置为 NaN */
                error_occurred = 1;
                nan_matrix((typ*)args[2], &r_out);
            }
        /* 结束外层循环 */
        END_OUTER_LOOP

        /* 释放 params 结构体所占用的资源 */
        release_gesv(&params);
    }

    /* 设置浮点无效标志 */
    set_fp_invalid_or_clear(error_occurred);
}


template<typename typ>
static void
solve1(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func))
{
    /* 定义 ftyp 为 typ 对应的 Fortran 类型 */
    using ftyp = fortran_type_t<typ>;
    /* 定义 GESV_PARAMS_t 结构体实例 params */
    GESV_PARAMS_t<ftyp> params;
    /* 获取并清除浮点无效标志 */
    int error_occurred = get_fp_invalid_and_clear();
    /* 定义 fortran_int 类型变量 n */
    fortran_int n;
    /* 初始化外层循环控制 */
    INIT_OUTER_LOOP_3

    /* 将 dimensions 数组中的第一个元素转换为 fortran_int 类型赋值给 n */
    n = (fortran_int)dimensions[0];
    /* 如果初始化 gesv 函数返回非零值，执行以下代码块 */
    if (init_gesv(&params, n, 1)) {
        /* 初始化 linearize_data 结构体 a_in、b_in、r_out */
        linearize_data a_in = init_linearize_data(n, n, steps[1], steps[0]);
        linearize_data b_in = init_linearize_data(1, n, 1, steps[2]);
        linearize_data r_out = init_linearize_data(1, n, 1, steps[3]);

        /* 开始外层循环 */
        BEGIN_OUTER_LOOP_3
            int not_ok;
            /* 将 args[0] 的数据线性化到 params.A 中 */
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            /* 将 args[1] 的数据线性化到 params.B 中 */
            linearize_matrix((typ*)params.B, (typ*)args[1], &b_in);
            /* 调用 gesv 函数并将返回值赋给 not_ok */
            not_ok = call_gesv(&params);
            /* 如果 gesv 函数返回正常，将 params.B 的数据反线性化到 args[2] 中 */
            if (!not_ok) {
                delinearize_matrix((typ*)args[2], (typ*)params.B, &r_out);
            } else {
                /* 如果 gesv 函数返回异常，将 args[2] 的数据设置为 NaN */
                error_occurred = 1;
                nan_matrix((typ*)args[2], &r_out);
            }
        /* 结束外层循环 */
        END_OUTER_LOOP

        /* 释放 params 结构体所占用的资源 */
        release_gesv(&params);
    }

    /* 设置浮点无效标志 */
    set_fp_invalid_or_clear(error_occurred);
}

template<typename typ>
static void
inv(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    /* 定义 ftyp 为 typ 对应的 Fortran 类型 */
    using ftyp = fortran_type_t<typ>;
    /* 定义 GESV_PARAMS_t 结构体实例 params */
    GESV_PARAMS_t<ftyp> params;
    /* 定义 fortran_int 类型变量 n */
    fortran_int n;
    /* 获取并清除浮点无效标志 */
    int error_occurred = get_fp_invalid_and_clear();
    /* 初始化外层循环控制 */
    INIT_OUTER_LOOP_2

    /* 将 dimensions 数组中的第一个元素转换为 fortran_int 类型赋值给 n */
    n = (fortran_int)dimensions[0];
    // 如果初始化参数失败，则不执行以下代码块
    if (init_gesv(&params, n, n)) {
        // 初始化用于线性化数据的输入和输出结构体
        linearize_data a_in = init_linearize_data(n, n, steps[1], steps[0]);
        linearize_data r_out = init_linearize_data(n, n, steps[3], steps[2]);

        // 开始外层循环
        BEGIN_OUTER_LOOP_2
            int not_ok;
            // 将参数 A 线性化为输入矩阵 a_in
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            // 初始化参数 B 为单位矩阵
            identity_matrix((typ*)params.B, n);
            // 调用线性方程求解函数 gesv
            not_ok = call_gesv(&params);
            // 如果求解成功，则将结果反线性化到输出矩阵 args[1]
            if (!not_ok) {
                delinearize_matrix((typ*)args[1], (typ*)params.B, &r_out);
            } else {
                // 如果求解失败，则标记错误，并将输出矩阵设置为 NaN
                error_occurred = 1;
                nan_matrix((typ*)args[1], &r_out);
            }
        // 结束外层循环
        END_OUTER_LOOP

        // 释放参数对象的资源
        release_gesv(&params);
    }

    // 设置浮点数异常标志或清除错误状态
    set_fp_invalid_or_clear(error_occurred);
/* -------------------------------------------------------------------------- */
/* Cholesky decomposition */

template<typename typ>
struct POTR_PARAMS_t
{
    typ *A;         // 指向矩阵的指针
    fortran_int N;  // 矩阵的维度
    fortran_int LDA; // 矩阵的leading dimension
    char UPLO;      // 指示矩阵是上三角还是下三角
};

/* zero the undefined part in a upper/lower triangular matrix */
/* Note: matrix from fortran routine, so column-major order */

template<typename typ>
static inline void
zero_lower_triangle(POTR_PARAMS_t<typ> *params)
{
    fortran_int n = params->N;   // 获取矩阵的维度
    typ *matrix = params->A;     // 获取矩阵的起始地址
    fortran_int i, j;
    for (i = 0; i < n-1; ++i) {  // 循环遍历矩阵的行
        for (j = i+1; j < n; ++j) { // 循环遍历矩阵的列
            matrix[j] = numeric_limits<typ>::zero; // 将矩阵的下三角部分清零
        }
        matrix += n;  // 移动到下一行的起始位置
    }
}

template<typename typ>
static inline void
zero_upper_triangle(POTR_PARAMS_t<typ> *params)
{
    fortran_int n = params->N;   // 获取矩阵的维度
    typ *matrix = params->A;     // 获取矩阵的起始地址
    fortran_int i, j;
    matrix += n;  // 移动到矩阵的第二行起始位置
    for (i = 1; i < n; ++i) {    // 循环遍历矩阵的行
        for (j = 0; j < i; ++j) { // 循环遍历矩阵的列
            matrix[j] = numeric_limits<typ>::zero; // 将矩阵的上三角部分清零
        }
        matrix += n;  // 移动到下一行的起始位置
    }
}

static inline fortran_int
call_potrf(POTR_PARAMS_t<fortran_real> *params)
{
    fortran_int rv;
    LAPACK(spotrf)(&params->UPLO,
                          &params->N, params->A, &params->LDA,
                          &rv);
    return rv;
}

static inline fortran_int
call_potrf(POTR_PARAMS_t<fortran_doublereal> *params)
{
    fortran_int rv;
    LAPACK(dpotrf)(&params->UPLO,
                          &params->N, params->A, &params->LDA,
                          &rv);
    return rv;
}

static inline fortran_int
call_potrf(POTR_PARAMS_t<fortran_complex> *params)
{
    fortran_int rv;
    LAPACK(cpotrf)(&params->UPLO,
                          &params->N, params->A, &params->LDA,
                          &rv);
    return rv;
}

static inline fortran_int
call_potrf(POTR_PARAMS_t<fortran_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zpotrf)(&params->UPLO,
                          &params->N, params->A, &params->LDA,
                          &rv);
    return rv;
}

template<typename ftyp>
static inline int
init_potrf(POTR_PARAMS_t<ftyp> *params, char UPLO, fortran_int N)
{
    npy_uint8 *mem_buff = NULL;  // 内存缓冲区指针
    npy_uint8 *a;                // 矩阵起始地址指针
    size_t safe_N = N;
    fortran_int lda = fortran_int_max(N, 1); // 计算leading dimension

    mem_buff = (npy_uint8 *)malloc(safe_N * safe_N * sizeof(ftyp)); // 分配内存缓冲区
    if (!mem_buff) {
        goto error;
    }

    a = mem_buff;

    params->A = (ftyp*)a;  // 设置矩阵指针
    params->N = N;         // 设置矩阵维度
    params->LDA = lda;     // 设置leading dimension
    params->UPLO = UPLO;   // 设置矩阵的上/下三角属性

    return 1;
 error:
    free(mem_buff);        // 释放内存缓冲区
    memset(params, 0, sizeof(*params)); // 将参数结构体清零

    return 0;
}

template<typename ftyp>
static inline void
release_potrf(POTR_PARAMS_t<ftyp> *params)
{
    /* memory block base in A */
    free(params->A);       // 释放矩阵内存
    memset(params, 0, sizeof(*params)); // 将参数结构体清零
}

template<typename typ>
static void
cholesky(char uplo, char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    using ftyp = fortran_type_t<typ>; // 定义Fortran类型
    POTR_PARAMS_t<ftyp> params;
    # 声明一个名为params的变量，类型为POTR_PARAMS_t<ftyp>，这是一个模板类实例

    int error_occurred = get_fp_invalid_and_clear();
    # 调用函数get_fp_invalid_and_clear()，获取并清除浮点无效状态，将结果保存在error_occurred变量中

    fortran_int n;
    # 声明一个名为n的变量，类型为fortran_int

    INIT_OUTER_LOOP_2
    # 执行宏或函数INIT_OUTER_LOOP_2，用于初始化外层循环的控制结构

    n = (fortran_int)dimensions[0];
    # 将dimensions数组的第一个元素转换为fortran_int类型，并赋值给n

    if (init_potrf(&params, uplo, n)) {
        # 如果调用init_potrf(&params, uplo, n)返回非零值（表示初始化成功）
        
        linearize_data a_in = init_linearize_data(n, n, steps[1], steps[0]);
        # 声明并初始化一个名为a_in的linearize_data结构，用于将n x n矩阵线性化，具体步长由steps数组指定

        linearize_data r_out = init_linearize_data(n, n, steps[3], steps[2]);
        # 声明并初始化一个名为r_out的linearize_data结构，用于将n x n矩阵线性化，具体步长由steps数组指定

        BEGIN_OUTER_LOOP_2
        # 执行宏或函数BEGIN_OUTER_LOOP_2，开始外层循环的操作

            int not_ok;
            # 声明一个名为not_ok的整型变量

            linearize_matrix(params.A, (ftyp*)args[0], &a_in);
            # 调用linearize_matrix函数，将params.A的内容线性化存储到args[0]指向的位置，使用a_in作为配置信息

            not_ok = call_potrf(&params);
            # 调用call_potrf(&params)，并将返回值赋给not_ok

            if (!not_ok) {
                # 如果not_ok为假（即返回值为0，表示调用成功）

                if (uplo == 'L') {
                    zero_upper_triangle(&params);
                    # 如果uplo为'L'，调用zero_upper_triangle函数清零params.A的上三角部分
                }
                else {
                    zero_lower_triangle(&params);
                    # 否则调用zero_lower_triangle函数清零params.A的下三角部分
                }

                delinearize_matrix((ftyp*)args[1], params.A, &r_out);
                # 调用delinearize_matrix函数，将params.A的内容从线性化状态反解为args[1]指向的位置，使用r_out作为配置信息

            } else {
                # 如果not_ok为真（返回值非0，表示调用失败）

                error_occurred = 1;
                # 将error_occurred设置为1，表示出现错误

                nan_matrix((ftyp*)args[1], &r_out);
                # 调用nan_matrix函数，将args[1]位置的矩阵数据设置为NaN，使用r_out作为配置信息
            }

        END_OUTER_LOOP
        # 执行宏或函数END_OUTER_LOOP，结束外层循环的操作

        release_potrf(&params);
        # 调用release_potrf函数释放params结构体占用的资源
    }

    set_fp_invalid_or_clear(error_occurred);
    # 调用set_fp_invalid_or_clear函数，根据error_occurred的值来设置浮点无效状态或清除当前状态
}

// 模板函数：调用 cholesky 函数处理下三角矩阵
template<typename typ>
static void
cholesky_lo(char **args, npy_intp const *dimensions, npy_intp const *steps,
                void *NPY_UNUSED(func))
{
    cholesky<typ>('L', args, dimensions, steps);
}

// 模板函数：调用 cholesky 函数处理上三角矩阵
template<typename typ>
static void
cholesky_up(char **args, npy_intp const *dimensions, npy_intp const *steps,
                void *NPY_UNUSED(func))
{
    cholesky<typ>('U', args, dimensions, steps);
}

/* -------------------------------------------------------------------------- */
                          /* eig family  */

// 结构体模板：包含用于 LAPACK 中 GEEV 函数的参数
template<typename typ>
struct GEEV_PARAMS_t {
    typ *A;            // 矩阵 A
    basetype_t<typ> *WR; /* RWORK in complex versions, REAL W buffer for (sd)geev*/
    typ *WI;           // 虚部分数组
    typ *VLR;          // 左本征向量部分
    typ *VRR;          // 右本征向量部分
    typ *WORK;         // 工作区数组
    typ *W;            // 最终本征值数组
    typ *VL;           // 最终左本征向量数组
    typ *VR;           // 最终右本征向量数组

    fortran_int N;     // 矩阵的阶数
    fortran_int LDA;   // A 的第一维长度
    fortran_int LDVL;  // VL 的第一维长度
    fortran_int LDVR;  // VR 的第一维长度
    fortran_int LWORK; // 工作区长度

    char JOBVL;        // 计算左本征向量的选项
    char JOBVR;        // 计算右本征向量的选项
};

// 函数模板：打印 GEEV 参数结构体的内容
template<typename typ>
static inline void
dump_geev_params(const char *name, GEEV_PARAMS_t<typ>* params)
{
    TRACE_TXT("\n%s\n"

              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\

              "\t%10s: %d\n"\
              "\t%10s: %d\n"\
              "\t%10s: %d\n"\
              "\t%10s: %d\n"\
              "\t%10s: %d\n"\

              "\t%10s: %c\n"\
              "\t%10s: %c\n",

              name,

              "A", params->A,
              "WR", params->WR,
              "WI", params->WI,
              "VLR", params->VLR,
              "VRR", params->VRR,
              "WORK", params->WORK,
              "W", params->W,
              "VL", params->VL,
              "VR", params->VR,

              "N", (int)params->N,
              "LDA", (int)params->LDA,
              "LDVL", (int)params->LDVL,
              "LDVR", (int)params->LDVR,
              "LWORK", (int)params->LWORK,

              "JOBVL", params->JOBVL,
              "JOBVR", params->JOBVR);
}

// 函数：调用 LAPACK 中的 sgeev 函数处理 float 类型的 GEEV 参数
static inline fortran_int
call_geev(GEEV_PARAMS_t<float>* params)
{
    fortran_int rv;
    LAPACK(sgeev)(&params->JOBVL, &params->JOBVR,
                          &params->N, params->A, &params->LDA,
                          params->WR, params->WI,
                          params->VLR, &params->LDVL,
                          params->VRR, &params->LDVR,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}

// 函数：调用 LAPACK 中的 sgeev 函数处理 double 类型的 GEEV 参数
static inline fortran_int
call_geev(GEEV_PARAMS_t<double>* params)
{
    fortran_int rv;
    LAPACK(dgeev)(&params->JOBVL, &params->JOBVR,
                          &params->N, params->A, &params->LDA,
                          params->WR, params->WI,
                          params->VLR, &params->LDVL,
                          params->VRR, &params->LDVR,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}
    # 调用 LAPACK 库中的 dgeev 函数，用于计算特征值和特征向量
    LAPACK(dgeev)(&params->JOBVL, &params->JOBVR,
                              &params->N, params->A, &params->LDA,
                              params->WR, params->WI,
                              params->VLR, &params->LDVL,
                              params->VRR, &params->LDVR,
                              params->WORK, &params->LWORK,
                              &rv);
    # 返回 LAPACK 函数的执行结果，通常是一个整数指示操作的成功与否
    return rv;
template<typename typ>
static inline int
init_geev(GEEV_PARAMS_t<typ> *params, char jobvl, char jobvr, fortran_int n,
scalar_trait)
{
    // 指向内存缓冲区的指针，初始化为NULL
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    // 指向各种数据的指针，初始为NULL
    npy_uint8 *a, *wr, *wi, *vlr, *vrr, *work, *w, *vl, *vr;
    // 将 n 转为 size_t 类型，确保安全使用
    size_t safe_n = n;
    // 计算各个数组所需内存大小
    size_t a_size = safe_n * safe_n * sizeof(typ);
    size_t wr_size = safe_n * sizeof(typ);
    size_t wi_size = safe_n * sizeof(typ);
    size_t vlr_size = jobvl=='V' ? safe_n * safe_n * sizeof(typ) : 0;
    size_t vrr_size = jobvr=='V' ? safe_n * safe_n * sizeof(typ) : 0;
    size_t w_size = wr_size * 2;
    size_t vl_size = vlr_size * 2;
    size_t vr_size = vrr_size * 2;
    size_t work_count = 0;
    // 计算 ld 的值
    fortran_int ld = fortran_int_max(n, 1);

    /* allocate data for known sizes (all but work) */
    // 分配内存缓冲区，包括所有数组的大小，除了 work
    mem_buff = (npy_uint8 *)malloc(a_size + wr_size + wi_size +
                      vlr_size + vrr_size +
                      w_size + vl_size + vr_size);
    if (!mem_buff) {
        goto error;  // 分配失败，跳转到错误处理
    }

    // 分配各个数组的内存位置
    a = mem_buff;
    wr = a + a_size;
    wi = wr + wr_size;
    vlr = wi + wi_size;
    vrr = vlr + vlr_size;
    w = vrr + vrr_size;
    vl = w + w_size;
    vr = vl + vl_size;

    // 将参数结构体中的指针指向对应的数组
    params->A = (typ*)a;
    params->WR = (typ*)wr;
    params->WI = (typ*)wi;
    params->VLR = (typ*)vlr;
    params->VRR = (typ*)vrr;
    params->W = (typ*)w;
    params->VL = (typ*)vl;
    params->VR = (typ*)vr;
    params->N = n;
    params->LDA = ld;
    params->LDVL = ld;
    params->LDVR = ld;
    params->JOBVL = jobvl;
    params->JOBVR = jobvr;

    /* Work size query */
    // 查询所需的 work 大小
    {
        typ work_size_query;

        params->LWORK = -1;  // 设置 LWORK 为 -1
        params->WORK = &work_size_query;  // 将 WORK 指向 work_size_query

        // 调用 call_geev 函数查询工作空间大小
        if (call_geev(params) != 0) {
            goto error;  // 调用失败，跳转到错误处理
        }

        work_count = (size_t)work_size_query;  // 获取查询得到的工作空间大小
    }

    // 分配工作空间的内存
    mem_buff2 = (npy_uint8 *)malloc(work_count * sizeof(typ));
    if (!mem_buff2) {
        goto error;  // 分配失败，跳转到错误处理
    }
    work = mem_buff2;  // 设置 work 指向分配的内存

    // 设置参数结构体中的 LWORK 和 WORK
    params->LWORK = (fortran_int)work_count;
    params->WORK = (typ*)work;

    return 1;  // 成功初始化，返回1

error:
    free(mem_buff2);  // 释放分配的内存
    free(mem_buff);
    memset(params, 0, sizeof(*params));  // 将 params 结构体清零

    return 0;  // 返回初始化失败
}
    # 迭代处理，从 0 到 n-1
    for (iter = 0; iter < n; ++iter) {
        # 获取实部
        typ re = r[iter];
        # 获取虚部
        typ im = r[iter+n];
        # 将实部赋给对应的复数结构体的实部
        c[iter].r = re;
        # 将虚部赋给对应的复数结构体的虚部
        c[iter].i = im;
        # 将实部赋给对应的复数结构体的第二部分实部（对应于位移n后的位置）
        c[iter+n].r = re;
        # 将负的虚部赋给对应的复数结构体的第二部分虚部（对应于位移n后的位置）
        c[iter+n].i = -im;
    }
/*
 * make the complex eigenvectors from the real array produced by sgeev/zgeev.
 * c is the array where the results will be left.
 * r is the source array of reals produced by sgeev/zgeev
 * i is the eigenvalue imaginary part produced by sgeev/zgeev
 * n is so that the order of the matrix is n by n
 */
template<typename complextyp, typename typ>
static inline void
mk_geev_complex_eigenvectors(complextyp *c,
                              const typ *r,
                              const typ *i,
                              size_t n)
{
    // 初始化迭代器
    size_t iter = 0;
    // 迭代处理每个特征值
    while (iter < n)
    {
        // 如果特征值的虚部为零，则为实特征值，生成实特征向量数组
        if (i[iter] ==  numeric_limits<typ>::zero) {
            /* eigenvalue was real, eigenvectors as well...  */
            mk_complex_array_from_real(c, r, n);
            c += n;  // 移动复数结果数组指针到下一个位置
            r += n;  // 移动实数源数组指针到下一个位置
            iter ++;  // 增加迭代器
        } else {
            // 特征值为复数，则生成一对复数特征向量
            mk_complex_array_conjugate_pair(c, r, n);
            c += 2*n;  // 移动复数结果数组指针到下一个位置的两倍
            r += 2*n;  // 移动实数源数组指针到下一个位置的两倍
            iter += 2;  // 增加迭代器两倍
        }
    }
}


template<typename complextyp, typename typ>
static inline void
process_geev_results(GEEV_PARAMS_t<typ> *params, scalar_trait)
{
    /* REAL versions of geev need the results to be translated
     * into complex versions. This is the way to deal with imaginary
     * results. In our gufuncs we will always return complex arrays!
     */
    // 将实数特征值结果转换为复数数组
    mk_complex_array((complextyp*)params->W, (typ*)params->WR, (typ*)params->WI, params->N);

    /* handle the eigenvectors */
    // 处理特征向量
    if ('V' == params->JOBVL) {
        // 如果需要左特征向量，则生成复数左特征向量数组
        mk_geev_complex_eigenvectors((complextyp*)params->VL, (typ*)params->VLR,
                                     (typ*)params->WI, params->N);
    }
    if ('V' == params->JOBVR) {
        // 如果需要右特征向量，则生成复数右特征向量数组
        mk_geev_complex_eigenvectors((complextyp*)params->VR, (typ*)params->VRR,
                                     (typ*)params->WI, params->N);
    }
}

#if 0
static inline fortran_int
call_geev(GEEV_PARAMS_t<fortran_complex>* params)
{
    fortran_int rv;

    LAPACK(cgeev)(&params->JOBVL, &params->JOBVR,
                          &params->N, params->A, &params->LDA,
                          params->W,
                          params->VL, &params->LDVL,
                          params->VR, &params->LDVR,
                          params->WORK, &params->LWORK,
                          params->WR, /* actually RWORK */
                          &rv);
    return rv;
}
#endif

static inline fortran_int
call_geev(GEEV_PARAMS_t<fortran_doublecomplex>* params)
{
    fortran_int rv;

    LAPACK(zgeev)(&params->JOBVL, &params->JOBVR,
                          &params->N, params->A, &params->LDA,
                          params->W,
                          params->VL, &params->LDVL,
                          params->VR, &params->LDVR,
                          params->WORK, &params->LWORK,
                          params->WR, /* actually RWORK */
                          &rv);
    return rv;
}
// 初始化 GEEV 算法的参数结构体
template<typename ftyp>
static inline int
init_geev(GEEV_PARAMS_t<ftyp>* params,
                   char jobvl,
                   char jobvr,
                   fortran_int n, complex_trait)
{
    // 定义实数类型为 ftyp 对应的基础类型
    using realtyp = basetype_t<ftyp>;

    // 初始化指针变量为 NULL
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *w, *vl, *vr, *work, *rwork;

    // 将 n 赋值给安全大小
    size_t safe_n = n;

    // 计算并分配内存空间大小
    size_t a_size = safe_n * safe_n * sizeof(ftyp);
    size_t w_size = safe_n * sizeof(ftyp);
    size_t vl_size = jobvl=='V'? safe_n * safe_n * sizeof(ftyp) : 0;
    size_t vr_size = jobvr=='V'? safe_n * safe_n * sizeof(ftyp) : 0;
    size_t rwork_size = 2 * safe_n * sizeof(realtyp);
    size_t work_count = 0;
    size_t total_size = a_size + w_size + vl_size + vr_size + rwork_size;

    // 计算矩阵 A 的列数
    fortran_int ld = fortran_int_max(n, 1);

    // 分配总内存空间
    mem_buff = (npy_uint8 *)malloc(total_size);
    if (!mem_buff) {
        goto error;
    }

    // 将内存空间分配给各个数组指针
    a = mem_buff;
    w = a + a_size;
    vl = w + w_size;
    vr = vl + vl_size;
    rwork = vr + vr_size;

    // 设置参数结构体中的成员指针
    params->A = (ftyp*)a;
    params->WR = (realtyp*)rwork;
    params->WI = NULL;
    params->VLR = NULL;
    params->VRR = NULL;
    params->VL = (ftyp*)vl;
    params->VR = (ftyp*)vr;
    params->W = (ftyp*)w;
    params->N = n;
    params->LDA = ld;
    params->LDVL = ld;
    params->LDVR = ld;
    params->JOBVL = jobvl;
    params->JOBVR = jobvr;

    /* Work size query */
    {
        // 定义用于查询工作空间大小的变量
        ftyp work_size_query;

        // 设置 LWORK 为 -1，以查询所需的工作空间大小
        params->LWORK = -1;
        params->WORK = &work_size_query;

        // 调用 GEEV 函数查询工作空间大小，若失败则跳转到错误处理
        if (call_geev(params) != 0) {
            goto error;
        }

        // 获取实际所需的工作空间大小并修复 Lapack 3.0.0 中的一个 bug
        work_count = (size_t) work_size_query.r;
        if(work_count == 0) work_count = 1;
    }

    // 分配所需的工作空间
    mem_buff2 = (npy_uint8 *)malloc(work_count*sizeof(ftyp));
    if (!mem_buff2) {
        goto error;
    }

    // 将工作空间内存指针赋给 params 结构体中的 WORK 成员
    work = mem_buff2;
    params->LWORK = (fortran_int)work_count;
    params->WORK = (ftyp*)work;

    // 初始化成功，返回 1
    return 1;

error:
    // 发生错误时释放内存空间并清空 params 结构体
    free(mem_buff2);
    free(mem_buff);
    memset(params, 0, sizeof(*params));

    // 返回 0 表示初始化失败
    return 0;
}

// 处理 GEEV 算法的结果，复数版本无需处理
template<typename complextyp, typename typ>
static inline void
process_geev_results(GEEV_PARAMS_t<typ> *NPY_UNUSED(params), complex_trait)
{
    /* nothing to do here, complex versions are ready to copy out */
}

// 释放 GEEV 算法所使用的内存空间
template<typename typ>
static inline void
release_geev(GEEV_PARAMS_t<typ> *params)
{
    // 释放工作空间和矩阵 A 的内存空间，并清空 params 结构体
    free(params->WORK);
    free(params->A);
    memset(params, 0, sizeof(*params));
}

// 对外提供的包装函数，用于调用 GEEV 算法
template<typename fctype, typename ftype>
static inline void
eig_wrapper(char JOBVL,
                   char JOBVR,
                   char**args,
                   npy_intp const *dimensions,
                   npy_intp const *steps)
{
    // 外部维度数组和迭代器变量
    ptrdiff_t outer_steps[4];
    size_t iter;
    size_t outer_dim = *dimensions++;
    size_t op_count = 2;
    int error_occurred = get_fp_invalid_and_clear();
    GEEV_PARAMS_t<ftype> geev_params;

    // 断言 JOBVL 必须为 'N'，即不计算左特征向量
    assert(JOBVL == 'N');

    // 堆栈追踪
    STACK_TRACE;

    // 根据 JOBVL 和 JOBVR 的值更新操作计数
    op_count += 'V'==JOBVL?1:0;
    op_count += 'V'==JOBVR?1:0;
}
    for (iter = 0; iter < op_count; ++iter) {
        outer_steps[iter] = (ptrdiff_t) steps[iter];
    }
    steps += op_count;

// 将 steps 数组中的前 op_count 个元素转换为 ptrdiff_t 类型并存储到 outer_steps 数组中，然后将 steps 指针向后移动 op_count 个位置。


    if (init_geev(&geev_params,
                           JOBVL, JOBVR,
                           (fortran_int)dimensions[0], dispatch_scalar<ftype>())) {

// 调用 init_geev 函数初始化 geev_params 结构体，传递 JOBVL、JOBVR、dimensions[0] 和 dispatch_scalar<ftype>() 的值作为参数。如果 init_geev 返回非零值（表示初始化成功），则执行以下代码块。


        linearize_data vl_out = {}; /* silence uninitialized warning */
        linearize_data vr_out = {}; /* silence uninitialized warning */

// 初始化 vl_out 和 vr_out 结构体，用于存储线性化的数据。这里使用空初始化列表来消除未初始化的警告。


        linearize_data a_in = init_linearize_data(
                            geev_params.N, geev_params.N,
                            steps[1], steps[0]);
        steps += 2;

// 调用 init_linearize_data 函数初始化 a_in 结构体，设置其维度和步长参数，并将 steps 指针向后移动两个位置。


        linearize_data w_out = init_linearize_data(
                            1, geev_params.N,
                            0, steps[0]);
        steps += 1;

// 调用 init_linearize_data 函数初始化 w_out 结构体，设置其维度和步长参数，并将 steps 指针向后移动一位。


        if ('V' == geev_params.JOBVL) {
            vl_out = init_linearize_data(
                                geev_params.N, geev_params.N,
                                steps[1], steps[0]);
            steps += 2;
        }

// 如果 geev_params.JOBVL 的值为 'V'，则调用 init_linearize_data 函数初始化 vl_out 结构体，设置其维度和步长参数，并将 steps 指针向后移动两个位置。


        if ('V' == geev_params.JOBVR) {
            vr_out = init_linearize_data(
                                geev_params.N, geev_params.N,
                                steps[1], steps[0]);
        }

// 如果 geev_params.JOBVR 的值为 'V'，则调用 init_linearize_data 函数初始化 vr_out 结构体，设置其维度和步长参数。


        for (iter = 0; iter < outer_dim; ++iter) {
            int not_ok;
            char **arg_iter = args;
            /* copy the matrix in */
            linearize_matrix((ftype*)geev_params.A, (ftype*)*arg_iter++, &a_in);
            not_ok = call_geev(&geev_params);

            if (!not_ok) {
                process_geev_results<fctype>(&geev_params,

// 循环遍历 outer_dim 次执行以下操作：将 args 中的矩阵复制到 geev_params.A 中，调用 call_geev 函数执行 geev 计算，并检查计算是否成功。
/* -------------------------------------------------------------------------- */
/*                            singular value decomposition                    */

template<typename ftyp>
struct GESDD_PARAMS_t
{
    ftyp *A;                      // 输入矩阵 A
    basetype_t<ftyp> *S;          // 奇异值数组 S
    ftyp *U;                      // 左奇异向量 U
    ftyp *VT;                     // 右奇异向量的转置 VT
    ftyp *WORK;                   // 工作数组 WORK
    basetype_t<ftyp> *RWORK;      // 实数类型工作数组 RWORK
    fortran_int *IWORK;           // 整数类型工作数组 IWORK

    fortran_int M;                // A 的行数
    fortran_int N;                // A 的列数
    fortran_int LDA;              // A 的 leading dimension
    fortran_int LDU;              // U 的 leading dimension
    fortran_int LDVT;             // VT 的 leading dimension
    fortran_int LWORK;            // WORK 数组的长度
    char JOBZ;                    // 指定是否计算 U 和 VT ('A', 'S', 'O', 'N')
} ;


template<typename ftyp>
static inline void
dump_gesdd_params(const char *name,
                  GESDD_PARAMS_t<ftyp> *params)
{
    # 打印调试信息，显示参数的详细值
    TRACE_TXT("\n%s:\n"\
    
              # 打印指针参数的地址
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
    
              # 打印整数参数的值
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
    
              # 打印字符参数的值
              "%14s: %15c'%c'\n",
    
              name,
    
              "A", params->A,
              "S", params->S,
              "U", params->U,
              "VT", params->VT,
              "WORK", params->WORK,
              "RWORK", params->RWORK,
              "IWORK", params->IWORK,
    
              "M", (int)params->M,
              "N", (int)params->N,
              "LDA", (int)params->LDA,
              "LDU", (int)params->LDU,
              "LDVT", (int)params->LDVT,
              "LWORK", (int)params->LWORK,
    
              "JOBZ", ' ', params->JOBZ);
}

static inline int
compute_urows_vtcolumns(char jobz,
                        fortran_int m, fortran_int n,
                        fortran_int *urows, fortran_int *vtcolumns)
{
    // 计算 m 和 n 的最小值
    fortran_int min_m_n = fortran_int_min(m, n);
    // 根据 jobz 的不同取值进行不同的操作
    switch(jobz)
    {
    case 'N':
        // 不计算 U 和 VT
        *urows = 0;
        *vtcolumns = 0;
        break;
    case 'A':
        // 计算全部的 U 和 VT
        *urows = m;
        *vtcolumns = n;
        break;
    case 'S':
        {
            // 计算较小的 min_m_n 个 U 和 VT
            *urows = min_m_n;
            *vtcolumns = min_m_n;
        }
        break;
    default:
        // 默认情况下返回错误
        return 0;
    }

    // 成功计算则返回 1
    return 1;
}

static inline fortran_int
call_gesdd(GESDD_PARAMS_t<fortran_real> *params)
{
    fortran_int rv;
    // 调用 LAPACK 库中的单精度实数版本 gesdd 函数
    LAPACK(sgesdd)(&params->JOBZ, &params->M, &params->N,
                          params->A, &params->LDA,
                          params->S,
                          params->U, &params->LDU,
                          params->VT, &params->LDVT,
                          params->WORK, &params->LWORK,
                          (fortran_int*)params->IWORK,
                          &rv);
    // 返回 LAPACK 函数的返回值
    return rv;
}

static inline fortran_int
call_gesdd(GESDD_PARAMS_t<fortran_doublereal> *params)
{
    fortran_int rv;
    // 调用 LAPACK 库中的双精度实数版本 gesdd 函数
    LAPACK(dgesdd)(&params->JOBZ, &params->M, &params->N,
                          params->A, &params->LDA,
                          params->S,
                          params->U, &params->LDU,
                          params->VT, &params->LDVT,
                          params->WORK, &params->LWORK,
                          (fortran_int*)params->IWORK,
                          &rv);
    // 返回 LAPACK 函数的返回值
    return rv;
}

template<typename ftyp>
static inline int
init_gesdd(GESDD_PARAMS_t<ftyp> *params,
                   char jobz,
                   fortran_int m,
                   fortran_int n, scalar_trait)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *s, *u, *vt, *work, *iwork;
    size_t safe_m = m;
    size_t safe_n = n;
    // 计算数组 a 的大小
    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    // 计算 min_m_n
    fortran_int min_m_n = fortran_int_min(m, n);
    size_t safe_min_m_n = min_m_n;
    // 计算数组 s 的大小
    size_t s_size = safe_min_m_n * sizeof(ftyp);
    fortran_int u_row_count, vt_column_count;
    size_t safe_u_row_count, safe_vt_column_count;
    size_t u_size, vt_size;
    fortran_int work_count;
    size_t work_size;
    // 计算 iwork 的大小
    size_t iwork_size = 8 * safe_min_m_n * sizeof(fortran_int);
    // 计算 ld
    fortran_int ld = fortran_int_max(m, 1);

    // 根据 jobz 的值计算 u_row_count 和 vt_column_count
    if (!compute_urows_vtcolumns(jobz, m, n, &u_row_count, &vt_column_count)) {
        // 如果计算失败则跳转到 error 标签
        goto error;
    }

    safe_u_row_count = u_row_count;
    safe_vt_column_count = vt_column_count;

    // 计算 u 和 vt 的大小
    u_size = safe_u_row_count * safe_m * sizeof(ftyp);
    vt_size = safe_n * safe_vt_column_count * sizeof(ftyp);

    // 分配内存空间
    mem_buff = (npy_uint8 *)malloc(a_size + s_size + u_size + vt_size + iwork_size);

    if (!mem_buff) {
        // 内存分配失败，跳转到 error 标签
        goto error;
    }

    // 设置各个指针指向相应的内存位置
    a = mem_buff;
    s = a + a_size;
    u = s + s_size;
    vt = u + u_size;
    iwork = vt + vt_size;
    /* 使 vt_column_count 成为有效的 LAPACK 参数（0 不是有效值） */
    vt_column_count = fortran_int_max(1, vt_column_count);

    // 设置参数结构体中的维度信息
    params->M = m;          // 矩阵 A 的行数
    params->N = n;          // 矩阵 A 的列数
    params->A = (ftyp*)a;   // 矩阵 A 的数据
    params->S = (ftyp*)s;   // 奇异值数组
    params->U = (ftyp*)u;   // 左奇异向量矩阵
    params->VT = (ftyp*)vt; // 右奇异向量转置矩阵
    params->RWORK = NULL;   // 用于实数工作空间的指针（未使用）
    params->IWORK = (fortran_int*)iwork; // 用于整数工作空间的指针
    params->LDA = ld;       // 矩阵 A 的列数
    params->LDU = ld;       // 矩阵 U 的列数
    params->LDVT = vt_column_count; // 矩阵 VT 的列数
    params->JOBZ = jobz;    // 指定计算选项

    /* 查询工作空间大小 */
    {
        ftyp work_size_query; // 用于存储工作空间大小的变量

        params->LWORK = -1;   // 设置为-1表示查询工作空间大小
        params->WORK = &work_size_query; // 指向存储工作空间大小的变量的指针

        if (call_gesdd(params) != 0) { // 调用 LAPACK 函数查询工作空间大小
            goto error; // 如果调用失败，跳转到错误处理
        }

        work_count = (fortran_int)work_size_query; // 获取查询得到的工作空间大小
        /* 修复 LAPACK 3.0.0 中的一个 bug */
        if (work_count == 0) work_count = 1; // 如果工作空间大小为 0，则设置为 1
        work_size = (size_t)work_count * sizeof(ftyp); // 计算需要分配的工作空间大小
    }

    mem_buff2 = (npy_uint8 *)malloc(work_size); // 分配实际的工作空间内存
    if (!mem_buff2) { // 内存分配失败处理
        goto error; // 跳转到错误处理
    }

    work = mem_buff2; // 将分配的内存地址存入 work 指针

    params->LWORK = work_count; // 设置参数结构体中的工作空间大小
    params->WORK = (ftyp*)work; // 设置参数结构体中的工作空间指针

    return 1; // 成功初始化，返回 1
error:
    TRACE_TXT("%s failed init\n", __FUNCTION__); // 输出错误信息
    free(mem_buff); // 释放之前分配的内存
    free(mem_buff2); // 释放当前分配的工作空间内存
    memset(params, 0, sizeof(*params)); // 将参数结构体清零

    return 0; // 返回初始化失败
}

static inline fortran_int
call_gesdd(GESDD_PARAMS_t<fortran_complex> *params)
{
    fortran_int rv;
    // 调用 LAPACK 库函数 cgesdd 进行复数类型的奇异值分解
    LAPACK(cgesdd)(&params->JOBZ, &params->M, &params->N,
                          params->A, &params->LDA,
                          params->S,
                          params->U, &params->LDU,
                          params->VT, &params->LDVT,
                          params->WORK, &params->LWORK,
                          params->RWORK,
                          params->IWORK,
                          &rv);
    // 返回 LAPACK 函数调用的返回值
    return rv;
}

static inline fortran_int
call_gesdd(GESDD_PARAMS_t<fortran_doublecomplex> *params)
{
    fortran_int rv;
    // 调用 LAPACK 库函数 zgesdd 进行双精度复数类型的奇异值分解
    LAPACK(zgesdd)(&params->JOBZ, &params->M, &params->N,
                          params->A, &params->LDA,
                          params->S,
                          params->U, &params->LDU,
                          params->VT, &params->LDVT,
                          params->WORK, &params->LWORK,
                          params->RWORK,
                          params->IWORK,
                          &rv);
    // 返回 LAPACK 函数调用的返回值
    return rv;
}

template<typename ftyp>
static inline int
init_gesdd(GESDD_PARAMS_t<ftyp> *params,
                   char jobz,
                   fortran_int m,
                   fortran_int n, complex_trait)
{
    using frealtyp = basetype_t<ftyp>;
    npy_uint8 *mem_buff = NULL, *mem_buff2 = NULL;
    npy_uint8 *a,*s, *u, *vt, *work, *rwork, *iwork;
    size_t a_size, s_size, u_size, vt_size, work_size, rwork_size, iwork_size;
    size_t safe_u_row_count, safe_vt_column_count;
    fortran_int u_row_count, vt_column_count, work_count;
    size_t safe_m = m;
    size_t safe_n = n;
    fortran_int min_m_n = fortran_int_min(m, n);
    size_t safe_min_m_n = min_m_n;
    fortran_int ld = fortran_int_max(m, 1);

    // 根据参数计算安全的 U 和 VT 的行数和列数
    if (!compute_urows_vtcolumns(jobz, m, n, &u_row_count, &vt_column_count)) {
        goto error;
    }

    safe_u_row_count = u_row_count;
    safe_vt_column_count = vt_column_count;

    // 计算各个缓冲区的大小
    a_size = safe_m * safe_n * sizeof(ftyp);
    s_size = safe_min_m_n * sizeof(frealtyp);
    u_size = safe_u_row_count * safe_m * sizeof(ftyp);
    vt_size = safe_n * safe_vt_column_count * sizeof(ftyp);
    rwork_size = 'N'==jobz?
        (7 * safe_min_m_n) :
        (5*safe_min_m_n * safe_min_m_n + 5*safe_min_m_n);
    rwork_size *= sizeof(ftyp);
    iwork_size = 8 * safe_min_m_n* sizeof(fortran_int);

    // 分配内存缓冲区
    mem_buff = (npy_uint8 *)malloc(a_size +
                      s_size +
                      u_size +
                      vt_size +
                      rwork_size +
                      iwork_size);
    if (!mem_buff) {
        goto error;
    }

    // 指定各个数据结构在内存缓冲区中的位置
    a = mem_buff;
    s = a + a_size;
    u = s + s_size;
    vt = u + u_size;
    rwork = vt + vt_size;
    iwork = rwork + rwork_size;

    /* fix vt_column_count so that it is a valid lapack parameter (0 is not) */
    // 调整 vt_column_count 以确保其为 LAPACK 函数的有效参数（0 是无效的）
    vt_column_count = fortran_int_max(1, vt_column_count);

    // 设置参数结构体中各个指针的值
    params->A = (ftyp*)a;
    params->S = (frealtyp*)s;
    params->U = (ftyp*)u;
    // 将参数指针指向传入的数组 vt
    params->VT = (ftyp*)vt;
    // 将参数指针指向传入的数组 rwork
    params->RWORK = (frealtyp*)rwork;
    // 将参数指针指向传入的数组 iwork
    params->IWORK = (fortran_int*)iwork;
    // 设置参数中的矩阵行数 M
    params->M = m;
    // 设置参数中的矩阵列数 N
    params->N = n;
    // 设置参数中的矩阵的 leading dimension LDA
    params->LDA = ld;
    // 设置参数中的左奇异矩阵的 leading dimension LDU
    params->LDU = ld;
    // 设置参数中的右奇异矩阵的 leading dimension LDVT
    params->LDVT = vt_column_count;
    // 设置参数中的计算选项 JOBZ

    /* Work size query */
    {
        // 定义用于工作空间查询的变量
        ftyp work_size_query;

        // 设置 LWORK 为 -1 以查询所需的工作空间大小
        params->LWORK = -1;
        // 将 WORK 指向工作空间查询变量的地址
        params->WORK = &work_size_query;

        // 调用 gesdd 函数进行工作空间大小查询，并检查返回值
        if (call_gesdd(params) != 0) {
            // 如果调用失败，跳转到错误处理
            goto error;
        }

        // 从工作空间查询结果中获取工作空间的大小
        work_count = (fortran_int)(*(frealtyp*)&work_size_query);
        // 修复 lapack 3.0.0 中的一个 bug
        if (work_count == 0) work_count = 1;
        // 计算实际需要的工作空间大小
        work_size = (size_t)work_count * sizeof(ftyp);
    }

    // 分配工作空间的内存
    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2) {
        // 如果内存分配失败，跳转到错误处理
        goto error;
    }

    // 将工作指针指向分配的内存空间
    work = mem_buff2;

    // 将参数中的 LWORK 设置为实际计算得到的工作空间大小
    params->LWORK = work_count;
    // 将参数中的 WORK 指针指向工作空间
    params->WORK = (ftyp*)work;

    // 成功初始化，返回 1
    return 1;
error:
    // 打印错误信息
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    // 释放分配的内存空间
    free(mem_buff2);
    free(mem_buff);
    // 将参数结构体清零
    memset(params, 0, sizeof(*params));

    // 返回初始化失败
    return 0;
}

template<typename typ>
static inline void
release_gesdd(GESDD_PARAMS_t<typ>* params)
{
    /* 释放 params 结构体中的 A 和 WORK 字段所分配的内存块 */
    free(params->A);
    free(params->WORK);
    // 将 params 结构体的内存清零
    memset(params, 0, sizeof(*params));
}

template<typename typ>
static inline void
svd_wrapper(char JOBZ,
                   char **args,
                   npy_intp const *dimensions,
                   npy_intp const *steps)
{
    // 使用 basetype_t<typ> 定义 basetyp 类型
    using basetyp = basetype_t<typ>;
    // 声明一个指针数组 outer_steps，包含四个元素
    ptrdiff_t outer_steps[4];
    // 获取浮点无效操作的状态并清除
    int error_occurred = get_fp_invalid_and_clear();
    // 定义迭代器 iter
    size_t iter;
    // 获取外部维度的大小
    size_t outer_dim = *dimensions++;
    // 确定操作计数 op_count，如果 JOBZ 为 'N' 则为 2，否则为 4
    size_t op_count = (JOBZ=='N')?2:4;
    // 定义 GESDD_PARAMS_t<typ> 结构体变量 params
    GESDD_PARAMS_t<typ> params;

    // 将 steps 数组中的值赋给 outer_steps 数组
    for (iter = 0; iter < op_count; ++iter) {
        outer_steps[iter] = (ptrdiff_t) steps[iter];
    }
    // 更新 steps 指针
    steps += op_count;

    // 初始化 gesdd 参数
    if (init_gesdd(&params,
                   JOBZ,
                   (fortran_int)dimensions[0],
                   (fortran_int)dimensions[1],


注：代码截至此处。
template<typename typ>
static void
svd_N(char **args,
             npy_intp const *dimensions,
             npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    /* 定义保存结果的变量 */
    linearize_data u_out = {}, s_out = {}, v_out = {};
    /* 计算最小的维度大小 */
    fortran_int min_m_n = params.M < params.N ? params.M : params.N;

    /* 初始化输入矩阵数据 */
    linearize_data a_in = init_linearize_data(params.N, params.M, steps[1], steps[0]);
    
    /* 根据 JOBZ 的值决定需要初始化的输出变量 */
    if ('N' == params.JOBZ) {
        /* 只需要奇异值 */
        s_out = init_linearize_data(1, min_m_n, 0, steps[2]);
    } else {
        fortran_int u_columns, v_rows;
        if ('S' == params.JOBZ) {
            /* 需要计算全部左奇异向量和右奇异向量 */
            u_columns = min_m_n;
            v_rows = min_m_n;
        } else { /* JOBZ == 'A' */
            /* 需要计算所有左右奇异向量 */
            u_columns = params.M;
            v_rows = params.N;
        }
        /* 初始化左奇异向量、奇异值、右奇异向量 */
        u_out = init_linearize_data(u_columns, params.M, steps[3], steps[2]);
        s_out = init_linearize_data(1, min_m_n, 0, steps[4]);
        v_out = init_linearize_data(params.N, v_rows, steps[6], steps[5]);
    }

    /* 外层循环迭代 */
    for (iter = 0; iter < outer_dim; ++iter) {
        int not_ok;
        /* 将矩阵复制到输入矩阵数据结构 */
        linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
        /* 调用 SVD 计算 */
        not_ok = call_gesdd(&params);
        if (!not_ok) {
            /* 如果计算成功 */
            if ('N' == params.JOBZ) {
                /* 只返回奇异值 */
                delinearize_matrix((basetyp*)args[1], (basetyp*)params.S, &s_out);
            } else {
                if ('A' == params.JOBZ && min_m_n == 0) {
                    /* 处理 Lapack 未初始化的情况，产生单位矩阵 */
                    identity_matrix((typ*)params.U, params.M);
                    identity_matrix((typ*)params.VT, params.N);
                }
                /* 返回左右奇异向量和奇异值 */
                delinearize_matrix((typ*)args[1], (typ*)params.U, &u_out);
                delinearize_matrix((basetyp*)args[2], (basetyp*)params.S, &s_out);
                delinearize_matrix((typ*)args[3], (typ*)params.VT, &v_out);
            }
        } else {
            /* 如果计算失败 */
            error_occurred = 1;
            if ('N' == params.JOBZ) {
                /* 返回 NaN 奇异值 */
                nan_matrix((basetyp*)args[1], &s_out);
            } else {
                /* 返回 NaN 左右奇异向量和奇异值 */
                nan_matrix((typ*)args[1], &u_out);
                nan_matrix((basetyp*)args[2], &s_out);
                nan_matrix((typ*)args[3], &v_out);
            }
        }
        /* 更新指针位置 */
        update_pointers((npy_uint8**)args, outer_steps, op_count);
    }

    /* 释放 SVD 计算资源 */
    release_gesdd(&params);
}

/* 设置错误标志 */
set_fp_invalid_or_clear(error_occurred);
    # 使用 svd_wrapper 函数对给定参数执行奇异值分解操作，使用 'N' 参数表示不返回特征向量
    svd_wrapper<fortran_type_t<typ>>('N', args, dimensions, steps);
/* -------------------------------------------------------------------------- */
/* svd_S: 执行SVD分解的静态函数，处理模板类型typ对应的单字母参数'S' */
template<typename typ>
static void
svd_S(char **args,
             npy_intp const *dimensions,
             npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    // 调用svd_wrapper函数处理SVD分解，传入模板类型typ对应的参数和相关信息
    svd_wrapper<fortran_type_t<typ>>('S', args, dimensions, steps);
}

/* svd_A: 执行SVD分解的静态函数，处理模板类型typ对应的单字母参数'A' */
template<typename typ>
static void
svd_A(char **args,
             npy_intp const *dimensions,
             npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    // 调用svd_wrapper函数处理SVD分解，传入模板类型typ对应的参数和相关信息
    svd_wrapper<fortran_type_t<typ>>('A', args, dimensions, steps);
}

/* -------------------------------------------------------------------------- */
/* qr (modes - r, raw) */

/* GEQRF_PARAMS_t: 用于存储GEQRF参数的结构模板 */
template<typename typ>
struct GEQRF_PARAMS_t
{
    fortran_int M;      // 矩阵A的行数
    fortran_int N;      // 矩阵A的列数
    typ *A;             // 指向矩阵A的指针
    fortran_int LDA;    // 矩阵A的leading dimension
    typ* TAU;           // 存储元素的向量tau
    typ *WORK;          // 存储工作区域的指针
    fortran_int LWORK;  // 工作区域的大小
};

/* dump_geqrf_params: 打印GEQRF参数的函数 */
template<typename typ>
static inline void
dump_geqrf_params(const char *name,
                  GEQRF_PARAMS_t<typ> *params)
{
    // 使用TRACE_TXT打印GEQRF参数的详细信息，包括矩阵A、向量TAU、工作区等
    TRACE_TXT("\n%s:\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n",

              name,

              "A", params->A,
              "TAU", params->TAU,
              "WORK", params->WORK,

              "M", (int)params->M,
              "N", (int)params->N,
              "LDA", (int)params->LDA,
              "LWORK", (int)params->LWORK);
}

/* call_geqrf: 调用LAPACK库中的dgeqrf函数执行QR分解 */
static inline fortran_int
call_geqrf(GEQRF_PARAMS_t<double> *params)
{
    fortran_int rv; // 存储函数返回值
    // 调用LAPACK库中的dgeqrf函数执行QR分解
    LAPACK(dgeqrf)(&params->M, &params->N,
                          params->A, &params->LDA,
                          params->TAU,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv; // 返回函数执行结果
}

/* call_geqrf: 调用LAPACK库中的zgeqrf函数执行复数类型的QR分解 */
static inline fortran_int
call_geqrf(GEQRF_PARAMS_t<f2c_doublecomplex> *params)
{
    fortran_int rv; // 存储函数返回值
    // 调用LAPACK库中的zgeqrf函数执行复数类型的QR分解
    LAPACK(zgeqrf)(&params->M, &params->N,
                          params->A, &params->LDA,
                          params->TAU,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv; // 返回函数执行结果
}

/* init_geqrf: 初始化GEQRF参数结构 */
static inline int
init_geqrf(GEQRF_PARAMS_t<fortran_doublereal> *params,
                   fortran_int m,
                   fortran_int n)
{
    using ftyp = fortran_doublereal;
    npy_uint8 *mem_buff = NULL; // 内存缓冲区指针
    npy_uint8 *mem_buff2 = NULL; // 第二个内存缓冲区指针
    npy_uint8 *a, *tau, *work; // 指向矩阵A、向量TAU、工作区的指针
    fortran_int min_m_n = fortran_int_min(m, n); // m和n的最小值
    size_t safe_min_m_n = min_m_n; // 安全的最小值
    size_t safe_m = m; // 安全的m值
    size_t safe_n = n; // 安全的n值

    size_t a_size = safe_m * safe_n * sizeof(ftyp); // 计算矩阵A所需内存大小
    size_t tau_size = safe_min_m_n * sizeof(ftyp); // 计算向量TAU所需内存大小

    fortran_int work_count; // 工作区域计数
    size_t work_size; // 工作区域大小
    fortran_int lda = fortran_int_max(1, m); // 确定矩阵A的leading dimension

    mem_buff = (npy_uint8 *)malloc(a_size + tau_size); // 分配内存空间

    if (!mem_buff) // 内存分配失败处理
        goto error;

    a = mem_buff; // 指向矩阵A的指针
    tau = a + a_size; // 指向向量TAU的指针
    memset(tau, 0, tau_size); // 将向量TAU的内存清零

    // 初始化GEQRF参数结构
    params->M = m; // 设置矩阵A的行数
    params->N = n; // 设置矩阵A的列数
    params->A = (ftyp*)a; // 设置矩阵A的指针
    params->TAU = (ftyp*)tau; // 设置向量TAU的指针
    params->LDA = lda; // 设置矩阵A的leading dimension
    {
        /* 计算最佳工作大小 */
    
        // 声明一个变量来存储工作大小查询的结果
        ftyp work_size_query;
    
        // 将参数结构体中的 WORK 指针指向工作大小查询的结果变量
        params->WORK = &work_size_query;
    
        // 设置参数结构体中的 LWORK 为 -1，用于查询工作大小
        params->LWORK = -1;
    
        // 调用函数进行 QR 分解，如果返回值不为 0，则跳转到错误处理标签
        if (call_geqrf(params) != 0)
            goto error;
    
        // 将工作大小查询结果转换为整数类型并存储在 work_count 中
        work_count = (fortran_int) *(ftyp*) params->WORK;
    
    }
    
    // 根据计算得到的工作大小和 n 的最大值来确定最终的 LWORK
    params->LWORK = fortran_int_max(fortran_int_max(1, n), work_count);
    
    // 计算所需的内存大小，并分配相应的内存空间
    work_size = (size_t) params->LWORK * sizeof(ftyp);
    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2)
        goto error;
    
    // 将分配的内存空间的起始地址赋给工作指针
    work = mem_buff2;
    
    // 将工作指针的地址转换为特定类型的指针，并存储在参数结构体的 WORK 中
    params->WORK = (ftyp*)work;
    
    // 成功完成初始化，返回 1
    return 1;
    
    error:
    // 如果发生错误，输出跟踪信息
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    
    // 释放已分配的内存空间
    free(mem_buff);
    free(mem_buff2);
    
    // 将参数结构体清零，以便在错误情况下的状态重置
    memset(params, 0, sizeof(*params));
    
    // 返回 0，表示初始化失败
    return 0;
    }
static inline int
init_geqrf(GEQRF_PARAMS_t<fortran_doublecomplex> *params,
                   fortran_int m,
                   fortran_int n)
{
    // 定义指向 GEQRF 参数结构体的指针，初始化为 NULL
    using ftyp = fortran_doublecomplex;
    
    // 初始化内存缓冲区指针
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *tau, *work;
    
    // 计算最小的 m 和 n
    fortran_int min_m_n = fortran_int_min(m, n);
    // 将 min_m_n 转换为 size_t 类型
    size_t safe_min_m_n = min_m_n;
    // 将 m 和 n 转换为 size_t 类型
    size_t safe_m = m;
    size_t safe_n = n;

    // 计算 A 和 TAU 的大小
    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t tau_size = safe_min_m_n * sizeof(ftyp);

    // 声明变量用于存储工作区的大小和个数
    fortran_int work_count;
    size_t work_size;
    // 计算 lda，确保不小于 1
    fortran_int lda = fortran_int_max(1, m);

    // 分配内存并检查分配情况
    mem_buff = (npy_uint8 *)malloc(a_size + tau_size);
    if (!mem_buff)
        goto error;

    // 设置 a 指针和 tau 指针
    a = mem_buff;
    tau = a + a_size;
    // 将 tau 内存块清零
    memset(tau, 0, tau_size);

    // 初始化 GEQRF_PARAMS_t 结构体的参数
    params->M = m;
    params->N = n;
    params->A = (ftyp*)a;
    params->TAU = (ftyp*)tau;
    params->LDA = lda;

    {
        /* 计算最优工作区大小 */

        // 声明一个用于查询工作区大小的变量
        ftyp work_size_query;

        // 设置 params 的 WORK 指针为 work_size_query 的地址，并设置 LWORK 为 -1
        params->WORK = &work_size_query;
        params->LWORK = -1;

        // 调用 call_geqrf 函数进行计算，并检查返回值
        if (call_geqrf(params) != 0)
            goto error;

        // 将计算得到的工作区大小转换为 fortran_int 类型
        work_count = (fortran_int) ((ftyp*)params->WORK)->r;
    }

    // 设置 LWORK 为 n 和 work_count 中的较大值
    params->LWORK = fortran_int_max(fortran_int_max(1, n),
                                    work_count);

    // 计算工作区的实际大小
    work_size = (size_t) params->LWORK * sizeof(ftyp);

    // 分配内存并检查分配情况
    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2)
        goto error;

    // 设置 work 指针
    work = mem_buff2;

    // 将 params 的 WORK 指针设置为 work 的地址
    params->WORK = (ftyp*)work;

    // 返回成功状态
    return 1;

error:
    // 打印错误消息并释放内存
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    // 将 params 结构体清零
    memset(params, 0, sizeof(*params));

    // 返回失败状态
    return 0;
}

template<typename ftyp>
static inline void
release_geqrf(GEQRF_PARAMS_t<ftyp>* params)
{
    // 释放 A 和 WORK 所指向的内存块
    free(params->A);
    free(params->WORK);
    // 将 params 结构体清零
    memset(params, 0, sizeof(*params));
}

template<typename typ>
static void
qr_r_raw(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
    // 定义 ftyp 为 typ 对应的 fortran 类型
    using ftyp = fortran_type_t<typ>;

    // 声明 GEQRF_PARAMS_t 结构体的变量 params
    GEQRF_PARAMS_t<ftyp> params;
    // 声明变量用于指示是否发生错误
    int error_occurred = get_fp_invalid_and_clear();
    // 声明变量 n 和 m
    fortran_int n, m;

    // 初始化外层循环，由外部定义
    INIT_OUTER_LOOP_2

    // 从 dimensions 数组中获取 m 和 n
    m = (fortran_int)dimensions[0];
    n = (fortran_int)dimensions[1];

    // 调用 init_geqrf 函数初始化 params，如果初始化成功则执行以下代码块
    if (init_geqrf(&params, m, n)) {

        // 初始化线性化数据结构 a_in 和 tau_out
        linearize_data a_in = init_linearize_data(n, m, steps[1], steps[0]);
        linearize_data tau_out = init_linearize_data(1, fortran_int_min(m, n), 1, steps[2]);

        // 开始外层循环
        BEGIN_OUTER_LOOP_2
            int not_ok;
            // 线性化矩阵数据并调用 GEQRF 函数
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            not_ok = call_geqrf(&params);
            // 如果执行成功，则反线性化矩阵数据
            if (!not_ok) {
                delinearize_matrix((typ*)args[0], (typ*)params.A, &a_in);
                delinearize_matrix((typ*)args[1], (typ*)params.TAU, &tau_out);
            } else {
                // 如果执行失败，设置错误标志，并用 NaN 填充矩阵
                error_occurred = 1;
                nan_matrix((typ*)args[1], &tau_out);
            }
        // 结束外层循环
        END_OUTER_LOOP

        // 释放 params 所指向的内存块
        release_geqrf(&params);
    }
}
    }



    // 这里是一个函数或者代码块的结束
    // 没有具体代码行为展示，可能是函数的最后一行或者是一个条件语句的结尾
    // 程序员用大括号来标识代码块的开始和结束
    // 这里的大括号可能是 if、for、while 或者函数定义的一部分



    set_fp_invalid_or_clear(error_occurred);



    // 调用函数 set_fp_invalid_or_clear，传递参数 error_occurred
    // 函数的具体作用可以根据上下文来理解，它可能用于设置浮点处理器状态为无效或者清除错误标志
    // error_occurred 可能是一个布尔变量或者整数，用来指示是否发生了错误
/* -------------------------------------------------------------------------- */
                 /* qr common code (modes - reduced and complete) */

template<typename typ>
struct GQR_PARAMS_t
{
    fortran_int M;             // Number of rows in matrix A
    fortran_int MC;            // Column dimension of the input matrix Q
    fortran_int MN;            // Minimum of M and N
    void* A;                   // Pointer to matrix A
    typ *Q;                    // Pointer to matrix Q
    fortran_int LDA;           // Leading dimension of A
    typ* TAU;                  // Scalar factors of the elementary reflectors
    typ *WORK;                 // Workspace
    fortran_int LWORK;         // Length of the workspace array
} ;

static inline fortran_int
call_gqr(GQR_PARAMS_t<double> *params)
{
    fortran_int rv;
    LAPACK(dorgqr)(&params->M, &params->MC, &params->MN,
                          params->Q, &params->LDA,
                          params->TAU,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}

static inline fortran_int
call_gqr(GQR_PARAMS_t<f2c_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zungqr)(&params->M, &params->MC, &params->MN,
                          params->Q, &params->LDA,
                          params->TAU,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}

static inline int
init_gqr_common(GQR_PARAMS_t<fortran_doublereal> *params,
                          fortran_int m,
                          fortran_int n,
                          fortran_int mc)
{
    using ftyp = fortran_doublereal;
    npy_uint8 *mem_buff = NULL;   // Memory buffer for workspace and matrices
    npy_uint8 *mem_buff2 = NULL;  // Additional memory buffer for workspace
    npy_uint8 *a, *q, *tau, *work; // Pointers to different parts of mem_buff
    fortran_int min_m_n = fortran_int_min(m, n);  // Minimum of M and N
    size_t safe_mc = mc;          // Column dimension of Q, safely cast to size_t
    size_t safe_min_m_n = min_m_n; // Safely cast min_m_n to size_t
    size_t safe_m = m;            // Safely cast m to size_t
    size_t safe_n = n;            // Safely cast n to size_t
    size_t a_size = safe_m * safe_n * sizeof(ftyp); // Size of matrix A in bytes
    size_t q_size = safe_m * safe_mc * sizeof(ftyp); // Size of matrix Q in bytes
    size_t tau_size = safe_min_m_n * sizeof(ftyp);   // Size of tau vector in bytes

    fortran_int work_count;       // Size of workspace as returned by LAPACK
    size_t work_size;             // Size of workspace in bytes
    fortran_int lda = fortran_int_max(1, m);  // Leading dimension of A

    mem_buff = (npy_uint8 *)malloc(q_size + tau_size + a_size); // Allocate memory for Q, tau, and A

    if (!mem_buff)
        goto error;

    q = mem_buff;                 // Q starts at the beginning of mem_buff
    tau = q + q_size;             // tau follows Q in memory
    a = tau + tau_size;           // A follows tau in memory

    params->M = m;                // Set struct members based on input parameters
    params->MC = mc;
    params->MN = min_m_n;
    params->A = a;
    params->Q = (ftyp*)q;
    params->TAU = (ftyp*)tau;
    params->LDA = lda;

    {
        /* compute optimal work size */
        ftyp work_size_query;    // Temporary variable to query workspace size

        params->WORK = &work_size_query;  // Set WORK to point to work_size_query
        params->LWORK = -1;               // Set LWORK to -1 to query optimal workspace size

        if (call_gqr(params) != 0)        // Call LAPACK function to query workspace size
            goto error;

        work_count = (fortran_int) *(ftyp*) params->WORK;  // Extract workspace size from WORK
    }

    params->LWORK = fortran_int_max(fortran_int_max(1, n), work_count); // Set LWORK to the maximum needed workspace size

    work_size = (size_t) params->LWORK * sizeof(ftyp);  // Calculate total workspace size in bytes

    mem_buff2 = (npy_uint8 *)malloc(work_size);   // Allocate additional memory for workspace

    if (!mem_buff2)
        goto error;

    work = mem_buff2;             // Set work pointer to point to the allocated workspace

    params->WORK = (ftyp*)work;   // Set params->WORK to point to the allocated workspace

    return 1;  // Return success
 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);  // Log initialization failure
    free(mem_buff);               // Free memory buffers
    free(mem_buff2);
    memset(params, 0, sizeof(*params));  // Clear params structure

    return 0;  // Return failure
}
/*
初始化通用 GQR 参数，使用指定的参数和尺寸。

参数说明：
- params: GQR 参数结构体指针，用于存储初始化后的参数
- m: 矩阵维度 m
- n: 矩阵维度 n
- mc: 矩阵列维度，通常为 m 和 n 中较小的一个

返回值：
- 返回 1 表示初始化成功，返回 0 表示初始化失败

注意事项：
- 函数内部会动态分配内存，需要在使用后手动释放以避免内存泄漏
- 初始化过程中若调用 call_gqr 函数失败会立即返回并释放之前分配的内存
*/
init_gqr_common(GQR_PARAMS_t<fortran_doublecomplex> *params,
                          fortran_int m,
                          fortran_int n,
                          fortran_int mc)
{
    using ftyp=fortran_doublecomplex;
    npy_uint8 *mem_buff = NULL;  // 初始化内存缓冲区指针
    npy_uint8 *mem_buff2 = NULL;  // 初始化第二个内存缓冲区指针
    npy_uint8 *a, *q, *tau, *work;  // 定义指向不同数据区域的指针
    fortran_int min_m_n = fortran_int_min(m, n);  // 计算 m 和 n 中的较小值
    size_t safe_mc = mc;  // 将 mc 转为 size_t 类型的安全变量
    size_t safe_min_m_n = min_m_n;  // 将 min_m_n 转为 size_t 类型的安全变量
    size_t safe_m = m;  // 将 m 转为 size_t 类型的安全变量
    size_t safe_n = n;  // 将 n 转为 size_t 类型的安全变量

    size_t a_size = safe_m * safe_n * sizeof(ftyp);  // 计算 A 数据区域的大小
    size_t q_size = safe_m * safe_mc * sizeof(ftyp);  // 计算 Q 数据区域的大小
    size_t tau_size = safe_min_m_n * sizeof(ftyp);  // 计算 TAU 数据区域的大小

    fortran_int work_count;  // 工作区域计数变量
    size_t work_size;  // 工作区域的大小变量
    fortran_int lda = fortran_int_max(1, m);  // 计算 lda，即 m 和 1 中的较大值

    mem_buff = (npy_uint8 *)malloc(q_size + tau_size + a_size);  // 分配内存用于 Q、TAU 和 A 数据区域

    if (!mem_buff)
        goto error;  // 内存分配失败，跳转到 error 标签处

    q = mem_buff;  // 设置 Q 指针
    tau = q + q_size;  // 设置 TAU 指针
    a = tau + tau_size;  // 设置 A 指针

    params->M = m;  // 设置参数结构体中的 M 字段
    params->MC = mc;  // 设置参数结构体中的 MC 字段
    params->MN = min_m_n;  // 设置参数结构体中的 MN 字段
    params->A = a;  // 设置参数结构体中的 A 指针
    params->Q = (ftyp*)q;  // 设置参数结构体中的 Q 指针
    params->TAU = (ftyp*)tau;  // 设置参数结构体中的 TAU 指针
    params->LDA = lda;  // 设置参数结构体中的 LDA 字段

    {
        /* 计算最优工作区域大小 */
        ftyp work_size_query;

        params->WORK = &work_size_query;  // 设置 WORK 指针为工作区域大小查询变量的地址
        params->LWORK = -1;  // 设置 LWORK 为 -1，表示进行大小查询

        if (call_gqr(params) != 0)  // 调用 call_gqr 函数进行工作区域大小查询，如果失败则跳转到 error 标签处
            goto error;

        work_count = (fortran_int) ((ftyp*)params->WORK)->r;  // 获取工作区域的大小
    }

    params->LWORK = fortran_int_max(fortran_int_max(1, n),
                                    work_count);  // 设置 LWORK 为 n 和工作区域大小中的较大值

    work_size = (size_t) params->LWORK * sizeof(ftyp);  // 计算实际工作区域的大小

    mem_buff2 = (npy_uint8 *)malloc(work_size);  // 分配内存用于工作区域

    if (!mem_buff2)
        goto error;  // 内存分配失败，跳转到 error 标签处

    work = mem_buff2;  // 设置工作区域的指针

    params->WORK = (ftyp*)work;  // 设置参数结构体中的 WORK 指针为工作区域的地址
    params->LWORK = work_count;  // 设置参数结构体中的 LWORK 为实际工作区域的大小

    return 1;  // 初始化成功，返回 1

error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);  // 输出初始化失败的错误信息
    free(mem_buff);  // 释放分配的内存
    free(mem_buff2);  // 释放分配的内存
    memset(params, 0, sizeof(*params));  // 将 params 结构体清零

    return 0;  // 初始化失败，返回 0
}

/* -------------------------------------------------------------------------- */
                 /* qr (modes - reduced) */

/*
以文本形式输出 GQR 参数结构体中的内容。

参数说明：
- name: 输出的名称字符串
- params: GQR 参数结构体指针，包含要输出的数据

注意事项：
- 输出格式固定，包含 Q、TAU、WORK、M、MC、MN、LDA、LWORK 等信息
*/
template<typename typ>
static inline void
dump_gqr_params(const char *name,
                GQR_PARAMS_t<typ> *params)
{
    TRACE_TXT("\n%s:\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n",

              name,

              "Q", params->Q,
              "TAU", params->TAU,
              "WORK", params->WORK,

              "M", (int)params->M,
              "MC", (int)params->MC,
              "MN", (int)params->MN,
              "LDA", (int)params->LDA,
              "LWORK", (int)params->LWORK);
}

/*
初始化 GQR 参数结构体，简化接口。

参数说明：
- params: GQR 参数结构体指针，用于存储初始化后的参数
- m: 矩阵维度 m
- n: 矩阵维度 n

返回值：
- 返回初始化结果，调用 init_gqr_common 函数进行初始化
*/
template<typename ftyp>
static inline int
init_gqr(GQR_PARAMS_t<ftyp> *params,
                   fortran_int m,
                   fortran_int n)
{
    return init_gqr_common(
        params, m, n,
        fortran_int_min(m, n));  // 调用通用初始化函数，并传入 m、n 和它们的最小值作为参数
}
/* 释放 GQR_PARAMS_t 结构体中动态分配的 Q 和 WORK 内存块，并将 params 结构体清零 */
release_gqr(GQR_PARAMS_t<typ>* params)
{
    free(params->Q);  // 释放 Q 指向的内存块
    free(params->WORK);  // 释放 WORK 指向的内存块
    memset(params, 0, sizeof(*params));  // 将 params 结构体清零
}

/* 使用模板 typename typ 实例化的静态函数，执行 QR 分解 */
template<typename typ>
static void
qr_reduced(char **args, npy_intp const *dimensions, npy_intp const *steps,
                  void *NPY_UNUSED(func))
{
    using ftyp = fortran_type_t<typ>;  // 定义 fortran 类型别名 ftyp

    GQR_PARAMS_t<ftyp> params;  // 声明 GQR_PARAMS_t 结构体对象 params
    int error_occurred = get_fp_invalid_and_clear();  // 获取并清除浮点错误标志
    fortran_int n, m;  // 声明 fortran 整型变量 n 和 m

    INIT_OUTER_LOOP_3  // 宏定义的外部循环初始化

    m = (fortran_int)dimensions[0];  // 从 dimensions 数组获取 m
    n = (fortran_int)dimensions[1];  // 从 dimensions 数组获取 n

    // 如果初始化 GQR_PARAMS_t 结构体失败，则跳过 QR 分解
    if (init_gqr(&params, m, n)) {
        // 初始化线性化数据结构
        linearize_data a_in = init_linearize_data(n, m, steps[1], steps[0]);
        linearize_data tau_in = init_linearize_data(1, fortran_int_min(m, n), 1, steps[2]);
        linearize_data q_out = init_linearize_data(fortran_int_min(m, n), m, steps[4], steps[3]);

        BEGIN_OUTER_LOOP_3  // 外部循环开始

            int not_ok;  // 定义整型变量 not_ok
            // 将 args[0] 中的数据线性化到 params.A 中
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            // 将 args[0] 中的数据线性化到 params.Q 中
            linearize_matrix((typ*)params.Q, (typ*)args[0], &a_in);
            // 将 args[1] 中的数据线性化到 params.TAU 中
            linearize_matrix((typ*)params.TAU, (typ*)args[1], &tau_in);
            // 调用 GQR 过程，将结果保存到 not_ok 中
            not_ok = call_gqr(&params);

            // 如果 GQR 过程执行成功
            if (!not_ok) {
                // 将 params.Q 的结果反线性化到 args[2] 中
                delinearize_matrix((typ*)args[2], (typ*)params.Q, &q_out);
            } else {
                // 如果 GQR 过程执行失败，标记错误，并将 args[2] 矩阵置为 NaN
                error_occurred = 1;
                nan_matrix((typ*)args[2], &q_out);
            }

        END_OUTER_LOOP  // 外部循环结束

        // 释放 GQR_PARAMS_t 结构体中的资源
        release_gqr(&params);
    }

    // 设置浮点错误标志
    set_fp_invalid_or_clear(error_occurred);
}

/* 初始化 GQR_PARAMS_t 结构体用于 QR 完整模式 */
template<typename ftyp>
static inline int
init_gqr_complete(GQR_PARAMS_t<ftyp> *params,
                            fortran_int m,
                            fortran_int n)
{
    return init_gqr_common(params, m, n, m);  // 调用通用初始化函数
}

/* QR 完整模式的模板函数 */
template<typename typ>
static void
qr_complete(char **args, npy_intp const *dimensions, npy_intp const *steps,
                  void *NPY_UNUSED(func))
{
    using ftyp = fortran_type_t<typ>;  // 定义 fortran 类型别名 ftyp

    GQR_PARAMS_t<ftyp> params;  // 声明 GQR_PARAMS_t 结构体对象 params
    int error_occurred = get_fp_invalid_and_clear();  // 获取并清除浮点错误标志
    fortran_int n, m;  // 声明 fortran 整型变量 n 和 m

    INIT_OUTER_LOOP_3  // 宏定义的外部循环初始化

    m = (fortran_int)dimensions[0];  // 从 dimensions 数组获取 m
    n = (fortran_int)dimensions[1];  // 从 dimensions 数组获取 n
    # 检查是否成功初始化了 GQR 参数结构，并进行相应操作
    if (init_gqr_complete(&params, m, n)) {
        # 初始化将被线性化的输入矩阵数据结构 a_in
        linearize_data a_in = init_linearize_data(n, m, steps[1], steps[0]);
        # 初始化将被线性化的输入矩阵数据结构 tau_in
        linearize_data tau_in = init_linearize_data(1, fortran_int_min(m, n), 1, steps[2]);
        # 初始化将被线性化的输出矩阵数据结构 q_out
        linearize_data q_out = init_linearize_data(m, m, steps[4], steps[3]);

        # 开始外层循环，具体实现可能被宏定义或者内联代码所替换
        BEGIN_OUTER_LOOP_3
            # 定义变量 not_ok，用于记录 GQR 调用是否成功
            int not_ok;
            # 将 params.A 矩阵线性化到 args[0]，使用 a_in 数据结构描述
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            # 将 params.Q 矩阵线性化到 args[0]，使用 a_in 数据结构描述
            linearize_matrix((typ*)params.Q, (typ*)args[0], &a_in);
            # 将 params.TAU 矩阵线性化到 args[1]，使用 tau_in 数据结构描述
            linearize_matrix((typ*)params.TAU, (typ*)args[1], &tau_in);
            # 调用 GQR 算法，返回值表示是否发生错误
            not_ok = call_gqr(&params);
            # 如果 GQR 调用成功
            if (!not_ok) {
                # 将 params.Q 矩阵反线性化到 args[2]，使用 q_out 数据结构描述
                delinearize_matrix((typ*)args[2], (typ*)params.Q, &q_out);
            } else {
                # 如果 GQR 调用失败，标记错误发生并将 args[2] 矩阵设置为 NaN
                error_occurred = 1;
                nan_matrix((typ*)args[2], &q_out);
            }
        # 结束外层循环
        END_OUTER_LOOP

        # 释放 GQR 参数结构占用的资源
        release_gqr(&params);
    }

    # 根据错误标记设置浮点操作的无效状态或清除错误状态
    set_fp_invalid_or_clear(error_occurred);
/* -------------------------------------------------------------------------- */
/* least squares */

/* 定义模板结构体 GELSD_PARAMS_t，用于存储调用最小二乘求解函数参数 */
template<typename typ>
struct GELSD_PARAMS_t
{
    fortran_int M;         // 行数
    fortran_int N;         // 列数
    fortran_int NRHS;      // 右侧矩阵的列数
    typ *A;                // 输入矩阵 A
    fortran_int LDA;       // A 的列数
    typ *B;                // 输出矩阵 B
    fortran_int LDB;       // B 的列数
    basetype_t<typ> *S;    // 奇异值数组
    basetype_t<typ> *RCOND;// 条件数的阈值
    fortran_int RANK;      // 矩阵的秩
    typ *WORK;             // 工作空间数组
    fortran_int LWORK;     // 工作空间的大小
    basetype_t<typ> *RWORK;// 实数工作空间
    fortran_int *IWORK;    // 整数工作空间
};

/* 打印输出 GELSD_PARAMS_t 结构体的内容 */
template<typename typ>
static inline void
dump_gelsd_params(const char *name,
                  GELSD_PARAMS_t<typ> *params)
{
    TRACE_TXT("\n%s:\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18p\n",
              name,
              "A", params->A,
              "B", params->B,
              "S", params->S,
              "WORK", params->WORK,
              "RWORK", params->RWORK,
              "IWORK", params->IWORK,
              "M", (int)params->M,
              "N", (int)params->N,
              "NRHS", (int)params->NRHS,
              "LDA", (int)params->LDA,
              "LDB", (int)params->LDB,
              "LWORK", (int)params->LWORK,
              "RANK", (int)params->RANK,
              "RCOND", params->RCOND);
}

/* 调用 LAPACK 库中的 sgelsd 函数解决最小二乘问题（单精度实数版本） */
static inline fortran_int
call_gelsd(GELSD_PARAMS_t<fortran_real> *params)
{
    fortran_int rv;
    LAPACK(sgelsd)(&params->M, &params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->B, &params->LDB,
                          params->S,
                          params->RCOND, &params->RANK,
                          params->WORK, &params->LWORK,
                          params->IWORK,
                          &rv);
    return rv;
}

/* 调用 LAPACK 库中的 dgelsd 函数解决最小二乘问题（双精度实数版本） */
static inline fortran_int
call_gelsd(GELSD_PARAMS_t<fortran_doublereal> *params)
{
    fortran_int rv;
    LAPACK(dgelsd)(&params->M, &params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->B, &params->LDB,
                          params->S,
                          params->RCOND, &params->RANK,
                          params->WORK, &params->LWORK,
                          params->IWORK,
                          &rv);
    return rv;
}

/* 初始化 GELSD_PARAMS_t 结构体，准备调用最小二乘求解函数 */
template<typename ftyp>
static inline int
init_gelsd(GELSD_PARAMS_t<ftyp> *params,
                   fortran_int m,
                   fortran_int n,
                   fortran_int nrhs,
                   scalar_trait)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *s, *work, *iwork;
    fortran_int min_m_n = fortran_int_min(m, n);
    fortran_int max_m_n = fortran_int_max(m, n);
    # 复制最小和最大的m和n值到安全变量中
    size_t safe_min_m_n = min_m_n;
    size_t safe_max_m_n = max_m_n;
    size_t safe_m = m;
    size_t safe_n = n;
    size_t safe_nrhs = nrhs;

    # 计算数组a、b和s的内存大小，分别用于存储矩阵A、矩阵B和求解器
    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t b_size = safe_max_m_n * safe_nrhs * sizeof(ftyp);
    size_t s_size = safe_min_m_n * sizeof(ftyp);

    # 初始化工作区大小和工作数组的计数
    fortran_int work_count;
    size_t work_size;
    size_t iwork_size;

    # 确定矩阵A和矩阵B的列主元大小
    fortran_int lda = fortran_int_max(1, m);
    fortran_int ldb = fortran_int_max(1, fortran_int_max(m,n));

    # 计算总内存大小，分配内存缓冲区
    size_t msize = a_size + b_size + s_size;
    mem_buff = (npy_uint8 *)malloc(msize != 0 ? msize : 1);

    # 检查内存分配是否成功，若失败则跳转到no_memory标签处理
    if (!mem_buff) {
        goto no_memory;
    }

    # 设置数组a、b和s的起始位置
    a = mem_buff;
    b = a + a_size;
    s = b + b_size;

    # 将参数结构体params中的值设置为当前矩阵和数组的尺寸和位置信息
    params->M = m;
    params->N = n;
    params->NRHS = nrhs;
    params->A = (ftyp*)a;
    params->B = (ftyp*)b;
    params->S = (ftyp*)s;
    params->LDA = lda;
    params->LDB = ldb;

    {
        /* 计算最优工作区大小 */
        ftyp work_size_query;
        fortran_int iwork_size_query;

        # 设置params结构体以查询工作区大小
        params->WORK = &work_size_query;
        params->IWORK = &iwork_size_query;
        params->RWORK = NULL;
        params->LWORK = -1;

        # 调用函数call_gelsd获取工作区大小
        if (call_gelsd(params) != 0) {
            goto error;
        }
        work_count = (fortran_int)work_size_query;

        # 计算实际工作区和整数工作区的内存大小
        work_size  = (size_t) work_size_query * sizeof(ftyp);
        iwork_size = (size_t)iwork_size_query * sizeof(fortran_int);
    }

    # 分配工作区和整数工作区的内存
    mem_buff2 = (npy_uint8 *)malloc(work_size + iwork_size);
    if (!mem_buff2) {
        goto no_memory;
    }
    work = mem_buff2;
    iwork = work + work_size;

    # 设置params结构体中的工作区指针和工作区大小
    params->WORK = (ftyp*)work;
    params->RWORK = NULL;
    params->IWORK = (fortran_int*)iwork;
    params->LWORK = work_count;

    # 返回成功标志
    return 1;

 no_memory:
    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API;
    PyErr_NoMemory();
    NPY_DISABLE_C_API;

 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));
    return 0;
}

static inline fortran_int
call_gelsd(GELSD_PARAMS_t<fortran_complex> *params)
{
    // 声明返回值变量
    fortran_int rv;
    // 调用 LAPACK 库中的复数版本的 GELSD 函数
    LAPACK(cgelsd)(&params->M, &params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->B, &params->LDB,
                          params->S,
                          params->RCOND, &params->RANK,
                          params->WORK, &params->LWORK,
                          params->RWORK, (fortran_int*)params->IWORK,
                          &rv);
    // 返回 LAPACK 函数的返回值
    return rv;
}

static inline fortran_int
call_gelsd(GELSD_PARAMS_t<fortran_doublecomplex> *params)
{
    // 声明返回值变量
    fortran_int rv;
    // 调用 LAPACK 库中的双精度复数版本的 GELSD 函数
    LAPACK(zgelsd)(&params->M, &params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->B, &params->LDB,
                          params->S,
                          params->RCOND, &params->RANK,
                          params->WORK, &params->LWORK,
                          params->RWORK, (fortran_int*)params->IWORK,
                          &rv);
    // 返回 LAPACK 函数的返回值
    return rv;
}


template<typename ftyp>
static inline int
init_gelsd(GELSD_PARAMS_t<ftyp> *params,
                   fortran_int m,
                   fortran_int n,
                   fortran_int nrhs,
                   complex_trait)
{
    // 使用模板类型确定实数类型
    using frealtyp = basetype_t<ftyp>;
    // 初始化指针变量
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *s, *work, *iwork, *rwork;
    // 计算最小值和最大值
    fortran_int min_m_n = fortran_int_min(m, n);
    fortran_int max_m_n = fortran_int_max(m, n);
    // 安全转换为大小类型
    size_t safe_min_m_n = min_m_n;
    size_t safe_max_m_n = max_m_n;
    size_t safe_m = m;
    size_t safe_n = n;
    size_t safe_nrhs = nrhs;

    // 计算各个缓冲区的大小
    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t b_size = safe_max_m_n * safe_nrhs * sizeof(ftyp);
    size_t s_size = safe_min_m_n * sizeof(frealtyp);

    // 初始化工作变量的计数和大小
    fortran_int work_count;
    size_t work_size, rwork_size, iwork_size;
    // 计算 LDA 和 LDB 的值
    fortran_int lda = fortran_int_max(1, m);
    fortran_int ldb = fortran_int_max(1, fortran_int_max(m,n));

    // 计算总内存大小并分配内存
    size_t msize = a_size + b_size + s_size;
    mem_buff = (npy_uint8 *)malloc(msize != 0 ? msize : 1);

    // 内存分配失败处理
    if (!mem_buff) {
        goto no_memory;
    }

    // 设置缓冲区指针
    a = mem_buff;
    b = a + a_size;
    s = b + b_size;

    // 初始化参数结构体的各个字段
    params->M = m;
    params->N = n;
    params->NRHS = nrhs;
    params->A = (ftyp*)a;
    params->B = (ftyp*)b;
    params->S = (frealtyp*)s;
    params->LDA = lda;
    params->LDB = ldb;
    {
        /* 计算最优工作大小 */
        ftyp work_size_query;  // 定义工作大小查询变量
        frealtyp rwork_size_query;  // 定义实数工作大小查询变量
        fortran_int iwork_size_query;  // 定义整数工作大小查询变量

        params->WORK = &work_size_query;  // 设置参数结构体中的 WORK 指针
        params->IWORK = &iwork_size_query;  // 设置参数结构体中的 IWORK 指针
        params->RWORK = &rwork_size_query;  // 设置参数结构体中的 RWORK 指针
        params->LWORK = -1;  // 设置工作大小为 -1，用于查询最优工作大小

        if (call_gelsd(params) != 0) {  // 调用函数查询最优工作大小，如果不成功则跳转到错误处理
            goto error;
        }

        work_count = (fortran_int)work_size_query.r;  // 将查询到的工作大小转换为整数存储
        work_size  = (size_t )work_size_query.r * sizeof(ftyp);  // 计算工作大小的内存空间大小
        rwork_size = (size_t)rwork_size_query * sizeof(frealtyp);  // 计算实数工作大小的内存空间大小
        iwork_size = (size_t)iwork_size_query * sizeof(fortran_int);  // 计算整数工作大小的内存空间大小
    }

    mem_buff2 = (npy_uint8 *)malloc(work_size + rwork_size + iwork_size);  // 分配工作空间内存
    if (!mem_buff2) {  // 如果内存分配失败，则跳转到无内存处理
        goto no_memory;
    }

    work = mem_buff2;  // 将分配的内存空间指针赋值给工作指针
    rwork = work + work_size;  // 计算实数工作空间指针
    iwork = rwork + rwork_size;  // 计算整数工作空间指针

    params->WORK = (ftyp*)work;  // 设置参数结构体中的 WORK 指针
    params->RWORK = (frealtyp*)rwork;  // 设置参数结构体中的 RWORK 指针
    params->IWORK = (fortran_int*)iwork;  // 设置参数结构体中的 IWORK 指针
    params->LWORK = work_count;  // 设置工作大小为查询到的最优工作大小

    return 1;  // 返回成功标识

 no_memory:
    NPY_ALLOW_C_API_DEF  // 允许调用C API宏定义
    NPY_ALLOW_C_API;  // 允许调用C API
    PyErr_NoMemory();  // 报告内存不足错误
    NPY_DISABLE_C_API;  // 禁用调用C API

 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);  // 输出错误信息到日志中
    free(mem_buff);  // 释放之前分配的内存
    free(mem_buff2);  // 释放工作空间内存
    memset(params, 0, sizeof(*params));  // 将参数结构体的内容清零

    return 0;  // 返回失败标识
}

/** 
 * 释放 GELSD 参数结构体中的内存资源
 * @param params GELSD 参数结构体指针
 */
template<typename ftyp>
static inline void
release_gelsd(GELSD_PARAMS_t<ftyp>* params)
{
    /* A and WORK contain allocated blocks */
    // 释放 params 结构体中 A 成员指向的内存块
    free(params->A);
    // 释放 params 结构体中 WORK 成员指向的内存块
    free(params->WORK);
    // 将 params 结构体清零，大小为其本身的大小
    memset(params, 0, sizeof(*params));
}

/** 
 * 计算连续向量的平方 L2 范数
 * @tparam typ 向量元素类型
 * @param p 指向向量的指针
 * @param n 向量的长度
 * @param scalar_trait 标量特性类型
 * @return 返回平方 L2 范数的结果
 */
template<typename typ>
static basetype_t<typ>
abs2(typ *p, npy_intp n, scalar_trait) {
    npy_intp i;
    basetype_t<typ> res = 0;
    for (i = 0; i < n; i++) {
        typ el = p[i];
        res += el*el;
    }
    return res;
}

/** 
 * 计算复数向量的平方 L2 范数
 * @tparam typ 向量元素类型
 * @param p 指向向量的指针
 * @param n 向量的长度
 * @param complex_trait 复数特性类型
 * @return 返回平方 L2 范数的结果
 */
template<typename typ>
static basetype_t<typ>
abs2(typ *p, npy_intp n, complex_trait) {
    npy_intp i;
    basetype_t<typ> res = 0;
    for (i = 0; i < n; i++) {
        typ el = p[i];
        res += RE(&el)*RE(&el) + IM(&el)*IM(&el);
    }
    return res;
}

/** 
 * 最小二乘法求解函数
 * @tparam typ 求解的数据类型
 * @param args 参数列表
 * @param dimensions 数组维度信息
 * @param steps 步幅信息
 * @param NPY_UNUSED(func) 未使用的函数参数
 */
template<typename typ>
static void
lstsq(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func))
{
using ftyp = fortran_type_t<typ>;
using basetyp = basetype_t<typ>;
    GELSD_PARAMS_t<ftyp> params;
    int error_occurred = get_fp_invalid_and_clear();
    fortran_int n, m, nrhs;
    fortran_int excess;

    INIT_OUTER_LOOP_7

    // 获取维度信息
    m = (fortran_int)dimensions[0];
    n = (fortran_int)dimensions[1];
    nrhs = (fortran_int)dimensions[2];
    excess = m - n;

    // 初始化 GELSD 参数结构体
    if (init_gelsd(&params, m, n, nrhs, dispatch_scalar<ftyp>{})) {
        // 初始化线性化数据结构体
        linearize_data a_in = init_linearize_data(n, m, steps[1], steps[0]);
        linearize_data b_in = init_linearize_data_ex(nrhs, m, steps[3], steps[2], fortran_int_max(n, m));
        linearize_data x_out = init_linearize_data_ex(nrhs, n, steps[5], steps[4], fortran_int_max(n, m));
        linearize_data r_out = init_linearize_data(1, nrhs, 1, steps[6]);
        linearize_data s_out = init_linearize_data(1, fortran_int_min(n, m), 1, steps[7]);

        BEGIN_OUTER_LOOP_7
            int not_ok;
            // 线性化矩阵数据
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            linearize_matrix((typ*)params.B, (typ*)args[1], &b_in);
            // 设置 RCOND 参数
            params.RCOND = (basetyp*)args[2];
            // 调用 GELSD 函数
            not_ok = call_gelsd(&params);
            if (!not_ok) {
                // 反线性化矩阵数据
                delinearize_matrix((typ*)args[3], (typ*)params.B, &x_out);
                *(npy_int*) args[5] = params.RANK;
                delinearize_matrix((basetyp*)args[6], (basetyp*)params.S, &s_out);

                /* Note that linalg.lstsq discards this when excess == 0 */
                // 当 excess >= 0 且 params.RANK == n 时，计算残差的平方和
                if (excess >= 0 && params.RANK == n) {
                    /* Compute the residuals as the square sum of each column */
                    int i;
                    char *resid = args[4];
                    ftyp *components = (ftyp *)params.B + n;
                    for (i = 0; i < nrhs; i++) {
                        ftyp *vector = components + i*m;
                        /* Numpy and fortran floating types are the same size,
                         * so this cast is safe */
                        // 计算绝对值的平方
                        basetyp abs = abs2((typ *)vector, excess,
/* -------------------------------------------------------------------------- */
/* gufunc registration  */

static void *array_of_nulls[] = {
    (void *)NULL,  // 初始化一个包含空指针的数组，长度为16，用于后续的初始化
    (void *)NULL,
    (void *)NULL,
    (void *)NULL,

    (void *)NULL,
    (void *)NULL,
    (void *)NULL,
    (void *)NULL,

    (void *)NULL,
    (void *)NULL,
    (void *)NULL,
    (void *)NULL,

    (void *)NULL,
    (void *)NULL,
    (void *)NULL,
    (void *)NULL
};

#define FUNC_ARRAY_NAME(NAME) NAME ## _funcs

#define GUFUNC_FUNC_ARRAY_REAL(NAME)                    \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        FLOAT_ ## NAME,                                 // 定义一个包含不同函数指针的数组，用于注册 gufunc 的实数版本
        DOUBLE_ ## NAME                                 // 包含 FLOAT_NAME 和 DOUBLE_NAME 函数指针
    }

#define GUFUNC_FUNC_ARRAY_REAL_COMPLEX(NAME)            \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        FLOAT_ ## NAME,                                 // 定义一个包含不同函数指针的数组，用于注册 gufunc 的实数和复数版本
        DOUBLE_ ## NAME,                                // 包含 FLOAT_NAME、DOUBLE_NAME、CFLOAT_NAME、CDOUBLE_NAME 函数指针
        CFLOAT_ ## NAME,
        CDOUBLE_ ## NAME
    }

#define GUFUNC_FUNC_ARRAY_REAL_COMPLEX_(NAME)            \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        NAME<npy_float, npy_float>,                     // 定义一个包含不同模板化函数指针的数组，用于注册 gufunc 的实数和复数版本
        NAME<npy_double, npy_double>,                   // 使用不同的模板参数进行实例化，如 npy_float 和 npy_double
        NAME<npy_cfloat, npy_float>,
        NAME<npy_cdouble, npy_double>
    }

#define GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(NAME)            \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        NAME<npy_float>,                                // 定义一个包含不同模板化函数指针的数组，用于注册 gufunc 的实数和复数版本
        NAME<npy_double>,                               // 使用不同的模板参数进行实例化，如 npy_float 和 npy_double
        NAME<npy_cfloat>,
        NAME<npy_cdouble>
    }

/* There are problems with eig in complex single precision.
 * That kernel is disabled
 */
/* 定义一个宏，用于生成通用函数数组，该数组包含不同数据类型的函数指针 */

#define GUFUNC_FUNC_ARRAY_EIG(NAME)                     \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        NAME<fortran_complex,fortran_real>,                                 \ // 为NAME宏展开创建一个函数指针，处理复数到实数
        NAME<fortran_doublecomplex,fortran_doublereal>,                                \ // 为NAME宏展开创建一个函数指针，处理双精度复数到双精度实数
        NAME<fortran_doublecomplex,fortran_doublecomplex>                                \ // 为NAME宏展开创建一个函数指针，处理双精度复数到双精度复数
    }

/* 单精度函数不会被使用，因为Python中的输入数据会被提升为双精度，
 * 所以这里没有实现它们。
 */
#define GUFUNC_FUNC_ARRAY_QR(NAME)                      \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        DOUBLE_ ## NAME,                                \ // 为NAME宏展开创建一个函数指针，处理双精度实数
        CDOUBLE_ ## NAME                                \ // 为NAME宏展开创建一个函数指针，处理双精度复数
    }

#define GUFUNC_FUNC_ARRAY_QR__(NAME)                      \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        NAME<npy_double>,                                \ // 为NAME宏展开创建一个函数指针，处理npy_double类型的数据
        NAME<npy_cdouble>                                \ // 为NAME宏展开创建一个函数指针，处理npy_cdouble类型的数据
    }


/* 下面是一系列宏的展开，为不同的数学函数名称(NAME)创建通用函数数组 */

GUFUNC_FUNC_ARRAY_REAL_COMPLEX_(slogdet); // 创建slogdet函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX_(det); // 创建det函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(eighlo); // 创建eighlo函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(eighup); // 创建eighup函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(eigvalshlo); // 创建eigvalshlo函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(eigvalshup); // 创建eigvalshup函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(solve); // 创建solve函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(solve1); // 创建solve1函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(inv); // 创建inv函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(cholesky_lo); // 创建cholesky_lo函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(cholesky_up); // 创建cholesky_up函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(svd_N); // 创建svd_N函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(svd_S); // 创建svd_S函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(svd_A); // 创建svd_A函数的通用函数数组
GUFUNC_FUNC_ARRAY_QR__(qr_r_raw); // 创建qr_r_raw函数的通用函数数组
GUFUNC_FUNC_ARRAY_QR__(qr_reduced); // 创建qr_reduced函数的通用函数数组
GUFUNC_FUNC_ARRAY_QR__(qr_complete); // 创建qr_complete函数的通用函数数组
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(lstsq); // 创建lstsq函数的通用函数数组
GUFUNC_FUNC_ARRAY_EIG(eig); // 创建eig函数的通用函数数组
GUFUNC_FUNC_ARRAY_EIG(eigvals); // 创建eigvals函数的通用函数数组

/* 定义常量数组，指定不同函数所接受的输入数据类型 */

static const char equal_2_types[] = {
    NPY_FLOAT, NPY_FLOAT, // 两个浮点数
    NPY_DOUBLE, NPY_DOUBLE, // 两个双精度浮点数
    NPY_CFLOAT, NPY_CFLOAT, // 两个复数
    NPY_CDOUBLE, NPY_CDOUBLE // 两个双精度复数
};

static const char equal_3_types[] = {
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, // 三个浮点数
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, // 三个双精度浮点数
    NPY_CFLOAT, NPY_CFLOAT, NPY_CFLOAT, // 三个复数
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE // 三个双精度复数
};

/* 第二个结果是logdet，它总是一个实数 */
static const char slogdet_types[] = {
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, // 三个浮点数
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, // 三个双精度浮点数
    NPY_CFLOAT, NPY_CFLOAT, NPY_FLOAT, // 两个复数和一个浮点数（实部为浮点数）
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_DOUBLE // 两个双精度复数和一个双精度浮点数（实部为双精度浮点数）
};

static const char eigh_types[] = {
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, // 三个浮点数
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, // 三个双精度浮点数
    NPY_CFLOAT, NPY_FLOAT, NPY_CFLOAT, // 两个复数和一个浮点数（实部为浮点数）
    NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE // 两个双精度复数和一个双精度浮点数（实部为双精度浮点数）
};

static const char eighvals_types[] = {
    NPY_FLOAT, NPY_FLOAT, // 两个浮点数
    NPY_DOUBLE, NPY_DOUBLE, // 两个双精度浮点数
    NPY_CFLOAT, NPY_FLOAT, // 一个复数和一个浮点数（实部为浮点数）
    NPY_CDOUBLE, NPY_DOUBLE // 一个双精度复数和一个双精度浮点数（实部为双精度浮点数）
};

static const char eig_types[] = {
    /* 省略了该数组的展开，需要根据实际情况补充 */
};
    # 定义了一组常量，表示不同数据类型的NumPy类型码
    NPY_FLOAT, NPY_CFLOAT, NPY_CFLOAT,
    NPY_DOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE
    # 这些常量分别代表单精度浮点数、复数浮点数、双精度浮点数、复数双精度浮点数的NumPy类型码
    # 它们用于指定和区分NumPy数组中的不同数据类型
};

// 定义一个静态常量数组，包含特征值操作的数据类型
static const char eigvals_types[] = {
    NPY_FLOAT, NPY_CFLOAT,
    NPY_DOUBLE, NPY_CDOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE
};

// 定义一个静态常量数组，包含 SVD 分解中形状为 (1,1) 的操作的数据类型
static const char svd_1_1_types[] = {
    NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_FLOAT,
    NPY_CDOUBLE, NPY_DOUBLE
};

// 定义一个静态常量数组，包含 SVD 分解中形状为 (1,3) 的操作的数据类型
static const char svd_1_3_types[] = {
    NPY_FLOAT,   NPY_FLOAT,   NPY_FLOAT,  NPY_FLOAT,
    NPY_DOUBLE,  NPY_DOUBLE,  NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT,  NPY_CFLOAT,  NPY_FLOAT,  NPY_CFLOAT,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE
};

// 定义一个静态常量数组，包含 QR 分解中形状为 (raw) 的操作的数据类型
/* A, tau */
static const char qr_r_raw_types[] = {
    NPY_DOUBLE,  NPY_DOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE,
};

// 定义一个静态常量数组，包含 QR 分解中形状为 (reduced) 的操作的数据类型
/* A, tau, q */
static const char qr_reduced_types[] = {
    NPY_DOUBLE,  NPY_DOUBLE,  NPY_DOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
};

// 定义一个静态常量数组，包含 QR 分解中形状为 (complete) 的操作的数据类型
/* A, tau, q */
static const char qr_complete_types[] = {
    NPY_DOUBLE,  NPY_DOUBLE,  NPY_DOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
};

// 定义一个静态常量数组，包含最小二乘解法的数据类型
/*  A,           b,           rcond,      x,           resid,      rank,    s,        */
static const char lstsq_types[] = {
    NPY_FLOAT,   NPY_FLOAT,   NPY_FLOAT,  NPY_FLOAT,   NPY_FLOAT,  NPY_INT, NPY_FLOAT,
    NPY_DOUBLE,  NPY_DOUBLE,  NPY_DOUBLE, NPY_DOUBLE,  NPY_DOUBLE, NPY_INT, NPY_DOUBLE,
    NPY_CFLOAT,  NPY_CFLOAT,  NPY_FLOAT,  NPY_CFLOAT,  NPY_FLOAT,  NPY_INT, NPY_FLOAT,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_INT, NPY_DOUBLE,
};

// 定义一个结构体描述符，用于描述通用函数（ufunc）
typedef struct gufunc_descriptor_struct {
    const char *name;
    const char *signature;
    const char *doc;
    int ntypes;
    int nin;
    int nout;
    PyUFuncGenericFunction *funcs;
    const char *types;
} GUFUNC_DESCRIPTOR_t;

// 定义一个数组，包含通用函数描述符结构体，描述不同的通用函数
GUFUNC_DESCRIPTOR_t gufunc_descriptors [] = {
    {
        "slogdet",
        "(m,m)->(),()",
        "slogdet on the last two dimensions and broadcast on the rest. \n"\
        "Results in two arrays, one with sign and the other with log of the"\
        " determinants. \n"\
        "    \"(m,m)->(),()\" \n",
        4, 1, 2,
        FUNC_ARRAY_NAME(slogdet),
        slogdet_types
    },
    {
        "det",
        "(m,m)->()",
        "det of the last two dimensions and broadcast on the rest. \n"\
        "    \"(m,m)->()\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(det),
        equal_2_types
    },
    {
        "eigh_lo",
        "(m,m)->(m),(m,m)",
        "eigh on the last two dimension and broadcast to the rest, using"\
        " lower triangle \n"\
        "Results in a vector of eigenvalues and a matrix with the"\
        "eigenvectors. \n"\
        "    \"(m,m)->(m),(m,m)\" \n",
        4, 1, 2,
        FUNC_ARRAY_NAME(eighlo),
        eigh_types
    },
    {
        "eigh_up",
        "(m,m)->(m),(m,m)",
        "eigh on the last two dimension and broadcast to the rest, using"\
        " upper triangle. \n"\
        "Results in a vector of eigenvalues and a matrix with the"\
        " eigenvectors. \n"\
        "    \"(m,m)->(m),(m,m)\" \n",
        4, 1, 2,
        FUNC_ARRAY_NAME(eighup),
        eigh_types
    },
    {
        "eigvalsh_lo",
        "(m,m)->(m)",
        "eigh on the last two dimension and broadcast to the rest, using"\
        " lower triangle. \n"\
        "Results in a vector of eigenvalues and a matrix with the"\
        " eigenvectors. \n"\
        "    \"(m,m)->(m)\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(eigvalshlo),
        eighvals_types
    },
    {
        "eigvalsh_up",
        "(m,m)->(m)",
        "eigvalsh on the last two dimension and broadcast to the rest,"\
        " using upper triangle. \n"\
        "Results in a vector of eigenvalues and a matrix with the"\
        " eigenvectors.\n"\
        "    \"(m,m)->(m)\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(eigvalshup),
        eighvals_types
    },
    {
        "solve",
        "(m,m),(m,n)->(m,n)",
        "solve the system a x = b, on the last two dimensions, broadcast"\
        " to the rest. \n"\
        "Results in matrices with the solutions. \n"\
        "    \"(m,m),(m,n)->(m,n)\" \n",
        4, 2, 1,
        FUNC_ARRAY_NAME(solve),
        equal_3_types
    },
    {
        "solve1",
        "(m,m),(m)->(m)",
        "solve the system a x = b, for b being a vector, broadcast in"\
        " the outer dimensions. \n"\
        "Results in vectors with the solutions. \n"\
        "    \"(m,m),(m)->(m)\" \n",
        4, 2, 1,
        FUNC_ARRAY_NAME(solve1),
        equal_3_types
    },
    {
        "inv",
        "(m, m)->(m, m)",
        "compute the inverse of the last two dimensions and broadcast"\
        " to the rest. \n"\
        "Results in the inverse matrices. \n"\
        "    \"(m,m)->(m,m)\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(inv),
        equal_2_types
    },
    {
        "cholesky_lo",
        "(m,m)->(m,m)",
        "cholesky decomposition of hermitian positive-definite matrices,\n"\
        "using lower triangle. Broadcast to all outer dimensions.\n"\
        "    \"(m,m)->(m,m)\"\n",
        4, 1, 1,
        FUNC_ARRAY_NAME(cholesky_lo),
        equal_2_types
    },
    {
        "cholesky_up",
        "(m,m)->(m,m)",
        "cholesky decomposition of hermitian positive-definite matrices,\n"\
        "using upper triangle. Broadcast to all outer dimensions.\n"\
        "    \"(m,m)->(m,m)\"\n",
        4, 1, 1,
        FUNC_ARRAY_NAME(cholesky_up),
        equal_2_types
    },
    {
        "svd_m",
        "(m,n)->(m)",
        "svd when n>=m. ",
        4, 1, 1,
        FUNC_ARRAY_NAME(svd_N),
        svd_1_1_types
    },
    {
        "svd_n",
        "(m,n)->(n)",
        "svd when n<=m",
        4, 1, 1,
        FUNC_ARRAY_NAME(svd_N),
        svd_1_1_types
    },
    {
        "svd_m_s",
        "(m,n)->(m,m),(m),(m,n)",
        "svd when m<=n",
        4, 1, 3,
        FUNC_ARRAY_NAME(svd_S),
        svd_1_3_types
    },
    {
        "svd_n_s",
        "(m,n)->(m,n),(n),(n,n)",
        "svd when m>=n",
        4, 1, 3,
        FUNC_ARRAY_NAME(svd_S),
        svd_1_3_types
    },
    {
        "svd_m_f",
        "(m,n)->(m,m),(m),(n,n)",
        "svd when m<=n",
        4, 1, 3,
        FUNC_ARRAY_NAME(svd_A),
        svd_1_3_types
    },
    {
        "svd_n_f",
        "(m,n)->(m,m),(n),(n,n)",
        "svd when m>=n",
        4, 1, 3,
        FUNC_ARRAY_NAME(svd_A),
        svd_1_3_types
    },
    {
        "eig",
        "(m,m)->(m),(m,m)",
        "eig on the last two dimension and broadcast to the rest. \n"\
        "Results in a vector with the  eigenvalues and a matrix with the"\
        " eigenvectors. \n"\
        "    \"(m,m)->(m),(m,m)\" \n",
        3, 1, 2,
        FUNC_ARRAY_NAME(eig),
        eig_types
    },
    {
        "eigvals",
        "(m,m)->(m)",
        "eigvals on the last two dimension and broadcast to the rest. \n"\
        "Results in a vector of eigenvalues. \n",
        3, 1, 1,
        FUNC_ARRAY_NAME(eigvals),
        eigvals_types
    },
    {
        "qr_r_raw_m",
        "(m,n)->(m)",
        "Compute TAU vector for the last two dimensions \n"\
        "and broadcast to the rest. For m <= n. \n",
        2, 1, 1,
        FUNC_ARRAY_NAME(qr_r_raw),
        qr_r_raw_types
    },
    {
        "qr_r_raw_n",
        "(m,n)->(n)",
        "Compute TAU vector for the last two dimensions \n"\
        "and broadcast to the rest. For m > n. \n",
        2, 1, 1,
        FUNC_ARRAY_NAME(qr_r_raw),
        qr_r_raw_types
    },
    {
        "qr_reduced",
        "(m,n),(k)->(m,k)",
        "Compute Q matrix for the last two dimensions \n"\
        "and broadcast to the rest. \n",
        2, 2, 1,
        FUNC_ARRAY_NAME(qr_reduced),
        qr_reduced_types
    },
    {
        "qr_complete",
        "(m,n),(n)->(m,m)",
        "Compute Q matrix for the last two dimensions \n"\
        "and broadcast to the rest. For m > n. \n",
        2, 2, 1,
        FUNC_ARRAY_NAME(qr_complete),
        qr_complete_types
    },
    {
        "lstsq_m",
        "(m,n),(m,nrhs),()->(n,nrhs),(nrhs),(),(m)",
        "least squares on the last two dimensions and broadcast to the rest. \n"\
        "For m <= n. \n",
        4, 3, 4,
        FUNC_ARRAY_NAME(lstsq),
        lstsq_types
    },
    {
        "lstsq_n",
        "(m,n),(m,nrhs),()->(n,nrhs),(nrhs),(),(n)",
        "least squares on the last two dimensions and broadcast to the rest. \n"\
        "For m >= n, meaning that residuals are produced. \n",
        4, 3, 4,
        FUNC_ARRAY_NAME(lstsq),
        lstsq_types
    }



    "svd_n": {
        "description": "svd when n<=m",
        "inputs": 4,
        "outputs": 1,
        "dimensionality": 1,
        "function_name": FUNC_ARRAY_NAME(svd_N),
        "types": svd_1_1_types
    },
    "svd_m_s": {
        "description": "svd when m<=n",
        "inputs": 4,
        "outputs": 1,
        "dimensionality": 3,
        "function_name": FUNC_ARRAY_NAME(svd_S),
        "types": svd_1_3_types
    },
    "svd_n_s": {
        "description": "svd when m>=n",
        "inputs": 4,
        "outputs": 1,
        "dimensionality": 3,
        "function_name": FUNC_ARRAY_NAME(svd_S),
        "types": svd_1_3_types
    },
    "svd_m_f": {
        "description": "svd when m<=n",
        "inputs": 4,
        "outputs": 1,
        "dimensionality": 3,
        "function_name": FUNC_ARRAY_NAME(svd_A),
        "types": svd_1_3_types
    },
    "svd_n_f": {
        "description": "svd when m>=n",
        "inputs": 4,
        "outputs": 1,
        "dimensionality": 3,
        "function_name": FUNC_ARRAY_NAME(svd_A),
        "types": svd_1_3_types
    },
    "eig": {
        "description": "eig on the last two dimension and broadcast to the rest. Results in a vector with the eigenvalues and a matrix with the eigenvectors.",
        "inputs": 3,
        "outputs": 1,
        "dimensionality": 2,
        "function_name": FUNC_ARRAY_NAME(eig),
        "types": eig_types
    },
    "eigvals": {
        "description": "eigvals on the last two dimension and broadcast to the rest. Results in a vector of eigenvalues.",
        "inputs": 3,
        "outputs": 1,
        "dimensionality": 1,
        "function_name": FUNC_ARRAY_NAME(eigvals),
        "types": eigvals_types
    },
    "qr_r_raw_m": {
        "description": "Compute TAU vector for the last two dimensions and broadcast to the rest. For m <= n.",
        "inputs": 2,
        "outputs": 1,
        "dimensionality": 1,
        "function_name": FUNC_ARRAY_NAME(qr_r_raw),
        "types": qr_r_raw_types
    },
    "qr_r_raw_n": {
        "description": "Compute TAU vector for the last two dimensions and broadcast to the rest. For m > n.",
        "inputs": 2,
        "outputs": 1,
        "dimensionality": 1,
        "function_name": FUNC_ARRAY_NAME(qr_r_raw),
        "types": qr_r_raw_types
    },
    "qr_reduced": {
        "description": "Compute Q matrix for the last two dimensions and broadcast to the rest.",
        "inputs": 2,
        "outputs": 2,
        "dimensionality": 1,
        "function_name": FUNC_ARRAY_NAME(qr_reduced),
        "types": qr_reduced_types
    },
    "qr_complete": {
        "description": "Compute Q matrix for the last two dimensions and broadcast to the rest. For m > n.",
        "inputs": 2,
        "outputs": 2,
        "dimensionality": 1,
        "function_name": FUNC_ARRAY_NAME(qr_complete),
        "types": qr_complete_types
    },
    "lstsq_m": {
        "description": "least squares on the last two dimensions and broadcast to the rest. For m <= n.",
        "inputs": 4,
        "outputs": 3,
        "dimensionality": 4,
        "function_name": FUNC_ARRAY_NAME(lstsq),
        "types": lstsq_types
    },
    "lstsq_n": {
        "description": "least squares on the last two dimensions and broadcast to the rest. For m >= n, meaning that residuals are produced.",
        "inputs": 4,
        "outputs": 3,
        "dimensionality": 4,
        "function_name": FUNC_ARRAY_NAME(lstsq),
        "types": lstsq_types
    }
};

static int
addUfuncs(PyObject *dictionary) {
    PyObject *f;
    int i;
    const int gufunc_count = sizeof(gufunc_descriptors)/
        sizeof(gufunc_descriptors[0]);
    // 遍历 gufunc_descriptors 数组，为每个描述符创建 PyUFuncObject 对象并添加到给定的字典中
    for (i = 0; i < gufunc_count; i++) {
        GUFUNC_DESCRIPTOR_t* d = &gufunc_descriptors[i];
        // 使用描述符 d 创建 PyUFuncObject 对象 f
        f = PyUFunc_FromFuncAndDataAndSignature(d->funcs,
                                                array_of_nulls,
                                                d->types,
                                                d->ntypes,
                                                d->nin,
                                                d->nout,
                                                PyUFunc_None,
                                                d->name,
                                                d->doc,
                                                0,
                                                d->signature);
        if (f == NULL) {
            return -1;  // 如果创建失败，返回 -1 表示错误
        }
#if _UMATH_LINALG_DEBUG
        // 在调试模式下，输出 PyUFuncObject 对象的信息
        dump_ufunc_object((PyUFuncObject*) f);
#endif
        // 将创建的 PyUFuncObject 对象 f 添加到字典中，使用描述符的名称作为键
        int ret = PyDict_SetItemString(dictionary, d->name, f);
        Py_DECREF(f);
        if (ret < 0) {
            return -1;  // 如果添加失败，返回 -1 表示错误
        }
    }
    return 0;  // 成功添加所有的 ufunc 到字典中，返回 0 表示成功
}



/* -------------------------------------------------------------------------- */
                  /* Module initialization stuff  */

static PyMethodDef UMath_LinAlgMethods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        UMATH_LINALG_MODULE_NAME,
        NULL,
        -1,
        UMath_LinAlgMethods,
        NULL,
        NULL,
        NULL,
        NULL
};

// Python 模块初始化函数，创建并返回一个新的 PyModuleObject 对象
PyMODINIT_FUNC PyInit__umath_linalg(void)
{
    PyObject *m;
    PyObject *d;
    PyObject *version;

    // 创建一个新的 Python 模块对象
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;  // 如果创建失败，返回 NULL 表示错误
    }

    // 导入 NumPy 的 C API，确保 NumPy 的数据结构可以在模块中使用
    import_array();
    // 导入 NumPy 的通用函数 API，确保通用函数可以在模块中使用
    import_ufunc();

    // 获取模块 m 的字典对象
    d = PyModule_GetDict(m);
    if (d == NULL) {
        return NULL;  // 如果获取失败，返回 NULL 表示错误
    }

    // 创建包含模块版本信息的字符串对象
    version = PyUnicode_FromString(umath_linalg_version_string);
    if (version == NULL) {
        return NULL;  // 如果创建失败，返回 NULL 表示错误
    }
    // 将版本信息添加到模块的字典中
    int ret = PyDict_SetItemString(d, "__version__", version);
    Py_DECREF(version);
    if (ret < 0) {
        return NULL;  // 如果添加失败，返回 NULL 表示错误
    }

    /* Load the ufunc operators into the module's namespace */
    // 将 ufunc 操作符加载到模块的命名空间中
    if (addUfuncs(d) < 0) {
        return NULL;  // 如果加载失败，返回 NULL 表示错误
    }

    // 根据编译配置添加 _ilp64 到模块的字典中，表示是否支持 64 位整数 BLAS 操作
#ifdef HAVE_BLAS_ILP64
    PyDict_SetItemString(d, "_ilp64", Py_True);
#else
    PyDict_SetItemString(d, "_ilp64", Py_False);
#endif

    // 返回创建的 Python 模块对象
    return m;
}
```