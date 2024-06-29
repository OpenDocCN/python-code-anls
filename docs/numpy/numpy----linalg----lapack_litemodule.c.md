# `.\numpy\numpy\linalg\lapack_litemodule.c`

```py
/*
 * 此模块由Doug Heisterkamp贡献
 * Jim Hugunin进行了修改
 * Jeff Whitaker进行了更多修改
 */

/* 定义取消过时 API 的宏 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* 清除以前的 ssize_t 定义，以使用更安全的定义 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* 包含 NumPy 的数组对象头文件 */
#include "numpy/arrayobject.h"

/* 包含 NumPy 的 CBLAS 头文件 */
#include "npy_cblas.h"

/* 定义一个宏，用于生成 BLAS 函数名 */
#define FNAME(name) BLAS_FUNC(name)

/* 定义 Fortran 中使用的整型 */
typedef CBLAS_INT fortran_int;

#ifdef HAVE_BLAS_ILP64

/* 根据不同的数据类型和平台选择对应的 Fortran 整型格式 */
#if NPY_BITSOF_SHORT == 64
#define FINT_PYFMT       "h"
#elif NPY_BITSOF_INT == 64
#define FINT_PYFMT       "i"
#elif NPY_BITSOF_LONG == 64
#define FINT_PYFMT       "l"
#elif NPY_BITSOF_LONGLONG == 64
#define FINT_PYFMT       "L"
#else
/* 如果没有找到兼容的 64 位整型大小，则抛出错误 */
#error No compatible 64-bit integer size. \
       Please contact NumPy maintainers and give detailed information about your \
       compiler and platform, or dont try to use ILP64 BLAS
#endif

#else
/* 默认使用标准的 32 位整型 */
#define FINT_PYFMT       "i"
#endif

/* 定义复数类型结构体 */
typedef struct { float r, i; } f2c_complex;
typedef struct { double r, i; } f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

/* 声明外部链接的 BLAS 和 LAPACK 函数 */

extern fortran_int FNAME(dgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
                          double a[], fortran_int *lda, double b[], fortran_int *ldb,
                          double s[], double *rcond, fortran_int *rank,
                          double work[], fortran_int *lwork, fortran_int iwork[], fortran_int *info);

extern fortran_int FNAME(zgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
                          f2c_doublecomplex a[], fortran_int *lda,
                          f2c_doublecomplex b[], fortran_int *ldb,
                          double s[], double *rcond, fortran_int *rank,
                          f2c_doublecomplex work[], fortran_int *lwork,
                          double rwork[], fortran_int iwork[], fortran_int *info);

extern fortran_int FNAME(dgeqrf)(fortran_int *m, fortran_int *n, double a[], fortran_int *lda,
                          double tau[], double work[],
                          fortran_int *lwork, fortran_int *info);

extern fortran_int FNAME(zgeqrf)(fortran_int *m, fortran_int *n, f2c_doublecomplex a[], fortran_int *lda,
                          f2c_doublecomplex tau[], f2c_doublecomplex work[],
                          fortran_int *lwork, fortran_int *info);

extern fortran_int FNAME(dorgqr)(fortran_int *m, fortran_int *n, fortran_int *k, double a[], fortran_int *lda,
                          double tau[], double work[],
                          fortran_int *lwork, fortran_int *info);

extern fortran_int FNAME(zungqr)(fortran_int *m, fortran_int *n, fortran_int *k, f2c_doublecomplex a[],
                          fortran_int *lda, f2c_doublecomplex tau[],
                          f2c_doublecomplex work[], fortran_int *lwork, fortran_int *info);

extern fortran_int FNAME(xerbla)(char *srname, fortran_int *info);

/* 定义 LapackError 的静态 PyObject 指针 */
static PyObject *LapackError;

/* 定义一个宏，用于简化错误处理 */
#define TRY(E) if (!(E)) return NULL

/* 定义一个静态函数，用于检查 Python 对象 */
static int
check_object(PyObject *ob, int t, char *obname,
                        char *tname, char *funname)
{
    # 检查参数 ob 是否为 NumPy 数组，如果不是则抛出错误信息并返回 0
    if (!PyArray_Check(ob)) {
        PyErr_Format(LapackError,
                     "Expected an array for parameter %s in lapack_lite.%s",
                     obname, funname);
        return 0;
    }
    # 检查参数 ob 是否为 C 连续存储的 NumPy 数组，如果不是则抛出错误信息并返回 0
    else if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject *)ob)) {
        PyErr_Format(LapackError,
                     "Parameter %s is not contiguous in lapack_lite.%s",
                     obname, funname);
        return 0;
    }
    # 检查参数 ob 是否为指定类型 t 的 NumPy 数组，如果不是则抛出错误信息并返回 0
    else if (!(PyArray_TYPE((PyArrayObject *)ob) == t)) {
        PyErr_Format(LapackError,
                     "Parameter %s is not of type %s in lapack_lite.%s",
                     obname, tname, funname);
        return 0;
    }
    # 检查参数 ob 是否具有非本机字节顺序，如果是则抛出错误信息并返回 0
    else if (PyArray_ISBYTESWAPPED((PyArrayObject *)ob)) {
        PyErr_Format(LapackError,
                     "Parameter %s has non-native byte order in lapack_lite.%s",
                     obname, funname);
        return 0;
    }
    # 如果所有条件都满足，则返回 1 表示通过参数检查
    else {
        return 1;
    }
# 定义宏 CHDATA，将指针 p 强制转换为 char* 类型，并获取其指向的 NumPy 数组数据的指针
#define CHDATA(p) ((char *) PyArray_DATA((PyArrayObject *)p))
# 定义宏 SHDATA，将指针 p 强制转换为 short int* 类型，并获取其指向的 NumPy 数组数据的指针
#define SHDATA(p) ((short int *) PyArray_DATA((PyArrayObject *)p))
# 定义宏 DDATA，将指针 p 强制转换为 double* 类型，并获取其指向的 NumPy 数组数据的指针
#define DDATA(p) ((double *) PyArray_DATA((PyArrayObject *)p))
# 定义宏 FDATA，将指针 p 强制转换为 float* 类型，并获取其指向的 NumPy 数组数据的指针
#define FDATA(p) ((float *) PyArray_DATA((PyArrayObject *)p))
# 定义宏 CDATA，将指针 p 强制转换为 f2c_complex* 类型，并获取其指向的 NumPy 数组数据的指针
#define CDATA(p) ((f2c_complex *) PyArray_DATA((PyArrayObject *)p))
# 定义宏 ZDATA，将指针 p 强制转换为 f2c_doublecomplex* 类型，并获取其指向的 NumPy 数组数据的指针
#define ZDATA(p) ((f2c_doublecomplex *) PyArray_DATA((PyArrayObject *)p))
# 定义宏 IDATA，将指针 p 强制转换为 fortran_int* 类型，并获取其指向的 NumPy 数组数据的指针
#define IDATA(p) ((fortran_int *) PyArray_DATA((PyArrayObject *)p))

# 定义 lapack_lite_dgelsd 函数，接收 Python 的调用，执行 dgelsd LAPACK 求解操作
static PyObject *
lapack_lite_dgelsd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    fortran_int lapack_lite_status;  // LAPACK 操作返回状态
    fortran_int m;                   // 矩阵 A 的行数
    fortran_int n;                   // 矩阵 A 的列数
    fortran_int nrhs;                // 矩阵 B 的列数
    PyObject *a;                     // NumPy 数组对象，存储矩阵 A 的数据
    fortran_int lda;                 // 矩阵 A 的列数
    PyObject *b;                     // NumPy 数组对象，存储矩阵 B 的数据
    fortran_int ldb;                 // 矩阵 B 的列数
    PyObject *s;                     // NumPy 数组对象，存储奇异值分解结果 S 的数据
    double rcond;                    // 控制奇异值的截断参数
    fortran_int rank;                // 估计的矩阵 A 的秩
    PyObject *work;                  // NumPy 数组对象，提供给 LAPACK 的工作空间
    PyObject *iwork;                 // NumPy 数组对象，提供给 LAPACK 的整数工作空间
    fortran_int lwork;               // 工作空间的长度
    fortran_int info;                // LAPACK 操作的信息

    TRY(PyArg_ParseTuple(args,
                         (FINT_PYFMT FINT_PYFMT FINT_PYFMT "O" FINT_PYFMT "O"
                          FINT_PYFMT "O" "d" FINT_PYFMT "O" FINT_PYFMT "O"
                          FINT_PYFMT ":dgelsd"),
                         &m,&n,&nrhs,&a,&lda,&b,&ldb,&s,&rcond,
                         &rank,&work,&lwork,&iwork,&info));
    
    TRY(check_object(a,NPY_DOUBLE,"a","NPY_DOUBLE","dgelsd"));  // 检查参数 a 是否为 NPY_DOUBLE 类型的 NumPy 数组
    TRY(check_object(b,NPY_DOUBLE,"b","NPY_DOUBLE","dgelsd"));  // 检查参数 b 是否为 NPY_DOUBLE 类型的 NumPy 数组
    TRY(check_object(s,NPY_DOUBLE,"s","NPY_DOUBLE","dgelsd"));  // 检查参数 s 是否为 NPY_DOUBLE 类型的 NumPy 数组
    TRY(check_object(work,NPY_DOUBLE,"work","NPY_DOUBLE","dgelsd"));  // 检查参数 work 是否为 NPY_DOUBLE 类型的 NumPy 数组
#ifndef NPY_UMATH_USE_BLAS64_
    TRY(check_object(iwork,NPY_INT,"iwork","NPY_INT","dgelsd"));  // 检查参数 iwork 是否为 NPY_INT 类型的 NumPy 数组
#else
    TRY(check_object(iwork,NPY_INT64,"iwork","NPY_INT64","dgelsd"));  // 检查参数 iwork 是否为 NPY_INT64 类型的 NumPy 数组
#endif

    // 调用 LAPACK 中的 dgelsd 函数，进行最小二乘法求解
    lapack_lite_status =
            FNAME(dgelsd)(&m,&n,&nrhs,DDATA(a),&lda,DDATA(b),&ldb,
                          DDATA(s),&rcond,&rank,DDATA(work),&lwork,
                          IDATA(iwork),&info);
    if (PyErr_Occurred()) {
        return NULL;
    }

    // 构建返回值，包括 dgelsd 操作的结果信息
    return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:d,s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT "}"),
                         "dgelsd_",lapack_lite_status,"m",m,"n",n,"nrhs",nrhs,
                         "lda",lda,"ldb",ldb,"rcond",rcond,"rank",rank,
                         "lwork",lwork,"info",info);
}
{
    fortran_int lapack_lite_status;  // 定义 LAPACK 函数返回状态变量
    fortran_int m, n, lwork;  // 定义矩阵维度 m, n 和工作数组大小 lwork
    PyObject *a, *tau, *work;  // 定义 Python 对象指针，分别用于矩阵 A、TAU 和工作数组
    fortran_int lda;  // 定义矩阵 A 的 leading dimension
    fortran_int info;  // 定义 LAPACK 函数信息变量

    TRY(PyArg_ParseTuple(args,
                         (FINT_PYFMT FINT_PYFMT "O" FINT_PYFMT "OO"
                          FINT_PYFMT FINT_PYFMT ":dgeqrf"),
                         &m, &n, &a, &lda, &tau, &work, &lwork, &info));
    // 解析传入的 Python 元组参数，获取 m, n, a, lda, tau, work, lwork 和 info 的值

    /* check objects and convert to right storage order */
    TRY(check_object(a, NPY_DOUBLE, "a", "NPY_DOUBLE", "dgeqrf"));
    TRY(check_object(tau, NPY_DOUBLE, "tau", "NPY_DOUBLE", "dgeqrf"));
    TRY(check_object(work, NPY_DOUBLE, "work", "NPY_DOUBLE", "dgeqrf"));
    // 检查对象类型并确保存储顺序正确，针对矩阵 a、tau 和 work

    lapack_lite_status =
            FNAME(dgeqrf)(&m, &n, DDATA(a), &lda, DDATA(tau),
                          DDATA(work), &lwork, &info);
    // 调用 LAPACK 的 dgeqrf 函数进行 QR 分解计算

    if (PyErr_Occurred()) {
        return NULL;
    }
    // 检查是否有 Python 异常发生，如有则返回空指针

    return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT "}"),
                         "dgeqrf_",
                         lapack_lite_status, "m", m, "n", n, "lda", lda,
                         "lwork", lwork, "info", info);
    // 构建并返回包含函数执行结果的 Python 元组对象
}


static PyObject *
lapack_lite_dorgqr(PyObject *NPY_UNUSED(self), PyObject *args)
{
    fortran_int lapack_lite_status;
    fortran_int m, n, k, lwork;
    PyObject *a, *tau, *work;
    fortran_int lda;
    fortran_int info;

    TRY(PyArg_ParseTuple(args,
                         (FINT_PYFMT FINT_PYFMT FINT_PYFMT "O"
                          FINT_PYFMT "OO" FINT_PYFMT FINT_PYFMT
                          ":dorgqr"),
                         &m, &n, &k, &a, &lda, &tau, &work, &lwork, &info));
    // 解析传入的 Python 元组参数，获取 m, n, k, a, lda, tau, work, lwork 和 info 的值

    TRY(check_object(a, NPY_DOUBLE, "a", "NPY_DOUBLE", "dorgqr"));
    TRY(check_object(tau, NPY_DOUBLE, "tau", "NPY_DOUBLE", "dorgqr"));
    TRY(check_object(work, NPY_DOUBLE, "work", "NPY_DOUBLE", "dorgqr"));
    // 检查对象类型并确保存储顺序正确，针对矩阵 a、tau 和 work

    lapack_lite_status =
        FNAME(dorgqr)(&m, &n, &k, DDATA(a), &lda, DDATA(tau), DDATA(work),
                      &lwork, &info);
    // 调用 LAPACK 的 dorgqr 函数进行 QR 分解计算

    if (PyErr_Occurred()) {
        return NULL;
    }
    // 检查是否有 Python 异常发生，如有则返回空指针

    return Py_BuildValue("{s:i,s:i}", "dorgqr_", lapack_lite_status,
                         "info", info);
    // 构建并返回包含函数执行结果的 Python 字典对象
}


static PyObject *
lapack_lite_zgelsd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    fortran_int lapack_lite_status;
    fortran_int m;
    fortran_int n;
    fortran_int nrhs;
    PyObject *a;
    fortran_int lda;
    PyObject *b;
    fortran_int ldb;
    PyObject *s;
    double rcond;
    fortran_int rank;
    PyObject *work;
    fortran_int lwork;
    PyObject *rwork;
    PyObject *iwork;
    fortran_int info;
    # 尝试解析传入的参数元组，根据指定格式字符串匹配参数，对应关系如下：
    # m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork, iwork, info
    TRY(PyArg_ParseTuple(args,
                         (FINT_PYFMT FINT_PYFMT FINT_PYFMT "O" FINT_PYFMT
                          "O" FINT_PYFMT "Od" FINT_PYFMT "O" FINT_PYFMT
                          "OO" FINT_PYFMT ":zgelsd"),
                         &m,&n,&nrhs,&a,&lda,&b,&ldb,&s,&rcond,
                         &rank,&work,&lwork,&rwork,&iwork,&info));

    # 尝试检查参数对象 a，确保其为 NPY_CDOUBLE 类型，用于 zgelsd 函数
    TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zgelsd"));
    # 尝试检查参数对象 b，确保其为 NPY_CDOUBLE 类型，用于 zgelsd 函数
    TRY(check_object(b,NPY_CDOUBLE,"b","NPY_CDOUBLE","zgelsd"));
    # 尝试检查参数对象 s，确保其为 NPY_DOUBLE 类型，用于 zgelsd 函数
    TRY(check_object(s,NPY_DOUBLE,"s","NPY_DOUBLE","zgelsd"));
    # 尝试检查参数对象 work，确保其为 NPY_CDOUBLE 类型，用于 zgelsd 函数
    TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zgelsd"));
    # 尝试检查参数对象 rwork，确保其为 NPY_DOUBLE 类型，用于 zgelsd 函数
    TRY(check_object(rwork,NPY_DOUBLE,"rwork","NPY_DOUBLE","zgelsd"));
#ifndef NPY_UMATH_USE_BLAS64_
    // 如果未定义 NPY_UMATH_USE_BLAS64_，调用 check_object 函数检查 iwork 对象，期望其类型为 NPY_INT，用于函数 zgelsd
    TRY(check_object(iwork,NPY_INT,"iwork","NPY_INT","zgelsd"));
#else
    // 如果定义了 NPY_UMATH_USE_BLAS64_，调用 check_object 函数检查 iwork 对象，期望其类型为 NPY_INT64，用于函数 zgelsd
    TRY(check_object(iwork,NPY_INT64,"iwork","NPY_INT64","zgelsd"));
#endif

    // 调用 LAPACK 函数 zgelsd 执行最小二乘解，对矩阵进行奇异值分解
    lapack_lite_status =
        FNAME(zgelsd)(&m,&n,&nrhs,ZDATA(a),&lda,ZDATA(b),&ldb,DDATA(s),&rcond,
                      &rank,ZDATA(work),&lwork,DDATA(rwork),IDATA(iwork),&info);
    // 检查是否发生了 Python 异常，若有则返回空指针
    if (PyErr_Occurred()) {
        return NULL;
    }

    // 返回 LAPACK 函数 zgelsd 的执行结果和相关参数的 Python 对象表示
    return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          "}"),
                         "zgelsd_",
                         lapack_lite_status,"m",m,"n",n,"nrhs",nrhs,"lda",lda,
                         "ldb",ldb,"rank",rank,"lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_zgeqrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    fortran_int lapack_lite_status;
    fortran_int m, n, lwork;
    PyObject *a, *tau, *work;
    fortran_int lda;
    fortran_int info;

    // 尝试解析 Python 元组参数 args，解析后的参数依次赋值给 m, n, a, lda, tau, work, lwork, info
    TRY(PyArg_ParseTuple(args,
                         (FINT_PYFMT FINT_PYFMT "O" FINT_PYFMT "OO"
                          FINT_PYFMT "" FINT_PYFMT ":zgeqrf"),
                         &m,&n,&a,&lda,&tau,&work,&lwork,&info));

    // 检查并转换对象 a, tau, work 的存储顺序为 NPY_CDOUBLE
    TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zgeqrf"));
    TRY(check_object(tau,NPY_CDOUBLE,"tau","NPY_CDOUBLE","zgeqrf"));
    TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zgeqrf"));

    // 调用 LAPACK 函数 zgeqrf 执行 QR 分解
    lapack_lite_status =
        FNAME(zgeqrf)(&m, &n, ZDATA(a), &lda, ZDATA(tau), ZDATA(work),
                      &lwork, &info);
    // 检查是否发生了 Python 异常，若有则返回空指针
    if (PyErr_Occurred()) {
        return NULL;
    }

    // 返回 LAPACK 函数 zgeqrf 的执行结果和相关参数的 Python 对象表示
    return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT
                          ",s:" FINT_PYFMT ",s:" FINT_PYFMT "}"),
                         "zgeqrf_",lapack_lite_status,"m",m,"n",n,"lda",lda,"lwork",lwork,"info",info);
}


static PyObject *
lapack_lite_zungqr(PyObject *NPY_UNUSED(self), PyObject *args)
{
        // lapack_lite 模块中的 zungqr 函数实现
        fortran_int lapack_lite_status;  // Lapack 函数调用返回状态
        fortran_int m, n, k, lwork;  // 整型变量 m, n, k, lwork
        PyObject *a, *tau, *work;  // Python 对象 a, tau, work
        fortran_int lda;  // Lapack 函数调用中的 lda 参数
        fortran_int info;  // Lapack 函数调用返回的信息状态

        // 解析 Python 函数参数，格式化为 Lapack 函数调用的输入
        TRY(PyArg_ParseTuple(args,
                             (FINT_PYFMT FINT_PYFMT FINT_PYFMT "O"
                              FINT_PYFMT "OO" FINT_PYFMT "" FINT_PYFMT
                              ":zungqr"),
                             &m, &n, &k, &a, &lda, &tau, &work, &lwork, &info));
        // 检查 Python 对象 a, tau, work 的类型是否符合预期
        TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zungqr"));
        TRY(check_object(tau,NPY_CDOUBLE,"tau","NPY_CDOUBLE","zungqr"));
        TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zungqr"));

        // 调用 Lapack 中的 zungqr 函数进行 QR 分解
        lapack_lite_status =
            FNAME(zungqr)(&m, &n, &k, ZDATA(a), &lda, ZDATA(tau), ZDATA(work),
                          &lwork, &info);
        // 检查是否有 Python 异常发生，若有则返回空指针
        if (PyErr_Occurred()) {
            return NULL;
        }

        // 构建 Python 对象返回结果，包括 zungqr 函数的状态和信息
        return Py_BuildValue(("{s:" FINT_PYFMT ",s:" FINT_PYFMT "}"),
                             "zungqr_",lapack_lite_status,
                             "info",info);
}


static PyObject *
lapack_lite_xerbla(PyObject *NPY_UNUSED(self), PyObject *args)
{
    // 初始化 Lapack 错误信息的状态为 -1
    fortran_int info = -1;

    NPY_BEGIN_THREADS_DEF;  // 定义多线程使用的变量
    NPY_BEGIN_THREADS;  // 开始多线程执行

    // 调用 Lapack 错误处理函数 xerbla
    FNAME(xerbla)("test", &info);

    NPY_END_THREADS;  // 结束多线程执行

    // 检查是否有 Python 异常发生，若有则返回空指针
    if (PyErr_Occurred()) {
        return NULL;
    }
    // 返回 None
    Py_RETURN_NONE;
}


#define STR(x) #x
#define lameth(name) {STR(name), lapack_lite_##name, METH_VARARGS, NULL}
static struct PyMethodDef lapack_lite_module_methods[] = {
    lameth(dgelsd),  // lapack_lite 模块中的 dgelsd 函数
    lameth(dgeqrf),  // lapack_lite 模块中的 dgeqrf 函数
    lameth(dorgqr),  // lapack_lite 模块中的 dorgqr 函数
    lameth(zgelsd),  // lapack_lite 模块中的 zgelsd 函数
    lameth(zgeqrf),  // lapack_lite 模块中的 zgeqrf 函数
    lameth(zungqr),  // lapack_lite 模块中的 zungqr 函数
    lameth(xerbla),  // lapack_lite 模块中的 xerbla 函数
    { NULL,NULL,0, NULL}  // 方法定义结束的标志
};


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,  // 初始化 Python 模块定义头部
        "lapack_lite",  // 模块名为 lapack_lite
        NULL,  // 模块文档字符串为空
        -1,  // 模块状态为 -1，表示模块全局变量和线程的隔离
        lapack_lite_module_methods,  // 模块中包含的方法列表
        NULL,  // 模块的全局变量为 NULL
        NULL,  // 模块的 slot 常规方法为 NULL
        NULL,  // 模块的特殊内存分配器为 NULL
        NULL  // 模块的清理函数为 NULL
};

/* 模块初始化函数 */
PyMODINIT_FUNC PyInit_lapack_lite(void)
{
    PyObject *m,*d;  // Python 对象 m 和 d
    m = PyModule_Create(&moduledef);  // 创建名为 lapack_lite 的 Python 模块对象
    if (m == NULL) {
        return NULL;
    }
    import_array();  // 导入 NumPy 的数组对象

    d = PyModule_GetDict(m);  // 获取模块 m 的字典对象
    LapackError = PyErr_NewException("lapack_lite.LapackError", NULL, NULL);
    PyDict_SetItemString(d, "LapackError", LapackError);  // 将 LapackError 异常对象加入模块字典

#ifdef HAVE_BLAS_ILP64
    PyDict_SetItemString(d, "_ilp64", Py_True);  // 设置模块字典中的 _ilp64 为 Py_True
#else
    PyDict_SetItemString(d, "_ilp64", Py_False);  // 设置模块字典中的 _ilp64 为 Py_False
#endif

    return m;  // 返回创建的 Python 模块对象
}
```