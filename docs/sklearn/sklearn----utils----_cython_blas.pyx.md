# `D:\src\scipysrc\scikit-learn\sklearn\utils\_cython_blas.pyx`

```
# 导入 Cython 模块中的 floating 类型
from cython cimport floating

# 从 scipy.linalg.cython_blas 中导入相应的函数
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport sasum, dasum
from scipy.linalg.cython_blas cimport saxpy, daxpy
from scipy.linalg.cython_blas cimport snrm2, dnrm2
from scipy.linalg.cython_blas cimport scopy, dcopy
from scipy.linalg.cython_blas cimport sscal, dscal
from scipy.linalg.cython_blas cimport srotg, drotg
from scipy.linalg.cython_blas cimport srot, drot
from scipy.linalg.cython_blas cimport sgemv, dgemv
from scipy.linalg.cython_blas cimport sger, dger
from scipy.linalg.cython_blas cimport sgemm, dgemm


################
# BLAS Level 1 #
################

# 定义函数 _dot，计算两个向量的内积 x.T.y
cdef floating _dot(int n, const floating *x, int incx,
                   const floating *y, int incy) noexcept nogil:
    """x.T.y"""
    # 根据浮点数类型选择调用单精度或双精度的 BLAS 函数
    if floating is float:
        return sdot(&n, <float *> x, &incx, <float *> y, &incy)
    else:
        return ddot(&n, <double *> x, &incx, <double *> y, &incy)


# Python 可调用版本，接受内存视图作为参数
cpdef _dot_memview(const floating[::1] x, const floating[::1] y):
    return _dot(x.shape[0], &x[0], 1, &y[0], 1)


# 定义函数 _asum，计算向量的绝对值之和 sum(|x_i|)
cdef floating _asum(int n, const floating *x, int incx) noexcept nogil:
    """sum(|x_i|)"""
    # 根据浮点数类型选择调用单精度或双精度的 BLAS 函数
    if floating is float:
        return sasum(&n, <float *> x, &incx)
    else:
        return dasum(&n, <double *> x, &incx)


# Python 可调用版本，接受内存视图作为参数
cpdef _asum_memview(const floating[::1] x):
    return _asum(x.shape[0], &x[0], 1)


# 定义函数 _axpy，执行向量的线性组合操作 y := alpha * x + y
cdef void _axpy(int n, floating alpha, const floating *x, int incx,
                floating *y, int incy) noexcept nogil:
    """y := alpha * x + y"""
    # 根据浮点数类型选择调用单精度或双精度的 BLAS 函数
    if floating is float:
        saxpy(&n, &alpha, <float *> x, &incx, y, &incy)
    else:
        daxpy(&n, &alpha, <double *> x, &incx, y, &incy)


# Python 可调用版本，接受内存视图作为参数
cpdef _axpy_memview(floating alpha, const floating[::1] x, floating[::1] y):
    _axpy(x.shape[0], alpha, &x[0], 1, &y[0], 1)


# 定义函数 _nrm2，计算向量的二范数 sqrt(sum((x_i)^2))
cdef floating _nrm2(int n, const floating *x, int incx) noexcept nogil:
    """sqrt(sum((x_i)^2))"""
    # 根据浮点数类型选择调用单精度或双精度的 BLAS 函数
    if floating is float:
        return snrm2(&n, <float *> x, &incx)
    else:
        return dnrm2(&n, <double *> x, &incx)


# Python 可调用版本，接受内存视图作为参数
cpdef _nrm2_memview(const floating[::1] x):
    return _nrm2(x.shape[0], &x[0], 1)


# 定义函数 _copy，实现向量的复制操作 y := x
cdef void _copy(int n, const floating *x, int incx, const floating *y, int incy) noexcept nogil:
    """y := x"""
    # 根据浮点数类型选择调用单精度或双精度的 BLAS 函数
    if floating is float:
        scopy(&n, <float *> x, &incx, <float *> y, &incy)
    else:
        dcopy(&n, <double *> x, &incx, <double *> y, &incy)


# Python 可调用版本，接受两个内存视图作为参数
cpdef _copy_memview(const floating[::1] x, const floating[::1] y):
    _copy(x.shape[0], &x[0], 1, &y[0], 1)


# 定义函数 _scal，执行向量的数乘操作 x := alpha * x
cdef void _scal(int n, floating alpha, const floating *x, int incx) noexcept nogil:
    """x := alpha * x"""
    # 根据浮点数类型选择调用单精度或双精度的 BLAS 函数
    if floating is float:
        sscal(&n, &alpha, <float *> x, &incx)
    else:
        dscal(&n, &alpha, <double *> x, &incx)


# Python 可调用版本，接受数值 alpha 和内存视图作为参数
cpdef _scal_memview(floating alpha, const floating[::1] x):
    _scal(x.shape[0], alpha, &x[0], 1)


# 定义函数 _rotg，生成平面旋转
cdef void _rotg(floating *a, floating *b, floating *c, floating *s) noexcept nogil:
    """Generate plane rotation"""
    # 如果变量 `floating` 的类型是 float 类型
    if floating is float:
        # 调用 srotg 函数，执行特定操作
        srotg(a, b, c, s)
    # 否则，如果 `floating` 的类型不是 float
    else:
        # 调用 drotg 函数，执行另一种特定操作
        drotg(a, b, c, s)
cpdef _rotg_memview(floating a, floating b, floating c, floating s):
    # 调用 C 函数 _rotg，并传递参数 a, b, c, s 的地址
    _rotg(&a, &b, &c, &s)
    # 返回经过 _rotg 函数处理后的 a, b, c, s 值
    return a, b, c, s


cdef void _rot(int n, floating *x, int incx, floating *y, int incy,
               floating c, floating s) noexcept nogil:
    """Apply plane rotation"""
    # 根据参数类型决定调用 srot 或 drot 函数
    if floating is float:
        srot(&n, x, &incx, y, &incy, &c, &s)
    else:
        drot(&n, x, &incx, y, &incy, &c, &s)


cpdef _rot_memview(floating[::1] x, floating[::1] y, floating c, floating s):
    # 调用 _rot 函数来应用平面旋转
    _rot(x.shape[0], &x[0], 1, &y[0], 1, c, s)


################
# BLAS Level 2 #
################

cdef void _gemv(BLAS_Order order, BLAS_Trans ta, int m, int n, floating alpha,
                const floating *A, int lda, const floating *x, int incx,
                floating beta, floating *y, int incy) noexcept nogil:
    """y := alpha * op(A).x + beta * y"""
    # 根据 BLAS_Order 和 BLAS_Trans 的值选择操作方式
    cdef char ta_ = ta
    if order == RowMajor:
        # 在行主序下，调整 ta_ 的值以匹配 BLAS 库的要求
        ta_ = NoTrans if ta == Trans else Trans
        # 根据浮点类型调用对应的 sgemv 或 dgemv 函数
        if floating is float:
            sgemv(&ta_, &n, &m, &alpha, <float *> A, &lda, <float *> x,
                  &incx, &beta, y, &incy)
        else:
            dgemv(&ta_, &n, &m, &alpha, <double *> A, &lda, <double *> x,
                  &incx, &beta, y, &incy)
    else:
        # 在列主序下，直接调用对应的 sgemv 或 dgemv 函数
        if floating is float:
            sgemv(&ta_, &m, &n, &alpha, <float *> A, &lda, <float *> x,
                  &incx, &beta, y, &incy)
        else:
            dgemv(&ta_, &m, &n, &alpha, <double *> A, &lda, <double *> x,
                  &incx, &beta, y, &incy)


cpdef _gemv_memview(BLAS_Trans ta, floating alpha, const floating[:, :] A,
                    const floating[::1] x, floating beta, floating[::1] y):
    cdef:
        int m = A.shape[0]
        int n = A.shape[1]
        # 根据 A 的内存布局确定 BLAS_Order
        BLAS_Order order = ColMajor if A.strides[0] == A.itemsize else RowMajor
        int lda = m if order == ColMajor else n

    # 调用 _gemv 函数来执行矩阵向量乘法
    _gemv(order, ta, m, n, alpha, &A[0, 0], lda, &x[0], 1, beta, &y[0], 1)


cdef void _ger(BLAS_Order order, int m, int n, floating alpha,
               const floating *x, int incx, const floating *y,
               int incy, floating *A, int lda) noexcept nogil:
    """A := alpha * x.y.T + A"""
    # 根据 BLAS_Order 调用对应的 sger 或 dger 函数
    if order == RowMajor:
        if floating is float:
            sger(&n, &m, &alpha, <float *> y, &incy, <float *> x, &incx, A, &lda)
        else:
            dger(&n, &m, &alpha, <double *> y, &incy, <double *> x, &incx, A, &lda)
    else:
        if floating is float:
            sger(&m, &n, &alpha, <float *> x, &incx, <float *> y, &incy, A, &lda)
        else:
            dger(&m, &n, &alpha, <double *> x, &incx, <double *> y, &incy, A, &lda)


cpdef _ger_memview(floating alpha, const floating[::1] x,
                   const floating[::1] y, floating[:, :] A):
    cdef:
        int m = A.shape[0]
        int n = A.shape[1]
        # 根据 A 的内存布局确定 BLAS_Order
        BLAS_Order order = ColMajor if A.strides[0] == A.itemsize else RowMajor
        int lda = m if order == ColMajor else n

    # 调用 _ger 函数来执行矩阵外积的 BLAS 操作
    _ger(order, m, n, alpha, &x[0], 1, &y[0], 1, &A[0, 0], lda)


################
# BLAS Level 3 #
################

# 定义一个 C 语言函数，用于实现 BLAS Level 3 中的矩阵乘法运算 C := alpha * op(A).op(B) + beta * C
cdef void _gemm(BLAS_Order order, BLAS_Trans ta, BLAS_Trans tb, int m, int n,
                int k, floating alpha, const floating *A, int lda, const floating *B,
                int ldb, floating beta, floating *C, int ldc) noexcept nogil:
    """C := alpha * op(A).op(B) + beta * C"""
    # TODO: 移除指针类型转换，一旦 SciPy 使用了 const 限定符
    # 参考：https://github.com/scipy/scipy/issues/14262
    cdef:
        char ta_ = ta  # 将枚举类型 BLAS_Trans 转换为 char 类型
        char tb_ = tb  # 将枚举类型 BLAS_Trans 转换为 char 类型

    # 根据矩阵的存储顺序和数据类型调用不同的 BLAS 函数实现矩阵乘法
    if order == RowMajor:
        if floating is float:
            sgemm(&tb_, &ta_, &n, &m, &k, &alpha, <float*>B,
                  &ldb, <float*>A, &lda, &beta, C, &ldc)
        else:
            dgemm(&tb_, &ta_, &n, &m, &k, &alpha, <double*>B,
                  &ldb, <double*>A, &lda, &beta, C, &ldc)
    else:
        if floating is float:
            sgemm(&ta_, &tb_, &m, &n, &k, &alpha, <float*>A,
                  &lda, <float*>B, &ldb, &beta, C, &ldc)
        else:
            dgemm(&ta_, &tb_, &m, &n, &k, &alpha, <double*>A,
                  &lda, <double*>B, &ldb, &beta, C, &ldc)

# 使用内存视图进行矩阵乘法运算
cpdef _gemm_memview(BLAS_Trans ta, BLAS_Trans tb, floating alpha,
                    const floating[:, :] A, const floating[:, :] B, floating beta,
                    floating[:, :] C):
    cdef:
        int m = A.shape[0] if ta == NoTrans else A.shape[1]  # 确定矩阵 C 的行数 m
        int n = B.shape[1] if tb == NoTrans else B.shape[0]  # 确定矩阵 C 的列数 n
        int k = A.shape[1] if ta == NoTrans else A.shape[0]  # 确定中间维度 k
        int lda, ldb, ldc  # 定义矩阵 A, B, C 的 leading dimensions
        BLAS_Order order = ColMajor if A.strides[0] == A.itemsize else RowMajor  # 确定矩阵的存储顺序

    # 根据存储顺序设置矩阵 A, B 的 leading dimensions
    if order == RowMajor:
        lda = k if ta == NoTrans else m
        ldb = n if tb == NoTrans else k
        ldc = n
    else:
        lda = m if ta == NoTrans else k
        ldb = k if tb == NoTrans else n
        ldc = m

    # 调用底层的 _gemm 函数进行矩阵乘法计算
    _gemm(order, ta, tb, m, n, k, alpha, &A[0, 0],
          lda, &B[0, 0], ldb, beta, &C[0, 0], ldc)
```