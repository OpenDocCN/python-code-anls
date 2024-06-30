# `D:\src\scipysrc\scikit-learn\sklearn\utils\_cython_blas.pxd`

```
# 导入 Cython 模块中的 floating 类型
from cython cimport floating

# 定义 BLAS_Order 枚举类型，用于指定矩阵存储顺序
cpdef enum BLAS_Order:
    RowMajor  # 行主序，对应 C 连续存储方式
    ColMajor  # 列主序，对应 Fortran 连续存储方式

# 定义 BLAS_Trans 枚举类型，用于指定矩阵转置选项
cpdef enum BLAS_Trans:
    NoTrans = 110  # 不进行转置，对应 'n'
    Trans = 116    # 进行转置，对应 't'

# BLAS Level 1 ################################################################

# 计算两个向量的内积
cdef floating _dot(int, const floating*, int, const floating*, int) noexcept nogil

# 计算向量的绝对值之和
cdef floating _asum(int, const floating*, int) noexcept nogil

# 向量相加：y = a * x + y
cdef void _axpy(int, floating, const floating*, int, floating*, int) noexcept nogil

# 计算向量的二范数
cdef floating _nrm2(int, const floating*, int) noexcept nogil

# 向量复制：y = x
cdef void _copy(int, const floating*, int, const floating*, int) noexcept nogil

# 向量数乘：x = a * x
cdef void _scal(int, floating, const floating*, int) noexcept nogil

# Givens 旋转
cdef void _rotg(floating*, floating*, floating*, floating*) noexcept nogil

# 应用 Givens 旋转到向量中
cdef void _rot(int, floating*, int, floating*, int, floating, floating) noexcept nogil

# BLAS Level 2 ################################################################

# 矩阵向量乘法：y = alpha * A * x + beta * y
cdef void _gemv(BLAS_Order, BLAS_Trans, int, int, floating, const floating*, int,
                const floating*, int, floating, floating*, int) noexcept nogil

# 外积：A = alpha * x * y^T + A
cdef void _ger(BLAS_Order, int, int, floating, const floating*, int, const floating*,
               int, floating*, int) noexcept nogil

# BLAS Level 3 ################################################################

# 矩阵乘法：C = alpha * op(A) * op(B) + beta * C
cdef void _gemm(BLAS_Order, BLAS_Trans, BLAS_Trans, int, int, int, floating,
                const floating*, int, const floating*, int, floating, floating*,
                int) noexcept nogil
```