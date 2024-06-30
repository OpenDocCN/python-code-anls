# `D:\src\scipysrc\scipy\scipy\linalg\tests\_cython_examples\extending.pyx`

```
#!/usr/bin/env python3
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

# 从scipy.linalg.cython_blas模块中导入 cdotu 函数
from scipy.linalg.cython_blas cimport cdotu
# 从scipy.linalg.cython_lapack模块中导入 dgtsv 函数
from scipy.linalg.cython_lapack cimport dgtsv

# 定义一个 Cython 函数 tridiag，接受四个一维数组参数
cpdef tridiag(double[:] a, double[:] b, double[:] c, double[:] x):
    """ Solve the system A y = x for y where A is the tridiagonal matrix with
    subdiagonal 'a', diagonal 'b', and superdiagonal 'c'. """
    # 确定数组长度和求解参数
    cdef int n=b.shape[0], nrhs=1, info
    # 在数组 x 上直接写入解
    dgtsv(&n, &nrhs, &a[0], &b[0], &c[0], &x[0], &n, &info)

# 定义一个 Cython 函数 complex_dot，接受两个复数向量参数
cpdef float complex complex_dot(float complex[:] cx, float complex[:] cy):
    """ Take dot product of two complex vectors """
    cdef:
        int n = cx.shape[0]  # 向量的长度
        int incx = cx.strides[0] // sizeof(cx[0])  # cx 中相邻元素之间的字节数
        int incy = cy.strides[0] // sizeof(cy[0])  # cy 中相邻元素之间的字节数
    # 调用 cdotu 函数计算复数向量的点积并返回结果
    return cdotu(&n, &cx[0], &incx, &cy[0], &incy)
```