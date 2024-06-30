# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\givens_elimination.pyx`

```
# 导入Cython模块声明
cimport cython
# 从Cython版本的SciPy线性代数模块中导入dlartg函数
from scipy.linalg.cython_lapack cimport dlartg
# 从Cython版本的SciPy BLAS模块中导入drot函数
from scipy.linalg.cython_blas cimport drot

# 导入NumPy模块，并将其重命名为np
import numpy as np

# 声明函数givens_elimination，禁用边界检查和数组包装
@cython.boundscheck(False)
@cython.wraparound(False)
def givens_elimination(double[:, ::1] S, double[:] v, const double[:] diag):
    """通过Givens旋转将矩阵对角块归零。

    参数说明：
    S: 双精度浮点数二维数组，表示形状为(n, n)的上三角矩阵。
       该数组将被原地修改。
    v: 双精度浮点数一维数组，形状为(n,)，是形状为(2*n,)完整向量的一部分。
       Givens旋转将应用于该数组，并原地修改，使得在退出时，它包含上述向量的前n个分量。
    diag: 双精度浮点数一维数组，形状为(n,)，表示对角矩阵的对角元素。

    返回值：
    无返回值，函数原地修改输入的S和v数组。
    """
    # 获取对角元素的数量
    cdef int n = diag.shape[0]
    # 声明变量
    cdef int k
    cdef int i, j
    cdef double f, g, r
    cdef double cs, sn
    cdef int one = 1
    # 创建一个空的numpy数组，用于存储对角元素
    cdef double [:] diag_row = np.empty(n)
    cdef double u  # 用于v向量的旋转

    # 对每个对角元素进行循环
    for i in range(n):
        # 如果对角元素为0，则继续下一次循环
        if diag[i] == 0:
            continue
        
        # 将diag_row数组的剩余元素置零，并将当前对角元素复制到diag_row中
        diag_row[i+1:] = 0
        diag_row[i] = diag[i]
        u = 0

        # 对当前对角元素后的每一行进行循环
        for j in range(i, n):
            if diag_row[j] != 0:
                f = S[j, j]
                g = diag_row[j]

                # 计算旋转角的余弦和正弦值
                dlartg(&f, &g, &cs, &sn, &r)
                S[j, j] = r
                # diag_row[j]现在隐含为0

                # 对行中剩余的元素进行旋转
                k = n - j - 1
                if k > 0:
                    drot(&k, &S[j, j+1], &one, &diag_row[j+1], &one, &cs, &sn)

                # 对v数组进行旋转
                f = v[j]
                v[j] = cs * f + sn * u
                u = -sn * f + cs * u
```