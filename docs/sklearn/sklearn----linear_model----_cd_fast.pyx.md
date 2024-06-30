# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_cd_fast.pyx`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入需要的 Cython 头文件和库
from libc.math cimport fabs  # 导入 fabs 函数
import numpy as np  # 导入 NumPy 库

# 导入需要的 Cython 函数和异常处理相关的模块
from cython cimport floating  # 导入 floating 类型
import warnings  # 导入警告处理模块
from ..exceptions import ConvergenceWarning  # 导入自定义的收敛警告异常

# 导入 Cython 加速的 BLAS 函数和相关常量定义
from ..utils._cython_blas cimport (
    _axpy, _dot, _asum, _gemv, _nrm2, _copy, _scal  # 导入 BLAS 函数
)
from ..utils._cython_blas cimport ColMajor, Trans, NoTrans  # 导入 BLAS 常量定义
from ..utils._typedefs cimport uint32_t  # 导入 uint32_t 类型定义
from ..utils._random cimport our_rand_r  # 导入自定义的随机数生成函数


# 下面两个函数 shamelessly copied from the tree code.

# 定义一个枚举类型，用于指定 rand_r 替代函数的最大值
cdef enum:
    RAND_R_MAX = 2147483647  # rand_r 替代函数的最大返回值，对应 2^31 - 1


# 定义一个内联函数，生成一个 [0, end) 范围内的随机整数
cdef inline uint32_t rand_int(uint32_t end, uint32_t* random_state) noexcept nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


# 定义一个内联函数，返回两个浮点数中的较大者
cdef inline floating fmax(floating x, floating y) noexcept nogil:
    if x > y:
        return x
    return y


# 定义一个内联函数，返回浮点数的符号
cdef inline floating fsign(floating f) noexcept nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


# 定义一个函数，计算数组中绝对值最大的元素
cdef floating abs_max(int n, const floating* a) noexcept nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


# 定义一个函数，计算数组中数值最大的元素
cdef floating max(int n, floating* a) noexcept nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


# 定义一个函数，计算两个数组对应元素之差的绝对值的最大值
cdef floating diff_abs_max(int n, const floating* a, floating* b) noexcept nogil:
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef floating m = fabs(a[0] - b[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m


# 定义主函数，实现 Elastic-Net 回归的坐标下降算法
def enet_coordinate_descent(
    floating[::1] w,  # ElasticNet 系数 w，一维数组
    floating alpha,  # ElasticNet 惩罚项 L1 的系数
    floating beta,  # ElasticNet 惩罚项 L2 的系数
    const floating[::1, :] X,  # 输入特征矩阵 X
    const floating[::1] y,  # 输出向量 y
    unsigned int max_iter,  # 最大迭代次数
    floating tol,  # 容忍度
    object rng,  # 随机数生成器对象
    bint random=0,  # 是否使用随机数
    bint positive=0  # 是否强制系数为非负数
):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2

    Returns
    -------
    w : ndarray of shape (n_features,)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    n_iter : int
        Number of coordinate descent iterations.
    """

    # 根据浮点数类型选择合适的数据类型
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # 将数据信息转换为易于使用的变量
    # 获取样本数量
    cdef unsigned int n_samples = X.shape[0]
    # 获取特征数量
    cdef unsigned int n_features = X.shape[1]

    # 计算矩阵 X 每列的范数的平方
    cdef floating[::1] norm_cols_X = np.square(X).sum(axis=0)

    # 初始化残差向量 R
    cdef floating[::1] R = np.empty(n_samples, dtype=dtype)
    # 初始化向量 XtA
    cdef floating[::1] XtA = np.empty(n_features, dtype=dtype)

    # 定义临时变量 tmp
    cdef floating tmp
    # 定义权重参数 w_ii
    cdef floating w_ii
    # 定义最大的权重变化量 d_w_max
    cdef floating d_w_max
    # 定义最大权重 w_max
    cdef floating w_max
    # 定义权重变化量 d_w_ii
    cdef floating d_w_ii
    # 定义误差值 gap，初始为 tol + 1.0
    cdef floating gap = tol + 1.0
    # 定义权重变化阈值 d_w_tol，初始为 tol
    cdef floating d_w_tol = tol
    # 定义对偶范数 dual_norm_XtA
    cdef floating dual_norm_XtA
    # 定义残差的范数平方 R_norm2
    cdef floating R_norm2
    # 定义权重的范数平方 w_norm2
    cdef floating w_norm2
    # 定义 L1 范数 l1_norm
    cdef floating l1_norm
    # 定义常数值 const
    cdef floating const
    # 定义矩阵 A 的范数平方 A_norm2
    cdef floating A_norm2
    # 定义迭代计数器 ii
    cdef unsigned int ii
    # 初始化迭代次数 n_iter 为 0
    cdef unsigned int n_iter = 0
    # 定义用于生成随机数种子的状态变量 rand_r_state_seed
    cdef uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    # 定义指向随机数种子状态的指针 rand_r_state
    cdef uint32_t* rand_r_state = &rand_r_state_seed

    # 如果 alpha 和 beta 均为 0，则发出警告
    if alpha == 0 and beta == 0:
        warnings.warn("Coordinate descent with no regularization may lead to "
                      "unexpected results and is discouraged.")

    # 返回结果：权重向量 w 的 NumPy 数组表示、误差值 gap、容差 tol、迭代次数 n_iter + 1
    return np.asarray(w), gap, tol, n_iter + 1
# 定义稀疏弹性网络协调下降算法的函数，使用了Cython实现

# Notes for sample_weight:
# For dense X, one centers X and y and then rescales them by sqrt(sample_weight).
# Here, for sparse X, we get the sample_weight averaged center X_mean. We take care
# that every calculation results as if we had rescaled y and X (and therefore also
# X_mean) by sqrt(sample_weight) without actually calculating the square root.
# We work with:
#     yw = sample_weight
#     R = sample_weight * residual
#     norm_cols_X = np.sum(sample_weight * (X - X_mean)**2, axis=0)

# 获取数据信息到易于处理的变量中
cdef unsigned int n_samples = y.shape[0]  # 样本数
cdef unsigned int n_features = w.shape[0]  # 特征数

# 计算 X 列的范数
cdef unsigned int ii
cdef floating[:] norm_cols_X

# X_indptr 数组的起始指针值
cdef unsigned int startptr = X_indptr[0]
cdef unsigned int endptr

# 初始化残差的值
# R = y - Zw，加权版本为 R = sample_weight * (y - Zw)
cdef floating[::1] R
cdef floating[::1] XtA
cdef const floating[::1] yw

# 根据浮点数类型设置 dtype
if floating is float:
    dtype = np.float32
else:
    dtype = np.float64

norm_cols_X = np.zeros(n_features, dtype=dtype)  # 初始化 X 列的范数数组
XtA = np.zeros(n_features, dtype=dtype)  # 初始化 XtA 数组

cdef floating tmp
cdef floating w_ii
cdef floating d_w_max
cdef floating w_max
cdef floating d_w_ii
cdef floating X_mean_ii
cdef floating R_sum = 0.0
cdef floating R_norm2
cdef floating w_norm2
cdef floating A_norm2
cdef floating l1_norm
cdef floating normalize_sum
cdef floating gap = tol + 1.0  # 初始间隙为设定的 tol 加 1.0
cdef floating d_w_tol = tol
cdef floating dual_norm_XtA
cdef unsigned int jj
cdef unsigned int n_iter = 0  # 迭代次数
cdef unsigned int f_iter
cdef uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)  # 随机数生成器种子
cdef uint32_t* rand_r_state = &rand_r_state_seed
cdef bint center = False  # 是否对数据进行中心化的标志位
    # 定义一个 C 语言风格的布尔变量，检查是否没有样本权重（sample_weight 是 None）
    cdef bint no_sample_weights = sample_weight is None
    # 定义一个 C 语言风格的整型变量 kk

    # 如果没有样本权重
    if no_sample_weights:
        # 将 yw 初始化为 y 的引用
        yw = y
        # 将 R 初始化为 y 的拷贝
        R = y.copy()
    else:
        # 将 yw 初始化为 sample_weight 与 y 的按元素乘积
        yw = np.multiply(sample_weight, y)
        # 将 R 初始化为 yw 的拷贝
        R = yw.copy()

    # 将 w 转换为 NumPy 数组，并返回 w, gap, tol, n_iter + 1
    return np.asarray(w), gap, tol, n_iter + 1
#python
def enet_coordinate_descent_gram(
    floating[::1] w,
    floating alpha,
    floating beta,
    const floating[:, ::1] Q,
    const floating[::1] q,
    const floating[:] y,
    unsigned int max_iter,
    floating tol,
    object rng,
    bint random=0,
    bint positive=0
):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * w^T Q w - q^T w + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2

        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y

    Returns
    -------
    w : ndarray of shape (n_features,)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    n_iter : int
        Number of coordinate descent iterations.
    """

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # get the data information into easy vars
    cdef unsigned int n_features = Q.shape[0]  # Number of features in Q (Gram matrix)

    # initial value "Q w" which will be kept of up to date in the iterations
    cdef floating[:] H = np.dot(Q, w)  # Compute Q * w and store in H

    cdef floating[:] XtA = np.zeros(n_features, dtype=dtype)  # Initialize XtA as zeros
    cdef floating tmp  # Temporary floating point variable
    cdef floating w_ii  # Individual element of w
    cdef floating d_w_max  # Maximum change in w across iterations
    cdef floating w_max  # Maximum value in w across iterations
    cdef floating d_w_ii  # Change in individual element of w
    cdef floating q_dot_w  # Dot product of q and w
    cdef floating w_norm2  # L2 norm squared of w
    cdef floating gap = tol + 1.0  # Initial value for the dual gap
    cdef floating d_w_tol = tol  # Tolerance for changes in w
    cdef floating dual_norm_XtA  # Dual norm of XtA
    cdef unsigned int ii  # Index variable
    cdef unsigned int n_iter = 0  # Iteration counter
    cdef unsigned int f_iter  # Temporary unsigned integer for iterations
    cdef uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)  # Seed for random number generation
    cdef uint32_t* rand_r_state = &rand_r_state_seed  # Pointer to the random state seed

    cdef floating y_norm2 = np.dot(y, y)  # Compute L2 norm squared of y
    cdef floating* w_ptr = &w[0]  # Pointer to the first element of w
    cdef const floating* Q_ptr = &Q[0, 0]  # Pointer to the first element of Q
    cdef const floating* q_ptr = &q[0]  # Pointer to the first element of q
    cdef floating* H_ptr = &H[0]  # Pointer to the first element of H
    cdef floating* XtA_ptr = &XtA[0]  # Pointer to the first element of XtA
    tol = tol * y_norm2  # Update tol to be tol * y_norm2

    if alpha == 0:
        warnings.warn(
            "Coordinate descent without L1 regularization may "
            "lead to unexpected results and is discouraged. "
            "Set l1_ratio > 0 to add L1 regularization."
        )

    return np.asarray(w), gap, tol, n_iter + 1



def enet_coordinate_descent_multi_task(
    const floating[::1, :] W,
    floating l1_reg,
    floating l2_reg,
    const floating[::1, :] X,
    const floating[::1, :] Y,
    unsigned int max_iter,
    floating tol,
    object rng,
    bint random=0
):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net multi-task regression

        We minimize

        0.5 * norm(Y - X W.T, 2)^2 + l1_reg ||W.T||_21 + 0.5 * l2_reg norm(W.T, 2)^2

    Returns
    -------
    W : ndarray of shape (n_tasks, n_features)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    """
    n_iter : int
        Number of coordinate descent iterations.
    """

    # 检查浮点类型，确定数据类型
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # 将数据信息存入易于访问的变量中
    cdef unsigned int n_samples = X.shape[0]  # 样本数
    cdef unsigned int n_features = X.shape[1]  # 特征数
    cdef unsigned int n_tasks = Y.shape[1]  # 任务数

    # 初始化存储 XtA 的数组
    cdef floating[:, ::1] XtA = np.zeros((n_features, n_tasks), dtype=dtype)
    cdef floating XtA_axis1norm
    cdef floating dual_norm_XtA

    # 初始化残差的初始值
    cdef floating[::1, :] R = np.zeros((n_samples, n_tasks), dtype=dtype, order='F')

    # 初始化其他需要用到的数组
    cdef floating[::1] norm_cols_X = np.zeros(n_features, dtype=dtype)
    cdef floating[::1] tmp = np.zeros(n_tasks, dtype=dtype)
    cdef floating[::1] w_ii = np.zeros(n_tasks, dtype=dtype)
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating nn
    cdef floating W_ii_abs_max
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating R_norm
    cdef floating w_norm
    cdef floating ry_sum
    cdef floating l21_norm
    cdef unsigned int ii
    cdef unsigned int jj
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef uint32_t* rand_r_state = &rand_r_state_seed

    # 获取输入数据的指针
    cdef const floating* X_ptr = &X[0, 0]
    cdef const floating* Y_ptr = &Y[0, 0]

    # 如果 l1_reg 为 0，发出警告
    if l1_reg == 0:
        warnings.warn(
            "Coordinate descent with l1_reg=0 may lead to unexpected"
            " results and is discouraged."
        )

    # 返回结果数组 W，以及迭代次数、间隙和容差的值
    return np.asarray(W), gap, tol, n_iter + 1
```