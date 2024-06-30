# `D:\src\scipysrc\scipy\scipy\stats\_stats_pythran.py`

```
import numpy as np  # 导入NumPy库，用于处理数组和数值计算


#pythran export _Aij(float[:,:], int, int)
#pythran export _Aij(int[:,:], int, int)
def _Aij(A, i, j):
    """计算列联表的左上角和右下角块的元素之和。"""
    # 参见 `somersd` 参考文献 [2] 第309页底部
    return A[:i, :j].sum() + A[i+1:, j+1:].sum()


#pythran export _Dij(float[:,:], int, int)
#pythran export _Dij(int[:,:], int, int)
def _Dij(A, i, j):
    """计算列联表的左下角和右上角块的元素之和。"""
    # 参见 `somersd` 参考文献 [2] 第309页底部
    return A[i+1:, :j].sum() + A[:i, j+1:].sum()


# pythran export _concordant_pairs(float[:,:])
# pythran export _concordant_pairs(int[:,:])
def _concordant_pairs(A):
    """计算同序对的两倍数目，排除并列的情况。"""
    # 参见 `somersd` 参考文献 [2] 第309页底部
    m, n = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j]*_Aij(A, i, j)
    return count


# pythran export _discordant_pairs(float[:,:])
# pythran export _discordant_pairs(int[:,:])
def _discordant_pairs(A):
    """计算不同序对的两倍数目，排除并列的情况。"""
    # 参见 `somersd` 参考文献 [2] 第309页底部
    m, n = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j]*_Dij(A, i, j)
    return count


#pythran export _a_ij_Aij_Dij2(float[:,:])
#pythran export _a_ij_Aij_Dij2(int[:,:])
def _a_ij_Aij_Dij2(A):
    """Kendall's tau和Somers' D的ASE中出现的一个项。"""
    # 参见 `somersd` 参考文献 [2] 第4节:
    # 修改的ASE用于检验零假设...
    m, n = A.shape
    count = 0
    for i in range(m):
        for j in range(n):
            count += A[i, j]*(_Aij(A, i, j) - _Dij(A, i, j))**2
    return count


#pythran export _compute_outer_prob_inside_method(int64, int64, int64, int64)
def _compute_outer_prob_inside_method(m, n, g, h):
    """
    计算不严格位于两条对角线内部的路径的比例。

    Parameters
    ----------
    m : 整数
        m > 0
    n : 整数
        n > 0
    g : 整数
        m和n的最大公约数
    h : 整数
        0 <= h <= m和n的最小公倍数

    Returns
    -------
    p : 浮点数
        不严格位于两条对角线内部的路径的比例。

    经典算法计算从 (0, 0) 到 (m, n) 的整数格子路径，
    这些路径满足 |x/m - y/n| < h / lcm(m, n)。
    路径以+1的步长在正x或正y方向上移动。
    然而，我们关心的是1 - 比例以计算p值，
    因此我们修改递归直接计算1 - p，
    同时保持在由Hodges描述的“inside method”内部。

    我们一般遵循Hodges对Drion/Gnedenko/Korolyuk的处理。
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.

    有关1-p递归的详细信息请参见
    """
    Viehmann, T.: "Numerically more stable computation of the p-values
    for the two-sample Kolmogorov-Smirnov test," arXiv: 2102.08037

    """
    # 概率在 m, n 中是对称的。下面的计算假设 m >= n。
    if m < n:
        m, n = n, m
    # 将 m 和 n 各自除以 g 得到的商
    mg = m // g
    ng = n // g

    # 计算从 (0, 0) 到 (m, n) 的整数格路径的数量，满足 |nx/g - my/g| < h。
    # 计算矩阵 A 满足以下条件：
    #  A(x, 0) = A(0, y) = 1
    #  A(x, y) = A(x, y-1) + A(x-1, y), 对于 x, y >= 1，但如果 |x/m - y/n| >= h，则 A(x, y) = 0
    # 概率为 A(m, n)/binom(m+n, n)
    # 对于 m==n, m==n*p 的情况有优化
    # 只需保留 A 的单列，并且只需滑动窗口的一部分
    # minj 用于追踪滑动窗口的起始位置。
    minj, maxj = 0, min(int(np.ceil(h / mg)), n + 1)
    curlen = maxj - minj
    # 创建一个足够长的向量以容纳可能需要的最大窗口。
    lenA = min(2 * maxj + 2, n + 1)
    # 这是一个整数计算，但条目本质上是二项式系数，因此增长迅速。
    # 在计算每列后进行缩放，避免在最后除以大的二项式系数，但不足以避免计算过程中出现的大动态范围。
    # 相反，基于列中最右边项的大小重缩放，并单独跟踪指数，并在计算结束时应用它。同样，当乘以二项式系数时也是如此。
    dtype = np.float64
    A = np.ones(lenA, dtype=dtype)
    # 初始化第一列
    A[minj:maxj] = 0.0
    for i in range(1, m + 1):
        # 生成下一列。
        # 首先计算滑动窗口
        lastminj, lastlen = minj, curlen
        minj = max(int(np.floor((ng * i - h) / mg)) + 1, 0)
        minj = min(minj, n)
        maxj = min(int(np.ceil((ng * i + h) / mg)), n + 1)
        if maxj <= minj:
            return 1.0
        # 填充值。不幸的是，我们不能使用累加和。
        val = 0.0 if minj == 0 else 1.0
        for jj in range(maxj - minj):
            j = jj + minj
            val = (A[jj + minj - lastminj] * i + val * j) / (i + j)
            A[jj] = val
        curlen = maxj - minj
        if lastlen > curlen:
            # 将一些未使用的元素设置为 1
            A[maxj - minj:maxj - minj + (lastlen - curlen)] = 1

    return A[maxj - minj - 1]
# 定义 siegelslopes 函数，用于计算斜率和截距的中位数
# 函数可以被 Pythran 编译器导出，支持两种数据类型的输入：float32 和 float64
def siegelslopes(y, x, method):
    # 计算 x 向量的扩展矩阵，并与原始 x 向量的每个元素做差，得到 deltax 矩阵
    deltax = np.expand_dims(x, 1) - x
    # 计算 y 向量的扩展矩阵，并与原始 y 向量的每个元素做差，得到 deltay 矩阵
    deltay = np.expand_dims(y, 1) - y
    # 初始化斜率和截距列表
    slopes, intercepts = [], []

    # 遍历 x 向量的每个元素
    for j in range(len(x)):
        # 找出当前行中非零元素的索引
        id_nonzero, = np.nonzero(deltax[j, :])
        # 根据非零元素索引，计算斜率 slopes_j
        slopes_j = deltay[j, id_nonzero] / deltax[j, id_nonzero]
        # 计算斜率的中位数 medslope_j
        medslope_j = np.median(slopes_j)
        # 将中位数斜率添加到 slopes 列表中
        slopes.append(medslope_j)
        
        # 如果方法为 'separate'
        if method == 'separate':
            # 计算 z 值
            z = y * x[j] - y[j] * x
            # 根据非零元素索引，计算截距的中位数 medintercept_j
            medintercept_j = np.median(z[id_nonzero] / deltax[j, id_nonzero])
            # 将中位数截距添加到 intercepts 列表中
            intercepts.append(medintercept_j)

    # 计算斜率的全局中位数 medslope
    medslope = np.median(np.asarray(slopes))
    # 根据方法计算最终的截距 medinter
    if method == "separate":
        medinter = np.median(np.asarray(intercepts))
    else:
        medinter = np.median(y - medslope * x)

    # 返回斜率的中位数 medslope 和截距的中位数 medinter
    return medslope, medinter
```