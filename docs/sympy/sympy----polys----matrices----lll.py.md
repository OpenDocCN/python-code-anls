# `D:\src\scipysrc\sympy\sympy\polys\matrices\lll.py`

```
from __future__ import annotations

from math import floor as mfloor  # 导入 math 模块中的 floor 函数，并重命名为 mfloor

from sympy.polys.domains import ZZ, QQ  # 从 sympy.polys.domains 模块导入 ZZ 和 QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError  # 从 sympy.polys.matrices.exceptions 模块导入异常类

# 定义一个函数 _ddm_lll，用于执行 LLL 算法
def _ddm_lll(x, delta=QQ(3, 4), return_transform=False):
    if QQ(1, 4) >= delta or delta >= QQ(1, 1):  # 如果 delta 不在 (0.25, 1) 的范围内，抛出 DMValueError 异常
        raise DMValueError("delta must lie in range (0.25, 1)")
    if x.shape[0] > x.shape[1]:  # 如果输入矩阵的行数大于列数，抛出 DMShapeError 异常
        raise DMShapeError("input matrix must have shape (m, n) with m <= n")
    if x.domain != ZZ:  # 如果输入矩阵的域不是整数环 ZZ，抛出 DMDomainError 异常
        raise DMDomainError("input matrix domain must be ZZ")
    
    m = x.shape[0]  # 获取输入矩阵的行数
    n = x.shape[1]  # 获取输入矩阵的列数
    k = 1  # 初始化变量 k 为 1
    y = x.copy()  # 复制输入矩阵 x，赋值给 y
    y_star = x.zeros((m, n), QQ)  # 创建一个 m x n 的零矩阵 y_star，元素类型为 QQ
    mu = x.zeros((m, m), QQ)  # 创建一个 m x m 的零矩阵 mu，元素类型为 QQ
    g_star = [QQ(0, 1) for _ in range(m)]  # 创建一个长度为 m 的列表 g_star，每个元素为 QQ(0, 1)
    half = QQ(1, 2)  # 定义 half 为 QQ(1, 2)，即 1/2
    T = x.eye(m, ZZ) if return_transform else None  # 如果 return_transform 为 True，则创建一个 m x m 的单位矩阵 T；否则 T 为 None
    linear_dependent_error = "input matrix contains linearly dependent rows"  # 定义线性相关错误信息字符串

    # 定义一个函数 closest_integer，用于返回最接近 x 的整数
    def closest_integer(x):
        return ZZ(mfloor(x + half))  # 返回 x 加上 half 后最接近的整数，转换为 ZZ 类型

    # 定义一个函数 lovasz_condition，判断 Lovász 条件是否满足
    def lovasz_condition(k: int) -> bool:
        return g_star[k] >= ((delta - mu[k][k - 1] ** 2) * g_star[k - 1])

    # 定义一个函数 mu_small，判断 mu[k][j] 的绝对值是否小于等于 half
    def mu_small(k: int, j: int) -> bool:
        return abs(mu[k][j]) <= half

    # 定义一个函数 dot_rows，计算两行在指定列上的点积
    def dot_rows(x, y, rows: tuple[int, int]):
        return sum(x[rows[0]][z] * y[rows[1]][z] for z in range(x.shape[1]))

    # 定义一个函数 reduce_row，对指定的行进行行变换，使之满足 LLL 算法的条件
    def reduce_row(T, mu, y, rows: tuple[int, int]):
        r = closest_integer(mu[rows[0]][rows[1]])  # 计算 mu[rows[0]][rows[1]] 最接近的整数 r
        y[rows[0]] = [y[rows[0]][z] - r * y[rows[1]][z] for z in range(n)]  # 更新 y[rows[0]] 的值
        mu[rows[0]][:rows[1]] = [mu[rows[0]][z] - r * mu[rows[1]][z] for z in range(rows[1])]  # 更新 mu[rows[0]] 的值
        mu[rows[0]][rows[1]] -= r  # 更新 mu[rows[0]][rows[1]] 的值
        if return_transform:
            T[rows[0]] = [T[rows[0]][z] - r * T[rows[1]][z] for z in range(m)]  # 如果需要返回变换矩阵，则更新 T[rows[0]] 的值

    # 主循环，对输入矩阵的每一行应用 LLL 算法
    for i in range(m):
        y_star[i] = [QQ.convert_from(z, ZZ) for z in y[i]]  # 将 y[i] 中的元素转换为 QQ 类型，存入 y_star[i]
        for j in range(i):
            row_dot = dot_rows(y, y_star, (i, j))  # 计算 y[i] 和 y_star[j] 的点积
            try:
                mu[i][j] = row_dot / g_star[j]  # 计算 mu[i][j] 的值
            except ZeroDivisionError:
                raise DMRankError(linear_dependent_error)  # 如果除以 g_star[j] 时出现 ZeroDivisionError，则抛出 DMRankError 异常
            y_star[i] = [y_star[i][z] - mu[i][j] * y_star[j][z] for z in range(n)]  # 更新 y_star[i] 的值
        g_star[i] = dot_rows(y_star, y_star, (i, i))  # 计算 g_star[i] 的值
    # 当 k 小于 m 时执行循环
    while k < m:
        # 如果 mu_small(k, k - 1) 不成立，则调用 reduce_row 函数处理矩阵 T、mu、y
        if not mu_small(k, k - 1):
            reduce_row(T, mu, y, (k, k - 1))
        # 如果满足 Lovasz 条件
        if lovasz_condition(k):
            # 从 k-2 到 0 的范围内遍历
            for l in range(k - 2, -1, -1):
                # 如果 mu_small(k, l) 不成立，则调用 reduce_row 函数处理矩阵 T、mu、y
                if not mu_small(k, l):
                    reduce_row(T, mu, y, (k, l))
            # 增加 k 的值
            k += 1
        else:
            # 计算 nu 和 alpha 的值
            nu = mu[k][k - 1]
            alpha = g_star[k] + nu ** 2 * g_star[k - 1]
            try:
                # 计算 beta 的值
                beta = g_star[k - 1] / alpha
            except ZeroDivisionError:
                # 如果 alpha 为零，则抛出 DMRankError 异常
                raise DMRankError(linear_dependent_error)
            # 更新 mu[k][k-1]、g_star[k] 和 g_star[k-1] 的值
            mu[k][k - 1] = nu * beta
            g_star[k] = g_star[k] * beta
            g_star[k - 1] = alpha
            # 交换 y[k] 和 y[k-1] 的值
            y[k], y[k - 1] = y[k - 1], y[k]
            # 交换 mu[k][:k-1] 和 mu[k-1][:k-1] 的值
            mu[k][:k - 1], mu[k - 1][:k - 1] = mu[k - 1][:k - 1], mu[k][:k - 1]
            # 对于 k+1 到 m-1 的范围内的每一个 i
            for i in range(k + 1, m):
                # 计算 xi 的值
                xi = mu[i][k]
                # 更新 mu[i][k] 和 mu[i][k-1] 的值
                mu[i][k] = mu[i][k - 1] - nu * xi
                mu[i][k - 1] = mu[k][k - 1] * mu[i][k] + xi
            # 如果 return_transform 为真，则交换 T[k] 和 T[k-1] 的值
            if return_transform:
                T[k], T[k - 1] = T[k - 1], T[k]
            # 更新 k 的值为 k-1 和 1 之间的最大值
            k = max(k - 1, 1)
    # 断言从 1 到 m-1 的每个 i 都满足 Lovasz 条件
    assert all(lovasz_condition(i) for i in range(1, m))
    # 断言对于每对 (i, j)，其中 0 <= j < i < m，都满足 mu_small(i, j)
    assert all(mu_small(i, j) for i in range(m) for j in range(i))
    # 返回 y 和 T
    return y, T
# 使用 delta-delta modulation (DDM) 算法来进行低位限幅（LLL）处理
def ddm_lll(x, delta=QQ(3, 4)):
    # 调用底层函数 _ddm_lll 进行处理，返回结果中仅返回处理后的数据，而不返回变换
    return _ddm_lll(x, delta=delta, return_transform=False)[0]


# 使用 delta-delta modulation (DDM) 算法来进行低位限幅（LLL）处理，并返回处理后的变换
def ddm_lll_transform(x, delta=QQ(3, 4)):
    # 调用底层函数 _ddm_lll 进行处理，返回结果中包含处理后的数据及其变换
    return _ddm_lll(x, delta=delta, return_transform=True)
```