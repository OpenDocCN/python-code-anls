# `D:\src\scipysrc\scipy\scipy\interpolate\_pade.py`

```
# 从 numpy 模块中导入所需的函数和对象
from numpy import zeros, asarray, eye, poly1d, hstack, r_
# 从 scipy 模块中导入 linalg 函数
from scipy import linalg

# 定义模块的公开接口列表，用于模块导入时的限定
__all__ = ["pade"]

# 定义函数 pade，用于返回一个多项式的 Pade 近似
def pade(an, m, n=None):
    """
    Return Pade approximation to a polynomial as the ratio of two polynomials.

    Parameters
    ----------
    an : (N,) array_like
        Taylor series coefficients.
    m : int
        The order of the returned approximating polynomial `q`.
    n : int, optional
        The order of the returned approximating polynomial `p`. By default,
        the order is ``len(an)-1-m``.

    Returns
    -------
    p, q : Polynomial class
        The Pade approximation of the polynomial defined by `an` is
        ``p(x)/q(x)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import pade
    >>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]
    >>> p, q = pade(e_exp, 2)

    >>> e_exp.reverse()
    >>> e_poly = np.poly1d(e_exp)

    Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``

    >>> e_poly(1)
    2.7166666666666668

    >>> p(1)/q(1)
    2.7179487179487181

    """
    # 将输入的系数数组 an 转换为 numpy 数组
    an = asarray(an)
    # 如果未指定 n 的值，则计算默认值 len(an)-1-m
    if n is None:
        n = len(an) - 1 - m
        # 检查默认计算的 n 是否合理，若不合理则抛出异常
        if n < 0:
            raise ValueError("Order of q <m> must be smaller than len(an)-1.")
    # 若 n 小于 0，抛出异常
    if n < 0:
        raise ValueError("Order of p <n> must be greater than 0.")
    # 计算多项式总阶数 N
    N = m + n
    # 检查阶数 N 是否超出系数数组 an 的长度，若超出则抛出异常
    if N > len(an)-1:
        raise ValueError("Order of q+p <m+n> must be smaller than len(an).")
    # 截取前 N+1 项系数
    an = an[:N+1]
    # 创建单位矩阵 Akj，大小为 (N+1) x (n+1)
    Akj = eye(N+1, n+1, dtype=an.dtype)
    # 创建零矩阵 Bkj，大小为 (N+1) x m
    Bkj = zeros((N+1, m), dtype=an.dtype)
    # 填充 Bkj 的上三角部分，用于构造解线性方程组所需的矩阵
    for row in range(1, m+1):
        Bkj[row,:row] = -(an[:row])[::-1]
    for row in range(m+1, N+1):
        Bkj[row,:] = -(an[row-m:row])[::-1]
    # 拼接 Akj 和 Bkj，形成完整的系数矩阵 C
    C = hstack((Akj, Bkj))
    # 解线性方程组 C * pq = an，得到系数向量 pq
    pq = linalg.solve(C, an)
    # 提取出分子多项式 p 的系数向量
    p = pq[:n+1]
    # 构造分母多项式 q，首项为 1.0，其余为 pq 中剩余部分
    q = r_[1.0, pq[n+1:]]
    # 返回 p 和 q 构成的多项式对象
    return poly1d(p[::-1]), poly1d(q[::-1])
```