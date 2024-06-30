# `D:\src\scipysrc\sympy\sympy\ntheory\multinomial.py`

```
# 导入 sympy 库中的 as_int 函数，用于将输入转换为整数
from sympy.utilities.misc import as_int

# 计算给定整数 n 的二项式系数，并以字典形式返回所有满足 k1 + k2 = n 的组合对 (k1, k2) 及其对应的二项式系数 C_kn
def binomial_coefficients(n):
    n = as_int(n)  # 将 n 转换为整数
    d = {(0, n): 1, (n, 0): 1}  # 初始化字典，包含 (0, n) 和 (n, 0) 对应的系数为 1
    a = 1  # 初始化系数 a 为 1
    for k in range(1, n//2 + 1):
        a = (a * (n - k + 1))//k  # 计算当前 k 对应的二项式系数 a
        d[k, n - k] = d[n - k, k] = a  # 将 (k, n-k) 和 (n-k, k) 的二项式系数设置为 a
    return d  # 返回包含所有二项式系数的字典


# 计算给定整数 n 的二项式系数，并以列表形式返回 Pascal 三角形的第 n 行
def binomial_coefficients_list(n):
    n = as_int(n)  # 将 n 转换为整数
    d = [1] * (n + 1)  # 初始化列表 d，长度为 n+1，所有元素初始值为 1
    a = 1  # 初始化系数 a 为 1
    for k in range(1, n//2 + 1):
        a = (a * (n - k + 1))//k  # 计算当前 k 对应的二项式系数 a
        d[k] = d[n - k] = a  # 将索引 k 和 n-k 处的元素设置为 a
    return d  # 返回包含所有二项式系数的列表


# 计算给定整数 m 和 n 的多项式系数，并以字典形式返回所有满足 k1 + k2 + ... + km = n 的组合对 (k1, k2, ..., km) 及其对应的多项式系数 C_kn
def multinomial_coefficients(m, n):
    m = as_int(m)  # 将 m 转换为整数
    n = as_int(n)  # 将 n 转换为整数
    if not m:  # 如果 m 为 0
        if n:
            return {}  # 若 n 不为 0，则返回空字典
        return {(): 1}  # 若 n 为 0，则返回包含空元组和系数 1 的字典
    if m == 2:  # 如果 m 等于 2
        return binomial_coefficients(n)  # 返回 n 的二项式系数字典
    if m >= 2*n and n > 1:  # 如果 m 大于等于 2*n 并且 n 大于 1
        return dict(multinomial_coefficients_iterator(m, n))  # 返回多项式系数的迭代器生成的字典
    t = [n] + [0] * (m - 1)  # 创建长度为 m 的列表 t，包含 n 和 m-1 个 0
    r = {tuple(t): 1}  # 初始化字典 r，包含元组 t 和系数 1
    if n:
        j = 0  # 初始化 j 为 0，将是最左侧非零位置
    else:
        j = m  # 如果 n 为 0，则将 j 设置为 m
    # 枚举按字典逆序的元组
    # 当 j 小于 m - 1 时，执行循环
    while j < m - 1:
        # 计算下一个元组
        tj = t[j]
        # 如果 j 不为 0，则将 t[j] 置为 0，将原先的 tj 放到 t[0]
        if j:
            t[j] = 0
            t[0] = tj
        # 如果 tj 大于 1
        if tj > 1:
            # 增加 t[j+1] 的值，重置 j 为 0，并初始化 start 为 1 和 v 为 0
            t[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            # 否则增加 j 的值，设置 start 为 j+1，v 为 r[tuple(t)] 的值，增加 t[j] 的值
            j += 1
            start = j + 1
            v = r[tuple(t)]
            t[j] += 1
        # 计算值
        # 注意：v 的初始化在上面完成
        for k in range(start, m):
            # 如果 t[k] 不为 0，则减少 t[k] 的值，增加 v 的值，然后恢复 t[k] 的值
            if t[k]:
                t[k] -= 1
                v += r[tuple(t)]
                t[k] += 1
        # 减少 t[0] 的值
        t[0] -= 1
        # 将结果存储在 r[tuple(t)] 中，值为 (v * tj) // (n - t[0])
        r[tuple(t)] = (v * tj) // (n - t[0])
    # 返回结果字典 r
    return r
# 定义一个生成多项式系数的迭代器函数
def multinomial_coefficients_iterator(m, n, _tuple=tuple):
    """multinomial coefficient iterator

    This routine has been optimized for `m` large with respect to `n` by taking
    advantage of the fact that when the monomial tuples `t` are stripped of
    zeros, their coefficient is the same as that of the monomial tuples from
    ``multinomial_coefficients(n, n)``. Therefore, the latter coefficients are
    precomputed to save memory and time.

    >>> from sympy.ntheory.multinomial import multinomial_coefficients
    >>> m53, m33 = multinomial_coefficients(5,3), multinomial_coefficients(3,3)
    >>> m53[(0,0,0,1,2)] == m53[(0,0,1,0,2)] == m53[(1,0,2,0,0)] == m33[(0,1,2)]
    True

    Examples
    ========

    >>> from sympy.ntheory.multinomial import multinomial_coefficients_iterator
    >>> it = multinomial_coefficients_iterator(20,3)
    >>> next(it)
    ((3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1)
    """

    # 强制转换输入参数为整数
    m = as_int(m)
    n = as_int(n)

    # 如果 m 比较小或者 n 等于 1，则直接计算多项式系数并生成迭代器
    if m < 2*n or n == 1:
        mc = multinomial_coefficients(m, n)
        yield from mc.items()
    else:
        # 否则，使用预先计算好的 n 个数的多项式系数
        mc = multinomial_coefficients(n, n)
        mc1 = {}
        # 对于每个键值对，将其中的零元素滤除并重新组织
        for k, v in mc.items():
            mc1[_tuple(filter(None, k))] = v
        mc = mc1

        # 初始化一个长度为 m 的列表，第一个元素为 n，其余为零
        t = [n] + [0] * (m - 1)
        t1 = _tuple(t)
        b = _tuple(filter(None, t1))
        yield (t1, mc[b])

        # 如果 n 不为零，则 j 为最左边的非零位置
        if n:
            j = 0
        else:
            j = m

        # 按逆字典序枚举元组
        while j < m - 1:
            # 计算下一个元组
            tj = t[j]
            if j:
                t[j] = 0
                t[0] = tj
            if tj > 1:
                t[j + 1] += 1
                j = 0
            else:
                j += 1
                t[j] += 1

            t[0] -= 1
            t1 = _tuple(t)
            b = _tuple(filter(None, t1))
            yield (t1, mc[b])
```