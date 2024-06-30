# `D:\src\scipysrc\scipy\scipy\special\_precompute\utils.py`

```
# 尝试导入 mpmath 库，如果导入失败则忽略
try:
    import mpmath as mp
except ImportError:
    pass

# 尝试从 sympy.abc 中导入变量 x，如果导入失败则忽略
try:
    from sympy.abc import x
except ImportError:
    pass

# 定义函数 lagrange_inversion，使用拉格朗日反演公式计算系列 b，使得 f(g(x)) = g(f(x)) = x mod x**n，其中 f(x) 是输入系列 a 的多项式
def lagrange_inversion(a):
    """Given a series

    f(x) = a[1]*x + a[2]*x**2 + ... + a[n-1]*x**(n - 1),

    use the Lagrange inversion formula to compute a series

    g(x) = b[1]*x + b[2]*x**2 + ... + b[n-1]*x**(n - 1)

    so that f(g(x)) = g(f(x)) = x mod x**n. We must have a[0] = 0, so
    necessarily b[0] = 0 too.

    The algorithm is naive and could be improved, but speed isn't an
    issue here and it's easy to read.

    """
    # 系数个数 n 等于输入数组 a 的长度
    n = len(a)
    # 计算多项式 f(x) = a[1]*x + a[2]*x**2 + ... + a[n-1]*x**(n - 1)
    f = sum(a[i]*x**i for i in range(n))
    # 计算 h(x) = (x/f(x)).series(x, 0, n).removeO()，即 f(x) 的反函数的前 n 项
    h = (x/f).series(x, 0, n).removeO()
    # 计算 h(x) 的幂级数展开，存储在列表 hpower 中
    hpower = [h**0]
    for k in range(n):
        hpower.append((hpower[-1]*h).expand())
    # 计算系列 b，满足 g(x) = b[1]*x + b[2]*x**2 + ... + b[n-1]*x**(n - 1)
    b = [mp.mpf(0)]  # 初始设置 b[0] = 0
    for k in range(1, n):
        b.append(hpower[k].coeff(x, k - 1)/k)
    # 将 b 中的每个元素转换为 mpmath 中的 mp.mpf 类型
    b = [mp.mpf(x) for x in b]
    # 返回计算得到的系列 b
    return b
```