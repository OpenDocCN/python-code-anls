# `D:\src\scipysrc\sympy\sympy\series\approximants.py`

```
# 导入从 sympy 库中需要的类和函数
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.polytools import lcm
from sympy.utilities import public

# 定义一个公共函数，用于生成序列的 Pade 近似
@public
def approximants(l, X=Symbol('x'), simplify=False):
    """
    Return a generator for consecutive Pade approximants for a series.
    It can also be used for computing the rational generating function of a
    series when possible, since the last approximant returned by the generator
    will be the generating function (if any).

    Explanation
    ===========

    The input list can contain more complex expressions than integer or rational
    numbers; symbols may also be involved in the computation. An example below
    show how to compute the generating function of the whole Pascal triangle.

    The generator can be asked to apply the sympy.simplify function on each
    generated term, which will make the computation slower; however it may be
    useful when symbols are involved in the expressions.

    Examples
    ========

    >>> from sympy.series import approximants
    >>> from sympy import lucas, fibonacci, symbols, binomial
    >>> g = [lucas(k) for k in range(16)]
    >>> [e for e in approximants(g)]
    [2, -4/(x - 2), (5*x - 2)/(3*x - 1), (x - 2)/(x**2 + x - 1)]

    >>> h = [fibonacci(k) for k in range(16)]
    >>> [e for e in approximants(h)]
    [x, -x/(x - 1), (x**2 - x)/(2*x - 1), -x/(x**2 + x - 1)]

    >>> x, t = symbols("x,t")
    >>> p=[sum(binomial(k,i)*x**i for i in range(k+1)) for k in range(16)]
    >>> y = approximants(p, t)
    >>> for k in range(3): print(next(y))
    1
    (x + 1)/((-x - 1)*(t*(x + 1) + (x + 1)/(-x - 1)))
    nan

    >>> y = approximants(p, t, simplify=True)
    >>> for k in range(3): print(next(y))
    1
    -1/(t*(x + 1) - 1)
    nan

    See Also
    ========

    sympy.concrete.guess.guess_generating_function_rational
    mpmath.pade
    """
    
    # 从 sympy.simplify 模块导入 simplify 函数，并从 sympy.simplify.radsimp 模块导入 denom 函数
    from sympy.simplify import simplify as simp
    from sympy.simplify.radsimp import denom
    
    # 初始化 Pade 近似的分子和分母系数
    p1, q1 = [S.One], [S.Zero]
    p2, q2 = [S.Zero], [S.One]
    # 当列表 l 的长度不为零时执行循环
    while len(l):
        # 初始化变量 b 为 0
        b = 0
        # 当列表 l 中索引为 b 的元素为 0 时执行循环
        while l[b]==0:
            # b 自增
            b += 1
            # 如果 b 等于列表 l 的长度，返回，结束函数
            if b == len(l):
                return
        # 初始化列表 m，第一个元素为 S.One/l[b]
        m = [S.One/l[b]]
        # 从索引 b+1 开始到列表 l 的末尾遍历
        for k in range(b+1, len(l)):
            # 初始化 s 为 0
            s = 0
            # 从索引 b 到索引 k-1 遍历
            for j in range(b, k):
                # 计算 s -= l[j+1] * m[b-j-1]
                s -= l[j+1] * m[b-j-1]
            # 将 s/l[b] 添加到列表 m 中
            m.append(s/l[b])
        # 更新列表 l 为列表 m
        l = m
        # 将 l 的第一个元素赋值给 a，将 l 的第一个元素设为 0
        a, l[0] = l[0], 0
        # 初始化列表 p，长度为 max(len(p2), b+len(p1))，元素全部为 0
        p = [0] * max(len(p2), b+len(p1))
        # 初始化列表 q，长度为 max(len(q2), b+len(q1))，元素全部为 0
        q = [0] * max(len(q2), b+len(q1))
        # 遍历 p2 列表的索引
        for k in range(len(p2)):
            # 计算 a*p2[k] 并赋值给 p[k]
            p[k] = a*p2[k]
        # 遍历从 b 开始到 b+len(p1)-1 的索引
        for k in range(b, b+len(p1)):
            # 将 p1[k-b] 加到 p[k]
            p[k] += p1[k-b]
        # 遍历 q2 列表的索引
        for k in range(len(q2)):
            # 计算 a*q2[k] 并赋值给 q[k]
            q[k] = a*q2[k]
        # 遍历从 b 开始到 b+len(q1)-1 的索引
        for k in range(b, b+len(q1)):
            # 将 q1[k-b] 加到 q[k]
            q[k] += q1[k-b]
        # 移除 p 中末尾为 0 的元素
        while p[-1]==0: p.pop()
        # 移除 q 中末尾为 0 的元素
        while q[-1]==0: q.pop()
        # 更新 p1 为 p2，p2 为 p；更新 q1 为 q2，q2 为 q

        # 计算分母 c
        c = 1
        # 遍历 p 中的元素，计算它们的最小公倍数并赋值给 c
        for x in p:
            c = lcm(c, denom(x))
        # 遍历 q 中的元素，计算它们的最小公倍数并赋值给 c
        for x in q:
            c = lcm(c, denom(x))
        # 计算输出值 out
        out = ( sum(c*e*X**k for k, e in enumerate(p))
              / sum(c*e*X**k for k, e in enumerate(q)) )
        # 如果需要简化结果
        if simplify:
            # 使用 simp 函数简化 out 并返回
            yield(simp(out))
        else:
            # 直接返回 out
            yield out
    # 循环结束，返回
    return
```