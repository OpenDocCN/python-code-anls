# `D:\src\scipysrc\sympy\sympy\polys\heuristicgcd.py`

```
"""Heuristic polynomial GCD algorithm (HEUGCD). """

from .polyerrors import HeuristicGCDFailed  # 导入自定义异常类 HeuristicGCDFailed

HEU_GCD_MAX = 6  # 设置最大迭代次数为 6

def heugcd(f, g):
    """
    Heuristic polynomial GCD in ``Z[X]``.

    Given univariate polynomials ``f`` and ``g`` in ``Z[X]``, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    ``f`` and ``g`` at certain points and computing (fast) integer GCD
    of those evaluations. The polynomial GCD is recovered from the integer
    image by interpolation. The evaluation process reduces f and g variable
    by variable into a large integer. The final step is to verify if the
    interpolated polynomial is the correct GCD. This gives cofactors of
    the input polynomials as a side effect.

    Examples
    ========

    >>> from sympy.polys.heuristicgcd import heugcd
    >>> from sympy.polys import ring, ZZ

    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> h, cff, cfg = heugcd(f, g)
    >>> h, cff, cfg
    (x + y, x + y, x)

    >>> cff*h == f
    True
    >>> cfg*h == g
    True

    References
    ==========

    .. [1] [Liao95]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ  # 断言：多项式 f 和 g 在同一个环上，并且环的域是整数环

    ring = f.ring  # 获取多项式环
    x0 = ring.gens[0]  # 获取多项式环的第一个生成元
    domain = ring.domain  # 获取环的域

    gcd, f, g = f.extract_ground(g)  # 提取 f 和 g 的公因子

    f_norm = f.max_norm()  # 计算 f 的最大范数
    g_norm = g.max_norm()  # 计算 g 的最大范数

    B = domain(2*min(f_norm, g_norm) + 29)  # 计算 B 的值

    x = max(min(B, 99*domain.sqrt(B)),  # 计算 x 的初值
            2*min(f_norm // abs(f.LC), g_norm // abs(g.LC)) + 4)

    for i in range(0, HEU_GCD_MAX):  # 迭代 HEU_GCD_MAX 次数
        ff = f.evaluate(x0, x)  # 在 x0 处对 f 进行求值
        gg = g.evaluate(x0, x)  # 在 x0 处对 g 进行求值

        if ff and gg:  # 如果 ff 和 gg 非零
            if ring.ngens == 1:  # 如果环中只有一个生成元
                h, cff, cfg = domain.cofactors(ff, gg)  # 计算 ff 和 gg 的最大公因子及其余因子
            else:
                h, cff, cfg = heugcd(ff, gg)  # 递归调用 heugcd 函数

            h = _gcd_interpolate(h, x, ring)  # 插值得到 h 的多项式
            h = h.primitive()[1]  # 取 h 的原始多项式部分

            cff_, r = f.div(h)  # 将 f 除以 h

            if not r:  # 如果余数为零
                cfg_, r = g.div(h)  # 将 g 除以 h

                if not r:  # 如果余数为零
                    h = h.mul_ground(gcd)  # h 乘以 gcd
                    return h, cff_, cfg_

            cff = _gcd_interpolate(cff, x, ring)  # 插值得到 cff 的多项式

            h, r = f.div(cff)  # 将 f 除以 cff

            if not r:  # 如果余数为零
                cfg_, r = g.div(h)  # 将 g 除以 h

                if not r:  # 如果余数为零
                    h = h.mul_ground(gcd)  # h 乘以 gcd
                    return h, cff, cfg_

            cfg = _gcd_interpolate(cfg, x, ring)  # 插值得到 cfg 的多项式

            h, r = g.div(cfg)  # 将 g 除以 cfg

            if not r:  # 如果余数为零
                cff_, r = f.div(h)  # 将 f 除以 h

                if not r:  # 如果余数为零
                    h = h.mul_ground(gcd)  # h 乘以 gcd
                    return h, cff_, cfg

        x = 73794*x * domain.sqrt(domain.sqrt(x)) // 27011  # 更新 x 的值
    # 抛出 HeuristicGCDFailed 异常，说明启发式最大公约数算法失败且没有找到合适的解决方法
    raise HeuristicGCDFailed('no luck')
# 根据整数最大公约数（GCD），插值多项式的 GCD 结果
def _gcd_interpolate(h, x, ring):
    # 初始化多项式 f 和指数 i
    f, i = ring.zero, 0

    # 如果环中只有一个生成元（即一维环）
    if ring.ngens == 1:
        # 当 h 非零时循环执行以下操作
        while h:
            # 计算 h 对 x 的模
            g = h % x
            # 如果 g 大于 x 的一半，则减去 x
            if g > x // 2:
                g -= x
            # 更新 h 为 (h - g) / x
            h = (h - g) // x

            # 如果 g 非零，将 X**i*g 加入多项式 f 中
            if g:
                f[(i,)] = g
            # 增加指数 i
            i += 1
    else:
        # 当 h 非零时循环执行以下操作
        while h:
            # 通过 trunc_ground 方法得到 h 对 x 的地板除结果 g
            g = h.trunc_ground(x)
            # 更新 h 为 (h - g) / x
            h = (h - g).quo_ground(x)

            # 如果 g 非零，将 X**i*g 中的每一项 (monom, coeff) 加入多项式 f 中
            if g:
                for monom, coeff in g.iterterms():
                    f[(i,) + monom] = coeff
            # 增加指数 i
            i += 1

    # 如果多项式 f 的最高次项系数小于 0，则返回负数 f
    if f.LC < 0:
        return -f
    else:
        return f
```