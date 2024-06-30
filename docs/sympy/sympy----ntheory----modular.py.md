# `D:\src\scipysrc\sympy\sympy\ntheory\modular.py`

```
from math import prod  # 导入 math 模块中的 prod 函数，用于计算可迭代对象的乘积

from sympy.external.gmpy import gcd, gcdext  # 导入 sympy 外部的 gmpy 模块中的 gcd 和 gcdext 函数
from sympy.ntheory.primetest import isprime  # 导入 sympy 中的 primetest 模块中的 isprime 函数
from sympy.polys.domains import ZZ  # 导入 sympy 中的 polys 模块中的 domains 子模块，以及 ZZ 对象
from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2  # 导入 sympy 中的 polys 模块中的 galoistools 子模块中的 gf_crt, gf_crt1, gf_crt2 函数
from sympy.utilities.misc import as_int  # 导入 sympy 中的 utilities 模块中的 misc 子模块中的 as_int 函数


def symmetric_residue(a, m):
    """Return the residual mod m such that it is within half of the modulus.

    >>> from sympy.ntheory.modular import symmetric_residue
    >>> symmetric_residue(1, 6)
    1
    >>> symmetric_residue(4, 6)
    -2
    """
    if a <= m // 2:
        return a  # 如果 a 小于等于 m 的一半，则直接返回 a
    return a - m  # 否则返回 a - m


def crt(m, v, symmetric=False, check=True):
    r"""Chinese Remainder Theorem.

    The moduli in m are assumed to be pairwise coprime.  The output
    is then an integer f, such that f = v_i mod m_i for each pair out
    of v and m. If ``symmetric`` is False a positive integer will be
    returned, else \|f\| will be less than or equal to the LCM of the
    moduli, and thus f may be negative.

    If the moduli are not co-prime the correct result will be returned
    if/when the test of the result is found to be incorrect. This result
    will be None if there is no solution.

    The keyword ``check`` can be set to False if it is known that the moduli
    are coprime.

    Examples
    ========

    As an example consider a set of residues ``U = [49, 76, 65]``
    and a set of moduli ``M = [99, 97, 95]``. Then we have::

       >>> from sympy.ntheory.modular import crt

       >>> crt([99, 97, 95], [49, 76, 65])
       (639985, 912285)

    This is the correct result because::

       >>> [639985 % m for m in [99, 97, 95]]
       [49, 76, 65]

    If the moduli are not co-prime, you may receive an incorrect result
    if you use ``check=False``:

       >>> crt([12, 6, 17], [3, 4, 2], check=False)
       (954, 1224)
       >>> [954 % m for m in [12, 6, 17]]
       [6, 0, 2]
       >>> crt([12, 6, 17], [3, 4, 2]) is None
       True
       >>> crt([3, 6], [2, 5])
       (5, 6)

    Note: the order of gf_crt's arguments is reversed relative to crt,
    and that solve_congruence takes residue, modulus pairs.

    Programmer's note: rather than checking that all pairs of moduli share
    no GCD (an O(n**2) test) and rather than factoring all moduli and seeing
    that there is no factor in common, a check that the result gives the
    indicated residuals is performed -- an O(n) operation.

    See Also
    ========

    solve_congruence
    sympy.polys.galoistools.gf_crt : low level crt routine used by this routine
    """
    if check:
        m = list(map(as_int, m))  # 如果 check=True，将 m 中的每个元素转换为整数
        v = list(map(as_int, v))  # 如果 check=True，将 v 中的每个元素转换为整数

    result = gf_crt(v, m, ZZ)  # 使用 gf_crt 函数计算 v 和 m 的同余方程的解，使用 ZZ 域
    mm = prod(m)  # 计算 m 中所有元素的乘积，即 mm = m1 * m2 * ... * mn

    if check:
        # 如果 check=True，验证计算出的结果是否满足给定的同余方程
        if not all(v % m == result % m for v, m in zip(v, m)):
            # 如果不满足，则调用 solve_congruence 函数重新计算结果
            result = solve_congruence(*list(zip(v, m)),
                    check=False, symmetric=symmetric)
            if result is None:
                return result  # 如果无法找到解，则返回 None
            result, mm = result  # 否则更新 result 和 mm

    if symmetric:
        return int(symmetric_residue(result, mm)), int(mm)  # 如果 symmetric=True，返回对称余数及 mm
    # 将 result 和 mm 转换为整数后作为元组返回
    return int(result), int(mm)
# 定义函数 `crt1`，用于多次应用中的中国剩余定理的第一部分
def crt1(m):
    # 调用低级别的 `gf_crt1` 函数来执行具体计算，使用整数环 `ZZ`
    return gf_crt1(m, ZZ)


# 定义函数 `crt2`，用于多次应用中的中国剩余定理的第二部分
def crt2(m, v, mm, e, s, symmetric=False):
    # 调用低级别的 `gf_crt2` 函数来执行具体计算，使用整数环 `ZZ`
    result = gf_crt2(v, m, mm, e, s, ZZ)
    
    # 如果 `symmetric` 参数为 True，则返回对称余数及模数的整数值
    if symmetric:
        return int(symmetric_residue(result, mm)), int(mm)
    # 否则，返回计算结果及模数的整数值
    return int(result), int(mm)


# 定义函数 `solve_congruence`，用于求解一组同余方程
def solve_congruence(*remainder_modulus_pairs, **hint):
    """Compute the integer ``n`` that has the residual ``ai`` when it is
    divided by ``mi`` where the ``ai`` and ``mi`` are given as pairs to
    this function: ((a1, m1), (a2, m2), ...). If there is no solution,
    return None. Otherwise return ``n`` and its modulus.

    The ``mi`` values need not be co-prime. If it is known that the moduli are
    not co-prime then the hint ``check`` can be set to False (default=True) and
    the check for a quicker solution via crt() (valid when the moduli are
    co-prime) will be skipped.

    If the hint ``symmetric`` is True (default is False), the value of ``n``
    will be within 1/2 of the modulus, possibly negative.

    Examples
    ========

    >>> from sympy.ntheory.modular import solve_congruence

    What number is 2 mod 3, 3 mod 5 and 2 mod 7?

    >>> solve_congruence((2, 3), (3, 5), (2, 7))
    (23, 105)
    >>> [23 % m for m in [3, 5, 7]]
    [2, 3, 2]

    If you prefer to work with all remainder in one list and
    all moduli in another, send the arguments like this:

    >>> solve_congruence(*zip((2, 3, 2), (3, 5, 7)))
    (23, 105)

    The moduli need not be co-prime; in this case there may or
    may not be a solution:

    >>> solve_congruence((2, 3), (4, 6)) is None
    True

    >>> solve_congruence((2, 3), (5, 6))
    (5, 6)

    The symmetric flag will make the result be within 1/2 of the modulus:
    """
    pass  # 该函数目前没有实现具体功能，只是提供了文档字符串和示例
    def combine(c1, c2):
        """合并两个余数和模数的元组，返回满足条件的 (a, m) 元组。

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Method_of_successive_substitution
        """
        a1, m1 = c1
        a2, m2 = c2
        a, b, c = m1, a2 - a1, m2
        g = gcd(a, b, c)
        a, b, c = [i//g for i in [a, b, c]]
        if a != 1:
            g, inv_a, _ = gcdext(a, c)
            if g != 1:
                return None
            b *= inv_a
        a, m = a1 + m1*b, m1*c
        return a, m

    rm = remainder_modulus_pairs
    symmetric = hint.get('symmetric', False)

    if hint.get('check', True):
        # 将余数-模数对转换为整数，并忽略冗余对，确保唯一的基数集合传递给gf_crt函数，如果它们都是素数。
        rm = [(as_int(r), as_int(m)) for r, m in rm]

        uniq = {}
        for r, m in rm:
            r %= m
            if m in uniq:
                if r != uniq[m]:
                    return None
                continue
            uniq[m] = r
        rm = [(r, m) for m, r in uniq.items()]
        del uniq

        # 如果模数互质，使用CRT会更快；检查所有模数对是否互质变得很慢，但是素数测试是一个很好的权衡。
        if all(isprime(m) for r, m in rm):
            r, m = list(zip(*rm))
            return crt(m, r, symmetric=symmetric, check=False)

    rv = (0, 1)
    for rmi in rm:
        rv = combine(rv, rmi)
        if rv is None:
            break
        n, m = rv
        n = n % m
    else:
        if symmetric:
            return symmetric_residue(n, m), m
        return n, m


这段代码实现了解决线性同余方程组的功能。具体注释如下：

1. `def combine(c1, c2):`
   - 定义了一个函数 `combine`，用于合并两个余数和模数的元组，返回满足条件的 (a, m) 元组。
   - 引用了维基百科中的连续替换法的方法 [1]。

2. `rm = remainder_modulus_pairs`
   - 将输入的余数-模数对列表存储在变量 `rm` 中。

3. `symmetric = hint.get('symmetric', False)`
   - 从 `hint` 中获取 `symmetric` 参数的值，默认为 `False`。

4. `if hint.get('check', True):`
   - 如果 `hint` 中的 `check` 参数为 `True`，执行以下操作：
     - 将余数和模数转换为整数，并确保唯一的基数集合传递给下一个函数（`gf_crt`）。
     - 确保所有模数对互质，以便优化计算速度。
     - 如果所有模数都是素数，则调用 `crt` 函数计算结果。

5. `rv = (0, 1)`
   - 初始化变量 `rv` 为 `(0, 1)`，用于存储累计的余数和模数。

6. `for rmi in rm:`
   - 对余数-模数对列表 `rm` 进行迭代处理。

7. `rv = combine(rv, rmi)`
   - 调用 `combine` 函数将当前的累计余数和模数与当前余数-模数对合并，更新 `rv`。

8. `if rv is None:`
   - 如果合并结果为 `None`，则跳出循环。

9. `else:`
   - 否则，在循环结束时：
     - 如果 `symmetric` 参数为 `True`，返回对称的余数解。
     - 否则，返回标准的余数解 `(n, m)`。

这段代码利用数论中的方法解决了一组线性同余方程，同时通过参数 `symmetric` 控制返回结果的对称性。
```