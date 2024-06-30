# `D:\src\scipysrc\sympy\sympy\ntheory\egyptian_fraction.py`

```
# 导入需要的模块和类
from sympy.core.containers import Tuple
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
import sympy.polys

# 导入 math 模块中的 gcd 函数
from math import gcd

# 定义一个函数用于生成埃及分数
def egyptian_fraction(r, algorithm="Greedy"):
    """
    返回一个有理数 `r` 的埃及分数展开的分母列表。

    Parameters
    ==========

    r : Rational or (p, q)
        一个正有理数，表示为 ``p/q``。
    algorithm : { "Greedy", "Graham Jewett", "Takenouchi", "Golomb" }, optional
        指定使用的算法（默认为 "Greedy"）。

    Examples
    ========

    >>> from sympy import Rational
    >>> from sympy.ntheory.egyptian_fraction import egyptian_fraction
    >>> egyptian_fraction(Rational(3, 7))
    [3, 11, 231]
    >>> egyptian_fraction((3, 7), "Graham Jewett")
    [7, 8, 9, 56, 57, 72, 3192]
    >>> egyptian_fraction((3, 7), "Takenouchi")
    [4, 7, 28]
    >>> egyptian_fraction((3, 7), "Golomb")
    [3, 15, 35]
    >>> egyptian_fraction((11, 5), "Golomb")
    [1, 2, 3, 4, 9, 234, 1118, 2580]

    See Also
    ========

    sympy.core.numbers.Rational

    Notes
    =====

    目前支持以下算法：

    1) Greedy Algorithm

       也称为 Fibonacci-Sylvester 算法 [2]_。
       每一步中，选择小于目标的最大单位分数，并用余数替换目标。

       具有以下显著特性：

       a) 对于最简分数 `p/q`，生成最长为 `p` 的展开序列。即使是大数值的分子，也很少超过一小撮项。

       b) 使用的内存量最小。

       c) 分母可能会急剧增加（标准示例为 5/121 和 31/311）。每一步中分母最多是其平方（双指数增长），通常表现为单指数增长。

    2) Graham Jewett Algorithm

       根据 Graham 和 Jewett 的结果建议的算法。
       注意其有扩展的倾向：结果展开的长度始终为 ``2**(x/gcd(x, y)) - 1``。参见 [3]_。

    3) Takenouchi Algorithm

       根据 Takenouchi（1921）建议的算法。
       与 Graham-Jewett 算法在重复项处理上有所不同。参见 [3]_。

    4) Golomb's Algorithm

       Golumb（1962）提出的一种方法，使用模算术和倒数。
       与 Bleicher（1972）提出的连分数方法产生相同结果。参见 [4]_。

    如果给定的有理数大于或等于 1，使用贪婪算法来计算调和序列 1/1 + 1/2 + 1/3 + ... 的总和，直到再添加一个项会使结果大于给定数值为止。
    这些分母列表会作为前缀添加到所请求算法产生的剩余部分的结果中。
    """
    """
    Convert a rational number `r` into an Egyptian fraction representation using
    the specified algorithm.

    Parameters:
    - r: Rational number to convert into Egyptian fractions
    - algorithm: String specifying the algorithm to use ("Greedy", "Graham Jewett",
                 "Takenouchi", "Golomb")

    Returns:
    - List of integers representing the Egyptian fraction representation of `r`

    Raises:
    - ValueError if `r` is not a Rational or a tuple of ints, or if `r` is non-positive
    - ValueError if an invalid algorithm is specified

    References:
    ===========
    - [1] https://en.wikipedia.org/wiki/Egyptian_fraction
    - [2] https://en.wikipedia.org/wiki/Greedy_algorithm_for_Egyptian_fractions
    - [3] https://www.ics.uci.edu/~eppstein/numth/egypt/conflict.html
    - [4] https://web.archive.org/web/20180413004012/https://ami.ektf.hu/uploads/papers/finalpdf/AMI_42_from129to134.pdf
    """

    # Validate and normalize the rational input `r`
    if not isinstance(r, Rational):
        if isinstance(r, (Tuple, tuple)) and len(r) == 2:
            r = Rational(*r)
        else:
            raise ValueError("Value must be a Rational or tuple of ints")
    if r <= 0:
        raise ValueError("Value must be positive")

    # Handle common cases directly for efficiency
    x, y = r.as_numer_denom()
    if y == 1 and x == 2:
        return [Integer(i) for i in [1, 2, 3, 6]]  # Return precomputed Egyptian fraction

    if x == y + 1:
        return [S.One, y]  # Return precomputed Egyptian fraction

    # Compute prefix and remainder using the harmonic function
    prefix, rem = egypt_harmonic(r)
    if rem == 0:
        return prefix  # Return prefix if remainder is zero

    # Convert remainder into Python integers
    x, y = rem.p, rem.q

    # Choose the appropriate algorithm based on the input
    if algorithm == "Greedy":
        postfix = egypt_greedy(x, y)
    elif algorithm == "Graham Jewett":
        postfix = egypt_graham_jewett(x, y)
    elif algorithm == "Takenouchi":
        postfix = egypt_takenouchi(x, y)
    elif algorithm == "Golomb":
        postfix = egypt_golomb(x, y)
    else:
        raise ValueError("Entered invalid algorithm")

    return prefix + [Integer(i) for i in postfix]  # Combine prefix and postfix to form the final Egyptian fraction
# 使用贪婪算法求解埃及分数表示，假设 gcd(x, y) == 1
def egypt_greedy(x, y):
    # 如果 x 等于 1，直接返回 [y]
    if x == 1:
        return [y]
    else:
        # 计算 a 和 b 的值
        a = (-y) % x
        b = y*(y//x + 1)
        # 计算 a 和 b 的最大公约数
        c = gcd(a, b)
        if c > 1:
            # 如果最大公约数大于 1，化简分数
            num, denom = a//c, b//c
        else:
            num, denom = a, b
        # 递归调用 egypt_greedy 函数，返回结果列表
        return [y//x + 1] + egypt_greedy(num, denom)


# 使用 Graham 和 Jewett 方法求解埃及分数表示，假设 gcd(x, y) == 1
def egypt_graham_jewett(x, y):
    l = [y] * x

    # l 现在是一个包含 x 个元素的列表，其倒数之和为 x/y。
    # 现在我们将继续操作 l 的元素，保持倒数之和不变直到所有元素都是唯一的。

    while len(l) != len(set(l)):
        l.sort()  # 排序列表，以便找到重复的最小对
        for i in range(len(l) - 1):
            if l[i] == l[i + 1]:
                break
        # 找到重复的元素对 l[i] 和 l[i + 1]
        # 根据 Graham 和 Jewett 的结果应用操作
        l[i + 1] = l[i] + 1
        # 继续迭代，直到列表中没有重复元素
        l.append(l[i]*(l[i] + 1))
    # 返回排序后的列表 l
    return sorted(l)


# 使用 Takenouchi 方法求解埃及分数表示，假设 gcd(x, y) == 1
def egypt_takenouchi(x, y):
    # 对于 x 等于 3 的特殊情况
    if x == 3:
        if y % 2 == 0:
            return [y//2, y]
        i = (y - 1)//2
        j = i + 1
        k = j + i
        return [j, k, j*k]
    
    l = [y] * x
    while len(l) != len(set(l)):
        l.sort()
        for i in range(len(l) - 1):
            if l[i] == l[i + 1]:
                break
        k = l[i]
        if k % 2 == 0:
            l[i] = l[i] // 2
            del l[i + 1]
        else:
            l[i], l[i + 1] = (k + 1)//2, k*(k + 1)//2
    # 返回排序后的列表 l
    return sorted(l)


# 使用 Golomb 方法求解埃及分数表示，假设 x < y 且 gcd(x, y) == 1
def egypt_golomb(x, y):
    if x == 1:
        return [y]
    # 计算 x 关于 y 的模反元素
    xp = sympy.polys.ZZ.invert(int(x), int(y))
    rv = [xp*y]
    # 递归调用 egypt_golomb 函数，返回排序后的结果列表
    rv.extend(egypt_golomb((x*xp - 1)//y, xp))
    return sorted(rv)


# 使用调和级数求解埃及分数表示，假设 r 是 Rational 类型
def egypt_harmonic(r):
    rv = []
    d = S.One
    acc = S.Zero
    while acc + 1/d <= r:
        acc += 1/d
        rv.append(d)
        d += 1
    # 返回结果元组，包含调和级数列表和剩余部分
    return (rv, r - acc)
```