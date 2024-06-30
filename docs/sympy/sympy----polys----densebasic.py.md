# `D:\src\scipysrc\sympy\sympy\polys\densebasic.py`

```
"""Basic tools for dense recursive polynomials in ``K[x]`` or ``K[X]``. """

# 从 sympy.core 导入 igcd 函数
from sympy.core import igcd
# 从 sympy.polys.monomials 导入 monomial_min 和 monomial_div 函数
from sympy.polys.monomials import monomial_min, monomial_div
# 从 sympy.polys.orderings 导入 monomial_key 函数
from sympy.polys.orderings import monomial_key

# 导入 random 模块
import random

# 将负无穷定义为浮点数 -inf
ninf = float('-inf')

# 定义函数 poly_LC，用于返回多项式 f 的首项系数
def poly_LC(f, K):
    """
    Return leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import poly_LC

    >>> poly_LC([], ZZ)
    0
    >>> poly_LC([ZZ(1), ZZ(2), ZZ(3)], ZZ)
    1

    """
    # 如果 f 为空列表，则返回域 K 中的零元素
    if not f:
        return K.zero
    else:
        # 否则返回 f 的第一个元素，即首项系数
        return f[0]

# 定义函数 poly_TC，用于返回多项式 f 的尾项系数
def poly_TC(f, K):
    """
    Return trailing coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import poly_TC

    >>> poly_TC([], ZZ)
    0
    >>> poly_TC([ZZ(1), ZZ(2), ZZ(3)], ZZ)
    3

    """
    # 如果 f 为空列表，则返回域 K 中的零元素
    if not f:
        return K.zero
    else:
        # 否则返回 f 的最后一个元素，即尾项系数
        return f[-1]

# 定义别名 dup_LC 和 dmp_LC，它们分别等同于 poly_LC 函数
dup_LC = dmp_LC = poly_LC

# 定义别名 dup_TC 和 dmp_TC，它们分别等同于 poly_TC 函数
dup_TC = dmp_TC = poly_TC

# 定义函数 dmp_ground_LC，用于返回多重多项式 f 的地位（最高）首项系数
def dmp_ground_LC(f, u, K):
    """
    Return the ground leading coefficient.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_LC

    >>> f = ZZ.map([[[1], [2, 3]]])

    >>> dmp_ground_LC(f, 2, ZZ)
    1

    """
    # 当 u 大于 0 时循环
    while u:
        # 调用 dmp_LC 函数获取 f 的首项系数
        f = dmp_LC(f, K)
        # u 减 1
        u -= 1

    # 返回多重多项式 f 的最高地位（最高次）首项系数
    return dup_LC(f, K)

# 定义函数 dmp_ground_TC，用于返回多重多项式 f 的地位（最高）尾项系数
def dmp_ground_TC(f, u, K):
    """
    Return the ground trailing coefficient.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_TC

    >>> f = ZZ.map([[[1], [2, 3]]])

    >>> dmp_ground_TC(f, 2, ZZ)
    3

    """
    # 当 u 大于 0 时循环
    while u:
        # 调用 dmp_TC 函数获取 f 的尾项系数
        f = dmp_TC(f, K)
        # u 减 1
        u -= 1

    # 返回多重多项式 f 的最高地位（最高次）尾项系数
    return dup_TC(f, K)

# 定义函数 dmp_true_LT，用于返回多重多项式 f 的真实领导项
def dmp_true_LT(f, u, K):
    """
    Return the leading term ``c * x_1**n_1 ... x_k**n_k``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_true_LT

    >>> f = ZZ.map([[4], [2, 0], [3, 0, 0]])

    >>> dmp_true_LT(f, 1, ZZ)
    ((2, 0), 4)

    """
    # 初始化空列表 monom
    monom = []

    # 当 u 大于 0 时循环
    while u:
        # 将 f 的长度减 1 添加到 monom 中
        monom.append(len(f) - 1)
        # 将 f 更新为 f 的第一个元素
        f, u = f[0], u - 1

    # 如果 f 为空列表，则将 0 添加到 monom 中
    if not f:
        monom.append(0)
    else:
        # 否则将 f 的长度减 1 添加到 monom 中
        monom.append(len(f) - 1)

    # 返回元组 (monom, dup_LC(f, K))，表示真实领导项及其首项系数
    return tuple(monom), dup_LC(f, K)

# 定义函数 dup_degree，用于返回单变量多项式 f 的最高次数
def dup_degree(f):
    """
    Return the leading degree of ``f`` in ``K[x]``.

    Note that the degree of 0 is negative infinity (``float('-inf')``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_degree

    >>> f = ZZ.map([1, 2, 0, 3])

    >>> dup_degree(f)
    3

    """
    # 如果 f 为空列表，则返回负无穷
    if not f:
        return ninf
    # 否则返回 f 的长度减 1，即最高次数
    return len(f) - 1

# 定义函数 dmp_degree，用于返回多变量多项式 f 在第一个变量 x_0 中的最高次数
def dmp_degree(f, u):
    """
    Return the leading degree of ``f`` in ``x_0`` in ``K[X]``.

    Note that the degree of 0 is negative infinity (``float('-inf')``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree
    # 调用 dmp_degree 函数计算给定参数 [[[]]] 的结果，期望结果为负无穷
    >>> dmp_degree([[[]]], 2)
    -inf

    # 将列表 [[2], [1, 2, 3]] 映射为整数环 ZZ 中的元素
    >>> f = ZZ.map([[2], [1, 2, 3]])

    # 调用 dmp_degree 函数计算给定参数 f 和 1 的结果，期望结果为 1
    >>> dmp_degree(f, 1)
    1

    """
    # 如果 dmp_zero_p 函数对参数 f 和 u 返回 True，则返回负无穷
    if dmp_zero_p(f, u):
        return ninf
    # 否则，返回 f 的长度减 1
    else:
        return len(f) - 1
# 递归辅助函数，用于计算在多项式环中给定变量的最高次数
def _rec_degree_in(g, v, i, j):
    """Recursive helper function for :func:`dmp_degree_in`."""
    # 如果当前层次达到目标层次，则返回多项式 g 在变量 v 中的最高次数
    if i == j:
        return dmp_degree(g, v)

    # 将变量 v 和层次 i 向前移动一步
    v, i = v - 1, i + 1

    # 递归计算 g 中每个子多项式在更低一级的变量中的最高次数，并返回最大值
    return max(_rec_degree_in(c, v, i, j) for c in g)


def dmp_degree_in(f, j, u):
    """
    Return the leading degree of ``f`` in ``x_j`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree_in

    >>> f = ZZ.map([[2], [1, 2, 3]])

    >>> dmp_degree_in(f, 0, 1)
    1
    >>> dmp_degree_in(f, 1, 1)
    2

    """
    # 如果 j 为 0，则直接返回 f 关于变量 u 的最高次数
    if not j:
        return dmp_degree(f, u)
    # 如果 j 不在合法范围内，则抛出索引错误
    if j < 0 or j > u:
        raise IndexError("0 <= j <= %s expected, got %s" % (u, j))

    # 调用递归函数 _rec_degree_in 计算 f 在变量 x_j 中的最高次数
    return _rec_degree_in(f, u, 0, j)


def _rec_degree_list(g, v, i, degs):
    """Recursive helper for :func:`dmp_degree_list`."""
    # 更新 degs[i] 为多项式 g 在变量 v 中的最高次数
    degs[i] = max(degs[i], dmp_degree(g, v))

    # 如果变量 v 大于 0，则向下一级变量 v - 1 和 i + 1 递归处理 g 中的每个子多项式
    if v > 0:
        v, i = v - 1, i + 1

        for c in g:
            _rec_degree_list(c, v, i, degs)


def dmp_degree_list(f, u):
    """
    Return a list of degrees of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree_list

    >>> f = ZZ.map([[1], [1, 2, 3]])

    >>> dmp_degree_list(f, 1)
    (1, 2)

    """
    # 初始化 degs 列表为负无穷，长度为 u + 1
    degs = [ninf]*(u + 1)
    # 调用递归函数 _rec_degree_list 更新 degs 列表中每个元素的值
    _rec_degree_list(f, u, 0, degs)
    # 返回 degs 列表的元组形式
    return tuple(degs)


def dup_strip(f):
    """
    Remove leading zeros from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_strip

    >>> dup_strip([0, 0, 1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    # 如果 f 为空或者第一个非零系数不为 0，则直接返回 f
    if not f or f[0]:
        return f

    # 否则从头开始查找第一个非零系数的位置 i，并返回 f[i:] 列表
    i = 0

    for cf in f:
        if cf:
            break
        else:
            i += 1

    return f[i:]


def dmp_strip(f, u):
    """
    Remove leading zeros from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_strip

    >>> dmp_strip([[], [0, 1, 2], [1]], 1)
    [[0, 1, 2], [1]]

    """
    # 如果变量 u 为 0，则调用 dup_strip 函数处理 f
    if not u:
        return dup_strip(f)

    # 如果 f 是零多项式，则直接返回 f
    if dmp_zero_p(f, u):
        return f

    # 初始化 i 和 v
    i, v = 0, u - 1

    # 从头开始查找第一个非零子多项式的位置 i
    for c in f:
        if not dmp_zero_p(c, v):
            break
        else:
            i += 1

    # 如果 i 等于 f 的长度，则返回零多项式，否则返回 f[i:]
    if i == len(f):
        return dmp_zero(u)
    else:
        return f[i:]


def _rec_validate(f, g, i, K):
    """Recursive helper for :func:`dmp_validate`."""
    # 如果 g 不是列表，则验证其类型是否与 K 的类型匹配，不匹配则抛出类型错误
    if not isinstance(g, list):
        if K is not None and not K.of_type(g):
            raise TypeError("%s in %s in not of type %s" % (g, f, K.dtype))

        return {i - 1}
    # 如果 g 是空列表，则返回 {i}
    elif not g:
        return {i}
    else:
        levels = set()

        # 递归处理 g 中的每个子多项式，并将结果合并到 levels 中
        for c in g:
            levels |= _rec_validate(f, c, i + 1, K)

        return levels


def _rec_strip(g, v):
    """Recursive helper for :func:`_rec_strip`."""
    # 如果变量 v 为 0，则调用 dup_strip 处理 g
    if not v:
        return dup_strip(g)

    # 否则将 v 减少 1，并对 g 中的每个子多项式递归调用 _rec_strip
    w = v - 1

    return dmp_strip([ _rec_strip(c, w) for c in g ], v)


def dmp_validate(f, K=None):
    """
    Return the number of levels in ``f`` and recursively strip it.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_validate

    >>> dmp_validate([[], [0, 1, 2], [1]])
    ([[1, 2], [1]], 1)

    >>> dmp_validate([[1], 1])
    Traceback (most recent call last):
    ...
    ValueError: invalid data structure for a multivariate polynomial

    """
    # 调用 _rec_validate 函数，计算多项式 f 的层级并进行验证
    levels = _rec_validate(f, f, 0, K)

    # 弹出 levels 列表的最后一个元素并赋值给 u
    u = levels.pop()

    # 如果 levels 列表为空，则调用 _rec_strip 函数对 f 进行剥离并返回结果及 u
    if not levels:
        return _rec_strip(f, u), u
    else:
        # 如果 levels 列表不为空，则抛出 ValueError 异常，表明多变量多项式的数据结构无效
        raise ValueError(
            "invalid data structure for a multivariate polynomial")
def dup_reverse(f):
    """
    计算 ``x**n * f(1/x)``, 即在 ``K[x]`` 中反转 ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_reverse

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_reverse(f)
    [3, 2, 1]

    """
    # 调用 dup_strip 函数，传入 f 的反转列表作为参数，返回结果
    return dup_strip(list(reversed(f)))


def dup_copy(f):
    """
    在 ``K[x]`` 中创建多项式 ``f`` 的新副本.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_copy

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_copy([1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    # 返回列表 f 的浅拷贝
    return list(f)


def dmp_copy(f, u):
    """
    在 ``K[X]`` 中创建多项式 ``f`` 的新副本.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_copy

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_copy(f, 1)
    [[1], [1, 2]]

    """
    # 如果 u 为 0，则返回 f 的列表形式的拷贝
    if not u:
        return list(f)

    v = u - 1

    # 返回递归地对 f 中每个子列表应用 dmp_copy 的结果所构成的列表
    return [ dmp_copy(c, v) for c in f ]


def dup_to_tuple(f):
    """
    将多项式 ``f`` 转换为元组形式.

    用于哈希计算。与 dup_copy() 类似.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_copy

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_copy([1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    # 返回列表 f 的元组形式
    return tuple(f)


def dmp_to_tuple(f, u):
    """
    将多项式 ``f`` 转换为嵌套元组的形式.

    用于哈希计算。与 dmp_copy() 类似.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_to_tuple

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_to_tuple(f, 1)
    ((1,), (1, 2))

    """
    # 如果 u 为 0，则返回 f 的元组形式
    if not u:
        return tuple(f)
    v = u - 1

    # 返回递归地对 f 中每个子列表应用 dmp_to_tuple 的结果所构成的元组
    return tuple(dmp_to_tuple(c, v) for c in f)


def dup_normal(f, K):
    """
    在给定域 K 中标准化一元多项式.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_normal

    >>> dup_normal([0, 1, 2, 3], ZZ)
    [1, 2, 3]

    """
    # 返回对 f 中每个系数应用 K.normal 函数后所得到的列表，然后调用 dup_strip 函数
    return dup_strip([ K.normal(c) for c in f ])


def dmp_normal(f, u, K):
    """
    在给定域 K 中标准化多元多项式.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_normal

    >>> dmp_normal([[], [0, 1, 2]], 1, ZZ)
    [[1, 2]]

    """
    # 如果 u 为 0，则直接调用 dup_normal 函数标准化 f
    if not u:
        return dup_normal(f, K)

    v = u - 1

    # 返回递归地对 f 中每个子列表应用 dmp_normal 函数后所得到的列表，然后调用 dmp_strip 函数
    return dmp_strip([ dmp_normal(c, v, K) for c in f ], u)


def dup_convert(f, K0, K1):
    """
    将多项式 ``f`` 的基础域从 ``K0`` 转换到 ``K1``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_convert

    >>> R, x = ring("x", ZZ)

    >>> dup_convert([R(1), R(2)], R.to_domain(), ZZ)
    [1, 2]

    """
    # 返回对 f 中每个元素应用 K1.convert 函数所得到的结果的列表
    return [ K1.convert(c) for c in f ]
    # 调用 dup_convert 函数并传入参数 [ZZ(1), ZZ(2)]，期望返回 ZZ 类型的列表
    >>> dup_convert([ZZ(1), ZZ(2)], ZZ, R.to_domain())
    
    # 下面是一个条件语句，检查 K0 是否不为 None 并且 K0 是否等于 K1
    if K0 is not None and K0 == K1:
        # 如果条件成立，直接返回输入的参数 f
        return f
    else:
        # 如果条件不成立，则执行下面的操作
    
    # 返回一个由 K1.convert(c, K0) 转换后的列表，其中 c 是 f 中的每个元素
    return dup_strip([ K1.convert(c, K0) for c in f ])
# 返回 ``f`` 的地域域从 ``K0`` 到 ``K1`` 的转换。
def dmp_convert(f, u, K0, K1):
    # 如果 ``u`` 为假（0），调用 ``dup_convert`` 函数进行 ``f`` 的转换
    if not u:
        return dup_convert(f, K0, K1)
    # 如果 ``K0`` 不为 ``None`` 并且等于 ``K1``，直接返回 ``f``
    if K0 is not None and K0 == K1:
        return f

    # 计算 ``v = u - 1``
    v = u - 1

    # 递归地转换 ``f`` 中的每个系数到新的地域域 ``K1``，并使用 ``dmp_strip`` 剥离结果
    return dmp_strip([ dmp_convert(c, v, K0, K1) for c in f ], u)


# 将 ``f`` 的地域域从 SymPy 转换到 ``K``
def dup_from_sympy(f, K):
    # 使用列表推导式将 ``f`` 中的每个元素从 SymPy 转换到 ``K``
    return dup_strip([ K.from_sympy(c) for c in f ])


# 将 ``f`` 的地域域从 SymPy 转换到 ``K``，支持多项式
def dmp_from_sympy(f, u, K):
    # 如果 ``u`` 为假（0），调用 ``dup_from_sympy`` 函数进行 ``f`` 的转换
    if not u:
        return dup_from_sympy(f, K)

    # 计算 ``v = u - 1``
    v = u - 1

    # 递归地将 ``f`` 中的每个系数从 SymPy 转换到 ``K``，并使用 ``dmp_strip`` 剥离结果
    return dmp_strip([ dmp_from_sympy(c, v, K) for c in f ], u)


# 返回 ``f`` 在 ``K[x]`` 中的第 ``n`` 个系数
def dup_nth(f, n, K):
    # 如果 ``n`` 小于 0，抛出异常
    if n < 0:
        raise IndexError("'n' must be non-negative, got %i" % n)
    # 如果 ``n`` 大于等于 ``f`` 的长度，返回 ``K.zero``
    elif n >= len(f):
        return K.zero
    else:
        # 返回 ``f`` 在 ``K[x]`` 中从高次到低次的第 ``n`` 个系数
        return f[dup_degree(f) - n]


# 返回 ``f`` 在 ``K[x]`` 中的第 ``n`` 个系数，支持多项式
def dmp_nth(f, n, u, K):
    # 如果 ``n`` 小于 0，抛出异常
    if n < 0:
        raise IndexError("'n' must be non-negative, got %i" % n)
    # 如果 ``n`` 大于等于 ``f`` 的长度，返回一个 ``u - 1`` 零多项式
    elif n >= len(f):
        return dmp_zero(u - 1)
    else:
        # 返回 ``f`` 在 ``K[x]`` 中从高次到低次的第 ``n`` 个系数
        return f[dmp_degree(f, u) - n]


# 返回 ``f`` 在 ``K[x]`` 中的地 ``N`` 个系数
def dmp_ground_nth(f, N, u, K):
    # 将 ``u`` 赋值给 ``v``
    v = u
    # 对于列表 N 中的每个元素 n，依次进行以下操作
    for n in N:
        # 如果 n 小于 0，则抛出索引错误，要求 n 必须是非负数，给出具体的错误信息
        if n < 0:
            raise IndexError("`n` must be non-negative, got %i" % n)
        # 如果 n 大于等于列表 f 的长度，则返回零多项式 K.zero
        elif n >= len(f):
            return K.zero
        else:
            # 计算多项式 f 关于变量 v 的阶数
            d = dmp_degree(f, v)
            # 如果阶数 d 是负无穷，则将 d 置为 -1
            if d == ninf:
                d = -1
            # 从多项式 f 中取出特定的子多项式，更新 f 和变量 v 的次数
            f, v = f[d - n], v - 1

    # 返回处理后的多项式 f
    return f
# 如果在多项式环 K[X] 中 f 是零，则返回 True
def dmp_zero_p(f, u):
    while u:
        # 如果 f 的长度不为 1，说明多项式不是零
        if len(f) != 1:
            return False
        # 取多项式的第一个元素，继续向内层查看
        f = f[0]
        u -= 1
    # 返回 f 是否为零
    return not f


# 返回一个多元零多项式
def dmp_zero(u):
    r = []
    # 生成多元零多项式的层次结构
    for i in range(u):
        r = [r]
    return r


# 如果在多项式环 K[X] 中 f 是单位元，则返回 True
def dmp_one_p(f, u, K):
    # 调用 dmp_ground_p 判断 f 是否为 K 的单位元
    return dmp_ground_p(f, K.one, u)


# 返回一个多元单位元
def dmp_one(u, K):
    # 调用 dmp_ground 返回 K 的单位元
    return dmp_ground(K.one, u)


# 判断在多项式环 K[X] 中 f 是否为常数
def dmp_ground_p(f, c, u):
    # 如果 c 是 None 或者 c 为零，则调用 dmp_zero_p 判断 f 是否为零多项式
    if c is not None and not c:
        return dmp_zero_p(f, u)

    while u:
        # 如果 f 的长度不为 1，说明多项式不是常数
        if len(f) != 1:
            return False
        # 取多项式的第一个元素，继续向内层查看
        f = f[0]
        u -= 1

    # 如果 c 是 None，判断 f 的长度是否小于等于 1；否则，判断 f 是否与常数 c 相等
    if c is None:
        return len(f) <= 1
    else:
        return f == [c]


# 返回一个多元常数
def dmp_ground(c, u):
    # 如果 c 为零，返回一个多元零多项式
    if not c:
        return dmp_zero(u)

    # 生成多元常数的层次结构
    for i in range(u + 1):
        c = [c]

    return c


# 返回一个包含多个多元零多项式的列表
def dmp_zeros(n, u, K):
    # 如果 n 为零，返回空列表
    if not n:
        return []

    # 如果 u 小于零，返回包含 n 个 K 的零元素的列表；否则，返回包含 n 个多元零多项式的列表
    if u < 0:
        return [K.zero]*n
    else:
        return [dmp_zero(u) for i in range(n)]


# 返回一个包含多个多元常数的列表
def dmp_grounds(c, n, u):
    # 如果 n 为零，返回空列表
    if not n:
        return []

    # 如果 u 小于零，返回包含 n 个常数 c 的列表；否则，返回包含 n 个多元常数的列表
    if u < 0:
        return [c]*n
    else:
        return [dmp_ground(c, u) for i in range(n)]


# 判断在多项式环 K[X] 中 f 是否为负数
def dmp_negative_p(f, u, K):
    # 此函数未实现完整，暂时没有具体的代码实现
    Return ``True`` if ``LC(f)`` is negative.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_negative_p

    >>> dmp_negative_p([[ZZ(1)], [-ZZ(1)]], 1, ZZ)
    False
    >>> dmp_negative_p([[-ZZ(1)], [ZZ(1)]], 1, ZZ)
    True

    """
    # 调用 dmp_ground_LC 函数获取多项式 f 的领导系数 LC(f)，并检查其是否为负数
    return K.is_negative(dmp_ground_LC(f, u, K))
# 将 ``K[x]`` 多项式转换为 ``dict`` 类型表示
def dup_to_raw_dict(f, K=None, zero=False):
    """
    Convert ``K[x]`` polynomial to a raw ``dict``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_to_raw_dict

    >>> dup_to_raw_dict([1, 0, 5, 0, 7])
    {0: 7, 2: 5, 4: 1}
    >>> dup_to_raw_dict([])
    {}

    """
    if not f and zero:
        return {0: K.zero}

    # 确定多项式的最高次数
    n, result = len(f) - 1, {}

    # 从高次到低次遍历多项式系数
    for k in range(0, n + 1):
        # 如果系数不为零，则加入结果字典中
        if f[n - k]:
            result[k] = f[n - k]

    return result
    # 将 ``K[x]`` 多项式转换为原始的 ``dict`` 格式。
    
    Examples
    ========
    
    >>> from sympy.polys.densebasic import dup_to_raw_dict
    
    >>> dup_to_raw_dict([1, 0, 5, 0, 7])
    {0: 7, 2: 5, 4: 1}
    
    """
    # 如果输入的多项式 f 是空且 zero 标志为真，则返回一个包含单项式 0: K.zero 的字典
    if not f and zero:
        return {0: K.zero}
    
    # 计算多项式 f 的最高次数 n
    n, result = len(f) - 1, {}
    
    # 遍历多项式的每一个系数
    for k in range(0, n + 1):
        # 如果 f 的第 n-k 项系数不为零
        if f[n - k]:
            # 将非零系数添加到结果字典中，键为 k，值为对应的系数值
            result[k] = f[n - k]
    
    # 返回结果字典，其中包含了多项式的非零系数
    return result
# 将 ``K[X]`` 多项式转换为字典表示
def dmp_to_dict(f, u, K=None, zero=False):
    # 如果度数为零，调用 ``dup_to_dict`` 函数将多项式转换为字典
    if not u:
        return dup_to_dict(f, K, zero=zero)

    # 如果多项式为零且需要返回零多项式，则返回一个度数为 ``u+1`` 的零多项式字典
    if dmp_zero_p(f, u) and zero:
        return {(0,)*(u + 1): K.zero}

    # 计算多项式的最高次数和减少一个变量后的次数
    n, v, result = dmp_degree(f, u), u - 1, {}

    # 处理特殊情况：多项式的次数为无穷
    if n == ninf:
        n = -1

    # 遍历多项式的每一项
    for k in range(0, n + 1):
        # 递归调用 ``dmp_to_dict`` 函数，处理减少一个变量后的多项式
        h = dmp_to_dict(f[n - k], v)

        # 将每一项的指数和系数加入结果字典中
        for exp, coeff in h.items():
            result[(k,) + exp] = coeff

    return result


# 将 ``K[..x_i..x_j..]`` 多项式转换为 ``K[..x_j..x_i..]`` 多项式
def dmp_swap(f, i, j, u, K):
    # 如果变量索引超出范围，抛出索引错误
    if i < 0 or j < 0 or i > u or j > u:
        raise IndexError("0 <= i < j <= %s expected" % u)
    elif i == j:
        return f

    # 将多项式转换为字典表示
    F, H = dmp_to_dict(f, u), {}

    # 遍历原多项式的每一项
    for exp, coeff in F.items():
        # 根据给定的索引重新排列指数，构造新的多项式字典
        H[exp[:i] + (exp[j],) +
          exp[i + 1:j] +
          (exp[i],) + exp[j + 1:]] = coeff

    # 根据新的多项式字典构造多项式对象并返回
    return dmp_from_dict(H, u, K)


# 返回在 ``K[x_{P(1)},..,x_{P(n)}]`` 下排列的多项式
def dmp_permute(f, P, u, K):
    # 将多项式转换为字典表示
    F, H = dmp_to_dict(f, u), {}

    # 遍历原多项式的每一项
    for exp, coeff in F.items():
        new_exp = [0]*len(exp)

        # 根据排列列表 P，重新排列每一项的指数
        for e, p in zip(exp, P):
            new_exp[p] = e

        # 将重新排列后的指数和系数加入结果字典中
        H[tuple(new_exp)] = coeff

    # 根据结果字典构造多项式对象并返回
    return dmp_from_dict(H, u, K)


# 返回一个 ``l`` 级别嵌套的多变量值
def dmp_nest(f, l, K):
    # 如果输入不是列表，则将其视为常数，使用 ``dmp_ground`` 函数构造多项式
    if not isinstance(f, list):
        return dmp_ground(f, l)

    # 将多项式逐级嵌套 ``l`` 次
    for i in range(l):
        f = [f]

    return f


# 返回 ``l`` 级别提升的多变量多项式
def dmp_raise(f, l, u, K):
    # 如果提升级别为零，直接返回多项式
    if not l:
        return f

    # 如果变量数为零且多项式为空，则返回零多项式
    if not u:
        if not f:
            return dmp_zero(l)

        # 否则，对每一个系数应用 ``dmp_ground`` 函数，构造提升级别 ``l-1`` 的多项式列表
        k = l - 1
        return [ dmp_ground(c, k) for c in f ]
    # 计算 v 的值，其中 u 是输入参数减去 1
    v = u - 1

    # 返回一个列表，列表的每个元素是调用 dmp_raise 函数得到的结果，参数分别为 c, l, v, K
    # f 是输入的列表，在列表推导式中遍历每个元素 c
    return [ dmp_raise(c, l, v, K) for c in f ]
# 定义一个函数，用于将多项式集合中的每个多项式按照指定规则进行压缩或简化处理，以在域 K[x] 中映射 x**m 到 y。
def dup_multi_deflate(polys, K):
    # 初始化整体最大公约数为 0
    G = 0
    
    # 遍历多项式集合 polys 中的每个多项式 p
    for p in polys:
        # 如果多项式 p 的次数小于等于 0，返回 1 和原始多项式集合 polys
        if dup_degree(p) <= 0:
            return 1, polys
        
        # 初始化当前多项式 p 的最大公约数为 0
        g = 0
        
        # 遍历当前多项式 p 的系数列表
        for i in range(len(p)):
            # 如果当前系数 p[-i-1] 为 0，则跳过当前循环
            if not p[-i - 1]:
                continue
            
            # 计算当前系数的下标和 g 的最大公约数
            g = igcd(g, i)
            
            # 如果 g 等于 1，返回 1 和原始多项式集合 polys
            if g == 1:
                return 1, polys
        
        # 计算整体最大公约数 G 和当前多项式 p 的最大公约数 g 的最大公约数
        G = igcd(G, g)
    
    # 对多项式集合 polys 中的每个多项式 p，对其进行 G 阶压缩
    return G, tuple([ p[::G] for p in polys ])
    # 如果列表 B 中所有元素都等于 1，则直接返回 B 和 polys
    if all(b == 1 for b in B):
        return B, polys
    
    # 创建空列表 H，用于存储多项式转换后的结果
    H = []
    
    # 遍历 F 中的每一个字典 f
    for f in F:
        # 创建空字典 h，用于存储转换后的多项式系数
        h = {}
        
        # 遍历 f 中的每一项 (A, coeff)，其中 A 是一个列表，coeff 是系数
        for A, coeff in f.items():
            # 计算列表 N，其中每个元素是 A 中对应位置上元素与 B 中对应位置上元素的整数除法结果
            N = [ a // b for a, b in zip(A, B) ]
            # 将转换后的系数 coeff 存储在 h 字典中，键为元组 tuple(N)
            h[tuple(N)] = coeff
        
        # 将转换后的多项式系数字典 h 转换成多项式对象，并添加到列表 H 中
        H.append(dmp_from_dict(h, u, K))
    
    # 返回结果列表 B 和转换后的多项式列表 H
    return B, tuple(H)
# 将多项式 ``f`` 中的每个系数 ``x`` 映射为 ``x**m``，``m`` 为正整数，在域 ``K[x]`` 中操作
def dup_inflate(f, m, K):
    # 若 ``m`` 小于等于 0，则抛出异常
    if m <= 0:
        raise IndexError("'m' must be positive, got %s" % m)
    # 若 ``m`` 等于 1 或者 ``f`` 为空，则直接返回 ``f``
    if m == 1 or not f:
        return f

    # 初始化结果列表，并将 ``f`` 的第一个系数加入其中
    result = [f[0]]

    # 遍历 ``f`` 的每个系数（除第一个外）
    for coeff in f[1:]:
        # 在结果列表中扩展 m-1 个零系数
        result.extend([K.zero]*(m - 1))
        # 将当前系数添加到结果列表中
        result.append(coeff)

    return result


# 递归辅助函数，用于 ``dmp_inflate``
def _rec_inflate(g, M, v, i, K):
    """Recursive helper for :func:`dmp_inflate`."""
    # 若 v 为假值（0），则调用 ``dup_inflate`` 处理 ``g``，使用 M[i] 和域 ``K``
    if not v:
        return dup_inflate(g, M[i], K)
    # 若 M[i] 小于等于 0，则抛出异常
    if M[i] <= 0:
        raise IndexError("all M[i] must be positive, got %s" % M[i])

    # 更新递归参数 w 和 j
    w, j = v - 1, i + 1

    # 递归地对 ``g`` 中的每个系数调用当前函数
    g = [ _rec_inflate(c, M, w, j, K) for c in g ]

    # 初始化结果列表，并将 ``g`` 的第一个系数加入其中
    result = [g[0]]

    # 遍历 ``g`` 的每个系数（除第一个外）
    for coeff in g[1:]:
        # 将 M[i]-1 个零系数插入到结果列表中
        for _ in range(1, M[i]):
            result.append(dmp_zero(w))

        # 将当前系数添加到结果列表中
        result.append(coeff)

    return result


# 主函数，将多项式 ``f`` 中的每个系数映射为 ``x_i**k_i``，在多项式环 ``K[X]`` 中操作
def dmp_inflate(f, M, u, K):
    # 若 u 为假值（0），则调用 ``dup_inflate`` 处理 ``f``，使用 M[0] 和域 ``K``
    if not u:
        return dup_inflate(f, M[0], K)

    # 若所有 M 中的元素均为 1，则直接返回 ``f``
    if all(m == 1 for m in M):
        return f
    else:
        # 否则调用递归辅助函数 ``_rec_inflate`` 处理 ``f``
        return _rec_inflate(f, M, u, 0, K)


# 排除多项式 ``f`` 中无用的级别
def dmp_exclude(f, u, K):
    # 若 u 为假值（0）或者多项式 ``f`` 是常数，则返回空列表、``f`` 和 ``u``
    if not u or dmp_ground_p(f, None, u):
        return [], f, u

    # 初始化空列表 J 和多项式字典 F
    J, F = [], dmp_to_dict(f, u)

    # 遍历级别范围 [0, u+1)
    for j in range(0, u + 1):
        # 遍历多项式字典中的每个单项式及其系数
        for monom in F.keys():
            # 如果单项式 monom 的第 j 个元素不为零，则跳出循环
            if monom[j]:
                break
        else:
            # 如果所有单项式 monom 的第 j 个元素均为零，则将 j 加入到列表 J 中
            J.append(j)

    # 若列表 J 为空，则返回空列表、``f`` 和 ``u``
    if not J:
        return [], f, u

    # 初始化空字典 f 和多项式字典 F
    f = {}

    # 遍历多项式字典 F 中的每个单项式及其系数
    for monom, coeff in F.items():
        # 将单项式 monom 转换为列表
        monom = list(monom)

        # 倒序遍历列表 J
        for j in reversed(J):
            # 删除列表 monom 中的第 j 个元素
            del monom[j]

        # 将修改后的单项式 monom 和对应的系数 coeff 添加到字典 f 中
        f[tuple(monom)] = coeff

    # 更新级别 u 为原级别 u 减去列表 J 的长度
    u -= len(J)

    return J, dmp_from_dict(f, u, K), u


# 在多项式 ``f`` 中包含无用级别
def dmp_include(f, J, u, K):
    # 若列表 J 为空，则直接返回多项式 ``f``
    if not J:
        return f

    # 将多项式 ``f`` 转换为多项式字典 F，并初始化空字典 f
    F, f = dmp_to_dict(f, u), {}

    # 遍历多项式字典 F 中的每个单项式及其系数
    for monom, coeff in F.items():
        # 将单项式 monom 转换为列表
        monom = list(monom)

        # 遍历列表 J
        for j in J:
            # 在列表 monom 中的第 j 个位置插入零
            monom.insert(j, 0)

        # 将修改后的单项式 monom 和对应的系数 coeff 添加到字典 f 中
        f[tuple(monom)] = coeff

    # 更新级别 u 为原级别 u 加上列表 J 的长度
    u += len(J)

    return dmp_from_dict(f, u, K)
    # 调用名为 dmp_from_dict 的函数，并返回其结果
    return dmp_from_dict(f, u, K)
# 将多项式 f 从环 K[X][Y] 转换为环 K[X,Y] 的函数
def dmp_inject(f, u, K, front=False):
    f, h = dmp_to_dict(f, u), {}  # 将多项式 f 转换为字典表示，并初始化空字典 h

    v = K.ngens - 1  # 计算环 K 的生成元数减一

    for f_monom, g in f.items():  # 遍历多项式字典 f 的每一项
        g = g.to_dict()  # 将多项式系数 g 转换为字典形式

        for g_monom, c in g.items():  # 遍历 g 的每一项
            if front:
                h[g_monom + f_monom] = c  # 若 front 为 True，将 g_monom 和 f_monom 相加作为新的键，值为 c
            else:
                h[f_monom + g_monom] = c  # 否则将 f_monom 和 g_monom 相加作为新的键，值为 c

    w = u + v + 1  # 计算新的多项式字典 h 的最高次数

    return dmp_from_dict(h, w, K.dom), w  # 根据字典 h、次数 w 和域 K.dom 返回转换后的多项式及次数


# 将多项式 f 从环 K[X,Y] 转换为环 K[X][Y] 的函数
def dmp_eject(f, u, K, front=False):
    f, h = dmp_to_dict(f, u), {}  # 将多项式 f 转换为字典表示，并初始化空字典 h

    n = K.ngens  # 计算环 K 的生成元数
    v = u - K.ngens + 1  # 计算新的字典 h 的最高次数

    for monom, c in f.items():  # 遍历多项式字典 f 的每一项
        if front:
            g_monom, f_monom = monom[:n], monom[n:]  # 若 front 为 True，分割 monom 为 g_monom 和 f_monom
        else:
            g_monom, f_monom = monom[-n:], monom[:-n]  # 否则分割 monom 为 f_monom 和 g_monom

        if f_monom in h:
            h[f_monom][g_monom] = c  # 若 f_monom 已在 h 中，则将 g_monom 和 c 添加到 h[f_monom] 中
        else:
            h[f_monom] = {g_monom: c}  # 否则创建新的键值对 {g_monom: c}

    for monom, c in h.items():
        h[monom] = K(c)  # 将字典 h 中的每个值转换为域 K 中的元素

    return dmp_from_dict(h, v - 1, K)  # 根据字典 h、次数 v-1 和域 K 返回转换后的多项式


# 从多项式 f 中移除在环 K[x] 中的项的最大公因子
def dup_terms_gcd(f, K):
    if dup_TC(f, K) or not f:
        return 0, f  # 如果多项式 f 是零或在环 K[x] 中没有常数项，则返回 (0, f)

    i = 0

    for c in reversed(f):
        if not c:
            i += 1
        else:
            break

    return i, f[:-i]  # 返回最大公因子的次数 i 和移除后的多项式 f


# 从多项式 f 中移除在环 K[X] 中的项的最大公因子
def dmp_terms_gcd(f, u, K):
    if dmp_ground_TC(f, u, K) or dmp_zero_p(f, u):
        return (0,) * (u + 1), f  # 如果多项式 f 是零或在环 K[X] 中没有项，则返回对应元组和 f

    F = dmp_to_dict(f, u)  # 将多项式 f 转换为字典 F
    G = monomial_min(*list(F.keys()))  # 找出字典 F 的所有键的最小公倍数作为 G

    if all(g == 0 for g in G):
        return G, f  # 如果 G 中所有元素均为零，则返回 G 和 f

    f = {}

    for monom, coeff in F.items():  # 遍历多项式字典 F 的每一项
        f[monomial_div(monom, G)] = coeff  # 将 monom 除以 G 得到新的字典 f

    return G, dmp_from_dict(f, u, K)  # 根据字典 f、次数 u 和域 K 返回转换后的多项式


# 递归辅助函数，用于 :func:`dmp_list_terms`
def _rec_list_terms(g, v, monom):
    d, terms = dmp_degree(g, v), []  # 计算多项式 g 的最高次数，并初始化空列表 terms

    if not v:
        for i, c in enumerate(g):
            if not c:
                continue

            terms.append((monom + (d - i,), c))  # 将 (monom + (d - i,), c) 添加到 terms 中
    else:
        # 如果条件不满足，则将变量 w 设置为 v 减 1
        w = v - 1

        # 遍历列表 g 中的元素，同时获取索引 i 和元素值 c
        for i, c in enumerate(g):
            # 递归调用 _rec_list_terms 函数，并将结果扩展到列表 terms 中
            # 参数分别为 c（当前元素），w（更新后的变量 w），monom 增加了一个元组 (d - i,)
            terms.extend(_rec_list_terms(c, w, monom + (d - i,)))

    # 返回处理后的 terms 列表作为函数的结果
    return terms
# 在多项式字典 ``f`` 中，列出所有非零项，按照给定的顺序 ``order``。
def dmp_list_terms(f, u, K, order=None):
    # 定义一个排序函数，根据顺序 ``O`` 对项进行排序
    def sort(terms, O):
        return sorted(terms, key=lambda term: O(term[0]), reverse=True)

    # 递归地列出多项式 ``f`` 的所有项
    terms = _rec_list_terms(f, u, ())

    # 如果没有项，则返回包含一个零项的列表
    if not terms:
        return [((0,)*(u + 1), K.zero)]

    # 如果未指定顺序，则直接返回项的列表
    if order is None:
        return terms
    else:
        # 否则，按照指定的顺序 ``order`` 对项进行排序
        return sort(terms, monomial_key(order))


# 应用函数 ``h`` 到多项式 ``f`` 和 ``g`` 的系数对上。
def dup_apply_pairs(f, g, h, args, K):
    # 获取多项式 ``f`` 和 ``g`` 的长度
    n, m = len(f), len(g)

    # 如果两个多项式长度不同，则补齐短的多项式以使它们相同长度
    if n != m:
        if n > m:
            g = [K.zero]*(n - m) + g
        else:
            f = [K.zero]*(m - n) + f

    # 初始化结果列表
    result = []

    # 对多项式 ``f`` 和 ``g`` 的对应系数应用函数 ``h``，并将结果添加到结果列表中
    for a, b in zip(f, g):
        result.append(h(a, b, *args))

    # 返回处理后的结果列表
    return dup_strip(result)


# 应用函数 ``h`` 到多项式 ``f`` 和 ``g`` 的系数对上，其中 ``f`` 和 ``g`` 是多项式列表。
def dmp_apply_pairs(f, g, h, args, u, K):
    # 如果 ``u`` 为零，则调用 ``dup_apply_pairs`` 处理
    if not u:
        return dup_apply_pairs(f, g, h, args, K)

    # 获取多项式 ``f`` 和 ``g`` 的长度以及次数 ``v``
    n, m, v = len(f), len(g), u - 1

    # 如果两个多项式长度不同，则补齐短的多项式以使它们相同长度
    if n != m:
        if n > m:
            g = dmp_zeros(n - m, v, K) + g
        else:
            f = dmp_zeros(m - n, v, K) + f

    # 初始化结果列表
    result = []

    # 对多项式 ``f`` 和 ``g`` 的对应系数应用函数 ``h``，并将结果添加到结果列表中
    for a, b in zip(f, g):
        result.append(dmp_apply_pairs(a, b, h, args, v, K))

    # 返回处理后的结果列表
    return dmp_strip(result, u)


# 在多项式 ``f`` 中，取连续的项的子序列，``f`` 是 ``K[x]`` 中的多项式。
def dup_slice(f, m, n, K):
    # 获取多项式 ``f`` 的长度
    k = len(f)

    # 计算截取的起始索引 M 和结束索引 N
    if k >= m:
        M = k - m
    else:
        M = 0
    if k >= n:
        N = k - n
    else:
        N = 0

    # 对多项式 ``f`` 进行切片操作
    f = f[N:M]

    # 去除切片后多项式开头的零系数
    while f and f[0] == K.zero:
        f.pop(0)

    # 如果切片后为空，则返回空列表；否则，添加零系数以使长度为 ``m`` 并返回结果
    if not f:
        return []
    else:
        return f + [K.zero]*m


# 在多项式 ``f`` 中，取连续的项的子序列，``f`` 是 ``K[X]`` 中的多项式。
def dmp_slice(f, m, n, u, K):
    # 调用 ``dmp_slice_in`` 函数进行处理
    return dmp_slice_in(f, m, n, 0, u, K)


# 在多项式 ``f`` 中，取 ``x_j`` 上连续的项的子序列，``f`` 是 ``K[X]`` 中的多项式。
def dmp_slice_in(f, m, n, j, u, K):
    # 如果索引 ``j`` 超出范围，则抛出索引错误
    if j < 0 or j > u:
        raise IndexError("-%s <= j < %s expected, got %s" % (u, u, j))

    # 如果 ``u`` 为零，则调用 ``dup_slice`` 函数处理
    if not u:
        return dup_slice(f, m, n, K)

    # 否则，将多项式 ``f`` 转换为字典形式，并初始化结果字典 ``g``
    f, g = dmp_to_dict(f, u), {}
    # 对字典 f 中的每一个项进行迭代，其中 monom 是键，coeff 是对应的值
    for monom, coeff in f.items():
        # 获取 monom 中索引为 j 的元素 k
        k = monom[j]

        # 如果 k 小于 m 或者大于等于 n，则将 monom 中索引为 j 的位置替换为 0
        if k < m or k >= n:
            monom = monom[:j] + (0,) + monom[j + 1:]

        # 如果 monom 已经在字典 g 中，则将 coeff 累加到 g[monom] 上
        if monom in g:
            g[monom] += coeff
        else:
            # 如果 monom 不在字典 g 中，则将 monom 添加到 g 中，并将 coeff 作为其值
            g[monom] = coeff

    # 调用函数 dmp_from_dict，将字典 g 转换为对应的对象，并返回结果
    return dmp_from_dict(g, u, K)
# 定义一个函数 dup_random，用于生成指定次数和系数范围的随机多项式。
def dup_random(n, a, b, K):
    """
    Return a polynomial of degree ``n`` with coefficients in ``[a, b]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_random

    >>> dup_random(3, -10, 10, ZZ) #doctest: +SKIP
    [-2, -8, 9, -4]

    """
    # 生成一个包含 n+1 个元素的列表，每个元素为 K.convert(random.randint(a, b)) 的结果，表示随机生成的多项式的系数。
    f = [ K.convert(random.randint(a, b)) for _ in range(0, n + 1) ]

    # 如果生成的多项式的最高次项系数为零，则重新生成，直到不为零。
    while not f[0]:
        f[0] = K.convert(random.randint(a, b))

    # 返回生成的随机多项式的系数列表。
    return f
```