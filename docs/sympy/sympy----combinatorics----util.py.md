# `D:\src\scipysrc\sympy\sympy\combinatorics\util.py`

```
# 导入必要的函数和类
from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime

# 获取 Permutation 类的 rmul 方法
rmul = Permutation.rmul
# 获取 Permutation 类的 _af_new 方法
_af_new = Permutation._af_new

############################################
#
# Utilities for computational group theory
#
############################################

# 定义一个函数用于对底部点集排序，使得 base 点首先且按顺序排列
def _base_ordering(base, degree):
    r"""
    Order `\{0, 1, \dots, n-1\}` so that base points come first and in order.

    Parameters
    ==========

    base : the base
    degree : the degree of the associated permutation group

    Returns
    =======

    A list ``base_ordering`` such that ``base_ordering[point]`` is the
    number of ``point`` in the ordering.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _base_ordering
    >>> S = SymmetricGroup(4)
    >>> S.schreier_sims()
    >>> _base_ordering(S.base, S.degree)
    [0, 1, 2, 3]

    Notes
    =====

    This is used in backtrack searches, when we define a relation `\ll` on
    the underlying set for a permutation group of degree `n`,
    `\{0, 1, \dots, n-1\}`, so that if `(b_1, b_2, \dots, b_k)` is a base we
    have `b_i \ll b_j` whenever `i<j` and `b_i \ll a` for all
    `i\in\{1,2, \dots, k\}` and `a` is not in the base. The idea is developed
    and applied to backtracking algorithms in [1], pp.108-132. The points
    that are not in the base are taken in increasing order.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    # 初始化一个顺序列表
    ordering = [0]*degree
    # 将 base 中的点按顺序映射到 ordering 中
    for i in range(len(base)):
        ordering[base[i]] = i
    # 处理不在 base 中的点，按顺序添加到 ordering 中
    current = len(base)
    for i in range(degree):
        if i not in base:
            ordering[i] = current
            current += 1
    return ordering


# 检查是否存在长度为 p 的素数周期的循环
def _check_cycles_alt_sym(perm):
    """
    Checks for cycles of prime length p with n/2 < p < n-2.

    Explanation
    ===========

    Here `n` is the degree of the permutation. This is a helper function for
    the function is_alt_sym from sympy.combinatorics.perm_groups.

    Examples
    ========

    >>> from sympy.combinatorics.util import _check_cycles_alt_sym
    >>> from sympy.combinatorics import Permutation
    >>> a = Permutation([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    >>> _check_cycles_alt_sym(a)
    False
    >>> b = Permutation([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]])
    >>> _check_cycles_alt_sym(b)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.is_alt_sym

    """
    n = perm.size  # 获取置换的阶数
    af = perm.array_form  # 获取置换的数组表示
    current_len = 0  # 当前循环的长度
    total_len = 0  # 总循环长度
    used = set()  # 已使用的索引集合
    # 遍历范围为 n 的一半，向下取整
    for i in range(n//2):
        # 检查 i 是否未被使用且小于 n 的一半减去已使用的总长度
        if i not in used and i < n//2 - total_len:
            # 初始化当前长度为 1
            current_len = 1
            # 将 i 添加到已使用集合中
            used.add(i)
            # 设定 j 的初始值为 i
            j = i
            # 循环直到找到循环链的起点 i
            while af[j] != i:
                # 增加当前长度
                current_len += 1
                # 更新 j 为下一个节点
                j = af[j]
                # 将 j 添加到已使用集合中
                used.add(j)
            # 增加总长度
            total_len += current_len
            # 如果当前长度大于 n 的一半且小于 n-2，并且是素数，则返回 True
            if current_len > n//2 and current_len < n - 2 and isprime(current_len):
                return True
    # 如果没有找到满足条件的循环链，返回 False
    return False
def _handle_precomputed_bsgs(base, strong_gens, transversals=None,
                             basic_orbits=None, strong_gens_distr=None):
    """
    Calculate BSGS-related structures from those present.

    Explanation
    ===========

    The base and strong generating set must be provided; if any of the
    transversals, basic orbits or distributed strong generators are not
    provided, they will be calculated from the base and strong generating set.

    Parameters
    ==========

    base : the base
        A sequence of points that define the base for the group action.
    strong_gens : the strong generators
        A list of elements that generate a permutation group.
    transversals : basic transversals
        Optional. Precomputed basic transversals for the base.
    basic_orbits : basic orbits
        Optional. Precomputed basic orbits under the group action.
    strong_gens_distr : strong generators distributed by membership in basic stabilizers
        Optional. Precomputed strong generators distributed by membership in basic stabilizers.

    Returns
    =======

    (transversals, basic_orbits, strong_gens_distr)
        where *transversals* are the basic transversals, *basic_orbits* are the
        basic orbits, and *strong_gens_distr* are the strong generators distributed
        by membership in basic stabilizers.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> D = DihedralGroup(3)
    >>> D.schreier_sims()
    >>> D.strong_gens
    [(0 1 2), (0 2), (1 2)]
    >>> D.base
    [0, 1]
    >>> _handle_precomputed_bsgs(D.base, D.strong_gens)
    (None, None, [[(0 1 2), (0 2), (1 2)],
                  [(1 2)]])

    See Also
    ========

    _strong_gens_from_distr, _orbits_transversals_from_bsgs,
    _handle_precomputed_bsgs

    """
    base_len = len(base)  # Length of the base sequence
    degree = strong_gens[0].size  # Degree of the permutation group
    stabs = [[] for _ in range(base_len)]  # Initialize a list of empty lists for stabilizers
    max_stab_index = 0  # Initialize the maximum stabilizer index

    # Iterate over each generator in the strong generators list
    for gen in strong_gens:
        j = 0
        # Find the largest j such that gen fixes the first j elements of base
        while j < base_len - 1 and gen._array_form[base[j]] == base[j]:
            j += 1
        if j > max_stab_index:
            max_stab_index = j
        # Append gen to the lists corresponding to the first j+1 elements of base
        for k in range(j + 1):
            stabs[k].append(gen)

    # For elements where no generator fixes up to base_len elements, add the identity element
    for i in range(max_stab_index + 1, base_len):
        stabs[i].append(_af_new(list(range(degree))))

    # Return the list of stabilizers
    return stabs
    # 如果没有提供强生成分布，则使用基础和强生成元素来计算它们的分布
    if strong_gens_distr is None:
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    
    # 如果没有提供横截面，则根据基础和强生成分布计算基本轨道和横截面
    if transversals is None:
        if basic_orbits is None:
            # 从基础和强生成分布计算基本轨道和横截面
            basic_orbits, transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr)
        else:
            # 从基础和强生成分布计算横截面（仅限横截面）
            transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr,
                                               transversals_only=True)
    else:
        # 如果没有提供基本轨道，则初始化基本轨道列表并填充
        if basic_orbits is None:
            base_len = len(base)
            basic_orbits = [None]*base_len
            for i in range(base_len):
                basic_orbits[i] = list(transversals[i].keys())
    
    # 返回计算得到的结果：横截面、基本轨道和强生成分布
    return transversals, basic_orbits, strong_gens_distr
def _orbits_transversals_from_bsgs(base, strong_gens_distr,
                                   transversals_only=False, slp=False):
    """
    Compute basic orbits and transversals from a base and strong generating set.

    Explanation
    ===========

    The generators are provided as distributed across the basic stabilizers.
    If the optional argument ``transversals_only`` is set to True, only the
    transversals are returned.

    Parameters
    ==========

    base : The base.
        List representing the base of the permutation group.
    strong_gens_distr : Strong generators distributed by membership in basic stabilizers.
        List of lists, where each sublist contains generators associated with
        a specific stabilizer of the base.
    transversals_only : bool, default: False
        A flag switching between returning only the
        transversals and both orbits and transversals.
    slp : bool, default: False
        If ``True``, return a list of dictionaries containing the
        generator presentations of the elements of the transversals,
        i.e. the list of indices of generators from ``strong_gens_distr[i]``
        such that their product is the relevant transversal element.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> (S.base, strong_gens_distr)
    ([0, 1], [[(0 1 2), (2)(0 1), (1 2)], [(1 2)]])

    See Also
    ========

    _distribute_gens_by_base, _handle_precomputed_bsgs

    """
    from sympy.combinatorics.perm_groups import _orbit_transversal
    
    # Determine the length of the base and the degree of the group
    base_len = len(base)
    degree = strong_gens_distr[0][0].size
    
    # Initialize lists to store transversals and optionally SLPs
    transversals = [None]*base_len
    slps = [None]*base_len
    
    # Initialize basic_orbits list if transversals_only is False
    if transversals_only is False:
        basic_orbits = [None]*base_len
    
    # Iterate over each element in the base to compute its transversal
    for i in range(base_len):
        # Compute the orbit and transversal using _orbit_transversal function
        transversals[i], slps[i] = _orbit_transversal(degree, strong_gens_distr[i],
                                 base[i], pairs=True, slp=True)
        
        # Convert transversals[i] to a dictionary format
        transversals[i] = dict(transversals[i])
        
        # If transversals_only is False, compute and store basic orbits
        if transversals_only is False:
            basic_orbits[i] = list(transversals[i].keys())
    
    # Return transversals if transversals_only is True, otherwise return basic_orbits and transversals
    if transversals_only:
        return transversals
    else:
        if not slp:
            return basic_orbits, transversals
        return basic_orbits, transversals, slps


def _remove_gens(base, strong_gens, basic_orbits=None, strong_gens_distr=None):
    """
    Remove redundant generators from a strong generating set.

    Parameters
    ==========

    base : a base
        A list representing the base of the permutation group.
    strong_gens : a strong generating set relative to *base*
        List of generators forming a strong generating set relative to the base.
    basic_orbits : basic orbits
        Optional. List representing basic orbits associated with the base.
    strong_gens_distr : strong generators distributed by membership in basic stabilizers
        Optional. List of lists, where each sublist contains generators associated with
        a specific stabilizer of the base.

    Returns
    =======

    A strong generating set with respect to ``base`` which is a subset of
    ``strong_gens``.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _remove_gens
    >>> from sympy.combinatorics.testutil import _verify_bsgs

    """
    # 使用 SymmetricGroup 类创建一个阶为 15 的对称群对象 S
    S = SymmetricGroup(15)
    # 调用 S 对象的 schreier_sims_incremental 方法，获取基和强发生器
    base, strong_gens = S.schreier_sims_incremental()
    # 调用 _remove_gens 函数，移除基和强发生器中的一些生成元，返回新的生成元列表 new_gens
    new_gens = _remove_gens(base, strong_gens)
    # 计算新生成元列表 new_gens 的长度，并断言其结果为 14
    len(new_gens)
    # 调用 _verify_bsgs 函数，验证给定的基和新的生成元 new_gens 是否生成对称群 S
    _verify_bsgs(S, base, new_gens)
    # 返回验证结果 True

Notes
=====

This procedure is outlined in [1],p.95.

References
==========

.. [1] Holt, D., Eick, B., O'Brien, E.
       "Handbook of computational group theory"

"""
# 从 sympy.combinatorics.perm_groups 模块导入 _orbit 函数
from sympy.combinatorics.perm_groups import _orbit
# 计算基的长度
base_len = len(base)
# 获取第一个强发生器的阶数，即基的第一个元素的大小
degree = strong_gens[0].size
# 如果 strong_gens_distr 为 None，则调用 _distribute_gens_by_base 函数分配基和强发生器
if strong_gens_distr is None:
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
# 如果 basic_orbits 为 None，则初始化为空列表，并计算基轨道
if basic_orbits is None:
    basic_orbits = []
    for i in range(base_len):
        basic_orbit = _orbit(degree, strong_gens_distr[i], base[i])
        basic_orbits.append(basic_orbit)
# 将一个空列表附加到 strong_gens_distr 的末尾
strong_gens_distr.append([])
# 复制强发生器列表到结果 res
res = strong_gens[:]
# 从后向前遍历基列表
for i in range(base_len - 1, -1, -1):
    # 复制当前基的强发生器列表
    gens_copy = strong_gens_distr[i][:]
    # 遍历当前基的强发生器
    for gen in strong_gens_distr[i]:
        # 如果 gen 不在下一个基的强发生器列表中，则从 gens_copy 中移除 gen
        if gen not in strong_gens_distr[i + 1]:
            temp_gens = gens_copy[:]
            temp_gens.remove(gen)
            # 如果 temp_gens 为空列表，则继续下一次循环
            if temp_gens == []:
                continue
            # 计算使用 temp_gens 的轨道
            temp_orbit = _orbit(degree, temp_gens, base[i])
            # 如果 temp_orbit 与基轨道相等，则从 gens_copy 和 res 中移除 gen
            if temp_orbit == basic_orbits[i]:
                gens_copy.remove(gen)
                res.remove(gen)
# 返回结果列表 res
return res
def _strip(g, base, orbits, transversals):
    """
    Attempt to decompose a permutation using a (possibly partial) BSGS
    structure.

    Explanation
    ===========

    This is done by treating the sequence ``base`` as an actual base, and
    the orbits ``orbits`` and transversals ``transversals`` as basic orbits and
    transversals relative to it.

    This process is called "sifting". A sift is unsuccessful when a certain
    orbit element is not found or when after the sift the decomposition
    does not end with the identity element.

    The argument ``transversals`` is a list of dictionaries that provides
    transversal elements for the orbits ``orbits``.

    Parameters
    ==========

    g : permutation to be decomposed
    base : sequence of points
    orbits : list
        A list in which the ``i``-th entry is an orbit of ``base[i]``
        under some subgroup of the pointwise stabilizer of `
        `base[0], base[1], ..., base[i - 1]``. The groups themselves are implicit
        in this function since the only information we need is encoded in the orbits
        and transversals
    transversals : list
        A list of orbit transversals associated with the orbits *orbits*.

    Examples
    ========

    >>> from sympy.combinatorics import Permutation, SymmetricGroup
    >>> from sympy.combinatorics.util import _strip
    >>> S = SymmetricGroup(5)
    >>> S.schreier_sims()
    >>> g = Permutation([0, 2, 3, 1, 4])
    >>> _strip(g, S.base, S.basic_orbits, S.basic_transversals)
    ((4), 5)

    Notes
    =====

    The algorithm is described in [1],pp.89-90. The reason for returning
    both the current state of the element being decomposed and the level
    at which the sifting ends is that they provide important information for
    the randomized version of the Schreier-Sims algorithm.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E."Handbook of computational group theory"

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims
    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims_random

    """
    h = g._array_form  # Get the array form of the permutation
    base_len = len(base)  # Determine the length of the base sequence
    for i in range(base_len):  # Iterate over the indices of the base sequence
        beta = h[base[i]]  # Obtain the image of base[i] under permutation g
        if beta == base[i]:  # If beta is fixed by the permutation
            continue  # Skip to the next iteration
        if beta not in orbits[i]:  # If beta is not in the i-th orbit
            return _af_new(h), i + 1  # Return the new array form of h and i + 1
        u = transversals[i][beta]._array_form  # Get the array form of the transversal element
        h = _af_rmul(_af_invert(u), h)  # Update h by multiplying with the inverse of u in array form
    return _af_new(h), base_len + 1  # Return the final array form of h and base_len + 1


def _strip_af(h, base, orbits, transversals, j, slp=[], slps={}):
    """
    optimized _strip, with h, transversals and result in array form
    if the stripped elements is the identity, it returns False, base_len + 1

    j    h[base[i]] == base[i] for i <= j

    """
    base_len = len(base)  # Determine the length of the base sequence
    # 循环遍历从 j+1 到 base_len 的索引 i
    for i in range(j+1, base_len):
        # 获取 base[i] 对应的 h 中的值，并赋给 beta
        beta = h[base[i]]
        # 如果 beta 等于 base[i]，则继续下一次循环
        if beta == base[i]:
            continue
        # 如果 beta 不在 orbits[i] 中
        if beta not in orbits[i]:
            # 如果 slp 为空，则返回 h 和 i+1
            if not slp:
                return h, i + 1
            # 否则返回 h, i+1, slp
            return h, i + 1, slp
        # 获取 transversals[i][beta] 赋值给 u
        u = transversals[i][beta]
        # 如果 h 等于 u
        if h == u:
            # 如果 slp 为空，则返回 False 和 base_len + 1
            if not slp:
                return False, base_len + 1
            # 否则返回 False, base_len + 1, slp
            return False, base_len + 1, slp
        # 将 _af_invert(u) 与 h 作右乘，结果赋给 h
        h = _af_rmul(_af_invert(u), h)
        # 如果 slp 不为空
        if slp:
            # 将 slps[i][beta] 复制一份并逆序
            u_slp = slps[i][beta][:]
            u_slp.reverse()
            # 将逆序后的每个元素构造为 (i, (g,)) 形式，并加到 slp 前面
            u_slp = [(i, (g,)) for g in u_slp]
            slp = u_slp + slp
    # 如果 slp 为空，则返回 h 和 base_len + 1
    if not slp:
        return h, base_len + 1
    # 否则返回 h, base_len + 1, slp
    return h, base_len + 1, slp
# 从基本稳定子的生成器中提取强生成集

def _strong_gens_from_distr(strong_gens_distr):
    """
    Retrieve strong generating set from generators of basic stabilizers.

    This is just the union of the generators of the first and second basic
    stabilizers.

    Parameters
    ==========

    strong_gens_distr : list
        List of lists representing strong generators distributed by membership in basic stabilizers.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import (_strong_gens_from_distr,
    ... _distribute_gens_by_base)
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> S.strong_gens
    [(0 1 2), (2)(0 1), (1 2)]
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> _strong_gens_from_distr(strong_gens_distr)
    [(0 1 2), (2)(0 1), (1 2)]

    See Also
    ========

    _distribute_gens_by_base

    """
    # 如果只有一个基本稳定子的生成器集合，则直接返回该生成器集合的副本
    if len(strong_gens_distr) == 1:
        return strong_gens_distr[0][:]
    else:
        result = strong_gens_distr[0]
        # 遍历第二个基本稳定子的生成器集合，将不在结果集中的生成器添加进去
        for gen in strong_gens_distr[1]:
            if gen not in result:
                result.append(gen)
        return result
```