# `D:\src\scipysrc\sympy\sympy\combinatorics\named_groups.py`

```
# 从 sympy.combinatorics.group_constructs 模块导入 DirectProduct 类
# 从 sympy.combinatorics.perm_groups 模块导入 PermutationGroup 类
from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.perm_groups import PermutationGroup
# 从 sympy.combinatorics.permutations 模块导入 Permutation 类
from sympy.combinatorics.permutations import Permutation

# 从 Permutation 类中导入 _af_new 方法
_af_new = Permutation._af_new

# 定义 AbelianGroup 函数，接收多个循环群的阶数作为参数
def AbelianGroup(*cyclic_orders):
    """
    Returns the direct product of cyclic groups with the given orders.

    Explanation
    ===========

    According to the structure theorem for finite abelian groups ([1]),
    every finite abelian group can be written as the direct product of
    finitely many cyclic groups.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import AbelianGroup
    >>> AbelianGroup(3, 4)
    PermutationGroup([
            (6)(0 1 2),
            (3 4 5 6)])
    >>> _.is_group
    True

    See Also
    ========

    DirectProduct

    References
    ==========

    .. [1] https://groupprops.subwiki.org/wiki/Structure_theorem_for_finitely_generated_abelian_groups

    """
    # 初始化空列表存储循环群
    groups = []
    # 初始化 degree 和 order 变量
    degree = 0
    order = 1
    # 遍历每个循环群的阶数
    for size in cyclic_orders:
        # 更新 degree 和 order
        degree += size
        order *= size
        # 创建循环群对象并加入 groups 列表
        groups.append(CyclicGroup(size))
    # 创建直积群 G
    G = DirectProduct(*groups)
    # 设置 G 的属性为交换群
    G._is_abelian = True
    G._degree = degree
    G._order = order

    return G


# 定义 AlternatingGroup 函数，生成由 n 个元素的交错群作为置换群
def AlternatingGroup(n):
    """
    Generates the alternating group on ``n`` elements as a permutation group.

    Explanation
    ===========

    For ``n > 2``, the generators taken are ``(0 1 2), (0 1 2 ... n-1)`` for
    ``n`` odd
    and ``(0 1 2), (1 2 ... n-1)`` for ``n`` even (See [1], p.31, ex.6.9.).
    After the group is generated, some of its basic properties are set.
    The cases ``n = 1, 2`` are handled separately.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import AlternatingGroup
    >>> G = AlternatingGroup(4)
    >>> G.is_group
    True
    >>> a = list(G.generate_dimino())
    >>> len(a)
    12
    >>> all(perm.is_even for perm in a)
    True

    See Also
    ========

    SymmetricGroup, CyclicGroup, DihedralGroup

    References
    ==========

    .. [1] Armstrong, M. "Groups and Symmetry"

    """
    # 处理特殊情况：当 n 为 1 或 2 时，返回包含单个置换的置换群对象
    if n in (1, 2):
        return PermutationGroup([Permutation([0])])

    # 对于 n > 2 的情况，根据 n 的奇偶性选择生成元
    a = list(range(n))
    a[0], a[1], a[2] = a[1], a[2], a[0]
    gen1 = a
    if n % 2:
        a = list(range(1, n))
        a.append(0)
        gen2 = a
    else:
        a = list(range(2, n))
        a.append(1)
        a.insert(0, 0)
        gen2 = a
    gens = [gen1, gen2]
    # 如果 gen1 和 gen2 相等，则只保留一个生成元
    if gen1 == gen2:
        gens = gens[:1]
    # 使用生成元列表创建置换群对象 G
    G = PermutationGroup([_af_new(a) for a in gens], dups=False)

    # 设置交错群的属性
    set_alternating_group_properties(G, n, n)
    G._is_alt = True
    return G


# 定义 set_alternating_group_properties 函数，设置交错群的已知属性
def set_alternating_group_properties(G, n, degree):
    """Set known properties of an alternating group. """
    # 根据 n 的大小设置交错群的属性
    if n < 4:
        G._is_abelian = True
        G._is_nilpotent = True
    else:
        G._is_abelian = False
        G._is_nilpotent = False
    if n < 5:
        G._is_solvable = True
    # 如果条件不满足，将图的可解性标志设为 False
    else:
        G._is_solvable = False
    # 设置图的度属性为给定的度数
    G._degree = degree
    # 设置图的传递性属性为 True
    G._is_transitive = True
    # 设置图的二面角属性为 False
    G._is_dihedral = False
def CyclicGroup(n):
    """
    Generates the cyclic group of order ``n`` as a permutation group.

    Explanation
    ===========

    The generator taken is the ``n``-cycle ``(0 1 2 ... n-1)``
    (in cycle notation). After the group is generated, some of its basic
    properties are set.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import CyclicGroup
    >>> G = CyclicGroup(6)
    >>> G.is_group
    True
    >>> G.order()
    6
    >>> list(G.generate_schreier_sims(af=True))
    [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 0, 1],
    [3, 4, 5, 0, 1, 2], [4, 5, 0, 1, 2, 3], [5, 0, 1, 2, 3, 4]]

    See Also
    ========

    SymmetricGroup, DihedralGroup, AlternatingGroup

    """
    # 创建一个从 0 到 n-1 的列表，并将最后一个元素移到列表开头，形成一个 n-cycle
    a = list(range(1, n))
    a.append(0)
    # 根据生成的 n-cycle 创建置换对象
    gen = _af_new(a)
    # 使用生成的置换对象创建置换群
    G = PermutationGroup([gen])

    # 设置置换群的基本属性
    G._is_abelian = True
    G._is_nilpotent = True
    G._is_solvable = True
    G._degree = n
    G._is_transitive = True
    G._order = n
    # 如果 n 等于 2，则设置为二面体群
    G._is_dihedral = (n == 2)
    return G


def DihedralGroup(n):
    r"""
    Generates the dihedral group `D_n` as a permutation group.

    Explanation
    ===========

    The dihedral group `D_n` is the group of symmetries of the regular
    ``n``-gon. The generators taken are the ``n``-cycle ``a = (0 1 2 ... n-1)``
    (a rotation of the ``n``-gon) and ``b = (0 n-1)(1 n-2)...``
    (a reflection of the ``n``-gon) in cycle rotation. It is easy to see that
    these satisfy ``a**n = b**2 = 1`` and ``bab = ~a`` so they indeed generate
    `D_n` (See [1]). After the group is generated, some of its basic properties
    are set.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> G = DihedralGroup(5)
    >>> G.is_group
    True
    >>> a = list(G.generate_dimino())
    >>> [perm.cyclic_form for perm in a]
    [[], [[0, 1, 2, 3, 4]], [[0, 2, 4, 1, 3]],
    [[0, 3, 1, 4, 2]], [[0, 4, 3, 2, 1]], [[0, 4], [1, 3]],
    [[1, 4], [2, 3]], [[0, 1], [2, 4]], [[0, 2], [3, 4]],
    [[0, 3], [1, 2]]]

    See Also
    ========

    SymmetricGroup, CyclicGroup, AlternatingGroup

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dihedral_group

    """
    # 小型情况下的特殊处理
    if n == 1:
        # 返回包含逆序置换的置换群对象
        return PermutationGroup([Permutation([1, 0])])
    if n == 2:
        # 返回包含三个适当置换的置换群对象
        return PermutationGroup([Permutation([1, 0, 3, 2]),
               Permutation([2, 3, 0, 1]), Permutation([3, 2, 1, 0])])

    # 创建一个从 1 到 n-1 的列表，并将最后一个元素移到列表开头，形成一个 n-cycle
    a = list(range(1, n))
    a.append(0)
    # 根据生成的 n-cycle 创建第一个置换对象
    gen1 = _af_new(a)
    # 创建一个从 0 到 n-1 的列表，并将元素逆序，形成另一个 n-cycle
    a = list(range(n))
    a.reverse()
    # 根据生成的逆序 n-cycle 创建第二个置换对象
    gen2 = _af_new(a)
    # 使用生成的两个置换对象创建置换群
    G = PermutationGroup([gen1, gen2])
    # 如果 n 是 2 的幂，则设置群为幂零群
    if n & (n-1) == 0:
        G._is_nilpotent = True
    else:
        G._is_nilpotent = False
    # 设置置换群的基本属性
    G._is_dihedral = True
    G._is_abelian = False
    G._is_solvable = True
    G._degree = n
    G._is_transitive = True
    G._order = 2*n
    return G


def SymmetricGroup(n):
    """
    # 如果 n 等于 1，生成只包含一个单位置换 [0] 的置换群 G
    if n == 1:
        G = PermutationGroup([Permutation([0])])
    # 如果 n 等于 2，生成包含一个逆序对 [1, 0] 的置换群 G
    elif n == 2:
        G = PermutationGroup([Permutation([1, 0])])
    else:
        # 对于 n 大于 2 的情况，生成两个生成元 gen1 和 gen2
        a = list(range(1, n))  # 生成序列 [1, 2, ..., n-1]
        a.append(0)  # 在末尾添加元素 0，得到 [1, 2, ..., n-1, 0]
        gen1 = _af_new(a)  # 使用 _af_new 函数创建置换 gen1
        a = list(range(n))  # 生成序列 [0, 1, ..., n-1]
        a[0], a[1] = a[1], a[0]  # 交换序列的第一个和第二个元素，得到 [1, 0, 2, ..., n-1]
        gen2 = _af_new(a)  # 使用 _af_new 函数创建置换 gen2
        G = PermutationGroup([gen1, gen2])  # 使用 gen1 和 gen2 创建置换群 G
    
    # 设置置换群 G 的基本属性，如阶数和生成 Schreier Sims 表
    set_symmetric_group_properties(G, n, n)
    # 将 _is_sym 标记设置为 True，表示这是一个对称群
    G._is_sym = True
    # 返回生成的对称群 G
    return G
# 设置对称群的已知属性
def set_symmetric_group_properties(G, n, degree):
    """Set known properties of a symmetric group. """
    # 如果 n 小于 3，对称群是阿贝尔群和幂零群
    if n < 3:
        G._is_abelian = True
        G._is_nilpotent = True
    else:
        # 否则，对称群不是阿贝尔群和幂零群
        G._is_abelian = False
        G._is_nilpotent = False
    # 如果 n 小于 5，对称群是可解群
    if n < 5:
        G._is_solvable = True
    else:
        # 否则，对称群不是可解群
        G._is_solvable = False
    # 设置对称群的阶（度）
    G._degree = degree
    # 对称群是传递群
    G._is_transitive = True
    # 如果 n 是 2 或 3，则对称群是二面体群，参考 Landau 的函数和 Stirling 的近似
    G._is_dihedral = (n in [2, 3])  # cf Landau's func and Stirling's approx


def RubikGroup(n):
    """Return a group of Rubik's cube generators

    >>> from sympy.combinatorics.named_groups import RubikGroup
    >>> RubikGroup(2).is_group
    True
    """
    # 导入魔方生成器模块
    from sympy.combinatorics.generators import rubik
    # 如果 n 小于等于 1，抛出值错误异常
    if n <= 1:
        raise ValueError("Invalid cube. n has to be greater than 1")
    # 返回一个由 rubik(n) 生成器生成的排列群对象
    return PermutationGroup(rubik(n))
```