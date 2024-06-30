# `D:\src\scipysrc\sympy\sympy\combinatorics\galois.py`

```
"""
Construct transitive subgroups of symmetric groups, useful in Galois theory.

Besides constructing instances of the :py:class:`~.PermutationGroup` class to
represent the transitive subgroups of $S_n$ for small $n$, this module provides
*names* for these groups.

In some applications, it may be preferable to know the name of a group,
rather than receive an instance of the :py:class:`~.PermutationGroup`
class, and then have to do extra work to determine which group it is, by
checking various properties.

Names are instances of ``Enum`` classes defined in this module. With a name in
hand, the name's ``get_perm_group`` method can then be used to retrieve a
:py:class:`~.PermutationGroup`.

The names used for groups in this module are taken from [1].

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""

from collections import defaultdict
from enum import Enum
import itertools

from sympy.combinatorics.named_groups import (
    SymmetricGroup, AlternatingGroup, CyclicGroup, DihedralGroup,
    set_symmetric_group_properties, set_alternating_group_properties,
)
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation


class S1TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S1.
    """
    S1 = "S1"

    def get_perm_group(self):
        return SymmetricGroup(1)


class S2TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S2.
    """
    S2 = "S2"

    def get_perm_group(self):
        return SymmetricGroup(2)


class S3TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S3.
    """
    A3 = "A3"
    S3 = "S3"

    def get_perm_group(self):
        if self == S3TransitiveSubgroups.A3:
            return AlternatingGroup(3)
        elif self == S3TransitiveSubgroups.S3:
            return SymmetricGroup(3)


class S4TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S4.
    """
    C4 = "C4"
    V = "V"
    D4 = "D4"
    A4 = "A4"
    S4 = "S4"

    def get_perm_group(self):
        if self == S4TransitiveSubgroups.C4:
            return CyclicGroup(4)
        elif self == S4TransitiveSubgroups.V:
            return four_group()  # Assuming four_group is a defined function elsewhere
        elif self == S4TransitiveSubgroups.D4:
            return DihedralGroup(4)
        elif self == S4TransitiveSubgroups.A4:
            return AlternatingGroup(4)
        elif self == S4TransitiveSubgroups.S4:
            return SymmetricGroup(4)


class S5TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S5.
    """
    C5 = "C5"
    D5 = "D5"
    M20 = "M20"
    A5 = "A5"
    S5 = "S5"
    # 定义一个方法用于获取置换群
    def get_perm_group(self):
        # 如果 self 等于 S5TransitiveSubgroups.C5，则返回一个阶为 5 的循环群对象
        if self == S5TransitiveSubgroups.C5:
            return CyclicGroup(5)
        # 如果 self 等于 S5TransitiveSubgroups.D5，则返回一个阶为 5 的二面角群对象
        elif self == S5TransitiveSubgroups.D5:
            return DihedralGroup(5)
        # 如果 self 等于 S5TransitiveSubgroups.M20，则返回一个 M20 对象
        elif self == S5TransitiveSubgroups.M20:
            return M20()
        # 如果 self 等于 S5TransitiveSubgroups.A5，则返回一个阶为 5 的交错群对象
        elif self == S5TransitiveSubgroups.A5:
            return AlternatingGroup(5)
        # 如果 self 等于 S5TransitiveSubgroups.S5，则返回一个阶为 5 的对称群对象
        elif self == S5TransitiveSubgroups.S5:
            return SymmetricGroup(5)
class S6TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S6.
    """

    # Enum members representing different transitive subgroups of S6
    C6 = "C6"
    S3 = "S3"
    D6 = "D6"
    A4 = "A4"
    G18 = "G18"
    A4xC2 = "A4 x C2"
    S4m = "S4-"
    S4p = "S4+"
    G36m = "G36-"
    G36p = "G36+"
    S4xC2 = "S4 x C2"
    PSL2F5 = "PSL2(F5)"
    G72 = "G72"
    PGL2F5 = "PGL2(F5)"
    A6 = "A6"
    S6 = "S6"

    def get_perm_group(self):
        # Method to return the permutation group associated with each subgroup
        if self == S6TransitiveSubgroups.C6:
            return CyclicGroup(6)
        elif self == S6TransitiveSubgroups.S3:
            return S3_in_S6()
        elif self == S6TransitiveSubgroups.D6:
            return DihedralGroup(6)
        elif self == S6TransitiveSubgroups.A4:
            return A4_in_S6()
        elif self == S6TransitiveSubgroups.G18:
            return G18()
        elif self == S6TransitiveSubgroups.A4xC2:
            return A4xC2()
        elif self == S6TransitiveSubgroups.S4m:
            return S4m()
        elif self == S6TransitiveSubgroups.S4p:
            return S4p()
        elif self == S6TransitiveSubgroups.G36m:
            return G36m()
        elif self == S6TransitiveSubgroups.G36p:
            return G36p()
        elif self == S6TransitiveSubgroups.S4xC2:
            return S4xC2()
        elif self == S6TransitiveSubgroups.PSL2F5:
            return PSL2F5()
        elif self == S6TransitiveSubgroups.G72:
            return G72()
        elif self == S6TransitiveSubgroups.PGL2F5:
            return PGL2F5()
        elif self == S6TransitiveSubgroups.A6:
            return AlternatingGroup(6)
        elif self == S6TransitiveSubgroups.S6:
            return SymmetricGroup(6)


def four_group():
    """
    Return a representation of the Klein four-group as a transitive subgroup
    of S4.
    """
    return PermutationGroup(
        Permutation(0, 1)(2, 3),
        Permutation(0, 2)(1, 3)
    )


def M20():
    """
    Return a representation of the metacyclic group M20, a transitive subgroup
    of S5 that is one of the possible Galois groups for polys of degree 5.

    Notes
    =====

    See [1], Page 323.

    """
    # Create and define properties of the M20 group
    G = PermutationGroup(Permutation(0, 1, 2, 3, 4), Permutation(1, 2, 4, 3))
    G._degree = 5
    G._order = 20
    G._is_transitive = True
    G._is_sym = False
    G._is_alt = False
    G._is_cyclic = False
    G._is_dihedral = False
    return G


def S3_in_S6():
    """
    Return a representation of S3 as a transitive subgroup of S6.

    Notes
    =====

    The representation is found by viewing the group as the symmetries of a
    triangular prism.

    """
    # Create and return the S3 subgroup of S6
    G = PermutationGroup(Permutation(0, 1, 2)(3, 4, 5), Permutation(0, 3)(2, 4)(1, 5))
    set_symmetric_group_properties(G, 3, 6)
    return G


def A4_in_S6():
    """
    Return a representation of A4 as a transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    # Create and return the A4 subgroup of S6
    G = PermutationGroup(Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4))
    ```
    # 调用函数 set_alternating_group_properties 设置交替群 G 的性质，参数为 4 和 6
    set_alternating_group_properties(G, 4, 6)
    # 返回设置后的交替群 G
    return G
# 定义函数 S4m，返回 S6 中 S4- 的传递子群的表示
def S4m():
    # 创建包含两个置换的置换群 G：(1, 4, 5, 3) 和 (0, 4)(1, 5)(2, 3)
    G = PermutationGroup(Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3))
    # 设置对称群属性为 4 和 6
    set_symmetric_group_properties(G, 4, 6)
    return G


# 定义函数 S4p，返回 S6 中 S4+ 的传递子群的表示
def S4p():
    # 创建包含两个置换的置换群 G：(0, 2, 4, 1)(3, 5) 和 (0, 3)(4, 5)
    G = PermutationGroup(Permutation(0, 2, 4, 1)(3, 5), Permutation(0, 3)(4, 5))
    # 设置对称群属性为 4 和 6
    set_symmetric_group_properties(G, 4, 6)
    return G


# 定义函数 A4xC2，返回 S6 中 (A4 x C2) 的传递子群的表示
def A4xC2():
    # 创建包含三个置换的置换群，表示 (0, 4, 5)(1, 3, 2), (0, 1, 2)(3, 5, 4), (5)(2, 4)
    return PermutationGroup(
        Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4),
        Permutation(5)(2, 4))


# 定义函数 S4xC2，返回 S6 中 (S4 x C2) 的传递子群的表示
def S4xC2():
    # 创建包含三个置换的置换群，表示 (1, 4, 5, 3), (0, 4)(1, 5)(2, 3), (1, 4)(3, 5)
    return PermutationGroup(
        Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3),
        Permutation(1, 4)(3, 5))


# 定义函数 G18，返回 S6 中 G18 的传递子群的表示
def G18():
    # 创建包含三个置换的置换群，表示 (5)(0, 1, 2), (3, 4, 5), (0, 4)(1, 5)(2, 3)
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(0, 4)(1, 5)(2, 3))


# 定义函数 G36m，返回 S6 中 G36- 的传递子群的表示
def G36m():
    # 创建包含四个置换的置换群，表示 (5)(0, 1, 2), (3, 4, 5), (1, 2)(3, 5), (0, 4)(1, 5)(2, 3)
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(1, 2)(3, 5), Permutation(0, 4)(1, 5)(2, 3))


# 定义函数 G36p，返回 S6 中 G36+ 的传递子群的表示
def G36p():
    # 创建包含四个置换的置换群，表示 (5)(0, 1, 2), (3, 4, 5), (0, 5, 2, 3)(1, 4)
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(0, 5, 2, 3)(1, 4))


# 定义函数 G72，返回 S6 中 G72 的传递子群的表示
def G72():
    # 创建包含三个置换的置换群，表示 (5)(0, 1, 2), (0, 4, 1, 3)(2, 5), (0, 3)(1, 4)(2, 5)
    return PermutationGroup(
        Permutation(5)(0, 1, 2),
        Permutation(0, 4, 1, 3)(2, 5), Permutation(0, 3)(1, 4)(2, 5))


# 定义函数 PSL2F5，返回 S6 中 PSL2(F5) 的传递子群的表示，等同于 A5
def PSL2F5():
    # 这个函数返回一个具体的置换群，但是具体置换的定义在此处并未提供
    # 可能的实现需要参考额外的文档或代码
    pass
    # 这个函数使用了 :py:func:`~.find_transitive_subgroups_of_S6` 计算得出的结果。
    
    G = PermutationGroup(
        # 创建置换群 G，使用两个置换：(0, 4, 5)(1, 3, 2) 和 (0, 4, 3, 1, 5)
        Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 4, 3, 1, 5)
    )
    # 设置置换群 G 的性质，第一个参数是阶数为 5，第二个参数是阶数为 6
    set_alternating_group_properties(G, 5, 6)
    # 返回置换群 G
    return G
# 定义函数 PGL2F5，返回群 $PGL_2(\mathbb{F}_5)$ 在 S6 中的表示，同构于 $S_5$
def PGL2F5():
    # 创建置换群对象 G，包含两个置换：(0, 1, 2, 3, 4) 和 (0, 5)(1, 2)(3, 4)
    G = PermutationGroup(
        Permutation(0, 1, 2, 3, 4), Permutation(0, 5)(1, 2)(3, 4))
    # 设置对称群的特性，群阶为 5，置换集的阶为 6
    set_symmetric_group_properties(G, 5, 6)
    # 返回创建的置换群对象 G
    return G


# 定义函数 find_transitive_subgroups_of_S6，搜索 S6 中的特定传递子群
def find_transitive_subgroups_of_S6(*targets, print_report=False):
    r"""
    搜索 $S_6$ 中的特定传递子群。

    对称群 $S_6$ 有 16 个不同的传递子群，经过共轭。一些比其他更容易构造。例如，对角群 $D_6$
    可以立即找到，但如何在 $S_6$ 中传递实现 $S_4$ 或 $S_5$ 则不那么明显。

    在某些情况下，存在可以使用的知名构造。例如，$S_5$ 同构于 $PGL_2(\mathbb{F}_5)$，
    自然地作用于项目线 $P^1(\mathbb{F}_5)$，一个阶为 6 的集合。

    然而，如果没有这样的特殊构造，我们可以简单地搜索生成器。例如，可以通过此方法在 $S_6$
    中找到 $A_4$ 和 $S_4$ 的传递实例。

    一旦我们参与这样的搜索，即使像 $S_5$ 这样确实有特殊构造的群也可以更容易（虽然不那么优雅）地找到。

    此函数在 $S_6$ 中定位以下子群的生成器：

    * $A_4$
    * $S_4^-$（$S_4$ 不包含在 $A_6$ 中）
    * $S_4^+$（$S_4$ 包含在 $A_6$ 中）
    * $A_4 \times C_2$
    * $S_4 \times C_2$
    * $G_{18}   = C_3^2 \rtimes C_2$
    * $G_{36}^- = C_3^2 \rtimes C_2^2$
    * $G_{36}^+ = C_3^2 \rtimes C_4$
    * $G_{72}   = C_3^2 \rtimes D_4$
    * $A_5$
    * $S_5$

    注意：这些群中的每一个在此模块中也有一个专用函数，立即返回群，使用此搜索过程找到的生成器。

    由于排列群元素的生成具有随机性，搜索过程可以再次调用，以（可能）获取相同群的不同生成器。

    Parameters
    ==========

    targets : list of :py:class:`~.S6TransitiveSubgroups` values
        要查找的群。

    print_report : bool（默认为 False）
        如果为 True，则将找到的每个群的生成器打印到 stdout。

    Returns
    =======

    dict
        将 *targets* 中的每个名称映射到找到的 :py:class:`~.PermutationGroup`

    References
    ==========

    .. [2] https://en.wikipedia.org/wiki/Projective_linear_group#Exceptional_isomorphisms
    .. [3] https://en.wikipedia.org/wiki/Automorphisms_of_the_symmetric_and_alternating_groups#PGL%282,5%29
    """
    def elts_by_order(G):
        """根据元素的阶排序一个群的元素。"""
        elts = defaultdict(list)
        for g in G.elements:
            # 将元素按其阶数分组存储
            elts[g.order()].append(g)
        return elts

    def order_profile(G, name=None):
        """确定群中每个阶数的元素数量。"""
        elts = elts_by_order(G)
        # 创建一个字典，记录每个阶数的元素数量
        profile = {o:len(e) for o, e in elts.items()}
        if name:
            # 如果指定了名称，则打印群的阶数分布信息
            print(f'{name}: ' + ' '.join(f'{len(profile[r])}@{r}' for r in sorted(profile.keys())))
        return profile

    S6 = SymmetricGroup(6)
    A6 = AlternatingGroup(6)
    S6_by_order = elts_by_order(S6)

    def search(existing_gens, needed_gen_orders, order, alt=None, profile=None, anti_profile=None):
        """
        查找 S6 的一个传递子群。

        Parameters
        ==========

        existing_gens : list of Permutation
            可选的生成元列表，必须包含在所找到的群中。

        needed_gen_orders : list of positive int
            非空的生成元阶数列表，需要找到这些阶数的额外生成元。

        order: int
            要查找的群的阶数。

        alt: bool, None
            如果为 True，则要求找到的群必须包含在 A6 中。
            如果为 False，则要求找到的群不能包含在 A6 中。

        profile : dict
            如果提供了，则要求找到的群的阶数分布必须等于此分布。

        anti_profile : dict
            如果提供了，则要求找到的群的阶数分布不能等于此分布。
        """
        for gens in itertools.product(*[S6_by_order[n] for n in needed_gen_orders]):
            if len(set(gens)) < len(gens):
                continue
            G = PermutationGroup(existing_gens + list(gens))
            if G.order() == order and G.is_transitive():
                if alt is not None and G.is_subgroup(A6) != alt:
                    continue
                if profile and order_profile(G) != profile:
                    continue
                if anti_profile and order_profile(G) == anti_profile:
                    continue
                return G

    def match_known_group(G, alt=None):
        needed = [g.order() for g in G.generators]
        return search([], needed, G.order(), alt=alt, profile=order_profile(G))

    found = {}

    def finish_up(name, G):
        found[name] = G
        if print_report:
            print("=" * 40)
            print(f"{name}:")
            print(G.generators)

    if S6TransitiveSubgroups.A4 in targets or S6TransitiveSubgroups.A4xC2 in targets:
        A4_in_S6 = match_known_group(AlternatingGroup(4))
        finish_up(S6TransitiveSubgroups.A4, A4_in_S6)

    if S6TransitiveSubgroups.S4m in targets or S6TransitiveSubgroups.S4xC2 in targets:
        S4m_in_S6 = match_known_group(SymmetricGroup(4), alt=False)
        finish_up(S6TransitiveSubgroups.S4m, S4m_in_S6)
    # 如果 S4p 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.S4p in targets:
        # 在对称群 S4 中搜索已知的匹配群，并设置 alt=True
        S4p_in_S6 = match_known_group(SymmetricGroup(4), alt=True)
        # 完成处理 S4p 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.S4p, S4p_in_S6)

    # 如果 A4xC2 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.A4xC2 in targets:
        # 在 A4 在 S6 中的生成器中搜索特定条件下的匹配
        A4xC2_in_S6 = search(A4_in_S6.generators, [2], 24, anti_profile=order_profile(SymmetricGroup(4)))
        # 完成处理 A4xC2 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.A4xC2, A4xC2_in_S6)

    # 如果 S4xC2 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.S4xC2 in targets:
        # 在 S4m 在 S6 中的生成器中搜索特定条件下的匹配
        S4xC2_in_S6 = search(S4m_in_S6.generators, [2], 48)
        # 完成处理 S4xC2 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.S4xC2, S4xC2_in_S6)

    # 对于在任何 G_n 子群中正规因子 N = C3^2 的明显实例，我们取一个显而易见的实例在 S6 中：
    N_gens = [Permutation(5)(0, 1, 2), Permutation(5)(3, 4, 5)]

    # 如果 G18 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.G18 in targets:
        # 在 N_gens 中搜索特定条件下的 G18 匹配
        G18_in_S6 = search(N_gens, [2], 18)
        # 完成处理 G18 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.G18, G18_in_S6)

    # 如果 G36m 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.G36m in targets:
        # 在 N_gens 中搜索特定条件下的 G36m 匹配，设置 alt=False
        G36m_in_S6 = search(N_gens, [2, 2], 36, alt=False)
        # 完成处理 G36m 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.G36m, G36m_in_S6)

    # 如果 G36p 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.G36p in targets:
        # 在 N_gens 中搜索特定条件下的 G36p 匹配，设置 alt=True
        G36p_in_S6 = search(N_gens, [4], 36, alt=True)
        # 完成处理 G36p 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.G36p, G36p_in_S6)

    # 如果 G72 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.G72 in targets:
        # 在 N_gens 中搜索特定条件下的 G72 匹配
        G72_in_S6 = search(N_gens, [4, 2], 72)
        # 完成处理 G72 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.G72, G72_in_S6)

    # PSL2(F5) 和 PGL2(F5) 子群分别同构于 A5 和 S5。
    
    # 如果 PSL2F5 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.PSL2F5 in targets:
        # 在交错群 A5 中搜索已知的匹配群
        PSL2F5_in_S6 = match_known_group(AlternatingGroup(5))
        # 完成处理 PSL2F5 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.PSL2F5, PSL2F5_in_S6)

    # 如果 PGL2F5 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.PGL2F5 in targets:
        # 在对称群 S5 中搜索已知的匹配群
        PGL2F5_in_S6 = match_known_group(SymmetricGroup(5))
        # 完成处理 PGL2F5 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.PGL2F5, PGL2F5_in_S6)

    # 对于 C6、S3、D6、A6 或 S6，由于它们在 S6 中有明显的实现，很少需要“搜索”。
    # 然而，我们在这里支持它们，以防需要随机表示。

    # 如果 C6 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.C6 in targets:
        # 在循环群 C6 中搜索已知的匹配群
        C6 = match_known_group(CyclicGroup(6))
        # 完成处理 C6 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.C6, C6)

    # 如果 S3 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.S3 in targets:
        # 在对称群 S3 中搜索已知的匹配群
        S3 = match_known_group(SymmetricGroup(3))
        # 完成处理 S3 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.S3, S3)

    # 如果 D6 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.D6 in targets:
        # 在二面体群 D6 中搜索已知的匹配群
        D6 = match_known_group(DihedralGroup(6))
        # 完成处理 D6 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.D6, D6)

    # 如果 A6 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.A6 in targets:
        # 在 A6 中搜索已知的匹配群
        A6 = match_known_group(A6)
        # 完成处理 A6 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.A6, A6)

    # 如果 S6 是目标列表中的一个子群，执行以下操作：
    if S6TransitiveSubgroups.S6 in targets:
        # 在 S6 中搜索已知的匹配群
        S6 = match_known_group(S6)
        # 完成处理 S6 子群在 S6 中的转换
        finish_up(S6TransitiveSubgroups.S6, S6)

    # 返回找到的结果
    return found
```