# `D:\src\scipysrc\sympy\sympy\combinatorics\fp_groups.py`

```
"""Finitely Presented Groups and its algorithms. """

# 导入所需的符号和函数
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
                                                free_group)
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
                                             coset_enumeration_r,
                                             coset_enumeration_c)
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute

from itertools import product

# 定义公共函数 fp_group，用于创建 FpGroup 实例并返回其生成器
@public
def fp_group(fr_grp, relators=()):
    _fp_group = FpGroup(fr_grp, relators)
    return (_fp_group,) + tuple(_fp_group._generators)

# 定义公共函数 xfp_group，用于创建 FpGroup 实例并返回实例及其生成器
@public
def xfp_group(fr_grp, relators=()):
    _fp_group = FpGroup(fr_grp, relators)
    return (_fp_group, _fp_group._generators)

# 定义公共函数 vfp_group，用于创建 FpGroup 实例并对符号进行污染
# 注意：该函数当前无法工作，因为 symbols 和 pollute 未定义，从未经过测试
@public
def vfp_group(fr_grpm, relators):
    _fp_group = FpGroup(symbols, relators)
    pollute([sym.name for sym in _fp_group.symbols], _fp_group.generators)
    return _fp_group

# 定义函数 _parse_relators，用于解析传入的关系列表
def _parse_relators(rels):
    """Parse the passed relators."""
    return rels


###############################################################################
#                           FINITELY PRESENTED GROUPS                         #
###############################################################################

# 定义类 FpGroup，继承自 DefaultPrinting
class FpGroup(DefaultPrinting):
    """
    The FpGroup would take a FreeGroup and a list/tuple of relators, the
    relators would be specified in such a way that each of them be equal to the
    identity of the provided free group.

    """
    is_group = True
    is_FpGroup = True
    is_PermutationGroup = False

    # 初始化方法，接受自由群和关系列表作为参数
    def __init__(self, fr_grp, relators):
        relators = _parse_relators(relators)
        self.free_group = fr_grp
        self.relators = relators
        self.generators = self._generators()  # 计算并存储自由群的生成器
        self.dtype = type("FpGroupElement", (FpGroupElement,), {"group": self})

        # CosetTable 实例，用于标识恒等子群
        self._coset_table = None
        # 标识恒等子群的 coset table 是否已标准化
        self._is_standardized = False

        self._order = None  # 群的阶
        self._center = None  # 群的中心

        self._rewriting_system = RewritingSystem(self)  # 创建重写系统对象
        self._perm_isomorphism = None  # 置换同构映射
        return

    # 获取自由群的生成器
    def _generators(self):
        return self.free_group.generators

    # 尝试使群的重写系统成为合流状态
    def make_confluent(self):
        '''
        Try to make the group's rewriting system confluent

        '''
        self._rewriting_system.make_confluent()
        return
    # 返回 `word` 在当前群组中的规约形式，根据群组的重写系统。如果重写系统是可交错的，则规约形式是该词在群组中的唯一正规形式。
    def reduce(self, word):
        return self._rewriting_system.reduce(word)

    # 在群组中比较 `word1` 和 `word2` 是否相等，使用群组的重写系统。如果系统是可交错的，则返回的答案必然正确。
    # 如果不是，某些情况下可能返回 `False`，即使 `word1 == word2`。
    def equals(self, word1, word2):
        if self.reduce(word1 * word2**-1) == self.identity:
            return True
        elif self._rewriting_system.is_confluent:
            return False
        return None

    # 返回当前群组的单位元素
    @property
    def identity(self):
        return self.free_group.identity

    # 检查元素 `g` 是否属于自由群 `self.free_group`
    def __contains__(self, g):
        return g in self.free_group

    # 使用Reidemeister-Schreier算法返回由 `gens` 生成的子群
    # homomorphism -- 当设置为True时，返回一个包含原始群中演示生成器映像的字典
    def subgroup(self, gens, C=None, homomorphism=False):
        if not all(isinstance(g, FreeGroupElement) for g in gens):
            raise ValueError("Generators must be `FreeGroupElement`s")
        if not all(g.group == self.free_group for g in gens):
            raise ValueError("Given generators are not members of the group")
        if homomorphism:
            # 获取Reidemeister演示生成器的展示，包括映射到原始群中的映像
            g, rels, _gens = reidemeister_presentation(self, gens, C=C, homomorphism=True)
        else:
            # 获取Reidemeister演示生成器的展示
            g, rels = reidemeister_presentation(self, gens, C=C)
        if g:
            # 构建FpGroup对象，使用演示生成器和关系
            g = FpGroup(g[0].group, rels)
        else:
            # 如果没有演示生成器，创建一个空的FpGroup对象
            g = FpGroup(free_group('')[0], [])
        if homomorphism:
            # 如果homomorphism为True，返回群组和映射到原始群中的生成器映射
            from sympy.combinatorics.homomorphisms import homomorphism
            return g, homomorphism(g, self, g.generators, _gens, check=False)
        # 否则，只返回群组对象
        return g
    def coset_enumeration(self, H, strategy="relator_based", max_cosets=None,
                                                        draft=None, incomplete=False):
        """
        使用 Todd-Coxeter 算法在当前对象 `self` 上运行，以 `H` 作为子群，使用 `strategy`
        参数作为策略，返回一个 `coset table` 实例。返回的 coset table 是压缩的但不是标准化的。

        在关键字参数 `draft` 中可以传递 `fp_grp` 的 `CosetTable` 实例，此时 coset 枚举将从
        该实例开始并尝试完成它。

        当 `incomplete` 为 `True` 且由于某些原因函数无法完成时，将返回部分完成的表格。
        """
        if not max_cosets:
            max_cosets = CosetTable.coset_table_max_limit
        if strategy == 'relator_based':
            C = coset_enumeration_r(self, H, max_cosets=max_cosets,
                                                    draft=draft, incomplete=incomplete)
        else:
            C = coset_enumeration_c(self, H, max_cosets=max_cosets,
                                                    draft=draft, incomplete=incomplete)
        if C.is_complete():
            C.compress()
        return C

    def standardize_coset_table(self):
        """
        标准化 coset table `self` 并将内部变量 `_is_standardized` 设置为 `True`。
        """
        self._coset_table.standardize()
        self._is_standardized = True

    def coset_table(self, H, strategy="relator_based", max_cosets=None,
                                                 draft=None, incomplete=False):
        """
        返回 `self` 在 `H` 中的数学 coset table。
        """
        if not H:
            if self._coset_table is not None:
                if not self._is_standardized:
                    self.standardize_coset_table()
            else:
                C = self.coset_enumeration([], strategy, max_cosets=max_cosets,
                                            draft=draft, incomplete=incomplete)
                self._coset_table = C
                self.standardize_coset_table()
            return self._coset_table.table
        else:
            C = self.coset_enumeration(H, strategy, max_cosets=max_cosets,
                                            draft=draft, incomplete=incomplete)
            C.standardize()
            return C.table
    # 返回有限表示群 ``self`` 的阶数。使用余陪枚举，其中单位群作为子群，即 ``H=[]``。
    def order(self, strategy="relator_based"):
        """
        Returns the order of the finitely presented group ``self``. It uses
        the coset enumeration with identity group as subgroup, i.e ``H=[]``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x, y**2])
        >>> f.order(strategy="coset_table_based")
        2

        """
        # 如果已经计算过阶数，则直接返回缓存的结果
        if self._order is not None:
            return self._order
        # 如果已经计算过余陪表，则返回余陪表的大小作为阶数
        if self._coset_table is not None:
            self._order = len(self._coset_table.table)
        # 如果没有定义关系，则使用自由群的阶数作为阶数
        elif len(self.relators) == 0:
            self._order = self.free_group.order()
        # 如果只有一个生成元，则计算所有关系的第一个分量的最大公约数的绝对值作为阶数
        elif len(self.generators) == 1:
            self._order = abs(gcd([r.array_form[0][1] for r in self.relators]))
        # 如果群是无限的，则将阶数设为正无穷
        elif self._is_infinite():
            self._order = S.Infinity
        else:
            # 计算有限指数子群的生成元和余陪表大小的乘积作为阶数
            gens, C = self._finite_index_subgroup()
            if C:
                ind = len(C.table)
                self._order = ind * self.subgroup(gens, C=C).order()
            else:
                # 否则，计算相对于空子群的指数
                self._order = self.index([])
        # 返回计算得到的阶数
        return self._order

    # 检测群是否是无限的。如果测试成功则返回 `True`，否则返回 `None`
    def _is_infinite(self):
        '''
        Test if the group is infinite. Return `True` if the test succeeds
        and `None` otherwise

        '''
        # 收集所有在关系中使用过的生成元
        used_gens = set()
        for r in self.relators:
            used_gens.update(r.contains_generators())
        # 如果不是所有生成元都在使用过的生成元集合中，则认为群是无限的
        if not set(self.generators) <= used_gens:
            return True
        # 阿贝尔化测试：检查阿贝尔化是否无限
        abelian_rels = []
        for rel in self.relators:
            abelian_rels.append([rel.exponent_sum(g) for g in self.generators])
        m = Matrix(Matrix(abelian_rels))
        # 检查阿贝尔化的不变因子是否包含零元素
        if 0 in invariant_factors(m):
            return True
        else:
            return None
    def _finite_index_subgroup(self, s=None):
        '''
        Find the elements of `self` that generate a finite index subgroup
        and, if found, return the list of elements and the coset table of `self` by
        the subgroup, otherwise return `(None, None)`

        '''
        # 获取最频繁生成元素
        gen = self.most_frequent_generator()
        # 生成关系列表
        rels = list(self.generators)
        rels.extend(self.relators)
        # 如果没有给定 s，则根据条件生成 s
        if not s:
            if len(self.generators) == 2:
                s = [gen] + [g for g in self.generators if g != gen]
            else:
                rand = self.free_group.identity
                i = 0
                # 选择一个随机元素 rand，直到不在关系列表中且不是单位元或逆元
                while ((rand in rels or rand**-1 in rels or rand.is_identity)
                        and i < 10):
                    rand = self.random()
                    i += 1
                s = [gen, rand] + [g for g in self.generators if g != gen]
        # 计算中点
        mid = (len(s) + 1) // 2
        half1 = s[:mid]  # 第一个子集
        half2 = s[mid:]  # 第二个子集
        draft1 = None
        draft2 = None
        m = 200
        C = None
        # 循环直到找到完整的陪集表或超过最大限制
        while not C and (m / 2 < CosetTable.coset_table_max_limit):
            m = min(m, CosetTable.coset_table_max_limit)
            # 枚举第一个子集的陪集
            draft1 = self.coset_enumeration(half1, max_cosets=m,
                                            draft=draft1, incomplete=True)
            if draft1.is_complete():
                C = draft1
                half = half1
            else:
                # 枚举第二个子集的陪集
                draft2 = self.coset_enumeration(half2, max_cosets=m,
                                                draft=draft2, incomplete=True)
                if draft2.is_complete():
                    C = draft2
                    half = half2
            if not C:
                m *= 2
        # 如果没有找到完整的陪集表，则返回 (None, None)
        if not C:
            return None, None
        # 压缩陪集表
        C.compress()
        # 返回找到的半群生成元和陪集表
        return half, C

    def most_frequent_generator(self):
        # 获取生成元素列表
        gens = self.generators
        # 获取关系列表
        rels = self.relators
        # 计算每个生成元素在关系列表中出现的频率
        freqs = [sum(r.generator_count(g) for r in rels) for g in gens]
        # 返回频率最高的生成元素
        return gens[freqs.index(max(freqs))]

    def random(self):
        import random
        r = self.free_group.identity
        # 随机选择若干次生成元素及其逆元，并组合它们
        for i in range(random.randint(2, 3)):
            r = r * random.choice(self.generators) ** random.choice([1, -1])
        return r

    def index(self, H, strategy="relator_based"):
        """
        Return the index of subgroup ``H`` in group ``self``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x**5, y**4, y*x*y**3*x**3])
        >>> f.index([x])
        4

        """
        # TODO: use |G:H| = |G|/|H| (currently H can't be made into a group)
        # when we know |G| and |H|

        # 如果 H 是空列表，则返回群的阶
        if H == []:
            return self.order()
        else:
            # 构建并返回由 H 生成的陪集表的长度，作为 H 在 self 中的指数
            C = self.coset_enumeration(H, strategy)
            return len(C.table)
    # 定义对象的字符串表示方法
    def __str__(self):
        # 如果自由群的秩大于30
        if self.free_group.rank > 30:
            # 使用较简短的形式表示自由生成元素的群
            str_form = "<fp group with %s generators>" % self.free_group.rank
        else:
            # 否则使用详细的生成元素列表形式表示自由群
            str_form = "<fp group on the generators %s>" % str(self.generators)
        # 返回对象的字符串形式
        return str_form

    # 将对象的 __repr__ 方法设置为与 __str__ 方法相同
    __repr__ = __str__
#==============================================================================
#                       PERMUTATION GROUP METHODS
#==============================================================================

    def _to_perm_group(self):
        '''
        Return an isomorphic permutation group and the isomorphism.
        The implementation is dependent on coset enumeration so
        will only terminate for finite groups.

        '''
        # 导入需要的符号计算库中的排列和同态映射函数
        from sympy.combinatorics import Permutation
        from sympy.combinatorics.homomorphisms import homomorphism
        # 如果群的阶是无穷的，抛出未实现错误
        if self.order() is S.Infinity:
            raise NotImplementedError("Permutation presentation of infinite "
                                      "groups is not implemented")
        # 如果已经计算过同构映射，则直接使用已存的映射和其像
        if self._perm_isomorphism:
            T = self._perm_isomorphism
            P = T.image()
        else:
            # 获取余类表
            C = self.coset_table([])
            # 获取群的生成元
            gens = self.generators
            # 获取生成元在余类表中的映射
            images = [[C[i][2*gens.index(g)] for i in range(len(C))] for g in gens]
            # 转换映射为排列对象
            images = [Permutation(i) for i in images]
            # 创建排列群对象
            P = PermutationGroup(images)
            # 创建同态映射
            T = homomorphism(self, P, gens, images, check=False)
            self._perm_isomorphism = T
        return P, T

    def _perm_group_list(self, method_name, *args):
        '''
        Given the name of a `PermutationGroup` method (returning a subgroup
        or a list of subgroups) and (optionally) additional arguments it takes,
        return a list or a list of lists containing the generators of this (or
        these) subgroups in terms of the generators of `self`.

        '''
        # 获取同构的排列群和同态映射
        P, T = self._to_perm_group()
        # 调用指定方法名的排列群方法，返回子群或子群列表
        perm_result = getattr(P, method_name)(*args)
        single = False
        # 如果返回的是单个排列群对象，则转为列表形式
        if isinstance(perm_result, PermutationGroup):
            perm_result, single = [perm_result], True
        result = []
        # 对每个子群对象，通过同态映射逆转生成元的映射关系
        for group in perm_result:
            gens = group.generators
            result.append(T.invert(gens))
        return result[0] if single else result

    def derived_series(self):
        '''
        Return the list of lists containing the generators
        of the subgroups in the derived series of `self`.

        '''
        return self._perm_group_list('derived_series')

    def lower_central_series(self):
        '''
        Return the list of lists containing the generators
        of the subgroups in the lower central series of `self`.

        '''
        return self._perm_group_list('lower_central_series')

    def center(self):
        '''
        Return the list of generators of the center of `self`.

        '''
        return self._perm_group_list('center')

    def derived_subgroup(self):
        '''
        Return the list of generators of the derived subgroup of `self`.

        '''
        return self._perm_group_list('derived_subgroup')
    def centralizer(self, other):
        '''
        Return the list of generators of the centralizer of `other`
        (a list of elements of `self`) in `self`.
        '''
        # 转换为置换群并获取其转换器
        T = self._to_perm_group()[1]
        # 将参数 `other` 转换为置换群 `T` 中的元素
        other = T(other)
        # 调用 `_perm_group_list` 方法，返回 `self` 中与 `other` 的中心化子的生成器列表
        return self._perm_group_list('centralizer', other)

    def normal_closure(self, other):
        '''
        Return the list of generators of the normal closure of `other`
        (a list of elements of `self`) in `self`.
        '''
        # 转换为置换群并获取其转换器
        T = self._to_perm_group()[1]
        # 将参数 `other` 转换为置换群 `T` 中的元素
        other = T(other)
        # 调用 `_perm_group_list` 方法，返回 `self` 中与 `other` 的正规闭包的生成器列表
        return self._perm_group_list('normal_closure', other)

    def _perm_property(self, attr):
        '''
        Given an attribute of a `PermutationGroup`, return
        its value for a permutation group isomorphic to `self`.
        '''
        # 转换为置换群并获取其转换器
        P = self._to_perm_group()[0]
        # 返回与 `self` 同构的置换群 `P` 的属性 `attr` 的值
        return getattr(P, attr)

    @property
    def is_abelian(self):
        '''
        Check if `self` is abelian.
        '''
        # 调用 `_perm_property` 方法，检查 `self` 是否为阿贝尔群
        return self._perm_property("is_abelian")

    @property
    def is_nilpotent(self):
        '''
        Check if `self` is nilpotent.
        '''
        # 调用 `_perm_property` 方法，检查 `self` 是否为幂零群
        return self._perm_property("is_nilpotent")

    @property
    def is_solvable(self):
        '''
        Check if `self` is solvable.
        '''
        # 调用 `_perm_property` 方法，检查 `self` 是否为可解群
        return self._perm_property("is_solvable")

    @property
    def elements(self):
        '''
        List the elements of `self`.
        '''
        # 转换为置换群并获取其置换器
        P, T = self._to_perm_group()
        # 返回 `self` 的元素列表，通过置换器 `T` 反转置换群 `P` 的元素
        return T.invert(P.elements)

    @property
    def is_cyclic(self):
        """
        Return ``True`` if group is Cyclic.
        """
        # 如果生成器数量小于等于1，则返回 True
        if len(self.generators) <= 1:
            return True
        try:
            # 转换为置换群并获取其置换器
            P, T = self._to_perm_group()
        except NotImplementedError:
            # 如果无法实现无限循环群检查，则引发异常
            raise NotImplementedError("Check for infinite Cyclic group "
                                      "is not implemented")
        # 返回置换群 `P` 是否为循环群的结果
        return P.is_cyclic

    def abelian_invariants(self):
        """
        Return Abelian Invariants of a group.
        """
        try:
            # 转换为置换群并获取其置换器
            P, T = self._to_perm_group()
        except NotImplementedError:
            # 如果无法实现无限群的阿贝尔不变量，则引发异常
            raise NotImplementedError("abelian invariants is not implemented"
                                      "for infinite group")
        # 返回置换群 `P` 的阿贝尔不变量
        return P.abelian_invariants()

    def composition_series(self):
        """
        Return subnormal series of maximum length for a group.
        """
        try:
            # 转换为置换群并获取其置换器
            P, T = self._to_perm_group()
        except NotImplementedError:
            # 如果无法实现无限群的组成序列，则引发异常
            raise NotImplementedError("composition series is not implemented"
                                      "for infinite group")
        # 返回置换群 `P` 的最大长度子正规系列
        return P.composition_series()
class FpSubgroup(DefaultPrinting):
    '''
    The class implementing a subgroup of an FpGroup or a FreeGroup
    (only finite index subgroups are supported at this point). This
    is to be used if one wishes to check if an element of the original
    group belongs to the subgroup

    '''
    def __init__(self, G, gens, normal=False):
        super().__init__()
        # 设置父群对象
        self.parent = G
        # 确定子群的生成元列表，剔除恒等元素
        self.generators = list({g for g in gens if g != G.identity})
        # 用于 __contains__ 方法的最小词（未定义）
        self._min_words = None
        # 未定义的 C 值
        self.C = None
        # 是否为正规子群
        self.normal = normal

    def order(self):
        if not self.generators:
            return S.One
        if isinstance(self.parent, FreeGroup):
            return S.Infinity
        if self.C is None:
            # 计算余陪列表 C
            C = self.parent.coset_enumeration(self.generators)
            self.C = C
        # 返回群的阶，通过余陪表长度除以群的阶来估算
        return self.parent.order()/len(self.C.table)

    def to_FpGroup(self):
        if isinstance(self.parent, FreeGroup):
            # 如果父群是自由群，返回相应的自由群对象
            gen_syms = [('x_%d'%i) for i in range(len(self.generators))]
            return free_group(', '.join(gen_syms))[0]
        # 否则返回基于余陪表 C 的子群
        return self.parent.subgroup(C=self.C)

    def __str__(self):
        if len(self.generators) > 30:
            # 如果生成元超过 30 个，返回简略形式的字符串表示
            str_form = "<fp subgroup with %s generators>" % len(self.generators)
        else:
            # 否则返回具体的生成元信息的字符串表示
            str_form = "<fp subgroup on the generators %s>" % str(self.generators)
        return str_form

    __repr__ = __str__


###############################################################################
#                           LOW INDEX SUBGROUPS                               #
###############################################################################

def low_index_subgroups(G, N, Y=()):
    """
    Implements the Low Index Subgroups algorithm, i.e find all subgroups of
    ``G`` upto a given index ``N``. This implements the method described in
    [Sim94]. This procedure involves a backtrack search over incomplete Coset
    Tables, rather than over forced coincidences.

    Parameters
    ==========

    G: An FpGroup < X|R >
    N: positive integer, representing the maximum index value for subgroups
    Y: (an optional argument) specifying a list of subgroup generators, such
    that each of the resulting subgroup contains the subgroup generated by Y.

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, low_index_subgroups
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**4])
    >>> L = low_index_subgroups(f, 4)
    >>> for coset_table in L:
    ...     print(coset_table.table)
    [[0, 0, 0, 0]]
    [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 3, 3]]
    [[0, 0, 1, 2], [2, 2, 2, 0], [1, 1, 0, 1]]
    [[1, 1, 0, 0], [0, 0, 1, 1]]

    References
    ==========

    """
    """
    根据引用 [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"
           第 5.4 节
    
    根据引用 [2] Marston Conder 和 Peter Dobcsanyi
           "Applications and Adaptions of the Low Index Subgroups Procedure"
    """
    
    # 创建余类表对象 C，初始化为空列表
    C = CosetTable(G, [])
    # 获取群 G 的生成元（relators）
    R = G.relators
    # 设定短生成元的长度阈值
    len_short_rel = 5
    # 筛选出长度超过 len_short_rel 的生成元，放入集合 R2
    R2 = {rel for rel in R if len(rel) > len_short_rel}
    # 从 R 中减去 R2，计算其循环归约（identity_cyclic_reduction），并放入集合 R1
    R1 = {rel.identity_cyclic_reduction() for rel in set(R) - R2}
    # 计算 R1 的共轭类列表
    R1_c_list = C.conjugates(R1)
    # 初始化子群列表 S
    S = []
    # 调用 descendant_subgroups 函数，计算子群 S，传入参数为 S, C, R1_c_list, C.A[0], R2, N, Y
    descendant_subgroups(S, C, R1_c_list, C.A[0], R2, N, Y)
    # 返回计算得到的子群列表 S
    return S
# 定义函数 `descendant_subgroups`，接受多个参数 S, C, R1_c_list, x, R2, N, Y
def descendant_subgroups(S, C, R1_c_list, x, R2, N, Y):
    # 从对象 C 中获取 A_dict 字段，并赋值给 A_dict
    A_dict = C.A_dict
    # 从对象 C 中获取 A_dict_inv 字段，并赋值给 A_dict_inv
    A_dict_inv = C.A_dict_inv
    # 如果对象 C 是完全的
    if C.is_complete():
        # 遍历 R2 和 C 的 omega 中的每个元素
        for w, alpha in product(R2, C.omega):
            # 如果 C 的 scan_check 方法返回 False，则返回
            if not C.scan_check(alpha, w):
                return
        # 如果所有 R2 中的关系都被满足，则将 C 添加到 S 列表中
        S.append(C)
    else:
        # 在 Coset Table 中找到第一个未定义的条目
        for alpha, x in product(range(len(C.table)), C.A):
            if C.table[alpha][A_dict[x]] is None:
                # 将未定义的 coset 和生成器分别赋值给 undefined_coset 和 undefined_gen
                undefined_coset, undefined_gen = alpha, x
                break
        # 为填充未定义的条目，尝试所有可能的 beta 值
        reach = C.omega + [C.n]
        for beta in reach:
            if beta < N:
                # 如果 beta 等于 C.n 或者表中的某一项未定义
                if beta == C.n or C.table[beta][A_dict_inv[undefined_gen]] is None:
                    # 调用 try_descendant 函数进行后续处理
                    try_descendant(S, C, R1_c_list, R2, N, undefined_coset, \
                            undefined_gen, beta, Y)


# 定义函数 `try_descendant`，接受多个参数 S, C, R1_c_list, R2, N, alpha, x, beta, Y
def try_descendant(S, C, R1_c_list, R2, N, alpha, x, beta, Y):
    """
    尝试每个可能的 `\alpha^x` 的问题解决方法
    """
    # 创建对象 D 作为 C 的副本
    D = C.copy()
    # 如果 beta 等于 D.n 并且 beta 小于 N
    if beta == D.n and beta < N:
        # 在 D 的表中追加一行未定义项
        D.table.append([None]*len(D.A))
        # 将 beta 添加到 D.p 中
        D.p.append(beta)
    # 在 D 的表中设置 alpha^x 的值为 beta
    D.table[alpha][D.A_dict[x]] = beta
    # 在 D 的表中设置 beta^(x^-1) 的值为 alpha
    D.table[beta][D.A_dict_inv[x]] = alpha
    # 将 (alpha, x) 元组添加到 D 的 deduction_stack 中
    D.deduction_stack.append((alpha, x))
    # 如果不满足 R1_c_list[D.A_dict[x]] 和 R1_c_list[D.A_dict_inv[x]] 的处理规则
    if not D.process_deductions_check(R1_c_list[D.A_dict[x]], \
            R1_c_list[D.A_dict_inv[x]]):
        return
    # 遍历 Y 中的每个元素
    for w in Y:
        # 如果 D 的 scan_check 方法返回 False，则返回
        if not D.scan_check(0, w):
            return
    # 如果 D 是其类中的第一个
    if first_in_class(D, Y):
        # 递归调用 descendant_subgroups 函数处理 D
        descendant_subgroups(S, D, R1_c_list, x, R2, N, Y)


# 定义函数 `first_in_class`，接受一个参数 C 和一个可选参数 Y
def first_in_class(C, Y=()):
    """
    检查与 Coset Table 对应的子群 ``H=G1`` 是否可能是其共轭类的规范代表。

    """
    # 如果 first_in_class 返回 False，则放弃 C
    # 如果返回 True，则需要进一步处理与 C 对应的搜索树节点，因此递归调用 ``descendant_subgroups``
    # 在处理完 C 之后，返回 True 或者 False
    >>> C.p = [0, 1, 2]
    # 设置 C 类的属性 p 为列表 [0, 1, 2]

    >>> first_in_class(C)
    # 调用函数 first_in_class，检查是否存在类 C 的首次出现（首次出现返回 True）

    >>> C.table = [[1, 1, 1, 2], [0, 0, 2, 0], [2, None, 0, 1]]
    # 设置 C 类的属性 table 为二维列表 [[1, 1, 1, 2], [0, 0, 2, 0], [2, None, 0, 1]]

    >>> first_in_class(C)
    # 再次调用函数 first_in_class，检查是否存在类 C 的首次出现

    # TODO:: Sims points out in [Sim94] that performance can be improved by
    # remembering some of the information computed by ``first_in_class``. If
    # the ``continue alpha`` statement is executed at line 14, then the same thing
    # will happen for that value of alpha in any descendant of the table C, and so
    # the values the values of alpha for which this occurs could profitably be
    # stored and passed through to the descendants of C. Of course this would
    # make the code more complicated.
    # Sims 在 [Sim94] 中指出可以通过记住 ``first_in_class`` 计算的一些信息来提高性能。
    # 如果在第 14 行执行 ``continue alpha`` 语句，则对于表 C 的任何后代，alpha 的相同情况也将发生，
    # 因此可以将发生这种情况的 alpha 值存储起来，并传递给 C 的后代。当然，这会使代码变得更加复杂。

    # The code below is taken directly from the function on page 208 of [Sim94]
    # nu[alpha]

    """
    n = C.n
    # 获取 C 类的属性 n 的值，n 表示某个数量

    # lamda is the largest numbered point in Omega_c_alpha which is currently defined
    lamda = -1
    # lamda 是 Omega_c_alpha 中当前定义的最大编号点，初始值为 -1

    # for alpha in Omega_c, nu[alpha] is the point in Omega_c_alpha corresponding to alpha
    nu = [None]*n
    # 对于 Omega_c 中的每个 alpha，nu[alpha] 是与 alpha 对应的 Omega_c_alpha 中的点

    # for alpha in Omega_c_alpha, mu[alpha] is the point in Omega_c corresponding to alpha
    mu = [None]*n
    # 对于 Omega_c_alpha 中的每个 alpha，mu[alpha] 是与 alpha 对应的 Omega_c 中的点

    # mutually nu and mu are the mutually-inverse equivalence maps between
    # Omega_c_alpha and Omega_c
    # nu 和 mu 是 Omega_c_alpha 和 Omega_c 之间互逆的等价映射

    next_alpha = False
    # 初始化 next_alpha 为 False

    # For each 0!=alpha in [0 .. nc-1], we start by constructing the equivalent
    # standardized coset table C_alpha corresponding to H_alpha
    # 对于 [0 .. nc-1] 中的每个非零 alpha，我们首先构造与 H_alpha 对应的等价标准化陪集表 C_alpha
    # 对于每个 alpha 从 1 到 n-1 进行循环
    for alpha in range(1, n):
        # 在每次 alpha 变化后将 nu 重置为 "None"
        for beta in range(lamda+1):
            nu[mu[beta]] = None
        # 我们只希望在 G_alpha 包含给定子群时，拒绝当前表，并选择之前的表
        for w in Y:
            # TODO: 这应该支持一系列一般词语的输入，而不仅限于在 "A" 中的词语（即 gen 和 gen^-1）
            if C.table[alpha][C.A_dict[w]] != alpha:
                # 如果不相等，则继续处理下一个 alpha
                next_alpha = True
                break
        if next_alpha:
            next_alpha = False
            continue
        # 尝试将 alpha 作为 Omega_C_alpha 中的新点 0
        mu[0] = alpha
        nu[alpha] = 0
        # 比较 C 和 C_alpha 中对应的条目
        lamda = 0
        for beta in range(n):
            for x in C.A:
                gamma = C.table[beta][C.A_dict[x]]
                delta = C.table[mu[beta]][C.A_dict[x]]
                # 如果任何一个条目未定义，则继续处理下一个 alpha
                if gamma is None or delta is None:
                    # 继续处理下一个 alpha
                    next_alpha = True
                    break
                if nu[delta] is None:
                    # delta 成为 Omega_C_alpha 中的下一个点
                    lamda += 1
                    nu[delta] = lamda
                    mu[lamda] = delta
                if nu[delta] < gamma:
                    # 如果 nu[delta] 小于 gamma，则返回 False
                    return False
                if nu[delta] > gamma:
                    # 继续处理下一个 alpha
                    next_alpha = True
                    break
            if next_alpha:
                next_alpha = False
                break
    # 如果所有的 alpha 都通过了检查，则返回 True
    return True
#========================================================================
#                    Simplifying Presentation
#========================================================================

# 定义函数 `simplify_presentation`，用于简化 `FpGroup` 的表示
def simplify_presentation(*args, change_gens=False):
    '''
    For an instance of `FpGroup`, return a simplified isomorphic copy of
    the group (e.g. remove redundant generators or relators). Alternatively,
    a list of generators and relators can be passed in which case the
    simplified lists will be returned.

    By default, the generators of the group are unchanged. If you would
    like to remove redundant generators, set the keyword argument
    `change_gens = True`.

    '''
    # 处理单个参数的情况
    if len(args) == 1:
        # 检查参数是否为 `FpGroup` 的实例，否则引发类型错误
        if not isinstance(args[0], FpGroup):
            raise TypeError("The argument must be an instance of FpGroup")
        G = args[0]
        # 递归调用 `simplify_presentation` 处理生成元和关系
        gens, rels = simplify_presentation(G.generators, G.relators,
                                              change_gens=change_gens)
        # 如果存在简化后的生成元，则返回一个新的 `FpGroup` 实例
        if gens:
            return FpGroup(gens[0].group, rels)
        return FpGroup(FreeGroup([]), [])
    # 处理两个参数的情况
    elif len(args) == 2:
        gens, rels = args[0][:], args[1][:]
        # 如果生成元列表为空，直接返回生成元和关系列表
        if not gens:
            return gens, rels
        # 获取第一个生成元所属的群的单位元素
        identity = gens[0].group.identity
    # 处理其他参数数量的情况
    else:
        if len(args) == 0:
            m = "Not enough arguments"
        else:
            m = "Too many arguments"
        raise RuntimeError(m)

    # 初始化前一次迭代的生成元和关系列表
    prev_gens = []
    prev_rels = []
    # 迭代直到关系列表不再改变
    while not set(prev_rels) == set(rels):
        prev_rels = rels
        # 如果需要改变生成元且生成元列表不再改变，则进行生成元消除技术处理
        while change_gens and not set(prev_gens) == set(gens):
            prev_gens = gens
            # 调用 `elimination_technique_1` 进行生成元消除技术处理
            gens, rels = elimination_technique_1(gens, rels, identity)
        # 简化关系列表
        rels = _simplify_relators(rels)

    # 如果需要改变生成元，则重新生成群并替换关系中的符号
    if change_gens:
        syms = [g.array_form[0][0] for g in gens]
        F = free_group(syms)[0]
        identity = F.identity
        gens = F.generators
        subs = dict(zip(syms, gens))
        for j, r in enumerate(rels):
            a = r.array_form
            rel = identity
            # 根据替换关系中的符号重建关系
            for sym, p in a:
                rel = rel*subs[sym]**p
            rels[j] = rel
    # 返回简化后的生成元和关系列表
    return gens, rels

# 定义内部函数 `_simplify_relators`，用于简化一组关系列表
def _simplify_relators(rels):
    """
    Simplifies a set of relators. All relators are checked to see if they are
    of the form `gen^n`. If any such relators are found then all other relators
    are processed for strings in the `gen` known order.

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import _simplify_relators
    >>> F, x, y = free_group("x, y")
    >>> w1 = [x**2*y**4, x**3]
    >>> _simplify_relators(w1)
    [x**3, x**-1*y**4]

    >>> w2 = [x**2*y**-4*x**5, x**3, x**2*y**8, y**5]
    >>> _simplify_relators(w2)
    [x**-1*y**-2, x**-1*y*x**-1, x**3, y**5]

    >>> w3 = [x**6*y**4, x**4]
    >>> _simplify_relators(w3)
    [x**4, x**2*y**4]

    >>> w4 = [x**2, x**5, y**3]
    >>> _simplify_relators(w4)
    [x, y**3]

    """
    rels = rels[:]
    # 返回简化后的关系列表
    return rels
    # 如果 rels 列表为空，则返回空列表
    if not rels:
        return []

    # 获取第一个关系对象的身份标识
    identity = rels[0].group.identity

    # 构建字典，键为生成器 g，值为 g 的幂次方，其中 g^n 是关系的一部分
    exps = {}
    for i in range(len(rels)):
        rel = rels[i]
        # 如果关系只有一个音节
        if rel.number_syllables() == 1:
            g = rel[0]  # 获取生成器 g
            exp = abs(rel.array_form[0][1])  # 获取 g 的幂次方
            # 如果幂次方为负数，将关系和 g 都取反
            if rel.array_form[0][1] < 0:
                rels[i] = rels[i]**-1
                g = g**-1
            # 如果 g 在 exps 中已存在，则取其与当前幂次方的最大公约数
            if g in exps:
                exp = gcd(exp, exps[g].array_form[0][1])
            exps[g] = g**exp  # 将 g 的幂次方加入 exps 字典中

    # 将 exps 字典中的所有值（即单音节词）转为列表
    one_syllables_words = list(exps.values())

    # 使用单音节词减少关系中的一些幂次，使其更简洁
    for i, rel in enumerate(rels):
        if rel in one_syllables_words:
            continue
        rel = rel.eliminate_words(one_syllables_words, _all=True)
        # 对于包含的每个生成器 g，如果其在 exps 中存在，则根据条件减少幂次
        for g in rel.contains_generators():
            if g in exps:
                exp = exps[g].array_form[0][1]
                max_exp = (exp + 1) // 2
                rel = rel.eliminate_word(g**(max_exp), g**(max_exp - exp), _all=True)
                rel = rel.eliminate_word(g**(-max_exp), g**(-(max_exp - exp)), _all=True)
        rels[i] = rel

    # 对每个关系进行环归约操作
    rels = [r.identity_cyclic_reduction() for r in rels]

    # 将单音节词加入到关系列表中
    rels += one_syllables_words
    # 获取关系列表中的唯一值
    rels = list(set(rels))
    # 对关系列表进行排序
    rels.sort()

    # 尝试从关系列表中移除标识符 identity
    try:
        rels.remove(identity)
    except ValueError:
        pass
    # 返回处理后的关系列表
    return rels
# Pg 350, section 2.5.1 from [2]
def elimination_technique_1(gens, rels, identity):
    rels = rels[:]  # 复制一份关系列表以保持原始数据不变
    # 按照长度排序关系，以便优先处理较短的关系
    rels.sort()
    gens = gens[:]  # 复制一份生成器列表以保持原始数据不变
    redundant_gens = {}  # 存储多余的生成器及其消去后的表达式
    redundant_rels = []  # 存储已处理过的冗余关系
    used_gens = set()  # 存储已使用过的生成器集合

    # 遍历每个关系，寻找只出现一次的生成器
    for rel in rels:
        # 如果关系中包含已知的冗余生成器，则跳过
        contained_gens = rel.contains_generators()
        if any(g in contained_gens for g in redundant_gens):
            continue
        contained_gens = list(contained_gens)
        contained_gens.sort(reverse=True)

        for gen in contained_gens:
            # 如果该生成器只在关系中出现一次且未被使用过，则处理之
            if rel.generator_count(gen) == 1 and gen not in used_gens:
                k = rel.exponent_sum(gen)
                gen_index = rel.index(gen**k)
                bk = rel.subword(gen_index + 1, len(rel))
                fw = rel.subword(0, gen_index)
                chi = bk * fw
                redundant_gens[gen] = chi**(-1 * k)
                used_gens.update(chi.contains_generators())
                redundant_rels.append(rel)
                break

    # 移除已处理过的冗余关系
    rels = [r for r in rels if r not in redundant_rels]

    # 从剩余的关系中消去冗余的生成器，并进行循环归约
    rels = [r.eliminate_words(redundant_gens, _all=True).identity_cyclic_reduction() for r in rels]

    # 将关系列表转换为集合以去重，再转换回列表形式
    rels = list(set(rels))

    try:
        # 尝试移除恒等关系
        rels.remove(identity)
    except ValueError:
        pass

    # 从生成器列表中移除冗余的生成器
    gens = [g for g in gens if g not in redundant_gens]

    # 返回更新后的生成器列表和关系列表
    return gens, rels

###############################################################################
#                           SUBGROUP PRESENTATIONS                            #
###############################################################################

# Pg 175 [1]
def define_schreier_generators(C, homomorphism=False):
    '''
    Parameters
    ==========

    C -- Coset table.
    homomorphism -- When set to True, return a dictionary containing the images
                     of the presentation generators in the original group.
    '''
    y = []  # 初始化一个空列表
    gamma = 1  # 初始化一个整数变量
    f = C.fp_group  # 从输入的Coset表中获取自由群
    X = f.generators  # 获取自由群的生成器列表

    if homomorphism:
        # `_gens`存储与Schreier生成器对应的父群元素
        _gens = {}
        # 计算Schreier遍历
        tau = {}
        tau[0] = f.identity

    # 初始化一个二维列表，大小为(C.n) * len(C.A)，元素全部为None
    C.P = [[None]*len(C.A) for i in range(C.n)]
    # 遍历 C.omega 和 C.A 的笛卡尔积中的每个元素，alpha 和 x 是迭代变量
    for alpha, x in product(C.omega, C.A):
        # 从 C.table 中获取 beta，其值为 C.table[alpha][C.A_dict[x]]
        beta = C.table[alpha][C.A_dict[x]]
        # 如果 beta 等于 gamma，则执行以下操作
        if beta == gamma:
            # 将 C.P[alpha][C.A_dict[x]] 设置为 "<identity>"
            C.P[alpha][C.A_dict[x]] = "<identity>"
            # 将 C.P[beta][C.A_dict_inv[x]] 设置为 "<identity>"
            C.P[beta][C.A_dict_inv[x]] = "<identity>"
            # 增加 gamma 的值
            gamma += 1
            # 如果 homomorphism 为真，则执行以下操作
            if homomorphism:
                # 计算 tau[beta] = tau[alpha] * x
                tau[beta] = tau[alpha] * x
        # 如果 x 在 X 中并且 C.P[alpha][C.A_dict[x]] 是 None，则执行以下操作
        elif x in X and C.P[alpha][C.A_dict[x]] is None:
            # 构建 y_alpha_x，其值为 '%s_%s' % (x, alpha)
            y_alpha_x = '%s_%s' % (x, alpha)
            # 将 y_alpha_x 添加到 y 列表中
            y.append(y_alpha_x)
            # 将 C.P[alpha][C.A_dict[x]] 设置为 y_alpha_x
            C.P[alpha][C.A_dict[x]] = y_alpha_x
            # 如果 homomorphism 为真，则执行以下操作
            if homomorphism:
                # 计算 _gens[y_alpha_x] = tau[alpha] * x * tau[beta]**-1
                _gens[y_alpha_x] = tau[alpha] * x * tau[beta]**-1

    # 将以逗号分隔的 y 列表中的元素作为参数，创建自由群的生成器列表 grp_gens
    grp_gens = list(free_group(', '.join(y)))
    # 从 grp_gens 中移除第一个元素，并将其设置为 C._schreier_free_group
    C._schreier_free_group = grp_gens.pop(0)
    # 将 grp_gens 设置为 C._schreier_generators
    C._schreier_generators = grp_gens
    # 如果 homomorphism 为真，则将 _gens 设置为 C._schreier_gen_elem
    if homomorphism:
        C._schreier_gen_elem = _gens

    # 替换 P 中的所有元素为自由群元素
    for i, j in product(range(len(C.P)), range(len(C.A))):
        # 如果 C.P[i][j] 等于 "<identity>"，则将其替换为 C._schreier_free_group.identity
        if C.P[i][j] == "<identity>":
            C.P[i][j] = C._schreier_free_group.identity
        # 如果 C.P[i][j] 是字符串，则执行以下操作
        elif isinstance(C.P[i][j], str):
            # 获取 y 中 C.P[i][j] 的索引，并将其对应的生成器赋给 r
            r = C._schreier_generators[y.index(C.P[i][j])]
            # 将 C.P[i][j] 设置为 r
            C.P[i][j] = r
            # 获取 beta，其值为 C.table[i][j]
            beta = C.table[i][j]
            # 将 C.P[beta][j + 1] 设置为 r 的逆
            C.P[beta][j + 1] = r**-1
def reidemeister_relators(C):
    # 获取 C 对象的 fp_group 属性下的 relators
    R = C.fp_group.relators
    # 对每个 relator 和每个余类进行重写操作，生成 rels 列表
    rels = [rewrite(C, coset, word) for word in R for coset in range(C.n)]
    # 找出所有长度为 1 的生成器，存放在 order_1_gens 中
    order_1_gens = {i for i in rels if len(i) == 1}

    # 从 rels 中移除所有长度为 1 的生成器
    rels = list(filter(lambda rel: rel not in order_1_gens, rels))

    # 将 reidemeister relators 中的长度为 1 的生成器替换为单位元素
    for i in range(len(rels)):
        w = rels[i]
        w = w.eliminate_words(order_1_gens, _all=True)
        rels[i] = w

    # 从 C._schreier_generators 中移除所有 order 1 生成器及其逆元
    C._schreier_generators = [i for i in C._schreier_generators
                    if not (i in order_1_gens or i**-1 in order_1_gens)]

    # Tietze 变换 1，即 TT_1
    # 从 rels 中移除循环共轭元素
    i = 0
    while i < len(rels):
        w = rels[i]
        j = i + 1
        while j < len(rels):
            if w.is_cyclic_conjugate(rels[j]):
                del rels[j]
            else:
                j += 1
        i += 1

    # 将处理后的 rels 赋给 C._reidemeister_relators
    C._reidemeister_relators = rels


def rewrite(C, alpha, w):
    """
    Parameters
    ==========

    C: CosetTable
    alpha: A live coset
    w: A word in `A*`

    Returns
    =======

    rho(tau(alpha), w)

    Examples
    ========

    >>> from sympy.combinatorics.fp_groups import FpGroup, CosetTable, define_schreier_generators, rewrite
    >>> from sympy.combinatorics import free_group
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**6])
    >>> C = CosetTable(f, [])
    >>> C.table = [[1, 1, 2, 3], [0, 0, 4, 5], [4, 4, 3, 0], [5, 5, 0, 2], [2, 2, 5, 1], [3, 3, 1, 4]]
    >>> C.p = [0, 1, 2, 3, 4, 5]
    >>> define_schreier_generators(C)
    >>> rewrite(C, 0, (x*y)**6)
    x_4*y_2*x_3*x_1*x_2*y_4*x_5

    """
    v = C._schreier_free_group.identity
    # 依次计算词 w 中的每个生成器在 alpha 上的作用，并更新 alpha
    for i in range(len(w)):
        x_i = w[i]
        v = v*C.P[alpha][C.A_dict[x_i]]
        alpha = C.table[alpha][C.A_dict[x_i]]
    return v

# Pg 350, section 2.5.2 from [2]
def elimination_technique_2(C):
    """
    This technique eliminates one generator at a time. Heuristically this
    seems superior in that we may select for elimination the generator with
    shortest equivalent string at each stage.

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r, \
            reidemeister_relators, define_schreier_generators, elimination_technique_2
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**5, (x*y)**2]); H = [x*y, x**-1*y**-1*x*y*x]
    >>> C = coset_enumeration_r(f, H)
    >>> C.compress(); C.standardize()
    >>> define_schreier_generators(C)
    >>> reidemeister_relators(C)
    >>> elimination_technique_2(C)
    ([y_1, y_2], [y_2**-3, y_2*y_1*y_2*y_1*y_2*y_1, y_1**2])

    """
    # 获取 C 对象的 _reidemeister_relators
    rels = C._reidemeister_relators
    # 对 rels 进行降序排序
    rels.sort(reverse=True)
    # 获取 C 对象的 _schreier_generators
    gens = C._schreier_generators
    # 逆序遍历gens列表中的元素索引
    for i in range(len(gens) - 1, -1, -1):
        # 获取当前索引处的关系对象
        rel = rels[i]
        # 逆序遍历gens列表中的元素索引
        for j in range(len(gens) - 1, -1, -1):
            # 获取当前索引处的生成器对象
            gen = gens[j]
            # 如果关系对象中生成器的数量为1
            if rel.generator_count(gen) == 1:
                # 计算生成器的指数和
                k = rel.exponent_sum(gen)
                # 获取生成器的索引
                gen_index = rel.index(gen**k)
                # 获取从生成器索引处到关系对象末尾的子词
                bk = rel.subword(gen_index + 1, len(rel))
                # 获取从关系对象开头到生成器索引处的子词
                fw = rel.subword(0, gen_index)
                # 计算生成器的替换值
                rep_by = (bk * fw)**(-1*k)
                # 删除rels列表中的当前关系对象
                del rels[i]
                # 删除gens列表中的当前生成器对象
                del gens[j]
                # 更新rels列表，替换其中的关系对象
                rels = [rel.eliminate_word(gen, rep_by) for rel in rels]
                # 结束内层循环
                break
    # 更新对象C的Schreier生成器列表为更新后的rels列表
    C._reidemeister_relators = rels
    # 更新对象C的Reidemeister关系列表为更新后的gens列表
    C._schreier_generators = gens
    # 返回更新后的Schreier生成器列表和Reidemeister关系列表
    return C._schreier_generators, C._reidemeister_relators
# 定义函数 reidemeister_presentation，生成一个给定的 FpGroup 实例的表示
def reidemeister_presentation(fp_grp, H, C=None, homomorphism=False):
    """
    Parameters
    ==========

    fp_group: A finitely presented group, an instance of FpGroup
        有限呈现群，即 FpGroup 的实例
    H: A subgroup whose presentation is to be found, given as a list
        of words in generators of `fp_grp`
        要查找表示的子群，以 `fp_grp` 生成器的单词列表给出
    homomorphism: When set to True, return a homomorphism from the subgroup
                    to the parent group
        当设置为 True 时，返回子群到父群的同态映射

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, reidemeister_presentation
    >>> F, x, y = free_group("x, y")

    Example 5.6 Pg. 177 from [1]
    >>> f = FpGroup(F, [x**3, y**5, (x*y)**2])
    >>> H = [x*y, x**-1*y**-1*x*y*x]
    >>> reidemeister_presentation(f, H)
    ((y_1, y_2), (y_1**2, y_2**3, y_2*y_1*y_2*y_1*y_2*y_1))

    Example 5.8 Pg. 183 from [1]
    >>> f = FpGroup(F, [x**3, y**3, (x*y)**3])
    >>> H = [x*y, x*y**-1]
    >>> reidemeister_presentation(f, H)
    ((x_0, y_0), (x_0**3, y_0**3, x_0*y_0*x_0*y_0*x_0*y_0))

    Exercises Q2. Pg 187 from [1]
    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    >>> H = [x]
    >>> reidemeister_presentation(f, H)
    ((x_0,), (x_0**4,))

    Example 5.9 Pg. 183 from [1]
    >>> f = FpGroup(F, [x**3*y**-3, (x*y)**3, (x*y**-1)**2])
    >>> H = [x]
    >>> reidemeister_presentation(f, H)
    ((x_0,), (x_0**6,))

    """
    # 如果 C 未提供，则进行余等式枚举
    if not C:
        C = coset_enumeration_r(fp_grp, H)
    # 压缩余等式枚举结果并标准化
    C.compress(); C.standardize()
    # 定义 Schreier 生成器
    define_schreier_generators(C, homomorphism=homomorphism)
    # 计算 Reidemeister 关系
    reidemeister_relators(C)
    # 获取生成器和关系
    gens, rels = C._schreier_generators, C._reidemeister_relators
    # 简化表示
    gens, rels = simplify_presentation(gens, rels, change_gens=True)

    # 更新余等式枚举对象的生成器和关系
    C.schreier_generators = tuple(gens)
    C.reidemeister_relators = tuple(rels)

    # 如果需要同态映射，则返回 Schreier 生成器、Reidemeister 关系和相应的元素
    if homomorphism:
        _gens = [C._schreier_gen_elem[str(gen)] for gen in gens]
        return C.schreier_generators, C.reidemeister_relators, _gens

    # 否则，仅返回 Schreier 生成器和 Reidemeister 关系
    return C.schreier_generators, C.reidemeister_relators


# 定义 FpGroupElement 作为 FreeGroupElement 的别名
FpGroupElement = FreeGroupElement
```