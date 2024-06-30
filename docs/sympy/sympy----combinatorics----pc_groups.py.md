# `D:\src\scipysrc\sympy\sympy\combinatorics\pc_groups.py`

```
# 从 sympy.ntheory.primetest 模块中导入 isprime 函数，用于检查一个数是否为素数
from sympy.ntheory.primetest import isprime
# 从 sympy.combinatorics.perm_groups 模块中导入 PermutationGroup 类，用于处理置换群
from sympy.combinatorics.perm_groups import PermutationGroup
# 从 sympy.printing.defaults 模块中导入 DefaultPrinting 类，这是一个默认的打印设置类
from sympy.printing.defaults import DefaultPrinting
# 从 sympy.combinatorics.free_groups 模块中导入 free_group 函数，用于创建自由群
from sympy.combinatorics.free_groups import free_group


class PolycyclicGroup(DefaultPrinting):
    # 定义 PolycyclicGroup 类，继承自 DefaultPrinting 类

    is_group = True
    is_solvable = True

    def __init__(self, pc_sequence, pc_series, relative_order, collector=None):
        """
        初始化方法，创建 PolycyclicGroup 实例

        Parameters
        ==========

        pc_sequence : list
            一个元素序列，它们的类生成了 pc_series 的循环因子群。
        pc_series : list
            一个子正规序列的子群，其中每个因子群都是循环的。
        relative_order : list
            pc_series 的因子群的阶。
        collector : Collector
            默认为 None。Collector 类提供带有各种其他功能的多项式表示。

        """
        self.pcgs = pc_sequence  # 设置 pcgs 属性为 pc_sequence
        self.pc_series = pc_series  # 设置 pc_series 属性为 pc_series
        self.relative_order = relative_order  # 设置 relative_order 属性为 relative_order
        # 如果 collector 为 None，则使用给定参数创建 Collector 实例；否则使用已有的 collector
        self.collector = Collector(self.pcgs, pc_series, relative_order) if not collector else collector

    def is_prime_order(self):
        # 检查 relative_order 中所有的数是否都是素数，返回布尔值
        return all(isprime(order) for order in self.relative_order)

    def length(self):
        # 返回 pcgs 列表的长度
        return len(self.pcgs)


class Collector(DefaultPrinting):
    # 定义 Collector 类，继承自 DefaultPrinting 类

    """
    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"
           Section 8.1.3
    """

    def __init__(self, pcgs, pc_series, relative_order, free_group_=None, pc_presentation=None):
        """
        初始化方法，创建 Collector 实例

        Most of the parameters for the Collector class are the same as for PolycyclicGroup.
        Others are described below.

        Parameters
        ==========

        free_group_ : tuple
            free_group_ 提供了多项式生成序列与自由群元素之间的映射。
        pc_presentation : dict
            使用幂次和共轭关系提供了多项式群的表示。

        See Also
        ========

        PolycyclicGroup

        """
        self.pcgs = pcgs  # 设置 pcgs 属性为 pcgs
        self.pc_series = pc_series  # 设置 pc_series 属性为 pc_series
        self.relative_order = relative_order  # 设置 relative_order 属性为 relative_order
        # 如果 free_group_ 为 None，则创建自由群 'x:{}'.format(len(pcgs)) 的第一个元素，否则使用给定的 free_group_
        self.free_group = free_group('x:{}'.format(len(pcgs)))[0] if not free_group_ else free_group_
        # 创建 self.free_group.symbols 到索引的映射字典
        self.index = {s: i for i, s in enumerate(self.free_group.symbols)}
        # 调用 pc_relators 方法创建 pc_presentation 属性，表示多项式群的关系
        self.pc_presentation = self.pc_relators()
    def minimal_uncollected_subword(self, word):
        r"""
        Returns the minimal uncollected subwords.

        Explanation
        ===========

        A word ``v`` defined on generators in ``X`` is a minimal
        uncollected subword of the word ``w`` if ``v`` is a subword
        of ``w`` and it has one of the following form

        * `v = {x_{i+1}}^{a_j}x_i`

        * `v = {x_{i+1}}^{a_j}{x_i}^{-1}`

        * `v = {x_i}^{a_j}`

        for `a_j` not in `\{1, \ldots, s-1\}`. Where, ``s`` is the power
        exponent of the corresponding generator.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> collector.minimal_uncollected_subword(word)
        ((x2, 2),)

        """
        # To handle the case word = <identity>
        if not word:
            return None

        # Retrieve the array form of the word
        array = word.array_form
        # Get the relative order and index mappings
        re = self.relative_order
        index = self.index

        # Iterate over each pair in the array form of the word
        for i in range(len(array)):
            s1, e1 = array[i]

            # Check conditions for minimal uncollected subwords
            if re[index[s1]] and (e1 < 0 or e1 > re[index[s1]]-1):
                return ((s1, e1), )

        # Iterate over pairs of elements in the array form
        for i in range(len(array)-1):
            s1, e1 = array[i]
            s2, e2 = array[i+1]

            # Check conditions for minimal uncollected subwords
            if index[s1] > index[s2]:
                e = 1 if e2 > 0 else -1
                return ((s1, e1), (s2, e))

        # If no minimal uncollected subword found, return None
        return None

    def relations(self):
        """
        Separates the given relators of pc presentation in power and
        conjugate relations.

        Returns
        =======

        (power_rel, conj_rel)
            Separates pc presentation into power and conjugate relations.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> power_rel, conj_rel = collector.relations()
        >>> power_rel
        {x0**2: (), x1**3: ()}
        >>> conj_rel
        {x0**-1*x1*x0: x1**2}

        See Also
        ========

        pc_relators

        """
        # Initialize dictionaries for power and conjugate relations
        power_relators = {}
        conjugate_relators = {}

        # Iterate through the pc presentation items
        for key, value in self.pc_presentation.items():
            # Classify based on the length of array form
            if len(key.array_form) == 1:
                power_relators[key] = value
            else:
                conjugate_relators[key] = value

        # Return separated power and conjugate relations
        return power_relators, conjugate_relators
    def subword_index(self, word, w):
        """
        Returns the start and ending index of a given
        subword in a word.

        Parameters
        ==========

        word : FreeGroupElement
            word defined on free group elements for a
            polycyclic group.
        w : FreeGroupElement
            subword of a given word, whose starting and
            ending index to be computed.

        Returns
        =======

        (i, j)
            A tuple containing starting and ending index of ``w``
            in the given word. If not exists, (-1,-1) is returned.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> w = x2**2*x1
        >>> collector.subword_index(word, w)
        (0, 3)
        >>> w = x1**7
        >>> collector.subword_index(word, w)
        (2, 9)
        >>> w = x1**8
        >>> collector.subword_index(word, w)
        (-1, -1)

        """
        # 初始化起始和结束索引为 -1
        low = -1
        high = -1
        # 遍历主单词中所有可能的起始位置
        for i in range(len(word)-len(w)+1):
            # 检查从当前位置开始的子单词是否与给定子单词相同
            if word.subword(i, i+len(w)) == w:
                # 如果找到匹配，更新起始和结束索引，并结束循环
                low = i
                high = i+len(w)
                break
        # 返回起始和结束索引
        return low, high

    def map_relation(self, w):
        """
        Return a conjugate relation.

        Explanation
        ===========

        Given a word formed by two free group elements, the
        corresponding conjugate relation with those free
        group elements is formed and mapped with the collected
        word in the polycyclic presentation.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x0, x1 = free_group("x0, x1")
        >>> w = x1*x0
        >>> collector.map_relation(w)
        x1**2

        See Also
        ========

        pc_presentation

        """
        # 获取给定单词的数组形式
        array = w.array_form
        # 提取数组的第一个元素和第二个元素作为自由群元素
        s1 = array[0][0]
        s2 = array[1][0]
        # 定义一个键来表示共轭关系
        key = ((s2, -1), (s1, 1), (s2, 1))
        # 根据键获取并返回在多项式展示中对应的结果
        key = self.free_group.dtype(key)
        return self.pc_presentation[key]
    def exponent_vector(self, element):
        r"""
        Return the exponent vector of length equal to the
        length of polycyclic generating sequence.

        Explanation
        ===========

        For a given generator/element ``g`` of the polycyclic group,
        it can be represented as `g = {x_1}^{e_1}, \ldots, {x_n}^{e_n}`,
        where `x_i` represents polycyclic generators and ``n`` is
        the number of generators in the free_group equal to the length
        of pcgs.

        Parameters
        ==========

        element : Permutation
            Generator of a polycyclic group.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> pcgs = PcGroup.pcgs
        >>> collector.exponent_vector(G[0])
        [1, 0, 0, 0]
        >>> exp = collector.exponent_vector(G[1])
        >>> g = Permutation()
        >>> for i in range(len(exp)):
        ...     g = g*pcgs[i]**exp[i] if exp[i] else g
        >>> assert g == G[1]

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of Computational Group Theory"
               Section 8.1.1, Definition 8.4

        """
        # 获取自由群对象
        free_group = self.free_group
        # 创建置换群对象
        G = PermutationGroup()
        # 将pcgs中的生成元加入置换群生成器中
        for g in self.pcgs:
            G = PermutationGroup([g] + G.generators)
        # 获取元素的生成器积，并反转顺序
        gens = G.generator_product(element, original=True)
        gens.reverse()

        # 将置换映射到自由生成器的字典
        perm_to_free = {}
        for sym, g in zip(free_group.generators, self.pcgs):
            perm_to_free[g**-1] = sym**-1
            perm_to_free[g] = sym
        # 初始化单词为自由群的单位元素
        w = free_group.identity
        # 根据生成器序列更新单词
        for g in gens:
            w = w * perm_to_free[g]

        # 获取收集单词的数组形式
        word = self.collected_word(w)

        # 获取索引
        index = self.index
        # 初始化指数向量
        exp_vector = [0]*len(free_group)
        # 遍历单词数组，更新指数向量
        word = word.array_form
        for t in word:
            exp_vector[index[t[0]]] = t[1]
        # 返回指数向量
        return exp_vector
    def _sift(self, z, g):
        r"""
        Perform the sift operation in the context of a collector.

        Explanation
        ===========

        The sift operation modifies the element `g` based on the sequence `z`.
        It reduces `g` step-by-step according to certain conditions until `g`
        satisfies specific criteria.

        Parameters
        ==========
        z : list
            A list representing a sequence related to `g`.
        g : object
            An element on which the sift operation is performed.

        Returns
        =======
        object
            The sifted element `h`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> z = [1, 2, 0]
        >>> g = G[0]
        >>> collector._sift(z, g)
        G[1]

        Notes
        =====

        The sift operation is crucial in computational group theory,
        specifically in algorithms involving polycyclic presentations.

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of Computational Group Theory"
               Section 8.1.3, Algorithm 8.6

        """
        h = g
        # Calculate the depth of the element `h` using the `depth` method
        d = self.depth(h)
        # Perform the sift operation until certain conditions are met
        while d < len(self.pcgs) and z[d-1] != 1:
            # Extract the exponent `k` from `z` at position `d-1`
            k = z[d-1]
            # Calculate `e`, which is a modified exponent based on `h` and `k`
            e = self.leading_exponent(h) * (self.leading_exponent(k))**-1
            e = e % self.relative_order[d-1]
            # Update `h` according to the sift operation
            h = k**-e * h
            # Update the depth of `h`
            d = self.depth(h)
        return h
    def induced_pcgs(self, gens):
        """
        根据给定的生成元列表计算诱导多项循环子群（induced polycyclic subgroup）。

        Parameters
        ==========

        gens : list
            要定义多项循环子群的生成元列表。

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(8)
        >>> G = S.sylow_subgroup(2)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [2, 2, 2]
        >>> G = S.sylow_subgroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [3]

        """
        z = [1]*len(self.pcgs)  # 初始化指数向量z，长度与self.pcgs相同
        G = gens  # 将gens赋值给G
        while G:
            g = G.pop(0)  # 从G中移除并返回第一个元素作为g
            h = self._sift(z, g)  # 调用_sift方法，更新h
            d = self.depth(h)  # 计算h的深度d
            if d < len(self.pcgs):  # 如果d小于self.pcgs的长度
                for gen in z:
                    if gen != 1:
                        G.append(h**-1*gen**-1*h*gen)  # 更新G，添加新的生成元
                z[d-1] = h  # 更新指数向量z的第d-1个元素为h
        z = [gen for gen in z if gen != 1]  # 移除z中的单位元素
        return z  # 返回计算得到的诱导多项循环子群的生成元列表z

    def constructive_membership_test(self, ipcgs, g):
        """
        返回诱导多项循环子群的指数向量。

        Parameters
        ==========

        ipcgs : list
            诱导多项循环子群的生成元列表。

        g : object
            待测试成员身份的元素。

        """
        e = [0]*len(ipcgs)  # 初始化指数向量e，长度与ipcgs相同
        h = g  # 将g赋值给h
        d = self.depth(h)  # 计算h的深度d
        for i, gen in enumerate(ipcgs):
            while self.depth(gen) == d:
                f = self.leading_exponent(h)*self.leading_exponent(gen)
                f = f % self.relative_order[d-1]
                h = gen**(-f)*h  # 更新h
                e[i] = f  # 更新指数向量e的第i个元素为f
                d = self.depth(h)  # 更新h的深度d
        if h == 1:
            return e  # 如果h为单位元素，返回指数向量e
        return False  # 否则返回False，表示g不属于诱导多项循环子群ipcgs
```