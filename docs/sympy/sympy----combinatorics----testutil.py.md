# `D:\src\scipysrc\sympy\sympy\combinatorics\testutil.py`

```
# 从 sympy.combinatorics 模块中导入 Permutation 类
from sympy.combinatorics import Permutation
# 从 sympy.combinatorics.util 模块中导入 _distribute_gens_by_base 函数
from sympy.combinatorics.util import _distribute_gens_by_base

# 将 Permutation 类的 rmul 方法赋值给 rmul 变量
rmul = Permutation.rmul


def _cmp_perm_lists(first, second):
    """
    Compare two lists of permutations as sets.

    Explanation
    ===========

    This is used for testing purposes. Since the array form of a
    permutation is currently a list, Permutation is not hashable
    and cannot be put into a set.

    Examples
    ========

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.testutil import _cmp_perm_lists
    >>> a = Permutation([0, 2, 3, 4, 1])
    >>> b = Permutation([1, 2, 0, 4, 3])
    >>> c = Permutation([3, 4, 0, 1, 2])
    >>> ls1 = [a, b, c]
    >>> ls2 = [b, c, a]
    >>> _cmp_perm_lists(ls1, ls2)
    True

    """
    # 将每个排列转换为元组，并比较作为集合的相等性
    return {tuple(a) for a in first} == \
           {tuple(a) for a in second}


def _naive_list_centralizer(self, other, af=False):
    # 从 sympy.combinatorics.perm_groups 模块导入 PermutationGroup 类
    from sympy.combinatorics.perm_groups import PermutationGroup
    """
    Return a list of elements for the centralizer of a subgroup/set/element.

    Explanation
    ===========

    This is a brute force implementation that goes over all elements of the
    group and checks for membership in the centralizer. It is used to
    test ``.centralizer()`` from ``sympy.combinatorics.perm_groups``.

    Examples
    ========

    >>> from sympy.combinatorics.testutil import _naive_list_centralizer
    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> D = DihedralGroup(4)
    >>> _naive_list_centralizer(D, D)
    [Permutation([0, 1, 2, 3]), Permutation([2, 3, 0, 1])]

    See Also
    ========

    sympy.combinatorics.perm_groups.centralizer

    """
    # 从 sympy.combinatorics.permutations 模块导入 _af_commutes_with 函数
    from sympy.combinatorics.permutations import _af_commutes_with
    # 如果 other 具有 generators 属性，获取 self 的维数生成元列表
    if hasattr(other, 'generators'):
        elements = list(self.generate_dimino(af=True))
        gens = [x._array_form for x in other.generators]
        # 检查元素与生成元是否可交换，并将属于中心化器的元素加入列表
        commutes_with_gens = lambda x: all(_af_commutes_with(x, gen) for gen in gens)
        centralizer_list = []
        if not af:
            for element in elements:
                if commutes_with_gens(element):
                    centralizer_list.append(Permutation._af_new(element))
        else:
            for element in elements:
                if commutes_with_gens(element):
                    centralizer_list.append(element)
        return centralizer_list
    # 如果 other 具有 getitem 属性，将其作为 PermutationGroup 处理
    elif hasattr(other, 'getitem'):
        return _naive_list_centralizer(self, PermutationGroup(other), af)
    # 如果 other 具有 array_form 属性，将其作为 PermutationGroup 处理
    elif hasattr(other, 'array_form'):
        return _naive_list_centralizer(self, PermutationGroup([other]), af)


def _verify_bsgs(group, base, gens):
    """
    Verify the correctness of a base and strong generating set.

    Explanation
    ===========

    This is a naive implementation using the definition of a base and a strong
    generating set relative to it. There are other procedures for
    verifying a base and strong generating set, but this one will

    """
    # 这是一个验证基和强生成集正确性的简单实现
    # 相对于其他验证方法，这个方法是基于基和强生成集的定义
    """
    Verify if the given base and generators define a strong generating set for the group.

    Parameters
    ==========
    base : list
        A list representing the base of the strong generating set.
    gens : list
        A list of generators of the group.
    group : PermutationGroup
        The permutation group to be verified.

    Returns
    =======
    bool
        True if the base and generators form a strong generating set for the group, False otherwise.

    Notes
    =====
    This function checks if the generators distributed by base create a strong generating set
    that is equivalent in order to the given group.

    Examples
    ========
    
    >>> from sympy.combinatorics.named_groups import AlternatingGroup
    >>> from sympy.combinatorics.testutil import _verify_bsgs
    >>> A = AlternatingGroup(4)
    >>> A.schreier_sims()
    >>> _verify_bsgs(A, A.base, A.strong_gens)
    True

    See Also
    ========
    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims
    """
    from sympy.combinatorics.perm_groups import PermutationGroup
    
    # Distribute generators by base to form strong generating sets
    strong_gens_distr = _distribute_gens_by_base(base, gens)
    
    # Start with the entire group as the current stabilizer
    current_stabilizer = group
    
    # Iterate over each element in the base
    for i in range(len(base)):
        # Create a PermutationGroup from the generators distributed by base
        candidate = PermutationGroup(strong_gens_distr[i])
        
        # Check if the order of the current stabilizer matches the candidate group
        if current_stabilizer.order() != candidate.order():
            return False
        
        # Update the current stabilizer to be the stabilizer of the current base element
        current_stabilizer = current_stabilizer.stabilizer(base[i])
    
    # Finally, check if the order of the last stabilizer is 1
    if current_stabilizer.order() != 1:
        return False
    
    # If all checks pass, return True indicating the base and generators form a strong generating set
    return True
def _verify_centralizer(group, arg, centr=None):
    """
    Verify the centralizer of a group/set/element inside another group.

    This is used for testing ``.centralizer()`` from
    ``sympy.combinatorics.perm_groups``

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
    ... AlternatingGroup)
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.testutil import _verify_centralizer
    >>> S = SymmetricGroup(5)
    >>> A = AlternatingGroup(5)
    >>> centr = PermutationGroup([Permutation([0, 1, 2, 3, 4])])
    >>> _verify_centralizer(S, A, centr)
    True

    See Also
    ========

    _naive_list_centralizer,
    sympy.combinatorics.perm_groups.PermutationGroup.centralizer,
    _cmp_perm_lists

    """
    # 如果未提供中心化子（centralizer），则通过 group.centralizer(arg) 计算
    if centr is None:
        centr = group.centralizer(arg)
    # 使用 DIMINO 算法生成中心化子的生成器列表
    centr_list = list(centr.generate_dimino(af=True))
    # 使用朴素方法计算中心化子的生成器列表
    centr_list_naive = _naive_list_centralizer(group, arg, af=True)
    # 比较两个生成器列表是否相等
    return _cmp_perm_lists(centr_list, centr_list_naive)


def _verify_normal_closure(group, arg, closure=None):
    from sympy.combinatorics.perm_groups import PermutationGroup
    """
    Verify the normal closure of a subgroup/subset/element in a group.

    This is used to test
    sympy.combinatorics.perm_groups.PermutationGroup.normal_closure

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
    ... AlternatingGroup)
    >>> from sympy.combinatorics.testutil import _verify_normal_closure
    >>> S = SymmetricGroup(3)
    >>> A = AlternatingGroup(3)
    >>> _verify_normal_closure(S, A, closure=A)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.normal_closure

    """
    # 如果未提供闭包（closure），则通过 group.normal_closure(arg) 计算
    if closure is None:
        closure = group.normal_closure(arg)
    # 初始化共轭集合
    conjugates = set()
    # 根据 arg 类型获取子群生成器列表
    if hasattr(arg, 'generators'):
        subgr_gens = arg.generators
    elif hasattr(arg, '__getitem__'):
        subgr_gens = arg
    elif hasattr(arg, 'array_form'):
        subgr_gens = [arg]
    # 生成 group 的 DIMINO 生成器并更新共轭集合
    for el in group.generate_dimino():
        conjugates.update(gen ^ el for gen in subgr_gens)
    # 使用共轭集合构建朴素闭包的置换群
    naive_closure = PermutationGroup(list(conjugates))
    # 检查给定闭包是否为朴素闭包的子群
    return closure.is_subgroup(naive_closure)


def canonicalize_naive(g, dummies, sym, *v):
    """
    Canonicalize tensor formed by tensors of the different types.

    Explanation
    ===========

    sym_i symmetry under exchange of two component tensors of type `i`
          None  no symmetry
          0     commuting
          1     anticommuting

    Parameters
    ==========

    g : Permutation representing the tensor.
    dummies : List of dummy indices.
    msym : Symmetry of the metric.
    v : A list of (base_i, gens_i, n_i, sym_i) for tensors of type `i`.
        base_i, gens_i BSGS for tensors of this type
        n_i  number of tensors of type `i`

    Returns
    =======

    """
    # 省略了函数体，暂无代码需要注释
    Returns 0 if the tensor is zero, else returns the array form of
    the permutation representing the canonical form of the tensor.

    Examples
    ========

    >>> from sympy.combinatorics.testutil import canonicalize_naive
    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs
    >>> from sympy.combinatorics import Permutation
    >>> g = Permutation([1, 3, 2, 0, 4, 5])
    >>> base2, gens2 = get_symmetric_group_sgs(2)
    >>> canonicalize_naive(g, [2, 3], 0, (base2, gens2, 2, 0))
    [0, 2, 1, 3, 4, 5]
    """
    # 导入需要的模块和函数
    from sympy.combinatorics.perm_groups import PermutationGroup
    from sympy.combinatorics.tensor_can import gens_products, dummy_sgs
    from sympy.combinatorics.permutations import _af_rmul
    
    # 初始化空列表 v1
    v1 = []
    # 遍历 v 中的元素
    for i in range(len(v)):
        # 解包 v[i] 中的元组
        base_i, gens_i, n_i, sym_i = v[i]
        # 将解包后的元组组成新的元组，添加到 v1 列表中
        v1.append((base_i, gens_i, [[]]*n_i, sym_i))
    
    # 调用 gens_products 函数，生成 size, sbase, sgens
    size, sbase, sgens = gens_products(*v1)
    # 调用 dummy_sgs 函数，生成 dgens
    dgens = dummy_sgs(dummies, sym, size-2)
    
    # 如果 sym 是整数，则将其包装成列表
    if isinstance(sym, int):
        num_types = 1
        dummies = [dummies]
        sym = [sym]
    else:
        num_types = len(sym)
    
    # 清空 dgens 列表
    dgens = []
    # 遍历 num_types
    for i in range(num_types):
        # 调用 dummy_sgs 函数，将生成的结果扩展到 dgens 中
        dgens.extend(dummy_sgs(dummies[i], sym[i], size - 2))
    
    # 创建置换群对象 S，传入 sgens
    S = PermutationGroup(sgens)
    # 创建置换群对象 D，将 dgens 转换为 Permutation 对象后传入
    D = PermutationGroup([Permutation(x) for x in dgens])
    
    # 生成 D 的所有生成元，以列表形式存储在 dlist 中
    dlist = list(D.generate(af=True))
    # 将 g 转换为其数组形式
    g = g.array_form
    # 创建空集合 st
    st = set()
    
    # 遍历 S 的所有生成元
    for s in S.generate(af=True):
        # 计算 h = g * s
        h = _af_rmul(g, s)
        # 遍历 dlist 中的每个置换 d
        for d in dlist:
            # 计算 q = d * h
            q = tuple(_af_rmul(d, h))
            # 将 q 添加到集合 st 中
            st.add(q)
    
    # 将集合 st 转换为列表 a，并进行排序
    a = list(st)
    a.sort()
    
    # 初始化 prev 为全零元组
    prev = (0,)*size
    # 遍历 a 中的每个元组 h
    for h in a:
        # 如果 h 的前面部分（除最后两个元素外）与 prev 相同
        if h[:-2] == prev[:-2]:
            # 如果 h 的倒数第一个元素与 prev 的倒数第一个元素不同，返回 0
            if h[-1] != prev[-1]:
                return 0
        # 更新 prev 为当前 h
        prev = h
    
    # 返回排序后的 a 的第一个元素（即规范形式的数组形式）
    return list(a[0])
# 返回图的证书
def graph_certificate(gr):
    """
    返回图的证书

    参数
    ==========

    gr : 邻接列表

    说明
    ===========

    假设图是无向且没有外部线。

    将图的每个顶点关联到一个对称张量，其索引数等于顶点的度；当索引对应于图的同一条线时，进行收缩。
    张量的规范形式给出了图的证书。

    这不是获取图证书的高效算法。

    示例
    ========

    >>> from sympy.combinatorics.testutil import graph_certificate
    >>> gr1 = {0:[1, 2, 3, 5], 1:[0, 2, 4], 2:[0, 1, 3, 4], 3:[0, 2, 4], 4:[1, 2, 3, 5], 5:[0, 4]}
    >>> gr2 = {0:[1, 5], 1:[0, 2, 3, 4], 2:[1, 3, 5], 3:[1, 2, 4, 5], 4:[1, 3, 5], 5:[0, 2, 3, 4]}
    >>> c1 = graph_certificate(gr1)
    >>> c2 = graph_certificate(gr2)
    >>> c1
    [0, 2, 4, 6, 1, 8, 10, 12, 3, 14, 16, 18, 5, 9, 15, 7, 11, 17, 13, 19, 20, 21]
    >>> c1 == c2
    True
    """
    from sympy.combinatorics.permutations import _af_invert
    from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize

    # 按照邻接点数量排序顶点，并生成其反序列
    items = list(gr.items())
    items.sort(key=lambda x: len(x[1]), reverse=True)
    pvert = [x[0] for x in items]
    pvert = _af_invert(pvert)

    # 张量的索引数是图的边数的两倍
    num_indices = 0
    for v, neigh in items:
        num_indices += len(neigh)

    # 为每个顶点关联其索引；对于每条边，将偶数索引分配给在items中排名靠前的顶点，将奇数索引分配给另一个顶点
    vertices = [[] for i in items]
    i = 0
    for v, neigh in items:
        for v2 in neigh:
            if pvert[v] < pvert[v2]:
                vertices[pvert[v]].append(i)
                vertices[pvert[v2]].append(i+1)
                i += 2

    # 将所有顶点的索引扁平化为一个列表
    g = []
    for v in vertices:
        g.extend(v)

    # 检查总索引数是否正确，并添加两个额外索引用于规范化
    assert len(g) == num_indices
    g += [num_indices, num_indices + 1]
    size = num_indices + 2
    assert sorted(g) == list(range(size))

    # 生成置换对象
    g = Permutation(g)

    # 统计每个顶点度数的出现次数
    vlen = [0]*(len(vertices[0])+1)
    for neigh in vertices:
        vlen[len(neigh)] += 1

    # 生成用于规范化的信息列表
    v = []
    for i in range(len(vlen)):
        n = vlen[i]
        if n:
            base, gens = get_symmetric_group_sgs(i)
            v.append((base, gens, n, 0))
    v.reverse()

    # 规范化张量并返回
    dummies = list(range(num_indices))
    can = canonicalize(g, dummies, 0, *v)
    return can
```