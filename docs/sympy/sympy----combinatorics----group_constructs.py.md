# `D:\src\scipysrc\sympy\sympy\combinatorics\group_constructs.py`

```
# 从 sympy.combinatorics.perm_groups 模块中导入 PermutationGroup 类
# 从 sympy.combinatorics.permutations 模块中导入 Permutation 类
# 从 sympy.utilities.iterables 模块中导入 uniq 函数
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
from sympy.utilities.iterables import uniq

# 从 Permutation 类中导入 _af_new 方法
_af_new = Permutation._af_new

# 定义函数 DirectProduct，用于计算多个置换群的直积并返回一个置换群对象
def DirectProduct(*groups):
    """
    Returns the direct product of several groups as a permutation group.

    Explanation
    ===========

    This is implemented much like the __mul__ procedure for taking the direct
    product of two permutation groups, but the idea of shifting the
    generators is realized in the case of an arbitrary number of groups.
    A call to DirectProduct(G1, G2, ..., Gn) is generally expected to be faster
    than a call to G1*G2*...*Gn (and thus the need for this algorithm).

    Examples
    ========

    >>> from sympy.combinatorics.group_constructs import DirectProduct
    >>> from sympy.combinatorics.named_groups import CyclicGroup
    >>> C = CyclicGroup(4)
    >>> G = DirectProduct(C, C, C)
    >>> G.order()
    64

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.__mul__

    """
    # 初始化空列表 degrees 和 gens_count 用于存储每个群的度和生成元个数
    degrees = []
    gens_count = []
    total_degree = 0
    total_gens = 0
    
    # 遍历传入的每个群对象
    for group in groups:
        # 获取当前群的度和生成元个数
        current_deg = group.degree
        current_num_gens = len(group.generators)
        # 将当前群的度和生成元个数分别存入 degrees 和 gens_count 列表
        degrees.append(current_deg)
        total_degree += current_deg
        gens_count.append(current_num_gens)
        total_gens += current_num_gens
    
    # 初始化二维列表 array_gens，用于存储直积群的生成元
    array_gens = []
    for i in range(total_gens):
        # 每个生成元都是一个长度为 total_degree 的列表，初始化为 [0, 1, ..., total_degree-1]
        array_gens.append(list(range(total_degree)))
    
    # 初始化当前生成元的索引和当前度的索引
    current_gen = 0
    current_deg = 0
    
    # 遍历每个群对象
    for i in range(len(gens_count)):
        # 遍历当前群的每个生成元
        for j in range(current_gen, current_gen + gens_count[i]):
            # 获取当前生成元的数组形式
            gen = groups[i].generators[j - current_gen].array_form
            # 将当前生成元的数组形式映射到 array_gens 中相应的位置
            array_gens[j][current_deg:current_deg + degrees[i]] = \
                [x + current_deg for x in gen]
        # 更新当前生成元和当前度的索引
        current_gen += gens_count[i]
        current_deg += degrees[i]
    
    # 使用 _af_new 方法对 array_gens 中的每个生成元进行唯一化处理，得到置换生成元列表 perm_gens
    perm_gens = list(uniq([_af_new(list(a)) for a in array_gens]))
    
    # 返回一个新的 PermutationGroup 对象，其中包含处理后的置换生成元列表 perm_gens，且不允许重复
    return PermutationGroup(perm_gens, dups=False)
```