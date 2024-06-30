# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_enumerative.py`

```
from itertools import zip_longest

from sympy.utilities.enumerative import (
    list_visitor,
    MultisetPartitionTraverser,
    multiset_partitions_taocp
    )
from sympy.utilities.iterables import _set_partitions

# first some functions only useful as test scaffolding - these provide
# straightforward, but slow reference implementations against which to
# compare the real versions, and also a comparison to verify that
# different versions are giving identical results.

def part_range_filter(partition_iterator, lb, ub):
    """
    Filters (on the number of parts) a multiset partition enumeration

    Arguments
    =========

    lb, and ub are a range (in the Python slice sense) on the lpart
    variable returned from a multiset partition enumeration.  Recall
    that lpart is 0-based (it points to the topmost part on the part
    stack), so if you want to return parts of sizes 2,3,4,5 you would
    use lb=1 and ub=5.
    """
    # 迭代多重集分区枚举结果，根据部分数量范围进行过滤
    for state in partition_iterator:
        f, lpart, pstack = state
        # 如果部分数量在 lb 和 ub 之间，则将当前状态返回
        if lpart >= lb and lpart < ub:
            yield state

def multiset_partitions_baseline(multiplicities, components):
    """Enumerates partitions of a multiset

    Parameters
    ==========

    multiplicities
         list of integer multiplicities of the components of the multiset.

    components
         the components (elements) themselves

    Returns
    =======

    Set of partitions.  Each partition is tuple of parts, and each
    part is a tuple of components (with repeats to indicate
    multiplicity)

    Notes
    =====

    Multiset partitions can be created as equivalence classes of set
    partitions, and this function does just that.  This approach is
    slow and memory intensive compared to the more advanced algorithms
    available, but the code is simple and easy to understand.  Hence
    this routine is strictly for testing -- to provide a
    straightforward baseline against which to regress the production
    versions.  (This code is a simplified version of an earlier
    production implementation.)
    """

    canon = []                  # list of components with repeats
    # 根据组件的重复次数，构建具有重复的组件列表
    for ct, elem in zip(multiplicities, components):
        canon.extend([elem]*ct)

    # accumulate the multiset partitions in a set to eliminate dups
    cache = set()
    n = len(canon)
    # 使用 _set_partitions 函数进行集合的分区
    for nc, q in _set_partitions(n):
        rv = [[] for i in range(nc)]
        for i in range(n):
            rv[q[i]].append(canon[i])
        canonical = tuple(
            sorted([tuple(p) for p in rv]))
        cache.add(canonical)
    return cache


def compare_multiset_w_baseline(multiplicities):
    """
    Enumerates the partitions of multiset with AOCP algorithm and
    baseline implementation, and compare the results.

    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    # 使用基准实现获取多重集的分区
    bl_partitions = multiset_partitions_baseline(multiplicities, letters)

    # The partitions returned by the different algorithms may have
    # 创建一个空集合，用于存储生成的分区结果
    aocp_partitions = set()
    
    # 使用 multiset_partitions_taocp 函数生成多重集的所有可能分区，并逐个处理
    for state in multiset_partitions_taocp(multiplicities):
        # 对于每个生成的分区 state，使用 list_visitor 函数根据字母顺序访问并生成元组列表
        # 然后对这些元组列表进行排序，并将排序后的元组转换成元组的元组，形成 p1
        p1 = tuple(sorted(
                [tuple(p) for p in list_visitor(state, letters)]))
        
        # 将处理好的分区结果 p1 添加到 aocp_partitions 集合中
        aocp_partitions.add(p1)
    
    # 断言语句，用于确保两个集合 bl_partitions 和 aocp_partitions 相等
    assert bl_partitions == aocp_partitions
# 比较两个多重集分割状态实例的相等性
def compare_multiset_states(s1, s2):
    """compare for equality two instances of multiset partition states

    This is useful for comparing different versions of the algorithm
    to verify correctness."""
    # 解包元组 s1 和 s2，分别获取第一个元素 f1、分割点 lpart1 和堆栈 pstack1
    f1, lpart1, pstack1 = s1
    f2, lpart2, pstack2 = s2

    # 如果分割点相等并且 f1 在区间 [0:lpart1+1] 内与 f2 在区间 [0:lpart2+1] 内相等
    if (lpart1 == lpart2) and (f1[0:lpart1+1] == f2[0:lpart2+1]):
        # 如果 pstack1 在区间 [0:f1[lpart1+1]] 内与 pstack2 在区间 [0:f2[lpart2+1]] 内相等
        if pstack1[0:f1[lpart1+1]] == pstack2[0:f2[lpart2+1]]:
            return True
    return False

# 测试 multiset_partitions_taocp 与基准（基于集合分割的）实现的输出是否一致
def test_multiset_partitions_taocp():
    """Compares the output of multiset_partitions_taocp with a baseline
    (set partition based) implementation."""

    # 测试案例不应过大，因为基准实现速度较慢
    multiplicities = [2,2]
    compare_multiset_w_baseline(multiplicities)

    multiplicities = [4,3,1]
    compare_multiset_w_baseline(multiplicities)

# 比较基于 Knuth 的 multiset_partitions 的不同版本
def test_multiset_partitions_versions():
    """Compares Knuth-based versions of multiset_partitions"""
    multiplicities = [5,2,2,1]
    m = MultisetPartitionTraverser()
    for s1, s2 in zip_longest(m.enum_all(multiplicities),
                              multiset_partitions_taocp(multiplicities)):
        assert compare_multiset_states(s1, s2)

# 比较基于过滤器和更优化的子范围实现
def subrange_exercise(mult, lb, ub):
    """Compare filter-based and more optimized subrange implementations

    Helper for tests, called with both small and larger multisets.
    """
    m = MultisetPartitionTraverser()
    # 断言使用普通和慢速计算的分割数量相同
    assert m.count_partitions(mult) == \
        m.count_partitions_slow(mult)

    # 注意 - 同一个 MultisetPartitionTraverser 对象上的多个遍历不能同时执行，因此在此创建多个实例。
    ma = MultisetPartitionTraverser()
    mc = MultisetPartitionTraverser()
    md = MultisetPartitionTraverser()

    # 多个路径计算仅包含大小为二的分割
    a_it = ma.enum_range(mult, lb, ub)
    b_it = part_range_filter(multiset_partitions_taocp(mult), lb, ub)
    c_it = part_range_filter(mc.enum_small(mult, ub), lb, sum(mult))
    d_it = part_range_filter(md.enum_large(mult, lb), 0, ub)

    for sa, sb, sc, sd in zip_longest(a_it, b_it, c_it, d_it):
        # 分别比较 sa 与 sb、sa 与 sc、sa 与 sd 的状态
        assert compare_multiset_states(sa, sb)
        assert compare_multiset_states(sa, sc)
        assert compare_multiset_states(sa, sd)

# 测试 subrange_exercise 函数的子范围功能
def test_subrange():
    # 快速测试，但不涵盖所有边界情况
    mult = [4,4,2,1] # mississippi
    lb = 1
    ub = 2
    subrange_exercise(mult, lb, ub)


def test_subrange_large():
    # 时间较长，取决于 CPU、Python 版本等因素
    mult = [6,3,2,1]
    lb = 4
    ub = 7
    subrange_exercise(mult, lb, ub)
```