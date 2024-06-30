# `D:\src\scipysrc\sympy\sympy\combinatorics\partitions.py`

```
from sympy.core import Basic, Dict, sympify, Tuple  # 导入符号计算相关模块
from sympy.core.numbers import Integer  # 导入整数相关模块
from sympy.core.sorting import default_sort_key  # 导入排序相关模块
from sympy.core.sympify import _sympify  # 导入符号化相关函数
from sympy.functions.combinatorial.numbers import bell  # 导入贝尔数相关函数
from sympy.matrices import zeros  # 导入矩阵相关模块
from sympy.sets.sets import FiniteSet, Union  # 导入集合相关类和函数
from sympy.utilities.iterables import flatten, group  # 导入迭代工具类函数
from sympy.utilities.misc import as_int  # 导入类型转换函数


from collections import defaultdict  # 导入默认字典相关模块


class Partition(FiniteSet):
    """
    This class represents an abstract partition.

    A partition is a set of disjoint sets whose union equals a given set.

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions
    """

    _rank = None  # 分区的秩
    _partition = None  # 分区的具体分组

    def __new__(cls, *partition):
        """
        Generates a new partition object.

        This method also verifies if the arguments passed are
        valid and raises a ValueError if they are not.

        Examples
        ========

        Creating Partition from Python lists:

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a
        Partition({3}, {1, 2})
        >>> a.partition
        [[1, 2], [3]]
        >>> len(a)
        2
        >>> a.members
        (1, 2, 3)

        Creating Partition from Python sets:

        >>> Partition({1, 2, 3}, {4, 5})
        Partition({4, 5}, {1, 2, 3})

        Creating Partition from SymPy finite sets:

        >>> from sympy import FiniteSet
        >>> a = FiniteSet(1, 2, 3)
        >>> b = FiniteSet(4, 5)
        >>> Partition(a, b)
        Partition({4, 5}, {1, 2, 3})
        """
        args = []
        dups = False
        for arg in partition:
            if isinstance(arg, list):
                as_set = set(arg)  # 将列表转换为集合
                if len(as_set) < len(arg):
                    dups = True
                    break  # 如果存在重复元素，抛出错误
                arg = as_set
            args.append(_sympify(arg))  # 符号化每个参数

        if not all(isinstance(part, FiniteSet) for part in args):
            raise ValueError(
                "Each argument to Partition should be " \
                "a list, set, or a FiniteSet")  # 参数必须是列表、集合或有限集合

        # sort so we have a canonical reference for RGS
        U = Union(*args)  # 创建参数的并集
        if dups or len(U) < sum(len(arg) for arg in args):
            raise ValueError("Partition contained duplicate elements.")  # 如果有重复元素，抛出错误

        obj = FiniteSet.__new__(cls, *args)  # 创建新的有限集合对象
        obj.members = tuple(U)  # 设置对象的成员属性为并集元素的元组形式
        obj.size = len(U)  # 设置对象的大小属性为并集的长度
        return obj
    def sort_key(self, order=None):
        """Return a canonical key that can be used for sorting.

        Ordering is based on the size and sorted elements of the partition
        and ties are broken with the rank.

        Examples
        ========

        >>> from sympy import default_sort_key
        >>> from sympy.combinatorics import Partition
        >>> from sympy.abc import x
        >>> a = Partition([1, 2])
        >>> b = Partition([3, 4])
        >>> c = Partition([1, x])
        >>> d = Partition(list(range(4)))
        >>> l = [d, b, a + 1, a, c]
        >>> l.sort(key=default_sort_key); l
        [Partition({1, 2}), Partition({1}, {2}), Partition({1, x}), Partition({3, 4}), Partition({0, 1, 2, 3})]
        """
        if order is None:
            members = self.members
        else:
            # 对 self.members 进行排序，排序规则由 default_sort_key(w, order) 决定
            members = tuple(sorted(self.members,
                             key=lambda w: default_sort_key(w, order)))
        # 返回一个元组，包含 self.size、members 和 self.rank 经 default_sort_key 处理后的结果
        return tuple(map(default_sort_key, (self.size, members, self.rank)))

    @property
    def partition(self):
        """Return partition as a sorted list of lists.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> Partition([1], [2, 3]).partition
        [[1], [2, 3]]
        """
        if self._partition is None:
            # 如果 self._partition 尚未计算，对 self.args 中的每个分区进行排序，并使用 default_sort_key 进行元素排序
            self._partition = sorted([sorted(p, key=default_sort_key)
                                      for p in self.args])
        # 返回已排序的分区列表
        return self._partition

    def __add__(self, other):
        """
        Return permutation whose rank is ``other`` greater than current rank,
        (mod the maximum rank for the set).

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a.rank
        1
        >>> (a + 1).rank
        2
        >>> (a + 100).rank
        1
        """
        other = as_int(other)
        # 计算新的排列的偏移量，确保在最大排列数的模下
        offset = self.rank + other
        # 使用 RGS_unrank 函数根据偏移量和大小创建新的排列
        result = RGS_unrank((offset) %
                            RGS_enum(self.size),
                            self.size)
        # 返回一个新的 Partition 对象，使用从 RGS_unrank 得到的结果和当前成员
        return Partition.from_rgs(result, self.members)

    def __sub__(self, other):
        """
        Return permutation whose rank is ``other`` less than current rank,
        (mod the maximum rank for the set).

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a.rank
        1
        >>> (a - 1).rank
        0
        >>> (a - 100).rank
        1
        """
        # 实现减法运算为加上负数
        return self.__add__(-other)
    def __le__(self, other):
        """
        Checks if a partition is less than or equal to
        the other based on rank.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3, 4, 5])
        >>> b = Partition([1], [2, 3], [4], [5])
        >>> a.rank, b.rank
        (9, 34)
        >>> a <= a
        True
        >>> a <= b
        True
        """
        # 使用排序键比较自身和其他分区对象的排序键，判断是否小于或等于
        return self.sort_key() <= sympify(other).sort_key()

    def __lt__(self, other):
        """
        Checks if a partition is less than the other.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3, 4, 5])
        >>> b = Partition([1], [2, 3], [4], [5])
        >>> a.rank, b.rank
        (9, 34)
        >>> a < b
        True
        """
        # 使用排序键比较自身和其他分区对象的排序键，判断是否严格小于
        return self.sort_key() < sympify(other).sort_key()

    @property
    def rank(self):
        """
        Gets the rank of a partition.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3], [4, 5])
        >>> a.rank
        13
        """
        # 如果已经计算过分区的等级，则直接返回
        if self._rank is not None:
            return self._rank
        # 否则计算分区的等级并缓存
        self._rank = RGS_rank(self.RGS)
        return self._rank

    @property
    def RGS(self):
        """
        Returns the "restricted growth string" of the partition.

        Explanation
        ===========

        The RGS is returned as a list of indices, L, where L[i] indicates
        the block in which element i appears. For example, in a partition
        of 3 elements (a, b, c) into 2 blocks ([c], [a, b]) the RGS is
        [1, 1, 0]: "a" is in block 1, "b" is in block 1 and "c" is in block 0.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3], [4, 5])
        >>> a.members
        (1, 2, 3, 4, 5)
        >>> a.RGS
        (0, 0, 1, 2, 2)
        >>> a + 1
        Partition({3}, {4}, {5}, {1, 2})
        >>> _.RGS
        (0, 0, 1, 2, 3)
        """
        # 创建一个空字典来存储元素到块的映射关系
        rgs = {}
        partition = self.partition
        # 遍历分区中的每个部分，并将每个元素映射到其块的索引
        for i, part in enumerate(partition):
            for j in part:
                rgs[j] = i
        # 返回排序后的元素的映射结果，形成元组
        return tuple([rgs[i] for i in sorted(
            [i for p in partition for i in p], key=default_sort_key)])

    @classmethod
    def from_rgs(self, rgs, elements):
        """
        Creates a set partition from a restricted growth string.

        Explanation
        ===========

        The indices given in rgs are assumed to be the index
        of the element as given in elements *as provided* (the
        elements are not sorted by this routine). Block numbering
        starts from 0. If any block was not referenced in ``rgs``
        an error will be raised.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))
        Partition({c}, {a, d}, {b, e})
        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))
        Partition({e}, {a, c}, {b, d})
        >>> a = Partition([1, 4], [2], [3, 5])
        >>> Partition.from_rgs(a.RGS, a.members)
        Partition({2}, {1, 4}, {3, 5})
        """
        # 检查输入的 rgs 和 elements 的长度是否一致
        if len(rgs) != len(elements):
            raise ValueError('mismatch in rgs and element lengths')
        
        # 计算 rgs 中的最大值，并作为分区的块数目
        max_elem = max(rgs) + 1
        # 初始化一个空列表，用于存放每个块的元素
        partition = [[] for i in range(max_elem)]
        j = 0
        # 根据 rgs 将 elements 中的元素分配到对应的块中
        for i in rgs:
            partition[i].append(elements[j])
            j += 1
        
        # 检查是否所有的分区块都非空
        if not all(p for p in partition):
            raise ValueError('some blocks of the partition were empty.')
        
        # 返回创建的 Partition 对象，使用 partition 列表作为参数
        return Partition(*partition)
# IntegerPartition 类定义，继承自 Basic 类
class IntegerPartition(Basic):
    """
    This class represents an integer partition.

    Explanation
    ===========

    In number theory and combinatorics, a partition of a positive integer,
    ``n``, also called an integer partition, is a way of writing ``n`` as a
    list of positive integers that sum to n. Two partitions that differ only
    in the order of summands are considered to be the same partition; if order
    matters then the partitions are referred to as compositions. For example,
    4 has five partitions: [4], [3, 1], [2, 2], [2, 1, 1], and [1, 1, 1, 1];
    the compositions [1, 2, 1] and [1, 1, 2] are the same as partition
    [2, 1, 1].

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_%28number_theory%29
    """

    # 存储分区的字典，默认为 None
    _dict = None
    # 存储键的列表，默认为 None
    _keys = None
    def __new__(cls, partition, integer=None):
        """
        从列表或字典生成一个新的IntegerPartition对象。

        Explanation
        ===========

        partition可以作为正整数列表或(integer, multiplicity)项的字典提供。如果partition前面有一个整数，
        如果partition的总和不等于给定的整数，则会引发错误。

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([5, 4, 3, 1, 1])
        >>> a
        IntegerPartition(14, (5, 4, 3, 1, 1))
        >>> print(a)
        [5, 4, 3, 1, 1]
        >>> IntegerPartition({1:3, 2:1})
        IntegerPartition(5, (2, 1, 1, 1))

        如果partition应该总和的值首先给出，将检查是否有任何不一致：

        >>> IntegerPartition(10, [5, 4, 3, 1])
        Traceback (most recent call last):
        ...
        ValueError: The partition is not valid

        """
        # 如果integer不是None，则将partition和integer交换
        if integer is not None:
            integer, partition = partition, integer
        # 如果partition是字典或Dict类型
        if isinstance(partition, (dict, Dict)):
            # 创建一个空列表
            _ = []
            # 遍历排序后的partition.items()，降序
            for k, v in sorted(partition.items(), reverse=True):
                # 如果v为假值，则继续下一次循环
                if not v:
                    continue
                # 将k和v转换为整数类型
                k, v = as_int(k), as_int(v)
                # 将k重复v次添加到列表_
                _.extend([k]*v)
            # 将列表_转换为元组，并赋值给partition
            partition = tuple(_)
        else:
            # 将partition转换为整数类型的列表，排序后降序，并转换为元组
            partition = tuple(sorted(map(as_int, partition), reverse=True))
        # 初始化sum_ok为False
        sum_ok = False
        # 如果integer为None
        if integer is None:
            # 计算partition的总和，并将结果赋值给integer，将sum_ok设置为True
            integer = sum(partition)
            sum_ok = True
        else:
            # 将integer转换为整数类型
            integer = as_int(integer)

        # 如果sum_ok不为True且partition的总和不等于integer，则引发ValueError异常
        if not sum_ok and sum(partition) != integer:
            raise ValueError("Partition did not add to %s" % integer)
        # 如果partition中有任何小于1的整数，则引发ValueError异常
        if any(i < 1 for i in partition):
            raise ValueError("All integer summands must be greater than one")

        # 创建一个新的对象，基于cls类，传入整数部分和分区的元组
        obj = Basic.__new__(cls, Integer(integer), Tuple(*partition))
        # 将partition转换为列表，并赋值给obj.partition
        obj.partition = list(partition)
        # 将integer赋值给obj.integer
        obj.integer = integer
        # 返回新创建的对象obj
        return obj
    def prev_lex(self):
        """Return the previous partition of the integer, n, in lexical order,
        wrapping around to [1, ..., 1] if the partition is [n].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([4])
        >>> print(p.prev_lex())
        [3, 1]
        >>> p.partition > p.prev_lex().partition
        True
        """
        # 使用 defaultdict 创建一个空的默认字典 d
        d = defaultdict(int)
        # 将当前对象的分区表示转换为字典形式，并更新到字典 d 中
        d.update(self.as_dict())
        # 获取当前分区的键列表 keys
        keys = self._keys
        # 如果 keys 只包含 [1]，返回一个新的 IntegerPartition 对象，表示分区 [n]
        if keys == [1]:
            return IntegerPartition({self.integer: 1})
        # 如果 keys 中最后一个元素不为 1
        if keys[-1] != 1:
            # 将 d 中 keys 中最后一个元素对应的值减 1
            d[keys[-1]] -= 1
            # 如果 keys 中最后一个元素为 2
            if keys[-1] == 2:
                # 设置 d 中键为 1 的值为 2
                d[1] = 2
            else:
                # 否则设置 d 中 keys 中最后一个元素减 1 和键为 1 的值均为 1
                d[keys[-1] - 1] = d[1] = 1
        else:
            # 如果 keys 中最后一个元素为 1
            # 将 d 中倒数第二个键对应的值减 1
            d[keys[-2]] -= 1
            # 计算 left 为 d 中键为 1 的值加上 keys 中倒数第二个元素
            left = d[1] + keys[-2]
            # 设置 new 为 keys 中倒数第二个元素
            new = keys[-2]
            # 将 d 中键为 1 的值置为 0
            d[1] = 0
            # 当 left 不为 0 时进行循环
            while left:
                # new 减 1
                new -= 1
                # 如果 left 减 new 大于等于 0
                if left - new >= 0:
                    # 将 d 中键为 new 的值加上 left 除以 new 的整数部分
                    d[new] += left // new
                    # left 减去 d 中键为 new 的值乘以 new
                    left -= d[new] * new
        # 返回一个新的 IntegerPartition 对象，表示计算得到的分区
        return IntegerPartition(self.integer, d)

    def next_lex(self):
        """Return the next partition of the integer, n, in lexical order,
        wrapping around to [n] if the partition is [1, ..., 1].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([3, 1])
        >>> print(p.next_lex())
        [4]
        >>> p.partition < p.next_lex().partition
        True
        """
        # 使用 defaultdict 创建一个空的默认字典 d
        d = defaultdict(int)
        # 将当前对象的分区表示转换为字典形式，并更新到字典 d 中
        d.update(self.as_dict())
        # 获取当前分区的键列表 key
        key = self._keys
        # 取 key 中最后一个元素的值为 a
        a = key[-1]
        # 如果 a 等于 self.integer，将 d 清空，并设置键为 1 的值为 self.integer
        if a == self.integer:
            d.clear()
            d[1] = self.integer
        # 否则如果 a 等于 1
        elif a == 1:
            # 如果 d 中 a 的值大于 1
            if d[a] > 1:
                # 将 d 中 a+1 的值加 1，将 d 中 a 的值减 2
                d[a + 1] += 1
                d[a] -= 2
            else:
                # 否则取 key 中倒数第二个元素为 b
                b = key[-2]
                # 将 d 中 b+1 的值加 1，将 d 中 1 的值置为 (d 中 b 的值减 1) 乘以 b
                d[b + 1] += 1
                d[1] = (d[b] - 1) * b
                d[b] = 0
        else:
            # 否则如果 d 中 a 的值大于 1
            if d[a] > 1:
                # 如果 key 的长度为 1
                if len(key) == 1:
                    # 将 d 清空，将 d 中 a+1 的值置为 1，将 d 中 1 的值置为 self.integer 减 a-1
                    d.clear()
                    d[a + 1] = 1
                    d[1] = self.integer - a - 1
                else:
                    # 否则取 a+1 为 a1
                    a1 = a + 1
                    # 将 d 中 a1 的值加 1，将 d 中 1 的值置为 d 中 a 的值乘以 a 减 a1
                    d[a1] += 1
                    d[1] = d[a] * a - a1
                    d[a] = 0
            else:
                # 否则取 key 中倒数第二个元素为 b
                b = key[-2]
                # 取 b+1 为 b1
                b1 = b + 1
                # 将 d 中 b1 的值加 1，计算 need 为 d 中 b 的值乘以 b 加上 d 中 a 的值乘以 a 减 b1
                need = d[b] * b + d[a] * a - b1
                d[a] = d[b] = 0
                # 将 d 中 1 的值置为 need
                d[1] = need
        # 返回一个新的 IntegerPartition 对象，表示计算得到的分区
        return IntegerPartition(self.integer, d)
    def as_dict(self):
        """
        Return the partition as a dictionary whose keys are the
        partition integers and the values are the multiplicity of that
        integer.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> IntegerPartition([1]*3 + [2] + [3]*4).as_dict()
        {1: 3, 2: 1, 3: 4}
        """
        # 如果尚未计算过分区的字典表示
        if self._dict is None:
            # 使用 `group` 函数对分区进行分组处理
            groups = group(self.partition, multiple=False)
            # 提取分区中的唯一整数作为键
            self._keys = [g[0] for g in groups]
            # 构建分区的字典表示
            self._dict = dict(groups)
        # 返回分区的字典表示
        return self._dict

    @property
    def conjugate(self):
        """
        Computes the conjugate partition of itself.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([6, 3, 3, 2, 1])
        >>> a.conjugate
        [5, 4, 3, 1, 1, 1]
        """
        # 初始化计算所需的变量
        j = 1
        temp_arr = list(self.partition) + [0]
        k = temp_arr[0]
        b = [0]*k
        # 根据分区计算其共轭分区
        while k > 0:
            while k > temp_arr[j]:
                b[k - 1] = j
                k -= 1
            j += 1
        return b

    def __lt__(self, other):
        """
        Return True if self is less than other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([3, 1])
        >>> a < a
        False
        >>> b = a.next_lex()
        >>> a < b
        True
        >>> a == b
        False
        """
        # 按照从大到小的顺序比较分区，返回比较结果
        return list(reversed(self.partition)) < list(reversed(other.partition))

    def __le__(self, other):
        """
        Return True if self is less than or equal to other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([4])
        >>> a <= a
        True
        """
        # 按照从大到小的顺序比较分区，返回比较结果
        return list(reversed(self.partition)) <= list(reversed(other.partition))

    def as_ferrers(self, char='#'):
        """
        Prints the ferrer diagram of a partition.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> print(IntegerPartition([1, 1, 5]).as_ferrers())
        #####
        #
        #
        """
        # 打印分区的费雷图形式，使用指定字符
        return "\n".join([char*i for i in self.partition])

    def __str__(self):
        # 返回分区的字符串表示
        return str(list(self.partition))
# 导入 as_int 函数，用于确保 n 是整数
from sympy.core.numbers import as_int
# 导入 _randint 函数，用于生成随机整数
from sympy.core.random import _randint

# 生成随机整数分区，使其总和为 n，并以逆序排序的整数列表形式返回
def random_integer_partition(n, seed=None):
    n = as_int(n)  # 确保 n 是整数
    if n < 1:
        raise ValueError('n must be a positive integer')

    randint = _randint(seed)  # 使用 seed 生成随机数函数

    partition = []
    while (n > 0):
        k = randint(1, n)  # 随机生成一个不超过 n 的整数 k
        mult = randint(1, n//k)  # 随机生成一个不超过 n//k 的整数 mult
        partition.append((k, mult))  # 将 k 和 mult 组成元组追加到 partition 中
        n -= k * mult  # 更新 n 的值

    partition.sort(reverse=True)  # 将 partition 按第一个元素逆序排序
    partition = flatten([[k]*m for k, m in partition])  # 将 partition 扁平化成整数列表
    return partition


# 计算 m + 1 个广义不受限制生长字符串，并以矩阵的行形式返回
def RGS_generalized(m):
    from sympy import zeros, Matrix
    d = zeros(m + 1)  # 创建一个大小为 (m+1) x (m+1) 的零矩阵

    # 填充矩阵 d
    for i in range(m + 1):
        d[0, i] = 1

    for i in range(1, m + 1):
        for j in range(m):
            if j <= m - i:
                d[i, j] = j * d[i - 1, j] + d[i - 1, j + 1]
            else:
                d[i, j] = 0

    return d


# 计算给定超集大小 m 的受限生长字符串的总数
def RGS_enum(m):
    if m < 1:
        return 0
    elif m == 1:
        return 1
    else:
        return bell(m)  # 调用 bell 函数计算受限生长字符串的数目


# 给定排名和超集大小 m，返回未排名的受限生长字符串
def RGS_unrank(rank, m):
    if m < 1:
        raise ValueError("The superset size must be >= 1")
    # 检查排名是否有效，如果无效则抛出 ValueError 异常
    if rank < 0 or RGS_enum(m) <= rank:
        raise ValueError("Invalid arguments")
    
    # 初始化列表 L，将其长度设置为 m+1，并且每个元素初始化为 1
    L = [1] * (m + 1)
    # 初始化变量 j 为 1
    j = 1
    # 调用 RGS_generalized 函数，返回结果赋值给变量 D
    D = RGS_generalized(m)
    
    # 遍历 i 从 2 到 m+1（不包含 m+1）
    for i in range(2, m + 1):
        # 从 D 中获取值 v，并且 D 是一个二维数组
        v = D[m - i, j]
        # 计算 cr 的值，等于 j*v
        cr = j * v
        # 如果 cr 小于等于 rank
        if cr <= rank:
            # 将 L[i] 设置为 j+1
            L[i] = j + 1
            # 将 rank 减去 cr
            rank -= cr
            # j 值加 1
            j += 1
        else:
            # 否则将 L[i] 设置为 int(rank / v + 1)
            L[i] = int(rank / v + 1)
            # rank 取余 v 的值赋给 rank
            rank %= v
    
    # 返回列表 L 中从第二个元素开始到末尾的所有元素减去 1 的结果
    return [x - 1 for x in L[1:]]
# 计算给定受限增长字符串的排名（即将其映射为整数）。
def RGS_rank(rgs):
    """
    Computes the rank of a restricted growth string.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_rank, RGS_unrank
    >>> RGS_rank([0, 1, 2, 1, 3])
    42
    >>> RGS_rank(RGS_unrank(4, 7))
    4
    """
    # 获取受限增长字符串的长度
    rgs_size = len(rgs)
    # 初始排名设为0
    rank = 0
    # 计算受限增长字符串的广义版本
    D = RGS_generalized(rgs_size)
    # 遍历受限增长字符串中的每一个位置（从第二个位置开始）
    for i in range(1, rgs_size):
        # 计算剩余字符串长度
        n = len(rgs[(i + 1):])
        # 计算当前位置之前的最大值
        m = max(rgs[0:i])
        # 根据受限增长字符串的广义版本计算排名贡献
        rank += D[n, m + 1] * rgs[i]
    # 返回计算得到的排名
    return rank
```