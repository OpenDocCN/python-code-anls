# `D:\src\scipysrc\sympy\sympy\combinatorics\subsets.py`

```
from itertools import combinations  # 导入 combinations 函数，用于生成组合
from sympy.combinatorics.graycode import GrayCode  # 导入 GrayCode 类，用于生成格雷码序列


class Subset():
    """
    Represents a basic subset object.

    Explanation
    ===========

    We generate subsets using essentially two techniques,
    binary enumeration and lexicographic enumeration.
    The Subset class takes two arguments, the first one
    describes the initial subset to consider and the second
    describes the superset.

    Examples
    ========

    >>> from sympy.combinatorics import Subset
    >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
    >>> a.next_binary().subset
    ['b']
    >>> a.prev_binary().subset
    ['c']
    """

    _rank_binary = None  # 用于存储二进制排列的排名
    _rank_lex = None  # 用于存储词典排序的排名
    _rank_graycode = None  # 用于存储格雷码排序的排名
    _subset = None  # 存储当前子集
    _superset = None  # 存储超集

    def __new__(cls, subset, superset):
        """
        Default constructor.

        It takes the ``subset`` and its ``superset`` as its parameters.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.subset
        ['c', 'd']
        >>> a.superset
        ['a', 'b', 'c', 'd']
        >>> a.size
        2
        """
        if len(subset) > len(superset):
            raise ValueError('Invalid arguments have been provided. The '
                             'superset must be larger than the subset.')
        for elem in subset:
            if elem not in superset:
                raise ValueError('The superset provided is invalid as it does '
                                 'not contain the element {}'.format(elem))
        obj = object.__new__(cls)
        obj._subset = subset  # 初始化 _subset 属性为 subset
        obj._superset = superset  # 初始化 _superset 属性为 superset
        return obj

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        whether both objects are of the class Subset and if the values
        of the subset and superset attributes are the same.
        """
        if not isinstance(other, Subset):
            return NotImplemented
        return self.subset == other.subset and self.superset == other.superset

    def iterate_binary(self, k):
        """
        This is a helper function. It iterates over the
        binary subsets by ``k`` steps. This variable can be
        both positive or negative.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(-2).subset
        ['d']
        >>> a = Subset(['a', 'b', 'c'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(2).subset
        []

        See Also
        ========

        next_binary, prev_binary
        """
        bin_list = Subset.bitlist_from_subset(self.subset, self.superset)  # 调用 bitlist_from_subset 方法生成子集的位列表
        n = (int(''.join(bin_list), 2) + k) % 2**self.superset_size  # 计算下一个二进制子集的编号
        bits = bin(n)[2:].rjust(self.superset_size, '0')  # 将编号转换为二进制字符串，并补齐长度
        return Subset.subset_from_bitlist(self.superset, bits)  # 根据二进制位列表生成子集
    def next_binary(self):
        """
        生成下一个二进制排序的子集。

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_binary().subset
        ['b']
        >>> a = Subset(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_binary().subset
        []

        See Also
        ========

        prev_binary, iterate_binary
        """
        # 调用 iterate_binary 方法生成下一个二进制排序的子集
        return self.iterate_binary(1)

    def prev_binary(self):
        """
        生成前一个二进制排序的子集。

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_binary().subset
        ['a', 'b', 'c', 'd']
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.prev_binary().subset
        ['c']

        See Also
        ========

        next_binary, iterate_binary
        """
        # 调用 iterate_binary 方法生成前一个二进制排序的子集
        return self.iterate_binary(-1)

    def next_lexicographic(self):
        """
        生成下一个字典序排序的子集。

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        ['d']
        >>> a = Subset(['d'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        []

        See Also
        ========

        prev_lexicographic
        """
        # 获取当前子集在超集中的最大索引
        i = self.superset_size - 1
        # 获取当前子集在超集中的索引列表
        indices = Subset.subset_indices(self.subset, self.superset)

        # 根据字典序规则调整索引列表
        if i in indices:
            if i - 1 in indices:
                indices.remove(i - 1)
            else:
                indices.remove(i)
                i = i - 1
                while i >= 0 and i not in indices:
                    i = i - 1
                if i >= 0:
                    indices.remove(i)
                    indices.append(i+1)
        else:
            while i not in indices and i >= 0:
                i = i - 1
            indices.append(i + 1)

        # 根据调整后的索引列表构建新的子集
        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)
    def prev_lexicographic(self):
        """
        Generates the previous lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['d']
        >>> a = Subset(['c','d'], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['c']

        See Also
        ========

        next_lexicographic
        """
        # 初始化索引为超集大小减一
        i = self.superset_size - 1
        # 获取当前子集在超集中的索引列表
        indices = Subset.subset_indices(self.subset, self.superset)

        # 循环直到找到不在当前子集中的最大索引 i
        while i >= 0 and i not in indices:
            i = i - 1

        # 根据 i 的值更新索引列表
        if i == 0 or i - 1 in indices:
            indices.remove(i)
        else:
            if i >= 0:
                indices.remove(i)
                indices.append(i - 1)
            indices.append(self.superset_size - 1)

        # 根据更新后的索引列表构建返回的子集列表
        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        # 返回新的 Subset 对象
        return Subset(ret_set, super_set)

    def iterate_graycode(self, k):
        """
        Helper function used for prev_gray and next_gray.
        It performs ``k`` step overs to get the respective Gray codes.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.iterate_graycode(3).subset
        [1, 4]
        >>> a.iterate_graycode(-2).subset
        [1, 2, 4]

        See Also
        ========

        next_gray, prev_gray
        """
        # 根据当前 Gray 码的排名和步长 k 计算新的未排名 Gray 码
        unranked_code = GrayCode.unrank(self.superset_size,
                                       (self.rank_gray + k) % self.cardinality)
        # 根据未排名 Gray 码生成对应的子集并返回
        return Subset.subset_from_bitlist(self.superset,
                                          unranked_code)

    def next_gray(self):
        """
        Generates the next Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.next_gray().subset
        [1, 3]

        See Also
        ========

        iterate_graycode, prev_gray
        """
        # 调用 iterate_graycode 方法来生成下一个 Gray 码顺序的子集
        return self.iterate_graycode(1)

    def prev_gray(self):
        """
        Generates the previous Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([2, 3, 4], [1, 2, 3, 4, 5])
        >>> a.prev_gray().subset
        [2, 3, 4, 5]

        See Also
        ========

        iterate_graycode, next_gray
        """
        # 调用 iterate_graycode 方法来生成前一个 Gray 码顺序的子集
        return self.iterate_graycode(-1)

    @property
    def rank_binary(self):
        """
        Computes the binary ordered rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a','b','c','d'])
        >>> a.rank_binary
        0
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_binary
        3

        See Also
        ========

        iterate_binary, unrank_binary
        """
        # 如果尚未计算二进制有序的排名
        if self._rank_binary is None:
            # 将子集的位列表转换为二进制字符串，并转换为整数
            self._rank_binary = int("".join(
                Subset.bitlist_from_subset(self.subset,
                                           self.superset)), 2)
        return self._rank_binary

    @property
    def rank_lexicographic(self):
        """
        Computes the lexicographic ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_lexicographic
        14
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_lexicographic
        43
        """
        # 如果尚未计算字典序排名
        if self._rank_lex is None:
            # 定义内部函数 _ranklex 递归计算排名
            def _ranklex(self, subset_index, i, n):
                if subset_index == [] or i > n:
                    return 0
                if i in subset_index:
                    subset_index.remove(i)
                    return 1 + _ranklex(self, subset_index, i + 1, n)
                return 2**(n - i - 1) + _ranklex(self, subset_index, i + 1, n)
            
            # 获取子集相对于超集的索引列表
            indices = Subset.subset_indices(self.subset, self.superset)
            # 计算字典序排名
            self._rank_lex = _ranklex(self, indices, 0, self.superset_size)
        return self._rank_lex

    @property
    def rank_gray(self):
        """
        Computes the Gray code ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c','d'], ['a','b','c','d'])
        >>> a.rank_gray
        2
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_gray
        27

        See Also
        ========

        iterate_graycode, unrank_gray
        """
        # 如果尚未计算格雷码排名
        if self._rank_graycode is None:
            # 获取子集的位列表并计算格雷码排名
            bits = Subset.bitlist_from_subset(self.subset, self.superset)
            self._rank_graycode = GrayCode(len(bits), start=bits).rank
        return self._rank_graycode

    @property
    def subset(self):
        """
        Gets the subset represented by the current instance.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.subset
        ['c', 'd']

        See Also
        ========

        superset, size, superset_size, cardinality
        """
        return self._subset
    def size(self):
        """
        Gets the size of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.size
        2

        See Also
        ========

        subset, superset, superset_size, cardinality
        """
        # 返回当前子集的元素个数（即子集的大小）
        return len(self.subset)

    @property
    def superset(self):
        """
        Gets the superset of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset
        ['a', 'b', 'c', 'd']

        See Also
        ========

        subset, size, superset_size, cardinality
        """
        # 返回当前子集的超集
        return self._superset

    @property
    def superset_size(self):
        """
        Returns the size of the superset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset_size
        4

        See Also
        ========

        subset, superset, size, cardinality
        """
        # 返回当前子集的超集的大小（即超集包含的元素个数）
        return len(self.superset)

    @property
    def cardinality(self):
        """
        Returns the number of all possible subsets.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.cardinality
        16

        See Also
        ========

        subset, superset, size, superset_size
        """
        # 返回当前超集可能的所有子集的数量（即 2 的超集大小次方）
        return 2**(self.superset_size)

    @classmethod
    def subset_from_bitlist(self, super_set, bitlist):
        """
        Gets the subset defined by the bitlist.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.subset_from_bitlist(['a', 'b', 'c', 'd'], '0011').subset
        ['c', 'd']

        See Also
        ========

        bitlist_from_subset
        """
        # 根据给定的比特列表 bitlist 返回对应的子集
        if len(super_set) != len(bitlist):
            raise ValueError("The sizes of the lists are not equal")
        ret_set = []
        for i in range(len(bitlist)):
            if bitlist[i] == '1':
                ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    @classmethod
    def bitlist_from_subset(self, subset, superset):
        """
        Gets the bitlist corresponding to a subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.bitlist_from_subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        '0011'

        See Also
        ========

        subset_from_bitlist
        """
        # 返回给定子集在超集中对应的比特列表表示
        bitlist = ['0'] * len(superset)
        if isinstance(subset, Subset):
            subset = subset.subset
        for i in Subset.subset_indices(subset, superset):
            bitlist[i] = '1'
        return ''.join(bitlist)

    @classmethod
    def unrank_binary(self, rank, superset):
        """
        Gets the binary ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_binary(4, ['a', 'b', 'c', 'd']).subset
        ['b']

        See Also
        ========

        iterate_binary, rank_binary
        """
        # Convert the rank to a binary string representation padded to the length of superset
        bits = bin(rank)[2:].rjust(len(superset), '0')
        # Return the subset of superset based on the binary representation bits
        return Subset.subset_from_bitlist(superset, bits)

    @classmethod
    def unrank_gray(cls, rank, superset):
        """
        Gets the Gray code ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_gray(4, ['a', 'b', 'c']).subset
        ['a', 'b']
        >>> Subset.unrank_gray(0, ['a', 'b', 'c']).subset
        []

        See Also
        ========

        iterate_graycode, rank_gray
        """
        # Obtain the Gray code bitlist for the given rank and superset length
        graycode_bitlist = GrayCode.unrank(len(superset), rank)
        # Return the subset of superset based on the Gray code bitlist
        return Subset.subset_from_bitlist(superset, graycode_bitlist)

    @classmethod
    def subset_indices(cls, subset, superset):
        """Return indices of subset in superset in a list; the list is empty
        if all elements of ``subset`` are not in ``superset``.

        Examples
        ========

            >>> from sympy.combinatorics import Subset
            >>> superset = [1, 3, 2, 5, 4]
            >>> Subset.subset_indices([3, 2, 1], superset)
            [1, 2, 0]
            >>> Subset.subset_indices([1, 6], superset)
            []
            >>> Subset.subset_indices([], superset)
            []

        """
        # Assign shorter variable names for convenience
        a, b = superset, subset
        # Convert subset b into a set for quick lookup
        sb = set(b)
        # Dictionary to store indices of elements from subset b found in superset a
        d = {}
        # Iterate through superset a with index enumeration
        for i, ai in enumerate(a):
            # If the element ai from a is in sb (subset b)
            if ai in sb:
                # Store the index of ai in d dictionary
                d[ai] = i
                # Remove ai from sb, indicating it's found
                sb.remove(ai)
                # If sb is empty (all elements of subset b are found), exit the loop
                if not sb:
                    break
        else:
            # If sb is not empty after the loop, return an empty list (subset not fully found)
            return []
        # Return indices of elements from subset b found in superset a, in the order of subset b
        return [d[bi] for bi in b]
# 定义函数 ksubsets，用于找出给定集合的大小为 k 的所有子集，按照词典顺序排序

def ksubsets(superset, k):
    """
    Finds the subsets of size ``k`` in lexicographic order.

    This uses the itertools generator.

    Examples
    ========

    >>> from sympy.combinatorics.subsets import ksubsets
    >>> list(ksubsets([1, 2, 3], 2))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(ksubsets([1, 2, 3, 4, 5], 2))
    [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), \
    (2, 5), (3, 4), (3, 5), (4, 5)]

    See Also
    ========

    Subset
    """
    使用 itertools 库中的 combinations 函数生成所有长度为 k 的子集
    return combinations(superset, k)
```