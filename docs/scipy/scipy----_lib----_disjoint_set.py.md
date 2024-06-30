# `D:\src\scipysrc\scipy\scipy\_lib\_disjoint_set.py`

```
"""
Disjoint set data structure
"""


class DisjointSet:
    """ Disjoint set data structure for incremental connectivity queries.

    .. versionadded:: 1.6.0

    Attributes
    ----------
    n_subsets : int
        The number of subsets.

    Methods
    -------
    add
    merge
    connected
    subset
    subset_size
    subsets
    __getitem__

    Notes
    -----
    This class implements the disjoint set [1]_, also known as the *union-find*
    or *merge-find* data structure. The *find* operation (implemented in
    `__getitem__`) implements the *path halving* variant. The *merge* method
    implements the *merge by size* variant.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    Examples
    --------
    >>> from scipy.cluster.hierarchy import DisjointSet

    Initialize a disjoint set:

    >>> disjoint_set = DisjointSet([1, 2, 3, 'a', 'b'])

    Merge some subsets:

    >>> disjoint_set.merge(1, 2)
    True
    >>> disjoint_set.merge(3, 'a')
    True
    >>> disjoint_set.merge('a', 'b')
    True
    >>> disjoint_set.merge('b', 'b')
    False

    Find root elements:

    >>> disjoint_set[2]
    1
    >>> disjoint_set['b']
    3

    Test connectivity:

    >>> disjoint_set.connected(1, 2)
    True
    >>> disjoint_set.connected(1, 'b')
    False

    List elements in disjoint set:

    >>> list(disjoint_set)
    [1, 2, 3, 'a', 'b']

    Get the subset containing 'a':

    >>> disjoint_set.subset('a')
    {'a', 3, 'b'}

    Get the size of the subset containing 'a' (without actually instantiating
    the subset):

    >>> disjoint_set.subset_size('a')
    3

    Get all subsets in the disjoint set:

    >>> disjoint_set.subsets()
    [{1, 2}, {'a', 3, 'b'}]
    """
    def __init__(self, elements=None):
        # Initialize the DisjointSet with default values and structures
        self.n_subsets = 0  # Number of subsets initially set to 0
        self._sizes = {}    # Dictionary to store sizes of subsets
        self._parents = {}  # Dictionary to store parent elements
        # _nbrs is a circular linked list which links connected elements.
        self._nbrs = {}     # Dictionary for circular linked list structure
        # _indices tracks the element insertion order in `__iter__`.
        self._indices = {}  # Dictionary to track insertion order of elements

        # If initial elements are provided, add each element to the DisjointSet
        if elements is not None:
            for x in elements:
                self.add(x)

    def __iter__(self):
        """Returns an iterator of the elements in the disjoint set.

        Elements are ordered by insertion order.
        """
        return iter(self._indices)

    def __len__(self):
        # Return the number of elements in the DisjointSet
        return len(self._indices)

    def __contains__(self, x):
        # Check if element x is in the DisjointSet
        return x in self._indices
    def __getitem__(self, x):
        """Find the root element of `x`.

        Parameters
        ----------
        x : hashable object
            Input element.

        Returns
        -------
        root : hashable object
            Root element of `x`.
        """
        # 如果 x 不在索引中，则抛出 KeyError 异常
        if x not in self._indices:
            raise KeyError(x)

        # 使用路径压缩（path halving）的方式找到 x 的根节点
        parents = self._parents
        while self._indices[x] != self._indices[parents[x]]:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def add(self, x):
        """Add element `x` to disjoint set
        """
        # 如果 x 已经存在于索引中，则直接返回
        if x in self._indices:
            return

        # 初始化 x 的大小为 1，父节点指向自己，邻居节点也指向自己，并更新索引
        self._sizes[x] = 1
        self._parents[x] = x
        self._nbrs[x] = x
        self._indices[x] = len(self._indices)
        self.n_subsets += 1

    def merge(self, x, y):
        """Merge the subsets of `x` and `y`.

        The smaller subset (the child) is merged into the larger subset (the
        parent). If the subsets are of equal size, the root element which was
        first inserted into the disjoint set is selected as the parent.

        Parameters
        ----------
        x, y : hashable object
            Elements to merge.

        Returns
        -------
        merged : bool
            True if `x` and `y` were in disjoint sets, False otherwise.
        """
        # 找到 x 和 y 的根节点
        xr = self[x]
        yr = self[y]

        # 如果 x 和 y 已经在同一个子集中，则返回 False
        if self._indices[xr] == self._indices[yr]:
            return False

        sizes = self._sizes
        # 将较小的子集合并到较大的子集中，或者如果大小相等，则选择先插入的根节点作为父节点
        if (sizes[xr], self._indices[yr]) < (sizes[yr], self._indices[xr]):
            xr, yr = yr, xr
        self._parents[yr] = xr
        self._sizes[xr] += self._sizes[yr]
        self._nbrs[xr], self._nbrs[yr] = self._nbrs[yr], self._nbrs[xr]
        self.n_subsets -= 1
        return True

    def connected(self, x, y):
        """Test whether `x` and `y` are in the same subset.

        Parameters
        ----------
        x, y : hashable object
            Elements to test.

        Returns
        -------
        result : bool
            True if `x` and `y` are in the same set, False otherwise.
        """
        # 检查 x 和 y 是否在同一个子集中
        return self._indices[self[x]] == self._indices[self[y]]

    def subset(self, x):
        """Get the subset containing `x`.

        Parameters
        ----------
        x : hashable object
            Input element.

        Returns
        -------
        result : set
            Subset containing `x`.
        """
        # 如果 x 不在索引中，则抛出 KeyError 异常
        if x not in self._indices:
            raise KeyError(x)

        # 从 x 开始，找到包含 x 的子集
        result = [x]
        nxt = self._nbrs[x]
        while self._indices[nxt] != self._indices[x]:
            result.append(nxt)
            nxt = self._nbrs[nxt]
        return set(result)
    def subset_size(self, x):
        """Get the size of the subset containing `x`.
        
        Note that this method is faster than ``len(self.subset(x))`` because
        the size is directly read off an internal field, without the need to
        instantiate the full subset.
        
        Parameters
        ----------
        x : hashable object
            Input element.
        
        Returns
        -------
        result : int
            Size of the subset containing `x`.
        """
        # 返回包含元素 x 的子集的大小，直接从内部字段 self._sizes 中读取
        return self._sizes[self[x]]

    def subsets(self):
        """Get all the subsets in the disjoint set.
        
        Returns
        -------
        result : list
            Subsets in the disjoint set.
        """
        # 初始化结果列表
        result = []
        # 初始化已访问过的元素集合
        visited = set()
        # 遍历所有元素 x 在 self 中
        for x in self:
            # 如果元素 x 尚未被访问过
            if x not in visited:
                # 获取包含元素 x 的子集
                xset = self.subset(x)
                # 将子集 xset 中的元素标记为已访问
                visited.update(xset)
                # 将子集 xset 添加到结果列表中
                result.append(xset)
        # 返回所有子集的列表
        return result
```