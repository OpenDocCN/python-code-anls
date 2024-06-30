# `D:\src\scipysrc\sympy\sympy\combinatorics\prufer.py`

```
    def rank(self):
        """Returns the rank of the Prufer object.

        The rank is defined as the sum of the elements in the Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).rank
        13
        >>> Prufer([1, 0, 0]).rank
        1

        """
        if self._rank is None:
            self._rank = sum(self.prufer_repr)
        return self._rank



    def __new__(cls, arg):
        """
        Creates a new instance of Prufer from a given argument.

        Parameters
        ==========

        arg : list or Tuple
            If arg is a list of lists or tuples of length 2,
            it will be interpreted as an adjacency list and used to construct
            a Prufer object. If arg is a list or Tuple of numbers,
            it will be interpreted as a Prufer sequence and used to construct
            a Prufer object.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]])
        Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]])
        >>> Prufer([1, 0, 0])
        Prufer([[1, 2], [0, 1], [0, 3], [0, 4]])

        """
        obj = Basic.__new__(cls)
        obj._rank = None
        obj._prufer_repr = None
        obj._tree_repr = None
        obj._nodes = None

        if isinstance(arg, (list, Tuple)):
            if all(isinstance(item, (list, Tuple)) and len(item) == 2 for item in arg):
                # If arg is an adjacency list, store it as tree representation
                obj._tree_repr = arg
                obj._nodes = max(flatten(arg)) + 1
            elif all(isinstance(item, (int, Integer)) for item in arg):
                # If arg is a Prufer sequence, store it as Prufer representation
                obj._prufer_repr = arg
                obj._nodes = len(arg) + 1

        return obj



    def to_prufer(self, tree, num_nodes):
        """Converts a tree representation to a Prufer sequence.

        Parameters
        ==========

        tree : list
            The tree representation as an adjacency list.
        num_nodes : int
            The total number of nodes in the tree.

        Returns
        =======

        list
            The Prufer sequence corresponding to the tree representation.

        """
        # Implementation of the algorithm to convert tree representation to Prufer sequence
        # Omitted for brevity

    def to_tree(self, prufer):
        """Converts a Prufer sequence to a tree representation.

        Parameters
        ==========

        prufer : list
            The Prufer sequence.

        Returns
        =======

        list
            The tree representation as an adjacency list.

        """
        # Implementation of the algorithm to convert Prufer sequence to tree representation
        # Omitted for brevity



    def __hash__(self):
        """Returns the hash value of the Prufer object."""
        # Hash value is computed based on the hash of the Prufer sequence
        return hash(tuple(self.prufer_repr))

    def __eq__(self, other):
        """Checks if two Prufer objects are equal."""
        # Equality check based on Prufer sequence comparison
        return isinstance(other, Prufer) and self.prufer_repr == other.prufer_repr
    def rank(self):
        """Returns the rank of the Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> p = Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]])
        >>> p.rank
        778
        >>> p.next(1).rank
        779
        >>> p.prev().rank
        777

        See Also
        ========

        prufer_rank, next, prev, size

        """
        # 如果未计算过排名，调用 prufer_rank() 方法计算并缓存排名
        if self._rank is None:
            self._rank = self.prufer_rank()
        # 返回已计算的排名
        return self._rank

    @property
    def size(self):
        """Return the number of possible trees of this Prufer object.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([0]*4).size == Prufer([6]*4).size == 1296
        True

        See Also
        ========

        prufer_rank, rank, next, prev

        """
        # 计算当前 Prufer 对象的可能树的数量
        return self.prev(self.rank).prev().rank + 1

    @staticmethod
    def to_prufer(tree, n):
        """Return the Prufer sequence for a tree given as a list of edges where
        ``n`` is the number of nodes in the tree.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> a.prufer_repr
        [0, 0]
        >>> Prufer.to_prufer([[0, 1], [0, 2], [0, 3]], 4)
        [0, 0]

        See Also
        ========
        prufer_repr: returns Prufer sequence of a Prufer object.

        """
        # 创建一个默认字典和空列表
        d = defaultdict(int)
        L = []
        # 遍历树中的每条边
        for edge in tree:
            # 增加涉及节点在度数列表中的计数
            d[edge[0]] += 1
            d[edge[1]] += 1
        # 构造 Prufer 序列
        for i in range(n - 2):
            # 找到度数为 1 的最小叶子节点
            for x in range(n):
                if d[x] == 1:
                    break
            # 找到与之连接的节点
            y = None
            for edge in tree:
                if x == edge[0]:
                    y = edge[1]
                elif x == edge[1]:
                    y = edge[0]
                if y is not None:
                    break
            # 记录并更新 Prufer 序列和度数列表
            L.append(y)
            for j in (x, y):
                d[j] -= 1
                if not d[j]:
                    d.pop(j)
            tree.remove(edge)
        return L
    def to_tree(prufer):
        """Return the tree (as a list of edges) of the given Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([0, 2], 4)
        >>> a.tree_repr
        [[0, 1], [0, 2], [2, 3]]
        >>> Prufer.to_tree([0, 2])
        [[0, 1], [0, 2], [2, 3]]

        References
        ==========

        .. [1] https://hamberg.no/erlend/posts/2010-11-06-prufer-sequence-compact-tree-representation.html

        See Also
        ========
        tree_repr: returns tree representation of a Prufer object.

        """
        tree = []  # 初始化树的边列表
        last = []  # 初始化最后一个节点
        n = len(prufer) + 2  # 计算节点数目
        d = defaultdict(lambda: 1)  # 创建默认字典，用于统计每个节点的度数
        for p in prufer:
            d[p] += 1  # 统计每个节点在Prufer序列中出现的次数
        for i in prufer:
            for j in range(n):
                if d[j] == 1:  # 找到度数为1的最小叶子节点
                    break
            # 将(i, j)作为新的边添加到树中，并更新字典中对应节点的度数
            d[i] -= 1
            d[j] -= 1
            tree.append(sorted([i, j]))  # 将边按顺序加入树中
        last = [i for i in range(n) if d[i] == 1] or [0, 1]  # 找到最后剩余的节点，添加到树中作为最后一个边
        tree.append(last)  # 将最后一个边添加到树中

        return tree  # 返回构建好的树的边列表
    # 定义一个函数edges，接受多个runs作为参数，返回一个连接整数标记树节点的边列表及节点数
    def edges(*runs):
        """Return a list of edges and the number of nodes from the given runs
        that connect nodes in an integer-labelled tree.

        All node numbers will be shifted so that the minimum node is 0. It is
        not a problem if edges are repeated in the runs; only unique edges are
        returned. There is no assumption made about what the range of the node
        labels should be, but all nodes from the smallest through the largest
        must be present.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer.edges([1, 2, 3], [2, 4, 5]) # a T
        ([[0, 1], [1, 2], [1, 3], [3, 4]], 5)

        Duplicate edges are removed:

        >>> Prufer.edges([0, 1, 2, 3], [1, 4, 5], [1, 4, 6]) # a K
        ([[0, 1], [1, 2], [1, 4], [2, 3], [4, 5], [4, 6]], 7)

        """
        # 使用集合e存储唯一的边
        e = set()
        # 初始化节点数的最小值为第一个runs的第一个元素
        nmin = runs[0][0]
        # 遍历每个runs
        for r in runs:
            # 遍历当前run中的节点对
            for i in range(len(r) - 1):
                # 获取当前节点对
                a, b = r[i: i + 2]
                # 确保a小于等于b
                if b < a:
                    a, b = b, a
                # 将节点对(a, b)添加到集合e中
                e.add((a, b))
        # 初始化结果列表rv
        rv = []
        # 初始化集合got，用于跟踪已经添加的节点
        got = set()
        # 初始化节点数的最小值和最大值为None
        nmin = nmax = None
        # 遍历集合e中的每个边ei
        for ei in e:
            # 更新got集合，将ei中的节点加入到got中
            got.update(ei)
            # 更新节点数的最小值和最大值
            nmin = min(ei[0], nmin) if nmin is not None else ei[0]
            nmax = max(ei[1], nmax) if nmax is not None else ei[1]
            # 将边ei添加到结果列表rv中，转换为列表形式
            rv.append(list(ei))
        # 计算缺失的节点
        missing = set(range(nmin, nmax + 1)) - got
        # 如果存在缺失节点
        if missing:
            # 将缺失节点列表化
            missing = [i + nmin for i in missing]
            # 根据缺失节点数目生成错误信息msg
            if len(missing) == 1:
                msg = 'Node %s is missing.' % missing.pop()
            else:
                msg = 'Nodes %s are missing.' % sorted(missing)
            # 抛出数值错误异常，包含错误信息msg
            raise ValueError(msg)
        # 如果节点数的最小值不为0
        if nmin != 0:
            # 对结果列表rv中的每个边ei，将其节点标签减去nmin
            for i, ei in enumerate(rv):
                rv[i] = [n - nmin for n in ei]
            # 更新节点数的最大值
            nmax -= nmin
        # 返回按节点标签排序的结果列表rv及节点数nmax加1
        return sorted(rv), nmax + 1

    def prufer_rank(self):
        """Computes the rank of a Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> a.prufer_rank()
        0

        See Also
        ========

        rank, next, prev, size

        """
        # 初始化排名r为0，乘积p为1
        r = 0
        p = 1
        # 从self.nodes-3到0逆序遍历Prufer表示
        for i in range(self.nodes - 3, -1, -1):
            # 计算当前位置i的贡献，加到排名r中
            r += p*self.prufer_repr[i]
            # 更新乘积p
            p *= self.nodes
        # 返回计算得到的排名r
        return r

    @classmethod
    def unrank(self, rank, n):
        """Finds the unranked Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer.unrank(0, 4)
        Prufer([0, 0])

        """
        # 将n和rank转换为整数类型
        n, rank = as_int(n), as_int(rank)
        # 使用defaultdict初始化L
        L = defaultdict(int)
        # 从n-3到0逆序遍历
        for i in range(n - 3, -1, -1):
            # 计算L[i]的值
            L[i] = rank % n
            # 更新rank的值
            rank = (rank - L[i]) // n
        # 返回Prufer序列，由L中的值组成的列表
        return Prufer([L[i] for i in range(len(L))])
    def __new__(cls, *args, **kw_args):
        """创建 Prufer 对象的构造函数。

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer

        可以通过边的列表构建 Prufer 对象：

        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> a.prufer_repr
        [0, 0]

        如果给出节点数，不会检查节点的存在性；假设节点从 0 到 n-1 都存在：

        >>> Prufer([[0, 1], [0, 2], [0, 3]], 4)
        Prufer([[0, 1], [0, 2], [0, 3]], 4)

        可以通过 Prufer 序列构建 Prufer 对象：

        >>> b = Prufer([1, 3])
        >>> b.tree_repr
        [[0, 1], [1, 3], [2, 3]]

        """
        # 将第一个参数转换为 Array 类型，如果为空则转换为 Tuple 类型
        arg0 = Array(args[0]) if args[0] else Tuple()
        # 对参数进行符号化处理
        args = (arg0,) + tuple(_sympify(arg) for arg in args[1:])
        # 调用基类 Basic 的构造函数生成对象
        ret_obj = Basic.__new__(cls, *args, **kw_args)
        # 将参数重新赋值为列表形式
        args = [list(args[0])]
        # 如果参数不为空且可迭代
        if args[0] and iterable(args[0][0]):
            # 如果列表为空则引发 ValueError
            if not args[0][0]:
                raise ValueError(
                    'Prufer expects at least one edge in the tree.')
            # 如果参数个数大于 1，则取第二个参数作为节点数
            if len(args) > 1:
                nnodes = args[1]
            else:
                # 将所有节点放入集合并计算节点数
                nodes = set(flatten(args[0]))
                nnodes = max(nodes) + 1
                # 如果计算出的节点数与集合中节点数量不一致，则报错
                if nnodes != len(nodes):
                    missing = set(range(nnodes)) - nodes
                    if len(missing) == 1:
                        msg = 'Node %s is missing.' % missing.pop()
                    else:
                        msg = 'Nodes %s are missing.' % sorted(missing)
                    raise ValueError(msg)
            # 将对象的 _tree_repr 属性设置为参数的列表形式
            ret_obj._tree_repr = [list(i) for i in args[0]]
            # 将对象的 _nodes 属性设置为节点数
            ret_obj._nodes = nnodes
        else:
            # 将对象的 _prufer_repr 属性设置为第一个参数
            ret_obj._prufer_repr = args[0]
            # 将对象的 _nodes 属性设置为 Prufer 序列长度加 2
            ret_obj._nodes = len(ret_obj._prufer_repr) + 2
        # 返回创建的对象
        return ret_obj

    def next(self, delta=1):
        """生成当前 Prufer 序列之后 delta 个位置的 Prufer 序列。

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> b = a.next(1) # == a.next()
        >>> b.tree_repr
        [[0, 2], [0, 1], [1, 3]]
        >>> b.rank
        1

        See Also
        ========

        prufer_rank, rank, prev, size

        """
        # 返回根据当前排名加上 delta 的节点数生成的 Prufer 序列
        return Prufer.unrank(self.rank + delta, self.nodes)

    def prev(self, delta=1):
        """生成当前 Prufer 序列之前 delta 个位置的 Prufer 序列。

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [1, 2], [2, 3], [1, 4]])
        >>> a.rank
        36
        >>> b = a.prev()
        >>> b
        Prufer([1, 2, 0])
        >>> b.rank
        35

        See Also
        ========

        prufer_rank, rank, next, size

        """
        # 返回根据当前排名减去 delta 的节点数生成的 Prufer 序列
        return Prufer.unrank(self.rank - delta, self.nodes)
```