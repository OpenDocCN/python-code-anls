# `numpy-ml\numpy_ml\utils\data_structures.py`

```
# 导入 heapq 模块，用于实现优先队列
import heapq
# 从 copy 模块中导入 copy 函数
from copy import copy
# 从 collections 模块中导入 Hashable 类
from collections import Hashable
# 导入 numpy 模块，并使用 np 别名
import numpy as np
# 从当前目录下的 distance_metrics 模块中导入 euclidean 函数
from .distance_metrics import euclidean

#######################################################################
#                           Priority Queue                            #
#######################################################################

# 定义优先队列节点类
class PQNode(object):
    def __init__(self, key, val, priority, entry_id, **kwargs):
        """A generic node object for holding entries in :class:`PriorityQueue`"""
        # 初始化节点对象的属性
        self.key = key
        self.val = val
        self.entry_id = entry_id
        self.priority = priority

    def __repr__(self):
        # 返回节点对象的字符串表示
        fstr = "PQNode(key={}, val={}, priority={}, entry_id={})"
        return fstr.format(self.key, self.val, self.priority, self.entry_id)

    def to_dict(self):
        """Return a dictionary representation of the node's contents"""
        # 返回节点对象的字典表示
        d = self.__dict__
        d["id"] = "PQNode"
        return d

    def __gt__(self, other):
        # 实现大于比较操作符
        if not isinstance(other, PQNode):
            return -1
        if self.priority == other.priority:
            return self.entry_id > other.entry_id
        return self.priority > other.priority

    def __ge__(self, other):
        # 实现大于等于比较操作符
        if not isinstance(other, PQNode):
            return -1
        return self.priority >= other.priority

    def __lt__(self, other):
        # 实现小于比较操作符
        if not isinstance(other, PQNode):
            return -1
        if self.priority == other.priority:
            return self.entry_id < other.entry_id
        return self.priority < other.priority

    def __le__(self, other):
        # 实现小于等于比较操作符
        if not isinstance(other, PQNode):
            return -1
        return self.priority <= other.priority

# 定义优先队列类
class PriorityQueue:
    # 初始化优先队列对象
    def __init__(self, capacity, heap_order="max"):
        """
        A priority queue implementation using a binary heap.

        Notes
        -----
        A priority queue is a data structure useful for storing the top
        `capacity` largest or smallest elements in a collection of values. As a
        result of using a binary heap, ``PriorityQueue`` offers `O(log N)`
        :meth:`push` and :meth:`pop` operations.

        Parameters
        ----------
        capacity: int
            The maximum number of items that can be held in the queue.
        heap_order: {"max", "min"}
            Whether the priority queue should retain the items with the
            `capacity` smallest (`heap_order` = 'min') or `capacity` largest
            (`heap_order` = 'max') priorities.
        """
        # 检查堆序是否合法
        assert heap_order in ["max", "min"], "heap_order must be either 'max' or 'min'"
        # 初始化容量和堆序
        self.capacity = capacity
        self.heap_order = heap_order

        # 初始化优先队列列表、计数器和条目计数器
        self._pq = []
        self._count = 0
        self._entry_counter = 0

    # 返回优先队列的字符串表示
    def __repr__(self):
        fstr = "PriorityQueue(capacity={}, heap_order={}) with {} items"
        return fstr.format(self.capacity, self.heap_order, self._count)

    # 返回优先队列中元素的数量
    def __len__(self):
        return self._count

    # 返回优先队列的迭代器
    def __iter__(self):
        return iter(self._pq)
    def push(self, key, priority, val=None):
        """
        Add a new (key, value) pair with priority `priority` to the queue.

        Notes
        -----
        If the queue is at capacity and `priority` exceeds the priority of the
        item with the largest/smallest priority currently in the queue, replace
        the current queue item with (`key`, `val`).

        Parameters
        ----------
        key : hashable object
            The key to insert into the queue.
        priority : comparable
            The priority for the `key`, `val` pair.
        val : object
            The value associated with `key`. Default is None.
        """
        # 如果队列是最大堆，则将优先级取负数
        if self.heap_order == "max":
            priority = -1 * priority

        # 创建一个新的 PQNode 对象
        item = PQNode(key=key, val=val, priority=priority, entry_id=self._entry_counter)
        # 将新节点加入堆中
        heapq.heappush(self._pq, item)

        # 更新计数器
        self._count += 1
        self._entry_counter += 1

        # 当队列超过容量时，弹出队列中的元素
        while self._count > self.capacity:
            self.pop()

    def pop(self):
        """
        Remove the item with the largest/smallest (depending on
        ``self.heap_order``) priority from the queue and return it.

        Notes
        -----
        In contrast to :meth:`peek`, this operation is `O(log N)`.

        Returns
        -------
        item : :class:`PQNode` instance or None
            Item with the largest/smallest priority, depending on
            ``self.heap_order``.
        """
        # 从堆中弹出具有最大/最小优先级的元素，并将其转换为字典
        item = heapq.heappop(self._pq).to_dict()
        # 如果队列是最大堆，则将优先级取负数恢复原值
        if self.heap_order == "max":
            item["priority"] = -1 * item["priority"]
        # 更新计数器
        self._count -= 1
        return item
    # 返回具有最大/最小优先级（取决于self.heap_order）的项目，但不从队列中删除它
    def peek(self):
        """
        Return the item with the largest/smallest (depending on
        ``self.heap_order``) priority *without* removing it from the queue.

        Notes
        -----
        In contrast to :meth:`pop`, this operation is O(1).

        Returns
        -------
        item : :class:`PQNode` instance or None
            Item with the largest/smallest priority, depending on
            ``self.heap_order``.
        """
        # 初始化item为None
        item = None
        # 如果队列中有元素
        if self._count > 0:
            # 复制队列中第一个元素的字典表示
            item = copy(self._pq[0].to_dict())
            # 如果堆的顺序是最大堆
            if self.heap_order == "max":
                # 将item的优先级取反
                item["priority"] = -1 * item["priority"]
        # 返回item
        return item
# 定义 BallTreeNode 类，表示 Ball 树的节点
class BallTreeNode:
    # 初始化方法，设置节点的属性
    def __init__(self, centroid=None, X=None, y=None):
        self.left = None
        self.right = None
        self.radius = None
        self.is_leaf = False

        self.data = X
        self.targets = y
        self.centroid = centroid

    # 重写 __repr__ 方法，返回节点的字符串表示
    def __repr__(self):
        fstr = "BallTreeNode(centroid={}, is_leaf={})"
        return fstr.format(self.centroid, self.is_leaf)

    # 将节点转换为字典形式
    def to_dict(self):
        d = self.__dict__
        d["id"] = "BallTreeNode"
        return d


# 定义 BallTree 类
class BallTree:
    # 初始化 BallTree 数据结构
    def __init__(self, leaf_size=40, metric=None):
        """
        A ball tree data structure.

        Notes
        -----
        A ball tree is a binary tree in which every node defines a
        `D`-dimensional hypersphere ("ball") containing a subset of the points
        to be searched. Each internal node of the tree partitions the data
        points into two disjoint sets which are associated with different
        balls. While the balls themselves may intersect, each point is assigned
        to one or the other ball in the partition according to its distance
        from the ball's center. Each leaf node in the tree defines a ball and
        enumerates all data points inside that ball.

        Parameters
        ----------
        leaf_size : int
            The maximum number of datapoints at each leaf. Default is 40.
        metric : :doc:`Distance metric <numpy_ml.utils.distance_metrics>` or None
            The distance metric to use for computing nearest neighbors. If
            None, use the :func:`~numpy_ml.utils.distance_metrics.euclidean`
            metric. Default is None.

        References
        ----------
        .. [1] Omohundro, S. M. (1989). "Five balltree construction algorithms". *ICSI
           Technical Report TR-89-063*.
        .. [2] Liu, T., Moore, A., & Gray A. (2006). "New algorithms for efficient
           high-dimensional nonparametric classification". *J. Mach. Learn. Res.,
           7*, 1135-1158.
        """
        # 初始化根节点为 None
        self.root = None
        # 设置叶子节点最大数据点数，默认为 40
        self.leaf_size = leaf_size
        # 设置距离度量方式，如果未指定则使用欧氏距离
        self.metric = metric if metric is not None else euclidean
    def fit(self, X, y=None):
        """
        Build a ball tree recursively using the O(M log N) `k`-d construction
        algorithm.

        Notes
        -----
        Recursively divides data into nodes defined by a centroid `C` and radius
        `r` such that each point below the node lies within the hyper-sphere
        defined by `C` and `r`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            An array of `N` examples each with `M` features.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)` or None
            An array of target values / labels associated with the entries in
            `X`. Default is None.
        """
        # 分割数据集，得到中心点、左子集、右子集
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)
        # 创建根节点，设置中心点和半径
        self.root = BallTreeNode(centroid=centroid)
        self.root.radius = np.max([self.metric(centroid, x) for x in X])
        # 递归构建左右子树
        self.root.left = self._build_tree(left_X, left_y)
        self.root.right = self._build_tree(right_X, right_y)

    def _build_tree(self, X, y):
        # 分割数据集，得到中心点、左子集、右子集
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)

        # 如果数据集大小小于等于叶子节点大小阈值，则创建叶子节点
        if X.shape[0] <= self.leaf_size:
            leaf = BallTreeNode(centroid=centroid, X=X, y=y)
            leaf.radius = np.max([self.metric(centroid, x) for x in X])
            leaf.is_leaf = True
            return leaf

        # 创建内部节点，设置中心点和半径
        node = BallTreeNode(centroid=centroid)
        node.radius = np.max([self.metric(centroid, x) for x in X])
        # 递归构建左右子树
        node.left = self._build_tree(left_X, left_y)
        node.right = self._build_tree(right_X, right_y)
        return node
    # 将数据集 X 拆分成两部分，根据最大方差的维度进行拆分
    def _split(self, X, y=None):
        # 找到方差最大的维度
        split_dim = np.argmax(np.var(X, axis=0))

        # 沿着 split_dim 对 X 和 y 进行排序
        sort_ixs = np.argsort(X[:, split_dim])
        X, y = X[sort_ixs], y[sort_ixs] if y is not None else None

        # 在 split_dim 的中位数处划分数据
        med_ix = X.shape[0] // 2
        centroid = X[med_ix]  # , split_dim

        # 在中心点（中位数总是出现在右侧划分）处将数据分成两半
        left_X, left_y = X[:med_ix], y[:med_ix] if y is not None else None
        right_X, right_y = X[med_ix:], y[med_ix:] if y is not None else None
        return centroid, left_X, left_y, right_X, right_y

    # 使用 KNS1 算法找到球树中距离查询向量 x 最近的 k 个邻居
    def nearest_neighbors(self, k, x):
        """
        Find the `k` nearest neighbors in the ball tree to a query vector `x`
        using the KNS1 algorithm.

        Parameters
        ----------
        k : int
            The number of closest points in `X` to return
        x : :py:class:`ndarray <numpy.ndarray>` of shape `(1, M)`
            The query vector.

        Returns
        -------
        nearest : list of :class:`PQNode` s of length `k`
            List of the `k` points in `X` to closest to the query vector. The
            ``key`` attribute of each :class:`PQNode` contains the point itself, the
            ``val`` attribute contains its target, and the ``distance``
            attribute contains its distance to the query vector.
        """
        # 维护一个最大优先队列，优先级为到 x 的距离
        PQ = PriorityQueue(capacity=k, heap_order="max")
        # 调用 _knn 方法找到最近的 k 个邻居
        nearest = self._knn(k, x, PQ, self.root)
        # 计算每个邻居点到查询向量 x 的距离
        for n in nearest:
            n.distance = self.metric(x, n.key)
        return nearest
    # 使用 k 近邻算法查找最近的 k 个点
    def _knn(self, k, x, PQ, root):
        # 定义距离度量函数
        dist = self.metric
        # 计算点 x 到当前节点球体表面的距离
        dist_to_ball = dist(x, root.centroid) - root.radius
        # 计算点 x 到当前优先队列中最远邻居的距离
        dist_to_farthest_neighbor = dist(x, PQ.peek()["key"]) if len(PQ) > 0 else np.inf

        # 如果点 x 到球体表面的距离大于等于到最远邻居的距离，并且优先队列中已经有 k 个点，则直接返回当前优先队列
        if dist_to_ball >= dist_to_farthest_neighbor and len(PQ) == k:
            return PQ
        # 如果当前节点是叶子节点
        if root.is_leaf:
            # 初始化目标值列表
            targets = [None] * len(root.data) if root.targets is None else root.targets
            # 遍历当前节点的数据点和目标值
            for point, target in zip(root.data, targets):
                # 计算点 x 到当前数据点的距离
                dist_to_x = dist(x, point)
                # 如果优先队列中已经有 k 个点，并且点 x 到当前数据点的距离小于到最远邻居的距离，则将当前数据点加入优先队列
                if len(PQ) == k and dist_to_x < dist_to_farthest_neighbor:
                    PQ.push(key=point, val=target, priority=dist_to_x)
                else:
                    PQ.push(key=point, val=target, priority=dist_to_x)
        else:
            # 判断点 x 更接近左子节点还是右子节点
            l_closest = dist(x, root.left.centroid) < dist(x, root.right.centroid)
            # 递归调用_knn函数，继续在左子树中查找最近的 k 个点
            PQ = self._knn(k, x, PQ, root.left if l_closest else root.right)
            # 递归调用_knn函数，继续在右子树中查找最近的 k 个点
            PQ = self._knn(k, x, PQ, root.right if l_closest else root.left)
        # 返回最终的优先队列
        return PQ
#######################################################################
#                         Multinomial Sampler                         #
#######################################################################

# 定义一个离散采样器类
class DiscreteSampler:
    # 从概率分布`probs`中生成随机抽样整数，范围在[0, N)之间
    def __call__(self, n_samples=1):
        """
        Generate random draws from the `probs` distribution over integers in
        [0, N).

        Parameters
        ----------
        n_samples: int
            The number of samples to generate. Default is 1.

        Returns
        -------
        sample : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples,)`
            A collection of draws from the distribution defined by `probs`.
            Each sample is an int in the range `[0, N)`.
        """
        return self.sample(n_samples)

    # 从概率分布`probs`中生成随机抽样整数，范围在[0, N)之间
    def sample(self, n_samples=1):
        """
        Generate random draws from the `probs` distribution over integers in
        [0, N).

        Parameters
        ----------
        n_samples: int
            The number of samples to generate. Default is 1.

        Returns
        -------
        sample : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples,)`
            A collection of draws from the distribution defined by `probs`.
            Each sample is an int in the range `[0, N)`.
        """
        # 生成随机整数索引
        ixs = np.random.randint(0, self.N, n_samples)
        # 计算概率值
        p = np.exp(self.prob_table[ixs]) if self.log else self.prob_table[ixs]
        # 生成二项分布随机数
        flips = np.random.binomial(1, p)
        # 根据二项分布结果生成抽样
        samples = [ix if f else self.alias_table[ix] for ix, f in zip(ixs, flips)]

        # 使用递归拒绝采样来进行无重复抽样
        if not self.with_replacement:
            unique = list(set(samples))
            while len(samples) != len(unique):
                n_new = len(samples) - len(unique)
                samples = unique + self.sample(n_new).tolist()
                unique = list(set(samples))

        return np.array(samples, dtype=int)
# 定义一个名为 Dict 的字典子类
class Dict(dict):
    # 初始化方法，接受一个编码器参数
    def __init__(self, encoder=None):
        """
        A dictionary subclass which returns the key value if it is not in the
        dict.

        Parameters
        ----------
        encoder : function or None
            A function which is applied to a key before adding / retrieving it
            from the dictionary. If None, the function defaults to the
            identity. Default is None.
        """
        # 调用父类的初始化方法
        super(Dict, self).__init__()
        # 设置编码器和最大 ID
        self._encoder = encoder
        self._id_max = 0

    # 重写 __setitem__ 方法
    def __setitem__(self, key, value):
        # 如果存在编码器，则对键进行编码
        if self._encoder is not None:
            key = self._encoder(key)
        # 如果键不可哈希，则转换为元组
        elif not isinstance(key, Hashable):
            key = tuple(key)
        # 调用父类的 __setitem__ 方法
        super(Dict, self).__setitem__(key, value)

    # 编码键的方法
    def _encode_key(self, key):
        # 获取父类的实例
        D = super(Dict, self)
        # 对键进行编码
        enc_key = self._encoder(key)
        # 如果编码后的键存在于字典中，则返回对应的值，否则添加新键值对
        if D.__contains__(enc_key):
            val = D.__getitem__(enc_key)
        else:
            val = self._id_max
            D.__setitem__(enc_key, val)
            self._id_max += 1
        return val

    # 重写 __getitem__ 方法
    def __getitem__(self, key):
        # 深拷贝键
        self._key = copy.deepcopy(key)
        # 如果存在编码器，则返回编码后的键对应的值
        if self._encoder is not None:
            return self._encode_key(key)
        # 如果键不可哈希，则转换为元组
        elif not isinstance(key, Hashable):
            key = tuple(key)
        # 调用父类的 __getitem__ 方法
        return super(Dict, self).__getitem__(key)

    # 处理缺失键的方法
    def __missing__(self, key):
        return self._key
```