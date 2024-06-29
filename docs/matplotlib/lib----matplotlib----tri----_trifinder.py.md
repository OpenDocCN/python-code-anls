# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_trifinder.py`

```py
# 导入NumPy库，用于处理数组和数值计算
import numpy as np

# 导入Matplotlib内部API
from matplotlib import _api
# 导入Triangulation类，用于处理三角剖分
from matplotlib.tri import Triangulation

# TriFinder类：抽象基类，用于查找(x, y)点所在的三角剖分中的三角形
class TriFinder:
    """
    Abstract base class for classes used to find the triangles of a
    Triangulation in which (x, y) points lie.

    Rather than instantiate an object of a class derived from TriFinder, it is
    usually better to use the function `.Triangulation.get_trifinder`.

    Derived classes implement __call__(x, y) where x and y are array-like point
    coordinates of the same shape.
    """

    # 构造方法：初始化TriFinder对象
    def __init__(self, triangulation):
        # 检查triangulation参数是否为Triangulation实例
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        # 设置_triangulation属性为传入的triangulation对象
        self._triangulation = triangulation

    # __call__方法：抽象方法，子类必须实现以返回(x, y)点所在的三角形的索引
    def __call__(self, x, y):
        # 抛出未实现错误，子类需覆盖此方法
        raise NotImplementedError


# TrapezoidMapTriFinder类：使用梯形映射算法实现的TriFinder类
class TrapezoidMapTriFinder(TriFinder):
    """
    `~matplotlib.tri.TriFinder` class implemented using the trapezoid
    map algorithm from the book "Computational Geometry, Algorithms and
    Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
    and O. Schwarzkopf.

    The triangulation must be valid, i.e. it must not have duplicate points,
    triangles formed from colinear points, or overlapping triangles.  The
    algorithm has some tolerance to triangles formed from colinear points, but
    this should not be relied upon.
    """

    # 构造方法：初始化TrapezoidMapTriFinder对象
    def __init__(self, triangulation):
        # 导入_tri模块
        from matplotlib import _tri
        # 调用父类的构造方法
        super().__init__(triangulation)
        # 调用C++实现的梯形映射TriFinder对象
        self._cpp_trifinder = _tri.TrapezoidMapTriFinder(
            triangulation.get_cpp_triangulation())
        # 执行初始化操作
        self._initialize()

    # __call__方法：查找(x, y)点所在的三角形的索引
    def __call__(self, x, y):
        """
        Return an array containing the indices of the triangles in which the
        specified *x*, *y* points lie, or -1 for points that do not lie within
        a triangle.

        *x*, *y* are array-like x and y coordinates of the same shape and any
        number of dimensions.

        Returns integer array with the same shape and *x* and *y*.
        """
        # 将x和y转换为NumPy数组，并指定数据类型为float64
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # 检查x和y的形状是否相同，若不同则抛出值错误
        if x.shape != y.shape:
            raise ValueError("x and y must be array-like with the same shape")

        # 调用C++实现的梯形映射TriFinder对象的find_many方法，查找(x, y)点所在的三角形的索引
        indices = (self._cpp_trifinder.find_many(x.ravel(), y.ravel())
                   .reshape(x.shape))
        # 返回包含三角形索引的整数数组，形状与x和y相同
        return indices
    # 返回一个包含关于节点树统计信息的 Python 列表：
    # 0: 节点数（树的大小）
    # 1: 唯一节点数
    # 2: 梯形数（树的叶节点数）
    # 3: 唯一梯形数
    # 4: 最大父节点计数（节点在树中重复的最大次数）
    # 5: 树的最大深度（搜索树需要的最大比较次数加一）
    # 6: 所有梯形深度的平均值（搜索树需要的平均比较次数加一）
    def _get_tree_stats(self):
        """
        Return a python list containing the statistics about the node tree:
            0: number of nodes (tree size)
            1: number of unique nodes
            2: number of trapezoids (tree leaf nodes)
            3: number of unique trapezoids
            4: maximum parent count (max number of times a node is repeated in
                   tree)
            5: maximum depth of tree (one more than the maximum number of
                   comparisons needed to search through the tree)
            6: mean of all trapezoid depths (one more than the average number
                   of comparisons needed to search through the tree)
        """
        return self._cpp_trifinder.get_tree_stats()

    # 初始化底层的 C++ 对象。如果需要，可以多次调用，例如在修改三角剖分时。
    def _initialize(self):
        """
        Initialize the underlying C++ object.  Can be called multiple times if,
        for example, the triangulation is modified.
        """
        self._cpp_trifinder.initialize()

    # 打印节点树的文本表示，用于调试目的。
    def _print_tree(self):
        """
        Print a text representation of the node tree, which is useful for
        debugging purposes.
        """
        self._cpp_trifinder.print_tree()
```