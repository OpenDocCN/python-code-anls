# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_pairwise_distances_reduction\__init__.py`

```
#
# Pairwise Distances Reductions
# =============================
#
#   Authors: The scikit-learn developers.
#   License: BSD 3 clause
#

# 概述
# --------
#
#    该模块提供了计算X的一组行向量与Y的另一组行向量之间的成对距离，并在其上应用归约的例程。经典的例子是通过利用arg-k-min归约来进行前k个最近邻的蛮力计算。
#
#    归约接受作为输入的X和Y行之间的成对距离矩阵，并为X的每一行输出一个聚合数据结构。聚合值通常小于Y行数，因此称为归约。
#
#    为了计算效率，归约在X和Y行的分块上实时执行，以保持CPU缓存中的中间数据结构，并避免不必要地大内存距离数组的往返，否则会严重降低速度，使整体处理变为内存受限。
#
#    最后，这些例程遵循通用的并行化模板，使用OpenMP循环（通过Cython prange）处理数据块，可以是X的行或Y的行，取决于它们各自的大小。
#

# 分派到专门实现
# ------------------------------------------
#
#    分派器用于在Python代码中使用。在内部，分派器必须仅根据X和Y的dtype定义逻辑，以在运行时选择正确的dtype专用的:class:`BaseDistancesReductionDispatcher`实现。
#

# 高级图示
# ------------------
#
#    图例:
#
#      A ---⊳ B: A继承自B
#      A ---x B: A分派给B
#
#
#                                      (base dispatcher)
#                               BaseDistancesReductionDispatcher
#                                              ∆
#                                              |
#                                              |
#           +------------------+---------------+---------------+------------------+
#           |                  |                               |                  |
#           |             (dispatcher)                    (dispatcher)            |
#           |               ArgKmin                      RadiusNeighbors          |
#           |                  |                               |                  |
#           |                  |                               |                  |
#           |                  |     (float{32,64} implem.)    |                  |
#           |                  | BaseDistancesReduction{32,64} |                  |
#           |                  |               ∆               |                  |
#      (dispatcher)            |               |               |             (dispatcher)
#    ArgKminClassMode          |               |               |        RadiusNeighborsClassMode
#           |                  |    +----------+----------+    |                  |
#           |                  |    |                     |    |                  |
#           |                  x    |                     |    x                  |
#           |     +-------⊳ ArgKmin{32,64}         RadiusNeighbors{32,64} ⊲---+   |
#           x     |            |    ∆                     ∆    |              |   x
#   ArgKminClassMode{32,64}    |    |                     |    |   RadiusNeighborsClassMode{32,64}
# ===================================== Specializations ============================================
#                              |    |                     |    |
#                              |    |                     |    |
#                              x    |                     |    x
#                      EuclideanArgKmin{32,64}    EuclideanRadiusNeighbors{32,64}
#
#
#    For instance :class:`ArgKmin` dispatches to:
#      - :class:`ArgKmin64` if X and Y are two `float64` array-likes
#      - :class:`ArgKmin32` if X and Y are two `float32` array-likes
#
#    In addition, if the metric parameter is set to "euclidean" or "sqeuclidean",
#    then some direct subclass of `BaseDistancesReduction{32,64}` further dispatches
#    to one of their subclass for euclidean-specialized implementation. For instance,
#    :class:`ArgKmin64` dispatches to :class:`EuclideanArgKmin64`.
#
#    Those Euclidean-specialized implementations rely on optimal implementations of
#    a decomposition of the squared euclidean distance matrix into a sum of three terms
#    (see :class:`MiddleTermComputer{32,64}`).
#

from ._dispatcher import (
    ArgKmin,                     # 导入 ArgKmin 类，用于 K 最小值算法的调度
    ArgKminClassMode,            # 导入 ArgKminClassMode 类，用于 K 最小值算法的类别模式调度
    BaseDistancesReductionDispatcher,  # 导入 BaseDistancesReductionDispatcher 类，用于距离缩减的基本调度
    RadiusNeighbors,             # 导入 RadiusNeighbors 类，用于半径最近邻算法的调度
    RadiusNeighborsClassMode,    # 导入 RadiusNeighborsClassMode 类，用于半径最近邻算法的类别模式调度
    sqeuclidean_row_norms,       # 导入 sqeuclidean_row_norms 函数，用于计算平方欧氏距离的行规范
)

__all__ = [
    "BaseDistancesReductionDispatcher",   # 将 BaseDistancesReductionDispatcher 添加到模块的公共接口中
    "ArgKmin",                           # 将 ArgKmin 添加到模块的公共接口中
    "RadiusNeighbors",                   # 将 RadiusNeighbors 添加到模块的公共接口中
    "ArgKminClassMode",                  # 将 ArgKminClassMode 添加到模块的公共接口中
    "RadiusNeighborsClassMode",          # 将 RadiusNeighborsClassMode 添加到模块的公共接口中
    "sqeuclidean_row_norms",             # 将 sqeuclidean_row_norms 添加到模块的公共接口中
]

# ruff: noqa: E501  # 忽略 PEP 8 中的 E501 错误（行长度超过限制）
```