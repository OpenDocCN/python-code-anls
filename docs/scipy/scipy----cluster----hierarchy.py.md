# `D:\src\scipysrc\scipy\scipy\cluster\hierarchy.py`

```
"""
Hierarchical clustering (:mod:`scipy.cluster.hierarchy`)
========================================================

.. currentmodule:: scipy.cluster.hierarchy

These functions cut hierarchical clusterings into flat clusterings
or find the roots of the forest formed by a cut by providing the flat
cluster ids of each observation.

.. autosummary::
   :toctree: generated/

   fcluster
   fclusterdata
   leaders

These are routines for agglomerative clustering.

.. autosummary::
   :toctree: generated/

   linkage
   single
   complete
   average
   weighted
   centroid
   median
   ward

These routines compute statistics on hierarchies.

.. autosummary::
   :toctree: generated/

   cophenet
   from_mlab_linkage
   inconsistent
   maxinconsts
   maxdists
   maxRstat
   to_mlab_linkage

Routines for visualizing flat clusters.

.. autosummary::
   :toctree: generated/

   dendrogram

These are data structures and routines for representing hierarchies as
tree objects.

.. autosummary::
   :toctree: generated/

   ClusterNode
   leaves_list
   to_tree
   cut_tree
   optimal_leaf_ordering

These are predicates for checking the validity of linkage and
inconsistency matrices as well as for checking isomorphism of two
flat cluster assignments.

.. autosummary::
   :toctree: generated/

   is_valid_im
   is_valid_linkage
   is_isomorphic
   is_monotonic
   correspond
   num_obs_linkage

Utility routines for plotting:

.. autosummary::
   :toctree: generated/

   set_link_color_palette

Utility classes:

.. autosummary::
   :toctree: generated/

   DisjointSet -- data structure for incremental connectivity queries

"""
# Copyright (C) Damian Eads, 2007-2008. New BSD License.

# hierarchy.py (derived from cluster.py, http://scipy-cluster.googlecode.com)
#
# Author: Damian Eads
# Date:   September 22, 2007
#
# Copyright (c) 2007, 2008, Damian Eads
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   - Redistributions of source code must retain the above
#     copyright notice, this list of conditions and the
#     following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer
#     in the documentation and/or other materials provided with the
#     distribution.
#   - Neither the name of the author nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 导入警告模块，用于生成警告信息
import warnings
# bisect模块提供了用于操作有序列表的函数
import bisect
# collections模块中的deque用于创建双端队列数据结构
from collections import deque

# 导入numpy库，并使用np作为别名
import numpy as np
# 从当前目录下的_hierarchy和_optimal_leaf_ordering模块中导入内容
from . import _hierarchy, _optimal_leaf_ordering
# 导入scipy.spatial.distance模块中的distance别名
import scipy.spatial.distance as distance
# 导入scipy._lib._array_api模块中的部分内容
from scipy._lib._array_api import array_namespace, _asarray, copy, is_jax
# 导入scipy._lib._disjoint_set模块中的DisjointSet类
from scipy._lib._disjoint_set import DisjointSet

# 定义_LINKAGE_METHODS字典，将字符串表示的链接方法映射为整数
_LINKAGE_METHODS = {'single': 0, 'complete': 1, 'average': 2, 'centroid': 3,
                    'median': 4, 'ward': 5, 'weighted': 6}
# _EUCLIDEAN_METHODS元组包含了使用欧氏距离的链接方法名称
_EUCLIDEAN_METHODS = ('centroid', 'median', 'ward')

# __all__列表定义了在当前模块中导出的公共对象名称
__all__ = ['ClusterNode', 'DisjointSet', 'average', 'centroid', 'complete',
           'cophenet', 'correspond', 'cut_tree', 'dendrogram', 'fcluster',
           'fclusterdata', 'from_mlab_linkage', 'inconsistent',
           'is_isomorphic', 'is_monotonic', 'is_valid_im', 'is_valid_linkage',
           'leaders', 'leaves_list', 'linkage', 'maxRstat', 'maxdists',
           'maxinconsts', 'median', 'num_obs_linkage', 'optimal_leaf_ordering',
           'set_link_color_palette', 'single', 'to_mlab_linkage', 'to_tree',
           'ward', 'weighted']

# 定义ClusterWarning类，继承自UserWarning，用于在聚类过程中生成警告信息
class ClusterWarning(UserWarning):
    pass

# _warning函数定义，用于发出聚类过程中的警告信息
def _warning(s):
    # 发出警告，包含'scipy.cluster: '作为前缀
    warnings.warn('scipy.cluster: %s' % s, ClusterWarning, stacklevel=3)

# int_floor函数定义，将浮点数数组转换为整数类型，array_api_strict模块对此特别敏感
def int_floor(arr, xp):
    # array_api_strict对浮点数组不允许直接使用`int()`，这里进行显式转换
    return int(xp.astype(xp.asarray(arr), xp.int64))

# single函数定义，执行单连接/最小连接/最近连接的层次聚类算法
def single(y):
    """
    Perform single/min/nearest linkage on the condensed distance matrix ``y``.

    Parameters
    ----------
    y : ndarray
        The upper triangular of the distance matrix. The result of
        ``pdist`` is returned in this form.

    Returns
    -------
    Z : ndarray
        The linkage matrix.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import single, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = single(y)
    >>> Z
    """
    array([[ 0.,  1.,  1.,  2.],
           [ 2., 12.,  1.,  3.],
           [ 3.,  4.,  1.,  2.],
           [ 5., 14.,  1.,  3.],
           [ 6.,  7.,  1.,  2.],
           [ 8., 16.,  1.,  3.],
           [ 9., 10.,  1.,  2.],
           [11., 18.,  1.,  3.],
           [13., 15.,  2.,  6.],
           [17., 20.,  2.,  9.],
           [19., 21.,  2., 12.]])



    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.



    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:



    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 7,  8,  9, 10, 11, 12,  4,  5,  6,  1,  2,  3], dtype=int32)
    >>> fcluster(Z, 1, criterion='distance')
    array([3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1], dtype=int32)
    >>> fcluster(Z, 2, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)



    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.
    """
    return linkage(y, method='single', metric='euclidean')
# 定义一个函数，用于对一个压缩的距离矩阵执行完全/最大/最远点连接法的层次聚类

def complete(y):
    """
    Perform complete/max/farthest point linkage on a condensed distance matrix.

    Parameters
    ----------
    y : ndarray
        The upper triangular of the distance matrix. The result of
        ``pdist`` is returned in this form.

    Returns
    -------
    Z : ndarray
        A linkage matrix containing the hierarchical clustering. See
        the `linkage` function documentation for more information
        on its structure.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import complete, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = complete(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.41421356,  3.        ],
           [ 5.        , 13.        ,  1.41421356,  3.        ],
           [ 8.        , 14.        ,  1.41421356,  3.        ],
           [11.        , 15.        ,  1.41421356,  3.        ],
           [16.        , 17.        ,  4.12310563,  6.        ],
           [18.        , 19.        ,  4.12310563,  6.        ],
           [20.        , 21.        ,  5.65685425, 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
    >>> fcluster(Z, 1.5, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
    >>> fcluster(Z, 4.5, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=int32)
    >>> fcluster(Z, 6, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.
    """
    # 调用 scipy.cluster.hierarchy.linkage 函数执行层次聚类，使用完全连接法和欧氏距离
    return linkage(y, method='complete', metric='euclidean')
    y : ndarray
        距离矩阵的上三角形式。``pdist`` 函数的返回结果保存在这里。

    Returns
    -------
    Z : ndarray
        包含层次聚类结果的链接矩阵。详细结构请参考 `linkage` 的文档。

    See Also
    --------
    linkage : 更高级的层次聚类创建方法。
    scipy.spatial.distance.pdist : 用于计算成对距离的函数。

    Examples
    --------
    >>> from scipy.cluster.hierarchy import average, fcluster
    >>> from scipy.spatial.distance import pdist

    首先，我们需要一个玩具数据集来演示::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    然后，我们从这个数据集中得到压缩距离矩阵：

    >>> y = pdist(X)

    最后，我们可以执行层次聚类：

    >>> Z = average(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.20710678,  3.        ],
           [ 5.        , 13.        ,  1.20710678,  3.        ],
           [ 8.        , 14.        ,  1.20710678,  3.        ],
           [11.        , 15.        ,  1.20710678,  3.        ],
           [16.        , 17.        ,  3.39675184,  6.        ],
           [18.        , 19.        ,  3.39675184,  6.        ],
           [20.        , 21.        ,  4.09206523, 12.        ]])

    链接矩阵 ``Z`` 表示一个树状图 - 参见 `scipy.cluster.hierarchy.linkage` 获取其详细内容的解释。

    我们可以使用 `scipy.cluster.hierarchy.fcluster` 函数根据距离阈值查看每个初始点属于哪个聚类：

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
    >>> fcluster(Z, 1.5, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
    >>> fcluster(Z, 4, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=int32)
    >>> fcluster(Z, 6, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    另外，可以使用 `scipy.cluster.hierarchy.dendrogram` 来生成树状图的绘图。
    """
    return linkage(y, method='average', metric='euclidean')
# 定义函数，执行加权（weighted）或WPGMA（加权平均群聚）链接，基于压缩的距离矩阵。

"""
Perform weighted/WPGMA linkage on the condensed distance matrix.

See `linkage` for more information on the return
structure and algorithm.

Parameters
----------
y : ndarray
    The upper triangular of the distance matrix. The result of
    ``pdist`` is returned in this form.

Returns
-------
Z : ndarray
    A linkage matrix containing the hierarchical clustering. See
    `linkage` for more information on its structure.

See Also
--------
linkage : for advanced creation of hierarchical clusterings.
scipy.spatial.distance.pdist : pairwise distance metrics

Examples
--------
>>> from scipy.cluster.hierarchy import weighted, fcluster
>>> from scipy.spatial.distance import pdist

First, we need a toy dataset to play with::

    x x    x x
    x        x

    x        x
    x x    x x

>>> X = [[0, 0], [0, 1], [1, 0],
...      [0, 4], [0, 3], [1, 4],
...      [4, 0], [3, 0], [4, 1],
...      [4, 4], [3, 4], [4, 3]]

Then, we get a condensed distance matrix from this dataset:

>>> y = pdist(X)

Finally, we can perform the clustering:

>>> Z = weighted(y)
>>> Z
array([[ 0.        ,  1.        ,  1.        ,  2.        ],
       [ 6.        ,  7.        ,  1.        ,  2.        ],
       [ 3.        ,  4.        ,  1.        ,  2.        ],
       [ 9.        , 11.        ,  1.        ,  2.        ],
       [ 2.        , 12.        ,  1.20710678,  3.        ],
       [ 8.        , 13.        ,  1.20710678,  3.        ],
       [ 5.        , 14.        ,  1.20710678,  3.        ],
       [10.        , 15.        ,  1.20710678,  3.        ],
       [18.        , 19.        ,  3.05595762,  6.        ],
       [16.        , 17.        ,  3.32379407,  6.        ],
       [20.        , 21.        ,  4.06357713, 12.        ]])

The linkage matrix ``Z`` represents a dendrogram - see
`scipy.cluster.hierarchy.linkage` for a detailed explanation of its
contents.

We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
each initial point would belong given a distance threshold:

>>> fcluster(Z, 0.9, criterion='distance')
array([ 7,  8,  9,  1,  2,  3, 10, 11, 12,  4,  6,  5], dtype=int32)
>>> fcluster(Z, 1.5, criterion='distance')
array([3, 3, 3, 1, 1, 1, 4, 4, 4, 2, 2, 2], dtype=int32)
>>> fcluster(Z, 4, criterion='distance')
array([2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1], dtype=int32)
>>> fcluster(Z, 6, criterion='distance')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
plot of the dendrogram.

"""
return linkage(y, method='weighted', metric='euclidean')
    return structure, and algorithm.  # 返回的结构和算法。

    The following are common calling conventions:  # 以下是常见的调用约定：

    1. ``Z = centroid(y)``

       Performs centroid/UPGMC linkage on the condensed distance
       matrix ``y``.  # 对压缩距离矩阵 ``y`` 执行质心/UPGMC联接。

    2. ``Z = centroid(X)``

       Performs centroid/UPGMC linkage on the observation matrix ``X``
       using Euclidean distance as the distance metric.  # 使用欧几里德距离作为距离度量，在观察矩阵 ``X`` 上执行质心/UPGMC联接。

    Parameters
    ----------
    y : ndarray
        A condensed distance matrix. A condensed
        distance matrix is a flat array containing the upper
        triangular of the distance matrix. This is the form that
        ``pdist`` returns. Alternatively, a collection of
        m observation vectors in n dimensions may be passed as
        an m by n array.  # 压缩距离矩阵 ``y``。压缩距离矩阵是一个扁平数组，包含距离矩阵的上三角部分。这是 ``pdist`` 返回的形式。或者可以传递一个 m 行 n 列的数组，表示 m 个 n 维观测向量。

    Returns
    -------
    Z : ndarray
        A linkage matrix containing the hierarchical clustering. See
        the `linkage` function documentation for more information
        on its structure.  # 包含层次聚类的联接矩阵。详细结构信息请参阅 `linkage` 函数的文档。

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.  # 用于创建高级层次聚类的 `linkage` 函数。
    scipy.spatial.distance.pdist : pairwise distance metrics  # scipy 中的 pairwise 距离度量函数 pdist。

    Examples
    --------
    >>> from scipy.cluster.hierarchy import centroid, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = centroid(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.33333333,  6.        ],
           [16.        , 17.        ,  3.33333333,  6.        ],
           [20.        , 21.        ,  3.33333333, 12.        ]]) # may vary

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.  # 联接矩阵 ``Z`` 表示一棵树状图 - 详细内容解释请参阅 `scipy.cluster.hierarchy.linkage`。

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6], dtype=int32) # may vary
    >>> fcluster(Z, 1.1, criterion='distance')
    array([5, 5, 6, 7, 7, 8, 1, 1, 2, 3, 3, 4], dtype=int32) # may vary
    >>> fcluster(Z, 2, criterion='distance')
    # 创建一个包含距离矩阵 Z 的层次聚类结果，使用 4 个聚类，并基于距离判据进行聚类
    >>> fcluster(Z, 4, criterion='distance')

    # 返回一个包含 12 个元素的数组，表示每个观测所属的聚类编号，这里每个观测都属于第一个聚类
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    # 此外，可以使用 `scipy.cluster.hierarchy.dendrogram` 生成一幅树状图谱
    Also, `scipy.cluster.hierarchy.dendrogram` can be used to generate a
    plot of the dendrogram.

    """
    # 使用层次聚类函数 linkage 对观测数据 y 进行聚类，方法为 'centroid'，距离度量为欧氏距离，并返回结果
    return linkage(y, method='centroid', metric='euclidean')
# 定义了一个名为 median 的函数，用于执行中位数/平均连接（median/WPGMC linkage）的层次聚类算法。
def median(y):
    """
    Perform median/WPGMC linkage.

    See `linkage` for more information on the return structure
    and algorithm.

     The following are common calling conventions:

     1. ``Z = median(y)``

        Performs median/WPGMC linkage on the condensed distance matrix
        ``y``.  See ``linkage`` for more information on the return
        structure and algorithm.

     2. ``Z = median(X)``

        Performs median/WPGMC linkage on the observation matrix ``X``
        using Euclidean distance as the distance metric. See `linkage`
        for more information on the return structure and algorithm.

    Parameters
    ----------
    y : ndarray
        A condensed distance matrix. A condensed
        distance matrix is a flat array containing the upper
        triangular of the distance matrix. This is the form that
        ``pdist`` returns.  Alternatively, a collection of
        m observation vectors in n dimensions may be passed as
        an m by n array.

    Returns
    -------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = median(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.        ,  6.        ],
           [16.        , 17.        ,  3.5       ,  6.        ],
           [20.        , 21.        ,  3.25      , 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6], dtype=int32)
    """
    # 调用 scipy 的层次聚类函数，执行中位数/平均连接算法，并返回聚类结果的链接矩阵 Z
    Z = linkage(y, method='median')
    # 返回聚类结果的链接矩阵 Z
    return Z
    # 使用 linkage 函数计算层次聚类的结果，返回一个链接矩阵
    >>> fcluster(Z, 1.1, criterion='distance')
    # 使用给定的阈值参数（1.1），基于距离标准执行层次聚类，返回聚类结果的整数数组
    array([5, 5, 6, 7, 7, 8, 1, 1, 2, 3, 3, 4], dtype=int32)
    >>> fcluster(Z, 2, criterion='distance')
    # 使用另一个阈值参数（2），基于距离标准再次执行层次聚类，返回聚类结果的整数数组
    array([3, 3, 3, 4, 4, 4, 1, 1, 1, 2, 2, 2], dtype=int32)
    >>> fcluster(Z, 4, criterion='distance')
    # 使用另一个阈值参数（4），基于距离标准再次执行层次聚类，返回聚类结果的整数数组
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)
    
    # 此外，`scipy.cluster.hierarchy.dendrogram` 可用于生成树状图谱
    """
    使用 linkage 函数进行层次聚类分析，返回聚类后的链接矩阵。
    """
    return linkage(y, method='median', metric='euclidean')
# 定义函数 ward，执行 Ward's linkage 算法来计算层次聚类的链接矩阵
def ward(y):
    """
    Perform Ward's linkage on a condensed distance matrix.

    See `linkage` for more information on the return structure
    and algorithm.

    The following are common calling conventions:

    1. ``Z = ward(y)``
       Performs Ward's linkage on the condensed distance matrix ``y``.

    2. ``Z = ward(X)``
       Performs Ward's linkage on the observation matrix ``X`` using
       Euclidean distance as the distance metric.

    Parameters
    ----------
    y : ndarray
        A condensed distance matrix. A condensed
        distance matrix is a flat array containing the upper
        triangular of the distance matrix. This is the form that
        ``pdist`` returns.  Alternatively, a collection of
        m observation vectors in n dimensions may be passed as
        an m by n array.

    Returns
    -------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix. See
        `linkage` for more information on the return structure and
        algorithm.

    See Also
    --------
    linkage : for advanced creation of hierarchical clusterings.
    scipy.spatial.distance.pdist : pairwise distance metrics

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, fcluster
    >>> from scipy.spatial.distance import pdist

    First, we need a toy dataset to play with::

        x x    x x
        x        x

        x        x
        x x    x x

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    Then, we get a condensed distance matrix from this dataset:

    >>> y = pdist(X)

    Finally, we can perform the clustering:

    >>> Z = ward(y)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    The linkage matrix ``Z`` represents a dendrogram - see
    `scipy.cluster.hierarchy.linkage` for a detailed explanation of its
    contents.

    We can use `scipy.cluster.hierarchy.fcluster` to see to which cluster
    each initial point would belong given a distance threshold:

    >>> fcluster(Z, 0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
    >>> fcluster(Z, 1.1, criterion='distance')
    array([1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8], dtype=int32)

    """
    # 使用给定的层次聚类结果 Z 和距离标准为 'distance'，对数据进行聚类，并返回聚类结果的标签数组
    >>> fcluster(Z, 3, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)
    
    # 使用给定的层次聚类结果 Z 和距离标准为 'distance'，对数据进行聚类，并返回聚类结果的标签数组
    >>> fcluster(Z, 9, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)
    
    # 此函数的返回值是一个聚类树的链接矩阵，可以使用 `scipy.cluster.hierarchy.dendrogram` 函数生成树状图。
    """
    根据输入的数据 y，使用 ward 方法和欧氏距离进行层次聚类，返回聚类树的链接矩阵。
    """
    return linkage(y, method='ward', metric='euclidean')
def linkage(y, method='single', metric='euclidean', optimal_ordering=False):
    """
    Perform hierarchical/agglomerative clustering.

    The input y may be either a 1-D condensed distance matrix
    or a 2-D array of observation vectors.

    If y is a 1-D condensed distance matrix,
    then y must be a :math:`\\binom{n}{2}` sized
    vector, where n is the number of original observations paired
    in the distance matrix. The behavior of this function is very
    similar to the MATLAB linkage function.

    A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
    fourth value ``Z[i, 3]`` represents the number of original
    observations in the newly formed cluster.

    The following linkage methods are used to compute the distance
    :math:`d(s, t)` between two clusters :math:`s` and
    :math:`t`. The algorithm begins with a forest of clusters that
    have yet to be used in the hierarchy being formed. When two
    clusters :math:`s` and :math:`t` from this forest are combined
    into a single cluster :math:`u`, :math:`s` and :math:`t` are
    removed from the forest, and :math:`u` is added to the
    forest. When only one cluster remains in the forest, the algorithm
    stops, and this cluster becomes the root.

    A distance matrix is maintained at each iteration. The ``d[i,j]``
    entry corresponds to the distance between cluster :math:`i` and
    :math:`j` in the original forest.

    At each iteration, the algorithm must update the distance matrix
    to reflect the distance of the newly formed cluster u with the
    remaining clusters in the forest.

    Suppose there are :math:`|u|` original observations
    :math:`u[0], \\ldots, u[|u|-1]` in cluster :math:`u` and
    :math:`|v|` original objects :math:`v[0], \\ldots, v[|v|-1]` in
    cluster :math:`v`. Recall, :math:`s` and :math:`t` are
    combined to form cluster :math:`u`. Let :math:`v` be any
    remaining cluster in the forest that is not :math:`u`.

    The following are methods for calculating the distance between the
    clusters:

    """

    # 以下代码段为具体的聚类算法实现，根据输入参数进行不同的聚类方法计算
    # 如果输入的 y 是一个 1-D 紧缩距离矩阵，则进行相应处理
    if np.ndim(y) == 1:
        # 根据距离矩阵的大小计算观测值的数量 n
        n = int(np.ceil(np.sqrt(2 * len(y))))
        # 构建一个方形形式的距离矩阵
        Z = np.zeros((n - 1, 4))
        # 计算形成距离矩阵 Z 的过程
        combine(y, Z)
    # 如果输入的 y 是一个二维的观测向量数组，则进行不同的处理
    elif np.ndim(y) == 2:
        # 初始化观测向量的数量为 n
        Z = np.zeros((y.shape[0] - 1, 4))
        # 根据不同的方法进行聚类的计算过程
        if y.dtype.kind == 'f':
            # 如果 y 的数据类型是浮点数，根据给定的方法和度量进行聚类计算
            _hierarchical.hc(y, Z, int(n), int(m), l)
        elif y.dtype.kind == 'u':
            # 如果 y 的数据类型是无符号整数，进行另一种计算方式
            _hierarchical.hac(x, 'e', k, i)
    return Z
    newly formed cluster :math:`u` and each :math:`v`.

      * method='single' assigns

        .. math::
           d(u,v) = \\min(dist(u[i],v[j]))

        for all points :math:`i` in cluster :math:`u` and
        :math:`j` in cluster :math:`v`. This is also known as the
        Nearest Point Algorithm.

      * method='complete' assigns

        .. math::
           d(u, v) = \\max(dist(u[i],v[j]))

        for all points :math:`i` in cluster u and :math:`j` in
        cluster :math:`v`. This is also known by the Farthest Point
        Algorithm or Voor Hees Algorithm.

      * method='average' assigns

        .. math::
           d(u,v) = \\sum_{ij} \\frac{d(u[i], v[j])}
                                   {(|u|*|v|)}

        for all points :math:`i` and :math:`j` where :math:`|u|`
        and :math:`|v|` are the cardinalities of clusters :math:`u`
        and :math:`v`, respectively. This is also called the UPGMA
        algorithm.

      * method='weighted' assigns

        .. math::
           d(u,v) = (dist(s,v) + dist(t,v))/2

        where cluster u was formed with cluster s and t and v
        is a remaining cluster in the forest (also called WPGMA).

      * method='centroid' assigns

        .. math::
           dist(s,t) = ||c_s-c_t||_2

        where :math:`c_s` and :math:`c_t` are the centroids of
        clusters :math:`s` and :math:`t`, respectively. When two
        clusters :math:`s` and :math:`t` are combined into a new
        cluster :math:`u`, the new centroid is computed over all the
        original objects in clusters :math:`s` and :math:`t`. The
        distance then becomes the Euclidean distance between the
        centroid of :math:`u` and the centroid of a remaining cluster
        :math:`v` in the forest. This is also known as the UPGMC
        algorithm.

      * method='median' assigns :math:`d(s,t)` like the ``centroid``
        method. When two clusters :math:`s` and :math:`t` are combined
        into a new cluster :math:`u`, the average of centroids s and t
        give the new centroid :math:`u`. This is also known as the
        WPGMC algorithm.

      * method='ward' uses the Ward variance minimization algorithm.
        The new entry :math:`d(u,v)` is computed as follows,

        .. math::

           d(u,v) = \\sqrt{\\frac{|v|+|s|}
                               {T}d(v,s)^2
                        + \\frac{|v|+|t|}
                               {T}d(v,t)^2
                        - \\frac{|v|}
                               {T}d(s,t)^2}

        where :math:`u` is the newly joined cluster consisting of
        clusters :math:`s` and :math:`t`, :math:`v` is an unused
        cluster in the forest, :math:`T=|v|+|s|+|t|`, and
        :math:`|*|` is the cardinality of its argument. This is also
        known as the incremental algorithm.

    Warning: When the minimum distance pair in the forest is chosen, there
    may be two or more pairs with the same minimum distance. This
    # 导入所需的库和模块
    from scipy.spatial.distance import pdist
    from scipy.cluster import hierarchy
    
    # 定义层次聚类函数
    def linkage(y, method='single', metric='euclidean', optimal_ordering=False):
        # 返回一个层次聚类的链接矩阵
        Z = hierarchy.linkage(y, method=method, metric=metric, optimal_ordering=optimal_ordering)
        # 返回层次聚类的链接矩阵
        return Z
    >>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]


    # 创建一个二维列表 X，包含多个单元素列表，用于聚类分析的数据
    X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

    >>> Z = linkage(X, 'ward')
    # 使用 ward 方法进行层次聚类，并返回聚类结果的链接矩阵 Z
    Z = linkage(X, 'ward')
    # 创建一个大小为 (25, 10) 的新图形对象
    fig = plt.figure(figsize=(25, 10))
    # 绘制树状图（树状图用于可视化层次聚类结果），并将其分配给变量 dn
    dn = dendrogram(Z)

    >>> Z = linkage(X, 'single')
    # 使用 single 方法进行层次聚类，并返回聚类结果的链接矩阵 Z
    Z = linkage(X, 'single')
    # 创建一个大小为 (25, 10) 的新图形对象
    fig = plt.figure(figsize=(25, 10))
    # 绘制树状图，将其分配给变量 dn，并显示图形
    dn = dendrogram(Z)
    plt.show()
    """
    # 以下代码段来自一个函数或方法，接受 y 作为输入参数，并执行一系列数据验证和处理步骤
    xp = array_namespace(y)
    # 将 y 转换为数组，并确保其为 float64 类型的 C 顺序数组
    y = _asarray(y, order='C', dtype=xp.float64, xp=xp)

    # 如果指定的聚类方法不在预定义的方法列表中，则抛出 ValueError 异常
    if method not in _LINKAGE_METHODS:
        raise ValueError(f"Invalid method: {method}")

    # 对于欧氏距离相关的方法，要求距离度量必须为欧氏距离
    if method in _EUCLIDEAN_METHODS and metric != 'euclidean' and y.ndim == 2:
        msg = f"`method={method}` requires the distance metric to be Euclidean"
        raise ValueError(msg)

    # 如果 y 是一维数组，验证其有效性，否则如果是二维数组，则进行额外的验证和处理
    if y.ndim == 1:
        distance.is_valid_y(y, throw=True, name='y')
    elif y.ndim == 2:
        # 如果 y 是对称的非负空矩阵，并且符合特定条件，则发出警告
        if (y.shape[0] == y.shape[1] and np.allclose(np.diag(y), 0) and
                xp.all(y >= 0) and np.allclose(y, y.T)):
            warnings.warn('The symmetric non-negative hollow observation '
                          'matrix looks suspiciously like an uncondensed '
                          'distance matrix',
                          ClusterWarning, stacklevel=2)
        # 计算 y 的紧致形式的距离矩阵，并将结果赋给 y
        y = distance.pdist(y, metric)
        y = xp.asarray(y)
    else:
        # 如果 y 不是 1 或 2 维，则引发 ValueError 异常
        raise ValueError("`y` must be 1 or 2 dimensional.")

    # 确保 y 中的所有值都是有限的，否则引发 ValueError 异常
    if not xp.all(xp.isfinite(y)):
        raise ValueError("The condensed distance matrix must contain only "
                         "finite values.")

    # 计算观测数 n，并根据指定的方法编码获取相应的方法代码
    n = int(distance.num_obs_y(y))
    method_code = _LINKAGE_METHODS[method]

    # 将 y 转换为 NumPy 数组
    y = np.asarray(y)
    # 根据不同的聚类方法调用相应的聚类函数，生成聚类结果
    if method == 'single':
        result = _hierarchy.mst_single_linkage(y, n)
    elif method in ['complete', 'average', 'weighted', 'ward']:
        result = _hierarchy.nn_chain(y, n, method_code)
    else:
        result = _hierarchy.fast_linkage(y, n, method_code)
    # 将结果转换为 xp 的数组形式
    result = xp.asarray(result)

    # 如果指定了 optimal_ordering 参数为 True，则执行优化叶子节点的顺序操作
    if optimal_ordering:
        y = xp.asarray(y)
        return optimal_leaf_ordering(result, y)
    else:
        # 否则直接返回聚类结果
        return result
class ClusterNode:
    """
    A tree node class for representing a cluster.

    Leaf nodes correspond to original observations, while non-leaf nodes
    correspond to non-singleton clusters.

    The `to_tree` function converts a matrix returned by the linkage
    function into an easy-to-use tree representation.

    All parameter names are also attributes.

    Parameters
    ----------
    id : int
        The node id.
    left : ClusterNode instance, optional
        The left child tree node.
    right : ClusterNode instance, optional
        The right child tree node.
    dist : float, optional
        Distance for this cluster in the linkage matrix.
    count : int, optional
        The number of samples in this cluster.

    See Also
    --------
    to_tree : for converting a linkage matrix ``Z`` into a tree object.

    """

    def __init__(self, id, left=None, right=None, dist=0, count=1):
        # 检查节点 id 和距离 dist 是否为非负数
        if id < 0:
            raise ValueError('The id must be non-negative.')
        if dist < 0:
            raise ValueError('The distance must be non-negative.')
        # 检查是否为全树或者适当的二叉树结构
        if (left is None and right is not None) or \
           (left is not None and right is None):
            raise ValueError('Only full or proper binary trees are permitted.'
                             '  This node has one child.')
        # 检查聚类中至少有一个原始观察样本
        if count < 1:
            raise ValueError('A cluster must contain at least one original '
                             'observation.')
        
        # 初始化节点的属性
        self.id = id
        self.left = left
        self.right = right
        self.dist = dist
        # 如果是叶子节点，则设置 count 为节点本身的样本数，否则为左右子树样本数之和
        if self.left is None:
            self.count = count
        else:
            self.count = left.count + right.count

    def __lt__(self, node):
        # 定义小于比较操作，按节点之间的距离比较
        if not isinstance(node, ClusterNode):
            raise ValueError("Can't compare ClusterNode "
                             f"to type {type(node)}")
        return self.dist < node.dist

    def __gt__(self, node):
        # 定义大于比较操作，按节点之间的距离比较
        if not isinstance(node, ClusterNode):
            raise ValueError("Can't compare ClusterNode "
                             f"to type {type(node)}")
        return self.dist > node.dist

    def __eq__(self, node):
        # 定义等于比较操作，按节点之间的距离比较
        if not isinstance(node, ClusterNode):
            raise ValueError("Can't compare ClusterNode "
                             f"to type {type(node)}")
        return self.dist == node.dist

    def get_id(self):
        """
        The identifier of the target node.

        For ``0 <= i < n``, `i` corresponds to original observation i.
        For ``n <= i < 2n-1``, `i` corresponds to non-singleton cluster formed
        at iteration ``i-n``.

        Returns
        -------
        id : int
            The identifier of the target node.

        """
        # 返回节点的 id
        return self.id
    def get_count(self):
        """
        The number of leaf nodes (original observations) belonging to
        the cluster node nd. If the target node is a leaf, 1 is
        returned.

        Returns
        -------
        get_count : int
            The number of leaf nodes below the target node.

        """
        # 返回当前节点的叶子节点数量
        return self.count

    def get_left(self):
        """
        Return a reference to the left child tree object.

        Returns
        -------
        left : ClusterNode
            The left child of the target node. If the node is a leaf,
            None is returned.

        """
        # 返回当前节点的左子树对象引用，如果节点是叶子节点则返回None
        return self.left

    def get_right(self):
        """
        Return a reference to the right child tree object.

        Returns
        -------
        right : ClusterNode
            The right child of the target node. If the node is a leaf,
            None is returned.

        """
        # 返回当前节点的右子树对象引用，如果节点是叶子节点则返回None
        return self.right

    def is_leaf(self):
        """
        Return True if the target node is a leaf.

        Returns
        -------
        leafness : bool
            True if the target node is a leaf node.

        """
        # 判断当前节点是否为叶子节点，返回布尔值
        return self.left is None
    def pre_order(self, func=(lambda x: x.id)):
        """
        Perform pre-order traversal without recursive function calls.

        When a leaf node is first encountered, ``func`` is called with
        the leaf node as its argument, and its result is appended to
        the list.

        For example, the statement::

           ids = root.pre_order(lambda x: x.id)

        returns a list of the node ids corresponding to the leaf nodes
        of the tree as they appear from left to right.

        Parameters
        ----------
        func : function
            Applied to each leaf ClusterNode object in the pre-order traversal.
            Given the ``i``-th leaf node in the pre-order traversal ``n[i]``,
            the result of ``func(n[i])`` is stored in ``L[i]``. If not
            provided, the index of the original observation to which the node
            corresponds is used.

        Returns
        -------
        L : list
            The pre-order traversal.

        """
        # Do a preorder traversal, caching the result. To avoid having to do
        # recursion, we'll store the previous index we've visited in a vector.
        n = self.count  # 获取节点总数，用于初始化数组长度

        curNode = [None] * (2 * n)  # 初始化一个长度为 2*n 的数组 curNode，用于存储当前访问的节点
        lvisited = set()  # 初始化一个集合 lvisited，用于存储已访问过左子节点的节点 ID
        rvisited = set()  # 初始化一个集合 rvisited，用于存储已访问过右子节点的节点 ID
        curNode[0] = self  # 将当前节点设置为根节点，存放在 curNode 数组的第一个位置
        k = 0  # 初始化 k 为 0，表示当前访问的节点索引
        preorder = []  # 初始化一个空列表 preorder，用于存储前序遍历结果

        while k >= 0:
            nd = curNode[k]  # 获取当前访问的节点
            ndid = nd.id  # 获取当前节点的 ID

            if nd.is_leaf():  # 判断当前节点是否为叶子节点
                preorder.append(func(nd))  # 如果是叶子节点，将 func(nd) 的结果添加到 preorder 列表中
                k = k - 1  # 回溯到上一个节点
            else:
                if ndid not in lvisited:  # 如果当前节点的左子节点未被访问过
                    curNode[k + 1] = nd.left  # 将左子节点添加到 curNode 数组中
                    lvisited.add(ndid)  # 将当前节点的 ID 添加到 lvisited 集合中
                    k = k + 1  # 移动到左子节点
                elif ndid not in rvisited:  # 如果当前节点的右子节点未被访问过
                    curNode[k + 1] = nd.right  # 将右子节点添加到 curNode 数组中
                    rvisited.add(ndid)  # 将当前节点的 ID 添加到 rvisited 集合中
                    k = k + 1  # 移动到右子节点
                else:
                    k = k - 1  # 如果当前节点的左右子节点都已经访问过，则回溯到上一个节点

        return preorder  # 返回前序遍历结果列表
# 创建一个 ClusterNode 对象，表示一个裸节点，节点编号为 0
_cnode_bare = ClusterNode(0)
# 获取 ClusterNode 类型对象的引用
_cnode_type = type(ClusterNode)

# 定义函数 _order_cluster_tree，用于返回按距离从低到高排序的聚类节点列表
def _order_cluster_tree(Z):
    """
    Return clustering nodes in bottom-up order by distance.

    Parameters
    ----------
    Z : scipy.cluster.linkage array
        The linkage matrix.

    Returns
    -------
    nodes : list
        A list of ClusterNode objects.
    """
    # 初始化一个双端队列
    q = deque()
    # 将 linkage matrix Z 转换为树形结构
    tree = to_tree(Z)
    # 将树的根节点放入队列
    q.append(tree)
    # 初始化空列表 nodes，用于存储聚类节点
    nodes = []

    # 遍历队列
    while q:
        # 弹出队列左侧的节点
        node = q.popleft()
        # 如果节点不是叶子节点
        if not node.is_leaf():
            # 将节点插入到已排序的 nodes 列表中
            bisect.insort_left(nodes, node)
            # 将节点的右子节点和左子节点依次放入队列尾部
            q.append(node.get_right())
            q.append(node.get_left())
    # 返回按照距离从低到高排序的聚类节点列表
    return nodes


# 定义函数 cut_tree，根据给定的 linkage matrix Z 返回切割后的树
def cut_tree(Z, n_clusters=None, height=None):
    """
    Given a linkage matrix Z, return the cut tree.

    Parameters
    ----------
    Z : scipy.cluster.linkage array
        The linkage matrix.
    n_clusters : array_like, optional
        Number of clusters in the tree at the cut point.
    height : array_like, optional
        The height at which to cut the tree. Only possible for ultrametric
        trees.

    Returns
    -------
    cutree : array
        An array indicating group membership at each agglomeration step. I.e.,
        for a full cut tree, in the first column each data point is in its own
        cluster. At the next step, two nodes are merged. Finally, all
        singleton and non-singleton clusters are in one group. If `n_clusters`
        or `height` are given, the columns correspond to the columns of
        `n_clusters` or `height`.

    Examples
    --------
    >>> from scipy import cluster
    >>> import numpy as np
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> X = rng.random((50, 4))
    >>> Z = cluster.hierarchy.ward(X)
    >>> cutree = cluster.hierarchy.cut_tree(Z, n_clusters=[5, 10])
    >>> cutree[:10]
    array([[0, 0],
           [1, 1],
           [2, 2],
           [3, 3],
           [3, 4],
           [2, 2],
           [0, 0],
           [1, 5],
           [3, 6],
           [4, 7]])  # random

    """
    # 使用 array_namespace 函数创建 XP 数组命名空间
    xp = array_namespace(Z)
    # 获取 linkage matrix Z 的观测数目
    nobs = num_obs_linkage(Z)
    # 获取聚类节点列表，按距离从低到高排序
    nodes = _order_cluster_tree(Z)

    # 如果同时指定了 height 和 n_clusters，则抛出 ValueError
    if height is not None and n_clusters is not None:
        raise ValueError("At least one of either height or n_clusters "
                         "must be None")
    # 如果 height 为 None 且 n_clusters 为 None，则返回完整的切割树
    elif height is None and n_clusters is None:
        cols_idx = xp.arange(nobs)
    # 如果指定了 height，则根据 height 切割树
    elif height is not None:
        height = xp.asarray(height)
        heights = xp.asarray([x.dist for x in nodes])
        cols_idx = xp.searchsorted(heights, height)
    # 如果指定了 n_clusters，则根据 n_clusters 切割树
    else:
        n_clusters = xp.asarray(n_clusters)
        cols_idx = nobs - xp.searchsorted(xp.arange(nobs), n_clusters)

    # 尝试获取 cols_idx 的长度，如果是标量，则转换为长度为 1 的数组
    try:
        n_cols = len(cols_idx)
    except TypeError:  # scalar
        n_cols = 1
        cols_idx = xp.asarray([cols_idx])

    # 初始化一个全零矩阵，用于存储各个切割步骤的群组成员关系
    groups = xp.zeros((n_cols, nobs), dtype=xp.int64)
    last_group = xp.arange(nobs)
    # 如果 0 在 cols_idx 中，则将最初的组成员关系保存在第一个切割步骤中
    if 0 in cols_idx:
        groups[0] = last_group
    # 遍历节点列表 `nodes`，同时获取索引 `i` 和节点 `node`
    for i, node in enumerate(nodes):
        # 获取节点 `node` 的先序遍历索引 `idx`
        idx = node.pre_order()
        
        # 复制 `last_group` 并命名为 `this_group`，使用当前的数组库 `xp`
        this_group = copy(last_group, xp=xp)
        
        # TODO ARRAY_API 复杂的索引不支持
        # 用 `xp` 库计算 `last_group[idx]` 的最小值，并将结果赋值给 `this_group[idx]`
        this_group[idx] = xp.min(last_group[idx])
        
        # 将 `this_group[idx]` 大于 `last_group[idx]` 的元素减去 1
        this_group[this_group > xp.max(last_group[idx])] -= 1
        
        # 如果 `i + 1` 存在于 `cols_idx` 中，则将 `this_group` 的对应列赋值给 `groups`
        if i + 1 in cols_idx:
            groups[np.nonzero(i + 1 == cols_idx)[0]] = this_group
        
        # 更新 `last_group` 为当前的 `this_group`
        last_group = this_group

    # 返回转置后的 `groups`
    return groups.T
def to_tree(Z, rd=False):
    """
    Convert a linkage matrix into an easy-to-use tree object.

    The reference to the root `ClusterNode` object is returned (by default).

    Each `ClusterNode` object has a ``left``, ``right``, ``dist``, ``id``,
    and ``count`` attribute. The left and right attributes point to
    ClusterNode objects that were combined to generate the cluster.
    If both are None then the `ClusterNode` object is a leaf node, its count
    must be 1, and its distance is meaningless but set to 0.

    *Note: This function is provided for the convenience of the library
    user. ClusterNodes are not used as input to any of the functions in this
    library.*

    Parameters
    ----------
    Z : ndarray
        The linkage matrix in proper form (see the `linkage`
        function documentation).
    rd : bool, optional
        When False (default), a reference to the root `ClusterNode` object is
        returned.  Otherwise, a tuple ``(r, d)`` is returned. ``r`` is a
        reference to the root node while ``d`` is a list of `ClusterNode`
        objects - one per original entry in the linkage matrix plus entries
        for all clustering steps. If a cluster id is
        less than the number of samples ``n`` in the data that the linkage
        matrix describes, then it corresponds to a singleton cluster (leaf
        node).
        See `linkage` for more information on the assignment of cluster ids
        to clusters.

    Returns
    -------
    tree : ClusterNode or tuple (ClusterNode, list of ClusterNode)
        If ``rd`` is False, a `ClusterNode`.
        If ``rd`` is True, a list of length ``2*n - 1``, with ``n`` the number
        of samples.  See the description of `rd` above for more details.

    See Also
    --------
    linkage, is_valid_linkage, ClusterNode

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> rng = np.random.default_rng()
    >>> x = rng.random((5, 2))
    >>> Z = hierarchy.linkage(x)
    >>> hierarchy.to_tree(Z)
    <scipy.cluster.hierarchy.ClusterNode object at ...
    >>> rootnode, nodelist = hierarchy.to_tree(Z, rd=True)
    >>> rootnode
    <scipy.cluster.hierarchy.ClusterNode object at ...
    >>> len(nodelist)
    9

    """
    # Import the array namespace from NumPy
    xp = array_namespace(Z)
    # Convert Z to a contiguous array in C-order using the provided array namespace
    Z = _asarray(Z, order='c', xp=xp)
    # Validate the linkage matrix Z, throwing an error if invalid
    is_valid_linkage(Z, throw=True, name='Z')

    # Number of original objects is equal to the number of rows plus 1.
    n = Z.shape[0] + 1

    # Create a list full of None's to store the node objects
    d = [None] * (n * 2 - 1)

    # Create the nodes corresponding to the n original objects.
    for i in range(0, n):
        d[i] = ClusterNode(i)

    nd = None
    # 遍历矩阵 Z 的每一行
    for i in range(Z.shape[0]):
        # 获取当前行数据
        row = Z[i, :]

        # 计算左子节点的索引 fi 和右子节点的索引 fj
        fi = int_floor(row[0], xp)
        fj = int_floor(row[1], xp)

        # 检查 fi 和 fj 是否超出索引范围 i + n，如果是则抛出数值错误异常
        if fi > i + n:
            raise ValueError(('Corrupt matrix Z. Index to derivative cluster '
                              'is used before it is formed. See row %d, '
                              'column 0') % fi)
        if fj > i + n:
            raise ValueError(('Corrupt matrix Z. Index to derivative cluster '
                              'is used before it is formed. See row %d, '
                              'column 1') % fj)

        # 创建一个 ClusterNode 对象 nd，表示当前节点
        nd = ClusterNode(i + n, d[fi], d[fj], row[2])
        #                ^ id   ^ left ^ right ^ dist

        # 检查当前节点的 count 值是否与 row[3] 的值相符，如果不符则抛出数值错误异常
        if row[3] != nd.count:
            raise ValueError(('Corrupt matrix Z. The count Z[%d,3] is '
                              'incorrect.') % i)

        # 将当前节点 nd 加入到字典 d 中
        d[n + i] = nd

    # 如果 rd 为 True，则返回一个元组 (nd, d)，否则返回 nd
    if rd:
        return (nd, d)
    else:
        return nd
    """
    Calculate the cophenetic distances between each observation in
    the hierarchical clustering defined by the linkage ``Z``.

    Suppose ``p`` and ``q`` are original observations in
    disjoint clusters ``s`` and ``t``, respectively and
    ``s`` and ``t`` are joined by a direct parent cluster
    ``u``. The cophenetic distance between observations
    ``i`` and ``j`` is simply the distance between
    clusters ``s`` and ``t``.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix. See
        `linkage` for more information on the return structure and
        algorithm.
    Y : ndarray, optional
        Ignored parameter; maintained for backward compatibility.

    Returns
    -------
    res : ndarray
        The cophenetic distances.

    Notes
    -----
    The cophenetic distance between two observations is defined as the
    distance between the two clusters to which the observations belong
    at the time they are first combined.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import linkage, cophenet
    >>> X = [[0], [1], [3], [6], [7], [8]]
    >>> Z = linkage(X, 'ward')
    >>> cophenet(Z)
    array([1., 2., 3., 3., 4., 3., 4., 5.])
    """
    # Z: ndarray
    #   表示作为数组编码的层次聚类（参见`linkage`函数）。

    # Y: ndarray（可选）
    #   计算由链接矩阵`Z`定义的层次聚类的科菲尼特相关系数``c``。
    #   `Y`是从中生成`Z`的一组包含距离矩阵。

    # Returns
    # -------
    # c: ndarray
    #   科菲尼特相关距离（如果传入`Y`）。
    # d: ndarray
    #   紧凑形式的科菲尼特距离矩阵。第`ij`个条目是原始观察`i`和`j`之间的科菲尼特距离。

    # See Also
    # --------
    # linkage:
    #   关于链接矩阵的描述。
    # scipy.spatial.distance.squareform:
    #   将紧凑形式的矩阵转换为正方形矩阵的方法。

    # Examples
    # --------
    # >>> from scipy.cluster.hierarchy import single, cophenet
    # >>> from scipy.spatial.distance import pdist, squareform

    # 给定数据集`X`和链接矩阵`Z`，两点之间的科菲尼特距离是每个点的最大两个不同聚类之间的距离：

    # >>> X = [[0, 0], [0, 1], [1, 0],
    # ...      [0, 4], [0, 3], [1, 4],
    # ...      [4, 0], [3, 0], [4, 1],
    # ...      [4, 4], [3, 4], [4, 3]]

    # `X`对应于以下数据集：
    #
    #     x x    x x
    #     x        x
    #
    #     x        x
    #     x x    x x

    # >>> Z = single(pdist(X))
    # >>> Z
    # array([[ 0.,  1.,  1.,  2.],
    #        [ 2., 12.,  1.,  3.],
    #        [ 3.,  4.,  1.,  2.],
    #        [ 5., 14.,  1.,  3.],
    #        [ 6.,  7.,  1.,  2.],
    #        [ 8., 16.,  1.,  3.],
    #        [ 9., 10.,  1.,  2.],
    #        [11., 18.,  1.,  3.],
    #        [13., 15.,  2.,  6.],
    #        [17., 20.,  2.,  9.],
    #        [19., 21.,  2., 12.]])
    #
    # >>> cophenet(Z)
    # array([1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2.,
    #        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 2., 2.,
    #        2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
    #        1., 1., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 1., 1., 1.])

    # `scipy.cluster.hierarchy.cophenet`方法的输出以紧凑形式表示。
    # 我们可以使用`scipy.spatial.distance.squareform`来将输出看作是一个常规矩阵
    # （其中每个元素``ij``表示``X``中每对点``i``、``j``之间的科菲尼特距离）：

    # >>> squareform(cophenet(Z))
    # 创建一个二维数组，表示点之间的距离矩阵
    array([[0., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [1., 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [1., 1., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 0., 1., 1., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 1., 0., 1., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 1., 1., 0., 2., 2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 0., 1., 1., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 1., 0., 1., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 1., 1., 0., 2., 2., 2.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 1., 1.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0., 1.],
           [2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 0.]])
    
    # 在这个例子中，X 中非常接近的点（即在同一角落）的科菲尼系数距离为1。
    # 对于其他点对，因为它们位于不同角落的聚类中，所以它们之间的距离是2。
    # 因此，这些聚类之间的距离会更大。
    
    """
    xp = array_namespace(Z, Y)
    # 使用 array_namespace 函数将 Z 和 Y 转换为 xp 命名空间中的数组表示
    
    # 确保 Z 是 float64 类型的 C 连续数组。因为 Cython 代码不能处理步幅。
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    # 将 Z 转换为指定的数组表示，使用 C 顺序，数据类型为 float64
    
    is_valid_linkage(Z, throw=True, name='Z')
    # 检查 Z 是否为有效的链接矩阵，如果无效则抛出异常，名称为 'Z'
    
    n = Z.shape[0] + 1
    # 计算链接矩阵 Z 的行数加一，得到 n 的值
    
    zz = np.zeros((n * (n-1)) // 2, dtype=np.float64)
    # 创建一个长度为 n*(n-1)/2 的全零数组，数据类型为 float64，用于存储科菲尼系数距离
    
    Z = np.asarray(Z)
    # 将 Z 转换为 NumPy 数组
    
    _hierarchy.cophenetic_distances(Z, zz, int(n))
    # 计算链接矩阵 Z 的科菲尼系数距离，存储在 zz 中，n 需要转换为整数类型
    
    zz = xp.asarray(zz)
    # 将 zz 转换为 xp 命名空间中的数组表示
    
    if Y is None:
        return zz
    # 如果 Y 为 None，则直接返回 zz
    
    Y = _asarray(Y, order='C', xp=xp)
    # 将 Y 转换为指定的数组表示，使用 C 顺序，命名空间为 xp
    
    distance.is_valid_y(Y, throw=True, name='Y')
    # 检查 Y 是否为有效的距离矩阵，如果无效则抛出异常，名称为 'Y'
    
    z = xp.mean(zz)
    # 计算 zz 数组的均值，使用 xp 命名空间
    
    y = xp.mean(Y)
    # 计算 Y 数组的均值，使用 xp 命名空间
    
    Yy = Y - y
    # Y 中每个元素减去均值 y
    
    Zz = zz - z
    # zz 中每个元素减去均值 z
    
    numerator = (Yy * Zz)
    # 计算分子，即 Yy 和 Zz 对应位置元素的乘积
    
    denomA = Yy**2
    # 计算分母A，即 Yy 的平方
    
    denomB = Zz**2
    # 计算分母B，即 Zz 的平方
    
    c = xp.sum(numerator) / xp.sqrt(xp.sum(denomA) * xp.sum(denomB))
    # 计算最终的相关系数 c，使用 xp 命名空间进行计算
    
    return (c, zz)
    # 返回相关系数 c 和 zz 数组
# 定义一个函数，用于计算链接矩阵的不一致性统计信息
def inconsistent(Z, d=2):
    r"""
    Calculate inconsistency statistics on a linkage matrix.

    Parameters
    ----------
    Z : ndarray
        The :math:`(n-1)` by 4 matrix encoding the linkage (hierarchical
        clustering).  See `linkage` documentation for more information on its
        form.
    d : int, optional
        The number of links up to `d` levels below each non-singleton cluster.

    Returns
    -------
    R : ndarray
        A :math:`(n-1)` by 4 matrix where the ``i``'th row contains the link
        statistics for the non-singleton cluster ``i``. The link statistics are
        computed over the link heights for links :math:`d` levels below the
        cluster ``i``. ``R[i,0]`` and ``R[i,1]`` are the mean and standard
        deviation of the link heights, respectively; ``R[i,2]`` is the number
        of links included in the calculation; and ``R[i,3]`` is the
        inconsistency coefficient,

        .. math:: \frac{\mathtt{Z[i,2]} - \mathtt{R[i,0]}} {R[i,1]}

    Notes
    -----
    This function behaves similarly to the MATLAB(TM) ``inconsistent``
    function.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import inconsistent, linkage
    >>> from matplotlib import pyplot as plt
    >>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
    >>> Z = linkage(X, 'ward')
    >>> print(Z)
    [[ 5.          6.          0.          2.        ]
     [ 2.          7.          0.          2.        ]
     [ 0.          4.          1.          2.        ]
     [ 1.          8.          1.15470054  3.        ]
     [ 9.         10.          2.12132034  4.        ]
     [ 3.         12.          4.11096096  5.        ]
     [11.         13.         14.07183949  8.        ]]
    >>> inconsistent(Z)
    array([[ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 1.        ,  0.        ,  1.        ,  0.        ],
           [ 0.57735027,  0.81649658,  2.        ,  0.70710678],
           [ 1.04044011,  1.06123822,  3.        ,  1.01850858],
           [ 3.11614065,  1.40688837,  2.        ,  0.70710678],
           [ 6.44583366,  6.76770586,  3.        ,  1.12682288]])

    """
    # 将 Z 转换为适当的数组格式，确保使用正确的数据类型和内存布局
    xp = array_namespace(Z)
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    # 检查 Z 是否是有效的链接矩阵，如果不是会抛出异常
    is_valid_linkage(Z, throw=True, name='Z')

    # 检查参数 d 是否为非负整数，如果不是则抛出 ValueError
    if (not d == np.floor(d)) or d < 0:
        raise ValueError('The second argument d must be a nonnegative '
                         'integer value.')

    # 计算样本点数 n
    n = Z.shape[0] + 1
    # 创建一个全零数组 R，用于存储统计结果，数据类型为 float64
    R = np.zeros((n - 1, 4), dtype=np.float64)

    # 将 Z 转换为 np.ndarray 类型
    Z = np.asarray(Z)
    # 调用 C 语言实现的函数 _hierarchy.inconsistent 计算不一致性统计信息并存储到 R 中
    _hierarchy.inconsistent(Z, R, int(n), int(d))
    # 将 R 转换为 xp 数组格式
    R = xp.asarray(R)
    # 返回计算结果 R
    return R


def from_mlab_linkage(Z):
    """
    Convert a linkage matrix generated by MATLAB(TM) to a new
    linkage matrix compatible with this module.

    """
    """
    The conversion does two things:

     * the indices are converted from ``1..N`` to ``0..(N-1)`` form,
       and

     * a fourth column ``Z[:,3]`` is added where ``Z[i,3]`` represents the
       number of original observations (leaves) in the non-singleton
       cluster ``i``.

    This function is useful when loading in linkages from legacy data
    files generated by MATLAB.

    Parameters
    ----------
    Z : ndarray
        A linkage matrix generated by MATLAB(TM).

    Returns
    -------
    ZS : ndarray
        A linkage matrix compatible with ``scipy.cluster.hierarchy``.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    to_mlab_linkage : transform from SciPy to MATLAB format.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster.hierarchy import ward, from_mlab_linkage

    Given a linkage matrix in MATLAB format ``mZ``, we can use
    `scipy.cluster.hierarchy.from_mlab_linkage` to import
    it into SciPy format:

    >>> mZ = np.array([[1, 2, 1], [4, 5, 1], [7, 8, 1],
    ...                [10, 11, 1], [3, 13, 1.29099445],
    ...                [6, 14, 1.29099445],
    ...                [9, 15, 1.29099445],
    ...                [12, 16, 1.29099445],
    ...                [17, 18, 5.77350269],
    ...                [19, 20, 5.77350269],
    ...                [21, 22,  8.16496581]])

    >>> Z = from_mlab_linkage(mZ)
    >>> Z
    array([[  0.        ,   1.        ,   1.        ,   2.        ],
           [  3.        ,   4.        ,   1.        ,   2.        ],
           [  6.        ,   7.        ,   1.        ,   2.        ],
           [  9.        ,  10.        ,   1.        ,   2.        ],
           [  2.        ,  12.        ,   1.29099445,   3.        ],
           [  5.        ,  13.        ,   1.29099445,   3.        ],
           [  8.        ,  14.        ,   1.29099445,   3.        ],
           [ 11.        ,  15.        ,   1.29099445,   3.        ],
           [ 16.        ,  17.        ,   5.77350269,   6.        ],
           [ 18.        ,  19.        ,   5.77350269,   6.        ],
           [ 20.        ,  21.        ,   8.16496581,  12.        ]])

    As expected, the linkage matrix ``Z`` returned includes an
    additional column counting the number of original samples in
    each cluster. Also, all cluster indices are reduced by 1
    (MATLAB format uses 1-indexing, whereas SciPy uses 0-indexing).

    """

    # Convert the input array to the appropriate array namespace
    xp = array_namespace(Z)
    # Convert Z to a numpy array with specific dtype and order
    Z = _asarray(Z, dtype=xp.float64, order='C', xp=xp)
    # Get the shape of Z
    Zs = Z.shape

    # If Z is empty or contains only one element which is zero, return a copy of Z
    if len(Zs) == 0 or (len(Zs) == 1 and Zs[0] == 0):
        return copy(Z, xp=xp)

    # If Z does not have exactly two dimensions, raise a ValueError
    if len(Zs) != 2:
        raise ValueError("The linkage array must be rectangular.")

    # If Z has zero rows, return a copy of Z
    if Zs[0] == 0:
        return copy(Z, xp=xp)
    # 检查 Z 矩阵的第一列和第二列是否符合特定的值范围条件，如果不符合，则抛出数值错误异常
    if xp.min(Z[:, 0:2]) != 1.0 and xp.max(Z[:, 0:2]) != 2 * Zs[0]:
        raise ValueError('The format of the indices is not 1..N')

    # 将 Z 矩阵的第一列和第二列减去 1.0，并与第三列拼接，形成新的 Zpart 矩阵
    Zpart = xp.concat((Z[:, 0:2] - 1.0, Z[:, 2:]), axis=1)
    
    # 创建一个全零数组 CS，用于存储聚类大小信息
    CS = np.zeros((Zs[0],), dtype=np.float64)
    
    # 如果使用的是 JAX 库，则需要复制 Zpart 数组，因为 calculate_cluster_sizes 函数不接受只读数组
    if is_jax(xp):
        # 使用 np.array 进行深拷贝，以确保 Zpart 是可写的数组
        Zpart = np.array(Zpart, copy=True)
    else:
        # 否则，将 Zpart 转换为 NumPy 数组
        Zpart = np.asarray(Zpart)
    
    # 调用 _hierarchy.calculate_cluster_sizes 函数计算聚类大小，将结果存储在 CS 数组中
    _hierarchy.calculate_cluster_sizes(Zpart, CS, int(Zs[0]) + 1)
    
    # 将 Zpart 和 CS 水平堆叠，形成最终的结果 res
    res = np.hstack([Zpart, CS.reshape(Zs[0], 1)])
    
    # 将结果 res 转换为 xp 对应的数组类型并返回
    return xp.asarray(res)
def to_mlab_linkage(Z):
    """
    Convert a linkage matrix to a MATLAB(TM) compatible one.

    Converts a linkage matrix ``Z`` generated by the linkage function
    of this module to a MATLAB(TM) compatible one. The return linkage
    matrix has the last column removed and the cluster indices are
    converted to ``1..N`` indexing.

    Parameters
    ----------
    Z : ndarray
        A linkage matrix generated by ``scipy.cluster.hierarchy``.

    Returns
    -------
    to_mlab_linkage : ndarray
        A linkage matrix compatible with MATLAB(TM)'s hierarchical
        clustering functions.

        The return linkage matrix has the last column removed
        and the cluster indices are converted to ``1..N`` indexing.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    from_mlab_linkage : transform from Matlab to SciPy format.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, to_mlab_linkage
    >>> from scipy.spatial.distance import pdist

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    After a linkage matrix ``Z`` has been created, we can use
    `scipy.cluster.hierarchy.to_mlab_linkage` to convert it
    into MATLAB format:

    >>> mZ = to_mlab_linkage(Z)
    >>> mZ
    array([[  1.        ,   2.        ,   1.        ],
           [  4.        ,   5.        ,   1.        ],
           [  7.        ,   8.        ,   1.        ],
           [ 10.        ,  11.        ,   1.        ],
           [  3.        ,  13.        ,   1.29099445],
           [  6.        ,  14.        ,   1.29099445],
           [  9.        ,  15.        ,   1.29099445],
           [ 12.        ,  16.        ,   1.29099445],
           [ 17.        ,  18.        ,   5.77350269],
           [ 19.        ,  20.        ,   5.77350269],
           [ 21.        ,  22.        ,   8.16496581]])

    The new linkage matrix ``mZ`` uses 1-indexing for all the
    clusters (instead of 0-indexing). Also, the last column of
    the original linkage matrix has been dropped.

    """
    xp = array_namespace(Z)  # 使用特定命名空间处理数组，通常是NumPy或类似的数组库
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)  # 将输入的Z转换为NumPy数组，按C顺序（行主序），指定数据类型为float64
    # 获取数组 Z 的形状
    Zs = Z.shape
    # 检查数组 Z 是否为空数组或者形状为 (0,) 的数组
    if len(Zs) == 0 or (len(Zs) == 1 and Zs[0] == 0):
        # 如果满足上述条件，返回 Z 的拷贝（使用相同的计算环境 xp）
        return copy(Z, xp=xp)
    # 检查 linkage 数组 Z 是否符合有效的层次聚类链接条件，如果不符合则抛出异常
    is_valid_linkage(Z, throw=True, name='Z')
    
    # 返回一个新的数组，该数组由 Z 的前两列增加 1.0 后和 Z 的第三列组成，沿着 axis=1 连接
    return xp.concat((Z[:, :2] + 1.0, Z[:, 2:3]), axis=1)
# 定义一个函数用于检查给定的聚类链接是否是单调的
def is_monotonic(Z):
    """
    Return True if the linkage passed is monotonic.

    The linkage is monotonic if for every cluster :math:`s` and :math:`t`
    joined, the distance between them is no less than the distance
    between any previously joined clusters.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix to check for monotonicity.

    Returns
    -------
    b : bool
        A boolean indicating whether the linkage is monotonic.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, ward, is_monotonic
    >>> from scipy.spatial.distance import pdist

    By definition, some hierarchical clustering algorithms - such as
    `scipy.cluster.hierarchy.ward` - produce monotonic assignments of
    samples to clusters; however, this is not always true for other
    hierarchical methods - e.g. `scipy.cluster.hierarchy.median`.

    Given a linkage matrix ``Z`` (as the result of a hierarchical clustering
    method) we can test programmatically whether it has the monotonicity
    property or not, using `scipy.cluster.hierarchy.is_monotonic`:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])
    >>> is_monotonic(Z)
    True

    >>> Z = median(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.        ,  6.        ],
           [16.        , 17.        ,  3.5       ,  6.        ],
           [20.        , 21.        ,  3.25      , 12.        ]])
    >>> is_monotonic(Z)
    False

    Note that this method is equivalent to just verifying that the distances
    """
    # 通过对比相邻聚类间的距离，判断给定的聚类链接矩阵是否满足单调性条件
    return np.all(Z[1:, 2] >= Z[:-1, 2])
    in the third column of the linkage matrix appear in a monotonically
    increasing order.

    """
    # 转换 Z 到指定命名空间的数组表示形式
    xp = array_namespace(Z)
    # 将 Z 转换为 C 风格的数组（行优先），使用指定的数组命名空间 xp
    Z = _asarray(Z, order='c', xp=xp)
    # 检查 Z 是否为有效的聚类链接矩阵，如果不是则抛出异常，使用名称 'Z'
    is_valid_linkage(Z, throw=True, name='Z')

    # 我们期望第 i 个值大于其后继值。
    # 检查 linkage 矩阵的第三列（索引为2），确保其每个元素都大于或等于其前一个元素
    return xp.all(Z[1:, 2] >= Z[:-1, 2])
# 检查给定的不一致性矩阵是否有效
def is_valid_im(R, warning=False, throw=False, name=None):
    """Return True if the inconsistency matrix passed is valid.
    
    它必须是一个大小为n×4的双精度数组。标准差 `R[:,1]` 必须是非负的。
    连接计数 `R[:,2]` 必须是正数，且不大于n-1。

    Parameters
    ----------
    R : ndarray
        要检查有效性的不一致性矩阵。
    warning : bool, optional
        当为True时，如果传入的连接矩阵无效，发出Python警告。
    throw : bool, optional
        当为True时，如果传入的连接矩阵无效，抛出Python异常。
    name : str, optional
        无效连接矩阵的变量名字符串。

    Returns
    -------
    b : bool
        如果不一致性矩阵有效，则返回True。

    See Also
    --------
    linkage : 描述连接矩阵的内容。
    inconsistent : 创建不一致性矩阵。

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, inconsistent, is_valid_im
    >>> from scipy.spatial.distance import pdist

    给定数据集 `X`，我们可以应用聚类方法来获得连接矩阵 `Z`。`scipy.cluster.hierarchy.inconsistent`
    也可以用来获得与此聚类过程相关的不一致性矩阵 `R`：

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> R = inconsistent(Z)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])
    >>> R

    """
    # 定义一个多维数组，表示一个矩阵，每行包含四个浮点数
    array([[1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [1.14549722, 0.20576415, 2.        , 0.70710678],
           [2.78516386, 2.58797734, 3.        , 1.15470054],
           [2.78516386, 2.58797734, 3.        , 1.15470054],
           [6.57065706, 1.38071187, 3.        , 1.15470054]])
    
    Now we can use `scipy.cluster.hierarchy.is_valid_im` to verify that
    ``R`` is correct:
    
    >>> is_valid_im(R)
    True
    
    However, if ``R`` is wrongly constructed (e.g., one of the standard
    deviations is set to a negative value), then the check will fail:
    
    >>> R[-1,1] = R[-1,1] * -1
    >>> is_valid_im(R)
    False
    """
    xp = array_namespace(R)
    # 使用给定的数组命名空间创建一个数组实例
    R = _asarray(R, order='c', xp=xp)
    # 将输入转换为数组表示，使用C顺序存储，使用指定的数组命名空间
    valid = True
    name_str = "%r " % name if name else ''
    try:
        # 检查矩阵的数据类型是否为float64
        if R.dtype != xp.float64:
            raise TypeError('Inconsistency matrix %smust contain doubles '
                            '(double).' % name_str)
        # 检查矩阵的维度是否为2
        if len(R.shape) != 2:
            raise ValueError('Inconsistency matrix %smust have shape=2 (i.e. '
                             'be two-dimensional).' % name_str)
        # 检查矩阵的列数是否为4
        if R.shape[1] != 4:
            raise ValueError('Inconsistency matrix %smust have 4 columns.' %
                             name_str)
        # 检查矩阵的行数是否至少为1
        if R.shape[0] < 1:
            raise ValueError('Inconsistency matrix %smust have at least one '
                             'row.' % name_str)
        # 检查矩阵中第一列是否包含负值
        if xp.any(R[:, 0] < 0):
            raise ValueError('Inconsistency matrix %scontains negative link '
                             'height means.' % name_str)
        # 检查矩阵中第二列是否包含负值
        if xp.any(R[:, 1] < 0):
            raise ValueError('Inconsistency matrix %scontains negative link '
                             'height standard deviations.' % name_str)
        # 检查矩阵中第三列是否包含负值
        if xp.any(R[:, 2] < 0):
            raise ValueError('Inconsistency matrix %scontains negative link '
                             'counts.' % name_str)
    except Exception as e:
        # 如果发生异常，根据参数设置决定是否抛出异常或警告
        if throw:
            raise
        if warning:
            _warning(str(e))
        valid = False
    
    return valid
def is_valid_linkage(Z, warning=False, throw=False, name=None):
    """
    Check the validity of a linkage matrix.

    A linkage matrix is valid if it is a 2-D array (type double)
    with :math:`n` rows and 4 columns. The first two columns must contain
    indices between 0 and :math:`2n-1`. For a given row ``i``, the following
    two expressions have to hold:

    .. math::

        0 \\leq \\mathtt{Z[i,0]} \\leq i+n-1
        0 \\leq Z[i,1] \\leq i+n-1

    I.e., a cluster cannot join another cluster unless the cluster being joined
    has been generated.

    Parameters
    ----------
    Z : array_like
        Linkage matrix.
    warning : bool, optional
        When True, issues a Python warning if the linkage
        matrix passed is invalid.
    throw : bool, optional
        When True, throws a Python exception if the linkage
        matrix passed is invalid.
    name : str, optional
        This string refers to the variable name of the invalid
        linkage matrix.

    Returns
    -------
    b : bool
        True if the inconsistency matrix is valid.

    See Also
    --------
    linkage: for a description of what a linkage matrix is.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, is_valid_linkage
    >>> from scipy.spatial.distance import pdist

    All linkage matrices generated by the clustering methods in this module
    will be valid (i.e., they will have the appropriate dimensions and the two
    required expressions will hold for all the rows).

    We can check this using `scipy.cluster.hierarchy.is_valid_linkage`:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])
    >>> is_valid_linkage(Z)
    True

    However, if we create a linkage matrix in a wrong way - or if we modify
    a valid one in a way that any of the required expressions don't hold
    anymore, then the check will fail:

    >>> Z[3][1] = 20    # the cluster number 20 is not defined at this point
    >>> is_valid_linkage(Z)
    False

    """
    xp = array_namespace(Z)  # 使用array_namespace函数获取数组的执行命名空间
    Z = _asarray(Z, order='c', xp=xp)  # 将Z转换为C顺序的数组，并使用指定的命名空间xp
    valid = True  # 初始化valid变量为True，用于记录链接矩阵的有效性
    name_str = "%r " % name if name else ''  # 如果提供了name参数，则将其格式化为字符串name_str；否则为空字符串
    try:
        # 检查 Z 的数据类型是否为 float64，如果不是则抛出类型错误
        if Z.dtype != xp.float64:
            raise TypeError('Linkage matrix %smust contain doubles.' % name_str)
        
        # 检查 Z 的维度是否为 2，如果不是则抛出数值错误
        if len(Z.shape) != 2:
            raise ValueError('Linkage matrix %smust have shape=2 (i.e. be '
                             'two-dimensional).' % name_str)
        
        # 检查 Z 的第二维度是否为 4，如果不是则抛出数值错误
        if Z.shape[1] != 4:
            raise ValueError('Linkage matrix %smust have 4 columns.' % name_str)
        
        # 检查 Z 的第一维度是否为至少 2，如果不是则抛出数值错误
        if Z.shape[0] == 0:
            raise ValueError('Linkage must be computed on at least two '
                             'observations.')
        
        # 获取 Z 的第一维度大小
        n = Z.shape[0]
        
        # 如果 n 大于 1，则进一步检查以下条件
        if n > 1:
            # 如果 Z[:, 0] 或 Z[:, 1] 存在小于 0 的值，则抛出数值错误
            if (xp.any(Z[:, 0] < 0) or xp.any(Z[:, 1] < 0)):
                raise ValueError('Linkage %scontains negative indices.' %
                                 name_str)
            
            # 如果 Z[:, 2] 存在小于 0 的值，则抛出数值错误
            if xp.any(Z[:, 2] < 0):
                raise ValueError('Linkage %scontains negative distances.' %
                                 name_str)
            
            # 如果 Z[:, 3] 存在小于 0 的值，则抛出数值错误
            if xp.any(Z[:, 3] < 0):
                raise ValueError('Linkage %scontains negative counts.' %
                                 name_str)
        
        # 检查是否存在在形成之前就使用了非单例集群，如果是则抛出数值错误
        if _check_hierarchy_uses_cluster_before_formed(Z):
            raise ValueError('Linkage %suses non-singleton cluster before '
                             'it is formed.' % name_str)
        
        # 检查是否存在使用了超过一次的相同集群，如果是则抛出数值错误
        if _check_hierarchy_uses_cluster_more_than_once(Z):
            raise ValueError('Linkage %suses the same cluster more than once.'
                             % name_str)
    
    # 捕获所有异常
    except Exception as e:
        # 如果设置了 throw 参数，则继续抛出异常
        if throw:
            raise
        
        # 如果设置了 warning 参数，则记录警告信息
        if warning:
            _warning(str(e))
        
        # 将 valid 设为 False，表示验证未通过
        valid = False
    
    # 返回验证结果 valid
    return valid
# 检查层次聚类是否在形成之前使用了集群
def _check_hierarchy_uses_cluster_before_formed(Z):
    # 计算集群数目
    n = Z.shape[0] + 1
    # 遍历所有链接
    for i in range(0, n - 1):
        # 检查链接是否超出了已有集群数目范围
        if Z[i, 0] >= n + i or Z[i, 1] >= n + i:
            return True
    return False


# 检查层次聚类是否使用了超过一次的集群
def _check_hierarchy_uses_cluster_more_than_once(Z):
    # 计算集群数目
    n = Z.shape[0] + 1
    # 记录已经选择的集群
    chosen = set()
    # 遍历所有链接
    for i in range(0, n - 1):
        # 检查集群是否被多次使用
        used_more_than_once = (
            (float(Z[i, 0]) in chosen)
            or (float(Z[i, 1]) in chosen)
            or Z[i, 0] == Z[i, 1]
        )
        if used_more_than_once:
            return True
        chosen.add(float(Z[i, 0]))
        chosen.add(float(Z[i, 1]))
    return False


# 检查层次聚类是否未使用所有集群
def _check_hierarchy_not_all_clusters_used(Z):
    # 计算集群数目
    n = Z.shape[0] + 1
    # 记录已经选择的集群
    chosen = set()
    # 遍历所有链接
    for i in range(0, n - 1):
        chosen.add(int(Z[i, 0]))
        chosen.add(int(Z[i, 1]))
    # 必须被选择的集群
    must_chosen = set(range(0, 2 * n - 2))
    # 检查是否有集群未被使用
    return len(must_chosen.difference(chosen)) > 0


# 返回链接矩阵中的原始观测数目
def num_obs_linkage(Z):
    """
    Parameters
    ----------
    Z : ndarray
        要执行操作的链接矩阵。

    Returns
    -------
    n : int
        链接中的原始观测数。

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, num_obs_linkage
    >>> from scipy.spatial.distance import pdist

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))

    ``Z`` 是使用 ``X`` 数据集进行 Ward 聚类方法后得到的链接矩阵，其中包含 12 个数据点。

    >>> num_obs_linkage(Z)
    12

    """
    # 从数组命名空间创建 xp
    xp = array_namespace(Z)
    # 将 Z 转换为数组，按照指定顺序进行排列
    Z = _asarray(Z, order='c', xp=xp)
    # 检查 Z 是否为有效的链接矩阵
    is_valid_linkage(Z, throw=True, name='Z')
    # 返回原始观测数目
    return (Z.shape[0] + 1)


# 检查链接矩阵和距离矩阵之间是否存在对应关系
def correspond(Z, Y):
    """
    Parameters
    ----------
    Z : array_like
        要检查对应关系的链接矩阵。
    Y : array_like
        要检查对应关系的压缩距离矩阵。

    Returns
    -------
    b : bool
        指示链接矩阵和距离矩阵是否可能相对应的布尔值。

    See Also
    --------
    linkage : 链接矩阵的描述。

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, correspond
    >>> from scipy.spatial.distance import pdist

    此方法可用于检查给定链接矩阵 ``Z`` 是否是通过对数据集 ``X`` 应用聚类方法得到的：

    """

    # 返回链接和距离矩阵是否可能相对应的布尔值
    return Z.shape[0] + 1 == Y.shape[0]
    is_valid_linkage(Z, throw=True)
    # 调用函数检查 Z 是否为有效的聚类链接数组，并在无效时抛出异常

    distance.is_valid_y(Y, throw=True)
    # 调用函数检查 Y 是否为有效的距离矩阵，并在无效时抛出异常

    xp = array_namespace(Z, Y)
    # 调用函数创建一个混合命名空间，以便适应 Z 和 Y 的数据类型

    Z = _asarray(Z, order='c', xp=xp)
    # 将 Z 转换为 C 风格的数组，并使用创建的混合命名空间 xp

    Y = _asarray(Y, order='c', xp=xp)
    # 将 Y 转换为 C 风格的数组，并使用创建的混合命名空间 xp

    return distance.num_obs_y(Y) == num_obs_linkage(Z)
    # 返回 Y 中的观测数量是否等于 Z 的链接数目
# 从给定的链接矩阵 Z 中形成扁平聚类
def fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
    """
    Form flat clusters from the hierarchical clustering defined by
    the given linkage matrix.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded with the matrix returned
        by the `linkage` function.
    t : scalar
        For criteria 'inconsistent', 'distance' or 'monocrit',
         this is the threshold to apply when forming flat clusters.
        For 'maxclust' or 'maxclust_monocrit' criteria,
         this would be max number of clusters requested.
    criterion : str, optional
        The criterion to use in forming flat clusters. This can
        be any of the following values:

          ``inconsistent`` :
              If a cluster node and all its
              descendants have an inconsistent value less than or equal
              to `t`, then all its leaf descendants belong to the
              same flat cluster. When no non-singleton cluster meets
              this criterion, every node is assigned to its own
              cluster. (Default)

          ``distance`` :
              Forms flat clusters so that the original
              observations in each flat cluster have no greater a
              cophenetic distance than `t`.

          ``maxclust`` :
              Finds a minimum threshold ``r`` so that
              the cophenetic distance between any two original
              observations in the same flat cluster is no more than
              ``r`` and no more than `t` flat clusters are formed.

          ``monocrit`` :
              Forms a flat cluster from a cluster node c
              with index i when ``monocrit[j] <= t``.

              For example, to threshold on the maximum mean distance
              as computed in the inconsistency matrix R with a
              threshold of 0.8 do::

                  MR = maxRstat(Z, R, 3)
                  fcluster(Z, t=0.8, criterion='monocrit', monocrit=MR)

          ``maxclust_monocrit`` :
              Forms a flat cluster from a
              non-singleton cluster node ``c`` when ``monocrit[i] <=
              r`` for all cluster indices ``i`` below and including
              ``c``. ``r`` is minimized such that no more than ``t``
              flat clusters are formed. monocrit must be
              monotonic. For example, to minimize the threshold t on
              maximum inconsistency values so that no more than 3 flat
              clusters are formed, do::

                  MI = maxinconsts(Z, R)
                  fcluster(Z, t=3, criterion='maxclust_monocrit', monocrit=MI)
    depth : int, optional
        The maximum depth to perform the inconsistency calculation.
        It has no meaning for the other criteria. Default is 2.
    R : ndarray, optional
        The inconsistency matrix to use for the ``'inconsistent'``
        criterion. This matrix is computed if not provided.
    """
    pass
    monocrit : ndarray, optional
        An array of length n-1. `monocrit[i]` is the
        statistics upon which non-singleton i is thresholded. The
        monocrit vector must be monotonic, i.e., given a node c with
        index i, for all node indices j corresponding to nodes
        below c, ``monocrit[i] >= monocrit[j]``.

    Returns
    -------
    fcluster : ndarray
        An array of length ``n``. ``T[i]`` is the flat cluster number to
        which original observation ``i`` belongs.

    See Also
    --------
    linkage : for information about hierarchical clustering methods work.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, fcluster
    >>> from scipy.spatial.distance import pdist

    All cluster linkage methods - e.g., `scipy.cluster.hierarchy.ward`
    generate a linkage matrix ``Z`` as their output:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))

    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    This matrix represents a dendrogram, where the first and second elements
    are the two clusters merged at each step, the third element is the
    distance between these clusters, and the fourth element is the size of
    the new cluster - the number of original data points included.

    `scipy.cluster.hierarchy.fcluster` can be used to flatten the
    dendrogram, obtaining as a result an assignation of the original data
    points to single clusters.

    This assignation mostly depends on a distance threshold ``t`` - the maximum
    inter-cluster distance allowed:

    >>> fcluster(Z, t=0.9, criterion='distance')
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)

    >>> fcluster(Z, t=1.1, criterion='distance')
    array([1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8], dtype=int32)

    >>> fcluster(Z, t=3, criterion='distance')
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

    >>> fcluster(Z, t=9, criterion='distance')
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

    In the first case, the threshold ``t`` is too small to allow any two
    samples in the data to form a cluster, so 12 different clusters are
    returned.
    # 将输入数组 Z 转换为 array_namespace 中定义的数组类型
    xp = array_namespace(Z)
    
    # 将 Z 转换为 ndarray 对象，确保其顺序为 C 风格，数据类型为 float64，使用 xp 中的数组类型
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    
    # 验证 Z 是否为有效的层次聚类结果，如果不是则抛出异常
    is_valid_linkage(Z, throw=True, name='Z')
    
    # 计算节点数量 n，层次聚类结果中的节点数为 Z 的行数加 1
    n = Z.shape[0] + 1
    
    # 创建一个全零数组 T，用于存储聚类结果，数据类型为整型
    T = np.zeros((n,), dtype='i')
    
    # 如果指定了 monocrit 参数，则将其转换为 ndarray 对象，数据类型为 float64
    if monocrit is not None:
        monocrit = np.asarray(monocrit, order='C', dtype=np.float64)
    
    # 将 Z 和 monocrit 转换为 ndarray 对象
    Z = np.asarray(Z)
    monocrit = np.asarray(monocrit)
    
    # 根据不同的聚类形成标准 criterion 进行不同的处理
    if criterion == 'inconsistent':
        # 如果 R 为 None，则计算层次聚类的不一致性
        if R is None:
            R = inconsistent(Z, depth)
        else:
            # 否则将 R 转换为 ndarray 对象，确保其顺序为 C 风格，数据类型为 float64，使用 xp 中的数组类型
            R = _asarray(R, order='C', dtype=xp.float64, xp=xp)
            # 验证 R 是否为有效的不一致性矩阵，如果不是则抛出异常
            is_valid_im(R, throw=True, name='R')
            # 使用给定的 Z 和 R 计算层次聚类
            R = np.asarray(R)
        # 调用 C 代码执行层次聚类，传入 Z, R, T, t, n 参数
        _hierarchy.cluster_in(Z, R, T, float(t), int(n))
    elif criterion == 'distance':
        # 使用 Z 计算层次聚类，传入 Z, T, t, n 参数
        _hierarchy.cluster_dist(Z, T, float(t), int(n))
    elif criterion == 'maxclust':
        # 使用 Z 计算最大聚类数的层次聚类，传入 Z, T, n, t 参数
        _hierarchy.cluster_maxclust_dist(Z, T, int(n), t)
    elif criterion == 'monocrit':
        # 使用 Z 和 monocrit 计算层次聚类，传入 Z, monocrit, T, t, n 参数
        _hierarchy.cluster_monocrit(Z, monocrit, T, float(t), int(n))
    elif criterion == 'maxclust_monocrit':
        # 使用 Z 和 monocrit 计算最大聚类数的层次聚类，传入 Z, monocrit, T, n, t 参数
        _hierarchy.cluster_maxclust_monocrit(Z, monocrit, T, int(n), int(t))
    else:
        # 若 criterion 不是以上任何一种，则抛出 ValueError 异常
        raise ValueError('Invalid cluster formation criterion: %s' % str(criterion))
    
    # 返回 T 转换为 xp 中定义的数组类型的数组对象
    return xp.asarray(T)
# 定义函数 fclusterdata，用于对给定的观测数据进行聚类
def fclusterdata(X, t, criterion='inconsistent',
                 metric='euclidean', depth=2, method='single', R=None):
    """
    Cluster observation data using a given metric.

    Clusters the original observations in the n-by-m data
    matrix X (n observations in m dimensions), using the euclidean
    distance metric to calculate distances between original observations,
    performs hierarchical clustering using the single linkage algorithm,
    and forms flat clusters using the inconsistency method with `t` as the
    cut-off threshold.

    A 1-D array ``T`` of length ``n`` is returned. ``T[i]`` is
    the index of the flat cluster to which the original observation ``i``
    belongs.

    Parameters
    ----------
    X : (N, M) ndarray
        N by M data matrix with N observations in M dimensions.
    t : scalar
        For criteria 'inconsistent', 'distance' or 'monocrit',
         this is the threshold to apply when forming flat clusters.
        For 'maxclust' or 'maxclust_monocrit' criteria,
         this would be max number of clusters requested.
    criterion : str, optional
        Specifies the criterion for forming flat clusters. Valid
        values are 'inconsistent' (default), 'distance', or 'maxclust'
        cluster formation algorithms. See `fcluster` for descriptions.
    metric : str or function, optional
        The distance metric for calculating pairwise distances. See
        ``distance.pdist`` for descriptions and linkage to verify
        compatibility with the linkage method.
    depth : int, optional
        The maximum depth for the inconsistency calculation. See
        `inconsistent` for more information.
    method : str, optional
        The linkage method to use (single, complete, average,
        weighted, median centroid, ward). See `linkage` for more
        information. Default is "single".
    R : ndarray, optional
        The inconsistency matrix. It will be computed if necessary
        if it is not passed.

    Returns
    -------
    fclusterdata : ndarray
        A vector of length n. T[i] is the flat cluster number to
        which original observation i belongs.

    See Also
    --------
    scipy.spatial.distance.pdist : pairwise distance metrics

    Notes
    -----
    This function is similar to the MATLAB function ``clusterdata``.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import fclusterdata

    This is a convenience method that abstracts all the steps to perform in a
    typical SciPy's hierarchical clustering workflow.

    * Transform the input data into a condensed matrix with
      `scipy.spatial.distance.pdist`.

    * Apply a clustering method.

    * Obtain flat clusters at a user defined distance threshold ``t`` using
      `scipy.cluster.hierarchy.fcluster`.

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> fclusterdata(X, t=1)
    """
    # 在给定数据 X 上计算成对距离，使用指定的距离度量 metric
    pdist_result = distance.pdist(X, metric)

    # 使用层次聚类方法 method 对成对距离矩阵进行聚类
    Z = linkage(pdist_result, method=method)

    # 根据指定的准则 criterion 和阈值 t，使用 `fcluster` 获取平坦的聚类
    T = fcluster(Z, t, criterion=criterion, depth=depth, R=R)

    # 返回平坦聚类结果向量 T
    return T
    # 创建一个名为 `xp` 的变量，将其设置为 `array_namespace(X)` 的结果，可能是一个数组命名空间的操作。
    xp = array_namespace(X)
    
    # 将变量 `X` 转换为一个 `xp.float64` 类型的二维数组，使用 C 风格的存储顺序。
    X = _asarray(X, order='C', dtype=xp.float64, xp=xp)
    
    # 检查 `X` 的维度是否为二维，如果不是则抛出类型错误。
    if X.ndim != 2:
        raise TypeError('The observation matrix X must be an n by m array.')
    
    # 计算 `X` 中数据点之间的成对距离，并存储在 `Y` 中，使用指定的距离度量 `metric`。
    Y = distance.pdist(X, metric=metric)
    
    # 将 `Y` 转换为 `xp` 数组类型。
    Y = xp.asarray(Y)
    
    # 根据距离数组 `Y` 创建层次聚类的连接矩阵 `Z`，使用指定的聚类方法 `method`。
    Z = linkage(Y, method=method)
    
    # 如果 `R` 参数为 None，则计算层次聚类的不一致性统计量，深度为 `depth`。
    if R is None:
        R = inconsistent(Z, d=depth)
    else:
        # 否则将 `R` 转换为 `xp` 数组类型。
        R = _asarray(R, order='c', xp=xp)
    
    # 根据给定的连接矩阵 `Z` 和阈值 `t` 进行层次聚类并返回聚类结果 `T`。
    T = fcluster(Z, criterion=criterion, depth=depth, R=R, t=t)
    
    # 返回层次聚类的结果 `T`。
    return T
# 导入需要的库和函数
def _plot_dendrogram(icoords, dcoords, ivl, p, n, mh, orientation,
                     no_labels, color_list, leaf_font_size=None,
                     leaf_rotation=None, contraction_marks=None,
                     ax=None, above_threshold_color='C0'):
    # 在这里导入 matplotlib 库，仅在需要绘制树状图时导入，如果导入失败则抛出相关错误信息。
    # 此处函数定义了绘制树状图的各种参数和选项，具体功能和使用方法可以参考相关文档和示例。

_dtextsizes = {20: 12, 30: 10, 50: 8, 85: 6, np.inf: 5}
_drotation = {20: 0, 40: 45, np.inf: 90}
_dtextsortedkeys = list(_dtextsizes.keys())
_dtextsortedkeys.sort()
_drotationsortedkeys = list(_drotation.keys())
_drotationsortedkeys.sort()


def _remove_dups(L):
    """
    从列表中移除重复元素，并保持原始元素的顺序。

    使用一个集合来实现去重操作，同时维护元素的原始顺序。
    """
    seen_before = set()
    L2 = []
    for i in L:
        if i not in seen_before:
            seen_before.add(i)
            L2.append(i)
    return L2


def _get_tick_text_size(p):
    """
    根据叶节点数量 `p` 返回对应的文本大小。

    根据全局变量 `_dtextsizes` 中的设定，确定叶节点数量所对应的文本大小。
    """
    for k in _dtextsortedkeys:
        if p <= k:
            return _dtextsizes[k]


def _get_tick_rotation(p):
    """
    根据叶节点数量 `p` 返回对应的文本旋转角度。

    根据全局变量 `_drotation` 中的设定，确定叶节点数量所对应的文本旋转角度。
    """
    for k in _drotationsortedkeys:
        if p <= k:
            return _drotation[k]


def leaves_list(Z):
    """
    返回叶节点 id 的列表。

    返回的列表对应于观察向量索引在树中从左到右的顺序。Z 是一个链接矩阵。

    Parameters
    ----------
    Z : ndarray
        编码为矩阵的层次聚类。`Z` 是一个链接矩阵。详见 `linkage` 获取更多信息。

    Returns
    -------
    leaves_list : ndarray
        叶节点 id 的列表。

    See Also
    --------
    dendrogram : 关于树状图结构的信息。

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
    >>> from scipy.spatial.distance import pdist
    >>> from matplotlib import pyplot as plt

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))

    链接矩阵 ``Z`` 表示一个树状图，即编码了执行聚类的结构的树。
    `scipy.cluster.hierarchy.leaves_list` 显示了 `X` 数据集中索引与树状图中叶子之间的映射：

    >>> leaves_list(Z)
    array([ 2,  0,  1,  5,  3,  4,  8,  6,  7, 11,  9, 10], dtype=int32)

    >>> fig = plt.figure(figsize=(25, 10))
    >>> dn = dendrogram(Z)
    >>> plt.show()

    """
    xp = array_namespace(Z)
    Z = _asarray(Z, order='C', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    n = Z.shape[0] + 1
    ML = np.zeros((n,), dtype='i')
    Z = np.asarray(Z)
    _hierarchy.prelist(Z, ML, n)
    return xp.asarray(ML)
    try:
        # 如果提供了轴（ax），则完全不使用 pylab
        if ax is None:
            import matplotlib.pylab
        import matplotlib.patches
        import matplotlib.collections
    except ImportError as e:
        # 抛出 ImportError 如果没有安装 matplotlib 库，同时提供安装指引
        raise ImportError("You must install the matplotlib library to plot "
                          "the dendrogram. Use no_plot=True to calculate the "
                          "dendrogram without plotting.") from e

    if ax is None:
        # 如果没有提供轴（ax），则获取当前轴对象
        ax = matplotlib.pylab.gca()
        # 如果使用 pylab，最后触发绘图更新
        trigger_redraw = True
    else:
        trigger_redraw = False

    # 独立变量绘图宽度
    ivw = len(ivl) * 10
    # 依赖变量绘图高度
    dvw = mh + mh * 0.05

    # 独立变量的刻度位置
    iv_ticks = np.arange(5, len(ivl) * 10 + 5, 10)
    if orientation in ('top', 'bottom'):
        if orientation == 'top':
            # 设置顶部方向的坐标轴范围
            ax.set_ylim([0, dvw])
            ax.set_xlim([0, ivw])
        else:
            # 设置底部方向的坐标轴范围
            ax.set_ylim([dvw, 0])
            ax.set_xlim([0, ivw])

        xlines = icoords
        ylines = dcoords
        if no_labels:
            # 如果不显示标签，则设置无刻度线和标签
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            # 设置 x 轴的刻度位置和标签
            ax.set_xticks(iv_ticks)

            if orientation == 'top':
                # 如果方向是顶部，则设置 x 轴刻度位置在底部
                ax.xaxis.set_ticks_position('bottom')
            else:
                # 如果方向是底部，则设置 x 轴刻度位置在顶部
                ax.xaxis.set_ticks_position('top')

            # 将刻度线设置为不可见，因为它们会遮挡链接
            for line in ax.get_xticklines():
                line.set_visible(False)

            # 计算叶子节点的旋转角度和字体大小
            leaf_rot = (float(_get_tick_rotation(len(ivl)))
                        if (leaf_rotation is None) else leaf_rotation)
            leaf_font = (float(_get_tick_text_size(len(ivl)))
                         if (leaf_font_size is None) else leaf_font_size)
            ax.set_xticklabels(ivl, rotation=leaf_rot, size=leaf_font)

    elif orientation in ('left', 'right'):
        if orientation == 'left':
            # 设置左侧方向的坐标轴范围
            ax.set_xlim([dvw, 0])
            ax.set_ylim([0, ivw])
        else:
            # 设置右侧方向的坐标轴范围
            ax.set_xlim([0, dvw])
            ax.set_ylim([0, ivw])

        xlines = dcoords
        ylines = icoords
        if no_labels:
            # 如果不显示标签，则设置无刻度线和标签
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            # 设置 y 轴的刻度位置和标签
            ax.set_yticks(iv_ticks)

            if orientation == 'left':
                # 如果方向是左侧，则设置 y 轴刻度位置在右侧
                ax.yaxis.set_ticks_position('right')
            else:
                # 如果方向是右侧，则设置 y 轴刻度位置在左侧
                ax.yaxis.set_ticks_position('left')

            # 将刻度线设置为不可见，因为它们会遮挡链接
            for line in ax.get_yticklines():
                line.set_visible(False)

            # 计算叶子节点的字体大小，并根据需要设置旋转角度
            leaf_font = (float(_get_tick_text_size(len(ivl)))
                         if (leaf_font_size is None) else leaf_font_size)

            if leaf_rotation is not None:
                ax.set_yticklabels(ivl, rotation=leaf_rotation, size=leaf_font)
            else:
                ax.set_yticklabels(ivl, size=leaf_font)
    # 使用 collections 来处理颜色分组，确保每个树组合都有单独的图例项，而不是每条线段都有一个图例项。
    colors_used = _remove_dups(color_list)
    # 创建一个空字典，用于将每种颜色映射到对应的线段列表
    color_to_lines = {}
    # 遍历每种颜色，将其初始化为空列表
    for color in colors_used:
        color_to_lines[color] = []
    # 将每条线段按照其颜色分组，存入对应颜色的列表中
    for (xline, yline, color) in zip(xlines, ylines, color_list):
        color_to_lines[color].append(list(zip(xline, yline)))

    # 创建一个空字典，用于将每种颜色映射到对应的 LineCollection 对象
    colors_to_collections = {}
    # 根据每种颜色的线段列表，构造 LineCollection 对象
    for color in colors_used:
        coll = matplotlib.collections.LineCollection(color_to_lines[color],
                                                     colors=(color,))
        colors_to_collections[color] = coll

    # 将颜色阈值以下的所有分组添加到绘图对象中
    for color in colors_used:
        if color != above_threshold_color:
            ax.add_collection(colors_to_collections[color])
    # 如果存在颜色阈值以上的链接分组，则将其放在最后绘制
    if above_threshold_color in colors_to_collections:
        ax.add_collection(colors_to_collections[above_threshold_color])

    # 如果存在收缩标记，则添加椭圆形标记到图中
    if contraction_marks is not None:
        Ellipse = matplotlib.patches.Ellipse
        # 遍历每个收缩标记的坐标，根据方向创建对应的椭圆形标记，并添加到图中
        for (x, y) in contraction_marks:
            if orientation in ('left', 'right'):
                e = Ellipse((y, x), width=dvw / 100, height=1.0)
            else:
                e = Ellipse((x, y), width=1.0, height=dvw / 100)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.5)
            e.set_facecolor('k')

    # 如果需要重新绘制图形，则触发 Matplotlib 的绘图操作
    if trigger_redraw:
        matplotlib.pylab.draw_if_interactive()
# 默认链接线颜色列表，用于指定 dendrogram 中链接线的颜色
_link_line_colors_default = ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9')
# 创建一个可变列表，复制默认颜色列表，用于动态修改链接线颜色
_link_line_colors = list(_link_line_colors_default)

def set_link_color_palette(palette):
    """
    设置用于 dendrogram 的 matplotlib 颜色代码列表。

    注意，这个调色板是全局的（设置一次会影响到后续所有对 `dendrogram` 的调用），
    只影响低于 `color_threshold` 的颜色。

    `dendrogram` 函数还可以通过其 `link_color_func` 关键字接受自定义着色函数，
    这种方式更加灵活且不是全局的。

    Parameters
    ----------
    palette : list of str or None
        matplotlib 颜色代码的列表。颜色代码的顺序决定了在 dendrogram 中进行颜色阈值处理时的循环顺序。

        如果为 ``None``，则重置调色板为其默认值（即 matplotlib 默认颜色 C1 到 C9）。

    Returns
    -------
    None

    See Also
    --------
    dendrogram

    Notes
    -----
    SciPy 0.17.0 版本增加了使用 ``None`` 重置调色板的功能。

    Examples
    --------
    参考 SciPy 文档中的示例
    """

    if palette is None:
        # 如果 palette 为 None，则重置为默认调色板
        palette = _link_line_colors_default
    elif not isinstance(palette, (list, tuple)):
        # 如果 palette 不是列表或元组，则抛出类型错误
        raise TypeError("palette must be a list or tuple")
    _ptypes = [isinstance(p, str) for p in palette]

    if False in _ptypes:
        # 如果 palette 中有非字符串元素，则抛出类型错误
        raise TypeError("all palette list elements must be color strings")

    # 使用全局变量来更新调色板
    global _link_line_colors
    _link_line_colors = palette


def dendrogram(Z, p=30, truncate_mode=None, color_threshold=None,
               get_leaves=True, orientation='top', labels=None,
               count_sort=False, distance_sort=False, show_leaf_counts=True,
               no_plot=False, no_labels=False, leaf_font_size=None,
               leaf_rotation=None, leaf_label_func=None,
               show_contracted=False, link_color_func=None, ax=None,
               above_threshold_color='C0'):
    """
    绘制层次聚类结果的树状图（树枝图）。

    这个函数用于将层次聚类结果绘制成树状图（树枝图）。

    """
    # 以下是绘制树状图的说明，展示了如何通过绘制U形链接来展示每个聚类的构成。
    # U形链接的顶部表示聚类的合并，两条腿表示被合并的具体聚类，其长度表示子聚类之间的距离。
    # 这也是两个子聚类中原始观察之间的共同距离（cophenetic distance）。

    Parameters
    ----------
    Z : ndarray
        编码层次聚类的链接矩阵，用于绘制树状图。详见 ``linkage`` 函数有关 ``Z`` 格式的更多信息。
    p : int, optional
        ``truncate_mode`` 的参数 ``p``。
    truncate_mode : str, optional
        当从派生链接的原始观察矩阵较大时，树状图可能难以阅读。使用截断来压缩树状图，有多种模式：
        
        ``None``
          不执行截断（默认值）。
          注意：``'none'`` 是 ``None`` 的别名，保留了向后兼容性。
        
        ``'lastp'``
          链接中形成的最后 ``p`` 个非单例聚类是链接中唯一的非叶节点；它们对应于 ``Z[n-p-2:end]`` 中的行。
          所有其他非单例聚类都被收缩为叶节点。
        
        ``'level'``
          仅显示不超过 ``p`` 级别的树状图树。一个“级别”包括所有距最终合并 ``p`` 次的节点。
          注意：``'mtica'`` 是 ``'level'`` 的别名，保留了向后兼容性。
    
    color_threshold : double, optional
        为了简洁起见，设 :math:`t` 为 ``color_threshold``。
        如果 :math:`k` 是第一个在切割阈值 :math:`t` 以下的节点，将所有该聚类节点下方的链接着色为相同颜色。
        所有连接节点间距离大于或等于阈值的链接将使用默认的 matplotlib 颜色 ``'C0'`` 进行着色。
        如果 :math:`t` 小于或等于零，则所有节点都着色为 ``'C0'``。
        如果 ``color_threshold`` 为 None 或 'default'，与 MATLAB(TM) 行为相对应，阈值将设置为 ``0.7*max(Z[:,2])``。
    
    get_leaves : bool, optional
        在结果字典中包含一个列表 ``R['leaves']=H``。对于每个 :math:`i`，``H[i] == j``，聚类节点 ``j`` 在从左到右遍历叶节点时出现在位置 ``i``，
        其中 :math:`j < 2n-1` 且 :math:`i < n`。
    orientation : str, optional
        # 可选参数，用于指定树状图的方向，可以是以下字符串之一：

        ``'top'``
          # 将根节点放置在顶部，子节点向下延伸的方向。（默认）

        ``'bottom'``
          # 将根节点放置在底部，子节点向上延伸的方向。

        ``'left'``
          # 将根节点放置在左侧，子节点向右延伸的方向。

        ``'right'``
          # 将根节点放置在右侧，子节点向左延伸的方向。

    labels : ndarray, optional
        # 默认情况下，``labels`` 为 None，因此使用原始观测的索引来标记叶节点。
        # 否则，这是一个长度为 :math:`n` 的序列，其中 ``n == Z.shape[0] + 1``。
        # 如果它对应于一个原始观测而不是非单例聚类，则 ``labels[i]`` 的值是要放置在第 :math:`i` 个叶节点下的文本。

    count_sort : str or bool, optional
        # 对于每个节点 n，确定其两个子节点链接在可视上（从左到右）的顺序由此参数决定，可以是以下值之一：

        ``False``
          # 无操作。

        ``'ascending'`` or ``True``
          # 首先绘制具有其聚类中最少原始对象数量的子节点。

        ``'descending'``
          # 首先绘制具有其聚类中最多原始对象数量的子节点。

        注意，``distance_sort`` 和 ``count_sort`` 不能同时为 True。

    distance_sort : str or bool, optional
        # 对于每个节点 n，确定其两个子节点链接在可视上（从左到右）的顺序由此参数决定，可以是以下值之一：

        ``False``
          # 无操作。

        ``'ascending'`` or ``True``
          # 首先绘制具有其直接子节点之间最小距离的子节点。

        ``'descending'``
          # 首先绘制具有其直接子节点之间最大距离的子节点。

        注意，``distance_sort`` 和 ``count_sort`` 不能同时为 True。

    show_leaf_counts : bool, optional
         # 当为 True 时，叶节点上表示包含 :math:`k>1` 个原始观测的叶节点将用括号中的观测数量标记。

    no_plot : bool, optional
        # 当为 True 时，不进行最终的渲染。这在只需要渲染所计算的数据结构或者没有可用的 matplotlib 时非常有用。

    no_labels : bool, optional
        # 当为 True 时，在树状图的叶节点旁边不显示标签。
    # 叶子旋转角度（以度为单位），用于旋转叶子标签。未指定时，角度基于树状图中节点数量（默认为0）。
    leaf_rotation : double, optional
    
    # 叶子标签的字体大小（以点为单位）。未指定时，字体大小基于树状图中节点数量。
    leaf_font_size : int, optional
    
    # 叶子标签函数，可以是 lambda 函数或普通函数。对于每个叶子，其集群索引为 k < 2n-1。该函数应返回叶子的标签字符串。
    
    # 当 k < n 时，对应于原始观测值；
    # 当 k >= n 时，对应于非单例集群。
    
    # 例如，要将单例标记为其节点 ID，将非单例标记为其 ID、计数和不一致系数，可以简单地编写如下：
    
    # 定义叶子标签函数 llf。
    def llf(id):
        if id < n:
            return str(id)
        else:
            return '[%d %d %1.2f]' % (id, count, R[n-id,3])
    
    # 如果叶子节点的文本很大，则强制旋转90度。
    dendrogram(Z, leaf_label_func=llf, leaf_rotation=90)
    
    # leaf_label_func 也可以与 truncate_mode 一起使用，这种情况下，叶子将在截断后进行标记：
    dendrogram(Z, leaf_label_func=llf, leaf_rotation=90,
               truncate_mode='level', p=2)
    
    leaf_label_func : lambda or function, optional
    
    # 当 show_contracted 为 True 时，将非单例节点收缩至叶子节点的高度，并沿连接该叶子节点的链上绘制十字。这在使用截断时非常有用（参见 truncate_mode 参数）。
    show_contracted : bool, optional
    
    # 如果给定，link_color_func 将对每个非单例 ID 调用，对应于每个 U 形链接将会绘制的颜色。该函数应返回绘制链接的颜色，使用 matplotlib 颜色字符串代码表示。
    
    # 例如：
    # 使用 colors[k] 对每个未截断的非单例节点下方的直接链接进行颜色设置。
    dendrogram(Z, link_color_func=lambda k: colors[k])
    
    link_color_func : callable, optional
    
    # matplotlib Axes 实例。如果为 None 并且 no_plot 不为 True，则树状图将绘制在当前坐标轴上。
    # 如果 no_plot 不为 True，则树状图将绘制在给定的 Axes 实例上。如果树状图是更复杂图形的一部分，则这非常有用。
    ax : matplotlib Axes instance, optional
    
    # 设置超过 color_threshold 的链接颜色。默认为 'C0'。
    above_threshold_color : str, optional
    # 定义变量 R，存储计算树状图渲染所需的数据结构字典
    R : dict
        A dictionary of data structures computed to render the
        dendrogram. Its has the following keys:
    
        ``'color_list'``
          A list of color names. The k'th element represents the color of the
          k'th link.
    
        ``'icoord'`` and ``'dcoord'``
          Each of them is a list of lists. Let ``icoord = [I1, I2, ..., Ip]``
          where ``Ik = [xk1, xk2, xk3, xk4]`` and ``dcoord = [D1, D2, ..., Dp]``
          where ``Dk = [yk1, yk2, yk3, yk4]``, then the k'th link painted is
          ``(xk1, yk1)`` - ``(xk2, yk2)`` - ``(xk3, yk3)`` - ``(xk4, yk4)``.
    
        ``'ivl'``
          A list of labels corresponding to the leaf nodes.
    
        ``'leaves'``
          For each i, ``H[i] == j``, cluster node ``j`` appears in position
          ``i`` in the left-to-right traversal of the leaves, where
          :math:`j < 2n-1` and :math:`i < n`. If ``j`` is less than ``n``, the
          ``i``-th leaf node corresponds to an original observation.
          Otherwise, it corresponds to a non-singleton cluster.
    
        ``'leaves_color_list'``
          A list of color names. The k'th element represents the color of the
          k'th leaf.
    
    See Also
    --------
    linkage, set_link_color_palette
    
    Notes
    -----
    It is expected that the distances in ``Z[:,2]`` be monotonic, otherwise
    crossings appear in the dendrogram.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> import matplotlib.pyplot as plt
    
    A very basic example:
    
    >>> ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
    ...                    400., 754., 564., 138., 219., 869., 669.])
    >>> Z = hierarchy.linkage(ytdist, 'single')
    >>> plt.figure()
    >>> dn = hierarchy.dendrogram(Z)
    
    Now, plot in given axes, improve the color scheme and use both vertical and
    horizontal orientations:
    
    >>> hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
    >>> fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    >>> dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
    ...                            orientation='top')
    >>> dn2 = hierarchy.dendrogram(Z, ax=axes[1],
    ...                            above_threshold_color='#bcbddc',
    ...                            orientation='right')
    >>> hierarchy.set_link_color_palette(None)  # reset to default after use
    >>> plt.show()
    
    """
    # This feature was thought about but never implemented (still useful?):
    #
    #         ... = dendrogram(..., leaves_order=None)
    #
    #         Plots the leaves in the order specified by a vector of
    #         original observation indices. If the vector contains duplicates
    #         or results in a crossing, an exception will be thrown. Passing
    #         None orders leaf nodes based on the order they appear in the
    #         pre-order traversal.
    # 创建数组命名空间 xp，存储 Z 的副本
    xp = array_namespace(Z)
    # 将 Z 转换为 NumPy 数组，按 C 顺序排列
    Z = _asarray(Z, order='c', xp=xp)
    # 检查给定的方向是否为预定义的选项之一，如果不是则抛出值错误异常
    if orientation not in ["top", "left", "bottom", "right"]:
        raise ValueError("orientation must be one of 'top', 'left', "
                         "'bottom', or 'right'")
    
    # 如果标签不为None，则检查其长度，并与Z的行数进行一致性检查
    if labels is not None:
        try:
            len_labels = len(labels)
        except (TypeError, AttributeError):
            len_labels = labels.shape[0]
        if Z.shape[0] + 1 != len_labels:
            raise ValueError("Dimensions of Z and labels must be consistent.")
    
    # 检查Z的有效性，确保其符合聚类链接矩阵的规范
    is_valid_linkage(Z, throw=True, name='Z')
    
    # 获取Z的形状，并计算节点数目n
    Zs = Z.shape
    n = Zs[0] + 1
    
    # 将第二个参数p转换为整数（如果可能的话）
    if isinstance(p, (int, float)):
        p = int(p)
    else:
        raise TypeError('The second argument must be a number')
    
    # 检查截断模式是否为预定义的选项之一
    if truncate_mode not in ('lastp', 'mtica', 'level', 'none', None):
        # 对于'lastp'模式，如果p大于n或者p为0，则将其设为n
        raise ValueError('Invalid truncation mode.')
    
    # 如果截断模式为'mtica'，则将其作为别名替换为'level'
    if truncate_mode == 'mtica':
        # 'mtica' is an alias
        truncate_mode = 'level'
    
    # 如果截断模式为'level'，则确保p大于0，否则设为无穷大
    if truncate_mode == 'level':
        if p <= 0:
            p = np.inf
    
    # 如果需要获取叶子节点，则初始化一个空列表lvs，否则设为None
    if get_leaves:
        lvs = []
    else:
        lvs = None
    
    # 初始化用于存储图表信息的各种列表和字典
    icoord_list = []
    dcoord_list = []
    color_list = []
    current_color = [0]
    currently_below_threshold = [False]
    ivl = []  # list of leaves
    
    # 如果颜色阈值为None或者为'default'字符串，则将其设为Z矩阵中最大值的70%
    if color_threshold is None or (isinstance(color_threshold, str) and
                                   color_threshold == 'default'):
        color_threshold = max(Z[:, 2]) * 0.7
    
    # 创建存储计算结果的字典R
    R = {'icoord': icoord_list, 'dcoord': dcoord_list, 'ivl': ivl,
         'leaves': lvs, 'color_list': color_list}
    
    # 如果show_contracted为True，则初始化一个空列表contraction_marks，否则设为None
    contraction_marks = [] if show_contracted else None
    
    # 调用_dendrogram_calculate_info函数，计算树状图的绘制信息
    _dendrogram_calculate_info(
        Z=Z, p=p,
        truncate_mode=truncate_mode,
        color_threshold=color_threshold,
        get_leaves=get_leaves,
        orientation=orientation,
        labels=labels,
        count_sort=count_sort,
        distance_sort=distance_sort,
        show_leaf_counts=show_leaf_counts,
        i=2*n - 2,
        iv=0.0,
        ivl=ivl,
        n=n,
        icoord_list=icoord_list,
        dcoord_list=dcoord_list,
        lvs=lvs,
        current_color=current_color,
        color_list=color_list,
        currently_below_threshold=currently_below_threshold,
        leaf_label_func=leaf_label_func,
        contraction_marks=contraction_marks,
        link_color_func=link_color_func,
        above_threshold_color=above_threshold_color)
    
    # 如果不禁用绘图，则计算最大的Z矩阵值并调用_plot_dendrogram函数绘制树状图
    if not no_plot:
        mh = max(Z[:, 2])
        _plot_dendrogram(icoord_list, dcoord_list, ivl, p, n, mh, orientation,
                         no_labels, color_list,
                         leaf_font_size=leaf_font_size,
                         leaf_rotation=leaf_rotation,
                         contraction_marks=contraction_marks,
                         ax=ax,
                         above_threshold_color=above_threshold_color)
    # 将"leaves_color_list"字段赋值为调用_get_leaves_color_list函数的结果
    R["leaves_color_list"] = _get_leaves_color_list(R)
    
    # 返回变量R作为函数的结果
    return R
# 为给定的树状结构 R 提取叶子节点的颜色列表
def _get_leaves_color_list(R):
    # 创建一个与叶子节点数量相同的空列表，用于存储叶子节点的颜色
    leaves_color_list = [None] * len(R['leaves'])
    # 遍历 R 中的每个连接线，包括其 x 和 y 坐标以及颜色列表
    for link_x, link_y, link_color in zip(R['icoord'],
                                          R['dcoord'],
                                          R['color_list']):
        # 遍历每条连接线上的点 (xi, yi)
        for (xi, yi) in zip(link_x, link_y):
            # 如果 yi 为 0.0 并且 xi 是 5 的倍数且为奇数
            if yi == 0.0 and (xi % 5 == 0 and xi % 2 == 1):
                # 计算该叶子节点在 leaves_color_list 中的索引
                leaf_index = (int(xi) - 5) // 10
                # 将叶子节点的颜色设置为对应连接线的颜色
                leaves_color_list[leaf_index] = link_color
    # 返回叶子节点的颜色列表
    return leaves_color_list


# 向树结构 Z 中追加单个叶子节点
def _append_singleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func,
                                i, labels):
    # 如果 lvs 不为 None，则将当前叶子节点的索引 i 加入 lvs 中
    if lvs is not None:
        lvs.append(int(i))

    # 如果 ivl 不为 None，则处理叶子节点标签 ivl
    if ivl is not None:
        # 如果提供了 leaf_label_func 函数，则使用它获取叶子节点的标签
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i)))
        else:
            # 否则，如果调用者提供了 labels 列表，则使用它来获取叶子节点的标签
            if labels is not None:
                ivl.append(labels[int(i - n)])
            else:
                # 否则，将叶子节点的 id 转换为字符串作为其标签
                ivl.append(str(int(i)))


# 向树结构 Z 中追加非单个叶子节点
def _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl, leaf_label_func,
                                   i, labels, show_leaf_counts):
    # 如果 lvs 不为 None，则将当前叶子节点的索引 i 加入 lvs 中
    if lvs is not None:
        lvs.append(int(i))
    # 如果 ivl 不为 None，则处理叶子节点标签 ivl
    if ivl is not None:
        # 如果提供了 leaf_label_func 函数，则使用它获取叶子节点的标签
        if leaf_label_func:
            ivl.append(leaf_label_func(int(i)))
        else:
            # 否则，如果 show_leaf_counts 为 True，则将叶子节点的聚类计数显示为标签
            if show_leaf_counts:
                ivl.append("(" + str(np.asarray(Z[i - n, 3], dtype=np.int64)) + ")")
            else:
                # 否则，将 ivl 留空
                ivl.append("")


# 向树结构 Z 中追加收缩标记
def _append_contraction_marks(Z, iv, i, n, contraction_marks, xp):
    # 调用 _append_contraction_marks_sub 函数处理 Z[i-n, 0] 和 Z[i-n, 1] 的收缩标记
    _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 0], xp),
                                  n, contraction_marks, xp)
    _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 1], xp),
                                  n, contraction_marks, xp)


# 辅助函数：向树结构 Z 中追加具体的收缩标记
def _append_contraction_marks_sub(Z, iv, i, n, contraction_marks, xp):
    # 如果 i 大于等于 n，则将 (iv, Z[i - n, 2]) 添加到收缩标记列表中
    if i >= n:
        contraction_marks.append((iv, Z[i - n, 2]))
        # 调用递归函数 _append_contraction_marks_sub，处理 Z[i - n, 0] 向下取整后的结果
        _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 0], xp),
                                      n, contraction_marks, xp)
        # 调用递归函数 _append_contraction_marks_sub，处理 Z[i - n, 1] 向下取整后的结果
        _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 1], xp),
                                      n, contraction_marks, xp)
    # 定义函数_dendrogram_calculate_info，计算树状图的信息
    # Z: 聚类树的连接矩阵
    # p: 截断模式
    # truncate_mode: 截断模式
    # color_threshold: 颜色阈值，默认为无穷大
    # get_leaves: 是否获取叶子节点，默认为True
    # orientation: 树状图的方向，默认为'top'
    # labels: 节点的标签列表，可选
    # count_sort: 是否按计数排序节点，默认为False
    # distance_sort: 是否按距离排序节点，默认为False
    # show_leaf_counts: 是否显示叶子节点的计数，默认为False
    # i: 根节点的索引，默认为-1
    # iv: 独立变量的值，用于绘制根节点下最左侧的叶节点
    # ivl: 叶子节点的标签列表
    # n: 单例聚类的数量
    # icoord_list: 用于存储X轴坐标的列表
    # dcoord_list: 用于存储Y轴坐标的列表
    # lvs: 可选的聚类树等级列表
    # mhr: 是否进行中位数合并，默认为False
    # current_color: 当前节点的颜色
    # color_list: 节点颜色的列表
    # currently_below_threshold: 当前是否低于阈值的节点列表
    # leaf_label_func: 叶子节点标签函数
    # level: 节点层级
    # contraction_marks: 收缩标记
    # link_color_func: 连接颜色函数
    # above_threshold_color: 高于阈值的颜色，默认为'C0'

    """
    Calculate the endpoints of the links as well as the labels for the
    the dendrogram rooted at the node with index i. iv is the independent
    variable value to plot the left-most leaf node below the root node i
    (if orientation='top', this would be the left-most x value where the
    plotting of this root node i and its descendents should begin).

    ivl is a list to store the labels of the leaf nodes. The leaf_label_func
    is called whenever ivl != None, labels == None, and
    leaf_label_func != None. When ivl != None and labels != None, the
    labels list is used only for labeling the leaf nodes. When
    ivl == None, no labels are generated for leaf nodes.

    When get_leaves==True, a list of leaves is built as they are visited
    in the dendrogram.

    Returns a tuple with l being the independent variable coordinate that
    corresponds to the midpoint of cluster to the left of cluster i if
    i is non-singleton, otherwise the independent coordinate of the leaf
    node if i is a leaf node.

    Returns
    -------
    A tuple (left, w, h, md), where:
        * left is the independent variable coordinate of the center of the
          the U of the subtree

        * w is the amount of space used for the subtree (in independent
          variable units)

        * h is the height of the subtree in dependent variable units

        * md is the ``max(Z[*,2]``) for all nodes ``*`` below and including
          the target node.
    """
    
    xp = array_namespace(Z)  # 将Z转换为数组命名空间
    if n == 0:
        raise ValueError("Invalid singleton cluster count n.")  # 如果单例聚类数量为0，则引发值错误异常

    if i == -1:
        raise ValueError("Invalid root cluster index i.")  # 如果根集群索引为-1，则引发值错误异常
    if truncate_mode == 'lastp':
        # 如果截断模式为 'lastp'，表示根据节点是否为叶节点但对应非单例聚类，其标签可能为空字符串或聚类 i 的原始观测数量。
        if 2*n - p > i >= n:
            # 如果节点为叶节点且对应非单例聚类，并且在节点范围内，获取距离值 d
            d = Z[i - n, 2]
            # 添加非单例叶节点到树状图，更新叶节点标签和属性
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts)
            if contraction_marks is not None:
                # 如果存在收缩标记，则添加收缩标记到树状图
                _append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks, xp)
            # 返回节点的位置和相关参数
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            # 如果节点是单例叶节点，在树状图上添加单例叶节点
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                        leaf_label_func, i, labels)
            # 返回节点的位置和相关参数
            return (iv + 5.0, 10.0, 0.0, 0.0)
    elif truncate_mode == 'level':
        # 如果截断模式为 'level'，表示根据节点是否为叶节点但对应非单例聚类，并且节点水平高于给定阈值
        if i > n and level > p:
            # 如果节点为叶节点且对应非单例聚类，并且在节点范围内，获取距离值 d
            d = Z[i - n, 2]
            # 添加非单例叶节点到树状图，更新叶节点标签和属性
            _append_nonsingleton_leaf_node(Z, p, n, level, lvs, ivl,
                                           leaf_label_func, i, labels,
                                           show_leaf_counts)
            if contraction_marks is not None:
                # 如果存在收缩标记，则添加收缩标记到树状图
                _append_contraction_marks(Z, iv + 5.0, i, n, contraction_marks, xp)
            # 返回节点的位置和相关参数
            return (iv + 5.0, 10.0, 0.0, d)
        elif i < n:
            # 如果节点是单例叶节点，在树状图上添加单例叶节点
            _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                        leaf_label_func, i, labels)
            # 返回节点的位置和相关参数
            return (iv + 5.0, 10.0, 0.0, 0.0)

    # 否则，只有当我们有一个叶节点时才截断。
    #
    # 只有当叶节点对应于原始观测时，才放置叶节点。
    if i < n:
        # 如果节点是单例叶节点，在树状图上添加单例叶节点
        _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
                                    leaf_label_func, i, labels)
        # 返回节点的位置和相关参数
        return (iv + 5.0, 10.0, 0.0, 0.0)

    # 否则，我们没有叶节点，因此要继续绘制非叶节点。
    # 实际的 a 和 b 索引
    aa = int_floor(Z[i - n, 0], xp)
    ab = int_floor(Z[i - n, 1], xp)
    if aa >= n:
        # 聚类 a 下的单例数目
        na = Z[aa - n, 3]
        # a 的两个直接子节点之间的距离
        da = Z[aa - n, 2]
    else:
        na = 1
        da = 0.0
    if ab >= n:
        nb = Z[ab - n, 3]
        db = Z[ab - n, 2]
    else:
        nb = 1
        db = 0.0

    if count_sort == 'ascending' or count_sort is True:
        # 如果 a 的计数大于 b，则它及其后代应该向右绘制；否则向左绘制。
        if na > nb:
            # 要向左绘制的聚类索引（ua）将是 ab
            # 要向右绘制的聚类索引（ub）将是 aa
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif count_sort == 'descending':
        # 如果 count_sort 是 'descending'，意味着如果 a 的计数小于等于 b，
        # 则它及其后代应该被绘制到左侧，否则绘制到右侧。
        if na > nb:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    elif distance_sort == 'ascending' or distance_sort is True:
        # 如果 distance_sort 是 'ascending' 或者为 True，意味着如果 a 的距离大于 b，
        # 则它及其后代应该被绘制到右侧，否则绘制到左侧。
        if da > db:
            ua = ab
            ub = aa
        else:
            ua = aa
            ub = ab
    elif distance_sort == 'descending':
        # 如果 distance_sort 是 'descending'，意味着如果 a 的距离小于等于 b，
        # 则它及其后代应该被绘制到左侧，否则绘制到右侧。
        if da > db:
            ua = aa
            ub = ab
        else:
            ua = ab
            ub = aa
    else:
        # 默认情况下，设置 ua 为 aa，ub 为 ab。
        ua = aa
        ub = ab

    # 更新 iv 变量和使用的空间量。
    (uiva, uwa, uah, uamd) = \
        _dendrogram_calculate_info(
            Z=Z, p=p,
            truncate_mode=truncate_mode,
            color_threshold=color_threshold,
            get_leaves=get_leaves,
            orientation=orientation,
            labels=labels,
            count_sort=count_sort,
            distance_sort=distance_sort,
            show_leaf_counts=show_leaf_counts,
            i=ua, iv=iv, ivl=ivl, n=n,
            icoord_list=icoord_list,
            dcoord_list=dcoord_list, lvs=lvs,
            current_color=current_color,
            color_list=color_list,
            currently_below_threshold=currently_below_threshold,
            leaf_label_func=leaf_label_func,
            level=level + 1, contraction_marks=contraction_marks,
            link_color_func=link_color_func,
            above_threshold_color=above_threshold_color)

    h = Z[i - n, 2]
    # 如果 h 大于等于 color_threshold 或 color_threshold 小于等于 0，
    # 则设置 c 为 above_threshold_color。
    if h >= color_threshold or color_threshold <= 0:
        c = above_threshold_color

        # 如果 currently_below_threshold 的第一个元素为真，
        # 则更新 current_color[0]，循环使用链接线颜色列表。
        if currently_below_threshold[0]:
            current_color[0] = (current_color[0] + 1) % len(_link_line_colors)
        currently_below_threshold[0] = False
    else:
        # 否则，设置 currently_below_threshold 的第一个元素为真，并设置 c 为
        # _link_line_colors 中当前颜色。
        currently_below_threshold[0] = True
        c = _link_line_colors[current_color[0]]
    # 调用 _dendrogram_calculate_info 函数计算并获取返回值 (uivb, uwb, ubh, ubmd)
    (uivb, uwb, ubh, ubmd) = \
        _dendrogram_calculate_info(
            Z=Z,  # 系统聚类树的连接矩阵
            p=p,  # 系统树的阶数
            truncate_mode=truncate_mode,  # 截断模式
            color_threshold=color_threshold,  # 聚类颜色阈值
            get_leaves=get_leaves,  # 是否获取叶节点
            orientation=orientation,  # 方向
            labels=labels,  # 标签
            count_sort=count_sort,  # 是否对计数进行排序
            distance_sort=distance_sort,  # 是否对距离进行排序
            show_leaf_counts=show_leaf_counts,  # 是否显示叶节点计数
            i=ub,  # 当前节点索引
            iv=iv + uwa,  # 索引加权值
            ivl=ivl,  # 索引值列表
            n=n,  # 节点数量
            icoord_list=icoord_list,  # x 坐标列表
            dcoord_list=dcoord_list,  # y 坐标列表
            lvs=lvs,  # 等级值
            current_color=current_color,  # 当前颜色
            color_list=color_list,  # 颜色列表
            currently_below_threshold=currently_below_threshold,  # 当前是否低于阈值
            leaf_label_func=leaf_label_func,  # 叶标签函数
            level=level + 1,  # 层级加一
            contraction_marks=contraction_marks,  # 收缩标记
            link_color_func=link_color_func,  # 链接颜色函数
            above_threshold_color=above_threshold_color  # 阈值以上颜色
        )

    # 计算最大距离
    max_dist = max(uamd, ubmd, h)

    # 添加当前节点的 x 坐标范围到 x 坐标列表
    icoord_list.append([uiva, uiva, uivb, uivb])
    # 添加当前节点的 y 坐标范围到 y 坐标列表
    dcoord_list.append([uah, h, h, ubh])

    # 如果链接颜色函数存在
    if link_color_func is not None:
        # 获取颜色值
        v = link_color_func(int(i))
        # 如果颜色值不是字符串，抛出类型错误
        if not isinstance(v, str):
            raise TypeError("link_color_func must return a matplotlib "
                            "color string!")
        # 将颜色值添加到颜色列表中
        color_list.append(v)
    else:
        # 将默认颜色添加到颜色列表中
        color_list.append(c)

    # 返回当前节点的中心 x 坐标、累积宽度、高度、最大距离
    return (((uiva + uivb) / 2), uwa + uwb, h, max_dist)
# 确定两个不同的簇分配是否等价
def is_isomorphic(T1, T2):
    # 将输入的 T1 和 T2 转换为 NumPy 数组，使用 C 风格内存布局
    T1 = np.asarray(T1, order='c')
    T2 = np.asarray(T2, order='c')

    # 获取 T1 和 T2 的形状
    T1S = T1.shape
    T2S = T2.shape

    # 检查 T1 和 T2 是否为一维数组
    if len(T1S) != 1:
        raise ValueError('T1 must be one-dimensional.')
    if len(T2S) != 1:
        raise ValueError('T2 must be one-dimensional.')

    # 检查 T1 和 T2 是否具有相同的元素数量
    if T1S[0] != T2S[0]:
        raise ValueError('T1 and T2 must have the same number of elements.')

    # 获取数组的长度
    n = T1S[0]

    # 创建空字典来存储 T1 和 T2 中的簇关系
    d1 = {}
    d2 = {}

    # 遍历数组，检查簇分配是否等价
    for i in range(0, n):
        if T1[i] in d1:
            # 如果 T1[i] 在 d1 中已经有映射，检查与 T2[i] 的对应关系
            if T2[i] not in d2:
                return False
            if d1[T1[i]] != T2[i] or d2[T2[i]] != T1[i]:
                return False
        elif T2[i] in d2:
            # 如果 T2[i] 在 d2 中已经有映射，返回 False
            return False
        else:
            # 将 T1[i] 映射到 T2[i]，同时将 T2[i] 映射到 T1[i]
            d1[T1[i]] = T2[i]
            d2[T2[i]] = T1[i]

    # 如果所有检查都通过，则返回 True，表示簇分配是等价的
    return True
    xp = array_namespace(Z)
    # 将 Z 转换为适当的数组命名空间 xp
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    # 将 Z 转换为 C 顺序的 float64 类型的数组，使用 xp 命名空间
    is_valid_linkage(Z, throw=True, name='Z')
    # 检查 Z 是否为有效的 linkage matrix，如果不是则抛出异常

    n = Z.shape[0] + 1
    # 计算节点数 n，Z 的行数加 1
    MD = np.zeros((n - 1,))
    # 创建一个长度为 n-1 的全零 numpy 数组 MD
    Z = np.asarray(Z)
    # 将 Z 转换为 numpy 数组，确保 Z 是 ndarray 类型
    # 使用_hierarchy对象的方法get_max_dist_for_each_cluster来计算每个簇的最大距离，并将结果存储在MD列表中
    _hierarchy.get_max_dist_for_each_cluster(Z, MD, int(n))
    # 将MD列表转换为NumPy数组，并赋值给MD变量
    MD = xp.asarray(MD)
    # 返回MD数组作为函数的结果
    return MD
# 定义函数 maxinconsts，用于计算每个非单元素聚类及其子类的最大不一致系数
def maxinconsts(Z, R):
    """
    Return the maximum inconsistency coefficient for each
    non-singleton cluster and its children.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a matrix. See
        `linkage` for more information.
    R : ndarray
        The inconsistency matrix.

    Returns
    -------
    MI : ndarray
        A monotonic ``(n-1)``-sized numpy array of doubles.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    inconsistent : for the creation of a inconsistency matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, inconsistent, maxinconsts
    >>> from scipy.spatial.distance import pdist

    Given a data set ``X``, we can apply a clustering method to obtain a
    linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
    be also used to obtain the inconsistency matrix ``R`` associated to
    this clustering process:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = median(pdist(X))
    >>> R = inconsistent(Z)
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.11803399,  3.        ],
           [ 5.        , 13.        ,  1.11803399,  3.        ],
           [ 8.        , 15.        ,  1.11803399,  3.        ],
           [11.        , 14.        ,  1.11803399,  3.        ],
           [18.        , 19.        ,  3.        ,  6.        ],
           [16.        , 17.        ,  3.5       ,  6.        ],
           [20.        , 21.        ,  3.25      , 12.        ]])
    >>> R
    array([[1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.74535599, 1.08655358, 3.        , 1.15470054],
           [1.91202266, 1.37522872, 3.        , 1.15470054],
           [3.25      , 0.25      , 3.        , 0.        ]])

    Here, `scipy.cluster.hierarchy.maxinconsts` can be used to compute
    the maximum value of the inconsistency statistic (the last column of
    ``R``) for each non-singleton cluster and its children:

    # 调用 maxinconsts 函数计算每个非单元素聚类及其子类的不一致统计的最大值
    >>> maxinconsts(Z, R)
    array([0.        , 0.        , 0.        , 0.        , 0.70710678,
           0.70710678, 0.70710678, 0.70710678, 1.15470054, 1.15470054,
           1.15470054])
    """
    """
    xp = array_namespace(Z, R)
    # 从给定的 Z 和 R 创建适当的数组命名空间，返回对应的执行环境（如 numpy 或 cupy）
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    # 将 Z 转换为特定执行环境下的数组，使用 C 风格顺序存储，数据类型为 float64
    R = _asarray(R, order='C', dtype=xp.float64, xp=xp)
    # 将 R 转换为特定执行环境下的数组，使用 C 风格顺序存储，数据类型为 float64
    is_valid_linkage(Z, throw=True, name='Z')
    # 检查 Z 是否是有效的链接矩阵，如果无效则抛出异常，名称为 'Z'
    is_valid_im(R, throw=True, name='R')
    # 检查 R 是否是有效的im矩阵，如果无效则抛出异常，名称为 'R'

    n = Z.shape[0] + 1
    # 计算簇的数量 n，等于 Z 矩阵的行数加一
    if Z.shape[0] != R.shape[0]:
        raise ValueError("The inconsistency matrix and linkage matrix each "
                         "have a different number of rows.")
    # 如果 Z 和 R 的行数不相等，则抛出 ValueError 异常，指出两个矩阵行数不一致
    MI = np.zeros((n - 1,))
    # 创建一个大小为 n-1 的全零数组 MI
    Z = np.asarray(Z)
    # 将 Z 转换为 numpy 数组
    R = np.asarray(R)
    # 将 R 转换为 numpy 数组
    _hierarchy.get_max_Rfield_for_each_cluster(Z, R, MI, int(n), 3)
    # 使用特定函数计算每个簇的最大 R 值，并更新到 MI 数组中
    MI = xp.asarray(MI)
    # 将 MI 转换为特定执行环境下的数组
    return MI
    # 返回计算得到的 MI 数组
# 定义函数 maxRstat，计算每个非单例聚类及其子节点的最大统计值
def maxRstat(Z, R, i):
    """
    Return the maximum statistic for each non-singleton cluster and its
    children.

    Parameters
    ----------
    Z : array_like
        The hierarchical clustering encoded as a matrix. See `linkage` for more
        information.
    R : array_like
        The inconsistency matrix.
    i : int
        The column of `R` to use as the statistic.

    Returns
    -------
    MR : ndarray
        Calculates the maximum statistic for the i'th column of the
        inconsistency matrix `R` for each non-singleton cluster
        node. ``MR[j]`` is the maximum over ``R[Q(j)-n, i]``, where
        ``Q(j)`` the set of all node ids corresponding to nodes below
        and including ``j``.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.
    inconsistent : for the creation of an inconsistency matrix.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import median, inconsistent, maxRstat
    >>> from scipy.spatial.distance import pdist

    Given a data set ``X``, we can apply a clustering method to obtain a
    linkage matrix ``Z``. `scipy.cluster.hierarchy.inconsistent` can
    be also used to obtain the inconsistency matrix ``R`` associated to
    this clustering process:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = median(pdist(X))
    >>> R = inconsistent(Z)
    >>> R
    array([[1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.        , 0.        , 1.        , 0.        ],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.05901699, 0.08346263, 2.        , 0.70710678],
           [1.74535599, 1.08655358, 3.        , 1.15470054],
           [1.91202266, 1.37522872, 3.        , 1.15470054],
           [3.25      , 0.25      , 3.        , 0.        ]])

    `scipy.cluster.hierarchy.maxRstat` can be used to compute
    the maximum value of each column of ``R``, for each non-singleton
    cluster and its children:

    >>> maxRstat(Z, R, 0)
    array([1.        , 1.        , 1.        , 1.        , 1.05901699,
           1.05901699, 1.05901699, 1.05901699, 1.74535599, 1.91202266,
           3.25      ])
    >>> maxRstat(Z, R, 1)
    array([0.        , 0.        , 0.        , 0.        , 0.08346263,
           0.08346263, 0.08346263, 0.08346263, 1.08655358, 1.37522872,
           1.37522872])
    >>> maxRstat(Z, R, 3)
    array([0.        , 0.        , 0.        , 0.        , 0.70710678,
           0.70710678, 0.70710678, 0.70710678, 1.15470054, 1.15470054,
           1.15470054])

    """
    # 获取适当的数组命名空间，这里假设为 xp
    xp = array_namespace(Z, R)
    # 将 Z 转换为浮点类型 ndarray，按 C 风格顺序排列
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)
    # 将 R 转换为 NumPy 数组，确保其顺序为 C 风格，数据类型为 float64
    R = _asarray(R, order='C', dtype=xp.float64, xp=xp)
    
    # 检查 Z 是否为有效的 linkage 矩阵，如果不是则抛出异常
    is_valid_linkage(Z, throw=True, name='Z')
    
    # 检查 R 是否为有效的 inconsistency matrix，如果不是则抛出异常
    is_valid_im(R, throw=True, name='R')

    # 检查 i 是否为整数，如果不是则抛出类型错误异常
    if not isinstance(i, int):
        raise TypeError('The third argument must be an integer.')

    # 检查 i 是否在指定范围内，如果不是则抛出值错误异常
    if i < 0 or i > 3:
        raise ValueError('i must be an integer between 0 and 3 inclusive.')

    # 检查 Z 和 R 的行数是否一致，如果不一致则抛出值错误异常
    if Z.shape[0] != R.shape[0]:
        raise ValueError("The inconsistency matrix and linkage matrix each "
                         "have a different number of rows.")

    # 计算聚类数目 n
    n = Z.shape[0] + 1
    
    # 创建一个全零数组 MR，形状为 (n-1,)
    MR = np.zeros((n - 1,))
    
    # 将 Z 和 R 转换为 NumPy 数组
    Z = np.asarray(Z)
    R = np.asarray(R)
    
    # 调用 _hierarchy 模块中的函数，获取每个聚类的最大 R 值，并存储在 MR 数组中
    _hierarchy.get_max_Rfield_for_each_cluster(Z, R, MR, int(n), i)
    
    # 将 MR 数组转换为 xp (可能是 NumPy 或者类似库的数组)
    MR = xp.asarray(MR)
    
    # 返回计算得到的 MR 数组作为结果
    return MR
# 定义函数 leaders，返回层次聚类中的根节点
def leaders(Z, T):
    """
    Return the root nodes in a hierarchical clustering.

    Returns the root nodes in a hierarchical clustering corresponding
    to a cut defined by a flat cluster assignment vector ``T``. See
    the ``fcluster`` function for more information on the format of ``T``.

    For each flat cluster :math:`j` of the :math:`k` flat clusters
    represented in the n-sized flat cluster assignment vector ``T``,
    this function finds the lowest cluster node :math:`i` in the linkage
    tree Z, such that:

      * leaf descendants belong only to flat cluster j
        (i.e., ``T[p]==j`` for all :math:`p` in :math:`S(i)`, where
        :math:`S(i)` is the set of leaf ids of descendant leaf nodes
        with cluster node :math:`i`)

      * there does not exist a leaf that is not a descendant with
        :math:`i` that also belongs to cluster :math:`j`
        (i.e., ``T[q]!=j`` for all :math:`q` not in :math:`S(i)`). If
        this condition is violated, ``T`` is not a valid cluster
        assignment vector, and an exception will be thrown.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a matrix. See
        `linkage` for more information.
    T : ndarray
        The flat cluster assignment vector.

    Returns
    -------
    L : ndarray
        The leader linkage node id's stored as a k-element 1-D array,
        where ``k`` is the number of flat clusters found in ``T``.

        ``L[j]=i`` is the linkage cluster node id that is the
        leader of flat cluster with id M[j]. If ``i < n``, ``i``
        corresponds to an original observation, otherwise it
        corresponds to a non-singleton cluster.
    M : ndarray
        The leader linkage node id's stored as a k-element 1-D array, where
        ``k`` is the number of flat clusters found in ``T``. This allows the
        set of flat cluster ids to be any arbitrary set of ``k`` integers.

        For example: if ``L[3]=2`` and ``M[3]=8``, the flat cluster with
        id 8's leader is linkage node 2.

    See Also
    --------
    fcluster : for the creation of flat cluster assignments.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, fcluster, leaders
    >>> from scipy.spatial.distance import pdist

    Given a linkage matrix ``Z`` - obtained after apply a clustering method
    to a dataset ``X`` - and a flat cluster assignment array ``T``:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    """
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    >>> T = fcluster(Z, 3, criterion='distance')
    >>> T
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

    `scipy.cluster.hierarchy.leaders` returns the indices of the nodes
    in the dendrogram that are the leaders of each flat cluster:

    >>> L, M = leaders(Z, T)
    >>> L
    array([16, 17, 18, 19], dtype=int32)

    (remember that indices 0-11 point to the 12 data points in ``X``,
    whereas indices 12-22 point to the 11 rows of ``Z``)

    `scipy.cluster.hierarchy.leaders` also returns the indices of
    the flat clusters in ``T``:

    >>> M
    array([1, 2, 3, 4], dtype=int32)

    """
    xp = array_namespace(Z, T)  # 使用给定的 Z 和 T 创建适当的数组命名空间
    Z = _asarray(Z, order='C', dtype=xp.float64, xp=xp)  # 将 Z 转换为 C 顺序的 float64 类型数组
    T = _asarray(T, order='C', xp=xp)  # 将 T 转换为 C 顺序的 xp 类型数组
    is_valid_linkage(Z, throw=True, name='Z')  # 检查 Z 是否是有效的 linkage

    if T.dtype != xp.int32:  # 如果 T 的数据类型不是 int32
        raise TypeError('T must be a 1-D array of dtype int32.')  # 抛出类型错误异常

    if T.shape[0] != Z.shape[0] + 1:  # 如果 T 的长度不等于 Z 的行数加 1
        raise ValueError('Mismatch: len(T)!=Z.shape[0] + 1.')  # 抛出值错误异常

    n_clusters = int(xp.unique_values(T).shape[0])  # 计算 T 中唯一值的数量作为群集数
    n_obs = int(Z.shape[0] + 1)  # 计算观察值的数量
    L = np.zeros(n_clusters, dtype=np.int32)  # 创建一个全零的 int32 类型数组 L
    M = np.zeros(n_clusters, dtype=np.int32)  # 创建一个全零的 int32 类型数组 M
    Z = np.asarray(Z)  # 将 Z 转换为 NumPy 数组
    T = np.asarray(T, dtype=np.int32)  # 将 T 转换为 int32 类型的 NumPy 数组
    s = _hierarchy.leaders(Z, T, L, M, n_clusters, n_obs)  # 调用层次聚类 leaders 方法
    if s >= 0:  # 如果返回的 s 大于等于 0
        raise ValueError(('T is not a valid assignment vector. Error found '
                          'when examining linkage node %d (< 2n-1).') % s)  # 抛出值错误异常
    L, M = xp.asarray(L), xp.asarray(M)  # 将 L 和 M 转换为 xp 类型的数组
    return (L, M)  # 返回元组 (L, M)
```