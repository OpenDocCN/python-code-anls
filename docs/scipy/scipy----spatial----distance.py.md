# `D:\src\scipysrc\scipy\scipy\spatial\distance.py`

```
"""
Distance computations (:mod:`scipy.spatial.distance`)
=====================================================

.. sectionauthor:: Damian Eads

Function reference
------------------

Distance matrix computation from a collection of raw observation vectors
stored in a rectangular array.

.. autosummary::
   :toctree: generated/

   pdist   -- pairwise distances between observation vectors.
   cdist   -- distances between two collections of observation vectors
   squareform -- convert distance matrix to a condensed one and vice versa
   directed_hausdorff -- directed Hausdorff distance between arrays

Predicates for checking the validity of distance matrices, both
condensed and redundant. Also contained in this module are functions
for computing the number of observations in a distance matrix.

.. autosummary::
   :toctree: generated/

   is_valid_dm -- checks for a valid distance matrix
   is_valid_y  -- checks for a valid condensed distance matrix
   num_obs_dm  -- # of observations in a distance matrix
   num_obs_y   -- # of observations in a condensed distance matrix

Distance functions between two numeric vectors ``u`` and ``v``. Computing
distances over a large collection of vectors is inefficient for these
functions. Use ``pdist`` for this purpose.

.. autosummary::
   :toctree: generated/

   braycurtis       -- the Bray-Curtis distance.
   canberra         -- the Canberra distance.
   chebyshev        -- the Chebyshev distance.
   cityblock        -- the Manhattan distance.
   correlation      -- the Correlation distance.
   cosine           -- the Cosine distance.
   euclidean        -- the Euclidean distance.
   jensenshannon    -- the Jensen-Shannon distance.
   mahalanobis      -- the Mahalanobis distance.
   minkowski        -- the Minkowski distance.
   seuclidean       -- the normalized Euclidean distance.
   sqeuclidean      -- the squared Euclidean distance.

Distance functions between two boolean vectors (representing sets) ``u`` and
``v``.  As in the case of numerical vectors, ``pdist`` is more efficient for
computing the distances between all pairs.

.. autosummary::
   :toctree: generated/

   dice             -- the Dice dissimilarity.
   hamming          -- the Hamming distance.
   jaccard          -- the Jaccard distance.
   kulczynski1      -- the Kulczynski 1 distance.
   rogerstanimoto   -- the Rogers-Tanimoto dissimilarity.
   russellrao       -- the Russell-Rao dissimilarity.
   sokalmichener    -- the Sokal-Michener dissimilarity.
   sokalsneath      -- the Sokal-Sneath dissimilarity.
   yule             -- the Yule dissimilarity.

:func:`hamming` also operates over discrete numerical vectors.
"""

# Copyright (C) Damian Eads, 2007-2008. New BSD License.

# 导入模块，用于计算距离和矩阵转换
from . import _distance_wrap
from . import _hausdorff

# 导入必要的模块和函数
from numpy import (asarray, Inf, minimum, sqrt, double, any, zeros,
                   arccos, clip, dot, pi, sin, where, isscalar)
import numpy as np
from . import distance as dist

# 定义允许导出的函数和类名
__all__ = [
    'braycurtis',      # Bray-Curtis 距离函数
    'canberra',        # Canberra 距离函数
    'cdist',           # 两组观测向量之间的距离
    'chebyshev',       # Chebyshev 距离函数
    'cityblock',       # 曼哈顿距离函数
    'correlation',     # 相关距离函数
    'cosine',          # 余弦距离函数
    'dice',            # Dice 不相似度函数
    'directed_hausdorff',  # 有向豪斯多夫距离函数
    'euclidean',       # 欧几里得距离函数
    'hamming',         # 汉明距离函数
    'is_valid_dm',     # 检查是否为有效的距离矩阵函数
    'is_valid_y',      # 检查是否为有效的压缩距离矩阵函数
    'jaccard',         # Jaccard 距离函数
    'jensenshannon',        # 字符串，标识一个特定的计算方法或指标名称
    'kulczynski1',          # 字符串，标识一个特定的计算方法或指标名称
    'mahalanobis',          # 字符串，标识一个特定的计算方法或指标名称
    'minkowski',            # 字符串，标识一个特定的计算方法或指标名称
    'num_obs_dm',           # 字符串，标识一个特定的变量或参数名称
    'num_obs_y',            # 字符串，标识一个特定的变量或参数名称
    'pdist',                # 字符串，标识一个特定的计算方法或指标名称
    'rogerstanimoto',       # 字符串，标识一个特定的计算方法或指标名称
    'russellrao',           # 字符串，标识一个特定的计算方法或指标名称
    'seuclidean',           # 字符串，标识一个特定的计算方法或指标名称
    'sokalmichener',        # 字符串，标识一个特定的计算方法或指标名称
    'sokalsneath',          # 字符串，标识一个特定的计算方法或指标名称
    'sqeuclidean',          # 字符串，标识一个特定的计算方法或指标名称
    'squareform',           # 字符串，标识一个特定的计算方法或指标名称
    'yule'                  # 字符串，标识一个特定的计算方法或指标名称
def _copy_array_if_base_present(a):
    """Copy the array if its base points to a parent array."""
    # 如果数组的 base 属性不为 None，则进行复制操作
    if a.base is not None:
        return a.copy()
    return a


def _correlation_cdist_wrap(XA, XB, dm, **kwargs):
    """Wrap function for computing correlation distance between two arrays."""
    # 对第一个数组 XA 和第二个数组 XB 进行去中心化处理
    XA = XA - XA.mean(axis=1, keepdims=True)
    XB = XB - XB.mean(axis=1, keepdims=True)
    # 调用底层的 C 函数计算余弦距离
    _distance_wrap.cdist_cosine_double_wrap(XA, XB, dm, **kwargs)


def _correlation_pdist_wrap(X, dm, **kwargs):
    """Wrap function for computing correlation distance for a single array."""
    # 对数组 X 进行去中心化处理
    X2 = X - X.mean(axis=1, keepdims=True)
    # 调用底层的 C 函数计算余弦距离
    _distance_wrap.pdist_cosine_double_wrap(X2, dm, **kwargs)


def _convert_to_type(X, out_type):
    """Convert array X to the specified dtype."""
    return np.ascontiguousarray(X, dtype=out_type)


def _nbool_correspond_all(u, v, w=None):
    """Count occurrences of different boolean combinations between arrays u and v."""
    if u.dtype == v.dtype == bool and w is None:
        not_u = ~u
        not_v = ~v
        nff = (not_u & not_v).sum()
        nft = (not_u & v).sum()
        ntf = (u & not_v).sum()
        ntt = (u & v).sum()
    else:
        dtype = np.result_type(int, u.dtype, v.dtype)
        u = u.astype(dtype)
        v = v.astype(dtype)
        not_u = 1.0 - u
        not_v = 1.0 - v
        if w is not None:
            not_u = w * not_u
            u = w * u
        nff = (not_u * not_v).sum()
        nft = (not_u * v).sum()
        ntf = (u * not_v).sum()
        ntt = (u * v).sum()
    return (nff, nft, ntf, ntt)


def _nbool_correspond_ft_tf(u, v, w=None):
    """Count occurrences of specific boolean combinations between arrays u and v."""
    if u.dtype == v.dtype == bool and w is None:
        not_u = ~u
        not_v = ~v
        nft = (not_u & v).sum()
        ntf = (u & not_v).sum()
    else:
        dtype = np.result_type(int, u.dtype, v.dtype)
        u = u.astype(dtype)
        v = v.astype(dtype)
        not_u = 1.0 - u
        not_v = 1.0 - v
        if w is not None:
            not_u = w * not_u
            u = w * u
        nft = (not_u * v).sum()
        ntf = (u * not_v).sum()
    return (nft, ntf)


def _validate_cdist_input(XA, XB, mA, mB, n, metric_info, **kwargs):
    """Validate and prepare input arrays for distance computation."""
    # 获取支持的数据类型
    types = metric_info.types
    # 选择最佳的数据类型
    typ = types[types.index(XA.dtype)] if XA.dtype in types else types[0]
    # 将输入数组转换为指定的数据类型
    XA = _convert_to_type(XA, out_type=typ)
    XB = _convert_to_type(XB, out_type=typ)

    # 验证额外的关键字参数
    _validate_kwargs = metric_info.validator
    if _validate_kwargs:
        kwargs = _validate_kwargs((XA, XB), mA + mB, n, **kwargs)
    return XA, XB, typ, kwargs


def _validate_weight_with_size(X, m, n, **kwargs):
    """Validate weight array size and return updated keyword arguments."""
    w = kwargs.pop('w', None)
    if w is None:
        return kwargs

    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError("Weights must have same size as input vector. "
                         f"{w.shape[0]} vs. {n}")

    kwargs['w'] = _validate_weights(w)
    return kwargs
    return kwargs


# 返回关键字参数字典
返回传递给函数的关键字参数字典。kwargs 是一个包含所有关键字参数的字典，这里直接返回它。
# 验证汉明距离计算函数的关键字参数，确保权重参数有效，默认为全一数组
def _validate_hamming_kwargs(X, m, n, **kwargs):
    # 获取关键字参数中的权重参数，如果未指定则使用全一数组
    w = kwargs.get('w', np.ones((n,), dtype='double'))

    # 检查权重数组的维度和长度是否与输入向量相同，若不同则引发数值错误异常
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(
            "Weights must have same size as input vector. %d vs. %d" % (w.shape[0], n)
        )

    # 验证并更新权重数组，确保所有权重均为非负数
    kwargs['w'] = _validate_weights(w)
    return kwargs


# 验证马氏距离计算函数的关键字参数
def _validate_mahalanobis_kwargs(X, m, n, **kwargs):
    # 弹出关键字参数中的协方差矩阵逆矩阵，如果不存在则进行计算
    VI = kwargs.pop('VI', None)
    if VI is None:
        # 如果观测数量小于等于观测维度，引发数值错误异常
        if m <= n:
            raise ValueError("The number of observations (%d) is too "
                             "small; the covariance matrix is "
                             "singular. For observations with %d "
                             "dimensions, at least %d observations "
                             "are required." % (m, n, n + 1))
        # 如果输入是元组，则按行堆叠成数组，计算协方差矩阵并求其逆矩阵
        if isinstance(X, tuple):
            X = np.vstack(X)
        # 计算协方差矩阵并求其逆矩阵的转置
        CV = np.atleast_2d(np.cov(X.astype(np.float64, copy=False).T))
        VI = np.linalg.inv(CV).T.copy()
    # 将验证后的逆协方差矩阵更新至关键字参数中
    kwargs["VI"] = _convert_to_double(VI)
    return kwargs


# 验证闵可夫斯基距离计算函数的关键字参数
def _validate_minkowski_kwargs(X, m, n, **kwargs):
    # 使用内部函数验证权重，并确保关键字参数中包含距离度量参数 p，默认为 2
    kwargs = _validate_weight_with_size(X, m, n, **kwargs)
    if 'p' not in kwargs:
        kwargs['p'] = 2.
    else:
        # 如果 p 值小于等于 0，引发数值错误异常
        if kwargs['p'] <= 0:
            raise ValueError("p must be greater than 0")

    return kwargs


# 验证距离度量输入的关键字参数
def _validate_pdist_input(X, m, n, metric_info, **kwargs):
    # 获取支持的数据类型
    types = metric_info.types
    # 根据输入数据的数据类型选择最佳类型
    typ = types[types.index(X.dtype)] if X.dtype in types else types[0]
    # 将输入数据转换为指定类型
    X = _convert_to_type(X, out_type=typ)

    # 验证其他关键字参数
    _validate_kwargs = metric_info.validator
    if _validate_kwargs:
        # 如果存在验证函数，则调用该函数验证关键字参数
        kwargs = _validate_kwargs(X, m, n, **kwargs)
    return X, typ, kwargs


# 验证标准化欧氏距离计算函数的关键字参数
def _validate_seuclidean_kwargs(X, m, n, **kwargs):
    # 弹出关键字参数中的方差向量 V，如果不存在则计算样本方差
    V = kwargs.pop('V', None)
    if V is None:
        # 如果输入是元组，则按行堆叠成数组，计算每个维度的样本方差
        if isinstance(X, tuple):
            X = np.vstack(X)
        # 计算每个维度的样本方差并作为方差向量
        V = np.var(X.astype(np.float64, copy=False), axis=0, ddof=1)
    else:
        # 如果指定了方差向量 V，则确保其为一维向量且长度与向量维度相同
        V = np.asarray(V, order='c')
        if len(V.shape) != 1:
            raise ValueError('Variance vector V must '
                             'be one-dimensional.')
        if V.shape[0] != n:
            raise ValueError('Variance vector V must be of the same '
                             'dimension as the vectors on which the distances '
                             'are computed.')
    # 将验证后的方差向量更新至关键字参数中
    kwargs['V'] = _convert_to_double(V)
    return kwargs


# 验证输入向量是否为有效向量的函数
def _validate_vector(u, dtype=None):
    # 将输入数据转换为指定类型的一维数组（必要时按 C 顺序存储）
    u = np.asarray(u, dtype=dtype, order='c')
    # 如果转换后的数组为一维，则返回该数组；否则引发数值错误异常
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")


# 验证权重数组的有效性函数
def _validate_weights(w, dtype=np.float64):
    # 验证输入数组是否为有效的一维数组，并确保所有权重均为非负数
    w = _validate_vector(w, dtype=dtype)
    if np.any(w < 0):
        raise ValueError("Input weights should be all non-negative")
    return w


# 计算有向豪斯多夫距离的函数，该函数未完成定义
def directed_hausdorff(u, v, seed=0):
    """
    Compute the directed Hausdorff distance between two 2-D arrays.

    Distances between pairs are calculated using a Euclidean metric.

    Parameters
    ----------
    u : (M,N) array_like
        Input array with M points in N dimensions.
    v : (O,N) array_like
        Input array with O points in N dimensions.
    seed : int or None, optional
        Local `numpy.random.RandomState` seed. Default is 0, a random
        shuffling of u and v that guarantees reproducibility.

    Returns
    -------
    d : double
        The directed Hausdorff distance between arrays `u` and `v`,

    index_1 : int
        index of point contributing to Hausdorff pair in `u`

    index_2 : int
        index of point contributing to Hausdorff pair in `v`

    Raises
    ------
    ValueError
        An exception is thrown if `u` and `v` do not have
        the same number of columns.

    See Also
    --------
    scipy.spatial.procrustes : Another similarity test for two data sets

    Notes
    -----
    Uses the early break technique and the random sampling approach
    described by [1]_. Although worst-case performance is ``O(m * o)``
    (as with the brute force algorithm), this is unlikely in practice
    as the input data would have to require the algorithm to explore
    every single point interaction, and after the algorithm shuffles
    the input points at that. The best case performance is O(m), which
    is satisfied by selecting an inner loop distance that is less than
    cmax and leads to an early break as often as possible. The authors
    have formally shown that the average runtime is closer to O(m).

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] A. A. Taha and A. Hanbury, "An efficient algorithm for
           calculating the exact Hausdorff distance." IEEE Transactions On
           Pattern Analysis And Machine Intelligence, vol. 37 pp. 2153-63,
           2015.

    Examples
    --------
    Find the directed Hausdorff distance between two 2-D arrays of
    coordinates:

    >>> from scipy.spatial.distance import directed_hausdorff
    >>> import numpy as np
    >>> u = np.array([(1.0, 0.0),
    ...               (0.0, 1.0),
    ...               (-1.0, 0.0),
    ...               (0.0, -1.0)])
    >>> v = np.array([(2.0, 0.0),
    ...               (0.0, 2.0),
    ...               (-2.0, 0.0),
    ...               (0.0, -4.0)])

    >>> directed_hausdorff(u, v)[0]
    2.23606797749979
    >>> directed_hausdorff(v, u)[0]
    3.0

    Find the general (symmetric) Hausdorff distance between two 2-D
    arrays of coordinates:

    >>> max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    3.0

    Find the indices of the points that generate the Hausdorff distance
    (the Hausdorff pair):

    >>> directed_hausdorff(v, u)[1:]
    (3, 3)

    """
    # Convert input arrays `u` and `v` to numpy arrays of type float64, ensuring they are C-contiguous
    u = np.asarray(u, dtype=np.float64, order='c')
    v = np.asarray(v, dtype=np.float64, order='c')
    # 检查矩阵 u 和 v 的列数是否相同，如果不同则抛出值错误异常
    if u.shape[1] != v.shape[1]:
        raise ValueError('u and v need to have the same '
                         'number of columns')
    # 调用 _hausdorff 模块中的 directed_hausdorff 函数计算有向哈尔斯多夫距离
    result = _hausdorff.directed_hausdorff(u, v, seed)
    # 返回计算结果
    return result
# 定义一个函数，计算两个一维数组之间的Minkowski距离
def minkowski(u, v, p=2, w=None):
    """
    Compute the Minkowski distance between two 1-D arrays.

    The Minkowski distance between 1-D arrays `u` and `v`,
    is defined as

    .. math::

       {\\|u-v\\|}_p = (\\sum{|u_i - v_i|^p})^{1/p}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    p : scalar
        The order of the norm of the difference :math:`{\\|u-v\\|}_p`. Note
        that for :math:`0 < p < 1`, the triangle inequality only holds with
        an additional multiplicative factor, i.e. it is only a quasi-metric.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    minkowski : double
        The Minkowski distance between vectors `u` and `v`.
    """
    # 确保输入的向量是有效的，调用 _validate_vector 函数
    u = _validate_vector(u)
    v = _validate_vector(v)
    # 如果 p 小于等于 0，则抛出 ValueError
    if p <= 0:
        raise ValueError("p must be greater than 0")
    # 计算向量 u 和 v 的差
    u_v = u - v
    # 如果提供了权重 w，则进行下列操作
    if w is not None:
        # 确保权重向量是有效的，调用 _validate_weights 函数
        w = _validate_weights(w)
        # 根据 p 的不同取值，计算权重的根
        if p == 1:
            root_w = w
        elif p == 2:
            # 更好的精度和速度，计算权重的平方根
            root_w = np.sqrt(w)
        elif p == np.inf:
            # 对于无穷范数，权重根为 w 不等于 0 的元素
            root_w = (w != 0)
        else:
            # 一般情况下，计算权重的 p 次方根
            root_w = np.power(w, 1/p)
        # 将 u_v 乘以权重的根
        u_v = root_w * u_v
    # 计算修正后的向量 u_v 的 p 范数距离
    dist = norm(u_v, ord=p)
    return dist


# 定义一个函数，计算两个一维数组之间的欧几里德距离
def euclidean(u, v, w=None):
    """
    Computes the Euclidean distance between two 1-D arrays.

    The Euclidean distance between 1-D arrays `u` and `v`, is defined as

    .. math::

       {\\|u-v\\|}_2

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    euclidean : double
        The Euclidean distance between vectors `u` and `v`.
    """
    # 调用 minkowski 函数计算 p=2 时的 Minkowski 距离，即欧几里德距离
    return minkowski(u, v, p=2, w=w)


# 定义一个函数，计算两个一维数组之间的平方欧几里德距离
def sqeuclidean(u, v, w=None):
    """
    Compute the squared Euclidean distance between two 1-D arrays.

    The squared Euclidean distance between `u` and `v` is defined as
    """
    # 计算加权的平方欧氏距离
    def sqeuclidean(u, v, w=None):
        # 保持浮点类型，但将其他类型转换为 np.float64 以提高稳定性
        utype, vtype = None, None
        if not (hasattr(u, "dtype") and np.issubdtype(u.dtype, np.inexact)):
            utype = np.float64
        if not (hasattr(v, "dtype") and np.issubdtype(v.dtype, np.inexact)):
            vtype = np.float64
        
        # 确保向量 u 和 v 是合法的向量，并且类型为指定的 utype 和 vtype
        u = _validate_vector(u, dtype=utype)
        v = _validate_vector(v, dtype=vtype)
        
        # 计算向量 u 和 v 的差
        u_v = u - v
        
        # 如果提供了权重 w，则将权重应用到 u_v 上
        u_v_w = u_v  # 只希望权重应用一次
        if w is not None:
            w = _validate_weights(w)
            u_v_w = w * u_v
        
        # 返回向量 u_v 和 u_v_w 的点积，即加权的平方欧氏距离
        return np.dot(u_v, u_v_w)
# 导入必要的库函数
from scipy.spatial import distance

# 计算两个一维数组之间的相关距离
def correlation(u, v, w=None, centered=True):
    """
    Compute the correlation distance between two 1-D arrays.

    The correlation distance between `u` and `v`, is
    defined as

    .. math::

        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}

    where :math:`\\bar{u}` is the mean of the elements of `u`
    and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0
    centered : bool, optional
        If True, `u` and `v` will be centered. Default is True.

    Returns
    -------
    correlation : double
        The correlation distance between 1-D array `u` and `v`.

    Examples
    --------
    Find the correlation between two arrays.

    >>> from scipy.spatial.distance import correlation
    >>> correlation([1, 0, 1], [1, 1, 0])
    1.5

    Using a weighting array, the correlation can be calculated as:

    >>> correlation([1, 0, 1], [1, 1, 0], w=[0.9, 0.1, 0.1])
    1.1

    If centering is not needed, the correlation can be calculated as:

    >>> correlation([1, 0, 1], [1, 1, 0], centered=False)
    0.5
    """
    # 确保输入的向量格式正确
    u = _validate_vector(u)
    v = _validate_vector(v)
    # 如果有权重向量，进行有效性验证并归一化
    if w is not None:
        w = _validate_weights(w)
        w = w / w.sum()
    # 如果需要中心化处理
    if centered:
        if w is not None:
            umu = np.dot(u, w)
            vmu = np.dot(v, w)
        else:
            umu = np.mean(u)
            vmu = np.mean(v)
        u = u - umu
        v = v - vmu
    # 计算加权后的向量乘积
    if w is not None:
        vw = v * w
        uw = u * w
    else:
        vw, uw = v, u
    # 计算相关距离
    uv = np.dot(u, vw)
    uu = np.dot(u, uw)
    vv = np.dot(v, vw)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    # 修剪结果以避免舍入误差
    return np.clip(dist, 0.0, 2.0)


# 计算两个一维数组之间的余弦距离
def cosine(u, v, w=None):
    """
    Compute the Cosine distance between 1-D arrays.

    The Cosine distance between `u` and `v`, is defined as

    .. math::

        1 - \\frac{u \\cdot v}
                  {\\|u\\|_2 \\|v\\|_2}.

    where :math:`u \\cdot v` is the dot product of :math:`u` and
    :math:`v`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    cosine : double
        The Cosine distance between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.cosine([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.cosine([100, 0, 0], [0, 1, 0])
    1.0
    >>> distance.cosine([1, 1, 0], [0, 1, 0])
    0.29289321881345254

    """
    # 使用余弦距离计算两个向量 u 和 v 之间的相关性，此距离也称为“未居中的相关性”或“反映性相关性”。
    return correlation(u, v, w=w, centered=False)
# 计算两个一维数组之间的汉明距离

def hamming(u, v, w=None):
    """
    Compute the Hamming distance between two 1-D arrays.

    The Hamming distance between 1-D arrays `u` and `v`, is simply the
    proportion of disagreeing components in `u` and `v`. If `u` and `v` are
    boolean vectors, the Hamming distance is

    .. math::

       \\frac{c_{01} + c_{10}}{n}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    hamming : double
        The Hamming distance between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.hamming([1, 0, 0], [0, 1, 0])
    0.66666666666666663
    >>> distance.hamming([1, 0, 0], [1, 1, 0])
    0.33333333333333331
    >>> distance.hamming([1, 0, 0], [2, 0, 0])
    0.33333333333333331
    >>> distance.hamming([1, 0, 0], [3, 0, 0])
    0.33333333333333331

    """
    # 验证并转换输入数组为合适的向量格式
    u = _validate_vector(u)
    v = _validate_vector(v)
    # 如果数组长度不同，抛出异常
    if u.shape != v.shape:
        raise ValueError('The 1d arrays must have equal lengths.')
    # 计算不同元素的数量
    u_ne_v = u != v
    # 如果有权重向量，进行加权求和
    if w is not None:
        # 验证权重向量并进行归一化
        w = _validate_weights(w)
        if w.shape != u.shape:
            raise ValueError("'w' should have the same length as 'u' and 'v'.")
        w = w / w.sum()
        return np.dot(u_ne_v, w)
    # 否则返回不同元素的比例均值
    return np.mean(u_ne_v)


# 计算两个布尔型一维数组之间的Jaccard-Needham不相似度

def jaccard(u, v, w=None):
    """
    Compute the Jaccard-Needham dissimilarity between two boolean 1-D arrays.

    The Jaccard-Needham dissimilarity between 1-D boolean arrays `u` and `v`,
    is defined as

    .. math::

       \\frac{c_{TF} + c_{FT}}
            {c_{TT} + c_{FT} + c_{TF}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    jaccard : double
        The Jaccard distance between vectors `u` and `v`.

    Notes
    -----
    When both `u` and `v` lead to a ``0/0`` division i.e. there is no overlap
    between the items in the vectors the returned distance is 0. See the
    Wikipedia page on the Jaccard index [1]_, and this paper [2]_.

    .. versionchanged:: 1.2.0
        Previously, when `u` and `v` lead to a ``0/0`` division, the function
        would return NaN. This was changed to return 0 instead.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Jaccard_index

    """
    # 验证并转换输入数组为合适的向量格式
    u = _validate_vector(u)
    v = _validate_vector(v)
    # 如果数组长度不同，抛出异常
    if u.shape != v.shape:
        raise ValueError('The 1d arrays must have equal lengths.')
    # 计算不同元素的数量
    u_ne_v = u != v
    # 如果有权重向量，进行加权求和
    if w is not None:
        # 验证权重向量并进行归一化
        w = _validate_weights(w)
        if w.shape != u.shape:
            raise ValueError("'w' should have the same length as 'u' and 'v'.")
        w = w / w.sum()
        # 返回加权后的Jaccard距离
        return np.dot(u_ne_v, w) / (1 - np.dot(u == v, w))
    # 否则返回Jaccard距离
    return np.mean(u_ne_v)
    # 确保向量 u 合法，即如果 u 不是一个有效的向量，则做必要的处理使其合法化
    u = _validate_vector(u)
    
    # 确保向量 v 合法，同上
    v = _validate_vector(v)
    
    # 计算非零元素的掩码，即找出向量 u 和 v 中不为零的元素的位置
    nonzero = np.bitwise_or(u != 0, v != 0)
    
    # 找出不相等且非零的元素的位置，即 u 和 v 中对应位置元素不相等且不为零的位置
    unequal_nonzero = np.bitwise_and((u != v), nonzero)
    
    # 如果指定了权重向量 w，则将非零元素掩码和不相等且非零元素掩码都乘以 w 对应的值
    if w is not None:
        w = _validate_weights(w)
        nonzero = w * nonzero
        unequal_nonzero = w * unequal_nonzero
    
    # 计算不相等且非零元素的数量，并转换为浮点数
    a = np.float64(unequal_nonzero.sum())
    
    # 计算非零元素的数量，并转换为浮点数
    b = np.float64(nonzero.sum())
    
    # 如果 b 不为零，则返回 a/b，否则返回 0，这是 Jaccard 距离的计算公式
    return (a / b) if b != 0 else 0
# 计算两个布尔型一维数组之间的 Kulczynski 1 不相似度。

def kulczynski1(u, v, *, w=None):
    """
    Compute the Kulczynski 1 dissimilarity between two boolean 1-D arrays.

    The Kulczynski 1 dissimilarity between two boolean 1-D arrays `u` and `v`
    of length ``n``, is defined as

    .. math::

         \\frac{c_{11}}
              {c_{01} + c_{10}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k \\in {0, 1, ..., n-1}`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    kulczynski1 : float
        The Kulczynski 1 distance between vectors `u` and `v`.

    Notes
    -----
    This measure has a minimum value of 0 and no upper limit.
    It is un-defined when there are no non-matches.

    .. versionadded:: 1.8.0

    References
    ----------
    .. [1] Kulczynski S. et al. Bulletin
           International de l'Academie Polonaise des Sciences
           et des Lettres, Classe des Sciences Mathematiques
           et Naturelles, Serie B (Sciences Naturelles). 1927;
           Supplement II: 57-203.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.kulczynski1([1, 0, 0], [0, 1, 0])
    0.0
    >>> distance.kulczynski1([True, False, False], [True, True, False])
    1.0
    >>> distance.kulczynski1([True, False, False], [True])
    0.5
    >>> distance.kulczynski1([1, 0, 0], [3, 1, 0])
    -3.0

    """
    # 验证输入向量 u 和 v，确保它们是有效的向量
    u = _validate_vector(u)
    v = _validate_vector(v)
    # 如果给定了权重向量 w，则验证其有效性
    if w is not None:
        w = _validate_weights(w)
    # 获取 u 和 v 的布尔值匹配情况（非匹配、u 中为 false、v 中为 false、u 和 v 都为 true 的次数）
    (_, nft, ntf, ntt) = _nbool_correspond_all(u, v, w=w)
    # 返回 Kulczynski 1 不相似度，计算公式为 ntt / (ntf + nft)
    return ntt / (ntf + nft)


# 计算两个一维数组之间的标准化欧氏距离。

def seuclidean(u, v, V):
    """
    Return the standardized Euclidean distance between two 1-D arrays.

    The standardized Euclidean distance between two n-vectors `u` and `v` is

    .. math::

       \\sqrt{\\sum\\limits_i \\frac{1}{V_i} \\left(u_i-v_i \\right)^2}

    ``V`` is the variance vector; ``V[I]`` is the variance computed over all the i-th
    components of the points. If not passed, it is automatically computed.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    V : (N,) array_like
        `V` is an 1-D array of component variances. It is usually computed
        among a larger collection vectors.

    Returns
    -------
    seuclidean : double
        The standardized Euclidean distance between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.seuclidean([1, 0, 0], [0, 1, 0], [0.1, 0.1, 0.1])
    4.4721359549995796
    >>> distance.seuclidean([1, 0, 0], [0, 1, 0], [1, 0.1, 0.1])
    3.3166247903553998
    >>> distance.seuclidean([1, 0, 0], [0, 1, 0], [10, 0.1, 0.1])

    """
    # 返回标准化欧氏距离，计算公式为 sqrt(sum(1/V_i * (u_i - v_i)^2))
    return np.sqrt(np.sum((u - v)**2 / V))
    3.1780497164141406



"""
Validate and compute the weighted Euclidean distance between vectors u and v.
"""
    # 确保向量 u 是有效的并返回有效向量
    u = _validate_vector(u)
    # 确保向量 v 是有效的并返回有效向量
    v = _validate_vector(v)
    # 确保向量 V 是有效的并返回有效向量，要求数据类型为 np.float64
    V = _validate_vector(V, dtype=np.float64)
    # 检查向量 V 的维度与向量 u 和 v 的维度是否一致，若不一致则抛出类型错误
    if V.shape[0] != u.shape[0] or u.shape[0] != v.shape[0]:
        raise TypeError('V must be a 1-D array of the same dimension '
                        'as u and v.')
    # 返回使用权重向量 V 计算的 u 和 v 之间的加权欧氏距离
    return euclidean(u, v, w=1/V)
# 定义函数，计算两个向量之间的曼哈顿距离（城市街区距离）
def cityblock(u, v, w=None):
    # 确保输入向量 u 是有效的向量形式
    u = _validate_vector(u)
    # 确保输入向量 v 是有效的向量形式
    v = _validate_vector(v)
    # 计算绝对值差，即曼哈顿距离的中间结果
    l1_diff = abs(u - v)
    # 如果有权重向量 w，则进行加权操作
    if w is not None:
        # 确保权重向量 w 是有效的向量形式
        w = _validate_weights(w)
        # 对差值应用权重
        l1_diff = w * l1_diff
    # 返回曼哈顿距离，即加权差值的总和
    return l1_diff.sum()


# 定义函数，计算两个向量之间的马氏距离
def mahalanobis(u, v, VI):
    # 确保输入向量 u 是有效的向量形式
    u = _validate_vector(u)
    # 确保输入向量 v 是有效的向量形式
    v = _validate_vector(v)
    # 确保 VI 是至少二维的数组
    VI = np.atleast_2d(VI)
    # 计算向量差
    delta = u - v
    # 计算马氏距离的中间量，使用向量差、VI 和转置的乘积
    m = np.dot(np.dot(delta, VI), delta)
    # 返回马氏距离的平方根，即最终的马氏距离
    return np.sqrt(m)


# 定义函数，计算两个向量之间的切比雪夫距离
def chebyshev(u, v, w=None):
    # 确保输入向量 u 是有效的向量形式
    u = _validate_vector(u)
    # 确保输入向量 v 是有效的向量形式
    v = _validate_vector(v)
    # 切比雪夫距离直接计算为两向量对应元素差的最大值
    chebyshev = np.abs(u - v).max()
    # 返回切比雪夫距离
    return chebyshev
    # 对向量 v 进行验证，确保其格式和内容符合要求
    v = _validate_vector(v)
    
    # 如果权重向量 w 不为 None，则对其进行验证，确保格式和内容符合要求
    if w is not None:
        w = _validate_weights(w)
        
        # 检查权重向量中大于零的元素数量，以确定有效的权重数量
        has_weight = w > 0
        
        # 如果有效权重的数量小于权重向量的大小，则根据有效权重筛选 u 和 v
        if has_weight.sum() < w.size:
            u = u[has_weight]
            v = v[has_weight]
    
    # 返回 u 和 v 中元素差的绝对值的最大值
    return max(abs(u - v))
def braycurtis(u, v, w=None):
    """
    Compute the Bray-Curtis distance between two 1-D arrays.

    Bray-Curtis distance is defined as

    .. math::

       \\sum{|u_i-v_i|} / \\sum{|u_i+v_i|}

    The Bray-Curtis distance is in the range [0, 1] if all coordinates are
    positive, and is undefined if the inputs are of length zero.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    braycurtis : double
        The Bray-Curtis distance between 1-D arrays `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.braycurtis([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.braycurtis([1, 1, 0], [0, 1, 0])
    0.33333333333333331

    """
    # Validate input vectors `u` and `v`, ensuring they are 1-D arrays
    u = _validate_vector(u)
    v = _validate_vector(v, dtype=np.float64)
    
    # Compute L1 (Manhattan) distance components
    l1_diff = abs(u - v)
    l1_sum = abs(u + v)
    
    # Apply weights `w` if provided
    if w is not None:
        w = _validate_weights(w)
        l1_diff = w * l1_diff
        l1_sum = w * l1_sum
    
    # Compute Bray-Curtis distance as the ratio of summed differences to summed sums
    return l1_diff.sum() / l1_sum.sum()


def canberra(u, v, w=None):
    """
    Compute the Canberra distance between two 1-D arrays.

    The Canberra distance is defined as

    .. math::

         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    canberra : double
        The Canberra distance between vectors `u` and `v`.

    Notes
    -----
    When ``u[i]`` and ``v[i]`` are 0 for given i, then the fraction 0/0 = 0 is
    used in the calculation.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.canberra([1, 0, 0], [0, 1, 0])
    2.0
    >>> distance.canberra([1, 1, 0], [0, 1, 0])
    1.0

    """
    # Validate input vectors `u` and `v`, ensuring they are 1-D arrays
    u = _validate_vector(u)
    v = _validate_vector(v, dtype=np.float64)
    
    # Apply weights `w` if provided
    if w is not None:
        w = _validate_weights(w)
    
    # Calculate Canberra distance using numpy functions, handling division by zero with NaN handling
    with np.errstate(invalid='ignore'):
        abs_uv = abs(u - v)
        abs_u = abs(u)
        abs_v = abs(v)
        d = abs_uv / (abs_u + abs_v)
        if w is not None:
            d = w * d
        d = np.nansum(d)
    
    return d


def jensenshannon(p, q, base=None, *, axis=0, keepdims=False):
    """
    Compute the Jensen-Shannon distance (metric) between
    two probability arrays. This is the square root
    of the Jensen-Shannon divergence.

    The Jensen-Shannon distance between two probability
    vectors `p` and `q` is defined as,

    .. math::

       \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}

    where :math:`m` is the pointwise mean of :math:`p` and :math:`q`

    Parameters
    ----------
    p : (N,) array_like
        Input probability vector.
    q : (N,) array_like
        Input probability vector.
    base : double, optional
        The base of the logarithm used in the Jensen-Shannon divergence.
        Default is None, which implies the natural logarithm (base e).
    axis : int, optional
        The axis along which the Jensen-Shannon divergence is computed.
        Default is 0.
    keepdims : bool, optional
        If True, the original dimensions of the array are kept.
        Default is False.

    Returns
    -------
    jensenshannon : double
        The Jensen-Shannon distance between probability arrays `p` and `q`.

    """
    # Validate input probability vectors `p` and `q`
    p = _validate_vector(p, dtype=np.float64)
    q = _validate_vector(q, dtype=np.float64)
    
    # Compute the pointwise mean of `p` and `q`
    m = (p + q) / 2.0
    
    # Compute the Jensen-Shannon divergence between `p` and `q`
    d_p = entropy(p, m, base=base)
    d_q = entropy(q, m, base=base)
    
    # Compute the Jensen-Shannon distance as the square root of the average divergence
    return np.sqrt((d_p + d_q) / 2.0)
    # 将输入的概率向量转换为 NumPy 数组
    p = np.asarray(p)
    q = np.asarray(q)
    # 对 p 和 q 进行归一化，使它们的元素和为1，axis 参数指定归一化操作的轴
    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)
    # 计算 p 和 q 的平均值向量
    m = (p + q) / 2.0
    # 计算 p 和 m 的相对熵（Kullback-Leibler 散度）
    left = rel_entr(p, m)
    # 计算 q 和 m 的相对熵（Kullback-Leibler 散度）
    right = rel_entr(q, m)
    # 按指定轴对相对熵进行求和，得到左侧和右侧的和
    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    # 计算 Jensen-Shannon 距离
    js = left_sum + right_sum
    # 如果给定了 base 参数，则除以 log(base)，以改变计算的基数
    if base is not None:
        js /= np.log(base)
    # 返回 Jensen-Shannon 距离的平方根除以 2
    return np.sqrt(js / 2.0)
def dice(u, v, w=None):
    """
    Compute the Dice dissimilarity between two boolean 1-D arrays.

    The Dice dissimilarity between `u` and `v`, is

    .. math::

         \\frac{c_{TF} + c_{FT}}
              {2c_{TT} + c_{FT} + c_{TF}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input 1-D array.
    v : (N,) array_like, bool
        Input 1-D array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    dice : double
        The Dice dissimilarity between 1-D arrays `u` and `v`.

    Notes
    -----
    This function computes the Dice dissimilarity index. To compute the
    Dice similarity index, convert one to the other with similarity =
    1 - dissimilarity.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.dice([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.dice([1, 0, 0], [1, 1, 0])
    0.3333333333333333
    >>> distance.dice([1, 0, 0], [2, 0, 0])
    -0.3333333333333333

    """
    # Validate input vectors `u` and `v`, ensuring they are boolean arrays
    u = _validate_vector(u)
    v = _validate_vector(v)
    
    # Validate optional weights `w`, if provided
    if w is not None:
        w = _validate_weights(w)
    
    # If `u` and `v` are boolean and no weights are provided, compute `ntt` directly
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()
    else:
        # Convert `u` and `v` to appropriate type for element-wise multiplication
        dtype = np.result_type(int, u.dtype, v.dtype)
        u = u.astype(dtype)
        v = v.astype(dtype)
        
        # Calculate `ntt` considering weights `w`, if provided
        if w is None:
            ntt = (u * v).sum()
        else:
            ntt = (u * v * w).sum()
    
    # Calculate `nft` and `ntf` using helper function `_nbool_correspond_ft_tf`
    (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
    # 返回一个浮点数，计算 (ntf + nft) / np.array(2.0 * ntt + ntf + nft) 的结果
    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))
def rogerstanimoto(u, v, w=None):
    """
    Compute the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.

    The Rogers-Tanimoto dissimilarity between two boolean 1-D arrays
    `u` and `v`, is defined as

    .. math::
       \\frac{R}
            {c_{TT} + c_{FF} + R}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    rogerstanimoto : double
        The Rogers-Tanimoto dissimilarity between vectors
        `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.rogerstanimoto([1, 0, 0], [0, 1, 0])
    0.8
    >>> distance.rogerstanimoto([1, 0, 0], [1, 1, 0])
    0.5
    >>> distance.rogerstanimoto([1, 0, 0], [2, 0, 0])
    -1.0

    """
    # Validate input vectors `u` and `v`
    u = _validate_vector(u)
    v = _validate_vector(v)
    # Validate weights `w` if provided
    if w is not None:
        w = _validate_weights(w)
    # Compute counts of different boolean combinations
    (nff, nft, ntf, ntt) = _nbool_correspond_all(u, v, w=w)
    # Calculate Rogers-Tanimoto dissimilarity using the formula
    return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))


def russellrao(u, v, w=None):
    """
    Compute the Russell-Rao dissimilarity between two boolean 1-D arrays.

    The Russell-Rao dissimilarity between two boolean 1-D arrays, `u` and
    `v`, is defined as

    .. math::

      \\frac{n - c_{TT}}
           {n}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    russellrao : double
        The Russell-Rao dissimilarity between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.russellrao([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.russellrao([1, 0, 0], [1, 1, 0])
    0.6666666666666666
    >>> distance.russellrao([1, 0, 0], [2, 0, 0])
    0.3333333333333333

    """
    # Validate input vectors `u` and `v`
    u = _validate_vector(u)
    v = _validate_vector(v)
    # Check for boolean type and handle weights `w`
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()  # Count true-true occurrences
        n = float(len(u))  # Total number of elements
    elif w is None:
        ntt = (u * v).sum()  # Count true-true occurrences with non-boolean input
        n = float(len(u))  # Total number of elements
    else:
        w = _validate_weights(w)  # Validate weights `w`
        ntt = (u * v * w).sum()  # Weighted count of true-true occurrences
        n = w.sum()  # Sum of weights
    # Calculate Russell-Rao dissimilarity using the formula
    return float(n - ntt) / n


def sokalmichener(u, v, w=None):
    """
    Compute the Sokal-Michener dissimilarity between two boolean 1-D arrays.

    """
    """
    Calculate the Sokal-Michener dissimilarity between boolean 1-D arrays `u` and `v`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array `u`.
    v : (N,) array_like, bool
        Input array `v`.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None, which gives each value a weight of 1.0.

    Returns
    -------
    sokalmichener : double
        The Sokal-Michener dissimilarity between vectors `u` and `v`.

    Notes
    -----
    The Sokal-Michener dissimilarity is defined as:

    .. math::

       \\frac{2 * (c_{TF} + c_{FT})}
            {c_{FF} + c_{TT} + 2 * (c_{TF} + c_{FT})}

    where:
    - :math:`c_{ij}` is the count of occurrences where `u[k] = i` and `v[k] = j` for `k < n`.
    - :math:`c_{TF}` is the count where `u[k] = True` and `v[k] = False`.
    - :math:`c_{FT}` is the count where `u[k] = False` and `v[k] = True`.
    - :math:`c_{FF}` is the count where `u[k] = False` and `v[k] = False`.
    - :math:`c_{TT}` is the count where `u[k] = True` and `v[k] = True`.
    - :math:`N` is the length of arrays `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.sokalmichener([1, 0, 0], [0, 1, 0])
    0.8
    >>> distance.sokalmichener([1, 0, 0], [1, 1, 0])
    0.5
    >>> distance.sokalmichener([1, 0, 0], [2, 0, 0])
    -1.0
    """
    # Validate input vectors u and v
    u = _validate_vector(u)
    v = _validate_vector(v)
    
    # Validate weights w if provided
    if w is not None:
        w = _validate_weights(w)
    
    # Calculate the counts of all combinations of boolean values between u and v
    nff, nft, ntf, ntt = _nbool_correspond_all(u, v, w=w)
    
    # Compute the Sokal-Michener dissimilarity using the counts
    # Formula: 2 * (ntf + nft) / (ntt + nff + 2 * (ntf + nft))
    return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))
def sokalsneath(u, v, w=None):
    """
    Compute the Sokal-Sneath dissimilarity between two boolean 1-D arrays.

    The Sokal-Sneath dissimilarity between `u` and `v`,

    .. math::

       \\frac{R}
            {c_{TT} + R}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    sokalsneath : double
        The Sokal-Sneath dissimilarity between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.sokalsneath([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.sokalsneath([1, 0, 0], [1, 1, 0])
    0.66666666666666663
    >>> distance.sokalsneath([1, 0, 0], [2, 1, 0])
    0.0
    >>> distance.sokalsneath([1, 0, 0], [3, 1, 0])
    -2.0

    """
    # 验证输入向量 `u` 和 `v`，确保它们是有效的向量
    u = _validate_vector(u)
    v = _validate_vector(v)
    # 如果 `u` 和 `v` 的数据类型为布尔型，并且没有提供权重 `w`，则计算 `c_{TT}`
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()
    # 如果没有提供权重 `w`，则计算 `c_{TT}`
    elif w is None:
        ntt = (u * v).sum()
    # 否则，验证权重 `w` 并计算 `c_{TT}`
    else:
        w = _validate_weights(w)
        ntt = (u * v * w).sum()
    # 计算 `c_{FT}` 和 `c_{TF}`
    (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
    # 计算分母 `c_{TT} + 2.0 * (c_{TF} + c_{FT})`
    denom = np.array(ntt + 2.0 * (ntf + nft))
    # 如果分母为零，则抛出异常
    if not denom.any():
        raise ValueError('Sokal-Sneath dissimilarity is not defined for '
                         'vectors that are entirely false.')
    # 返回 Sokal-Sneath dissimilarity 指数
    return float(2.0 * (ntf + nft)) / denom


# 使用 partial 函数将 _convert_to_type 函数应用到双精度浮点类型的转换
_convert_to_double = partial(_convert_to_type, out_type=np.float64)
# 使用 partial 函数将 _convert_to_type 函数应用到布尔类型的转换
_convert_to_bool = partial(_convert_to_type, out_type=bool)

# 将 python-only wrappers 添加到 _distance_wrap 模块中
_distance_wrap.pdist_correlation_double_wrap = _correlation_pdist_wrap
_distance_wrap.cdist_correlation_double_wrap = _correlation_cdist_wrap


@dataclasses.dataclass(frozen=True)
class CDistMetricWrapper:
    metric_name: str

    def __call__(self, XA, XB, *, out=None, **kwargs):
        # 将输入数组 `XA` 和 `XB` 转换为连续的内存布局
        XA = np.ascontiguousarray(XA)
        XB = np.ascontiguousarray(XB)
        # 获取数组 `XA` 的形状
        mA, n = XA.shape
        # 获取数组 `XB` 的形状
        mB, _ = XB.shape
        # 获取度量名称 `metric_name`
        metric_name = self.metric_name
        # 获取度量信息 `_METRICS[metric_name]`
        metric_info = _METRICS[metric_name]
        # 验证输入的 cdist 参数，确保其有效性
        XA, XB, typ, kwargs = _validate_cdist_input(
            XA, XB, mA, mB, n, metric_info, **kwargs)

        # 弹出参数中的权重 `w`
        w = kwargs.pop('w', None)
        # 如果提供了权重 `w`，则使用 `_cdist_callable` 计算结果
        if w is not None:
            metric = metric_info.dist_func
            return _cdist_callable(
                XA, XB, metric=metric, out=out, w=w, **kwargs)

        # 准备输出参数 `out`，使用双精度浮点类型和形状 `(mA, mB)`
        dm = _prepare_out_argument(out, np.float64, (mA, mB))
        # 获取 cdist 包装函数 `cdist_fn`
        cdist_fn = getattr(_distance_wrap, f'cdist_{metric_name}_{typ}_wrap')
        # 调用 cdist 包装函数 `cdist_fn`
        cdist_fn(XA, XB, dm, **kwargs)
        # 返回计算结果 `dm`
        return dm


@dataclasses.dataclass(frozen=True)
class PDistMetricWrapper:
    metric_name: str
    # 定义类的调用方法，将参数X转换为连续存储的NumPy数组
    def __call__(self, X, *, out=None, **kwargs):
        X = np.ascontiguousarray(X)
        # 获取数组X的形状，m为行数，n为列数
        m, n = X.shape
        # 获取距离度量的名称
        metric_name = self.metric_name
        # 根据度量名称从预定义的度量信息字典_METRICS中获取相应的信息
        metric_info = _METRICS[metric_name]
        # 验证输入数据X，确保符合距离度量函数的要求，并返回验证后的结果
        X, typ, kwargs = _validate_pdist_input(
            X, m, n, metric_info, **kwargs)
        # 计算输出向量的长度，对应于X中所有两两组合的距离
        out_size = (m * (m - 1)) // 2
        # 从kwargs中获取权重w，如果存在，则使用度量信息中的距离函数进行计算
        w = kwargs.pop('w', None)
        if w is not None:
            metric = metric_info.dist_func
            # 如果有权重w，则调用_pdist_callable计算加权距离
            return _pdist_callable(
                X, metric=metric, out=out, w=w, **kwargs)

        # 准备输出参数dm，用于存储计算得到的距离
        dm = _prepare_out_argument(out, np.float64, (out_size,))
        # 获取对应度量函数的封装函数，用于计算距离矩阵
        pdist_fn = getattr(_distance_wrap, f'pdist_{metric_name}_{typ}_wrap')
        # 调用距离计算函数pdist_fn计算距离，并将结果存储在dm中
        pdist_fn(X, dm, **kwargs)
        # 返回计算得到的距离矩阵dm
        return dm
@dataclasses.dataclass(frozen=True)
class MetricInfo:
    # Name of python distance function
    canonical_name: str
    # All aliases, including canonical_name
    aka: set[str]
    # unvectorized distance function
    dist_func: Callable
    # Optimized cdist function
    cdist_func: Callable
    # Optimized pdist function
    pdist_func: Callable
    # function that checks kwargs and computes default values:
    # f(X, m, n, **kwargs)
    validator: Optional[Callable] = None
    # list of supported types:
    # X (pdist) and XA (cdist) are used to choose the type. if there is no
    # match the first type is used. Default double
    types: list[str] = dataclasses.field(default_factory=lambda: ['double'])
    # true if out array must be C-contiguous
    requires_contiguous_out: bool = True


# Registry of implemented metrics:
_METRIC_INFOS = [
    MetricInfo(
        canonical_name='braycurtis',
        aka={'braycurtis'},
        dist_func=braycurtis,
        cdist_func=_distance_pybind.cdist_braycurtis,
        pdist_func=_distance_pybind.pdist_braycurtis,
    ),
    MetricInfo(
        canonical_name='canberra',
        aka={'canberra'},
        dist_func=canberra,
        cdist_func=_distance_pybind.cdist_canberra,
        pdist_func=_distance_pybind.pdist_canberra,
    ),
    MetricInfo(
        canonical_name='chebyshev',
        aka={'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch'},
        dist_func=chebyshev,
        cdist_func=_distance_pybind.cdist_chebyshev,
        pdist_func=_distance_pybind.pdist_chebyshev,
    ),
    MetricInfo(
        canonical_name='cityblock',
        aka={'cityblock', 'cblock', 'cb', 'c'},
        dist_func=cityblock,
        cdist_func=_distance_pybind.cdist_cityblock,
        pdist_func=_distance_pybind.pdist_cityblock,
    ),
    MetricInfo(
        canonical_name='correlation',
        aka={'correlation', 'co'},
        dist_func=correlation,
        cdist_func=CDistMetricWrapper('correlation'),
        pdist_func=PDistMetricWrapper('correlation'),
    ),
    MetricInfo(
        canonical_name='cosine',
        aka={'cosine', 'cos'},
        dist_func=cosine,
        cdist_func=CDistMetricWrapper('cosine'),
        pdist_func=PDistMetricWrapper('cosine'),
    ),
    MetricInfo(
        canonical_name='dice',
        aka={'dice'},
        types=['bool'],
        dist_func=dice,
        cdist_func=_distance_pybind.cdist_dice,
        pdist_func=_distance_pybind.pdist_dice,
    ),
    MetricInfo(
        canonical_name='euclidean',
        aka={'euclidean', 'euclid', 'eu', 'e'},
        dist_func=euclidean,
        cdist_func=_distance_pybind.cdist_euclidean,
        pdist_func=_distance_pybind.pdist_euclidean,
    ),
]


注释：

# 数据类 MetricInfo 的定义，用于存储距离函数相关信息，不可变
@dataclasses.dataclass(frozen=True)
class MetricInfo:
    # Python 距离函数的标准名称
    canonical_name: str
    # 包括 canonical_name 在内的所有别名集合
    aka: set[str]
    # 未向量化的距离函数
    dist_func: Callable
    # 优化后的 cdist 函数
    cdist_func: Callable
    # 优化后的 pdist 函数
    pdist_func: Callable
    # 检查关键字参数并计算默认值的函数
    # f(X, m, n, **kwargs)
    validator: Optional[Callable] = None
    # 支持的类型列表
    # X（pdist）和 XA（cdist）用于选择类型。如果没有匹配，则使用第一个类型。默认为 double
    types: list[str] = dataclasses.field(default_factory=lambda: ['double'])
    # 如果输出数组必须是 C 连续的，则为 True
    requires_contiguous_out: bool = True

# 已实现距离度量函数的注册表
_METRIC_INFOS = [
    MetricInfo(
        canonical_name='braycurtis',
        aka={'braycurtis'},
        dist_func=braycurtis,
        cdist_func=_distance_pybind.cdist_braycurtis,
        pdist_func=_distance_pybind.pdist_braycurtis,
    ),
    MetricInfo(
        canonical_name='canberra',
        aka={'canberra'},
        dist_func=canberra,
        cdist_func=_distance_pybind.cdist_canberra,
        pdist_func=_distance_pybind.pdist_canberra,
    ),
    MetricInfo(
        canonical_name='chebyshev',
        aka={'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch'},
        dist_func=chebyshev,
        cdist_func=_distance_pybind.cdist_chebyshev,
        pdist_func=_distance_pybind.pdist_chebyshev,
    ),
    MetricInfo(
        canonical_name='cityblock',
        aka={'cityblock', 'cblock', 'cb', 'c'},
        dist_func=cityblock,
        cdist_func=_distance_pybind.cdist_cityblock,
        pdist_func=_distance_pybind.pdist_cityblock,
    ),
    MetricInfo(
        canonical_name='correlation',
        aka={'correlation', 'co'},
        dist_func=correlation,
        cdist_func=CDistMetricWrapper('correlation'),
        pdist_func=PDistMetricWrapper('correlation'),
    ),
    MetricInfo(
        canonical_name='cosine',
        aka={'cosine', 'cos'},
        dist_func=cosine,
        cdist_func=CDistMetricWrapper('cosine'),
        pdist_func=PDistMetricWrapper('cosine'),
    ),
    MetricInfo(
        canonical_name='dice',
        aka={'dice'},
        types=['bool'],
        dist_func=dice,
        cdist_func=_distance_pybind.cdist_dice,
        pdist_func=_distance_pybind.pdist_dice,
    ),
    MetricInfo(
        canonical_name='euclidean',
        aka={'euclidean', 'euclid', 'eu', 'e'},
        dist_func=euclidean,
        cdist_func=_distance_pybind.cdist_euclidean,
        pdist_func=_distance_pybind.pdist_euclidean,
    ),
]
    MetricInfo(
        canonical_name='hamming',  # 指定距离度量的规范名称为'hamming'
        aka={'matching', 'hamming', 'hamm', 'ha', 'h'},  # 别名集合包括'matching', 'hamming', 'hamm', 'ha', 'h'
        types=['double', 'bool'],  # 支持的数据类型为双精度浮点数和布尔型
        validator=_validate_hamming_kwargs,  # 使用_validate_hamming_kwargs函数验证参数
        dist_func=hamming,  # 使用hamming函数计算距离
        cdist_func=_distance_pybind.cdist_hamming,  # 使用_distance_pybind.cdist_hamming函数计算距离矩阵
        pdist_func=_distance_pybind.pdist_hamming,  # 使用_distance_pybind.pdist_hamming函数计算距离矩阵
    ),
    MetricInfo(
        canonical_name='jaccard',  # 指定距离度量的规范名称为'jaccard'
        aka={'jaccard', 'jacc', 'ja', 'j'},  # 别名集合包括'jaccard', 'jacc', 'ja', 'j'
        types=['double', 'bool'],  # 支持的数据类型为双精度浮点数和布尔型
        dist_func=jaccard,  # 使用jaccard函数计算距离
        cdist_func=_distance_pybind.cdist_jaccard,  # 使用_distance_pybind.cdist_jaccard函数计算距离矩阵
        pdist_func=_distance_pybind.pdist_jaccard,  # 使用_distance_pybind.pdist_jaccard函数计算距离矩阵
    ),
    MetricInfo(
        canonical_name='jensenshannon',  # 指定距离度量的规范名称为'jensenshannon'
        aka={'jensenshannon', 'js'},  # 别名集合包括'jensenshannon', 'js'
        dist_func=jensenshannon,  # 使用jensenshannon函数计算距离
        cdist_func=CDistMetricWrapper('jensenshannon'),  # 使用CDistMetricWrapper('jensenshannon')对象计算距离矩阵
        pdist_func=PDistMetricWrapper('jensenshannon'),  # 使用PDistMetricWrapper('jensenshannon')对象计算距离矩阵
    ),
    MetricInfo(
        canonical_name='kulczynski1',  # 指定距离度量的规范名称为'kulczynski1'
        aka={'kulczynski1'},  # 别名集合仅包括'kulczynski1'
        types=['bool'],  # 支持的数据类型为布尔型
        dist_func=kulczynski1,  # 使用kulczynski1函数计算距离
        cdist_func=_distance_pybind.cdist_kulczynski1,  # 使用_distance_pybind.cdist_kulczynski1函数计算距离矩阵
        pdist_func=_distance_pybind.pdist_kulczynski1,  # 使用_distance_pybind.pdist_kulczynski1函数计算距离矩阵
    ),
    MetricInfo(
        canonical_name='mahalanobis',  # 指定距离度量的规范名称为'mahalanobis'
        aka={'mahalanobis', 'mahal', 'mah'},  # 别名集合包括'mahalanobis', 'mahal', 'mah'
        validator=_validate_mahalanobis_kwargs,  # 使用_validate_mahalanobis_kwargs函数验证参数
        dist_func=mahalanobis,  # 使用mahalanobis函数计算距离
        cdist_func=CDistMetricWrapper('mahalanobis'),  # 使用CDistMetricWrapper('mahalanobis')对象计算距离矩阵
        pdist_func=PDistMetricWrapper('mahalanobis'),  # 使用PDistMetricWrapper('mahalanobis')对象计算距离矩阵
    ),
    MetricInfo(
        canonical_name='minkowski',  # 指定距离度量的规范名称为'minkowski'
        aka={'minkowski', 'mi', 'm', 'pnorm'},  # 别名集合包括'minkowski', 'mi', 'm', 'pnorm'
        validator=_validate_minkowski_kwargs,  # 使用_validate_minkowski_kwargs函数验证参数
        dist_func=minkowski,  # 使用minkowski函数计算距离
        cdist_func=_distance_pybind.cdist_minkowski,  # 使用_distance_pybind.cdist_minkowski函数计算距离矩阵
        pdist_func=_distance_pybind.pdist_minkowski,  # 使用_distance_pybind.pdist_minkowski函数计算距离矩阵
    ),
    MetricInfo(
        canonical_name='rogerstanimoto',  # 指定距离度量的规范名称为'rogerstanimoto'
        aka={'rogerstanimoto'},  # 别名集合仅包括'rogerstanimoto'
        types=['bool'],  # 支持的数据类型为布尔型
        dist_func=rogerstanimoto,  # 使用rogerstanimoto函数计算距离
        cdist_func=_distance_pybind.cdist_rogerstanimoto,  # 使用_distance_pybind.cdist_rogerstanimoto函数计算距离矩阵
        pdist_func=_distance_pybind.pdist_rogerstanimoto,  # 使用_distance_pybind.pdist_rogerstanimoto函数计算距离矩阵
    ),
    MetricInfo(
        canonical_name='russellrao',  # 指定距离度量的规范名称为'russellrao'
        aka={'russellrao'},  # 别名集合仅包括'russellrao'
        types=['bool'],  # 支持的数据类型为布尔型
        dist_func=russellrao,  # 使用russellrao函数计算距离
        cdist_func=_distance_pybind.cdist_russellrao,  # 使用_distance_pybind.cdist_russellrao函数计算距离矩阵
        pdist_func=_distance_pybind.pdist_russellrao,  # 使用_distance_pybind.pdist_russellrao函数计算距离矩阵
    ),
    MetricInfo(
        canonical_name='seuclidean',  # 指定距离度量的规范名称为'seuclidean'
        aka={'seuclidean', 'se', 's'},  # 别名集合包括'seuclidean', 'se', 's'
        validator=_validate_seuclidean_kwargs,  # 使用_validate_seuclidean_kwargs函数验证参数
        dist_func=seuclidean,  # 使用seuclidean函数计算距离
        cdist_func=CDistMetricWrapper('seuclidean'),  # 使用CDistMetricWrapper('seuclidean')对象计算距离矩阵
        pdist_func=PDistMetricWrapper('seuclidean'),  # 使用PDistMetricWrapper('seuclidean')对象计算距离矩阵
    ),
    MetricInfo(
        canonical_name='sokalmichener',  # 指定距离度量的规范名称为'sokalmichener'
        aka={'sokalmichener'},  # 别名集合仅包括'sokalmichener'
        types=['bool'],  # 支持的数据类型为布尔型
        dist_func=sokalmichener,  # 使用sokalmichener函数计算距离
        cdist_func=_distance_pybind.cdist_sokalmichener,  # 使用_distance_pybind.cdist_sokalmichener函数计算距离矩阵
        pdist_func=_distance_pybind.pdist_sokalmichener,  # 使用_distance_pybind.pdist_sokalmichener函数计算距离矩阵
    ),
    MetricInfo(
        canonical_name='sokalsneath',  # 指定距离度量的规范名称为'sokalsneath'
        aka={'sokalsneath'},  #
    # 创建 MetricInfo 对象，表示 sqeuclidean 度量
    MetricInfo(
        canonical_name='sqeuclidean',  # 规范名称为 'sqeuclidean'
        aka={'sqeuclidean', 'sqe', 'sqeuclid'},  # 别名集合包括 'sqeuclidean', 'sqe', 'sqeuclid'
        dist_func=sqeuclidean,  # 距离函数使用 sqeuclidean
        cdist_func=_distance_pybind.cdist_sqeuclidean,  # 使用 C 扩展的 cdist_sqeuclidean 函数
        pdist_func=_distance_pybind.pdist_sqeuclidean,  # 使用 C 扩展的 pdist_sqeuclidean 函数
    ),

    # 创建 MetricInfo 对象，表示 yule 度量
    MetricInfo(
        canonical_name='yule',  # 规范名称为 'yule'
        aka={'yule'},  # 别名集合仅包括 'yule'
        types=['bool'],  # 度量适用于布尔类型的数据
        dist_func=yule,  # 距离函数使用 yule
        cdist_func=_distance_pybind.cdist_yule,  # 使用 C 扩展的 cdist_yule 函数
        pdist_func=_distance_pybind.pdist_yule,  # 使用 C 扩展的 pdist_yule 函数
    ),
# 将 _METRIC_INFOS 列表中的每个元素 info 转换为以 info.canonical_name 为键的字典，并存储在 _METRICS 中
_METRICS = {info.canonical_name: info for info in _METRIC_INFOS}

# 将 _METRIC_INFOS 列表中的每个元素 info 转换为字典，其中键为 info.aka 中的每个别名 alias，值为 info
_METRIC_ALIAS = {alias: info
                     for info in _METRIC_INFOS
                     for alias in info.aka}

# 生成包含 _METRICS 字典中所有键的列表，并将其存储在 _METRICS_NAMES 中
_METRICS_NAMES = list(_METRICS.keys())

# 将 _METRIC_INFOS 列表中的每个元素 info 转换为一个新字典，其中键为 'test_' + info.canonical_name，值为 info
_TEST_METRICS = {'test_' + info.canonical_name: info for info in _METRIC_INFOS}


def pdist(X, metric='euclidean', *, out=None, **kwargs):
    """
    Pairwise distances between observations in n-dimensional space.

    See Notes for common calling conventions.

    Parameters
    ----------
    X : array_like
        An m by n array of m original observations in an
        n-dimensional space.
    metric : str or function, optional
        The distance metric to use. The distance function can
        be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'jensenshannon', 'kulczynski1',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'yule'.
    out : ndarray, optional
        The output array.
        If not None, condensed distance matrix Y is stored in this array.
    **kwargs : dict, optional
        Extra arguments to `metric`: refer to each metric documentation for a
        list of all possible arguments.

        Some possible arguments:

        p : scalar
            The p-norm to apply for Minkowski, weighted and unweighted.
            Default: 2.

        w : ndarray
            The weight vector for metrics that support weights (e.g., Minkowski).

        V : ndarray
            The variance vector for standardized Euclidean.
            Default: var(X, axis=0, ddof=1)

        VI : ndarray
            The inverse of the covariance matrix for Mahalanobis.
            Default: inv(cov(X.T)).T

    Returns
    -------
    Y : ndarray
        Returns a condensed distance matrix Y. For each :math:`i` and :math:`j`
        (where :math:`i<j<m`),where m is the number of original observations.
        The metric ``dist(u=X[i], v=X[j])`` is computed and stored in entry ``m
        * i + j - ((i + 2) * (i + 1)) // 2``.

    See Also
    --------
    squareform : converts between condensed distance matrices and
                 square distance matrices.

    Notes
    -----
    See ``squareform`` for information on how to calculate the index of
    this entry or to convert the condensed distance matrix to a
    redundant square matrix.

    The following are common calling conventions.

    1. ``Y = pdist(X, 'euclidean')``

       Computes the distance between m points using Euclidean distance
       (2-norm) as the distance metric between the points. The points
       are arranged as m n-dimensional row vectors in the matrix X.
    """
    # 计算使用 Minkowski 距离 :math:`\\|u-v\\|_p` (:math:`p`-norm) 的距离，这里的 `p=2.` 表示欧几里得距离。
    Y = pdist(X, 'minkowski', p=2.)
    
    # 计算城市街区距离（曼哈顿距离）。
    Y = pdist(X, 'cityblock')
    
    # 计算标准化欧几里得距离。对于两个 n-向量 u 和 v，标准化欧几里得距离定义如下：
    # .. math::
    #    \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}
    # 其中 V 是方差向量；V[i] 是计算所有点的第 i 个分量的方差。如果未传递 V，则自动计算。
    Y = pdist(X, 'seuclidean', V=None)
    
    # 计算平方欧几里得距离 :math:`\\|u-v\\|_2^2`。
    Y = pdist(X, 'sqeuclidean')
    
    # 计算向量 u 和 v 之间的余弦距离。
    # .. math::
    #    1 - \\frac{u \\cdot v}{{\\|u\\|}_2 {\\|v\\|}_2}
    # 其中 :math:`\\|*\\|_2` 是其参数的 2-范数，:math:`u \\cdot v` 是向量 u 和 v 的点积。
    Y = pdist(X, 'cosine')
    
    # 计算向量 u 和 v 之间的相关距离。
    # .. math::
    #    1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}{{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
    # 其中 :math:`\\bar{v}` 是向量 v 元素的均值，:math:`x \\cdot y` 是向量 x 和 y 的点积。
    Y = pdist(X, 'correlation')
    
    # 计算归一化汉明距离，即两个 n-向量 u 和 v 之间不同元素的比例。
    # 为节省内存，矩阵 X 可以是布尔类型。
    Y = pdist(X, 'hamming')
    
    # 计算 Jaccard 距离，即两个向量 u 和 v 之间不同元素的比例。
    Y = pdist(X, 'jaccard')
    
    # 计算 Jensen-Shannon 距离，用于两个概率数组之间的距离。
    # 给定两个概率向量 :math:`p` 和 :math:`q`，Jensen-Shannon 距离定义如下：
    # .. math::
    #    \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
    # 其中 :math:`m` 是 :math:`p` 和 :math:`q` 的逐点均值，:math:`D` 是 Kullback-Leibler 散度。
    Y = pdist(X, 'jensenshannon')
    
    # 计算 Chebyshev 距离，即两个 n-向量 u 和 v 之间的最大 L1 距离。
    # 具体来说，距离定义为：
    # .. math::
    #    d(u,v) = \\max_i {|u_i-v_i|}
    Y = pdist(X, 'chebyshev')
    12. ```Y = pdist(X, 'canberra')```
        
        计算点之间的 Canberra 距离。两点 u 和 v 之间的 Canberra 距离定义为：
        
        .. math::
        
          d(u,v) = \\sum_i \\frac{|u_i - v_i|}
                               {|u_i| + |v_i|}
                               
    13. ```Y = pdist(X, 'braycurtis')```
        
        计算点之间的 Bray-Curtis 距离。两点 u 和 v 之间的 Bray-Curtis 距离定义为：
        
        .. math::
        
             d(u,v) = \\frac{\\sum_i {|u_i - v_i|}}
                            {\\sum_i {|u_i + v_i|}}
                            
    14. ```Y = pdist(X, 'mahalanobis', VI=None)```
        
        计算点之间的 Mahalanobis 距离。两点 u 和 v 之间的 Mahalanobis 距离定义为：
        
        :math:`\\sqrt{(u-v)(1/V)(u-v)^T}`
        
        其中 :math:`(1/V)`（变量 `VI`）为协方差的逆矩阵。如果 `VI` 不为 None，则使用 `VI` 作为逆协方差矩阵。
        
    15. ```Y = pdist(X, 'yule')```
        
        计算布尔向量之间的 Yule 距离。（参见 yule 函数文档）
        
    16. ```Y = pdist(X, 'matching')```
        
        'hamming' 的同义词，计算布尔向量之间的匹配距离。
        
    17. ```Y = pdist(X, 'dice')```
        
        计算布尔向量之间的 Dice 距离。（参见 dice 函数文档）
        
    18. ```Y = pdist(X, 'kulczynski1')```
        
        计算布尔向量之间的 Kulczynski 1 距离。（参见 kulczynski1 函数文档）
        
    19. ```Y = pdist(X, 'rogerstanimoto')```
        
        计算布尔向量之间的 Rogers-Tanimoto 距离。（参见 rogerstanimoto 函数文档）
        
    20. ```Y = pdist(X, 'russellrao')```
        
        计算布尔向量之间的 Russell-Rao 距离。（参见 russellrao 函数文档）
        
    21. ```Y = pdist(X, 'sokalmichener')```
        
        计算布尔向量之间的 Sokal-Michener 距离。（参见 sokalmichener 函数文档）
        
    22. ```Y = pdist(X, 'sokalsneath')```
        
        计算布尔向量之间的 Sokal-Sneath 距离。（参见 sokalsneath 函数文档）
        
    23. ```Y = pdist(X, 'kulczynski1')```
        
        计算布尔向量之间的 Kulczynski 1 距离。（参见 kulczynski1 函数文档）
    # 将 X 转换为验证后的数组，确保其为非稀疏、允许对象和掩码，且不检查有限性
    X = _asarray_validated(X, sparse_ok=False, objects_ok=True, mask_ok=True,
                           check_finite=False)

    # 获取数组 X 的形状
    s = X.shape

    # 如果数组 X 不是二维的，则引发 ValueError 异常
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    # 获取数组 X 的行数 m 和列数 n
    m, n = s

    # 如果 metric 是可调用的函数，则获取其名称并查找是否有对应的度量信息
    if callable(metric):
        mstr = getattr(metric, '__name__', 'UnknownCustomMetric')
        metric_info = _METRIC_ALIAS.get(mstr, None)

        # 如果找到了度量信息，则验证输入并返回验证后的 X、类型和额外参数 kwargs
        if metric_info is not None:
            X, typ, kwargs = _validate_pdist_input(
                X, m, n, metric_info, **kwargs)

        # 调用 _pdist_callable 函数计算基于 metric 函数的距离，并返回结果
        return _pdist_callable(X, metric=metric, out=out, **kwargs)
    # 如果 metric 是字符串类型，则将其转换为小写
    elif isinstance(metric, str):
        mstr = metric.lower()
        # 从 _METRIC_ALIAS 字典中获取与 mstr 对应的 metric_info
        metric_info = _METRIC_ALIAS.get(mstr, None)

        # 如果找到了匹配的 metric_info
        if metric_info is not None:
            # 获取距离函数 pdist_fn
            pdist_fn = metric_info.pdist_func
            # 使用 pdist_fn 计算距离，并返回结果
            return pdist_fn(X, out=out, **kwargs)
        
        # 如果未在 _METRIC_ALIAS 中找到匹配项，但是以 "test_" 开头
        elif mstr.startswith("test_"):
            # 从 _TEST_METRICS 字典中获取与 mstr 对应的 metric_info
            metric_info = _TEST_METRICS.get(mstr, None)
            # 如果找不到对应的 metric_info，则引发异常
            if metric_info is None:
                raise ValueError(f'Unknown "Test" Distance Metric: {mstr[5:]}')
            # 验证输入并返回验证后的输入数据 X, typ, kwargs
            X, typ, kwargs = _validate_pdist_input(
                X, m, n, metric_info, **kwargs)
            # 使用 _pdist_callable 计算距离，并返回结果
            return _pdist_callable(
                X, metric=metric_info.dist_func, out=out, **kwargs)
        
        # 如果 metric 不以 "test_" 开头，并且未在 _METRIC_ALIAS 中找到匹配项
        else:
            # 抛出异常，说明找不到对应的距离度量标准
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    
    # 如果 metric 不是字符串类型，则抛出类型错误异常
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
# 将向量形式的距离向量或方阵转换为方阵形式的距离矩阵，反之亦然。
def squareform(X, force="no", checks=True):
    """
    Convert a vector-form distance vector to a square-form distance
    matrix, and vice-versa.

    Parameters
    ----------
    X : array_like
        Either a condensed or redundant distance matrix.
    force : str, optional
        As with MATLAB(TM), if force is equal to ``'tovector'`` or
        ``'tomatrix'``, the input will be treated as a distance matrix or
        distance vector respectively.
    checks : bool, optional
        If set to False, no checks will be made for matrix
        symmetry nor zero diagonals. This is useful if it is known that
        ``X - X.T1`` is small and ``diag(X)`` is close to zero.
        These values are ignored any way so they do not disrupt the
        squareform transformation.

    Returns
    -------
    Y : ndarray
        If a condensed distance matrix is passed, a redundant one is
        returned, or if a redundant one is passed, a condensed distance
        matrix is returned.

    Notes
    -----
    1. ``v = squareform(X)``

       Given a square n-by-n symmetric distance matrix ``X``,
       ``v = squareform(X)`` returns a ``n * (n-1) / 2``
       (i.e. binomial coefficient n choose 2) sized vector `v`
       where :math:`v[{n \\choose 2} - {n-i \\choose 2} + (j-i-1)]`
       is the distance between distinct points ``i`` and ``j``.
       If ``X`` is non-square or asymmetric, an error is raised.

    2. ``X = squareform(v)``

       Given a ``n * (n-1) / 2`` sized vector ``v``
       for some integer ``n >= 1`` encoding distances as described,
       ``X = squareform(v)`` returns a n-by-n distance matrix ``X``.
       The ``X[i, j]`` and ``X[j, i]`` values are set to
       :math:`v[{n \\choose 2} - {n-i \\choose 2} + (j-i-1)]`
       and all diagonal elements are zero.

    In SciPy 0.19.0, ``squareform`` stopped casting all input types to
    float64, and started returning arrays of the same dtype as the input.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import pdist, squareform

    ``x`` is an array of five points in three-dimensional space.

    >>> x = np.array([[2, 0, 2], [2, 2, 3], [-2, 4, 5], [0, 1, 9], [2, 2, 4]])

    ``pdist(x)`` computes the Euclidean distances between each pair of
    points in ``x``.  The distances are returned in a one-dimensional
    array with length ``5*(5 - 1)/2 = 10``.

    >>> distvec = pdist(x)
    >>> distvec
    array([2.23606798, 6.40312424, 7.34846923, 2.82842712, 4.89897949,
           6.40312424, 1.        , 5.38516481, 4.58257569, 5.47722558])

    ``squareform(distvec)`` returns the 5x5 distance matrix.

    >>> m = squareform(distvec)
    >>> m
    """
    array([[0.        , 2.23606798, 6.40312424, 7.34846923, 2.82842712],
           [2.23606798, 0.        , 4.89897949, 6.40312424, 1.        ],
           [6.40312424, 4.89897949, 0.        , 5.38516481, 4.58257569],
           [7.34846923, 6.40312424, 5.38516481, 0.        , 5.47722558],
           [2.82842712, 1.        , 4.58257569, 5.47722558, 0.        ]])

    When given a square distance matrix ``m``, ``squareform(m)`` returns
    the one-dimensional condensed distance vector associated with the
    matrix.  In this case, we recover ``distvec``.

    >>> squareform(m)
    array([2.23606798, 6.40312424, 7.34846923, 2.82842712, 4.89897949,
           6.40312424, 1.        , 5.38516481, 4.58257569, 5.47722558])
    """
    # 将输入数组 X 转换为一个连续的数组
    X = np.ascontiguousarray(X)

    # 获取数组 X 的形状
    s = X.shape

    # 如果指定了强制类型为 'tomatrix'
    if force.lower() == 'tomatrix':
        # 如果 X 不是一维数组，则抛出 ValueError 异常
        if len(s) != 1:
            raise ValueError("Forcing 'tomatrix' but input X is not a "
                             "distance vector.")
    # 如果指定了强制类型为 'tovector'
    elif force.lower() == 'tovector':
        # 如果 X 不是二维数组，则抛出 ValueError 异常
        if len(s) != 2:
            raise ValueError("Forcing 'tovector' but input X is not a "
                             "distance matrix.")

    # 如果 X 是一维数组
    if len(s) == 1:
        # 如果数组 X 中没有元素
        if s[0] == 0:
            # 返回一个包含单个零的数组
            return np.zeros((1, 1), dtype=X.dtype)

        # 计算数组元素个数的平方根乘以 2 的向上取整值
        d = int(np.ceil(np.sqrt(s[0] * 2)))

        # 检查数组 X 的维度是否符合二项式系数的要求
        if d * (d - 1) != s[0] * 2:
            raise ValueError('Incompatible vector size. It must be a binomial '
                             'coefficient n choose 2 for some integer n >= 2.')

        # 为距离矩阵分配内存空间
        M = np.zeros((d, d), dtype=X.dtype)

        # 如果数组 X 具有基类存在，则复制数组 X
        X = _copy_array_if_base_present(X)

        # 使用 C 代码填充距离矩阵的值
        _distance_wrap.to_squareform_from_vector_wrap(M, X)

        # 返回填充后的距离矩阵
        return M
    # 如果 X 是二维数组
    elif len(s) == 2:
        # 如果 X 不是方阵，则抛出 ValueError 异常
        if s[0] != s[1]:
            raise ValueError('The matrix argument must be square.')
        
        # 如果开启了检查选项，则验证 X 是否为有效的距离矩阵
        if checks:
            is_valid_dm(X, throw=True, name='X')

        # 获取矩阵 X 的维度
        d = s[0]

        # 如果矩阵的维度小于等于 1，则返回一个空数组
        if d <= 1:
            return np.array([], dtype=X.dtype)

        # 创建一个用于存储压缩距离矩阵的向量
        v = np.zeros((d * (d - 1)) // 2, dtype=X.dtype)

        # 如果数组 X 具有基类存在，则复制数组 X
        X = _copy_array_if_base_present(X)

        # 将方阵转换为压缩形式的向量
        _distance_wrap.to_vector_from_squareform_wrap(X, v)
        
        # 返回转换后的向量
        return v
    else:
        # 如果参数 s 是一个超过两维的数组，则抛出数值错误异常
        raise ValueError(('The first argument must be one or two dimensional '
                          'array. A %d-dimensional array is not '
                          'permitted') % len(s))
# 将输入参数 D 转换为 C 风格的 NumPy 数组，确保它是二维数组
D = np.asarray(D, order='c')
# 初始化变量 valid，假设输入的距离矩阵是有效的
valid = True
    # 尝试获取距离矩阵 D 的形状
    try:
        s = D.shape
        # 如果距离矩阵 D 的维度不是 2
        if len(D.shape) != 2:
            # 如果提供了名称 name，则抛出维度错误异常
            if name:
                raise ValueError(('Distance matrix \'%s\' must have shape=2 '
                                  '(i.e. be two-dimensional).') % name)
            else:
                # 否则抛出维度错误异常
                raise ValueError('Distance matrix must have shape=2 (i.e. '
                                 'be two-dimensional).')
        
        # 如果公差 tol 为 0.0
        if tol == 0.0:
            # 检查距离矩阵 D 是否是对称的
            if not (D == D.T).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' must be '
                                     'symmetric.') % name)
                else:
                    raise ValueError('Distance matrix must be symmetric.')
            
            # 检查距离矩阵 D 对角线元素是否为零
            if not (D[range(0, s[0]), range(0, s[0])] == 0).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' diagonal must '
                                      'be zero.') % name)
                else:
                    raise ValueError('Distance matrix diagonal must be zero.')
        else:
            # 如果公差 tol 不为 0.0，则检查距离矩阵 D 是否在公差范围内对称
            if not (D - D.T <= tol).all():
                if name:
                    raise ValueError(f'Distance matrix \'{name}\' must be '
                                     f'symmetric within tolerance {tol:5.5f}.')
                else:
                    raise ValueError('Distance matrix must be symmetric within '
                                     'tolerance %5.5f.' % tol)
            
            # 检查距离矩阵 D 对角线元素是否在公差范围内接近零
            if not (D[range(0, s[0]), range(0, s[0])] <= tol).all():
                if name:
                    raise ValueError(f'Distance matrix \'{name}\' diagonal must be '
                                     f'close to zero within tolerance {tol:5.5f}.')
                else:
                    raise ValueError(('Distance matrix \'{}\' diagonal must be close '
                                      'to zero within tolerance {:5.5f}.').format(*tol))
    except Exception as e:
        # 如果发生异常，并且 throw 标志为真，则重新抛出异常
        if throw:
            raise
        # 如果 warning 标志为真，则发出警告
        if warning:
            warnings.warn(str(e), stacklevel=2)
        # 设置 valid 变量为假
        valid = False
    
    # 返回 valid 变量
    return valid
# 返回 True 如果输入的数组是一个有效的压缩距离矩阵。
# 压缩距离矩阵必须是一维的 numpy 数组。
# 它们的长度必须是某个正整数 n 的二项式系数，即 :math:`{n \\choose 2}`。

def is_valid_y(y, warning=False, throw=False, name=None):
    """
    Return True if the input array is a valid condensed distance matrix.

    Condensed distance matrices must be 1-dimensional numpy arrays.
    Their length must be a binomial coefficient :math:`{n \\choose 2}`
    for some positive integer n.

    Parameters
    ----------
    y : array_like
        The condensed distance matrix.
    warning : bool, optional
        Invokes a warning if the variable passed is not a valid
        condensed distance matrix. The warning message explains why
        the distance matrix is not valid.  `name` is used when
        referencing the offending variable.
    throw : bool, optional
        Throws an exception if the variable passed is not a valid
        condensed distance matrix.
    name : bool, optional
        Used when referencing the offending variable in the
        warning or exception message.

    Returns
    -------
    bool
        True if the input array is a valid condensed distance matrix,
        False otherwise.

    Examples
    --------
    >>> from scipy.spatial.distance import is_valid_y

    This vector is a valid condensed distance matrix.  The length is 6,
    which corresponds to ``n = 4``, since ``4*(4 - 1)/2`` is 6.

    >>> v = [1.0, 1.2, 1.0, 0.5, 1.3, 0.9]
    >>> is_valid_y(v)
    True

    An input vector with length, say, 7, is not a valid condensed distance
    matrix.

    >>> is_valid_y([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    False

    """
    
    # 将输入转换为 numpy 数组，以确保其内存布局为 C 风格
    y = np.asarray(y, order='c')
    valid = True
    try:
        # 检查数组的维度是否为 1
        if len(y.shape) != 1:
            if name:
                raise ValueError(('Condensed distance matrix \'%s\' must '
                                  'have shape=1 (i.e. be one-dimensional).')
                                 % name)
            else:
                raise ValueError('Condensed distance matrix must have shape=1 '
                                 '(i.e. be one-dimensional).')
        
        # 计算矩阵的长度 n
        n = y.shape[0]
        # 计算 n 对应的 k
        d = int(np.ceil(np.sqrt(n * 2)))
        # 检查是否满足二项式系数的条件
        if (d * (d - 1) / 2) != n:
            if name:
                raise ValueError(('Length n of condensed distance matrix '
                                  '\'%s\' must be a binomial coefficient, i.e.'
                                  'there must be a k such that '
                                  '(k \\choose 2)=n)!') % name)
            else:
                raise ValueError('Length n of condensed distance matrix must '
                                 'be a binomial coefficient, i.e. there must '
                                 'be a k such that (k \\choose 2)=n)!')
    
    except Exception as e:
        # 处理异常情况
        if throw:
            raise
        if warning:
            # 触发警告，解释为何距离矩阵无效
            warnings.warn(str(e), stacklevel=2)
        valid = False
    
    # 返回验证结果
    return valid


# 返回与方形冗余距离矩阵对应的原始观测数量。
def num_obs_dm(d):
    """
    Return the number of original observations that correspond to a
    square, redundant distance matrix.

    Parameters
    ----------
    
    d : array_like
        The target distance matrix.
    返回一个距离矩阵的数组，通常是一个二维数组，表示目标距离矩阵。

    Returns
    -------
    num_obs_dm : int
        The number of observations in the redundant distance matrix.
    返回一个整数，表示冗余距离矩阵中的观测数量。

    Examples
    --------
    Find the number of original observations corresponding
    to a square redundant distance matrix d.
    找到对应于一个方形冗余距离矩阵 d 的原始观测数量。

    >>> from scipy.spatial.distance import num_obs_dm
    >>> d = [[0, 100, 200], [100, 0, 150], [200, 150, 0]]
    >>> num_obs_dm(d)
    3
    """
    将输入的距离矩阵 d 转换为一个 NumPy 数组，强制使用 C 顺序存储。
    d = np.asarray(d, order='c')
    调用 is_valid_dm 函数，检查距离矩阵 d 是否有效，tol 参数为无穷大，如果不合法则抛出异常，名称为 'd'。
    is_valid_dm(d, tol=np.inf, throw=True, name='d')
    返回距离矩阵 d 的第一维度大小，即观测数量。
    return d.shape[0]
# 计算给定压缩距离矩阵Y对应的原始观测数量
def num_obs_y(Y):
    Y = np.asarray(Y, order='c')  # 将输入Y转换为C顺序的NumPy数组
    is_valid_y(Y, throw=True, name='Y')  # 检查Y是否是有效的压缩距离矩阵
    k = Y.shape[0]  # 获取Y的长度（即观测数量）
    if k == 0:
        raise ValueError("The number of observations cannot be determined on "
                         "an empty distance matrix.")  # 如果Y为空，抛出值错误异常
    d = int(np.ceil(np.sqrt(k * 2)))  # 计算观测数量d，满足k=(n choose 2)，其中n >= 2
    if (d * (d - 1) / 2) != k:
        raise ValueError("Invalid condensed distance matrix passed. Must be "
                         "some k where k=(n choose 2) for some n >= 2.")  # 如果Y不是有效的压缩距离矩阵，抛出值错误异常
    return d  # 返回观测数量d


# 准备输出参数的辅助函数
def _prepare_out_argument(out, dtype, expected_shape):
    if out is None:
        return np.empty(expected_shape, dtype=dtype)  # 如果输出参数out为空，创建一个指定形状和数据类型的空NumPy数组

    if out.shape != expected_shape:
        raise ValueError("Output array has incorrect shape.")  # 如果输出参数out的形状与期望形状不符，抛出值错误异常
    if not out.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous.")  # 如果输出参数out不是C连续存储，抛出值错误异常
    if out.dtype != np.float64:
        raise ValueError("Output array must be double type.")  # 如果输出参数out的数据类型不是双精度浮点型，抛出值错误异常
    return out  # 返回准备好的输出参数out


# 计算距离度量的辅助函数（用于一对一的距离计算）
def _pdist_callable(X, *, out, metric, **kwargs):
    n = X.shape[0]  # 获取输入数组XA的行数n
    out_size = (n * (n - 1)) // 2  # 计算输出数组的大小，以存储所有可能的一对距离
    dm = _prepare_out_argument(out, np.float64, (out_size,))  # 准备输出数组dm，确保其形状、数据类型和连续性
    k = 0
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            dm[k] = metric(X[i], X[j], **kwargs)  # 使用指定的度量函数计算XA中每对观测之间的距离，并存储在dm中
            k += 1
    return dm  # 返回计算得到的距离数组dm


# 计算距离度量的辅助函数（用于两组观测之间的距离计算）
def _cdist_callable(XA, XB, *, out, metric, **kwargs):
    mA = XA.shape[0]  # 获取输入数组XA的行数mA
    mB = XB.shape[0]  # 获取输入数组XB的行数mB
    dm = _prepare_out_argument(out, np.float64, (mA, mB))  # 准备输出数组dm，确保其形状、数据类型和连续性
    for i in range(mA):
        for j in range(mB):
            dm[i, j] = metric(XA[i], XB[j], **kwargs)  # 使用指定的度量函数计算XA中的每个观测与XB中的每个观测之间的距离，并存储在dm中
    return dm  # 返回计算得到的距离矩阵dm


# 计算两组输入之间的距离（用于一对一和一对多的距离计算）
def cdist(XA, XB, metric='euclidean', *, out=None, **kwargs):
    """
    Compute distance between each pair of the two collections of inputs.

    See Notes for common calling conventions.

    Parameters
    ----------
    XA : array_like
        An :math:`m_A` by :math:`n` array of :math:`m_A`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    XB : array_like
        An :math:`m_B` by :math:`n` array of :math:`m_B`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    """
    # metric 参数可以接受字符串或者可调用对象，用于指定距离度量方法，可选的字符串包括多种距离度量方法的名称
    metric : str or callable, optional
        # 距离度量方法的文档字符串，列出可用的距离函数名称，例如 'braycurtis', 'canberra', 'chebyshev' 等
        The distance metric to use. If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule'.

    # **kwargs 是传递给 metric 参数的额外参数字典，具体参数应参考各个距离度量方法的文档说明
    **kwargs : dict, optional
        # 对 metric 参数的额外参数进行描述，具体可用的参数列表需要参考每种距离度量方法的文档
        Extra arguments to `metric`: refer to each metric documentation for a
        list of all possible arguments.

        Some possible arguments:

        # p 参数适用于 Minkowski 距离，可以指定 p-范数的大小，包括加权和未加权情况，默认为 2
        p : scalar
        The p-norm to apply for Minkowski, weighted and unweighted.
        Default: 2.

        # w 参数是一种权重向量，适用于支持权重的距离度量方法，例如 Minkowski 距离
        w : array_like
        The weight vector for metrics that support weights (e.g., Minkowski).

        # V 参数是标准化欧几里得距离的方差向量，如果未指定，将自动计算
        V : array_like
        The variance vector for standardized Euclidean.
        Default: var(vstack([XA, XB]), axis=0, ddof=1)

        # VI 参数是马氏距离的协方差矩阵的逆矩阵，如果未指定，将自动计算
        VI : array_like
        The inverse of the covariance matrix for Mahalanobis.
        Default: inv(cov(vstack([XA, XB].T))).T

        # out 参数是一个 ndarray，如果不为 None，则距离矩阵 Y 将存储在这个数组中
        out : ndarray
        The output array
        If not None, the distance matrix Y is stored in this array.

    # 返回一个 ndarray Y，表示由 :math:`m_A` 行 :math:`m_B` 列的距离矩阵
    Returns
    -------
    Y : ndarray
        A :math:`m_A` by :math:`m_B` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.

    # 如果 XA 和 XB 的列数不同，则引发 ValueError 异常
    Raises
    ------
    ValueError
        An exception is thrown if `XA` and `XB` do not have
        the same number of columns.

    # 下面是一些常见的调用约定的说明
    Notes
    -----
    The following are common calling conventions:

    # 使用 'euclidean' 方法计算欧氏距离
    1. ``Y = cdist(XA, XB, 'euclidean')``

       Computes the distance between :math:`m` points using
       Euclidean distance (2-norm) as the distance metric between the
       points. The points are arranged as :math:`m`
       :math:`n`-dimensional row vectors in the matrix X.

    # 使用 'minkowski' 方法计算 Minkowski 距离，指定 p=2
    2. ``Y = cdist(XA, XB, 'minkowski', p=2.)``

       Computes the distances using the Minkowski distance
       :math:`\\|u-v\\|_p` (:math:`p`-norm) where :math:`p > 0` (note
       that this is only a quasi-metric if :math:`0 < p < 1`).

    # 使用 'cityblock' 方法计算曼哈顿距离
    3. ``Y = cdist(XA, XB, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    # 使用 'seuclidean' 方法计算标准化欧几里得距离
    4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}.

       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points. If not passed, it is
       automatically computed.

    # 使用 'sqeuclidean' 方法计算平方欧氏距离
    5. ``Y = cdist(XA, XB, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`\\|u-v\\|_2^2` between
       the vectors.
    6. ``Y = cdist(XA, XB, 'cosine')``
    
       计算向量 u 和 v 之间的余弦距离，
       
       .. math::
       
          1 - \\frac{u \\cdot v}
                   {{\\|u\\|}_2 {\\|v\\|}_2}
       
       其中 :math:`\\|*\\|_2` 是参数 ``*`` 的2-范数，而 :math:`u \\cdot v` 是向量 :math:`u` 和 :math:`v` 的点积。

    7. ``Y = cdist(XA, XB, 'correlation')``
    
       计算向量 u 和 v 之间的相关距离。具体计算公式为
       
       .. math::
       
          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
       
       其中 :math:`\\bar{v}` 是向量 v 的元素均值，而 :math:`x \\cdot y` 是向量 :math:`x` 和 :math:`y` 的点积。

    8. ``Y = cdist(XA, XB, 'hamming')``
    
       计算归一化的汉明距离，即两个 n-向量 ``u`` 和 ``v`` 中不同元素的比例。为了节省内存，矩阵 ``X`` 可以是布尔类型。

    9. ``Y = cdist(XA, XB, 'jaccard')``
    
       计算点之间的Jaccard距离。给定两个向量 ``u`` 和 ``v``，Jaccard距离是在至少一个元素非零的情况下，元素 ``u[i]`` 和 ``v[i]`` 不同的比例。

    10. ``Y = cdist(XA, XB, 'jensenshannon')``
    
        计算两个概率数组之间的Jensen-Shannon距离。给定两个概率向量 :math:`p` 和 :math:`q`，Jensen-Shannon距离为
        
        .. math::
        
           \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
        
        其中 :math:`m` 是 :math:`p` 和 :math:`q` 的逐点均值，而 :math:`D` 是Kullback-Leibler散度。

    11. ``Y = cdist(XA, XB, 'chebyshev')``
    
        计算点之间的切比雪夫距离。两个 n-向量 ``u`` 和 ``v`` 之间的切比雪夫距离是它们各自元素的最大L1距离。
        
        .. math::
        
           d(u,v) = \\max_i {|u_i-v_i|}.

    12. ``Y = cdist(XA, XB, 'canberra')``
    
        计算点之间的坎贝拉距离。两个点 ``u`` 和 ``v`` 之间的坎贝拉距离是
        
        .. math::
        
          d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                               {|u_i|+|v_i|}.

    13. ``Y = cdist(XA, XB, 'braycurtis')``
    
        计算点之间的Bray-Curtis距离。两个点 ``u`` 和 ``v`` 之间的Bray-Curtis距离是
        
        .. math::
        
             d(u,v) = \\frac{\\sum_i (|u_i-v_i|)}
                           {\\sum_i (|u_i+v_i|)}
    14. ```
       Y = cdist(XA, XB, 'mahalanobis', VI=None)
       
           计算两组点之间的马氏距离。马氏距离定义为：sqrt((u-v)(1/V)(u-v)^T)，其中(1/V)为协方差的逆矩阵。
           如果VI不为None，则使用VI作为协方差的逆矩阵。
       ```

    15. ```
       Y = cdist(XA, XB, 'yule')
       
           计算布尔向量之间的Yule距离。详见'yule'函数的文档。
       ```

    16. ```
       Y = cdist(XA, XB, 'matching')
       
           'matching'的同义词为'hamming'，计算两个向量之间的汉明距离。
       ```

    17. ```
       Y = cdist(XA, XB, 'dice')
       
           计算布尔向量之间的Dice距离。详见'dice'函数的文档。
       ```

    18. ```
       Y = cdist(XA, XB, 'kulczynski1')
       
           计算布尔向量之间的Kulczynski距离。详见'kulczynski1'函数的文档。
       ```

    19. ```
       Y = cdist(XA, XB, 'rogerstanimoto')
       
           计算布尔向量之间的Rogers-Tanimoto距离。详见'rogerstanimoto'函数的文档。
       ```

    20. ```
       Y = cdist(XA, XB, 'russellrao')
       
           计算布尔向量之间的Russell-Rao距离。详见'russellrao'函数的文档。
       ```

    21. ```
       Y = cdist(XA, XB, 'sokalmichener')
       
           计算布尔向量之间的Sokal-Michener距离。详见'sokalmichener'函数的文档。
       ```

    22. ```
       Y = cdist(XA, XB, 'sokalsneath')
       
           计算向量之间的Sokal-Sneath距离。详见'sokalsneath'函数的文档。
       ```

    23. ```
       Y = cdist(XA, XB, f)
       
           使用用户提供的二元函数f计算X中所有向量之间的距离。例如，可以计算向量之间的欧氏距离如下：
           dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))
           
           注意不要将库中已定义的距离函数直接传递给f，如：
           dm = cdist(XA, XB, sokalsneath)
           这样做效率低下，应该使用优化的C版本，使用以下语法调用：
           dm = cdist(XA, XB, 'sokalsneath')
       ```
    """
    Find the Manhattan distance from a 3-D point to the corners of the unit
    cube:
    
    >>> a = np.array([[0, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 1, 0],
    ...               [0, 1, 1],
    ...               [1, 0, 0],
    ...               [1, 0, 1],
    ...               [1, 1, 0],
    ...               [1, 1, 1]])
    >>> b = np.array([[ 0.1,  0.2,  0.4]])
    >>> distance.cdist(a, b, 'cityblock')
    array([[ 0.7],
           [ 0.9],
           [ 1.3],
           [ 1.5],
           [ 1.5],
           [ 1.7],
           [ 2.1],
           [ 2.3]])
    
    """
    # You can also call this as:
    #     Y = cdist(XA, XB, 'test_abc')
    # where 'abc' is the metric being tested.  This computes the distance
    # between all pairs of vectors in XA and XB using the distance metric 'abc'
    # but with a more succinct, verifiable, but less efficient implementation.
    
    # Convert XA and XB to numpy arrays
    XA = np.asarray(XA)
    XB = np.asarray(XB)
    
    # Get the shapes of XA and XB
    s = XA.shape
    sB = XB.shape
    
    # Check if XA is a 2-dimensional array
    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    
    # Check if XB is a 2-dimensional array
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    
    # Check if XA and XB have the same number of columns
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')
    
    # Get the number of rows in XA and XB
    mA = s[0]
    mB = sB[0]
    n = s[1]
    
    # Check if metric is callable (a function)
    if callable(metric):
        # Get the name of the metric function
        mstr = getattr(metric, '__name__', 'Unknown')
        # Look up metric info in _METRIC_ALIAS dictionary
        metric_info = _METRIC_ALIAS.get(mstr, None)
        if metric_info is not None:
            # Validate and process inputs using _validate_cdist_input
            XA, XB, typ, kwargs = _validate_cdist_input(
                XA, XB, mA, mB, n, metric_info, **kwargs)
            # Call _cdist_callable function with metric function
            return _cdist_callable(XA, XB, metric=metric, out=out, **kwargs)
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    
    # Check if metric is a string
    elif isinstance(metric, str):
        # Convert metric string to lowercase
        mstr = metric.lower()
        # Look up metric info in _METRIC_ALIAS dictionary
        metric_info = _METRIC_ALIAS.get(mstr, None)
        if metric_info is not None:
            # Call cdist function associated with the metric
            cdist_fn = metric_info.cdist_func
            return cdist_fn(XA, XB, out=out, **kwargs)
        elif mstr.startswith("test_"):
            # Look up test metric info in _TEST_METRICS dictionary
            metric_info = _TEST_METRICS.get(mstr, None)
            if metric_info is None:
                raise ValueError(f'Unknown "Test" Distance Metric: {mstr[5:]}')
            # Validate and process inputs using _validate_cdist_input
            XA, XB, typ, kwargs = _validate_cdist_input(
                XA, XB, mA, mB, n, metric_info, **kwargs)
            # Call _cdist_callable function with test metric function
            return _cdist_callable(
                XA, XB, metric=metric_info.dist_func, out=out, **kwargs)
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    
    # If metric is neither callable nor a string, raise TypeError
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
```