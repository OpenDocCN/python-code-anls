# `D:\src\scipysrc\scipy\scipy\cluster\vq.py`

```
"""
K-means clustering and vector quantization (:mod:`scipy.cluster.vq`)
====================================================================

Provides routines for k-means clustering, generating code books
from k-means models and quantizing vectors by comparing them with
centroids in a code book.

.. autosummary::
   :toctree: generated/

   whiten -- Normalize a group of observations so each feature has unit variance
   vq -- Calculate code book membership of a set of observation vectors
   kmeans -- Perform k-means on a set of observation vectors forming k clusters
   kmeans2 -- A different implementation of k-means with more methods
           -- for initializing centroids

Background information
----------------------
The k-means algorithm takes as input the number of clusters to
generate, k, and a set of observation vectors to cluster. It
returns a set of centroids, one for each of the k clusters. An
observation vector is classified with the cluster number or
centroid index of the centroid closest to it.

A vector v belongs to cluster i if it is closer to centroid i than
any other centroid. If v belongs to i, we say centroid i is the
dominating centroid of v. The k-means algorithm tries to
minimize distortion, which is defined as the sum of the squared distances
between each observation vector and its dominating centroid.
The minimization is achieved by iteratively reclassifying
the observations into clusters and recalculating the centroids until
a configuration is reached in which the centroids are stable. One can
also define a maximum number of iterations.

Since vector quantization is a natural application for k-means,
information theory terminology is often used. The centroid index
or cluster index is also referred to as a "code" and the table
mapping codes to centroids and, vice versa, is often referred to as a
"code book". The result of k-means, a set of centroids, can be
used to quantize vectors. Quantization aims to find an encoding of
vectors that reduces the expected distortion.

All routines expect obs to be an M by N array, where the rows are
the observation vectors. The codebook is a k by N array, where the
ith row is the centroid of code word i. The observation vectors
and centroids have the same feature dimension.

As an example, suppose we wish to compress a 24-bit color image
(each pixel is represented by one byte for red, one for blue, and
one for green) before sending it over the web. By using a smaller
8-bit encoding, we can reduce the amount of data by two
thirds. Ideally, the colors for each of the 256 possible 8-bit
encoding values should be chosen to minimize distortion of the
color. Running k-means with k=256 generates a code book of 256
codes, which fills up all possible 8-bit sequences. Instead of
sending a 3-byte value for each pixel, the 8-bit centroid index
(or code word) of the dominating centroid is transmitted. The code
book is also sent over the wire so each 8-bit code can be
"""

# 导入必要的模块
from scipy.cluster.vq import whiten, vq, kmeans, kmeans2

# whiten函数：对观测数据进行归一化，使每个特征具有单位方差
# vq函数：计算一组观测向量的代码簿成员资格
# kmeans函数：对一组观测向量执行k均值聚类，形成k个簇
# kmeans2函数：另一种实现k均值的方法，包含更多用于初始化质心的方法
# 导入警告模块，用于在运行时可能出现的问题时发出警告
import warnings
# 导入 NumPy 库，用于处理数组和数学运算
import numpy as np
# 导入 collections 模块中的 deque 类，用于高效实现双向队列
from collections import deque
# 导入 scipy._lib._array_api 模块中的相关函数
from scipy._lib._array_api import (
    _asarray, array_namespace, size, atleast_nd, copy, cov
)
# 导入 scipy._lib._util 模块中的随机数生成和状态检查函数
from scipy._lib._util import check_random_state, rng_integers
# 导入 scipy.spatial.distance 模块中的距离计算函数
from scipy.spatial.distance import cdist

# 导入本地的 _vq 模块
from . import _vq

# 设置文档字符串的格式为 reStructuredText
__docformat__ = 'restructuredtext'

# 定义公开的函数列表
__all__ = ['whiten', 'vq', 'kmeans', 'kmeans2']


# 定义一个异常类 ClusterError，用于处理聚类过程中的错误
class ClusterError(Exception):
    pass


# 定义函数 whiten，用于对观测数据进行白化处理
def whiten(obs, check_finite=True):
    """
    Normalize a group of observations on a per feature basis.

    Before running k-means, it is beneficial to rescale each feature
    dimension of the observation set by its standard deviation (i.e. "whiten"
    it - as in "white noise" where each frequency has equal power).
    Each feature is divided by its standard deviation across all observations
    to give it unit variance.

    Parameters
    ----------
    obs : ndarray
        Each row of the array is an observation.  The
        columns are the features seen during each observation.

        >>> #         f0    f1    f2
        >>> obs = [[  1.,   1.,   1.],  #o0
        ...        [  2.,   2.,   2.],  #o1
        ...        [  3.,   3.,   3.],  #o2
        ...        [  4.,   4.,   4.]]  #o3

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    result : ndarray
        Contains the values in `obs` scaled by the standard deviation
        of each column.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster.vq import whiten
    >>> features  = np.array([[1.9, 2.3, 1.7],
    ...                       [1.5, 2.5, 2.2],
    ...                       [0.8, 0.6, 1.7,]])
    >>> whiten(features)
    array([[ 4.17944278,  2.69811351,  7.21248917],
           [ 3.29956009,  2.93273208,  9.33380951],
           [ 1.75976538,  0.7038557 ,  7.21248917]])

    """
    # 获取数组的命名空间（可能是 NumPy 或其他实现），并将 obs 转换为数组
    xp = array_namespace(obs)
    obs = _asarray(obs, check_finite=check_finite, xp=xp)
    # 计算每列的标准差
    std_dev = xp.std(obs, axis=0)
    # 创建一个标志位，标记标准差为零的列
    zero_std_mask = std_dev == 0
    # 如果存在标准差为零的列，将其标准差设为1，并发出警告
    if xp.any(zero_std_mask):
        std_dev[zero_std_mask] = 1.0
        warnings.warn("Some columns have standard deviation zero. "
                      "The values of these columns will not change.",
                      RuntimeWarning, stacklevel=2)
    # 返回白化后的观测数据
    return obs / std_dev


# 定义函数 vq，用于将观测数据分配到给定的码书中
def vq(obs, code_book, check_finite=True):
    """
    Assign codes from a code book to observations.

    Assigns a code from a code book to each observation. Each
    """
    xp = array_namespace(obs, code_book)
    obs = _asarray(obs, xp=xp, check_finite=check_finite)
    code_book = _asarray(code_book, xp=xp, check_finite=check_finite)
    ct = xp.result_type(obs, code_book)

    c_obs = xp.astype(obs, ct, copy=False)
    c_code_book = xp.astype(code_book, ct, copy=False)

    if xp.isdtype(ct, kind='real floating'):
        c_obs = np.asarray(c_obs)
        c_code_book = np.asarray(c_code_book)
        result = _vq.vq(c_obs, c_code_book)
        return xp.asarray(result[0]), xp.asarray(result[1])
    return py_vq(obs, code_book, check_finite=False)



    xp = array_namespace(obs, code_book)
    # 使用 array_namespace 函数创建一个与输入数据相关的数据结构，以优化数组操作
    obs = _asarray(obs, xp=xp, check_finite=check_finite)
    # 将输入数据 obs 转换为数组，使用 xp 指定的数据结构
    code_book = _asarray(code_book, xp=xp, check_finite=check_finite)
    # 将 code_book 转换为数组，使用 xp 指定的数据结构
    ct = xp.result_type(obs, code_book)
    # 根据 obs 和 code_book 的类型，确定结果数组的类型

    c_obs = xp.astype(obs, ct, copy=False)
    # 将 obs 转换为指定类型 ct 的数组，使用 xp 指定的数据结构，无需复制
    c_code_book = xp.astype(code_book, ct, copy=False)
    # 将 code_book 转换为指定类型 ct 的数组，使用 xp 指定的数据结构，无需复制

    if xp.isdtype(ct, kind='real floating'):
        # 如果结果类型 ct 是实数浮点类型
        c_obs = np.asarray(c_obs)
        # 将 c_obs 转换为 NumPy 数组
        c_code_book = np.asarray(c_code_book)
        # 将 c_code_book 转换为 NumPy 数组
        result = _vq.vq(c_obs, c_code_book)
        # 使用 c_obs 和 c_code_book 进行向量量化，返回结果
        return xp.asarray(result[0]), xp.asarray(result[1])
        # 将结果转换为 xp 指定的数据结构并返回

    return py_vq(obs, code_book, check_finite=False)
    # 如果结果类型不是实数浮点类型，则调用 py_vq 函数进行向量量化，返回结果
# 定义一个名为 py_vq 的函数，实现了 vq 算法的 Python 版本
def py_vq(obs, code_book, check_finite=True):
    """ Python version of vq algorithm.

    The algorithm computes the Euclidean distance between each
    observation and every frame in the code_book.

    Parameters
    ----------
    obs : ndarray
        Expects a rank 2 array. Each row is one observation.
    code_book : ndarray
        Code book to use. Same format than obs. Should have same number of
        features (e.g., columns) than obs.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    code : ndarray
        code[i] gives the label of the ith obversation; its code is
        code_book[code[i]].
    mind_dist : ndarray
        min_dist[i] gives the distance between the ith observation and its
        corresponding code.

    Notes
    -----
    This function is slower than the C version but works for
    all input types. If the inputs have the wrong types for the
    C versions of the function, this one is called as a last resort.

    It is about 20 times slower than the C version.

    """
    # 为了与底层库兼容，使用 array_namespace 函数重新定义 xp
    xp = array_namespace(obs, code_book)
    # 将观测数据 obs 转换为数组，并确保其类型和有限性
    obs = _asarray(obs, xp=xp, check_finite=check_finite)
    # 将 code_book 转换为数组，并确保其类型和有限性
    code_book = _asarray(code_book, xp=xp, check_finite=check_finite)

    # 检查观测数据和 code_book 的维度是否一致
    if obs.ndim != code_book.ndim:
        raise ValueError("Observation and code_book should have the same rank")

    # 如果观测数据和 code_book 是一维的，则将它们转换为二维
    if obs.ndim == 1:
        obs = obs[:, xp.newaxis]
        code_book = code_book[:, xp.newaxis]

    # 一旦 cdist 支持数组 API，此 xp.asarray 调用可以移除
    # 计算观测数据和 code_book 之间的距离矩阵
    dist = xp.asarray(cdist(obs, code_book))
    # 对每个观测数据，找到距离最近的 code_book 中的代码
    code = xp.argmin(dist, axis=1)
    # 计算每个观测数据与其对应代码之间的最小距离
    min_dist = xp.min(dist, axis=1)
    # 返回计算得到的代码和最小距离
    return code, min_dist


# 定义一个名为 _kmeans 的函数，实现了 k-means 算法的原始版本
def _kmeans(obs, guess, thresh=1e-5, xp=None):
    """ "raw" version of k-means.

    Returns
    -------
    code_book
        The lowest distortion codebook found.
    avg_dist
        The average distance a observation is from a code in the book.
        Lower means the code_book matches the data better.

    See Also
    --------
    kmeans : wrapper around k-means

    Examples
    --------
    Note: not whitened in this example.

    >>> import numpy as np
    >>> from scipy.cluster.vq import _kmeans
    >>> features  = np.array([[ 1.9,2.3],
    ...                       [ 1.5,2.5],
    ...                       [ 0.8,0.6],
    ...                       [ 0.4,1.8],
    ...                       [ 1.0,1.0]])
    >>> book = np.array((features[0],features[2]))
    >>> _kmeans(features,book)
    (array([[ 1.7       ,  2.4       ],
           [ 0.73333333,  1.13333333]]), 0.40563916697728591)

    """
    # 如果 xp 未指定，则默认使用 numpy 库
    xp = np if xp is None else xp
    # 初始化代码本 guess
    code_book = guess
    # 初始化差异为无穷大
    diff = xp.inf
    # 使用 deque 存储前两次的平均距离
    prev_avg_dists = deque([diff], maxlen=2)
    # 当误差（diff）大于阈值（thresh）时执行循环
    while diff > thresh:
        # 计算观测值（obs）与码本（code_book）之间的成员关系和距离
        obs_code, distort = vq(obs, code_book, check_finite=False)
        
        # 将当前平均失真度（distort）添加到列表中
        prev_avg_dists.append(xp.mean(distort, axis=-1))
        
        # 更新码本（code_book），以观测值的关联族群的质心作为新的码本
        obs = np.asarray(obs)  # 将观测值转换为NumPy数组
        obs_code = np.asarray(obs_code)  # 将成员关系转换为NumPy数组
        code_book, has_members = _vq.update_cluster_means(obs, obs_code,
                                                          code_book.shape[0])
        
        obs = xp.asarray(obs)  # 将观测值转换为指定的数据类型（可能是NumPy或CuPy数组）
        obs_code = xp.asarray(obs_code)  # 将成员关系转换为指定的数据类型
        code_book = xp.asarray(code_book)  # 将码本转换为指定的数据类型
        has_members = xp.asarray(has_members)  # 将成员标记转换为指定的数据类型
        code_book = code_book[has_members]  # 根据有成员标记来筛选码本中的条目
        
        # 计算前后两次平均失真度的绝对差异，作为下一次循环是否执行的依据
        diff = xp.abs(prev_avg_dists[0] - prev_avg_dists[1])

    # 返回最终的码本和最终的平均失真度
    return code_book, prev_avg_dists[1]
# 定义一个函数，执行 k-means 聚类算法，用于对一组观测向量进行聚类
def kmeans(obs, k_or_guess, iter=20, thresh=1e-5, check_finite=True,
           *, seed=None):
    """
    Performs k-means on a set of observation vectors forming k clusters.

    The k-means algorithm adjusts the classification of the observations
    into clusters and updates the cluster centroids until the position of
    the centroids is stable over successive iterations. In this
    implementation of the algorithm, the stability of the centroids is
    determined by comparing the absolute value of the change in the average
    Euclidean distance between the observations and their corresponding
    centroids against a threshold. This yields
    a code book mapping centroids to codes and vice versa.

    Parameters
    ----------
    obs : ndarray
       Each row of the M by N array is an observation vector. The
       columns are the features seen during each observation.
       The features must be whitened first with the `whiten` function.

    k_or_guess : int or ndarray
       The number of centroids to generate. A code is assigned to
       each centroid, which is also the row index of the centroid
       in the code_book matrix generated.

       The initial k centroids are chosen by randomly selecting
       observations from the observation matrix. Alternatively,
       passing a k by N array specifies the initial k centroids.

    iter : int, optional
       The number of times to run k-means, returning the codebook
       with the lowest distortion. This argument is ignored if
       initial centroids are specified with an array for the
       ``k_or_guess`` parameter. This parameter does not represent the
       number of iterations of the k-means algorithm.

    thresh : float, optional
       Terminates the k-means algorithm if the change in
       distortion since the last k-means iteration is less than
       or equal to threshold.

    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        Seed for initializing the pseudo-random number generator.
        If `seed` is None (or `numpy.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        The default is None.

    Returns
    -------
    """
    codebook : ndarray
       k个N维数组，表示k个质心。第i个质心codebook[i]代表第i个代码。生成的质心和代码表示所见的最低失真，并不一定是全局最小失真。
       注意质心的数量不一定与k_or_guess参数相同，因为分配给没有观察到的质心在迭代期间将被移除。

    distortion : float
       观察到的数据点与生成的质心之间的平均（非平方）欧氏距离。请注意，这与k均值算法上下文中失真的标准定义有所不同，后者是平方距离的总和。

    See Also
    --------
    kmeans2 : k均值聚类的另一种实现，具有更多生成初始质心方法，但不使用失真变化阈值作为停止准则。

    whiten : 在将观察矩阵传递给kmeans之前必须调用此函数。

    Notes
    -----
    要获得更多功能或最佳性能，您可以使用 `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_。
    `这里 <https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html#comparison-of-high-performance-implementations>`_
    是几种实现的基准结果比较。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster.vq import vq, kmeans, whiten
    >>> import matplotlib.pyplot as plt
    >>> features  = np.array([[ 1.9,2.3],
    ...                       [ 1.5,2.5],
    ...                       [ 0.8,0.6],
    ...                       [ 0.4,1.8],
    ...                       [ 0.1,0.1],
    ...                       [ 0.2,1.8],
    ...                       [ 2.0,0.5],
    ...                       [ 0.3,1.5],
    ...                       [ 1.0,1.0]])
    >>> whitened = whiten(features)
    >>> book = np.array((whitened[0],whitened[2]))
    >>> kmeans(whitened,book)
    (array([[ 2.3110306 ,  2.86287398],    # 随机生成
           [ 0.93218041,  1.24398691]]), 0.85684700941625547)

    >>> codes = 3
    >>> kmeans(whitened,codes)
    (array([[ 2.3110306 ,  2.86287398],    # 随机生成
           [ 1.32544402,  0.65607529],
           [ 0.40782893,  2.02786907]]), 0.5196582527686241)

    >>> # 创建50个数据点，分为两个簇a和b
    >>> pts = 50
    >>> rng = np.random.default_rng()
    >>> a = rng.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
    >>> b = rng.multivariate_normal([30, 10],
    ...                             [[10, 2], [2, 1]],
    ...                             size=pts)
    >>> features = np.concatenate((a, b))
    >>> # 白化数据
    >>> whitened = whiten(features)
    >>> # 在数据中找到2个簇
    >>> codebook, distortion = kmeans(whitened, 2)
    >>> # 绘制白化数据和聚类中心点（红色）
    >>> plt.scatter(whitened[:, 0], whitened[:, 1])
    >>> plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    >>> plt.show()

    """
    绘制散点图，显示经过白化处理后的数据集的第一列和第二列特征
    if isinstance(k_or_guess, int):
        xp = array_namespace(obs)
    else:
        xp = array_namespace(obs, k_or_guess)
    将观测数据和簇数量或猜测作为参数，转换为数组命名空间
    obs = _asarray(obs, xp=xp, check_finite=check_finite)
    guess = _asarray(k_or_guess, xp=xp, check_finite=check_finite)
    如果迭代次数小于1，引发值错误异常
    if iter < 1:
        raise ValueError("iter must be at least 1, got %s" % iter)

    # 确定传入的是数量（标量）还是初始猜测（数组）
    if size(guess) != 1:
        如果猜测的大小小于1，引发值错误异常
        if size(guess) < 1:
            raise ValueError("Asked for 0 clusters. Initial book was %s" %
                             guess)
        返回_kmeans函数的结果
        return _kmeans(obs, guess, thresh=thresh, xp=xp)

    # 如果k_or_guess是标量，验证它是否是整数
    k = int(guess)
    如果k不等于guess，引发值错误异常
    if k != guess:
        raise ValueError("If k_or_guess is a scalar, it must be an integer.")
    如果k小于1，引发值错误异常
    if k < 1:
        raise ValueError("Asked for %d clusters." % k)

    验证随机状态
    rng = check_random_state(seed)

    初始化最佳距离值为一个大值
    best_dist = xp.inf
    迭代次数范围内执行以下操作
    for i in range(iter):
        从观察值中随机选择初始代码本
        猜测=_kpoints（obs，k，rng，xp）
        book，dist=_kmeans（obs，guess，thresh=thresh，xp=xp）
        如果dist<best_dist
        选择best_book= book
 best_dist= vari
# 从数据中随机选择 k 个点作为初始质心，用于聚类算法中的初始化过程
def _kpoints(data, k, rng, xp):
    idx = rng.choice(data.shape[0], size=int(k), replace=False)
    # 将选取的索引转换为指定类型的数组，避免 numpy#25607 问题
    idx = xp.asarray(idx, dtype=xp.asarray([1]).dtype)
    return xp.take(data, idx, axis=0)


# 根据数据的统计特性，返回 k 个服从高斯分布的随机样本作为初始质心
def _krandinit(data, k, rng, xp):
    mu = xp.mean(data, axis=0)
    k = np.asarray(k)  # 将 k 转换为 NumPy 数组格式

    if data.ndim == 1:
        _cov = cov(data)  # 计算数据的协方差
        x = rng.standard_normal(size=k)  # 生成 k 个标准正态分布的随机数
        x = xp.asarray(x)  # 将生成的随机数转换为指定类型的数组
        x *= xp.sqrt(_cov)  # 对每个随机数乘以数据的标准差
    elif data.shape[1] > data.shape[0]:
        # 处理数据的协方差矩阵秩不足的情况
        _, s, vh = xp.linalg.svd(data - mu, full_matrices=False)
        x = rng.standard_normal(size=(k, s.size))  # 生成 k 行 s.size 列的标准正态分布随机数
        x = xp.asarray(x)  # 将生成的随机数转换为指定类型的数组
        sVh = s[:, None] * vh / xp.sqrt(data.shape[0] - xp.asarray(1.))
        x = x @ sVh  # 乘以特征值和右奇异矩阵的乘积
    else:
        _cov = atleast_nd(cov(data.T), ndim=2)  # 计算数据转置后的协方差矩阵

        # 生成 k 行，mu.size 列的标准正态分布随机数，然后乘以数据协方差的 Cholesky 分解的转置
        x = rng.standard_normal(size=(k, mu.size))
        x = xp.asarray(x)  # 将生成的随机数转换为指定类型的数组
        x = x @ xp.linalg.cholesky(_cov).T  # 乘以协方差矩阵的 Cholesky 分解的转置

    x += mu  # 将生成的随机数加上数据的均值
    return x


# 使用 kmeans++ 方法从数据中选择 k 个初始质心点
def _kpp(data, k, rng, xp):
    init = xp.zeros((k, data.shape[1]))  # 创建一个 k 行，data.shape[1] 列的零数组

    # 从数据中随机选择一个点作为第一个初始质心
    init[0] = data[rng.choice(data.shape[0])]

    # 使用 kmeans++ 算法选择剩余 k-1 个初始质心
    for j in range(1, k):
        D_sq = xp.array([min([xp.linalg.norm(init[i] - x)**2 for i in range(j)]) for x in data])
        probs = D_sq / D_sq.sum()
        cumulative_probs = probs.cumsum()
        r = rng.random()
        for i, cum_prob in enumerate(cumulative_probs):
            if r < cum_prob:
                init[j] = data[i]
                break

    return init
    # 引用的文献参考了k-means++算法的原始论文[1]，这篇论文探讨了谨慎选择初始点对k-means聚类的优势。
    """
    
    # 计算数据的维度数目
    ndim = len(data.shape)
    
    # 如果数据是一维的，则将其转换为二维，以便后续处理
    if ndim == 1:
        data = data[:, None]
    
    # 获取数据的维度
    dims = data.shape[1]
    
    # 初始化聚类中心矩阵，形状为 (k, dims)
    init = xp.empty((int(k), dims))
    
    # 循环k次，选择初始聚类中心
    for i in range(k):
        # 如果是第一个聚类中心，随机选择一个数据点作为初始点
        if i == 0:
            init[i, :] = data[rng_integers(rng, data.shape[0]), :]
    
        else:
            # 计算已选取的聚类中心与所有数据点之间的最小平方欧氏距离
            D2 = cdist(init[:i,:], data, metric='sqeuclidean').min(axis=0)
            # 计算每个数据点被选为下一个聚类中心的概率
            probs = D2 / D2.sum()
            # 计算累积概率
            cumprobs = probs.cumsum()
            # 生成一个均匀分布随机数r
            r = rng.uniform()
            cumprobs = np.asarray(cumprobs)
            # 根据随机数r选择下一个聚类中心
            init[i, :] = data[np.searchsorted(cumprobs, r), :]
    
    # 如果数据最初是一维的，最终结果只取第一列作为输出
    if ndim == 1:
        init = init[:, 0]
    
    # 返回初始化后的聚类中心矩阵
    return init
# 有效的初始化方法字典，将字符串映射到对应的初始化函数
_valid_init_meth = {'random': _krandinit, 'points': _kpoints, '++': _kpp}


def _missing_warn():
    """打印警告信息的函数，当调用时使用。"""
    warnings.warn("One of the clusters is empty. "
                  "Re-run kmeans with a different initialization.",
                  stacklevel=3)


def _missing_raise():
    """抛出 ClusterError 异常的函数，当调用时使用。"""
    raise ClusterError("One of the clusters is empty. "
                       "Re-run kmeans with a different initialization.")


# 有效的处理空簇方法字典，将字符串映射到对应的处理函数
_valid_miss_meth = {'warn': _missing_warn, 'raise': _missing_raise}


def kmeans2(data, k, iter=10, thresh=1e-5, minit='random',
            missing='warn', check_finite=True, *, seed=None):
    """
    使用 k-means 算法将一组观测分类成 k 个簇。

    该算法试图最小化观测值与质心之间的欧几里德距离。包含多种初始化方法。

    Parameters
    ----------
    data : ndarray
        'M' 行 'N' 列的数组，包含 'M' 个观测值，每个观测值 'N' 维度；或者包含 'M' 个 1-D 观测值的长度为 'M' 的数组。
    k : int or ndarray
        要形成的簇数，同时也是要生成的质心数。如果 `minit` 的初始化字符串是 'matrix'，或者给定一个 ndarray，则会解释为要使用的初始簇。
    iter : int, optional
        运行 k-means 算法的迭代次数。注意，这与 kmeans 函数的 iters 参数含义不同。
    thresh : float, optional
        阈值（目前未使用）。
    minit : str, optional
        初始化方法。可用方法有 'random'、'points'、'++' 和 'matrix'：

        'random': 从数据的均值和方差估计中生成 k 个质心。
        
        'points': 从数据中随机选择 k 个观测值（行）作为初始质心。
        
        '++': 根据 kmeans++ 方法选择 k 个观测值（需要小心的种子选择）。

        'matrix': 将 k 参数解释为 k 行 M 列（或者对于 1-D 数据的长度为 k 的数组）的初始质心。
    missing : str, optional
        处理空簇的方法。可用方法有 'warn' 和 'raise'：

        'warn': 发出警告并继续执行。
        
        'raise': 抛出 ClusterError 并终止算法。
    check_finite : bool, optional
        是否检查输入矩阵只包含有限数。禁用可能会提高性能，但如果输入包含无穷大或 NaN，可能会导致问题（崩溃、不终止）。
        默认值为 True。
    seed : int or None, optional
        随机数种子，用于初始化。默认为 None。
    """
    # 如果迭代次数 iter 不是正整数，则抛出数值错误异常
    if int(iter) < 1:
        raise ValueError("Invalid iter (%s), "
                         "must be a positive integer." % iter)
    try:
        # 尝试根据 missing 的值从 _valid_miss_meth 字典中获取相应的方法
        miss_meth = _valid_miss_meth[missing]
    except KeyError as e:
        # 如果 missing 的值在 _valid_miss_meth 字典中不存在，则抛出值错误异常
        raise ValueError(f"Unknown missing method {missing!r}") from e

    # 如果 k 是整数，则使用 data 的命名空间创建 xp 数组
    if isinstance(k, int):
        xp = array_namespace(data)
    else:
        # 否则，使用 data 和 k 的命名空间创建 xp 数组
        xp = array_namespace(data, k)
    
    # 将 data 转换为数组，并检查其是否有限
    data = _asarray(data, xp=xp, check_finite=check_finite)
    
    # 复制 k 到 code_book，使用 xp 数组
    code_book = copy(k, xp=xp)
    
    # 如果 data 的维度为 1，则将 d 设置为 1
    if data.ndim == 1:
        d = 1
    # 如果数据的维度为2，获取数据的第二个维度的大小
    elif data.ndim == 2:
        d = data.shape[1]
    else:
        # 如果数据的维度大于2，则抛出数值错误异常
        raise ValueError("Input of rank > 2 is not supported.")

    # 如果数据或者码本为空，则抛出数值错误异常
    if size(data) < 1 or size(code_book) < 1:
        raise ValueError("Empty input is not supported.")

    # 如果 minit 是 'matrix' 或者码本大小大于1，则检查 k 是否与数据的形状兼容
    if minit == 'matrix' or size(code_book) > 1:
        # 如果数据维度与码本维度不一致，则抛出数值错误异常
        if data.ndim != code_book.ndim:
            raise ValueError("k array doesn't match data rank")
        # 获取码本的聚类数目 nc
        nc = code_book.shape[0]
        # 如果数据的维度大于1且码本的第二维度不等于 d，则抛出数值错误异常
        if data.ndim > 1 and code_book.shape[1] != d:
            raise ValueError("k array doesn't match data dimension")
    else:
        # 将码本转换为整数类型
        nc = int(code_book)

        # 如果 nc 小于1，则抛出数值错误异常
        if nc < 1:
            raise ValueError("Cannot ask kmeans2 for %d clusters"
                             " (k was %s)" % (nc, code_book))
        # 如果 nc 不等于原始的 code_book 值，则发出警告并进行转换
        elif nc != code_book:
            warnings.warn("k was not an integer, was converted.", stacklevel=2)

        try:
            # 获取有效的初始化方法
            init_meth = _valid_init_meth[minit]
        except KeyError as e:
            # 如果初始化方法未知，则抛出数值错误异常
            raise ValueError(f"Unknown init method {minit!r}") from e
        else:
            # 检查随机状态
            rng = check_random_state(seed)
            # 使用初始化方法初始化码本
            code_book = init_meth(data, code_book, rng, xp)

    # 将数据和码本转换为 NumPy 数组
    data = np.asarray(data)
    code_book = np.asarray(code_book)

    # 迭代 iter 次数
    for i in range(iter):
        # 使用当前码本计算每个观测的最近邻居
        label = vq(data, code_book, check_finite=check_finite)[0]
        # 更新码本，计算新的质心
        new_code_book, has_members = _vq.update_cluster_means(data, label, nc)
        # 如果存在空的聚类，则调用 miss_meth 函数
        if not has_members.all():
            miss_meth()
            # 将空的聚类设为其先前的位置
            new_code_book[~has_members] = code_book[~has_members]
        # 更新码本
        code_book = new_code_book

    # 返回码本和标签的 NumPy 数组表示
    return xp.asarray(code_book), xp.asarray(label)
```