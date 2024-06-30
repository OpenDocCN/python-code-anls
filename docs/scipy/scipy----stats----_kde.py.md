# `D:\src\scipysrc\scipy\scipy\stats\_kde.py`

```
#-------------------------------------------------------------------------------
#
#  Define classes for (uni/multi)-variate kernel density estimation.
#  定义用于单变量和多变量核密度估计的类。
#
#  Currently, only Gaussian kernels are implemented.
#  目前仅实现了高斯核函数。
#
#  Written by: Robert Kern
#  作者：Robert Kern
#
#  Date: 2004-08-09
#  日期：2004年8月9日
#
#  Modified: 2005-02-10 by Robert Kern.
#  修改：2005年2月10日，Robert Kern进行了修改。
#              Contributed to SciPy
#              贡献给了SciPy
#            2005-10-07 by Robert Kern.
#            修改：2005年10月7日，Robert Kern进行了一些修复以匹配新的scipy_core。
#
#  Copyright 2004-2005 by Enthought, Inc.
#  版权所有2004-2005年 Enthought, Inc.
#
#-------------------------------------------------------------------------------

# Standard library imports.
# 标准库导入。
import warnings

# SciPy imports.
# SciPy导入。
from scipy import linalg, special
from scipy._lib._util import check_random_state

# NumPy imports.
# NumPy导入。
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
                   sqrt, ravel, power, atleast_1d, squeeze, sum, transpose,
                   ones, cov)
import numpy as np

# Local imports.
# 本地导入。
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log

# Exported symbols.
# 导出的符号。
__all__ = ['gaussian_kde']

# Class for Gaussian Kernel Density Estimation.
# 用于高斯核密度估计的类。
class gaussian_kde:
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, optional
        weights of datapoints. This must be the same shape as dataset.
        If None (default), the samples are assumed to be equally weighted

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : int
        Effective number of datapoints.

        .. versionadded:: 1.2.0
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`. The square
        of `kde.factor` multiplies the covariance matrix of the data in the kde
        estimation.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    """
    # 表示使用高斯核的核密度估计的类。

    def __init__(self, dataset, bw_method=None, weights=None):
        """Initialize a gaussian_kde instance.

        Parameters
        ----------
        dataset : array_like
            Datapoints to estimate from. In case of univariate data this is a
            1-D array, otherwise a 2-D array with shape (# of dims, # of data).
            数据点用于估计。对于单变量数据，这是一个1维数组，对于多变量数据，这是一个形状为(# of dims, # of data)的2维数组。
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a scalar,
            this will be used directly as `kde.factor`.  If a callable, it should
            take a `gaussian_kde` instance as only parameter and return a scalar.
            If None (default), 'scott' is used.  See Notes for more details.
            用于计算估计带宽的方法。可以是'scott'、'silverman'、标量常数或可调用对象。
            如果是标量，将直接用作`kde.factor`。
            如果是可调用对象，应该以`gaussian_kde`实例为唯一参数，并返回一个标量。
            如果为None（默认），则使用'scott'方法。详见Notes。
        weights : array_like, optional
            weights of datapoints. This must be the same shape as dataset.
            If None (default), the samples are assumed to be equally weighted
            数据点的权重。必须与dataset具有相同的形状。
            如果为None（默认），则假定样本具有相等的权重。
        """
        self.dataset = atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        self.neff = int(sum(weights) ** 2 / sum(weights ** 2))

        if weights is not None:
            self.weights = asarray(weights)
        else:
            self.weights = ones(self.n) / self.n

        # Compute the covariance matrix scaled by the bandwidth factor.
        # 计算通过带宽因子缩放的协方差矩阵。
        self.factor = self.d ** -0.5

        # Covariance matrix of the dataset, scaled by the bandwidth.
        # 数据集的协方差矩阵，通过带宽缩放。
        self.covariance = atleast_2d(cov(self.dataset, aweights=self.weights))

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the estimated pdf.

        Returns
        -------
        values : ndarray
            The values of the estimated pdf evaluated at `points`.
        """
        points = atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            raise ValueError("points have dimension %s, dataset has dimension %s" % (d, self.d))

        # Compute the kernel density estimate.
        # 计算核密度估计。
        kde = zeros(m)
        # Loop over all data points.
        # 遍历所有数据点。
        for i in range(self.n):
            diff = self.dataset[:, i, newaxis] - points
            tdiff = np.dot(self.covariance, diff)
            energy = sum(diff * tdiff, axis=0) / 2.0
            kde += self.weights[i] * exp(-energy)
        # Normalize the result.
        # 标准化结果。
        kde /= sqrt((2 * pi) ** self.d * linalg.det(self.covariance))
        return kde

    def __call__(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the estimated pdf.

        Returns
        -------
        values : ndarray
            The values of the estimated pdf evaluated at `points`.
        """
        return self.evaluate(points)
    # inv_cov : ndarray
    #     The inverse of `covariance`.

    # Methods
    # -------
    # evaluate
    # __call__
    # integrate_gaussian
    # integrate_box_1d
    # integrate_box
    # integrate_kde
    # pdf
    # logpdf
    # resample
    # set_bandwidth
    # covariance_factor

    # Notes
    # -----
    # Bandwidth selection strongly influences the estimate obtained from the KDE
    # (much more so than the actual shape of the kernel).  Bandwidth selection
    # can be done by a "rule of thumb", by cross-validation, by "plug-in
    # methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    # uses a rule of thumb, the default is Scott's Rule.

    # Scott's Rule [1]_, implemented as `scotts_factor`, is::
    # 
    #     n**(-1./(d+4)),
    # 
    # with ``n`` the number of data points and ``d`` the number of dimensions.
    # In the case of unequally weighted points, `scotts_factor` becomes::
    # 
    #     neff**(-1./(d+4)),
    # 
    # with ``neff`` the effective number of datapoints.
    # Silverman's Rule [2]_, implemented as `silverman_factor`, is::
    # 
    #     (n * (d + 2) / 4.)**(-1. / (d + 4)).
    # 
    # or in the case of unequally weighted points::
    # 
    #     (neff * (d + 2) / 4.)**(-1. / (d + 4)).
    # 
    # Good general descriptions of kernel density estimation can be found in [1]_
    # and [2]_, the mathematics for this multi-dimensional implementation can be
    # found in [1]_.
    # 
    # With a set of weighted samples, the effective number of datapoints ``neff``
    # is defined by::
    # 
    #     neff = sum(weights)^2 / sum(weights^2)
    # 
    # as detailed in [5]_.
    # 
    # `gaussian_kde` does not currently support data that lies in a
    # lower-dimensional subspace of the space in which it is expressed. For such
    # data, consider performing principle component analysis / dimensionality
    # reduction and using `gaussian_kde` with the transformed data.
    # 
    # References
    # ----------
    # .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
    #        Visualization", John Wiley & Sons, New York, Chicester, 1992.
    # .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
    #        Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
    #        Chapman and Hall, London, 1986.
    # .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
    #        Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    # .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
    #        conditional density estimation", Computational Statistics & Data
    #        Analysis, Vol. 36, pp. 279-298, 2001.
    # .. [5] Gray P. G., 1969, Journal of the Royal Statistical Society.
    #        Series A (General), 132, 272

    # Examples
    # --------
    # Generate some random two-dimensional data:
    # 
    # >>> import numpy as np
    # >>> from scipy import stats
    # >>> def measure(n):
    # ...     "Measurement model, return two coupled measurements."
    # ...     m1 = np.random.normal(size=n)
    # ...     m2 = np.random.normal(scale=0.5, size=n)
    # 返回两个数的和与差
    return m1+m2, m1-m2

    # 使用函数 measure(2000) 获取 m1 和 m2 的值
    m1, m2 = measure(2000)
    # 计算 m1 的最小值和最大值
    xmin = m1.min()
    xmax = m1.max()
    # 计算 m2 的最小值和最大值
    ymin = m2.min()
    ymax = m2.max()

    # 对数据执行核密度估计
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # 绘制结果图像
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax])
    ax.plot(m1, m2, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()

    # 初始化 Gaussian KDE 对象
    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = atleast_2d(asarray(dataset))
        # 检查输入数据集是否大于一个元素
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        # 如果指定了权重，对权重进行处理
        if weights is not None:
            self._weights = atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1/sum(self._weights**2)

        # 如果数据维度大于样本数，抛出异常
        if self.d > self.n:
            msg = ("Number of dimensions is greater than number of samples. "
                   "This results in a singular data covariance matrix, which "
                   "cannot be treated using the algorithms implemented in "
                   "`gaussian_kde`. Note that `gaussian_kde` interprets each "
                   "*column* of `dataset` to be a point; consider transposing "
                   "the input to `dataset`.")
            raise ValueError(msg)

        # 尝试设置带宽参数，处理可能出现的线性代数错误
        try:
            self.set_bandwidth(bw_method=bw_method)
        except linalg.LinAlgError as e:
            msg = ("The data appears to lie in a lower-dimensional subspace "
                   "of the space in which it is expressed. This has resulted "
                   "in a singular data covariance matrix, which cannot be "
                   "treated using the algorithms implemented in "
                   "`gaussian_kde`. Consider performing principle component "
                   "analysis / dimensionality reduction and using "
                   "`gaussian_kde` with the transformed data.")
            raise linalg.LinAlgError(msg) from e
    # 将评估函数定义为类的方法，用于计算估计的概率密度函数在一组点上的值
    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        # 将点至少处理为二维数组
        points = atleast_2d(asarray(points))

        # 获取点的维度 d 和点的数量 m
        d, m = points.shape

        # 如果点的维度 d 不等于 KDE 的维度 self.d
        if d != self.d:
            # 如果 d 是 1 并且 m 等于 self.d，则将 points 视为行向量处理
            if d == 1 and m == self.d:
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                # 否则，抛出异常，指示点的维度与数据集的维度不匹配
                msg = (f"points have dimension {d}, "
                       f"dataset has dimension {self.d}")
                raise ValueError(msg)

        # 根据数据点的特性和协方差类型获取输出的数据类型和规格
        output_dtype, spec = _get_output_dtype(self.covariance, points)

        # 使用高斯核函数评估估计的密度函数在给定点上的值
        result = gaussian_kernel_estimate[spec](
            self.dataset.T, self.weights[:, None],
            points.T, self.cho_cov, output_dtype)

        # 返回结果的第一列（因为结果是二维数组，我们只需要第一列的值）
        return result[:, 0]

    # 将 __call__ 方法指定为 evaluate 方法，使对象可调用
    __call__ = evaluate

    # 定义一个方法，用于将估计的密度乘以多元高斯分布并在整个空间上进行积分
    def integrate_gaussian(self, mean, cov):
        """
        Multiply estimated density by a multivariate Gaussian and integrate
        over the whole space.

        Parameters
        ----------
        mean : aray_like
            A 1-D array, specifying the mean of the Gaussian.
        cov : array_like
            A 2-D array, specifying the covariance matrix of the Gaussian.

        Returns
        -------
        result : scalar
            The value of the integral.

        Raises
        ------
        ValueError
            If the mean or covariance of the input Gaussian differs from
            the KDE's dimensionality.

        """
        # 将 mean 至少处理为一维数组并去除多余的维度
        mean = atleast_1d(squeeze(mean))

        # 将 cov 至少处理为二维数组
        cov = atleast_2d(cov)

        # 如果 mean 的形状不等于 self.d，则抛出异常
        if mean.shape != (self.d,):
            raise ValueError("mean does not have dimension %s" % self.d)
        
        # 如果 cov 的形状不等于 (self.d, self.d)，则抛出异常
        if cov.shape != (self.d, self.d):
            raise ValueError("covariance does not have dimension %s" % self.d)

        # 将 mean 转换为列向量
        mean = mean[:, newaxis]

        # 计算合并后的协方差矩阵 sum_cov
        sum_cov = self.covariance + cov

        # 使用 cho_factor 对合并后的协方差矩阵进行 Cholesky 分解
        # 如果新的协方差矩阵不是正定的，将会引发 LinAlgError 异常
        sum_cov_chol = linalg.cho_factor(sum_cov)

        # 计算数据集与 mean 之间的差异
        diff = self.dataset - mean

        # 使用 sum_cov_chol 对 diff 进行 Cholesky 解
        tdiff = linalg.cho_solve(sum_cov_chol, diff)

        # 计算 Cholesky 分解的行列式的平方根
        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))

        # 计算归一化常数
        norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det

        # 计算能量值
        energies = sum(diff * tdiff, axis=0) / 2.0

        # 计算结果，即估计的密度乘以多元高斯分布在整个空间上的积分值
        result = sum(exp(-energies)*self.weights, axis=0) / norm_const

        # 返回计算得到的积分结果
        return result
    def integrate_box_1d(self, low, high):
        """
        Computes the integral of a 1D pdf between two bounds.

        Parameters
        ----------
        low : scalar
            Lower bound of integration.
        high : scalar
            Upper bound of integration.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDE is over more than one dimension.

        """
        # 检查核密度估计是否为一维，否则抛出值错误异常
        if self.d != 1:
            raise ValueError("integrate_box_1d() only handles 1D pdfs")

        # 计算标准差作为矩阵的平方根
        stdev = ravel(sqrt(self.covariance))[0]

        # 标准化低边界和高边界
        normalized_low = ravel((low - self.dataset) / stdev)
        normalized_high = ravel((high - self.dataset) / stdev)

        # 计算积分值，使用权重乘以标准正态分布的差值
        value = np.sum(self.weights * (
                        special.ndtr(normalized_high) -
                        special.ndtr(normalized_low)))
        return value

    def integrate_box(self, low_bounds, high_bounds, maxpts=None):
        """Computes the integral of a pdf over a rectangular interval.

        Parameters
        ----------
        low_bounds : array_like
            A 1-D array containing the lower bounds of integration.
        high_bounds : array_like
            A 1-D array containing the upper bounds of integration.
        maxpts : int, optional
            The maximum number of points to use for integration.

        Returns
        -------
        value : scalar
            The result of the integral.

        """
        # 如果提供了最大点数，使用额外的关键字参数传递给 _mvn.mvnun_weighted
        if maxpts is not None:
            extra_kwds = {'maxpts': maxpts}
        else:
            extra_kwds = {}

        # 调用 _mvn.mvnun_weighted 计算多元正态分布的加权积分值和通知信息
        value, inform = _mvn.mvnun_weighted(low_bounds, high_bounds,
                                            self.dataset, self.weights,
                                            self.covariance, **extra_kwds)
        
        # 如果有通知信息，发出警告并指定警告的堆栈级别
        if inform:
            msg = ('An integral in _mvn.mvnun requires more points than %s' %
                   (self.d * 1000))
            warnings.warn(msg, stacklevel=2)

        return value
    def integrate_kde(self, other):
        """
        Computes the integral of the product of this kernel density estimate
        with another.

        Parameters
        ----------
        other : gaussian_kde instance
            The other kde.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDEs have different dimensionality.

        """
        # 检查两个 KDE 的维度是否相同，如果不同则抛出异常
        if other.d != self.d:
            raise ValueError("KDEs are not the same dimensionality")

        # 确定要迭代的点数量较少的对象
        if other.n < self.n:
            small = other
            large = self
        else:
            small = self
            large = other

        # 计算两个协方差的和
        sum_cov = small.covariance + large.covariance
        # 对和的 Cholesky 分解
        sum_cov_chol = linalg.cho_factor(sum_cov)
        result = 0.0
        # 遍历小数据集中的每个点
        for i in range(small.n):
            mean = small.dataset[:, i, newaxis]
            # 计算大数据集与当前均值的差异
            diff = large.dataset - mean
            # 解 Cholesky 分解后的方程
            tdiff = linalg.cho_solve(sum_cov_chol, diff)

            # 计算能量
            energies = sum(diff * tdiff, axis=0) / 2.0
            # 计算积分的一部分
            result += sum(exp(-energies) * large.weights, axis=0) * small.weights[i]

        # 计算 Cholesky 分解的平方根的行列式
        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
        # 计算归一化常数
        norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det

        # 归一化结果
        result /= norm_const

        return result

    def resample(self, size=None, seed=None):
        """Randomly sample a dataset from the estimated pdf.

        Parameters
        ----------
        size : int, optional
            The number of samples to draw. If not provided, then the size is
            the same as the effective number of samples in the underlying
            dataset.
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance then
            that instance is used.

        Returns
        -------
        resample : (self.d, `size`) ndarray
            The sampled dataset.

        """ # numpy/numpydoc#87  # noqa: E501
        # 如果 size 未指定，则使用有效样本数 self.neff
        if size is None:
            size = int(self.neff)

        # 获取随机数生成器状态
        random_state = check_random_state(seed)
        # 生成多元正态分布随机数，进行转置
        norm = transpose(random_state.multivariate_normal(
            zeros((self.d,), float), self.covariance, size=size
        ))
        # 使用权重从数据集中选择索引
        indices = random_state.choice(self.n, size=size, p=self.weights)
        # 获取对应索引的均值
        means = self.dataset[:, indices]

        return means + norm

    def scotts_factor(self):
        """Compute Scott's factor.

        Returns
        -------
        s : float
            Scott's factor.
        """
        # 计算 Scott's factor
        return power(self.neff, -1./(self.d+4))
    # 计算 Silverman 因子的方法
    def silverman_factor(self):
        """Compute the Silverman factor.

        Returns
        -------
        s : float
            The silverman factor.
        """
        # 使用公式计算 Silverman 因子
        return power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))

    # 默认的带宽计算方法，可以被子类覆盖
    covariance_factor = scotts_factor
    covariance_factor.__doc__ = """Computes the coefficient (`kde.factor`) that
        multiplies the data covariance matrix to obtain the kernel covariance
        matrix. The default is `scotts_factor`.  A subclass can overwrite this
        method to provide a different method, or set it through a call to
        `kde.set_bandwidth`."""
    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> x1 = np.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = np.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x1, np.full(x1.shape, 1 / (4. * x1.size)), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        # 如果 bw_method 为 None，则不执行任何操作，保持当前的带宽计算方法不变
        if bw_method is None:
            pass
        # 如果 bw_method 为 'scott'，则设置带宽计算方法为 Scott 方法
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        # 如果 bw_method 为 'silverman'，则设置带宽计算方法为 Silverman 方法
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        # 如果 bw_method 是标量且不是字符串，则设置带宽计算方法为常数
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        # 如果 bw_method 是可调用对象，则设置带宽计算方法为该对象返回值
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        # 如果 bw_method 类型不符合预期，则抛出 ValueError 异常
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        # 计算协方差矩阵
        self._compute_covariance()
    def _compute_covariance(self):
        """计算每个高斯核的协方差矩阵，使用covariance_factor()函数计算因子。
        """
        self.factor = self.covariance_factor()
        # 缓存协方差和协方差的Cholesky分解
        if not hasattr(self, '_data_cho_cov'):
            self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1,
                                               bias=False,
                                               aweights=self.weights))
            self._data_cho_cov = linalg.cholesky(self._data_covariance,
                                                 lower=True)

        self.covariance = self._data_covariance * self.factor**2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
        self.log_det = 2*np.log(np.diag(self.cho_cov
                                        * np.sqrt(2*pi))).sum()

    @property
    def inv_cov(self):
        # 每次从头计算，因为不确定如何在实际中使用。可能用户会改变`dataset`，因为它不是私有属性？
        # `_compute_covariance`用于重新计算所有这些，所以现在这是一个属性，我们将重新计算一切。
        self.factor = self.covariance_factor()
        self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1,
                                           bias=False, aweights=self.weights))
        return linalg.inv(self._data_covariance) / self.factor**2

    def pdf(self, x):
        """
        在提供的一组点上评估估计的概率密度函数（PDF）。

        Notes
        -----
        这是`gaussian_kde.evaluate`的别名。查看`evaluate`的文档字符串以获取更多细节。

        """
        return self.evaluate(x)

    def logpdf(self, x):
        """
        在提供的一组点上评估估计概率密度函数（PDF）的对数。
        """
        points = atleast_2d(x)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points被传递为行向量
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = (f"points have dimension {d}, "
                       f"dataset has dimension {self.d}")
                raise ValueError(msg)

        output_dtype, spec = _get_output_dtype(self.covariance, points)
        result = gaussian_kernel_estimate_log[spec](
            self.dataset.T, self.weights[:, None],
            points.T, self.cho_cov, output_dtype)

        return result[:, 0]
    def marginal(self, dimensions):
        """Return a marginal KDE distribution

        Parameters
        ----------
        dimensions : int or 1-d array_like
            The dimensions of the multivariate distribution corresponding
            with the marginal variables, that is, the indices of the dimensions
            that are being retained. The other dimensions are marginalized out.

        Returns
        -------
        marginal_kde : gaussian_kde
            An object representing the marginal distribution.

        Notes
        -----
        .. versionadded:: 1.10.0

        """

        # 将输入的维度参数转换为至少一维的 numpy 数组
        dims = np.atleast_1d(dimensions)

        # 检查维度参数是否都为整数
        if not np.issubdtype(dims.dtype, np.integer):
            msg = ("Elements of `dimensions` must be integers - the indices "
                   "of the marginal variables being retained.")
            raise ValueError(msg)

        # 计算数据集的维度数目
        n = len(self.dataset)  # number of dimensions
        # 保存原始的维度参数
        original_dims = dims.copy()

        # 处理负数的维度索引，转换为正数索引
        dims[dims < 0] = n + dims[dims < 0]

        # 检查维度参数是否都是唯一的
        if len(np.unique(dims)) != len(dims):
            msg = ("All elements of `dimensions` must be unique.")
            raise ValueError(msg)

        # 检查维度参数是否在有效范围内
        i_invalid = (dims < 0) | (dims >= n)
        if np.any(i_invalid):
            msg = (f"Dimensions {original_dims[i_invalid]} are invalid "
                   f"for a distribution in {n} dimensions.")
            raise ValueError(msg)

        # 根据给定的维度参数获取数据集的子集
        dataset = self.dataset[dims]
        # 获取权重数据
        weights = self.weights

        # 使用 gaussian_kde 函数创建并返回边际 KDE 分布对象
        return gaussian_kde(dataset, bw_method=self.covariance_factor(),
                            weights=weights)

    @property
    def weights(self):
        try:
            # 如果已经存在权重属性，则直接返回
            return self._weights
        except AttributeError:
            # 如果不存在权重属性，则根据数据集的大小设置均匀权重
            self._weights = ones(self.n)/self.n
            return self._weights

    @property
    def neff(self):
        try:
            # 如果已经存在有效样本数属性，则直接返回
            return self._neff
        except AttributeError:
            # 如果不存在有效样本数属性，则根据权重计算并返回
            self._neff = 1/sum(self.weights**2)
            return self._neff
# 计算输出的数据类型和C类型名称的函数
def _get_output_dtype(covariance, points):
    """
    Calculates the output dtype and the "spec" (=C type name).

    This was necessary in order to deal with the fused types in the Cython
    routine `gaussian_kernel_estimate`. See gh-10824 for details.
    """
    # 获取输入数组 covariance 和 points 的公共数据类型
    output_dtype = np.common_type(covariance, points)
    # 获取数据类型对象的字节大小
    itemsize = np.dtype(output_dtype).itemsize
    # 根据数据类型对象的字节大小确定 C 类型名称
    if itemsize == 4:
        spec = 'float'
    elif itemsize == 8:
        spec = 'double'
    elif itemsize in (12, 16):
        spec = 'long double'
    else:
        # 如果字节大小不符合预期，则抛出 ValueError 异常
        raise ValueError(
                f"{output_dtype} has unexpected item size: {itemsize}"
            )

    # 返回计算得到的输出数据类型和对应的 C 类型名称
    return output_dtype, spec
```