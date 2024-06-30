# `D:\src\scipysrc\seaborn\seaborn\external\kde.py`

```
# 导入 NumPy 库，并从中导入所需函数和模块
import numpy as np
# 从 NumPy 中导入常用函数和工具函数，如数组处理函数和维度操作函数
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, dot, exp, pi,
                   sqrt, power, atleast_1d, sum, ones, cov)
# 从 NumPy 中导入线性代数模块
from numpy import linalg

# 定义模块中公开的类名称，用于高斯核密度估计
__all__ = ['gaussian_kde']

# 定义一个用于高斯核密度估计的类
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
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    evaluate
        Evaluates the estimated PDF at given points.
    __call__
        Enables instance to be called like a function, equivalent to `evaluate`.
    integrate_gaussian
        Integrates the estimated PDF over a Gaussian region.
    integrate_box_1d
        Integrates the estimated PDF over a 1D box region.
    integrate_box
        Integrates the estimated PDF over a hyper-rectangle region.
    integrate_kde
        Integrates the estimated PDF over a KDE region.
    pdf
        Computes the estimated PDF at given points.
    logpdf
        Computes the logarithm of the estimated PDF at given points.
    resample
        Generates random samples from the estimated PDF.
    set_bandwidth
        Sets the bandwidth used in the KDE estimation.
    covariance_factor
        Computes the factor used for bandwidth selection.

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    In the case of unequally weighted points, `scotts_factor` becomes::

        neff**(-1./(d+4)),

    with ``neff`` the effective number of datapoints.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    or in the case of unequally weighted points::

        (neff * (d + 2) / 4.)**(-1. / (d + 4)).
    """
    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    With a set of weighted samples, the effective number of datapoints ``neff``
    is defined by::

        neff = sum(weights)^2 / sum(weights^2)

    as detailed in [5]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.
    .. [5] Gray P. G., 1969, Journal of the Royal Statistical Society.
           Series A (General), 132, 272

    """
    # 定义一个核密度估计类
    def __init__(self, dataset, bw_method=None, weights=None):
        # 将输入数据集转换为至少是二维的数组
        self.dataset = atleast_2d(asarray(dataset))
        # 检查数据集大小是否大于1，否则引发值错误异常
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        # 获取数据集的维度（d）和样本数量（n）
        self.d, self.n = self.dataset.shape

        # 如果提供了权重（weights）参数
        if weights is not None:
            # 将权重数组转换为至少是一维的浮点型数组
            self._weights = atleast_1d(weights).astype(float)
            # 标准化权重数组使其总和为1
            self._weights /= sum(self._weights)
            # 检查权重数组是否是一维的
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            # 检查权重数组长度是否与样本数量相等
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            # 计算有效样本数（neff）
            self._neff = 1/sum(self._weights**2)

        # 设置带宽（bandwidth）参数
        self.set_bandwidth(bw_method=bw_method)
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
        # Ensure points is at least 2-dimensional
        points = atleast_2d(asarray(points))

        # Retrieve dimensions of points
        d, m = points.shape

        # Check if dimensions of points match KDE's dimensionality
        if d != self.d:
            # Check if points can be reshaped to match dataset dimensions
            if d == 1 and m == self.d:
                # Reshape points as a row vector if possible
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = f"points have dimension {d}, dataset has dimension {self.d}"
                raise ValueError(msg)

        # Determine output data type for results
        output_dtype = np.common_type(self.covariance, points)

        # Initialize result array
        result = zeros((m,), dtype=output_dtype)

        # Compute whitening matrix using Cholesky decomposition of inverse covariance
        whitening = linalg.cholesky(self.inv_cov)

        # Scale dataset and points using whitening matrix
        scaled_dataset = dot(whitening, self.dataset)
        scaled_points = dot(whitening, points)

        if m >= self.n:
            # Loop over dataset because there are more points than data
            for i in range(self.n):
                # Compute squared differences and energy for each point
                diff = scaled_dataset[:, i, newaxis] - scaled_points
                energy = sum(diff * diff, axis=0) / 2.0
                result += self.weights[i] * exp(-energy)
        else:
            # Loop over points because there are fewer points than data
            for i in range(m):
                diff = scaled_dataset - scaled_points[:, i, newaxis]
                energy = sum(diff * diff, axis=0) / 2.0
                result[i] = sum(exp(-energy) * self.weights, axis=0)

        # Normalize result by the normalization factor
        result = result / self._norm_factor

        # Return the evaluated results
        return result

    __call__ = evaluate

    def scotts_factor(self):
        """Compute Scott's factor.

        Returns
        -------
        s : float
            Scott's factor.
        """
        return power(self.neff, -1. / (self.d + 4))

    def silverman_factor(self):
        """Compute the Silverman factor.

        Returns
        -------
        s : float
            The silverman factor.
        """
        return power(self.neff * (self.d + 2.0) / 4.0, -1. / (self.d + 4))

    # Default method to calculate bandwidth, can be overwritten by subclass
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

        """
        # 如果 bw_method 为 None，则不进行任何操作，保持当前的带宽计算方法
        if bw_method is None:
            pass
        # 如果 bw_method 为 'scott'，使用 Scott's 方法计算带宽
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        # 如果 bw_method 为 'silverman'，使用 Silverman's 方法计算带宽
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        # 如果 bw_method 是一个标量且不是字符串，则使用一个常数作为带宽因子
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        # 如果 bw_method 是可调用的函数，则使用该函数计算带宽
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        # 如果 bw_method 不符合预期，则抛出 ValueError 异常
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        # 计算协方差矩阵
        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        # 获取带宽因子
        self.factor = self.covariance_factor()
        # 缓存数据的协方差矩阵和逆协方差矩阵
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1,
                                               bias=False,
                                               aweights=self.weights))
            self._data_inv_cov = linalg.inv(self._data_covariance)

        # 计算协方差矩阵、逆协方差矩阵以及归一化因子
        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = sqrt(linalg.det(2*pi*self.covariance))

    def pdf(self, x):
        """
        Evaluate the estimated pdf on a provided set of points.

        Notes
        -----
        This is an alias for `gaussian_kde.evaluate`.  See the ``evaluate``
        docstring for more details.

        """
        # 对提供的点集计算估计的概率密度函数值
        return self.evaluate(x)

    @property
    def weights(self):
        try:
            # 返回已经存在的权重值
            return self._weights
        except AttributeError:
            # 如果权重值不存在，则初始化权重为均匀分布
            self._weights = ones(self.n)/self.n
            return self._weights
    # 定义一个方法 `neff`，用于计算有效样本数
    def neff(self):
        try:
            # 尝试返回已计算的有效样本数 `_neff`
            return self._neff
        except AttributeError:
            # 如果属性 `_neff` 不存在，则进行计算
            # 计算有效样本数 `_neff`，根据权重数组 `weights` 的平方和的倒数
            self._neff = 1/sum(self.weights**2)
            # 返回计算得到的有效样本数 `_neff`
            return self._neff
```