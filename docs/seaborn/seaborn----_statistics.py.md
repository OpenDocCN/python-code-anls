# `D:\src\scipysrc\seaborn\seaborn\_statistics.py`

```
# 导入必要的模块和库
from numbers import Number  # 导入数字类型相关的模块
from statistics import NormalDist  # 导入统计模块中的正态分布
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理和分析
try:
    from scipy.stats import gaussian_kde  # 尝试从SciPy中导入高斯核密度估计函数
    _no_scipy = False  # 如果成功导入SciPy，设置标志为False
except ImportError:
    from .external.kde import gaussian_kde  # 如果导入失败，从外部模块导入高斯核密度估计函数
    _no_scipy = True  # 设置标志为True，表示没有成功导入SciPy

from .algorithms import bootstrap  # 从本地algorithms模块中导入bootstrap算法
from .utils import _check_argument  # 从本地utils模块中导入_check_argument函数


class KDE:
    """Univariate and bivariate kernel density estimator."""

    def __init__(
        self, *,
        bw_method=None,  # 带宽选择方法，用于核密度估计的参数
        bw_adjust=1,  # 带宽调整系数，默认为1
        gridsize=200,  # 网格大小，默认为200
        cut=3,  # 截断范围，控制核密度估计的范围
        clip=None,  # 剪切参数，用于限制估计密度的范围
        cumulative=False,  # 是否累积计算，默认为False，表示非累积计算
    ):
        """
        Initialize the estimator with its parameters.

        Parameters
        ----------
        bw_method : string, scalar, or callable, optional
            Method for determining the smoothing bandwidth to use; passed to
            :class:`scipy.stats.gaussian_kde`.
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using
            ``bw_method``. Increasing will make the curve smoother. See Notes.
        gridsize : int, optional
            Number of points on each dimension of the evaluation grid.
        cut : number, optional
            Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        clip : pair of numbers or None, or a pair of such pairs
            Do not evaluate the density outside of these limits.
        cumulative : bool, optional
            If True, estimate a cumulative distribution function. Requires scipy.

        """
        if clip is None:
            clip = None, None

        # 设置对象的各个参数
        self.bw_method = bw_method
        self.bw_adjust = bw_adjust
        self.gridsize = gridsize
        self.cut = cut
        self.clip = clip
        self.cumulative = cumulative

        # 如果需要累积分布函数且没有安装 scipy，则抛出运行时错误
        if cumulative and _no_scipy:
            raise RuntimeError("Cumulative KDE evaluation requires scipy")

        # 初始化支持度属性为 None
        self.support = None

    def _define_support_grid(self, x, bw, cut, clip, gridsize):
        """
        Create the grid of evaluation points depending for vector x.
        """
        # 确定剪切的上下界
        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        
        # 计算网格的最小值和最大值
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        
        # 在最小值和最大值之间生成均匀分布的网格点
        return np.linspace(gridmin, gridmax, gridsize)

    def _define_support_univariate(self, x, weights):
        """
        Create a 1D grid of evaluation points.
        """
        # 根据输入数据和权重创建 KDE 对象
        kde = self._fit(x, weights)
        
        # 计算带宽
        bw = np.sqrt(kde.covariance.squeeze())
        
        # 根据带宽和其它参数定义支持度网格
        grid = self._define_support_grid(
            x, bw, self.cut, self.clip, self.gridsize
        )
        return grid

    def _define_support_bivariate(self, x1, x2, weights):
        """
        Create a 2D grid of evaluation points.
        """
        # 处理剪切参数
        clip = self.clip
        if clip[0] is None or np.isscalar(clip[0]):
            clip = (clip, clip)
        
        # 根据输入数据和权重创建 KDE 对象
        kde = self._fit([x1, x2], weights)
        
        # 计算带宽
        bw = np.sqrt(np.diag(kde.covariance).squeeze())

        # 根据带宽和其它参数定义两个维度上的支持度网格
        grid1 = self._define_support_grid(
            x1, bw[0], self.cut, clip[0], self.gridsize
        )
        grid2 = self._define_support_grid(
            x2, bw[1], self.cut, clip[1], self.gridsize
        )

        return grid1, grid2
    # 定义一个方法，用于生成给定数据集的评估网格
    def define_support(self, x1, x2=None, weights=None, cache=True):
        """Create the evaluation grid for a given data set."""
        if x2 is None:
            # 如果数据是单变量的，则调用单变量支持定义方法
            support = self._define_support_univariate(x1, weights)
        else:
            # 如果数据是双变量的，则调用双变量支持定义方法
            support = self._define_support_bivariate(x1, x2, weights)

        # 如果 cache 参数为 True，则将生成的支持网格保存到对象属性中
        if cache:
            self.support = support

        # 返回生成的支持网格
        return support

    # 私有方法，用于拟合 scipy 的 KDE（核密度估计），同时添加带宽调整逻辑和版本检查
    def _fit(self, fit_data, weights=None):
        """Fit the scipy kde while adding bw_adjust logic and version check."""
        # 根据当前对象的带宽方法创建拟合参数
        fit_kws = {"bw_method": self.bw_method}
        # 如果有权重参数，则添加到拟合参数中
        if weights is not None:
            fit_kws["weights"] = weights

        # 使用拟合数据和拟合参数创建 Gaussian KDE 对象
        kde = gaussian_kde(fit_data, **fit_kws)
        # 设置 KDE 对象的带宽，基于当前对象的带宽调整因子
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        # 返回拟合好的 KDE 对象
        return kde

    # 私有方法，用于在单变量数据上拟合和评估单变量 KDE
    def _eval_univariate(self, x, weights=None):
        """Fit and evaluate a univariate on univariate data."""
        # 获取或定义支持网格
        support = self.support
        if support is None:
            support = self.define_support(x, cache=False)

        # 在数据 x 上拟合 KDE
        kde = self._fit(x, weights)

        # 如果需要累积分布，则计算累积密度估计
        if self.cumulative:
            s_0 = support[0]
            # 计算在每个支持点上的累积分布函数值
            density = np.array([
                kde.integrate_box_1d(s_0, s_i) for s_i in support
            ])
        else:
            # 计算在支持网格上的密度估计
            density = kde(support)

        # 返回计算得到的密度估计值和支持网格
        return density, support

    # 私有方法，用于在双变量数据上拟合和评估双变量 KDE
    def _eval_bivariate(self, x1, x2, weights=None):
        """Fit and evaluate a univariate on bivariate data."""
        # 获取或定义支持网格
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, cache=False)

        # 在数据 [x1, x2] 上拟合 KDE
        kde = self._fit([x1, x2], weights)

        # 如果需要累积分布，则计算累积密度估计
        if self.cumulative:
            # 获取支持网格的两个维度
            grid1, grid2 = support
            # 初始化一个用于存储密度值的数组
            density = np.zeros((grid1.size, grid2.size))
            p0 = grid1.min(), grid2.min()
            # 遍历支持网格的每一个点，计算其处的累积分布函数值
            for i, xi in enumerate(grid1):
                for j, xj in enumerate(grid2):
                    density[i, j] = kde.integrate_box(p0, (xi, xj))

        else:
            # 生成支持网格的网格点
            xx1, xx2 = np.meshgrid(*support)
            # 计算在网格点上的密度估计
            density = kde([xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

        # 返回计算得到的密度估计值和支持网格
        return density, support

    # 对象的调用方法，用于在单变量或双变量数据上拟合和评估 KDE
    def __call__(self, x1, x2=None, weights=None):
        """Fit and evaluate on univariate or bivariate data."""
        if x2 is None:
            # 如果没有第二个变量，则调用单变量拟合和评估方法
            return self._eval_univariate(x1, weights)
        else:
            # 如果有第二个变量，则调用双变量拟合和评估方法
            return self._eval_bivariate(x1, x2, weights)
# Note: we no longer use this for univariate histograms in histplot,
# preferring _stats.Hist. We'll deprecate this once we have a bivariate Stat class.
class Histogram:
    """Univariate and bivariate histogram estimator."""

    def __init__(
        self,
        stat="count",
        bins="auto",
        binwidth=None,
        binrange=None,
        discrete=False,
        cumulative=False,
    ):
        """Initialize the estimator with its parameters.

        Parameters
        ----------
        stat : str
            Aggregate statistic to compute in each bin.

            - `count`: show the number of observations in each bin
            - `frequency`: show the number of observations divided by the bin width
            - `probability` or `proportion`: normalize such that bar heights sum to 1
            - `percent`: normalize such that bar heights sum to 100
            - `density`: normalize such that the total area of the histogram equals 1

        bins : str, number, vector, or a pair of such values
            Generic bin parameter that can be the name of a reference rule,
            the number of bins, or the breaks of the bins.
            Passed to :func:`numpy.histogram_bin_edges`.
        binwidth : number or pair of numbers
            Width of each bin, overrides ``bins`` but can be used with
            ``binrange``.
        binrange : pair of numbers or a pair of pairs
            Lowest and highest value for bin edges; can be used either
            with ``bins`` or ``binwidth``. Defaults to data extremes.
        discrete : bool or pair of bools
            If True, set ``binwidth`` and ``binrange`` such that bin
            edges cover integer values in the dataset.
        cumulative : bool
            If True, return the cumulative statistic.

        """
        # List of valid choices for the `stat` parameter
        stat_choices = [
            "count", "frequency", "density", "probability", "proportion", "percent",
        ]
        # Validate the `stat` parameter against the list of valid choices
        _check_argument("stat", stat_choices, stat)

        # Assigning the initialized parameters to the instance variables
        self.stat = stat
        self.bins = bins
        self.binwidth = binwidth
        self.binrange = binrange
        self.discrete = discrete
        self.cumulative = cumulative

        self.bin_kws = None
    def _define_bin_edges(self, x, weights, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments."""
        # 根据传入参数确定数据的起始和结束点
        if binrange is None:
            start, stop = x.min(), x.max()
        else:
            start, stop = binrange

        # 如果数据是离散的，使用整数值作为 bin 边界
        if discrete:
            bin_edges = np.arange(start - .5, stop + 1.5)
        # 否则，根据指定的 bin 宽度生成 bin 边界
        elif binwidth is not None:
            step = binwidth
            bin_edges = np.arange(start, stop + step, step)
            # 处理舍入误差（也许有更优雅的方法？）
            if bin_edges.max() < stop or len(bin_edges) < 2:
                bin_edges = np.append(bin_edges, bin_edges.max() + step)
        else:
            # 使用 numpy 的直方图边界函数生成 bin 边界
            bin_edges = np.histogram_bin_edges(
                x, bins, binrange, weights,
            )
        return bin_edges

    def define_bin_params(self, x1, x2=None, weights=None, cache=True):
        """Given data, return numpy.histogram parameters to define bins."""
        if x2 is None:
            # 对单个变量 x1 定义 bin 参数

            bin_edges = self._define_bin_edges(
                x1, weights, self.bins, self.binwidth, self.binrange, self.discrete,
            )

            if isinstance(self.bins, (str, Number)):
                # 如果 bins 是字符串或数值，确定 bins 的数量和范围
                n_bins = len(bin_edges) - 1
                bin_range = bin_edges.min(), bin_edges.max()
                bin_kws = dict(bins=n_bins, range=bin_range)
            else:
                # 否则直接将 bin 边界作为参数
                bin_kws = dict(bins=bin_edges)

        else:
            # 对两个变量 x1 和 x2 同时定义 bin 参数

            bin_edges = []
            for i, x in enumerate([x1, x2]):

                # 确定每个变量的 bin 参数是共享的还是特定的

                bins = self.bins
                if not bins or isinstance(bins, (str, Number)):
                    pass
                elif isinstance(bins[i], str):
                    bins = bins[i]
                elif len(bins) == 2:
                    bins = bins[i]

                binwidth = self.binwidth
                if binwidth is None:
                    pass
                elif not isinstance(binwidth, Number):
                    binwidth = binwidth[i]

                binrange = self.binrange
                if binrange is None:
                    pass
                elif not isinstance(binrange[0], Number):
                    binrange = binrange[i]

                discrete = self.discrete
                if not isinstance(discrete, bool):
                    discrete = discrete[i]

                # 为每个变量定义对应的 bin 边界
                bin_edges.append(self._define_bin_edges(
                    x, weights, bins, binwidth, binrange, discrete,
                ))

            bin_kws = dict(bins=tuple(bin_edges))

        if cache:
            # 如果指定了缓存，将计算得到的 bin 参数保存在实例的 bin_kws 属性中
            self.bin_kws = bin_kws

        return bin_kws
    def _eval_bivariate(self, x1, x2, weights):
        """Inner function for histogram of two variables."""
        # 获取直方图参数，如果未指定则根据 x1 和 x2 定义参数，不缓存结果
        bin_kws = self.bin_kws
        if bin_kws is None:
            bin_kws = self.define_bin_params(x1, x2, cache=False)

        # 确定是否为密度直方图
        density = self.stat == "density"

        # 计算二维直方图和对应的边界
        hist, *bin_edges = np.histogram2d(
            x1, x2, **bin_kws, weights=weights, density=density
        )

        # 计算每个小区域的面积
        area = np.outer(
            np.diff(bin_edges[0]),
            np.diff(bin_edges[1]),
        )

        # 根据统计类型对直方图进行归一化处理
        if self.stat == "probability" or self.stat == "proportion":
            hist = hist.astype(float) / hist.sum()
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100
        elif self.stat == "frequency":
            hist = hist.astype(float) / area

        # 如果要计算累积分布
        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * area).cumsum(axis=0).cumsum(axis=1)
            else:
                hist = hist.cumsum(axis=0).cumsum(axis=1)

        # 返回直方图和边界
        return hist, bin_edges

    def _eval_univariate(self, x, weights):
        """Inner function for histogram of one variable."""
        # 获取直方图参数，如果未指定则根据 x 和 weights 定义参数，不缓存结果
        bin_kws = self.bin_kws
        if bin_kws is None:
            bin_kws = self.define_bin_params(x, weights=weights, cache=False)

        # 确定是否为密度直方图
        density = self.stat == "density"

        # 计算一维直方图和对应的边界
        hist, bin_edges = np.histogram(
            x, **bin_kws, weights=weights, density=density,
        )

        # 根据统计类型对直方图进行归一化处理
        if self.stat == "probability" or self.stat == "proportion":
            hist = hist.astype(float) / hist.sum()
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100
        elif self.stat == "frequency":
            hist = hist.astype(float) / np.diff(bin_edges)

        # 如果要计算累积分布
        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * np.diff(bin_edges)).cumsum()
            else:
                hist = hist.cumsum()

        # 返回直方图和边界
        return hist, bin_edges

    def __call__(self, x1, x2=None, weights=None):
        """Count the occurrences in each bin, maybe normalize."""
        # 如果只有一个变量，则调用一元直方图函数
        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            # 否则调用二元直方图函数
            return self._eval_bivariate(x1, x2, weights)
class ECDF:
    """Univariate empirical cumulative distribution estimator."""
    
    def __init__(self, stat="proportion", complementary=False):
        """Initialize the class with its parameters

        Parameters
        ----------
        stat : {{"proportion", "percent", "count"}}
            Distribution statistic to compute.
        complementary : bool
            If True, use the complementary CDF (1 - CDF)

        """
        # 检查并确保 stat 参数的合法性
        _check_argument("stat", ["count", "percent", "proportion"], stat)
        self.stat = stat
        self.complementary = complementary

    def _eval_bivariate(self, x1, x2, weights):
        """Inner function for ECDF of two variables."""
        # 抛出未实现错误，因为双变量 ECDF 暂未实现
        raise NotImplementedError("Bivariate ECDF is not implemented")

    def _eval_univariate(self, x, weights):
        """Inner function for ECDF of one variable."""
        # 对输入数据进行排序并计算累积权重
        sorter = x.argsort()
        x = x[sorter]
        weights = weights[sorter]
        y = weights.cumsum()

        # 根据统计类型对累积值进行调整
        if self.stat in ["percent", "proportion"]:
            y = y / y.max()
        if self.stat == "percent":
            y = y * 100

        # 在数组前面添加负无穷以及在累积分布前面添加零
        x = np.r_[-np.inf, x]
        y = np.r_[0, y]

        # 如果设置了 complementary 参数，计算互补的累积分布函数
        if self.complementary:
            y = y.max() - y

        return y, x

    def __call__(self, x1, x2=None, weights=None):
        """Return proportion or count of observations below each sorted datapoint."""
        # 将输入参数转换为 NumPy 数组
        x1 = np.asarray(x1)
        if weights is None:
            weights = np.ones_like(x1)
        else:
            weights = np.asarray(weights)

        # 根据输入参数的不同调用对应的计算函数
        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            return self._eval_bivariate(x1, x2, weights)


class EstimateAggregator:

    def __init__(self, estimator, errorbar=None, **boot_kws):
        """
        Data aggregator that produces an estimate and error bar interval.

        Parameters
        ----------
        estimator : callable or string
            Function (or method name) that maps a vector to a scalar.
        errorbar : string, (string, number) tuple, or callable
            Name of errorbar method (either "ci", "pi", "se", or "sd"), or a tuple
            with a method name and a level parameter, or a function that maps from a
            vector to a (min, max) interval, or None to hide errorbar. See the
            :doc:`errorbar tutorial </tutorial/error_bars>` for more information.
        boot_kws
            Additional keywords are passed to bootstrap when error_method is "ci".

        """
        # 初始化数据聚合器对象，并设置估计方法、误差条参数、以及引导参数
        self.estimator = estimator

        # 验证并设置误差条的计算方法和级别
        method, level = _validate_errorbar_arg(errorbar)
        self.error_method = method
        self.error_level = level

        self.boot_kws = boot_kws
    # 定义一个类方法，接收数据和变量名，并返回包含估计和误差区间的聚合结果
    def __call__(self, data, var):
        """Aggregate over `var` column of `data` with estimate and error interval."""
        # 从数据中选择特定列 `var` 的值
        vals = data[var]
        
        # 如果估计器是可调用的函数，则使用它来估计 `vals` 的值
        if callable(self.estimator):
            # 注意：我们本可以将 `vals` 传递给 `vals.agg`，但实际上不行
            # 参考：https://github.com/mwaskom/seaborn/issues/2943
            estimate = self.estimator(vals)
        else:
            # 否则，使用 `vals` 的聚合函数 `self.estimator` 来计算估计值
            estimate = vals.agg(self.estimator)

        # 如果未指定误差条的生成方法，则设定错误范围为 NaN
        if self.error_method is None:
            err_min = err_max = np.nan
        # 如果数据量小于等于1，则无法生成有意义的误差条
        elif len(data) <= 1:
            err_min = err_max = np.nan

        # 如果误差条的生成方法是可调用的函数，则使用它来计算误差范围
        elif callable(self.error_method):
            err_min, err_max = self.error_method(vals)

        # 如果误差条的生成方法是参数化方法之一
        elif self.error_method == "sd":
            # 使用标准偏差乘以误差级别来计算误差条
            half_interval = vals.std() * self.error_level
            err_min, err_max = estimate - half_interval, estimate + half_interval
        elif self.error_method == "se":
            # 使用标准误乘以误差级别来计算误差条
            half_interval = vals.sem() * self.error_level
            err_min, err_max = estimate - half_interval, estimate + half_interval

        # 如果误差条的生成方法是非参数化方法之一
        elif self.error_method == "pi":
            # 使用百分位数区间函数 `_percentile_interval` 来计算误差条
            err_min, err_max = _percentile_interval(vals, self.error_level)
        elif self.error_method == "ci":
            # 如果有单位数据，则进行自举法计算，否则直接用 `vals` 进行计算
            units = data.get("units", None)
            boots = bootstrap(vals, units=units, func=self.estimator, **self.boot_kws)
            err_min, err_max = _percentile_interval(boots, self.error_level)

        # 返回一个 Pandas Series，包含估计值、最小误差值和最大误差值
        return pd.Series({var: estimate, f"{var}min": err_min, f"{var}max": err_max})
# 定义一个加权聚合器类，用于生成加权估计值和误差条区间
class WeightedAggregator:

    def __init__(self, estimator, errorbar=None, **boot_kws):
        """
        Data aggregator that produces a weighted estimate and error bar interval.

        Parameters
        ----------
        estimator : string
            Function (or method name) that maps a vector to a scalar. Currently
            supports only "mean".
        errorbar : string or (string, number) tuple
            Name of errorbar method or a tuple with a method name and a level parameter.
            Currently the only supported method is "ci".
        boot_kws
            Additional keywords are passed to bootstrap when error_method is "ci".

        """
        # 如果估计器不是"mean"，则引发值错误
        if estimator != "mean":
            raise ValueError(f"Weighted estimator must be 'mean', not {estimator!r}.")
        self.estimator = estimator

        # 验证并设置误差条方法和级别
        method, level = _validate_errorbar_arg(errorbar)
        if method is not None and method != "ci":
            raise ValueError(f"Error bar method must be 'ci', not {method!r}.")
        self.error_method = method
        self.error_level = level

        # 保存引导参数
        self.boot_kws = boot_kws

    def __call__(self, data, var):
        """Aggregate over `var` column of `data` with estimate and error interval."""
        # 获取数据列和权重列
        vals = data[var]
        weights = data["weight"]

        # 使用权重计算平均值作为估计值
        estimate = np.average(vals, weights=weights)

        # 如果误差条方法是"ci"并且数据长度大于1
        if self.error_method == "ci" and len(data) > 1:

            # 定义误差函数用于引导采样
            def error_func(x, w):
                return np.average(x, weights=w)

            # 进行引导采样，计算误差条的最小和最大值
            boots = bootstrap(vals, weights, func=error_func, **self.boot_kws)
            err_min, err_max = _percentile_interval(boots, self.error_level)

        else:
            err_min = err_max = np.nan

        # 返回包含估计值和误差条区间的 Pandas Series 对象
        return pd.Series({var: estimate, f"{var}min": err_min, f"{var}max": err_max})
    def __init__(self, k_depth, outlier_prop, trust_alpha):
        """
        Compute percentiles of a distribution using various tail stopping rules.

        Parameters
        ----------
        k_depth: "tukey", "proportion", "trustworthy", or "full"
            Stopping rule for choosing tail percentiles to show:
            
            - tukey: Show a similar number of outliers as in a conventional boxplot.
            - proportion: Show approximately `outlier_prop` outliers.
            - trust_alpha: Use `trust_alpha` level for the most extreme tail percentile.

        outlier_prop: float
            Parameter for `k_depth="proportion"` setting the expected outlier rate.
        trust_alpha: float
            Parameter for `k_depth="trustworthy"` setting the confidence threshold.

        Notes
        -----
        Based on the proposal in this paper:
        https://vita.had.co.nz/papers/letter-value-plot.pdf

        """
        k_options = ["tukey", "proportion", "trustworthy", "full"]
        # 检查传入的 k_depth 是否为合法选项之一
        if isinstance(k_depth, str):
            _check_argument("k_depth", k_options, k_depth)
        # 如果 k_depth 不是字符串，也不是整数，抛出类型错误异常
        elif not isinstance(k_depth, int):
            err = (
                "The `k_depth` parameter must be either an integer or string "
                f"(one of {k_options}), not {k_depth!r}."
            )
            raise TypeError(err)

        # 将参数保存到对象实例中
        self.k_depth = k_depth
        self.outlier_prop = outlier_prop
        self.trust_alpha = trust_alpha

    def _compute_k(self, n):

        # 根据不同的 k_depth 方法选择深度（即要绘制的箱子数量）
        if self.k_depth == "full":
            # 将箱子扩展到覆盖整个数据集
            k = int(np.log2(n)) + 1
        elif self.k_depth == "tukey":
            # 这将导致每个尾部大约有5-8个点
            k = int(np.log2(n)) - 3
        elif self.k_depth == "proportion":
            # 根据异常值比例计算所需的箱子数量
            k = int(np.log2(n)) - int(np.log2(n * self.outlier_prop)) + 1
        elif self.k_depth == "trustworthy":
            # 计算可信区间，确定需要绘制的箱子数量
            normal_quantile_func = np.vectorize(NormalDist().inv_cdf)
            point_conf = 2 * normal_quantile_func(1 - self.trust_alpha / 2) ** 2
            k = int(np.log2(n / point_conf)) + 1
        else:
            # 允许直接指定 k 作为输入
            k = int(self.k_depth)

        return max(k, 1)
    # 定义一个特殊方法 __call__，用于对输入 x 进行评估并返回结果字典
    def __call__(self, x):
        """Evaluate the letter values."""
        # 根据输入 x 的长度计算 k 值
        k = self._compute_k(len(x))
        # 计算指数数组 exp，其中包括 k+1 到 2 的逆序排列
        exp = np.arange(k + 1, 1, -1), np.arange(2, k + 2)
        # 计算水平数 levels，由 k+1 减去指数数组的连接部分得到
        levels = k + 1 - np.concatenate([exp[0], exp[1][1:]])
        # 计算百分位数 percentiles，基于指数数组的百分比计算
        percentiles = 100 * np.concatenate([0.5 ** exp[0], 1 - 0.5 ** exp[1]])
        # 如果 k_depth 设为 "full"，则调整百分位数的边界值
        if self.k_depth == "full":
            percentiles[0] = 0
            percentiles[-1] = 100
        # 计算数据 x 的百分位数 values
        values = np.percentile(x, percentiles)
        # 找出数据 x 中的异常值 fliers，即超出 values 范围的数据
        fliers = np.asarray(x[(x < values.min()) | (x > values.max())])
        # 计算数据 x 的中位数 median
        median = np.percentile(x, 50)

        # 返回包含评估结果的字典
        return {
            "k": k,
            "levels": levels,
            "percs": percentiles,
            "values": values,
            "fliers": fliers,
            "median": median,
        }
# 返回一个给定宽度的数据百分位数区间
def _percentile_interval(data, width):
    # 计算百分位数的边界
    edge = (100 - width) / 2
    # 定义百分位数的范围
    percentiles = edge, 100 - edge
    # 使用 numpy 计算给定百分位数的数据区间
    return np.nanpercentile(data, percentiles)


# 检查 errorbar 参数的类型和值，并分配默认级别
def _validate_errorbar_arg(arg):
    # 默认级别字典
    DEFAULT_LEVELS = {
        "ci": 95,
        "pi": 95,
        "se": 1,
        "sd": 1,
    }

    # 错误信息
    usage = "`errorbar` must be a callable, string, or (string, number) tuple"

    # 如果参数为 None，则返回空
    if arg is None:
        return None, None
    # 如果参数是可调用对象，则返回该对象和空级别
    elif callable(arg):
        return arg, None
    # 如果参数是字符串，则将其作为方法，并获取默认级别
    elif isinstance(arg, str):
        method = arg
        level = DEFAULT_LEVELS.get(method, None)
    # 否则尝试解包参数，获取方法和级别
    else:
        try:
            method, level = arg
        # 捕获可能的异常并重新抛出
        except (ValueError, TypeError) as err:
            raise err.__class__(usage) from err

    # 检查方法是否有效
    _check_argument("errorbar", list(DEFAULT_LEVELS), method)
    # 如果级别不为 None 且不是数字类型，则抛出类型错误
    if level is not None and not isinstance(level, Number):
        raise TypeError(usage)

    # 返回方法和级别
    return method, level
```