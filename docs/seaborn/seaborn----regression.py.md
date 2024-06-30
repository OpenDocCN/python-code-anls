# `D:\src\scipysrc\seaborn\seaborn\regression.py`

```
"""Plotting functions for linear models (broadly construed)."""

# 导入必要的库和模块
import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 尝试导入 statsmodels 库，标记是否成功导入
try:
    import statsmodels
    assert statsmodels
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

# 导入本地的工具函数和算法模块
from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs

# 定义导出的函数列表
__all__ = ["lmplot", "regplot", "residplot"]


class _LinearPlotter:
    """Base class for plotting relational data in tidy format.

    To get anything useful done you'll have to inherit from this, but setup
    code that can be abstracted out should be put here.

    """
    def establish_variables(self, data, **kws):
        """Extract variables from data or use directly."""
        # 将数据存储到实例变量中
        self.data = data

        # 检验输入的有效性
        any_strings = any([isinstance(v, str) for v in kws.values()])
        if any_strings and data is None:
            raise ValueError("Must pass `data` if using named variables.")

        # 设置变量
        for var, val in kws.items():
            if isinstance(val, str):
                vector = data[val]
            elif isinstance(val, list):
                vector = np.asarray(val)
            else:
                vector = val
            if vector is not None and vector.shape != (1,):
                vector = np.squeeze(vector)
            if np.ndim(vector) > 1:
                err = "regplot inputs must be 1d"
                raise ValueError(err)
            setattr(self, var, vector)

    def dropna(self, *vars):
        """Remove observations with missing data."""
        # 获取变量的值
        vals = [getattr(self, var) for var in vars]
        vals = [v for v in vals if v is not None]
        # 根据缺失值情况筛选数据
        not_na = np.all(np.column_stack([pd.notnull(v) for v in vals]), axis=1)
        for var in vars:
            val = getattr(self, var)
            if val is not None:
                setattr(self, var, val[not_na])

    def plot(self, ax):
        # 抽象方法，需要子类实现具体的绘图逻辑
        raise NotImplementedError


class _RegressionPlotter(_LinearPlotter):
    """Plotter for numeric independent variables with regression model.

    This does the computations and drawing for the `regplot` function, and
    is thus also used indirectly by `lmplot`.
    """
    def __init__(self, x, y, data=None, x_estimator=None, x_bins=None,
                 x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
                 units=None, seed=None, order=1, logistic=False, lowess=False,
                 robust=False, logx=False, x_partial=None, y_partial=None,
                 truncate=False, dropna=True, x_jitter=None, y_jitter=None,
                 color=None, label=None):
        # 初始化方法，设置对象的各个属性
        self.x_estimator = x_estimator  # 设置 x 估计器
        self.ci = ci  # 置信区间的百分比
        self.x_ci = ci if x_ci == "ci" else x_ci  # 设置 x 置信区间
        self.n_boot = n_boot  # bootstrap 重抽样次数
        self.seed = seed  # 随机数种子
        self.scatter = scatter  # 是否绘制散点图
        self.fit_reg = fit_reg  # 是否拟合回归线
        self.order = order  # 多项式拟合的阶数
        self.logistic = logistic  # 是否逻辑回归
        self.lowess = lowess  # 是否使用低通滤波
        self.robust = robust  # 是否使用鲁棒回归
        self.logx = logx  # x 轴是否取对数
        self.truncate = truncate  # 是否截断数据
        self.x_jitter = x_jitter  # x 轴是否添加抖动
        self.y_jitter = y_jitter  # y 轴是否添加抖动
        self.color = color  # 绘图颜色
        self.label = label  # 图例标签

        # 检查回归选项的互斥性
        if sum((order > 1, logistic, robust, lowess, logx)) > 1:
            raise ValueError("Mutually exclusive regression options.")

        # 从参数或传递的数据框架中提取数据
        self.establish_variables(data, x=x, y=y, units=units,
                                 x_partial=x_partial, y_partial=y_partial)

        # 删除空观测
        if dropna:
            self.dropna("x", "y", "units", "x_partial", "y_partial")

        # 从数据中回归掉干扰变量
        if self.x_partial is not None:
            self.x = self.regress_out(self.x, self.x_partial)
        if self.y_partial is not None:
            self.y = self.regress_out(self.y, self.y_partial)

        # 可能对预测变量进行分箱，这意味着需要进行点估计
        if x_bins is not None:
            self.x_estimator = np.mean if x_estimator is None else x_estimator
            x_discrete, x_bins = self.bin_predictor(x_bins)
            self.x_discrete = x_discrete
        else:
            self.x_discrete = self.x

        # 如果输入数据太少，则禁用回归
        if len(self.x) <= 1:
            self.fit_reg = False

        # 保存 x 变量的范围，以备后续使用网格时需要
        if self.fit_reg:
            self.x_range = self.x.min(), self.x.max()

    @property
    def scatter_data(self):
        """每个观测点作为数据的数据。"""
        x_j = self.x_jitter
        if x_j is None:
            x = self.x
        else:
            x = self.x + np.random.uniform(-x_j, x_j, len(self.x))

        y_j = self.y_jitter
        if y_j is None:
            y = self.y
        else:
            y = self.y + np.random.uniform(-y_j, y_j, len(self.y))

        return x, y
    def estimate_data(self):
        """Data with a point estimate and CI for each discrete x value."""
        # 获取离散的 x 值和对应的 y 值
        x, y = self.x_discrete, self.y
        # 对 x 值进行排序并去重
        vals = sorted(np.unique(x))
        points, cis = [], []

        for val in vals:

            # 获取 y 变量的点估计值
            _y = y[x == val]
            # 使用给定的估计器计算估计值
            est = self.x_estimator(_y)
            points.append(est)

            # 计算该估计值的置信区间
            if self.x_ci is None:
                cis.append(None)
            else:
                units = None
                if self.x_ci == "sd":
                    # 如果置信区间类型为标准差，计算标准差
                    sd = np.std(_y)
                    _ci = est - sd, est + sd
                else:
                    # 如果置信区间类型为其它，且有指定单位，则获取对应单位
                    if self.units is not None:
                        units = self.units[x == val]
                    # 使用 bootstrap 方法生成 bootstrap 样本，并计算置信区间
                    boots = algo.bootstrap(_y,
                                           func=self.x_estimator,
                                           n_boot=self.n_boot,
                                           units=units,
                                           seed=self.seed)
                    _ci = utils.ci(boots, self.x_ci)
                cis.append(_ci)

        return vals, points, cis

    def _check_statsmodels(self):
        """Check whether statsmodels is installed if any boolean options require it."""
        # 检查是否需要 statsmodels，并且确认其是否已安装
        options = "logistic", "robust", "lowess"
        err = "`{}=True` requires statsmodels, an optional dependency, to be installed."
        for option in options:
            if getattr(self, option) and not _has_statsmodels:
                raise RuntimeError(err.format(option))
    def fit_regression(self, ax=None, x_range=None, grid=None):
        """Fit the regression model."""
        self._check_statsmodels()  # 调用内部方法检查是否加载了 statsmodels 库

        # Create the grid for the regression
        if grid is None:
            if self.truncate:
                x_min, x_max = self.x_range  # 如果设置了截断，使用预定义的 x 范围
            else:
                if ax is None:
                    x_min, x_max = x_range  # 如果没有提供轴对象，使用传入的 x 范围
                else:
                    x_min, x_max = ax.get_xlim()  # 否则，从轴对象获取当前的 x 范围
            grid = np.linspace(x_min, x_max, 100)  # 创建包含 100 个点的等间距网格
        ci = self.ci  # 将置信区间参数存储在变量 ci 中

        # Fit the regression
        if self.order > 1:
            yhat, yhat_boots = self.fit_poly(grid, self.order)  # 如果是多项式回归，调用 fit_poly 方法拟合数据
        elif self.logistic:
            from statsmodels.genmod.generalized_linear_model import GLM
            from statsmodels.genmod.families import Binomial
            yhat, yhat_boots = self.fit_statsmodels(grid, GLM,
                                                    family=Binomial())  # 如果是逻辑回归，使用 statsmodels 进行拟合
        elif self.lowess:
            ci = None
            grid, yhat = self.fit_lowess()  # 如果是 LOESS 回归，调用 fit_lowess 方法拟合数据
        elif self.robust:
            from statsmodels.robust.robust_linear_model import RLM
            yhat, yhat_boots = self.fit_statsmodels(grid, RLM)  # 如果是鲁棒回归，使用 statsmodels 进行拟合
        elif self.logx:
            yhat, yhat_boots = self.fit_logx(grid)  # 如果是对数回归，调用 fit_logx 方法拟合数据
        else:
            yhat, yhat_boots = self.fit_fast(grid)  # 默认情况下，使用快速回归拟合数据

        # Compute the confidence interval at each grid point
        if ci is None:
            err_bands = None  # 如果置信区间为 None，则错误带为空
        else:
            err_bands = utils.ci(yhat_boots, ci, axis=0)  # 否则，使用 utils 模块计算置信区间

        return grid, yhat, err_bands  # 返回网格、预测值和置信区间
    def fit_statsmodels(self, grid, model, **kwargs):
        """More general regression function using statsmodels objects."""
        # 将自变量和因变量整理成适合回归模型的格式
        X, y = np.c_[np.ones(len(self.x)), self.x], self.y
        # 将网格数据整理成适合回归模型预测的格式
        grid = np.c_[np.ones(len(grid)), grid]

        def reg_func(_x, _y):
            # 定义回归函数，并处理可能的分离异常
            err_classes = (sme.PerfectSeparationError,)
            try:
                with warnings.catch_warnings():
                    if hasattr(sme, "PerfectSeparationWarning"):
                        # 处理完全分离警告
                        warnings.simplefilter("error", sme.PerfectSeparationWarning)
                        err_classes = (*err_classes, sme.PerfectSeparationWarning)
                    # 使用给定的模型拟合数据并进行预测
                    yhat = model(_y, _x, **kwargs).fit().predict(grid)
            except err_classes:
                # 处理分离异常，返回空值
                yhat = np.empty(len(grid))
                yhat.fill(np.nan)
            return yhat

        # 调用回归函数进行预测
        yhat = reg_func(X, y)
        # 如果没有置信区间要求，则只返回预测结果
        if self.ci is None:
            return yhat, None

        # 使用自助法进行预测结果的区间估计
        yhat_boots = algo.bootstrap(X, y,
                                    func=reg_func,
                                    n_boot=self.n_boot,
                                    units=self.units,
                                    seed=self.seed)
        return yhat, yhat_boots

    def fit_lowess(self):
        """Fit a locally-weighted regression, which returns its own grid."""
        # 使用低通滤波平滑函数对数据进行回归拟合
        from statsmodels.nonparametric.smoothers_lowess import lowess
        # 对自变量和因变量进行低通滤波拟合
        grid, yhat = lowess(self.y, self.x).T
        return grid, yhat

    def fit_logx(self, grid):
        """Fit the model in log-space."""
        # 将自变量和因变量整理成适合对数空间回归模型的格式
        X, y = np.c_[np.ones(len(self.x)), self.x], self.y
        # 将网格数据整理成适合对数空间回归模型预测的格式
        grid = np.c_[np.ones(len(grid)), np.log(grid)]

        def reg_func(_x, _y):
            # 定义对数空间回归函数
            _x = np.c_[_x[:, 0], np.log(_x[:, 1])]
            return np.linalg.pinv(_x).dot(_y)

        # 使用对数空间回归函数进行预测
        yhat = grid.dot(reg_func(X, y))
        # 如果没有置信区间要求，则只返回预测结果
        if self.ci is None:
            return yhat, None

        # 使用自助法进行预测结果的区间估计
        beta_boots = algo.bootstrap(X, y,
                                    func=reg_func,
                                    n_boot=self.n_boot,
                                    units=self.units,
                                    seed=self.seed).T
        yhat_boots = grid.dot(beta_boots).T
        return yhat, yhat_boots

    def bin_predictor(self, bins):
        """Discretize a predictor by assigning value to closest bin."""
        # 将自变量转换为数组
        x = np.asarray(self.x)
        if np.isscalar(bins):
            # 如果 bins 是标量，则按百分位数生成分位数
            percentiles = np.linspace(0, 100, bins + 2)[1:-1]
            bins = np.percentile(x, percentiles)
        else:
            bins = np.ravel(bins)

        # 计算自变量到最近分位数的距离，并离散化预测值
        dist = np.abs(np.subtract.outer(x, bins))
        x_binned = bins[np.argmin(dist, axis=1)].ravel()

        return x_binned, bins
    def regress_out(self, a, b):
        """从 a 中回归掉 b，但保持 a 的原始均值。"""
        # 计算 a 的均值
        a_mean = a.mean()
        # 将 a 减去均值，得到零均值的 a
        a = a - a_mean
        # 将 b 减去均值，得到零均值的 b
        b = b - b.mean()
        # 将 b 转换为列向量
        b = np.c_[b]
        # 使用广义逆计算回归后的 a
        a_prime = a - b.dot(np.linalg.pinv(b).dot(a))
        # 将结果加上原始均值，返回形状与 a 相同的数组
        return np.asarray(a_prime + a_mean).reshape(a.shape)

    def plot(self, ax, scatter_kws, line_kws):
        """绘制完整的图形。"""
        # 将标签插入正确的关键字参数集合中
        if self.scatter:
            scatter_kws["label"] = self.label
        else:
            line_kws["label"] = self.label

        # 使用当前颜色循环状态作为默认值
        if self.color is None:
            # 创建一个空的线条对象并获取其颜色
            lines, = ax.plot([], [])
            color = lines.get_color()
            lines.remove()
        else:
            color = self.color

        # 确保颜色是十六进制，以避免 matplotlib 的奇怪行为
        color = mpl.colors.rgb2hex(mpl.colors.colorConverter.to_rgb(color))

        # 让关键字参数中的颜色覆盖整体绘图颜色
        scatter_kws.setdefault("color", color)
        line_kws.setdefault("color", color)

        # 绘制各个组成部分的图形
        if self.scatter:
            self.scatterplot(ax, scatter_kws)

        if self.fit_reg:
            self.lineplot(ax, line_kws)

        # 标记坐标轴
        if hasattr(self.x, "name"):
            ax.set_xlabel(self.x.name)
        if hasattr(self.y, "name"):
            ax.set_ylabel(self.y.name)

    def scatterplot(self, ax, kws):
        """绘制散点图。"""
        # 特别处理基于线的标记，明确设置比 seaborn 默认样式提供的更大的线宽
        line_markers = ["1", "2", "3", "4", "+", "x", "|", "_"]
        if self.x_estimator is None:
            if "marker" in kws and kws["marker"] in line_markers:
                lw = mpl.rcParams["lines.linewidth"]
            else:
                lw = mpl.rcParams["lines.markeredgewidth"]
            kws.setdefault("linewidths", lw)

            # 如果颜色没有 shape 属性或其列数小于 4，则设置默认的透明度
            if not hasattr(kws['color'], 'shape') or kws['color'].shape[1] < 4:
                kws.setdefault("alpha", .8)

            # 获取散点数据并绘制散点图
            x, y = self.scatter_data
            ax.scatter(x, y, **kws)
        else:
            # TODO 抽象化
            ci_kws = {"color": kws["color"]}
            if "alpha" in kws:
                ci_kws["alpha"] = kws["alpha"]
            ci_kws["linewidth"] = mpl.rcParams["lines.linewidth"] * 1.75
            kws.setdefault("s", 50)

            # 获取估计数据并绘制散点图或误差线
            xs, ys, cis = self.estimate_data
            if [ci for ci in cis if ci is not None]:
                for x, ci in zip(xs, cis):
                    ax.plot([x, x], ci, **ci_kws)
            ax.scatter(xs, ys, **kws)
    # 绘制线图，将结果绘制在给定的轴 `ax` 上，使用参数 `kws` 控制绘图样式
    def lineplot(self, ax, kws):
        """Draw the model."""
        # 拟合回归模型，获取网格、预测值和误差边界
        grid, yhat, err_bands = self.fit_regression(ax)
        # 提取网格的边缘值
        edges = grid[0], grid[-1]

        # 获取并设置默认的美学属性
        fill_color = kws["color"]
        lw = kws.pop("lw", mpl.rcParams["lines.linewidth"] * 1.5)  # 提取或设置线宽度
        kws.setdefault("linewidth", lw)  # 设置线的宽度为默认值

        # 绘制回归线和置信区间
        line, = ax.plot(grid, yhat, **kws)
        if not self.truncate:
            line.sticky_edges.x[:] = edges  # 防止 mpl 添加边缘空白
        if err_bands is not None:
            ax.fill_between(grid, *err_bands, facecolor=fill_color, alpha=.15)  # 填充置信区间
# 定义回归绘图函数的文档字符串字典，包含不同选项的说明
_regression_docs = dict(

    model_api=dedent("""\
    There are a number of mutually exclusive options for estimating the
    regression model. See the :ref:`tutorial <regression_tutorial>` for more
    information.\
    """),

    # 说明 regplot 和 lmplot 函数的关系及其功能区别
    regplot_vs_lmplot=dedent("""\
    The :func:`regplot` and :func:`lmplot` functions are closely related, but
    the former is an axes-level function while the latter is a figure-level
    function that combines :func:`regplot` and :class:`FacetGrid`.\
    """),

    # 关于 x_estimator 参数的说明，指定了可选参数及其作用
    x_estimator=dedent("""\
    x_estimator : callable that maps vector -> scalar, optional
        Apply this function to each unique value of ``x`` and plot the
        resulting estimate. This is useful when ``x`` is a discrete variable.
        If ``x_ci`` is given, this estimate will be bootstrapped and a
        confidence interval will be drawn.\
    """),

    # 关于 x_bins 参数的说明，指定了可选参数及其作用
    x_bins=dedent("""\
    x_bins : int or vector, optional
        Bin the ``x`` variable into discrete bins and then estimate the central
        tendency and a confidence interval. This binning only influences how
        the scatterplot is drawn; the regression is still fit to the original
        data.  This parameter is interpreted either as the number of
        evenly-sized (not necessary spaced) bins or the positions of the bin
        centers. When this parameter is used, it implies that the default of
        ``x_estimator`` is ``numpy.mean``.\
    """),

    # 关于 x_ci 参数的说明，指定了可选参数及其作用
    x_ci=dedent("""\
    x_ci : "ci", "sd", int in [0, 100] or None, optional
        Size of the confidence interval used when plotting a central tendency
        for discrete values of ``x``. If ``"ci"``, defer to the value of the
        ``ci`` parameter. If ``"sd"``, skip bootstrapping and show the
        standard deviation of the observations in each bin.\
    """),

    # 关于 scatter 参数的说明，指定了可选参数及其作用
    scatter=dedent("""\
    scatter : bool, optional
        If ``True``, draw a scatterplot with the underlying observations (or
        the ``x_estimator`` values).\
    """),

    # 关于 fit_reg 参数的说明，指定了可选参数及其作用
    fit_reg=dedent("""\
    fit_reg : bool, optional
        If ``True``, estimate and plot a regression model relating the ``x``
        and ``y`` variables.\
    """),

    # 关于 ci 参数的说明，指定了可选参数及其作用
    ci=dedent("""\
    ci : int in [0, 100] or None, optional
        Size of the confidence interval for the regression estimate. This will
        be drawn using translucent bands around the regression line. The
        confidence interval is estimated using a bootstrap; for large
        datasets, it may be advisable to avoid that computation by setting
        this parameter to None.\
    """),

    # 关于 n_boot 参数的说明，指定了可选参数及其作用
    n_boot=dedent("""\
    n_boot : int, optional
        Number of bootstrap resamples used to estimate the ``ci``. The default
        value attempts to balance time and stability; you may want to increase
        this value for "final" versions of plots.\
    """),

    # 关于 units 参数的说明，指定了可选参数及其作用
    units=dedent("""\
    units : variable name in ``data``, optional
        # 定义变量名，用于指定数据中的抽样单元，如果观测数据 ``x`` 和 ``y`` 是在这些抽样单元内部嵌套的。
        # 在计算置信区间时会使用多层次自举法，对抽样单元和观测数据（在单元内部）进行重采样。
        # 这不会影响如何估计或绘制回归线。
    """),
    seed=dedent("""\
    seed : int, numpy.random.Generator, or numpy.random.RandomState, optional
        # 种子或用于可重复自举抽样的随机数生成器，可以是整数、numpy.random.Generator 或 numpy.random.RandomState 类型。
    """),
    order=dedent("""\
    order : int, optional
        # 如果 ``order`` 大于1，则使用 ``numpy.polyfit`` 来估计多项式回归。
    """),
    logistic=dedent("""\
    logistic : bool, optional
        # 如果为 ``True``，假设 ``y`` 是二进制变量，并使用 ``statsmodels`` 来估计 logistic 回归模型。
        # 注意，这比线性回归计算量大得多，因此可能需要减少自举重采样次数（``n_boot``），或将 ``ci`` 设为 None。
    """),
    lowess=dedent("""\
    lowess : bool, optional
        # 如果为 ``True``，使用 ``statsmodels`` 来估计非参数的 lowess 模型（局部加权线性回归）。
        # 注意，目前无法为这种模型绘制置信区间。
    """),
    robust=dedent("""\
    robust : bool, optional
        # 如果为 ``True``，使用 ``statsmodels`` 来估计鲁棒回归。这会减少异常值的权重。
        # 注意，这比标准线性回归计算量大得多，因此可能需要减少自举重采样次数（``n_boot``），或将 ``ci`` 设为 None。
    """),
    logx=dedent("""\
    logx : bool, optional
        # 如果为 ``True``，估计形式为 y ~ log(x) 的线性回归，但在输入空间中绘制散点图和回归模型。
        # 注意，``x`` 必须为正数才能工作。
    """),
    xy_partial=dedent("""\
    {x,y}_partial : strings in ``data`` or matrices
        # 在绘图之前要从 ``x`` 或 ``y`` 变量中回归出的混杂变量。
    """),
    truncate=dedent("""\
    truncate : bool, optional
        # 如果为 ``True``，回归线将被数据限制边界截断。如果为 ``False``，它将延伸到 ``x`` 轴限制。
    """),
    dropna=dedent("""\
    dropna : bool, optional
        # 如果为 ``True``，从绘图中删除具有缺失数据的观测。
    """),
    xy_jitter=dedent("""\
    {x,y}_jitter : floats, optional
        # 可选参数 {x,y}_jitter：浮点数
        在“x”或“y”变量上添加均匀分布的随机噪声。这些噪声添加到拟合回归之后的数据副本上，仅影响散点图的外观。
        这在绘制取离散值的变量时很有帮助。

    scatter_line_kws=dedent("""\
    {scatter,line}_kws : dictionaries
        # scatter_line_kws：参数为字典
        传递给“plt.scatter”和“plt.plot”的额外关键字参数。
    """),
# 更新回归图绘制的文档说明字典，将_facet_docs内容添加到_regression_docs中
_regression_docs.update(_facet_docs)


def lmplot(
    data, *,
    x=None, y=None, hue=None, col=None, row=None,
    palette=None, col_wrap=None, height=5, aspect=1, markers="o",
    sharex=None, sharey=None, hue_order=None, col_order=None, row_order=None,
    legend=True, legend_out=None, x_estimator=None, x_bins=None,
    x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
    units=None, seed=None, order=1, logistic=False, lowess=False,
    robust=False, logx=False, x_partial=None, y_partial=None,
    truncate=True, x_jitter=None, y_jitter=None, scatter_kws=None,
    line_kws=None, facet_kws=None,
):
    # 如果facet_kws为None，则初始化为一个空字典
    if facet_kws is None:
        facet_kws = {}

    # 定义用于废弃facet_kws参数的函数
    def facet_kw_deprecation(key, val):
        msg = (
            f"{key} is deprecated from the `lmplot` function signature. "
            "Please update your code to pass it using `facet_kws`."
        )
        # 如果val不为None，则发出警告并将其添加到facet_kws中
        if val is not None:
            warnings.warn(msg, UserWarning)
            facet_kws[key] = val

    # 检查data是否为None，如果是则引发TypeError异常
    if data is None:
        raise TypeError("Missing required keyword argument `data`.")

    # 筛选出数据框中需要的列，并更新data为包含这些列的数据
    need_cols = [x, y, hue, col, row, units, x_partial, y_partial]
    cols = np.unique([a for a in need_cols if a is not None]).tolist()
    data = data[cols]

    # 初始化FacetGrid对象
    facets = FacetGrid(
        data, row=row, col=col, hue=hue,
        palette=palette,
        row_order=row_order, col_order=col_order, hue_order=hue_order,
        height=height, aspect=aspect, col_wrap=col_wrap,
        **facet_kws,
    )

    # 在这里添加标记，因为FacetGrid已经确定了hue变量的级别数
    # 我们不希望重复该过程
    if facets.hue_names is None:
        n_markers = 1
    else:
        n_markers = len(facets.hue_names)
    # 如果markers不是列表，则将其重复n_markers次组成列表
    if not isinstance(markers, list):
        markers = [markers] * n_markers
    # 如果markers列表长度与hue变量的级别数不匹配，则引发ValueError异常
    if len(markers) != n_markers:
        raise ValueError("markers must be a singleton or a list of markers "
                         "for each level of the hue variable")
    # 设置hue_kws字典中的'marker'键对应的值为markers列表
    facets.hue_kws = {"marker": markers}

    # 定义更新数据极限的函数，用于每个facet中的数据极限更新
    def update_datalim(data, x, y, ax, **kws):
        xys = data[[x, y]].to_numpy().astype(float)
        ax.update_datalim(xys, updatey=False)
        ax.autoscale_view(scaley=False)

    # 对每个facet应用map_dataframe方法，更新数据极限
    facets.map_dataframe(update_datalim, x=x, y=y)

    # 绘制每个facet上的回归图
    regplot_kws = dict(
        x_estimator=x_estimator, x_bins=x_bins, x_ci=x_ci,
        scatter=scatter, fit_reg=fit_reg, ci=ci, n_boot=n_boot, units=units,
        seed=seed, order=order, logistic=logistic, lowess=lowess,
        robust=robust, logx=logx, x_partial=x_partial, y_partial=y_partial,
        truncate=truncate, x_jitter=x_jitter, y_jitter=y_jitter,
        scatter_kws=scatter_kws, line_kws=line_kws,
    )
    # 对数据框中的每个子图进行绘制，使用 regplot 函数绘制 x 和 y 的关系图，根据 regplot_kws 参数设置样式
    facets.map_dataframe(regplot, x=x, y=y, **regplot_kws)
    # 设置子图的坐标轴标签为 x 和 y
    facets.set_axis_labels(x, y)

    # 添加图例（如果需要且数据中有分类变量 hue，且 hue 不是列名或行名之一）
    if legend and (hue is not None) and (hue not in [col, row]):
        facets.add_legend()
    # 返回已创建的子图对象
    return facets
lmplot.__doc__ = dedent("""\
    # 设置 lmplot 的文档字符串，用于描述其功能和使用方法

    Plot data and regression model fits across a FacetGrid.

    This function combines :func:`regplot` and :class:`FacetGrid`. It is
    intended as a convenient interface to fit regression models across
    conditional subsets of a dataset.

    When thinking about how to assign variables to different facets, a general
    rule is that it makes sense to use ``hue`` for the most important
    comparison, followed by ``col`` and ``row``. However, always think about
    your particular dataset and the goals of the visualization you are
    creating.

    {model_api}  # 描述模型 API 的文档占位符

    The parameters to this function span most of the options in
    :class:`FacetGrid`, although there may be occasional cases where you will
    want to use that class and :func:`regplot` directly.

    Parameters
    ----------
    {data}  # 描述数据参数的文档占位符
    x, y : strings, optional
        Input variables; these should be column names in ``data``.
    hue, col, row : strings
        Variables that define subsets of the data, which will be drawn on
        separate facets in the grid. See the ``*_order`` parameters to control
        the order of levels of this variable.
    {palette}  # 描述调色板参数的文档占位符
    {col_wrap}  # 描述列包装参数的文档占位符
    {height}  # 描述高度参数的文档占位符
    {aspect}  # 描述宽高比参数的文档占位符
    markers : matplotlib marker code or list of marker codes, optional
        Markers for the scatterplot. If a list, each marker in the list will be
        used for each level of the ``hue`` variable.
    {share_xy}  # 描述共享坐标轴参数的文档占位符

        .. deprecated:: 0.12.0
            Pass using the `facet_kws` dictionary.

    {{hue,col,row}}_order : lists, optional
        Order for the levels of the faceting variables. By default, this will
        be the order that the levels appear in ``data`` or, if the variables
        are pandas categoricals, the category order.
    legend : bool, optional
        If ``True`` and there is a ``hue`` variable, add a legend.
    {legend_out}  # 描述图例位置参数的文档占位符

        .. deprecated:: 0.12.0
            Pass using the `facet_kws` dictionary.

    {x_estimator}  # 描述 x 估计器参数的文档占位符
    {x_bins}  # 描述 x 分箱参数的文档占位符
    {x_ci}  # 描述 x 置信区间参数的文档占位符
    {scatter}  # 描述散点图参数的文档占位符
    {fit_reg}  # 描述拟合回归线参数的文档占位符
    {ci}  # 描述置信区间参数的文档占位符
    {n_boot}  # 描述 bootstrap 参数的文档占位符
    {units}  # 描述单元参数的文档占位符
    {seed}  # 描述随机种子参数的文档占位符
    {order}  # 描述排序参数的文档占位符
    {logistic}  # 描述 logistic 参数的文档占位符
    {lowess}  # 描述 lowess 参数的文档占位符
    {robust}  # 描述 robust 参数的文档占位符
    {logx}  # 描述对数 x 参数的文档占位符
    {xy_partial}  # 描述部分 xy 参数的文档占位符
    {truncate}  # 描述截断参数的文档占位符
    {xy_jitter}  # 描述 xy 抖动参数的文档占位符
    {scatter_line_kws}  # 描述散点线参数的文档占位符
    facet_kws : dict
        Dictionary of keyword arguments for :class:`FacetGrid`.

    Returns
    -------
    :class:`FacetGrid`
        The :class:`FacetGrid` object with the plot on it for further tweaking.

    See Also
    --------
    regplot : Plot data and a conditional model fit.
    FacetGrid : Subplot grid for plotting conditional relationships.
    pairplot : Combine :func:`regplot` and :class:`PairGrid` (when used with
               ``kind="reg"``).

    Notes
    -----

    {regplot_vs_lmplot}  # 描述 regplot 与 lmplot 区别的文档占位符

    Examples
    --------

    .. include:: ../docstrings/lmplot.rst

    """).format(**_regression_docs)
    scatter=True,  # 是否显示散点图，默认为True
    fit_reg=True,  # 是否显示拟合的回归线，默认为True
    ci=95,  # 置信区间的大小，默认为95（表示95%置信区间）
    n_boot=1000,  # 使用bootstrap方法计算置信区间时的重抽样次数，默认为1000
    units=None,  # 数据分组的变量，用于拟合线性模型的每个独立单元，默认为None
    seed=None,  # 随机数种子，用于控制bootstrap的随机性，默认为None
    order=1,  # 多项式拟合的阶数，默认为1（线性拟合）
    logistic=False,  # 是否使用逻辑回归模型进行拟合，默认为False
    lowess=False,  # 是否使用LOESS拟合（局部加权线性回归），默认为False
    robust=False,  # 是否使用稳健回归，默认为False
    logx=False,  # 是否对x轴取对数，默认为False
    x_partial=None,  # x变量的偏导数，用于条件效应图，默认为None
    y_partial=None,  # y变量的偏导数，用于条件效应图，默认为None
    truncate=True,  # 是否截断回归线以适应坐标轴的边界，默认为True
    dropna=True,  # 是否删除包含缺失值的行，默认为True
    x_jitter=None,  # x轴数据抖动（添加噪声）的幅度，默认为None
    y_jitter=None,  # y轴数据抖动（添加噪声）的幅度，默认为None
    label=None,  # 图例标签的名称，默认为None
    color=None,  # 散点的颜色，默认为None（使用默认颜色）
    marker="o",  # 散点的标记样式，默认为圆圈
    scatter_kws=None,  # 用于传递给散点图绘制函数的其他关键字参数，默认为None
    line_kws=None,  # 用于传递给拟合线绘制函数的其他关键字参数，默认为None
    ax=None  # 绘图所使用的坐标轴对象，默认为None（使用当前活动的坐标轴）
# 绘制线性回归模型的残差图
def residplot(
    data=None, *, x=None, y=None,  # 定义函数参数：数据集、自变量、因变量
    x_partial=None, y_partial=None,  # 部分自变量和因变量
    lowess=False,  # 是否使用局部加权回归平滑（Loess）
    order=1,  # 多项式拟合的阶数
    robust=False,  # 是否使用鲁棒回归
    dropna=True,  # 是否丢弃缺失值
    label=None,  # 图例标签
    color=None,  # 绘图颜色
    scatter_kws=None,  # 散点图参数
    line_kws=None,  # 线性拟合参数
    ax=None  # Matplotlib 的 Axes 对象，可选
):
    """Plot the residuals of a linear regression.

    Parameters
    ----------
    data : DataFrame
        输入数据集
    x, y : string, series, or vector array
        自变量和因变量，可以是字符串、序列或向量数组
    x_partial, y_partial : string, series, or vector array, optional
        部分自变量和因变量
    lowess : bool, optional
        是否使用局部加权回归平滑（Loess）
    order : int, optional
        多项式拟合的阶数
    robust : bool, optional
        是否使用鲁棒回归
    dropna : bool, optional
        是否丢弃缺失值
    label : string, optional
        图例标签，适用于散点图或回归线（如果 ``scatter`` 为 ``False``）
    color : matplotlib 颜色，optional
        应用于所有图形元素的颜色；将被传递给 ``scatter_kws`` 或 ``line_kws`` 的颜色所覆盖
    scatter_kws : dict, optional
        散点图参数
    line_kws : dict, optional
        线性拟合参数
    ax : matplotlib Axes, optional
        绘制图形的 Axes 对象；否则使用当前 Axes。

    Returns
    -------
    ax : matplotlib Axes
        包含绘图的 Axes 对象

    See Also
    --------
    regplot : 绘制数据和线性回归模型拟合线

    Notes
    -----
    可以通过 :func:`jointplot` 或 :func:`pairplot` 函数将 :func:`regplot` 和 :class:`JointGrid` 或 :class:`PairGrid` 结合起来，
    不过这些函数并不直接接受 :func:`regplot` 的所有参数。

    Examples
    --------
    .. include:: ../docstrings/regplot.rst

    """
    """
    This function performs a regression of y on x, possibly using robust or polynomial methods,
    and then visualizes the residuals using a scatterplot. Optionally, a lowess smoother can
    be fitted to the residuals plot to detect any underlying structure.
    
    Parameters
    ----------
    data : DataFrame, optional
        DataFrame containing the dataset if `x` and `y` are column names.
    x : vector or string
        Predictor variable data or column name in `data`.
    y : vector or string
        Response variable data or column name in `data`.
    {x, y}_partial : vectors or string(s), optional
        Variables treated as confounding, to be removed from `x` or `y` before plotting.
    lowess : boolean, optional
        Flag indicating whether to fit a lowess smoother to the residual scatterplot.
    order : int, optional
        Degree of polynomial for polynomial regression when calculating residuals.
    robust : boolean, optional
        Flag indicating whether to use robust linear regression when calculating residuals.
    dropna : boolean, optional
        If True, ignore observations with missing data when fitting and plotting.
    label : string, optional
        Label used in plot legends.
    color : matplotlib color, optional
        Color for all elements of the plot.
    {scatter, line}_kws : dictionaries, optional
        Additional keyword arguments passed to scatter() and plot() functions for drawing components.
    ax : matplotlib axis, optional
        Axis to plot into; if None, uses current axis or creates a new one.
    
    Returns
    -------
    ax: matplotlib axes
        Axes with the regression plot.
    
    See Also
    --------
    regplot : Plot a simple linear regression model.
    jointplot : Draw a `residplot` with univariate marginal distributions (when used with `kind="resid"`).
    
    Examples
    --------
    
    .. include:: ../docstrings/residplot.rst
    """
    
    plotter = _RegressionPlotter(x, y, data, ci=None,
                                 order=order, robust=robust,
                                 x_partial=x_partial, y_partial=y_partial,
                                 dropna=dropna, color=color, label=label)
    
    if ax is None:
        ax = plt.gca()
    
    # Calculate the residuals from a linear regression
    _, yhat, _ = plotter.fit_regression(grid=plotter.x)
    plotter.y = plotter.y - yhat
    
    # Set the plotter's regression and lowess options
    if lowess:
        plotter.lowess = True
    else:
        plotter.fit_reg = False
    
    # Plot a horizontal line at y=0 to mark residuals center
    ax.axhline(0, ls=":", c=".2")
    
    # Draw the scatterplot of residuals
    scatter_kws = {} if scatter_kws is None else scatter_kws.copy()
    line_kws = {} if line_kws is None else line_kws.copy()
    plotter.plot(ax, scatter_kws, line_kws)
    
    return ax
```