# `D:\src\scipysrc\scipy\scipy\stats\_fit.py`

```
import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state

def _combine_bounds(name, user_bounds, shape_domain, integral):
    """Intersection of user-defined bounds and distribution PDF/PMF domain"""

    # 将用户定义的边界至少转换为一维数组
    user_bounds = np.atleast_1d(user_bounds)

    # 检查用户定义的边界是否有效，若下界大于上界则引发异常
    if user_bounds[0] > user_bounds[1]:
        message = (f"There are no values for `{name}` on the interval "
                   f"{list(user_bounds)}.")
        raise ValueError(message)

    # 计算用户定义边界和分布域的交集
    bounds = (max(user_bounds[0], shape_domain[0]),
              min(user_bounds[1], shape_domain[1]))

    # 若需要整数值并且边界上界小于下界，则引发异常
    if integral and (np.ceil(bounds[0]) > np.floor(bounds[1])):
        message = (f"There are no integer values for `{name}` on the interval "
                   f"defined by the user-provided bounds and the domain "
                   "of the distribution.")
        raise ValueError(message)
    # 若不需要整数值并且边界上界小于下界，则引发异常
    elif not integral and (bounds[0] > bounds[1]):
        message = (f"There are no values for `{name}` on the interval "
                   f"defined by the user-provided bounds and the domain "
                   "of the distribution.")
        raise ValueError(message)

    # 若边界不是有限的，则引发异常
    if not np.all(np.isfinite(bounds)):
        message = (f"The intersection of user-provided bounds for `{name}` "
                   f"and the domain of the distribution is not finite. Please "
                   f"provide finite bounds for shape `{name}` in `bounds`.")
        raise ValueError(message)

    # 返回计算得到的边界
    return bounds


class FitResult:
    r"""Result of fitting a discrete or continuous distribution to data

    Attributes
    ----------
    params : namedtuple
        A namedtuple containing the maximum likelihood estimates of the
        shape parameters, location, and (if applicable) scale of the
        distribution.
    success : bool or None
        Whether the optimizer considered the optimization to terminate
        successfully or not.
    message : str or None
        Any status message provided by the optimizer.

    """

    def __init__(self, dist, data, discrete, res):
        # 初始化 FitResult 对象，保存分布、数据、是否离散、优化结果等属性
        self._dist = dist
        self._data = data
        self.discrete = discrete
        # 设置概率质量函数或概率密度函数的引用
        self.pxf = getattr(dist, "pmf", None) or getattr(dist, "pdf", None)

        # 根据分布是否有形状参数，创建对应的命名元组 FitParams
        shape_names = [] if dist.shapes is None else dist.shapes.split(", ")
        if not discrete:
            FitParams = namedtuple('FitParams', shape_names + ['loc', 'scale'])
        else:
            FitParams = namedtuple('FitParams', shape_names + ['loc'])

        # 将优化得到的参数作为命名元组 FitParams 的实例保存在 params 属性中
        self.params = FitParams(*res.x)

        # 如果优化成功但负对数似然函数为无穷大，则将优化结果标记为失败
        if res.success and not np.isfinite(self.nllf()):
            res.success = False
            res.message = ("Optimization converged to parameter values that "
                           "are inconsistent with the data.")
        self.success = getattr(res, "success", None)
        self.message = getattr(res, "message", None)
    def __repr__(self):
        keys = ["params", "success", "message"]
        # 计算最长的键的长度，并加1，用于美化输出格式
        m = max(map(len, keys)) + 1
        # 使用列表推导式生成每个键值对应的格式化字符串，并拼接成最终的输出字符串
        return '\n'.join([key.rjust(m) + ': ' + repr(getattr(self, key))
                          for key in keys if getattr(self, key) is not None])

    def nllf(self, params=None, data=None):
        """Negative log-likelihood function

        Evaluates the negative of the log-likelihood function of the provided
        data at the provided parameters.

        Parameters
        ----------
        params : tuple, optional
            The shape parameters, location, and (if applicable) scale of the
            distribution as a single tuple. Default is the maximum likelihood
            estimates (``self.params``).
        data : array_like, optional
            The data for which the log-likelihood function is to be evaluated.
            Default is the data to which the distribution was fit.

        Returns
        -------
        nllf : float
            The negative of the log-likelihood function.

        """
        # 如果参数 params 和 data 未提供，则使用对象的默认参数 self.params 和 self._data
        params = params if params is not None else self.params
        data = data if data is not None else self._data
        # 调用分布对象的 nnlf 方法计算负对数似然函数的值，并返回结果
        return self._dist.nnlf(theta=params, x=data)

    def _hist_plot(self, ax, fit_params):
        from matplotlib.ticker import MaxNLocator

        # 获取分布的支持范围，用于绘图的边界
        support = self._dist.support(*fit_params)
        lb = support[0] if np.isfinite(support[0]) else min(self._data)
        ub = support[1] if np.isfinite(support[1]) else max(self._data)
        pxf = "PMF" if self.discrete else "PDF"

        # 根据数据分布是否离散选择不同的 x 值和绘图方式
        if self.discrete:
            x = np.arange(lb, ub + 2)
            y = self.pxf(x, *fit_params)
            # 绘制离散分布的 PMF 曲线和直方图
            ax.vlines(x[:-1], 0, y[:-1], label='Fitted Distribution PMF',
                      color='C0')
            options = dict(density=True, bins=x, align='left', color='C1')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('k')
            ax.set_ylabel('PMF')
        else:
            x = np.linspace(lb, ub, 200)
            y = self.pxf(x, *fit_params)
            # 绘制连续分布的 PDF 曲线和直方图
            ax.plot(x, y, '--', label='Fitted Distribution PDF', color='C0')
            options = dict(density=True, bins=50, align='mid', color='C1')
            ax.set_xlabel('x')
            ax.set_ylabel('PDF')

        # 根据数据量的大小决定使用直方图或者散点图展示数据
        if len(self._data) > 50 or self.discrete:
            ax.hist(self._data, label="Histogram of Data", **options)
        else:
            ax.plot(self._data, np.zeros_like(self._data), "*",
                    label='Data', color='C1')

        # 设置图表标题，并显示图例
        ax.set_title(rf"Fitted $\tt {self._dist.name}$ {pxf} and Histogram")
        ax.legend(*ax.get_legend_handles_labels())
        return ax
    def _qp_plot(self, ax, fit_params, qq):
        # 对数据进行排序
        data = np.sort(self._data)
        # 计算绘图位置参数
        ps = self._plotting_positions(len(self._data))

        if qq:
            qp = "Quantiles"
            plot_type = 'Q-Q'
            # 计算理论分位数对应的横坐标
            x = self._dist.ppf(ps, *fit_params)
            y = data
        else:
            qp = "Percentiles"
            plot_type = 'P-P'
            # 计算累积分布函数值对应的横坐标
            x = ps
            y = self._dist.cdf(data, *fit_params)

        # 绘制散点图或者步阶图
        ax.plot(x, y, '.', label=f'Fitted Distribution {plot_type}',
                color='C0', zorder=1)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        if not qq:
            # 确保 y 轴范围在 [0, 1] 之间
            lim = max(lim[0], 0), min(lim[1], 1)

        if self.discrete and qq:
            # 若数据为离散型且为 Q-Q 图，则绘制参考线
            q_min, q_max = int(lim[0]), int(lim[1]+1)
            q_ideal = np.arange(q_min, q_max)
            ax.plot(q_ideal, q_ideal, 'o', label='Reference', color='k',
                    alpha=0.25, markerfacecolor='none', clip_on=True)
        elif self.discrete and not qq:
            # 若数据为离散型且为 P-P 图，则绘制参考步阶图
            p_min, p_max = lim
            a, b = self._dist.support(*fit_params)
            p_min = max(p_min, 0 if np.isfinite(a) else 1e-3)
            p_max = min(p_max, 1 if np.isfinite(b) else 1-1e-3)
            q_min, q_max = self._dist.ppf([p_min, p_max], *fit_params)
            qs = np.arange(q_min-1, q_max+1)
            ps = self._dist.cdf(qs, *fit_params)
            ax.step(ps, ps, '-', label='Reference', color='k', alpha=0.25,
                    clip_on=True)
        else:
            # 若数据为连续型，则绘制参考线
            ax.plot(lim, lim, '-', label='Reference', color='k', alpha=0.25,
                    clip_on=True)

        # 设置 x 和 y 轴的范围
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        # 设置 x 和 y 轴标签
        ax.set_xlabel(rf"Fitted $\tt {self._dist.name}$ Theoretical {qp}")
        ax.set_ylabel(f"Data {qp}")
        # 设置图表标题
        ax.set_title(rf"Fitted $\tt {self._dist.name}$ {plot_type} Plot")
        # 添加图例
        ax.legend(*ax.get_legend_handles_labels())
        # 设置图表纵横比为等比例
        ax.set_aspect('equal')
        return ax

    def _qq_plot(self, **kwargs):
        # 绘制 Q-Q 图
        return self._qp_plot(qq=True, **kwargs)

    def _pp_plot(self, **kwargs):
        # 绘制 P-P 图
        return self._qp_plot(qq=False, **kwargs)

    def _plotting_positions(self, n, a=.5):
        # 计算绘图位置参数，参考 https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Plotting_positions
        k = np.arange(1, n+1)
        return (k-a) / (n + 1 - 2*a)
    # 定义一个私有方法，用于绘制累积分布函数图
    def _cdf_plot(self, ax, fit_params):
        # 对数据进行排序，得到有序数据
        data = np.sort(self._data)
        # 使用绘图位置函数计算经验累积分布函数
        ecdf = self._plotting_positions(len(self._data))
        # 如果唯一值数量少于30个，则使用虚线，否则使用点线
        ls = '--' if len(np.unique(data)) < 30 else '.'
        # 如果是离散分布，则设置 x 轴标签为 'k'，否则为 'x'
        xlabel = 'k' if self.discrete else 'x'
        # 绘制经验累积分布函数的步阶图
        ax.step(data, ecdf, ls, label='Empirical CDF', color='C1', zorder=0)

        # 获取当前坐标轴的 x 范围
        xlim = ax.get_xlim()
        # 在 x 范围内生成300个均匀分布的点作为查询点
        q = np.linspace(*xlim, 300)
        # 使用拟合参数计算拟合分布的累积分布函数
        tcdf = self._dist.cdf(q, *fit_params)

        # 绘制拟合分布的累积分布函数曲线
        ax.plot(q, tcdf, label='Fitted Distribution CDF', color='C0', zorder=1)
        # 设置坐标轴的 x 范围为之前获取的范围
        ax.set_xlim(xlim)
        # 设置坐标轴的 y 范围为 [0, 1]
        ax.set_ylim(0, 1)
        # 设置 x 轴标签为之前定义的 xlabel
        ax.set_xlabel(xlabel)
        # 设置 y 轴标签为 "CDF"
        ax.set_ylabel("CDF")
        # 设置图表标题，包含拟合分布的名称
        ax.set_title(rf"Fitted $\tt {self._dist.name}$ and Empirical CDF")
        # 获取图例中的句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 倒序排列图例中的句柄和标签
        ax.legend(handles[::-1], labels[::-1])
        # 返回绘制后的坐标轴对象
        return ax
# 定义一个函数 fit，用于拟合离散或连续分布到数据
def fit(dist, data, bounds=None, *, guess=None, method='mle',
        optimizer=optimize.differential_evolution):
    r"""Fit a discrete or continuous distribution to data

    Given a distribution, data, and bounds on the parameters of the
    distribution, return maximum likelihood estimates of the parameters.

    Parameters
    ----------
    dist : `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete`
        表示要拟合到数据的分布对象。
    data : 1D array_like
        要拟合的数据。如果数据中包含 np.nan、np.inf 或 -np.inf，则拟合方法会引发 ValueError。
    bounds : dict or sequence of tuples, optional
        如果是字典，则每个键是分布的参数名，对应的值是包含该参数的下限和上限的元组。
        如果分布仅对参数的有限范围有定义，则不需要该参数的条目；例如，某些分布的参数必须在区间 [0, 1] 上。位置（loc）和尺度（scale）的边界是可选的；默认分别固定为 0 和 1。
        
        如果是序列，则第 i 个元素是包含第 i 个分布参数的下限和上限的元组。在这种情况下，必须提供所有分布形状参数的边界。还可以选择在分布形状参数后提供位置和尺度的边界。
        
        如果要保持形状固定（例如已知），则下限和上限可以相等。如果用户提供的下限或上限超出了分布定义的域的边界，则分布域的边界将替换用户提供的值。类似地，必须是整数的参数将在用户提供的边界内被约束为整数值。
    guess : dict or array_like, optional
        如果是字典，则每个键是分布的参数名，对应的值是参数值的猜测。
        
        如果是序列，则第 i 个元素是第 i 个分布参数的猜测。在这种情况下，必须提供所有分布形状参数的猜测。
        
        如果未提供 guess，则不会将决策变量的猜测传递给优化器。如果提供了 guess，则会将任何缺失的参数猜测设置为下限和上限的平均值。必须是整数的参数的猜测将四舍五入为整数值，并且位于用户提供的边界和分布域的交集之外的猜测将被修剪。
    method : str, optional
        用于拟合分布的方法，默认为 'mle'（最大似然估计）。
    optimizer : function, optional
        用于优化的函数，默认为 optimize.differential_evolution。

    Returns
    -------
    `scipy.optimize.OptimizeResult`
        一个包含最优化结果的对象，包括拟合分布的参数估计和优化器的其他信息。

    Raises
    ------
    ValueError
        如果数据中包含 np.nan、np.inf 或 -np.inf。

    Notes
    -----
    This function uses the maximum likelihood estimation method to fit a
    distribution to data. It can handle both discrete and continuous
    distributions represented by `scipy.stats.rv_discrete` or
    `scipy.stats.rv_continuous` objects.
    """
    method : {'mle', 'mse'}
        # 参数method可以是'mle'或'mse'。当method="mle"（默认）时，通过最小化负对数似然函数来进行拟合。
        # 对于分布支持范围之外的观测值，会施加一个大的有限惩罚（而不是无限负对数似然）。
        # 当method="mse"时，通过最小化负对数乘积间距函数来进行拟合。对于支持范围之外的观测值，会施加相同的惩罚。
        # 我们遵循文献[1]中的方法，该方法适用于具有重复观测样本的情况。

    optimizer : callable, optional
        # 参数optimizer是一个可调用对象，接受以下位置参数：

        fun : callable
            # 要优化的目标函数。`fun`接受一个参数`x`，即分布的候选形状参数，并返回给定`x`、`dist`和提供的`data`的目标函数值。
            # `optimizer`的任务是找到最小化`fun`的决策变量值。

        # `optimizer`还必须接受以下关键字参数：

        bounds : sequence of tuples
            # 决策变量值的范围限制；每个元素是一个包含决策变量下界和上界的元组。

        如果提供了`guess`，`optimizer`还必须接受以下关键字参数：

        x0 : array_like
            # 每个决策变量的初始猜测值。

        如果分布具有必须是整数的形状参数，或者分布是离散的且位置参数不固定，`optimizer`还必须接受以下关键字参数：

        integrality : array_like of bools
            # 每个决策变量的布尔数组。如果决策变量必须被限制为整数值，则为True；如果决策变量是连续的，则为False。

        # `optimizer`必须返回一个对象，例如`scipy.optimize.OptimizeResult`的实例，该对象在属性`x`中保存决策变量的最优值。
        # 如果提供了`fun`、`status`或`message`属性，则它们将包含在`fit`返回的结果对象中。
    # result 是一个 `~scipy.stats._result_classes.FitResult` 对象，包含以下字段和方法。

    params : namedtuple
        # params 是一个命名元组，包含分布的最大似然估计的形状参数、位置参数，以及（如果适用）尺度参数。

    success : bool or None
        # success 表示优化器是否认为优化成功终止。

    message : str or None
        # message 是优化器提供的任何状态消息。

    The object has the following method:

    nllf(params=None, data=None)
        # 默认情况下，对给定数据的拟合参数 `params` 的负对数似然函数。
        # 接受一个包含替代形状、位置和尺度的元组，以及一个替代数据的数组。

    plot(ax=None)
        # 在数据的归一化直方图上叠加拟合分布的概率密度函数或概率质量函数。

See Also
--------
rv_continuous,  rv_discrete

Notes
-----
# 当用户提供包含最大似然估计的紧密边界时，优化更可能收敛于最大似然估计。
# 例如，在将二项分布拟合到数据时，每个样本背后的实验次数可能已知，此时对应的形状参数 ``n`` 可以固定。

References
----------
.. [1] Shao, Yongzhao, and Marjorie G. Hahn. "Maximum product of spacings
       method: a unified formulation with illustration of strong
       consistency." Illinois Journal of Mathematics 43.3 (1999): 489-499.

Examples
--------
# 假设我们希望将分布拟合到以下数据。

>>> import numpy as np
>>> from scipy import stats
>>> rng = np.random.default_rng()
>>> dist = stats.nbinom
>>> shapes = (5, 0.5)
>>> data = dist.rvs(*shapes, size=1000, random_state=rng)

# 假设我们不知道数据是如何生成的，但我们怀疑它遵循负二项分布，具有参数 *n* 和 *p*。
# （参见 `scipy.stats.nbinom`。）我们相信参数 *n* 少于30，而且我们知道参数 *p* 必须位于区间 [0, 1]。
# 我们将这些信息记录在变量 `bounds` 中，并将其传递给 `fit`。

>>> bounds = [(0, 30), (0, 1)]
>>> res = stats.fit(dist, data, bounds)

# `fit` 在用户指定的 `bounds` 范围内搜索最能匹配数据的值（以最大似然估计的意义上）。
# 在这种情况下，它找到了与实际生成数据相似的形状值。

>>> res.params
# FitParams(n=5.0, p=0.5028157644634368, loc=0.0)  # 结果可能有所不同

# 我们可以通过在归一化的直方图上叠加分布的概率质量函数来可视化结果。
    """
    This block of code demonstrates the fitting of a statistical distribution
    to empirical data using the `fit` function from `scipy.stats`.

    The examples illustrate various scenarios:
    1. Fitting the distribution without specifying parameter bounds.
    2. Fitting with a specified upper bound for the parameter `n`.
    3. Fitting with a fixed value for the parameter `n`.
    4. Customizing the optimizer used in the fitting process for reproducibility.

    Each example includes details on the resulting parameter estimates and, where applicable,
    comparisons between different fitting approaches.

    """

    # --- Input Validation / Standardization --- #
    # Extract user-provided bounds and guess for parameter estimation
    user_bounds = bounds
    user_guess = guess

    # Determine distribution type and default bounds based on its attributes
    if hasattr(dist, "pdf"):  # Continuous distribution check
        default_bounds = {'loc': (0, 0), 'scale': (1, 1)}
        discrete = False
    elif hasattr(dist, "pmf"):  # Discrete distribution check
        default_bounds = {'loc': (0, 0)}
        discrete = True
    else:
        # Raise an error if `dist` does not match expected distribution types
        message = ("`dist` must be an instance of `rv_continuous` "
                   "or `rv_discrete.`")
        raise ValueError(message)

    try:
        # Attempt to retrieve parameter information from the distribution
        param_info = dist._param_info()
    except AttributeError as e:
        # Raise an error if shape information is not defined for the distribution
        message = (f"Distribution `{dist.name}` is not yet supported by "
                   "`scipy.stats.fit` because shape information has "
                   "not been defined.")
        raise ValueError(message) from e
    # data input validation
    # 将数据转换为 NumPy 数组
    data = np.asarray(data)
    # 检查数据维度是否为一维
    if data.ndim != 1:
        message = "`data` must be exactly one-dimensional."
        raise ValueError(message)
    # 检查数据元素是否为有限数值
    if not (np.issubdtype(data.dtype, np.number)
            and np.all(np.isfinite(data))):
        message = "All elements of `data` must be finite numbers."
        raise ValueError(message)

    # bounds input validation and information collection
    # 获取参数信息列表长度
    n_params = len(param_info)
    # 计算形状参数个数
    n_shapes = n_params - (1 if discrete else 2)
    # 构建参数名列表
    param_list = [param.name for param in param_info]
    # 将参数名列表转换为字符串
    param_names = ", ".join(param_list)
    # 获取形状参数名列表
    shape_names = ", ".join(param_list[:n_shapes])

    # 如果用户未提供边界信息，则使用默认空字典
    if user_bounds is None:
        user_bounds = {}

    # 如果用户边界是字典类型
    if isinstance(user_bounds, dict):
        # 更新默认边界信息
        default_bounds.update(user_bounds)
        user_bounds = default_bounds
        # 创建用户边界数组
        user_bounds_array = np.empty((n_params, 2))
        # 遍历参数信息
        for i in range(n_params):
            param_name = param_info[i].name
            # 获取用户指定的边界，若未指定则使用参数的默认域
            user_bound = user_bounds.pop(param_name, None)
            if user_bound is None:
                user_bound = param_info[i].domain
            # 将边界值存入用户边界数组
            user_bounds_array[i] = user_bound
        # 若仍有未识别的参数边界，则发出警告
        if user_bounds:
            message = ("Bounds provided for the following unrecognized "
                       f"parameters will be ignored: {set(user_bounds)}")
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    # 如果用户边界是其他类型
    else:
        try:
            # 尝试将边界信息转换为浮点数数组
            user_bounds = np.asarray(user_bounds, dtype=float)
            if user_bounds.size == 0:
                user_bounds = np.empty((0, 2))
        except ValueError as e:
            message = ("Each element of a `bounds` sequence must be a tuple "
                       "containing two elements: the lower and upper bound of "
                       "a distribution parameter.")
            raise ValueError(message) from e
        # 检查边界数组的维度和形状
        if (user_bounds.ndim != 2 or user_bounds.shape[1] != 2):
            message = ("Each element of `bounds` must be a tuple specifying "
                       "the lower and upper bounds of a shape parameter")
            raise ValueError(message)
        # 检查边界数组的长度是否符合形状参数的数量要求
        if user_bounds.shape[0] < n_shapes:
            message = (f"A `bounds` sequence must contain at least {n_shapes} "
                       "elements: tuples specifying the lower and upper "
                       f"bounds of all shape parameters {shape_names}.")
            raise ValueError(message)
        # 检查边界数组的长度是否超出参数信息列表的长度
        if user_bounds.shape[0] > n_params:
            message = ("A `bounds` sequence may not contain more than "
                       f"{n_params} elements: tuples specifying the lower and "
                       "upper bounds of distribution parameters "
                       f"{param_names}.")
            raise ValueError(message)

        # 创建用户边界数组
        user_bounds_array = np.empty((n_params, 2))
        user_bounds_array[n_shapes:] = list(default_bounds.values())
        user_bounds_array[:len(user_bounds)] = user_bounds

    # 将用户边界数组赋给 user_bounds 变量
    user_bounds = user_bounds_array
    # 初始化验证后的边界列表
    validated_bounds = []
    # 遍历参数数量范围内的每一个参数
    for i in range(n_params):
        # 获取第 i 个参数的名称
        name = param_info[i].name
        # 获取用户定义的边界数组中第 i 个参数的边界
        user_bound = user_bounds_array[i]
        # 获取第 i 个参数的定义域
        param_domain = param_info[i].domain
        # 获取第 i 个参数的整数性质
        integral = param_info[i].integrality
        # 组合第 i 个参数的名称、用户边界、参数域和整数性，生成一个组合后的边界信息
        combined = _combine_bounds(name, user_bound, param_domain, integral)
        # 将组合后的边界信息添加到验证过的边界列表中
        validated_bounds.append(combined)

    # 将验证过的边界列表转换为 NumPy 数组
    bounds = np.asarray(validated_bounds)
    # 从参数信息列表中提取每个参数的整数性质，并组成列表
    integrality = [param.integrality for param in param_info]

    # 猜测输入的有效性验证

    # 如果用户提供的猜测值为 None，则设置猜测数组为 None
    if user_guess is None:
        guess_array = None
    # 如果用户提供的猜测值为字典类型
    elif isinstance(user_guess, dict):
        # 创建默认的猜测字典，每个参数使用其边界的平均值作为猜测值
        default_guess = {param.name: np.mean(bound)
                         for param, bound in zip(param_info, bounds)}
        # 找出用户提供但程序未识别的参数，并给出警告信息
        unrecognized = set(user_guess) - set(default_guess)
        if unrecognized:
            message = ("Guesses provided for the following unrecognized "
                       f"parameters will be ignored: {unrecognized}")
            # 发出运行时警告，指明警告发生的堆栈位置
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        # 将用户提供的猜测值更新到默认猜测字典中
        default_guess.update(user_guess)

        # 检查每个猜测值是否是标量值，否则抛出值错误异常
        message = ("Each element of `guess` must be a scalar "
                   "guess for a distribution parameter.")
        try:
            # 将默认猜测字典转换为 NumPy 数组作为最终的猜测数组
            guess_array = np.asarray([default_guess[param.name]
                                      for param in param_info], dtype=float)
        except ValueError as e:
            # 如果转换过程中出现错误，将捕获的值错误异常重新抛出，并附带消息
            raise ValueError(message) from e

    # 如果用户提供的猜测值不是字典类型
    else:
        # 检查每个猜测值是否是标量值，否则抛出值错误异常
        message = ("Each element of `guess` must be a scalar "
                   "guess for a distribution parameter.")
        try:
            # 将用户提供的猜测值转换为 NumPy 数组
            user_guess = np.asarray(user_guess, dtype=float)
        except ValueError as e:
            # 如果转换过程中出现错误，将捕获的值错误异常重新抛出，并附带消息
            raise ValueError(message) from e
        # 检查用户提供的猜测值数组是否是一维数组
        if user_guess.ndim != 1:
            # 如果不是一维数组，抛出值错误异常
            raise ValueError(message)
        # 检查用户提供的猜测值数组长度是否小于期望的形状数量
        if user_guess.shape[0] < n_shapes:
            # 如果长度小于期望的形状数量，抛出值错误异常，指明期望的形状名称
            message = (f"A `guess` sequence must contain at least {n_shapes} "
                       "elements: scalar guesses for the distribution shape "
                       f"parameters {shape_names}.")
            raise ValueError(message)
        # 检查用户提供的猜测值数组长度是否大于参数数量
        if user_guess.shape[0] > n_params:
            # 如果长度大于参数数量，抛出值错误异常，指明参数名称
            message = ("A `guess` sequence may not contain more than "
                       f"{n_params} elements: scalar guesses for the "
                       f"distribution parameters {param_names}.")
            raise ValueError(message)

        # 将所有参数的边界的平均值作为猜测数组的初始值
        guess_array = np.mean(bounds, axis=1)
        # 将用户提供的猜测值填充到猜测数组的前段
        guess_array[:len(user_guess)] = user_guess
    # 如果猜测数组不为空，则进行以下操作
    if guess_array is not None:
        # 复制猜测数组以避免直接修改原始数据
        guess_rounded = guess_array.copy()

        # 根据 integrality 数组的指示，对猜测值数组进行四舍五入
        guess_rounded[integrality] = np.round(guess_rounded[integrality])

        # 找出被四舍五入改变的参数索引并生成警告信息
        rounded = np.where(guess_rounded != guess_array)[0]
        for i in rounded:
            message = (f"Guess for parameter `{param_info[i].name}` "
                       f"rounded from {guess_array[i]} to {guess_rounded[i]}.")
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        # 将四舍五入后的猜测值数组限制在给定的边界范围内
        guess_clipped = np.clip(guess_rounded, bounds[:, 0], bounds[:, 1])

        # 找出被剪裁改变的参数索引并生成警告信息
        clipped = np.where(guess_clipped != guess_rounded)[0]
        for i in clipped:
            message = (f"Guess for parameter `{param_info[i].name}` "
                       f"clipped from {guess_rounded[i]} to "
                       f"{guess_clipped[i]}.")
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        # 将剪裁后的猜测值数组赋给 guess 变量
        guess = guess_clipped
    else:
        # 如果猜测数组为空，则将 guess 设置为 None
        guess = None

    # --- Fitting --- #

    # 定义负对数似然函数，通过闭包绑定 data 数据
    def nllf(free_params, data=data):
        with np.errstate(invalid='ignore', divide='ignore'):
            return dist._penalized_nnlf(free_params, data)

    # 定义负对数概率密度函数，通过闭包绑定 data 数据
    def nlpsf(free_params, data=data):
        with np.errstate(invalid='ignore', divide='ignore'):
            return dist._penalized_nlpsf(free_params, data)

    # 根据 method 参数选择优化的目标函数
    methods = {'mle': nllf, 'mse': nlpsf}
    objective = methods[method.lower()]

    # 设置优化过程中的参数和选项
    with np.errstate(invalid='ignore', divide='ignore'):
        kwds = {}
        if bounds is not None:
            kwds['bounds'] = bounds
        if np.any(integrality):
            kwds['integrality'] = integrality
        if guess is not None:
            kwds['x0'] = guess

        # 调用优化器执行优化过程
        res = optimizer(objective, **kwds)

    # 返回拟合结果对象
    return FitResult(dist, data, discrete, res)
# 定义命名元组 GoodnessOfFitResult，表示拟合结果和拟合统计信息的命名元组
GoodnessOfFitResult = namedtuple('GoodnessOfFitResult',
                                 ('fit_result', 'statistic', 'pvalue',
                                  'null_distribution'))

# 定义函数 goodness_of_fit，用于执行拟合优度检验，比较数据和分布族之间的拟合情况
def goodness_of_fit(dist, data, *, known_params=None, fit_params=None,
                    guessed_params=None, statistic='ad', n_mc_samples=9999,
                    random_state=None):
    r"""
    Perform a goodness of fit test comparing data to a distribution family.

    Given a distribution family and data, perform a test of the null hypothesis
    that the data were drawn from a distribution in that family. Any known
    parameters of the distribution may be specified. Remaining parameters of
    the distribution will be fit to the data, and the p-value of the test
    is computed accordingly. Several statistics for comparing the distribution
    to data are available.

    Parameters
    ----------
    dist : `scipy.stats.rv_continuous`
        The object representing the distribution family under the null
        hypothesis.
    data : 1D array_like
        Finite, uncensored data to be tested.
    known_params : dict, optional
        A dictionary containing name-value pairs of known distribution
        parameters. Monte Carlo samples are randomly drawn from the
        null-hypothesized distribution with these values of the parameters.
        Before the statistic is evaluated for each Monte Carlo sample, only
        remaining unknown parameters of the null-hypothesized distribution
        family are fit to the samples; the known parameters are held fixed.
        If all parameters of the distribution family are known, then the step
        of fitting the distribution family to each sample is omitted.
    fit_params : dict, optional
        A dictionary containing name-value pairs of distribution parameters
        that have already been fit to the data, e.g. using `scipy.stats.fit`
        or the ``fit`` method of `dist`. Monte Carlo samples are drawn from the
        null-hypothesized distribution with these specified values of the
        parameter. On those Monte Carlo samples, however, these and all other
        unknown parameters of the null-hypothesized distribution family are
        fit before the statistic is evaluated.
    guessed_params : dict, optional
        A dictionary containing name-value pairs of distribution parameters
        which have been guessed. These parameters are always considered as
        free parameters and are fit both to the provided `data` as well as
        to the Monte Carlo samples drawn from the null-hypothesized
        distribution. The purpose of these `guessed_params` is to be used as
        initial values for the numerical fitting procedure.
    statistic : {"ad", "ks", "cvm", "filliben"} or callable, optional
        # 统计量，可选参数，可以是字符串集合 {"ad", "ks", "cvm", "filliben"} 或可调用对象
        用于在拟合分布未知参数到数据后，将数据与分布进行比较的统计量。
        可选的统计量包括 Anderson-Darling ("ad") [1]_, Kolmogorov-Smirnov ("ks") [1]_,
        Cramer-von Mises ("cvm") [1]_, 和 Filliben ("filliben") [7]_ 统计量。
        或者可以提供一个带有签名 ``(dist, data, axis)`` 的可调用对象来计算统计量。
        这里，``dist`` 是一个冻结的分布对象（可能带有数组参数），``data`` 是蒙特卡洛样本的数组（具有兼容的形状），``axis`` 是沿着其计算统计量的数据轴。

    n_mc_samples : int, default: 9999
        # 蒙特卡洛样本数，默认为 9999
        从零假设分布中抽取的蒙特卡洛样本数，用于形成统计量的零分布。
        每个样本的大小与给定的 `data` 相同。

    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        # 伪随机数生成器状态，用于生成蒙特卡洛样本
        用于生成蒙特卡洛样本的伪随机数生成器状态。

        如果 `random_state` 是 ``None``（默认），则使用 `numpy.random.RandomState` 单例。
        如果 `random_state` 是一个整数，则使用一个新的 ``RandomState`` 实例，并使用 `random_state` 作为种子。
        如果 `random_state` 已经是 ``Generator`` 或 ``RandomState`` 实例，则使用提供的实例。
    # 导入需要的库
    from data. We describe the test using "Monte Carlo" rather than
    # 通过 "Monte Carlo" 方法描述测试，而不是使用 "parametric bootstrap"，以避免与更熟悉的非参数自举方法混淆
    "parametric bootstrap" throughout to avoid confusion with the more familiar
    nonparametric bootstrap, and describe how the test is performed below.

    *Traditional goodness of fit tests*

    Traditionally, critical values corresponding with a fixed set of
    # 传统上，根据一组固定的显著性水平预先计算临界值，使用 Monte Carlo 方法
    significance levels are pre-calculated using Monte Carlo methods. Users
    # 用户通过仅计算他们观察到的 `data` 的检验统计量的值，并将其与表格中的临界值进行比较来执行测试
    perform the test by calculating the value of the test statistic only for
    their observed `data` and comparing this value to tabulated critical
    values. This practice is not very flexible, as tables are not available for
    all distributions and combinations of known and unknown parameter values.
    Also, results can be inaccurate when critical values are interpolated from
    limited tabulated data to correspond with the user's sample size and
    fitted parameter values. To overcome these shortcomings, this function
    allows the user to perform the Monte Carlo trials adapted to their
    particular data.

    *Algorithmic overview*

    In brief, this routine executes the following steps:

      1. Fit unknown parameters to the given `data`, thereby forming the
         "null-hypothesized" distribution, and compute the statistic of
         this pair of data and distribution.
      # 将未知参数拟合到给定的 `data` 中，从而形成 "零假设" 分布，并计算该数据和分布的统计量
      2. Draw random samples from this null-hypothesized distribution.
      # 从这个零假设分布中抽取随机样本
      3. Fit the unknown parameters to each random sample.
      # 将未知参数拟合到每个随机样本中
      4. Calculate the statistic between each sample and the distribution that
         has been fit to the sample.
      # 计算每个样本与拟合到样本的分布之间的统计量
      5. Compare the value of the statistic corresponding with `data` from (1)
         against the values of the statistic corresponding with the random
         samples from (4). The p-value is the proportion of samples with a
         statistic value greater than or equal to the statistic of the observed
         data.
      # 将 (1) 中对应的数据统计值与 (4) 中随机样本的统计值进行比较。p 值是具有大于或等于观察数据统计量的统计值的样本比例

    In more detail, the steps are as follows.

    First, any unknown parameters of the distribution family specified by
    `dist` are fit to the provided `data` using maximum likelihood estimation.
    (One exception is the normal distribution with unknown location and scale:
    we use the bias-corrected standard deviation ``np.std(data, ddof=1)`` for
    the scale as recommended in [1]_.)
    # 首先，使用最大似然估计将分布族 `dist` 指定的任何未知参数拟合到提供的 `data` 中
    These values of the parameters specify a particular member of the
    distribution family referred to as the "null-hypothesized distribution",
    that is, the distribution from which the data were sampled under the null
    hypothesis. The `statistic`, which compares data to a distribution, is
    computed between `data` and the null-hypothesized distribution.

    Next, many (specifically `n_mc_samples`) new samples, each containing the
    same number of observations as `data`, are drawn from the
    null-hypothesized distribution. All unknown parameters of the distribution
    family `dist` are fit to *each resample*, and the `statistic` is computed
    # 接下来，从零假设分布中抽取许多（特别是 `n_mc_samples`）新样本，每个样本包含与 `data` 相同数量的观测值
    to *each resample*, and the `statistic` is computed
    # 计算自定义数据与其拟合分布之间的卡方统计量
    def chisquare_custom(data, fitted_distribution, n_mc_samples=1000):
        # 初始化一个空列表，用于存储每个蒙特卡罗样本的卡方统计量
        statistic_values = []
    
        # 循环生成指定数量的蒙特卡罗样本
        for _ in range(n_mc_samples):
            # 从拟合分布中生成一个随机样本
            sample = fitted_distribution.rvs(size=len(data))
            # 计算当前蒙特卡罗样本的卡方统计量，并将其添加到列表中
            statistic = chisquare(data, f_exp=sample)[0]
            statistic_values.append(statistic)
    
        # 将卡方统计量列表按照从大到小排序
        statistic_values.sort(reverse=True)
    
        # 计算 p 值，根据蒙特卡罗 null 分布中大于等于数据统计量的样本比例
        p_value = (sum(stat >= chisquare(data)[0] for stat in statistic_values) + 1) / (n_mc_samples + 1)
    
        # 返回计算得到的 p 值
        return p_value
    # 文献引用 [3] C. Genest 和 B Rémillard (2008)，描述了半参数模型中参数自举法在拟合优度检验中的有效性。
    # 文献引用 [4] I. Kojadinovic 和 J. Yan (2012)，介绍了基于加权自举法的拟合优度检验，是参数自举法的一种快速大样本替代方法。
    # 文献引用 [5] B. Phipson 和 G. K. Smyth (2010)，讨论了当排列随机抽取时，排列 p 值永远不应为零的问题，提出了计算精确 p 值的方法。
    # 文献引用 [6] H. W. Lilliefors (1967)，描述了在未知均值和方差情况下，Kolmogorov-Smirnov 正态性检验的具体方法。
    # 文献引用 [7] Filliben, James J. (1975)，介绍了用于正态性检验的概率图相关系数检验方法。

    # 示例
    # --------
    # SciPy 中的 Kolmogorov-Smirnov (KS) 检验是一种检验数据是否来自给定分布的常用方法，可以通过 `scipy.stats.ks_1samp` 访问。
    # 假设我们希望测试以下数据是否来自正态分布：
    # >>> import numpy as np
    # >>> from scipy import stats
    # >>> rng = np.random.default_rng()
    # >>> x = stats.uniform.rvs(size=75, random_state=rng)
    #
    # 要执行 KS 检验，将观察数据的经验分布函数与正态分布的累积分布函数（理论值）进行比较。
    # 在进行这项测试之前，必须完全指定零假设下的正态分布。通常是通过首先对观察数据拟合分布的 `loc` 和 `scale` 参数来完成的。
    # >>> loc, scale = np.mean(x), np.std(x, ddof=1)
    # >>> cdf = stats.norm(loc, scale).cdf
    # >>> stats.ks_1samp(x, cdf)
    # KstestResult(statistic=0.1119257570456813,
    #              pvalue=0.2827756409939257,
    #              statistic_location=0.7751845155861765,
    #              statistic_sign=-1)
    #
    # KS 检验的优点是，可以精确和高效地计算 p 值，即在零假设下获得与观察数据所得测试统计量一样极端值的概率。
    # `goodness_of_fit` 只能近似这些结果。
    #
    # >>> known_params = {'loc': loc, 'scale': scale}
    # >>> res = stats.goodness_of_fit(stats.norm, x, known_params=known_params,
    # ...                             statistic='ks', random_state=rng)
    # >>> res.statistic, res.pvalue
    # (0.1119257570456813, 0.2788)
    #
    # 统计量完全匹配，但通过形成
    # 使用 Monte Carlo 方法生成空假设分布，即从 `scipy.stats.norm` 中使用给定的参数显式绘制随机样本，并计算每个样本的统计量。
    # 比较那些统计量值至少与 ``res.statistic`` 一样极端的比例，以近似由 `scipy.stats.ks_1samp` 计算的精确 p 值。
    
    # 然而，在许多情况下，我们更倾向于仅测试数据是否来自正态分布家族中的任何一个成员，而不是具体来自已观察样本的位置和尺度拟合的正态分布。
    # 在这种情况下，Lilliefors [6]_ 认为 KS 检验过于保守（即 p 值高估拒绝真实零假设的实际概率），因此缺乏功效 - 即在真实零假设为假时拒绝零假设的能力。
    # 实际上，上面的 p 值约为 0.28，这远远大于在任何常见显著性水平下拒绝零假设所需的值。
    
    # 考虑为什么会出现这种情况。请注意，在上述 KS 检验中，统计量始终将数据与拟合到*观察数据*的正态分布的累积分布函数（CDF）进行比较。
    # 这倾向于减少观察数据的统计量值，但在计算其他样本（如我们随机绘制以形成 Monte Carlo 空假设分布的样本）的统计量时，却是“不公平”的。
    # 可以很容易地纠正这一点：每当计算样本的 KS 统计量时，我们使用拟合到*该样本*的正态分布的 CDF。在这种情况下，空假设分布未经精确计算，通常使用上述的 Monte Carlo 方法进行近似。这就是 `goodness_of_fit` 卓越之处。
    
    >>> res = stats.goodness_of_fit(stats.norm, x, statistic='ks',
    ...                             random_state=rng)
    >>> res.statistic, res.pvalue
    (0.1119257570456813, 0.0196)
    
    # 实际上，这个 p 值要小得多，并足够小以在常见显著性水平（包括 5% 和 2.5%）下（正确地）拒绝零假设。
    
    # 然而，KS 统计量对于所有偏离正态性的情况并不是非常敏感。KS 统计量最初的优势在于能够从理论上计算空假设分布，但现在我们可以通过计算方法近似空假设分布，
    # 可以使用更敏感的统计量 - 这会导致更高的测试功效。Anderson-Darling 统计量 [1]_ 倾向于更为敏感，且已使用 Monte Carlo 方法为不同显著性水平和样本大小制表了临界值。
    
    >>> res = stats.anderson(x, 'norm')
    >>> print(res.statistic)
    1.2139573337497467
    >>> print(res.critical_values)
    [0.549 0.625 0.75  0.875 1.041]
    >>> print(res.significance_level)
    [15.  10.   5.   2.5  1. ]
    """
    This block of code demonstrates the usage and interpretation of the `goodness_of_fit`
    function from the `stats` module for statistical analysis.

    >>> res = stats.goodness_of_fit(stats.norm, x, statistic='ad', random_state=rng)
    >>> res.statistic, res.pvalue
    (1.2139573337497467, 0.0034)

    The above lines perform a goodness of fit test using the Anderson-Darling statistic
    (`statistic='ad'`) against a normal distribution (`stats.norm`) fitted to the data `x`.
    `res.statistic` provides the computed statistic value and `res.pvalue` gives the
    corresponding p-value.

    >>> res = stats.goodness_of_fit(stats.rayleigh, x, statistic='cvm',
    ...                             known_params={'loc': 0}, random_state=rng)

    This snippet conducts another goodness of fit test, this time using the Cramer-von Mises
    statistic (`statistic='cvm'`) against a Rayleigh distribution (`stats.rayleigh`) with
    a known location parameter (`{'loc': 0}`) and unknown scale, fitted to the same data `x`.

    >>> res.fit_result  # location is as specified, and scale is reasonable
      params: FitParams(loc=0.0, scale=2.1026719844231243)
     success: True
     message: 'The fit was performed successfully.'

    After fitting the distribution, `res.fit_result` provides information about the fitted
    parameters (`loc` and `scale`), success of the fitting process, and any relevant messages.

    >>> ax.hist(np.log10(res.null_distribution))
    >>> ax.set_xlabel("log10 of CVM statistic under the null hypothesis")
    >>> ax.set_ylabel("Frequency")
    >>> ax.set_title("Histogram of the Monte Carlo null distribution")
    >>> plt.show()

    The histogram plot visualizes the distribution of the Cramer-von Mises (CVM) statistic
    under the null hypothesis obtained through Monte Carlo simulations (`res.null_distribution`).

    >>> res.statistic, res.pvalue
    (0.2231991510248692, 0.0525)

    Finally, `res.statistic` presents the computed CVM statistic value, and `res.pvalue`
    provides the corresponding p-value after performing the goodness of fit test.

    """
    args = _gof_iv(dist, data, known_params, fit_params, guessed_params,
                   statistic, n_mc_samples, random_state)
    # 将元组 args 解包，分别赋值给不同的变量
    (dist, data, fixed_nhd_params, fixed_rfd_params, guessed_nhd_params,
     guessed_rfd_params, statistic, n_mc_samples_int, random_state) = args

    # 根据数据拟合空假设分布
    nhd_fit_fun = _get_fit_fun(dist, data, guessed_nhd_params,
                               fixed_nhd_params)
    # 使用拟合函数计算空假设分布的参数值
    nhd_vals = nhd_fit_fun(data)
    # 基于参数值创建空假设分布对象
    nhd_dist = dist(*nhd_vals)

    # 定义随机变量生成函数
    def rvs(size):
        return nhd_dist.rvs(size=size, random_state=random_state)

    # 定义统计量函数
    fit_fun = _get_fit_fun(dist, data, guessed_rfd_params, fixed_rfd_params)
    # 确定用于比较的函数，可以是预定义的比较函数或自定义函数
    if callable(statistic):
        compare_fun = statistic
    else:
        compare_fun = _compare_dict[statistic]
    # 获取比较函数的备择假设类型
    alternative = getattr(compare_fun, 'alternative', 'greater')

    # 定义统计量计算函数
    def statistic_fun(data, axis):
        # 将数据沿指定轴移动到最后一个维度，简化计算
        data = np.moveaxis(data, axis, -1)
        # 使用拟合函数计算拟合分布的参数值
        rfd_vals = fit_fun(data)
        # 基于参数值创建拟合分布对象
        rfd_dist = dist(*rfd_vals)
        # 返回比较结果
        return compare_fun(rfd_dist, data, axis=-1)

    # 进行蒙特卡洛检验，评估拟合分布的好坏
    res = stats.monte_carlo_test(data, rvs, statistic_fun, vectorized=True,
                                 n_resamples=n_mc_samples, axis=-1,
                                 alternative=alternative)
    
    # 创建优化结果对象，表示拟合成功
    opt_res = optimize.OptimizeResult()
    opt_res.success = True
    opt_res.message = "The fit was performed successfully."
    opt_res.x = nhd_vals
    
    # 只支持连续分布，因此设置 discrete=False
    # 这并非基本限制；只是因为我们没有使用 stats.fit，离散分布没有 `fit` 方法，
    # 并且我们还没有为离散分布编写任何矢量化拟合函数。
    # 目前仅支持连续分布。
    return GoodnessOfFitResult(FitResult(dist, data, False, opt_res),
                               res.statistic, res.pvalue,
                               res.null_distribution)
# 定义私有函数 _get_fit_fun，接受分布 dist、数据 data、猜测参数 guessed_params 和固定参数 fixed_params
def _get_fit_fun(dist, data, guessed_params, fixed_params):

    # 如果 dist 没有形状参数，则 shape_names 为空列表，否则根据逗号和空格分割 dist.shapes 成列表
    shape_names = [] if dist.shapes is None else dist.shapes.split(", ")
    # 将形状参数和固定参数 'loc'、'scale' 放入 param_names 列表中
    param_names = shape_names + ['loc', 'scale']
    # 在 param_names 列表中的每个参数名前加上 'f'，形成 fparam_names 列表
    fparam_names = ['f'+name for name in param_names]
    # 如果所有的 fparam_names 都在 fixed_params 中，则 all_fixed 为 True，否则为 False
    all_fixed = not set(fparam_names).difference(fixed_params)
    # 将 guessed_params 中与 shape_names 匹配的键对应的值移除到 guessed_shapes 列表中
    guessed_shapes = [guessed_params.pop(x, None)
                      for x in shape_names if x in guessed_params]

    # 如果所有参数都是固定的，则定义 fit_fun 函数以返回固定参数值的列表
    if all_fixed:
        def fit_fun(data):
            return [fixed_params[name] for name in fparam_names]
    # 如果 dist 存在于 _fit_funs 字典中，则定义 fit_fun 函数以返回拟合后的参数数组
    elif dist in _fit_funs:
        def fit_fun(data):
            # 使用 _fit_funs[dist] 函数拟合数据，并将结果广播为数组
            params = _fit_funs[dist](data, **fixed_params)
            params = np.asarray(np.broadcast_arrays(*params))
            # 如果 params 的维度大于 1，则在倒数第二维度后增加一个新维度
            if params.ndim > 1:
                params = params[..., np.newaxis]
            return params
    # 否则定义 fit_fun_1d 函数和 fit_fun 函数，以对数据进行一维拟合
    else:
        def fit_fun_1d(data):
            return dist.fit(data, *guessed_shapes, **guessed_params,
                            **fixed_params)

        def fit_fun(data):
            # 使用 apply_along_axis 在最后一个轴上应用 fit_fun_1d 函数
            params = np.apply_along_axis(fit_fun_1d, axis=-1, arr=data)
            # 如果 params 的维度大于 1，则将其转置并在倒数第二维度后增加一个新维度
            if params.ndim > 1:
                params = params.T[..., np.newaxis]
            return params

    # 返回适用于给定分布的拟合函数 fit_fun
    return fit_fun


# 定义正态分布拟合函数 _fit_norm，接受数据 data 和可选的固定参数 floc 和 fscale
def _fit_norm(data, floc=None, fscale=None):
    # 如果 floc 和 fscale 都为 None，则分别计算数据的均值和标准差
    loc = floc
    scale = fscale
    if loc is None and scale is None:
        loc = np.mean(data, axis=-1)
        scale = np.std(data, ddof=1, axis=-1)
    # 如果 loc 为 None，则计算数据的均值
    elif loc is None:
        loc = np.mean(data, axis=-1)
    # 如果 scale 为 None，则计算数据的标准差
    elif scale is None:
        scale = np.sqrt(((data - loc)**2).mean(axis=-1))
    # 返回计算得到的 loc 和 scale
    return loc, scale


# 定义 _fit_funs 字典，将 stats.norm 映射到 _fit_norm 函数
_fit_funs = {stats.norm: _fit_norm}  # type: ignore[attr-defined]


# 定义私有函数 _anderson_darling，接受分布 dist、数据 data 和轴 axis
def _anderson_darling(dist, data, axis):
    # 对数据按最后一个轴排序，并计算相关的统计量
    x = np.sort(data, axis=-1)
    n = data.shape[-1]
    i = np.arange(1, n+1)
    # 计算 Anderson-Darling 统计量 Si，并对其求和
    Si = (2*i - 1)/n * (dist.logcdf(x) + dist.logsf(x[..., ::-1]))
    S = np.sum(Si, axis=-1)
    # 返回计算结果
    return -n - S


# 定义私有函数 _compute_dplus，接受累积分布函数值 cdfvals，计算 D+ 统计量
def _compute_dplus(cdfvals):  # adapted from _stats_py before gh-17062
    n = cdfvals.shape[-1]
    # 计算 D+ 统计量
    return (np.arange(1.0, n + 1) / n - cdfvals).max(axis=-1)


# 定义私有函数 _compute_dminus，接受累积分布函数值 cdfvals，计算 D- 统计量
def _compute_dminus(cdfvals):
    n = cdfvals.shape[-1]
    # 计算 D- 统计量
    return (cdfvals - np.arange(0.0, n)/n).max(axis=-1)


# 定义私有函数 _kolmogorov_smirnov，接受分布 dist、数据 data 和轴 axis
def _kolmogorov_smirnov(dist, data, axis):
    # 对数据按最后一个轴排序，并计算累积分布函数值 cdfvals
    x = np.sort(data, axis=-1)
    cdfvals = dist.cdf(x)
    # 计算 Kolmogorov-Smirnov 统计量 D+ 和 D-
    Dplus = _compute_dplus(cdfvals)  # 总是沿着最后一个轴操作
    Dminus = _compute_dminus(cdfvals)
    # 返回 D+ 和 D- 的最大值作为结果
    return np.maximum(Dplus, Dminus)


# 定义私有函数 _corr，接受数据 X 和 M，计算简化的相关系数 r
def _corr(X, M):
    # 简化的向量化相关系数计算
    # （此处代码缺失）
    # 计算向量 X 和 M 的均值，axis=-1 表示沿着最后一个维度进行均值计算，keepdims=True 保持维度信息
    Xm = X.mean(axis=-1, keepdims=True)
    Mm = M.mean(axis=-1, keepdims=True)
    # 计算 Pearson 相关系数的分子部分
    num = np.sum((X - Xm) * (M - Mm), axis=-1)
    # 计算 Pearson 相关系数的分母部分，使用 np.sqrt 对分母进行开方计算
    den = np.sqrt(np.sum((X - Xm)**2, axis=-1) * np.sum((M - Mm)**2, axis=-1))
    # 返回计算得到的 Pearson 相关系数
    return num/den
# [7] Section 8 # 1
# 对输入数据沿着指定轴(axis)进行排序，并赋值给变量X
X = np.sort(data, axis=-1)

# [7] Section 8 # 2
# 获取数据的最后一个维度的大小，赋值给变量n
n = data.shape[-1]
# 生成一个从1到n的整数数组，赋值给变量k
k = np.arange(1, n+1)
# 使用Beta分布的中位数作为Filliben方法中的参数m
m = stats.beta(k, n + 1 - k).median()

# [7] Section 8 # 3
# 根据累积分布函数的中位数m，计算对应分布的百分位点，并赋值给变量M
M = dist.ppf(m)

# [7] Section 8 # 4
# 调用_corr函数，计算X和M之间的相关性，并返回结果
return _corr(X, M)
    # 如果 `n_mc_samples_int` 不等于 `n_mc_samples`，则抛出类型错误异常
    if n_mc_samples_int != n_mc_samples:
        message = "`n_mc_samples` must be an integer."
        raise TypeError(message)

    # 检查并确保随机状态参数 `random_state` 是一个有效的随机状态对象
    random_state = check_random_state(random_state)

    # 返回一个包含多个元素的元组，包括分布 `dist`、数据 `data`、固定的近邻直径参数 `fixed_nhd_params`、
    # 固定的距离函数参数 `fixed_rfd_params`、推测的近邻直径参数 `guessed_nhd_params`、
    # 推测的距离函数参数 `guessed_rfd_params`、统计量 `statistic`、整数化的蒙特卡洛采样次数 `n_mc_samples_int`、
    # 以及随机状态对象 `random_state`
    return (dist, data, fixed_nhd_params, fixed_rfd_params, guessed_nhd_params,
            guessed_rfd_params, statistic, n_mc_samples_int, random_state)
```