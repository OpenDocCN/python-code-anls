# `D:\src\scipysrc\scikit-learn\sklearn\inspection\_plot\partial_dependence.py`

```
# 导入必要的模块和函数
import numbers                          # 导入 numbers 模块，用于数值类型的判断
from itertools import chain             # 导入 itertools 模块中的 chain 函数，用于迭代器操作
from math import ceil                   # 导入 math 模块中的 ceil 函数，用于向上取整

import numpy as np                      # 导入 NumPy 库，用于科学计算
from scipy import sparse                # 导入 SciPy 库中的 sparse 模块，用于稀疏矩阵
from scipy.stats.mstats import mquantiles  # 从 SciPy 库中的 mstats 模块导入 mquantiles 函数，用于计算分位数

from ...base import is_regressor        # 导入 base 模块中的 is_regressor 函数，用于检查是否为回归器
from ...utils import (                  # 导入 utils 模块中的多个函数和类
    Bunch,                              # 导入 Bunch 类，用于封装任意数据
    _safe_indexing,                     # 导入 _safe_indexing 函数，用于安全地索引操作
    check_array,                        # 导入 check_array 函数，用于验证输入是否为数组
    check_random_state,                 # 导入 check_random_state 函数，用于验证随机状态
)
from ...utils._encode import _unique    # 导入 _encode 模块中的 _unique 函数，用于获取唯一值
from ...utils._optional_dependencies import check_matplotlib_support  # 导入 _optional_dependencies 模块中的 check_matplotlib_support 函数，用于检查 Matplotlib 支持性
from ...utils.parallel import Parallel, delayed  # 导入 parallel 模块中的 Parallel 和 delayed，用于并行处理
from .. import partial_dependence       # 导入当前包中的 partial_dependence 模块
from .._pd_utils import (               # 导入 _pd_utils 模块中的多个函数
    _check_feature_names,               # 导入 _check_feature_names 函数，用于检查特征名
    _get_feature_index                  # 导入 _get_feature_index 函数，用于获取特征索引
)


class PartialDependenceDisplay:
    """Partial Dependence Plot (PDP).

    This can also display individual partial dependencies which are often
    referred to as: Individual Condition Expectation (ICE).

    It is recommended to use
    :func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` to create a
    :class:`~sklearn.inspection.PartialDependenceDisplay`. All parameters are
    stored as attributes.

    Read more in
    :ref:`sphx_glr_auto_examples_miscellaneous_plot_partial_dependence_visualization_api.py`
    and the :ref:`User Guide <partial_dependence>`.

        .. versionadded:: 0.22

    Parameters
    ----------
    pd_results : list of Bunch
        Results of :func:`~sklearn.inspection.partial_dependence` for
        ``features``.

    features : list of (int,) or list of (int, int)
        Indices of features for a given plot. A tuple of one integer will plot
        a partial dependence curve of one feature. A tuple of two integers will
        plot a two-way partial dependence curve as a contour plot.

    feature_names : list of str
        Feature names corresponding to the indices in ``features``.

    target_idx : int

        - In a multiclass setting, specifies the class for which the PDPs
          should be computed. Note that for binary classification, the
          positive class (index 1) is always used.
        - In a multioutput setting, specifies the task for which the PDPs
          should be computed.

        Ignored in binary classification or classical regression settings.

    deciles : dict
        Deciles for feature indices in ``features``.
    # 定义参数 `kind`，控制偏依赖图的类型。可以是单个字符串或字符串列表，
    # 默认为 'average'。支持的选项有 {'average', 'individual', 'both'}。
    # 单独字符串含义如下：
    # - 'average'：传统的偏依赖（PD）图；
    # - 'individual'：每个样本的独立条件期望（ICE）图；
    # - 'both'：同时绘制 ICE 和 PD 在同一图上。
    # 如果提供字符串列表，则每个字符串指定一个偏依赖图类型，数量应与 `features` 中请求的交互数量相同。
    kind : {'average', 'individual', 'both'} or list of such str, \
            default='average'

        # 说明 ICE 图（'individual' 或 'both'）不适用于二维交互作用的情况。
        # 因此，如果 `features` 请求二维交互作用的图，则会引发错误。
        # 二维交互作用图应始终配置为使用 'average' 类型。

        # 添加于版本 0.24
        # 添加 `kind` 参数，包括 `'average'`、`'individual'` 和 `'both'` 选项。

        # 添加于版本 1.1
        # 支持通过传递字符串列表为每个图指定 `kind` 类型的可能性。

        .. note::
           ICE 图（'individual' 或 'both'）不适用于二维交互作用的情况。
           因此，如果 `features` 请求二维交互作用的图，则会引发错误。
           二维交互作用图应始终配置为使用 'average' 类型。

    # 定义参数 `subsample`，用于控制当 `kind` 为 'individual' 或 'both' 时的 ICE 曲线采样。
    # 如果是浮点数，则应在 0.0 到 1.0 之间，表示要用于绘制 ICE 曲线的数据集比例。
    # 如果是整数，则表示要使用的最大绝对样本数。
    # 当 `kind='both'` 时，仍然使用完整数据集计算偏依赖。

    subsample : float, int or None, default=1000

        # 添加于版本 0.24
        # 为 `kind` 参数添加了 `'individual'` 或 `'both'` 时的 ICE 曲线采样控制。

    # 定义参数 `random_state`，用于控制在 `subsample` 不为 `None` 时选定样本的随机性。
    # 可以是整数、RandomState 实例或 None。
    # 详细信息请参见术语表中的 "随机状态"。

    random_state : int, RandomState instance or None, default=None

        # 添加于版本 0.24
        # 为 `subsample` 参数添加了样本选择随机性的控制。

    # 定义参数 `is_categorical`，指定 `features` 中每个目标特征是否是分类特征。
    # 应为布尔值的列表，列表大小应与 `features` 相同。
    # 如果为 `None`，则所有特征都假定为连续特征。

    is_categorical : list of (bool,) or list of (bool, bool), default=None

        # 添加于版本 1.2
        # 为 `features` 参数添加了每个目标特征是否是分类特征的指定。

    # 定义属性 `bounding_ax_`，表示偏依赖图网格绘制的轴。
    # 如果 `ax` 是一个轴或 None，则 `bounding_ax_` 是绘制偏依赖图网格的轴。
    # 如果 `ax` 是轴的列表或 numpy 轴数组，则 `bounding_ax_` 为 None。

    Attributes
    ----------
    bounding_ax_ : matplotlib Axes or None

        # 如果 `ax` 是一个轴或 None，则 `bounding_ax_` 是绘制偏依赖图网格的轴。
        # 如果 `ax` 是轴的列表或 numpy 轴数组，则 `bounding_ax_` 为 None。
    axes_ : ndarray of matplotlib Axes
        # 存储 matplotlib 的 Axes 对象数组
        If `ax` is an axes or None, `axes_[i, j]` is the axes on the i-th row
        and j-th column. If `ax` is a list of axes, `axes_[i]` is the i-th item
        in `ax`. Elements that are None correspond to a nonexisting axes in
        that position.

    lines_ : ndarray of matplotlib Artists
        # 存储 matplotlib 的 Artists 对象数组，用于存放偏导曲线
        If `ax` is an axes or None, `lines_[i, j]` is the partial dependence
        curve on the i-th row and j-th column. If `ax` is a list of axes,
        `lines_[i]` is the partial dependence curve corresponding to the i-th
        item in `ax`. Elements that are None correspond to a nonexisting axes
        or an axes that does not include a line plot.

    deciles_vlines_ : ndarray of matplotlib LineCollection
        # 存储 matplotlib 的 LineCollection 对象数组，表示x轴十分位数
        If `ax` is an axes or None, `vlines_[i, j]` is the line collection
        representing the x axis deciles of the i-th row and j-th column. If
        `ax` is a list of axes, `vlines_[i]` corresponds to the i-th item in
        `ax`. Elements that are None correspond to a nonexisting axes or an
        axes that does not include a PDP plot.

        .. versionadded:: 0.23

    deciles_hlines_ : ndarray of matplotlib LineCollection
        # 存储 matplotlib 的 LineCollection 对象数组，表示y轴十分位数
        If `ax` is an axes or None, `vlines_[i, j]` is the line collection
        representing the y axis deciles of the i-th row and j-th column. If
        `ax` is a list of axes, `vlines_[i]` corresponds to the i-th item in
        `ax`. Elements that are None correspond to a nonexisting axes or an
        axes that does not include a 2-way plot.

        .. versionadded:: 0.23

    contours_ : ndarray of matplotlib Artists
        # 存储 matplotlib 的 Artists 对象数组，表示偏导图
        If `ax` is an axes or None, `contours_[i, j]` is the partial dependence
        plot on the i-th row and j-th column. If `ax` is a list of axes,
        `contours_[i]` is the partial dependence plot corresponding to the i-th
        item in `ax`. Elements that are None correspond to a nonexisting axes
        or an axes that does not include a contour plot.

    bars_ : ndarray of matplotlib Artists
        # 存储 matplotlib 的 Artists 对象数组，表示偏导条形图（用于类别特征）
        If `ax` is an axes or None, `bars_[i, j]` is the partial dependence bar
        plot on the i-th row and j-th column (for a categorical feature).
        If `ax` is a list of axes, `bars_[i]` is the partial dependence bar
        plot corresponding to the i-th item in `ax`. Elements that are None
        correspond to a nonexisting axes or an axes that does not include a
        bar plot.

        .. versionadded:: 1.2

    heatmaps_ : ndarray of matplotlib Artists
        # 存储 matplotlib 的 Artists 对象数组，表示偏导热图（用于一对类别特征）
        If `ax` is an axes or None, `heatmaps_[i, j]` is the partial dependence
        heatmap on the i-th row and j-th column (for a pair of categorical
        features) . If `ax` is a list of axes, `heatmaps_[i]` is the partial
        dependence heatmap corresponding to the i-th item in `ax`. Elements
        that are None correspond to a nonexisting axes or an axes that does not
        include a heatmap.

        .. versionadded:: 1.2
    figure_ : matplotlib Figure
        Figure containing partial dependence plots.
    
    See Also
    --------
    partial_dependence : Compute Partial Dependence values.
    PartialDependenceDisplay.from_estimator : Plot Partial Dependence.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from sklearn.inspection import PartialDependenceDisplay
    >>> from sklearn.inspection import partial_dependence
    >>> X, y = make_friedman1()
    >>> clf = GradientBoostingRegressor(n_estimators=10).fit(X, y)
    >>> features, feature_names = [(0,)], [f"Features #{i}" for i in range(X.shape[1])]
    >>> deciles = {0: np.linspace(0, 1, num=5)}
    >>> pd_results = partial_dependence(
    ...     clf, X, features=0, kind="average", grid_resolution=5)
    >>> display = PartialDependenceDisplay(
    ...     [pd_results], features=features, feature_names=feature_names,
    ...     target_idx=0, deciles=deciles
    ... )
    >>> display.plot(pdp_lim={1: (-1.38, 0.66)})
    <...>
    >>> plt.show()
    """

    def __init__(
        self,
        pd_results,
        *,
        features,
        feature_names,
        target_idx,
        deciles,
        kind="average",
        subsample=1000,
        random_state=None,
        is_categorical=None,
    ):
        """
        Initialize PartialDependenceDisplay object.

        Parameters
        ----------
        pd_results : array-like of shape (n_features, grid_resolution)
            The results of the partial dependence computation.
        features : list of tuples
            The indices of the features for which partial dependence is computed.
        feature_names : list of str
            The names of the features.
        target_idx : int
            The index of the target feature.
        deciles : dict
            Dictionary specifying the deciles.
        kind : str, default='average'
            The kind of partial dependence ('average' or 'individual').
        subsample : int or float, default=1000
            Maximum number of samples used to plot ICE curves.
        random_state : int, RandomState instance, or None, default=None
            Random state used for subsampling and other random operations.
        is_categorical : list of bool, default=None
            List indicating whether each feature is categorical.
        """
        self.pd_results = pd_results
        self.features = features
        self.feature_names = feature_names
        self.target_idx = target_idx
        self.deciles = deciles
        self.kind = kind
        self.subsample = subsample
        self.random_state = random_state
        self.is_categorical = is_categorical

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        features,
        *,
        sample_weight=None,
        categorical_features=None,
        feature_names=None,
        target=None,
        response_method="auto",
        n_cols=3,
        grid_resolution=100,
        percentiles=(0.05, 0.95),
        method="auto",
        n_jobs=None,
        verbose=0,
        line_kw=None,
        ice_lines_kw=None,
        pd_line_kw=None,
        contour_kw=None,
        ax=None,
        kind="average",
        centered=False,
        subsample=1000,
        random_state=None,
    ):
        """
        Create a PartialDependenceDisplay object from a given estimator and data.

        Parameters
        ----------
        estimator : estimator object
            The estimator from which to compute partial dependence.
        X : array-like of shape (n_samples, n_features)
            The input samples.
        features : int, tuple, or list of tuples
            The indices or names of the features for which to compute partial dependence.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights used in fitting the estimator.
        categorical_features : list of int or array of bool, default=None
            Indices of categorical features or boolean mask indicating categorical features.
        feature_names : list of str, default=None
            Names of features.
        target : int, default=None
            Index of the target feature.
        response_method : str, default='auto'
            Method to interpret response of the estimator ('auto', 'predict_proba', 'decision_function').
        n_cols : int, default=3
            Number of columns in the resulting plot grid.
        grid_resolution : int, default=100
            Number of points on the grid.
        percentiles : tuple of float, default=(0.05, 0.95)
            Percentiles for the confidence interval.
        method : str, default='auto'
            Method for calculating the partial dependence ('auto', 'brute', 'recursion').
        n_jobs : int, default=None
            Number of jobs to run in parallel.
        verbose : int, default=0
            Verbosity level.
        line_kw : dict, default=None
            Keyword arguments for the main line plot.
        ice_lines_kw : dict, default=None
            Keyword arguments for ICE line plots.
        pd_line_kw : dict, default=None
            Keyword arguments for the partial dependence line plot.
        contour_kw : dict, default=None
            Keyword arguments for contour plots.
        ax : matplotlib axes, default=None
            Existing axes on which to plot.
        kind : str, default='average'
            The kind of partial dependence ('average' or 'individual').
        centered : bool, default=False
            Whether to center the plot grid on the data.
        subsample : int or float, default=1000
            Maximum number of samples used to plot ICE curves.
        random_state : int, RandomState instance, or None, default=None
            Random state used for subsampling and other random operations.
        """

    def _get_sample_count(self, n_samples):
        """
        Compute the number of samples to be used for plotting ICE curves.

        Parameters
        ----------
        n_samples : int
            Number of samples available.

        Returns
        -------
        int
            Number of samples to be used for plotting.
        """
        if isinstance(self.subsample, numbers.Integral):
            if self.subsample < n_samples:
                return self.subsample
            return n_samples
        elif isinstance(self.subsample, numbers.Real):
            return ceil(n_samples * self.subsample)
        return n_samples

    def _plot_ice_lines(
        self,
        preds,
        feature_values,
        n_ice_to_plot,
        ax,
        pd_plot_idx,
        n_total_lines_by_plot,
        individual_line_kw,
    ):
        """
        Plot ICE lines for a specific feature.

        Parameters
        ----------
        preds : array-like of shape (n_samples,)
            Predictions of the estimator.
        feature_values : array-like of shape (n_samples,)
            Feature values.
        n_ice_to_plot : int
            Number of ICE lines to plot.
        ax : matplotlib axes
            Axes on which to plot.
        pd_plot_idx : int
            Index of the partial dependence plot.
        n_total_lines_by_plot : int
            Total number of lines to be plotted.
        individual_line_kw : dict
            Keyword arguments for individual ICE lines.
        """
    ):
        """
        Plot the ICE lines.

        Parameters
        ----------
        preds : ndarray of shape \
                (n_instances, n_grid_points)
            The predictions computed for all points of `feature_values` for a
            given feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        n_ice_to_plot : int
            The number of ICE lines to plot.
        ax : Matplotlib axes
            The axis on which to plot the ICE lines.
        pd_plot_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        n_total_lines_by_plot : int
            The total number of lines expected to be plot on the axis.
        individual_line_kw : dict
            Dict with keywords passed when plotting the ICE lines.
        """
        rng = check_random_state(self.random_state)
        # 使用随机数生成器从预测数据中随机选择一部分 ICE 曲线的索引
        ice_lines_idx = rng.choice(
            preds.shape[0],  # ICE 曲线的数量为 preds 的行数
            n_ice_to_plot,   # 需要选择的 ICE 曲线数量
            replace=False,   # 不允许重复选择同一个 ICE 曲线
        )
        # 从预测数据中抽取被选中的 ICE 曲线数据
        ice_lines_subsampled = preds[ice_lines_idx, :]
        # 绘制被抽样的 ICE 曲线
        for ice_idx, ice in enumerate(ice_lines_subsampled):
            # 计算当前 ICE 曲线在总体布局中的索引位置
            line_idx = np.unravel_index(
                pd_plot_idx * n_total_lines_by_plot + ice_idx, self.lines_.shape
            )
            # 在指定的轴上绘制 ICE 曲线，并将其保存到对应的 lines_ 属性中
            self.lines_[line_idx] = ax.plot(
                feature_values, ice.ravel(), **individual_line_kw
            )[0]

    def _plot_average_dependence(
        self,
        avg_preds,
        feature_values,
        ax,
        pd_line_idx,
        line_kw,
        categorical,
        bar_kw,
    ):
        """Plot the average partial dependence.

        Parameters
        ----------
        avg_preds : ndarray of shape (n_grid_points,)
            The average predictions for all points of `feature_values` for a
            given feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        ax : Matplotlib axes
            The axis on which to plot the average PD.
        pd_line_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        line_kw : dict
            Dict with keywords passed when plotting the PD plot.
        categorical : bool
            Whether feature is categorical.
        bar_kw: dict
            Dict with keywords passed when plotting the PD bars (categorical).
        """
        if categorical:
            # If the feature is categorical, convert the sequential index to
            # 2D coordinates in the grid of categorical bars, then plot the bar
            # chart for the corresponding feature values and average predictions.
            bar_idx = np.unravel_index(pd_line_idx, self.bars_.shape)
            self.bars_[bar_idx] = ax.bar(feature_values, avg_preds, **bar_kw)[0]
            ax.tick_params(axis="x", rotation=90)
        else:
            # If the feature is not categorical, convert the sequential index to
            # 2D coordinates in the grid of line plots, then plot the line plot
            # for the corresponding feature values and average predictions.
            line_idx = np.unravel_index(pd_line_idx, self.lines_.shape)
            self.lines_[line_idx] = ax.plot(
                feature_values,
                avg_preds,
                **line_kw,
            )[0]

    def _plot_one_way_partial_dependence(
        self,
        kind,
        preds,
        avg_preds,
        feature_values,
        feature_idx,
        n_ice_lines,
        ax,
        n_cols,
        pd_plot_idx,
        n_lines,
        ice_lines_kw,
        pd_line_kw,
        categorical,
        bar_kw,
        pdp_lim,
    ):
        """Plot one-way partial dependence.

        Parameters
        ----------
        kind : str
            Type of partial dependence plot.
        preds : ndarray of shape (n_samples,)
            Predictions for all samples in X.
        avg_preds : ndarray of shape (n_grid_points,)
            Average predictions for all points of `feature_values` for a given
            feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        feature_idx : int
            Index of the feature for which partial dependence is being plotted.
        n_ice_lines : int
            Number of ICE lines to plot.
        ax : Matplotlib axes
            The axis on which to plot the PD.
        n_cols : int
            Number of columns in the grid of subplots.
        pd_plot_idx : int
            Index of the current PD plot in the grid of subplots.
        n_lines : int
            Total number of lines to plot.
        ice_lines_kw : dict
            Dict with keywords passed when plotting ICE lines.
        pd_line_kw : dict
            Dict with keywords passed when plotting PD lines.
        categorical : bool
            Whether feature is categorical.
        bar_kw : dict
            Dict with keywords passed when plotting PD bars (categorical).
        pdp_lim : tuple (min, max)
            Limits for the partial dependence plot.
        """

    def _plot_two_way_partial_dependence(
        self,
        avg_preds,
        feature_values,
        feature_idx,
        ax,
        pd_plot_idx,
        Z_level,
        contour_kw,
        categorical,
        heatmap_kw,
    ):
        """Plot two-way partial dependence.

        Parameters
        ----------
        avg_preds : ndarray of shape (n_grid_points,)
            Average predictions for all points of `feature_values` for a given
            feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        feature_idx : tuple (int, int)
            Indices of the features for which partial dependence is being plotted.
        ax : Matplotlib axes
            The axis on which to plot the PD.
        pd_plot_idx : int
            Index of the current PD plot in the grid of subplots.
        Z_level : ndarray of shape (n_grid_points, n_grid_points)
            Contour levels for the 2D PD plot.
        contour_kw : dict
            Dict with keywords passed when plotting contours.
        categorical : bool
            Whether feature is categorical.
        heatmap_kw : dict
            Dict with keywords passed when plotting heatmaps.
        """

    def plot(
        self,
        *,
        ax=None,
        n_cols=3,
        line_kw=None,
        ice_lines_kw=None,
        pd_line_kw=None,
        contour_kw=None,
        bar_kw=None,
        heatmap_kw=None,
        pdp_lim=None,
        centered=False,
    ):
        """Plot partial dependence plots.

        Parameters
        ----------
        ax : Matplotlib axes, optional
            The axes on which to plot. If None, new axes will be created.
        n_cols : int, optional
            Number of columns in the grid layout for the plots.
        line_kw : dict, optional
            Keywords passed when plotting line plots for PD.
        ice_lines_kw : dict, optional
            Keywords passed when plotting ICE lines.
        pd_line_kw : dict, optional
            Keywords passed when plotting PD lines.
        contour_kw : dict, optional
            Keywords passed when plotting contours for 2D PD.
        bar_kw : dict, optional
            Keywords passed when plotting PD bars for categorical features.
        heatmap_kw : dict, optional
            Keywords passed when plotting heatmaps for 2D PD.
        pdp_lim : tuple (min, max), optional
            Limits for the partial dependence plots.
        centered : bool, optional
            Whether to center the plots.

        Notes
        -----
        This method plots partial dependence plots based on the provided
        parameters, utilizing Matplotlib for visualization.
        """
```