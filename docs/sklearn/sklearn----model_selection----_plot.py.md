# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\_plot.py`

```
# 导入 NumPy 库，用于处理数值数据
import numpy as np

# 导入检查 Matplotlib 支持的函数
from ..utils._optional_dependencies import check_matplotlib_support
# 导入绘图相关函数和验证函数
from ..utils._plotting import _interval_max_min_ratio, _validate_score_name
# 导入学习曲线和验证曲线函数
from ._validation import learning_curve, validation_curve


class _BaseCurveDisplay:
    def _plot_curve(
        self,
        x_data,
        *,
        ax=None,
        negate_score=False,
        score_name=None,
        score_type="test",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None,
    ):
        pass


class LearningCurveDisplay(_BaseCurveDisplay):
    """Learning Curve visualization.

    It is recommended to use
    :meth:`~sklearn.model_selection.LearningCurveDisplay.from_estimator` to
    create a :class:`~sklearn.model_selection.LearningCurveDisplay` instance.
    All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>` for general information
    about the visualization API and
    :ref:`detailed documentation <learning_curve>` regarding the learning
    curve visualization.

    .. versionadded:: 1.2

    Parameters
    ----------
    train_sizes : ndarray of shape (n_unique_ticks,)
        Numbers of training examples that has been used to generate the
        learning curve.

    train_scores : ndarray of shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : ndarray of shape (n_ticks, n_cv_folds)
        Scores on test set.

    score_name : str, default=None
        The name of the score used in `learning_curve`. It will override the name
        inferred from the `scoring` parameter. If `score` is `None`, we use `"Score"` if
        `negate_score` is `False` and `"Negative score"` otherwise. If `scoring` is a
        string or a callable, we infer the name. We replace `_` by spaces and capitalize
        the first letter. We remove `neg_` and replace it by `"Negative"` if
        `negate_score` is `False` or just remove it otherwise.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with the learning curve.

    figure_ : matplotlib Figure
        Figure containing the learning curve.

    errorbar_ : list of matplotlib Artist or None
        When the `std_display_style` is `"errorbar"`, this is a list of
        `matplotlib.container.ErrorbarContainer` objects. If another style is
        used, `errorbar_` is `None`.

    lines_ : list of matplotlib Artist or None
        When the `std_display_style` is `"fill_between"`, this is a list of
        `matplotlib.lines.Line2D` objects corresponding to the mean train and
        test scores. If another style is used, `line_` is `None`.

    fill_between_ : list of matplotlib Artist or None
        When the `std_display_style` is `"fill_between"`, this is a list of
        `matplotlib.collections.PolyCollection` objects. If another style is
        used, `fill_between_` is `None`.

    See Also
    --------
    sklearn.model_selection.learning_curve : Compute the learning curve.
    """
    pass
    """
    构造函数，初始化学习曲线显示对象。

    Parameters
    ----------
    train_sizes : array-like of shape (n_ticks,)
        训练集大小的数组，用于绘制学习曲线。
    train_scores : array-like of shape (n_ticks, n_cv_folds)
        训练集上的分数，每个交叉验证折叠的结果。
    test_scores : array-like of shape (n_ticks, n_cv_folds)
        测试集上的分数，每个交叉验证折叠的结果。
    score_name : str or None, optional, default: None
        分数的名称，用于在图表中显示标签。

    Methods
    -------
    plot(ax=None, negate_score=False, score_name=None, score_type="both",
         std_display_style="fill_between", line_kw=None, fill_between_kw=None,
         errorbar_kw=None)
        绘制学习曲线，可选择多种显示选项。
    """
    def __init__(self, *, train_sizes, train_scores, test_scores, score_name=None):
        self.train_sizes = train_sizes  # 初始化训练集大小数组
        self.train_scores = train_scores  # 初始化训练集分数数组
        self.test_scores = test_scores  # 初始化测试集分数数组
        self.score_name = score_name  # 设置分数的名称，用于图表显示标签

    def plot(
        self,
        ax=None,
        *,
        negate_score=False,
        score_name=None,
        score_type="both",
        std_display_style="fill_between",
        line_kw=None,
        fill_between_kw=None,
        errorbar_kw=None,
    ):
        """
        Plot visualization.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        negate_score : bool, default=False
            Whether or not to negate the scores obtained through
            :func:`~sklearn.model_selection.learning_curve`. This is
            particularly useful when using the error denoted by `neg_*` in
            `scikit-learn`.

        score_name : str, default=None
            The name of the score used to decorate the y-axis of the plot. It will
            override the name inferred from the `scoring` parameter. If `score` is
            `None`, we use `"Score"` if `negate_score` is `False` and `"Negative score"`
            otherwise. If `scoring` is a string or a callable, we infer the name. We
            replace `_` by spaces and capitalize the first letter. We remove `neg_` and
            replace it by `"Negative"` if `negate_score` is
            `False` or just remove it otherwise.

        score_type : {"test", "train", "both"}, default="both"
            The type of score to plot. Can be one of `"test"`, `"train"`, or
            `"both"`.

        std_display_style : {"errorbar", "fill_between"} or None, default="fill_between"
            The style used to display the score standard deviation around the
            mean score. If None, no standard deviation representation is
            displayed.

        line_kw : dict, default=None
            Additional keyword arguments passed to the `plt.plot` used to draw
            the mean score.

        fill_between_kw : dict, default=None
            Additional keyword arguments passed to the `plt.fill_between` used
            to draw the score standard deviation.

        errorbar_kw : dict, default=None
            Additional keyword arguments passed to the `plt.errorbar` used to
            draw mean score and standard deviation score.

        Returns
        -------
        display : :class:`~sklearn.model_selection.LearningCurveDisplay`
            Object that stores computed values.
        """
        # 调用内部方法 `_plot_curve` 绘制学习曲线
        self._plot_curve(
            self.train_sizes,  # 使用的训练集大小数组
            ax=ax,  # 绘图所用的坐标轴对象
            negate_score=negate_score,  # 是否对得分进行取反处理
            score_name=score_name,  # y 轴得分的名称
            score_type=score_type,  # 绘制的得分类型
            std_display_style=std_display_style,  # 标准差显示的样式
            line_kw=line_kw,  # 传递给 `plt.plot` 的额外参数
            fill_between_kw=fill_between_kw,  # 传递给 `plt.fill_between` 的额外参数
            errorbar_kw=errorbar_kw,  # 传递给 `plt.errorbar` 的额外参数
        )
        # 设置 x 轴标签
        self.ax_.set_xlabel("Number of samples in the training set")
        # 返回当前对象自身，用于方法链
        return self
    # 定义一个类方法，用于从给定的估计器（estimator）生成学习曲线
    cls,
    # estimator: 给定的机器学习模型
    estimator,
    # X: 输入特征数据
    X,
    # y: 目标数据
    y,
    # groups: 分组信息（可选）
    *,
    # train_sizes: 训练集大小的数组，默认从0.1到1.0等间隔取5个值
    train_sizes=np.linspace(0.1, 1.0, 5),
    # cv: 交叉验证生成器或可迭代器（可选）
    cv=None,
    # scoring: 评分方法（可选）
    scoring=None,
    # exploit_incremental_learning: 是否利用增量学习（默认为False）
    exploit_incremental_learning=False,
    # n_jobs: 并行运行的作业数（可选）
    n_jobs=None,
    # pre_dispatch: 控制并行执行时任务的数量或内存消耗（可选）
    pre_dispatch="all",
    # verbose: 控制详细程度的输出（可选）
    verbose=0,
    # shuffle: 是否在每次迭代前对数据进行洗牌（可选）
    shuffle=False,
    # random_state: 随机数种子（可选）
    random_state=None,
    # error_score: 如果拟合失败时的分数值（默认为NaN）
    error_score=np.nan,
    # fit_params: 传递给拟合方法的额外参数（可选）
    fit_params=None,
    # ax: 可视化图形的轴（可选）
    ax=None,
    # negate_score: 是否对评分进行反转（可选）
    negate_score=False,
    # score_name: 评分名称（可选）
    score_name=None,
    # score_type: 评分类型，包括 'train'、'test' 或两者皆有（默认为两者都有）
    score_type="both",
    # std_display_style: 标准差展示风格（默认为 'fill_between'）
    std_display_style="fill_between",
    # line_kw: 线条属性的字典（可选）
    line_kw=None,
    # fill_between_kw: 填充区域属性的字典（可选）
    fill_between_kw=None,
    # errorbar_kw: 错误条属性的字典（可选）
    errorbar_kw=None,
class ValidationCurveDisplay(_BaseCurveDisplay):
    """Validation Curve visualization.

    It is recommended to use
    :meth:`~sklearn.model_selection.ValidationCurveDisplay.from_estimator` to
    create a :class:`~sklearn.model_selection.ValidationCurveDisplay` instance.
    All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>` for general information
    about the visualization API and :ref:`detailed documentation
    <validation_curve>` regarding the validation curve visualization.

    .. versionadded:: 1.3

    Parameters
    ----------
    param_name : str
        Name of the parameter that has been varied.

    param_range : array-like of shape (n_ticks,)
        The values of the parameter that have been evaluated.

    train_scores : ndarray of shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : ndarray of shape (n_ticks, n_cv_folds)
        Scores on test set.

    score_name : str, default=None
        The name of the score used in `validation_curve`. It will override the name
        inferred from the `scoring` parameter. If `score` is `None`, we use `"Score"` if
        `negate_score` is `False` and `"Negative score"` otherwise. If `scoring` is a
        string or a callable, we infer the name. We replace `_` by spaces and capitalize
        the first letter. We remove `neg_` and replace it by `"Negative"` if
        `negate_score` is `False` or just remove it otherwise.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with the validation curve.

    figure_ : matplotlib Figure
        Figure containing the validation curve.

    errorbar_ : list of matplotlib Artist or None
        When the `std_display_style` is `"errorbar"`, this is a list of
        `matplotlib.container.ErrorbarContainer` objects. If another style is
        used, `errorbar_` is `None`.

    lines_ : list of matplotlib Artist or None
        When the `std_display_style` is `"fill_between"`, this is a list of
        `matplotlib.lines.Line2D` objects corresponding to the mean train and
        test scores. If another style is used, `line_` is `None`.

    fill_between_ : list of matplotlib Artist or None
        When the `std_display_style` is `"fill_between"`, this is a list of
        `matplotlib.collections.PolyCollection` objects. If another style is
        used, `fill_between_` is `None`.

    See Also
    --------
    sklearn.model_selection.validation_curve : Compute the validation curve.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import ValidationCurveDisplay, validation_curve
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=1_000, random_state=0)
    >>> logistic_regression = LogisticRegression()
    >>> param_name, param_range = "C", np.logspace(-8, 3, 10)

    Notes
    -----
    This class provides a visualization of validation curves, allowing users to
    inspect how the performance metric of a model varies with different parameter
    values. It encapsulates functionality to display these curves using matplotlib,
    offering both visual insight and quantitative understanding of model behavior
    across parameter values.
    """
    """
    >>> train_scores, test_scores = validation_curve(
    ...     logistic_regression, X, y, param_name=param_name, param_range=param_range
    ... )
    >>> display = ValidationCurveDisplay(
    ...     param_name=param_name, param_range=param_range,
    ...     train_scores=train_scores, test_scores=test_scores, score_name="Score"
    ... )
    >>> display.plot()
    <...>
    >>> plt.show()
    """
    
    class ValidationCurveDisplay:
        """
        初始化方法，接受参数用于初始化对象。
    
        Args:
            param_name (str): 参数名称，用于绘制验证曲线。
            param_range (ndarray or list): 参数范围，用于绘制验证曲线。
            train_scores (ndarray): 训练集分数数组。
            test_scores (ndarray): 测试集分数数组。
            score_name (str, optional): 分数的名称，用于标记绘图中的分数。
    
        Attributes:
            param_name (str): 参数名称，用于绘制验证曲线。
            param_range (ndarray or list): 参数范围，用于绘制验证曲线。
            train_scores (ndarray): 训练集分数数组。
            test_scores (ndarray): 测试集分数数组。
            score_name (str or None): 分数的名称，用于标记绘图中的分数。
        """
    
        def __init__(
            self, *, param_name, param_range, train_scores, test_scores, score_name=None
        ):
            self.param_name = param_name  # 初始化对象的参数名称
            self.param_range = param_range  # 初始化对象的参数范围
            self.train_scores = train_scores  # 初始化对象的训练集分数数组
            self.test_scores = test_scores  # 初始化对象的测试集分数数组
            self.score_name = score_name  # 初始化对象的分数名称
    
        def plot(
            self,
            ax=None,
            *,
            negate_score=False,
            score_name=None,
            score_type="both",
            std_display_style="fill_between",
            line_kw=None,
            fill_between_kw=None,
            errorbar_kw=None,
        ):
            """
            绘制验证曲线。
    
            Args:
                ax (matplotlib.axes.Axes, optional): 用于绘图的轴对象。
                negate_score (bool, optional): 是否对分数进行否定处理。
                score_name (str or None, optional): 覆盖对象初始化时的分数名称。
                score_type (str, optional): 绘制的分数类型，可以是 "train"、"test" 或 "both"。
                std_display_style (str, optional): 标准差显示样式，例如 "fill_between"。
                line_kw (dict, optional): 线条属性的关键字参数。
                fill_between_kw (dict, optional): 填充区域属性的关键字参数。
                errorbar_kw (dict, optional): 误差条属性的关键字参数。
            """
    ):
        """
        Plot visualization.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        negate_score : bool, default=False
            Whether or not to negate the scores obtained through
            :func:`~sklearn.model_selection.validation_curve`. This is
            particularly useful when using the error denoted by `neg_*` in
            `scikit-learn`.

        score_name : str, default=None
            The name of the score used to decorate the y-axis of the plot. It will
            override the name inferred from the `scoring` parameter. If `score` is
            `None`, we use `"Score"` if `negate_score` is `False` and `"Negative score"`
            otherwise. If `scoring` is a string or a callable, we infer the name. We
            replace `_` by spaces and capitalize the first letter. We remove `neg_` and
            replace it by `"Negative"` if `negate_score` is
            `False` or just remove it otherwise.

        score_type : {"test", "train", "both"}, default="both"
            The type of score to plot. Can be one of `"test"`, `"train"`, or
            `"both"`.

        std_display_style : {"errorbar", "fill_between"} or None, default="fill_between"
            The style used to display the score standard deviation around the
            mean score. If None, no standard deviation representation is
            displayed.

        line_kw : dict, default=None
            Additional keyword arguments passed to the `plt.plot` used to draw
            the mean score.

        fill_between_kw : dict, default=None
            Additional keyword arguments passed to the `plt.fill_between` used
            to draw the score standard deviation.

        errorbar_kw : dict, default=None
            Additional keyword arguments passed to the `plt.errorbar` used to
            draw mean score and standard deviation score.

        Returns
        -------
        display : :class:`~sklearn.model_selection.ValidationCurveDisplay`
            Object that stores computed values.
        """
        # 调用内部方法 _plot_curve() 绘制曲线图
        self._plot_curve(
            self.param_range,
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )
        # 设置 x 轴标签为参数的名称
        self.ax_.set_xlabel(f"{self.param_name}")
        # 返回当前对象实例，用于链式调用
        return self
    # 定义一个类方法，用于从一个机器学习估计器对象中进行评分曲线的生成
    def from_estimator(
        cls,
        estimator,              # 机器学习估计器对象，用于生成评分曲线
        X,                      # 特征数据集
        y,                      # 目标数据集
        *,                     # 以下参数为关键字参数
        param_name,             # 待调参数的名称
        param_range,            # 待调参数的取值范围
        groups=None,            # 分组标签（如果适用）
        cv=None,                # 交叉验证生成器或迭代器
        scoring=None,           # 评分标准
        n_jobs=None,            # 并行运行的作业数量
        pre_dispatch="all",     # 用于并行执行时预调度的作业数量或方式
        verbose=0,              # 控制详细程度的整数
        error_score=np.nan,     # 在评分期间出现错误时返回的值
        fit_params=None,        # 传递给拟合方法的参数
        ax=None,                # matplotlib.axes.Axes 对象用于绘图
        negate_score=False,     # 是否反转评分的符号
        score_name=None,        # 评分曲线的名称
        score_type="both",      # 评分曲线的类型（训练、测试或两者）
        std_display_style="fill_between",  # 显示标准差的样式
        line_kw=None,           # 传递给曲线绘制方法的关键字参数
        fill_between_kw=None,   # 传递给填充方法的关键字参数
        errorbar_kw=None,       # 传递给误差条绘制方法的关键字参数
```