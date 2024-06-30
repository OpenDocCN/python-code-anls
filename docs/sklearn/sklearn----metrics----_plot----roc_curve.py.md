# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\roc_curve.py`

```
# 导入所需模块，从utils._plotting中导入_BinaryClassifierCurveDisplayMixin类
# 从_ranking模块中导入auc和roc_curve函数
from ...utils._plotting import _BinaryClassifierCurveDisplayMixin
from .._ranking import auc, roc_curve

# 定义RocCurveDisplay类，继承自_BinaryClassifierCurveDisplayMixin类
class RocCurveDisplay(_BinaryClassifierCurveDisplayMixin):
    """ROC Curve visualization.

    It is recommend to use
    :func:`~sklearn.metrics.RocCurveDisplay.from_estimator` or
    :func:`~sklearn.metrics.RocCurveDisplay.from_predictions` to create
    a :class:`~sklearn.metrics.RocCurveDisplay`. All parameters are
    stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    fpr : ndarray
        False positive rate.

    tpr : ndarray
        True positive rate.

    roc_auc : float, default=None
        Area under ROC curve. If None, the roc_auc score is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class when computing the roc auc
        metrics. By default, `estimators.classes_[1]` is considered
        as the positive class.

        .. versionadded:: 0.24

    Attributes
    ----------
    line_ : matplotlib Artist
        ROC Curve.

    chance_level_ : matplotlib Artist or None
        The chance level line. It is `None` if the chance level is not plotted.

        .. versionadded:: 1.3

    ax_ : matplotlib Axes
        Axes with ROC Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    RocCurveDisplay.from_estimator : Plot Receiver Operating Characteristic
        (ROC) curve given an estimator and some data.
    RocCurveDisplay.from_predictions : Plot Receiver Operating Characteristic
        (ROC) curve given the true and predicted values.
    roc_auc_score : Compute the area under the ROC curve.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([0, 0, 1, 1])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    >>> roc_auc = metrics.auc(fpr, tpr)
    >>> display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    ...                                   estimator_name='example estimator')
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    # 初始化函数，接收fpr、tpr、roc_auc、estimator_name和pos_label作为参数
    def __init__(self, *, fpr, tpr, roc_auc=None, estimator_name=None, pos_label=None):
        self.estimator_name = estimator_name  # 设置属性：评估器名称
        self.fpr = fpr  # 设置属性：假正率（False Positive Rate）
        self.tpr = tpr  # 设置属性：真正率（True Positive Rate）
        self.roc_auc = roc_auc  # 设置属性：ROC曲线下的面积
        self.pos_label = pos_label  # 设置属性：正类标签

    # 绘图函数，接收参数ax、name、plot_chance_level、chance_level_kw和kwargs
    def plot(
        self,
        ax=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        **kwargs,
    ):
    ):
        """
        Plot visualization.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use `estimator_name` if
            not `None`, otherwise no labeling is shown.

        plot_chance_level : bool, default=False
            Whether to plot the chance level.

            .. versionadded:: 1.3

        chance_level_kw : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

            .. versionadded:: 1.3

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.RocCurveDisplay`
            Object that stores computed values.
        """
        # Validate plot parameters and retrieve necessary objects
        self.ax_, self.figure_, name = self._validate_plot_params(ax=ax, name=name)

        # Prepare line properties for the ROC curve
        line_kwargs = {}
        if self.roc_auc is not None and name is not None:
            line_kwargs["label"] = f"{name} (AUC = {self.roc_auc:0.2f})"
        elif self.roc_auc is not None:
            line_kwargs["label"] = f"AUC = {self.roc_auc:0.2f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        # Prepare properties for the chance level line
        chance_level_line_kw = {
            "label": "Chance level (AUC = 0.5)",
            "color": "k",
            "linestyle": "--",
        }

        if chance_level_kw is not None:
            chance_level_line_kw.update(**chance_level_kw)

        # Plot the ROC curve using matplotlib
        (self.line_,) = self.ax_.plot(self.fpr, self.tpr, **line_kwargs)

        # Determine the label for positive label in the plot's information
        info_pos_label = (
            f" (Positive label: {self.pos_label})" if self.pos_label is not None else ""
        )

        # Set labels for x and y axes, considering the positive label
        xlabel = "False Positive Rate" + info_pos_label
        ylabel = "True Positive Rate" + info_pos_label
        self.ax_.set(
            xlabel=xlabel,
            xlim=(-0.01, 1.01),
            ylabel=ylabel,
            ylim=(-0.01, 1.01),
            aspect="equal",
        )

        # Optionally plot the chance level line
        if plot_chance_level:
            (self.chance_level_,) = self.ax_.plot(
                (0, 1), (0, 1), **chance_level_line_kw
            )
        else:
            self.chance_level_ = None

        # Add legend if any label is specified
        if "label" in line_kwargs or "label" in chance_level_line_kw:
            self.ax_.legend(loc="lower right")

        # Return the object containing computed values for display
        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        sample_weight=None,
        drop_intermediate=True,
        response_method="auto",
        pos_label=None,
        name=None,
        ax=None,
        plot_chance_level=False,
        chance_level_kw=None,
        **kwargs,
    ):
        """
        Create ROC curve visualization from a given estimator.

        Parameters
        ----------
        estimator : estimator object
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a classifier.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target values (true labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        drop_intermediate : bool, default=True
            Whether to drop some suboptimal thresholds which would not appear
            on a plotted ROC curve. This is useful in order to create lighter
            ROC curves.

        response_method : {'predict_proba', 'decision_function', 'auto'}, \
                default='auto'
            Specifies whether to use the 'predict_proba' method of the estimator,
            its 'decision_function' method, or let it be inferred automatically.

        pos_label : int or str, default=None
            Label considered as positive in the binary classification task.
            If `None`, the estimator's default positive label will be used.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, the estimator's name will be used.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        plot_chance_level : bool, default=False
            Whether to plot the chance level.

        chance_level_kw : dict, default=None
            Keyword arguments to be passed to matplotlib's `plot` for rendering
            the chance level line.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~sklearn.metrics.RocCurveDisplay`
            Object that stores computed values.
        """
    # 定义一个类方法 `from_predictions`，用于根据预测值和真实值创建一个 ROC 曲线对象
    def from_predictions(
        cls,
        # 真实的类别标签
        y_true,
        # 预测的类别标签
        y_pred,
        # 可选参数：样本权重，默认为 None
        *,
        sample_weight=None,
        # 可选参数：是否丢弃中间计算结果，默认为 True
        drop_intermediate=True,
        # 可选参数：正类别标签，默认为 None
        pos_label=None,
        # 可选参数：ROC 曲线的名称，默认为 None
        name=None,
        # 可选参数：绘图使用的轴对象，默认为 None
        ax=None,
        # 可选参数：是否绘制机会水平线，默认为 False
        plot_chance_level=False,
        # 可选参数：机会水平线的关键字参数，默认为 None
        chance_level_kw=None,
        # 其他关键字参数，用于接收未命名的额外参数
        **kwargs,
```