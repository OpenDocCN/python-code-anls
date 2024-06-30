# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\precision_recall_curve.py`

```
# 从 collections 模块导入 Counter 类
from collections import Counter

# 从 utils._plotting 模块导入 _BinaryClassifierCurveDisplayMixin 类
from ...utils._plotting import _BinaryClassifierCurveDisplayMixin

# 从 _ranking 模块导入 average_precision_score 和 precision_recall_curve 函数
from .._ranking import average_precision_score, precision_recall_curve


class PrecisionRecallDisplay(_BinaryClassifierCurveDisplayMixin):
    """Precision Recall visualization.

    It is recommend to use
    :func:`~sklearn.metrics.PrecisionRecallDisplay.from_estimator` or
    :func:`~sklearn.metrics.PrecisionRecallDisplay.from_predictions` to create
    a :class:`~sklearn.metrics.PrecisionRecallDisplay`. All parameters are
    stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    precision : ndarray
        Precision values.

    recall : ndarray
        Recall values.

    average_precision : float, default=None
        Average precision. If None, the average precision is not shown.

    estimator_name : str, default=None
        Name of estimator. If None, then the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The class considered as the positive class. If None, the class will not
        be shown in the legend.

        .. versionadded:: 0.24

    prevalence_pos_label : float, default=None
        The prevalence of the positive label. It is used for plotting the
        chance level line. If None, the chance level line will not be plotted
        even if `plot_chance_level` is set to True when plotting.

        .. versionadded:: 1.3

    Attributes
    ----------
    line_ : matplotlib Artist
        Precision recall curve.

    chance_level_ : matplotlib Artist or None
        The chance level line. It is `None` if the chance level is not plotted.

        .. versionadded:: 1.3

    ax_ : matplotlib Axes
        Axes with precision recall curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    precision_recall_curve : Compute precision-recall pairs for different
        probability thresholds.
    PrecisionRecallDisplay.from_estimator : Plot Precision Recall Curve given
        a binary classifier.
    PrecisionRecallDisplay.from_predictions : Plot Precision Recall Curve
        using predictions from a binary classifier.

    Notes
    -----
    The average precision (cf. :func:`~sklearn.metrics.average_precision_score`) in
    scikit-learn is computed without any interpolation. To be consistent with
    this metric, the precision-recall curve is plotted without any
    interpolation as well (step-wise style).

    You can change this style by passing the keyword argument
    `drawstyle="default"` in :meth:`plot`, :meth:`from_estimator`, or
    :meth:`from_predictions`. However, the curve will not be strictly
    consistent with the reported average precision.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import (precision_recall_curve,
    """
    PrecisionRecallDisplay 是一个用于展示精确率-召回率曲线的类

    def __init__(
        self,
        precision,
        recall,
        *,
        average_precision=None,
        estimator_name=None,
        pos_label=None,
        prevalence_pos_label=None,
    ):
        # 初始化 PrecisionRecallDisplay 对象
        self.estimator_name = estimator_name  # 设定估算器名称
        self.precision = precision  # 设置精确率
        self.recall = recall  # 设置召回率
        self.average_precision = average_precision  # 平均精确率，可选
        self.pos_label = pos_label  # 正类标签，可选
        self.prevalence_pos_label = prevalence_pos_label  # 正类标签的流行度，可选

    def plot(
        self,
        ax=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        **kwargs,
    ):
        # 绘制精确率-召回率曲线
        # ax: matplotlib 的轴对象，如果为 None 则创建一个新轴
        # name: 图表的名称，可选
        # plot_chance_level: 是否绘制机会水平线，可选
        # chance_level_kw: 机会水平线的参数，可选
        # kwargs: 其他绘图参数

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        sample_weight=None,
        pos_label=None,
        drop_intermediate=False,
        response_method="auto",
        name=None,
        ax=None,
        plot_chance_level=False,
        chance_level_kw=None,
        **kwargs,
    ):
        # 从估算器对象创建 PrecisionRecallDisplay 实例
        # estimator: 用于预测的估算器对象
        # X: 特征数据
        # y: 标签数据
        # sample_weight: 样本权重，可选
        # pos_label: 正类标签，可选
        # drop_intermediate: 是否丢弃中间结果，可选
        # response_method: 响应方法，可选
        # name: 图表的名称，可选
        # ax: matplotlib 的轴对象，可选
        # plot_chance_level: 是否绘制机会水平线，可选
        # chance_level_kw: 机会水平线的参数，可选
        # kwargs: 其他参数

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        sample_weight=None,
        pos_label=None,
        drop_intermediate=False,
        name=None,
        ax=None,
        plot_chance_level=False,
        chance_level_kw=None,
        **kwargs,
    ):
        # 从预测结果创建 PrecisionRecallDisplay 实例
        # y_true: 真实标签
        # y_pred: 预测标签
        # sample_weight: 样本权重，可选
        # pos_label: 正类标签，可选
        # drop_intermediate: 是否丢弃中间结果，可选
        # name: 图表的名称，可选
        # ax: matplotlib 的轴对象，可选
        # plot_chance_level: 是否绘制机会水平线，可选
        # chance_level_kw: 机会水平线的参数，可选
        # kwargs: 其他参数
```