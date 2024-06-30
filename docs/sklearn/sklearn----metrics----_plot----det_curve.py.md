# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\det_curve.py`

```
import scipy as sp  # 导入 scipy 库，用别名 sp 表示

from ...utils._plotting import _BinaryClassifierCurveDisplayMixin  # 从私有模块中导入 _BinaryClassifierCurveDisplayMixin 类
from .._ranking import det_curve  # 从上级包中的 _ranking 模块导入 det_curve 函数


class DetCurveDisplay(_BinaryClassifierCurveDisplayMixin):
    """DET curve visualization.

    It is recommend to use :func:`~sklearn.metrics.DetCurveDisplay.from_estimator`
    or :func:`~sklearn.metrics.DetCurveDisplay.from_predictions` to create a
    visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    fpr : ndarray
        False positive rate.

    fnr : ndarray
        False negative rate.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The label of the positive class.

    Attributes
    ----------
    line_ : matplotlib Artist
        DET Curve.

    ax_ : matplotlib Axes
        Axes with DET Curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    det_curve : Compute error rates for different probability thresholds.
    DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
        some data.
    DetCurveDisplay.from_predictions : Plot DET curve given the true and
        predicted labels.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import det_curve, DetCurveDisplay
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, random_state=0)
    >>> clf = SVC(random_state=0).fit(X_train, y_train)
    >>> y_pred = clf.decision_function(X_test)
    >>> fpr, fnr, _ = det_curve(y_test, y_pred)
    >>> display = DetCurveDisplay(
    ...     fpr=fpr, fnr=fnr, estimator_name="SVC"
    ... )
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, *, fpr, fnr, estimator_name=None, pos_label=None):
        self.fpr = fpr  # 初始化对象的 false positive rate 属性
        self.fnr = fnr  # 初始化对象的 false negative rate 属性
        self.estimator_name = estimator_name  # 初始化对象的估计器名称属性
        self.pos_label = pos_label  # 初始化对象的正类标签属性

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        sample_weight=None,
        response_method="auto",
        pos_label=None,
        name=None,
        ax=None,
        **kwargs,
    ):
        pass  # 类方法，根据估计器和数据创建 DET 曲线显示器，具体实现未给出

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        sample_weight=None,
        pos_label=None,
        name=None,
        ax=None,
        **kwargs,
    ):
        pass  # 类方法，根据真实标签和预测标签创建 DET 曲线显示器，具体实现未给出
    ):
        """
        Plot the DET curve given the true and predicted labels.

        Read more in the :ref:`User Guide <visualizations>`.

        .. versionadded:: 1.0

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_pred : array-like of shape (n_samples,)
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by `decision_function` on some classifiers).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        pos_label : int, float, bool or str, default=None
            The label of the positive class. When `pos_label=None`, if `y_true`
            is in {-1, 1} or {0, 1}, `pos_label` is set to 1, otherwise an
            error will be raised.

        name : str, default=None
            Name of DET curve for labeling. If `None`, name will be set to
            `"Classifier"`.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Additional keywords arguments passed to matplotlib `plot` function.

        Returns
        -------
        display : :class:`~sklearn.metrics.DetCurveDisplay`
            Object that stores computed values.

        See Also
        --------
        det_curve : Compute error rates for different probability thresholds.
        DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
            some data.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.metrics import DetCurveDisplay
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.svm import SVC
        >>> X, y = make_classification(n_samples=1000, random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.4, random_state=0)
        >>> clf = SVC(random_state=0).fit(X_train, y_train)
        >>> y_pred = clf.decision_function(X_test)
        >>> DetCurveDisplay.from_predictions(
        ...    y_test, y_pred)
        <...>
        >>> plt.show()
        """

        # Validate the positive label and determine the curve name
        pos_label_validated, name = cls._validate_from_predictions_params(
            y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label, name=name
        )

        # Compute false positive rate (fpr), false negative rate (fnr), and thresholds
        fpr, fnr, _ = det_curve(
            y_true,
            y_pred,
            pos_label=pos_label,
            sample_weight=sample_weight,
        )

        # Create DetCurveDisplay object with computed rates and labels
        viz = cls(
            fpr=fpr,
            fnr=fnr,
            estimator_name=name,
            pos_label=pos_label_validated,
        )

        # Plot the DET curve on the specified axes with optional additional arguments
        return viz.plot(ax=ax, name=name, **kwargs)
    def plot(self, ax=None, *, name=None, **kwargs):
        """
        Plot visualization.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of DET curve for labeling. If `None`, use `estimator_name` if
            it is not `None`, otherwise no labeling is shown.

        **kwargs : dict
            Additional keywords arguments passed to matplotlib `plot` function.

        Returns
        -------
        display : :class:`~sklearn.metrics.DetCurveDisplay`
            Object that stores computed values.
        """
        # Validate plot parameters and retrieve necessary attributes for plotting
        self.ax_, self.figure_, name = self._validate_plot_params(ax=ax, name=name)

        # Determine kwargs for line plot, including label if name is provided
        line_kwargs = {} if name is None else {"label": name}
        line_kwargs.update(**kwargs)

        # Plot DET curve using FPR and FNR transformed via inverse normal CDF
        (self.line_,) = self.ax_.plot(
            sp.stats.norm.ppf(self.fpr),
            sp.stats.norm.ppf(self.fnr),
            **line_kwargs,
        )

        # Generate label strings for x and y axes, incorporating positive label information
        info_pos_label = (
            f" (Positive label: {self.pos_label})" if self.pos_label is not None else ""
        )
        xlabel = "False Positive Rate" + info_pos_label
        ylabel = "False Negative Rate" + info_pos_label
        self.ax_.set(xlabel=xlabel, ylabel=ylabel)

        # Add legend to the plot if a label is specified in line_kwargs
        if "label" in line_kwargs:
            self.ax_.legend(loc="lower right")

        # Define tick locations and labels for x and y axes
        ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
        tick_locations = sp.stats.norm.ppf(ticks)
        tick_labels = [
            "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
            for s in ticks
        ]

        # Set x-axis properties: ticks, tick labels, and limits
        self.ax_.set_xticks(tick_locations)
        self.ax_.set_xticklabels(tick_labels)
        self.ax_.set_xlim(-3, 3)

        # Set y-axis properties: ticks, tick labels, and limits
        self.ax_.set_yticks(tick_locations)
        self.ax_.set_yticklabels(tick_labels)
        self.ax_.set_ylim(-3, 3)

        return self
```