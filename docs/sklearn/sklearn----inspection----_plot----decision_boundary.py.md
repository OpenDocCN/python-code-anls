# `D:\src\scipysrc\scikit-learn\sklearn\inspection\_plot\decision_boundary.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from ...base import is_regressor  # 导入模型基类中的 is_regressor 函数
from ...preprocessing import LabelEncoder  # 导入标签编码器
from ...utils import _safe_indexing  # 导入安全索引工具函数
from ...utils._optional_dependencies import check_matplotlib_support  # 导入检查 Matplotlib 支持的函数
from ...utils._response import _get_response_values  # 导入获取响应值的函数
from ...utils._set_output import _get_adapter_from_container  # 导入从容器获取适配器的函数
from ...utils.validation import (  # 导入验证模块中的多个函数
    _is_arraylike_not_scalar,
    _is_pandas_df,
    _is_polars_df,
    _num_features,
    check_is_fitted,
)


def _check_boundary_response_method(estimator, response_method, class_of_interest):
    """Validate the response methods to be used with the fitted estimator.

    Parameters
    ----------
    estimator : object
        Fitted estimator to check.

    response_method : {'auto', 'predict_proba', 'decision_function', 'predict'}
        Specifies whether to use :term:`predict_proba`,
        :term:`decision_function`, :term:`predict` as the target response.
        If set to 'auto', the response method is tried in the following order:
        :term:`decision_function`, :term:`predict_proba`, :term:`predict`.

    class_of_interest : int, float, bool, str or None
        The class considered when plotting the decision. If the label is specified, it
        is then possible to plot the decision boundary in multiclass settings.

        .. versionadded:: 1.4

    Returns
    -------
    prediction_method : list of str or str
        The name or list of names of the response methods to use.
    """
    has_classes = hasattr(estimator, "classes_")
    # 检查估计器是否具有类属性，即是否为多类分类器

    if has_classes and _is_arraylike_not_scalar(estimator.classes_[0]):
        # 如果估计器是多标签或多输出的多类分类器，则抛出错误
        msg = "Multi-label and multi-output multi-class classifiers are not supported"
        raise ValueError(msg)

    if has_classes and len(estimator.classes_) > 2:
        # 如果估计器具有超过两个类，并且响应方法不是 'predict' 或 'auto'，且未提供 class_of_interest，则抛出错误
        if response_method not in {"auto", "predict"} and class_of_interest is None:
            msg = (
                "Multiclass classifiers are only supported when `response_method` is "
                "'predict' or 'auto'. Else you must provide `class_of_interest` to "
                "plot the decision boundary of a specific class."
            )
            raise ValueError(msg)
        prediction_method = "predict" if response_method == "auto" else response_method
    elif response_method == "auto":
        # 如果响应方法为 'auto'
        if is_regressor(estimator):
            # 如果估计器是回归器
            prediction_method = "predict"
        else:
            # 否则，使用 'decision_function', 'predict_proba', 'predict' 的顺序作为响应方法
            prediction_method = ["decision_function", "predict_proba", "predict"]
    else:
        # 否则，使用指定的响应方法
        prediction_method = response_method

    return prediction_method


class DecisionBoundaryDisplay:
    """Decisions boundary visualization.

    It is recommended to use
    :func:`~sklearn.inspection.DecisionBoundaryDisplay.from_estimator`
    to create a :class:`DecisionBoundaryDisplay`. All parameters are stored as
    attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    """
    xx0 : ndarray of shape (grid_resolution, grid_resolution)
        第一个参数：meshgrid 函数的第一个输出。

    xx1 : ndarray of shape (grid_resolution, grid_resolution)
        第二个参数：meshgrid 函数的第二个输出。

    response : ndarray of shape (grid_resolution, grid_resolution)
        响应函数的值。

    xlabel : str, default=None
        x 轴的默认标签。

    ylabel : str, default=None
        y 轴的默认标签。

    Attributes
    ----------
    surface_ : matplotlib `QuadContourSet` or `QuadMesh`
        如果 `plot_method` 是 'contour' 或 'contourf'，`surface_` 是一个
        :class:`QuadContourSet <matplotlib.contour.QuadContourSet>`。如果
        `plot_method` 是 'pcolormesh'，`surface_` 是一个
        :class:`QuadMesh <matplotlib.collections.QuadMesh>`。

    ax_ : matplotlib Axes
        带有决策边界的坐标轴。

    figure_ : matplotlib Figure
        包含决策边界的图形。

    See Also
    --------
    DecisionBoundaryDisplay.from_estimator : 根据估计器绘制决策边界的方法。

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.inspection import DecisionBoundaryDisplay
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> iris = load_iris()
    >>> feature_1, feature_2 = np.meshgrid(
    ...     np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max()),
    ...     np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max())
    ... )
    >>> grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    >>> tree = DecisionTreeClassifier().fit(iris.data[:, :2], iris.target)
    >>> y_pred = np.reshape(tree.predict(grid), feature_1.shape)
    >>> display = DecisionBoundaryDisplay(
    ...     xx0=feature_1, xx1=feature_2, response=y_pred
    ... )
    >>> display.plot()
    <...>
    >>> display.ax_.scatter(
    ...     iris.data[:, 0], iris.data[:, 1], c=iris.target, edgecolor="black"
    ... )
    <...>
    >>> plt.show()
    """

    def __init__(self, *, xx0, xx1, response, xlabel=None, ylabel=None):
        self.xx0 = xx0
        self.xx1 = xx1
        self.response = response
        self.xlabel = xlabel
        self.ylabel = ylabel
    def plot(self, plot_method="contourf", ax=None, xlabel=None, ylabel=None, **kwargs):
        """Plot visualization.

        Parameters
        ----------
        plot_method : {'contourf', 'contour', 'pcolormesh'}, default='contourf'
            Plotting method to call when plotting the response. Please refer
            to the following matplotlib documentation for details:
            :func:`contourf <matplotlib.pyplot.contourf>`,
            :func:`contour <matplotlib.pyplot.contour>`,
            :func:`pcolormesh <matplotlib.pyplot.pcolormesh>`.

        ax : Matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        xlabel : str, default=None
            Overwrite the x-axis label.

        ylabel : str, default=None
            Overwrite the y-axis label.

        **kwargs : dict
            Additional keyword arguments to be passed to the `plot_method`.

        Returns
        -------
        display: :class:`~sklearn.inspection.DecisionBoundaryDisplay`
            Object that stores computed values.
        """
        # 检查是否支持 matplotlib 库
        check_matplotlib_support("DecisionBoundaryDisplay.plot")
        # 导入 matplotlib.pyplot 模块
        import matplotlib.pyplot as plt  # noqa

        # 如果指定的 plot_method 不在支持的列表中，抛出 ValueError 异常
        if plot_method not in ("contourf", "contour", "pcolormesh"):
            raise ValueError(
                "plot_method must be 'contourf', 'contour', or 'pcolormesh'"
            )

        # 如果未提供 axes 对象，则创建新的 figure 和 axes
        if ax is None:
            _, ax = plt.subplots()

        # 获取指定的绘图函数对象
        plot_func = getattr(ax, plot_method)
        # 使用指定的方法绘制图形，并存储结果
        self.surface_ = plot_func(self.xx0, self.xx1, self.response, **kwargs)

        # 如果指定了 xlabel 或者当前没有设置 x 轴标签，则设置 x 轴标签
        if xlabel is not None or not ax.get_xlabel():
            xlabel = self.xlabel if xlabel is None else xlabel
            ax.set_xlabel(xlabel)
        
        # 如果指定了 ylabel 或者当前没有设置 y 轴标签，则设置 y 轴标签
        if ylabel is not None or not ax.get_ylabel():
            ylabel = self.ylabel if ylabel is None else ylabel
            ax.set_ylabel(ylabel)

        # 存储当前使用的 axes 对象和对应的 figure
        self.ax_ = ax
        self.figure_ = ax.figure
        return self
```