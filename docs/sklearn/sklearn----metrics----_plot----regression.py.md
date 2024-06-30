# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\regression.py`

```
import numbers  # 导入用于数值判断的模块 numbers

import numpy as np  # 导入 NumPy 库，用于数值计算

from ...utils import _safe_indexing, check_random_state  # 导入本地项目中的工具函数
from ...utils._optional_dependencies import check_matplotlib_support  # 导入本地项目中的 Matplotlib 支持检查函数


class PredictionErrorDisplay:
    """Visualization of the prediction error of a regression model.

    This tool can display "residuals vs predicted" or "actual vs predicted"
    using scatter plots to qualitatively assess the behavior of a regressor,
    preferably on held-out data points.

    See the details in the docstrings of
    :func:`~sklearn.metrics.PredictionErrorDisplay.from_estimator` or
    :func:`~sklearn.metrics.PredictionErrorDisplay.from_predictions` to
    create a visualizer. All parameters are stored as attributes.

    For general information regarding `scikit-learn` visualization tools, read
    more in the :ref:`Visualization Guide <visualizations>`.
    For details regarding interpreting these plots, refer to the
    :ref:`Model Evaluation Guide <visualization_regression_evaluation>`.

    .. versionadded:: 1.2

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True values.

    y_pred : ndarray of shape (n_samples,)
        Prediction values.

    Attributes
    ----------
    line_ : matplotlib Artist
        Optimal line representing `y_true == y_pred`. Therefore, it is a
        diagonal line for `kind="predictions"` and a horizontal line for
        `kind="residuals"`.

    errors_lines_ : matplotlib Artist or None
        Residual lines. If `with_errors=False`, then it is set to `None`.

    scatter_ : matplotlib Artist
        Scatter data points.

    ax_ : matplotlib Axes
        Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the scatter and lines.

    See Also
    --------
    PredictionErrorDisplay.from_estimator : Prediction error visualization
        given an estimator and some data.
    PredictionErrorDisplay.from_predictions : Prediction error visualization
        given the true and predicted targets.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.metrics import PredictionErrorDisplay
    >>> X, y = load_diabetes(return_X_y=True)
    >>> ridge = Ridge().fit(X, y)
    >>> y_pred = ridge.predict(X)
    >>> display = PredictionErrorDisplay(y_true=y, y_pred=y_pred)
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, *, y_true, y_pred):
        self.y_true = y_true  # 初始化真实值
        self.y_pred = y_pred  # 初始化预测值

    def plot(
        self,
        ax=None,
        *,
        kind="residual_vs_predicted",  # 绘图类型，默认为残差与预测值之间的关系
        scatter_kwargs=None,  # 散点图参数
        line_kwargs=None,  # 线条参数
    ):
        """Plotting function for displaying prediction error.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on. If not provided, a new figure and axis will be created.
        
        kind : str, default="residual_vs_predicted"
            Type of plot to create. Can be "residual_vs_predicted" or "actual_vs_predicted".
        
        scatter_kwargs : dict, optional
            Additional keyword arguments to be passed to the scatter plot function.
        
        line_kwargs : dict, optional
            Additional keyword arguments to be passed to the line plot function.
        """
        # Plotting logic goes here
    # 定义一个类方法，用于从一个机器学习估计器（estimator）中生成残差图或其他预测相关图形
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        kind="residual_vs_predicted",  # 图形类型，默认为残差与预测值的关系图
        subsample=1_000,  # 子采样数量，默认为1000
        random_state=None,  # 随机数种子，默认为None
        ax=None,  # matplotlib 的轴对象，用于绘制图形，默认为None
        scatter_kwargs=None,  # 散点图的参数配置字典，默认为None
        line_kwargs=None,  # 线图的参数配置字典，默认为None
    ):
    
    # 定义一个类方法，从预测值生成残差图或其他相关图形
    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        kind="residual_vs_predicted",  # 图形类型，默认为残差与预测值的关系图
        subsample=1_000,  # 子采样数量，默认为1000
        random_state=None,  # 随机数种子，默认为None
        ax=None,  # matplotlib 的轴对象，用于绘制图形，默认为None
        scatter_kwargs=None,  # 散点图的参数配置字典，默认为None
        line_kwargs=None,  # 线图的参数配置字典，默认为None
```