# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\confusion_matrix.py`

```
from itertools import product  # 导入 itertools 模块中的 product 函数，用于生成迭代器的笛卡尔积

import numpy as np  # 导入 numpy 库，用于数值计算

from ...base import is_classifier  # 导入 is_classifier 函数，用于检查对象是否为分类器
from ...utils._optional_dependencies import check_matplotlib_support  # 导入 check_matplotlib_support 函数，用于检查 matplotlib 支持情况
from ...utils.multiclass import unique_labels  # 导入 unique_labels 函数，用于获取唯一的类标签
from .. import confusion_matrix  # 导入 confusion_matrix 函数，用于计算混淆矩阵


class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.

    It is recommend to use
    :func:`~sklearn.metrics.ConfusionMatrixDisplay.from_estimator` or
    :func:`~sklearn.metrics.ConfusionMatrixDisplay.from_predictions` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    display_labels : ndarray of shape (n_classes,), default=None
        Display labels for plot. If None, display labels are set from 0 to
        `n_classes - 1`.

    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.

    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.

    ax_ : matplotlib Axes
        Axes with confusion matrix.

    figure_ : matplotlib Figure
        Figure containing the confusion matrix.

    See Also
    --------
    confusion_matrix : Compute Confusion Matrix to evaluate the accuracy of a
        classification.
    ConfusionMatrixDisplay.from_estimator : Plot the confusion matrix
        given an estimator, the data, and the label.
    ConfusionMatrixDisplay.from_predictions : Plot the confusion matrix
        given the true and predicted labels.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> clf = SVC(random_state=0)
    >>> clf.fit(X_train, y_train)
    SVC(random_state=0)
    >>> predictions = clf.predict(X_test)
    >>> cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    >>> disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    ...                               display_labels=clf.classes_)
    >>> disp.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, confusion_matrix, *, display_labels=None):
        self.confusion_matrix = confusion_matrix  # 初始化混淆矩阵属性
        self.display_labels = display_labels  # 初始化显示标签属性

    def plot(
        self,
        *,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
        colorbar=True,
        im_kw=None,
        text_kw=None,
    # 类方法装饰器，表示下面定义的方法是一个类方法，可以通过类调用，而不是实例调用
    @classmethod
    # 从一个机器学习估计器对象中创建混淆矩阵。使用以下参数：
    # - cls: 表示类本身，即当前类对象
    # - estimator: 机器学习估计器对象，可以是分类器或回归器
    # - X: 特征数据集，通常是二维数组或DataFrame
    # - y: 目标数据集，通常是一维数组或Series
    # - 其余参数用于自定义混淆矩阵的绘制和格式
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        labels=None,
        sample_weight=None,
        normalize=None,
        display_labels=None,
        include_values=True,
        xticks_rotation="horizontal",
        values_format=None,
        cmap="viridis",
        ax=None,
        colorbar=True,
        im_kw=None,
        text_kw=None,



    # 类方法装饰器，表示下面定义的方法是一个类方法，可以通过类调用，而不是实例调用
    @classmethod
    # 从预测结果中创建混淆矩阵。使用以下参数：
    # - cls: 表示类本身，即当前类对象
    # - y_true: 真实的目标值，通常是一维数组或Series
    # - y_pred: 预测的目标值，通常是一维数组或Series，与y_true的维度相同
    # - 其余参数用于自定义混淆矩阵的绘制和格式
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        labels=None,
        sample_weight=None,
        normalize=None,
        display_labels=None,
        include_values=True,
        xticks_rotation="horizontal",
        values_format=None,
        cmap="viridis",
        ax=None,
        colorbar=True,
        im_kw=None,
        text_kw=None,
```