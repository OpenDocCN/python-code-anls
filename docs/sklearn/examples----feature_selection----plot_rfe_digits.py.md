# `D:\src\scipysrc\scikit-learn\examples\feature_selection\plot_rfe_digits.py`

```
"""
=============================
Recursive feature elimination
=============================

This example demonstrates how Recursive Feature Elimination
(:class:`~sklearn.feature_selection.RFE`) can be used to determine the
importance of individual pixels for classifying handwritten digits.
:class:`~sklearn.feature_selection.RFE` recursively removes the least
significant features, assigning ranks based on their importance, where higher
`ranking_` values denote lower importance. The ranking is visualized using both
shades of blue and pixel annotations for clarity. As expected, pixels positioned
at the center of the image tend to be more predictive than those near the edges.

.. note::

    See also :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`

"""  # noqa: E501

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 导入手写数字数据集和相关库
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# 加载手写数字数据集
digits = load_digits()
# 重塑数据集，将每张图片转换成一维数组
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# 创建管道，包括数据归一化和递归特征消除 (RFE) 步骤
pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),  # 使用 MinMaxScaler 进行数据归一化
        ("rfe", RFE(estimator=LogisticRegression(), n_features_to_select=1, step=1)),  # 使用逻辑回归作为评估器进行 RFE
    ]
)

# 在数据上拟合管道
pipe.fit(X, y)
# 获取特征排名
ranking = pipe.named_steps["rfe"].ranking_.reshape(digits.images[0].shape)

# 绘制像素排名图像
plt.matshow(ranking, cmap=plt.cm.Blues)

# 添加像素编号的注释
for i in range(ranking.shape[0]):
    for j in range(ranking.shape[1]):
        plt.text(j, i, str(ranking[i, j]), ha="center", va="center", color="black")

# 添加颜色条
plt.colorbar()
# 设置图像标题
plt.title("Ranking of pixels with RFE\n(Logistic Regression)")
# 显示图像
plt.show()
```