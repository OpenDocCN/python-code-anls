# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_partial_dependence_visualization_api.py`

```
"""
=========================================
Advanced Plotting With Partial Dependence
=========================================
The :class:`~sklearn.inspection.PartialDependenceDisplay` object can be used
for plotting without needing to recalculate the partial dependence. In this
example, we show how to plot partial dependence plots and how to quickly
customize the plot with the visualization API.

.. note::

    See also :ref:`sphx_glr_auto_examples_miscellaneous_plot_roc_curve_visualization_api.py`

"""  # noqa: E501

# 引入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块用于绘图
import pandas as pd  # 导入 pandas 库用于数据处理

from sklearn.datasets import load_diabetes  # 导入 sklearn 的 load_diabetes 函数用于加载数据集
from sklearn.inspection import PartialDependenceDisplay  # 导入 sklearn 的 PartialDependenceDisplay 类
from sklearn.neural_network import MLPRegressor  # 导入 sklearn 的 MLPRegressor 类
from sklearn.pipeline import make_pipeline  # 导入 sklearn 的 make_pipeline 函数用于创建管道
from sklearn.preprocessing import StandardScaler  # 导入 sklearn 的 StandardScaler 类用于数据标准化
from sklearn.tree import DecisionTreeRegressor  # 导入 sklearn 的 DecisionTreeRegressor 类用于决策树回归

# %%
# 在糖尿病数据集上训练模型
# ================================================
#
# 首先，在糖尿病数据集上训练一个决策树和一个多层感知器模型。

diabetes = load_diabetes()  # 加载糖尿病数据集
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)  # 将数据集特征转换为 pandas 的 DataFrame
y = diabetes.target  # 获取目标值

tree = DecisionTreeRegressor()  # 创建决策树回归模型对象
mlp = make_pipeline(  # 创建多层感知器模型管道
    StandardScaler(),  # 数据标准化
    MLPRegressor(hidden_layer_sizes=(100, 100), tol=1e-2, max_iter=500, random_state=0),  # 多层感知器模型
)
tree.fit(X, y)  # 在数据上训练决策树模型
mlp.fit(X, y)  # 在数据上训练多层感知器模型

# %%
# 绘制两个特征的偏依赖图
# ============================================
#
# 我们绘制决策树模型的特征 "age" 和 "bmi" (身体质量指数) 的偏依赖曲线。
# 使用两个特征，:func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` 函数预计会绘制两条曲线。
# 这里的 plot 函数在 `ax` 定义的空间内放置一个包含两个图的网格。
fig, ax = plt.subplots(figsize=(12, 6))  # 创建一个包含两个子图的图形对象
ax.set_title("Decision Tree")  # 设置子图标题为 "Decision Tree"
tree_disp = PartialDependenceDisplay.from_estimator(tree, X, ["age", "bmi"], ax=ax)  # 使用决策树模型创建偏依赖显示对象

# %%
# 可以为多层感知器绘制偏依赖曲线。
# 在这种情况下，通过 `line_kw` 传递给 :func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` 函数来改变曲线的颜色。
fig, ax = plt.subplots(figsize=(12, 6))  # 创建一个包含两个子图的图形对象
ax.set_title("Multi-layer Perceptron")  # 设置子图标题为 "Multi-layer Perceptron"
mlp_disp = PartialDependenceDisplay.from_estimator(  # 使用多层感知器模型创建偏依赖显示对象
    mlp, X, ["age", "bmi"], ax=ax, line_kw={"color": "red"}
)

# %%
# 将两个模型的偏依赖绘图放在一起
# ======================================================
#
# `tree_disp` 和 `mlp_disp` :class:`~sklearn.inspection.PartialDependenceDisplay` 对象包含了重建偏依赖曲线所需的所有计算信息。
# 这意味着我们可以轻松创建额外的图形而无需重新计算曲线。
#
# 绘制曲线的一种方法是将它们放在同一图中，每个模型的曲线放在每行中。
# 首先，我们创建一个具有两行和一列的图形，其中两个轴被传递给
# 创建一个包含两个子图的图像，每个子图的大小为 10x10 英寸
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# 使用 `tree_disp` 的 `plot` 函数在第一个子图 `ax1` 上绘制决策树的偏依赖图
tree_disp.plot(ax=ax1)
# 设置第一个子图的标题为 "Decision Tree"
ax1.set_title("Decision Tree")

# 使用 `mlp_disp` 的 `plot` 函数在第二个子图 `ax2` 上绘制多层感知器的偏依赖图，
# 并设置线条的颜色为红色
mlp_disp.plot(ax=ax2, line_kw={"color": "red"})
# 设置第二个子图的标题为 "Multi-layer Perceptron"
ax2.set_title("Multi-layer Perceptron")

# %%
# 另一种比较曲线的方法是将它们绘制在同一组坐标系上。在这里，我们创建一个一行两列的图像。
# 将子图的列表传递给 `tree_disp` 和 `mlp_disp` 的 `plot` 函数，用于在同一坐标系上绘制
# 每个模型的偏依赖曲线。
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
tree_disp.plot(ax=[ax1, ax2], line_kw={"label": "Decision Tree"})
mlp_disp.plot(
    ax=[ax1, ax2], line_kw={"label": "Multi-layer Perceptron", "color": "red"}
)
# 在两个子图上显示图例
ax1.legend()
ax2.legend()

# %%
# `tree_disp.axes_` 是一个包含用于绘制偏依赖图的坐标系的 numpy 数组。可以将其传递给 `mlp_disp`，
# 以在相同的坐标系上绘制多层感知器的偏依赖图。此外，`mlp_disp.figure_` 存储了图像对象，
# 允许在调用 `plot` 后调整图像大小。在这种情况下，`tree_disp.axes_` 有两个维度，因此 `plot`
# 函数只会在最左侧的坐标系上显示 y 标签和 y 刻度。
tree_disp.plot(line_kw={"label": "Decision Tree"})
mlp_disp.plot(
    line_kw={"label": "Multi-layer Perceptron", "color": "red"}, ax=tree_disp.axes_
)
# 设置决策树图像对象的大小为 10x6 英寸
tree_disp.figure_.set_size_inches(10, 6)
# 在第一个子图上显示图例
tree_disp.axes_[0, 0].legend()
# 在第二个子图上显示图例
tree_disp.axes_[0, 1].legend()
# 显示图像
plt.show()

# %%
# 绘制单个特征的偏依赖图
# =======================
#
# 在这里，我们在同一坐标系上绘制了单个特征 "age" 的偏依赖曲线。在这种情况下，将 `tree_disp.axes_`
# 传递给第二个 `plot` 函数。
tree_disp = PartialDependenceDisplay.from_estimator(tree, X, ["age"])
mlp_disp = PartialDependenceDisplay.from_estimator(
    mlp, X, ["age"], ax=tree_disp.axes_, line_kw={"color": "red"}
)
```