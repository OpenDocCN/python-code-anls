# `D:\src\scipysrc\scikit-learn\examples\compose\plot_digits_pipe.py`

```
"""
=========================================================
Pipelining: chaining a PCA and a logistic regression
=========================================================

The PCA does an unsupervised dimensionality reduction, while the logistic
regression does the prediction.

We use a GridSearchCV to set the dimensionality of the PCA

"""

# 代码来源: Gaël Varoquaux
# 由 Jaques Grobler 修改以供文档化
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import numpy as np  # 导入数值计算库numpy
import polars as pl  # 导入数据处理库polars

from sklearn import datasets  # 导入sklearn的数据集模块
from sklearn.decomposition import PCA  # 导入PCA算法模块
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模块
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉验证模块
from sklearn.pipeline import Pipeline  # 导入管道模块
from sklearn.preprocessing import StandardScaler  # 导入标准化模块

# 定义一个管道，用于搜索最佳的PCA截断和分类器正则化参数组合
pca = PCA()
# 定义一个标准化处理器，用于归一化输入数据
scaler = StandardScaler()

# 将容忍度设置为较大的值，以加快示例的执行速度
logistic = LogisticRegression(max_iter=10000, tol=0.1)
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])

X_digits, y_digits = datasets.load_digits(return_X_y=True)  # 载入手写数字数据集
# 管道的参数可以使用 '__' 分隔的参数名称设置:
param_grid = {
    "pca__n_components": [5, 15, 30, 45, 60],  # PCA的组件数
    "logistic__C": np.logspace(-4, 4, 4),  # Logistic回归的正则化参数C
}
search = GridSearchCV(pipe, param_grid, n_jobs=2)  # 使用网格搜索交叉验证寻找最佳参数
search.fit(X_digits, y_digits)  # 对数据进行拟合

print("Best parameter (CV score=%0.3f):" % search.best_score_)  # 打印最佳模型的交叉验证得分
print(search.best_params_)  # 打印最佳模型的参数

# 绘制PCA的谱图
pca.fit(X_digits)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))  # 创建一个包含两个子图的图形
ax0.plot(
    np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
)  # 绘制PCA解释的方差比率
ax0.set_ylabel("PCA explained variance ratio")  # 设置y轴标签

ax0.axvline(
    search.best_estimator_.named_steps["pca"].n_components,
    linestyle=":",
    label="n_components chosen",
)  # 在图上标记选择的PCA组件数
ax0.legend(prop=dict(size=12))  # 设置图例

# 对于每个组件数，找到最佳分类器结果
components_col = "param_pca__n_components"
is_max_test_score = pl.col("mean_test_score") == pl.col("mean_test_score").max()
best_clfs = (
    pl.LazyFrame(search.cv_results_)  # 使用polars转换为延迟计算框架
    .filter(is_max_test_score.over(components_col))  # 过滤出测试分数最高的结果
    .unique(components_col)  # 唯一化组件数列
    .sort(components_col)  # 按组件数排序
    .collect()  # 收集结果
)
ax1.errorbar(
    best_clfs[components_col],
    best_clfs["mean_test_score"],
    yerr=best_clfs["std_test_score"],
)  # 绘制误差条形图
ax1.set_ylabel("Classification accuracy (val)")  # 设置y轴标签
ax1.set_xlabel("n_components")  # 设置x轴标签

plt.xlim(-1, 70)  # 设置x轴限制

plt.tight_layout()  # 自动调整子图参数以便紧凑显示
plt.show()  # 显示图形
```