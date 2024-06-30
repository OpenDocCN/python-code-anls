# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_voting_regressor.py`

```
"""
=================================================
Plot individual and voting regression predictions
=================================================

.. currentmodule:: sklearn

A voting regressor is an ensemble meta-estimator that fits several base
regressors, each on the whole dataset. Then it averages the individual
predictions to form a final prediction.
We will use three different regressors to predict the data:
:class:`~ensemble.GradientBoostingRegressor`,
:class:`~ensemble.RandomForestRegressor`, and
:class:`~linear_model.LinearRegression`).
Then the above 3 regressors will be used for the
:class:`~ensemble.VotingRegressor`.

Finally, we will plot the predictions made by all models for comparison.

We will work with the diabetes dataset which consists of 10 features
collected from a cohort of diabetes patients. The target is a quantitative
measure of disease progression one year after baseline.

"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression

# %%
# Training classifiers
# --------------------------------
#
# First, we will load the diabetes dataset and initiate a gradient boosting
# regressor, a random forest regressor and a linear regression. Next, we will
# use the 3 regressors to build the voting regressor:

# 加载糖尿病数据集，并将其分配给特征矩阵 X 和目标向量 y
X, y = load_diabetes(return_X_y=True)

# 初始化三个回归器：GradientBoostingRegressor，RandomForestRegressor 和 LinearRegression
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()

# 分别使用每个回归器拟合数据集
reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)

# 创建 VotingRegressor，使用上述三个回归器
ereg = VotingRegressor([("gb", reg1), ("rf", reg2), ("lr", reg3)])
ereg.fit(X, y)

# %%
# Making predictions
# --------------------------------
#
# Now we will use each of the regressors to make the 20 first predictions.

# 选择前 20 个样本作为测试集
xt = X[:20]

# 使用每个回归器进行预测
pred1 = reg1.predict(xt)
pred2 = reg2.predict(xt)
pred3 = reg3.predict(xt)
pred4 = ereg.predict(xt)

# %%
# Plot the results
# --------------------------------
#
# Finally, we will visualize the 20 predictions. The red stars show the average
# prediction made by :class:`~ensemble.VotingRegressor`.

# 绘制预测结果图表
plt.figure()
plt.plot(pred1, "gd", label="GradientBoostingRegressor")  # 使用绿色菱形标记绘制 GradientBoostingRegressor 的预测结果
plt.plot(pred2, "b^", label="RandomForestRegressor")     # 使用蓝色三角形标记绘制 RandomForestRegressor 的预测结果
plt.plot(pred3, "ys", label="LinearRegression")          # 使用黄色正方形标记绘制 LinearRegression 的预测结果
plt.plot(pred4, "r*", ms=10, label="VotingRegressor")    # 使用红色星形标记绘制 VotingRegressor 的平均预测结果，星形的大小为10

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("predicted")  # 设置 y 轴标签为 "predicted"
plt.xlabel("training samples")  # 设置 x 轴标签为 "training samples"
plt.legend(loc="best")  # 设置图例位置为最佳位置
plt.title("Regressor predictions and their average")  # 设置图表标题为 "Regressor predictions and their average"

plt.show()  # 显示图表
```