# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_ransac.py`

```
"""
===========================================
Robust linear model estimation using RANSAC
===========================================

In this example, we see how to robustly fit a linear model to faulty data using
the :ref:`RANSAC <ransac_regression>` algorithm.

The ordinary linear regressor is sensitive to outliers, and the fitted line can
easily be skewed away from the true underlying relationship of data.

The RANSAC regressor automatically splits the data into inliers and outliers,
and the fitted line is determined only by the identified inliers.


"""

# 导入必要的库
import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets, linear_model

# 设置样本和异常值的数量
n_samples = 1000
n_outliers = 50

# 生成具有异常值的数据集
X, y, coef = datasets.make_regression(
    n_samples=n_samples,
    n_features=1,
    n_informative=1,
    noise=10,
    coef=True,
    random_state=0,
)

# 添加异常值数据
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# 使用所有数据拟合直线
lr = linear_model.LinearRegression()
lr.fit(X, y)

# 使用 RANSAC 算法健壮拟合线性模型
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# 预测估计模型的数据
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# 比较估计的系数
print("Estimated coefficients (true, linear regression, RANSAC):")
print(coef, lr.coef_, ransac.estimator_.coef_)

lw = 2
# 绘制散点图和拟合直线
plt.scatter(
    X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
)
plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=lw,
    label="RANSAC regressor",
)
plt.legend(loc="lower right")
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
```