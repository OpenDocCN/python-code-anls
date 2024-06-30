# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_feature_agglomeration_vs_univariate_selection.py`

```
# ==============================================================================
# Feature agglomeration vs. univariate selection
# ==============================================================================
#
# This example compares 2 dimensionality reduction strategies:
#
# - univariate feature selection with Anova
#
# - feature agglomeration with Ward hierarchical clustering
#
# Both methods are compared in a regression problem using
# a BayesianRidge as supervised estimator.
#

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
#

# %%
# Import necessary libraries
import shutil        # 用于高级文件操作，例如移动文件
import tempfile      # 用于创建临时文件和目录

import matplotlib.pyplot as plt  # 引入绘图工具
import numpy as np               # 引入数值计算工具
from joblib import Memory        # 用于缓存函数计算结果
from scipy import linalg, ndimage  # 引入线性代数和图像处理工具

from sklearn import feature_selection  # 引入特征选择模块
from sklearn.cluster import FeatureAgglomeration  # 引入特征聚类模块
from sklearn.feature_extraction.image import grid_to_graph  # 引入图像网格转图模块
from sklearn.linear_model import BayesianRidge   # 引入贝叶斯岭回归模型
from sklearn.model_selection import GridSearchCV, KFold  # 引入网格搜索和交叉验证工具
from sklearn.pipeline import Pipeline   # 引入管道工具，用于构建工作流

# %%
# Set parameters
n_samples = 200   # 样本数
size = 40         # 图像尺寸
roi_size = 15     # 区域尺寸
snr = 5.0         # 信噪比
np.random.seed(0)  # 设定随机种子

# %%
# Generate data
coef = np.zeros((size, size))   # 初始化系数矩阵
coef[0:roi_size, 0:roi_size] = -1.0   # 在左上角区域设置系数为-1
coef[-roi_size:, -roi_size:] = 1.0     # 在右下角区域设置系数为1

X = np.random.randn(n_samples, size**2)   # 生成服从正态分布的数据矩阵X
for x in X:   # 对每个样本进行数据平滑处理
    x[:] = ndimage.gaussian_filter(x.reshape(size, size), sigma=1.0).ravel()
X -= X.mean(axis=0)   # 对数据进行标准化处理
X /= X.std(axis=0)

y = np.dot(X, coef.ravel())   # 计算目标值y

# %%
# Add noise
noise = np.random.randn(y.shape[0])   # 生成噪声
noise_coef = (linalg.norm(y, 2) / np.exp(snr / 20.0)) / linalg.norm(noise, 2)   # 计算噪声系数
y += noise_coef * noise   # 添加噪声到目标值y中

# %%
# Compute the coefs of a Bayesian Ridge with GridSearch
cv = KFold(2)   # 创建交叉验证生成器
ridge = BayesianRidge()   # 创建贝叶斯岭回归模型
cachedir = tempfile.mkdtemp()   # 创建临时目录作为缓存目录
mem = Memory(location=cachedir, verbose=1)   # 创建内存缓存对象

# %%
# Ward agglomeration followed by BayesianRidge
connectivity = grid_to_graph(n_x=size, n_y=size)   # 根据图像网格创建连接图
ward = FeatureAgglomeration(n_clusters=10, connectivity=connectivity, memory=mem)   # 创建特征聚类对象
clf = Pipeline([("ward", ward), ("ridge", ridge)])   # 创建工作流管道
# Select the optimal number of parcels with grid search
clf = GridSearchCV(clf, {"ward__n_clusters": [10, 20, 30]}, n_jobs=1, cv=cv)   # 创建网格搜索对象
clf.fit(X, y)   # 拟合模型，寻找最佳参数
coef_ = clf.best_estimator_.steps[-1][1].coef_   # 提取最佳模型的系数
coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_)   # 反向转换特征
coef_agglomeration_ = coef_.reshape(size, size)   # 重塑系数为图像大小

# %%
# Anova univariate feature selection followed by BayesianRidge
f_regression = mem.cache(feature_selection.f_regression)   # 缓存特征选择函数
anova = feature_selection.SelectPercentile(f_regression)   # 创建基于百分比选择特征的对象
clf = Pipeline([("anova", anova), ("ridge", ridge)])   # 创建工作流管道
# Select the optimal percentage of features with grid search
clf = GridSearchCV(clf, {"anova__percentile": [5, 10, 20]}, cv=cv)   # 创建网格搜索对象
clf.fit(X, y)   # 拟合模型，寻找最佳参数
coef_ = clf.best_estimator_.steps[-1][1].coef_   # 提取最佳模型的系数
coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_.reshape(1, -1))   # 反向转换特征
coef_selection_ = coef_.reshape(size, size)   # 重塑系数为图像大小

# %%
# Inverse the transformation to plot the results on an image
plt.close("all")   # 关闭所有已打开的图像窗口
# 创建一个 7.3x2.7 英寸大小的图形窗口
plt.figure(figsize=(7.3, 2.7))

# 在图形窗口中创建一个 1x3 的子图布局，并选取第一个子图
plt.subplot(1, 3, 1)

# 在当前子图中显示 coef 数据，使用最近邻插值，颜色映射为红蓝渐变
plt.imshow(coef, interpolation="nearest", cmap=plt.cm.RdBu_r)

# 设置当前子图的标题为 "True weights"
plt.title("True weights")

# 在图形窗口中创建一个 1x3 的子图布局，并选取第二个子图
plt.subplot(1, 3, 2)

# 在当前子图中显示 coef_selection_ 数据，使用最近邻插值，颜色映射为红蓝渐变
plt.imshow(coef_selection_, interpolation="nearest", cmap=plt.cm.RdBu_r)

# 设置当前子图的标题为 "Feature Selection"
plt.title("Feature Selection")

# 在图形窗口中创建一个 1x3 的子图布局，并选取第三个子图
plt.subplot(1, 3, 3)

# 在当前子图中显示 coef_agglomeration_ 数据，使用最近邻插值，颜色映射为红蓝渐变
plt.imshow(coef_agglomeration_, interpolation="nearest", cmap=plt.cm.RdBu_r)

# 设置当前子图的标题为 "Feature Agglomeration"
plt.title("Feature Agglomeration")

# 调整子图的布局，设置边距和相对位置
plt.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.16, 0.26)

# 显示整个图形窗口
plt.show()



# %%
# 尝试移除临时缓存目录 cachedir，如果移除失败则忽略错误
shutil.rmtree(cachedir, ignore_errors=True)
```