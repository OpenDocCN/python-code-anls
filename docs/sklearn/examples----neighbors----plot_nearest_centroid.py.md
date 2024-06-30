# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_nearest_centroid.py`

```
"""
===============================
Nearest Centroid Classification
===============================

Sample usage of Nearest Centroid classification.
It will plot the decision boundaries for each class.

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy
from matplotlib.colors import ListedColormap  # 导入颜色映射工具类

from sklearn import datasets  # 导入 sklearn 中的数据集模块
from sklearn.inspection import DecisionBoundaryDisplay  # 导入显示决策边界的工具类
from sklearn.neighbors import NearestCentroid  # 导入最近质心分类器

# import some data to play with
iris = datasets.load_iris()  # 载入鸢尾花数据集
# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]  # 仅使用前两个特征作为输入特征
y = iris.target  # 目标变量为鸢尾花的分类标签

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])  # 定义浅色调色板
cmap_bold = ListedColormap(["darkorange", "c", "darkblue"])  # 定义深色调色板

# Iterate over different shrinkage values
for shrinkage in [None, 0.2]:
    # we create an instance of Nearest Centroid Classifier and fit
```