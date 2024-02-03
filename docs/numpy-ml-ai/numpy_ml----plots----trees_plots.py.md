# `numpy-ml\numpy_ml\plots\trees_plots.py`

```
# 禁用 flake8 的警告
# 导入 NumPy 库，并使用 np 别名
import numpy as np

# 从 sklearn 库中导入准确率评估、均方误差评估、生成聚类数据、生成回归数据、划分训练集和测试集的函数
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_blobs, make_regression
from sklearn.model_selection import train_test_split

# 导入 matplotlib.pyplot 库，并使用 plt 别名
import matplotlib.pyplot as plt

# 导入 seaborn 库，并设置图表样式为白色，字体比例为 0.9
# 参考链接：https://seaborn.pydata.org/generated/seaborn.set_context.html
# 参考链接：https://seaborn.pydata.org/generated/seaborn.set_style.html
import seaborn as sns
sns.set_style("white")
sns.set_context("paper", font_scale=0.9)

# 从 numpy_ml.trees 模块中导入梯度提升决策树、决策树、随机森林类
from numpy_ml.trees import GradientBoostedDecisionTree, DecisionTree, RandomForest

# 定义一个绘图函数
def plot():
    # 创建一个 4x4 的子图，并设置图表大小为 10x10
    fig, axes = plt.subplots(4, 4)
    fig.set_size_inches(10, 10)
    # 保存图表为 plot.png，分辨率为 300 dpi
    plt.savefig("plot.png", dpi=300)
    # 关闭所有图表
    plt.close("all")
```