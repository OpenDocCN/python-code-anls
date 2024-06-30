# `D:\src\scipysrc\seaborn\examples\many_pairwise_correlations.py`

```
"""
Plotting a diagonal correlation matrix
======================================

_thumb: .3, .6
"""
# 导入必要的库和模块
from string import ascii_letters  # 导入字符串模块中的字母序列
import numpy as np  # 导入NumPy库
import pandas as pd  # 导入Pandas库
import seaborn as sns  # 导入Seaborn库
import matplotlib.pyplot as plt  # 导入Matplotlib库

# 设定Seaborn的主题样式为白色背景
sns.set_theme(style="white")

# 生成一个大的随机数据集
rs = np.random.RandomState(33)  # 使用随机种子33生成一个随机状态对象
d = pd.DataFrame(data=rs.normal(size=(100, 26)),  # 生成一个100行26列的随机正态分布数据DataFrame
                 columns=list(ascii_letters[26:]))

# 计算数据集的相关系数矩阵
corr = d.corr()

# 生成一个上三角形式的掩码数组，用于在热图中隐藏下三角部分
mask = np.triu(np.ones_like(corr, dtype=bool))

# 设置Matplotlib图形的大小为11x9英寸
f, ax = plt.subplots(figsize=(11, 9))

# 生成一个自定义的发散调色板
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# 使用掩码和正确的纵横比绘制热图
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```