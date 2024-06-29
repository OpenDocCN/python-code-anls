# `.\numpy\doc\source\user\plots\meshgrid_plot.py`

```
# 导入 NumPy 库，用于科学计算
import numpy as np
# 导入 Matplotlib 库中的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt

# 创建 NumPy 数组 x 和 y，分别表示 x 和 y 轴上的坐标点
x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 2, 3, 4, 5])

# 使用 meshgrid 函数生成网格数据，xx 和 yy 是网格化后的 x 和 y 值
xx, yy = np.meshgrid(x, y)

# 使用 Matplotlib 的 pyplot 模块绘制散点图
plt.plot(xx, yy, marker='o', color='k', linestyle='none')
```