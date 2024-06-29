# `D:\src\scipysrc\matplotlib\galleries\examples\scales\log_bar.py`

```
"""
=======
Log Bar
=======

Plotting a bar chart with a logarithmic y-axis.
"""
# 导入matplotlib.pyplot模块，用于绘图
import matplotlib.pyplot as plt
# 导入numpy模块，用于处理数据
import numpy as np

# 定义数据集
data = ((3, 1000), (10, 3), (100, 30), (500, 800), (50, 1))

# 获取数据维度
dim = len(data[0])
# 设置每个条形图的宽度
w = 0.75
dimw = w / dim

# 创建图形和坐标轴对象
fig, ax = plt.subplots()

# 生成x轴的位置
x = np.arange(len(data))

# 遍历数据集中的每个维度，绘制条形图
for i in range(len(data[0])):
    # 提取当前维度的数据
    y = [d[i] for d in data]
    # 绘制条形图
    b = ax.bar(x + i * dimw, y, dimw, bottom=0.001)

# 设置x轴刻度及其标签
ax.set_xticks(x + dimw / 2, labels=map(str, x))
# 设置y轴为对数坐标轴
ax.set_yscale('log')

# 设置x轴和y轴标签
ax.set_xlabel('x')
ax.set_ylabel('y')

# 显示图形
plt.show()
```