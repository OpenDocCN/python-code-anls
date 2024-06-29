# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\dollar_ticks.py`

```
"""
============
Dollar ticks
============

Use a format string to prepend dollar signs on y-axis labels.

.. redirect-from:: /gallery/pyplots/dollar_ticks
"""

# 导入matplotlib.pyplot模块，并重命名为plt
import matplotlib.pyplot as plt
# 导入numpy模块，并重命名为np
import numpy as np

# 设置随机数种子以保证可复现性
np.random.seed(19680801)

# 创建图形和子图对象
fig, ax = plt.subplots()

# 生成随机数据并绘制折线图
ax.plot(100*np.random.rand(20))

# 使用自动的StrMethodFormatter，设置y轴主要刻度的格式为美元符号加浮点数，保留两位小数
ax.yaxis.set_major_formatter('${x:1.2f}')

# 设置y轴主要刻度的参数，包括标签颜色为绿色，左侧标签不显示，右侧标签显示
ax.yaxis.set_tick_params(which='major', labelcolor='green',
                         labelleft=False, labelright=True)

# 显示图形
plt.show()
```