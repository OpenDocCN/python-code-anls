# `.\numpy\doc\source\user\plots\matplotlib2.py`

```
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块，命名为plt
import numpy as np  # 导入numpy数值计算库，命名为np

x = np.linspace(0, 5, 20)  # 在0到5之间生成20个等间距的数值，作为x轴数据
y = np.linspace(0, 10, 20)  # 在0到10之间生成20个等间距的数值，作为y轴数据

plt.plot(x, y, 'purple')  # 绘制以(x, y)为坐标的紫色线条，用于展示连接的数据点
plt.plot(x, y, 'o')       # 绘制以(x, y)为坐标的散点，用圆圈表示每个数据点
plt.show()  # 显示图形，展示绘制的图像
```