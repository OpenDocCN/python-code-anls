# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_axis.py`

```py
import numpy as np  # 导入 NumPy 库，用于处理数值数组

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
from matplotlib.axis import XTick  # 从 matplotlib.axis 模块导入 XTick 类，用于处理坐标轴刻度


def test_tick_labelcolor_array():
    # 测试：创建一个 Tick 对象，使用数组作为 labelcolor 的颜色值
    ax = plt.axes()  # 创建一个图轴对象
    XTick(ax, 0, labelcolor=np.array([1, 0, 0, 1]))  # 创建一个 XTick 对象，设置标签颜色为红色（RGBA值为[1, 0, 0, 1]）
```