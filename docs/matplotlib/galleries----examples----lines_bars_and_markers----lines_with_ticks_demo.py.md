# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\lines_with_ticks_demo.py`

```
"""
==============================
Lines with a ticked patheffect
==============================

Ticks can be added along a line to mark one side as a barrier using
`~matplotlib.patheffects.TickedStroke`.  You can control the angle,
spacing, and length of the ticks.

The ticks will also appear appropriately in the legend.

"""

# 导入 matplotlib 库中的 pyplot 模块并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np
# 从 matplotlib 中导入 patheffects 模块
from matplotlib import patheffects

# 创建一个图形和轴对象，设置图形大小为 6x6
fig, ax = plt.subplots(figsize=(6, 6))

# 绘制一条直线，起点 (0, 0) 终点 (1, 1)，标签为 "Line"，
# 应用 path_effects，使用 withTickedStroke 方法，设置间距为 7，角度为 135
ax.plot([0, 1], [0, 1], label="Line",
        path_effects=[patheffects.withTickedStroke(spacing=7, angle=135)])

# 绘制一条曲线，使用 numpy 生成 0 到 1 之间的 101 个点作为 x 坐标，
# 对应的 y 坐标为 0.3*sin(8*x) + 0.4，标签为 "Curve"，
# 应用 path_effects，使用 withTickedStroke 方法，默认参数
ax.plot(x, y, label="Curve", path_effects=[patheffects.withTickedStroke()])

# 添加图例
ax.legend()

# 显示图形
plt.show()
```