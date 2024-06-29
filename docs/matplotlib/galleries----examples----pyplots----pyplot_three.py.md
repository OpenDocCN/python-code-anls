# `D:\src\scipysrc\matplotlib\galleries\examples\pyplots\pyplot_three.py`

```py
"""
===========================
Multiple lines using pyplot
===========================

Plot three datasets with a single call to `~matplotlib.pyplot.plot`.
"""

# 导入 matplotlib 的 pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并简写为 np
import numpy as np

# 创建一个包含从0到4.8（不包括5）的数列，间隔为0.2
t = np.arange(0., 5., 0.2)

# 使用单个 plot 调用绘制三组数据：
# 第一组：红色虚线 ('r--')，表示 t vs. t
# 第二组：蓝色方块 ('bs')，表示 t vs. t^2
# 第三组：绿色三角形 ('g^')，表示 t vs. t^3
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
```