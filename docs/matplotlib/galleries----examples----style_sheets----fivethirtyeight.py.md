# `D:\src\scipysrc\matplotlib\galleries\examples\style_sheets\fivethirtyeight.py`

```py
"""
===========================
FiveThirtyEight style sheet
===========================

This shows an example of the "fivethirtyeight" styling, which
tries to replicate the styles from FiveThirtyEight.com.
"""

# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并重命名为 np
import numpy as np

# 使用 'fivethirtyeight' 样式
plt.style.use('fivethirtyeight')

# 生成从 0 到 10 的等间隔数字序列
x = np.linspace(0, 10)

# 设置随机数种子以确保结果的可重复性
np.random.seed(19680801)

# 创建一个新的图形和一个子图
fig, ax = plt.subplots()

# 绘制六条曲线，每条曲线是 sin(x) 加上不同倍数的 x 和随机噪声的结果
ax.plot(x, np.sin(x) + x + np.random.randn(50))
ax.plot(x, np.sin(x) + 0.5 * x + np.random.randn(50))
ax.plot(x, np.sin(x) + 2 * x + np.random.randn(50))
ax.plot(x, np.sin(x) - 0.5 * x + np.random.randn(50))
ax.plot(x, np.sin(x) - 2 * x + np.random.randn(50))
ax.plot(x, np.sin(x) + np.random.randn(50))

# 设置子图的标题
ax.set_title("'fivethirtyeight' style sheet")

# 显示图形
plt.show()
```