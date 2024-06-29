# `D:\src\scipysrc\matplotlib\galleries\examples\animation\unchained.py`

```
"""
========================
MATPLOTLIB **UNCHAINED**
========================

Comparative path demonstration of frequency from a fake signal of a pulsar
(mostly known because of the cover for Joy Division's Unknown Pleasures).

Author: Nicolas P. Rougier

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入 matplotlib 库中的 pyplot 模块并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 matplotlib 库中的 animation 模块
import matplotlib.animation as animation

# 设置随机数种子以确保可重现性
np.random.seed(19680801)

# 创建一个带有黑色背景的新 Figure 对象
fig = plt.figure(figsize=(8, 8), facecolor='black')

# 在 Figure 对象中添加一个无边框的子图
ax = plt.subplot(frameon=False)

# 生成随机数据
data = np.random.uniform(0, 1, (64, 75))
X = np.linspace(-1, 1, data.shape[-1])
G = 1.5 * np.exp(-4 * X ** 2)

# 生成线条图形
lines = []
for i in range(len(data)):
    # 稍微减少 X 范围以获得一种简单的透视效果
    xscale = 1 - i / 200.
    # 线条宽度也相应地减少（底部线条较粗）
    lw = 1.5 - i / 100.0
    line, = ax.plot(xscale * X, i + G * data[i], color="w", lw=lw)
    lines.append(line)

# 设置 y 轴的限制（否则第一条线因为线条粗细会被裁切）
ax.set_ylim(-1, 70)

# 不显示刻度
ax.set_xticks([])
ax.set_yticks([])

# 使用两部分标题以获得不同的字体粗细
ax.text(0.5, 1.0, "MATPLOTLIB ", transform=ax.transAxes,
        ha="right", va="bottom", color="w",
        family="sans-serif", fontweight="light", fontsize=16)
ax.text(0.5, 1.0, "UNCHAINED", transform=ax.transAxes,
        ha="left", va="bottom", color="w",
        family="sans-serif", fontweight="bold", fontsize=16)

# 定义更新函数，用于更新数据
def update(*args):
    # 将所有数据向右移动一列
    data[:, 1:] = data[:, :-1]

    # 填充新的值
    data[:, 0] = np.random.uniform(0, 1, len(data))

    # 更新数据
    for i in range(len(data)):
        lines[i].set_ydata(i + G * data[i])

    # 返回更新后的图形对象
    return lines

# 构造动画，使用 update 函数作为动画的更新函数，并设置更新间隔和保存帧数
anim = animation.FuncAnimation(fig, update, interval=10, save_count=100)

# 显示动画
plt.show()
```