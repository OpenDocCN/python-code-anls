# `D:\src\scipysrc\matplotlib\galleries\examples\animation\simple_scatter.py`

```
"""
=============================
Animated scatter saved as GIF
=============================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np
# 导入 matplotlib.animation 库中的 animation 模块
import matplotlib.animation as animation

# 创建一个图形 fig 和一个轴 ax
fig, ax = plt.subplots()
# 设置 x 轴的范围为 0 到 10
ax.set_xlim([0, 10])

# 在轴 ax 上创建一个散点图 scat，并设置初始位置为 (1, 0)
scat = ax.scatter(1, 0)
# 创建一个从 0 到 10 的均匀分布的数组 x
x = np.linspace(0, 10)

# 定义动画函数 animate，每次更新散点的位置
def animate(i):
    # 设置散点的偏移位置为 (x[i], 0)
    scat.set_offsets((x[i], 0))
    return scat,

# 创建动画对象 ani，使用 FuncAnimation 函数，设定重复播放，总帧数为 len(x) - 1，每帧间隔 50 毫秒
ani = animation.FuncAnimation(fig, animate, repeat=True,
                              frames=len(x) - 1, interval=50)

# 展示动画
plt.show()

# 若要将动画保存为 GIF，可以使用 PillowWriter，以下是保存代码：
# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('scatter.gif', writer=writer)
```