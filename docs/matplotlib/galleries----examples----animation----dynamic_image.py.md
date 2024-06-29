# `D:\src\scipysrc\matplotlib\galleries\examples\animation\dynamic_image.py`

```
"""
=================================================
Animated image using a precomputed list of images
=================================================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块并简称为 plt
import numpy as np  # 导入 numpy 库并简称为 np

import matplotlib.animation as animation  # 导入 matplotlib 的 animation 模块

# 创建一个图形窗口和一个轴
fig, ax = plt.subplots()

# 定义一个函数 f(x, y)，用于计算图像的值
def f(x, y):
    return np.sin(x) + np.cos(y)

# 创建 x 和 y 的数组，用于绘制图像
x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims 是一个列表的列表，每一行都是一个在当前帧中绘制的艺术家列表；
# 这里我们只是在每一帧中绘制一个艺术家，即图像。
ims = []
for i in range(60):
    x += np.pi / 15
    y += np.pi / 30
    # 创建一个图像对象并添加到 ims 列表中
    im = ax.imshow(f(x, y), animated=True)
    if i == 0:
        ax.imshow(f(x, y))  # 显示初始图像
    ims.append([im])

# 使用 ArtistAnimation 类创建动画对象 ani
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# 若要保存动画，可以使用以下代码：
# 
# ani.save("movie.mp4")
# 
# 或者可以自定义保存选项，比如使用 FFMpegWriter：
# 
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

# 显示动画
plt.show()
```