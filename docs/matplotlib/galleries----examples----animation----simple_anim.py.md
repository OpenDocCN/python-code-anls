# `D:\src\scipysrc\matplotlib\galleries\examples\animation\simple_anim.py`

```py
"""
==================
Animated line plot
==================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入 matplotlib 库中需要的模块
import matplotlib.pyplot as plt
import numpy as np

# 导入动画模块
import matplotlib.animation as animation

# 创建一个图形窗口和一个轴对象
fig, ax = plt.subplots()

# 生成 x 数据，范围是 [0, 2π)，步长为 0.01
x = np.arange(0, 2*np.pi, 0.01)

# 绘制初始的正弦波线
line, = ax.plot(x, np.sin(x))


# 定义更新数据的函数，i 是动画帧数
def animate(i):
    # 更新线的 y 数据为 sin(x + i / 50)
    line.set_ydata(np.sin(x + i / 50))  # update the data.
    return line,

# 创建动画对象，fig 是图形对象，animate 是更新函数，interval 是更新间隔，blit=True 表示仅绘制变化的部分，save_count 是保存帧数
ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)

# 显示动画
plt.show()

# 要保存动画为视频文件，可以使用 ani.save("movie.mp4")，也可以指定写入器参数进行更详细的配置
#
# ani.save("movie.mp4")
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
```