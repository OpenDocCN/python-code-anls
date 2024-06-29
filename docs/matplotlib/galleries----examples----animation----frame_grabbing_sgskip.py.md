# `D:\src\scipysrc\matplotlib\galleries\examples\animation\frame_grabbing_sgskip.py`

```py
"""
==============
Frame grabbing
==============

Use a MovieWriter directly to grab individual frames and write them to a
file.  This avoids any event loop integration, and thus works even with the Agg
backend.  This is not recommended for use in an interactive setting.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

import numpy as np

import matplotlib

# 设置 matplotlib 使用 "Agg" 后端，用于无需事件循环的图形绘制和动画生成
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter

# 设定随机种子以保证结果的可复现性
np.random.seed(19680801)

# 定义动画的元数据，包括标题、艺术家和注释
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')

# 创建一个 FFMpegWriter 对象，设置帧率为 15，并传入元数据
writer = FFMpegWriter(fps=15, metadata=metadata)

# 创建一个新的图形窗口
fig = plt.figure()
# 在图形中绘制一条空的黑色线条
l, = plt.plot([], [], 'k-o')

# 设置 x 和 y 轴的范围
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# 初始化起始点坐标
x0, y0 = 0, 0

# 使用 with 语句调用 writer.saving 方法，保存动画到文件 "writer_test.mp4"，总共 100 帧
with writer.saving(fig, "writer_test.mp4", 100):
    # 循环生成动画的每一帧
    for i in range(100):
        # 更新 x0 和 y0 的值，模拟随机漫步
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        # 设置线条数据为当前的 x0 和 y0
        l.set_data(x0, y0)
        # 抓取当前帧并添加到动画中
        writer.grab_frame()
```