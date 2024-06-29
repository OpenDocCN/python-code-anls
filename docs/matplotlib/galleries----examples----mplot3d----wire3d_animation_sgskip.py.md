# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\wire3d_animation_sgskip.py`

```py
"""
===========================
Animate a 3D wireframe plot
===========================

A very simple "animation" of a 3D plot.  See also :doc:`rotate_axes3d_sgskip`.

(This example is skipped when building the documentation gallery because it
intentionally takes a long time to run.)
"""

# 导入必要的库
import time

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并简写为 plt
import numpy as np  # 导入 numpy 库，并简写为 np

# 创建一个新的 figure 对象
fig = plt.figure()
# 在 figure 上添加一个 3D subplot
ax = fig.add_subplot(projection='3d')

# 创建 X, Y 的网格
xs = np.linspace(-1, 1, 50)  # 在 -1 到 1 之间生成 50 个等间距的数值作为 X 轴坐标
ys = np.linspace(-1, 1, 50)  # 在 -1 到 1 之间生成 50 个等间距的数值作为 Y 轴坐标
X, Y = np.meshgrid(xs, ys)  # 使用 X, Y 轴坐标创建网格数据

# 设置 Z 轴的限制，使得每帧不重新计算
ax.set_zlim(-1, 1)

# 开始绘图循环
wframe = None  # 初始化一个变量用于存储绘制的线框图对象
tstart = time.time()  # 记录开始时间
for phi in np.linspace(0, 180. / np.pi, 100):  # 在 0 到 180 度之间均匀生成 100 个数值
    if wframe:
        wframe.remove()  # 如果已经有线框图对象存在，则先移除它

    # 生成新的数据
    Z = np.cos(2 * np.pi * X + phi) * (1 - np.hypot(X, Y))
    # 绘制新的线框图，并在继续之前暂停一小段时间
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    plt.pause(.001)

# 打印平均帧率
print('Average FPS: %f' % (100 / (time.time() - tstart)))
```