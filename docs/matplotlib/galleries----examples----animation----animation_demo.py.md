# `D:\src\scipysrc\matplotlib\galleries\examples\animation\animation_demo.py`

```
"""
================
pyplot animation
================

Generating an animation by calling `~.pyplot.pause` between plotting commands.

The method shown here is only suitable for simple, low-performance use.  For
more demanding applications, look at the :mod:`.animation` module and the
examples that use it.

Note that calling `time.sleep` instead of `~.pyplot.pause` would *not* work.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入 matplotlib.pyplot 库并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np

# 设置随机数种子，以便结果可重现
np.random.seed(19680801)
# 生成一个形状为 (50, 50, 50) 的随机数据数组
data = np.random.random((50, 50, 50))

# 创建一个新的图形和子图
fig, ax = plt.subplots()

# 遍历随机数据数组的每一帧图像
for i, img in enumerate(data):
    # 清空当前子图内容
    ax.clear()
    # 在子图上显示当前帧的图像
    ax.imshow(img)
    # 设置子图的标题，显示当前帧的索引
    ax.set_title(f"frame {i}")
    # 在每帧显示后暂停一段时间，以实现动画效果
    plt.pause(0.1)
```