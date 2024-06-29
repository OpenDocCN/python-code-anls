# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_imagegrid_aspect.py`

```
"""
=========================================
Setting a fixed aspect on ImageGrid cells
=========================================
"""

# 导入需要的库
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# 创建一个新的图形对象
fig = plt.figure()

# 创建第一个图形网格对象，包含2x2个子图，间距为0.1，保持纵横比，共享所有轴
grid1 = ImageGrid(fig, 121, (2, 2), axes_pad=0.1,
                  aspect=True, share_all=True)
# 设置网格1中第0和第1个子图的纵横比为2
for i in [0, 1]:
    grid1[i].set_aspect(2)

# 创建第二个图形网格对象，包含2x2个子图，间距为0.1，保持纵横比，共享所有轴
grid2 = ImageGrid(fig, 122, (2, 2), axes_pad=0.1,
                  aspect=True, share_all=True)
# 设置网格2中第1和第3个子图的纵横比为2
for i in [1, 3]:
    grid2[i].set_aspect(2)

# 显示绘制的图形
plt.show()
```