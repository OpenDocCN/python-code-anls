# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\rotate_axes3d_sgskip.py`

```
"""
==================
Rotating a 3D plot
==================

A very simple animation of a rotating 3D plot about all three axes.

See :doc:`wire3d_animation_sgskip` for another example of animating a 3D plot.

(This example is skipped when building the documentation gallery because it
intentionally takes a long time to run)
"""

# 导入 matplotlib 的 pyplot 模块，并简称为 plt
import matplotlib.pyplot as plt

# 从 mpl_toolkits.mplot3d 模块中导入 axes3d 子模块
from mpl_toolkits.mplot3d import axes3d

# 创建一个新的图形窗口
fig = plt.figure()

# 在图形窗口中添加一个 3D 坐标轴
ax = fig.add_subplot(projection='3d')

# 获取一些示例数据，并绘制一个基本的线框图
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

# 设置坐标轴的标签
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# 旋转坐标轴并更新画面
for angle in range(0, 360*4 + 1):
    # 将角度规范化到 [-180, 180] 的范围，便于显示
    angle_norm = (angle + 180) % 360 - 180

    # 根据角度进行不同轴向的旋转
    elev = azim = roll = 0
    if angle <= 360:
        elev = angle_norm
    elif angle <= 360*2:
        azim = angle_norm
    elif angle <= 360*3:
        roll = angle_norm
    else:
        elev = azim = roll = angle_norm

    # 更新坐标轴的视角和标题
    ax.view_init(elev, azim, roll)
    plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

    # 绘制更新后的画面，并暂停一小段时间以展示动画效果
    plt.draw()
    plt.pause(.001)
```