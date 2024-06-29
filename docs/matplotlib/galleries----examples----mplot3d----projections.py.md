# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\projections.py`

```
"""
========================
3D plot projection types
========================

Demonstrates the different camera projections for 3D plots, and the effects of
changing the focal length for a perspective projection. Note that Matplotlib
corrects for the 'zoom' effect of changing the focal length.

The default focal length of 1 corresponds to a Field of View (FOV) of 90 deg.
An increased focal length between 1 and infinity "flattens" the image, while a
decreased focal length between 1 and 0 exaggerates the perspective and gives
the image more apparent depth. In the limiting case, a focal length of
infinity corresponds to an orthographic projection after correction of the
zoom effect.

You can calculate focal length from a FOV via the equation:

.. math::

    1 / \\tan (\\mathrm{FOV} / 2)

Or vice versa:

.. math::

    \\mathrm{FOV} = 2 \\arctan (1 / \\mathrm{focal length})

"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块

from mpl_toolkits.mplot3d import axes3d  # 从 mpl_toolkits.mplot3d 导入 axes3d 模块

fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'})  # 创建一个包含3个子图的Figure对象和Axes3D对象的数组

# Get the test data
X, Y, Z = axes3d.get_test_data(0.05)  # 获取用于绘图的测试数据 X, Y, Z

# Plot the data
for ax in axs:
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)  # 在每个子图上绘制线框图

# Set the orthographic projection.
axs[0].set_proj_type('ortho')  # 设置第一个子图的投影类型为正交投影（ortho），对应 FOV = 0 deg
axs[0].set_title("'ortho'\nfocal_length = ∞", fontsize=10)  # 设置第一个子图的标题，显示投影类型和无限远的焦距

# Set the perspective projections
axs[1].set_proj_type('persp')  # 设置第二个子图的投影类型为透视投影（persp），对应 FOV = 90 deg
axs[1].set_title("'persp'\nfocal_length = 1 (default)", fontsize=10)  # 设置第二个子图的标题，显示投影类型和默认焦距

axs[2].set_proj_type('persp', focal_length=0.2)  # 设置第三个子图的投影类型为透视投影（persp），并指定焦距为 0.2，对应 FOV = 157.4 deg
axs[2].set_title("'persp'\nfocal_length = 0.2", fontsize=10)  # 设置第三个子图的标题，显示投影类型和指定的焦距

plt.show()  # 显示图形
```