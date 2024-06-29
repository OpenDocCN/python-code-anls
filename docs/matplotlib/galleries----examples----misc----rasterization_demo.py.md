# `D:\src\scipysrc\matplotlib\galleries\examples\misc\rasterization_demo.py`

```py
"""
=================================
Rasterization for vector graphics
=================================

Rasterization converts vector graphics into a raster image (pixels). It can
speed up rendering and produce smaller files for large data sets, but comes
at the cost of a fixed resolution.

Whether rasterization should be used can be specified per artist.  This can be
useful to reduce the file size of large artists, while maintaining the
advantages of vector graphics for other artists such as the Axes
and text.  For instance a complicated `~.Axes.pcolormesh` or
`~.Axes.contourf` can be made significantly simpler by rasterizing.
Setting rasterization only affects vector backends such as PDF, SVG, or PS.

Rasterization is disabled by default. There are two ways to enable it, which
can also be combined:

- Set `~.Artist.set_rasterized` on individual artists, or use the keyword
  argument *rasterized* when creating the artist.
- Set `.Axes.set_rasterization_zorder` to rasterize all artists with a zorder
  less than the given value.

The storage size and the resolution of the rasterized artist is determined by
its physical size and the value of the ``dpi`` parameter passed to
`~.Figure.savefig`.

.. note::

    The image of this example shown in the HTML documentation is not a vector
    graphic. Therefore, it cannot illustrate the rasterization effect. Please
    run this example locally and check the generated graphics files.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于生成数据

d = np.arange(100).reshape(10, 10)  # 创建一个 10x10 的数组作为颜色映射的值
x, y = np.meshgrid(np.arange(11), np.arange(11))  # 生成网格坐标

theta = 0.25*np.pi
xx = x*np.cos(theta) - y*np.sin(theta)  # 绕原点旋转 x 坐标
yy = x*np.sin(theta) + y*np.cos(theta)  # 绕原点旋转 y 坐标

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, layout="constrained")

# pcolormesh without rasterization
ax1.set_aspect(1)  # 设置纵横比为1
ax1.pcolormesh(xx, yy, d)  # 使用 pcolormesh 绘制颜色图
ax1.set_title("No Rasterization")  # 设置子图标题为 "No Rasterization"

# pcolormesh with rasterization; enabled by keyword argument
ax2.set_aspect(1)  # 设置纵横比为1
ax2.set_title("Rasterization")  # 设置子图标题为 "Rasterization"
ax2.pcolormesh(xx, yy, d, rasterized=True)  # 使用 rasterized=True 启用栅格化

# pcolormesh with an overlaid text without rasterization
ax3.set_aspect(1)  # 设置纵横比为1
ax3.pcolormesh(xx, yy, d)  # 使用 pcolormesh 绘制颜色图
ax3.text(0.5, 0.5, "Text", alpha=0.2,  # 添加文本 "Text"，设置透明度为0.2
         va="center", ha="center", size=50, transform=ax3.transAxes)  # 设置文本属性
ax3.set_title("No Rasterization")  # 设置子图标题为 "No Rasterization"

# pcolormesh with an overlaid text without rasterization; enabled by zorder.
# Setting the rasterization zorder threshold to 0 and a negative zorder on the
# pcolormesh rasterizes it. All artists have a non-negative zorder by default,
# so they (e.g. the text here) are not affected.
ax4.set_aspect(1)  # 设置纵横比为1
m = ax4.pcolormesh(xx, yy, d, zorder=-10)  # 使用 zorder 设置图层顺序，并启用栅格化
ax4.text(0.5, 0.5, "Text", alpha=0.2,  # 添加文本 "Text"，设置透明度为0.2
         va="center", ha="center", size=50, transform=ax4.transAxes)  # 设置文本属性
ax4.set_rasterization_zorder(0)  # 设置栅格化的 zorder 阈值为0
ax4.set_title("Rasterization z$<-10$")  # 设置子图标题为 "Rasterization z$<-10$"

# Save files in pdf and eps format
plt.savefig("test_rasterization.pdf", dpi=150)  # 将图保存为 pdf 格式，分辨率为150 dpi
# 保存当前图形为 EPS 格式，设置分辨率为 150 DPI
plt.savefig("test_rasterization.eps", dpi=150)

# 检查是否启用了 LaTeX 渲染文本，如果没有，则保存为 SVG 格式，分辨率为 150 DPI
if not plt.rcParams["text.usetex"]:
    plt.savefig("test_rasterization.svg", dpi=150)
    # SVG 后端目前忽略 DPI 设置

# %%
#
# .. admonition:: References
#
#    在本示例中展示了以下函数、方法、类和模块的使用:
#
#    - `matplotlib.artist.Artist.set_rasterized`
#    - `matplotlib.axes.Axes.set_rasterization_zorder`
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
```