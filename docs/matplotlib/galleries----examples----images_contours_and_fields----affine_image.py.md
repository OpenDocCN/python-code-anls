# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\affine_image.py`

```py
"""
============================
Affine transform of an image
============================


Prepending an affine transformation (`~.transforms.Affine2D`) to the :ref:`data
transform <data-coords>` of an image allows to manipulate the image's shape and
orientation.  This is an example of the concept of :ref:`transform chaining
<transformation-pipeline>`.

The image of the output should have its boundary match the dashed yellow
rectangle.
"""

# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np

# 导入用于图形变换的模块
import matplotlib.transforms as mtransforms

# 定义获取图像数据的函数
def get_image():
    delta = 0.25
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2)
    return Z

# 定义绘制图像的函数
def do_plot(ax, Z, transform):
    # 绘制图像
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   extent=[-2, 4, -3, 2], clip_on=True)

    # 应用数据变换
    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # 显示图像的预期范围
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)


# 准备图像和图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
Z = get_image()

# 对图像进行旋转
do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(30))

# 对图像进行倾斜
do_plot(ax2, Z, mtransforms.Affine2D().skew_deg(30, 15))

# 对图像进行缩放和反射
do_plot(ax3, Z, mtransforms.Affine2D().scale(-1, .5))

# 对图像进行旋转、倾斜、缩放和平移
do_plot(ax4, Z, mtransforms.Affine2D().
        rotate_deg(30).skew_deg(30, 15).scale(-1, .5).translate(.5, -1))

plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.transforms.Affine2D`
```