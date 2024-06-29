# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\shading_example.py`

```
"""
===============
Shading example
===============

Example showing how to make shaded relief plots like Mathematica_ or
`Generic Mapping Tools`_.

.. _Mathematica: http://reference.wolfram.com/mathematica/ref/ReliefPlot.html
.. _Generic Mapping Tools: https://www.generic-mapping-tools.org/
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，并简写为plt
import numpy as np  # 导入numpy库，并简写为np

from matplotlib import cbook  # 从matplotlib中导入cbook模块
from matplotlib.colors import LightSource  # 从matplotlib.colors中导入LightSource类


def main():
    # Test data
    x, y = np.mgrid[-5:5:0.05, -5:5:0.05]  # 生成二维坐标网格
    z = 5 * (np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2))  # 计算测试数据 z

    # 从示例数据中获取高程数据
    dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
    elev = dem['elevation']

    # 绘制两个图形，并设置标题
    fig = compare(z, plt.cm.copper)
    fig.suptitle('HSV Blending Looks Best with Smooth Surfaces', y=0.95)

    fig = compare(elev, plt.cm.gist_earth, ve=0.05)
    fig.suptitle('Overlay Blending Looks Best with Rough Surfaces', y=0.95)

    plt.show()  # 显示图形


def compare(z, cmap, ve=1):
    # Create subplots and hide ticks
    fig, axs = plt.subplots(ncols=2, nrows=2)  # 创建2x2的子图
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])  # 隐藏子图的刻度

    # Illuminate the scene from the northwest
    ls = LightSource(azdeg=315, altdeg=45)  # 创建光源对象，角度设置为315度方位，45度高度

    # 在第一个子图中显示 z 的彩色映射数据
    axs[0, 0].imshow(z, cmap=cmap)
    axs[0, 0].set(xlabel='Colormapped Data')  # 设置x轴标签为 'Colormapped Data'

    # 在第二个子图中显示 z 的阴影强度，使用灰度图显示
    axs[0, 1].imshow(ls.hillshade(z, vert_exag=ve), cmap='gray')
    axs[0, 1].set(xlabel='Illumination Intensity')  # 设置x轴标签为 'Illumination Intensity'

    # 使用HSV混合模式对 z 进行着色，并在第三个子图中显示
    rgb = ls.shade(z, cmap=cmap, vert_exag=ve, blend_mode='hsv')
    axs[1, 0].imshow(rgb)
    axs[1, 0].set(xlabel='Blend Mode: "hsv" (default)')  # 设置x轴标签为 'Blend Mode: "hsv" (default)'

    # 使用overlay混合模式对 z 进行着色，并在第四个子图中显示
    rgb = ls.shade(z, cmap=cmap, vert_exag=ve, blend_mode='overlay')
    axs[1, 1].imshow(rgb)
    axs[1, 1].set(xlabel='Blend Mode: "overlay"')  # 设置x轴标签为 'Blend Mode: "overlay"'

    return fig  # 返回绘制好的图形对象


if __name__ == '__main__':
    main()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.colors.LightSource`
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
```