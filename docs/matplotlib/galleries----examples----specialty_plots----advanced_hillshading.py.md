# `D:\src\scipysrc\matplotlib\galleries\examples\specialty_plots\advanced_hillshading.py`

```
"""
===========
Hillshading
===========

Demonstrates a few common tricks with shaded plots.
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 库

from matplotlib.colors import LightSource, Normalize  # 从 matplotlib 的 colors 模块导入 LightSource 和 Normalize 类


def display_colorbar():
    """Display a correct numeric colorbar for a shaded plot."""
    # 创建一个网格数据
    y, x = np.mgrid[-4:2:200j, -4:2:200j]
    # 根据网格数据计算高度值
    z = 10 * np.cos(x**2 + y**2)

    # 设置色彩映射
    cmap = plt.cm.copper
    # 创建一个光源对象
    ls = LightSource(315, 45)
    # 对高度值进行着色处理
    rgb = ls.shade(z, cmap)

    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上显示着色后的图像
    ax.imshow(rgb, interpolation='bilinear')

    # 使用一个代理艺术家来显示色彩条
    im = ax.imshow(z, cmap=cmap)
    im.remove()
    # 添加色彩条到轴上
    fig.colorbar(im, ax=ax)

    # 设置图表标题
    ax.set_title('Using a colorbar with a shaded plot', size='x-large')


def avoid_outliers():
    """Use a custom norm to control the displayed z-range of a shaded plot."""
    # 创建一个网格数据
    y, x = np.mgrid[-4:2:200j, -4:2:200j]
    # 根据网格数据计算高度值
    z = 10 * np.cos(x**2 + y**2)

    # 向数据中添加一些异常值
    z[100, 105] = 2000
    z[120, 110] = -9000

    # 创建光源对象
    ls = LightSource(315, 45)
    # 创建包含两个子图的图形对象
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4.5))

    # 对数据进行着色处理并在第一个子图上显示
    rgb = ls.shade(z, plt.cm.copper)
    ax1.imshow(rgb, interpolation='bilinear')
    ax1.set_title('Full range of data')

    # 手动设置显示范围，并在第二个子图上显示
    rgb = ls.shade(z, plt.cm.copper, vmin=-10, vmax=10)
    ax2.imshow(rgb, interpolation='bilinear')
    ax2.set_title('Manually set range')

    # 设置整体标题
    fig.suptitle('Avoiding Outliers in Shaded Plots', size='x-large')


def shade_other_data():
    """Demonstrates displaying different variables through shade and color."""
    # 创建一个网格数据
    y, x = np.mgrid[-4:2:200j, -4:2:200j]
    # 根据网格数据计算两个不同的数据集
    z1 = np.sin(x**2)  # 用于山体阴影的数据
    z2 = np.cos(x**2 + y**2)  # 用于着色的数据

    # 创建一个规范化对象
    norm = Normalize(z2.min(), z2.max())
    # 设置色彩映射
    cmap = plt.cm.RdBu

    # 创建光源对象
    ls = LightSource(315, 45)
    # 对颜色和阴影数据进行处理并在图上显示
    rgb = ls.shade_rgb(cmap(norm(z2)), z1)

    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上显示着色后的图像
    ax.imshow(rgb, interpolation='bilinear')
    # 设置图表标题
    ax.set_title('Shade by one variable, color by another', size='x-large')


# 调用三个函数来展示示例
display_colorbar()
avoid_outliers()
shade_other_data()
# 显示图形
plt.show()
```