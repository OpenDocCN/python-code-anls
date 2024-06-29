# `D:\src\scipysrc\matplotlib\galleries\examples\specialty_plots\leftventricle_bullseye.py`

```py
"""
=======================
Left ventricle bullseye
=======================

This example demonstrates how to create the 17 segment model for the left
ventricle recommended by the American Heart Association (AHA).

.. redirect-from:: /gallery/specialty_plots/leftventricle_bulleye

See also the :doc:`/gallery/pie_and_polar_charts/nested_pie` example.
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入numpy数值计算库

import matplotlib as mpl  # 导入matplotlib的配置模块


def bullseye_plot(ax, data, seg_bold=None, cmap="viridis", norm=None):
    """
    Bullseye representation for the left ventricle.

    Parameters
    ----------
    ax : Axes
        绘图的坐标轴
    data : list[float]
        包含17个区段强度值的列表
    seg_bold : list[int], optional
        需要突出显示的区段列表
    cmap : colormap, default: "viridis"
        数据的颜色映射
    norm : Normalize or None, optional
        数据的归一化器

    Notes
    -----
    This function creates the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and
        nomenclature for tomographic imaging of the heart",
        Circulation, vol. 105, no. 4, pp. 539-542, 2002.
    """

    # 将数据展平为一维数组
    data = np.ravel(data)
    if seg_bold is None:
        seg_bold = []  # 如果未提供需要突出显示的区段列表，则设为空列表
    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())  # 如果未提供归一化器，则使用默认的归一化范围

    r = np.linspace(0.2, 1, 4)  # 在径向上均匀分布4个半径值

    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])  # 设置坐标轴的范围和刻度标签
    ax.grid(False)  # 移除网格线

    # 填充区段 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
            (16, 17, 0, r[0]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               color=cmap(norm(data[start:stop])))

    # 现在，绘制区段的边界。为了使外部加粗边界不被内部区段覆盖，所有边界在填充所有区段后分开绘制。
    # 我们还禁用了裁剪，以防影响最外侧区段边缘。
    # 绘制区段 1-6, 7-12, 13-16 的边界。
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="k", linewidth=[
                   4 if i + 1 in seg_bold else 2 for i in range(start, stop)])
    # 绘制区段 17 的边界，此处需要使用 plot() 不同方式绘制边界。
    # 在 ax 对象上绘制一条水平线，x 轴范围从 0 到 2*pi，y 值始终为 r[0]
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
            linewidth=(4 if 17 in seg_bold else 2))
# Create the fake data
data = np.arange(17) + 1


# Make a figure and Axes with dimensions as desired.
fig = plt.figure(figsize=(10, 5), layout="constrained")
fig.get_layout_engine().set(wspace=.1, w_pad=.2)
axs = fig.subplots(1, 3, subplot_kw=dict(projection='polar'))
fig.canvas.manager.set_window_title('Left Ventricle Bulls Eyes (AHA)')


# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=1, vmax=17)
# Create an empty ScalarMappable to set the colorbar's colormap and norm.
# The following gives a basic continuous colorbar with ticks and labels.
# 创建一个空的 ScalarMappable 对象来设置颜色条的颜色映射和归一化。
# 下面的代码创建一个带有刻度和标签的基本连续颜色条。
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
             cax=axs[0].inset_axes([0, -.15, 1, .1]),
             orientation='horizontal', label='Some units')


# And again for the second colorbar.
cmap2 = mpl.cm.cool
norm2 = mpl.colors.Normalize(vmin=1, vmax=17)
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap2, norm=norm2),
             cax=axs[1].inset_axes([0, -.15, 1, .1]),
             orientation='horizontal', label='Some other units')


# The second example illustrates the use of a ListedColormap, a
# BoundaryNorm, and extended ends to show the "over" and "under"
# value colors.
# 第二个例子展示了如何使用 ListedColormap、BoundaryNorm 和扩展的颜色以展示“超过”和“低于”值的颜色。
cmap3 = (mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
         .with_extremes(over='0.35', under='0.75'))
# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
# 如果使用 ListedColormap，bounds 数组的长度必须比颜色列表的长度大一。
# bounds 必须单调递增。
bounds = [2, 3, 7, 9, 15]
norm3 = mpl.colors.BoundaryNorm(bounds, cmap3.N)
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap3, norm=norm3),
             cax=axs[2].inset_axes([0, -.15, 1, .1]),
             extend='both',
             ticks=bounds,  # optional
             spacing='proportional',
             orientation='horizontal',
             label='Discrete intervals, some other units')


# Create the 17 segment model
bullseye_plot(axs[0], data, cmap=cmap, norm=norm)
axs[0].set_title('Bulls Eye (AHA)')

bullseye_plot(axs[1], data, cmap=cmap2, norm=norm2)
axs[1].set_title('Bulls Eye (AHA)')

bullseye_plot(axs[2], data, seg_bold=[3, 5, 6, 11, 12, 16],
              cmap=cmap3, norm=norm3)
axs[2].set_title('Segments [3, 5, 6, 11, 12, 16] in bold')

plt.show()
```