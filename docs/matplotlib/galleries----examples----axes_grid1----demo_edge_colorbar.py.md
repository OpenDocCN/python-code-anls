# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_edge_colorbar.py`

```py
"""
===============================
Per-row or per-column colorbars
===============================

This example shows how to use one common colorbar for each row or column
of an image grid.
"""

import matplotlib.pyplot as plt

from matplotlib import cbook
from mpl_toolkits.axes_grid1 import AxesGrid


def get_demo_image():
    # 从示例数据集中获取一个15x15的二维数组作为图像数据
    z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")
    return z, (-3, 4, -4, 3)


def demo_bottom_cbar(fig):
    """
    A grid of 2x2 images with a colorbar for each column.
    """
    # 创建一个2x2的图像网格，每列有一个颜色条
    grid = AxesGrid(fig, 121,  # 类似于subplot(121)
                    nrows_ncols=(2, 2),
                    axes_pad=0.10,
                    share_all=True,
                    label_mode="1",
                    cbar_location="bottom",
                    cbar_mode="edge",
                    cbar_pad=0.25,
                    cbar_size="15%",
                    direction="column"
                    )

    Z, extent = get_demo_image()
    cmaps = ["autumn", "summer"]
    for i in range(4):
        # 在每个子图中显示图像，并设置颜色映射
        im = grid[i].imshow(Z, extent=extent, cmap=cmaps[i//2])
        if i % 2:
            # 如果是奇数列，为该列添加颜色条
            grid.cbar_axes[i//2].colorbar(im)

    for cax in grid.cbar_axes:
        # 设置每个颜色条的标签
        cax.axis[cax.orientation].set_label("Bar")

    # 由于 share_all=True，以下设置影响所有子图
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])


def demo_right_cbar(fig):
    """
    A grid of 2x2 images. Each row has its own colorbar.
    """
    # 创建一个2x2的图像网格，每行有一个颜色条
    grid = AxesGrid(fig, 122,  # 类似于subplot(122)
                    nrows_ncols=(2, 2),
                    axes_pad=0.10,
                    label_mode="1",
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="7%",
                    cbar_pad="2%",
                    )
    Z, extent = get_demo_image()
    cmaps = ["spring", "winter"]
    for i in range(4):
        # 在每个子图中显示图像，并设置颜色映射
        im = grid[i].imshow(Z, extent=extent, cmap=cmaps[i//2])
        if i % 2:
            # 如果是奇数行，为该行添加颜色条
            grid.cbar_axes[i//2].colorbar(im)

    for cax in grid.cbar_axes:
        # 设置每个颜色条的标签
        cax.axis[cax.orientation].set_label('Foo')

    # 由于 share_all=True，以下设置影响所有子图
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])


fig = plt.figure()

# 在图形上展示具有不同位置颜色条的子图网格
demo_bottom_cbar(fig)
demo_right_cbar(fig)

plt.show()
```