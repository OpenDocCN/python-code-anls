# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_curvelinear_grid2.py`

```py
"""
======================
Demo CurveLinear Grid2
======================

Custom grid and ticklines.

This example demonstrates how to use GridHelperCurveLinear to define
custom grids and ticklines by applying a transformation on the grid.
As showcase on the plot, a 5x5 matrix is displayed on the Axes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axisartist.axislines import Axes  # 导入绘图的轴类
from mpl_toolkits.axisartist.grid_finder import (ExtremeFinderSimple,  # 导入坐标网格相关类
                                                 MaxNLocator)
from mpl_toolkits.axisartist.grid_helper_curvelinear import \  # 导入曲线线性网格助手类
    GridHelperCurveLinear


def curvelinear_test1(fig):
    """Grid for custom transform."""
    
    # 定义转换函数及其反函数
    def tr(x, y):
        return np.sign(x)*abs(x)**0.5, y

    def inv_tr(x, y):
        return np.sign(x)*x**2, y
    
    # 创建曲线线性网格助手对象
    grid_helper = GridHelperCurveLinear(
        (tr, inv_tr),  # 设置转换函数和反函数
        extreme_finder=ExtremeFinderSimple(20, 20),  # 设置极值点查找器
        grid_locator1=MaxNLocator(nbins=6),  # 设置第一个坐标轴的网格定位器
        grid_locator2=MaxNLocator(nbins=6)   # 设置第二个坐标轴的网格定位器
    )

    # 在图形上添加子图，使用自定义的轴和网格助手
    ax1 = fig.add_subplot(axes_class=Axes, grid_helper=grid_helper)
    # ax1将根据给定的转换函数（+轴的transData）定义刻度和网格线。
    # 注意：轴本身的转换（即transData）不受给定转换函数的影响。

    # 在子图上显示一个5x5的矩阵，设置颜色映射和原点位置
    ax1.imshow(np.arange(25).reshape(5, 5),
               vmax=50, cmap=plt.cm.gray_r, origin="lower")


if __name__ == "__main__":
    # 创建一个图形对象，设置尺寸
    fig = plt.figure(figsize=(7, 4))
    # 调用curvelinear_test1函数，传入图形对象，进行绘制
    curvelinear_test1(fig)
    # 显示绘制的图形
    plt.show()
```