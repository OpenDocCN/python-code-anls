# `D:\src\scipysrc\matplotlib\galleries\examples\specialty_plots\hinton_demo.py`

```
"""
===============
Hinton diagrams
===============

Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
a weight matrix): Positive and negative values are represented by white and
black squares, respectively, and the size of each square represents the
magnitude of each value.

Initial idea from David Warde-Farley on the SciPy Cookbook
"""

# 导入需要的库
import matplotlib.pyplot as plt
import numpy as np

# 定义绘制 Hinton 图的函数
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    # 如果未提供轴对象，使用当前的轴对象或者创建新的轴对象
    ax = ax if ax is not None else plt.gca()

    # 如果未指定最大权重，根据矩阵中数值的绝对值计算最大权重
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    # 设置背景颜色为灰色
    ax.patch.set_facecolor('gray')
    # 设置图像比例为等比例方框
    ax.set_aspect('equal', 'box')
    # 隐藏 x 轴和 y 轴的主刻度
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # 遍历矩阵中的每个元素的索引和值
    for (x, y), w in np.ndenumerate(matrix):
        # 根据值的正负确定颜色（正值白色，负值黑色）
        color = 'white' if w > 0 else 'black'
        # 计算方块的大小，大小与数值的绝对值成平方根关系，除以最大权重
        size = np.sqrt(abs(w) / max_weight)
        # 创建方块对象并添加到图像中
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # 自动调整视图范围
    ax.autoscale_view()
    # 反转 y 轴，使图像正向表示矩阵的上下方向
    ax.invert_yaxis()


if __name__ == '__main__':
    # 设置随机种子以便结果可重现
    np.random.seed(19680801)

    # 调用 hinton 函数，绘制一个随机的 20x20 的矩阵的 Hinton 图
    hinton(np.random.rand(20, 20) - 0.5)
    # 显示绘制的图像
    plt.show()
```