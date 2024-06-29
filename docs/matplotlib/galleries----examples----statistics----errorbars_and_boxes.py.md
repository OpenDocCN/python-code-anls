# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\errorbars_and_boxes.py`

```
# 导入 matplotlib 的 pyplot 模块，并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并命名为 np
import numpy as np

# 导入 PatchCollection 和 Rectangle 类
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# 数据点数量
n = 5

# 创建随机种子以便复现结果
np.random.seed(19680801)
# 生成 x 轴数据
x = np.arange(0, n, 1)
# 生成随机的 y 轴数据
y = np.random.rand(n) * 5.

# 创建虚拟的 x 方向和 y 方向的误差数据
xerr = np.random.rand(2, n) + 0.1
yerr = np.random.rand(2, n) + 0.2

# 定义函数 make_error_boxes，用于在每个数据点处根据误差创建方框
def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='none', alpha=0.5):
    # 遍历数据点，为每个数据点创建由误差定义的方框
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # 创建 PatchCollection，设置颜色和透明度
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # 将 PatchCollection 添加到 Axes 对象中
    ax.add_collection(pc)

    # 绘制误差条
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='none', ecolor='k')

    return artists


# 创建图和 Axes 对象
fig, ax = plt.subplots(1)

# 调用函数 make_error_boxes 创建误差方框
_ = make_error_boxes(ax, x, y, xerr, yerr)

# 显示图形
plt.show()
```