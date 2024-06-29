# `D:\src\scipysrc\matplotlib\galleries\examples\specialty_plots\sankey_links.py`

```py
"""
======================================
Long chain of connections using Sankey
======================================

Demonstrate/test the Sankey class by producing a long chain of connections.
"""

# 导入 matplotlib.pyplot 模块作为 plt
import matplotlib.pyplot as plt

# 从 matplotlib.sankey 中导入 Sankey 类
from matplotlib.sankey import Sankey

# 定义每一侧的连接数
links_per_side = 6

# 定义生成侧链的函数
def side(sankey, n=1):
    """Generate a side chain."""
    # 获取当前图表中已有图表数量
    prior = len(sankey.diagrams)
    # 循环生成侧链
    for i in range(0, 2*n, 2):
        # 添加 Sankey 图表，表示从前一个图表到当前图表的连接
        sankey.add(flows=[1, -1], orientations=[-1, -1],
                   patchlabel=str(prior + i),
                   prior=prior + i - 1, connect=(1, 0), alpha=0.5)
        # 添加 Sankey 图表，表示从前一个图表到当前图表的连接
        sankey.add(flows=[1, -1], orientations=[1, 1],
                   patchlabel=str(prior + i + 1),
                   prior=prior + i, connect=(1, 0), alpha=0.5)

# 定义生成角连接的函数
def corner(sankey):
    """Generate a corner link."""
    # 获取当前图表中已有图表数量
    prior = len(sankey.diagrams)
    # 添加 Sankey 图表，表示从前一个图表到当前图表的连接
    sankey.add(flows=[1, -1], orientations=[0, 1],
               patchlabel=str(prior), facecolor='k',
               prior=prior - 1, connect=(1, 0), alpha=0.5)

# 创建一个图形对象
fig = plt.figure()
# 添加子图到图形中，无 x 轴和 y 轴刻度，设置标题
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Why would you want to do this?\n(But you could.)")
# 创建 Sankey 图对象，关联到 ax 绘图区域，unit 参数为 None
sankey = Sankey(ax=ax, unit=None)
# 添加 Sankey 图表，表示从一个起始点到第一个侧链的连接
sankey.add(flows=[1, -1], orientations=[0, 1],
           patchlabel="0", facecolor='k',
           rotation=45)
# 生成多个侧链
side(sankey, n=links_per_side)
# 生成一个角连接
corner(sankey)
# 生成多个侧链
side(sankey, n=links_per_side)
# 生成一个角连接
corner(sankey)
# 生成多个侧链
side(sankey, n=links_per_side)
# 生成一个角连接
corner(sankey)
# 生成多个侧链
side(sankey, n=links_per_side)
# 结束 Sankey 图的绘制
sankey.finish()

# 输出注意事项
# 1. 对齐不会明显（甚至根本不会）偏离；有 16007 个子图时仍然闭合。
# 2. 第一个图表旋转了 45 度，因此所有其他图表也相应旋转。

# 显示图形
plt.show()
```