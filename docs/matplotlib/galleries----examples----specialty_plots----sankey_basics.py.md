# `D:\src\scipysrc\matplotlib\galleries\examples\specialty_plots\sankey_basics.py`

```
"""
================
The Sankey class
================

Demonstrate the Sankey class by producing three basic diagrams.
"""

# 导入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt
# 从 matplotlib.sankey 模块导入 Sankey 类
from matplotlib.sankey import Sankey

# %%
# Example 1 -- Mostly defaults
#
# This demonstrates how to create a simple diagram by implicitly calling the
# Sankey.add() method and by appending finish() to the call to the class.

# 创建 Sankey 图表示例，使用默认设置
Sankey(flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
       labels=['', '', '', 'First', 'Second', 'Third', 'Fourth', 'Fifth'],
       orientations=[-1, 1, 0, 1, 1, 1, 0, -1]).finish()
plt.title("The default settings produce a diagram like this.")

# %%
# Notice:
#
# 1. Axes weren't provided when Sankey() was instantiated, so they were
#    created automatically.
# 2. The scale argument wasn't necessary since the data was already
#    normalized.
# 3. By default, the lengths of the paths are justified.


# %%
# Example 2
#
# This demonstrates:
#
# 1. Setting one path longer than the others
# 2. Placing a label in the middle of the diagram
# 3. Using the scale argument to normalize the flows
# 4. Implicitly passing keyword arguments to PathPatch()
# 5. Changing the angle of the arrow heads
# 6. Changing the offset between the tips of the paths and their labels
# 7. Formatting the numbers in the path labels and the associated unit
# 8. Changing the appearance of the patch and the labels after the figure is
#    created

# 创建一个新的图形窗口
fig = plt.figure()
# 添加一个子图到图形窗口，设定标题和坐标轴标签
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Flow Diagram of a Widget")
# 创建 Sankey 图表示例，并指定绘图参数
sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=180,
                format='%.0f', unit='%')
sankey.add(flows=[25, 0, 60, -10, -20, -5, -15, -10, -40],
           labels=['', '', '', 'First', 'Second', 'Third', 'Fourth',
                   'Fifth', 'Hurray!'],
           orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0],
           pathlengths=[0.25, 0.25, 0.25, 0.25, 0.25, 0.6, 0.25, 0.25,
                        0.25],
           patchlabel="Widget\nA")  # Arguments to matplotlib.patches.PathPatch
# 完成 Sankey 图的绘制，并返回图表中的图示对象
diagrams = sankey.finish()
# 修改图示对象中最后一个文本的颜色为红色
diagrams[0].texts[-1].set_color('r')
# 设置图示对象中所有文本的字体加粗
diagrams[0].text.set_fontweight('bold')

# %%
# Notice:
#
# 1. Since the sum of the flows is nonzero, the width of the trunk isn't
#    uniform.  The matplotlib logging system logs this at the DEBUG level.
# 2. The second flow doesn't appear because its value is zero.  Again, this is
#    logged at the DEBUG level.


# %%
# Example 3
#
# This demonstrates:
#
# 1. Connecting two systems
# 2. Turning off the labels of the quantities
# 3. Adding a legend

# 创建一个新的图形窗口
fig = plt.figure()
# 添加一个子图到图形窗口，设定标题和坐标轴标签
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
# 定义流量数据
flows = [0.25, 0.15, 0.60, -0.10, -0.05, -0.25, -0.15, -0.10, -0.35]
# 创建 Sankey 图表示例，不指定单位
sankey = Sankey(ax=ax, unit=None)
# 添加第一个系统的流量数据和标签
sankey.add(flows=flows, label='one',
           orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0])
# 添加第二个系统的流量数据和标签，并连接到第一个系统
sankey.add(flows=[-0.25, 0.15, 0.1], label='two',
           orientations=[-1, -1, -1], prior=0, connect=(0, 0))
# 获取 Sankey 图的所有图表对象，并存储在 diagrams 变量中
diagrams = sankey.finish()
# 从 diagrams 中取出最后一个图表对象，并设置其 hatch 样式为 '/'
diagrams[-1].patch.set_hatch('/')
# 在图中添加图例
plt.legend()

# %%
# 注意这里只指定了一个连接，但系统形成一个闭路，原因是：(1) 路径的长度合理，
# (2) 流的方向和顺序是镜像对称的。
# 展示绘制好的图形
plt.show()

# %%
#
# .. admonition:: References
#
#    在这个示例中展示了以下函数、方法、类和模块的使用：
#
#    - `matplotlib.sankey`
#    - `matplotlib.sankey.Sankey`
#    - `matplotlib.sankey.Sankey.add`
#    - `matplotlib.sankey.Sankey.finish`
```