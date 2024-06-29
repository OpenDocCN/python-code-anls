# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\demo_gridspec01.py`

```
"""
=================
subplot2grid demo
=================

This example demonstrates the use of `.pyplot.subplot2grid` to generate
subplots.  Using `.GridSpec`, as demonstrated in
:doc:`/gallery/userdemo/demo_gridspec03` is generally preferred.
"""

# 导入 matplotlib.pyplot 库，用于绘制图形
import matplotlib.pyplot as plt

# 定义一个函数，用于给图形中的每个子图添加标注
def annotate_axes(fig):
    # 遍历图形中的每个子图
    for i, ax in enumerate(fig.axes):
        # 在每个子图的中心位置添加文本标注，显示子图的序号
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        # 设置子图的坐标轴参数，隐藏标签
        ax.tick_params(labelbottom=False, labelleft=False)

# 创建一个新的图形窗口
fig = plt.figure()

# 在图形窗口中创建多个子图，使用subplot2grid方法进行布局
# 第一个子图，跨3列
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
# 第二个子图，跨2列
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# 第三个子图，跨2行
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# 第四个子图，位于第3行第1列
ax4 = plt.subplot2grid((3, 3), (2, 0))
# 第五个子图，位于第3行第2列
ax5 = plt.subplot2grid((3, 3), (2, 1))

# 调用annotate_axes函数，为图形中的每个子图添加标注
annotate_axes(fig)

# 显示图形
plt.show()
```