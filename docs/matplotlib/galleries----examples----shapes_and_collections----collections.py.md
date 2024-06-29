# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\collections.py`

```
# 导入需要的库
import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy

from matplotlib import collections, transforms  # 从 matplotlib 中导入 collections 和 transforms 模块

# 设定顶点数和点数
nverts = 50  # 多边形的顶点数
npts = 100  # 点的数量

# 创建螺旋线数据
r = np.arange(nverts)
theta = np.linspace(0, 2*np.pi, nverts)
xx = r * np.sin(theta)
yy = r * np.cos(theta)
spiral = np.column_stack([xx, yy])  # 将 x 和 y 组合成一个 (nverts, 2) 的数组

# 固定随机种子以保证可重复性
rs = np.random.RandomState(19680801)  # 创建一个随机数生成器对象

# 创建偏移量
xyo = rs.randn(npts, 2)  # 创建一个形状为 (npts, 2) 的随机数组，用于偏移

# 创建颜色列表，使用默认的颜色系列
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # 获取默认颜色循环

# 创建 2x2 的子图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)

# 创建 LineCollection 对象，使用偏移量和偏移变换
col = collections.LineCollection(
    [spiral], offsets=xyo, offset_transform=ax1.transData)
trans = fig.dpi_scale_trans + transforms.Affine2D().scale(1.0/72.0)
col.set_transform(trans)  # 将点转换为像素坐标
# 注意：集合初始化器的第一个参数必须是 (x, y) 元组序列的列表；我们只有一个序列，但仍然必须将其放入列表中。
ax1.add_collection(col, autolim=True)
# autolim=True 启用自动缩放。对于带有偏移量的集合，这不是高效或准确的，但足以生成一个可以用作起点的图形。
# 如果事先知道要显示的 x 和 y 的范围，最好显式设置它们，省略 'ax1.autoscale_view()' 调用。

# 创建线段的变换，使其大小以点为单位给出：
col.set_color(colors)

ax1.autoscale_view()  # 查看上面的注释，在 ax1.add_collection 之后。
ax1.set_title('LineCollection using offsets')  # 设置子图标题

# 使用相同的数据，但填充曲线。
col = collections.PolyCollection(
    [spiral], offsets=xyo, offset_transform=ax2.transData)
trans = transforms.Affine2D().scale(fig.dpi/72.0)
col.set_transform(trans)  # 将点转换为像素坐标
ax2.add_collection(col, autolim=True)
col.set_color(colors)

ax2.autoscale_view()
ax2.set_title('PolyCollection using offsets')  # 设置子图标题

# 7边形的正多边形
# 创建一个包含正多边形的集合对象，具有7个边形，每个边形的大小由xx的绝对值乘以10.0决定，
# 偏移量为xyo，偏移变换使用ax3的数据坐标系
col = collections.RegularPolyCollection(
    7, sizes=np.abs(xx) * 10.0, offsets=xyo, offset_transform=ax3.transData)

# 创建一个仿射变换对象，用于将点转换为像素坐标，缩放比例为fig.dpi / 72.0
trans = transforms.Affine2D().scale(fig.dpi / 72.0)
col.set_transform(trans)  # 设置点到像素的转换

ax3.add_collection(col, autolim=True)  # 将集合对象添加到ax3中，并自动调整显示范围
col.set_color(colors)  # 设置集合对象的颜色
ax3.autoscale_view()  # 自动调整视图范围
ax3.set_title('RegularPolyCollection using offsets')  # 设置图表标题


# 模拟一系列海洋流速剖面，每个剖面逐步偏移0.1 m/s，形成所谓的“瀑布”图或“错开”图。
nverts = 60
ncurves = 20
offs = (0.1, 0.0)

yy = np.linspace(0, 2*np.pi, nverts)
ym = np.max(yy)
xx = (0.2 + (ym - yy) / ym) ** 2 * np.cos(yy - 0.4) * 0.5
segs = []
for i in range(ncurves):
    xxx = xx + 0.02*rs.randn(nverts)
    curve = np.column_stack([xxx, yy * 100])
    segs.append(curve)

# 创建一条由线段组成的集合对象，使用segs作为线段数据，偏移量为offs
col = collections.LineCollection(segs, offsets=offs)
ax4.add_collection(col, autolim=True)  # 将集合对象添加到ax4中，并自动调整显示范围
col.set_color(colors)  # 设置集合对象的颜色
ax4.autoscale_view()  # 自动调整视图范围
ax4.set_title('Successive data offsets')  # 设置图表标题
ax4.set_xlabel('Zonal velocity component (m/s)')  # 设置x轴标签
ax4.set_ylabel('Depth (m)')  # 设置y轴标签
# 反转y轴，使深度向下增加
ax4.set_ylim(ax4.get_ylim()[::-1])


plt.show()  # 显示图表
```