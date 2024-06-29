# `D:\src\scipysrc\matplotlib\galleries\examples\misc\svg_filter_line.py`

```py
"""
===============
SVG Filter Line
===============

Demonstrate SVG filtering effects which might be used with Matplotlib.

Note that the filtering effects are only effective if your SVG renderer
support it.
"""

import io  # 导入 io 模块用于字节流操作
import xml.etree.ElementTree as ET  # 导入 ElementTree 模块，用于处理 XML 数据

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import matplotlib.transforms as mtransforms  # 导入 matplotlib.transforms 模块，用于图形变换

fig1 = plt.figure()  # 创建一个图形对象
ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])  # 添加一个坐标轴对象到图形中

# draw lines
l1, = ax.plot([0.1, 0.5, 0.9], [0.1, 0.9, 0.5], "bo-",  # 绘制蓝色线条
              mec="b", lw=5, ms=10, label="Line 1")
l2, = ax.plot([0.1, 0.5, 0.9], [0.5, 0.2, 0.7], "rs-",  # 绘制红色线条
              mec="r", lw=5, ms=10, label="Line 2")

for l in [l1, l2]:

    # draw shadows with same lines with slight offset and gray colors.

    xx = l.get_xdata()  # 获取线条的 x 坐标数据
    yy = l.get_ydata()  # 获取线条的 y 坐标数据
    shadow, = ax.plot(xx, yy)  # 绘制阴影效果的线条
    shadow.update_from(l)  # 更新阴影线条与原始线条的属性

    # adjust color
    shadow.set_color("0.2")  # 设置阴影线条颜色为灰色
    # adjust zorder of the shadow lines so that it is drawn below the
    # original lines
    shadow.set_zorder(l.get_zorder() - 0.5)  # 调整阴影线条的绘制顺序，使其位于原始线条下方

    # offset transform
    transform = mtransforms.offset_copy(l.get_transform(), fig1,
                                        x=4.0, y=-6.0, units='points')  # 创建偏移变换对象
    shadow.set_transform(transform)  # 应用偏移变换到阴影线条上

    # set the id for a later use
    shadow.set_gid(l.get_label() + "_shadow")  # 设置阴影线条的 id 属性，以便后续使用

ax.set_xlim(0., 1.)  # 设置 x 轴范围
ax.set_ylim(0., 1.)  # 设置 y 轴范围

# save the figure as a bytes string in the svg format.
f = io.BytesIO()  # 创建一个字节流对象
plt.savefig(f, format="svg")  # 将图形保存为 SVG 格式，并写入字节流中

# filter definition for a gaussian blur
filter_def = """
  <defs xmlns='http://www.w3.org/2000/svg'
        xmlns:xlink='http://www.w3.org/1999/xlink'>
    <filter id='dropshadow' height='1.2' width='1.2'>
      <feGaussianBlur result='blur' stdDeviation='3'/>
    </filter>
  </defs>
"""

# read in the saved svg
tree, xmlid = ET.XMLID(f.getvalue())  # 从保存的 SVG 中读取 XML 数据并获取其 id

# insert the filter definition in the svg dom tree.
tree.insert(0, ET.XML(filter_def))  # 将滤镜定义插入 SVG 的 DOM 树中的首部位置

for l in [l1, l2]:
    # pick up the svg element with given id
    shadow = xmlid[l.get_label() + "_shadow"]  # 获取具有特定 id 的 SVG 元素
    # apply shadow filter
    shadow.set("filter", 'url(#dropshadow)')  # 应用阴影滤镜效果

fn = "svg_filter_line.svg"  # 设置保存文件的文件名
print(f"Saving '{fn}'")  # 打印保存文件名信息
ET.ElementTree(tree).write(fn)  # 将修改后的 SVG DOM 树写入文件
```