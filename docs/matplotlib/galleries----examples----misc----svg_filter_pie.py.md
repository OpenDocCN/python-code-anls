# `D:\src\scipysrc\matplotlib\galleries\examples\misc\svg_filter_pie.py`

```
"""
==============
SVG filter pie
==============

Demonstrate SVG filtering effects which might be used with Matplotlib.
The pie chart drawing code is borrowed from pie_demo.py

Note that the filtering effects are only effective if your SVG renderer
support it.
"""

import io  # 导入io模块，用于处理文件流
import xml.etree.ElementTree as ET  # 导入xml.etree.ElementTree模块，并简写为ET，用于处理XML文档

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，并简写为plt，用于绘图

from matplotlib.patches import Shadow  # 从matplotlib.patches模块导入Shadow类，用于创建阴影效果的图形元素

# make a square figure and Axes
fig = plt.figure(figsize=(6, 6))  # 创建一个大小为6x6的图形窗口
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # 在图形窗口上添加一个坐标轴，位置和大小为[0.1, 0.1, 0.8, 0.8]

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'  # 设置饼图的标签
fracs = [15, 30, 45, 10]  # 设置饼图每一部分的比例

explode = (0, 0.05, 0, 0)  # 设置饼图每一部分的突出显示

# We want to draw the shadow for each pie, but we will not use "shadow"
# option as it doesn't save the references to the shadow patches.
pies = ax.pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%')  # 在坐标轴上绘制饼图，并保存饼图对象到pies变量

for w in pies[0]:
    # set the id with the label.
    w.set_gid(w.get_label())  # 为每个饼图部分设置全局唯一标识符，使用饼图的标签作为ID

    # we don't want to draw the edge of the pie
    w.set_edgecolor("none")  # 设置饼图部分边缘不可见

for w in pies[0]:
    # create shadow patch
    s = Shadow(w, -0.01, -0.01)  # 创建阴影对象s，参数为原始图形对象w以及阴影偏移量
    s.set_gid(w.get_gid() + "_shadow")  # 为阴影对象设置全局唯一标识符，使用原始图形的ID加上"_shadow"
    s.set_zorder(w.get_zorder() - 0.1)  # 设置阴影对象的绘制顺序，稍微在原始图形的下层
    ax.add_patch(s)  # 将阴影对象添加到坐标轴上


# save
f = io.BytesIO()  # 创建一个字节流对象f，用于保存绘制的图形
plt.savefig(f, format="svg")  # 将绘制的图形保存到字节流f中，格式为SVG


# Filter definition for shadow using a gaussian blur and lighting effect.
# The lighting filter is copied from http://www.w3.org/TR/SVG/filters.html

# I tested it with Inkscape and Firefox3. "Gaussian blur" is supported
# in both, but the lighting effect only in Inkscape. Also note
# that, Inkscape's exporting also may not support it.

filter_def = """
  <defs xmlns='http://www.w3.org/2000/svg'
        xmlns:xlink='http://www.w3.org/1999/xlink'>
    <filter id='dropshadow' height='1.2' width='1.2'>
      <feGaussianBlur result='blur' stdDeviation='2'/>
    </filter>

    <filter id='MyFilter' filterUnits='objectBoundingBox'
            x='0' y='0' width='1' height='1'>
      <feGaussianBlur in='SourceAlpha' stdDeviation='4%' result='blur'/>
      <feOffset in='blur' dx='4%' dy='4%' result='offsetBlur'/>
      <feSpecularLighting in='blur' surfaceScale='5' specularConstant='.75'
           specularExponent='20' lighting-color='#bbbbbb' result='specOut'>
        <fePointLight x='-5000%' y='-10000%' z='20000%'/>
      </feSpecularLighting>
      <feComposite in='specOut' in2='SourceAlpha'
                   operator='in' result='specOut'/>
      <feComposite in='SourceGraphic' in2='specOut' operator='arithmetic'
    k1='0' k2='1' k3='1' k4='0'/>
    </filter>
  </defs>
"""

tree, xmlid = ET.XMLID(f.getvalue())  # 解析SVG图形的XML内容，并返回XML树和ID映射

# insert the filter definition in the svg dom tree.
tree.insert(0, ET.XML(filter_def))  # 将滤镜定义插入SVG文档的DOM树中，作为第一个子元素

for i, pie_name in enumerate(labels):
    pie = xmlid[pie_name]
    pie.set("filter", 'url(#MyFilter)')  # 为每个饼图部分设置滤镜效果，使用ID为MyFilter的滤镜

    shadow = xmlid[pie_name + "_shadow"]
    shadow.set("filter", 'url(#dropshadow)')  # 为每个饼图部分的阴影设置滤镜效果，使用ID为dropshadow的滤镜

fn = "svg_filter_pie.svg"
print(f"Saving '{fn}'")
ET.ElementTree(tree).write(fn)  # 将包含滤镜效果的SVG文档树写入文件fn中
```