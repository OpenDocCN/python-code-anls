# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\text_alignment.py`

```
# %%
# The following plot demonstrates text alignment relative to a rectangle within the plot.

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

fig, ax = plt.subplots()  # 创建一个图形对象和一个坐标轴对象

# 在坐标轴的相对坐标系中创建一个矩形
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
p = plt.Rectangle((left, bottom), width, height, fill=False)
p.set_transform(ax.transAxes)  # 设置矩形的变换方式为坐标轴的相对坐标系
p.set_clip_on(False)
ax.add_patch(p)  # 将矩形添加到坐标轴上

# 添加文本到指定位置，并设置其水平和垂直对齐方式
ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, 0.5 * (bottom + top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)
# 在绘图对象 ax 上添加文本，显示在左上角
ax.text(left, 0.5 * (bottom + top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

# 在绘图对象 ax 上添加文本，显示在中间
ax.text(0.5 * (left + right), 0.5 * (bottom + top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)

# 在绘图对象 ax 上添加文本，水平和垂直居中
ax.text(right, 0.5 * (bottom + top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

# 在绘图对象 ax 上添加文本，带有换行符的旋转文本
ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes)

# 关闭图形对象 ax 的坐标轴显示
ax.set_axis_off()

# 显示绘图
plt.show()
```