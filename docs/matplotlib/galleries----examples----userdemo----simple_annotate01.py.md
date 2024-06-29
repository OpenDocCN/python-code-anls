# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\simple_annotate01.py`

```py
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图

import matplotlib.patches as mpatches  # 导入matplotlib的patches模块，用于绘制图形

fig, axs = plt.subplots(2, 4)  # 创建2行4列的子图布局，返回Figure对象和Axes对象的数组

x1, y1 = 0.3, 0.3  # 设置起始点的x和y坐标
x2, y2 = 0.7, 0.7  # 设置结束点的x和y坐标

ax = axs.flat[0]  # 获取第一个子图
ax.plot([x1, x2], [y1, y2], "o")  # 在子图中绘制以(x1, y1)和(x2, y2)为端点的线段和圆点
ax.annotate("",  # 添加注释，无文本
            xy=(x1, y1), xycoords='data',  # 箭头起始点在数据坐标系中的位置
            xytext=(x2, y2), textcoords='data',  # 箭头终点在数据坐标系中的位置
            arrowprops=dict(arrowstyle="->"))  # 箭头样式为箭头
ax.text(.05, .95, "A $->$ B",  # 在子图中添加文本"A -> B"
        transform=ax.transAxes, ha="left", va="top")  # 文本位置为相对坐标(0.05, 0.95)

ax = axs.flat[2]  # 获取第三个子图
ax.plot([x1, x2], [y1, y2], "o")  # 在子图中绘制以(x1, y1)和(x2, y2)为端点的线段和圆点
ax.annotate("",  # 添加注释，无文本
            xy=(x1, y1), xycoords='data',  # 箭头起始点在数据坐标系中的位置
            xytext=(x2, y2), textcoords='data',  # 箭头终点在数据坐标系中的位置
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3",  # 箭头样式为箭头，连接样式为arc3，弧度为0.3
                            shrinkB=5))  # 箭头尾部缩进5个单位
ax.text(.05, .95, "shrinkB=5",  # 在子图中添加文本"shrinkB=5"
        transform=ax.transAxes, ha="left", va="top")  # 文本位置为相对坐标(0.05, 0.95)

ax = axs.flat[3]  # 获取第四个子图
ax.plot([x1, x2], [y1, y2], "o")  # 在子图中绘制以(x1, y1)和(x2, y2)为端点的线段和圆点
ax.annotate("",  # 添加注释，无文本
            xy=(x1, y1), xycoords='data',  # 箭头起始点在数据坐标系中的位置
            xytext=(x2, y2), textcoords='data',  # 箭头终点在数据坐标系中的位置
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))  # 箭头样式为箭头，连接样式为arc3，弧度为0.3
ax.text(.05, .95, "connectionstyle=arc3",  # 在子图中添加文本"connectionstyle=arc3"
        transform=ax.transAxes, ha="left", va="top")  # 文本位置为相对坐标(0.05, 0.95)

ax = axs.flat[4]  # 获取第五个子图
ax.plot([x1, x2], [y1, y2], "o")  # 在子图中绘制以(x1, y1)和(x2, y2)为端点的线段和圆点
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.5)  # 创建椭圆对象
ax.add_artist(el)  # 将椭圆添加到子图中
ax.annotate("",  # 添加注释，无文本
            xy=(x1, y1), xycoords='data',  # 箭头起始点在数据坐标系中的位置
            xytext=(x2, y2), textcoords='data',  # 箭头终点在数据坐标系中的位置
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))  # 箭头样式为箭头，连接样式为arc3，弧度为0.2
ax = axs.flat[5]  # 获取第六个子图
ax.plot([x1, x2], [y1, y2], "o") , the
```