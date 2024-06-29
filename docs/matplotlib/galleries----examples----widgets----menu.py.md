# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\menu.py`

```py
`
"""
====
Menu
====

Using texts to construct a simple menu.
"""

from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器，方便创建数据类

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图

import matplotlib.artist as artist  # 导入 matplotlib.artist 模块，提供艺术家类基础
import matplotlib.patches as patches  # 导入 matplotlib.patches 模块，用于绘制形状
from matplotlib.typing import ColorType  # 导入 ColorType 类型，用于指定颜色类型

@dataclass  # 使用 dataclass 装饰器定义 ItemProperties 类，简化类的定义
class ItemProperties:
    fontsize: float = 14  # 字体大小，默认为 14
    labelcolor: ColorType = 'black'  # 标签颜色，默认为黑色
    bgcolor: ColorType = 'yellow'  # 背景颜色，默认为黄色
    alpha: float = 1.0  # 透明度，默认为 1.0

class MenuItem(artist.Artist):  # 定义 MenuItem 类，继承自 matplotlib.artist.Artist
    padx = 0.05  # 水平内边距，单位为英寸
    pady = 0.05  # 垂直内边距，单位为英寸

    def __init__(self, fig, labelstr, props=None, hoverprops=None, on_select=None):
        super().__init__()  # 调用父类的初始化方法

        self.set_figure(fig)  # 设置图形对象
        self.labelstr = labelstr  # 设置菜单项的标签字符串

        self.props = props if props is not None else ItemProperties()  # 设置属性，默认为 ItemProperties 的实例
        self.hoverprops = hoverprops if hoverprops is not None else ItemProperties()  # 设置悬停属性，默认为 ItemProperties 的实例
        if self.props.fontsize != self.hoverprops.fontsize:  # 检查字体大小是否一致
            raise NotImplementedError('support for different font sizes not implemented')  # 抛出异常，未实现不同字体大小支持

        self.on_select = on_select  # 设置选择事件回调函数

        # 在英寸坐标系中指定标签的位置
        self.label = fig.text(0, 0, labelstr, transform=fig.dpi_scale_trans, size=props.fontsize)
        self.text_bbox = self.label.get_window_extent(fig.canvas.get_renderer())  # 获取标签的边界框
        self.text_bbox = fig.dpi_scale_trans.inverted().transform_bbox(self.text_bbox)  # 转换边界框坐标

        self.rect = patches.Rectangle((0, 0), 1, 1, transform=fig.dpi_scale_trans)  # 创建矩形，后续会更新

        self.set_hover_props(False)  # 设置悬停状态，默认为不悬停

        fig.canvas.mpl_connect('button_release_event', self.check_select)  # 连接鼠标释放事件，调用 check_select 方法

    def check_select(self, event):
        over, _ = self.rect.contains(event)  # 检查鼠标是否在矩形区域内
        if not over:
            return  # 如果不在区域内，直接返回
        if self.on_select is not None:
            self.on_select(self)  # 如果定义了选择事件回调，调用它

    def set_extent(self, x, y, w, h, depth):
        self.rect.set(x=x, y=y, width=w, height=h)  # 设置矩形的位置和大小
        self.label.set(position=(x + self.padx, y + depth + self.pady / 2))  # 设置标签的位置
        self.hover = False  # 重置悬停状态

    def draw(self, renderer):
        self.rect.draw(renderer)  # 绘制矩形
        self.label.draw(renderer)  # 绘制标签

    def set_hover_props(self, b):
        props = self.hoverprops if b else self.props  # 根据悬停状态选择属性
        self.label.set(color=props.labelcolor)  # 设置标签颜色
        self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)  # 设置矩形背景颜色和透明度

    def set_hover(self, event):
        """
        Update the hover status of event and return whether it was changed.
        """
        b, _ = self.rect.contains(event)  # 检查鼠标是否在矩形区域内
        changed = (b != self.hover)  # 判断悬停状态是否改变
        if changed:
            self.set_hover_props(b)  # 更新悬停属性
        self.hover = b  # 更新悬停状态
        return changed  # 返回悬停状态是否改变

class Menu:
    # 初始化方法，接受一个图形对象和菜单项列表作为参数
    def __init__(self, fig, menuitems):
        # 将图形对象赋给实例属性
        self.figure = fig
        # 将菜单项列表赋给实例属性
        self.menuitems = menuitems

        # 计算菜单项中文本框的最大宽度、最大高度和深度
        maxw = max(item.text_bbox.width for item in menuitems)
        maxh = max(item.text_bbox.height for item in menuitems)
        depth = max(-item.text_bbox.y0 for item in menuitems)

        # 设置绘制起点的坐标
        x0 = 1
        y0 = 4

        # 计算菜单项的总宽度和高度
        width = maxw + 2 * MenuItem.padx
        height = maxh + MenuItem.pady

        # 遍历菜单项列表，设置每个菜单项的位置和大小
        for item in menuitems:
            left = x0
            bottom = y0 - maxh - MenuItem.pady

            item.set_extent(left, bottom, width, height, depth)

            # 将菜单项添加到图形对象的艺术家列表中
            fig.artists.append(item)
            y0 -= maxh + MenuItem.pady

        # 为图形对象的画布连接鼠标移动事件触发方法
        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    # 鼠标移动事件触发方法
    def on_move(self, event):
        # 如果有任何菜单项的悬停状态被设置
        if any(item.set_hover(event) for item in self.menuitems):
            # 重新绘制图形对象的画布
            self.figure.canvas.draw()
# 创建一个新的图形对象
fig = plt.figure()

# 调整子图的边界，使得左边界为0.3
fig.subplots_adjust(left=0.3)

# 定义一个属性对象，用于设置菜单项的标签颜色、背景色、字体大小和透明度
props = ItemProperties(labelcolor='black', bgcolor='yellow',
                       fontsize=15, alpha=0.2)

# 定义一个属性对象，用于设置菜单项鼠标悬停时的标签颜色、背景色、字体大小和透明度
hoverprops = ItemProperties(labelcolor='white', bgcolor='blue',
                            fontsize=15, alpha=0.2)

# 初始化一个空的菜单项列表
menuitems = []

# 遍历每个菜单项的标签，并为每个标签创建一个 MenuItem 对象
for label in ('open', 'close', 'save', 'save as', 'quit'):
    # 定义一个当菜单项被选中时触发的函数
    def on_select(item):
        print(f'you selected {item.labelstr}')
    
    # 创建一个 MenuItem 对象，使用指定的标签、属性和鼠标悬停属性，并指定选中时的回调函数
    item = MenuItem(fig, label, props=props, hoverprops=hoverprops,
                    on_select=on_select)
    
    # 将创建的 MenuItem 对象添加到菜单项列表中
    menuitems.append(item)

# 创建一个 Menu 对象，使用包含所有 MenuItem 对象的菜单项列表
menu = Menu(fig, menuitems)

# 显示图形界面，显示菜单
plt.show()
```