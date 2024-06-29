# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\pylab_with_gtk3_sgskip.py`

```
"""
================
pyplot with GTK3
================

使用 pyplot 来管理图形窗口，并通过访问底层的 GTK 小部件修改 GUI 的示例。
"""

import matplotlib

matplotlib.use('GTK3Agg')  # 使用GTK3Agg作为后端，或者'GTK3Cairo'
import gi

import matplotlib.pyplot as plt

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

# 创建一个图形和轴
fig, ax = plt.subplots()
# 在轴上绘制线条，并添加标签
ax.plot([1, 2, 3], 'ro-', label='easy as 1 2 3')
ax.plot([1, 4, 9], 'gs--', label='easy as 1 2 3 squared')
ax.legend()  # 显示图例

# 获取图形的管理器，可以访问窗口或vbox属性
manager = fig.canvas.manager
toolbar = manager.toolbar  # 获取工具栏
vbox = manager.vbox  # 获取垂直布局框

# 在工具栏上添加一个按钮
button = Gtk.Button(label='Click me')  # 创建按钮对象
button.show()  # 显示按钮
button.connect('clicked', lambda button: print('hi mom'))  # 按钮点击事件处理

toolitem = Gtk.ToolItem()  # 创建工具项
toolitem.show()  # 显示工具项
toolitem.set_tooltip_text('Click me for fun and profit')  # 设置工具提示文本
toolitem.add(button)  # 将按钮添加到工具项中

pos = 8  # 插入工具项的位置
toolbar.insert(toolitem, pos)  # 在工具栏指定位置插入工具项

# 在垂直布局框中添加一个小部件
label = Gtk.Label()  # 创建标签对象
label.set_markup('Drag mouse over axes for position')  # 设置标签文本
label.show()  # 显示标签
vbox.pack_start(label, False, False, 0)  # 将标签添加到垂直布局框中
vbox.reorder_child(toolbar, -1)  # 重新排列工具栏的顺序

# 更新函数，根据鼠标事件更新标签内容
def update(event):
    if event.xdata is None:
        label.set_markup('Drag mouse over axes for position')  # 如果没有xdata，显示默认文本
    else:
        label.set_markup(
            f'<span color="#ef0000">x,y=({event.xdata}, {event.ydata})</span>')  # 显示鼠标位置的x和y坐标

fig.canvas.mpl_connect('motion_notify_event', update)  # 连接鼠标移动事件到更新函数

plt.show()  # 显示图形界面
```