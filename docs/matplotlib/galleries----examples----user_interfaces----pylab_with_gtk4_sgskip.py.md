# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\pylab_with_gtk4_sgskip.py`

```
"""
================
pyplot with GTK4
================

An example of how to use pyplot to manage your figure windows, but modify the
GUI by accessing the underlying GTK widgets.
"""

# 导入 matplotlib 库
import matplotlib

# 设置使用 GTK4Agg 或 GTK4Cairo 后端
matplotlib.use('GTK4Agg')  # or 'GTK4Cairo'

# 导入 GTK 对应的库
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

# 导入 matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt

# 创建一个图形和坐标系
fig, ax = plt.subplots()

# 在坐标系上绘制两条曲线
ax.plot([1, 2, 3], 'ro-', label='easy as 1 2 3')
ax.plot([1, 4, 9], 'gs--', label='easy as 1 2 3 squared')

# 添加图例
ax.legend()

# 获取图形的画布管理器
manager = fig.canvas.manager

# 可以通过 manager 访问工具栏（toolbar）或垂直布局（vbox）属性
toolbar = manager.toolbar
vbox = manager.vbox

# 在工具栏上添加一个按钮
button = Gtk.Button(label='Click me')
button.connect('clicked', lambda button: print('hi mom'))
button.set_tooltip_text('Click me for fun and profit')
toolbar.append(button)

# 在垂直布局中添加一个小部件（widget）
label = Gtk.Label()
label.set_markup('Drag mouse over axes for position')
vbox.insert_child_after(label, fig.canvas)

# 定义一个更新函数，根据鼠标事件更新标签内容
def update(event):
    if event.xdata is None:
        label.set_markup('Drag mouse over axes for position')
    else:
        label.set_markup(
            f'<span color="#ef0000">x,y=({event.xdata}, {event.ydata})</span>')

# 连接图形画布的鼠标移动事件到更新函数
fig.canvas.mpl_connect('motion_notify_event', update)

# 显示图形界面
plt.show()
```