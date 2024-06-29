# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\gtk3_spreadsheet_sgskip.py`

```py
"""
================
GTK3 spreadsheet
================

Example of embedding Matplotlib in an application and interacting with a
treeview to store data.  Double-click on an entry to update plot data.
"""

import gi

gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk, Gtk

from numpy.random import random

from matplotlib.backends.backend_gtk3agg import FigureCanvas  # or gtk3cairo.
from matplotlib.figure import Figure


class DataManager(Gtk.Window):
    num_rows, num_cols = 20, 10  # 定义数据表格的行数和列数

    data = random((num_rows, num_cols))  # 随机生成数据矩阵

    def __init__(self):
        super().__init__()
        self.set_default_size(600, 600)  # 设置窗口默认大小
        self.connect('destroy', lambda win: Gtk.main_quit())  # 关联窗口关闭事件到 Gtk.main_quit()

        self.set_title('GtkListStore demo')  # 设置窗口标题
        self.set_border_width(8)  # 设置窗口边框宽度

        vbox = Gtk.VBox(homogeneous=False, spacing=8)  # 创建垂直布局容器
        self.add(vbox)  # 将垂直布局容器添加到窗口中

        label = Gtk.Label(label='Double click a row to plot the data')  # 创建标签控件
        vbox.pack_start(label, False, False, 0)  # 将标签控件添加到垂直布局容器中

        sw = Gtk.ScrolledWindow()  # 创建滚动窗口
        sw.set_shadow_type(Gtk.ShadowType.ETCHED_IN)  # 设置滚动窗口的阴影类型
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)  # 设置滚动窗口的滚动策略
        vbox.pack_start(sw, True, True, 0)  # 将滚动窗口添加到垂直布局容器中

        model = self.create_model()  # 调用方法创建数据模型

        self.treeview = Gtk.TreeView(model=model)  # 创建树形视图控件

        # Matplotlib stuff
        fig = Figure(figsize=(6, 4))  # 创建 Matplotlib 图形对象

        self.canvas = FigureCanvas(fig)  # 创建 Matplotlib 图形的绘制区域
        vbox.pack_start(self.canvas, True, True, 0)  # 将绘制区域添加到垂直布局容器中
        ax = fig.add_subplot()  # 添加子图到 Figure 对象
        self.line, = ax.plot(self.data[0, :], 'go')  # 绘制第一行数据并返回线条对象

        self.treeview.connect('row-activated', self.plot_row)  # 连接树形视图的行激活事件到绘制方法
        sw.add(self.treeview)  # 将树形视图添加到滚动窗口中

        self.add_columns()  # 调用方法添加树形视图的列

        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.KEY_PRESS_MASK |
                        Gdk.EventMask.KEY_RELEASE_MASK)  # 添加按钮和键盘事件掩码

    def plot_row(self, treeview, path, view_column):
        ind, = path  # 获取索引以访问数据
        points = self.data[ind, :]  # 获取指定行的数据
        self.line.set_ydata(points)  # 更新线条的 Y 轴数据
        self.canvas.draw()  # 重新绘制图形

    def add_columns(self):
        for i in range(self.num_cols):
            column = Gtk.TreeViewColumn(str(i), Gtk.CellRendererText(), text=i)  # 创建树形视图的列对象
            self.treeview.append_column(column)  # 将列对象添加到树形视图中

    def create_model(self):
        types = [float] * self.num_cols  # 创建数据列的类型列表
        store = Gtk.ListStore(*types)  # 创建 Gtk.ListStore 对象
        for row in self.data:
            store.append(tuple(row))  # 将数据逐行添加到 Gtk.ListStore 对象中
        return store  # 返回创建的数据模型对象


manager = DataManager()  # 创建 DataManager 类的实例
manager.show_all()  # 显示窗口及其所有子控件
Gtk.main()  # 运行 Gtk 主循环
```