# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\gtk4_spreadsheet_sgskip.py`

```
"""
================
GTK4 spreadsheet
================

Example of embedding Matplotlib in an application and interacting with a
treeview to store data.  Double-click on an entry to update plot data.
"""

import gi

gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk

from numpy.random import random

from matplotlib.backends.backend_gtk4agg import FigureCanvas  # or gtk4cairo.
from matplotlib.figure import Figure


class DataManager(Gtk.ApplicationWindow):
    num_rows, num_cols = 20, 10

    data = random((num_rows, num_cols))  # 生成一个随机数据矩阵

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_default_size(600, 600)  # 设置窗口默认大小

        self.set_title('GtkListStore demo')  # 设置窗口标题

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, homogeneous=False,
                       spacing=8)  # 创建垂直布局的容器 vbox
        self.set_child(vbox)  # 设置窗口的子部件为 vbox

        label = Gtk.Label(label='Double click a row to plot the data')  # 创建一个标签
        vbox.append(label)  # 将标签添加到 vbox 中显示

        sw = Gtk.ScrolledWindow()  # 创建一个滚动窗口
        sw.set_has_frame(True)  # 设置滚动窗口有边框
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)  # 设置滚动策略
        sw.set_hexpand(True)  # 横向扩展
        sw.set_vexpand(True)  # 纵向扩展
        vbox.append(sw)  # 将滚动窗口添加到 vbox 中显示

        model = self.create_model()  # 创建数据模型
        self.treeview = Gtk.TreeView(model=model)  # 创建树视图
        self.treeview.connect('row-activated', self.plot_row)  # 连接行激活事件到 plot_row 方法
        sw.set_child(self.treeview)  # 将树视图添加到滚动窗口中显示

        # Matplotlib stuff
        fig = Figure(figsize=(6, 4), layout='constrained')  # 创建 matplotlib 图形对象

        self.canvas = FigureCanvas(fig)  # 创建用于显示 matplotlib 图形的画布
        self.canvas.set_hexpand(True)  # 横向扩展
        self.canvas.set_vexpand(True)  # 纵向扩展
        vbox.append(self.canvas)  # 将画布添加到 vbox 中显示
        ax = fig.add_subplot()  # 添加子图到图形对象
        self.line, = ax.plot(self.data[0, :], 'go')  # 绘制数据的第一行，并得到线条对象 self.line

        self.add_columns()  # 添加列到树视图中

    def plot_row(self, treeview, path, view_column):
        ind, = path  # 获取数据的索引
        points = self.data[ind, :]  # 获取对应行的数据
        self.line.set_ydata(points)  # 更新绘图数据
        self.canvas.draw()  # 重新绘制画布

    def add_columns(self):
        for i in range(self.num_cols):
            column = Gtk.TreeViewColumn(str(i), Gtk.CellRendererText(), text=i)  # 创建树视图列
            self.treeview.append_column(column)  # 将列添加到树视图中

    def create_model(self):
        types = [float] * self.num_cols  # 创建数据类型列表
        store = Gtk.ListStore(*types)  # 创建列表存储模型
        for row in self.data:
            # Gtk.ListStore.append is broken in PyGObject, so insert manually.
            it = store.insert(-1)  # 手动插入新行
            store.set(it, {i: val for i, val in enumerate(row)})  # 设置行数据
        return store  # 返回数据模型


def on_activate(app):
    manager = DataManager(application=app)  # 创建 DataManager 实例
    manager.show()  # 显示窗口


app = Gtk.Application(application_id='org.matplotlib.examples.GTK4Spreadsheet')  # 创建 Gtk 应用
app.connect('activate', on_activate)  # 连接激活应用事件到 on_activate 方法
app.run()  # 运行应用
```