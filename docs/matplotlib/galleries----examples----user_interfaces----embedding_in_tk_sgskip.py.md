# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_tk_sgskip.py`

```py
"""
===============
Embedding in Tk
===============

"""

# 导入 tkinter 库，用于创建 GUI 应用程序
import tkinter

# 导入 numpy 库，用于数值计算
import numpy as np

# 导入 Matplotlib 相关模块
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

# 创建一个 Tkinter 窗口
root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# 创建一个 Matplotlib 图形对象
fig = Figure(figsize=(5, 4), dpi=100)

# 生成时间序列数据
t = np.arange(0, 3, .01)

# 向图形对象中添加子图
ax = fig.add_subplot()
# 绘制正弦波曲线并保存线对象
line, = ax.plot(t, 2 * np.sin(2 * np.pi * t))
# 设置 x 轴和 y 轴标签
ax.set_xlabel("time [s]")
ax.set_ylabel("f(t)")

# 创建一个 FigureCanvasTkAgg 对象，用于在 Tkinter 窗口中绘制 Matplotlib 图形
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()

# 创建 Matplotlib 工具栏对象
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

# 定义键盘事件处理函数
canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

# 创建一个 Quit 按钮，用于关闭应用程序
button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy)

# 定义一个更新频率的函数，并创建一个滑动条
def update_frequency(new_val):
    # 获取滑动条当前值作为频率
    f = float(new_val)
    
    # 更新正弦波数据
    y = 2 * np.sin(2 * np.pi * f * t)
    line.set_data(t, y)
    
    # 更新画布和工具栏显示
    canvas.draw()

# 创建一个水平方向的滑动条，用于调整频率
slider_update = tkinter.Scale(root, from_=1, to=5, orient=tkinter.HORIZONTAL,
                              command=update_frequency, label="Frequency [Hz]")

# 确保窗口布局中的控件顺序正确，canvas 被放置在最后以确保界面控件尽可能显示完整
button_quit.pack(side=tkinter.BOTTOM)
slider_update.pack(side=tkinter.BOTTOM)
toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

# 启动 Tkinter 的事件循环，开始运行应用程序
tkinter.mainloop()
```