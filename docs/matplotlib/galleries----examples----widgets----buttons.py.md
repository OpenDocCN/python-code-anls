# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\buttons.py`

```py
"""
=======
Buttons
=======

Constructing a simple button GUI to modify a sine wave.

The ``next`` and ``previous`` button widget helps visualize the wave with
new frequencies.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库并命名为 plt
import numpy as np  # 导入 numpy 库并命名为 np
from matplotlib.widgets import Button  # 从 matplotlib.widgets 模块导入 Button 类

# 创建频率数组
freqs = np.arange(2, 20, 3)

# 创建图形和坐标系
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l, = ax.plot(t, s, lw=2)  # 绘制正弦波，并保存 Line2D 对象到 l

# 定义一个类 Index，用于处理按钮点击事件
class Index:
    ind = 0  # 类属性，记录当前频率数组的索引位置

    # 处理 Next 按钮点击事件
    def next(self, event):
        self.ind += 1  # 索引加一
        i = self.ind % len(freqs)  # 计算取余后的索引
        ydata = np.sin(2*np.pi*freqs[i]*t)  # 根据新索引生成新的正弦波数据
        l.set_ydata(ydata)  # 更新图形的数据
        plt.draw()  # 重新绘制图形

    # 处理 Previous 按钮点击事件
    def prev(self, event):
        self.ind -= 1  # 索引减一
        i = self.ind % len(freqs)  # 计算取余后的索引
        ydata = np.sin(2*np.pi*freqs[i]*t)  # 根据新索引生成新的正弦波数据
        l.set_ydata(ydata)  # 更新图形的数据
        plt.draw()  # 重新绘制图形

callback = Index()  # 创建 Index 类的实例 callback

# 添加 Previous 和 Next 按钮，并绑定点击事件处理函数
axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)  # 将 next 方法绑定到 Next 按钮的点击事件
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)  # 将 prev 方法绑定到 Previous 按钮的点击事件

plt.show()  # 显示图形界面
```