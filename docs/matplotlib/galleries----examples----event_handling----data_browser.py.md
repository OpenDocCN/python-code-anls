# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\data_browser.py`

```
"""
============
Data browser
============

Connecting data between multiple canvases.

This example covers how to interact data with multiple canvases. This
lets you select and highlight a point on one axis, and generating the
data of that point on the other axis.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""
import numpy as np

class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower Axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self):
        # 初始化最后选择的索引为0
        self.lastind = 0

        # 在Axes上方创建文本框，用于显示当前选择的信息
        self.text = ax.text(0.05, 0.95, 'selected: none',
                            transform=ax.transAxes, va='top')
        
        # 创建一个可视化的选中点，初始位置为第一个点的位置
        self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
                                 color='yellow', visible=False)

    def on_press(self, event):
        # 如果最后选择的索引为None，则返回
        if self.lastind is None:
            return
        # 如果按键不是'n'或'p'，则返回
        if event.key not in ('n', 'p'):
            return
        # 根据按键确定增加还是减少索引
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        # 更新最后选择的索引，并限制在有效范围内
        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(xs) - 1)
        # 更新显示
        self.update()

    def on_pick(self, event):
        # 如果点击的不是数据点，则返回
        if event.artist != line:
            return True

        N = len(event.ind)
        # 如果没有点击到点，则返回
        if not N:
            return True

        # 获取点击的位置
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        # 计算点击位置与所有数据点的距离
        distances = np.hypot(x - xs[event.ind], y - ys[event.ind])
        # 找到距离最近的点的索引
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        # 更新最后选择的索引
        self.lastind = dataind
        # 更新显示
        self.update()

    def update(self):
        # 如果最后选择的索引为None，则返回
        if self.lastind is None:
            return

        # 获取最后选择的数据点索引
        dataind = self.lastind

        # 清空第二个Axes并绘制选择的数据点的时间序列
        ax2.clear()
        ax2.plot(X[dataind])

        # 在第二个Axes上方显示数据点的统计信息
        ax2.text(0.05, 0.9, f'mu={xs[dataind]:1.3f}\nsigma={ys[dataind]:1.3f}',
                 transform=ax2.transAxes, va='top')
        ax2.set_ylim(-0.5, 1.5)

        # 设置选中点的位置和可见性
        self.selected.set_visible(True)
        self.selected.set_data(xs[dataind], ys[dataind])

        # 更新文本框显示的选择信息
        self.text.set_text('selected: %d' % dataind)
        
        # 刷新整个图形
        fig.canvas.draw()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 设置随机种子以便复现结果
    np.random.seed(19680801)

    # 创建随机数据
    X = np.random.rand(100, 200)
    xs = np.mean(X, axis=1)
    ys = np.std(X, axis=1)

    # 创建包含两个子图的图形窗口
    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.set_title('click on point to plot time series')
    
    # 在第一个Axes上绘制数据点，设置为可选
    line, = ax.plot(xs, ys, 'o', picker=True, pickradius=5)

    # 创建PointBrowser实例
    browser = PointBrowser()

    # 连接鼠标点击事件和键盘按下事件到对应的处理函数
    fig.canvas.mpl_connect('pick_event', browser.on_pick)
    fig.canvas.mpl_connect('key_press_event', browser.on_press)
    # 显示当前 matplotlib 中的所有图形
    plt.show()
```