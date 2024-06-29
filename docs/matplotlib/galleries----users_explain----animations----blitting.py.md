# `D:\src\scipysrc\matplotlib\galleries\users_explain\animations\blitting.py`

```py
# 导入必要的库
import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 库，用于生成数组和进行数学运算

# 生成一个包含 0 到 2π 之间 100 个点的数组
x = np.linspace(0, 2 * np.pi, 100)

# 创建一个图形窗口和一个坐标轴对象
fig, ax = plt.subplots()

# 创建一个绘图对象，绘制正弦函数曲线，并标记为动画对象
(ln,) = ax.plot(x, np.sin(x), animated=True)

# 显示图形窗口，但不阻塞当前进程继续执行
plt.show(block=False)

# 暂停 0.1 秒，确保窗口至少绘制一次
plt.pause(0.1)

# 复制整个图形窗口的背景（除了动画对象）
bg = fig.canvas.copy_from_bbox(fig.bbox)
# 绘制动画对象，这里使用了缓存的渲染器
ax.draw_artist(ln)
# show the result to the screen, this pushes the updated RGBA buffer from the
# renderer to the GUI framework so you can see it
fig.canvas.blit(fig.bbox)

for j in range(100):
    # reset the background back in the canvas state, screen unchanged
    fig.canvas.restore_region(bg)
    # update the artist, neither the canvas state nor the screen have changed
    ln.set_ydata(np.sin(x + (j / 100) * np.pi))
    # re-render the artist, updating the canvas state, but not the screen
    ax.draw_artist(ln)
    # copy the image to the GUI state, but screen might not be changed yet
    fig.canvas.blit(fig.bbox)
    # flush any pending GUI events, re-painting the screen if needed
    fig.canvas.flush_events()
    # you can put a pause in if you want to slow things down
    # plt.pause(.1)

# %%
# This example works and shows a simple animation, however because we
# are only grabbing the background once, if the size of the figure in
# pixels changes (due to either the size or dpi of the figure
# changing) , the background will be invalid and result in incorrect
# (but sometimes cool looking!) images.  There is also a global
# variable and a fair amount of boilerplate which suggests we should
# wrap this in a class.
#
# Class-based example
# -------------------
#
# We can use a class to encapsulate the boilerplate logic and state of
# restoring the background, drawing the artists, and then blitting the
# result to the screen.  Additionally, we can use the ``'draw_event'``
# callback to capture a new background whenever a full re-draw
# happens to handle resizes correctly.

class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        # copy the current figure's background into _bg
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        # redraw all animated artists
        self._draw_animated()
    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist
            要添加的艺术家对象。将其设置为 'animated' (动画) (为了安全起见)。*art* 必须在与此类管理的画布相关联的图形中。

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)  # 设置艺术家对象为动画状态
        self._artists.append(art)  # 将艺术家对象添加到管理列表中

    def _draw_animated(self):
        """
        Draw all of the animated artists.
        """
        fig = self.canvas.figure  # 获取画布所属的图形对象
        for a in self._artists:  # 遍历所有管理的艺术家对象
            fig.draw_artist(a)  # 在图形上绘制每个艺术家对象

    def update(self):
        """
        Update the screen with animated artists.
        """
        cv = self.canvas  # 获取画布对象
        fig = cv.figure  # 获取画布所属的图形对象

        # 如果还未保存背景，以防我们错过了绘制事件，
        if self._bg is None:
            self.on_draw(None)  # 调用on_draw方法来绘制

        else:
            # 恢复背景
            cv.restore_region(self._bg)
            # 绘制所有动画艺术家对象
            self._draw_animated()
            # 更新 GUI 状态
            cv.blit(fig.bbox)  # 在画布上绘制图形的指定区域

        # 让 GUI 事件循环处理它需要处理的任何事情
        cv.flush_events()
# %%
# 这里展示了如何使用我们的类。这个例子比第一个例子稍微复杂一些，因为我们添加了一个文本帧计数器。

# 创建一个新的图形和轴对象
fig, ax = plt.subplots()
# 添加一条曲线
(ln,) = ax.plot(x, np.sin(x), animated=True)
# 添加一个帧数显示文本
fr_number = ax.annotate(
    "0",  # 初始显示帧数为0
    (0, 1),  # 文本的位置在图的左上角
    xycoords="axes fraction",  # 使用轴坐标系
    xytext=(10, -10),  # 文本的偏移量
    textcoords="offset points",  # 偏移量的坐标系为 points
    ha="left",  # 水平对齐方式为左对齐
    va="top",  # 垂直对齐方式为顶部对齐
    animated=True,  # 允许动画
)
# 创建一个 BlitManager 对象来管理需要被 blit（局部更新）的艺术家对象
bm = BlitManager(fig.canvas, [ln, fr_number])
# 确保窗口在屏幕上并且绘制完成
plt.show(block=False)
plt.pause(.1)

# 循环更新动画效果，重复100次
for j in range(100):
    # 更新曲线的纵坐标数据
    ln.set_ydata(np.sin(x + (j / 100) * np.pi))
    # 更新帧数文本内容
    fr_number.set_text(f"frame: {j}")
    # 告诉 blitting 管理器执行更新操作
    bm.update()

# %%
# 这个类不依赖于 `.pyplot`，适合嵌入到更大的 GUI 应用程序中。
```