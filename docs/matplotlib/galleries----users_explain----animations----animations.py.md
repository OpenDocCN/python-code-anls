# `D:\src\scipysrc\matplotlib\galleries\users_explain\animations\animations.py`

```py
"""
.. redirect-from:: /tutorials/introductory/animation_tutorial

.. _animations:

===========================
Animations using Matplotlib
===========================

Based on its plotting functionality, Matplotlib also provides an interface to
generate animations using the `~matplotlib.animation` module. An
animation is a sequence of frames where each frame corresponds to a plot on a
`~matplotlib.figure.Figure`. This tutorial covers a general guideline on
how to create such animations and the different options available.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并用 plt 别名引用
import numpy as np  # 导入 numpy 模块，并用 np 别名引用

import matplotlib.animation as animation  # 导入 matplotlib 的 animation 模块，并用 animation 别名引用

# %%
# Animation classes
# =================
#
# The animation process in Matplotlib can be thought of in 2 different ways:
#
# - `~matplotlib.animation.FuncAnimation`: Generate data for first
#   frame and then modify this data for each frame to create an animated plot.
#
# - `~matplotlib.animation.ArtistAnimation`: Generate a list (iterable)
#   of artists that will draw in each frame in the animation.
#
# `~matplotlib.animation.FuncAnimation` is more efficient in terms of
# speed and memory as it draws an artist once and then modifies it. On the
# other hand `~matplotlib.animation.ArtistAnimation` is flexible as it
# allows any iterable of artists to be animated in a sequence.
#
# ``FuncAnimation``
# -----------------
#
# The `~matplotlib.animation.FuncAnimation` class allows us to create an
# animation by passing a function that iteratively modifies the data of a plot.
# This is achieved by using the *setter* methods on various
# `~matplotlib.artist.Artist` (examples: `~matplotlib.lines.Line2D`,
# `~matplotlib.collections.PathCollection`, etc.). A usual
# `~matplotlib.animation.FuncAnimation` object takes a
# `~matplotlib.figure.Figure` that we want to animate and a function
# *func* that modifies the data plotted on the figure. It uses the *frames*
# parameter to determine the length of the animation. The *interval* parameter
# is used to determine time in milliseconds between drawing of two frames.
# Animating using `.FuncAnimation` typically requires these steps:
#
# 1) Plot the initial figure as you would in a static plot. Save all the created
#    artists, which are returned by the plot functions, in variables so that you can
#    access and modify them later in the animation function.
# 2) Create an animation function that updates the artists for a given frame.
#    Typically, this calls ``set_*`` methods of the artists.
# 3) Create a `.FuncAnimation`, passing the `.Figure` and the animation function.
# 4) Save or show the animation using one of the following methods:
#
#    - `.pyplot.show` to show the animation in a window
#    - `.Animation.to_html5_video` to create a HTML ``<video>`` tag
#    - `.Animation.to_jshtml` to create HTML code with interactive JavaScript animation
#      controls
#    - `.Animation.save` to save the animation to a file
#
# 下面的表格展示了几种绘图方法，它们返回的图形对象以及一些常用的 `set_*` 方法，这些方法用于更新底层数据。虽然在动画中更新数据是最常见的操作，但您也可以更新其他方面，如颜色或文本位置。
#
# ========================================  =============================  ===========================
# 绘图方法                                   图形对象                         数据集方法
# ========================================  =============================  ===========================
# `.Axes.plot`                              `.lines.Line2D`                `~.Line2D.set_data`,
#                                                                          `~.Line2D.set_xdata`,
#                                                                          `~.Line2D.set_ydata`
# `.Axes.scatter`                           `.collections.PathCollection`  `~.collections.\
#                                                                          PathCollection.set_offsets`
# `.Axes.imshow`                            `.image.AxesImage`             ``AxesImage.set_data``
# `.Axes.annotate`                          `.text.Annotation`             `~.text.Annotation.\
#                                                                          update_positions`
# `.Axes.barh`                              `.patches.Rectangle`           `~.Rectangle.set_angle`,
#                                                                          `~.Rectangle.set_bounds`,
#                                                                          `~.Rectangle.set_height`,
#                                                                          `~.Rectangle.set_width`,
#                                                                          `~.Rectangle.set_x`,
#                                                                          `~.Rectangle.set_y`,
#                                                                          `~.Rectangle.set_xy`
# `.Axes.fill`                              `.patches.Polygon`             `~.Polygon.set_xy`
# `.Axes.add_patch`\(`.patches.Ellipse`\)   `.patches.Ellipse`             `~.Ellipse.set_angle`,
#                                                                          `~.Ellipse.set_center`,
#                                                                          `~.Ellipse.set_height`,
#                                                                          `~.Ellipse.set_width`
# `.Axes.set_title`, `.Axes.text`           `.text.Text`                   `~.Text.set_text`
# ========================================  =============================  ===========================
#
# 涵盖所有类型图形对象的设置方法超出了本教程的范围，但可以在它们各自的文档中找到。下面是使用 `.Axes.scatter` 和 `.Axes.plot` 的示例。
fig, ax = plt.subplots()
t = np.linspace(0, 3, 40)
# 创建一个包含40个均匀间隔的时间点的数组，范围从0到3秒
g = -9.81
# 设置重力加速度为-9.81米/秒²
v0 = 12
# 设置初始速度为12米/秒
z = g * t**2 / 2 + v0 * t
# 计算抛物线运动的高度，z为时间t下的高度数组

v02 = 5
# 设置另一个初始速度为5米/秒
z2 = g * t**2 / 2 + v02 * t
# 计算另一个初始速度下的抛物线运动的高度数组

scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
# 在图形上绘制散点图的起始点，使用蓝色标记，大小为5像素，并标记初始速度v0
line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
# 在图形上绘制折线图的起始点，标记初始速度v02，并获取折线对象
ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
# 设置图形的x轴和y轴范围，以及x轴和y轴的标签
ax.legend()
# 在图形上添加图例

def update(frame):
    # 每个帧更新每个艺术家存储的数据。
    x = t[:frame]
    y = z[:frame]
    # 更新散点图：
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # 更新折线图：
    line2.set_xdata(t[:frame])
    line2.set_ydata(z2[:frame])
    return (scat, line2)

ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
# 创建动画对象，每30毫秒更新一次，总共40帧
plt.show()

# %%
# ``ArtistAnimation``
# -------------------
#
# `~matplotlib.animation.ArtistAnimation` 可以用来生成动画，如果有数据存储在不同的艺术家中。
# 这些艺术家的列表然后逐帧转换成动画。例如，当使用 `.Axes.barh` 绘制条形图时，它为每个条和误差条创建多个艺术家。
# 要更新图表，需要逐个更新容器中的每个条，并重新绘制它们。相反，可以使用 `.animation.ArtistAnimation` 来单独绘制每一帧，然后将它们组合成动画。这在条形图比赛中是一个简单的示例。

fig, ax = plt.subplots()
# 创建一个新的图形和轴
rng = np.random.default_rng(19680801)
# 创建一个随机数生成器实例
data = np.array([20, 20, 20, 20])
x = np.array([1, 2, 3, 4])

artists = []
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
for i in range(20):
    data += rng.integers(low=0, high=10, size=data.shape)
    container = ax.barh(x, data, color=colors)
    artists.append(container)
    # 创建每帧的艺术家列表

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
# 创建艺术家动画对象，每400毫秒更新一次
plt.show()

# %%
# Animation writers
# =================
#
# 动画对象可以使用各种多媒体编写器（例如Pillow，*ffpmeg*，*imagemagick*）保存到磁盘。
# 并非所有的视频格式都被所有的编写器支持。有4种主要类型的编写器：
#
# - `~matplotlib.animation.PillowWriter` - 使用Pillow库创建动画。
#
# - `~matplotlib.animation.HTMLWriter` - 用于创建基于JavaScript的动画。
#
# - 基于管道的编写器 - `~matplotlib.animation.FFMpegWriter` 和 `~matplotlib.animation.ImageMagickWriter` 是基于管道的编写器。
#   这些编写器将每一帧传输给实用程序（*ffmpeg* / *imagemagick*），然后将它们全部拼接在一起以创建动画。
#
# - 基于文件的编写器 - `~matplotlib.animation.FFMpegFileWriter` 和 `~matplotlib.animation.ImageMagickFileWriter` 是基于文件的编写器。
#   这些编写器比基于管道的替代方案慢，但对于调试更有用，因为它们会将每一帧保存到文件中，然后再将它们拼接成动画。
#
# 保存动画
# -----------------
#
# .. list-table::
#    :header-rows: 1
#
#    * - Writer
#      - Supported Formats
#    * - `~matplotlib.animation.PillowWriter`
#      - .gif, .apng, .webp
#    * - `~matplotlib.animation.HTMLWriter`
#      - .htm, .html, .png
#    * - | `~matplotlib.animation.FFMpegWriter`
#        | `~matplotlib.animation.FFMpegFileWriter`
#      - All formats supported by |ffmpeg|_: ``ffmpeg -formats``
#    * - | `~matplotlib.animation.ImageMagickWriter`
#        | `~matplotlib.animation.ImageMagickFileWriter`
#      - All formats supported by |imagemagick|_: ``magick -list format``
#
# .. _ffmpeg: https://www.ffmpeg.org/general.html#Supported-File-Formats_002c-Codecs-or-Features
# .. |ffmpeg| replace:: *ffmpeg*
#
# .. _imagemagick: https://imagemagick.org/script/formats.php#supported
# .. |imagemagick| replace:: *imagemagick*
#
# To save animations using any of the writers, we can use the
# `.animation.Animation.save` method. It takes the *filename* that we want to
# save the animation as and the *writer*, which is either a string or a writer
# object. It also takes an *fps* argument. This argument is different than the
# *interval* argument that `~.animation.FuncAnimation` or
# `~.animation.ArtistAnimation` uses. *fps* determines the frame rate that the
# **saved** animation uses, whereas *interval* determines the frame rate that
# the **displayed** animation uses.
#
# Below are a few examples that show how to save an animation with different
# writers.
#
#
# Pillow writers::
#
#   ani.save(filename="/tmp/pillow_example.gif", writer="pillow")
#   ani.save(filename="/tmp/pillow_example.apng", writer="pillow")
#
# HTML writers::
#
#   ani.save(filename="/tmp/html_example.html", writer="html")
#   ani.save(filename="/tmp/html_example.htm", writer="html")
#   ani.save(filename="/tmp/html_example.png", writer="html")
#
# FFMpegWriter::
#
#   ani.save(filename="/tmp/ffmpeg_example.mkv", writer="ffmpeg")
#   ani.save(filename="/tmp/ffmpeg_example.mp4", writer="ffmpeg")
#   ani.save(filename="/tmp/ffmpeg_example.mjpeg", writer="ffmpeg")
#
# Imagemagick writers::
#
#   ani.save(filename="/tmp/imagemagick_example.gif", writer="imagemagick")
#   ani.save(filename="/tmp/imagemagick_example.webp", writer="imagemagick")
#   ani.save(filename="apng:/tmp/imagemagick_example.apng",
#            writer="imagemagick", extra_args=["-quality", "100"])
#
# (the ``extra_args`` for *apng* are needed to reduce filesize by ~10x)
```