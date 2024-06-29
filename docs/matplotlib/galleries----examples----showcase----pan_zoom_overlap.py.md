# `D:\src\scipysrc\matplotlib\galleries\examples\showcase\pan_zoom_overlap.py`

```
# 导入 matplotlib 库
import matplotlib.pyplot as plt

# 创建一个新的图形对象，指定尺寸为 11x6 英寸
fig = plt.figure(figsize=(11, 6))

# 设置整个图形的标题
fig.suptitle("Showcase for pan/zoom events on overlapping axes.")

# 在图形中添加一个主要的坐标轴，位置从左下角 (0.05, 0.05) 开始，宽度为 90%，高度为 90%
ax = fig.add_axes((.05, .05, .9, .9))

# 设置主要坐标轴的补丁（背景）颜色为浅灰色
ax.patch.set_color(".75")

# 在主要坐标轴上添加一个与其共享 x 轴的副坐标轴
ax_twin = ax.twinx()

# 在图形中添加第一个子图，编号为 221
ax1 = fig.add_subplot(221)

# 在第一个子图上添加一个与其共享 x 轴的副坐标轴
ax1_twin = ax1.twinx()

# 在第一个子图上添加文本说明，说明内容包括：
# - 可见的补丁
# - 指示不将平移/缩放事件转发到下方的坐标轴
ax1.text(.5, .5,
         "Visible patch\n\n"
         "Pan/zoom events are NOT\n"
         "forwarded to axes below",
         ha="center", va="center", transform=ax1.transAxes)

# 在图形中添加第二个子图，编号为 222
ax2 = fig.add_subplot(222)

# 在第二个子图上添加一个与其共享 x 轴的副坐标轴
ax2_twin = ax2.twinx()

# 设置第二个子图的补丁（背景）不可见
ax2.patch.set_visible(False)

# 在第二个子图上添加文本说明，说明内容包括：
# - 不可见的补丁
# - 指示将平移/缩放事件转发到下方的坐标轴
ax2.text(.5, .5,
         "Invisible patch\n\n"
         "Pan/zoom events are\n"
         "forwarded to axes below",
         ha="center", va="center", transform=ax2.transAxes)

# 在图形中添加第三个子图，编号为 223，并且与第一个子图共享 x 和 y 轴
ax11 = fig.add_subplot(223, sharex=ax1, sharey=ax1)

# 设置第三个子图将平移/缩放事件转发到下方的坐标轴
ax11.set_forward_navigation_events(True)

# 在第三个子图上添加文本说明，说明内容包括：
# - 可见的补丁
# - 指示覆盖捕捉行为，将平移/缩放事件转发到下方的坐标轴
ax11.text(.5, .5,
          "Visible patch\n\n"
          "Override capture behavior:\n\n"
          "ax.set_forward_navigation_events(True)",
          ha="center", va="center", transform=ax11.transAxes)

# 在图形中添加第四个子图，编号为 224，并且与第二个子图共享 x 和 y 轴
ax22 = fig.add_subplot(224, sharex=ax2, sharey=ax2)

# 设置第四个子图不转发平移/缩放事件到下方的坐标轴
ax22.set_forward_navigation_events(False)

# 在第四个子图上添加文本说明，说明内容包括：
# - 不可见的补丁
# - 指示覆盖捕捉行为，不将平移/缩放事件转发到下方的坐标轴
ax22.text(.5, .5,
          "Invisible patch\n\n"
          "Override capture behavior:\n\n"
          "ax.set_forward_navigation_events(False)",
          ha="center", va="center", transform=ax22.transAxes)
```