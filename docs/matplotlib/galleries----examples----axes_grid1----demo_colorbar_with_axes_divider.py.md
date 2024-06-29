# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_colorbar_with_axes_divider.py`

```py
# 导入 matplotlib.pyplot 库，用于绘图操作
import matplotlib.pyplot as plt

# 从 mpl_toolkits.axes_grid1.axes_divider 模块中导入 make_axes_locatable 函数
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# 创建一个包含两个子图的图像对象，子图水平排列，调整子图之间的间距
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.subplots_adjust(wspace=0.5)

# 在第一个子图 ax1 上显示二维图像，传入二维数组作为数据
im1 = ax1.imshow([[1, 2], [3, 4]])
# 使用 make_axes_locatable 函数创建 ax1 的 AxesDivider 对象
ax1_divider = make_axes_locatable(ax1)
# 在 ax1 的右侧添加一个新的 Axes 对象，设置它的宽度为原始 Axes 的百分之七，间距为百分之二
cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
# 在图像 fig 上创建一个颜色条，将其绑定到新添加的 Axes 对象 cax1 上
cb1 = fig.colorbar(im1, cax=cax1)

# 在第二个子图 ax2 上显示二维图像，传入二维数组作为数据
im2 = ax2.imshow([[1, 2], [3, 4]])
# 使用 make_axes_locatable 函数创建 ax2 的 AxesDivider 对象
ax2_divider = make_axes_locatable(ax2)
# 在 ax2 的顶部添加一个新的 Axes 对象，设置它的高度为原始 Axes 的百分之七，间距为百分之二
cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
# 在图像 fig 上创建一个水平方向的颜色条，将其绑定到新添加的 Axes 对象 cax2 上
cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
# 更改颜色条的刻度位置为顶部，默认位置为底部，这样可以避免刻度与图像重叠
cax2.xaxis.set_ticks_position("top")

# 显示图形
plt.show()
```