# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\image_clip_path.py`

```py
```python`
"""
============================
Clipping images with patches
============================

Demo of image that's been clipped by a circular patch.
"""
# 导入 matplotlib.pyplot 库，通常用于绘图
import matplotlib.pyplot as plt

# 导入 matplotlib.cbook 库，提供一些工具函数
import matplotlib.cbook as cbook

# 导入 matplotlib.patches 库，提供绘制各种形状的工具
import matplotlib.patches as patches

# 使用 cbook.get_sample_data 方法获取样本图片文件路径，并读取图片数据
with cbook.get_sample_data('grace_hopper.jpg') as image_file:
    image = plt.imread(image_file)

# 创建一个图形和一个坐标轴对象
fig, ax = plt.subplots()
# 在坐标轴上显示读取的图片
im = ax.imshow(image)
# 创建一个圆形补丁，圆心在 (260, 200)，半径为 200，使用坐标轴数据的变换
patch = patches.Circle((260, 200), radius=200, transform=ax.transData)
# 设置图片的裁剪路径为圆形补丁
im.set_clip_path(patch)

# 关闭坐标轴的显示
ax.axis('off')
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.artist.Artist.set_clip_path`
```