# `D:\src\scipysrc\matplotlib\galleries\tutorials\images.py`

```py
# %%
# .. _importing_data:
#
# Importing image data into Numpy arrays
# ======================================
#
# Matplotlib relies on the Pillow_ library to load image data.
#
# .. _Pillow: https://pillow.readthedocs.io/en/latest/
#
# Here's the image we're going to play with:
#
# .. image:: ../_static/stinkbug.png
#
# It's a 24-bit RGB PNG image (8 bits for each of R, G, B).  Depending
# on where you get your data, the other kinds of image that you'll most
# likely encounter are RGBA images, which allow for transparency, or
# single-channel grayscale (luminosity) images.  Download `stinkbug.png
# <https://raw.githubusercontent.com/matplotlib/matplotlib/main/doc/_static/stinkbug.png>`_
# to your computer for the rest of this tutorial.
#
# 使用 Pillow 打开图像文件，并将其转换为一个 8 位无符号整数（dtype=uint8）的 numpy 数组
img = np.asarray(Image.open('../../doc/_static/stinkbug.png'))
print(repr(img))

# %%
# 每个内部列表代表一个像素点。对于 RGB 图像，每个像素有 3 个值。
# 由于这是一张黑白图像，R、G 和 B 都是相似的。对于 RGBA 图像（其中 A 表示透明度），每个像素有 4 个值。
# 而灰度图像只有一个值，因此只是一个 2-D 数组，而不是 3-D 数组。
# 对于 RGB 和 RGBA 图像，Matplotlib 支持 float32 和 uint8 数据类型。
# 对于灰度图像，Matplotlib 只支持 float32。如果你的数组数据不符合这些描述，你需要重新缩放它。
#
# .. _plotting_data:
#
# 将 numpy 数组绘制为图像
# ===================================
#
# 现在，你已经有了一个 numpy 数组（通过导入或生成）。让我们来渲染它。
# 在 Matplotlib 中，使用 :func:`~matplotlib.pyplot.imshow` 函数来完成这个操作。
# 这里我们获取绘图对象。这个对象提供了一个简单的方式来从提示符操作绘图。
imgplot = plt.imshow(img)

# %%
# 你也可以绘制任何 numpy 数组。
#
# .. _Pseudocolor:
#
# 将伪彩色方案应用于图像绘制
# -------------------------------------------------
#
# 伪彩色是增强对比度和更轻松地可视化数据的有用工具。
# 在使用投影仪进行数据演示时特别有用，因为它们的对比度通常非常低。
# 伪彩色只适用于单通道、灰度、亮度图像。我们目前有一幅 RGB 图像。
# 由于 R、G 和 B 都是相似的（可以在上面或你的数据中自行查看），
# 我们可以只选择数据的一个通道，使用数组切片（你可以在 `Numpy 教程 <https://numpy.org/doc/stable/user/quickstart.html#indexing-slicing-and-iterating>`_ 中了解更多）：
lum_img = img[:, :, 0]
plt.imshow(lum_img)

# %%
# 现在，对于亮度（2D、无颜色）图像，默认使用 colormap（又称查找表，LUT），默认为 viridis。
# 还有许多其他选择。
plt.imshow(lum_img, cmap="hot")

# %%
# 注意，你也可以使用 :meth:`~matplotlib.cm.ScalarMappable.set_cmap` 方法在现有的绘图对象上更改 colormap：
imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')

# %%
#
# .. note::
#
#    请记住，在具有 inline 后端的 Jupyter Notebook 中，你无法更改已渲染的绘图。
#    如果你在一个单元格中创建了 imgplot，那么你不能在后续单元格中调用 set_cmap() 来更改它并期望早期的绘图会改变。
#    确保你在一个单元格中一起输入这些命令。plt 命令不会更改较早单元格中的绘图。
#
# 图像显示函数，显示灰度图像 lum_img
imgplot = plt.imshow(lum_img)
# 添加颜色条，用于标识图像的数值范围
plt.colorbar()

# %%
# 显示灰度图像 lum_img 的直方图，以了解数据分布
plt.hist(lum_img.ravel(), bins=range(256), fc='k', ec='k')

# %%
# 调整图像显示范围，设置色彩限制 clim=(0, 175)，放大感兴趣区域
plt.imshow(lum_img, clim=(0, 175))

# %%
# 通过调用 imgplot 对象的 set_clim 方法，再次设置图像的色彩限制
imgplot = plt.imshow(lum_img)
imgplot.set_clim(0, 175)

# %%
# 加载图像并调整尺寸，使用 Pillow 库
img = Image.open('../../doc/_static/stinkbug.png')
img.thumbnail((64, 64))  # 原地调整图像大小
imgplot = plt.imshow(img, interpolation='nearest')

# %%
# 默认使用最近邻插值方法进行图像显示，以填充图像调整后的空白部分
# 导入 matplotlib.pyplot 库，并使用其 imshow 函数显示图像，设置插值方式为 "bilinear"
imgplot = plt.imshow(img, interpolation="bilinear")

# %%
# 导入 matplotlib.pyplot 库，并使用其 imshow 函数显示图像，设置插值方式为 "bicubic"
imgplot = plt.imshow(img, interpolation="bicubic")

# %%
# bicubic 插值通常用于放大照片时，人们倾向于选择模糊而不是像素化的效果。
# bicubic 插值在放大图像时比较常用，它能够在保持图像平滑的同时，尽可能减少像素化的现象。
```