# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\image_annotated_heatmap.py`

```
"""
===========================
Creating annotated heatmaps
===========================

It is often desirable to show data which depends on two independent
variables as a color coded image plot. This is often referred to as a
heatmap. If the data is categorical, this would be called a categorical
heatmap.

Matplotlib's `~matplotlib.axes.Axes.imshow` function makes
production of such plots particularly easy.

The following examples show how to create a heatmap with annotations.
We will start with an easy example and expand it to be usable as a
universal function.
"""


# %%
#
# A simple categorical heatmap
# ----------------------------
#
# We may start by defining some data. What we need is a 2D list or array
# which defines the data to color code. We then also need two lists or arrays
# of categories; of course the number of elements in those lists
# need to match the data along the respective axes.
# The heatmap itself is an `~matplotlib.axes.Axes.imshow` plot
# with the labels set to the categories we have.
# Note that it is important to set both, the tick locations
# (`~matplotlib.axes.Axes.set_xticks`) as well as the
# tick labels (`~matplotlib.axes.Axes.set_xticklabels`),
# otherwise they would become out of sync. The locations are just
# the ascending integer numbers, while the ticklabels are the labels to show.
# Finally, we can label the data itself by creating a `~matplotlib.text.Text`
# within each cell showing the value of that cell.

# Importing necessary libraries
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于处理数值数组

import matplotlib  # 导入 matplotlib 库
import matplotlib as mpl  # 导入 matplotlib 库并使用 mpl 别名

# sphinx_gallery_thumbnail_number = 2

# Define categories
vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]  # 定义蔬菜类别列表
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]  # 定义农民类别列表

# Define data as a 2D numpy array
harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

# Create a figure and axis for the plot
fig, ax = plt.subplots()  # 创建绘图的图形和坐标轴对象
im = ax.imshow(harvest)  # 在坐标轴上绘制热图

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(farmers)))  # 设置 x 轴刻度位置为农民列表的长度
ax.set_xticklabels(farmers)  # 设置 x 轴刻度标签为农民列表的名称
ax.set_yticks(np.arange(len(vegetables)))  # 设置 y 轴刻度位置为蔬菜列表的长度
ax.set_yticklabels(vegetables)  # 设置 y 轴刻度标签为蔬菜列表的名称

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# 设置 x 轴刻度标签的旋转角度为 45 度，水平对齐方式为右对齐

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):  # 遍历蔬菜列表的长度
    for j in range(len(farmers)):  # 遍历农民列表的长度
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")  # 在每个单元格中心添加文本标注

ax.set_title("Harvest of local farmers (in tons/year)")  # 设置图表标题
fig.tight_layout()  # 调整图表布局以防止重叠
plt.show()  # 显示图表


# %%
# Using the helper function code style
# ------------------------------------
#
# As discussed in the :ref:`Coding styles <coding_styles>`
# one might want to reuse such code to create some kind of heatmap
# for different input data and/or on different axes.
# We create a function that takes the data and the row and column labels as
# input, and allows arguments that are used to customize the plot
#
# Here, in addition to the above we also want to create a colorbar and
# position the labels above of the heatmap instead of below it.
# The annotations shall get different colors depending on a threshold
# for better contrast against the pixel color.
# Finally, we turn the surrounding axes spines off and create
# a grid of white lines to separate the cells.


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()  # 如果未提供轴对象，则使用当前的Axes对象

    if cbar_kw is None:
        cbar_kw = {}  # 如果未提供colorbar的关键字参数字典，则初始化为空字典

    # 绘制热图
    im = ax.imshow(data, **kwargs)

    # 创建colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # 显示所有刻度，并用相应的列表条目标记它们
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # 让水平轴标签出现在顶部
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # 旋转刻度标签并设置它们的对齐方式
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # 关闭边框并创建白色网格
    ax.spines[:].set_visible(False)  # 关闭所有边框

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)  # 设置x轴次要刻度
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)  # 设置y轴次要刻度
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)  # 创建白色网格
    ax.tick_params(which="minor", bottom=False, left=False)  # 关闭次要刻度的刻度线

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    # 根据提供的数据和参数绘制热图上的注释文本，并返回所有创建的文本对象列表
    def annotate_heatmap(data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """
        # 如果数据不是列表或者 NumPy 数组，将从图像对象中获取数据
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()
    
        # 将阈值标准化到图像颜色范围内
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            # 如果阈值为 None，则将其设置为数据最大值的一半作为默认值
            threshold = im.norm(data.max())/2.
    
        # 设置默认文本对齐方式为居中，但允许通过 textkw 参数覆盖
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)
    
        # 如果 valfmt 是字符串，则将其转换为格式化文本的方法
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    
        # 遍历数据并为每个“像素”创建一个 `Text` 对象
        # 根据数据的值改变文本的颜色
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # 根据数据的规范化值决定文本的颜色
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                # 在图像的坐标位置 (j, i) 创建文本标签，并使用给定的格式化方法格式化数据值
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
    
        return texts
# %%
# 上面的代码现在允许我们将实际的绘图过程保持相当紧凑。
#

fig, ax = plt.subplots()

# 调用 heatmap 函数生成热图，并返回图像对象 im 和 colorbar 对象 cbar
im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="YlGn", cbarlabel="harvest [t/year]")

# 调用 annotate_heatmap 函数对热图进行标注，并使用 valfmt 格式化标注文本
texts = annotate_heatmap(im, valfmt="{x:.1f} t")

# 调整图像布局，使其紧凑
fig.tight_layout()

# 显示图形
plt.show()


# %%
# 更复杂的热图示例
# ----------------------------------
#
# 在接下来的代码中，我们展示了之前创建的函数的多样性，通过在不同情况下应用它们和使用不同的参数来展示。
#

np.random.seed(19680801)

# 创建一个 2x2 的子图布局，并设置图像大小为 (8, 6)
fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# 使用不同的字体大小和颜色图表，复制上面的示例
im, _ = heatmap(harvest, vegetables, farmers, ax=ax,
                cmap="Wistia", cbarlabel="harvest [t/year]")
annotate_heatmap(im, valfmt="{x:.1f}", size=7)

# 创建一些新数据，给 imshow 提供更多的参数（vmin），在标注上使用整数格式，并指定一些颜色。
data = np.random.randint(2, 100, size=(7, 7))
y = [f"Book {i}" for i in range(1, 8)]
x = [f"Store {i}" for i in list("ABCDEFG")]
im, _ = heatmap(data, y, x, ax=ax2, vmin=0,
                cmap="magma_r", cbarlabel="weekly sold copies")
annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=20,
                 textcolors=("red", "white"))

# 有时数据本身是分类的。这里使用 `matplotlib.colors.BoundaryNorm` 将数据分成类别，并用于着色图，同时从类别数组获取类别标签。
data = np.random.randn(6, 6)
y = [f"Prod. {i}" for i in range(10, 70, 10)]
x = [f"Cycle {i}" for i in range(1, 7)]

qrates = list("ABCDEFG")
norm = matplotlib.colors.BoundaryNorm(np.linspace(-3.5, 3.5, 8), 7)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

im, _ = heatmap(data, y, x, ax=ax3,
                cmap=mpl.colormaps["PiYG"].resampled(7), norm=norm,
                cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
                cbarlabel="Quality Rating")

annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
                 textcolors=("red", "black"))

# 我们可以很好地绘制一个相关矩阵。由于相关系数介于 -1 和 1 之间，我们使用这些作为 vmin 和 vmax。我们还可以通过 `matplotlib.ticker.FuncFormatter` 移除前导零，并隐藏对角线元素（它们全部为 1）。

corr_matrix = np.corrcoef(harvest)
im, _ = heatmap(corr_matrix, vegetables, vegetables, ax=ax4,
                cmap="PuOr", vmin=-1, vmax=1,
                cbarlabel="correlation coeff.")

# 定义一个格式化函数 func，用于标注
def func(x, pos):
    return f"{x:.2f}".replace("0.", ".").replace("1.00", "")

annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

# 调整子图布局，使其紧凑
plt.tight_layout()

# 显示图形
plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
# 导入 `matplotlib` 库中的 `Axes` 类和 `Figure` 类
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# 导入 `imshow` 函数，用于在图形中显示图像数据
from matplotlib.axes.Axes.imshow import imshow
from matplotlib.pyplot.imshow import imshow
# 导入 `colorbar` 函数，用于在图形中添加颜色条
from matplotlib.figure.Figure.colorbar import colorbar
from matplotlib.pyplot.colorbar import colorbar
```