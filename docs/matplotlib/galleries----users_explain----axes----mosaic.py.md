# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\mosaic.py`

```
# 导入 matplotlib.pyplot 库，命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，命名为 np
import numpy as np

# 辅助函数，用于在下面的示例中识别 Axes
def identify_axes(ax_dict, fontsize=48):
    """
    辅助函数，用于在下面的示例中识别 Axes。

    在 Axes 的中心绘制大字体标签。

    参数
    ----------
    ax_dict : dict[str, Axes]
        标题或标签与 Axes 对象之间的映射。
    fontsize : int, optional
        标签的字体大小。
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

# %%
# 如果我们想要一个 2x2 的网格，可以使用 `.Figure.subplots`，它返回一个 2D 数组
# 包含 `.axes.Axes`，我们可以通过索引来绘制图表。
np.random.seed(19680801)
hist_data = np.random.randn(1_500)

# 创建一个带有约束布局的新图形对象
fig = plt.figure(layout="constrained")
# 使用 `.Figure.subplots` 创建一个 2x2 的 Axes 数组，squeeze=False 确保返回的是二维数组
ax_array = fig.subplots(2, 2, squeeze=False)

# 在不同的 Axes 上绘制不同的图表
ax_array[0, 0].bar(["a", "b", "c"], [5, 7, 9])
ax_array[0, 1].plot([1, 2, 3])
ax_array[1, 0].hist(hist_data, bins="auto")
ax_array[1, 1].imshow([[1, 2], [2, 1]])

# 调用辅助函数，标识每个 Axes 的位置
identify_axes(
    {(j, k): a for j, r in enumerate(ax_array) for k, a in enumerate(r)},
)

# %%
# 使用 `.Figure.subplot_mosaic` 可以生成相同的图表，但是可以为 Axes 分配语义化的名称

# 创建一个带有约束布局的新图形对象
fig = plt.figure(layout="constrained")
# 使用 `.Figure.subplot_mosaic` 根据给定的布局列表创建一个命名的 Axes 字典
ax_dict = fig.subplot_mosaic(
    [
        ["bar", "plot"],
        ["hist", "image"],
    ],
)
# 在每个命名的 Axes 上绘制不同的图表
ax_dict["bar"].bar(["a", "b", "c"], [5, 7, 9])
ax_dict["plot"].plot([1, 2, 3])
ax_dict["hist"].hist(hist_data)
ax_dict["image"].imshow([[1, 2], [2, 1]])
# 调用辅助函数，标识每个命名的 Axes 的位置
identify_axes(ax_dict)

# %%
# %%
# 输出当前存储的轴字典，显示由subplot_mosaic方法生成的轴布局
print(ax_dict)

# %%
# 字符串简写法
# =================
#
# 通过将轴标签限制为单个字符，我们可以将所需的轴"绘制"为"ASCII艺术"。
# 下面的mosaic字符串

mosaic = """
    AB
    CD
    """

# %%
# 将在2x2网格中给我们生成4个轴，并生成与上述相同的图形mosaic（但现在标记为{"A", "B", "C", "D"}，而不是{"bar", "plot", "hist", "image"}）。

fig = plt.figure(layout="constrained")
ax_dict = fig.subplot_mosaic(mosaic)
identify_axes(ax_dict)

# %%
# 或者，您可以使用更紧凑的字符串表示法
mosaic = "AB;CD"

# %%
# 将给出相同的组合，其中“;”用作行分隔符，而不是换行符。

fig = plt.figure(layout="constrained")
ax_dict = fig.subplot_mosaic(mosaic)
identify_axes(ax_dict)

# %%
# 跨越多行/列的轴
# ===================================
#
# 使用`.Figure.subplot_mosaic`可以实现的一件事情是指定一个轴跨越多行或列。

# %%
# 如果我们想要重新排列我们的四个轴，使“C”成为底部的水平跨度，“D”成为右侧的垂直跨度，则可以执行以下操作

axd = plt.figure(layout="constrained").subplot_mosaic(
    """
    ABD
    CCD
    """
)
identify_axes(axd)

# %%
# 如果我们不想用Axes填满图中的所有空间，我们可以指定网格中的某些空间为空白

axd = plt.figure(layout="constrained").subplot_mosaic(
    """
    A.C
    BBB
    .D.
    """
)
identify_axes(axd)

# %%
# 如果我们希望使用另一个字符（而不是句号“.”）来标记空白空间，可以使用empty_sentinel指定要使用的字符。

axd = plt.figure(layout="constrained").subplot_mosaic(
    """
    aX
    Xb
    """,
    empty_sentinel="X",
)
identify_axes(axd)

# %%
#
# 内部对我们使用的字母没有附加任何含义，任何Unicode代码点都是有效的！

axd = plt.figure(layout="constrained").subplot_mosaic(
    """αб
       ℝ☢"""
)
identify_axes(axd)

# %%
# 不建议在字符串简写法中使用空格作为标签或空标记，因为在处理输入时可能会被剥离。
#
# 控制mosaic的创建
# ===========================
#
# 此功能构建在.gridspec之上，您可以将关键字参数传递给底层的.gridspec.GridSpec（与.Figure.subplots相同）。
#
# 在这种情况下，我们想要使用输入来指定排列，但设置行/列的相对宽度。为了方便起见，
# .gridspec.GridSpec的height_ratios和width_ratios在
# `.Figure.subplot_mosaic` 调用序列。

axd = plt.figure(layout="constrained").subplot_mosaic(
    """
    .a.
    bAc
    .d.
    """,
    # 设置行的高度比例
    height_ratios=[1, 3.5, 1],
    # 设置列的宽度比例
    width_ratios=[1, 3.5, 1],
)
# 识别和标识每个子图的轴
identify_axes(axd)

# %%
# 其它 `.gridspec.GridSpec` 的关键字可以通过 *gridspec_kw* 参数传递。例如，
# 使用 {*left*, *right*, *bottom*, *top*} 关键字参数可以定位整体的镶嵌图，
# 在一个图中放置多个版本的相同镶嵌图。

mosaic = """AA
            BC"""
fig = plt.figure()
axd = fig.subplot_mosaic(
    mosaic,
    gridspec_kw={
        "bottom": 0.25,
        "top": 0.95,
        "left": 0.1,
        "right": 0.5,
        "wspace": 0.5,
        "hspace": 0.5,
    },
)
# 识别和标识每个子图的轴
identify_axes(axd)

axd = fig.subplot_mosaic(
    mosaic,
    gridspec_kw={
        "bottom": 0.05,
        "top": 0.75,
        "left": 0.6,
        "right": 0.95,
        "wspace": 0.5,
        "hspace": 0.5,
    },
)
# 识别和标识每个子图的轴
identify_axes(axd)

# %%
# 或者，您可以使用子图功能：

mosaic = """AA
            BC"""
fig = plt.figure(layout="constrained")
left, right = fig.subfigures(nrows=1, ncols=2)
axd = left.subplot_mosaic(mosaic)
# 识别和标识每个子图的轴
identify_axes(axd)

axd = right.subplot_mosaic(mosaic)
# 识别和标识每个子图的轴
identify_axes(axd)


# %%
# 控制子图的创建
# ============================
#
# 我们还可以传递用于创建子图的参数（与 `.Figure.subplots` 相同），
# 这些参数将应用于创建的所有 Axes。

axd = plt.figure(layout="constrained").subplot_mosaic(
    "AB", subplot_kw={"projection": "polar"}
)
# 识别和标识每个子图的轴
identify_axes(axd)

# %%
# 每个子图的关键字参数
# ----------------------------------
#
# 如果需要分别控制每个子图的参数，可以使用 *per_subplot_kw*，
# 通过将 Axes 标识符（或 Axes 标识符的元组）映射到关键字字典来传递。

fig, axd = plt.subplot_mosaic(
    "AB;CD",
    per_subplot_kw={
        "A": {"projection": "polar"},
        ("C", "D"): {"xscale": "log"}
    },
)
# 识别和标识每个子图的轴
identify_axes(axd)

# %%
# 如果布局使用字符串简写指定，那么我们知道 Axes 标签将是单个字符，
# 可以明确解释 *per_subplot_kw* 中较长的字符串，以指定要应用关键字的一组 Axes：

fig, axd = plt.subplot_mosaic(
    "AB;CD",
    per_subplot_kw={
        "AD": {"projection": "polar"},
        "BC": {"facecolor": ".9"}
    },
)
# 识别和标识每个子图的轴
identify_axes(axd)

# %%
# 如果 *subplot_kw* 和 *per_subplot_kw* 一起使用，则它们将合并，
# *per_subplot_kw* 具有优先权：

axd = plt.figure(layout="constrained").subplot_mosaic(
    "AB;CD",
    subplot_kw={"facecolor": "xkcd:tangerine"},
    per_subplot_kw={
        "B": {"facecolor": "xkcd:water blue"},
        "D": {"projection": "polar", "facecolor": "w"},
    }
)
# 调用函数 identify_axes，传入参数 axd，用于标识和处理绘图中的坐标轴
identify_axes(axd)


# %%
# 嵌套列表作为输入
# =================
#
# 当我们传入一个列表时（内部转换为嵌套列表），我们可以像使用字符串简写一样使用 spans、blanks 和 *gridspec_kw*，例如：

# 使用 layout="constrained" 创建一个画布，并使用 subplot_mosaic 方法布局子图
# 将子图分布定义为一个嵌套列表
axd = plt.figure(layout="constrained").subplot_mosaic(
    [
        ["main", "zoom"],
        ["main", "BLANK"],
    ],
    empty_sentinel="BLANK",
    width_ratios=[2, 1],
)
# 标识和处理布局后的坐标轴
identify_axes(axd)


# %%
# 此外，使用列表输入，我们可以指定嵌套的mosaic。内部列表的任意元素可以是另一个嵌套列表：

# 定义一个内部嵌套列表
inner = [
    ["inner A"],
    ["inner B"],
]

# 定义一个外部嵌套 mosaic
outer_nested_mosaic = [
    ["main", inner],
    ["bottom", "bottom"],
]
# 使用 layout="constrained" 创建一个画布，并使用 subplot_mosaic 方法布局子图
axd = plt.figure(layout="constrained").subplot_mosaic(
    outer_nested_mosaic, empty_sentinel=None
)
# 标识和处理布局后的坐标轴，设置字体大小为 36
identify_axes(axd, fontsize=36)


# %%
# 我们还可以传入一个二维的 NumPy 数组来做一些事情，比如：

# 创建一个 4x4 的全零整型 NumPy 数组
mosaic = np.zeros((4, 4), dtype=int)
for j in range(4):
    mosaic[j, j] = j + 1
# 使用 layout="constrained" 创建一个画布，并使用 subplot_mosaic 方法布局子图
axd = plt.figure(layout="constrained").subplot_mosaic(
    mosaic,
    empty_sentinel=0,
)
# 标识和处理布局后的坐标轴
identify_axes(axd)
```