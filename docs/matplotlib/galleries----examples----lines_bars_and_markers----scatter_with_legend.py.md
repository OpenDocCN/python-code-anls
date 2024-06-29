# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\scatter_with_legend.py`

```py
"""
===========================
Scatter plots with a legend
===========================

To create a scatter plot with a legend one may use a loop and create one
`~.Axes.scatter` plot per item to appear in the legend and set the ``label``
accordingly.

The following also demonstrates how transparency of the markers
can be adjusted by giving ``alpha`` a value between 0 and 1.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 设置随机种子，以便结果可重现
np.random.seed(19680801)

# 创建一个图形窗口和一个坐标系
fig, ax = plt.subplots()

# 循环三次，每次创建一个散点图并设置不同的颜色
for color in ['tab:blue', 'tab:orange', 'tab:green']:
    # 创建随机数据
    n = 750
    x, y = np.random.rand(2, n)
    scale = 200.0 * np.random.rand(n)
    # 绘制散点图，设置颜色、大小、标签和透明度
    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=0.3, edgecolors='none')

# 添加图例和网格
ax.legend()
ax.grid(True)

# 显示图形
plt.show()


# %%
# .. _automatedlegendcreation:
#
# Automated legend creation
# -------------------------
#
# Another option for creating a legend for a scatter is to use the
# `.PathCollection.legend_elements` method.  It will automatically try to
# determine a useful number of legend entries to be shown and return a tuple of
# handles and labels. Those can be passed to the call to `~.axes.Axes.legend`.


# 创建一些随机数据
N = 45
x, y = np.random.rand(2, N)
c = np.random.randint(1, 5, size=N)
s = np.random.randint(10, 220, size=N)

# 创建图形窗口和坐标系
fig, ax = plt.subplots()

# 绘制散点图，并保留返回的散点对象
scatter = ax.scatter(x, y, c=c, s=s)

# 使用散点对象的方法创建图例，显示不同颜色的分类
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
# 将图例添加到坐标系中
ax.add_artist(legend1)

# 使用散点对象的方法创建另一个图例，显示不同大小的分类
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

# 显示图形
plt.show()


# %%
# Further arguments to the `.PathCollection.legend_elements` method
# can be used to steer how many legend entries are to be created and how they
# should be labeled. The following shows how to use some of them.

# 创建更多的随机数据
volume = np.random.rayleigh(27, size=40)
amount = np.random.poisson(10, size=40)
ranking = np.random.normal(size=40)
price = np.random.uniform(1, 10, size=40)

# 创建图形窗口和坐标系
fig, ax = plt.subplots()

# 绘制散点图，设置不同的大小和颜色
# 通过将价格转换为合适的点大小，同时设置颜色和标签
scatter = ax.scatter(volume, amount, c=ranking, s=0.3*(price*3)**2,
                     vmin=-3, vmax=3, cmap="Spectral")

# 创建图例，显示不同排名的颜色分类
legend1 = ax.legend(*scatter.legend_elements(num=5),
                    loc="upper left", title="Ranking")
# 将图例添加到坐标系中
ax.add_artist(legend1)

# 创建图例，显示不同价格的点大小分类，并以美元形式显示价格
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper right", title="Prices (dollars)")

# 显示图形
plt.show()
# 定义关键字参数字典，用于设置图例属性：
# - prop="sizes": 图例中显示的尺寸属性
# - num=5: 图例中显示的数量
# - color=scatter.cmap(0.7): 图例的颜色，使用散点图的 colormap 中的 70% 颜色
# - fmt="$ {x:.2f}": 标签格式，显示为货币格式，保留两位小数
# - func=lambda s: np.sqrt(s/.3)/3: 自定义函数，用于处理尺寸数据
kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="$ {x:.2f}",
          func=lambda s: np.sqrt(s/.3)/3)

# 创建散点图的图例元素，并赋值给 legend2
legend2 = ax.legend(*scatter.legend_elements(**kw),
                    loc="lower right", title="Price")

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    本示例中展示了以下函数、方法、类和模块的使用：
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`: 创建散点图
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`: 添加图例
#    - `matplotlib.collections.PathCollection.legend_elements`: 生成图例元素
```