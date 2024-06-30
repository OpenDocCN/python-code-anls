# `D:\src\scipysrc\seaborn\examples\pairgrid_dotplot.py`

```
"""
Dot plot with several variables
===============================

_thumb: .3, .3
"""
# 导入 seaborn 库
import seaborn as sns
# 设置 seaborn 样式为白色网格
sns.set_theme(style="whitegrid")

# 加载数据集
crashes = sns.load_dataset("car_crashes")

# 创建 PairGrid 对象
g = sns.PairGrid(crashes.sort_values("total", ascending=False),
                 x_vars=crashes.columns[:-3], y_vars=["abbrev"],
                 height=10, aspect=.25)

# 使用 stripplot 函数绘制点图
g.map(sns.stripplot, size=10, orient="h", jitter=False,
      palette="flare_r", linewidth=1, edgecolor="w")

# 设置所有列的 x 轴限制并添加更好的标签
g.set(xlim=(0, 25), xlabel="Crashes", ylabel="")

# 使用语义化的标题命名列
titles = ["Total crashes", "Speeding crashes", "Alcohol crashes",
          "Not distracted crashes", "No previous crashes"]

# 为每个子图设置不同的标题
for ax, title in zip(g.axes.flat, titles):
    ax.set(title=title)  # 设置每个子图的标题

    # 将网格设置为水平而非垂直
    ax.xaxis.grid(False)  # 关闭竖直方向的网格线
    ax.yaxis.grid(True)   # 打开水平方向的网格线

# 去除左侧和底部的轴线
sns.despine(left=True, bottom=True)
```