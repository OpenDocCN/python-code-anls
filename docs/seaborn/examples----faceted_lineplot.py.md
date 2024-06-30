# `D:\src\scipysrc\seaborn\examples\faceted_lineplot.py`

```
"""
Line plots on multiple facets
=============================

_thumb: .48, .42

"""
# 导入 seaborn 库，用于绘图
import seaborn as sns
# 设置 seaborn 的绘图风格为 "ticks"
sns.set_theme(style="ticks")

# 加载名为 "dots" 的数据集
dots = sns.load_dataset("dots")

# 将调色板定义为列表，以指定精确的颜色值
palette = sns.color_palette("rocket_r")

# 在两个分面上绘制线图
sns.relplot(
    data=dots,                   # 使用 dots 数据集
    x="time", y="firing_rate",   # x 轴为时间，y 轴为射频
    hue="coherence",             # 根据 coherence 参数设置颜色
    size="choice",               # 根据 choice 参数设置点的大小
    col="align",                 # 根据 align 参数分列显示
    kind="line",                 # 绘制线图
    size_order=["T1", "T2"],     # 指定点大小的顺序
    palette=palette,             # 使用定义的调色板
    height=5,                    # 图的高度为 5
    aspect=.75,                  # 图的长宽比为 0.75
    facet_kws=dict(sharex=False),  # 分面设置，不共享 x 轴
)
```