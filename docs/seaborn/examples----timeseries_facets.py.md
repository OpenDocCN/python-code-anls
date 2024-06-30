# `D:\src\scipysrc\seaborn\examples\timeseries_facets.py`

```
"""
Small multiple time series
--------------------------

_thumb: .42, .58

"""
# 导入 seaborn 库
import seaborn as sns

# 设置 seaborn 的主题为暗色风格
sns.set_theme(style="dark")

# 加载名为 "flights" 的示例数据集
flights = sns.load_dataset("flights")

# 绘制每年乘客数量随月份变化的时间序列，每年的时间序列绘制在单独的面板中
g = sns.relplot(
    data=flights,  # 使用 flights 数据集
    x="month", y="passengers", col="year", hue="year",  # x 轴为月份，y 轴为乘客数量，按年份分列，按年份着色
    kind="line", palette="crest", linewidth=4, zorder=5,  # 使用线图类型，使用 "crest" 调色板，设置线宽为 4，设置 Z 轴顺序为 5
    col_wrap=3, height=2, aspect=1.5, legend=False,  # 每行显示 3 列面板，设置面板高度为 2，宽高比为 1.5，不显示图例
)

# 遍历每个子图面板以进一步自定义
for year, ax in g.axes_dict.items():

    # 将年份作为标题注释添加到图中
    ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

    # 绘制每年的时间序列在背景中
    sns.lineplot(
        data=flights, x="month", y="passengers", units="year",  # x 轴为月份，y 轴为乘客数量，每年一单位
        estimator=None, color=".7", linewidth=1, ax=ax,  # 不使用估计器，颜色为灰度 0.7，线宽为 1，在当前子图面板中绘制
    )

# 减少 x 轴刻度的频率
ax.set_xticks(ax.get_xticks()[::2])

# 调整图的支持元素
g.set_titles("")  # 设置空标题
g.set_axis_labels("", "Passengers")  # 设置 x 轴和 y 轴标签
g.tight_layout()  # 调整布局使得子图之间紧凑排列
```