# `D:\src\scipysrc\seaborn\examples\anscombes_quartet.py`

```
"""
Anscombe's quartet
==================

_thumb: .4, .4
"""
# 导入 seaborn 库
import seaborn as sns
# 设置 seaborn 样式为 "ticks"
sns.set_theme(style="ticks")

# 加载 Anscombe's quartet 的示例数据集
df = sns.load_dataset("anscombe")

# 在每个数据集内展示线性回归的结果
sns.lmplot(
    # 使用 df 数据集，x 轴为 "x" 列，y 轴为 "y" 列
    data=df, x="x", y="y",
    # 按照 "dataset" 列分组，每行显示两个图，palette 使用 "muted" 调色板
    col="dataset", hue="dataset", col_wrap=2, palette="muted",
    # 置信区间设为 None
    ci=None,
    # 图的高度设为 4
    height=4,
    # 散点图的样式设置，点的大小为 50，透明度为 1
    scatter_kws={"s": 50, "alpha": 1}
)
```