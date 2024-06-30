# `D:\src\scipysrc\seaborn\examples\faceted_histogram.py`

```
"""
Facetting histograms by subsets of data
=======================================

_thumb: .33, .57
"""
# 导入 seaborn 库，用于数据可视化
import seaborn as sns

# 设置 seaborn 的主题风格为 darkgrid
sns.set_theme(style="darkgrid")

# 加载名为 "penguins" 的数据集，并将其存储在变量 df 中
df = sns.load_dataset("penguins")

# 绘制多个直方图，根据不同的数据子集进行分面展示
sns.displot(
    df,  # 使用数据集 df 进行绘图
    x="flipper_length_mm",  # 指定 x 轴的数据为 flipper_length_mm 列
    col="species",  # 按照 species 列的不同取值进行列分面
    row="sex",  # 按照 sex 列的不同取值进行行分面
    binwidth=3,  # 指定直方图的柱宽度为 3
    height=3,  # 指定每个分面的高度为 3
    facet_kws=dict(margin_titles=True),  # 设置分面参数，包括显示标题边距
)
```