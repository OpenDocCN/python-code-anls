# `D:\src\scipysrc\seaborn\examples\simple_violinplots.py`

```
"""
Horizontal, unfilled violinplots
================================

_thumb: .5, .45
"""
# 导入 seaborn 库
import seaborn as sns

# 设置 seaborn 的主题样式
sns.set_theme()

# 加载名为 "seaice" 的数据集
seaice = sns.load_dataset("seaice")

# 将日期数据取年份后四舍五入到最近的十年，存储到新的列 "Decade"
seaice["Decade"] = seaice["Date"].dt.year.round(-1)

# 绘制水平方向的小提琴图，不填充区域
sns.violinplot(seaice, x="Extent", y="Decade", orient="y", fill=False)
```