# `D:\src\scipysrc\seaborn\examples\grouped_boxplot.py`

```
"""
Grouped boxplots
================

_thumb: .66, .45

"""
# 导入 seaborn 库，并设置绘图风格为 ticks，调色板为 pastel
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

# 加载示例数据集 tips
tips = sns.load_dataset("tips")

# 绘制嵌套箱线图，显示不同天和时间段的账单数据，按吸烟者与非吸烟者分类，颜色分别为绿色和品红色
sns.boxplot(x="day", y="total_bill",
            hue="smoker", palette=["m", "g"],
            data=tips)

# 移除图形周围的轴线，设置偏移量为10，并修剪多余的空白区域
sns.despine(offset=10, trim=True)
```