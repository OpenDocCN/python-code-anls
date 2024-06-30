# `D:\src\scipysrc\seaborn\examples\grouped_violinplots.py`

```
"""
Grouped violinplots with split violins
======================================

_thumb: .44, .47
"""
# 导入 seaborn 库，用于绘图
import seaborn as sns
# 设置 seaborn 的绘图主题为暗色风格
sns.set_theme(style="dark")

# 加载示例数据集 tips
# 这里的 tips 是一个包含餐厅小费信息的数据集
tips = sns.load_dataset("tips")

# 绘制分组的小提琴图，并且分裂小提琴以便比较
# 使用 data=tips 指定数据集，x="day" 表示按照天数分组，y="total_bill" 表示小费总额，hue="smoker" 表示按吸烟者分颜色
# split=True 表示分裂小提琴，inner="quart" 表示小提琴内部展示四分位数，fill=False 表示不填充小提琴内部
# palette={"Yes": "g", "No": ".35"} 设置调色板，表示吸烟者为 "Yes" 的小提琴为绿色 "g"，"No" 的小提琴为灰色 ".35"
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker",
               split=True, inner="quart", fill=False,
               palette={"Yes": "g", "No": ".35"})
```