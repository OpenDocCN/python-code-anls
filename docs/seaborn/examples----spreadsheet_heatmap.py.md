# `D:\src\scipysrc\seaborn\examples\spreadsheet_heatmap.py`

```
"""
Annotated heatmaps
==================

"""
# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt
# 导入seaborn库，并设置其默认主题
import seaborn as sns
sns.set_theme()

# 加载示例航班数据集，并将其转换为长格式（long-form）
flights_long = sns.load_dataset("flights")
# 使用pivot方法将长格式数据转换为以月份为索引、年份为列名、乘客数为值的宽格式数据
flights = (
    flights_long
    .pivot(index="month", columns="year", values="passengers")
)

# 创建一个新的图形和轴对象，指定图形大小为9x6英寸
f, ax = plt.subplots(figsize=(9, 6))
# 绘制热力图，其中每个单元格显示数值，并标注在每个单元格内，使用整数格式显示数值
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
```