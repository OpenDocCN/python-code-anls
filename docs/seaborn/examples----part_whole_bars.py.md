# `D:\src\scipysrc\seaborn\examples\part_whole_bars.py`

```
"""
Horizontal bar plots
====================

"""
# 导入 seaborn 和 matplotlib.pyplot 库
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# 初始化 matplotlib 图形
# 创建一个 6x15 英寸大小的子图
f, ax = plt.subplots(figsize=(6, 15))

# 载入示例数据集 'car_crashes'，并按 'total' 列降序排序
crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

# 绘制总碰撞次数的水平条形图
sns.set_color_codes("pastel")
sns.barplot(x="total", y="abbrev", data=crashes,
            label="Total", color="b")

# 绘制涉及酒精的碰撞次数的水平条形图
sns.set_color_codes("muted")
sns.barplot(x="alcohol", y="abbrev", data=crashes,
            label="Alcohol-involved", color="b")

# 添加图例和信息性的坐标轴标签
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)
```