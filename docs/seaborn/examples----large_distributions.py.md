# `D:\src\scipysrc\seaborn\examples\large_distributions.py`

```
"""
Plotting large distributions
============================

"""
# 导入 seaborn 库并设置主题为白色网格样式
import seaborn as sns
sns.set_theme(style="whitegrid")

# 加载钻石数据集
diamonds = sns.load_dataset("diamonds")
# 定义钻石清晰度的排序列表
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# 绘制箱线图，显示钻石清晰度与克拉数之间的关系
sns.boxenplot(
    diamonds, x="clarity", y="carat",  # x 轴为清晰度，y 轴为克拉数
    color="b",  # 箱线图的颜色为蓝色
    order=clarity_ranking,  # 使用预定义的清晰度顺序
    width_method="linear",  # 箱线图宽度的计算方法为线性
)
```