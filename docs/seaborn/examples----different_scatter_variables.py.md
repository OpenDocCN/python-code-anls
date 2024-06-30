# `D:\src\scipysrc\seaborn\examples\different_scatter_variables.py`

```
"""
Scatterplot with multiple semantics
===================================

_thumb: .45, .5

"""
# 导入 seaborn 和 matplotlib.pyplot 库
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 seaborn 主题样式为白色网格
sns.set_theme(style="whitegrid")

# 加载示例数据集 diamonds
diamonds = sns.load_dataset("diamonds")

# 创建一个大小为 6.5x6.5 英寸的子图
f, ax = plt.subplots(figsize=(6.5, 6.5))

# 移除子图 f 的左边和底部的轴线
sns.despine(f, left=True, bottom=True)

# 确定钻石清晰度的排序，用于后续图例颜色的排列
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# 绘制散点图，同时将点的颜色和大小分配给数据集中的不同变量
sns.scatterplot(x="carat", y="price",  # x 轴为克拉数，y 轴为价格
                hue="clarity",          # 根据清晰度分配颜色
                size="depth",           # 根据深度分配点的大小
                palette="ch:r=-.2,d=.3_r",  # 自定义调色板
                hue_order=clarity_ranking,  # 按照指定顺序排列清晰度
                sizes=(1, 8),           # 设置点的大小范围
                linewidth=0,            # 线宽为 0，即无边框
                data=diamonds,          # 使用 diamonds 数据集
                ax=ax)                  # 绘制到子图 ax 上
```