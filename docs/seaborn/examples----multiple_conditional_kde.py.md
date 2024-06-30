# `D:\src\scipysrc\seaborn\examples\multiple_conditional_kde.py`

```
"""
Conditional kernel density estimate
===================================

_thumb: .4, .5
"""
# 导入 seaborn 库，并设置主题为白色网格
import seaborn as sns
sns.set_theme(style="whitegrid")

# 加载钻石数据集
diamonds = sns.load_dataset("diamonds")

# 绘制基于条件的核密度估计图
sns.displot(
    # 使用 diamonds 数据集
    data=diamonds,
    # 指定 x 轴为 carat（克拉数），hue 为 cut（切工）
    x="carat", hue="cut",
    # 设置图类型为核密度估计（kde），图高度为 6
    kind="kde", height=6,
    # 填充多个密度曲线，裁剪 x 轴范围为 (0, 无穷大)
    multiple="fill", clip=(0, None),
    # 使用指定调色板进行着色
    palette="ch:rot=-.25,hue=1,light=.75",
)
```