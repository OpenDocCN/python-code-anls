# `D:\src\scipysrc\seaborn\examples\three_variable_histogram.py`

```
"""
Trivariate histogram with two categorical variables
===================================================

_thumb: .32, .55

"""
# 导入 seaborn 库，用于数据可视化
import seaborn as sns
# 设置 seaborn 的主题样式为暗色系列
sns.set_theme(style="dark")

# 载入名为 "diamonds" 的示例数据集
diamonds = sns.load_dataset("diamonds")
# 绘制分布图，显示价格与颜色关系，根据透明度细分
sns.displot(
    data=diamonds, x="price", y="color", col="clarity",  # 设置 x 轴、y 轴、以及列分组的变量
    log_scale=(True, False),  # 在 x 轴上使用对数尺度，y 轴不使用对数尺度
    col_wrap=4,  # 每行显示的列数限制为 4 列
    height=4, aspect=.7,  # 每个图的高度设为 4，宽高比为 0.7
)
```