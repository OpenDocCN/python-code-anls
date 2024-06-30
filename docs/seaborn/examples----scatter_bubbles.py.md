# `D:\src\scipysrc\seaborn\examples\scatter_bubbles.py`

```
"""
Scatterplot with varying point sizes and hues
==============================================

_thumb: .45, .5

"""
# 导入 seaborn 库，用于绘图
import seaborn as sns
# 设置 seaborn 主题为白色风格
sns.set_theme(style="white")

# 加载示例数据集 mpg
# mpg 数据集包含汽车燃油效率等信息
mpg = sns.load_dataset("mpg")

# 绘制散点图，显示马力与每加仑英里数（mpg）的关系，并根据不同的语义进行渲染
# hue 参数表示按照 'origin' 列的值来区分不同的点，每个不同的 'origin' 用不同的颜色表示
# size 参数表示按照 'weight' 列的值来调整点的大小
# sizes=(40, 400) 表示点的大小范围从40到400之间变化
# alpha=.5 表示点的透明度为0.5，即半透明
# palette="muted" 使用 "muted" 调色板，渲染颜色
# height=6 表示图的高度为6英寸
# data=mpg 表示数据来源为 mpg 数据集
sns.relplot(x="horsepower", y="mpg", hue="origin", size="weight",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=mpg)
```