# `D:\src\scipysrc\seaborn\examples\multiple_ecdf.py`

```
"""
Facetted ECDF plots
===================

_thumb: .30, .49
"""
# 导入 seaborn 库，用于数据可视化
import seaborn as sns
# 设置 seaborn 的绘图主题为 "ticks"
sns.set_theme(style="ticks")
# 加载示例数据集 "mpg"
mpg = sns.load_dataset("mpg")

# 定义两种颜色元组，用于创建颜色映射
colors = (250, 70, 50), (350, 70, 50)
# 使用 husl 颜色空间创建混合调色板，作为颜色映射
cmap = sns.blend_palette(colors, input="husl", as_cmap=True)
# 绘制基于数据集 mpg 的 ECDF 图，根据 origin 列分列显示，使用 model_year 列作为颜色分组
sns.displot(
    mpg,
    x="displacement", col="origin", hue="model_year",
    kind="ecdf", aspect=.75, linewidth=2, palette=cmap,
)
```