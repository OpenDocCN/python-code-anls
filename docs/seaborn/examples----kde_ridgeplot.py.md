# `D:\src\scipysrc\seaborn\examples\kde_ridgeplot.py`

```
"""
Overlapping densities ('ridge plot')
====================================

This script generates a ridge plot using seaborn to visualize overlapped densities of data.

"""
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理
import seaborn as sns  # 导入 Seaborn 库，用于统计数据可视化
import matplotlib.pyplot as plt  # 导入 Matplotlib 库，用于绘图
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})  # 设置 Seaborn 主题及背景色

# Create the data
rs = np.random.RandomState(1979)  # 使用特定种子生成随机数
x = rs.randn(500)  # 生成服从标准正态分布的随机数数组
g = np.tile(list("ABCDEFGHIJ"), 50)  # 重复列表生成数组 g，用于分组
df = pd.DataFrame(dict(x=x, g=g))  # 创建包含数据 x 和分组 g 的 DataFrame
m = df.g.map(ord)  # 将分组 g 映射为整数值
df["x"] += m  # 修改数据 x，使其依赖于分组 g

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)  # 创建调色板 palette
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)  # 创建 FacetGrid 对象

# Draw the densities in a few steps
g.map(sns.kdeplot, "x",  # 绘制核密度估计图
      bw_adjust=.5, clip_on=False,  # 调整带宽参数及绘图选项
      fill=True, alpha=1, linewidth=1.5)  # 填充密度曲线并设置透明度、线宽
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)  # 绘制白色轮廓

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)  # 绘制参考线

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):  # 定义标签函数，用于在图中标注文字
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "x")  # 在图中应用标签函数

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)  # 调整子图重叠参数

# Remove axes details that don't play well with overlap
g.set_titles("")  # 设置子图标题为空字符串
g.set(yticks=[], ylabel="")  # 移除 y 轴刻度和标签
g.despine(bottom=True, left=True)  # 移除底部和左侧的轴线
```