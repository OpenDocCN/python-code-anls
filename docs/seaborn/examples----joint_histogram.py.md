# `D:\src\scipysrc\seaborn\examples\joint_histogram.py`

```
"""
Joint and marginal histograms
=============================

_thumb: .52, .505

"""
# 导入 seaborn 库，用于数据可视化
import seaborn as sns
# 设置 seaborn 的主题风格为 "ticks"
sns.set_theme(style="ticks")

# 加载行星数据集，并初始化绘图区域
planets = sns.load_dataset("planets")
# 创建一个联合网格图，设置 x 轴为 "year"，y 轴为 "distance"，同时显示边际直方图的刻度
g = sns.JointGrid(data=planets, x="year", y="distance", marginal_ticks=True)

# 设置 y 轴为对数刻度
g.ax_joint.set(yscale="log")

# 在图形中添加一个颜色条的直方图图例
cax = g.figure.add_axes([.15, .55, .02, .2])

# 绘制联合直方图和边际直方图
g.plot_joint(
    sns.histplot, discrete=(True, False),
    cmap="light:#03012d", pmax=.8, cbar=True, cbar_ax=cax
)
# 绘制边缘直方图，元素类型为 "step"，颜色为 "#03012d"
g.plot_marginals(sns.histplot, element="step", color="#03012d")
```