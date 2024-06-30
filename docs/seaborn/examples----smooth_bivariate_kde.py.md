# `D:\src\scipysrc\seaborn\examples\smooth_bivariate_kde.py`

```
"""
Smooth kernel density with marginal histograms
==============================================

_thumb: .48, .41
"""
# 导入 seaborn 库
import seaborn as sns
# 设置 seaborn 主题为白色风格
sns.set_theme(style="white")

# 使用 seaborn 加载内置数据集 'penguins'，存储在 DataFrame 中
df = sns.load_dataset("penguins")

# 创建 JointGrid 对象 g，设置数据来源为 df，x 轴为 'body_mass_g'，y 轴为 'bill_depth_mm'，间隙为 0
g = sns.JointGrid(data=df, x="body_mass_g", y="bill_depth_mm", space=0)

# 绘制二维联合图，使用核密度估计绘制填充图，剪切范围限制在 x 轴 (2200, 6800)，y 轴 (10, 25)，阈值为 0，绘制级别为 100，使用 'rocket' 颜色映射
g.plot_joint(sns.kdeplot,
             fill=True, clip=((2200, 6800), (10, 25)),
             thresh=0, levels=100, cmap="rocket")

# 绘制边缘直方图，使用 histplot，颜色为 '#03051A'，不透明度为 1，分为 25 个箱子
g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)
```