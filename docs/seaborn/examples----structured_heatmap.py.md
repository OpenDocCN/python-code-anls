# `D:\src\scipysrc\seaborn\examples\structured_heatmap.py`

```
"""
Discovering structure in heatmap data
=====================================

_thumb: .3, .25
"""
# 导入 pandas 和 seaborn 库
import pandas as pd
import seaborn as sns

# 设置 seaborn 的主题样式
sns.set_theme()

# 加载示例数据集 brain_networks
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# 选择部分网络进行分析
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
# 筛选出包含指定网络编号的列
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# 创建一个分类调色板，用于标识网络
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# 将调色板转换为向量，以便在矩阵的侧边绘制颜色
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

# 绘制完整的聚类热图
g = sns.clustermap(df.corr(), center=0, cmap="vlag",
                   row_colors=network_colors, col_colors=network_colors,
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))

# 移除行向谱图，因为在此示例中没有行向谱图
g.ax_row_dendrogram.remove()
```