# `D:\src\scipysrc\seaborn\examples\wide_form_violinplot.py`

```
"""
Violinplot from a wide-form dataset
===================================

_thumb: .6, .45
"""
# 导入 seaborn 和 matplotlib.pyplot 库
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 seaborn 的主题样式为 whitegrid
sns.set_theme(style="whitegrid")

# 加载示例数据集，这里是大脑网络相关性的数据集
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# 从数据集中选择特定的网络子集
used_networks = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# 计算相关性矩阵并按网络进行平均
corr_df = df.corr().groupby(level="network").mean()
corr_df.index = corr_df.index.astype(int)
corr_df = corr_df.sort_index().T

# 设置 matplotlib 图形的大小
f, ax = plt.subplots(figsize=(11, 6))

# 绘制小提琴图，带有比默认更窄的带宽调整
sns.violinplot(data=corr_df, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")

# 完成图形设置
ax.set(ylim=(-.7, 1.05))  # 设置 y 轴的范围
sns.despine(left=True, bottom=True)  # 去除左边和底部的轴线
```