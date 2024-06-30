# `D:\src\scipysrc\seaborn\examples\heat_scatter.py`

```
"""
Scatterplot heatmap
-------------------

_thumb: .5, .5

"""
# 导入 seaborn 库，并设置主题样式为白色网格
import seaborn as sns
sns.set_theme(style="whitegrid")

# 加载脑网络数据集，选择子集，并且将多级索引合并
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# 定义要使用的网络编号列表
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]

# 选择包含特定网络编号的列，并更新数据集
used_columns = (df.columns
                  .get_level_values("network")
                  .astype(int)
                  .isin(used_networks))
df = df.loc[:, used_columns]

# 将列名映射为使用连接符连接的字符串
df.columns = df.columns.map("-".join)

# 计算相关性矩阵，并将其转换为长格式
corr_mat = df.corr().stack().reset_index(name="correlation")

# 绘制散点图热图，每个单元格表示为散点，大小和颜色变化
g = sns.relplot(
    data=corr_mat,
    x="level_0", y="level_1", hue="correlation", size="correlation",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(50, 250), size_norm=(-.2, .8),
)

# 调整图形细节以完成最终布局
g.set(xlabel="", ylabel="", aspect="equal")  # 设置 x 和 y 轴标签为空，保持纵横比
g.despine(left=True, bottom=True)  # 移除左侧和底部的轴线
g.ax.margins(.02)  # 设置图形边界的外边距为 0.02
for label in g.ax.get_xticklabels():
    label.set_rotation(90)  # 将 x 轴刻度标签旋转 90 度以便显示
```