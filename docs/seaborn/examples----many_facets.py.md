# `D:\src\scipysrc\seaborn\examples\many_facets.py`

```
"""
Plotting on a large number of facets
====================================

_thumb: .4, .3

"""
# 导入所需的库
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理
import seaborn as sns  # 导入 Seaborn 库，用于数据可视化
import matplotlib.pyplot as plt  # 导入 Matplotlib 库的 pyplot 模块，用于绘图

# 设置 Seaborn 的绘图主题
sns.set_theme(style="ticks")

# 创建一个包含多个短随机漫步的数据集
rs = np.random.RandomState(4)  # 使用种子为4的随机数生成器
pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)  # 生成随机步长矩阵并累积求和
pos -= pos[:, 0, np.newaxis]  # 每行数据减去起始点的偏移量
step = np.tile(range(5), 20)  # 复制步长序列多次以匹配数据集大小
walk = np.repeat(range(20), 5)  # 重复生成序列以匹配数据集大小
df = pd.DataFrame(np.c_[pos.flat, step, walk],  # 创建 Pandas DataFrame，包含位置、步数和漫步编号
                  columns=["position", "step", "walk"])

# 初始化包含每个漫步的图表网格
grid = sns.FacetGrid(df, col="walk", hue="walk", palette="tab20c",  # 创建基于数据集的图表网格，按漫步编号分列
                     col_wrap=4, height=1.5)

# 绘制水平参考线以显示起始点
grid.refline(y=0, linestyle=":")

# 绘制线图显示每个随机漫步的轨迹
grid.map(plt.plot, "step", "position", marker="o")

# 调整刻度位置和标签
grid.set(xticks=np.arange(5), yticks=[-3, 3],  # 设置 x 和 y 轴刻度的位置和标签
         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))  # 设置 x 和 y 轴的显示范围

# 调整图表布局
grid.fig.tight_layout(w_pad=1)  # 调整图表的紧凑布局，设置水平间距
```