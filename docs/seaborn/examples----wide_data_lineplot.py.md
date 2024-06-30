# `D:\src\scipysrc\seaborn\examples\wide_data_lineplot.py`

```
"""
Lineplot from a wide-form dataset
=================================

_thumb: .52, .5

"""
# 导入必要的库
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理
import seaborn as sns  # 导入 Seaborn 库，用于数据可视化
sns.set_theme(style="whitegrid")  # 设置 Seaborn 的主题为白色网格风格

# 创建一个随机数生成器，用于生成确定性随机数序列
rs = np.random.RandomState(365)

# 生成一个随机数矩阵，形状为 (365, 4)，并对每列进行累积和计算
values = rs.randn(365, 4).cumsum(axis=0)

# 创建一个日期索引，从 "1 1 2016" 开始，间隔为 1 天，共 365 个日期
dates = pd.date_range("1 1 2016", periods=365, freq="D")

# 用随机数数据创建一个 Pandas 数据帧（DataFrame），列名分别为 ["A", "B", "C", "D"]
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])

# 对数据帧中的数据进行滚动均值计算，窗口大小为 7 天
data = data.rolling(7).mean()

# 使用 Seaborn 绘制线图，数据来源于 data 数据帧，调色板为 "tab10"，线宽为 2.5
sns.lineplot(data=data, palette="tab10", linewidth=2.5)
```