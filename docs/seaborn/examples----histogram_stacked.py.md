# `D:\src\scipysrc\seaborn\examples\histogram_stacked.py`

```
"""
Stacked histogram on a log scale
================================

_thumb: .5, .45

"""
# 导入 seaborn、matplotlib 的必要模块
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置 seaborn 的绘图主题为 ticks
sns.set_theme(style="ticks")

# 从 seaborn 内置数据集中加载 diamonds 数据集
diamonds = sns.load_dataset("diamonds")

# 创建一个大小为 7x5 英寸的绘图对象，并返回图形和轴对象
f, ax = plt.subplots(figsize=(7, 5))

# 去除图形的上和右边框
sns.despine(f)

# 绘制堆叠直方图，横轴为 diamond 数据集中的价格（'price'），根据切割质量（'cut'）分组堆叠
sns.histplot(
    diamonds,
    x="price", hue="cut",           # x 轴使用价格，颜色按切割质量分组
    multiple="stack",               # 堆叠方式为 stack
    palette="light:m_r",            # 使用色板 'light:m_r'
    edgecolor=".3",                 # 直方图边缘颜色
    linewidth=.5,                   # 直方图边缘线宽度
    log_scale=True,                 # 使用对数刻度绘制直方图
)

# 设置 x 轴主刻度格式为标量格式
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

# 设置 x 轴刻度值为指定的值
ax.set_xticks([500, 1000, 2000, 5000, 10000])
```