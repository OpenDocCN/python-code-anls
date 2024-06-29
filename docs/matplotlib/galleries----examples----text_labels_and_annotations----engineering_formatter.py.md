# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\engineering_formatter.py`

```
"""
=========================================
Labeling ticks using engineering notation
=========================================

Use of the engineering Formatter.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

from matplotlib.ticker import EngFormatter  # 从 matplotlib 的 ticker 模块中导入 EngFormatter 工具类，用于工程格式化

# 设定随机种子以便结果可重现性
prng = np.random.RandomState(19680801)

# 创建人工数据进行绘图
# x 数据跨越多个数量级，以展示不同的国际单位前缀
xs = np.logspace(1, 9, 100)  # 创建一个对数间隔的数组，包含 100 个点，范围从 10^1 到 10^9
ys = (0.8 + 0.4 * prng.uniform(size=100)) * np.log10(xs)**2  # 根据 xs 计算对应的 ys 值

# 将图的宽度加倍（2*6.4），以便侧边显示两个子图
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(7, 9.6))  # 创建一个包含两个子图的图形窗口

for ax in (ax0, ax1):
    ax.set_xscale('log')  # 设置 x 轴的刻度为对数刻度

# 演示使用默认设置，带有用户定义的单位标签
ax0.set_title('Full unit ticklabels, w/ default precision & space separator')  # 设置子图 ax0 的标题
formatter0 = EngFormatter(unit='Hz')  # 创建一个工程格式化对象，单位为 Hz
ax0.xaxis.set_major_formatter(formatter0)  # 设置 ax0 的 x 轴主刻度格式化器为 formatter0
ax0.plot(xs, ys)  # 在 ax0 上绘制图像
ax0.set_xlabel('Frequency')  # 设置 x 轴标签为 Frequency

# 演示选项 `places`（小数点后的位数）和 `sep`（数值与前缀/单位之间的分隔符）
ax1.set_title('SI-prefix only ticklabels, 1-digit precision & '
              'thin space separator')  # 设置子图 ax1 的标题
formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # 创建一个工程格式化对象，小数点后保留 1 位，使用细空格作为分隔符
ax1.xaxis.set_major_formatter(formatter1)  # 设置 ax1 的 x 轴主刻度格式化器为 formatter1
ax1.plot(xs, ys)  # 在 ax1 上绘制图像
ax1.set_xlabel('Frequency [Hz]')  # 设置 x 轴标签为 Frequency [Hz]

plt.tight_layout()  # 调整子图之间的布局，使其紧凑显示
plt.show()  # 显示绘制的图形
```