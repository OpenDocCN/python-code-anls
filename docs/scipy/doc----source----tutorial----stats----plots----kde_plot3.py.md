# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\kde_plot3.py`

```
# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 创建一个随机数生成器对象
rng = np.random.default_rng()
# 生成服从正态分布的随机数据
x1 = rng.normal(size=200)  # random data, normal distribution
# 在数据范围内生成一系列均匀分布的点，用于绘制平滑曲线
xs = np.linspace(x1.min()-1, x1.max()+1, 200)

# 使用 Scott's Rule 计算正态分布的核密度估计
kde1 = stats.gaussian_kde(x1)
# 使用 Silverman's Rule 计算正态分布的核密度估计
kde2 = stats.gaussian_kde(x1, bw_method='silverman')

# 创建一个图形窗口对象，设置尺寸为 8x6 英寸
fig = plt.figure(figsize=(8, 6))

# 添加第一个子图，位于整个图形窗口的上半部分
ax1 = fig.add_subplot(211)
# 绘制 rug plot，显示数据点的分布
ax1.plot(x1, np.zeros(x1.shape), 'b+', ms=12)  # rug plot
# 绘制核密度估计曲线，使用 Scott's Rule
ax1.plot(xs, kde1(xs), 'k-', label="Scott's Rule")
# 绘制核密度估计曲线，使用 Silverman's Rule
ax1.plot(xs, kde2(xs), 'b-', label="Silverman's Rule")
# 绘制真实概率密度函数的曲线，这里使用正态分布的概率密度函数
ax1.plot(xs, stats.norm.pdf(xs), 'r--', label="True PDF")

# 设置第一个子图的 x 轴标签
ax1.set_xlabel('x')
# 设置第一个子图的 y 轴标签
ax1.set_ylabel('Density')
# 设置第一个子图的标题
ax1.set_title("Normal (top) and Student's T$_{df=5}$ (bottom) distributions")
# 设置第一个子图的图例位置
ax1.legend(loc=1)

# 生成服从自由度为 5 的 t 分布的随机数据
x2 = stats.t.rvs(5, size=200, random_state=rng)  # random data, T distribution
# 在数据范围内生成一系列均匀分布的点，用于绘制平滑曲线
xs = np.linspace(x2.min() - 1, x2.max() + 1, 200)

# 使用 Scott's Rule 计算 t 分布的核密度估计
kde3 = stats.gaussian_kde(x2)
# 使用 Silverman's Rule 计算 t 分布的核密度估计
kde4 = stats.gaussian_kde(x2, bw_method='silverman')

# 添加第二个子图，位于整个图形窗口的下半部分
ax2 = fig.add_subplot(212)
# 绘制 rug plot，显示数据点的分布
ax2.plot(x2, np.zeros(x2.shape), 'b+', ms=12)  # rug plot
# 绘制核密度估计曲线，使用 Scott's Rule
ax2.plot(xs, kde3(xs), 'k-', label="Scott's Rule")
# 绘制核密度估计曲线，使用 Silverman's Rule
ax2.plot(xs, kde4(xs), 'b-', label="Silverman's Rule")
# 绘制真实概率密度函数的曲线，这里使用 t 分布的概率密度函数
ax2.plot(xs, stats.t.pdf(xs, 5), 'r--', label="True PDF")

# 设置第二个子图的 x 轴标签
ax2.set_xlabel('x')
# 设置第二个子图的 y 轴标签
ax2.set_ylabel('Density')

# 展示图形窗口
plt.show()
```