# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\normdiscr_plot1.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import matplotlib.pyplot as plt  # 导入 matplotlib 库的 pyplot 模块，用于绘图
from scipy import stats  # 从 SciPy 库导入 stats 模块，用于统计函数

npoints = 20  # 分布的整数支持点数减去1
npointsh = npoints // 2  # 整数支持点数的一半
npointsf = float(npoints)  # 将整数支持点数转换为浮点数
nbound = 4  # 截断正态分布的边界
normbound = (1 + 1/npointsf) * nbound  # 实际截断正态分布的边界
grid = np.arange(-npointsh, npointsh+2, 1)  # 创建整数网格
gridlimitsnorm = (grid-0.5) / npointsh * nbound  # 截断正态分布的区间限制
gridlimits = grid - 0.5  # 网格的边界
grid = grid[:-1]  # 去除最后一个元素
probs = np.diff(stats.truncnorm.cdf(gridlimitsnorm, -normbound, normbound))  # 计算截断正态分布的概率
gridint = grid  # 网格中的整数值
normdiscrete = stats.rv_discrete(
                        values=(gridint, np.round(probs, decimals=7)),
                        name='normdiscrete')  # 创建离散随机变量对象

n_sample = 500  # 样本数
rng = np.random.default_rng()  # 创建随机数生成器对象
rvs = normdiscrete.rvs(size=n_sample, random_state=rng)  # 生成离散随机变量的样本
f, l = np.histogram(rvs, bins=gridlimits)  # 计算样本的直方图
sfreq = np.vstack([gridint, f, probs*n_sample]).T  # 堆叠整数网格、直方图频率和理论概率
fs = sfreq[:,1] / float(n_sample)  # 计算频率
ft = sfreq[:,2] / float(n_sample)  # 计算理论概率
nd_std = np.sqrt(normdiscrete.stats(moments='v'))  # 计算离散随机变量的标准差

ind = gridint  # 分组的 x 轴位置
width = 0.35       # 条形图的宽度

plt.subplot(111)  # 创建一个 subplot

rects1 = plt.bar(ind, ft, width, color='b')  # 绘制条形图，表示理论概率
rects2 = plt.bar(ind+width, fs, width, color='r')  # 绘制条形图，表示样本频率
normline = plt.plot(ind+width/2.0, stats.norm.pdf(ind, scale=nd_std),
                    color='b')  # 绘制正态分布的概率密度函数曲线

plt.ylabel('Frequency')  # 设置 y 轴标签
plt.title('Frequency and Probability of normdiscrete')  # 设置图表标题
plt.xticks(ind+width, ind)  # 设置 x 轴刻度及标签
plt.legend((rects1[0], rects2[0]), ('true', 'sample'))  # 设置图例

plt.show()  # 显示图表
```