# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\normdiscr_plot2.py`

```
# 导入所需的库：numpy（数值计算库）、matplotlib.pyplot（绘图库）、scipy.stats（统计函数库）
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 分布的支持点数减1
npoints = 20
# 支持点数的一半
npointsh = npoints // 2
# 支持点数的浮点数表示
npointsf = float(npoints)
# 截断正态分布的边界
nbound = 4
# 截断正态分布的实际边界
normbound = (1 + 1/npointsf) * nbound
# 创建一个整数网格，范围从-npointsh到npointsh+1
grid = np.arange(-npointsh, npointsh+2, 1)
# 用于截断正态分布的箱体边界
gridlimitsnorm = (grid - 0.5) / npointsh * nbound
# 网格的中心点
gridlimits = grid - 0.5
# 去掉最后一个元素后的网格
grid = grid[:-1]
# 使用差分计算截断正态分布的概率密度函数值
probs = np.diff(stats.truncnorm.cdf(gridlimitsnorm, -normbound, normbound))
# 网格的整数表示
gridint = grid

# 创建一个默认随机数生成器对象
rng = np.random.default_rng()
# 创建一个离散随机变量对象，以gridint为值，probs为概率
normdiscrete = stats.rv_discrete(
                        values=(gridint, np.round(probs, decimals=7)),
                        name='normdiscrete')

# 生成样本的数量
n_sample = 500
# 从离散随机变量中抽取样本
rvs = normdiscrete.rvs(size=n_sample, random_state=rng)
# 计算直方图
f, l = np.histogram(rvs, bins=gridlimits)
# 计算频率、真实概率、样本概率的组合
sfreq = np.vstack([gridint, f, probs*n_sample]).T
# 计算频率的比例
fs = sfreq[:, 1] / float(n_sample)
# 计算真实概率的累积和
ft = sfreq[:, 2].cumsum() / float(n_sample)
# 计算样本概率的累积和
fs = sfreq[:, 1].cumsum() / float(n_sample)
# 计算样本概率的累积和
ft = sfreq[:, 2].cumsum() / float(n_sample)
# 计算离散随机变量的标准差
nd_std = np.sqrt(normdiscrete.stats(moments='v'))

# x轴的位置
ind = gridint
# 条形图的宽度
width = 0.35

# 创建新的图形
plt.figure()
# 创建子图
plt.subplot(111)
# 绘制第一个条形图，显示累积分布函数
rects1 = plt.bar(ind, ft, width, color='b')
# 绘制第二个条形图，显示样本频率
rects2 = plt.bar(ind+width, fs, width, color='r')
# 绘制正态分布的累积分布函数曲线
normline = plt.plot(ind+width/2.0, stats.norm.cdf(ind+0.5, scale=nd_std),
                    color='b')

# 设置y轴标签
plt.ylabel('cdf')
# 设置图表标题
plt.title('Cumulative Frequency and CDF of normdiscrete')
# 设置x轴刻度
plt.xticks(ind+width, ind)
# 添加图例
plt.legend((rects1[0], rects2[0]), ('true', 'sample'))

# 显示图形
plt.show()
```