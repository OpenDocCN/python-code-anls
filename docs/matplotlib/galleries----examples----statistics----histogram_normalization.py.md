# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\histogram_normalization.py`

```
"""
.. redirect-from:: /gallery/statistics/histogram_features

===================================
Histogram bins, density, and weight
===================================

The `.Axes.hist` method can flexibly create histograms in a few different ways,
which is flexible and helpful, but can also lead to confusion.  In particular,
you can:

- bin the data as you want, either with an automatically chosen number of
  bins, or with fixed bin edges,
- normalize the histogram so that its integral is one,
- and assign weights to the data points, so that each data point affects the
  count in its bin differently.

The Matplotlib ``hist`` method calls `numpy.histogram` and plots the results,
therefore users should consult the numpy documentation for a definitive guide.

Histograms are created by defining bin edges, and taking a dataset of values
and sorting them into the bins, and counting or summing how much data is in
each bin.  In this simple example, 9 numbers between 1 and 4 are sorted into 3
bins:
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(19680801)

xdata = np.array([1.2, 2.3, 3.3, 3.1, 1.7, 3.4, 2.1, 1.25, 1.3])
xbins = np.array([1, 2, 3, 4])

# changing the style of the histogram bars just to make it
# very clear where the boundaries of the bins are:
style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}

fig, ax = plt.subplots()
# Create a histogram of xdata with specified bin edges (xbins) and styling options (style)
ax.hist(xdata, bins=xbins, **style)

# plot the xdata locations on the x axis:
ax.plot(xdata, 0*xdata, 'd')
ax.set_ylabel('Number per bin')  # Set y-axis label
ax.set_xlabel('x bins (dx=1.0)')  # Set x-axis label

# %%
# Modifying bins
# ==============
#
# Changing the bin size changes the shape of this sparse histogram, so its a
# good idea to choose bins with some care with respect to your data.  Here we
# make the bins half as wide.

xbins = np.arange(1, 4.5, 0.5)

fig, ax = plt.subplots()
# Create a histogram with modified bin sizes, using the updated xbins and style
ax.hist(xdata, bins=xbins, **style)
ax.plot(xdata, 0*xdata, 'd')  # plot data points on x axis
ax.set_ylabel('Number per bin')  # Set y-axis label
ax.set_xlabel('x bins (dx=0.5)')  # Set x-axis label

# %%
# We can also let numpy (via Matplotlib) choose the bins automatically, or
# specify a number of bins to choose automatically:

fig, ax = plt.subplot_mosaic([['auto', 'n4']],
                             sharex=True, sharey=True, layout='constrained')

# Plot histogram with automatic bin selection ('auto')
ax['auto'].hist(xdata, **style)
ax['auto'].plot(xdata, 0*xdata, 'd')
ax['auto'].set_ylabel('Number per bin')  # Set y-axis label
ax['auto'].set_xlabel('x bins (auto)')  # Set x-axis label

# Plot histogram with 4 specified bins ('n4')
ax['n4'].hist(xdata, bins=4, **style)
ax['n4'].plot(xdata, 0*xdata, 'd')
ax['n4'].set_xlabel('x bins ("bins=4")')  # Set x-axis label

# %%
# Normalizing histograms: density and weight
# ==========================================
#
# Counts-per-bin is the default length of each bar in the histogram.  However,
# we can also normalize the bar lengths as a probability density function using
# the ``density`` parameter:

fig, ax = plt.subplots()
# Create a histogram with density normalization (density=True), using xbins and style
ax.hist(xdata, bins=xbins, density=True, **style)
ax.set_ylabel('Probability density [$V^{-1}$])')  # Set y-axis label
ax.set_xlabel('x bins (dx=0.5 $V$)')  # Set x-axis label

# %%
# This normalization can be a little hard to interpret when just exploring the
# data. The value attached to each bar is divided by the total number of data
# points *and* the width of the bin, and thus the values _integrate_ to one
# when integrating across the full range of data.
# e.g. ::
#
#     density = counts / (sum(counts) * np.diff(bins))
#     np.sum(density * np.diff(bins)) == 1
#
# This normalization is how `probability density functions
# <https://en.wikipedia.org/wiki/Probability_density_function>`_ are defined in
# statistics.  If :math:`X` is a random variable on :math:`x`, then :math:`f_X`
# is is the probability density function if :math:`P[a<X<b] = \int_a^b f_X dx`.
# If the units of x are Volts, then the units of :math:`f_X` are :math:`V^{-1}`
# or probability per change in voltage.
#
# The usefulness of this normalization is a little more clear when we draw from
# a known distribution and try to compare with theory.  So, choose 1000 points
# from a `normal distribution
# <https://en.wikipedia.org/wiki/Normal_distribution>`_, and also calculate the
# known probability density function:

xdata = rng.normal(size=1000)  # 从正态分布中生成1000个随机数据点
xpdf = np.arange(-4, 4, 0.1)   # 定义横坐标范围为从-4到4，步长为0.1
pdf = 1 / (np.sqrt(2 * np.pi)) * np.exp(-xpdf**2 / 2)  # 计算标准正态分布的概率密度函数

# %%
# If we don't use ``density=True``, we need to scale the expected probability
# distribution function by both the length of the data and the width of the
# bins:

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained')  # 创建一个子图布局
dx = 0.1
xbins = np.arange(-4, 4, dx)   # 设置直方图的区间范围和宽度
ax['False'].hist(xdata, bins=xbins, density=False, histtype='step', label='Counts')  # 绘制直方图，未进行密度归一化
ax['False'].plot(xpdf, pdf * len(xdata) * dx, label=r'$N\,f_X(x)\,\delta x$')  # 绘制预期的概率密度函数，需手动缩放
ax['False'].set_ylabel('Count per bin')  # 设置y轴标签
ax['False'].set_xlabel('x bins [V]')   # 设置x轴标签
ax['False'].legend()   # 添加图例

ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label='density')  # 绘制直方图，进行密度归一化
ax['True'].plot(xpdf, pdf, label='$f_X(x)$')   # 绘制标准正态分布的概率密度函数
ax['True'].set_ylabel('Probability density [$V^{-1}$]')   # 设置y轴标签
ax['True'].set_xlabel('x bins [$V$]')   # 设置x轴标签
ax['True'].legend()   # 添加图例

# %%
# One advantage of using the density is therefore that the shape and amplitude
# of the histogram does not depend on the size of the bins.  Consider an
# extreme case where the bins do not have the same width.  In this example, the
# bins below ``x=-1.25`` are six times wider than the rest of the bins.   By
# normalizing by density, we preserve the shape of the distribution, whereas if
# we do not, then the wider bins have much higher counts than the thinner bins:

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained')   # 创建一个子图布局
dx = 0.1
xbins = np.hstack([np.arange(-4, -1.25, 6*dx), np.arange(-1.25, 4, dx)])  # 设置不同宽度的区间范围
ax['False'].hist(xdata, bins=xbins, density=False, histtype='step', label='Counts')   # 绘制直方图，未进行密度归一化
ax['False'].plot(xpdf, pdf * len(xdata) * dx, label=r'$N\,f_X(x)\,\delta x_0$')   # 绘制预期的概率密度函数，需手动缩放
ax['False'].set_ylabel('Count per bin')   # 设置y轴标签
ax['False'].set_xlabel('x bins [V]')   # 设置x轴标签
ax['False'].legend()   # 添加图例
ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label='density')
# 在ax['True']上绘制直方图，参数包括数据xdata、箱数xbins、密度为True（归一化）、直方图类型为step、标签为'density'

ax['True'].plot(xpdf, pdf, label='$f_X(x)$')
# 在ax['True']上绘制折线图，参数包括数据xpdf和pdf，标签为'$f_X(x)$'

ax['True'].set_ylabel('Probability density [$V^{-1}$]')
# 设置ax['True']的y轴标签为'Probability density [$V^{-1}$]'

ax['True'].set_xlabel('x bins [$V$]')
# 设置ax['True']的x轴标签为'x bins [$V$]'

ax['True'].legend()
# 在ax['True']上创建图例



fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained')
# 创建一个包含两个子图ax['False']和ax['True']的图形fig，布局为constrained

ax['True'].plot(xpdf, pdf, '--', label='$f_X(x)$', color='k')
# 在ax['True']上绘制带有虚线的折线图，参数包括数据xpdf和pdf，标签为'$f_X(x)$'，颜色为黑色

for nn, dx in enumerate([0.1, 0.4, 1.2]):
    xbins = np.arange(-4, 4, dx)
    # 对于列表[0.1, 0.4, 1.2]中的每个值dx，生成区间为[-4, 4)、步长为dx的箱子列表xbins
    
    ax['False'].plot(xpdf, pdf*1000*dx, '--', color=f'C{nn}')
    # 在ax['False']上绘制带有虚线的折线图，参数包括数据xpdf、pdf乘以1000再乘以dx，颜色使用C{nn}的颜色索引
    
    ax['False'].hist(xdata, bins=xbins, density=False, histtype='step')
    # 在ax['False']上绘制直方图，参数包括数据xdata、箱数xbins、密度为False（不归一化）、直方图类型为step

    ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label=dx)
    # 在ax['True']上绘制直方图，参数包括数据xdata、箱数xbins、密度为True（归一化）、直方图类型为step，标签为dx

ax['False'].set_xlabel('x bins [$V$]')
# 设置ax['False']的x轴标签为'x bins [$V$]'

ax['False'].set_ylabel('Count per bin')
# 设置ax['False']的y轴标签为'Count per bin'

ax['True'].set_ylabel('Probability density [$V^{-1}$]')
# 设置ax['True']的y轴标签为'Probability density [$V^{-1}$]'

ax['True'].set_xlabel('x bins [$V$]')
# 设置ax['True']的x轴标签为'x bins [$V$]'

ax['True'].legend(fontsize='small', title='bin width:')
# 在ax['True']上创建图例，设置字体大小为'small'，标题为'bin width:'



fig, ax = plt.subplots(layout='constrained', figsize=(3.5, 3))
# 创建一个包含单个子图ax的图形fig，布局为constrained，尺寸为3.5x3英寸

for nn, dx in enumerate([0.1, 0.4, 1.2]):
    xbins = np.arange(-4, 4, dx)
    # 对于列表[0.1, 0.4, 1.2]中的每个值dx，生成区间为[-4, 4)、步长为dx的箱子列表xbins
    
    ax.hist(xdata, bins=xbins, weights=1/len(xdata) * np.ones(len(xdata)),
                   histtype='step', label=f'{dx}')
    # 在ax上绘制加权直方图，参数包括数据xdata、箱数xbins、权重为1/len(xdata)、直方图类型为step，标签为f'{dx}'

ax.set_xlabel('x bins [$V$]')
# 设置ax的x轴标签为'x bins [$V$]'

ax.set_ylabel('Bin count / N')
# 设置ax的y轴标签为'Bin count / N'

ax.legend(fontsize='small', title='bin width:')
# 在ax上创建图例，设置字体大小为'small'，标题为'bin width:'



fig, ax = plt.subplot_mosaic([['no_norm', 'density', 'weight']],
                             layout='constrained', figsize=(8, 4))
# 创建一个包含三个子图ax['no_norm']、ax['density']、ax['weight']的图形fig，布局为constrained，尺寸为8x4英寸

ax['no_norm'].hist(xdata, bins=xbins, histtype='step')
# 在ax['no_norm']上绘制直方图，参数包括数据xdata、箱数xbins、直方图类型为step

ax['no_norm'].hist(xdata2, bins=xbins, histtype='step')
# 在ax['no_norm']上绘制直方图，参数包括数据xdata2、箱数xbins、直方图类型为step

ax['no_norm'].set_ylabel('Counts')
# 设置ax['no_norm']的y轴标签为'Counts'

ax['no_norm'].set_xlabel('x bins [$V$]')
# 设置ax['no_norm']的x轴标签为'x bins [$V$]'

ax['no_norm'].set_title('No normalization')
# 设置ax['no_norm']的标题为'No normalization'

ax['density'].hist(xdata, bins=xbins, histtype='step', density=True)
# 在ax['density']上绘制密度归一化的直方图，参数包括数据xdata、箱数xbins、直方图类型为step

ax['density'].hist(xdata2, bins=xbins, histtype='step', density=True)
# 在ax['density']上绘制密度归一化的直方图，参数包括数据xdata2、箱数xbins、直方图类型为step

ax['density'].set_ylabel('Probability density [$V^{-1}$]')
# 设置ax['density']的y轴标签为'Probability density [$V^{-1}$]'

ax['density'].set_title('Density=True')
# 设置ax['density']的标题为'Density=True'

ax['density'].set_xlabel('x bins [$V$]')
# 设置ax['density']的x轴标签为'x bins [$V$]'

ax['weight'].hist(xdata, bins=xbins, histtype='step',
                  weights=1 / len(xdata) * np.ones(len(xdata)),
                  label='N=1000')
# 在ax['weight']上绘制加权直方图，参数包括数据xdata、箱数xbins、权重为1/len(xdata)、直方图类型为step，标签为'N=1000'
# 绘制直方图到指定的轴对象 ax['weight']，使用数据 xdata2，设置直方图的边界数为 xbins，绘制的类型为 step
# 使用权重参数，每个数据点的权重为 1 / len(xdata2)
ax['weight'].hist(xdata2, bins=xbins, histtype='step',
                  weights=1 / len(xdata2) * np.ones(len(xdata2)),
                  label='N=100')

# 设置轴对象 ax['weight'] 的 x 轴标签文本为 'x bins [$V$]'
ax['weight'].set_xlabel('x bins [$V$]')

# 设置轴对象 ax['weight'] 的 y 轴标签文本为 'Counts / N'
ax['weight'].set_ylabel('Counts / N')

# 在轴对象 ax['weight'] 上添加图例，图例标签为 'N=100'，设置图例字体大小为 'small'
ax['weight'].legend(fontsize='small')

# 设置轴对象 ax['weight'] 的标题为 'Weight = 1/N'
ax['weight'].set_title('Weight = 1/N')

# 显示绘制的图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.set_xlabel`
#    - `matplotlib.axes.Axes.set_ylabel`
#    - `matplotlib.axes.Axes.legend`
```