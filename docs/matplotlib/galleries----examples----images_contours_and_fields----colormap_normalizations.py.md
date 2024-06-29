# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\colormap_normalizations.py`

```py
"""
=======================
Colormap normalizations
=======================

Demonstration of using norm to map colormaps onto data in non-linear ways.

.. redirect-from:: /gallery/userdemo/colormap_normalizations
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors

N = 100

# %%
# LogNorm
# -------
# This example data has a low hump with a spike coming out of its center. If plotted
# using a linear colour scale, then only the spike will be visible. To see both hump and
# spike, this requires the z/colour axis on a log scale.
#
# Instead of transforming the data with ``pcolor(log10(Z))``, the color mapping can be
# made logarithmic using a `.LogNorm`.

# 生成一个复杂网格的 X 和 Y 坐标
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
# 计算数据 Z1 和 Z2，Z1 是一个低峰和一个从中心伸出的尖峰，Z2 是一个放大了10倍的类似形状
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
Z = Z1 + 50 * Z2

fig, ax = plt.subplots(2, 1)

# 在第一个子图上绘制彩色图，使用 'PuBu_r' 颜色映射，线条描绘方式，线性缩放
pcm = ax[0].pcolor(X, Y, Z, cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='max', label='linear scaling')

# 在第二个子图上绘制彩色图，使用 'PuBu_r' 颜色映射，线条描绘方式，使用 LogNorm 进行对数缩放
pcm = ax[1].pcolor(X, Y, Z, cmap='PuBu_r', shading='nearest',
                   norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))
fig.colorbar(pcm, ax=ax[1], extend='max', label='LogNorm')

# %%
# PowerNorm
# ---------
# This example data mixes a power-law trend in X with a rectified sine wave in Y. If
# plotted using a linear colour scale, then the power-law trend in X partially obscures
# the sine wave in Y.
#
# The power law can be removed using a `.PowerNorm`.

# 生成另一组复杂网格的 X 和 Y 坐标
X, Y = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
# 计算数据 Z，其中 X 带有幂律趋势，Y 带有修正正弦波
Z = (1 + np.sin(Y * 10)) * X**2

fig, ax = plt.subplots(2, 1)

# 在第一个子图上绘制彩色图，使用 'PuBu_r' 颜色映射，网格填充方式，线性缩放
pcm = ax[0].pcolormesh(X, Y, Z, cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='max', label='linear scaling')

# 在第二个子图上绘制彩色图，使用 'PuBu_r' 颜色映射，网格填充方式，使用 PowerNorm 进行幂律缩放
pcm = ax[1].pcolormesh(X, Y, Z, cmap='PuBu_r', shading='nearest',
                       norm=colors.PowerNorm(gamma=0.5))
fig.colorbar(pcm, ax=ax[1], extend='max', label='PowerNorm')

# %%
# SymLogNorm
# ----------
# This example data has two humps, one negative and one positive, The positive hump has
# 5 times the amplitude of the negative. If plotted with a linear colour scale, then
# the detail in the negative hump is obscured.
#
# Here we logarithmically scale the positive and negative data separately with
# `.SymLogNorm`.
#
# Note that colorbar labels do not come out looking very good.

# 生成另一组复杂网格的 X 和 Y 坐标
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
# 计算数据 Z，包含一个正和一个负的两个波峰，其中正的波峰振幅是负的五倍
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (5 * Z1 - Z2) * 2

fig, ax = plt.subplots(2, 1)

# 在第一个子图上绘制彩色图，使用 'RdBu_r' 颜色映射，网格填充方式，线性缩放，设置最小值为 Z 的最大值的相反数
pcm = ax[0].pcolormesh(X, Y, Z, cmap='RdBu_r', shading='nearest',
                       vmin=-np.max(Z))
fig.colorbar(pcm, ax=ax[0], extend='both', label='linear scaling')

# 在第二个子图上绘制彩色图，使用 'RdBu_r' 颜色映射，网格填充方式，使用 SymLogNorm 进行对称对数缩放
pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', shading='nearest',
                       norm=colors.SymLogNorm(linthresh=0.015,
                                              vmin=-10.0, vmax=10.0, base=10))
fig.colorbar(pcm, ax=ax[1], extend='both', label='SymLogNorm')

# %%
# Custom Norm
# -----------
# Placeholder for demonstrating a custom normalization if needed.
# %%
# 创建一个包含两个子图的图形窗口
fig, ax = plt.subplots(2, 1)

# 在第一个子图上绘制伪彩图（pcolormesh），使用'RdBu_r'颜色映射，最小值为-Z的最大值
pcm = ax[0].pcolormesh(X, Y, Z, cmap='RdBu_r', shading='nearest',
                       vmin=-np.max(Z))
# 添加颜色条到第一个子图，扩展颜色条至两端，标签为'linear scaling'
fig.colorbar(pcm, ax=ax[0], extend='both', label='linear scaling')

# 在第二个子图上绘制伪彩图（pcolormesh），使用'RdBu_r'颜色映射，使用自定义的规范化（MidpointNormalize类）
pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', shading='nearest',
                       norm=MidpointNormalize(midpoint=0))
# 添加颜色条到第二个子图，扩展颜色条至两端，标签为'Custom norm'
fig.colorbar(pcm, ax=ax[1], extend='both', label='Custom norm')

# %%
# BoundaryNorm
# ------------
# 用于将颜色刻度任意分割的BoundaryNorm，通过提供颜色的边界值，这个规范化方法将第一个颜色放在第一对边界之间，
# 第二个颜色放在第二对边界之间，依此类推。

# 创建一个包含三个子图的图形窗口，使用约束布局
fig, ax = plt.subplots(3, 1, layout='constrained')

# 在第一个子图上绘制伪彩图（pcolormesh），使用'RdBu_r'颜色映射，最小值为-Z的最大值
pcm = ax[0].pcolormesh(X, Y, Z, cmap='RdBu_r', shading='nearest',
                       vmin=-np.max(Z))
# 添加颜色条到第一个子图，扩展颜色条至两端，垂直方向，标签为'linear scaling'
fig.colorbar(pcm, ax=ax[0], extend='both', orientation='vertical',
             label='linear scaling')

# 创建均匀间隔的边界，产生类似等高线的效果
bounds = np.linspace(-2, 2, 11)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
# 在第二个子图上绘制伪彩图（pcolormesh），使用'RdBu_r'颜色映射和自定义的BoundaryNorm规范化
pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', shading='nearest',
                       norm=norm)
# 添加颜色条到第二个子图，扩展颜色条至两端，垂直方向，标签包含'BoundaryNorm'和边界定义[-2, 2, 11]
fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical',
             label='BoundaryNorm\nlinspace(-2, 2, 11)')

# 创建非均匀间隔的边界，改变颜色映射
bounds = np.array([-1, -0.5, 0, 2.5, 5])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
# 在第三个子图上绘制伪彩图（pcolormesh），使用'RdBu_r'颜色映射和自定义的BoundaryNorm规范化
pcm = ax[2].pcolormesh(X, Y, Z, cmap='RdBu_r', shading='nearest',
                       norm=norm)
# 添加颜色条到第三个子图，扩展颜色条至两端，垂直方向，标签包含'BoundaryNorm'和边界定义[-1, -0.5, 0, 2.5, 5]
fig.colorbar(pcm, ax=ax[2], extend='both', orientation='vertical',
             label='BoundaryNorm\n[-1, -0.5, 0, 2.5, 5]')

# 显示图形
plt.show()
```