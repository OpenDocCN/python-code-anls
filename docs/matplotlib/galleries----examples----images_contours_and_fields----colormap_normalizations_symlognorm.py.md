# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\colormap_normalizations_symlognorm.py`

```
# %%
# 合成数据集，包含两个驼峰形状，一个为负值，一个为正值，正值的振幅是负值的8倍。
# 在线性情况下，负值的驼峰几乎不可见，很难看到其轮廓的任何细节。
# 应用对正负值都进行对数缩放后，更容易看到每个驼峰的形状。
#
# 参见 `~.colors.SymLogNorm`。

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors


def rbf(x, y):
    return 1.0 / (1 + 5 * ((x ** 2) + (y ** 2)))

N = 200
gain = 8
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z1 = rbf(X + 0.5, Y + 0.5)
Z2 = rbf(X - 0.5, Y - 0.5)
Z = gain * Z1 - Z2

shadeopts = {'cmap': 'PRGn', 'shading': 'gouraud'}
colormap = 'PRGn'
lnrwidth = 0.5

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

# 在第一个子图上绘制彩色网格图，使用 SymLogNorm 进行归一化
pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=lnrwidth, linscale=1,
                                              vmin=-gain, vmax=gain, base=10),
                       **shadeopts)
# 添加颜色条到第一个子图
fig.colorbar(pcm, ax=ax[0], extend='both')
# 在第一个子图上添加文本 'symlog'
ax[0].text(-2.5, 1.5, 'symlog')

# 在第二个子图上绘制彩色网格图，使用线性缩放进行归一化
pcm = ax[1].pcolormesh(X, Y, Z, vmin=-gain, vmax=gain,
                       **shadeopts)
# 添加颜色条到第二个子图
fig.colorbar(pcm, ax=ax[1], extend='both')
# 在第二个子图上添加文本 'linear'
ax[1].text(-2.5, 1.5, 'linear')


# %%
# 为了找到特定数据集的最佳可视化方式，可能需要尝试多种不同的颜色尺度。
# 除了 `~.colors.SymLogNorm` 缩放之外，还可以选择使用 `~.colors.AsinhNorm`（实验性的），
# 它在应用于数据值 "Z" 的线性和对数区域之间具有更平滑的过渡。
# 在下面的图中，可能会看到每个驼峰周围的轮廓样的伪影，尽管数据集本身没有尖锐的特征。
# “asinh” 缩放显示了每个驼峰的更平滑的阴影。

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

# 在第一个子图上绘制彩色网格图，使用 SymLogNorm 进行归一化
pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=lnrwidth, linscale=1,
                                              vmin=-gain, vmax=gain, base=10),
                       **shadeopts)
# 添加颜色条到第一个子图
fig.colorbar(pcm, ax=ax[0], extend='both')
# 在第一个子图上添加文本 'symlog'
ax[0].text(-2.5, 1.5, 'symlog')

# 在第二个子图上绘制彩色网格图，使用 AsinhNorm 进行归一化
pcm = ax[1].pcolormesh(X, Y, Z,
                       norm=colors.AsinhNorm(linear_width=lnrwidth,
                                             vmin=-gain, vmax=gain),
                       **shadeopts)
# 添加颜色条到第二个子图
fig.colorbar(pcm, ax=ax[1], extend='both')
# 在第二个子图上添加文本 'asinh'


plt.show()
```