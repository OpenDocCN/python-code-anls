# `D:\src\scipysrc\matplotlib\galleries\users_explain\colors\colormapnorms.py`

```py
# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import numpy as np  # 导入 numpy 库，并重命名为 np

# 导入 matplotlib 的色彩映射和一些辅助函数
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors

# 定义矩阵的大小 N
N = 100
# 创建一个二维网格，范围是 [-3, 3] 和 [-2, 2]，包含 N 个复杂数点
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]

# 创建两个数据集 Z1 和 Z2，分别表示两个高度不同的“山丘”，Z 是它们的和
Z1 = np.exp(-X**2 - Y**2)  # 第一个“山丘”
Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)  # 第二个“山丘”，高度更高
Z = Z1 + 50 * Z2  # 总数据集 Z，结合了 Z1 和 Z2

# 创建一个包含两个子图的图形对象和子图数组
fig, ax = plt.subplots(2, 1)

# 在第一个子图中绘制颜色图，使用对数标准化（LogNorm），将 Z 数据映射到 PuBu_r 色彩映射上
pcm = ax[0].pcolor(X, Y, Z,
                   norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),  # 对数标准化器，设置最小值和最大值
                   cmap='PuBu_r', shading='auto')  # 使用 PuBu_r 色彩映射，自动着色
fig.colorbar(pcm, ax=ax[0], extend='max')  # 在第一个子图中添加颜色条

# 在第二个子图中绘制颜色图，使用线性映射和 PuBu_r 色彩映射
pcm = ax[1].pcolor(X, Y, Z, cmap='PuBu_r', shading='auto')  # 线性映射，PuBu_r 色彩映射
fig.colorbar(pcm, ax=ax[1], extend='max')  # 在第二个子图中添加颜色条

# 显示图形
plt.show()
# 定义步长
delta = 0.1
# 创建一维数组 x 和 y，覆盖指定范围，步长为 delta
x = np.arange(-3.0, 4.001, delta)
y = np.arange(-4.0, 3.001, delta)
# 创建网格 X 和 Y，用于生成二维坐标系
X, Y = np.meshgrid(x, y)
# 计算 Z1 和 Z2 的值
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# 计算合成的 Z 值
Z = (0.9*Z1 - 0.5*Z2) * 2

# 选择一个发散型的颜色映射
cmap = cm.coolwarm

# 创建包含两个子图的图形对象
fig, (ax1, ax2) = plt.subplots(ncols=2)
# 在第一个子图上绘制彩色网格图
pc = ax1.pcolormesh(Z, cmap=cmap)
fig.colorbar(pc, ax=ax1)
ax1.set_title('Normalize()')

# 在第二个子图上绘制彩色网格图，使用 CenteredNorm 进行归一化
pc = ax2.pcolormesh(Z, norm=colors.CenteredNorm(), cmap=cmap)
fig.colorbar(pc, ax=ax2)
ax2.set_title('CenteredNorm()')

plt.show()

# %%
# 对数对称尺度
# ---------------------
#
# 有时数据既有正值又有负值，但我们仍希望在两者上应用对数尺度。
# 在这种情况下，负数也按对数尺度进行缩放，并映射到较小的数值；
# 例如，如果 ``vmin=-vmax``，那么负数将从0映射到0.5，正数从0.5映射到1。
#
# 由于接近零的值的对数趋于无穷大，需要线性地映射零附近的一小范围。
# 参数 *linthresh* 允许用户指定这个范围的大小（-linthresh，linthresh）。
# 在颜色映射中，*linscale* 设置了这个范围的大小。
# 当 *linscale* == 1.0（默认值）时，用于正负线性范围的空间将等于对数范围内的一个数量级。

# 创建网格 X 和 Y
N = 100
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
# 计算 Z1 和 Z2 的值
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# 计算合成的 Z 值
Z = (Z1 - Z2) * 2

# 创建包含两个子图的图形对象
fig, ax = plt.subplots(2, 1)

# 在第一个子图上绘制彩色网格图，使用 SymLogNorm 进行归一化
pcm = ax[0].pcolormesh(X, Y, Z,
                       norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=-1.0, vmax=1.0, base=10),
                       cmap='RdBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[0], extend='both')

# 在第二个子图上绘制彩色网格图，简单地使用线性范围进行归一化
pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z), shading='auto')
fig.colorbar(pcm, ax=ax[1], extend='both')
plt.show()

# %%
# 幂律
# ---------
#
# 有时将颜色重新映射到幂律关系上很有用（即 :math:`y=x^{\gamma}`，其中 :math:`\gamma` 是
# 幂指数）。为此我们使用 `.colors.PowerNorm`。它接受参数 *gamma*（*gamma* == 1.0 将产生默认的线性归一化）：
#
# .. note::
#
#    应该有充分的理由使用这种类型的转换来绘制数据。技术观众习惯于线性和对数坐标轴及数据变换。
#    幂律较少见，观众应明确知道已经使用了这种方法。

# 创建网格 X 和 Y
N = 100
X, Y = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
Z1 = (1 + np.sin(Y * 10.)) * X**2
# 计算 Z1，根据 Y 值的正弦值和 X 的平方计算得出

fig, ax = plt.subplots(2, 1, layout='constrained')
# 创建一个包含两个子图的图形对象 fig 和对应的轴对象 ax，子图布局为 constrained

pcm = ax[0].pcolormesh(X, Y, Z1, norm=colors.PowerNorm(gamma=0.5),
                       cmap='PuBu_r', shading='auto')
# 在第一个子图 ax[0] 上绘制彩色网格，使用 PowerNorm 规范化器，颜色映射为 'PuBu_r'，自动着色
fig.colorbar(pcm, ax=ax[0], extend='max')
# 在 ax[0] 上添加颜色条，颜色范围扩展到最大值
ax[0].set_title('PowerNorm()')
# 设置第一个子图的标题为 'PowerNorm()'

pcm = ax[1].pcolormesh(X, Y, Z1, cmap='PuBu_r', shading='auto')
# 在第二个子图 ax[1] 上绘制彩色网格，使用默认的彩色映射 'PuBu_r'，自动着色
fig.colorbar(pcm, ax=ax[1], extend='max')
# 在 ax[1] 上添加颜色条，颜色范围扩展到最大值
ax[1].set_title('Normalize()')
# 设置第二个子图的标题为 'Normalize()'
plt.show()
# 显示图形

# %%
# Discrete bounds
# ---------------
#
# Another normalization that comes with Matplotlib is `.colors.BoundaryNorm`.
# In addition to *vmin* and *vmax*, this takes as arguments boundaries between
# which data is to be mapped.  The colors are then linearly distributed between
# these "bounds".  It can also take an *extend* argument to add upper and/or
# lower out-of-bounds values to the range over which the colors are
# distributed. For instance:
#
# .. code-block:: pycon
#
#   >>> import matplotlib.colors as colors
#   >>> bounds = np.array([-0.25, -0.125, 0, 0.5, 1])
#   >>> norm = colors.BoundaryNorm(boundaries=bounds, ncolors=4)
#   >>> print(norm([-0.2, -0.15, -0.02, 0.3, 0.8, 0.99]))
#   [0 0 1 2 3 3]
#
# Note: Unlike the other norms, this norm returns values from 0 to *ncolors*-1.

N = 100
X, Y = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-2, 2, N))
# 创建一个 N x N 的网格，X 和 Y 分别表示 X 和 Y 轴的坐标点

Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = ((Z1 - Z2) * 2)[:-1, :-1]
# 计算 Z1 和 Z2 的指数函数值，并计算它们的差值乘以 2，然后截取除了最后一行和最后一列的所有元素作为 Z

fig, ax = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
# 创建一个包含四个子图的图形对象 fig 和对应的轴对象 ax，子图布局为 constrained，并指定图形大小为 8 x 6

ax = ax.flatten()
# 将二维数组的轴对象展平为一维数组

# Default norm:
pcm = ax[0].pcolormesh(X, Y, Z, cmap='RdBu_r')
# 在第一个子图 ax[0] 上绘制彩色网格，使用默认的彩色映射 'RdBu_r'
fig.colorbar(pcm, ax=ax[0], orientation='vertical')
# 在 ax[0] 上添加垂直方向的颜色条
ax[0].set_title('Default norm')
# 设置第一个子图的标题为 'Default norm'

# Even bounds give a contour-like effect:
bounds = np.linspace(-1.5, 1.5, 7)
# 生成一个包含 7 个均匀分布值的数组，作为彩色映射的边界
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
# 创建一个 BoundaryNorm 规范化器，使用指定的边界和颜色数目
pcm = ax[1].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
# 在第二个子图 ax[1] 上绘制彩色网格，使用 BoundaryNorm 规范化器和彩色映射 'RdBu_r'
fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical')
# 在 ax[1] 上添加垂直方向的颜色条，扩展颜色条到两侧
ax[1].set_title('BoundaryNorm: 7 boundaries')
# 设置第二个子图的标题为 'BoundaryNorm: 7 boundaries'

# Bounds may be unevenly spaced:
bounds = np.array([-0.2, -0.1, 0, 0.5, 1])
# 生成一个包含 5 个不均匀分布值的数组，作为彩色映射的边界
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
# 创建一个 BoundaryNorm 规范化器，使用指定的边界和颜色数目
pcm = ax[2].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
# 在第三个子图 ax[2] 上绘制彩色网格，使用 BoundaryNorm 规范化器和彩色映射 'RdBu_r'
fig.colorbar(pcm, ax=ax[2], extend='both', orientation='vertical')
# 在 ax[2] 上添加垂直方向的颜色条，扩展颜色条到两侧
ax[2].set_title('BoundaryNorm: nonuniform')
# 设置第三个子图的标题为 'BoundaryNorm: nonuniform'

# With out-of-bounds colors:
bounds = np.linspace(-1.5, 1.5, 7)
# 生成一个包含 7 个均匀分布值的数组，作为彩色映射的边界
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')
# 创建一个 BoundaryNorm 规范化器，使用指定的边界、颜色数目和扩展参数
pcm = ax[3].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
# 在第四个子图 ax[3] 上绘制彩色网格，使用 BoundaryNorm 规范化器和彩色映射 'RdBu_r'
# 颜色条继承 BoundaryNorm 的 "extend" 参数
fig.colorbar(pcm, ax=ax[3], orientation='vertical')
# 在 ax[3] 上添加垂直方向的颜色条
ax[3].set_title('BoundaryNorm: extend="both"')
# 设置第四个子图的标题为 'BoundaryNorm: extend="both"'
plt.show()
# 显示图形

# %%
# TwoSlopeNorm: Different mapping on either side of a center
# ----------------------------------------------------------
#
# Sometimes we want to have a different colormap on either side of a
# conceptual center point, and we want those two colormaps to have
# different linear scales.  An example is a topographic map where the land
# Load sample topography and bathymetry data from a file
dem = cbook.get_sample_data('topobathy.npz')
topo = dem['topo']  # Extract topography data
longitude = dem['longitude']  # Extract longitude data
latitude = dem['latitude']  # Extract latitude data

# Create a new figure and axis for plotting
fig, ax = plt.subplots()

# Generate a custom colormap for terrain with distinct colors for land and ocean
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = colors.LinearSegmentedColormap.from_list(
    'terrain_map', all_colors)

# Define a normalization instance that centers on land to emphasize dynamic range
divnorm = colors.TwoSlopeNorm(vmin=-500., vcenter=0, vmax=4000)

# Create a pseudo-color plot mesh (pcolormesh) on the axis using the data
pcm = ax.pcolormesh(longitude, latitude, topo, rasterized=True, norm=divnorm,
                    cmap=terrain_map, shading='auto')

# Adjust the aspect ratio to compensate for latitude-dependent longitude spacing
ax.set_aspect(1 / np.cos(np.deg2rad(49)))
ax.set_title('TwoSlopeNorm(x)')  # Set the plot title

# Add a colorbar for the pseudo-color plot with specified tick marks
cb = fig.colorbar(pcm, shrink=0.6)
cb.set_ticks([-500, 0, 1000, 2000, 3000, 4000])
plt.show()



# Define a custom normalization function using `FuncNorm` for arbitrary transformation
def _forward(x):
    return np.sqrt(x)

def _inverse(x):
    return x**2

N = 100
X, Y = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
Z1 = (1 + np.sin(Y * 10.)) * X**2
fig, ax = plt.subplots()

# Create a `FuncNorm` instance using the defined forward and inverse functions
norm = colors.FuncNorm((_forward, _inverse), vmin=0, vmax=20)

# Create a pseudo-color plot mesh using the custom normalization
pcm = ax.pcolormesh(X, Y, Z1, norm=norm, cmap='PuBu_r', shading='auto')
ax.set_title('FuncNorm(x)')  # Set the plot title

# Add a colorbar for the pseudo-color plot with specified tick marks
fig.colorbar(pcm, shrink=0.6)
plt.show()



# Define a custom normalization class `MidpointNormalize` derived from `Normalize`
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Simple interpolation function ignoring edge cases and masked values
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        # Inverse interpolation function for the custom normalization
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)
# 创建一个新的图形窗口和轴对象
fig, ax = plt.subplots()

# 创建一个自定义的归一化器对象，用于对颜色映射中的值进行中心化处理
midnorm = MidpointNormalize(vmin=-500., vcenter=0, vmax=4000)

# 在轴上绘制一个伪彩色网格图，用于显示地形数据，使用自定义的归一化器和地形地图的颜色映射
pcm = ax.pcolormesh(longitude, latitude, topo, rasterized=True, norm=midnorm,
                    cmap=terrain_map, shading='auto')

# 设置轴的纵横比例，使其在投影时保持正确的地理形状
ax.set_aspect(1 / np.cos(np.deg2rad(49)))

# 设置图形的标题
ax.set_title('Custom norm')

# 在图形上添加一个颜色条，关联到之前绘制的伪彩色网格图对象，设置颜色条的缩放比例和延伸方式
cb = fig.colorbar(pcm, shrink=0.6, extend='both')

# 设置颜色条上的刻度位置
cb.set_ticks([-500, 0, 1000, 2000, 3000, 4000])

# 显示图形
plt.show()
```