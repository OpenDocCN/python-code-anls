# `D:\src\scipysrc\matplotlib\galleries\examples\scales\scales.py`

```
"""
======
Scales
======

Illustrate the scale transformations applied to axes, e.g. log, symlog, logit.

The last two examples are examples of using the ``'function'`` scale by
supplying forward and inverse functions for the scale transformation.
"""

# 导入matplotlib.pyplot库，并简写为plt
import matplotlib.pyplot as plt
# 导入numpy库，并简写为np
import numpy as np

# 从matplotlib.ticker模块中导入FixedLocator和NullFormatter类
from matplotlib.ticker import FixedLocator, NullFormatter

# 设定随机种子以保证结果可复现性
np.random.seed(19680801)

# 生成一些位于(0, 1)区间内的随机数据
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
# 筛选出位于(0, 1)区间内的数据并排序
y = y[(y > 0) & (y < 1)]
y.sort()
# 生成与y长度相同的序列作为x轴数据
x = np.arange(len(y))

# 创建包含6个子图的画布，大小为6x8，并使用constrained布局
fig, axs = plt.subplots(3, 2, figsize=(6, 8), layout='constrained')

# 第一个子图：线性轴
ax = axs[0, 0]
ax.plot(x, y)
ax.set_yscale('linear')
ax.set_title('linear')
ax.grid(True)

# 第二个子图：对数轴
ax = axs[0, 1]
ax.plot(x, y)
ax.set_yscale('log')
ax.set_title('log')
ax.grid(True)

# 第三个子图：对称对数轴
ax = axs[1, 1]
ax.plot(x, y - y.mean())
ax.set_yscale('symlog', linthresh=0.02)
ax.set_title('symlog')
ax.grid(True)

# 第四个子图：logit轴
ax = axs[1, 0]
ax.plot(x, y)
ax.set_yscale('logit')
ax.set_title('logit')
ax.grid(True)

# 第五个子图：自定义函数轴，使用x的平方根作为函数
def forward(x):
    return x**(1/2)

def inverse(x):
    return x**2

ax = axs[2, 0]
ax.plot(x, y)
ax.set_yscale('function', functions=(forward, inverse))
ax.set_title('function: $x^{1/2}$')
ax.grid(True)
ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 1, 0.2)**2))

# 第六个子图：Mercator变换函数轴
def forward(a):
    a = np.deg2rad(a)
    return np.rad2deg(np.log(np.abs(np.tan(a) + 1.0 / np.cos(a))))

def inverse(a):
    a = np.deg2rad(a)
    return np.rad2deg(np.arctan(np.sinh(a)))

ax = axs[2, 1]
t = np.arange(0, 170.0, 0.1)
s = t / 2.
ax.plot(t, s, '-', lw=2)
ax.set_yscale('function', functions=(forward, inverse))
ax.set_title('function: Mercator')
ax.grid(True)
ax.set_xlim([0, 180])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 90, 10)))

# 显示图形
plt.show()
```