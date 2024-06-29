# `D:\src\scipysrc\matplotlib\galleries\examples\units\artist_tests.py`

```py
"""
============
Artist tests
============

Test unit support with each of the Matplotlib primitive artist types.

The axis handles unit conversions and the artists keep a pointer to their axis
parent. You must initialize the artists with the axis instance if you want to
use them with unit data, or else they will not know how to convert the units
to scalars.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""
# 导入随机模块
import random

# 从basic_units模块中导入cm和inch单位
from basic_units import cm, inch

# 导入matplotlib.pyplot和numpy模块
import matplotlib.pyplot as plt
import numpy as np

# 导入matplotlib的collections、lines、patches和text模块
import matplotlib.collections as collections
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.text as text

# 创建一个新的图形和坐标系对象
fig, ax = plt.subplots()

# 设置X轴和Y轴的单位为厘米
ax.xaxis.set_units(cm)
ax.yaxis.set_units(cm)

# 设置随机种子以保证结果可复现性
np.random.seed(19680801)

if 0:
    # 测试线段集合（目前不支持）
    verts = []
    for i in range(10):
        # 生成一个随机的英寸单位的线段
        verts.append(zip(*inch*10*np.random.rand(2, random.randint(2, 15))))
    lc = collections.LineCollection(verts, axes=ax)
    ax.add_collection(lc)

# 测试简单的线段
line = lines.Line2D([0*cm, 1.5*cm], [0*cm, 2.5*cm],
                    lw=2, color='black', axes=ax)
ax.add_line(line)

if 0:
    # 测试图形块（目前不支持）
    rect = patches.Rectangle((1*cm, 1*cm), width=5*cm, height=2*cm,
                             alpha=0.2, axes=ax)
    ax.add_patch(rect)

# 创建一个文本对象
t = text.Text(3*cm, 2.5*cm, 'text label', ha='left', va='bottom', axes=ax)
ax.add_artist(t)

# 设置X轴和Y轴的显示范围
ax.set_xlim(-1*cm, 10*cm)
ax.set_ylim(-1*cm, 10*cm)

# 打开网格显示
ax.grid(True)

# 设置图形的标题
ax.set_title("Artists with units")

# 显示图形
plt.show()
```