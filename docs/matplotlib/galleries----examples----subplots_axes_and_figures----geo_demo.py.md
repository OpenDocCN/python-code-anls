# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\geo_demo.py`

```py
"""
======================
Geographic Projections
======================

This shows 4 possible geographic projections.  Cartopy_ supports more
projections.

.. _Cartopy: https://scitools.org.uk/cartopy/
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# %%

# 创建新的图形窗口
plt.figure()
# 在当前图形窗口中创建子图，投影方式为 Aitoff 投影
plt.subplot(projection="aitoff")
# 设置子图标题为 "Aitoff"
plt.title("Aitoff")
# 打开子图网格线显示
plt.grid(True)

# %%

# 创建新的图形窗口
plt.figure()
# 在当前图形窗口中创建子图，投影方式为 Hammer 投影
plt.subplot(projection="hammer")
# 设置子图标题为 "Hammer"
plt.title("Hammer")
# 打开子图网格线显示
plt.grid(True)

# %%

# 创建新的图形窗口
plt.figure()
# 在当前图形窗口中创建子图，投影方式为 Lambert 投影
plt.subplot(projection="lambert")
# 设置子图标题为 "Lambert"
plt.title("Lambert")
# 打开子图网格线显示
plt.grid(True)

# %%

# 创建新的图形窗口
plt.figure()
# 在当前图形窗口中创建子图，投影方式为 Mollweide 投影
plt.subplot(projection="mollweide")
# 设置子图标题为 "Mollweide"
plt.title("Mollweide")
# 打开子图网格线显示
plt.grid(True)

# 显示所有创建的图形窗口
plt.show()
```