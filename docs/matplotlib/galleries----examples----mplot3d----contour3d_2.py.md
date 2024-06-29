# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\contour3d_2.py`

```
"""
===========================================================
Plot contour (level) curves in 3D using the extend3d option
===========================================================

This modification of the :doc:`contour3d` example uses ``extend3d=True`` to
extend the curves vertically into 'ribbons'.
"""

# 导入 matplotlib 的 pyplot 模块作为 plt
import matplotlib.pyplot as plt

# 从 matplotlib 中导入颜色映射和三维坐标轴工具包
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

# 创建一个带有三维投影的图形子图对象
ax = plt.figure().add_subplot(projection='3d')

# 获取测试数据 X, Y, Z，数据间隔为 0.05
X, Y, Z = axes3d.get_test_data(0.05)

# 在三维坐标轴上绘制等高线曲面，使用 extend3d=True 扩展曲面成为“带状”
ax.contour(X, Y, Z, extend3d=True, cmap=cm.coolwarm)

# 显示图形
plt.show()
```