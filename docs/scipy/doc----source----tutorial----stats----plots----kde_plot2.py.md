# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\kde_plot2.py`

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x1 = np.array([-7, -5, 1, 4, 5], dtype=float)  # 创建一个包含浮点数的 NumPy 数组 x1
x_eval = np.linspace(-10, 10, num=200)  # 在 -10 到 10 之间生成 200 个均匀间隔的值，存储在 x_eval 中
kde1 = stats.gaussian_kde(x1)  # 使用默认带宽生成 x1 数组的高斯核密度估计对象
kde2 = stats.gaussian_kde(x1, bw_method='silverman')  # 使用 Silverman 方法生成 x1 数组的高斯核密度估计对象

def my_kde_bandwidth(obj, fac=1./5):
    """We use Scott's Rule, multiplied by a constant factor."""
    # 自定义函数，基于 Scott's Rule 计算 KDE 带宽，乘以 fac 参数作为缩放因子
    return np.power(obj.n, -1./(obj.d+4)) * fac

fig = plt.figure()  # 创建一个新的图形对象
ax = fig.add_subplot(111)  # 在图形上添加一个子图，1 行 1 列的第一个子图

ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)  # 在子图上绘制 x1 数组的 rug plot，表示数据点的位置
kde3 = stats.gaussian_kde(x1, bw_method=my_kde_bandwidth)  # 使用自定义带宽函数生成 x1 数组的高斯核密度估计对象
ax.plot(x_eval, kde3(x_eval), 'g-', label="With smaller BW")  # 在子图上绘制使用自定义带宽的 KDE 曲线，并标注为 "With smaller BW"

plt.show()  # 显示绘制的图形
```