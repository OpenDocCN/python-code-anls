# `D:\src\scipysrc\matplotlib\galleries\examples\scales\aspect_loglog.py`

```
`
"""
=============
Loglog Aspect
=============

"""
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图

fig, (ax1, ax2) = plt.subplots(1, 2)  # 创建一个包含两个子图的图形窗口，布局为 1 行 2 列
ax1.set_xscale("log")  # 设置第一个子图的 x 轴为对数刻度
ax1.set_yscale("log")  # 设置第一个子图的 y 轴为对数刻度
ax1.set_xlim(1e1, 1e3)  # 设置第一个子图的 x 轴范围从 10 到 1000
ax1.set_ylim(1e2, 1e3)  # 设置第一个子图的 y 轴范围从 100 到 1000
ax1.set_aspect(1)  # 设置第一个子图的纵横比为 1，使得 x 和 y 轴的单位长度相同
ax1.set_title("adjustable = box")  # 为第一个子图设置标题 "adjustable = box"

ax2.set_xscale("log")  # 设置第二个子图的 x 轴为对数刻度
ax2.set_yscale("log")  # 设置第二个子图的 y 轴为对数刻度
ax2.set_adjustable("datalim")  # 设置第二个子图的调整方式为 "datalim"
ax2.plot([1, 3, 10], [1, 9, 100], "o-")  # 在第二个子图中绘制数据点 [1, 3, 10] 和 [1, 9, 100]，使用圆点和线连接
ax2.set_xlim(1e-1, 1e2)  # 设置第二个子图的 x 轴范围从 0.1 到 100
ax2.set_ylim(1e-1, 1e3)  # 设置第二个子图的 y 轴范围从 0.1 到 1000
ax2.set_aspect(1)  # 设置第二个子图的纵横比为 1，使得 x 和 y 轴的单位长度相同
ax2.set_title("adjustable = datalim")  # 为第二个子图设置标题 "adjustable = datalim"

plt.show()  # 显示图形窗口
```