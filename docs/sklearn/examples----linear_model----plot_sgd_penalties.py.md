# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sgd_penalties.py`

```
"""
==============
SGD: Penalties
==============

Contours of where the penalty is equal to 1
for the three penalties L1, L2 and elastic-net.

All of the above are supported by :class:`~sklearn.linear_model.SGDClassifier`
and :class:`~sklearn.linear_model.SGDRegressor`.

"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于科学计算
import numpy as np

# 定义颜色变量
l1_color = "navy"
l2_color = "c"
elastic_net_color = "darkorange"

# 在区间 [-1.5, 1.5] 内生成均匀间隔的1001个点作为 x 和 y 轴坐标
line = np.linspace(-1.5, 1.5, 1001)
# 构建坐标网格
xx, yy = np.meshgrid(line, line)

# 计算 L2 penalty 的等值线数据
l2 = xx**2 + yy**2
# 计算 L1 penalty 的等值线数据
l1 = np.abs(xx) + np.abs(yy)
# 定义 elastic-net penalty 的混合参数 rho
rho = 0.5
# 计算 elastic-net penalty 的等值线数据
elastic_net = rho * l1 + (1 - rho) * l2

# 创建绘图区域
plt.figure(figsize=(10, 10), dpi=100)
# 获取当前的 Axes 对象
ax = plt.gca()

# 绘制 elastic-net penalty 的等值线图
elastic_net_contour = plt.contour(xx, yy, elastic_net, levels=[1], colors=elastic_net_color)
# 绘制 L2 penalty 的等值线图
l2_contour = plt.contour(xx, yy, l2, levels=[1], colors=l2_color)
# 绘制 L1 penalty 的等值线图
l1_contour = plt.contour(xx, yy, l1, levels=[1], colors=l1_color)

# 设置坐标轴纵横比为相等
ax.set_aspect("equal")
# 将左边框移动到坐标原点
ax.spines["left"].set_position("center")
# 右边框设为无色（即隐藏）
ax.spines["right"].set_color("none")
# 将底边框移动到坐标原点
ax.spines["bottom"].set_position("center")
# 顶边框设为无色（即隐藏）
ax.spines["top"].set_color("none")

# 给 elastic-net penalty 的等值线标上标签
plt.clabel(elastic_net_contour, inline=1, fontsize=18, fmt={1.0: "elastic-net"}, manual=[(-1, -1)])
# 给 L2 penalty 的等值线标上标签
plt.clabel(l2_contour, inline=1, fontsize=18, fmt={1.0: "L2"}, manual=[(-1, -1)])
# 给 L1 penalty 的等值线标上标签
plt.clabel(l1_contour, inline=1, fontsize=18, fmt={1.0: "L1"}, manual=[(-1, -1)])

# 调整图形布局，使得各元素之间的间距合适
plt.tight_layout()
# 显示绘制的图形
plt.show()
```