# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\multivariate_marker_plot.py`

```
"""
==============================================
Mapping marker properties to multivariate data
==============================================

This example shows how to use different properties of markers to plot
multivariate datasets. Here we represent a successful baseball throw as a
smiley face with marker size mapped to the skill of thrower, marker rotation to
the take-off angle, and thrust to the marker color.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

from matplotlib.colors import Normalize  # 从 matplotlib 中导入 Normalize 类，用于颜色映射的归一化
from matplotlib.markers import MarkerStyle  # 导入 MarkerStyle 类，用于设置标记样式
from matplotlib.text import TextPath  # 导入 TextPath 类，用于在图中绘制文本路径
from matplotlib.transforms import Affine2D  # 导入 Affine2D 类，用于定义仿射变换

SUCCESS_SYMBOLS = [  # 定义成功标记的符号列表
    TextPath((0, 0), "☹"),  # 第一个符号，哭脸
    TextPath((0, 0), "😒"),  # 第二个符号，苦脸
    TextPath((0, 0), "☺"),  # 第三个符号，笑脸
]

N = 25  # 设置数据点数量
np.random.seed(42)  # 设置随机种子，以便结果可重现
skills = np.random.uniform(5, 80, size=N) * 0.1 + 5  # 生成技能值数组
takeoff_angles = np.random.normal(0, 90, N)  # 生成起飞角度数组
thrusts = np.random.uniform(size=N)  # 生成推力数组
successful = np.random.randint(0, 3, size=N)  # 生成成功标记数组
positions = np.random.normal(size=(N, 2)) * 5  # 生成位置坐标数组
data = zip(skills, takeoff_angles, thrusts, successful, positions)  # 组合数据为一个迭代器

cmap = plt.colormaps["plasma"]  # 使用 plasma 颜色映射
fig, ax = plt.subplots()  # 创建图形和坐标轴
fig.suptitle("Throwing success", size=14)  # 设置主标题

# 遍历数据并绘制图形
for skill, takeoff, thrust, mood, pos in data:
    t = Affine2D().scale(skill).rotate_deg(takeoff)  # 创建仿射变换对象 t，用于设置标记的大小和旋转角度
    m = MarkerStyle(SUCCESS_SYMBOLS[mood], transform=t)  # 创建标记样式对象 m，设置标记的符号和变换
    ax.plot(pos[0], pos[1], marker=m, color=cmap(thrust))  # 在坐标轴上绘制点，设置标记样式和颜色

# 添加颜色条
fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
             ax=ax, label="Normalized Thrust [a.u.]")  # 添加颜色条到图中，设置标签和颜色映射范围
ax.set_xlabel("X position [m]")  # 设置 X 轴标签
ax.set_ylabel("Y position [m]")  # 设置 Y 轴标签

plt.show()  # 显示图形
```