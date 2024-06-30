# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\optimize_global_1.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import matplotlib.pyplot as plt  # 导入 matplotlib 库的 pyplot 模块，用于绘图
from scipy import optimize  # 导入 scipy 库的 optimize 模块，用于优化算法

# 定义 Eggholder 函数，计算特定输入 x 的值
def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
            - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

bounds = [(-512, 512), (-512, 512)]  # 设置变量边界

x = np.arange(-512, 513)  # 生成从 -512 到 512 的数组
y = np.arange(-512, 513)  # 生成从 -512 到 512 的数组
xgrid, ygrid = np.meshgrid(x, y)  # 创建网格
xy = np.stack([xgrid, ygrid])  # 将网格堆叠成一个数组

results = dict()  # 创建空字典存储优化结果

# 使用 SHGO 方法进行优化并存储结果
results['shgo'] = optimize.shgo(eggholder, bounds)
# 使用 Dual Annealing 方法进行优化并存储结果
results['DA'] = optimize.dual_annealing(eggholder, bounds)
# 使用 Differential Evolution 方法进行优化并存储结果
results['DE'] = optimize.differential_evolution(eggholder, bounds)
# 使用 SHGO 方法（Sobol采样）进行优化并存储结果
results['shgo_sobol'] = optimize.shgo(eggholder, bounds, n=256, iters=5,
                                      sampling_method='sobol')

fig = plt.figure(figsize=(4.5, 4.5))  # 创建绘图窗口
ax = fig.add_subplot(111)  # 添加子图到绘图窗口
im = ax.imshow(eggholder(xy), interpolation='bilinear', origin='lower',
               cmap='gray')  # 绘制函数的二维图像

ax.set_xlabel('x')  # 设置 x 轴标签
ax.set_ylabel('y')  # 设置 y 轴标签

# 定义 plot_point 函数，绘制优化结果点
def plot_point(res, marker='o', color=None):
    ax.plot(512+res.x[0], 512+res.x[1], marker=marker, color=color, ms=10)

plot_point(results['DE'], color='c')  # 绘制 Differential Evolution 方法的优化结果（青色）
plot_point(results['DA'], color='w')  # 绘制 Dual Annealing 方法的优化结果（白色）

# SHGO 方法产生多个最小值，绘制所有最小值点（使用较小的标记）
plot_point(results['shgo'], color='r', marker='+')
plot_point(results['shgo_sobol'], color='r', marker='x')
for i in range(results['shgo_sobol'].xl.shape[0]):
    ax.plot(512 + results['shgo_sobol'].xl[i, 0],
            512 + results['shgo_sobol'].xl[i, 1],
            'ro', ms=2)

ax.set_xlim([-4, 514*2])  # 设置 x 轴范围
ax.set_ylim([-4, 514*2])  # 设置 y 轴范围

fig.tight_layout()  # 调整图像布局
plt.show()  # 显示绘图窗口
```