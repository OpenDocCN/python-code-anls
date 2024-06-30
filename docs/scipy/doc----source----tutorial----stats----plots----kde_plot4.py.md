# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\kde_plot4.py`

```
# 导入 functools 模块中的 partial 函数，用于创建带有预设参数的可调用对象
from functools import partial

# 导入 numpy 库，并用 np 别名引用
import numpy as np
# 从 scipy 库中导入 stats 模块
from scipy import stats
# 导入 matplotlib.pyplot 模块，并用 plt 别名引用
import matplotlib.pyplot as plt

# 定义一个函数 my_kde_bandwidth，计算 KDE（核密度估计）的带宽
def my_kde_bandwidth(obj, fac=1./5):
    """We use Scott's Rule, multiplied by a constant factor."""
    # 使用 Scott's Rule 计算带宽，乘以一个常数因子 fac
    return np.power(obj.n, -1./(obj.d+4)) * fac

# 设置第一个正态分布的参数 loc1, scale1, size1
loc1, scale1, size1 = (-2, 1, 175)
# 设置第二个正态分布的参数 loc2, scale2, size2
loc2, scale2, size2 = (2, 0.2, 50)

# 生成符合第一组参数的随机数列 x2，并与符合第二组参数的随机数列连接起来
x2 = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1),
                     np.random.normal(loc=loc2, scale=scale2, size=size2)])

# 在 x2 的最小值减去1到最大值加1之间生成500个等间距的点，作为评估点 x_eval
x_eval = np.linspace(x2.min() - 1, x2.max() + 1, 500)

# 使用 Gaussian KDE (stats.gaussian_kde) 对 x2 进行核密度估计，生成不同的核密度估计对象
kde = stats.gaussian_kde(x2)
kde2 = stats.gaussian_kde(x2, bw_method='silverman')
kde3 = stats.gaussian_kde(x2, bw_method=partial(my_kde_bandwidth, fac=0.2))
kde4 = stats.gaussian_kde(x2, bw_method=partial(my_kde_bandwidth, fac=0.5))

# 计算正态分布的概率密度函数（PDF），以及双峰分布的 PDF
pdf = stats.norm.pdf
bimodal_pdf = pdf(x_eval, loc=loc1, scale=scale1) * float(size1) / x2.size + \
              pdf(x_eval, loc=loc2, scale=scale2) * float(size2) / x2.size

# 创建一个图形窗口，并设置尺寸为 8x6 英寸
fig = plt.figure(figsize=(8, 6))
# 添加一个子图到图形窗口，编号为 111
ax = fig.add_subplot(111)

# 在子图上绘制蓝色十字标记，标记 x2 的位置
ax.plot(x2, np.zeros(x2.shape), 'b+', ms=12)
# 绘制 Scott's Rule 方法生成的核密度估计曲线
ax.plot(x_eval, kde(x_eval), 'k-', label="Scott's Rule")
# 绘制 Silverman's Rule 方法生成的核密度估计曲线
ax.plot(x_eval, kde2(x_eval), 'b-', label="Silverman's Rule")
# 绘制使用 my_kde_bandwidth 函数生成的带有自定义带宽的核密度估计曲线
ax.plot(x_eval, kde3(x_eval), 'g-', label="Scott * 0.2")
ax.plot(x_eval, kde4(x_eval), 'c-', label="Scott * 0.5")
# 绘制双峰分布的实际概率密度函数曲线
ax.plot(x_eval, bimodal_pdf, 'r--', label="Actual PDF")

# 设置 x 轴的显示范围为 x_eval 的最小值到最大值
ax.set_xlim([x_eval.min(), x_eval.max()])
# 在图形中添加图例，位置为 2（即左上角）
ax.legend(loc=2)
# 设置 x 轴标签为 'x'
ax.set_xlabel('x')
# 设置 y 轴标签为 'Density'
ax.set_ylabel('Density')

# 显示绘制的图形
plt.show()
```