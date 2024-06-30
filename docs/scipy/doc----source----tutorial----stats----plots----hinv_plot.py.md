# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\hinv_plot.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
from scipy.stats.sampling import NumericalInverseHermite  # 导入NumericalInverseHermite采样方法
from scipy.stats import norm  # 导入正态分布
from scipy.special import ndtr  # 导入标准正态分布的累积分布函数
import matplotlib.pyplot as plt  # 导入matplotlib库中的绘图模块


class StandardNormal:
    def pdf(self, x):
        return 1/np.sqrt(2*np.pi) * np.exp(-x**2 / 2)  # 标准正态分布的概率密度函数

    def cdf(self, x):
        return ndtr(x)  # 使用标准正态分布的累积分布函数


dist = StandardNormal()  # 创建StandardNormal类的实例dist，用于描述标准正态分布
urng = np.random.default_rng()  # 使用NumPy中的随机数生成器创建一个新的随机数生成器实例
rng = NumericalInverseHermite(dist, random_state=urng)  # 创建NumericalInverseHermite采样器实例rng，使用dist描述的分布和urng作为随机数生成器
rvs = rng.rvs(10000)  # 从rng中生成10000个随机变量样本
x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, 1000)  # 生成用于绘图的x坐标，范围包括rvs的最小值减去0.1到最大值加上0.1，总共1000个点
fx = norm.pdf(x)  # 计算标准正态分布在x处的概率密度函数值
plt.plot(x, fx, 'r-', lw=2, label='true distribution')  # 绘制标准正态分布的真实概率密度函数曲线
plt.hist(rvs, bins=20, density=True, alpha=0.8, label='random variates')  # 绘制生成的随机变量样本的直方图
plt.xlabel('x')  # 设置x轴标签
plt.ylabel('PDF(x)')  # 设置y轴标签
plt.title('Numerical Inverse Hermite Samples')  # 设置图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图形
```