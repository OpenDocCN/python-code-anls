# `D:\src\scipysrc\seaborn\examples\residplot.py`

```
"""
Plotting model residuals
========================

"""
# 导入必要的库：numpy用于数值计算，seaborn用于绘图
import numpy as np
import seaborn as sns
# 设置seaborn的主题样式为白色网格
sns.set_theme(style="whitegrid")

# 使用随机种子生成器创建一个示例数据集，其中 y 与 x 大致符合线性关系
rs = np.random.RandomState(7)
# 生成正态分布的 x 数据，均值为2，标准差为1，共75个样本
x = rs.normal(2, 1, 75)
# 生成 y 数据，满足 y = 2 + 1.5*x + 噪声，其中噪声为正态分布，均值为0，标准差为2
y = 2 + 1.5 * x + rs.normal(0, 2, 75)

# 使用线性模型拟合后，绘制残差图
# residplot 函数用于绘制残差图，x 为自变量，y 为因变量，lowess=True 表示使用局部加权回归平滑拟合残差，color="g" 表示绘图颜色为绿色
sns.residplot(x=x, y=y, lowess=True, color="g")
```