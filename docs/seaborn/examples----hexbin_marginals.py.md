# `D:\src\scipysrc\seaborn\examples\hexbin_marginals.py`

```
"""
Hexbin plot with marginal distributions
=======================================

_thumb: .45, .4
"""
# 导入 NumPy 库，用于生成随机数据
import numpy as np
# 导入 seaborn 库，并设置风格为 "ticks"
import seaborn as sns
sns.set_theme(style="ticks")

# 使用种子为 11 的随机数生成器创建 RandomState 对象
rs = np.random.RandomState(11)
# 生成服从 gamma 分布的随机数 x，共 1000 个
x = rs.gamma(2, size=1000)
# 生成 y 数据，与 x 相关，同时添加正态分布的噪声，共 1000 个
y = -.5 * x + rs.normal(size=1000)

# 创建关联图，x 为横轴，y 为纵轴，使用 hex 形式的二维直方图，颜色为 "#4CB391"
sns.jointplot(x=x, y=y, kind="hex", color="#4CB391")
```