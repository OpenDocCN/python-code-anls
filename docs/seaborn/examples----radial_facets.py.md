# `D:\src\scipysrc\seaborn\examples\radial_facets.py`

```
"""
FacetGrid with custom projection
================================

_thumb: .33, .5

"""
# 导入必要的库
import numpy as np
import pandas as pd
import seaborn as sns

# 设置 seaborn 的主题
sns.set_theme()

# 生成一个示例的径向数据集
r = np.linspace(0, 10, num=100)
df = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})

# 将数据框转换为长格式或“整洁”格式
df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='theta')

# 使用极坐标投影设置一个带子图的坐标轴网格
g = sns.FacetGrid(df, col="speed", hue="speed",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

# 在网格中的每个子图上绘制散点图
g.map(sns.scatterplot, "theta", "r")
```