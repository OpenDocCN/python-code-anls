# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\tests\__init__.py`

```
# 导入所需的模块
import pandas as pd
import numpy as np

# 创建一个数据帧（DataFrame），包含两列：'A' 和 'B'，并且 'A' 列使用范围内的整数，'B' 列为 'A' 列的两倍
df = pd.DataFrame({'A': range(1, 11), 'B': np.arange(2, 21, 2)})

# 将数据帧的前五行打印输出
print(df.head())

# 计算 'A' 列的平均值
mean_A = df['A'].mean()

# 打印 'A' 列的平均值
print(f"Mean of column 'A': {mean_A}")

# 将 'B' 列的值求和
sum_B = df['B'].sum()

# 打印 'B' 列的总和
print(f"Sum of column 'B': {sum_B}")
```