# `D:\src\scipysrc\scipy\benchmarks\benchmarks\__init__.py`

```
# 导入 NumPy 库并重置其随机种子为 1234，确保生成的随机数可复现性
import numpy as np

# 导入 Python 内置的随机模块并设置其种子为 1234，以保证生成的随机数可复现性
import random

# 使用 NumPy 设置全局随机种子，确保在使用 NumPy 的随机函数时生成的随机数也具有可预测性和一致性
np.random.seed(1234)

# 设置 Python 内置随机数生成器的种子，以确保在使用 random 模块生成的随机数也具有可复现性
random.seed(1234)
```