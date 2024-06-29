# `.\numpy\doc\source\user\plots\matplotlib1.py`

```
# 导入 matplotlib.pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并简写为 np
import numpy as np

# 创建一个包含指定数据的 NumPy 数组
a = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22])

# 使用 matplotlib.pyplot 的 plot 函数绘制 a 数组的折线图
plt.plot(a) 
# 显示绘制的图形
plt.show()
```