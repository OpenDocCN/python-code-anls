# `.\pandas-ta\tests\__init__.py`

```py
# 导入必要的模块
import numpy as np
import matplotlib.pyplot as plt

# 生成一组随机数据
x = np.random.randn(1000)

# 创建一个频率直方图
plt.hist(x, bins=30, edgecolor='black')

# 设置图表标题
plt.title('Histogram of Random Data')

# 设置 X 轴标签
plt.xlabel('Value')

# 设置 Y 轴标签
plt.ylabel('Frequency')

# 显示图表
plt.show()
```