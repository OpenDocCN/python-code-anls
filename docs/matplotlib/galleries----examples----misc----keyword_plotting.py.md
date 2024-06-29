# `D:\src\scipysrc\matplotlib\galleries\examples\misc\keyword_plotting.py`

```
"""
======================
Plotting with keywords
======================

Some data structures, like dict, `structured numpy array
<https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays>`_
or `pandas.DataFrame` provide access to labelled data via string index access
``data[key]``.

For these data types, Matplotlib supports passing the whole datastructure via the
``data`` keyword argument, and using the string names as plot function parameters,
where you'd normally pass in your data.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，并使用别名 np
import numpy as np

# 设定随机数种子，以便结果可重现
np.random.seed(19680801)

# 创建一个字典 data，包含以下键值对：
# 'a': 从 0 到 49 的整数数组
# 'c': 包含 50 个在 [0, 50) 范围内的随机整数
# 'd': 包含 50 个标准正态分布随机数，再取绝对值并乘以 100
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
# 添加新键 'b' 到 data，其值为 'a' 对应值加上标准正态分布随机数乘以 10
data['b'] = data['a'] + 10 * np.random.randn(50)
# 将 'd' 对应值取绝对值后乘以 100，更新 data 中 'd' 对应的值
data['d'] = np.abs(data['d']) * 100

# 创建一个图形和坐标系对象
fig, ax = plt.subplots()
# 在坐标系 ax 上绘制散点图，使用 data 参数传入数据：
# x 轴使用 data['a'] 的数据
# y 轴使用 data['b'] 的数据
# 颜色参数 c 使用 data['c'] 的数据
# 点的大小参数 s 使用 data['d'] 的数据
ax.scatter('a', 'b', c='c', s='d', data=data)
# 设置 x 轴标签为 'entry a'，y 轴标签为 'entry b'
ax.set(xlabel='entry a', ylabel='entry b')

# 显示图形
plt.show()
```