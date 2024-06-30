# `D:\src\scipysrc\scipy\scipy\fftpack\tests\gendata.py`

```
# 导入 NumPy 库，简写为 np
import numpy as np
# 从 SciPy 库的 io 模块中导入 loadmat 函数
from scipy.io import loadmat

# 使用 loadmat 函数加载名为 'test.mat' 的 MATLAB 文件，将其解压缩为 Python 对象
# squeeze_me=True 将尝试压缩结构数组中的单元素维度
# struct_as_record=True 将 MATLAB 结构体转换为 Python 命名元组
# mat_dtype=True 将数据以 MATLAB 的原始数据类型加载
m = loadmat('test.mat', squeeze_me=True, struct_as_record=True,
            mat_dtype=True)

# 使用 np.savez 函数将加载的 MATLAB 数据保存为名为 'test.npz' 的 NumPy 压缩存档文件
# **m 将字典 m 中的所有键值对作为关键字参数传递给 np.savez 函数，保存所有加载的数据
np.savez('test.npz', **m)
```