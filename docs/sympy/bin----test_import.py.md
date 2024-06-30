# `D:\src\scipysrc\sympy\bin\test_import.py`

```
# 导入用于支持 Python 未来版本的 print_function
from __future__ import print_function

# 导入默认计时器作为 clock，并使用起始时间记录当前时间
from timeit import default_timer as clock

# 导入 get_sympy 模块中的 path_hack 函数，用于修改路径以支持 sympy 导入
from get_sympy import path_hack

# 调用 path_hack 函数，执行路径修改
path_hack()

# 记录当前时间到变量 t，作为开始计时
t = clock()

# 导入 sympy 模块，如果需要的话，这将导致模块的初始化
import sympy

# 计算从记录时间到当前时间的时间差，并将结果保存在变量 t 中
t = clock() - t

# 输出变量 t，即程序运行所消耗的时间
print(t)
```