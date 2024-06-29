# `D:\src\scipysrc\matplotlib\galleries\examples\misc\print_stdout_sgskip.py`

```py
"""
============
Print Stdout
============

print png to standard out

usage: python print_stdout.py > somefile.png

"""

# 导入系统模块
import sys

# 导入 matplotlib 库
import matplotlib

# 设置 matplotlib 使用的后端为 'Agg'
matplotlib.use('Agg')
# 导入 pyplot 模块
import matplotlib.pyplot as plt

# 绘制简单的折线图，数据为 [1, 2, 3]
plt.plot([1, 2, 3])
# 将图形保存为 PNG 格式，并输出到标准输出流
plt.savefig(sys.stdout.buffer)
```