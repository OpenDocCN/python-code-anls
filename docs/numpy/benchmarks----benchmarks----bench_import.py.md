# `.\numpy\benchmarks\benchmarks\bench_import.py`

```py
# 导入 subprocess 模块中的 call 函数，用于执行外部命令
# 导入 sys 模块中的 executable 变量，表示当前 Python 解释器的路径
# 导入 timeit 模块中的 default_timer 函数，用于获取当前时间
from subprocess import call
from sys import executable
from timeit import default_timer

# 导入 Benchmark 类，该类在 common 模块中定义
from .common import Benchmark

# 定义一个名为 Import 的类，继承自 Benchmark 类
class Import(Benchmark):
    # timer 属性指定为 default_timer 函数，用于计时
    timer = default_timer

    # 定义一个执行外部命令的方法，参数为命令字符串
    def execute(self, command):
        # 调用 subprocess 模块中的 call 函数执行命令
        call((executable, '-c', command))

    # 定义一个测试导入 numpy 模块耗时的方法
    def time_numpy(self):
        self.execute('import numpy')

    # 定义一个测试导入 numpy 和 inspect 模块耗时的方法
    # 此处注释提问了避免导入 inspect 模块可能带来的效率提升
    def time_numpy_inspect(self):
        self.execute('import numpy, inspect')

    # 定义一个测试从 numpy 模块导入 fft 子模块耗时的方法
    def time_fft(self):
        self.execute('from numpy import fft')

    # 定义一个测试从 numpy 模块导入 linalg 子模块耗时的方法
    def time_linalg(self):
        self.execute('from numpy import linalg')

    # 定义一个测试从 numpy 模块导入 ma 子模块耗时的方法
    def time_ma(self):
        self.execute('from numpy import ma')

    # 定义一个测试从 numpy 模块导入 matlib 子模块耗时的方法
    def time_matlib(self):
        self.execute('from numpy import matlib')

    # 定义一个测试从 numpy 模块导入 random 子模块耗时的方法
    def time_random(self):
        self.execute('from numpy import random')
```