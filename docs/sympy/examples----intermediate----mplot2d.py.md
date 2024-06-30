# `D:\src\scipysrc\sympy\examples\intermediate\mplot2d.py`

```
#!/usr/bin/env python

"""Matplotlib 2D plotting example

Demonstrates plotting with matplotlib.
"""

import sys

from sample import sample  # 导入sample模块中的sample函数

from sympy import sqrt, Symbol  # 导入sympy库中的sqrt和Symbol类
from sympy.utilities.iterables import is_sequence  # 导入sympy库中iterables模块的is_sequence函数
from sympy.external import import_module  # 导入sympy库中external模块的import_module函数


def mplot2d(f, var, *, show=True):
    """
    Plot a 2d function using matplotlib/Tk.
    """
    
    import warnings  # 导入warnings模块
    warnings.filterwarnings("ignore", r"Could not match \S")  # 忽略特定警告信息

    p = import_module('pylab')  # 尝试导入pylab模块
    if not p:
        sys.exit("Matplotlib is required to use mplot2d.")  # 若导入失败则退出程序并显示错误信息

    if not is_sequence(f):  # 检查f是否为可迭代对象，若不是则转为单元素列表
        f = [f, ]

    for f_i in f:  # 遍历函数列表
        x, y = sample(f_i, var)  # 对每个函数f_i进行采样，获取x和y坐标
        p.plot(x, y)  # 使用matplotlib绘制图形

    p.draw()  # 绘制图形
    if show:  # 若show为True，则显示图形界面
        p.show()


def main():
    x = Symbol('x')  # 创建符号变量x

    # mplot2d(log(x), (x, 0, 2, 100))
    # mplot2d([sin(x), -sin(x)], (x, float(-2*pi), float(2*pi), 50))
    mplot2d([sqrt(x), -sqrt(x), sqrt(-x), -sqrt(-x)], (x, -40.0, 40.0, 80))  # 调用mplot2d函数绘制多个函数图像

if __name__ == "__main__":
    main()
```