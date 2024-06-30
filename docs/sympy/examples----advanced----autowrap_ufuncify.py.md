# `D:\src\scipysrc\sympy\examples\advanced\autowrap_ufuncify.py`

```
#!/usr/bin/env python
"""
Setup ufuncs for the legendre polynomials
-----------------------------------------

This example demonstrates how you can use the ufuncify utility in SymPy
to create fast, customized universal functions for use with numpy
arrays. An autowrapped sympy expression can be significantly faster than
what you would get by applying a sequence of the ufuncs shipped with
numpy. [0]

You need to have numpy installed to run this example, as well as a
working fortran compiler.


[0]:
http://ojensen.wordpress.com/2010/08/10/fast-ufunc-ish-hydrogen-solutions/
"""

import sys

# 导入 SymPy 的外部模块 import_module
from sympy.external import import_module

# 尝试导入 numpy 模块
np = import_module('numpy')
if not np:
    # 如果导入失败，输出错误信息并退出程序
    sys.exit("Cannot import numpy. Exiting.")

# 尝试导入 matplotlib.pyplot 模块
plt = import_module('matplotlib.pyplot')
if not plt:
    # 如果导入失败，输出错误信息并退出程序
    sys.exit("Cannot import matplotlib.pyplot. Exiting.")

# 导入 mpmath 模块
import mpmath

# 导入 sympy 的 ufuncify 函数和其他需要的类和函数
from sympy.utilities.autowrap import ufuncify
from sympy import symbols, legendre, pprint


def main():

    # 打印本文件的文档字符串
    print(__doc__)

    # 定义符号变量 x
    x = symbols('x')

    # 创建一个 numpy 数组，用于应用 ufuncs
    grid = np.linspace(-1, 1, 1000)

    # 设置 mpmath 的精度为 20 个有效数字，用于验证
    mpmath.mp.dps = 20

    print("Compiling legendre ufuncs and checking results:")

    # 循环生成并检查 legendre 多项式
    for n in range(6):

        # 设置 SymPy 表达式以进行 ufuncify
        expr = legendre(n, x)
        print("The polynomial of degree %i is" % n)
        pprint(expr)

        # 使用 ufuncify 将 SymPy 表达式转换为二进制形式
        binary_poly = ufuncify(x, expr)

        # 现在可以在 numpy 数组上使用生成的函数
        polyvector = binary_poly(grid)

        # 检查生成的值与 mpmath 的 legendre 函数的值
        maxdiff = 0
        for j in range(len(grid)):
            precise_val = mpmath.legendre(n, grid[j])
            diff = abs(polyvector[j] - precise_val)
            if diff > maxdiff:
                maxdiff = diff
        print("The largest error in applied ufunc was %e" % maxdiff)
        assert maxdiff < 1e-14

        # 将 autowrapped 的 legendre 多项式附加到 sympy 函数，并绘制计算值
        plot1 = plt.pyplot.plot(grid, polyvector, hold=True)


    # 显示包含由二进制函数计算的值的图形
    print("Here's a plot with values calculated by the wrapped binary functions")
    plt.pyplot.show()

if __name__ == '__main__':
    main()
```