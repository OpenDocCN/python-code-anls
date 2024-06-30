# `D:\src\scipysrc\sympy\examples\beginner\plotting_nice_plot.py`

```
#!/usr/bin/env python
"""
Plotting example

Demonstrates simple plotting.
"""

# 从 sympy 库导入所需的符号和函数
from sympy import Symbol, cos, sin, log, tan
# 从 sympy.plotting 模块导入 PygletPlot 类
from sympy.plotting import PygletPlot
# 从 sympy.abc 模块导入符号 x 和 y
from sympy.abc import x, y

# 主函数，程序的入口
def main():
    # 定义三个函数表达式
    fun1 = cos(x)*sin(y)  # 第一个函数表达式
    fun2 = sin(x)*sin(y)  # 第二个函数表达式
    fun3 = cos(y) + log(tan(y/2)) + 0.2*x  # 第三个函数表达式

    # 创建 PygletPlot 对象并进行绘图
    PygletPlot(fun1, fun2, fun3, [x, -0.00, 12.4, 40], [y, 0.1, 2, 40])

# 如果该脚本作为主程序运行，则调用主函数 main()
if __name__ == "__main__":
    main()
```