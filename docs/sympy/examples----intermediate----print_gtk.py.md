# `D:\src\scipysrc\sympy\examples\intermediate\print_gtk.py`

```
#!/usr/bin/env python

"""print_gtk example

Demonstrates printing with gtkmathview using mathml
"""

# 导入 sympy 库中的 Integral, Limit, print_gtk, sin, Symbol 函数和类
from sympy import Integral, Limit, print_gtk, sin, Symbol

# 主函数定义
def main():
    # 创建符号变量 x
    x = Symbol('x')

    # 创建一个极限对象 example_limit，表示 sin(x)/x 当 x 趋向于 0 时的极限
    example_limit = Limit(sin(x)/x, x, 0)
    # 使用 gtkmathview 打印 example_limit 的数学表达式
    print_gtk(example_limit)

    # 创建一个积分对象 example_integral，表示从 0 到 1 的 x 积分
    example_integral = Integral(x, (x, 0, 1))
    # 使用 gtkmathview 打印 example_integral 的数学表达式
    print_gtk(example_integral)

# 如果该脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```