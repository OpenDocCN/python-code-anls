# `D:\src\scipysrc\sympy\examples\beginner\substitution.py`

```
#!/usr/bin/env python

"""Substitution example

Demonstrates substitution.
"""

import sympy
from sympy import pprint


def main():
    # 创建符号变量 x 和 y
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')

    # 定义表达式 e = 1/cos(x)
    e = 1/sympy.cos(x)
    # 打印表达式 e
    print()
    pprint(e)
    print('\n')
    # 打印将 cos(x) 替换为 y 后的表达式
    pprint(e.subs(sympy.cos(x), y))
    print('\n')
    # 打印将 cos(x) 替换为 y，并将 y 替换为 x**2 后的表达式
    pprint(e.subs(sympy.cos(x), y).subs(y, x**2))

    # 定义表达式 e = 1/log(x)，并将 x 替换为 2.71828
    e = 1/sympy.log(x)
    e = e.subs(x, sympy.Float("2.71828"))
    print('\n')
    # 打印表达式 e
    pprint(e)
    print('\n')
    # 打印表达式 e 的数值近似值
    pprint(e.evalf())
    print()

    # 创建符号变量 a 和 b
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    # 定义表达式 e = a*2 + a**b/a
    e = a*2 + a**b/a
    print('\n')
    # 打印表达式 e
    pprint(e)
    # 将 a 替换为 8，并打印结果
    a = 2
    print('\n')
    pprint(e.subs(a, 8))
    print()


if __name__ == "__main__":
    main()
```