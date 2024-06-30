# `D:\src\scipysrc\sympy\examples\beginner\basic.py`

```
#!/usr/bin/env python

"""Basic example

Demonstrates how to create symbols and print some algebra operations.
"""

# 导入 sympy 库中的 Symbol 和 pprint 函数
from sympy import Symbol, pprint

# 定义程序的主函数
def main():
    # 创建符号变量 a, b, c
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    # 定义表达式 e，包含符号变量和运算
    e = (a*b*b + 2*b*a*b)**c

    # 打印空行
    print('')
    # 使用 pprint 函数美观地打印表达式 e
    pprint(e)
    # 打印空行
    print('')

# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```