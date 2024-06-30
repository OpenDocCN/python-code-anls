# `D:\src\scipysrc\sympy\examples\beginner\differentiation.py`

```
#!/usr/bin/env python

"""Differentiation example

Demonstrates some differentiation operations.
"""

# 导入 sympy 模块中的 pprint 和 Symbol 类
from sympy import pprint, Symbol

# 定义程序的主函数
def main():
    # 创建符号变量 a 和 b
    a = Symbol('a')
    b = Symbol('b')
    # 定义表达式 e
    e = (a + 2*b)**5

    # 输出表达式 e
    print("\nExpression : ")
    print()
    pprint(e)

    # 输出关于变量 a 的一阶导数
    print("\n\nDifferentiating w.r.t. a:")
    print()
    pprint(e.diff(a))

    # 输出关于变量 b 的一阶导数
    print("\n\nDifferentiating w.r.t. b:")
    print()
    pprint(e.diff(b))

    # 输出关于变量 b 的一阶导数后，再对结果关于变量 a 求二阶导数
    print("\n\nSecond derivative of the above result w.r.t. a:")
    print()
    pprint(e.diff(b).diff(a, 2))

    # 将表达式 e 先展开，再对展开后的结果关于变量 b 求一阶导数，再对结果关于变量 a 求二阶导数
    print("\n\nExpanding the above result:")
    print()
    pprint(e.expand().diff(b).diff(a, 2))

    # 输出空行
    print()

# 如果该脚本作为主程序运行，则调用主函数 main()
if __name__ == "__main__":
    main()
```