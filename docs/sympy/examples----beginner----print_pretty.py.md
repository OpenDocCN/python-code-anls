# `D:\src\scipysrc\sympy\examples\beginner\print_pretty.py`

```
#!/usr/bin/env python

"""Pretty print example

Demonstrates pretty printing.
"""

# 从 sympy 库导入所需的符号和打印函数
from sympy import Symbol, pprint, sin, cos, exp, sqrt, MatrixSymbol, KroneckerProduct

# 定义主函数
def main():
    # 创建符号变量 x 和 y
    x = Symbol("x")
    y = Symbol("y")

    # 创建矩阵符号 a, b, c，均为 1x1 矩阵
    a = MatrixSymbol("a", 1, 1)
    b = MatrixSymbol("b", 1, 1)
    c = MatrixSymbol("c", 1, 1)

    # 打印 x 的 x 次幂
    pprint( x**x )
    # 打印空行，用于分隔输出
    print('\n')

    # 打印 x 的平方加上 y 加上 x
    pprint(x**2 + y + x)
    print('\n')

    # 打印 sin(x) 的 x 次幂
    pprint(sin(x)**x)
    print('\n')

    # 打印 sin(x) 的 cos(x) 次幂
    pprint( sin(x)**cos(x) )
    print('\n')

    # 打印 sin(x) 除以 (cos(x) 的平方乘以 x 的 x 次幂加上 2*y)
    pprint( sin(x)/(cos(x)**2 * x**x + (2*y)) )
    print('\n')

    # 打印 sin(x^2 + exp(x))
    pprint( sin(x**2 + exp(x)) )
    print('\n')

    # 打印 sqrt(exp(x))
    pprint( sqrt(exp(x)) )
    print('\n')

    # 打印 sqrt(sqrt(exp(x)))
    pprint( sqrt(sqrt(exp(x))) )
    print('\n')

    # 打印 (1/cos(x)) 在 x=0 处展开的前 10 项级数
    pprint( (1/cos(x)).series(x, 0, 10) )
    print('\n')

    # 打印 a 乘以 KroneckerProduct(b, c)
    pprint(a*(KroneckerProduct(b, c)))
    print('\n')

# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```