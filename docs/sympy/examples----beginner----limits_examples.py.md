# `D:\src\scipysrc\sympy\examples\beginner\limits_examples.py`

```
# 使用 Python 解释器运行该脚本
#!/usr/bin/env python

# 演示极限的示例
"""Limits Example

Demonstrates limits.
"""

# 从 sympy 库中导入所需函数和符号
from sympy import exp, log, Symbol, Rational, sin, limit, sqrt, oo


# 定义一个函数，计算 x 的立方根
def sqrt3(x):
    return x**Rational(1, 3)


# 定义一个辅助函数，用于打印计算结果和预期结果
def show(computed, correct):
    print("computed:", computed, "correct:", correct)


# 主函数
def main():
    # 定义符号 x
    x = Symbol("x")

    # 计算极限：sqrt(x**2 - 5*x + 6) - x 当 x 趋向无穷大时的极限，预期结果为 -5/2
    show( limit(sqrt(x**2 - 5*x + 6) - x, x, oo), -Rational(5)/2 )

    # 计算极限：x*(sqrt(x**2 + 1) - x) 当 x 趋向无穷大时的极限，预期结果为 1/2
    show( limit(x*(sqrt(x**2 + 1) - x), x, oo), Rational(1)/2 )

    # 计算极限：x - sqrt3(x**3 - 1) 当 x 趋向无穷大时的极限，预期结果为 0
    show( limit(x - sqrt3(x**3 - 1), x, oo), Rational(0) )

    # 计算极限：log(1 + exp(x))/x 当 x 趋向负无穷大时的极限，预期结果为 0
    show( limit(log(1 + exp(x))/x, x, -oo), Rational(0) )

    # 计算极限：log(1 + exp(x))/x 当 x 趋向无穷大时的极限，预期结果为 1
    show( limit(log(1 + exp(x))/x, x, oo), Rational(1) )

    # 计算极限：sin(3*x)/x 当 x 趋向 0 时的极限，预期结果为 3
    show( limit(sin(3*x)/x, x, 0), Rational(3) )

    # 计算极限：sin(5*x)/sin(2*x) 当 x 趋向 0 时的极限，预期结果为 5/2
    show( limit(sin(5*x)/sin(2*x), x, 0), Rational(5)/2 )

    # 计算极限：((x - 1)/(x + 1))**x 当 x 趋向无穷大时的极限，预期结果为 e^(-2)
    show( limit(((x - 1)/(x + 1))**x, x, oo), exp(-2))

if __name__ == "__main__":
    main()
```