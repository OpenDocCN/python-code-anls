# `D:\src\scipysrc\sympy\examples\intermediate\trees.py`

```
# 指定脚本运行环境为 Python
#!/usr/bin/env python

# 脚本的简要描述和用途
"""
Calculates the Sloane's A000055 integer sequence, i.e. the "Number of
trees with n unlabeled nodes."

You can also google for "The On-Line Encyclopedia of Integer Sequences"
and paste in the sequence returned by this script:

1, 1, 1, 1, 2, 3, 6, 11, 23, 47, 106

and it will show you the A000055
"""

# 导入 sympy 库中的 Symbol 和 Poly 类
from sympy import Symbol, Poly


# 定义函数 T(x)，计算特定数学表达式
def T(x):
    return x + x**2 + 2*x**3 + 4*x**4 + 9*x**5 + 20*x**6 + 48 * x**7 + \
        115*x**8 + 286*x**9 + 719*x**10


# 定义函数 A(x)，根据给定的数学表达式计算结果
def A(x):
    return 1 + T(x) - T(x)**2/2 + T(x**2)/2


# 主函数，用于计算并打印 Sloane's A000055 数列的表达式及前11个系数
def main():
    # 创建符号变量 x
    x = Symbol("x")
    # 构造多项式对象 s，其表达式为 A(x)
    s = Poly(A(x), x)
    # 获取多项式 s 的前11个系数，以列表形式逆序排列
    num = list(reversed(s.coeffs()))[:11]

    # 打印多项式 s 的表达式
    print(s.as_expr())
    # 打印 num 列表，即前11个系数
    print(num)

# 如果脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```