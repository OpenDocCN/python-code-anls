# `D:\src\scipysrc\sympy\examples\beginner\precision.py`

```
#!/usr/bin/env python

"""Precision Example

Demonstrates SymPy's arbitrary integer precision abilities
"""

# 导入 sympy 库及相关模块
import sympy
from sympy import Mul, Pow, S

# 主函数
def main():
    # 创建一个指数运算表达式，不进行求值
    x = Pow(2, 50, evaluate=False)
    # 创建一个小数指数运算表达式，不进行求值
    y = Pow(10, -50, evaluate=False)
    # 创建一个大的、未求值的乘法表达式
    m = Mul(x, y, evaluate=False)
    # 对乘法表达式进行求值
    e = S(2)**50/S(10)**50
    # 打印比较两种表达式的结果
    print("{} == {}".format(m, e))

# 如果这个脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```