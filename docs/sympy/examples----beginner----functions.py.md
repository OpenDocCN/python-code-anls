# `D:\src\scipysrc\sympy\examples\beginner\functions.py`

```
#!/usr`
#!/usr/bin/env python

"""Functions example

Demonstrates functions defined in SymPy.
"""

# 从 sympy 模块中导入所需的函数和类
from sympy import pprint, Symbol, log, exp

# 主函数，程序的入口
def main():
    # 创建符号变量 a 和 b
    a = Symbol('a')
    b = Symbol('b')
    
    # 计算并打印 log((a + b)^5)，并使用 sympy 的美观打印函数 pprint 进行输出
    e = log((a + b)**5)
    print()
    pprint(e)
    print('\n')

    # 计算 e 的指数函数，并打印结果
    e = exp(e)
    pprint(e)
    print('\n')

    # 直接计算 log(exp((a + b)^5))，并使用 pprint 打印结果
    e = log(exp((a + b)**5))
    pprint(e)
    print()

# 如果当前脚本被直接执行，则执行主函数 main()
if __name__ == "__main__":
    main()
```