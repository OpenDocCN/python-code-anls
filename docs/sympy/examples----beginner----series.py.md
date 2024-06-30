# `D:\src\scipysrc\sympy\examples\beginner\series.py`

```
#!/usr/bin/env python

"""Series example

Demonstrates series.
"""

# 从 sympy 库导入需要的符号、余弦、正弦、以及打印函数
from sympy import Symbol, cos, sin, pprint

# 主函数定义
def main():
    # 定义符号变量 x
    x = Symbol('x')

    # 计算 sec(x) 的级数展开
    e = 1/cos(x)
    print('')
    print("Series for sec(x):")
    print('')
    # 使用 sympy 的 pprint 函数打印 sec(x) 的级数展开结果
    pprint(e.series(x, 0, 10))
    print("\n")

    # 计算 csc(x) 的级数展开
    e = 1/sin(x)
    print("Series for csc(x):")
    print('')
    # 使用 sympy 的 pprint 函数打印 csc(x) 的级数展开结果
    pprint(e.series(x, 0, 4))
    print('')

# 如果此脚本被直接执行，则调用 main 函数
if __name__ == "__main__":
    main()
```