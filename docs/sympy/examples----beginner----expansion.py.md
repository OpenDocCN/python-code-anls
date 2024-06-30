# `D:\src\scipysrc\sympy\examples\beginner\expansion.py`

```
#!/usr/bin/env python
# 指定使用Python解释器运行该脚本

"""Expansion Example

Demonstrates how to expand expressions.
"""
# 脚本的简要描述和说明

from sympy import pprint, Symbol
# 导入Sympy库中的pprint函数和Symbol类

def main():
    # 定义符号变量a和b
    a = Symbol('a')
    b = Symbol('b')
    # 定义表达式e为(a + b)的五次幂
    e = (a + b)**5

    print("\nExpression:")
    # 打印原始表达式e
    pprint(e)
    print('\nExpansion of the above expression:')
    # 打印表达式e的展开结果
    pprint(e.expand())
    print()

if __name__ == "__main__":
    # 如果脚本作为主程序运行，则调用main函数
    main()
```