# `D:\src\scipysrc\sympy\examples\advanced\hydrogen.py`

```
#!/usr/bin/env python

"""
This example shows how to work with the Hydrogen radial wavefunctions.
"""

# 导入所需的模块和函数
from sympy import Eq, Integral, oo, pprint, symbols
from sympy.physics.hydrogen import R_nl

# 定义主函数
def main():
    # 打印标题
    print("Hydrogen radial wavefunctions:")
    
    # 定义符号变量
    a, r = symbols("a r")
    
    # 打印并计算 R_{21} 的径向波函数
    print("R_{21}:")
    pprint(R_nl(2, 1, a, r))
    
    # 打印并计算 R_{60} 的径向波函数
    print("R_{60}:")
    pprint(R_nl(6, 0, a, r))

    # 打印归一化计算的标题
    print("Normalization:")
    
    # 计算并打印 R_{10} 的归一化积分
    i = Integral(R_nl(1, 0, 1, r)**2 * r**2, (r, 0, oo))
    pprint(Eq(i, i.doit()))
    
    # 计算并打印 R_{20} 的归一化积分
    i = Integral(R_nl(2, 0, 1, r)**2 * r**2, (r, 0, oo))
    pprint(Eq(i, i.doit()))
    
    # 计算并打印 R_{21} 的归一化积分
    i = Integral(R_nl(2, 1, 1, r)**2 * r**2, (r, 0, oo))
    pprint(Eq(i, i.doit()))

# 如果作为主程序运行，则执行主函数
if __name__ == '__main__':
    main()
```