# `D:\src\scipysrc\scipy\scipy\interpolate\_interpnd_info.py`

```
"""
Here we perform some symbolic computations required for the N-D
interpolation routines in `interpnd.pyx`.

"""
# 导入 sympy 库中的符号、二项式系数计算和矩阵功能
from sympy import symbols, binomial, Matrix


def _estimate_gradients_2d_global():
    # 定义符号变量：f1, f2, df1, df2, x
    f1, f2, df1, df2, x = symbols(['f1', 'f2', 'df1', 'df2', 'x'])
    # 定义系数列表 c，用于符号计算
    c = [f1, (df1 + 3*f1)/3, (df2 + 3*f2)/3, f2]

    # 计算加权函数 w
    w = 0
    for k in range(4):
        w += binomial(3, k) * c[k] * x**k*(1-x)**(3-k)

    # 计算 w 对 x 的二阶导数，并展开
    wpp = w.diff(x, 2).expand()
    # 计算 wpp 的平方并对 x 在 [0, 1] 上积分并展开
    intwpp2 = (wpp**2).integrate((x, 0, 1)).expand()

    # 构造矩阵 A，包含 df1^2 和 df1*df2 的系数
    A = Matrix([[intwpp2.coeff(df1**2), intwpp2.coeff(df1*df2)/2],
                [intwpp2.coeff(df1*df2)/2, intwpp2.coeff(df2**2)]])

    # 构造矩阵 B，包含 df1 和 df2 的系数
    B = Matrix([[intwpp2.coeff(df1).subs(df2, 0)],
                [intwpp2.coeff(df2).subs(df1, 0)]]) / 2

    # 输出矩阵 A
    print("A")
    print(A)
    # 输出矩阵 B
    print("B")
    print(B)
    # 输出解向量
    print("solution")
    print(A.inv() * B)
```