# `D:\src\scipysrc\sympy\sympy\diffgeom\tests\test_hyperbolic_space.py`

```
r'''
unit test describing the hyperbolic half-plane with the Poincare metric. This
is a basic model of hyperbolic geometry on the (positive) half-space

{(x,y) \in R^2 | y > 0}

with the Riemannian metric

ds^2 = (dx^2 + dy^2)/y^2

It has constant negative scalar curvature = -2

https://en.wikipedia.org/wiki/Poincare_half-plane_model
'''
# 从 sympy 库中导入需要的模块和函数
from sympy.matrices.dense import diag
from sympy.diffgeom import (twoform_to_matrix,
                            metric_to_Christoffel_1st, metric_to_Christoffel_2nd,
                            metric_to_Riemann_components, metric_to_Ricci_components)
import sympy.diffgeom.rn
from sympy.tensor.array import ImmutableDenseNDimArray


# 定义测试函数 test_H2()
def test_H2():
    # 导入 tensor product (TP) 和 R2 空间对象
    TP = sympy.diffgeom.TensorProduct
    R2 = sympy.diffgeom.rn.R2
    # 获取 R2 空间对象的 y, dy, dx 符号
    y = R2.y
    dy = R2.dy
    dx = R2.dx
    # 定义 Poincare 半平面上的度量 g
    g = (TP(dx, dx) + TP(dy, dy))*y**(-2)
    # 使用度量转换为矩阵形式
    automat = twoform_to_matrix(g)
    # 创建对角矩阵 mat
    mat = diag(y**(-2), y**(-2))
    # 断言 mat 和 automat 相等
    assert mat == automat

    # 计算第一类 Christoffel 符号 gamma1
    gamma1 = metric_to_Christoffel_1st(g)
    assert gamma1[0, 0, 0] == 0
    assert gamma1[0, 0, 1] == -y**(-3)
    assert gamma1[0, 1, 0] == -y**(-3)
    assert gamma1[0, 1, 1] == 0

    assert gamma1[1, 1, 1] == -y**(-3)
    assert gamma1[1, 1, 0] == 0
    assert gamma1[1, 0, 1] == 0
    assert gamma1[1, 0, 0] == y**(-3)

    # 计算第二类 Christoffel 符号 gamma2
    gamma2 = metric_to_Christoffel_2nd(g)
    assert gamma2[0, 0, 0] == 0
    assert gamma2[0, 0, 1] == -y**(-1)
    assert gamma2[0, 1, 0] == -y**(-1)
    assert gamma2[0, 1, 1] == 0

    assert gamma2[1, 1, 1] == -y**(-1)
    assert gamma2[1, 1, 0] == 0
    assert gamma2[1, 0, 1] == 0
    assert gamma2[1, 0, 0] == y**(-1)

    # 计算 Riemann 张量分量 Rm
    Rm = metric_to_Riemann_components(g)
    assert Rm[0, 0, 0, 0] == 0
    assert Rm[0, 0, 0, 1] == 0
    assert Rm[0, 0, 1, 0] == 0
    assert Rm[0, 0, 1, 1] == 0

    assert Rm[0, 1, 0, 0] == 0
    assert Rm[0, 1, 0, 1] == -y**(-2)
    assert Rm[0, 1, 1, 0] == y**(-2)
    assert Rm[0, 1, 1, 1] == 0

    assert Rm[1, 0, 0, 0] == 0
    assert Rm[1, 0, 0, 1] == y**(-2)
    assert Rm[1, 0, 1, 0] == -y**(-2)
    assert Rm[1, 0, 1, 1] == 0

    assert Rm[1, 1, 0, 0] == 0
    assert Rm[1, 1, 0, 1] == 0
    assert Rm[1, 1, 1, 0] == 0
    assert Rm[1, 1, 1, 1] == 0

    # 计算 Ricci 张量分量 Ric
    Ric = metric_to_Ricci_components(g)
    assert Ric[0, 0] == -y**(-2)
    assert Ric[0, 1] == 0
    assert Ric[1, 0] == 0
    assert Ric[1, 1] == -y**(-2)

    assert Ric == ImmutableDenseNDimArray([-y**(-2), 0, 0, -y**(-2)], (2, 2))

    ## scalar curvature is -2
    # 断言标量曲率 R 等于 -2
    R = (Ric[0, 0] + Ric[1, 1])*y**2
    assert R == -2

    ## Gauss curvature is -1
    # 断言高斯曲率为 -1
    assert R/2 == -1
```