# `D:\src\scipysrc\sympy\examples\advanced\curvilinear_coordinates.py`

```
#!/usr/bin/env python

"""
This example shows how to work with coordinate transformations, curvilinear
coordinates and a little bit with differential geometry.

It takes polar, cylindrical, spherical, rotating disk coordinates and others
and calculates all kinds of interesting properties, like Jacobian, metric
tensor, Laplace operator, ...
"""

# 导入必要的库和函数
from sympy import var, sin, cos, pprint, Matrix, eye, trigsimp, Eq, \
    Function, simplify, sinh, cosh, expand, symbols


def laplace(f, g_inv, g_det, X):
    """
    Calculates Laplace(f), using the inverse metric g_inv, the determinant of
    the metric g_det, all in variables X.
    """
    # 初始化结果为零
    r = 0
    # 计算 Laplace 算子的第一部分
    for i in range(len(X)):
        for j in range(len(X)):
            r += g_inv[i, j]*f.diff(X[i]).diff(X[j])
    # 计算 Laplace 算子的第二部分
    for sigma in range(len(X)):
        for alpha in range(len(X)):
            r += g_det.diff(X[sigma]) * g_inv[sigma, alpha] * \
                f.diff(X[alpha]) / (2*g_det)
    return r


def transform(name, X, Y, *, g_correct=None, recursive=False):
    """
    Transforms from cartesian coordinates X to any curvilinear coordinates Y.

    It printing useful information, like Jacobian, metric tensor, determinant
    of metric, Laplace operator in the new coordinates, ...

    g_correct ... if not None, it will be taken as the metric --- this is
                  useful if sympy's trigsimp() is not powerful enough to
                  simplify the metric so that it is usable for later
                  calculation. Leave it as None, only if the metric that
                  transform() prints is not simplified, you can help it by
                  specifying the correct one.

    recursive ... apply recursive trigonometric simplification (use only when
                  needed, as it is an expensive operation)
    """
    # 打印变换名称
    print("_"*80)
    print("Transformation:", name)
    # 打印坐标变换关系
    for x, y in zip(X, Y):
        pprint(Eq(y, x))
    # 计算雅可比矩阵
    J = X.jacobian(Y)
    print("Jacobian:")
    pprint(J)
    # 计算度量张量 g_{ij}
    g = J.T * eye(J.shape[0]) * J
    g = g.applyfunc(expand)
    print("metric tensor g_{ij}:")
    pprint(g)
    # 如果提供了手动修正的度量张量 g_{ij}，则使用它
    if g_correct is not None:
        g = g_correct
        print("metric tensor g_{ij} specified by hand:")
        pprint(g)
    # 计算逆度量张量 g^{ij}
    print("inverse metric tensor g^{ij}:")
    g_inv = g.inv(method="ADJ")
    g_inv = g_inv.applyfunc(simplify)
    pprint(g_inv)
    # 计算度量张量的行列式 det g_{ij}
    print("det g_{ij}:")
    g_det = g.det()
    pprint(g_det)
    # 定义一个函数 f(*Y)，表示在新坐标系下的函数
    f = Function("f")(*list(Y))
    # 计算 Laplace 算子
    print("Laplace:")
    pprint(laplace(f, g_inv, g_det, Y))


def main():
    # 定义符号变量
    mu, nu, rho, theta, phi, sigma, tau, a, t, x, y, z, w = symbols(
        "mu, nu, rho, theta, phi, sigma, tau, a, t, x, y, z, w")

    # 进行极坐标变换
    transform("polar", Matrix([rho*cos(phi), rho*sin(phi)]), [rho, phi])

    # 进行柱坐标变换
    transform("cylindrical", Matrix([rho*cos(phi), rho*sin(phi), z]),
              [rho, phi, z])
    # 应用名为 "spherical" 的变换，使用给定的三维球坐标公式
    transform("spherical",
              Matrix([rho*sin(theta)*cos(phi), rho*sin(theta)*sin(phi),
                      rho*cos(theta)]),
              [rho, theta, phi],
              recursive=True
              )
    
    # 应用名为 "rotating disk" 的变换，使用给定的旋转盘坐标公式
    transform("rotating disk",
              Matrix([t,
                      x*cos(w*t) - y*sin(w*t),
                      x*sin(w*t) + y*cos(w*t),
                      z]),
              [t, x, y, z])
    
    # 应用名为 "parabolic" 的变换，使用给定的抛物线坐标公式
    transform("parabolic",
              Matrix([sigma*tau, (tau**2 - sigma**2) / 2]),
              [sigma, tau])
    
    # 应用名为 "bipolar" 的变换，使用给定的双极坐标公式
    transform("bipolar",
            Matrix([a*sinh(tau)/(cosh(tau)-cos(sigma)),
                a*sin(sigma)/(cosh(tau)-cos(sigma))]),
            [sigma, tau]
            )
    
    # 应用名为 "elliptic" 的变换，使用给定的椭圆坐标公式
    transform("elliptic",
              Matrix([a*cosh(mu)*cos(nu), a*sinh(mu)*sin(nu)]),
              [mu, nu]
              )
# 如果当前脚本被直接执行（而不是被导入到其他模块中），那么执行下面的代码
if __name__ == "__main__":
    # 调用主函数，开始程序的执行
    main()
```