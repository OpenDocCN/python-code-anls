# `D:\src\scipysrc\sympy\examples\advanced\fem.py`

```
#!/usr/bin/env python

"""FEM library

Demonstrates some simple finite element definitions, and computes a mass
matrix

$ python fem.py
[  1/60,     0, -1/360,     0, -1/90, -1/360]
[     0,  4/45,      0,  2/45,  2/45,  -1/90]
[-1/360,     0,   1/60, -1/90,     0, -1/360]
[     0,  2/45,  -1/90,  4/45,  2/45,      0]
[ -1/90,  2/45,      0,  2/45,  4/45,      0]
[-1/360, -1/90, -1/360,     0,     0,   1/60]

"""

from sympy import symbols, Symbol, factorial, Rational, zeros, eye, \
    integrate, diff, pprint, reduced

x, y, z = symbols('x,y,z')

class ReferenceSimplex:
    # 参考单纯形类，用于处理简单有限元素定义
    def __init__(self, nsd):
        # 初始化方法，接受空间维度参数
        self.nsd = nsd
        if nsd <= 3:
            coords = symbols('x,y,z')[:nsd]
        else:
            coords = [Symbol("x_%d" % d) for d in range(nsd)]
        self.coords = coords

    def integrate(self, f):
        # 对指定函数进行积分
        coords = self.coords
        nsd = self.nsd

        limit = 1
        for p in coords:
            limit -= p

        intf = f
        for d in range(0, nsd):
            p = coords[d]
            limit += p
            intf = integrate(intf, (p, 0, limit))
        return intf


def bernstein_space(order, nsd):
    # Bernstein空间生成函数，生成指定阶数和空间维度的基函数和系数
    if nsd > 3:
        raise RuntimeError("Bernstein only implemented in 1D, 2D, and 3D")
    sum = 0
    basis = []
    coeff = []

    if nsd == 1:
        b1, b2 = x, 1 - x
        for o1 in range(0, order + 1):
            for o2 in range(0, order + 1):
                if o1 + o2 == order:
                    aij = Symbol("a_%d_%d" % (o1, o2))
                    sum += aij*binomial(order, o1)*pow(b1, o1)*pow(b2, o2)
                    basis.append(binomial(order, o1)*pow(b1, o1)*pow(b2, o2))
                    coeff.append(aij)

    if nsd == 2:
        b1, b2, b3 = x, y, 1 - x - y
        for o1 in range(0, order + 1):
            for o2 in range(0, order + 1):
                for o3 in range(0, order + 1):
                    if o1 + o2 + o3 == order:
                        aij = Symbol("a_%d_%d_%d" % (o1, o2, o3))
                        fac = factorial(order) / (factorial(o1)*factorial(o2)*factorial(o3))
                        sum += aij*fac*pow(b1, o1)*pow(b2, o2)*pow(b3, o3)
                        basis.append(fac*pow(b1, o1)*pow(b2, o2)*pow(b3, o3))
                        coeff.append(aij)

    if nsd == 3:
        b1, b2, b3, b4 = x, y, z, 1 - x - y - z
        for o1 in range(0, order + 1):
            for o2 in range(0, order + 1):
                for o3 in range(0, order + 1):
                    for o4 in range(0, order + 1):
                        if o1 + o2 + o3 + o4 == order:
                            aij = Symbol("a_%d_%d_%d_%d" % (o1, o2, o3, o4))
                            fac = factorial(order)/(factorial(o1)*factorial(o2)*factorial(o3)*factorial(o4))
                            sum += aij*fac*pow(b1, o1)*pow(b2, o2)*pow(b3, o3)*pow(b4, o4)
                            basis.append(fac*pow(b1, o1)*pow(b2, o2)*pow(b3, o3)*pow(b4, o4))
                            coeff.append(aij)
    # 返回函数的三个变量：sum, coeff, basis
    return sum, coeff, basis
def main():
    # 创建一个参考简单形状对象，此处为二维三角形
    t = ReferenceSimplex(2)
    # 创建一个拉格朗日插值对象，二维二阶
    fe = Lagrange(2, 2)

    u = 0
    # 计算 u = sum_i u_i N_i，初始化 u 列表
    us = []
    # 遍历基函数数量
    for i in range(0, fe.nbf()):
        # 创建符号变量 u_i
        ui = Symbol("u_%d" % i)
        us.append(ui)
        # 构建 u = sum_i u_i N_i
        u += ui * fe.N[i]

    # 创建一个零矩阵 J，维度为基函数数量 x 基函数数量
    J = zeros(fe.nbf())
    # 遍历基函数数量
    for i in range(0, fe.nbf()):
        # 构建 Fi = u * N_i
        Fi = u * fe.N[i]
        # 打印 Fi
        print(Fi)
        # 再次遍历基函数数量
        for j in range(0, fe.nbf()):
            # 获取 u_j 符号变量
            uj = us[j]
            # 计算 Fi 对 u_j 的导数
            integrands = diff(Fi, uj)
            # 打印积分项
            print(integrands)
            # 将积分结果放入矩阵 J 中
            J[j, i] = t.integrate(integrands)

    # 美观打印输出矩阵 J
    pprint(J)


if __name__ == "__main__":
    main()
```