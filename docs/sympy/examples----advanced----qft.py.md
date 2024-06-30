# `D:\src\scipysrc\sympy\examples\advanced\qft.py`

```
# 用于指定 Python 解释器的路径，使得脚本可以在不同的环境中运行
#!/usr/bin/env python

"""Quantum field theory example

* https://en.wikipedia.org/wiki/Quantum_field_theory

This particular example is a work in progress. Currently it calculates the
scattering amplitude of the process:

    electron + positron -> photon -> electron + positron

in QED (https://en.wikipedia.org/wiki/Quantum_electrodynamics). The aim
is to be able to do any kind of calculations in QED or standard model in
SymPy, but that's a long journey.

"""

# 导入 SymPy 库中需要使用的模块和函数
from sympy import Basic, Symbol, Matrix, \
    ones, sqrt, pprint, Eq, sympify

# 导入 SymPy 物理模块中定义的矩阵和符号
from sympy.physics import msigma, mgamma

# gamma^mu
# 定义四个 gamma 矩阵，用于量子场论中的计算
gamma0 = mgamma(0)
gamma1 = mgamma(1)
gamma2 = mgamma(2)
gamma3 = mgamma(3)
gamma5 = mgamma(5)

# sigma_i
# 定义三个 Pauli 矩阵，用于量子场论中的计算
sigma1 = msigma(1)
sigma2 = msigma(2)
sigma3 = msigma(3)

E = Symbol("E", real=True)
m = Symbol("m", real=True)


def u(p, r):
    """ p = (p1, p2, p3); r = 0,1 """
    # 检查 r 是否在有效范围内
    if r not in [1, 2]:
        raise ValueError("Value of r should lie between 1 and 2")
    p1, p2, p3 = p
    # 根据 r 的值选择合适的 ksi 矩阵
    if r == 1:
        ksi = Matrix([[1], [0]])
    else:
        ksi = Matrix([[0], [1]])
    # 计算 u spinor
    a = (sigma1*p1 + sigma2*p2 + sigma3*p3) / (E + m)*ksi
    if a == 0:
        a = zeros(2, 1)  # 如果 a 为零，则返回零矩阵
    # 返回 u spinor
    return sqrt(E + m) *\
        Matrix([[ksi[0, 0]], [ksi[1, 0]], [a[0, 0]], [a[1, 0]]])


def v(p, r):
    """ p = (p1, p2, p3); r = 0,1 """
    # 检查 r 是否在有效范围内
    if r not in [1, 2]:
        raise ValueError("Value of r should lie between 1 and 2")
    p1, p2, p3 = p
    # 根据 r 的值选择合适的 ksi 矩阵
    if r == 1:
        ksi = Matrix([[1], [0]])
    else:
        ksi = -Matrix([[0], [1]])
    # 计算 v spinor
    a = (sigma1*p1 + sigma2*p2 + sigma3*p3) / (E + m)*ksi
    if a == 0:
        a = zeros(2, 1)  # 如果 a 为零，则返回零矩阵
    # 返回 v spinor
    return sqrt(E + m) *\
        Matrix([[a[0, 0]], [a[1, 0]], [ksi[0, 0]], [ksi[1, 0]]])


def pslash(p):
    # 计算 p slash
    p1, p2, p3 = p
    p0 = sqrt(m**2 + p1**2 + p2**2 + p3**2)
    return gamma0*p0 - gamma1*p1 - gamma2*p2 - gamma3*p3


def Tr(M):
    # 返回矩阵 M 的迹
    return M.trace()


def xprint(lhs, rhs):
    # 使用 SymPy 的 pprint 函数输出等式 lhs = rhs
    pprint(Eq(sympify(lhs), rhs))


def main():
    # 定义实数符号 a, b, c
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    c = Symbol("c", real=True)

    p = (a, b, c)  # 定义动量 p

    # 断言两个 spinor 的内积为零
    assert u(p, 1).D*u(p, 2) == Matrix(1, 1, [0])
    assert u(p, 2).D*u(p, 1) == Matrix(1, 1, [0])

    # 定义实数符号 p1, p2, p3, pp1, pp2, pp3, k1, k2, k3, kp1, kp2, kp3, mu
    p1, p2, p3 = [Symbol(x, real=True) for x in ["p1", "p2", "p3"]]
    pp1, pp2, pp3 = [Symbol(x, real=True) for x in ["pp1", "pp2", "pp3"]]
    k1, k2, k3 = [Symbol(x, real=True) for x in ["k1", "k2", "k3"]]
    kp1, kp2, kp3 = [Symbol(x, real=True) for x in ["kp1", "kp2", "kp3"]]

    p = (p1, p2, p3)  # 定义动量 p
    pp = (pp1, pp2, pp3)  # 定义动量 pp
    k = (k1, k2, k3)  # 定义动量 k
    kp = (kp1, kp2, kp3)  # 定义动量 kp

    mu = Symbol("mu")  # 定义 mu 符号

    e = (pslash(p) + m*ones(4))*(pslash(k) - m*ones(4))  # 计算 e
    f = pslash(p) + m*ones(4)  # 计算 f
    g = pslash(p) - m*ones(4)  # 计算 g

    xprint('Tr(f*g)', Tr(f*g))  # 打印 Tr(f*g)

    # 计算 M 矩阵的每一项，并将它们相加
    M0 = [(v(pp, 1).D*mgamma(mu)*u(p, 1))*(u(k, 1).D*mgamma(mu, True) *
                                           v(kp, 1)) for mu in range(4)]
    M = M0[0] + M0[1] + M0[2] + M0[3]
    M = M[0]  # 取出 M 的第一个元素
    if not isinstance(M, Basic):
        raise TypeError("Invalid type of variable")  # 如果 M 的类型不正确，则抛出 TypeError
    # 定义一个符号变量 d，声明其为实数，符号表达式为 d=E+m
    d = Symbol("d", real=True)  # d=E+m

    # 打印 M 的值
    xprint('M', M)
    # 打印一行分隔符 "-" * 40
    print("-"*40)

    # 将表达式 M 中的 E 替换为 d - m，然后展开表达式并乘以 d 的平方，再次展开
    M = ((M.subs(E, d - m)).expand()*d**2).expand()
    # 打印 M2 的值
    xprint('M2', 1 / (E + m)**2*M)
    # 打印一行分隔符 "-" * 40
    print("-"*40)

    # 获取 M 的实部和虚部，并分别打印
    x, y = M.as_real_imag()
    xprint('Re(M)', x)
    xprint('Im(M)', y)

    # 计算 M 的模的平方，并打印结果
    e = x**2 + y**2
    xprint('abs(M)**2', e)
    # 打印一行分隔符 "-" * 40
    print("-"*40)

    # 展开表达式 e，并打印其展开后的结果
    xprint('Expand(abs(M)**2)', e.expand())
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 main() 函数
if __name__ == "__main__":
    main()
```