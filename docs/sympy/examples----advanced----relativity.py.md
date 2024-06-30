# `D:\src\scipysrc\sympy\examples\advanced\relativity.py`

```
#!/usr/bin/env python

"""
This example calculates the Ricci tensor from the metric and does this
on the example of Schwarzschild solution.

If you want to derive this by hand, follow the wiki page here:

https://en.wikipedia.org/wiki/Deriving_the_Schwarzschild_solution

Also read the above wiki and follow the references from there if
something is not clear, like what the Ricci tensor is, etc.

"""

from sympy import (exp, Symbol, sin, dsolve, Function,
                  Matrix, Eq, pprint, solve)


def grad(f, X):
    # Compute the gradient of function f with respect to variables in X
    a = []
    for x in X:
        a.append(f.diff(x))
    return a


def d(m, x):
    # Compute the gradient of the function m[0, 0] with respect to x
    return grad(m[0, 0], x)


class MT:
    def __init__(self, m):
        # Initialize the metric tensor with m and compute its inverse
        self.gdd = m
        self.guu = m.inv()

    def __str__(self):
        # Return a string representation of the metric tensor g_dd
        return "g_dd =\n" + str(self.gdd)

    def dd(self, i, j):
        # Return the (i, j) component of the metric tensor g_dd
        return self.gdd[i, j]

    def uu(self, i, j):
        # Return the (i, j) component of the inverse metric tensor g_uu
        return self.guu[i, j]


class G:
    def __init__(self, g, x):
        # Initialize the connection coefficients with metric g and coordinates x
        self.g = g
        self.x = x

    def udd(self, i, k, l):
        # Compute the Christoffel symbols of the second kind (connection coefficients) 
        # for indices (i, k, l) using the metric g and coordinates x
        g = self.g
        x = self.x
        r = 0
        for m in [0, 1, 2, 3]:
            r += g.uu(i, m)/2 * (g.dd(m, k).diff(x[l]) + g.dd(m, l).diff(x[k])
                    - g.dd(k, l).diff(x[m]))
        return r


class Riemann:
    def __init__(self, G, x):
        # Initialize the Riemann curvature tensor with connection coefficients G and coordinates x
        self.G = G
        self.x = x

    def uddd(self, rho, sigma, mu, nu):
        # Compute the components of the Riemann curvature tensor for indices (rho, sigma, mu, nu)
        G = self.G
        x = self.x
        r = G.udd(rho, nu, sigma).diff(x[mu]) - G.udd(rho, mu, sigma).diff(x[nu])
        for lam in [0, 1, 2, 3]:
            r += G.udd(rho, mu, lam)*G.udd(lam, nu, sigma) \
                - G.udd(rho, nu, lam)*G.udd(lam, mu, sigma)
        return r


class Ricci:
    def __init__(self, R, x):
        # Initialize the Ricci curvature tensor with Riemann tensor R, coordinates x, and metric g
        self.R = R
        self.x = x
        self.g = R.G.g

    def dd(self, mu, nu):
        # Compute the (mu, nu) component of the Ricci tensor using the Riemann tensor R
        R = self.R
        x = self.x
        r = 0
        for lam in [0, 1, 2, 3]:
            r += R.uddd(lam, mu, lam, nu)
        return r

    def ud(self, mu, nu):
        # Compute the contracted (mu, nu) component of the Ricci tensor using the metric g
        r = 0
        for lam in [0, 1, 2, 3]:
            r += self.g.uu(mu, lam)*self.dd(lam, nu)
        return r.expand()


def curvature(Rmn):
    # Compute the scalar curvature from the Ricci tensor components
    return Rmn.ud(0, 0) + Rmn.ud(1, 1) + Rmn.ud(2, 2) + Rmn.ud(3, 3)

nu = Function("nu")
lam = Function("lambda")

t = Symbol("t")
r = Symbol("r")
theta = Symbol(r"theta")
phi = Symbol(r"phi")

# general, spherically symmetric metric
gdd = Matrix((
    (-exp(nu(r)), 0, 0, 0),
    (0, exp(lam(r)), 0, 0),
    (0, 0, r**2, 0),
    (0, 0, 0, r**2*sin(theta)**2)
))
g = MT(gdd)
X = (t, r, theta, phi)
Gamma = G(g, X)
Rmn = Ricci(Riemann(Gamma, X), X)


def pprint_Gamma_udd(i, k, l):
    # Pretty-print the Christoffel symbols of the second kind
    pprint(Eq(Symbol('Gamma^%i_%i%i' % (i, k, l)), Gamma.udd(i, k, l)))


def pprint_Rmn_dd(i, j):
    # Pretty-print the components of the Ricci tensor
    pprint(Eq(Symbol('R_%i%i' % (i, j)), Rmn.dd(i, j)))


# from Differential Equations example
def eq1():
    # Solve the first differential equation related to the Ricci tensor component R_{00}
    r = Symbol("r")
    e = Rmn.dd(0, 0)
    e = e.subs(nu(r), -lam(r))
    pprint(dsolve(e, lam(r)))


def eq2():
    # Prepare to solve the second differential equation related to the Ricci tensor component R_{11}
    r = Symbol("r")
    e = Rmn.dd(1, 1)
    C = Symbol("CC")
    e = e.subs(nu(r), -lam(r))
    # 使用 pprint 模块打印调用 dsolve 函数后返回的表达式解
    pprint(dsolve(e, lam(r)))
# 定义解方程的函数eq3，求解与符号r相关的方程
def eq3():
    # 创建一个符号r
    r = Symbol("r")
    # 使用Rmn.dd(2, 2)计算张量e
    e = Rmn.dd(2, 2)
    # 将nu(r)替换为-lam(r)
    e = e.subs(nu(r), -lam(r))
    # 打印解dsolve(e, lam(r))的结果
    pprint(dsolve(e, lam(r)))

# 定义解方程的函数eq4，求解与符号r相关的方程
def eq4():
    # 创建一个符号r
    r = Symbol("r")
    # 使用Rmn.dd(3, 3)计算张量e
    e = Rmn.dd(3, 3)
    # 将nu(r)替换为-lam(r)
    e = e.subs(nu(r), -lam(r))
    # 打印使用不同方法求解dsolve(e, lam(r))的结果
    pprint(dsolve(e, lam(r)))
    pprint(dsolve(e, lam(r), 'best'))

# 主函数main，用于输出不同的量和方程解的结果
def main():
    # 输出初始度量
    print("Initial metric:")
    pprint(gdd)
    print("-"*40)
    # 输出Christoffel符号
    print("Christoffel symbols:")
    pprint_Gamma_udd(0, 1, 0)
    pprint_Gamma_udd(0, 0, 1)
    print()
    pprint_Gamma_udd(1, 0, 0)
    pprint_Gamma_udd(1, 1, 1)
    pprint_Gamma_udd(1, 2, 2)
    pprint_Gamma_udd(1, 3, 3)
    print()
    pprint_Gamma_udd(2, 2, 1)
    pprint_Gamma_udd(2, 1, 2)
    pprint_Gamma_udd(2, 3, 3)
    print()
    pprint_Gamma_udd(3, 2, 3)
    pprint_Gamma_udd(3, 3, 2)
    pprint_Gamma_udd(3, 1, 3)
    pprint_Gamma_udd(3, 3, 1)
    print("-"*40)
    # 输出Ricci张量
    print("Ricci tensor:")
    pprint_Rmn_dd(0, 0)
    # 使用Rmn.dd(1, 1)计算张量e
    e = Rmn.dd(1, 1)
    pprint_Rmn_dd(1, 1)
    pprint_Rmn_dd(2, 2)
    pprint_Rmn_dd(3, 3)
    print("-"*40)
    # 输出求解Einstein方程的结果
    print("Solve Einstein's equations:")
    # 将nu(r)替换为-lam(r)，并计算
    e = e.subs(nu(r), -lam(r)).doit()
    # 求解dsolve(e, lam(r))的结果，并打印
    l = dsolve(e, lam(r))
    pprint(l)
    # 解出lam(r)的值，并用gdd替换lam(r)，再用-nu(r)替换nu(r)，打印结果
    lamsol = solve(l, lam(r))[0]
    metric = gdd.subs(lam(r), lamsol).subs(nu(r), -lamsol)  # .combine()
    print("metric:")
    pprint(metric)

if __name__ == "__main__":
    main()
```