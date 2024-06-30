# `D:\src\scipysrc\sympy\sympy\polys\benchmarks\bench_galoispolys.py`

```
"""
Benchmarks for polynomials over Galois fields.
"""

从 sympy.polys.galoistools 导入 gf_from_dict 和 gf_factor_sqf
从 sympy.polys.domains 导入 ZZ
从 sympy.core.numbers 导入 pi
从 sympy.ntheory.generate 导入 nextprime


def gathen_poly(n, p, K):
    返回一个 Galois 域上的多项式，其系数由字典 {n: K.one, 1: K.one, 0: K.one} 给出
    return gf_from_dict({n: K.one, 1: K.one, 0: K.one}, p, K)


def shoup_poly(n, p, K):
    创建一个 Shoup 多项式的系数列表，长度为 n+1
    f = [K.one] * (n + 1)
    对于 i 从 1 到 n+1：
        使用 Shoup 的递归关系计算多项式系数
        f[i] = (f[i - 1]**2 + K.one) % p
    返回系数列表 f
    return f


def genprime(n, K):
    生成一个 n 比特长度的素数
    return K(nextprime(int((2**n * pi).evalf())))


生成一个长度为 10 的素数 p_10
生成 Galois 域上的度为 10 的 Gathen 多项式 f_10
生成一个长度为 20 的素数 p_20
生成 Galois 域上的度为 20 的 Gathen 多项式 f_20


定义函数 timeit_gathen_poly_f10_zassenhaus():
    使用 Zassenhaus 方法对 f_10 进行平方自由因式分解
    gf_factor_sqf(f_10, p_10, ZZ, method='zassenhaus')


定义函数 timeit_gathen_poly_f10_shoup():
    使用 Shoup 方法对 f_10 进行平方自由因式分解
    gf_factor_sqf(f_10, p_10, ZZ, method='shoup')


定义函数 timeit_gathen_poly_f20_zassenhaus():
    使用 Zassenhaus 方法对 f_20 进行平方自由因式分解
    gf_factor_sqf(f_20, p_20, ZZ, method='zassenhaus')


定义函数 timeit_gathen_poly_f20_shoup():
    使用 Shoup 方法对 f_20 进行平方自由因式分解
    gf_factor_sqf(f_20, p_20, ZZ, method='shoup')


生成一个长度为 8 的素数 P_08
生成 Shoup 多项式 F_10，其系数由 shoup_poly(10, P_08, ZZ) 返回
生成一个长度为 18 的素数 P_18
生成 Shoup 多项式 F_20，其系数由 shoup_poly(20, P_18, ZZ) 返回


定义函数 timeit_shoup_poly_F10_zassenhaus():
    使用 Zassenhaus 方法对 F_10 进行平方自由因式分解
    gf_factor_sqf(F_10, P_08, ZZ, method='zassenhaus')


定义函数 timeit_shoup_poly_F10_shoup():
    使用 Shoup 方法对 F_10 进行平方自由因式分解
    gf_factor_sqf(F_10, P_08, ZZ, method='shoup')


定义函数 timeit_shoup_poly_F20_zassenhaus():
    使用 Zassenhaus 方法对 F_20 进行平方自由因式分解
    gf_factor_sqf(F_20, P_18, ZZ, method='zassenhaus')


定义函数 timeit_shoup_poly_F20_shoup():
    使用 Shoup 方法对 F_20 进行平方自由因式分解
    gf_factor_sqf(F_20, P_18, ZZ, method='shoup')
```