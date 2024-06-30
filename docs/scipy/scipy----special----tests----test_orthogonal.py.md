# `D:\src\scipysrc\scipy\scipy\special\tests\test_orthogonal.py`

```
import numpy as np  # 导入 NumPy 库，通常用 np 别名表示

from numpy import array, sqrt  # 从 NumPy 中导入 array 和 sqrt 函数
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_almost_equal, assert_allclose)  # 从 NumPy testing 模块导入多个断言函数
from pytest import raises as assert_raises  # 导入 pytest 库中的 raises 函数并起别名为 assert_raises

from scipy import integrate  # 导入 SciPy 库中的 integrate 模块
import scipy.special as sc  # 导入 SciPy 库中的 special 模块并起别名为 sc
from scipy.special import gamma  # 从 SciPy special 模块中导入 gamma 函数
import scipy.special._orthogonal as orth  # 导入 SciPy special 模块中的 _orthogonal 子模块并起别名为 orth

class TestCheby:
    def test_chebyc(self):
        C0 = orth.chebyc(0)  # 计算第一类切比雪夫多项式的系数 C0
        C1 = orth.chebyc(1)  # 计算第一类切比雪夫多项式的系数 C1
        with np.errstate(all='ignore'):  # 忽略 NumPy 中的所有错误
            C2 = orth.chebyc(2)  # 计算第一类切比雪夫多项式的系数 C2
            C3 = orth.chebyc(3)  # 计算第一类切比雪夫多项式的系数 C3
            C4 = orth.chebyc(4)  # 计算第一类切比雪夫多项式的系数 C4
            C5 = orth.chebyc(5)  # 计算第一类切比雪夫多项式的系数 C5

        assert_array_almost_equal(C0.c,[2],13)  # 断言 C0 的系数
        assert_array_almost_equal(C1.c,[1,0],13)  # 断言 C1 的系数
        assert_array_almost_equal(C2.c,[1,0,-2],13)  # 断言 C2 的系数
        assert_array_almost_equal(C3.c,[1,0,-3,0],13)  # 断言 C3 的系数
        assert_array_almost_equal(C4.c,[1,0,-4,0,2],13)  # 断言 C4 的系数
        assert_array_almost_equal(C5.c,[1,0,-5,0,5,0],13)  # 断言 C5 的系数

    def test_chebys(self):
        S0 = orth.chebys(0)  # 计算第二类切比雪夫多项式的系数 S0
        S1 = orth.chebys(1)  # 计算第二类切比雪夫多项式的系数 S1
        S2 = orth.chebys(2)  # 计算第二类切比雪夫多项式的系数 S2
        S3 = orth.chebys(3)  # 计算第二类切比雪夫多项式的系数 S3
        S4 = orth.chebys(4)  # 计算第二类切比雪夫多项式的系数 S4
        S5 = orth.chebys(5)  # 计算第二类切比雪夫多项式的系数 S5
        assert_array_almost_equal(S0.c,[1],13)  # 断言 S0 的系数
        assert_array_almost_equal(S1.c,[1,0],13)  # 断言 S1 的系数
        assert_array_almost_equal(S2.c,[1,0,-1],13)  # 断言 S2 的系数
        assert_array_almost_equal(S3.c,[1,0,-2,0],13)  # 断言 S3 的系数
        assert_array_almost_equal(S4.c,[1,0,-3,0,1],13)  # 断言 S4 的系数
        assert_array_almost_equal(S5.c,[1,0,-4,0,3,0],13)  # 断言 S5 的系数

    def test_chebyt(self):
        T0 = orth.chebyt(0)  # 计算第三类切比雪夫多项式的系数 T0
        T1 = orth.chebyt(1)  # 计算第三类切比雪夫多项式的系数 T1
        T2 = orth.chebyt(2)  # 计算第三类切比雪夫多项式的系数 T2
        T3 = orth.chebyt(3)  # 计算第三类切比雪夫多项式的系数 T3
        T4 = orth.chebyt(4)  # 计算第三类切比雪夫多项式的系数 T4
        T5 = orth.chebyt(5)  # 计算第三类切比雪夫多项式的系数 T5
        assert_array_almost_equal(T0.c,[1],13)  # 断言 T0 的系数
        assert_array_almost_equal(T1.c,[1,0],13)  # 断言 T1 的系数
        assert_array_almost_equal(T2.c,[2,0,-1],13)  # 断言 T2 的系数
        assert_array_almost_equal(T3.c,[4,0,-3,0],13)  # 断言 T3 的系数
        assert_array_almost_equal(T4.c,[8,0,-8,0,1],13)  # 断言 T4 的系数
        assert_array_almost_equal(T5.c,[16,0,-20,0,5,0],13)  # 断言 T5 的系数

    def test_chebyu(self):
        U0 = orth.chebyu(0)  # 计算第四类切比雪夫多项式的系数 U0
        U1 = orth.chebyu(1)  # 计算第四类切比雪夫多项式的系数 U1
        U2 = orth.chebyu(2)  # 计算第四类切比雪夫多项式的系数 U2
        U3 = orth.chebyu(3)  # 计算第四类切比雪夫多项式的系数 U3
        U4 = orth.chebyu(4)  # 计算第四类切比雪夫多项式的系数 U4
        U5 = orth.chebyu(5)  # 计算第四类切比雪夫多项式的系数 U5
        assert_array_almost_equal(U0.c,[1],13)  # 断言 U0 的系数
        assert_array_almost_equal(U1.c,[2,0],13)  # 断言 U1 的系数
        assert_array_almost_equal(U2.c,[4,0,-1],13)  # 断言 U2 的系数
        assert_array_almost_equal(U3.c,[8,0,-4,0],13)  # 断言 U3 的系数
        assert_array_almost_equal(U4.c,[16,0,-12,0,1],13)  # 断言 U4 的系数
        assert_array_almost_equal(U5.c,[32,0,-32,0,6,0],13)  # 断言 U5 的系数

class TestGegenbauer:
    # 这里是 TestGegenbauer 类的定义，未提供其余代码，因此没有更多需要添加的注释
    # 定义一个测试函数 test_gegenbauer，用于测试 Gegenbauer 正交多项式的计算
    def test_gegenbauer(self):
        # 生成一个随机数 a，范围为 -0.5 到 4.5
        a = 5*np.random.random() - 0.5
        # 如果随机数 a 中有任何一个值为 0，则将 a 设为 -0.2
        if np.any(a == 0):
            a = -0.2
        # 计算 Gegenbauer 多项式的前几个阶数
        Ca0 = orth.gegenbauer(0,a)
        Ca1 = orth.gegenbauer(1,a)
        Ca2 = orth.gegenbauer(2,a)
        Ca3 = orth.gegenbauer(3,a)
        Ca4 = orth.gegenbauer(4,a)
        Ca5 = orth.gegenbauer(5,a)

        # 断言 Ca0 的系数数组近似等于 [1]，精度为 13 位小数
        assert_array_almost_equal(Ca0.c,array([1]),13)
        # 断言 Ca1 的系数数组近似等于 [2*a, 0]，精度为 13 位小数
        assert_array_almost_equal(Ca1.c,array([2*a,0]),13)
        # 断言 Ca2 的系数数组近似等于 [2*a*(a+1), 0, -a]，精度为 13 位小数
        assert_array_almost_equal(Ca2.c,array([2*a*(a+1),0,-a]),13)
        # 断言 Ca3 的系数数组近似等于 [4*sc.poch(a,3), 0, -6*a*(a+1), 0] / 3.0，精度为 11 位小数
        assert_array_almost_equal(Ca3.c,array([4*sc.poch(a,3),0,-6*a*(a+1),
                                               0])/3.0,11)
        # 断言 Ca4 的系数数组近似等于 [4*sc.poch(a,4), 0, -12*sc.poch(a,3), 0, 3*a*(a+1)] / 6.0，精度为 11 位小数
        assert_array_almost_equal(Ca4.c,array([4*sc.poch(a,4),0,-12*sc.poch(a,3),
                                               0,3*a*(a+1)])/6.0,11)
        # 断言 Ca5 的系数数组近似等于 [4*sc.poch(a,5), 0, -20*sc.poch(a,4), 0, 15*sc.poch(a,3), 0] / 15.0，精度为 11 位小数
        assert_array_almost_equal(Ca5.c,array([4*sc.poch(a,5),0,-20*sc.poch(a,4),
                                               0,15*sc.poch(a,3),0])/15.0,11)
class TestHermite:
    def test_hermite(self):
        # 计算 Hermite 多项式 H_n(x)，其中 n 分别为 0 到 5
        H0 = orth.hermite(0)
        H1 = orth.hermite(1)
        H2 = orth.hermite(2)
        H3 = orth.hermite(3)
        H4 = orth.hermite(4)
        H5 = orth.hermite(5)
        # 断言每个 Hermite 多项式的系数是否接近预期值，使用绝对误差限为 1e-13
        assert_array_almost_equal(H0.c,[1],13)
        assert_array_almost_equal(H1.c,[2,0],13)
        assert_array_almost_equal(H2.c,[4,0,-2],13)
        assert_array_almost_equal(H3.c,[8,0,-12,0],13)
        assert_array_almost_equal(H4.c,[16,0,-48,0,12],12)
        assert_array_almost_equal(H5.c,[32,0,-160,0,120,0],12)

    def test_hermitenorm(self):
        # 计算归一化 Hermite 多项式 He_n(x) = 2**(-n/2) * H_n(x/sqrt(2))
        psub = np.poly1d([1.0/sqrt(2),0])
        H0 = orth.hermitenorm(0)
        H1 = orth.hermitenorm(1)
        H2 = orth.hermitenorm(2)
        H3 = orth.hermitenorm(3)
        H4 = orth.hermitenorm(4)
        H5 = orth.hermitenorm(5)
        # 计算归一化 Hermite 多项式的期望值
        he0 = orth.hermite(0)(psub)
        he1 = orth.hermite(1)(psub) / sqrt(2)
        he2 = orth.hermite(2)(psub) / 2.0
        he3 = orth.hermite(3)(psub) / (2*sqrt(2))
        he4 = orth.hermite(4)(psub) / 4.0
        he5 = orth.hermite(5)(psub) / (4.0*sqrt(2))
        # 断言归一化 Hermite 多项式的系数是否接近期望值，使用绝对误差限为 1e-13
        assert_array_almost_equal(H0.c,he0.c,13)
        assert_array_almost_equal(H1.c,he1.c,13)
        assert_array_almost_equal(H2.c,he2.c,13)
        assert_array_almost_equal(H3.c,he3.c,13)
        assert_array_almost_equal(H4.c,he4.c,13)
        assert_array_almost_equal(H5.c,he5.c,13)


class TestShLegendre:
    def test_sh_legendre(self):
        # 计算球谐函数的 Legendre 形式 P*_n(x) = P_n(2x-1)
        psub = np.poly1d([2,-1])
        Ps0 = orth.sh_legendre(0)
        Ps1 = orth.sh_legendre(1)
        Ps2 = orth.sh_legendre(2)
        Ps3 = orth.sh_legendre(3)
        Ps4 = orth.sh_legendre(4)
        Ps5 = orth.sh_legendre(5)
        # 计算球谐函数的 Legendre 形式的期望值
        pse0 = orth.legendre(0)(psub)
        pse1 = orth.legendre(1)(psub)
        pse2 = orth.legendre(2)(psub)
        pse3 = orth.legendre(3)(psub)
        pse4 = orth.legendre(4)(psub)
        pse5 = orth.legendre(5)(psub)
        # 断言球谐函数的 Legendre 形式的系数是否接近期望值，使用绝对误差限为 1e-13 或 1e-12
        assert_array_almost_equal(Ps0.c,pse0.c,13)
        assert_array_almost_equal(Ps1.c,pse1.c,13)
        assert_array_almost_equal(Ps2.c,pse2.c,13)
        assert_array_almost_equal(Ps3.c,pse3.c,13)
        assert_array_almost_equal(Ps4.c,pse4.c,12)
        assert_array_almost_equal(Ps5.c,pse5.c,12)


class TestShChebyt:
    # 定义一个测试方法，用于验证正交多项式的 Chebyshev T 函数的计算结果
    def test_sh_chebyt(self):
        # 定义变量 psub，表示用于生成 Chebyshev T 函数的多项式的系数
        psub = np.poly1d([2,-1])
        # 计算 Chebyshev T 函数的零阶到五阶结果，存储在 Ts0 到 Ts5 中
        Ts0 = orth.sh_chebyt(0)
        Ts1 = orth.sh_chebyt(1)
        Ts2 = orth.sh_chebyt(2)
        Ts3 = orth.sh_chebyt(3)
        Ts4 = orth.sh_chebyt(4)
        Ts5 = orth.sh_chebyt(5)
        # 计算通过多项式 psub 生成的 Chebyshev T 函数的零阶到五阶结果，存储在 tse0 到 tse5 中
        tse0 = orth.chebyt(0)(psub)
        tse1 = orth.chebyt(1)(psub)
        tse2 = orth.chebyt(2)(psub)
        tse3 = orth.chebyt(3)(psub)
        tse4 = orth.chebyt(4)(psub)
        tse5 = orth.chebyt(5)(psub)
        # 使用 assert_array_almost_equal 方法断言 Ts0 到 Ts5 的系数与 tse0 到 tse5 的系数近似相等，
        # 允许最大的绝对误差为 13 或 12（对应不同的阶数）
        assert_array_almost_equal(Ts0.c, tse0.c, 13)
        assert_array_almost_equal(Ts1.c, tse1.c, 13)
        assert_array_almost_equal(Ts2.c, tse2.c, 13)
        assert_array_almost_equal(Ts3.c, tse3.c, 13)
        assert_array_almost_equal(Ts4.c, tse4.c, 12)
        assert_array_almost_equal(Ts5.c, tse5.c, 12)
class TestShChebyu:
    def test_sh_chebyu(self):
        # U*_n(x) = U_n(2x-1)
        # 创建一个一元多项式对象，代表多项式 2*x - 1
        psub = np.poly1d([2,-1])
        # 计算 Chebyshev-U 多项式 U_n(2x-1)，其中 n=0 到 n=5
        Us0 = orth.sh_chebyu(0)
        Us1 = orth.sh_chebyu(1)
        Us2 = orth.sh_chebyu(2)
        Us3 = orth.sh_chebyu(3)
        Us4 = orth.sh_chebyu(4)
        Us5 = orth.sh_chebyu(5)
        # 计算 Chebyshev 多项式 U_n(psub)，其中 n=0 到 n=5
        use0 = orth.chebyu(0)(psub)
        use1 = orth.chebyu(1)(psub)
        use2 = orth.chebyu(2)(psub)
        use3 = orth.chebyu(3)(psub)
        use4 = orth.chebyu(4)(psub)
        use5 = orth.chebyu(5)(psub)
        # 断言 Chebyshev-U 多项式与 Chebyshev 多项式的系数数组几乎相等，精度为13位小数
        assert_array_almost_equal(Us0.c, use0.c, 13)
        assert_array_almost_equal(Us1.c, use1.c, 13)
        assert_array_almost_equal(Us2.c, use2.c, 13)
        assert_array_almost_equal(Us3.c, use3.c, 13)
        assert_array_almost_equal(Us4.c, use4.c, 12)
        assert_array_almost_equal(Us5.c, use5.c, 11)


class TestShJacobi:
    def test_sh_jacobi(self):
        # G^(p,q)_n(x) = n! gamma(n+p)/gamma(2*n+p) * P^(p-q,q-1)_n(2*x-1)
        # 定义一个函数，计算 Jacobi 多项式的系数修正系数
        def conv(n, p):
            return gamma(n + 1) * gamma(n + p) / gamma(2 * n + p)
        # 创建一个一元多项式对象，代表多项式 2*x - 1
        psub = np.poly1d([2,-1])
        # 随机生成 p 和 q 的值
        q = 4 * np.random.random()
        p = q - 1 + 2 * np.random.random()
        # 计算 shifted Jacobi 多项式 G^(p,q)_n(x)，其中 n=0 到 n=5
        G0 = orth.sh_jacobi(0, p, q)
        G1 = orth.sh_jacobi(1, p, q)
        G2 = orth.sh_jacobi(2, p, q)
        G3 = orth.sh_jacobi(3, p, q)
        G4 = orth.sh_jacobi(4, p, q)
        G5 = orth.sh_jacobi(5, p, q)
        # 计算 Jacobi 多项式 P^(p-q,q-1)_n(psub) * conv(n,p)，其中 n=0 到 n=5
        ge0 = orth.jacobi(0, p - q, q - 1)(psub) * conv(0, p)
        ge1 = orth.jacobi(1, p - q, q - 1)(psub) * conv(1, p)
        ge2 = orth.jacobi(2, p - q, q - 1)(psub) * conv(2, p)
        ge3 = orth.jacobi(3, p - q, q - 1)(psub) * conv(3, p)
        ge4 = orth.jacobi(4, p - q, q - 1)(psub) * conv(4, p)
        ge5 = orth.jacobi(5, p - q, q - 1)(psub) * conv(5, p)
        # 断言 shifted Jacobi 多项式与计算得到的值的系数数组几乎相等，精度为13位小数
        assert_array_almost_equal(G0.c, ge0.c, 13)
        assert_array_almost_equal(G1.c, ge1.c, 13)
        assert_array_almost_equal(G2.c, ge2.c, 13)
        assert_array_almost_equal(G3.c, ge3.c, 13)
        assert_array_almost_equal(G4.c, ge4.c, 13)
        assert_array_almost_equal(G5.c, ge5.c, 13)
    # 定义测试方法 `test_call`
    def test_call(self):
        # 初始化空列表 `poly`
        poly = []
        # 循环迭代范围为 0 到 4
        for n in range(5):
            # 将多行字符串按换行符分割成列表，然后去除每行两侧的空格并添加到 `poly` 列表中
            poly.extend([x.strip() for x in
                ("""
                orth.jacobi(%(n)d,0.3,0.9)
                orth.sh_jacobi(%(n)d,0.3,0.9)
                orth.genlaguerre(%(n)d,0.3)
                orth.laguerre(%(n)d)
                orth.hermite(%(n)d)
                orth.hermitenorm(%(n)d)
                orth.gegenbauer(%(n)d,0.3)
                orth.chebyt(%(n)d)
                orth.chebyu(%(n)d)
                orth.chebyc(%(n)d)
                orth.chebys(%(n)d)
                orth.sh_chebyt(%(n)d)
                orth.sh_chebyu(%(n)d)
                orth.legendre(%(n)d)
                orth.sh_legendre(%(n)d)
                """ % dict(n=n)).split()
            ])
        # 设置 numpy 的错误状态为忽略所有错误
        with np.errstate(all='ignore'):
            # 遍历 `poly` 列表中的每个字符串 `pstr`
            for pstr in poly:
                # 通过 `eval` 执行字符串表示的函数调用，将结果赋给变量 `p`
                p = eval(pstr)
                # 使用 `assert_almost_equal` 断言，检查 p(0.315) 是否接近于 np.poly1d(p.coef)(0.315)
                assert_almost_equal(p(0.315), np.poly1d(p.coef)(0.315),
                                    err_msg=pstr)
class TestGenlaguerre:
    # 定义一个测试类 TestGenlaguerre

    def test_regression(self):
        # 测试函数，验证 genlaguerre 函数的回归结果是否正确
        assert_equal(orth.genlaguerre(1, 1, monic=False)(0), 2.)
        # 断言：非单一形式的 genlaguerre 函数在 x=0 处的返回值应为 2.0
        assert_equal(orth.genlaguerre(1, 1, monic=True)(0), -2.)
        # 断言：单一形式的 genlaguerre 函数在 x=0 处的返回值应为 -2.0
        assert_equal(orth.genlaguerre(1, 1, monic=False), np.poly1d([-1, 2]))
        # 断言：非单一形式的 genlaguerre 函数应与给定的多项式对象相等
        assert_equal(orth.genlaguerre(1, 1, monic=True), np.poly1d([1, -2]))
        # 断言：单一形式的 genlaguerre 函数应与给定的多项式对象相等


def verify_gauss_quad(root_func, eval_func, weight_func, a, b, N,
                      rtol=1e-15, atol=5e-14):
    # 验证 Gauss-Quadrature 的函数
    # 这个测试从 numpy 的 test_hermite.py 中复制而来

    x, w, mu = root_func(N, True)
    # 使用 root_func 获取 Gauss-Jacobi 积分的根 x, 权重 w, 和 μ 值

    n = np.arange(N, dtype=np.dtype("long"))
    # 创建一个长度为 N 的整数数组 n

    v = eval_func(n[:,np.newaxis], x)
    # 使用 eval_func 评估 Jacobi 多项式在点集 x 上的值

    vv = np.dot(v*w, v.T)
    # 计算 v * w * v^T

    vd = 1 / np.sqrt(vv.diagonal())
    # 对角线元素的倒数的平方根

    vv = vd[:, np.newaxis] * vv * vd
    # 对称正定矩阵 vv

    assert_allclose(vv, np.eye(N), rtol, atol)
    # 断言：vv 应该接近单位矩阵

    # 检查积分结果是否正确
    assert_allclose(w.sum(), mu, rtol, atol)

    # 使用 quad 比较积分结果的一致性
    def f(x):
        return x ** 3 - 3 * x ** 2 + x - 2
    # 定义函数 f(x) = x^3 - 3x^2 + x - 2

    resI = integrate.quad(lambda x: f(x)*weight_func(x), a, b)
    # 使用 integrate.quad 计算函数 f(x) * weight_func(x) 在区间 [a, b] 上的积分

    resG = np.vdot(f(x), w)
    # 计算 f(x) 和权重 w 的内积

    rtol = 1e-6 if 1e-6 < resI[1] else resI[1] * 10
    # 相对误差的计算

    assert_allclose(resI[0], resG, rtol=rtol)
    # 断言：积分结果应该与内积结果接近


def test_roots_jacobi():
    # 测试 Jacobi 多项式的根

    def rf(a, b):
        return lambda n, mu: sc.roots_jacobi(n, a, b, mu)
    # 返回一个函数，用于计算 Jacobi 多项式的根

    def ef(a, b):
        return lambda n, x: sc.eval_jacobi(n, a, b, x)
    # 返回一个函数，用于评估 Jacobi 多项式在给定点集 x 上的值

    def wf(a, b):
        return lambda x: (1 - x) ** a * (1 + x) ** b
    # 返回一个函数，用于计算 Jacobi 多项式的权重函数

    vgq = verify_gauss_quad

    vgq(rf(-0.5, -0.75), ef(-0.5, -0.75), wf(-0.5, -0.75), -1., 1., 5)
    # 验证 Gauss-Quadrature 在 Jacobi 多项式的根、评估函数和权重函数上的结果

    vgq(rf(-0.5, -0.75), ef(-0.5, -0.75), wf(-0.5, -0.75), -1., 1.,
        25, atol=1e-12)
    # 使用更高的精度验证 Gauss-Quadrature 的结果

    vgq(rf(-0.5, -0.75), ef(-0.5, -0.75), wf(-0.5, -0.75), -1., 1.,
        100, atol=1e-11)
    # 使用更高的精度验证 Gauss-Quadrature 的结果

    # 更多的测试用例，验证不同参数组合下的 Gauss-Quadrature 结果
    # 使用 Gauss-Jacobi 方法进行数值积分，计算积分区间 [-1, 1] 上的积分，使用不同的节点数和绝对误差容限进行计算
    vgq(rf(1., 658.), ef(1., 658.), wf(1., 658.), -1., 1., 25, atol=1e-12)
    vgq(rf(1., 658.), ef(1., 658.), wf(1., 658.), -1., 1., 100, atol=1e-11)
    vgq(rf(1., 658.), ef(1., 658.), wf(1., 658.), -1., 1., 250, atol=1e-11)

    # 使用 Gauss-Jacobi 方法进行数值积分，计算积分区间 [-1, 1] 上的积分，使用不同的节点数和绝对误差容限进行计算
    vgq(rf(511., 511.), ef(511., 511.), wf(511., 511.), -1., 1., 5,
        atol=1e-12)
    vgq(rf(511., 511.), ef(511., 511.), wf(511., 511.), -1., 1., 25,
        atol=1e-11)
    vgq(rf(511., 511.), ef(511., 511.), wf(511., 511.), -1., 1., 100,
        atol=1e-10)

    # 使用 Gauss-Jacobi 方法进行数值积分，计算积分区间 [-1, 1] 上的积分，使用不同的节点数和绝对误差容限进行计算
    vgq(rf(511., 512.), ef(511., 512.), wf(511., 512.), -1., 1., 5,
        atol=1e-12)
    vgq(rf(511., 512.), ef(511., 512.), wf(511., 512.), -1., 1., 25,
        atol=1e-11)
    vgq(rf(511., 512.), ef(511., 512.), wf(511., 512.), -1., 1., 100,
        atol=1e-10)

    # 使用 Gauss-Jacobi 方法进行数值积分，计算积分区间 [-1, 1] 上的积分，使用不同的节点数和绝对误差容限进行计算
    vgq(rf(1000., 500.), ef(1000., 500.), wf(1000., 500.), -1., 1., 5,
        atol=1e-12)
    vgq(rf(1000., 500.), ef(1000., 500.), wf(1000., 500.), -1., 1., 25,
        atol=1e-11)
    vgq(rf(1000., 500.), ef(1000., 500.), wf(1000., 500.), -1., 1., 100,
        atol=1e-10)

    # 使用 Gauss-Jacobi 方法进行数值积分，计算积分区间 [-1, 1] 上的积分，使用不同的节点数和绝对误差容限进行计算
    vgq(rf(2.25, 68.9), ef(2.25, 68.9), wf(2.25, 68.9), -1., 1., 5)
    vgq(rf(2.25, 68.9), ef(2.25, 68.9), wf(2.25, 68.9), -1., 1., 25,
        atol=1e-13)
    vgq(rf(2.25, 68.9), ef(2.25, 68.9), wf(2.25, 68.9), -1., 1., 100,
        atol=1e-13)

    # 计算 Jacobi 多项式的根和权重，以及 Legendre 多项式的根和权重，并确保它们在给定的容差下相等
    xj, wj = sc.roots_jacobi(6, 0.0, 0.0)
    xl, wl = sc.roots_legendre(6)
    assert_allclose(xj, xl, 1e-14, 1e-14)
    assert_allclose(wj, wl, 1e-14, 1e-14)

    # 计算 Jacobi 多项式的根和权重，以及 Gegenbauer 多项式的根和权重，并确保它们在给定的容差下相等
    xj, wj = sc.roots_jacobi(6, 4.0, 4.0)
    xc, wc = sc.roots_gegenbauer(6, 4.5)
    assert_allclose(xj, xc, 1e-14, 1e-14)
    assert_allclose(wj, wc, 1e-14, 1e-14)

    # 计算 Jacobi 多项式的根和权重，同时将结果分别存储在 x, w 中，并确保它们在给定的容差下相等
    x, w = sc.roots_jacobi(5, 2, 3, False)
    y, v, m = sc.roots_jacobi(5, 2, 3, True)
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)

    # 使用 Gauss-Jacobi 方法进行数值积分，计算给定权重函数 wf(2,3) 在积分区间 [-1, 1] 上的积分，返回积分值和误差
    muI, muI_err = integrate.quad(wf(2,3), -1, 1)
    assert_allclose(m, muI, rtol=muI_err)

    # 确保在特定参数条件下，调用 roots_jacobi 函数会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_jacobi, 0, 1, 1)
    assert_raises(ValueError, sc.roots_jacobi, 3.3, 1, 1)
    assert_raises(ValueError, sc.roots_jacobi, 3, -2, 1)
    assert_raises(ValueError, sc.roots_jacobi, 3, 1, -2)
    assert_raises(ValueError, sc.roots_jacobi, 3, -2, -2)
def test_roots_sh_jacobi():
    # 定义内部函数 rf，返回带有固定 a, b 参数的 roots_sh_jacobi 函数
    def rf(a, b):
        return lambda n, mu: sc.roots_sh_jacobi(n, a, b, mu)
    
    # 定义内部函数 ef，返回带有固定 a, b 参数的 eval_sh_jacobi 函数
    def ef(a, b):
        return lambda n, x: sc.eval_sh_jacobi(n, a, b, x)
    
    # 定义内部函数 wf，返回带有固定 a, b 参数的权重函数
    def wf(a, b):
        return lambda x: (1.0 - x) ** (a - b) * x ** (b - 1.0)

    # 验证高斯积分法的函数对象
    vgq = verify_gauss_quad
    
    # 对不同参数组合调用高斯积分验证函数
    vgq(rf(-0.5, 0.25), ef(-0.5, 0.25), wf(-0.5, 0.25), 0., 1., 5)
    vgq(rf(-0.5, 0.25), ef(-0.5, 0.25), wf(-0.5, 0.25), 0., 1., 25, atol=1e-12)
    vgq(rf(-0.5, 0.25), ef(-0.5, 0.25), wf(-0.5, 0.25), 0., 1., 100, atol=1e-11)

    vgq(rf(0.5, 0.5), ef(0.5, 0.5), wf(0.5, 0.5), 0., 1., 5)
    vgq(rf(0.5, 0.5), ef(0.5, 0.5), wf(0.5, 0.5), 0., 1., 25, atol=1e-13)
    vgq(rf(0.5, 0.5), ef(0.5, 0.5), wf(0.5, 0.5), 0., 1., 100, atol=1e-12)

    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), 0., 1., 5)
    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), 0., 1., 25, atol=1.5e-13)
    vgq(rf(1, 0.5), ef(1, 0.5), wf(1, 0.5), 0., 1., 100, atol=2e-12)

    vgq(rf(2, 0.9), ef(2, 0.9), wf(2, 0.9), 0., 1., 5)
    vgq(rf(2, 0.9), ef(2, 0.9), wf(2, 0.9), 0., 1., 25, atol=1e-13)
    vgq(rf(2, 0.9), ef(2, 0.9), wf(2, 0.9), 0., 1., 100, atol=1e-12)

    vgq(rf(27.3, 18.24), ef(27.3, 18.24), wf(27.3, 18.24), 0., 1., 5)
    vgq(rf(27.3, 18.24), ef(27.3, 18.24), wf(27.3, 18.24), 0., 1., 25)
    vgq(rf(27.3, 18.24), ef(27.3, 18.24), wf(27.3, 18.24), 0., 1., 100, atol=1e-13)

    vgq(rf(47.1, 0.2), ef(47.1, 0.2), wf(47.1, 0.2), 0., 1., 5, atol=1e-12)
    vgq(rf(47.1, 0.2), ef(47.1, 0.2), wf(47.1, 0.2), 0., 1., 25, atol=1e-11)
    vgq(rf(47.1, 0.2), ef(47.1, 0.2), wf(47.1, 0.2), 0., 1., 100, atol=1e-10)

    vgq(rf(68.9, 2.25), ef(68.9, 2.25), wf(68.9, 2.25), 0., 1., 5, atol=3.5e-14)
    vgq(rf(68.9, 2.25), ef(68.9, 2.25), wf(68.9, 2.25), 0., 1., 25, atol=2e-13)
    vgq(rf(68.9, 2.25), ef(68.9, 2.25), wf(68.9, 2.25), 0., 1., 100, atol=1e-12)

    # 获取 roots_sh_jacobi 的返回值 x, w
    x, w = sc.roots_sh_jacobi(5, 3, 2, False)
    # 获取 roots_sh_jacobi 的返回值 y, v, m
    y, v, m = sc.roots_sh_jacobi(5, 3, 2, True)
    # 检验 x, y 和 w, v 的接近程度
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)

    # 计算权重函数在 [0, 1] 上的积分值 muI
    muI, muI_err = integrate.quad(wf(3,2), 0, 1)
    # 检验 m 和 muI 的接近程度，使用相对误差 muI_err
    assert_allclose(m, muI, rtol=muI_err)

    # 断言错误情况：传入无效参数时会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_sh_jacobi, 0, 1, 1)
    assert_raises(ValueError, sc.roots_sh_jacobi, 3.3, 1, 1)
    assert_raises(ValueError, sc.roots_sh_jacobi, 3, 1, 2)    # p - q <= -1
    assert_raises(ValueError, sc.roots_sh_jacobi, 3, 2, -1)   # q <= 0
    assert_raises(ValueError, sc.roots_sh_jacobi, 3, -2, -1)  # both

def test_roots_hermite():
    # 获取 roots_hermite 函数的引用
    rootf = sc.roots_hermite
    # 获取 eval_hermite 函数的引用
    evalf = sc.eval_hermite
    # 获取 Hermite 多项式的权重函数
    weightf = orth.hermite(5).weight_func

    # 验证高斯积分法的函数对象
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 5)
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 25, atol=1e-13)
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 100, atol=1e-12)

    # Golub-Welsch 分支
    # 获取 Golub-Welsch 分支的返回值 x, w
    x, w = sc.roots_hermite(5, False)
    # 获取 Golub-Welsch 分支的返回值 y, v, m
    y, v, m = sc.roots_hermite(5, True)
    # 检验 x, y 和 w, v 的接近程度
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)
    # 使用积分函数 `quad` 计算权重函数 `weightf` 在整个实数轴上的积分值 `muI` 和误差 `muI_err`
    muI, muI_err = integrate.quad(weightf, -np.inf, np.inf)
    
    # 断言检查计算得到的 `muI` 是否与预期的 `m` 值非常接近，允许的相对误差为 `muI_err`
    assert_allclose(m, muI, rtol=muI_err)
    
    # 在渐近分支（当 n >= 150 时切换）
    # 使用 `roots_hermite` 函数计算 Hermite 多项式的根 `x` 和权重 `w`，`200` 表示计算的阶数，`False` 表示不返回多项式值
    x, w = sc.roots_hermite(200, False)
    # 再次使用 `roots_hermite` 函数计算 Hermite 多项式的根 `y`、权重 `v` 和规范化因子 `m`，`True` 表示返回规范化因子
    y, v, m = sc.roots_hermite(200, True)
    
    # 断言检查计算得到的 Hermite 多项式根 `x` 与 `y` 是否非常接近，允许的相对误差为 `1e-14`
    assert_allclose(x, y, 1e-14, 1e-14)
    # 断言检查计算得到的 Hermite 多项式权重 `w` 与 `v` 是否非常接近，允许的相对误差为 `1e-14`
    assert_allclose(w, v, 1e-14, 1e-14)
    # 断言检查计算得到的 Hermite 多项式权重之和 `sum(v)` 是否与规范化因子 `m` 非常接近，允许的相对误差为 `1e-14`
    assert_allclose(sum(v), m, 1e-14, 1e-14)
    
    # 断言检查当输入参数为非法值时，`roots_hermite` 函数是否会引发 `ValueError` 异常
    assert_raises(ValueError, sc.roots_hermite, 0)
    assert_raises(ValueError, sc.roots_hermite, 3.3)
def test_roots_hermite_asy():
    # 定义一个函数，用于计算 Hermite 函数的递归序列
    def hermite_recursion(n, nodes):
        # 创建一个 n x nodes.size 的零矩阵
        H = np.zeros((n, nodes.size))
        # 初始化第一行 Hermite 函数的值
        H[0,:] = np.pi**(-0.25) * np.exp(-0.5*nodes**2)
        # 如果 n 大于 1，继续计算后续的 Hermite 函数值
        if n > 1:
            # 计算第二行 Hermite 函数的值
            H[1,:] = sqrt(2.0) * nodes * H[0,:]
            # 使用递推公式计算剩余的 Hermite 函数值
            for k in range(2, n):
                H[k,:] = sqrt(2.0/k) * nodes * H[k-1,:] - sqrt((k-1.0)/k) * H[k-2,:]
        return H

    # 测试 Hermite 函数的节点
    def test(N, rtol=1e-15, atol=1e-14):
        # 调用 Hermite 函数的节点计算函数
        x, w = orth._roots_hermite_asy(N)
        # 计算 Hermite 递归序列
        H = hermite_recursion(N+1, x)
        # 断言 Hermite 递归序列的最后一行应该全为零
        assert_allclose(H[-1,:], np.zeros(N), rtol, atol)
        # 断言权重和等于根号下的 π
        assert_allclose(sum(w), sqrt(np.pi), rtol, atol)

    # 使用不同的 N 值进行测试
    test(150, atol=1e-12)
    test(151, atol=1e-12)
    test(300, atol=1e-12)
    test(301, atol=1e-12)
    test(500, atol=1e-12)
    test(501, atol=1e-12)
    test(999, atol=1e-12)
    test(1000, atol=1e-12)
    test(2000, atol=1e-12)
    test(5000, atol=1e-12)

def test_roots_hermitenorm():
    # 定义使用正规 Hermite 多项式的函数
    rootf = sc.roots_hermitenorm
    evalf = sc.eval_hermitenorm
    weightf = orth.hermitenorm(5).weight_func

    # 验证高斯积分法的正确性
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 5)
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 25, atol=1e-13)
    verify_gauss_quad(rootf, evalf, weightf, -np.inf, np.inf, 100, atol=1e-12)

    # 获取正规 Hermite 多项式的根和权重
    x, w = sc.roots_hermitenorm(5, False)
    y, v, m = sc.roots_hermitenorm(5, True)
    # 断言计算出的根和权重应该非常接近
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)

    # 计算 Hermite 多项式的积分与理论值比较
    muI, muI_err = integrate.quad(weightf, -np.inf, np.inf)
    assert_allclose(m, muI, rtol=muI_err)

    # 断言对于无效的参数，应该引发 ValueError
    assert_raises(ValueError, sc.roots_hermitenorm, 0)
    assert_raises(ValueError, sc.roots_hermitenorm, 3.3)

def test_roots_gegenbauer():
    # 定义 Gegenbauer 多项式的根、值、权重函数
    def rootf(a):
        return lambda n, mu: sc.roots_gegenbauer(n, a, mu)
    def evalf(a):
        return lambda n, x: sc.eval_gegenbauer(n, a, x)
    def weightf(a):
        return lambda x: (1 - x ** 2) ** (a - 0.5)

    # 验证高斯积分法的正确性
    vgq = verify_gauss_quad
    vgq(rootf(-0.25), evalf(-0.25), weightf(-0.25), -1., 1., 5)
    vgq(rootf(-0.25), evalf(-0.25), weightf(-0.25), -1., 1., 25, atol=1e-12)
    vgq(rootf(-0.25), evalf(-0.25), weightf(-0.25), -1., 1., 100, atol=1e-11)

    vgq(rootf(0.1), evalf(0.1), weightf(0.1), -1., 1., 5)
    vgq(rootf(0.1), evalf(0.1), weightf(0.1), -1., 1., 25, atol=1e-13)
    vgq(rootf(0.1), evalf(0.1), weightf(0.1), -1., 1., 100, atol=1e-12)

    vgq(rootf(1), evalf(1), weightf(1), -1., 1., 5)
    vgq(rootf(1), evalf(1), weightf(1), -1., 1., 25, atol=1e-13)
    vgq(rootf(1), evalf(1), weightf(1), -1., 1., 100, atol=1e-12)

    vgq(rootf(10), evalf(10), weightf(10), -1., 1., 5)
    vgq(rootf(10), evalf(10), weightf(10), -1., 1., 25, atol=1e-13)
    vgq(rootf(10), evalf(10), weightf(10), -1., 1., 100, atol=1e-12)

    vgq(rootf(50), evalf(50), weightf(50), -1., 1., 5, atol=1e-13)
    vgq(rootf(50), evalf(50), weightf(50), -1., 1., 25, atol=1e-12)
    vgq(rootf(50), evalf(50), weightf(50), -1., 1., 100, atol=1e-11)
    # 使用 vgq 函数进行数值积分，计算 Gegenbauer 多项式的 Gauss 积分
    # 对 alpha=170 的 Gegenbauer 多项式进行积分近似，使用不同的积分节点数和绝对误差容限
    vgq(rootf(170), evalf(170), weightf(170), -1., 1., 5, atol=1e-13)
    vgq(rootf(170), evalf(170), weightf(170), -1., 1., 25, atol=1e-12)
    vgq(rootf(170), evalf(170), weightf(170), -1., 1., 100, atol=1e-11)
    
    # 对 alpha=170.5 的 Gegenbauer 多项式进行积分近似，使用不同的积分节点数和绝对误差容限
    vgq(rootf(170.5), evalf(170.5), weightf(170.5), -1., 1., 5, atol=1.25e-13)
    vgq(rootf(170.5), evalf(170.5), weightf(170.5), -1., 1., 25, atol=1e-12)
    vgq(rootf(170.5), evalf(170.5), weightf(170.5), -1., 1., 100, atol=1e-11)

    # 测试大 alpha 值可能导致的失败，如溢出
    vgq(rootf(238), evalf(238), weightf(238), -1., 1., 5, atol=1e-13)
    vgq(rootf(238), evalf(238), weightf(238), -1., 1., 25, atol=1e-12)
    vgq(rootf(238), evalf(238), weightf(238), -1., 1., 100, atol=1e-11)
    vgq(rootf(512.5), evalf(512.5), weightf(512.5), -1., 1., 5, atol=1e-12)
    vgq(rootf(512.5), evalf(512.5), weightf(512.5), -1., 1., 25, atol=1e-11)
    vgq(rootf(512.5), evalf(512.5), weightf(512.5), -1., 1., 100, atol=1e-10)

    # 对 alpha=0 的特殊情况进行测试
    # 当 alpha=0 时，Gegenbauer 多项式均为常数 0，但在这里会变成 T_n(x) 的缩放版本
    vgq(rootf(0), sc.eval_chebyt, weightf(0), -1., 1., 5)
    vgq(rootf(0), sc.eval_chebyt, weightf(0), -1., 1., 25)
    vgq(rootf(0), sc.eval_chebyt, weightf(0), -1., 1., 100, atol=1e-12)

    # 计算 Gegenbauer 多项式的根和权重，进行比较和断言检验
    x, w = sc.roots_gegenbauer(5, 2, False)
    y, v, m = sc.roots_gegenbauer(5, 2, True)
    assert_allclose(x, y, 1e-14, 1e-14)  # 检验根的近似性
    assert_allclose(w, v, 1e-14, 1e-14)  # 检验权重的近似性

    # 使用积分函数计算权重函数在区间 [-1, 1] 上的积分，与根的积分进行比较
    muI, muI_err = integrate.quad(weightf(2), -1, 1)
    assert_allclose(m, muI, rtol=muI_err)  # 检验积分结果的近似性

    # 测试在非法输入情况下是否引发 ValueError 异常
    assert_raises(ValueError, sc.roots_gegenbauer, 0, 2)
    assert_raises(ValueError, sc.roots_gegenbauer, 3.3, 2)
    assert_raises(ValueError, sc.roots_gegenbauer, 3, -.75)
def test_roots_chebyt():
    # 获取 Chebyshev T 多项式的权重函数
    weightf = orth.chebyt(5).weight_func
    # 验证 roots_chebyt 函数计算的 Gauss 积分精度
    verify_gauss_quad(sc.roots_chebyt, sc.eval_chebyt, weightf, -1., 1., 5)
    verify_gauss_quad(sc.roots_chebyt, sc.eval_chebyt, weightf, -1., 1., 25)
    verify_gauss_quad(sc.roots_chebyt, sc.eval_chebyt, weightf, -1., 1., 100,
                      atol=1e-12)

    # 获取非正交化和正交化的 Chebyshev T 多项式的根和权重
    x, w = sc.roots_chebyt(5, False)
    y, v, m = sc.roots_chebyt(5, True)
    # 断言非正交化和正交化的根和权重相等
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)

    # 计算权重函数在 [-1, 1] 上的积分以及误差
    muI, muI_err = integrate.quad(weightf, -1, 1)
    # 断言计算得到的积分值与预期的正交化根 m 值相等，误差在可接受范围内
    assert_allclose(m, muI, rtol=muI_err)

    # 断言 roots_chebyt 函数对于无效参数会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_chebyt, 0)
    assert_raises(ValueError, sc.roots_chebyt, 3.3)

def test_chebyt_symmetry():
    # 计算 Chebyshev T 多项式的根和权重
    x, w = sc.roots_chebyt(21)
    # 分割根以获得正负部分
    pos, neg = x[:10], x[11:]
    # 断言负部分是正部分的逆序
    assert_equal(neg, -pos[::-1])
    # 断言中间值为零
    assert_equal(x[10], 0)

def test_roots_chebyu():
    # 获取 Chebyshev U 多项式的权重函数
    weightf = orth.chebyu(5).weight_func
    # 验证 roots_chebyu 函数计算的 Gauss 积分精度
    verify_gauss_quad(sc.roots_chebyu, sc.eval_chebyu, weightf, -1., 1., 5)
    verify_gauss_quad(sc.roots_chebyu, sc.eval_chebyu, weightf, -1., 1., 25)
    verify_gauss_quad(sc.roots_chebyu, sc.eval_chebyu, weightf, -1., 1., 100)

    # 获取非正交化和正交化的 Chebyshev U 多项式的根和权重
    x, w = sc.roots_chebyu(5, False)
    y, v, m = sc.roots_chebyu(5, True)
    # 断言非正交化和正交化的根和权重相等
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)

    # 计算权重函数在 [-1, 1] 上的积分以及误差
    muI, muI_err = integrate.quad(weightf, -1, 1)
    # 断言计算得到的积分值与预期的正交化根 m 值相等，误差在可接受范围内
    assert_allclose(m, muI, rtol=muI_err)

    # 断言 roots_chebyu 函数对于无效参数会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_chebyu, 0)
    assert_raises(ValueError, sc.roots_chebyu, 3.3)

def test_roots_chebyc():
    # 获取 Chebyshev C 多项式的权重函数
    weightf = orth.chebyc(5).weight_func
    # 验证 roots_chebyc 函数计算的 Gauss 积分精度
    verify_gauss_quad(sc.roots_chebyc, sc.eval_chebyc, weightf, -2., 2., 5)
    verify_gauss_quad(sc.roots_chebyc, sc.eval_chebyc, weightf, -2., 2., 25)
    verify_gauss_quad(sc.roots_chebyc, sc.eval_chebyc, weightf, -2., 2., 100,
                      atol=1e-12)

    # 获取非正交化和正交化的 Chebyshev C 多项式的根和权重
    x, w = sc.roots_chebyc(5, False)
    y, v, m = sc.roots_chebyc(5, True)
    # 断言非正交化和正交化的根和权重相等
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)

    # 计算权重函数在 [-2, 2] 上的积分以及误差
    muI, muI_err = integrate.quad(weightf, -2, 2)
    # 断言计算得到的积分值与预期的正交化根 m 值相等，误差在可接受范围内
    assert_allclose(m, muI, rtol=muI_err)

    # 断言 roots_chebyc 函数对于无效参数会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_chebyc, 0)
    assert_raises(ValueError, sc.roots_chebyc, 3.3)

def test_roots_chebys():
    # 获取 Chebyshev S 多项式的权重函数
    weightf = orth.chebys(5).weight_func
    # 验证 roots_chebys 函数计算的 Gauss 积分精度
    verify_gauss_quad(sc.roots_chebys, sc.eval_chebys, weightf, -2., 2., 5)
    verify_gauss_quad(sc.roots_chebys, sc.eval_chebys, weightf, -2., 2., 25)
    verify_gauss_quad(sc.roots_chebys, sc.eval_chebys, weightf, -2., 2., 100)

    # 获取非正交化和正交化的 Chebyshev S 多项式的根和权重
    x, w = sc.roots_chebys(5, False)
    y, v, m = sc.roots_chebys(5, True)
    # 断言非正交化和正交化的根和权重相等
    assert_allclose(x, y, 1e-14, 1e-14)
    assert_allclose(w, v, 1e-14, 1e-14)

    # 计算权重函数在 [-2, 2] 上的积分以及误差
    muI, muI_err = integrate.quad(weightf, -2, 2)
    # 断言计算得到的积分值与预期的正交化根 m 值相等，误差在可接受范围内
    assert_allclose(m, muI, rtol=muI_err)

    # 断言 roots_chebys 函数对于无效参数会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_chebys, 0)
    assert_raises(ValueError, sc.roots_chebys, 3.3)

def test_roots_sh_chebyt():
    # 获取 shifted Chebyshev T 多项式的权重函数
    weightf = orth.sh_chebyt(5).weight_func
    # 验证 roots_sh_chebyt 函数计算的 Gauss 积分精度
    verify_gauss_quad(sc.roots_sh_chebyt, sc.eval_sh_chebyt, weightf, 0., 1., 5)
    # 验证 Gauss-Chebyshev 积分的准确性，使用切比雪夫多项式的根和评估函数
    verify_gauss_quad(sc.roots_sh_chebyt, sc.eval_sh_chebyt, weightf, 0., 1., 25)
    # 再次验证 Gauss-Chebyshev 积分的准确性，增加更高的积分点数
    verify_gauss_quad(sc.roots_sh_chebyt, sc.eval_sh_chebyt, weightf, 0., 1., 100, atol=1e-13)

    # 获取切比雪夫多项式的根和权重，分别保存在 x 和 w 中
    x, w = sc.roots_sh_chebyt(5, False)
    # 获取切比雪夫多项式的根、权重和对应的导数，保存在 y、v 和 m 中
    y, v, m = sc.roots_sh_chebyt(5, True)
    # 断言两组根 x 和 y 在给定的容差范围内相等
    assert_allclose(x, y, 1e-14, 1e-14)
    # 断言两组权重 w 和 v 在给定的容差范围内相等
    assert_allclose(w, v, 1e-14, 1e-14)

    # 计算权重函数在区间 [0, 1] 上的积分值及其估计误差
    muI, muI_err = integrate.quad(weightf, 0, 1)
    # 断言多项式导数 m 与权重函数积分值 muI 在相对误差 muI_err 允许的范围内相等
    assert_allclose(m, muI, rtol=muI_err)

    # 断言在给定参数下，函数 sc.roots_sh_chebyt 应引发 ValueError 异常
    assert_raises(ValueError, sc.roots_sh_chebyt, 0)
    # 断言在给定参数下，函数 sc.roots_sh_chebyt 应引发 ValueError 异常
    assert_raises(ValueError, sc.roots_sh_chebyt, 3.3)
# 测试正交多项式的施莱比谢夫（Chebyshev）型一级和二级
def test_roots_sh_chebyu():
    # 获取施莱比谢夫型一级正交多项式的权重函数
    weightf = orth.sh_chebyu(5).weight_func
    # 验证施莱比谢夫型一级正交多项式的高斯积分
    verify_gauss_quad(sc.roots_sh_chebyu, sc.eval_sh_chebyu, weightf, 0., 1., 5)
    # 验证施莱比谢夫型一级正交多项式的高斯积分，增加采样点以提高精度
    verify_gauss_quad(sc.roots_sh_chebyu, sc.eval_sh_chebyu, weightf, 0., 1., 25)
    # 验证施莱比谢夫型一级正交多项式的高斯积分，非常高的采样点数，并设置绝对误差容限
    verify_gauss_quad(sc.roots_sh_chebyu, sc.eval_sh_chebyu, weightf, 0., 1., 100, atol=1e-13)

    # 获取施莱比谢夫型一级正交多项式的根（x）和权重（w）
    x, w = sc.roots_sh_chebyu(5, False)
    # 获取施莱比谢夫型一级正交多项式的根（y）、权重（v）和模（m）
    y, v, m = sc.roots_sh_chebyu(5, True)
    # 断言两种方法计算的根（x, y）非常接近，设置绝对容差
    assert_allclose(x, y, 1e-14, 1e-14)
    # 断言两种方法计算的权重（w, v）非常接近，设置绝对容差
    assert_allclose(w, v, 1e-14, 1e-14)

    # 使用积分计算施莱比谢夫型一级正交多项式的权重函数在 [0, 1] 上的积分值和误差
    muI, muI_err = integrate.quad(weightf, 0, 1)
    # 断言模（m）与积分计算的值（muI）非常接近，设置相对误差容限
    assert_allclose(m, muI, rtol=muI_err)

    # 断言 ValueError 当输入参数为非法值时会被引发
    assert_raises(ValueError, sc.roots_sh_chebyu, 0)
    assert_raises(ValueError, sc.roots_sh_chebyu, 3.3)

# 测试勒让德（Legendre）型正交多项式的根
def test_roots_legendre():
    # 获取勒让德型正交多项式的权重函数
    weightf = orth.legendre(5).weight_func
    # 验证勒让德型正交多项式的高斯积分
    verify_gauss_quad(sc.roots_legendre, sc.eval_legendre, weightf, -1., 1., 5)
    # 验证勒让德型正交多项式的高斯积分，增加采样点以提高精度
    verify_gauss_quad(sc.roots_legendre, sc.eval_legendre, weightf, -1., 1., 25, atol=1e-13)
    # 验证勒让德型正交多项式的高斯积分，非常高的采样点数，并设置绝对误差容限
    verify_gauss_quad(sc.roots_legendre, sc.eval_legendre, weightf, -1., 1., 100, atol=1e-12)

    # 获取勒让德型正交多项式的根（x）和权重（w）
    x, w = sc.roots_legendre(5, False)
    # 获取勒让德型正交多项式的根（y）、权重（v）和模（m）
    y, v, m = sc.roots_legendre(5, True)
    # 断言两种方法计算的根（x, y）非常接近，设置绝对容差
    assert_allclose(x, y, 1e-14, 1e-14)
    # 断言两种方法计算的权重（w, v）非常接近，设置绝对容差
    assert_allclose(w, v, 1e-14, 1e-14)

    # 使用积分计算勒让德型正交多项式的权重函数在 [-1, 1] 上的积分值和误差
    muI, muI_err = integrate.quad(weightf, -1, 1)
    # 断言模（m）与积分计算的值（muI）非常接近，设置相对误差容限
    assert_allclose(m, muI, rtol=muI_err)

    # 断言 ValueError 当输入参数为非法值时会被引发
    assert_raises(ValueError, sc.roots_legendre, 0)
    assert_raises(ValueError, sc.roots_legendre, 3.3)

# 测试勒让德型正交多项式的施莱比谢夫（Chebyshev）型二级
def test_roots_sh_legendre():
    # 获取施莱比谢夫型二级正交多项式的权重函数
    weightf = orth.sh_legendre(5).weight_func
    # 验证施莱比谢夫型二级正交多项式的高斯积分
    verify_gauss_quad(sc.roots_sh_legendre, sc.eval_sh_legendre, weightf, 0., 1., 5)
    # 验证施莱比谢夫型二级正交多项式的高斯积分，增加采样点以提高精度
    verify_gauss_quad(sc.roots_sh_legendre, sc.eval_sh_legendre, weightf, 0., 1., 25, atol=1e-13)
    # 验证施莱比谢夫型二级正交多项式的高斯积分，非常高的采样点数，并设置绝对误差容限
    verify_gauss_quad(sc.roots_sh_legendre, sc.eval_sh_legendre, weightf, 0., 1., 100, atol=1e-12)

    # 获取施莱比谢夫型二级正交多项式的根（x）和权重（w）
    x, w = sc.roots_sh_legendre(5, False)
    # 获取施莱比谢夫型二级正交多项式的根（y）、权重（v）和模（m）
    y, v, m = sc.roots_sh_legendre(5, True)
    # 断言两种方法计算的根（x, y）非常接近，设置绝对容差
    assert_allclose(x, y, 1e-14, 1e-14)
    # 断言两种方法计算的权重（w, v）非常接近，设置绝对容差
    assert_allclose(w, v, 1e-14, 1e-14)

    # 使用积分计算施莱比谢夫型二级正交多项式的权重函数在 [0, 1] 上的积分值和误差
    muI, muI_err = integrate.quad(weightf, 0, 1)
    # 断言模（m）与积分计算的值（muI）非常接近，设置相对误差容限
    assert_allclose(m, muI, rtol=muI_err)

    # 断言 ValueError 当输入参数为非法值时会被引发
    assert_raises(ValueError, sc.roots_sh_legendre, 0)
    assert_raises(ValueError, sc.roots_sh_legendre, 3.3)

# 测试拉盖尔（Laguerre）型正交多项式的根
def test_roots_laguerre():
    # 获取拉盖
# 定义测试函数 test_roots_genlaguerre，用于测试 roots_genlaguerre 相关功能
def test_roots_genlaguerre():
    # 定义 rootf 函数，返回一个 lambda 函数，用于计算一般拉盖尔多项式的根
    def rootf(a):
        return lambda n, mu: sc.roots_genlaguerre(n, a, mu)
    
    # 定义 evalf 函数，返回一个 lambda 函数，用于计算一般拉盖尔多项式的值
    def evalf(a):
        return lambda n, x: sc.eval_genlaguerre(n, a, x)
    
    # 定义 weightf 函数，返回一个 lambda 函数，用于计算拉盖尔多项式的权重函数
    def weightf(a):
        return lambda x: x ** a * np.exp(-x)

    # 将 verify_gauss_quad 函数赋值给 vgq 变量
    vgq = verify_gauss_quad

    # 使用不同的参数调用 vgq 函数进行验证高斯积分
    vgq(rootf(-0.5), evalf(-0.5), weightf(-0.5), 0., np.inf, 5)
    vgq(rootf(-0.5), evalf(-0.5), weightf(-0.5), 0., np.inf, 25, atol=1e-13)
    vgq(rootf(-0.5), evalf(-0.5), weightf(-0.5), 0., np.inf, 100, atol=1e-12)

    vgq(rootf(0.1), evalf(0.1), weightf(0.1), 0., np.inf, 5)
    vgq(rootf(0.1), evalf(0.1), weightf(0.1), 0., np.inf, 25, atol=1e-13)
    vgq(rootf(0.1), evalf(0.1), weightf(0.1), 0., np.inf, 100, atol=1.6e-13)

    vgq(rootf(1), evalf(1), weightf(1), 0., np.inf, 5)
    vgq(rootf(1), evalf(1), weightf(1), 0., np.inf, 25, atol=1e-13)
    vgq(rootf(1), evalf(1), weightf(1), 0., np.inf, 100, atol=1.03e-13)

    vgq(rootf(10), evalf(10), weightf(10), 0., np.inf, 5)
    vgq(rootf(10), evalf(10), weightf(10), 0., np.inf, 25, atol=1e-13)
    vgq(rootf(10), evalf(10), weightf(10), 0., np.inf, 100, atol=1e-12)

    vgq(rootf(50), evalf(50), weightf(50), 0., np.inf, 5)
    vgq(rootf(50), evalf(50), weightf(50), 0., np.inf, 25, atol=1e-13)
    vgq(rootf(50), evalf(50), weightf(50), 0., np.inf, 100, rtol=1e-14, atol=2e-13)

    # 获取一般拉盖尔多项式的根 x 和权重 w
    x, w = sc.roots_genlaguerre(5, 2, False)
    # 获取带参数的一般拉盖尔多项式的根 y、权重 v 和 m
    y, v, m = sc.roots_genlaguerre(5, 2, True)
    # 断言 x 与 y 的值在给定的精度范围内相等
    assert_allclose(x, y, 1e-14, 1e-14)
    # 断言 w 与 v 的值在给定的精度范围内相等
    assert_allclose(w, v, 1e-14, 1e-14)

    # 使用 integrate.quad 函数计算权重函数 weightf(2.) 的积分值及其误差
    muI, muI_err = integrate.quad(weightf(2.), 0., np.inf)
    # 断言带参数的一般拉盖尔多项式的 m 值与积分结果 muI 的值在相对误差范围内相等
    assert_allclose(m, muI, rtol=muI_err)

    # 断言 sc.roots_genlaguerre(0, 2) 会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_genlaguerre, 0, 2)
    # 断言 sc.roots_genlaguerre(3.3, 2) 会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_genlaguerre, 3.3, 2)
    # 断言 sc.roots_genlaguerre(3, -1.1) 会引发 ValueError 异常
    assert_raises(ValueError, sc.roots_genlaguerre, 3, -1.1)


# 定义测试函数 test_gh_6721，用于测试 gh_6721 的回归情况，验证不应该出现异常
def test_gh_6721():
    # 调用 sc.chebyt(65)(0.2)，用于回归测试 gh_6721，不应该引发异常
    sc.chebyt(65)(0.2)
```