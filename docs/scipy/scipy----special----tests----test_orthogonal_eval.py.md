# `D:\src\scipysrc\scipy\scipy\special\tests\test_orthogonal_eval.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_, assert_allclose  # 导入 NumPy 测试模块中的断言函数
import pytest  # 导入 Pytest 测试框架

from scipy.special import _ufuncs  # 导入 SciPy 库中的特殊函数 _ufuncs
import scipy.special._orthogonal as orth  # 导入 SciPy 库中的正交多项式模块
from scipy.special._testutils import FuncData  # 导入 SciPy 测试工具中的 FuncData 类


def test_eval_chebyt():
    # 创建一个长整型数组 n，范围为从0到10000，步长为7
    n = np.arange(0, 10000, 7, dtype=np.dtype("long"))
    # 随机生成一个在[-1, 1]之间的浮点数 x
    x = 2*np.random.rand() - 1
    # 计算余弦函数和特定角度的乘积，存储在 v1 中
    v1 = np.cos(n*np.arccos(x))
    # 使用 SciPy 的 eval_chebyt 函数计算切比雪夫多项式并存储在 v2 中
    v2 = _ufuncs.eval_chebyt(n, x)
    # 断言 v1 和 v2 在指定的相对误差范围内相等
    assert_(np.allclose(v1, v2, rtol=1e-15))


def test_eval_chebyt_gh20129():
    # 对 issue 20129 进行测试，验证 eval_chebyt 在复数输入 (2 + 0j) 时的输出是否为 5042.0
    assert _ufuncs.eval_chebyt(7, 2 + 0j) == 5042.0


def test_eval_genlaguerre_restriction():
    # 检查 eval_genlaguerre 在 alpha <= -1 时返回 NaN
    assert_(np.isnan(_ufuncs.eval_genlaguerre(0, -1, 0)))
    assert_(np.isnan(_ufuncs.eval_genlaguerre(0.1, -1, 0)))


def test_warnings():
    # 对于 ticket 1334，验证以下函数在无浮点数警告情况下是否正常工作
    with np.errstate(all='raise'):
        _ufuncs.eval_legendre(1, 0)
        _ufuncs.eval_laguerre(1, 1)
        _ufuncs.eval_gegenbauer(1, 1, 0)


class TestPolys:
    """
    检查 eval_* 函数与构建的多项式是否一致

    """

    def check_poly(self, func, cls, param_ranges=[], x_range=[], nn=10,
                   nparam=10, nx=10, rtol=1e-8):
        np.random.seed(1234)

        dataset = []
        for n in np.arange(nn):
            # 根据 param_ranges 和 x_range 生成多项式数据集
            params = [a + (b-a)*np.random.rand(nparam) for a,b in param_ranges]
            params = np.asarray(params).T
            if not param_ranges:
                params = [0]
            for p in params:
                if param_ranges:
                    p = (n,) + tuple(p)
                else:
                    p = (n,)
                x = x_range[0] + (x_range[1] - x_range[0])*np.random.rand(nx)
                x[0] = x_range[0]  # 始终包含域起始点
                x[1] = x_range[1]  # 始终包含域结束点
                poly = np.poly1d(cls(*p).coef)
                z = np.c_[np.tile(p, (nx,1)), x, poly(x)]
                dataset.append(z)

        dataset = np.concatenate(dataset, axis=0)

        def polyfunc(*p):
            p = (p[0].astype(np.dtype("long")),) + p[1:]
            return func(*p)

        with np.errstate(all='raise'):
            # 使用 FuncData 类检查多项式函数的结果
            ds = FuncData(polyfunc, dataset, list(range(len(param_ranges)+2)), -1,
                          rtol=rtol)
            ds.check()

    def test_jacobi(self):
        # 检查 eval_jacobi 函数与 jacobi 多项式的一致性
        self.check_poly(_ufuncs.eval_jacobi, orth.jacobi,
                        param_ranges=[(-0.99, 10), (-0.99, 10)],
                        x_range=[-1, 1], rtol=1e-5)

    def test_sh_jacobi(self):
        # 检查 eval_sh_jacobi 函数与 sh_jacobi 多项式的一致性
        self.check_poly(_ufuncs.eval_sh_jacobi, orth.sh_jacobi,
                        param_ranges=[(1, 10), (0, 1)], x_range=[0, 1],
                        rtol=1e-5)

    def test_gegenbauer(self):
        # 检查 eval_gegenbauer 函数与 gegenbauer 多项式的一致性
        self.check_poly(_ufuncs.eval_gegenbauer, orth.gegenbauer,
                        param_ranges=[(-0.499, 10)], x_range=[-1, 1],
                        rtol=1e-7)
    # 测试 Chebyshev 多项式函数 eval_chebyt 的功能
    def test_chebyt(self):
        # 调用 self.check_poly 方法，检查 eval_chebyt 函数与 orth.chebyt 的一致性
        self.check_poly(_ufuncs.eval_chebyt, orth.chebyt,
                        param_ranges=[], x_range=[-1, 1])

    # 测试 Chebyshev 多项式函数 eval_chebyu 的功能
    def test_chebyu(self):
        # 调用 self.check_poly 方法，检查 eval_chebyu 函数与 orth.chebyu 的一致性
        self.check_poly(_ufuncs.eval_chebyu, orth.chebyu,
                        param_ranges=[], x_range=[-1, 1])

    # 测试 shifted Chebyshev 多项式函数 eval_chebys 的功能
    def test_chebys(self):
        # 调用 self.check_poly 方法，检查 eval_chebys 函数与 orth.chebys 的一致性
        self.check_poly(_ufuncs.eval_chebys, orth.chebys,
                        param_ranges=[], x_range=[-2, 2])

    # 测试 shifted Chebyshev 第二类多项式函数 eval_chebyc 的功能
    def test_chebyc(self):
        # 调用 self.check_poly 方法，检查 eval_chebyc 函数与 orth.chebyc 的一致性
        self.check_poly(_ufuncs.eval_chebyc, orth.chebyc,
                        param_ranges=[], x_range=[-2, 2])

    # 测试 spherical harmonics Chebyshev 多项式函数 eval_sh_chebyt 的功能
    def test_sh_chebyt(self):
        # 使用 np.errstate 忽略所有错误
        with np.errstate(all='ignore'):
            # 调用 self.check_poly 方法，检查 eval_sh_chebyt 函数与 orth.sh_chebyt 的一致性
            self.check_poly(_ufuncs.eval_sh_chebyt, orth.sh_chebyt,
                            param_ranges=[], x_range=[0, 1])

    # 测试 spherical harmonics Chebyshev 多项式函数 eval_sh_chebyu 的功能
    def test_sh_chebyu(self):
        # 调用 self.check_poly 方法，检查 eval_sh_chebyu 函数与 orth.sh_chebyu 的一致性
        self.check_poly(_ufuncs.eval_sh_chebyu, orth.sh_chebyu,
                        param_ranges=[], x_range=[0, 1])

    # 测试 Legendre 多项式函数 eval_legendre 的功能
    def test_legendre(self):
        # 调用 self.check_poly 方法，检查 eval_legendre 函数与 orth.legendre 的一致性
        self.check_poly(_ufuncs.eval_legendre, orth.legendre,
                        param_ranges=[], x_range=[-1, 1])

    # 测试 spherical harmonics Legendre 多项式函数 eval_sh_legendre 的功能
    def test_sh_legendre(self):
        # 使用 np.errstate 忽略所有错误
        with np.errstate(all='ignore'):
            # 调用 self.check_poly 方法，检查 eval_sh_legendre 函数与 orth.sh_legendre 的一致性
            self.check_poly(_ufuncs.eval_sh_legendre, orth.sh_legendre,
                            param_ranges=[], x_range=[0, 1])

    # 测试 generalized Laguerre 多项式函数 eval_genlaguerre 的功能
    def test_genlaguerre(self):
        # 调用 self.check_poly 方法，检查 eval_genlaguerre 函数与 orth.genlaguerre 的一致性
        self.check_poly(_ufuncs.eval_genlaguerre, orth.genlaguerre,
                        param_ranges=[(-0.99, 10)], x_range=[0, 100])

    # 测试 Laguerre 多项式函数 eval_laguerre 的功能
    def test_laguerre(self):
        # 调用 self.check_poly 方法，检查 eval_laguerre 函数与 orth.laguerre 的一致性
        self.check_poly(_ufuncs.eval_laguerre, orth.laguerre,
                        param_ranges=[], x_range=[0, 100])

    # 测试 Hermite 多项式函数 eval_hermite 的功能
    def test_hermite(self):
        # 调用 self.check_poly 方法，检查 eval_hermite 函数与 orth.hermite 的一致性
        self.check_poly(_ufuncs.eval_hermite, orth.hermite,
                        param_ranges=[], x_range=[-100, 100])

    # 测试 normalized Hermite 多项式函数 eval_hermitenorm 的功能
    def test_hermitenorm(self):
        # 调用 self.check_poly 方法，检查 eval_hermitenorm 函数与 orth.hermitenorm 的一致性
        self.check_poly(_ufuncs.eval_hermitenorm, orth.hermitenorm,
                        param_ranges=[], x_range=[-100, 100])
class TestRecurrence:
    """
    Check that the eval_* functions sig='ld->d' and 'dd->d' agree.
    """

    # 检查多项式函数的一致性
    def check_poly(self, func, param_ranges=[], x_range=[], nn=10,
                   nparam=10, nx=10, rtol=1e-8):
        # 设定随机种子以保证结果可复现
        np.random.seed(1234)

        # 初始化数据集
        dataset = []
        # 对每个 n 进行迭代
        for n in np.arange(nn):
            # 根据参数范围生成随机参数
            params = [a + (b-a)*np.random.rand(nparam) for a,b in param_ranges]
            params = np.asarray(params).T
            # 如果没有参数范围，则使用默认参数值
            if not param_ranges:
                params = [0]
            # 对每组参数进行迭代
            for p in params:
                if param_ranges:
                    p = (n,) + tuple(p)
                else:
                    p = (n,)
                # 生成随机的 x 值，确保包含域的起始和结束点
                x = x_range[0] + (x_range[1] - x_range[0])*np.random.rand(nx)
                x[0] = x_range[0]  # 始终包含域的起始点
                x[1] = x_range[1]  # 始终包含域的结束点
                # 设置函数调用的参数签名
                kw = dict(sig=(len(p)+1)*'d'+'->d')
                # 构建数据点并添加到数据集中
                z = np.c_[np.tile(p, (nx,1)), x, func(*(p + (x,)), **kw)]
                dataset.append(z)

        # 将数据集沿着指定轴连接起来
        dataset = np.concatenate(dataset, axis=0)

        # 定义多项式函数
        def polyfunc(*p):
            p = (p[0].astype(int),) + p[1:]
            kw = dict(sig='l'+(len(p)-1)*'d'+'->d')
            return func(*p, **kw)

        # 在计算期间捕获所有的浮点错误
        with np.errstate(all='raise'):
            # 使用 FuncData 对象初始化数据集并进行检查
            ds = FuncData(polyfunc, dataset, list(range(len(param_ranges)+2)), -1,
                          rtol=rtol)
            ds.check()

    # 测试 Jacobi 函数
    def test_jacobi(self):
        self.check_poly(_ufuncs.eval_jacobi,
                        param_ranges=[(-0.99, 10), (-0.99, 10)],
                        x_range=[-1, 1])

    # 测试 shifted Jacobi 函数
    def test_sh_jacobi(self):
        self.check_poly(_ufuncs.eval_sh_jacobi,
                        param_ranges=[(1, 10), (0, 1)], x_range=[0, 1])

    # 测试 Gegenbauer 函数
    def test_gegenbauer(self):
        self.check_poly(_ufuncs.eval_gegenbauer,
                        param_ranges=[(-0.499, 10)], x_range=[-1, 1])

    # 测试 Chebyshev T 函数
    def test_chebyt(self):
        self.check_poly(_ufuncs.eval_chebyt,
                        param_ranges=[], x_range=[-1, 1])

    # 测试 Chebyshev U 函数
    def test_chebyu(self):
        self.check_poly(_ufuncs.eval_chebyu,
                        param_ranges=[], x_range=[-1, 1])

    # 测试 shifted Chebyshev T 函数
    def test_sh_chebyt(self):
        self.check_poly(_ufuncs.eval_sh_chebyt,
                        param_ranges=[], x_range=[0, 1])

    # 测试 shifted Chebyshev U 函数
    def test_sh_chebyu(self):
        self.check_poly(_ufuncs.eval_sh_chebyu,
                        param_ranges=[], x_range=[0, 1])

    # 测试 Chebyshev S 函数
    def test_chebys(self):
        self.check_poly(_ufuncs.eval_chebys,
                        param_ranges=[], x_range=[-2, 2])

    # 测试 Chebyshev C 函数
    def test_chebyc(self):
        self.check_poly(_ufuncs.eval_chebyc,
                        param_ranges=[], x_range=[-2, 2])

    # 测试 Legendre 函数
    def test_legendre(self):
        self.check_poly(_ufuncs.eval_legendre,
                        param_ranges=[], x_range=[-1, 1])

    # 测试 shifted Legendre 函数
    def test_sh_legendre(self):
        self.check_poly(_ufuncs.eval_sh_legendre,
                        param_ranges=[], x_range=[0, 1])
    # 定义测试函数 test_genlaguerre，用于测试 eval_genlaguerre 函数
    def test_genlaguerre(self):
        # 调用 self.check_poly 方法，检查 eval_genlaguerre 函数的多项式性质
        self.check_poly(_ufuncs.eval_genlaguerre,
                        param_ranges=[(-0.99, 10)], x_range=[0, 100])

    # 定义测试函数 test_laguerre，用于测试 eval_laguerre 函数
    def test_laguerre(self):
        # 调用 self.check_poly 方法，检查 eval_laguerre 函数的多项式性质
        self.check_poly(_ufuncs.eval_laguerre,
                        param_ranges=[], x_range=[0, 100])

    # 定义测试函数 test_hermite，用于测试 eval_hermite 函数
    def test_hermite(self):
        # 调用 _ufuncs.eval_hermite 函数计算 Hermite 函数的值，参数为 (70, 1.0)
        v = _ufuncs.eval_hermite(70, 1.0)
        # 定义预期的 Hermite 函数值
        a = -1.457076485701412e60
        # 使用 assert_allclose 断言函数，验证计算得到的 v 和预期值 a 在允许误差范围内是否相等
        assert_allclose(v, a)
# 为 hermite_domain 函数编写回归测试，验证 gh-11091 问题
def test_hermite_domain():
    # 断言 eval_hermite(-1, 1.0) 返回 NaN
    assert np.isnan(_ufuncs.eval_hermite(-1, 1.0))
    # 断言 eval_hermitenorm(-1, 1.0) 返回 NaN
    assert np.isnan(_ufuncs.eval_hermitenorm(-1, 1.0))


# 使用参数化装饰器为 hermite_nan 函数编写回归测试，验证 gh-11369 问题
@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("x", [0, 1, np.nan])
def test_hermite_nan(n, x):
    # 断言 eval_hermite(n, x) 返回 NaN 的结果是否与 [n, x] 中是否存在 NaN 相匹配
    assert np.isnan(_ufuncs.eval_hermite(n, x)) == np.any(np.isnan([n, x]))
    # 断言 eval_hermitenorm(n, x) 返回 NaN 的结果是否与 [n, x] 中是否存在 NaN 相匹配
    assert np.isnan(_ufuncs.eval_hermitenorm(n, x)) == np.any(np.isnan([n, x]))


# 使用参数化装饰器为 genlaguerre_nan 函数编写回归测试，验证 gh-11361 问题
@pytest.mark.parametrize('n', [0, 1, 2, 3.2])
@pytest.mark.parametrize('alpha', [1, np.nan])
@pytest.mark.parametrize('x', [2, np.nan])
def test_genlaguerre_nan(n, alpha, x):
    # 断言 eval_genlaguerre(n, alpha, x) 返回 NaN 的结果是否与 [n, alpha, x] 中是否存在 NaN 相匹配
    nan_laguerre = np.isnan(_ufuncs.eval_genlaguerre(n, alpha, x))
    nan_arg = np.any(np.isnan([n, alpha, x]))
    assert nan_laguerre == nan_arg


# 使用参数化装饰器为 gegenbauer_nan 函数编写回归测试，验证 gh-11370 问题
@pytest.mark.parametrize('n', [0, 1, 2, 3.2])
@pytest.mark.parametrize('alpha', [0.0, 1, np.nan])
@pytest.mark.parametrize('x', [1e-6, 2, np.nan])
def test_gegenbauer_nan(n, alpha, x):
    # 断言 eval_gegenbauer(n, alpha, x) 返回 NaN 的结果是否与 [n, alpha, x] 中是否存在 NaN 相匹配
    nan_gegenbauer = np.isnan(_ufuncs.eval_gegenbauer(n, alpha, x))
    nan_arg = np.any(np.isnan([n, alpha, x]))
    assert nan_gegenbauer == nan_arg
```