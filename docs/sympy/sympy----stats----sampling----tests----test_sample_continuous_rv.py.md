# `D:\src\scipysrc\sympy\sympy\stats\sampling\tests\test_sample_continuous_rv.py`

```
# 从 sympy 库中导入无穷大 oo，用于表示无限大的数值
from sympy.core.numbers import oo
# 从 sympy 库中导入符号操作相关的类 Symbol
from sympy.core.symbol import Symbol
# 从 sympy 库中导入指数函数 exp
from sympy.functions.elementary.exponential import exp
# 从 sympy 库中导入集合 Interval，用于表示区间
from sympy.sets.sets import Interval
# 从 sympy.external 模块中导入 import_module 函数，用于动态导入模块
from sympy.external import import_module
# 从 sympy.stats 模块中导入一系列概率分布类和相关函数
from sympy.stats import Beta, Chi, Normal, Gamma, Exponential, LogNormal, Pareto, ChiSquared, Uniform, sample, \
    BetaPrime, Cauchy, GammaInverse, GaussianInverse, StudentT, Weibull, density, ContinuousRV, FDistribution, \
    Gumbel, Laplace, Logistic, Rayleigh, Triangular
# 从 sympy.testing.pytest 中导入 skip 和 raises 函数，用于测试跳过和异常抛出
from sympy.testing.pytest import skip, raises


# 定义测试函数 test_sample_numpy
def test_sample_numpy():
    # 定义一个包含 numpy 库中支持的概率分布对象的列表
    distribs_numpy = [
        Beta("B", 1, 1),
        Normal("N", 0, 1),
        Gamma("G", 2, 7),
        Exponential("E", 2),
        LogNormal("LN", 0, 1),
        Pareto("P", 1, 1),
        ChiSquared("CS", 2),
        Uniform("U", 0, 1),
        FDistribution("FD", 1, 2),
        Gumbel("GB", 1, 2),
        Laplace("L", 1, 2),
        Logistic("LO", 1, 2),
        Rayleigh("R", 1),
        Triangular("T", 1, 2, 2),
    ]
    # 设置样本大小
    size = 3
    # 动态导入 numpy 模块
    numpy = import_module('numpy')
    # 如果未成功导入 numpy 模块，则跳过使用 numpy 的测试
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        # 遍历 distribs_numpy 列表中的每个分布对象 X
        for X in distribs_numpy:
            # 从分布对象 X 中生成样本数据，使用 numpy 库，指定样本大小为 size
            samps = sample(X, size=size, library='numpy')
            # 验证生成的每个样本在该分布对象 X 的定义域内
            for sam in samps:
                assert sam in X.pspace.domain.set
        # 测试抛出 NotImplementedError 异常，因为 Chi 分布不支持使用 numpy 库生成样本
        raises(NotImplementedError,
               lambda: sample(Chi("C", 1), library='numpy'))
    # 测试抛出 NotImplementedError 异常，因为 Chi 分布的样本生成函数在 library 参数为 'tensorflow' 时未实现
    raises(NotImplementedError,
           lambda: Chi("C", 1).pspace.distribution.sample(library='tensorflow'))


# 定义测试函数 test_sample_scipy
def test_sample_scipy():
    # 定义一个包含 scipy 库中支持的概率分布对象的列表
    distribs_scipy = [
        Beta("B", 1, 1),
        BetaPrime("BP", 1, 1),
        Cauchy("C", 1, 1),
        Chi("C", 1),
        Normal("N", 0, 1),
        Gamma("G", 2, 7),
        GammaInverse("GI", 1, 1),
        GaussianInverse("GUI", 1, 1),
        Exponential("E", 2),
        LogNormal("LN", 0, 1),
        Pareto("P", 1, 1),
        StudentT("S", 2),
        ChiSquared("CS", 2),
        Uniform("U", 0, 1)
    ]
    # 设置样本大小
    size = 3
    # 动态导入 scipy 模块
    scipy = import_module('scipy')
    # 如果未成功导入 scipy 模块，则跳过使用 scipy 的测试
    if not scipy:
        skip('Scipy is not installed. Abort tests for _sample_scipy.')
    else:
        # 遍历 distribs_scipy 列表中的每个分布对象 X
        for X in distribs_scipy:
            # 从分布对象 X 中生成样本数据，使用 scipy 库，指定样本大小为 size
            samps = sample(X, size=size, library='scipy')
            # 验证生成的每个样本在该分布对象 X 的定义域内
            for sam in samps:
                assert sam in X.pspace.domain.set
            # 生成一个二维样本数组，使用 scipy 库，验证每个样本在分布对象 X 的定义域内
            samps2 = sample(X, size=(2, 2), library='scipy')
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set


# 定义测试函数 test_sample_pymc
def test_sample_pymc():
    # 定义一个包含 pymc 库中支持的概率分布对象的列表
    distribs_pymc = [
        Beta("B", 1, 1),
        Cauchy("C", 1, 1),
        Normal("N", 0, 1),
        Gamma("G", 2, 7),
        GaussianInverse("GI", 1, 1),
        Exponential("E", 2),
        LogNormal("LN", 0, 1),
        Pareto("P", 1, 1),
        ChiSquared("CS", 2),
        Uniform("U", 0, 1)
    ]
    # 设置样本大小
    size = 3
    # 动态导入 pymc 模块
    pymc = import_module('pymc')
    # 如果未成功导入 pymc 模块，则跳过使用 pymc 的测试
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        # 对于 distribs_pymc 中的每个分布 X，进行采样
        for X in distribs_pymc:
            # 使用 pymc 库对分布 X 进行 size 大小的采样
            samps = sample(X, size=size, library='pymc')
            # 对每个样本 sam 进行断言，确保其在 X 的取值域内
            for sam in samps:
                assert sam in X.pspace.domain.set
        # 断言捕获 NotImplementedError 异常，确保 sample 函数对 Chi("C", 1) 分布的调用未实现
        raises(NotImplementedError,
               lambda: sample(Chi("C", 1), library='pymc'))
def test_sampling_gamma_inverse():
    # 导入 scipy 模块
    scipy = import_module('scipy')
    # 如果 scipy 模块未安装，则跳过测试并显示消息
    if not scipy:
        skip('Scipy not installed. Abort tests for sampling of gamma inverse.')
    # 创建 GammaInverse 对象 X，参数为 "x", 1, 1
    X = GammaInverse("x", 1, 1)
    # 确保从 X 的概率空间中采样的结果在定义域中
    assert sample(X) in X.pspace.domain.set


def test_lognormal_sampling():
    # 目前仅支持密度函数和采样功能
    scipy = import_module('scipy')
    # 如果 scipy 模块未安装，则跳过测试并显示消息
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    # 对于范围为 0 到 2 的三个值 i，创建 LogNormal 对象 X
    for i in range(3):
        X = LogNormal('x', i, 1)
        # 确保从 X 的概率空间中采样的结果在定义域中
        assert sample(X) in X.pspace.domain.set

    size = 5
    # 从 X 中采样指定大小的样本集合，并逐个检查是否在定义域中
    samps = sample(X, size=size)
    for samp in samps:
        assert samp in X.pspace.domain.set


def test_sampling_gaussian_inverse():
    # 导入 scipy 模块
    scipy = import_module('scipy')
    # 如果 scipy 模块未安装，则跳过测试并显示消息
    if not scipy:
        skip('Scipy not installed. Abort tests for sampling of Gaussian inverse.')
    # 创建 GaussianInverse 对象 X，参数为 "x", 1, 1
    X = GaussianInverse("x", 1, 1)
    # 确保使用 scipy 库从 X 的概率空间中采样的结果在定义域中
    assert sample(X, library='scipy') in X.pspace.domain.set


def test_prefab_sampling():
    # 导入 scipy 模块
    scipy = import_module('scipy')
    # 如果 scipy 模块未安装，则跳过测试并显示消息
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    # 创建不同类型的随机变量对象并进行采样测试
    N = Normal('X', 0, 1)
    L = LogNormal('L', 0, 1)
    E = Exponential('Ex', 1)
    P = Pareto('P', 1, 3)
    W = Weibull('W', 1, 1)
    U = Uniform('U', 0, 1)
    B = Beta('B', 2, 5)
    G = Gamma('G', 1, 3)

    variables = [N, L, E, P, W, U, B, G]
    niter = 10
    size = 5
    # 对每个变量进行多次采样测试
    for var in variables:
        for _ in range(niter):
            # 确保从 var 的概率空间中采样的结果在定义域中
            assert sample(var) in var.pspace.domain.set
            # 从 var 中采样指定大小的样本集合，并逐个检查是否在定义域中
            samps = sample(var, size=size)
            for samp in samps:
                assert samp in var.pspace.domain.set


def test_sample_continuous():
    # 创建符号 z
    z = Symbol('z')
    # 创建 ContinuousRV 对象 Z，指定密度函数和定义域为区间 [0, oo)
    Z = ContinuousRV(z, exp(-z), set=Interval(0, oo))
    # 确保在 -1 处 Z 的密度函数值为 0
    assert density(Z)(-1) == 0

    # 导入 scipy 模块
    scipy = import_module('scipy')
    # 如果 scipy 模块未安装，则跳过测试并显示消息
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    # 确保从 Z 的概率空间中采样的结果在定义域中
    assert sample(Z) in Z.pspace.domain.set
    # 从 Z 的概率空间中随机采样，并确保符号和值在定义域中
    sym, val = list(Z.pspace.sample().items())[0]
    assert sym == Z and val in Interval(0, oo)

    libraries = ['scipy', 'numpy', 'pymc']
    # 使用不同的库进行多次采样测试
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0, s1, s2 = [], [], []
                # 使用指定库进行从 Z 中采样大小为 10 的样本集合，并检查结果
                s0 = sample(Z, size=10, library=lib, seed=0)
                s1 = sample(Z, size=10, library=lib, seed=0)
                s2 = sample(Z, size=10, library=lib, seed=1)
                assert all(s0 == s1)
                assert all(s1 != s2)
        except NotImplementedError:
            continue
```