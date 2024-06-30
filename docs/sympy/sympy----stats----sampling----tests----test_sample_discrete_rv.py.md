# `D:\src\scipysrc\sympy\sympy\stats\sampling\tests\test_sample_discrete_rv.py`

```
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.stats import Geometric, Poisson, Zeta, sample, Skellam, DiscreteRV, Logarithmic, NegativeBinomial, YuleSimon
from sympy.testing.pytest import skip, raises, slow


def test_sample_numpy():
    # 创建包含几何分布、泊松分布和杂项分布的列表
    distribs_numpy = [
        Geometric('G', 0.5),
        Poisson('P', 1),
        Zeta('Z', 2)
    ]
    size = 3
    # 导入 numpy 模块
    numpy = import_module('numpy')
    # 如果没有安装 numpy，则跳过这个测试
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        for X in distribs_numpy:
            # 使用 numpy 库对分布进行抽样
            samps = sample(X, size=size, library='numpy')
            for sam in samps:
                # 确保抽样值在分布的定义域内
                assert sam in X.pspace.domain.set
        # 测试 Skellam 分布抽样时应引发 NotImplementedError
        raises(NotImplementedError,
               lambda: sample(Skellam('S', 1, 1), library='numpy'))
    # 测试 Skellam 分布的抽样方法是否引发 NotImplementedError
    raises(NotImplementedError,
           lambda: Skellam('S', 1, 1).pspace.distribution.sample(library='tensorflow'))


def test_sample_scipy():
    p = S(2)/3
    x = Symbol('x', integer=True, positive=True)
    pdf = p*(1 - p)**(x - 1) # 几何分布的概率密度函数
    # 创建包含不同分布的列表
    distribs_scipy = [
        DiscreteRV(x, pdf, set=S.Naturals),
        Geometric('G', 0.5),
        Logarithmic('L', 0.5),
        NegativeBinomial('N', 5, 0.4),
        Poisson('P', 1),
        Skellam('S', 1, 1),
        YuleSimon('Y', 1),
        Zeta('Z', 2)
    ]
    size = 3
    # 导入 scipy 模块
    scipy = import_module('scipy')
    # 如果没有安装 scipy，则跳过这个测试
    if not scipy:
        skip('Scipy is not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            # 使用 scipy 库对分布进行抽样
            samps = sample(X, size=size, library='scipy')
            # 对于单个抽样结果，确保其在分布的定义域内
            for sam in samps:
                assert sam in X.pspace.domain.set
            # 对于 size=(2, 2) 的抽样结果，确保每个值在分布的定义域内
            samps2 = sample(X, size=(2, 2), library='scipy')
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set


def test_sample_pymc():
    # 创建包含几何分布、泊松分布和负二项分布的列表
    distribs_pymc = [
        Geometric('G', 0.5),
        Poisson('P', 1),
        NegativeBinomial('N', 5, 0.4)
    ]
    size = 3
    # 导入 pymc 模块
    pymc = import_module('pymc')
    # 如果没有安装 pymc，则跳过这个测试
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            # 使用 pymc 库对分布进行抽样
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                # 确保抽样值在分布的定义域内
                assert sam in X.pspace.domain.set
        # 测试 Skellam 分布抽样时应引发 NotImplementedError
        raises(NotImplementedError,
               lambda: sample(Skellam('S', 1, 1), library='pymc'))


@slow
def test_sample_discrete():
    X = Geometric('X', S.Half)
    # 导入 scipy 模块
    scipy = import_module('scipy')
    # 如果没有安装 scipy，则跳过这个测试
    if not scipy:
        skip('Scipy not installed. Abort tests')
    # 确保单次抽样结果在分布的定义域内
    assert sample(X) in X.pspace.domain.set
    # 对 size=2 的抽样结果，确保每个值在分布的定义域内；这可能需要较长时间，如果没有安装 scipy
    samps = sample(X, size=2)
    for samp in samps:
        assert samp in X.pspace.domain.set

    libraries = ['scipy', 'numpy', 'pymc']
    # 遍历给定的库列表
    for lib in libraries:
        try:
            # 尝试导入当前循环中的库模块
            imported_lib = import_module(lib)
            # 如果成功导入了库模块
            if imported_lib:
                # 初始化三个空列表 s0, s1, s2
                s0, s1, s2 = [], [], []
                # 使用给定库从数据集 X 中抽样大小为 10 的数据，种子为 0
                s0 = sample(X, size=10, library=lib, seed=0)
                # 使用给定库从数据集 X 中抽样大小为 10 的数据，种子为 0
                s1 = sample(X, size=10, library=lib, seed=0)
                # 使用给定库从数据集 X 中抽样大小为 10 的数据，种子为 1
                s2 = sample(X, size=10, library=lib, seed=1)
                # 断言：检查 s0 和 s1 的所有元素是否相等
                assert all(s0 == s1)
                # 断言：检查 s1 和 s2 的所有元素是否不完全相等
                assert not all(s1 == s2)
        # 捕获 NotImplementedError 异常，继续循环下一个库
        except NotImplementedError:
            continue
```