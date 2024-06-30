# `D:\src\scipysrc\sympy\sympy\stats\sampling\tests\test_sample_finite_rv.py`

```
# 从 sympy 库中导入 Rational 类
from sympy.core.numbers import Rational
# 从 sympy 库中导入 S 单例
from sympy.core.singleton import S
# 从 sympy.external 模块中导入 import_module 函数
from sympy.external import import_module
# 从 sympy.stats 模块中导入多个随机变量类
from sympy.stats import Binomial, sample, Die, FiniteRV, DiscreteUniform, Bernoulli, BetaBinomial, Hypergeometric, Rademacher
# 从 sympy.testing.pytest 模块中导入 skip 和 raises 函数
from sympy.testing.pytest import skip, raises


# 定义测试函数 test_given_sample
def test_given_sample():
    # 创建一个六面骰子 X
    X = Die('X', 6)
    # 尝试导入 scipy 库
    scipy = import_module('scipy')
    # 如果 scipy 未安装，则跳过测试
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    # 断言 sample(X, X > 5) 的结果为 6
    assert sample(X, X > 5) == 6


# 定义测试函数 test_sample_numpy
def test_sample_numpy():
    # 定义 numpy 需要使用的分布列表
    distribs_numpy = [
        Binomial("B", 5, 0.4),
        Hypergeometric("H", 2, 1, 1)
    ]
    size = 3
    # 尝试导入 numpy 库
    numpy = import_module('numpy')
    # 如果 numpy 未安装，则跳过测试
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        # 遍历 distribs_numpy 中的每个分布 X
        for X in distribs_numpy:
            # 使用 numpy 库对分布 X 进行抽样，抽样大小为 size
            samps = sample(X, size=size, library='numpy')
            # 断言抽样结果中的每个样本 sam 都属于分布 X 的定义域
            for sam in samps:
                assert sam in X.pspace.domain.set
        # 断言对 Die("D") 使用 numpy 库进行抽样会引发 NotImplementedError 异常
        raises(NotImplementedError,
               lambda: sample(Die("D"), library='numpy'))
    # 断言对 Die("D") 使用 tensorflow 库进行抽样会引发 NotImplementedError 异常
    raises(NotImplementedError,
           lambda: Die("D").pspace.sample(library='tensorflow'))


# 定义测试函数 test_sample_scipy
def test_sample_scipy():
    # 定义 scipy 需要使用的分布列表
    distribs_scipy = [
        FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)}),
        DiscreteUniform("Y", list(range(5))),
        Die("D"),
        Bernoulli("Be", 0.3),
        Binomial("Bi", 5, 0.4),
        BetaBinomial("Bb", 2, 1, 1),
        Hypergeometric("H", 1, 1, 1),
        Rademacher("R")
    ]
    
    size = 3
    # 尝试导入 scipy 库
    scipy = import_module('scipy')
    # 如果 scipy 未安装，则跳过测试
    if not scipy:
        skip('Scipy not installed. Abort tests for _sample_scipy.')
    else:
        # 遍历 distribs_scipy 中的每个分布 X
        for X in distribs_scipy:
            # 使用 scipy 库对分布 X 进行抽样，抽样大小为 size
            samps = sample(X, size=size)
            # 使用 scipy 库对分布 X 进行多维抽样，大小为 (2, 2)
            samps2 = sample(X, size=(2, 2))
            # 断言抽样结果中的每个样本 sam 都属于分布 X 的定义域
            for sam in samps:
                assert sam in X.pspace.domain.set
            # 断言多维抽样结果中的每个样本都属于分布 X 的定义域
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set


# 定义测试函数 test_sample_pymc
def test_sample_pymc():
    # 定义 pymc 需要使用的分布列表
    distribs_pymc = [
        Bernoulli('B', 0.2),
        Binomial('N', 5, 0.4)
    ]
    size = 3
    # 尝试导入 pymc 库
    pymc = import_module('pymc')
    # 如果 pymc 未安装，则跳过测试
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        # 遍历 distribs_pymc 中的每个分布 X
        for X in distribs_pymc:
            # 使用 pymc 库对分布 X 进行抽样，抽样大小为 size
            samps = sample(X, size=size, library='pymc')
            # 断言抽样结果中的每个样本 sam 都属于分布 X 的定义域
            for sam in samps:
                assert sam in X.pspace.domain.set
        # 断言对 Die("D") 使用 pymc 库进行抽样会引发 NotImplementedError 异常
        raises(NotImplementedError,
               lambda: (sample(Die("D"), library='pymc')))


# 定义测试函数 test_sample_seed
def test_sample_seed():
    # 创建一个有限随机变量 F
    F = FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)})
    size = 10
    # 定义需要测试的库列表
    libraries = ['scipy', 'numpy', 'pymc']
    # 遍历给定的库列表
    for lib in libraries:
        # 尝试导入当前循环中的库
        try:
            imported_lib = import_module(lib)
            # 如果成功导入库
            if imported_lib:
                # 使用指定的库从 F 中生成随机样本 s0，使用相同的种子数（seed=0）
                s0 = sample(F, size=size, library=lib, seed=0)
                # 使用相同的库和种子数（seed=0），再次生成随机样本 s1
                s1 = sample(F, size=size, library=lib, seed=0)
                # 使用相同的库但不同的种子数（seed=1），生成随机样本 s2
                s2 = sample(F, size=size, library=lib, seed=1)
                # 断言 s0 和 s1 的所有元素相等
                assert all(s0 == s1)
                # 断言 s1 和 s2 的至少一个元素不相等
                assert not all(s1 == s2)
        # 如果当前库抛出 NotImplementedError 异常，则继续处理下一个库
        except NotImplementedError:
            continue
```