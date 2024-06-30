# `D:\src\scipysrc\sympy\sympy\stats\tests\test_matrix_distributions.py`

```
# 导入 SymPy 中的具体模块和函数
from sympy.concrete.products import Product
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices import Determinant, Matrix, Trace, MatrixSymbol, MatrixSet
from sympy.stats import density, sample
from sympy.stats.matrix_distributions import (MatrixGammaDistribution,
                MatrixGamma, MatrixPSpace, Wishart, MatrixNormal, MatrixStudentT)
from sympy.testing.pytest import raises, skip
from sympy.external import import_module

# 定义测试函数 test_MatrixPSpace
def test_MatrixPSpace():
    # 创建一个 MatrixGammaDistribution 对象 M
    M = MatrixGammaDistribution(1, 2, [[2, 1], [1, 2]])
    # 创建一个 MatrixPSpace 对象 MP，使用 M 作为其分布
    MP = MatrixPSpace('M', M, 2, 2)
    # 断言 MP 的分布与 M 相同
    assert MP.distribution == M
    # 预期会抛出 ValueError 异常，因为参数不合法
    raises(ValueError, lambda: MatrixPSpace('M', M, 1.2, 2))

# 定义测试函数 test_MatrixGamma
def test_MatrixGamma():
    # 创建一个 MatrixGamma 对象 M
    M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
    # 断言 M 的概率空间的分布集合是一个实数矩阵集合
    assert M.pspace.distribution.set == MatrixSet(2, 2, S.Reals)
    # 断言 density(M) 返回的对象是 MatrixGammaDistribution 类的实例
    assert isinstance(density(M), MatrixGammaDistribution)
    # 创建一个 MatrixSymbol 对象 X
    X = MatrixSymbol('X', 2, 2)
    # 计算密度函数 density(M)(X) 的表达式并进行断言
    num = exp(Trace(Matrix([[-S(1)/2, 0], [0, -S(1)/2]])*X))
    assert density(M)(X).doit() == num/(4*pi*sqrt(Determinant(X)))
    # 检查 density(M)([[2, 1], [1, 2]]) 的值
    assert density(M)([[2, 1], [1, 2]]).doit() == sqrt(3)*exp(-2)/(12*pi)
    # 创建 MatrixSymbol 对象 X 和 Y
    X = MatrixSymbol('X', 1, 2)
    Y = MatrixSymbol('Y', 1, 2)
    # 检查 density(M)([X, Y]) 的值
    assert density(M)([X, Y]).doit() == exp(-X[0, 0]/2 - Y[0, 1]/2)/(4*pi*sqrt(
                                X[0, 0]*Y[0, 1] - X[0, 1]*Y[0, 0]))
    # 创建符号变量和 MatrixSymbol 对象
    a, b = symbols('a b', positive=True)
    d = symbols('d', positive=True, integer=True)
    Y = MatrixSymbol('Y', d, d)
    Z = MatrixSymbol('Z', 2, 2)
    SM = MatrixSymbol('SM', d, d)
    # 创建 MatrixGamma 对象 M2 和 M3，并进行密度函数表达式的断言
    M2 = MatrixGamma('M2', a, b, SM)
    M3 = MatrixGamma('M3', 2, 3, [[2, 1], [1, 2]])
    k = Dummy('k')
    exprd = pi**(-d*(d - 1)/4)*b**(-a*d)*exp(Trace((-1/b)*SM**(-1)*Y)
        )*Determinant(SM)**(-a)*Determinant(Y)**(a - d/2 - S(1)/2)/Product(
        gamma(-k/2 + a + S(1)/2), (k, 1, d))
    assert density(M2)(Y).dummy_eq(exprd)
    # 预期会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: density(M3 + M)(Z))
    # 预期会抛出 ValueError 异常，因为参数不合法
    raises(ValueError, lambda: density(M)(1))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, -2, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixGamma('M', -1, 2, [[1, 0], [0]]))

# 定义测试函数 test_Wishart
def test_Wishart():
    # 创建一个 Wishart 分布对象 W
    W = Wishart('W', 5, [[1, 0], [0, 1]])
    # 断言 W 的概率空间的分布集合是一个实数矩阵集合
    assert W.pspace.distribution.set == MatrixSet(2, 2, S.Reals)
    # 创建 MatrixSymbol 对象 X
    X = MatrixSymbol('X', 2, 2)
    # 计算密度函数 density(W)(X) 的表达式并进行断言
    term1 = exp(Trace(Matrix([[-S(1)/2, 0], [0, -S(1)/2]])*X))
    assert density(W)(X).doit() == term1 * Determinant(X)/(24*pi)
    # 检查 density(W)([[2, 1], [1, 2]]) 的值
    assert density(W)([[2, 1], [1, 2]]).doit() == exp(-2)/(8*pi)
    # 创建符号变量
    n = symbols('n', positive=True)
    d = symbols('d', positive=True, integer=True)
    Y = MatrixSymbol('Y', d, d)
    SM = MatrixSymbol('SM', d, d)
    # 创建一个 Wishart 分布对象 W，指定名称为 'W'，自由度为 n，尺度矩阵为 SM
    W = Wishart('W', n, SM)
    # 创建一个虚拟变量 k
    k = Dummy('k')
    # 计算 Wishart 分布的密度表达式 exprd
    exprd = 2**(-d*n/2)*pi**(-d*(d - 1)/4)*exp(Trace(-(S(1)/2)*SM**(-1)*Y)
    )*Determinant(SM)**(-n/2)*Determinant(Y)**(
    -d/2 + n/2 - S(1)/2)/Product(gamma(-k/2 + n/2 + S(1)/2), (k, 1, d))
    # 使用 assert 语句检查 Wishart 分布对象 W 在随机矩阵 Y 处的密度是否等于 exprd
    assert density(W)(Y).dummy_eq(exprd)
    # 使用 lambda 表达式和 raises 函数测试 density(W)(1) 是否会引发 ValueError 异常
    raises(ValueError, lambda: density(W)(1))
    # 使用 lambda 表达式和 raises 函数测试创建 Wishart 分布对象时传入无效参数会引发 ValueError 异常
    raises(ValueError, lambda: Wishart('W', -1, [[1, 0], [0, 1]]))
    raises(ValueError, lambda: Wishart('W', -1, [[1, 0], [2, 1]]))
    raises(ValueError, lambda: Wishart('W',  2, [[1, 0], [0]]))
# 定义一个测试函数，用于测试 MatrixNormal 分布的功能
def test_MatrixNormal():
    # 创建一个 MatrixNormal 对象 M，指定名称为 'M'，均值为 [[5, 6]]，标准差为 [4]，协方差为 [[2, 1], [1, 2]]
    M = MatrixNormal('M', [[5, 6]], [4], [[2, 1], [1, 2]])
    # 断言 M 的概率空间分布的集合属性为 MatrixSet(1, 2, S.Reals)
    assert M.pspace.distribution.set == MatrixSet(1, 2, S.Reals)
    
    # 定义一个 1x2 的 MatrixSymbol 对象 X
    X = MatrixSymbol('X', 1, 2)
    # 计算 term1 的值，这是一个复杂的数学表达式
    term1 = exp(-Trace(Matrix([[ S(2)/3, -S(1)/3], [-S(1)/3, S(2)/3]]) * (
            Matrix([[-5], [-6]]) + X.T) * Matrix([[S(1)/4]]) * (Matrix([[-5, -6]]) + X))/2)
    # 断言使用 M 的概率密度函数对 X 进行计算得到的结果等于给定的表达式
    assert density(M)(X).doit() == (sqrt(3)) * term1 / (24 * pi)
    
    # 断言使用 M 的概率密度函数对 [[7, 8]] 进行计算得到的结果
    assert density(M)([[7, 8]]).doit() == sqrt(3) * exp(-S(1)/3) / (24 * pi)
    
    # 定义两个符号 d 和 n，分别为正整数
    d, n = symbols('d n', positive=True, integer=True)
    # 创建两个 MatrixSymbol 对象 SM2 和 SM1，以及 LM 和 Y
    SM2 = MatrixSymbol('SM2', d, d)
    SM1 = MatrixSymbol('SM1', n, n)
    LM = MatrixSymbol('LM', n, d)
    Y = MatrixSymbol('Y', n, d)
    # 创建一个新的 MatrixNormal 对象 M，指定名称为 'M'，均值为 LM，标准差为 SM1，协方差为 SM2
    M = MatrixNormal('M', LM, SM1, SM2)
    # 计算 exprd 的值，这是一个复杂的数学表达式
    exprd = (2*pi)**(-d*n/2) * exp(-Trace(SM2**(-1) * (-LM.T + Y.T) * SM1**(-1) * (-LM + Y)) / 2) * \
            Determinant(SM1)**(-d/2) * Determinant(SM2)**(-n/2)
    # 断言使用 M 的概率密度函数对 Y 进行计算得到的结果等于给定的表达式
    assert density(M)(Y).doit() == exprd
    
    # 测试异常情况：传入非法参数 1 给 density 函数，预期抛出 ValueError
    raises(ValueError, lambda: density(M)(1))
    
    # 以下多行分别测试不同的非法参数情况，预期抛出 ValueError
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [0, 1]], [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2]], [[1, 0], [0, 1]]))
    raises(ValueError, lambda: MatrixNormal('M', [1, 2], [[1, 0], [2, 1]], [[1, 0], [0]]))
    raises(ValueError, lambda: MatrixNormal('M', [[1, 2]], [[1, 0], [0, 1]], [[1, 0]]))

# 定义一个测试函数，用于测试 MatrixStudentT 分布的功能
def test_MatrixStudentT():
    # 创建一个 MatrixStudentT 对象 M，指定名称为 'M'，自由度为 2，均值为 [[5, 6]]，协方差为 [[2, 1], [1, 2]]，缩放参数为 [4]
    M = MatrixStudentT('M', 2, [[5, 6]], [[2, 1], [1, 2]], [4])
    # 断言 M 的概率空间分布的集合属性为 MatrixSet(1, 2, S.Reals)
    assert M.pspace.distribution.set == MatrixSet(1, 2, S.Reals)
    
    # 定义一个 1x2 的 MatrixSymbol 对象 X
    X = MatrixSymbol('X', 1, 2)
    # 计算 D 的值，这是一个复杂的数学表达式
    D = pi ** (-1.0) * Determinant(Matrix([[4]])) ** (-1.0) * Determinant(Matrix([[2, 1], [1, 2]])) \
        ** (-0.5) / Determinant(Matrix([[S(1) / 4]]) * (Matrix([[-5, -6]]) + X) \
                                * Matrix([[S(2) / 3, -S(1) / 3], [-S(1) / 3, S(2) / 3]]) * (
                                        Matrix([[-5], [-6]]) + X.T) + Matrix([[1]])) ** 2
    # 断言使用 M 的概率密度函数对 X 进行计算得到的结果等于给定的表达式 D
    assert density(M)(X) == D
    
    # 定义一个符号 v，表示正数
    v = symbols('v', positive=True)
    n, p = 1, 2
    # 创建两个 MatrixSymbol 对象 Omega 和 Sigma，以及 Location 和 Y
    Omega = MatrixSymbol('Omega', p, p)
    Sigma = MatrixSymbol('Sigma', n, n)
    Location = MatrixSymbol('Location', n, p)
    Y = MatrixSymbol('Y', n, p)
    # 创建一个新的 MatrixStudentT 对象 M，指定名称为 'M'，自由度为 v，均值为 Location，协方差为 Omega，缩放参数为 Sigma
    M = MatrixStudentT('M', v, Location, Omega, Sigma)
    
    # 计算 exprd 的值，这是一个复杂的数学表达式
    exprd = gamma(v/2 + 1) * Determinant(Matrix([[1]]) + Sigma**(-1) * (-Location + Y) * Omega**(-1) * (-Location.T + Y.T))**(-v/2 - 1) / \
            (pi * gamma(v/2) * sqrt(Determinant(Omega)) * Determinant(Sigma))
    # 断言使用 M 的概率密度函数对 Y 进行计算得到的结果等于给定的表达式 exprd
    assert density(M)(Y) == exprd
    
    # 测试异常情况：传入非法参数 1 给 density 函数，预期抛出 ValueError
    raises(ValueError, lambda: density(M)(1))
    
    # 以下多行分别测试不同的非法参数情况，预期抛出 ValueError
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [0, 1]], [[1, 0], [2, 1]]))
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2, 1]], [[1, 0], [0, 1]]))
    # 抛出 ValueError 异常，测试 MatrixStudentT 类的参数验证
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
    # 抛出 ValueError 异常，测试 MatrixStudentT 类的参数验证
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2]], [[1, 0], [0, 1]]))
    # 抛出 ValueError 异常，测试 MatrixStudentT 类的参数验证
    raises(ValueError, lambda: MatrixStudentT('M', 1, [1, 2], [[1, 0], [2, 1]], [[1], [2]]))
    # 抛出 ValueError 异常，测试 MatrixStudentT 类的参数验证
    raises(ValueError, lambda: MatrixStudentT('M', 1, [[1, 2]], [[1, 0], [0, 1]], [[1, 0]]))
    # 抛出 ValueError 异常，测试 MatrixStudentT 类的参数验证
    raises(ValueError, lambda: MatrixStudentT('M', 1, [[1, 2]], [1], [[1, 0]]))
    # 抛出 ValueError 异常，测试 MatrixStudentT 类的参数验证
    raises(ValueError, lambda: MatrixStudentT('M', -1, [1, 2], [[1, 0], [0, 1]], [4]))
# 定义一个测试函数，用于测试使用 Scipy 库生成样本数据
def test_sample_scipy():
    # 创建两个 Scipy 库中的分布对象，一个是 MatrixNormal，另一个是 Wishart
    distribs_scipy = [
        MatrixNormal('M', [[5, 6]], [4], [[2, 1], [1, 2]]),  # 创建 MatrixNormal 分布对象
        Wishart('W', 5, [[1, 0], [0, 1]])  # 创建 Wishart 分布对象
    ]

    # 设置每个分布对象生成样本的大小为 5
    size = 5
    # 导入 Scipy 库
    scipy = import_module('scipy')
    # 如果 Scipy 未安装，则跳过测试，并提示 Scipy 未安装
    if not scipy:
        skip('Scipy not installed. Abort tests for _sample_scipy.')
    else:
        # 遍历每个分布对象
        for X in distribs_scipy:
            # 生成指定大小的样本集合
            samps = sample(X, size=size)
            # 验证每个样本是否在分布对象的样本空间中
            for sam in samps:
                assert Matrix(sam) in X.pspace.distribution.set
        # 创建一个 MatrixGamma 分布对象 M，并断言调用 sample 函数会抛出 NotImplementedError
        M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
        raises(NotImplementedError, lambda: sample(M, size=3))

# 定义一个测试函数，用于测试使用 PyMC 库生成样本数据
def test_sample_pymc():
    # 创建两个 PyMC 库中的分布对象，一个是 MatrixNormal，另一个是 Wishart
    distribs_pymc = [
        MatrixNormal('M', [[5, 6], [3, 4]], [[1, 0], [0, 1]], [[2, 1], [1, 2]]),  # 创建 MatrixNormal 分布对象
        Wishart('W', 7, [[2, 1], [1, 2]])  # 创建 Wishart 分布对象
    ]
    # 设置每个分布对象生成样本的大小为 3
    size = 3
    # 导入 PyMC 库
    pymc = import_module('pymc')
    # 如果 PyMC 未安装，则跳过测试，并提示 PyMC 未安装
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        # 遍历每个分布对象
        for X in distribs_pymc:
            # 生成指定大小的样本集合，并指定使用 PyMC 库
            samps = sample(X, size=size, library='pymc')
            # 验证每个样本是否在分布对象的样本空间中
            for sam in samps:
                assert Matrix(sam) in X.pspace.distribution.set
        # 创建一个 MatrixGamma 分布对象 M，并断言调用 sample 函数会抛出 NotImplementedError
        M = MatrixGamma('M', 1, 2, [[1, 0], [0, 1]])
        raises(NotImplementedError, lambda: sample(M, size=3))

# 定义一个测试函数，用于测试设置随机种子的情况下生成样本数据
def test_sample_seed():
    # 创建一个 MatrixNormal 分布对象 X
    X = MatrixNormal('M', [[5, 6], [3, 4]], [[1, 0], [0, 1]], [[2, 1], [1, 2]])

    # 定义需要测试的库列表
    libraries = ['scipy', 'numpy', 'pymc']
    # 遍历每个库
    for lib in libraries:
        try:
            # 导入指定库
            imported_lib = import_module(lib)
            # 如果导入成功
            if imported_lib:
                # 初始化三个空列表 s0, s1, s2
                s0, s1, s2 = [], [], []
                # 生成使用相同种子的三组样本数据集合
                s0 = sample(X, size=10, library=lib, seed=0)
                s1 = sample(X, size=10, library=lib, seed=0)
                s2 = sample(X, size=10, library=lib, seed=1)
                # 遍历每个样本集合中的样本数据
                for i in range(10):
                    # 断言第一组和第二组的样本数据完全相同
                    assert (s0[i] == s1[i]).all()
                    # 断言第一组和第三组的样本数据不完全相同
                    assert (s1[i] != s2[i]).all()

        # 捕获 NotImplementedError 异常，并继续执行下一个库的测试
        except NotImplementedError:
            continue
```