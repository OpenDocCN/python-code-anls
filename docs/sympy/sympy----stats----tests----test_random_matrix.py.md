# `D:\src\scipysrc\sympy\sympy\stats\tests\test_random_matrix.py`

```
# 从 sympy.concrete.products 模块中导入 Product 类
from sympy.concrete.products import Product
# 从 sympy.core.function 模块中导入 Lambda 类
from sympy.core.function import Lambda
# 从 sympy.core.numbers 模块中导入 I, Rational, pi 类
from sympy.core.numbers import (I, Rational, pi)
# 从 sympy.core.singleton 模块中导入 S 类
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块中导入 Dummy 类
from sympy.core.symbol import Dummy
# 从 sympy.functions.elementary.complexes 模块中导入 Abs 函数
from sympy.functions.elementary.complexes import Abs
# 从 sympy.functions.elementary.exponential 模块中导入 exp 函数
from sympy.functions.elementary.exponential import exp
# 从 sympy.functions.elementary.miscellaneous 模块中导入 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.integrals.integrals 模块中导入 Integral 类
from sympy.integrals.integrals import Integral
# 从 sympy.matrices.dense 模块中导入 Matrix 类
from sympy.matrices.dense import Matrix
# 从 sympy.matrices.expressions.matexpr 模块中导入 MatrixSymbol 类
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 从 sympy.matrices.expressions.trace 模块中导入 Trace 类
from sympy.matrices.expressions.trace import Trace
# 从 sympy.tensor.indexed 模块中导入 IndexedBase 类
from sympy.tensor.indexed import IndexedBase
# 从 sympy.stats 模块中导入以下类和函数
from sympy.stats import (GaussianUnitaryEnsemble as GUE, density,
                         GaussianOrthogonalEnsemble as GOE,
                         GaussianSymplecticEnsemble as GSE,
                         joint_eigen_distribution,
                         CircularUnitaryEnsemble as CUE,
                         CircularOrthogonalEnsemble as COE,
                         CircularSymplecticEnsemble as CSE,
                         JointEigenDistribution,
                         level_spacing_distribution,
                         Normal, Beta)
# 从 sympy.stats.joint_rv_types 模块中导入 JointDistributionHandmade 类
from sympy.stats.joint_rv_types import JointDistributionHandmade
# 从 sympy.stats.rv 模块中导入 RandomMatrixSymbol 类
from sympy.stats.rv import RandomMatrixSymbol
# 从 sympy.stats.random_matrix_models 模块中导入 GaussianEnsemble, RandomMatrixPSpace 类
from sympy.stats.random_matrix_models import GaussianEnsemble, RandomMatrixPSpace
# 从 sympy.testing.pytest 模块中导入 raises 函数
from sympy.testing.pytest import raises

# 定义测试函数 test_GaussianEnsemble
def test_GaussianEnsemble():
    # 创建一个高斯集成对象 G，包含 3 个维度
    G = GaussianEnsemble('G', 3)
    # 断言 G 的概率密度函数等于 G 的概率空间模型
    assert density(G) == G.pspace.model
    # 使用 lambda 表达式测试，预期会抛出 ValueError 异常
    raises(ValueError, lambda: GaussianEnsemble('G', 3.5))

# 定义测试函数 test_GaussianUnitaryEnsemble
def test_GaussianUnitaryEnsemble():
    # 创建一个随机矩阵符号 H，是一个 3x3 的矩阵
    H = RandomMatrixSymbol('H', 3, 3)
    # 创建一个高斯酉集成对象 G，包含 3 个维度
    G = GUE('U', 3)
    # 断言 G 的概率密度函数应用于 H 的结果等于特定表达式
    assert density(G)(H) == sqrt(2)*exp(-3*Trace(H**2)/2)/(4*pi**Rational(9, 2))
    # 创建两个虚拟符号 i 和 j，表示整数且为正数
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    # 创建一个 IndexedBase 符号 l
    l = IndexedBase('l')
    # 断言 G 的联合特征值分布与特定表达式相等
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            27*sqrt(6)*exp(-3*(l[1]**2)/2 - 3*(l[2]**2)/2 - 3*(l[3]**2)/2)*
            Product(Abs(l[i] - l[j])**2, (j, i + 1, 3), (i, 1, 2))/(16*pi**Rational(3, 2))))
    # 创建一个虚拟符号 s
    s = Dummy('s')
    # 断言 G 的能级间距分布与特定表达式相等
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, 32*s**2*exp(-4*s**2/pi)/pi**2))

# 定义测试函数 test_GaussianOrthogonalEnsemble
def test_GaussianOrthogonalEnsemble():
    # 创建一个随机矩阵符号 H，是一个 3x3 的矩阵
    H = RandomMatrixSymbol('H', 3, 3)
    # 创建一个高斯正交集成对象 G，包含 3 个维度
    G = GOE('O', 3)
    # 断言 G 的概率密度函数应用于 H 的结果等于特定表达式
    assert density(G)(H) == exp(-3*Trace(H**2)/4)/Integral(exp(-3*Trace(_H**2)/4), _H)
    # 创建两个虚拟符号 i 和 j，表示整数且为正数
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    # 创建一个 IndexedBase 符号 l
    l = IndexedBase('l')
    # 断言 G 的联合特征值分布与特定表达式相等
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            9*sqrt(2)*exp(-3*l[1]**2/2 - 3*l[2]**2/2 - 3*l[3]**2/2)*
            Product(Abs(l[i] - l[j]), (j, i + 1, 3), (i, 1, 2))/(32*pi)))
    # 创建一个虚拟符号 s
    s = Dummy('s')
    # 断言 G 的能级间距分布与特定表达式相等
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, s*pi*exp(-s**2*pi/4)/2))

# 定义测试函数 test_GaussianSymplecticEnsemble
def test_GaussianSymplecticEnsemble():
    # 此函数未完成，需要进一步补充
    # 定义一个随机矩阵符号 H，是一个 3x3 的矩阵符号
    H = RandomMatrixSymbol('H', 3, 3)
    # 定义另一个矩阵符号 _H，也是一个 3x3 的矩阵符号
    _H = MatrixSymbol('_H', 3, 3)
    # 创建一个 GSE 对象 'O'，代表一个三维的 GSE（Gaussian Symplectic Ensemble）
    G = GSE('O', 3)
    # 断言密度函数 density(G) 在矩阵 H 上的计算结果等于指定的表达式
    assert density(G)(H) == exp(-3*Trace(H**2))/Integral(exp(-3*Trace(_H**2)), _H)
    # 定义两个整数虚拟变量 i 和 j，分别为正整数
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    # 创建一个索引基 l
    l = IndexedBase('l')
    # 断言联合特征分布 joint_eigen_distribution(G) 等同于指定的 Lambda 函数
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            162*sqrt(3)*exp(-3*l[1]**2/2 - 3*l[2]**2/2 - 3*l[3]**2/2)*
            Product(Abs(l[i] - l[j])**4, (j, i + 1, 3), (i, 1, 2))/(5*pi**Rational(3, 2))))
    # 创建一个虚拟变量 s
    s = Dummy('s')
    # 断言级距分布函数 level_spacing_distribution(G) 等同于指定的 Lambda 函数
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, S(262144)*s**4*exp(-64*s**2/(9*pi))/(729*pi**3)))
# 定义测试函数，用于测试 Circular Unitary Ensemble (CUE) 的性质
def test_CircularUnitaryEnsemble():
    # 创建一个 CUE 对象，参数为 'U' 和 3
    CU = CUE('U', 3)
    # 定义两个虚拟变量 j 和 k，均为正整数
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    # 创建一个 IndexedBase 对象 t
    t = IndexedBase('t')
    # 断言语句：验证 joint_eigen_distribution(CU) 的结果是否与 Lambda 函数 dummy_eq
    assert joint_eigen_distribution(CU).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            # 计算 Product 内部的表达式，这里是绝对值的平方
            Product(Abs(exp(I*t[j]) - exp(I*t[k]))**2,
            # j 从 k+1 到 3，k 从 1 到 2 的乘积
            (j, k + 1, 3), (k, 1, 2))/(48*pi**3))
    )

# 定义测试函数，用于测试 Circular Orthogonal Ensemble (COE) 的性质
def test_CircularOrthogonalEnsemble():
    # 创建一个 COE 对象，参数为 'U' 和 3
    CO = COE('U', 3)
    # 定义两个虚拟变量 j 和 k，均为正整数
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    # 创建一个 IndexedBase 对象 t
    t = IndexedBase('t')
    # 断言语句：验证 joint_eigen_distribution(CO) 的结果是否与 Lambda 函数 dummy_eq
    assert joint_eigen_distribution(CO).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            # 计算 Product 内部的表达式，这里是绝对值
            Product(Abs(exp(I*t[j]) - exp(I*t[k])),
            # j 从 k+1 到 3，k 从 1 到 2 的乘积
            (j, k + 1, 3), (k, 1, 2))/(48*pi**2))
    )

# 定义测试函数，用于测试 Circular Symplectic Ensemble (CSE) 的性质
def test_CircularSymplecticEnsemble():
    # 创建一个 CSE 对象，参数为 'U' 和 3
    CS = CSE('U', 3)
    # 定义两个虚拟变量 j 和 k，均为正整数
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    # 创建一个 IndexedBase 对象 t
    t = IndexedBase('t')
    # 断言语句：验证 joint_eigen_distribution(CS) 的结果是否与 Lambda 函数 dummy_eq
    assert joint_eigen_distribution(CS).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            # 计算 Product 内部的表达式，这里是绝对值的四次方
            Product(Abs(exp(I*t[j]) - exp(I*t[k]))**4,
            # j 从 k+1 到 3，k 从 1 到 2 的乘积
            (j, k + 1, 3), (k, 1, 2))/(720*pi**3))
    )

# 定义测试函数，用于测试 JointEigenDistribution 的性质
def test_JointEigenDistribution():
    # 创建一个 2x2 的随机矩阵 A，元素为正态分布和贝塔分布
    A = Matrix([[Normal('A00', 0, 1), Normal('A01', 1, 1)],
                [Beta('A10', 1, 1), Beta('A11', 1, 1)]])
    # 断言语句：验证 JointEigenDistribution(A) 是否等于手动创建的 JointDistributionHandmade
    assert JointEigenDistribution(A) == \
    JointDistributionHandmade(-sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] + A[1, 1]**2)/2 +
    A[0, 0]/2 + A[1, 1]/2, sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] + A[1, 1]**2)/2 + A[0, 0]/2 + A[1, 1]/2)
    # 断言语句：验证当输入的矩阵不满足特定条件时，会引发 ValueError
    raises(ValueError, lambda: JointEigenDistribution(Matrix([[1, 0], [2, 1]])))

# 定义测试函数，用于测试 issue 19841
def test_issue_19841():
    # 创建一个 2x2 的 GUE 对象
    G1 = GUE('U', 2)
    # 使用 xreplace 将 G1 中的 2 替换为 2，生成 G2
    G2 = G1.xreplace({2: 2})
    # 断言语句：验证 G1 和 G2 的参数列表是否相同
    assert G1.args == G2.args

    # 创建一个 2x2 的 MatrixSymbol 对象 X
    X = MatrixSymbol('X', 2, 2)
    # 创建一个 2x2 的 GSE 对象 G
    G = GSE('U', 2)
    # 创建一个随机矩阵概率空间对象 h_pspace，其模型为 G 的密度函数
    h_pspace = RandomMatrixPSpace('P', model=density(G))
    # 创建一个随机矩阵符号 H，维度为 2x2，概率空间为 h_pspace
    H = RandomMatrixSymbol('H', 2, 2, pspace=h_pspace)
    # 创建一个随机矩阵符号 H2，维度为 2x2，概率空间为 None
    H2 = RandomMatrixSymbol('H', 2, 2, pspace=None)
    # 断言语句：验证 doit() 方法是否返回 H 本身
    assert H.doit() == H

    # 断言语句：验证在表达式中使用 xreplace 将 H 替换为 X 是否得到正确结果
    assert (2*H).xreplace({H: X}) == 2*X
    # 断言语句：验证在表达式中使用 xreplace 将 H2 替换为 X 是否得到正确结果
    assert (2*H).xreplace({H2: X}) == 2*H
    # 断言语句：验证在表达式中使用 xreplace 将 H 替换为 X 是否得到正确结果
    assert (2*H2).xreplace({H: X}) == 2*H2
    # 断言语句：验证在表达式中使用 xreplace 将 H2 替换为 X 是否得到正确结果
    assert (2*H2).xreplace({H2: X}) == 2*X
```