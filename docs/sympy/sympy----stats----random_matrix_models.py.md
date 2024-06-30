# `D:\src\scipysrc\sympy\sympy\stats\random_matrix_models.py`

```
# 导入具体类和函数，以便在代码中使用
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.core.sympify import _sympify
from sympy.stats.rv import _symbol_converter, Density, RandomMatrixSymbol, is_random
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.random_matrix import RandomMatrixPSpace
from sympy.tensor.array import ArrayComprehension

# 定义 __all__ 列表，指定模块导入时允许导入的公共标识符
__all__ = [
    'CircularEnsemble',
    'CircularUnitaryEnsemble',
    'CircularOrthogonalEnsemble',
    'CircularSymplecticEnsemble',
    'GaussianEnsemble',
    'GaussianUnitaryEnsemble',
    'GaussianOrthogonalEnsemble',
    'GaussianSymplecticEnsemble',
    'joint_eigen_distribution',
    'JointEigenDistribution',
    'level_spacing_distribution'
]

# 注册 RandomMatrixSymbol 类，使其被识别为随机变量
@is_random.register(RandomMatrixSymbol)
def _(x):
    return True


class RandomMatrixEnsembleModel(Basic):
    """
    Base class for random matrix ensembles.
    It acts as an umbrella and contains
    the methods common to all the ensembles
    defined in sympy.stats.random_matrix_models.
    """
    
    def __new__(cls, sym, dim=None):
        # 将符号和维度参数转换为适当的形式
        sym, dim = _symbol_converter(sym), _sympify(dim)
        # 如果维度不是整数，则引发错误
        if dim.is_integer == False:
            raise ValueError("Dimension of the random matrices must be "
                             "integers, received %s instead."%(dim))
        return Basic.__new__(cls, sym, dim)

    # 符号属性，返回实例的第一个参数
    symbol = property(lambda self: self.args[0])
    # 维度属性，返回实例的第二个参数
    dimension = property(lambda self: self.args[1])

    # 创建并返回一个 Density 对象，用于表示密度函数
    def density(self, expr):
        return Density(expr)

    # 调用 density 方法，用于计算密度函数
    def __call__(self, expr):
        return self.density(expr)


class GaussianEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Gaussian ensembles.
    Contains the properties common to all the
    gaussian ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Gaussian_ensembles
    .. [2] https://arxiv.org/pdf/1712.07903.pdf
    """
    def _compute_normalization_constant(self, beta, n):
        """
        Helper function for computing normalization
        constant for joint probability density of eigen
        values of Gaussian ensembles.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Selberg_integral#Mehta's_integral
        """
        # 将 n 转换为 SymPy 的符号类型 S(n)
        n = S(n)
        # 定义一个 lambda 函数 prod_term，计算 gamma 函数的乘积
        prod_term = lambda j: gamma(1 + beta*S(j)/2)/gamma(S.One + beta/S(2))
        # 定义符号 j 为整数且为正数的虚拟变量
        j = Dummy('j', integer=True, positive=True)
        # 计算第一个 term，是 gamma 函数乘积的积分结果
        term1 = Product(prod_term(j), (j, 1, n)).doit()
        # 计算第二个 term，是公式中的 (2/(beta*n))**(...) 部分
        term2 = (2/(beta*n))**(beta*n*(n - 1)/4 + n/2)
        # 计算第三个 term，是公式中的 (2*pi)**(n/2) 部分
        term3 = (2*pi)**(n/2)
        # 返回三个 term 的乘积作为归一化常数
        return term1 * term2 * term3

    def _compute_joint_eigen_distribution(self, beta):
        """
        Helper function for computing the joint
        probability distribution of eigen values
        of the random matrix.
        """
        # 获取矩阵的维度 n
        n = self.dimension
        # 计算 normalization constant Zbn
        Zbn = self._compute_normalization_constant(beta, n)
        # 定义符号 l 为 IndexedBase 类型
        l = IndexedBase('l')
        # 定义符号 i 为整数且为正数的虚拟变量
        i = Dummy('i', integer=True, positive=True)
        # 定义符号 j 和 k 为整数且为正数的虚拟变量
        j = Dummy('j', integer=True, positive=True)
        k = Dummy('k', integer=True, positive=True)
        # 计算第一个 term，是指数部分 exp((-S(n)/2) * Sum(l[k]**2, (k, 1, n)).doit())
        term1 = exp((-S(n)/2) * Sum(l[k]**2, (k, 1, n)).doit())
        # 定义 sub_term 为 lambda 函数，计算 Abs(l[j] - l[i])**beta 的乘积
        sub_term = Lambda(i, Product(Abs(l[j] - l[i])**beta, (j, i + 1, n)))
        # 计算第二个 term，是指数部分的乘积 Product(sub_term(i).doit(), (i, 1, n - 1)).doit()
        term2 = Product(sub_term(i).doit(), (i, 1, n - 1)).doit()
        # 定义符号 syms 为 l[k] 的数组推导式
        syms = ArrayComprehension(l[k], (k, 1, n)).doit()
        # 返回 lambda 函数，表示联合特征值分布的计算结果
        return Lambda(tuple(syms), (term1 * term2)/Zbn)
class GaussianUnitaryEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        # 返回高斯酉集合的归一化常数
        n = self.dimension
        return 2**(S(n)/2) * pi**(S(n**2)/2)

    def density(self, expr):
        # 计算高斯酉集合的密度函数
        n, ZGUE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n)/2 * Trace(H**2))/ZGUE)(expr)

    def joint_eigen_distribution(self):
        # 计算高斯酉集合的联合特征值分布
        return self._compute_joint_eigen_distribution(S(2))

    def level_spacing_distribution(self):
        # 返回高斯酉集合的级间距分布函数
        s = Dummy('s')
        f = (32/pi**2)*(s**2)*exp((-4/pi)*s**2)
        return Lambda(s, f)

class GaussianOrthogonalEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        # 返回高斯正交集合的归一化常数
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n)/4 * Trace(_H**2)))

    def density(self, expr):
        # 计算高斯正交集合的密度函数
        n, ZGOE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n)/4 * Trace(H**2))/ZGOE)(expr)

    def joint_eigen_distribution(self):
        # 计算高斯正交集合的联合特征值分布
        return self._compute_joint_eigen_distribution(S.One)

    def level_spacing_distribution(self):
        # 返回高斯正交集合的级间距分布函数
        s = Dummy('s')
        f = (pi/2)*s*exp((-pi/4)*s**2)
        return Lambda(s, f)

class GaussianSymplecticEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        # 返回高斯辛集合的归一化常数
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n) * Trace(_H**2)))

    def density(self, expr):
        # 计算高斯辛集合的密度函数
        n, ZGSE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n) * Trace(H**2))/ZGSE)(expr)

    def joint_eigen_distribution(self):
        # 计算高斯辛集合的联合特征值分布
        return self._compute_joint_eigen_distribution(S(4))

    def level_spacing_distribution(self):
        # 返回高斯辛集合的级间距分布函数
        s = Dummy('s')
        f = ((S(2)**18)/((S(3)**6)*(pi**3)))*(s**4)*exp((-64/(9*pi))*s**2)
        return Lambda(s, f)

def GaussianEnsemble(sym, dim):
    # 创建一个随机矩阵符号表示的高斯集合
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianUnitaryEnsemble(sym, dim):
    """
    表示高斯酉集合。

    Examples
    ========

    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE, density
    >>> from sympy import MatrixSymbol
    >>> G = GUE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-Trace(X**2))/(2*pi**2)
    """
    # 创建一个高斯酉集合
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianUnitaryEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)
def GaussianOrthogonalEnsemble(sym, dim):
    """
    Represents Gaussian Orthogonal Ensembles.

    Examples
    ========

    >>> from sympy.stats import GaussianOrthogonalEnsemble as GOE, density
    >>> from sympy import MatrixSymbol
    >>> G = GOE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-Trace(X**2)/2)/Integral(exp(-Trace(_H**2)/2), _H)
    """
    # 将符号和维度转换为适当的符号和Sympy表达式
    sym, dim = _symbol_converter(sym), _sympify(dim)
    # 创建高斯正交集合模型
    model = GaussianOrthogonalEnsembleModel(sym, dim)
    # 创建随机矩阵概率空间对象
    rmp = RandomMatrixPSpace(sym, model=model)
    # 返回随机矩阵符号对象
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianSymplecticEnsemble(sym, dim):
    """
    Represents Gaussian Symplectic Ensembles.

    Examples
    ========

    >>> from sympy.stats import GaussianSymplecticEnsemble as GSE, density
    >>> from sympy import MatrixSymbol
    >>> G = GSE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-2*Trace(X**2))/Integral(exp(-2*Trace(_H**2)), _H)
    """
    # 将符号和维度转换为适当的符号和Sympy表达式
    sym, dim = _symbol_converter(sym), _sympify(dim)
    # 创建高斯辛集合模型
    model = GaussianSymplecticEnsembleModel(sym, dim)
    # 创建随机矩阵概率空间对象
    rmp = RandomMatrixPSpace(sym, model=model)
    # 返回随机矩阵符号对象
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

class CircularEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Circular ensembles.
    Contains the properties and methods
    common to all the circular ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Circular_ensemble
    """
    def density(self, expr):
        # TODO : Add support for Lie groups(as extensions of sympy.diffgeom)
        #        and define measures on them
        # 抛出未实现错误，因为尚未支持Haar测度
        raise NotImplementedError("Support for Haar measure hasn't been "
                                  "implemented yet, therefore the density of "
                                  "%s cannot be computed."%(self))

    def _compute_joint_eigen_distribution(self, beta):
        """
        Helper function to compute the joint distribution of phases
        of the complex eigen values of matrices belonging to any
        circular ensembles.
        """
        # 获取维度大小
        n = self.dimension
        # 计算Zbn参数
        Zbn = ((2*pi)**n)*(gamma(beta*n/2 + 1)/S(gamma(beta/2 + 1))**n)
        # 定义指数化的基础
        t = IndexedBase('t')
        # 定义用于迭代的虚拟变量
        i, j, k = (Dummy('i', integer=True), Dummy('j', integer=True),
                   Dummy('k', integer=True))
        # 创建符号数组
        syms = ArrayComprehension(t[i], (i, 1, n)).doit()
        # 定义内部关系
        f = Product(Product(Abs(exp(I*t[k]) - exp(I*t[j]))**beta, (j, k + 1, n)).doit(),
                    (k, 1, n - 1)).doit()
        # 返回模板表达式
        return Lambda(tuple(syms), f/Zbn)

class CircularUnitaryEnsembleModel(CircularEnsembleModel):
    def joint_eigen_distribution(self):
        # 返回S的值
        return self._compute_joint_eigen_distribution(S(2))

class CircularOrthogonalEnsembleModel(CircularEnsembleModel):
    def joint_eigen_distribution(self):
        # 返回S.One的值
        return self._compute_joint_eigen_distribution(S.One)

class CircularSymplecticEnsembleModel(CircularEnsembleModel):
    #
    
# 定义函数 CircularEnsemble，表示循环集合
def CircularEnsemble(sym, dim):
    # 将符号 sym 和维度 dim 转换为适当的表示
    sym, dim = _symbol_converter(sym), _sympify(dim)
    # 创建 CircularEnsembleModel 模型对象
    model = CircularEnsembleModel(sym, dim)
    # 创建 RandomMatrixPSpace 随机矩阵概率空间对象
    rmp = RandomMatrixPSpace(sym, model=model)
    # 返回 RandomMatrixSymbol 随机矩阵符号对象
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

# 定义函数 CircularUnitaryEnsemble，表示循环酉集合
def CircularUnitaryEnsemble(sym, dim):
    """
    表示循环酉集合。

    Examples
    ========

    >>> from sympy.stats import CircularUnitaryEnsemble as CUE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = CUE('U', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**2, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    正如上例所示，循环酉集合的密度未被评估，因为其确切定义基于酉群的哈尔测度，这是不唯一的。
    """
    # 将符号 sym 和维度 dim 转换为适当的表示
    sym, dim = _symbol_converter(sym), _sympify(dim)
    # 创建 CircularUnitaryEnsembleModel 模型对象
    model = CircularUnitaryEnsembleModel(sym, dim)
    # 创建 RandomMatrixPSpace 随机矩阵概率空间对象
    rmp = RandomMatrixPSpace(sym, model=model)
    # 返回 RandomMatrixSymbol 随机矩阵符号对象
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

# 定义函数 CircularOrthogonalEnsemble，表示循环正交集合
def CircularOrthogonalEnsemble(sym, dim):
    """
    表示循环正交集合。

    Examples
    ========

    >>> from sympy.stats import CircularOrthogonalEnsemble as COE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = COE('O', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k])), (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    正如上例所示，循环正交集合的密度未被评估，因为其确切定义基于正交群的哈尔测度，这是不唯一的。
    """
    # 将符号 sym 和维度 dim 转换为适当的表示
    sym, dim = _symbol_converter(sym), _sympify(dim)
    # 创建 CircularOrthogonalEnsembleModel 模型对象
    model = CircularOrthogonalEnsembleModel(sym, dim)
    # 创建 RandomMatrixPSpace 随机矩阵概率空间对象
    rmp = RandomMatrixPSpace(sym, model=model)
    # 返回 RandomMatrixSymbol 随机矩阵符号对象
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

# 定义函数 CircularSymplecticEnsemble，表示循环辛集合
def CircularSymplecticEnsemble(sym, dim):
    """
    表示循环辛集合。

    Examples
    ========

    >>> from sympy.stats import CircularSymplecticEnsemble as CSE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = CSE('S', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**4, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    正如上例所示，循环辛集合的密度未被评估，因为其确切定义基于辛群的哈尔测度，这是不唯一的。
    """
    # 将符号 sym 和维度 dim 转换为适当的表示
    sym, dim = _symbol_converter(sym), _sympify(dim)
    # 创建 CircularSymplecticEnsembleModel 模型对象
    model = CircularSymplecticEnsembleModel(sym, dim)
    # 创建 RandomMatrixPSpace 随机矩阵概率空间对象
    rmp = RandomMatrixPSpace(sym, model=model)
    # 返回 RandomMatrixSymbol 随机矩阵符号对象
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

# 定义函数 joint_eigen_distribution，用于获取随机矩阵的特征值的联合概率分布
def joint_eigen_distribution(mat):
    """
    用于获取随机矩阵特征值的联合概率分布。

    Parameters
    ==========

    mat: RandomMatrixSymbol
        要考虑其特征值的矩阵符号。

    """
    如果输入的 mat 不是 RandomMatrixSymbol 类型的对象，则抛出 ValueError 异常
    raise ValueError("%s is not of type, RandomMatrixSymbol."%(mat))
    调用 RandomMatrixSymbol 对象的 pspace 属性获取概率空间，然后调用 model 属性获取模型，最后调用 joint_eigen_distribution 方法得到联合特征值分布函数
    return mat.pspace.model.joint_eigen_distribution()
# 定义一个函数，用于创建随机表达式矩阵的特征值的联合分布
def JointEigenDistribution(mat):
    """
    Creates joint distribution of eigen values of matrices with random
    expressions.

    Parameters
    ==========

    mat: Matrix
        The matrix under consideration.

    Returns
    =======

    JointDistributionHandmade

    Examples
    ========

    >>> from sympy.stats import Normal, JointEigenDistribution
    >>> from sympy import Matrix
    >>> A = [[Normal('A00', 0, 1), Normal('A01', 0, 1)],
    ... [Normal('A10', 0, 1), Normal('A11', 0, 1)]]
    >>> JointEigenDistribution(Matrix(A))
    JointDistributionHandmade(-sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2
    + A00/2 + A11/2, sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2 + A00/2 + A11/2)

    """
    # 获取矩阵的特征值，允许多个特征值
    eigenvals = mat.eigenvals(multiple=True)
    # 检查所有特征值是否都包含随机表达式
    if not all(is_random(eigenval) for eigenval in set(eigenvals)):
        # 若特征值中有任何一个不包含随机表达式，抛出数值错误异常
        raise ValueError("Eigen values do not have any random expression, "
                         "joint distribution cannot be generated.")
    # 返回特征值的手工联合分布
    return JointDistributionHandmade(*eigenvals)

# 定义一个函数，用于获取级别间隔的分布
def level_spacing_distribution(mat):
    """
    For obtaining distribution of level spacings.

    Parameters
    ==========

    mat: RandomMatrixSymbol
        The random matrix symbol whose eigen values are
        to be considered for finding the level spacings.

    Returns
    =======

    Lambda

    Examples
    ========

    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE
    >>> from sympy.stats import level_spacing_distribution
    >>> U = GUE('U', 2)
    >>> level_spacing_distribution(U)
    Lambda(_s, 32*_s**2*exp(-4*_s**2/pi)/pi**2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Distribution_of_level_spacings
    """
    # 返回随机矩阵符号的概率空间模型中的级别间隔分布
    return mat.pspace.model.level_spacing_distribution()
```