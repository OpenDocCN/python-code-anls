# `D:\src\scipysrc\sympy\sympy\stats\matrix_distributions.py`

```
from math import prod  # 导入 math 模块中的 prod 函数

from sympy.core.basic import Basic  # 导入 sympy.core.basic 模块中的 Basic 类
from sympy.core.numbers import pi  # 导入 sympy.core.numbers 模块中的 pi 常数
from sympy.core.singleton import S  # 导入 sympy.core.singleton 模块中的 S 单例
from sympy.functions.elementary.exponential import exp  # 导入 sympy.functions.elementary.exponential 模块中的 exp 函数
from sympy.functions.special.gamma_functions import multigamma  # 导入 sympy.functions.special.gamma_functions 模块中的 multigamma 函数
from sympy.core.sympify import sympify, _sympify  # 导入 sympy.core.sympify 模块中的 sympify 和 _sympify 函数
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,  # 导入 sympy.matrices 模块中的多个类和函数
                            MatrixSymbol, MatrixBase, Transpose, MatrixSet,
                            matrix2numpy)
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,  # 导入 sympy.stats.rv 模块中的多个类和函数
                            _symbol_converter, MatrixDomain, Distribution)
from sympy.external import import_module  # 导入 sympy.external 模块中的 import_module 函数


################################################################################
#------------------------Matrix Probability Space------------------------------#
################################################################################
class MatrixPSpace(PSpace):
    """
    表示矩阵分布的概率空间。
    """
    def __new__(cls, sym, distribution, dim_n, dim_m):
        sym = _symbol_converter(sym)  # 将符号转换为合适的类型
        dim_n, dim_m = _sympify(dim_n), _sympify(dim_m)  # 将维度转换为 sympy 符号类型
        if not (dim_n.is_integer and dim_m.is_integer):  # 检查维度是否为整数
            raise ValueError("Dimensions should be integers")  # 抛出值错误异常，提示维度应为整数
        return Basic.__new__(cls, sym, distribution, dim_n, dim_m)  # 调用基类的构造方法创建新的对象

    distribution = property(lambda self: self.args[1])  # 定义 distribution 属性，返回参数中的分布对象
    symbol = property(lambda self: self.args[0])  # 定义 symbol 属性，返回参数中的符号对象

    @property
    def domain(self):
        return MatrixDomain(self.symbol, self.distribution.set)  # 返回矩阵域对象，使用符号和分布的集合

    @property
    def value(self):
        return RandomMatrixSymbol(self.symbol, self.args[2], self.args[3], self)  # 返回随机矩阵符号对象，使用符号和维度参数

    @property
    def values(self):
        return {self.value}  # 返回包含当前 value 属性的集合

    def compute_density(self, expr, *args):
        rms = expr.atoms(RandomMatrixSymbol)  # 获取表达式中所有的随机矩阵符号
        if len(rms) > 1 or (not isinstance(expr, RandomMatrixSymbol)):  # 如果表达式包含多个矩阵符号或者不是单一矩阵符号
            raise NotImplementedError("Currently, no algorithm has been "
                                      "implemented to handle general expressions containing "
                                      "multiple matrix distributions.")  # 抛出未实现错误，暂时不支持处理包含多个矩阵分布的一般表达式
        return self.distribution.pdf(expr)  # 返回表达式在当前分布下的概率密度函数值

    def sample(self, size=(), library='scipy', seed=None):
        """
        内部采样方法

        返回字典，将随机矩阵符号映射到实现值。
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}  # 返回包含随机矩阵符号和样本值的字典


def rv(symbol, cls, args):
    args = list(map(sympify, args))  # 将参数列表中的每个参数转换为 sympy 符号类型
    dist = cls(*args)  # 使用参数创建分布对象
    dist.check(*args)  # 检查分布参数的有效性
    dim = dist.dimension  # 获取分布的维度
    pspace = MatrixPSpace(symbol, dist, dim[0], dim[1])  # 创建矩阵概率空间对象
    return pspace.value  # 返回矩阵概率空间的值


class SampleMatrixScipy:
    """返回给定分布的 scipy 采样样本"""
    def __new__(cls, dist, size, seed=None):
        return cls._sample_scipy(dist, size, seed)  # 调用类方法 _sample_scipy 返回采样结果

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

        # 导入需要的库
        from scipy import stats as scipy_stats
        import numpy

        # 定义映射字典，用于根据分布类型选择相应的随机抽样函数
        scipy_rv_map = {
            'WishartDistribution': lambda dist, size, rand_state: scipy_stats.wishart.rvs(
                df=int(dist.n), scale=matrix2numpy(dist.scale_matrix, float), size=size),
            'MatrixNormalDistribution': lambda dist, size, rand_state: scipy_stats.matrix_normal.rvs(
                mean=matrix2numpy(dist.location_matrix, float),
                rowcov=matrix2numpy(dist.scale_matrix_1, float),
                colcov=matrix2numpy(dist.scale_matrix_2, float), size=size, random_state=rand_state)
        }

        # 定义获取样本形状的字典，根据分布类型返回相应的形状
        sample_shape = {
            'WishartDistribution': lambda dist: dist.scale_matrix.shape,
            'MatrixNormalDistribution' : lambda dist: dist.location_matrix.shape
        }

        # 获取所有支持的分布名称列表
        dist_list = scipy_rv_map.keys()

        # 如果给定的分布类型不在支持列表中，则返回空
        if dist.__class__.__name__ not in dist_list:
            return None

        # 根据种子值初始化随机数生成器
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed

        # 调用相应分布类型的随机抽样函数，获取样本
        samp = scipy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)

        # 重新整形样本以匹配指定的大小和形状
        return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))
class SampleMatrixNumpy:
    """Returns the sample from numpy of the given distribution"""

    ### TODO: Add tests after adding matrix distributions in numpy_rv_map
    def __new__(cls, dist, size, seed=None):
        # 调用内部方法 _sample_numpy 处理抽样请求
        return cls._sample_numpy(dist, size, seed)

    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

        numpy_rv_map = {
        }

        sample_shape = {
        }

        dist_list = numpy_rv_map.keys()

        # 检查给定的分布类型是否在 numpy_rv_map 中，若不在则返回 None
        if dist.__class__.__name__ not in dist_list:
            return None

        import numpy
        # 根据是否提供了种子值，创建随机数生成器
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        # 调用对应分布类型的随机抽样方法，并返回抽样结果
        samp = numpy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
        return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))


class SampleMatrixPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        # 调用内部方法 _sample_pymc 处理抽样请求
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

        try:
            import pymc
        except ImportError:
            import pymc3 as pymc
        # 定义 pymc_rv_map 包含的分布类型及其对应的抽样方法
        pymc_rv_map = {
            'MatrixNormalDistribution': lambda dist: pymc.MatrixNormal('X',
                mu=matrix2numpy(dist.location_matrix, float),
                rowcov=matrix2numpy(dist.scale_matrix_1, float),
                colcov=matrix2numpy(dist.scale_matrix_2, float),
                shape=dist.location_matrix.shape),
            'WishartDistribution': lambda dist: pymc.WishartBartlett('X',
                nu=int(dist.n), S=matrix2numpy(dist.scale_matrix, float))
        }

        # 定义每个分布类型对应的样本形状
        sample_shape = {
            'WishartDistribution': lambda dist: dist.scale_matrix.shape,
            'MatrixNormalDistribution' : lambda dist: dist.location_matrix.shape
        }

        dist_list = pymc_rv_map.keys()

        # 检查给定的分布类型是否在 pymc_rv_map 中，若不在则返回 None
        if dist.__class__.__name__ not in dist_list:
            return None
        import logging
        # 设置 PyMC 的日志级别为错误，避免输出冗余信息
        logging.getLogger("pymc").setLevel(logging.ERROR)
        # 在 PyMC 的上下文中，进行分布抽样并返回结果
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samps = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)['X']
        return samps.reshape(size + sample_shape[dist.__class__.__name__](dist))

_get_sample_class_matrixrv = {
    'scipy': SampleMatrixScipy,
    'pymc3': SampleMatrixPymc,
    'pymc': SampleMatrixPymc,
    'numpy': SampleMatrixNumpy
}

################################################################################
#-------------------------Matrix Distribution----------------------------------#
################################################################################

class MatrixDistribution(Distribution, NamedArgsMixin):
    """
    Abstract class for Matrix Distribution.
    """
    """
    创建一个新的不可变矩阵对象，继承自 Basic 类

    实现了 __new__ 方法，用于处理输入参数，将列表转换为 ImmutableMatrix 对象，将其他类型参数转换为符号表达式
    """
    def __new__(cls, *args):
        args = [ImmutableMatrix(arg) if isinstance(arg, list)
                else _sympify(arg) for arg in args]
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        """
        静态方法，用于检查参数，但实际上没有实现具体的检查动作，只是占位符
        """
        pass

    def __call__(self, expr):
        """
        实例方法，允许对象被调用，传入参数 expr，如果 expr 是列表，则将其转换为 ImmutableMatrix 对象，
        然后调用 self.pdf 方法处理该对象并返回结果
        """
        if isinstance(expr, list):
            expr = ImmutableMatrix(expr)
        return self.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        实例方法，用于生成随机样本数据

        Args:
        - size: 可选参数，用于指定样本的大小
        - library: 可选参数，指定使用的数学库，支持的库包括 'scipy', 'numpy', 'pymc3', 'pymc'
        - seed: 可选参数，用于控制随机数生成的种子值

        Returns:
        - 如果成功生成样本，则返回 RandomSymbol 到实际值的映射字典
        - 如果指定的库不支持，则抛出 NotImplementedError
        - 如果无法导入指定的库，则抛出 ValueError
        """
        libraries = ['scipy', 'numpy', 'pymc3', 'pymc']
        if library not in libraries:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                        % str(library))
        if not import_module(library):
            raise ValueError("Failed to import %s" % library)

        # 调用相应库的 _get_sample_class_matrixrv 方法来生成样本数据
        samps = _get_sample_class_matrixrv[library](self, size, seed)

        if samps is not None:
            return samps
        raise NotImplementedError(
                "Sampling for %s is not currently implemented from %s"
                % (self.__class__.__name__, library)
                )
################################################################################
#------------------------Matrix Distribution Types-----------------------------#
################################################################################

#-------------------------------------------------------------------------------
# Matrix Gamma distribution ----------------------------------------------------

class MatrixGammaDistribution(MatrixDistribution):
    # 定义参数名
    _argnames = ('alpha', 'beta', 'scale_matrix')

    @staticmethod
    def check(alpha, beta, scale_matrix):
        # 检查规则：确保scale_matrix是MatrixSymbol类型
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        # 检查规则：确保scale_matrix是方阵
        _value_check(scale_matrix.is_square, "Should "
        "be square matrix")
        # 检查规则：确保alpha为正数
        _value_check(alpha.is_positive, "Shape parameter should be positive.")
        # 检查规则：确保beta为正数
        _value_check(beta.is_positive, "Scale parameter should be positive.")

    @property
    def set(self):
        # 返回一个矩阵集合，元素为实数
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        # 返回scale_matrix的形状
        return self.scale_matrix.shape

    def pdf(self, x):
        # 从实例中获取参数alpha, beta, scale_matrix
        alpha, beta, scale_matrix = self.alpha, self.beta, self.scale_matrix
        p = scale_matrix.shape[0]
        # 如果x是列表，则将其转换为ImmutableMatrix
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        # 如果x不是MatrixBase或MatrixSymbol类型，则引发值错误异常
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        # 计算pdf函数的返回值
        sigma_inv_x = - Inverse(scale_matrix)*x / beta
        term1 = exp(Trace(sigma_inv_x))/((beta**(p*alpha)) * multigamma(alpha, p))
        term2 = (Determinant(scale_matrix))**(-alpha)
        term3 = (Determinant(x))**(alpha - S(p + 1)/2)
        return term1 * term2 * term3

def MatrixGamma(symbol, alpha, beta, scale_matrix):
    """
    Creates a random variable with Matrix Gamma Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    alpha: Positive Real number
        Shape Parameter
    beta: Positive Real number
        Scale Parameter
    scale_matrix: Positive definite real square matrix
        Scale Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, MatrixGamma
    >>> from sympy import MatrixSymbol, symbols
    >>> a, b = symbols('a b', positive=True)
    >>> M = MatrixGamma('M', a, b, [[2, 1], [1, 2]])
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(M)(X).doit()
    exp(Trace(Matrix([
    [-2/3,  1/3],
    [ 1/3, -2/3]])*X)/b)*Determinant(X)**(a - 3/2)/(3**a*sqrt(pi)*b**(2*a)*gamma(a)*gamma(a - 1/2))
    >>> density(M)([[1, 0], [0, 1]]).doit()
    exp(-4/(3*b))/(3**a*sqrt(pi)*b**(2*a)*gamma(a)*gamma(a - 1/2))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_gamma_distribution

    """
    # 检查 scale_matrix 是否为列表类型
    if isinstance(scale_matrix, list):
        # 如果是列表类型，则将其转换为 ImmutableMatrix 对象
        scale_matrix = ImmutableMatrix(scale_matrix)
    # 调用 rv 函数，返回一个符号表示的 MatrixGammaDistribution 随机变量
    return rv(symbol, MatrixGammaDistribution, (alpha, beta, scale_matrix))
#-------------------------------------------------------------------------------
# Wishart Distribution ---------------------------------------------------------

class WishartDistribution(MatrixDistribution):
    # Wishart 分布类，继承自 MatrixDistribution

    _argnames = ('n', 'scale_matrix')

    @staticmethod
    def check(n, scale_matrix):
        # 静态方法，用于检查参数是否符合要求
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix.is_square, "Should "
        "be square matrix")
        _value_check(n.is_positive, "Shape parameter should be positive.")

    @property
    def set(self):
        # 返回一个表示可能值集合的 MatrixSet 对象
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        # 返回尺寸信息，即 scale_matrix 的形状
        return self.scale_matrix.shape

    def pdf(self, x):
        # 概率密度函数定义
        n, scale_matrix = self.n, self.scale_matrix
        p = scale_matrix.shape[0]
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        # 计算 Wishart 分布的概率密度函数
        sigma_inv_x = - Inverse(scale_matrix)*x / S(2)
        term1 = exp(Trace(sigma_inv_x))/((2**(p*n/S(2))) * multigamma(n/S(2), p))
        term2 = (Determinant(scale_matrix))**(-n/S(2))
        term3 = (Determinant(x))**(S(n - p - 1)/2)
        return term1 * term2 * term3

def Wishart(symbol, n, scale_matrix):
    """
    创建一个服从 Wishart 分布的随机变量。

    该分布的密度函数可以在 [1] 中找到。

    Parameters
    ==========

    n: 正实数
        自由度参数
    scale_matrix: 正定实方阵
        规模矩阵

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, Wishart
    >>> from sympy import MatrixSymbol, symbols
    >>> n = symbols('n', positive=True)
    >>> W = Wishart('W', n, [[2, 1], [1, 2]])
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(W)(X).doit()
    exp(Trace(Matrix([
    [-1/3,  1/6],
    [ 1/6, -1/3]])*X))*Determinant(X)**(n/2 - 3/2)/(2**n*3**(n/2)*sqrt(pi)*gamma(n/2)*gamma(n/2 - 1/2))
    >>> density(W)([[1, 0], [0, 1]]).doit()
    exp(-2/3)/(2**n*3**(n/2)*sqrt(pi)*gamma(n/2)*gamma(n/2 - 1/2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Wishart_distribution

    """
    if isinstance(scale_matrix, list):
        scale_matrix = ImmutableMatrix(scale_matrix)
    return rv(symbol, WishartDistribution, (n, scale_matrix))

#-------------------------------------------------------------------------------
# Matrix Normal distribution ---------------------------------------------------

class MatrixNormalDistribution(MatrixDistribution):
    # 矩阵正态分布类，继承自 MatrixDistribution

    _argnames = ('location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    # 检查参数的合法性，确保 scale_matrix_1 是 MatrixSymbol 类型
    def check(location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            # 如果 scale_matrix_1 不是 MatrixSymbol 类型，抛出异常
            _value_check(scale_matrix_1.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        if not isinstance(scale_matrix_2, MatrixSymbol):
            # 如果 scale_matrix_2 不是 MatrixSymbol 类型，抛出异常
            _value_check(scale_matrix_2.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        # 检查 scale_matrix_1 是否为方阵
        _value_check(scale_matrix_1.is_square, "Scale matrix 1 should be "
        "be square matrix")
        # 检查 scale_matrix_2 是否为方阵
        _value_check(scale_matrix_2.is_square, "Scale matrix 2 should be "
        "be square matrix")
        # 获取 location_matrix 的行数和列数
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        # 检查 scale_matrix_1 的形状是否与 location_matrix 的行数相同
        _value_check(scale_matrix_1.shape[0] == n, "Scale matrix 1 should be"
        " of shape %s x %s"% (str(n), str(n)))
        # 检查 scale_matrix_2 的形状是否与 location_matrix 的列数相同
        _value_check(scale_matrix_2.shape[0] == p, "Scale matrix 2 should be"
        " of shape %s x %s"% (str(p), str(p)))

    # 返回 location_matrix 的维度作为 MatrixSet 的属性
    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    # 返回 location_matrix 的形状作为 dimension 方法的返回值
    @property
    def dimension(self):
        return self.location_matrix.shape

    # 计算概率密度函数（PDF），接受一个向量 x 作为参数
    def pdf(self, x):
        # 提取对象中的 location_matrix, scale_matrix_1, scale_matrix_2
        M, U, V = self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        # 获取 location_matrix 的行数和列数
        n, p = M.shape
        # 如果 x 是列表，则转换为不可变矩阵 ImmutableMatrix
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        # 如果 x 不是 MatrixBase 或 MatrixSymbol 类型，则抛出 ValueError 异常
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        # 计算 PDF 中的第一个术语
        term1 = Inverse(V)*Transpose(x - M)*Inverse(U)*(x - M)
        # 计算 PDF 的分子部分
        num = exp(-Trace(term1)/S(2))
        # 计算 PDF 的分母部分
        den = (2*pi)**(S(n*p)/2) * Determinant(U)**(S(p)/2) * Determinant(V)**(S(n)/2)
        # 返回最终的 PDF 值
        return num/den
# 定义一个函数，用于创建具有矩阵正态分布的随机变量
def MatrixNormal(symbol, location_matrix, scale_matrix_1, scale_matrix_2):
    """
    Creates a random variable with Matrix Normal Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    location_matrix: Real ``n x p`` matrix
        Represents degrees of freedom
    scale_matrix_1: Positive definite matrix
        Scale Matrix of shape ``n x n``
    scale_matrix_2: Positive definite matrix
        Scale Matrix of shape ``p x p``

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> from sympy.stats import density, MatrixNormal
    >>> M = MatrixNormal('M', [[1, 2]], [1], [[1, 0], [0, 1]])
    >>> X = MatrixSymbol('X', 1, 2)
    >>> density(M)(X).doit()
    exp(-Trace((Matrix([
    [-1],
    [-2]]) + X.T)*(Matrix([[-1, -2]]) + X))/2)/(2*pi)
    >>> density(M)([[3, 4]]).doit()
    exp(-4)/(2*pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_normal_distribution

    """
    # 如果location_matrix是列表，则转换为ImmutableMatrix
    if isinstance(location_matrix, list):
        location_matrix = ImmutableMatrix(location_matrix)
    # 如果scale_matrix_1是列表，则转换为ImmutableMatrix
    if isinstance(scale_matrix_1, list):
        scale_matrix_1 = ImmutableMatrix(scale_matrix_1)
    # 如果scale_matrix_2是列表，则转换为ImmutableMatrix
    if isinstance(scale_matrix_2, list):
        scale_matrix_2 = ImmutableMatrix(scale_matrix_2)
    # 将所有参数封装成元组
    args = (location_matrix, scale_matrix_1, scale_matrix_2)
    # 调用rv函数创建一个随机变量，使用MatrixNormalDistribution分布
    return rv(symbol, MatrixNormalDistribution, args)

#-------------------------------------------------------------------------------
# Matrix Student's T distribution ---------------------------------------------------

class MatrixStudentTDistribution(MatrixDistribution):
    # 类属性_argnames，包含了参数nu, location_matrix, scale_matrix_1, scale_matrix_2
    _argnames = ('nu', 'location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(nu, location_matrix, scale_matrix_1, scale_matrix_2):
        # 检查 scale_matrix_1 是否为 MatrixSymbol 类型，如果不是，则抛出异常
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite != False, "The shape "
                                                              "matrix must be positive definite.")
        # 检查 scale_matrix_2 是否为 MatrixSymbol 类型，如果不是，则抛出异常
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite != False, "The shape "
                                                              "matrix must be positive definite.")
        # 检查 scale_matrix_1 是否为方阵，如果不是，则抛出异常
        _value_check(scale_matrix_1.is_square != False, "Scale matrix 1 should be "
                                               "be square matrix")
        # 检查 scale_matrix_2 是否为方阵，如果不是，则抛出异常
        _value_check(scale_matrix_2.is_square != False, "Scale matrix 2 should be "
                                               "be square matrix")
        # 获取 location_matrix 的行数和列数
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        # 检查 scale_matrix_1 的行数是否与 location_matrix 的列数相同，否则抛出异常
        _value_check(scale_matrix_1.shape[0] == p, "Scale matrix 1 should be"
                                                   " of shape %s x %s" % (str(p), str(p)))
        # 检查 scale_matrix_2 的行数是否与 location_matrix 的行数相同，否则抛出异常
        _value_check(scale_matrix_2.shape[0] == n, "Scale matrix 2 should be"
                                                   " of shape %s x %s" % (str(n), str(n)))
        # 检查 nu（自由度）是否为正数，否则抛出异常
        _value_check(nu.is_positive != False, "Degrees of freedom must be positive")

    @property
    def set(self):
        # 获取 location_matrix 的行数和列数，并返回一个 MatrixSet 对象
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        # 返回 location_matrix 的形状（行数和列数）
        return self.location_matrix.shape

    def pdf(self, x):
        # 导入所需的库函数
        from sympy.matrices.dense import eye
        # 如果 x 是列表，则将其转换为 ImmutableMatrix 对象
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        # 如果 x 不是 MatrixBase 或 MatrixSymbol 类型，则抛出异常
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                             "or MatrixSymbol" % str(x))
        # 获取 nu（自由度）、M（位置矩阵）、Omega（规模矩阵1）、Sigma（规模矩阵2）的形状信息
        nu, M, Omega, Sigma = self.nu, self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        n, p = M.shape

        # 计算多元 Gamma 函数的值，并组合成 PDF 的标准化常数 K
        K = multigamma((nu + n + p - 1)/2, p) * Determinant(Omega)**(-n/2) * Determinant(Sigma)**(-p/2) \
            / ((pi)**(n*p/2) * multigamma((nu + p - 1)/2, p))
        
        # 计算多元 T 分布的概率密度函数值并返回
        return K * (Determinant(eye(n) + Inverse(Sigma)*(x - M)*Inverse(Omega)*Transpose(x - M))) \
               **(-(nu + n + p -1)/2)
# 定义一个函数，用于生成具有矩阵学生 t 分布的随机变量

"""
Creates a random variable with Matrix Student T Distribution.

The density of the said distribution can be found at [1].

Parameters
==========

nu: Positive Real number
    自由度
location_matrix: Positive definite real square matrix
    形状为 ``n x p`` 的位置矩阵
scale_matrix_1: Positive definite real square matrix
    形状为 ``p x p`` 的比例矩阵 1
scale_matrix_2: Positive definite real square matrix
    形状为 ``n x n`` 的比例矩阵 2

Returns
=======

RandomSymbol

Examples
========

>>> from sympy import MatrixSymbol, symbols
>>> from sympy.stats import density, MatrixStudentT
>>> v = symbols('v', positive=True)
>>> M = MatrixStudentT('M', v, [[1, 2]], [[1, 0], [0, 1]], [1])
>>> X = MatrixSymbol('X', 1, 2)
>>> density(M)(X)
gamma(v/2 + 1)*Determinant((Matrix([[-1, -2]]) + X)*(Matrix([
[-1],
[-2]]) + X.T) + Matrix([[1]]))**(-v/2 - 1)/(pi**1.0*gamma(v/2)*Determinant(Matrix([[1]]))**1.0*Determinant(Matrix([
[1, 0],
[0, 1]]))**0.5)

References
==========

.. [1] https://en.wikipedia.org/wiki/Matrix_t-distribution
"""

# 如果位置矩阵是列表，则将其转换为 ImmutableMatrix 类型
if isinstance(location_matrix, list):
    location_matrix = ImmutableMatrix(location_matrix)

# 如果比例矩阵 1 是列表，则将其转换为 ImmutableMatrix 类型
if isinstance(scale_matrix_1, list):
    scale_matrix_1 = ImmutableMatrix(scale_matrix_1)

# 如果比例矩阵 2 是列表，则将其转换为 ImmutableMatrix 类型
if isinstance(scale_matrix_2, list):
    scale_matrix_2 = ImmutableMatrix(scale_matrix_2)

# 组装参数元组
args = (nu, location_matrix, scale_matrix_1, scale_matrix_2)

# 调用 rv 函数生成随机变量，使用 MatrixStudentTDistribution 类和参数 args
return rv(symbol, MatrixStudentTDistribution, args)
```