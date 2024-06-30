# `D:\src\scipysrc\scipy\scipy\odr\_models.py`

```
""" Collection of Model instances for use with the odrpack fitting package.
"""
# 导入必要的库
import numpy as np
from scipy.odr._odrpack import Model

# 定义本模块中公开的符号列表
__all__ = ['Model', 'exponential', 'multilinear', 'unilinear', 'quadratic',
           'polynomial']

# 定义一个线性函数 _lin_fcn，用于计算线性模型的预测值
def _lin_fcn(B, x):
    # 拆解 B，获取截距 a 和系数向量 b
    a, b = B[0], B[1:]
    # 调整系数向量 b 的形状，使其变为列向量
    b.shape = (b.shape[0], 1)
    # 计算线性模型的预测值并返回
    return a + (x*b).sum(axis=0)

# 定义一个线性函数的雅可比矩阵 _lin_fjb，用于计算线性函数关于参数的雅可比矩阵
def _lin_fjb(B, x):
    # 创建一个全为 1 的向量 a，长度与 x 的最后一个维度相同
    a = np.ones(x.shape[-1], float)
    # 将 a 和 x 平坦化后的数组连接起来，形成雅可比矩阵
    res = np.concatenate((a, x.ravel()))
    # 调整雅可比矩阵的形状
    res.shape = (B.shape[-1], x.shape[-1])
    return res

# 定义一个线性函数的对角雅可比矩阵 _lin_fjd，用于计算线性函数关于数据的对角雅可比矩阵
def _lin_fjd(B, x):
    # 获取系数向量 b
    b = B[1:]
    # 将 b 按列的方式重复，使其与 x 的形状相匹配
    b = np.repeat(b, (x.shape[-1],)*b.shape[-1], axis=0)
    # 调整结果的形状为 x 的形状
    b.shape = x.shape
    return b

# 定义一个线性函数的估计函数 _lin_est，返回估计值
def _lin_est(data):
    # 如果数据 x 的维度为 2，则获取数据的行数 m，否则 m 为 1
    if len(data.x.shape) == 2:
        m = data.x.shape[0]
    else:
        m = 1
    # 返回一个全为 1 的浮点数数组，长度为 m+1
    return np.ones((m + 1,), float)

# 定义一个多项式函数 _poly_fcn，用于计算多项式模型的预测值
def _poly_fcn(B, x, powers):
    # 拆解 B，获取截距 a 和系数向量 b
    a, b = B[0], B[1:]
    # 调整系数向量 b 的形状，使其变为列向量
    b.shape = (b.shape[0], 1)
    # 计算多项式模型的预测值并返回
    return a + np.sum(b * np.power(x, powers), axis=0)

# 定义一个多项式函数的雅可比矩阵 _poly_fjacb，用于计算多项式函数关于参数的雅可比矩阵
def _poly_fjacb(B, x, powers):
    # 构造雅可比矩阵的一部分，包含常数项 1 和 x 的 powers 次方
    res = np.concatenate((np.ones(x.shape[-1], float),
                          np.power(x, powers).flat))
    # 调整雅可比矩阵的形状
    res.shape = (B.shape[-1], x.shape[-1])
    return res

# 定义一个多项式函数的对角雅可比矩阵 _poly_fjacd，用于计算多项式函数关于数据的对角雅可比矩阵
def _poly_fjacd(B, x, powers):
    # 获取系数向量 b
    b = B[1:]
    # 调整系数向量 b 的形状，使其变为列向量
    b.shape = (b.shape[0], 1)
    # 计算多项式函数关于数据的对角雅可比矩阵并返回
    b = b * powers
    return np.sum(b * np.power(x, powers-1), axis=0)

# 定义一个指数函数 _exp_fcn，用于计算指数模型的预测值
def _exp_fcn(B, x):
    # 计算指数模型的预测值并返回
    return B[0] + np.exp(B[1] * x)

# 定义一个指数函数的对角雅可比矩阵 _exp_fjd，用于计算指数函数关于数据的对角雅可比矩阵
def _exp_fjd(B, x):
    # 计算指数函数关于数据的对角雅可比矩阵并返回
    return B[1] * np.exp(B[1] * x)

# 定义一个指数函数的雅可比矩阵 _exp_fjb，用于计算指数函数关于参数的雅可比矩阵
def _exp_fjb(B, x):
    # 构造雅可比矩阵的一部分，包含常数项 1 和 x * exp(B[1] * x)
    res = np.concatenate((np.ones(x.shape[-1], float), x * np.exp(B[1] * x)))
    # 调整雅可比矩阵的形状
    res.shape = (2, x.shape[-1])
    return res

# 定义一个指数函数的估计函数 _exp_est，返回估计值
def _exp_est(data):
    # 返回一个包含两个浮点数 1 的数组
    return np.array([1., 1.])

# 定义一个多维线性模型 _MultilinearModel，继承自 Model 类
class _MultilinearModel(Model):
    r"""
    Arbitrary-dimensional linear model

    This model is defined by :math:`y=\beta_0 + \sum_{i=1}^m \beta_i x_i`

    Examples
    --------
    We can calculate orthogonal distance regression with an arbitrary
    dimensional linear model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 10.0 + 5.0 * x
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.multilinear)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [10.  5.]

    """

    def __init__(self):
        # 调用父类 Model 的构造函数，初始化线性函数及其相关参数
        super().__init__(
            _lin_fcn, fjacb=_lin_fjb, fjacd=_lin_fjd, estimate=_lin_est,
            meta={'name': 'Arbitrary-dimensional Linear',
                  'equ': 'y = B_0 + Sum[i=1..m, B_i * x_i]',
                  'TeXequ': r'$y=\beta_0 + \sum_{i=1}^m \beta_i x_i'})

# 创建一个多维线性模型的实例 multilinear
multilinear = _MultilinearModel()

# 定义一个多项式模型的工厂函数 polynomial，用于生成指定阶数的多项式模型
def polynomial(order):
    """
    Factory function for a general polynomial model.

    Parameters
    ----------
    order : int
        多项式的阶数
    order : int or sequence
        如果是整数，将作为拟合多项式的阶数。如果是数字序列，将作为多项式中各项的指数。
        常数项（幂为0的项）总是包含在内，所以不需要包含0。
        因此，polynomial(n) 相当于 polynomial(range(1, n+1))。

    Returns
    -------
    polynomial : Model instance
        返回一个模型实例。

    Examples
    --------
    我们可以使用正交距离回归（ODR）拟合输入数据，使用多项式模型：

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import odr
    >>> x = np.linspace(0.0, 5.0)
    >>> y = np.sin(x)
    >>> poly_model = odr.polynomial(3)  # 使用三阶多项式模型
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, poly_model)
    >>> output = odr_obj.run()  # 运行ODR拟合
    >>> poly = np.poly1d(output.beta[::-1])
    >>> poly_y = poly(x)
    >>> plt.plot(x, y, label="输入数据")
    >>> plt.plot(x, poly_y, label="多项式ODR")
    >>> plt.legend()
    >>> plt.show()

    """

    powers = np.asarray(order)
    if powers.shape == ():
        # 如果是标量。
        # 将其视为单一整数，生成从1到该整数的整数序列。
        powers = np.arange(1, powers + 1)

    powers.shape = (len(powers), 1)
    len_beta = len(powers) + 1

    def _poly_est(data, len_beta=len_beta):
        # 啊。忽略数据并返回全为1的数组。
        return np.ones((len_beta,), float)

    return Model(_poly_fcn, fjacd=_poly_fjacd, fjacb=_poly_fjacb,
                 estimate=_poly_est, extra_args=(powers,),
                 meta={'name': 'Sorta-general Polynomial',
                 'equ': 'y = B_0 + Sum[i=1..%s, B_i * (x**i)]' % (len_beta-1),
                 'TeXequ': r'$y=\beta_0 + \sum_{i=1}^{%s} \beta_i x^i$' %
                        (len_beta-1)})
class _ExponentialModel(Model):
    r"""
    Exponential model

    This model is defined by :math:`y=\beta_0 + e^{\beta_1 x}`

    Examples
    --------
    We can calculate orthogonal distance regression with an exponential model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = -10.0 + np.exp(0.5*x)
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.exponential)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [-10.    0.5]

    """

    def __init__(self):
        # 调用父类构造函数初始化模型
        super().__init__(_exp_fcn, fjacd=_exp_fjd, fjacb=_exp_fjb,
                         estimate=_exp_est,
                         meta={'name': 'Exponential',
                               'equ': 'y= B_0 + exp(B_1 * x)',
                               'TeXequ': r'$y=\beta_0 + e^{\beta_1 x}$'})


exponential = _ExponentialModel()


def _unilin(B, x):
    # 定义一元线性模型函数 y = B[0] * x + B[1]
    return x * B[0] + B[1]


def _unilin_fjd(B, x):
    # 返回一元线性模型在雅可比矩阵的偏导数
    return np.ones(x.shape, float) * B[0]


def _unilin_fjb(B, x):
    # 返回一元线性模型在雅可比矩阵的偏导数，包含 x 和常数项
    _ret = np.concatenate((x, np.ones(x.shape, float)))
    _ret.shape = (2,) + x.shape
    return _ret


def _unilin_est(data):
    # 返回一元线性模型的参数初始估计值
    return (1., 1.)


def _quadratic(B, x):
    # 定义二次模型函数 y = B[0] * x^2 + B[1] * x + B[2]
    return x * (x * B[0] + B[1]) + B[2]


def _quad_fjd(B, x):
    # 返回二次模型在雅可比矩阵的偏导数
    return 2 * x * B[0] + B[1]


def _quad_fjb(B, x):
    # 返回二次模型在雅可比矩阵的偏导数，包含 x^2、x 和常数项
    _ret = np.concatenate((x * x, x, np.ones(x.shape, float)))
    _ret.shape = (3,) + x.shape
    return _ret


def _quad_est(data):
    # 返回二次模型的参数初始估计值
    return (1., 1., 1.)


class _UnilinearModel(Model):
    r"""
    Univariate linear model

    This model is defined by :math:`y = \beta_0 x + \beta_1`

    Examples
    --------
    We can calculate orthogonal distance regression with an unilinear model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 1.0 * x + 2.0
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.unilinear)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [1. 2.]

    """

    def __init__(self):
        # 调用父类构造函数初始化模型
        super().__init__(_unilin, fjacd=_unilin_fjd, fjacb=_unilin_fjb,
                         estimate=_unilin_est,
                         meta={'name': 'Univariate Linear',
                               'equ': 'y = B_0 * x + B_1',
                               'TeXequ': '$y = \\beta_0 x + \\beta_1$'})


unilinear = _UnilinearModel()


class _QuadraticModel(Model):
    r"""
    Quadratic model

    This model is defined by :math:`y = \beta_0 x^2 + \beta_1 x + \beta_2`

    Examples
    --------
    We can calculate orthogonal distance regression with a quadratic model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 1.0 * x ** 2 + 2.0 * x + 3.0
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.quadratic)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [1. 2. 3.]

    """
    # 定义 Quadratic 类的初始化方法
    def __init__(self):
        # 调用父类的初始化方法，传递参数和关键字参数
        super().__init__(
            _quadratic, fjacd=_quad_fjd, fjacb=_quad_fjb, estimate=_quad_est,
            # 设置元数据，包括名称、方程式字符串、TeX 格式的方程式字符串
            meta={'name': 'Quadratic',
                  'equ': 'y = B_0*x**2 + B_1*x + B_2',
                  'TeXequ': '$y = \\beta_0 x^2 + \\beta_1 x + \\beta_2'})
# 创建一个名为 quadratic 的对象实例，使用 _QuadraticModel 类的默认构造函数
quadratic = _QuadraticModel()
```