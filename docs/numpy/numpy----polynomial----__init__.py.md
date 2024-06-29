# `.\numpy\numpy\polynomial\__init__.py`

```
"""
A sub-package for efficiently dealing with polynomials.

Within the documentation for this sub-package, a "finite power series,"
i.e., a polynomial (also referred to simply as a "series") is represented
by a 1-D numpy array of the polynomial's coefficients, ordered from lowest
order term to highest.  For example, array([1,2,3]) represents
``P_0 + 2*P_1 + 3*P_2``, where P_n is the n-th order basis polynomial
applicable to the specific module in question, e.g., `polynomial` (which
"wraps" the "standard" basis) or `chebyshev`.  For optimal performance,
all operations on polynomials, including evaluation at an argument, are
implemented as operations on the coefficients.  Additional (module-specific)
information can be found in the docstring for the module of interest.

This package provides *convenience classes* for each of six different kinds
of polynomials:

========================    ================
**Name**                    **Provides**
========================    ================
`~polynomial.Polynomial`    Power series
`~chebyshev.Chebyshev`      Chebyshev series
`~legendre.Legendre`        Legendre series
`~laguerre.Laguerre`        Laguerre series
`~hermite.Hermite`          Hermite series
`~hermite_e.HermiteE`       HermiteE series
========================    ================

These *convenience classes* provide a consistent interface for creating,
manipulating, and fitting data with polynomials of different bases.
The convenience classes are the preferred interface for the `~numpy.polynomial`
package, and are available from the ``numpy.polynomial`` namespace.
This eliminates the need to navigate to the corresponding submodules, e.g.
``np.polynomial.Polynomial`` or ``np.polynomial.Chebyshev`` instead of
``np.polynomial.polynomial.Polynomial`` or
``np.polynomial.chebyshev.Chebyshev``, respectively.
The classes provide a more consistent and concise interface than the
type-specific functions defined in the submodules for each type of polynomial.
For example, to fit a Chebyshev polynomial with degree ``1`` to data given
by arrays ``xdata`` and ``ydata``, the
`~chebyshev.Chebyshev.fit` class method::

    >>> from numpy.polynomial import Chebyshev
    >>> xdata = [1, 2, 3, 4]
    >>> ydata = [1, 4, 9, 16]
    >>> c = Chebyshev.fit(xdata, ydata, deg=1)

is preferred over the `chebyshev.chebfit` function from the
``np.polynomial.chebyshev`` module::

    >>> from numpy.polynomial.chebyshev import chebfit
    >>> c = chebfit(xdata, ydata, deg=1)

See :doc:`routines.polynomials.classes` for more details.

Convenience Classes
===================

The following lists the various constants and methods common to all of
the classes representing the various kinds of polynomials. In the following,
the term ``Poly`` represents any one of the convenience classes (e.g.
`~polynomial.Polynomial`, `~chebyshev.Chebyshev`, `~hermite.Hermite`, etc.)
while the lowercase ``p`` represents an **instance** of a polynomial class.

Constants
---------
"""
# 设置默认的多项式字符串表示格式
def set_default_printstyle(style):
    """
    Set the default format for the string representation of polynomials.

    Values for ``style`` must be valid inputs to ``__format__``, i.e. 'ascii'
    or 'unicode'.

    Parameters
    ----------
    style : str
        Format string for default printing style. Must be either 'ascii' or
        'unicode'.

    Notes
    -----
    The default format depends on the platform: 'unicode' is used on
    Unix-based systems and 'ascii' on Windows. This determination is based on
    default font support for the unicode superscript and subscript ranges.

    Examples
    --------
    >>> p = np.polynomial.Polynomial([1, 2, 3])
    >>> c = np.polynomial.Chebyshev([1, 2, 3])
    >>> np.polynomial.set_default_printstyle('unicode')
    >>> print(p)
    1.0 + 2.0·x + 3.0·x²
    """
    pass



# 默认域
Poly.domain



# 默认窗口
Poly.window



# 用于表示基础的字符串
Poly.basis_name



# 允许的最大幂值，使得 p**n 是被允许的
Poly.maxpower



# 打印时使用的字符串
Poly.nickname



# 创建
# 返回给定阶数的基础多项式
Poly.basis(degree)



# 创建
# 返回多项式 p，其中 p(x) = x 对所有 x 成立
Poly.identity()



# 创建
# 返回与数据 x、y 最小二乘拟合的阶数为 deg 的多项式 p
Poly.fit(x, y, deg)



# 创建
# 返回具有指定根的多项式 p
Poly.fromroots(roots)



# 创建
# 复制多项式 p
p.copy()



# 转换
# 将 p 转换为指定类别 Poly 的实例
p.cast(Poly)



# 转换
# 将 p 转换为指定类别 Poly 的实例或在 domain 和 window 之间进行映射
p.convert(Poly)



# 微积分
# 对 p 求导数
p.deriv()



# 微积分
# 对 p 积分
p.integ()



# 验证
# 检查多项式 p1 和 p2 的系数是否匹配
Poly.has_samecoef(p1, p2)



# 验证
# 检查多项式 p1 和 p2 的域是否匹配
Poly.has_samedomain(p1, p2)



# 验证
# 检查多项式 p1 和 p2 的类型是否匹配
Poly.has_sametype(p1, p2)



# 验证
# 检查多项式 p1 和 p2 的窗口是否匹配
Poly.has_samewindow(p1, p2)



# 杂项
# 在默认域内返回均匀间隔点上的 x, p(x)
p.linspace()



# 杂项
# 返回在 domain 和 window 之间的线性映射参数
p.mapparms()



# 杂项
# 返回多项式 p 的根
p.roots()



# 杂项
# 去除多项式 p 的尾系数
p.trim()



# 杂项
# 截断多项式 p 到给定阶数
p.cutdeg(degree)



# 杂项
# 截断多项式 p 到给定大小
p.truncate(size)
    >>> print(c)
    1.0 + 2.0·T₁(x) + 3.0·T₂(x)
    >>> np.polynomial.set_default_printstyle('ascii')
    >>> print(p)
    1.0 + 2.0 x + 3.0 x**2
    >>> print(c)
    1.0 + 2.0 T_1(x) + 3.0 T_2(x)
    >>> # Formatting supersedes all class/package-level defaults
    >>> print(f"{p:unicode}")
    1.0 + 2.0·x + 3.0·x²
    """
    if style not in ('unicode', 'ascii'):
        raise ValueError(
            f"Unsupported format string '{style}'. Valid options are 'ascii' "
            f"and 'unicode'"
        )
    _use_unicode = True
    if style == 'ascii':
        _use_unicode = False
    from ._polybase import ABCPolyBase
    ABCPolyBase._use_unicode = _use_unicode
    
    
    注释：
    
    
    # 如果样式不是 'unicode' 或 'ascii'，则抛出值错误异常，提示有效选项为 'ascii' 和 'unicode'
    if style not in ('unicode', 'ascii'):
        raise ValueError(
            f"Unsupported format string '{style}'. Valid options are 'ascii' "
            f"and 'unicode'"
        )
    # 默认使用 Unicode 格式打印多项式
    _use_unicode = True
    # 如果选择了 'ascii' 格式，则设置为不使用 Unicode
    if style == 'ascii':
        _use_unicode = False
    # 导入多项式基类，并设置其打印格式为当前选择的格式
    from ._polybase import ABCPolyBase
    ABCPolyBase._use_unicode = _use_unicode
# 从numpy._pytesttester模块中导入PytestTester类
from numpy._pytesttester import PytestTester
# 创建一个PytestTester对象，并将当前模块的名称作为参数传递
test = PytestTester(__name__)
# 删除当前作用域中的PytestTester类的引用，通常用于清理不再需要的变量或类
del PytestTester
```