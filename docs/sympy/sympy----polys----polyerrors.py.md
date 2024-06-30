# `D:\src\scipysrc\sympy\sympy\polys\polyerrors.py`

```
"""
`polys` 模块的常见异常定义。
"""

# 导入 sympy.utilities 中的 public 函数
from sympy.utilities import public

# 定义一个公开的基础多项式错误类
@public
class BasePolynomialError(Exception):
    """多项式相关异常的基类。"""

    # 定义一个未实现的方法 new，用于抛出 NotImplementedError 异常
    def new(self, *args):
        raise NotImplementedError("abstract base class")

# 定义一个公开的精确除法失败异常类，继承自 BasePolynomialError
@public
class ExactQuotientFailed(BasePolynomialError):

    # 初始化方法，接收参数 f、g 和 dom
    def __init__(self, f, g, dom=None):
        self.f, self.g, self.dom = f, g, dom

    # 字符串表示方法
    def __str__(self):  # pragma: no cover
        # 导入 sympy.printing.str 中的 sstr 函数
        from sympy.printing.str import sstr

        # 根据 dom 的存在与否返回不同的字符串表示
        if self.dom is None:
            return "%s does not divide %s" % (sstr(self.g), sstr(self.f))
        else:
            return "%s does not divide %s in %s" % (sstr(self.g), sstr(self.f), sstr(self.dom))

    # 定义一个 new 方法，返回一个新的 ExactQuotientFailed 异常对象
    def new(self, f, g):
        return self.__class__(f, g, self.dom)

# 定义一个公开的多项式除法失败异常类，继承自 BasePolynomialError
@public
class PolynomialDivisionFailed(BasePolynomialError):

    # 初始化方法，接收参数 f、g 和 domain
    def __init__(self, f, g, domain):
        self.f = f
        self.g = g
        self.domain = domain

    # 字符串表示方法
    def __str__(self):
        # 根据 domain 的不同状态返回不同的错误信息
        if self.domain.is_EX:
            msg = "You may want to use a different simplification algorithm. Note " \
                  "that in general it's not possible to guarantee to detect zero "  \
                  "in this domain."
        elif not self.domain.is_Exact:
            msg = "Your working precision or tolerance of computations may be set " \
                  "improperly. Adjust those parameters of the coefficient domain "  \
                  "and try again."
        else:
            msg = "Zero detection is guaranteed in this coefficient domain. This "  \
                  "may indicate a bug in SymPy or the domain is user defined and "  \
                  "doesn't implement zero detection properly."

        return "couldn't reduce degree in a polynomial division algorithm when "    \
               "dividing %s by %s. This can happen when it's not possible to "      \
               "detect zero in the coefficient domain. The domain of computation "  \
               "is %s. %s" % (self.f, self.g, self.domain, msg)

# 定义一个公开的不支持的操作异常类，继承自 BasePolynomialError
@public
class OperationNotSupported(BasePolynomialError):

    # 初始化方法，接收参数 poly 和 func
    def __init__(self, poly, func):
        self.poly = poly
        self.func = func

    # 字符串表示方法
    def __str__(self):  # pragma: no cover
        return "`%s` operation not supported by %s representation" % (self.func, self.poly.rep.__class__.__name__)

# 定义一个公开的启发式最大公因数算法失败异常类，继承自 BasePolynomialError
@public
class HeuristicGCDFailed(BasePolynomialError):
    pass

# 定义一个模块化最大公因数算法失败异常类，继承自 BasePolynomialError
class ModularGCDFailed(BasePolynomialError):
    pass

# 定义一个公开的同态映射失败异常类，继承自 BasePolynomialError
@public
class HomomorphismFailed(BasePolynomialError):
    pass

# 定义一个公开的同构映射失败异常类，继承自 BasePolynomialError
@public
class IsomorphismFailed(BasePolynomialError):
    pass

# 定义一个公开的多余因子异常类，继承自 BasePolynomialError
@public
class ExtraneousFactors(BasePolynomialError):
    pass

# 定义一个公开的求值失败异常类，继承自 BasePolynomialError
@public
class EvaluationFailed(BasePolynomialError):
    pass

# 定义一个公开的精炼失败异常类，继承自 BasePolynomialError
@public
class RefinementFailed(BasePolynomialError):
    pass

# 定义一个公开的强制转换失败异常类，继承自 BasePolynomialError
@public
class CoercionFailed(BasePolynomialError):
    pass

# 定义一个公开的不可逆异常类，继承自 BasePolynomialError
@public
class NotInvertible(BasePolynomialError):
    pass

# 定义一个公开的不可逆转异常类，继承自 BasePolynomialError
@public
class NotReversible(BasePolynomialError):
    pass

# 定义一个公开的
@public
# 定义一个继承自 BasePolynomialError 的异常类 NotAlgebraic
class NotAlgebraic(BasePolynomialError):
    pass

# 定义一个公开的域错误异常类 DomainError，继承自 BasePolynomialError
@public
class DomainError(BasePolynomialError):
    pass

# 定义一个公开的多项式错误异常类 PolynomialError，继承自 BasePolynomialError
@public
class PolynomialError(BasePolynomialError):
    pass

# 定义一个公开的无法统一的异常类 UnificationFailed，继承自 BasePolynomialError
@public
class UnificationFailed(BasePolynomialError):
    pass

# 定义一个公开的无法解的因子错误异常类 UnsolvableFactorError，继承自 BasePolynomialError
@public
class UnsolvableFactorError(BasePolynomialError):
    """Raised if ``roots`` is called with strict=True and a polynomial
     having a factor whose solutions are not expressible in radicals
     is encountered."""

# 定义一个公开的生成器错误异常类 GeneratorsError，继承自 BasePolynomialError
@public
class GeneratorsError(BasePolynomialError):
    pass

# 定义一个公开的需要生成器的异常类 GeneratorsNeeded，继承自 GeneratorsError
@public
class GeneratorsNeeded(GeneratorsError):
    pass

# 定义一个公开的计算失败异常类 ComputationFailed，继承自 BasePolynomialError
@public
class ComputationFailed(BasePolynomialError):

    def __init__(self, func, nargs, exc):
        self.func = func
        self.nargs = nargs
        self.exc = exc

    def __str__(self):
        return "%s(%s) failed without generators" % (self.func, ', '.join(map(str, self.exc.exprs[:self.nargs])))

# 定义一个公开的一元多项式错误异常类 UnivariatePolynomialError，继承自 PolynomialError
@public
class UnivariatePolynomialError(PolynomialError):
    pass

# 定义一个公开的多元多项式错误异常类 MultivariatePolynomialError，继承自 PolynomialError
@public
class MultivariatePolynomialError(PolynomialError):
    pass

# 定义一个公开的多项式构建失败异常类 PolificationFailed，继承自 PolynomialError
@public
class PolificationFailed(PolynomialError):

    def __init__(self, opt, origs, exprs, seq=False):
        if not seq:
            self.orig = origs
            self.expr = exprs
            self.origs = [origs]
            self.exprs = [exprs]
        else:
            self.origs = origs
            self.exprs = exprs

        self.opt = opt
        self.seq = seq

    def __str__(self):  # pragma: no cover
        if not self.seq:
            return "Cannot construct a polynomial from %s" % str(self.orig)
        else:
            return "Cannot construct polynomials from %s" % ', '.join(map(str, self.origs))

# 定义一个公开的选项错误异常类 OptionError，继承自 BasePolynomialError
@public
class OptionError(BasePolynomialError):
    pass

# 定义一个公开的标志错误异常类 FlagError，继承自 OptionError
@public
class FlagError(OptionError):
    pass
```