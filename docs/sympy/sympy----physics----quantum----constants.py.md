# `D:\src\scipysrc\sympy\sympy\physics\quantum\constants.py`

```
"""Constants (like hbar) related to quantum mechanics."""

# 导入所需模块和类
from sympy.core.numbers import NumberSymbol
from sympy.core.singleton import Singleton
from sympy.printing.pretty.stringpict import prettyForm
import mpmath.libmp as mlib

#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

# 定义导出的常量列表
__all__ = [
    'hbar',
    'HBar',
]

# 定义 HBar 类，继承自 NumberSymbol，并使用 Singleton 元类
class HBar(NumberSymbol, metaclass=Singleton):
    """Reduced Plank's constant in numerical and symbolic form [1]_.

    Examples
    ========

        >>> from sympy.physics.quantum.constants import hbar
        >>> hbar.evalf()
        1.05457162000000e-34

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Planck_constant
    """

    # 常数的属性
    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True

    # 禁止实例的动态属性
    __slots__ = ()

    # 将常数转换为浮点数的方法，使用 mlib 提供的精度
    def _as_mpf_val(self, prec):
        return mlib.from_float(1.05457162e-34, prec)

    # 用于 SymPy 表示的方法，返回符号表达式 'HBar()'
    def _sympyrepr(self, printer, *args):
        return 'HBar()'

    # 用于 SymPy 字符串表示的方法，返回符号字符串 'hbar'
    def _sympystr(self, printer, *args):
        return 'hbar'

    # 用于美观打印的方法，根据是否使用 Unicode 返回不同的表达形式
    def _pretty(self, printer, *args):
        if printer._use_unicode:
            return prettyForm('\N{PLANCK CONSTANT OVER TWO PI}')
        return prettyForm('hbar')

    # 用于 LaTeX 表示的方法，返回 LaTeX 字符串 '\hbar'
    def _latex(self, printer, *args):
        return r'\hbar'

# 创建一个全局的 hbar 常量实例供所有人使用
hbar = HBar()
```