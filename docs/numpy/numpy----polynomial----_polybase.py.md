# `.\numpy\numpy\polynomial\_polybase.py`

```
"""
Abstract base class for the various polynomial Classes.

The ABCPolyBase class provides the methods needed to implement the common API
for the various polynomial classes. It operates as a mixin, but uses the
abc module from the stdlib, hence it is only available for Python >= 2.6.

"""
import os  # 导入标准库中的 os 模块
import abc  # 导入标准库中的 abc 模块
import numbers  # 导入标准库中的 numbers 模块
from typing import Callable  # 从 typing 模块中导入 Callable 类型

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from . import polyutils as pu  # 从当前包中导入 polyutils 模块，并使用 pu 别名

__all__ = ['ABCPolyBase']  # 定义当前模块中导出的符号列表

class ABCPolyBase(abc.ABC):
    """An abstract base class for immutable series classes.

    ABCPolyBase provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' along with the
    methods listed below.

    .. versionadded:: 1.9.0

    Parameters
    ----------
    coef : array_like
        Series coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``, where
        ``P_i`` is the basis polynomials of degree ``i``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is the derived class domain.
    window : (2,) array_like, optional
        Window, see domain for its use. The default value is the
        derived class window.
    symbol : str, optional
        Symbol used to represent the independent variable in string 
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    Attributes
    ----------
    coef : (N,) ndarray
        Series coefficients in order of increasing degree.
    domain : (2,) ndarray
        Domain that is mapped to window.
    window : (2,) ndarray
        Window that domain is mapped to.
    symbol : str
        Symbol representing the independent variable.

    Class Attributes
    ----------------
    maxpower : int
        Maximum power allowed, i.e., the largest number ``n`` such that
        ``p(x)**n`` is allowed. This is to limit runaway polynomial size.
    domain : (2,) ndarray
        Default domain of the class.
    window : (2,) ndarray
        Default window of the class.

    """

    # Not hashable
    __hash__ = None  # 禁用对象的哈希特性

    # Opt out of numpy ufuncs and Python ops with ndarray subclasses.
    __array_ufunc__ = None  # 禁用 ndarray 子类的 NumPy 和 Python 操作

    # Limit runaway size. T_n^m has degree n*m
    maxpower = 100  # 定义类属性 maxpower，限制多项式的最大次数

    # Unicode character mappings for improved __str__
    _superscript_mapping = str.maketrans({
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    })  # 定义将数字字符转换为上标 Unicode 字符的映射表

    _subscript_mapping = str.maketrans({
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉"
    })  # 定义将数字字符转换为下标 Unicode 字符的映射表
    # 某些字体不支持完整的 Unicode 字符范围，这对于包括 Windows shell/终端中常见的默认字体在内的一些字体来说是必需的，因此在 Windows 平台上默认只打印 ASCII 字符。
    _use_unicode = not os.name == 'nt'

    @property
    def symbol(self):
        # 返回属性 _symbol 的值
        return self._symbol

    @property
    @abc.abstractmethod
    def domain(self):
        # 抽象方法，子类需要实现，返回对象的 domain 属性

    @property
    @abc.abstractmethod
    def window(self):
        # 抽象方法，子类需要实现，返回对象的 window 属性

    @property
    @abc.abstractmethod
    def basis_name(self):
        # 抽象方法，子类需要实现，返回对象的 basis_name 属性

    @staticmethod
    @abc.abstractmethod
    def _add(c1, c2):
        # 抽象静态方法，子类需要实现，用于执行加法操作

    @staticmethod
    @abc.abstractmethod
    def _sub(c1, c2):
        # 抽象静态方法，子类需要实现，用于执行减法操作

    @staticmethod
    @abc.abstractmethod
    def _mul(c1, c2):
        # 抽象静态方法，子类需要实现，用于执行乘法操作

    @staticmethod
    @abc.abstractmethod
    def _div(c1, c2):
        # 抽象静态方法，子类需要实现，用于执行除法操作

    @staticmethod
    @abc.abstractmethod
    def _pow(c, pow, maxpower=None):
        # 抽象静态方法，子类需要实现，用于执行幂运算

    @staticmethod
    @abc.abstractmethod
    def _val(x, c):
        # 抽象静态方法，子类需要实现，用于计算特定值

    @staticmethod
    @abc.abstractmethod
    def _int(c, m, k, lbnd, scl):
        # 抽象静态方法，子类需要实现，用于计算积分

    @staticmethod
    @abc.abstractmethod
    def _der(c, m, scl):
        # 抽象静态方法，子类需要实现，用于计算导数

    @staticmethod
    @abc.abstractmethod
    def _fit(x, y, deg, rcond, full):
        # 抽象静态方法，子类需要实现，用于拟合数据

    @staticmethod
    @abc.abstractmethod
    def _line(off, scl):
        # 抽象静态方法，子类需要实现，用于创建线性对象

    @staticmethod
    @abc.abstractmethod
    def _roots(c):
        # 抽象静态方法，子类需要实现，用于计算根

    @staticmethod
    @abc.abstractmethod
    def _fromroots(r):
        # 抽象静态方法，子类需要实现，用于根据给定的根创建对象

    def has_samecoef(self, other):
        """Check if coefficients match.

        .. versionadded:: 1.6.0

        Parameters
        ----------
        other : class instance
            The other class must have the ``coef`` attribute.

        Returns
        -------
        bool : boolean
            True if the coefficients are the same, False otherwise.

        """
        # 检查两个对象的 coef 属性长度是否相等，如果不相等返回 False；否则比较它们的值是否完全相同，相同则返回 True，否则返回 False
        if len(self.coef) != len(other.coef):
            return False
        elif not np.all(self.coef == other.coef):
            return False
        else:
            return True

    def has_samedomain(self, other):
        """Check if domains match.

        .. versionadded:: 1.6.0

        Parameters
        ----------
        other : class instance
            The other class must have the ``domain`` attribute.

        Returns
        -------
        bool : boolean
            True if the domains are the same, False otherwise.

        """
        # 检查两个对象的 domain 属性是否完全相同，使用 NumPy 的 all 函数比较
        return np.all(self.domain == other.domain)

    def has_samewindow(self, other):
        """Check if windows match.

        .. versionadded:: 1.6.0

        Parameters
        ----------
        other : class instance
            The other class must have the ``window`` attribute.

        Returns
        -------
        bool : boolean
            True if the windows are the same, False otherwise.

        """
        # 检查两个对象的 window 属性是否完全相同，使用 NumPy 的 all 函数比较
        return np.all(self.window == other.window)
    def has_sametype(self, other):
        """
        Check if types match.

        .. versionadded:: 1.7.0

        Parameters
        ----------
        other : object
            Class instance.

        Returns
        -------
        bool : boolean
            True if `other` is of the same class as `self`.

        """
        return isinstance(other, self.__class__)

    def _get_coefficients(self, other):
        """
        Interpret `other` as polynomial coefficients.

        The `other` argument is checked to see if it is of the same
        class as `self` with identical domain and window. If so,
        return its coefficients, otherwise return `other`.

        .. versionadded:: 1.9.0

        Parameters
        ----------
        other : anything
            Object to be checked.

        Returns
        -------
        coef
            The coefficients of `other` if it is a compatible instance
            of ABCPolyBase, otherwise `other`.

        Raises
        ------
        TypeError
            When `other` is an incompatible instance of ABCPolyBase.

        """
        if isinstance(other, ABCPolyBase):
            if not isinstance(other, self.__class__):
                raise TypeError("Polynomial types differ")
            elif not np.all(self.domain == other.domain):
                raise TypeError("Domains differ")
            elif not np.all(self.window == other.window):
                raise TypeError("Windows differ")
            elif self.symbol != other.symbol:
                raise ValueError("Polynomial symbols differ")
            return other.coef
        return other

    def __init__(self, coef, domain=None, window=None, symbol='x'):
        """
        Initialize a polynomial object with coefficients, domain, window, and symbol.

        Parameters
        ----------
        coef : array_like
            Coefficients of the polynomial.
        domain : array_like or None, optional
            Domain of the polynomial.
        window : array_like or None, optional
            Window of the polynomial.
        symbol : str, optional
            Symbol representing the polynomial variable.

        Raises
        ------
        ValueError
            If domain or window has incorrect number of elements,
            or if symbol is not a valid Python identifier.
        TypeError
            If symbol is not a string.

        """
        [coef] = pu.as_series([coef], trim=False)
        self.coef = coef

        if domain is not None:
            [domain] = pu.as_series([domain], trim=False)
            if len(domain) != 2:
                raise ValueError("Domain has wrong number of elements.")
            self.domain = domain

        if window is not None:
            [window] = pu.as_series([window], trim=False)
            if len(window) != 2:
                raise ValueError("Window has wrong number of elements.")
            self.window = window

        # Validation for symbol
        try:
            if not symbol.isidentifier():
                raise ValueError(
                    "Symbol string must be a valid Python identifier"
                )
        except AttributeError:
            raise TypeError("Symbol must be a non-empty string")

        self._symbol = symbol
    # 定义对象的字符串表示形式，返回一个描述对象的字符串
    def __repr__(self):
        # 将系数转换为字符串表示，并去除开头和结尾的多余字符
        coef = repr(self.coef)[6:-1]
        # 将定义域转换为字符串表示，并去除开头和结尾的多余字符
        domain = repr(self.domain)[6:-1]
        # 将窗口大小转换为字符串表示，并去除开头和结尾的多余字符
        window = repr(self.window)[6:-1]
        # 获取对象的类名
        name = self.__class__.__name__
        # 返回格式化后的字符串表示，包括类名、系数、定义域、窗口大小和符号
        return (f"{name}({coef}, domain={domain}, window={window}, "
                f"symbol='{self.symbol}')")

    # 格式化对象的表示形式，支持 'ascii' 和 'unicode' 两种格式
    def __format__(self, fmt_str):
        # 如果格式字符串为空，则返回对象的字符串表示
        if fmt_str == '':
            return self.__str__()
        # 如果格式字符串不是 'ascii' 或 'unicode'，抛出值错误异常
        if fmt_str not in ('ascii', 'unicode'):
            raise ValueError(
                f"Unsupported format string '{fmt_str}' passed to "
                f"{self.__class__}.__format__. Valid options are "
                f"'ascii' and 'unicode'"
            )
        # 根据格式字符串调用相应的方法生成对应格式的字符串表示
        if fmt_str == 'ascii':
            return self._generate_string(self._str_term_ascii)
        return self._generate_string(self._str_term_unicode)

    # 返回对象的字符串表示，默认为 unicode 格式
    def __str__(self):
        # 如果使用 unicode 表示，则生成 unicode 格式的字符串表示
        if self._use_unicode:
            return self._generate_string(self._str_term_unicode)
        # 否则生成 ascii 格式的字符串表示
        return self._generate_string(self._str_term_ascii)

    # 根据给定的方法生成多项式的完整字符串表示
    def _generate_string(self, term_method):
        """
        Generate the full string representation of the polynomial, using
        ``term_method`` to generate each polynomial term.
        """
        # 获取打印选项中的行宽设置
        linewidth = np.get_printoptions().get('linewidth', 75)
        # 如果行宽小于 1，则将其设置为 1
        if linewidth < 1:
            linewidth = 1
        # 初始化输出字符串，将第一个系数转换为字符串后添加到输出中
        out = pu.format_float(self.coef[0])

        # 获取映射参数的偏移量和比例
        off, scale = self.mapparms()

        # 根据给定的方法和参数格式化项，获取缩放后的符号和是否需要括号
        scaled_symbol, needs_parens = self._format_term(pu.format_float,
                                                        off, scale)
        # 如果需要括号，则在符号周围添加括号
        if needs_parens:
            scaled_symbol = '(' + scaled_symbol + ')'

        # 遍历多项式的每个系数和幂次
        for i, coef in enumerate(self.coef[1:]):
            out += " "
            power = str(i + 1)
            # 处理多项式系数，如果系数为正则添加 '+'，否则添加 '-'
            try:
                if coef >= 0:
                    next_term = "+ " + pu.format_float(coef, parens=True)
                else:
                    next_term = "- " + pu.format_float(-coef, parens=True)
            except TypeError:
                next_term = f"+ {coef}"
            # 生成多项式的每一项，并使用给定方法生成术语
            next_term += term_method(power, scaled_symbol)
            # 计算添加下一项后当前行的长度
            line_len = len(out.split('\n')[-1]) + len(next_term)
            # 如果不是多项式的最后一项，则由于符号的存在，长度增加 2
            if i < len(self.coef[1:]) - 1:
                line_len += 2
            # 处理行长度超过设定的情况，进行换行处理
            if line_len >= linewidth:
                next_term = next_term.replace(" ", "\n", 1)
            out += next_term
        # 返回生成的多项式字符串表示
        return out
    def _str_term_unicode(cls, i, arg_str):
        """
        使用 Unicode 字符表示单个多项式项的字符串形式，包括上标和下标。
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis_name, or override "
                "_str_term_unicode(cls, i, arg_str)"
            )
        返回格式化的多项式项字符串，包括基函数名称和下标的 Unicode 表示
        return (f"·{cls.basis_name}{i.translate(cls._subscript_mapping)}"
                f"({arg_str})")

    @classmethod
    def _str_term_ascii(cls, i, arg_str):
        """
        使用 ** 和 _ 表示上标和下标，生成单个多项式项的 ASCII 字符串表示。
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis_name, or override "
                "_str_term_ascii(cls, i, arg_str)"
            )
        返回格式化的多项式项字符串，包括基函数名称和下标的 ASCII 表示
        return f" {cls.basis_name}_{i}({arg_str})"

    @classmethod
    def _repr_latex_term(cls, i, arg_str, needs_parens):
        """
        生成单个多项式项的 LaTeX 字符串表示。
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis name, or override "
                "_repr_latex_term(i, arg_str, needs_parens)")
        # 因为我们总是添加括号，所以不需要关心表达式是否需要括号
        return f"{{{cls.basis_name}}}_{{{i}}}({arg_str})"

    @staticmethod
    def _repr_latex_scalar(x, parens=False):
        """
        生成 LaTeX 表示的标量值。
        TODO: 在这个函数处理指数之前，我们禁用数学格式化。
        """
        return r'\text{{{}}}'.format(pu.format_float(x, parens=parens))

    def _format_term(self, scalar_format: Callable, off: float, scale: float):
        """
        格式化展开中的单个项。
        """
        if off == 0 and scale == 1:
            term = self.symbol
            needs_parens = False
        elif scale == 1:
            term = f"{scalar_format(off)} + {self.symbol}"
            needs_parens = True
        elif off == 0:
            term = f"{scalar_format(scale)}{self.symbol}"
            needs_parens = True
        else:
            term = (
                f"{scalar_format(off)} + "
                f"{scalar_format(scale)}{self.symbol}"
            )
            needs_parens = True
        return term, needs_parens
    # 定义用于生成 LaTeX 表示的方法
    def _repr_latex_(self):
        # 获取基函数的偏移和缩放参数
        off, scale = self.mapparms()
        # 格式化表示基函数的项
        term, needs_parens = self._format_term(self._repr_latex_scalar,
                                               off, scale)

        # 静音模式，将文本用浅灰色显示
        mute = r"\color{{LightGray}}{{{}}}".format

        parts = []
        # 遍历系数列表
        for i, c in enumerate(self.coef):
            # 防止+和-符号重复出现
            if i == 0:
                coef_str = f"{self._repr_latex_scalar(c)}"
            elif not isinstance(c, numbers.Real):
                coef_str = f" + ({self._repr_latex_scalar(c)})"
            elif c >= 0:
                coef_str = f" + {self._repr_latex_scalar(c, parens=True)}"
            else:
                coef_str = f" - {self._repr_latex_scalar(-c, parens=True)}"

            # 生成项的字符串表示
            term_str = self._repr_latex_term(i, term, needs_parens)
            if term_str == '1':
                part = coef_str
            else:
                part = rf"{coef_str}\,{term_str}"

            # 如果系数为0，则使用静音模式
            if c == 0:
                part = mute(part)

            parts.append(part)

        # 如果存在部件，则将它们连接成字符串
        if parts:
            body = ''.join(parts)
        else:
            # 如果系数全为0，则返回'0'
            body = '0'

        # 返回 LaTeX 格式的表达式
        return rf"${self.symbol} \mapsto {body}$"



    # Pickle 和 copy

    def __getstate__(self):
        # 复制对象的字典表示
        ret = self.__dict__.copy()
        # 复制系数列表
        ret['coef'] = self.coef.copy()
        # 复制定义域
        ret['domain'] = self.domain.copy()
        # 复制窗口
        ret['window'] = self.window.copy()
        # 复制符号
        ret['symbol'] = self.symbol
        return ret

    def __setstate__(self, dict):
        # 恢复对象状态
        self.__dict__ = dict

    # 调用对象

    def __call__(self, arg):
        # 将参数映射到定义域和窗口
        arg = pu.mapdomain(arg, self.domain, self.window)
        # 计算对象在参数处的值
        return self._val(arg, self.coef)

    def __iter__(self):
        # 返回系数列表的迭代器
        return iter(self.coef)

    def __len__(self):
        # 返回系数列表的长度
        return len(self.coef)

    # 数值属性

    def __neg__(self):
        # 返回对象的负值
        return self.__class__(
            -self.coef, self.domain, self.window, self.symbol
        )

    def __pos__(self):
        # 返回对象本身
        return self

    def __add__(self, other):
        # 获取另一个对象的系数
        othercoef = self._get_coefficients(other)
        try:
            # 尝试执行加法操作
            coef = self._add(self.coef, othercoef)
        except Exception:
            # 如果出现异常，则返回NotImplemented
            return NotImplemented
        # 返回新对象，带有相加后的系数
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __sub__(self, other):
        # 获取另一个对象的系数
        othercoef = self._get_coefficients(other)
        try:
            # 尝试执行减法操作
            coef = self._sub(self.coef, othercoef)
        except Exception:
            # 如果出现异常，则返回NotImplemented
            return NotImplemented
        # 返回新对象，带有相减后的系数
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __mul__(self, other):
        # 获取另一个对象的系数
        othercoef = self._get_coefficients(other)
        try:
            # 尝试执行乘法操作
            coef = self._mul(self.coef, othercoef)
        except Exception:
            # 如果出现异常，则返回NotImplemented
            return NotImplemented
        # 返回新对象，带有相乘后的系数
        return self.__class__(coef, self.domain, self.window, self.symbol)
    # 实现真除法的特殊方法，处理除数不是数字或布尔类型的情况，抛出TypeError异常
    def __truediv__(self, other):
        if not isinstance(other, numbers.Number) or isinstance(other, bool):
            raise TypeError(
                f"unsupported types for true division: "
                f"'{type(self)}', '{type(other)}'"
            )
        # 调用整除的特殊方法处理
        return self.__floordiv__(other)

    # 实现整除的特殊方法，调用__divmod__方法获得结果
    def __floordiv__(self, other):
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    # 实现取模的特殊方法，调用__divmod__方法获得结果
    def __mod__(self, other):
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    # 实现divmod的特殊方法，进行多项式除法计算，处理ZeroDivisionError和其他异常
    def __divmod__(self, other):
        # 获取其他操作数的系数
        othercoef = self._get_coefficients(other)
        try:
            # 调用内部方法执行多项式除法
            quo, rem = self._div(self.coef, othercoef)
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        # 创建新的多项式实例表示商和余数
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return quo, rem

    # 实现幂运算的特殊方法，调用内部方法计算多项式的幂
    def __pow__(self, other):
        coef = self._pow(self.coef, other, maxpower=self.maxpower)
        res = self.__class__(coef, self.domain, self.window, self.symbol)
        return res

    # 实现右加法的特殊方法，调用内部方法执行多项式加法
    def __radd__(self, other):
        try:
            coef = self._add(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    # 实现右减法的特殊方法，调用内部方法执行多项式减法
    def __rsub__(self, other):
        try:
            coef = self._sub(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    # 实现右乘法的特殊方法，调用内部方法执行多项式乘法
    def __rmul__(self, other):
        try:
            coef = self._mul(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    # 实现右真除法的特殊方法，委托给__rfloordiv__方法处理
    def __rdiv__(self, other):
        return self.__rfloordiv__(other)

    # 实现右真除法的特殊方法，返回NotImplemented表示不支持
    def __rtruediv__(self, other):
        return NotImplemented

    # 实现右整除的特殊方法，调用__rdivmod__方法获得结果
    def __rfloordiv__(self, other):
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    # 实现右取模的特殊方法，调用__rdivmod__方法获得结果
    def __rmod__(self, other):
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[1]
    def __rdivmod__(self, other):
        try:
            # 调用自定义的 _div 方法进行除法运算
            quo, rem = self._div(other, self.coef)
        except ZeroDivisionError:
            # 如果除数为零，则抛出 ZeroDivisionError 异常
            raise
        except Exception:
            # 捕获所有其他异常，返回 NotImplemented 表示不支持该操作
            return NotImplemented
        # 根据结果创建新的实例并返回
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return quo, rem

    def __eq__(self, other):
        # 检查是否与另一个对象相等的方法
        res = (isinstance(other, self.__class__) and  # 检查类型是否相同
               np.all(self.domain == other.domain) and  # 比较 domain 属性是否相等
               np.all(self.window == other.window) and  # 比较 window 属性是否相等
               (self.coef.shape == other.coef.shape) and  # 比较 coef 的形状是否相等
               np.all(self.coef == other.coef) and  # 比较 coef 的值是否相等
               (self.symbol == other.symbol))  # 比较 symbol 属性是否相等
        return res

    def __ne__(self, other):
        # 检查是否与另一个对象不相等的方法，通过调用 __eq__ 方法的结果取反得到
        return not self.__eq__(other)

    #
    # Extra methods.
    #

    def copy(self):
        """Return a copy.

        Returns
        -------
        new_series : series
            Copy of self.

        """
        # 返回当前对象的一个副本
        return self.__class__(self.coef, self.domain, self.window, self.symbol)

    def degree(self):
        """The degree of the series.

        .. versionadded:: 1.5.0

        Returns
        -------
        degree : int
            Degree of the series, one less than the number of coefficients.

        Examples
        --------

        Create a polynomial object for ``1 + 7*x + 4*x**2``:

        >>> poly = np.polynomial.Polynomial([1, 7, 4])
        >>> print(poly)
        1.0 + 7.0·x + 4.0·x²
        >>> poly.degree()
        2

        Note that this method does not check for non-zero coefficients.
        You must trim the polynomial to remove any trailing zeroes:

        >>> poly = np.polynomial.Polynomial([1, 7, 0])
        >>> print(poly)
        1.0 + 7.0·x + 0.0·x²
        >>> poly.degree()
        2
        >>> poly.trim().degree()
        1

        """
        # 返回当前系列的阶数，即系数数量减一
        return len(self) - 1

    def cutdeg(self, deg):
        """Truncate series to the given degree.

        Reduce the degree of the series to `deg` by discarding the
        high order terms. If `deg` is greater than the current degree a
        copy of the current series is returned. This can be useful in least
        squares where the coefficients of the high degree terms may be very
        small.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        deg : non-negative int
            The series is reduced to degree `deg` by discarding the high
            order terms. The value of `deg` must be a non-negative integer.

        Returns
        -------
        new_series : series
            New instance of series with reduced degree.

        """
        # 调用 truncate 方法将当前系列截断至给定的阶数 deg+1
        return self.truncate(deg + 1)
    # 修剪级数，移除尾部系数直到找到一个绝对值大于 `tol` 的系数或者到达系列的开头
    # 如果所有系数都被移除，则将系列设置为 `[0]`。返回一个新的系列实例，保持当前实例不变。
    def trim(self, tol=0):
        """Remove trailing coefficients

        Remove trailing coefficients until a coefficient is reached whose
        absolute value greater than `tol` or the beginning of the series is
        reached. If all the coefficients would be removed the series is set
        to ``[0]``. A new series instance is returned with the new
        coefficients.  The current instance remains unchanged.

        Parameters
        ----------
        tol : non-negative number.
            All trailing coefficients less than `tol` will be removed.

        Returns
        -------
        new_series : series
            New instance of series with trimmed coefficients.

        """
        # 调用外部函数 pu.trimcoef() 对当前系列的系数进行修剪
        coef = pu.trimcoef(self.coef, tol)
        # 返回一个新的系列实例，使用修剪后的系数以及当前实例的其它属性
        return self.__class__(coef, self.domain, self.window, self.symbol)

    # 将系列截断至指定长度 `size`
    def truncate(self, size):
        """Truncate series to length `size`.

        Reduce the series to length `size` by discarding the high
        degree terms. The value of `size` must be a positive integer. This
        can be useful in least squares where the coefficients of the
        high degree terms may be very small.

        Parameters
        ----------
        size : positive int
            The series is reduced to length `size` by discarding the high
            degree terms. The value of `size` must be a positive integer.

        Returns
        -------
        new_series : series
            New instance of series with truncated coefficients.

        """
        # 将 size 转换为整数
        isize = int(size)
        # 如果转换后的大小与原始大小不同或者小于 1，则引发 ValueError
        if isize != size or isize < 1:
            raise ValueError("size must be a positive integer")
        # 如果要截断的长度大于等于当前系列长度，则保持系数不变
        if isize >= len(self.coef):
            coef = self.coef
        else:
            # 否则，截取前 isize 个系数
            coef = self.coef[:isize]
        # 返回一个新的系列实例，使用截断后的系数以及当前实例的其它属性
        return self.__class__(coef, self.domain, self.window, self.symbol)
    # 将系列转换为不同种类和/或域和/或窗口。

    # Parameters
    # ----------
    # domain : array_like, optional
    #     转换后系列的域。如果值为None，则使用`kind`的默认域。
    # kind : class, optional
    #     要转换为的多项式系列类型类。如果kind为None，则使用当前实例的类。
    # window : array_like, optional
    #     转换后系列的窗口。如果值为None，则使用`kind`的默认窗口。

    # Returns
    # -------
    # new_series : series
    #     返回的类可以与当前实例的类型不同，也可以具有不同的域和/或不同的窗口。

    # Notes
    # -----
    # 在域和类类型之间的转换可能导致数值上不稳定的系列。

    def convert(self, domain=None, kind=None, window=None):
        if kind is None:
            kind = self.__class__
        if domain is None:
            domain = kind.domain
        if window is None:
            window = kind.window
        return self(kind.identity(domain, window=window, symbol=self.symbol))

    # 返回映射参数。

    # 返回的值定义了应用于系列评估之前的输入参数的线性映射``off + scl*x``。
    # 映射取决于``domain``和``window``；如果当前``domain``等于``window``，则结果映射为恒等映射。
    # 如果系列实例的系数在类外部独立使用，则必须将线性函数替换为标准基多项式表示中的``x``。

    # Returns
    # -------
    # off, scl : float or complex
    #     映射函数由``off + scl*x``定义。

    # Notes
    # -----
    # 如果当前域是区间``[l1, r1]``，窗口是``[l2, r2]``，则线性映射函数``L``由以下方程定义：
    # 
    #     L(l1) = l2
    #     L(r1) = r2
    def mapparms(self):
        return pu.mapparms(self.domain, self.window)
    def integ(self, m=1, k=[], lbnd=None):
        """
        Integrate.

        Return a series instance that is the definite integral of the
        current series.

        Parameters
        ----------
        m : non-negative int
            The number of integrations to perform.
        k : array_like
            Integration constants. The first constant is applied to the
            first integration, the second to the second, and so on. The
            list of values must less than or equal to `m` in length and any
            missing values are set to zero.
        lbnd : Scalar
            The lower bound of the definite integral.

        Returns
        -------
        new_series : series
            A new series representing the integral. The domain is the same
            as the domain of the integrated series.
        """
        # Extract offset and scale parameters from the series
        off, scl = self.mapparms()

        # Set default lower bound for integration if not provided
        if lbnd is None:
            lbnd = 0
        else:
            lbnd = off + scl * lbnd  # Adjust lower bound based on series parameters

        # Perform the integration on the coefficient array
        coef = self._int(self.coef, m, k, lbnd, 1. / scl)

        # Return a new series instance representing the integral
        return self.__class__(coef, self.domain, self.window, self.symbol)


    def deriv(self, m=1):
        """
        Differentiate.

        Return a series instance that is the derivative of the current
        series.

        Parameters
        ----------
        m : non-negative int
            Find the derivative of order `m`.

        Returns
        -------
        new_series : series
            A new series representing the derivative. The domain is the same
            as the domain of the differentiated series.
        """
        # Extract offset and scale parameters from the series
        off, scl = self.mapparms()

        # Compute the derivative of the coefficient array
        coef = self._der(self.coef, m, scl)

        # Return a new series instance representing the derivative
        return self.__class__(coef, self.domain, self.window, self.symbol)


    def roots(self):
        """
        Return the roots of the series polynomial.

        Compute the roots for the series. Note that the accuracy of the
        roots decreases the further outside the `domain` they lie.

        Returns
        -------
        roots : ndarray
            Array containing the roots of the series.
        """
        # Compute the roots of the polynomial represented by the coefficient array
        roots = self._roots(self.coef)

        # Map the computed roots to the correct domain using mapdomain function
        return pu.mapdomain(roots, self.window, self.domain)
    def linspace(self, n=100, domain=None):
        """Return x, y values at equally spaced points in domain.

        Returns the x, y values at `n` linearly spaced points across the
        domain.  Here y is the value of the polynomial at the points x. By
        default the domain is the same as that of the series instance.
        This method is intended mostly as a plotting aid.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        n : int, optional
            Number of point pairs to return. The default value is 100.
        domain : {None, array_like}, optional
            If not None, the specified domain is used instead of that of
            the calling instance. It should be of the form ``[beg,end]``.
            The default is None which case the class domain is used.

        Returns
        -------
        x, y : ndarray
            x is equal to linspace(self.domain[0], self.domain[1], n) and
            y is the series evaluated at element of x.

        """
        if domain is None:
            domain = self.domain  # 如果 domain 参数为 None，则使用当前实例的 domain 属性
        x = np.linspace(domain[0], domain[1], n)  # 生成一个在指定 domain 区间内的 n 个等间距点的数组
        y = self(x)  # 计算这些 x 点上多项式的值
        return x, y

    @classmethod
    @classmethod
    def fromroots(cls, roots, domain=[], window=None, symbol='x'):
        """Return series instance that has the specified roots.

        Returns a series representing the product
        ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is a
        list of roots.

        Parameters
        ----------
        roots : array_like
            List of roots.
        domain : {[], None, array_like}, optional
            Domain for the resulting series. If None the domain is the
            interval from the smallest root to the largest. If [] the
            domain is the class domain. The default is [].
        window : {None, array_like}, optional
            Window for the returned series. If None the class window is
            used. The default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            Series with the specified roots.

        """
        [roots] = pu.as_series([roots], trim=False)  # 将输入的 roots 转换为 series 格式
        if domain is None:
            domain = pu.getdomain(roots)  # 如果 domain 参数为 None，则根据 roots 计算 domain
        elif type(domain) is list and len(domain) == 0:
            domain = cls.domain  # 如果 domain 是空列表，则使用类的 domain 属性

        if window is None:
            window = cls.window  # 如果 window 参数为 None，则使用类的 window 属性

        deg = len(roots)  # 计算 roots 的个数
        off, scl = pu.mapparms(domain, window)  # 计算 domain 和 window 的映射参数
        rnew = off + scl * roots  # 对 roots 进行线性映射
        coef = cls._fromroots(rnew) / scl**deg  # 计算系数
        return cls(coef, domain=domain, window=window, symbol=symbol)  # 返回新的 series 实例
    @classmethod
    def identity(cls, domain=None, window=None, symbol='x'):
        """Identity function.

        If ``p`` is the returned series, then ``p(x) == x`` for all
        values of x.

        Parameters
        ----------
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
             Series of representing the identity.

        """
        # 如果未指定 domain，则使用类属性中的 domain
        if domain is None:
            domain = cls.domain
        # 如果未指定 window，则使用类属性中的 window
        if window is None:
            window = cls.window
        # 根据给定的 window 和 domain 计算偏移量和缩放因子
        off, scl = pu.mapparms(window, domain)
        # 使用计算得到的偏移量和缩放因子生成系数
        coef = cls._line(off, scl)
        # 返回一个新的系列对象，表示恒等函数
        return cls(coef, domain, window, symbol)

    @classmethod
    def basis(cls, deg, domain=None, window=None, symbol='x'):
        """Series basis polynomial of degree `deg`.

        Returns the series representing the basis polynomial of degree `deg`.

        .. versionadded:: 1.7.0

        Parameters
        ----------
        deg : int
            Degree of the basis polynomial for the series. Must be >= 0.
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            A series with the coefficient of the `deg` term set to one and
            all others zero.

        """
        # 如果未指定 domain，则使用类属性中的 domain
        if domain is None:
            domain = cls.domain
        # 如果未指定 window，则使用类属性中的 window
        if window is None:
            window = cls.window
        # 将 deg 转换为整数类型
        ideg = int(deg)

        # 如果 ideg 不等于 deg 或者 ideg 小于 0，则抛出数值错误
        if ideg != deg or ideg < 0:
            raise ValueError("deg must be non-negative integer")
        # 返回一个新的系列对象，表示指定阶数的基础多项式
        return cls([0]*ideg + [1], domain, window, symbol)
    def cast(cls, series, domain=None, window=None):
        """Convert series to series of this class.

        The `series` is expected to be an instance of some polynomial
        series of one of the types supported by by the numpy.polynomial
        module, but could be some other class that supports the convert
        method.

        .. versionadded:: 1.7.0

        Parameters
        ----------
        series : series
            The series instance to be converted.
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.

        Returns
        -------
        new_series : series
            A series of the same kind as the calling class and equal to
            `series` when evaluated.

        See Also
        --------
        convert : similar instance method

        """
        # 如果未提供特定的 domain，则使用类的默认 domain
        if domain is None:
            domain = cls.domain
        # 如果未提供特定的 window，则使用类的默认 window
        if window is None:
            window = cls.window
        # 调用 series 对象的 convert 方法，将其转换为当前类的实例
        return series.convert(domain, cls, window)
```