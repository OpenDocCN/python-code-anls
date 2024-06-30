# `D:\src\scipysrc\sympy\sympy\polys\polytools.py`

```
# 提供友好的公共接口用于多项式函数操作

from functools import wraps, reduce  # 导入 wraps 和 reduce 函数
from operator import mul  # 导入 mul 运算符
from typing import Optional  # 导入 Optional 类型提示

from sympy.core import (  # 导入 SymPy 核心模块的一些类和函数
    S, Expr, Add, Tuple
)
from sympy.core.basic import Basic  # 导入 SymPy 核心模块中的 Basic 类
from sympy.core.decorators import _sympifyit  # 导入 SymPy 核心模块中的 _sympifyit 装饰器
from sympy.core.exprtools import Factors, factor_nc, factor_terms  # 导入 SymPy 核心模块中的因子相关函数
from sympy.core.evalf import (  # 导入 SymPy 核心模块中的数值计算函数
    pure_complex, evalf, fastlog, _evalf_with_bounded_error, quad_to_mpmath)
from sympy.core.function import Derivative  # 导入 SymPy 核心模块中的 Derivative 类
from sympy.core.mul import Mul, _keep_coeff  # 导入 SymPy 核心模块中的乘法相关函数
from sympy.core.intfunc import ilcm  # 导入 SymPy 核心模块中的整数相关函数
from sympy.core.numbers import I, Integer, equal_valued  # 导入 SymPy 核心模块中的数值类型和比较函数
from sympy.core.relational import Relational, Equality  # 导入 SymPy 核心模块中的关系运算类和相等性类
from sympy.core.sorting import ordered  # 导入 SymPy 核心模块中的排序函数
from sympy.core.symbol import Dummy, Symbol  # 导入 SymPy 核心模块中的符号类
from sympy.core.sympify import sympify, _sympify  # 导入 SymPy 核心模块中的符号转换函数
from sympy.core.traversal import preorder_traversal, bottom_up  # 导入 SymPy 核心模块中的遍历函数
from sympy.logic.boolalg import BooleanAtom  # 导入 SymPy 逻辑模块中的布尔原子类
from sympy.polys import polyoptions as options  # 导入 SymPy 多项式模块中的选项配置
from sympy.polys.constructor import construct_domain  # 导入 SymPy 多项式模块中的域构造函数
from sympy.polys.domains import FF, QQ, ZZ  # 导入 SymPy 多项式模块中的有限域和整数域类
from sympy.polys.domains.domainelement import DomainElement  # 导入 SymPy 多项式模块中的域元素类
from sympy.polys.fglmtools import matrix_fglm  # 导入 SymPy 多项式模块中的 FGLM 矩阵函数
from sympy.polys.groebnertools import groebner as _groebner  # 导入 SymPy 多项式模块中的格罗布纳基函数
from sympy.polys.monomials import Monomial  # 导入 SymPy 多项式模块中的单项式类
from sympy.polys.orderings import monomial_key  # 导入 SymPy 多项式模块中的单项式排序函数
from sympy.polys.polyclasses import DMP, DMF, ANP  # 导入 SymPy 多项式模块中的多项式类
from sympy.polys.polyerrors import (  # 导入 SymPy 多项式模块中的错误类和异常
    OperationNotSupported, DomainError,
    CoercionFailed, UnificationFailed,
    GeneratorsNeeded, PolynomialError,
    MultivariatePolynomialError,
    ExactQuotientFailed,
    PolificationFailed,
    ComputationFailed,
    GeneratorsError,
)
from sympy.polys.polyutils import (  # 导入 SymPy 多项式模块中的实用函数
    basic_from_dict,
    _sort_gens,
    _unify_gens,
    _dict_reorder,
    _dict_from_expr,
    _parallel_dict_from_expr,
)
from sympy.polys.rationaltools import together  # 导入 SymPy 多项式模块中的有理数工具函数
from sympy.polys.rootisolation import dup_isolate_real_roots_list  # 导入 SymPy 多项式模块中的实根隔离函数
from sympy.utilities import group, public, filldedent  # 导入 SymPy 实用工具模块中的一些函数和修饰器
from sympy.utilities.exceptions import sympy_deprecation_warning  # 导入 SymPy 异常模块中的警告类
from sympy.utilities.iterables import iterable, sift  # 导入 SymPy 实用工具模块中的可迭代对象处理函数

# Required to avoid errors
import sympy.polys  # 导入 sympy.polys 模块，用于避免错误

import mpmath  # 导入 mpmath 数学库
from mpmath.libmp.libhyper import NoConvergence  # 导入 mpmath 中的 NoConvergence 异常类



def _polifyit(func):  # 定义内部函数 _polifyit，用作装饰器
    @wraps(func)  # 使用 functools.wraps 装饰器，保留被装饰函数的元信息
    # 定义一个装饰器函数 wrapper，接受两个参数 f 和 g
    def wrapper(f, g):
        # 将 g 转换为符号表达式
        g = _sympify(g)
        # 如果 g 是多项式（Poly）类型，则调用 func 函数处理 f 和 g
        if isinstance(g, Poly):
            return func(f, g)
        # 如果 g 是整数（Integer）类型
        elif isinstance(g, Integer):
            # 将 g 转换为 f 所在环域中的表达式
            g = f.from_expr(g, *f.gens, domain=f.domain)
            # 调用 func 函数处理 f 和 g
            return func(f, g)
        # 如果 g 是表达式（Expr）类型
        elif isinstance(g, Expr):
            try:
                # 尝试将 g 转换为 f 所在环域中的表达式
                g = f.from_expr(g, *f.gens)
            except PolynomialError:
                # 如果 g 是矩阵（Matrix）类型，则返回 NotImplemented
                if g.is_Matrix:
                    return NotImplemented
                # 否则，使用 f 的表达式方法名调用 g
                expr_method = getattr(f.as_expr(), func.__name__)
                result = expr_method(g)
                # 如果结果不是 NotImplemented，则发出 sympy_deprecation_warning
                if result is not NotImplemented:
                    sympy_deprecation_warning(
                        """
                        在二元操作中混合多项式和非多项式表达式已被弃用。请显式将非多项式操作数转换为多项式（使用 as_poly()）或将多项式转换为表达式（使用 as_expr()）。
                        """,
                        deprecated_since_version="1.6",
                        active_deprecations_target="deprecated-poly-nonpoly-binary-operations",
                    )
                return result
            else:
                # 如果成功将 g 转换为表达式，则调用 func 函数处理 f 和 g
                return func(f, g)
        else:
            # 如果 g 不是上述任何类型，则返回 NotImplemented
            return NotImplemented
    # 返回装饰器函数 wrapper
    return wrapper
# 定义一个公共类 Poly，继承自 Basic 类
@public
class Poly(Basic):
    """
    用于表示和操作多项式表达式的通用类。

    参见 :ref:`polys-docs` 获取一般文档信息。

    Poly 是 Basic 的子类，但实例可以通过 :py:meth:`~.Poly.as_expr` 方法转换为 Expr。

    .. deprecated:: 1.6
       在二元操作中将 Poly 与非 Poly 对象结合使用已被弃用。请首先显式将这两个对象转换为 Poly 或 Expr。参见 :ref:`deprecated-poly-nonpoly-binary-operations`。

    Examples
    ========

    >>> from sympy import Poly
    >>> from sympy.abc import x, y

    创建一个一元多项式：

    >>> Poly(x*(x**2 + x - 1)**2)
    Poly(x**5 + 2*x**4 - x**3 - 2*x**2 + x, x, domain='ZZ')

    创建一个指定域的一元多项式：

    >>> from sympy import sqrt
    >>> Poly(x**2 + 2*x + sqrt(3), domain='R')
    Poly(1.0*x**2 + 2.0*x + 1.73205080756888, x, domain='RR')

    创建一个多元多项式：

    >>> Poly(y*x**2 + x*y + 1)
    Poly(x**2*y + x*y + 1, x, y, domain='ZZ')

    创建一个一元多项式，其中 y 是常数：

    >>> Poly(y*x**2 + x*y + 1,x)
    Poly(y*x**2 + y*x + 1, x, domain='ZZ[y]')

    您可以将上述多项式作为 y 的函数进行评估：

    >>> Poly(y*x**2 + x*y + 1,x).eval(2)
    6*y + 1

    See Also
    ========

    sympy.core.expr.Expr

    """

    __slots__ = ('rep', 'gens')  # 限定实例只能有 rep 和 gens 这两个属性

    is_commutative = True  # 多项式是可交换的
    is_Poly = True  # 标识这是一个多项式类
    _op_priority = 10.001  # 运算优先级为 10.001

    def __new__(cls, rep, *gens, **args):
        """根据给定的 rep 创建一个新的多项式实例。"""
        opt = options.build_options(gens, args)  # 构建选项

        if 'order' in opt:
            raise NotImplementedError("'order' keyword is not implemented yet")  # 抛出未实现错误

        # 根据 rep 的类型选择不同的初始化方法
        if isinstance(rep, (DMP, DMF, ANP, DomainElement)):
            return cls._from_domain_element(rep, opt)
        elif iterable(rep, exclude=str):
            if isinstance(rep, dict):
                return cls._from_dict(rep, opt)
            else:
                return cls._from_list(list(rep), opt)
        else:
            rep = sympify(rep, evaluate=type(rep) is not str)  # 符号化 rep

            if rep.is_Poly:
                return cls._from_poly(rep, opt)
            else:
                return cls._from_expr(rep, opt)

    # Poly 类不会将其参数传递给 Basic.__new__ 来存储在 _args 中，因此需要在这里使用一个 args 属性来模拟这些参数，这些参数来自 rep 和 gens，它们是实例属性。
    # 这意味着我们需要定义 _hashable_content。_hashable_content 包括 rep 和 gens，但 args 使用 expr 而不是 rep（expr 是 rep 的 Basic 版本）。
    # 通过 args 传递 expr 意味着 Basic 类的方法如 subs 应该正常工作。而使用 rep 可以使 Poly 类比 Basic 类更有效率，因为它避免了创建一个仅用于可哈希的 Basic 实例。

    @classmethod
    def new(cls, rep, *gens):
        """Construct :class:`Poly` instance from raw representation. """
        # 检查 rep 是否为 DMP 类的实例，如果不是则抛出 PolynomialError 异常
        if not isinstance(rep, DMP):
            raise PolynomialError(
                "invalid polynomial representation: %s" % rep)
        # 检查 rep 的级数是否与 gens 参数个数减一相符，如果不符则抛出 PolynomialError 异常
        elif rep.lev != len(gens) - 1:
            raise PolynomialError("invalid arguments: %s, %s" % (rep, gens))

        # 创建一个新的 Basic 类实例
        obj = Basic.__new__(cls)
        obj.rep = rep
        obj.gens = gens

        return obj

    @property
    def expr(self):
        # 返回基于 rep 属性生成的表达式
        return basic_from_dict(self.rep.to_sympy_dict(), *self.gens)

    @property
    def args(self):
        # 返回包含表达式和 gens 参数的元组
        return (self.expr,) + self.gens

    def _hashable_content(self):
        # 返回一个元组，包含 rep 和 gens 属性，用于哈希
        return (self.rep,) + self.gens

    @classmethod
    def from_dict(cls, rep, *gens, **args):
        """Construct a polynomial from a ``dict``. """
        # 使用 gens 和 args 构建选项 opt
        opt = options.build_options(gens, args)
        # 调用 _from_dict 方法创建多项式对象并返回
        return cls._from_dict(rep, opt)

    @classmethod
    def from_list(cls, rep, *gens, **args):
        """Construct a polynomial from a ``list``. """
        # 使用 gens 和 args 构建选项 opt
        opt = options.build_options(gens, args)
        # 调用 _from_list 方法创建多项式对象并返回
        return cls._from_list(rep, opt)

    @classmethod
    def from_poly(cls, rep, *gens, **args):
        """Construct a polynomial from a polynomial. """
        # 使用 gens 和 args 构建选项 opt
        opt = options.build_options(gens, args)
        # 调用 _from_poly 方法创建多项式对象并返回
        return cls._from_poly(rep, opt)

    @classmethod
    def from_expr(cls, rep, *gens, **args):
        """Construct a polynomial from an expression. """
        # 使用 gens 和 args 构建选项 opt
        opt = options.build_options(gens, args)
        # 调用 _from_expr 方法创建多项式对象并返回
        return cls._from_expr(rep, opt)

    @classmethod
    def _from_dict(cls, rep, opt):
        """Construct a polynomial from a ``dict``. """
        # 从选项 opt 中获取 gens
        gens = opt.gens

        # 如果 gens 为空，则抛出 GeneratorsNeeded 异常
        if not gens:
            raise GeneratorsNeeded(
                "Cannot initialize from 'dict' without generators")

        # 计算 gens 的级数
        level = len(gens) - 1
        # 获取域 domain
        domain = opt.domain

        # 如果 domain 为 None，则调用 construct_domain 函数构建域
        if domain is None:
            domain, rep = construct_domain(rep, opt=opt)
        else:
            # 否则，对于 rep 中的每个项，将其值转换为 domain 的值
            for monom, coeff in rep.items():
                rep[monom] = domain.convert(coeff)

        # 使用 DMP.from_dict 方法创建多项式对象，并返回 cls.new 方法创建的实例
        return cls.new(DMP.from_dict(rep, level, domain), *gens)

    @classmethod
    def _from_list(cls, rep, opt):
        """Construct a polynomial from a ``list``. """
        # 从选项 opt 中获取 gens
        gens = opt.gens

        # 如果 gens 为空，则抛出 GeneratorsNeeded 异常
        if not gens:
            raise GeneratorsNeeded(
                "Cannot initialize from 'list' without generators")
        # 如果 gens 的长度不为 1，则抛出 MultivariatePolynomialError 异常
        elif len(gens) != 1:
            raise MultivariatePolynomialError(
                "'list' representation not supported")

        # 计算 gens 的级数
        level = len(gens) - 1
        # 获取域 domain
        domain = opt.domain

        # 如果 domain 为 None，则调用 construct_domain 函数构建域
        if domain is None:
            domain, rep = construct_domain(rep, opt=opt)
        else:
            # 否则，将 rep 中的每个值转换为 domain 的值
            rep = list(map(domain.convert, rep))

        # 使用 DMP.from_list 方法创建多项式对象，并返回 cls.new 方法创建的实例
        return cls.new(DMP.from_list(rep, level, domain), *gens)

    @classmethod
    def _from_poly(cls, rep, opt):
        """Construct a polynomial from a polynomial."""
        # 如果给定的类和 rep 的类不同，将 rep 转换为当前类的实例
        if cls != rep.__class__:
            rep = cls.new(rep.rep, *rep.gens)

        # 从 opt 参数中获取生成元和域信息
        gens = opt.gens
        field = opt.field
        domain = opt.domain

        # 如果给定了生成元，并且 rep 的生成元与之不同，根据情况进行重排或者重新构建
        if gens and rep.gens != gens:
            if set(rep.gens) != set(gens):
                # 如果生成元不匹配，将表达式转换为当前类的实例
                return cls._from_expr(rep.as_expr(), opt)
            else:
                # 否则按照给定的生成元顺序重新排列 rep
                rep = rep.reorder(*gens)

        # 如果 opt 中包含 'domain'，并且 domain 不为空，则设置 rep 的域
        if 'domain' in opt and domain:
            rep = rep.set_domain(domain)
        elif field is True:
            # 如果 field 标志为 True，则将 rep 转换为域中的元素
            rep = rep.to_field()

        # 返回构建好的多项式 rep
        return rep

    @classmethod
    def _from_expr(cls, rep, opt):
        """Construct a polynomial from an expression."""
        # 从表达式 rep 和选项 opt 中获取字典表示
        rep, opt = _dict_from_expr(rep, opt)
        # 使用字典表示构建多项式
        return cls._from_dict(rep, opt)

    @classmethod
    def _from_domain_element(cls, rep, opt):
        # 获取生成元和域信息
        gens = opt.gens
        domain = opt.domain

        # 计算生成元的层级
        level = len(gens) - 1
        # 将 rep 转换为域元素，并构建一个列表
        rep = [domain.convert(rep)]

        # 使用列表中的元素构建一个新的多项式对象
        return cls.new(DMP.from_list(rep, level, domain), *gens)

    def __hash__(self):
        # 调用父类的哈希方法
        return super().__hash__()

    @property
    def free_symbols(self):
        """
        Free symbols of a polynomial expression.
        
        Examples
        ========
        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x**2 + 1).free_symbols
        {x}
        >>> Poly(x**2 + y).free_symbols
        {x, y}
        >>> Poly(x**2 + y, x).free_symbols
        {x, y}
        >>> Poly(x**2 + y, x, z).free_symbols
        {x, y}
        """
        # 初始化一个空集合来存储自由符号
        symbols = set()
        # 获取多项式的生成元
        gens = self.gens
        # 遍历每个生成元的每个单项式，如果单项式非零，则将其自由符号添加到集合中
        for i in range(len(gens)):
            for monom in self.monoms():
                if monom[i]:
                    symbols |= gens[i].free_symbols
                    break

        # 返回生成元的自由符号与域中的自由符号的并集
        return symbols | self.free_symbols_in_domain

    @property
    def free_symbols_in_domain(self):
        """
        Free symbols of the domain of ``self``.
        
        Examples
        ========
        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 1).free_symbols_in_domain
        set()
        >>> Poly(x**2 + y).free_symbols_in_domain
        set()
        >>> Poly(x**2 + y, x).free_symbols_in_domain
        {y}
        """
        # 获取多项式的底层表示和相关的符号集合
        domain, symbols = self.rep.dom, set()

        # 根据域的类型不同，获取不同的自由符号
        if domain.is_Composite:
            for gen in domain.symbols:
                symbols |= gen.free_symbols
        elif domain.is_EX:
            for coeff in self.coeffs():
                symbols |= coeff.free_symbols

        # 返回域中的自由符号集合
        return symbols

    @property
    def gen(self):
        """
        Return the principal generator.
        
        Examples
        ========
        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).gen
        x
        """
        # 返回多项式的主要生成元
        return self.gens[0]

    @property
    def domain(self):
        """
        Return the domain of the polynomial.
        
        Examples
        ========
        >>> from sympy import Poly, FF

        >>> Poly(x**2 + 1, x, domain=FF(3)).domain
        FF(3)
        """
        # 返回多项式的域信息
        return self.rep.dom
    def domain(self):
        """
        Get the ground domain of a :py:class:`~.Poly`.

        Returns
        =======
        :py:class:`~.Domain`:
            Ground domain of the :py:class:`~.Poly`.

        Examples
        ========
        Demonstrates usage examples of getting the domain of a polynomial.

        >>> from sympy import Poly, Symbol
        >>> x = Symbol('x')
        >>> p = Poly(x**2 + x)
        >>> p
        Poly(x**2 + x, x, domain='ZZ')
        >>> p.domain
        ZZ
        """
        return self.get_domain()

    @property
    def zero(self):
        """
        Return zero polynomial with ``self``'s properties.

        Returns
        =======
        :py:class:`~.Poly`:
            Zero polynomial with the same properties as ``self``.
        """
        return self.new(self.rep.zero(self.rep.lev, self.rep.dom), *self.gens)

    @property
    def one(self):
        """
        Return one polynomial with ``self``'s properties.

        Returns
        =======
        :py:class:`~.Poly`:
            One polynomial with the same properties as ``self``.
        """
        return self.new(self.rep.one(self.rep.lev, self.rep.dom), *self.gens)

    @property
    def unit(self):
        """
        Return unit polynomial with ``self``'s properties.

        Returns
        =======
        :py:class:`~.Poly`:
            Unit polynomial with the same properties as ``self``.
        """
        return self.new(self.rep.unit(self.rep.lev, self.rep.dom), *self.gens)

    def unify(f, g):
        """
        Make ``f`` and ``g`` belong to the same domain.

        Examples
        ========
        Demonstrates how to unify two polynomials to the same domain.

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f, g = Poly(x/2 + 1), Poly(2*x + 1)

        >>> f
        Poly(1/2*x + 1, x, domain='QQ')
        >>> g
        Poly(2*x + 1, x, domain='ZZ')

        >>> F, G = f.unify(g)

        >>> F
        Poly(1/2*x + 1, x, domain='QQ')
        >>> G
        Poly(2*x + 1, x, domain='QQ')

        Returns
        =======
        Tuple of :py:class:`~.Poly`:
            Polynomials ``f`` and ``g`` after unification to the same domain.
        """
        _, per, F, G = f._unify(g)
        return per(F), per(G)
   `
# 定义一个名为 `_unify` 的函数，用于多项式对象的统一操作，接受两个参数 f 和 g
def _unify(f, g):
    # 将 g 转换为符号表达式（sympify）
    g = sympify(g)

    # 如果 g 不是多项式对象（is_Poly），则尝试将 g 转换为 f.rep.dom 的符号
    if not g.is_Poly:
        try:
            g_coeff = f.rep.dom.from_sympy(g)
        except CoercionFailed:
            # 如果转换失败，则抛出异常，指示无法统一 f 和 g
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))
        else:
            # 如果转换成功，则返回 f.rep.dom、f.per、f.rep 和根据 g_coeff 生成的新多项式对象
            return f.rep.dom, f.per, f.rep, f.rep.ground_new(g_coeff)

    # 如果 f.rep 和 g.rep 都是多项式对象（DMP 类型）
    if isinstance(f.rep, DMP) and isinstance(g.rep, DMP):
        # 统一多项式的生成器（gens）
        gens = _unify_gens(f.gens, g.gens)

        # 将 f.rep.dom 和 g.rep.dom 统一化，gens 是生成器的列表
        dom, lev = f.rep.dom.unify(g.rep.dom, gens), len(gens) - 1

        # 如果 f.gens 与 gens 不同，则重新排列 f.rep 的字典表示
        if f.gens != gens:
            f_monoms, f_coeffs = _dict_reorder(
                f.rep.to_dict(), f.gens, gens)

            # 如果 f.rep.dom 与 dom 不同，则将 f_coeffs 转换为 dom 的元素类型
            if f.rep.dom != dom:
                f_coeffs = [dom.convert(c, f.rep.dom) for c in f_coeffs]

            # 创建新的 DMP 对象 F
            F = DMP.from_dict(dict(list(zip(f_monoms, f_coeffs))), lev, dom)
        else:
            # 否则，直接将 f.rep 转换为 dom 的类型
            F = f.rep.convert(dom)

        # 如果 g.gens 与 gens 不同，则重新排列 g.rep 的字典表示
        if g.gens != gens:
            g_monoms, g_coeffs = _dict_reorder(
                g.rep.to_dict(), g.gens, gens)

            # 如果 g.rep.dom 与 dom 不同，则将 g_coeffs 转换为 dom 的元素类型
            if g.rep.dom != dom:
                g_coeffs = [dom.convert(c, g.rep.dom) for c in g_coeffs]

            # 创建新的 DMP 对象 G
            G = DMP.from_dict(dict(list(zip(g_monoms, g_coeffs))), lev, dom)
        else:
            # 否则，直接将 g.rep 转换为 dom 的类型
            G = g.rep.convert(dom)
    else:
        # 如果 f.rep 和 g.rep 不是 DMP 类型，则抛出无法统一的异常
        raise UnificationFailed("Cannot unify %s with %s" % (f, g))

    # 获取 f 的类，并定义 per 方法
    cls = f.__class__

    def per(rep, dom=dom, gens=gens, remove=None):
        # 如果 remove 不为 None，则从 gens 中移除指定索引的生成器
        if remove is not None:
            gens = gens[:remove] + gens[remove + 1:]

            # 如果 gens 为空，则将 rep 转换为 dom 的符号表达式并返回
            if not gens:
                return dom.to_sympy(rep)

        # 创建并返回 f 类的新实例，传入 rep 和 gens
        return cls.new(rep, *gens)

    # 返回 dom（统一的基础域）、per（用于创建多项式的方法）、F 和 G
    return dom, per, F, G


# 定义一个名为 per 的方法，用于创建多项式对象
def per(f, rep, gens=None, remove=None):
    """
    Create a Poly out of the given representation.

    Examples
    ========

    >>> from sympy import Poly, ZZ
    >>> from sympy.abc import x, y

    >>> from sympy.polys.polyclasses import DMP

    >>> a = Poly(x**2 + 1)

    >>> a.per(DMP([ZZ(1), ZZ(1)], ZZ), gens=[y])
    Poly(y + 1, y, domain='ZZ')

    """
    # 如果未提供 gens，则使用 f 的生成器
    if gens is None:
        gens = f.gens

    # 如果 remove 不为 None，则从 gens 中移除指定索引的生成器
    if remove is not None:
        gens = gens[:remove] + gens[remove + 1:]

        # 如果 gens 为空，则将 rep 转换为 f.rep.dom 的符号表达式并返回
        if not gens:
            return f.rep.dom.to_sympy(rep)

    # 创建并返回 f 类的新实例，传入 rep 和 gens
    return f.__class__.new(rep, *gens)


# 定义一个名为 set_domain 的方法，用于设置多项式的基础域
def set_domain(f, domain):
    """Set the ground domain of ``f``. """
    # 构建基于新域的选项
    opt = options.build_options(f.gens, {'domain': domain})
    # 使用 per 方法将 f.rep 转换为新域的类型并返回
    return f.per(f.rep.convert(opt.domain))


# 定义一个名为 get_domain 的方法，用于获取多项式的基础域
def get_domain(f):
    """Get the ground domain of ``f``. """
    # 返回 f.rep.dom，即多项式的当前基础域
    return f.rep.dom
    def set_modulus(f, modulus):
        """
        设置 ``f`` 的模数。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(5*x**2 + 2*x - 1, x).set_modulus(2)
        Poly(x**2 + 1, x, modulus=2)

        """
        # 预处理模数
        modulus = options.Modulus.preprocess(modulus)
        # 设置域为有限域，并返回新的多项式对象
        return f.set_domain(FF(modulus))

    def get_modulus(f):
        """
        获取 ``f`` 的模数。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, modulus=2).get_modulus()
        2

        """
        # 获取多项式的域
        domain = f.get_domain()

        # 如果域是有限域，则返回其特征作为模数
        if domain.is_FiniteField:
            return Integer(domain.characteristic())
        else:
            # 否则抛出异常，说明多项式不是在 Galois 域上定义的
            raise PolynomialError("not a polynomial over a Galois field")

    def _eval_subs(f, old, new):
        """:func:`subs` 的内部实现。"""
        # 如果旧生成元在多项式的生成元列表中
        if old in f.gens:
            # 如果新值是数字，则求旧生成元对应的新值表达式
            if new.is_number:
                return f.eval(old, new)
            else:
                try:
                    # 否则尝试用新值替换旧生成元
                    return f.replace(old, new)
                except PolynomialError:
                    pass

        # 返回多项式的表达式，并在其中替换旧值为新值
        return f.as_expr().subs(old, new)

    def exclude(f):
        """
        从 ``f`` 中移除不必要的生成元。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import a, b, c, d, x

        >>> Poly(a + x, a, b, c, d, x).exclude()
        Poly(a + x, a, x, domain='ZZ')

        """
        # 获取不必要生成元的下标 J 以及新的表示形式 new
        J, new = f.rep.exclude()
        # 根据 J 列表筛选生成元
        gens = [gen for j, gen in enumerate(f.gens) if j not in J]

        # 返回新的多项式对象，更新生成元列表
        return f.per(new, gens=gens)

    def replace(f, x, y=None, **_ignore):
        # XXX this does not match Basic's signature
        """
        在生成元列表中将 ``x`` 替换为 ``y``。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 1, x).replace(x, y)
        Poly(y**2 + 1, y, domain='ZZ')

        """
        # 如果未提供 y，则在单变量情况下交换 x 和 y
        if y is None:
            if f.is_univariate:
                x, y = f.gen, x
            else:
                raise PolynomialError(
                    "syntax supported only in univariate case")

        # 如果 x 等于 y 或者 x 不在生成元列表中，则直接返回多项式 f
        if x == y or x not in f.gens:
            return f

        # 如果 x 在生成元列表中且 y 不在生成元列表中
        if x in f.gens and y not in f.gens:
            dom = f.get_domain()

            # 如果域不是复合域或者 y 不在符号列表中，则更新生成元列表并返回新的多项式对象
            if not dom.is_Composite or y not in dom.symbols:
                gens = list(f.gens)
                gens[gens.index(x)] = y
                return f.per(f.rep, gens=gens)

        # 抛出异常，表示无法在多项式 f 中用 y 替换 x
        raise PolynomialError("Cannot replace %s with %s in %s" % (x, y, f))

    def match(f, *args, **kwargs):
        """从 Poly 中匹配表达式。参见 Basic.match()"""
        # 返回多项式的表达式，并进行匹配操作
        return f.as_expr().match(*args, **kwargs)
    # 定义函数 `reorder`，用于重新排列生成器的顺序
    def reorder(f, *gens, **args):
        """
        Efficiently apply new order of generators.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x*y**2, x, y).reorder(y, x)
        Poly(y**2*x + x**2, y, x, domain='ZZ')

        """
        # 根据传入的参数创建选项对象
        opt = options.Options((), args)

        # 如果未指定生成器，则按照默认顺序重新排列
        if not gens:
            gens = _sort_gens(f.gens, opt=opt)
        # 如果指定的生成器集合与 f 的生成器集合不同，抛出异常
        elif set(f.gens) != set(gens):
            raise PolynomialError(
                "generators list can differ only up to order of elements")

        # 通过重新排列后的生成器创建表示字典的 rep
        rep = dict(list(zip(*_dict_reorder(f.rep.to_dict(), f.gens, gens))))

        # 返回按照新顺序重排后的多项式
        return f.per(DMP.from_dict(rep, len(gens) - 1, f.rep.dom), gens=gens)

    # 定义函数 `ltrim`，用于从左侧删除虚拟生成器
    def ltrim(f, gen):
        """
        Remove dummy generators from ``f`` that are to the left of
        specified ``gen`` in the generators as ordered. When ``gen``
        is an integer, it refers to the generator located at that
        position within the tuple of generators of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(y**2 + y*z**2, x, y, z).ltrim(y)
        Poly(y**2 + y*z**2, y, z, domain='ZZ')
        >>> Poly(z, x, y, z).ltrim(-1)
        Poly(z, z, domain='ZZ')

        """
        # 将多项式 f 转换为字典表示
        rep = f.as_dict(native=True)
        # 获取要删除的生成器在生成器列表中的位置
        j = f._gen_to_level(gen)

        terms = {}

        # 遍历多项式的项，将不受影响的项加入 terms 字典中
        for monom, coeff in rep.items():
            if any(monom[:j]):
                # 如果要删除的部分中包含使用的生成器，抛出异常
                raise PolynomialError("Cannot left trim %s" % f)
            terms[monom[j:]] = coeff

        # 更新生成器列表
        gens = f.gens[j:]

        # 返回按新生成器列表重新构建的多项式
        return f.new(DMP.from_dict(terms, len(gens) - 1, f.rep.dom), *gens)

    # 定义函数 `has_only_gens`，检查多项式中是否只包含指定的生成器
    def has_only_gens(f, *gens):
        """
        Return ``True`` if ``Poly(f, *gens)`` retains ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x*y + 1, x, y, z).has_only_gens(x, y)
        True
        >>> Poly(x*y + z, x, y, z).has_only_gens(x, y)
        False

        """
        # 存储生成器的索引集合
        indices = set()

        # 遍历传入的生成器列表
        for gen in gens:
            try:
                # 获取生成器在 f 的生成器列表中的索引
                index = f.gens.index(gen)
            except ValueError:
                # 如果生成器不在 f 的生成器列表中，抛出异常
                raise GeneratorsError(
                    "%s doesn't have %s as generator" % (f, gen))
            else:
                indices.add(index)

        # 检查多项式的每一项是否只包含指定索引的生成器
        for monom in f.monoms():
            for i, elt in enumerate(monom):
                if i not in indices and elt:
                    return False

        # 如果所有项都符合条件，则返回 True
        return True
    ```python`
        def to_ring(f):
            """
            Make the ground domain a ring.
    
            Examples
            ========
    
            >>> from sympy import Poly, QQ
            >>> from sympy.abc import x
    
            >>> Poly(x**2 + 1, domain=QQ).to_ring()
            Poly(x**2 + 1, x, domain='ZZ')
    
            """
            # 检查 f.rep 是否具有 to_ring 方法，如果有则调用它
            if hasattr(f.rep, 'to_ring'):
                result = f.rep.to_ring()
            else:  # 如果 f.rep 没有 to_ring 方法，则抛出异常
                raise OperationNotSupported(f, 'to_ring')
    
            # 返回经过处理后的多项式 f 的表示形式
            return f.per(result)
    
        def to_field(f):
            """
            Make the ground domain a field.
    
            Examples
            ========
    
            >>> from sympy import Poly, ZZ
            >>> from sympy.abc import x
    
            >>> Poly(x**2 + 1, x, domain=ZZ).to_field()
            Poly(x**2 + 1, x, domain='QQ')
    
            """
            # 检查 f.rep 是否具有 to_field 方法，如果有则调用它
            if hasattr(f.rep, 'to_field'):
                result = f.rep.to_field()
            else:  # 如果 f.rep 没有 to_field 方法，则抛出异常
                raise OperationNotSupported(f, 'to_field')
    
            # 返回经过处理后的多项式 f 的表示形式
            return f.per(result)
    
        def to_exact(f):
            """
            Make the ground domain exact.
    
            Examples
            ========
    
            >>> from sympy import Poly, RR
            >>> from sympy.abc import x
    
            >>> Poly(x**2 + 1.0, x, domain=RR).to_exact()
            Poly(x**2 + 1, x, domain='QQ')
    
            """
            # 检查 f.rep 是否具有 to_exact 方法，如果有则调用它
            if hasattr(f.rep, 'to_exact'):
                result = f.rep.to_exact()
            else:  # 如果 f.rep 没有 to_exact 方法，则抛出异常
                raise OperationNotSupported(f, 'to_exact')
    
            # 返回经过处理后的多项式 f 的表示形式
            return f.per(result)
    
        def retract(f, field=None):
            """
            Recalculate the ground domain of a polynomial.
    
            Examples
            ========
    
            >>> from sympy import Poly
            >>> from sympy.abc import x
    
            >>> f = Poly(x**2 + 1, x, domain='QQ[y]')
            >>> f
            Poly(x**2 + 1, x, domain='QQ[y]')
    
            >>> f.retract()
            Poly(x**2 + 1, x, domain='ZZ')
            >>> f.retract(field=True)
            Poly(x**2 + 1, x, domain='QQ')
    
            """
            # 构造域和表示
            dom, rep = construct_domain(f.as_dict(zero=True),
                field=field, composite=f.domain.is_Composite or None)
            # 使用新的域和表示构造多项式，并返回
            return f.from_dict(rep, f.gens, domain=dom)
    
        def slice(f, x, m, n=None):
            """Take a continuous subsequence of terms of ``f``. """
            # 如果 n 为 None，则将 j 设置为 0，m 设置为 x，n 设置为 m
            if n is None:
                j, m, n = 0, x, m
            else:
                j = f._gen_to_level(x)
    
            # 将 m 和 n 转换为整数
            m, n = int(m), int(n)
    
            # 检查 f.rep 是否具有 slice 方法，如果有则调用它
            if hasattr(f.rep, 'slice'):
                result = f.rep.slice(m, n, j)
            else:  # 如果 f.rep 没有 slice 方法，则抛出异常
                raise OperationNotSupported(f, 'slice')
    
            # 返回经过切片处理后的多项式 f 的表示形式
            return f.per(result)
    
        def coeffs(f, order=None):
            """
            Returns all non-zero coefficients from ``f`` in lex order.
    
            Examples
            ========
    
            >>> from sympy import Poly
            >>> from sympy.abc import x
    
            >>> Poly(x**3 + 2*x + 3, x).coeffs()
            [1, 2, 3]
    
            See Also
            ========
            all_coeffs
            coeff_monomial
            nth
    
            """
            # 返回 f.rep 的所有非零系数，按照字典序排序
            return [f.rep.dom.to_sympy(c) for c in f.rep.coeffs(order=order)]
    # 返回多项式 f 中按词典序排列的所有非零单项式
    def monoms(f, order=None):
        return f.rep.monoms(order=order)
    
    # 返回多项式 f 中按词典序排列的所有非零项
    def terms(f, order=None):
        return [(m, f.rep.dom.to_sympy(c)) for m, c in f.rep.terms(order=order)]
    
    # 返回一元多项式 f 的所有系数
    def all_coeffs(f):
        return [f.rep.dom.to_sympy(c) for c in f.rep.all_coeffs()]
    
    # 返回一元多项式 f 的所有单项式
    def all_monoms(f):
        return f.rep.all_monoms()
    
    # 返回一元多项式 f 的所有项
    def all_terms(f):
        return [(m, f.rep.dom.to_sympy(c)) for m, c in f.rep.all_terms()]
    def termwise(f, func, *gens, **args):
        """
        对多项式 ``f`` 的每个项应用函数 ``func``。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> def func(k, coeff):
        ...     k = k[0]
        ...     return coeff//10**(2-k)

        >>> Poly(x**2 + 20*x + 400).termwise(func)
        Poly(x**2 + 2*x + 4, x, domain='ZZ')

        """
        # 初始化空字典来存储处理后的项
        terms = {}

        # 遍历多项式 ``f`` 的所有项和系数
        for monom, coeff in f.terms():
            # 调用给定的函数 ``func`` 处理当前项和系数，得到处理结果
            result = func(monom, coeff)

            # 如果处理结果是一个元组，则更新项和系数
            if isinstance(result, tuple):
                monom, coeff = result
            else:
                coeff = result

            # 如果系数不为零
            if coeff:
                # 如果当前项不在字典中，则添加；否则抛出多项式错误
                if monom not in terms:
                    terms[monom] = coeff
                else:
                    raise PolynomialError(
                        "%s monomial was generated twice" % monom)

        # 从字典重构多项式对象并返回
        return f.from_dict(terms, *(gens or f.gens), **args)

    def length(f):
        """
        返回多项式 ``f`` 中非零项的数量。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 2*x - 1).length()
        3

        """
        # 返回多项式对象转换为字典后的长度
        return len(f.as_dict())

    def as_dict(f, native=False, zero=False):
        """
        转换为 ``dict`` 表示形式。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x*y**2 - y, x, y).as_dict()
        {(0, 1): -1, (1, 2): 2, (2, 0): 1}

        """
        # 如果指定使用本地表示，则转换为字典
        if native:
            return f.rep.to_dict(zero=zero)
        else:
            return f.rep.to_sympy_dict(zero=zero)

    def as_list(f, native=False):
        """转换为 ``list`` 表示形式。"""
        # 如果指定使用本地表示，则转换为列表
        if native:
            return f.rep.to_list()
        else:
            return f.rep.to_sympy_list()

    def as_expr(f, *gens):
        """
        将多项式实例转换为表达式实例。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2 + 2*x*y**2 - y, x, y)

        >>> f.as_expr()
        x**2 + 2*x*y**2 - y
        >>> f.as_expr({x: 5})
        10*y**2 - y + 25
        >>> f.as_expr(5, 6)
        379

        """
        # 如果未指定生成器，则直接返回多项式的表达式
        if not gens:
            return f.expr

        # 如果生成器是一个字典，则进行映射转换
        if len(gens) == 1 and isinstance(gens[0], dict):
            mapping = gens[0]
            gens = list(f.gens)

            for gen, value in mapping.items():
                try:
                    index = gens.index(gen)
                except ValueError:
                    raise GeneratorsError(
                        "%s doesn't have %s as generator" % (f, gen))
                else:
                    gens[index] = value

        # 从字典转换为基本表达式并返回
        return basic_from_dict(f.rep.to_sympy_dict(), *gens)
    def as_poly(self, *gens, **args):
        """
        Converts ``self`` to a polynomial or returns ``None``.

        >>> from sympy import sin
        >>> from sympy.abc import x, y

        >>> print((x**2 + x*y).as_poly())
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + x*y).as_poly(x, y))
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + sin(y)).as_poly(x, y))
        None

        """
        try:
            # 尝试将 self 转换为多项式对象 Poly
            poly = Poly(self, *gens, **args)

            # 如果转换后的对象不是 Poly 类型，则返回 None
            if not poly.is_Poly:
                return None
            else:
                return poly
        except PolynomialError:
            # 捕获多项式错误，返回 None
            return None

    def lift(f):
        """
        Convert algebraic coefficients to rationals.

        Examples
        ========

        >>> from sympy import Poly, I
        >>> from sympy.abc import x

        >>> Poly(x**2 + I*x + 1, x, extension=I).lift()
        Poly(x**4 + 3*x**2 + 1, x, domain='QQ')

        """
        if hasattr(f.rep, 'lift'):
            # 如果 f.rep 具有 lift 方法，则调用 lift 方法
            result = f.rep.lift()
        else:  # pragma: no cover
            # 如果没有 lift 方法，抛出不支持操作异常
            raise OperationNotSupported(f, 'lift')

        # 返回 f 的结果对象
        return f.per(result)

    def deflate(f):
        """
        Reduce degree of ``f`` by mapping ``x_i**m`` to ``y_i``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**6*y**2 + x**3 + 1, x, y).deflate()
        ((3, 2), Poly(x**2*y + x + 1, x, y, domain='ZZ'))

        """
        if hasattr(f.rep, 'deflate'):
            # 如果 f.rep 具有 deflate 方法，则调用 deflate 方法
            J, result = f.rep.deflate()
        else:  # pragma: no cover
            # 如果没有 deflate 方法，抛出不支持操作异常
            raise OperationNotSupported(f, 'deflate')

        # 返回 deflate 操作的结果元组 (J, f 的结果对象)
        return J, f.per(result)

    def inject(f, front=False):
        """
        Inject ground domain generators into ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x)

        >>> f.inject()
        Poly(x**2*y + x*y**3 + x*y + 1, x, y, domain='ZZ')
        >>> f.inject(front=True)
        Poly(y**3*x + y*x**2 + y*x + 1, y, x, domain='ZZ')

        """
        dom = f.rep.dom

        # 如果 dom 是数值类型，则直接返回 f
        if dom.is_Numerical:
            return f
        # 如果 dom 不是多项式类型，则抛出域错误异常
        elif not dom.is_Poly:
            raise DomainError("Cannot inject generators over %s" % dom)

        # 如果 f.rep 具有 inject 方法，则调用 inject 方法
        if hasattr(f.rep, 'inject'):
            result = f.rep.inject(front=front)
        else:  # pragma: no cover
            # 如果没有 inject 方法，抛出不支持操作异常
            raise OperationNotSupported(f, 'inject')

        # 根据 front 参数选择合适的生成器顺序，并返回结果
        if front:
            gens = dom.symbols + f.gens
        else:
            gens = f.gens + dom.symbols

        return f.new(result, *gens)
    def eject(f, *gens):
        """
        Eject selected generators into the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x, y)

        >>> f.eject(x)
        Poly(x*y**3 + (x**2 + x)*y + 1, y, domain='ZZ[x]')
        >>> f.eject(y)
        Poly(y*x**2 + (y**3 + y)*x + 1, x, domain='ZZ[y]')

        """
        # 获取多项式 ``f`` 的表示域
        dom = f.rep.dom

        # 如果表示域不是数值类型，则抛出域错误
        if not dom.is_Numerical:
            raise DomainError("Cannot eject generators over %s" % dom)

        # 计算生成器的数量
        k = len(gens)

        # 根据生成器的情况选择是在前面还是后面剔除
        if f.gens[:k] == gens:
            _gens, front = f.gens[k:], True
        elif f.gens[-k:] == gens:
            _gens, front = f.gens[:-k], False
        else:
            raise NotImplementedError(
                "can only eject front or back generators")

        # 将生成器注入到表示域中
        dom = dom.inject(*gens)

        # 如果 ``f.rep`` 具有 'eject' 方法，则执行剔除操作
        if hasattr(f.rep, 'eject'):
            result = f.rep.eject(dom, front=front)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'eject')

        # 返回新的多项式对象
        return f.new(result, *_gens)

    def terms_gcd(f):
        """
        Remove GCD of terms from the polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**6*y**2 + x**3*y, x, y).terms_gcd()
        ((3, 1), Poly(x**3*y + 1, x, y, domain='ZZ'))

        """
        # 如果 ``f.rep`` 具有 'terms_gcd' 方法，则执行项的最大公约数操作
        if hasattr(f.rep, 'terms_gcd'):
            J, result = f.rep.terms_gcd()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'terms_gcd')

        # 返回项的最大公约数和处理结果后的多项式
        return J, f.per(result)

    def add_ground(f, coeff):
        """
        Add an element of the ground domain to ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).add_ground(2)
        Poly(x + 3, x, domain='ZZ')

        """
        # 如果 ``f.rep`` 具有 'add_ground' 方法，则执行向多项式添加地域元素的操作
        if hasattr(f.rep, 'add_ground'):
            result = f.rep.add_ground(coeff)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'add_ground')

        # 返回处理结果后的多项式
        return f.per(result)

    def sub_ground(f, coeff):
        """
        Subtract an element of the ground domain from ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).sub_ground(2)
        Poly(x - 1, x, domain='ZZ')

        """
        # 如果 ``f.rep`` 具有 'sub_ground' 方法，则执行从多项式减去地域元素的操作
        if hasattr(f.rep, 'sub_ground'):
            result = f.rep.sub_ground(coeff)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'sub_ground')

        # 返回处理结果后的多项式
        return f.per(result)
    def mul_ground(f, coeff):
        """
        Multiply ``f`` by an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).mul_ground(2)
        Poly(2*x + 2, x, domain='ZZ')

        """
        # 检查 f.rep 是否具有 mul_ground 方法，如果有，则调用该方法
        if hasattr(f.rep, 'mul_ground'):
            result = f.rep.mul_ground(coeff)
        else:  # 如果没有该方法，则抛出 OperationNotSupported 异常
            raise OperationNotSupported(f, 'mul_ground')

        # 调用 f.per 方法，将结果 result 封装并返回
        return f.per(result)

    def quo_ground(f, coeff):
        """
        Quotient of ``f`` by an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x + 4).quo_ground(2)
        Poly(x + 2, x, domain='ZZ')

        >>> Poly(2*x + 3).quo_ground(2)
        Poly(x + 1, x, domain='ZZ')

        """
        # 检查 f.rep 是否具有 quo_ground 方法，如果有，则调用该方法
        if hasattr(f.rep, 'quo_ground'):
            result = f.rep.quo_ground(coeff)
        else:  # 如果没有该方法，则抛出 OperationNotSupported 异常
            raise OperationNotSupported(f, 'quo_ground')

        # 调用 f.per 方法，将结果 result 封装并返回
        return f.per(result)

    def exquo_ground(f, coeff):
        """
        Exact quotient of ``f`` by an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x + 4).exquo_ground(2)
        Poly(x + 2, x, domain='ZZ')

        >>> Poly(2*x + 3).exquo_ground(2)
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2 does not divide 3 in ZZ

        """
        # 检查 f.rep 是否具有 exquo_ground 方法，如果有，则调用该方法
        if hasattr(f.rep, 'exquo_ground'):
            result = f.rep.exquo_ground(coeff)
        else:  # 如果没有该方法，则抛出 OperationNotSupported 异常
            raise OperationNotSupported(f, 'exquo_ground')

        # 调用 f.per 方法，将结果 result 封装并返回
        return f.per(result)

    def abs(f):
        """
        Make all coefficients in ``f`` positive.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).abs()
        Poly(x**2 + 1, x, domain='ZZ')

        """
        # 检查 f.rep 是否具有 abs 方法，如果有，则调用该方法
        if hasattr(f.rep, 'abs'):
            result = f.rep.abs()
        else:  # 如果没有该方法，则抛出 OperationNotSupported 异常
            raise OperationNotSupported(f, 'abs')

        # 调用 f.per 方法，将结果 result 封装并返回
        return f.per(result)

    def neg(f):
        """
        Negate all coefficients in ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).neg()
        Poly(-x**2 + 1, x, domain='ZZ')

        >>> -Poly(x**2 - 1, x)
        Poly(-x**2 + 1, x, domain='ZZ')

        """
        # 检查 f.rep 是否具有 neg 方法，如果有，则调用该方法
        if hasattr(f.rep, 'neg'):
            result = f.rep.neg()
        else:  # 如果没有该方法，则抛出 OperationNotSupported 异常
            raise OperationNotSupported(f, 'neg')

        # 调用 f.per 方法，将结果 result 封装并返回
        return f.per(result)
    def add(f, g):
        """
        Add two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).add(Poly(x - 2, x))
        Poly(x**2 + x - 1, x, domain='ZZ')

        >>> Poly(x**2 + 1, x) + Poly(x - 2, x)
        Poly(x**2 + x - 1, x, domain='ZZ')

        """
        # 将 g 转换为 sympy 的表达式
        g = sympify(g)

        # 如果 g 不是多项式，则调用 f 的 add_ground 方法
        if not g.is_Poly:
            return f.add_ground(g)

        # 将 f 和 g 统一到相同的表示形式
        _, per, F, G = f._unify(g)

        # 如果 f.rep 具有 'add' 属性，则执行多项式相加操作
        if hasattr(f.rep, 'add'):
            result = F.add(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'add')

        # 返回结果并恢复原始的表达形式
        return per(result)

    def sub(f, g):
        """
        Subtract two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).sub(Poly(x - 2, x))
        Poly(x**2 - x + 3, x, domain='ZZ')

        >>> Poly(x**2 + 1, x) - Poly(x - 2, x)
        Poly(x**2 - x + 3, x, domain='ZZ')

        """
        # 将 g 转换为 sympy 的表达式
        g = sympify(g)

        # 如果 g 不是多项式，则调用 f 的 sub_ground 方法
        if not g.is_Poly:
            return f.sub_ground(g)

        # 将 f 和 g 统一到相同的表示形式
        _, per, F, G = f._unify(g)

        # 如果 f.rep 具有 'sub' 属性，则执行多项式相减操作
        if hasattr(f.rep, 'sub'):
            result = F.sub(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'sub')

        # 返回结果并恢复原始的表达形式
        return per(result)

    def mul(f, g):
        """
        Multiply two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).mul(Poly(x - 2, x))
        Poly(x**3 - 2*x**2 + x - 2, x, domain='ZZ')

        >>> Poly(x**2 + 1, x)*Poly(x - 2, x)
        Poly(x**3 - 2*x**2 + x - 2, x, domain='ZZ')

        """
        # 将 g 转换为 sympy 的表达式
        g = sympify(g)

        # 如果 g 不是多项式，则调用 f 的 mul_ground 方法
        if not g.is_Poly:
            return f.mul_ground(g)

        # 将 f 和 g 统一到相同的表示形式
        _, per, F, G = f._unify(g)

        # 如果 f.rep 具有 'mul' 属性，则执行多项式相乘操作
        if hasattr(f.rep, 'mul'):
            result = F.mul(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'mul')

        # 返回结果并恢复原始的表达形式
        return per(result)

    def sqr(f):
        """
        Square a polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x - 2, x).sqr()
        Poly(x**2 - 4*x + 4, x, domain='ZZ')

        >>> Poly(x - 2, x)**2
        Poly(x**2 - 4*x + 4, x, domain='ZZ')

        """
        # 如果 f.rep 具有 'sqr' 属性，则执行多项式平方操作
        if hasattr(f.rep, 'sqr'):
            result = f.rep.sqr()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'sqr')

        # 返回结果并恢复原始的表达形式
        return f.per(result)
    # 定义一个函数 `pow`，用于计算多项式 `f` 的非负整数次幂 `n`
    def pow(f, n):
        # 将 `n` 转换为整数类型
        n = int(n)

        # 如果多项式 `f` 的表示有 `pow` 方法，则调用其 `pow` 方法计算结果
        if hasattr(f.rep, 'pow'):
            result = f.rep.pow(n)
        else:  # 如果不支持 `pow` 方法，则抛出不支持操作的异常
            raise OperationNotSupported(f, 'pow')

        # 调用 `f` 的 `per` 方法，将计算得到的结果 `result` 转换为相应的多项式返回
        return f.per(result)

    # 定义一个函数 `pdiv`，用于进行多项式 `f` 除以 `g` 的伪除法
    def pdiv(f, g):
        # 使用 `_unify` 方法获取多项式 `f` 和 `g` 的一致表示
        _, per, F, G = f._unify(g)

        # 如果多项式 `f` 的表示有 `pdiv` 方法，则调用其 `pdiv` 方法进行伪除法计算
        if hasattr(f.rep, 'pdiv'):
            q, r = F.pdiv(G)
        else:  # 如果不支持 `pdiv` 方法，则抛出不支持操作的异常
            raise OperationNotSupported(f, 'pdiv')

        # 使用 `per` 方法将计算得到的商 `q` 和余数 `r` 转换为相应的多项式返回
        return per(q), per(r)

    # 定义一个函数 `prem`，用于计算多项式 `f` 除以 `g` 的伪余数
    def prem(f, g):
        # 使用 `_unify` 方法获取多项式 `f` 和 `g` 的一致表示
        _, per, F, G = f._unify(g)

        # 如果多项式 `f` 的表示有 `prem` 方法，则调用其 `prem` 方法进行伪余数计算
        if hasattr(f.rep, 'prem'):
            result = F.prem(G)
        else:  # 如果不支持 `prem` 方法，则抛出不支持操作的异常
            raise OperationNotSupported(f, 'prem')

        # 使用 `per` 方法将计算得到的余数 `result` 转换为相应的多项式返回
        return per(result)
    def pquo(f, g):
        """
        Polynomial pseudo-quotient of ``f`` by ``g``.

        See the Caveat note in the function prem(f, g).

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).pquo(Poly(2*x - 4, x))
        Poly(2*x + 4, x, domain='ZZ')

        >>> Poly(x**2 - 1, x).pquo(Poly(2*x - 2, x))
        Poly(2*x + 2, x, domain='ZZ')

        """
        # 获取 f 和 g 的统一表示，以及相应的置换函数
        _, per, F, G = f._unify(g)

        # 如果 f 的表示有 pquo 方法，则调用 F.pquo(G) 计算伪商
        if hasattr(f.rep, 'pquo'):
            result = F.pquo(G)
        else:  # 如果没有 pquo 方法，引发 OperationNotSupported 异常
            raise OperationNotSupported(f, 'pquo')

        # 返回应用置换函数后的结果
        return per(result)

    def pexquo(f, g):
        """
        Polynomial exact pseudo-quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).pexquo(Poly(2*x - 2, x))
        Poly(2*x + 2, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).pexquo(Poly(2*x - 4, x))
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1

        """
        # 获取 f 和 g 的统一表示，以及相应的置换函数
        _, per, F, G = f._unify(g)

        # 如果 f 的表示有 pexquo 方法，则尝试计算精确伪商 F.pexquo(G)
        if hasattr(f.rep, 'pexquo'):
            try:
                result = F.pexquo(G)
            except ExactQuotientFailed as exc:
                # 如果计算失败，引发新的 ExactQuotientFailed 异常，指明具体的表达式
                raise exc.new(f.as_expr(), g.as_expr())
        else:  # 如果没有 pexquo 方法，引发 OperationNotSupported 异常
            raise OperationNotSupported(f, 'pexquo')

        # 返回应用置换函数后的结果
        return per(result)

    def div(f, g, auto=True):
        """
        Polynomial division with remainder of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).div(Poly(2*x - 4, x))
        (Poly(1/2*x + 1, x, domain='QQ'), Poly(5, x, domain='QQ'))

        >>> Poly(x**2 + 1, x).div(Poly(2*x - 4, x), auto=False)
        (Poly(0, x, domain='ZZ'), Poly(x**2 + 1, x, domain='ZZ'))

        """
        # 获取 f 和 g 的统一表示，以及相应的置换函数
        dom, per, F, G = f._unify(g)
        retract = False

        # 如果 auto=True 且 dom 是环而不是域，则将 F 和 G 转换为域，设置 retract 为 True
        if auto and dom.is_Ring and not dom.is_Field:
            F, G = F.to_field(), G.to_field()
            retract = True

        # 如果 f 的表示有 div 方法，则调用 F.div(G) 计算除法
        if hasattr(f.rep, 'div'):
            q, r = F.div(G)
        else:  # 如果没有 div 方法，引发 OperationNotSupported 异常
            raise OperationNotSupported(f, 'div')

        # 如果之前进行了转换，则尝试将商和余数重新转换为环表示
        if retract:
            try:
                Q, R = q.to_ring(), r.to_ring()
            except CoercionFailed:
                pass
            else:
                q, r = Q, R

        # 返回应用置换函数后的商和余数
        return per(q), per(r)
    def rem(f, g, auto=True):
        """
        Computes the polynomial remainder of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).rem(Poly(2*x - 4, x))
        Poly(5, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).rem(Poly(2*x - 4, x), auto=False)
        Poly(x**2 + 1, x, domain='ZZ')

        """
        # 将 f 和 g 统一到相同的环或域上，并获取相应的操作函数
        dom, per, F, G = f._unify(g)
        retract = False

        # 如果 auto=True 并且 dom 是环但不是域，则将 F 和 G 转换为域，并标记需要回退
        if auto and dom.is_Ring and not dom.is_Field:
            F, G = F.to_field(), G.to_field()
            retract = True

        # 如果 f 的表示具有 'rem' 方法，则计算 F 除以 G 的余数
        if hasattr(f.rep, 'rem'):
            r = F.rem(G)
        else:  # pragma: no cover
            # 如果没有 'rem' 方法，抛出不支持的操作异常
            raise OperationNotSupported(f, 'rem')

        # 如果进行了域转换，尝试将余数 r 转换回环
        if retract:
            try:
                r = r.to_ring()
            except CoercionFailed:
                pass

        # 返回处理后的结果，应用可能的变换函数 per
        return per(r)

    def quo(f, g, auto=True):
        """
        Computes polynomial quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).quo(Poly(2*x - 4, x))
        Poly(1/2*x + 1, x, domain='QQ')

        >>> Poly(x**2 - 1, x).quo(Poly(x - 1, x))
        Poly(x + 1, x, domain='ZZ')

        """
        # 将 f 和 g 统一到相同的环或域上，并获取相应的操作函数
        dom, per, F, G = f._unify(g)
        retract = False

        # 如果 auto=True 并且 dom 是环但不是域，则将 F 和 G 转换为域，并标记需要回退
        if auto and dom.is_Ring and not dom.is_Field:
            F, G = F.to_field(), G.to_field()
            retract = True

        # 如果 f 的表示具有 'quo' 方法，则计算 F 除以 G 的商
        if hasattr(f.rep, 'quo'):
            q = F.quo(G)
        else:  # pragma: no cover
            # 如果没有 'quo' 方法，抛出不支持的操作异常
            raise OperationNotSupported(f, 'quo')

        # 如果进行了域转换，尝试将商 q 转换回环
        if retract:
            try:
                q = q.to_ring()
            except CoercionFailed:
                pass

        # 返回处理后的结果，应用可能的变换函数 per
        return per(q)

    def exquo(f, g, auto=True):
        """
        Computes polynomial exact quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).exquo(Poly(x - 1, x))
        Poly(x + 1, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).exquo(Poly(2*x - 4, x))
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1

        """
        # 将 f 和 g 统一到相同的环或域上，并获取相应的操作函数
        dom, per, F, G = f._unify(g)
        retract = False

        # 如果 auto=True 并且 dom 是环但不是域，则将 F 和 G 转换为域，并标记需要回退
        if auto and dom.is_Ring and not dom.is_Field:
            F, G = F.to_field(), G.to_field()
            retract = True

        # 如果 f 的表示具有 'exquo' 方法，则计算 F 除以 G 的精确商
        if hasattr(f.rep, 'exquo'):
            try:
                q = F.exquo(G)
            except ExactQuotientFailed as exc:
                # 如果出现精确商计算失败的异常，抛出新的异常
                raise exc.new(f.as_expr(), g.as_expr())
        else:  # pragma: no cover
            # 如果没有 'exquo' 方法，抛出不支持的操作异常
            raise OperationNotSupported(f, 'exquo')

        # 如果进行了域转换，尝试将商 q 转换回环
        if retract:
            try:
                q = q.to_ring()
            except CoercionFailed:
                pass

        # 返回处理后的结果，应用可能的变换函数 per
        return per(q)
    def _gen_to_level(f, gen):
        """Returns level associated with the given generator. """
        # 检查 gen 是否为整数
        if isinstance(gen, int):
            # 获取生成器列表的长度
            length = len(f.gens)

            # 判断 gen 是否在有效范围内
            if -length <= gen < length:
                if gen < 0:
                    # 返回负索引对应的正索引
                    return length + gen
                else:
                    # 返回非负索引
                    return gen
            else:
                # 抛出异常，gen 超出范围
                raise PolynomialError("-%s <= gen < %s expected, got %s" %
                                      (length, length, gen))
        else:
            try:
                # 尝试在生成器列表中查找 gen，并返回其索引
                return f.gens.index(sympify(gen))
            except ValueError:
                # 抛出异常，gen 不是有效的生成器
                raise PolynomialError(
                    "a valid generator expected, got %s" % gen)

    def degree(f, gen=0):
        """
        Returns degree of ``f`` in ``x_j``.

        The degree of 0 is negative infinity.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).degree()
        2
        >>> Poly(x**2 + y*x + y, x, y).degree(y)
        1
        >>> Poly(0, x).degree()
        -oo

        """
        # 获取与给定生成器关联的级数 j
        j = f._gen_to_level(gen)

        # 检查是否支持 rep 对象的 degree 方法
        if hasattr(f.rep, 'degree'):
            # 调用 rep 对象的 degree 方法获取 f 在 j 上的度数
            d = f.rep.degree(j)
            # 如果度数为负数，返回负无穷
            if d < 0:
                d = S.NegativeInfinity
            return d
        else:  # pragma: no cover
            # 如果不支持 degree 方法，抛出操作不支持异常
            raise OperationNotSupported(f, 'degree')

    def degree_list(f):
        """
        Returns a list of degrees of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).degree_list()
        (2, 1)

        """
        # 检查是否支持 rep 对象的 degree_list 方法
        if hasattr(f.rep, 'degree_list'):
            # 调用 rep 对象的 degree_list 方法获取 f 的度数列表
            return f.rep.degree_list()
        else:  # pragma: no cover
            # 如果不支持 degree_list 方法，抛出操作不支持异常
            raise OperationNotSupported(f, 'degree_list')

    def total_degree(f):
        """
        Returns the total degree of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).total_degree()
        2
        >>> Poly(x + y**5, x, y).total_degree()
        5

        """
        # 检查是否支持 rep 对象的 total_degree 方法
        if hasattr(f.rep, 'total_degree'):
            # 调用 rep 对象的 total_degree 方法获取 f 的总度数
            return f.rep.total_degree()
        else:  # pragma: no cover
            # 如果不支持 total_degree 方法，抛出操作不支持异常
            raise OperationNotSupported(f, 'total_degree')
    def homogenize(f, s):
        """
        Returns the homogeneous polynomial of ``f``.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. If you only
        want to check if a polynomial is homogeneous, then use
        :func:`Poly.is_homogeneous`. If you want not only to check if a
        polynomial is homogeneous but also compute its homogeneous order,
        then use :func:`Poly.homogeneous_order`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> f = Poly(x**5 + 2*x**2*y**2 + 9*x*y**3)
        >>> f.homogenize(z)
        Poly(x**5 + 2*x**2*y**2*z + 9*x*y**3*z, x, y, z, domain='ZZ')

        """
        # 检查参数 s 是否为符号类型，如果不是则抛出类型错误
        if not isinstance(s, Symbol):
            raise TypeError("``Symbol`` expected, got %s" % type(s))
        # 如果 s 在 f 的生成器列表中，则记录其索引 i，并使用现有生成器列表
        if s in f.gens:
            i = f.gens.index(s)
            gens = f.gens
        else:
            # 否则，将 s 添加到生成器列表中，记录其索引 i
            i = len(f.gens)
            gens = f.gens + (s,)
        # 如果 f.rep 对象具有 'homogenize' 方法，则调用该方法返回处理后的多项式
        if hasattr(f.rep, 'homogenize'):
            return f.per(f.rep.homogenize(i), gens=gens)
        # 如果没有 'homogenize' 方法，抛出不支持操作的异常
        raise OperationNotSupported(f, 'homogeneous_order')

    def homogeneous_order(f):
        """
        Returns the homogeneous order of ``f``.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. This degree is
        the homogeneous order of ``f``. If you only want to check if a
        polynomial is homogeneous, then use :func:`Poly.is_homogeneous`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**5 + 2*x**3*y**2 + 9*x*y**4)
        >>> f.homogeneous_order()
        5

        """
        # 如果 f.rep 对象具有 'homogeneous_order' 方法，则返回其结果
        if hasattr(f.rep, 'homogeneous_order'):
            return f.rep.homogeneous_order()
        else:  # pragma: no cover
            # 如果没有 'homogeneous_order' 方法，抛出不支持操作的异常
            raise OperationNotSupported(f, 'homogeneous_order')

    def LC(f, order=None):
        """
        Returns the leading coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(4*x**3 + 2*x**2 + 3*x, x).LC()
        4

        """
        # 如果指定了 order 参数，则返回对应阶数的系数
        if order is not None:
            return f.coeffs(order)[0]

        # 如果 f.rep 对象具有 'LC' 方法，则返回其结果
        if hasattr(f.rep, 'LC'):
            result = f.rep.LC()
        else:  # pragma: no cover
            # 如果没有 'LC' 方法，抛出不支持操作的异常
            raise OperationNotSupported(f, 'LC')

        # 将 f.rep.dom.to_sympy 方法应用于结果，返回 SymPy 对象
        return f.rep.dom.to_sympy(result)

    def TC(f):
        """
        Returns the trailing coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x**2 + 3*x, x).TC()
        0

        """
        # 如果 f.rep 对象具有 'TC' 方法，则返回其结果
        if hasattr(f.rep, 'TC'):
            result = f.rep.TC()
        else:  # pragma: no cover
            # 如果没有 'TC' 方法，抛出不支持操作的异常
            raise OperationNotSupported(f, 'TC')

        # 将 f.rep.dom.to_sympy 方法应用于结果，返回 SymPy 对象
        return f.rep.dom.to_sympy(result)
    def EC(f, order=None):
        """
        Returns the last non-zero coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x**2 + 3*x, x).EC()
        3

        """
        # 检查是否 f.rep 具有 'coeffs' 属性
        if hasattr(f.rep, 'coeffs'):
            # 调用 f.coeffs(order) 获取系数列表，并返回最后一个非零系数
            return f.coeffs(order)[-1]
        else:  # pragma: no cover
            # 如果 f.rep 不支持 'coeffs'，抛出不支持操作的异常
            raise OperationNotSupported(f, 'EC')

    def coeff_monomial(f, monom):
        """
        Returns the coefficient of ``monom`` in ``f`` if there, else None.

        Examples
        ========

        >>> from sympy import Poly, exp
        >>> from sympy.abc import x, y

        >>> p = Poly(24*x*y*exp(8) + 23*x, x, y)

        >>> p.coeff_monomial(x)
        23
        >>> p.coeff_monomial(y)
        0
        >>> p.coeff_monomial(x*y)
        24*exp(8)

        Note that ``Expr.coeff()`` behaves differently, collecting terms
        if possible; the Poly must be converted to an Expr to use that
        method, however:

        >>> p.as_expr().coeff(x)
        24*y*exp(8) + 23
        >>> p.as_expr().coeff(y)
        24*x*exp(8)
        >>> p.as_expr().coeff(x*y)
        24*exp(8)

        See Also
        ========
        nth: more efficient query using exponents of the monomial's generators

        """
        # 返回 monom 在 f 中的系数，如果不存在则返回 None
        return f.nth(*Monomial(monom, f.gens).exponents)

    def nth(f, *N):
        """
        Returns the ``n``-th coefficient of ``f`` where ``N`` are the
        exponents of the generators in the term of interest.

        Examples
        ========

        >>> from sympy import Poly, sqrt
        >>> from sympy.abc import x, y

        >>> Poly(x**3 + 2*x**2 + 3*x, x).nth(2)
        2
        >>> Poly(x**3 + 2*x*y**2 + y**2, x, y).nth(1, 2)
        2
        >>> Poly(4*sqrt(x)*y)
        Poly(4*y*(sqrt(x)), y, sqrt(x), domain='ZZ')
        >>> _.nth(1, 1)
        4

        See Also
        ========
        coeff_monomial

        """
        # 检查 f.rep 是否具有 'nth' 属性
        if hasattr(f.rep, 'nth'):
            # 如果 N 的长度不等于生成器的个数，抛出 ValueError
            if len(N) != len(f.gens):
                raise ValueError('exponent of each generator must be specified')
            # 调用 f.rep.nth(*list(map(int, N))) 获取指定位置的系数
            result = f.rep.nth(*list(map(int, N)))
        else:  # pragma: no cover
            # 如果 f.rep 不支持 'nth'，抛出不支持操作的异常
            raise OperationNotSupported(f, 'nth')

        # 将结果转换为 SymPy 表达式返回
        return f.rep.dom.to_sympy(result)

    def coeff(f, x, n=1, right=False):
        # 'coeff_monomial' 和 'Expr.coeff' 的语义不同；
        # 如果使用 Poly，应该了解它们之间的差异，并选择最适合查询的方法。
        # 或者可以在这里编写一个纯粹的多项式方法，
        # 但此时 'right' 关键字将被忽略，因为 Poly 不适用于非交换环。
        raise NotImplementedError(
            'Either convert to Expr with `as_expr` method '
            'to use Expr\'s coeff method or else use the '
            '`coeff_monomial` method of Polys.')
    def LM(f, order=None):
        """
        Returns the leading monomial of ``f``.

        The Leading monomial signifies the monomial having
        the highest power of the principal generator in the
        expression f.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).LM()
        x**2*y**0

        """
        # 获取表达式 f 的首项单项式
        return Monomial(f.monoms(order)[0], f.gens)

    def EM(f, order=None):
        """
        Returns the last non-zero monomial of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).EM()
        x**0*y**1

        """
        # 获取表达式 f 的最后一个非零单项式
        return Monomial(f.monoms(order)[-1], f.gens)

    def LT(f, order=None):
        """
        Returns the leading term of ``f``.

        The Leading term signifies the term having
        the highest power of the principal generator in the
        expression f along with its coefficient.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).LT()
        (x**2*y**0, 4)

        """
        # 获取表达式 f 的首项及其系数
        monom, coeff = f.terms(order)[0]
        return Monomial(monom, f.gens), coeff

    def ET(f, order=None):
        """
        Returns the last non-zero term of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).ET()
        (x**0*y**1, 3)

        """
        # 获取表达式 f 的最后一个非零项及其系数
        monom, coeff = f.terms(order)[-1]
        return Monomial(monom, f.gens), coeff

    def max_norm(f):
        """
        Returns maximum norm of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(-x**2 + 2*x - 3, x).max_norm()
        3

        """
        # 如果 f 的表示支持 max_norm 操作，则计算其最大范数
        if hasattr(f.rep, 'max_norm'):
            result = f.rep.max_norm()
        else:  # pragma: no cover
            # 如果不支持 max_norm 操作，则抛出异常
            raise OperationNotSupported(f, 'max_norm')

        # 将结果转换为 SymPy 表达式返回
        return f.rep.dom.to_sympy(result)

    def l1_norm(f):
        """
        Returns l1 norm of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(-x**2 + 2*x - 3, x).l1_norm()
        6

        """
        # 如果 f 的表示支持 l1_norm 操作，则计算其 l1 范数
        if hasattr(f.rep, 'l1_norm'):
            result = f.rep.l1_norm()
        else:  # pragma: no cover
            # 如果不支持 l1_norm 操作，则抛出异常
            raise OperationNotSupported(f, 'l1_norm')

        # 将结果转换为 SymPy 表达式返回
        return f.rep.dom.to_sympy(result)
    # 清除多项式的分母，保持其所在域不变
    def clear_denoms(self, convert=False):
        """
        Clear denominators, but keep the ground domain.

        Examples
        ========

        >>> from sympy import Poly, S, QQ
        >>> from sympy.abc import x

        >>> f = Poly(x/2 + S(1)/3, x, domain=QQ)

        >>> f.clear_denoms()
        (6, Poly(3*x + 2, x, domain='QQ'))
        >>> f.clear_denoms(convert=True)
        (6, Poly(3*x + 2, x, domain='ZZ'))

        """
        f = self

        # 如果多项式所在环不是域，则返回多项式的系数为1和原多项式本身
        if not f.rep.dom.is_Field:
            return S.One, f

        # 获取多项式的域
        dom = f.get_domain()
        # 如果域具有关联环，将域设为其关联环
        if dom.has_assoc_Ring:
            dom = f.rep.dom.get_ring()

        # 如果多项式的表示具有清除分母的方法，则调用该方法
        if hasattr(f.rep, 'clear_denoms'):
            coeff, result = f.rep.clear_denoms()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'clear_denoms')

        # 将系数转换为 SymPy 格式，并将结果应用到多项式上
        coeff, f = dom.to_sympy(coeff), f.per(result)

        # 如果不需要转换或者域没有关联环，则返回系数和处理后的多项式
        if not convert or not dom.has_assoc_Ring:
            return coeff, f
        else:
            # 否则，将处理后的多项式转换为环
            return coeff, f.to_ring()

    # 清除有理函数的分母
    def rat_clear_denoms(self, g):
        """
        Clear denominators in a rational function ``f/g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2/y + 1, x)
        >>> g = Poly(x**3 + y, x)

        >>> p, q = f.rat_clear_denoms(g)

        >>> p
        Poly(x**2 + y, x, domain='ZZ[y]')
        >>> q
        Poly(y*x**3 + y**2, x, domain='ZZ[y]')

        """
        f = self

        # 统一两个多项式的域和表示
        dom, per, f, g = f._unify(g)

        f = per(f)
        g = per(g)

        # 如果域不是域或者没有关联环，则直接返回 f 和 g
        if not (dom.is_Field and dom.has_assoc_Ring):
            return f, g

        # 分别清除 f 和 g 的分母，并转换为整数系数
        a, f = f.clear_denoms(convert=True)
        b, g = g.clear_denoms(convert=True)

        # 将 f 和 g 的分母乘到多项式上
        f = f.mul_ground(b)
        g = g.mul_ground(a)

        return f, g

    # 计算多项式的不定积分
    def integrate(self, *specs, **args):
        """
        Computes indefinite integral of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x + 1, x).integrate()
        Poly(1/3*x**3 + x**2 + x, x, domain='QQ')

        >>> Poly(x*y**2 + x, x, y).integrate((0, 1), (1, 0))
        Poly(1/2*x**2*y**2 + 1/2*x**2, x, y, domain='QQ')

        """
        f = self

        # 如果设置了自动模式并且多项式所在环是环而非域，则转换为域
        if args.get('auto', True) and f.rep.dom.is_Ring:
            f = f.to_field()

        # 如果多项式的表示具有积分方法，则调用该方法
        if hasattr(f.rep, 'integrate'):
            if not specs:
                return f.per(f.rep.integrate(m=1))

            rep = f.rep

            for spec in specs:
                if isinstance(spec, tuple):
                    gen, m = spec
                else:
                    gen, m = spec, 1

                rep = rep.integrate(int(m), f._gen_to_level(gen))

            return f.per(rep)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'integrate')
    # 定义一个函数 `diff`，用于计算函数 `f` 的偏导数
    def diff(f, *specs, **kwargs):
        """
        Computes partial derivative of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x + 1, x).diff()
        Poly(2*x + 2, x, domain='ZZ')

        >>> Poly(x*y**2 + x, x, y).diff((0, 0), (1, 1))
        Poly(2*x*y, x, y, domain='ZZ')

        """
        # 如果 `evaluate` 关键字参数为 False，则返回 `Derivative` 对象
        if not kwargs.get('evaluate', True):
            return Derivative(f, *specs, **kwargs)

        # 如果 `f.rep` 具有 `diff` 属性，执行下面的逻辑
        if hasattr(f.rep, 'diff'):
            # 如果没有给定偏导数规范 `specs`，则对 `f` 进行一阶偏导数计算
            if not specs:
                return f.per(f.rep.diff(m=1))

            # 使用 `f.rep` 进行偏导数计算，并根据给定的规范 `specs` 迭代计算
            rep = f.rep
            for spec in specs:
                # 如果规范 `spec` 是一个元组，则解析出生成元和阶数 `m`
                if isinstance(spec, tuple):
                    gen, m = spec
                else:
                    gen, m = spec, 1

                # 根据指定的阶数 `m` 和生成元 `gen` 进行偏导数计算
                rep = rep.diff(int(m), f._gen_to_level(gen))

            # 返回计算结果
            return f.per(rep)
        else:  # pragma: no cover
            # 如果 `f.rep` 没有 `diff` 属性，抛出不支持的操作异常
            raise OperationNotSupported(f, 'diff')

    # 将 `diff` 函数赋值给 `_eval_derivative` 变量
    _eval_derivative = diff
    def eval(self, x, a=None, auto=True):
        """
        Evaluate ``f`` at ``a`` in the given variable.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x**2 + 2*x + 3, x).eval(2)
        11

        >>> Poly(2*x*y + 3*x + y + 2, x, y).eval(x, 2)
        Poly(5*y + 8, y, domain='ZZ')

        >>> f = Poly(2*x*y + 3*x + y + 2*z, x, y, z)

        >>> f.eval({x: 2})
        Poly(5*y + 2*z + 6, y, z, domain='ZZ')
        >>> f.eval({x: 2, y: 5})
        Poly(2*z + 31, z, domain='ZZ')
        >>> f.eval({x: 2, y: 5, z: 7})
        45

        >>> f.eval((2, 5))
        Poly(2*z + 31, z, domain='ZZ')
        >>> f(2, 5)
        Poly(2*z + 31, z, domain='ZZ')

        """
        f = self  # 将当前对象赋给变量 f，即将要进行求值操作的多项式对象

        if a is None:
            if isinstance(x, dict):
                mapping = x
                # 如果 x 是字典，则逐一对字典中的变量进行求值，并返回结果多项式
                for gen, value in mapping.items():
                    f = f.eval(gen, value)

                return f
            elif isinstance(x, (tuple, list)):
                values = x
                # 如果 x 是元组或列表，则对应元组或列表中的值依次对多项式的变量进行求值，并返回结果多项式
                if len(values) > len(f.gens):
                    raise ValueError("too many values provided")

                for gen, value in zip(f.gens, values):
                    f = f.eval(gen, value)

                return f
            else:
                j, a = 0, x
        else:
            j = f._gen_to_level(x)  # 根据给定的变量 x，获取其对应的级别 j

        if not hasattr(f.rep, 'eval'):  # pragma: no cover
            raise OperationNotSupported(f, 'eval')

        try:
            result = f.rep.eval(a, j)  # 调用底层表示的 eval 方法来计算多项式在点 a 处的值
        except CoercionFailed:
            if not auto:
                raise DomainError("Cannot evaluate at %s in %s" % (a, f.rep.dom))
            else:
                a_domain, [a] = construct_domain([a])
                new_domain = f.get_domain().unify_with_symbols(a_domain, f.gens)

                f = f.set_domain(new_domain)
                a = new_domain.convert(a, a_domain)

                result = f.rep.eval(a, j)

        return f.per(result, remove=j)  # 对结果进行处理并返回

    def __call__(f, *values):
        """
        Evaluate ``f`` at the give values.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> f = Poly(2*x*y + 3*x + y + 2*z, x, y, z)

        >>> f(2)
        Poly(5*y + 2*z + 6, y, z, domain='ZZ')
        >>> f(2, 5)
        Poly(2*z + 31, z, domain='ZZ')
        >>> f(2, 5, 7)
        45

        """
        return f.eval(values)  # 调用 eval 方法来对多项式进行求值，并返回结果
    # 使用半扩展欧几里得算法计算多项式 f 和 g 的半 gcdex
    def half_gcdex(f, g, auto=True):
        """
        Half extended Euclidean algorithm of ``f`` and ``g``.

        Returns ``(s, h)`` such that ``h = gcd(f, g)`` and ``s*f = h (mod g)``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
        >>> g = x**3 + x**2 - 4*x - 4

        >>> Poly(f).half_gcdex(Poly(g))
        (Poly(-1/5*x + 3/5, x, domain='QQ'), Poly(x + 1, x, domain='QQ'))

        """
        # 统一多项式的环境和表示
        dom, per, F, G = f._unify(g)

        # 如果自动转换开启并且多项式环是一个环
        if auto and dom.is_Ring:
            F, G = F.to_field(), G.to_field()

        # 如果多项式 F 具有 half_gcdex 方法，则调用该方法
        if hasattr(f.rep, 'half_gcdex'):
            s, h = F.half_gcdex(G)
        else:  # pragma: no cover
            # 如果没有支持的操作，则引发操作不支持异常
            raise OperationNotSupported(f, 'half_gcdex')

        # 返回结果，转换回原始多项式环境
        return per(s), per(h)

    # 使用扩展欧几里得算法计算多项式 f 和 g 的 gcdex
    def gcdex(f, g, auto=True):
        """
        Extended Euclidean algorithm of ``f`` and ``g``.

        Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
        >>> g = x**3 + x**2 - 4*x - 4

        >>> Poly(f).gcdex(Poly(g))
        (Poly(-1/5*x + 3/5, x, domain='QQ'),
         Poly(1/5*x**2 - 6/5*x + 2, x, domain='QQ'),
         Poly(x + 1, x, domain='QQ'))

        """
        # 统一多项式的环境和表示
        dom, per, F, G = f._unify(g)

        # 如果自动转换开启并且多项式环是一个环
        if auto and dom.is_Ring:
            F, G = F.to_field(), G.to_field()

        # 如果多项式 F 具有 gcdex 方法，则调用该方法
        if hasattr(f.rep, 'gcdex'):
            s, t, h = F.gcdex(G)
        else:  # pragma: no cover
            # 如果没有支持的操作，则引发操作不支持异常
            raise OperationNotSupported(f, 'gcdex')

        # 返回结果，转换回原始多项式环境
        return per(s), per(t), per(h)

    # 在可能的情况下，计算多项式 f 对于 g 的模反元素
    def invert(f, g, auto=True):
        """
        Invert ``f`` modulo ``g`` when possible.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).invert(Poly(2*x - 1, x))
        Poly(-4/3, x, domain='QQ')

        >>> Poly(x**2 - 1, x).invert(Poly(x - 1, x))
        Traceback (most recent call last):
        ...
        NotInvertible: zero divisor

        """
        # 统一多项式的环境和表示
        dom, per, F, G = f._unify(g)

        # 如果自动转换开启并且多项式环是一个环
        if auto and dom.is_Ring:
            F, G = F.to_field(), G.to_field()

        # 如果多项式 F 具有 invert 方法，则调用该方法
        if hasattr(f.rep, 'invert'):
            result = F.invert(G)
        else:  # pragma: no cover
            # 如果没有支持的操作，则引发操作不支持异常
            raise OperationNotSupported(f, 'invert')

        # 返回结果，转换回原始多项式环境
        return per(result)
    def revert(f, n):
        """
        Compute ``f**(-1)`` mod ``x**n``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(1, x).revert(2)
        Poly(1, x, domain='ZZ')

        >>> Poly(1 + x, x).revert(1)
        Poly(1, x, domain='ZZ')

        >>> Poly(x**2 - 2, x).revert(2)
        Traceback (most recent call last):
        ...
        NotReversible: only units are reversible in a ring

        >>> Poly(1/x, x).revert(1)
        Traceback (most recent call last):
        ...
        PolynomialError: 1/x contains an element of the generators set

        """
        # 检查是否存在 `revert` 方法，如果存在则调用它计算逆元
        if hasattr(f.rep, 'revert'):
            result = f.rep.revert(int(n))
        else:  # pragma: no cover
            # 如果不存在 `revert` 方法，则抛出不支持操作的异常
            raise OperationNotSupported(f, 'revert')

        # 返回计算结果，并调用 `per` 方法对结果进行处理
        return f.per(result)

    def subresultants(f, g):
        """
        Computes the subresultant PRS of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).subresultants(Poly(x**2 - 1, x))
        [Poly(x**2 + 1, x, domain='ZZ'),
         Poly(x**2 - 1, x, domain='ZZ'),
         Poly(-2, x, domain='ZZ')]

        """
        # 将输入多项式统一，获取相应的转换函数及多项式对象
        _, per, F, G = f._unify(g)

        # 检查是否存在 `subresultants` 方法，如果存在则调用它计算子结果多项式序列
        if hasattr(f.rep, 'subresultants'):
            result = F.subresultants(G)
        else:  # pragma: no cover
            # 如果不存在 `subresultants` 方法，则抛出不支持操作的异常
            raise OperationNotSupported(f, 'subresultants')

        # 对计算结果列表中的每个多项式应用转换函数，并返回结果列表
        return list(map(per, result))

    def resultant(f, g, includePRS=False):
        """
        Computes the resultant of ``f`` and ``g`` via PRS.

        If includePRS=True, it includes the subresultant PRS in the result.
        Because the PRS is used to calculate the resultant, this is more
        efficient than calling :func:`subresultants` separately.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(x**2 + 1, x)

        >>> f.resultant(Poly(x**2 - 1, x))
        4
        >>> f.resultant(Poly(x**2 - 1, x), includePRS=True)
        (4, [Poly(x**2 + 1, x, domain='ZZ'), Poly(x**2 - 1, x, domain='ZZ'),
             Poly(-2, x, domain='ZZ')])

        """
        # 将输入多项式统一，获取相应的转换函数及多项式对象
        _, per, F, G = f._unify(g)

        # 检查是否存在 `resultant` 方法，如果存在则调用它计算结果式
        if hasattr(f.rep, 'resultant'):
            if includePRS:
                result, R = F.resultant(G, includePRS=includePRS)
            else:
                result = F.resultant(G)
        else:  # pragma: no cover
            # 如果不存在 `resultant` 方法，则抛出不支持操作的异常
            raise OperationNotSupported(f, 'resultant')

        # 如果 `includePRS` 为 True，则返回结果及其转换后的子结果多项式序列
        if includePRS:
            return (per(result, remove=0), list(map(per, R)))
        # 否则，只返回结果的转换后值
        return per(result, remove=0)
    def discriminant(f):
        """
        Computes the discriminant of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 2*x + 3, x).discriminant()
        -8

        """
        # 检查 f.rep 是否有 discriminant 方法
        if hasattr(f.rep, 'discriminant'):
            # 计算 f.rep 的 discriminant
            result = f.rep.discriminant()
        else:  # pragma: no cover
            # 如果没有该方法，则抛出异常
            raise OperationNotSupported(f, 'discriminant')

        # 返回 f.per(result, remove=0)
        return f.per(result, remove=0)

    def dispersionset(f, g=None):
        r"""Compute the *dispersion set* of two polynomials.

        For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
        and `\deg g > 0` the dispersion set `\operatorname{J}(f, g)` is defined as:

        .. math::
            \operatorname{J}(f, g)
            & := \{a \in \mathbb{N}_0 | \gcd(f(x), g(x+a)) \neq 1\} \\
            &  = \{a \in \mathbb{N}_0 | \deg \gcd(f(x), g(x+a)) \geq 1\}

        For a single polynomial one defines `\operatorname{J}(f) := \operatorname{J}(f, f)`.

        Examples
        ========

        >>> from sympy import poly
        >>> from sympy.polys.dispersion import dispersion, dispersionset
        >>> from sympy.abc import x

        Dispersion set and dispersion of a simple polynomial:

        >>> fp = poly((x - 3)*(x + 3), x)
        >>> sorted(dispersionset(fp))
        [0, 6]
        >>> dispersion(fp)
        6

        Note that the definition of the dispersion is not symmetric:

        >>> fp = poly(x**4 - 3*x**2 + 1, x)
        >>> gp = fp.shift(-3)
        >>> sorted(dispersionset(fp, gp))
        [2, 3, 4]
        >>> dispersion(fp, gp)
        4
        >>> sorted(dispersionset(gp, fp))
        []
        >>> dispersion(gp, fp)
        -oo

        Computing the dispersion also works over field extensions:

        >>> from sympy import sqrt
        >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
        >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
        >>> sorted(dispersionset(fp, gp))
        [2]
        >>> sorted(dispersionset(gp, fp))
        [1, 4]

        We can even perform the computations for polynomials
        having symbolic coefficients:

        >>> from sympy.abc import a
        >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
        >>> sorted(dispersionset(fp))
        [0, 1]

        See Also
        ========

        dispersion

        References
        ==========

        1. [ManWright94]_
        2. [Koepf98]_
        3. [Abramov71]_
        4. [Man93]_
        """
        # 导入 dispersionset 函数
        from sympy.polys.dispersion import dispersionset
        # 返回 dispersionset(f, g)
        return dispersionset(f, g)
    def dispersion(f, g=None):
        r"""Compute the *dispersion* of polynomials.
        
        For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
        and `\deg g > 0` the dispersion `\operatorname{dis}(f, g)` is defined as:
        
        .. math::
            \operatorname{dis}(f, g)
            & := \max\{ J(f,g) \cup \{0\} \} \\
            &  = \max\{ \{a \in \mathbb{N} | \gcd(f(x), g(x+a)) \neq 1\} \cup \{0\} \}
        
        and for a single polynomial `\operatorname{dis}(f) := \operatorname{dis}(f, f)`.
        
        Examples
        ========
        
        >>> from sympy import poly
        >>> from sympy.polys.dispersion import dispersion, dispersionset
        >>> from sympy.abc import x
        
        Dispersion set and dispersion of a simple polynomial:
        
        >>> fp = poly((x - 3)*(x + 3), x)
        >>> sorted(dispersionset(fp))
        [0, 6]
        >>> dispersion(fp)
        6
        
        Note that the definition of the dispersion is not symmetric:
        
        >>> fp = poly(x**4 - 3*x**2 + 1, x)
        >>> gp = fp.shift(-3)
        >>> sorted(dispersionset(fp, gp))
        [2, 3, 4]
        >>> dispersion(fp, gp)
        4
        >>> sorted(dispersionset(gp, fp))
        []
        >>> dispersion(gp, fp)
        -oo
        
        Computing the dispersion also works over field extensions:
        
        >>> from sympy import sqrt
        >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
        >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
        >>> sorted(dispersionset(fp, gp))
        [2]
        >>> sorted(dispersionset(gp, fp))
        [1, 4]
        
        We can even perform the computations for polynomials
        having symbolic coefficients:
        
        >>> from sympy.abc import a
        >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
        >>> sorted(dispersionset(fp))
        [0, 1]
        
        See Also
        ========
        
        dispersionset
        
        References
        ==========
        
        1. [ManWright94]_
        2. [Koepf98]_
        3. [Abramov71]_
        4. [Man93]_
        """
        # 导入 dispersion 函数并调用，返回 dispersion(f, g) 的计算结果
        from sympy.polys.dispersion import dispersion
        return dispersion(f, g)
    def cofactors(f, g):
        """
        Returns the GCD of ``f`` and ``g`` and their cofactors.

        Returns polynomials ``(h, cff, cfg)`` such that ``h = gcd(f, g)``, and
        ``cff = quo(f, h)`` and ``cfg = quo(g, h)`` are, so called, cofactors
        of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).cofactors(Poly(x**2 - 3*x + 2, x))
        (Poly(x - 1, x, domain='ZZ'),
         Poly(x + 1, x, domain='ZZ'),
         Poly(x - 2, x, domain='ZZ'))

        """
        _, per, F, G = f._unify(g)  # 获取 ``f`` 和 ``g`` 的统一表示形式

        if hasattr(f.rep, 'cofactors'):  # 如果 ``f`` 的表示对象支持 cofactors 操作
            h, cff, cfg = F.cofactors(G)  # 计算 ``f`` 和 ``g`` 的 cofactors
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'cofactors')  # 抛出不支持操作的异常

        return per(h), per(cff), per(cfg)  # 返回结果，确保结果采用统一表示形式

    def gcd(f, g):
        """
        Returns the polynomial GCD of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).gcd(Poly(x**2 - 3*x + 2, x))
        Poly(x - 1, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)  # 获取 ``f`` 和 ``g`` 的统一表示形式

        if hasattr(f.rep, 'gcd'):  # 如果 ``f`` 的表示对象支持 gcd 操作
            result = F.gcd(G)  # 计算 ``f`` 和 ``g`` 的 gcd
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'gcd')  # 抛出不支持操作的异常

        return per(result)  # 返回结果，确保结果采用统一表示形式

    def lcm(f, g):
        """
        Returns polynomial LCM of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).lcm(Poly(x**2 - 3*x + 2, x))
        Poly(x**3 - 2*x**2 - x + 2, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)  # 获取 ``f`` 和 ``g`` 的统一表示形式

        if hasattr(f.rep, 'lcm'):  # 如果 ``f`` 的表示对象支持 lcm 操作
            result = F.lcm(G)  # 计算 ``f`` 和 ``g`` 的 lcm
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'lcm')  # 抛出不支持操作的异常

        return per(result)  # 返回结果，确保结果采用统一表示形式

    def trunc(f, p):
        """
        Reduce ``f`` modulo a constant ``p``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**3 + 3*x**2 + 5*x + 7, x).trunc(3)
        Poly(-x**3 - x + 1, x, domain='ZZ')

        """
        p = f.rep.dom.convert(p)  # 将常数 ``p`` 转换为与 ``f`` 同一域的表示形式

        if hasattr(f.rep, 'trunc'):  # 如果 ``f`` 的表示对象支持 trunc 操作
            result = f.rep.trunc(p)  # 对 ``f`` 进行截断操作
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'trunc')  # 抛出不支持操作的异常

        return f.per(result)  # 返回结果，确保结果采用原来的表示形式
    def monic(self, auto=True):
        """
        Divides all coefficients by ``LC(f)``.

        Examples
        ========

        >>> from sympy import Poly, ZZ
        >>> from sympy.abc import x

        >>> Poly(3*x**2 + 6*x + 9, x, domain=ZZ).monic()
        Poly(x**2 + 2*x + 3, x, domain='QQ')

        >>> Poly(3*x**2 + 4*x + 2, x, domain=ZZ).monic()
        Poly(x**2 + 4/3*x + 2/3, x, domain='QQ')

        """
        f = self  # 将当前对象赋值给变量 f

        # 如果 auto 为真且 f 的表示域是一个环（Ring），则将 f 转换为域（Field）
        if auto and f.rep.dom.is_Ring:
            f = f.to_field()

        # 如果 f 的表示对象有 monic 方法，则调用 monic 方法
        if hasattr(f.rep, 'monic'):
            result = f.rep.monic()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'monic')  # 抛出不支持的操作异常

        return f.per(result)  # 返回经过操作后的结果

    def content(f):
        """
        Returns the GCD of polynomial coefficients.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(6*x**2 + 8*x + 12, x).content()
        2

        """
        # 如果 f 的表示对象有 content 方法，则调用 content 方法
        if hasattr(f.rep, 'content'):
            result = f.rep.content()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'content')  # 抛出不支持的操作异常

        return f.rep.dom.to_sympy(result)  # 返回结果转换为 SymPy 的对象

    def primitive(f):
        """
        Returns the content and a primitive form of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**2 + 8*x + 12, x).primitive()
        (2, Poly(x**2 + 4*x + 6, x, domain='ZZ'))

        """
        # 如果 f 的表示对象有 primitive 方法，则调用 primitive 方法
        if hasattr(f.rep, 'primitive'):
            cont, result = f.rep.primitive()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'primitive')  # 抛出不支持的操作异常

        return f.rep.dom.to_sympy(cont), f.per(result)  # 返回内容和操作后的结果

    def compose(f, g):
        """
        Computes the functional composition of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + x, x).compose(Poly(x - 1, x))
        Poly(x**2 - x, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)  # 解构 _unify 方法的返回值

        # 如果 f 的表示对象有 compose 方法，则调用 compose 方法
        if hasattr(f.rep, 'compose'):
            result = F.compose(G)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'compose')  # 抛出不支持的操作异常

        return per(result)  # 返回操作后的结果

    def decompose(f):
        """
        Computes a functional decomposition of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**4 + 2*x**3 - x - 1, x, domain='ZZ').decompose()
        [Poly(x**2 - x - 1, x, domain='ZZ'), Poly(x**2 + x, x, domain='ZZ')]

        """
        # 如果 f 的表示对象有 decompose 方法，则调用 decompose 方法
        if hasattr(f.rep, 'decompose'):
            result = f.rep.decompose()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'decompose')  # 抛出不支持的操作异常

        return list(map(f.per, result))  # 返回操作后的结果列表
    def shift(f, a):
        """
        Efficiently compute Taylor shift ``f(x + a)``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 2*x + 1, x).shift(2)
        Poly(x**2 + 2*x + 1, x, domain='ZZ')

        See Also
        ========

        shift_list: Analogous method for multivariate polynomials.
        """
        # 调用多项式对象的表示式的平移方法，并返回结果多项式对象
        return f.per(f.rep.shift(a))

    def shift_list(f, a):
        """
        Efficiently compute Taylor shift ``f(X + A)``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x*y, [x,y]).shift_list([1, 2]) == Poly((x+1)*(y+2), [x,y])
        True

        See Also
        ========

        shift: Analogous method for univariate polynomials.
        """
        # 调用多项式对象的表示式的列表平移方法，并返回结果多项式对象
        return f.per(f.rep.shift_list(a))

    def transform(f, p, q):
        """
        Efficiently evaluate the functional transformation ``q**n * f(p/q)``.


        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1, x), Poly(x - 1, x))
        Poly(4, x, domain='ZZ')

        """
        # 将多项式 f、p、q 统一到相同的领域上
        P, Q = p.unify(q)
        F, P = f.unify(P)
        F, Q = F.unify(Q)

        # 如果 F.rep 具有 transform 方法，则调用该方法，否则抛出异常
        if hasattr(F.rep, 'transform'):
            result = F.rep.transform(P.rep, Q.rep)
        else:  # pragma: no cover
            raise OperationNotSupported(F, 'transform')

        # 返回经过变换后的多项式对象
        return F.per(result)

    def sturm(self, auto=True):
        """
        Computes the Sturm sequence of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 - 2*x**2 + x - 3, x).sturm()
        [Poly(x**3 - 2*x**2 + x - 3, x, domain='QQ'),
         Poly(3*x**2 - 4*x + 1, x, domain='QQ'),
         Poly(2/9*x + 25/9, x, domain='QQ'),
         Poly(-2079/4, x, domain='QQ')]

        """
        # 将 self 赋值给 f
        f = self

        # 如果 auto 为真且 f.rep.dom 是环，则将 f 转换为域
        if auto and f.rep.dom.is_Ring:
            f = f.to_field()

        # 如果 f.rep 具有 sturm 方法，则调用该方法，否则抛出异常
        if hasattr(f.rep, 'sturm'):
            result = f.rep.sturm()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'sturm')

        # 将结果映射为多项式对象列表并返回
        return list(map(f.per, result))

    def gff_list(f):
        """
        Computes greatest factorial factorization of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**5 + 2*x**4 - x**3 - 2*x**2

        >>> Poly(f).gff_list()
        [(Poly(x, x, domain='ZZ'), 1), (Poly(x + 2, x, domain='ZZ'), 4)]

        """
        # 如果 f.rep 具有 gff_list 方法，则调用该方法，否则抛出异常
        if hasattr(f.rep, 'gff_list'):
            result = f.rep.gff_list()
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'gff_list')

        # 将结果映射为带有 f.per 处理的元组列表并返回
        return [(f.per(g), k) for g, k in result]
    def norm(f):
        """
        计算多项式 ``f`` 在数域 ``K`` 上定义的共轭的乘积，即 ``Norm(f)``。

        Examples
        ========

        >>> from sympy import Poly, sqrt
        >>> from sympy.abc import x

        >>> a, b = sqrt(2), sqrt(3)

        二次扩展域上的多项式。
        两个共轭 x - a 和 x + a。

        >>> f = Poly(x - a, x, extension=a)
        >>> f.norm()
        Poly(x**2 - 2, x, domain='QQ')

        四次扩展域上的多项式。
        四个共轭 x - a, x - a, x + a 和 x + a。

        >>> f = Poly(x - a, x, extension=(a, b))
        >>> f.norm()
        Poly(x**4 - 4*x**2 + 4, x, domain='QQ')

        """
        if hasattr(f.rep, 'norm'):
            # 如果多项式对象具有 norm 方法，则调用其 norm 方法
            r = f.rep.norm()
        else:  # pragma: no cover
            # 如果没有 norm 方法，抛出异常
            raise OperationNotSupported(f, 'norm')

        # 返回结果多项式的乘积
        return f.per(r)

    def sqf_norm(f):
        """
        计算多项式 ``f`` 的平方自由范数。

        返回 ``s``, ``f``, ``r``, 其中 ``g(x) = f(x-sa)`` 且
        ``r(x) = Norm(g(x))`` 是 ``K`` 上的平方自由多项式，
        其中 ``a`` 是底域的代数扩展。

        Examples
        ========

        >>> from sympy import Poly, sqrt
        >>> from sympy.abc import x

        >>> s, f, r = Poly(x**2 + 1, x, extension=[sqrt(3)]).sqf_norm()

        >>> s
        [1]
        >>> f
        Poly(x**2 - 2*sqrt(3)*x + 4, x, domain='QQ<sqrt(3)>')
        >>> r
        Poly(x**4 - 4*x**2 + 16, x, domain='QQ')

        """
        if hasattr(f.rep, 'sqf_norm'):
            # 如果多项式对象具有 sqf_norm 方法，则调用其 sqf_norm 方法
            s, g, r = f.rep.sqf_norm()
        else:  # pragma: no cover
            # 如果没有 sqf_norm 方法，抛出异常
            raise OperationNotSupported(f, 'sqf_norm')

        # 返回结果：平方自由范数的 s, f 和 r
        return s, f.per(g), f.per(r)

    def sqf_part(f):
        """
        计算多项式 ``f`` 的平方自由部分。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 - 3*x - 2, x).sqf_part()
        Poly(x**2 - x - 2, x, domain='ZZ')

        """
        if hasattr(f.rep, 'sqf_part'):
            # 如果多项式对象具有 sqf_part 方法，则调用其 sqf_part 方法
            result = f.rep.sqf_part()
        else:  # pragma: no cover
            # 如果没有 sqf_part 方法，抛出异常
            raise OperationNotSupported(f, 'sqf_part')

        # 返回结果多项式的平方自由部分
        return f.per(result)
    def sqf_list(f, all=False):
        """
        Returns a list of square-free factors of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

        >>> Poly(f).sqf_list()
        (2, [(Poly(x + 1, x, domain='ZZ'), 2),
             (Poly(x + 2, x, domain='ZZ'), 3)])

        >>> Poly(f).sqf_list(all=True)
        (2, [(Poly(1, x, domain='ZZ'), 1),
             (Poly(x + 1, x, domain='ZZ'), 2),
             (Poly(x + 2, x, domain='ZZ'), 3)])

        """
        # 检查 f 的表示是否有 sqf_list 方法
        if hasattr(f.rep, 'sqf_list'):
            # 调用 sqf_list 方法并获取返回的系数和因子列表
            coeff, factors = f.rep.sqf_list(all)
        else:  # pragma: no cover
            # 如果没有 sqf_list 方法，则抛出不支持的操作异常
            raise OperationNotSupported(f, 'sqf_list')

        # 将系数转换为 SymPy 表达式，并对每个因子应用 f.per 方法，构成新的因子列表
        return f.rep.dom.to_sympy(coeff), [(f.per(g), k) for g, k in factors]

    def sqf_list_include(f, all=False):
        """
        Returns a list of square-free factors of ``f``.

        Examples
        ========

        >>> from sympy import Poly, expand
        >>> from sympy.abc import x

        >>> f = expand(2*(x + 1)**3*x**4)
        >>> f
        2*x**7 + 6*x**6 + 6*x**5 + 2*x**4

        >>> Poly(f).sqf_list_include()
        [(Poly(2, x, domain='ZZ'), 1),
         (Poly(x + 1, x, domain='ZZ'), 3),
         (Poly(x, x, domain='ZZ'), 4)]

        >>> Poly(f).sqf_list_include(all=True)
        [(Poly(2, x, domain='ZZ'), 1),
         (Poly(1, x, domain='ZZ'), 2),
         (Poly(x + 1, x, domain='ZZ'), 3),
         (Poly(x, x, domain='ZZ'), 4)]

        """
        # 检查 f 的表示是否有 sqf_list_include 方法
        if hasattr(f.rep, 'sqf_list_include'):
            # 调用 sqf_list_include 方法并获取返回的因子列表
            factors = f.rep.sqf_list_include(all)
        else:  # pragma: no cover
            # 如果没有 sqf_list_include 方法，则抛出不支持的操作异常
            raise OperationNotSupported(f, 'sqf_list_include')

        # 对每个因子应用 f.per 方法，构成新的因子列表
        return [(f.per(g), k) for g, k in factors]

    def factor_list(f):
        """
        Returns a list of irreducible factors of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = 2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y

        >>> Poly(f).factor_list()
        (2, [(Poly(x + y, x, y, domain='ZZ'), 1),
             (Poly(x**2 + 1, x, y, domain='ZZ'), 2)])

        """
        # 检查 f 的表示是否有 factor_list 方法
        if hasattr(f.rep, 'factor_list'):
            try:
                # 尝试调用 factor_list 方法并获取返回的系数和因子列表
                coeff, factors = f.rep.factor_list()
            except DomainError:
                # 如果出现 DomainError 异常，则根据多项式的度返回不可约因子的表达式
                if f.degree() == 0:
                    return f.as_expr(), []
                else:
                    return S.One, [(f, 1)]
        else:  # pragma: no cover
            # 如果没有 factor_list 方法，则抛出不支持的操作异常
            raise OperationNotSupported(f, 'factor_list')

        # 将系数转换为 SymPy 表达式，并对每个因子应用 f.per 方法，构成新的因子列表
        return f.rep.dom.to_sympy(coeff), [(f.per(g), k) for g, k in factors]
    def factor_list_include(f):
        """
        Returns a list of irreducible factors of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = 2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y

        >>> Poly(f).factor_list_include()
        [(Poly(2*x + 2*y, x, y, domain='ZZ'), 1),
         (Poly(x**2 + 1, x, y, domain='ZZ'), 2)]

        """
        # 检查 f.rep 是否有 factor_list_include 方法
        if hasattr(f.rep, 'factor_list_include'):
            try:
                # 调用 f.rep 的 factor_list_include 方法计算因子
                factors = f.rep.factor_list_include()
            except DomainError:
                # 如果计算中出现 DomainError，则返回 f 本身作为一个因子
                return [(f, 1)]
        else:  # pragma: no cover
            # 如果 f.rep 没有 factor_list_include 方法，则抛出 OperationNotSupported 异常
            raise OperationNotSupported(f, 'factor_list_include')

        # 对每个因子 (g, k)，将 f 除以 g，并返回 (f/g, k)
        return [(f.per(g), k) for g, k in factors]
    def intervals(f, all=False, eps=None, inf=None, sup=None, fast=False, sqf=False):
        """
        Compute isolating intervals for roots of ``f``.

        For real roots the Vincent-Akritas-Strzebonski (VAS) continued fractions method is used.

        References
        ==========
        .. [#] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative Study of Two Real Root
            Isolation Methods . Nonlinear Analysis: Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
        .. [#] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S. Vigklas: Improving the
            Performance of the Continued Fractions Method Using new Bounds of Positive Roots. Nonlinear
            Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 3, x).intervals()
        [((-2, -1), 1), ((1, 2), 1)]
        >>> Poly(x**2 - 3, x).intervals(eps=1e-2)
        [((-26/15, -19/11), 1), ((19/11, 26/15), 1)]

        """
        # 如果 eps 参数不为 None，则转换为有理数 QQ
        if eps is not None:
            eps = QQ.convert(eps)

            # 如果 eps 小于等于 0，则抛出值错误异常
            if eps <= 0:
                raise ValueError("'eps' must be a positive rational")

        # 如果 inf 参数不为 None，则转换为有理数 QQ
        if inf is not None:
            inf = QQ.convert(inf)
        # 如果 sup 参数不为 None，则转换为有理数 QQ
        if sup is not None:
            sup = QQ.convert(sup)

        # 如果 f.rep 具有 intervals 方法，则调用其 intervals 方法
        if hasattr(f.rep, 'intervals'):
            result = f.rep.intervals(
                all=all, eps=eps, inf=inf, sup=sup, fast=fast, sqf=sqf)
        else:  # pragma: no cover
            # 如果 f.rep 不支持 intervals 方法，则引发 OperationNotSupported 异常
            raise OperationNotSupported(f, 'intervals')

        # 如果 sqf 为 True，则处理结果为平方自由形式
        if sqf:
            # 定义处理实部结果的函数 _real
            def _real(interval):
                s, t = interval
                return (QQ.to_sympy(s), QQ.to_sympy(t))

            # 如果非 all，则返回映射后的实部结果列表
            if not all:
                return list(map(_real, result))

            # 定义处理复数结果的函数 _complex
            def _complex(rectangle):
                (u, v), (s, t) = rectangle
                return (QQ.to_sympy(u) + I*QQ.to_sympy(v),
                        QQ.to_sympy(s) + I*QQ.to_sympy(t))

            # 拆分结果为实部和复数部分，并映射处理后返回列表
            real_part, complex_part = result

            return list(map(_real, real_part)), list(map(_complex, complex_part))
        else:
            # 定义处理实部结果的函数 _real
            def _real(interval):
                (s, t), k = interval
                return ((QQ.to_sympy(s), QQ.to_sympy(t)), k)

            # 如果非 all，则返回映射后的实部结果列表
            if not all:
                return list(map(_real, result))

            # 定义处理复数结果的函数 _complex
            def _complex(rectangle):
                ((u, v), (s, t)), k = rectangle
                return ((QQ.to_sympy(u) + I*QQ.to_sympy(v),
                         QQ.to_sympy(s) + I*QQ.to_sympy(t)), k)

            # 拆分结果为实部和复数部分，并映射处理后返回列表
            real_part, complex_part = result

            return list(map(_real, real_part)), list(map(_complex, complex_part))
    def refine_root(f, s, t, eps=None, steps=None, fast=False, check_sqf=False):
        """
        Refine an isolating interval of a root to the given precision.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 3, x).refine_root(1, 2, eps=1e-2)
        (19/11, 26/15)

        """
        # 如果需要检查多项式是否是平方自由的，并且不是，抛出异常
        if check_sqf and not f.is_sqf:
            raise PolynomialError("only square-free polynomials supported")

        # 将 s 和 t 转换为有理数
        s, t = QQ.convert(s), QQ.convert(t)

        # 如果指定了 eps，则将其转换为有理数，并检查是否为正数
        if eps is not None:
            eps = QQ.convert(eps)

            if eps <= 0:
                raise ValueError("'eps' must be a positive rational")

        # 如果指定了 steps，则转换为整数；否则默认为 1
        if steps is not None:
            steps = int(steps)
        elif eps is None:
            steps = 1

        # 如果多项式对象具有 refine_root 方法，则调用它进行根的细化
        if hasattr(f.rep, 'refine_root'):
            S, T = f.rep.refine_root(s, t, eps=eps, steps=steps, fast=fast)
        else:  # pragma: no cover
            # 如果不支持 refine_root 方法，则抛出操作不支持异常
            raise OperationNotSupported(f, 'refine_root')

        # 将结果转换为 SymPy 的表示并返回
        return QQ.to_sympy(S), QQ.to_sympy(T)

    def count_roots(f, inf=None, sup=None):
        """
        Return the number of roots of ``f`` in ``[inf, sup]`` interval.

        Examples
        ========

        >>> from sympy import Poly, I
        >>> from sympy.abc import x

        >>> Poly(x**4 - 4, x).count_roots(-3, 3)
        2
        >>> Poly(x**4 - 4, x).count_roots(0, 1 + 3*I)
        1

        """
        # 默认情况下，inf 和 sup 都是实数
        inf_real, sup_real = True, True

        # 如果 inf 不为 None，则将其转换为符号对象
        if inf is not None:
            inf = sympify(inf)

            # 如果 inf 是负无穷，则将其置为 None
            if inf is S.NegativeInfinity:
                inf = None
            else:
                re, im = inf.as_real_imag()

                # 如果 inf 是实数，则将其转换为有理数；否则转换为复数并更新标志
                if not im:
                    inf = QQ.convert(inf)
                else:
                    inf, inf_real = list(map(QQ.convert, (re, im))), False

        # 如果 sup 不为 None，则将其转换为符号对象
        if sup is not None:
            sup = sympify(sup)

            # 如果 sup 是正无穷，则将其置为 None
            if sup is S.Infinity:
                sup = None
            else:
                re, im = sup.as_real_imag()

                # 如果 sup 是实数，则将其转换为有理数；否则转换为复数并更新标志
                if not im:
                    sup = QQ.convert(sup)
                else:
                    sup, sup_real = list(map(QQ.convert, (re, im))), False

        # 如果 inf 和 sup 都是实数，则调用 count_real_roots 方法计算实根的数量
        if inf_real and sup_real:
            if hasattr(f.rep, 'count_real_roots'):
                count = f.rep.count_real_roots(inf=inf, sup=sup)
            else:  # pragma: no cover
                # 如果不支持 count_real_roots 方法，则抛出操作不支持异常
                raise OperationNotSupported(f, 'count_real_roots')
        else:
            # 如果 inf 或 sup 不是实数，则调整它们的表示形式为包含复数部分的元组
            if inf_real and inf is not None:
                inf = (inf, QQ.zero)

            if sup_real and sup is not None:
                sup = (sup, QQ.zero)

            # 如果多项式对象具有 count_complex_roots 方法，则调用它计算复根的数量
            if hasattr(f.rep, 'count_complex_roots'):
                count = f.rep.count_complex_roots(inf=inf, sup=sup)
            else:  # pragma: no cover
                # 如果不支持 count_complex_roots 方法，则抛出操作不支持异常
                raise OperationNotSupported(f, 'count_complex_roots')

        # 返回根的数量作为整数
        return Integer(count)
    def root(f, index, radicals=True):
        """
        Get an indexed root of a polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(2*x**3 - 7*x**2 + 4*x + 4)

        >>> f.root(0)
        -1/2
        >>> f.root(1)
        2
        >>> f.root(2)
        2
        >>> f.root(3)
        Traceback (most recent call last):
        ...
        IndexError: root index out of [-3, 2] range, got 3

        >>> Poly(x**5 + x + 1).root(0)
        CRootOf(x**3 - x**2 + 1, 0)

        """
        # 使用 sympy.polys.rootoftools.rootof 函数计算多项式 f 的第 index 个根
        return sympy.polys.rootoftools.rootof(f, index, radicals=radicals)

    def real_roots(f, multiple=True, radicals=True):
        """
        Return a list of real roots with multiplicities.

        See :func:`real_roots` for more explanation.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**3 - 7*x**2 + 4*x + 4).real_roots()
        [-1/2, 2, 2]
        >>> Poly(x**3 + x + 1).real_roots()
        [CRootOf(x**3 + x + 1, 0)]
        """
        # 使用 sympy.polys.rootoftools.CRootOf.real_roots 函数计算多项式 f 的实根列表
        reals = sympy.polys.rootoftools.CRootOf.real_roots(f, radicals=radicals)

        if multiple:
            return reals
        else:
            # 如果 multiple 为 False，则对根进行分组
            return group(reals, multiple=False)

    def all_roots(f, multiple=True, radicals=True):
        """
        Return a list of real and complex roots with multiplicities.

        See :func:`all_roots` for more explanation.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**3 - 7*x**2 + 4*x + 4).all_roots()
        [-1/2, 2, 2]
        >>> Poly(x**3 + x + 1).all_roots()
        [CRootOf(x**3 + x + 1, 0),
         CRootOf(x**3 + x + 1, 1),
         CRootOf(x**3 + x + 1, 2)]

        """
        # 使用 sympy.polys.rootoftools.CRootOf.all_roots 函数计算多项式 f 的所有根列表
        roots = sympy.polys.rootoftools.CRootOf.all_roots(f, radicals=radicals)

        if multiple:
            return roots
        else:
            # 如果 multiple 为 False，则对根进行分组
            return group(roots, multiple=False)

    def ground_roots(f):
        """
        Compute roots of ``f`` by factorization in the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**6 - 4*x**4 + 4*x**3 - x**2).ground_roots()
        {0: 2, 1: 2}

        """
        # 如果 f 是多变量多项式，则抛出异常
        if f.is_multivariate:
            raise MultivariatePolynomialError(
                "Cannot compute ground roots of %s" % f)

        roots = {}

        # 对 f 进行因式分解，并计算线性因子的根及其重数
        for factor, k in f.factor_list()[1]:
            if factor.is_linear:
                a, b = factor.all_coeffs()
                roots[-b/a] = k

        return roots
    def nth_power_roots_poly(f, n):
        """
        构造一个多项式，其根的 n 次幂为 f 的根。

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(x**4 - x**2 + 1)

        >>> f.nth_power_roots_poly(2)
        Poly(x**4 - 2*x**3 + 3*x**2 - 2*x + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(3)
        Poly(x**4 + 2*x**2 + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(4)
        Poly(x**4 + 2*x**3 + 3*x**2 + 2*x + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(12)
        Poly(x**4 - 4*x**3 + 6*x**2 - 4*x + 1, x, domain='ZZ')

        """
        # 检查是否为多元多项式，如果是则抛出异常
        if f.is_multivariate:
            raise MultivariatePolynomialError(
                "must be a univariate polynomial")

        # 将 n 转换为符号表达式
        N = sympify(n)

        # 检查 n 是否为正整数，若是则转换为整数类型，否则抛出值错误异常
        if N.is_Integer and N >= 1:
            n = int(N)
        else:
            raise ValueError("'n' must an integer and n >= 1, got %s" % n)

        # 获取多项式的符号变量
        x = f.gen
        # 创建一个虚拟符号变量 t
        t = Dummy('t')

        # 计算多项式 f 与 x**n - t 的结果式的 resultant
        r = f.resultant(f.__class__.from_expr(x**n - t, x, t))

        return r.replace(t, x)

    def same_root(f, a, b):
        """
        判断多项式的两个根是否相等。

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly, exp, I, pi
        >>> f = Poly(cyclotomic_poly(5))
        >>> r0 = exp(2*I*pi/5)
        >>> indices = [i for i, r in enumerate(f.all_roots()) if f.same_root(r, r0)]
        >>> print(indices)
        [3]

        Raises
        ======

        DomainError
            如果多项式的定义域不是 :ref:`ZZ`, :ref:`QQ`, :ref:`RR`, 或者 :ref:`CC`。
        MultivariatePolynomialError
            如果多项式不是单变量多项式。
        PolynomialError
            如果多项式的次数小于 2。

        """
        # 检查是否为多元多项式，如果是则抛出异常
        if f.is_multivariate:
            raise MultivariatePolynomialError(
                "Must be a univariate polynomial")

        # 计算多项式的表示形式的 Mignotte 分隔边界的平方值
        dom_delta_sq = f.rep.mignotte_sep_bound_squared()
        # 获取域的字段并将其转换为 sympy 格式
        delta_sq = f.domain.get_field().to_sympy(dom_delta_sq)
        # 我们有 delta_sq = delta**2，其中 delta 是两个根之间最小分隔的下界。
        # 定义 eps = delta/3，并且定义 eps_sq = eps**2 = delta**2/9。
        eps_sq = delta_sq / 9

        # 计算 evalf(1/eps_sq, 1, {}) 的结果
        r, _, _, _ = evalf(1/eps_sq, 1, {})
        # 计算 log(2, r) 的快速值
        n = fastlog(r)
        # 计算 m = floor(n/2) + (n mod 2)
        m = (n // 2) + (n % 2)

        # 定义 ev 函数，其结果为 quad_to_mpmath(_evalf_with_bounded_error(x, m=m))
        ev = lambda x: quad_to_mpmath(_evalf_with_bounded_error(x, m=m))

        # 对于任意复数 a, b，满足 |a - ev(a)| < eps 和 |b - ev(b)| < eps。
        # 所以如果 |ev(a) - ev(b)|**2 < eps**2，则 |ev(a) - ev(b)| < eps，
        # 因此 |a - b| < 3*eps = delta。
        A, B = ev(a), ev(b)
        return (A.real - B.real)**2 + (A.imag - B.imag)**2 < eps_sq
    def cancel(f, g, include=False):
        """
        Cancel common factors in a rational function ``f/g``.
        
        Examples
        ========
        
        >>> from sympy import Poly
        >>> from sympy.abc import x
        
        >>> Poly(2*x**2 - 2, x).cancel(Poly(x**2 - 2*x + 1, x))
        (1, Poly(2*x + 2, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'))
        
        >>> Poly(2*x**2 - 2, x).cancel(Poly(x**2 - 2*x + 1, x), include=True)
        (Poly(2*x + 2, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'))
        
        """
        # 将输入的有理函数 f/g 统一到一个标准形式
        dom, per, F, G = f._unify(g)
        
        # 检查 F 是否支持 cancel 方法，然后执行因式约分
        if hasattr(F, 'cancel'):
            result = F.cancel(G, include=include)
        else:  # pragma: no cover
            raise OperationNotSupported(f, 'cancel')
        
        # 根据 include 参数选择返回的结果格式
        if not include:
            # 如果定义域具有关联环属性，则获取其环
            if dom.has_assoc_Ring:
                dom = dom.get_ring()
            
            cp, cq, p, q = result
            
            # 将分子和分母转换为 SymPy 表达式
            cp = dom.to_sympy(cp)
            cq = dom.to_sympy(cq)
            
            # 返回约分后的有理函数的分子、分母
            return cp/cq, per(p), per(q)
        else:
            # 返回映射 per 后的结果
            return tuple(map(per, result))
    
    def make_monic_over_integers_by_scaling_roots(f):
        """
        Turn any univariate polynomial over :ref:`QQ` or :ref:`ZZ` into a monic
        polynomial over :ref:`ZZ`, by scaling the roots as necessary.
        
        Explanation
        ===========
        
        This operation can be performed whether or not *f* is irreducible; when
        it is, this can be understood as determining an algebraic integer
        generating the same field as a root of *f*.
        
        Examples
        ========
        
        >>> from sympy import Poly, S
        >>> from sympy.abc import x
        >>> f = Poly(x**2/2 + S(1)/4 * x + S(1)/8, x, domain='QQ')
        >>> f.make_monic_over_integers_by_scaling_roots()
        (Poly(x**2 + 2*x + 4, x, domain='ZZ'), 4)
        
        Returns
        =======
        
        Pair ``(g, c)``
            g is the polynomial
            
            c is the integer by which the roots had to be scaled
        
        """
        # 检查输入多项式是否是一元的，且定义域为 QQ 或 ZZ
        if not f.is_univariate or f.domain not in [ZZ, QQ]:
            raise ValueError('Polynomial must be univariate over ZZ or QQ.')
        
        # 如果多项式已经是单位化且定义域为 ZZ，则直接返回
        if f.is_monic and f.domain == ZZ:
            return f, ZZ.one
        else:
            # 将多项式单位化，并得到清除分母的结果
            fm = f.monic()
            c, _ = fm.clear_denoms()
            
            # 将多项式转换为环 ZZ 上的多项式，并返回结果
            return fm.transform(Poly(fm.gen), c).to_ring(), c
    @property
    def is_zero(f):
        """
        如果 ``f`` 是零多项式则返回 ``True``。

        示例
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(0, x).is_zero
        True
        >>> Poly(1, x).is_zero
        False

        """
        # 返回内部表示是否为零
        return f.rep.is_zero

    @property
    def is_one(f):
        """
        如果 ``f`` 是单位多项式则返回 ``True``。

        示例
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(0, x).is_one
        False
        >>> Poly(1, x).is_one
        True

        """
        # 返回内部表示是否为单位元素
        return f.rep.is_one

    @property
    def is_sqf(f):
        """
        如果 ``f`` 是无平方因子的多项式则返回 ``True``。

        示例
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 2*x + 1, x).is_sqf
        False
        >>> Poly(x**2 - 1, x).is_sqf
        True

        """
        # 返回内部表示是否为无平方因子多项式
        return f.rep.is_sqf
    @property
    def is_monic(f):
        """
        Returns ``True`` if the leading coefficient of ``f`` is one.
    
        Examples
        ========
    
        >>> from sympy import Poly
        >>> from sympy.abc import x
    
        >>> Poly(x + 2, x).is_monic
        True
        >>> Poly(2*x + 2, x).is_monic
        False
    
        """
        # 返回 f 对象的表示(rep)是否为首一多项式的属性值
        return f.rep.is_monic
    
    
    @property
    def is_primitive(f):
        """
        Returns ``True`` if GCD of the coefficients of ``f`` is one.
    
        Examples
        ========
    
        >>> from sympy import Poly
        >>> from sympy.abc import x
    
        >>> Poly(2*x**2 + 6*x + 12, x).is_primitive
        False
        >>> Poly(x**2 + 3*x + 6, x).is_primitive
        True
    
        """
        # 返回 f 对象的表示(rep)是否为原始多项式的属性值
        return f.rep.is_primitive
    
    
    @property
    def is_ground(f):
        """
        Returns ``True`` if ``f`` is an element of the ground domain.
    
        Examples
        ========
    
        >>> from sympy import Poly
        >>> from sympy.abc import x, y
    
        >>> Poly(x, x).is_ground
        False
        >>> Poly(2, x).is_ground
        True
        >>> Poly(y, x).is_ground
        True
    
        """
        # 返回 f 对象的表示(rep)是否为基础域元素的属性值
        return f.rep.is_ground
    
    
    @property
    def is_linear(f):
        """
        Returns ``True`` if ``f`` is linear in all its variables.
    
        Examples
        ========
    
        >>> from sympy import Poly
        >>> from sympy.abc import x, y
    
        >>> Poly(x + y + 2, x, y).is_linear
        True
        >>> Poly(x*y + 2, x, y).is_linear
        False
    
        """
        # 返回 f 对象的表示(rep)是否为线性多项式的属性值
        return f.rep.is_linear
    
    
    @property
    def is_quadratic(f):
        """
        Returns ``True`` if ``f`` is quadratic in all its variables.
    
        Examples
        ========
    
        >>> from sympy import Poly
        >>> from sympy.abc import x, y
    
        >>> Poly(x*y + 2, x, y).is_quadratic
        True
        >>> Poly(x*y**2 + 2, x, y).is_quadratic
        False
    
        """
        # 返回 f 对象的表示(rep)是否为二次多项式的属性值
        return f.rep.is_quadratic
    
    
    @property
    def is_monomial(f):
        """
        Returns ``True`` if ``f`` is zero or has only one term.
    
        Examples
        ========
    
        >>> from sympy import Poly
        >>> from sympy.abc import x
    
        >>> Poly(3*x**2, x).is_monomial
        True
        >>> Poly(3*x**2 + 1, x).is_monomial
        False
    
        """
        # 返回 f 对象的表示(rep)是否为单项式的属性值
        return f.rep.is_monomial
    def is_homogeneous(f):
        """
        Returns ``True`` if ``f`` is a homogeneous polynomial.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. If you want not
        only to check if a polynomial is homogeneous but also compute its
        homogeneous order, then use :func:`Poly.homogeneous_order`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x*y, x, y).is_homogeneous
        True
        >>> Poly(x**3 + x*y, x, y).is_homogeneous
        False

        """
        return f.rep.is_homogeneous
        # 返回 f 对象的表示形式是否为齐次多项式的布尔值

    @property
    def is_irreducible(f):
        """
        Returns ``True`` if ``f`` has no factors over its domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + x + 1, x, modulus=2).is_irreducible
        True
        >>> Poly(x**2 + 1, x, modulus=2).is_irreducible
        False

        """
        return f.rep.is_irreducible
        # 返回 f 对象的表示形式是否为不可约多项式的布尔值

    @property
    def is_univariate(f):
        """
        Returns ``True`` if ``f`` is a univariate polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x + 1, x).is_univariate
        True
        >>> Poly(x*y**2 + x*y + 1, x, y).is_univariate
        False
        >>> Poly(x*y**2 + x*y + 1, x).is_univariate
        True
        >>> Poly(x**2 + x + 1, x, y).is_univariate
        False

        """
        return len(f.gens) == 1
        # 返回 f 对象的生成器数量是否为 1，即判断是否为一元多项式的布尔值

    @property
    def is_multivariate(f):
        """
        Returns ``True`` if ``f`` is a multivariate polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x + 1, x).is_multivariate
        False
        >>> Poly(x*y**2 + x*y + 1, x, y).is_multivariate
        True
        >>> Poly(x*y**2 + x*y + 1, x).is_multivariate
        False
        >>> Poly(x**2 + x + 1, x, y).is_multivariate
        True

        """
        return len(f.gens) != 1
        # 返回 f 对象的生成器数量是否不为 1，即判断是否为多元多项式的布尔值

    @property
    def is_cyclotomic(f):
        """
        Returns ``True`` if ``f`` is a cyclotomic polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1

        >>> Poly(f).is_cyclotomic
        False

        >>> g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1

        >>> Poly(g).is_cyclotomic
        True

        """
        return f.rep.is_cyclotomic
        # 返回 f 对象的表示形式是否为旋转多项式的布尔值

    def __abs__(f):
        return f.abs()
        # 返回 f 的绝对值

    def __neg__(f):
        return f.neg()
        # 返回 f 的相反数

    @_polifyit
    def __add__(f, g):
        return f.add(g)
        # 返回 f 与 g 的加法结果，可能是多项式或其他合适的数学对象

    @_polifyit
    def __radd__(f, g):
        return g.add(f)
        # 返回 g 与 f 的加法结果，可能是多项式或其他合适的数学对象

    @_polifyit
    def __sub__(f, g):
        return f.sub(g)
        # 返回 f 减去 g 的结果，可能是多项式或其他合适的数学对象

    @_polifyit
    def __rsub__(f, g):
        return g.sub(f)
        # 返回 g 减去 f 的结果，可能是多项式或其他合适的数学对象
    # 定义特殊方法 __rsub__，实现 g - f 的操作
    def __rsub__(f, g):
        return g.sub(f)

    # 使用 @_polifyit 装饰器装饰特殊方法 __mul__，实现 f * g 的操作
    @_polifyit
    def __mul__(f, g):
        return f.mul(g)

    # 使用 @_polifyit 装饰器装饰特殊方法 __rmul__，实现 g * f 的操作
    @_polifyit
    def __rmul__(f, g):
        return g.mul(f)

    # 使用 @_sympifyit 装饰器装饰特殊方法 __pow__，实现 f ** n 的操作
    @_sympifyit('n', NotImplemented)
    def __pow__(f, n):
        if n.is_Integer and n >= 0:
            return f.pow(n)
        else:
            return NotImplemented

    # 使用 @_polifyit 装饰器装饰特殊方法 __divmod__，实现 f // g 的操作
    @_polifyit
    def __divmod__(f, g):
        return f.div(g)

    # 使用 @_polifyit 装饰器装饰特殊方法 __rdivmod__，实现 g // f 的操作
    @_polifyit
    def __rdivmod__(f, g):
        return g.div(f)

    # 使用 @_polifyit 装饰器装饰特殊方法 __mod__，实现 f % g 的操作
    @_polifyit
    def __mod__(f, g):
        return f.rem(g)

    # 使用 @_polifyit 装饰器装饰特殊方法 __rmod__，实现 g % f 的操作
    @_polifyit
    def __rmod__(f, g):
        return g.rem(f)

    # 使用 @_polifyit 装饰器装饰特殊方法 __floordiv__，实现 f // g 的操作
    @_polifyit
    def __floordiv__(f, g):
        return f.quo(g)

    # 使用 @_polifyit 装饰器装饰特殊方法 __rfloordiv__，实现 g // f 的操作
    @_polifyit
    def __rfloordiv__(f, g):
        return g.quo(f)

    # 使用 @_sympifyit 装饰器装饰特殊方法 __truediv__，实现 f / g 的操作
    @_sympifyit('g', NotImplemented)
    def __truediv__(f, g):
        return f.as_expr() / g.as_expr()

    # 使用 @_sympifyit 装饰器装饰特殊方法 __rtruediv__，实现 g / f 的操作
    @_sympifyit('g', NotImplemented)
    def __rtruediv__(f, g):
        return g.as_expr() / f.as_expr()

    # 使用 @_sympifyit 装饰器装饰特殊方法 __eq__，实现 f == g 的操作
    @_sympifyit('other', NotImplemented)
    def __eq__(self, other):
        f, g = self, other

        # 如果 g 不是多项式对象，则尝试用 g 创建一个与 f 类型相同的对象
        if not g.is_Poly:
            try:
                g = f.__class__(g, f.gens, domain=f.get_domain())
            except (PolynomialError, DomainError, CoercionFailed):
                return False

        # 检查多项式的生成器是否相同
        if f.gens != g.gens:
            return False

        # 检查多项式的表示域是否相同
        if f.rep.dom != g.rep.dom:
            return False

        # 比较多项式的表示是否相等
        return f.rep == g.rep

    # 使用 @_sympifyit 装饰器装饰特殊方法 __ne__，实现 f != g 的操作
    @_sympifyit('g', NotImplemented)
    def __ne__(f, g):
        return not f == g

    # 定义方法 __bool__，实现布尔测试 f 的操作
    def __bool__(f):
        return not f.is_zero

    # 定义方法 eq，比较 f 和 g 是否相等，可选是否严格比较
    def eq(f, g, strict=False):
        if not strict:
            return f == g
        else:
            return f._strict_eq(sympify(g))

    # 定义方法 ne，比较 f 和 g 是否不相等，可选是否严格比较
    def ne(f, g, strict=False):
        return not f.eq(g, strict=strict)

    # 定义方法 _strict_eq，严格比较 f 和 g 是否相等
    def _strict_eq(f, g):
        return isinstance(g, f.__class__) and f.gens == g.gens and f.rep.eq(g.rep, strict=True)
@public
class PurePoly(Poly):
    """Class for representing pure polynomials. """

    def _hashable_content(self):
        """Allow SymPy to hash Poly instances. """
        # 返回元组，用于唯一标识多项式
        return (self.rep,)

    def __hash__(self):
        # 调用父类的哈希方法，返回哈希值
        return super().__hash__()

    @property
    def free_symbols(self):
        """
        Free symbols of a polynomial.

        Examples
        ========

        >>> from sympy import PurePoly
        >>> from sympy.abc import x, y

        >>> PurePoly(x**2 + 1).free_symbols
        set()
        >>> PurePoly(x**2 + y).free_symbols
        set()
        >>> PurePoly(x**2 + y, x).free_symbols
        {y}

        """
        # 返回多项式在定义域内的自由符号集合
        return self.free_symbols_in_domain

    @_sympifyit('other', NotImplemented)
    def __eq__(self, other):
        f, g = self, other

        if not g.is_Poly:
            # 如果 g 不是 Poly 实例，则尝试将其转换为与 f 相同类型的 Poly
            try:
                g = f.__class__(g, f.gens, domain=f.get_domain())
            except (PolynomialError, DomainError, CoercionFailed):
                return False

        if len(f.gens) != len(g.gens):
            return False

        if f.rep.dom != g.rep.dom:
            # 尝试统一 f 和 g 的定义域
            try:
                dom = f.rep.dom.unify(g.rep.dom, f.gens)
            except UnificationFailed:
                return False

            f = f.set_domain(dom)
            g = g.set_domain(dom)

        # 比较 f 和 g 的表示是否相同
        return f.rep == g.rep

    def _strict_eq(f, g):
        # 判断 g 是否与 f 类型相同，并且严格比较它们的表示
        return isinstance(g, f.__class__) and f.rep.eq(g.rep, strict=True)

    def _unify(f, g):
        g = sympify(g)

        if not g.is_Poly:
            # 如果 g 不是 Poly 实例，则尝试将其转换为 Poly 实例
            try:
                return f.rep.dom, f.per, f.rep, f.rep.per(f.rep.dom.from_sympy(g))
            except CoercionFailed:
                raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if len(f.gens) != len(g.gens):
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if not (isinstance(f.rep, DMP) and isinstance(g.rep, DMP)):
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        cls = f.__class__
        gens = f.gens

        # 统一 f 和 g 的定义域
        dom = f.rep.dom.unify(g.rep.dom, gens)

        F = f.rep.convert(dom)
        G = g.rep.convert(dom)

        def per(rep, dom=dom, gens=gens, remove=None):
            if remove is not None:
                gens = gens[:remove] + gens[remove + 1:]

                if not gens:
                    return dom.to_sympy(rep)

            # 使用类方法创建新的 Poly 实例
            return cls.new(rep, *gens)

        return dom, per, F, G


@public
def poly_from_expr(expr, *gens, **args):
    """Construct a polynomial from an expression. """
    opt = options.build_options(gens, args)
    return _poly_from_expr(expr, opt)


def _poly_from_expr(expr, opt):
    """Construct a polynomial from an expression. """
    orig, expr = expr, sympify(expr)

    if not isinstance(expr, Basic):
        raise PolificationFailed(opt, orig, expr)
    # 如果表达式是多项式（Poly），则将其转换为多项式对象
    elif expr.is_Poly:
        poly = expr.__class__._from_poly(expr, opt)

        # 设置选项对象的生成器为多项式的生成器
        opt.gens = poly.gens
        # 设置选项对象的域为多项式的域
        opt.domain = poly.domain

        # 如果选项中的 polys 属性为 None，则设置为 True
        if opt.polys is None:
            opt.polys = True

        # 返回转换后的多项式对象和更新后的选项对象
        return poly, opt
    # 如果选项中指定要展开表达式
    elif opt.expand:
        # 对表达式进行展开
        expr = expr.expand()

    # 将表达式转换为字典表示形式，并获取更新后的选项对象
    rep, opt = _dict_from_expr(expr, opt)
    
    # 如果没有设置生成器，则抛出多项式转换失败的异常
    if not opt.gens:
        raise PolificationFailed(opt, orig, expr)

    # 将表达式字典分解为单项式列表和系数列表
    monoms, coeffs = list(zip(*list(rep.items())))
    # 获取选项对象的域
    domain = opt.domain

    # 如果域为 None，则根据系数构建适当的域，并更新选项中的域属性
    if domain is None:
        opt.domain, coeffs = construct_domain(coeffs, opt=opt)
    else:
        # 否则，将系数转换为域中对应的表示形式
        coeffs = list(map(domain.from_sympy, coeffs))

    # 重新构建表达式字典，以确保使用正确的域
    rep = dict(list(zip(monoms, coeffs)))
    # 根据更新后的字典和选项对象，创建多项式对象
    poly = Poly._from_dict(rep, opt)

    # 如果选项中的 polys 属性为 None，则设置为 False
    if opt.polys is None:
        opt.polys = False

    # 返回构建好的多项式对象和最终的选项对象
    return poly, opt
# 声明一个公共装饰器，使函数能被外部访问
@public
# 从表达式构造多项式
def parallel_poly_from_expr(exprs, *gens, **args):
    """Construct polynomials from expressions. """
    # 构建选项参数
    opt = options.build_options(gens, args)
    # 调用内部函数来处理表达式列表和选项
    return _parallel_poly_from_expr(exprs, opt)


# 内部函数：从表达式构造多项式
def _parallel_poly_from_expr(exprs, opt):
    """Construct polynomials from expressions. """
    # 如果表达式列表长度为2
    if len(exprs) == 2:
        # 分别取出两个表达式
        f, g = exprs
        
        # 如果 f 和 g 都是 Poly 对象
        if isinstance(f, Poly) and isinstance(g, Poly):
            # 将 f 和 g 转换为相同类的 Poly 对象
            f = f.__class__._from_poly(f, opt)
            g = g.__class__._from_poly(g, opt)

            # 统一 f 和 g 的变量
            f, g = f.unify(g)

            # 将选项中的变量设为 f 的变量
            opt.gens = f.gens
            # 将选项中的域设为 f 的域
            opt.domain = f.domain

            # 如果选项中未指定多项式，则设为 True
            if opt.polys is None:
                opt.polys = True

            # 返回处理后的多项式列表和选项
            return [f, g], opt

    # 复制原始表达式列表，并清空表达式列表
    origs, exprs = list(exprs), []
    # 用于存放表达式和多项式的索引列表
    _exprs, _polys = [], []

    # 是否有处理失败的表达式标志
    failed = False

    # 遍历原始表达式列表
    for i, expr in enumerate(origs):
        # 将表达式转换为 sympy 的基本对象
        expr = sympify(expr)

        # 如果是基本对象
        if isinstance(expr, Basic):
            # 如果是多项式
            if expr.is_Poly:
                # 将索引添加到多项式索引列表中
                _polys.append(i)
            else:
                # 将索引添加到表达式索引列表中
                _exprs.append(i)

                # 如果选项中指定要展开表达式，则展开之
                if opt.expand:
                    expr = expr.expand()
        else:
            # 如果不是基本对象，则表示处理失败
            failed = True

        # 将处理后的表达式添加到表达式列表中
        exprs.append(expr)

    # 如果存在处理失败的表达式，则抛出异常
    if failed:
        raise PolificationFailed(opt, origs, exprs, True)

    # 如果有多项式
    if _polys:
        # 对多项式进行临时处理
        for i in _polys:
            exprs[i] = exprs[i].as_expr()

    # 调用内部函数，从表达式构造字典映射并返回结果
    reps, opt = _parallel_dict_from_expr(exprs, opt)
    
    # 如果没有生成变量列表，则抛出异常
    if not opt.gens:
        raise PolificationFailed(opt, origs, exprs, True)

    # 导入 Piecewise 函数并检查生成器列表
    from sympy.functions.elementary.piecewise import Piecewise
    # 遍历生成器列表
    for k in opt.gens:
        # 如果生成器是 Piecewise 类型，则抛出多项式错误异常
        if isinstance(k, Piecewise):
            raise PolynomialError("Piecewise generators do not make sense")

    # 系数列表、长度列表的初始化
    coeffs_list, lengths = [], []

    # 所有单项式和系数的列表初始化
    all_monoms = []
    all_coeffs = []

    # 遍历映射列表
    for rep in reps:
        # 将单项式和系数分别存入列表
        monoms, coeffs = list(zip(*list(rep.items())))

        # 扩展系数列表
        coeffs_list.extend(coeffs)
        # 将单项式列表添加到所有单项式列表中
        all_monoms.append(monoms)

        # 记录长度
        lengths.append(len(coeffs))

    # 获取选项中的域
    domain = opt.domain

    # 如果域为空，则构建域和系数列表
    if domain is None:
        opt.domain, coeffs_list = construct_domain(coeffs_list, opt=opt)
    else:
        # 否则，将系数列表转换为域
        coeffs_list = list(map(domain.from_sympy, coeffs_list))

    # 遍历长度列表
    for k in lengths:
        # 添加部分系数列表，并更新剩余系数列表
        all_coeffs.append(coeffs_list[:k])
        coeffs_list = coeffs_list[k:]

    # 初始化多项式列表
    polys = []

    # 遍历所有单项式和系数列表
    for monoms, coeffs in zip(all_monoms, all_coeffs):
        # 创建字典映射
        rep = dict(list(zip(monoms, coeffs)))
        # 从字典映射创建多项式
        poly = Poly._from_dict(rep, opt)
        # 将多项式添加到多项式列表中
        polys.append(poly)

    # 如果选项中未指定多项式，则设为 _polys 的布尔值
    if opt.polys is None:
        opt.polys = bool(_polys)

    # 返回多项式列表和选项
    return polys, opt


# 更新参数字典，添加新的 (key, value) 对
def _update_args(args, key, value):
    """Add a new ``(key, value)`` pair to arguments ``dict``. """
    # 将参数字典转换为新的字典
    args = dict(args)

    # 如果 key 不在参数字典中，则添加新的 (key, value) 对
    if key not in args:
        args[key] = value

    # 返回更新后的参数字典
    return args


# 公共装饰器：返回多项式的次数
@public
def degree(f, gen=0):
    """
    Return the degree of ``f`` in the given variable.

    The degree of 0 is negative infinity.

    Examples
    ========

    >>> from sympy import degree
    >>> from sympy.abc import x, y

    >>> degree(x**2 + y*x + 1, gen=x)
    2
    """
    # 返回在给定变量中多项式的次数
    return f.degree(gen)
    # 计算多项式 f 关于变量 gen 的次数
    >>> degree(x**2 + y*x + 1, gen=y)
    # 返回 1，因为该多项式关于 y 的次数为 1
    1
    # 计算零多项式关于变量 x 的次数
    >>> degree(0, x)
    # 返回负无穷，因为零多项式在任何变量上的次数都是负无穷
    -oo

    # 参见
    # ========

    # sympy.polys.polytools.Poly.total_degree
    # degree_list
    """

    # 将输入的表达式转换为 sympy 的表达式对象，严格模式
    f = sympify(f, strict=True)
    # 将输入的 gen 转换为 sympy 的对象，并检查是否为数值类型
    gen_is_Num = sympify(gen, strict=True).is_Number
    # 如果 f 是多项式对象
    if f.is_Poly:
        # 将 f 转换为 p
        p = f
        # 检查 p 是否表示为数值
        isNum = p.as_expr().is_Number
    else:
        # 检查 f 是否为数值
        isNum = f.is_Number
        # 如果 f 不是数值
        if not isNum:
            # 如果 gen 是数值类型
            if gen_is_Num:
                # 从 f 构造多项式对象 p
                p, _ = poly_from_expr(f)
            else:
                # 使用 gen 作为符号变量从 f 构造多项式对象 p
                p, _ = poly_from_expr(f, gen)

    # 如果 f 是数值
    if isNum:
        # 如果 f 是零，则返回零，否则返回负无穷
        return S.Zero if f else S.NegativeInfinity

    # 如果 gen 不是数值类型
    if not gen_is_Num:
        # 如果 f 是多项式且 gen 不在 p 的生成器列表中
        if f.is_Poly and gen not in p.gens:
            # 尝试重新转换不带显式生成器的多项式
            p, _ = poly_from_expr(f.as_expr())
        # 如果 gen 仍然不在 p 的生成器列表中，则返回零
        if gen not in p.gens:
            return S.Zero
    # 如果 f 不是多项式且其自由符号大于 1
    elif not f.is_Poly and len(f.free_symbols) > 1:
        # 抛出类型错误，因为多变量表达式需要一个感兴趣的符号生成器
        raise TypeError(filldedent('''
         A symbolic generator of interest is required for a multivariate
         expression like func = %s, e.g. degree(func, gen = %s) instead of
         degree(func, gen = %s).
        ''' % (f, next(ordered(f.free_symbols)), gen)))
    # 计算多项式 p 关于 gen 的次数，并返回结果
    result = p.degree(gen)
    # 如果结果是整数类型，则返回 Integer 类型结果，否则返回负无穷
    return Integer(result) if isinstance(result, int) else S.NegativeInfinity
@public
def total_degree(f, *gens):
    """
    Return the total_degree of ``f`` in the given variables.

    Examples
    ========
    >>> from sympy import total_degree, Poly
    >>> from sympy.abc import x, y

    >>> total_degree(1)
    0
    >>> total_degree(x + x*y)
    2
    >>> total_degree(x + x*y, x)
    1

    If the expression is a Poly and no variables are given
    then the generators of the Poly will be used:

    >>> p = Poly(x + x*y, y)
    >>> total_degree(p)
    1

    To deal with the underlying expression of the Poly, convert
    it to an Expr:

    >>> total_degree(p.as_expr())
    2

    This is done automatically if any variables are given:

    >>> total_degree(p, x)
    1

    See also
    ========
    degree
    """

    # 将输入的表达式转换为 SymPy 的表达式对象
    p = sympify(f)
    # 如果 p 是多项式对象，则转换为其表达式形式
    if p.is_Poly:
        p = p.as_expr()
    # 如果 p 是数值，则度数为 0
    if p.is_Number:
        rv = 0
    else:
        # 如果输入的 f 是多项式，则使用给定的生成器计算其总次数
        if f.is_Poly:
            gens = gens or f.gens
        rv = Poly(p, gens).total_degree()

    return Integer(rv)


@public
def degree_list(f, *gens, **args):
    """
    Return a list of degrees of ``f`` in all variables.

    Examples
    ========

    >>> from sympy import degree_list
    >>> from sympy.abc import x, y

    >>> degree_list(x**2 + y*x + 1)
    (2, 1)

    """
    # 验证并处理函数的可选参数
    options.allowed_flags(args, ['polys'])

    try:
        # 将输入表达式转换为多项式对象，使用给定的生成器和参数
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 捕获多项式转换失败的异常
        raise ComputationFailed('degree_list', 1, exc)

    # 获取多项式每个变量的次数列表，并将其转换为整数类型的元组
    degrees = F.degree_list()

    return tuple(map(Integer, degrees))


@public
def LC(f, *gens, **args):
    """
    Return the leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy import LC
    >>> from sympy.abc import x, y

    >>> LC(4*x**2 + 2*x*y**2 + x*y + 3*y)
    4

    """
    # 验证并处理函数的可选参数
    options.allowed_flags(args, ['polys'])

    try:
        # 将输入表达式转换为多项式对象，使用给定的生成器和参数
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 捕获多项式转换失败的异常
        raise ComputationFailed('LC', 1, exc)

    # 返回多项式的首项系数
    return F.LC(order=opt.order)


@public
def LM(f, *gens, **args):
    """
    Return the leading monomial of ``f``.

    Examples
    ========

    >>> from sympy import LM
    >>> from sympy.abc import x, y

    >>> LM(4*x**2 + 2*x*y**2 + x*y + 3*y)
    x**2

    """
    # 验证并处理函数的可选参数
    options.allowed_flags(args, ['polys'])

    try:
        # 将输入表达式转换为多项式对象，使用给定的生成器和参数
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 捕获多项式转换失败的异常
        raise ComputationFailed('LM', 1, exc)

    # 返回多项式的首项单项式
    monom = F.LM(order=opt.order)
    return monom.as_expr()


@public
def LT(f, *gens, **args):
    """
    Return the leading term of ``f``.

    Examples
    ========

    >>> from sympy import LT
    >>> from sympy.abc import x, y

    >>> LT(4*x**2 + 2*x*y**2 + x*y + 3*y)
    4*x**2

    """
    # 验证并处理函数的可选参数
    options.allowed_flags(args, ['polys'])

    try:
        # 将输入表达式转换为多项式对象，使用给定的生成器和参数
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 捕获多项式转换失败的异常
        raise ComputationFailed('LT', 1, exc)

    # 返回多项式的首项，包括系数和单项式
    monom, coeff = F.LT(order=opt.order)
    return coeff*monom.as_expr()
# 定义函数 div，用于计算多项式 f 和 g 的除法
def div(f, g, *gens, **args):
    # 使用 options 模块检查并允许 args 中的 'auto' 和 'polys' 标志
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 调用 parallel_poly_from_expr 函数，将 f 和 g 转化为多项式 F 和 G，并获取选项 opt
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，则抛出 ComputationFailed 异常，传递异常信息
        raise ComputationFailed('div', 2, exc)

    # 使用 F 的 div 方法计算多项式除法，得到商 q 和余数 r
    q, r = F.div(G)

    # 如果 opt.polys 为 False，返回结果的表达式形式
    if not opt.polys:
        return q.as_expr(), r.as_expr()
    else:
        return q, r



# 定义函数 prem，用于计算多项式 f 和 g 的伪余数
@public
def prem(f, g, *gens, **args):
    # 使用 options 模块检查并允许 args 中的 'polys' 标志
    options.allowed_flags(args, ['polys'])

    try:
        # 调用 parallel_poly_from_expr 函数，将 f 和 g 转化为多项式 F 和 G，并获取选项 opt
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，则抛出 ComputationFailed 异常，传递异常信息
        raise ComputationFailed('prem', 2, exc)

    # 使用 F 的 prem 方法计算多项式伪余数 r
    r = F.prem(G)

    # 如果 opt.polys 为 False，返回结果的表达式形式
    if not opt.polys:
        return r.as_expr()
    else:
        return r



# 定义函数 pquo，用于计算多项式 f 和 g 的伪商
@public
def pquo(f, g, *gens, **args):
    # 使用 options 模块检查并允许 args 中的 'polys' 标志
    options.allowed_flags(args, ['polys'])

    try:
        # 调用 parallel_poly_from_expr 函数，将 f 和 g 转化为多项式 F 和 G，并获取选项 opt
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，则抛出 ComputationFailed 异常，传递异常信息
        raise ComputationFailed('pquo', 2, exc)

    try:
        # 使用 F 的 pquo 方法计算多项式伪商 q
        q = F.pquo(G)
    except ExactQuotientFailed:
        # 如果无法得到精确的商，抛出 ExactQuotientFailed 异常
        raise ExactQuotientFailed(f, g)

    # 如果 opt.polys 为 False，返回结果的表达式形式
    if not opt.polys:
        return q.as_expr()
    else:
        return q



# 定义函数 pexquo，用于计算多项式 f 和 g 的精确伪商
@public
def pexquo(f, g, *gens, **args):
    # 使用 options 模块检查并允许 args 中的 'polys' 标志
    options.allowed_flags(args, ['polys'])

    try:
        # 调用 parallel_poly_from_expr 函数，将 f 和 g 转化为多项式 F 和 G，并获取选项 opt
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，则抛出 ComputationFailed 异常，传递异常信息
        raise ComputationFailed('pexquo', 2, exc)

    # 使用 F 的 pexquo 方法计算多项式精确伪商 q
    q = F.pexquo(G)

    # 如果 opt.polys 为 False，返回结果的表达式形式
    if not opt.polys:
        return q.as_expr()
    else:
        return q



# 定义函数 pdiv，用于计算多项式 f 和 g 的伪除法
def pdiv(f, g, *gens, **args):
    # 使用 options 模块检查并允许 args 中的 'polys' 标志
    options.allowed_flags(args, ['polys'])

    try:
        # 调用 parallel_poly_from_expr 函数，将 f 和 g 转化为多项式 F 和 G，并获取选项 opt
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，则抛出 ComputationFailed 异常，传递异常信息
        raise ComputationFailed('pdiv', 2, exc)

    # 使用 F 的 pdiv 方法计算多项式伪除法，得到商 q 和余数 r
    q, r = F.pdiv(G)

    # 如果 opt.polys 为 False，返回结果的表达式形式
    if not opt.polys:
        return q.as_expr(), r.as_expr()
    else:
        return q, r
    # 捕获 PolificationFailed 异常，将其转换为 ComputationFailed 异常，并指明失败原因为 'div'、参数为 2、异常信息为 exc
    except PolificationFailed as exc:
        raise ComputationFailed('div', 2, exc)

    # 对 F 除以 G，使用自动选项为 opt.auto
    q, r = F.div(G, auto=opt.auto)

    # 如果 opt.polys 为 False，将商 q 和余数 r 转换为表达式并返回
    if not opt.polys:
        return q.as_expr(), r.as_expr()
    else:
        # 如果 opt.polys 为 True，则直接返回商 q 和余数 r
        return q, r
@public
def rem(f, g, *gens, **args):
    """
    Compute polynomial remainder of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy import rem, ZZ, QQ
    >>> from sympy.abc import x

    >>> rem(x**2 + 1, 2*x - 4, domain=ZZ)
    x**2 + 1
    >>> rem(x**2 + 1, 2*x - 4, domain=QQ)
    5

    """
    # 检查传入的参数中是否包含指定的标志
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 将输入的表达式转换为多项式，获取相关选项
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，则抛出计算失败的异常
        raise ComputationFailed('rem', 2, exc)

    # 计算多项式 F 除以 G 的余数
    r = F.rem(G, auto=opt.auto)

    if not opt.polys:
        # 如果不需要返回多项式对象，则将余数 r 转换为表达式形式返回
        return r.as_expr()
    else:
        # 否则直接返回多项式对象 r
        return r


@public
def quo(f, g, *gens, **args):
    """
    Compute polynomial quotient of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy import quo
    >>> from sympy.abc import x

    >>> quo(x**2 + 1, 2*x - 4)
    x/2 + 1
    >>> quo(x**2 - 1, x - 1)
    x + 1

    """
    # 检查传入的参数中是否包含指定的标志
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 将输入的表达式转换为多项式，获取相关选项
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，则抛出计算失败的异常
        raise ComputationFailed('quo', 2, exc)

    # 计算多项式 F 除以 G 的商
    q = F.quo(G, auto=opt.auto)

    if not opt.polys:
        # 如果不需要返回多项式对象，则将商 q 转换为表达式形式返回
        return q.as_expr()
    else:
        # 否则直接返回多项式对象 q
        return q


@public
def exquo(f, g, *gens, **args):
    """
    Compute polynomial exact quotient of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy import exquo
    >>> from sympy.abc import x

    >>> exquo(x**2 - 1, x - 1)
    x + 1

    >>> exquo(x**2 + 1, 2*x - 4)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1

    """
    # 检查传入的参数中是否包含指定的标志
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 将输入的表达式转换为多项式，获取相关选项
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，则抛出计算失败的异常
        raise ComputationFailed('exquo', 2, exc)

    # 计算多项式 F 除以 G 的精确商
    q = F.exquo(G, auto=opt.auto)

    if not opt.polys:
        # 如果不需要返回多项式对象，则将精确商 q 转换为表达式形式返回
        return q.as_expr()
    else:
        # 否则直接返回多项式对象 q
        return q


@public
def half_gcdex(f, g, *gens, **args):
    """
    Half extended Euclidean algorithm of ``f`` and ``g``.

    Returns ``(s, h)`` such that ``h = gcd(f, g)`` and ``s*f = h (mod g)``.

    Examples
    ========

    >>> from sympy import half_gcdex
    >>> from sympy.abc import x

    >>> half_gcdex(x**4 - 2*x**3 - 6*x**2 + 12*x + 15, x**3 + x**2 - 4*x - 4)
    (3/5 - x/5, x + 1)

    """
    # 检查传入的参数中是否包含指定的标志
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 尝试将输入的表达式转换为多项式，获取相关选项
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，则根据异常构造定义域并尝试使用其半扩展欧几里得算法
        domain, (a, b) = construct_domain(exc.exprs)

        try:
            # 调用定义域的半扩展欧几里得算法
            s, h = domain.half_gcdex(a, b)
        except NotImplementedError:
            # 如果算法未实现，则抛出计算失败的异常
            raise ComputationFailed('half_gcdex', 2, exc)
        else:
            # 将结果转换为 SymPy 表达式并返回
            return domain.to_sympy(s), domain.to_sympy(h)

    # 调用多项式 F 的半扩展欧几里得算法
    s, h = F.half_gcdex(G, auto=opt.auto)

    if not opt.polys:
        # 如果不需要返回多项式对象，则将结果转换为表达式形式返回
        return s.as_expr(), h.as_expr()
    else:
        # 否则直接返回多项式对象 s, h
        return s, h


@public
def gcdex(f, g, *gens, **args):
    """
    """
    Extended Euclidean algorithm of ``f`` and ``g``.

    Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.

    Examples
    ========

    >>> from sympy import gcdex
    >>> from sympy.abc import x

    >>> gcdex(x**4 - 2*x**3 - 6*x**2 + 12*x + 15, x**3 + x**2 - 4*x - 4)
    (3/5 - x/5, x**2/5 - 6*x/5 + 2, x + 1)

    """
    # 检查并允许特定的选项标志，例如 'auto' 或 'polys'
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 尝试将表达式转化为多项式，并返回多项式列表 (F, G) 和选项 opt
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，构造领域并重新尝试
        domain, (a, b) = construct_domain(exc.exprs)

        try:
            # 在新的领域中计算 a 和 b 的扩展欧几里得算法结果 s, t, h
            s, t, h = domain.gcdex(a, b)
        except NotImplementedError:
            # 如果领域不支持扩展欧几里得算法，则抛出计算失败的异常
            raise ComputationFailed('gcdex', 2, exc)
        else:
            # 将领域元素转换为 SymPy 表达式并返回结果
            return domain.to_sympy(s), domain.to_sympy(t), domain.to_sympy(h)

    # 在多项式环 F 和 G 中执行扩展欧几里得算法，并返回结果 s, t, h
    s, t, h = F.gcdex(G, auto=opt.auto)

    # 根据选项返回符号表达式或多项式对象
    if not opt.polys:
        return s.as_expr(), t.as_expr(), h.as_expr()
    else:
        return s, t, h
# 公共函数装饰器，使该函数可以在外部被访问
@public
# 定义函数invert，用于计算在可能情况下，对于给定的f和g，对g取模后f的逆
def invert(f, g, *gens, **args):
    """
    Invert ``f`` modulo ``g`` when possible.

    Examples
    ========

    >>> from sympy import invert, S, mod_inverse
    >>> from sympy.abc import x

    >>> invert(x**2 - 1, 2*x - 1)
    -4/3

    >>> invert(x**2 - 1, x - 1)
    Traceback (most recent call last):
    ...
    NotInvertible: zero divisor

    For more efficient inversion of Rationals,
    use the :obj:`sympy.core.intfunc.mod_inverse` function:

    >>> mod_inverse(3, 5)
    2
    >>> (S(2)/5).invert(S(7)/3)
    5/2

    See Also
    ========
    sympy.core.intfunc.mod_inverse

    """
    # 检查并允许特定的标志位
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 尝试并行将表达式f和g转化为多项式，并获取选项
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，则构造相关领域，并尝试执行逆操作
        domain, (a, b) = construct_domain(exc.exprs)

        try:
            return domain.to_sympy(domain.invert(a, b))
        except NotImplementedError:
            raise ComputationFailed('invert', 2, exc)

    # 对F在G模下的逆运算，根据选项自动执行
    h = F.invert(G, auto=opt.auto)

    if not opt.polys:
        return h.as_expr()
    else:
        return h


# 公共函数装饰器，使该函数可以在外部被访问
@public
# 定义函数subresultants，计算f和g的子结果多项式序列
def subresultants(f, g, *gens, **args):
    """
    Compute subresultant PRS of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy import subresultants
    >>> from sympy.abc import x

    >>> subresultants(x**2 + 1, x**2 - 1)
    [x**2 + 1, x**2 - 1, -2]

    """
    # 检查并允许特定的标志位
    options.allowed_flags(args, ['polys'])

    try:
        # 尝试并行将表达式f和g转化为多项式，并获取选项
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，则抛出计算失败异常
        raise ComputationFailed('subresultants', 2, exc)

    # 计算F和G的子结果多项式序列
    result = F.subresultants(G)

    if not opt.polys:
        return [r.as_expr() for r in result]
    else:
        return result


# 公共函数装饰器，使该函数可以在外部被访问
@public
# 定义函数resultant，计算f和g的结果多项式
def resultant(f, g, *gens, includePRS=False, **args):
    """
    Compute resultant of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy import resultant
    >>> from sympy.abc import x

    >>> resultant(x**2 + 1, x**2 - 1)
    4

    """
    # 检查并允许特定的标志位
    options.allowed_flags(args, ['polys'])

    try:
        # 尝试并行将表达式f和g转化为多项式，并获取选项
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果转化失败，则抛出计算失败异常
        raise ComputationFailed('resultant', 2, exc)

    if includePRS:
        # 如果包括子结果多项式序列，计算F和G的结果多项式及其子结果多项式序列
        result, R = F.resultant(G, includePRS=includePRS)
    else:
        # 否则，仅计算F和G的结果多项式
        result = F.resultant(G)

    if not opt.polys:
        # 如果不是多项式形式，将结果转换为表达式形式返回
        if includePRS:
            return result.as_expr(), [r.as_expr() for r in R]
        return result.as_expr()
    else:
        # 否则直接返回多项式形式的结果
        if includePRS:
            return result, R
        return result


# 公共函数装饰器，使该函数可以在外部被访问
@public
# 定义函数discriminant，计算f的判别式
def discriminant(f, *gens, **args):
    """
    Compute discriminant of ``f``.

    Examples
    ========

    >>> from sympy import discriminant
    >>> from sympy.abc import x

    >>> discriminant(x**2 + 2*x + 3)
    -8

    """
    # 检查并允许特定的标志位
    options.allowed_flags(args, ['polys'])

    try:
        # 尝试将表达式f转化为多项式，并获取选项
        F, opt = poly_from_expr(f, *gens, **args)
    # 如果捕获到 PolificationFailed 异常，则将其转换为 ComputationFailed 异常，并指明是因为计算 'discriminant' 时出错
    except PolificationFailed as exc:
        raise ComputationFailed('discriminant', 1, exc)

    # 调用 F 对象的 discriminant 方法计算结果
    result = F.discriminant()

    # 如果选项中没有指定 polys 参数，返回结果的表达式形式
    if not opt.polys:
        return result.as_expr()
    else:
        # 否则，直接返回计算结果
        return result
# 定义一个公共函数用于计算多项式 f 和 g 的最大公因数 (GCD) 及其余因子
@public
def cofactors(f, g, *gens, **args):
    """
    Compute GCD and cofactors of ``f`` and ``g``.

    Returns polynomials ``(h, cff, cfg)`` such that ``h = gcd(f, g)``, and
    ``cff = quo(f, h)`` and ``cfg = quo(g, h)`` are, so called, cofactors
    of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy import cofactors
    >>> from sympy.abc import x

    >>> cofactors(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    """
    # 检查是否允许给定的选项标志（目前仅支持 'polys'）
    options.allowed_flags(args, ['polys'])

    try:
        # 并行地从表达式中构建多项式 F 和 G
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        # 如果多项式构建失败，则处理异常并尝试另一种方法
        domain, (a, b) = construct_domain(exc.exprs)

        try:
            # 使用定义域的方法计算多项式 a 和 b 的因数
            h, cff, cfg = domain.cofactors(a, b)
        except NotImplementedError:
            # 如果计算方法未实现，则引发计算失败异常
            raise ComputationFailed('cofactors', 2, exc)
        else:
            # 将结果转换为 SymPy 多项式并返回
            return domain.to_sympy(h), domain.to_sympy(cff), domain.to_sympy(cfg)

    # 使用 F 对象的方法计算 G 的因数
    h, cff, cfg = F.cofactors(G)

    # 如果不需要保留多项式对象，则将结果转换为表达式形式并返回
    if not opt.polys:
        return h.as_expr(), cff.as_expr(), cfg.as_expr()
    else:
        return h, cff, cfg


# 定义一个公共函数用于计算多项式列表的最大公因数 (GCD)
@public
def gcd_list(seq, *gens, **args):
    """
    Compute GCD of a list of polynomials.

    Examples
    ========

    >>> from sympy import gcd_list
    >>> from sympy.abc import x

    >>> gcd_list([x**3 - 1, x**2 - 1, x**2 - 3*x + 2])
    x - 1

    """
    # 将输入的多项式列表转换为 Sympy 多项式对象
    seq = sympify(seq)

    # 尝试使用非多项式方法计算 GCD
    def try_non_polynomial_gcd(seq):
        if not gens and not args:
            # 构建定义域并尝试计算数值列表的 GCD
            domain, numbers = construct_domain(seq)

            if not numbers:
                return domain.zero
            elif domain.is_Numerical:
                result, numbers = numbers[0], numbers[1:]

                for number in numbers:
                    # 在定义域中计算多个数值的 GCD
                    result = domain.gcd(result, number)

                    if domain.is_one(result):
                        break

                # 将结果转换为 SymPy 表达式并返回
                return domain.to_sympy(result)

        return None

    # 尝试使用非多项式方法计算 GCD
    result = try_non_polynomial_gcd(seq)

    if result is not None:
        return result

    # 检查是否允许给定的选项标志（目前仅支持 'polys'）
    options.allowed_flags(args, ['polys'])

    try:
        # 并行地从表达式中构建多项式列表
        polys, opt = parallel_poly_from_expr(seq, *gens, **args)

        # 对于定义域为 Q[irrational] 的多项式，计算其 GCD
        if len(seq) > 1 and all(elt.is_algebraic and elt.is_irrational for elt in seq):
            a = seq[-1]
            lst = [ (a/elt).ratsimp() for elt in seq[:-1] ]
            if all(frc.is_rational for frc in lst):
                lc = 1
                for frc in lst:
                    lc = lcm(lc, frc.as_numer_denom()[0])
                # 取绝对值以确保 GCD 始终为非负数
                return abs(a/lc)

    except PolificationFailed as exc:
        # 如果多项式构建失败，则尝试使用非多项式方法计算 GCD
        result = try_non_polynomial_gcd(exc.exprs)

        if result is not None:
            return result
        else:
            # 如果仍然失败，则引发计算失败异常
            raise ComputationFailed('gcd_list', len(seq), exc)

    # 如果多项式列表为空，则返回零多项式或多项式对象
    if not polys:
        if not opt.polys:
            return S.Zero
        else:
            return Poly(0, opt=opt)

    # 返回计算结果的第一个多项式或多项式对象
    result, polys = polys[0], polys[1:]
    for poly in polys:
        # 对于每个多项式 poly 在 polys 列表中进行迭代
        result = result.gcd(poly)
        # 使用 result 对象的 gcd 方法计算 result 和当前 poly 的最大公约数，并更新 result
        
        if result.is_one:
            # 如果 result 是单位元（即最大公约数为1），则跳出循环
            break

    if not opt.polys:
        # 如果 opt.polys 为空（假值），返回 result 的表达式形式
        return result.as_expr()
    else:
        # 否则，返回 result 的值
        return result
# 定义一个公共函数 gcd，用于计算 f 和 g 的最大公约数
@public
def gcd(f, g=None, *gens, **args):
    """
    Compute GCD of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy import gcd
    >>> from sympy.abc import x

    >>> gcd(x**2 - 1, x**2 - 3*x + 2)
    x - 1

    """
    # 如果 f 是可迭代对象，则调用 gcd_list 处理多个输入的情况
    if hasattr(f, '__iter__'):
        # 如果同时提供了 g 参数，则将 g 加入生成器列表中
        if g is not None:
            gens = (g,) + gens
        return gcd_list(f, *gens, **args)
    # 如果 g 为 None，则抛出类型错误
    elif g is None:
        raise TypeError("gcd() takes 2 arguments or a sequence of arguments")

    # 检查并允许特定的参数标志
    options.allowed_flags(args, ['polys'])

    try:
        # 并行地将表达式转换为多项式
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)

        # 对于域 Q[irrational]（纯代数无理数）的 gcd 计算
        a, b = map(sympify, (f, g))
        if a.is_algebraic and a.is_irrational and b.is_algebraic and b.is_irrational:
            frc = (a/b).ratsimp()
            if frc.is_rational:
                # abs 确保返回的 gcd 总是非负的
                return abs(a/frc.as_numer_denom()[0])

    except PolificationFailed as exc:
        # 处理多项式化失败的异常
        domain, (a, b) = construct_domain(exc.exprs)

        try:
            return domain.to_sympy(domain.gcd(a, b))
        except NotImplementedError:
            raise ComputationFailed('gcd', 2, exc)

    # 返回多项式 F 和 G 的 gcd 结果
    result = F.gcd(G)

    # 如果不需要多项式对象，则将结果作为表达式返回
    if not opt.polys:
        return result.as_expr()
    else:
        return result


# 定义一个公共函数 lcm_list，用于计算多项式列表的最小公倍数
@public
def lcm_list(seq, *gens, **args):
    """
    Compute LCM of a list of polynomials.

    Examples
    ========

    >>> from sympy import lcm_list
    >>> from sympy.abc import x

    >>> lcm_list([x**3 - 1, x**2 - 1, x**2 - 3*x + 2])
    x**5 - x**4 - 2*x**3 - x**2 + x + 2

    """
    # 将 seq 转换为 Sympy 的表达式对象
    seq = sympify(seq)

    # 尝试非多项式 lcm 的计算
    def try_non_polynomial_lcm(seq) -> Optional[Expr]:
        if not gens and not args:
            # 构造域并获取其中的数字
            domain, numbers = construct_domain(seq)

            if not numbers:
                return domain.to_sympy(domain.one)
            elif domain.is_Numerical:
                result, numbers = numbers[0], numbers[1:]

                for number in numbers:
                    result = domain.lcm(result, number)

                return domain.to_sympy(result)

        return None

    # 尝试计算非多项式 lcm
    result = try_non_polynomial_lcm(seq)

    # 如果成功计算出结果，则返回
    if result is not None:
        return result

    # 检查并允许特定的参数标志
    options.allowed_flags(args, ['polys'])

    try:
        # 并行地将表达式转换为多项式
        polys, opt = parallel_poly_from_expr(seq, *gens, **args)

        # 对于域 Q[irrational]（纯代数无理数）的 lcm 计算
        if len(seq) > 1 and all(elt.is_algebraic and elt.is_irrational for elt in seq):
            a = seq[-1]
            lst = [ (a/elt).ratsimp() for elt in seq[:-1] ]
            if all(frc.is_rational for frc in lst):
                lc = 1
                for frc in lst:
                    lc = lcm(lc, frc.as_numer_denom()[1])
                return a*lc

    except PolificationFailed as exc:
        domain, (a, b) = construct_domain(exc.exprs)

        try:
            return domain.to_sympy(domain.gcd(a, b))
        except NotImplementedError:
            raise ComputationFailed('gcd', 2, exc)

    # 如果以上条件都不满足，则返回多项式列表的 lcm 结果
    return result
    # 如果捕获到 PolificationFailed 异常，则执行以下代码块
    except PolificationFailed as exc:
        # 尝试使用非多项式的最小公倍数函数来处理异常表达式
        result = try_non_polynomial_lcm(exc.exprs)

        # 如果成功得到结果，则返回该结果
        if result is not None:
            return result
        # 否则，抛出 ComputationFailed 异常，指示 lcm_list 计算失败
        else:
            raise ComputationFailed('lcm_list', len(seq), exc)

    # 如果没有多项式存在
    if not polys:
        # 如果选项中不包含多项式，则返回 1
        if not opt.polys:
            return S.One
        # 否则，返回一个多项式对象，表示 1，根据选项
        else:
            return Poly(1, opt=opt)

    # 从 polys 列表中取出第一个多项式作为结果的初始值，其余多项式放入 polys 列表中
    result, polys = polys[0], polys[1:]

    # 遍历剩余的 polys 列表中的多项式，并计算其与 result 的最小公倍数
    for poly in polys:
        result = result.lcm(poly)

    # 如果选项中不包含多项式，则将最终结果转换为表达式形式并返回
    if not opt.polys:
        return result.as_expr()
    # 否则，直接返回多项式形式的最终结果
    else:
        return result
# 定义一个公共函数 lcm，计算 f 和 g 的最小公倍数
@public
def lcm(f, g=None, *gens, **args):
    """
    Compute LCM of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy import lcm
    >>> from sympy.abc import x

    >>> lcm(x**2 - 1, x**2 - 3*x + 2)
    x**3 - 2*x**2 - x + 2

    """
    # 如果 f 是可迭代对象
    if hasattr(f, '__iter__'):
        # 如果同时提供了 g 参数，则将 g 加入到 gens 中
        if g is not None:
            gens = (g,) + gens

        # 调用 lcm_list 函数计算参数序列 f 和 gens 的最小公倍数
        return lcm_list(f, *gens, **args)
    # 如果 f 不是可迭代对象但 g 参数为 None，则抛出类型错误
    elif g is None:
        raise TypeError("lcm() takes 2 arguments or a sequence of arguments")

    # 检查并允许特定的 args 标志
    options.allowed_flags(args, ['polys'])

    try:
        # 从表达式 (f, g) 中并行构造多项式 F 和 G，以及选项 opt
        (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)

        # 对于定义域 Q[irrational]（纯代数无理数）的最小公倍数
        a, b = map(sympify, (f, g))
        if a.is_algebraic and a.is_irrational and b.is_algebraic and b.is_irrational:
            # 计算有理化的比值 frc
            frc = (a/b).ratsimp()
            if frc.is_rational:
                # 返回 a 乘以 frc 的分母，作为最小公倍数结果
                return a*frc.as_numer_denom()[1]

    except PolificationFailed as exc:
        # 构造 domain 和 (a, b) 对，用于处理 PolificationFailed 异常表达式
        domain, (a, b) = construct_domain(exc.exprs)

        try:
            # 调用 domain 的 to_sympy 方法计算 a 和 b 的最小公倍数，并返回结果
            return domain.to_sympy(domain.lcm(a, b))
        except NotImplementedError:
            # 如果方法未实现，则抛出计算失败异常
            raise ComputationFailed('lcm', 2, exc)

    # 计算 F 和 G 的多项式最小公倍数，并返回结果
    result = F.lcm(G)

    # 如果不需要 polys 模式，则作为表达式返回结果
    if not opt.polys:
        return result.as_expr()
    else:
        return result



# 定义一个公共函数 terms_gcd，从 f 中移除项的最大公因数
@public
def terms_gcd(f, *gens, **args):
    """
    Remove GCD of terms from ``f``.

    If the ``deep`` flag is True, then the arguments of ``f`` will have
    terms_gcd applied to them.

    If a fraction is factored out of ``f`` and ``f`` is an Add, then
    an unevaluated Mul will be returned so that automatic simplification
    does not redistribute it. The hint ``clear``, when set to False, can be
    used to prevent such factoring when all coefficients are not fractions.

    Examples
    ========

    >>> from sympy import terms_gcd, cos
    >>> from sympy.abc import x, y
    >>> terms_gcd(x**6*y**2 + x**3*y, x, y)
    x**3*y*(x**3*y + 1)

    The default action of polys routines is to expand the expression
    given to them. terms_gcd follows this behavior:

    >>> terms_gcd((3+3*x)*(x+x*y))
    3*x*(x*y + x + y + 1)

    If this is not desired then the hint ``expand`` can be set to False.
    In this case the expression will be treated as though it were comprised
    of one or more terms:

    >>> terms_gcd((3+3*x)*(x+x*y), expand=False)
    (3*x + 3)*(x*y + x)

    In order to traverse factors of a Mul or the arguments of other
    functions, the ``deep`` hint can be used:

    >>> terms_gcd((3 + 3*x)*(x + x*y), expand=False, deep=True)
    3*x*(x + 1)*(y + 1)
    >>> terms_gcd(cos(x + x*y), deep=True)
    cos(x*(y + 1))

    Rationals are factored out by default:

    >>> terms_gcd(x + y/2)
    (2*x + y)/2

    Only the y-term had a coefficient that was a fraction; if one
    does not want to factor out the 1/2 in cases like this, the
    flag ``clear`` can be set to False:

    >>> terms_gcd(x + y/2, clear=False)
    x + y/2
    >>> terms_gcd(x*y/2 + y**2, clear=False)
    y*(x/2 + y)

    """
    # 如果 deep 标志为 True，则对 f 的参数递归应用 terms_gcd
    if args.get('deep', False):
        pass  # 略过，不在此展示递归应用的具体逻辑

    # 如果 clear 标志为 False，则阻止将分数系数提取出来
    if not args.get('clear', True):
        pass  # 略过，不在此展示阻止分数系数提取的具体逻辑

    # 根据 args 中的标志执行多项式函数，默认会展开表达式
    options.allowed_flags(args, ['expand'])

    # 省略掉对表达式的展开细节

    # 返回从 f 中移除项的最大公因数后的结果
    return result
    # 如果所有系数都是分数，则忽略“clear”标志：
    # 示例：terms_gcd(x/3 + y/2, clear=False)
    # 结果为 (2*x + 3*y)/6
    
    # 参见
    # ========
    # sympy.core.exprtools.gcd_terms, sympy.core.exprtools.factor_terms
    
    """
    
    # 将输入的表达式转换为 SymPy 表达式对象
    orig = sympify(f)
    
    # 如果输入的表达式是等式类型，递归地对左右两边进行 terms_gcd 处理
    if isinstance(f, Equality):
        return Equality(*(terms_gcd(s, *gens, **args) for s in [f.lhs, f.rhs]))
    
    # 如果输入的表达式是关系型（不等式）类型，抛出 TypeError
    elif isinstance(f, Relational):
        raise TypeError("Inequalities cannot be used with terms_gcd. Found: %s" %(f,))
    
    # 如果输入的表达式不是 SymPy 的表达式对象或者是原子表达式，则直接返回原始输入
    if not isinstance(f, Expr) or f.is_Atom:
        return orig
    
    # 如果设置了深度处理标志，则深度处理表达式的每个参数
    if args.get('deep', False):
        new = f.func(*[terms_gcd(a, *gens, **args) for a in f.args])
        args.pop('deep')
        args['expand'] = False
        return terms_gcd(new, *gens, **args)
    
    # 获取并移除清除标志，默认为 True
    clear = args.pop('clear', True)
    
    # 校验并获取选项中的“polys”标志
    options.allowed_flags(args, ['polys'])
    
    # 尝试将表达式转换为多项式形式，获取多项式对象 F 和选项 opt
    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        return exc.expr
    
    # 计算 F 的最大公因子（terms_gcd）J 和 f
    J, f = F.terms_gcd()
    
    # 如果定义域是环（Ring）
    if opt.domain.is_Ring:
        # 如果定义域是域（Field），清除 f 的分母并将其转换为整数形式
        if opt.domain.is_Field:
            denom, f = f.clear_denoms(convert=True)
    
        # 计算 f 的原始形式并提取系数
        coeff, f = f.primitive()
    
        # 如果定义域是域（Field），将系数除以分母
        if opt.domain.is_Field:
            coeff /= denom
    else:
        coeff = S.One
    
    # 计算 term，即根据 f.gens 和 J 构造的多项式
    term = Mul(*[x**j for x, j in zip(f.gens, J)])
    
    # 如果系数 coeff 等于 1，则设置 coeff 为 SymPy 中的表示 1
    if equal_valued(coeff, 1):
        coeff = S.One
    
    # 如果清除标志为 True，则返回清除系数和 term*f 的结果
    if clear:
        return _keep_coeff(coeff, term*f.as_expr())
    
    # 否则基于原始表达式的形式而不是当前的 Mul 来进行清除
    coeff, f = _keep_coeff(coeff, f.as_expr(), clear=False).as_coeff_Mul()
    return _keep_coeff(coeff, term*f, clear=False)
# 定义一个公共函数，用于计算多项式 f 关于 p 取模的结果
@public
def trunc(f, p, *gens, **args):
    """
    Reduce ``f`` modulo a constant ``p``.

    Examples
    ========

    >>> from sympy import trunc
    >>> from sympy.abc import x

    >>> trunc(2*x**3 + 3*x**2 + 5*x + 7, 3)
    -x**3 - x + 1

    """
    # 检查并限制输入的选项参数
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 将表达式 f 转换为多项式 F，并获取选项信息
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，抛出计算失败异常
        raise ComputationFailed('trunc', 1, exc)

    # 对多项式 F 进行 p 取模运算
    result = F.trunc(sympify(p))

    # 如果不需要返回多项式形式，则返回表达式形式的结果
    if not opt.polys:
        return result.as_expr()
    else:
        return result


# 定义一个公共函数，用于将多项式 f 转换为首一形式
@public
def monic(f, *gens, **args):
    """
    Divide all coefficients of ``f`` by ``LC(f)``.

    Examples
    ========

    >>> from sympy import monic
    >>> from sympy.abc import x

    >>> monic(3*x**2 + 4*x + 2)
    x**2 + 4*x/3 + 2/3

    """
    # 检查并限制输入的选项参数
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 将表达式 f 转换为多项式 F，并获取选项信息
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，抛出计算失败异常
        raise ComputationFailed('monic', 1, exc)

    # 计算多项式 F 的首一形式
    result = F.monic(auto=opt.auto)

    # 如果不需要返回多项式形式，则返回表达式形式的结果
    if not opt.polys:
        return result.as_expr()
    else:
        return result


# 定义一个公共函数，计算多项式 f 的系数的最大公约数
@public
def content(f, *gens, **args):
    """
    Compute GCD of coefficients of ``f``.

    Examples
    ========

    >>> from sympy import content
    >>> from sympy.abc import x

    >>> content(6*x**2 + 8*x + 12)
    2

    """
    # 检查并限制输入的选项参数
    options.allowed_flags(args, ['polys'])

    try:
        # 将表达式 f 转换为多项式 F，并获取选项信息
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，抛出计算失败异常
        raise ComputationFailed('content', 1, exc)

    # 计算多项式 F 的系数的最大公约数
    return F.content()


# 定义一个公共函数，计算多项式 f 的内容和其原始形式
@public
def primitive(f, *gens, **args):
    """
    Compute content and the primitive form of ``f``.

    Examples
    ========

    >>> from sympy.polys.polytools import primitive
    >>> from sympy.abc import x

    >>> primitive(6*x**2 + 8*x + 12)
    (2, 3*x**2 + 4*x + 6)

    >>> eq = (2 + 2*x)*x + 2

    Expansion is performed by default:

    >>> primitive(eq)
    (2, x**2 + x + 1)

    Set ``expand`` to False to shut this off. Note that the
    extraction will not be recursive; use the as_content_primitive method
    for recursive, non-destructive Rational extraction.

    >>> primitive(eq, expand=False)
    (1, x*(2*x + 2) + 2)

    >>> eq.as_content_primitive()
    (2, x*(x + 1) + 1)

    """
    # 检查并限制输入的选项参数
    options.allowed_flags(args, ['polys'])

    try:
        # 将表达式 f 转换为多项式 F，并获取选项信息
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，抛出计算失败异常
        raise ComputationFailed('primitive', 1, exc)

    # 计算多项式 F 的内容和原始形式
    cont, result = F.primitive()
    # 如果不需要返回多项式形式，则返回表达式形式的结果
    if not opt.polys:
        return cont, result.as_expr()
    else:
        return cont, result


# 定义一个公共函数，计算函数复合 f(g)
@public
def compose(f, g, *gens, **args):
    """
    Compute functional composition ``f(g)``.

    Examples
    ========

    >>> from sympy import compose
    >>> from sympy.abc import x

    >>> compose(x**2 + x, x - 1)
    x**2 - x

    """
    # 检查并限制输入的选项参数
    options.allowed_flags(args, ['polys'])

    # 这里可以添加更多的功能实现
    # 尝试并行构造多项式 F 和 G，从表达式 f 和 g 中生成
    (F, G), opt = parallel_poly_from_expr((f, g), *gens, **args)

    # 将多项式 G 组合到多项式 F 中得到结果
    result = F.compose(G)

    # 如果选项 opt 中没有多项式，则返回结果的表达式形式
    if not opt.polys:
        return result.as_expr()
    else:
        # 否则直接返回结果
        return result
# 定义一个公共函数 `decompose`，用于计算函数 `f` 的功能分解
@public
def decompose(f, *gens, **args):
    """
    Compute functional decomposition of ``f``.

    Examples
    ========

    >>> from sympy import decompose
    >>> from sympy.abc import x

    >>> decompose(x**4 + 2*x**3 - x - 1)
    [x**2 - x - 1, x**2 + x]

    """
    # 根据传入的参数 `args`，确认只使用了允许的选项标志
    options.allowed_flags(args, ['polys'])

    try:
        # 将表达式 `f` 转换为多项式对象 `F`，并获取相关选项 `opt`
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，则抛出 `ComputationFailed` 异常
        raise ComputationFailed('decompose', 1, exc)

    # 对多项式 `F` 进行功能分解，返回结果 `result`
    result = F.decompose()

    # 如果没有指定 `polys` 标志，将结果转换为表达式返回；否则直接返回多项式对象
    if not opt.polys:
        return [r.as_expr() for r in result]
    else:
        return result


# 定义一个公共函数 `sturm`，用于计算函数 `f` 的斯特尔姆序列
@public
def sturm(f, *gens, **args):
    """
    Compute Sturm sequence of ``f``.

    Examples
    ========

    >>> from sympy import sturm
    >>> from sympy.abc import x

    >>> sturm(x**3 - 2*x**2 + x - 3)
    [x**3 - 2*x**2 + x - 3, 3*x**2 - 4*x + 1, 2*x/9 + 25/9, -2079/4]

    """
    # 根据传入的参数 `args`，确认只使用了允许的选项标志
    options.allowed_flags(args, ['auto', 'polys'])

    try:
        # 将表达式 `f` 转换为多项式对象 `F`，并获取相关选项 `opt`
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，则抛出 `ComputationFailed` 异常
        raise ComputationFailed('sturm', 1, exc)

    # 对多项式 `F` 计算斯特尔姆序列，使用 `auto` 参数控制是否自动计算
    result = F.sturm(auto=opt.auto)

    # 如果没有指定 `polys` 标志，将结果转换为表达式返回；否则直接返回多项式对象
    if not opt.polys:
        return [r.as_expr() for r in result]
    else:
        return result


# 定义一个公共函数 `gff_list`，用于计算函数 `f` 的最大阶乘因子列表
@public
def gff_list(f, *gens, **args):
    """
    Compute a list of greatest factorial factors of ``f``.

    Note that the input to ff() and rf() should be Poly instances to use the
    definitions here.

    Examples
    ========

    >>> from sympy import gff_list, ff, Poly
    >>> from sympy.abc import x

    >>> f = Poly(x**5 + 2*x**4 - x**3 - 2*x**2, x)

    >>> gff_list(f)
    [(Poly(x, x, domain='ZZ'), 1), (Poly(x + 2, x, domain='ZZ'), 4)]

    >>> (ff(Poly(x), 1)*ff(Poly(x + 2), 4)) == f
    True

    >>> f = Poly(x**12 + 6*x**11 - 11*x**10 - 56*x**9 + 220*x**8 + 208*x**7 - \
        1401*x**6 + 1090*x**5 + 2715*x**4 - 6720*x**3 - 1092*x**2 + 5040*x, x)

    >>> gff_list(f)
    [(Poly(x**3 + 7, x, domain='ZZ'), 2), (Poly(x**2 + 5*x, x, domain='ZZ'), 3)]

    >>> ff(Poly(x**3 + 7, x), 2)*ff(Poly(x**2 + 5*x, x), 3) == f
    True

    """
    # 根据传入的参数 `args`，确认只使用了允许的选项标志
    options.allowed_flags(args, ['polys'])

    try:
        # 将表达式 `f` 转换为多项式对象 `F`，并获取相关选项 `opt`
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，则抛出 `ComputationFailed` 异常
        raise ComputationFailed('gff_list', 1, exc)

    # 计算多项式 `F` 的最大阶乘因子列表 `factors`
    factors = F.gff_list()

    # 如果没有指定 `polys` 标志，将结果转换为表达式返回；否则直接返回多项式对象
    if not opt.polys:
        return [(g.as_expr(), k) for g, k in factors]
    else:
        return factors


# 定义一个公共函数 `gff`，用于计算函数 `f` 的最大阶乘因子分解
@public
def gff(f, *gens, **args):
    """Compute greatest factorial factorization of ``f``. """
    # 抛出未实现的异常，因为该函数未被实现
    raise NotImplementedError('symbolic falling factorial')


# 定义一个公共函数 `sqf_norm`，用于计算函数 `f` 的平方自由标准化
@public
def sqf_norm(f, *gens, **args):
    """
    Compute square-free norm of ``f``.

    Returns ``s``, ``f``, ``r``, such that ``g(x) = f(x-sa)`` and
    ``r(x) = Norm(g(x))`` is a square-free polynomial over ``K``,
    where ``a`` is the algebraic extension of the ground domain.

    Examples
    ========

    >>> from sympy import sqf_norm, sqrt
    >>> from sympy.abc import x
    """
    >>> sqf_norm(x**2 + 1, extension=[sqrt(3)])
    ([1], x**2 - 2*sqrt(3)*x + 4, x**4 - 4*x**2 + 16)

    """
    # 检查传入参数中是否包含有效的 'polys' 标志，确保参数合法性
    options.allowed_flags(args, ['polys')

    # 尝试将输入的表达式转换为多项式 F，并获取相应的选项
    try:
        F, opt = poly_from_expr(f, *gens, **args)
    # 如果转换失败，则抛出 ComputationFailed 异常，传递 PolificationFailed 作为原因
    except PolificationFailed as exc:
        raise ComputationFailed('sqf_norm', 1, exc)

    # 计算多项式 F 的平方因式范式 s, g, r
    s, g, r = F.sqf_norm()

    # 将 s 转换为整数表示
    s_expr = [Integer(si) for si in s]

    # 如果未指定 'polys' 标志，则返回结果作为表达式
    if not opt.polys:
        return s_expr, g.as_expr(), r.as_expr()
    # 否则返回多项式形式的 g 和 r
    else:
        return s_expr, g, r
# 定义一个公共函数，计算多项式 `f` 的平方因子
@public
def sqf_part(f, *gens, **args):
    """
    Compute square-free part of ``f``.

    Examples
    ========

    >>> from sympy import sqf_part
    >>> from sympy.abc import x

    >>> sqf_part(x**3 - 3*x - 2)
    x**2 - x - 2

    """
    # 检查传入的参数中是否包含 'polys' 标志，并进行相应处理
    options.allowed_flags(args, ['polys'])

    try:
        # 将表达式 `f` 转换为多项式对象 `F`，并获取相关选项 `opt`
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        # 如果多项式转换失败，抛出异常并捕获，报告计算失败
        raise ComputationFailed('sqf_part', 1, exc)

    # 计算多项式 `F` 的平方因子部分
    result = F.sqf_part()

    if not opt.polys:
        return result.as_expr()  # 如果不需要保留多项式形式，则返回表达式形式的结果
    else:
        return result  # 否则返回多项式形式的结果


def _sorted_factors(factors, method):
    """Sort a list of ``(expr, exp)`` pairs. """
    if method == 'sqf':
        # 如果方法为 'sqf'，定义排序关键字函数以优化平方自由因子排序
        def key(obj):
            poly, exp = obj
            rep = poly.rep.to_list()
            return (exp, len(rep), len(poly.gens), str(poly.domain), rep)
    else:
        # 对于其他方法，定义通用的排序关键字函数
        def key(obj):
            poly, exp = obj
            rep = poly.rep.to_list()
            return (len(rep), len(poly.gens), exp, str(poly.domain), rep)

    # 使用定义的关键字函数对因子列表进行排序并返回
    return sorted(factors, key=key)


def _factors_product(factors):
    """Multiply a list of ``(expr, exp)`` pairs. """
    # 将一组 ``(expr, exp)`` 对乘积化为表达式形式
    return Mul(*[f.as_expr()**k for f, k in factors])


def _symbolic_factor_list(expr, opt, method):
    """Helper function for :func:`_symbolic_factor`. """
    coeff, factors = S.One, []

    # 将表达式 `expr` 中的因子化为符号化的因子列表
    args = [i._eval_factor() if hasattr(i, '_eval_factor') else i
        for i in Mul.make_args(expr)]
    for arg in args:
        if arg.is_Number or (isinstance(arg, Expr) and pure_complex(arg)):
            coeff *= arg
            continue
        elif arg.is_Pow and arg.base != S.Exp1:
            base, exp = arg.args
            if base.is_Number and exp.is_Number:
                coeff *= arg
                continue
            if base.is_Number:
                factors.append((base, exp))
                continue
        else:
            base, exp = arg, S.One

        try:
            # 尝试将基本表达式 `base` 转换为多项式 `poly`
            poly, _ = _poly_from_expr(base, opt)
        except PolificationFailed as exc:
            # 如果转换失败，将异常表达式作为因子添加到列表中
            factors.append((exc.expr, exp))
        else:
            # 否则，调用多项式对象的特定方法计算其符号化的因子列表
            func = getattr(poly, method + '_list')

            _coeff, _factors = func()
            if _coeff is not S.One:
                if exp.is_Integer:
                    coeff *= _coeff**exp
                elif _coeff.is_positive:
                    factors.append((_coeff, exp))
                else:
                    _factors.append((_coeff, S.One))

            if exp is S.One:
                factors.extend(_factors)
            elif exp.is_integer:
                factors.extend([(f, k*exp) for f, k in _factors])
            else:
                other = []

                for f, k in _factors:
                    if f.as_expr().is_positive:
                        factors.append((f, k*exp))
                    else:
                        other.append((f, k))

                factors.append((_factors_product(other), exp))
    # 如果方法为 'sqf'，执行以下代码块
    if method == 'sqf':
        # 使用生成器推导式计算每个 (f, _) 中 _ 等于 k 的因子乘积，与 k 组成元组
        factors = [(reduce(mul, (f for f, _ in factors if _ == k)), k)
                   for k in {i for _, i in factors}]
    
    # 返回 coeff 和 factors 变量作为结果
    return coeff, factors
# 定义一个帮助函数用于符号因式分解表达式
def _symbolic_factor(expr, opt, method):
    """Helper function for :func:`_factor`. """
    # 如果表达式是 SymPy 的表达式对象
    if isinstance(expr, Expr):
        # 如果表达式有 '_eval_factor' 方法，则调用该方法求因式分解结果
        if hasattr(expr, '_eval_factor'):
            return expr._eval_factor()
        # 将表达式转化为一起的形式，并获取系数和因子的列表
        coeff, factors = _symbolic_factor_list(together(expr, fraction=opt['fraction']), opt, method)
        # 保留系数并将因子相乘得到最终结果
        return _keep_coeff(coeff, _factors_product(factors))
    # 如果表达式有 args 属性，则递归调用 _symbolic_factor 处理其每个参数
    elif hasattr(expr, 'args'):
        return expr.func(*[_symbolic_factor(arg, opt, method) for arg in expr.args])
    # 如果表达式是可迭代对象，则对其每个元素递归调用 _symbolic_factor 处理
    elif hasattr(expr, '__iter__'):
        return expr.__class__([_symbolic_factor(arg, opt, method) for arg in expr])
    # 如果表达式不属于上述任何情况，则直接返回其本身
    else:
        return expr


# 定义一个帮助函数用于处理一般的因式分解列表
def _generic_factor_list(expr, gens, args, method):
    """Helper function for :func:`sqf_list` and :func:`factor_list`. """
    # 检查并设置允许的选项标志
    options.allowed_flags(args, ['frac', 'polys'])
    # 构建选项对象
    opt = options.build_options(gens, args)

    # 将输入的表达式转化为 SymPy 的表达式对象
    expr = sympify(expr)

    # 如果表达式是 SymPy 的表达式或多项式对象
    if isinstance(expr, (Expr, Poly)):
        # 如果表达式是多项式对象
        if isinstance(expr, Poly):
            numer, denom = expr, 1
        else:
            # 将表达式整合为分子分母形式
            numer, denom = together(expr).as_numer_denom()

        # 对分子和分母分别进行符号因式分解列表处理
        cp, fp = _symbolic_factor_list(numer, opt, method)
        cq, fq = _symbolic_factor_list(denom, opt, method)

        # 如果存在分母的因子且不允许分数形式，则抛出多项式错误
        if fq and not opt.frac:
            raise PolynomialError("a polynomial expected, got %s" % expr)

        # 克隆选项对象并设置扩展为 True
        _opt = opt.clone({"expand": True})

        # 对每个因子列表进行处理，确保其为多项式对象
        for factors in (fp, fq):
            for i, (f, k) in enumerate(factors):
                if not f.is_Poly:
                    f, _ = _poly_from_expr(f, _opt)
                    factors[i] = (f, k)

        # 对因子列表按指定方法排序
        fp = _sorted_factors(fp, method)
        fq = _sorted_factors(fq, method)

        # 如果不要求返回多项式对象，则将因子列表中的多项式转化为表达式形式
        if not opt.polys:
            fp = [(f.as_expr(), k) for f, k in fp]
            fq = [(f.as_expr(), k) for f, k in fq]

        # 计算并返回系数和分子因子列表
        coeff = cp / cq

        # 如果不要求分数形式，则只返回系数和分子因子列表
        if not opt.frac:
            return coeff, fp
        else:
            # 否则返回系数、分子因子列表和分母因子列表
            return coeff, fp, fq
    # 如果表达式不是 SymPy 的表达式或多项式对象，则抛出多项式错误
    else:
        raise PolynomialError("a polynomial expected, got %s" % expr)


# 定义一个帮助函数用于一般的因式分解
def _generic_factor(expr, gens, args, method):
    """Helper function for :func:`sqf` and :func:`factor`. """
    # 弹出 'fraction' 参数，默认为 True
    fraction = args.pop('fraction', True)
    # 检查并设置允许的选项标志
    options.allowed_flags(args, [])
    # 构建选项对象
    opt = options.build_options(gens, args)
    # 将输入的表达式转化为 SymPy 的表达式对象
    opt['fraction'] = fraction
    return _symbolic_factor(sympify(expr), opt, method)


# 尝试将多项式转化为有理系数形式的函数
def to_rational_coeffs(f):
    """
    try to transform a polynomial to have rational coefficients

    try to find a transformation ``x = alpha*y``

    ``f(x) = lc*alpha**n * g(y)`` where ``g`` is a polynomial with
    rational coefficients, ``lc`` the leading coefficient.

    If this fails, try ``x = y + beta``
    ``f(x) = g(y)``

    Returns ``None`` if ``g`` not found;
    ``(lc, alpha, None, g)`` in case of rescaling
    ``(None, None, beta, g)`` in case of translation

    Notes
    =====

    Currently it transforms only polynomials without roots larger than 2.

    Examples
    ========

    >>> from sympy import sqrt, Poly, simplify
    >>> from sympy.polys.polytools import to_rational_coeffs

    """
    # 导入符号 x 作为 sympy 的符号变量
    from sympy.abc import x
    # 计算表达式 (x**2-1)*(x-2) 在 x 被赋值为 x*(1 + sqrt(2)) 时的值，创建多项式对象 p
    p = Poly(((x**2-1)*(x-2)).subs({x:x*(1 + sqrt(2))}), x, domain='EX')
    # 将多项式 p 转换为有理系数的形式，返回其中的首项系数 lc 和转换后的多项式 r
    lc, r, _, g = to_rational_coeffs(p)
    # 输出 lc 和 r 的值
    lc, r
    # 输出 g 的值，g 被转换为多项式对象 Poly 类型，其域为有理数域 QQ
    g
    # 计算 1/r 的简化形式，并赋值给 r1
    r1 = simplify(1/r)
    # 创建一个新的多项式对象，其表达式为 g 在 x 被赋值为 r1 时的值，验证其是否等于 p
    Poly(lc*r**3*(g.as_expr()).subs({x:x*r1}), x, domain='EX') == p
    # 输出结果为 True，验证成功
    True
    
    """
    from sympy.simplify.simplify import simplify
    
    # 尝试对多项式 f 进行重新缩放，使其系数变为有理数
    def _try_rescale(f, f1=None):
        """
        尝试将多项式 f 重新缩放为具有有理系数的多项式，通过变换 x -> alpha*x 实现。
        返回 alpha 和转换后的多项式 f；如果缩放成功，alpha 是缩放因子，f 是缩放后的多项式；否则返回 None。
        """
        if not len(f.gens) == 1 or not (f.gens[0]).is_Atom:
            return None, f
        n = f.degree()
        lc = f.LC()
        f1 = f1 or f.monic()
        coeffs = f1.all_coeffs()[1:]
        # 对系数进行简化
        coeffs = [simplify(coeffx) for coeffx in coeffs]
        if len(coeffs) > 1 and coeffs[-2]:
            rescale1_x = simplify(coeffs[-2]/coeffs[-1])
            coeffs1 = []
            for i in range(len(coeffs)):
                coeffx = simplify(coeffs[i]*rescale1_x**(i + 1))
                if not coeffx.is_rational:
                    break
                coeffs1.append(coeffx)
            else:
                rescale_x = simplify(1/rescale1_x)
                x = f.gens[0]
                v = [x**n]
                for i in range(1, n + 1):
                    v.append(coeffs1[i - 1]*x**(n - i))
                f = Add(*v)
                f = Poly(f)
                return lc, rescale_x, f
        return None
    
    # 尝试对多项式 f 进行平移变换，使其系数变为有理数
    def _try_translate(f, f1=None):
        """
        尝试将多项式 f 通过变换 x -> x + alpha 转换为具有有理系数的多项式。
        返回 alpha 和转换后的多项式 f；如果转换成功，alpha 是平移因子，f 是平移后的多项式；否则返回 None。
        """
        if not len(f.gens) == 1 or not (f.gens[0]).is_Atom:
            return None, f
        n = f.degree()
        f1 = f1 or f.monic()
        coeffs = f1.all_coeffs()[1:]
        c = simplify(coeffs[0])
        if c.is_Add and not c.is_rational:
            # 将常数项进行分解，取出其中的有理数部分
            rat, nonrat = sift(c.args,
                lambda z: z.is_rational is True, binary=True)
            alpha = -c.func(*nonrat)/n
            # 对多项式进行平移变换
            f2 = f1.shift(alpha)
            return alpha, f2
        return None
    # 定义一个内部函数 `_has_square_roots`，用于判断多项式 `p` 是否仅包含平方根而没有其他根
    def _has_square_roots(p):
        """
        Return True if ``f`` is a sum with square roots but no other root
        """
        # 获取多项式 `p` 的系数
        coeffs = p.coeffs()
        # 初始化是否存在平方根的标志为 False
        has_sq = False
        # 遍历多项式的系数
        for y in coeffs:
            # 将系数展开为加法操作数
            for x in Add.make_args(y):
                # 将每个加法操作数因式分解，并获取因式的系数信息
                f = Factors(x).factors
                # 获取所有因式中是数字且是有理数且大于等于2的因式的分母部分
                r = [wx.q for b, wx in f.items() if
                    b.is_number and wx.is_Rational and wx.q >= 2]
                # 如果没有找到满足条件的因式，则继续下一个操作数的处理
                if not r:
                    continue
                # 如果找到了最小的分母为2的因式，则表示存在平方根
                if min(r) == 2:
                    has_sq = True
                # 如果存在分母大于2的因式，则直接返回 False，表示不仅仅是平方根
                if max(r) > 2:
                    return False
        # 返回是否存在平方根的标志
        return has_sq

    # 如果多项式 `f` 的定义域是扩展实数域并且 `_has_square_roots(f)` 返回 True
    if f.get_domain().is_EX and _has_square_roots(f):
        # 将多项式 `f` 转化为首一多项式
        f1 = f.monic()
        # 尝试对多项式 `f` 和 `f1` 进行重新缩放操作
        r = _try_rescale(f, f1)
        # 如果成功得到结果，则返回结果的第一个、第二个元素和 `None`，以及结果的第三个元素
        if r:
            return r[0], r[1], None, r[2]
        else:
            # 否则尝试对多项式 `f` 和 `f1` 进行平移操作
            r = _try_translate(f, f1)
            # 如果成功得到结果，则返回 `None`、`None`、结果的第一个元素和第二个元素
            if r:
                return None, None, r[0], r[1]
    # 如果条件不满足，则返回 `None`
    return None
def _torational_factor_list(p, x):
    """
    helper function to factor polynomial using to_rational_coeffs

    Examples
    ========

    >>> from sympy.polys.polytools import _torational_factor_list
    >>> from sympy.abc import x
    >>> from sympy import sqrt, expand, Mul
    >>> p = expand(((x**2-1)*(x-2)).subs({x:x*(1 + sqrt(2))}))
    >>> factors = _torational_factor_list(p, x); factors
    (-2, [(-x*(1 + sqrt(2))/2 + 1, 1), (-x*(1 + sqrt(2)) - 1, 1), (-x*(1 + sqrt(2)) + 1, 1)])
    >>> expand(factors[0]*Mul(*[z[0] for z in factors[1]])) == p
    True
    >>> p = expand(((x**2-1)*(x-2)).subs({x:x + sqrt(2)}))
    >>> factors = _torational_factor_list(p, x); factors
    (1, [(x - 2 + sqrt(2), 1), (x - 1 + sqrt(2), 1), (x + 1 + sqrt(2), 1)])
    >>> expand(factors[0]*Mul(*[z[0] for z in factors[1]])) == p
    True

    """
    # 导入必要的模块
    from sympy.simplify.simplify import simplify
    # 将多项式 p 转换为 Poly 类型
    p1 = Poly(p, x, domain='EX')
    # 获取多项式的次数
    n = p1.degree()
    # 调用 to_rational_coeffs 函数进行有理系数化处理
    res = to_rational_coeffs(p1)
    # 如果转换结果为空，则返回 None
    if not res:
        return None
    # 解构转换结果
    lc, r, t, g = res
    # 对多项式 g 进行因式分解
    factors = factor_list(g.as_expr())
    # 如果 lc 不为零
    if lc:
        # 计算首项系数 c
        c = simplify(factors[0]*lc*r**n)
        # 计算 r 的倒数
        r1 = simplify(1/r)
        # 初始化空列表 a
        a = []
        # 遍历 factors[1] 中的每个元素
        for z in factors[1:][0]:
            # 将简化后的表达式添加到列表 a 中
            a.append((simplify(z[0].subs({x: x*r1})), z[1]))
    else:
        # 否则，直接取 factors 的首项作为 c
        c = factors[0]
        # 初始化空列表 a
        a = []
        # 遍历 factors[1] 中的每个元素
        for z in factors[1:][0]:
            # 将简化后的表达式添加到列表 a 中
            a.append((z[0].subs({x: x - t}), z[1]))
    # 返回结果元组 (c, a)
    return (c, a)


@public
def sqf_list(f, *gens, **args):
    """
    Compute a list of square-free factors of ``f``.

    Examples
    ========

    >>> from sympy import sqf_list
    >>> from sympy.abc import x

    >>> sqf_list(2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16)
    (2, [(x + 1, 2), (x + 2, 3)])

    """
    # 调用 _generic_factor_list 函数计算 f 的平方自由因子列表
    return _generic_factor_list(f, gens, args, method='sqf')


@public
def sqf(f, *gens, **args):
    """
    Compute square-free factorization of ``f``.

    Examples
    ========

    >>> from sympy import sqf
    >>> from sympy.abc import x

    >>> sqf(2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16)
    2*(x + 1)**2*(x + 2)**3

    """
    # 调用 _generic_factor 函数计算 f 的平方自由因子
    return _generic_factor(f, gens, args, method='sqf')


@public
def factor_list(f, *gens, **args):
    """
    Compute a list of irreducible factors of ``f``.

    Examples
    ========

    >>> from sympy import factor_list
    >>> from sympy.abc import x, y

    >>> factor_list(2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y)
    (2, [(x + y, 1), (x**2 + 1, 2)])

    """
    # 调用 _generic_factor_list 函数计算 f 的不可约因子列表
    return _generic_factor_list(f, gens, args, method='factor')


@public
def factor(f, *gens, deep=False, **args):
    """
    Compute the factorization of expression, ``f``, into irreducibles. (To
    factor an integer into primes, use ``factorint``.)

    There two modes implemented: symbolic and formal. If ``f`` is not an
    instance of :class:`Poly` and generators are not specified, then the
    former mode is used. Otherwise, the formal mode is used.

    In symbolic mode, :func:`factor` will traverse the expression tree and
    """
    # 在符号模式下，使用 _generic_factor 函数计算表达式 f 的因式分解
    return _generic_factor(f, gens, args, method='factor')
    """
    Factorizes an expression `f` into its components, handling large or symbolic exponents.
    If an instance of :class:`~.Add` is encountered, formal factorization is used.

    By default, factorization is over the rationals. Use options like `extension`, `modulus`, or `domain`
    to factor over other domains such as algebraic or finite fields.

    Examples
    ========

    >>> from sympy import factor, sqrt, exp
    >>> from sympy.abc import x, y

    >>> factor(2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y)
    2*(x + y)*(x**2 + 1)**2

    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, modulus=2)
    (x + 1)**2
    >>> factor(x**2 + 1, gaussian=True)
    (x - I)*(x + I)

    >>> factor(x**2 - 2, extension=sqrt(2))
    (x - sqrt(2))*(x + sqrt(2))

    >>> factor((x**2 - 1)/(x**2 + 4*x + 4))
    (x - 1)*(x + 1)/(x + 2)**2
    >>> factor((x**2 + 4*x + 4)**10000000*(x**2 + 1))
    (x + 2)**20000000*(x**2 + 1)

    By default, `factor` treats an expression as a whole:

    >>> eq = 2**(x**2 + 2*x + 1)
    >>> factor(eq)
    2**(x**2 + 2*x + 1)

    If the `deep` flag is True, subexpressions will be factored:

    >>> factor(eq, deep=True)
    2**((x + 1)**2)

    If the `fraction` flag is False, rational expressions will not be combined. Default is True.

    >>> factor(5*x + 3*exp(2 - 7*x), deep=True)
    (5*x*exp(7*x) + 3*exp(2))*exp(-7*x)
    >>> factor(5*x + 3*exp(2 - 7*x), deep=True, fraction=False)
    5*x + 3*exp(2)*exp(-7*x)

    See Also
    ========
    sympy.ntheory.factor_.factorint
    """

    # Convert `f` into a SymPy expression if not already
    f = sympify(f)

    # If deep flag is True, attempt to factor subexpressions recursively
    if deep:
        def _try_factor(expr):
            """
            Attempt to factor `expr`, preserving unchanged when unable to factor.
            """
            fac = factor(expr, *gens, **args)
            if fac.is_Mul or fac.is_Pow:
                return fac
            return expr
        
        # Apply `_try_factor` to `f` in a bottom-up manner
        f = bottom_up(f, _try_factor)
        
        # Clean up any subexpressions that may have been expanded during factoring
        partials = {}
        muladd = f.atoms(Mul, Add)
        for p in muladd:
            fac = factor(p, *gens, **args)
            if (fac.is_Mul or fac.is_Pow) and fac != p:
                partials[p] = fac
        
        # Replace subexpressions with their factored versions
        return f.xreplace(partials)

    # If deep flag is False, attempt generic factorization
    try:
        return _generic_factor(f, gens, args, method='factor')
    except PolynomialError:
        # Handle cases where `_generic_factor` fails, typically due to non-commutative expressions
        if not f.is_commutative:
            return factor_nc(f)
        else:
            raise
@public
def intervals(F, all=False, eps=None, inf=None, sup=None, strict=False, fast=False, sqf=False):
    """
    Compute isolating intervals for roots of ``f``.

    Examples
    ========

    >>> from sympy import intervals
    >>> from sympy.abc import x

    >>> intervals(x**2 - 3)
    [((-2, -1), 1), ((1, 2), 1)]
    >>> intervals(x**2 - 3, eps=1e-2)
    [((-26/15, -19/11), 1), ((19/11, 26/15), 1)]

    """
    # 如果 F 不可迭代，则尝试将其转换为多项式
    if not hasattr(F, '__iter__'):
        try:
            F = Poly(F)
        except GeneratorsNeeded:
            return []

        # 调用 Poly 类的 intervals 方法计算根的隔离区间并返回结果
        return F.intervals(all=all, eps=eps, inf=inf, sup=sup, fast=fast, sqf=sqf)
    else:
        # 对于可迭代的 F，使用并行处理多项式的函数获取多项式列表和选项
        polys, opt = parallel_poly_from_expr(F, domain='QQ')

        # 如果多项式的变量数大于 1，抛出多变量多项式错误
        if len(opt.gens) > 1:
            raise MultivariatePolynomialError

        # 将每个多项式转换为系数列表表示
        for i, poly in enumerate(polys):
            polys[i] = poly.rep.to_list()

        # 如果指定了 eps，将其转换为 opt 域的有理数
        if eps is not None:
            eps = opt.domain.convert(eps)

            # 如果 eps 小于等于 0，则引发值错误异常
            if eps <= 0:
                raise ValueError("'eps' must be a positive rational")

        # 如果指定了 inf，则将其转换为 opt 域的值
        if inf is not None:
            inf = opt.domain.convert(inf)
        # 如果指定了 sup，则将其转换为 opt 域的值
        if sup is not None:
            sup = opt.domain.convert(sup)

        # 调用 dup_isolate_real_roots_list 函数计算多项式的实根的隔离区间
        intervals = dup_isolate_real_roots_list(polys, opt.domain,
            eps=eps, inf=inf, sup=sup, strict=strict, fast=fast)

        # 将区间转换为 sympy 中的表示，并构建最终结果列表
        result = []

        for (s, t), indices in intervals:
            s, t = opt.domain.to_sympy(s), opt.domain.to_sympy(t)
            result.append(((s, t), indices))

        return result


@public
def refine_root(f, s, t, eps=None, steps=None, fast=False, check_sqf=False):
    """
    Refine an isolating interval of a root to the given precision.

    Examples
    ========

    >>> from sympy import refine_root
    >>> from sympy.abc import x

    >>> refine_root(x**2 - 3, 1, 2, eps=1e-2)
    (19/11, 26/15)

    """
    try:
        # 尝试将 f 转换为 Poly 类，并检查生成器是否为符号变量
        F = Poly(f)
        if not isinstance(f, Poly) and not F.gen.is_Symbol:
            # 如果传递的表达式不是 Poly 类且生成器不是符号变量，则引发多项式错误异常
            raise PolynomialError("generator must be a Symbol")
    except GeneratorsNeeded:
        # 如果无法转换 f 为多项式，则引发多项式错误异常
        raise PolynomialError(
            "Cannot refine a root of %s, not a polynomial" % f)

    # 调用 Poly 类的 refine_root 方法，精确化根的隔离区间并返回结果
    return F.refine_root(s, t, eps=eps, steps=steps, fast=fast, check_sqf=check_sqf)


@public
def count_roots(f, inf=None, sup=None):
    """
    Return the number of roots of ``f`` in ``[inf, sup]`` interval.

    If one of ``inf`` or ``sup`` is complex, it will return the number of roots
    in the complex rectangle with corners at ``inf`` and ``sup``.

    Examples
    ========

    >>> from sympy import count_roots, I
    >>> from sympy.abc import x

    >>> count_roots(x**4 - 4, -3, 3)
    2
    >>> count_roots(x**4 - 4, 0, 1 + 3*I)
    1

    """
    # 尝试将 f 转换为多项式对象 Poly，禁用贪婪模式
    F = Poly(f, greedy=False)
    # 如果 f 不是 Poly 类型，并且生成器不是符号类型
    if not isinstance(f, Poly) and not F.gen.is_Symbol:
        # 抛出多项式错误，要求生成器必须是符号类型
        raise PolynomialError("generator must be a Symbol")
    except GeneratorsNeeded:
        # 如果 GeneratorsNeeded 异常被抛出，说明 f 不是多项式
        raise PolynomialError("Cannot count roots of %s, not a polynomial" % f)

    # 返回多项式对象 F 在区间 [inf, sup] 内的根数
    return F.count_roots(inf=inf, sup=sup)
# 定义一个公共函数，用于计算给定多项式的所有实数和复数根，包括它们的重数。
@public
def all_roots(f, multiple=True, radicals=True):
    """
    Returns the real and complex roots of ``f`` with multiplicities.

    Explanation
    ===========
    
    Finds all real and complex roots of a univariate polynomial with rational
    coefficients of any degree exactly. The roots are represented in the form
    given by :func:`~.rootof`. This is equivalent to using :func:`~.rootof` to
    find each of the indexed roots.

    Examples
    ========
    
    >>> from sympy import all_roots
    >>> from sympy.abc import x, y

    >>> print(all_roots(x**3 + 1))
    [-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2]

    Simple radical formulae are used in some cases but the cubic and quartic
    formulae are avoided. Instead most non-rational roots will be represented
    as :class:`~.ComplexRootOf`:

    >>> print(all_roots(x**3 + x + 1))
    [CRootOf(x**3 + x + 1, 0), CRootOf(x**3 + x + 1, 1), CRootOf(x**3 + x + 1, 2)]

    All roots of any polynomial with rational coefficients of any degree can be
    represented using :py:class:`~.ComplexRootOf`. The use of
    :py:class:`~.ComplexRootOf` bypasses limitations on the availability of
    radical formulae for quintic and higher degree polynomials _[1]:

    >>> p = x**5 - x - 1
    >>> for r in all_roots(p): print(r)
    CRootOf(x**5 - x - 1, 0)
    CRootOf(x**5 - x - 1, 1)
    CRootOf(x**5 - x - 1, 2)
    CRootOf(x**5 - x - 1, 3)
    CRootOf(x**5 - x - 1, 4)
    >>> [r.evalf(3) for r in all_roots(p)]
    [1.17, -0.765 - 0.352*I, -0.765 + 0.352*I, 0.181 - 1.08*I, 0.181 + 1.08*I]

    Irrational algebraic or transcendental coefficients cannot currently be
    handled by :func:`all_roots` (or :func:`~.rootof` more generally):

    >>> from sympy import sqrt, expand
    >>> p = expand((x - sqrt(2))*(x - sqrt(3)))
    >>> print(p)
    x**2 - sqrt(3)*x - sqrt(2)*x + sqrt(6)
    >>> all_roots(p)
    Traceback (most recent call last):
    ...
    NotImplementedError: sorted roots not supported over EX

    In the case of algebraic or transcendental coefficients
    :func:`~.ground_roots` might be able to find some roots by factorisation:

    >>> from sympy import ground_roots
    >>> ground_roots(p, x, extension=True)
    {sqrt(2): 1, sqrt(3): 1}

    If the coefficients are numeric then :func:`~.nroots` can be used to find
    all roots approximately:

    >>> from sympy import nroots
    >>> nroots(p, 5)
    [1.4142, 1.732]

    If the coefficients are symbolic then :func:`sympy.polys.polyroots.roots`
    or :func:`~.ground_roots` should be used instead:

    >>> from sympy import roots, ground_roots
    >>> p = x**2 - 3*x*y + 2*y**2
    >>> roots(p, x)
    {y: 1, 2*y: 1}
    >>> ground_roots(p, x)
    {y: 1, 2*y: 1}

    Parameters
    ==========

    f : :class:`~.Expr` or :class:`~.Poly`
        A univariate polynomial with rational (or ``Float``) coefficients.
    multiple : ``bool`` (default ``True``).
        Whether to return a ``list`` of roots or a list of root/multiplicity
        pairs.
    radicals : ``bool`` (default ``True``)
        Whether to include radicals in the root representation.

    """
    # radicals 参数，指定是否使用简单的根式公式而不是 ComplexRootOf 来表示一些无理根
    radicals : ``bool`` (default ``True``)
        Use simple radical formulae rather than :py:class:`~.ComplexRootOf`
        for some irrational roots.

    # 返回值说明
    Returns
    =======

    # 返回一个 :class:`~.Expr` 类型的列表，通常是 :class:`~.ComplexRootOf`，表示多项式 `f` 的根，
    # 每个根根据其重数重复出现。根总是按唯一顺序排列，实根排在复根之前，实根按增序排列，
    # 复根先按实部增序排列，然后按虚部增序排列。
    A list of :class:`~.Expr` (usually :class:`~.ComplexRootOf`) representing
    the roots is returned with each root repeated according to its multiplicity
    as a root of ``f``. The roots are always uniquely ordered with real roots
    coming before complex roots. The real roots are in increasing order.
    Complex roots are ordered by increasing real part and then increasing
    imaginary part.

    # 如果 multiple=False，则返回一个根/重数对的列表。
    If ``multiple=False`` is passed then a list of root/multiplicity pairs is
    returned instead.

    # 如果 radicals=False，则所有根都将以有理数或 :class:`~.ComplexRootOf` 表示。
    If ``radicals=False`` is passed then all roots will be represented as
    either rational numbers or :class:`~.ComplexRootOf`.

    # 查看也可以参考以下内容
    See also
    ========

    # Poly.all_roots 方法：被 :func:`~.all_roots` 使用的 :class:`Poly` 的底层方法。
    Poly.all_roots:
        The underlying :class:`Poly` method used by :func:`~.all_roots`.
    
    # rootof 函数：计算单个多项式的编号根。
    rootof:
        Compute a single numbered root of a univariate polynomial.
    
    # real_roots 函数：使用 :func:`~.rootof` 计算所有实根。
    real_roots:
        Compute all the real roots using :func:`~.rootof`.
    
    # ground_roots 函数：通过因式分解在地域域中计算一些根。
    ground_roots:
        Compute some roots in the ground domain by factorisation.
    
    # nroots 函数：使用近似数值技术计算所有根。
    nroots:
        Compute all roots using approximate numerical techniques.
    
    # sympy.polys.polyroots.roots：使用根式公式计算根的符号表达式。
    sympy.polys.polyroots.roots:
        Compute symbolic expressions for roots using radical formulae.

    # 参考文献
    References
    ==========

    # 阿贝尔－鲁菲尼定理的参考链接
    .. [1] https://en.wikipedia.org/wiki/Abel%E2%80%93Ruffini_theorem
    """
    # 尝试将 f 转换为 Poly 类型的对象 F，如果失败则抛出异常
    try:
        F = Poly(f, greedy=False)
        # 如果 f 不是 Poly 类型且 F.gen 不是 Symbol 类型，则抛出多项式错误
        if not isinstance(f, Poly) and not F.gen.is_Symbol:
            raise PolynomialError("generator must be a Symbol")
    except GeneratorsNeeded:
        # 如果需要生成器则抛出多项式错误
        raise PolynomialError(
            "Cannot compute real roots of %s, not a polynomial" % f)

    # 调用 F.all_roots 方法计算根，传入参数 multiple 和 radicals
    return F.all_roots(multiple=multiple, radicals=radicals)
# 声明一个公共函数，计算多项式 f 的实根，并可选择返回多重根和根式形式
@public
def real_roots(f, multiple=True, radicals=True):
    """
    Returns the real roots of ``f`` with multiplicities.

    Explanation
    ===========

    Finds all real roots of a univariate polynomial with rational coefficients
    of any degree exactly. The roots are represented in the form given by
    :func:`~.rootof`. This is equivalent to using :func:`~.rootof` or
    :func:`~.all_roots` and filtering out only the real roots. However if only
    the real roots are needed then :func:`real_roots` is more efficient than
    :func:`~.all_roots` because it computes only the real roots and avoids
    costly complex root isolation routines.

    Examples
    ========

    >>> from sympy import real_roots
    >>> from sympy.abc import x, y

    >>> real_roots(2*x**3 - 7*x**2 + 4*x + 4)
    [-1/2, 2, 2]
    >>> real_roots(2*x**3 - 7*x**2 + 4*x + 4, multiple=False)
    [(-1/2, 1), (2, 2)]

    Real roots of any polynomial with rational coefficients of any degree can
    be represented using :py:class:`~.ComplexRootOf`:

    >>> p = x**9 + 2*x + 2
    >>> print(real_roots(p))
    [CRootOf(x**9 + 2*x + 2, 0)]
    >>> [r.evalf(3) for r in real_roots(p)]
    [-0.865]

    All rational roots will be returned as rational numbers. Roots of some
    simple factors will be expressed using radical or other formulae (unless
    ``radicals=False`` is passed). All other roots will be expressed as
    :class:`~.ComplexRootOf`.

    >>> p = (x + 7)*(x**2 - 2)*(x**3 + x + 1)
    >>> print(real_roots(p))
    [-7, -sqrt(2), CRootOf(x**3 + x + 1, 0), sqrt(2)]
    >>> print(real_roots(p, radicals=False))
    [-7, CRootOf(x**2 - 2, 0), CRootOf(x**3 + x + 1, 0), CRootOf(x**2 - 2, 1)]

    All returned root expressions will numerically evaluate to real numbers
    with no imaginary part. This is in contrast to the expressions generated by
    the cubic or quartic formulae as used by :func:`~.roots` which suffer from
    casus irreducibilis [1]_:

    >>> from sympy import roots
    >>> p = 2*x**3 - 9*x**2 - 6*x + 3
    >>> [r.evalf(5) for r in roots(p, multiple=True)]
    [5.0365 - 0.e-11*I, 0.33984 + 0.e-13*I, -0.87636 + 0.e-10*I]
    >>> [r.evalf(5) for r in real_roots(p, x)]
    [-0.87636, 0.33984, 5.0365]
    >>> [r.is_real for r in roots(p, multiple=True)]
    [None, None, None]
    >>> [r.is_real for r in real_roots(p)]
    [True, True, True]

    Using :func:`real_roots` is equivalent to using :func:`~.all_roots` (or
    :func:`~.rootof`) and filtering out only the real roots:

    >>> from sympy import all_roots
    >>> r = [r for r in all_roots(p) if r.is_real]
    >>> real_roots(p) == r
    True

    If only the real roots are wanted then using :func:`real_roots` is faster
    than using :func:`~.all_roots`. Using :func:`real_roots` avoids complex root
    isolation which can be a lot slower than real root isolation especially for
    polynomials of high degree which typically have many more complex roots
    than real roots.
    """
    # 实数根不能处理含有无理代数或超越系数的多项式，例如使用 :func:`real_roots`（或更一般的 :func:`~.rootof`）：

    >>> from sympy import sqrt, expand
    >>> p = expand((x - sqrt(2))*(x - sqrt(3)))
    >>> print(p)
    x**2 - sqrt(3)*x - sqrt(2)*x + sqrt(6)
    >>> real_roots(p)
    Traceback (most recent call last):
    ...
    NotImplementedError: sorted roots not supported over EX

    # 对于含有代数或超越系数的情况，可能可以通过因式分解使用 :func:`~.ground_roots` 找到一些根：

    >>> from sympy import ground_roots
    >>> ground_roots(p, x, extension=True)
    {sqrt(2): 1, sqrt(3): 1}

    # 如果系数是数值型的，则可以使用 :func:`~.nroots` 来近似找到所有根：

    >>> from sympy import nroots
    >>> nroots(p, 5)
    [1.4142, 1.732]

    # 如果系数是符号型的，则应该使用 :func:`sympy.polys.polyroots.roots` 或 :func:`~.ground_roots`。

    >>> from sympy import roots, ground_roots
    >>> p = x**2 - 3*x*y + 2*y**2
    >>> roots(p, x)
    {y: 1, 2*y: 1}
    >>> ground_roots(p, x)
    {y: 1, 2*y: 1}

    """
    # 尝试将输入的表达式 f 转换为多项式对象 F，禁用贪婪模式
    try:
        F = Poly(f, greedy=False)
        # 如果 f 不是多项式对象并且 F 的生成器不是符号（Symbol）
        if not isinstance(f, Poly) and not F.gen.is_Symbol:
            # 抛出多项式错误，要求生成器必须是符号（Symbol）
            raise PolynomialError("generator must be a Symbol")
    # 如果 GeneratorsNeeded 异常被抛出
    except GeneratorsNeeded:
        # 抛出多项式错误，说明无法计算实根，因为 f 不是多项式
        raise PolynomialError(
            "Cannot compute real roots of %s, not a polynomial" % f)

    # 返回多项式对象 F 的实数根，支持多重根和使用根式（radicals）
    return F.real_roots(multiple=multiple, radicals=radicals)
# 定义一个公共函数，计算多项式 `f` 的数值近似根
@public
def nroots(f, n=15, maxsteps=50, cleanup=True):
    """
    Compute numerical approximations of roots of ``f``.

    Examples
    ========

    >>> from sympy import nroots
    >>> from sympy.abc import x

    >>> nroots(x**2 - 3, n=15)
    [-1.73205080756888, 1.73205080756888]
    >>> nroots(x**2 - 3, n=30)
    [-1.73205080756887729352744634151, 1.73205080756887729352744634151]

    """
    try:
        # 尝试将输入的表达式 `f` 转换为多项式对象 `F`
        F = Poly(f, greedy=False)
        # 如果 `f` 不是多项式对象且其生成器不是符号类型，则引发异常
        if not isinstance(f, Poly) and not F.gen.is_Symbol:
            # root of sin(x) + 1 is -1 but when someone
            # passes an Expr instead of Poly they may not expect
            # that the generator will be sin(x), not x
            raise PolynomialError("generator must be a Symbol")
    except GeneratorsNeeded:
        # 如果无法生成多项式对象，则引发多项式错误
        raise PolynomialError(
            "Cannot compute numerical roots of %s, not a polynomial" % f)

    # 调用多项式对象 `F` 的 `nroots` 方法计算数值近似根，并返回结果
    return F.nroots(n=n, maxsteps=maxsteps, cleanup=cleanup)


# 定义一个公共函数，通过在基域中进行因式分解来计算多项式 `f` 的根
@public
def ground_roots(f, *gens, **args):
    """
    Compute roots of ``f`` by factorization in the ground domain.

    Examples
    ========

    >>> from sympy import ground_roots
    >>> from sympy.abc import x

    >>> ground_roots(x**6 - 4*x**4 + 4*x**3 - x**2)
    {0: 2, 1: 2}

    """
    # 检查是否有合法的选项标志，这里不允许任何标志参数
    options.allowed_flags(args, [])

    try:
        # 尝试将输入的表达式 `f` 转换为多项式对象 `F`，并获取相关选项
        F, opt = poly_from_expr(f, *gens, **args)
        # 如果 `f` 不是多项式对象且其生成器不是符号类型，则引发异常
        if not isinstance(f, Poly) and not F.gen.is_Symbol:
            # root of sin(x) + 1 is -1 but when someone
            # passes an Expr instead of Poly they may not expect
            # that the generator will be sin(x), not x
            raise PolynomialError("generator must be a Symbol")
    except PolificationFailed as exc:
        # 如果无法将表达式转换为多项式，则引发计算失败异常
        raise ComputationFailed('ground_roots', 1, exc)

    # 调用多项式对象 `F` 的 `ground_roots` 方法计算根，并返回结果
    return F.ground_roots()


# 定义一个公共函数，构造一个多项式，其根的 n 次幂是多项式 `f` 的根的 n 次幂
@public
def nth_power_roots_poly(f, n, *gens, **args):
    """
    Construct a polynomial with n-th powers of roots of ``f``.

    Examples
    ========

    >>> from sympy import nth_power_roots_poly, factor, roots
    >>> from sympy.abc import x

    >>> f = x**4 - x**2 + 1
    >>> g = factor(nth_power_roots_poly(f, 2))

    >>> g
    (x**2 - x + 1)**2

    >>> R_f = [ (r**2).expand() for r in roots(f) ]
    >>> R_g = roots(g).keys()

    >>> set(R_f) == set(R_g)
    True

    """
    # 检查是否有合法的选项标志，这里不允许任何标志参数
    options.allowed_flags(args, [])

    try:
        # 尝试将输入的表达式 `f` 转换为多项式对象 `F`，并获取相关选项
        F, opt = poly_from_expr(f, *gens, **args)
        # 如果 `f` 不是多项式对象且其生成器不是符号类型，则引发异常
        if not isinstance(f, Poly) and not F.gen.is_Symbol:
            # root of sin(x) + 1 is -1 but when someone
            # passes an Expr instead of Poly they may not expect
            # that the generator will be sin(x), not x
            raise PolynomialError("generator must be a Symbol")
    except PolificationFailed as exc:
        # 如果无法将表达式转换为多项式，则引发计算失败异常
        raise ComputationFailed('nth_power_roots_poly', 1, exc)

    # 调用多项式对象 `F` 的 `nth_power_roots_poly` 方法构造 n 次幂根的多项式，并返回结果
    result = F.nth_power_roots_poly(n)

    # 如果选项中没有 `polys` 标志，则将结果作为表达式返回，否则作为多项式返回
    if not opt.polys:
        return result.as_expr()
    else:
        return result


# 定义一个公共函数，消除有理函数 `f` 中的公共因子
@public
def cancel(f, *gens, _signsimp=True, **args):
    """
    Cancel common factors in a rational function ``f``.

    """
    from sympy.simplify.simplify import signsimp  # 导入符号简化模块的符号简化函数
    from sympy.polys.rings import sring  # 导入多项式环模块的sring函数
    options.allowed_flags(args, ['polys'])  # 检查并允许args参数中的'polys'标志

    f = sympify(f)  # 将输入的表达式f转换为Sympy表达式对象
    if _signsimp:  # 如果_signsimp为真
        f = signsimp(f)  # 对表达式f进行符号简化

    opt = {}  # 初始化一个空的选项字典
    if 'polys' in args:  # 如果args参数中包含'polys'
        opt['polys'] = args['polys']  # 将'polys'选项设置为args中对应的值

    if not isinstance(f, (tuple, Tuple)):  # 如果f不是元组或Tuple类型
        if f.is_Number or isinstance(f, Relational) or not isinstance(f, Expr):  # 如果f是数值、关系表达式或者不是表达式类型
            return f  # 直接返回f
        f = factor_terms(f, radical=True)  # 对表达式f进行因子项的处理，包括根式项的因子化
        p, q = f.as_numer_denom()  # 将f转换为分子p和分母q

    elif len(f) == 2:  # 如果f是长度为2的元组
        p, q = f  # 将f分解为分子p和分母q
        if isinstance(p, Poly) and isinstance(q, Poly):  # 如果p和q都是多项式类型
            opt['gens'] = p.gens  # 设置'gens'选项为p的生成器
            opt['domain'] = p.domain  # 设置'domain'选项为p的定义域
            opt['polys'] = opt.get('polys', True)  # 设置'polys'选项，默认为True
        p, q = p.as_expr(), q.as_expr()  # 将p和q转换为表达式

    elif isinstance(f, Tuple):  # 如果f是Tuple类型
        return factor_terms(f)  # 对f进行因子项处理

    else:  # 否则
        raise ValueError('unexpected argument: %s' % f)  # 抛出值错误，表示未预期的参数类型：%s

    from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数模块的Piecewise类
    try:
        if f.has(Piecewise):  # 如果表达式f包含Piecewise函数
            raise PolynomialError()  # 抛出多项式错误
        R, (F, G) = sring((p, q), *gens, **args)  # 使用sring函数处理(p, q)，获取返回的环R和元组(F, G)
        if not R.ngens:  # 如果R中没有生成器
            if not isinstance(f, (tuple, Tuple)):  # 如果f不是元组或Tuple类型
                return f.expand()  # 返回表达式f的展开形式
            else:  # 否则
                return S.One, p, q  # 返回S.One, p, q

    except PolynomialError as msg:  # 捕获多项式错误，并将错误信息存储为msg
        if f.is_commutative and not f.has(Piecewise):  # 如果f是可交换的，并且不包含Piecewise函数
            raise PolynomialError(msg)  # 抛出多项式错误，使用msg作为错误信息
        # 处理非可交换和/或分段表达式
        if f.is_Add or f.is_Mul:  # 如果f是加法或乘法表达式
            c, nc = sift(f.args, lambda x:
                x.is_commutative is True and not x.has(Piecewise),
                binary=True)  # 使用sift函数将f的参数按条件分组成可交换项c和非可交换项nc
            nc = [cancel(i) for i in nc]  # 对nc中的每一项应用cancel函数进行化简
            return f.func(cancel(f.func(*c)), *nc)  # 返回f的函数形式，对可交换项和非可交换项应用cancel函数
        else:  # 否则
            reps = []  # 初始化一个替换列表
            pot = preorder_traversal(f)  # 对f进行前序遍历
            next(pot)  # 移动到下一个节点
            for e in pot:  # 遍历pot中的每个元素e
                # XXX: This should really skip anything that's not Expr.
                if isinstance(e, (tuple, Tuple, BooleanAtom)):  # 如果e是元组、Tuple或BooleanAtom类型
                    continue  # 继续下一个循环
                try:
                    reps.append((e, cancel(e)))  # 尝试将(e, cancel(e))添加到替换列表中
                    pot.skip()  # 跳过处理成功的节点
                except NotImplementedError:
                    pass  # 如果抛出未实现错误，则忽略该异常
            return f.xreplace(dict(reps))  # 返回使用替换列表替换f后的表达式

    c, (P, Q) = 1, F.cancel(G)  # 对F.cancel(G)的结果进行分解，得到c和(P, Q)
    if opt.get('polys', False) and 'gens' not in opt:  # 如果'polys'选项为真，并且'gens'不在选项中
        opt['gens'] = R.symbols  # 设置'gens'选项为环R中的符号

    if not isinstance(f, (tuple, Tuple)):  # 如果f不是元组或Tuple类型
        return c*(P.as_expr()/Q.as_expr())  # 返回c乘以P除以Q的表达式形式
    else:
        # 将 P 和 Q 转换为表达式对象
        P, Q = P.as_expr(), Q.as_expr()
        # 如果 opt 字典中未设置 'polys' 键或者其值为 False，则直接返回 c, P, Q
        if not opt.get('polys', False):
            return c, P, Q
        else:
            # 否则，将 P 和 Q 转换为多项式对象，并使用 gens 和 opt 作为参数
            return c, Poly(P, *gens, **opt), Poly(Q, *gens, **opt)
@public
def reduced(f, G, *gens, **args):
    """
    Reduces a polynomial ``f`` modulo a set of polynomials ``G``.

    Given a polynomial ``f`` and a set of polynomials ``G = (g_1, ..., g_n)``,
    computes a set of quotients ``q = (q_1, ..., q_n)`` and the remainder ``r``
    such that ``f = q_1*g_1 + ... + q_n*g_n + r``, where ``r`` vanishes or ``r``
    is a completely reduced polynomial with respect to ``G``.

    Examples
    ========

    >>> from sympy import reduced
    >>> from sympy.abc import x, y

    >>> reduced(2*x**4 + y**2 - x**2 + y**3, [x**3 - x, y**3 - y])
    ([2*x, 1], x**2 + y**2 + y)

    """
    # 检查并允许特定的参数标志
    options.allowed_flags(args, ['polys', 'auto'])

    try:
        # 将输入的多项式转换为多项式列表，并根据参数并行处理
        polys, opt = parallel_poly_from_expr([f] + list(G), *gens, **args)
    except PolificationFailed as exc:
        # 如果转换失败，抛出计算失败异常
        raise ComputationFailed('reduced', 0, exc)

    domain = opt.domain
    retract = False

    # 如果设置了自动模式，并且当前环域是环而不是域，尝试转换为域
    if opt.auto and domain.is_Ring and not domain.is_Field:
        opt = opt.clone({"domain": domain.get_field()})
        retract = True

    from sympy.polys.rings import xring
    # 使用选项中的生成器、环域和排序创建一个新的多项式环
    _ring, _ = xring(opt.gens, opt.domain, opt.order)

    for i, poly in enumerate(polys):
        # 将每个多项式转换为字典表示，然后从字典创建多项式对象
        poly = poly.set_domain(opt.domain).rep.to_dict()
        polys[i] = _ring.from_dict(poly)

    # 对第一个多项式进行多项式除法，得到商和余式
    Q, r = polys[0].div(polys[1:])

    # 将每个多项式字典转换回多项式对象
    Q = [Poly._from_dict(dict(q), opt) for q in Q]
    r = Poly._from_dict(dict(r), opt)

    # 如果之前进行了环到域的转换，则尝试将结果转换回环
    if retract:
        try:
            _Q, _r = [q.to_ring() for q in Q], r.to_ring()
        except CoercionFailed:
            pass
        else:
            Q, r = _Q, _r

    # 根据选项返回结果，如果不需要多项式对象，则转换为表达式形式返回
    if not opt.polys:
        return [q.as_expr() for q in Q], r.as_expr()
    else:
        return Q, r


@public
def groebner(F, *gens, **args):
    """
    Computes the reduced Groebner basis for a set of polynomials.

    Use the ``order`` argument to set the monomial ordering that will be
    used to compute the basis. Allowed orders are ``lex``, ``grlex`` and
    ``grevlex``. If no order is specified, it defaults to ``lex``.

    For more information on Groebner bases, see the references and the docstring
    of :func:`~.solve_poly_system`.

    Examples
    ========

    Example taken from [1].

    >>> from sympy import groebner
    >>> from sympy.abc import x, y

    >>> F = [x*y - 2*y, 2*y**2 - x**2]

    >>> groebner(F, x, y, order='lex')
    GroebnerBasis([x**2 - 2*y**2, x*y - 2*y, y**3 - 2*y], x, y,
                  domain='ZZ', order='lex')
    >>> groebner(F, x, y, order='grlex')
    GroebnerBasis([y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y], x, y,
                  domain='ZZ', order='grlex')
    >>> groebner(F, x, y, order='grevlex')
    GroebnerBasis([y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y], x, y,
                  domain='ZZ', order='grevlex')

    By default, an improved implementation of the Buchberger algorithm is
    used. Optionally, an implementation of the F5B algorithm can be used. The
    algorithm can be set using the ``method`` flag or with the

    """
    # 计算一组多项式的简化格罗布纳基础

    # 使用'order'参数设置计算基础时要使用的单项式排序方式，允许的排序包括lex、grlex和grevlex
    # 如果未指定排序方式，默认使用lex排序

    # 有关格罗布纳基础的更多信息，请参阅参考文献和solve_poly_system函数的文档字符串
    # 定义一个函数，用于计算给定多项式列表的格罗布纳基。
    def groebner(F, *gens, **args):
        # 使用 'buchberger' 方法计算格罗布纳基
        GroebnerBasis([x**2 - x - 1, y - 55], x, y, domain='ZZ', order='lex')
        # 使用 'f5b' 方法计算格罗布纳基
        GroebnerBasis([x**2 - x - 1, y - 55], x, y, domain='ZZ', order='lex')
        # 返回计算出的格罗布纳基对象
        return GroebnerBasis(F, *gens, **args)
@public
# 定义一个公共函数，用于检查由Groebner基生成的理想是否是零维的
def is_zero_dimensional(F, *gens, **args):
    """
    Checks if the ideal generated by a Groebner basis is zero-dimensional.

    The algorithm checks if the set of monomials not divisible by the
    leading monomial of any element of ``F`` is bounded.

    References
    ==========

    David A. Cox, John B. Little, Donal O'Shea. Ideals, Varieties and
    Algorithms, 3rd edition, p. 230

    """
    # 调用GroebnerBasis类的构造函数来计算Groebner基，并返回其是否是零维的布尔值
    return GroebnerBasis(F, *gens, **args).is_zero_dimensional


@public
# 定义一个公共类GroebnerBasis，表示一个简化的Groebner基
class GroebnerBasis(Basic):
    """Represents a reduced Groebner basis. """

    # 构造函数，计算给定多项式系统的简化Groebner基
    def __new__(cls, F, *gens, **args):
        """Compute a reduced Groebner basis for a system of polynomials. """
        # 检查并允许特定的选项标志
        options.allowed_flags(args, ['polys', 'method'])

        try:
            # 尝试从表达式中获取并并行生成多项式
            polys, opt = parallel_poly_from_expr(F, *gens, **args)
        except PolificationFailed as exc:
            # 如果多项式生成失败，则引发计算失败的异常
            raise ComputationFailed('groebner', len(F), exc)

        # 导入多项式环PolyRing
        from sympy.polys.rings import PolyRing
        # 创建多项式环对象
        ring = PolyRing(opt.gens, opt.domain, opt.order)

        # 将多项式列表转换为PolyRing中的字典表示，并生成新的多项式列表
        polys = [ring.from_dict(poly.rep.to_dict()) for poly in polys if poly]

        # 计算Groebner基
        G = _groebner(polys, ring, method=opt.method)
        # 将结果转换为Poly对象的列表
        G = [Poly._from_dict(g, opt) for g in G]

        # 返回GroebnerBasis类的新实例
        return cls._new(G, opt)

    @classmethod
    # 类方法，用于创建GroebnerBasis类的新实例
    def _new(cls, basis, options):
        obj = Basic.__new__(cls)

        obj._basis = tuple(basis)  # 设置Groebner基
        obj._options = options     # 设置选项参数

        return obj

    @property
    # 返回Groebner基的表达式列表
    def exprs(self):
        return [poly.as_expr() for poly in self._basis]

    @property
    # 返回Groebner基的多项式列表
    def polys(self):
        return list(self._basis)

    @property
    # 返回Groebner基的生成元列表
    def gens(self):
        return self._options.gens

    @property
    # 返回Groebner基的定义域
    def domain(self):
        return self._options.domain

    @property
    # 返回Groebner基的排序方式
    def order(self):
        return self._options.order

    # 返回Groebner基的长度（即基的多项式数目）
    def __len__(self):
        return len(self._basis)

    # 返回Groebner基的迭代器，根据选项返回多项式或表达式
    def __iter__(self):
        if self._options.polys:
            return iter(self.polys)
        else:
            return iter(self.exprs)

    # 获取Groebner基中指定索引位置的多项式或表达式
    def __getitem__(self, item):
        if self._options.polys:
            basis = self.polys
        else:
            basis = self.exprs

        return basis[item]

    # 返回Groebner基的哈希值
    def __hash__(self):
        return hash((self._basis, tuple(self._options.items())))

    # 判断当前Groebner基与另一个对象是否相等
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._basis == other._basis and self._options == other._options
        elif iterable(other):
            return self.polys == list(other) or self.exprs == list(other)
        else:
            return False

    # 判断当前Groebner基与另一个对象是否不相等
    def __ne__(self, other):
        return not self == other
    # 检查通过Groebner基生成的理想是否是零维的
    """
    Checks if the ideal generated by a Groebner basis is zero-dimensional.

    The algorithm checks if the set of monomials not divisible by the
    leading monomial of any element of ``F`` is bounded.

    References
    ==========

    David A. Cox, John B. Little, Donal O'Shea. Ideals, Varieties and
    Algorithms, 3rd edition, p. 230

    """
    # 定义一个函数，判断单项式是否为单变量
    def single_var(monomial):
        return sum(map(bool, monomial)) == 1

    # 初始化一个单项式，各变量指数为0
    exponents = Monomial([0]*len(self.gens))
    # 获取Groebner基使用的排序方式
    order = self._options.order

    # 遍历每个多项式对象
    for poly in self.polys:
        # 获取当前多项式的领先单项式
        monomial = poly.LM(order=order)

        # 如果领先单项式是单变量
        if single_var(monomial):
            # 更新指数向量
            exponents *= monomial

    # 如果指数向量中的每个元素都不为零，说明每个变量都有有限的次数上界，
    # 这意味着通过此Groebner基生成的理想是零维的。
    return all(exponents)
    def fglm(self, order):
        """
        Convert a Groebner basis from one ordering to another.

        The FGLM algorithm converts reduced Groebner bases of zero-dimensional
        ideals from one ordering to another. This method is often used when it
        is infeasible to compute a Groebner basis with respect to a particular
        ordering directly.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import groebner

        >>> F = [x**2 - 3*y - x + 1, y**2 - 2*x + y - 1]
        >>> G = groebner(F, x, y, order='grlex')

        >>> list(G.fglm('lex'))
        [2*x - y**2 - y + 1, y**4 + 2*y**3 - 3*y**2 - 16*y + 7]
        >>> list(groebner(F, x, y, order='lex'))
        [2*x - y**2 - y + 1, y**4 + 2*y**3 - 3*y**2 - 16*y + 7]

        References
        ==========

        .. [1] J.C. Faugere, P. Gianni, D. Lazard, T. Mora (1994). Efficient
               Computation of Zero-dimensional Groebner Bases by Change of
               Ordering

        """
        opt = self._options  # 获取当前对象的选项设置

        src_order = opt.order  # 获取当前对象的排序方式
        dst_order = monomial_key(order)  # 根据给定的排序方式生成目标排序的关键字

        if src_order == dst_order:  # 如果当前排序方式和目标排序方式相同，则直接返回当前对象
            return self

        if not self.is_zero_dimensional:  # 如果当前对象不是零维理想的Groebner基，则抛出未实现错误
            raise NotImplementedError("Cannot convert Groebner bases of ideals with positive dimension")

        polys = list(self._basis)  # 获取当前对象的基础多项式列表
        domain = opt.domain  # 获取当前对象的定义域

        opt = opt.clone({
            "domain": domain.get_field(),  # 将定义域设置为其域
            "order": dst_order,  # 将排序方式设置为目标排序方式
        })

        from sympy.polys.rings import xring
        _ring, _ = xring(opt.gens, opt.domain, src_order)  # 根据当前选项创建多项式环

        for i, poly in enumerate(polys):
            poly = poly.set_domain(opt.domain).rep.to_dict()  # 将多项式转换为指定域，并转换为字典形式
            polys[i] = _ring.from_dict(poly)  # 根据多项式字典在新的多项式环中创建多项式对象

        G = matrix_fglm(polys, _ring, dst_order)  # 使用FGLM算法将多项式列表转换为目标排序的Groebner基
        G = [Poly._from_dict(dict(g), opt) for g in G]  # 将结果转换为多项式对象列表

        if not domain.is_Field:  # 如果当前定义域不是域，则对结果进行通分处理
            G = [g.clear_denoms(convert=True)[1] for g in G]
            opt.domain = domain  # 恢复原始定义域

        return self._new(G, opt)  # 返回一个新的对象，其基础多项式为转换后的Groebner基，选项为更新后的选项
    def reduce(self, expr, auto=True):
        """
        Reduces a polynomial modulo a Groebner basis.

        Given a polynomial ``f`` and a set of polynomials ``G = (g_1, ..., g_n)``,
        computes a set of quotients ``q = (q_1, ..., q_n)`` and the remainder ``r``
        such that ``f = q_1*f_1 + ... + q_n*f_n + r``, where ``r`` vanishes or ``r``
        is a completely reduced polynomial with respect to ``G``.

        Examples
        ========

        >>> from sympy import groebner, expand
        >>> from sympy.abc import x, y

        >>> f = 2*x**4 - x**2 + y**3 + y**2
        >>> G = groebner([x**3 - x, y**3 - y])

        >>> G.reduce(f)
        ([2*x, 1], x**2 + y**2 + y)
        >>> Q, r = _

        >>> expand(sum(q*g for q, g in zip(Q, G)) + r)
        2*x**4 - x**2 + y**3 + y**2
        >>> _ == f
        True

        """
        # 将表达式转换为多项式对象
        poly = Poly._from_expr(expr, self._options)
        # 将当前多项式和基底多项式列表合并
        polys = [poly] + list(self._basis)

        opt = self._options
        domain = opt.domain

        retract = False

        # 如果自动模式打开且当前环为环但不是域，则切换到相应的域
        if auto and domain.is_Ring and not domain.is_Field:
            opt = opt.clone({"domain": domain.get_field()})
            retract = True

        # 导入多项式环
        from sympy.polys.rings import xring
        _ring, _ = xring(opt.gens, opt.domain, opt.order)

        # 将每个多项式转换为字典表示并从字典创建多项式环中的对象
        for i, poly in enumerate(polys):
            poly = poly.set_domain(opt.domain).rep.to_dict()
            polys[i] = _ring.from_dict(poly)

        # 对第一个多项式进行多项式长除法，返回商和余数
        Q, r = polys[0].div(polys[1:])

        # 将结果转换回多项式对象形式
        Q = [Poly._from_dict(dict(q), opt) for q in Q]
        r = Poly._from_dict(dict(r), opt)

        # 如果需要回退到原始环，则进行回退操作
        if retract:
            try:
                _Q, _r = [q.to_ring() for q in Q], r.to_ring()
            except CoercionFailed:
                pass
            else:
                Q, r = _Q, _r

        # 如果选项中不包含多项式信息，则将结果转换回表达式形式
        if not opt.polys:
            return [q.as_expr() for q in Q], r.as_expr()
        else:
            return Q, r

    def contains(self, poly):
        """
        Check if ``poly`` belongs the ideal generated by ``self``.

        Examples
        ========

        >>> from sympy import groebner
        >>> from sympy.abc import x, y

        >>> f = 2*x**3 + y**3 + 3*y
        >>> G = groebner([x**2 + y**2 - 1, x*y - 2])

        >>> G.contains(f)
        True
        >>> G.contains(f + 1)
        False

        """
        # 检查多项式是否属于由当前Groebner基底生成的理想
        return self.reduce(poly)[1] == 0
# 声明一个公共函数 poly，用于将表达式转换为多项式的形式
@public
def poly(expr, *gens, **args):
    """
    Efficiently transform an expression into a polynomial.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.abc import x

    >>> poly(x*(x**2 + x - 1)**2)
    Poly(x**5 + 2*x**4 - x**3 - 2*x**2 + x, x, domain='ZZ')

    """
    # 检查并设置传递给选项的标志，这里不允许传递任何标志
    options.allowed_flags(args, [])

    # 定义内部函数 _poly，用于递归地处理表达式并生成多项式
    def _poly(expr, opt):
        terms, poly_terms = [], []

        # 将表达式 expr 拆解成加法操作的项
        for term in Add.make_args(expr):
            factors, poly_factors = [], []

            # 将每个项进一步拆解成乘法操作的因子
            for factor in Mul.make_args(term):
                if factor.is_Add:
                    # 如果因子是加法操作，递归地处理它并生成对应的多项式因子
                    poly_factors.append(_poly(factor, opt))
                elif factor.is_Pow and factor.base.is_Add and \
                        factor.exp.is_Integer and factor.exp >= 0:
                    # 如果因子是指数操作，且指数为非负整数，处理基数并生成对应的多项式
                    poly_factors.append(
                        _poly(factor.base, opt).pow(factor.exp))
                else:
                    # 否则，将因子添加到因子列表中
                    factors.append(factor)

            # 如果没有生成多项式因子，则将当前项直接添加到 terms 中
            if not poly_factors:
                terms.append(term)
            else:
                # 否则，将多项式因子相乘得到 product
                product = poly_factors[0]
                for factor in poly_factors[1:]:
                    product = product.mul(factor)

                # 如果还有剩余因子，则将它们乘到 product 中
                if factors:
                    factor = Mul(*factors)
                    if factor.is_Number:
                        product *= factor
                    else:
                        product = product.mul(Poly._from_expr(factor, opt))

                # 将得到的 product 添加到 poly_terms 中
                poly_terms.append(product)

        # 如果没有生成多项式项，则从原始表达式 expr 直接生成结果
        if not poly_terms:
            result = Poly._from_expr(expr, opt)
        else:
            # 否则，将多项式项相加得到 result
            result = poly_terms[0]
            for term in poly_terms[1:]:
                result = result.add(term)

            # 如果还有剩余项，则将它们加到 result 中
            if terms:
                term = Add(*terms)
                if term.is_Number:
                    result += term
                else:
                    result = result.add(Poly._from_expr(term, opt))

        # 重新排序并返回最终的多项式结果，根据 opt 中的设置重新排序变量
        return result.reorder(*opt.get('gens', ()), **args)

    # 将输入的 expr 转换为 SymPy 的表达式
    expr = sympify(expr)

    # 如果输入的 expr 已经是多项式，则直接返回对应的 Poly 对象
    if expr.is_Poly:
        return Poly(expr, *gens, **args)

    # 如果未设置 'expand' 参数，则默认为 False
    if 'expand' not in args:
        args['expand'] = False

    # 构建选项 opt，并返回 _poly 函数的结果
    opt = options.build_options(gens, args)
    return _poly(expr, opt)
    # 如果 n 小于 0，抛出数值错误，提示不能生成给定索引 n 的内容
    if n < 0:
        raise ValueError("Cannot generate %s of index %s" % (name, n))
    
    # 将 x 序列分解为头部和尾部
    head, tail = x[0], x[1:]
    
    # 如果 K 为 None，则调用 construct_domain 函数构造域 K，并从 tail 中移除构造参数
    if K is None:
        K, tail = construct_domain(tail, field=True)
    
    # 调用函数 f 生成一个多项式，并将结果用 DMP 封装成多项式 poly
    poly = DMP(f(int(n), *tail, K), K)
    
    # 根据头部值 head 是否为 None，创建对应的 PurePoly 或 Poly 对象
    if head is None:
        poly = PurePoly.new(poly, Dummy('x'))
    else:
        poly = Poly.new(poly, head)
    
    # 如果 polys 参数为 True，则返回 poly 对象；否则返回其表达式形式
    return poly if polys else poly.as_expr()
```