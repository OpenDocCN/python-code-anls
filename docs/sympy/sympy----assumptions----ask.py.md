# `D:\src\scipysrc\sympy\sympy\assumptions\ask.py`

```
# 导入 SymPy 模块中的各种功能和类
"""Module for querying SymPy objects about assumptions."""

# 导入全局假设、谓词和应用谓词相关的模块和类
from sympy.assumptions.assume import (global_assumptions, Predicate,
        AppliedPredicate)

# 导入 CNF（合取范式）、EncodedCNF（编码的合取范式）和 Literal（文字）类
from sympy.assumptions.cnf import CNF, EncodedCNF, Literal

# 导入 sympify 函数，用于将输入转换为 SymPy 对象
from sympy.core import sympify

# 导入 BooleanKind 类，用于表示布尔类型
from sympy.core.kind import BooleanKind

# 导入关系运算符的类：Eq（等于）、Ne（不等于）、Gt（大于）、Lt（小于）、Ge（大于等于）、Le（小于等于）
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le

# 导入逻辑推理模块中的 satisfiable 函数，用于求解逻辑表达式的可满足性
from sympy.logic.inference import satisfiable

# 导入 memoize_property 装饰器，用于实现属性的记忆化（memoization）
from sympy.utilities.decorator import memoize_property

# 导入异常相关模块中的警告类：sympy_deprecation_warning、SymPyDeprecationWarning、ignore_warnings
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning,
                                        ignore_warnings)


# Memoization is necessary for the properties of AssumptionKeys to
# ensure that only one object of Predicate objects are created.
# This is because assumption handlers are registered on those objects.

# AssumptionKeys 类用于管理所有支持的谓词键值对，用于“ask”查询
class AssumptionKeys:
    """
    This class contains all the supported keys by ``ask``.
    It should be accessed via the instance ``sympy.Q``.

    """

    # 不要在此类中添加方法或属性，除非是谓词键
    # SAT 求解器检查 Q 的属性，并使用它们计算事实系统。非谓词属性会破坏这一机制。

    # 装饰器 memoize_property 用于记忆化谓词属性 hermitian
    @memoize_property
    def hermitian(self):
        from .handlers.sets import HermitianPredicate
        return HermitianPredicate
    # 定义一个装饰器函数 memoize_property，用于缓存属性的值，避免重复计算
    @memoize_property
    # 定义一个属性方法 infinite，返回无穷的谓词对象
    def infinite(self):
        # 从 .handlers.calculus 模块导入 InfinitePredicate 类
        from .handlers.calculus import InfinitePredicate
        # 返回 InfinitePredicate 的实例
        return InfinitePredicate()

    @memoize_property
    # 定义一个属性方法 positive_infinite，返回正无穷的谓词对象
    def positive_infinite(self):
        # 从 .handlers.calculus 模块导入 PositiveInfinitePredicate 类
        from .handlers.calculus import PositiveInfinitePredicate
        # 返回 PositiveInfinitePredicate 的实例
        return PositiveInfinitePredicate()

    @memoize_property
    # 定义一个属性方法 negative_infinite，返回负无穷的谓词对象
    def negative_infinite(self):
        # 从 .handlers.calculus 模块导入 NegativeInfinitePredicate 类
        from .handlers.calculus import NegativeInfinitePredicate
        # 返回 NegativeInfinitePredicate 的实例
        return NegativeInfinitePredicate()

    @memoize_property
    # 定义一个属性方法 positive，返回正数的谓词对象
    def positive(self):
        # 从 .handlers.order 模块导入 PositivePredicate 类
        from .handlers.order import PositivePredicate
        # 返回 PositivePredicate 的实例
        return PositivePredicate()

    @memoize_property
    # 定义一个属性方法 negative，返回负数的谓词对象
    def negative(self):
        # 从 .handlers.order 模块导入 NegativePredicate 类
        from .handlers.order import NegativePredicate
        # 返回 NegativePredicate 的实例
        return NegativePredicate()

    @memoize_property
    # 定义一个属性方法 zero，返回零的谓词对象
    def zero(self):
        # 从 .handlers.order 模块导入 ZeroPredicate 类
        from .handlers.order import ZeroPredicate
        # 返回 ZeroPredicate 的实例
        return ZeroPredicate()

    @memoize_property
    # 定义一个属性方法 extended_positive，返回扩展正数的谓词对象
    def extended_positive(self):
        # 从 .handlers.order 模块导入 ExtendedPositivePredicate 类
        from .handlers.order import ExtendedPositivePredicate
        # 返回 ExtendedPositivePredicate 的实例
        return ExtendedPositivePredicate()

    @memoize_property
    # 定义一个属性方法 extended_negative，返回扩展负数的谓词对象
    def extended_negative(self):
        # 从 .handlers.order 模块导入 ExtendedNegativePredicate 类
        from .handlers.order import ExtendedNegativePredicate
        # 返回 ExtendedNegativePredicate 的实例
        return ExtendedNegativePredicate()

    @memoize_property
    # 定义一个属性方法 nonzero，返回非零数的谓词对象
    def nonzero(self):
        # 从 .handlers.order 模块导入 NonZeroPredicate 类
        from .handlers.order import NonZeroPredicate
        # 返回 NonZeroPredicate 的实例
        return NonZeroPredicate()

    @memoize_property
    # 定义一个属性方法 nonpositive，返回非正数的谓词对象
    def nonpositive(self):
        # 从 .handlers.order 模块导入 NonPositivePredicate 类
        from .handlers.order import NonPositivePredicate
        # 返回 NonPositivePredicate 的实例
        return NonPositivePredicate()

    @memoize_property
    # 定义一个属性方法 nonnegative，返回非负数的谓词对象
    def nonnegative(self):
        # 从 .handlers.order 模块导入 NonNegativePredicate 类
        from .handlers.order import NonNegativePredicate
        # 返回 NonNegativePredicate 的实例
        return NonNegativePredicate()

    @memoize_property
    # 定义一个属性方法 extended_nonzero，返回扩展非零数的谓词对象
    def extended_nonzero(self):
        # 从 .handlers.order 模块导入 ExtendedNonZeroPredicate 类
        from .handlers.order import ExtendedNonZeroPredicate
        # 返回 ExtendedNonZeroPredicate 的实例
        return ExtendedNonZeroPredicate()

    @memoize_property
    # 定义一个属性方法 extended_nonpositive，返回扩展非正数的谓词对象
    def extended_nonpositive(self):
        # 从 .handlers.order 模块导入 ExtendedNonPositivePredicate 类
        from .handlers.order import ExtendedNonPositivePredicate
        # 返回 ExtendedNonPositivePredicate 的实例
        return ExtendedNonPositivePredicate()

    @memoize_property
    # 定义一个属性方法 extended_nonnegative，返回扩展非负数的谓词对象
    def extended_nonnegative(self):
        # 从 .handlers.order 模块导入 ExtendedNonNegativePredicate 类
        from .handlers.order import ExtendedNonNegativePredicate
        # 返回 ExtendedNonNegativePredicate 的实例
        return ExtendedNonNegativePredicate()

    @memoize_property
    # 定义一个属性方法 even，返回偶数的谓词对象
    def even(self):
        # 从 .handlers.ntheory 模块导入 EvenPredicate 类
        from .handlers.ntheory import EvenPredicate
        # 返回 EvenPredicate 的实例
        return EvenPredicate()

    @memoize_property
    # 定义一个属性方法 odd，返回奇数的谓词对象
    def odd(self):
        # 从 .handlers.ntheory 模块导入 OddPredicate 类
        from .handlers.ntheory import OddPredicate
        # 返回 OddPredicate 的实例
        return OddPredicate()

    @memoize_property
    # 定义一个属性方法 prime，返回素数的谓词对象
    def prime(self):
        # 从 .handlers.ntheory 模块导入 PrimePredicate 类
        from .handlers.ntheory import PrimePredicate
        # 返回 PrimePredicate 的实例
        return PrimePredicate()

    @memoize_property
    # 定义一个属性方法 composite，返回合数的谓词对象
    def composite(self):
        # 从 .handlers.ntheory 模块导入 CompositePredicate 类
        from .handlers.ntheory import CompositePredicate
        # 返回 CompositePredicate 的实例
        return CompositePredicate()

    @memoize_property
    # 定义一个属性方法 commutative，返回可交换的谓词对象
    def commutative(self):
        # 从 .handlers.common 模块导入 CommutativePredicate 类
        from .handlers.common import CommutativePredicate
        # 返回 CommutativePredicate 的实例
        return CommutativePredicate()

    @memoize_property
    # 定义一个属性方法 is_true，返回为真的谓词对象
    def is_true(self):
        # 从 .handlers.common 模块导入 IsTruePredicate 类
        from .handlers.common import IsTruePredicate
        # 返回 IsTruePredicate 的实例
        return IsTruePredicate()

    @memoize_property
    # 定义一个属性方法
    # 这是一个未完成的注释块，代码没有提供完整的内容
    # 返回一个 SymmetricPredicate 实例，用于检查矩阵是否对称
    def symmetric(self):
        from .handlers.matrices import SymmetricPredicate
        return SymmetricPredicate()

    # 返回一个 InvertiblePredicate 实例，用于检查矩阵是否可逆
    @memoize_property
    def invertible(self):
        from .handlers.matrices import InvertiblePredicate
        return InvertiblePredicate()

    # 返回一个 OrthogonalPredicate 实例，用于检查矩阵是否正交
    @memoize_property
    def orthogonal(self):
        from .handlers.matrices import OrthogonalPredicate
        return OrthogonalPredicate()

    # 返回一个 UnitaryPredicate 实例，用于检查矩阵是否酉（幺正）
    @memoize_property
    def unitary(self):
        from .handlers.matrices import UnitaryPredicate
        return UnitaryPredicate()

    # 返回一个 PositiveDefinitePredicate 实例，用于检查矩阵是否正定
    @memoize_property
    def positive_definite(self):
        from .handlers.matrices import PositiveDefinitePredicate
        return PositiveDefinitePredicate()

    # 返回一个 UpperTriangularPredicate 实例，用于检查矩阵是否上三角
    @memoize_property
    def upper_triangular(self):
        from .handlers.matrices import UpperTriangularPredicate
        return UpperTriangularPredicate()

    # 返回一个 LowerTriangularPredicate 实例，用于检查矩阵是否下三角
    @memoize_property
    def lower_triangular(self):
        from .handlers.matrices import LowerTriangularPredicate
        return LowerTriangularPredicate()

    # 返回一个 DiagonalPredicate 实例，用于检查矩阵是否对角
    @memoize_property
    def diagonal(self):
        from .handlers.matrices import DiagonalPredicate
        return DiagonalPredicate()

    # 返回一个 FullRankPredicate 实例，用于检查矩阵是否满秩
    @memoize_property
    def fullrank(self):
        from .handlers.matrices import FullRankPredicate
        return FullRankPredicate()

    # 返回一个 SquarePredicate 实例，用于检查矩阵是否方阵
    @memoize_property
    def square(self):
        from .handlers.matrices import SquarePredicate
        return SquarePredicate()

    # 返回一个 IntegerElementsPredicate 实例，用于检查矩阵元素是否为整数
    @memoize_property
    def integer_elements(self):
        from .handlers.matrices import IntegerElementsPredicate
        return IntegerElementsPredicate()

    # 返回一个 RealElementsPredicate 实例，用于检查矩阵元素是否为实数
    @memoize_property
    def real_elements(self):
        from .handlers.matrices import RealElementsPredicate
        return RealElementsPredicate()

    # 返回一个 ComplexElementsPredicate 实例，用于检查矩阵元素是否为复数
    @memoize_property
    def complex_elements(self):
        from .handlers.matrices import ComplexElementsPredicate
        return ComplexElementsPredicate()

    # 返回一个 SingularPredicate 实例，用于检查矩阵是否奇异（非可逆）
    @memoize_property
    def singular(self):
        from .predicates.matrices import SingularPredicate
        return SingularPredicate()

    # 返回一个 NormalPredicate 实例，用于检查矩阵是否正规
    @memoize_property
    def normal(self):
        from .predicates.matrices import NormalPredicate
        return NormalPredicate()

    # 返回一个 TriangularPredicate 实例，用于检查矩阵是否三角
    @memoize_property
    def triangular(self):
        from .predicates.matrices import TriangularPredicate
        return TriangularPredicate()

    # 返回一个 UnitTriangularPredicate 实例，用于检查矩阵是否单位三角
    @memoize_property
    def unit_triangular(self):
        from .predicates.matrices import UnitTriangularPredicate
        return UnitTriangularPredicate()

    # 返回一个 EqualityPredicate 实例，用于检查两个矩阵是否相等
    @memoize_property
    def eq(self):
        from .relation.equality import EqualityPredicate
        return EqualityPredicate()

    # 返回一个 UnequalityPredicate 实例，用于检查两个矩阵是否不等
    @memoize_property
    def ne(self):
        from .relation.equality import UnequalityPredicate
        return UnequalityPredicate()

    # 返回一个 StrictGreaterThanPredicate 实例，用于检查一个矩阵是否严格大于另一个矩阵
    @memoize_property
    def gt(self):
        from .relation.equality import StrictGreaterThanPredicate
        return StrictGreaterThanPredicate()
    # 定义一个方法 `ge`，返回大于比较谓词对象
    def ge(self):
        # 导入严格大于比较谓词类
        from .relation.equality import GreaterThanPredicate
        # 返回一个 GreaterThanPredicate 的实例
        return GreaterThanPredicate()
    
    # 定义一个装饰为缓存属性的方法 `lt`，返回严格小于比较谓词对象
    @memoize_property
    def lt(self):
        # 导入严格小于比较谓词类
        from .relation.equality import StrictLessThanPredicate
        # 返回一个 StrictLessThanPredicate 的实例
        return StrictLessThanPredicate()
    
    # 定义一个装饰为缓存属性的方法 `le`，返回小于等于比较谓词对象
    @memoize_property
    def le(self):
        # 导入小于等于比较谓词类
        from .relation.equality import LessThanPredicate
        # 返回一个 LessThanPredicate 的实例
        return LessThanPredicate()
# 创建一个名为 Q 的对象，该对象用于表示假设的键值
Q = AssumptionKeys()

# 定义一个函数 _extract_all_facts，用于从给定的假设中提取与表达式相关的所有事实
def _extract_all_facts(assump, exprs):
    """
    Extract all relevant assumptions from *assump* with respect to given *exprs*.

    Parameters
    ==========

    assump : sympy.assumptions.cnf.CNF
        给定的假设，类型为 CNF（合取范式）

    exprs : tuple of expressions
        包含要检查的表达式的元组

    Returns
    =======

    sympy.assumptions.cnf.CNF
        包含从假设中提取的所有相关事实的新 CNF 对象

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.ask import _extract_all_facts
    >>> from sympy.abc import x, y
    >>> assump = CNF.from_prop(Q.positive(x) & Q.integer(y))
    >>> exprs = (x,)
    >>> cnf = _extract_all_facts(assump, exprs)
    >>> cnf.clauses
    {frozenset({Literal(Q.positive, False)})}

    """
    # 初始化一个空集合用于存储事实
    facts = set()

    # 遍历假设的每个子句
    for clause in assump.clauses:
        # 初始化一个空列表用于存储符合条件的 literal
        args = []
        # 遍历子句中的每个 literal
        for literal in clause:
            # 检查 literal 是否是 AppliedPredicate 类型，并且其参数长度为 1
            if isinstance(literal.lit, AppliedPredicate) and len(literal.lit.arguments) == 1:
                # 如果 literal 的参数在 exprs 中，则添加到 args 列表中
                if literal.lit.arg in exprs:
                    args.append(Literal(literal.lit.function, literal.is_Not))
                else:
                    # 如果任何一个 literal 的参数不在 exprs 中，则不添加整个子句
                    break
            else:
                # 如果任何一个 literal 不是一元谓词，则不添加整个子句
                break
        else:
            # 如果 args 列表不为空，则将其转换为不可变集合并添加到 facts 中
            if args:
                facts.add(frozenset(args))
    
    # 返回一个新的 CNF 对象，其中包含从假设中提取的所有相关事实
    return CNF(facts)


# 定义一个函数 ask，用于评估带有假设的命题的真假
def ask(proposition, assumptions=True, context=global_assumptions):
    """
    Function to evaluate the proposition with assumptions.

    Explanation
    ===========

    This function evaluates the proposition to ``True`` or ``False`` if
    the truth value can be determined. If not, it returns ``None``.

    It should be discerned from :func:`~.refine()` which, when applied to a
    proposition, simplifies the argument to symbolic ``Boolean`` instead of
    Python built-in ``True``, ``False`` or ``None``.

    **Syntax**

        * ask(proposition)
            Evaluate the *proposition* in global assumption context.

        * ask(proposition, assumptions)
            Evaluate the *proposition* with respect to *assumptions* in
            global assumption context.

    Parameters
    ==========

    proposition : Boolean
        Proposition which will be evaluated to boolean value. If this is
        not ``AppliedPredicate``, it will be wrapped by ``Q.is_true``.

    assumptions : Boolean, optional
        Local assumptions to evaluate the *proposition*.

    context : AssumptionsContext, optional
        Default assumptions to evaluate the *proposition*. By default,
        this is ``sympy.assumptions.global_assumptions`` variable.

    Returns
    =======

    ``True``, ``False``, or ``None``

    Raises
    ======

    TypeError : *proposition* or *assumptions* is not valid logical expression.

    ValueError : assumptions are inconsistent.

    Examples
    ========

    """
    # 函数的具体实现被省略，在这里不进行详细讨论
    # 导入必要的库和模块
    from sympy import ask, Q, pi
    from sympy.abc import x, y

    # 使用 ask 函数查询 pi 是否为有理数，返回 False
    ask(Q.rational(pi))

    # 使用 ask 函数查询 x*y 是否为偶数，给定 x 是偶数且 y 是整数，返回 True
    ask(Q.even(x*y), Q.even(x) & Q.integer(y))

    # 使用 ask 函数查询 4*x 是否为素数，给定 x 是整数，返回 False
    ask(Q.prime(4*x), Q.integer(x))

    # 如果无法确定真值，返回 None
    print(ask(Q.odd(3*x)))  # 除非知道 x 的具体值，否则无法确定

    # 如果假设不一致，抛出 ValueError
    ask(Q.integer(x), Q.even(x) & Q.odd(x))

    # 以下是一些关于假设的说明和提示
    """
    Relations in assumptions are not implemented (yet), so the following
    will not give a meaningful result.

    sympy.assumptions.refine.refine : Simplification using assumptions.
        Proposition is not reduced to ``None`` if the truth value cannot
        be determined.
    """

    # 导入必要的函数和类
    from sympy.assumptions.satask import satask
    from sympy.assumptions.lra_satask import lra_satask
    from sympy.logic.algorithms.lra_theory import UnhandledInput

    # 将命题和假设符号化
    proposition = sympify(proposition)
    assumptions = sympify(assumptions)

    # 检查命题和假设的合法性
    if isinstance(proposition, Predicate) or proposition.kind is not BooleanKind:
        raise TypeError("proposition must be a valid logical expression")

    if isinstance(assumptions, Predicate) or assumptions.kind is not BooleanKind:
        raise TypeError("assumptions must be a valid logical expression")

    # 定义二元关系谓词映射
    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}

    # 根据命题类型确定关键字和参数
    if isinstance(proposition, AppliedPredicate):
        key, args = proposition.function, proposition.arguments
    elif proposition.func in binrelpreds:
        key, args = binrelpreds[type(proposition)], proposition.args
    else:
        key, args = Q.is_true, (proposition,)

    # 将局部和全局假设转换为合取范式 (CNF)
    assump_cnf = CNF.from_prop(assumptions)
    assump_cnf.extend(context)

    # 从假设中提取与参数相关的事实
    local_facts = _extract_all_facts(assump_cnf, args)

    # 获取所有已知事实的编码合取范式
    known_facts_cnf = get_all_known_facts()
    enc_cnf = EncodedCNF()
    enc_cnf.from_cnf(CNF(known_facts_cnf))
    enc_cnf.add_from_cnf(local_facts)

    # 检查给定假设的可满足性
    if local_facts.clauses and satisfiable(enc_cnf) is False:
        raise ValueError("inconsistent assumptions %s" % assumptions)

    # 快速计算单个事实
    res = _ask_single_fact(key, local_facts)
    if res is not None:
        return res

    # 直接解析方法，无逻辑
    res = key(*args)._eval_ask(assumptions)
    if res is not None:
        return bool(res)

    # 使用 satask 进行解析（成本高）
    res = satask(proposition, assumptions=assumptions, context=context)
    if res is not None:
        return res
    # 尝试调用 lra_satask 函数进行逻辑推理分析
    try:
        res = lra_satask(proposition, assumptions=assumptions, context=context)
    # 处理 UnhandledInput 异常，如果捕获到此异常，则返回 None
    except UnhandledInput:
        return None

    # 返回逻辑推理分析的结果
    return res
def _ask_single_fact(key, local_facts):
    """
    Compute the truth value of a single predicate using assumptions.

    Parameters
    ==========

    key : sympy.assumptions.assume.Predicate
        Proposition predicate.

    local_facts : sympy.assumptions.cnf.CNF
        Local assumptions in CNF form.

    Returns
    =======

    ``True``, ``False`` or ``None``

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.ask import _ask_single_fact

    If the assumption rejects the prerequisite of the proposition,
    return ``False``.

    >>> key, assump = Q.zero, ~Q.zero
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False
    >>> key, assump = Q.zero, ~Q.even
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False

    If the assumption implies the proposition, return ``True``.

    >>> key, assump = Q.even, Q.zero
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    True

    If the proposition rejects the assumption, return ``False``.

    >>> key, assump = Q.even, Q.odd
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False
    """
    if local_facts.clauses:
        # 获取已知事实字典
        known_facts_dict = get_known_facts_dict()

        if len(local_facts.clauses) == 1:
            cl, = local_facts.clauses
            if len(cl) == 1:
                f, = cl
                # 获取关于 key 的已知事实集合
                prop_facts = known_facts_dict.get(key, None)
                prop_req = prop_facts[0] if prop_facts is not None else set()
                if f.is_Not and f.arg in prop_req:
                    # 命题的先决条件被拒绝
                    return False

        for clause in local_facts.clauses:
            if len(clause) == 1:
                f, = clause
                # 获取关于 f.arg 的已知事实元组 (prop_req, prop_rej)
                prop_facts = known_facts_dict.get(f.arg, None) if not f.is_Not else None
                if prop_facts is None:
                    continue

                prop_req, prop_rej = prop_facts
                if key in prop_req:
                    # 假设暗示命题成立
                    return True
                elif key in prop_rej:
                    # 命题拒绝假设
                    return False

    return None


def register_handler(key, handler):
    """
    Register a handler in the ask system. key must be a string and handler a
    class inheriting from AskHandler.

    .. deprecated:: 1.8.
        Use multipledispatch handler instead. See :obj:`~.Predicate`.

    """
    sympy_deprecation_warning(
        """
        The AskHandler system is deprecated. The register_handler() function
        should be replaced with the multipledispatch handler of Predicate.
        """,
        deprecated_since_version="1.8",
        active_deprecations_target='deprecated-askhandler',
    )
    # 如果 key 是 Predicate 的实例，则将 key 设置为 key.name.name，即取其名称属性的名称
    if isinstance(key, Predicate):
        key = key.name.name
    
    # 获取 Q 对象中名为 key 的属性，并赋值给 Qkey
    Qkey = getattr(Q, key, None)
    
    # 如果 Qkey 不为 None，则将 handler 添加到 Qkey 的处理器列表中
    if Qkey is not None:
        Qkey.add_handler(handler)
    # 如果 Qkey 为 None，则在 Q 对象中设置名为 key 的属性，属性值为一个新的 Predicate 对象，带有 handler 列表作为参数
    else:
        setattr(Q, key, Predicate(key, handlers=[handler]))
# 定义一个函数，用于从 ask 系统中移除一个处理程序
def remove_handler(key, handler):
    """
    Removes a handler from the ask system.

    .. deprecated:: 1.8.
        Use multipledispatch handler instead. See :obj:`~.Predicate`.

    """
    # 发出 SymPy 弃用警告，提醒用户不再使用 AskHandler 系统
    sympy_deprecation_warning(
        """
        The AskHandler system is deprecated. The remove_handler() function
        should be replaced with the multipledispatch handler of Predicate.
        """,
        deprecated_since_version="1.8",
        active_deprecations_target='deprecated-askhandler',
    )
    
    # 如果 key 是 Predicate 类型的实例，将其转换为关联的名称字符串
    if isinstance(key, Predicate):
        key = key.name.name
    
    # 使用 ignore_warnings 上下文管理器，忽略 SymPy 的警告，避免递归显示相同警告
    with ignore_warnings(SymPyDeprecationWarning):
        # 获取 Q 对象中 key 所对应的属性，并移除其中的处理程序 handler
        getattr(Q, key).remove_handler(handler)


# 导入 get_all_known_facts 和 get_known_facts_dict 函数
from sympy.assumptions.ask_generated import (get_all_known_facts,
    get_known_facts_dict)
```