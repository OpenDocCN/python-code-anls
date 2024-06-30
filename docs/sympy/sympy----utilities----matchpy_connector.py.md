# `D:\src\scipysrc\sympy\sympy\utilities\matchpy_connector.py`

```
"""
The objects in this module allow the usage of the MatchPy pattern matching
library on SymPy expressions.
"""

# 导入正则表达式模块
import re
# 导入类型提示模块
from typing import List, Callable, NamedTuple, Any, Dict

# 导入 SymPy 表达式的核心功能
from sympy.core.sympify import _sympify
# 导入外部模块
from sympy.external import import_module
# 导入数学函数，包括常见的对数、三角函数和特殊函数
from sympy.functions import (log, sin, cos, tan, cot, csc, sec, erf, gamma, uppergamma)
# 导入双曲函数
from sympy.functions.elementary.hyperbolic import acosh, asinh, atanh, acoth, acsch, asech, cosh, sinh, tanh, coth, sech, csch
# 导入逆三角函数
from sympy.functions.elementary.trigonometric import atan, acsc, asin, acot, acos, asec
# 导入特殊误差函数
from sympy.functions.special.error_functions import fresnelc, fresnels, erfc, erfi, Ei
# 导入 SymPy 的基本类
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
# 导入指数函数
from sympy.functions.elementary.exponential import exp
# 导入积分类
from sympy.integrals.integrals import Integral
# 导入打印模块
from sympy.printing.repr import srepr
# 导入工具装饰器
from sympy.utilities.decorator import doctest_depends_on

# 尝试导入 matchpy 模块
matchpy = import_module("matchpy")

# 指定文档测试所需的模块
__doctest_requires__ = {('*',): ['matchpy']}

# 如果 matchpy 成功导入
if matchpy:
    # 导入 matchpy 的操作类型和函数
    from matchpy import Operation, CommutativeOperation, AssociativeOperation, OneIdentityOperation
    from matchpy.expressions.functions import op_iter, create_operation_expression, op_len

    # 向 Operation 类注册以下类型
    Operation.register(Integral)
    Operation.register(Pow)
    OneIdentityOperation.register(Pow)

    # 向 Operation 类注册以下类型，并将它们标记为一元操作
    Operation.register(Add)
    OneIdentityOperation.register(Add)
    # 将 Add 标记为可交换的操作
    CommutativeOperation.register(Add)
    # 将 Add 标记为可关联的操作
    AssociativeOperation.register(Add)

    # 向 Operation 类注册以下类型，并将它们标记为一元操作
    Operation.register(Mul)
    OneIdentityOperation.register(Mul)
    # 将 Mul 标记为可交换的操作
    CommutativeOperation.register(Mul)
    # 将 Mul 标记为可关联的操作
    AssociativeOperation.register(Mul)

    # 向 Operation 类注册以下类型
    Operation.register(Equality)
    # 将 Equality 标记为可交换的操作
    CommutativeOperation.register(Equality)
    # 向 Operation 类注册以下类型
    Operation.register(Unequality)
    # 将 Unequality 标记为可交换的操作
    CommutativeOperation.register(Unequality)

    # 向 Operation 类注册以下函数
    Operation.register(exp)
    Operation.register(log)
    Operation.register(gamma)
    Operation.register(uppergamma)
    Operation.register(fresnels)
    Operation.register(fresnelc)
    Operation.register(erf)
    Operation.register(Ei)
    Operation.register(erfc)
    Operation.register(erfi)
    Operation.register(sin)
    Operation.register(cos)
    Operation.register(tan)
    Operation.register(cot)
    Operation.register(csc)
    Operation.register(sec)
    Operation.register(sinh)
    Operation.register(cosh)
    Operation.register(tanh)
    Operation.register(coth)
    Operation.register(csch)
    Operation.register(sech)
    Operation.register(asin)
    Operation.register(acos)
    Operation.register(atan)
    Operation.register(acot)
    Operation.register(acsc)
    Operation.register(asec)
    Operation.register(asinh)
    Operation.register(acosh)
    Operation.register(atanh)
    Operation.register(acoth)
    Operation.register(acsch)
    Operation.register(asech)

    # 为 Integral 类型注册 op_iter 方法
    @op_iter.register(Integral)  # type: ignore
    # 定义一个函数 `_`，接受一个操作 `operation` 作为参数，返回一个迭代器
    def _(operation):
        # 返回一个迭代器，包含操作 `operation` 的第一个参数和其余参数组成的元组
        return iter((operation._args[0],) + operation._args[1])

    # 注册一个操作 `op_iter`，对于类型为 `Basic` 的操作进行处理（忽略类型检查）
    @op_iter.register(Basic)  # type: ignore
    def _(operation):
        # 返回一个迭代器，包含操作 `operation` 的所有参数
        return iter(operation._args)

    # 注册一个操作 `op_len`，对于 `Integral` 类型的操作进行处理（忽略类型检查）
    @op_len.register(Integral)  # type: ignore
    def _(operation):
        # 返回操作 `operation` 的参数数量加一
        return 1 + len(operation._args[1])

    # 注册一个操作 `op_len`，对于 `Basic` 类型的操作进行处理（忽略类型检查）
    @op_len.register(Basic)  # type: ignore
    def _(operation):
        # 返回操作 `operation` 的参数数量
        return len(operation._args)

    # 定义一个函数 `create_operation_expression.register`，用于创建基本操作表达式
    @create_operation_expression.register(Basic)
    def sympy_op_factory(old_operation, new_operands, variable_name=True):
        # 返回一个类型为 `old_operation` 的新操作，使用 `new_operands` 替换原操作的参数
        return type(old_operation)(*new_operands)
if matchpy:
    from matchpy import Wildcard  # 如果有matchpy模块，则导入Wildcard类
else:
    class Wildcard:  # 如果没有matchpy模块，则定义一个Wildcard类，忽略类型检查
        def __init__(self, min_length, fixed_size, variable_name, optional):
            self.min_count = min_length  # 设置最小计数
            self.fixed_size = fixed_size  # 设置固定大小
            self.variable_name = variable_name  # 设置变量名
            self.optional = optional  # 设置可选性


@doctest_depends_on(modules=('matchpy',))
class _WildAbstract(Wildcard, Symbol):
    min_length: int  # 抽象字段，在子类中需要定义
    fixed_size: bool  # 抽象字段，在子类中需要定义

    def __init__(self, variable_name=None, optional=None, **assumptions):
        min_length = self.min_length  # 使用抽象字段设置最小长度
        fixed_size = self.fixed_size  # 使用抽象字段设置固定大小
        if optional is not None:
            optional = _sympify(optional)  # 如果可选项不为空，则转换为符号表达式
        Wildcard.__init__(self, min_length, fixed_size, str(variable_name), optional)  # 初始化Wildcard父类

    def __getstate__(self):
        return {
            "min_length": self.min_length,  # 返回对象状态的最小长度
            "fixed_size": self.fixed_size,  # 返回对象状态的固定大小
            "min_count": self.min_count,  # 返回对象状态的最小计数
            "variable_name": self.variable_name,  # 返回对象状态的变量名
            "optional": self.optional,  # 返回对象状态的可选性
        }

    def __new__(cls, variable_name=None, optional=None, **assumptions):
        cls._sanitize(assumptions, cls)  # 调用静态方法_sanitize以确保符号假设的正确性
        return _WildAbstract.__xnew__(cls, variable_name, optional, **assumptions)  # 调用静态方法__xnew__创建新对象

    def __getnewargs__(self):
        return self.variable_name, self.optional  # 返回对象的变量名和可选性作为新参数

    @staticmethod
    def __xnew__(cls, variable_name=None, optional=None, **assumptions):
        obj = Symbol.__xnew__(cls, variable_name, **assumptions)  # 调用Symbol类的静态方法__xnew__创建符号对象
        return obj

    def _hashable_content(self):
        if self.optional:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name, self.optional)  # 如果可选项存在，则返回可哈希内容
        else:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name)  # 否则，返回不包含可选项的可哈希内容

    def __copy__(self) -> '_WildAbstract':
        return type(self)(variable_name=self.variable_name, optional=self.optional)  # 返回副本对象

    def __repr__(self):
        return str(self)  # 返回对象的字符串表示形式

    def __str__(self):
        return self.name  # 返回对象的名称


@doctest_depends_on(modules=('matchpy',))
class WildDot(_WildAbstract):
    min_length = 1  # 设置最小长度为1
    fixed_size = True  # 设置固定大小为True


@doctest_depends_on(modules=('matchpy',))
class WildPlus(_WildAbstract):
    min_length = 1  # 设置最小长度为1
    fixed_size = False  # 设置固定大小为False


@doctest_depends_on(modules=('matchpy',))
class WildStar(_WildAbstract):
    min_length = 0  # 设置最小长度为0
    fixed_size = False  # 设置固定大小为False


def _get_srepr(expr):
    s = srepr(expr)  # 获取表达式的规范字符串表示形式
    s = re.sub(r"WildDot\('(\w+)'\)", r"\1", s)  # 将WildDot替换为其变量名
    s = re.sub(r"WildPlus\('(\w+)'\)", r"*\1", s)  # 将WildPlus替换为"*变量名"
    s = re.sub(r"WildStar\('(\w+)'\)", r"*\1", s)  # 将WildStar替换为"*变量名"
    return s  # 返回替换后的字符串表示形式


class ReplacementInfo(NamedTuple):
    replacement: Any  # 替换的内容
    info: Any  # 相关信息


@doctest_depends_on(modules=('matchpy',))
class Replacer:
    """
    Replacer object to perform multiple pattern matching and subexpression
    replacements in SymPy expressions.

    Examples
    ========

    Example to construct a simple first degree equation solver:
    """
    # 导入所需模块和类
    >>> from sympy.utilities.matchpy_connector import WildDot, Replacer
    >>> from sympy import Equality, Symbol
    
    # 定义符号变量 x
    >>> x = Symbol("x")
    
    # 定义两个通配符 WildDot，用于匹配方程 a*x + b = 0 的系数
    >>> a_ = WildDot("a_", optional=1)
    >>> b_ = WildDot("b_", optional=0)

    # 上述代码定义了两个通配符 a_ 和 b_，它们分别表示方程 a*x + b = 0 中的系数 a 和 b。
    # optional 参数指定了在没有匹配的情况下返回的表达式，对于 a*x = 0 和 x + b = 0 的情况，这些设置是必要的。

    # 创建两个约束条件，确保 a_ 和 b_ 不会匹配包含变量 x 的任何表达式
    >>> from matchpy import CustomConstraint
    >>> free_x_a = CustomConstraint(lambda a_: not a_.has(x))
    >>> free_x_b = CustomConstraint(lambda b_: not b_.has(x))

    # 创建 Replacer 对象，带有自定义的约束条件
    >>> replacer = Replacer(common_constraints=[free_x_a, free_x_b])

    # 添加匹配规则：当匹配到方程 a_*x + b_ = 0 时，替换为 -b_/a_
    >>> replacer.add(Equality(a_*x + b_, 0), -b_/a_)

    # 测试匹配功能
    >>> replacer.replace(Equality(3*x + 4, 0))
    -4/3

    # 注意，它不会匹配其他模式的方程：
    >>> eq = Equality(3*x, 4)
    >>> replacer.replace(eq)
    Eq(3*x, 4)

    # 为了扩展匹配模式，定义另一个匹配规则，并清除缓存以确保重新匹配
    >>> replacer.add(Equality(a_*x, b_), b_/a_)
    >>> replacer._matcher.clear()
    >>> replacer.replace(eq)
    4/3
    """

    # 定义 Replacer 类，用于模式匹配和替换
    def __init__(self, common_constraints: list = [], lambdify: bool = False, info: bool = False):
        # 初始化多对一匹配器
        self._matcher = matchpy.ManyToOneMatcher()
        # 设置公共约束条件
        self._common_constraint = common_constraints
        # 是否进行 lambdify 操作的标志
        self._lambdify = lambdify
        # 是否输出详细信息的标志
        self._info = info
        # 用于存储通配符的字典
        self._wildcards: Dict[str, Wildcard] = {}

    # 从字符串获取 Lambda 表达式的函数
    def _get_lambda(self, lambda_str: str) -> Callable[..., Expr]:
        # 动态导入 sympy 模块中的所有内容
        exec("from sympy import *")
        # 返回 Lambda 表达式的可调用对象
        return eval(lambda_str, locals())

    # 获取自定义约束条件的函数，基于给定的约束表达式和条件模板
    def _get_custom_constraint(self, constraint_expr: Expr, condition_template: str) -> Callable[..., Expr]:
        # 获取约束表达式中的通配符名称
        wilds = [x.name for x in constraint_expr.atoms(_WildAbstract)]
        # 构建 Lambda 表达式的参数列表
        lambdaargs = ', '.join(wilds)
        # 获取完整的表达式字符串表示
        fullexpr = _get_srepr(constraint_expr)
        # 根据条件模板生成约束条件字符串
        condition = condition_template.format(fullexpr)
        # 返回自定义约束条件对象
        return matchpy.CustomConstraint(
            self._get_lambda(f"lambda {lambdaargs}: ({condition})"))

    # 获取不为 False 的自定义约束条件的函数
    def _get_custom_constraint_nonfalse(self, constraint_expr: Expr) -> Callable[..., Expr]:
        return self._get_custom_constraint(constraint_expr, "({}) != False")

    # 获取为 True 的自定义约束条件的函数
    def _get_custom_constraint_true(self, constraint_expr: Expr) -> Callable[..., Expr]:
        return self._get_custom_constraint(constraint_expr, "({}) == True")
    def add(self, expr: Expr, replacement, conditions_true: List[Expr] = [],
            conditions_nonfalse: List[Expr] = [], info: Any = None) -> None:
        # 将输入的表达式和替换项转换为 SymPy 表达式对象
        expr = _sympify(expr)
        replacement = _sympify(replacement)
        # 复制公共约束条件列表，以确保不修改原始列表
        constraints = self._common_constraint[:]
        # 为真条件列表生成自定义约束条件
        constraint_conditions_true = [
            self._get_custom_constraint_true(cond) for cond in conditions_true]
        # 为非假条件列表生成自定义约束条件
        constraint_conditions_nonfalse = [
            self._get_custom_constraint_nonfalse(cond) for cond in conditions_nonfalse]
        # 将所有约束条件合并到约束列表中
        constraints.extend(constraint_conditions_true)
        constraints.extend(constraint_conditions_nonfalse)
        # 创建匹配模式对象，用于匹配表达式
        pattern = matchpy.Pattern(expr, *constraints)
        
        # 如果启用了 lambda 替换方式
        if self._lambdify:
            # 构建 lambda 表达式字符串
            lambda_str = f"lambda {', '.join((x.name for x in expr.atoms(_WildAbstract)))}: {_get_srepr(replacement)}"
            # 调用内部方法获取 lambda 表达式对象
            lambda_expr = self._get_lambda(lambda_str)
            # 更新替换项为 lambda 表达式
            replacement = lambda_expr
        else:
            # 更新通配符字典，以便匹配任意通配符
            self._wildcards.update({str(i): i for i in expr.atoms(Wildcard)})
        
        # 如果有额外信息，创建 ReplacementInfo 对象来包装替换项和信息
        if self._info:
            replacement = ReplacementInfo(replacement, info)
        
        # 将模式和替换项添加到匹配器中
        self._matcher.add(pattern, replacement)

    def replace(self, expression, max_count: int = -1):
        # 本方法部分重写了 MatchPy 中 ManyToOneReplacer 类的 .replace 方法
        # 许可证：https://github.com/HPAC/matchpy/blob/master/LICENSE
        # 初始化信息列表
        infos = []
        # 标记是否替换成功的标志
        replaced = True
        # 初始化替换计数器
        replace_count = 0
        
        # 循环直到未替换任何内容或达到最大替换次数
        while replaced and (max_count < 0 or replace_count < max_count):
            # 每次循环开始时标记为未替换
            replaced = False
            # 使用前序遍历及位置信息遍历表达式的每个子表达式
            for subexpr, pos in matchpy.preorder_iter_with_position(expression):
                try:
                    # 尝试寻找匹配的替换数据和替换字典
                    replacement_data, subst = next(iter(self._matcher.match(subexpr)))
                    # 如果需要信息，获取替换项的信息
                    if self._info:
                        replacement = replacement_data.replacement
                        infos.append(replacement_data.info)
                    else:
                        replacement = replacement_data
                    
                    # 如果启用了 lambda 替换方式，应用 lambda 函数
                    if self._lambdify:
                        result = replacement(**subst)
                    else:
                        # 否则，使用 subst 字典替换通配符
                        result = replacement.xreplace({self._wildcards[k]: v for k, v in subst.items()})
                    
                    # 使用 MatchPy 函数替换表达式中的子表达式
                    expression = matchpy.functions.replace(expression, pos, result)
                    # 标记已替换成功
                    replaced = True
                    # 退出当前循环
                    break
                except StopIteration:
                    # 如果未找到匹配，继续循环查找下一个子表达式
                    pass
            
            # 增加替换计数器
            replace_count += 1
        
        # 如果需要返回信息列表，一并返回表达式和信息列表；否则，只返回表达式
        if self._info:
            return expression, infos
        else:
            return expression
```