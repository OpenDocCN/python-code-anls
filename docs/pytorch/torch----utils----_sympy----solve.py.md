# `.\pytorch\torch\utils\_sympy\solve.py`

```py
# 导入 logging 模块，用于记录日志
import logging

# 导入类型提示模块
from typing import Dict, Optional, Tuple, Type

# 导入 sympy 符号计算库
import sympy

# 导入 torch 中的 FloorDiv 函数
from torch.utils._sympy.functions import FloorDiv

# 设置日志记录器
log = logging.getLogger(__name__)

# 定义一个字典，用于将某些 sympy.Basic 类型的关系运算符映射为对应的反向关系运算符
_MIRROR_REL_OP: Dict[Type[sympy.Basic], Type[sympy.Rel]] = {
    sympy.Eq: sympy.Eq,
    sympy.Ne: sympy.Ne,
    sympy.Ge: sympy.Le,
    sympy.Gt: sympy.Lt,
    sympy.Le: sympy.Ge,
    sympy.Lt: sympy.Gt,
}

# 定义一个元组，包含支持的不等式类型
INEQUALITY_TYPES = (sympy.Gt, sympy.Ge, sympy.Lt, sympy.Le)


# 函数：获取给定关系运算符类型的反向关系运算符类型
# 返回值：如果存在，返回对应的反向关系运算符类型；否则返回 None
def mirror_rel_op(type: Type) -> Optional[Type[sympy.Rel]]:
    return _MIRROR_REL_OP.get(type, None)


# 函数：尝试简化表达式 'expr'，使左侧仅保留 'thing'
# 返回值：返回一个元组，包含简化后的表达式和右侧表达式；如果无法仅保留左侧 'thing'，返回 None
# 参数：
#   - expr: 待简化的 sympy.Basic 类型的表达式
#   - thing: 需要保留在左侧的 sympy.Basic 类型的对象
#   - trials: 尝试次数，默认为 5
#   - floordiv_inequality: 是否启用将 'FloorDiv' 转换为不等式的标志，默认为 True
def try_solve(
    expr: sympy.Basic,
    thing: sympy.Basic,
    trials: int = 5,
    floordiv_inequality: bool = True,
) -> Optional[Tuple[sympy.Rel, sympy.Basic]]:
    # 获取当前表达式类型的反向关系运算符
    mirror = mirror_rel_op(type(expr))

    # 忽略不支持的表达式：
    #   - 非关系运算符
    #   - 没有反向运算符的表达式
    if not isinstance(expr, sympy.Rel) or mirror is None:
        log.debug("expression with unsupported type: %s", type(expr))
        return None

    # 检查左侧和右侧是否包含 'thing'
    lhs_has_thing = expr.lhs.has(thing)
    rhs_has_thing = expr.rhs.has(thing)

    # 如果 'thing' 同时出现在左侧和右侧，放弃简化
    if lhs_has_thing and rhs_has_thing:
        log.debug("thing (%s) found in both sides of expression: %s", thing, expr)
        return None

    # 尝试考虑 'expr' 的镜像版本：如果 'thing' 在左侧，添加原始表达式；如果在右侧，添加镜像后的版本
    expressions = []
    if lhs_has_thing:
        expressions.append(expr)
    if rhs_has_thing:
        expressions.append(mirror(expr.rhs, expr.lhs))

    for e in expressions:
        if e is None:
            continue

        assert isinstance(e, sympy.Rel)

        for _ in range(trials):
            # 尝试将 'thing' 隔离到左侧
            trial = _try_isolate_lhs(e, thing, floordiv_inequality=floordiv_inequality)
            # 如果本次试验没有改变表达式，则停止
            if trial == e:
                break
            e = trial  # type: ignore[assignment]

        # 如果成功将 'thing' 隔离到左侧，返回简化后的表达式和右侧表达式
        if isinstance(e, sympy.Rel) and e.lhs == thing:
            log.debug("solved: %s ---> %s", expr, e)
            return e, e.rhs

    return None


# 函数：尝试将表达式 'expr' 中的 'thing' 隔离到左侧
# 返回值：返回简化后的 sympy.Basic 类型的表达式
# 参数：
#   - expr: 待简化的 sympy.Basic 类型的表达式
#   - thing: 需要保留在左侧的 sympy.Basic 类型的对象
#   - floordiv_inequality: 是否启用将 'FloorDiv' 转换为不等式的标志
def _try_isolate_lhs(
    expr: sympy.Basic, thing: sympy.Basic, floordiv_inequality: bool
) -> sympy.Basic:
    e = expr
    op = type(expr)

    if isinstance(e, sympy.Rel):
        # 如果表达式 e 是 sympy.Rel 类型的实例

        # 将左侧不含 thing 的常量移动到右侧。
        lhs_not_thing = (
            sum(a for a in e.lhs.args if not a.has(thing))
            if isinstance(e.lhs, sympy.Add)
            else 0
        )
        # 使用 op 类型构造新的表达式，更新 e
        e = op(expr.lhs - lhs_not_thing, expr.rhs - lhs_not_thing)  # type: ignore[attr-defined]

    # 将两侧都除以不包含 thing 的因子。
    if isinstance(e, sympy.Rel) and isinstance(e.lhs, sympy.Mul):
        lhs, rhs = e.args
        other = sympy.Mul(*[a for a in lhs.args if not a.has(thing)])

        # 如果我们无法确定 'other' 是正数还是负数，我们不进行操作。
        # 这是因为我们不知道是否需要翻转操作。
        if not (isinstance(e, INEQUALITY_TYPES) and other.is_negative is None):
            # 将两侧都除以 'other'。
            lhs = lhs / other
            rhs = rhs / other

            # 如果 'e' 是不等式且 'other' 是负数，我们需要翻转表达式。
            if isinstance(e, INEQUALITY_TYPES) and other.is_negative:
                op = mirror_rel_op(op)  # type: ignore[assignment]

            assert op is not None
            e = op(lhs, rhs)

    ################################################################################
    # 左侧是 FloorDiv
    ################################################################################
    #
    # 给定表达式: a // b op c
    # 其中 'op' 是一个关系操作符，以下规则仅在以下条件下有效:
    #   - b > 0
    #   - c 是整数
    if (
        floordiv_inequality
        and isinstance(e, sympy.Rel)
        and isinstance(e.lhs, FloorDiv)
        and e.lhs.divisor.is_positive
        and e.rhs.is_integer
    ):
        # 如果 floordiv_inequality 为真且 e 是 sympy.Rel 类型的实例，
        # 并且 e 的左侧是 FloorDiv 类型，并且除数为正，右侧为整数。
    ):
        # 如果表达式是等式，将其左侧分子和分母分别赋值给变量
        if isinstance(expr, sympy.Eq):
            numerator, denominator = e.lhs.args
            # 返回一个逻辑与条件，确保左侧分子大于等于右侧乘以分母，且小于右侧乘以（分母 + 1）
            return sympy.And(
                sympy.Ge(numerator, (e.rhs * denominator)),  # type: ignore[arg-type]
                sympy.Lt(numerator, ((e.rhs + 1) * denominator)),  # type: ignore[arg-type]
            )
        # 如果表达式是不等式，将其左侧分子和分母分别赋值给变量
        elif isinstance(expr, sympy.Ne):
            numerator, denominator = e.lhs.args
            # 返回一个逻辑或条件，确保左侧分子小于右侧乘以分母，或者大于等于右侧乘以（分母 + 1）
            return sympy.Or(
                sympy.Lt(numerator, (e.rhs * denominator)),  # type: ignore[arg-type]
                sympy.Ge(numerator, ((e.rhs + 1) * denominator)),  # type: ignore[arg-type]
            )
        
        # 以下变换仅在分母为正数时有效。
        # 注意：我们只有常数时才能知道这些信息。
        
        # 如果表达式是大于或者大于等于
        elif isinstance(expr, (sympy.Gt, sympy.Ge)):
            quotient = e.rhs if isinstance(expr, sympy.Ge) else (e.rhs + 1)  # type: ignore[arg-type]
            # 返回一个大于或者大于等于条件，确保左侧第一个参数大于等于右侧乘以第二个参数
            return sympy.Ge(e.lhs.args[0], (quotient * e.lhs.args[1]))  # type: ignore[arg-type]
        
        # 如果表达式是小于或者小于等于
        elif isinstance(expr, (sympy.Lt, sympy.Le)):
            quotient = e.rhs if isinstance(expr, sympy.Lt) else (e.rhs + 1)  # type: ignore[arg-type]
            # 返回一个小于或者小于等于条件，确保左侧第一个参数小于右侧乘以第二个参数
            return sympy.Lt(e.lhs.args[0], (quotient * e.lhs.args[1]))  # type: ignore[arg-type]

    # 如果不符合以上任何情况，则返回原始表达式
    return e
```