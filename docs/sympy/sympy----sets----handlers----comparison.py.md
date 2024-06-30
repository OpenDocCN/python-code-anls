# `D:\src\scipysrc\sympy\sympy\sets\handlers\comparison.py`

```
# 导入必要的库函数来定义特定的比较函数
from sympy.core.relational import Eq, is_eq  # 导入相等性比较相关函数
from sympy.core.basic import Basic  # 导入基本的 Sympy 类
from sympy.core.logic import fuzzy_and, fuzzy_bool  # 导入模糊逻辑运算函数
from sympy.logic.boolalg import And  # 导入布尔代数中的 AND 运算函数
from sympy.multipledispatch import dispatch  # 导入多重分派函数
from sympy.sets.sets import tfn, ProductSet, Interval, FiniteSet, Set  # 导入集合和区间相关函数


@dispatch(Interval, FiniteSet) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return False  # 如果左操作数是 Interval 而右操作数是 FiniteSet，则返回 False


@dispatch(FiniteSet, Interval) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return False  # 如果左操作数是 FiniteSet 而右操作数是 Interval，则返回 False


@dispatch(Interval, Interval) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return And(Eq(lhs.left, rhs.left),  # 如果两个操作数都是 Interval，比较左边界和右边界是否相等
               Eq(lhs.right, rhs.right),  # 比较右边界是否相等
               lhs.left_open == rhs.left_open,  # 比较左边界是否开放
               lhs.right_open == rhs.right_open)  # 比较右边界是否开放


@dispatch(FiniteSet, FiniteSet) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    def all_in_both():
        s_set = set(lhs.args)  # 将 lhs 的元素转换为集合
        o_set = set(rhs.args)  # 将 rhs 的元素转换为集合
        yield fuzzy_and(lhs._contains(e) for e in o_set - s_set)  # 检查 o_set 中的元素是否都在 s_set 中
        yield fuzzy_and(rhs._contains(e) for e in s_set - o_set)  # 检查 s_set 中的元素是否都在 o_set 中

    return tfn[fuzzy_and(all_in_both())]  # 返回模糊逻辑 AND 运算结果


@dispatch(ProductSet, ProductSet) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    if len(lhs.sets) != len(rhs.sets):  # 如果两个 ProductSet 中的子集数量不相等，则返回 False
        return False

    eqs = (is_eq(x, y) for x, y in zip(lhs.sets, rhs.sets))  # 逐个比较 lhs 和 rhs 的每个子集是否相等
    return tfn[fuzzy_and(map(fuzzy_bool, eqs))]  # 返回模糊逻辑 AND 运算结果


@dispatch(Set, Basic) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return False  # 如果左操作数是 Set 而右操作数是 Basic 类型，则返回 False


@dispatch(Set, Set) # type:ignore
def _eval_is_eq(lhs, rhs): # noqa: F811
    return tfn[fuzzy_and(a.is_subset(b) for a, b in [(lhs, rhs), (rhs, lhs)])]  # 检查 lhs 是否是 rhs 和 rhs 是否是 lhs 的子集，返回模糊逻辑 AND 运算结果
```