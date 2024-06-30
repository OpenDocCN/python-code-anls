# `D:\src\scipysrc\sympy\sympy\sets\handlers\issubset.py`

```
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.logic import fuzzy_and, fuzzy_bool, fuzzy_not, fuzzy_or
from sympy.core.relational import Eq
from sympy.sets.sets import FiniteSet, Interval, Set, Union, ProductSet
from sympy.sets.fancysets import Complexes, Reals, Range, Rationals
from sympy.multipledispatch import Dispatcher

# 定义一组无穷集合，包括自然数、非负整数、整数、有理数、实数和复数
_inf_sets = [S.Naturals, S.Naturals0, S.Integers, S.Rationals, S.Reals, S.Complexes]

# 创建一个分派器，用于判断集合是否为另一个集合的子集
is_subset_sets = Dispatcher('is_subset_sets')

# 注册函数，用于判断两个一般集合之间的子集关系
@is_subset_sets.register(Set, Set)
def _(a, b):
    return None

# 注册函数，用于判断两个区间集合之间的子集关系
@is_subset_sets.register(Interval, Interval)
def _(a, b):
    # 这段代码逻辑正确，但可以更加全面...
    if fuzzy_bool(a.start < b.start):
        return False
    if fuzzy_bool(a.end > b.end):
        return False
    if (b.left_open and not a.left_open and fuzzy_bool(Eq(a.start, b.start))):
        return False
    if (b.right_open and not a.right_open and fuzzy_bool(Eq(a.end, b.end))):
        return False

# 注册函数，用于判断区间集合是否是有限集合的子集
@is_subset_sets.register(Interval, FiniteSet)
def _(a_interval, b_fs):
    # 只有当区间集合的测度为零时，它才能是有限集合的子集
    if fuzzy_not(a_interval.measure.is_zero):
        return False

# 注册函数，用于判断区间集合是否是并集的子集
@is_subset_sets.register(Interval, Union)
def _(a_interval, b_u):
    if all(isinstance(s, (Interval, FiniteSet)) for s in b_u.args):
        intervals = [s for s in b_u.args if isinstance(s, Interval)]
        if all(fuzzy_bool(a_interval.start < s.start) for s in intervals):
            return False
        if all(fuzzy_bool(a_interval.end > s.end) for s in intervals):
            return False
        if a_interval.measure.is_nonzero:
            no_overlap = lambda s1, s2: fuzzy_or([
                    fuzzy_bool(s1.end <= s2.start),
                    fuzzy_bool(s1.start >= s2.end),
                    ])
            if all(no_overlap(s, a_interval) for s in intervals):
                return False

# 注册函数，用于判断范围集合是否是另一个范围集合的子集
@is_subset_sets.register(Range, Range)
def _(a, b):
    if a.step == b.step == 1:
        return fuzzy_and([fuzzy_bool(a.start >= b.start),
                          fuzzy_bool(a.stop <= b.stop)])

# 注册函数，用于判断范围集合是否是区间集合的子集
@is_subset_sets.register(Range, Interval)
def _(a_range, b_interval):
    if a_range.step.is_positive:
        if b_interval.left_open and a_range.inf.is_finite:
            cond_left = a_range.inf > b_interval.left
        else:
            cond_left = a_range.inf >= b_interval.left
        if b_interval.right_open and a_range.sup.is_finite:
            cond_right = a_range.sup < b_interval.right
        else:
            cond_right = a_range.sup <= b_interval.right
        return fuzzy_and([cond_left, cond_right])

# 注册函数，用于判断范围集合是否是有限集合的子集
@is_subset_sets.register(Range, FiniteSet)
def _(a_range, b_finiteset):
    try:
        a_size = a_range.size
    except ValueError:
        # 未知大小的符号范围
        return None
    if a_size > len(b_finiteset):
        return False
    # 如果 a_range 中的任何参数包含 Symbol 符号，则调用 fuzzy_and 函数进行模糊与操作
    elif any(arg.has(Symbol) for arg in a_range.args):
        return fuzzy_and(b_finiteset.contains(x) for x in a_range)
    else:
        # 检查 A \ B == EmptySet 比在任意 FiniteSet 上反复进行朴素成员检查更有效率。
        # 将 a_range 转换为集合 a_set
        a_set = set(a_range)
        # 获取集合 b_finiteset 的长度，作为 b_remaining 的初始值
        b_remaining = len(b_finiteset)
        # 对于符号表达式和未知类型的数字（整数或其他），都视为 "候选项"，即可能匹配 a_range 中某个元素
        cnt_candidate = 0
        # 遍历集合 b_finiteset 中的每个元素 b
        for b in b_finiteset:
            # 如果 b 是整数，则从 a_set 中移除 b
            if b.is_Integer:
                a_set.discard(b)
            # 如果 b 不是整数（模糊不是整数）
            elif fuzzy_not(b.is_integer):
                pass
            # 否则，认定为候选项，增加 cnt_candidate 计数
            else:
                cnt_candidate += 1
            # 每处理一个 b，减少 b_remaining 计数
            b_remaining -= 1
            # 如果 a_set 中剩余的元素个数大于 b_remaining 加上 cnt_candidate 的数量，返回 False
            if len(a_set) > b_remaining + cnt_candidate:
                return False
            # 如果 a_set 中的元素数量为 0，返回 True
            if len(a_set) == 0:
                return True
        # 如果循环结束仍未返回结果，则返回 None
        return None
# 为 Interval 和 Range 类型的对象注册子集关系判断函数
@is_subset_sets.register(Interval, Range)
def _(a_interval, b_range):
    # 如果 a_interval 的测度（measure）是扩展非零的，则返回 False
    if a_interval.measure.is_extended_nonzero:
        return False

# 为 Interval 和 Rationals 类型的对象注册子集关系判断函数
@is_subset_sets.register(Interval, Rationals)
def _(a_interval, b_rationals):
    # 如果 a_interval 的测度（measure）是扩展非零的，则返回 False
    if a_interval.measure.is_extended_nonzero:
        return False

# 为 Range 和 Complexes 类型的对象注册子集关系判断函数
@is_subset_sets.register(Range, Complexes)
def _(a, b):
    # 直接返回 True，表明 Range 是 Complexes 的子集
    return True

# 为 Complexes 和 Interval 类型的对象注册子集关系判断函数
@is_subset_sets.register(Complexes, Interval)
def _(a, b):
    # 直接返回 False，表明 Complexes 不是 Interval 的子集
    return False

# 为 Complexes 和 Range 类型的对象注册子集关系判断函数
@is_subset_sets.register(Complexes, Range)
def _(a, b):
    # 直接返回 False，表明 Complexes 不是 Range 的子集
    return False

# 为 Complexes 和 Rationals 类型的对象注册子集关系判断函数
@is_subset_sets.register(Complexes, Rationals)
def _(a, b):
    # 直接返回 False，表明 Complexes 不是 Rationals 的子集
    return False

# 为 Rationals 和 Reals 类型的对象注册子集关系判断函数
@is_subset_sets.register(Rationals, Reals)
def _(a, b):
    # 直接返回 True，表明 Rationals 是 Reals 的子集
    return True

# 为 Rationals 和 Range 类型的对象注册子集关系判断函数
@is_subset_sets.register(Rationals, Range)
def _(a, b):
    # 直接返回 False，表明 Rationals 不是 Range 的子集
    return False

# 为 ProductSet 和 FiniteSet 类型的对象注册子集关系判断函数
@is_subset_sets.register(ProductSet, FiniteSet)
def _(a_ps, b_fs):
    # 返回模糊逻辑与运算的结果，检查 a_ps 中的所有元素是否都包含在 b_fs 中
    return fuzzy_and(b_fs.contains(x) for x in a_ps)
```