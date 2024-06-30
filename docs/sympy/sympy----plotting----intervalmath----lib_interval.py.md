# `D:\src\scipysrc\sympy\sympy\plotting\intervalmath\lib_interval.py`

```
""" The module contains implemented functions for interval arithmetic."""
from functools import reduce

from sympy.plotting.intervalmath import interval
from sympy.external import import_module


def Abs(x):
    if isinstance(x, (int, float)):
        return interval(abs(x))
    elif isinstance(x, interval):
        if x.start < 0 and x.end > 0:
            return interval(0, max(abs(x.start), abs(x.end)), is_valid=x.is_valid)
        else:
            return interval(abs(x.start), abs(x.end))
    else:
        raise NotImplementedError


#Monotonic
def exp(x):
    """evaluates the exponential of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.exp(x), np.exp(x))
    elif isinstance(x, interval):
        return interval(np.exp(x.start), np.exp(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


#Monotonic
def log(x):
    """evaluates the natural logarithm of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.log(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-np.inf, np.inf, is_valid=x.is_valid)
        elif x.end <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        elif x.start <= 0:
            return interval(-np.inf, np.inf, is_valid=None)

        return interval(np.log(x.start), np.log(x.end))
    else:
        raise NotImplementedError


#Monotonic
def log10(x):
    """evaluates the logarithm to the base 10 of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.log10(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-np.inf, np.inf, is_valid=x.is_valid)
        elif x.end <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        elif x.start <= 0:
            return interval(-np.inf, np.inf, is_valid=None)
        return interval(np.log10(x.start), np.log10(x.end))
    else:
        raise NotImplementedError


#Monotonic
def atan(x):
    """evaluates the tan inverse of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.arctan(x))
    elif isinstance(x, interval):
        start = np.arctan(x.start)
        end = np.arctan(x.end)
        return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


#periodic
def sin(x):
    """evaluates the sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        # Evaluate sine of the interval [x.start, x.end]
        return interval(np.sin(x.start), np.sin(x.end))
    else:
        raise NotImplementedError
    elif isinstance(x, interval):
        # 如果 x 是 interval 类型的实例，则执行以下操作
        if not x.is_valid:
            # 如果 x 不是有效的区间，则返回一个无效区间对象
            return interval(-1, 1, is_valid=x.is_valid)
        # 计算 x.start 和 x.end 除以 np.pi / 2.0 的商和余数
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        # 计算 np.sin(x.start) 和 np.sin(x.end) 的最小值和最大值
        start = min(np.sin(x.start), np.sin(x.end))
        end = max(np.sin(x.start), np.sin(x.end))
        # 如果区间跨越超过四个 π，则返回一个 [-1, 1] 的区间对象
        if nb - na > 4:
            return interval(-1, 1, is_valid=x.is_valid)
        # 如果 na 和 nb 相等，则返回一个有效区间对象，其范围为 start 到 end
        elif na == nb:
            return interval(start, end, is_valid=x.is_valid)
        else:
            # 如果 na 和 nb 不相等，则根据 na 和 nb 的差别进行调整 start 和 end 的值
            if (na - 1) // 4 != (nb - 1) // 4:
                # 如果 (na - 1) // 4 != (nb - 1) // 4，则 end 设置为 1，表示 sin 函数达到最大值
                end = 1
            if (na - 3) // 4 != (nb - 3) // 4:
                # 如果 (na - 3) // 4 != (nb - 3) // 4，则 start 设置为 -1，表示 sin 函数达到最小值
                start = -1
            # 返回根据调整后的 start 和 end 构建的区间对象
            return interval(start, end)
    else:
        # 如果 x 不是 interval 类型的实例，则抛出未实现错误
        raise NotImplementedError
#periodic
def cos(x):
    """计算区间的余弦值"""
    # 导入 numpy 模块
    np = import_module('numpy')
    # 如果 x 是 int 或 float 类型，则返回其余弦值所构成的区间
    if isinstance(x, (int, float)):
        return interval(np.cos(x))
    # 如果 x 是 interval 类型
    elif isinstance(x, interval):
        # 如果 x 的起始或结束值不是有限的，则返回一个 [-1, 1] 的区间，有效性与 x 一致
        if not (np.isfinite(x.start) and np.isfinite(x.end)):
            return interval(-1, 1, is_valid=x.is_valid)
        # 计算 x.start 和 x.end 分别除以 π/2 后的商和余数
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        # 计算区间起始和结束点的余弦值
        start = min(np.cos(x.start), np.cos(x.end))
        end = max(np.cos(x.start), np.cos(x.end))
        # 如果 nb - na 大于 4，表示区间超过了 2*pi 的差异，返回 [-1, 1] 区间
        if nb - na > 4:
            return interval(-1, 1, is_valid=x.is_valid)
        # 如果 na 和 nb 相等，即在同一个象限内，返回 [start, end] 区间
        elif na == nb:
            return interval(start, end, is_valid=x.is_valid)
        else:
            # 否则根据象限进行调整
            if (na) // 4 != (nb) // 4:
                end = 1
            if (na - 2) // 4 != (nb - 2) // 4:
                start = -1
            return interval(start, end, is_valid=x.is_valid)
    else:
        # 如果 x 不是支持的类型，抛出未实现的异常
        raise NotImplementedError


def tan(x):
    """计算区间的正切值"""
    # 调用 sin(x) / cos(x) 计算正切值
    return sin(x) / cos(x)


#Monotonic
def sqrt(x):
    """计算区间的平方根"""
    np = import_module('numpy')
    # 如果 x 是 int 或 float 类型
    if isinstance(x, (int, float)):
        # 如果 x 大于 0，则返回其平方根构成的区间
        if x > 0:
            return interval(np.sqrt(x))
        # 否则返回 (-∞, ∞) 区间，有效性为 False
        else:
            return interval(-np.inf, np.inf, is_valid=False)
    # 如果 x 是 interval 类型
    elif isinstance(x, interval):
        # 如果 x 的结束值小于 0，则返回 (-∞, ∞) 区间，有效性为 False
        if x.end < 0:
            return interval(-np.inf, np.inf, is_valid=False)
        # 如果 x 的起始值小于 0，则返回 (-∞, ∞) 区间，有效性为 None
        elif x.start < 0:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            # 否则返回 [sqrt(x.start), sqrt(x.end)] 区间，有效性与 x 一致
            return interval(np.sqrt(x.start), np.sqrt(x.end),
                            is_valid=x.is_valid)
    else:
        # 如果 x 不是支持的类型，抛出未实现的异常
        raise NotImplementedError


def imin(*args):
    """计算区间列表的最小值"""
    np = import_module('numpy')
    # 如果所有参数都是 int、float 或 interval 类型
    if not all(isinstance(arg, (int, float, interval)) for arg in args):
        return NotImplementedError
    else:
        # 筛选出所有有效参数
        new_args = [a for a in args if isinstance(a, (int, float))
                    or a.is_valid]
        # 如果没有有效参数
        if len(new_args) == 0:
            # 如果所有参数的有效性都为 False，则返回 (-∞, ∞) 区间，有效性为 False
            if all(a.is_valid is False for a in args):
                return interval(-np.inf, np.inf, is_valid=False)
            # 否则返回 (-∞, ∞) 区间，有效性为 None
            else:
                return interval(-np.inf, np.inf, is_valid=None)
        # 提取所有有效参数的起始值组成的列表
        start_array = [a if isinstance(a, (int, float)) else a.start
                       for a in new_args]
        # 提取所有有效参数的结束值组成的列表
        end_array = [a if isinstance(a, (int, float)) else a.end
                     for a in new_args]
        # 返回最小值构成的区间
        return interval(min(start_array), min(end_array))


def imax(*args):
    """计算区间列表的最大值"""
    np = import_module('numpy')
    # 如果所有参数都是 int、float 或 interval 类型
    if not all(isinstance(arg, (int, float, interval)) for arg in args):
        return NotImplementedError
    # 如果条件不成立，执行以下代码块
    else:
        # 从参数列表中筛选出整数、浮点数或具有 is_valid 属性的参数
        new_args = [a for a in args if isinstance(a, (int, float))
                    or a.is_valid]
        
        # 如果筛选后的参数列表长度为 0
        if len(new_args) == 0:
            # 如果所有参数的 is_valid 属性均为 False
            if all(a.is_valid is False for a in args):
                # 返回一个表示整个实数轴区间的 interval 对象，无效
                return interval(-np.inf, np.inf, is_valid=False)
            else:
                # 返回一个表示整个实数轴区间的 interval 对象，未定义有效性
                return interval(-np.inf, np.inf, is_valid=None)
        
        # 从筛选后的参数列表中提取起始值，如果参数是数值类型则直接使用该值
        start_array = [a if isinstance(a, (int, float)) else a.start
                       for a in new_args]

        # 从筛选后的参数列表中提取结束值，如果参数是数值类型则直接使用该值
        end_array = [a if isinstance(a, (int, float)) else a.end
                     for a in new_args]

        # 返回一个 interval 对象，其起始值为 start_array 中的最大值，结束值为 end_array 中的最大值
        return interval(max(start_array), max(end_array))
#Monotonic
def sinh(x):
    """Evaluates the hyperbolic sine of an interval"""
    np = import_module('numpy')
    # 如果 x 是整数或浮点数，则计算单个值的双曲正弦
    if isinstance(x, (int, float)):
        return interval(np.sinh(x), np.sinh(x))
    # 如果 x 是区间对象，则计算区间的双曲正弦
    elif isinstance(x, interval):
        return interval(np.sinh(x.start), np.sinh(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


def cosh(x):
    """Evaluates the hyperbolic cos of an interval"""
    np = import_module('numpy')
    # 如果 x 是整数或浮点数，则计算单个值的双曲余弦
    if isinstance(x, (int, float)):
        return interval(np.cosh(x), np.cosh(x))
    elif isinstance(x, interval):
        # 如果区间同时包含正负号，则取最大值
        if x.start < 0 and x.end > 0:
            end = max(np.cosh(x.start), np.cosh(x.end))
            return interval(1, end, is_valid=x.is_valid)
        else:
            #Monotonic
            # 否则计算区间的双曲余弦
            start = np.cosh(x.start)
            end = np.cosh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


#Monotonic
def tanh(x):
    """Evaluates the hyperbolic tan of an interval"""
    np = import_module('numpy')
    # 如果 x 是整数或浮点数，则计算单个值的双曲正切
    if isinstance(x, (int, float)):
        return interval(np.tanh(x), np.tanh(x))
    elif isinstance(x, interval):
        # 否则计算区间的双曲正切
        return interval(np.tanh(x.start), np.tanh(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


def asin(x):
    """Evaluates the inverse sine of an interval"""
    np = import_module('numpy')
    # 如果 x 是整数或浮点数
    if isinstance(x, (int, float)):
        # 如果 x 超出定义域 [-1, 1]，则返回无穷区间
        if abs(x) > 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            # 否则计算单个值的反正弦
            return interval(np.arcsin(x), np.arcsin(x))
    elif isinstance(x, interval):
        # 如果区间无效或者完全超出定义域 [-1, 1]，则返回无穷区间
        if x.is_valid is False or x.start > 1 or x.end < -1:
            return interval(-np.inf, np.inf, is_valid=False)
        # 如果区间部分超出定义域 [-1, 1]，则返回无穷区间，但有效性未确定
        elif x.start < -1 or x.end > 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            # 否则计算区间的反正弦
            start = np.arcsin(x.start)
            end = np.arcsin(x.end)
            return interval(start, end, is_valid=x.is_valid)


def acos(x):
    """Evaluates the inverse cos of an interval"""
    np = import_module('numpy')
    # 如果 x 是整数或浮点数
    if isinstance(x, (int, float)):
        # 如果 x 超出定义域 [-1, 1]，则返回无穷区间
        if abs(x) > 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            # 否则计算单个值的反余弦
            return interval(np.arccos(x), np.arccos(x))
    elif isinstance(x, interval):
        # 如果区间无效或者完全超出定义域 [-1, 1]，则返回无穷区间
        if x.is_valid is False or x.start > 1 or x.end < -1:
            return interval(-np.inf, np.inf, is_valid=False)
        # 如果区间部分超出定义域 [-1, 1]，则返回无穷区间，但有效性未确定
        elif x.start < -1 or x.end > 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            # 否则计算区间的反余弦
            start = np.arccos(x.start)
            end = np.arccos(x.end)
            return interval(start, end, is_valid=x.is_valid)


def ceil(x):
    """Evaluates the ceiling of an interval"""
    np = import_module('numpy')
    # 如果 x 是 int 或 float 类型，返回 x 的上取整后的区间
    if isinstance(x, (int, float)):
        return interval(np.ceil(x))
    # 如果 x 是 interval 类型
    elif isinstance(x, interval):
        # 如果 x 是无效区间
        if x.is_valid is False:
            # 返回一个表示全集的区间，is_valid 设为 False
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            # 对区间的起始点和终点进行上取整操作
            start = np.ceil(x.start)
            end = np.ceil(x.end)
            # 如果起始点和终点相等，表示区间内连续
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                # 如果起始点和终点不相等，表示区间内不连续
                return interval(start, end, is_valid=None)
    else:
        # 如果 x 不是 int、float 或 interval 类型，则抛出未实现的错误
        return NotImplementedError
# 定义函数 `floor`，用于计算数值或区间的下取整操作
def floor(x):
    # 导入 numpy 模块
    np = import_module('numpy')
    # 如果 x 是 int 或 float 类型
    if isinstance(x, (int, float)):
        # 返回 x 的下取整后的区间对象
        return interval(np.floor(x))
    # 如果 x 是 interval 类型
    elif isinstance(x, interval):
        # 如果 x 无效
        if x.is_valid is False:
            # 返回无效区间对象
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            # 对区间起始和结束进行下取整操作
            start = np.floor(x.start)
            end = np.floor(x.end)
            # 如果起始和结束相等
            if start == end:
                # 返回有效区间对象
                return interval(start, end, is_valid=x.is_valid)
            else:
                # 返回无效区间对象（不连续区间）
                return interval(start, end, is_valid=None)
    else:
        # 如果类型不支持，返回未实现错误
        return NotImplementedError


# 定义函数 `acosh`，用于计算数值或区间的反双曲余弦操作
def acosh(x):
    # 导入 numpy 模块
    np = import_module('numpy')
    # 如果 x 是 int 或 float 类型
    if isinstance(x, (int, float)):
        # 如果 x 小于 1，返回无效区间对象
        if x < 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            # 返回 x 的反双曲余弦值组成的区间对象
            return interval(np.arccosh(x))
    # 如果 x 是 interval 类型
    elif isinstance(x, interval):
        # 如果 x 的结束值小于 1，返回无效区间对象
        if x.end < 1:
            return interval(-np.inf, np.inf, is_valid=False)
        # 如果 x 的起始值小于 1，返回部分无效区间对象
        elif x.start < 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            # 对区间起始和结束值进行反双曲余弦操作
            start = np.arccosh(x.start)
            end = np.arccosh(x.end)
            # 返回结果区间对象
            return interval(start, end, is_valid=x.is_valid)
    else:
        # 如果类型不支持，返回未实现错误
        return NotImplementedError


# 定义函数 `asinh`，用于计算数值或区间的反双曲正弦操作
def asinh(x):
    # 导入 numpy 模块
    np = import_module('numpy')
    # 如果 x 是 int 或 float 类型
    if isinstance(x, (int, float)):
        # 返回 x 的反双曲正弦值组成的区间对象
        return interval(np.arcsinh(x))
    # 如果 x 是 interval 类型
    elif isinstance(x, interval):
        # 对区间起始和结束值进行反双曲正弦操作
        start = np.arcsinh(x.start)
        end = np.arcsinh(x.end)
        # 返回结果区间对象
        return interval(start, end, is_valid=x.is_valid)
    else:
        # 如果类型不支持，返回未实现错误
        return NotImplementedError


# 定义函数 `atanh`，用于计算数值或区间的反双曲正切操作
def atanh(x):
    # 导入 numpy 模块
    np = import_module('numpy')
    # 如果 x 是 int 或 float 类型
    if isinstance(x, (int, float)):
        # 如果 x 的绝对值大于等于 1，返回无效区间对象
        if abs(x) >= 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            # 返回 x 的反双曲正切值组成的区间对象
            return interval(np.arctanh(x))
    # 如果 x 是 interval 类型
    elif isinstance(x, interval):
        # 如果 x 无效或者起始值大于等于 1或结束值小于等于 -1，返回无效区间对象
        if x.is_valid is False or x.start >= 1 or x.end <= -1:
            return interval(-np.inf, np.inf, is_valid=False)
        # 如果 x 的起始值小于等于 -1 或者结束值大于等于 1，返回部分无效区间对象
        elif x.start <= -1 or x.end >= 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            # 对区间起始和结束值进行反双曲正切操作
            start = np.arctanh(x.start)
            end = np.arctanh(x.end)
            # 返回结果区间对象
            return interval(start, end, is_valid=x.is_valid)
    else:
        # 如果类型不支持，返回未实现错误
        return NotImplementedError


# 定义函数 `And`，用于定义两个三值逻辑值的三值“与”操作
def And(*args):
    """Defines the three valued ``And`` behaviour for a 2-tuple of
     three valued logic values"""
    # 该函数用于定义两个三值逻辑值的三值“与”行为，但是函数体未提供，需要进一步实现。
    # 定义一个函数，用于对比两个区间的条件，返回一个元组
    def reduce_and(cmp_intervala, cmp_intervalb):
        # 检查两个区间是否有一个为 False，如果有，则返回 False
        if cmp_intervala[0] is False or cmp_intervalb[0] is False:
            first = False
        # 检查两个区间是否有一个为 None，如果有，则返回 None
        elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
            first = None
        # 否则，返回 True
        else:
            first = True
        # 同样的方式检查第二个值
        if cmp_intervala[1] is False or cmp_intervalb[1] is False:
            second = False
        elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
            second = None
        else:
            second = True
        # 返回两个值组成的元组
        return (first, second)
    # 使用 reduce 函数，对参数列表 args 应用 reduce_and 函数
    return reduce(reduce_and, args)
# 定义了三值逻辑的“或”操作函数，适用于两个元组的三值逻辑值
def Or(*args):
    """Defines the three valued ``Or`` behaviour for a 2-tuple of
     three valued logic values"""
    # 定义内部函数，实现两个三值逻辑元组的“或”运算
    def reduce_or(cmp_intervala, cmp_intervalb):
        # 如果第一个元组的第一个值为 True 或者第二个元组的第一个值为 True，则结果第一个值为 True
        if cmp_intervala[0] is True or cmp_intervalb[0] is True:
            first = True
        # 如果其中一个元组的第一个值为 None，则结果第一个值为 None
        elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
            first = None
        # 否则结果第一个值为 False
        else:
            first = False

        # 如果第一个元组的第二个值为 True 或者第二个元组的第二个值为 True，则结果第二个值为 True
        if cmp_intervala[1] is True or cmp_intervalb[1] is True:
            second = True
        # 如果其中一个元组的第二个值为 None，则结果第二个值为 None
        elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
            second = None
        # 否则结果第二个值为 False
        else:
            second = False
        return (first, second)
    
    # 使用 functools.reduce 函数，对所有参数使用 reduce_or 函数进行归约操作
    return reduce(reduce_or, args)
```