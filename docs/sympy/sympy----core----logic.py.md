# `D:\src\scipysrc\sympy\sympy\core\logic.py`

```
# 导入必要的模块和类型定义
from __future__ import annotations
from typing import Optional

# 模糊布尔类型定义
FuzzyBool = Optional[bool]


def _torf(args):
    """根据参数列表的逻辑值返回 True、False 或 None。

    >>> from sympy.core.logic import _torf
    >>> _torf((True, True))
    True
    >>> _torf((False, False))
    False
    >>> _torf((True, False))
    """
    sawT = sawF = False  # 初始化 True 和 False 的标志位
    for a in args:
        if a is True:
            if sawF:
                return  # 如果之前出现过 False，则返回 None
            sawT = True  # 设置 True 的标志位为 True
        elif a is False:
            if sawT:
                return  # 如果之前出现过 True，则返回 None
            sawF = True  # 设置 False 的标志位为 True
        else:
            return  # 如果参数不是 True 或 False，则返回 None
    return sawT  # 返回最终的 True 标志位


def _fuzzy_group(args, quick_exit=False):
    """根据参数列表的逻辑值返回 True、False 或 None。

    如果参数列表中有 None，则返回 None；如果参数列表中有多个 False，
    根据 quick_exit 参数的设置决定是返回 False 还是 None。

    Examples
    ========

    >>> from sympy.core.logic import _fuzzy_group

    默认情况下，多个 False 表示返回 False：

    >>> _fuzzy_group([False, False, True])
    False

    如果希望当第二个 False 出现时返回 None，则设置 quick_exit 为 True：

    >>> _fuzzy_group([False, False, True], quick_exit=True)

    如果只有一个 False 出现，则返回 False：

    >>> _fuzzy_group([False, True, True], quick_exit=True)
    False

    """
    saw_other = False  # 初始化其它值（非 True 或 False）的标志位为 False
    for a in args:
        if a is True:
            continue  # 如果是 True，则继续循环
        if a is None:
            return  # 如果是 None，则直接返回 None
        if quick_exit and saw_other:
            return  # 如果 quick_exit 为 True 并且之前已经看到其它值，则返回 None
        saw_other = True  # 设置其它值标志位为 True
    return not saw_other  # 返回最终的结果，如果没有其它值，则返回 True；否则返回 False


def fuzzy_bool(x):
    """根据输入 x 返回 True、False 或 None。

    与 bool(x) 不同的是，fuzzy_bool 允许返回 None 和非 False 的值也被转换为 None。

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_bool
    >>> from sympy.abc import x
    >>> fuzzy_bool(x), fuzzy_bool(None)
    (None, None)
    >>> bool(x), bool(None)
    (True, False)

    """
    if x is None:
        return None  # 如果 x 是 None，则返回 None
    if x in (True, False):
        return bool(x)  # 如果 x 是 True 或 False，则返回其布尔值
    # 初始化返回值为 True
    rv = True
    # 遍历参数列表 args 中的每个元素 ai
    for ai in args:
        # 将 ai 转换为模糊布尔值
        ai = fuzzy_bool(ai)
        # 如果 ai 为 False，则直接返回 False
        if ai is False:
            return False
        # 如果 rv 仍为 True，更新 rv 为 ai 的值
        if rv:  # 如果 rv 为 True，将会停止更新如果发现一个 None 被捕获
            rv = ai
    # 返回最终的 rv 值，即可能是 True 或者是最后一个非 False 的 ai 的值
    return rv
def fuzzy_not(v):
    """
    Not in fuzzy logic

    Return None if `v` is None else `not v`.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_not
    >>> fuzzy_not(True)
    False
    >>> fuzzy_not(None)
    >>> fuzzy_not(False)
    True

    """
    # 如果输入值 v 是 None，则返回 None
    if v is None:
        return v
    else:
        # 否则返回逻辑非的结果
        return not v


def fuzzy_or(args):
    """
    Or in fuzzy logic. Returns True (any True), False (all False), or None

    See the docstrings of fuzzy_and and fuzzy_not for more info.  fuzzy_or is
    related to the two by the standard De Morgan's law.

    >>> from sympy.core.logic import fuzzy_or
    >>> fuzzy_or([True, False])
    True
    >>> fuzzy_or([True, None])
    True
    >>> fuzzy_or([False, False])
    False
    >>> print(fuzzy_or([False, None]))
    None

    """
    # 初始化返回值为 False
    rv = False
    # 遍历输入的参数列表 args
    for ai in args:
        # 将每个元素 ai 转换为模糊逻辑布尔值
        ai = fuzzy_bool(ai)
        # 如果 ai 是 True，则直接返回 True
        if ai is True:
            return True
        # 如果 rv 是 False，则更新为 ai 的值（可能为 True 或 False）
        if rv is False:  # 如果 rv 是 False，一旦捕获到 None，就不再更新
            rv = ai
    # 返回最终的 rv，可能为 False 或者是最后一个非 None 的 ai 值
    return rv


def fuzzy_xor(args):
    """Return None if any element of args is not True or False, else
    True (if there are an odd number of True elements), else False."""
    # 初始化计数器 t 和 f
    t = f = 0
    # 遍历输入的参数列表 args
    for a in args:
        # 将每个元素 a 转换为模糊逻辑布尔值
        ai = fuzzy_bool(a)
        # 如果 ai 是 True，则 t 加一
        if ai:
            t += 1
        # 如果 ai 是 False，则 f 加一
        elif ai is False:
            f += 1
        else:
            # 如果 ai 是 None，则直接返回 None
            return
    # 返回 True，如果 t 是奇数，否则返回 False
    return t % 2 == 1


def fuzzy_nand(args):
    """Return False if all args are True, True if they are all False,
    else None."""
    # 使用模糊逻辑中的非操作来计算与非结果
    return fuzzy_not(fuzzy_and(args))


class Logic:
    """Logical expression"""
    # {} 'op' -> LogicClass
    # 类变量 op_2class 是一个字典，用于存储逻辑操作符对应的逻辑类
    op_2class: dict[str, type[Logic]] = {}

    def __new__(cls, *args):
        # 创建新的逻辑表达式对象，初始化其参数
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __getnewargs__(self):
        return self.args

    def __hash__(self):
        # 计算逻辑表达式对象的哈希值
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(a, b):
        # 比较两个逻辑表达式对象是否相等
        if not isinstance(b, type(a)):
            return False
        else:
            return a.args == b.args

    def __ne__(a, b):
        # 比较两个逻辑表达式对象是否不相等
        if not isinstance(b, type(a)):
            return True
        else:
            return a.args != b.args

    def __lt__(self, other):
        # 判断当前逻辑表达式对象是否小于另一个对象
        if self.__cmp__(other) == -1:
            return True
        return False

    def __cmp__(self, other):
        # 比较两个逻辑表达式对象的大小关系
        if type(self) is not type(other):
            a = str(type(self))
            b = str(type(other))
        else:
            a = self.args
            b = other.args
        return (a > b) - (a < b)

    def __str__(self):
        # 返回逻辑表达式对象的字符串表示形式
        return '%s(%s)' % (self.__class__.__name__,
                           ', '.join(str(a) for a in self.args))

    __repr__ = __str__

    @staticmethod
    def fromstring(text):
        """从字符串中构建逻辑表达式，其中 & 和 | 周围有空格，但!后面没有空格。

           例如：

           !a & b | c
        """
        lexpr = None  # 当前逻辑表达式
        schedop = None  # 待定的操作
        for term in text.split():
            # 操作符号
            if term in '&|':
                if schedop is not None:
                    raise ValueError(
                        'double op forbidden: "%s %s"' % (term, schedop))
                if lexpr is None:
                    raise ValueError(
                        '%s cannot be in the beginning of expression' % term)
                schedop = term
                continue
            if '&' in term or '|' in term:
                raise ValueError('& 和 | 必须在它们周围有空格')
            if term[0] == '!':
                if len(term) == 1:
                    raise ValueError('在 "!" 后面不要包含空格')
                term = Not(term[1:])

            # 已经预定的操作，例如 '&'
            if schedop:
                lexpr = Logic.op_2class[schedop](lexpr, term)
                schedop = None
                continue

            # 这应该是原子表达式
            if lexpr is not None:
                raise ValueError(
                    '缺少 "%s" 和 "%s" 之间的操作符' % (lexpr, term))

            lexpr = term

        # 检查最终状态是否正确
        if schedop is not None:
            raise ValueError('在 "%s" 中存在表达式提前结束的情况' % text)
        if lexpr is None:
            raise ValueError('"%s" 是空的' % text)

        # 现在一切看起来都很好
        return lexpr
class AndOr_Base(Logic):
    # AndOr_Base 类继承自 Logic 类，用于实现逻辑运算的基础功能

    def __new__(cls, *args):
        # 定义 __new__ 方法用于创建类的新实例，接受任意数量的参数 args

        bargs = []
        for a in args:
            if a == cls.op_x_notx:
                # 如果 a 等于类属性 op_x_notx，则直接返回 a
                return a
            elif a == (not cls.op_x_notx):
                # 如果 a 等于 not cls.op_x_notx，则跳过此参数
                continue    # 跳过这个参数
            bargs.append(a)

        args = sorted(set(cls.flatten(bargs)), key=hash)
        # 将 bargs 中的元素去重后转为列表并排序，然后赋值给 args

        for a in args:
            if Not(a) in args:
                # 如果 Not(a) 在 args 中，则返回类属性 op_x_notx
                return cls.op_x_notx

        if len(args) == 1:
            # 如果 args 中只有一个元素
            return args.pop()
        elif len(args) == 0:
            # 如果 args 中没有元素
            return not cls.op_x_notx

        return Logic.__new__(cls, *args)
        # 调用 Logic 类的 __new__ 方法创建新实例，并传入参数 args

    @classmethod
    def flatten(cls, args):
        # 类方法 flatten，用于快速展开 And 和 Or 类的参数

        args_queue = list(args)
        res = []

        while True:
            try:
                arg = args_queue.pop(0)
            except IndexError:
                break
            if isinstance(arg, Logic):
                if isinstance(arg, cls):
                    args_queue.extend(arg.args)
                    continue
            res.append(arg)

        args = tuple(res)
        return args
        # 返回展开后的参数元组


class And(AndOr_Base):
    # And 类继承自 AndOr_Base 类，用于实现逻辑与运算
    op_x_notx = False
    # 设置类属性 op_x_notx 为 False，表示逻辑与的默认状态为真

    def _eval_propagate_not(self):
        # 定义 _eval_propagate_not 方法，处理逻辑与运算中的取非操作
        # !(a&b&c ...) == !a | !b | !c ...
        return Or(*[Not(a) for a in self.args])
        # 返回一个逻辑或操作，对 self.args 中的每个元素取非后组成列表，并传入 Or 类中

    def expand(self):
        # 定义 expand 方法，用于展开逻辑与运算

        # 首先定位逻辑或
        for i, arg in enumerate(self.args):
            if isinstance(arg, Or):
                arest = self.args[:i] + self.args[i + 1:]

                orterms = [And(*(arest + (a,))) for a in arg.args]
                for j in range(len(orterms)):
                    if isinstance(orterms[j], Logic):
                        orterms[j] = orterms[j].expand()

                res = Or(*orterms)
                return res
                # 如果找到逻辑或，则展开逻辑与运算

        return self
        # 如果没有找到逻辑或，则返回原实例


class Or(AndOr_Base):
    # Or 类继承自 AndOr_Base 类，用于实现逻辑或运算
    op_x_notx = True
    # 设置类属性 op_x_notx 为 True，表示逻辑或的默认状态为假

    def _eval_propagate_not(self):
        # 定义 _eval_propagate_not 方法，处理逻辑或运算中的取非操作
        # !(a|b|c ...) == !a & !b & !c ...
        return And(*[Not(a) for a in self.args])
        # 返回一个逻辑与操作，对 self.args 中的每个元素取非后组成列表，并传入 And 类中


class Not(Logic):
    # Not 类继承自 Logic 类，用于实现逻辑非运算

    def __new__(cls, arg):
        # 定义 __new__ 方法用于创建类的新实例，接受参数 arg

        if isinstance(arg, str):
            return Logic.__new__(cls, arg)
            # 如果 arg 是字符串，则调用 Logic 类的 __new__ 方法创建新实例

        elif isinstance(arg, bool):
            return not arg
            # 如果 arg 是布尔值，则返回其取反值

        elif isinstance(arg, Not):
            return arg.args[0]
            # 如果 arg 是 Not 类的实例，则返回其参数的第一个元素

        elif isinstance(arg, Logic):
            # 如果 arg 是 Logic 类的实例
            # XXX this is a hack to expand right from the beginning
            arg = arg._eval_propagate_not()
            return arg
            # 对参数进行取非操作后返回

        else:
            raise ValueError('Not: unknown argument %r' % (arg,))
            # 如果 arg 类型不在预期范围内，则抛出 ValueError 异常

    @property
    def arg(self):
        return self.args[0]
        # 返回 Not 类的参数的第一个元素


Logic.op_2class['&'] = And
Logic.op_2class['|'] = Or
Logic.op_2class['!'] = Not
# 将逻辑运算符与对应的类关联起来，存储在 Logic 类的 op_2class 字典中
```