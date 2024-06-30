# `D:\src\scipysrc\sympy\sympy\assumptions\sathandlers.py`

```
from collections import defaultdict  # 导入 Python 标准库中的 defaultdict 数据结构

from sympy.assumptions.ask import Q  # 导入 SymPy 中用于假设推断的 Q 对象
from sympy.core import (Add, Mul, Pow, Number, NumberSymbol, Symbol)  # 导入 SymPy 核心模块中的多个类
from sympy.core.numbers import ImaginaryUnit  # 导入 SymPy 中的虚数单位
from sympy.functions.elementary.complexes import Abs  # 导入 SymPy 中的复数函数
from sympy.logic.boolalg import (Equivalent, And, Or, Implies)  # 导入 SymPy 中的逻辑运算类
from sympy.matrices.expressions import MatMul  # 导入 SymPy 中的矩阵乘法表达式类

# APIs here may be subject to change
# 此处的 API 可能会发生更改


### Helper functions ###

def allargs(symbol, fact, expr):
    """
    Apply all arguments of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import allargs
    >>> from sympy.abc import x, y
    >>> allargs(x, Q.negative(x) | Q.positive(x), x*y)
    (Q.negative(x) | Q.positive(x)) & (Q.negative(y) | Q.positive(y))

    """
    # 对表达式的所有参数应用给定的事实结构，并返回逻辑与的结果
    return And(*[fact.subs(symbol, arg) for arg in expr.args])


def anyarg(symbol, fact, expr):
    """
    Apply any argument of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import anyarg
    >>> from sympy.abc import x, y
    >>> anyarg(x, Q.negative(x) & Q.positive(x), x*y)
    (Q.negative(x) & Q.positive(x)) | (Q.negative(y) & Q.positive(y))

    """
    # 对表达式的任意参数应用给定的事实结构，并返回逻辑或的结果
    return Or(*[fact.subs(symbol, arg) for arg in expr.args])


def exactlyonearg(symbol, fact, expr):
    """
    Apply exactly one argument of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import exactlyonearg
    >>> from sympy.abc import x, y
    >>> exactlyonearg(x, Q.positive(x), x*y)
    (Q.positive(x) & ~Q.positive(y)) | (Q.positive(y) & ~Q.positive(x))

    """
    # 对表达式的一个参数应用给定的事实结构，确保只有一个参数满足，返回逻辑或的结果
    pred_args = [fact.subs(symbol, arg) for arg in expr.args]
    res = Or(*[And(pred_args[i], *[~lit for lit in pred_args[:i] +
        pred_args[i+1:]]) for i in range(len(pred_args))])
    return res


### Fact registry ###

class ClassFactRegistry:
    """
    Register handlers against classes.

    Explanation
    ===========

    ``register`` method registers the handler function for a class. Here,
    handler function should return a single fact. ``multiregister`` method
    registers the handler function for multiple classes. Here, handler function
    should return a container of multiple facts.

    ``registry(expr)`` returns a set of facts for *expr*.

    Examples
    ========

    Here, we register the facts for ``Abs``.

    >>> from sympy import Abs, Equivalent, Q

    """
    # 用于注册处理器与类之间关系的类
    # register 方法用于为类注册处理函数，该函数返回一个事实
    # multiregister 方法用于为多个类注册处理函数，该函数返回多个事实的容器
    # registry(expr) 方法返回表达式 expr 的一组事实集合
    >>> from sympy.assumptions.sathandlers import ClassFactRegistry
    # 导入 ClassFactRegistry 类，用于管理和查询符号表达式的事实
    >>> reg = ClassFactRegistry()
    # 创建 ClassFactRegistry 的实例 reg，用于注册和存储类的事实处理函数

    >>> @reg.register(Abs)
    # 使用 reg.register 方法注册 Abs 类的事实处理函数
    ... def f1(expr):
    # 定义一个处理函数 f1，接收表达式 expr 作为参数
    ...     return Q.nonnegative(expr)
    # 返回表达式 expr 非负的事实

    >>> @reg.register(Abs)
    # 继续使用 reg.register 方法注册 Abs 类的另一个事实处理函数
    ... def f2(expr):
    # 定义第二个处理函数 f2，接收表达式 expr 作为参数
    ...     arg = expr.args[0]
    # 获取表达式 expr 的第一个参数
    ...     return Equivalent(~Q.zero(arg), ~Q.zero(expr))
    # 返回表达式 expr 的参数不为零等价于表达式 expr 不为零的事实

    Calling the registry with expression returns the defined facts for the
    expression.
    # 使用注册表和表达式调用时，返回表达式的定义事实

    >>> from sympy.abc import x
    # 导入 sympy 库中的符号 x
    >>> reg(Abs(x))
    # 使用注册表 reg 处理 Abs(x) 表达式，返回该表达式的所有已定义事实
    {Q.nonnegative(Abs(x)), Equivalent(~Q.zero(x), ~Q.zero(Abs(x)))}

    Multiple facts can be registered at once by ``multiregister`` method.
    # 可以使用 multiregister 方法一次注册多个事实处理函数

    >>> reg2 = ClassFactRegistry()
    # 创建另一个 ClassFactRegistry 实例 reg2

    >>> @reg2.multiregister(Abs)
    # 使用 reg2.multiregister 方法注册 Abs 类的多个事实处理函数
    ... def _(expr):
    # 定义一个匿名函数，接收表达式 expr 作为参数
    ...     arg = expr.args[0]
    # 获取表达式 expr 的第一个参数
    ...     return [Q.even(arg) >> Q.even(expr), Q.odd(arg) >> Q.odd(expr)]
    # 返回一个列表，其中包含对参数为偶数和奇数情况下 Abs(expr) 的定义事实

    >>> reg2(Abs(x))
    # 使用注册表 reg2 处理 Abs(x) 表达式，返回该表达式的所有已定义事实
    {Implies(Q.even(x), Q.even(Abs(x))), Implies(Q.odd(x), Q.odd(Abs(x)))}

    """
    def __init__(self):
        # 初始化函数，创建 ClassFactRegistry 实例时调用
        self.singlefacts = defaultdict(frozenset)
        # 创建一个 defaultdict，用于存储单个类事实处理函数的集合
        self.multifacts = defaultdict(frozenset)
        # 创建一个 defaultdict，用于存储多个类事实处理函数的集合

    def register(self, cls):
        # 注册单个类的事实处理函数
        def _(func):
            # 定义一个闭包函数，接收事实处理函数 func 作为参数
            self.singlefacts[cls] |= {func}
            # 将 func 添加到 cls 类的事实处理函数集合中
            return func
            # 返回事实处理函数 func
        return _

    def multiregister(self, *classes):
        # 一次注册多个类的事实处理函数
        def _(func):
            # 定义一个闭包函数，接收事实处理函数 func 作为参数
            for cls in classes:
                self.multifacts[cls] |= {func}
                # 将 func 添加到每个类 cls 的事实处理函数集合中
            return func
            # 返回事实处理函数 func
        return _

    def __getitem__(self, key):
        # 获取指定键 key 对应的值
        ret1 = self.singlefacts[key]
        # 获取单个类事实处理函数集合中键为 key 的值
        for k in self.singlefacts:
            if issubclass(key, k):
                ret1 |= self.singlefacts[k]
                # 如果 key 是 k 的子类，则将 k 类的事实处理函数集合合并到 ret1 中

        ret2 = self.multifacts[key]
        # 获取多个类事实处理函数集合中键为 key 的值
        for k in self.multifacts:
            if issubclass(key, k):
                ret2 |= self.multifacts[k]
                # 如果 key 是 k 的子类，则将 k 类的事实处理函数集合合并到 ret2 中

        return ret1, ret2
        # 返回两个集合 ret1 和 ret2

    def __call__(self, expr):
        # 当实例被调用时执行的方法
        ret = set()
        # 初始化一个空集合 ret

        handlers1, handlers2 = self[type(expr)]
        # 获取表达式 expr 的单个类和多个类的事实处理函数集合

        ret.update(h(expr) for h in handlers1)
        # 将 handlers1 中每个处理函数 h 对表达式 expr 执行并将结果添加到 ret 中
        for h in handlers2:
            ret.update(h(expr))
            # 对 handlers2 中的每个处理函数 h 对表达式 expr 执行并将结果添加到 ret 中
        return ret
        # 返回包含所有执行结果的集合
class_fact_registry = ClassFactRegistry()

### Class fact registration ###

# 创建一个 ClassFactRegistry 的实例，用于注册类相关的事实
x = Symbol('x')
# 创建一个符号 x，用于表示表达式中的符号变量

## Abs ##

# 将 Abs 类注册到 class_fact_registry
@class_fact_registry.multiregister(Abs)
def _(expr):
    # 定义 Abs 函数的多个事实
    arg = expr.args[0]
    return [Q.nonnegative(expr),  # 表达式非负
            Equivalent(~Q.zero(arg), ~Q.zero(expr)),  # 表达式非零等价于参数非零
            Q.even(arg) >> Q.even(expr),  # 参数为偶数则表达式为偶数
            Q.odd(arg) >> Q.odd(expr),  # 参数为奇数则表达式为奇数
            Q.integer(arg) >> Q.integer(expr),  # 参数为整数则表达式为整数
            ]


### Add ##

# 将 Add 类注册到 class_fact_registry 的多重注册
@class_fact_registry.multiregister(Add)
def _(expr):
    # 定义 Add 函数的多个事实
    return [allargs(x, Q.positive(x), expr) >> Q.positive(expr),  # 所有参数为正数则表达式为正数
            allargs(x, Q.negative(x), expr) >> Q.negative(expr),  # 所有参数为负数则表达式为负数
            allargs(x, Q.real(x), expr) >> Q.real(expr),  # 所有参数为实数则表达式为实数
            allargs(x, Q.rational(x), expr) >> Q.rational(expr),  # 所有参数为有理数则表达式为有理数
            allargs(x, Q.integer(x), expr) >> Q.integer(expr),  # 所有参数为整数则表达式为整数
            exactlyonearg(x, ~Q.integer(x), expr) >> ~Q.integer(expr),  # 恰好一个参数非整数则表达式非整数
            ]

# 将 Add 类注册到 class_fact_registry 的单一注册
@class_fact_registry.register(Add)
def _(expr):
    # 定义 Add 函数的一个事实
    allargs_real = allargs(x, Q.real(x), expr)
    onearg_irrational = exactlyonearg(x, Q.irrational(x), expr)
    return Implies(allargs_real, Implies(onearg_irrational, Q.irrational(expr)))


### Mul ###

# 将 Mul 类注册到 class_fact_registry 的多重注册
@class_fact_registry.multiregister(Mul)
def _(expr):
    # 定义 Mul 函数的多个事实
    return [Equivalent(Q.zero(expr), anyarg(x, Q.zero(x), expr)),  # 表达式为零等价于存在参数为零
            allargs(x, Q.positive(x), expr) >> Q.positive(expr),  # 所有参数为正数则表达式为正数
            allargs(x, Q.real(x), expr) >> Q.real(expr),  # 所有参数为实数则表达式为实数
            allargs(x, Q.rational(x), expr) >> Q.rational(expr),  # 所有参数为有理数则表达式为有理数
            allargs(x, Q.integer(x), expr) >> Q.integer(expr),  # 所有参数为整数则表达式为整数
            exactlyonearg(x, ~Q.rational(x), expr) >> ~Q.integer(expr),  # 恰好一个参数非有理数则表达式非整数
            allargs(x, Q.commutative(x), expr) >> Q.commutative(expr),  # 所有参数满足交换律则表达式满足交换律
            ]

# 将 Mul 类注册到 class_fact_registry 的单一注册
@class_fact_registry.register(Mul)
def _(expr):
    # 定义 Mul 函数的一个事实
    allargs_prime = allargs(x, Q.prime(x), expr)
    return Implies(allargs_prime, ~Q.prime(expr))

# 将 Mul 类注册到 class_fact_registry 的单一注册
@class_fact_registry.register(Mul)
def _(expr):
    # 定义 Mul 函数的一个事实
    allargs_imag_or_real = allargs(x, Q.imaginary(x) | Q.real(x), expr)
    onearg_imaginary = exactlyonearg(x, Q.imaginary(x), expr)
    return Implies(allargs_imag_or_real, Implies(onearg_imaginary, Q.imaginary(expr)))

# 将 Mul 类注册到 class_fact_registry 的单一注册
@class_fact_registry.register(Mul)
def _(expr):
    # 定义 Mul 函数的一个事实
    allargs_real = allargs(x, Q.real(x), expr)
    onearg_irrational = exactlyonearg(x, Q.irrational(x), expr)
    return Implies(allargs_real, Implies(onearg_irrational, Q.irrational(expr)))

# 将 Mul 类注册到 class_fact_registry 的单一注册
@class_fact_registry.register(Mul)
def _(expr):
    # 定义 Mul 函数的一个事实
    allargs_integer = allargs(x, Q.integer(x), expr)
    anyarg_even = anyarg(x, Q.even(x), expr)
    # 返回一个条件表达式，当所有参数都是整数时成立，当任意参数为偶数时等价于 Q.even(expr)
    return Implies(allargs_integer, Equivalent(anyarg_even, Q.even(expr)))
### MatMul ###

# 注册一个函数，用于处理 MatMul 类型的表达式
@class_fact_registry.register(MatMul)
def _(expr):
    # 创建一个条件，要求所有参数均为平方数
    allargs_square = allargs(x, Q.square(x), expr)
    # 创建一个条件，要求所有参数均为可逆元素
    allargs_invertible = allargs(x, Q.invertible(x), expr)
    # 返回一个蕴含式，表明所有参数均为平方数时，表达式可逆当且仅当所有参数均为可逆元素
    return Implies(allargs_square, Equivalent(Q.invertible(expr), allargs_invertible))


### Pow ###

# 多重注册 Pow 类型的表达式处理函数
@class_fact_registry.multiregister(Pow)
def _(expr):
    # 获取 Pow 表达式的底数和指数
    base, exp = expr.base, expr.exp
    # 返回一个列表，包含不同的蕴含式，根据底数和指数的特性确定表达式的性质
    return [
        (Q.real(base) & Q.even(exp) & Q.nonnegative(exp)) >> Q.nonnegative(expr),
        (Q.nonnegative(base) & Q.odd(exp) & Q.nonnegative(exp)) >> Q.nonnegative(expr),
        (Q.nonpositive(base) & Q.odd(exp) & Q.nonnegative(exp)) >> Q.nonpositive(expr),
        Equivalent(Q.zero(expr), Q.zero(base) & Q.positive(exp))
    ]


### Numbers ###

# 旧的前提获取器字典
_old_assump_getters = {
    Q.positive: lambda o: o.is_positive,
    Q.zero: lambda o: o.is_zero,
    Q.negative: lambda o: o.is_negative,
    Q.rational: lambda o: o.is_rational,
    Q.irrational: lambda o: o.is_irrational,
    Q.even: lambda o: o.is_even,
    Q.odd: lambda o: o.is_odd,
    Q.imaginary: lambda o: o.is_imaginary,
    Q.prime: lambda o: o.is_prime,
    Q.composite: lambda o: o.is_composite,
}

# 多重注册处理 Number、NumberSymbol、ImaginaryUnit 类型的表达式函数
@class_fact_registry.multiregister(Number, NumberSymbol, ImaginaryUnit)
def _(expr):
    ret = []
    # 遍历旧的前提获取器字典
    for p, getter in _old_assump_getters.items():
        # 获取表达式 expr 满足前提 p 的值
        pred = p(expr)
        # 获取表达式 expr 的属性 getter 的值
        prop = getter(expr)
        # 如果属性不为 None，添加等价式到结果列表中
        if prop is not None:
            ret.append(Equivalent(pred, prop))
    # 返回结果列表
    return ret
```