# `D:\src\scipysrc\sympy\sympy\core\random.py`

```
"""
当在 SymPy 库代码中需要使用随机数时，从这里导入以确保只有一个生成器为 SymPy 工作。
从这里导入的行为应与从 Python 的 random 模块导入时相同。但此处仅包含 SymPy 目前使用的例程。
要使用其他例程，请导入 ``rng`` 并直接访问方法。例如，要捕获生成器的当前状态，请使用 ``rng.getstate()``。

有意地没有从这里导入 Random。如果要控制生成器的状态，请导入 ``seed`` 并调用它，带或不带参数都可以设置状态。

示例
========

>>> from sympy.core.random import random, seed
>>> assert random() < 1
>>> seed(1); a = random()
>>> b = random()
>>> seed(1); c = random()
>>> assert a == c
>>> assert a != b  # 这个可能性很小会失败

"""
# 从 sympy.utilities.iterables 导入 is_sequence 函数
from sympy.utilities.iterables import is_sequence
# 从 sympy.utilities.misc 导入 as_int 函数
from sympy.utilities.misc import as_int

# 导入 random 模块并将其重命名为 _random
import random as _random
# 创建 rng 实例作为随机数生成器
rng = _random.Random()

# 将以下函数和方法赋值为 rng 实例对应的函数和方法
choice = rng.choice
random = rng.random
randint = rng.randint
randrange = rng.randrange
sample = rng.sample
# seed = rng.seed  # 此行被注释掉了，不会执行
shuffle = rng.shuffle
uniform = rng.uniform

# 创建 _assumptions_rng 实例作为另一个独立的随机数生成器
_assumptions_rng = _random.Random()
# 将 _assumptions_rng 的 shuffle 方法赋值给 _assumptions_shuffle
_assumptions_shuffle = _assumptions_rng.shuffle

# 定义 seed 函数用于设置随机数生成器的种子状态
def seed(a=None, version=2):
    rng.seed(a=a, version=version)
    _assumptions_rng.seed(a=a, version=version)

# 定义 random_complex_number 函数用于生成随机复数
def random_complex_number(a=2, b=-1, c=3, d=1, rational=False, tolerance=None):
    """
    返回一个随机复数。

    若要减少命中分支切割或其他问题的机会，我们保证 b <= Im z <= d, a <= Re z <= c

    当 rational 为 True 时，会在指定的容差范围内获得随机数的有理近似。
    """
    from sympy.core.numbers import I
    from sympy.simplify.simplify import nsimplify
    A, B = uniform(a, c), uniform(b, d)
    if not rational:
        return A + I*B
    return (nsimplify(A, rational=True, tolerance=tolerance) +
        I*nsimplify(B, rational=True, tolerance=tolerance))

# 定义 verify_numerically 函数用于数值测试 f 和 g 在参数 z 处的一致性
def verify_numerically(f, g, z=None, tol=1.0e-6, a=2, b=-1, c=3, d=1):
    """
    在数值上测试 f 和 g 在参数 z 处的一致性。

    如果 z 是 None，则测试所有符号。此例程不会测试是否存在精度高于 15 位的浮点数，
    如果有的话，由于舍入误差，结果可能不如预期。

    示例
    ========

    >>> from sympy import sin, cos
    >>> from sympy.abc import x
    >>> from sympy.core.random import verify_numerically as tn
    >>> tn(sin(x)**2 + cos(x)**2, 1, x)
    True
    """
    from sympy.core.symbol import Symbol
    from sympy.core.sympify import sympify
    from sympy.core.numbers import comp
    f, g = (sympify(i) for i in (f, g))
    if z is None:
        z = f.free_symbols | g.free_symbols
    elif isinstance(z, Symbol):
        z = [z]
    # 为 z 中的每个符号生成一个随机复数
    reps = list(zip(z, [random_complex_number(a, b, c, d) for _ in z]))
    # 使用 subs 方法替换表达式 f 中的变量，然后计算数值结果
    z1 = f.subs(reps).n()
    # 使用 subs 方法替换表达式 g 中的变量，然后计算数值结果
    z2 = g.subs(reps).n()
    # 调用 comp 函数比较 z1 和 z2 的数值结果，使用给定的容差 tol
    return comp(z1, z2, tol)
# 定义一个函数，用于数值上测试函数 f 对变量 z 的符号计算导数的准确性
def test_derivative_numerically(f, z, tol=1.0e-6, a=2, b=-1, c=3, d=1):
    """
    Test numerically that the symbolically computed derivative of f
    with respect to z is correct.

    This routine does not test whether there are Floats present with
    precision higher than 15 digits so if there are, your results may
    not be what you expect due to round-off errors.

    Examples
    ========

    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> from sympy.core.random import test_derivative_numerically as td
    >>> td(sin(x), x)
    True
    """
    # 导入需要的数值计算库和函数
    from sympy.core.numbers import comp
    from sympy.core.function import Derivative
    # 生成一个随机复数作为测试点
    z0 = random_complex_number(a, b, c, d)
    # 计算 f 对 z 的符号导数，并在 z=z0 处求值
    f1 = f.diff(z).subs(z, z0)
    # 使用数值方法计算 f 对 z 的导数，并在 z=z0 处求值
    f2 = Derivative(f, z).doit_numerically(z0)
    # 比较两个数值结果是否在指定容差范围内相等，返回比较结果
    return comp(f1.n(), f2.n(), tol)


# 定义一个内部函数，返回一个随机数生成器
def _randrange(seed=None):
    """Return a randrange generator.

    ``seed`` can be

    * None - return randomly seeded generator
    * int - return a generator seeded with the int
    * list - the values to be returned will be taken from the list
      in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.core.random import _randrange
    >>> rr = _randrange()
    >>> rr(1000) # doctest: +SKIP
    999
    >>> rr = _randrange(3)
    >>> rr(1000) # doctest: +SKIP
    238
    >>> rr = _randrange([0, 5, 1, 3, 4])
    >>> rr(3), rr(3)
    (0, 1)
    """
    # 根据不同的 seed 类型返回不同的随机数生成器
    if seed is None:
        return randrange
    elif isinstance(seed, int):
        # 使用指定的整数种子初始化随机数生成器
        rng.seed(seed)
        return randrange
    elif is_sequence(seed):
        seed = list(seed)  # 复制 seed 列表，以免修改原始列表

        def give(a, b=None, seq=seed):
            """内部函数，返回指定范围内的随机数，从 seq 中取值"""
            if b is None:
                a, b = 0, a
            a, b = as_int(a), as_int(b)
            w = b - a
            if w < 1:
                raise ValueError('_randrange got empty range')
            try:
                x = seq.pop()
            except IndexError:
                raise ValueError('_randrange sequence was too short')
            if a <= x < b:
                return x
            else:
                return give(a, b, seq)
        return give
    else:
        raise ValueError('_randrange got an unexpected seed')


# 定义一个内部函数，返回一个 randint 生成器
def _randint(seed=None):
    """Return a randint generator.

    ``seed`` can be

    * None - return randomly seeded generator
    * int - return a generator seeded with the int
    * list - the values to be returned will be taken from the list
      in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.core.random import _randint
    >>> ri = _randint()
    >>> ri(1, 1000) # doctest: +SKIP
    999
    >>> ri = _randint(3)
    >>> ri(1, 1000) # doctest: +SKIP
    238
    >>> ri = _randint([0, 5, 1, 2, 4])
    >>> ri(1, 3), ri(1, 3)
    (1, 2)
    """
    # 根据不同的 seed 类型返回不同的随机整数生成器
    if seed is None:
        return randint
    elif isinstance(seed, int):
        # 使用指定的整数种子初始化随机数生成器
        rng.seed(seed)
        return randint
    else:
        raise ValueError('_randint got an unexpected seed')
    elif is_sequence(seed):
        seed = list(seed)  # make a copy  # 将 seed 转换为列表，以便复制并反转
        seed.reverse()  # 反转列表元素顺序

        def give(a, b, seq=seed):
            a, b = as_int(a), as_int(b)  # 将 a 和 b 转换为整数
            w = b - a  # 计算范围宽度
            if w < 0:
                raise ValueError('_randint got empty range')  # 如果范围为空，则抛出值错误异常
            try:
                x = seq.pop()  # 弹出序列中的一个元素作为随机数 x
            except IndexError:
                raise ValueError('_randint sequence was too short')  # 如果序列太短，则抛出值错误异常
            if a <= x <= b:  # 如果 x 在指定范围内
                return x  # 返回 x
            else:
                return give(a, b, seq)  # 否则递归调用 give 函数
        return give  # 返回内部函数 give 作为结果
    else:
        raise ValueError('_randint got an unexpected seed')  # 如果 seed 不是整数或序列，则抛出值错误异常
```