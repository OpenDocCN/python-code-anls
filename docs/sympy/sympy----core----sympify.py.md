# `D:\src\scipysrc\sympy\sympy\core\sympify.py`

```
# 从将对象转换为 SymPy 内部格式的函数 sympify 中导入类型提示
from __future__ import annotations
from typing import Any, Callable

# 导入 SymPy 中用于多精度数学运算的 mpmath 库
import mpmath.libmp as mlib

# 从 inspect 模块中导入获取方法解析顺序的函数 getmro
from inspect import getmro

# 导入字符串处理相关的模块
import string

# 从 sympy.core.random 模块中导入 choice 函数
from sympy.core.random import choice

# 从当前包中的 parameters 模块中导入 global_parameters 对象
from .parameters import global_parameters

# 从 sympy.utilities.iterables 模块中导入 iterable 函数
from sympy.utilities.iterables import iterable

# 定义一个自定义异常类 SympifyError，继承自 ValueError
class SympifyError(ValueError):
    def __init__(self, expr, base_exc=None):
        self.expr = expr
        self.base_exc = base_exc

    def __str__(self):
        if self.base_exc is None:
            return "SympifyError: %r" % (self.expr,)
        return ("Sympify of expression '%s' failed, because of exception being "
            "raised:\n%s: %s" % (self.expr, self.base_exc.__class__.__name__,
            str(self.base_exc)))

# 定义一个字典 converter，用于存储类型到转换函数的映射关系
converter: dict[type[Any], Callable[[Any], Basic]] = {}

# 定义一个字典 _sympy_converter，用于存储 SymPy 自身定义的类型到转换函数的映射关系
_sympy_converter: dict[type[Any], Callable[[Any], Basic]] = {}

# 定义一个别名 _external_converter，指向 converter 字典，以提高库中代码的清晰度
_external_converter = converter

# 定义一个特性类 CantSympify，用于混入到其它类中，阻止其实例被 sympify 转换
class CantSympify:
    """
    Mix in this trait to a class to disallow sympification of its instances.

    Examples
    ========

    >>> from sympy import sympify
    >>> from sympy.core.sympify import CantSympify

    >>> class Something(dict):
    ...     pass
    ...
    >>> sympify(Something())
    {}

    >>> class Something(dict, CantSympify):
    ...     pass
    ...
    >>> sympify(Something())
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: {}

    """

    __slots__ = ()

# 定义一个内部函数 _is_numpy_instance，用于检查对象是否为 numpy 模块的实例
def _is_numpy_instance(a):
    """
    Checks if an object is an instance of a type from the numpy module.
    """
    # 避免不必要地导入 NumPy，检查整个 __mro__ 确保任何基类是 numpy 类型
    return any(type_.__module__ == 'numpy'
               for type_ in type(a).__mro__)

# 定义一个内部函数 _convert_numpy_types，用于将 numpy 数据类型转换为适当的 SymPy 类型
def _convert_numpy_types(a, **sympify_args):
    """
    Converts a numpy datatype input to an appropriate SymPy type.
    """
    import numpy as np
    if not isinstance(a, np.floating):
        if np.iscomplex(a):
            return _sympy_converter[complex](a.item())
        else:
            return sympify(a.item(), **sympify_args)
    else:
        from .numbers import Float
        # 获取浮点数的精度信息并转换为 SymPy 中的 Float 类型
        prec = np.finfo(a).nmant + 1
        p, q = a.as_integer_ratio()
        a = mlib.from_rational(p, q, prec)
        return Float(a, precision=prec)

# 定义函数 sympify，用于将任意表达式转换为 SymPy 中可以使用的类型
def sympify(a, locals=None, convert_xor=True, strict=False, rational=False,
        evaluate=None):
    """
    Converts an arbitrary expression to a type that can be used inside SymPy.

    Explanation
    ===========

    It will convert Python ints into instances of :class:`~.Integer`, floats
    into instances of :class:`~.Float`, etc. It is also able to coerce
    """
    # sympify 函数用于将输入的字符串或数字转换为 Sympy 中的表达式对象。
    # 如果输入已经是 Sympy 可识别的类型，则直接返回该值。
    # 该函数可以处理整数、实数、科学计数法表示的数值，返回的表达式对象具有相应的属性，如 is_integer 和 is_real。
    
    >>> from sympy import sympify
    
    >>> sympify(2).is_integer
    True
    >>> sympify(2).is_real
    True
    
    >>> sympify(2.0).is_real
    True
    >>> sympify("2.0").is_real
    True
    >>> sympify("2e-45").is_real
    True
    
    # 如果无法将输入字符串解析为有效的表达式对象，则会引发 SympifyError。
    # 例如，对于 "x***2" 这样的表达式会引发 SympifyError。
    
    >>> sympify("x***2")
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: "could not parse 'x***2'"
    
    # 当尝试使用 sympify 解析非 Python 语法时，也会引发 SympifyError。
    # 例如，"2x+1" 这样的表达式会导致 SympifyError。
    
    >>> sympify("2x+1")
    Traceback (most recent call last):
    ...
    SympifyError: Sympify of expression 'could not parse '2x+1'' failed
    
    # 若要解析非 Python 语法，可以使用 sympy.parsing.sympy_parser 中的 parse_expr 函数。
    
    >>> from sympy.parsing.sympy_parser import parse_expr
    >>> parse_expr("2x+1", transformations="all")
    2*x + 1
    
    # 有关 transformations 的更多详细信息，请参阅 sympy.parsing.sympy_parser.parse_expr 函数的文档。
    
    # 在执行 sympify 过程中，可以通过 locals 参数指定额外的局部变量和函数，这些变量和函数可以在解析字符串时访问。
    # 例如，可以将自定义的函数 bitcount 或类似的符号如 O 加入 locals 字典中，以确保能够正确识别和处理。
    
    >>> s = 'bitcount(42)'
    >>> sympify(s)
    bitcount(42)
    >>> sympify("O(x)")
    O(x)
    >>> sympify("O + 1")
    Traceback (most recent call last):
    ...
    TypeError: unbound method...
    
    # 若要使 bitcount 能够被正确识别，可以将其导入到一个命名空间字典中并通过 locals 参数传递。
    
    >>> ns = {}
    >>> exec('from sympy.core.evalf import bitcount', ns)
    >>> sympify(s, locals=ns)
    6
    
    # 同样地，对于符号 O，也可以通过不同的方式在 locals 字典中进行识别，确保它被正确解析为 Symbol 对象。
    
    >>> from sympy import Symbol
    >>> ns["O"] = Symbol("O")  # 方法 1
    >>> exec('from sympy.abc import O', ns)  # 方法 2
    >>> ns.update(dict(O=Symbol("O")))  # 方法 3
    >>> sympify("O + 1", locals=ns)
    O + 1
    
    # 如果希望所有单字母和希腊字母变量都被识别为符号，则可以使用私有变量 _clash1 和相关的冲突符号字典。
    ``_clash2`` (the multi-letter Greek names) or ``_clash`` (both single and
    multi-letter names that are defined in ``abc``).


    >>> from sympy.abc import _clash1
    >>> set(_clash1)  # if this fails, see issue #23903
    {'E', 'I', 'N', 'O', 'Q', 'S'}


    >>> sympify('I & Q', _clash1)
    I & Q


    Strict
    ------

    If the option ``strict`` is set to ``True``, only the types for which an
    explicit conversion has been defined are converted. In the other
    cases, a SympifyError is raised.


    >>> print(sympify(None))
    None
    >>> sympify(None, strict=True)
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: None


    .. deprecated:: 1.6

       ``sympify(obj)`` automatically falls back to ``str(obj)`` when all
       other conversion methods fail, but this is deprecated. ``strict=True``
       will disable this deprecated behavior. See
       :ref:`deprecated-sympify-string-fallback`.


    Evaluation
    ----------

    If the option ``evaluate`` is set to ``False``, then arithmetic and
    operators will be converted into their SymPy equivalents and the
    ``evaluate=False`` option will be added. Nested ``Add`` or ``Mul`` will
    be denested first. This is done via an AST transformation that replaces
    operators with their SymPy equivalents, so if an operand redefines any
    of those operations, the redefined operators will not be used. If
    argument a is not a string, the mathematical expression is evaluated
    before being passed to sympify, so adding ``evaluate=False`` will still
    return the evaluated result of expression.


    >>> sympify('2**2 / 3 + 5')
    19/3
    >>> sympify('2**2 / 3 + 5', evaluate=False)
    2**2/3 + 5
    >>> sympify('4/2+7', evaluate=True)
    9
    >>> sympify('4/2+7', evaluate=False)
    4/2 + 7
    >>> sympify(4/2+7, evaluate=False)
    9.00000000000000


    Extending
    ---------

    To extend ``sympify`` to convert custom objects (not derived from ``Basic``),
    just define a ``_sympy_`` method to your class. You can do that even to
    classes that you do not own by subclassing or adding the method at runtime.


    >>> from sympy import Matrix
    >>> class MyList1(object):
    ...     def __iter__(self):
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    ...     def _sympy_(self): return Matrix(self)
    >>> sympify(MyList1())
    Matrix([
    [1],
    [2]])


    If you do not have control over the class definition you could also use the
    ``converter`` global dictionary. The key is the class and the value is a
    function that takes a single argument and returns the desired SymPy
    object, e.g. ``converter[MyList] = lambda x: Matrix(x)``.


    >>> class MyList2(object):   # XXX Do not do this if you control the class!
    ...     def __iter__(self):  #     Use _sympy_!
    ...         yield 1
    ...         yield 2
    ...         return
    # 定义了一个特殊方法 __getitem__，用于通过索引访问 MyList2 对象的元素，返回列表形式中的第 i 个元素
    def __getitem__(self, i): return list(self)[i]
    # 获取对象 a 的 '__sympy__' 属性的值，如果不存在则返回 None
    is_sympy = getattr(a, '__sympy__', None)
    # 如果 __sympy__ 属性为 True，则直接返回 a，表示 a 已经是符号表达式
    if is_sympy is True:
        return a
    # 如果 __sympy__ 属性不为 None 且 strict 参数为 False，则直接返回 a
    elif is_sympy is not None:
        if not strict:
            return a
        else:
            # 如果 strict 参数为 True，则抛出 SympifyError 异常，表示无法符号化 a
            raise SympifyError(a)

    # 如果 a 是 CantSympify 类型的实例，则抛出 SympifyError 异常，表示无法符号化 a
    if isinstance(a, CantSympify):
        raise SympifyError(a)

    # 获取对象 a 的 "__class__" 属性
    cls = getattr(a, "__class__", None)

    # 检查类的方法解析顺序 (Method Resolution Order, MRO) 中的每个超类
    for superclass in getmro(cls):
        # 首先查找用户定义的转换器
        conv = _external_converter.get(superclass)
        if conv is None:
            # 如果找不到用户定义的转换器，则查找 SymPy 定义的转换器
            conv = _sympy_converter.get(superclass)
        # 如果找到合适的转换器，则使用转换器将 a 转换为符号表达式并返回
        if conv is not None:
            return conv(a)

    # 如果 a 的类是 NoneType，并且 strict 参数为 True，则抛出 SympifyError 异常
    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a

    # 如果 evaluate 参数为 None，则使用全局参数中的 evaluate 设置
    if evaluate is None:
        evaluate = global_parameters.evaluate

    # 支持基本的 numpy 数据类型转换
    if _is_numpy_instance(a):
        import numpy as np
        # 如果 a 是 numpy 标量，则调用 _convert_numpy_types 函数进行转换
        if np.isscalar(a):
            return _convert_numpy_types(a, locals=locals,
                convert_xor=convert_xor, strict=strict, rational=rational,
                evaluate=evaluate)

    # 获取对象 a 的 '_sympy_' 属性的值
    _sympy_ = getattr(a, "_sympy_", None)
    # 如果 _sympy_ 属性不为 None，则调用 a 的 '_sympy_' 方法并返回结果
    if _sympy_ is not None:
        return a._sympy_()

    # 如果 strict 参数为 False，则尝试进行其他非严格模式下的转换操作
    if not strict:
        # 在执行 float/int 转换之前，先检查是否为 numpy 数组，避免影响转换顺序
        flat = getattr(a, "flat", None)
        if flat is not None:
            shape = getattr(a, "shape", None)
            if shape is not None:
                from sympy.tensor.array import Array
                # 使用 sympy 的 Array 类处理 numpy 数组，返回符号表达式
                return Array(a.flat, a.shape)  # works with e.g. NumPy arrays

    # 如果 a 不是字符串类型
    if not isinstance(a, str):
        # 如果 a 是 numpy 实例
        if _is_numpy_instance(a):
            import numpy as np
            assert not isinstance(a, np.number)
            # 如果 a 是 numpy.ndarray 类型
            if isinstance(a, np.ndarray):
                # 对标量数组（维度为零的数组）调用 sympify 函数进行符号化
                if a.ndim == 0:
                    try:
                        return sympify(a.item(),
                                       locals=locals,
                                       convert_xor=convert_xor,
                                       strict=strict,
                                       rational=rational,
                                       evaluate=evaluate)
                    except SympifyError:
                        pass
        # 如果 a 具有 '__float__' 方法，则尝试将其转换为 float 类型并返回符号表达式
        elif hasattr(a, '__float__'):
            return sympify(float(a))
        # 如果 a 具有 '__int__' 方法，则尝试将其转换为 int 类型并返回符号表达式
        elif hasattr(a, '__int__'):
            return sympify(int(a))

    # 如果 strict 参数为 True，则最终无法进行符号化时抛出 SympifyError 异常
    if strict:
        raise SympifyError(a)
    # 如果输入的对象 a 是一个可迭代对象
    if iterable(a):
        try:
            # 尝试使用 sympify 函数逐个处理可迭代对象 a 的元素，并构建一个相同类型的对象返回
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor,
                rational=rational, evaluate=evaluate) for x in a])
        except TypeError:
            # 某些可迭代对象无法通过其类型完全重建
            # 如果出现类型错误，继续执行后续的判断
            pass

    # 如果 a 不是字符串类型，则抛出 SympifyError 异常
    if not isinstance(a, str):
        raise SympifyError('cannot sympify object of type %r' % type(a))

    # 导入需要的模块和函数
    from sympy.parsing.sympy_parser import (parse_expr, TokenError,
                                            standard_transformations)
    from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
    from sympy.parsing.sympy_parser import rationalize as t_rationalize

    # 初始化转换列表
    transformations = standard_transformations

    # 根据参数设置是否进行有理数化的转换
    if rational:
        transformations += (t_rationalize,)
    # 根据参数设置是否进行 XOR 转换的转换
    if convert_xor:
        transformations += (t_convert_xor,)

    try:
        # 移除字符串 a 中的换行符，然后使用 parse_expr 函数解析表达式
        expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
    except (TokenError, SyntaxError) as exc:
        # 如果解析过程中出现 TokenError 或 SyntaxError 异常，抛出 SympifyError 异常
        raise SympifyError('could not parse %r' % a, exc)

    # 返回解析得到的表达式对象
    return expr
# 定义一个内部使用的简化符号处理函数，用于 __add__ 和 __eq__ 方法中，允许一些内容（如 Python 整数和浮点数），但不包括不适合的内容（如字符串）。
def _sympify(a):
    """
    Short version of :func:`~.sympify` for internal usage for ``__add__`` and
    ``__eq__`` methods where it is ok to allow some things (like Python
    integers and floats) in the expression. This excludes things (like strings)
    that are unwise to allow into such an expression.

    >>> from sympy import Integer
    >>> Integer(1) == 1
    True

    >>> Integer(1) == '1'
    False

    >>> from sympy.abc import x
    >>> x + 1
    x + 1

    >>> x + '1'
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for +: 'Symbol' and 'str'

    see: sympify
    """
    # 调用 sympify 函数，严格模式将参数转换为 SymPy 对象，并返回结果
    return sympify(a, strict=True)


def kernS(s):
    """
    Use a hack to try keep autosimplification from distributing a
    a number into an Add; this modification does not
    prevent the 2-arg Mul from becoming an Add, however.

    Examples
    ========

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x, y

    The 2-arg Mul distributes a number (or minus sign) across the terms
    of an expression, but kernS will prevent that:

    >>> 2*(x + y), -(x + 1)
    (2*x + 2*y, -x - 1)
    >>> kernS('2*(x + y)')
    2*(x + y)
    >>> kernS('-(x + 1)')
    -(x + 1)

    If use of the hack fails, the un-hacked string will be passed to sympify...
    and you get what you get.

    XXX This hack should not be necessary once issue 4596 has been resolved.
    """
    # 初始化变量 hit 为 False
    hit = False
    # 检查参数 s 是否包含单引号或双引号，并将结果赋给 quoted
    quoted = '"' in s or "'" in s
    # 如果字符串 s 中包含 '(' 且没有引号括起来
    if '(' in s and not quoted:
        # 如果左括号 '(' 的数量不等于右括号 ')' 的数量，抛出异常
        if s.count('(') != s.count(")"):
            raise SympifyError('unmatched left parenthesis')

        # 去除字符串 s 中所有的空格
        s = ''.join(s.split())
        # 保存原始字符串 s 的备份
        olds = s

        # 步骤 1：将可能的二元乘法 Muls 转换为三元版本
        # 1a. *( -> * *(
        s = s.replace('*(', '* *(')
        # 1b. 关闭指数运算符 '** *'
        s = s.replace('** *', '**')

        # 步骤 2：处理否定的括号表达式的隐式乘法
        # 2a: -(...)  -->  -( *(...)
        target = '-( *('
        s = s.replace('-(', target)
        
        # 2b: 加倍匹配的右括号
        # -( *(...)  -->  -( *(...))
        i = nest = 0
        assert target.endswith('(')  # 假设以下操作
        while True:
            j = s.find(target, i)
            if j == -1:
                break
            j += len(target) - 1
            for j in range(j, len(s)):
                if s[j] == "(":
                    nest += 1
                elif s[j] == ")":
                    nest -= 1
                if nest == 0:
                    break
            # 在匹配位置 j 处插入一个右括号 ")"
            s = s[:j] + ")" + s[j:]
            # 更新下一次搜索的起始位置
            i = j + 2  # 第二个 ")" 后的第一个字符位置

        # 如果字符串 s 中包含空格
        if ' ' in s:
            # 获取一个唯一的 kern
            kern = '_'
            while kern in s:
                kern += choice(string.ascii_letters + string.digits)
            # 将空格替换为 kern
            s = s.replace(' ', kern)
            # 检查 kern 是否在 s 中
            hit = kern in s
        else:
            hit = False

    # 尝试用 sympify 函数解析字符串 s，最多尝试两次
    for i in range(2):
        try:
            expr = sympify(s)
            break
        except TypeError:  # 可能是 kern 导致未知错误
            if hit:
                # 如果出现错误且 hit 为真，则使用未经 kern 处理的 olds
                s = olds
                hit = False
                continue
            # 否则，让原始的错误再次抛出
            expr = sympify(s)

    # 如果没有 hit，则返回表达式 expr
    if not hit:
        return expr

    # 导入符号类 Symbol
    from .symbol import Symbol
    # 创建一个替换字典 rep，将 kern 替换为 1
    rep = {Symbol(kern): 1}
    
    # 定义一个内部函数 _clear，用于递归清理表达式中的 kern
    def _clear(expr):
        if isinstance(expr, (list, tuple, set)):
            return type(expr)([_clear(e) for e in expr])
        if hasattr(expr, 'subs'):
            return expr.subs(rep, hack2=True)
        return expr

    # 对表达式 expr 应用 _clear 函数，以清理 kern
    expr = _clear(expr)
    # 希望此时 kern 不再存在于表达式中
    return expr
# 避免循环导入
from .basic import Basic
```