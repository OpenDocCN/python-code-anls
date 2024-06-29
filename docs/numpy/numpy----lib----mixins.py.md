# `.\numpy\numpy\lib\mixins.py`

```
"""
Mixin classes for custom array types that don't inherit from ndarray.
"""
# 导入 umath 模块作为 um 的别名，用于执行数学运算
from numpy._core import umath as um

# 定义模块公开的类名列表
__all__ = ['NDArrayOperatorsMixin']


def _disables_array_ufunc(obj):
    """True when __array_ufunc__ is set to None."""
    # 检查对象是否有 __array_ufunc__ 属性，并返回是否为 None
    try:
        return obj.__array_ufunc__ is None
    except AttributeError:
        return False


def _binary_method(ufunc, name):
    """Implement a forward binary method with a ufunc, e.g., __add__."""
    # 定义一个前向二元方法的函数，使用给定的 ufunc 函数进行操作
    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(self, other)
    func.__name__ = '__{}__'.format(name)
    return func


def _reflected_binary_method(ufunc, name):
    """Implement a reflected binary method with a ufunc, e.g., __radd__."""
    # 定义一个反向二元方法的函数，使用给定的 ufunc 函数进行反向操作
    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(other, self)
    func.__name__ = '__r{}__'.format(name)
    return func


def _inplace_binary_method(ufunc, name):
    """Implement an in-place binary method with a ufunc, e.g., __iadd__."""
    # 定义一个就地二元方法的函数，使用给定的 ufunc 函数进行操作并更新自身
    def func(self, other):
        return ufunc(self, other, out=(self,))
    func.__name__ = '__i{}__'.format(name)
    return func


def _numeric_methods(ufunc, name):
    """Implement forward, reflected and inplace binary methods with a ufunc."""
    # 返回包含前向、反向和就地二元方法函数的元组，使用给定的 ufunc 函数进行操作
    return (_binary_method(ufunc, name),
            _reflected_binary_method(ufunc, name),
            _inplace_binary_method(ufunc, name))


def _unary_method(ufunc, name):
    """Implement a unary special method with a ufunc."""
    # 定义一个一元特殊方法的函数，使用给定的 ufunc 函数进行操作
    def func(self):
        return ufunc(self)
    func.__name__ = '__{}__'.format(name)
    return func


class NDArrayOperatorsMixin:
    """Mixin defining all operator special methods using __array_ufunc__.

    This class implements the special methods for almost all of Python's
    builtin operators defined in the `operator` module, including comparisons
    (``==``, ``>``, etc.) and arithmetic (``+``, ``*``, ``-``, etc.), by
    deferring to the ``__array_ufunc__`` method, which subclasses must
    implement.

    It is useful for writing classes that do not inherit from `numpy.ndarray`,
    but that should support arithmetic and numpy universal functions like
    arrays as described in `A Mechanism for Overriding Ufuncs
    <https://numpy.org/neps/nep-0013-ufunc-overrides.html>`_.

    As an trivial example, consider this implementation of an ``ArrayLike``
    class that simply wraps a NumPy array and ensures that the result of any
    """
    # 此处省略部分注释，因为不在示例范围内，按要求只注释示例中的一部分代码块
    # 此处定义了一个类 ArrayLike，它继承自 np.lib.mixins.NDArrayOperatorsMixin 类。
    # ArrayLike 类模拟了 numpy 数组的行为，支持与数字和 numpy 数组之间的操作。
    class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
        # 初始化方法，接受一个 value 参数，并将其转换为 numpy 数组存储在 self.value 中。
        def __init__(self, value):
            self.value = np.asarray(value)
        
        # 定义了 __array_ufunc__ 方法，用于处理 numpy 的通用函数操作。
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            # 获取 'out' 参数，如果没有则为空元组。
            out = kwargs.get('out', ())
            # 遍历输入和输出，检查是否都是 _HANDLED_TYPES 中定义的类型或者 ArrayLike 类型。
            for x in inputs + out:
                if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):
                    # 如果不是支持的类型，则返回 NotImplemented。
                    return NotImplemented
            
            # 将输入中的 ArrayLike 对象转换为其内部存储的值（self.value）。
            inputs = tuple(x.value if isinstance(x, ArrayLike) else x for x in inputs)
            # 如果存在输出，同样将输出中的 ArrayLike 对象转换为其内部存储的值。
            if out:
                kwargs['out'] = tuple(x.value if isinstance(x, ArrayLike) else x for x in out)
            
            # 调用 ufunc 对象的指定方法（如 'add'、'subtract' 等），处理输入和参数，返回结果。
            result = getattr(ufunc, method)(*inputs, **kwargs)
            
            # 如果结果是一个元组，说明有多个返回值，则将每个返回值转换为 ArrayLike 对象并返回。
            if type(result) is tuple:
                return tuple(type(self)(x) for x in result)
            # 如果方法是 'at'，表示没有返回值，则返回 None。
            elif method == 'at':
                return None
            # 否则，将单个返回值转换为 ArrayLike 对象并返回。
            else:
                return type(self)(result)
        
        # 返回对象的字符串表示形式，格式为类名和 self.value 的表示。
        def __repr__(self):
            return '%s(%r)' % (type(self).__name__, self.value)
    
    # 设置 __slots__ 属性为空元组，这意味着此类不允许动态添加新的实例属性。
    __slots__ = ()
    # 类似于 np.ndarray，此混合类实现了 ufunc 覆盖 NEP 的 "Option 1"。
    # comparisons don't have reflected and in-place versions
    __lt__ = _binary_method(um.less, 'lt')
    __le__ = _binary_method(um.less_equal, 'le')
    __eq__ = _binary_method(um.equal, 'eq')
    __ne__ = _binary_method(um.not_equal, 'ne')
    __gt__ = _binary_method(um.greater, 'gt')
    __ge__ = _binary_method(um.greater_equal, 'ge')
    
    # numeric methods
    # 设置数值方法 __add__, __radd__, __iadd__，使用 um.add 方法进行操作
    __add__, __radd__, __iadd__ = _numeric_methods(um.add, 'add')
    # 设置数值方法 __sub__, __rsub__, __isub__，使用 um.subtract 方法进行操作
    __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, 'sub')
    # 设置数值方法 __mul__, __rmul__, __imul__，使用 um.multiply 方法进行操作
    __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, 'mul')
    # 设置数值方法 __matmul__, __rmatmul__, __imatmul__，使用 um.matmul 方法进行操作
    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(um.matmul, 'matmul')
    # Python 3 不使用 __div__, __rdiv__, 或 __idiv__
    # 设置数值方法 __truediv__, __rtruediv__, __itruediv__，使用 um.true_divide 方法进行操作
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(um.true_divide, 'truediv')
    # 设置数值方法 __floordiv__, __rfloordiv__, __ifloordiv__，使用 um.floor_divide 方法进行操作
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(um.floor_divide, 'floordiv')
    # 设置数值方法 __mod__, __rmod__, __imod__，使用 um.remainder 方法进行操作
    __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, 'mod')
    # 设置二元方法 __divmod__，使用 um.divmod 方法进行操作
    __divmod__ = _binary_method(um.divmod, 'divmod')
    # 设置反射二元方法 __rdivmod__，使用 um.divmod 方法进行操作
    __rdivmod__ = _reflected_binary_method(um.divmod, 'divmod')
    # __idivmod__ 不存在
    # TODO: 处理 __pow__ 的可选第三个参数？
    # 设置数值方法 __pow__, __rpow__, __ipow__，使用 um.power 方法进行操作
    __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, 'pow')
    # 设置数值方法 __lshift__, __rlshift__, __ilshift__，使用 um.left_shift 方法进行操作
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods(um.left_shift, 'lshift')
    # 设置数值方法 __rshift__, __rrshift__, __irshift__，使用 um.right_shift 方法进行操作
    __rshift__, __rrshift__, __irshift__ = _numeric_methods(um.right_shift, 'rshift')
    # 设置数值方法 __and__, __rand__, __iand__，使用 um.bitwise_and 方法进行操作
    __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, 'and')
    # 设置数值方法 __xor__, __rxor__, __ixor__，使用 um.bitwise_xor 方法进行操作
    __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, 'xor')
    # 设置数值方法 __or__, __ror__, __ior__，使用 um.bitwise_or 方法进行操作
    __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, 'or')
    
    # unary methods
    # 设置一元方法 __neg__，使用 um.negative 方法进行操作
    __neg__ = _unary_method(um.negative, 'neg')
    # 设置一元方法 __pos__，使用 um.positive 方法进行操作
    __pos__ = _unary_method(um.positive, 'pos')
    # 设置一元方法 __abs__，使用 um.absolute 方法进行操作
    __abs__ = _unary_method(um.absolute, 'abs')
    # 设置一元方法 __invert__，使用 um.invert 方法进行操作
    __invert__ = _unary_method(um.invert, 'invert')
```