# `D:\src\scipysrc\sympy\sympy\core\_print_helpers.py`

```
"""
Base class to provide str and repr hooks that `init_printing` can overwrite.

This is exposed publicly in the `printing.defaults` module,
but cannot be defined there without causing circular imports.
"""

# 定义一个可打印类，提供`str`和`repr`钩子，以便`init_printing`可以覆盖
class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way. Natively Python uses ``repr()``
    even if ``str()`` was explicitly requested. Mix in this trait into
    a class to get proper default printing.

    This also adds support for LaTeX printing in jupyter notebooks.
    """

    # 由于这个类作为一个mixin使用，我们设置了空的slots。这意味着任何使用slots的子类的实例都不需要有__dict__。
    __slots__ = ()

    # 注意，我们总是在__str__和__repr__中使用默认的排序方式（lex），而不管全局设置如何。参见issue 5487。
    def __str__(self):
        # 导入`sstr`函数并使用它来返回对象的字符串表示
        from sympy.printing.str import sstr
        return sstr(self, order=None)

    __repr__ = __str__  # `__repr__`方法与`__str__`相同

    def _repr_disabled(self):
        """
        No-op repr function used to disable jupyter display hooks.

        When :func:`sympy.init_printing` is used to disable certain display
        formats, this function is copied into the appropriate ``_repr_*_``
        attributes.

        While we could just set the attributes to `None``, doing it this way
        allows derived classes to call `super()`.
        """
        # 用于禁用Jupyter显示钩子的空操作repr函数
        return None

    # 在这里不实现`_repr_png_`，因为它会向包含SymPy表达式的任何笔记本添加大量数据，而不增加任何有用信息。
    _repr_png_ = _repr_disabled

    _repr_svg_ = _repr_disabled  # 同上，禁用SVG显示钩子

    def _repr_latex_(self):
        """
        IPython/Jupyter LaTeX printing

        To change the behavior of this (e.g., pass in some settings to LaTeX),
        use init_printing(). init_printing() will also enable LaTeX printing
        for built in numeric types like ints and container types that contain
        SymPy objects, like lists and dictionaries of expressions.
        """
        # 导入`latex`函数并使用它来返回对象的LaTeX表示
        from sympy.printing.latex import latex
        s = latex(self, mode='plain')
        return "$\\displaystyle %s$" % s
```