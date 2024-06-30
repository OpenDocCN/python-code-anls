# `D:\src\scipysrc\sympy\sympy\printing\printer.py`

```
"""Printing subsystem driver

SymPy's printing system works the following way: Any expression can be
passed to a designated Printer who then is responsible to return an
adequate representation of that expression.

**The basic concept is the following:**

1.  Let the object print itself if it knows how.
2.  Take the best fitting method defined in the printer.
3.  As fall-back use the emptyPrinter method for the printer.

Which Method is Responsible for Printing?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The whole printing process is started by calling ``.doprint(expr)`` on the printer
which you want to use. This method looks for an appropriate method which can
print the given expression in the given style that the printer defines.
While looking for the method, it follows these steps:

1.  **Let the object print itself if it knows how.**

    The printer looks for a specific method in every object. The name of that method
    depends on the specific printer and is defined under ``Printer.printmethod``.
    For example, StrPrinter calls ``_sympystr`` and LatexPrinter calls ``_latex``.
    Look at the documentation of the printer that you want to use.
    The name of the method is specified there.

    This was the original way of doing printing in sympy. Every class had
    its own latex, mathml, str and repr methods, but it turned out that it
    is hard to produce a high quality printer, if all the methods are spread
    out that far. Therefore all printing code was combined into the different
    printers, which works great for built-in SymPy objects, but not that
    good for user defined classes where it is inconvenient to patch the
    printers.

2.  **Take the best fitting method defined in the printer.**

    The printer loops through expr classes (class + its bases), and tries
    to dispatch the work to ``_print_<EXPR_CLASS>``

    e.g., suppose we have the following class hierarchy::

            Basic
            |
            Atom
            |
            Number
            |
        Rational

    then, for ``expr=Rational(...)``, the Printer will try
    to call printer methods in the order as shown in the figure below::

        p._print(expr)
        |
        |-- p._print_Rational(expr)
        |
        |-- p._print_Number(expr)
        |
        |-- p._print_Atom(expr)
        |
        `-- p._print_Basic(expr)

    if ``._print_Rational`` method exists in the printer, then it is called,
    and the result is returned back. Otherwise, the printer tries to call
    ``._print_Number`` and so on.

3.  **As a fall-back use the emptyPrinter method for the printer.**

    As fall-back ``self.emptyPrinter`` will be called with the expression. If
    not defined in the Printer subclass this will be the same as ``str(expr)``.

.. _printer_example:

Example of Custom Printer
^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, we have a printer which prints the derivative of a function
in a shorter form.

.. code-block:: python
"""
    # 导入符号操作相关的类 Symbol
    from sympy.core.symbol import Symbol
    # 导入 LaTeX 打印相关的类 LatexPrinter 和打印函数 print_latex
    from sympy.printing.latex import LatexPrinter, print_latex
    # 导入未定义函数和函数类 UndefinedFunction, Function
    from sympy.core.function import UndefinedFunction, Function


    # 定义一个自定义的 LaTeX 打印器 MyLatexPrinter，继承自 LatexPrinter 类
    class MyLatexPrinter(LatexPrinter):
        """Print derivative of a function of symbols in a shorter form.
        """
        # 重写 _print_Derivative 方法，用于打印符号导数的特定格式
        def _print_Derivative(self, expr):
            # 获取函数及其参数
            function, *vars = expr.args
            # 检查函数是否为未定义函数且所有参数均为符号
            if not isinstance(type(function), UndefinedFunction) or \\
               not all(isinstance(i, Symbol) for i in vars):
                # 如果不符合特定格式，调用父类的打印方法
                return super()._print_Derivative(expr)

            # 如果希望打印器在嵌套表达式中正确工作，应使用 self._print() 而不是 str() 或 latex()
            # 参见下面在自定义打印方法部分中的嵌套模数的示例
            return "{}_{{{}}}".format(
                self._print(Symbol(function.func.__name__)),
                            ''.join(self._print(i) for i in vars))


    # 定义打印自定义 LaTeX 表达式的函数 print_my_latex
    def print_my_latex(expr):
        """Most of the printers define their own wrappers for print().
        These wrappers usually take printer settings. Our printer does not have
        any settings.
        """
        # 创建 MyLatexPrinter 实例并打印表达式
        print(MyLatexPrinter().doprint(expr))


    # 创建符号 y 和 x
    y = Symbol("y")
    x = Symbol("x")
    # 创建函数符号 f
    f = Function("f")
    # 创建表达式 f(x, y) 的 x, y 偏导数
    expr = f(x, y).diff(x, y)

    # 使用普通的 LaTeX 打印器和我们的自定义打印器打印表达式
    print_latex(expr)
    print_my_latex(expr)
# 导入需要的符号、模块和函数
>>> from sympy import Symbol, Mod, Integer, print_latex

# 自定义 Mod 操作符的子类 ModOp
>>> class ModOp(Mod):
...     # 重载 _latex 方法，用于自定义 LaTeX 输出格式
...     def _latex(self, printer):
...         # 使用 printer._print 方法来打印 ModOp 对象的参数
...         a, b = [printer._print(i) for i in self.args]
...         # 返回自定义的 LaTeX 格式化字符串
...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

# 比较自定义操作符输出和内置操作符输出的差异
>>> x = Symbol('x')
>>> m = Symbol('m')
>>> print_latex(Mod(x, m))
x \\bmod m
>>> print_latex(ModOp(x, m))
\\operatorname{Mod}{\\left(x, m\\right)}
    # 计算 1.0 x 对 m 取模
    \operatorname{Mod}{\left(1.0 x, m\right)}
"""

from __future__ import annotations  # 导入用于支持注释中的类型标注
import sys  # 导入系统相关的模块
from typing import Any, Type  # 导入用于类型标注的模块
import inspect  # 导入用于检查对象的模块
from contextlib import contextmanager  # 导入用于上下文管理的模块
from functools import cmp_to_key, update_wrapper  # 导入用于函数包装和比较器转换的模块

from sympy.core.add import Add  # 导入 Sympy 中的加法类
from sympy.core.basic import Basic  # 导入 Sympy 中的基本类

from sympy.core.function import AppliedUndef, UndefinedFunction, Function  # 导入 Sympy 中的函数相关类


@contextmanager
def printer_context(printer, **kwargs):
    """定义一个上下文管理器，用于在打印过程中管理打印机的上下文设置。"""
    original = printer._context.copy()  # 备份当前打印机的上下文设置
    try:
        printer._context.update(kwargs)  # 更新打印机的上下文设置
        yield  # 执行上下文管理器中的代码块
    finally:
        printer._context = original  # 恢复原始的打印机上下文设置


class Printer:
    """通用打印机类

    该类提供了实现新打印机的基础设施。

    如果要定义自定义的打印机或自定义类的自定义打印方法，
    可参考上面的示例 printer_example_ 。
    """

    _global_settings: dict[str, Any] = {}  # 全局设置，用于存储全局打印设置

    _default_settings: dict[str, Any] = {}  # 默认设置，用于存储默认的打印设置

    printmethod = None  # type: str  # 打印方法，未指定具体类型的字符串

    @classmethod
    def _get_initial_settings(cls):
        """获取初始设置的类方法

        返回一个基于默认设置和全局设置的初始设置字典。
        """
        settings = cls._default_settings.copy()  # 拷贝默认设置
        for key, val in cls._global_settings.items():
            if key in cls._default_settings:
                settings[key] = val  # 更新全局设置中存在于默认设置的键的值
        return settings  # 返回初始设置字典

    def __init__(self, settings=None):
        """打印机类的初始化方法

        初始化打印机对象，设置默认的字符串转换函数和初始设置。
        """
        self._str = str  # 设置默认的字符串转换函数为内置的 str 函数

        self._settings = self._get_initial_settings()  # 获取初始设置
        self._context = {}  # mutable during printing  # 打印过程中可变的上下文设置

        if settings is not None:
            self._settings.update(settings)  # 更新打印机的设置

            if len(self._settings) > len(self._default_settings):
                for key in self._settings:
                    if key not in self._default_settings:
                        raise TypeError("Unknown setting '%s'." % key)

        # _print_level 是 self._print() 递归调用的次数。参见 StrPrinter._print_Float() 的使用示例
        self._print_level = 0  # 初始化打印级别为 0

    @classmethod
    def set_global_settings(cls, **settings):
        """设置全局打印设置的类方法

        允许设置系统范围的打印设置。
        """
        for key, val in settings.items():
            if val is not None:
                cls._global_settings[key] = val  # 更新全局设置

    @property
    def order(self):
        """属性方法 order

        获取打印机的设置中的 'order' 键的值。
        如果未定义 'order'，则抛出 AttributeError 异常。
        """
        if 'order' in self._settings:
            return self._settings['order']  # 返回设置中的 'order' 值
        else:
            raise AttributeError("No order defined.")  # 抛出未定义 'order' 的异常

    def doprint(self, expr):
        """打印方法 doprint

        返回表达式 expr 的打印表示（字符串形式）。
        """
        return self._str(self._print(expr))  # 返回打印结果的字符串形式
    def _print(self, expr, **kwargs) -> str:
        """Internal dispatcher

        Tries the following concepts to print an expression:
            1. Let the object print itself if it knows how.
            2. Take the best fitting method defined in the printer.
            3. As fall-back use the emptyPrinter method for the printer.
        """
        # 增加打印级别计数器，用于追踪嵌套调用
        self._print_level += 1
        try:
            # 如果打印机定义了打印方法的名称 (Printer.printmethod)，并且表达式对象知道如何打印自己，则使用该方法。
            if self.printmethod and hasattr(expr, self.printmethod):
                # 检查表达式对象是否不是类型对象并且不是 Basic 的子类，然后调用相应的打印方法。
                if not (isinstance(expr, type) and issubclass(expr, Basic)):
                    return getattr(expr, self.printmethod)(self, **kwargs)

            # 查看表达式 expr 的类及其超类是否有已定义的打印函数
            # 异常情况：忽略 Undefined 的子类，例如 Function('gamma') 不应分派到 _print_gamma
            classes = type(expr).__mro__
            if AppliedUndef in classes:
                classes = classes[classes.index(AppliedUndef):]
            if UndefinedFunction in classes:
                classes = classes[classes.index(UndefinedFunction):]
            # 另一个异常情况：如果有人定义了已知函数的子类，例如 gamma，并更改了名称，则忽略 _print_gamma
            if Function in classes:
                i = classes.index(Function)
                classes = tuple(c for c in classes[:i] if \
                    c.__name__ == classes[0].__name__ or \
                    c.__name__.endswith("Base")) + classes[i:]
            # 遍历类列表，尝试获取每个类对应的打印方法，并调用之
            for cls in classes:
                printmethodname = '_print_' + cls.__name__
                printmethod = getattr(self, printmethodname, None)
                if printmethod is not None:
                    return printmethod(expr, **kwargs)
            # 未知对象类型，使用空打印方法 emptyPrinter 作为后备
            return self.emptyPrinter(expr)
        finally:
            # 减少打印级别计数器，确保在函数返回前进行清理
            self._print_level -= 1

    def emptyPrinter(self, expr):
        # 默认的空打印方法，将表达式转换为字符串
        return str(expr)

    def _as_ordered_terms(self, expr, order=None):
        """A compatibility function for ordering terms in Add. """
        # 如果未指定排序顺序，则使用默认的 self.order
        order = order or self.order

        if order == 'old':
            # 按照旧的方式对 Add 类型的表达式进行排序，并返回排序后的结果
            return sorted(Add.make_args(expr), key=cmp_to_key(Basic._compare_pretty))
        elif order == 'none':
            # 返回未排序的表达式的参数列表
            return list(expr.args)
        else:
            # 使用 expr 的 as_ordered_terms 方法进行排序，并传递指定的 order 参数
            return expr.as_ordered_terms(order=order)
class _PrintFunction:
    """
    Function wrapper to replace ``**settings`` in the signature with printer defaults
    """

    def __init__(self, f, print_cls: Type[Printer]):
        # 获取函数 f 的参数列表
        params = list(inspect.signature(f).parameters.values())
        # 移除最后一个参数，应为 **settings
        assert params.pop(-1).kind == inspect.Parameter.VAR_KEYWORD
        self.__other_params = params

        self.__print_cls = print_cls
        # 更新包装器，将当前对象包装成函数 f
        update_wrapper(self, f)

    def __reduce__(self):
        # 由于此对象用作装饰器，替换原始函数
        # 默认的 pickle 会尝试对 self.__wrapped__ 进行 pickle，并失败，
        # 因为无法通过名称检索包装的函数。
        return self.__wrapped__.__qualname__

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        # 获取打印设置的初始值
        settings = self.__print_cls._get_initial_settings()
        return inspect.Signature(
            # 构建新的函数签名，添加默认参数为打印设置
            parameters=self.__other_params + [
                inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
                for k, v in settings.items()
            ],
            return_annotation=self.__wrapped__.__annotations__.get('return', inspect.Signature.empty)  # type:ignore
        )


def print_function(print_cls):
    """ A decorator to replace kwargs with the printer settings in __signature__ """
    def decorator(f):
        if sys.version_info < (3, 9):
            # 在旧版本 Python 中，必须创建一个子类，
            # 以便 `help` 在帮助文档中显示 docstring。
            # IPython 和 Sphinx 不需要这样做，只有原始的 Python 控制台需要。
            cls = type(f'{f.__qualname__}_PrintFunction', (_PrintFunction,), {"__doc__": f.__doc__})
        else:
            cls = _PrintFunction
        # 返回函数 f 的一个新实例，通过 cls 包装，使用 print_cls 参数
        return cls(f, print_cls)
    return decorator
```