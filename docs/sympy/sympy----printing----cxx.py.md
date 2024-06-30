# `D:\src\scipysrc\sympy\sympy\printing\cxx.py`

```
"""
C++ code printer
"""

# 从 itertools 模块导入 chain 函数
from itertools import chain
# 从 sympy.codegen.ast 模块导入 Type 和 none
from sympy.codegen.ast import Type, none
# 导入 requires 函数和 C89CodePrinter、C99CodePrinter 类
from .codeprinter import requires
from .c import C89CodePrinter, C99CodePrinter

# 这些在另一个文件中定义，以避免从顶层 'import sympy' 导入，将其在此导出
from sympy.printing.codeprinter import cxxcode  # noqa:F401

# 定义保留关键字字典，来源于 https://en.cppreference.com/w/cpp/keyword
reserved = {
    'C++98': [
        'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break',
        'case', 'catch,', 'char', 'class', 'compl', 'const', 'const_cast',
        'continue', 'default', 'delete', 'do', 'double', 'dynamic_cast',
        'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float',
        'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable',
        'namespace', 'new', 'not', 'not_eq', 'operator', 'or', 'or_eq',
        'private', 'protected', 'public', 'register', 'reinterpret_cast',
        'return', 'short', 'signed', 'sizeof', 'static', 'static_cast',
        'struct', 'switch', 'template', 'this', 'throw', 'true', 'try',
        'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using',
        'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'
    ]
}

# 将 C++11 版本的保留关键字添加到 reserved 字典中
reserved['C++11'] = reserved['C++98'][:] + [
    'alignas', 'alignof', 'char16_t', 'char32_t', 'constexpr', 'decltype',
    'noexcept', 'nullptr', 'static_assert', 'thread_local'
]
# 将 C++17 版本的保留关键字添加到 reserved 字典中，并移除 'register'
reserved['C++17'] = reserved['C++11'][:]
reserved['C++17'].remove('register')

# 定义数学函数的映射字典
_math_functions = {
    'C++98': {
        'Mod': 'fmod',
        'ceiling': 'ceil',
    },
    'C++11': {
        'gamma': 'tgamma',
    },
    'C++17': {
        'beta': 'beta',
        'Ei': 'expint',
        'zeta': 'riemann_zeta',
    }
}

# 将 C++98 版本的数学函数添加到 _math_functions 字典中
# 来自 https://en.cppreference.com/w/cpp/header/cmath
for k in ('Abs', 'exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 'tan',
          'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'floor'):
    _math_functions['C++98'][k] = k.lower()

# 将 C++11 版本的数学函数添加到 _math_functions 字典中
for k in ('asinh', 'acosh', 'atanh', 'erf', 'erfc'):
    _math_functions['C++11'][k] = k.lower()


def _attach_print_method(cls, sympy_name, func_name):
    # 定义打印方法名称
    meth_name = '_print_%s' % sympy_name
    # 如果 cls 已经有了相同名称的方法，则引发 ValueError
    if hasattr(cls, meth_name):
        raise ValueError("Edit method (or subclass) instead of overwriting.")
    # 定义新的打印方法
    def _print_method(self, expr):
        return '{}{}({})'.format(self._ns, func_name, ', '.join(map(self._print, expr.args)))
    # 设置打印方法的文档字符串
    _print_method.__doc__ = "Prints code for %s" % k
    # 将打印方法设置为 cls 的属性
    setattr(cls, meth_name, _print_method)


def _attach_print_methods(cls, cont):
    # 遍历 cont 中 cls.standard 对应的项，并添加打印方法
    for sympy_name, cxx_name in cont[cls.standard].items():
        _attach_print_method(cls, sympy_name, cxx_name)


class _CXXCodePrinterBase:
    # 设置打印方法名
    printmethod = "_cxxcode"
    # 设置语言为 'C++'
    language = 'C++'
    # 设置命名空间为 'std::'
    _ns = 'std::'  # namespace
    # 初始化函数，接受一个可选的设置字典作为参数，并调用父类的初始化方法
    def __init__(self, settings=None):
        super().__init__(settings or {})

    # 带有装饰器的方法，要求设置 headers={'algorithm'}，用于打印表达式中的最大值
    @requires(headers={'algorithm'})
    def _print_Max(self, expr):
        # 导入 SymPy 中的最大值函数 Max
        from sympy.functions.elementary.miscellaneous import Max
        # 如果表达式参数中只有一个元素，则直接打印该元素
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        # 否则，返回格式化后的字符串，包含最大值函数的调用
        return "%smax(%s, %s)" % (self._ns, self._print(expr.args[0]),
                                  self._print(Max(*expr.args[1:])))

    # 带有装饰器的方法，要求设置 headers={'algorithm'}，用于打印表达式中的最小值
    @requires(headers={'algorithm'})
    def _print_Min(self, expr):
        # 导入 SymPy 中的最小值函数 Min
        from sympy.functions.elementary.miscellaneous import Min
        # 如果表达式参数中只有一个元素，则直接打印该元素
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        # 否则，返回格式化后的字符串，包含最小值函数的调用
        return "%smin(%s, %s)" % (self._ns, self._print(expr.args[0]),
                                  self._print(Min(*expr.args[1:])))

    # 用于打印 using 表达式，根据表达式中的 alias 属性返回相应字符串
    def _print_using(self, expr):
        # 如果 alias 属性为 none，则返回使用该表达式的类型信息
        if expr.alias == none:
            return 'using %s' % expr.type
        # 否则，抛出异常，因为 C++98 不支持类型别名
        else:
            raise ValueError("C++98 does not support type aliases")

    # 用于打印 raise 表达式，返回格式化的 throw 语句，抛出表达式中的异常
    def _print_Raise(self, rs):
        # 解包 rs.args 中的唯一参数
        arg, = rs.args
        return 'throw %s' % self._print(arg)

    # 带有装饰器的方法，要求设置 headers={'stdexcept'}，用于打印 runtime_error 异常
    @requires(headers={'stdexcept'})
    def _print_RuntimeError_(self, re):
        # 解包 re.args 中的唯一参数，即异常消息
        message, = re.args
        # 返回格式化后的字符串，创建一个 runtime_error 异常对象
        return "%sruntime_error(%s)" % (self._ns, self._print(message))
# 定义一个代码打印器类，用于打印符合 C++98 标准的代码，继承自 _CXXCodePrinterBase 和 C89CodePrinter
class CXX98CodePrinter(_CXXCodePrinterBase, C89CodePrinter):
    # 设定标准为 'C++98'
    standard = 'C++98'
    # 设置保留字为 'C++98' 标准中的保留字集合
    reserved_words = set(reserved['C++98'])


# _attach_print_methods(CXX98CodePrinter, _math_functions)


# 定义一个代码打印器类，用于打印符合 C++11 标准的代码，继承自 _CXXCodePrinterBase 和 C99CodePrinter
class CXX11CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    # 设定标准为 'C++11'
    standard = 'C++11'
    # 设置保留字为 'C++11' 标准中的保留字集合
    reserved_words = set(reserved['C++11'])
    # 定义类型映射，扩展自 CXX98CodePrinter 的类型映射，并添加了新类型映射
    type_mappings = dict(chain(
        CXX98CodePrinter.type_mappings.items(),
        {
            Type('int8'): ('int8_t', {'cstdint'}),
            Type('int16'): ('int16_t', {'cstdint'}),
            Type('int32'): ('int32_t', {'cstdint'}),
            Type('int64'): ('int64_t', {'cstdint'}),
            Type('uint8'): ('uint8_t', {'cstdint'}),
            Type('uint16'): ('uint16_t', {'cstdint'}),
            Type('uint32'): ('uint32_t', {'cstdint'}),
            Type('uint64'): ('uint64_t', {'cstdint'}),
            Type('complex64'): ('std::complex<float>', {'complex'}),
            Type('complex128'): ('std::complex<double>', {'complex'}),
            Type('bool'): ('bool', None),
        }.items()
    ))

    # 定义 _print_using 方法，根据表达式是否有别名决定打印 'using' 语句
    def _print_using(self, expr):
        if expr.alias == none:
            return super()._print_using(expr)
        else:
            return 'using %(alias)s = %(type)s' % expr.kwargs(apply=self._print)

# _attach_print_methods(CXX11CodePrinter, _math_functions)


# 定义一个代码打印器类，用于打印符合 C++17 标准的代码，继承自 _CXXCodePrinterBase 和 C99CodePrinter
class CXX17CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    # 设定标准为 'C++17'
    standard = 'C++17'
    # 设置保留字为 'C++17' 标准中的保留字集合
    reserved_words = set(reserved['C++17'])
    
    # 合并 C99CodePrinter 类的关键字字典和 _math_functions['C++17'] 字典到 _kf 中
    _kf = dict(C99CodePrinter._kf, **_math_functions['C++17'])

    # 定义 _print_beta 方法，用于打印数学函数表达式 'beta'
    def _print_beta(self, expr):
        return self._print_math_func(expr)

    # 定义 _print_Ei 方法，用于打印数学函数表达式 'Ei'
    def _print_Ei(self, expr):
        return self._print_math_func(expr)

    # 定义 _print_zeta 方法，用于打印数学函数表达式 'zeta'
    def _print_zeta(self, expr):
        return self._print_math_func(expr)

# _attach_print_methods(CXX17CodePrinter, _math_functions)


# 定义一个字典，将不同版本的 C++ 代码打印器类映射到相应的字符串键上
cxx_code_printers = {
    'c++98': CXX98CodePrinter,
    'c++11': CXX11CodePrinter,
    'c++17': CXX17CodePrinter
}
```