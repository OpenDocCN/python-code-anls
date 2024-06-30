# `D:\src\scipysrc\sympy\sympy\printing\python.py`

```
import keyword as kw
import sympy
from .repr import ReprPrinter
from .str import StrPrinter

# 定义需要使用 StrPrinter 而不是 ReprPrinter 的类列表
STRPRINT = ("Add", "Infinity", "Integer", "Mul", "NegativeInfinity", "Pow")

# PythonPrinter 类继承自 ReprPrinter 和 StrPrinter，用于将表达式转换为其 Python 表示形式
class PythonPrinter(ReprPrinter, StrPrinter):
    """A printer which converts an expression into its Python interpretation."""

    def __init__(self, settings=None):
        super().__init__(settings)
        self.symbols = []  # 用于存储发现的符号变量
        self.functions = []  # 用于存储发现的函数

        # 为需要使用 StrPrinter 的类创建打印方法
        for name in STRPRINT:
            f_name = "_print_%s" % name
            f = getattr(StrPrinter, f_name)
            setattr(PythonPrinter, f_name, f)

    # 处理 Function 类的打印方法，记录未知函数名称到 functions 列表
    def _print_Function(self, expr):
        func = expr.func.__name__
        if not hasattr(sympy, func) and func not in self.functions:
            self.functions.append(func)
        return StrPrinter._print_Function(self, expr)

    # 处理 Symbol 类的打印方法，记录符号名称到 symbols 列表
    def _print_Symbol(self, expr):
        symbol = self._str(expr)
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        return StrPrinter._print_Symbol(self, expr)

    # 当表达式中包含 module 类时引发错误，不允许处理模块
    def _print_module(self, expr):
        raise ValueError('Modules in the expression are unacceptable')


# python 函数将表达式转换为其 Python 解释形式，可直接传递给 exec() 函数
def python(expr, **settings):
    """Return Python interpretation of passed expression
    (can be passed to the exec() function without any modifications)"""

    printer = PythonPrinter(settings)
    exprp = printer.doprint(expr)  # 执行打印转换操作

    result = ''
    # 返回找到的符号和函数
    renamings = {}
    for symbolname in printer.symbols:
        # 如果符号名称中包含大括号，则去除大括号
        if '{' in symbolname:
            newsymbolname = symbolname.replace('{', '').replace('}', '')
            renamings[sympy.Symbol(symbolname)] = newsymbolname
        else:
            newsymbolname = symbolname

        # 如果符号名称是 Python 保留关键字，则进行转义处理
        if kw.iskeyword(newsymbolname):
            while True:
                newsymbolname += "_"
                if (newsymbolname not in printer.symbols and
                        newsymbolname not in printer.functions):
                    renamings[sympy.Symbol(
                        symbolname)] = sympy.Symbol(newsymbolname)
                    break
        result += newsymbolname + ' = Symbol(\'' + symbolname + '\')\n'
    # 遍历打印机对象中的函数名列表
    for functionname in printer.functions:
        # 复制函数名到新变量
        newfunctionname = functionname
        # 如果函数名是Python保留关键字，则进行转义处理
        if kw.iskeyword(newfunctionname):
            # 循环直到找到一个不与打印机符号或函数名重复的新函数名
            while True:
                newfunctionname += "_"
                # 如果新函数名既不在打印机的符号列表中，也不在函数名列表中，即可使用
                if (newfunctionname not in printer.symbols and
                        newfunctionname not in printer.functions):
                    # 将原始函数名映射到新函数名，并添加到重命名字典中
                    renamings[sympy.Function(
                        functionname)] = sympy.Function(newfunctionname)
                    break
        # 将新函数名和赋值语句添加到结果字符串中
        result += newfunctionname + ' = Function(\'' + functionname + '\')\n'

    # 如果存在需要重命名的函数映射，则用重命名后的表达式进行替换
    if renamings:
        exprp = expr.subs(renamings)
    # 将表达式字符串化并添加到结果字符串末尾
    result += 'e = ' + printer._str(exprp)
    # 返回最终的结果字符串
    return result
# 定义一个函数 print_python，用于打印 python() 函数的输出结果
def print_python(expr, **settings):
    """Print output of python() function"""
    # 调用 python() 函数，并将结果打印出来
    print(python(expr, **settings))
```