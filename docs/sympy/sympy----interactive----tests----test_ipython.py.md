# `D:\src\scipysrc\sympy\sympy\interactive\tests\test_ipython.py`

```
"""Tests of tools for setting up interactive IPython sessions. """

# 从 sympy.interactive.session 模块导入所需的函数和类
from sympy.interactive.session import (
    init_ipython_session,
    enable_automatic_symbols,
    enable_automatic_int_sympification
)

# 从 sympy.core 模块导入 Symbol, Rational, Integer 类
from sympy.core import Symbol, Rational, Integer

# 从 sympy.external 模块导入 import_module 函数
from sympy.external import import_module

# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises

# 检查是否成功导入 IPython 模块，并根据导入结果设置 disabled 变量
ipython = import_module("IPython", min_module_version="0.11")
if not ipython:
    # 如果未成功导入 IPython，将 disabled 设为 True
    disabled = True

# WARNING: These tests will modify the existing IPython environment. IPython
# uses a single instance for its interpreter, so there is no way to isolate
# the test from another IPython session. It also means that if this test is
# run twice in the same Python session it will fail. This isn't usually a
# problem because the test suite is run in a subprocess by default, but if the
# tests are run with subprocess=False it can pollute the current IPython
# session. See the discussion in issue #15149.

# 定义测试函数 test_automatic_symbols，用于测试自动符号功能
def test_automatic_symbols():
    # NOTE: Because of the way the hook works, you have to use run_cell(code,
    # True).  This means that the code must have no Out, or it will be printed
    # during the tests.
    
    # 初始化 IPython 会话对象
    app = init_ipython_session()
    # 在 IPython 中运行命令，导入 sympy 库的所有内容
    app.run_cell("from sympy import *")

    # 启用自动符号功能
    enable_automatic_symbols(app)

    # 定义一个很长的符号名
    symbol = "verylongsymbolname"
    # 检查符号名在当前命名空间中不存在
    assert symbol not in app.user_ns
    # 在 IPython 中运行命令，尝试创建一个符号
    app.run_cell("a = %s" % symbol, True)
    # 再次检查符号名在当前命名空间中不存在
    assert symbol not in app.user_ns
    # 在 IPython 中运行命令，检查符号的类型
    app.run_cell("a = type(%s)" % symbol, True)
    # 检查运行后符号的类型是否为 Symbol
    assert app.user_ns['a'] == Symbol
    # 在 IPython 中运行命令，创建一个具体的符号对象
    app.run_cell("%s = Symbol('%s')" % (symbol, symbol), True)
    # 检查符号名是否出现在当前命名空间中
    assert symbol in app.user_ns

    # 检查内置名称是否被覆盖
    app.run_cell("a = all == __builtin__.all", True)
    # 检查内置名称 'all' 是否不在当前命名空间中
    assert "all" not in app.user_ns
    # 检查 app.user_ns 中 'a' 的值是否为 True
    assert app.user_ns['a'] is True

    # 检查 SymPy 的名称是否被覆盖
    app.run_cell("import sympy")
    app.run_cell("a = factorial == sympy.factorial", True)
    # 检查 app.user_ns 中 'a' 的值是否为 True
    assert app.user_ns['a'] is True

# 定义测试函数 test_int_to_Integer，用于测试整数转换为 Integer 对象的功能
def test_int_to_Integer():
    # XXX: Warning, don't test with == here.  0.5 == Rational(1, 2) is True!
    
    # 初始化 IPython 会话对象
    app = init_ipython_session()
    # 在 IPython 中运行命令，导入 Integer 类
    app.run_cell("from sympy import Integer")
    # 在 IPython 中运行命令，设置变量 'a' 的值为整数 1
    app.run_cell("a = 1")
    # 检查 'a' 是否为 int 类型
    assert isinstance(app.user_ns['a'], int)

    # 启用自动整数 sympification 功能
    enable_automatic_int_sympification(app)
    # 在 IPython 中运行命令，设置变量 'a' 的值为分数 1/2
    app.run_cell("a = 1/2")
    # 检查 'a' 是否为 Rational 类型
    assert isinstance(app.user_ns['a'], Rational)
    # 在 IPython 中运行命令，设置变量 'a' 的值为 Integer 对象
    app.run_cell("a = 1")
    # 检查 'a' 是否为 Integer 类型
    assert isinstance(app.user_ns['a'], Integer)
    # 在 IPython 中运行命令，设置变量 'a' 的值为整数 1
    app.run_cell("a = int(1)")
    # 检查 'a' 是否为 int 类型
    assert isinstance(app.user_ns['a'], int)
    # 在 IPython 中运行命令，设置变量 'a' 的值为表达式 (1/2)
    app.run_cell("a = (1/\n2)")
    # 检查 'a' 是否等于 Rational(1, 2)
    assert app.user_ns['a'] == Rational(1, 2)
    # TODO: How can we test that the output of a SyntaxError is the original
    # input, not the transformed input?

# 定义测试函数 test_ipythonprinting，用于测试 IPython 的打印功能
def test_ipythonprinting():
    # 初始化 IPython 会话对象
    app = init_ipython_session()
    # 运行代码单元，获取当前 IPython 实例
    app.run_cell("ip = get_ipython()")
    # 获取 IPython 实例对象
    app.run_cell("inst = ip.instance()")
    # 获取显示格式化对象
    app.run_cell("format = inst.display_formatter.format")
    # 导入 sympy 的 Symbol 符号
    app.run_cell("from sympy import Symbol")

    # 在没有打印扩展的情况下进行打印
    app.run_cell("a = format(Symbol('pi'))")
    app.run_cell("a2 = format(Symbol('pi')**2)")
    # 处理 IPython 1.0 开始的 API 变化
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] == "pi"
        assert app.user_ns['a2']['text/plain'] == "pi**2"
    else:
        assert app.user_ns['a'][0]['text/plain'] == "pi"
        assert app.user_ns['a2'][0]['text/plain'] == "pi**2"

    # 加载打印扩展
    app.run_cell("from sympy import init_printing")
    app.run_cell("init_printing()")
    # 在打印扩展下进行打印
    app.run_cell("a = format(Symbol('pi'))")
    app.run_cell("a2 = format(Symbol('pi')**2)")
    # 处理 IPython 1.0 开始的 API 变化
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] in ('\N{GREEK SMALL LETTER PI}', 'pi')
        assert app.user_ns['a2']['text/plain'] in (' 2\n\N{GREEK SMALL LETTER PI} ', '  2\npi ')
    else:
        assert app.user_ns['a'][0]['text/plain'] in ('\N{GREEK SMALL LETTER PI}', 'pi')
        assert app.user_ns['a2'][0]['text/plain'] in (' 2\n\N{GREEK SMALL LETTER PI} ', '  2\npi ')
def test_print_builtin_option():
    # 初始化并设置 IPython 会话
    app = init_ipython_session()
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("from sympy import Symbol")
    app.run_cell("from sympy import init_printing")

    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # 处理 IPython 1.0 之后的 API 更改
    if int(ipython.__version__.split(".")[0]) < 1:
        text = app.user_ns['a']['text/plain']
        raises(KeyError, lambda: app.user_ns['a']['text/latex'])
    else:
        text = app.user_ns['a'][0]['text/plain']
        raises(KeyError, lambda: app.user_ns['a'][0]['text/latex'])
    
    # XXX: 如何使此测试忽略终端宽度？如果终端宽度太窄，此测试将失败。
    assert text in ("{pi: 3.14, n_i: 3}",
                    '{n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3, \N{GREEK SMALL LETTER PI}: 3.14}',
                    "{n_i: 3, pi: 3.14}",
                    '{\N{GREEK SMALL LETTER PI}: 3.14, n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3}')

    # 如果启用默认打印设置，则字典应该呈现为整个字典的 LaTeX 版本：${\pi: 3.14, n_i: 3}$
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("init_printing(use_latex=True)")
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # 处理 IPython 1.0 之后的 API 更改
    if int(ipython.__version__.split(".")[0]) < 1:
        text = app.user_ns['a']['text/plain']
        latex = app.user_ns['a']['text/latex']
    else:
        text = app.user_ns['a'][0]['text/plain']
        latex = app.user_ns['a'][0]['text/latex']
    assert text in ("{pi: 3.14, n_i: 3}",
                    '{n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3, \N{GREEK SMALL LETTER PI}: 3.14}',
                    "{n_i: 3, pi: 3.14}",
                    '{\N{GREEK SMALL LETTER PI}: 3.14, n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3}')
    assert latex == r'$\displaystyle \left\{ n_{i} : 3, \  \pi : 3.14\right\}$'

    # 具有 _latex 重载的对象也应该被我们的元组打印器处理。
    app.run_cell("""\
    class WithOverload:
        def _latex(self, printer):
            return r"\\LaTeX"
    """)
    app.run_cell("a = format((WithOverload(),))")
    # 处理 IPython 1.0 之后的 API 更改
    if int(ipython.__version__.split(".")[0]) < 1:
        latex = app.user_ns['a']['text/latex']
    else:
        latex = app.user_ns['a'][0]['text/latex']
    assert latex == r'$\displaystyle \left( \LaTeX,\right)$'

    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("init_printing(use_latex=True, print_builtin=False)")
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # 处理 IPython 1.0 之后的 API 更改
    # 检查 IPython 库的版本，判断执行不同的代码路径
    if int(ipython.__version__.split(".")[0]) < 1:
        # 如果 IPython 版本小于 1，从用户命名空间中获取 'a' 变量的 'text/plain' 数据
        text = app.user_ns['a']['text/plain']
        # 预期会抛出 KeyError 异常，因为 'text/latex' 在 'a' 变量中不存在
        raises(KeyError, lambda: app.user_ns['a']['text/latex'])
    else:
        # 如果 IPython 版本大于等于 1，从用户命名空间中获取 'a[0]' 变量的 'text/plain' 数据
        text = app.user_ns['a'][0]['text/plain']
        # 预期会抛出 KeyError 异常，因为 'text/latex' 在 'a[0]' 变量中不存在
        raises(KeyError, lambda: app.user_ns['a'][0]['text/latex'])
    
    # 断言 text 变量的值应该是以下字符串中的一个
    assert text in ("{pi: 3.14, n_i: 3}", "{n_i: 3, pi: 3.14}")
# 测试内置容器的函数

def test_builtin_containers():
    # 初始化并设置 IPython 会话
    app = init_ipython_session()
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("from sympy import init_printing, Matrix")
    app.run_cell('init_printing(use_latex=True, use_unicode=False)')

    # 确保不会对不应该漂亮打印的容器进行漂亮打印
    app.run_cell('a = format((True, False))')
    app.run_cell('import sys')
    app.run_cell('b = format(sys.flags)')
    app.run_cell('c = format((Matrix([1, 2]),))')

    # 处理 IPython 1.0 开始的 API 变化
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] ==  '(True, False)'
        assert 'text/latex' not in app.user_ns['a']
        assert app.user_ns['b']['text/plain'][:10] == 'sys.flags('
        assert 'text/latex' not in app.user_ns['b']
        assert app.user_ns['c']['text/plain'] == \
"""\
 [1]  \n\
([ ],)
 [2]  \
"""
        assert app.user_ns['c']['text/latex'] == '$\\displaystyle \\left( \\left[\\begin{matrix}1\\\\2\\end{matrix}\\right],\\right)$'
    else:
        assert app.user_ns['a'][0]['text/plain'] ==  '(True, False)'
        assert 'text/latex' not in app.user_ns['a'][0]
        assert app.user_ns['b'][0]['text/plain'][:10] == 'sys.flags('
        assert 'text/latex' not in app.user_ns['b'][0]
        assert app.user_ns['c'][0]['text/plain'] == \
"""\
 [1]  \n\
([ ],)
 [2]  \
"""
        assert app.user_ns['c'][0]['text/latex'] == '$\\displaystyle \\left( \\left[\\begin{matrix}1\\\\2\\end{matrix}\\right],\\right)$'


def test_matplotlib_bad_latex():
    # 初始化并设置 IPython 会话
    app = init_ipython_session()
    app.run_cell("import IPython")
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("from sympy import init_printing, Matrix")
    app.run_cell("init_printing(use_latex='matplotlib')")

    # 在此上下文中，默认情况下不启用 png 格式化程序
    app.run_cell("inst.display_formatter.formatters['image/png'].enabled = True")

    # 确保 IPython 不会引发任何警告
    app.run_cell("import warnings")
    # IPython.core.formatters.FormatterWarning 在 IPython 2.0 中引入
    if int(ipython.__version__.split(".")[0]) < 2:
        app.run_cell("warnings.simplefilter('error')")
    else:
        app.run_cell("warnings.simplefilter('error', IPython.core.formatters.FormatterWarning)")

    # 这应该不会引发异常
    app.run_cell("a = format(Matrix([1, 2, 3]))")

    # issue 9799
    app.run_cell("from sympy import Piecewise, Symbol, Eq")
    app.run_cell("x = Symbol('x'); pw = format(Piecewise((1, Eq(x, 0)), (0, True)))")


def test_override_repr_latex():
    # 初始化并设置 IPython 会话
    # 初始化一个 IPython 会话应用程序
    app = init_ipython_session()
    # 在 IPython 中运行一条命令，导入 IPython 库
    app.run_cell("import IPython")
    # 在 IPython 中运行一条命令，获取当前 IPython 实例
    app.run_cell("ip = get_ipython()")
    # 在 IPython 中运行一条命令，获取 IPython 实例的 instance 对象
    app.run_cell("inst = ip.instance()")
    # 在 IPython 中运行一条命令，获取显示格式化函数 format
    app.run_cell("format = inst.display_formatter.format")
    # 在 IPython 中运行一条命令，启用 'text/latex' 格式的显示格式化
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    # 在 IPython 中运行一条命令，从 sympy 库中导入 init_printing 函数
    app.run_cell("from sympy import init_printing")
    # 在 IPython 中运行一条命令，从 sympy 库中导入 Symbol 类
    app.run_cell("from sympy import Symbol")
    # 在 IPython 中运行一条命令，初始化 sympy 打印系统使用 LaTeX
    app.run_cell("init_printing(use_latex=True)")
    # 在 IPython 中运行多行命令，定义一个名为 SymbolWithOverload 的类，继承自 Symbol 类
    app.run_cell("""\
    class SymbolWithOverload(Symbol):
        def _repr_latex_(self):
            return r"Hello " + super()._repr_latex_() + " world"
    """)
    # 在 IPython 中运行一条命令，格式化并获取 SymbolWithOverload 实例 's' 的输出
    app.run_cell("a = format(SymbolWithOverload('s'))")
    
    # 根据 IPython 版本判断条件，选择合适的方式获取 LaTeX 表示
    if int(ipython.__version__.split(".")[0]) < 1:
        # 如果 IPython 版本小于 1，获取 'text/latex' 格式的输出
        latex = app.user_ns['a']['text/latex']
    else:
        # 如果 IPython 版本不小于 1，获取索引为 0 的 'text/latex' 格式的输出
        latex = app.user_ns['a'][0]['text/latex']
    
    # 断言获取到的 LaTeX 输出符合预期的格式
    assert latex == r'Hello $\displaystyle s$ world'
```