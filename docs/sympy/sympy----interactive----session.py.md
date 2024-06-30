# `D:\src\scipysrc\sympy\sympy\interactive\session.py`

```
"""Tools for setting up interactive sessions. """

# 导入必要的模块和函数
from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import version_tuple
from sympy.interactive.printing import init_printing
from sympy.utilities.misc import ARCH

# 预设的执行代码，用于交互式会话初始化
preexec_source = """\
from sympy import *
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)
init_printing()
"""

# 详细信息的消息模板，显示命令执行情况和文档链接
verbose_message = """\
These commands were executed:
%(source)s
Documentation can be found at https://docs.sympy.org/%(version)s
"""

# IPython 未找到时的警告消息
no_ipython = """\
Could not locate IPython. Having IPython installed is greatly recommended.
See http://ipython.scipy.org for more details. If you use Debian/Ubuntu,
just install the 'ipython' package and start isympy again.
"""


def _make_message(ipython=True, quiet=False, source=None):
    """Create a banner for an interactive session. """
    # 导入所需的模块和函数
    from sympy import __version__ as sympy_version
    from sympy import SYMPY_DEBUG
    import sys
    import os

    # 如果 quiet 为 True，返回空字符串
    if quiet:
        return ""

    # 获取当前 Python 的版本号
    python_version = "%d.%d.%d" % sys.version_info[:3]

    # 根据是否使用 IPython 设置交互环境名称
    if ipython:
        shell_name = "IPython"
    else:
        shell_name = "Python"

    # 组装环境信息的列表
    info = ['ground types: %s' % GROUND_TYPES]

    # 检查是否禁用了缓存，并在信息中添加相应条目
    cache = os.getenv('SYMPY_USE_CACHE')
    if cache is not None and cache.lower() == 'no':
        info.append('cache: off')

    # 如果 SYMPY_DEBUG 开启，则在信息中添加调试状态
    if SYMPY_DEBUG:
        info.append('debugging: on')

    # 组合包含环境信息的字符串
    args = shell_name, sympy_version, python_version, ARCH, ', '.join(info)
    message = "%s console for SymPy %s (Python %s-%s) (%s)\n" % args

    # 如果未提供源代码，则使用预设的执行代码
    if source is None:
        source = preexec_source

    # 初始化用于显示的源代码字符串
    _source = ""

    # 拆分源代码为行并添加提示符 '>>> '，然后连接为单个字符串
    for line in source.split('\n')[:-1]:
        if not line:
            _source += '\n'
        else:
            _source += '>>> ' + line + '\n'

    # 根据 SymPy 版本信息，设置文档链接的版本号
    doc_version = sympy_version
    if 'dev' in doc_version:
        doc_version = "dev"
    else:
        doc_version = "%s/" % doc_version

    # 将详细消息格式化为包含源代码和文档链接的最终消息
    message += '\n' + verbose_message % {'source': _source,
                                         'version': doc_version}

    return message


def int_to_Integer(s):
    """
    Wrap integer literals with Integer.

    This is based on the decistmt example from
    https://docs.python.org/3/library/tokenize.html.

    Only integer literals are converted.  Float literals are left alone.

    Examples
    ========

    >>> from sympy import Integer # noqa: F401
    >>> from sympy.interactive.session import int_to_Integer
    >>> s = '1.2 + 1/2 - 0x12 + a1'
    >>> int_to_Integer(s)
    '1.2 +Integer (1 )/Integer (2 )-Integer (0x12 )+a1 '
    >>> s = 'print (1/2)'
    >>> int_to_Integer(s)
    'print (Integer (1 )/Integer (2 ))'
    >>> exec(s)
    0.5
    >>> exec(int_to_Integer(s))
    1/2
    """
    from tokenize import generate_tokens, untokenize, NUMBER, NAME, OP
    from io import StringIO
    def _is_int(num):
        """
        Returns true if string value num (with token NUMBER) represents an integer.
        """
        # 检查字符串 num 是否表示一个整数，排除包含小数点、虚部标记 'j' 或指数 'e' 的情况
        if '.' in num or 'j' in num.lower() or 'e' in num.lower():
            return False
        return True

    # 从字符串 s 中生成标记流
    g = generate_tokens(StringIO(s).readline)

    result = []
    # 遍历标记流中的每个标记
    for toknum, tokval, _, _, _ in g:
        # 如果标记类型是 NUMBER 并且 tokval 是一个整数
        if toknum == NUMBER and _is_int(tokval):
            # 将替换后的标记序列添加到结果列表中
            result.extend([
                (NAME, 'Integer'),  # 替换为标识整数的名称
                (OP, '('),           # 插入左括号
                (NUMBER, tokval),    # 插入整数值
                (OP, ')')            # 插入右括号
            ])
        else:
            # 对于其他类型的标记，直接添加到结果列表中
            result.append((toknum, tokval))
    
    # 将结果列表重新组装成未标记化的字符串并返回
    return untokenize(result)
def enable_automatic_int_sympification(shell):
    """
    Allow IPython to automatically convert integer literals to Integer.
    """
    import ast
    # 保存旧的 run_cell 方法，以便稍后调用
    old_run_cell = shell.run_cell

    def my_run_cell(cell, *args, **kwargs):
        try:
            # 检查单元格是否存在语法错误。这样，语法错误将显示原始输入，而不是转换后的输入。
            # 这种方式的缺点是，IPython 魔术命令如 %timeit 将无法处理转换后的输入
            # （但另一方面，不期望转换后输入的 IPython 魔术命令将继续工作）。
            ast.parse(cell)
        except SyntaxError:
            pass
        else:
            # 如果没有语法错误，将单元格中的整数转换为 Integer
            cell = int_to_Integer(cell)
        # 调用原始的 run_cell 方法，并返回其结果
        return old_run_cell(cell, *args, **kwargs)

    # 替换 shell 对象的 run_cell 方法为自定义的 my_run_cell 方法
    shell.run_cell = my_run_cell


def enable_automatic_symbols(shell):
    """Allow IPython to automatically create symbols (``isympy -a``). """
    # XXX: 可能应该像上面的 int_to_Integer() 一样使用 tokenize。
    # 这可以避免重新执行代码，这可能会导致细微的问题。
    # 例如：
    #
    # In [1]: a = 1
    #
    # In [2]: for i in range(10):
    #    ...:     a += 1
    #    ...:
    #
    # In [3]: a
    # Out[3]: 11
    #
    # In [4]: a = 1
    #
    # In [5]: for i in range(10):
    #    ...:     a += 1
    #    ...:     print b
    #    ...:
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    #
    # In [6]: a
    # Out[6]: 12
    #
    # 注意，因为 `b` 未定义，for 循环再次执行，但 `a` 已经增加了一次，因此结果是多次增加。
    
    import re
    # 编译用于匹配 NameError 的正则表达式
    re_nameerror = re.compile(
        "name '(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)' is not defined")

    def _handler(self, etype, value, tb, tb_offset=None):
        """Handle :exc:`NameError` exception and allow injection of missing symbols. """
        if etype is NameError and tb.tb_next and not tb.tb_next.tb_next:
            match = re_nameerror.match(str(value))

            if match is not None:
                # XXX: 确保 Symbol 在作用域内。否则会出现无限递归。
                self.run_cell("%(symbol)s = Symbol('%(symbol)s')" %
                              {'symbol': match.group("symbol")}, store_history=False)

                try:
                    # 获取最后一个用户输入的代码行
                    code = self.user_ns['In'][-1]
                except (KeyError, IndexError):
                    pass
                else:
                    # 重新执行最后一个代码行
                    self.run_cell(code, store_history=False)
                    return None
                finally:
                    # 删除刚刚定义的 Symbol
                    self.run_cell("del %s" % match.group("symbol"),
                                  store_history=False)

        # 处理异常，并显示详细的 traceback 信息
        stb = self.InteractiveTB.structured_traceback(
            etype, value, tb, tb_offset=tb_offset)
        self._showtraceback(etype, value, stb)

    # 设置自定义的异常处理器，用于处理 NameError 异常
    shell.set_custom_exc((NameError,), _handler)
# 初始化 IPython 会话或 Python 会话，可以根据提供的参数选择性地启用各种特性和配置
def init_session(ipython=None, pretty_print=True, order=None,
                 use_unicode=None, use_latex=None, quiet=False, auto_symbols=False,
                 auto_int_to_Integer=False, str_printer=None, pretty_printer=None,
                 latex_printer=None, argv=[]):
    """
    Initialize an embedded IPython or Python session. The IPython session is
    initiated with the --pylab option, without the numpy imports, so that
    matplotlib plotting can be interactive.

    Parameters
    ==========

    pretty_print: boolean
        If True, use pretty_print to stringify;
        if False, use sstrrepr to stringify.
    """

    # 如果 IPython 版本大于等于 0.11，则执行以下操作
    import IPython
    if version_tuple(IPython.__version__) >= version_tuple('0.11'):
        if not ipython:
            # 如果未提供 ipython 对象，则根据 IPython 版本选择合适的终端应用对象
            if version_tuple(IPython.__version__) >= version_tuple('1.0'):
                from IPython.terminal import ipapp
            else:
                from IPython.frontend.terminal import ipapp
            app = ipapp.TerminalIPythonApp()

            # 初始化应用对象，设置不显示 IPython 横幅
            app.display_banner = False
            app.initialize(argv)

            ipython = app.shell  # 将 shell 对象设置为 IPython 应用的 shell

        # 如果启用了自动符号功能，则调用相应函数
        if auto_symbols:
            enable_automatic_symbols(ipython)
        # 如果启用了自动将整数转为 SymPy Integer 的功能，则调用相应函数
        if auto_int_to_Integer:
            enable_automatic_int_sympification(ipython)

        # 返回初始化好的 IPython shell 对象
        return ipython
    else:
        # 如果 IPython 版本低于 0.11，则使用旧版 API 创建 IPython session
        from IPython.Shell import make_IPython
        return make_IPython(argv)


# 初始化一个普通的 Python 会话，包含 readline 支持和 tab 自动补全功能
def init_python_session():
    """Construct new Python session. """
    from code import InteractiveConsole

    class SymPyConsole(InteractiveConsole):
        """An interactive console with readline support. """

        def __init__(self):
            ns_locals = {}
            InteractiveConsole.__init__(self, locals=ns_locals)
            try:
                import rlcompleter
                import readline
            except ImportError:
                pass
            else:
                import os
                import atexit

                # 设置 tab 自动补全和历史记录功能
                readline.set_completer(rlcompleter.Completer(ns_locals).complete)
                readline.parse_and_bind('tab: complete')

                if hasattr(readline, 'read_history_file'):
                    history = os.path.expanduser('~/.sympy-history')

                    try:
                        readline.read_history_file(history)
                    except OSError:
                        pass

                    atexit.register(readline.write_history_file, history)

    # 返回自定义的 SymPyConsole 实例
    return SymPyConsole()
    order: string or None
        # 参数order用于指定多项式的排序方式，默认为lex字典序
        There are a few different settings for this parameter:
        lex (default), which is lexographic order;
        grlex, which is graded lexographic order;
        grevlex, which is reversed graded lexographic order;
        old, which is used for compatibility reasons and for long expressions;
        None, which sets it to lex.

    use_unicode: boolean or None
        # 参数use_unicode用于指定是否使用Unicode字符，默认为None
        If True, use unicode characters;
        if False, do not use unicode characters.

    use_latex: boolean or None
        # 参数use_latex用于指定是否在IPython GUI中使用LaTeX渲染，默认为None
        If True, use latex rendering if IPython GUI's;
        if False, do not use latex rendering.

    quiet: boolean
        # 参数quiet用于指定是否在init_session初始化时打印状态信息，默认为False
        If True, init_session will not print messages regarding its status;
        if False, init_session will print messages regarding its status.

    auto_symbols: boolean
        # 参数auto_symbols用于指定IPython是否自动创建符号，默认为False
        If True, IPython will automatically create symbols for you.
        If False, it will not.
        The default is False.

    auto_int_to_Integer: boolean
        # 参数auto_int_to_Integer用于指定IPython是否将整数自动包装为Integer类型，默认为False
        If True, IPython will automatically wrap int literals with Integer, so
        that things like 1/2 give Rational(1, 2).
        If False, it will not.
        The default is False.

    ipython: boolean or None
        # 参数ipython用于指定初始化是否为IPython控制台，默认为None
        If True, printing will initialize for an IPython console;
        if False, printing will initialize for a normal console;
        The default is None, which automatically determines whether we are in
        an ipython instance or not.

    str_printer: function, optional, default=None
        # 参数str_printer用于指定自定义的字符串打印函数，默认为None
        A custom string printer function. This should mimic
        sympy.printing.sstrrepr().

    pretty_printer: function, optional, default=None
        # 参数pretty_printer用于指定自定义的pretty printer函数，默认为None
        A custom pretty printer. This should mimic sympy.printing.pretty().

    latex_printer: function, optional, default=None
        # 参数latex_printer用于指定自定义的LaTeX打印函数，默认为None
        A custom LaTeX printer. This should mimic sympy.printing.latex().
        This should mimic sympy.printing.latex().

    argv: list of arguments for IPython
        # 参数argv用于指定传递给IPython的参数列表
        See sympy.bin.isympy for options that can be used to initialize IPython.

    See Also
    ========

    sympy.interactive.printing.init_printing: for examples and the rest of the parameters.

    Examples
    ========

    >>> from sympy import init_session, Symbol, sin, sqrt
    >>> sin(x) #doctest: +SKIP
    NameError: name 'x' is not defined
    >>> init_session() #doctest: +SKIP
    >>> sin(x) #doctest: +SKIP
    sin(x)
    >>> sqrt(5) #doctest: +SKIP
      ___
    \\/ 5
    >>> init_session(pretty_print=False) #doctest: +SKIP
    >>> sqrt(5) #doctest: +SKIP
    sqrt(5)
    >>> y + x + y**2 + x**2 #doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_session(order='grlex') #doctest: +SKIP
    >>> y + x + y**2 + x**2 #doctest: +SKIP
    x**2 + y**2 + x + y
    >>> init_session(order='grevlex') #doctest: +SKIP
    >>> y * x**2 + x * y**2 #doctest: +SKIP
    x**2*y + x*y**2
    >>> init_session(order='old') #doctest: +SKIP
    >>> x**2 + y**2 + x + y #doctest: +SKIP
    x + y + x**2 + y**2
    >>> theta = Symbol('theta') #doctest: +SKIP
    >>> theta #doctest: +SKIP
    theta
    >>> init_session(use_unicode=True) #doctest: +SKIP
    >>> theta # doctest: +SKIP
    \u03b8
    """
    import sys  # 导入系统模块，用于访问系统相关功能

    in_ipython = False  # 初始化变量，表示当前不在 IPython 环境中

    if ipython is not False:  # 如果 ipython 不为 False，则尝试加载 IPython 模块
        try:
            import IPython
        except ImportError:  # 如果导入失败
            if ipython is True:
                raise RuntimeError("IPython is not available on this system")  # 如果 ipython 要求为 True，抛出运行时错误
            ip = None
        else:
            try:
                from IPython import get_ipython
                ip = get_ipython()  # 获取当前的 IPython 实例
            except ImportError:
                ip = None
        in_ipython = bool(ip)  # 更新 in_ipython 标志，表示是否在 IPython 环境中
        if ipython is None:
            ipython = in_ipython  # 如果 ipython 为 None，则根据实际情况更新 ipython 变量

    if ipython is False:  # 如果不在 IPython 环境中
        ip = init_python_session()  # 初始化一个 Python 会话
        mainloop = ip.interact  # 将 mainloop 设置为与用户交互的函数
    else:
        ip = init_ipython_session(ip, argv=argv, auto_symbols=auto_symbols,
                                  auto_int_to_Integer=auto_int_to_Integer)  # 初始化一个 IPython 会话

        if version_tuple(IPython.__version__) >= version_tuple('0.11'):
            # IPython 版本大于等于 0.11，使用 run_cell 替代 runsource，不保存到 IPython 历史记录
            ip.runsource = lambda src, symbol='exec': ip.run_cell(src, False)

            # 尝试启用 pylab 进行交互式绘图
            try:
                ip.enable_pylab(import_all=False)
            except Exception:
                # 如果 matplotlib 没有安装，则抛出 ImportError
                # 在没有显示器或后端问题时可能会引发其他错误，这里使用通用的异常捕获
                pass
        if not in_ipython:
            mainloop = ip.mainloop  # 如果不在 IPython 环境中，将 mainloop 设置为 IPython 的主循环函数

    if auto_symbols and (not ipython or version_tuple(IPython.__version__) < version_tuple('0.11')):
        raise RuntimeError("automatic construction of symbols is possible only in IPython 0.11 or above")
    if auto_int_to_Integer and (not ipython or version_tuple(IPython.__version__) < version_tuple('0.11')):
        raise RuntimeError("automatic int to Integer transformation is possible only in IPython 0.11 or above")

    _preexec_source = preexec_source  # 获取预执行的源码

    ip.runsource(_preexec_source, symbol='exec')  # 在 IPython 或 Python 环境中执行预执行的源码

    init_printing(pretty_print=pretty_print, order=order,
                  use_unicode=use_unicode, use_latex=use_latex, ip=ip,
                  str_printer=str_printer, pretty_printer=pretty_printer,
                  latex_printer=latex_printer)  # 初始化打印设置

    message = _make_message(ipython, quiet, _preexec_source)  # 根据参数生成消息

    if not in_ipython:
        print(message)  # 如果不在 IPython 环境中，打印消息
        mainloop()  # 执行主循环
        sys.exit('Exiting ...')  # 退出程序
    else:
        print(message)  # 如果在 IPython 环境中，打印消息
        import atexit
        atexit.register(lambda: print("Exiting ...\n"))  # 在程序退出时注册退出消息的打印函数
```