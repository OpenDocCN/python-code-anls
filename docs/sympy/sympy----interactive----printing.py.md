# `D:\src\scipysrc\sympy\sympy\interactive\printing.py`

```
"""Tools for setting up printing in interactive sessions. """

# 导入 sympy 外部的版本工具
from sympy.external.importtools import version_tuple
# 导入 BytesIO 类
from io import BytesIO

# 导入 sympy 的默认 LaTeX 打印函数
from sympy.printing.latex import latex as default_latex
# 导入 sympy 的预览函数
from sympy.printing.preview import preview
# 导入调试工具
from sympy.utilities.misc import debug
# 导入打印默认设置
from sympy.printing.defaults import Printable


def _init_python_printing(stringify_func, **settings):
    """Setup printing in Python interactive session. """
    import sys
    import builtins

    def _displayhook(arg):
        """Python's pretty-printer display hook.

           This function was adapted from:

            https://www.python.org/dev/peps/pep-0217/

        """
        # 如果参数不为空，则进行打印和保存上一个值
        if arg is not None:
            builtins._ = None
            print(stringify_func(arg, **settings))
            builtins._ = arg

    sys.displayhook = _displayhook


def _init_ipython_printing(ip, stringify_func, use_latex, euler, forecolor,
                           backcolor, fontsize, latex_mode, print_builtin,
                           latex_printer, scale, **settings):
    """Setup printing in IPython interactive session. """
    try:
        from IPython.lib.latextools import latex_to_png
    except ImportError:
        pass

    # 根据 IPython 的主题色自动推测最佳前景色
    if forecolor is None:
        color = ip.colors.lower()
        if color == 'lightbg':
            forecolor = 'Black'
        elif color == 'linux':
            forecolor = 'White'
        else:
            # 如果无法推测，选择灰色作为前景色
            forecolor = 'Gray'
        debug("init_printing: Automatic foreground color:", forecolor)

    # 根据 LaTeX 使用模式设置额外的前导信息
    if use_latex == "svg":
        extra_preamble = "\n\\special{color %s}" % forecolor
    else:
        extra_preamble = ""

    # 设置图像大小和偏移量
    imagesize = 'tight'
    offset = "0cm,0cm"
    resolution = round(150*scale)
    dvi = r"-T %s -D %d -bg %s -fg %s -O %s" % (
        imagesize, resolution, backcolor, forecolor, offset)
    dvioptions = dvi.split()

    # 计算 SVG 比例
    svg_scale = 150/72*scale
    dvioptions_svg = ["--no-fonts", "--scale={}".format(svg_scale)]

    debug("init_printing: DVIOPTIONS:", dvioptions)
    debug("init_printing: DVIOPTIONS_SVG:", dvioptions_svg)

    # 设置 LaTeX 函数
    latex = latex_printer or default_latex

    def _print_plain(arg, p, cycle):
        """caller for pretty, for use in IPython 0.11"""
        # 如果可以打印，则使用 stringify_func 打印
        if _can_print(arg):
            p.text(stringify_func(arg))
        else:
            p.text(IPython.lib.pretty.pretty(arg))
    # 定义一个函数，用于生成预览的 PNG 图像数据
    def _preview_wrapper(o):
        # 创建一个字节流对象，用于存储表达式的预览数据
        exprbuffer = BytesIO()
        try:
            # 调用预览函数生成 PNG 格式的表达式预览图，并将结果写入 exprbuffer
            preview(o, output='png', viewer='BytesIO', euler=euler,
                    outputbuffer=exprbuffer, extra_preamble=extra_preamble,
                    dvioptions=dvioptions, fontsize=fontsize)
        except Exception as e:
            # 如果发生异常，打印调试信息并重新抛出异常
            debug("png printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        # 返回生成的 PNG 图像数据
        return exprbuffer.getvalue()

    # 定义一个函数，用于生成预览的 SVG 图像数据
    def _svg_wrapper(o):
        # 创建一个字节流对象，用于存储表达式的 SVG 格式预览数据
        exprbuffer = BytesIO()
        try:
            # 调用预览函数生成 SVG 格式的表达式预览图，并将结果写入 exprbuffer
            preview(o, output='svg', viewer='BytesIO', euler=euler,
                    outputbuffer=exprbuffer, extra_preamble=extra_preamble,
                    dvioptions=dvioptions_svg, fontsize=fontsize)
        except Exception as e:
            # 如果发生异常，打印调试信息并重新抛出异常
            debug("svg printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        # 返回生成的 SVG 图像数据，并转换为 UTF-8 编码的字符串格式
        return exprbuffer.getvalue().decode('utf-8')

    # 定义一个函数，用于生成 Matplotlib 渲染的 PNG 图像数据
    def _matplotlib_wrapper(o):
        # Mathtext 不能渲染某些 LaTeX 命令，例如无法渲染 array 或 matrix 等环境。因此在这里，
        # 确保如果 Mathtext 无法渲染，则返回 None。
        try:
            try:
                # 尝试将 LaTeX 表达式转换为 PNG 图像数据
                return latex_to_png(o, color=forecolor, scale=scale)
            except TypeError:  # 旧版本的 IPython 不支持 color 和 scale 参数
                return latex_to_png(o)
        except ValueError as e:
            # 捕获 ValueError 异常，打印调试信息，并返回 None
            debug('matplotlib exception caught:', repr(e))
            return None

    # 内置 SymPy 打印器的钩子方法列表
    printing_hooks = ('_latex', '_sympystr', '_pretty', '_sympyrepr')
    def _can_print(o):
        """检查对象 o 是否可以使用 SymPy 打印机打印。

        如果 o 是一个容器类型，只有当 o 中的每个元素都可以以这种方式打印时，返回 True。
        """

        try:
            # 如果你添加了另一种类型，请确保在本文件后面的 printable_types 中也添加了它

            builtin_types = (list, tuple, set, frozenset)
            if isinstance(o, builtin_types):
                # 如果对象是自定义子类，并且具有自定义的 str 或 repr，则使用该自定义方法。
                if (type(o).__str__ not in (i.__str__ for i in builtin_types) or
                    type(o).__repr__ not in (i.__repr__ for i in builtin_types)):
                    return False
                return all(_can_print(i) for i in o)
            elif isinstance(o, dict):
                return all(_can_print(i) and _can_print(o[i]) for i in o)
            elif isinstance(o, bool):
                return False
            elif isinstance(o, Printable):
                # SymPy 已知的类型
                return True
            elif any(hasattr(o, hook) for hook in printing_hooks):
                # 类型自行添加支持的
                return True
            elif isinstance(o, (float, int)) and print_builtin:
                return True
            return False
        except RuntimeError:
            return False
            # 当达到最大递归深度时使用此处。
            # 由于 RecursionError 适用于 Python 3.5+ 版本
            # 因此这里是为了防止旧版本的 RecursionError。

    def _print_latex_png(o):
        """
        返回一个由外部 LaTeX 发行版渲染的 PNG，如果失败则回退到 matplotlib 渲染。
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _preview_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e),
                      ' Falling back to matplotlib backend')
                if latex_mode != 'inline':
                    s = latex(o, mode='inline', **settings)
                return _matplotlib_wrapper(s)

    def _print_latex_svg(o):
        """
        返回一个由外部 LaTeX 发行版渲染的 SVG，如果失败则无法回退。
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _svg_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e),
                      ' No fallback available.')
    def _print_latex_matplotlib(o):
        """
        A function that returns a PNG rendered by mathtext.

        Parameters:
        o -- Object to be printed using LaTeX mathtext.

        Returns:
        PNG image rendered from LaTeX mathtext.
        """
        if _can_print(o):
            # Convert object o to LaTeX inline mode string
            s = latex(o, mode='inline', **settings)
            # Return PNG image rendered by matplotlib wrapper
            return _matplotlib_wrapper(s)

    def _print_latex_text(o):
        """
        A function to generate the LaTeX representation of SymPy expressions.

        Parameters:
        o -- Object to be converted to LaTeX.

        Returns:
        LaTeX representation of SymPy expression.
        """
        if _can_print(o):
            # Convert object o to LaTeX with specified mode and settings
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                # Return LaTeX string for plain mode
                return '$\\displaystyle %s$' % s
            # Return regular LaTeX string
            return s

    def _result_display(self, arg):
        """
        IPython's pretty-printer display hook, for use in IPython 0.10

        Parameters:
        self -- IPython instance.
        arg -- Argument to be displayed.

        Prints the formatted output based on IPython's pprint setting.
        """
        if self.rc.pprint:
            # Convert argument to string representation
            out = stringify_func(arg)

            if '\n' in out:
                print()  # Print empty line if output contains newline

            print(out)  # Print formatted output
        else:
            print(repr(arg))  # Print the raw representation of the argument

    import IPython
    else:
        ip.set_hook('result_display', _result_display)
# 判断给定的 shell 实例是否是 IPython shell
def _is_ipython(shell):
    # 快速检查是否需要导入 IPython 模块
    from sys import modules
    # 如果没有导入 IPython 模块，则返回 False
    if 'IPython' not in modules:
        return False
    try:
        # 尝试导入 IPython 核心交互式 shell 类
        from IPython.core.interactiveshell import InteractiveShell
    except ImportError:
        # 处理 IPython 版本小于 0.11 的情况
        try:
            from IPython.iplib import InteractiveShell
        except ImportError:
            # 如果 IPython 发生不向后兼容的变化，则返回 False
            # 这种情况下我们可能需要发出警告
            return False
    # 返回 shell 实例是否是 InteractiveShell 类型的结果
    return isinstance(shell, InteractiveShell)

# 用于 doctester 的全局变量，用来覆盖默认的 no_global 设置
NO_GLOBAL = False

# 初始化打印设置函数，根据环境不同初始化漂亮打印机
def init_printing(pretty_print=True, order=None, use_unicode=None,
                  use_latex=None, wrap_line=None, num_columns=None,
                  no_global=False, ip=None, euler=False, forecolor=None,
                  backcolor='Transparent', fontsize='10pt',
                  latex_mode='plain', print_builtin=True,
                  str_printer=None, pretty_printer=None,
                  latex_printer=None, scale=1.0, **settings):
    """
    根据环境初始化漂亮打印设置。

    参数
    ==========

    pretty_print : bool, 默认为 True
        如果为 True，则使用 pretty_print 函数或提供的漂亮打印机打印字符串；
        如果为 False，则使用 sstrrepr 函数或提供的字符串打印机打印。
    order : string 或 None，默认为 'lex'
        此参数有几种不同的设置：
        'lex'（默认）：词法顺序；
        'grlex'：分级词法顺序；
        'grevlex'：反向分级词法顺序；
        'old'：出于兼容性和长表达式的原因；
        None：设置为 lex。
    use_unicode : bool 或 None，默认为 None
        如果为 True，则使用 Unicode 字符；
        如果为 False，则不使用 Unicode 字符；
        如果为 None，则根据环境猜测。
    use_latex : string, bool 或 None，默认为 None
        如果为 True，则在 GUI 接口中使用默认的 LaTeX 渲染（png 和 mathjax）；
        如果为 False，则不使用 LaTeX 渲染；
        如果为 None，则根据环境猜测；
        如果为 'png'，则启用外部 LaTeX 编译器的 LaTeX 渲染，失败时回退到 matplotlib；
        如果为 'matplotlib'，则使用 matplotlib 进行 LaTeX 渲染；
        如果为 'mathjax'，则在 IPython notebook 中启用 MathJax 渲染或在 LaTeX 文档中进行文本渲染；
        如果为 'svg'，则使用外部 LaTeX 编译器进行 LaTeX 渲染，没有回退。
    # 是否将行包装在结束时；如果为 True，则行将换行；如果为 False，则不换行，而是继续作为一行。仅在 pretty_print 为 True 时有效。
    wrap_line: bool

    # 如果为 int，则设置换行前的列数为 num_columns；如果为 None，则设置换行前的列数为终端宽度。仅在 pretty_print 为 True 时有效。
    num_columns: int or None, default=None

    # 如果为 True，则设置成全局系统范围内的设置；如果为 False，则仅在此控制台/会话中使用。
    no_global: bool, default=False

    # 一个交互式控制台的实例，可以是 IPython 的实例，或者从 code.InteractiveConsole 派生的类。
    ip: An interactive console

    # 是否加载 euler 包到 LaTeX 导言区，以使用手写风格的字体（https://www.ctan.org/pkg/euler）。可选，默认为 False。
    euler: bool, optional, default=False

    # 前景色的 DVI 设置。None 意味着根据 IPython 终端颜色设置的猜测选择 'Black'、'White' 或 'Gray' 中的一个。见注意事项。
    forecolor: string or None, optional, default=None

    # 背景色的 DVI 设置。见注意事项。
    backcolor: string, optional, default='Transparent'

    # 传递给 LaTeX documentclass 函数的字体大小。注意选项受 documentclass 的限制。考虑使用 scale 替代。
    fontsize: string or int, optional, default='10pt'

    # LaTeX 打印机中使用的模式。可以是 'inline'、'plain'、'equation' 或 'equation*' 中的一个。
    latex_mode: string, optional, default='plain'

    # 是否打印浮点数和整数。如果为 True，则打印；如果为 False，则只打印 SymPy 类型。
    print_builtin: boolean, optional, default=True

    # 自定义字符串打印函数。应模仿 .sstrrepr() 的行为。
    str_printer: function, optional, default=None

    # 自定义漂亮打印函数。应模仿 .pretty() 的行为。
    pretty_printer: function, optional, default=None

    # 自定义 LaTeX 打印函数。应模仿 .latex() 的行为。
    latex_printer: function, optional, default=None

    # 当使用 'png' 或 'svg' 后端时，缩放 LaTeX 输出。对高 DPI 屏幕很有用。
    scale: float, optional, default=1.0

    # 任何额外的设置，用于细化 'latex' 和 'pretty' 命令的输出。
    settings:
    # 初始化打印设置，不使用 Unicode 输出
    >>> init_printing(use_unicode=False) # doctest: +SKIP
    # 打印 theta 变量
    >>> theta # doctest: +SKIP
    theta
    # 设置打印顺序为词典序
    >>> init_printing(order='lex') # doctest: +SKIP
    # 打印表达式 y + x + y**2 + x**2
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    # 设置打印顺序为基于总次数的词典序
    >>> init_printing(order='grlex') # doctest: +SKIP
    # 打印表达式 y * x**2 + x * y**2
    >>> str(y * x**2 + x * y**2) # doctest: +SKIP
    x**2*y + x*y**2
    # 设置打印顺序为反词典序
    >>> init_printing(order='grevlex') # doctest: +SKIP
    # 打印表达式 x**2 + y**2 + x + y
    >>> str(x**2 + y**2 + x + y) # doctest: +SKIP
    x**2 + x + y**2 + y
    # 设置打印列数为 10
    >>> init_printing(num_columns=10) # doctest: +SKIP
    # 打印表达式 x**2 + x + y**2 + y
    >>> x**2 + x + y**2 + y # doctest: +SKIP
    x + y +
    x**2 + y**2
    
    Notes
    =====
    
    The foreground and background colors can be selected when using ``'png'`` or
    ``'svg'`` LaTeX rendering. Note that before the ``init_printing`` command is
    executed, the LaTeX rendering is handled by the IPython console and not SymPy.
    
    The colors can be selected among the 68 standard colors known to ``dvips``,
    for a list see [1]_. In addition, the background color can be
    set to  ``'Transparent'`` (which is the default value).
    
    When using the ``'Auto'`` foreground color, the guess is based on the
    ``colors`` variable in the IPython console, see [2]_. Hence, if
    that variable is set correctly in your IPython console, there is a high
    chance that the output will be readable, although manual settings may be
    needed.
    
    
    References
    ==========
    
    .. [1] https://en.wikibooks.org/wiki/LaTeX/Colors#The_68_standard_colors_known_to_dvips
    
    .. [2] https://ipython.readthedocs.io/en/stable/config/details.html#terminal-colors
    
    See Also
    ========
    
    sympy.printing.latex
    sympy.printing.pretty
    
    """
    import sys
    from sympy.printing.printer import Printer
    
    if pretty_print:
        # 如果启用漂亮打印并且提供了自定义的漂亮打印函数，则使用自定义函数
        if pretty_printer is not None:
            stringify_func = pretty_printer
        else:
            # 否则使用默认的漂亮打印函数
            from sympy.printing import pretty as stringify_func
    else:
        # 如果不启用漂亮打印并且提供了自定义的字符串打印函数，则使用自定义函数
        if str_printer is not None:
            stringify_func = str_printer
        else:
            # 否则使用默认的字符串打印函数
            from sympy.printing import sstrrepr as stringify_func
    
    # 检查是否在 IPython 中运行
    in_ipython = False
    if ip is None:
        try:
            # 尝试获取 IPython 环境
            ip = get_ipython()
        except NameError:
            pass
        else:
            # 如果获取到 IPython 环境，则确认在 IPython 中运行
            in_ipython = (ip is not None)
    
    # 如果传入了 IPython 环境并且不在 IPython 中，则再次确认不在 IPython shell 中
    if ip and not in_ipython:
        in_ipython = _is_ipython(ip)
    # 如果在 IPython 环境中且需要美化打印输出
    if in_ipython and pretty_print:
        try:
            import IPython
            # IPython 1.0 以后不再使用 frontend 模块，直接从 terminal 模块导入以避免显示废弃消息
            if version_tuple(IPython.__version__) >= version_tuple('1.0'):
                from IPython.terminal.interactiveshell import TerminalInteractiveShell
            else:
                from IPython.frontend.terminal.interactiveshell import TerminalInteractiveShell
            from code import InteractiveConsole
        except ImportError:
            pass
        else:
            # 如果不是在 InteractiveConsole 或 TerminalInteractiveShell 中，并且不是在 ipython-console 中
            if not isinstance(ip, (InteractiveConsole, TerminalInteractiveShell)) \
                    and 'ipython-console' not in ''.join(sys.argv):
                # 如果 use_unicode 未设置，则设置为 True
                if use_unicode is None:
                    debug("init_printing: Setting use_unicode to True")
                    use_unicode = True
                # 如果 use_latex 未设置，则设置为 True
                if use_latex is None:
                    debug("init_printing: Setting use_latex to True")
                    use_latex = True

    # 如果不禁用全局设置
    if not NO_GLOBAL and not no_global:
        # 设置全局打印设置，包括顺序、unicode 使用、换行、列数等
        Printer.set_global_settings(order=order, use_unicode=use_unicode,
                                    wrap_line=wrap_line, num_columns=num_columns)
    else:
        _stringify_func = stringify_func

        # 如果需要美化打印输出
        if pretty_print:
            # 定义一个 lambda 函数用于字符串化表达式，继承之前的设置
            stringify_func = lambda expr, **settings: \
                             _stringify_func(expr, order=order,
                                             use_unicode=use_unicode,
                                             wrap_line=wrap_line,
                                             num_columns=num_columns,
                                             **settings)
        else:
            # 否则直接使用之前定义的字符串化函数
            stringify_func = \
                lambda expr, **settings: _stringify_func(
                    expr, order=order, **settings)

    # 如果在 IPython 环境中
    if in_ipython:
        # 从 settings 中移除 mode 设置，因为无法通过 IPython 的打印机制设置 mode
        mode_in_settings = settings.pop("mode", None)
        if mode_in_settings:
            debug("init_printing: Mode is not able to be set due to internals"
                  "of IPython printing")
        # 初始化 IPython 的打印设置
        _init_ipython_printing(ip, stringify_func, use_latex, euler,
                               forecolor, backcolor, fontsize, latex_mode,
                               print_builtin, latex_printer, scale,
                               **settings)
    else:
        # 初始化普通 Python 的打印设置
        _init_python_printing(stringify_func, **settings)
```