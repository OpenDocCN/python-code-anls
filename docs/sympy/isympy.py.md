# `D:\src\scipysrc\sympy\isympy.py`

```
"""
Python shell for SymPy.

This is just a normal Python shell (IPython shell if you have the
IPython package installed), that executes the following commands for
the user:

    >>> from __future__ import division
    >>> from sympy import *
    >>> x, y, z, t = symbols('x y z t')
    >>> k, m, n = symbols('k m n', integer=True)
    >>> f, g, h = symbols('f g h', cls=Function)
    >>> init_printing()

So starting 'isympy' is equivalent to starting Python (or IPython) and
executing the above commands by hand.  It is intended for easy and quick
experimentation with SymPy. isympy is a good way to use SymPy as an
interactive calculator. If you have IPython and Matplotlib installed, then
interactive plotting is enabled by default.
"""

# COMMAND LINE OPTIONS
# --------------------

-c CONSOLE, --console=CONSOLE
# 指定使用的控制台类型（Python 或 IPython），默认情况下使用 IPython（如果安装了），否则使用 Python。
# 例如：
#     $ isympy -c python
# CONSOLE 必须是 'ipython' 或 'python'

-p PRETTY, --pretty PRETTY
# 设置 SymPy 的漂亮打印选项。启用漂亮打印时，可以使用 Unicode 或 ASCII 进行打印。
# 默认情况下使用漂亮打印（如果终端支持的话使用 Unicode）。
# 当 PRETTY 设为 'no' 时，表达式将不会漂亮打印，而是使用 ASCII。
# 例如：
#     $ isympy -p no
# PRETTY 必须是 'unicode'、'ascii' 或 'no'

-t TYPES, --types=TYPES
# 设置多项式的基础类型。默认情况下，如果安装了 gmpy2 或 gmpy，则使用 gmpy 基础类型，否则回退到 Python 基础类型，速度稍慢。
# 可以手动选择即使安装了 gmpy 也使用 Python 基础类型（例如，用于测试目的）。
# 例如：
#     $ isympy -t python
# TYPES 必须是 'gmpy'、'gmpy1' 或 'python'

-o ORDER, --order ORDER
# 设置打印项的顺序。默认情况下是 lex（按字典顺序排序项，例如 x**2 + x + 1）。
# 可以选择其他排序方式，如 rev-lex（使用反向字典顺序，例如 1 + x + x**2）。
# 例如：
#     $ isympy -o rev-lex
# ORDER 必须是 'lex'、'rev-lex'、'grlex'、'rev-grlex'、'grevlex'、'rev-grevlex'、'old' 或 'none'。

-q, --quiet
# 在启动时只打印 Python 和 SymPy 的版本信息到标准输出。

-d, --doctest
# 使用应该用于 doctest 的相同格式。这是
    # 与 `-c python -p no.` 等效。
-C, --no-cache

    # 禁用缓存机制。禁用缓存可能会显著减慢某些操作的速度。这在测试缓存或基准测试时很有用，因为缓存可能导致时间计算出现误差。

    # 等效于将环境变量 SYMPY_USE_CACHE 设置为 'no'。

-a, --auto-symbols (requires at least IPython 0.11)

    # 自动创建缺失的符号。通常情况下，如果使用尚未实例化的符号名称，则会引发 NameError，但启用此选项后，任何未定义的名称都将自动创建为符号。

    # 注意，此选项仅适用于交互式计算器样式的使用场景。在使用 SymPy 的脚本中，应在顶部实例化符号，以便清楚其含义。

    # 这不会覆盖已定义的任何名称，包括由助记符 QCOSINE 表示的单字符字母（请参阅文档中的“Gotchas and Pitfalls”文档）。可以通过执行 "del name" 删除现有名称。如果名称已定义，执行 "'name' in dir()" 将返回 True。

    # 使用此选项创建的符号具有默认假设。如果想要对符号进行假设，请使用 symbols() 或 var() 创建它们。

    # 最后，此选项仅在顶层命名空间中起作用。例如，在 isympy 中定义一个带有未定义符号的函数将不起作用。

    # 另请参阅 -i 和 -I 选项。

-i, --int-to-Integer (requires at least IPython 0.11)

    # 自动将 int 文字型转换为 Integer。这样做使得像 1/2 这样的表达式将输出 Rational(1, 2)，而不是 0.5。它通过预处理源代码，在所有 int 文字型周围包装 Integer 来实现。请注意，这不会更改分配给变量的 int 文字型的行为，也不会更改返回 int 文字型的函数的行为。

    # 如果需要一个 int，可以使用 int() 包装文字型，例如 int(3)/int(2) 将给出 1.5（假定使用了 __future__ 中的 division）。

-I, --interactive (requires at least IPython 0.11)

    # 等效于 --auto-symbols --int-to-Integer。未来为交互使用方便设计的选项可能会添加到此选项中。

-D, --debug

    # 启用调试输出。这与设置环境变量 SYMPY_DEBUG 为 'True' 是一样的。调试状态在 isympy 中的 SYMPY_DEBUG 变量中设置。

-- IPython options

    # 此外，您可以直接将命令行选项传递给 IPython 解释器（不支持标准 Python shell）。但是，需要在两种类型的选项之间添加 '--' 分隔符，例如启动横幅选项和颜色选项。您还需要按照您使用的 IPython 版本的要求输入选项，例如在 IPython 0.11 中，

    #     $isympy -q -- --colors=NoColor
    or older versions of IPython,

        $isympy -q -- -colors NoColor


# 如果使用较旧版本的 IPython 或其他兼容的环境，

    # 以静默模式启动 SymPy，关闭颜色输出功能
    $isympy -q -- -colors NoColor
"""
See also isympy --help.
"""

# 导入必要的标准库模块
import os
import sys

# DO NOT IMPORT SYMPY HERE! Or the setting of the sympy environment variables
# by the command line will break.

# 定义主函数
def main() -> None:
    # 导入需要的参数解析器和帮助信息格式化类
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    # 初始化版本信息为 None
    VERSION = None
    # 如果命令行参数包含 '--version'
    if '--version' in sys.argv:
        # 我们不能在此之前导入 sympy，因为像 -C 和 -t 这样的标志会设置环境变量，
        # 这些环境变量必须在导入 SymPy 之前设置好。我们唯一需要导入它的是为了获取版本信息，
        # 这仅在使用 --version 标志时才重要。
        import sympy
        # 获取 SymPy 的版本信息
        VERSION = sympy.__version__

    # 定义用法信息
    usage = 'isympy [options] -- [ipython options]'
    # 创建参数解析器对象
    parser = ArgumentParser(
        usage=usage,
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )

    # 添加 --version 参数，用于显示版本信息
    parser.add_argument('--version', action='version', version=VERSION)

    # 添加 --console 参数，选择交互会话类型
    parser.add_argument(
        '-c', '--console',
        dest='console',
        action='store',
        default=None,
        choices=['ipython', 'python'],
        metavar='CONSOLE',
        help='select type of interactive session: ipython | python; defaults '
        'to ipython if IPython is installed, otherwise python')

    # 添加 --pretty 参数，设置漂亮打印选项
    parser.add_argument(
        '-p', '--pretty',
        dest='pretty',
        action='store',
        default=None,
        metavar='PRETTY',
        choices=['unicode', 'ascii', 'no'],
        help='setup pretty printing: unicode | ascii | no; defaults to '
        'unicode printing if the terminal supports it, otherwise ascii')

    # 添加 --types 参数，设置基础类型选项
    parser.add_argument(
        '-t', '--types',
        dest='types',
        action='store',
        default=None,
        metavar='TYPES',
        choices=['gmpy', 'gmpy1', 'python'],
        help='setup ground types: gmpy | gmpy1 | python; defaults to gmpy if gmpy2 '
        'or gmpy is installed, otherwise python')

    # 添加 --order 参数，设置项的排序方式选项
    parser.add_argument(
        '-o', '--order',
        dest='order',
        action='store',
        default=None,
        metavar='ORDER',
        choices=['lex', 'grlex', 'grevlex', 'rev-lex', 'rev-grlex', 'rev-grevlex', 'old', 'none'],
        help='setup ordering of terms: [rev-]lex | [rev-]grlex | [rev-]grevlex | old | none; defaults to lex')

    # 添加 --quiet 参数，设置安静模式选项
    parser.add_argument(
        '-q', '--quiet',
        dest='quiet',
        action='store_true',
        default=False,
        help='print only version information at startup')

    # 添加 --doctest 参数，设置使用 doctest 格式输出选项
    parser.add_argument(
        '-d', '--doctest',
        dest='doctest',
        action='store_true',
        default=False,
        help='use the doctest format for output (you can just copy and paste it)')

    # 添加 -C 或 --no-cache 参数，设置禁用缓存机制选项
    parser.add_argument(
        '-C', '--no-cache',
        dest='cache',
        action='store_false',
        default=True,
        help='disable caching mechanism')
    # 添加一个解析器参数，用于自动构造缺失的符号
    parser.add_argument(
        '-a', '--auto-symbols',
        dest='auto_symbols',
        action='store_true',
        default=False,
        help='automatically construct missing symbols')

    # 添加一个解析器参数，用于自动将整型字面量包装成整数类型
    parser.add_argument(
        '-i', '--int-to-Integer',
        dest='auto_int_to_Integer',
        action='store_true',
        default=False,
        help="automatically wrap int literals with Integer")

    # 添加一个解析器参数，启用交互模式，并等效于 -a -i 选项
    parser.add_argument(
        '-I', '--interactive',
        dest='interactive',
        action='store_true',
        default=False,
        help="equivalent to -a -i")

    # 添加一个解析器参数，启用调试输出
    parser.add_argument(
        '-D', '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='enable debugging output')

    # 解析命令行参数，并将解析结果存储在 options 变量中
    (options, ipy_args) = parser.parse_known_args()

    # 如果命令行参数列表中包含 '--'，则移除该项
    if '--' in ipy_args:
        ipy_args.remove('--')

    # 如果未启用缓存选项，则设置环境变量 SYMPY_USE_CACHE 为 'no'
    if not options.cache:
        os.environ['SYMPY_USE_CACHE'] = 'no'

    # 如果设置了 types 选项，则设置环境变量 SYMPY_GROUND_TYPES
    if options.types:
        os.environ['SYMPY_GROUND_TYPES'] = options.types

    # 如果启用了 debug 选项，则将 SYMPY_DEBUG 环境变量设置为选项值的字符串形式
    if options.debug:
        os.environ['SYMPY_DEBUG'] = str(options.debug)

    # 如果设置了 doctest 选项，则禁用 pretty_print 和 console 选项
    if options.doctest:
        options.pretty = 'no'
        options.console = 'python'

    # 将 console 选项的值存储在 session 变量中
    session = options.console

    # 如果 session 不为 None，则判断是否为 IPython 环境
    if session is not None:
        ipython = session == 'ipython'
    else:
        # 如果未指定 session，尝试导入 IPython 模块，如果导入失败且未设置 quiet 选项，则输出错误消息
        try:
            import IPython
            ipython = True
        except ImportError:
            if not options.quiet:
                from sympy.interactive.session import no_ipython
                print(no_ipython)
            ipython = False

    # 初始化参数字典 args，设置默认值和从命令行参数获取的值
    args = {
        'pretty_print': True,
        'use_unicode':  None,
        'use_latex':    None,
        'order':        None,
        'argv':         ipy_args,
    }

    # 根据 pretty 选项设置 use_unicode 和 pretty_print 选项
    if options.pretty == 'unicode':
        args['use_unicode'] = True
    elif options.pretty == 'ascii':
        args['use_unicode'] = False
    elif options.pretty == 'no':
        args['pretty_print'] = False

    # 如果设置了 order 选项，则将其设置到 args 中
    if options.order is not None:
        args['order'] = options.order

    # 设置 quiet、auto_symbols 和 auto_int_to_Integer 选项到 args 中
    args['quiet'] = options.quiet
    args['auto_symbols'] = options.auto_symbols or options.interactive
    args['auto_int_to_Integer'] = options.auto_int_to_Integer or options.interactive

    # 导入 sympy.interactive 模块，并初始化会话环境
    from sympy.interactive import init_session
    init_session(ipython, **args)
# 如果这个脚本被直接执行（而不是被导入为模块），那么执行以下代码块
if __name__ == "__main__":
    # 调用主函数main()
    main()
```