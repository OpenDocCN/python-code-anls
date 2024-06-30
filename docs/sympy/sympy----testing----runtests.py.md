# `D:\src\scipysrc\sympy\sympy\testing\runtests.py`

```
"""
这是我们的测试框架。

目标：

* 兼容 py.test 并且操作方式非常相似（或完全相同）
* 不需要任何外部依赖
* 最好所有功能都在这个文件中
* 没有魔法，只需导入测试文件并执行测试函数，就是这样
* 可移植
"""

import os  # 导入操作系统相关的功能
import sys  # 导入系统相关的功能
import platform  # 导入平台相关的功能
import inspect  # 导入检查模块
import traceback  # 导入追踪错误信息的功能
import pdb  # 导入 Python 调试器
import re  # 导入正则表达式模块
import linecache  # 导入缓存行的模块
import time  # 导入时间相关的功能
from fnmatch import fnmatch  # 导入通配符匹配功能
from timeit import default_timer as clock  # 导入计时器
import doctest as pdoctest  # 避免与我们的 doctest() 函数冲突
from doctest import DocTestFinder, DocTestRunner  # 导入 doctest 测试相关的功能
import random  # 导入随机数生成功能
import subprocess  # 导入子进程管理功能
import shutil  # 导入文件操作功能
import signal  # 导入信号处理功能
import stat  # 导入文件状态相关功能
import tempfile  # 导入临时文件和目录的功能
import warnings  # 导入警告管理功能
from contextlib import contextmanager  # 导入上下文管理器
from inspect import unwrap  # 导入取消装饰器的功能

from sympy.core.cache import clear_cache  # 导入清除缓存的功能
from sympy.external import import_module  # 导入模块导入功能
from sympy.external.gmpy import GROUND_TYPES  # 导入 gmpy 相关类型

IS_WINDOWS = (os.name == 'nt')  # 检查操作系统是否为 Windows
ON_CI = os.getenv('CI', None)  # 检查是否在 CI 环境下运行

# 实验性生成的测试时间分配比例列表
# 这些值应周期性地更新
# 可以通过以下代码生成这些值：
# from time import time
# import sympy
# import os
# os.environ["CI"] = 'true' # 模拟 CI 环境以获取更准确的分布
# delays, num_splits = [], 30
# for i in range(1, num_splits + 1):
#     tic = time()
#     sympy.test(split='{}/{}'.format(i, num_splits), time_balance=False) # 添加 slow=True 用于慢速测试
#     delays.append(time() - tic)
# tot = sum(delays)
# print([round(x / tot, 4) for x in delays])
SPLIT_DENSITY = [
    0.0059, 0.0027, 0.0068, 0.0011, 0.0006,
    0.0058, 0.0047, 0.0046, 0.004, 0.0257,
    0.0017, 0.0026, 0.004, 0.0032, 0.0016,
    0.0015, 0.0004, 0.0011, 0.0016, 0.0014,
    0.0077, 0.0137, 0.0217, 0.0074, 0.0043,
    0.0067, 0.0236, 0.0004, 0.1189, 0.0142,
    0.0234, 0.0003, 0.0003, 0.0047, 0.0006,
    0.0013, 0.0004, 0.0008, 0.0007, 0.0006,
    0.0139, 0.0013, 0.0007, 0.0051, 0.002,
    0.0004, 0.0005, 0.0213, 0.0048, 0.0016,
    0.0012, 0.0014, 0.0024, 0.0015, 0.0004,
    0.0005, 0.0007, 0.011, 0.0062, 0.0015,
    0.0021, 0.0049, 0.0006, 0.0006, 0.0011,
    0.0006, 0.0019, 0.003, 0.0044, 0.0054,
    0.0057, 0.0049, 0.0016, 0.0006, 0.0009,
    0.0006, 0.0012, 0.0006, 0.0149, 0.0532,
    0.0076, 0.0041, 0.0024, 0.0135, 0.0081,
    0.2209, 0.0459, 0.0438, 0.0488, 0.0137,
    0.002, 0.0003, 0.0008, 0.0039, 0.0024,
    0.0005, 0.0004, 0.003, 0.056, 0.0026
]
# 定义一个包含多个浮点数的列表，表示分割密度（慢速版本）
SPLIT_DENSITY_SLOW = [0.0086, 0.0004, 0.0568, 0.0003, 0.0032, 0.0005, 0.0004, 0.0013, 0.0016, 0.0648,
                      0.0198, 0.1285, 0.098, 0.0005, 0.0064, 0.0003, 0.0004, 0.0026, 0.0007, 0.0051,
                      0.0089, 0.0024, 0.0033, 0.0057, 0.0005, 0.0003, 0.001, 0.0045, 0.0091, 0.0006,
                      0.0005, 0.0321, 0.0059, 0.1105, 0.216, 0.1489, 0.0004, 0.0003, 0.0006, 0.0483]

# 定义一个自定义异常类，用于表示跳过异常
class Skipped(Exception):
    pass

# 定义一个自定义异常类，用于表示超时异常
class TimeOutError(Exception):
    pass

# 定义一个自定义异常类，用于表示依赖错误异常
class DependencyError(Exception):
    pass


def _indent(s, indent=4):
    """
    给每一行（非空白行）添加指定数量的空格字符，返回结果字符串。
    如果字符串 ``s`` 是 Unicode 类型，则使用标准输出编码和 ``backslashreplace`` 错误处理器。
    """
    # 正则表达式匹配非空白行的开头：
    return re.sub('(?m)^(?!$)', indent*' ', s)


pdoctest._indent = _indent  # type: ignore

# 重写报告函数以维护 Windows 和 Python3 的兼容性


def _report_failure(self, out, test, example, got):
    """
    报告给定示例失败的情况。
    """
    s = self._checker.output_difference(example, got, self.optionflags)
    s = s.encode('raw_unicode_escape').decode('utf8', 'ignore')
    out(self._failure_header(test, example) + s)


if IS_WINDOWS:
    DocTestRunner.report_failure = _report_failure  # type: ignore


def convert_to_native_paths(lst):
    """
    将一个包含 '/' 分隔路径的列表转换为本地（os.sep 分隔）路径列表，并在系统不区分大小写时转换为小写。
    """
    newlst = []
    for i, rv in enumerate(lst):
        rv = os.path.join(*rv.split("/"))
        # 在 Windows 上，冒号后的斜杠会被去除
        if sys.platform == "win32":
            pos = rv.find(':')
            if pos != -1:
                if rv[pos + 1] != '\\':
                    rv = rv[:pos + 1] + '\\' + rv[pos + 1:]
        newlst.append(os.path.normcase(rv))
    return newlst


def get_sympy_dir():
    """
    返回 SymPy 根目录，并设置全局值指示系统是否区分大小写。
    """
    this_file = os.path.abspath(__file__)
    sympy_dir = os.path.join(os.path.dirname(this_file), "..", "..")
    sympy_dir = os.path.normpath(sympy_dir)
    return os.path.normcase(sympy_dir)


def setup_pprint(disable_line_wrap=True):
    from sympy.interactive.printing import init_printing
    from sympy.printing.pretty.pretty import pprint_use_unicode
    import sympy.interactive.printing as interactive_printing
    from sympy.printing.pretty import stringpict

    # 防止 doctest 中的 init_printing() 影响其他 doctest
    interactive_printing.NO_GLOBAL = True

    # 强制 pprint 在 doctest 中使用 ASCII 模式
    use_unicode_prev = pprint_use_unicode(False)

    # 禁用 pprint() 输出的行包裹
    wrap_line_prev = stringpict._GLOBAL_WRAP_LINE
    if disable_line_wrap:
        stringpict._GLOBAL_WRAP_LINE = False
    # 初始化打印设置，使用稳定的哈希值打印器并禁用漂亮打印选项
    init_printing(pretty_print=False)

    # 返回使用 Unicode 预设值和换行预设值
    return use_unicode_prev, wrap_line_prev
@contextmanager
def raise_on_deprecated():
    """Context manager to make DeprecationWarning raise an error

    This is to catch SymPyDeprecationWarning from library code while running
    tests and doctests. It is important to use this context manager around
    each individual test/doctest in case some tests modify the warning
    filters.
    """
    # 在警告处理上下文中，捕获 SymPyDeprecationWarning 并将其转换为错误
    with warnings.catch_warnings():
        # 设置警告过滤器，使得 DeprecationWarning 类型的警告被转换为错误
        warnings.filterwarnings('error', '.*', DeprecationWarning, module='sympy.*')
        # 返回上下文管理器的控制权，以便执行包含在上下文中的代码
        yield


def run_in_subprocess_with_hash_randomization(
        function, function_args=(),
        function_kwargs=None, command=sys.executable,
        module='sympy.testing.runtests', force=False):
    """
    Run a function in a Python subprocess with hash randomization enabled.

    If hash randomization is not supported by the version of Python given, it
    returns False.  Otherwise, it returns the exit value of the command.  The
    function is passed to sys.exit(), so the return value of the function will
    be the return value.

    The environment variable PYTHONHASHSEED is used to seed Python's hash
    randomization.  If it is set, this function will return False, because
    starting a new subprocess is unnecessary in that case.  If it is not set,
    one is set at random, and the tests are run.  Note that if this
    environment variable is set when Python starts, hash randomization is
    automatically enabled.  To force a subprocess to be created even if
    PYTHONHASHSEED is set, pass ``force=True``.  This flag will not force a
    subprocess in Python versions that do not support hash randomization (see
    below), because those versions of Python do not support the ``-R`` flag.

    ``function`` should be a string name of a function that is importable from
    the module ``module``, like "_test".  The default for ``module`` is
    "sympy.testing.runtests".  ``function_args`` and ``function_kwargs``
    should be a repr-able tuple and dict, respectively.  The default Python
    command is sys.executable, which is the currently running Python command.

    This function is necessary because the seed for hash randomization must be
    set by the environment variable before Python starts.  Hence, in order to
    use a predetermined seed for tests, we must start Python in a separate
    subprocess.

    Hash randomization was added in the minor Python versions 2.6.8, 2.7.3,
    3.1.5, and 3.2.3, and is enabled by default in all Python versions after
    and including 3.3.0.

    Examples
    ========

    >>> from sympy.testing.runtests import (
    ... run_in_subprocess_with_hash_randomization)
    >>> # run the core tests in verbose mode
    >>> run_in_subprocess_with_hash_randomization("_test",
    ... function_args=("core",),
    ... function_kwargs={'verbose': True}) # doctest: +SKIP
    # Will return 0 if sys.executable supports hash randomization and tests
    # pass, 1 if they fail, and False if it does not support hash
    # randomization.

    """
    # 获取当前工作目录路径
    cwd = get_sympy_dir()
    # 注意，必须在每处返回 False，而不是 None，因为 subprocess.call 有时会返回 None。

    # 首先检查 Python 版本是否支持哈希随机化
    # 如果不支持，将无法识别 -R 标志
    p = subprocess.Popen([command, "-RV"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, cwd=cwd)
    p.communicate()
    # 如果返回码不为 0，则表示命令执行失败，返回 False
    if p.returncode != 0:
        return False

    # 获取环境变量中的 PYTHONHASHSEED
    hash_seed = os.getenv("PYTHONHASHSEED")
    # 如果没有设置 PYTHONHASHSEED，则随机生成一个 32 位整数作为种子
    if not hash_seed:
        os.environ["PYTHONHASHSEED"] = str(random.randrange(2**32))
    else:
        # 如果不是强制重新设置哈希种子，并且已经有设置了的情况下，返回 False
        if not force:
            return False

    # 如果未提供 function_kwargs，则将其设为一个空字典
    function_kwargs = function_kwargs or {}

    # 构建要执行的命令字符串
    commandstring = ("import sys; from %s import %s;sys.exit(%s(*%s, **%s))" %
                     (module, function, function, repr(function_args),
                      repr(function_kwargs)))

    try:
        # 使用 subprocess.Popen 执行命令
        p = subprocess.Popen([command, "-R", "-c", commandstring], cwd=cwd)
        p.communicate()
    except KeyboardInterrupt:
        # 如果捕获到键盘中断异常，等待子进程结束
        p.wait()
    finally:
        # 恢复环境变量设置，确保当前 Python 进程正常读取
        if hash_seed is None:
            # 如果原先没有设置 PYTHONHASHSEED，删除环境变量中的设置
            del os.environ["PYTHONHASHSEED"]
        else:
            # 否则，将环境变量设置为之前保存的值
            os.environ["PYTHONHASHSEED"] = hash_seed
        # 返回子进程的返回码
        return p.returncode
def run_all_tests(test_args=(), test_kwargs=None,
                  doctest_args=(), doctest_kwargs=None,
                  examples_args=(), examples_kwargs=None):
    """
    运行所有的测试。

    目前，这会运行常规测试（bin/test）、doctests（bin/doctest）以及示例（examples/all.py）。

    这是 ``setup.py test`` 使用的方式。

    您可以向支持的测试函数传递参数和关键字参数（目前支持 test、doctest 和示例）。请参阅这些函数的文档字符串以了解可用选项的描述。

    例如，要关闭求解器测试的颜色，可以这样做：

    >>> from sympy.testing.runtests import run_all_tests
    >>> run_all_tests(test_args=("solvers",),
    ... test_kwargs={"colors:False"}) # doctest: +SKIP

    """
    tests_successful = True  # 初始化测试成功标志

    test_kwargs = test_kwargs or {}  # 如果 test_kwargs 为 None，则初始化为空字典
    doctest_kwargs = doctest_kwargs or {}  # 如果 doctest_kwargs 为 None，则初始化为空字典
    examples_kwargs = examples_kwargs or {'quiet': True}  # 如果 examples_kwargs 为 None，则初始化为 {'quiet': True}

    try:
        # Regular tests
        if not test(*test_args, **test_kwargs):
            # 如果常规测试中有任何测试失败，则设置 tests_successful 为 False 并继续运行 doctests
            tests_successful = False

        # Doctests
        print()
        if not doctest(*doctest_args, **doctest_kwargs):
            # 如果 doctests 中有任何测试失败，则设置 tests_successful 为 False
            tests_successful = False

        # Examples
        print()
        sys.path.append("examples")   # 将 "examples" 路径添加到 sys.path 中，以便引入 examples/all.py
        from all import run_examples  # 导入 examples/all.py 中的 run_examples 函数（类型忽略）

        if not run_examples(*examples_args, **examples_kwargs):
            # 如果示例中有任何测试失败，则设置 tests_successful 为 False
            tests_successful = False

        if tests_successful:
            return  # 如果所有测试都成功，则直接返回
        else:
            # 如果有测试失败，则以非零退出码退出程序
            sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("DO *NOT* COMMIT!")  # 在键盘中断情况下输出警告信息
        sys.exit(1)  # 以非零退出码退出程序


def test(*paths, subprocess=True, rerun=0, **kwargs):
    """
    在指定的 test_*.py 文件中运行测试。

    如果 ``paths`` 中的任何字符串与测试文件的路径的一部分匹配，则运行该特定的 test_*.py 文件中的测试。如果 ``paths=[]``，则运行所有 test_*.py 文件中的测试。

    注意：

    - 如果 sort=False，则测试将以随机顺序运行（而不是默认顺序）。
    - 路径可以使用本地系统格式或 Unix 前斜杠格式输入。
    - 可以通过提供其路径来测试黑名单中的文件；只有在没有给出路径时才排除这些文件。

    **测试结果的说明**

    ======  ===============================================================
    输出    含义
    ======  ===============================================================
    .       通过
    F       失败
    X       XPassed（预期失败但通过）
    f       XFAILed（预期失败且确实失败）
    s       跳过
    w       较慢
    T       超时（例如使用 ``--timeout`` 时）
    ```
    K       KeyboardInterrupt (when running the slow tests with ``--slow``,
            you can interrupt one of them without killing the test runner)
    ======  ===============================================================


    Colors have no additional meaning and are used just to facilitate
    interpreting the output.

    Examples
    ========

    >>> import sympy

    Run all tests:

    >>> sympy.test()    # doctest: +SKIP

    Run one file:

    >>> sympy.test("sympy/core/tests/test_basic.py")    # doctest: +SKIP
    >>> sympy.test("_basic")    # doctest: +SKIP

    Run all tests in sympy/functions/ and some particular file:

    >>> sympy.test("sympy/core/tests/test_basic.py",
    ...        "sympy/functions")    # doctest: +SKIP

    Run all tests in sympy/core and sympy/utilities:

    >>> sympy.test("/core", "/util")    # doctest: +SKIP

    Run specific test from a file:

    >>> sympy.test("sympy/core/tests/test_basic.py",
    ...        kw="test_equality")    # doctest: +SKIP

    Run specific test from any file:

    >>> sympy.test(kw="subs")    # doctest: +SKIP

    Run the tests with verbose mode on:

    >>> sympy.test(verbose=True)    # doctest: +SKIP

    Do not sort the test output:

    >>> sympy.test(sort=False)    # doctest: +SKIP

    Turn on post-mortem pdb:

    >>> sympy.test(pdb=True)    # doctest: +SKIP

    Turn off colors:

    >>> sympy.test(colors=False)    # doctest: +SKIP

    Force colors, even when the output is not to a terminal (this is useful,
    e.g., if you are piping to ``less -r`` and you still want colors)

    >>> sympy.test(force_colors=False)    # doctest: +SKIP

    The traceback verboseness can be set to "short" or "no" (default is
    "short")

    >>> sympy.test(tb='no')    # doctest: +SKIP

    The ``split`` option can be passed to split the test run into parts. The
    split currently only splits the test files, though this may change in the
    future. ``split`` should be a string of the form 'a/b', which will run
    part ``a`` of ``b``. For instance, to run the first half of the test suite:

    >>> sympy.test(split='1/2')  # doctest: +SKIP

    The ``time_balance`` option can be passed in conjunction with ``split``.
    If ``time_balance=True`` (the default for ``sympy.test``), SymPy will attempt
    to split the tests such that each split takes equal time.  This heuristic
    for balancing is based on pre-recorded test data.

    >>> sympy.test(split='1/2', time_balance=True)  # doctest: +SKIP

    You can disable running the tests in a separate subprocess using
    ``subprocess=False``.  This is done to support seeding hash randomization,
    which is enabled by default in the Python versions where it is supported.
    If subprocess=False, hash randomization is enabled/disabled according to
    whether it has been enabled or not in the calling Python process.
    However, even if it is enabled, the seed cannot be printed unless it is
    called from a new Python process.



    K       KeyboardInterrupt (when running the slow tests with ``--slow``,
            you can interrupt one of them without killing the test runner)
    ======  ===============================================================


    Colors have no additional meaning and are used just to facilitate
    interpreting the output.

    Examples
    ========

    >>> import sympy

    Run all tests:

    >>> sympy.test()    # doctest: +SKIP

    Run one file:

    >>> sympy.test("sympy/core/tests/test_basic.py")    # doctest: +SKIP
    >>> sympy.test("_basic")    # doctest: +SKIP

    Run all tests in sympy/functions/ and some particular file:

    >>> sympy.test("sympy/core/tests/test_basic.py",
    ...        "sympy/functions")    # doctest: +SKIP

    Run all tests in sympy/core and sympy/utilities:

    >>> sympy.test("/core", "/util")    # doctest: +SKIP

    Run specific test from a file:

    >>> sympy.test("sympy/core/tests/test_basic.py",
    ...        kw="test_equality")    # doctest: +SKIP

    Run specific test from any file:

    >>> sympy.test(kw="subs")    # doctest: +SKIP

    Run the tests with verbose mode on:

    >>> sympy.test(verbose=True)    # doctest: +SKIP

    Do not sort the test output:

    >>> sympy.test(sort=False)    # doctest: +SKIP

    Turn on post-mortem pdb:

    >>> sympy.test(pdb=True)    # doctest: +SKIP

    Turn off colors:

    >>> sympy.test(colors=False)    # doctest: +SKIP

    Force colors, even when the output is not to a terminal (this is useful,
    e.g., if you are piping to ``less -r`` and you still want colors)

    >>> sympy.test(force_colors=False)    # doctest: +SKIP

    The traceback verboseness can be set to "short" or "no" (default is
    "short")

    >>> sympy.test(tb='no')    # doctest: +SKIP

    The ``split`` option can be passed to split the test run into parts. The
    split currently only splits the test files, though this may change in the
    future. ``split`` should be a string of the form 'a/b', which will run
    part ``a`` of ``b``. For instance, to run the first half of the test suite:

    >>> sympy.test(split='1/2')  # doctest: +SKIP

    The ``time_balance`` option can be passed in conjunction with ``split``.
    If ``time_balance=True`` (the default for ``sympy.test``), SymPy will attempt
    to split the tests such that each split takes equal time.  This heuristic
    for balancing is based on pre-recorded test data.

    >>> sympy.test(split='1/2', time_balance=True)  # doctest: +SKIP

    You can disable running the tests in a separate subprocess using
    ``subprocess=False``.  This is done to support seeding hash randomization,
    which is enabled by default in the Python versions where it is supported.
    If subprocess=False, hash randomization is enabled/disabled according to
    whether it has been enabled or not in the calling Python process.
    However, even if it is enabled, the seed cannot be printed unless it is
    called from a new Python process.
    """
    # 从 0 开始计数，不打印 0
    print_counter = lambda i : (print("rerun %d" % (rerun-i))
                                if rerun-i else None)

    # 如果使用子进程
    if subprocess:
        # 倒序循环，直到 i 为 0
        for i in range(rerun, -1, -1):
            # 打印计数器
            print_counter(i)
            # 在具有哈希随机化的子进程中运行测试函数
            ret = run_in_subprocess_with_hash_randomization("_test",
                        function_args=paths, function_kwargs=kwargs)
            # 如果返回 False，则中断循环
            if ret is False:
                break
            # 根据返回值判断是否继续循环
            val = not bool(ret)
            # 在第一次失败或者完成时退出循环
            if not val or i == 0:
                return val

    # 即使哈希随机化不受支持也重新运行
    for i in range(rerun, -1, -1):
        # 打印计数器
        print_counter(i)
        # 调用 _test 函数进行测试
        val = not bool(_test(*paths, **kwargs))
        # 在第一次失败或者完成时退出循环
        if not val or i == 0:
            return val
    """
def _test(*paths,
        verbose=False, tb="short", kw=None, pdb=False, colors=True,
        force_colors=False, sort=True, seed=None, timeout=False,
        fail_on_timeout=False, slow=False, enhance_asserts=False, split=None,
        time_balance=True, blacklist=(),
        fast_threshold=None, slow_threshold=None):
    """
    Internal function that actually runs the tests.

    All keyword arguments from ``test()`` are passed to this function except for
    ``subprocess``.

    Returns 0 if tests passed and 1 if they failed.  See the docstring of
    ``test()`` for more information.
    """
    # Initialize kw to an empty tuple if None
    kw = kw or ()
    # Ensure kw is converted to a tuple if it's currently a string
    if isinstance(kw, str):
        kw = (kw,)
    
    # Determine if post mortem debugging should be enabled based on pdb flag
    post_mortem = pdb
    
    # Generate a random seed if not provided
    if seed is None:
        seed = random.randrange(100000000)
    
    # Adjust timeout settings if running on a CI environment
    if ON_CI and timeout is False:
        timeout = 595
        fail_on_timeout = True
    
    # Modify the blacklist to include specific paths if running on CI
    if ON_CI:
        blacklist = list(blacklist) + ['sympy/plotting/pygletplot/tests']
    
    # Convert paths in blacklist to native system format
    blacklist = convert_to_native_paths(blacklist)
    
    # Initialize a PyTestReporter instance with specified parameters
    r = PyTestReporter(verbose=verbose, tb=tb, colors=colors,
        force_colors=force_colors, split=split)
    
    # Comment explaining the purpose of the loop
    _paths = []
    for path in paths:
        # Split the path and keyword if '::' is present and add to kw
        if '::' in path:
            path, _kw = path.split('::', 1)
            kw += (_kw,)
        _paths.append(path)
    paths = _paths

    # Initialize SymPyTests instance with necessary parameters
    t = SymPyTests(r, kw, post_mortem, seed,
                   fast_threshold=fast_threshold,
                   slow_threshold=slow_threshold)

    # Retrieve the list of test files for SymPy
    test_files = t.get_test_files('sympy')

    # Filter out blacklisted files from test_files
    not_blacklisted = [f for f in test_files
                       if not any(b in f for b in blacklist)]

    # Determine which test files to execute based on provided paths
    if len(paths) == 0:
        matched = not_blacklisted
    else:
        # Convert paths to native system format
        paths = convert_to_native_paths(paths)
        matched = []
        for f in not_blacklisted:
            basename = os.path.basename(f)
            for p in paths:
                # Check if path or basename matches the filter pattern
                if p in f or fnmatch(basename, p):
                    matched.append(f)
                    break

    # Adjust test file distribution based on time_balance and split options
    density = None
    if time_balance:
        if slow:
            density = SPLIT_DENSITY_SLOW
        else:
            density = SPLIT_DENSITY

    # Split matched test files into smaller lists if split option is specified
    if split:
        matched = split_list(matched, split, density=density)

    # Extend the list of test files to be executed by SymPyTests instance
    t._testfiles.extend(matched)

    # Return 0 if all tests pass, otherwise return 1
    return int(not t.test(sort=sort, timeout=timeout, slow=slow,
        enhance_asserts=enhance_asserts, fail_on_timeout=fail_on_timeout))
    # 定义一个匿名函数 print_counter，用于打印计数器 i 的值，除非 i 为 0
    print_counter = lambda i : (print("rerun %d" % (rerun-i))
                                if rerun-i else None)
    
    # 如果 subprocess 参数为真，则执行以下代码块
    if subprocess:
        # 反向循环，从 rerun 到 0
        for i in range(rerun, -1, -1):
            # 调用 print_counter 函数打印当前的计数器值
            print_counter(i)
            # 在子进程中使用带有哈希随机化的函数运行 "_doctest"，传入 paths 和 kwargs 作为参数和关键字参数
            ret = run_in_subprocess_with_hash_randomization("_doctest",
                        function_args=paths, function_kwargs=kwargs)
            # 如果返回 False，则跳出循环
            if ret is False:
                break
            # 将 ret 转换为布尔值，并取反得到 val
            val = not bool(ret)
            # 在第一个失败或者完成时退出循环并返回 val
            if not val or i == 0:
                return val
    
    # 无论哈希随机化是否支持，都重新运行以下代码块
    for i in range(rerun, -1, -1):
        # 调用 print_counter 函数打印当前的计数器值
        print_counter(i)
        # 调用 _doctest 函数，传入 paths 和 kwargs 作为参数和关键字参数，并将返回值转换为布尔值取反得到 val
        val = not bool(_doctest(*paths, **kwargs))
        # 在第一个失败或者完成时退出循环并返回 val
        if not val or i == 0:
            return val
# 获取用于 doctest 的默认黑名单列表
def _get_doctest_blacklist():
    # 初始化空的黑名单列表
    blacklist = []

    # 手动添加特定文件路径到黑名单中，这些文件会导致特定问题或警告
    blacklist.extend([
        "doc/src/modules/plotting.rst",  # 生成实时图表
        "doc/src/modules/physics/mechanics/autolev_parser.rst",  # 自动生成的文件
        "sympy/codegen/array_utils.py",  # 引发弃用警告
        "sympy/core/compatibility.py",  # 后向兼容性垫片，导入它会触发弃用警告
        "sympy/core/trace.py",  # 后向兼容性垫片，导入它会触发弃用警告
        "sympy/galgebra.py",  # 不再是 SymPy 的一部分
        "sympy/parsing/autolev/_antlr/autolevlexer.py",  # 自动生成的代码
        "sympy/parsing/autolev/_antlr/autolevlistener.py",  # 自动生成的代码
        "sympy/parsing/autolev/_antlr/autolevparser.py",  # 自动生成的代码
        "sympy/parsing/latex/_antlr/latexlexer.py",  # 自动生成的代码
        "sympy/parsing/latex/_antlr/latexparser.py",  # 自动生成的代码
        "sympy/plotting/pygletplot/__init__.py",  # 在某些系统上崩溃
        "sympy/plotting/pygletplot/plot.py",  # 在某些系统上崩溃
        "sympy/printing/ccode.py",  # 后向兼容性垫片，导入它会破坏代码生成的 doctest
        "sympy/printing/cxxcode.py",  # 后向兼容性垫片，导入它会破坏代码生成的 doctest
        "sympy/printing/fcode.py",  # 后向兼容性垫片，导入它会破坏代码生成的 doctest
        "sympy/testing/randtest.py",  # 后向兼容性垫片，导入它会触发弃用警告
        "sympy/this.py",  # 输出文本
    ])
    
    # 自动添加一组 autolev 解析器测试文件到黑名单中
    num = 12
    for i in range(1, num + 1):
        blacklist.append("sympy/parsing/autolev/test-examples/ruletest" + str(i) + ".py")
    
    # 手动添加特定 autolev 示例测试文件到黑名单中
    blacklist.extend([
        "sympy/parsing/autolev/test-examples/pydy-example-repo/mass_spring_damper.py",
        "sympy/parsing/autolev/test-examples/pydy-example-repo/chaos_pendulum.py",
        "sympy/parsing/autolev/test-examples/pydy-example-repo/double_pendulum.py",
        "sympy/parsing/autolev/test-examples/pydy-example-repo/non_min_pendulum.py"
    ])

    # 如果未成功导入 numpy 模块，将进一步添加到黑名单中
    if import_module('numpy') is None:
        blacklist.extend([
            "sympy/plotting/experimental_lambdify.py",
            "sympy/plotting/plot_implicit.py",
            "examples/advanced/autowrap_integrators.py",
            "examples/advanced/autowrap_ufuncify.py",
            "examples/intermediate/sample.py",
            "examples/intermediate/mplot2d.py",
            "examples/intermediate/mplot3d.py",
            "doc/src/modules/numeric-computation.rst",
            "doc/src/explanation/best-practices.md",
            "doc/src/tutorials/physics/biomechanics/biomechanical-model-example.rst",
            "doc/src/tutorials/physics/biomechanics/biomechanics.rst",
        ])
    else:
        # 如果没有导入 matplotlib 模块，则将以下文件添加到黑名单
        if import_module('matplotlib') is None:
            blacklist.extend([
                "examples/intermediate/mplot2d.py",
                "examples/intermediate/mplot3d.py"
            ])
        else:
            # 如果导入了 matplotlib 模块，则设置非窗口化的后端，以便在 CI 环境中运行测试
            import matplotlib
            matplotlib.use('Agg')

    # 如果在 CI 环境中或未导入 pyglet 模块，则将以下文件添加到黑名单
    if ON_CI or import_module('pyglet') is None:
        blacklist.extend(["sympy/plotting/pygletplot"])

    # 如果未导入 aesara 模块，则将以下文件添加到黑名单
    if import_module('aesara') is None:
        blacklist.extend([
            "sympy/printing/aesaracode.py",
            "doc/src/modules/numeric-computation.rst",
        ])

    # 如果未导入 cupy 模块，则将以下文件添加到黑名单
    if import_module('cupy') is None:
        blacklist.extend([
            "doc/src/modules/numeric-computation.rst",
        ])

    # 如果未导入 jax 模块，则将以下文件添加到黑名单
    if import_module('jax') is None:
        blacklist.extend([
            "doc/src/modules/numeric-computation.rst",
        ])

    # 如果未导入 antlr4 模块，则将以下文件添加到黑名单
    if import_module('antlr4') is None:
        blacklist.extend([
            "sympy/parsing/autolev/__init__.py",
            "sympy/parsing/latex/_parse_latex_antlr.py",
        ])

    # 如果未导入 lfortran 模块，则将以下文件添加到黑名单
    if import_module('lfortran') is None:
        # 当 lfortran 未安装时会抛出 ImportError
        blacklist.extend([
            "sympy/parsing/sym_expr.py",
        ])

    # 如果未导入 scipy 模块，则将以下文件添加到黑名单
    if import_module("scipy") is None:
        # 当 scipy 未安装时会抛出 ModuleNotFoundError
        blacklist.extend([
            "doc/src/guides/solving/solve-numerically.md",
            "doc/src/guides/solving/solve-ode.md",
        ])

    # 如果未导入 numpy 模块，则将以下文件添加到黑名单
    if import_module("numpy") is None:
        # 当 numpy 未安装时会抛出 ModuleNotFoundError
        blacklist.extend([
                "doc/src/guides/solving/solve-ode.md",
                "doc/src/guides/solving/solve-numerically.md",
        ])

    # 由于在 asmeurer 的机器人中出现 doctest 失败，以下文件被禁用
    blacklist.extend([
        "sympy/utilities/autowrap.py",
        "examples/advanced/autowrap_integrators.py",
        "examples/advanced/autowrap_ufuncify.py"
        ])

    # 这些是已弃用的存根文件，将会被移除
    blacklist.extend([
        "sympy/utilities/tmpfiles.py",
        "sympy/utilities/pytest.py",
        "sympy/utilities/runtests.py",
        "sympy/utilities/quality_unicode.py",
        "sympy/utilities/randtest.py",
    ])

    # 将黑名单转换为本地路径格式
    blacklist = convert_to_native_paths(blacklist)
    return blacklist
def _doctest(*paths, **kwargs):
    """
    Internal function that actually runs the doctests.

    All keyword arguments from ``doctest()`` are passed to this function
    except for ``subprocess``.

    Returns 0 if tests passed and 1 if they failed.  See the docstrings of
    ``doctest()`` and ``test()`` for more information.
    """
    # 导入需要的模块和函数
    from sympy.printing.pretty.pretty import pprint_use_unicode
    from sympy.printing.pretty import stringpict

    # 从参数中获取选项设置，默认为 False 或空列表
    normal = kwargs.get("normal", False)
    verbose = kwargs.get("verbose", False)
    colors = kwargs.get("colors", True)
    force_colors = kwargs.get("force_colors", False)
    blacklist = kwargs.get("blacklist", [])
    split  = kwargs.get('split', None)

    # 将默认的黑名单项扩展到用户指定的黑名单中
    blacklist.extend(_get_doctest_blacklist())

    # 在 CI 环境下，使用非窗口化的后端以确保测试可以正常运行
    if import_module('matplotlib') is not None:
        import matplotlib
        matplotlib.use('Agg')

    # 禁止对外部模块的警告信息显示
    import sympy.external
    sympy.external.importtools.WARN_OLD_VERSION = False
    sympy.external.importtools.WARN_NOT_INSTALLED = False

    # 禁止显示绘图窗口
    from sympy.plotting.plot import unset_show
    unset_show()

    # 创建测试报告对象和 SymPy 的文档测试对象
    r = PyTestReporter(verbose, split=split, colors=colors,\
                       force_colors=force_colors)
    t = SymPyDocTests(r, normal)

    # 获取符合测试条件的测试文件列表
    test_files = t.get_test_files('sympy')
    test_files.extend(t.get_test_files('examples', init_only=False))

    # 过滤掉黑名单中的文件，得到不在黑名单中的文件列表
    not_blacklisted = [f for f in test_files
                       if not any(b in f for b in blacklist)]
    
    # 如果未指定具体文件路径，则使用所有符合条件的文件
    if len(paths) == 0:
        matched = not_blacklisted
    else:
        # 根据用户指定的路径进行匹配，同时不包括黑名单中的项
        paths = convert_to_native_paths(paths)
        matched = []
        for f in not_blacklisted:
            basename = os.path.basename(f)
            for p in paths:
                if p in f or fnmatch(basename, p):
                    matched.append(f)
                    break

    # 按文件名排序匹配到的文件列表
    matched.sort()

    # 如果指定了 split 参数，则将匹配到的文件列表按照 split 数目分组
    if split:
        matched = split_list(matched, split)

    # 将匹配到的文件列表添加到 SymPyDocTests 对象的测试文件列表中
    t._testfiles.extend(matched)

    # 运行测试，并记录测试结果，failed 表示是否有测试失败
    if t._testfiles:
        failed = not t.test()
    else:
        failed = False

    # N.B.
    # --------------------------------------------------------------------
    # Here we test *.rst and *.md files at or below doc/src. Code from these
    # must be self supporting in terms of imports since there is no importing
    # of necessary modules by doctest.testfile. If you try to pass *.py files
    # through this they might fail because they will lack the needed imports
    # and smarter parsing that can be done with source code.
    #
    # 获取 doc/src 目录及其子目录下所有符合 *.rst 扩展名的文件作为测试文件列表
    test_files_rst = t.get_test_files('doc/src', '*.rst', init_only=False)
    # 获取 doc/src 目录及其子目录下所有符合 *.md 扩展名的文件作为测试文件列表
    test_files_md = t.get_test_files('doc/src', '*.md', init_only=False)
    # 将测试文件列表合并并排序
    test_files = test_files_rst + test_files_md
    test_files.sort()

    # 根据黑名单过滤出不在黑名单中的文件列表
    not_blacklisted = [f for f in test_files
                       if not any(b in f for b in blacklist)]

    # 如果 paths 列表为空，则匹配所有不在黑名单中的文件
    if len(paths) == 0:
        matched = not_blacklisted
    else:
        # 根据 paths 列表匹配文件，避免重复匹配 *py 测试文件
        matched = []
        for f in not_blacklisted:
            basename = os.path.basename(f)
            for p in paths:
                if p in f or fnmatch(basename, p):
                    matched.append(f)
                    break

    # 如果 split 标志为真，则根据指定规则拆分匹配列表
    if split:
        matched = split_list(matched, split)

    # 初始化第一次报告标志为真
    first_report = True

    # 遍历匹配的 rst 文件进行测试
    for rst_file in matched:
        # 如果 rst_file 不是文件，则跳过
        if not os.path.isfile(rst_file):
            continue
        
        # 保存旧的 displayhook 设置
        old_displayhook = sys.displayhook
        try:
            # 设置 pprint 使用的 Unicode 和换行方式
            use_unicode_prev, wrap_line_prev = setup_pprint()
            
            # 运行 sympytestfile 测试函数
            out = sympytestfile(
                rst_file, module_relative=False, encoding='utf-8',
                optionflags=pdoctest.ELLIPSIS | pdoctest.NORMALIZE_WHITESPACE |
                pdoctest.IGNORE_EXCEPTION_DETAIL)
        finally:
            # 恢复原始的 displayhook 设置
            sys.displayhook = old_displayhook
            
            # 重置 sympy.interactive.printing 中的全局标志
            import sympy.interactive.printing as interactive_printing
            interactive_printing.NO_GLOBAL = False
            
            # 恢复 pprint 使用的 Unicode 设置
            pprint_use_unicode(use_unicode_prev)
            
            # 恢复 stringpict 模块的全局换行设置
            stringpict._GLOBAL_WRAP_LINE = wrap_line_prev

        # 获取测试结果和是否测试过的标志
        rstfailed, tested = out
        
        # 如果有测试过，则更新失败标志
        if tested:
            failed = rstfailed or failed
            
            # 如果是第一次报告，则打印起始信息
            if first_report:
                first_report = False
                msg = 'rst/md doctests start'
                if not t._testfiles:
                    r.start(msg=msg)
                else:
                    r.write_center(msg)
                    print()
            
            # 提取文件名后的标识作为 ID
            file_id = rst_file[rst_file.find('sympy') + len('sympy') + 1:]
            print(file_id, end=" ")
            
            # 更新报告宽度
            wid = r.terminal_width - len(file_id) - 1
            
            # 准备测试文件和报告字符串
            test_file = '[%s]' % (tested)
            report = '[%s]' % (rstfailed or 'OK')
            
            # 打印测试结果的报告
            print(''.join(
                [test_file, ' '*(wid - len(test_file) - len(report)), report])
            )

    # 如果 *py 的 doctest 已经输出了失败信息，这里不再输出消息
    # 只有在 *rst 测试报告第一次输出后，才会显示此消息
    # 如果不是第一次报告且存在失败的情况，执行以下操作
    if not first_report and failed:
        # 打印空行
        print()
        # 打印警告信息，提示不要提交
        print("DO *NOT* COMMIT!")

    # 将失败的计数转换为整数并返回
    return int(failed)
# 编译正则表达式，匹配形如 'a/b' 的字符串，其中 a 和 b 是整数
sp = re.compile(r'([0-9]+)/([1-9][0-9]*)')

def split_list(l, split, density=None):
    """
    将列表分割成若干部分

    split 应为形如 'a/b' 的字符串，例如 '1/3' 表示将列表分割为三等分之一的第一部分。

    如果列表的长度不能被分割数整除，最后一个部分将含有更多的元素。

    如果指定了 `density` 参数为一个列表，将会尽量平衡各部分的质量。

    >>> from sympy.testing.runtests import split_list
    >>> a = list(range(10))
    >>> split_list(a, '1/3')
    [0, 1, 2]
    >>> split_list(a, '2/3')
    [3, 4, 5]
    >>> split_list(a, '3/3')
    [6, 7, 8, 9]
    """
    # 匹配 split 字符串的格式是否符合要求
    m = sp.match(split)
    if not m:
        raise ValueError("split must be a string of the form a/b where a and b are ints")
    # 提取出分子和分母
    i, t = map(int, m.groups())

    if not density:
        # 如果未指定 density，则按照分割比例直接返回列表的对应部分
        return l[(i - 1)*len(l)//t : i*len(l)//t]

    # 标准化 density 列表
    tot = sum(density)
    density = [x / tot for x in density]

    def density_inv(x):
        """插值密度的累积分布函数的反函数"""
        if x <= 0:
            return 0
        if x >= sum(density):
            return 1

        # 找到第一次累积和超过 x 的位置，并进行线性插值
        cumm = 0
        for i, d in enumerate(density):
            cumm += d
            if cumm >= x:
                break
        frac = (d - (cumm - x)) / d
        return (i + frac) / len(density)

    # 计算较低和较高分位点对应的密度反函数值，然后在列表中切分
    lower_frac = density_inv((i - 1) / t)
    higher_frac = density_inv(i / t)
    return l[int(lower_frac*len(l)) : int(higher_frac*len(l))]

from collections import namedtuple
SymPyTestResults = namedtuple('SymPyTestResults', 'failed attempted')

def sympytestfile(filename, module_relative=True, name=None, package=None,
             globs=None, verbose=None, report=True, optionflags=0,
             extraglobs=None, raise_on_error=False,
             parser=pdoctest.DocTestParser(), encoding=None):

    """
    测试给定文件中的示例。返回 (#失败数, #测试总数)。

    可选关键字参数 `module_relative` 指定文件名的解释方式：

    - 如果 `module_relative` 为 True（默认），则 `filename` 指定一个相对于调用模块目录的模块相对路径。
      如果指定了 `package` 参数，则相对于该包。
      为了确保跨平台兼容性，`filename` 应使用 "/" 分隔路径段，且不应为绝对路径（即不能以 "/" 开头）。

    - 如果 `module_relative` 为 False，则 `filename` 指定一个特定于操作系统的路径。
      路径可以是绝对的或相对于当前工作目录的。

    可选关键字参数 `name` 指定测试的名称；默认使用文件的基本名称。
    """
    pass  # 这里只是一个文档字符串示例，未实现具体的功能逻辑，因此留空
    """
    Optional keyword argument ``package`` is a Python package or the
    name of a Python package whose directory should be used as the
    base directory for a module relative filename.  If no package is
    specified, then the calling module's directory is used as the base
    directory for module relative filenames.  It is an error to
    specify ``package`` if ``module_relative`` is False.

    Optional keyword arg ``globs`` gives a dict to be used as the globals
    when executing examples; by default, use {}.  A copy of this dict
    is actually used for each docstring, so that each docstring's
    examples start with a clean slate.

    Optional keyword arg ``extraglobs`` gives a dictionary that should be
    merged into the globals that are used to execute examples.  By
    default, no extra globals are used.

    Optional keyword arg ``verbose`` prints lots of stuff if true, prints
    only failures if false; by default, it's true iff "-v" is in sys.argv.

    Optional keyword arg ``report`` prints a summary at the end when true,
    else prints nothing at the end.  In verbose mode, the summary is
    detailed, else very brief (in fact, empty if all tests passed).

    Optional keyword arg ``optionflags`` or's together module constants,
    and defaults to 0.  Possible values (see the docs for details):

    - DONT_ACCEPT_TRUE_FOR_1
    - DONT_ACCEPT_BLANKLINE
    - NORMALIZE_WHITESPACE
    - ELLIPSIS
    - SKIP
    - IGNORE_EXCEPTION_DETAIL
    - REPORT_UDIFF
    - REPORT_CDIFF
    - REPORT_NDIFF
    - REPORT_ONLY_FIRST_FAILURE

    Optional keyword arg ``raise_on_error`` raises an exception on the
    first unexpected exception or failure. This allows failures to be
    post-mortem debugged.

    Optional keyword arg ``parser`` specifies a DocTestParser (or
    subclass) that should be used to extract tests from the files.

    Optional keyword arg ``encoding`` specifies an encoding that should
    be used to convert the file to unicode.

    Advanced tomfoolery:  testmod runs methods of a local instance of
    class doctest.Tester, then merges the results into (or creates)
    global Tester instance doctest.master.  Methods of doctest.master
    can be called directly too, if you want to do something unusual.
    Passing report=0 to testmod is especially useful then, to delay
    displaying a summary.  Invoke doctest.master.summarize(verbose)
    when you're done fiddling.
    """
    # 如果指定了 package 但 module_relative 不是 True，则抛出 ValueError
    if package and not module_relative:
        raise ValueError("Package may only be specified for module-"
                         "relative paths.")

    # 使用 pdoctest._load_testfile 函数加载测试文件内容并获取文件名
    text, filename = pdoctest._load_testfile(
        filename, package, module_relative, encoding)

    # 如果未指定 name，则使用文件的基本名称
    if name is None:
        name = os.path.basename(filename)

    # 组装全局变量字典，如果 globs 为 None，则使用空字典 {}，否则复制 globs
    if globs is None:
        globs = {}
    else:
        globs = globs.copy()
    # 如果 extraglobs 参数不为空，则更新当前的全局变量字典 globs
    if extraglobs is not None:
        globs.update(extraglobs)
    
    # 如果全局变量字典 globs 中不存在 '__name__' 键，则将其设置为 '__main__'
    if '__name__' not in globs:
        globs['__name__'] = '__main__'

    # 如果 raise_on_error 为 True，则创建一个 pdoctest.DebugRunner 对象作为测试运行器
    # 并设置其参数为 verbose 和 optionflags
    if raise_on_error:
        runner = pdoctest.DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        # 如果 raise_on_error 为 False，则创建一个 SymPyDocTestRunner 对象作为测试运行器
        # 同时设置其输出检查器为 SymPyOutputChecker 对象
        runner = SymPyDocTestRunner(verbose=verbose, optionflags=optionflags)
        runner._checker = SymPyOutputChecker()

    # 使用 parser.get_doctest 方法从文本 text 中解析并获取 doctest 测试对象
    test = parser.get_doctest(text, globs, name, filename, 0)
    
    # 运行测试对象 test
    runner.run(test)

    # 如果设置了 report 标志为 True，则打印测试运行的总结信息
    if report:
        runner.summarize()

    # 如果 pdoctest.master 为 None，则将当前 runner 赋值给 pdoctest.master
    # 否则，将当前 runner 的结果合并到 pdoctest.master 中
    if pdoctest.master is None:
        pdoctest.master = runner
    else:
        pdoctest.master.merge(runner)

    # 返回 SymPyTestResults 对象，其中包含 runner 的失败和尝试次数
    return SymPyTestResults(runner.failures, runner.tries)
class SymPyTests:
    # SymPy 测试类

    def __init__(self, reporter, kw="", post_mortem=False,
                 seed=None, fast_threshold=None, slow_threshold=None):
        # 初始化方法，设置测试报告器、关键字、是否自动调试、随机种子、快速和慢速阈值
        self._post_mortem = post_mortem  # 是否自动调试
        self._kw = kw  # 关键字
        self._count = 0  # 计数器初始化为0
        self._root_dir = get_sympy_dir()  # 获取 SymPy 目录路径
        self._reporter = reporter  # 测试报告器
        self._reporter.root_dir(self._root_dir)  # 设置报告器的根目录为 SymPy 目录
        self._testfiles = []  # 测试文件列表初始化为空
        self._seed = seed if seed is not None else random.random()  # 设置随机种子

        # 默认阈值，单位为秒，来自人类/用户体验设计限制
        # http://www.nngroup.com/articles/response-times-3-important-limits/
        #
        # 这些默认值并非铁板钉钉，因为我们在测量不同的事物，所以其他人可以提出更好的标准 :)
        if fast_threshold:
            self._fast_threshold = float(fast_threshold)  # 设置快速阈值
        else:
            self._fast_threshold = 8  # 默认快速阈值为8秒
        if slow_threshold:
            self._slow_threshold = float(slow_threshold)  # 设置慢速阈值
        else:
            self._slow_threshold = 10  # 默认慢速阈值为10秒

    def test(self, sort=False, timeout=False, slow=False,
            enhance_asserts=False, fail_on_timeout=False):
        """
        Runs the tests returning True if all tests pass, otherwise False.

        If sort=False run tests in random order.
        """
        # 执行测试，如果所有测试通过返回True，否则返回False
        if sort:
            self._testfiles.sort()  # 如果sort为True，按顺序运行测试文件
        elif slow:
            pass  # 如果slow为True，则不做任何排序
        else:
            random.seed(self._seed)
            random.shuffle(self._testfiles)  # 使用随机种子打乱测试文件顺序
        self._reporter.start(self._seed)  # 启动测试报告器
        for f in self._testfiles:
            try:
                self.test_file(f, sort, timeout, slow,
                    enhance_asserts, fail_on_timeout)  # 对每个测试文件执行测试
            except KeyboardInterrupt:
                print(" interrupted by user")  # 如果被用户中断，则打印信息
                self._reporter.finish()  # 结束测试报告
                raise
        return self._reporter.finish()  # 返回最终测试结果
    # 导入需要的模块和类
    def _enhance_asserts(self, source):
        from ast import (NodeTransformer, Compare, Name, Store, Load, Tuple,
            Assign, BinOp, Str, Mod, Assert, parse, fix_missing_locations)

        # 操作符字典，将比较操作的类名映射为对应的字符串表示
        ops = {"Eq": '==', "NotEq": '!=', "Lt": '<', "LtE": '<=',
                "Gt": '>', "GtE": '>=', "Is": 'is', "IsNot": 'is not',
                "In": 'in', "NotIn": 'not in'}

        # 定义一个 AST 转换类
        class Transform(NodeTransformer):
            def visit_Assert(self, stmt):
                # 如果断言语句的测试部分是比较操作
                if isinstance(stmt.test, Compare):
                    compare = stmt.test
                    # 提取比较操作的左操作数和右操作数
                    values = [compare.left] + compare.comparators
                    # 为每个值生成一个唯一的名称
                    names = [ "_%s" % i for i, _ in enumerate(values) ]
                    # 创建用于存储值的 Name 节点列表
                    names_store = [ Name(n, Store()) for n in names ]
                    # 创建用于加载值的 Name 节点列表
                    names_load = [ Name(n, Load()) for n in names ]
                    # 创建一个 Tuple 节点作为赋值的目标
                    target = Tuple(names_store, Store())
                    # 创建一个 Tuple 节点作为表达式的值
                    value = Tuple(values, Load())
                    # 创建赋值语句
                    assign = Assign([target], value)
                    # 创建新的比较操作
                    new_compare = Compare(names_load[0], compare.ops, names_load[1:])
                    # 创建断言消息的格式字符串
                    msg_format = "\n%s " + "\n%s ".join([ ops[op.__class__.__name__] for op in compare.ops ]) + "\n%s"
                    # 创建消息表达式
                    msg = BinOp(Str(msg_format), Mod(), Tuple(names_load, Load()))
                    # 创建新的断言语句并保留原始的行号和列偏移
                    test = Assert(new_compare, msg, lineno=stmt.lineno, col_offset=stmt.col_offset)
                    # 返回赋值语句和新的断言语句的列表
                    return [assign, test]
                else:
                    # 如果不是比较操作，则返回原始的断言语句
                    return stmt

        # 解析源代码生成抽象语法树
        tree = parse(source)
        # 使用定义的 AST 转换器进行转换
        new_tree = Transform().visit(tree)
        # 修复缺失的位置信息并返回新的抽象语法树
        return fix_missing_locations(new_tree)

    # 设置函数超时处理
    def _timeout(self, function, timeout, fail_on_timeout):
        def callback(x, y):
            # 响应超时信号，关闭当前的超时报警
            signal.alarm(0)
            # 如果设置了超时后失败选项，则抛出超时异常
            if fail_on_timeout:
                raise TimeOutError("Timed out after %d seconds" % timeout)
            else:
                # 否则抛出跳过异常
                raise Skipped("Timeout")
        # 设置信号处理函数
        signal.signal(signal.SIGALRM, callback)
        # 设置定时器超时时间
        signal.alarm(timeout)  # Set an alarm with a given timeout
        # 执行指定函数
        function()
        # 关闭定时器超时报警
        signal.alarm(0)  # Disable the alarm

    # 检查关键字表达式是否匹配给定的对象
    def matches(self, x):
        """
        Does the keyword expression self._kw match "x"? Returns True/False.

        Always returns True if self._kw is "".
        """
        # 如果关键字表达式为空，则始终返回 True
        if not self._kw:
            return True
        # 遍历关键字列表，检查是否有关键字在对象名称中出现
        for kw in self._kw:
            if x.__name__.lower().find(kw.lower()) != -1:
                return True
        # 如果没有匹配的关键字，则返回 False
        return False

    # 获取指定目录下匹配指定模式的测试文件列表
    def get_test_files(self, dir, pat='test_*.py'):
        """
        Returns the list of test_*.py (default) files at or below directory
        ``dir`` relative to the SymPy home directory.
        """
        # 将目录路径转换为本地路径
        dir = os.path.join(self._root_dir, convert_to_native_paths([dir])[0])

        # 初始化一个空列表用于存储匹配的文件路径
        g = []
        # 遍历指定目录及其子目录下的所有文件和文件夹
        for path, folders, files in os.walk(dir):
            # 将符合指定模式的文件路径添加到列表中
            g.extend([os.path.join(path, f) for f in files if fnmatch(f, pat)])

        # 返回排序后的文件路径列表
        return sorted([os.path.normcase(gi) for gi in g])
# 定义一个类 SymPyDocTests，用于执行 SymPy 文档测试
class SymPyDocTests:

    # 初始化方法，接受两个参数：reporter 和 normal
    def __init__(self, reporter, normal):
        # 初始化测试计数器
        self._count = 0
        # 获取 SymPy 的根目录路径
        self._root_dir = get_sympy_dir()
        # 设置报告器对象，并传入 SymPy 根目录路径
        self._reporter = reporter
        self._reporter.root_dir(self._root_dir)
        # 设置 normal 属性
        self._normal = normal

        # 初始化测试文件列表为空
        self._testfiles = []

    # 定义测试方法，运行测试文件并返回测试结果
    def test(self):
        """
        Runs the tests and returns True if all tests pass, otherwise False.
        """
        # 启动报告器
        self._reporter.start()
        # 遍历测试文件列表
        for f in self._testfiles:
            try:
                # 执行单个测试文件的测试
                self.test_file(f)
            except KeyboardInterrupt:
                # 捕获用户中断异常，打印中断消息，完成报告并重新抛出异常
                print(" interrupted by user")
                self._reporter.finish()
                raise
        # 完成所有测试后，结束报告并返回结果
        return self._reporter.finish()

    # 获取测试文件列表的方法，可以指定目录和文件匹配模式，默认为 *.py 文件
    def get_test_files(self, dir, pat='*.py', init_only=True):
        r"""
        Returns the list of \*.py files (default) from which docstrings
        will be tested which are at or below directory ``dir``. By default,
        only those that have an __init__.py in their parent directory
        and do not start with ``test_`` will be included.
        """
        # 内部函数，用于检查给定路径是否可导入为模块
        def importable(x):
            """
            Checks if given pathname x is an importable module by checking for
            __init__.py file.

            Returns True/False.

            Currently we only test if the __init__.py file exists in the
            directory with the file "x" (in theory we should also test all the
            parent dirs).
            """
            # 构建 __init__.py 文件路径
            init_py = os.path.join(os.path.dirname(x), "__init__.py")
            # 检查文件是否存在
            return os.path.exists(init_py)

        # 将目录路径转换为本地路径格式，并加入 SymPy 根目录路径
        dir = os.path.join(self._root_dir, convert_to_native_paths([dir])[0])

        # 初始化空列表 g，用于存储符合条件的文件路径
        g = []
        # 遍历目录及其子目录下的文件
        for path, folders, files in os.walk(dir):
            # 将符合条件的文件路径添加到 g 列表中
            g.extend([os.path.join(path, f) for f in files
                      if not f.startswith('test_') and fnmatch(f, pat)])
        
        # 如果指定仅包括可导入模块文件，则筛选出符合条件的文件路径
        if init_only:
            g = [x for x in g if importable(x)]

        # 返回符合条件的文件路径列表，统一使用小写路径格式
        return [os.path.normcase(gi) for gi in g]
    def _check_dependencies(self,
                            executables=(),
                            modules=(),
                            disable_viewers=(),
                            python_version=(3, 5),
                            ground_types=None):
        """
        Checks if the dependencies for the test are installed.

        Raises ``DependencyError`` it at least one dependency is not installed.
        """

        # 检查是否所有的可执行文件都可以在系统的 PATH 中找到
        for executable in executables:
            if not shutil.which(executable):
                raise DependencyError("Could not find %s" % executable)

        # 检查是否所有的模块都能成功导入
        for module in modules:
            if module == 'matplotlib':
                # 特别处理 matplotlib 模块的导入
                matplotlib = import_module(
                    'matplotlib',
                    import_kwargs={'fromlist':
                                      ['pyplot', 'cm', 'collections']},
                    min_module_version='1.0.0', catch=(RuntimeError,))
                if matplotlib is None:
                    raise DependencyError("Could not import matplotlib")
            else:
                # 普通模块的导入检查
                if not import_module(module):
                    raise DependencyError("Could not import %s" % module)

        # 如果有禁用的视图程序，则将临时目录添加到系统 PATH 中
        if disable_viewers:
            tempdir = tempfile.mkdtemp()
            os.environ['PATH'] = '%s:%s' % (tempdir, os.environ['PATH'])

            # 为每个禁用的视图程序创建临时的可执行文件
            vw = ('#!/usr/bin/env python3\n'
                  'import sys\n'
                  'if len(sys.argv) <= 1:\n'
                  '    exit("wrong number of args")\n')

            for viewer in disable_viewers:
                with open(os.path.join(tempdir, viewer), 'w') as fh:
                    fh.write(vw)

                # 设置文件为可执行
                os.chmod(os.path.join(tempdir, viewer),
                         stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)

        # 检查 Python 版本是否满足要求
        if python_version:
            if sys.version_info < python_version:
                raise DependencyError("Requires Python >= " + '.'.join(map(str, python_version)))

        # 检查是否指定了 ground_types，并且是否符合要求
        if ground_types is not None:
            if GROUND_TYPES not in ground_types:
                raise DependencyError("Requires ground_types in " + str(ground_types))

        # 如果 'pyglet' 在模块列表中，执行以下代码
        if 'pyglet' in modules:
            # monkey-patch pyglet，使其在文档测试时不打开窗口
            import pyglet
            class DummyWindow:
                def __init__(self, *args, **kwargs):
                    self.has_exit = True
                    self.width = 600
                    self.height = 400

                def set_vsync(self, x):
                    pass

                def switch_to(self):
                    pass

                def push_handlers(self, x):
                    pass

                def close(self):
                    pass

            pyglet.window.Window = DummyWindow
class SymPyDocTestFinder(DocTestFinder):
    """
    SymPyDocTestFinder 类继承自 DocTestFinder 类，用于提取与给定对象相关的文档测试。
    文档测试可以从以下对象类型中提取：模块（modules）、函数（functions）、类（classes）、
    方法（methods）、静态方法（staticmethods）、类方法（classmethods）和属性（properties）。

    从 doctest 的版本修改而来，以更深入地查找看起来来自不同模块的代码。例如，@vectorize 装饰器使得函数看起来来自
    multidimensional.py，尽管它们的代码实际上存在于其他地方。
    """
    def _get_test(self, obj, name, module, globs, source_lines):
        """
        Return a DocTest for the given object, if it defines a docstring;
        otherwise, return None.
        """

        lineno = None

        # Extract the object's docstring.  If it does not have one,
        # then return None (no test for this object).
        if isinstance(obj, str):
            # obj is a string in the case for objects in the polys package.
            # Note that source_lines is a binary string (compiled polys
            # modules), which can't be handled by _find_lineno so determine
            # the line number here.

            # 将 obj 当作字符串处理，通常出现在 polys 包中的对象中。
            # 注意 source_lines 是二进制字符串（编译后的 polys 模块），
            # 无法由 _find_lineno 处理，因此在这里确定行号。
            docstring = obj

            # 查找字符串中的行号信息并验证
            matches = re.findall(r"line \d+", name)
            assert len(matches) == 1, \
                "string '%s' does not contain lineno " % name

            # 提取行号，这里获取的不是确切的行号，但总比没有要好
            lineno = int(matches[0][5:])

        else:
            # 否则从 obj 中获取 docstring，如果不存在则设为空字符串
            docstring = getattr(obj, '__doc__', '')
            if docstring is None:
                docstring = ''
            if not isinstance(docstring, str):
                docstring = str(docstring)

        # 如果设定了排除空 docstring 并且 docstring 为空，则返回 None
        if self._exclude_empty and not docstring:
            return None

        # 对于属性，确保有 docstring，因为 _find_lineno 假定其存在
        if isinstance(obj, property):
            if obj.fget.__doc__ is None:
                return None

        # 查找 docstring 在文件中的位置
        if lineno is None:
            obj = unwrap(obj)
            # 对于属性的处理并未在 _find_lineno 中实现，因此在此处进行处理
            if hasattr(obj, 'func_closure') and obj.func_closure is not None:
                tobj = obj.func_closure[0].cell_contents
            elif isinstance(obj, property):
                tobj = obj.fget
            else:
                tobj = obj
            lineno = self._find_lineno(tobj, source_lines)

        if lineno is None:
            return None

        # 返回给定对象的 DocTest
        if module is None:
            filename = None
        else:
            filename = getattr(module, '__file__', module.__name__)
            if filename[-4:] in (".pyc", ".pyo"):
                filename = filename[:-1]

        # 设置全局变量中的 _doctest_depends_on，用于获取对象的依赖信息
        globs['_doctest_depends_on'] = getattr(obj, '_doctest_depends_on', {})

        # 调用解析器获取给定对象的 DocTest
        return self._parser.get_doctest(docstring, globs, name,
                                        filename, lineno)
class SymPyDocTestRunner(DocTestRunner):
    """
    用于运行 DocTest 测试用例并累积统计信息的类。
    其中 ``run`` 方法用于处理单个 DocTest 测试用例。
    返回一个元组 ``(f, t)``, 其中 ``t`` 是尝试的测试用例数，
    ``f`` 是失败的测试用例数。

    从 doctest 版本修改而来，不重置 sys.displayhook（参见问题 5140）。

    更多信息请参阅原始 DocTestRunner 的文档字符串。
    """
    # 定义一个方法 run，用于执行测试例 test，并使用指定的编译标志 compileflags 和输出函数 out
    def run(self, test, compileflags=None, out=None, clear_globs=True):
        """
        Run the examples in ``test``, and display the results using the
        writer function ``out``.

        The examples are run in the namespace ``test.globs``.  If
        ``clear_globs`` is true (the default), then this namespace will
        be cleared after the test runs, to help with garbage
        collection.  If you would like to examine the namespace after
        the test completes, then use ``clear_globs=False``.

        ``compileflags`` gives the set of flags that should be used by
        the Python compiler when running the examples.  If not
        specified, then it will default to the set of future-import
        flags that apply to ``globs``.

        The output of each example is checked using
        ``SymPyDocTestRunner.check_output``, and the results are
        formatted by the ``SymPyDocTestRunner.report_*`` methods.
        """
        self.test = test

        # 遍历测试例中的每个例子
        for example in test.examples:
            # 替换例子中可能出现的 Markdown 文件的结束符 '```'
            example.want = example.want.replace('```\n', '')
            if example.exc_msg:
                example.exc_msg = example.exc_msg.replace('```\n', '')

        # 如果未指定编译标志，使用 test.globs 的未来导入标志
        if compileflags is None:
            compileflags = pdoctest._extract_future_flags(test.globs)

        save_stdout = sys.stdout
        # 如果未提供输出函数，使用默认的 sys.stdout.write
        if out is None:
            out = save_stdout.write
        sys.stdout = self._fakeout

        # 重定向 pdb.set_trace，以便在交互式调试期间恢复 sys.stdout
        save_set_trace = pdb.set_trace
        self.debugger = pdoctest._OutputRedirectingPdb(save_stdout)
        self.debugger.reset()
        pdb.set_trace = self.debugger.set_trace

        # 重定向 linecache.getlines，以便在调试器内部查看例子的源代码
        self.save_linecache_getlines = pdoctest.linecache.getlines
        linecache.getlines = self.__patched_linecache_getlines

        # 检查是否存在弃用警告，若存在则引发异常
        with raise_on_deprecated():
            try:
                # 调用 __run 方法执行测试，传入测试例、编译标志和输出函数
                return self.__run(test, compileflags, out)
            finally:
                # 恢复 sys.stdout、pdb.set_trace 和 linecache.getlines 的原始状态
                sys.stdout = save_stdout
                pdb.set_trace = save_set_trace
                linecache.getlines = self.save_linecache_getlines
                # 如果需要清空命名空间中的全局变量，则执行清空操作
                if clear_globs:
                    test.globs.clear()
# 需要重写的方法名列表，这些方法名带有名称混淆（name mangling）
monkeypatched_methods = [
    'patched_linecache_getlines',
    'run',
    'record_outcome'
]

# 遍历要重写的方法名列表
for method in monkeypatched_methods:
    # 构造旧名称和新名称，通过名称混淆规则进行修改
    oldname = '_DocTestRunner__' + method
    newname = '_SymPyDocTestRunner__' + method
    # 设置 SymPyDocTestRunner 类的新方法，使其等于 DocTestRunner 类的旧方法
    setattr(SymPyDocTestRunner, newname, getattr(DocTestRunner, oldname))


class SymPyOutputChecker(pdoctest.OutputChecker):
    """
    相较于标准库中的 OutputChecker，我们的 OutputChecker 类支持对 doctest 示例中出现的浮点数进行数值比较
    """

    def __init__(self):
        # 注意，OutputChecker 是一个旧式类，没有 __init__ 方法，因此无法调用基类的 __init__ 方法

        # 匹配浮点数格式的正则表达式
        got_floats = r'(\d+\.\d*|\.\d+)'

        # 'want' 字符串中的浮点数可能包含省略号
        want_floats = got_floats + r'(\.{3})?'

        # 浮点数前面的分隔符，可以是空白字符或者运算符
        front_sep = r'\s|\+|\-|\*|,'
        # 浮点数后面的分隔符，包括前面的分隔符以及可能的虚部或指数部分
        back_sep = front_sep + r'|j|e'

        # 匹配 'got' 字符串中浮点数的正则表达式
        fbeg = r'^%s(?=%s|$)' % (got_floats, back_sep)
        fmidend = r'(?<=%s)%s(?=%s|$)' % (front_sep, got_floats, back_sep)
        self.num_got_rgx = re.compile(r'(%s|%s)' %(fbeg, fmidend))

        # 匹配 'want' 字符串中浮点数的正则表达式
        fbeg = r'^%s(?=%s|$)' % (want_floats, back_sep)
        fmidend = r'(?<=%s)%s(?=%s|$)' % (front_sep, want_floats, back_sep)
        self.num_want_rgx = re.compile(r'(%s|%s)' %(fbeg, fmidend))
    def check_output(self, want, got, optionflags):
        """
        Return True iff the actual output from an example (`got`)
        matches the expected output (`want`).  These strings are
        always considered to match if they are identical; but
        depending on what option flags the test runner is using,
        several non-exact match types are also possible.  See the
        documentation for `TestRunner` for more information about
        option flags.
        """
        # 处理最常见的情况，效率最高：
        # 如果字符串完全相同，则直接返回True。
        if got == want:
            return True

        # TODO 是否也应该解析整数？
        # 解析浮点数并进行比较。如果某些解析后的浮点数包含省略号，则跳过比较。
        matches = self.num_got_rgx.finditer(got)
        numbers_got = [match.group(1) for match in matches]  # 匹配到的浮点数字符串列表
        matches = self.num_want_rgx.finditer(want)
        numbers_want = [match.group(1) for match in matches]  # 匹配到的期望浮点数字符串列表
        if len(numbers_got) != len(numbers_want):
            return False

        if len(numbers_got) > 0:
            nw_ = []
            for ng, nw in zip(numbers_got, numbers_want):
                if '...' in nw:
                    nw_.append(ng)
                    continue
                else:
                    nw_.append(nw)

                if abs(float(ng) - float(nw)) > 1e-5:
                    return False

            got = self.num_got_rgx.sub(r'%s', got)
            got = got % tuple(nw_)

        # <BLANKLINE> 可以作为特殊序列表示空行，除非使用了 DONT_ACCEPT_BLANKLINE 标志。
        if not (optionflags & pdoctest.DONT_ACCEPT_BLANKLINE):
            # 用空行替换 want 中的 <BLANKLINE>。
            want = re.sub(r'(?m)^%s\s*?$' % re.escape(pdoctest.BLANKLINE_MARKER),
                          '', want)
            # 如果 got 中的某行只包含空格，则移除这些空格。
            got = re.sub(r'(?m)^\s*?$', '', got)
            if got == want:
                return True

        # 此标志导致 doctest 忽略空白字符串内容的差异。注意，这可以与 ELLIPSIS 标志一起使用。
        if optionflags & pdoctest.NORMALIZE_WHITESPACE:
            got = ' '.join(got.split())
            want = ' '.join(want.split())
            if got == want:
                return True

        # ELLIPSIS 标志允许 "..." 在 want 中匹配 got 中的任何子串。
        if optionflags & pdoctest.ELLIPSIS:
            if pdoctest._ellipsis_match(want, got):
                return True

        # 没有找到匹配项，返回 False。
        return False
    """
    Parent class for all reporters.
    """
    pass



    """
    Py.test like reporter. Should produce output identical to py.test.
    """

    def __init__(self, verbose=False, tb="short", colors=True,
                 force_colors=False, split=None):
        # 初始化测试报告的各种属性
        self._verbose = verbose  # 是否详细输出
        self._tb_style = tb  # 回溯信息的样式
        self._colors = colors  # 是否使用彩色输出
        self._force_colors = force_colors  # 是否强制使用彩色
        self._xfailed = 0  # 记录失败但标记为预期失败的测试数量
        self._xpassed = []  # 记录通过但标记为预期失败的测试列表
        self._failed = []  # 记录失败的测试列表
        self._failed_doctest = []  # 记录失败的文档测试列表
        self._passed = 0  # 记录通过的测试数量
        self._skipped = 0  # 记录跳过的测试数量
        self._exceptions = []  # 记录发生异常的测试列表
        self._terminal_width = None  # 终端的宽度
        self._default_width = 80  # 默认的终端宽度
        self._split = split  # 分割输出的标记
        self._active_file = ''  # 当前活动的文件名
        self._active_f = None  # 当前活动的文件对象

        # TODO: Should these be protected?
        # 这些属性用于记录慢速和快速测试函数的列表，应该考虑将其保护起来吗？
        self.slow_test_functions = []  # 慢速测试函数列表
        self.fast_test_functions = []  # 快速测试函数列表

        # this tracks the x-position of the cursor (useful for positioning
        # things on the screen), without the need for any readline library:
        # 用于跟踪光标的 x 位置，有助于屏幕上的定位，无需 readline 库：
        self._write_pos = 0  # 光标的 x 位置
        self._line_wrap = False  # 是否换行输出
    def terminal_width(self):
        # 如果已经计算过终端宽度，则直接返回之前保存的值
        if self._terminal_width is not None:
            return self._terminal_width

        def findout_terminal_width():
            # 如果运行在 Windows 平台
            if sys.platform == "win32":
                # Windows 平台的支持基于以下链接中的代码：
                #
                #  http://code.activestate.com/recipes/
                #  440694-determine-size-of-console-window-on-windows/

                from ctypes import windll, create_string_buffer

                h = windll.kernel32.GetStdHandle(-12)
                csbi = create_string_buffer(22)
                res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)

                if res:
                    import struct
                    (_, _, _, _, _, left, _, right, _, _, _) = \
                        struct.unpack("hhhhHhhhhhh", csbi.raw)
                    return right - left
                else:
                    return self._default_width

            # 如果 stdout 不是交互式终端，则使用默认宽度
            if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
                return self._default_width  # leave PIPEs alone

            try:
                # 尝试运行 'stty -a' 命令获取终端宽度信息
                process = subprocess.Popen(['stty', '-a'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                stdout = stdout.decode("utf-8")
            except OSError:
                pass
            else:
                # 支持从 stty 命令获取的以下输出格式：
                #
                # 1) Linux   -> columns 80
                # 2) OS X    -> 80 columns
                # 3) Solaris -> columns = 80

                re_linux = r"columns\s+(?P<columns>\d+);"
                re_osx = r"(?P<columns>\d+)\s*columns;"
                re_solaris = r"columns\s+=\s+(?P<columns>\d+);"

                for regex in (re_linux, re_osx, re_solaris):
                    match = re.search(regex, stdout)

                    if match is not None:
                        columns = match.group('columns')

                        try:
                            width = int(columns)
                        except ValueError:
                            pass
                        if width != 0:
                            return width

            return self._default_width

        # 调用内部函数获取终端宽度
        width = findout_terminal_width()
        # 将计算得到的宽度保存在实例变量中
        self._terminal_width = width

        # 返回最终计算得到的终端宽度
        return width

    def write_center(self, text, delim="="):
        # 获取终端宽度
        width = self.terminal_width
        # 如果文本不为空，则在文本两侧添加空格
        if text != "":
            text = " %s " % text
        # 计算居中对齐后的字符串
        idx = (width - len(text)) // 2
        t = delim*idx + text + delim*(width - idx - len(text))
        # 输出居中对齐的字符串，并换行
        self.write(t + "\n")

    def write_exception(self, e, val, tb):
        # 移除 traceback 链中的第一个元素，因为它总是 runtests.py
        tb = tb.tb_next
        # 格式化异常信息
        t = traceback.format_exception(e, val, tb)
        # 将格式化后的异常信息写入输出
        self.write("".join(t))
    # 启动函数，用于初始化并输出测试进程的相关信息
    def start(self, seed=None, msg="test process starts"):
        # 在控制台居中输出指定消息
        self.write_center(msg)
        # 获取当前 Python 解释器的可执行路径
        executable = sys.executable
        # 获取当前 Python 解释器的版本信息，并格式化为字符串
        v = tuple(sys.version_info)
        python_version = "%s.%s.%s-%s-%s" % v
        # 获取 Python 解释器的实现（如 CPython、PyPy 等）
        implementation = platform.python_implementation()
        # 如果是 PyPy，添加 PyPy 版本信息到实现名称中
        if implementation == 'PyPy':
            implementation += " %s.%s.%s-%s-%s" % sys.pypy_version_info
        # 输出 Python 解释器的相关信息：可执行路径、版本、实现名称
        self.write("executable:         %s  (%s) [%s]\n" %
            (executable, python_version, implementation))
        # 导入并输出 Sympy 库的架构信息
        from sympy.utilities.misc import ARCH
        self.write("architecture:       %s\n" % ARCH)
        # 导入并输出 Sympy 库的缓存使用情况
        from sympy.core.cache import USE_CACHE
        self.write("cache:              %s\n" % USE_CACHE)
        # 如果使用的是 gmpy 类型，则导入 gmpy2 并输出其版本信息
        version = ''
        if GROUND_TYPES =='gmpy':
            import gmpy2 as gmpy
            version = gmpy.version()
        self.write("ground types:       %s %s\n" % (GROUND_TYPES, version))
        # 导入并输出 numpy 库的版本信息
        numpy = import_module('numpy')
        self.write("numpy:              %s\n" % (None if not numpy else numpy.__version__))
        # 如果设置了随机种子，则输出随机种子值
        if seed is not None:
            self.write("random seed:        %d\n" % seed)
        # 导入并输出 Sympy 库的哈希随机化设置
        from sympy.utilities.misc import HASH_RANDOMIZATION
        self.write("hash randomization: ")
        # 获取环境变量中的 PYTHONHASHSEED 值或使用默认值 '0'
        hash_seed = os.getenv("PYTHONHASHSEED") or '0'
        # 如果启用了哈希随机化并且 PYTHONHASHSEED 是 "random" 或一个整数，则输出相关信息
        if HASH_RANDOMIZATION and (hash_seed == "random" or int(hash_seed)):
            self.write("on (PYTHONHASHSEED=%s)\n" % hash_seed)
        else:
            self.write("off\n")
        # 如果设置了分割点，则输出分割点信息
        if self._split:
            self.write("split:              %s\n" % self._split)
        # 输出空行
        self.write('\n')
        # 记录当前时间为测试开始时间
        self._t_start = clock()

    # 进入指定文件名的测试过程
    def entering_filename(self, filename, n):
        # 计算相对于根目录的文件名
        rel_name = filename[len(self._root_dir) + 1:]
        # 设置当前活动文件名
        self._active_file = rel_name
        # 初始化当前文件测试错误状态为 False
        self._active_file_error = False
        # 在控制台输出相对文件名及其编号
        self.write(rel_name)
        self.write("[%d] " % n)

    # 完成指定文件名的测试过程
    def leaving_filename(self):
        # 在控制台输出空格
        self.write(" ")
        # 根据当前活动文件的测试结果输出相应状态标记（FAIL 或 OK）
        if self._active_file_error:
            self.write("[FAIL]", "Red", align="right")
        else:
            self.write("[OK]", "Green", align="right")
        # 在控制台输出换行符
        self.write("\n")
        # 如果设置了详细输出模式，则输出额外的空行
        if self._verbose:
            self.write("\n")

    # 进入指定测试函数的测试过程
    def entering_test(self, f):
        # 设置当前活动测试函数
        self._active_f = f
        # 如果设置了详细输出模式，则在控制台输出测试函数名
        if self._verbose:
            self.write("\n" + f.__name__ + " ")

    # 标记当前测试为预期失败
    def test_xfail(self):
        # 增加预期失败测试计数
        self._xfailed += 1
        # 在控制台输出 'f' 并使用绿色显示
        self.write("f", "Green")

    # 标记当前测试为预期通过
    def test_xpass(self, v):
        # 获取消息字符串表示
        message = str(v)
        # 记录当前测试为预期通过，并记录消息
        self._xpassed.append((self._active_file, message))
        # 在控制台输出 'X' 并使用绿色显示
        self.write("X", "Green")

    # 标记当前测试为失败
    def test_fail(self, exc_info):
        # 记录当前测试失败的文件、测试函数和异常信息
        self._failed.append((self._active_file, self._active_f, exc_info))
        # 在控制台输出 'F' 并使用红色显示
        self.write("F", "Red")
        # 设置当前活动文件测试错误状态为 True
        self._active_file_error = True

    # 标记当前 doctest 测试为失败
    def doctest_fail(self, name, error_msg):
        # 第一行包含 "******"，将其从错误消息中移除
        error_msg = "\n".join(error_msg.split("\n")[1:])
        # 记录当前 doctest 测试失败的文件名和错误消息
        self._failed_doctest.append((name, error_msg))
        # 在控制台输出 'F' 并使用红色显示
        self.write("F", "Red")
        # 设置当前活动文件测试错误状态为 True
        self._active_file_error = True
    # 测试通过的情况下，增加通过计数
    def test_pass(self, char="."):
        self._passed += 1
        # 如果设置了详细输出模式
        if self._verbose:
            # 输出 "ok" 并使用绿色文本
            self.write("ok", "Green")
        else:
            # 否则输出指定字符，并使用绿色文本
            self.write(char, "Green")

    # 测试跳过的情况下，处理跳过逻辑
    def test_skip(self, v=None):
        # 默认字符为 "s"
        char = "s"
        # 增加跳过计数
        self._skipped += 1
        # 如果传入了自定义消息
        if v is not None:
            message = str(v)
            # 根据特定消息修改输出字符
            if message == "KeyboardInterrupt":
                char = "K"
            elif message == "Timeout":
                char = "T"
            elif message == "Slow":
                char = "w"
        # 如果设置了详细输出模式
        if self._verbose:
            # 如果有自定义消息，输出消息并使用蓝色文本
            if v is not None:
                self.write(message + ' ', "Blue")
            else:
                self.write(" - ", "Blue")
        # 输出指定字符，并使用蓝色文本
        self.write(char, "Blue")

    # 处理测试中的异常情况
    def test_exception(self, exc_info):
        # 将异常信息和相关文件信息添加到异常列表中
        self._exceptions.append((self._active_file, self._active_f, exc_info))
        # 如果异常是 TimeOutError 类型
        if exc_info[0] is TimeOutError:
            # 输出 "T" 表示超时，并使用红色文本
            self.write("T", "Red")
        else:
            # 否则输出 "E" 表示异常，并使用红色文本
            self.write("E", "Red")
        # 将活动文件错误标志设为 True
        self._active_file_error = True

    # 处理导入错误的情况
    def import_error(self, filename, exc_info):
        # 将异常信息和文件名添加到异常列表中
        self._exceptions.append((filename, None, exc_info))
        # 计算相对于根目录的文件名
        rel_name = filename[len(self._root_dir) + 1:]
        # 输出相对文件名
        self.write(rel_name)
        # 输出 "[?]   Failed to import"，并使用红色文本
        self.write("[?]   Failed to import", "Red")
        # 输出空格
        self.write(" ")
        # 输出 "[FAIL]"，右对齐，并使用红色文本
        self.write("[FAIL]", "Red", align="right")
        # 输出换行符
        self.write("\n")
```