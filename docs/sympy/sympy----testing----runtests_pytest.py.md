# `D:\src\scipysrc\sympy\sympy\testing\runtests_pytest.py`

```
    # 导入 functools 模块，用于函数装饰器和高阶函数
    import functools
    # 导入 importlib.util 模块，用于检查和加载模块
    import importlib.util
    # 导入 os 模块，提供与操作系统交互的功能
    import os
    # 导入 pathlib 模块，用于操作路径和文件系统
    import pathlib
    # 导入 re 模块，提供正则表达式操作支持
    import re
    # 导入 fnmatch 模块，提供文件名匹配支持
    from fnmatch import fnmatch
    # 导入 typing 模块，用于类型提示
    from typing import List, Optional, Tuple

    try:
        # 尝试导入 pytest 模块，用于运行测试用例
        import pytest
    except ImportError:
        # 如果导入失败，定义一个自定义异常类 NoPytestError
        class NoPytestError(Exception):
            """当试图使用 pytest 但未安装时抛出的异常。"""

        # 如果导入失败，定义一个虚拟的 pytest 类，用于模拟 pytest 的特性
        class pytest:  # type: ignore
            """当无法导入 pytest 时，用于支持 pytest 功能的影子类。"""

            @staticmethod
            def main(*args, **kwargs):
                """模拟 pytest 的主函数，抛出未安装 pytest 的异常信息。"""
                msg = 'pytest must be installed to run tests via this function'
                raise NoPytestError(msg)

    # 从 sympy.testing.runtests 模块中导入 test 函数
    from sympy.testing.runtests import test as test_sympy

    # 默认的测试路径列表
    TESTPATHS_DEFAULT = (
        pathlib.Path('sympy'),
        pathlib.Path('doc', 'src'),
    )
    # 默认的黑名单路径列表
    BLACKLIST_DEFAULT = (
        'sympy/integrals/rubi/rubi_tests/tests',
    )


    class PytestPluginManager:
        """SymPy 使用的 pytest 插件的模块名称。"""

        # 定义常量表示各个 pytest 插件的模块名
        PYTEST: str = 'pytest'
        RANDOMLY: str = 'pytest_randomly'
        SPLIT: str = 'pytest_split'
        TIMEOUT: str = 'pytest_timeout'
        XDIST: str = 'xdist'

        @functools.cached_property
        def has_pytest(self) -> bool:
            """检查是否安装了 pytest 插件。"""
            return bool(importlib.util.find_spec(self.PYTEST))

        @functools.cached_property
        def has_randomly(self) -> bool:
            """检查是否安装了 pytest-randomly 插件。"""
            return bool(importlib.util.find_spec(self.RANDOMLY))

        @functools.cached_property
        def has_split(self) -> bool:
            """检查是否安装了 pytest-split 插件。"""
            return bool(importlib.util.find_spec(self.SPLIT))

        @functools.cached_property
        def has_timeout(self) -> bool:
            """检查是否安装了 pytest-timeout 插件。"""
            return bool(importlib.util.find_spec(self.TIMEOUT))

        @functools.cached_property
        def has_xdist(self) -> bool:
            """检查是否安装了 pytest-xdist 插件。"""
            return bool(importlib.util.find_spec(self.XDIST))


    # 编译用于分割测试进度信息的正则表达式模式
    split_pattern = re.compile(r'([1-9][0-9]*)/([1-9][0-9]*)')


    @functools.lru_cache
    def sympy_dir() -> pathlib.Path:
        """返回 SymPy 根目录的路径。"""
        return pathlib.Path(__file__).parents[2]


    def update_args_with_rootdir(args: List[str]) -> List[str]:
        """将 `--rootdir` 和路径添加到传递给 `pytest.main` 的参数列表中。"""
    This is required to ensure that pytest is able to find the SymPy tests in
    instances where it gets confused determining the root directory, e.g. when
    running with Pyodide (e.g. `bin/test_pyodide.mjs`).

    """
    将 `--rootdir` 参数添加到 pytest 的参数列表中，以确保 pytest 能够正确地找到 SymPy 测试的根目录。
    这对于在运行时可能因为特定环境（如 Pyodide）而导致 pytest 无法准确确定根目录时特别重要。
    args.extend(['--rootdir', str(sympy_dir())])
    返回更新后的参数列表 args。
    ```
# 定义一个函数 `update_args_with_paths`，用于更新 `args` 列表，以便传递给 `pytest.main`。

def update_args_with_paths(
    paths: List[str],
    keywords: Optional[Tuple[str]],
    args: List[str],
) -> List[str]:
    """Appends valid paths and flags to the args `list` passed to `pytest.main`.

    The are three different types of "path" that a user may pass to the `paths`
    positional arguments, all of which need to be handled slightly differently:

    1. Nothing is passed
        The paths to the `testpaths` defined in `pytest.ini` need to be appended
        to the arguments list.
    2. Full, valid paths are passed
        These paths need to be validated but can then be directly appended to
        the arguments list.
    3. Partial paths are passed.
        The `testpaths` defined in `pytest.ini` need to be recursed and any
        matches be appended to the arguments list.

    """

    # 定义一个内部函数 `find_paths_matching_partial`，用于查找与部分路径匹配的路径

    def find_paths_matching_partial(partial_paths):
        partial_path_file_patterns = []
        # 遍历传入的部分路径列表
        for partial_path in partial_paths:
            # 根据部分路径长度进行不同情况的判断
            if len(partial_path) >= 4:
                has_test_prefix = partial_path[:4] == 'test'
                has_py_suffix = partial_path[-3:] == '.py'
            elif len(partial_path) >= 3:
                has_test_prefix = False
                has_py_suffix = partial_path[-3:] == '.py'
            else:
                has_test_prefix = False
                has_py_suffix = False
            # 根据前缀和后缀情况添加匹配模式到列表中
            if has_test_prefix and has_py_suffix:
                partial_path_file_patterns.append(partial_path)
            elif has_test_prefix:
                partial_path_file_patterns.append(f'{partial_path}*.py')
            elif has_py_suffix:
                partial_path_file_patterns.append(f'test*{partial_path}')
            else:
                partial_path_file_patterns.append(f'test*{partial_path}*.py')

        matches = []
        # 遍历默认有效的测试路径列表
        for testpath in valid_testpaths_default:
            # 使用 os.walk 遍历路径、目录和文件
            for path, dirs, files in os.walk(testpath, topdown=True):
                # 使用 zip 函数组合部分路径和部分路径匹配模式
                zipped = zip(partial_paths, partial_path_file_patterns)
                for (partial_path, partial_path_file) in zipped:
                    # 使用 fnmatch 函数检查路径是否与部分路径匹配
                    if fnmatch(path, f'*{partial_path}*'):
                        matches.append(str(pathlib.Path(path)))
                        # 如果匹配成功，停止遍历当前路径的子目录
                        dirs[:] = []
                    else:
                        # 如果路径不匹配，则检查当前目录下的文件是否与匹配模式匹配
                        for file in files:
                            if fnmatch(file, partial_path_file):
                                matches.append(str(pathlib.Path(path, file)))
        return matches

    # 定义一个内部函数 `is_tests_file`，用于判断给定文件路径是否为测试文件

    def is_tests_file(filepath: str) -> bool:
        path = pathlib.Path(filepath)
        # 检查路径是否为文件
        if not path.is_file():
            return False
        # 检查文件名是否以 'test_' 开头且文件后缀为 '.py'
        if not path.parts[-1].startswith('test_'):
            return False
        if not path.suffix == '.py':
            return False
        return True
    def find_tests_matching_keywords(keywords, filepath):
        matches = []  # 初始化一个空列表，用于存储匹配到的测试文件路径
        with open(filepath, encoding='utf-8') as tests_file:
            source = tests_file.read()  # 读取整个测试文件内容
            for line in source.splitlines():  # 遍历文件的每一行
                if line.lstrip().startswith('def '):  # 如果以'def '开头（即定义函数的行）
                    for kw in keywords:  # 遍历关键词列表
                        if line.lower().find(kw.lower()) != -1:  # 如果关键词在该行中出现（不区分大小写）
                            test_name = line.split(' ')[1].split('(')[0]  # 提取函数名
                            full_test_path = filepath + '::' + test_name  # 构建完整的测试路径
                            matches.append(full_test_path)  # 将完整测试路径添加到匹配列表中
        return matches  # 返回匹配到的测试文件路径列表

    valid_testpaths_default = []  # 初始化一个空列表，用于存储默认的有效测试文件路径
    for testpath in TESTPATHS_DEFAULT:  # 遍历默认测试文件路径列表
        absolute_testpath = pathlib.Path(sympy_dir(), testpath)  # 获取测试文件的绝对路径
        if absolute_testpath.exists():  # 如果路径存在
            valid_testpaths_default.append(str(absolute_testpath))  # 将路径转换为字符串并添加到有效路径列表中

    candidate_paths = []  # 初始化一个空列表，用于存储候选路径

    if paths:  # 如果给定了路径列表
        full_paths = []  # 初始化一个空列表，用于存储完整路径
        partial_paths = []  # 初始化一个空列表，用于存储部分路径
        for path in paths:  # 遍历给定的路径列表
            if pathlib.Path(path).exists():  # 如果路径存在
                full_paths.append(str(pathlib.Path(sympy_dir(), path)))  # 将完整路径添加到完整路径列表中
            else:  # 如果路径不存在
                partial_paths.append(path)  # 将路径添加到部分路径列表中
        matched_paths = find_paths_matching_partial(partial_paths)  # 查找与部分路径匹配的路径
        candidate_paths.extend(full_paths)  # 将完整路径列表中的路径添加到候选路径列表中
        candidate_paths.extend(matched_paths)  # 将匹配路径列表中的路径添加到候选路径列表中
    else:  # 如果未给定路径列表
        candidate_paths.extend(valid_testpaths_default)  # 将默认有效路径列表中的路径添加到候选路径列表中

    if keywords is not None and keywords != ():  # 如果给定了关键词且关键词不为空元组
        matches = []  # 初始化一个空列表，用于存储关键词匹配到的测试文件路径
        for path in candidate_paths:  # 遍历候选路径列表
            if is_tests_file(path):  # 如果路径是测试文件
                test_matches = find_tests_matching_keywords(keywords, path)  # 查找关键词匹配的测试文件
                matches.extend(test_matches)  # 将匹配到的测试文件路径列表添加到匹配列表中
            else:  # 如果路径不是测试文件
                for root, dirnames, filenames in os.walk(path):  # 遍历路径下的所有文件和子目录
                    for filename in filenames:  # 遍历所有文件名
                        absolute_filepath = str(pathlib.Path(root, filename))  # 获取文件的绝对路径
                        if is_tests_file(absolute_filepath):  # 如果是测试文件
                            test_matches = find_tests_matching_keywords(
                                keywords,
                                absolute_filepath,
                            )  # 查找关键词匹配的测试文件
                            matches.extend(test_matches)  # 将匹配到的测试文件路径列表添加到匹配列表中
        args.extend(matches)  # 将匹配到的测试文件路径列表添加到参数列表中
    else:  # 如果未给定关键词或关键词为空元组
        args.extend(candidate_paths)  # 将候选路径列表添加到参数列表中

    return args  # 返回参数列表
def make_absolute_path(partial_path: str) -> str:
    """Convert a partial path to an absolute path.

    A path such a `sympy/core` might be needed. However, absolute paths should
    be used in the arguments to pytest in all cases as it avoids errors that
    arise from nonexistent paths.

    This function assumes that partial_paths will be passed in such that they
    begin with the explicit `sympy` directory, i.e. `sympy/...`.

    """

    def is_valid_partial_path(partial_path: str) -> bool:
        """Assumption that partial paths are defined from the `sympy` root."""
        return pathlib.Path(partial_path).parts[0] == 'sympy'
        # 检查部分路径是否以 `sympy` 目录为根目录

    if not is_valid_partial_path(partial_path):
        msg = (
            f'Partial path {dir(partial_path)} is invalid, partial paths are '
            f'expected to be defined with the `sympy` directory as the root.'
        )
        raise ValueError(msg)
        # 如果部分路径无效，抛出 ValueError 异常

    absolute_path = str(pathlib.Path(sympy_dir(), partial_path))
    # 构建完整的绝对路径，基于 `sympy_dir()` 返回的路径和部分路径

    return absolute_path
    # 返回构建的绝对路径


def test(*paths, subprocess=True, rerun=0, **kwargs):
    """Interface to run tests via pytest compatible with SymPy's test runner.

    Explanation
    ===========

    Note that a `pytest.ExitCode`, which is an `enum`, is returned. This is
    different to the legacy SymPy test runner which would return a `bool`. If
    all tests sucessfully pass the `pytest.ExitCode.OK` with value `0` is
    returned, whereas the legacy SymPy test runner would return `True`. In any
    other scenario, a non-zero `enum` value is returned, whereas the legacy
    SymPy test runner would return `False`. Users need to, therefore, be careful
    if treating the pytest exit codes as booleans because
    `bool(pytest.ExitCode.OK)` evaluates to `False`, the opposite of legacy
    behaviour.

    Examples
    ========

    >>> import sympy  # doctest: +SKIP

    Run one file:

    >>> sympy.test('sympy/core/tests/test_basic.py')  # doctest: +SKIP
    >>> sympy.test('_basic')  # doctest: +SKIP

    Run all tests in sympy/functions/ and some particular file:

    >>> sympy.test("sympy/core/tests/test_basic.py",
    ...            "sympy/functions")  # doctest: +SKIP

    Run all tests in sympy/core and sympy/utilities:

    >>> sympy.test("/core", "/util")  # doctest: +SKIP

    Run specific test from a file:

    >>> sympy.test("sympy/core/tests/test_basic.py",
    ...            kw="test_equality")  # doctest: +SKIP

    Run specific test from any file:

    >>> sympy.test(kw="subs")  # doctest: +SKIP

    Run the tests using the legacy SymPy runner:

    >>> sympy.test(use_sympy_runner=True)  # doctest: +SKIP

    Note that this option is slated for deprecation in the near future and is
    only currently provided to ensure users have an alternative option while the
    pytest-based runner receives real-world testing.

    Parameters
    ==========
    # 定义函数参数
    paths : first n positional arguments of strings
        Paths, both partial and absolute, describing which subset(s) of the test
        suite are to be run.
    # 子进程标志，该选项已不再使用
    subprocess : bool, default is True
        Legacy option, is currently ignored.
    # 重新运行次数，该选项已不再使用
    rerun : int, default is 0
        Legacy option, is ignored.
    # 暂时选项，用于调用旧版 SymPy 测试运行器而非 `pytest.main`，将在不久的将来移除
    use_sympy_runner : bool or None, default is None
        Temporary option to invoke the legacy SymPy test runner instead of
        `pytest.main`. Will be removed in the near future.
    # 设置 pytest 输出的详细程度。设置为 `True` 将在 pytest 调用中添加 `--verbose` 选项。
    verbose : bool, default is False
        Sets the verbosity of the pytest output. Using `True` will add the
        `--verbose` option to the pytest call.
    # 设置 pytest 使用的回溯打印模式，使用 `--tb` 选项。
    tb : str, 'auto', 'long', 'short', 'line', 'native', or 'no'
        Sets the traceback print mode of pytest using the `--tb` option.
    # 只运行与给定子字符串表达式匹配的测试。表达式是一个可评估的 Python 表达式，其中所有名称都与测试名称及其父类进行子字符串匹配。
    kw : str
        Only run tests which match the given substring expression. An expression
        is a Python evaluatable expression where all names are substring-matched
        against test names and their parent classes. Example: -k 'test_method or
        test_other' matches all test functions and classes whose name contains
        'test_method' or 'test_other', while -k 'not test_method' matches those
        that don't contain 'test_method' in their names. -k 'not test_method and
        not test_other' will eliminate the matches. Additionally keywords are
        matched to classes and functions containing extra names in their
        'extra_keyword_matches' set, as well as functions which have names
        assigned directly to them. The matching is case-insensitive.
    # 在错误或 `KeyboardInterrupt` 时启动交互式 Python 调试器
    pdb : bool, default is False
        Start the interactive Python debugger on errors or `KeyboardInterrupt`.
    # 是否使用彩色终端输出
    colors : bool, default is True
        Color terminal output.
    # 按排序顺序运行测试。pytest 默认使用排序的测试顺序。需要 pytest-randomly。
    sort : bool, default is True
        Run the tests in sorted order. pytest uses a sorted test order by
        default. Requires pytest-randomly.
    # 用于随机数生成的种子。需要 pytest-randomly。
    seed : int
        Seed to use for random number generation. Requires pytest-randomly.
    # 在堆栈转储前的超时时间（秒）。0 表示无超时。需要 pytest-timeout。
    timeout : int, default is 0
        Timeout in seconds before dumping the stacks. 0 means no timeout.
        Requires pytest-timeout.
    # 超时时是否失败。该选项已不再使用。
    fail_on_timeout : bool, default is False
        Legacy option, is currently ignored.
    # 运行标记为 `slow` 的测试子集
    slow : bool, default is False
        Run the subset of tests marked as `slow`.
    # 增强断言，该选项已不再使用。
    enhance_asserts : bool, default is False
        Legacy option, is currently ignored.
    # 用于分割测试的字符串形式 `<SPLIT>/<GROUPS>` 或 None。默认为 None。
    split : string in form `<SPLIT>/<GROUPS>` or None, default is None
        Used to split the tests up. As an example, if `split='2/3' is used then
        only the middle third of tests are run. Requires pytest-split.
    # 时间平衡，该选项已不再使用。
    time_balance : bool, default is True
        Legacy option, is currently ignored.
    # 黑名单：以字符串形式表示的测试路径的可迭代对象，默认为 BLACKLIST_DEFAULT
    # 使用 `--ignore` 选项忽略黑名单中的测试路径。路径可以是部分路径或绝对路径。如果是部分路径，则会与 pytest 测试路径中的所有路径进行匹配。
    blacklist : iterable of test paths as strings, default is BLACKLIST_DEFAULT

    # 并行运行：布尔值，默认为 False
    # 使用 pytest-xdist 并行运行测试。如果为 `True`，pytest 将自动检测可用的 CPU 核心数并利用全部核心。需要安装 pytest-xdist。
    parallel : bool, default is False

    # 存储测试持续时间：布尔值，默认为 False
    # 将测试持续时间存储到文件 `.test_durations` 中。这个文件由 `pytest-split` 使用，帮助在使用多个测试组时确定更均匀的拆分。需要安装 pytest-split。
    store_durations : bool, False

    """
    # 注意：与 SymPy 测试运行器一起删除
    if kwargs.get('use_sympy_runner', False):
        # 如果 `use_sympy_runner` 参数为 True，则移除下列参数
        kwargs.pop('parallel', False)
        kwargs.pop('store_durations', False)
        kwargs.pop('use_sympy_runner', True)
        # 如果 `slow` 参数未设置，则将其设为 False
        if kwargs.get('slow') is None:
            kwargs['slow'] = False
        # 调用 test_sympy 函数，传递路径、子进程运行标志以及其他关键字参数
        return test_sympy(*paths, subprocess=True, rerun=0, **kwargs)

    # 创建 PytestPluginManager 实例
    pytest_plugin_manager = PytestPluginManager()
    # 如果没有找到 pytest，则调用 pytest.main() 运行 pytest

    # 更新参数列表 `args`，包含根目录
    args = []
    args = update_args_with_rootdir(args)

    # 如果设置了 `verbose` 参数，则添加 `--verbose` 到参数列表 `args` 中
    if kwargs.get('verbose', False):
        args.append('--verbose')

    # 如果设置了 `tb` 参数，则添加 `--tb` 和对应的值到参数列表 `args
    # 如果参数中有 'split' 关键字
    if (split := kwargs.get('split')) is not None:
        # 检查 pytest 插件管理器是否有 split 功能
        if not pytest_plugin_manager.has_split:
            msg = '`pytest-split` 插件需要用来按组运行测试。'
            raise ModuleNotFoundError(msg)
        # 使用正则表达式匹配 split 的格式
        match = split_pattern.match(split)
        # 如果匹配不成功，抛出值错误异常
        if not match:
            msg = ('split 必须是形如 a/b 的字符串，其中 a 和 b 是正的非零整数')
            raise ValueError(msg)
        # 从匹配结果中提取组号和分组数，并转换为字符串类型
        group, splits = map(str, match.groups())
        # 如果组号大于分组数，抛出值错误异常
        if group > splits:
            msg = (f'组号 {group} 不能大于仅有 {splits} 个分组')
            raise ValueError(msg)
        # 向参数列表中添加分组信息
        args.extend(['--group', group, '--splits', splits])

    # 如果参数中有 'blacklist' 关键字
    if blacklist := kwargs.get('blacklist', BLACKLIST_DEFAULT):
        # 遍历黑名单中的路径，将其转换为绝对路径并添加到参数列表中
        for path in blacklist:
            args.extend(['--ignore', make_absolute_path(path)])

    # 如果参数中有 'parallel' 关键字且为 True
    if kwargs.get('parallel', False):
        # 检查 pytest 插件管理器是否有 xdist 功能
        if not pytest_plugin_manager.has_xdist:
            msg = '`pytest-xdist` 插件需要用来并行运行测试。'
            raise ModuleNotFoundError(msg)
        # 向参数列表中添加并行测试选项
        args.extend(['-n', 'auto'])

    # 如果参数中有 'store_durations' 关键字且为 True
    if kwargs.get('store_durations', False):
        # 检查 pytest 插件管理器是否有 split 功能
        if not pytest_plugin_manager.has_split:
            msg = '`pytest-split` 插件需要用来存储测试持续时间。'
            raise ModuleNotFoundError(msg)
        # 向参数列表中添加存储持续时间选项
        args.append('--store-durations')

    # 如果参数中有 'kw' 关键字
    if (keywords := kwargs.get('kw')) is not None:
        # 将关键字转换为字符串元组
        keywords = tuple(str(kw) for kw in keywords)
    else:
        # 否则设置关键字为空元组
        keywords = ()

    # 更新参数列表中的路径和关键字
    args = update_args_with_paths(paths, keywords, args)
    # 运行 pytest 并获取退出码
    exit_code = pytest.main(args)
    # 返回退出码
    return exit_code
# 定义一个函数 doctest，用于通过 pytest 运行 doctest，与 SymPy 的测试运行器兼容
def doctest():
    # 抛出 NotImplementedError 异常，表示该函数的具体实现尚未完成
    raise NotImplementedError
```