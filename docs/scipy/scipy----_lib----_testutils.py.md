# `D:\src\scipysrc\scipy\scipy\_lib\_testutils.py`

```
# 导入必要的模块和库
"""
Generic test utilities.
"""
import inspect  # 用于检查对象的成员的工具
import os  # 提供与操作系统相关的功能
import re  # 提供正则表达式操作
import shutil  # 提供高级文件操作
import subprocess  # 允许创建新进程、连接它们的输入/输出/错误管道，并获取它们的返回代码
import sys  # 提供与 Python 解释器及其环境相关的功能
import sysconfig  # 提供与当前 Python 解释器配置相关的功能
from importlib.util import module_from_spec, spec_from_file_location  # 提供动态导入模块的工具

import numpy as np  # 数值计算库
import scipy  # 科学计算库

try:
    # 需要 type: ignore[import-untyped] 来忽略类型检查
    import cython  # type: ignore[import-untyped]
    from Cython.Compiler.Version import (  # type: ignore[import-untyped]
        version as cython_version,
    )
except ImportError:
    cython = None
else:
    from scipy._lib import _pep440
    required_version = '3.0.8'
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # 过旧或错误的 Cython 版本，跳过 Cython API 测试
        cython = None

__all__ = ['PytestTester', 'check_free_memory', '_TestPythranFunc', 'IS_MUSL']

IS_MUSL = False
# 从 sysconfig 获取主机 GNU 类型的配置变量，并检查是否包含 'musl'
_v = sysconfig.get_config_var('HOST_GNU_TYPE') or ''
if 'musl' in _v:
    IS_MUSL = True

IS_EDITABLE = 'editable' in scipy.__path__[0]

class FPUModeChangeWarning(RuntimeWarning):
    """Warning about FPU mode change"""
    pass

class PytestTester:
    """
    Run tests for this namespace

    ``scipy.test()`` runs tests for all of SciPy, with the default settings.
    When used from a submodule (e.g., ``scipy.cluster.test()``, only the tests
    for that namespace are run.

    Parameters
    ----------
    label : {'fast', 'full'}, optional
        Whether to run only the fast tests, or also those marked as slow.
        Default is 'fast'.
    verbose : int, optional
        Test output verbosity. Default is 1.
    extra_argv : list, optional
        Arguments to pass through to Pytest.
    doctests : bool, optional
        Whether to run doctests or not. Default is False.
    coverage : bool, optional
        Whether to run tests with code coverage measurements enabled.
        Default is False.
    tests : list of str, optional
        List of module names to run tests for. By default, uses the module
        from which the ``test`` function is called.
    parallel : int, optional
        Run tests in parallel with pytest-xdist, if number given is larger than
        1. Default is 1.
    """
    
    def __init__(self, module_name):
        self.module_name = module_name
    # 定义一个方法，使对象可以像函数一样被调用，运行 pytest 测试
    def __call__(self, label="fast", verbose=1, extra_argv=None, doctests=False,
                 coverage=False, tests=None, parallel=None):
        # 导入 pytest 模块
        import pytest

        # 获取模块对象
        module = sys.modules[self.module_name]
        # 获取模块的绝对路径
        module_path = os.path.abspath(module.__path__[0])

        # 初始化 pytest 参数列表，设置一些常用的选项
        pytest_args = ['--showlocals', '--tb=short']

        # 如果有额外的参数，将其添加到 pytest 参数列表中
        if extra_argv:
            pytest_args += list(extra_argv)

        # 根据 verbose 参数设置 pytest 的详细输出级别
        if verbose and int(verbose) > 1:
            pytest_args += ["-" + "v"*(int(verbose)-1)]

        # 如果启用了 coverage 选项，设置 pytest 来计算代码覆盖率
        if coverage:
            pytest_args += ["--cov=" + module_path]

        # 根据 label 参数设置 pytest 执行的标记条件
        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]

        # 如果 tests 为空，则默认测试当前模块
        if tests is None:
            tests = [self.module_name]

        # 如果启用了并行测试，并且有多个并行进程
        if parallel is not None and parallel > 1:
            # 检查是否安装了 pytest-xdist 插件，若安装则使用多进程执行测试
            if _pytest_has_xdist():
                pytest_args += ['-n', str(parallel)]
            else:
                # 若未安装 pytest-xdist 插件，给出警告信息
                import warnings
                warnings.warn('Could not run tests in parallel because '
                              'pytest-xdist plugin is not available.',
                              stacklevel=2)

        # 添加 --pyargs 标志以指示 pytest 查找 Python 包中的模块
        pytest_args += ['--pyargs'] + list(tests)

        try:
            # 运行 pytest 并捕获其退出状态码
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            # 如果 pytest 抛出 SystemExit 异常，获取其退出状态码
            code = exc.code

        # 返回 pytest 的运行结果，成功返回 True（0），失败返回 False（非0）
        return (code == 0)
class _TestPythranFunc:
    '''
    These are situations that can be tested in our pythran tests:
    - A function with multiple array arguments and then
      other positional and keyword arguments.
    - A function with array-like keywords (e.g. `def somefunc(x0, x1=None)`.
    Note: list/tuple input is not yet tested!

    `self.arguments`: A dictionary which key is the index of the argument,
                      value is tuple(array value, all supported dtypes)
    `self.partialfunc`: A function used to freeze some non-array argument
                        that of no interests in the original function
    '''

    ALL_INTEGER = [np.int8, np.int16, np.int32, np.int64, np.intc, np.intp]
    ALL_FLOAT = [np.float32, np.float64]
    ALL_COMPLEX = [np.complex64, np.complex128]

    def setup_method(self):
        # Initialize an empty dictionary to store arguments
        self.arguments = {}
        # Initialize a variable to store a partial function (not used initially)
        self.partialfunc = None
        # Initialize a variable to store expected result (not used initially)
        self.expected = None

    def get_optional_args(self, func):
        # Retrieve optional arguments with their default values from the function signature
        signature = inspect.signature(func)
        optional_args = {}
        for k, v in signature.parameters.items():
            if v.default is not inspect.Parameter.empty:
                optional_args[k] = v.default
        return optional_args

    def get_max_dtype_list_length(self):
        # Determine the maximum length of supported dtypes lists among all arguments
        max_len = 0
        for arg_idx in self.arguments:
            cur_len = len(self.arguments[arg_idx][1])
            if cur_len > max_len:
                max_len = cur_len
        return max_len

    def get_dtype(self, dtype_list, dtype_idx):
        # Retrieve the dtype from dtype_list based on index
        # If index is out of range, return the last dtype in the list
        if dtype_idx > len(dtype_list)-1:
            return dtype_list[-1]
        else:
            return dtype_list[dtype_idx]

    def test_all_dtypes(self):
        # Test the function with all combinations of dtypes for arguments
        for type_idx in range(self.get_max_dtype_list_length()):
            args_array = []
            for arg_idx in self.arguments:
                new_dtype = self.get_dtype(self.arguments[arg_idx][1],
                                           type_idx)
                args_array.append(self.arguments[arg_idx][0].astype(new_dtype))
            self.pythranfunc(*args_array)

    def test_views(self):
        # Test the function with arguments that are views (reversed arrays)
        args_array = []
        for arg_idx in self.arguments:
            args_array.append(self.arguments[arg_idx][0][::-1][::-1])
        self.pythranfunc(*args_array)

    def test_strided(self):
        # Test the function with arguments that are strided arrays
        args_array = []
        for arg_idx in self.arguments:
            args_array.append(np.repeat(self.arguments[arg_idx][0],
                                        2, axis=0)[::2])
        self.pythranfunc(*args_array)


def _pytest_has_xdist():
    """
    Check if the pytest-xdist plugin is installed, providing parallel tests
    """
    # Check if pytest-xdist plugin is installed without importing it
    # This prevents pytest from emitting warnings related to imports
    # 导入模块 importlib.util 中的 find_spec 函数，用于查找指定模块是否可用
    from importlib.util import find_spec
    # 使用 find_spec 函数查找名为 'xdist' 的模块，检查其是否存在
    return find_spec('xdist') is not None
# 检查可用内存是否足够，否则跳过测试
def check_free_memory(free_mb):
    import pytest  # 导入 pytest 模块

    try:
        # 尝试解析环境变量 SCIPY_AVAILABLE_MEM 的值作为可用内存大小
        mem_free = _parse_size(os.environ['SCIPY_AVAILABLE_MEM'])
        # 创建消息，指示所需的内存大小及实际环境变量的值
        msg = '{} MB memory required, but environment SCIPY_AVAILABLE_MEM={}'.format(
            free_mb, os.environ['SCIPY_AVAILABLE_MEM'])
    except KeyError:
        # 如果环境变量未设置，则获取当前可用内存大小
        mem_free = _get_mem_available()
        if mem_free is None:
            # 如果无法获取可用内存大小，则跳过测试，并给出相应的提示信息
            pytest.skip("Could not determine available memory; set SCIPY_AVAILABLE_MEM "
                        "variable to free memory in MB to run the test.")
        # 创建消息，指示所需的内存大小及实际可用内存大小
        msg = f'{free_mb} MB memory required, but {mem_free/1e6} MB available'

    # 如果可用内存小于所需内存的兆字节表示，则跳过测试，并附带相应的消息
    if mem_free < free_mb * 1e6:
        pytest.skip(msg)


# 解析给定大小字符串，返回以字节为单位的大小
def _parse_size(size_str):
    suffixes = {'': 1e6,
                'b': 1.0,
                'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
                'kb': 1e3, 'Mb': 1e6, 'Gb': 1e9, 'Tb': 1e12,
                'kib': 1024.0, 'Mib': 1024.0**2, 'Gib': 1024.0**3, 'Tib': 1024.0**4}
    # 匹配给定大小字符串的大小及其单位，返回字节表示的大小
    m = re.match(r'^\s*(\d+)\s*({})\s*$'.format('|'.join(suffixes.keys())),
                 size_str,
                 re.I)
    if not m or m.group(2) not in suffixes:
        # 如果匹配失败或者单位不在预定义的单位列表中，则抛出值错误异常
        raise ValueError("Invalid size string")

    return float(m.group(1)) * suffixes[m.group(2)]


# 获取可用内存信息（不包括交换空间）
def _get_mem_available():
    try:
        import psutil
        # 使用 psutil 获取虚拟内存的可用大小
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass

    if sys.platform.startswith('linux'):
        # 如果运行在 Linux 平台上，则尝试从 /proc/meminfo 文件获取内存信息
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                p = line.split()
                info[p[0].strip(':').lower()] = float(p[1]) * 1e3

        if 'memavailable' in info:
            # 如果支持 memavailable 字段，则返回其值（适用于 Linux >= 3.14）
            return info['memavailable']
        else:
            # 否则返回可用内存和缓存的总和
            return info['memfree'] + info['cached']

    return None


# 辅助函数，用于测试构建和导入使用 Cython APIs 的 Cython 模块
def _test_cython_extension(tmp_path, srcdir):
    import pytest  # 导入 pytest 模块
    try:
        # 检查是否能找到可用的 'meson' 构建工具
        subprocess.check_call(["meson", "--version"])
    except FileNotFoundError:
        # 如果找不到 'meson'，则跳过测试，并给出相应的提示信息
        pytest.skip("No usable 'meson' found")

    # 复制源代码目录到临时目录下的模块名称目录中
    mod_name = os.path.split(srcdir)[1]
    shutil.copytree(srcdir, tmp_path / mod_name)
    # 设置构建目录和目标目录的路径
    build_dir = tmp_path / mod_name / 'tests' / '_cython_examples'
    target_dir = build_dir / 'build'
    os.makedirs(target_dir, exist_ok=True)

    # 确保使用正确的 Python 解释器，即使 'meson' 安装在不同的 Python 环境中
    native_file = str(build_dir / 'interpreter-native-file.ini')
    with open(native_file, 'w') as f:
        f.write("[binaries]\n")
        f.write(f"python = '{sys.executable}'")
    # 如果操作系统为 Windows
    if sys.platform == "win32":
        # 在指定目录下使用 subprocess 启动 meson 的 setup 命令，配置构建类型为 release，
        # 使用本地文件和构建目录作为参数，使用 Visual Studio 环境
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release",
                               "--native-file", native_file,
                               "--vsenv", str(build_dir)],
                              cwd=target_dir,
                              )
    else:
        # 在指定目录下使用 subprocess 启动 meson 的 setup 命令，使用本地文件和构建目录作为参数
        subprocess.check_call(["meson", "setup",
                               "--native-file", native_file, str(build_dir)],
                              cwd=target_dir
                              )

    # 在指定目录下使用 subprocess 编译 meson 项目，输出详细编译信息
    subprocess.check_call(["meson", "compile", "-vv"], cwd=target_dir)

    # 使用 sysconfig 获取当前平台的扩展模块后缀名
    suffix = sysconfig.get_config_var('EXT_SUFFIX')

    # 定义一个函数 load，用于动态加载指定名称的模块
    def load(modname):
        # 构建扩展模块的路径，加上平台特定的扩展名
        so = (target_dir / modname).with_suffix(suffix)
        # 根据文件路径创建模块规范
        spec = spec_from_file_location(modname, so)
        # 根据规范加载模块
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # 测试扩展模块是否可以成功导入
    return load("extending"), load("extending_cpp")
```