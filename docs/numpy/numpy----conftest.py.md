# `.\numpy\numpy\conftest.py`

```py
"""
Pytest configuration and fixtures for the Numpy test suite.
"""
# 导入必要的库和模块
import os  # 导入操作系统接口
import tempfile  # 导入临时文件和目录创建的模块
from contextlib import contextmanager  # 导入上下文管理器
import warnings  # 导入警告模块

import hypothesis  # 导入假设测试库
import pytest  # 导入pytest测试框架
import numpy  # 导入NumPy库

from numpy._core._multiarray_tests import get_fpu_mode  # 导入获取FPU模式的函数

# 尝试导入scipy_doctest.conftest模块，标记是否成功导入
try:
    from scipy_doctest.conftest import dt_config
    HAVE_SCPDT = True
except ModuleNotFoundError:
    HAVE_SCPDT = False

# 初始化旧的FPU模式和结果收集字典
_old_fpu_mode = None
_collect_results = {}

# 设置hypothesis缓存的主目录，使用已知且持久的临时目录
hypothesis.configuration.set_hypothesis_home_dir(
    os.path.join(tempfile.gettempdir(), ".hypothesis")
)

# 注册两个自定义的Hypothesis配置文件，用于NumPy测试
hypothesis.settings.register_profile(
    name="numpy-profile", deadline=None, print_blob=True,
)
hypothesis.settings.register_profile(
    name="np.test() profile",
    deadline=None, print_blob=True, database=None, derandomize=True,
    suppress_health_check=list(hypothesis.HealthCheck),
)

# 根据pytest.ini文件的存在与否加载默认的Hypothesis配置文件
_pytest_ini = os.path.join(os.path.dirname(__file__), "..", "pytest.ini")
hypothesis.settings.load_profile(
    "numpy-profile" if os.path.isfile(_pytest_ini) else "np.test() profile"
)

# 设置NUMPY_EXPERIMENTAL_DTYPE_API环境变量为1，用于_umath_tests
os.environ["NUMPY_EXPERIMENTAL_DTYPE_API"] = "1"

# 定义pytest的配置函数，添加自定义标记
def pytest_configure(config):
    config.addinivalue_line("markers",
        "valgrind_error: Tests that are known to error under valgrind.")
    config.addinivalue_line("markers",
        "leaks_references: Tests that are known to leak references.")
    config.addinivalue_line("markers",
        "slow: Tests that are very slow.")
    config.addinivalue_line("markers",
        "slow_pypy: Tests that are very slow on pypy.")

# 定义pytest的命令行选项，用于设置可用内存量
def pytest_addoption(parser):
    parser.addoption("--available-memory", action="store", default=None,
                     help=("Set amount of memory available for running the "
                           "test suite. This can result to tests requiring "
                           "especially large amounts of memory to be skipped. "
                           "Equivalent to setting environment variable "
                           "NPY_AVAILABLE_MEM. Default: determined"
                           "automatically."))

# 在测试会话开始时，根据命令行选项设置环境变量NPY_AVAILABLE_MEM
def pytest_sessionstart(session):
    available_mem = session.config.getoption('available_memory')
    if available_mem is not None:
        os.environ['NPY_AVAILABLE_MEM'] = available_mem

# TODO: 移除yield测试后修复此函数
@pytest.hookimpl()
def pytest_itemcollected(item):
    """
    Check FPU precision mode was not changed during test collection.
    """
    The clumsy way we do it here is mainly necessary because numpy
    still uses yield tests, which can execute code at test collection
    time.
    """
    # 声明全局变量 _old_fpu_mode，用于存储旧的浮点数处理单元模式
    global _old_fpu_mode

    # 获取当前的浮点数处理单元模式
    mode = get_fpu_mode()

    # 如果 _old_fpu_mode 还未设置，则将当前模式赋给它
    if _old_fpu_mode is None:
        _old_fpu_mode = mode
    # 否则，如果当前模式与旧模式不同，则记录结果到 _collect_results 字典中，并更新 _old_fpu_mode
    elif mode != _old_fpu_mode:
        _collect_results[item] = (_old_fpu_mode, mode)
        _old_fpu_mode = mode
# 如果 HAVE_SCPDT 可用，则定义一个上下文管理器 warnings_errors_and_rng
if HAVE_SCPDT:
    @contextmanager
    def warnings_errors_and_rng(test=None):
        """Filter out the wall of DeprecationWarnings.
        """
        # 定义需要忽略的 DeprecationWarning 的消息列表
        msgs = ["The numpy.linalg.linalg",
                "The numpy.fft.helper",
                "dep_util",
                "pkg_resources",
                "numpy.core.umath",
                "msvccompiler",
                "Deprecated call",
                "numpy.core",
                "`np.compat`",
                "Importing from numpy.matlib",
                "This function is deprecated.",    # random_integers
                "Data type alias 'a'",     # numpy.rec.fromfile
                "Arrays of 2-dimensional vectors",   # matlib.cross
                "`in1d` is deprecated", ]
        msg = "|".join(msgs)

        # 定义需要忽略的 RuntimeWarning 的消息列表
        msgs_r = [
            "invalid value encountered",
            "divide by zero encountered"
        ]
        msg_r = "|".join(msgs_r)

        # 忽略特定类型的警告消息
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=DeprecationWarning, message=msg
            )
            warnings.filterwarnings(
                'ignore', category=RuntimeWarning, message=msg_r
            )
            yield

    # 将定义好的上下文管理器应用于用户配置的上下文管理器
    dt_config.user_context_mgr = warnings_errors_and_rng

    # 为 doctests 添加特定于 numpy 的标记以处理未初始化情况
    dt_config.rndm_markers.add('#uninitialized')
    dt_config.rndm_markers.add('# uninitialized')

    # 导入 doctest 模块，用于查找和检查此上下文管理器下的文档测试
    import doctest
    # 设置 doctest 的选项标志，用于规范化空白和省略号处理
    dt_config.optionflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

    # 将 'StringDType' 识别为 numpy.dtypes.StringDType 的命名空间检查
    dt_config.check_namespace['StringDType'] = numpy.dtypes.StringDType

    # 设置临时跳过列表，避免在 doctest 中处理以下函数
    dt_config.skiplist = set([
        'numpy.savez',    # 文件未关闭
        'numpy.matlib.savez',
        'numpy.__array_namespace_info__',
        'numpy.matlib.__array_namespace_info__',
    ])

    # 标记无法通过测试的教程文件为 xfail（预期失败），附加信息为空字符串
    dt_config.pytest_extra_xfail = {
        'how-to-verify-bug.rst': '',
        'c-info.ufunc-tutorial.rst': '',
        'basics.interoperability.rst': 'needs pandas',  # 需要 pandas
        'basics.dispatch.rst': 'errors out in /testing/overrides.py',  # 在 /testing/overrides.py 中出错
        'basics.subclassing.rst': '.. testcode:: admonitions not understood'  # 不理解警告
    }

    # 设置额外的忽略列表，用于不希望进行 doctest 集合的内容（例如可选内容）
    dt_config.pytest_extra_ignore = [
        'numpy/distutils',
        'numpy/_core/cversions.py',
        'numpy/_pyinstaller',
        'numpy/random/_examples',
        'numpy/compat',
        'numpy/f2py/_backends/_distutils.py',
    ]
```