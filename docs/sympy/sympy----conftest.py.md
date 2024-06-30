# `D:\src\scipysrc\sympy\sympy\conftest.py`

```
import sys

sys._running_pytest = True  # 设置一个全局变量，用于指示正在运行 pytest 测试
from sympy.external.importtools import version_tuple  # 导入版本元组工具

import pytest  # 导入 pytest 测试框架
from sympy.core.cache import clear_cache, USE_CACHE  # 导入清除缓存函数和缓存使用标志
from sympy.external.gmpy import GROUND_TYPES  # 导入 GMPY 外部库的地面类型
from sympy.utilities.misc import ARCH  # 导入系统架构信息
import re  # 导入正则表达式模块

try:
    import hypothesis  # 尝试导入 hypothesis 库

    # 注册和加载 hypothesis 的测试配置文件
    hypothesis.settings.register_profile("sympy_hypothesis_profile", deadline=None)
    hypothesis.settings.load_profile("sympy_hypothesis_profile")
except ImportError:
    # 如果导入失败，则抛出 ImportError 异常并提供安装提示信息
    raise ImportError(
        "hypothesis is a required dependency to run the SymPy test suite. "
        "Install it with 'pip install hypothesis' or 'conda install -c conda-forge hypothesis'"
    )


# 定义一个正则表达式对象，用于匹配数字/数字格式的字符串
sp = re.compile(r"([0-9]+)/([1-9][0-9]*)")


def process_split(config, items):
    # 根据 pytest 的命令行选项 --split 处理测试项列表的分割
    split = config.getoption("--split")
    if not split:
        return  # 如果没有指定 --split 选项，则直接返回

    m = sp.match(split)  # 使用正则表达式匹配 --split 的字符串格式
    if not m:
        # 如果匹配失败，则抛出 ValueError 异常
        raise ValueError(
            "split must be a string of the form a/b " "where a and b are ints."
        )

    i, t = map(int, m.groups())  # 解析匹配结果为整数 i 和 t
    start, end = (i - 1) * len(items) // t, i * len(items) // t

    if i < t:
        # 如果当前索引 i 小于总数 t，则删除列表末尾的元素
        del items[end:]
    del items[:start]  # 删除列表起始处到指定索引 start 的元素


def pytest_report_header(config):
    # 生成用于 pytest 报告头部的信息字符串
    s = "architecture: %s\n" % ARCH  # 添加系统架构信息
    s += "cache:        %s\n" % USE_CACHE  # 添加缓存使用信息
    version = ""
    if GROUND_TYPES == "gmpy":
        import gmpy2
        version = gmpy2.version()  # 如果使用 gmpy，则添加 gmpy2 的版本信息
    elif GROUND_TYPES == "flint":
        try:
            from flint import __version__
        except ImportError:
            version = "unknown"  # 如果使用 flint 但导入失败，则版本信息为 unknown
        else:
            version = f'(python-flint=={__version__})'  # 否则添加 python-flint 的版本信息
    s += "ground types: %s %s\n" % (GROUND_TYPES, version)  # 添加地面类型和版本信息
    return s  # 返回生成的报告头部信息字符串


def pytest_terminal_summary(terminalreporter):
    # 在 pytest 执行结束后生成终端摘要信息
    if terminalreporter.stats.get("error", None) or terminalreporter.stats.get(
        "failed", None
    ):
        # 如果有错误或失败的测试结果，则输出警告信息
        terminalreporter.write_sep(" ", "DO *NOT* COMMIT!", red=True, bold=True)


def pytest_addoption(parser):
    # 添加一个命令行选项 --split，用于分割测试
    parser.addoption("--split", action="store", default="", help="split tests")


def pytest_collection_modifyitems(config, items):
    """pytest hook."""
    # 修改测试项集合，根据配置和命令行选项处理分割
    process_split(config, items)


@pytest.fixture(autouse=True, scope="module")
def file_clear_cache():
    # 每个模块级别的自动使用 fixture，用于清除缓存
    clear_cache()


@pytest.fixture(autouse=True, scope="module")
def check_disabled(request):
    # 检查是否需要跳过测试
    if getattr(request.module, "disabled", False):
        pytest.skip("test requirements not met.")  # 如果模块被禁用，则跳过测试
    elif getattr(request.module, "ipython", False):
        # 针对 ipython 测试，需要检查版本和选项
        if (
            version_tuple(pytest.__version__) < version_tuple("2.6.3")
            and pytest.config.getvalue("-s") != "no"
        ):
            pytest.skip("run py.test with -s or upgrade to newer version.")  # 如果条件不满足，则跳过测试
```