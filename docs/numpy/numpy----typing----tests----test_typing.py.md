# `.\numpy\numpy\typing\tests\test_typing.py`

```py
# 从未来模块导入 annotations 特性，用于支持类型注解
from __future__ import annotations

# 导入标准库中的模块
import importlib.util  # 导入动态加载模块的实用工具
import os  # 导入操作系统功能的模块
import re  # 导入正则表达式模块
import shutil  # 导入高级文件操作模块
from collections import defaultdict  # 导入默认字典容器
from collections.abc import Iterator  # 导入迭代器抽象基类
from typing import TYPE_CHECKING  # 导入类型提示中的 TYPE_CHECKING 特性

# 导入第三方库和模块
import pytest  # 导入用于编写测试的 pytest
from numpy.typing.mypy_plugin import _EXTENDED_PRECISION_LIST  # 导入特定的 numpy 类型提示

# 环境变量检查，确定是否需要在测试套件中运行完整的 mypy 检查
RUN_MYPY = "NPY_RUN_MYPY_IN_TESTSUITE" in os.environ
if RUN_MYPY and RUN_MYPY not in ('0', '', 'false'):
    RUN_MYPY = True

# 标记当前文件中的所有函数为 pytest 测试的跳过状态，条件是不在指定的环境变量设置下运行 mypy
pytestmark = pytest.mark.skipif(
    not RUN_MYPY,
    reason="`NPY_RUN_MYPY_IN_TESTSUITE` not set"
)

try:
    # 尝试导入 mypy 的 API 模块
    from mypy import api
except ImportError:
    NO_MYPY = True  # 如果导入失败，标记为无 mypy 环境
else:
    NO_MYPY = False  # 否则标记为有 mypy 环境

if TYPE_CHECKING:
    # 如果在类型检查模式下
    # 需要这个作为注解，但它位于私有命名空间中
    # 作为一种折中，不在运行时导入它
    from _pytest.mark.structures import ParameterSet

# 指定数据文件夹路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# 定义不同类型的测试数据子文件夹路径
PASS_DIR = os.path.join(DATA_DIR, "pass")
FAIL_DIR = os.path.join(DATA_DIR, "fail")
REVEAL_DIR = os.path.join(DATA_DIR, "reveal")
MISC_DIR = os.path.join(DATA_DIR, "misc")
# 指定 mypy 配置文件路径
MYPY_INI = os.path.join(DATA_DIR, "mypy.ini")
# 指定 mypy 缓存文件夹路径
CACHE_DIR = os.path.join(DATA_DIR, ".mypy_cache")

# 用于存储 mypy 输出的字典，键为文件名，值为 mypy 的标准输出列表
OUTPUT_MYPY: defaultdict[str, list[str]] = defaultdict(list)

def _key_func(key: str) -> str:
    """根据第一个出现的 ':' 字符分割键。

    Windows 下的驱动器号（例如 'C:'）会被忽略。
    """
    drive, tail = os.path.splitdrive(key)
    return os.path.join(drive, tail.split(":", 1)[0])

def _strip_filename(msg: str) -> tuple[int, str]:
    """从 mypy 消息中剥离文件名和行号。

    返回一个包含行号和剥离后消息的元组。
    """
    _, tail = os.path.splitdrive(msg)
    _, lineno, msg = tail.split(":", 2)
    return int(lineno), msg.strip()

def strip_func(match: re.Match[str]) -> str:
    """用于 re.sub 的辅助函数，用于剥离模块名。

    返回匹配组的第二个元素。
    """

@pytest.fixture(scope="module", autouse=True)
def run_mypy() -> None:
    """在运行任何类型检查测试之前清除缓存并运行 mypy。

    mypy 的结果会缓存到 OUTPUT_MYPY 中以便后续使用。

    可以通过设置环境变量 NUMPY_TYPING_TEST_CLEAR_CACHE=0 来跳过缓存刷新。

    """
    if (
        os.path.isdir(CACHE_DIR)
        and bool(os.environ.get("NUMPY_TYPING_TEST_CLEAR_CACHE", True))
    ):
        shutil.rmtree(CACHE_DIR)

    split_pattern = re.compile(r"(\s+)?\^(\~+)?")
    # 遍历指定的目录列表，依次处理每个目录
    for directory in (PASS_DIR, REVEAL_DIR, FAIL_DIR, MISC_DIR):
        # 运行 mypy 静态类型检查工具
        stdout, stderr, exit_code = api.run([
            "--config-file",  # 指定配置文件路径
            MYPY_INI,
            "--cache-dir",    # 指定缓存目录路径
            CACHE_DIR,
            directory,        # 当前处理的目录路径
        ])
        # 检查是否有标准错误输出，如果有则抛出测试失败异常
        if stderr:
            pytest.fail(f"Unexpected mypy standard error\n\n{stderr}")
        # 检查 mypy 的退出码是否为 0 或 1，否则抛出测试失败异常
        elif exit_code not in {0, 1}:
            pytest.fail(f"Unexpected mypy exit code: {exit_code}\n\n{stdout}")

        # 初始化一个空字符串用于存储每行输出内容的累加
        str_concat = ""
        # 初始化一个变量用于存储当前处理的文件名，初始值为 None
        filename: str | None = None
        # 遍历 mypy 输出的每一行
        for i in stdout.split("\n"):
            # 如果当前行包含 "note:"，则跳过处理
            if "note:" in i:
                continue
            # 如果当前文件名尚未设置，则调用 _key_func 函数获取文件名
            if filename is None:
                filename = _key_func(i)

            # 将当前行内容添加到 str_concat 中
            str_concat += f"{i}\n"
            # 如果当前行符合分隔模式，将累积的输出添加到 OUTPUT_MYPY 中对应的文件名下
            if split_pattern.match(i) is not None:
                OUTPUT_MYPY[filename].append(str_concat)
                # 重置 str_concat 和 filename，准备处理下一个文件的输出
                str_concat = ""
                filename = None
# 从指定目录递归遍历文件和子目录
def get_test_cases(directory: str) -> Iterator[ParameterSet]:
    for root, _, files in os.walk(directory):
        for fname in files:
            # 分离文件名和扩展名
            short_fname, ext = os.path.splitext(fname)
            # 如果文件扩展名是 .pyi 或者 .py
            if ext in (".pyi", ".py"):
                # 构建文件的完整路径
                fullpath = os.path.join(root, fname)
                # 使用 pytest.param 创建测试参数，以文件名作为测试用例的 id
                yield pytest.param(fullpath, id=short_fname)


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR))
def test_success(path) -> None:
    # 别名 OUTPUT_MYPY，使其在本地命名空间中可见
    output_mypy = OUTPUT_MYPY
    # 如果路径在 output_mypy 中
    if path in output_mypy:
        # 构建错误消息
        msg = "Unexpected mypy output\n\n"
        # 添加每个错误行到消息中
        msg += "\n".join(_strip_filename(v)[1] for v in output_mypy[path])
        # 抛出断言错误，显示错误消息
        raise AssertionError(msg)


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(FAIL_DIR))
def test_fail(path: str) -> None:
    # 隐藏 traceback 信息
    __tracebackhide__ = True

    # 打开文件，并按行读取内容
    with open(path) as fin:
        lines = fin.readlines()

    # 创建默认字典，用于存储错误信息
    errors = defaultdict(lambda: "")

    # 别名 OUTPUT_MYPY，使其在本地命名空间中可见
    output_mypy = OUTPUT_MYPY
    # 断言路径在 output_mypy 中
    assert path in output_mypy

    # 遍历 output_mypy 中的每个错误行
    for error_line in output_mypy[path]:
        # 获取错误行号和错误内容
        lineno, error_line = _strip_filename(error_line)
        # 将错误内容添加到对应行号的错误字典中
        errors[lineno] += f'{error_line}\n'

    # 遍历文件的每一行
    for i, line in enumerate(lines):
        # 获取行号
        lineno = i + 1
        # 如果行以 '#' 开头或者（不包含 ' E:' 且行号不在错误字典中）
        if (
            line.startswith('#')
            or (" E:" not in line and lineno not in errors)
        ):
            continue

        # 获取目标行内容
        target_line = lines[lineno - 1]
        # 如果目标行包含 "# E:"
        if "# E:" in target_line:
            # 分离表达式、期望错误和标记
            expression, _, marker = target_line.partition("  # E: ")
            # 获取期望的错误信息
            expected_error = errors[lineno].strip()
            marker = marker.strip()
            # 调用 _test_fail 函数进行测试失败断言
            _test_fail(path, expression, marker, expected_error, lineno)
        else:
            # 抛出断言错误，显示未预期的 mypy 输出
            pytest.fail(
                f"Unexpected mypy output at line {lineno}\n\n{errors[lineno]}"
            )


_FAIL_MSG1 = """Extra error at line {}

Expression: {}
Extra error: {!r}
"""

_FAIL_MSG2 = """Error mismatch at line {}

Expression: {}
Expected error: {}
Observed error: {!r}
"""


def _test_fail(
    path: str,
    expression: str,
    error: str,
    expected_error: None | str,
    lineno: int,
) -> None:
    # 如果期望的错误信息为 None
    if expected_error is None:
        # 抛出断言错误，显示额外错误信息
        raise AssertionError(_FAIL_MSG1.format(lineno, expression, error))
    # 如果错误不在期望的错误信息中
    elif error not in expected_error:
        # 抛出断言错误，显示错误不匹配信息
        raise AssertionError(_FAIL_MSG2.format(
            lineno, expression, expected_error, error
        ))


_REVEAL_MSG = """Reveal mismatch at line {}

{}
"""


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(REVEAL_DIR))
def test_reveal(path: str) -> None:
    """Validate that mypy correctly infers the return-types of
    the expressions in `path`.
    """
    # 隐藏 traceback 信息
    __tracebackhide__ = True

    # 别名 OUTPUT_MYPY，使其在本地命名空间中可见
    output_mypy = OUTPUT_MYPY
    # 如果路径不在 output_mypy 中，则返回
    if path not in output_mypy:
        return
    # 遍历 output_mypy[path] 中的每一行错误信息
    for error_line in output_mypy[path]:
        # 调用 _strip_filename 函数，解析出行号和错误行内容
        lineno, error_line = _strip_filename(error_line)
        # 抛出断言错误，格式化错误消息，包含行号和错误行内容
        raise AssertionError(_REVEAL_MSG.format(lineno, error_line))
# 标记该测试为慢速测试，运行时若未安装 Mypy 则跳过
@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
# 使用参数化测试，使用 PASS_DIR 中的测试用例路径进行参数化
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR))
def test_code_runs(path: str) -> None:
    """验证 `path` 中的代码在运行时是否正常执行。"""
    # 获取文件路径去除扩展名后的部分
    path_without_extension, _ = os.path.splitext(path)
    # 分割路径，获取目录名和文件名
    dirname, filename = path.split(os.sep)[-2:]

    # 根据文件路径创建一个导入规范对象
    spec = importlib.util.spec_from_file_location(
        f"{dirname}.{filename}", path
    )
    # 断言规范对象不为空
    assert spec is not None
    # 断言规范对象的加载器不为空

    assert spec.loader is not None

    # 使用规范对象的加载器执行模块
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)


# 行号与扩展精度类型的映射关系
LINENO_MAPPING = {
    11: "uint128",
    12: "uint256",
    14: "int128",
    15: "int256",
    17: "float80",
    18: "float96",
    19: "float128",
    20: "float256",
    22: "complex160",
    23: "complex192",
    24: "complex256",
    25: "complex512",
}


# 标记该测试为慢速测试，运行时若未安装 Mypy 则跳过
@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
def test_extended_precision() -> None:
    # 设置扩展精度文件的路径
    path = os.path.join(MISC_DIR, "extended_precision.pyi")
    # 确认 Mypy 输出中包含该路径
    output_mypy = OUTPUT_MYPY
    assert path in output_mypy

    # 打开扩展精度文件，读取所有行到表达式列表中
    with open(path) as f:
        expression_list = f.readlines()

    # 遍历 Mypy 输出中的每一条消息
    for _msg in output_mypy[path]:
        # 解析出行号和消息
        lineno, msg = _strip_filename(_msg)
        # 获取指定行号的表达式，并去除末尾的换行符
        expression = expression_list[lineno - 1].rstrip("\n")

        # 如果该行号在扩展精度类型映射中存在
        if LINENO_MAPPING[lineno] in _EXTENDED_PRECISION_LIST:
            # 抛出断言错误，显示消息和行号
            raise AssertionError(_REVEAL_MSG.format(lineno, msg))
        # 否则，如果消息中不包含错误关键词
        elif "error" not in msg:
            # 调用 _test_fail 函数，传递文件路径、表达式、消息、'Expression is of type "Any"' 以及行号
            _test_fail(
                path, expression, msg, 'Expression is of type "Any"', lineno
            )
```