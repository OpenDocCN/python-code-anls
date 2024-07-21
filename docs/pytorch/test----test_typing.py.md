# `.\pytorch\test\test_typing.py`

```py
# 导入所需的模块和库
import itertools  # 提供迭代工具的函数
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式操作的支持
import shutil  # 提供高级文件操作功能

import unittest  # 提供单元测试框架
from collections import defaultdict  # 提供默认字典功能
from threading import Lock  # 提供线程同步的锁对象
from typing import Dict, IO, List, Optional  # 提供类型提示相关的功能

from torch.testing._internal.common_utils import (  # 导入 Torch 测试相关的工具函数
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

try:
    from mypy import api  # 尝试导入 mypy 的 API 模块
except ImportError:
    NO_MYPY = True  # 如果导入失败，设置标志位 NO_MYPY 为 True
else:
    NO_MYPY = False  # 如果成功导入，设置标志位 NO_MYPY 为 False


# 定义各种目录和文件路径常量
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "typing"))
REVEAL_DIR = os.path.join(DATA_DIR, "reveal")  # 解析类型提示的目录
PASS_DIR = os.path.join(DATA_DIR, "pass")  # 包含类型提示通过测试的目录
FAIL_DIR = os.path.join(DATA_DIR, "fail")  # 包含类型提示未通过测试的目录
MYPY_INI = os.path.join(DATA_DIR, os.pardir, os.pardir, "mypy.ini")  # mypy 的配置文件路径
CACHE_DIR = os.path.join(DATA_DIR, ".mypy_cache")  # mypy 的缓存目录路径


def _key_func(key: str) -> str:
    """根据第一个出现的 ':' 字符拆分键名。

    在此函数中忽略 Windows 系统下的驱动器盘符（例如 'C:'）。
    """
    drive, tail = os.path.splitdrive(key)  # 分离驱动器盘符
    return os.path.join(drive, tail.split(":", 1)[0])  # 返回拆分后的路径


def _strip_filename(msg: str) -> str:
    """从 mypy 的消息中去除文件名部分。"""
    _, tail = os.path.splitdrive(msg)  # 分离驱动器盘符
    return tail.split(":", 1)[-1]  # 返回去除文件名后的消息部分


def _run_mypy() -> Dict[str, List[str]]:
    """在运行任何类型提示测试之前，清理缓存并运行 mypy。"""
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)  # 如果存在缓存目录，递归删除

    rc: Dict[str, List[str]] = {}  # 初始化结果字典
    for directory in (REVEAL_DIR, PASS_DIR, FAIL_DIR):
        # 运行 mypy
        stdout, stderr, _ = api.run(
            [
                "--show-absolute-path",
                "--config-file",
                MYPY_INI,
                "--cache-dir",
                CACHE_DIR,
                directory,
            ]
        )
        assert not stderr, stderr  # 断言标准错误输出为空

        stdout = stdout.replace("*", "")  # 去除输出中的 '*'

        # 解析输出
        iterator = itertools.groupby(stdout.split("\n"), key=_key_func)
        rc.update((k, list(v)) for k, v in iterator if k)  # 更新结果字典
    return rc  # 返回结果字典


def get_test_cases(directory):
    """生成指定目录下的测试文件路径生成器。"""
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.startswith("disabled_"):
                continue  # 忽略以 'disabled_' 开头的文件
            if os.path.splitext(fname)[-1] == ".py":
                fullpath = os.path.join(root, fname)
                yield fullpath  # 返回每个 Python 文件的完整路径


_FAIL_MSG1 = """Extra error at line {}
Extra error: {!r}
"""

_FAIL_MSG2 = """Error mismatch at line {}
Expected error: {!r}
Observed error: {!r}
"""


def _test_fail(
    path: str, error: str, expected_error: Optional[str], lineno: int
) -> None:
    """测试函数，用于检查失败的测试。

    Args:
        path: 文件路径
        error: 实际错误消息
        expected_error: 期望的错误消息
        lineno: 出错的行号
    """
    if expected_error is None:
        raise AssertionError(_FAIL_MSG1.format(lineno, error))  # 抛出断言错误异常
    elif error not in expected_error:
        raise AssertionError(_FAIL_MSG2.format(lineno, expected_error, error))  # 抛出断言错误异常


def _construct_format_dict():
    # 此处未完整提供代码，无法为其余部分添加注释
    pass
    # 创建一个字典 dct，包含了各种 PyTorch 模块和类的名称与它们的完整路径
    dct = {
        "ModuleList": "torch.nn.modules.container.ModuleList",  # ModuleList 类的完整路径
        "AdaptiveAvgPool2d": "torch.nn.modules.pooling.AdaptiveAvgPool2d",  # AdaptiveAvgPool2d 类的完整路径
        "AdaptiveMaxPool2d": "torch.nn.modules.pooling.AdaptiveMaxPool2d",  # AdaptiveMaxPool2d 类的完整路径
        "Tensor": "torch._tensor.Tensor",  # Tensor 类的完整路径
        "Adagrad": "torch.optim.adagrad.Adagrad",  # Adagrad 优化器类的完整路径
        "Adam": "torch.optim.adam.Adam",  # Adam 优化器类的完整路径
    }
    # 返回定义好的包含 PyTorch 模块名称与完整路径的字典 dct
    return dct
#: 包含所有支持格式键的字典（作为键）及其匹配值
FORMAT_DICT: Dict[str, str] = _construct_format_dict()


def _parse_reveals(file: IO[str]) -> List[str]:
    """从传入的类文件对象中提取并解析所有 ``"  # E: "`` 注释。

    所有格式键将被替换为它们在 `FORMAT_DICT` 中对应的值，
    例如 ``"{Tensor}"`` 将变成 ``"torch.tensor.Tensor"``。
    """
    string = file.read().replace("*", "")

    # 获取所有基于 `# E:` 的注释
    comments_array = [str.partition("  # E: ")[2] for str in string.split("\n")]
    comments = "/n".join(comments_array)

    # 只在注释中搜索 `{*}` 模式，以避免意外抓取字典和集合
    key_set = set(re.findall(r"\{(.*?)\}", comments))
    kwargs = {
        k: FORMAT_DICT.get(k, f"<UNRECOGNIZED FORMAT KEY {k!r}>") for k in key_set
    }
    fmt_str = comments.format(**kwargs)

    return fmt_str.split("/n")


_REVEAL_MSG = """在第 {} 行发现显示不匹配

期望的显示: {!r}
实际的显示: {!r}
"""


def _test_reveal(path: str, reveal: str, expected_reveal: str, lineno: int) -> None:
    if reveal not in expected_reveal:
        raise AssertionError(_REVEAL_MSG.format(lineno, expected_reveal, reveal))


@unittest.skipIf(NO_MYPY, reason="Mypy 未安装")
class TestTyping(TestCase):
    _lock = Lock()
    _cached_output: Optional[Dict[str, List[str]]] = None

    @classmethod
    def get_mypy_output(cls) -> Dict[str, List[str]]:
        with cls._lock:
            if cls._cached_output is None:
                cls._cached_output = _run_mypy()
            return cls._cached_output

    @parametrize(
        "path",
        get_test_cases(PASS_DIR),
        name_fn=lambda b: os.path.relpath(b, start=PASS_DIR),
    )
    def test_success(self, path) -> None:
        output_mypy = self.get_mypy_output()
        if path in output_mypy:
            msg = "Unexpected mypy output\n\n"
            msg += "\n".join(_strip_filename(v) for v in output_mypy[path])
            raise AssertionError(msg)

    @parametrize(
        "path",
        get_test_cases(FAIL_DIR),
        name_fn=lambda b: os.path.relpath(b, start=FAIL_DIR),
    )
    # 定义一个测试函数，用于测试失败情况下的行为
    def test_fail(self, path):
        # 隐藏测试失败的回溯信息
        __tracebackhide__ = True

        # 打开指定路径的文件，并读取所有行到列表中
        with open(path) as fin:
            lines = fin.readlines()

        # 创建一个默认字典，用于存储错误信息
        errors = defaultdict(lambda: "")

        # 获取当前路径下的 mypy 输出结果
        output_mypy = self.get_mypy_output()
        # 断言指定路径在 mypy 输出结果中
        self.assertIn(path, output_mypy)

        # 遍历 mypy 输出中的每一行错误信息
        for error_line in output_mypy[path]:
            # 去除文件名后的错误行
            error_line = _strip_filename(error_line)
            # 使用正则表达式匹配错误行的格式
            match = re.match(
                r"(?P<lineno>\d+):(?P<colno>\d+): (error|note): .+$",
                error_line,
            )
            # 如果匹配结果为 None，则抛出异常
            if match is None:
                raise ValueError(f"Unexpected error line format: {error_line}")
            # 提取行号并转换为整数
            lineno = int(match.group("lineno"))
            # 将错误信息按行号存入字典中
            errors[lineno] += f"{error_line}\n"

        # 遍历文件的每一行，验证是否包含预期的错误信息
        for i, line in enumerate(lines):
            lineno = i + 1
            # 如果行以 '#' 开头或者行号不在错误字典中，跳过当前循环
            if line.startswith("#") or (" E:" not in line and lineno not in errors):
                continue

            # 获取目标行
            target_line = lines[lineno - 1]
            # 断言目标行包含预期的错误信息
            self.assertIn(
                "# E:", target_line, f"Unexpected mypy output\n\n{errors[lineno]}"
            )
            # 提取错误标记信息
            marker = target_line.split("# E:")[-1].strip()
            # 获取预期的错误信息
            expected_error = errors.get(lineno)
            # 调用测试失败的辅助函数，验证行为
            _test_fail(path, marker, expected_error, lineno)

    # 参数化测试函数，用于测试 reveal 情况下的行为
    @parametrize(
        "path",
        get_test_cases(REVEAL_DIR),
        # 计算相对路径函数，作为测试用例的名称
        name_fn=lambda b: os.path.relpath(b, start=REVEAL_DIR),
    )
    def test_reveal(self, path):
        # 隐藏测试失败的回溯信息
        __tracebackhide__ = True

        # 打开指定路径的文件，并解析其中的 reveal 行
        with open(path) as fin:
            lines = _parse_reveals(fin)

        # 获取当前路径下的 mypy 输出结果
        output_mypy = self.get_mypy_output()
        # 断言指定路径在 mypy 输出结果中
        assert path in output_mypy

        # 遍历 mypy 输出中的每一行错误信息
        for error_line in output_mypy[path]:
            # 使用正则表达式匹配 reveal 行的格式
            match = re.match(
                r"^.+\.py:(?P<lineno>\d+):(?P<colno>\d+): note: .+$",
                error_line,
            )
            # 如果匹配结果为 None，则抛出异常
            if match is None:
                raise ValueError(f"Unexpected reveal line format: {error_line}")
            # 提取行号并减去 1，因为列表从 0 开始
            lineno = int(match.group("lineno")) - 1
            # 断言 reveal 行中包含特定的文本信息
            assert "Revealed type is" in error_line

            # 获取行标记信息
            marker = lines[lineno]
            # 调用测试 reveal 的辅助函数，验证行为
            _test_reveal(path, marker, error_line, 1 + lineno)
# 实例化参数化测试，使用 TestTyping 类来创建测试实例
instantiate_parametrized_tests(TestTyping)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```