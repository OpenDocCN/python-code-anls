# `.\pytorch\tools\linter\adapters\flake8_linter.py`

```py
# 从 __future__ 模块中导入 annotations 功能，使得类型提示中的字符串支持 | 运算符
from __future__ import annotations

# 导入命令行参数解析模块 argparse
import argparse
# 导入 JSON 操作模块 json
import json
# 导入日志记录模块 logging
import logging
# 导入操作系统相关模块 os
import os
# 导入正则表达式模块 re
import re
# 导入子进程管理模块 subprocess
import subprocess
# 导入系统相关模块 sys
import sys
# 导入时间模块 time
import time
# 导入枚举模块 Enum
from enum import Enum
# 导入类型提示相关模块 NamedTuple
from typing import Any, NamedTuple

# 检测当前操作系统是否为 Windows，将结果保存到 IS_WINDOWS 变量中
IS_WINDOWS: bool = os.name == "nt"


# 定义一个打印到标准错误流的函数 eprint
def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


# 定义一个枚举类 LintSeverity，表示代码检查的严重程度
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


# 定义一个命名元组 LintMessage，用于表示代码检查结果中的消息
class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


# 定义一个函数 as_posix，将给定的文件名转换为 POSIX 风格的路径
def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


# fmt: off
# 定义一个集合 DOCUMENTED_IN_FLAKE8RULES，包含在 Flake8 规则中有文档记录的错误代码
DOCUMENTED_IN_FLAKE8RULES: set[str] = {
    "E101", "E111", "E112", "E113", "E114", "E115", "E116", "E117",
    "E121", "E122", "E123", "E124", "E125", "E126", "E127", "E128", "E129",
    "E131", "E133",
    "E201", "E202", "E203",
    "E211",
    "E221", "E222", "E223", "E224", "E225", "E226", "E227", "E228",
    "E231",
    "E241", "E242",
    "E251",
    "E261", "E262", "E265", "E266",
    "E271", "E272", "E273", "E274", "E275",
    "E301", "E302", "E303", "E304", "E305", "E306",
    "E401", "E402",
    "E501", "E502",
    "E701", "E702", "E703", "E704",
    "E711", "E712", "E713", "E714",
    "E721", "E722",
    "E731",
    "E741", "E742", "E743",
    "E901", "E902", "E999",
    "W191",
    "W291", "W292", "W293",
    "W391",
    "W503", "W504",
    "W601", "W602", "W603", "W604", "W605",
    "F401", "F402", "F403", "F404", "F405",
    "F811", "F812",
    "F821", "F822", "F823",
    "F831",
    "F841",
    "F901",
    "C901",
}

# 定义一个集合 DOCUMENTED_IN_FLAKE8COMPREHENSIONS，包含在 Flake8 comprehensions 规则中有文档记录的错误代码
DOCUMENTED_IN_FLAKE8COMPREHENSIONS: set[str] = {
    "C400", "C401", "C402", "C403", "C404", "C405", "C406", "C407", "C408", "C409",
    "C410",
    "C411", "C412", "C413", "C414", "C415", "C416",
}

# 定义一个集合 DOCUMENTED_IN_BUGBEAR，包含在 flake8-bugbear 规则中有文档记录的错误代码
DOCUMENTED_IN_BUGBEAR: set[str] = {
    "B001", "B002", "B003", "B004", "B005", "B006", "B007", "B008", "B009", "B010",
    "B011", "B012", "B013", "B014", "B015",
    "B301", "B302", "B303", "B304", "B305", "B306",
    "B901", "B902", "B903", "B950",
}
# fmt: on


# 定义一个正则表达式模式 RESULTS_RE，用于解析代码检查结果的输出格式
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<code>\S+?):?
    \s(?P<message>.*)
    $
    """
)


# 定义一个函数 _test_results_re，用于测试 RESULTS_RE 正则表达式的匹配功能
def _test_results_re() -> None:
    """
    >>> def t(s): return RESULTS_RE.search(s).groupdict()

    >>> t(r"file.py:80:1: E302 expected 2 blank lines, found 1")
    ... # doctest: +NORMALIZE_WHITESPACE
    # 创建一个字典，包含静态分析器返回的错误或警告的详细信息
    {
        'file': 'file.py',        # 文件名为 'file.py'
        'line': '80',             # 行号为 80
        'column': '1',            # 列号为 1
        'code': 'E302',           # 错误代码为 E302
        'message': 'expected 2 blank lines, found 1'  # 错误消息为 'expected 2 blank lines, found 1'
    }
    
    >>> t(r"file.py:7:1: P201: Resource `stdout` is acquired but not always released.")
    ... # doctest: +NORMALIZE_WHITESPACE
    {
        'file': 'file.py',        # 文件名为 'file.py'
        'line': '7',              # 行号为 7
        'column': '1',            # 列号为 1
        'code': 'P201',           # 错误代码为 P201
        'message': 'Resource `stdout` is acquired but not always released.'  # 错误消息为 'Resource `stdout` is acquired but not always released.'
    }
    
    >>> t(r"file.py:8:-10: W605 invalid escape sequence '/'")
    ... # doctest: +NORMALIZE_WHITESPACE
    {
        'file': 'file.py',        # 文件名为 'file.py'
        'line': '8',              # 行号为 8
        'column': '-10',          # 列号为 -10
        'code': 'W605',           # 错误代码为 W605
        'message': "invalid escape sequence '/'"  # 错误消息为 "invalid escape sequence '/'"
    }
# 定义一个私有函数 `_run_command`，用于执行系统命令并返回其完成的进程对象
def _run_command(
    args: list[str],  # 接收命令及其参数的列表
    *,  # 使用命名关键字参数来标识后续参数
    extra_env: dict[str, str] | None,  # 可选的额外环境变量字典，可以为 None
) -> subprocess.CompletedProcess[str]:  # 返回 subprocess.CompletedProcess[str] 类型的对象
    # 记录调试信息，打印执行的命令
    logging.debug(
        "$ %s",
        " ".join(
            ([f"{k}={v}" for (k, v) in extra_env.items()] if extra_env else []) + args
        ),
    )
    start_time = time.monotonic()  # 记录函数开始执行的时间
    try:
        # 调用 subprocess.run 执行命令，捕获输出，检查返回状态，使用 UTF-8 解码
        return subprocess.run(
            args,
            capture_output=True,
            check=True,
            encoding="utf-8",
        )
    finally:
        end_time = time.monotonic()  # 记录函数执行完毕的时间
        logging.debug("took %dms", (end_time - start_time) * 1000)  # 打印函数执行时间（毫秒）的调试信息


# 定义函数 `run_command`，用于执行命令，并在失败时进行重试
def run_command(
    args: list[str],  # 接收命令及其参数的列表
    *,  # 使用命名关键字参数来标识后续参数
    extra_env: dict[str, str] | None,  # 可选的额外环境变量字典，可以为 None
    retries: int,  # 重试次数
) -> subprocess.CompletedProcess[str]:  # 返回 subprocess.CompletedProcess[str] 类型的对象
    remaining_retries = retries  # 设置剩余重试次数为总重试次数
    while True:
        try:
            return _run_command(args, extra_env=extra_env)  # 调用 _run_command 函数执行命令
        except subprocess.CalledProcessError as err:  # 捕获命令执行失败的异常
            if remaining_retries == 0 or not re.match(
                r"^ERROR:1:1: X000 linting with .+ timed out after \d+ seconds",
                err.stdout,
            ):
                raise err  # 如果没有剩余重试次数或者不匹配特定错误消息，则抛出异常
            remaining_retries -= 1  # 减少剩余重试次数
            logging.warning(
                "(%s/%s) Retrying because command failed with: %r",
                retries - remaining_retries,
                retries,
                err,
            )  # 记录警告信息，显示重试次数及原因
            time.sleep(1)  # 等待 1 秒后进行重试


# 定义函数 `get_issue_severity`，根据 lint 错误代码返回对应的严重性
def get_issue_severity(code: str) -> LintSeverity:
    # 如果错误代码以指定的前缀开始，则返回建议级别
    if any(
        code.startswith(x)
        for x in [
            "B9",
            "C4",
            "C9",
            "E2",
            "E3",
            "E5",
            "F401",
            "F403",
            "F405",
            "T400",
            "T49",
        ]
    ):
        return LintSeverity.ADVICE

    # 如果错误代码以指定的前缀开始，则返回错误级别
    if any(code.startswith(x) for x in ["F821", "E999"]):
        return LintSeverity.ERROR

    # 否则，返回警告级别
    return LintSeverity.WARNING


# 定义函数 `get_issue_documentation_url`，根据 lint 错误代码返回对应的文档 URL
def get_issue_documentation_url(code: str) -> str:
    if code in DOCUMENTED_IN_FLAKE8RULES:  # 如果错误代码在 flake8 规则文档中
        return f"https://www.flake8rules.com/rules/{code}.html"  # 返回对应的文档 URL

    if code in DOCUMENTED_IN_FLAKE8COMPREHENSIONS:  # 如果错误代码在 flake8-comprehensions 规则文档中
        return "https://pypi.org/project/flake8-comprehensions/#rules"  # 返回对应的文档 URL
    # 如果给定的 `code` 存在于 `DOCUMENTED_IN_BUGBEAR` 列表中
    if code in DOCUMENTED_IN_BUGBEAR:
        # 返回包含警告列表的 GitHub 链接地址
        return "https://github.com/PyCQA/flake8-bugbear#list-of-warnings"
    
    # 如果 `code` 不存在于 `DOCUMENTED_IN_BUGBEAR` 列表中，则返回空字符串
    return ""
# 定义一个函数，用于检查指定文件的代码规范性并返回 lint 消息列表
def check_files(
    filenames: list[str],                           # 输入参数：文件名列表
    flake8_plugins_path: str | None,                # 输入参数：flake8 插件路径，可为 None
    severities: dict[str, LintSeverity],            # 输入参数：lint 错误级别字典
    retries: int,                                   # 输入参数：重试次数
) -> list[LintMessage]:                            # 返回类型注释：LintMessage 对象列表

    # 尝试运行 flake8 命令检查文件
    try:
        proc = run_command(
            [sys.executable, "-mflake8", "--exit-zero"] + filenames,  # 运行的命令参数列表
            extra_env={"FLAKE8_PLUGINS_PATH": flake8_plugins_path}   # 如果指定了插件路径，将其作为环境变量传递
            if flake8_plugins_path                                  # 条件语句：如果 flake8 插件路径不为 None
            else None,                                               # 则传递该环境变量，否则传递 None
            retries=retries,                                          # 指定重试次数
        )
    except (OSError, subprocess.CalledProcessError) as err:
        # 如果捕获到 OSError 或 subprocess.CalledProcessError 异常，则返回一个特定的 lint 消息列表
        return [
            LintMessage(
                path=None,                                          # 文件路径为空
                line=None,                                          # 错误所在行为空
                char=None,                                          # 错误所在列为空
                code="FLAKE8",                                      # 错误代码标识为 FLAKE8
                severity=LintSeverity.ERROR,                        # 错误级别为 ERROR
                name="command-failed",                              # 错误名称为 command-failed
                original=None,                                      # 原始内容为空
                replacement=None,                                   # 替换内容为空
                description=(                                       # 错误描述信息，根据不同异常类型格式化不同的内容
                    f"Failed due to {err.__class__.__name__}:\n{err}"   # 如果是 OSError，则说明失败的原因
                    if not isinstance(err, subprocess.CalledProcessError)  # 如果不是 CalledProcessError
                    else (                                              # 则格式化为 Command failed 错误信息
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,                      # 格式化命令的返回码
                        command=" ".join(as_posix(x) for x in err.cmd), # 格式化命令内容
                        stderr=err.stderr.strip() or "(empty)",         # 格式化标准错误输出
                        stdout=err.stdout.strip() or "(empty)",         # 格式化标准输出
                    )
                ),
            )
        ]

    # 如果没有捕获异常，则根据 flake8 的输出结果生成 lint 消息列表
    return [
        LintMessage(
            path=match["file"],                                     # 错误所在文件路径
            name=match["code"],                                     # 错误代码
            description=f"{match['message']}\nSee {get_issue_documentation_url(match['code'])}",  # 错误描述信息
            line=int(match["line"]),                                # 错误所在行号
            char=int(match["column"])                               # 错误所在列号（如果有，并且不以 "-" 开头）
            if match["column"] is not None and not match["column"].startswith("-") else None,
            code="FLAKE8",                                          # 错误代码标识为 FLAKE8
            severity=severities.get(match["code"]) or get_issue_severity(match["code"]),  # 错误级别
            original=None,                                          # 原始内容为空
            replacement=None,                                       # 替换内容为空
        )
        for match in RESULTS_RE.finditer(proc.stdout)                # 遍历 flake8 输出结果，匹配 lint 错误信息
    ]


# 主函数，用于解析命令行参数并调用检查文件函数
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flake8 wrapper linter.",                     # 程序描述信息
        fromfile_prefix_chars="@",                                # 文件前缀字符为 @，支持从文件读取参数
    )
    parser.add_argument(
        "--flake8-plugins-path",                                  # 可选参数：flake8 插件路径
        help="FLAKE8_PLUGINS_PATH env value",                     # 参数帮助信息
    )
    parser.add_argument(
        "--severity",                                             # 可选参数：指定错误代码到错误级别的映射关系
        action="append",                                          # 追加模式，允许多次指定该参数
        help="map code to severity (e.g. `B950:advice`)",         # 参数帮助信息
    )
    parser.add_argument(
        "--retries",                                              # 可选参数：指定重试次数
        default=3,                                                # 默认值为 3
        type=int,                                                 # 参数类型为整数
        help="times to retry timed out flake8",                   # 参数帮助信息
    )
    parser.add_argument(
        "--verbose",                                              # 可选参数：是否启用详细日志
        action="store_true",                                      # 存在则为 True，否则为 False
        help="verbose logging",                                   # 参数帮助信息
    )
    parser.add_argument(
        "filenames",                                              # 位置参数：需要 lint 的文件路径列表
        nargs="+",                                                # 至少需要一个参数
        help="paths to lint",                                     # 参数帮助信息
    )
    args = parser.parse_args()                                     # 解析命令行参数
    # 设置日志系统的配置，指定日志格式和级别，输出到标准错误流
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET  # 如果启用了详细模式，则不设置日志级别限制
        if args.verbose  # 如果命令行参数指定了 verbose 标志
        else logging.DEBUG  # 如果文件名数目少于 1000，则设置为 DEBUG 级别
        if len(args.filenames) < 1000  # 判断文件名列表长度是否小于 1000
        else logging.INFO,  # 否则设置为 INFO 级别
        stream=sys.stderr,  # 日志输出到标准错误流
    )

    # 设置 flake8 插件路径为实际路径，或者为 None（如果未指定）
    flake8_plugins_path = (
        None
        if args.flake8_plugins_path is None  # 如果未提供 flake8 插件路径参数
        else os.path.realpath(args.flake8_plugins_path)  # 否则设置为指定路径的实际路径
    )

    # 初始化一个空的严重性字典
    severities: dict[str, LintSeverity] = {}

    # 如果命令行参数指定了 lint 严重性级别
    if args.severity:
        # 遍历每个指定的严重性级别字符串
        for severity in args.severity:
            # 拆分严重性字符串，应该以 ":" 分隔为两部分
            parts = severity.split(":", 1)
            # 如果拆分后的部分不是两个，则抛出异常，显示无效的严重性设置
            assert len(parts) == 2, f"invalid severity `{severity}`"
            # 将严重性级别与其对应的LintSeverity对象存入字典中
            severities[parts[0]] = LintSeverity(parts[1])

    # 检查指定文件的 lint 问题，返回 lint 消息列表
    lint_messages = check_files(
        args.filenames, flake8_plugins_path, severities, args.retries
    )

    # 遍历 lint 消息列表，将每条 lint 消息转换为 JSON 格式并输出到标准输出
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
if __name__ == "__main__":
    # 当作为主程序执行时，执行 main 函数
    main()
```