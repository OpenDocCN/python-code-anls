# `.\pytorch\tools\linter\adapters\mypy_linter.py`

```py
# 从未来模块导入 annotations 功能，以支持类型注解中的 str 类型
from __future__ import annotations

# 导入必要的模块
import argparse  # 命令行参数解析模块
import json  # JSON 数据处理模块
import logging  # 日志记录模块
import os  # 系统相关操作模块
import re  # 正则表达式模块
import subprocess  # 子进程管理模块
import sys  # 系统相关功能模块
import time  # 时间操作模块
from enum import Enum  # 枚举类型模块
from pathlib import Path  # 文件路径操作模块
from typing import Any, NamedTuple  # 类型提示相关模块


# 判断当前操作系统是否为 Windows
IS_WINDOWS: bool = os.name == "nt"


# 将输出写入标准错误流的函数定义
def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


# 定义代码检查的严重性枚举类
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


# 定义代码检查消息的命名元组
class LintMessage(NamedTuple):
    path: str | None  # 文件路径或 None
    line: int | None  # 行号或 None
    char: int | None  # 字符位置或 None
    code: str  # 错误代码
    severity: LintSeverity  # 错误严重性
    name: str  # 错误名称
    original: str | None  # 原始内容或 None
    replacement: str | None  # 替换内容或 None
    description: str | None  # 错误描述或 None


# 将路径名转换为 POSIX 格式的函数定义
def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


# 定义匹配代码检查结果的正则表达式对象
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<severity>\S+?):?
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)

# 定义匹配内部错误消息的正则表达式对象
INTERNAL_ERROR_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    \s(?P<severity>\S+?):?
    \s(?P<message>INTERNAL\sERROR.*)
    $
    """
)


# 运行系统命令的函数定义
def run_command(
    args: list[str],  # 命令及其参数列表
    *,
    extra_env: dict[str, str] | None,  # 额外的环境变量字典或 None
    retries: int,  # 重试次数
) -> subprocess.CompletedProcess[bytes]:  # 返回子进程完成后的字节串结果
    logging.debug("$ %s", " ".join(args))  # 记录调试信息，输出运行的命令
    start_time = time.monotonic()  # 记录开始时间
    try:
        return subprocess.run(
            args,
            capture_output=True,  # 捕获命令输出
        )
    finally:
        end_time = time.monotonic()  # 记录结束时间
        logging.debug("took %dms", (end_time - start_time) * 1000)  # 输出运行时间


# 定义 mypy 错误严重性映射字典
severities = {
    "error": LintSeverity.ERROR,  # 错误严重性映射为 LintSeverity 的 ERROR 类型
    "note": LintSeverity.ADVICE,  # 通知信息严重性映射为 LintSeverity 的 ADVICE 类型
}


# 检查 mypy 是否已安装的函数定义
def check_mypy_installed(code: str) -> list[LintMessage]:
    cmd = [sys.executable, "-mmypy", "-V"]  # 构造检查 mypy 版本的命令
    try:
        subprocess.run(cmd, check=True, capture_output=True)  # 运行命令并捕获输出
        return []  # 返回空的消息列表，表示 mypy 已安装
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace")  # 解码并处理错误消息
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=f"Could not run '{' '.join(cmd)}': {msg}",  # 返回运行命令失败的详细描述
            )
        ]


# 检查指定文件的函数定义
def check_files(
    filenames: list[str],  # 文件名列表
    config: str,  # 配置文件名
    retries: int,  # 重试次数
    code: str,  # 错误代码
) -> list[LintMessage]:  # 返回错误消息列表
    # dmypy 存在 bug，不能处理绝对路径文件名，详见 https://github.com/python/mypy/issues/16768
    # 将文件名列表转换为相对路径列表
    filenames = [os.path.relpath(f) for f in filenames]
    
    try:
        # 执行指定的命令，运行 dmypy 工具进行类型检查
        proc = run_command(
            ["dmypy", "run", "--", f"--config={config}"] + filenames,
            extra_env={},
            retries=retries,
        )
    except OSError as err:
        # 如果执行命令过程中出现 OSError 异常，则返回包含错误信息的 lint 消息列表
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    
    # 获取命令执行的标准输出，并转换为 UTF-8 编码的字符串，去除首尾空白字符
    stdout = str(proc.stdout, "utf-8").strip()
    # 获取命令执行的标准错误输出，并转换为 UTF-8 编码的字符串，去除首尾空白字符
    stderr = str(proc.stderr, "utf-8").strip()
    
    # 解析标准输出中的结果，生成 lint 消息列表
    rc = [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            # 如果存在列号且不以 "-" 开头，则将其转换为整数；否则设为 None
            char=int(match["column"])
            if match["column"] is not None and not match["column"].startswith("-")
            else None,
            code=code,
            # 根据严重性字符串获取对应的 LintSeverity 枚举值，默认为 ERROR
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)  # 遍历结果匹配对象列表
    ] + [
        # 解析标准错误输出中的内部错误，生成 lint 消息列表
        LintMessage(
            path=match["file"],
            name="INTERNAL ERROR",
            description=match["message"],
            line=int(match["line"]),
            char=None,  # 内部错误没有列号，设为 None
            code=code,
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in INTERNAL_ERROR_RE.finditer(stderr)  # 遍历内部错误匹配对象列表
    ]
    
    # 返回 lint 消息列表 rc
    return rc
# 定义主函数，程序的入口点，不返回任何值
def main() -> None:
    # 创建参数解析器对象，用于处理命令行参数
    parser = argparse.ArgumentParser(
        description="mypy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    # 添加命令行参数 --retries，用于指定超时重试次数，默认为3次
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out mypy",
    )
    # 添加命令行参数 --config，必选参数，指定 mypy 的 .ini 配置文件路径
    parser.add_argument(
        "--config",
        required=True,
        help="path to an mypy .ini config file",
    )
    # 添加命令行参数 --code，用于指定 lint 结果报告的代码标识，默认为 "MYPY"
    parser.add_argument(
        "--code",
        default="MYPY",
        help="the code this lint should report as",
    )
    # 添加命令行参数 --verbose，开启详细日志记录的开关
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    # 添加位置参数 filenames，表示需要进行 lint 的文件路径列表，至少需要指定一个文件
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    # 解析命令行参数，将结果存储在 args 变量中
    args = parser.parse_args()

    # 根据 --verbose 参数设置日志格式和日志级别
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    # 使用字典来存储文件名及其处理标记，保持顺序以满足 mypy 的需求
    filenames: dict[str, bool] = {}

    # 遍历位置参数中的文件名列表
    for filename in args.filenames:
        # 如果文件名以 .pyi 结尾，按照 PEP-484 的建议，使用 stub 文件进行检查
        if filename.endswith(".pyi"):
            filenames[filename] = True
            continue

        # 构造对应的 stub 文件名
        stub_filename = filename.replace(".py", ".pyi")
        # 如果 stub 文件存在，则使用 stub 文件进行检查
        if Path(stub_filename).exists():
            filenames[stub_filename] = True
        else:
            # 否则，使用原始文件进行检查
            filenames[filename] = True

    # 调用 check_mypy_installed 函数检查 mypy 是否安装，并获取检查消息
    lint_messages = check_mypy_installed(args.code) + check_files(
        list(filenames), args.config, args.retries, args.code
    )
    # 遍历 lint 消息列表，将消息以 JSON 格式输出到标准输出流，并刷新缓冲区
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```