# `.\pytorch\tools\linter\adapters\grep_linter.py`

```
"""
Generic linter that greps for a pattern and optionally suggests replacements.
"""

# 导入必要的库
from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from typing import Any, NamedTuple

# 检查操作系统是否为 Windows
IS_WINDOWS: bool = os.name == "nt"

# 打印错误信息到标准错误流
def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)

# 定义枚举类型LintSeverity
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

# 定义LintMessage命名元组
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

# 将路径名转换为 POSIX 风格
def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name

# 运行系统命令
def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)

# 对文件进行 lint 操作
def lint_file(
    matching_line: str,
    allowlist_pattern: str,
    replace_pattern: str,
    linter_name: str,
    error_name: str,
    error_description: str,
) -> LintMessage | None:
    # 解析匹配行，获取文件名
    split = matching_line.split(":")
    filename = split[0]

    # 如果存在 allowlist_pattern，则执行 grep 命令
    if allowlist_pattern:
        try:
            proc = run_command(["grep", "-nEHI", allowlist_pattern, filename])
        except Exception as err:
            return LintMessage(
                path=None,
                line=None,
                char=None,
                code=linter_name,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )

        # 如果找到 allowlist_pattern，则终止 lint 操作
        if proc.returncode == 0:
            return None

    original = None
    replacement = None
    # 如果存在替换模式
    if replace_pattern:
        # 打开指定文件并读取其内容到变量 original
        with open(filename) as f:
            original = f.read()

        try:
            # 运行 sed 命令来替换文件中的文本内容，将结果保存到 replacement 中
            proc = run_command(["sed", "-r", replace_pattern, filename])
            replacement = proc.stdout.decode("utf-8")
        except Exception as err:
            # 如果命令执行失败，返回相应的错误消息对象
            return LintMessage(
                path=None,
                line=None,
                char=None,
                code=linter_name,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )

    # 构造并返回一个LintMessage对象，表示Lint检查的错误消息
    return LintMessage(
        path=split[0],
        line=int(split[1]) if len(split) > 1 else None,
        char=None,
        code=linter_name,
        severity=LintSeverity.ERROR,
        name=error_name,
        original=original,
        replacement=replacement,
        description=error_description,
    )
# 定义主函数，用于执行程序的主要逻辑
def main() -> None:
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(
        description="grep wrapper linter.",
        fromfile_prefix_chars="@",
    )
    # 添加用于匹配的模式参数
    parser.add_argument(
        "--pattern",
        required=True,
        help="pattern to grep for",
    )
    # 添加允许列表模式参数，如果文件中存在该模式则不进行匹配
    parser.add_argument(
        "--allowlist-pattern",
        help="if this pattern is true in the file, we don't grep for pattern",
    )
    # 添加 linter 名称参数，指定使用的 linter 名称
    parser.add_argument(
        "--linter-name",
        required=True,
        help="name of the linter",
    )
    # 添加匹配仅第一个结果的标志参数
    parser.add_argument(
        "--match-first-only",
        action="store_true",
        help="only match the first hit in the file",
    )
    # 添加错误名称参数，用于描述错误的可读名称
    parser.add_argument(
        "--error-name",
        required=True,
        help="human-readable description of what the error is",
    )
    # 添加错误描述参数，当匹配到模式时显示的消息
    parser.add_argument(
        "--error-description",
        required=True,
        help="message to display when the pattern is found",
    )
    # 添加替换模式参数，用于传递给 `sed -r` 的替换文本形式
    parser.add_argument(
        "--replace-pattern",
        help=(
            "the form of a pattern passed to `sed -r`. "
            "If specified, this will become proposed replacement text."
        ),
    )
    # 添加详细日志记录标志参数
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    # 添加文件名参数，指定要进行检查的文件路径列表
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 根据是否启用详细日志设置日志格式和级别
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    # 如果设置了匹配仅第一个结果的标志，准备将其添加到 grep 命令参数中
    files_with_matches = []
    if args.match_first_only:
        files_with_matches = ["--files-with-matches"]

    try:
        # 执行 grep 命令，查找指定模式在指定文件中的匹配结果
        proc = run_command(
            ["grep", "-nEHI", *files_with_matches, args.pattern, *args.filenames]
        )
    except Exception as err:
        # 处理命令执行失败的情况，创建相应的 lint 消息
        err_msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code=args.linter_name,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"Failed due to {err.__class__.__name__}:\n{err}"
                if not isinstance(err, subprocess.CalledProcessError)
                else (
                    "COMMAND (exit code {returncode})\n"
                    "{command}\n\n"
                    "STDERR\n{stderr}\n\n"
                    "STDOUT\n{stdout}"
                ).format(
                    returncode=err.returncode,
                    command=" ".join(as_posix(x) for x in err.cmd),
                    stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                    stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                )
            ),
        )
        # 将 lint 消息转换为 JSON 格式输出，并刷新输出缓冲区
        print(json.dumps(err_msg._asdict()), flush=True)
        # 退出程序
        sys.exit(0)
    # 将子进程的标准输出解码为字符串，并按行分割成列表
    lines = proc.stdout.decode().splitlines()
    
    # 遍历每一行输出
    for line in lines:
        # 调用 lint_file 函数对每行进行代码审查
        lint_message = lint_file(
            line,  # 当前行的代码内容
            args.allowlist_pattern,  # 允许列表模式
            args.replace_pattern,    # 替换模式
            args.linter_name,        # 代码审查工具名称
            args.error_name,         # 错误名称
            args.error_description   # 错误描述
        )
        # 如果 lint_message 不为空，则将其转换为字典并打印为 JSON 格式，刷新输出
        if lint_message is not None:
            print(json.dumps(lint_message._asdict()), flush=True)
# 如果当前模块是作为主程序运行
if __name__ == "__main__":
    # 调用主函数 main()
    main()
```