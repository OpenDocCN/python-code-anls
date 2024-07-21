# `.\pytorch\tools\linter\adapters\shellcheck_linter.py`

```py
from __future__ import annotations
# 导入必要的未来注释以支持类型注解

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from enum import Enum
from typing import NamedTuple

# 定义一个常量，表示代码检查工具为 ShellCheck
LINTER_CODE = "SHELLCHECK"

# 枚举类型，表示代码检查结果的严重程度
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

# 命名元组，用于存储代码检查结果的详细信息
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

# 函数：运行系统命令，并记录调试日志
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

# 函数：检查给定文件列表的代码风格和错误
def check_files(
    files: list[str],
) -> list[LintMessage]:
    try:
        # 调用 run_command 函数执行 shellcheck 命令
        proc = run_command(
            ["shellcheck", "--external-sources", "--format=json1"] + files
        )
    except OSError as err:
        # 处理执行 shellcheck 命令失败的情况
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    results = json.loads(stdout)["comments"]
    # 将 shellcheck 的结果转换为 LintMessage 对象列表
    return [
        LintMessage(
            path=result["file"],
            name=f"SC{result['code']}",
            description=result["message"],
            line=result["line"],
            char=result["column"],
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            original=None,
            replacement=None,
        )
        for result in results
    ]

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="shellcheck runner",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    # 检查是否安装了 shellcheck 命令
    if shutil.which("shellcheck") is None:
        # 如果没有安装，输出错误信息并退出程序
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description="shellcheck is not installed, did you forget to run `lintrunner init`?",
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        sys.exit(0)

    # 解析命令行参数
    args = parser.parse_args()

    # 对指定文件进行代码检查
    lint_messages = check_files(args.filenames)
    for lint_message in lint_messages:
        # 输出每个 lint_message 对象的详细信息
        print(json.dumps(lint_message._asdict()), flush=True)
```