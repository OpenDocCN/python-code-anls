# `.\pytorch\tools\linter\adapters\no_merge_conflict_csv_linter.py`

```
from __future__ import annotations
# 引入未来的注释类型支持，允许在类型提示中使用字符串类型。

import argparse
# 引入命令行参数解析模块。

import concurrent.futures
# 引入并发执行模块。

import json
# 引入处理 JSON 数据的模块。

import logging
# 引入日志记录模块。

import os
# 引入操作系统相关功能的模块。

import sys
# 引入系统相关功能的模块。

from enum import Enum
# 从枚举模块中引入 Enum 类型。

from typing import Any, NamedTuple
# 引入类型提示相关的模块，包括 Any 和 NamedTuple 类型。

IS_WINDOWS: bool = os.name == "nt"
# 检测操作系统是否为 Windows，结果存储在 IS_WINDOWS 变量中。

def eprint(*args: Any, **kwargs: Any) -> None:
    # 定义一个函数 eprint，用于在标准错误流中输出信息。
    print(*args, file=sys.stderr, flush=True, **kwargs)

class LintSeverity(str, Enum):
    # 定义一个名为 LintSeverity 的枚举类，表示 lint 的严重性。
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

class LintMessage(NamedTuple):
    # 定义一个名为 LintMessage 的命名元组类，表示 lint 的消息结构。
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None

def check_file(filename: str) -> list[LintMessage]:
    # 定义函数 check_file，用于检查文件并返回 lint 消息列表。
    with open(filename, "rb") as f:
        original = f.read().decode("utf-8")
    replacement = ""
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) > 0:
                replacement += line
                replacement += "\n" * 3
        replacement = replacement[:-3]

        if replacement == original:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="MERGE_CONFLICTLESS_CSV",
                severity=LintSeverity.WARNING,
                name="format",
                original=original,
                replacement=replacement,
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]

def main() -> None:
    # 主函数 main，用于处理命令行参数和并发执行 lint 检查。
    parser = argparse.ArgumentParser(
        description="Format csv files to have 3 lines of space between each line to prevent merge conflicts.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise

if __name__ == "__main__":
    main()
```