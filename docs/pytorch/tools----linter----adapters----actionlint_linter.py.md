# `.\pytorch\tools\linter\adapters\actionlint_linter.py`

```
# 从未来版本导入注解功能，支持类型注解
from __future__ import annotations

# 导入命令行参数解析模块
import argparse
# 导入并发执行模块
import concurrent.futures
# 导入 JSON 模块
import json
# 导入日志记录模块
import logging
# 导入操作系统相关功能模块
import os
# 导入正则表达式模块
import re
# 导入子进程管理模块
import subprocess
# 导入系统相关功能模块
import sys
# 导入时间相关功能模块
import time
# 导入枚举类型模块
from enum import Enum
# 导入命名元组类型模块
from typing import NamedTuple

# 定义一个常量，用于指定 Linter 的代码标识
LINTER_CODE = "ACTIONLINT"

# 定义LintSeverity枚举类型，包含严重性级别：错误、警告、建议、禁用
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

# 定义LintMessage命名元组类型，表示一个Lint消息
class LintMessage(NamedTuple):
    path: str | None  # 文件路径或空
    line: int | None  # 行号或空
    char: int | None  # 字符位置或空
    code: str          # 代码标识
    severity: LintSeverity  # 严重性级别
    name: str          # 消息名称
    original: str | None  # 原始内容或空
    replacement: str | None  # 替换内容或空
    description: str | None  # 描述信息或空

# 定义一个正则表达式模式对象，用于匹配Lint结果
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?P<char>\d+):
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)

# 定义一个函数，用于运行系统命令并返回完成的进程对象
def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))  # 记录调试信息，记录执行的命令
    start_time = time.monotonic()  # 记录开始时间
    try:
        return subprocess.run(
            args,
            capture_output=True,  # 捕获命令的输出
        )
    finally:
        end_time = time.monotonic()  # 记录结束时间
        logging.debug("took %dms", (end_time - start_time) * 1000)  # 记录执行时间

# 定义一个函数，用于检查单个文件的Lint情况
def check_file(
    binary: str,
    file: str,
) -> list[LintMessage]:
    try:
        proc = run_command(
            [
                binary,
                "-ignore",
                '"runs-on" section must be sequence node but got mapping node with "!!map" tag',
                file,
            ]
        )
    except OSError as err:
        # 如果命令执行失败，返回一个带有错误信息的LintMessage对象列表
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
    stdout = str(proc.stdout, "utf-8").strip()  # 获取命令输出并去除首尾空白字符
    # 使用正则表达式匹配输出，生成LintMessage对象列表
    return [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            char=int(match["char"]),
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ]

# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="actionlint runner",
        fromfile_prefix_chars="@",
    )
    # 添加命令行参数：指定actionlint二进制文件路径
    parser.add_argument(
        "--binary",
        required=True,
        help="actionlint binary path",
    )
    # 添加命令行参数：指定需要检查的文件路径列表
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 如果指定的二进制文件路径不存在
    if not os.path.exists(args.binary):
        # 创建LintMessage对象，用于描述lint错误信息
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"Could not find actionlint binary at {args.binary},"
                " you may need to run `lintrunner init`."
            ),
        )
        # 将LintMessage对象转换为字典并打印出来
        print(json.dumps(err_msg._asdict()), flush=True)
        # 程序正常退出
        sys.exit(0)

    # 使用ThreadPoolExecutor创建线程池
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),  # 设置最大工作线程数为CPU核心数
        thread_name_prefix="Thread",  # 线程名称前缀为"Thread"
    ) as executor:
        # 使用executor.submit提交任务，并将future对象与filename关联
        futures = {
            executor.submit(
                check_file,  # 要执行的函数是check_file
                args.binary,  # 第一个参数是args.binary
                filename,  # 第二个参数是当前循环中的filename
            ): filename
            for filename in args.filenames  # 遍历args.filenames中的每个filename
        }
        # 遍历已完成的任务future
        for future in concurrent.futures.as_completed(futures):
            try:
                # 获取future的结果，即lint消息列表
                for lint_message in future.result():
                    # 将lint消息转换为字典并打印出来
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                # 发生异常时记录错误日志
                logging.critical('Failed at "%s".', futures[future])
                # 抛出异常以结束程序执行
                raise
```