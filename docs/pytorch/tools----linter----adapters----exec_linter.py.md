# `.\pytorch\tools\linter\adapters\exec_linter.py`

```py
"""
EXEC: Ensure that source files are not executable.
"""

# 从未来版本导入注释类型（用于支持类型提示）
from __future__ import annotations

# 导入必要的库
import argparse        # 解析命令行参数
import json            # 处理 JSON 数据
import logging         # 记录日志
import os              # 提供操作系统相关功能
import sys             # 提供与 Python 解释器相关的功能
from enum import Enum  # 创建枚举类型
from typing import NamedTuple  # 提供命名元组的支持


# 定义 Linter 代码
LINTER_CODE = "EXEC"


# 定义 lint 严重性级别的枚举
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


# 定义 lint 消息的命名元组结构
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


# 检查文件是否具有可执行权限，如果是则返回 lint 消息，否则返回 None
def check_file(filename: str) -> LintMessage | None:
    is_executable = os.access(filename, os.X_OK)
    if is_executable:
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="executable-permissions",
            original=None,
            replacement=None,
            description="This file has executable permission; please remove it by using `chmod -x`.",
        )
    return None


# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="exec linter",
        fromfile_prefix_chars="@",
    )
    # 添加 --verbose 选项，用于控制详细输出
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    # 添加必选参数 filenames，表示需要进行 lint 的文件路径
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 配置日志输出格式和级别
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    # 存储 lint 消息的列表
    lint_messages = []
    # 遍历每个指定的文件路径进行 lint
    for filename in args.filenames:
        lint_message = check_file(filename)
        if lint_message is not None:
            lint_messages.append(lint_message)

    # 打印 lint 消息的 JSON 格式表示，确保刷新输出
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
```