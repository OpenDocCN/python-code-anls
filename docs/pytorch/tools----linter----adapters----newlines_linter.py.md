# `.\pytorch\tools\linter\adapters\newlines_linter.py`

```
"""
NEWLINE: Checks files to make sure there are no trailing newlines.
"""

# 导入所需的库和模块
from __future__ import annotations  # 允许类型注解中使用字符串形式的类型名

import argparse  # 解析命令行参数的库
import json  # 处理 JSON 格式的数据的库
import logging  # 记录日志信息的库
import sys  # 提供对 Python 解释器的访问
from enum import Enum  # 创建枚举类型的支持
from typing import NamedTuple  # 定义命名元组的支持

# 定义常量
NEWLINE = 10  # ASCII "\n"
CARRIAGE_RETURN = 13  # ASCII "\r"
LINTER_CODE = "NEWLINE"  # 代码检查标识符

# 枚举类型，定义代码检查的严重程度
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

# 命名元组，定义代码检查的消息结构
class LintMessage(NamedTuple):
    path: str | None  # 文件路径或者为 None
    line: int | None  # 行号或者为 None
    char: int | None  # 字符位置或者为 None
    code: str  # 代码检查标识符
    severity: LintSeverity  # 代码检查严重程度
    name: str  # 消息名称
    original: str | None  # 原始内容或者为 None
    replacement: str | None  # 替换内容或者为 None
    description: str | None  # 描述信息或者为 None

# 检查文件中是否存在尾随的换行符
def check_file(filename: str) -> LintMessage | None:
    logging.debug("Checking file %s", filename)  # 记录调试信息，检查文件名

    with open(filename, "rb") as f:
        lines = f.readlines()  # 读取文件的所有行并存储在列表中

    if len(lines) == 0:
        # 文件为空，不做处理
        return None

    if len(lines) == 1 and len(lines[0]) == 1:
        # 文件只有一个字节且为换行符，报告错误
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="testestTrailing newline",  # 消息名称
            original=None,
            replacement=None,
            description="Trailing newline found. Run `lintrunner --take NEWLINE -a` to apply changes.",
        )

    if len(lines[-1]) == 1 and lines[-1][0] == NEWLINE:
        try:
            original = b"".join(lines).decode("utf-8")  # 尝试将字节串连接成字符串并解码为 UTF-8
        except Exception as err:
            # 解码失败，报告错误
            return LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="Decoding failure",  # 消息名称
                original=None,
                replacement=None,
                description=f"utf-8 decoding failed due to {err.__class__.__name__}:\n{err}",
            )

        # 发现尾随的换行符，报告错误，并提供修复建议
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="Trailing newline",  # 消息名称
            original=original,
            replacement=original.rstrip("\n") + "\n",
            description="Trailing newline found. Run `lintrunner --take NEWLINE -a` to apply changes.",
        )

    has_changes = False
    original_lines: list[bytes] | None = None  # 原始文件行的备份
    for idx, line in enumerate(lines):
        if len(line) >= 2 and line[-1] == NEWLINE and line[-2] == CARRIAGE_RETURN:
            if not has_changes:
                original_lines = list(lines)  # 备份原始文件行
                has_changes = True
            lines[idx] = line[:-2] + b"\n"  # 移除尾随的换行符和回车符并添加换行符
    # 如果存在变更，执行以下逻辑
    if has_changes:
        try:
            # 断言原始行不为None
            assert original_lines is not None
            # 将原始行连接为字节串，并使用utf-8解码为字符串
            original = b"".join(original_lines).decode("utf-8")
            # 将变更后的行连接为字节串，并使用utf-8解码为字符串
            replacement = b"".join(lines).decode("utf-8")
        except Exception as err:
            # 如果解码失败，返回一个LintMessage对象，指示解码错误
            return LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="Decoding failure",
                original=None,
                replacement=None,
                description=f"utf-8 decoding failed due to {err.__class__.__name__}:\n{err}",
            )
        # 返回一个LintMessage对象，指示发现DOS换行符，并提供变更建议
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="DOS newline",
            original=original,
            replacement=replacement,
            description="DOS newline found. Run `lintrunner --take NEWLINE -a` to apply changes.",
        )

    # 如果没有变更，返回None
    return None
if __name__ == "__main__":
    # 检查当前模块是否作为主程序运行

    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    # 创建命令行参数解析器，描述为“native functions linter”，支持文件前缀字符“@”

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="location of native_functions.yaml",
    )
    # 添加命令行选项"--verbose"，如果存在则设置为True，用以开启详细输出模式，显示"location of native_functions.yaml"帮助信息

    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    # 添加位置参数"filenames"，要求至少提供一个文件路径，用以进行代码检查

    args = parser.parse_args()
    # 解析命令行参数并将其存储在args对象中

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )
    # 配置日志记录基本设置：格式为"<线程名:日志级别> 日志消息"，日志级别根据命令行参数决定：
    # - 如果--verbose选项存在，则设置为NOTSET级别
    # - 否则，如果提供的文件路径数小于1000，则设置为DEBUG级别
    # - 否则，设置为INFO级别
    # 所有日志消息输出到标准错误流sys.stderr

    lint_messages = []
    # 初始化空列表lint_messages，用于存储代码检查的消息

    for filename in args.filenames:
        # 遍历每个命令行参数中的文件路径

        lint_message = check_file(filename)
        # 对当前文件进行代码检查，返回检查结果消息

        if lint_message is not None:
            # 如果检查结果消息不为None（即有问题的消息）

            lint_messages.append(lint_message)
            # 将该消息添加到lint_messages列表中

    for lint_message in lint_messages:
        # 遍历lint_messages列表中的每条检查结果消息

        print(json.dumps(lint_message._asdict()), flush=True)
        # 将每条消息转换为字典形式，然后以JSON格式打印输出，同时刷新输出缓冲区
```