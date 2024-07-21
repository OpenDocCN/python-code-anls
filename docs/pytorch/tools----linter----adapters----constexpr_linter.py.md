# `.\pytorch\tools\linter\adapters\constexpr_linter.py`

```
"""
CONSTEXPR: Ensures users don't use vanilla constexpr since it causes issues
"""

# 导入所需模块和库
import argparse  # 用于解析命令行参数
import json  # 用于 JSON 数据的处理
import logging  # 日志记录模块
import sys  # 提供对 Python 解释器的访问
from enum import Enum  # 枚举类型的支持
from typing import NamedTuple  # 命名元组类型的支持

# 定义常量和变量
CONSTEXPR = "constexpr char"
CONSTEXPR_MACRO = "CONSTEXPR_EXCEPT_WIN_CUDA char"
LINTER_CODE = "CONSTEXPR"

# 定义枚举类型LintSeverity，表示检查消息的严重程度
class LintSeverity(str, Enum):
    ERROR = "error"

# 定义命名元组LintMessage，表示Lint检查的消息
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

# 定义函数check_file，用于检查文件中是否使用了CONSTEXPR
def check_file(filename: str) -> LintMessage | None:
    logging.debug("Checking file %s", filename)

    # 打开文件并读取所有行到变量lines中
    with open(filename) as f:
        lines = f.readlines()

    # 遍历文件的每一行
    for idx, line in enumerate(lines):
        if CONSTEXPR in line:
            # 如果在某行中找到了CONSTEXPR，则进行替换操作
            original = "".join(lines)
            replacement = original.replace(CONSTEXPR, CONSTEXPR_MACRO)
            logging.debug("replacement: %s", replacement)
            # 返回LintMessage对象，表示找到了CONSTEXPR的使用
            return LintMessage(
                path=filename,
                line=idx,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="Vanilla constexpr used, prefer macros",
                original=original,
                replacement=replacement,
                description="Vanilla constexpr used, prefer macros run `lintrunner --take CONSTEXPR -a` to apply changes.",
            )
    
    # 如果文件中没有发现CONSTEXPR的使用，则返回None
    return None

# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="CONSTEXPR linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    # 配置日志记录器
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET if args.verbose else logging.DEBUG if len(args.filenames) < 1000 else logging.INFO,
        stream=sys.stderr,
    )

    # 存储Lint检查结果的列表
    lint_messages = []
    # 遍历所有指定的文件路径进行Lint检查
    for filename in args.filenames:
        lint_message = check_file(filename)
        if lint_message is not None:
            lint_messages.append(lint_message)

    # 输出Lint检查结果，以JSON格式打印
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
```