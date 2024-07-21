# `.\pytorch\tools\linter\adapters\testowners_linter.py`

```
# 指定 Python 解释器路径为 /usr/bin/env python3
#!/usr/bin/env python3

# 以下是脚本的简要描述和链接引用
"""
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

This lint verifies that every Python test file (file that matches test_*.py or *_test.py in the test folder)
has valid ownership information in a comment header. Valid means:
  - The format of the header follows the pattern "# Owner(s): ["list", "of owner", "labels"]
  - Each owner label actually exists in PyTorch
  - Each owner label starts with "module: " or "oncall: " or is in ACCEPTABLE_OWNER_LABELS
"""

# 引入未来版本的特性，支持类型注解
from __future__ import annotations

# 引入必要的库
import argparse
import json
from enum import Enum
from typing import Any, NamedTuple
from urllib.request import urlopen

# 定义常量，表示此 lint 的代码标识为 "TESTOWNERS"
LINTER_CODE = "TESTOWNERS"

# 枚举类定义 lint 的严重程度
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

# 定义 NamedTuple 类型 LintMessage，表示 lint 结果的消息
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

# 定义可接受的所有者标签的列表，通常以 "module: " 或 "oncall: " 开头
ACCEPTABLE_OWNER_LABELS = ["NNC", "high priority"]

# 代码文件中有效的所有者信息标识前缀
OWNERS_PREFIX = "# Owner(s): "

# 获取 PyTorch 的标签信息
def get_pytorch_labels() -> Any:
    labels = (
        urlopen("https://ossci-metrics.s3.amazonaws.com/pytorch_labels.json")
        .read()
        .decode("utf-8")
    )
    return json.loads(labels)

# 调用函数获取 PyTorch 标签信息
PYTORCH_LABELS = get_pytorch_labels()

# 再次定义可接受的所有者标签的列表（通常以 "module: " 或 "oncall: " 开头）
ACCEPTABLE_OWNER_LABELS = ["NNC", "high priority"]

# 全局的异常文件列表，这些文件不需要进行所有者信息的 lint
GLOB_EXCEPTIONS = ["**/test/run_test.py"]

# 检查标签函数，验证文件的所有者信息
def check_labels(
    labels: list[str], filename: str, line_number: int
) -> list[LintMessage]:
    lint_messages = []
    # 遍历给定的标签列表
    for label in labels:
        # 检查标签是否不在预定义的PyTorch标签列表中
        if label not in PYTORCH_LABELS:
            # 如果标签无效，添加LintMessage到lint_messages列表，指示错误
            lint_messages.append(
                LintMessage(
                    path=filename,                    # 错误出现的文件路径
                    line=line_number,                 # 错误出现的行号
                    char=None,                        # 错误位置的字符（此处无具体字符）
                    code=LINTER_CODE,                 # Linter的错误代码
                    severity=LintSeverity.ERROR,      # 错误严重程度为错误
                    name="[invalid-label]",           # 错误名称标识为无效标签
                    original=None,                    # 原始内容（此处无具体内容）
                    replacement=None,                 # 替换内容（此处无需替换）
                    description=(
                        f"{label} is not a PyTorch label "  # 错误描述，指出标签不是PyTorch的标签
                        "(please choose from https://github.com/pytorch/pytorch/labels)"
                    ),
                )
            )

        # 检查标签是否以"module:"或"oncall:"开头，或者是否在可接受的所有者标签列表中
        if label.startswith(("module:", "oncall:")) or label in ACCEPTABLE_OWNER_LABELS:
            # 如果是可接受的所有者标签，继续下一个标签的检查
            continue

        # 如果标签不是合适的所有者标签，添加LintMessage到lint_messages列表，指示错误
        lint_messages.append(
            LintMessage(
                path=filename,                    # 错误出现的文件路径
                line=line_number,                 # 错误出现的行号
                char=None,                        # 错误位置的字符（此处无具体字符）
                code=LINTER_CODE,                 # Linter的错误代码
                severity=LintSeverity.ERROR,      # 错误严重程度为错误
                name="[invalid-owner]",           # 错误名称标识为无效所有者
                original=None,                    # 原始内容（此处无具体内容）
                replacement=None,                 # 替换内容（此处无需替换）
                description=(
                    f"{label} is not an acceptable owner "  # 错误描述，指出标签不是合适的所有者标签
                    "(please update to another label or edit ACCEPTABLE_OWNERS_LABELS "
                    "in tools/linters/adapters/testowners_linter.py"
                ),
            )
        )

    return lint_messages
# 检查给定文件中的lint信息，并返回lint消息列表
def check_file(filename: str) -> list[LintMessage]:
    # 初始化lint消息列表
    lint_messages = []
    # 标记是否有所有权信息的标志位，默认为False
    has_ownership_info = False

    # 使用上下文管理器打开文件
    with open(filename) as f:
        # 逐行遍历文件内容
        for idx, line in enumerate(f):
            # 如果当前行不以OWNERS_PREFIX开头，则跳过当前循环
            if not line.startswith(OWNERS_PREFIX):
                continue

            # 如果找到以OWNERS_PREFIX开头的行，则表示有所有权信息
            has_ownership_info = True
            # 解析JSON格式的所有权信息，并调用检查函数，将返回的lint消息扩展到lint_messages列表中
            labels = json.loads(line[len(OWNERS_PREFIX):])
            lint_messages.extend(check_labels(labels, filename, idx + 1))

    # 如果未找到任何所有权信息，则添加一条lint消息，指示缺少所有权信息的错误
    if has_ownership_info is False:
        lint_messages.append(
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[no-owner-info]",
                original=None,
                replacement=None,
                description="Missing a comment header with ownership information.",
            )
        )

    # 返回lint消息列表
    return lint_messages


# 主函数，用于解析命令行参数并执行lint检查
def main() -> None:
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="test ownership linter",
        fromfile_prefix_chars="@",
    )
    # 添加位置参数，指定需要进行lint检查的文件路径列表
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 初始化lint消息列表
    lint_messages = []

    # 遍历每个指定的文件路径，执行lint检查并将返回的lint消息扩展到lint_messages列表中
    for filename in args.filenames:
        lint_messages.extend(check_file(filename))

    # 遍历lint_messages列表，将每条lint消息以JSON格式输出到标准输出，刷新缓冲区
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


# 如果当前脚本作为主程序执行，则调用main函数
if __name__ == "__main__":
    main()
```