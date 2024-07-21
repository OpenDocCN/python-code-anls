# `.\pytorch\tools\linter\adapters\lintrunner_version_linter.py`

```
# 引入 Python 未来版本的注解支持
from __future__ import annotations

# 引入处理 JSON 数据的模块
import json
# 引入执行子进程的模块
import subprocess
# 引入系统相关的模块
import sys
# 引入枚举类型支持的模块
from enum import Enum
# 引入命名元组的支持
from typing import NamedTuple

# 定义一个全局常量，用于指定 lint 错误码
LINTER_CODE = "LINTRUNNER_VERSION"

# 定义 lint 错误等级的枚举类型
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

# 定义 lint 错误信息的命名元组
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

# 定义一个函数，将版本号元组转换为字符串形式
def toVersionString(version_tuple: tuple[int, int, int]) -> str:
    return ".".join(str(x) for x in version_tuple)

# 如果作为主程序运行
if __name__ == "__main__":
    # 运行命令 `lintrunner -V` 获取 lintrunner 版本信息
    version_str = (
        subprocess.run(["lintrunner", "-V"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )

    # 导入正则表达式模块
    import re

    # 使用正则表达式匹配 lintrunner 版本号
    version_match = re.compile(r"lintrunner (\d+)\.(\d+)\.(\d+)").match(version_str)

    # 如果未匹配到版本号
    if not version_match:
        # 创建一个表示未安装 lintrunner 的错误消息对象
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description="Lintrunner is not installed, did you forget to run `make setup_lint && make lint`?",
        )
        # 退出程序
        sys.exit(0)

    # 解析当前 lintrunner 版本号
    curr_version = int(version_match[1]), int(version_match[2]), int(version_match[3])
    # 定义最低支持的 lintrunner 版本号
    min_version = (0, 10, 7)

    # 如果当前 lintrunner 版本号低于最低支持版本
    if curr_version < min_version:
        # 创建一个建议更新 lintrunner 的错误消息对象
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ADVICE,
            name="command-failed",
            original=None,
            replacement=None,
            description="".join(
                (
                    f"Lintrunner is out of date (you have v{toVersionString(curr_version)} ",
                    f"instead of v{toVersionString(min_version)}). ",
                    "Please run `pip install lintrunner -U` to update it",
                )
            ),
        )
        # 输出错误消息的 JSON 形式
        print(json.dumps(err_msg._asdict()), flush=True)
```