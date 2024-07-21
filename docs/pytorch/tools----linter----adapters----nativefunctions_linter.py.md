# `.\pytorch\tools\linter\adapters\nativefunctions_linter.py`

```py
"""
Verify that it is possible to round-trip native_functions.yaml via ruamel under some
configuration.  Keeping native_functions.yaml consistent in this way allows us to
run codemods on the file using ruamel without introducing line noise.  Note that we don't
want to normalize the YAML file, as that would to lots of spurious lint failures.  Anything
that ruamel understands how to roundtrip, e.g., whitespace and comments, is OK!

ruamel is a bit picky about inconsistent indentation, so you will have to indent your
file properly.  Also, if you are working on changing the syntax of native_functions.yaml,
you may find that you want to use some format that is not what ruamel prefers.  If so,
it is OK to modify this script (instead of reformatting native_functions.yaml)--the point
is simply to make sure that there is *some* configuration of ruamel that can round trip
the YAML, not to be prescriptive about it.
"""

from __future__ import annotations

import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import sys  # 导入与系统交互的模块
from enum import Enum  # 导入用于定义枚举的模块
from io import StringIO  # 导入用于在内存中创建文件对象的模块
from typing import NamedTuple  # 导入用于创建命名元组的模块

import ruamel.yaml  # 导入 ruamel.yaml 模块，用于处理 YAML 文件

class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",  # 解析器描述
        fromfile_prefix_chars="@",  # 指定从文件加载参数的前缀字符
    )
    parser.add_argument(
        "--native-functions-yml",  # 参数名
        required=True,  # 参数是否必需
        help="location of native_functions.yaml",  # 参数帮助信息
    )

    args = parser.parse_args()  # 解析命令行参数

    with open(args.native_functions_yml) as f:
        contents = f.read()  # 读取 native_functions.yaml 文件内容

    yaml = ruamel.yaml.YAML()  # 创建 ruamel.yaml 的 YAML 对象
    yaml.preserve_quotes = True  # 设置保留引号选项
    yaml.width = 1000  # 设置 YAML 输出宽度
    yaml.boolean_representation = ["False", "True"]  # 设置布尔值表示形式

    try:
        r = yaml.load(contents)  # 尝试加载 YAML 文件内容
    except Exception as err:
        msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code="NATIVEFUNCTIONS",
            severity=LintSeverity.ERROR,
            name="YAML load failure",
            original=None,
            replacement=None,
            description=f"Failed due to {err.__class__.__name__}:\n{err}",
        )
        print(json.dumps(msg._asdict()), flush=True)  # 打印加载失败的错误信息
        sys.exit(0)  # 退出程序

    # Cuz ruamel's author intentionally didn't include conversion to string
    # https://stackoverflow.com/questions/47614862/best-way-to-use-ruamel-yaml-to-dump-to-string-not-to-stream
    string_stream = StringIO()  # 创建一个内存中的字符串流对象
    yaml.dump(r, string_stream)  # 将加载的 YAML 数据转储到字符串流中
    new_contents = string_stream.getvalue()  # 获取字符串流中的内容
    string_stream.close()  # 关闭字符串流
    # 如果 contents 和 new_contents 不相等，则创建一个 lint 消息对象
    msg = LintMessage(
        # 设置消息对象的路径为 args.native_functions_yml
        path=args.native_functions_yml,
        # 行号为空
        line=None,
        # 字符位置为空
        char=None,
        # 错误代码为 "NATIVEFUNCTIONS"
        code="NATIVEFUNCTIONS",
        # 严重性设置为 ERROR
        severity=LintSeverity.ERROR,
        # 错误名称为 "roundtrip inconsistency"
        name="roundtrip inconsistency",
        # 原始内容为 contents
        original=contents,
        # 替换内容为 new_contents
        replacement=new_contents,
        # 错误描述信息包括如何修复以及如何联系相关工具
        description=(
            "YAML roundtrip failed; run `lintrunner --take NATIVEFUNCTIONS -a` to apply the suggested changes. "
            "If you think this is in error, please see tools/linter/adapters/nativefunctions_linter.py"
        ),
    )

    # 将消息对象转换为 JSON 格式，并输出到标准输出，刷新缓冲区
    print(json.dumps(msg._asdict()), flush=True)
```