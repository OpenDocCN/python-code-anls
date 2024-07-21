# `.\pytorch\torchgen\code_template.py`

```py
from __future__ import annotations

import re
from typing import Mapping, Sequence


# 匹配 $identifier 或 ${identifier} 并替换为环境中的值
# 如果该标识符位于行首的空白处，并且其值是列表，则视为块替换，根据缩进深度将列表元素单独放置在每一行
# 如果标识符位于非空白开头的行上，并且其值是列表，则用逗号分隔元素
# ${,foo} 在列表前插入逗号（如果列表非空），${foo,} 在列表后插入逗号（如果列表非空）


class CodeTemplate:
    # 匹配模板中的替换字符串，支持变量名或者带大括号的变量名
    substitution_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    substitution = re.compile(substitution_str, re.MULTILINE)

    pattern: str  # 模板内容
    filename: str  # 文件名（可选）

    @staticmethod
    def from_file(filename: str) -> CodeTemplate:
        # 从文件中读取模板内容并返回 CodeTemplate 实例
        with open(filename) as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str = "") -> None:
        # 初始化 CodeTemplate 实例
        self.pattern = pattern
        self.filename = filename

    def substitute(
        self, env: Mapping[str, object] | None = None, **kwargs: object
    ) -> str:
        # 执行模板内容的变量替换
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            # 查找变量在环境中的值
            assert env is not None
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            # 根据给定的缩进和值序列，生成缩进后的多行字符串
            return "".join(
                [indent + l + "\n" for e in v for l in str(e).splitlines()]
            ).rstrip()

        def replace(match: re.Match[str]) -> str:
            # 根据正则匹配结果替换模板中的变量
            indent = match.group(1)
            key = match.group(2)
            comma_before = ""
            comma_after = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ", "
                    key = key[1:]
                if key[-1] == ",":
                    comma_after = ", "
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ", ".join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)

        return self.substitution.sub(replace, self.pattern)


if __name__ == "__main__":
    c = CodeTemplate(
        """\
    int foo($args) {

        $bar
            $bar
        $a+$b
    }
    int commatest(int a${,stuff})
    int notest(int a${,empty,})
    """
    )
    print(
        c.substitute(
            args=["hi", 8],
            bar=["what", 7],
            a=3,
            b=4,
            stuff=["things...", "others"],
            empty=[],
        )
    )
```