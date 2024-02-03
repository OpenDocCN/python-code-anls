# `.\AutoGPT\autogpts\autogpt\autogpt\prompts\utils.py`

```py
# 导入 Any 类型用于泛型注解
from typing import Any

# 格式化带编号的列表，可以指定起始编号，默认从1开始
def format_numbered_list(items: list[Any], start_at: int = 1) -> str:
    # 使用列表推导式生成带编号的列表项字符串，每一项格式为"编号. 项内容"
    return "\n".join(f"{i}. {str(item)}" for i, item in enumerate(items, start_at)

# 缩进文本内容，可以指定缩进量，默认为4个空格
def indent(content: str, indentation: int | str = 4) -> str:
    # 如果缩进量为整数，则将其转换为对应数量的空格字符串
    if type(indentation) is int:
        indentation = " " * indentation
    # 在每个换行符后面添加相应数量的缩进空格，返回缩进后的文本内容
    return indentation + content.replace("\n", f"\n{indentation}")  # type: ignore
```