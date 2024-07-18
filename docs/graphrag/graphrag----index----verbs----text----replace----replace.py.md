# `.\graphrag\graphrag\index\verbs\text\replace\replace.py`

```py
# 版权声明及许可信息

"""包含 replace 和 _apply_replacements 方法的模块。"""

# 引入必要的模块和类型提示
from typing import cast

import pandas as pd
from datashaper import TableContainer, VerbInput, verb

# 引入自定义类型 Replacement
from .typing import Replacement


# 定义名为 'text_replace' 的动词（verb）函数
@verb(name="text_replace")
def text_replace(
    input: VerbInput,
    column: str,
    to: str,
    replacements: list[dict[str, str]],
    **_kwargs: dict,
) -> TableContainer:
    """
    应用一组替换规则到文本上。

    ## 用法
    ```yaml
    verb: text_replace
    args:
        column: <column name> # 包含待替换文本的列名
        to: <column name> # 写入替换后文本的列名
        replacements: # 需要应用的替换规则列表
            - pattern: <string> # 要查找的正则表达式模式
              replacement: <string> # 替换的字符串
    ```py
    """
    # 将输入转换为 pandas DataFrame 格式
    output = cast(pd.DataFrame, input.get_input())
    # 解析替换规则列表为 Replacement 对象的列表
    parsed_replacements = [Replacement(**r) for r in replacements]
    # 在输出 DataFrame 中新增一列，并应用 _apply_replacements 函数进行替换
    output[to] = output[column].apply(
        lambda text: _apply_replacements(text, parsed_replacements)
    )
    # 返回替换后的结果作为 TableContainer 封装的表格
    return TableContainer(table=output)


# 私有函数，用于应用替换规则到给定文本
def _apply_replacements(text: str, replacements: list[Replacement]) -> str:
    # 遍历替换规则列表，逐个替换文本中的匹配项
    for r in replacements:
        text = text.replace(r.pattern, r.replacement)
    # 返回替换后的文本
    return text
```