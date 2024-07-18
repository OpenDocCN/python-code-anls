# `.\graphrag\graphrag\index\verbs\text\split.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the text_split method definition."""

# 导入必要的模块和类
from typing import cast

import pandas as pd
from datashaper import TableContainer, VerbInput, verb

# 定义一个名为"text_split"的动词，用于将文本根据分隔符拆分为字符串列表，并输出包含拆分文本的新列
@verb(name="text_split")
def text_split(
    input: VerbInput,
    column: str,
    to: str,
    separator: str = ",",
    **_kwargs: dict,
) -> TableContainer:
    """
    Split a piece of text into a list of strings based on a delimiter. The verb outputs a new column containing a list of strings.

    ## Usage

    ```yaml
    verb: text_split
    args:
        column: text # The name of the column containing the text to split
        to: split_text # The name of the column to output the split text to
        separator: "," # The separator to split the text on, defaults to ","
    ```py
    """
    # 调用text_split_df函数处理输入数据框架，执行文本拆分操作，并将结果封装到TableContainer中返回
    output = text_split_df(cast(pd.DataFrame, input.get_input()), column, to, separator)
    return TableContainer(table=output)


def text_split_df(
    input: pd.DataFrame, column: str, to: str, separator: str = ","
) -> pd.DataFrame:
    """Split a column into a list of strings."""
    # 将输入数据框架赋值给输出
    output = input

    # 定义内部函数_apply_split，用于将指定列的文本根据分隔符拆分为字符串列表
    def _apply_split(row):
        # 如果指定列的值为空或者已经是列表，则直接返回该值
        if row[column] is None or isinstance(row[column], list):
            return row[column]
        # 如果指定列的值为空字符串，则返回空列表
        if row[column] == "":
            return []
        # 如果指定列的值不是字符串类型，则抛出类型错误异常
        if not isinstance(row[column], str):
            message = f"Expected {column} to be a string, but got {type(row[column])}"
            raise TypeError(message)
        # 使用指定分隔符拆分指定列的值，并返回拆分后的字符串列表
        return row[column].split(separator)

    # 将拆分后的结果存储到输出数据框架的新列中
    output[to] = output.apply(_apply_split, axis=1)
    return output
```