# `.\graphrag\graphrag\index\verbs\spread_json.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing spread_json method definition."""

import logging  # 导入日志模块

import pandas as pd  # 导入 pandas 库
from datashaper import TableContainer, VerbInput, verb  # 导入自定义模块

from graphrag.index.utils import is_null  # 从 graphrag.index.utils 模块导入 is_null 函数

# TODO: Check if this is already a thing
DEFAULT_COPY = ["level"]  # 默认拷贝的列名列表


@verb(name="spread_json")
def spread_json(
    input: VerbInput,  # 输入参数，类型为 VerbInput
    column: str,  # 列名参数，字符串类型
    copy: list[str] | None = None,  # 拷贝列名列表参数，默认为 None
    **_kwargs: dict,
) -> TableContainer:  # 返回类型为 TableContainer
    """
    Unpack a column containing a tuple into multiple columns.

    id|json|b
    1|{"x":5,"y":6}|b

    is converted to

    id|x|y|b
    --------
    1|5|6|b
    """
    if copy is None:  # 如果拷贝列名列表为 None，则使用默认拷贝列名列表
        copy = DEFAULT_COPY

    data = input.get_input()  # 获取输入数据

    results = []  # 初始化结果列表
    for _, row in data.iterrows():  # 遍历数据中的每一行
        try:
            cleaned_row = {col: row[col] for col in copy}  # 提取需要拷贝的列数据
            rest_row = row[column] if row[column] is not None else {}  # 提取指定列中的 JSON 数据，如果为空则为空字典

            if is_null(rest_row):  # 检查 JSON 数据是否为空
                rest_row = {}  # 如果为空则设置为空字典

            results.append({**cleaned_row, **rest_row})  # 将清理过的列数据与 JSON 数据合并，并添加到结果列表中
        except Exception:
            logging.exception("Error spreading row: %s", row)  # 记录异常情况到日志中
            raise  # 抛出异常

    data = pd.DataFrame(results, index=data.index)  # 根据结果列表创建新的 pandas DataFrame

    return TableContainer(table=data)  # 将结果封装为 TableContainer 类型并返回
```