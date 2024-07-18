# `.\graphrag\graphrag\index\verbs\genid.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing genid method definition."""

# 导入所需的模块和函数
from typing import cast

import pandas as pd
from datashaper import TableContainer, VerbInput, verb

# 从自定义模块中导入函数
from graphrag.index.utils import gen_md5_hash


# 使用装饰器定义名为 "genid" 的函数
@verb(name="genid")
def genid(
    input: VerbInput,
    to: str,
    method: str = "md5_hash",
    hash: list[str] = [],  # noqa A002
    **_kwargs: dict,
) -> TableContainer:
    """
    Generate a unique id for each row in the tabular data.

    ## Usage
    ### json
    ```json
    {
        "verb": "genid",
        "args": {
            "to": "id_output_column_name", /* The name of the column to output the id to */
            "method": "md5_hash", /* The method to use to generate the id */
            "hash": ["list", "of", "column", "names"] /* only if using md5_hash */,
            "seed": 034324 /* The random seed to use with UUID */
        }
    }
    ```py

    ### yaml
    ```yaml
    verb: genid
    args:
        to: id_output_column_name
        method: md5_hash
        hash:
            - list
            - of
            - column
            - names
        seed: 034324
    ```py
    """
    
    # 将输入数据转换为 pandas DataFrame
    data = cast(pd.DataFrame, input.source.table)

    # 根据方法选择生成唯一 ID 的方式
    if method == "md5_hash":
        # 如果方法为 md5_hash，则确保 hash 列表不为空
        if len(hash) == 0:
            msg = 'Must specify the "hash" columns to use md5_hash method'
            raise ValueError(msg)

        # 对数据应用 gen_md5_hash 函数生成唯一 ID，并赋值给指定的列 'to'
        data[to] = data.apply(lambda row: gen_md5_hash(row, hash), axis=1)
    elif method == "increment":
        # 如果方法为 increment，则直接使用行索引增量作为 ID，并赋值给指定的列 'to'
        data[to] = data.index + 1
    else:
        # 如果方法未知，则抛出异常
        msg = f"Unknown method {method}"
        raise ValueError(msg)
    
    # 将处理后的数据封装成 TableContainer 对象并返回
    return TableContainer(table=data)
```