# `.\graphrag\graphrag\index\verbs\zip.py`

```py
# 版权所有 (c) 2024 微软公司。
# 根据 MIT 许可证授权

"""定义包含 ds_zip 方法的模块。"""

# 导入必要的库和模块
from typing import cast

import pandas as pd
from datashaper import TableContainer, VerbInput, verb

# 使用装饰器定义名为 "zip" 的动词操作
@verb(name="zip")
def zip_verb(
    input: VerbInput,                           # 输入参数，用于获取数据
    to: str,                                    # 输出数据的目标列名
    columns: list[str],                         # 要合并的列的列表
    type: str | None = None,  # noqa A002       # 类型标志，默认为 None
    **_kwargs: dict,
) -> TableContainer:
    """
    将指定的列进行压缩合并。

    ## Usage
    TODO
    """

    # 将输入的表格数据转换为 pandas 的 DataFrame 格式
    table = cast(pd.DataFrame, input.get_input())

    # 如果类型标志为 None，则执行默认的合并操作
    if type is None:
        # 使用 zip 函数将指定的列并列合并，并将结果存入目标列中
        table[to] = list(zip(*[table[col] for col in columns]))

    # 如果类型标志为 "dict"，则执行特定的字典形式合并操作
    elif type == "dict":
        # 检查列数是否为两列，否则抛出异常
        if len(columns) != 2:
            msg = f"期望两列以生成字典，但得到了 {columns}"
            raise ValueError(msg)

        # 获取用作键和值的列名
        key_col, value_col = columns

        results = []
        # 遍历表格中的每一行数据
        for _, row in table.iterrows():
            keys = row[key_col]
            values = row[value_col]
            output = {}

            # 检查键值数量是否一致，不一致则抛出异常
            if len(keys) != len(values):
                msg = f"期望相同数量的键和值，但得到了 {len(keys)} 个键和 {len(values)} 个值"
                raise ValueError(msg)

            # 构建字典，将每对键值对应存入 output 字典中
            for idx, key in enumerate(keys):
                output[key] = values[idx]

            # 将构建好的字典添加到结果列表中
            results.append(output)

        # 将生成的结果列表存入目标列中
        table[to] = results

    # 将处理后的表格数据封装为 TableContainer 并返回
    return TableContainer(table=table.reset_index(drop=True))
```