# `.\graphrag\graphrag\index\verbs\overrides\merge.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing merge and _merge_json methods definition."""

# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
import logging  # 导入日志记录模块
from enum import Enum  # 导入枚举类型支持
from typing import Any, cast  # 导入类型提示支持

import pandas as pd  # 导入 pandas 库
from datashaper import TableContainer, VerbInput, VerbResult, verb  # 导入数据整形相关模块和类
from datashaper.engine.verbs.merge import merge as ds_merge  # 导入数据整形中的 merge 函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class MergeStrategyType(str, Enum):
    """MergeStrategy class definition."""
    
    json = "json"  # JSON 合并策略
    datashaper = "datashaper"  # 数据整形合并策略

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'  # 返回枚举值的字符串表示形式


# TODO: This thing is kinda gross
# Also, it diverges from the original aggregate verb, since it doesn't support the same syntax
@verb(name="merge_override")
def merge(
    input: VerbInput,
    to: str,
    columns: list[str],
    strategy: MergeStrategyType = MergeStrategyType.datashaper,
    delimiter: str = "",
    preserveSource: bool = False,  # noqa N806
    unhot: bool = False,
    prefix: str = "",
    **_kwargs: dict,
) -> TableContainer | VerbResult:
    """Merge method definition."""
    output: pd.DataFrame  # 输出结果为 pandas DataFrame 类型
    match strategy:  # 根据策略类型进行匹配
        case MergeStrategyType.json:
            output = _merge_json(input, to, columns)  # 调用 JSON 合并方法
            filtered_list: list[str] = []  # 初始化空的过滤列列表

            for col in output.columns:  # 遍历输出 DataFrame 的列
                try:
                    columns.index(col)  # 在输入的列中查找当前列
                except ValueError:
                    log.exception("Column %s not found in input columns", col)  # 如果列不存在，记录异常信息
                    filtered_list.append(col)  # 将不存在的列名添加到过滤列表中

            if not preserveSource:  # 如果不保留原始数据
                output = cast(Any, output[filtered_list])  # 只保留过滤后的列数据
            return TableContainer(table=output.reset_index())  # 返回重置索引后的 TableContainer 对象
        case _:  # 对于其他未指定的合并策略
            return ds_merge(  # 调用数据整形中的 merge 函数
                input, to, columns, strategy, delimiter, preserveSource, unhot, prefix
            )


def _merge_json(
    input: VerbInput,
    to: str,
    columns: list[str],
) -> pd.DataFrame:
    input_table = cast(pd.DataFrame, input.get_input())  # 获取输入数据并转换为 DataFrame
    output = input_table  # 将输入数据赋值给输出
    output[to] = output[columns].apply(
        lambda row: ({**row}),  # 对指定列进行操作，生成新的合并结果
        axis=1,
    )
    return output  # 返回处理后的 DataFrame
```