# `.\graphrag\graphrag\index\verbs\snapshot_rows.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'FormatSpecifier' model."""

import json  # 导入处理 JSON 的模块
from dataclasses import dataclass  # 导入用于创建数据类的装饰器
from typing import Any  # 导入用于类型提示的通用 Any 类型

from datashaper import TableContainer, VerbInput, verb  # 导入数据处理相关模块

from graphrag.index.storage import PipelineStorage  # 导入管道存储相关模块


@dataclass
class FormatSpecifier:
    """Format specifier class definition."""

    format: str  # 格式类型，例如 'json', 'text'
    extension: str  # 文件扩展名，例如 'json', 'txt'


@verb(name="snapshot_rows")
async def snapshot_rows(
    input: VerbInput,  # 输入数据对象
    column: str | None,  # 要处理的列名，可选
    base_name: str,  # 基本名称，用于生成文件名
    storage: PipelineStorage,  # 存储对象，用于存储快照数据
    formats: list[str | dict[str, Any]],  # 快照格式列表，支持字符串或字典格式
    row_name_column: str | None = None,  # 行名列名，用于生成每行的名称，可选
    **_kwargs: dict,  # 其他关键字参数
) -> TableContainer:
    """Take a by-row snapshot of the tabular data."""
    data = input.get_input()  # 获取输入数据
    parsed_formats = _parse_formats(formats)  # 解析快照格式列表

    num_rows = len(data)  # 获取数据行数

    def get_row_name(row: Any, row_idx: Any):
        """生成行名函数."""
        if row_name_column is None:  # 如果未指定行名列
            if num_rows == 1:
                return base_name  # 只有一行数据时使用基本名称
            return f"{base_name}.{row_idx}"  # 使用基本名称加行索引
        return f"{base_name}.{row[row_name_column]}"  # 使用基本名称加行名列的值作为行名

    for row_idx, row in data.iterrows():  # 遍历数据的每一行
        for fmt in parsed_formats:  # 遍历解析后的快照格式
            row_name = get_row_name(row, row_idx)  # 获取当前行的名称
            extension = fmt.extension  # 获取当前格式的文件扩展名
            if fmt.format == "json":  # 如果格式为 JSON
                await storage.set(  # 使用存储对象存储数据
                    f"{row_name}.{extension}",  # 构建文件名
                    json.dumps(row[column]) if column is not None else json.dumps(row.to_dict()),  # 转换为 JSON 字符串
                )
            elif fmt.format == "text":  # 如果格式为文本
                if column is None:
                    msg = "column must be specified for text format"
                    raise ValueError(msg)  # 抛出值错误异常，要求必须指定列名
                await storage.set(f"{row_name}.{extension}", str(row[column]))  # 存储文本数据

    return TableContainer(table=data)  # 返回包含原始数据的表格容器对象


def _parse_formats(formats: list[str | dict[str, Any]]) -> list[FormatSpecifier]:
    """解析格式列表为 FormatSpecifier 对象列表."""
    return [
        FormatSpecifier(**fmt) if isinstance(fmt, dict) else FormatSpecifier(format=fmt, extension=_get_format_extension(fmt))
        for fmt in formats
    ]


def _get_format_extension(fmt: str) -> str:
    """获取给定格式的文件扩展名."""
    if fmt == "json":
        return "json"
    if fmt == "text":
        return "txt"
    if fmt == "parquet":
        return "parquet"
    if fmt == "csv":
        return "csv"
    msg = f"Unknown format: {fmt}"  # 未知格式时抛出值错误异常
    raise ValueError(msg)
```