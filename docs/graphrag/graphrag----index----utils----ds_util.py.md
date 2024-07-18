# `.\graphrag\graphrag\index\utils\ds_util.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A utility module datashaper-specific utility methods."""

from typing import cast

# 定义一个常量，表示需要命名输入
_NAMED_INPUTS_REQUIRED = "Named inputs are required"

# 根据给定的名称从输入中获取所需的数据表，返回一个TableContainer对象
def get_required_input_table(input: VerbInput, name: str) -> TableContainer:
    """Get a required input table by name."""
    # 使用类型转换确保返回的是TableContainer类型的对象
    return cast(TableContainer, get_named_input_table(input, name, required=True))

# 根据名称从datashaper动词输入中获取输入表格，返回TableContainer对象或者None
def get_named_input_table(
    input: VerbInput, name: str, required: bool = False
) -> TableContainer | None:
    """Get an input table from datashaper verb-inputs by name."""
    # 获取命名输入的字典
    named_inputs = input.named
    # 如果没有命名输入
    if named_inputs is None:
        # 如果不是必需的，则返回None
        if not required:
            return None
        # 如果必需但没有命名输入，则抛出异常
        raise ValueError(_NAMED_INPUTS_REQUIRED)

    # 从命名输入中获取指定名称的表格
    result = named_inputs.get(name)
    # 如果找不到指定名称的表格，并且这是一个必需的操作，则抛出异常
    if result is None and required:
        msg = f"input '${name}' is required"
        raise ValueError(msg)
    return result
```