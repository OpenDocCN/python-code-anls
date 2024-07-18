# `.\graphrag\graphrag\query\input\loaders\utils.py`

```py
# 版权所有 (c) 2024 Microsoft Corporation.
# 根据 MIT 许可证授权

"""数据加载工具."""

# 导入必要的库
import numpy as np
import pandas as pd

# 将值转换并验证为字符串
def to_str(data: pd.Series, column_name: str | None) -> str:
    """Convert and validate a value to a string."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        return str(data[column_name])
    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


# 将值转换并验证为可选字符串
def to_optional_str(data: pd.Series, column_name: str | None) -> str | None:
    """Convert and validate a value to an optional string."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        value = data[column_name]
        if value is None:
            return None
        return str(data[column_name])
    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


# 将值转换并验证为列表
def to_list(
    data: pd.Series, column_name: str | None, item_type: type | None = None
) -> list:
    """Convert and validate a value to a list."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    if column_name in data:
        value = data[column_name]
        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, list):
            msg = f"value is not a list: {value} ({type(value)})"
            raise ValueError(msg)

        if item_type is not None:
            for v in value:
                if not isinstance(v, item_type):
                    msg = f"list item has item that is not {item_type}: {v} ({type(v)})"
                    raise TypeError(msg)
        return value

    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


# 将值转换并验证为可选列表
def to_optional_list(
    data: pd.Series, column_name: str | None, item_type: type | None = None
) -> list | None:
    """Convert and validate a value to an optional list."""
    if column_name is None:
        return None

    if column_name in data:
        value = data[column_name]  # type: ignore
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, list):
            msg = f"value is not a list: {value} ({type(value)})"
            raise ValueError(msg)

        if item_type is not None:
            for v in value:
                if not isinstance(v, item_type):
                    msg = f"list item has item that is not {item_type}: {v} ({type(v)})"
                    raise TypeError(msg)
        return value

    return None


# 将值转换并验证为整数
def to_int(data: pd.Series, column_name: str | None) -> int:
    """Convert and validate a value to an int."""
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)
    # 检查列名是否存在于数据中
    if column_name in data:
        # 如果列名存在，获取对应的数值
        value = data[column_name]
        
        # 如果数值是浮点数，转换为整数类型
        if isinstance(value, float):
            value = int(value)
        
        # 检查数值是否为整数类型，如果不是，抛出数值错误异常
        if not isinstance(value, int):
            msg = f"value is not an int: {value} ({type(value)})"
            raise ValueError(msg)
    else:
        # 如果列名不存在于数据中，抛出列未找到的数值错误异常
        msg = f"Column {column_name} not found in data"
        raise ValueError(msg)

    # 返回转换为整数类型的数值
    return int(value)
# 将数据系列中的值转换并验证为可选的整数
def to_optional_int(data: pd.Series, column_name: str | None) -> int | None:
    """Convert and validate a value to an optional int."""
    # 如果列名为None，返回None
    if column_name is None:
        return None

    # 如果列名存在于数据中
    if column_name in data:
        # 获取列名对应的值
        value = data[column_name]

        # 如果值为None，返回None
        if value is None:
            return None

        # 如果值是浮点数，转换为整数
        if isinstance(value, float):
            value = int(value)

        # 如果值不是整数，抛出值错误异常
        if not isinstance(value, int):
            msg = f"value is not an int: {value} ({type(value)})"
            raise ValueError(msg)
    else:
        # 如果列名不存在于数据中，抛出值错误异常
        msg = f"Column {column_name} not found in data"
        raise ValueError(msg)

    # 返回转换后的整数值
    return int(value)


# 将数据系列中的值转换并验证为浮点数
def to_float(data: pd.Series, column_name: str | None) -> float:
    """Convert and validate a value to a float."""
    # 如果列名为None，抛出值错误异常
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    # 如果列名存在于数据中
    if column_name in data:
        # 获取列名对应的值
        value = data[column_name]

        # 如果值不是浮点数，抛出值错误异常
        if not isinstance(value, float):
            msg = f"value is not a float: {value} ({type(value)})"
            raise ValueError(msg)
    else:
        # 如果列名不存在于数据中，抛出值错误异常
        msg = f"Column {column_name} not found in data"
        raise ValueError(msg)

    # 返回转换后的浮点数值
    return float(value)


# 将数据系列中的值转换并验证为可选的浮点数
def to_optional_float(data: pd.Series, column_name: str | None) -> float | None:
    """Convert and validate a value to an optional float."""
    # 如果列名为None，返回None
    if column_name is None:
        return None

    # 如果列名存在于数据中
    if column_name in data:
        # 获取列名对应的值
        value = data[column_name]

        # 如果值为None，返回None
        if value is None:
            return None

        # 如果值不是浮点数，抛出值错误异常
        if not isinstance(value, float):
            msg = f"value is not a float: {value} ({type(value)})"
            raise ValueError(msg)
    else:
        # 如果列名不存在于数据中，抛出值错误异常
        msg = f"Column {column_name} not found in data"
        raise ValueError(msg)

    # 返回转换后的浮点数值
    return float(value)


# 将数据系列中的值转换并验证为字典
def to_dict(
    data: pd.Series,
    column_name: str | None,
    key_type: type | None = None,
    value_type: type | None = None,
) -> dict:
    """Convert and validate a value to a dict."""
    # 如果列名为None，抛出值错误异常
    if column_name is None:
        msg = "Column name is None"
        raise ValueError(msg)

    # 如果列名存在于数据中
    if column_name in data:
        # 获取列名对应的值
        value = data[column_name]

        # 如果值不是字典，抛出值错误异常
        if not isinstance(value, dict):
            msg = f"value is not a dict: {value} ({type(value)})"
            raise ValueError(msg)

        # 如果指定了键类型
        if key_type is not None:
            # 遍历字典的键
            for v in value:
                # 如果键的类型不是指定的类型，抛出类型错误异常
                if not isinstance(v, key_type):
                    msg = f"dict key has item that is not {key_type}: {v} ({type(v)})"
                    raise TypeError(msg)

        # 如果指定了值类型
        if value_type is not None:
            # 遍历字典的值
            for v in value.values():
                # 如果值的类型不是指定的类型，抛出类型错误异常
                if not isinstance(v, value_type):
                    msg = f"dict value has item that is not {value_type}: {v} ({type(v)})"
                    raise TypeError(msg)
        
        # 返回验证通过的字典值
        return value

    # 如果列名不存在于数据中，抛出值错误异常
    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)


# 将数据系列中的值转换并验证为可选的字典
def to_optional_dict(
    data: pd.Series,
    column_name: str | None,
    key_type: type | None = None,
    value_type: type | None = None,
) -> dict | None:
    """Convert and validate a value to an optional dict."""
    # 如果列名为None，返回None
    if column_name is None:
        return None

    # 如果列名存在于数据中
    if column_name in data:
        # 获取列名对应的值
        value = data[column_name]

        # 如果值为None，返回None
        if value is None:
            return None

        # 如果值不是字典，抛出值错误异常
        if not isinstance(value, dict):
            msg = f"value is not a dict: {value} ({type(value)})"
            raise ValueError(msg)

        # 如果指定了键类型
        if key_type is not None:
            # 遍历字典的键
            for v in value:
                # 如果键的类型不是指定的类型，抛出类型错误异常
                if not isinstance(v, key_type):
                    msg = f"dict key has item that is not {key_type}: {v} ({type(v)})"
                    raise TypeError(msg)

        # 如果指定了值类型
        if value_type is not None:
            # 遍历字典的值
            for v in value.values():
                # 如果值的类型不是指定的类型，抛出类型错误异常
                if not isinstance(v, value_type):
                    msg = f"dict value has item that is not {value_type}: {v} ({type(v)})"
                    raise TypeError(msg)
        
        # 返回验证通过的字典值
        return value

    # 如果列名不存在于数据中，抛出值错误异常
    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)
    value_type: type | None = None,


    # 定义变量 value_type，用于表示值的类型，默认为 None
    # 这里使用了类型注解，指定了 value_type 的类型可以是 type 或者 NoneType
# 定义函数，将指定列名对应的值转换并验证为可选的字典类型
def convert_to_optional_dict(
    column_name: str, data: dict, key_type=None, value_type=None
) -> dict | None:
    """Convert and validate a value to an optional dict."""
    
    # 如果列名为 None，则返回 None
    if column_name is None:
        return None

    # 如果数据中存在指定的列名
    if column_name in data:
        # 获取列名对应的值
        value = data[column_name]
        
        # 如果值为 None，则返回 None
        if value is None:
            return None
        
        # 如果值不是字典类型，则抛出类型错误异常
        if not isinstance(value, dict):
            msg = f"value is not a dict: {value} ({type(value)})"
            raise TypeError(msg)

        # 如果指定了键类型 key_type，则验证字典的键是否符合指定类型
        if key_type is not None:
            for v in value:
                if not isinstance(v, key_type):
                    msg = f"dict key has item that is not {key_type}: {v} ({type(v)})"
                    raise TypeError(msg)

        # 如果指定了值类型 value_type，则验证字典的值是否符合指定类型
        if value_type is not None:
            for v in value.values():
                if not isinstance(v, value_type):
                    msg = f"dict value has item that is not {value_type}: {v} ({type(v)})"
                    raise TypeError(msg)

        # 返回验证通过的字典值
        return value

    # 如果数据中不存在指定的列名，则抛出值错误异常
    msg = f"Column {column_name} not found in data"
    raise ValueError(msg)
```