# `.\pytorch\torch\onnx\_internal\diagnostics\infra\formatter.py`

```
# 引入 Python 未来兼容性模块，用于支持注解类型
from __future__ import annotations

# 引入标准库模块
import dataclasses  # 用于数据类支持
import json  # 用于 JSON 操作
import re  # 用于正则表达式操作
import traceback  # 用于异常回溯操作
from typing import Any, Callable, Dict, List, Optional, Union  # 引入类型提示支持

# 引入 torch 库的日志相关模块
from torch._logging import LazyString  # 惰性字符串支持
from torch.onnx._internal import _beartype  # beartype 函数装饰器
from torch.onnx._internal.diagnostics.infra import sarif  # sarif 模块引入


# SARIF 模块中的类型列表，用于美化打印
# 主要用于以下函数的类型注解
_SarifClass = Union[
    sarif.SarifLog,
    sarif.Run,
    sarif.ReportingDescriptor,
    sarif.Result,
]


# 返回异常信息的惰性字符串表示
def lazy_format_exception(exception: Exception) -> LazyString:
    return LazyString(
        lambda: "\n".join(
            (
                "```",
                *traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                ),
                "```",
            )
        ),
    )


# 将蛇形命名转换为驼峰命名
@_beartype.beartype
def snake_case_to_camel_case(s: str) -> str:
    splits = s.split("_")
    if len(splits) <= 1:
        return s
    return "".join([splits[0], *map(str.capitalize, splits[1:])])


# 将驼峰命名转换为蛇形命名
@_beartype.beartype
def camel_case_to_snake_case(s: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", s).lower()


# 将短横线命名转换为蛇形命名
@_beartype.beartype
def kebab_case_to_snake_case(s: str) -> str:
    return s.replace("-", "_")


# 将字典中的键转换为指定格式
# 如果值为字典，则递归更新其键
# 如果值为列表，则递归搜索更新
@_beartype.beartype
def _convert_key(
    object: Union[Dict[str, Any], Any], convert: Callable[[str], str]
) -> Union[Dict[str, Any], Any]:
    """Convert and update keys in a dictionary with "convert".

    Any value that is a dictionary will be recursively updated.
    Any value that is a list will be recursively searched.

    Args:
        object: The object to update.
        convert: The function to convert the keys, e.g. `kebab_case_to_snake_case`.

    Returns:
        The updated object.
    """
    if not isinstance(object, Dict):
        return object
    new_dict = {}
    for k, v in object.items():
        new_k = convert(k)
        if isinstance(v, Dict):
            new_v = _convert_key(v, convert)
        elif isinstance(v, List):
            new_v = [_convert_key(elem, convert) for elem in v]
        else:
            new_v = v
        if new_v is None:
            # Otherwise unnecessarily bloated sarif log with "null"s.
            continue
        if new_v == -1:
            # WAR: -1 as default value shouldn't be logged into sarif.
            continue

        new_dict[new_k] = new_v

    return new_dict


# 将 SARIF 对象转换为 JSON 格式字符串
@_beartype.beartype
def sarif_to_json(attr_cls_obj: _SarifClass, indent: Optional[str] = " ") -> str:
    dict = dataclasses.asdict(attr_cls_obj)  # 将数据类对象转换为字典
    dict = _convert_key(dict, snake_case_to_camel_case)  # 转换字典中的键
    return json.dumps(dict, indent=indent, separators=(",", ":"))  # 将字典转换为 JSON 字符串


# 格式化参数对象的类型信息字符串
@_beartype.beartype
def format_argument(obj: Any) -> str:
    return f"{type(obj)}"


# 获取函数的显示名称
@_beartype.beartype
def display_name(fn: Callable) -> str:
    if hasattr(fn, "__qualname__"):
        return fn.__qualname__
    elif hasattr(fn, "__name__"):
        return fn.__name__
    else:
        # 如果条件不满足（即 fn 不是字符串类型），将 fn 转换为字符串并返回
        return str(fn)
```