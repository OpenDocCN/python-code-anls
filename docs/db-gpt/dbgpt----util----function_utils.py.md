# `.\DB-GPT-src\dbgpt\util\function_utils.py`

```py
# 导入必要的模块
import asyncio
import inspect
from functools import wraps
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    _UnionGenericAlias,
    get_args,
    get_origin,
    get_type_hints,
)

# 导入类型检查相关的函数
from typeguard import check_type
from typing_extensions import Annotated, Doc, _AnnotatedAlias

# 检查对象是否属于 typing 模块定义的类型
def _is_typing(obj):
    from typing import _Final  # type: ignore

    return isinstance(obj, _Final)

# 检查对象是否是泛型类型的实例
def _is_instance_of_generic_type(obj, generic_type):
    """Check if an object is an instance of a generic type."""
    if generic_type is Any:
        return True  # Any type is compatible with any object

    origin = get_origin(generic_type)
    if origin is None:
        return isinstance(obj, generic_type)  # Handle non-generic types

    args = get_args(generic_type)
    if not args:
        return isinstance(obj, origin)

    # Check if object matches the generic origin (like list, dict)
    if not _is_typing(origin):
        return isinstance(obj, origin)

    objs = [obj for _ in range(len(args)]

    # For each item in the object, check if it matches the corresponding type argument
    for sub_obj, arg in zip(objs, args):
        # Skip check if the type argument is Any
        if arg is not Any:
            if _is_typing(arg):
                sub_args = get_args(arg)
                if (
                    sub_args
                    and not _is_typing(sub_args[0])
                    and not isinstance(sub_obj, sub_args[0])
                ):
                    return False
            elif not isinstance(sub_obj, arg):
                return False
    return True

# 检查对象是否符合指定类型
def _check_type(obj, t) -> bool:
    try:
        check_type(obj, t)
        return True
    except Exception:
        return False

# 获取参数的排序顺序
def _get_orders(obj, arg_types):
    try:
        orders = [i for i, t in enumerate(arg_types) if _check_type(obj, t)]
        return orders[0] if orders else int(1e8)
    except Exception:
        return int(1e8)

# 根据参数类型对参数进行排序
def _sort_args(func, args, kwargs):
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    arg_types = [
        type_hints[param_name]
        for param_name in sig.parameters
        if param_name != "return" and param_name != "self"
    ]

    if "self" in sig.parameters:
        self_arg = [args[0]]
        other_args = args[1:]
    else:
        self_arg = []
        other_args = args

    sorted_args = sorted(
        other_args,
        key=lambda x: _get_orders(x, arg_types),
    )
    return (*self_arg, *sorted_args), kwargs

# 装饰器，根据参数类型重新排列函数的参数
def rearrange_args_by_type(func):
    """Decorator to rearrange the arguments of a function by type.
    # 使用 functools 库的 wraps 装饰器，将 func 的元数据复制到 sync_wrapper 函数中
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # 调用内部函数 _sort_args 对参数进行排序，获取排序后的位置参数和关键字参数
        sorted_args, sorted_kwargs = _sort_args(func, args, kwargs)
        # 调用原始函数 func，并使用排序后的参数进行调用
        return func(*sorted_args, **sorted_kwargs)
    
    # 使用 functools 库的 wraps 装饰器，将 func 的元数据复制到 async_wrapper 函数中
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # 调用内部函数 _sort_args 对参数进行排序，获取排序后的位置参数和关键字参数
        sorted_args, sorted_kwargs = _sort_args(func, args, kwargs)
        # 使用 await 调用原始函数 func，并使用排序后的参数进行调用
        return await func(*sorted_args, **sorted_kwargs)
    
    # 根据原始函数是否为异步函数（coroutine function），决定返回 sync_wrapper 或 async_wrapper
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
# 将类型转换为字符串表示形式
def type_to_string(obj: Any, default_type: str = "unknown") -> str:
    # 类型映射字典，将 Python 类型映射为字符串表示
    type_map = {
        int: "integer",
        str: "string",
        float: "float",
        bool: "boolean",
        Any: "any",
        List: "array",
        dict: "object",
    }

    # 检查是否为 NoneType
    if obj is type(None):
        return "null"

    # 获取类型的原始类型
    origin = getattr(obj, "__origin__", None)
    if origin:
        # 如果类型有原始类型，并且不是 UnionGenericAlias 的实例，则进行处理
        if _is_typing(origin) and not isinstance(obj, _UnionGenericAlias):
            obj = origin
            origin = origin.__origin__

        # 处理特殊情况，例如 List[int]
        if origin is Union and hasattr(obj, "__args__"):
            # 获取 Union 类型的子类型列表，并将其转换为字符串
            subtypes = ", ".join(
                type_to_string(t) for t in obj.__args__ if t is not type(None)
            )
            return subtypes
        elif origin is list or origin is List:
            # 获取 List 类型的子类型列表，并将其转换为字符串
            subtypes = ", ".join(type_to_string(t) for t in obj.__args__)
            return "array"
        elif origin in [dict, Dict]:
            # 获取 Dict 类型的键和值的类型，并将其转换为字符串
            key_type, value_type = (type_to_string(t) for t in obj.__args__)
            return "object"

        # 根据映射字典返回类型的字符串表示，如果不存在则返回默认类型
        return type_map.get(origin, default_type)
    else:
        # 如果没有原始类型，则进一步处理参数类型
        if hasattr(obj, "__args__"):
            # 获取参数类型的子类型列表，并将其转换为字符串
            subtypes = ", ".join(
                type_to_string(t) for t in obj.__args__ if t is not type(None)
            )
            return subtypes

    # 根据映射字典返回对象的字符串表示，如果不存在则返回默认类型
    return type_map.get(obj, default_type)


# 解析参数描述信息
def parse_param_description(name: str, obj: Any) -> str:
    # 将参数名称中的下划线替换为空格，并将首字母大写
    default_type_title = name.replace("_", " ").title()

    if isinstance(obj, _AnnotatedAlias):
        # 如果对象是 _AnnotatedAlias 的实例，则获取其元数据
        metadata = obj.__metadata__
        # 从元数据中提取文档信息，若存在则返回文档字符串，否则返回默认类型的标题化字符串
        docs = [arg for arg in metadata if isinstance(arg, Doc)]
        doc_str = docs[0].documentation if docs else default_type_title
    else:
        # 如果对象不是 _AnnotatedAlias 的实例，则直接返回默认类型的标题化字符串
        doc_str = default_type_title

    return doc_str
```