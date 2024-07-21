# `.\pytorch\torch\_library\infer_schema.py`

```
````
# 允许未类型化的函数定义，便于 mypy 的类型检查
# 该行指定了 mypy 的解析选项，允许未指定类型的函数
# pylint: disable=unused-import
# 引入标准库 inspect 模块，用于获取函数签名和参数信息
import inspect
import typing
from typing import List, Optional, Sequence, Union  # noqa: F401

# 引入 PyTorch 库，忽略 mypy 检查
import torch  # noqa: F401
from .. import device, dtype, Tensor, types  # noqa: F401

# 定义 infer_schema 函数，接收一个函数的类型提示和一个元组参数 mutates_args
def infer_schema(prototype_function: typing.Callable, mutates_args=()) -> str:
    """Given a function with type hints, parses a schema.

    We make some assumptions to make our lives easier that correspond to how people
    write custom ops in real life:
    - none of the outputs alias any of the inputs or each other.
    - only the args listed in mutates_args are being mutated.
    - string type annotations "device, dtype, Tensor, types" without library specification
      are assumed to be torch.*. Similarly, string type annotations "Optional, List, Sequence, Union"
      without library specification are assumed to be typing.*.

    Callers (e.g. the custom ops API) are responsible for checking these assumptions.
    """
    sig = inspect.signature(prototype_function)  # 获取函数签名

    # 定义 error_fn 函数，处理错误情况，抛出 ValueError 异常
    def error_fn(what):
        raise ValueError(
            f"infer_schema(func): {what} " f"Got func with signature {sig})"
        )

    # 定义 convert_type_string 函数，将字符串类型注解转换为实际类型
    def convert_type_string(annotation_type: str):
        try:
            return eval(annotation_type)
        except Exception as e:
            error_fn(
                f"Unsupported type annotation {annotation_type}. It is not a type."
            )

    params = []  # 初始化参数列表
    seen_args = set()  # 初始化已见参数集合
    saw_kwarg_only_arg = False  # 标记是否只见到关键字参数
    mutates_args_not_seen = set(mutates_args) - seen_args  # 计算未见到的 mutates_args 参数

    # 检查 mutates_args 中的参数是否都在函数签名中
    if len(mutates_args_not_seen) > 0:
        error_fn(
            f"{mutates_args_not_seen} in mutates_args were not found in "
            f"the custom op's signature. "
            f"mutates_args should contain the names of all args that the "
            f"custom op mutates."
        )

    # 获取函数返回注解
    return_annotation = sig.return_annotation
    # 将返回注解字符串转换为类型
    if type(return_annotation) == str:
        return_annotation = convert_type_string(return_annotation)
    ret = parse_return(return_annotation, error_fn)  # 解析返回类型
    # 返回参数的类型签名字符串
    return f"({', '.join(params)}) -> {ret}"


# 定义 derived_types 函数，生成基类型及其衍生类型
def derived_types(
    base_type, cpp_type, list_base, optional_base_list, optional_list_base
):
    # 初始化结果列表，包含基本类型和可选类型的元组
    result = [
        (base_type, cpp_type),
        (typing.Optional[base_type], f"{cpp_type}?"),
    ]

    # 定义 derived_seq_types 函数，生成序列类型
    def derived_seq_types(typ):
        return [
            typing.Sequence[typ],  # type: ignore[valid-type]
            typing.List[typ],  # type: ignore[valid-type]
        ]

    # 如果 list_base 为真，添加列表类型到结果列表
    if list_base:
        for seq_typ in derived_seq_types(base_type):
            result.append((seq_typ, f"{cpp_type}[]"))  # type: ignore[valid-type]
    # 如果 optional_base_list 为真，添加可选列表类型到结果列表
    if optional_base_list:
        for seq_typ in derived_seq_types(typing.Optional[base_type]):
            result.append((seq_typ, f"{cpp_type}?[]"))  # type: ignore[valid-type]
    # 如果 optional_list_base 为真，添加可选序列类型到结果列表
    if optional_list_base:
        for seq_typ in derived_seq_types(base_type):  # type: ignore[valid-type]
            result.append((typing.Optional[seq_typ], f"{cpp_type}[]?"))  # type: ignore[valid-type]
    return result  # 返回结果列表
def get_supported_param_types():
    # 定义一个包含不同参数类型及其变体的数据列表
    data = [
        (Tensor, "Tensor", True, True, False),       # Tensor 类型及其变体
        (int, "SymInt", True, False, True),          # int 类型及其变体
        (float, "float", True, False, True),         # float 类型及其变体
        (bool, "bool", True, False, True),           # bool 类型及其变体
        (str, "str", False, False, False),           # str 类型
        (types.Number, "Scalar", True, False, False),# Number 类型及其变体
        (dtype, "ScalarType", False, False, False),  # dtype 类型
        (device, "Device", False, False, False),     # device 类型
    ]
    result = []
    # 遍历数据列表，调用 derived_types 函数并将结果扩展到 result 列表中
    for line in data:
        result.extend(derived_types(*line))
    # 返回包含所有支持的参数类型的字典
    return dict(result)


SUPPORTED_RETURN_TYPES = {
    Tensor: "Tensor",                               # 支持的返回类型包括 Tensor
    typing.List[Tensor]: "Tensor[]",                # 支持的返回类型包括 Tensor 列表
    int: "SymInt",                                  # 支持的返回类型包括 int
    float: "float",                                 # 支持的返回类型包括 float
    bool: "bool",                                   # 支持的返回类型包括 bool
    types.Number: "Scalar",                         # 支持的返回类型包括 Number
}


def parse_return(annotation, error_fn):
    # 如果注解为 None，则返回空元组字符串表示无返回值
    if annotation is None:
        return "()"

    origin = typing.get_origin(annotation)
    # 如果注解的起源不是元组
    if origin is not tuple:
        # 如果注解不在支持的返回类型中，则调用错误处理函数
        if annotation not in SUPPORTED_RETURN_TYPES.keys():
            error_fn(
                f"Return has unsupported type {annotation}. "
                f"The valid types are: {SUPPORTED_RETURN_TYPES}."
            )
        # 返回对应注解的支持返回类型字符串
        return SUPPORTED_RETURN_TYPES[annotation]

    # 如果注解是元组类型，则获取其参数列表
    args = typing.get_args(annotation)
    for arg in args:
        # 如果元组的某个参数不在支持的返回类型中，则调用错误处理函数
        if arg not in SUPPORTED_RETURN_TYPES:
            error_fn(
                f"Return has unsupported type {annotation}. "
                f"The valid types are: {SUPPORTED_RETURN_TYPES}."
            )

    # 构建元组返回类型的字符串表示，包括各参数的支持返回类型
    return "(" + ", ".join([SUPPORTED_RETURN_TYPES[arg] for arg in args]) + ")"


SUPPORTED_PARAM_TYPES = get_supported_param_types()


def supported_param(param: inspect.Parameter) -> bool:
    # 判断参数是否为位置或关键字参数
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def tuple_to_list(tuple_type: typing.Type[typing.Tuple]) -> typing.Type[typing.List]:
    """
    将 `tuple_type` 转换为具有相同类型参数的列表类型。假定 `tuple_type` 是 typing.Tuple 类型。
    """
    type_args = getattr(tuple_type, "__args__", None)
    # 考虑不同的 Python 版本，例如 Python 3.8 给出 ()
    # 但 Python 3.12 给出 None。
    if tuple_type is typing.Tuple or type_args == () or type_args is None:
        # 处理空元组类型的情况
        return typing.List
    elif len(type_args) == 1:
        # 一般情况：创建具有相同类型参数的列表
        return typing.List[type_args[0]]  # type: ignore[valid-type]
    elif len(type_args) == 2 and type_args[1] is Ellipsis:  # type: ignore[valid-type]
        return typing.List[type_args[0]]  # type: ignore[valid-type]
    else:
        # 处理其他情况，创建具有 Union 类型参数的列表
        return typing.List[typing.Union[tuple(type_args)]]  # type: ignore[misc]
```