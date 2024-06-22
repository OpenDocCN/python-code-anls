# `.\transformers\hf_argparser.py`

```py
# 引入必要的模块和库
import dataclasses  # 用于定义数据类
import json  # 用于 JSON 数据的编解码
import sys  # 用于与 Python 解释器交互
import types  # 用于操作 Python 对象的类型
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError  # 用于解析命令行参数
from copy import copy  # 用于复制对象
from enum import Enum  # 用于定义枚举类型
from inspect import isclass  # 用于判断对象是否为类
from pathlib import Path  # 用于处理文件路径
from typing import Any, Callable, Dict, Iterable, List, Literal, NewType, Optional, Tuple, Union, get_type_hints  # 用于类型提示

import yaml  # 用于 YAML 数据的编解码

# 定义新类型 DataClass
DataClass = NewType("DataClass", Any)
# 定义新类型 DataClassType
DataClassType = NewType("DataClassType", Any)


# 从 https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse 引用的函数
# 将字符串转换为布尔值
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


# 创建从每个选项的字符串表示到实际值的映射函数，用于支持单个参数的多个值类型
def make_choice_type_function(choices: list) -> Callable[[str], Any]:
    """
    Creates a mapping function from each choices string representation to the actual value. Used to support multiple
    value types for a single argument.

    Args:
        choices (list): List of choices.

    Returns:
        Callable[[str], Any]: Mapping function from string representation to actual value for each choice.
    """
    # 将选项的字符串表示映射为实际值
    str_to_choice = {str(choice): choice for choice in choices}
    # 返回映射函数
    return lambda arg: str_to_choice.get(arg, arg)


# HfArg 辅助函数，使得创建用于解析的数据类字段更加简洁
def HfArg(
    *,
    aliases: Union[str, List[str]] = None,  # 参数的别名列表
    help: str = None,  # 参数的帮助文本
    default: Any = dataclasses.MISSING,  # 参数的默认值
    default_factory: Callable[[], Any] = dataclasses.MISSING,  # 参数的默认工厂函数
    meta dict = None,  # 参数的元数据
    **kwargs,  # 其它关键字参数
) -> dataclasses.Field:
    """Argument helper enabling a concise syntax to create dataclass fields for parsing with `HfArgumentParser`.

    Example comparing the use of `HfArg` and `dataclasses.field`:
    ```
    @dataclass
    class Args:
        regular_arg: str = dataclasses.field(default="Huggingface", metadata={"aliases": ["--example", "-e"], "help": "This syntax could be better!"})
        hf_arg: str = HfArg(default="Huggingface", aliases=["--example", "-e"], help="What a nice syntax!")
    ```py
    """
    def add_field_metadata(aliases=None, help=None, default=dataclasses.MISSING, default_factory=dataclasses.MISSING, metadata=None, **kwargs):
        """
        将字段的元数据添加到 `dataclasses.Field` 中。
    
        Args:
            aliases (Union[str, List[str]], optional):
                要传递给 argparse 的别名的单个字符串或字符串列表，例如 `aliases=["--example", "-e"]`。
                默认为 None。
            help (str, optional): 可以与 --help 一起显示的帮助字符串，要传递给 argparse。
                默认为 None。
            default (Any, optional):
                参数的默认值。如果未指定 default 或 default_factory，则参数是必需的。
                默认为 dataclasses.MISSING。
            default_factory (Callable[[], Any], optional):
                default_factory 是一个零参数函数，用于初始化字段的值。对于可变类型（例如列表）提供默认值非常有用：`default_factory=list`。
                与 `default=` 互斥。
                默认为 dataclasses.MISSING。
            metadata (dict, optional): 要传递给 `dataclasses.field` 的更多元数据。
                默认为 None。
    
        Returns:
            Field: 具有所需属性的 `dataclasses.Field`。
        """
        if metadata is None:
            # 重要提示：不要将 dict 用作函数签名中的默认参数，因为 dict 是可变的，并且在函数调用之间是共享的
            metadata = {}
        if aliases is not None:
            metadata["aliases"] = aliases
        if help is not None:
            metadata["help"] = help
    
        return dataclasses.field(metadata=metadata, default=default, default_factory=default_factory, **kwargs)
class HfArgumentParser(ArgumentParser):
    """
    这个 `argparse.ArgumentParser` 的子类使用数据类上的类型提示生成参数。

    这个类被设计成与原生的 argparse 很好地配合。特别地，你可以在初始化之后向解析器添加更多（非数据类支持的）参数，
    并在解析后将其作为额外的命名空间返回。可选：要创建子参数组，可以在数据类中使用 `_argument_group_name` 属性。
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                数据类类型，或要用解析后的参数“填充”实例的数据类类型列表。
            kwargs (`Dict[str, Any]`, *optional*):
                在常规方式下传递给 `argparse.ArgumentParser()` 的参数。
        """
        # 使得默认值在使用 --help 时出现
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 如果 `dataclass_types` 是数据类，则将其转换为列表
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = list(dataclass_types)
        # 为每个数据类添加参数
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    @staticmethod
    # 定义一个方法，用于向 ArgumentParser 添加数据类的参数
    def _add_dataclass_arguments(self, dtype: DataClassType):
        # 如果数据类具有 "_argument_group_name" 属性，则创建一个新的参数组
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            # 否则，使用当前参数解析器
            parser = self

        try:
            # 尝试获取数据类的类型提示
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except NameError:
            # 如果类型解析失败，引发运行时错误，提供相关建议
            raise RuntimeError(
                f"Type resolution failed for {dtype}. Try declaring the class in global scope or "
                "removing line of `from __future__ import annotations` which opts in Postponed "
                "Evaluation of Annotations (PEP 563)"
            )
        except TypeError as ex:
            # 在 Python 3.9 支持被移除后移除此块
            if sys.version_info[:2] < (3, 10) and "unsupported operand type(s) for |" in str(ex):
                # 如果是 Python 版本不支持的错误类型，则提供相关建议
                python_version = ".".join(map(str, sys.version_info[:3]))
                raise RuntimeError(
                    f"Type resolution failed for {dtype} on Python {python_version}. Try removing "
                    "line of `from __future__ import annotations` which opts in union types as "
                    "`X | Y` (PEP 604) via Postponed Evaluation of Annotations (PEP 563). To "
                    "support Python versions that lower than 3.10, you need to use "
                    "`typing.Union[X, Y]` instead of `X | Y` and `typing.Optional[X]` instead of "
                    "`X | None`."
                ) from ex
            raise

        # 遍历数据类的字段
        for field in dataclasses.fields(dtype):
            # 如果字段不是初始化参数，则跳过
            if not field.init:
                continue
            # 更新字段的类型提示
            field.type = type_hints[field.name]
            # 解析数据类字段并将其添加到参数解析器中
            self._parse_dataclass_field(parser, field)

    # 将参数解析为数据类的方法
    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
        look_for_args_file=True,
        args_filename=None,
        args_file_flag=None,
```  
    # 解析一个字典，将其值填充到数据类实例中，不使用 argparse，而是使用字典和数据类类型
    def parse_dict(self, args: Dict[str, Any], allow_extra_keys: bool = False) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.

        Args:
            args (`dict`):
                dict containing config values
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the dict contains keys that are not parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        # 创建一个未使用的键集合，用于跟踪未使用的键
        unused_keys = set(args.keys())
        # 创建一个空列表，用于存储数据类实例
        outputs = []
        # 遍历数据类类型列表
        for dtype in self.dataclass_types:
            # 获取数据类字段的名称集合
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            # 从参数中筛选出符合字段名称的键值对
            inputs = {k: v for k, v in args.items() if k in keys}
            # 更新未使用的键集合，去除已经使用的键
            unused_keys.difference_update(inputs.keys())
            # 使用筛选后的键值对创建数据类实例
            obj = dtype(**inputs)
            # 将数据类实例添加到输出列表中
            outputs.append(obj)
        # 如果不允许额外的键且存在未使用的键，则抛出异常
        if not allow_extra_keys and unused_keys:
            raise ValueError(f"Some keys are not used by the HfArgumentParser: {sorted(unused_keys)}")
        # 返回数据类实例的元组
        return tuple(outputs)

    # 解析一个 JSON 文件，将其值填充到数据类实例中，不使用 argparse，而是加载 JSON 文件和数据类类型
    def parse_json_file(self, json_file: str, allow_extra_keys: bool = False) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.

        Args:
            json_file (`str` or `os.PathLike`):
                File name of the json file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the json file contains keys that are not
                parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        # 使用 utf-8 编码打开 JSON 文件
        with open(Path(json_file), encoding="utf-8") as open_json_file:
            # 读取 JSON 文件内容并解析为字典
            data = json.loads(open_json_file.read())
        # 使用 parse_dict 方法解析字典数据，并返回数据类实例的元组
        outputs = self.parse_dict(data, allow_extra_keys=allow_extra_keys)
        # 返回数据类实例的元组
        return tuple(outputs)
    # 解析 YAML 文件的辅助方法，不使用 `argparse`，而是加载一个 YAML 文件并填充数据类类型
    def parse_yaml_file(self, yaml_file: str, allow_extra_keys: bool = False) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a yaml file and populating the
        dataclass types.

        Args:
            yaml_file (`str` or `os.PathLike`):
                File name of the yaml file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the json file contains keys that are not
                parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        # 使用 `yaml.safe_load` 方法加载 YAML 文件内容，返回一个字典
        outputs = self.parse_dict(yaml.safe_load(Path(yaml_file).read_text()), allow_extra_keys=allow_extra_keys)
        # 将数据传递给 `parse_dict` 方法进行解析，并返回结果的元组形式
        return tuple(outputs)
```