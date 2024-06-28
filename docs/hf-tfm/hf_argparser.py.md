# `.\hf_argparser.py`

```
# 版权声明及许可信息
#
# 在 Apache 许可证 2.0 版本下使用此文件的声明，表示除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“原样”分发的软件是根据许可证分发的，
# 没有任何形式的明示或暗示担保或条件。
# 有关更多详细信息，请参阅许可证。
#

import dataclasses  # 导入 dataclasses 模块
import json  # 导入 json 模块
import sys  # 导入 sys 模块
import types  # 导入 types 模块
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError  # 从 argparse 模块导入指定内容
from copy import copy  # 导入 copy 函数
from enum import Enum  # 导入 Enum 类
from inspect import isclass  # 导入 isclass 函数
from pathlib import Path  # 导入 Path 类
from typing import Any, Callable, Dict, Iterable, List, Literal, NewType, Optional, Tuple, Union, get_type_hints  # 导入 typing 模块中指定内容

import yaml  # 导入 yaml 模块


DataClass = NewType("DataClass", Any)  # 定义 DataClass 类型别名
DataClassType = NewType("DataClassType", Any)  # 定义 DataClassType 类型别名


def string_to_bool(v):
    """
    解析字符串表示的布尔值。

    Args:
        v (str): 输入的字符串值。

    Returns:
        bool: 如果字符串表示真值，则返回 True；否则返回 False。

    Raises:
        ArgumentTypeError: 如果无法解析字符串为布尔值，抛出异常。
    """
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


def make_choice_type_function(choices: list) -> Callable[[str], Any]:
    """
    创建从每个选择字符串表示到实际值的映射函数。用于支持单个参数的多个值类型。

    Args:
        choices (list): 选择列表。

    Returns:
        Callable[[str], Any]: 从字符串表示到每个选择的实际值的映射函数。
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def HfArg(
    *,
    aliases: Union[str, List[str]] = None,
    help: str = None,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[[], Any] = dataclasses.MISSING,
    metadata: dict = None,
    **kwargs,
) -> dataclasses.Field:
    """
    参数辅助函数，允许使用简洁的语法为 `HfArgumentParser` 创建数据类字段。

    Example comparing the use of `HfArg` and `dataclasses.field`:
    示例比较了 `HfArg` 和 `dataclasses.field` 的使用：
    ```
    @dataclass
    class Args:
        regular_arg: str = dataclasses.field(default="Huggingface", metadata={"aliases": ["--example", "-e"], "help": "This syntax could be better!"})
        hf_arg: str = HfArg(default="Huggingface", aliases=["--example", "-e"], help="What a nice syntax!")
    ```
    """
    pass  # HfArg 函数主体为空，实现在示例中展示
    def make_field(aliases=None, help=None, default=dataclasses.MISSING, default_factory=dataclasses.MISSING, metadata=None, **kwargs):
        """
        Construct a `dataclasses.Field` object with specified properties.
    
        Args:
            aliases (Union[str, List[str]], optional):
                Single string or list of strings of aliases to pass on to argparse, e.g. `aliases=["--example", "-e"]`.
                Defaults to None.
            help (str, optional): Help string to pass on to argparse that can be displayed with --help. Defaults to None.
            default (Any, optional):
                Default value for the argument. If not default or default_factory is specified, the argument is required.
                Defaults to dataclasses.MISSING.
            default_factory (Callable[[], Any], optional):
                The default_factory is a 0-argument function called to initialize a field's value. It is useful to provide
                default values for mutable types, e.g. lists: `default_factory=list`. Mutually exclusive with `default=`.
                Defaults to dataclasses.MISSING.
            metadata (dict, optional): Further metadata to pass on to `dataclasses.field`. Defaults to None.
    
        Returns:
            Field: A `dataclasses.Field` with the desired properties.
        """
        if metadata is None:
            # 如果 metadata 参数为 None，则创建一个空的字典，以避免在函数签名中使用默认参数，因为字典是可变的且在函数调用间共享
            metadata = {}
        if aliases is not None:
            # 如果传入了 aliases 参数，则将其添加到 metadata 字典中
            metadata["aliases"] = aliases
        if help is not None:
            # 如果传入了 help 参数，则将其添加到 metadata 字典中
            metadata["help"] = help
    
        # 创建并返回一个 `dataclasses.Field` 对象，传入指定的参数和 metadata 字典
        return dataclasses.field(metadata=metadata, default=default, default_factory=default_factory, **kwargs)
# 定义一个名为 HfArgumentParser 的类，它是 argparse.ArgumentParser 的子类
class HfArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace. Optional: To create sub argument groups use the `_argument_group_name` attribute in the dataclass.
    """

    # 定义一个名为 dataclass_types 的实例变量，用来存储数据类类型的可迭代对象
    dataclass_types: Iterable[DataClassType]

    # 初始化方法，接收 dataclass_types 和其他参数
    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs (`Dict[str, Any]`, *optional*):
                Passed to `argparse.ArgumentParser()` in the regular way.
        """
        # 如果 kwargs 中没有指定 formatter_class，则设置为 ArgumentDefaultsHelpFormatter
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        # 调用父类 ArgumentParser 的初始化方法，传入 kwargs
        super().__init__(**kwargs)
        # 如果 dataclass_types 是单个数据类而不是列表，则转换为列表
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        # 将 dataclass_types 转换为列表后存储在 self.dataclass_types 中
        self.dataclass_types = list(dataclass_types)
        # 遍历每个数据类类型，并为其添加参数到 argparse.ArgumentParser 实例中
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    # 静态方法，用来添加数据类的参数到 argparse.ArgumentParser 实例中
    @staticmethod
    # 将数据类的参数添加到命令行解析器中
    def _add_dataclass_arguments(self, dtype: DataClassType):
        # 检查数据类是否定义了参数组名称，如果是，则创建一个新的参数组；否则，使用当前解析器
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self

        try:
            # 获取数据类字段的类型提示字典
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except NameError:
            # 如果类型解析失败，通常是由于数据类不在全局范围内定义或使用了延迟注释的特性
            raise RuntimeError(
                f"Type resolution failed for {dtype}. Try declaring the class in global scope or "
                "removing line of `from __future__ import annotations` which opts in Postponed "
                "Evaluation of Annotations (PEP 563)"
            )
        except TypeError as ex:
            # 当 Python 版本低于 3.10 且涉及到 union 类型时，给出详细的错误信息和建议
            if sys.version_info[:2] < (3, 10) and "unsupported operand type(s) for |" in str(ex):
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

        # 遍历数据类的字段，并解析每个需要初始化的字段
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue  # 跳过不需要初始化的字段
            # 将字段的类型设定为从类型提示中获取的类型
            field.type = type_hints[field.name]
            # 调用私有方法，将数据类字段解析到命令行解析器中
            self._parse_dataclass_field(parser, field)

    # 解析命令行参数到数据类对象中
    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
        look_for_args_file=True,
        args_filename=None,
        args_file_flag=None,
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
        # 获取所有传入参数字典的键集合
        unused_keys = set(args.keys())
        # 初始化空列表，用于存储解析后的数据类实例
        outputs = []
        # 遍历数据类类型列表
        for dtype in self.dataclass_types:
            # 获取数据类字段的名称集合，仅包括可以初始化的字段
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            # 从参数字典中选取与数据类字段匹配的键值对
            inputs = {k: v for k, v in args.items() if k in keys}
            # 从未使用的键集合中移除已使用的键
            unused_keys.difference_update(inputs.keys())
            # 使用选取的键值对初始化数据类对象
            obj = dtype(**inputs)
            # 将初始化后的数据类对象添加到输出列表
            outputs.append(obj)
        # 如果不允许额外的键存在且有未使用的键，抛出异常
        if not allow_extra_keys and unused_keys:
            raise ValueError(f"Some keys are not used by the HfArgumentParser: {sorted(unused_keys)}")
        # 返回包含所有数据类实例的元组
        return tuple(outputs)

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
        # 打开并读取 JSON 文件
        with open(Path(json_file), encoding="utf-8") as open_json_file:
            data = json.loads(open_json_file.read())
        # 使用 parse_dict 方法解析 JSON 数据，并返回结果
        outputs = self.parse_dict(data, allow_extra_keys=allow_extra_keys)
        # 返回包含所有数据类实例的元组
        return tuple(outputs)
    # 定义一个方法用于解析 YAML 文件，并返回一个元组，其中包含数据类实例。
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
        # 使用 pathlib 读取 YAML 文件的文本内容，然后通过 yaml.safe_load 转换为 Python 对象
        outputs = self.parse_dict(yaml.safe_load(Path(yaml_file).read_text()), allow_extra_keys=allow_extra_keys)
        # 返回一个包含所有数据类实例的元组
        return tuple(outputs)
```