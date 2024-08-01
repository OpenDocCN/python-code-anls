# `.\DB-GPT-src\dbgpt\util\formatting.py`

```py
"""Utilities for formatting strings."""
# 导入必要的模块和类
import json  # 导入 JSON 模块
from string import Formatter  # 从字符串模块导入 Formatter 类
from typing import Any, List, Mapping, Sequence, Set, Union  # 导入类型提示相关的模块


class StrictFormatter(Formatter):
    """A subclass of formatter that checks for extra keys."""

    def check_unused_args(
        self,
        used_args: Set[Union[int, str]],
        args: Sequence,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Check to see if extra parameters are passed."""
        # 计算未使用的参数
        extra = set(kwargs).difference(used_args)
        # 如果存在未使用的参数，抛出 KeyError 异常
        if extra:
            raise KeyError(extra)

    def vformat(
        self, format_string: str, args: Sequence, kwargs: Mapping[str, Any]
    ) -> str:
        """Check that no arguments are provided."""
        # 检查是否提供了位置参数，如果提供了则抛出 ValueError 异常
        if len(args) > 0:
            raise ValueError(
                "No arguments should be provided, "
                "everything should be passed as keyword arguments."
            )
        # 调用父类的 vformat 方法进行格式化
        return super().vformat(format_string, args, kwargs)

    def validate_input_variables(
        self, format_string: str, input_variables: List[str]
    ) -> None:
        """Validate input variables."""
        # 创建虚拟输入变量字典
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        # 调用父类的 format 方法对格式字符串进行格式化，传入虚拟输入变量
        super().format(format_string, **dummy_inputs)


class NoStrictFormatter(StrictFormatter):
    def check_unused_args(
        self,
        used_args: Set[Union[int, str]],
        args: Sequence,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Not check unused args"""
        # 不执行未使用参数的检查，方法体为空
        pass


# 创建 StrictFormatter 的实例
formatter = StrictFormatter()
# 创建 NoStrictFormatter 的实例
no_strict_formatter = NoStrictFormatter()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        """Override JSONEncoder's default method."""
        # 如果对象是集合类型，则转换为列表
        if isinstance(obj, set):
            return list(obj)
        # 如果对象有 __dict__ 属性，则返回其字典形式
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        # 否则调用父类的 default 方法处理
        else:
            return json.JSONEncoder.default(self, obj)
```