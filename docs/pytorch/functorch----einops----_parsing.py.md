# `.\pytorch\functorch\einops\_parsing.py`

```py
"""Adapted from https://github.com/arogozhnikov/einops/blob/36c7bb16e57d6e57f8f3050f9e07abdf3f00469f/einops/parsing.py.

MIT License

Copyright (c) 2018 Alex Rogozhnikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import keyword
import warnings
from typing import Collection, List, Mapping, Optional, Set, Tuple, Union

_ellipsis: str = "\u2026"  # NB, this is a single unicode symbol. String is used as it is not a list, but can be iterated


class AnonymousAxis:
    """Represents an axis with a size (> 1) in parsed expressions, lacking an associated identifier."""

    def __init__(self, value: str) -> None:
        # Initialize with the provided value converted to an integer
        self.value = int(value)
        # Check if the value is less than 1 and raise an error if true
        if self.value < 1:
            raise ValueError(
                f"Anonymous axis should have positive length, not {self.value}"
            )

    def __repr__(self) -> str:
        # Return a string representation indicating the size of the anonymous axis
        return f"{self.value}-axis"


class ParsedExpression:
    """Structure containing information about one side of an `einops`-style pattern (e.g. 'b c (h w)')."""

    def __init__(
        self,
        expression: str,
        *,
        allow_underscore: bool = False,
        allow_duplicates: bool = False,
    ):
        # Initialize ParsedExpression object with the provided expression and optional flags
        pass  # Placeholder for future initialization code

    @staticmethod
    def check_axis_name_return_reason(
        name: str, allow_underscore: bool = False
    ) -> Optional[str]:
        # Static method to validate axis names and return reasons for invalid names
        pass  # Placeholder for method implementation
    ) -> Tuple[bool, str]:
        """检查给定的轴名称是否有效，并返回一个解释信息。

        有效的轴名称应该是 Python 的标识符，不能是关键字，并且不应以或以下划线结尾。

        Args:
            name (str): 要检查的轴名称
            allow_underscore (bool): 是否允许轴名称以下划线开头

        Returns:
            Tuple[bool, str]: 轴名称是否有效，如果无效则返回解释信息
        """
        if not str.isidentifier(name):
            return False, "not a valid python identifier"
        elif name[0] == "_" or name[-1] == "_":
            if name == "_" and allow_underscore:
                return True, ""
            return False, "axis name should should not start or end with underscore"
        else:
            if keyword.iskeyword(name):
                warnings.warn(
                    f"使用关键字作为轴名称不被推荐: {name}",
                    RuntimeWarning,
                )
            if name in ["axis"]:
                warnings.warn(
                    "不推荐使用 'axis' 作为轴名称，并且将在未来引发错误",
                    FutureWarning,
                )
            return True, ""

    @staticmethod
    def check_axis_name(name: str) -> bool:
        """检查给定的名称是否是有效的轴名称。

        Args:
            name (str): 要检查的轴名称

        Returns:
            bool: 轴名称是否有效
        """
        is_valid, _ = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid
# 将一个 `einops` 风格的模式解析为左侧和右侧的 `ParsedExpression` 对象。
def parse_pattern(
    pattern: str, axes_lengths: Mapping[str, int]
) -> Tuple[ParsedExpression, ParsedExpression]:
    """Parse an `einops`-style pattern into a left-hand side and right-hand side `ParsedExpression` object.

    Args:
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (Mapping[str, int]): any additional length specifications for dimensions

    Returns:
       Tuple[ParsedExpression, ParsedExpression]: a tuple containing the left-hand side and right-hand side expressions
    """
    # 从 `einops.einops._prepare_transformation_recipe` 改编而来
    # https://github.com/arogozhnikov/einops/blob/230ac1526c1f42c9e1f7373912c7f8047496df11/einops/einops.py
    try:
        left_str, right_str = pattern.split("->")
    except ValueError:
        raise ValueError("Pattern must contain a single '->' separator") from None

    # 检查是否有 `_ellipsis` 在 `axes_lengths` 中，若有则引发异常
    if _ellipsis in axes_lengths:
        raise ValueError(f"'{_ellipsis}' is not an allowed axis identifier")

    # 创建左侧和右侧的 `ParsedExpression` 对象
    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)

    # 如果左侧没有省略号但右侧有，则引发异常
    if not left.has_ellipsis and right.has_ellipsis:
        raise ValueError(
            f"Ellipsis found in right side, but not left side of a pattern {pattern}"
        )
    # 如果左侧有省略号且省略号被括号包围，则引发异常
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise ValueError(
            f"Ellipsis is parenthesis in the left side is not allowed: {pattern}"
        )

    # 返回左侧和右侧的 `ParsedExpression` 对象
    return left, right


# 执行特定于 `rearrange` 操作的表达式验证。
def validate_rearrange_expressions(
    left: ParsedExpression, right: ParsedExpression, axes_lengths: Mapping[str, int]
) -> None:
    """Perform expression validations that are specific to the `rearrange` operation.

    Args:
        left (ParsedExpression): left-hand side expression
        right (ParsedExpression): right-hand side expression
        axes_lengths (Mapping[str, int]): any additional length specifications for dimensions
    """
    # 检查 `axes_lengths` 中每个值是否为整数类型，若不是则引发异常
    for length in axes_lengths.values():
        if (length_type := type(length)) is not int:
            raise TypeError(
                f"rearrange axis lengths must be integers, got: {length_type}"
            )

    # 若左侧或右侧包含非单元素匿名轴，则引发异常
    if left.has_non_unitary_anonymous_axes or right.has_non_unitary_anonymous_axes:
        raise ValueError("rearrange only supports unnamed axes of size 1")

    # 找到左侧和右侧表达式中不同的标识符，若存在则引发异常
    difference = set.symmetric_difference(left.identifiers, right.identifiers)
    if len(difference) > 0:
        raise ValueError(
            f"Identifiers only on one side of rearrange expression (should be on both): {difference}"
        )

    # 找到未匹配的轴标识符，若存在则引发异常
    unmatched_axes = axes_lengths.keys() - left.identifiers
    if len(unmatched_axes) > 0:
        raise ValueError(
            f"Identifiers not found in rearrange expression: {unmatched_axes}"
        )


# 将表示一级尺寸的字符串集合转换为逗号分隔的字符串。
def comma_separate(collection: Collection[Union[str, Collection[str]]]) -> str:
    """Convert a collection of strings representing first class dims into a comma-separated string.
    Args:
        collection (Collection[Union[str, Collection[str]]]): 要转换的字符串集合，可以是字符串或字符串集合的混合类型。

    Returns:
        str: 逗号分隔的字符串结果。

    Examples:
        >>> comma_separate(('d0',))
        'd0'

        >>> comma_separate(('d0', 'd1', 'd2', 'd3'))
        'd0, d1, d2, d3'

        >>> comma_separate([('d1', 'd4')])
        '(d1, d4)'

        >>> comma_separate([('d0',), (), ('d1',), ('d2',), ('d3', 'd4')])
        '(d0,), (), (d1,), (d2,), (d3, d4)'
    """
    # 使用列表推导式处理集合中的每个元素
    return ", ".join(
        item
        if isinstance(item, str)  # 如果元素是字符串，则直接使用
        else f"({comma_separate(item)}{',' if len(item) == 1 else ''})"  # 如果元素是集合，则递归调用 comma_separate 处理并包装在括号内
        for item in collection  # 对于集合中的每个元素
    )
```