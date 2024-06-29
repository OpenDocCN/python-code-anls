# `D:\src\scipysrc\pandas\pandas\io\formats\css.py`

```
# 解释CSS解析器用于非HTML输出格式化的CSS样式表的实用程序
"""
Utilities for interpreting CSS from Stylers for formatting non-HTML outputs.
"""

# 导入未来的注释语法特性，用于类型检查
from __future__ import annotations

# 导入正则表达式模块和类型检查相关的库
import re
from typing import TYPE_CHECKING
import warnings

# 导入Pandas CSS警告和异常处理相关模块
from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level

# 如果类型检查开启，则导入迭代器和生成器等类型
if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Iterator,
    )

# 定义一个函数，用于将简写的CSS属性扩展为具体的top、right、bottom、left属性
def _side_expander(prop_fmt: str) -> Callable:
    """
    Wrapper to expand shorthand property into top, right, bottom, left properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """

    # 内部函数，用于实际执行扩展操作
    def expand(
        self: CSSResolver, prop: str, value: str
    ) -> Generator[tuple[str, str], None, None]:
        """
        Expand shorthand property into side-specific property (top, right, bottom, left)

        Parameters
        ----------
            prop (str): CSS property name
            value (str): String token for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        # 将属性值分割为多个token
        tokens = value.split()
        try:
            # 根据token数量选择相应的映射关系
            mapping = self.SIDE_SHORTHANDS[len(tokens)]
        except KeyError:
            # 如果找不到合适的映射关系，发出警告并返回
            warnings.warn(
                f'Could not expand "{prop}: {value}"',
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return
        # 使用映射关系生成扩展后的属性和值
        for key, idx in zip(self.SIDES, mapping):
            yield prop_fmt.format(key), tokens[idx]

    return expand


# 定义一个函数，用于将'border'属性扩展为颜色、样式和宽度等具体属性
def _border_expander(side: str = "") -> Callable:
    """
    Wrapper to expand 'border' property into border color, style, and width properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """
    # 如果指定了side参数，则设置为"-{side}"
    if side != "":
        side = f"-{side}"

    # 内部函数，用于实际执行扩展操作
    def expand(
        self: CSSResolver, prop: str, value: str
        ):
        # ...
    ) -> Generator[tuple[str, str], None, None]:
        """
        将边框属性扩展为颜色、样式和宽度的元组

        Parameters
        ----------
            prop : str
                传递给样式处理器的 CSS 属性名称
            value : str
                传递给属性的值

        Yields
        ------
            Tuple (str, str): 扩展后的属性, 值
        """
        # 将值按空格分割成单词
        tokens = value.split()
        # 如果值的个数为0或者超过3个，则发出警告
        if len(tokens) == 0 or len(tokens) > 3:
            warnings.warn(
                f'Too many tokens provided to "{prop}" (expected 1-3)',
                CSSWarning,
                stacklevel=find_stack_level(),
            )

        # TODO: 是否可以使用当前颜色作为初始值以符合 CSS 标准？
        # 初始化边框属性的默认值为黑色、无样式、中等宽度
        border_declarations = {
            f"border{side}-color": "black",
            f"border{side}-style": "none",
            f"border{side}-width": "medium",
        }
        # 遍历每个分割出的单词
        for token in tokens:
            # 如果单词属于已知的边框样式列表，则设置对应的样式
            if token.lower() in self.BORDER_STYLES:
                border_declarations[f"border{side}-style"] = token
            # 如果单词包含在已知的边框宽度比例列表中，则设置对应的宽度
            elif any(ratio in token.lower() for ratio in self.BORDER_WIDTH_RATIOS):
                border_declarations[f"border{side}-width"] = token
            # 否则，将单词视为颜色并设置边框颜色
            else:
                border_declarations[f"border{side}-color"] = token
            # TODO: 如果用户输入了重复的项（例如 "border: red green"），则发出警告

        # 根据 CSS 规范，"border" 将重置先前的 "border-*" 定义
        # 使用生成器将扩展后的边框属性逐个输出
        yield from self.atomize(border_declarations.items())

    # 返回扩展函数
    return expand
    """
    一个用于解析和解析CSS到原子属性的可调用对象。
    """

    # 各种单位与pt单位的比率映射表
    UNIT_RATIOS = {
        "pt": ("pt", 1),
        "em": ("em", 1),
        "rem": ("pt", 12),
        "ex": ("em", 0.5),
        "px": ("pt", 0.75),
        "pc": ("pt", 12),
        "in": ("pt", 72),
        "cm": ("in", 1 / 2.54),
        "mm": ("in", 1 / 25.4),
        "q": ("mm", 0.25),
        "!!default": ("em", 0),
    }

    # 字体大小与em单位的比率映射表，继承并扩展了单位比率映射表
    FONT_SIZE_RATIOS = UNIT_RATIOS.copy()
    FONT_SIZE_RATIOS.update(
        {
            "%": ("em", 0.01),
            "xx-small": ("rem", 0.5),
            "x-small": ("rem", 0.625),
            "small": ("rem", 0.8),
            "medium": ("rem", 1),
            "large": ("rem", 1.125),
            "x-large": ("rem", 1.5),
            "xx-large": ("rem", 2),
            "smaller": ("em", 1 / 1.2),
            "larger": ("em", 1.2),
            "!!default": ("em", 1),
        }
    )

    # 边距与单位比率的映射表，继承并扩展了单位比率映射表，并添加了none的定义
    MARGIN_RATIOS = UNIT_RATIOS.copy()
    MARGIN_RATIOS.update({"none": ("pt", 0)})

    # 边框宽度与单位比率的映射表，继承并扩展了单位比率映射表，并添加了特定边框宽度的定义
    BORDER_WIDTH_RATIOS = UNIT_RATIOS.copy()
    BORDER_WIDTH_RATIOS.update(
        {
            "none": ("pt", 0),
            "thick": ("px", 4),
            "medium": ("px", 2),
            "thin": ("px", 1),
            # 默认：如果是solid样式，则为medium
        }
    )

    # 边框样式列表
    BORDER_STYLES = [
        "none",
        "hidden",
        "dotted",
        "dashed",
        "solid",
        "double",
        "groove",
        "ridge",
        "inset",
        "outset",
        "mediumdashdot",
        "dashdotdot",
        "hair",
        "mediumdashdotdot",
        "dashdot",
        "slantdashdot",
        "mediumdashed",
    ]

    # 边框缩写映射表，根据值的数量将其分成各种边的缩写
    SIDE_SHORTHANDS = {
        1: [0, 0, 0, 0],
        2: [0, 1, 0, 1],
        3: [0, 1, 2, 1],
        4: [0, 1, 2, 3],
    }

    # 边的顺序元组，表示顶部、右侧、底部、左侧
    SIDES = ("top", "right", "bottom", "left")

    # CSS属性扩展字典，包括边框和填充等扩展
    CSS_EXPANSIONS = {
        **{
            (f"border-{prop}" if prop else "border"): _border_expander(prop)
            for prop in ["", "top", "right", "bottom", "left"]
        },
        **{
            f"border-{prop}": _side_expander(f"border-{{:s}}-{prop}")
            for prop in ["color", "style", "width"]
        },
        "margin": _side_expander("margin-{:s}"),
        "padding": _side_expander("padding-{:s}"),
    }

    def __call__(
        self,
        declarations: str | Iterable[tuple[str, str]],
        inherited: dict[str, str] | None = None,
        ```
    def _update_initial(
        self,
        props: dict[str, str],
        inherited: dict[str, str],
    ) -> dict[str, str]:
        """
        Update the properties dictionary with inherited values and initial values where necessary.

        Parameters
        ----------
        props : dict[str, str]
            Dictionary containing atomic CSS properties to be updated.
        inherited : dict[str, str]
            Dictionary representing inherited CSS properties.

        Returns
        -------
        dict[str, str]
            Updated dictionary of atomic CSS properties.

        Notes
        -----
        This method ensures that properties marked with 'inherit' or 'initial' are resolved
        according to the given inherited context.

        """
        # 1. resolve inherited, initial
        # Iterate over inherited properties and update props if not already present
        for prop, val in inherited.items():
            if prop not in props:
                props[prop] = val

        # Create a copy of props to modify
        new_props = props.copy()

        # Iterate over props to resolve 'inherit' and 'initial' values
        for prop, val in props.items():
            if val == "inherit":
                val = inherited.get(prop, "initial")

            if val in ("initial", None):
                # Remove properties marked as 'initial' or None
                del new_props[prop]
            else:
                new_props[prop] = val

        return new_props
    # 从属性字典中获取字体大小（如果存在）
    def _get_font_size(self, props: dict[str, str]) -> float | None:
        if props.get("font-size"):
            font_size_string = props["font-size"]
            return self._get_float_font_size_from_pt(font_size_string)
        return None

    # 将以 pt 结尾的字体大小字符串转换为浮点数
    def _get_float_font_size_from_pt(self, font_size_string: str) -> float:
        assert font_size_string.endswith("pt")
        return float(font_size_string.rstrip("pt"))

    # 更新其他属性中使用的单位（如边框宽度、边距等）为以 pt 为单位
    def _update_other_units(self, props: dict[str, str]) -> dict[str, str]:
        font_size = self._get_font_size(props)
        
        # 处理各个方向的边框宽度
        for side in self.SIDES:
            prop = f"border-{side}-width"
            if prop in props:
                props[prop] = self.size_to_pt(
                    props[prop],
                    em_pt=font_size,
                    conversions=self.BORDER_WIDTH_RATIOS,
                )

            # 处理各个方向的边距
            for prop in [f"margin-{side}", f"padding-{side}"]:
                if prop in props:
                    props[prop] = self.size_to_pt(
                        props[prop],
                        em_pt=font_size,
                        conversions=self.MARGIN_RATIOS,
                    )
        return props

    # 将各种单位（如 em、%、pt 等）转换为以 pt 为单位的字符串表示
    def size_to_pt(
        self, in_val: str, em_pt: float | None = None, conversions: dict = UNIT_RATIOS
    ) -> str:
        # 处理错误情况，并尝试进行默认处理
        def _error() -> str:
            warnings.warn(
                f"Unhandled size: {in_val!r}",
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return self.size_to_pt("1!!default", conversions=conversions)

        # 匹配输入值中的数值和单位部分
        match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
        if match is None:
            return _error()

        val, unit = match.groups()
        if val == "":
            # 对于 'large' 等特殊情况的处理
            val = 1
        else:
            try:
                val = float(val)
            except ValueError:
                return _error()

        # 将单位转换为 pt 单位
        while unit != "pt":
            if unit == "em":
                if em_pt is None:
                    unit = "rem"
                else:
                    val *= em_pt
                    unit = "pt"
                continue

            try:
                unit, mul = conversions[unit]
            except KeyError:
                return _error()
            val *= mul

        # 四舍五入保留五位小数，并格式化为字符串表示
        val = round(val, 5)
        if int(val) == val:
            size_fmt = f"{int(val):d}pt"
        else:
            size_fmt = f"{val:f}pt"
        return size_fmt

    # 将 CSS 属性声明原子化为 (属性, 值) 的生成器，支持扩展
    def atomize(self, declarations: Iterable) -> Generator[tuple[str, str], None, None]:
        for prop, value in declarations:
            prop = prop.lower()
            value = value.lower()
            if prop in self.CSS_EXPANSIONS:
                expand = self.CSS_EXPANSIONS[prop]
                yield from expand(self, prop, value)
            else:
                yield prop, value
    def parse(self, declarations_str: str) -> Iterator[tuple[str, str]]:
        """
        从声明字符串中生成 (属性, 值) 对。

        在将来的版本中可能会生成从 tinycss/tinycss2 解析出的标记。

        Parameters
        ----------
        declarations_str : str
            包含CSS声明的字符串

        """
        # 通过分号分隔每个声明，处理每一个声明
        for decl in declarations_str.split(";"):
            # 如果声明为空或只包含空白字符，则跳过处理
            if not decl.strip():
                continue
            # 使用冒号分割属性和值
            prop, sep, val = decl.partition(":")
            prop = prop.strip().lower()  # 去除首尾空白并转换为小写
            # TODO: 不要转换值中大小写敏感的部分（例如字符串）
            val = val.strip().lower()  # 去除首尾空白并转换为小写
            # 如果分割成功，生成 (属性, 值) 的元组
            if sep:
                yield prop, val
            # 如果分割不成功，发出警告并提示声明格式不正确
            else:
                warnings.warn(
                    f"Ill-formatted attribute: expected a colon in {decl!r}",
                    CSSWarning,
                    stacklevel=find_stack_level(),
                )
```