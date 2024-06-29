# `D:\src\scipysrc\matplotlib\lib\matplotlib\rcsetup.pyi`

```
from cycler import Cycler  # 导入 Cycler 类，用于循环参数生成

from collections.abc import Callable, Iterable  # 导入 Callable 和 Iterable 类型
from typing import Any, Literal, TypeVar  # 导入 Any、Literal 和 TypeVar 类型
from matplotlib.typing import ColorType, LineStyleType, MarkEveryType  # 导入 ColorType、LineStyleType 和 MarkEveryType 类型

interactive_bk: list[str]  # 定义 interactive_bk 变量，类型为字符串列表
non_interactive_bk: list[str]  # 定义 non_interactive_bk 变量，类型为字符串列表
all_backends: list[str]  # 定义 all_backends 变量，类型为字符串列表

_T = TypeVar("_T")  # 创建泛型变量 _T

def _listify_validator(s: Callable[[Any], _T]) -> Callable[[Any], list[_T]]: ...
    # 定义函数 _listify_validator，接受一个参数 s，类型为接受任意类型并返回 _T 类型的可调用对象，
    # 返回一个接受任意类型并返回 _T 类型列表的可调用对象

class ValidateInStrings:
    key: str  # 定义属性 key，类型为字符串
    ignorecase: bool  # 定义属性 ignorecase，类型为布尔值
    valid: dict[str, str]  # 定义属性 valid，类型为字符串键和字符串值的字典

    def __init__(
        self,
        key: str,
        valid: Iterable[str],
        ignorecase: bool = ...,
        *,
        _deprecated_since: str | None = ...
    ) -> None: ...
        # 初始化方法，接受 key（字符串）、valid（字符串可迭代对象）、ignorecase（布尔值，可选，默认...）、
        # _deprecated_since（字符串或空值，可选），无返回值

    def __call__(self, s: Any) -> str: ...
        # 实例可调用方法，接受参数 s（任意类型），返回字符串

def validate_any(s: Any) -> Any: ...
    # 验证函数，接受任意类型参数 s，返回任意类型

def validate_anylist(s: Any) -> list[Any]: ...
    # 验证函数，接受任意类型参数 s，返回任意类型列表

def validate_bool(b: Any) -> bool: ...
    # 验证函数，接受任意类型参数 b，返回布尔值

def validate_axisbelow(s: Any) -> bool | Literal["line"]: ...
    # 验证函数，接受任意类型参数 s，返回布尔值或特定字符串 "line"

def validate_dpi(s: Any) -> Literal["figure"] | float: ...
    # 验证函数，接受任意类型参数 s，返回 "figure" 字符串或浮点数

def validate_string(s: Any) -> str: ...
    # 验证函数，接受任意类型参数 s，返回字符串

def validate_string_or_None(s: Any) -> str | None: ...
    # 验证函数，接受任意类型参数 s，返回字符串或空值

def validate_stringlist(s: Any) -> list[str]: ...
    # 验证函数，接受任意类型参数 s，返回字符串列表

def validate_int(s: Any) -> int: ...
    # 验证函数，接受任意类型参数 s，返回整数

def validate_int_or_None(s: Any) -> int | None: ...
    # 验证函数，接受任意类型参数 s，返回整数或空值

def validate_float(s: Any) -> float: ...
    # 验证函数，接受任意类型参数 s，返回浮点数

def validate_float_or_None(s: Any) -> float | None: ...
    # 验证函数，接受任意类型参数 s，返回浮点数或空值

def validate_floatlist(s: Any) -> list[float]: ...
    # 验证函数，接受任意类型参数 s，返回浮点数列表

def _validate_marker(s: Any) -> int | str: ...
    # 验证函数，接受任意类型参数 s，返回整数或字符串

def _validate_markerlist(s: Any) -> list[int | str]: ...
    # 验证函数，接受任意类型参数 s，返回整数或字符串列表

def validate_fonttype(s: Any) -> int: ...
    # 验证函数，接受任意类型参数 s，返回整数

_auto_backend_sentinel: object  # 定义 _auto_backend_sentinel 变量，类型为对象

def validate_backend(s: Any) -> str: ...
    # 验证函数，接受任意类型参数 s，返回字符串

def validate_color_or_inherit(s: Any) -> Literal["inherit"] | ColorType: ...
    # 验证函数，接受任意类型参数 s，返回 "inherit" 字符串或颜色类型

def validate_color_or_auto(s: Any) -> ColorType | Literal["auto"]: ...
    # 验证函数，接受任意类型参数 s，返回颜色类型或 "auto" 字符串

def validate_color_for_prop_cycle(s: Any) -> ColorType: ...
    # 验证函数，接受任意类型参数 s，返回颜色类型

def validate_color(s: Any) -> ColorType: ...
    # 验证函数，接受任意类型参数 s，返回颜色类型

def validate_colorlist(s: Any) -> list[ColorType]: ...
    # 验证函数，接受任意类型参数 s，返回颜色类型列表

def _validate_color_or_linecolor(
    s: Any,
) -> ColorType | Literal["linecolor", "markerfacecolor", "markeredgecolor"] | None: ...
    # 验证函数，接受任意类型参数 s，返回颜色类型、特定字符串或空值

def validate_aspect(s: Any) -> Literal["auto", "equal"] | float: ...
    # 验证函数，接受任意类型参数 s，返回 "auto"、"equal" 字符串或浮点数

def validate_fontsize_None(
    s: Any,
) -> Literal[
    "xx-small",
    "x-small",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
    "smaller",
    "larger",
] | float | None: ...
    # 验证函数，接受任意类型参数 s，返回特定字符串、浮点数或空值

def validate_fontsize(
    s: Any,
) -> Literal[
    "xx-small",
    "x-small",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
    "smaller",
    "larger",
] | float: ...
    # 验证函数，接受任意类型参数 s，返回特定字符串或浮点数

def validate_fontsizelist(
    s: Any,
) -> list[
    Literal[
        "xx-small",
        "x-small",
        "small",
        "medium",
        "large",
        "x-large",
        "xx-large",
        "smaller",
        "larger",
    ]
    | float
]: ...
    # 验证函数，接受任意类型参数 s，返回特定字符串或浮点数列表

def validate_fontweight(
    s: Any,
) -> Literal[
    "ultralight",
    "light",
    "normal",
    "regular",
    "book",
    "medium",
    "roman",
    "semibold",
    "demibold",
    "demi",
    "bold",
    "heavy",
    "extra bold",
    "black",
] | int: ...
    # 验证函数，接受任意类型参数 s，返回特定字符串或整数

def validate_fontstretch(
    s: Any,
) -> Literal[
    "ultra-condensed",
    "extra-condensed",
    "condensed",
    "semi-condensed",
    "normal",
    "semi-expanded",
    "expanded",
    "extra-expanded",
    "ultra-expanded",
] | int: ...
    # 验证函数，接受任意类型参数 s，返回特定字符串或整数
# 定义了一个类型别名，表示可以是以下字符串之一或整数
def validate_font_properties(
    s: Any
) -> Literal[
    "ultra-condensed",
    "extra-condensed",
    "condensed",
    "semi-condensed",
    "normal",
    "semi-expanded",
    "expanded",
    "extra-expanded",
    "ultra-expanded",
] | int: ...

# 验证函数，接受任意类型的输入并返回一个字典
def validate_font_properties(s: Any) -> dict[str, Any]: ...

# 验证函数，接受任意类型的输入并返回浮点数列表或单个浮点数
def validate_whiskers(s: Any) -> list[float] | float: ...

# 验证函数，接受任意类型的输入并不返回任何值或指定字符串之一
def validate_ps_distiller(s: Any) -> None | Literal["ghostscript", "xpdf"]: ...

validate_fillstyle: ValidateInStrings

# 验证函数，接受任意类型的输入并返回包含特定字符串之一的列表
def validate_fillstylelist(
    s: Any,
) -> list[Literal["full", "left", "right", "bottom", "top", "none"]]: ...

# 验证函数，接受任意类型的输入并返回标记间隔类型
def validate_markevery(s: Any) -> MarkEveryType: ...

# 验证函数，接受任意类型的输入并返回线型类型
def _validate_linestyle(s: Any) -> LineStyleType: ...

# 验证函数，接受任意类型的输入并返回标记间隔类型的列表
def validate_markeverylist(s: Any) -> list[MarkEveryType]: ...

# 验证函数，接受任意类型的输入并返回特定字符串之一或无值
def validate_bbox(s: Any) -> Literal["tight", "standard"] | None: ...

# 验证函数，接受任意类型的输入并返回无值或包含三个浮点数的元组
def validate_sketch(s: Any) -> None | tuple[float, float, float]: ...

# 验证函数，接受任意类型的输入并返回字符串
def validate_hatch(s: Any) -> str: ...

# 验证函数，接受任意类型的输入并返回字符串列表
def validate_hatchlist(s: Any) -> list[str]: ...

# 验证函数，接受任意类型的输入并返回包含浮点数列表的列表
def validate_dashlist(s: Any) -> list[list[float]]: ...

# 函数签名为 `cycler` 的定义，可以接受任意参数并返回 `Cycler` 类型的结果
def cycler(*args, **kwargs) -> Cycler: ...

# 验证函数，接受任意类型的输入并返回 `Cycler` 类型的结果
def validate_cycler(s: Any) -> Cycler: ...

# 验证函数，接受任意类型的输入并返回特定字符串、整数或浮点数列表之一
def validate_hist_bins(
    s: Any,
) -> Literal["auto", "sturges", "fd", "doane", "scott", "rice", "sqrt"] | int | list[
    float
]: ...

# 运行时添加到 `__init__.py` 中的默认参数字典
defaultParams: dict[str, Any]
```