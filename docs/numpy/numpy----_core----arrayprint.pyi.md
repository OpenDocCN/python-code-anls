# `.\numpy\numpy\_core\arrayprint.pyi`

```py
# 导入必要的模块和类型
from collections.abc import Callable
from typing import Any, Literal, TypedDict, SupportsIndex

# 导入私有类以及 contextlib 模块中的 _GeneratorContextManager
# 注意，这是由于 `contextlib.context` 返回的结果是上述类的实例
from contextlib import _GeneratorContextManager

# 导入 numpy 库，并从中导入多个类型和函数
import numpy as np
from numpy import (
    integer,
    timedelta64,
    datetime64,
    floating,
    complexfloating,
    void,
    longdouble,
    clongdouble,
)
from numpy._typing import NDArray, _CharLike_co, _FloatLike_co

# 定义一个字面量类型 _FloatMode
_FloatMode = Literal["fixed", "unique", "maxprec", "maxprec_equal"]

# 定义一个 TypedDict 类型 _FormatDict，用于描述格式化函数的字典结构
class _FormatDict(TypedDict, total=False):
    bool: Callable[[np.bool], str]
    int: Callable[[integer[Any]], str]
    timedelta: Callable[[timedelta64], str]
    datetime: Callable[[datetime64], str]
    float: Callable[[floating[Any]], str]
    longfloat: Callable[[longdouble], str]
    complexfloat: Callable[[complexfloating[Any, Any]], str]
    longcomplexfloat: Callable[[clongdouble], str]
    void: Callable[[void], str]
    numpystr: Callable[[_CharLike_co], str]
    object: Callable[[object], str]
    all: Callable[[object], str]
    int_kind: Callable[[integer[Any]], str]
    float_kind: Callable[[floating[Any]], str]
    complex_kind: Callable[[complexfloating[Any, Any]], str]
    str_kind: Callable[[_CharLike_co], str]

# 定义一个 TypedDict 类型 _FormatOptions，描述打印选项的字典结构
class _FormatOptions(TypedDict):
    precision: int
    threshold: int
    edgeitems: int
    linewidth: int
    suppress: bool
    nanstr: str
    infstr: str
    formatter: None | _FormatDict
    sign: Literal["-", "+", " "]
    floatmode: _FloatMode
    legacy: Literal[False, "1.13", "1.21"]

# 定义 set_printoptions 函数，用于设置打印选项
def set_printoptions(
    precision: None | SupportsIndex = ...,
    threshold: None | int = ...,
    edgeitems: None | int = ...,
    linewidth: None | int = ...,
    suppress: None | bool = ...,
    nanstr: None | str = ...,
    infstr: None | str = ...,
    formatter: None | _FormatDict = ...,
    sign: Literal[None, "-", "+", " "] = ...,
    floatmode: None | _FloatMode = ...,
    *,
    legacy: Literal[None, False, "1.13", "1.21"] = ...,
    override_repr: None | Callable[[NDArray[Any]], str] = ...,
) -> None:
    ...

# 定义 get_printoptions 函数，用于获取当前的打印选项
def get_printoptions() -> _FormatOptions:
    ...

# 定义 array2string 函数，用于将数组转换为字符串表示
def array2string(
    a: NDArray[Any],
    max_line_width: None | int = ...,
    precision: None | SupportsIndex = ...,
    suppress_small: None | bool = ...,
    separator: str = ...,
    prefix: str = ...,
    # 注意：由于 `style` 参数已被弃用，
    # 在 `formatter` 和 `suffix` 之间的所有参数实际上是关键字参数
    *,
    formatter: None | _FormatDict = ...,
    threshold: None | int = ...,
    edgeitems: None | int = ...,
    sign: Literal[None, "-", "+", " "] = ...,
    floatmode: None | _FloatMode = ...,
    suffix: str = ...,
    legacy: Literal[None, False, "1.13", "1.21"] = ...,
) -> str:
    ...

# 定义 format_float_scientific 函数，用于科学计数法格式化浮点数
def format_float_scientific(
    x: _FloatLike_co,
    precision: None | int = ...,
    unique: bool = ...,
    ...
):  # 省略了部分函数定义，这里为示例注释，具体实现需要完整代码
    ...
    trim: Literal["k", ".", "0", "-"] = ...,
    # trim 变量类型为 Literal 类型，表示其取值只能是 "k", ".", "0", "-" 中的一种
    sign: bool = ...,
    # sign 变量类型为布尔型，表示其取值只能是 True 或 False
    pad_left: None | int = ...,
    # pad_left 变量类型为 None 或整数型，表示可以是 None 或任意整数值
    exp_digits: None | int = ...,
    # exp_digits 变量类型为 None 或整数型，表示可以是 None 或任意整数值
    min_digits: None | int = ...,
    # min_digits 变量类型为 None 或整数型，表示可以是 None 或任意整数值
# 定义格式化浮点数为字符串的函数，返回格式化后的字符串表示
def format_float_positional(
    x: _FloatLike_co,                   # 浮点数值或其它兼容类型的输入参数
    precision: None | int = ...,         # 精度参数，控制小数点后数字的位数，可以为None或整数
    unique: bool = ...,                  # 唯一性参数，控制是否强制唯一表示
    fractional: bool = ...,              # 分数参数，控制是否输出小数
    trim: Literal["k", ".", "0", "-"] = ...,  # 修剪参数，指定修剪策略的枚举值
    sign: bool = ...,                    # 符号参数，控制是否显示正负号
    pad_left: None | int = ...,          # 左填充参数，指定左侧填充的字符数，可以为None或整数
    pad_right: None | int = ...,         # 右填充参数，指定右侧填充的字符数，可以为None或整数
    min_digits: None | int = ...,        # 最小数字参数，指定输出的最小数字位数，可以为None或整数
) -> str: ...                            # 返回值为格式化后的字符串

# 返回数组的字符串表示形式
def array_repr(
    arr: NDArray[Any],                  # 输入数组，可以包含任何类型的元素
    max_line_width: None | int = ...,   # 最大行宽参数，指定每行的最大字符数，可以为None或整数
    precision: None | SupportsIndex = ...,  # 精度参数，控制小数点后数字的位数或索引支持，可以为None或支持索引的类型
    suppress_small: None | bool = ...,  # 抑制小数参数，控制是否抑制小数
) -> str: ...                           # 返回值为数组的字符串表示形式

# 返回数组的可打印字符串表示形式
def array_str(
    a: NDArray[Any],                    # 输入数组，可以包含任何类型的元素
    max_line_width: None | int = ...,   # 最大行宽参数，指定每行的最大字符数，可以为None或整数
    precision: None | SupportsIndex = ...,  # 精度参数，控制小数点后数字的位数或索引支持，可以为None或支持索引的类型
    suppress_small: None | bool = ...,  # 抑制小数参数，控制是否抑制小数
) -> str: ...                           # 返回值为数组的可打印字符串表示形式

# 设置打印选项的上下文管理器，返回格式选项的生成器
def printoptions(
    precision: None | SupportsIndex = ...,  # 精度参数，控制小数点后数字的位数或索引支持，可以为None或支持索引的类型
    threshold: None | int = ...,         # 阈值参数，控制元素的显示阈值，可以为None或整数
    edgeitems: None | int = ...,         # 边缘项目参数，指定在数组边缘显示的项目数，可以为None或整数
    linewidth: None | int = ...,         # 行宽参数，指定每行的最大字符数，可以为None或整数
    suppress: None | bool = ...,         # 抑制参数，控制是否抑制小数
    nanstr: None | str = ...,            # NaN字符串参数，指定NaN的字符串表示形式，可以为None或字符串
    infstr: None | str = ...,            # 无穷大字符串参数，指定无穷大的字符串表示形式，可以为None或字符串
    formatter: None | _FormatDict = ...,  # 格式化器参数，指定格式化字典，可以为None或格式化字典
    sign: Literal[None, "-", "+", " "] = ...,  # 符号参数，指定符号的表示形式，枚举值为None, "-", "+", " "
    floatmode: None | _FloatMode = ...,  # 浮点模式参数，指定浮点数的表示模式，可以为None或浮点模式类型
    *,
    legacy: Literal[None, False, "1.13", "1.21"] = ...  # 遗留模式参数，指定遗留模式的枚举值
) -> _GeneratorContextManager[_FormatOptions]: ...  # 返回值为格式选项的上下文管理器生成器
```