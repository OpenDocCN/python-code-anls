# `D:\src\scipysrc\matplotlib\lib\matplotlib\markers.pyi`

```py
from typing import Literal  # 引入Literal类型提示

from .path import Path  # 从当前包导入Path类
from .transforms import Affine2D, Transform  # 从当前包导入Affine2D和Transform类

from numpy.typing import ArrayLike  # 导入ArrayLike类型提示
from .typing import CapStyleType, FillStyleType, JoinStyleType  # 从当前包导入CapStyleType、FillStyleType和JoinStyleType类型

TICKLEFT: int  # 定义整数常量TICKLEFT
TICKRIGHT: int  # 定义整数常量TICKRIGHT
TICKUP: int  # 定义整数常量TICKUP
TICKDOWN: int  # 定义整数常量TICKDOWN
CARETLEFT: int  # 定义整数常量CARETLEFT
CARETRIGHT: int  # 定义整数常量CARETRIGHT
CARETUP: int  # 定义整数常量CARETUP
CARETDOWN: int  # 定义整数常量CARETDOWN
CARETLEFTBASE: int  # 定义整数常量CARETLEFTBASE
CARETRIGHTBASE: int  # 定义整数常量CARETRIGHTBASE
CARETUPBASE: int  # 定义整数常量CARETUPBASE
CARETDOWNBASE: int  # 定义整数常量CARETDOWNBASE

class MarkerStyle:
    markers: dict[str | int, str]  # markers属性是一个字典，键为str或int类型，值为str类型
    filled_markers: tuple[str, ...]  # filled_markers属性是一个元组，元素为str类型
    fillstyles: tuple[FillStyleType, ...]  # fillstyles属性是一个元组，元素为FillStyleType类型

    def __init__(
        self,
        marker: str | ArrayLike | Path | MarkerStyle,  # 初始化方法，接受marker参数，类型为str或ArrayLike或Path或MarkerStyle
        fillstyle: FillStyleType | None = ...,  # fillstyle参数，类型为FillStyleType或None，默认为Ellipsis（...）
        transform: Transform | None = ...,  # transform参数，类型为Transform或None，默认为Ellipsis（...）
        capstyle: CapStyleType | None = ...,  # capstyle参数，类型为CapStyleType或None，默认为Ellipsis（...）
        joinstyle: JoinStyleType | None = ...,  # joinstyle参数，类型为JoinStyleType或None，默认为Ellipsis（...）
    ) -> None: ...  # 初始化方法没有具体实现

    def __bool__(self) -> bool: ...  # 定义__bool__方法，返回布尔值
    def is_filled(self) -> bool: ...  # 定义is_filled方法，返回布尔值
    def get_fillstyle(self) -> FillStyleType: ...  # 定义get_fillstyle方法，返回FillStyleType类型
    def get_joinstyle(self) -> Literal["miter", "round", "bevel"]: ...  # 定义get_joinstyle方法，返回"miter"、"round"或"bevel"
    def get_capstyle(self) -> Literal["butt", "projecting", "round"]: ...  # 定义get_capstyle方法，返回"butt"、"projecting"或"round"
    def get_marker(self) -> str | ArrayLike | Path | None: ...  # 定义get_marker方法，返回str、ArrayLike、Path或None类型
    def get_path(self) -> Path: ...  # 定义get_path方法，返回Path类型
    def get_transform(self) -> Transform: ...  # 定义get_transform方法，返回Transform类型
    def get_alt_path(self) -> Path | None: ...  # 定义get_alt_path方法，返回Path或None类型
    def get_alt_transform(self) -> Transform: ...  # 定义get_alt_transform方法，返回Transform类型
    def get_snap_threshold(self) -> float | None: ...  # 定义get_snap_threshold方法，返回float或None类型
    def get_user_transform(self) -> Transform | None: ...  # 定义get_user_transform方法，返回Transform或None类型
    def transformed(self, transform: Affine2D) -> MarkerStyle: ...  # 定义transformed方法，接受Affine2D类型的参数，返回MarkerStyle类型
    def rotated(
        self, *, deg: float | None = ..., rad: float | None = ...
    ) -> MarkerStyle: ...  # 定义rotated方法，接受关键字参数deg和rad，类型为float或None，默认值为Ellipsis（...）
    def scaled(self, sx: float, sy: float | None = ...) -> MarkerStyle: ...  # 定义scaled方法，接受参数sx和sy，sx为float类型，sy为float或None类型，默认值为Ellipsis（...）
```