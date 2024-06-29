# `D:\src\scipysrc\matplotlib\lib\matplotlib\textpath.pyi`

```py
# 导入所需的模块和类
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.mathtext import MathTextParser, VectorParse
from matplotlib.path import Path

import numpy as np

from typing import Literal

# 定义一个名为TextToPath的类，用于将文本转换为路径
class TextToPath:
    FONT_SCALE: float  # 字体缩放比例
    DPI: float  # DPI（每英寸点数）
    mathtext_parser: MathTextParser[VectorParse]  # 数学文本解析器对象

    # 初始化方法，无具体实现
    def __init__(self) -> None: ...

    # 返回文本宽度、高度和下降距离的元组
    def get_text_width_height_descent(
        self, s: str, prop: FontProperties, ismath: bool | Literal["TeX"]
    ) -> tuple[float, float, float]: ...

    # 返回文本路径的列表
    def get_text_path(
        self, prop: FontProperties, s: str, ismath: bool | Literal["TeX"] = ...
    ) -> list[np.ndarray]: ...

    # 使用指定字体获取文本的字形信息
    def get_glyphs_with_font(
        self,
        font: FT2Font,
        s: str,
        glyph_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = ...,
        return_new_glyphs_only: bool = ...,
    ) -> tuple[
        list[tuple[str, float, float, float]],
        dict[str, tuple[np.ndarray, np.ndarray]],
        list[tuple[list[tuple[float, float]], list[int]]],
    ]: ...

    # 使用数学文本属性获取文本的字形信息
    def get_glyphs_mathtext(
        self,
        prop: FontProperties,
        s: str,
        glyph_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = ...,
        return_new_glyphs_only: bool = ...,
    ) -> tuple[
        list[tuple[str, float, float, float]],
        dict[str, tuple[np.ndarray, np.ndarray]],
        list[tuple[list[tuple[float, float]], list[int]]],
    ]: ...

    # 使用TeX文本属性获取文本的字形信息
    def get_glyphs_tex(
        self,
        prop: FontProperties,
        s: str,
        glyph_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = ...,
        return_new_glyphs_only: bool = ...,
    ) -> tuple[
        list[tuple[str, float, float, float]],
        dict[str, tuple[np.ndarray, np.ndarray]],
        list[tuple[list[tuple[float, float]], list[int]]],
    ]: ...

# 定义名为text_to_path的TextToPath类的实例
text_to_path: TextToPath

# 继承自Path类的文本路径类TextPath
class TextPath(Path):
    # 初始化方法，初始化文本路径对象
    def __init__(
        self,
        xy: tuple[float, float],  # 文本路径的位置坐标
        s: str,  # 文本内容
        size: float | None = ...,  # 字体大小（可选）
        prop: FontProperties | None = ...,  # 字体属性（可选）
        _interpolation_steps: int = ...,  # 插值步数
        usetex: bool = ...,  # 是否使用TeX
    ) -> None: ...

    # 设置文本路径对象的字体大小
    def set_size(self, size: float | None) -> None: ...

    # 获取文本路径对象的字体大小
    def get_size(self) -> float | None: ...

    # 只读属性：获取文本路径对象的顶点坐标
    @property  # type: ignore[misc]
    def vertices(self) -> np.ndarray: ...  # type: ignore[override]

    # 只读属性：获取文本路径对象的代码
    @property  # type: ignore[misc]
    def codes(self) -> np.ndarray: ...  # type: ignore[override]
```