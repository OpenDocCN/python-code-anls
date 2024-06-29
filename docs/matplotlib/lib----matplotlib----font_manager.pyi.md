# `D:\src\scipysrc\matplotlib\lib\matplotlib\font_manager.pyi`

```py
# 导入必要的模块和类
from dataclasses import dataclass
import os
from matplotlib._afm import AFM
from matplotlib import ft2font
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Literal

# 定义全局变量
font_scalings: dict[str | None, float]  # 字体缩放比例的字典
stretch_dict: dict[str, int]  # 字体拉伸属性的字典
weight_dict: dict[str, int]  # 字体粗细属性的字典
font_family_aliases: set[str]  # 字体族的别名集合
MSFolders: str  # Microsoft Windows 字体文件夹路径
MSFontDirectories: list[str]  # Microsoft Windows 字体文件夹路径列表
MSUserFontDirectories: list[str]  # Microsoft Windows 用户字体文件夹路径列表
X11FontDirectories: list[str]  # X11 字体文件夹路径列表
OSXFontDirectories: list[str]  # macOS 字体文件夹路径列表

# 定义函数签名，暂不提供具体实现
def get_fontext_synonyms(fontext: str) -> list[str]: ...
def list_fonts(directory: str, extensions: Iterable[str]) -> list[str]: ...
def win32FontDirectory() -> str: ...
def _get_fontconfig_fonts() -> list[Path]: ...
def findSystemFonts(
    fontpaths: Iterable[str | os.PathLike | Path] | None = ..., fontext: str = ...
) -> list[str]: ...

# 定义字体条目的数据类
@dataclass
class FontEntry:
    fname: str = ...  # 字体文件名
    name: str = ...  # 字体名称
    style: str = ...  # 字体风格
    variant: str = ...  # 字体变体
    weight: str | int = ...  # 字体粗细
    stretch: str = ...  # 字体拉伸
    size: str = ...  # 字体大小

    def _repr_html_(self) -> str: ...  # 返回 HTML 格式的字体条目表示
    def _repr_png_(self) -> bytes: ...  # 返回 PNG 格式的字体条目表示

# 定义 TTF 字体属性函数
def ttfFontProperty(font: ft2font.FT2Font) -> FontEntry: ...

# 定义 AFM 字体属性函数
def afmFontProperty(fontpath: str, font: AFM) -> FontEntry: ...

# 定义字体属性类
class FontProperties:
    def __init__(
        self,
        family: str | Iterable[str] | None = ...,
        style: Literal["normal", "italic", "oblique"] | None = ...,
        variant: Literal["normal", "small-caps"] | None = ...,
        weight: int | str | None = ...,
        stretch: int | str | None = ...,
        size: float | str | None = ...,
        fname: str | os.PathLike | Path | None = ...,
        math_fontfamily: str | None = ...,
    ) -> None: ...
    
    def __hash__(self) -> int: ...  # 返回对象的哈希值
    def __eq__(self, other: object) -> bool: ...  # 检查对象是否相等
    def get_family(self) -> list[str]: ...  # 返回字体族列表
    def get_name(self) -> str: ...  # 返回字体名称
    def get_style(self) -> Literal["normal", "italic", "oblique"]: ...  # 返回字体风格
    def get_variant(self) -> Literal["normal", "small-caps"]: ...  # 返回字体变体
    def get_weight(self) -> int | str: ...  # 返回字体粗细
    def get_stretch(self) -> int | str: ...  # 返回字体拉伸
    def get_size(self) -> float: ...  # 返回字体大小
    def get_file(self) -> str | bytes | None: ...  # 返回字体文件路径或内容
    def get_fontconfig_pattern(self) -> dict[str, list[Any]]: ...  # 返回字体配置模式
    def set_family(self, family: str | Iterable[str] | None) -> None: ...  # 设置字体族
    def set_style(
        self, style: Literal["normal", "italic", "oblique"] | None
    ) -> None: ...  # 设置字体风格
    def set_variant(self, variant: Literal["normal", "small-caps"] | None) -> None: ...  # 设置字体变体
    def set_weight(self, weight: int | str | None) -> None: ...  # 设置字体粗细
    def set_stretch(self, stretch: int | str | None) -> None: ...  # 设置字体拉伸
    def set_size(self, size: float | str | None) -> None: ...  # 设置字体大小
    def set_file(self, file: str | os.PathLike | Path | None) -> None: ...  # 设置字体文件路径或内容
    def set_fontconfig_pattern(self, pattern: str) -> None: ...  # 设置字体配置模式
    def get_math_fontfamily(self) -> str: ...  # 返回数学字体族
    def set_math_fontfamily(self, fontfamily: str | None) -> None: ...  # 设置数学字体族
    def copy(self) -> FontProperties: ...  # 复制当前对象的副本

    # 别名
    # 将 set_name 设置为 set_family 的引用
    set_name = set_family
    # 将 get_slant 设置为 get_style 的引用
    get_slant = get_style
    # 将 set_slant 设置为 set_style 的引用
    set_slant = set_style
    # 将 get_size_in_points 设置为 get_size 的引用
    get_size_in_points = get_size
# 将 FontManager 实例数据序列化为 JSON 文件
def json_dump(data: FontManager, filename: str | Path | os.PathLike) -> None:
    ...

# 从 JSON 文件加载数据并创建 FontManager 实例
def json_load(filename: str | Path | os.PathLike) -> FontManager:
    ...

class FontManager:
    __version__: int  # FontManager 的版本号
    default_size: float | None  # 默认字体大小，可能为 None
    defaultFamily: dict[str, str]  # 默认字体族的字典，映射字体类型到名称
    afmlist: list[FontEntry]  # AFM 格式字体文件的列表
    ttflist: list[FontEntry]  # TTF 格式字体文件的列表

    # 初始化 FontManager 实例，size 表示默认大小，weight 表示默认字体粗细
    def __init__(self, size: float | None = ..., weight: str = ...) -> None:
        ...

    # 添加字体文件到 FontManager 实例
    def addfont(self, path: str | Path | os.PathLike) -> None:
        ...

    @property
    # 获取默认字体的字典，映射字体类型到名称
    def defaultFont(self) -> dict[str, str]:
        ...

    # 获取默认字体的粗细
    def get_default_weight(self) -> str:
        ...

    @staticmethod
    # 静态方法，获取默认字体大小
    def get_default_size() -> float:
        ...

    # 设置默认字体的粗细
    def set_default_weight(self, weight: str) -> None:
        ...

    # 比较两个字体族的相似度得分
    def score_family(
        self, families: str | list[str] | tuple[str], family2: str
    ) -> float:
        ...

    # 比较两个字体风格的相似度得分
    def score_style(self, style1: str, style2: str) -> float:
        ...

    # 比较两个字体变体的相似度得分
    def score_variant(self, variant1: str, variant2: str) -> float:
        ...

    # 比较两个字体拉伸度的相似度得分
    def score_stretch(self, stretch1: str | int, stretch2: str | int) -> float:
        ...

    # 比较两个字体粗细的相似度得分
    def score_weight(self, weight1: str | float, weight2: str | float) -> float:
        ...

    # 比较两个字体大小的相似度得分
    def score_size(self, size1: str | float, size2: str | float) -> float:
        ...

    # 根据字体属性查找最合适的字体文件
    def findfont(
        self,
        prop: str | FontProperties,
        fontext: Literal["ttf", "afm"] = ...,
        directory: str | None = ...,
        fallback_to_default: bool = ...,
        rebuild_if_missing: bool = ...,
    ) -> str:
        ...

    # 获取当前 FontManager 实例管理的所有字体名称
    def get_font_names(self) -> list[str]:
        ...

# 判断给定文件名的字体文件是否为 OpenType CFF 格式
def is_opentype_cff_font(filename: str) -> bool:
    ...

# 根据给定的字体文件路径或字节流获取对应的 FT2Font 对象
def get_font(
    font_filepaths: Iterable[str | Path | bytes] | str | Path | bytes,
    hinting_factor: int | None = ...,
) -> ft2font.FT2Font:
    ...

fontManager: FontManager  # 全局 FontManager 实例

# 在全局 FontManager 实例中根据字体属性查找最合适的字体文件
def findfont(
    prop: str | FontProperties,
    fontext: Literal["ttf", "afm"] = ...,
    directory: str | None = ...,
    fallback_to_default: bool = ...,
    rebuild_if_missing: bool = ...,
) -> str:
    ...

# 获取全局 FontManager 实例管理的所有字体名称
def get_font_names() -> list[str]:
    ...
```