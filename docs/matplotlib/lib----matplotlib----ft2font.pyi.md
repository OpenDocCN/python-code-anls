# `D:\src\scipysrc\matplotlib\lib\matplotlib\ft2font.pyi`

```
from typing import BinaryIO, Literal, TypedDict, overload
# 引入需要的类型和注解相关模块

import numpy as np
# 引入 numpy 库

from numpy.typing import NDArray
# 从 numpy.typing 模块中引入 NDArray 类型注解

__freetype_build_type__: str
__freetype_version__: str
# 定义两个全局变量 __freetype_build_type__ 和 __freetype_version__

BOLD: int
EXTERNAL_STREAM: int
FAST_GLYPHS: int
FIXED_SIZES: int
FIXED_WIDTH: int
GLYPH_NAMES: int
HORIZONTAL: int
ITALIC: int
KERNING: int
KERNING_DEFAULT: int
KERNING_UNFITTED: int
KERNING_UNSCALED: int
LOAD_CROP_BITMAP: int
LOAD_DEFAULT: int
LOAD_FORCE_AUTOHINT: int
LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH: int
LOAD_IGNORE_TRANSFORM: int
LOAD_LINEAR_DESIGN: int
LOAD_MONOCHROME: int
LOAD_NO_AUTOHINT: int
LOAD_NO_BITMAP: int
LOAD_NO_HINTING: int
LOAD_NO_RECURSE: int
LOAD_NO_SCALE: int
LOAD_PEDANTIC: int
LOAD_RENDER: int
LOAD_TARGET_LCD: int
LOAD_TARGET_LCD_V: int
LOAD_TARGET_LIGHT: int
LOAD_TARGET_MONO: int
LOAD_TARGET_NORMAL: int
LOAD_VERTICAL_LAYOUT: int
MULTIPLE_MASTERS: int
SCALABLE: int
SFNT: int
VERTICAL: int
# 定义多个常量，可能用于指定不同的字体加载选项

class _SfntHeadDict(TypedDict):
    version: tuple[int, int]
    fontRevision: tuple[int, int]
    checkSumAdjustment: int
    magicNumber: int
    flags: int
    unitsPerEm: int
    created: tuple[int, int]
    modified: tuple[int, int]
    xMin: int
    yMin: int
    xMax: int
    yMax: int
    macStyle: int
    lowestRecPPEM: int
    fontDirectionHint: int
    indexToLocFormat: int
    glyphDataFormat: int
# 定义字典类型 _SfntHeadDict，包含字体文件头部信息的字段定义

class _SfntMaxpDict(TypedDict):
    version: tuple[int, int]
    numGlyphs: int
    maxPoints: int
    maxContours: int
    maxComponentPoints: int
    maxComponentContours: int
    maxZones: int
    maxTwilightPoints: int
    maxStorage: int
    maxFunctionDefs: int
    maxInstructionDefs: int
    maxStackElements: int
    maxSizeOfInstructions: int
    maxComponentElements: int
    maxComponentDepth: int
# 定义字典类型 _SfntMaxpDict，包含字体最大参数信息的字段定义

class _SfntOs2Dict(TypedDict):
    version: int
    xAvgCharWidth: int
    usWeightClass: int
    usWidthClass: int
    fsType: int
    ySubscriptXSize: int
    ySubscriptYSize: int
    ySubscriptXOffset: int
    ySubscriptYOffset: int
    ySuperscriptXSize: int
    ySuperscriptYSize: int
    ySuperscriptXOffset: int
    ySuperscriptYOffset: int
    yStrikeoutSize: int
    yStrikeoutPosition: int
    sFamilyClass: int
    panose: bytes
    ulCharRange: tuple[int, int, int, int]
    achVendID: bytes
    fsSelection: int
    fsFirstCharIndex: int
    fsLastCharIndex: int
# 定义字典类型 _SfntOs2Dict，包含字体操作系统特定信息的字段定义

class _SfntHheaDict(TypedDict):
    version: tuple[int, int]
    ascent: int
    descent: int
    lineGap: int
    advanceWidthMax: int
    minLeftBearing: int
    minRightBearing: int
    xMaxExtent: int
    caretSlopeRise: int
    caretSlopeRun: int
    caretOffset: int
    metricDataFormat: int
    numOfLongHorMetrics: int
# 定义字典类型 _SfntHheaDict，包含字体水平头部信息的字段定义

class _SfntVheaDict(TypedDict):
    version: tuple[int, int]
    vertTypoAscender: int
    vertTypoDescender: int
    vertTypoLineGap: int
    advanceHeightMax: int
    minTopSideBearing: int
    minBottomSizeBearing: int
    yMaxExtent: int
    caretSlopeRise: int
    caretSlopeRun: int
    caretOffset: int
    metricDataFormat: int
    numOfLongVerMetrics: int
# 定义字典类型 _SfntVheaDict，包含字体垂直头部信息的字段定义
# 定义一个类型字典，用于表示 "_SfntPostDict" 类型的数据结构
class _SfntPostDict(TypedDict):
    format: tuple[int, int]  # 指定 'format' 键对应的值是一个由两个整数组成的元组
    italicAngle: tuple[int, int]  # 指定 'italicAngle' 键对应的值是一个由两个整数组成的元组
    underlinePosition: int  # 指定 'underlinePosition' 键对应的值是一个整数
    underlineThickness: int  # 指定 'underlineThickness' 键对应的值是一个整数
    isFixedPitch: int  # 指定 'isFixedPitch' 键对应的值是一个整数
    minMemType42: int  # 指定 'minMemType42' 键对应的值是一个整数
    maxMemType42: int  # 指定 'maxMemType42' 键对应的值是一个整数
    minMemType1: int  # 指定 'minMemType1' 键对应的值是一个整数
    maxMemType1: int  # 指定 'maxMemType1' 键对应的值是一个整数

# 定义一个类型字典，用于表示 "_SfntPcltDict" 类型的数据结构
class _SfntPcltDict(TypedDict):
    version: tuple[int, int]  # 指定 'version' 键对应的值是一个由两个整数组成的元组
    fontNumber: int  # 指定 'fontNumber' 键对应的值是一个整数
    pitch: int  # 指定 'pitch' 键对应的值是一个整数
    xHeight: int  # 指定 'xHeight' 键对应的值是一个整数
    style: int  # 指定 'style' 键对应的值是一个整数
    typeFamily: int  # 指定 'typeFamily' 键对应的值是一个整数
    capHeight: int  # 指定 'capHeight' 键对应的值是一个整数
    symbolSet: int  # 指定 'symbolSet' 键对应的值是一个整数
    typeFace: bytes  # 指定 'typeFace' 键对应的值是一个字节序列
    characterComplement: bytes  # 指定 'characterComplement' 键对应的值是一个字节序列
    strokeWeight: int  # 指定 'strokeWeight' 键对应的值是一个整数
    widthType: int  # 指定 'widthType' 键对应的值是一个整数
    serifStyle: int  # 指定 'serifStyle' 键对应的值是一个整数

# 定义 FT2Font 类，表示一个字体对象
class FT2Font:
    ascender: int  # 字体的上升高度
    bbox: tuple[int, int, int, int]  # 字体的边界框，由四个整数组成的元组
    descender: int  # 字体的下降高度
    face_flags: int  # 字体的面部标志
    family_name: str  # 字体的家族名称
    fname: str  # 字体文件名
    height: int  # 字体的高度
    max_advance_height: int  # 字体的最大高度
    max_advance_width: int  # 字体的最大宽度
    num_charmaps: int  # 字体包含的字符映射数
    num_faces: int  # 字体包含的面数
    num_fixed_sizes: int  # 字体包含的固定尺寸数
    num_glyphs: int  # 字体包含的字形数
    postscript_name: str  # 字体的PostScript名称
    scalable: bool  # 字体是否可伸缩
    style_flags: int  # 字体的样式标志
    style_name: str  # 字体的样式名称
    underline_position: int  # 字体的下划线位置
    underline_thickness: int  # 字体的下划线厚度
    units_per_EM: int  # 字体的每EM单位

    def __init__(
        self,
        filename: str | BinaryIO,
        hinting_factor: int = ...,  # 指定字体的提示因子，默认为省略值
        *,
        _fallback_list: list[FT2Font] | None = ...,  # 字体的备用列表，默认为省略值
        _kerning_factor: int = ...  # 字体的调整因子，默认为省略值
    ) -> None: ...
    def _get_fontmap(self, string: str) -> dict[str, FT2Font]: ...  # 获取字体映射表
    def clear(self) -> None: ...  # 清除字体数据
    def draw_glyph_to_bitmap(
        self, image: FT2Image, x: float, y: float, glyph: Glyph, antialiased: bool = ...
    ) -> None: ...  # 将字形绘制到位图中
    def draw_glyphs_to_bitmap(self, antialiased: bool = ...) -> None: ...  # 将所有字形绘制到位图中
    def get_bitmap_offset(self) -> tuple[int, int]: ...  # 获取位图偏移量
    def get_char_index(self, codepoint: int) -> int: ...  # 获取字符的索引
    def get_charmap(self) -> dict[int, int]: ...  # 获取字符映射
    def get_descent(self) -> int: ...  # 获取字体的下行高度
    def get_glyph_name(self, index: int) -> str: ...  # 获取字形名称
    def get_image(self) -> NDArray[np.uint8]: ...  # 获取字体图像
    def get_kerning(self, left: int, right: int, mode: int) -> int: ...  # 获取字距调整
    def get_name_index(self, name: str) -> int: ...  # 获取名称的索引
    def get_num_glyphs(self) -> int: ...  # 获取字形数目
    def get_path(self) -> tuple[NDArray[np.float64], NDArray[np.int8]]: ...  # 获取字体路径
    def get_ps_font_info(
        self,
    ) -> tuple[str, str, str, str, str, int, int, int, int]: ...  # 获取PostScript字体信息
    def get_sfnt(self) -> dict[tuple[int, int, int, int], bytes]: ...  # 获取字体的SFNT数据
    @overload
    def get_sfnt_table(self, name: Literal["head"]) -> _SfntHeadDict | None: ...  # 获取特定SFNT表格的数据
    @overload
    def get_sfnt_table(self, name: Literal["maxp"]) -> _SfntMaxpDict | None: ...  # 获取特定SFNT表格的数据
    @overload
    def get_sfnt_table(self, name: Literal["OS/2"]) -> _SfntOs2Dict | None: ...  # 获取特定SFNT表格的数据
    @overload
    def get_sfnt_table(self, name: Literal["hhea"]) -> _SfntHheaDict | None: ...  # 获取特定SFNT表格的数据
    @overload
    def get_sfnt_table(self, name: Literal["vhea"]) -> _SfntVheaDict | None: ...  # 获取特定SFNT表格的数据
    @overload
    def get_sfnt_table(self, name: Literal["post"]) -> _SfntPostDict | None: ...  # 获取特定SFNT表格的数据
    @overload
    def get_sfnt_table(self, name: Literal["pclt"]) -> _SfntPcltDict | None: ...  # 获取特定SFNT表格的数据
    def get_width_height(self) -> tuple[int, int]: ...  # 获取宽度和高度
    # 获取 x, y 坐标值数组，返回一个包含浮点数的 NumPy 数组
    def get_xys(self, antialiased: bool = ...) -> NDArray[np.float64]:
        ...
    
    # 根据字符编码加载字符并返回 Glyph 对象
    def load_char(self, charcode: int, flags: int = ...) -> Glyph:
        ...
    
    # 根据字形索引加载字形并返回 Glyph 对象
    def load_glyph(self, glyphindex: int, flags: int = ...) -> Glyph:
        ...
    
    # 选择指定的字符映射表
    def select_charmap(self, i: int) -> None:
        ...
    
    # 设置当前使用的字符映射表
    def set_charmap(self, i: int) -> None:
        ...
    
    # 设置字体的大小（点数）和 DPI（每英寸点数）
    def set_size(self, ptsize: float, dpi: float) -> None:
        ...
    
    # 设置文本字符串，可选角度和标志，返回一个包含浮点数的 NumPy 数组
    def set_text(
        self, string: str, angle: float = ..., flags: int = ...
    ) -> NDArray[np.float64]:
        ...
# 定义一个名为 FT2Image 的类，用于操作图像和绘制矩形。
# TODO: 当更新到 mypy>=1.4 时，从 Buffer 类继承。
class FT2Image:
    # 初始化方法，接受浮点型的宽度和高度作为参数
    def __init__(self, width: float, height: float) -> None:
        ...

    # 绘制一个矩形，接受四个浮点型参数：左上角和右下角的坐标
    def draw_rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        ...

    # 绘制一个填充的矩形，接受四个浮点型参数：左上角和右下角的坐标
    def draw_rect_filled(self, x0: float, y0: float, x1: float, y1: float) -> None:
        ...

# 定义一个名为 Glyph 的类，用于表示字形的各种属性。
class Glyph:
    # 定义一些整数型的属性，表示字形的宽度、高度、水平和垂直的轴承、水平和垂直的前进
    width: int
    height: int
    horiBearingX: int
    horiBearingY: int
    horiAdvance: int
    linearHoriAdvance: int
    vertBearingX: int
    vertBearingY: int
    vertAdvance: int

    # 定义一个属性 bbox，返回一个包含四个整数的元组，表示字形的边界框
    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ...
```