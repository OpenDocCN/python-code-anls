# `D:\src\scipysrc\matplotlib\lib\matplotlib\_text_helpers.py`

```py
"""
Low-level text helper utilities.
"""

# 导入必要的模块和类
from __future__ import annotations  # 允许使用类型注解中的自引用

import dataclasses  # 导入数据类支持模块

from . import _api  # 导入本地的_api模块
from .ft2font import KERNING_DEFAULT, LOAD_NO_HINTING, FT2Font  # 从本地模块导入需要的常量和类


@dataclasses.dataclass(frozen=True)
class LayoutItem:
    ft_object: FT2Font  # 字体对象
    char: str  # 字符
    glyph_idx: int  # 字形索引
    x: float  # x轴位置
    prev_kern: float  # 前一个字符的字符间距


def warn_on_missing_glyph(codepoint, fontnames):
    # 发出警告，指出缺少的字形信息
    _api.warn_external(
        f"Glyph {codepoint} "
        f"({chr(codepoint).encode('ascii', 'namereplace').decode('ascii')}) "
        f"missing from font(s) {fontnames}.")

    # 根据Unicode代码点检查字符所属的语言区块并发出警告
    block = ("Hebrew" if 0x0590 <= codepoint <= 0x05ff else
             "Arabic" if 0x0600 <= codepoint <= 0x06ff else
             "Devanagari" if 0x0900 <= codepoint <= 0x097f else
             "Bengali" if 0x0980 <= codepoint <= 0x09ff else
             "Gurmukhi" if 0x0a00 <= codepoint <= 0x0a7f else
             "Gujarati" if 0x0a80 <= codepoint <= 0x0aff else
             "Oriya" if 0x0b00 <= codepoint <= 0x0b7f else
             "Tamil" if 0x0b80 <= codepoint <= 0x0bff else
             "Telugu" if 0x0c00 <= codepoint <= 0x0c7f else
             "Kannada" if 0x0c80 <= codepoint <= 0x0cff else
             "Malayalam" if 0x0d00 <= codepoint <= 0x0d7f else
             "Sinhala" if 0x0d80 <= codepoint <= 0x0dff else
             None)
    if block:
        _api.warn_external(
            f"Matplotlib currently does not support {block} natively.")


def layout(string, font, *, kern_mode=KERNING_DEFAULT):
    """
    使用指定的字体渲染字符串。

    对于字符串中的每个字符，生成一个LayoutItem实例。当生成此类实例时，将字体的字形设置为相应的字符。

    Parameters
    ----------
    string : str
        要渲染的字符串。
    font : FT2Font
        字体对象。
    kern_mode : int
        FreeType的字符间距模式。

    Yields
    ------
    LayoutItem
        包含字符布局信息的实例。
    """
    x = 0  # 初始化x轴位置
    prev_glyph_idx = None  # 初始化前一个字形索引为None
    char_to_font = font._get_fontmap(string)  # 获取字符到字体映射
    base_font = font  # 设置基础字体为输入字体
    for char in string:
        font = char_to_font.get(char, base_font)  # 获取字符对应的字体，如果不存在则使用基础字体
        glyph_idx = font.get_char_index(ord(char))  # 获取字符的字形索引
        kern = (
            base_font.get_kerning(prev_glyph_idx, glyph_idx, kern_mode) / 64  # 计算字符间距
            if prev_glyph_idx is not None else 0.
        )
        x += kern  # 更新x轴位置
        glyph = font.load_glyph(glyph_idx, flags=LOAD_NO_HINTING)  # 加载字形
        yield LayoutItem(font, char, glyph_idx, x, kern)  # 生成LayoutItem实例
        x += glyph.linearHoriAdvance / 65536  # 更新x轴位置，考虑字形水平进阶
        prev_glyph_idx = glyph_idx  # 更新前一个字形索引
```