# `D:\src\scipysrc\matplotlib\galleries\examples\misc\ftface_props.py`

```py
"""
===============
Font properties
===============

This example lists the attributes of an `.FT2Font` object, which describe
global font properties.  For individual character metrics, use the `.Glyph`
object, as returned by `.load_char`.
"""

import os  # 导入操作系统相关的模块

import matplotlib  # 导入 matplotlib 库
import matplotlib.ft2font as ft  # 导入 matplotlib 中的 ft2font 模块，命名为 ft

font = ft.FT2Font(
    # 使用 Matplotlib 提供的字体文件路径来创建一个 FT2Font 对象
    os.path.join(matplotlib.get_data_path(),
                 'fonts/ttf/DejaVuSans-Oblique.ttf'))

# 打印字体对象的属性
print('Num faces:  ', font.num_faces)        # 文件中的字体 face 数量
print('Num glyphs: ', font.num_glyphs)       # face 中的字形数量
print('Family name:', font.family_name)      # face 的家族名称
print('Style name: ', font.style_name)       # face 的风格名称
print('PS name:    ', font.postscript_name)  # postscript 名称
print('Num fixed:  ', font.num_fixed_sizes)  # 嵌入位图的数量

# 如果字体对象是可缩放的，则打印以下属性
if font.scalable:
    print('Bbox:               ', font.bbox)  # face 的全局边界框 (xmin, ymin, xmax, ymax)
    print('EM:                 ', font.units_per_EM)  # EM 表示的字体单元数量
    print('Ascender:           ', font.ascender)  # 26.6 单位下的上升部分
    print('Descender:          ', font.descender)  # 26.6 单位下的下降部分
    print('Height:             ', font.height)  # 26.6 单位下的高度
    print('Max adv width:      ', font.max_advance_width)  # 最大水平光标推进宽度
    print('Max adv height:     ', font.max_advance_height)  # 最大垂直光标推进高度
    print('Underline pos:      ', font.underline_position)  # 下划线的垂直位置
    print('Underline thickness:', font.underline_thickness)  # 下划线的垂直厚度

# 遍历给定的样式列表并打印每个样式的状态
for style in ('Italic',
              'Bold',
              'Scalable',
              'Fixed sizes',
              'Fixed width',
              'SFNT',
              'Horizontal',
              'Vertical',
              'Kerning',
              'Fast glyphs',
              'Multiple masters',
              'Glyph names',
              'External stream'):
    # 根据样式名称获取对应的标志位
    bitpos = getattr(ft, style.replace(' ', '_').upper()) - 1
    # 打印样式名称及其对应标志位的布尔值状态
    print(f"{style+':':17}", bool(font.style_flags & (1 << bitpos)))
```