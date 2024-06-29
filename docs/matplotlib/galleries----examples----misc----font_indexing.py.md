# `D:\src\scipysrc\matplotlib\galleries\examples\misc\font_indexing.py`

```
"""
=============
Font indexing
=============

This example shows how the font tables relate to one another.
"""

# 导入所需的模块
import os

# 导入 matplotlib，并从中导入相关的模块和类
import matplotlib
from matplotlib.ft2font import (KERNING_DEFAULT, KERNING_UNFITTED,
                                KERNING_UNSCALED, FT2Font)

# 使用指定的 TrueType 字体文件创建 FT2Font 对象
font = FT2Font(
    os.path.join(matplotlib.get_data_path(), 'fonts/ttf/DejaVuSans.ttf'))

# 设置字符映射表，这里将其设置为第一个映射表（通常是 Unicode）
font.set_charmap(0)

# 获取字体的字符映射表中的所有项
codes = font.get_charmap().items()

# 创建字符名到字符码和字形索引的映射字典
coded = {}
glyphd = {}

# 遍历字符映射表中的每一项
for ccode, glyphind in codes:
    # 获取字形索引对应的字形名称
    name = font.get_glyph_name(glyphind)
    # 将字符名称映射到字符码的字典中
    coded[name] = ccode
    # 将字符名称映射到字形索引的字典中
    glyphd[name] = glyphind
    # 打印字形索引、字符码、十六进制字符码、字形名称（注释掉的打印语句）
    # print(glyphind, ccode, hex(int(ccode)), name)

# 获取字符 'A' 对应的字符码
code = coded['A']
# 加载字符 'A' 的字形数据
glyph = font.load_char(code)

# 打印字符 'A' 的包围框信息
print(glyph.bbox)

# 打印字符 'A' 和 'V' 的字形索引及其对应的字符码
print(glyphd['A'], glyphd['V'], coded['A'], coded['V'])

# 打印字符 'A' 和 'V' 之间使用默认字距的信息
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_DEFAULT))

# 打印字符 'A' 和 'V' 之间使用适合但未缩放字距的信息
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_UNFITTED))

# 打印字符 'A' 和 'V' 之间未缩放的字距信息
print('AV', font.get_kerning(glyphd['A'], glyphd['V'], KERNING_UNSCALED))

# 打印字符 'A' 和 'T' 之间未缩放的字距信息
print('AT', font.get_kerning(glyphd['A'], glyphd['T'], KERNING_UNSCALED))
```