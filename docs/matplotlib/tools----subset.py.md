# `D:\src\scipysrc\matplotlib\tools\subset.py`

```
#!/usr/bin/env python
#
# Copyright 2010-2012, Google Inc.
# Author: Mikhail Kashkin (mkashkin@gmail.com)
# Author: Raph Levien (<firstname.lastname>@gmail.com)
# Author: Dave Crossland (dave@understandinglimited.com)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Version 1.01 Released 2012-03-27
#
# A script for subsetting a font, using FontForge. See README for details.

# TODO 2013-04-08 ensure the menu files are as compact as possible by default,
# similar to subset.pl
# TODO 2013-05-22 in Arimo, the latin subset doesn't include ; but the greek
# does. why on earth is this happening?

import getopt
import os
import struct
import subprocess
import sys

import fontforge


def log_namelist(name, unicode):
    # 如果文件名和 unicode 均不为空，则将 unicode 转换为十六进制格式并记录到文件中
    if name and isinstance(unicode, int):
        print(f"0x{unicode:04X}", fontforge.nameFromUnicode(unicode),
              file=name)


def select_with_refs(font, unicode, newfont, pe=None, name=None):
    # 使用 unicode 选择新字体中的字符集
    newfont.selection.select(('more', 'unicode'), unicode)
    # 记录 unicode 的名称和编码到指定的文件中
    log_namelist(name, unicode)
    if pe:
        # 如果存在脚本文件，则输出选择的 unicode 到脚本文件中
        print(f"SelectMore({unicode})", file=pe)
    try:
        # 遍历字符的引用并在新字体中选择它们
        for ref in font[unicode].references:
            newfont.selection.select(('more',), ref[0])
            # 记录引用字符的名称和编码到指定的文件中
            log_namelist(name, ref[0])
            if pe:
                # 如果存在脚本文件，则输出选择的引用到脚本文件中
                print(f'SelectMore("{ref[0]}")', file=pe)
    except Exception:
        # 若引用解析失败，则输出错误信息
        print(f'Resolving references on u+{unicode:04x} failed')


def subset_font_raw(font_in, font_out, unicodes, opts):
    if '--namelist' in opts:
        # 如果命令行参数包含 --namelist，则设置名单文件名并打开该文件以写入
        name_fn = f'{font_out}.name'
        name = open(name_fn, 'w')
    else:
        name = None

    if '--script' in opts:
        # 如果命令行参数包含 --script，则设置脚本文件的路径
        pe_fn = "/tmp/script.pe"
        pe = open(pe_fn, 'w')
    else:
        pe = None

    # 打开输入字体文件
    font = fontforge.open(font_in)

    if pe:
        # 如果存在脚本文件，则记录打开字体文件的操作到脚本文件中
        print(f'Open("{font_in}")', file=pe)
        extract_vert_to_script(font_in, pe)

    # 对于每个 unicode，选择并处理字体中的字符集
    for i in unicodes:
        select_with_refs(font, i, font, pe, name)

    addl_glyphs = []

    # 根据命令行参数选择需要额外处理的特殊字符集
    if '--nmr' in opts:
        addl_glyphs.append('nonmarkingreturn')
    if '--null' in opts:
        addl_glyphs.append('.null')
    if '--nd' in opts:
        addl_glyphs.append('.notdef')
    # 遍历附加字形列表中的每一个字形
    for glyph in addl_glyphs:
        # 选择指定的字形
        font.selection.select(('more',), glyph)
        # 如果指定了输出文件名，则打印字形的 Unicode 编码和字形名称到文件中
        if name:
            print(f"0x{fontforge.unicodeFromName(glyph):0.4X}", glyph,
                  file=name)
        # 如果指定了字形编辑器，则打印选中字形的命令到文件中
        if pe:
            print(f'SelectMore("{glyph}")', file=pe)

    # 定义一个空的标志元组
    flags = ()

    # 如果命令行参数中包含 '--opentype-features'
    if '--opentype-features' in opts:
        # 添加 'opentype' 到标志元组中
        flags += ('opentype',)

    # 如果命令行参数中包含 '--simplify'
    if '--simplify' in opts:
        # 简化字体轮廓
        font.simplify()
        # 四舍五入字体轮廓点
        font.round()
        # 添加 'omit-instructions' 到标志元组中
        flags += ('omit-instructions',)

    # 如果命令行参数中包含 '--strip_names'
    if '--strip_names' in opts:
        # 清空字体的名称表
        font.sfnt_names = ()

    # 如果命令行参数中包含 '--new'
    if '--new' in opts:
        # 复制当前字体对象
        font.copy()
        # 创建一个新的字体对象
        new = fontforge.font()
        # 将新字体对象的编码、字体全局指标与原字体对象相同
        new.encoding = font.encoding
        new.em = font.em
        new.layers['Fore'].is_quadratic = font.layers['Fore'].is_quadratic
        # 对于给定的 Unicode 编码列表中的每一个编码
        for i in unicodes:
            # 使用引用选择函数将原字体中的字符选择到新字体中
            select_with_refs(font, i, new, pe, name)
        # 粘贴选择的字符到新字体中
        new.paste()
        # 进行一个修补操作，本应该在上面的步骤中处理
        font.selection.select('space')
        font.copy()
        new.selection.select('space')
        new.paste()
        # 将原字体的名称表复制到新字体中
        new.sfnt_names = font.sfnt_names
        # 将当前字体对象指向新创建的字体对象
        font = new
    else:
        # 反选当前字体对象中的选择内容
        font.selection.invert()
        # 将反选的内容剪切
        print("SelectInvert()", file=pe)
        font.cut()
        print("Clear()", file=pe)

    # 如果命令行参数中包含 '--move-display'
    if '--move-display' in opts:
        # 输出信息，显示将显示字形移动到 Unicode 范围内
        print("Moving display glyphs into Unicode ranges...")
        # 修改字体的家族名称、全名、字体名称，加入 "Display" 后缀
        font.familyname += " Display"
        font.fullname += " Display"
        font.fontname += "Display"
        # 向字体添加英文名称条目
        font.appendSFNTName('English (US)', 'Family', font.familyname)
        font.appendSFNTName('English (US)', 16, font.familyname)
        font.appendSFNTName('English (US)', 17, 'Display')
        font.appendSFNTName('English (US)', 'Fullname', font.fullname)
        # 对于给定的 Unicode 编码列表中的每一个编码名称
        for glname in unicodes:
            # 清除当前字体的选择内容
            font.selection.none()
            # 如果编码名称是字符串类型，并以 '.display' 结尾
            if isinstance(glname, str):
                if glname.endswith('.display'):
                    # 选择以 '.display' 结尾的编码名称
                    font.selection.select(glname)
                    # 复制选中的内容
                    font.copy()
                    # 清除当前字体的选择内容
                    font.selection.none()
                    # 替换编码名称中的 '.display' 为新的编码名称
                    newgl = glname.replace('.display', '')
                    # 选择新的编码名称
                    font.selection.select(newgl)
                    # 粘贴复制的内容
                    font.paste()
                # 选择指定的编码名称
                font.selection.select(glname)
                # 剪切选中的内容
                font.cut()

    # 如果指定了输出文件名，则打印信息，写入名称列表完成
    if name:
        print("Writing NameList", end="")
        # 关闭名称文件
        name.close()

    # 如果指定了字形编辑器
    if pe:
        # 打印生成字体的命令到文件中
        print(f'Generate("{font_out}")', file=pe)
        # 关闭字形编辑器文件
        pe.close()
        # 调用外部命令行工具 fontforge，执行指定的脚本文件
        subprocess.call(["fontforge", "-script", pe_fn])
    else:
        # 生成字体文件到指定的输出文件名
        font.generate(font_out, flags=flags)
    # 关闭当前字体对象
    font.close()

    # 如果命令行参数中包含 '--roundtrip'
    if '--roundtrip' in opts:
        # 修复 FontForge 中的 bug，重新打开并生成字体文件
        font2 = fontforge.open(font_out)
        font2.generate(font_out, flags=flags)
# 定义函数 subset_font，用于从一个字体文件中提取特定字符集合并保存到另一个字体文件中
def subset_font(font_in, font_out, unicodes, opts):
    # 复制输出字体文件路径到一个新变量 font_out_raw
    font_out_raw = font_out
    # 如果输出字体文件路径不以 '.ttf' 结尾，则添加 '.ttf' 后缀
    if not font_out_raw.endswith('.ttf'):
        font_out_raw += '.ttf'
    # 调用 subset_font_raw 函数，从输入字体文件中提取指定字符集合到输出字体文件中
    subset_font_raw(font_in, font_out_raw, unicodes, opts)
    # 如果实际输出的字体文件路径与初始赋值的路径不同，进行文件重命名操作
    if font_out != font_out_raw:
        os.rename(font_out_raw, font_out)

# 定义函数 getsubset，用于根据给定的子集名称从字体文件中提取特定的字符集合
def getsubset(subset, font_in):
    # 将子集名称拆分成列表 subsets，以加号 '+' 为分隔符
    subsets = subset.split('+')
    # 定义引号字符集合列表 quotes
    quotes = [
        0x2013,  # endash
        0x2014,  # emdash
        0x2018,  # quoteleft
        0x2019,  # quoteright
        0x201A,  # quotesinglbase
        0x201C,  # quotedblleft
        0x201D,  # quotedblright
        0x201E,  # quotedblbase
        0x2022,  # bullet
        0x2039,  # guilsinglleft
        0x203A,  # guilsinglright
    ]
    
    # 定义拉丁字符集合列表 latin
    latin = [
        *range(0x20, 0x7f),  # Basic Latin (A-Z, a-z, numbers)
        *range(0xa0, 0x100),  # Western European symbols and diacritics
        0x20ac,  # Euro
        0x0152,  # OE
        0x0153,  # oe
        0x003b,  # semicolon
        0x00b7,  # periodcentered
        0x0131,  # dotlessi
        0x02c6,  # circumflex
        0x02da,  # ring
        0x02dc,  # tilde
        0x2074,  # foursuperior
        0x2215,  # division slash
        0x2044,  # fraction slash
        0xe0ff,  # PUA: Font logo
        0xeffd,  # PUA: Font version number
        0xf000,  # PUA: font ppem size indicator: run
                 # 使用 `ftview -f 1255 10 Ubuntu-Regular.ttf` 可以看到它的效果！
    ]
    
    # 初始化结果集合为 quotes
    result = quotes
    
    # 如果子集列表中包含 'menu'，则从字体文件中提取字体名称的 Unicode 编码并加入结果集合中
    if 'menu' in subsets:
        font = fontforge.open(font_in)
        result = [
            *map(ord, font.familyname),
            0x0020,
        ]
    
    # 如果子集列表中包含 'latin'，则将拉丁字符集合 latin 加入结果集合中
    if 'latin' in subsets:
        result += latin
    
    # 如果子集列表中包含 'latin-ext'，则将扩展拉丁字符集合加入结果集合中
    if 'latin-ext' in subsets:
        # 扩展拉丁字符集合包括多个范围的 Unicode 编码
        result += [
            *range(0x100, 0x370),
            *range(0x1d00, 0x1ea0),
            *range(0x1ef2, 0x1f00),
            *range(0x2070, 0x20d0),
            *range(0x2c60, 0x2c80),
            *range(0xa700, 0xa800),
        ]
    
    # 如果子集列表中包含 'vietnamese'，则将越南语字符集合加入结果集合中
    if 'vietnamese' in subsets:
        # 越南语字符集合的 Unicode 编码来自指定的来源
        result += [0x00c0, 0x00c1, 0x00c2, 0x00c3, 0x00C8, 0x00C9,
                   0x00CA, 0x00CC, 0x00CD, 0x00D2, 0x00D3, 0x00D4,
                   0x00D5, 0x00D9, 0x00DA, 0x00DD, 0x00E0, 0x00E1,
                   0x00E2, 0x00E3, 0x00E8, 0x00E9, 0x00EA, 0x00EC,
                   0x00ED, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F9,
                   0x00FA, 0x00FD, 0x0102, 0x0103, 0x0110, 0x0111,
                   0x0128, 0x0129, 0x0168, 0x0169, 0x01A0, 0x01A1,
                   0x01AF, 0x01B0, 0x20AB, *range(0x1EA0, 0x1EFA)]
    # 如果 'greek' 在子集中，将希腊字符的 Unicode 范围加入结果列表
    if 'greek' in subsets:
        # 这里可以更激进一些，排除古代字符，但是缺少数据支持
        result += [*range(0x370, 0x400)]
    
    # 如果 'greek-ext' 在子集中，将扩展希腊字符的 Unicode 范围加入结果列表
    if 'greek-ext' in subsets:
        result += [*range(0x370, 0x400), *range(0x1f00, 0x2000)]
    
    # 如果 'cyrillic' 在子集中，根据字符频率分析，将 Cyrillic 字符的 Unicode 范围加入结果列表
    if 'cyrillic' in subsets:
        result += [*range(0x400, 0x460), 0x490, 0x491, 0x4b0, 0x4b1, 0x2116]
    
    # 如果 'cyrillic-ext' 在子集中，将扩展 Cyrillic 字符的 Unicode 范围加入结果列表
    if 'cyrillic-ext' in subsets:
        result += [
            *range(0x400, 0x530),
            0x20b4,
            # 0x2116 是俄语中的 No，类似于拉丁文中的 #，由Alexei Vanyashin建议使用
            0x2116,
            *range(0x2de0, 0x2e00),
            *range(0xa640, 0xa6a0),
        ]
    
    # 如果 'dejavu-ext' 在子集中，将所有以 '.display' 结尾的字形名加入结果列表
    if 'dejavu-ext' in subsets:
        # 打开字体文件
        font = fontforge.open(font_in)
        # 遍历所有字形
        for glyph in font.glyphs():
            # 如果字形名以 '.display' 结尾，将其添加到结果列表
            if glyph.glyphname.endswith('.display'):
                result.append(glyph.glyphname)

    # 返回最终结果列表
    return result
# 从 TrueType 字体中提取垂直度量的代码
class Sfnt:
    def __init__(self, data):
        # 解析字体数据的头部信息
        _, numTables, _, _, _ = struct.unpack('>IHHHH', data[:12])
        self.tables = {}
        # 遍历字体数据中的表信息
        for i in range(numTables):
            tag, _, offset, length = struct.unpack(
                '>4sIII', data[12 + 16 * i: 28 + 16 * i])
            # 将每个表的数据存储在字典中
            self.tables[tag] = data[offset: offset + length]

    def hhea(self):
        r = {}
        d = self.tables['hhea']
        # 从'hhea'表中解析出 Ascender, Descender 和 LineGap 的值
        r['Ascender'], r['Descender'], r['LineGap'] = struct.unpack(
            '>hhh', d[4:10])
        return r

    def os2(self):
        r = {}
        d = self.tables['OS/2']
        # 从'OS/2'表中解析出 fsSelection, sTypoAscender, sTypoDescender 和 sTypoLineGap 的值
        r['fsSelection'], = struct.unpack('>H', d[62:64])
        r['sTypoAscender'], r['sTypoDescender'], r['sTypoLineGap'] = \
            struct.unpack('>hhh', d[68:74])
        r['usWinAscender'], r['usWinDescender'] = struct.unpack(
            '>HH', d[74:78])
        return r


def set_os2(pe, name, val):
    # 打印设置 OS/2 表数值的脚本代码
    print(f'SetOS2Value("{name}", {val:d})', file=pe)


def set_os2_vert(pe, name, val):
    # 设置垂直度量值
    set_os2(pe, name + 'IsOffset', 0)
    set_os2(pe, name, val)


# 从字体文件直接提取垂直度量数据，并生成设置字体值的脚本代码。
# 这是一个对于以下问题的（相当丑陋的）解决方法：
# https://sourceforge.net/p/fontforge/mailman/fontforge-users/thread/20100906085718.GB1907@khaled-laptop/
def extract_vert_to_script(font_in, pe):
    # 读取字体文件的二进制数据
    with open(font_in, 'rb') as in_file:
        data = in_file.read()
    # 创建字体对象
    sfnt = Sfnt(data)
    # 获取字体的'hhea'和'OS/2'表数据
    hhea = sfnt.hhea()
    os2 = sfnt.os2()
    # 设置字体垂直度量值到脚本
    set_os2_vert(pe, "WinAscent", os2['usWinAscender'])
    set_os2_vert(pe, "WinDescent", os2['usWinDescender'])
    set_os2_vert(pe, "TypoAscent", os2['sTypoAscender'])
    set_os2_vert(pe, "TypoDescent", os2['sTypoDescender'])
    set_os2_vert(pe, "HHeadAscent", hhea['Ascender'])
    set_os2_vert(pe, "HHeadDescent", hhea['Descender'])


def main(argv):
    # 解析命令行参数
    optlist, args = getopt.gnu_getopt(argv, '', [
        'string=', 'strip_names', 'opentype-features', 'simplify', 'new',
        'script', 'nmr', 'roundtrip', 'subset=', 'namelist', 'null', 'nd',
        'move-display'])

    font_in, font_out = args
    opts = dict(optlist)
    # 根据命令行参数设置子集，或者默认使用 Latin 子集
    if '--string' in opts:
        subset = map(ord, opts['--string'])
    else:
        subset = getsubset(opts.get('--subset', 'latin'), font_in)
    # 提取并生成子集字体
    subset_font(font_in, font_out, subset, opts)


if __name__ == '__main__':
    main(sys.argv[1:])
```