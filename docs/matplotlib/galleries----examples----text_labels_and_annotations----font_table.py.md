# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\font_table.py`

```py
"""
==========
Font table
==========

Matplotlib's font support is provided by the FreeType library.

Here, we use `~.Axes.table` to draw a table that shows the glyphs by Unicode
codepoint. For brevity, the table only contains the first 256 glyphs.

The example is a full working script. You can download it and use it to
investigate a font by running ::

    python font_table.py /path/to/font/file
"""

import os
from pathlib import Path
import unicodedata

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
from matplotlib.ft2font import FT2Font


def print_glyphs(path):
    """
    Print the all glyphs in the given font file to stdout.

    Parameters
    ----------
    path : str or None
        The path to the font file.  If None, use Matplotlib's default font.
    """
    if path is None:
        path = fm.findfont(fm.FontProperties())  # The default font.

    # 使用给定路径初始化一个 FT2Font 对象
    font = FT2Font(path)

    # 获取字体的字符映射表
    charmap = font.get_charmap()

    # 计算映射表中最大索引值的位数，用于格式化输出
    max_indices_len = len(str(max(charmap.values())))

    # 打印字体包含的所有字符
    print("The font face contains the following glyphs:")
    for char_code, glyph_index in charmap.items():
        char = chr(char_code)
        # 获取字符的名称，如果名称无法解析，则用字符代码和字形名称代替
        name = unicodedata.name(
                char,
                f"{char_code:#x} ({font.get_glyph_name(glyph_index)})")
        print(f"{glyph_index:>{max_indices_len}} {char} {name}")


def draw_font_table(path):
    """
    Draw a font table of the first 255 chars of the given font.

    Parameters
    ----------
    path : str or None
        The path to the font file.  If None, use Matplotlib's default font.
    """
    if path is None:
        path = fm.findfont(fm.FontProperties())  # The default font.

    # 使用给定路径初始化一个 FT2Font 对象
    font = FT2Font(path)

    # 获取字符映射表
    codes = font.get_charmap().items()

    # 创建标签列表，用于显示表格的行和列
    labelc = [f"{i:X}" for i in range(16)]
    labelr = [f"{i:02X}" for i in range(0, 16*16, 16)]
    chars = [["" for c in range(16)] for r in range(16)]

    # 填充字符二维列表，以便后续在表格中显示字符
    for char_code, glyph_index in codes:
        if char_code >= 256:
            continue
        row, col = divmod(char_code, 16)
        chars[row][col] = chr(char_code)

    # 创建 Matplotlib 图形和轴对象
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(os.path.basename(path))
    ax.set_axis_off()
    # 创建一个表格对象 `table`，并设置其内容和样式
    table = ax.table(
        cellText=chars,  # 将 `chars` 中的文本作为单元格内容
        rowLabels=labelr,  # 使用 `labelr` 作为行标签
        colLabels=labelc,  # 使用 `labelc` 作为列标签
        rowColours=["palegreen"] * 16,  # 设置所有行的背景颜色为淡绿色
        colColours=["palegreen"] * 16,  # 设置所有列的背景颜色为淡绿色
        cellColours=[[".95" for c in range(16)] for r in range(16)],  # 设置所有单元格的背景颜色为浅灰色
        cellLoc='center',  # 将单元格中的文本居中显示
        loc='upper left',  # 将表格放置在图像的左上角位置
    )
    
    # 遍历表格中的每个单元格，获取单元格对象 `cell`
    for key, cell in table.get_celld().items():
        row, col = key
        # 检查单元格的行索引大于0且列索引大于-1（因为表格的特殊索引...）
        if row > 0 and col > -1:
            # 设置单元格文本的字体属性为从指定路径 `path` 加载的字体
            cell.set_text_props(font=Path(path))
    
    # 调整图像布局以确保表格和其他元素合适地显示
    fig.tight_layout()
    
    # 显示图像
    plt.show()
if __name__ == "__main__":
    # 检查当前脚本是否作为主程序运行

    from argparse import ArgumentParser
    # 导入 ArgumentParser 类，用于处理命令行参数解析

    parser = ArgumentParser(description="Display a font table.")
    # 创建 ArgumentParser 对象，并设置描述信息

    parser.add_argument("path", nargs="?", help="Path to the font file.")
    # 添加位置参数 "path"，表示字体文件的路径，可选参数，用法示例中有说明

    parser.add_argument("--print-all", action="store_true",
                        help="Additionally, print all chars to stdout.")
    # 添加可选参数 "--print-all"，若设置则打印所有字符到标准输出

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    if args.print_all:
        # 如果指定了 --print-all 参数
        print_glyphs(args.path)
        # 调用 print_glyphs 函数，打印字体文件中的所有字符

    draw_font_table(args.path)
    # 调用 draw_font_table 函数，显示字体表格
```