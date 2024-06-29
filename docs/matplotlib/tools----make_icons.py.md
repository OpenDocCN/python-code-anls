# `D:\src\scipysrc\matplotlib\tools\make_icons.py`

```
#!/usr/bin/env python
"""
Generates the Matplotlib icon, and the toolbar icon images from the FontAwesome
font.

Generates SVG, PDF in one size (since they are vectors), and PNG in 24x24 and
48x48.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter  # 导入参数解析相关库
from io import BytesIO  # 导入字节流处理库
from pathlib import Path  # 导入路径处理库
import tarfile  # 导入tar文件处理库
import urllib.request  # 导入网络请求库

import matplotlib as mpl  # 导入matplotlib库并重命名为mpl
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块并重命名为plt
import numpy as np  # 导入numpy库并重命名为np


plt.rcdefaults()  # 重置matplotlib默认配置
plt.rcParams['svg.fonttype'] = 'path'  # 设置SVG输出字体类型为path
plt.rcParams['pdf.fonttype'] = 3  # 设置PDF输出字体类型为Type 3
plt.rcParams['pdf.compression'] = 9  # 设置PDF输出的压缩级别为9


def get_fontawesome():
    cached_path = Path(mpl.get_cachedir(), "FontAwesome.otf")  # 获取缓存目录并拼接FontAwesome.otf文件路径
    if not cached_path.exists():
        with urllib.request.urlopen(
                "https://github.com/FortAwesome/Font-Awesome"
                "/archive/v4.7.0.tar.gz") as req, \
             tarfile.open(fileobj=BytesIO(req.read()), mode="r:gz") as tf:
            cached_path.write_bytes(tf.extractfile(tf.getmember(
                "Font-Awesome-4.7.0/fonts/FontAwesome.otf")).read())  # 如果文件不存在，从网络下载并缓存FontAwesome.otf
    return cached_path  # 返回FontAwesome.otf的路径


def save_icon(fig, dest_dir, name, add_black_fg_color):
    if add_black_fg_color:
        # 对单色SVG图标添加显式的黑色前景色，以便后端可以添加暗色主题支持
        svg_bytes_io = BytesIO()
        fig.savefig(svg_bytes_io, format='svg')  # 将图形保存为SVG格式到字节流
        svg = svg_bytes_io.getvalue()
        before, sep, after = svg.rpartition(b'\nz\n"')
        svg = before + sep + b' style="fill:black;"' + after  # 在SVG中添加黑色样式
        (dest_dir / (name + '.svg')).write_bytes(svg)  # 将修改后的SVG写入目标目录
    else:
        fig.savefig(dest_dir / (name + '.svg'))  # 将图形保存为SVG格式到目标目录
    fig.savefig(dest_dir / (name + '.pdf'))  # 将图形保存为PDF格式到目标目录
    for dpi, suffix in [(24, ''), (48, '_large')]:
        fig.savefig(dest_dir / (name + suffix + '.png'), dpi=dpi)  # 将图形保存为指定dpi的PNG格式到目标目录


def make_icon(font_path, ccode):
    fig = plt.figure(figsize=(1, 1))  # 创建1x1尺寸的图形
    fig.patch.set_alpha(0.0)  # 设置图形背景透明度为0
    fig.text(0.5, 0.48, chr(ccode), ha='center', va='center',
             font=font_path, fontsize=68)  # 在图形中央绘制指定字体路径和Unicode码点的文本
    return fig  # 返回生成的图形对象


def make_matplotlib_icon():
    fig = plt.figure(figsize=(1, 1))  # 创建1x1尺寸的图形
    fig.patch.set_alpha(0.0)  # 设置图形背景透明度为0
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95], projection='polar')  # 在图形上添加极坐标轴
    ax.set_axisbelow(True)  # 将网格放置在柱状图之后

    N = 7
    arc = 2 * np.pi
    theta = np.arange(0, arc, arc / N)
    radii = 10 * np.array([0.2, 0.6, 0.8, 0.7, 0.4, 0.5, 0.8])
    width = np.pi / 4 * np.array([0.4, 0.4, 0.6, 0.8, 0.2, 0.5, 0.3])
    bars = ax.bar(theta, radii, width=width, bottom=0.0, linewidth=1,
                  edgecolor='k')  # 在极坐标轴上绘制柱状图

    for r, bar in zip(radii, bars):
        bar.set_facecolor(mpl.cm.jet(r / 10))  # 设置柱状图颜色为jet colormap

    ax.tick_params(labelleft=False, labelright=False,
                   labelbottom=False, labeltop=False)  # 设置坐标轴标签不可见
    ax.grid(lw=0.0)  # 设置坐标轴网格线宽度为0

    ax.set_yticks(np.arange(1, 9, 2))  # 设置极轴刻度
    ax.set_rmax(9)  # 设置极轴最大半径

    return fig  # 返回生成的图形对象


icon_defs = [
    ('home', 0xf015),
    ('back', 0xf060),
    ('forward', 0xf061),
    ('zoom_to_rect', 0xf002),
    ('move', 0xf047),
    ('filesave', 0xf0c7),
    ('subplots', 0xf1de),
    ('qt4_editor_options', 0xf201),
    ('help', 0xf128),


# 创建一个包含两个元组的列表，每个元组包含一个字符串和一个十六进制数
# 导入必要的模块和函数
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

# 定义函数 make_icons，用于生成图标
def make_icons():
    # 创建参数解析器
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    # 添加命令行参数选项
    parser.add_argument(
        "-d", "--dest-dir",
        type=Path,
        default=Path(__file__).parent / "../lib/matplotlib/mpl-data/images",
        help="Directory where to store the images.")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 获取 Font Awesome 字体文件的路径
    font_path = get_fontawesome()
    
    # 遍历图标定义列表，生成每个图标并保存
    for name, ccode in icon_defs:
        # 创建图标
        fig = make_icon(font_path, ccode)
        
        # 保存图标到指定目录
        save_icon(fig, args.dest_dir, name, True)
    
    # 创建 Matplotlib 图标
    fig = make_matplotlib_icon()
    
    # 保存 Matplotlib 图标到指定目录
    save_icon(fig, args.dest_dir, 'matplotlib', False)

# 程序入口：如果作为主程序运行，则调用 make_icons 函数生成图标
if __name__ == "__main__":
    make_icons()
```