# `D:\src\scipysrc\matplotlib\lib\matplotlib\font_manager.py`

```
# 导入必要的模块和库
from __future__ import annotations  # 导入 future 模块以支持类型注释

from base64 import b64encode  # 导入 base64 编码相关功能
from collections import namedtuple  # 导入 namedtuple 类型
import copy  # 导入复制相关功能
import dataclasses  # 导入数据类支持
from functools import lru_cache  # 导入 lru_cache 装饰器
from io import BytesIO  # 导入 BytesIO 类以处理二进制数据
import json  # 导入 JSON 格式支持
import logging  # 导入日志功能
from numbers import Number  # 导入数字类型支持
import os  # 导入操作系统功能
from pathlib import Path  # 导入路径处理支持
import plistlib  # 导入 plist 文件格式支持
import re  # 导入正则表达式支持
import subprocess  # 导入子进程管理支持
import sys  # 导入系统相关功能
import threading  # 导入多线程支持

import matplotlib as mpl  # 导入 matplotlib 库
from matplotlib import _api, _afm, cbook, ft2font  # 导入 matplotlib 内部模块
from matplotlib._fontconfig_pattern import (  # 导入字体配置相关模块
    parse_fontconfig_pattern, generate_fontconfig_pattern)
from matplotlib.rcsetup import _validators  # 导入 matplotlib 配置验证器

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 字体大小与缩放比例映射关系
font_scalings = {
    'xx-small': 0.579,
    'x-small':  0.694,
    'small':    0.833,
    'medium':   1.0,
    'large':    1.200,
    'x-large':  1.440,
    'xx-large': 1.728,
    'larger':   1.2,
    'smaller':  0.833,
    None:       1.0,  # 默认缩放比例为 1.0
}

# 字体拉伸样式与数值映射关系
stretch_dict = {
    'ultra-condensed': 100,
    'extra-condensed': 200,
    'condensed':       300,
    'semi-condensed':  400,
    'normal':          500,
    'semi-expanded':   600,
    'semi-extended':   600,
    'expanded':        700,
    'extended':        700,
    'extra-expanded':  800,
    'extra-extended':  800,
    'ultra-expanded':  900,
    'ultra-extended':  900,
}

# 字体粗细样式与数值映射关系
weight_dict = {
    'ultralight': 100,
    'light':      200,  # 'light' 实际上在 weight_dict 中是无效的，应该移除
    'normal':     400,
    'regular':    400,
    'book':       400,
    'medium':     500,
    'roman':      500,
    'semibold':   600,
    'demibold':   600,
    'demi':       600,
    'bold':       700,
    'heavy':      800,
    'extra bold': 800,
    'black':      900,
}

# 字体粗细样式的正则表达式匹配列表
_weight_regexes = [
    # 从 fontconfig 的 FcFreeTypeQueryFaceInternal 获取的，与 weight_dict 不同!
    ("thin", 100),
    ("extralight", 200),
    ("ultralight", 200),
    ("demilight", 350),
    ("semilight", 350),
    ("light", 300),  # 需要在 demi/semilight 之后出现!
    ("book", 380),
    ("regular", 400),
    ("normal", 400),
    ("medium", 500),
    ("demibold", 600),
    ("demi", 600),
    ("semibold", 600),
    ("extrabold", 800),
    ("superbold", 800),
    ("ultrabold", 800),
    ("bold", 700),  # 需要在 extra/super/ultrabold 之后！表示粗体字的标准重量
    ("ultrablack", 1000),
    ("superblack", 1000),
    ("extrablack", 1000),
    (r"\bultra", 1000),  # 包含 `\ultra` 的特定字体，重量为 1000
    ("black", 900),  # 需要在 ultra/super/extrablack 之后！表示最黑的标准重量
    ("heavy", 900),  # 表示较重的标准重量
# 定义一个字典，包含常见字体族的别名
font_family_aliases = {
    'serif',
    'sans-serif',
    'sans serif',
    'cursive',
    'fantasy',
    'monospace',
    'sans',
}

# 命名元组，用于代理异常信息
_ExceptionProxy = namedtuple('_ExceptionProxy', ['klass', 'message'])

# OS 字体路径

# 尝试获取当前用户的主目录路径
try:
    _HOME = Path.home()
# 如果无法获取主目录路径，将_HOME设置为一个没有子目录的虚拟路径
except Exception:  # Exceptions thrown by home() are not specified...
    _HOME = Path(os.devnull)  # Just an arbitrary path with no children.

# 定义 Windows 相关的字体目录列表
MSFolders = \
    r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
MSFontDirectories = [
    r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts',
    r'SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts']

# 定义 Windows 用户字体目录列表
MSUserFontDirectories = [
    str(_HOME / 'AppData/Local/Microsoft/Windows/Fonts'),
    str(_HOME / 'AppData/Roaming/Microsoft/Windows/Fonts'),
]

# 定义 X11（Unix/Linux）字体目录列表
X11FontDirectories = [
    # an old standard installation point
    "/usr/X11R6/lib/X11/fonts/TTF/",
    "/usr/X11/lib/X11/fonts",
    # here is the new standard location for fonts
    "/usr/share/fonts/",
    # documented as a good place to install new fonts
    "/usr/local/share/fonts/",
    # common application, not really useful
    "/usr/lib/openoffice/share/fonts/truetype/",
    # user fonts
    str((Path(os.environ.get('XDG_DATA_HOME') or _HOME / ".local/share"))
        / "fonts"),
    str(_HOME / ".fonts"),
]

# 定义 macOS 字体目录列表
OSXFontDirectories = [
    "/Library/Fonts/",
    "/Network/Library/Fonts/",
    "/System/Library/Fonts/",
    # fonts installed via MacPorts
    "/opt/local/share/fonts",
    # user fonts
    str(_HOME / "Library/Fonts"),
]

# 定义函数：根据字体文件扩展名获取其同义扩展名列表
def get_fontext_synonyms(fontext):
    """
    Return a list of file extensions that are synonyms for
    the given file extension *fileext*.
    """
    return {
        'afm': ['afm'],
        'otf': ['otf', 'ttc', 'ttf'],
        'ttc': ['otf', 'ttc', 'ttf'],
        'ttf': ['otf', 'ttc', 'ttf'],
    }[fontext]


# 定义函数：列出目录下所有符合指定扩展名的字体文件路径
def list_fonts(directory, extensions):
    """
    Return a list of all fonts matching any of the extensions, found
    recursively under the directory.
    """
    extensions = ["." + ext for ext in extensions]
    return [os.path.join(dirpath, filename)
            # os.walk ignores access errors, unlike Path.glob.
            for dirpath, _, filenames in os.walk(directory)
            for filename in filenames
            if Path(filename).suffix.lower() in extensions]


# 定义函数：返回 Win32 系统下的用户字体目录
def win32FontDirectory():
    r"""
    Return the user-specified font directory for Win32.  This is
    looked up from the registry key ::
    
      \\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders\Fonts
    
    If the key is not found, ``%WINDIR%\Fonts`` will be returned.
    """  # noqa: E501
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, MSFolders) as user:
            return winreg.QueryValueEx(user, 'Fonts')[0]
    except OSError:
        return os.path.join(os.environ['WINDIR'], 'Fonts')


# 定义函数：列出 Windows 系统注册表中已安装的字体路径
def _get_win32_installed_fonts():
    """List the font paths known to the Windows registry."""
    import winreg
    items = set()
    # 创建一个空的集合用于存储找到的字体文件的路径集合

    # 在注册表中搜索并解析列出的字体
    for domain, base_dirs in [
            (winreg.HKEY_LOCAL_MACHINE, [win32FontDirectory()]),  # 系统字体目录
            (winreg.HKEY_CURRENT_USER, MSUserFontDirectories),  # 用户字体目录
    ]:
        # 遍历每个基础目录
        for base_dir in base_dirs:
            # 遍历每个注册路径
            for reg_path in MSFontDirectories:
                try:
                    # 尝试打开注册表键并获取其信息
                    with winreg.OpenKey(domain, reg_path) as local:
                        # 遍历注册表键中的值
                        for j in range(winreg.QueryInfoKey(local)[1]):
                            # 值可能包含字体文件的文件名或绝对路径
                            key, value, tp = winreg.EnumValue(local, j)
                            if not isinstance(value, str):
                                continue
                            try:
                                # 如果值已经包含绝对路径，则不再改变
                                path = Path(base_dir, value).resolve()
                            except RuntimeError:
                                # 处理无效条目时不要失败
                                continue
                            # 将路径添加到集合中
                            items.add(path)
                except (OSError, MemoryError):
                    # 处理操作系统错误或内存错误时继续下一个注册表路径
                    continue
    # 返回收集到的所有字体文件的路径集合
    return items
# 使用 lru_cache 装饰器，缓存并列出由 ``fc-list`` 知道的字体路径
@lru_cache
def _get_fontconfig_fonts():
    """Cache and list the font paths known to ``fc-list``."""
    try:
        # 检查 ``fc-list`` 命令的输出中是否包含 '--format'，如果没有则提示警告
        if b'--format' not in subprocess.check_output(['fc-list', '--help']):
            _log.warning(  # fontconfig 2.7 implemented --format.
                'Matplotlib needs fontconfig>=2.7 to query system fonts.')
            return []
        # 执行 ``fc-list --format=%{file}\\n`` 命令，获取字体文件路径列表
        out = subprocess.check_output(['fc-list', '--format=%{file}\\n'])
    except (OSError, subprocess.CalledProcessError):
        return []
    # 将字体文件路径转换为 Path 对象，并返回列表
    return [Path(os.fsdecode(fname)) for fname in out.split(b'\n')]


# 使用 lru_cache 装饰器，缓存并列出由 ``system_profiler SPFontsDataType`` 知道的字体路径
@lru_cache
def _get_macos_fonts():
    """Cache and list the font paths known to ``system_profiler SPFontsDataType``."""
    # 执行 ``system_profiler -xml SPFontsDataType`` 命令，并解析输出的 plist 数据
    d, = plistlib.loads(
        subprocess.check_output(["system_profiler", "-xml", "SPFontsDataType"]))
    # 返回字体路径列表
    return [Path(entry["path"]) for entry in d["_items"]]


def findSystemFonts(fontpaths=None, fontext='ttf'):
    """
    Search for fonts in the specified font paths.  If no paths are
    given, will use a standard set of system paths, as well as the
    list of fonts tracked by fontconfig if fontconfig is installed and
    available.  A list of TrueType fonts are returned by default with
    AFM fonts as an option.
    """
    fontfiles = set()
    # 获取与 fontext 对应的文件扩展名列表
    fontexts = get_fontext_synonyms(fontext)

    if fontpaths is None:
        if sys.platform == 'win32':
            # 对于 Windows 平台，获取已安装的字体路径列表
            installed_fonts = _get_win32_installed_fonts()
            fontpaths = []
        else:
            # 对于非 Windows 平台，使用 fontconfig 获取已安装的字体路径列表
            installed_fonts = _get_fontconfig_fonts()
            if sys.platform == 'darwin':
                # 如果是 macOS 平台，添加 macOS 特定的字体路径
                installed_fonts += _get_macos_fonts()
                # 设置字体路径为 X11 和 macOS 字体目录的组合
                fontpaths = [*X11FontDirectories, *OSXFontDirectories]
            else:
                # 对于其他非 Windows 平台，设置字体路径为 X11 字体目录
                fontpaths = X11FontDirectories
        # 更新 fontfiles 集合，包含符合扩展名条件的字体文件路径
        fontfiles.update(str(path) for path in installed_fonts
                         if path.suffix.lower()[1:] in fontexts)

    elif isinstance(fontpaths, str):
        # 如果 fontpaths 是字符串，则转换为单元素列表
        fontpaths = [fontpaths]

    # 遍历指定的字体路径列表，更新 fontfiles 集合
    for path in fontpaths:
        fontfiles.update(map(os.path.abspath, list_fonts(path, fontexts)))

    # 返回存在的字体文件路径列表
    return [fname for fname in fontfiles if os.path.exists(fname)]


@dataclasses.dataclass(frozen=True)
class FontEntry:
    """
    A class for storing Font properties.

    It is used when populating the font lookup dictionary.
    """

    fname: str = ''
    name: str = ''
    style: str = 'normal'
    variant: str = 'normal'
    weight: str | int = 'normal'
    stretch: str = 'normal'
    size: str = 'medium'

    def _repr_html_(self) -> str:
        # 返回 HTML 格式的字符串，显示此 FontEntry 对象的图像表示
        png_stream = self._repr_png_()
        png_b64 = b64encode(png_stream).decode()
        return f"<img src=\"data:image/png;base64, {png_b64}\" />"
    # 定义一个方法 `_repr_png_`，用于生成表示对象的 PNG 图片数据，返回字节流形式的图片内容
    def _repr_png_(self) -> bytes:
        # 导入 Figure 类，用于创建图形对象，注意这里可能会出现循环导入的问题
        from matplotlib.figure import Figure  # Circular import.
        # 创建一个新的 Figure 对象
        fig = Figure()
        # 如果对象的文件名不为空，则将其作为字体路径
        font_path = Path(self.fname) if self.fname != '' else None
        # 在 Figure 对象上添加文本，位置为 (0, 0)，文本内容为对象的名称，使用指定的字体路径
        fig.text(0, 0, self.name, font=font_path)
        # 创建一个 BytesIO 对象作为图像数据的缓冲区
        with BytesIO() as buf:
            # 将 Figure 对象保存为 PNG 格式，存储到 buf 中，设置紧凑的边界框和透明背景
            fig.savefig(buf, bbox_inches='tight', transparent=True)
            # 返回 buf 中的所有数据，即 PNG 图片的字节流
            return buf.getvalue()
def ttfFontProperty(font):
    """
    Extract information from a TrueType font file.

    Parameters
    ----------
    font : `.FT2Font`
        The TrueType font file from which information will be extracted.

    Returns
    -------
    `FontEntry`
        The extracted font properties.

    """
    # 获取字体的家族名称
    name = font.family_name

    #  Styles are: italic, oblique, and normal (default)

    # 从字体对象中获取字形信息
    sfnt = font.get_sfnt()
    # 定义用于在字形信息中查找特定内容的键值对
    mac_key = (1,  # platform: macintosh
               0,  # id: roman
               0)  # langid: english
    ms_key = (3,   # platform: microsoft
              1,   # id: unicode_cs
              0x0409)  # langid: english_united_states

    # 检索字形信息中的特定表格内容，并转换为小写
    sfnt2 = (sfnt.get((*mac_key, 2), b'').decode('latin-1').lower() or
             sfnt.get((*ms_key, 2), b'').decode('utf-16-be').lower())
    sfnt4 = (sfnt.get((*mac_key, 4), b'').decode('latin-1').lower() or
             sfnt.get((*ms_key, 4), b'').decode('utf-16-be').lower())

    # 根据字形信息确定字体的样式
    if sfnt4.find('oblique') >= 0:
        style = 'oblique'
    elif sfnt4.find('italic') >= 0:
        style = 'italic'
    elif sfnt2.find('regular') >= 0:
        style = 'normal'
    elif font.style_flags & ft2font.ITALIC:
        style = 'italic'
    else:
        style = 'normal'

    #  Variants are: small-caps and normal (default)

    # 判断字体名称是否为小型大写字体
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'

    # The weight-guessing algorithm is directly translated from fontconfig
    # 2.13.1's FcFreeTypeQueryFaceInternal (fcfreetype.c).
    # 定义用于推测字体粗细的子系列索引
    wws_subfamily = 22
    typographic_subfamily = 16
    font_subfamily = 2
    # 检索特定子系列索引下的字体信息，并转换为字符串列表
    styles = [
        sfnt.get((*mac_key, wws_subfamily), b'').decode('latin-1'),
        sfnt.get((*mac_key, typographic_subfamily), b'').decode('latin-1'),
        sfnt.get((*mac_key, font_subfamily), b'').decode('latin-1'),
        sfnt.get((*ms_key, wws_subfamily), b'').decode('utf-16-be'),
        sfnt.get((*ms_key, typographic_subfamily), b'').decode('utf-16-be'),
        sfnt.get((*ms_key, font_subfamily), b'').decode('utf-16-be'),
    ]
    # 筛选出非空的样式列表或者使用字体对象的样式名称作为备选
    styles = [*filter(None, styles)] or [font.style_name]
    # 从 fontconfig 的 FcFreeTypeQueryFaceInternal 函数获取字体的权重信息。

    # 获取字体的 OS/2 表格，用于获取字体的权重信息。
    os2 = font.get_sfnt_table("OS/2")
    if os2 and os2["version"] != 0xffff:
        return os2["usWeightClass"]
    
    # 如果获取 OS/2 表格失败或者版本信息为 0xffff，则尝试获取字体的 PostScript 字体信息中的权重。
    try:
        ps_font_info_weight = (
            font.get_ps_font_info()["weight"].replace(" ", "") or "")
    except ValueError:
        pass
    else:
        # 使用正则表达式匹配 PS 字体信息中的权重，返回对应的权重值。
        for regex, weight in _weight_regexes:
            if re.fullmatch(regex, ps_font_info_weight, re.I):
                return weight
    
    # 如果无法从 OS/2 表格或 PS 字体信息中获取权重信息，则尝试从样式名中获取。
    for style in styles:
        style = style.replace(" ", "")
        for regex, weight in _weight_regexes:
            if re.search(regex, style, re.I):
                return weight
    
    # 如果字体的样式标志表明为粗体 (BOLD)，则返回 700，表示 "bold"。
    if font.style_flags & ft2font.BOLD:
        return 700  
    # 默认返回 500，表示 "medium"，并非 "regular"。
    return 500

weight = int(get_weight())

#  Stretch 可以是绝对和相对的拉伸程度
#  绝对拉伸程度有：ultra-condensed, extra-condensed, condensed,
#    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
#    和 ultra-expanded.
#  相对拉伸程度有：wider, narrower
#  Child 值是：inherit

if any(word in sfnt4 for word in ['narrow', 'condensed', 'cond']):
    stretch = 'condensed'
elif 'demi cond' in sfnt4:
    stretch = 'semi-condensed'
elif any(word in sfnt4 for word in ['wide', 'expanded', 'extended']):
    stretch = 'expanded'
else:
    stretch = 'normal'

#  Sizes 可以是绝对和相对的大小。
#  绝对大小有：xx-small, x-small, small, medium, large, x-large,
#    和 xx-large.
#  相对大小有：larger, smaller
#  Length 值是绝对字体大小，例如 12pt
#  百分比值以 'em' 为单位，是最健壮的规范。

# 如果字体不可缩放，则抛出未实现错误。
if not font.scalable:
    raise NotImplementedError("Non-scalable fonts are not supported")
size = 'scalable'

# 返回 FontEntry 对象，包括字体的各种属性：文件名、名称、样式、变体、权重、拉伸程度、大小。
return FontEntry(font.fname, name, style, variant, weight, stretch, size)
def afmFontProperty(fontpath, font):
    """
    Extract information from an AFM font file.

    Parameters
    ----------
    fontpath : str
        The filename corresponding to *font*.
    font : AFM
        The AFM font file from which information will be extracted.

    Returns
    -------
    `FontEntry`
        The extracted font properties.
    """

    # 获取字体的族名
    name = font.get_familyname()
    # 获取字体的名称，并转换为小写
    fontname = font.get_fontname().lower()

    # 根据字体的角度和族名判断字体样式
    if font.get_angle() != 0 or 'italic' in name.lower():
        style = 'italic'
    elif 'oblique' in name.lower():
        style = 'oblique'
    else:
        style = 'normal'

    # 检查是否为小型大写字体
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'

    # 获取字体的重量，并转换为小写
    weight = font.get_weight().lower()
    # 若字重不在预定义的字典中，则设为 'normal'
    if weight not in weight_dict:
        weight = 'normal'

    # 判断字体的拉伸程度
    if 'demi cond' in fontname:
        stretch = 'semi-condensed'
    elif any(word in fontname for word in ['narrow', 'cond']):
        stretch = 'condensed'
    elif any(word in fontname for word in ['wide', 'expanded', 'extended']):
        stretch = 'expanded'
    else:
        stretch = 'normal'

    # 字体大小始终为可伸缩
    size = 'scalable'

    # 返回字体属性对象 FontEntry
    return FontEntry(fontpath, name, style, variant, weight, stretch, size)



class FontProperties:
    """
    A class for storing and manipulating font properties.

    The font properties are the six properties described in the
    `W3C Cascading Style Sheet, Level 1
    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification and *math_fontfamily* for math fonts:

    - family: A list of font names in decreasing order of priority.
      The items may include a generic font family name, either 'sans-serif',
      'serif', 'cursive', 'fantasy', or 'monospace'.  In that case, the actual
      font to be used will be looked up from the associated rcParam during the
      search process in `.findfont`. Default: :rc:`font.family`

    - style: Either 'normal', 'italic' or 'oblique'.
      Default: :rc:`font.style`

    - variant: Either 'normal' or 'small-caps'.
      Default: :rc:`font.variant`
    """
    pass
    """
    - stretch: 在范围0-1000内的数值，或者以下之一：
               'ultra-condensed', 'extra-condensed', 'condensed',
               'semi-condensed', 'normal', 'semi-expanded', 'expanded',
               'extra-expanded' 或 'ultra-expanded'。默认值为：:rc:`font.stretch`

    - weight: 在范围0-1000内的数值，或者以下之一：
             'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
             'extra bold', 'black'。默认值为：:rc:`font.weight`

    - size: 相对值 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
            或绝对字体大小，例如 10。默认值为：:rc:`font.size`

    - math_fontfamily: 用于渲染数学文本的字体系列。支持的值有：
                      'dejavusans', 'dejavuserif', 'cm',
                      'stix', 'stixsans' 和 'custom'。默认值为：:rc:`mathtext.fontset`

    或者，可以使用字体文件的绝对路径，通过 *fname* 参数来指定。然而，在这种情况下，
    直接将路径（作为 `pathlib.Path` 而不是 `str`）传递给 `.Text` 对象的 *font* 参数通常更简单。

    推荐使用相对字体大小值，例如 'large'，而不是绝对字体大小值，例如 12。
    这种方法允许根据字体管理器的默认字体大小调整所有文本大小。

    此类还可以接受 fontconfig_ 模式_，如果它是唯一提供的参数。
    这种支持并不依赖于 fontconfig；我们只是借用其模式语法来在这里使用。

    .. _fontconfig: https://www.freedesktop.org/wiki/Software/fontconfig/
    .. _pattern:
       https://www.freedesktop.org/software/fontconfig/fontconfig-user.html

    注意，Matplotlib 的内部字体管理器和 fontconfig 使用不同的算法来查找字体，
    因此相同的模式在 Matplotlib 中的结果可能与其他使用 fontconfig 的应用程序不同。
    """

    def __init__(self, family=None, style=None, variant=None, weight=None,
                 stretch=None, size=None,
                 fname=None,  # 如果设置，这是要使用的硬编码文件名
                 math_fontfamily=None):
        # 设置字体系列
        self.set_family(family)
        # 设置字体样式
        self.set_style(style)
        # 设置字体变体
        self.set_variant(variant)
        # 设置字体粗细
        self.set_weight(weight)
        # 设置字体拉伸
        self.set_stretch(stretch)
        # 设置字体文件名
        self.set_file(fname)
        # 设置字体大小
        self.set_size(size)
        # 设置数学文本字体系列
        self.set_math_fontfamily(math_fontfamily)
        
        # 如果 family 是一个字符串，并且其他参数都为 None，则将其视为 fontconfig 模式
        # 即使在这种情况下，也要先调用其他设置器，将未指定的属性设置为 rcParams 默认值。
        if (isinstance(family, str)
                and style is None and variant is None and weight is None
                and stretch is None and size is None and fname is None):
            # 设置 fontconfig 模式
            self.set_fontconfig_pattern(family)
    @classmethod
    def _from_any(cls, arg):
        """
        从不同类型的输入构建一个 `.FontProperties` 对象。

        - 如果输入为 `None`，则返回使用 rc 值的 `.FontProperties` 对象；
        - 如果输入已经是 `.FontProperties` 对象，则直接返回；
        - 如果输入是 `os.PathLike` 类型，则将其作为字体文件的路径；
        - 如果输入是字符串类型，则解析为字体配置模式；
        - 如果输入是字典类型，则作为关键字参数 `**kwargs` 传递给 `.FontProperties` 构造函数。
        """
        if arg is None:
            return cls()
        elif isinstance(arg, cls):
            return arg
        elif isinstance(arg, os.PathLike):
            return cls(fname=arg)
        elif isinstance(arg, str):
            return cls(arg)
        else:
            return cls(**arg)

    def __hash__(self):
        """
        计算 `.FontProperties` 对象的哈希值。

        哈希值基于以下属性的元组：
        - 字体族列表
        - 字体样式
        - 字体变体
        - 字体粗细
        - 字体拉伸或宽度
        - 字体大小
        - 关联字体文件名
        - 数学字体族
        """
        l = (tuple(self.get_family()),
             self.get_slant(),
             self.get_variant(),
             self.get_weight(),
             self.get_stretch(),
             self.get_size(),
             self.get_file(),
             self.get_math_fontfamily())
        return hash(l)

    def __eq__(self, other):
        """
        判断两个 `.FontProperties` 对象是否相等。

        如果两个对象的哈希值相同，则认为它们相等。
        """
        return hash(self) == hash(other)

    def __str__(self):
        """
        返回 `.FontProperties` 对象的字体配置模式字符串表示。
        """
        return self.get_fontconfig_pattern()

    def get_family(self):
        """
        返回一个包含字体族名称或通用字体族名称的列表。

        返回的列表按优先顺序排列，用于匹配字体时解析相应的 rcParams。
        """
        return self._family

    def get_name(self):
        """
        返回与字体属性最匹配的字体名称。
        """
        return get_font(findfont(self)).family_name

    def get_style(self):
        """
        返回字体样式。可能的取值为：'normal'、'italic' 或 'oblique'。
        """
        return self._slant

    def get_variant(self):
        """
        返回字体变体。可能的取值为：'normal' 或 'small-caps'。
        """
        return self._variant

    def get_weight(self):
        """
        返回字体粗细。可以是 0-1000 范围内的数值，或以下字符串之一：
        'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold',
        'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'。
        """
        return self._weight

    def get_stretch(self):
        """
        返回字体拉伸或宽度。可能的取值为：
        'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed',
        'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'。
        """
        return self._stretch

    def get_size(self):
        """
        返回字体大小。
        """
        return self._size

    def get_file(self):
        """
        返回关联字体的文件名。
        """
        return self._file
    def get_fontconfig_pattern(self):
        """
        Get a fontconfig_ pattern_ suitable for looking up the font as
        specified with fontconfig's ``fc-match`` utility.

        This support does not depend on fontconfig; we are merely borrowing its
        pattern syntax for use here.
        """
        # 调用 generate_fontconfig_pattern 函数生成适用于 fontconfig 的查询模式
        return generate_fontconfig_pattern(self)

    def set_family(self, family):
        """
        Change the font family.  Can be either an alias (generic name
        is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',
        'fantasy', or 'monospace', a real font name or a list of real
        font names.  Real font names are not supported when
        :rc:`text.usetex` is `True`. Default: :rc:`font.family`
        """
        # 如果 family 为 None，则使用默认字体系列配置
        if family is None:
            family = mpl.rcParams['font.family']
        # 如果 family 是字符串，则转换为列表形式
        if isinstance(family, str):
            family = [family]
        # 设置对象的字体系列属性为给定的 family
        self._family = family

    def set_style(self, style):
        """
        Set the font style.

        Parameters
        ----------
        style : {'normal', 'italic', 'oblique'}, default: :rc:`font.style`
        """
        # 如果 style 为 None，则使用默认的字体样式配置
        if style is None:
            style = mpl.rcParams['font.style']
        # 检查 style 是否在有效的样式列表中
        _api.check_in_list(['normal', 'italic', 'oblique'], style=style)
        # 设置对象的字体样式属性为给定的 style
        self._slant = style

    def set_variant(self, variant):
        """
        Set the font variant.

        Parameters
        ----------
        variant : {'normal', 'small-caps'}, default: :rc:`font.variant`
        """
        # 如果 variant 为 None，则使用默认的字体变体配置
        if variant is None:
            variant = mpl.rcParams['font.variant']
        # 检查 variant 是否在有效的变体列表中
        _api.check_in_list(['normal', 'small-caps'], variant=variant)
        # 设置对象的字体变体属性为给定的 variant
        self._variant = variant

    def set_weight(self, weight):
        """
        Set the font weight.

        Parameters
        ----------
        weight : int or {'ultralight', 'light', 'normal', 'regular', 'book', \
                          'medium', 'roman', 'semibold', 'demibold', 'demi', \
                          'bold', 'heavy', 'extra bold', 'black'}, default: :rc:`font.weight`
        """
        # 如果 weight 为 None，则使用默认的字体粗细配置
        if weight is None:
            weight = mpl.rcParams['font.weight']
        # 检查 weight 是否在有效的粗细列表中
        _api.check_in_list(['ultralight', 'light', 'normal', 'regular', 'book',
                            'medium', 'roman', 'semibold', 'demibold', 'demi',
                            'bold', 'heavy', 'extra bold', 'black'], weight=weight)
        # 设置对象的字体粗细属性为给定的 weight
        self._weight = weight
    def set_weight(self, weight):
        """
        Set the font weight.

        Parameters
        ----------
        weight : int or {'medium', 'roman', 'semibold', 'demibold', 'demi',
                 'bold', 'heavy', 'extra bold', 'black'}, default: :rc:`font.weight`
            If int, must be in the range 0-1000.
        """
        # 如果 weight 参数为 None，则使用默认的 mpl.rcParams['font.weight']
        if weight is None:
            weight = mpl.rcParams['font.weight']
        # 如果 weight 在预定义的 weight_dict 中，直接设置字体的权重属性并返回
        if weight in weight_dict:
            self._weight = weight
            return
        # 尝试将 weight 转换为整数类型
        try:
            weight = int(weight)
        except ValueError:
            pass
        else:
            # 如果 weight 在合法的范围内 [0, 1000]，设置字体的权重属性并返回
            if 0 <= weight <= 1000:
                self._weight = weight
                return
        # 如果 weight 不在上述条件中，则抛出 ValueError 异常，提示 weight 参数无效
        raise ValueError(f"{weight=} is invalid")

    def set_stretch(self, stretch):
        """
        Set the font stretch or width.

        Parameters
        ----------
        stretch : int or {'ultra-condensed', 'extra-condensed', 'condensed',
                  'semi-condensed', 'normal', 'semi-expanded', 'expanded',
                  'extra-expanded', 'ultra-expanded'}, default: :rc:`font.stretch`
            If int, must be in the range 0-1000.
        """
        # 如果 stretch 参数为 None，则使用默认的 mpl.rcParams['font.stretch']
        if stretch is None:
            stretch = mpl.rcParams['font.stretch']
        # 如果 stretch 在预定义的 stretch_dict 中，直接设置字体的 stretch 属性并返回
        if stretch in stretch_dict:
            self._stretch = stretch
            return
        # 尝试将 stretch 转换为整数类型
        try:
            stretch = int(stretch)
        except ValueError:
            pass
        else:
            # 如果 stretch 在合法的范围内 [0, 1000]，设置字体的 stretch 属性并返回
            if 0 <= stretch <= 1000:
                self._stretch = stretch
                return
        # 如果 stretch 不在上述条件中，则抛出 ValueError 异常，提示 stretch 参数无效
        raise ValueError(f"{stretch=} is invalid")

    def set_size(self, size):
        """
        Set the font size.

        Parameters
        ----------
        size : float or {'xx-small', 'x-small', 'small', 'medium',
               'large', 'x-large', 'xx-large'}, default: :rc:`font.size`
            If a float, the font size in points. The string values denote
            sizes relative to the default font size.
        """
        # 如果 size 参数为 None，则使用默认的 mpl.rcParams['font.size']
        if size is None:
            size = mpl.rcParams['font.size']
        # 尝试将 size 转换为浮点数类型
        try:
            size = float(size)
        except ValueError:
            # 如果 size 是一个字符串，尝试查找对应的缩放比例，否则抛出 ValueError 异常
            try:
                scale = font_scalings[size]
            except KeyError as err:
                raise ValueError(
                    "Size is invalid. Valid font size are "
                    + ", ".join(map(str, font_scalings))) from err
            else:
                # 计算相对于默认字体大小的实际大小，并设置字体大小属性
                size = scale * FontManager.get_default_size()
        # 如果 size 小于 1.0，记录日志并将字体大小设置为 1.0
        if size < 1.0:
            _log.info('Fontsize %1.2f < 1.0 pt not allowed by FreeType. '
                      'Setting fontsize = 1 pt', size)
            size = 1.0
        # 设置字体的大小属性
        self._size = size

    def set_file(self, file):
        """
        Set the filename of the fontfile to use. In this case, all
        other properties will be ignored.
        """
        # 将 file 参数转换为文件路径的字符串表示，如果 file 为 None，则设置为 None
        self._file = os.fspath(file) if file is not None else None
    # 根据 fontconfig 的模式设置属性。

    def set_fontconfig_pattern(self, pattern):
        """
        Set the properties by parsing a fontconfig_ *pattern*.

        This support does not depend on fontconfig; we are merely borrowing its
        pattern syntax for use here.
        """
        # 解析 fontconfig 模式，并遍历解析后的键值对
        for key, val in parse_fontconfig_pattern(pattern).items():
            # 如果值是列表，则调用对应的 setter 方法，传入列表的第一个元素
            if type(val) is list:
                getattr(self, "set_" + key)(val[0])
            else:
                # 否则直接调用对应的 setter 方法，传入值
                getattr(self, "set_" + key)(val)

    def get_math_fontfamily(self):
        """
        Return the name of the font family used for math text.

        The default font is :rc:`mathtext.fontset`.
        """
        # 返回数学文本使用的字体系列的名称
        return self._math_fontfamily

    def set_math_fontfamily(self, fontfamily):
        """
        Set the font family for text in math mode.

        If not set explicitly, :rc:`mathtext.fontset` will be used.

        Parameters
        ----------
        fontfamily : str
            The name of the font family.

            Available font families are defined in the
            :ref:`default matplotlibrc file <customizing-with-matplotlibrc-files>`.

        See Also
        --------
        .text.Text.get_math_fontfamily
        """
        # 如果 fontfamily 为 None，则使用默认的 mathtext.fontset
        if fontfamily is None:
            fontfamily = mpl.rcParams['mathtext.fontset']
        else:
            # 否则验证 fontfamily 是否在有效的字体集合中
            valid_fonts = _validators['mathtext.fontset'].valid.values()
            _api.check_in_list(valid_fonts, math_fontfamily=fontfamily)
        # 将设置后的字体系列名称存储在 _math_fontfamily 中
        self._math_fontfamily = fontfamily

    def copy(self):
        """Return a copy of self."""
        # 返回当前对象的浅拷贝
        return copy.copy(self)

    # 别名定义
    set_name = set_family  # set_name 别名为 set_family
    get_slant = get_style  # get_slant 别名为 get_style
    set_slant = set_style  # set_slant 别名为 set_style
    get_size_in_points = get_size  # get_size_in_points 别名为 get_size
class _JSONEncoder(json.JSONEncoder):
    # 自定义 JSON 编码器，用于将 FontManager 和 FontEntry 对象编码为 JSON
    def default(self, o):
        # 如果 o 是 FontManager 类型，返回其字典表示，并添加 __class__ 属性
        if isinstance(o, FontManager):
            return dict(o.__dict__, __class__='FontManager')
        # 如果 o 是 FontEntry 类型，返回其字典表示，并处理其文件名路径
        elif isinstance(o, FontEntry):
            d = dict(o.__dict__, __class__='FontEntry')
            try:
                # 缓存与 Matplotlib 数据路径相关的字体路径，便于虚拟环境中使用
                d["fname"] = str(Path(d["fname"]).relative_to(mpl.get_data_path()))
            except ValueError:
                pass
            return d
        else:
            # 对于其他类型的对象，调用父类的 default 方法进行默认处理
            return super().default(o)


def _json_decode(o):
    # JSON 解码函数，根据字典中的 __class__ 属性来反序列化对象
    cls = o.pop('__class__', None)
    if cls is None:
        return o
    elif cls == 'FontManager':
        # 如果是 FontManager 类型，创建新的 FontManager 实例并更新其属性
        r = FontManager.__new__(FontManager)
        r.__dict__.update(o)
        return r
    elif cls == 'FontEntry':
        # 如果是 FontEntry 类型，处理其文件名路径，确保绝对路径存在
        if not os.path.isabs(o['fname']):
            o['fname'] = os.path.join(mpl.get_data_path(), o['fname'])
        r = FontEntry(**o)
        return r
    else:
        # 如果没有对应的 __class__ 属性，抛出 ValueError 异常
        raise ValueError("Don't know how to deserialize __class__=%s" % cls)


def json_dump(data, filename):
    """
    将 FontManager 数据以 JSON 格式写入到指定的文件 *filename* 中。

    See Also
    --------
    json_load

    Notes
    -----
    存储在 Matplotlib 数据路径下的文件路径（通常是 Matplotlib 提供的字体）会相对于
    数据路径进行存储，以确保在虚拟环境中的有效性。

    此函数会临时锁定输出文件，以防止多个进程互相覆盖输出。
    """
    with cbook._lock_path(filename), open(filename, 'w') as fh:
        try:
            # 使用自定义的 _JSONEncoder 类进行 JSON 编码，并写入文件
            json.dump(data, fh, cls=_JSONEncoder, indent=2)
        except OSError as e:
            _log.warning('Could not save font_manager cache %s', e)


def json_load(filename):
    """
    从指定的 JSON 文件 *filename* 中加载一个 FontManager 对象。

    See Also
    --------
    json_dump
    """
    # 打开指定文件并使用 _json_decode 函数作为对象解码的 hook
    with open(filename) as fh:
        return json.load(fh, object_hook=_json_decode)


class FontManager:
    """
    在导入时，FontManager 单例实例创建一个 ttf 和 afm 字体列表，并缓存其 FontProperties。
    FontManager.findfont 方法执行最近邻搜索，找到最匹配的字体规格。如果找不到足够好的匹配，
    则返回默认字体。

    使用 FontManager.addfont 方法添加的字体不会持久保存在缓存中；因此，每次导入 Matplotlib
    时都需要调用 addfont 方法。只有在操作系统无法通过其他方式安装字体时，才应使用此方法。

    Notes
    -----
    必须在全局 FontManager 实例上调用 FontManager.addfont 方法。
    """

class _JSONEncoder(json.JSONEncoder):
    # Custom JSON encoder for encoding FontManager and FontEntry objects to JSON
    def default(self, o):
        # If o is of type FontManager, return its dictionary representation with added __class__ attribute
        if isinstance(o, FontManager):
            return dict(o.__dict__, __class__='FontManager')
        # If o is of type FontEntry, return its dictionary representation and handle its filename path
        elif isinstance(o, FontEntry):
            d = dict(o.__dict__, __class__='FontEntry')
            try:
                # Cache paths of fonts shipped with Matplotlib relative to the Matplotlib data path
                d["fname"] = str(Path(d["fname"]).relative_to(mpl.get_data_path()))
            except ValueError:
                pass
            return d
        else:
            # For other types of objects, call the parent class's default method for default handling
            return super().default(o)


def _json_decode(o):
    # JSON decoding function that deserializes objects based on the __class__ attribute in the dictionary
    cls = o.pop('__class__', None)
    if cls is None:
        return o
    elif cls == 'FontManager':
        # If it's a FontManager type, create a new FontManager instance and update its attributes
        r = FontManager.__new__(FontManager)
        r.__dict__.update(o)
        return r
    elif cls == 'FontEntry':
        # If it's a FontEntry type, handle its filename path to ensure it's an absolute path
        if not os.path.isabs(o['fname']):
            o['fname'] = os.path.join(mpl.get_data_path(), o['fname'])
        r = FontEntry(**o)
        return r
    else:
        # If there's no __class__ attribute, raise a ValueError exception
        raise ValueError("Don't know how to deserialize __class__=%s" % cls)


def json_dump(data, filename):
    """
    Write FontManager data as JSON to the specified file *filename*.

    See Also
    --------
    json_load

    Notes
    -----
    File paths that are children of the Matplotlib data path (typically, fonts
    shipped with Matplotlib) are stored relative to that data path (to remain
    valid across virtualenvs).

    This function temporarily locks the output file to prevent multiple
    processes from overwriting one another's output.
    """
    with cbook._lock_path(filename), open(filename, 'w') as fh:
        try:
            # Use custom _JSONEncoder class for JSON encoding and write to file
            json.dump(data, fh, cls=_JSONEncoder, indent=2)
        except OSError as e:
            _log.warning('Could not save font_manager cache %s', e)


def json_load(filename):
    """
    Load a FontManager object from the JSON file named *filename*.

    See Also
    --------
    json_dump
    """
    # Open the specified file and use _json_decode function as the object decoding hook
    with open(filename) as fh:
        return json.load(fh, object_hook=_json_decode)


class FontManager:
    """
    On import, the FontManager singleton instance creates a list of ttf and
    afm fonts and caches their FontProperties. The FontManager.findfont method
    does a nearest neighbor search to find the font that most closely matches
    the specification. If no good enough match is found, the default font is returned.

    Fonts added with the FontManager.addfont method will not persist in the
    cache; therefore, addfont will need to be called every time Matplotlib is
    imported. This method should only be used if and when a font cannot be
    installed on your operating system by other means.

    Notes
    -----
    The FontManager.addfont method must be called on the global FontManager
    instance.
    """
    Example usage::

        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        font_dirs = ["/resources/fonts"]  # 定义自定义字体文件的路径
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
    """
    # 增加版本号以便在字体缓存数据格式或行为发生更改时重新构建现有字体缓存文件。
    __version__ = 390

    def __init__(self, size=None, weight='normal'):
        self._version = self.__version__  # 初始化版本号

        self.__default_weight = weight  # 设置默认字体粗细
        self.default_size = size  # 设置默认字体大小

        # 创建字体路径列表。
        paths = [cbook._get_data_path('fonts', subdir)
                 for subdir in ['ttf', 'afm', 'pdfcorefonts']]
        _log.debug('font search path %s', paths)  # 输出字体搜索路径日志

        self.defaultFamily = {
            'ttf': 'DejaVu Sans',  # 默认的 TrueType 字体
            'afm': 'Helvetica'}  # 默认的 Adobe 字体

        self.afmlist = []  # 初始化 AFM 字体列表
        self.ttflist = []  # 初始化 TTF 字体列表

        # 延迟 5 秒发送警告。
        timer = threading.Timer(5, lambda: _log.warning(
            'Matplotlib is building the font cache; this may take a moment.'))
        timer.start()
        try:
            for fontext in ["afm", "ttf"]:
                for path in [*findSystemFonts(paths, fontext=fontext),
                             *findSystemFonts(fontext=fontext)]:
                    try:
                        self.addfont(path)  # 添加字体文件到字体管理器
                    except OSError as exc:
                        _log.info("Failed to open font file %s: %s", path, exc)  # 记录打开字体文件失败日志
                    except Exception as exc:
                        _log.info("Failed to extract font properties from %s: "
                                  "%s", path, exc)  # 记录提取字体属性失败日志
        finally:
            timer.cancel()

    def addfont(self, path):
        """
        缓存位于 *path* 的字体的属性，以便让 `FontManager` 可用。从路径后缀推断字体类型。

        Parameters
        ----------
        path : str or path-like

        Notes
        -----
        此方法用于添加自定义字体，而无需将其安装在操作系统中。请参阅 `FontManager` 单例实例，了解此功能的用途和注意事项。
        """
        # 将路径转换为字符串，以便 afmFontProperty 和 FT2Font 可以处理
        path = os.fsdecode(path)
        if Path(path).suffix.lower() == ".afm":
            with open(path, "rb") as fh:
                font = _afm.AFM(fh)
            prop = afmFontProperty(path, font)  # 创建 AFM 字体属性对象
            self.afmlist.append(prop)  # 将属性对象添加到 AFM 字体列表
        else:
            font = ft2font.FT2Font(path)
            prop = ttfFontProperty(font)  # 创建 TTF 字体属性对象
            self.ttflist.append(prop)  # 将属性对象添加到 TTF 字体列表
        self._findfont_cached.cache_clear()  # 清除 findfont 缓存

    @property
    # 返回默认字体的字典，延迟评估以避免在 JSON 序列化中包含 venv 路径
    def defaultFont(self):
        return {ext: self.findfont(family, fontext=ext)
                for ext, family in self.defaultFamily.items()}

    # 返回默认字体的权重
    def get_default_weight(self):
        """
        Return the default font weight.
        """
        return self.__default_weight

    # 返回默认字体的大小
    @staticmethod
    def get_default_size():
        """
        Return the default font size.
        """
        return mpl.rcParams['font.size']

    # 设置默认字体的权重，初始值为 'normal'
    def set_default_weight(self, weight):
        """
        Set the default font weight.  The initial value is 'normal'.
        """
        self.__default_weight = weight

    # 将字体族名扩展为其配置的别名
    @staticmethod
    def _expand_aliases(family):
        if family in ('sans', 'sans serif'):
            family = 'sans-serif'
        return mpl.rcParams['font.' + family]

    # 计算字体族名列表与给定字体族名之间的匹配分数
    def score_family(self, families, family2):
        """
        Return a match score between the list of font families in
        *families* and the font family name *family2*.

        An exact match at the head of the list returns 0.0.

        A match further down the list will return between 0 and 1.

        No match will return 1.0.
        """
        if not isinstance(families, (list, tuple)):
            families = [families]
        elif len(families) == 0:
            return 1.0
        family2 = family2.lower()
        step = 1 / len(families)
        for i, family1 in enumerate(families):
            family1 = family1.lower()
            if family1 in font_family_aliases:
                options = [*map(str.lower, self._expand_aliases(family1))]
                if family2 in options:
                    idx = options.index(family2)
                    return (i + (idx / len(options))) * step
            elif family1 == family2:
                return i * step
        return 1.0

    # 计算字体风格之间的匹配分数
    def score_style(self, style1, style2):
        """
        Return a match score between *style1* and *style2*.

        An exact match returns 0.0.

        A match between 'italic' and 'oblique' returns 0.1.

        No match returns 1.0.
        """
        if style1 == style2:
            return 0.0
        elif (style1 in ('italic', 'oblique')
              and style2 in ('italic', 'oblique')):
            return 0.1
        return 1.0

    # 计算字体变体之间的匹配分数
    def score_variant(self, variant1, variant2):
        """
        Return a match score between *variant1* and *variant2*.

        An exact match returns 0.0, otherwise 1.0.
        """
        if variant1 == variant2:
            return 0.0
        else:
            return 1.0
    def score_stretch(self, stretch1, stretch2):
        """
        Return a match score between *stretch1* and *stretch2*.

        The result is the absolute value of the difference between the
        CSS numeric values of *stretch1* and *stretch2*, normalized
        between 0.0 and 1.0.
        """
        try:
            # Convert stretch1 to an integer if possible
            stretchval1 = int(stretch1)
        except ValueError:
            # If stretch1 is not a valid integer, try to get its value from stretch_dict; default to 500 if not found
            stretchval1 = stretch_dict.get(stretch1, 500)
        try:
            # Convert stretch2 to an integer if possible
            stretchval2 = int(stretch2)
        except ValueError:
            # If stretch2 is not a valid integer, try to get its value from stretch_dict; default to 500 if not found
            stretchval2 = stretch_dict.get(stretch2, 500)
        # Return the normalized absolute difference between stretchval1 and stretchval2
        return abs(stretchval1 - stretchval2) / 1000.0

    def score_weight(self, weight1, weight2):
        """
        Return a match score between *weight1* and *weight2*.

        The result is 0.0 if both weight1 and weight 2 are given as strings
        and have the same value.

        Otherwise, the result is the absolute value of the difference between
        the CSS numeric values of *weight1* and *weight2*, normalized between
        0.05 and 1.0.
        """
        # Exact match of the weight names, e.g. weight1 == weight2 == "regular"
        if cbook._str_equal(weight1, weight2):
            return 0.0
        # Determine the numeric value of weight1 if it's not already a number
        w1 = weight1 if isinstance(weight1, Number) else weight_dict[weight1]
        # Determine the numeric value of weight2 if it's not already a number
        w2 = weight2 if isinstance(weight2, Number) else weight_dict[weight2]
        # Return the normalized absolute difference between w1 and w2, adjusted to range between 0.05 and 1.0
        return 0.95 * (abs(w1 - w2) / 1000) + 0.05

    def score_size(self, size1, size2):
        """
        Return a match score between *size1* and *size2*.

        If *size2* (the size specified in the font file) is 'scalable', this
        function always returns 0.0, since any font size can be generated.

        Otherwise, the result is the absolute distance between *size1* and
        *size2*, normalized so that the usual range of font sizes (6pt -
        72pt) will lie between 0.0 and 1.0.
        """
        if size2 == 'scalable':
            # If size2 is 'scalable', return a perfect match score of 0.0
            return 0.0
        try:
            # Convert size1 to float if possible
            sizeval1 = float(size1)
        except ValueError:
            # If size1 is not a valid float, calculate its value based on default_size and font_scalings
            sizeval1 = self.default_size * font_scalings[size1]
        try:
            # Convert size2 to float if possible
            sizeval2 = float(size2)
        except ValueError:
            # If size2 is not a valid float, return the worst possible match score of 1.0
            return 1.0
        # Return the normalized absolute distance between sizeval1 and sizeval2, normalized to range between 0.0 and 1.0 based on typical font size range (6pt - 72pt)
        return abs(sizeval1 - sizeval2) / 72
    def findfont(self, prop, fontext='ttf', directory=None,
                 fallback_to_default=True, rebuild_if_missing=True):
        """
        Find the path to the font file most closely matching the given font properties.

        Parameters
        ----------
        prop : str or `~matplotlib.font_manager.FontProperties`
            The font properties to search for. This can be either a
            `.FontProperties` object or a string defining a
            `fontconfig patterns`_.

        fontext : {'ttf', 'afm'}, default: 'ttf'
            The extension of the font file:

            - 'ttf': TrueType and OpenType fonts (.ttf, .ttc, .otf)
            - 'afm': Adobe Font Metrics (.afm)

        directory : str, optional
            If given, only search this directory and its subdirectories.

        fallback_to_default : bool
            If True, will fall back to the default font family (usually
            "DejaVu Sans" or "Helvetica") if the first lookup hard-fails.

        rebuild_if_missing : bool
            Whether to rebuild the font cache and search again if the first
            match appears to point to a nonexisting font (i.e., the font cache
            contains outdated entries).

        Returns
        -------
        str
            The filename of the best matching font.

        Notes
        -----
        This performs a nearest neighbor search.  Each font is given a
        similarity score to the target font properties.  The first font with
        the highest score is returned.  If no matches below a certain
        threshold are found, the default font (usually DejaVu Sans) is
        returned.

        The result is cached, so subsequent lookups don't have to
        perform the O(n) nearest neighbor search.

        See the `W3C Cascading Style Sheet, Level 1
        <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation
        for a description of the font finding algorithm.

        .. _fontconfig patterns:
           https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
        """
        # 调用 _findfont_cached 方法以找到与给定字体属性最匹配的字体文件路径
        # 传递相关的 rcParams 和字体管理器 self，以防止在 rcParam 更改后使用过期的缓存条目
        rc_params = tuple(tuple(mpl.rcParams[key]) for key in [
            "font.serif", "font.sans-serif", "font.cursive", "font.fantasy",
            "font.monospace"])
        ret = self._findfont_cached(
            prop, fontext, directory, fallback_to_default, rebuild_if_missing,
            rc_params)
        # 如果返回值是 _ExceptionProxy 对象，则抛出其中的异常
        if isinstance(ret, _ExceptionProxy):
            raise ret.klass(ret.message)
        # 返回找到的字体文件路径
        return ret

    def get_font_names(self):
        """Return the list of available fonts."""
        # 返回当前字体管理器中可用字体名称的列表
        return list({font.name for font in self.ttflist})

    @lru_cache(1024)
# 使用 functools 模块中的 lru_cache 装饰器，将函数 is_opentype_cff_font 声明为带有缓存功能的函数
@lru_cache
def is_opentype_cff_font(filename):
    """
    Return whether the given font is a Postscript Compact Font Format Font
    embedded in an OpenType wrapper.  Used by the PostScript and PDF backends
    that cannot subset these fonts.
    """
    # 检查文件名的扩展名是否为 '.otf'，如果是则执行以下代码块
    if os.path.splitext(filename)[1].lower() == '.otf':
        # 使用二进制模式打开文件，并读取前四个字节，返回比较结果是否等于字节串 b"OTTO"
        with open(filename, 'rb') as fd:
            return fd.read(4) == b"OTTO"
    else:
        # 如果文件名不以 '.otf' 结尾，则返回 False
        return False


# 使用 functools 模块中的 lru_cache 装饰器，将函数 _get_font 声明为带有缓存功能的函数，缓存大小为 64
@lru_cache(64)
def _get_font(font_filepaths, hinting_factor, *, _kerning_factor, thread_id):
    # 取出字体文件路径列表中的第一个路径作为主要字体文件路径，其余路径作为备用列表
    first_fontpath, *rest = font_filepaths
    # 创建一个 FT2Font 对象，使用 hinting_factor 进行字体处理，_kerning_factor 用于字距调整
    return ft2font.FT2Font(
        first_fontpath, hinting_factor,
        _fallback_list=[
            # 为备用列表中的每个路径创建 FT2Font 对象，并传入 hinting_factor 和 _kerning_factor 参数
            ft2font.FT2Font(
                fpath, hinting_factor,
                _kerning_factor=_kerning_factor
            )
            for fpath in rest
        ],
        _kerning_factor=_kerning_factor
    )


# 如果操作系统支持在 fork 后注册操作，则调用 os.register_at_fork 方法，在子进程中清除 _get_font 函数的缓存
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_get_font.cache_clear)


# 使用 functools 模块中的 lru_cache 装饰器，将函数 _cached_realpath 声明为带有缓存功能的函数，缓存大小为 64
@lru_cache(64)
def _cached_realpath(path):
    # 返回给定路径的规范化版本，避免在 PDF/PS 输出中重复嵌入同一字体
    return os.path.realpath(path)


# 定义函数 get_font，用于获取给定字体文件路径列表的 FT2Font 对象
def get_font(font_filepaths, hinting_factor=None):
    """
    Get an `.ft2font.FT2Font` object given a list of file paths.

    Parameters
    ----------
    font_filepaths : Iterable[str, Path, bytes], str, Path, bytes
        Relative or absolute paths to the font files to be used.

        If a single string, bytes, or `pathlib.Path`, then it will be treated
        as a list with that entry only.

        If more than one filepath is passed, then the returned FT2Font object
        will fall back through the fonts, in the order given, to find a needed
        glyph.

    Returns
    -------
    `.ft2font.FT2Font`

    """
    # 如果 font_filepaths 是单个路径（字符串、字节串或 pathlib.Path 对象），则转为包含单个路径的元组
    if isinstance(font_filepaths, (str, Path, bytes)):
        paths = (_cached_realpath(font_filepaths),)
    else:
        # 否则，将所有路径规范化后存储在元组 paths 中
        paths = tuple(_cached_realpath(fname) for fname in font_filepaths)

    # 如果未提供 hinting_factor 参数，则使用默认配置中的 'text.hinting_factor'
    if hinting_factor is None:
        hinting_factor = mpl.rcParams['text.hinting_factor']

    # 调用 _get_font 函数，传入规范化后的路径元组 paths，hinting_factor 和 _kerning_factor 参数，
    # 并根据线程 ID 进行缓存区分，返回 FT2Font 对象
    return _get_font(
        paths,
        hinting_factor,
        _kerning_factor=mpl.rcParams['text.kerning_factor'],
        thread_id=threading.get_ident()
    )


def _load_fontmanager(*, try_read_cache=True):
    # 构建字体管理器缓存文件的路径
    fm_path = Path(
        mpl.get_cachedir(), f"fontlist-v{FontManager.__version__}.json")
    # 如果尝试读取缓存（try_read_cache 为真），则执行以下操作
    if try_read_cache:
        try:
            # 尝试从文件路径 fm_path 中加载 JSON 数据并赋值给 fm
            fm = json_load(fm_path)
        except Exception:
            # 如果加载 JSON 失败，则捕获异常并忽略
            pass
        else:
            # 如果成功加载 JSON 数据并且 fm 的 _version 属性与 FontManager.__version__ 相同
            if getattr(fm, "_version", object()) == FontManager.__version__:
                # 记录调试信息，使用从 fm_path 加载的 fontManager 实例
                _log.debug("Using fontManager instance from %s", fm_path)
                # 直接返回已加载的 fm
                return fm
    
    # 如果没有成功从缓存加载或者版本不匹配，则创建一个新的 FontManager 实例并赋值给 fm
    fm = FontManager()
    # 将新创建的 FontManager 实例 fm 序列化为 JSON，并写入 fm_path 指定的文件中
    json_dump(fm, fm_path)
    # 记录信息日志，表示生成了一个新的 fontManager 实例
    _log.info("generated new fontManager")
    # 返回新创建的 fontManager 实例 fm
    return fm
# 加载字体管理器并将其赋值给变量fontManager
fontManager = _load_fontmanager()
# 获取字体管理器中的findfont方法，并赋值给变量findfont
findfont = fontManager.findfont
# 获取字体管理器中的get_font_names方法，并赋值给变量get_font_names
get_font_names = fontManager.get_font_names
```