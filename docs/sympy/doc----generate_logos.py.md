# `D:\src\scipysrc\sympy\doc\generate_logos.py`

```
#!/usr/bin/env python

"""
This script creates logos of different formats from the source "sympy.svg"

Requirements:
    rsvg-convert    - for converting to *.png format
                    (librsvg2-bin deb package)
    imagemagick     - for converting to *.ico favicon format
"""

from argparse import ArgumentParser  # 导入参数解析器模块
import xml.dom.minidom  # 导入处理 XML 的模块
import os.path  # 导入操作路径的模块
import logging  # 导入日志记录模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关的模块
from platform import system  # 导入获取操作系统信息的模块

default_source_dir = os.path.join(os.path.dirname(__file__), "src", "logo")  # 默认的源文件目录路径
default_output_dir = os.path.join(os.path.dirname(__file__), "_build", "logo")  # 默认的输出目录路径
default_source_svg = "sympy.svg"  # 默认的源 SVG 文件名

# those are the options for resizing versions without tail or text
svg_sizes = {}
svg_sizes['notail'] = {
    "prefix":"notail", "dx":-70, "dy":-20, "size":690,
    "title":"SymPy Logo, with no tail"}  # 不带尾巴的 SymPy Logo 版本参数
svg_sizes['notail-notext'] = {
    "prefix":"notailtext", "dx":-70, "dy":60, "size":690,
    "title":"SymPy Logo, with no tail, no text"}  # 不带尾巴和文本的 SymPy Logo 版本参数
svg_sizes['notext'] = {
    "prefix":"notext", "dx":-7, "dy":90, "size":750,
    "title":"SymPy Logo, with no text"}  # 不带文本的 SymPy Logo 版本参数

# The list of identifiers of various versions
versions = ['notail', 'notail-notext', 'notext']  # 不同版本的标识符列表

parser = ArgumentParser(usage="%(prog)s [options ...]")  # 创建参数解析器实例

parser.add_argument("--source-dir", type=str, dest="source_dir",
    help="Directory of the source *.svg file [default: %(default)s]",
    default=default_source_dir)  # 源 SVG 文件目录选项

parser.add_argument("--source-svg", type=str, dest="source_svg",
    help="File name of the source *.svg file [default: %(default)s]",
    default=default_source_svg)  # 源 SVG 文件名选项

parser.add_argument("--svg", action="store_true", dest="generate_svg",
    help="Generate *.svg versions without tails " \
        "and without text 'SymPy' [default: %(default)s]",
    default=False)  # 生成不带尾巴和文本的 SVG 版本选项

parser.add_argument("--png", action="store_true", dest="generate_png",
    help="Generate *.png versions [default: %(default)s]",
    default=False)  # 生成 PNG 版本选项

parser.add_argument("--ico", action="store_true", dest="generate_ico",
    help="Generate *.ico versions [default: %(default)s]",
    default=False)  # 生成 ICO 版本选项

parser.add_argument("--clear", action="store_true", dest="clear",
    help="Remove temporary files [default: %(default)s]",
    default=False)  # 清除临时文件选项

parser.add_argument("-a", "--all", action="store_true", dest="generate_all",
    help="Shorthand for '--svg --png --ico --clear' options " \
        "[default: %(default)s]",
    default=True)  # 生成所有版本选项的简写

parser.add_argument("-s", "--sizes", type=str, dest="sizes",
    help="Sizes of png pictures [default: %(default)s]",
    default="160,500")  # PNG 图片的尺寸选项

parser.add_argument("--icon-sizes", type=str, dest="icon_sizes",
    help="Sizes of icons embedded in favicon file [default: %(default)s]",
    default="16,32,48,64")  # 嵌入到 favicon 文件中的图标尺寸选项

parser.add_argument("--output-dir", type=str, dest="output_dir",
    help="Output dir [default: %(default)s]",
    default=default_output_dir)  # 输出目录选项

parser.add_argument("-d", "--debug", action="store_true", dest="debug",
    help="Print debug log [default: %(default)s]",
    default=False)  # 打印调试日志选项
    default=False)



    # 设置函数参数的默认值为 False
    default=False


这行代码定义了一个函数的参数 `default`，并将其默认值设为 `False`。
def main():
    # 解析命令行参数
    options, args = parser.parse_known_args()
    # 如果启用调试模式，则配置日志级别为调试
    if options.debug:
        logging.basicConfig(level=logging.DEBUG)

    # 源文件路径为选项中指定的源目录和源 SVG 文件名的组合
    fn_source = os.path.join(options.source_dir, options.source_svg)

    # 如果需要生成 SVG 或者需要生成所有版本
    if options.generate_svg or options.generate_all:
        # 生成无尾无文本版本
        generate_notail_notext_versions(fn_source, options.output_dir)

    # 如果需要生成 PNG 或者需要生成所有版本
    if options.generate_png or options.generate_all:
        # 解析并转换尺寸参数为整数列表
        sizes = options.sizes.split(",")
        sizes = [int(s) for s in sizes]
        # 转换为 PNG 格式
        convert_to_png(fn_source, options.output_dir, sizes)

    # 如果需要生成 ICO 或者需要生成所有版本
    if options.generate_ico or options.generate_all:
        # 解析并转换图标尺寸参数为整数列表
        sizes = options.icon_sizes.split(",")
        sizes = [int(s) for s in sizes]
        # 转换为 ICO 格式
        convert_to_ico(fn_source, options.output_dir, sizes)

def generate_notail_notext_versions(fn_source, output_dir):
    # 遍历所有版本
    for ver in versions:
        # 获取版本属性
        properties = svg_sizes[ver]

        # 加载 SVG 文档
        doc = load_svg(fn_source)

        # 根据版本键转换为布尔值元组
        (notail, notext) = versionkey_to_boolean_tuple(ver)

        # 查找并处理尾部图像
        g_tail = searchElementById(doc, "SnakeTail", "g")
        if notail:
            g_tail.setAttribute("display", "none")

        # 查找并处理文本图像
        g_text = searchElementById(doc, "SymPy_text", "g")
        if notext:
            g_text.setAttribute("display", "none")

        # 查找并处理 Logo 图像
        g_logo = searchElementById(doc, "SympyLogo", "g")
        dx = properties["dx"]
        dy = properties["dy"]
        transform = "translate(%d,%d)" % (dx, dy)
        g_logo.setAttribute("transform", transform)

        # 查找 SVG 元素
        svg = searchElementById(doc, "svg_SympyLogo", "svg")
        newsize = properties["size"]
        svg.setAttribute("width", "%d" % newsize)
        svg.setAttribute("height", "%d" % newsize)

        # 更新标题
        title = svg.getElementsByTagName("title")[0]
        title.firstChild.data = properties["title"]

        # 更新描述信息
        desc = svg.getElementsByTagName("desc")[0]
        desc.appendChild(
            doc.createTextNode(
                "\n\nThis file is generated from %s !" % fn_source))

        # 生成输出文件名
        fn_out = get_svg_filename_from_versionkey(fn_source, ver)
        fn_out = os.path.join(output_dir, fn_out)
        # 保存 SVG 文档
        save_svg(fn_out, doc)

def convert_to_png(fn_source, output_dir, sizes):
    # 将所有版本插入到 SVG 列表中
    svgs = list(versions)
    svgs.insert(0, '')

    # 定义命令
    cmd = "rsvg-convert"
    # 启动子进程
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    # 等待子进程结束
    p.communicate()
    # 如果返回码为 127，显示错误信息并退出
    if p.returncode == 127:
        logging.error(
            "%s: command not found. Install librsvg" % cmd)
        sys.exit(p.returncode)
    for ver in svgs:
        # 遍历输入的版本列表 svgs
        if ver == '':
            # 如果版本号为空字符串，则使用原始文件名 fn_source
            fn_svg = fn_source
            # 如果运行系统是 Windows，切换到默认的源目录
            if system()[0:3].lower() == "win":
                os.chdir(default_source_dir)
        else:
            # 否则，根据版本号获取对应的 SVG 文件名
            fn_svg = get_svg_filename_from_versionkey(fn_source, ver)
            # 构建输出目录下的完整 SVG 文件路径
            fn_svg = os.path.join(output_dir, fn_svg)
            # 如果运行系统是 Windows，切换到默认的输出目录
            if system()[0:3].lower() == "win":
                os.chdir(default_output_dir)

        # 从完整路径中获取基本文件名
        basename = os.path.basename(fn_svg)
        # 分离基本文件名和扩展名
        name, ext = os.path.splitext(basename)
        
        # 遍历指定的尺寸列表 sizes
        for size in sizes:
            # 如果运行系统是 Windows
            if system()[0:3].lower() == "win":
                # 构建输出 PNG 文件的完整路径
                fn_out = "%s-%dpx.png" % (name, size)
                fn_out = os.path.join(os.pardir, os.pardir, "_build", "logo", fn_out)
                # 构建转换 SVG 到 PNG 的命令
                name_c = "%s.svg" % (name)
                cmd = "rsvg-convert %s -f png -h %d -w %d > %s" % (name_c,
                                                                   size, size,
                                                                   fn_out)
            else:
                # 构建输出 PNG 文件的完整路径
                fn_out = "%s-%dpx.png" % (name, size)
                fn_out = os.path.join(output_dir, fn_out)
                # 构建转换 SVG 到 PNG 的命令
                cmd = "rsvg-convert %s -f png -o %s -h %d -w %d" % (fn_svg,
                                                                    fn_out,
                                                                    size, size)

            # 执行命令并创建子进程
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            # 等待命令执行完成
            p.communicate()
            # 检查命令的返回码
            if p.returncode != 0:
                # 如果返回码不为 0，记录错误日志并退出程序
                logging.error("Return code is not 0: Command: %s" % cmd)
                logging.error("return code: %s" % p.returncode)
                sys.exit(p.returncode)
            else:
                # 否则，记录调试日志
                logging.debug("command: %s" % cmd)
                logging.debug("return code: %s" % p.returncode)
def convert_to_ico(fn_source, output_dir, sizes):
    # 首先准备将嵌入到 *.ico 文件中的 *.png 文件。
    convert_to_png(fn_source, output_dir, sizes)

    # 从 versions 列表创建 svgs 列表，并插入一个空字符串作为首个元素。
    svgs = list(versions)
    svgs.insert(0, '')

    # 根据操作系统判断使用的命令工具（ImageMagick 或 convert）
    if system()[0:3].lower() == "win":
        cmd = "magick"
    else:
        cmd = "convert"

    # 启动子进程，执行命令工具，捕获标准输入输出及错误
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    p.communicate()

    # 如果命令返回码为 127，则输出错误信息并退出程序
    if p.returncode == 127:
        logging.error("%s: command not found. Install imagemagick" % cmd)
        sys.exit(p.returncode)

    # 如果操作系统是 Windows，则切换至默认输出目录
    if system()[0:3].lower() == "win":
        os.chdir(default_output_dir)

    # 遍历 svgs 列表中的版本键
    for ver in svgs:
        # 根据版本键获取对应的 SVG 文件名
        if ver == '':
            fn_svg = fn_source
        else:
            fn_svg = get_svg_filename_from_versionkey(fn_source, ver)
            fn_svg = os.path.join(output_dir, fn_svg)

        # 从 SVG 文件名中提取基本文件名和扩展名
        basename = os.path.basename(fn_svg)
        name, ext = os.path.splitext(basename)

        # 计算生成的 *.png 文件列表
        pngs = []
        for size in sizes:
            fn_png= "%s-%dpx.png" % (name, size)
            if system()[0:3].lower() != "win":
                fn_png = os.path.join(output_dir, fn_png)
            pngs.append(fn_png)

        # 设置输出的 *.ico 文件名
        fn_out = "%s-favicon.ico" % name
        if system()[0:3].lower() != "win":
            fn_out = os.path.join(output_dir, fn_out)

        # 构建转换命令并执行，转换 *.png 文件为 *.ico 文件
        cmd = "{} {} {}".format(cmd, " ".join(pngs), fn_out)
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        p.communicate()

        # 检查命令执行返回码，如果不为 0，则输出错误信息并退出程序
        if p.returncode != 0:
            logging.error("Return code is not 0: Command: %s" % cmd)
            logging.error("return code: %s" % p.returncode)
            sys.exit(p.returncode)
        else:
            logging.debug("command: %s" % cmd)
            logging.debug("return code: %s" % p.returncode)
    # 使用 "wb" 模式打开文件 fn，准备写入二进制数据
    with open(fn, "wb") as f:
        # 将 XML 文档对象转换为 UTF-8 编码的 XML 字符串
        xmlstr = doc.toxml("utf-8")
        # 将 XML 字符串写入打开的文件对象 f
        f.write(xmlstr)
        # 记录日志信息，指示文件 fn 已保存
        logging.info(" File saved: %s" % fn)
# 调用主程序入口函数
main()
```