# `markdown\markdown\__main__.py`

```py

# Python Markdown

# A Python implementation of John Gruber's Markdown.

# Documentation: https://python-markdown.github.io/
# GitHub: https://github.com/Python-Markdown/markdown/
# PyPI: https://pypi.org/project/Markdown/

# Started by Manfred Stienstra (http://www.dwerg.net/).
# Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
# Currently maintained by Waylan Limberg (https://github.com/waylan),
# Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

# Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
# Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
# Copyright 2004 Manfred Stienstra (the original version)

# License: BSD (see LICENSE.md for details).

# 导入未来的注释语法
from __future__ import annotations

# 导入所需的模块
import sys
import optparse
import codecs
import warnings
import markdown
try:
    # 尝试导入 PyYAML 库的 unsafe_load 方法
    from yaml import unsafe_load as yaml_load
except ImportError:  # 如果导入失败，则尝试导入旧版本的 PyYAML
    try:
        from yaml import load as yaml_load
    except ImportError:  # 如果还是导入失败，则尝试导入 JSON 库
        from json import load as yaml_load

# 导入日志模块
import logging
from logging import DEBUG, WARNING, CRITICAL

# 获取名为 'MARKDOWN' 的日志记录器
logger = logging.getLogger('MARKDOWN')

# 定义并解析命令行选项
def parse_options(args=None, values=None):
    """
    Define and parse `optparse` options for command-line usage.
    """
    # 定义使用说明、描述和版本信息
    usage = """%prog [options] [INPUTFILE]
       (STDIN is assumed if no INPUTFILE is given)"""
    desc = "A Python implementation of John Gruber's Markdown. " \
           "https://Python-Markdown.github.io/"
    ver = "%%prog %s" % markdown.__version__

    # 创建命令行选项解析器
    parser = optparse.OptionParser(usage=usage, description=desc, version=ver)
    # 添加命令行选项
    parser.add_option("-f", "--file", dest="filename", default=None,
                      help="Write output to OUTPUT_FILE. Defaults to STDOUT.",
                      metavar="OUTPUT_FILE")
    parser.add_option("-e", "--encoding", dest="encoding",
                      help="Encoding for input and output files.",)
    parser.add_option("-o", "--output_format", dest="output_format",
                      default='xhtml', metavar="OUTPUT_FORMAT",
                      help="Use output format 'xhtml' (default) or 'html'.")
    parser.add_option("-n", "--no_lazy_ol", dest="lazy_ol",
                      action='store_false', default=True,
                      help="Observe number of first item of ordered lists.")
    parser.add_option("-x", "--extension", action="append", dest="extensions",
                      help="Load extension EXTENSION.", metavar="EXTENSION")
    parser.add_option("-c", "--extension_configs",
                      dest="configfile", default=None,
                      help="Read extension configurations from CONFIG_FILE. "
                      "CONFIG_FILE must be of JSON or YAML format. YAML "
                      "format requires that a python YAML library be "
                      "installed. The parsed JSON or YAML must result in a "
                      "python dictionary which would be accepted by the "
                      "'extension_configs' keyword on the markdown.Markdown "
                      "class. The extensions must also be loaded with the "
                      "`--extension` option.",
                      metavar="CONFIG_FILE")
    parser.add_option("-q", "--quiet", default=CRITICAL,
                      action="store_const", const=CRITICAL+10, dest="verbose",
                      help="Suppress all warnings.")
    parser.add_option("-v", "--verbose",
                      action="store_const", const=WARNING, dest="verbose",
                      help="Print all warnings.")
    parser.add_option("--noisy",
                      action="store_const", const=DEBUG, dest="verbose",
                      help="Print debug messages.")

    # 解析命令行选项
    (options, args) = parser.parse_args(args, values)

    if len(args) == 0:
        input_file = None
    else:
        input_file = args[0]

    if not options.extensions:
        options.extensions = []

    extension_configs = {}
    if options.configfile:
        with codecs.open(
            options.configfile, mode="r", encoding=options.encoding
        ) as fp:
            try:
                extension_configs = yaml_load(fp)
            except Exception as e:
                message = "Failed parsing extension config file: %s" % \
                          options.configfile
                e.args = (message,) + e.args[1:]
                raise

    opts = {
        'input': input_file,
        'output': options.filename,
        'extensions': options.extensions,
        'extension_configs': extension_configs,
        'encoding': options.encoding,
        'output_format': options.output_format,
        'lazy_ol': options.lazy_ol
    }

    return opts, options.verbose

# 运行 Markdown 命令行工具
def run():  # pragma: no cover
    """Run Markdown from the command line."""

    # 解析选项并根据需要调整日志级别
    options, logging_level = parse_options()
    if not options:
        sys.exit(2)
    logger.setLevel(logging_level)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    if logging_level <= WARNING:
        # 确保显示弃用警告
        warnings.filterwarnings('default')
        logging.captureWarnings(True)
        warn_logger = logging.getLogger('py.warnings')
        warn_logger.addHandler(console_handler)

    # 运行 Markdown
    markdown.markdownFromFile(**options)

# 如果作为模块被直接运行，则执行 run() 函数
if __name__ == '__main__':  # pragma: no cover
    # 支持将模块作为命令行命令运行
    #     python -m markdown [options] [args]
    run()

```