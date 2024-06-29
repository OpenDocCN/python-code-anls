# `.\numpy\tools\c_coverage\c_coverage_report.py`

```py
#!/usr/bin/env python3
"""
A script to create C code-coverage reports based on the output of
valgrind's callgrind tool.

"""
import os  # 导入操作系统相关的功能
import re  # 导入正则表达式模块
import sys  # 导入系统相关的功能
from xml.sax.saxutils import quoteattr, escape  # 导入 XML 相关工具

try:
    import pygments  # 尝试导入 Pygments
    if tuple([int(x) for x in pygments.__version__.split('.')]) < (0, 11):
        raise ImportError()
    from pygments import highlight  # 导入代码高亮函数
    from pygments.lexers import CLexer  # 导入 C 语言的代码词法分析器
    from pygments.formatters import HtmlFormatter  # 导入生成 HTML 的格式化器
    has_pygments = True  # 记录是否成功导入 Pygments
except ImportError:
    print("This script requires pygments 0.11 or greater to generate HTML")  # 提示需要 Pygments 0.11 或更高版本
    has_pygments = False  # 记录未成功导入 Pygments


class FunctionHtmlFormatter(HtmlFormatter):
    """Custom HTML formatter to insert extra information with the lines."""
    def __init__(self, lines, **kwargs):
        HtmlFormatter.__init__(self, **kwargs)  # 调用父类的初始化方法
        self.lines = lines  # 初始化行信息

    def wrap(self, source, outfile):
        for i, (c, t) in enumerate(HtmlFormatter.wrap(self, source, outfile)):
            as_functions = self.lines.get(i-1, None)  # 获取当前行的附加函数信息
            if as_functions is not None:
                yield 0, ('<div title=%s style="background: #ccffcc">[%2d]' %
                          (quoteattr('as ' + ', '.join(as_functions)),
                           len(as_functions)))  # 在 HTML 中插入附加函数信息的块
            else:
                yield 0, '    '  # 如果没有附加函数信息，则空行
            yield c, t
            if as_functions is not None:
                yield 0, '</div>'  # 结束附加函数信息的 HTML 块


class SourceFile:
    def __init__(self, path):
        self.path = path  # 初始化文件路径
        self.lines = {}  # 初始化行号和附加函数信息的字典

    def mark_line(self, lineno, as_func=None):
        line = self.lines.setdefault(lineno, set())  # 设置行号对应的附加函数信息
        if as_func is not None:
            as_func = as_func.split("'", 1)[0]  # 提取函数名称
            line.add(as_func)  # 添加函数名称到行信息中

    def write_text(self, fd):
        source = open(self.path, "r")  # 打开文件以读取文本
        for i, line in enumerate(source):
            if i + 1 in self.lines:
                fd.write("> ")  # 如果行号在附加函数信息中，则写入标记
            else:
                fd.write("! ")  # 否则写入另一种标记
            fd.write(line)  # 写入源代码行
        source.close()  # 关闭文件

    def write_html(self, fd):
        source = open(self.path, 'r')  # 打开文件以读取 HTML
        code = source.read()  # 读取文件内容
        lexer = CLexer()  # 创建 C 语言代码词法分析器
        formatter = FunctionHtmlFormatter(
            self.lines,
            full=True,
            linenos='inline')  # 使用自定义 HTML 格式化器生成 HTML
        fd.write(highlight(code, lexer, formatter))  # 使用 Pygments 高亮代码
        source.close()  # 关闭文件


class SourceFiles:
    def __init__(self):
        self.files = {}  # 初始化文件字典
        self.prefix = None  # 初始化文件路径前缀为 None

    def get_file(self, path):
        if path not in self.files:
            self.files[path] = SourceFile(path)  # 如果文件不在字典中，则创建新的 SourceFile 对象
            if self.prefix is None:
                self.prefix = path  # 设置文件路径前缀
            else:
                self.prefix = os.path.commonprefix([self.prefix, path])  # 更新文件路径前缀
        return self.files[path]  # 返回文件对象

    def clean_path(self, path):
        path = path[len(self.prefix):]  # 移除路径前缀
        return re.sub(r"[^A-Za-z0-9\.]", '_', path)  # 将非字母数字字符替换为下划线
    # 将数据写入文本文件的方法，接受一个根目录参数 root
    def write_text(self, root):
        # 遍历 self.files 中的路径和源数据
        for path, source in self.files.items():
            # 打开目标文件，使用写模式 'w'，路径为 root 下的清理后的路径
            fd = open(os.path.join(root, self.clean_path(path)), "w")
            # 将源数据写入文件
            source.write_text(fd)
            # 关闭文件描述符
            fd.close()

    # 将数据写入 HTML 文件的方法，接受一个根目录参数 root
    def write_html(self, root):
        # 遍历 self.files 中的路径和源数据
        for path, source in self.files.items():
            # 打开目标文件，使用写模式 'w'，路径为 root 下的清理后的路径后加上 ".html"
            fd = open(os.path.join(root, self.clean_path(path) + ".html"), "w")
            # 将源数据以 HTML 格式写入文件
            source.write_html(fd)
            # 关闭文件描述符
            fd.close()

        # 打开目标文件 'index.html'，使用写模式 'w'
        fd = open(os.path.join(root, 'index.html'), 'w')
        # 写入 HTML 开始标签
        fd.write("<html>")
        # 对文件路径按字母顺序排序
        paths = sorted(self.files.keys())
        # 遍历排序后的文件路径列表
        for path in paths:
            # 写入 HTML 标签，包含链接到各文件的内容
            fd.write('<p><a href="%s.html">%s</a></p>' %
                     (self.clean_path(path), escape(path[len(self.prefix):])))
        # 写入 HTML 结束标签
        fd.write("</html>")
        # 关闭文件描述符
        fd.close()
# 定义一个函数用于收集统计信息，处理压缩的 callgrind 文件
def collect_stats(files, fd, pattern):
    # 定义两个正则表达式模式，用于匹配不同的行格式
    line_regexs = [
        re.compile(r"(?P<lineno>[0-9]+)(\s[0-9]+)+"),  # 匹配形如 "123 456 789"
        re.compile(r"((jump)|(jcnd))=([0-9]+)\s(?P<lineno>[0-9]+)")  # 匹配形如 "jump=123 456"
    ]

    current_file = None  # 当前处理的文件对象
    current_function = None  # 当前处理的函数名
    for i, line in enumerate(fd):
        if re.match("f[lie]=.+", line):  # 如果行以 "f="、"l=" 或 "ie=" 开头
            path = line.split('=', 2)[1].strip()  # 提取路径信息
            if os.path.exists(path) and re.search(pattern, path):  # 如果路径存在且符合指定模式
                current_file = files.get_file(path)  # 获取当前文件对象
            else:
                current_file = None  # 否则置空当前文件对象
        elif re.match("fn=.+", line):  # 如果行以 "fn=" 开头
            current_function = line.split('=', 2)[1].strip()  # 提取函数名信息
        elif current_file is not None:  # 如果当前文件对象不为空
            for regex in line_regexs:  # 遍历每个正则表达式模式
                match = regex.match(line)  # 尝试匹配当前行
                if match:  # 如果匹配成功
                    lineno = int(match.group('lineno'))  # 提取行号
                    current_file.mark_line(lineno, current_function)  # 标记该行号属于当前函数

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)  # 创建命令行参数解析器
    parser.add_argument(
        'callgrind_file', nargs='+',
        help='One or more callgrind files')  # 接收一个或多个 callgrind 文件作为参数
    parser.add_argument(
        '-d', '--directory', default='coverage',
        help='Destination directory for output (default: %(default)s)')  # 指定输出目录，默认为 'coverage'
    parser.add_argument(
        '-p', '--pattern', default='numpy',
        help='Regex pattern to match against source file paths '
             '(default: %(default)s)')  # 指定用于匹配源文件路径的正则表达式模式，默认为 'numpy'
    parser.add_argument(
        '-f', '--format', action='append', default=[],
        choices=['text', 'html'],
        help="Output format(s) to generate. "
             "If option not provided, both will be generated.")  # 指定生成的输出格式，可选 'text' 或 'html'，支持多次指定
    args = parser.parse_args()  # 解析命令行参数

    files = SourceFiles()  # 创建 SourceFiles 对象，用于管理源文件信息
    for log_file in args.callgrind_file:  # 遍历每个输入的 callgrind 文件
        log_fd = open(log_file, 'r')  # 打开文件，准备读取
        collect_stats(files, log_fd, args.pattern)  # 收集统计信息
        log_fd.close()  # 关闭文件

    if not os.path.exists(args.directory):  # 如果指定的输出目录不存在
        os.makedirs(args.directory)  # 创建该目录

    if args.format == []:  # 如果未指定输出格式
        formats = ['text', 'html']  # 默认生成 'text' 和 'html'
    else:
        formats = args.format  # 使用用户指定的输出格式
    if 'text' in formats:  # 如果需要生成文本格式
        files.write_text(args.directory)  # 生成文本输出到指定目录
    if 'html' in formats:  # 如果需要生成 HTML 格式
        if not has_pygments:  # 检查是否具备生成 HTML 所需的 Pygments 库
            print("Pygments 0.11 or later is required to generate HTML")  # 提示需要 Pygments 版本 0.11 或更高
            sys.exit(1)  # 退出程序，返回错误状态码
        files.write_html(args.directory)  # 生成 HTML 输出到指定目录
```