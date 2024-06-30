# `D:\src\scipysrc\scipy\tools\authors.py`

```
#!/usr/bin/env python
"""
List the authors who contributed within a given revision interval::

    python tools/authors.py REV1..REV2

`REVx` being a commit hash.

To change the name mapping, edit .mailmap on the top-level of the
repository.

"""
# Author: Pauli Virtanen <pav@iki.fi>. This script is in the public domain.

import argparse  # 导入用于解析命令行参数的模块
import re  # 导入正则表达式模块
import sys  # 导入系统相关功能的模块
import os  # 导入与操作系统相关的功能模块
import subprocess  # 导入执行外部命令的模块
import collections  # 导入集合数据类型的模块

stdout_b = sys.stdout.buffer  # 获取标准输出的二进制模式流
MAILMAP_FILE = os.path.join(os.path.dirname(__file__), "..", ".mailmap")  # 设置.mailmap文件的路径，用于映射邮箱名

def main():
    p = argparse.ArgumentParser(__doc__.strip())  # 创建参数解析器对象，使用脚本顶部的文档字符串作为帮助信息
    p.add_argument("range", help=argparse.SUPPRESS)  # 添加位置参数range，用于指定版本区间，但不在帮助信息中显示
    p.add_argument("-d", "--debug", action="store_true",
                   help="print debug output")  # 添加可选参数-d或--debug，用于打印调试输出
    p.add_argument("-n", "--new", action="store_true",
                   help="print debug output")  # 添加可选参数-n或--new，用于打印调试输出
    options = p.parse_args()  # 解析命令行参数

    try:
        rev1, rev2 = options.range.split('..')  # 尝试从参数range中分割出两个版本号
    except ValueError:
        p.error("argument is not a revision range")  # 如果分割失败，则显示错误信息并退出

    NAME_MAP = load_name_map(MAILMAP_FILE)  # 加载.mailmap文件中的姓名映射表

    # Analyze log data
    all_authors = set()  # 创建一个空集合，用于存储所有的作者
    authors = collections.Counter()  # 创建一个计数器对象，用于存储各作者的贡献次数

    def analyze_line(line, names, disp=False):
        line = line.strip().decode('utf-8')  # 去除行首尾空白，并解码为UTF-8格式

        # Check the commit author name
        m = re.match('^@@@([^@]*)@@@', line)  # 使用正则表达式匹配行中的作者姓名
        if m:
            name = m.group(1)  # 获取匹配到的作者姓名
            line = line[m.end():]  # 截取匹配后的剩余部分
            name = NAME_MAP.get(name, name)  # 根据姓名映射表，将姓名转换为规范化的格式
            if disp:
                if name not in names:
                    stdout_b.write(("    - Author: %s\n" % name).encode('utf-8'))  # 如果是调试模式，输出作者信息
            names.update((name,))  # 将作者姓名加入到集合中

        # Look for "thanks to" messages in the commit log
        m = re.search(
            r'([Tt]hanks to|[Cc]ourtesy of|Co-authored-by:) '
            r'([A-Z][A-Za-z]*? [A-Z][A-Za-z]*? [A-Z][A-Za-z]*|[A-Z][A-Za-z]*? [A-Z]\.'
            r' [A-Z][A-Za-z]*|[A-Z][A-Za-z ]*? [A-Z][A-Za-z]*|[a-z0-9]+)($|\.| )',
            line,
        )  # 使用正则表达式查找提交日志中的"thanks to"消息
        if m:
            name = m.group(2)  # 获取匹配到的感谢消息中的姓名
            if name not in ('this',):
                if disp:
                    stdout_b.write("    - Log   : %s\n" % line.strip().encode('utf-8'))  # 如果是调试模式，输出感谢消息
                name = NAME_MAP.get(name, name)  # 根据姓名映射表，将姓名转换为规范化的格式
                names.update((name,))  # 将姓名加入到集合中

            line = line[m.end():].strip()  # 截取匹配后的剩余部分，并去除首尾空白
            line = re.sub(r'^(and|, and|, ) ', 'Thanks to ', line)  # 将行中的连接词替换为"Thanks to"
            analyze_line(line.encode('utf-8'), names)  # 递归调用analyze_line分析替换后的行

    # Find all authors before the named range
    for line in git.pipe('log', '--pretty=@@@%an@@@%n@@@%cn@@@%n%b',
                         f'{rev1}'):
        analyze_line(line, all_authors)  # 对指定版本区间前的所有作者进行分析

    # Find authors in the named range
    for line in git.pipe('log', '--pretty=@@@%an@@@%n@@@%cn@@@%n%b',
                         f'{rev1}..{rev2}'):
        analyze_line(line, authors, disp=options.debug)  # 对指定版本区间内的所有作者进行分析，并根据调试模式决定是否输出

    # Sort
    def name_key(fullname):
        # 使用正则表达式从完整姓名中提取姓和名
        m = re.search(' [a-z ]*[A-Za-z-]+$', fullname)
        if m:
            # 如果找到匹配项，提取名和姓
            forename = fullname[:m.start()].strip()  # 名
            surname = fullname[m.start():].strip()  # 姓
        else:
            # 如果没有找到匹配项，名为空，姓为整个字符串
            forename = ""
            surname = fullname.strip()  # 姓
        # 处理姓中的特殊前缀
        if surname.startswith('van der '):
            surname = surname[8:]
        if surname.startswith('de '):
            surname = surname[3:]
        if surname.startswith('von '):
            surname = surname[4:]
        # 返回以姓和名为元组的排序关键字，转换为小写
        return (surname.lower(), forename.lower())

    # 生成所有新作者的集合
    if vars(options)['new']:
        new_authors = set(authors.keys()).difference(all_authors)
        n_authors = list(new_authors)
        # 根据姓名关键字排序新作者列表
        n_authors.sort(key=name_key)
        # 打印空行以分隔输出
        stdout_b.write(b"\n\n")
        # 遍历新作者列表，将每个作者以特定格式写入 stdout_b
        for author in n_authors:
            stdout_b.write(("- %s\n" % author).encode('utf-8'))
        # 提前退出，只打印新作者信息
        return

    try:
        # 尝试从作者字典中移除键为 'GitHub' 的项
        authors.pop('GitHub')
    except KeyError:
        # 如果 'GitHub' 键不存在，捕获 KeyError 异常并继续
        pass

    # 按姓名排序作者列表。也可以按出现次数排序，使用 authors.most_common()
    authors = sorted(authors.items(), key=lambda i: name_key(i[0]))

    # 打印
    stdout_b.write(b"""
    Authors
    =======

    """

    for author, count in authors:
        # 如果作者只有 GitHub 用户名，去除开头的 '@'
        author_clean = author.strip('@')

        # 如果作者在总作者列表中已经存在
        if author in all_authors:
            # 将格式化后的作者名和贡献次数写入标准输出字节流
            stdout_b.write((f"* {author_clean} ({count})\n").encode())
        else:
            # 将格式化后的作者名、贡献次数和加号写入标准输出字节流（表示首次贡献）
            stdout_b.write((f"* {author_clean} ({count}) +\n").encode())

    # 将总贡献人数格式化后写入标准输出字节流
    stdout_b.write(("""
    A total of %(count)d people contributed to this release.
    People with a "+" by their names contributed a patch for the first time.
    This list of names is automatically generated, and may not be fully complete.

    """ % dict(count=len(authors))).encode('utf-8'))

    # 写入注意事项到标准输出字节流
    stdout_b.write(b"\nNOTE: Check this list manually! It is automatically generated "
                   b"and some names\n      may be missing.\n")


def load_name_map(filename):
    # 创建空的姓名映射字典
    name_map = {}

    # 使用 UTF-8 编码打开文件
    with open(filename, encoding='utf-8') as f:
        # 逐行读取文件内容
        for line in f:
            # 去除每行的首尾空白字符
            line = line.strip()
            # 如果行以 '#' 开头或者为空行，则跳过
            if line.startswith("#") or not line:
                continue

            # 使用正则表达式匹配 .mailmap 文件中的每一行
            m = re.match(r'^(.*?)\s*<(.*?)>(.*?)\s*<(.*?)>\s*$', line)
            # 如果匹配失败，则输出错误信息并退出程序
            if not m:
                print(f"Invalid line in .mailmap: '{line!r}'", file=sys.stderr)
                sys.exit(1)

            # 提取新姓名和旧姓名
            new_name = m.group(1).strip()
            old_name = m.group(3).strip()

            # 如果旧姓名和新姓名都存在，则将映射关系添加到字典中
            if old_name and new_name:
                name_map[old_name] = new_name

    # 返回姓名映射字典
    return name_map


#------------------------------------------------------------------------------
# Communicating with Git
#------------------------------------------------------------------------------

class Cmd:
    executable = None

    def __init__(self, executable):
        # 初始化 Cmd 类的 executable 属性
        self.executable = executable

    def _call(self, command, args, kw, repository=None, call=False):
        # 构建命令行参数列表
        cmd = [self.executable, command] + list(args)
        cwd = None

        # 如果指定了 repository，则切换到该目录
        if repository is not None:
            cwd = os.getcwd()
            os.chdir(repository)

        try:
            # 根据 call 参数决定是调用子进程还是返回子进程对象
            if call:
                return subprocess.call(cmd, **kw)
            else:
                return subprocess.Popen(cmd, **kw)
        finally:
            # 无论如何都要恢复当前工作目录
            if cwd is not None:
                os.chdir(cwd)

    def __call__(self, command, *a, **kw):
        # 调用 _call 方法执行命令，并检查返回值是否为 0，否则抛出运行时异常
        ret = self._call(command, a, {}, call=True, **kw)
        if ret != 0:
            raise RuntimeError("%s failed" % self.executable)

    def pipe(self, command, *a, **kw):
        # 执行命令并返回标准输出的管道对象
        stdin = kw.pop('stdin', None)
        p = self._call(command, a, dict(stdin=stdin, stdout=subprocess.PIPE),
                      call=False, **kw)
        return p.stdout

    def read(self, command, *a, **kw):
        # 执行命令并返回标准输出的内容
        p = self._call(command, a, dict(stdout=subprocess.PIPE),
                      call=False, **kw)
        out, err = p.communicate()
        # 如果返回值不为 0，则抛出运行时异常
        if p.returncode != 0:
            raise RuntimeError("%s failed" % self.executable)
        return out

    def readlines(self, command, *a, **kw):
        # 调用 read 方法获取标准输出内容，并按行分割返回结果列表
        out = self.read(command, *a, **kw)
        return out.rstrip("\n").split("\n")
    # 定义一个方法 test，接受命令和可变位置参数 *a 和可变关键字参数 **kw
    def test(self, command, *a, **kw):
        # 调用类中的 _call 方法，传入 command 参数和所有位置参数 a，
        # 并设置标准输出和标准错误都重定向到管道中，call 参数设置为 True
        ret = self._call(command, a, dict(stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE),
                        call=True, **kw)
        # 返回 _call 方法的返回值是否为 0 的布尔结果
        return (ret == 0)
# 导入Cmd类并创建一个名为git的Cmd对象
git = Cmd("git")

#------------------------------------------------------------------------------

# 如果当前脚本作为主程序执行，调用main函数
if __name__ == "__main__":
    main()
```