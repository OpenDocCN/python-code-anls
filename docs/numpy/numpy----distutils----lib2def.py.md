# `.\numpy\numpy\distutils\lib2def.py`

```
import re
import sys
import subprocess
# 导入re模块、sys模块和subprocess模块

__doc__ = """This module generates a DEF file from the symbols in
an MSVC-compiled DLL import library.  It correctly discriminates between
data and functions.  The data is collected from the output of the program
nm(1).

Usage:
    python lib2def.py [libname.lib] [output.def]
or
    python lib2def.py [libname.lib] > output.def

libname.lib defaults to python<py_ver>.lib and output.def defaults to stdout

Author: Robert Kern <kernr@mail.ncifcrf.gov>
Last Update: April 30, 1999
"""
# 模块文档字符串说明

__version__ = '0.1a'
# 模块的版本号

py_ver = "%d%d" % tuple(sys.version_info[:2])
# 获取当前Python版本信息拼接成字符串

DEFAULT_NM = ['nm', '-Cs']
# 默认的nm命令参数列表

DEF_HEADER = """LIBRARY         python%s.dll
;CODE           PRELOAD MOVEABLE DISCARDABLE
;DATA           PRELOAD SINGLE

EXPORTS
""" % py_ver
# DEF文件的头部信息

FUNC_RE = re.compile(r"^(.*) in python%s\.dll" % py_ver, re.MULTILINE)
# 匹配函数的正则表达式

DATA_RE = re.compile(r"^_imp__(.*) in python%s\.dll" % py_ver, re.MULTILINE)
# 匹配数据的正则表达式

def parse_cmd():
    """Parses the command-line arguments.

libfile, deffile = parse_cmd()"""
    # 解析命令行参数的函数说明

    if len(sys.argv) == 3:
        if sys.argv[1][-4:] == '.lib' and sys.argv[2][-4:] == '.def':
            libfile, deffile = sys.argv[1:]
        elif sys.argv[1][-4:] == '.def' and sys.argv[2][-4:] == '.lib':
            deffile, libfile = sys.argv[1:]
        else:
            print("I'm assuming that your first argument is the library")
            print("and the second is the DEF file.")
    elif len(sys.argv) == 2:
        if sys.argv[1][-4:] == '.def':
            deffile = sys.argv[1]
            libfile = 'python%s.lib' % py_ver
        elif sys.argv[1][-4:] == '.lib':
            deffile = None
            libfile = sys.argv[1]
    else:
        libfile = 'python%s.lib' % py_ver
        deffile = None
    return libfile, deffile
# 解析命令行参数的函数

def getnm(nm_cmd=['nm', '-Cs', 'python%s.lib' % py_ver], shell=True):
    """Returns the output of nm_cmd via a pipe.

nm_output = getnm(nm_cmd = 'nm -Cs py_lib')"""
    # 获取nm命令输出的函数说明

    p = subprocess.Popen(nm_cmd, shell=shell, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    nm_output, nm_err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError('failed to run "%s": "%s"' % (
                                     ' '.join(nm_cmd), nm_err))
    return nm_output
# 执行nm命令获取输出的函数

def parse_nm(nm_output):
    """Returns a tuple of lists: dlist for the list of data
symbols and flist for the list of function symbols.

dlist, flist = parse_nm(nm_output)"""
    # 解析nm命令输出内容的函数说明

    data = DATA_RE.findall(nm_output)
    func = FUNC_RE.findall(nm_output)

    flist = []
    for sym in data:
        if sym in func and (sym[:2] == 'Py' or sym[:3] == '_Py' or sym[:4] == 'init'):
            flist.append(sym)

    dlist = []
    for sym in data:
        if sym not in flist and (sym[:2] == 'Py' or sym[:3] == '_Py'):
            dlist.append(sym)

    dlist.sort()
    flist.sort()
    return dlist, flist
# 解析nm命令输出内容的函数

def output_def(dlist, flist, header, file = sys.stdout):
    """将最终的 DEF 文件输出到默认的 stdout 或指定的文件中。
def output_def(dlist, flist, header, file=sys.stdout):
    # 遍历数据符号列表，为每个符号生成一个数据行并添加到标题中
    for data_sym in dlist:
        header = header + '\t%s DATA\n' % data_sym
    # 添加一个空行到标题末尾
    header = header + '\n'
    # 遍历函数符号列表，为每个符号生成一个函数行并添加到标题中
    for func_sym in flist:
        header = header + '\t%s\n' % func_sym
    # 将最终的标题写入指定的文件对象中
    file.write(header)

if __name__ == '__main__':
    # 解析命令行参数，获取库文件和定义文件路径
    libfile, deffile = parse_cmd()
    # 如果定义文件路径为None，则将输出定向到标准输出流
    if deffile is None:
        deffile = sys.stdout
    else:
        # 否则，打开指定路径的文件以便写入
        deffile = open(deffile, 'w')
    # 构造执行 nm 命令的参数列表，并获取 nm 输出结果
    nm_cmd = DEFAULT_NM + [str(libfile)]
    nm_output = getnm(nm_cmd, shell=False)
    # 解析 nm 输出，获取数据符号列表和函数符号列表
    dlist, flist = parse_nm(nm_output)
    # 调用 output_def 函数，生成输出文件的定义部分
    output_def(dlist, flist, DEF_HEADER, deffile)
```