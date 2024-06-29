# `.\numpy\numpy\f2py\crackfortran.py`

```
#!/usr/bin/env python3
"""
crackfortran --- read fortran (77,90) code and extract declaration information.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

# crackfortran 模块用于解析 Fortran (77, 90) 代码并提取声明信息

# 导入所需的模块和库
import os
import re
import sys

# 用于解析命令行参数的正则表达式模式
cmdline_key_pattern = re.compile(r'-[a-zA-Z]+')

# 函数 crackfortran 的定义，用于处理 Fortran 代码
def crackfortran(source_code, options):
    """
    主函数 crackfortran，用于解析 Fortran 代码并提取声明信息。

    Parameters:
    source_code -- Fortran 代码的字符串表示
    options -- 包含各种选项的字典，影响解析过程

    Returns:
    declaration_info -- 提取的声明信息，可能是变量、函数、模块等
    """
    declaration_info = []

    # 用于解析 Fortran 90 类型声明语句的正则表达式模式
    type_decl_pattern = re.compile(r'<typespec>\s+[[<attrspec>]::]\s+<entitydecl>')

    # 匹配 Fortran 代码中的 type declaration 语句，提取类型声明信息
    matches = type_decl_pattern.findall(source_code)

    # 将匹配到的声明信息存储到 declaration_info 列表中
    for match in matches:
        declaration_info.append(match)

    return declaration_info

# 函数 crack2fortran 的定义，用于转换 Fortran 代码到其他格式（未在注释中提到具体功能）
def crack2fortran(source_code, options):
    """
    函数 crack2fortran，用于将 Fortran 代码转换为其他格式（具体格式未在注释中指定）。

    Parameters:
    source_code -- Fortran 代码的字符串表示
    options -- 包含各种选项的字典，影响转换过程

    Returns:
    transformed_code -- 转换后的代码，格式未在注释中指定
    """
    transformed_code = ''

    # 未提供具体转换实现，只返回空字符串
    return transformed_code

# 如果作为独立程序运行，则执行以下代码
if __name__ == '__main__':
    # 解析命令行参数，使用 getopt 或 argparse 更好，但示例中使用简单的方法
    cmdline_args = sys.argv[1:]

    # 使用正则表达式匹配命令行参数
    keys = cmdline_key_pattern.findall(' '.join(cmdline_args))

    # 打印匹配到的命令行参数
    print(f"Command line keys found: {keys}")
    * Apply 'parameter' attribute (e.g. 'integer parameter :: i=2' 'real x(i)'
                                   -> 'real x(2)')
    The above may be solved by creating appropriate preprocessor program, for example.
"""
import sys                      # 导入sys模块，用于系统相关操作
import string                   # 导入string模块，提供字符串相关操作
import fileinput                # 导入fileinput模块，用于迭代处理多个输入流
import re                       # 导入re模块，提供正则表达式操作
import os                       # 导入os模块，提供操作系统接口
import copy                     # 导入copy模块，用于复制对象
import platform                 # 导入platform模块，提供访问平台相关信息
import codecs                   # 导入codecs模块，提供编解码器注册和相关工具函数
from pathlib import Path        # 从pathlib模块导入Path类，提供面向对象的文件系统路径操作

try:
    import charset_normalizer   # 尝试导入charset_normalizer模块，用于字符集规范化
except ImportError:
    charset_normalizer = None   # 如果导入失败，设为None

from . import __version__       # 从当前包中导入__version__模块

# 从auxfuncs.py中导入所有函数和变量，以便在eval调用时使用必要的环境
from .auxfuncs import *

from . import symbolic          # 从当前包中导入symbolic模块

f2py_version = __version__.version  # 设置f2py_version为当前版本号

# Global flags: 全局标志变量
strictf77 = 1                   # 设置strictf77为1，忽略`!`开头的注释，除非行首为`!`
sourcecodeform = 'fix'          # 设置sourcecodeform为'fix'，表示源代码格式为固定格式
quiet = 0                       # 设置quiet为0，如果为0，则输出详细信息（已废弃）
verbose = 1                     # 设置verbose为1，如果为0则静默，大于1则输出额外详细信息
tabchar = 4 * ' '               # 设置tabchar为四个空格的字符串
pyffilename = ''                # 初始化pyffilename为空字符串
f77modulename = ''              # 初始化f77modulename为空字符串
skipemptyends = 0               # 对于老旧的F77程序，没有'program'语句时设置为0
ignorecontains = 1              # 设置ignorecontains为1，忽略contains语句
dolowercase = 1                 # 设置dolowercase为1，将所有标识符转换为小写
debug = []                      # 初始化debug为空列表

# Global variables: 全局变量
beginpattern = ''               # 初始化beginpattern为空字符串
currentfilename = ''            # 初始化currentfilename为空字符串
expectbegin = 1                 # 设置expectbegin为1，期望开始新的程序单元
f90modulevars = {}              # 初始化f90modulevars为空字典
filepositiontext = ''           # 初始化filepositiontext为空字符串
gotnextfile = 1                 # 初始化gotnextfile为1
groupcache = None               # 初始化groupcache为None
groupcounter = 0                # 初始化groupcounter为0
grouplist = {groupcounter: []}  # 初始化grouplist为包含空列表的字典，键为groupcounter
groupname = ''                  # 初始化groupname为空字符串
include_paths = []              # 初始化include_paths为空列表
neededmodule = -1               # 初始化neededmodule为-1
onlyfuncs = []                  # 初始化onlyfuncs为空列表
previous_context = None         # 初始化previous_context为None
skipblocksuntil = -1            # 初始化skipblocksuntil为-1
skipfuncs = []                  # 初始化skipfuncs为空列表
skipfunctions = []              # 初始化skipfunctions为空列表
usermodules = []                # 初始化usermodules为空列表


def reset_global_f2py_vars():
    global groupcounter, grouplist, neededmodule, expectbegin
    global skipblocksuntil, usermodules, f90modulevars, gotnextfile
    global filepositiontext, currentfilename, skipfunctions, skipfuncs
    global onlyfuncs, include_paths, previous_context
    global strictf77, sourcecodeform, quiet, verbose, tabchar, pyffilename
    global f77modulename, skipemptyends, ignorecontains, dolowercase, debug

    # 重置全局变量和标志
    strictf77 = 1
    sourcecodeform = 'fix'
    quiet = 0
    verbose = 1
    tabchar = 4 * ' '
    pyffilename = ''
    f77modulename = ''
    skipemptyends = 0
    ignorecontains = 1
    dolowercase = 1
    debug = []
    
    # 重置全局变量
    groupcounter = 0
    grouplist = {groupcounter: []}
    neededmodule = -1
    expectbegin = 1
    skipblocksuntil = -1
    usermodules = []
    f90modulevars = {}
    gotnextfile = 1
    filepositiontext = ''
    currentfilename = ''
    skipfunctions = []
    skipfuncs = []
    onlyfuncs = []
    include_paths = []
    previous_context = None


def outmess(line, flag=1):
    global filepositiontext

    # 如果不是verbose模式，直接返回
    if not verbose:
        return
    # 如果不是quiet模式，输出filepositiontext和line
    if not quiet:
        if flag:
            sys.stdout.write(filepositiontext)
        sys.stdout.write(line)

re._MAXCACHE = 50  # 设置re模块的_MAXCACHE为50，用于正则表达式缓存管理

defaultimplicitrules = {}
# 为字符集"abcdefghopqrstuvwxyz$_"设置默认的隐式规则为'real'
for c in "abcdefghopqrstuvwxyz$_":
    defaultimplicitrules[c] = {'typespec': 'real'}
# 为字符集"ijklmn"设置默认的隐式规则为'integer'
for c in "ijklmn":
    defaultimplicitrules[c] = {'typespec': 'integer'}

badnames = {}   # 初始化badnames为空字典
invbadnames = {}  # 初始化invbadnames为空字典
# 遍历包含关键字的列表，初始化badnames和invbadnames字典
for n in ['int', 'double', 'float', 'char', 'short', 'long', 'void', 'case', 'while',
          'return', 'signed', 'unsigned', 'if', 'for', 'typedef', 'sizeof', 'union',
          'struct', 'static', 'register', 'new', 'break', 'do', 'goto', 'switch',
          'continue', 'else', 'inline', 'extern', 'delete', 'const', 'auto',
          'len', 'rank', 'shape', 'index', 'slen', 'size', '_i',
          'max', 'min',
          'flen', 'fshape',
          'string', 'complex_double', 'float_double', 'stdin', 'stderr', 'stdout',
          'type', 'default']:
    # 将关键字和其加上'_bn'后的形式加入badnames字典
    badnames[n] = n + '_bn'
    # 将加上'_bn'后的形式和原关键字加入invbadnames字典，作为反向映射
    invbadnames[n + '_bn'] = n


def rmbadname1(name):
    # 如果name在badnames字典中
    if name in badnames:
        # 打印错误消息，提示替换name为badnames[name]
        errmess('rmbadname1: Replacing "%s" with "%s".\n' %
                (name, badnames[name]))
        # 返回替换后的名称
        return badnames[name]
    # 否则直接返回原名称
    return name


def rmbadname(names):
    # 对于names列表中的每个名称，应用rmbadname1函数进行处理
    return [rmbadname1(_m) for _m in names]


def undo_rmbadname1(name):
    # 如果name在invbadnames字典中
    if name in invbadnames:
        # 打印错误消息，提示替换name为invbadnames[name]
        errmess('undo_rmbadname1: Replacing "%s" with "%s".\n'
                % (name, invbadnames[name]))
        # 返回替换后的名称
        return invbadnames[name]
    # 否则直接返回原名称
    return name


def undo_rmbadname(names):
    # 对于names列表中的每个名称，应用undo_rmbadname1函数进行处理
    return [undo_rmbadname1(_m) for _m in names]


# 正则表达式，用于检测文件头部是否包含特定的Fortran格式标记
_has_f_header = re.compile(r'-\*-\s*fortran\s*-\*-', re.I).search
# 正则表达式，用于检测文件头部是否包含特定的F90格式标记
_has_f90_header = re.compile(r'-\*-\s*f90\s*-\*-', re.I).search
# 正则表达式，用于检测文件头部是否包含特定的Fixed格式标记
_has_fix_header = re.compile(r'-\*-\s*fix\s*-\*-', re.I).search
# 正则表达式，用于检测Free Format Fortran文件的起始位置
_free_f90_start = re.compile(r'[^c*]\s*[^\s\d\t]', re.I).match

# 常见的Free Format Fortran文件扩展名列表
COMMON_FREE_EXTENSIONS = ['.f90', '.f95', '.f03', '.f08']
# 常见的Fixed Format Fortran文件扩展名列表
COMMON_FIXED_EXTENSIONS = ['.for', '.ftn', '.f77', '.f']


def openhook(filename, mode):
    """确保使用正确的编码参数打开文件名。

    当可用时，此函数使用charset_normalizer包确定要打开的文件的编码。
    当charset_normalizer不可用时，函数仅检测UTF编码，否则使用ASCII作为后备编码。
    """
    # 读取整个文件。对编码进行鲁棒的检测。
    # 正确处理注释或晚期Unicode字符
    # gh-22871
    if charset_normalizer is not None:
        # 使用charset_normalizer确定最佳编码
        encoding = charset_normalizer.from_path(filename).best().encoding
    else:
        # 提示：安装charset_normalizer以进行正确的编码处理
        # 不需要读取整个文件来尝试startswith
        nbytes = min(32, os.path.getsize(filename))
        with open(filename, 'rb') as fhandle:
            raw = fhandle.read(nbytes)
            if raw.startswith(codecs.BOM_UTF8):
                encoding = 'UTF-8-SIG'
            elif raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
                encoding = 'UTF-32'
            elif raw.startswith((codecs.BOM_LE, codecs.BOM_BE)):
                encoding = 'UTF-16'
            else:
                # 无charset_normalizer时的后备，使用ASCII编码
                encoding = 'ascii'
    # 使用确定的编码参数打开文件
    return open(filename, mode, encoding=encoding)


def is_free_format(fname):
    """Check if file is in free format Fortran."""
    # 检查文件是否为自由格式Fortran

    result = False
    # 初始化结果为False，表示文件不是自由格式Fortran

    if Path(fname).suffix.lower() in COMMON_FREE_EXTENSIONS:
        # 如果文件的后缀在常见的自由格式Fortran文件后缀列表中
        result = True
        # 将结果设置为True，表示文件可能是自由格式Fortran

    with openhook(fname, 'r') as fhandle:
        # 使用openhook函数打开文件，并命名为fhandle，以只读方式打开

        line = fhandle.readline()
        # 读取文件的第一行

        n = 15  # the number of non-comment lines to scan for hints
        # 扫描文件以获取提示的非注释行数的初始值为15

        if _has_f_header(line):
            # 如果第一行具有Fortran标头
            n = 0
            # 直接将n设置为0，不需要继续扫描行数了
        elif _has_f90_header(line):
            # 如果第一行具有Fortran 90标头
            n = 0
            # 直接将n设置为0，不需要继续扫描行数了
            result = True
            # 将结果设置为True，表示文件是自由格式Fortran

        while n > 0 and line:
            # 当仍然需要扫描行数大于0并且还有行可读时，执行以下操作

            if line[0] != '!' and line.strip():
                # 如果行不以'!'开头且去除空白字符后不为空

                n -= 1
                # 扫描行数减1

                if (line[0] != '\t' and _free_f90_start(line[:5])) or line[-2:-1] == '&':
                    # 如果行不以制表符开头且前5个字符符合Fortran 90自由格式的起始条件
                    # 或者行的倒数第二到倒数第一字符是'&'

                    result = True
                    # 将结果设置为True，表示文件是自由格式Fortran
                    break
                    # 跳出循环

            line = fhandle.readline()
            # 继续读取下一行

    return result
    # 返回结果，指示文件是否为自由格式Fortran
# Read fortran (77,90) code
def readfortrancode(ffile, dowithline=show, istop=1):
    """
    Read fortran codes from files and
     1) Get rid of comments, line continuations, and empty lines; lower cases.
     2) Call dowithline(line) on every line.
     3) Recursively call itself when statement "include '<filename>'" is met.
    """
    # 声明全局变量，以便在函数内部使用
    global gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77
    global beginpattern, quiet, verbose, dolowercase, include_paths

    # 如果不是顶层调用，保存全局变量的当前状态
    if not istop:
        saveglobals = gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77,\
            beginpattern, quiet, verbose, dolowercase

    # 如果 ffile 是空列表，则直接返回
    if ffile == []:
        return

    # 根据全局变量决定是否将代码转换为小写
    localdolowercase = dolowercase

    # cont：当读取的最后一行内容表示语句继续时设置为 True
    cont = False
    finalline = ''
    ll = ''

    # 正则表达式模式：用于匹配包含文件的语句
    includeline = re.compile(
        r'\s*include\s*(\'|")(?P<name>[^\'"]*)(\'|")', re.I)
    cont1 = re.compile(r'(?P<line>.*)&\s*\Z')
    cont2 = re.compile(r'(\s*&|)(?P<line>.*)')
    mline_mark = re.compile(r".*?'''")

    # 如果是顶层调用，则调用 dowithline 处理空行
    if istop:
        dowithline('', -1)

    ll, l1 = '', ''

    # 文件位置文本初始化为空
    spacedigits = [' '] + [str(_m) for _m in range(10)]
    filepositiontext = ''

    # 使用 fileinput.FileInput 打开文件
    fin = fileinput.FileInput(ffile, openhook=openhook)

    # 根据 dolowercase 决定是否将行内容转换为小写
    if localdolowercase:
        finalline = ll.lower()
    else:
        finalline = ll
    origfinalline = ll

    # 设置文件位置文本，指示当前文件和行号
    filepositiontext = 'Line #%d in %s:"%s"\n\t' % (
        fin.filelineno() - 1, currentfilename, l1)

    # 如果 origfinalline 匹配 includeline 的模式
    m = includeline.match(origfinalline)
    if m:
        fn = m.group('name')

        # 如果文件存在，则递归调用 readfortrancode 处理 include 文件
        if os.path.isfile(fn):
            readfortrancode(fn, dowithline=dowithline, istop=0)
        else:
            # 否则在 include_paths 中寻找文件
            include_dirs = [os.path.dirname(currentfilename)] + include_paths
            foundfile = 0
            for inc_dir in include_dirs:
                fn1 = os.path.join(inc_dir, fn)
                if os.path.isfile(fn1):
                    foundfile = 1
                    readfortrancode(fn1, dowithline=dowithline, istop=0)
                    break
            if not foundfile:
                # 如果找不到文件，则输出警告信息
                outmess('readfortrancode: could not find include file %s in %s. Ignoring.\n' % (
                    repr(fn), os.pathsep.join(include_dirs)))
    else:
        # 否则调用 dowithline 处理当前行内容
        dowithline(finalline)

    # 清空文件位置文本
    filepositiontext = ''

    # 关闭文件输入流
    fin.close()

    # 如果是顶层调用，调用 dowithline 处理结束标志
    if istop:
        dowithline('', 1)
    else:
        # 否则恢复保存的全局变量状态
        gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77,\
            beginpattern, quiet, verbose, dolowercase = saveglobals
    '', fortrantypes + '|static|automatic|undefined', fortrantypes + '|static|automatic|undefined', '.*'), re.I)
# 定义一个正则表达式对象，用于匹配函数定义的模式
functionpattern = re.compile(beforethisafter % (
    r'([a-z]+[\w\s(=*+-/)]*?|)', 'function', 'function', '.*'), re.I), 'begin'

# 定义一个正则表达式对象，用于匹配子程序定义的模式
subroutinepattern = re.compile(beforethisafter % (
    r'[a-z\s]*?', 'subroutine', 'subroutine', '.*'), re.I), 'begin'

# 定义一个正则表达式对象，暂时注释掉，未使用
# modulepattern=re.compile(beforethisafter%('[a-z\s]*?','module','module','.*'),re.I),'begin'

# 定义一个正则表达式对象，用于匹配Fortran 77风格的程序或数据块起始的模式
groupbegins77 = r'program|block\s*data'
beginpattern77 = re.compile(
    beforethisafter % ('', groupbegins77, groupbegins77, '.*'), re.I), 'begin'

# 定义一个正则表达式对象，用于匹配Fortran 90及其后的程序或数据块起始的模式
groupbegins90 = groupbegins77 + \
    r'|module(?!\s*procedure)|python\s*module|(abstract|)\s*interface|' + \
    r'type(?!\s*\()'
beginpattern90 = re.compile(
    beforethisafter % ('', groupbegins90, groupbegins90, '.*'), re.I), 'begin'

# 定义一个正则表达式对象，用于匹配程序或数据块结束的模式
groupends = (r'end|endprogram|endblockdata|endmodule|endpythonmodule|'
             r'endinterface|endsubroutine|endfunction')
endpattern = re.compile(
    beforethisafter % ('', groupends, groupends, '.*'), re.I), 'end'

# 定义一个正则表达式对象，用于匹配Fortran 2008中特殊块（如if、do等）结束的模式
endifs = r'end\s*(if|do|where|select|while|forall|associate|' + \
         r'critical|enum|team)'
endifpattern = re.compile(
    beforethisafter % (r'[\w]*?', endifs, endifs, '.*'), re.I), 'endif'

# 定义一个正则表达式对象，用于匹配模块中的过程定义的模式
moduleprocedures = r'module\s*procedure'
moduleprocedurepattern = re.compile(
    beforethisafter % ('', moduleprocedures, moduleprocedures, '.*'), re.I), \
    'moduleprocedure'

# 定义一个正则表达式对象，用于匹配implicit声明的模式
implicitpattern = re.compile(
    beforethisafter % ('', 'implicit', 'implicit', '.*'), re.I), 'implicit'

# 定义一个正则表达式对象，用于匹配数组维度声明的模式
dimensionpattern = re.compile(beforethisafter % (
    '', 'dimension|virtual', 'dimension|virtual', '.*'), re.I), 'dimension'

# 定义一个正则表达式对象，用于匹配外部变量声明的模式
externalpattern = re.compile(
    beforethisafter % ('', 'external', 'external', '.*'), re.I), 'external'

# 定义一个正则表达式对象，用于匹配可选参数声明的模式
optionalpattern = re.compile(
    beforethisafter % ('', 'optional', 'optional', '.*'), re.I), 'optional'

# 定义一个正则表达式对象，用于匹配必须参数声明的模式
requiredpattern = re.compile(
    beforethisafter % ('', 'required', 'required', '.*'), re.I), 'required'

# 定义一个正则表达式对象，用于匹配公共变量声明的模式
publicpattern = re.compile(
    beforethisafter % ('', 'public', 'public', '.*'), re.I), 'public'

# 定义一个正则表达式对象，用于匹配私有变量声明的模式
privatepattern = re.compile(
    beforethisafter % ('', 'private', 'private', '.*'), re.I), 'private'

# 定义一个正则表达式对象，用于匹配内置函数声明的模式
intrinsicpattern = re.compile(
    beforethisafter % ('', 'intrinsic', 'intrinsic', '.*'), re.I), 'intrinsic'

# 定义一个正则表达式对象，用于匹配变量意图声明的模式
intentpattern = re.compile(beforethisafter % (
    '', 'intent|depend|note|check', 'intent|depend|note|check', r'\s*\(.*?\).*'), re.I), 'intent'

# 定义一个正则表达式对象，用于匹配参数声明的模式
parameterpattern = re.compile(
    beforethisafter % ('', 'parameter', 'parameter', r'\s*\(.*'), re.I), 'parameter'

# 定义一个正则表达式对象，用于匹配数据声明的模式
datapattern = re.compile(
    beforethisafter % ('', 'data', 'data', '.*'), re.I), 'data'

# 定义一个正则表达式对象，用于匹配调用语句的模式
callpattern = re.compile(
    beforethisafter % ('', 'call', 'call', '.*'), re.I), 'call'

# 定义一个正则表达式对象，用于匹配entry语句的模式
entrypattern = re.compile(
    beforethisafter % ('', 'entry', 'entry', '.*'), re.I), 'entry'

# 定义一个正则表达式对象，用于匹配调用函数的模式
callfunpattern = re.compile(
    beforethisafter % ('', 'callfun', 'callfun', '.*'), re.I), 'callfun'

# 定义一个正则表达式对象，用于匹配common声明的模式
commonpattern = re.compile(
    # 使用正则表达式替换字符串中的部分内容
    beforethisafter % ('', 'common', 'common', '.*'), re.I), 'common'
usepattern = re.compile(
    beforethisafter % ('', 'use', 'use', '.*'), re.I), 'use'
# 编译正则表达式，用于匹配包含 'use' 的语句，不区分大小写

containspattern = re.compile(
    beforethisafter % ('', 'contains', 'contains', ''), re.I), 'contains'
# 编译正则表达式，用于匹配包含 'contains' 的语句，不区分大小写

formatpattern = re.compile(
    beforethisafter % ('', 'format', 'format', '.*'), re.I), 'format'
# 编译正则表达式，用于匹配包含 'format' 的语句，不区分大小写

f2pyenhancementspattern = re.compile(beforethisafter % ('', 'threadsafe|fortranname|callstatement|callprotoargument|usercode|pymethoddef',
                                                        'threadsafe|fortranname|callstatement|callprotoargument|usercode|pymethoddef', '.*'), re.I | re.S), 'f2pyenhancements'
# 编译正则表达式，用于匹配包含 'threadsafe', 'fortranname', 'callstatement', 'callprotoargument', 'usercode', 'pymethoddef' 等关键词的语句，不区分大小写和多行模式

multilinepattern = re.compile(
    r"\s*(?P<before>''')(?P<this>.*?)(?P<after>''')\s*\Z", re.S), 'multiline'
# 编译正则表达式，用于匹配三引号之间的多行字符串

def split_by_unquoted(line, characters):
    """
    Splits the line into (line[:i], line[i:]),
    where i is the index of first occurrence of one of the characters
    not within quotes, or len(line) if no such index exists
    """
    assert not (set('"\'') & set(characters)), "cannot split by unquoted quotes"
    # 断言确保字符集中没有单引号或双引号，因为函数无法处理带引号的字符串

    r = re.compile(
        r"\A(?P<before>({single_quoted}|{double_quoted}|{not_quoted})*)"
        r"(?P<after>{char}.*)\Z".format(
            not_quoted="[^\"'{}]".format(re.escape(characters)),
            char="[{}]".format(re.escape(characters)),
            single_quoted=r"('([^'\\]|(\\.))*')",
            double_quoted=r'("([^"\\]|(\\.))*")'))
    # 编译正则表达式，用于将字符串分割为未在引号内的部分和之后的部分

    m = r.match(line)
    if m:
        d = m.groupdict()
        return (d["before"], d["after"])
    return (line, "")
    # 如果匹配成功，返回分割后的结果；否则返回原始字符串和空字符串

def _simplifyargs(argsline):
    a = []
    for n in markoutercomma(argsline).split('@,@'):
        for r in '(),':
            n = n.replace(r, '_')
        a.append(n)
    return ','.join(a)
# 简化函数参数列表的格式，将标记处的逗号替换为下划线，并返回处理后的字符串

crackline_re_1 = re.compile(r'\s*(?P<result>\b[a-z]+\w*\b)\s*=.*', re.I)
# 编译正则表达式，用于匹配形如 "result = ..." 的语句，不区分大小写

crackline_bind_1 = re.compile(r'\s*(?P<bind>\b[a-z]+\w*\b)\s*=.*', re.I)
# 编译正则表达式，用于匹配形如 "bind = ..." 的语句，不区分大小写

crackline_bindlang = re.compile(r'\s*bind\(\s*(?P<lang>[^,]+)\s*,\s*name\s*=\s*"(?P<lang_name>[^"]+)"\s*\)', re.I)
# 编译正则表达式，用于匹配 bind() 函数调用的语句，匹配其中的语言和语言名称，不区分大小写

def crackline(line, reset=0):
    """
    reset=-1  --- initialize
    reset=0   --- crack the line
    reset=1   --- final check if mismatch of blocks occurred

    Cracked data is saved in grouplist[0].
    """
    global beginpattern, groupcounter, groupname, groupcache, grouplist
    global filepositiontext, currentfilename, neededmodule, expectbegin
    global skipblocksuntil, skipemptyends, previous_context, gotnextfile

    _, has_semicolon = split_by_unquoted(line, ";")
    # 分割行，以找到第一个不在引号内的分号的位置
    # 如果代码行包含分号且不匹配特定的两种模式，则执行以下操作
    if has_semicolon and not (f2pyenhancementspattern[0].match(line) or
                               multilinepattern[0].match(line)):
        # 断言重置值为零，用于调试非零重置值的情况
        assert reset == 0, repr(reset)
        # 在未引号包围的分号上分割行
        line, semicolon_line = split_by_unquoted(line, ";")
        # 当存在分号行时循环执行破解行函数
        while semicolon_line:
            crackline(line, reset)
            line, semicolon_line = split_by_unquoted(semicolon_line[1:], ";")
        # 处理最后一个分号之后的行
        crackline(line, reset)
        return
    
    # 如果重置值小于零，则初始化各种组的数据结构并返回
    if reset < 0:
        groupcounter = 0
        groupname = {groupcounter: ''}
        groupcache = {groupcounter: {}}
        grouplist = {groupcounter: []}
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['block'] = ''
        groupcache[groupcounter]['name'] = ''
        neededmodule = -1
        skipblocksuntil = -1
        return
    
    # 如果重置值大于零，则执行以下操作
    if reset > 0:
        fl = 0
        # 如果存在 f77 模块名且需要模块等于组计数器，则设置 fl 为 2
        if f77modulename and neededmodule == groupcounter:
            fl = 2
        # 循环直到组计数器减少到 fl
        while groupcounter > fl:
            # 输出消息指示组计数器和组名
            outmess('crackline: groupcounter=%s groupname=%s\n' %
                    (repr(groupcounter), repr(groupname)))
            outmess(
                'crackline: Mismatch of blocks encountered. Trying to fix it by assuming "end" statement.\n')
            # 将当前组缓存添加到前一个组列表中，并设置其 body 为当前组列表
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            # 删除当前组缓存
            del grouplist[groupcounter]
            # 减少组计数器
            groupcounter = groupcounter - 1
        # 如果存在 f77 模块名且需要模块等于组计数器，则继续处理
        if f77modulename and neededmodule == groupcounter:
            # 将当前组缓存添加到前一个组列表中，并设置其 body 为当前组列表
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            # 删除当前组缓存
            del grouplist[groupcounter]
            # 减少组计数器（结束接口）
            groupcounter = groupcounter - 1
            # 将当前组缓存添加到前一个组列表中，并设置其 body 为当前组列表
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            # 删除当前组缓存（结束模块）
            del grouplist[groupcounter]
            # 减少组计数器
            groupcounter = groupcounter - 1
            # 重置需要的模块标志
            neededmodule = -1
        return
    
    # 如果行为空，则返回
    if line == '':
        return
    
    # 初始化标志为零
    flag = 0
    # 对于每个模式列表中的模式进行循环
    for pat in [dimensionpattern, externalpattern, intentpattern, optionalpattern,
                requiredpattern,
                parameterpattern, datapattern, publicpattern, privatepattern,
                intrinsicpattern,
                endifpattern, endpattern,
                formatpattern,
                beginpattern, functionpattern, subroutinepattern,
                implicitpattern, typespattern, commonpattern,
                callpattern, usepattern, containspattern,
                entrypattern,
                f2pyenhancementspattern,
                multilinepattern,
                moduleprocedurepattern
                ]:
        # 尝试使用当前模式匹配行
        m = pat[0].match(line)
        if m:
            break
        # 如果未匹配成功，增加标志计数器
        flag = flag + 1
    # 如果没有匹配到模式（pat为假值），则使用默认的crackline_re_1正则表达式模式
    if not m:
        re_1 = crackline_re_1
        # 如果skipblocksuntil大于等于0且小于等于groupcounter，表示需要跳过一定数量的块
        if 0 <= skipblocksuntil <= groupcounter:
            return
        # 如果当前组在groupcache中有'externals'键
        if 'externals' in groupcache[groupcounter]:
            # 遍历当前组的'externals'中的每个名称
            for name in groupcache[groupcounter]['externals']:
                # 如果名称在invbadnames字典中，将其替换为invbadnames中对应的值
                if name in invbadnames:
                    name = invbadnames[name]
                # 如果当前组中有'interfaced'键，并且名称在'interfaced'中，继续下一个循环
                if 'interfaced' in groupcache[groupcounter] and name in groupcache[groupcounter]['interfaced']:
                    continue
                # 使用正则表达式匹配名称在line中的位置，并提取相关信息
                m1 = re.match(
                    r'(?P<before>[^"]*)\b%s\b\s*@\(@(?P<args>[^@]*)@\)@.*\Z' % name, markouterparen(line), re.I)
                if m1:
                    # 使用crackline_re_1正则表达式匹配m1中的'before'部分
                    m2 = re_1.match(m1.group('before'))
                    # 简化参数列表
                    a = _simplifyargs(m1.group('args'))
                    if m2:
                        # 构造函数调用的描述信息
                        line = 'callfun %s(%s) result (%s)' % (
                            name, a, m2.group('result'))
                    else:
                        # 构造函数调用的描述信息（无结果）
                        line = 'callfun %s(%s)' % (name, a)
                    # 再次尝试匹配函数调用的模式
                    m = callfunpattern[0].match(line)
                    # 如果没有匹配成功，输出错误信息并返回
                    if not m:
                        outmess(
                            'crackline: could not resolve function call for line=%s.\n' % repr(line))
                        return
                    # 分析处理该函数调用行
                    analyzeline(m, 'callfun', line)
                    return
        # 如果verbose大于1，或者verbose等于1且当前文件名以'.pyf'结尾，输出无法匹配模式的错误信息
        if verbose > 1 or (verbose == 1 and currentfilename.lower().endswith('.pyf')):
            previous_context = None
            outmess('crackline:%d: No pattern for line\n' % (groupcounter))
        return
    # 如果pat的第二个元素为'end'
    elif pat[1] == 'end':
        # 处理结束块的情况
        if 0 <= skipblocksuntil < groupcounter:
            # 更新groupcounter为当前groupcounter减一
            groupcounter = groupcounter - 1
            # 如果skipblocksuntil小于等于当前groupcounter，返回
            if skipblocksuntil <= groupcounter:
                return
        # 如果groupcounter小于等于0，抛出异常
        if groupcounter <= 0:
            raise Exception('crackline: groupcounter(=%s) is nonpositive. '
                            'Check the blocks.'
                            % (groupcounter))
        # 使用beginpattern正则表达式匹配line，并检查匹配结果是否符合预期
        m1 = beginpattern[0].match((line))
        if (m1) and (not m1.group('this') == groupname[groupcounter]):
            # 如果匹配不符合预期，抛出异常
            raise Exception('crackline: End group %s does not match with '
                            'previous Begin group %s\n\t%s' %
                            (repr(m1.group('this')), repr(groupname[groupcounter]),
                             filepositiontext)
                            )
        # 如果skipblocksuntil等于当前groupcounter，将skipblocksuntil置为-1
        if skipblocksuntil == groupcounter:
            skipblocksuntil = -1
        # 将当前组添加到上一组的'body'中，并更新grouplist和groupcounter
        grouplist[groupcounter - 1].append(groupcache[groupcounter])
        grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
        del grouplist[groupcounter]
        groupcounter = groupcounter - 1
        # 如果不跳过空的结束块，设置expectbegin为1
        if not skipemptyends:
            expectbegin = 1
    # 如果pat的第二个元素为'begin'
    elif pat[1] == 'begin':
        # 处理开始块的情况
        if 0 <= skipblocksuntil <= groupcounter:
            # 更新groupcounter为当前groupcounter加一
            groupcounter = groupcounter + 1
            return
        # 分析处理当前行的内容，并设置expectbegin为0
        gotnextfile = 0
        analyzeline(m, pat[1], line)
        expectbegin = 0
    # 如果pat的第二个元素为'endif'，则什么也不做
    elif pat[1] == 'endif':
        pass
    # 如果匹配模式的第二个元素是 'moduleprocedure'，则调用 analyzeline 函数处理该行
    elif pat[1] == 'moduleprocedure':
        analyzeline(m, pat[1], line)
    # 如果匹配模式的第二个元素是 'contains'，则执行以下操作
    elif pat[1] == 'contains':
        # 如果 ignorecontains 为真，则直接返回
        if ignorecontains:
            return
        # 如果 skipblocksuntil 大于等于 0 且小于等于当前组计数器的值，则直接返回
        if 0 <= skipblocksuntil <= groupcounter:
            return
        # 将 skipblocksuntil 设置为当前组计数器的值，表示跳过后续代码块
        skipblocksuntil = groupcounter
    # 如果以上条件都不满足，则执行以下操作
    else:
        # 如果 skipblocksuntil 大于等于 0 且小于等于当前组计数器的值，则直接返回
        if 0 <= skipblocksuntil <= groupcounter:
            return
        # 调用 analyzeline 函数处理该行，使用匹配模式的第二个元素作为参数
        analyzeline(m, pat[1], line)
# 将输入的字符串 line 中最外层的括号替换为特定标记，返回修改后的字符串
def markouterparen(line):
    l = ''  # 初始化一个空字符串 l 用于存储处理后的结果
    f = 0  # 初始化计数器 f，用于跟踪当前在处理哪一层括号
    for c in line:  # 遍历输入字符串 line 中的每一个字符 c
        if c == '(':  # 如果字符 c 是左括号 (
            f = f + 1  # 计数器加一，表示进入了更深一层的括号
            if f == 1:  # 如果 f 变为 1，表示此时处理的是最外层的括号
                l = l + '@(@'  # 在结果字符串 l 中添加特定标记 '@(@'
                continue  # 继续处理下一个字符
        elif c == ')':  # 如果字符 c 是右括号 )
            f = f - 1  # 计数器减一，表示跳出了一层括号
            if f == 0:  # 如果 f 变回 0，表示此时处理的是最外层的右括号
                l = l + '@)@'  # 在结果字符串 l 中添加特定标记 '@)@'
                continue  # 继续处理下一个字符
        l = l + c  # 将当前字符 c 加入结果字符串 l 中
    return l  # 返回处理后的字符串


# 将输入的字符串 line 中最外层的逗号替换为特定标记，返回修改后的字符串
def markoutercomma(line, comma=','):
    l = ''  # 初始化一个空字符串 l 用于存储处理后的结果
    f = 0  # 初始化计数器 f，用于跟踪当前在处理哪一层括号
    before, after = split_by_unquoted(line, comma + '()')  # 使用 split_by_unquoted 函数分割字符串 line
    l += before  # 将分割后的第一部分添加到结果字符串 l 中
    while after:  # 循环处理剩余部分 after，直到处理完毕
        if (after[0] == comma) and (f == 0):  # 如果当前字符是逗号且不在括号内
            l += '@' + comma + '@'  # 在结果字符串 l 中添加特定标记 '@,@'
        else:
            l += after[0]  # 将当前字符加入结果字符串 l 中
            if after[0] == '(':  # 如果当前字符是左括号 (
                f += 1  # 计数器加一，表示进入了更深一层的括号
            elif after[0] == ')':  # 如果当前字符是右括号 )
                f -= 1  # 计数器减一，表示跳出了一层括号
        before, after = split_by_unquoted(after[1:], comma + '()')  # 继续使用 split_by_unquoted 函数分割剩余部分
        l += before  # 将分割后的部分添加到结果字符串 l 中
    assert not f, repr((f, line, l))  # 断言计数器 f 必须为零，否则抛出异常
    return l  # 返回处理后的字符串


# 将输入的字符串 line 中的特定标记 @(@ 和 @)@ 替换为括号 ( 和 )，返回修改后的字符串
def unmarkouterparen(line):
    r = line.replace('@(@', '(').replace('@)@', ')')  # 使用 replace 函数替换特定标记为括号
    return r  # 返回替换后的字符串


# 将两个声明字典 decl 和 decl2 合并，并返回合并后的结果字典
def appenddecl(decl, decl2, force=1):
    if not decl:  # 如果 decl 为空
        decl = {}  # 初始化 decl 为一个空字典
    if not decl2:  # 如果 decl2 为空
        return decl  # 直接返回 decl 字典
    if decl is decl2:  # 如果 decl 和 decl2 是同一个对象
        return decl  # 直接返回 decl 字典
    for k in list(decl2.keys()):  # 遍历 decl2 字典的所有键
        if k == 'typespec':  # 如果键是 'typespec'
            if force or k not in decl:  # 如果强制更新或者 decl 中不存在该键
                decl[k] = decl2[k]  # 将 decl2 中的 'typespec' 复制到 decl 中
        elif k == 'attrspec':  # 如果键是 'attrspec'
            for l in decl2[k]:  # 遍历 decl2 中 'attrspec' 对应的列表
                decl = setattrspec(decl, l, force)  # 调用 setattrspec 函数处理每一项
        elif k == 'kindselector':  # 如果键是 'kindselector'
            decl = setkindselector(decl, decl2[k], force)  # 调用 setkindselector 函数处理
        elif k == 'charselector':  # 如果键是 'charselector'
            decl = setcharselector(decl, decl2[k], force)  # 调用 setcharselector 函数处理
        elif k in ['=', 'typename']:  # 如果键是 '=' 或 'typename'
            if force or k not in decl:  # 如果强制更新或者 decl 中不存在该键
                decl[k] = decl2[k]  # 将 decl2 中的键复制到 decl 中
        elif k == 'note':  # 如果键是 'note'
            pass  # 不做任何操作
        elif k in ['intent', 'check', 'dimension', 'optional', 'required', 'depend']:
            errmess('appenddecl: "%s" not implemented.\n' % k)  # 抛出异常，指示未实现对应功能
        else:
            raise Exception('appenddecl: Unknown variable definition key: ' + str(k))  # 抛出异常，表示未知的键
    return decl  # 返回合并后的结果字典


# 编译正则表达式模式，用于匹配字符串中的模式
selectpattern = re.compile(
    r'\s*(?P<this>(@\(@.*?@\)@|\*[\d*]+|\*\s*@\(@.*?@\)@|))(?P<after>.*)\Z', re.I)
typedefpattern = re.compile(
    r'(?:,(?P<attributes>[\w(),]+))?(::)?(?P<name>\b[a-z$_][\w$]*\b)'
    r'(?:\((?P<params>[\w,]*)\))?\Z', re.I)
nameargspattern = re.compile(
    r'\s*(?P<name>\b[\w$]+\b)\s*(@\(@\s*(?P<args>[\w\s,]*)\s*@\)@|)\s*((result(\s*@\(@\s*(?P<result>\b[\w$]+\b)\s*@\)@|))|(bind\s*@\(@\s*(?P<bind>(?:(?!@\)@).)*)\s*@\)@))*\s*\Z', re.I)
operatorpattern = re.compile(
    r'\s*(?P<scheme>(operator|assignment))'
    r'@\(@\s*(?P<name>[^)]+)\s*@\)@\s*\Z', re.I)
callnameargspattern = re.compile(
    r'\s*(?P<name>\b[\w$]+\b)\s*@\(@\s*(?P<args>.*)\s*@\)@\s*\Z', re.I)
real16pattern = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\d*\.\d+))[dD]((?:[-+]?\d+)?)')
real8pattern = re.compile(
    r'([-+]?((?:\d+(?:\.\d*)?|\d*\.\d+))[eE]((?:[-+]?\d+)?)|(\d+\.\d*))')

_intentcallbackpattern = re.compile(r'intent\s*\(.*?\bcallback\b', re.I)
# 判断一个变量声明是否包含 intentcallback 属性
def _is_intent_callback(vdecl):
    # 遍历 vdecl 中的 attrspec 属性列表
    for a in vdecl.get('attrspec', []):
        # 使用正则表达式匹配是否存在 intentcallback 属性
        if _intentcallbackpattern.match(a):
            return 1  # 如果找到匹配的 intentcallback 属性，则返回 1
    return 0  # 如果未找到匹配的 intentcallback 属性，则返回 0


# 解析 typedefpattern 格式的字符串 line
def _resolvetypedefpattern(line):
    # 去除字符串 line 中的所有空白字符
    line = ''.join(line.split())  # removes whitespace
    # 使用 typedefpattern 正则表达式匹配 line
    m1 = typedefpattern.match(line)
    print(line, m1)  # 打印 line 和匹配结果 m1（用于调试）
    if m1:
        # 提取匹配结果中的 name, attributes 和 params
        attrs = m1.group('attributes')
        attrs = [a.lower() for a in attrs.split(',')] if attrs else []
        return m1.group('name'), attrs, m1.group('params')  # 返回 name, attributes 列表和 params
    return None, [], None  # 如果未匹配成功，则返回 None, 空列表和 None


# 解析 line 中的 bind 语句
def parse_name_for_bind(line):
    # 编译正则表达式 pattern，用于匹配 bind 语句
    pattern = re.compile(r'bind\(\s*(?P<lang>[^,]+)(?:\s*,\s*name\s*=\s*["\'](?P<name>[^"\']+)["\']\s*)?\)', re.I)
    match = pattern.search(line)  # 在 line 中搜索匹配的 pattern
    bind_statement = None
    if match:
        bind_statement = match.group(0)  # 获取整个匹配的 bind 语句
        # 从 line 中移除 bind 构造
        line = line[:match.start()] + line[match.end():]
    return line, bind_statement  # 返回移除 bind 构造后的 line 和 bind 语句


# 解析 line 中的 nameargspattern 格式的字符串
def _resolvenameargspattern(line):
    line, bind_cname = parse_name_for_bind(line)  # 解析 line 中的 bind 语句
    line = markouterparen(line)  # 对 line 进行标记外部括号的处理
    m1 = nameargspattern.match(line)  # 尝试使用 nameargspattern 正则表达式匹配 line
    if m1:
        # 如果匹配成功，返回 name, args, result 和 bind_cname
        return m1.group('name'), m1.group('args'), m1.group('result'), bind_cname
    m1 = operatorpattern.match(line)  # 尝试使用 operatorpattern 正则表达式匹配 line
    if m1:
        # 如果匹配成功，构造 name，并返回 name, 空列表，None 和 None
        name = m1.group('scheme') + '(' + m1.group('name') + ')'
        return name, [], None, None
    m1 = callnameargspattern.match(line)  # 尝试使用 callnameargspattern 正则表达式匹配 line
    if m1:
        # 如果匹配成功，返回 name, args, None 和 None
        return m1.group('name'), m1.group('args'), None, None
    return None, [], None, None  # 如果所有正则表达式均未匹配成功，则返回 None, 空列表，None 和 None


# 分析输入的每一行 line，并更新全局变量
def analyzeline(m, case, line):
    """
    依次读取输入文件中的每一行，并更新全局变量。

    从输入文件中有效地读取和收集信息到全局变量 groupcache 中，该变量包含有关 Fortran 模块的每个部分的信息。

    在 analyzeline 结束时，信息被过滤到正确的字典键中，但尚未解释参数值和维度。
    """
    global groupcounter, groupname, groupcache, grouplist, filepositiontext
    global currentfilename, f77modulename, neededinterface, neededmodule
    global expectbegin, gotnextfile, previous_context

    block = m.group('this')  # 获取匹配的 'this' 组
    if case != 'multiline':
        previous_context = None  # 如果 case 不是 'multiline'，则将 previous_context 设置为 None
    if expectbegin and case not in ['begin', 'call', 'callfun', 'type'] \
       and not skipemptyends and groupcounter < 1:
        newname = os.path.basename(currentfilename).split('.')[0]
        # 输出信息表明尚未找到组。创建名称为 newname 的程序组。
        outmess(
            'analyzeline: no group yet. Creating program group with name "%s".\n' % newname)
        gotnextfile = 0
        groupcounter = groupcounter + 1
        groupname[groupcounter] = 'program'
        groupcache[groupcounter] = {}
        grouplist[groupcounter] = []
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['block'] = 'program'
        groupcache[groupcounter]['name'] = newname
        groupcache[groupcounter]['from'] = 'fromsky'
        expectbegin = 0
    # 如果 case 等于 'entry'，则解析名称、参数、结果，并将其加入 groupcache
    elif case == 'entry':
        # 从 m.group('after') 解析名称、参数、结果、并忽略第四个值
        name, args, result, _= _resolvenameargspattern(m.group('after'))
        # 如果名称不为空，则处理参数
        if name is not None:
            # 如果有参数，则清除参数中的坏名称并分割成列表
            if args:
                args = rmbadname([x.strip()
                                  for x in markoutercomma(args).split('@,@')])
            else:
                args = []
            # 确保结果为 None，否则引发断言错误
            assert result is None, repr(result)
            # 将名称及其参数加入 groupcache 中的 'entry' 键下
            groupcache[groupcounter]['entry'][name] = args
            # 更新前一个上下文为 ('entry', name, groupcounter)
            previous_context = ('entry', name, groupcounter)
    
    # 如果 case 等于 'type'，则解析类型规范、选择器、属性和类型声明
    elif case == 'type':
        # 从 block 和 m.group('after') 解析类型规范等
        typespec, selector, attr, edecl = cracktypespec0(
            block, m.group('after'))
        # 更新变量并获取最后一个变量名称
        last_name = updatevars(typespec, selector, attr, edecl)
        # 如果最后一个名称不为空，则更新前一个上下文
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    
    # 如果 case 等于 'moduleprocedure'，则处理实现的模块过程
    elif case == 'moduleprocedure':
        # 将 m.group('after') 按逗号拆分并去除空白字符，存入 groupcache 中的 'implementedby' 键下
        groupcache[groupcounter]['implementedby'] = \
            [x.strip() for x in m.group('after').split(',')]
    
    # 如果 case 等于 'parameter'，则处理参数声明
    elif case == 'parameter':
        # 获取当前组缓存中的变量声明
        edecl = groupcache[groupcounter]['vars']
        # 去除首尾括号后，从 m.group('after') 中获取参数列表字符串
        ll = m.group('after').strip()[1:-1]
        # 初始化最后一个名称为 None
        last_name = None
        # 对每个经过处理的参数字符串进行循环处理
        for e in markoutercomma(ll).split('@,@'):
            try:
                # 尝试按 '=' 分割参数字符串，获取参数名及其初始表达式
                k, initexpr = [x.strip() for x in e.split('=')]
            except Exception:
                # 如果出现异常，则输出错误信息并继续下一个参数处理
                outmess(
                    'analyzeline: could not extract name,expr in parameter statement "%s" of "%s"\n' % (e, ll))
                continue
            # 获取当前参数声明的参数列表
            params = get_parameters(edecl)
            # 清理参数名称中的坏字符
            k = rmbadname1(k)
            # 如果参数名不在当前变量声明中，则添加之
            if k not in edecl:
                edecl[k] = {}
            # 如果已经有 '=' 的参数，并且不同于当前初始表达式，则输出警告信息
            if '=' in edecl[k] and (not edecl[k]['='] == initexpr):
                outmess('analyzeline: Overwriting the value of parameter "%s" ("%s") with "%s".\n' % (
                    k, edecl[k]['='], initexpr))
            # 确定初始表达式的类型
            t = determineexprtype(initexpr, params)
            # 如果类型存在
            if t:
                # 如果类型规范为 'real'，则处理其中的十六进制字符
                if t.get('typespec') == 'real':
                    tt = list(initexpr)
                    for m in real16pattern.finditer(initexpr):
                        tt[m.start():m.end()] = list(
                            initexpr[m.start():m.end()].lower().replace('d', 'e'))
                    initexpr = ''.join(tt)
                # 如果类型规范为 'complex'，则转换表达式中的字符
                elif t.get('typespec') == 'complex':
                    initexpr = initexpr[1:].lower().replace('d', 'e').\
                        replace(',', '+1j*(')
            try:
                # 尝试计算初始表达式的值，并将其赋给 v
                v = eval(initexpr, {}, params)
            except (SyntaxError, NameError, TypeError) as msg:
                # 如果计算失败，则输出错误信息并继续下一个参数处理
                errmess('analyzeline: Failed to evaluate %r. Ignoring: %s\n'
                        % (initexpr, msg))
                continue
            # 将计算后的值转换为字符串，并存入变量声明中对应参数名称的 '=' 键下
            edecl[k]['='] = repr(v)
            # 如果已经存在 'attrspec' 键，则将 'parameter' 添加到其值列表中，否则创建 'attrspec' 键
            if 'attrspec' in edecl[k]:
                edecl[k]['attrspec'].append('parameter')
            else:
                edecl[k]['attrspec'] = ['parameter']
            # 更新最后一个名称为当前参数名称
            last_name = k
        # 更新组缓存中的变量声明
        groupcache[groupcounter]['vars'] = edecl
        # 如果最后一个名称不为空，则更新前一个上下文
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    # 如果case等于'implicit'时执行以下逻辑
    elif case == 'implicit':
        # 检查正则匹配结果中的'after'组，并且去除首尾空白字符并转换为小写
        if m.group('after').strip().lower() == 'none':
            # 如果'after'的值为'none'，则将groupcache[groupcounter]['implicit']设为None
            groupcache[groupcounter]['implicit'] = None
        elif m.group('after'):
            # 如果'after'不为空
            if 'implicit' in groupcache[groupcounter]:
                # 如果'implicit'已经在groupcache[groupcounter]中存在，则获取其值
                impl = groupcache[groupcounter]['implicit']
            else:
                # 否则初始化一个空字典作为impl
                impl = {}
            # 如果impl为None，则输出警告信息并重新初始化为一个空字典
            if impl is None:
                outmess(
                    'analyzeline: Overwriting earlier "implicit none" statement.\n')
                impl = {}
            # 遍历经过处理后的m.group('after')中的每个元素
            for e in markoutercomma(m.group('after')).split('@,@'):
                # 初始化一个空字典作为声明信息
                decl = {}
                # 用正则表达式匹配m.group('after')中的每个元素，提取相关信息
                m1 = re.match(
                    r'\s*(?P<this>.*?)\s*(\(\s*(?P<after>[a-z-, ]+)\s*\)\s*|)\Z', e, re.I)
                # 如果匹配不成功，则输出错误信息并继续下一个元素的处理
                if not m1:
                    outmess(
                        'analyzeline: could not extract info of implicit statement part "%s"\n' % (e))
                    continue
                # 使用特定的正则表达式匹配impl的类型模式
                m2 = typespattern4implicit.match(m1.group('this'))
                # 如果匹配不成功，则输出错误信息并继续下一个元素的处理
                if not m2:
                    outmess(
                        'analyzeline: could not extract types pattern of implicit statement part "%s"\n' % (e))
                    continue
                # 使用特定的函数处理类型规范并返回类型选择器、字符选择器和类型名称
                typespec, selector, attr, edecl = cracktypespec0(
                    m2.group('this'), m2.group('after'))
                kindselect, charselect, typename = cracktypespec(
                    typespec, selector)
                # 将处理得到的信息存入声明信息字典中
                decl['typespec'] = typespec
                decl['kindselector'] = kindselect
                decl['charselector'] = charselect
                decl['typename'] = typename
                # 删除字典中值为None的键
                for k in list(decl.keys()):
                    if not decl[k]:
                        del decl[k]
                # 再次遍历经过处理后的m1.group('after')中的每个元素
                for r in markoutercomma(m1.group('after')).split('@,@'):
                    # 如果包含'-'符号，则尝试分割并获取起始字符和结束字符
                    if '-' in r:
                        try:
                            begc, endc = [x.strip() for x in r.split('-')]
                        except Exception:
                            # 如果分割出错则输出错误信息并继续下一个元素的处理
                            outmess(
                                'analyzeline: expected "<char>-<char>" instead of "%s" in range list of implicit statement\n' % r)
                            continue
                    else:
                        # 否则起始字符和结束字符相同
                        begc = endc = r.strip()
                    # 如果起始字符和结束字符长度不为1，则输出错误信息并继续下一个元素的处理
                    if not len(begc) == len(endc) == 1:
                        outmess(
                            'analyzeline: expected "<char>-<char>" instead of "%s" in range list of implicit statement (2)\n' % r)
                        continue
                    # 将从起始字符到结束字符的每个字符添加到impl中，并赋予相同的声明信息
                    for o in range(ord(begc), ord(endc) + 1):
                        impl[chr(o)] = decl
            # 将处理好的impl赋值给groupcache[groupcounter]['implicit']
            groupcache[groupcounter]['implicit'] = impl
    # 如果 case 变量的取值为 'common'
    elif case == 'common':
        # 提取正则匹配组 'after' 的内容，并去除首尾空白字符
        line = m.group('after').strip()
        # 如果行不以 '/' 开头，则在行首添加 '//'
        if not line[0] == '/':
            line = '//' + line
        # 初始化空列表 cl、标志位 f 设为 0、空字符串 bn 和 ol
        cl = []
        f = 0
        bn = ''
        ol = ''
        # 遍历行中的每个字符
        for c in line:
            # 如果字符为 '/', 增加标志位 f
            if c == '/':
                f = f + 1
                continue
            # 如果标志位 f 大于等于 3
            if f >= 3:
                # 去除首尾空白字符，并若为空则设为 '_BLNK_'
                bn = bn.strip()
                if not bn:
                    bn = '_BLNK_'
                # 添加元素 [bn, ol] 到列表 cl 中
                cl.append([bn, ol])
                # 减少标志位 f 以及重置 bn 和 ol
                f = f - 2
                bn = ''
                ol = ''
            # 如果标志位 f 是奇数
            if f % 2:
                bn = bn + c
            else:
                ol = ol + c
        # 处理最后的 bn
        bn = bn.strip()
        if not bn:
            bn = '_BLNK_'
        # 将最后的 [bn, ol] 添加到列表 cl 中
        cl.append([bn, ol])
        # 初始化空字典 commonkey
        commonkey = {}
        # 如果 groupcache[groupcounter] 中有 'common' 键，则取其值
        if 'common' in groupcache[groupcounter]:
            commonkey = groupcache[groupcounter]['common']
        # 将 cl 中的条目添加到 commonkey 中
        for c in cl:
            if c[0] not in commonkey:
                commonkey[c[0]] = []
            # 将 c[1] 按照特定格式分割后，添加到 commonkey[c[0]] 中
            for i in [x.strip() for x in markoutercomma(c[1]).split('@,@')]:
                if i:
                    commonkey[c[0]].append(i)
        # 将更新后的 commonkey 存入 groupcache[groupcounter]['common']
        groupcache[groupcounter]['common'] = commonkey
        # 设置 previous_context 为 ('common', bn, groupcounter)
        previous_context = ('common', bn, groupcounter)
    
    # 如果 case 变量的取值为 'use'
    elif case == 'use':
        # 用正则表达式匹配 m.group('after') 的内容
        m1 = re.match(
            r'\A\s*(?P<name>\b\w+\b)\s*((,(\s*\bonly\b\s*:|(?P<notonly>))\s*(?P<list>.*))|)\s*\Z', m.group('after'), re.I)
        if m1:
            # 提取匹配结果的字典
            mm = m1.groupdict()
            # 如果 groupcache[groupcounter] 中没有 'use' 键，则初始化为字典
            if 'use' not in groupcache[groupcounter]:
                groupcache[groupcounter]['use'] = {}
            # 提取 'name' 字段的值，并将其作为键添加到 groupcache[groupcounter]['use'] 中
            name = m1.group('name')
            groupcache[groupcounter]['use'][name] = {}
            # 初始化 isonly 标志为 0
            isonly = 0
            # 如果匹配结果中包含 'list' 字段并且不为 None
            if 'list' in mm and mm['list'] is not None:
                # 如果匹配结果中没有 'notonly' 字段或其为 None，则 isonly 设为 1
                if 'notonly' in mm and mm['notonly'] is None:
                    isonly = 1
                # 将 'only' 字段设为 isonly
                groupcache[groupcounter]['use'][name]['only'] = isonly
                # 将 'list' 字段按逗号分割后去除首尾空白字符形成列表 ll
                ll = [x.strip() for x in mm['list'].split(',')]
                # 初始化空字典 rl
                rl = {}
                # 遍历列表 ll
                for l in ll:
                    # 如果条目包含 '=' 符号
                    if '=' in l:
                        # 用正则表达式匹配 l，提取本地变量和使用变量，并加入字典 rl
                        m2 = re.match(
                            r'\A\s*(?P<local>\b\w+\b)\s*=\s*>\s*(?P<use>\b\w+\b)\s*\Z', l, re.I)
                        if m2:
                            rl[m2.group('local').strip()] = m2.group(
                                'use').strip()
                        else:
                            # 输出错误消息，指示未找到期望的 local=>use 模式
                            outmess(
                                'analyzeline: Not local=>use pattern found in %s\n' % repr(l))
                    else:
                        # 否则，将 l 添加到 rl 中，键和值相同
                        rl[l] = l
                # 将 rl 存入 groupcache[groupcounter]['use'][name]['map']
                groupcache[groupcounter]['use'][name]['map'] = rl
            else:
                # 如果 'list' 为 None，则 pass
                pass
        else:
            # 如果无法匹配，则输出匹配组的字典
            print(m.groupdict())
            # 输出错误消息，指示无法解析 use 语句
            outmess('analyzeline: Could not crack the use statement.\n')
    elif case in ['f2pyenhancements']:
        # 如果 case 在列表 ['f2pyenhancements'] 中
        if 'f2pyenhancements' not in groupcache[groupcounter]:
            # 如果 'f2pyenhancements' 不在 groupcache[groupcounter] 中，则初始化为空字典
            groupcache[groupcounter]['f2pyenhancements'] = {}
        # 将当前操作的字典指定为 d
        d = groupcache[groupcounter]['f2pyenhancements']
        # 如果匹配对象中 'this' 组的内容为 'usercode'，并且 'usercode' 已经存在于 d 中
        if m.group('this') == 'usercode' and 'usercode' in d:
            # 如果 d['usercode'] 是字符串，则转换为列表
            if isinstance(d['usercode'], str):
                d['usercode'] = [d['usercode']]
            # 将匹配对象中 'after' 组的内容添加到 d['usercode'] 列表中
            d['usercode'].append(m.group('after'))
        else:
            # 否则，将匹配对象中 'this' 组和 'after' 组的内容添加到 d 中
            d[m.group('this')] = m.group('after')
    elif case == 'multiline':
        # 如果 case 等于 'multiline'
        if previous_context is None:
            # 如果 previous_context 为 None
            if verbose:
                # 如果 verbose 不为 0，则输出消息指示无法分析多行块的上下文
                outmess('analyzeline: No context for multiline block.\n')
            # 返回空
            return
        # 将 groupcounter 的值赋给 gc
        gc = groupcounter
        # 在 groupcache[gc] 中附加多行内容的相关信息
        appendmultiline(groupcache[gc],
                        previous_context[:2],
                        m.group('this'))
    else:
        # 否则（如果 case 既不是 'f2pyenhancements' 也不是 'multiline'）
        if verbose > 1:
            # 如果 verbose 大于 1，则打印匹配对象的字典形式和指示未实现该行代码的消息
            print(m.groupdict())
            outmess('analyzeline: No code implemented for line.\n')
# 如果 'f2pymultilines' 不在组中，则将其初始化为空字典
def appendmultiline(group, context_name, ml):
    if 'f2pymultilines' not in group:
        group['f2pymultilines'] = {}
    # 获取 'f2pymultilines' 字典
    d = group['f2pymultilines']
    # 如果 context_name 不在 d 中，则将其初始化为一个空列表
    if context_name not in d:
        d[context_name] = []
    # 将 ml 添加到 d[context_name] 列表中
    d[context_name].append(ml)
    return

# 解析给定的类型说明符和行，返回类型、选择器、属性和剩余行
def cracktypespec0(typespec, ll):
    selector = None
    attr = None
    # 检查是否匹配 'double complex' 或 'double precision'，并标准化 typespec
    if re.match(r'double\s*complex', typespec, re.I):
        typespec = 'double complex'
    elif re.match(r'double\s*precision', typespec, re.I):
        typespec = 'double precision'
    else:
        typespec = typespec.strip().lower()
    # 使用 selectpattern 匹配 ll，获取分组字典 d
    m1 = selectpattern.match(markouterparen(ll))
    # 如果匹配失败，输出错误信息并返回
    if not m1:
        outmess(
            'cracktypespec0: no kind/char_selector pattern found for line.\n')
        return
    # 将 m1 的分组字典的值去除外部括号并更新到 d
    d = m1.groupdict()
    for k in list(d.keys()):
        d[k] = unmarkouterparen(d[k])
    # 如果 typespec 是基本类型之一，则从 d 中获取选择器和更新 ll
    if typespec in ['complex', 'integer', 'logical', 'real', 'character', 'type']:
        selector = d['this']
        ll = d['after']
    # 查找 ll 中 '::' 的位置，获取属性 attr 和更新 ll
    i = ll.find('::')
    if i >= 0:
        attr = ll[:i].strip()
        ll = ll[i + 2:]
    return typespec, selector, attr, ll

# 编译并初始化正则表达式对象，用于解析变量名称和其后的内容
namepattern = re.compile(r'\s*(?P<name>\b\w+\b)\s*(?P<after>.*)\s*\Z', re.I)
# 编译并初始化正则表达式对象，用于解析类型说明符中的 kind 选择器
kindselector = re.compile(
    r'\s*(\(\s*(kind\s*=)?\s*(?P<kind>.*)\s*\)|\*\s*(?P<kind2>.*?))\s*\Z', re.I)
# 编译并初始化正则表达式对象，用于解析字符长度选择器
charselector = re.compile(
    r'\s*(\((?P<lenkind>.*)\)|\*\s*(?P<charlen>.*))\s*\Z', re.I)
# 编译并初始化正则表达式对象，用于解析长度和种类
lenkindpattern = re.compile(
    r'\s*(kind\s*=\s*(?P<kind>.*?)\s*(@,@\s*len\s*=\s*(?P<len>.*)|)'
    r'|(len\s*=\s*|)(?P<len2>.*?)\s*(@,@\s*(kind\s*=\s*|)(?P<kind2>.*)'
    r'|(f2py_len\s*=\s*(?P<f2py_len>.*))|))\s*\Z', re.I)
# 编译并初始化正则表达式对象，用于解析长度数组
lenarraypattern = re.compile(
    r'\s*(@\(@\s*(?!/)\s*(?P<array>.*?)\s*@\)@\s*\*\s*(?P<len>.*?)|(\*\s*(?P<len2>.*?)|)\s*(@\(@\s*(?!/)\s*(?P<array2>.*?)\s*@\)@|))\s*(=\s*(?P<init>.*?)|(@\(@|)/\s*(?P<init2>.*?)\s*/(@\)@|)|)\s*\Z', re.I)

# 移除表达式中多余的空格，并确保在括号和操作符周围的空格正确
def removespaces(expr):
    expr = expr.strip()
    if len(expr) <= 1:
        return expr
    expr2 = expr[0]
    for i in range(1, len(expr) - 1):
        if (expr[i] == ' ' and
            ((expr[i + 1] in "()[]{}=+-/* ") or
                (expr[i - 1] in "()[]{}=+-/* "))):
            continue
        expr2 = expr2 + expr[i]
    expr2 = expr2 + expr[-1]
    return expr2

# 将输入行中被引号包围的空格替换为 "@_@"
def markinnerspaces(line):
    """
    The function replace all spaces in the input variable line which are 
    surrounded with quotation marks, with the triplet "@_@".

    For instance, for the input "a 'b c'" the function returns "a 'b@_@c'"

    Parameters
    ----------
    line : str

    Returns
    -------
    str

    """
    fragment = ''
    inside = False
    current_quote = None
    escaped = ''
    # 遍历字符串 line 中的每个字符 c
    for c in line:
        # 检查是否需要转义当前字符
        if escaped == '\\' and c in ['\\', '\'', '"']:
            # 如果上一个字符为反斜杠，并且当前字符是反斜杠、单引号或双引号，则将当前字符加入片段 fragment 中
            fragment += c
            # 更新转义状态为当前字符 c
            escaped = c
            # 继续处理下一个字符
            continue
        
        # 检查是否处于引号内
        if not inside and c in ['\'', '"']:
            # 如果不在引号内且当前字符是单引号或双引号，则更新当前引号类型
            current_quote = c
        
        # 检查当前字符是否与当前引号匹配
        if c == current_quote:
            # 如果当前字符与当前引号匹配，则切换引号内外状态
            inside = not inside
        elif c == ' ' and inside:
            # 如果当前字符是空格且处于引号内，则将 '@_@' 加入片段 fragment 中以表示空格
            fragment += '@_@'
            # 继续处理下一个字符
            continue
        
        # 将当前字符加入片段 fragment 中
        fragment += c
        # 重置转义状态为非反斜杠状态
        escaped = c
    
    # 返回处理后的片段 fragment
    return fragment
# 定义一个函数updatevars，用于更新变量信息和groupcache
def updatevars(typespec, selector, attrspec, entitydecl):
    """
    Returns last_name, the variable name without special chars, parenthesis
        or dimension specifiers.

    Alters groupcache to add the name, typespec, attrspec (and possibly value)
    of current variable.
    """
    # 声明引用全局变量groupcache和groupcounter
    global groupcache, groupcounter

    # 初始化last_name为None
    last_name = None

    # 调用cracktypespec函数解析typespec和selector，获取kindselect、charselect和typename
    kindselect, charselect, typename = cracktypespec(typespec, selector)

    # 清理attrspec中的外部逗号、空白和不需要的字符
    if attrspec:
        # 使用markoutercomma函数处理attrspec，分割后去除空白并存入列表l
        attrspec = [x.strip() for x in markoutercomma(attrspec).split('@,@')]
        l = []

        # 编译正则表达式，以匹配attrspec中的字符串开始部分的字母序列
        c = re.compile(r'(?P<start>[a-zA-Z]+)')

        # 遍历attrspec列表的每个元素a，根据正则匹配结果进行处理后添加到列表l
        for a in attrspec:
            if not a:
                continue
            m = c.match(a)
            if m:
                s = m.group('start').lower()
                a = s + a[len(s):]
            l.append(a)
        attrspec = l

    # 处理entitydecl，去除外部逗号并根据特定模式分割，去除空格后存入el列表
    el = [x.strip() for x in markoutercomma(entitydecl).split('@,@')]

    # 初始化el1列表
    el1 = []

    # 遍历el列表中的每个元素e，去除内部空格后根据特定模式分割，去除空项后添加到el1列表
    for e in el:
        for e1 in [x.strip() for x in markoutercomma(removespaces(markinnerspaces(e)), comma=' ').split('@ @')]:
            if e1:
                el1.append(e1.replace('@_@', ' '))

    # 返回last_name，此函数不涉及修改groupcache
    return last_name


# 定义函数cracktypespec，用于解析typespec和selector，返回kindselect、charselect和typename
def cracktypespec(typespec, selector):
    kindselect = None
    charselect = None
    typename = None
    # 如果 selector 非空
    if selector:
        # 如果 typespec 是 ['complex', 'integer', 'logical', 'real'] 中的一种
        if typespec in ['complex', 'integer', 'logical', 'real']:
            # 使用 kindselector 匹配 selector，获取 kindselect
            kindselect = kindselector.match(selector)
            # 如果 kindselect 为空
            if not kindselect:
                # 输出错误信息，指示未找到 kindselector 的模式
                outmess(
                    'cracktypespec: no kindselector pattern found for %s\n' % (repr(selector)))
                # 返回空值，结束函数
                return
            # 将 kindselect 转换为字典形式
            kindselect = kindselect.groupdict()
            # 将 '*' 键设为 'kind2' 值
            kindselect['*'] = kindselect['kind2']
            # 删除 'kind2' 键
            del kindselect['kind2']
            # 遍历 kindselect 的键列表
            for k in list(kindselect.keys()):
                # 如果键对应的值为空
                if not kindselect[k]:
                    # 删除该键
                    del kindselect[k]
            # 遍历 kindselect 的键值对
            for k, i in list(kindselect.items()):
                # 对每个值应用 rmbadname1 函数
                kindselect[k] = rmbadname1(i)
        # 如果 typespec 是 'character'
        elif typespec == 'character':
            # 使用 charselector 匹配 selector，获取 charselect
            charselect = charselector.match(selector)
            # 如果 charselect 为空
            if not charselect:
                # 输出错误信息，指示未找到 charselector 的模式
                outmess(
                    'cracktypespec: no charselector pattern found for %s\n' % (repr(selector)))
                # 返回空值，结束函数
                return
            # 将 charselect 转换为字典形式
            charselect = charselect.groupdict()
            # 将 '*' 键设为 'charlen' 值
            charselect['*'] = charselect['charlen']
            # 删除 'charlen' 键
            del charselect['charlen']
            # 如果 charselect 中有 'lenkind' 键
            if charselect['lenkind']:
                # 使用 lenkindpattern 匹配 markoutercomma(charselect['lenkind']) 的结果
                lenkind = lenkindpattern.match(
                    markoutercomma(charselect['lenkind']))
                # 将匹配结果转换为字典形式
                lenkind = lenkind.groupdict()
                # 遍历 ['len', 'kind'] 列表
                for lk in ['len', 'kind']:
                    # 如果 lenkind[lk + '2'] 存在
                    if lenkind[lk + '2']:
                        # 将 lenkind[lk + '2'] 的值赋给 lenkind[lk]
                        lenkind[lk] = lenkind[lk + '2']
                    # 将 lenkind[lk] 的值赋给 charselect[lk]
                    charselect[lk] = lenkind[lk]
                    # 删除 lenkind[lk + '2'] 键
                    del lenkind[lk + '2']
                # 如果 lenkind['f2py_len'] 不为空
                if lenkind['f2py_len'] is not None:
                    # 将 lenkind['f2py_len'] 的值赋给 charselect['f2py_len']
                    charselect['f2py_len'] = lenkind['f2py_len']
            # 删除 'lenkind' 键
            del charselect['lenkind']
            # 遍历 charselect 的键列表
            for k in list(charselect.keys()):
                # 如果键对应的值为空
                if not charselect[k]:
                    # 删除该键
                    del charselect[k]
            # 遍历 charselect 的键值对
            for k, i in list(charselect.items()):
                # 对每个值应用 rmbadname1 函数
                charselect[k] = rmbadname1(i)
        # 如果 typespec 是 'type'
        elif typespec == 'type':
            # 使用正则表达式匹配 selector 中的类型名称，获取 typename
            typename = re.match(r'\s*\(\s*(?P<name>\w+)\s*\)', selector, re.I)
            # 如果 typename 存在
            if typename:
                # 提取 typename 中的 'name' 组名对应的值
                typename = typename.group('name')
            else:
                # 输出错误信息，指示在 selector 中未找到 typename
                outmess('cracktypespec: no typename found in %s\n' %
                        (repr(typespec + selector)))
        else:
            # 输出错误信息，指示未使用 selector
            outmess('cracktypespec: no selector used for %s\n' %
                    (repr(selector)))
    # 返回 kindselect, charselect, typename 三个变量
    return kindselect, charselect, typename
# 设置声明的属性规范，将属性添加到声明中的属性规范列表中
def setattrspec(decl, attr, force=0):
    # 如果声明为空，初始化为空字典
    if not decl:
        decl = {}
    # 如果属性为空，返回声明
    if not attr:
        return decl
    # 如果声明中没有 'attrspec' 键，创建一个包含属性的列表，并返回声明
    if 'attrspec' not in decl:
        decl['attrspec'] = [attr]
        return decl
    # 如果 force 参数为真，强制将属性添加到 'attrspec' 列表中
    if force:
        decl['attrspec'].append(attr)
    # 如果属性已经存在于 'attrspec' 列表中，直接返回声明
    if attr in decl['attrspec']:
        return decl
    # 处理静态 ('static') 和自动 ('automatic') 属性的特殊情况
    if attr == 'static' and 'automatic' not in decl['attrspec']:
        decl['attrspec'].append(attr)
    elif attr == 'automatic' and 'static' not in decl['attrspec']:
        decl['attrspec'].append(attr)
    # 处理公共 ('public') 和私有 ('private') 属性的特殊情况
    elif attr == 'public':
        if 'private' not in decl['attrspec']:
            decl['attrspec'].append(attr)
    elif attr == 'private':
        if 'public' not in decl['attrspec']:
            decl['attrspec'].append(attr)
    else:
        # 添加属性到 'attrspec' 列表
        decl['attrspec'].append(attr)
    return decl


# 设置声明的种类选择器，指定声明的种类选择器
def setkindselector(decl, sel, force=0):
    # 如果声明为空，初始化为空字典
    if not decl:
        decl = {}
    # 如果选择器为空，返回声明
    if not sel:
        return decl
    # 如果声明中没有 'kindselector' 键，设置种类选择器并返回声明
    if 'kindselector' not in decl:
        decl['kindselector'] = sel
        return decl
    # 遍历选择器中的键，如果 force 为真或者键不存在于声明的种类选择器中，添加到种类选择器中
    for k in list(sel.keys()):
        if force or k not in decl['kindselector']:
            decl['kindselector'][k] = sel[k]
    return decl


# 设置声明的字符选择器，指定声明的字符选择器
def setcharselector(decl, sel, force=0):
    # 如果声明为空，初始化为空字典
    if not decl:
        decl = {}
    # 如果选择器为空，返回声明
    if not sel:
        return decl
    # 如果声明中没有 'charselector' 键，设置字符选择器并返回声明
    if 'charselector' not in decl:
        decl['charselector'] = sel
        return decl

    # 遍历选择器中的键，如果 force 为真或者键不存在于声明的字符选择器中，添加到字符选择器中
    for k in list(sel.keys()):
        if force or k not in decl['charselector']:
            decl['charselector'][k] = sel[k]
    return decl


# 获取代码块的名称，如果存在 'name' 键则返回其名称，否则返回指定的未知名称
def getblockname(block, unknown='unknown'):
    if 'name' in block:
        return block['name']
    return unknown


# 设置消息文本，在全局变量 filepositiontext 中设置文件位置信息
def setmesstext(block):
    global filepositiontext

    # 尝试从代码块中获取 'from' 和 'name' 字段，并设置文件位置文本
    try:
        filepositiontext = 'In: %s:%s\n' % (block['from'], block['name'])
    except Exception:
        pass


# 获取使用字典，递归获取父代码块的使用字典并更新当前代码块的使用字典
def get_usedict(block):
    usedict = {}
    # 如果存在父代码块，则递归获取其使用字典
    if 'parent_block' in block:
        usedict = get_usedict(block['parent_block'])
    # 如果存在 'use' 字段，则更新使用字典
    if 'use' in block:
        usedict.update(block['use'])
    return usedict


# 获取使用参数，获取当前代码块及其父代码块的使用字典中的参数映射
def get_useparameters(block, param_map=None):
    global f90modulevars

    # 如果参数映射为空，初始化为空字典
    if param_map is None:
        param_map = {}
    # 获取当前代码块的使用字典
    usedict = get_usedict(block)
    # 如果使用字典为空，返回参数映射
    if not usedict:
        return param_map
    # 遍历 usedict 字典中的每个键值对，键为 usename，值为 mapping
    for usename, mapping in list(usedict.items()):
        # 将 usename 转换为小写
        usename = usename.lower()
        # 如果 usename 不在 f90modulevars 字典的键集合中，则输出错误信息并继续下一次循环
        if usename not in f90modulevars:
            outmess('get_useparameters: no module %s info used by %s\n' %
                    (usename, block.get('name')))
            continue
        # 获取 usename 对应的模块变量字典
        mvars = f90modulevars[usename]
        # 调用函数 get_parameters 获取模块变量的参数信息
        params = get_parameters(mvars)
        # 如果 params 为空，则继续下一次循环
        if not params:
            continue
        # 如果存在 mapping 映射关系，则输出警告信息，暂时未实现该功能
        # XXX: apply mapping
        if mapping:
            errmess('get_useparameters: mapping for %s not impl.\n' % (mapping))
        # 遍历 params 字典中的每个键值对，如果键存在于 param_map 中，则输出信息进行参数覆盖操作
        for k, v in list(params.items()):
            if k in param_map:
                outmess('get_useparameters: overriding parameter %s with'
                        ' value from module %s\n' % (repr(k), repr(usename)))
            # 将参数键值对添加到 param_map 中
            param_map[k] = v

    # 返回最终的参数映射字典 param_map
    return param_map
# 如果全局变量 f90modulevars 为空，则直接返回传入的 block 参数
def postcrack2(block, tab='', param_map=None):
    global f90modulevars

    if not f90modulevars:
        return block
    # 如果 block 是列表，则递归调用 postcrack2 处理列表中的每个元素
    if isinstance(block, list):
        ret = [postcrack2(g, tab=tab + '\t', param_map=param_map)
               for g in block]
        return ret
    # 设置 block 的信息文本
    setmesstext(block)
    # 打印块的名称，带有缩进 tab，并换行输出
    outmess('%sBlock: %s\n' % (tab, block['name']), 0)

    # 如果 param_map 为空，则调用 get_useparameters 函数获取参数映射
    if param_map is None:
        param_map = get_useparameters(block)

    # 如果 param_map 不为空且 block 中包含 'vars' 键
    if param_map is not None and 'vars' in block:
        vars = block['vars']
        # 遍历 block['vars'] 中的每个变量
        for n in list(vars.keys()):
            var = vars[n]
            # 如果变量中包含 'kindselector' 键
            if 'kindselector' in var:
                kind = var['kindselector']
                # 如果 kind 中包含 'kind' 键
                if 'kind' in kind:
                    val = kind['kind']
                    # 如果 val 存在于 param_map 中，则替换 kind['kind'] 的值
                    if val in param_map:
                        kind['kind'] = param_map[val]
    
    # 对 block['body'] 中的每个元素递归调用 postcrack2 处理
    new_body = [postcrack2(b, tab=tab + '\t', param_map=param_map)
                for b in block['body']]
    block['body'] = new_body

    # 返回处理后的 block
    return block


# 如果 block 是列表，则递归调用 postcrack 处理列表中的每个元素
def postcrack(block, args=None, tab=''):
    """
    TODO:
          function return values
          determine expression types if in argument list
    """
    global usermodules, onlyfunctions

    if isinstance(block, list):
        gret = []
        uret = []
        for g in block:
            # 设置每个元素的信息文本
            setmesstext(g)
            # 递归调用 postcrack 处理元素 g，并根据名称将用户程序排在前面
            g = postcrack(g, tab=tab + '\t')
            if 'name' in g and '__user__' in g['name']:
                uret.append(g)
            else:
                gret.append(g)
        return uret + gret
    # 设置 block 的信息文本
    setmesstext(block)
    # 如果 block 不是字典类型或者不包含 'block' 键，则抛出异常
    if not isinstance(block, dict) and 'block' not in block:
        raise Exception('postcrack: Expected block dictionary instead of ' +
                        str(block))
    # 如果 block 中包含 'name' 键且不是 'unknown_interface'，则打印块的名称
    if 'name' in block and not block['name'] == 'unknown_interface':
        outmess('%sBlock: %s\n' % (tab, block['name']), 0)
    # 分析 block 的参数列表
    block = analyzeargs(block)
    # 分析 block 的常见内容
    block = analyzecommon(block)
    # 分析 block 中的变量并赋值给 block['vars']
    block['vars'] = analyzevars(block)
    # 对 block 中的变量名称进行排序并赋值给 block['sortvars']
    block['sortvars'] = sortvarnames(block['vars'])
    # 如果 block 中包含 'args' 键且其值为真值，则将其赋给 args
    if 'args' in block and block['args']:
        args = block['args']
    # 分析 block 的主体内容并赋值给 block['body']
    block['body'] = analyzebody(block, args, tab=tab)

    # 检查是否定义了用户模块
    userisdefined = []
    if 'use' in block:
        useblock = block['use']
        # 遍历 useblock 中的键，如果包含 '__user__'，则添加到 userisdefined
        for k in list(useblock.keys()):
            if '__user__' in k:
                userisdefined.append(k)
    else:
        useblock = {}
    name = ''
    if 'name' in block:
        name = block['name']
    # 如果 block 中包含 'name' 键，则将其赋值给 name

    # and not userisdefined: # Build a __user__ module
    # 检查是否在块中存在 'externals' 键，并且其值非空
    if 'externals' in block and block['externals']:
        # 初始化一个空列表 interfaced
        interfaced = []
        # 如果块中存在 'interfaced' 键，则将其赋值给 interfaced
        if 'interfaced' in block:
            interfaced = block['interfaced']
        # 复制块中的 'vars' 键对应的值，赋给 mvars
        mvars = copy.copy(block['vars'])
        # 根据条件设置模块名 mname
        if name:
            mname = name + '__user__routines'
        else:
            mname = 'unknown__user__routines'
        # 如果 mname 已经在 userisdefined 中定义过，则生成一个新的唯一的名称
        if mname in userisdefined:
            i = 1
            while '%s_%i' % (mname, i) in userisdefined:
                i = i + 1
            mname = '%s_%i' % (mname, i)
        # 初始化 interface 字典，表示一个接口块
        interface = {'block': 'interface', 'body': [],
                     'vars': {}, 'name': name + '_user_interface'}
        # 遍历 externals 中的每个外部变量 e
        for e in block['externals']:
            # 如果 e 在 interfaced 中，则处理其定义
            if e in interfaced:
                edef = []
                j = -1
                # 遍历块的 body 中的每个子块 b
                for b in block['body']:
                    j = j + 1
                    # 如果子块 b 是接口块
                    if b['block'] == 'interface':
                        i = -1
                        # 遍历接口块的 body 中的每个子块 bb
                        for bb in b['body']:
                            i = i + 1
                            # 如果 bb 中存在 'name' 键且等于 e，则找到了要处理的定义
                            if 'name' in bb and bb['name'] == e:
                                edef = copy.copy(bb)
                                # 从接口块的 body 中删除这个定义
                                del b['body'][i]
                                break
                        # 如果找到了 edef，则继续处理
                        if edef:
                            # 如果接口块的 body 已经为空，则从块的 body 中删除这个接口块
                            if not b['body']:
                                del block['body'][j]
                            # 从 interfaced 列表中删除 e
                            del interfaced[interfaced.index(e)]
                            break
                # 将 edef 添加到 interface 的 body 中
                interface['body'].append(edef)
            else:
                # 如果 e 不在 mvars 中或者在 mvars 中但不是外部变量，则将其添加到 interface 的 vars 中
                if e in mvars and not isexternal(mvars[e]):
                    interface['vars'][e] = mvars[e]
        # 如果 interface 的 vars 非空或者 body 非空，则表示有接口需要添加到用户模块中
        if interface['vars'] or interface['body']:
            # 更新 block 中的 interfaced
            block['interfaced'] = interfaced
            # 创建一个模块块 mblock，包含一个接口块
            mblock = {'block': 'python module', 'body': [
                interface], 'vars': {}, 'name': mname, 'interfaced': block['externals']}
            # 在 useblock 中添加 mname 的条目
            useblock[mname] = {}
            # 将 mblock 添加到 usermodules 列表中
            usermodules.append(mblock)
    # 如果 useblock 不为空，则将其添加到 block 的 use 中
    if useblock:
        block['use'] = useblock
    # 返回块 block
    return block
# 对变量名进行排序，分为独立变量和依赖变量两类
def sortvarnames(vars):
    indep = []  # 存储独立变量的列表
    dep = []    # 存储依赖变量的列表
    for v in list(vars.keys()):  # 遍历变量字典的键列表
        if 'depend' in vars[v] and vars[v]['depend']:  # 检查变量是否有依赖关系
            dep.append(v)  # 如果有依赖关系，将变量添加到依赖列表中
        else:
            indep.append(v)  # 如果没有依赖关系，将变量添加到独立列表中
    n = len(dep)  # 记录依赖列表的初始长度
    i = 0         # 初始化计数器
    while dep:    # 循环直到依赖列表为空
        v = dep[0]  # 取出依赖列表的第一个变量
        fl = 0      # 初始化标志位为0
        for w in dep[1:]:  # 遍历剩余的依赖列表
            if w in vars[v]['depend']:  # 检查是否存在循环依赖
                fl = 1  # 如果存在循环依赖，将标志位置为1
                break
        if fl:  # 如果存在循环依赖
            dep = dep[1:] + [v]  # 将第一个变量移到列表末尾，继续尝试解决循环依赖
            i = i + 1  # 增加计数器
            if i > n:  # 如果计数器超过初始长度
                errmess('sortvarnames: failed to compute dependencies because'
                        ' of cyclic dependencies between '
                        + ', '.join(dep) + '\n')  # 报告循环依赖错误
                indep = indep + dep  # 将剩余的依赖变量移动到独立变量中
                break
        else:
            indep.append(v)  # 如果没有循环依赖，将变量添加到独立变量中
            dep = dep[1:]  # 移除已处理的第一个依赖变量
            n = len(dep)  # 更新依赖列表的长度
            i = 0         # 重置计数器
    return indep  # 返回排序后的独立变量列表


# 分析代码块中的公共变量，并进行属性处理
def analyzecommon(block):
    if not hascommon(block):  # 检查代码块是否包含公共变量
        return block  # 如果没有公共变量，直接返回代码块
    commonvars = []  # 存储公共变量的列表
    for k in list(block['common'].keys()):  # 遍历公共变量字典的键列表
        comvars = []  # 存储当前公共变量的变量名列表
        for e in block['common'][k]:  # 遍历每个公共变量的元素列表
            m = re.match(  # 使用正则表达式匹配变量名及其维度
                r'\A\s*\b(?P<name>.*?)\b\s*(\((?P<dims>.*?)\)|)\s*\Z', e, re.I)
            if m:  # 如果匹配成功
                dims = []  # 存储变量维度的列表
                if m.group('dims'):  # 如果存在维度信息
                    dims = [x.strip()
                            for x in markoutercomma(m.group('dims')).split('@,@')]  # 处理维度信息
                n = rmbadname1(m.group('name').strip())  # 清理变量名中的非法字符
                if n in block['vars']:  # 如果变量名已存在于变量字典中
                    if 'attrspec' in block['vars'][n]:  # 如果已存在属性规范
                        block['vars'][n]['attrspec'].append(
                            'dimension(%s)' % (','.join(dims)))  # 添加维度属性规范
                    else:
                        block['vars'][n]['attrspec'] = [
                            'dimension(%s)' % (','.join(dims))]  # 创建新的维度属性规范
                else:
                    if dims:
                        block['vars'][n] = {
                            'attrspec': ['dimension(%s)' % (','.join(dims))]}  # 创建新的变量条目，包含维度属性
                    else:
                        block['vars'][n] = {}  # 创建新的变量条目，不含维度属性
                if n not in commonvars:  # 如果变量名不在公共变量列表中
                    commonvars.append(n)  # 将变量名添加到公共变量列表中
            else:
                n = e
                errmess(
                    'analyzecommon: failed to extract "<name>[(<dims>)]" from "%s" in common /%s/.\n' % (e, k))
            comvars.append(n)  # 将处理后的变量名添加到当前公共变量的列表中
        block['common'][k] = comvars  # 更新公共变量字典中的变量列表
    if 'commonvars' not in block:
        block['commonvars'] = commonvars  # 如果不存在公共变量列表，创建并赋值
    else:
        block['commonvars'] = block['commonvars'] + commonvars  # 否则将当前公共变量列表追加到已有列表中
    return block  # 返回更新后的代码块


# 分析代码块的主体部分，处理其中的参数和模块变量
def analyzebody(block, args, tab=''):
    global usermodules, skipfuncs, onlyfuncs, f90modulevars  # 引入全局变量

    setmesstext(block)  # 设置消息文本

    maybe_private = {  # 创建可能为私有的变量字典
        key: value
        for key, value in block['vars'].items()
        if 'attrspec' not in value or 'public' not in value['attrspec']
    }

    body = []  # 存储代码块主体的列表
    # 遍历当前代码块的每个子块
    for b in block['body']:
        # 将当前子块的父块设置为当前块
        b['parent_block'] = block
        # 如果当前子块是函数或子程序
        if b['block'] in ['function', 'subroutine']:
            # 如果提供了参数并且当前子块的名称不在参数列表中，则跳过
            if args is not None and b['name'] not in args:
                continue
            else:
                as_ = b['args']  # 将当前子块的参数列表赋值给 as_
            # 如果当前子块的名称在 maybe_private 字典的键中
            if b['name'] in maybe_private.keys():
                skipfuncs.append(b['name'])  # 将当前子块的名称添加到 skipfuncs 列表中
            # 如果当前子块的名称在 skipfuncs 列表中，则跳过
            if b['name'] in skipfuncs:
                continue
            # 如果仅允许特定函数，并且当前子块的名称不在允许列表中，则跳过
            if onlyfuncs and b['name'] not in onlyfuncs:
                continue
            # 对当前子块进行解析，并保存接口信息到 saved_interface 字段
            b['saved_interface'] = crack2fortrangen(
                b, '\n' + ' ' * 6, as_interface=True)

        else:
            as_ = args  # 否则将参数列表赋值给 as_
        
        # 对当前子块进行后续处理，并根据需要添加制表符
        b = postcrack(b, as_, tab=tab + '\t')
        
        # 如果当前子块是接口或抽象接口，并且没有子块且未实现
        if b['block'] in ['interface', 'abstract interface'] and \
           not b['body'] and not b.get('implementedby'):
            # 如果当前子块中未包含 'f2pyenhancements' 字段，则跳过
            if 'f2pyenhancements' not in b:
                continue
        
        # 如果当前子块的类型去除空格后为 'pythonmodule'，则将其添加到 usermodules 列表中
        if b['block'].replace(' ', '') == 'pythonmodule':
            usermodules.append(b)
        else:
            # 如果当前子块的类型为 'module'，将当前子块的名称与其变量列表添加到 f90modulevars 字典中
            if b['block'] == 'module':
                f90modulevars[b['name']] = b['vars']
            # 将当前子块添加到 body 列表中
            body.append(b)
    
    # 返回处理后的子块列表
    return body
# 定义一个函数，用于构建隐式规则集合，基于给定的块信息
def buildimplicitrules(block):
    # 调用函数设置块的消息文本
    setmesstext(block)
    # 初始化隐式规则为默认的隐式规则集合
    implicitrules = defaultimplicitrules
    # 初始化属性规则为空字典
    attrrules = {}
    
    # 检查块中是否包含 'implicit' 键
    if 'implicit' in block:
        # 如果 'implicit' 为 None，则将隐式规则设置为 None
        if block['implicit'] is None:
            implicitrules = None
            # 如果设置了详细输出（verbose > 1），输出相应消息
            if verbose > 1:
                outmess(
                    'buildimplicitrules: no implicit rules for routine %s.\n' % repr(block['name']))
        else:
            # 遍历块中 'implicit' 键的所有键
            for k in list(block['implicit'].keys()):
                # 检查键对应的值中的 'typespec' 是否不是 'static' 或 'automatic'
                if block['implicit'][k].get('typespec') not in ['static', 'automatic']:
                    # 如果符合条件，将其作为隐式规则添加到隐式规则集合中
                    implicitrules[k] = block['implicit'][k]
                else:
                    # 否则，将其作为属性规则添加到属性规则字典中
                    attrrules[k] = block['implicit'][k]['typespec']
    
    # 返回构建好的隐式规则集合和属性规则字典
    return implicitrules, attrrules


# 定义一个函数，类似于 eval 函数，但仅返回整数和浮点数
def myeval(e, g=None, l=None):
    r = eval(e, g, l)
    # 如果结果是整数或浮点数类型，则直接返回结果
    if type(r) in [int, float]:
        return r
    # 否则，抛出 ValueError 异常
    raise ValueError('r=%r' % (r))


# 编译正则表达式，用于匹配形如 'x' 的线性系数表达式
getlincoef_re_1 = re.compile(r'\A\b\w+\b\Z', re.I)


# 定义一个函数，用于从表达式中提取线性系数 a 和常数项 b，其中 x 是 xset 集合中的符号
def getlincoef(e, xset):
    """
    Obtain ``a`` and ``b`` when ``e == "a*x+b"``, where ``x`` is a symbol in
    xset.

    >>> getlincoef('2*x + 1', {'x'})
    (2, 1, 'x')
    >>> getlincoef('3*x + x*2 + 2 + 1', {'x'})
    (5, 3, 'x')
    >>> getlincoef('0', {'x'})
    (0, 0, None)
    >>> getlincoef('0*x', {'x'})
    (0, 0, 'x')
    >>> getlincoef('x*x', {'x'})
    (None, None, None)

    This can be tricked by sufficiently complex expressions

    >>> getlincoef('(x - 0.5)*(x - 1.5)*(x - 1)*x + 2*x + 3', {'x'})
    (2.0, 3.0, 'x')
    """
    # 尝试对表达式进行求值，并将结果转换为整数
    try:
        c = int(myeval(e, {}, {}))
        return 0, c, None
    except Exception:
        pass
    
    # 如果表达式符合形如 'x' 的简单模式，则返回默认的线性系数 a=1, b=0, x
    if getlincoef_re_1.match(e):
        return 1, 0, e
    
    # 返回空值，表示无法提取有效的线性系数
    len_e = len(e)
    # 对于集合 xset 中的每个元素 x 执行循环
    for x in xset:
        # 如果 x 的长度大于 len_e，则跳过当前循环，继续下一个 x
        if len(x) > len_e:
            continue
        # 如果在表达式 e 中找到类似于函数调用并且 x 是其中的一个参数，则跳过当前循环
        if re.search(r'\w\s*\([^)]*\b' + x + r'\b', e):
            continue
        
        # 创建一个正则表达式对象，用于匹配表达式 e 中的 x（不区分大小写）
        re_1 = re.compile(r'(?P<before>.*?)\b' + x + r'\b(?P<after>.*)', re.I)
        # 尝试匹配表达式 e，如果匹配成功则执行下面的代码块
        m = re_1.match(e)
        if m:
            try:
                # 再次尝试匹配表达式 e，并在其中循环直到找不到匹配项
                m1 = re_1.match(e)
                while m1:
                    # 将匹配到的部分替换为 0，并形成新的表达式 ee
                    ee = '%s(%s)%s' % (
                        m1.group('before'), 0, m1.group('after'))
                    m1 = re_1.match(ee)
                # 调用自定义函数 myeval 计算表达式 ee 的值，传入空的局部变量和全局变量字典
                b = myeval(ee, {}, {})
                
                # 重置匹配对象，再次循环匹配表达式 e
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (
                        m1.group('before'), 1, m1.group('after'))
                    m1 = re_1.match(ee)
                a = myeval(ee, {}, {}) - b  # 计算得到 a 的值
                
                # 重置匹配对象，再次循环匹配表达式 e
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (
                        m1.group('before'), 0.5, m1.group('after'))
                    m1 = re_1.match(ee)
                c = myeval(ee, {}, {})  # 计算得到 c 的值
                
                # 为了确保表达式是线性的，再计算另一个点的值
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (
                        m1.group('before'), 1.5, m1.group('after'))
                    m1 = re_1.match(ee)
                c2 = myeval(ee, {}, {})  # 计算得到另一个点的值
                
                # 如果满足线性关系的条件，则返回 a、b 和 x
                if (a * 0.5 + b == c and a * 1.5 + b == c2):
                    return a, b, x
            except Exception:
                # 捕获任何异常并忽略，继续执行下一个 x
                pass
            break
    # 如果没有找到符合条件的 x，则返回 None
    return None, None, None
# 定义一个正则表达式模式，用于匹配单词（以字母开头，后跟字母、数字、下划线或美元符号）
word_pattern = re.compile(r'\b[a-z][\w$]*\b', re.I)


# 获取变量依赖字典的内部函数
def _get_depend_dict(name, vars, deps):
    # 如果变量名已存在于给定的变量集合中
    if name in vars:
        # 获取变量的依赖列表
        words = vars[name].get('depend', [])

        # 如果变量名同时包含赋值操作符 '=' 且不是字符串类型
        if '=' in vars[name] and not isstring(vars[name]):
            # 遍历通过正则表达式模式匹配出的单词
            for word in word_pattern.findall(vars[name]['=']):
                # 只添加未包含在依赖列表中、且存在于变量集合中的单词
                if word not in words and word in vars and word != name:
                    words.append(word)

        # 对当前依赖列表中的每个单词进行遍历
        for word in words[:]:
            # 递归地获取每个单词的依赖列表，并将其添加到当前单词的依赖列表中
            for w in deps.get(word, []) or _get_depend_dict(word, vars, deps):
                if w not in words:
                    words.append(w)
    else:
        # 如果变量名不存在于变量集合中，则输出一条错误消息
        outmess('_get_depend_dict: no dependence info for %s\n' % (repr(name)))
        words = []

    # 将依赖列表存入依赖字典中
    deps[name] = words
    return words


# 计算所有变量的依赖字典
def _calc_depend_dict(vars):
    names = list(vars.keys())
    depend_dict = {}
    # 遍历所有变量名，并调用_get_depend_dict函数获取其依赖字典
    for n in names:
        _get_depend_dict(n, vars, depend_dict)
    return depend_dict


# 获取按依赖关系排序后的变量名列表
def get_sorted_names(vars):
    # 计算所有变量的依赖字典
    depend_dict = _calc_depend_dict(vars)
    names = []
    # 遍历依赖字典的键（变量名）
    for name in list(depend_dict.keys()):
        # 如果变量没有依赖项，则将其添加到结果列表中并从依赖字典中删除
        if not depend_dict[name]:
            names.append(name)
            del depend_dict[name]

    # 当依赖字典非空时，继续处理
    while depend_dict:
        for name, lst in list(depend_dict.items()):
            # 过滤出仍在依赖字典中存在的依赖项
            new_lst = [n for n in lst if n in depend_dict]
            # 如果没有新的依赖项，则将当前变量添加到结果列表并从依赖字典中删除
            if not new_lst:
                names.append(name)
                del depend_dict[name]
            else:
                depend_dict[name] = new_lst

    # 返回排序后的变量名列表（仅包含在给定变量集合中的变量）
    return [name for name in names if name in vars]


# 内部函数，根据字符串的起始字符和正则表达式模式判断其类型
def _kind_func(string):
    # XXX: 返回一个合理的值。
    if string[0] in "'\"":
        string = string[1:-1]
    # 如果字符串匹配实数类型的正则表达式模式，则返回实数类型
    if real16pattern.match(string):
        return 8
    elif real8pattern.match(string):
        return 4
    # 否则返回带有字符串描述的类型
    return 'kind(' + string + ')'


# 内部函数，根据整数的位数计算其类型
def _selected_int_kind_func(r):
    # XXX: 这应该是与处理器相关的。
    m = 10 ** r
    # 根据整数的大小返回对应的类型
    if m <= 2 ** 8:
        return 1
    if m <= 2 ** 16:
        return 2
    if m <= 2 ** 32:
        return 4
    if m <= 2 ** 63:
        return 8
    if m <= 2 ** 128:
        return 16
    return -1


# 内部函数，根据实数的精度和进制计算其类型
def _selected_real_kind_func(p, r=0, radix=0):
    # XXX: 这应该是与处理器相关的。
    # 这个函数仅针对 0 <= p <= 20 进行验证，对于 p <= 33 及以上可能也适用
    if p < 7:
        return 4
    if p < 16:
        return 8
    # 获取当前机器的处理器信息，并将其转换为小写形式
    machine = platform.machine().lower()
    if machine.startswith(('aarch64', 'alpha', 'arm64', 'loongarch', 'mips', 'power', 'ppc', 'riscv', 's390x', 'sparc')):
        if p <= 33:
            return 16
    else:
        if p < 19:
            return 10
        elif p <= 33:
            return 16
    return -1


# 获取参数列表的函数，包括全局参数和变量集合中的参数
def get_parameters(vars, global_params={}):
    # 复制全局参数到局部参数列表中
    params = copy.copy(global_params)
    g_params = copy.copy(global_params)
    # 遍历元组列表，元组中包含参数名和对应的函数，如 ('kind', _kind_func)
    for name, func in [('kind', _kind_func),
                       ('selected_int_kind', _selected_int_kind_func),
                       ('selected_real_kind', _selected_real_kind_func), ]:
        # 如果当前参数名不在全局参数字典 g_params 中
        if name not in g_params:
            # 将参数名与对应函数添加到全局参数字典 g_params 中
            g_params[name] = func
    
    # 初始化一个空列表，用于存储参数名
    param_names = []
    # 遍历变量字典 vars 中按名称排序的变量名列表
    for n in get_sorted_names(vars):
        # 如果当前变量具有 'attrspec' 属性并且其 'attrspec' 中包含 'parameter' 字段
        if 'attrspec' in vars[n] and 'parameter' in vars[n]['attrspec']:
            # 将当前变量名 n 添加到参数名列表 param_names 中
            param_names.append(n)
    
    # 创建一个正则表达式对象，用于匹配形如 'kind(value)' 的字符串，其中 value 为参数值
    kind_re = re.compile(r'\bkind\s*\(\s*(?P<value>.*)\s*\)', re.I)
    # 创建一个正则表达式对象，用于匹配形如 'selected_int_kind(value)' 的字符串
    selected_int_kind_re = re.compile(
        r'\bselected_int_kind\s*\(\s*(?P<value>.*)\s*\)', re.I)
    # 创建一个正则表达式对象，用于匹配形如 'selected_int_kind(value)' 或 'selected_real_kind(value)' 的字符串
    selected_kind_re = re.compile(
        r'\bselected_(int|real)_kind\s*\(\s*(?P<value>.*)\s*\)', re.I)
    
    # 返回参数列表 params
    return params
# 定义一个函数，用于评估长度信息
def _eval_length(length, params):
    # 如果长度信息是预定义的特殊值之一，返回'(*)'
    if length in ['(:)', '(*)', '*']:
        return '(*)'
    # 否则调用_eval_scalar函数进一步处理
    return _eval_scalar(length, params)


# 使用正则表达式检查是否是以数字结尾的字符串
_is_kind_number = re.compile(r'\d+_').match


# 评估标量值
def _eval_scalar(value, params):
    # 如果值符合数字结尾的格式，则取其前面的部分作为新的值
    if _is_kind_number(value):
        value = value.split('_')[0]
    try:
        # 尝试用参数params对value进行求值
        # 如果支持符号操作，则使用符号进行求值（从PR＃19805中）
        value = eval(value, {}, params)
        # 将值转换为字符串表示形式，或者如果是字符串，则使用其原始表示
        value = (repr if isinstance(value, str) else str)(value)
    except (NameError, SyntaxError, TypeError):
        # 如果发生特定的异常（NameError，SyntaxError，TypeError），返回原始值
        return value
    except Exception as msg:
        # 如果有其他异常，输出错误消息并返回原始值
        errmess('"%s" in evaluating %r '
                '(available names: %s)\n'
                % (msg, value, list(params.keys())))
    # 返回经过评估处理后的值
    return value


# 分析变量信息的函数
def analyzevars(block):
    """
    Sets correct dimension information for each variable/parameter
    """

    # 使用全局变量f90modulevars
    global f90modulevars

    # 设置信息文本
    setmesstext(block)
    # 构建隐式规则和属性规则
    implicitrules, attrrules = buildimplicitrules(block)
    # 复制变量字典
    vars = copy.copy(block['vars'])
    # 如果块类型为函数且名称不在变量字典中，则添加名称到变量字典中
    if block['block'] == 'function' and block['name'] not in vars:
        vars[block['name']] = {}
    # 如果变量字典中包含空键，则删除空键，并根据属性规则设置公共或私有属性
    if '' in block['vars']:
        del vars['']
        if 'attrspec' in block['vars']['']:
            gen = block['vars']['']['attrspec']
            for n in set(vars) | set(b['name'] for b in block['body']):
                for k in ['public', 'private']:
                    if k in gen:
                        vars[n] = setattrspec(vars.get(n, {}), k)
    # 初始化变量列表
    svars = []
    # 获取块的参数列表
    args = block['args']
    # 遍历参数列表，如果参数在变量字典中存在，则加入svars列表
    for a in args:
        try:
            vars[a]
            svars.append(a)
        except KeyError:
            pass
    # 遍历变量字典，将不在参数列表中的变量名加入svars列表
    for n in list(vars.keys()):
        if n not in args:
            svars.append(n)

    # 获取参数字典，包括定义变量时使用的参数
    params = get_parameters(vars, get_useparameters(block))
    # 在此时，params已经被读取和解释，但用于定义变量的参数尚未解析

    # 初始化依赖匹配字典
    dep_matches = {}
    # 匹配变量名的正则表达式
    name_match = re.compile(r'[A-Za-z][\w$]*').match
    # 遍历变量字典的键列表
    for v in list(vars.keys()):
        # 使用正则表达式匹配变量名
        m = name_match(v)
        if m:
            n = v[m.start():m.end()]
            try:
                # 尝试访问依赖匹配字典中的键
                dep_matches[n]
            except KeyError:
                # 如果键不存在，则将变量名作为正则表达式的模式进行匹配
                dep_matches[n] = re.compile(r'.*\b%s\b' % (v), re.I).match
    # 遍历 vars 字典中的键（变量名）
    for n in list(vars.keys()):
        # 检查当前变量名 n 是否与 block 字典中的 'name' 键值相同
        if n == block['name']:  # n is block name
            # 如果 vars[n] 中有 'note' 键，将其赋值给 block 字典中的 'note'
            if 'note' in vars[n]:
                block['note'] = vars[n]['note']
            # 如果 block 字典中的 'block' 键为 'function'
            if block['block'] == 'function':
                # 如果 block 字典中有 'result' 键，并且其值在 vars 中存在
                if 'result' in block and block['result'] in vars:
                    # 将 vars[block['result']] 的内容追加到 vars[n] 中
                    vars[n] = appenddecl(vars[n], vars[block['result']])
                # 如果 block 字典中有 'prefix' 键
                if 'prefix' in block:
                    # 将 block['prefix'] 的值赋给 pr
                    pr = block['prefix']
                    # 将 'pure' 替换为空字符串，判断是否是 pure 函数
                    pr1 = pr.replace('pure', '')
                    ispure = (not pr == pr1)
                    # 将 pr1 中的 'recursive' 替换为空字符串，判断是否是递归函数
                    pr = pr1.replace('recursive', '')
                    isrec = (not pr == pr1)
                    # 使用 typespattern[0] 中的正则表达式匹配 pr
                    m = typespattern[0].match(pr)
                    if m:
                        # 如果匹配成功，解析 typespec
                        typespec, selector, attr, edecl = cracktypespec0(
                            m.group('this'), m.group('after'))
                        # 解析类型选择器
                        kindselect, charselect, typename = cracktypespec(
                            typespec, selector)
                        # 将 typespec 存储到 vars[n]['typespec'] 中
                        vars[n]['typespec'] = typespec
                        try:
                            # 如果 block['result'] 存在，则将 typespec 存储到 vars[block['result']]['typespec'] 中
                            if block['result']:
                                vars[block['result']]['typespec'] = typespec
                        except Exception:
                            pass
                        # 如果 kindselect 存在且包含 'kind' 键
                        if kindselect:
                            if 'kind' in kindselect:
                                try:
                                    # 将 kindselect['kind'] 求值，并存储结果到 kindselect['kind'] 中
                                    kindselect['kind'] = eval(
                                        kindselect['kind'], {}, params)
                                except Exception:
                                    pass
                            # 将 kindselect 存储到 vars[n]['kindselector'] 中
                            vars[n]['kindselector'] = kindselect
                        # 将 charselect 存储到 vars[n]['charselector'] 中
                        if charselect:
                            vars[n]['charselector'] = charselect
                        # 将 typename 存储到 vars[n]['typename'] 中
                        if typename:
                            vars[n]['typename'] = typename
                        # 如果 ispure 为 True，将 'pure' 设置为 vars[n] 的属性
                        if ispure:
                            vars[n] = setattrspec(vars[n], 'pure')
                        # 如果 isrec 为 True，将 'recursive' 设置为 vars[n] 的属性
                        if isrec:
                            vars[n] = setattrspec(vars[n], 'recursive')
                    else:
                        # 如果未能成功匹配，输出错误信息
                        outmess(
                            'analyzevars: prefix (%s) were not used\n' % repr(block['prefix']))
    # 检查块的类型是否为模块、Python 模块、块数据，如果不是，则进行以下操作
    if not block['block'] in ['module', 'pythonmodule', 'python module', 'block data']:
        # 如果块中有 'commonvars' 键，则复制 'args' 和 'commonvars' 的内容到 neededvars
        if 'commonvars' in block:
            neededvars = copy.copy(block['args'] + block['commonvars'])
        else:
            # 否则，只复制 'args' 的内容到 neededvars
            neededvars = copy.copy(block['args'])
        
        # 遍历当前变量字典中的键，对符合 isintent_callback 或 isintent_aux 的变量进行处理
        for n in list(vars.keys()):
            if l_or(isintent_callback, isintent_aux)(vars[n]):
                neededvars.append(n)
        
        # 如果块中有 'entry' 键，则将其所有键合并到 neededvars 中
        if 'entry' in block:
            neededvars.extend(list(block['entry'].keys()))
            # 遍历 entry 中的每个键，将其值加入 neededvars，如果值不在 neededvars 中
            for k in list(block['entry'].keys()):
                for n in block['entry'][k]:
                    if n not in neededvars:
                        neededvars.append(n)
        
        # 如果块的类型为 'function'，则根据是否有 'result' 键或者 'name' 键，决定添加相应的变量到 neededvars
        if block['block'] == 'function':
            if 'result' in block:
                neededvars.append(block['result'])
            else:
                neededvars.append(block['name'])
        
        # 如果块的类型为 'subroutine' 或 'function'，并且块的名字在 vars 中且具有 'intent' 键，则将其赋给 block['intent']
        if block['block'] in ['subroutine', 'function']:
            name = block['name']
            if name in vars and 'intent' in vars[name]:
                block['intent'] = vars[name]['intent']
        
        # 如果块的类型为 'type'，则将当前变量字典中的所有键加入 neededvars
        if block['block'] == 'type':
            neededvars.extend(list(vars.keys()))
        
        # 再次遍历当前变量字典中的键，如果不在 neededvars 中，则删除该键
        for n in list(vars.keys()):
            if n not in neededvars:
                del vars[n]
    
    # 返回处理后的变量字典
    return vars
# 定义正则表达式对象，用于验证参数是否是有效的Fortran标识符
analyzeargs_re_1 = re.compile(r'\A[a-z]+[\w$]*\Z', re.I)

# 定义函数param_eval，用于创建参数数组的索引和值的字典，以便后续评估
def param_eval(v, g_params, params, dimspec=None):
    """
    创建参数数组的索引和值的字典，以便后续评估。

    警告：在这一点上不可能初始化多维数组参数，例如dimension(-3:1, 4, 3:5)。
    这是因为在Fortran中通过数组构造函数初始化需要使用RESHAPE内置函数。
    由于参数声明的右侧在f2py中不会执行，而是在编译后的c/fortran扩展中执行，
    因此无法执行参数数组的reshape。
    一个问题仍然存在：如果用户希望从Python访问数组参数，我们应该
    要么1）允许他们使用Python标准索引访问参数数组（这通常与原始Fortran索引不兼容）
    要么2）允许以Fortran索引为键将参数数组作为字典在Python中访问
    我们暂时选择第2种方式。
    """
    if dimspec is None:
        try:
            # 尝试使用全局参数和局部参数对变量v进行求值
            p = eval(v, g_params, params)
        except Exception as msg:
            # 如果出现异常，记录消息并返回原始字符串v
            p = v
            outmess(f'param_eval: got "{msg}" on {v!r}\n')
        return p

    # 这是一个数组参数。
    # 首先，我们解析维度信息
    if len(dimspec) < 2 or dimspec[::len(dimspec)-1] != "()":
        raise ValueError(f'param_eval: dimension {dimspec} can\'t be parsed')
    dimrange = dimspec[1:-1].split(',')
    if len(dimrange) == 1:
        # 例如dimension(2)或dimension(-1:1)
        dimrange = dimrange[0].split(':')
        # 现在，dimrange是一个包含1个或2个元素的列表
        if len(dimrange) == 1:
            # 如果只有一个元素，解析边界并生成范围
            bound = param_parse(dimrange[0], params)
            dimrange = range(1, int(bound)+1)
        else:
            # 如果有两个元素，解析下限和上限，并生成范围
            lbound = param_parse(dimrange[0], params)
            ubound = param_parse(dimrange[1], params)
            dimrange = range(int(lbound), int(ubound)+1)
    else:
        raise ValueError(f'param_eval: multidimensional array parameters '
                         f'{dimspec} not supported')

    # 解析参数值
    v = (v[2:-2] if v.startswith('(/') else v).split(',')
    v_eval = []
    for item in v:
        try:
            # 尝试使用全局参数和局部参数对项目进行求值
            item = eval(item, g_params, params)
        except Exception as msg:
            outmess(f'param_eval: got "{msg}" on {item!r}\n')
        v_eval.append(item)

    # 创建并返回维度范围和值的字典
    p = dict(zip(dimrange, v_eval))

    return p


def param_parse(d, params):
    """
    递归解析数组维度。

    解析数组变量或参数的声明，使用参数params中的先前定义的参数进行递归调用。

    Parameters
    ----------
    d : str
        描述数组维度的Fortran表达式。
    """
    params : dict
        存储了Fortran源文件中先前解析的参数。

    Returns
    -------
    out : str
        解析后的维度表达式。

    Examples
    --------

    * 如果被分析的行是

      `integer, parameter, dimension(2) :: pa = (/ 3, 5 /)`

      那么 `d = 2`，我们直接返回，结果为

    >>> d = '2'
    >>> param_parse(d, params)
    2

    * 如果被分析的行是

      `integer, parameter, dimension(pa) :: pb = (/1, 2, 3/)`

      那么 `d = 'pa'`；由于 `pa` 是一个先前解析的参数，并且 `pa = 3`，我们递归调用 `param_parse`，得到

    >>> d = 'pa'
    >>> params = {'pa': 3}
    >>> param_parse(d, params)
    3

    * 如果被分析的行是

      `integer, parameter, dimension(pa(1)) :: pb = (/1, 2, 3/)`

      那么 `d = 'pa(1)'`；由于 `pa` 是一个先前解析的参数，并且 `pa(1) = 3`，我们递归调用 `param_parse`，得到

    >>> d = 'pa(1)'
    >>> params = dict(pa={1: 3, 2: 5})
    >>> param_parse(d, params)
    3
    """
    if "(" in d:
        # 如果这个维度表达式是一个数组
        # 提取数组名
        dname = d[:d.find("(")]
        # 提取数组索引表达式
        ddims = d[d.find("(")+1:d.rfind(")")]
        # 如果这个维度表达式也是一个参数；
        # 递归解析它
        index = int(param_parse(ddims, params))
        return str(params[dname][index])
    elif d in params:
        # 如果维度表达式在参数列表中
        return str(params[d])
    else:
        # 否则，尝试替换所有参数名为其对应的值
        for p in params:
            # 创建正则表达式模式以查找参数名
            re_1 = re.compile(
                r'(?P<before>.*?)\b' + p + r'\b(?P<after>.*)', re.I
            )
            # 尝试匹配并替换参数名
            m = re_1.match(d)
            while m:
                d = m.group('before') + \
                    str(params[p]) + m.group('after')
                m = re_1.match(d)
        return d
# 检查给定的表达式是否不符合分析参数的正则表达式规则，返回其是否为表达式的布尔值
def expr2name(a, block, args=[]):
    # 保存原始表达式内容
    orig_a = a
    # 判断表达式是否不符合分析参数的正则表达式规则
    a_is_expr = not analyzeargs_re_1.match(a)
    if a_is_expr:  # `a` is an expression
        # 构建隐式规则和属性规则
        implicitrules, attrrules = buildimplicitrules(block)
        # 确定表达式类型
        at = determineexprtype(a, block['vars'], implicitrules)
        # 构建表达式的名称
        na = 'e_'
        for c in a:
            c = c.lower()
            if c not in string.ascii_lowercase + string.digits:
                c = '_'
            na = na + c
        # 确保名称以 'e' 结尾
        if na[-1] == '_':
            na = na + 'e'
        else:
            na = na + '_e'
        a = na
        # 确保新的表达式名称不与现有变量或参数冲突
        while a in block['vars'] or a in block['args']:
            a = a + 'r'
    # 如果表达式在参数列表中，则进行重命名处理
    if a in args:
        k = 1
        while a + str(k) in args:
            k = k + 1
        a = a + str(k)
    # 如果表达式是一个新的表达式，将其添加到块的变量字典中
    if a_is_expr:
        block['vars'][a] = at
    else:
        # 如果表达式不是一个新的表达式，则根据其原始名称决定如何处理
        if a not in block['vars']:
            if orig_a in block['vars']:
                block['vars'][a] = block['vars'][orig_a]
            else:
                block['vars'][a] = {}
        # 如果表达式在外部变量或接口中，则设置其属性为 'external'
        if 'externals' in block and orig_a in block['externals'] + block['interfaced']:
            block['vars'][a] = setattrspec(block['vars'][a], 'external')
    # 返回处理后的表达式名称
    return a


# 分析块中的参数，设置消息文本，构建隐式规则，并对参数列表中的每个参数应用表达式处理
def analyzeargs(block):
    # 设置消息文本
    setmesstext(block)
    # 构建隐式规则，忽略属性规则
    implicitrules, _ = buildimplicitrules(block)
    # 如果块中不存在参数列表，则创建一个空列表
    if 'args' not in block:
        block['args'] = []
    # 保存处理后的参数列表
    args = []
    # 对参数列表中的每个参数应用表达式处理，更新参数列表
    for a in block['args']:
        a = expr2name(a, block, args)
        args.append(a)
    block['args'] = args
    # 如果块中存在 'entry' 键，则处理其值
    if 'entry' in block:
        for k, args1 in list(block['entry'].items()):
            for a in args1:
                # 如果参数不在变量字典中，则将其添加到变量字典
                if a not in block['vars']:
                    block['vars'][a] = {}

    # 遍历块中的主体列表，检查参数是否在参数列表中，若不在则添加到外部变量列表中
    for b in block['body']:
        if b['name'] in args:
            if 'externals' not in block:
                block['externals'] = []
            if b['name'] not in block['externals']:
                block['externals'].append(b['name'])
    # 如果块中存在 'result' 键且其值不在变量字典中，则将其添加到变量字典中
    if 'result' in block and block['result'] not in block['vars']:
        block['vars'][block['result']] = {}
    # 返回处理后的块
    return block


# 编译正则表达式，用于识别特定类型的表达式
determineexprtype_re_1 = re.compile(r'\A\(.+?,.+?\)\Z', re.I)
determineexprtype_re_2 = re.compile(r'\A[+-]?\d+(_(?P<name>\w+)|)\Z', re.I)
determineexprtype_re_3 = re.compile(
    r'\A[+-]?[\d.]+[-\d+de.]*(_(?P<name>\w+)|)\Z', re.I)
determineexprtype_re_4 = re.compile(r'\A\(.*\)\Z', re.I)
determineexprtype_re_5 = re.compile(r'\A(?P<name>\w+)\s*\(.*?\)\s*\Z', re.I)


# 确保表达式类型的字典返回格式正确
def _ensure_exprdict(r):
    if isinstance(r, int):
        return {'typespec': 'integer'}
    if isinstance(r, float):
        return {'typespec': 'real'}
    if isinstance(r, complex):
        return {'typespec': 'complex'}
    if isinstance(r, dict):
        return r
    # 如果类型不在预期范围内，则引发断言错误
    raise AssertionError(repr(r))


# 确定给定表达式的类型，根据表达式本身或者已知的变量和规则
def determineexprtype(expr, vars, rules={}):
    # 如果表达式已经在变量字典中，则直接返回其类型字典
    if expr in vars:
        return _ensure_exprdict(vars[expr])
    # 去除表达式两端的空白字符
    expr = expr.strip()
    # 根据不同的正则表达式匹配类型，返回相应的类型字典
    if determineexprtype_re_1.match(expr):
        return {'typespec': 'complex'}
    m = determineexprtype_re_2.match(expr)
    # 如果正则表达式对象m匹配成功
    if m:
        # 检查匹配结果中是否有'name'字段，并且'name'字段不为空
        if 'name' in m.groupdict() and m.group('name'):
            # 输出警告信息，指示选择的类型不受支持
            outmess(
                'determineexprtype: selected kind types not supported (%s)\n' % repr(expr))
        # 返回一个包含类型为整数的字典
        return {'typespec': 'integer'}
    
    # 使用正则表达式对象determineexprtype_re_3尝试匹配表达式expr
    m = determineexprtype_re_3.match(expr)
    if m:
        # 再次检查'm'对象中是否有'name'字段，并且'name'字段不为空
        if 'name' in m.groupdict() and m.group('name'):
            # 输出警告信息，指示选择的类型不受支持
            outmess(
                'determineexprtype: selected kind types not supported (%s)\n' % repr(expr))
        # 返回一个包含类型为实数的字典
        return {'typespec': 'real'}
    
    # 遍历运算符列表['+', '-', '*', '/']
    for op in ['+', '-', '*', '/']:
        # 使用markoutercomma函数将表达式expr根据当前运算符op进行分割和处理
        for e in [x.strip() for x in markoutercomma(expr, comma=op).split('@' + op + '@')]:
            # 如果处理后的子表达式e在变量vars中
            if e in vars:
                # 返回变量e对应的表达式字典
                return _ensure_exprdict(vars[e])
    
    # 初始化空字典t
    t = {}
    
    # 如果表达式expr匹配determineexprtype_re_4，即位于括号内部
    if determineexprtype_re_4.match(expr):
        # 对expr去除外层括号后的部分进行类型推断
        t = determineexprtype(expr[1:-1], vars, rules)
    else:
        # 尝试使用determineexprtype_re_5匹配表达式expr
        m = determineexprtype_re_5.match(expr)
        if m:
            # 获取匹配结果中的'name'字段值
            rn = m.group('name')
            # 对'name'字段值进行类型推断
            t = determineexprtype(m.group('name'), vars, rules)
            # 如果推断结果中包含'attrspec'字段，则删除该字段
            if t and 'attrspec' in t:
                del t['attrspec']
            # 如果推断结果为空
            if not t:
                # 如果'name'字段值的首字符在规则集rules中
                if rn[0] in rules:
                    # 返回rules中首字符对应的表达式字典
                    return _ensure_exprdict(rules[rn[0]])
    
    # 如果表达式expr的第一个字符为单引号或双引号
    if expr[0] in '\'"':
        # 返回一个包含类型为字符的字典，并指定字符选择器为'*'
        return {'typespec': 'character', 'charselector': {'*': '*'}}
    
    # 如果t仍为空字典
    if not t:
        # 输出警告信息，指示无法确定表达式expr的类型
        outmess(
            'determineexprtype: could not determine expressions (%s) type.\n' % (repr(expr)))
    
    # 返回类型推断结果字典t
    return t
######
# 定义一个名为 crack2fortrangen 的函数，用于处理给定的代码块并生成对应的 Fortran 代码段
def crack2fortrangen(block, tab='\n', as_interface=False):
    global skipfuncs, onlyfuncs  # 声明使用全局变量 skipfuncs 和 onlyfuncs

    setmesstext(block)  # 调用 setmesstext 函数处理 block 参数
    ret = ''  # 初始化一个空字符串 ret 用于存储生成的代码段
    if isinstance(block, list):  # 如果 block 是一个列表
        for g in block:  # 遍历列表中的每个元素 g
            if g and g['block'] in ['function', 'subroutine']:  # 如果 g 是函数或子程序块
                if g['name'] in skipfuncs:  # 如果函数名在 skipfuncs 中，跳过处理
                    continue
                if onlyfuncs and g['name'] not in onlyfuncs:  # 如果 onlyfuncs 存在且函数名不在其中，跳过处理
                    continue
            ret = ret + crack2fortrangen(g, tab, as_interface=as_interface)  # 递归调用 crack2fortrangen 处理当前元素 g
        return ret  # 返回生成的代码段字符串
    prefix = ''  # 初始化前缀字符串为空
    name = ''  # 初始化名称字符串为空
    args = ''  # 初始化参数字符串为空
    blocktype = block['block']  # 获取 block 的类型
    if blocktype == 'program':  # 如果 block 类型为 'program'
        return ''  # 返回空字符串，不生成代码段
    argsl = []  # 初始化参数列表为空列表
    if 'name' in block:  # 如果 block 中有 'name' 键
        name = block['name']  # 获取 block 的名称
    if 'args' in block:  # 如果 block 中有 'args' 键
        vars = block['vars']  # 获取 block 的变量
        for a in block['args']:  # 遍历参数列表
            a = expr2name(a, block, argsl)  # 调用 expr2name 处理参数 a，并添加到 argsl 列表
            if not isintent_callback(vars[a]):  # 如果不是意图回调函数
                argsl.append(a)  # 将参数 a 添加到 argsl 列表
        if block['block'] == 'function' or argsl:  # 如果 block 类型是 'function' 或 argsl 不为空
            args = '(%s)' % ','.join(argsl)  # 格式化参数列表为字符串形式
    f2pyenhancements = ''  # 初始化 f2pyenhancements 字符串为空
    if 'f2pyenhancements' in block:  # 如果 block 中有 'f2pyenhancements' 键
        for k in list(block['f2pyenhancements'].keys()):  # 遍历 f2pyenhancements 键的所有键
            f2pyenhancements = '%s%s%s %s' % (  # 格式化 f2pyenhancements 字符串
                f2pyenhancements, tab + tabchar, k, block['f2pyenhancements'][k])
    intent_lst = block.get('intent', [])[:]  # 获取 block 中的 'intent' 键，若不存在则为空列表
    if blocktype == 'function' and 'callback' in intent_lst:  # 如果 block 类型是 'function' 并且 'callback' 在 intent_lst 中
        intent_lst.remove('callback')  # 移除 'callback' 从 intent_lst 中
    if intent_lst:  # 如果 intent_lst 不为空
        f2pyenhancements = '%s%sintent(%s) %s' %\  # 格式化 f2pyenhancements 字符串
                           (f2pyenhancements, tab + tabchar,
                            ','.join(intent_lst), name)
    use = ''  # 初始化 use 字符串为空
    if 'use' in block:  # 如果 block 中有 'use' 键
        use = use2fortran(block['use'], tab + tabchar)  # 调用 use2fortran 处理 'use' 键的内容
    common = ''  # 初始化 common 字符串为空
    if 'common' in block:  # 如果 block 中有 'common' 键
        common = common2fortran(block['common'], tab + tabchar)  # 调用 common2fortran 处理 'common' 键的内容
    if name == 'unknown_interface':  # 如果名称为 'unknown_interface'
        name = ''  # 将名称置空
    result = ''  # 初始化结果字符串为空
    if 'result' in block:  # 如果 block 中有 'result' 键
        result = ' result (%s)' % block['result']  # 设置结果字符串为 ' result (result)' 格式
        if block['result'] not in argsl:  # 如果结果不在参数列表中
            argsl.append(block['result'])  # 将结果添加到参数列表中
    body = crack2fortrangen(block['body'], tab + tabchar, as_interface=as_interface)  # 递归调用 crack2fortrangen 处理 'body' 键的内容
    vars = vars2fortran(  # 调用 vars2fortran 处理 block、block['vars']、argsl 等参数
        block, block['vars'], argsl, tab + tabchar, as_interface=as_interface)
    mess = ''  # 初始化消息字符串为空
    if 'from' in block and not as_interface:  # 如果 block 中有 'from' 键且不是作为接口处理
        mess = '! in %s' % block['from']  # 格式化消息字符串
    if 'entry' in block:  # 如果 block 中有 'entry' 键
        entry_stmts = ''  # 初始化 entry_stmts 字符串为空
        for k, i in list(block['entry'].items()):  # 遍历 entry 键的所有项
            entry_stmts = '%s%sentry %s(%s)' \  # 格式化 entry_stmts 字符串
                          % (entry_stmts, tab + tabchar, k, ','.join(i))
        body = body + entry_stmts  # 将 entry_stmts 添加到 body 字符串末尾
    if blocktype == 'block data' and name == '_BLOCK_DATA_':  # 如果 block 类型是 'block data' 并且名称为 '_BLOCK_DATA_'
        name = ''  # 将名称置空
    ret = '%s%s%s %s%s%s %s%s%s%s%s%s%send %s %s' % (  # 格式化生成的代码段字符串
        tab, prefix, blocktype, name, args, result, mess, f2pyenhancements, use, vars, common, body, tab, blocktype, name)
    return ret  # 返回生成的代码段字符串


# 定义一个名为 common2fortran 的函数，用于处理给定的 'common' 键内容并生成对应的 Fortran 代码段
def common2fortran(common, tab=''):
    ret = ''  # 初始化返回字符串为空
    # 遍历 common 字典的键列表
    for k in list(common.keys()):
        # 如果键为 '_BLNK_'，则格式化字符串添加到ret中
        if k == '_BLNK_':
            ret = '%s%scommon %s' % (ret, tab, ','.join(common[k]))
        # 如果键不为 '_BLNK_'，则格式化字符串添加到ret中
        else:
            ret = '%s%scommon /%s/ %s' % (ret, tab, k, ','.join(common[k]))
    # 返回结果字符串ret
    return ret
# 定义一个函数，生成用于Fortran代码中USE语句的字符串表示
def use2fortran(use, tab=''):
    ret = ''
    # 遍历use字典的键（可能是模块名）
    for m in list(use.keys()):
        # 拼接use语句的一部分，格式为 'use 模块名,'
        ret = '%s%suse %s,' % (ret, tab, m)
        # 如果模块对应的值是空字典，则跳过
        if use[m] == {}:
            if ret and ret[-1] == ',':
                ret = ret[:-1]
            continue
        # 如果模块对应的值中有'only'键，并且值为真，则添加 'only:' 到字符串中
        if 'only' in use[m] and use[m]['only']:
            ret = '%s only:' % (ret)
        # 如果模块对应的值中有'map'键，并且值为真，则生成映射关系字符串
        if 'map' in use[m] and use[m]['map']:
            c = ' '
            for k in list(use[m]['map'].keys()):
                if k == use[m]['map'][k]:
                    ret = '%s%s%s' % (ret, c, k)
                    c = ','
                else:
                    ret = '%s%s%s=>%s' % (ret, c, k, use[m]['map'][k])
                    c = ','
        # 如果字符串最后一个字符是逗号，则去除该逗号
        if ret and ret[-1] == ',':
            ret = ret[:-1]
    # 返回最终生成的use语句字符串
    return ret


# 根据变量的'intent'属性返回真实的意图列表
def true_intent_list(var):
    lst = var['intent']
    ret = []
    # 遍历意图列表
    for intent in lst:
        try:
            # 尝试获取全局函数'isintent_意图'
            f = globals()['isintent_%s' % intent]
        except KeyError:
            pass
        else:
            # 如果全局函数存在且返回真，则将意图添加到结果列表中
            if f(var):
                ret.append(intent)
    # 返回结果意图列表
    return ret


# 将变量转换为Fortran代码中的声明语句
def vars2fortran(block, vars, args, tab='', as_interface=False):
    # 设置消息文本
    setmesstext(block)
    ret = ''
    nout = []
    # 遍历参数列表，将在块中定义的变量添加到nout列表中
    for a in args:
        if a in block['vars']:
            nout.append(a)
    # 如果块中定义了'commonvars'，则将其添加到nout列表中
    if 'commonvars' in block:
        for a in block['commonvars']:
            if a in vars:
                if a not in nout:
                    nout.append(a)
            else:
                errmess(
                    'vars2fortran: Confused?!: "%s" is not defined in vars.\n' % a)
    # 如果块中定义了'varnames'，则将其添加到nout列表中
    if 'varnames' in block:
        nout.extend(block['varnames'])
    # 如果不是作为接口（interface）使用，则将所有变量添加到nout列表中
    if not as_interface:
        for a in list(vars.keys()):
            if a not in nout:
                nout.append(a)
    # 返回空字符串
    return ret


# 全局变量，用于存储后处理钩子函数
post_processing_hooks = []


# 解析Fortran代码文件
def crackfortran(files):
    global usermodules, post_processing_hooks

    # 输出消息，指示正在读取Fortran代码
    outmess('Reading fortran codes...\n', 0)
    # 读取Fortran代码文件并进行处理
    readfortrancode(files, crackline)
    # 输出消息，指示正在进行后处理
    outmess('Post-processing...\n', 0)
    # 初始化用户模块列表
    usermodules = []
    # 对组列表中的第一个组进行后处理
    postlist = postcrack(grouplist[0])
    # 输出消息，指示正在应用后处理钩子函数
    outmess('Applying post-processing hooks...\n', 0)
    # 遍历后处理钩子函数列表，对后处理结果进行遍历处理
    for hook in post_processing_hooks:
        outmess(f'  {hook.__name__}\n', 0)
        postlist = traverse(postlist, hook)
    # 输出消息，指示正在进行第二阶段后处理
    outmess('Post-processing (stage 2)...\n', 0)
    # 对第一阶段后处理结果再进行后处理
    postlist = postcrack2(postlist)
    # 返回用户模块列表与最终后处理结果列表的组合
    return usermodules + postlist


# 将块转换为Fortran代码
def crack2fortran(block):
    global f2py_version

    # 生成Fortran代码
    pyf = crack2fortrangen(block) + '\n'
    # 设置头部注释
    header = """!    -*- f90 -*-
! Note: the context of this file is case sensitive.
"""
    # 设置尾部注释
    footer = """
! This file was auto-generated with f2py (version:%s).
! See:
! https://web.archive.org/web/20140822061353/http://cens.ioc.ee/projects/f2py2e
""" % (f2py_version)
    # 返回整合了头部、生成代码和尾部的字符串
    return header + pyf + footer


# 检查对象是否是访问对
def _is_visit_pair(obj):
    return (isinstance(obj, tuple)
            and len(obj) == 2
            and isinstance(obj[0], (int, str)))
# 定义一个函数 traverse，用于遍历 f2py 数据结构，并调用指定的 visit 函数处理每个项
def traverse(obj, visit, parents=[], result=None, *args, **kwargs):
    '''
    Traverse f2py data structure with the following visit function:

    def visit(item, parents, result, *args, **kwargs):
        """
        parents is a list of key-"f2py data structure" pairs from which
        items are taken from.

        result is a f2py data structure that is filled with the
        return value of the visit function.

        item is 2-tuple (index, value) if parents[-1][1] is a list
        item is 2-tuple (key, value) if parents[-1][1] is a dict

        The return value of visit must be None, or of the same kind as
        item, that is, if parents[-1] is a list, the return value must
        be 2-tuple (new_index, new_value), or if parents[-1] is a
        dict, the return value must be 2-tuple (new_key, new_value).

        If new_index or new_value is None, the return value of visit
        is ignored, that is, it will not be added to the result.

        If the return value is None, the content of obj will be
        traversed, otherwise not.
        """
    '''

    # 如果 obj 是一个访问对，即 (index, value) 或 (key, value)
    if _is_visit_pair(obj):
        # 针对特定情况 'parent_block'，避免无限递归
        if obj[0] == 'parent_block':
            return obj
        # 调用 visit 函数处理当前对象 obj，并获取处理后的结果
        new_result = visit(obj, parents, result, *args, **kwargs)
        # 如果处理结果不为 None，则返回处理结果
        if new_result is not None:
            assert _is_visit_pair(new_result)
            return new_result
        # 更新 parent 和 obj，以备后续遍历使用
        parent = obj
        result_key, obj = obj
    else:
        # 如果 obj 不是访问对，创建一个虚拟的 parent
        parent = (None, obj)
        result_key = None

    # 根据 obj 的类型进行遍历处理
    if isinstance(obj, list):
        new_result = []
        # 遍历列表 obj 中的每个元素
        for index, value in enumerate(obj):
            # 递归调用 traverse 处理每个元素，并获取处理后的结果
            new_index, new_item = traverse((index, value), visit,
                                           parents=parents + [parent],
                                           result=result, *args, **kwargs)
            # 如果处理结果不为 None，则将其加入新的结果列表 new_result 中
            if new_index is not None:
                new_result.append(new_item)
    elif isinstance(obj, dict):
        new_result = dict()
        # 遍历字典 obj 中的每个键值对
        for key, value in obj.items():
            # 递归调用 traverse 处理每个键值对，并获取处理后的结果
            new_key, new_value = traverse((key, value), visit,
                                          parents=parents + [parent],
                                          result=result, *args, **kwargs)
            # 如果处理结果不为 None，则将其加入新的结果字典 new_result 中
            if new_key is not None:
                new_result[new_key] = new_value
    else:
        # 如果 obj 类型既不是列表也不是字典，直接将 obj 作为新结果
        new_result = obj

    # 如果 result_key 为 None，则返回新结果 new_result
    if result_key is None:
        return new_result
    # 否则，返回包含 result_key 的结果键值对
    return result_key, new_result


def character_backward_compatibility_hook(item, parents, result,
                                          *args, **kwargs):
    """
    Previously, Fortran character was incorrectly treated as
    character*1. This hook fixes the usage of the corresponding
    variables in `check`, `dimension`, `=`, and `callstatement`
    expressions.

    The usage of `char*` in `callprotoargument` expression can be left
    unchanged because C `character` is C typedef of `char`, although,
    new implementations should use `character*` in the corresponding
    expressions.
    """
    # 从父级列表中获取最后一个元素的键和值
    parent_key, parent_value = parents[-1]
    # 获取当前项的键和值
    key, value = item

    # 定义一个函数，用于修复变量使用方式中的特定格式问题
    def fix_usage(varname, value):
        # 替换形如 `* varname` 的格式为 varname
        value = re.sub(r'[*]\s*\b' + varname + r'\b', varname, value)
        # 替换形如 `varname [0]` 的格式为 varname
        value = re.sub(r'\b' + varname + r'\b\s*[\[]\s*0\s*[\]]',
                       varname, value)
        return value

    # 根据父级键的不同情况确定 vars_dict 的来源
    if parent_key in ['dimension', 'check']:
        # 确保父级列表倒数第三个元素的键为 'vars'
        assert parents[-3][0] == 'vars'
        # 获取 'vars' 对应的字典
        vars_dict = parents[-3][1]
    elif key == '=':
        # 确保父级列表倒数第二个元素的键为 'vars'
        assert parents[-2][0] == 'vars'
        # 获取 'vars' 对应的字典
        vars_dict = parents[-2][1]
    else:
        # 若以上条件均不满足，则 vars_dict 为空
        vars_dict = None

    # 初始化新值为 None
    new_value = None
    # 若 vars_dict 不为空，则进行以下处理
    if vars_dict is not None:
        # 将 new_value 初始化为当前值
        new_value = value
        # 遍历 vars_dict 中的每个变量名及其对应的值描述
        for varname, vd in vars_dict.items():
            # 如果 vd 被认为是字符类型
            if ischaracter(vd):
                # 修正 new_value 中的变量使用方式
                new_value = fix_usage(varname, new_value)
    elif key == 'callstatement':
        # 获取父级列表倒数第二个元素中的 'vars' 字典
        vars_dict = parents[-2][1]['vars']
        # 将 new_value 初始化为当前值
        new_value = value
        # 遍历 vars_dict 中的每个变量名及其对应的值描述
        for varname, vd in vars_dict.items():
            # 如果 vd 被认为是字符类型
            if ischaracter(vd):
                # 在参数传递中，替换所有 `<varname>` 的出现为 `&<varname>`
                new_value = re.sub(
                    r'(?<![&])\b' + varname + r'\b', '&' + varname, new_value)

    # 如果 new_value 不为 None，则进行以下处理
    if new_value is not None:
        # 如果 new_value 不等于原始值 value，则报告替换信息
        outmess(f'character_bc_hook[{parent_key}.{key}]:'
                f' replaced `{value}` -> `{new_value}`\n', 1)
        # 返回键和新值的元组
        return (key, new_value)
# 将一个函数添加到后处理钩子列表中，用于向后兼容角色
post_processing_hooks.append(character_backward_compatibility_hook)

if __name__ == "__main__":
    # 初始化空文件和函数列表，以及几个标志位
    files = []
    funcs = []
    f = 1
    f2 = 0
    f3 = 0
    showblocklist = 0
    
    # 遍历命令行参数
    for l in sys.argv[1:]:
        if l == '':
            pass  # 忽略空参数
        elif l[0] == ':':
            f = 0  # 标记为不读取文件
        elif l == '-quiet':
            quiet = 1  # 设置静默模式
            verbose = 0  # 取消详细模式
        elif l == '-verbose':
            verbose = 2  # 设置详细输出模式
            quiet = 0  # 取消静默模式
        elif l == '-fix':
            if strictf77:
                outmess(
                    'Use option -f90 before -fix if Fortran 90 code is in fix form.\n', 0)
            skipemptyends = 1  # 跳过空结束行
            sourcecodeform = 'fix'  # 设置源代码形式为固定格式
        elif l == '-skipemptyends':
            skipemptyends = 1  # 跳过空结束行
        elif l == '--ignore-contains':
            ignorecontains = 1  # 忽略包含关系检查
        elif l == '-f77':
            strictf77 = 1  # 使用严格的Fortran 77模式
            sourcecodeform = 'fix'  # 设置源代码形式为固定格式
        elif l == '-f90':
            strictf77 = 0  # 使用自由格式的Fortran 90模式
            sourcecodeform = 'free'  # 设置源代码形式为自由格式
            skipemptyends = 1  # 跳过空结束行
        elif l == '-h':
            f2 = 1  # 标记为读取pyf文件名
        elif l == '-show':
            showblocklist = 1  # 显示块列表
        elif l == '-m':
            f3 = 1  # 标记为读取f77模块名
        elif l[0] == '-':
            errmess('Unknown option %s\n' % repr(l))  # 报错，未知选项
        elif f2:
            f2 = 0  # 取消读取pyf文件名标记
            pyffilename = l  # 设置pyf文件名
        elif f3:
            f3 = 0  # 取消读取f77模块名标记
            f77modulename = l  # 设置f77模块名
        elif f:
            try:
                open(l).close()  # 尝试打开文件
                files.append(l)  # 将文件名添加到文件列表中
            except OSError as detail:
                errmess(f'OSError: {detail!s}\n')  # 报错，显示具体的OSError信息
        else:
            funcs.append(l)  # 将函数名添加到函数列表中
            
    # 检查非严格F77模式下是否设置了f77模块名且未跳过空结束行
    if not strictf77 and f77modulename and not skipemptyends:
        outmess("""\
  Warning: You have specified module name for non Fortran 77 code that
  should not need one (expect if you are scanning F90 code for non
  module blocks but then you should use flag -skipemptyends and also
  be sure that the files do not contain programs without program
  statement).
""", 0)  # 输出警告信息
    
    # 对Fortran文件进行解析得到后处理列表
    postlist = crackfortran(files)
    
    # 如果指定了pyf文件名，则将Fortran代码写入文件
    if pyffilename:
        outmess('Writing fortran code to file %s\n' % repr(pyffilename), 0)
        pyf = crack2fortran(postlist)
        with open(pyffilename, 'w') as f:
            f.write(pyf)  # 将生成的Fortran代码写入文件
    
    # 如果设置了-show标志，则显示解析后的块列表
    if showblocklist:
        show(postlist)  # 显示解析后的块列表
```