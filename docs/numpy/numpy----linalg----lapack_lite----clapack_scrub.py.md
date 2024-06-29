# `.\numpy\numpy\linalg\lapack_lite\clapack_scrub.py`

```
#!/usr/bin/env python2.7
# WARNING! This a Python 2 script. Read README.rst for rationale.
# 引入必要的模块和库
import os            # 导入操作系统功能模块
import re            # 导入正则表达式模块
import sys           # 导入系统相关的功能模块

# 从第三方库中导入需要的函数和类
from plex import Scanner, Str, Lexicon, Opt, Bol, State, AnyChar, TEXT, IGNORE
from plex.traditional import re as Re

try:
    from io import BytesIO as UStringIO  # 尝试在Python 2中导入BytesIO
except ImportError:
    from io import StringIO as UStringIO  # Python 3中导入StringIO替代BytesIO

# 自定义Scanner类，继承自plex的Scanner类
class MyScanner(Scanner):
    def __init__(self, info, name='<default>'):
        Scanner.__init__(self, self.lexicon, info, name)

    def begin(self, state_name):
        Scanner.begin(self, state_name)

# 函数：生成由给定序列和分隔符构成的模式
def sep_seq(sequence, sep):
    pat = Str(sequence[0])
    for s in sequence[1:]:
        pat += sep + Str(s)
    return pat

# 函数：运行扫描器，处理输入数据并返回处理结果和扫描器对象
def runScanner(data, scanner_class, lexicon=None):
    info = UStringIO(data)
    outfo = UStringIO()
    if lexicon is not None:
        scanner = scanner_class(lexicon, info)
    else:
        scanner = scanner_class(info)
    while True:
        value, text = scanner.read()
        if value is None:
            break
        elif value is IGNORE:
            pass
        else:
            outfo.write(value)
    return outfo.getvalue(), scanner

# 自定义的LenSubsScanner类，继承自MyScanner类
class LenSubsScanner(MyScanner):
    """Following clapack, we remove ftnlen arguments, which f2c puts after
    a char * argument to hold the length of the passed string. This is just
    a nuisance in C.
    """
    def __init__(self, info, name='<ftnlen>'):
        MyScanner.__init__(self, info, name)
        self.paren_count = 0

    # 进入参数处理状态
    def beginArgs(self, text):
        if self.paren_count == 0:
            self.begin('args')
        self.paren_count += 1
        return text

    # 结束参数处理状态
    def endArgs(self, text):
        self.paren_count -= 1
        if self.paren_count == 0:
            self.begin('')
        return text

    # 定义各种正则表达式模式
    digits = Re('[0-9]+')
    iofun = Re(r'\([^;]*;')
    decl = Re(r'\([^)]*\)[,;'+'\n]')
    any = Re('[.]*')
    S = Re('[ \t\n]*')
    cS = Str(',') + S
    len_ = Re('[a-z][a-z0-9]*_len')

    # 定义保留不移除ftnlen参数的函数名
    iofunctions = Str("s_cat", "s_copy", "s_stop", "s_cmp",
                      "i_len", "do_fio", "do_lio") + iofun

    # 定义需要保留ftnlen参数的函数的词法规则
    keep_ftnlen = (Str('ilaenv_') | Str('iparmq_') | Str('s_rnge')) + Str('(')

    # 定义扫描器的词法规则集合
    lexicon = Lexicon([
        (iofunctions,                           TEXT),
        (keep_ftnlen,                           beginArgs),
        State('args', [
            (Str(')'),   endArgs),
            (Str('('),   beginArgs),
            (AnyChar,    TEXT),
        ]),
        (cS+Re(r'[1-9][0-9]*L'),                IGNORE),
        (cS+Str('ftnlen')+Opt(S+len_),          IGNORE),
        (cS+sep_seq(['(', 'ftnlen', ')'], S)+S+digits,      IGNORE),
        (Bol+Str('ftnlen ')+len_+Str(';\n'),    IGNORE),
        (cS+len_,                               TEXT),
        (AnyChar,                               TEXT),
    ])

# 函数：移除源代码中的ftnlen参数
def scrubFtnlen(source):
    return runScanner(source, LenSubsScanner)[0]

# 函数：清理源代码
def cleanSource(source):
    # 移除每行末尾的空白字符
    source = re.sub(r'[\t ]+\n', '\n', source)
    # 移除类似于 ".. Scalar Arguments .." 的注释
    source = re.sub(r'(?m)^[\t ]*/\* *\.\. .*?\n', '', source)
    # 将连续超过两个空行合并为两个空行
    source = re.sub(r'\n\n\n\n+', r'\n\n\n', source)
    # 返回处理后的源代码
    return source
# 表示一个队列，用于存储文本行
class LineQueue:
    # 初始化 LineQueue 对象
    def __init__(self):
        object.__init__(self)
        self._queue = []

    # 向队列中添加一行文本
    def add(self, line):
        self._queue.append(line)

    # 清空队列
    def clear(self):
        self._queue = []

    # 将队列中的内容刷新到另一个队列中
    def flushTo(self, other_queue):
        # 遍历自身队列中的每一行，添加到另一个队列中
        for line in self._queue:
            other_queue.add(line)
        # 清空自身队列
        self.clear()

    # 获取队列中所有行的字符串表示
    def getValue(self):
        q = LineQueue()
        # 刷新到新的队列 q 中
        self.flushTo(q)
        # 将队列中的所有行连接成一个字符串
        s = ''.join(q._queue)
        # 清空当前队列
        self.clear()
        # 返回拼接后的字符串
        return s

# 表示用于存储注释行的特殊队列，继承自 LineQueue
class CommentQueue(LineQueue):
    # 初始化 CommentQueue 对象
    def __init__(self):
        LineQueue.__init__(self)

    # 向队列中添加一行注释
    def add(self, line):
        # 如果是空行，则添加一个换行符到队列中
        if line.strip() == '':
            LineQueue.add(self, '\n')
        else:
            # 否则，处理注释行格式，去掉开头结尾的多余部分，添加到队列中
            line = '  ' + line[2:-3].rstrip() + '\n'
            LineQueue.add(self, line)

    # 将注释队列刷新到另一个队列中
    def flushTo(self, other_queue):
        # 如果注释队列为空，则不执行任何操作
        if len(self._queue) == 0:
            pass
        # 如果只有一行注释，则将其格式化后添加到目标队列中
        elif len(self._queue) == 1:
            other_queue.add('/*' + self._queue[0][2:].rstrip() + ' */\n')
        else:
            # 否则，添加起始注释标记，然后刷新到目标队列，最后添加结束注释标记
            other_queue.add('/*\n')
            LineQueue.flushTo(self, other_queue)
            other_queue.add('*/\n')
        # 清空当前注释队列
        self.clear()

# 清理源代码中的注释，返回没有注释的源代码
def cleanComments(source):
    lines = LineQueue()
    comments = CommentQueue()

    # 判断是否为注释行
    def isCommentLine(line):
        return line.startswith('/*') and line.endswith('*/\n')

    blanks = LineQueue()

    # 判断是否为空行
    def isBlank(line):
        return line.strip() == ''

    # 处理源代码行为注释行
    def SourceLines(line):
        if isCommentLine(line):
            comments.add(line)
            return HaveCommentLines
        else:
            lines.add(line)
            return SourceLines

    # 处理存在注释行的状态
    def HaveCommentLines(line):
        if isBlank(line):
            blanks.add('\n')
            return HaveBlankLines
        elif isCommentLine(line):
            comments.add(line)
            return HaveCommentLines
        else:
            comments.flushTo(lines)
            lines.add(line)
            return SourceLines

    # 处理存在空行的状态
    def HaveBlankLines(line):
        if isBlank(line):
            blanks.add('\n')
            return HaveBlankLines
        elif isCommentLine(line):
            blanks.flushTo(comments)
            comments.add(line)
            return HaveCommentLines
        else:
            comments.flushTo(lines)
            blanks.flushTo(lines)
            lines.add(line)
            return SourceLines

    # 初始状态为处理源代码行
    state = SourceLines
    for line in UStringIO(source):
        state = state(line)
    
    # 将剩余的注释刷新到源代码行队列中，返回整理后的源代码字符串
    comments.flushTo(lines)
    return lines.getValue()

# 从源代码中移除头部信息
def removeHeader(source):
    lines = LineQueue()

    # 判断是否处于头部信息中
    def LookingForHeader(line):
        m = re.match(r'/\*[^\n]*-- translated', line)
        if m:
            return InHeader
        else:
            lines.add(line)
            return LookingForHeader

    # 处理头部信息中
    def InHeader(line):
        if line.startswith('*/'):
            return OutOfHeader
        else:
            return InHeader
    # 定义一个函数 OutOfHeader，用于从源代码中去除特定的头文件包含语句
    def OutOfHeader(line):
        # 检查当前行是否以 '#include "f2c.h"' 开头
        if line.startswith('#include "f2c.h"'):
            # 如果是，则跳过该行
            pass
        else:
            # 如果不是，则将该行添加到集合 lines 中
            lines.add(line)
        # 返回 OutOfHeader 函数本身，作为下一行处理函数的标识
        return OutOfHeader
    
    # 初始化状态变量 state，用于跟踪当前处理的状态
    state = LookingForHeader
    # 遍历源代码的每一行，根据当前状态 state 处理每一行内容
    for line in UStringIO(source):
        state = state(line)
    # 返回集合 lines 中存储的处理后的代码行
    return lines.getValue()
# 从源码中移除子程序原型声明
def removeSubroutinePrototypes(source):
    # 此函数从未按照其名称的声明正常工作：
    # - "/* Subroutine */" 声明可能跨越多行，并且不能通过逐行方法匹配。
    # - 初始正则表达式中的插入符号会阻止任何匹配，甚至是单行的 "/* Subroutine */" 声明。
    #
    # 虽然我们可以“修复”这个函数使其按照名称应该做的事情来执行，
    # 但我们没有任何线索表明它实际应该做什么。
    #
    # 因此，我们保持现有的（无）功能，并将此函数文档化为根本不做任何操作。
    return source

# 从源码中移除内建函数声明
def removeBuiltinFunctions(source):
    # 创建一个行队列对象
    lines = LineQueue()

    # 查找内建函数声明的状态机函数
    def LookingForBuiltinFunctions(line):
        if line.strip() == '/* Builtin functions */':
            return InBuiltInFunctions
        else:
            lines.add(line)
            return LookingForBuiltinFunctions

    # 处于内建函数声明中的状态机函数
    def InBuiltInFunctions(line):
        if line.strip() == '':
            return LookingForBuiltinFunctions
        else:
            return InBuiltInFunctions

    # 初始状态为查找内建函数声明
    state = LookingForBuiltinFunctions

    # 对源码进行逐行处理
    for line in UStringIO(source):
        state = state(line)

    # 返回处理后的行队列对象的值
    return lines.getValue()

# 将 dlamch_ 调用替换为适当的宏
def replaceDlamch(source):
    """Replace dlamch_ calls with appropriate macros"""
    def repl(m):
        s = m.group(1)
        return dict(E='EPSILON', P='PRECISION', S='SAFEMINIMUM',
                    B='BASE')[s[0]]
    
    # 替换所有 dlamch_ 调用
    source = re.sub(r'dlamch_\("(.*?)"\)', repl, source)
    
    # 移除所有 extern 声明的 dlamch_ 函数
    source = re.sub(r'^\s+extern.*? dlamch_.*?;$(?m)', '', source)
    
    # 返回替换后的源码
    return source

# 清理源码的所有步骤
def scrubSource(source, nsteps=None, verbose=False):
    # 定义所有需要执行的步骤
    steps = [
             ('scrubbing ftnlen', scrubFtnlen),
             ('remove header', removeHeader),
             ('clean source', cleanSource),
             ('clean comments', cleanComments),
             ('replace dlamch_() calls', replaceDlamch),
             ('remove prototypes', removeSubroutinePrototypes),
             ('remove builtin function prototypes', removeBuiltinFunctions),
            ]

    # 如果指定了步骤数量，则仅执行指定数量的步骤
    if nsteps is not None:
        steps = steps[:nsteps]

    # 逐步执行所有步骤
    for msg, step in steps:
        if verbose:
            print(msg)
        source = step(source)

    # 返回处理后的源码
    return source

# 如果作为脚本运行，则处理指定的源文件
if __name__ == '__main__':
    # 从命令行参数中获取输入文件名和输出文件名
    filename = sys.argv[1]
    outfilename = os.path.join(sys.argv[2], os.path.basename(filename))
    
    # 打开输入文件并读取源码
    with open(filename) as fo:
        source = fo.read()

    # 如果指定了第三个参数，则限制处理步骤数量
    if len(sys.argv) > 3:
        nsteps = int(sys.argv[3])
    else:
        nsteps = None

    # 执行源码处理并获取处理后的源码
    source = scrubSource(source, nsteps, verbose=True)

    # 将处理后的源码写入输出文件
    with open(outfilename, 'w') as writefo:
        writefo.write(source)
```