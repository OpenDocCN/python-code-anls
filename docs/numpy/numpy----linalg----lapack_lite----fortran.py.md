# `.\numpy\numpy\linalg\lapack_lite\fortran.py`

```
# WARNING! This a Python 2 script. Read README.rst for rationale.
# 引入 re 和 itertools 模块
import re
import itertools

# 检查给定行是否为空白行
def isBlank(line):
    return not line

# 检查给定行是否是标签行（以数字开头）
def isLabel(line):
    return line[0].isdigit()

# 检查给定行是否是注释行（不以空格开头）
def isComment(line):
    return line[0] != ' '

# 检查给定行是否是续行（第 6 个字符不是空格）
def isContinuation(line):
    return line[5] != ' '

# 定义常量 COMMENT, STATEMENT, CONTINUATION 用于表示不同类型的行
COMMENT, STATEMENT, CONTINUATION = 0, 1, 2

# 函数：确定 Fortran 代码行的类型
def lineType(line):
    """Return the type of a line of Fortran code."""
    if isBlank(line):
        return COMMENT
    elif isLabel(line):
        return STATEMENT
    elif isComment(line):
        return COMMENT
    elif isContinuation(line):
        return CONTINUATION
    else:
        return STATEMENT

# 类：LineIterator，用于迭代处理行并去除行尾空格
class LineIterator:
    """LineIterator(iterable)

    Return rstrip()'d lines from iterable, while keeping a count of the
    line number in the .lineno attribute.
    """
    def __init__(self, iterable):
        object.__init__(self)
        self.iterable = iter(iterable)
        self.lineno = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.lineno += 1
        line = next(self.iterable)
        line = line.rstrip()
        return line

    next = __next__


# 类：PushbackIterator，支持将元素推回迭代器
class PushbackIterator:
    """PushbackIterator(iterable)

    Return an iterator for which items can be pushed back into.
    Call the .pushback(item) method to have item returned as the next
    value of next().
    """
    def __init__(self, iterable):
        object.__init__(self)
        self.iterable = iter(iterable)
        self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.pop()
        else:
            return next(self.iterable)

    def pushback(self, item):
        self.buffer.append(item)

    next = __next__


# 函数：fortranSourceLines，返回一个迭代器，处理 Fortran 源文件的语句行
def fortranSourceLines(fo):
    """Return an iterator over statement lines of a Fortran source file.

    Comment and blank lines are stripped out, and continuation lines are
    merged.
    """
    numberingiter = LineIterator(fo)
    # 在末尾添加一个额外的空字符串，以处理续行
    with_extra = itertools.chain(numberingiter, [''])
    pushbackiter = PushbackIterator(with_extra)
    for line in pushbackiter:
        t = lineType(line)
        if t == COMMENT:
            continue
        elif t == STATEMENT:
            lines = [line]
            # 处理续行的逻辑，保证行的连续性
            for next_line in pushbackiter:
                t = lineType(next_line)
                if t == CONTINUATION:
                    lines.append(next_line[6:])
                else:
                    pushbackiter.pushback(next_line)
                    break
            yield numberingiter.lineno, ''.join(lines)
        else:
            raise ValueError("jammed: continuation line not expected: %s:%d" %
                             (fo.name, numberingiter.lineno))

# 函数：getDependencies(filename)
def getDependencies(filename):
    """
    对于一个 Fortran 源文件，返回其中声明为 EXTERNAL 的例程列表。
    """
    # 编译正则表达式模式，用于匹配以 EXTERNAL 开头的行（忽略大小写）
    external_pat = re.compile(r'^\s*EXTERNAL\s', re.I)
    # 初始化例程列表
    routines = []
    # 打开文件并进行迭代处理每一行
    with open(filename) as fo:
        # 使用自定义的函数迭代处理 Fortran 源文件的行
        for lineno, line in fortranSourceLines(fo):
            # 尝试在当前行中匹配 EXTERNAL 的模式
            m = external_pat.match(line)
            if m:
                # 如果匹配成功，提取 EXTERNAL 后面的例程名称列表
                names = line[m.end():].strip().split(',')
                # 去除每个名称的首尾空白字符并转换为小写
                names = [n.strip().lower() for n in names]
                # 过滤掉空的名称
                names = [n for n in names if n]
                # 将提取的例程名称列表添加到总例程列表中
                routines.extend(names)
    # 返回最终的例程名称列表作为结果
    return routines
```