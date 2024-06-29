# `.\numpy\numpy\distutils\conv_template.py`

```py
# 解析输入字符串中的重复块结构，返回第一级重复块的起始和结束标记
def parse_structure(astr, level):
    """
    The returned line number is from the beginning of the string, starting
    at zero. Returns an empty list if no loops found.

    """
    # 如果是第一级重复块，则使用标准的起始和结束标记
    if level == 0 :
        loopbeg = "/**begin repeat"
        loopend = "/**end repeat**/"
    # 对于嵌套的重复块，使用带有深度编号的起始和结束标记
    else :
        loopbeg = "/**begin repeat%d" % level
        loopend = "/**end repeat%d**/" % level

    ind = 0  # 初始化索引
    line = 0  # 初始化行数
    spanlist = []  # 初始化用于存储重复块位置的列表
    # 循环直到条件不再满足
    while True:
        # 在字符串 astr 中从索引 ind 开始查找字符串 loopbeg 的第一个出现位置
        start = astr.find(loopbeg, ind)
        # 如果未找到，则退出循环
        if start == -1:
            break
        # 继续在字符串 astr 中查找 "*/" 的第一个出现位置，起始索引为 start
        start2 = astr.find("*/", start)
        # 在找到的 "*/" 后查找下一个换行符的位置
        start2 = astr.find("\n", start2)
        # 在字符串 astr 中从 start2 开始查找字符串 loopend 的第一个出现位置
        fini1 = astr.find(loopend, start2)
        # 继续在找到的 loopend 后查找下一个换行符的位置
        fini2 = astr.find("\n", fini1)
        # 计算并累加位于 ind 和 start2+1 之间的换行符数量到 line 中
        line += astr.count("\n", ind, start2+1)
        # 将找到的起始和结束位置以及相关行号信息作为元组添加到 spanlist 中
        spanlist.append((start, start2+1, fini1, fini2+1, line))
        # 计算并累加位于 start2+1 和 fini2 之间的换行符数量到 line 中
        line += astr.count("\n", start2+1, fini2)
        # 更新查找的起始索引 ind 为 fini2
        ind = fini2
    # 对 spanlist 中的元组按照起始位置排序
    spanlist.sort()
    # 返回排序后的 spanlist 列表
    return spanlist
# 为给定的正则表达式模式创建一个编译对象，用于匹配字符串中的 '(a,b,c)*4' 格式
parenrep = re.compile(r"\(([^)]*)\)\*(\d+)")
# 为给定的正则表达式模式创建一个编译对象，用于匹配字符串中的 'xxx*3' 格式
plainrep = re.compile(r"([^*]+)\*(\d+)")

def parse_values(astr):
    """
    替换字符串 astr 中所有的 '(a,b,c)*4' 形式为 'a,b,c,a,b,c,a,b,c,a,b,c'。
    空括号生成空值，例如 '()*4' 返回 ',,,'。结果按 ',' 分割并返回值列表。
    替换字符串 astr 中所有的 'xxx*3' 形式为 'xxx,xxx,xxx'。
    """
    astr = parenrep.sub(paren_repl, astr)
    astr = ','.join([plainrep.sub(paren_repl, x.strip()) for x in astr.split(',')])
    return astr.split(',')

# 为给定的正则表达式模式创建一个编译对象，用于匹配字符串中的 '\n\s*\*?' 格式
stripast = re.compile(r"\n\s*\*?")
# 为给定的正则表达式模式创建一个编译对象，用于匹配字符串中的 '# name = value #' 格式
named_re = re.compile(r"#\s*(\w*)\s*=([^#]*)#")
# 为给定的正则表达式模式创建一个编译对象，用于匹配字符串中的 'var1=value1' 格式
exclude_vars_re = re.compile(r"(\w*)=(\w*)")
# 为给定的正则表达式模式创建一个编译对象，用于匹配字符串中的 ':exclude:' 格式
exclude_re = re.compile(":exclude:")

def parse_loop_header(loophead):
    """
    解析循环头部字符串，查找所有的命名替换和排除变量。

    返回两个部分：
    1. 一个字典列表，每个字典代表一个循环迭代，其中键是要替换的名称，值是替换的字符串。
    2. 一个排除列表，每个条目都是一个字典，包含要排除的变量名和对应的值。
    """
    loophead = stripast.sub("", loophead)
    names = []
    reps = named_re.findall(loophead)
    nsub = None
    for rep in reps:
        name = rep[0]
        vals = parse_values(rep[1])
        size = len(vals)
        if nsub is None:
            nsub = size
        elif nsub != size:
            msg = "Mismatch in number of values, %d != %d\n%s = %s"
            raise ValueError(msg % (nsub, size, name, vals))
        names.append((name, vals))

    excludes = []
    for obj in exclude_re.finditer(loophead):
        span = obj.span()
        endline = loophead.find('\n', span[1])
        substr = loophead[span[1]:endline]
        ex_names = exclude_vars_re.findall(substr)
        excludes.append(dict(ex_names))

    dlist = []
    if nsub is None:
        raise ValueError("No substitution variables found")
    for i in range(nsub):
        tmp = {name: vals[i] for name, vals in names}
        dlist.append(tmp)
    return dlist

# 为给定的正则表达式模式创建一个编译对象，用于匹配字符串中的 '@varname@' 格式
replace_re = re.compile(r"@(\w+)@")

def parse_string(astr, env, level, line):
    """
    解析字符串 astr，替换其中的 '@varname@' 格式为环境变量中对应的值。

    参数：
    - astr: 待解析的字符串
    - env: 环境变量字典，用于替换 '@varname@' 中的 varname
    - level: 解析层级
    - line: 当前处理的行数

    返回替换后的字符串以及包含当前行号信息的字符串
    """
    lineno = "#line %d\n" % line
    # 定义一个函数 replace，用于替换字符串中的变量名为对应的值
    def replace(match):
        # 获取匹配到的变量名
        name = match.group(1)
        try:
            # 尝试从环境变量中获取对应变量名的值
            val = env[name]
        except KeyError:
            # 如果变量名在环境变量中未定义，抛出错误并指明行号和变量名
            msg = 'line %d: no definition of key "%s"' % (line, name)
            raise ValueError(msg) from None
        # 返回变量名对应的值
        return val

    # 初始化代码列表，包含初始行号
    code = [lineno]
    # 解析给定的结构化字符串 astr，得到结构信息 struct
    struct = parse_structure(astr, level)
    if struct:
        # 如果存在结构信息，则递归处理内部循环
        oldend = 0
        newlevel = level + 1
        for sub in struct:
            # 分别获取前缀、头部、文本和结束位置等信息
            pref = astr[oldend:sub[0]]
            head = astr[sub[0]:sub[1]]
            text = astr[sub[1]:sub[2]]
            oldend = sub[3]
            newline = line + sub[4]
            # 将前缀部分中的变量名替换为对应的值
            code.append(replace_re.sub(replace, pref))
            try:
                # 解析循环头部，获取环境变量列表
                envlist = parse_loop_header(head)
            except ValueError as e:
                # 如果解析出错，抛出包含行号和错误信息的新错误
                msg = "line %d: %s" % (newline, e)
                raise ValueError(msg)
            for newenv in envlist:
                # 更新新环境变量列表，并解析文本内容得到新代码，追加到 code 中
                newenv.update(env)
                newcode = parse_string(text, newenv, newlevel, newline)
                code.extend(newcode)
        # 获取剩余的后缀部分并进行变量名替换
        suff = astr[oldend:]
        code.append(replace_re.sub(replace, suff))
    else:
        # 如果不存在结构信息，则直接将整个字符串 astr 中的变量名替换为对应的值
        code.append(replace_re.sub(replace, astr))
    # 将代码列表转换为字符串并返回
    code.append('\n')
    return ''.join(code)
# 处理给定字符串，返回处理后的代码字符串
def process_str(astr):
    # 创建一个空的代码列表，包含头部信息
    code = [header]
    # 将解析字符串后的代码块添加到代码列表中
    code.extend(parse_string(astr, global_names, 0, 1))
    # 将代码列表转换为一个字符串并返回
    return ''.join(code)


# 匹配以#include开头的行，并捕获引号内的文件名
include_src_re = re.compile(r"(\n|\A)#include\s*['\"]"
                            r"(?P<name>[\w\d./\\]+[.]src)['\"]", re.I)

# 解析源文件中的包含文件，并返回解析后的所有行
def resolve_includes(source):
    # 获取源文件的目录路径
    d = os.path.dirname(source)
    # 打开源文件并逐行处理
    with open(source) as fid:
        lines = []
        for line in fid:
            # 匹配当前行是否为#include开头的格式
            m = include_src_re.match(line)
            if m:
                # 提取引号内的文件名
                fn = m.group('name')
                # 如果文件名不是绝对路径，则加上源文件的目录路径
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                # 如果文件存在，则递归解析该文件的内容，否则直接添加当前行
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                lines.append(line)
    # 返回所有处理后的行组成的列表
    return lines


# 处理源文件，返回处理后的代码字符串
def process_file(source):
    # 解析源文件中的包含文件，获取所有处理后的行
    lines = resolve_includes(source)
    # 规范化源文件路径，并将路径中的反斜杠转义为双反斜杠
    sourcefile = os.path.normcase(source).replace("\\", "\\\\")
    try:
        # 调用process_str处理所有行组成的字符串，获取处理后的代码字符串
        code = process_str(''.join(lines))
    except ValueError as e:
        # 如果出现错误，则抛出新的错误，指示出错的源文件及位置
        raise ValueError('In "%s" loop at %s' % (sourcefile, e)) from None
    # 返回带有#line指令的处理后的代码字符串
    return '#line 1 "%s"\n%s' % (sourcefile, code)


# 给定一个字典，生成一个唯一的键
def unique_key(adict):
    # 获取字典中所有的键
    allkeys = list(adict.keys())
    done = False
    n = 1
    # 循环直到生成一个唯一的键
    while not done:
        # 按长度为n的方式生成新的键
        newkey = "".join([x[:n] for x in allkeys])
        # 如果新键已经存在于字典的键中，则增加n的长度
        if newkey in allkeys:
            n += 1
        else:
            done = True
    # 返回生成的唯一键
    return newkey


# 主程序入口
def main():
    try:
        # 尝试从命令行参数中获取文件名
        file = sys.argv[1]
    except IndexError:
        # 如果没有命令行参数，则使用标准输入和标准输出
        fid = sys.stdin
        outfile = sys.stdout
    else:
        # 如果有命令行参数，则打开对应的输入文件和输出文件
        fid = open(file, 'r')
        (base, ext) = os.path.splitext(file)
        newname = base
        outfile = open(newname, 'w')

    # 读取所有输入内容
    allstr = fid.read()
    try:
        # 处理所有输入内容，并获取处理后的字符串
        writestr = process_str(allstr)
    except ValueError as e:
        # 如果处理过程中出错，则抛出新的错误，指示出错的文件及位置
        raise ValueError("In %s loop at %s" % (file, e)) from None

    # 将处理后的字符串写入输出文件
    outfile.write(writestr)


# 如果当前脚本作为主程序运行，则调用main函数
if __name__ == "__main__":
    main()
```