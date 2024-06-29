# `.\numpy\numpy\distutils\from_template.py`

```py
"""
process_file(filename)

  takes templated file .xxx.src and produces .xxx file where .xxx
  is .pyf .f90 or .f using the following template rules:

  '<..>' denotes a template.

  All function and subroutine blocks in a source file with names that
  contain '<..>' will be replicated according to the rules in '<..>'.

  The number of comma-separated words in '<..>' will determine the number of
  replicates.

  '<..>' may have two different forms, named and short. For example,

  named:
   <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with
   'd', 's', 'z', and 'c' for each replicate of the block.

   <_c>  is already defined: <_c=s,d,c,z>
   <_t>  is already defined: <_t=real,double precision,complex,double complex>

  short:
   <s,d,c,z>, a short form of the named, useful when no <p> appears inside
   a block.

  In general, '<..>' contains a comma separated list of arbitrary
  expressions. If these expression must contain a comma|leftarrow|rightarrow,
  then prepend the comma|leftarrow|rightarrow with a backslash.

  If an expression matches '\\<index>' then it will be replaced
  by <index>-th expression.

  Note that all '<..>' forms in a block must have the same number of
  comma-separated entries.

 Predefined named template rules:
  <prefix=s,d,c,z>
  <ftype=real,double precision,complex,double complex>
  <ftypereal=real,double precision,\\0,\\1>
  <ctype=float,double,complex_float,complex_double>
  <ctypereal=float,double,\\0,\\1>

"""

__all__ = ['process_str', 'process_file']

import os
import sys
import re

# 正则表达式，用于识别函数和子程序的开始
routine_start_re = re.compile(r'(\n|\A)((     (\$|\*))|)\s*(subroutine|function)\b', re.I)
# 正则表达式，用于识别函数和子程序的结束
routine_end_re = re.compile(r'\n\s*end\s*(subroutine|function)\b.*(\n|\Z)', re.I)
# 正则表达式，用于识别函数的开始
function_start_re = re.compile(r'\n     (\$|\*)\s*function\b', re.I)

def parse_structure(astr):
    """ Return a list of tuples for each function or subroutine each
    tuple is the start and end of a subroutine or function to be
    expanded.
    """
    spanlist = []
    ind = 0
    # 循环直到找不到更多函数或子程序的起始位置
    while True:
        m = routine_start_re.search(astr, ind)
        if m is None:
            break
        start = m.start()
        if function_start_re.match(astr, start, m.end()):
            # 处理函数的情况，找到函数块的真正起始位置
            while True:
                i = astr.rfind('\n', ind, start)
                if i == -1:
                    break
                start = i
                if astr[i:i+7] != '\n     $':
                    break
        start += 1
        m = routine_end_re.search(astr, m.end())
        ind = end = m and m.end()-1 or len(astr)
        spanlist.append((start, end))
    return spanlist

# 正则表达式，用于识别模板
template_re = re.compile(r"<\s*(\w[\w\d]*)\s*>")
# 正则表达式，用于识别命名模板
named_re = re.compile(r"<\s*(\w[\w\d]*)\s*=\s*(.*?)\s*>")
# 正则表达式，用于识别简短模板
list_re = re.compile(r"<\s*((.*?))\s*>")

def find_repl_patterns(astr):
    reps = named_re.findall(astr)
    names = {}
    for rep in reps:
        # 遍历列表 `reps` 中的每个元素，每个元素是一个表示替换规则的列表 `rep`
        name = rep[0].strip() or unique_key(names)
        # 获取 `rep` 列表的第一个元素，并移除首尾空白，如果为空则调用 `unique_key(names)` 生成唯一键名
        repl = rep[1].replace(r'\,', '@comma@')
        # 获取 `rep` 列表的第二个元素，并替换其中的 `\,` 为 `@comma@`
        thelist = conv(repl)
        # 使用函数 `conv` 处理替换后的字符串 `repl`，并赋值给 `thelist`
        names[name] = thelist
        # 将 `name` 作为键，`thelist` 作为值添加到 `names` 字典中
    # 返回更新后的 `names` 字典
    return names
# 定义一个函数，用于查找并移除字符串中的替换模式，并返回处理后的字符串和模式名称列表
def find_and_remove_repl_patterns(astr):
    # 调用 find_repl_patterns 函数查找字符串中的替换模式，并将结果存储在 names 变量中
    names = find_repl_patterns(astr)
    # 使用正则表达式替换字符串中的命名替换模式，并获取替换后的字符串部分
    astr = re.subn(named_re, '', astr)[0]
    # 返回处理后的字符串和模式名称列表
    return astr, names

# 编译一个正则表达式，用于匹配形如 \数字 的字符串开头
item_re = re.compile(r"\A\\(?P<index>\d+)\Z")

# 定义一个函数，用于将输入字符串按逗号分割，并去除每个分割结果的首尾空格
def conv(astr):
    b = astr.split(',')
    l = [x.strip() for x in b]
    # 遍历列表中的每个元素，尝试用 item_re 正则表达式匹配开头是 \数字 的元素，并进行替换
    for i in range(len(l)):
        m = item_re.match(l[i])
        if m:
            j = int(m.group('index'))
            l[i] = l[j]
    # 将处理后的列表重新拼接成一个字符串，并用逗号连接
    return ','.join(l)

# 定义一个函数，用于生成一个唯一的键值，确保其不在给定字典中存在
def unique_key(adict):
    """ Obtain a unique key given a dictionary."""
    allkeys = list(adict.keys())
    done = False
    n = 1
    while not done:
        newkey = '__l%s' % (n)
        if newkey in allkeys:
            n += 1
        else:
            done = True
    return newkey

# 编译一个正则表达式，用于匹配模板名称的开头，形如 \w[\w\d]*
template_name_re = re.compile(r'\A\s*(\w[\w\d]*)\s*\Z')

# 定义一个函数，用于扩展子字符串，并处理特定的转义符号
def expand_sub(substr, names):
    # 将字符串中的特定转义符号替换为指定字符串
    substr = substr.replace(r'\>', '@rightarrow@')
    substr = substr.replace(r'\<', '@leftarrow@')
    # 调用 find_repl_patterns 函数查找字符串中的替换模式，并将结果存储在 lnames 变量中
    lnames = find_repl_patterns(substr)
    # 使用 named_re 正则表达式替换 substr 中的定义模板，将其替换为 <\1> 形式
    substr = named_re.sub(r"<\1>", substr)

    # 定义一个函数，用于处理列表的替换操作
    def listrepl(mobj):
        # 将匹配到的内容中的特定转义符号替换为 @comma@，然后调用 conv 函数处理
        thelist = conv(mobj.group(1).replace(r'\,', '@comma@'))
        # 如果列表模板名匹配 template_name_re，返回形如 <模板名> 的字符串
        if template_name_re.match(thelist):
            return "<%s>" % (thelist)
        name = None
        # 检查当前列表是否已经在字典中存在，如果不存在则生成唯一的键名
        for key in lnames.keys():
            if lnames[key] == thelist:
                name = key
        if name is None:
            name = unique_key(lnames)
            lnames[name] = thelist
        return "<%s>" % name

    # 使用 list_re 正则表达式将 substr 中的所有列表替换为命名模板
    substr = list_re.sub(listrepl, substr)

    # 初始化变量
    numsubs = None
    base_rule = None
    rules = {}

    # 遍历所有在模板正则表达式中匹配到的内容
    for r in template_re.findall(substr):
        if r not in rules:
            # 获取当前模板在 lnames 或 names 中对应的替换列表
            thelist = lnames.get(r, names.get(r, None))
            if thelist is None:
                raise ValueError('No replicates found for <%s>' % (r))
            if r not in names and not thelist.startswith('_'):
                names[r] = thelist
            # 将获取的替换列表转换为字符串数组，并获取其长度
            rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
            num = len(rule)

            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                print("Mismatch in number of replacements (base <%s=%s>)"
                      " for <%s=%s>. Ignoring." %
                      (base_rule, ','.join(rules[base_rule]), r, thelist))
    
    # 如果 rules 为空，直接返回 substr
    if not rules:
        return substr

    # 定义一个函数，用于替换 substr 中的模板名称为具体的替换内容
    def namerepl(mobj):
        name = mobj.group(1)
        return rules.get(name, (k+1)*[name])[k]

    # 初始化 newstr 变量为空字符串
    newstr = ''
    # 遍历 numsubs 次，依次替换 substr 中的模板名称为具体的替换内容，并添加换行符
    for k in range(numsubs):
        newstr += template_re.sub(namerepl, substr) + '\n\n'

    # 将字符串中的特定转义符号替换回原始符号
    newstr = newstr.replace('@rightarrow@', '>')
    newstr = newstr.replace('@leftarrow@', '<')
    # 返回处理后的字符串
    return newstr

# 定义一个函数，用于处理输入的字符串
def process_str(allstr):
    newstr = allstr
    # 初始化一个空字符串，用于存储最终生成的字符串
    writestr = ''

    # 调用 parse_structure 函数解析 newstr，返回结构化的数据结构
    struct = parse_structure(newstr)

    # 初始化旧结束位置为 0
    oldend = 0
    # 初始化一个空字典 names，并添加 _special_names 中的特殊名称
    names = {}
    names.update(_special_names)
    
    # 遍历结构化数据
    for sub in struct:
        # 在 newstr 的 oldend 到 sub[0] 之间查找和移除替换模式，并返回清理后的字符串和定义
        cleanedstr, defs = find_and_remove_repl_patterns(newstr[oldend:sub[0]])
        # 将清理后的字符串添加到 writestr 中
        writestr += cleanedstr
        # 更新 names 字典，添加新的定义
        names.update(defs)
        # 将 newstr 中 sub[0] 到 sub[1] 之间的子字符串进行展开，使用当前的 names 字典
        writestr += expand_sub(newstr[sub[0]:sub[1]], names)
        # 更新 oldend 为当前 sub 的结束位置 sub[1]
        oldend = sub[1]
    
    # 将剩余的 newstr 中 oldend 之后的部分添加到 writestr 中
    writestr += newstr[oldend:]

    # 返回生成的最终字符串 writestr
    return writestr
# 匹配包含语句的正则表达式，用于查找形如 `include 'filename.src'` 的字符串
include_src_re = re.compile(r"(\n|\A)\s*include\s*['\"](?P<name>[\w\d./\\]+\.src)['\"]", re.I)

# 解析包含语句并替换成实际内容后返回所有行的列表
def resolve_includes(source):
    # 获取源文件所在目录
    d = os.path.dirname(source)
    # 打开源文件并逐行处理
    with open(source) as fid:
        lines = []
        for line in fid:
            # 尝试匹配包含语句的正则表达式
            m = include_src_re.match(line)
            if m:
                # 获取包含文件的文件名
                fn = m.group('name')
                # 如果文件名不是绝对路径，则与源文件目录拼接
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                # 如果包含文件存在，则递归处理其内容，否则直接添加原始行
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                # 如果不是包含语句，则直接添加原始行
                lines.append(line)
    return lines

# 处理文件，首先解析包含语句，然后处理成字符串返回
def process_file(source):
    lines = resolve_includes(source)
    return process_str(''.join(lines))

# 查找特殊名称的替换模式并返回
_special_names = find_repl_patterns('''
<_c=s,d,c,z>
<_t=real,double precision,complex,double complex>
<prefix=s,d,c,z>
<ftype=real,double precision,complex,double complex>
<ctype=float,double,complex_float,complex_double>
<ftypereal=real,double precision,\\0,\\1>
<ctypereal=float,double,\\0,\\1>
''')

# 主函数，处理命令行参数，读取和写入文件内容，并进行处理
def main():
    try:
        file = sys.argv[1]
    except IndexError:
        fid = sys.stdin
        outfile = sys.stdout
    else:
        fid = open(file, 'r')
        (base, ext) = os.path.splitext(file)
        newname = base
        outfile = open(newname, 'w')

    # 读取输入文件的全部内容
    allstr = fid.read()
    # 处理全部内容成字符串
    writestr = process_str(allstr)
    # 将处理后的内容写入输出文件
    outfile.write(writestr)

# 如果作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```