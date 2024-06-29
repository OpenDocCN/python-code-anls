# `.\numpy\numpy\f2py\_src_pyf.py`

```
# 导入 re 模块，用于正则表达式操作
import re

# START OF CODE VENDORED FROM `numpy.distutils.from_template`
#############################################################
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

# 正则表达式，用于匹配子程序（subroutine）和函数（function）的开始
routine_start_re = re.compile(r'(\n|\A)((     (\$|\*))|)\s*(subroutine|function)\b', re.I)
# 正则表达式，用于匹配子程序（subroutine）和函数（function）的结束
routine_end_re = re.compile(r'\n\s*end\s*(subroutine|function)\b.*(\n|\Z)', re.I)
# 正则表达式，用于匹配函数（function）的开始
function_start_re = re.compile(r'\n     (\$|\*)\s*function\b', re.I)

def parse_structure(astr):
    """ Return a list of tuples for each function or subroutine each
    tuple is the start and end of a subroutine or function to be
    expanded.
    """
    # 初始化存储子程序和函数起始和结束位置的列表
    spanlist = []
    ind = 0
    while True:
        # 搜索子程序或函数的开始位置
        m = routine_start_re.search(astr, ind)
        if m is None:
            break
        start = m.start()
        if function_start_re.match(astr, start, m.end()):
            while True:
                # 找到函数的起始位置
                i = astr.rfind('\n', ind, start)
                if i == -1:
                    break
                start = i
                if astr[i:i+7] != '\n     $':
                    break
        start += 1
        # 搜索子程序或函数的结束位置
        m = routine_end_re.search(astr, m.end())
        ind = end = m and m.end()-1 or len(astr)
        spanlist.append((start, end))
    return spanlist

# 正则表达式，用于匹配模板标记（<..>）
template_re = re.compile(r"<\s*(\w[\w\d]*)\s*>")
# 正则表达式，用于匹配命名模板标记（<name=value>）
named_re = re.compile(r"<\s*(\w[\w\d]*)\s*=\s*(.*?)\s*>")
# 正则表达式，用于匹配简单的模板标记（<...>）
list_re = re.compile(r"<\s*((.*?))\s*>")

def find_repl_patterns(astr):
    # 使用命名模板正则表达式，找到所有命名模板标记并返回
    reps = named_re.findall(astr)
    names = {}
    # 对于每一个替换规则 rep 在 reps 列表中进行迭代
    for rep in reps:
        # 获取替换规则的名称，去除首尾空格，如果为空则使用唯一键生成一个新的名称
        name = rep[0].strip() or unique_key(names)
        # 获取替换规则的内容 repl，并将其中的逗号转换为特殊标记 '@comma@'
        repl = rep[1].replace(r'\,', '@comma@')
        # 使用 conv 函数对替换内容 repl 进行处理，转换为列表形式
        thelist = conv(repl)
        # 将处理后的替换规则名称 name 关联到处理后的列表内容 thelist
        names[name] = thelist
    # 返回处理后的所有替换规则名称及其对应的列表内容组成的字典 names
    return names
def find_and_remove_repl_patterns(astr):
    # 调用 find_repl_patterns 函数，找到并返回替换模式的名称列表
    names = find_repl_patterns(astr)
    # 使用 re.subn 函数将命名的替换模式替换为空字符串，并取替换后的字符串部分
    astr = re.subn(named_re, '', astr)[0]
    return astr, names

item_re = re.compile(r"\A\\(?P<index>\d+)\Z")
def conv(astr):
    # 将字符串按逗号分割成列表 b
    b = astr.split(',')
    # 对列表中的每个元素去除首尾空格，形成列表 l
    l = [x.strip() for x in b]
    # 遍历列表 l 的索引
    for i in range(len(l)):
        # 使用正则表达式 item_re 匹配列表元素 l[i]
        m = item_re.match(l[i])
        if m:
            # 如果匹配成功，提取匹配的索引号并转换为整数 j
            j = int(m.group('index'))
            # 将 l[i] 替换为列表中索引为 j 的元素
            l[i] = l[j]
    # 将列表 l 拼接成以逗号分隔的字符串并返回
    return ','.join(l)

def unique_key(adict):
    """ Obtain a unique key given a dictionary."""
    # 将字典的所有键转换成列表 allkeys
    allkeys = list(adict.keys())
    done = False
    n = 1
    while not done:
        # 构造新的键名
        newkey = '__l%s' % (n)
        if newkey in allkeys:
            n += 1
        else:
            done = True
    # 返回一个确保在字典中唯一的新键名
    return newkey

template_name_re = re.compile(r'\A\s*(\w[\w\d]*)\s*\Z')
def expand_sub(substr, names):
    # 将字符串中的特殊字符替换为标记，便于后续处理
    substr = substr.replace(r'\>', '@rightarrow@')
    substr = substr.replace(r'\<', '@leftarrow@')
    # 调用 find_repl_patterns 函数，找到并返回替换模式的名称列表
    lnames = find_repl_patterns(substr)
    # 使用 named_re 替换 substr 中的定义模板，将其替换为标准的 <name> 形式
    substr = named_re.sub(r"<\1>", substr)

    def listrepl(mobj):
        # 获取列表内容并进行处理，将特殊字符转换回逗号，并匹配已存在的列表名称
        thelist = conv(mobj.group(1).replace(r'\,', '@comma@'))
        if template_name_re.match(thelist):
            return "<%s>" % (thelist)
        name = None
        for key in lnames.keys():    # 查看列表是否已存在于字典中
            if lnames[key] == thelist:
                name = key
        if name is None:      # 若列表尚未存在于字典中
            name = unique_key(lnames)
            lnames[name] = thelist
        return "<%s>" % name

    substr = list_re.sub(listrepl, substr) # 将所有列表转换为命名模板
                                           # 根据需要构建新的名称

    numsubs = None
    base_rule = None
    rules = {}
    for r in template_re.findall(substr):
        if r not in rules:
            # 获取规则对应的替换列表
            thelist = lnames.get(r, names.get(r, None))
            if thelist is None:
                raise ValueError('No replicates found for <%s>' % (r))
            # 若规则不在 names 中，则添加到 names 中
            if r not in names and not thelist.startswith('_'):
                names[r] = thelist
            # 将规则转换为列表，并获取其长度
            rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
            num = len(rule)

            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                print("Mismatch in number of replacements (base <{}={}>) "
                      "for <{}={}>. Ignoring.".format(base_rule, ','.join(rules[base_rule]), r, thelist))
    if not rules:
        return substr

    def namerepl(mobj):
        # 获取命名规则并进行替换
        name = mobj.group(1)
        return rules.get(name, (k+1)*[name])[k]

    newstr = ''
    for k in range(numsubs):
        newstr += template_re.sub(namerepl, substr) + '\n\n'

    # 替换特殊标记为原字符
    newstr = newstr.replace('@rightarrow@', '>')
    newstr = newstr.replace('@leftarrow@', '<')
    return newstr

def process_str(allstr):
    # 初始化字符串处理
    newstr = allstr
    writestr = ''
    # 解析给定字符串的结构并返回结构化数据
    struct = parse_structure(newstr)
    
    # 初始化一个变量用于跟踪旧的结束位置
    oldend = 0
    
    # 初始化一个空字典用于存储变量名和对应的值，将特殊名称初始化到字典中
    names = {}
    names.update(_special_names)
    
    # 遍历结构化数据中的每个子结构
    for sub in struct:
        # 在原始字符串中查找和移除替换模式的定义，返回清理后的字符串和新定义的变量字典
        cleanedstr, defs = find_and_remove_repl_patterns(newstr[oldend:sub[0]])
        # 将清理后的字符串追加到输出字符串中
        writestr += cleanedstr
        # 更新变量名字典，添加新的定义
        names.update(defs)
        # 将子结构展开并添加到输出字符串中
        writestr += expand_sub(newstr[sub[0]:sub[1]], names)
        # 更新旧的结束位置为当前子结构的结束位置，以便继续处理下一个子结构
        oldend = sub[1]
    
    # 将剩余的未处理部分字符串添加到输出字符串中
    writestr += newstr[oldend:]
    
    # 返回最终生成的完整字符串
    return writestr
# 匹配包含文件名的行，以便解析 include 语句
include_src_re = re.compile(r"(\n|\A)\s*include\s*['\"](?P<name>[\w\d./\\]+\.src)['\"]", re.I)

# 解析源文件中的 include 语句，并展开所有包含的文件内容
def resolve_includes(source):
    # 获取源文件的目录路径
    d = os.path.dirname(source)
    # 打开源文件
    with open(source) as fid:
        lines = []
        # 逐行处理源文件内容
        for line in fid:
            # 尝试匹配 include_src_re 正则表达式
            m = include_src_re.match(line)
            if m:
                # 提取匹配到的文件名
                fn = m.group('name')
                # 如果文件名不是绝对路径，则与源文件目录路径拼接
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                # 如果文件存在，则递归处理其内容
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)  # 否则直接添加当前行到结果列表
            else:
                lines.append(line)  # 将未匹配到的行直接添加到结果列表
    return lines

# 处理给定源文件的包含语句，并返回处理后的结果
def process_file(source):
    # 解析所有 include 并展开内容后的源文件内容
    lines = resolve_includes(source)
    # 对合并后的字符串进行处理，返回处理后的结果
    return process_str(''.join(lines))

# 在字符串中查找和替换特殊模式，并返回替换模式的结果
_special_names = find_repl_patterns('''
<_c=s,d,c,z>
<_t=real,double precision,complex,double complex>
<prefix=s,d,c,z>
<ftype=real,double precision,complex,double complex>
<ctype=float,double,complex_float,complex_double>
<ftypereal=real,double precision,\\0,\\1>
<ctypereal=float,double,\\0,\\1>
''')

# 代码块结束，这段代码是从 `numpy.distutils.from_template` 中提取的
# END OF CODE VENDORED FROM `numpy.distutils.from_template`
###########################################################
```