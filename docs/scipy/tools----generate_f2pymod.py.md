# `D:\src\scipysrc\scipy\tools\generate_f2pymod.py`

```
#!/usr/bin/env python3
"""
Process f2py template files (`filename.pyf.src` -> `filename.pyf`)

Usage: python generate_pyf.py filename.pyf.src -o filename.pyf
"""

import argparse  # 导入处理命令行参数的模块
import os  # 导入操作系统相关功能的模块
import re  # 导入正则表达式模块
import subprocess  # 导入执行外部命令的模块


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

routine_start_re = re.compile(
    r'(\n|\A)((     (\$|\*))|)\s*(subroutine|function)\b',
    re.I
)
routine_end_re = re.compile(r'\n\s*end\s*(subroutine|function)\b.*(\n|\Z)', re.I)
function_start_re = re.compile(r'\n     (\$|\*)\s*function\b', re.I)

def parse_structure(astr):
    """ Return a list of tuples for each function or subroutine each
    tuple is the start and end of a subroutine or function to be
    expanded.
    """
    spanlist = []
    ind = 0
    while True:
        m = routine_start_re.search(astr, ind)
        if m is None:
            break
        start = m.start()
        if function_start_re.match(astr, start, m.end()):
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

template_re = re.compile(r"<\s*(\w[\w\d]*)\s*>")
named_re = re.compile(r"<\s*(\w[\w\d]*)\s*=\s*(.*?)\s*>")
# 定义一个正则表达式对象，用于匹配形如 "<...>" 的模式
list_re = re.compile(r"<\s*((.*?))\s*>")

# 查找并返回给定字符串中的替换模式及其对应的名称与替换内容的字典
def find_repl_patterns(astr):
    # 从字符串中提取所有命名的替换模式
    reps = named_re.findall(astr)
    # 创建一个空字典用于存储名称到替换内容的映射关系
    names = {}
    # 遍历每个替换模式及其对应的名称和内容
    for rep in reps:
        # 获取替换模式的名称，如果未提供名称则生成一个唯一的名称
        name = rep[0].strip() or unique_key(names)
        # 替换模式内容中的逗号转义为特定标记
        repl = rep[1].replace(r'\,', '@comma@')
        # 将替换内容转换为列表形式
        thelist = conv(repl)
        # 将名称与替换内容的列表关联存储到字典中
        names[name] = thelist
    # 返回名称到替换内容列表的映射字典
    return names

# 查找并移除给定字符串中的所有替换模式，并返回处理后的字符串及替换模式的字典
def find_and_remove_repl_patterns(astr):
    # 查找字符串中的所有替换模式及其对应的名称与替换内容的字典
    names = find_repl_patterns(astr)
    # 使用正则表达式将字符串中的所有命名替换模式替换为空字符串
    astr = re.subn(named_re, '', astr)[0]
    # 返回处理后的字符串和替换模式的字典
    return astr, names

# 定义一个正则表达式对象，用于匹配形如 "\d+" 的索引模式
item_re = re.compile(r"\A\\(?P<index>\d+)\Z")

# 将给定字符串按逗号分割为列表，并处理列表中的每个元素
def conv(astr):
    # 将字符串按逗号分割为列表
    b = astr.split(',')
    # 去除每个列表元素两端的空格
    l = [x.strip() for x in b]
    # 遍历列表中的每个元素
    for i in range(len(l)):
        # 如果元素匹配索引模式，则使用索引替换当前元素
        m = item_re.match(l[i])
        if m:
            j = int(m.group('index'))
            l[i] = l[j]
    # 将处理后的列表重新连接为字符串并返回
    return ','.join(l)

# 生成一个唯一的键名，确保在给定字典中不存在重复的键名
def unique_key(adict):
    """ Obtain a unique key given a dictionary."""
    # 获取字典中所有键名组成的列表
    allkeys = list(adict.keys())
    # 初始化循环控制变量和计数器
    done = False
    n = 1
    # 循环直到找到一个未在字典中使用的键名
    while not done:
        newkey = '__l%s' % (n)
        if newkey in allkeys:
            n += 1
        else:
            done = True
    # 返回新生成的唯一键名
    return newkey

# 定义一个正则表达式对象，用于匹配形如 "\w[\w\d]*" 的模板名称
template_name_re = re.compile(r'\A\s*(\w[\w\d]*)\s*\Z')

# 根据给定的子字符串和替换名称字典，扩展子字符串中的命名模板
def expand_sub(substr, names):
    # 将子字符串中的特定字符替换为标记
    substr = substr.replace(r'\>', '@rightarrow@')
    substr = substr.replace(r'\<', '@leftarrow@')
    # 查找子字符串中的所有替换模式及其对应的名称与替换内容的字典
    lnames = find_repl_patterns(substr)
    # 使用正则表达式将子字符串中的定义模板替换为空的模板形式
    substr = named_re.sub(r"<\1>", substr)  # get rid of definition templates

    # 定义一个函数，用于处理列表替换
    def listrepl(mobj):
        # 将正则匹配的列表内容转换为字符串
        thelist = conv(mobj.group(1).replace(r'\,', '@comma@'))
        # 如果列表内容符合模板名称，则直接返回原始模板形式
        if template_name_re.match(thelist):
            return "<%s>" % (thelist)
        name = None
        # 遍历替换名称字典中的每个键，检查是否已存在相同内容的列表
        for key in lnames.keys():
            if lnames[key] == thelist:
                name = key
        # 如果未找到相同内容的列表，则生成一个唯一的键名作为列表的名称
        if name is None:
            name = unique_key(lnames)
            lnames[name] = thelist
        # 返回列表的名称作为模板形式
        return "<%s>" % name

    # 使用正则表达式将子字符串中的所有列表替换为命名模板形式
    substr = list_re.sub(listrepl, substr) # convert all lists to named templates
                                           # newnames are constructed as needed

    # 初始化变量用于存储结果和规则
    numsubs = None
    base_rule = None
    rules = {}
    # 遍历子字符串中所有的模板名称，并处理每个模板的替换规则
    for r in template_re.findall(substr):
        if r not in rules:
            # 获取模板名称对应的替换内容列表
            thelist = lnames.get(r, names.get(r, None))
            # 如果未找到替换内容列表，则引发异常
            if thelist is None:
                raise ValueError('No replicates found for <%s>' % (r))
            # 如果模板名称不在全局名称字典中，则将其添加进去
            if r not in names and not thelist.startswith('_'):
                names[r] = thelist
            # 将替换内容列表转换为字符串形式的规则列表
            rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
            num = len(rule)

            # 初始化替换数目和基础规则
            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                # 如果替换数目不匹配，则打印警告信息并忽略当前规则
                print("Mismatch in number of replacements (base <{}={}>) "
                      "for <{}={}>. Ignoring."
                      .format(base_rule, ','.join(rules[base_rule]), r, thelist))
    # 如果未找到任何替换规则，则返回原始的子字符串
    if not rules:
        return substr
    # 定义一个函数 namerepl，用于替换模板中匹配的字符串
    def namerepl(mobj):
        # 获取匹配到的字符串中的名字部分
        name = mobj.group(1)
        # 根据规则字典获取替换后的名字，如果找不到则使用原名字的 k+1 倍重复列表中的第 k 个元素
        return rules.get(name, (k+1)*[name])[k]

    # 初始化一个空字符串用于存储处理后的文本
    newstr = ''
    # 遍历指定次数的模板替换操作
    for k in range(numsubs):
        # 使用正则表达式模板匹配并调用 namerepl 函数进行替换，并添加换行符
        newstr += template_re.sub(namerepl, substr) + '\n\n'

    # 将特定字符串 @rightarrow@ 替换为字符 '>'
    newstr = newstr.replace('@rightarrow@', '>')
    # 将特定字符串 @leftarrow@ 替换为字符 '<'
    newstr = newstr.replace('@leftarrow@', '<')
    # 返回处理后的文本结果
    return newstr
# 定义函数，处理输入的字符串并返回处理后的字符串
def process_str(allstr):
    # 复制输入的字符串到新变量
    newstr = allstr
    # 初始化一个空字符串用于存储处理后的结果
    writestr = ''

    # 解析输入字符串的结构，返回结构化的信息
    struct = parse_structure(newstr)

    # 初始化旧结束位置为0，用于追踪处理进度
    oldend = 0
    # 初始化一个空字典，用于存储特殊名称和替换模式
    names = {}
    # 将预定义的特殊名称加入到字典中
    names.update(_special_names)

    # 遍历结构化信息中的每个子结构
    for sub in struct:
        # 在当前处理段落中查找并移除替换模式，返回处理后的段落和定义的名称
        cleanedstr, defs = find_and_remove_repl_patterns(newstr[oldend:sub[0]])
        # 将处理后的段落添加到输出字符串中
        writestr += cleanedstr
        # 更新名称字典，加入新的定义
        names.update(defs)
        # 扩展并添加子结构的内容到输出字符串中
        writestr += expand_sub(newstr[sub[0]:sub[1]], names)
        # 更新旧结束位置为当前子结构的结束位置，以便继续处理下一段落
        oldend = sub[1]

    # 将剩余未处理的部分添加到输出字符串中
    writestr += newstr[oldend:]

    # 返回最终处理后的字符串
    return writestr

# 定义正则表达式模式，用于匹配包含源文件路径的包含语句
include_src_re = re.compile(
    r"(\n|\A)\s*include\s*['\"](?P<name>[\w\d./\\]+\.src)['\"]",
    re.I
)

# 定义函数，解析包含语句并递归处理源文件
def resolve_includes(source):
    # 获取源文件所在目录
    d = os.path.dirname(source)
    # 打开源文件，逐行处理
    with open(source) as fid:
        lines = []
        for line in fid:
            # 匹配包含语句的正则表达式
            m = include_src_re.match(line)
            if m:
                # 提取包含文件的名称
                fn = m.group('name')
                # 如果文件名不是绝对路径，构造完整路径
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                # 如果文件存在，递归解析包含的文件内容
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    # 如果文件不存在，直接添加当前行到输出列表中
                    lines.append(line)
            else:
                # 如果不是包含语句，直接添加当前行到输出列表中
                lines.append(line)
    # 返回处理后的所有行内容
    return lines

# 定义函数，处理源文件中的包含语句，并将结果传递给字符串处理函数
def process_file(source):
    # 解析并展开包含语句，获取所有源文件的内容
    lines = resolve_includes(source)
    # 将所有内容拼接成一个字符串，然后调用字符串处理函数进行处理
    return process_str(''.join(lines))

# 以下是从`numpy.distutils.from_template`供应的代码
###########################################################

# 定义特殊名称和替换模式的字典
_special_names = find_repl_patterns('''
<_c=s,d,c,z>
<_t=real,double precision,complex,double complex>
<prefix=s,d,c,z>
<ftype=real,double precision,complex,double complex>
<ctype=float,double,complex_float,complex_double>
<ftypereal=real,double precision,\\0,\\1>
<ctypereal=float,double,\\0,\\1>
''')

# 定义主函数，处理命令行参数并执行文件处理逻辑
def main():
    # 创建命令行解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数，输入文件路径
    parser.add_argument("infile", type=str,
                        help="Path to the input file")
    # 添加命令行选项，输出目录路径
    parser.add_argument("-o", "--outdir", type=str,
                        help="Path to the output directory")
    # 解析命令行参数
    args = parser.parse_args()

    # 检查输入文件的扩展名是否为支持的类型
    if not args.infile.endswith(('.pyf', '.pyf.src', '.f.src')):
        raise ValueError(f"Input file has unknown extension: {args.infile}")

    # 获取输出目录的绝对路径
    outdir_abs = os.path.join(os.getcwd(), args.outdir)

    # 如果输入文件是以'.pyf.src'或'.f.src'结尾，则处理文件并写出结果
    if args.infile.endswith(('.pyf.src', '.f.src')):
        # 处理输入文件，获取处理后的代码
        code = process_file(args.infile)
        # 构造输出文件的完整路径
        fname_pyf = os.path.join(args.outdir,
                                 os.path.splitext(os.path.split(args.infile)[1])[0])

        # 将处理后的代码写入到输出文件中
        with open(fname_pyf, 'w') as f:
            f.write(code)
    else:
        # 如果输入文件不需要处理，则直接使用输入文件的路径
        fname_pyf = args.infile

    # 现在调用f2py生成C API模块文件
    # 如果输入文件名以 '.pyf.src' 或 '.pyf' 结尾，则执行以下操作
    p = subprocess.Popen(['f2py', fname_pyf,
                          '--build-dir', outdir_abs], #'--quiet'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=os.getcwd())
    # 启动一个新的子进程来执行命令 'f2py'，传入参数 fname_pyf 和 '--build-dir'，并指定输出目录
    out, err = p.communicate()
    # 与子进程交互，获取其标准输出和标准错误输出
    if not (p.returncode == 0):
        # 如果子进程返回码不为 0，抛出运行时异常，显示错误信息和输出内容
        raise RuntimeError(f"Processing {fname_pyf} with f2py failed!\n"
                           f"{out.decode()}\n"
                           f"{err.decode()}")
# 如果当前脚本被直接执行（而不是被导入为模块），则执行下面的代码块
if __name__ == "__main__":
    # 调用主程序入口函数，这里假设主程序入口函数名为main()
    main()
```