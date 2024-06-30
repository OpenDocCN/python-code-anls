# `D:\src\scipysrc\sympy\doc\ext\docscrape.py`

```
"""
Extract reference documentation from the NumPy source tree.
"""

# 导入所需模块
import inspect             # 提供检查类和函数定义的工具
import textwrap            # 提供文本包装和填充的功能
import re                  # 提供正则表达式匹配操作
import pydoc               # 生成 Python 对象的文档
from collections.abc import Mapping   # 导入 Mapping 抽象基类
import sys                 # 提供与 Python 解释器交互的功能

# 定义一个字符串行读取器类
class Reader:
    """
    A line-based string reader.
    """
    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           String with lines separated by '\n'.

        """
        # 如果输入数据是列表，则直接使用；否则按行分割字符串保存为列表形式
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split('\n')  # 将字符串按行分割保存为列表

        self.reset()  # 调用 reset 方法初始化当前行指针

    def __getitem__(self, n):
        return self._str[n]

    def reset(self):
        self._l = 0  # 当前行号初始化为 0

    def read(self):
        # 如果当前行号小于列表长度，则返回当前行并将行号加一；否则返回空字符串
        if not self.eof():
            out = self[self._l]
            self._l += 1
            return out
        else:
            return ''

    def seek_next_non_empty_line(self):
        # 从当前行开始查找非空行，找到则跳出循环；否则当前行号加一
        for l in self[self._l:]:
            if l.strip():   # 如果去除首尾空白字符后不为空，则跳出循环
                break
            else:
                self._l += 1

    def eof(self):
        # 判断当前行号是否超过列表长度
        return self._l >= len(self._str)

    def read_to_condition(self, condition_func):
        start = self._l
        # 从当前行开始，循环遍历列表，直到满足条件函数返回真值时返回符合条件的行列表
        for line in self[start:]:
            if condition_func(line):
                return self[start:self._l]
            self._l += 1
            if self.eof():
                return self[start:self._l + 1]
        return []

    def read_to_next_empty_line(self):
        self.seek_next_non_empty_line()

        def is_empty(line):
            return not line.strip()

        return self.read_to_condition(is_empty)

    def read_to_next_unindented_line(self):
        def is_unindented(line):
            return line.strip() and (len(line.lstrip()) == len(line))
        return self.read_to_condition(is_unindented)

    def peek(self, n=0):
        # 返回当前行号加 n 后的行内容，如果超出列表范围则返回空字符串
        if self._l + n < len(self._str):
            return self[self._l + n]
        else:
            return ''

    def is_empty(self):
        # 判断所有行连接后的字符串是否为空
        return not ''.join(self._str).strip()


class NumpyDocString(Mapping):
    def __init__(self, docstring, config={}):
        docstring = textwrap.dedent(docstring).split('\n')

        # 使用 Reader 类处理文档字符串，初始化解析数据字典和其他键列表
        self._doc = Reader(docstring)
        self._parsed_data = {
            'Signature': '',
            'Summary': [''],
            'Extended Summary': [],
            'Parameters': [],
            'Returns': [],
            'Yields': [],
            'Raises': [],
            'Warns': [],
            'Other Parameters': [],
            'Attributes': [],
            'Methods': [],
            'See Also': [],
            'Warnings': [],
            'References': '',
            'index': {}
        }
        self._other_keys = []

        self._parse()  # 调用解析方法解析文档字符串

    def __getitem__(self, key):
        # 获取解析数据字典中的键值对
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        # 如果键不存在于解析数据字典中，则将键添加到其他键列表中
        if key not in self._parsed_data:
            self._other_keys.append(key)

        self._parsed_data[key] = val
    # 返回一个迭代器，用于迭代对象的_parsed_data成员
    def __iter__(self):
        return iter(self._parsed_data)

    # 返回对象_parsed_data成员的长度
    def __len__(self):
        return len(self._parsed_data)

    # 检查当前文档是否位于一个节(section)的起始位置
    def _is_at_section(self):
        # 移动文档指针到下一个非空行
        self._doc.seek_next_non_empty_line()

        # 如果已经到达文档末尾，返回False
        if self._doc.eof():
            return False

        # 获取当前行，并去除两端的空白
        l1 = self._doc.peek().strip()  # 例如：Parameters

        # 如果当前行以 '.. index::' 开头，则认为是一个节(section)
        if l1.startswith('.. index::'):
            return True

        # 获取下一行，并去除两端的空白
        l2 = self._doc.peek(1).strip()  # ---------- 或者 ==========

        # 如果下一行以 '-' 或 '=' 组成与l1相同长度的字符开头，则认为是一个节(section)
        return l2.startswith(('-' * len(l1), '=' * len(l1)))

    # 去除文档开头和结尾的空白行，并返回处理后的文档
    def _strip(self, doc):
        i = 0
        j = 0
        # 找到第一个非空行的索引i
        for i, line in enumerate(doc):
            if line.strip():
                break

        # 找到最后一个非空行的索引j
        for j, line in enumerate(doc[::-1]):
            if line.strip():
                break

        # 返回去除了开头和结尾空白行的文档片段
        return doc[i:len(doc) - j]

    # 读取到下一个节(section)的内容，并返回节(section)的列表
    def _read_to_next_section(self):
        # 读取到下一个空行结束，并作为起始节(section)的内容
        section = self._doc.read_to_next_empty_line()

        # 循环直到遇到下一个节(section)或文档末尾
        while not self._is_at_section() and not self._doc.eof():
            # 如果前一行为空白行，则添加一个空行到节(section)中
            if not self._doc.peek(-1).strip():  # 前一行是空行
                section += ['']

            # 继续读取到下一个空行结束，并添加到节(section)中
            section += self._doc.read_to_next_empty_line()

        # 返回完整的节(section)
        return section

    # 读取整个文档中的节(section)，并生成节(section)名称和内容的迭代器
    def _read_sections(self):
        while not self._doc.eof():
            # 读取到下一个节(section)的内容
            data = self._read_to_next_section()
            name = data[0].strip()

            # 如果节(section)名称以 '..' 开头，则作为索引节(index section)处理
            if name.startswith('..'):  # 索引节(index section)
                yield name, data[1:]
            # 如果节(section)内容少于2行，则返回StopIteration
            elif len(data) < 2:
                yield StopIteration
            # 否则，作为普通节(section)处理，并去除开头的2行空白行
            else:
                yield name, self._strip(data[2:])

    # 解析参数列表的内容，并返回参数的列表
    def _parse_param_list(self, content):
        # 使用Reader类初始化读取器r
        r = Reader(content)
        params = []
        # 循环直到读取器r到达结尾
        while not r.eof():
            # 读取并去除头部空白的行作为参数的头部
            header = r.read().strip()
            # 如果头部包含 ' : ' 分隔符，则将参数名和参数类型分割开来
            if ' : ' in header:
                arg_name, arg_type = header.split(' : ')[:2]
            else:
                arg_name, arg_type = header, ''

            # 读取到下一个非缩进行作为参数的描述，去除缩进后返回
            desc = r.read_to_next_unindented_line()
            desc = dedent_lines(desc)

            # 将参数名、参数类型和描述添加到参数列表中
            params.append((arg_name, arg_type, desc))

        # 返回解析后的参数列表
        return params

    # 匹配名称及角色的正则表达式对象
    _name_rgx = re.compile(r"^\s*(:(?P<role>\w+):`(?P<name>[a-zA-Z0-9_.-]+)`|"
                           r" (?P<name2>[a-zA-Z0-9_.-]+))\s*", re.X)
    def _parse_see_also(self, content):
        """
        解析文档中的'see also'部分，提取相关函数和描述信息

        Args:
            content (list): 包含文档内容的列表

        Returns:
            list: 包含元组的列表，每个元组包括函数名、描述文本和角色信息

        """
        items = []

        def parse_item_name(text):
            """解析函数名称，并返回名称和角色信息（如果存在）"""
            m = self._name_rgx.match(text)
            if m:
                g = m.groups()
                if g[1] is None:
                    return g[3], None
                else:
                    return g[2], g[1]
            raise ValueError("%s 不是一个有效的项目名称" % text)

        def push_item(name, rest):
            """将解析得到的函数名、描述和角色信息添加到项目列表中"""
            if not name:
                return
            name, role = parse_item_name(name)
            if '.' not in name:
                name = '~.' + name
            items.append((name, list(rest), role))
            del rest[:]

        current_func = None
        rest = []

        for line in content:
            if not line.strip():
                continue

            m = self._name_rgx.match(line)
            if m and line[m.end():].strip().startswith(':'):
                push_item(current_func, rest)
                current_func, line = line[:m.end()], line[m.end():]
                rest = [line.split(':', 1)[1].strip()]
                if not rest[0]:
                    rest = []
            elif not line.startswith(' '):
                push_item(current_func, rest)
                current_func = None
                if ',' in line:
                    for func in line.split(','):
                        if func.strip():
                            push_item(func, [])
                elif line.strip():
                    current_func = line
            elif current_func is not None:
                rest.append(line.strip())
        push_item(current_func, rest)
        return items

    def _parse_index(self, section, content):
        """
        解析文档索引部分，提取默认和其它指定的索引信息

        Args:
            section (str): 索引部分的内容
            content (list): 包含索引内容的列表

        Returns:
            dict: 包含解析后的索引信息的字典

        """
        def strip_each_in(lst):
            """去除列表中每个元素的前后空白"""
            return [s.strip() for s in lst]

        out = {}
        section = section.split('::')
        if len(section) > 1:
            out['default'] = strip_each_in(section[1].split(','))[0]
        for line in content:
            line = line.split(':')
            if len(line) > 2:
                out[line[1]] = strip_each_in(line[2].split(','))
        return out
    # 解析摘要信息，包括函数签名和摘要内容
    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        如果当前位置是在一个节(section)的开始，则直接返回，不进行解析
        if self._is_at_section():
            return

        # 如果存在多个签名信息，选择最后一个
        while True:
            # 读取至下一个空行为止的内容，构成摘要信息
            summary = self._doc.read_to_next_empty_line()
            # 将读取到的内容连接成一个字符串并去除两侧的空格
            summary_str = " ".join([s.strip() for s in summary]).strip()
            # 如果符合函数签名的正则表达式，则将其视为函数签名
            if re.compile(r'^([\w., ]+=)?\s*[\w\.]+\(.*\)$').match(summary_str):
                self['Signature'] = summary_str
                # 如果不是在一个节(section)的开始，则继续查找下一个签名
                if not self._is_at_section():
                    continue
            break

        # 如果存在摘要信息，则存储在 self['Summary'] 中
        if summary is not None:
            self['Summary'] = summary

        # 如果当前位置不在一个节(section)的开始，则继续读取扩展摘要信息
        if not self._is_at_section():
            self['Extended Summary'] = self._read_to_next_section()

    # 解析函数文档主体
    def _parse(self):
        # 重置文档读取位置
        self._doc.reset()
        # 解析摘要信息
        self._parse_summary()

        # 读取所有节(section)并将其转换为列表
        sections = list(self._read_sections())
        # 获取所有节(section)的名称集合
        section_names = {section for section, content in sections}

        # 检查是否包含 Returns 和 Yields 节(section)
        has_returns = 'Returns' in section_names
        has_yields = 'Yields' in section_names
        # 如果同时存在 Returns 和 Yields 节(section)，则抛出异常
        if has_returns and has_yields:
            msg = 'Docstring contains both a Returns and Yields section.'
            raise ValueError(msg)

        # 遍历所有节(section)
        for (section, content) in sections:
            # 将节(section)名称格式化为首字母大写的形式
            if not section.startswith('..'):
                section = (s.capitalize() for s in section.split(' '))
                section = ' '.join(section)
            # 根据不同的节(section)名称进行处理
            if section in ('Parameters', 'Returns', 'Yields', 'Raises',
                           'Warns', 'Other Parameters', 'Attributes',
                           'Methods'):
                # 解析节(section)内容，存储在对应的 self[section] 中
                self[section] = self._parse_param_list(content)
            elif section.startswith('.. index::'):
                # 解析索引信息
                self['index'] = self._parse_index(section, content)
            elif section == 'See Also':
                # 解析 See Also 节(section)
                self['See Also'] = self._parse_see_also(content)
            else:
                # 将未分类的节(section)内容直接存储
                self[section] = content

    # 创建字符串的标题行和下划线
    def _str_header(self, name, symbol='-'):
        return [name, len(name)*symbol]

    # 缩进文本内容，用于输出格式化
    def _str_indent(self, doc, indent=4):
        out = []
        # 对每一行进行缩进处理
        for line in doc:
            out += [' '*indent + line]
        return out

    # 返回函数签名的字符串表示
    def _str_signature(self):
        if self['Signature']:
            return [self['Signature'].replace('*', r'\*')] + ['']
        else:
            return ['']

    # 返回摘要内容的字符串表示
    def _str_summary(self):
        if self['Summary']:
            return self['Summary'] + ['']
        else:
            return []

    # 返回扩展摘要内容的字符串表示
    def _str_extended_summary(self):
        if self['Extended Summary']:
            return self['Extended Summary'] + ['']
        else:
            return []
    # 生成指定参数名对应的字符串列表
    def _str_param_list(self, name):
        out = []  # 初始化空列表用于存放输出结果
        if self[name]:  # 如果参数名在当前对象中存在
            out += self._str_header(name)  # 添加参数名对应的标题字符串列表
            for param, param_type, desc in self[name]:  # 遍历参数列表
                if param_type:  # 如果参数类型存在
                    out += ['{} : {}'.format(param, param_type)]  # 添加参数名和类型到输出列表
                else:
                    out += [param]  # 否则只添加参数名
                out += self._str_indent(desc)  # 添加参数描述的缩进文本到输出列表
            out += ['']  # 添加一个空行到输出列表
        return out  # 返回最终生成的输出列表

    # 生成指定名称对应的段落字符串列表
    def _str_section(self, name):
        out = []  # 初始化空列表用于存放输出结果
        if self[name]:  # 如果指定名称在当前对象中存在
            out += self._str_header(name)  # 添加指定名称对应的标题字符串列表
            out += self[name]  # 添加指定名称对应的段落内容字符串列表
            out += ['']  # 添加一个空行到输出列表
        return out  # 返回最终生成的输出列表

    # 生成"See Also"部分的字符串列表
    def _str_see_also(self, func_role):
        if not self['See Also']:  # 如果"See Also"部分不存在
            return []  # 返回空列表
        out = []  # 初始化空列表用于存放输出结果
        out += self._str_header("See Also")  # 添加"See Also"标题字符串列表
        last_had_desc = True  # 初始化描述状态为真
        for func, desc, role in self['See Also']:  # 遍历"See Also"部分的元组列表
            if role:  # 如果存在角色信息
                link = ':{}:`{}`'.format(role, func)  # 构建带角色的链接格式
            elif func_role:  # 如果给定了函数角色信息
                link = ':{}:`{}`'.format(func_role, func)  # 构建带函数角色的链接格式
            else:
                link = "`%s`_" % func  # 否则使用默认的链接格式
            if desc or last_had_desc:  # 如果存在描述或者上一个条目有描述
                out += ['']  # 添加一个空行到输出列表
                out += [link]  # 添加链接到输出列表
            else:
                out[-1] += ", %s" % link  # 否则在上一个条目的末尾添加链接
            if desc:  # 如果存在描述信息
                out += self._str_indent([' '.join(desc)])  # 添加描述的缩进文本到输出列表
                last_had_desc = True  # 更新描述状态为真
            else:
                last_had_desc = False  # 否则更新描述状态为假
        out += ['']  # 添加一个空行到输出列表
        return out  # 返回最终生成的输出列表

    # 生成索引部分的字符串列表
    def _str_index(self):
        idx = self['index']  # 获取索引字典
        out = []  # 初始化空列表用于存放输出结果
        out += ['.. index:: %s' % idx.get('default', '')]  # 添加默认索引到输出列表
        for section, references in idx.items():  # 遍历索引字典的每个部分和引用列表
            if section == 'default':  # 如果是默认部分则跳过
                continue
            out += ['   :{}: {}'.format(section, ', '.join(references))]  # 添加部分和引用列表到输出列表
        return out  # 返回最终生成的输出列表

    # 将对象转换为字符串形式的方法
    def __str__(self, func_role=''):
        out = []  # 初始化空列表用于存放输出结果
        out += self._str_signature()  # 添加方法签名字符串列表到输出列表
        out += self._str_summary()  # 添加方法摘要字符串列表到输出列表
        out += self._str_extended_summary()  # 添加方法扩展摘要字符串列表到输出列表
        for param_list in ('Parameters', 'Returns', 'Yields',
                           'Other Parameters', 'Raises', 'Warns'):  # 遍历不同参数列表
            out += self._str_param_list(param_list)  # 添加每个参数列表的字符串列表到输出列表
        out += self._str_section('Warnings')  # 添加"Warnings"部分的字符串列表到输出列表
        out += self._str_see_also(func_role)  # 添加"See Also"部分的字符串列表到输出列表
        for s in ('Notes', 'References', 'Examples'):  # 遍历不同段落
            out += self._str_section(s)  # 添加每个段落的字符串列表到输出列表
        for param_list in ('Attributes', 'Methods'):  # 遍历不同参数列表
            out += self._str_param_list(param_list)  # 添加每个参数列表的字符串列表到输出列表
        out += self._str_index()  # 添加索引部分的字符串列表到输出列表
        return '\n'.join(out)  # 将输出列表转换为单个字符串并返回
# 定义一个函数，用于给字符串每一行添加指定数量的缩进
def indent(str, indent=4):
    indent_str = ' '*indent
    # 如果输入的字符串为 None，则返回仅包含缩进的字符串
    if str is None:
        return indent_str
    # 将输入字符串按行分割
    lines = str.split('\n')
    # 给每一行添加缩进，并将它们拼接成一个新的字符串
    return '\n'.join(indent_str + l for l in lines)


# 定义一个函数，对列表中的每行文本进行最大程度的去缩进处理
def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    # 将列表中的多行文本连接成单个字符串，然后使用 textwrap 模块进行去除缩进处理
    return textwrap.dedent("\n".join(lines)).split("\n")


# 定义一个函数，生成带有标题风格的文本
def header(text, style='-'):
    # 返回标题文本及其下方用指定字符标记的横线
    return text + '\n' + style*len(text) + '\n'


# 定义一个类 FunctionDoc，继承自 NumpyDocString 类
class FunctionDoc(NumpyDocString):
    def __init__(self, func, role='func', doc=None, config={}):
        self._f = func
        self._role = role  # 指定函数角色，如 "func" 或 "meth"

        # 如果没有提供文档字符串，则尝试从函数对象中获取
        if doc is None:
            if func is None:
                raise ValueError("No function or docstring given")
            doc = inspect.getdoc(func) or ''
        # 调用父类 NumpyDocString 的初始化方法
        NumpyDocString.__init__(self, doc)

        # 如果文档字符串中没有函数签名，并且提供了函数对象，则尝试读取函数签名
        if not self['Signature'] and func is not None:
            func, func_name = self.get_func()
            try:
                # 尝试读取函数的签名信息
                argspec = str(inspect.signature(func))
                argspec = argspec.replace('*', r'\*')
                signature = '{}{}'.format(func_name, argspec)
            except TypeError as e:
                # 如果读取失败，则使用默认的函数名作为签名
                signature = '%s()' % func_name
            # 将获取到的签名信息存入文档字符串对象中
            self['Signature'] = signature

    # 获取函数对象及其名称
    def get_func(self):
        func_name = getattr(self._f, '__name__', self.__class__.__name__)
        if inspect.isclass(self._f):
            func = getattr(self._f, '__call__', self._f.__init__)
        else:
            func = self._f
        return func, func_name

    # 返回类的字符串表示形式
    def __str__(self):
        out = ''

        # 获取函数对象及其名称
        func, func_name = self.get_func()
        signature = self['Signature'].replace('*', r'\*')

        # 定义角色映射关系
        roles = {'func': 'function',
                 'meth': 'method'}

        # 如果指定了角色，则添加角色相关信息到输出字符串中
        if self._role:
            if self._role not in roles:
                print("Warning: invalid role %s" % self._role)
            out += '.. {}:: {}\n    \n\n'.format(roles.get(self._role, ''),
                                             func_name)

        # 调用父类的 __str__ 方法，并将结果添加到输出字符串中
        out += super().__str__(func_role=self._role)
        return out


# 定义一个类 ClassDoc，继承自 NumpyDocString 类
class ClassDoc(NumpyDocString):

    # 额外公共方法列表
    extra_public_methods = ['__call__']
    # 初始化方法，用于设置对象的初始状态
    def __init__(self, cls, doc=None, modulename='', func_doc=FunctionDoc,
                 config={}):
        # 如果 cls 不是类对象且不为 None，则抛出值错误异常
        if not inspect.isclass(cls) and cls is not None:
            raise ValueError("Expected a class or None, but got %r" % cls)
        # 将传入的类对象赋给 self._cls
        self._cls = cls

        # 根据配置字典获取是否显示继承的类成员，默认为 True
        self.show_inherited_members = config.get(
                    'show_inherited_class_members', True)

        # 如果 modulename 存在且不以 '.' 结尾，则加上 '.' 结尾
        if modulename and not modulename.endswith('.'):
            modulename += '.'
        # 将处理后的 modulename 赋给 self._mod
        self._mod = modulename

        # 如果 doc 为 None，则尝试从 cls 获取文档字符串
        if doc is None:
            if cls is None:
                raise ValueError("No class or documentation string given")
            doc = pydoc.getdoc(cls)

        # 调用 NumpyDocString 类的初始化方法，用传入的 doc 初始化对象
        NumpyDocString.__init__(self, doc)

        # 如果配置中指定显示类成员，则处理类的方法和属性
        if config.get('show_class_members', True):
            # 定义一个内部函数，用于将字符串按行拆分为列表
            def splitlines_x(s):
                if not s:
                    return []
                else:
                    return s.splitlines()

            # 遍历处理方法和属性
            for field, items in [('Methods', self.methods),
                                 ('Attributes', self.properties)]:
                # 如果当前字段没有内容，则初始化一个空列表
                if not self[field]:
                    doc_list = []
                    # 遍历属性或方法列表，获取对应的类对象并获取文档
                    for name in sorted(items):
                        clsname = getattr(self._cls, name, None)
                        if clsname is not None:
                            doc_item = pydoc.getdoc(clsname)
                            doc_list.append((name, '', splitlines_x(doc_item)))
                    # 将获取到的文档列表赋给当前字段
                    self[field] = doc_list

    # methods 属性的装饰器，用于返回类的方法列表
    @property
    def methods(self):
        # 如果 self._cls 为 None，则返回空列表
        if self._cls is None:
            return []
        # 返回类中所有公共方法的名称列表
        return [name for name, func in inspect.getmembers(self._cls)
                if ((not name.startswith('_')
                     or name in self.extra_public_methods)
                    and callable(func))]

    # properties 属性的装饰器，用于返回类的属性列表
    @property
    def properties(self):
        # 如果 self._cls 为 None，则返回空列表
        if self._cls is None:
            return []
        # 返回类中所有公共属性的名称列表
        return [name for name, func in inspect.getmembers(self._cls)
                if not name.startswith('_') and func is None]
```