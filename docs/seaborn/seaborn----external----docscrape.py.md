# `D:\src\scipysrc\seaborn\seaborn\external\docscrape.py`

```
"""
Extract reference documentation from the NumPy source tree.

Copyright (C) 2008 Stefan van der Walt <stefan@mentat.za.net>, Pauli Virtanen <pav@iki.fi>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""
import inspect  # 导入 inspect 模块，用于访问对象的属性和方法
import textwrap  # 导入 textwrap 模块，用于格式化文本段落
import re  # 导入 re 模块，用于正则表达式操作
import pydoc  # 导入 pydoc 模块，用于生成文档
from warnings import warn  # 从 warnings 模块导入 warn 函数，用于发出警告
from collections import namedtuple  # 导入 namedtuple 类型，用于创建具名元组
from collections.abc import Callable, Mapping  # 导入 Callable 和 Mapping 类型，用于抽象基类
import copy  # 导入 copy 模块，用于复制对象
import sys  # 导入 sys 模块，用于与解释器进行交互


def strip_blank_lines(l):
    "Remove leading and trailing blank lines from a list of lines"
    while l and not l[0].strip():
        del l[0]
    while l and not l[-1].strip():
        del l[-1]
    return l


class Reader:
    """A line-based string reader.

    """
    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           String with lines separated by '\n'.

        """
        if isinstance(data, list):
            self._str = data  # 如果 data 是 list 类型，则直接使用该列表作为内部数据
        else:
            self._str = data.split('\n')  # 否则，将 data 按行分割存储为列表

        self.reset()  # 初始化对象状态

    def __getitem__(self, n):
        return self._str[n]  # 获取第 n 行的文本行

    def reset(self):
        self._l = 0  # 当前行号，初始为 0

    def read(self):
        if not self.eof():
            out = self[self._l]  # 获取当前行内容
            self._l += 1  # 行号加一，准备读取下一行
            return out
        else:
            return ''  # 如果已经到达末尾，返回空字符串

    def seek_next_non_empty_line(self):
        for l in self[self._l:]:
            if l.strip():
                break  # 找到第一个非空行，退出循环
            else:
                self._l += 1  # 继续向下搜索空行

    def eof(self):
        return self._l >= len(self._str)  # 判断是否已经到达文本末尾

    def read_to_condition(self, condition_func):
        start = self._l  # 记录起始行号
        for line in self[start:]:
            if condition_func(line):
                return self[start:self._l]  # 如果满足条件，则返回起始到当前行之间的内容
            self._l += 1  # 否则，继续向下读取
            if self.eof():
                return self[start:self._l+1]  # 如果到达末尾仍未找到满足条件的行，则返回起始到当前行之间的内容
        return []  # 如果没有找到符合条件的内容，则返回空列表
    # 从当前位置读取直到下一个空行，并返回读取到的内容
    def read_to_next_empty_line(self):
        # 跳过到下一个非空行的位置
        self.seek_next_non_empty_line()

        # 定义用于检查行是否为空行的函数
        def is_empty(line):
            return not line.strip()

        # 调用通用的读取函数，根据条件读取内容直到遇到空行
        return self.read_to_condition(is_empty)

    # 从当前位置读取直到下一个非缩进行，并返回读取到的内容
    def read_to_next_unindented_line(self):
        # 定义用于检查行是否为非缩进行的函数
        def is_unindented(line):
            return line.strip() and (len(line.lstrip()) == len(line))

        # 调用通用的读取函数，根据条件读取内容直到遇到非缩进行
        return self.read_to_condition(is_unindented)

    # 返回当前位置加上指定偏移量处的字符，若超出字符串长度则返回空字符
    def peek(self, n=0):
        if self._l + n < len(self._str):
            return self[self._l + n]
        else:
            return ''

    # 检查当前对象所持有的字符串是否为空
    def is_empty(self):
        return not ''.join(self._str).strip()
# 自定义异常类，用于表示解析错误
class ParseError(Exception):
    def __str__(self):
        message = self.args[0]
        # 如果异常对象具有属性 'docstring'，则在错误信息中添加文档字符串的描述
        if hasattr(self, 'docstring'):
            message = f"{message} in {self.docstring!r}"
        return message


# 使用 namedtuple 定义一个名为 Parameter 的命名元组，包含 name、type 和 desc 三个字段
Parameter = namedtuple('Parameter', ['name', 'type', 'desc'])


# 定义一个名为 NumpyDocString 的类，继承自 Mapping 类
class NumpyDocString(Mapping):
    """Parses a numpydoc string to an abstract representation

    Instances define a mapping from section title to structured data.
    """

    # 定义类属性 sections，用于存储各部分标题到结构化数据的映射
    sections = {
        'Signature': '',
        'Summary': [''],
        'Extended Summary': [],
        'Parameters': [],
        'Returns': [],
        'Yields': [],
        'Receives': [],
        'Raises': [],
        'Warns': [],
        'Other Parameters': [],
        'Attributes': [],
        'Methods': [],
        'See Also': [],
        'Notes': [],
        'Warnings': [],
        'References': '',
        'Examples': '',
        'index': {}
    }

    # 初始化方法，接受 docstring 和可选的 config 参数
    def __init__(self, docstring, config={}):
        # 备份原始 docstring
        orig_docstring = docstring
        # 去除 docstring 的缩进并按行分割成列表
        docstring = textwrap.dedent(docstring).split('\n')

        # 使用 Reader 类封装 docstring 列表，创建 self._doc 对象
        self._doc = Reader(docstring)
        # 深拷贝 sections 字典，初始化 self._parsed_data
        self._parsed_data = copy.deepcopy(self.sections)

        try:
            # 调用 _parse 方法解析 docstring
            self._parse()
        except ParseError as e:
            # 如果解析过程中出现异常，将异常的 docstring 属性设置为原始 docstring，并重新抛出异常
            e.docstring = orig_docstring
            raise

    # 实现 Mapping 接口的 __getitem__ 方法，用于获取指定键的值
    def __getitem__(self, key):
        return self._parsed_data[key]

    # 实现 Mapping 接口的 __setitem__ 方法，用于设置指定键的值
    def __setitem__(self, key, val):
        if key not in self._parsed_data:
            self._error_location(f"Unknown section {key}", error=False)
        else:
            self._parsed_data[key] = val

    # 实现 Mapping 接口的 __iter__ 方法，返回迭代器遍历 _parsed_data 的键
    def __iter__(self):
        return iter(self._parsed_data)

    # 实现 Mapping 接口的 __len__ 方法，返回 _parsed_data 的长度
    def __len__(self):
        return len(self._parsed_data)

    # 判断当前位置是否处于一个文档部分的起始位置
    def _is_at_section(self):
        # 将文档指针移到下一个非空行
        self._doc.seek_next_non_empty_line()

        # 如果已经到达文档末尾，返回 False
        if self._doc.eof():
            return False

        # 获取当前行内容并去除首尾空白
        l1 = self._doc.peek().strip()  # 例如 'Parameters'

        # 如果当前行以 '.. index::' 开头，返回 True
        if l1.startswith('.. index::'):
            return True

        # 获取下一行内容并去除首尾空白
        l2 = self._doc.peek(1).strip()  # '----------' 或 '=========='
        
        # 判断下一行是否由 '-' 或 '=' 构成，并且长度与 l1 相同
        return l2.startswith('-'*len(l1)) or l2.startswith('='*len(l1))

    # 去除文档的前后空白行
    def _strip(self, doc):
        i = 0
        j = 0
        # 找到第一行不为空的位置 i
        for i, line in enumerate(doc):
            if line.strip():
                break

        # 找到最后一行不为空的位置 j
        for j, line in enumerate(doc[::-1]):
            if line.strip():
                break

        # 返回去除前后空白行后的文档片段
        return doc[i:len(doc)-j]

    # 读取直到下一个文档部分的内容
    def _read_to_next_section(self):
        # 读取当前部分的内容
        section = self._doc.read_to_next_empty_line()

        # 循环直到下一个部分的起始位置或文档末尾
        while not self._is_at_section() and not self._doc.eof():
            # 如果上一行为空行，则添加空行到当前部分
            if not self._doc.peek(-1).strip():  # 上一行为空
                section += ['']

            # 读取到下一个空行为止，添加到当前部分
            section += self._doc.read_to_next_empty_line()

        return section
    # 当尚未到达文档末尾时，循环读取各个部分的数据
    def _read_sections(self):
        while not self._doc.eof():
            # 读取到下一个部分的数据
            data = self._read_to_next_section()
            # 去除第一个元素的首尾空格，作为部分名称
            name = data[0].strip()

            # 如果部分名称以'..'开头，表示索引部分
            if name.startswith('..'):
                # 返回名称和剩余数据（不包括名称本身）
                yield name, data[1:]
            # 如果数据长度小于2，则返回StopIteration
            elif len(data) < 2:
                yield StopIteration
            else:
                # 否则，返回名称和处理后的数据（去除了缩进）
                yield name, self._strip(data[2:])

    # 解析参数列表的方法
    def _parse_param_list(self, content, single_element_is_type=False):
        # 使用Reader对象读取内容
        r = Reader(content)
        # 初始化参数列表
        params = []
        while not r.eof():
            # 读取并去除首尾空格
            header = r.read().strip()
            # 如果包含' : '，则将其分割为参数名和参数类型
            if ' : ' in header:
                arg_name, arg_type = header.split(' : ')[:2]
            else:
                # 否则，根据single_element_is_type决定如何处理
                if single_element_is_type:
                    arg_name, arg_type = '', header
                else:
                    arg_name, arg_type = header, ''

            # 读取到下一个非缩进行，并对描述进行缩进和空行处理
            desc = r.read_to_next_unindented_line()
            desc = dedent_lines(desc)
            desc = strip_blank_lines(desc)

            # 将参数信息添加到参数列表中
            params.append(Parameter(arg_name, arg_type, desc))

        return params

    # 下面是正则表达式用来匹配和解析函数格式的说明
    #
    # <FUNCNAME>
    # <FUNCNAME> SPACE* COLON SPACE+ <DESC> SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)+ (COMMA | PERIOD)? SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)* SPACE* COLON SPACE+ <DESC> SPACE*
    #
    # <FUNCNAME> 表示以下格式之一
    #   <PLAIN_FUNCNAME>
    #   COLON <ROLE> COLON BACKTICK <PLAIN_FUNCNAME> BACKTICK
    # 其中
    #   <PLAIN_FUNCNAME> 是合法的函数名称，而
    #   <ROLE> 是任意非空的词字符序列。
    # 例如：func_f1  :meth:`func_h1` :obj:`~baz.obj_r` :class:`class_j`
    # <DESC> 是描述函数的字符串。

    # 匹配具有角色标识符的函数名称
    _role = r":(?P<role>\w+):"
    # 匹配使用反引号标识的函数名称
    _funcbacktick = r"`(?P<name>(?:~\w+\.)?[a-zA-Z0-9_\.-]+)`"
    # 匹配普通的函数名称
    _funcplain = r"(?P<name2>[a-zA-Z0-9_\.-]+)"
    # 匹配函数名称，包括角色标识符和反引号标识
    _funcname = r"(" + _role + _funcbacktick + r"|" + _funcplain + r")"
    # 用于匹配下一个可能的函数名称（更具体的）
    _funcnamenext = _funcname.replace('role', 'rolenext')
    _funcnamenext = _funcnamenext.replace('name', 'namenext')
    # 匹配描述部分，可能为空
    _description = r"(?P<description>\s*:(\s+(?P<desc>\S+.*))?)?\s*$"
    # 用于匹配整行的正则表达式，包括函数名称和描述
    _func_rgx = re.compile(r"^\s*" + _funcname + r"\s*")
    # 用于匹配包含多个函数名称和描述的行
    _line_rgx = re.compile(
        r"^\s*" +
        r"(?P<allfuncs>" +        # group for all function names
        _funcname +
        r"(?P<morefuncs>([,]\s+" + _funcnamenext + r")*)" +
        r")" +                     # end of "allfuncs"
        r"(?P<trailing>[,\.])?" +   # Some function lists have a trailing comma (or period)
        _description)

    # 空的<DESC>元素被替换为'..'
    empty_description = '..'
    def _parse_see_also(self, content):
        """
        解析给定内容的函数关联信息。

        Args:
            content (list of str): 包含函数关联信息的内容列表

        Returns:
            list of tuple: 元组列表，每个元组包含函数列表和剩余描述信息

        Raises:
            ParseError: 如果解析过程中发现不符合预期的行格式

        """

        items = []

        def parse_item_name(text):
            """匹配 ':role:`name`' 或 'name'."""
            m = self._func_rgx.match(text)
            if not m:
                raise ParseError(f"{text} 不是一个有效的项目名")
            role = m.group('role')
            name = m.group('name') if role else m.group('name2')
            return name, role, m.end()

        rest = []
        for line in content:
            if not line.strip():
                continue

            line_match = self._line_rgx.match(line)
            description = None
            if line_match:
                description = line_match.group('desc')
                if line_match.group('trailing') and description:
                    self._error_location(
                        '在函数列表后发现意外的逗号或句号，位于行 "%s" 的索引 %d 处' %
                        (line, line_match.end('trailing')),
                        error=False)
            if not description and line.startswith(' '):
                rest.append(line.strip())
            elif line_match:
                funcs = []
                text = line_match.group('allfuncs')
                while True:
                    if not text.strip():
                        break
                    name, role, match_end = parse_item_name(text)
                    funcs.append((name, role))
                    text = text[match_end:].strip()
                    if text and text[0] == ',':
                        text = text[1:].strip()
                rest = list(filter(None, [description]))
                items.append((funcs, rest))
            else:
                raise ParseError(f"{line} 不是一个有效的项目名")
        return items

    def _parse_index(self, section, content):
        """
        解析给定部分和内容的索引信息。

        Args:
            section (str): 索引部分的内容
            content (list of str): 包含索引信息的内容列表

        Returns:
            dict: 解析后的索引信息字典

        """

        def strip_each_in(lst):
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
    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        # 如果当前位置处于一个小节的开始，则直接返回，不进行解析
        if self._is_at_section():
            return

        # 如果存在多个签名，选择最后一个
        while True:
            # 读取到下一个空行为止，作为摘要
            summary = self._doc.read_to_next_empty_line()
            # 将摘要内容连接成一个字符串并去除首尾空格
            summary_str = " ".join([s.strip() for s in summary]).strip()
            # 编译正则表达式，用于匹配函数签名的格式
            compiled = re.compile(r'^([\w., ]+=)?\s*[\w\.]+\(.*\)$')
            # 如果摘要符合函数签名的格式，则将其作为签名存储
            if compiled.match(summary_str):
                self['Signature'] = summary_str
                # 如果当前位置不是小节的开始，则继续查找
                if not self._is_at_section():
                    continue
            break

        # 如果存在摘要，则将其存储到字典中
        if summary is not None:
            self['Summary'] = summary

        # 如果当前位置不是小节的开始，则读取到下一个小节为止的内容，并存储为扩展摘要
        if not self._is_at_section():
            self['Extended Summary'] = self._read_to_next_section()

    def _parse(self):
        # 重置文档解析器状态
        self._doc.reset()
        # 解析摘要部分
        self._parse_summary()

        # 读取所有小节并将小节名称存储到集合中
        sections = list(self._read_sections())
        section_names = {section for section, content in sections}

        # 检查是否存在 Returns 和 Yields 小节
        has_returns = 'Returns' in section_names
        has_yields = 'Yields' in section_names
        # 如果同时存在 Returns 和 Yields 小节，则抛出异常
        if has_returns and has_yields:
            msg = 'Docstring contains both a Returns and Yields section.'
            raise ValueError(msg)
        # 如果不存在 Yields 小节但存在 Receives 小节，则抛出异常
        if not has_yields and 'Receives' in section_names:
            msg = 'Docstring contains a Receives section but not Yields.'
            raise ValueError(msg)

        # 遍历所有小节及其内容，根据小节类型进行解析和存储
        for (section, content) in sections:
            # 如果小节名称不是以 '..' 开头，则首字母大写并检查是否重复
            if not section.startswith('..'):
                section = (s.capitalize() for s in section.split(' '))
                section = ' '.join(section)
                if self.get(section):
                    self._error_location(f"The section {section} appears twice")

            # 根据小节名称类型进行解析和存储
            if section in ('Parameters', 'Other Parameters', 'Attributes',
                           'Methods'):
                self[section] = self._parse_param_list(content)
            elif section in ('Returns', 'Yields', 'Raises', 'Warns', 'Receives'):
                self[section] = self._parse_param_list(
                    content, single_element_is_type=True)
            elif section.startswith('.. index::'):
                self['index'] = self._parse_index(section, content)
            elif section == 'See Also':
                self['See Also'] = self._parse_see_also(content)
            else:
                self[section] = content

    def _error_location(self, msg, error=True):
        # 如果存在 _obj 属性，则尝试获取其源文件名
        if hasattr(self, '_obj'):
            # 我们知道文档的来源位置
            try:
                filename = inspect.getsourcefile(self._obj)
            except TypeError:
                filename = None
            # 在异常消息中添加文档所属对象的信息和文件名
            msg = msg + f" in the docstring of {self._obj} in {filename}."
        # 如果是错误，则抛出 ValueError 异常；否则发出警告
        if error:
            raise ValueError(msg)
        else:
            warn(msg)

    # 字符串转换函数

    def _str_header(self, name, symbol='-'):
        # 返回一个由名称和重复符号构成的列表
        return [name, len(name)*symbol]
    # 将给定文档（doc）中的每一行字符串进行缩进处理，并返回处理后的列表
    def _str_indent(self, doc, indent=4):
        out = []
        for line in doc:
            out += [' '*indent + line]
        return out

    # 构造并返回函数签名的字符串列表，如果存在签名，则将 '*' 替换为 '\*'
    def _str_signature(self):
        if self['Signature']:
            return [self['Signature'].replace('*', r'\*')] + ['']
        else:
            return ['']

    # 返回函数的简要概述信息的字符串列表
    def _str_summary(self):
        if self['Summary']:
            return self['Summary'] + ['']
        else:
            return []

    # 返回函数的扩展概述信息的字符串列表
    def _str_extended_summary(self):
        if self['Extended Summary']:
            return self['Extended Summary'] + ['']
        else:
            return []

    # 返回参数列表的字符串表示形式的列表，包括参数名、类型和描述信息
    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            for param in self[name]:
                parts = []
                if param.name:
                    parts.append(param.name)
                if param.type:
                    parts.append(param.type)
                out += [' : '.join(parts)]
                if param.desc and ''.join(param.desc).strip():
                    out += self._str_indent(param.desc)
            out += ['']
        return out

    # 返回指定部分（如参数、返回值等）的字符串表示形式的列表
    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += self[name]
            out += ['']
        return out

    # 返回"See Also"部分的字符串表示形式的列表，包括相关函数的链接和描述
    def _str_see_also(self, func_role):
        if not self['See Also']:
            return []
        out = []
        out += self._str_header("See Also")
        out += ['']
        last_had_desc = True
        for funcs, desc in self['See Also']:
            assert isinstance(funcs, list)
            links = []
            for func, role in funcs:
                if role:
                    link = f':{role}:`{func}`'
                elif func_role:
                    link = f':{func_role}:`{func}`'
                else:
                    link = f"`{func}`_"
                links.append(link)
            link = ', '.join(links)
            out += [link]
            if desc:
                out += self._str_indent([' '.join(desc)])
                last_had_desc = True
            else:
                last_had_desc = False
                out += self._str_indent([self.empty_description])

        if last_had_desc:
            out += ['']
        out += ['']
        return out

    # 返回索引部分的字符串表示形式的列表，包括默认索引和各个章节的索引
    def _str_index(self):
        idx = self['index']
        out = []
        output_index = False
        default_index = idx.get('default', '')
        if default_index:
            output_index = True
        out += [f'.. index:: {default_index}']
        for section, references in idx.items():
            if section == 'default':
                continue
            output_index = True
            out += [f"   :{section}: {', '.join(references)}"]
        if output_index:
            return out
        else:
            return ''
    # 定义一个特殊方法 __str__，用于返回对象的字符串表示
    def __str__(self, func_role=''):
        # 初始化一个空列表，用于存储生成的字符串表示的各部分
        out = []
        # 获取对象的签名字符串，并添加到输出列表中
        out += self._str_signature()
        # 获取对象的摘要信息字符串，并添加到输出列表中
        out += self._str_summary()
        # 获取对象的扩展摘要信息字符串，并添加到输出列表中
        out += self._str_extended_summary()
        # 遍历不同参数列表，如'Parameters', 'Returns'等，生成各自的字符串表示并添加到输出列表中
        for param_list in ('Parameters', 'Returns', 'Yields', 'Receives',
                           'Other Parameters', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        # 获取对象的警告信息字符串，并添加到输出列表中
        out += self._str_section('Warnings')
        # 根据给定的角色字符串，生成对象的相关参考链接信息字符串并添加到输出列表中
        out += self._str_see_also(func_role)
        # 遍历特定部分，如'Notes', 'References', 'Examples'，生成各自的字符串表示并添加到输出列表中
        for s in ('Notes', 'References', 'Examples'):
            out += self._str_section(s)
        # 遍历属性和方法参数列表，生成各自的字符串表示并添加到输出列表中
        for param_list in ('Attributes', 'Methods'):
            out += self._str_param_list(param_list)
        # 获取对象的索引信息字符串，并添加到输出列表中
        out += self._str_index()
        # 将所有生成的字符串表示通过换行符连接成最终的字符串，并返回
        return '\n'.join(out)
# 定义一个函数，用于缩进给定的字符串，默认缩进四个空格
def indent(str, indent=4):
    # 根据指定的缩进数量生成缩进字符串
    indent_str = ' ' * indent
    # 如果输入的字符串为空，返回仅含指定缩进的字符串
    if str is None:
        return indent_str
    # 将输入字符串按行分割
    lines = str.split('\n')
    # 将每一行字符串添加指定的缩进后，再以换行符连接起来
    return '\n'.join(indent_str + l for l in lines)


# 定义一个函数，用于将给定的行列表最大程度地去除缩进
def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    # 将输入的行列表合并成一个字符串，然后尝试最大程度地去除缩进
    return textwrap.dedent("\n".join(lines)).split("\n")


# 定义一个函数，用于生成带有样式线的标题文本
def header(text, style='-'):
    # 返回标题文本和使用样式字符重复文本长度的线
    return text + '\n' + style * len(text) + '\n'


# 定义一个类，继承自NumpyDocString，用于生成函数文档字符串
class FunctionDoc(NumpyDocString):
    def __init__(self, func, role='func', doc=None, config={}):
        self._f = func
        self._role = role  # 设置角色，如"func"或"meth"

        # 如果未提供文档字符串，尝试从函数中获取或设置为空字符串
        if doc is None:
            if func is None:
                raise ValueError("No function or docstring given")
            doc = inspect.getdoc(func) or ''
        # 调用父类的初始化方法，传入文档字符串和配置信息
        NumpyDocString.__init__(self, doc, config)

        # 如果未设置函数签名，并且函数不为空，则尝试获取函数的签名
        if not self['Signature'] and func is not None:
            func, func_name = self.get_func()
            try:
                try:
                    signature = str(inspect.signature(func))
                except (AttributeError, ValueError):
                    # 尝试读取签名，向后兼容旧版本的Python
                    if sys.version_info[0] >= 3:
                        argspec = inspect.getfullargspec(func)
                    else:
                        argspec = inspect.getargspec(func)
                    signature = inspect.formatargspec(*argspec)
                # 构建函数签名
                signature = f'{func_name}{signature}'
            except TypeError:
                signature = f'{func_name}()'
            # 设置文档字符串的签名部分
            self['Signature'] = signature

    # 获取函数对象及其名称的方法
    def get_func(self):
        func_name = getattr(self._f, '__name__', self.__class__.__name__)
        if inspect.isclass(self._f):
            func = getattr(self._f, '__call__', self._f.__init__)
        else:
            func = self._f
        return func, func_name

    # 将对象转换为字符串表示形式的方法
    def __str__(self):
        out = ''

        # 获取函数对象及其名称
        func, func_name = self.get_func()

        # 角色映射字典，用于生成文档字符串的角色行
        roles = {'func': 'function',
                 'meth': 'method'}

        # 如果指定了角色，确保角色有效，并添加角色行到输出中
        if self._role:
            if self._role not in roles:
                print(f"Warning: invalid role {self._role}")
            out += f".. {roles.get(self._role, '')}:: {func_name}\n    \n\n"

        # 调用父类的__str__方法生成文档字符串的其余部分，并添加到输出中
        out += super().__str__(func_role=self._role)
        return out


# 定义一个类，继承自NumpyDocString，用于生成类文档字符串
class ClassDoc(NumpyDocString):

    # 额外的公共方法列表，用于类文档字符串
    extra_public_methods = ['__call__']
    # 初始化方法，接受多个参数用于配置对象的属性
    def __init__(self, cls, doc=None, modulename='', func_doc=FunctionDoc,
                 config={}):
        # 检查 cls 是否为类或 None，否则引发异常
        if not inspect.isclass(cls) and cls is not None:
            raise ValueError(f"Expected a class or None, but got {cls!r}")
        self._cls = cls

        # 如果在 sys.modules 中存在 'sphinx' 模块，则导入其 ALL 变量，否则设为 object()
        if 'sphinx' in sys.modules:
            from sphinx.ext.autodoc import ALL
        else:
            ALL = object()

        # 获取配置字典中的 show_inherited_class_members 键对应的值，默认为 True
        self.show_inherited_members = config.get(
                    'show_inherited_class_members', True)

        # 如果 modulename 存在且不以 '.' 结尾，则在其末尾添加 '.'
        if modulename and not modulename.endswith('.'):
            modulename += '.'
        self._mod = modulename

        # 如果未提供 doc 参数，则尝试从 cls 中获取文档字符串
        if doc is None:
            if cls is None:
                raise ValueError("No class or documentation string given")
            doc = pydoc.getdoc(cls)

        # 使用 NumpyDocString 初始化当前对象，传入 doc 参数作为文档字符串
        NumpyDocString.__init__(self, doc)

        # 获取配置字典中的 members 键对应的值，默认为空列表
        _members = config.get('members', [])
        # 如果 _members 等于 ALL，则将其设为 None
        if _members is ALL:
            _members = None
        # 获取配置字典中的 exclude-members 键对应的值，默认为空列表
        _exclude = config.get('exclude-members', [])

        # 如果配置字典中的 show_class_members 键对应的值为 True，并且 exclude-members 不等于 ALL
        if config.get('show_class_members', True) and _exclude is not ALL:
            # 定义内部函数 splitlines_x，用于将字符串按行分割为列表
            def splitlines_x(s):
                if not s:
                    return []
                else:
                    return s.splitlines()
            # 遍历 [('Methods', self.methods), ('Attributes', self.properties)] 列表
            for field, items in [('Methods', self.methods),
                                 ('Attributes', self.properties)]:
                # 如果当前字段为空，则初始化为空列表
                if not self[field]:
                    doc_list = []
                    # 遍历 items 中的元素名字，按名称排序
                    for name in sorted(items):
                        # 如果 name 在 _exclude 中，或者 _members 存在且 name 不在 _members 中，则跳过当前循环
                        if (name in _exclude or
                                (_members and name not in _members)):
                            continue
                        try:
                            # 尝试获取属性或方法的文档字符串，并将其添加到 doc_list 中作为 Parameter 对象
                            doc_item = pydoc.getdoc(getattr(self._cls, name))
                            doc_list.append(
                                Parameter(name, '', splitlines_x(doc_item)))
                        except AttributeError:
                            pass  # method doesn't exist
                    # 将 doc_list 赋值给当前字段（Methods 或 Attributes）
                    self[field] = doc_list

    # 方法装饰器，定义 methods 属性，返回当前类的所有方法名列表
    @property
    def methods(self):
        # 如果 self._cls 为 None，则返回空列表
        if self._cls is None:
            return []
        # 使用 inspect.getmembers 获取 self._cls 中的成员，并过滤得到所有公共方法名列表
        return [name for name, func in inspect.getmembers(self._cls)
                if ((not name.startswith('_')
                     or name in self.extra_public_methods)
                    and isinstance(func, Callable)
                    and self._is_show_member(name))]

    # 方法装饰器，定义 properties 属性，返回当前类的所有属性名列表
    @property
    def properties(self):
        # 如果 self._cls 为 None，则返回空列表
        if self._cls is None:
            return []
        # 使用 inspect.getmembers 获取 self._cls 中的成员，并过滤得到所有公共属性名列表
        return [name for name, func in inspect.getmembers(self._cls)
                if (not name.startswith('_') and
                    (func is None or isinstance(func, property) or
                     inspect.isdatadescriptor(func))
                    and self._is_show_member(name))]

    # 内部方法，用于判断是否显示指定名称的成员
    def _is_show_member(self, name):
        # 如果允许显示继承成员，则返回 True
        if self.show_inherited_members:
            return True  # show all class members
        # 否则，如果 name 不在 self._cls 的 __dict__ 中，则返回 False（表示继承成员，不显示）
        if name not in self._cls.__dict__:
            return False  # class member is inherited, we do not show it
        # 否则返回 True
        return True
```