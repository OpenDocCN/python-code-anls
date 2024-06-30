# `D:\src\scipysrc\scipy\scipy\_lib\_docscrape.py`

```
"""
Extract reference documentation from the NumPy source tree.

"""
# 导入所需模块和库
import inspect                 # 用于检查对象信息
import textwrap                # 提供文本包装和填充功能
import re                      # 提供正则表达式匹配操作
import pydoc                   # 生成Python文档的工具
from warnings import warn      # 提供警告功能
from collections import namedtuple  # 命名元组，创建带有命名字段的元组子类
from collections.abc import Callable, Mapping  # 导入抽象基类
import copy                    # 提供深浅复制操作
import sys                     # 提供对Python解释器的访问和控制


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
           String with lines separated by '\\n'.

        """
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split('\n')  # 将字符串分割为行列表，并存储在实例变量中

        self.reset()  # 初始化重置当前行号

    def __getitem__(self, n):
        return self._str[n]

    def reset(self):
        self._l = 0  # 当前行号初始化为0

    def read(self):
        if not self.eof():
            out = self[self._l]  # 获取当前行内容
            self._l += 1  # 移动到下一行
            return out
        else:
            return ''  # 如果已经到达末尾，返回空字符串

    def seek_next_non_empty_line(self):
        for l in self[self._l:]:
            if l.strip():  # 如果行内容不为空
                break
            else:
                self._l += 1  # 否则继续下一行

    def eof(self):
        return self._l >= len(self._str)  # 检查是否到达文件末尾

    def read_to_condition(self, condition_func):
        start = self._l  # 记录开始位置
        for line in self[start:]:
            if condition_func(line):  # 根据条件函数判断是否满足条件
                return self[start:self._l]  # 返回满足条件的部分内容
            self._l += 1  # 继续下一行
            if self.eof():  # 如果到达文件末尾
                return self[start:self._l+1]  # 返回当前到结尾的内容
        return []  # 如果未找到满足条件的内容，返回空列表

    def read_to_next_empty_line(self):
        self.seek_next_non_empty_line()  # 移动到下一个非空行

        def is_empty(line):
            return not line.strip()  # 判断是否为空行

        return self.read_to_condition(is_empty)  # 返回到下一个空行的内容

    def read_to_next_unindented_line(self):
        def is_unindented(line):
            return line.strip() and (len(line.lstrip()) == len(line))  # 判断是否为非缩进行
        return self.read_to_condition(is_unindented)  # 返回到下一个非缩进行的内容

    def peek(self, n=0):
        if self._l + n < len(self._str):
            return self[self._l + n]  # 返回指定偏移量的行内容
        else:
            return ''  # 如果超出范围，返回空字符串

    def is_empty(self):
        return not ''.join(self._str).strip()  # 判断整个字符串是否为空


class ParseError(Exception):
    def __str__(self):
        message = self.args[0]
        if hasattr(self, 'docstring'):
            message = f"{message} in {self.docstring!r}"  # 如果有docstring属性，添加在错误信息中
        return message


Parameter = namedtuple('Parameter', ['name', 'type', 'desc'])  # 定义一个命名元组Parameter，包含name、type、desc字段


class NumpyDocString(Mapping):
    """Parses a numpydoc string to an abstract representation

    Instances define a mapping from section title to structured data.

    """
    # 定义一个字典，包含文档中可能出现的各种部分，每个部分都初始化为空或空列表
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

    # 初始化方法，接受文档字符串和配置字典作为参数
    def __init__(self, docstring, config={}):
        # 备份原始文档字符串
        orig_docstring = docstring
        # 去除文档字符串的缩进，并按行拆分成列表
        docstring = textwrap.dedent(docstring).split('\n')

        # 创建文档阅读器对象
        self._doc = Reader(docstring)
        # 深度复制预定义的部分字典，以便于后续解析使用
        self._parsed_data = copy.deepcopy(self.sections)

        try:
            # 解析文档内容
            self._parse()
        except ParseError as e:
            # 捕获解析错误，并将错误信息回溯到原始文档字符串
            e.docstring = orig_docstring
            raise

    # 实现获取操作符号[]的方法，返回指定键的解析数据
    def __getitem__(self, key):
        return self._parsed_data[key]

    # 实现设置操作符号[]=的方法，设置指定键的解析数据
    def __setitem__(self, key, val):
        # 如果键不存在于解析数据中，抛出错误提示未知部分
        if key not in self._parsed_data:
            self._error_location("Unknown section %s" % key, error=False)
        else:
            self._parsed_data[key] = val

    # 实现迭代操作符号iterator的方法，返回解析数据的迭代器
    def __iter__(self):
        return iter(self._parsed_data)

    # 实现长度操作符号len()的方法，返回解析数据的长度
    def __len__(self):
        return len(self._parsed_data)

    # 检查当前位置是否位于一个新的部分开始
    def _is_at_section(self):
        # 移动到下一个非空行
        self._doc.seek_next_non_empty_line()

        # 如果已经到达文档末尾，返回 False
        if self._doc.eof():
            return False

        # 读取当前行，并去除首尾空白
        l1 = self._doc.peek().strip()  # 例如 Parameters

        # 如果当前行以 '.. index::' 开头，说明是索引部分
        if l1.startswith('.. index::'):
            return True

        # 否则，读取下一行并去除首尾空白，检查是否由连字符或等号构成的分隔线
        l2 = self._doc.peek(1).strip()  # ---------- or ==========
        return l2.startswith('-'*len(l1)) or l2.startswith('='*len(l1))

    # 去除文档片段两端的空行
    def _strip(self, doc):
        i = 0
        j = 0
        # 找到第一个非空行的索引
        for i, line in enumerate(doc):
            if line.strip():
                break

        # 找到最后一个非空行的索引
        for j, line in enumerate(doc[::-1]):
            if line.strip():
                break

        # 返回去除两端空行后的文档片段
        return doc[i:len(doc)-j]

    # 读取到下一个部分开始的内容
    def _read_to_next_section(self):
        # 读取到下一个空行之前的内容，作为一个部分的开始
        section = self._doc.read_to_next_empty_line()

        # 循环直到遇到一个新部分开始或文档结束
        while not self._is_at_section() and not self._doc.eof():
            # 如果上一行是空行，则添加一个空行到当前部分
            if not self._doc.peek(-1).strip():  # previous line was empty
                section += ['']

            # 继续读取到下一个空行之前的内容，添加到当前部分
            section += self._doc.read_to_next_empty_line()

        return section

    # 读取整个文档的各个部分内容
    def _read_sections(self):
        # 循环直到文档结束
        while not self._doc.eof():
            # 读取到下一个部分开始的内容
            data = self._read_to_next_section()
            # 获取当前部分的名称，去除首尾空白
            name = data[0].strip()

            # 如果部分名称以 '..' 开头，表示索引部分
            if name.startswith('..'):
                yield name, data[1:]  # 返回索引部分的名称和内容
            # 否则，如果部分内容少于两行，结束迭代
            elif len(data) < 2:
                yield StopIteration
            else:
                # 否则，返回部分名称和去除两端空行后的内容
                yield name, self._strip(data[2:])
    # 解析参数列表的方法，从给定的内容中提取参数信息并返回参数对象列表
    def _parse_param_list(self, content, single_element_is_type=False):
        # 创建一个Reader对象来处理给定的内容
        r = Reader(content)
        # 初始化参数列表
        params = []
        # 当读取器未到达内容末尾时循环
        while not r.eof():
            # 读取并去除首尾空白的一行作为参数头部信息
            header = r.read().strip()
            # 如果参数头部包含 ' : ' 分隔符
            if ' : ' in header:
                # 使用 ' : ' 分割参数名和参数类型
                arg_name, arg_type = header.split(' : ')[:2]
            else:
                # 如果参数头部不包含 ' : '
                if single_element_is_type:
                    # 如果单个元素被视为类型，则参数名为空，参数类型为header内容
                    arg_name, arg_type = '', header
                else:
                    # 否则，参数名为header内容，参数类型为空字符串
                    arg_name, arg_type = header, ''

            # 读取并处理下一个非缩进行作为参数描述信息
            desc = r.read_to_next_unindented_line()
            # 去除缩进并处理描述信息
            desc = dedent_lines(desc)
            # 去除空白行并处理描述信息
            desc = strip_blank_lines(desc)

            # 将解析得到的参数名、参数类型和描述信息创建为Parameter对象并添加到参数列表中
            params.append(Parameter(arg_name, arg_type, desc))

        # 返回解析得到的参数对象列表
        return params

    # 下面的正则表达式定义用于匹配和解析函数名及描述信息
    #
    # <FUNCNAME>
    # <FUNCNAME> SPACE* COLON SPACE+ <DESC> SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)+ (COMMA | PERIOD)? SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)* SPACE* COLON SPACE+ <DESC> SPACE*
    #
    # <FUNCNAME> 是以下之一：
    #   <PLAIN_FUNCNAME>
    #   COLON <ROLE> COLON BACKTICK <PLAIN_FUNCNAME> BACKTICK
    # 其中：
    #   <PLAIN_FUNCNAME> 是合法的函数名，
    #   <ROLE> 是由字母数字字符组成的非空序列。
    # 示例：func_f1  :meth:`func_h1` :obj:`~baz.obj_r` :class:`class_j`
    # <DESC> 是描述函数的字符串。

    # 以下是定义的各个部分的正则表达式
    _role = r":(?P<role>\w+):"
    _funcbacktick = r"`(?P<name>(?:~\w+\.)?[a-zA-Z0-9_\.-]+)`"
    _funcplain = r"(?P<name2>[a-zA-Z0-9_\.-]+)"
    _funcname = r"(" + _role + _funcbacktick + r"|" + _funcplain + r")"
    _funcnamenext = _funcname.replace('role', 'rolenext')
    _funcnamenext = _funcnamenext.replace('name', 'namenext')
    _description = r"(?P<description>\s*:(\s+(?P<desc>\S+.*))?)?\s*$"

    # 用于匹配和解析整行函数名和描述信息的正则表达式
    _func_rgx = re.compile(r"^\s*" + _funcname + r"\s*")
    _line_rgx = re.compile(
        r"^\s*" +
        r"(?P<allfuncs>" +        # 所有函数名的组
        _funcname +
        r"(?P<morefuncs>([,]\s+" + _funcnamenext + r")*)" +
        r")" +                     # "allfuncs" 的结尾
        # 一些函数列表有一个尾随逗号（或句点）'\s*'
        r"(?P<trailing>[,\.])?" +
        _description)

    # 空的 <DESC> 元素被替换为 '..'
    empty_description = '..'
    def _parse_see_also(self, content):
        """
        解析“see also”部分的内容，返回一个列表，每个元素是一个包含函数及其描述的元组。

        content: 包含待解析文本行的列表

        """
        
        items = []  # 初始化空列表，用于存储解析后的条目
        
        def parse_item_name(text):
            """解析函数名及其角色，匹配':role:`name`'或'name'形式的文本。"""
            m = self._func_rgx.match(text)
            if not m:
                raise ParseError("%s is not a item name" % text)
            role = m.group('role')
            name = m.group('name') if role else m.group('name2')
            return name, role, m.end()
        
        rest = []  # 初始化空列表，用于存储未分配给任何函数的剩余行
        
        for line in content:
            if not line.strip():  # 跳过空行
                continue
            
            line_match = self._line_rgx.match(line)
            description = None
            
            if line_match:
                description = line_match.group('desc')
                if line_match.group('trailing') and description:
                    self._error_location(
                        'Unexpected comma or period after function list at '
                        'index %d of line "%s"' % (line_match.end('trailing'),
                                                   line),
                        error=False)
            if not description and line.startswith(' '):
                rest.append(line.strip())  # 将未分配给任何函数的行添加到rest列表中
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
                rest = list(filter(None, [description]))  # 将描述信息添加到rest列表中
                items.append((funcs, rest))  # 将函数列表及其描述添加到items列表中
            else:
                raise ParseError("%s is not a item name" % line)
        
        return items  # 返回解析后的条目列表

    def _parse_index(self, section, content):
        """
        解析索引部分的内容，返回一个包含索引项的字典。

        section: 包含索引类型和相关信息的字符串
        content: 包含索引项信息的列表

        """
        
        def strip_each_in(lst):
            """去除列表中每个元素的首尾空白。"""
            return [s.strip() for s in lst]

        out = {}  # 初始化空字典，用于存储解析后的索引项
        
        section = section.split('::')
        if len(section) > 1:
            out['default'] = strip_each_in(section[1].split(','))[0]  # 解析默认索引项
        for line in content:
            line = line.split(':')
            if len(line) > 2:
                out[line[1]] = strip_each_in(line[2].split(','))  # 解析其他索引项
        return out  # 返回解析后的索引项字典
    def _parse_summary(self):
        """解析文档摘要部分，提取签名（如果存在）和摘要"""
        if self._is_at_section():
            return

        # 如果存在多个签名，取最后一个
        while True:
            summary = self._doc.read_to_next_empty_line()  # 读取到下一个空行为止的内容作为摘要
            summary_str = " ".join([s.strip() for s in summary]).strip()  # 将摘要内容合并成字符串并去除首尾空格
            compiled = re.compile(r'^([\w., ]+=)?\s*[\w\.]+\(.*\)$')  # 编译正则表达式以匹配可能的函数签名格式
            if compiled.match(summary_str):  # 如果摘要符合函数签名的格式
                self['Signature'] = summary_str  # 将其作为签名存储
                if not self._is_at_section():  # 如果不在文档的节(section)部分，则继续查找
                    continue
            break

        if summary is not None:  # 如果摘要不为空
            self['Summary'] = summary  # 存储摘要内容

        if not self._is_at_section():  # 如果不在文档的节(section)部分
            self['Extended Summary'] = self._read_to_next_section()  # 读取到下一个节(section)之前的内容作为扩展摘要

    def _parse(self):
        self._doc.reset()  # 重置文档解析器的状态
        self._parse_summary()  # 解析摘要部分

        sections = list(self._read_sections())  # 获取文档中的所有节(section)
        section_names = {section for section, content in sections}  # 获取所有节(section)的名称集合

        has_returns = 'Returns' in section_names  # 判断是否包含返回值节(Returns)
        has_yields = 'Yields' in section_names  # 判断是否包含生成器产出节(Yields)
        # 还可以进行更多的测试，但我们并未进行。任意决定。
        if has_returns and has_yields:  # 如果同时包含Returns和Yields节
            msg = 'Docstring contains both a Returns and Yields section.'
            raise ValueError(msg)  # 抛出异常，不应同时包含Returns和Yields节
        if not has_yields and 'Receives' in section_names:  # 如果没有Yields节但包含Receives节
            msg = 'Docstring contains a Receives section but not Yields.'
            raise ValueError(msg)  # 抛出异常，应包含Yields节但不应包含Receives节

        for (section, content) in sections:
            if not section.startswith('..'):  # 如果节(section)不以'..'开头
                section = (s.capitalize() for s in section.split(' '))  # 将节(section)名称首字母大写
                section = ' '.join(section)  # 将首字母大写后的节(section)名称重新组合成字符串
                if self.get(section):  # 如果已经存在该节(section)
                    self._error_location("The section %s appears twice"
                                         % section)  # 报告错误，节(section)重复出现

            if section in ('Parameters', 'Other Parameters', 'Attributes',
                           'Methods'):
                self[section] = self._parse_param_list(content)  # 解析参数、属性或方法列表
            elif section in ('Returns', 'Yields', 'Raises', 'Warns',
                             'Receives'):
                self[section] = self._parse_param_list(
                    content, single_element_is_type=True)  # 解析返回值、生成器产出、异常、警告或接收到的参数列表
            elif section.startswith('.. index::'):
                self['index'] = self._parse_index(section, content)  # 解析索引部分
            elif section == 'See Also':
                self['See Also'] = self._parse_see_also(content)  # 解析参见部分
            else:
                self[section] = content  # 其他情况，直接存储节(section)内容

    def _error_location(self, msg, error=True):
        if hasattr(self, '_obj'):  # 如果存在属性'_obj'
            # 我们知道文档的来源：
            try:
                filename = inspect.getsourcefile(self._obj)  # 获取对象源代码所在文件名
            except TypeError:
                filename = None
            msg = msg + (f" in the docstring of {self._obj} in {filename}.")  # 添加文档字符串来源的信息
        if error:
            raise ValueError(msg)  # 如果有错误，则抛出值错误异常
        else:
            warn(msg, stacklevel=3)  # 否则，发出警告信息，告知调用栈层级

    # 字符串转换函数
    # 返回一个包含标题和横线的列表，用于文档中的节标题
    def _str_header(self, name, symbol='-'):
        return [name, len(name)*symbol]

    # 对给定的文档内容进行缩进处理，每行开头添加指定数目的空格
    def _str_indent(self, doc, indent=4):
        out = []
        for line in doc:
            out += [' '*indent + line]
        return out

    # 返回一个包含函数签名的列表，如果存在签名，则将其中的 '*' 替换为 '\*'
    def _str_signature(self):
        if self['Signature']:
            return [self['Signature'].replace('*', r'\*')] + ['']
        else:
            return ['']

    # 返回摘要内容的列表，如果存在摘要，则作为单独的行
    def _str_summary(self):
        if self['Summary']:
            return self['Summary'] + ['']
        else:
            return []

    # 返回扩展摘要内容的列表，如果存在扩展摘要，则作为单独的行
    def _str_extended_summary(self):
        if self['Extended Summary']:
            return self['Extended Summary'] + ['']
        else:
            return []

    # 返回参数列表的格式化输出，每个参数包含名称、类型和描述（如果存在的话）
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

    # 返回给定节名称的内容列表，如果该节存在的话
    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += self[name]
            out += ['']
        return out

    # 返回“参见”部分的格式化输出，包括相关函数的链接和描述（如果存在的话）
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
                    link = "`%s`_" % func
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

    # 返回索引部分的格式化输出，包括默认索引和各个章节的引用列表
    def _str_index(self):
        idx = self['index']
        out = []
        output_index = False
        default_index = idx.get('default', '')
        if default_index:
            output_index = True
        out += ['.. index:: %s' % default_index]
        for section, references in idx.items():
            if section == 'default':
                continue
            output_index = True
            out += ['   :{}: {}'.format(section, ', '.join(references))]
        if output_index:
            return out
        else:
            return ''
    # 定义一个特殊方法 __str__，用于返回对象的字符串表示形式
    def __str__(self, func_role=''):
        # 创建一个空列表，用于存储生成的字符串片段
        out = []
        # 调用对象的 _str_signature 方法，将其返回的字符串列表添加到 out 中
        out += self._str_signature()
        # 调用对象的 _str_summary 方法，将其返回的字符串列表添加到 out 中
        out += self._str_summary()
        # 调用对象的 _str_extended_summary 方法，将其返回的字符串列表添加到 out 中
        out += self._str_extended_summary()
        # 遍历给定的参数列表，依次调用 _str_param_list 方法，将返回的字符串列表添加到 out 中
        for param_list in ('Parameters', 'Returns', 'Yields', 'Receives',
                           'Other Parameters', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        # 调用对象的 _str_section 方法，将 'Warnings' 的相关内容字符串列表添加到 out 中
        out += self._str_section('Warnings')
        # 调用对象的 _str_see_also 方法，将指定角色相关的内容字符串列表添加到 out 中
        out += self._str_see_also(func_role)
        # 遍历给定的字符串列表，依次调用 _str_section 方法，将返回的字符串列表添加到 out 中
        for s in ('Notes', 'References', 'Examples'):
            out += self._str_section(s)
        # 遍历给定的参数列表，依次调用 _str_param_list 方法，将返回的字符串列表添加到 out 中
        for param_list in ('Attributes', 'Methods'):
            out += self._str_param_list(param_list)
        # 调用对象的 _str_index 方法，将其返回的字符串列表添加到 out 中
        out += self._str_index()
        # 将 out 中的所有字符串片段用换行符连接成一个完整的字符串并返回
        return '\n'.join(out)
# 定义一个函数，用于缩进字符串，可以指定缩进的空格数，默认为4个空格
def indent(str, indent=4):
    # 根据指定的缩进空格数生成缩进字符串
    indent_str = ' '*indent
    # 如果输入的字符串为空，则返回只有缩进的字符串
    if str is None:
        return indent_str
    # 将输入的字符串按行分割成列表
    lines = str.split('\n')
    # 对每一行字符串添加指定的缩进，并用换行符连接起来
    return '\n'.join(indent_str + l for l in lines)


# 定义一个函数，用于最大程度地去除一组字符串列表中的缩进
def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    # 将字符串列表连接成单个字符串，然后进行最大程度的去除缩进
    return textwrap.dedent("\n".join(lines)).split("\n")


# 定义一个函数，用于生成带标题的文本
def header(text, style='-'):
    # 返回标题文本和在下方用指定字符重复标题长度的字符串
    return text + '\n' + style*len(text) + '\n'


# 定义一个类，用于生成函数的文档字符串，继承自NumpyDocString类
class FunctionDoc(NumpyDocString):
    def __init__(self, func, role='func', doc=None, config={}):
        self._f = func
        self._role = role  # 函数角色，例如 "func" 或 "meth"

        # 如果未提供文档字符串，则尝试从函数对象中获取，如果函数对象为None，则抛出错误
        if doc is None:
            if func is None:
                raise ValueError("No function or docstring given")
            doc = inspect.getdoc(func) or ''
        # 调用父类的初始化方法，传入文档字符串和配置信息
        NumpyDocString.__init__(self, doc, config)

    # 获取函数对象的方法，返回函数对象和函数名称
    def get_func(self):
        func_name = getattr(self._f, '__name__', self.__class__.__name__)
        # 如果函数对象是类，则获取其调用方法或初始化方法作为函数对象
        if inspect.isclass(self._f):
            func = getattr(self._f, '__call__', self._f.__init__)
        else:
            func = self._f
        return func, func_name

    # 将函数文档对象转换为字符串表示形式
    def __str__(self):
        out = ''

        # 获取函数对象和函数名称
        func, func_name = self.get_func()

        # 定义函数角色与其对应的说明
        roles = {'func': 'function',
                 'meth': 'method'}

        # 如果指定了函数角色，但该角色不在预定义的角色列表中，则打印警告信息
        if self._role:
            if self._role not in roles:
                print("Warning: invalid role %s" % self._role)
            # 构建函数文档字符串的格式化输出
            out += '.. {}:: {}\n    \n\n'.format(roles.get(self._role, ''),
                                             func_name)

        # 调用父类的__str__()方法，传入函数角色作为参数，生成文档字符串的主体部分
        out += super().__str__(func_role=self._role)
        return out


# 定义一个类，用于生成类的文档字符串，继承自NumpyDocString类
class ClassDoc(NumpyDocString):

    # 额外公共方法列表，这些方法会包含在生成的文档字符串中
    extra_public_methods = ['__call__']
    # 初始化函数，接受多个参数：cls（类）、doc（文档字符串）、modulename（模块名称）、func_doc（函数文档类型）、config（配置字典）
    def __init__(self, cls, doc=None, modulename='', func_doc=FunctionDoc,
                 config={}):
        # 如果传入的cls不是类且不为None，则引发值错误异常
        if not inspect.isclass(cls) and cls is not None:
            raise ValueError("Expected a class or None, but got %r" % cls)
        # 将传入的类赋值给成员变量_cls
        self._cls = cls

        # 检查是否导入了sphinx模块，如果有则导入ALL常量，否则设为object()空对象
        if 'sphinx' in sys.modules:
            from sphinx.ext.autodoc import ALL
        else:
            ALL = object()

        # 根据配置字典获取是否显示继承成员的设置，默认为True
        self.show_inherited_members = config.get(
                    'show_inherited_class_members', True)

        # 如果modulename不为空且不以'.'结尾，则在其末尾添加'.'
        if modulename and not modulename.endswith('.'):
            modulename += '.'
        # 将处理后的modulename赋值给成员变量_mod
        self._mod = modulename

        # 如果未提供doc文档字符串，则尝试从cls获取文档字符串；若cls为None，则引发值错误异常
        if doc is None:
            if cls is None:
                raise ValueError("No class or documentation string given")
            doc = pydoc.getdoc(cls)

        # 使用NumpyDocString类的初始化方法初始化当前实例，传入doc作为参数
        NumpyDocString.__init__(self, doc)

        # 从配置字典中获取'members'字段，默认为空列表
        _members = config.get('members', [])
        # 如果_members等于ALL常量，则置为None
        if _members is ALL:
            _members = None
        # 从配置字典中获取'exclude-members'字段，默认为空列表
        _exclude = config.get('exclude-members', [])

        # 如果配置字典中的'show_class_members'字段为True且'exclude-members'不等于ALL常量
        if config.get('show_class_members', True) and _exclude is not ALL:
            # 定义一个内部函数splitlines_x，用于将字符串按行分割为列表
            def splitlines_x(s):
                if not s:
                    return []
                else:
                    return s.splitlines()
            # 遍历两个元组列表，每个元组包含字段名称和对应的类成员列表，如['Methods', self.methods]
            for field, items in [('Methods', self.methods),
                                 ('Attributes', self.properties)]:
                # 如果当前字段列表为空，则初始化doc_list列表
                if not self[field]:
                    doc_list = []
                    # 遍历类成员列表中的名称，按名称排序
                    for name in sorted(items):
                        # 如果名称在排除列表_exclude中或者_members列表中且不包含在其中
                        if (name in _exclude or
                                (_members and name not in _members)):
                            continue
                        try:
                            # 获取类成员对象的文档字符串
                            doc_item = pydoc.getdoc(getattr(self._cls, name))
                            # 将获取到的文档字符串作为参数创建Parameter对象，添加到doc_list列表中
                            doc_list.append(
                                Parameter(name, '', splitlines_x(doc_item)))
                        except AttributeError:
                            pass  # 方法不存在则跳过
                    # 将生成的doc_list列表赋值给当前字段
                    self[field] = doc_list

    # 方法装饰器，定义methods属性，返回当前类的所有方法名称列表
    @property
    def methods(self):
        # 如果_cls成员变量为None，则返回空列表
        if self._cls is None:
            return []
        # 使用inspect.getmembers获取_cls类的所有成员，筛选出符合条件的方法名称列表并返回
        return [name for name, func in inspect.getmembers(self._cls)
                if ((not name.startswith('_')
                     or name in self.extra_public_methods)
                    and isinstance(func, Callable)
                    and self._is_show_member(name))]

    # 方法装饰器，定义properties属性，返回当前类的所有属性名称列表
    @property
    def properties(self):
        # 如果_cls成员变量为None，则返回空列表
        if self._cls is None:
            return []
        # 使用inspect.getmembers获取_cls类的所有成员，筛选出符合条件的属性名称列表并返回
        return [name for name, func in inspect.getmembers(self._cls)
                if (not name.startswith('_') and
                    (func is None or isinstance(func, property) or
                     inspect.isdatadescriptor(func))
                    and self._is_show_member(name))]

    # 内部方法，用于判断是否显示成员名称
    def _is_show_member(self, name):
        # 如果show_inherited_members成员变量为True，则显示所有类成员
        if self.show_inherited_members:
            return True  # 显示所有类成员
        # 如果成员名称不在_cls类的__dict__中，则不显示继承的类成员
        if name not in self._cls.__dict__:
            return False  # 继承的类成员不显示
        return True
```