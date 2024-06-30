# `D:\src\scipysrc\scikit-learn\sklearn\externals\_arff.py`

```
# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2011 Renato de Pontes Pereira, renato.ppontes at gmail dot com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

'''
The liac-arff module implements functions to read and write ARFF files in
Python. It was created in the Connectionist Artificial Intelligence Laboratory
(LIAC), which takes place at the Federal University of Rio Grande do Sul
(UFRGS), in Brazil.

ARFF (Attribute-Relation File Format) is an file format specially created for
describe datasets which are commonly used for machine learning experiments and
software. This file format was created to be used in Weka, the best
representative software for machine learning automated experiments.

An ARFF file can be divided into two sections: header and data. The Header
describes the metadata of the dataset, including a general description of the
dataset, its name and its attributes. The source below is an example of a
header section in a XOR dataset::

    %
    % XOR Dataset
    %
    % Created by Renato Pereira
    %            rppereira@inf.ufrgs.br
    %            http://inf.ufrgs.br/~rppereira
    %
    %
    @RELATION XOR

    @ATTRIBUTE input1 REAL
    @ATTRIBUTE input2 REAL
    @ATTRIBUTE y REAL

The Data section of an ARFF file describes the observations of the dataset, in
the case of XOR dataset::

    @DATA
    0.0,0.0,0.0
    0.0,1.0,1.0
    1.0,0.0,1.0
    1.0,1.0,0.0
    %
    %
    %

Notice that several lines are starting with an ``%`` symbol, denoting a
comment, thus, lines with ``%`` at the beginning will be ignored, except by the
'''

# 本模块实现了读写 ARFF 文件的功能
# 创建于巴西联邦大学南大河州分校的连接主义人工智能实验室（LIAC）
import re

# 定义 ARFF 文件的关键字和符号
ATTRIBUTE = re.compile(r'^@ATTRIBUTE\s+([^\s]+)\s+(.*)$')
DATA = re.compile(r'^@DATA$')

def loads(input):
    '''
    从字符串中加载 ARFF 数据，返回包含 header 和 data 的元组。

    Parameters:
    input (str): 包含 ARFF 格式数据的字符串

    Returns:
    tuple: 包含 header（字符串）和 data（列表）的元组
    '''
    # 初始化 header 和 data 变量
    header = ''
    data = []

    # 逐行处理输入数据
    for line in input.splitlines():
        if line.startswith('%'):
            continue  # 跳过以 '%' 开头的注释行

        # 匹配 ATTRIBUTE 定义
        match = ATTRIBUTE.match(line)
        if match:
            header += line + '\n'
            continue

        # 匹配 DATA 定义
        match = DATA.match(line)
        if match:
            break

    # 返回 header 和 data 的元组
    return header.strip(), data
'''
This module provides utilities for working with ARFF (Attribute-Relation File Format) files.

'''
# 作者信息和版本号
__author__ = 'Renato de Pontes Pereira, Matthias Feurer, Joel Nothman'
__author_email__ = ('renato.ppontes@gmail.com, '
                    'feurerm@informatik.uni-freiburg.de, '
                    'joel.nothman@gmail.com')
__version__ = '2.4.0'

# 导入必要的库
import re  # 导入正则表达式模块
import csv  # 导入 CSV 文件读取模块
from typing import TYPE_CHECKING  # 导入类型检查相关的模块
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple  # 导入类型提示相关的模块
# CONSTANTS ===================================================================

# 定义简单类型列表，包括数值、实数、整数和字符串
_SIMPLE_TYPES = ['NUMERIC', 'REAL', 'INTEGER', 'STRING']

# 定义 ARFF 文件中的关键字符常量
_TK_DESCRIPTION = '%'
_TK_COMMENT     = '%'
_TK_RELATION    = '@RELATION'
_TK_ATTRIBUTE   = '@ATTRIBUTE'
_TK_DATA        = '@DATA'

# 正则表达式模式，用于匹配 ARFF 文件中的不同行类型
_RE_RELATION     = re.compile(r'^([^\{\}%,\s]*|\".*\"|\'.*\')$', re.UNICODE)
_RE_ATTRIBUTE    = re.compile(r'^(\".*\"|\'.*\'|[^\{\}%,\s]*)\s+(.+)$', re.UNICODE)
_RE_QUOTE_CHARS = re.compile(r'["\'\\\s%,\000-\031]', re.UNICODE)
_RE_ESCAPE_CHARS = re.compile(r'(?=["\'\\%])|[\n\r\t\000-\031]')
_RE_SPARSE_LINE = re.compile(r'^\s*\{.*\}\s*$', re.UNICODE)
_RE_NONTRIVIAL_DATA = re.compile('["\'{}\\s]', re.UNICODE)

# 定义 ARFF 文件中的数据类型别名
ArffDenseDataType = Iterator[List]
ArffSparseDataType = Tuple[List, ...]

# 根据类型检查设置，定义 ARFF 数据容器的类型
if TYPE_CHECKING:
    # 导入 typing_extensions 模块中的 TypedDict 类型
    from typing_extensions import TypedDict

    # 定义 ArffContainerType 类型，描述 ARFF 文件的结构
    class ArffContainerType(TypedDict):
        description: str
        relation: str
        attributes: List
        data: Union[ArffDenseDataType, ArffSparseDataType]

else:
    # 当没有类型检查时，使用通用的字典类型定义 ArffContainerType
    ArffContainerType = Dict[str, Any]

# 函数定义：构建用于解析 ARFF 文件值的正则表达式模式
def _build_re_values():
    # 定义匹配双引号包围值的正则表达式模式
    quoted_re = r'''
                    "      # 开始双引号
                    (?:
                        (?<!\\)    # 不匹配额外的反斜杠
                        (?:\\\\)*  # 可能有转义的反斜杠
                        \\"        # 转义的双引号
                    |
                        \\[^"]     # 转义非双引号字符
                    |
                        [^"\\]     # 非双引号字符
                    )*
                    "      # 结束双引号
                    '''
    # 定义值可能的正则表达式模式，可以被双引号或单引号包围或者不需要引号
    value_re = r'''(?:
        %s|          # 可以被双引号包围
        %s|          # 或者被单引号包围
        [^,\s"'{}]+  # 或者不包含需要引号的字符
        )''' % (quoted_re,
                quoted_re.replace('"', "'"))

    # 定义用于稠密格式数据行的正则表达式模式
    dense = re.compile(r'''(?x)
        ,                # 可能跟在', '后面
        \s*
        ((?=,)|$|{value_re})  # 空值或者值
        |
        (\S.*)           # 错误
        '''.format(value_re=value_re))

    # 定义用于稀疏格式数据行的正则表达式模式
    sparse = re.compile(r'''(?x)
        (?:^\s*\{|,)   # 可能跟在', '或者'{'后面
        \s*
        (\d+)          # 属性键
        \s+
        (%(value_re)s) # 值
        |
        (?!}\s*$)      # 不是错误，如果是 }$
        (?!^\s*{\s*}\s*$)  # 不是错误，如果是 ^{}$
        \S.*           # 错误
        ''' % {'value_re': value_re})

    return dense, sparse
# 定义常量_RE_DENSE_VALUES和_RE_SPARSE_KEY_VALUES，并调用函数_build_re_values()进行初始化
_RE_DENSE_VALUES, _RE_SPARSE_KEY_VALUES = _build_re_values()

# 定义用于转义的字典_ESCAPE_SUB_MAP，包含常见的转义序列及其对应的转义字符
_ESCAPE_SUB_MAP = {
    '\\\\': '\\',
    '\\"': '"',
    "\\'": "'",
    '\\t': '\t',
    '\\n': '\n',
    '\\r': '\r',
    '\\b': '\b',
    '\\f': '\f',
    '\\%': '%',
}

# 定义用于反转义的字典_UNESCAPE_SUB_MAP，将ASCII字符转换为八进制转义序列
_UNESCAPE_SUB_MAP = {chr(i): '\\%03o' % i for i in range(32)}
# 更新_UNESCAPE_SUB_MAP，将_ESCAPE_SUB_MAP的键值对进行反转
_UNESCAPE_SUB_MAP.update({v: k for k, v in _ESCAPE_SUB_MAP.items()})
# 将空字符串映射为反斜杠
_UNESCAPE_SUB_MAP[''] = '\\'
# 更新_ESCAPE_SUB_MAP，将'\0'到'\9'的转义序列映射为相应的ASCII字符
_ESCAPE_SUB_MAP.update({'\\%d' % i: chr(i) for i in range(10)})


def _escape_sub_callback(match):
    # 回调函数，用于替换正则表达式匹配的转义序列
    s = match.group()
    if len(s) == 2:
        try:
            return _ESCAPE_SUB_MAP[s]
        except KeyError:
            raise ValueError('Unsupported escape sequence: %s' % s)
    if s[1] == 'u':
        # 处理Unicode转义序列，将其转换为对应的字符
        return chr(int(s[2:], 16))
    else:
        # 处理八进制转义序列，将其转换为对应的字符
        return chr(int(s[1:], 8))


def _unquote(v):
    # 去除字符串v的引号，并将转义序列替换为实际字符
    if v[:1] in ('"', "'"):
        return re.sub(r'\\([0-9]{1,3}|u[0-9a-f]{4}|.)', _escape_sub_callback,
                      v[1:-1])
    elif v in ('?', ''):
        # 返回None或者空字符串
        return None
    else:
        return v


def _parse_values(s):
    '''(INTERNAL) Split a line into a list of values'''
    if not _RE_NONTRIVIAL_DATA.search(s):
        # 如果不包含非平凡数据，则快速处理为None或者空字符串
        return [None if s in ('?', '') else s
                for s in next(csv.reader([s]))]

    # _RE_DENSE_VALUES 根据正则表达式解析s中的值，忽略引号、空格等
    values, errors = zip(*_RE_DENSE_VALUES.findall(',' + s))
    if not any(errors):
        # 如果没有错误，则返回解析后的值列表，去除引号并替换转义序列
        return [_unquote(v) for v in values]
    if _RE_SPARSE_LINE.match(s):
        try:
            # 如果是稀疏数据，则按键值对的方式解析
            return {int(k): _unquote(v)
                    for k, v in _RE_SPARSE_KEY_VALUES.findall(s)}
        except ValueError:
            # 解析稀疏数据时出现ARFF语法错误
            for match in _RE_SPARSE_KEY_VALUES.finditer(s):
                if not match.group(1):
                    raise BadLayout('Error parsing %r' % match.group())
            raise BadLayout('Unknown parsing error')
    else:
        # 解析稠密数据时出现ARFF语法错误
        for match in _RE_DENSE_VALUES.finditer(s):
            if match.group(2):
                raise BadLayout('Error parsing %r' % match.group())
        raise BadLayout('Unknown parsing error')


# 常量定义部分
DENSE = 0     # 表示密集矩阵的常量值
COO = 1       # 表示坐标格式的稀疏矩阵的常量值
LOD = 2       # 表示字典列表格式的稀疏矩阵的常量值
DENSE_GEN = 3 # 字典生成器的常量值
LOD_GEN = 4   # 字典生成器的常量值
_SUPPORTED_DATA_STRUCTURES = [DENSE, COO, LOD, DENSE_GEN, LOD_GEN]


# 异常定义部分 ================================================================
class ArffException(Exception):
    message: Optional[str] = None

    def __init__(self):
        self.line = -1

    def __str__(self):
        return self.message % self.line


class BadRelationFormat(ArffException):
    # 定义一个错误消息，用于指示关系声明格式无效的情况，其中%d将被具体的行数替换。
    message = 'Bad @RELATION format, at line %d.'
class BadAttributeFormat(ArffException):
    '''当某些属性声明格式无效时引发的错误。'''
    message = 'Bad @ATTRIBUTE format, at line %d.'

class BadDataFormat(ArffException):
    '''当某些数据实例格式无效时引发的错误。'''
    def __init__(self, value):
        super().__init__()
        self.message = (
            'Bad @DATA instance format in line %d: ' +
            ('%s' % value)
        )

class BadAttributeType(ArffException):
    '''当在属性声明中提供了无效类型时引发的错误。'''
    message = 'Bad @ATTRIBUTE type, at line %d.'

class BadAttributeName(ArffException):
    '''当属性名在属性声明中重复时引发的错误。'''
    def __init__(self, value, value2):
        super().__init__()
        self.message = (
            ('Bad @ATTRIBUTE name %s at line' % value) +
            ' %d, this name is already in use in line' +
            (' %d.' % value2)
        )

class BadNominalValue(ArffException):
    '''当某个数据实例中使用了未在其相应属性声明中声明的值时引发的错误。'''
    def __init__(self, value):
        super().__init__()
        self.message = (
            ('Data value %s not found in nominal declaration, ' % value)
            + 'at line %d.'
        )

class BadNominalFormatting(ArffException):
    '''当未正确引用具有空格的名义值时引发的错误。'''
    def __init__(self, value):
        super().__init__()
        self.message = (
            ('Nominal data value "%s" not properly quoted in line ' % value) +
            '%d.'
        )

class BadNumericalValue(ArffException):
    '''当在某个数据实例中使用了无效的数值时引发的错误。'''
    message = 'Invalid numerical value, at line %d.'

class BadStringValue(ArffException):
    '''当字符串包含空格但未引用时引发的错误。'''
    message = 'Invalid string value at line %d.'

class BadLayout(ArffException):
    '''当ARFF文件的布局有误时引发的错误。'''
    message = 'Invalid layout of the ARFF file, at line %d.'

    def __init__(self, msg=''):
        super().__init__()
        if msg:
            self.message = BadLayout.message + ' ' + msg.replace('%', '%%')


class BadObject(ArffException):
    '''当表示ARFF文件的对象有误时引发的错误。'''
    def __init__(self, msg='Invalid object.'):
        self.msg = msg

    def __str__(self):
        return '%s' % self.msg

# =============================================================================

# INTERNAL ====================================================================
def _unescape_sub_callback(match):
    '''在字符串中查找匹配的子串并替换为预定义的字符。'''
    return _UNESCAPE_SUB_MAP[match.group()]

def encode_string(s):
    '''对字符串进行编码，确保引用了含有引号的字符。'''
    if _RE_QUOTE_CHARS.search(s):
        return "'%s'" % _RE_ESCAPE_CHARS.sub(_unescape_sub_callback, s)
    return s
class EncodedNominalConversor:
    # 编码名义变换器类，根据给定的值列表创建值到索引的映射字典
    def __init__(self, values):
        # 使用字典推导式创建值到索引的映射字典
        self.values = {v: i for i, v in enumerate(values)}
        # 强制将值 0 映射到索引 0
        self.values[0] = 0

    # 定义实例可调用方法，用于将值转换为对应的索引
    def __call__(self, value):
        try:
            # 尝试返回给定值对应的索引
            return self.values[value]
        except KeyError:
            # 如果值不在映射字典中，则抛出异常 BadNominalValue
            raise BadNominalValue(value)


class NominalConversor:
    # 名义变换器类，根据给定的值列表创建值集合和零值
    def __init__(self, values):
        # 使用集合存储给定的值列表
        self.values = set(values)
        # 将第一个值作为零值
        self.zero_value = values[0]

    # 定义实例可调用方法，用于将值转换为字符串表示
    def __call__(self, value):
        # 如果值不在值集合中
        if value not in self.values:
            # 如果值为 0，则返回预设的零值
            if value == 0:
                # 稀疏解码
                # 见问题 #52：在稀疏矩阵中，当未指定名义值时应使用它们的第一个值。这与EncodedNominalConversor的行为一致。
                return self.zero_value
            # 否则，抛出异常 BadNominalValue
            raise BadNominalValue(value)
        # 将值转换为字符串并返回
        return str(value)


class DenseGeneratorData:
    '''Internal helper class to allow for different matrix types without
    making the code a huge collection of if statements.'''

    # 解码行的内部辅助类方法，接受数据流和名义转换器列表作为参数
    def decode_rows(self, stream, conversors):
        # 对数据流中的每一行进行迭代处理
        for row in stream:
            # 解析行中的值
            values = _parse_values(row)

            # 如果解析后的值是字典类型
            if isinstance(values, dict):
                # 如果字典不为空且最大键大于等于名义转换器列表的长度，则抛出异常 BadDataFormat
                if values and max(values) >= len(conversors):
                    raise BadDataFormat(row)
                # 对于每个名义转换器，使用字典中的值或者默认为 0 的值
                values = [values[i] if i in values else 0 for i in range(len(conversors))]
            else:
                # 如果解析后的值不是字典，则要求其长度与名义转换器列表长度相等，否则抛出异常 BadDataFormat
                if len(values) != len(conversors):
                    raise BadDataFormat(row)

            # 调用内部方法，使用名义转换器列表解码处理后的值，并返回生成器对象
            yield self._decode_values(values, conversors)

    # 静态方法，用于解码给定的值列表
    @staticmethod
    def _decode_values(values, conversors):
        try:
            # 尝试对每个值使用对应的名义转换器进行转换
            values = [None if value is None else conversor(value)
                      for conversor, value
                      in zip(conversors, values)]
        except ValueError as exc:
            # 如果值转换时发生浮点数异常，抛出 BadNumericalValue 异常
            if 'float: ' in str(exc):
                raise BadNumericalValue()
        # 返回处理后的值列表
        return values
    # 定义一个方法，用于编码数据行（内部使用）

    # 当前数据行计数器，初始化为0
    current_row = 0

    # 遍历数据列表中的每个实例
    for inst in data:
        # 检查每个实例的属性数量是否与给定属性列表的数量相匹配
        if len(inst) != len(attributes):
            # 如果不匹配，抛出异常，指示实例的属性数量与期望的不同
            raise BadObject(
                'Instance %d has %d attributes, expected %d' %
                 (current_row, len(inst), len(attributes))
            )

        # 初始化一个新的数据列表，用于存储编码后的数据
        new_data = []
        # 遍历实例中的每个值
        for value in inst:
            # 如果值是None、空字符串或者不等于自身（NaN），将其编码为'?'
            if value is None or value == '' or value != value:
                s = '?'
            else:
                # 否则，将值转换为字符串并进行编码
                s = encode_string(str(value))
            # 将编码后的字符串添加到新数据列表中
            new_data.append(s)

        # 增加当前数据行计数器
        current_row += 1
        # 使用逗号将新数据列表中的各个元素连接为一个字符串，并返回生成器对象
        yield ','.join(new_data)
class _DataListMixin:
    """Mixin to return a list from decode_rows instead of a generator"""
    # 定义一个 Mixin 类，用于在 decode_rows 方法中返回列表而不是生成器

    def decode_rows(self, stream, conversors):
        # 解码行数据的方法
        data, rows, cols = [], [], []
        for i, row in enumerate(stream):
            # 遍历输入的流数据，每个元素包含一个行数据
            values = _parse_values(row)
            # 解析行数据，返回一个包含键值对的字典
            if not isinstance(values, dict):
                raise BadLayout()
                # 如果解析结果不是字典，则抛出 BadLayout 异常
            if not values:
                continue
                # 如果值为空，则继续下一次循环

            row_cols, values = zip(*sorted(values.items()))
            # 将字典中的键值对按键排序后分别存储到两个元组中
            try:
                values = [value if value is None else conversors[key](value)
                          for key, value in zip(row_cols, values)]
                # 尝试将值按照对应的转换器转换为指定类型
            except ValueError as exc:
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                    # 如果值转换出现 ValueError 异常，并且异常信息包含 'float: '，则抛出 BadNumericalValue 异常
                raise
                # 否则抛出原始异常

            except IndexError:
                # conversor out of range
                raise BadDataFormat(row)
                # 如果转换器超出范围，则抛出 BadDataFormat 异常

            data.extend(values)
            # 将转换后的值列表添加到 data 列表末尾
            rows.extend([i] * len(values))
            # 将当前行号 i 重复 len(values) 次，并将结果列表添加到 rows 列表末尾
            cols.extend(row_cols)
            # 将排序后的键列表添加到 cols 列表末尾

        return data, rows, cols
        # 返回三个列表：data（数据值列表）、rows（行索引列表）、cols（列索引列表）

    def encode_data(self, data, attributes):
        # 编码数据的方法，将数据转换为指定格式字符串
        num_attributes = len(attributes)
        # 计算属性的数量
        new_data = []
        # 新数据列表
        current_row = 0
        # 当前行号

        row = data.row
        # 数据的行索引
        col = data.col
        # 数据的列索引
        data = data.data
        # 数据的值

        # Check if the rows are sorted
        if not all(row[i] <= row[i + 1] for i in range(len(row) - 1)):
            raise ValueError("liac-arff can only output COO matrices with "
                             "sorted rows.")
            # 检查行索引是否已排序，如果未排序则抛出 ValueError 异常

        for v, col, row in zip(data, col, row):
            # 遍历数据、列索引和行索引的并行列表
            if row > current_row:
                # 如果当前行号小于数据行号
                while current_row < row:
                    yield " ".join(["{", ','.join(new_data), "}"])
                    # 使用新数据列表生成格式化的字符串，并作为生成器的一部分返回
                    new_data = []
                    # 重置新数据列表
                    current_row += 1
                    # 增加当前行号

            if col >= num_attributes:
                raise BadObject(
                    'Instance %d has at least %d attributes, expected %d' %
                    (current_row, col + 1, num_attributes)
                )
                # 如果列索引超出属性数量，则抛出 BadObject 异常

            if v is None or v == '' or v != v:
                s = '?'
                # 如果值为 None、空字符串或不等于自身，则设为 '?'
            else:
                s = encode_string(str(v))
                # 否则将值转换为字符串并进行编码
            new_data.append("%d %s" % (col, s))
            # 将格式化后的列索引和编码后的字符串添加到新数据列表末尾

        yield " ".join(["{", ','.join(new_data), "}"])
        # 返回最后一行生成的格式化字符串

class COOData:
    # COO 数据类，包含解码行和编码数据的方法
    def decode_rows(self, stream, conversors):
        # 解码行数据的方法
        data, rows, cols = [], [], []
        for i, row in enumerate(stream):
            # 遍历输入的流数据，每个元素包含一个行数据
            values = _parse_values(row)
            # 解析行数据，返回一个包含键值对的字典
            if not isinstance(values, dict):
                raise BadLayout()
                # 如果解析结果不是字典，则抛出 BadLayout 异常
            if not values:
                continue
                # 如果值为空，则继续下一次循环

            row_cols, values = zip(*sorted(values.items()))
            # 将字典中的键值对按键排序后分别存储到两个元组中
            try:
                values = [value if value is None else conversors[key](value)
                          for key, value in zip(row_cols, values)]
                # 尝试将值按照对应的转换器转换为指定类型
            except ValueError as exc:
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                    # 如果值转换出现 ValueError 异常，并且异常信息包含 'float: '，则抛出 BadNumericalValue 异常
                raise
                # 否则抛出原始异常

            except IndexError:
                # conversor out of range
                raise BadDataFormat(row)
                # 如果转换器超出范围，则抛出 BadDataFormat 异常

            data.extend(values)
            # 将转换后的值列表添加到 data 列表末尾
            rows.extend([i] * len(values))
            # 将当前行号 i 重复 len(values) 次，并将结果列表添加到 rows 列表末尾
            cols.extend(row_cols)
            # 将排序后的键列表添加到 cols 列表末尾

        return data, rows, cols
        # 返回三个列表：data（数据值列表）、rows（行索引列表）、cols（列索引列表）

    def encode_data(self, data, attributes):
        # 编码数据的方法，将数据转换为指定格式字符串
        num_attributes = len(attributes)
        # 计算属性的数量
        new_data = []
        # 新数据列表
        current_row = 0
        # 当前行号

        row = data.row
        # 数据的行索引
        col = data.col
        # 数据的列索引
        data = data.data
        # 数据的值

        # Check if the rows are sorted
        if not all(row[i] <= row[i + 1] for i in range(len(row) - 1)):
            raise ValueError("liac-arff can only output COO matrices with "
                             "sorted rows.")
            # 检查行索引是否已排序，如果未排序则抛出 ValueError 异常

        for v, col, row in zip(data, col, row):
            # 遍历数据、列索引和行索引的并行列表
            if row > current_row:
                # 如果当前行号小于数据行号
                while current_row < row:
                    yield " ".join(["{", ','.join(new_data), "}"])
                    # 使用新数据列表生成格式化的字符串，并作为生成器的一部分返回
                    new_data = []
                    # 重置新数据列表
                    current_row += 1
                    # 增加当前行号

            if col >= num_attributes:
                raise BadObject(
                    'Instance %d has at least %d attributes, expected %d' %
                    (current_row, col + 1, num_attributes)
                )
                # 如果列索引超出属性数量，则抛出 BadObject 异常

            if v is None or v == '' or v != v:
                s = '?'
                # 如果值为 None、空字符串或不等于自身，则设为 '?'
            else:
                s = encode_string(str(v))
                # 否则将值转换为字符串并进行编码
            new_data.append("%d %s" % (col, s))
            # 将格式化后的列索引和编码后的字符串添加到新数据列表
    # 解析数据流中的每一行数据，应用给定的转换器进行转换
    def decode_rows(self, stream, conversors):
        # 遍历数据流中的每一行
        for row in stream:
            # 解析行数据，获取键值对
            values = _parse_values(row)

            # 如果解析结果不是字典，则抛出异常 BadLayout
            if not isinstance(values, dict):
                raise BadLayout()

            try:
                # 尝试对每个键值对应用相应的转换器，生成新的字典
                yield {key: None if value is None else conversors[key](value)
                       for key, value in values.items()}
            except ValueError as exc:
                # 如果值转换失败，检查错误信息中是否包含 'float: '，如果是则抛出 BadNumericalValue 异常
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                raise  # 如果不是 'float: '，则继续抛出原始异常
            except IndexError:
                # 如果转换器超出范围，则抛出 BadDataFormat 异常
                raise BadDataFormat(row)

    # 编码数据为特定格式的字符串表示
    def encode_data(self, data, attributes):
        # 当前行号初始化为 0
        current_row = 0

        # 计算属性数量
        num_attributes = len(attributes)

        # 遍历数据中的每一行
        for row in data:
            # 初始化新数据列表
            new_data = []

            # 检查行长度和最大列数是否符合预期
            if len(row) > 0 and max(row) >= num_attributes:
                raise BadObject(
                    'Instance %d has %d attributes, expected %d' %
                    (current_row, max(row) + 1, num_attributes)
                )

            # 对行中的每一列进行排序后处理
            for col in sorted(row):
                v = row[col]
                # 根据值类型进行编码处理
                if v is None or v == '' or v != v:
                    s = '?'
                else:
                    s = encode_string(str(v))
                new_data.append("%d %s" % (col, s))

            # 更新当前行号
            current_row += 1

            # 生成格式化的输出字符串
            yield " ".join(["{", ','.join(new_data), "}"])
class LODData(_DataListMixin, LODGeneratorData):
    pass


def _get_data_object_for_decoding(matrix_type):
    if matrix_type == DENSE:
        return Data()
    elif matrix_type == COO:
        return COOData()
    elif matrix_type == LOD:
        return LODData()
    elif matrix_type == DENSE_GEN:
        return DenseGeneratorData()
    elif matrix_type == LOD_GEN:
        return LODGeneratorData()
    else:
        raise ValueError("Matrix type %s not supported." % str(matrix_type))


def _get_data_object_for_encoding(matrix):
    # Probably a scipy.sparse
    if hasattr(matrix, 'format'):
        if matrix.format == 'coo':
            return COOData()
        else:
            raise ValueError('Cannot guess matrix format!')
    elif isinstance(matrix[0], dict):
        return LODData()
    else:
        return Data()


# =============================================================================

# ADVANCED INTERFACE ==========================================================
class ArffDecoder:
    '''An ARFF decoder.'''

    def __init__(self):
        '''Constructor.'''
        self._conversors = []
        self._current_line = 0

    def _decode_comment(self, s):
        '''(INTERNAL) Decodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r\n" characters.

        :param s: a normalized string.
        :return: a string with the decoded comment.
        '''
        res = re.sub(r'^\%( )?', '', s)
        return res

    def _decode_relation(self, s):
        '''(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string. The string must
        start with alphabetic character and must be quoted if the name includes
        spaces, otherwise this method will raise a `BadRelationFormat` exception.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r\n" characters.

        :param s: a normalized string.
        :return: a string with the decoded relation name.
        '''
        _, v = s.split(' ', 1)
        v = v.strip()

        if not _RE_RELATION.match(v):
            raise BadRelationFormat()

        res = str(v.strip('"\''))
        return res
    # (INTERNAL) 解码一个属性行。
    #
    # 在 arff 文件中，属性是最复杂的声明。所有属性必须遵循以下模板：
    #
    #     @attribute <attribute-name> <datatype>
    #
    # 其中 `attribute-name` 是一个字符串，如果名称包含空白，则需要加引号，而 `datatype` 可以是：
    #
    # - 数值属性，如 `NUMERIC`、`INTEGER` 或 `REAL`。
    # - 字符串，使用 `STRING` 表示。
    # - 日期（未实现）。
    # - 标称属性的格式为：
    #
    #     {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}
    #
    # 标称名称遵循属性名称的规则，即如果名称包含空格，则必须加引号。
    #
    # 此方法接收一个规范化的字符串，即一个没有填充字符的字符串，包括 "\r\n" 字符。
    #
    # :param s: 一个规范化的字符串。
    # :return: 一个元组 (ATTRIBUTE_NAME, TYPE_OR_VALUES)。
    def _decode_attribute(self, s):
        _, v = s.split(' ', 1)  # 将输入字符串按第一个空格分割，获取属性值部分
        v = v.strip()  # 去除属性值部分的前后空白

        # 验证声明的一般结构
        m = _RE_ATTRIBUTE.match(v)
        if not m:
            raise BadAttributeFormat()

        # 提取原始名称和类型
        name, type_ = m.groups()

        # 提取最终名称
        name = str(name.strip('"\''))

        # 提取最终类型
        if type_[:1] == "{" and type_[-1:] == "}":  # 如果是标称属性
            try:
                type_ = _parse_values(type_.strip('{} '))  # 解析标称值
            except Exception:
                raise BadAttributeType()
            if isinstance(type_, dict):  # 如果解析结果是字典，抛出类型错误
                raise BadAttributeType()

        else:  # 如果不是标称属性，验证类型名称
            type_ = str(type_).upper()
            if type_ not in ['NUMERIC', 'REAL', 'INTEGER', 'STRING']:  # 验证类型是否合法
                raise BadAttributeType()

        return (name, type_)

    # 返回给定 ARFF 文件的 Python 表示形式。
    #
    # 当传递文件对象作为参数时，此方法逐行迭代地读取文件，避免将不必要的信息加载到内存中。
    #
    # :param s: 包含 ARFF 文件的字符串或文件对象。
    # :param encode_nominal: 布尔值，如果为 True，则在读取 .arff 文件时执行标签编码。
    # :param return_type: 确定用于存储数据集的数据结构。可以是 `arff.DENSE`、`arff.COO`、
    #                     `arff.LOD`、`arff.DENSE_GEN` 或 `arff.LOD_GEN` 中的一个。
    #                     参考 `working with sparse data`_ 和 `loading progressively`_ 章节。
    def decode(self, s, encode_nominal=False, return_type=DENSE):
        try:
            return self._decode(s, encode_nominal=encode_nominal,
                                matrix_type=return_type)  # 调用内部方法进行解码
        except ArffException as e:
            e.line = self._current_line
            raise e
    # 定义一个 ARFF 编码器类，用于编码 ARFF 文件
    '''An ARFF encoder.'''

    def _encode_comment(self, s=''):
        '''(INTERNAL) Encodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        If ``s`` is None, this method will simply return an empty comment.

        :param s: (OPTIONAL) string.
        :return: a string with the encoded comment line.
        '''
        # 如果传入了非空字符串 s，返回以 _TK_COMMENT 开头的格式化字符串
        if s:
            return '%s %s'%(_TK_COMMENT, s)
        else:
            # 否则，只返回 _TK_COMMENT
            return '%s' % _TK_COMMENT

    def _encode_relation(self, name):
        '''(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string.

        :param name: a string.
        :return: a string with the encoded relation declaration.
        '''
        # 检查 name 中是否包含特定字符，如果包含则将 name 用双引号括起来
        for char in ' %{},':
            if char in name:
                name = '"%s"'%name
                break

        # 返回以 _TK_RELATION 开头的格式化字符串，表示关系声明
        return '%s %s'%(_TK_RELATION, name)

    def _encode_attribute(self, name, type_):
        '''(INTERNAL) Encodes an attribute line.

        The attribute follow the template::

             @attribute <attribute-name> <datatype>

        where ``attribute-name`` is a string, and ``datatype`` can be:

        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.
        - Strings as ``STRING``.
        - Dates (NOT IMPLEMENTED).
        - Nominal attributes with format:

            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}

        This method must receive a the name of the attribute and its type, if
        the attribute type is nominal, ``type`` must be a list of values.

        :param name: a string.
        :param type_: a string or a list of string.
        :return: a string with the encoded attribute declaration.
        '''
        # 检查 name 中是否包含特定字符，如果包含则将 name 用双引号括起来
        for char in ' %{},':
            if char in name:
                name = '"%s"'%name
                break

        # 如果 type_ 是元组或列表，则将其中每个元素用 encode_string 函数处理后拼接
        if isinstance(type_, (tuple, list)):
            type_tmp = ['%s' % encode_string(type_k) for type_k in type_]
            type_ = '{%s}'%(', '.join(type_tmp))

        # 返回以 _TK_ATTRIBUTE 开头的格式化字符串，表示属性声明
        return '%s %s %s'%(_TK_ATTRIBUTE, name, type_)

    def encode(self, obj):
        '''Encodes a given object to an ARFF file.

        :param obj: the object containing the ARFF information.
        :return: the ARFF file as an string.
        '''
        # 通过迭代编码 obj 中的每行数据，生成一个列表
        data = [row for row in self.iter_encode(obj)]

        # 将列表中的每行数据用换行符连接成一个字符串，表示整个 ARFF 文件
        return '\n'.join(data)
    def iter_encode(self, obj):
        '''The iterative version of `arff.ArffEncoder.encode`.

        This encodes iteratively a given object and return, one-by-one, the
        lines of the ARFF file.

        :param obj: the object containing the ARFF information.
        :return: (yields) the ARFF file as strings.
        '''
        # 处理描述信息
        if obj.get('description', None):
            # 将描述信息按行分割，并逐行生成编码后的注释
            for row in obj['description'].split('\n'):
                yield self._encode_comment(row)

        # 处理关系（Relation）
        if not obj.get('relation'):
            # 如果关系名未找到或值无效，抛出异常
            raise BadObject('Relation name not found or with invalid value.')

        # 生成编码后的关系行
        yield self._encode_relation(obj['relation'])
        yield ''

        # 处理属性（Attributes）
        if not obj.get('attributes'):
            # 如果未找到属性信息，抛出异常
            raise BadObject('Attributes not found.')

        attribute_names = set()
        for attr in obj['attributes']:
            # 验证属性声明格式是否正确
            if not isinstance(attr, (tuple, list)) or \
               len(attr) != 2 or \
               not isinstance(attr[0], str):
                raise BadObject('Invalid attribute declaration "%s"'%str(attr))

            if isinstance(attr[1], str):
                # 验证属性类型是否有效
                if attr[1] not in _SIMPLE_TYPES:
                    raise BadObject('Invalid attribute type "%s"'%str(attr))

            # 验证属性类型格式是否正确
            elif not isinstance(attr[1], (tuple, list)):
                raise BadObject('Invalid attribute type "%s"'%str(attr))

            # 验证属性名是否重复使用
            if attr[0] in attribute_names:
                raise BadObject('Trying to use attribute name "%s" for the '
                                'second time.' % str(attr[0]))
            else:
                attribute_names.add(attr[0])

            # 生成编码后的属性行
            yield self._encode_attribute(attr[0], attr[1])
        yield ''
        attributes = obj['attributes']

        # 处理数据（Data）
        yield _TK_DATA
        if 'data' in obj:
            # 获取用于编码数据的数据对象，并生成数据行
            data = _get_data_object_for_encoding(obj.get('data'))
            yield from data.encode_data(obj.get('data'), attributes)

        yield ''
# =============================================================================

# BASIC INTERFACE =============================================================
# 加载 ARFF 文档的文件对象，并转换为 Python 对象
def load(fp, encode_nominal=False, return_type=DENSE):
    '''Load a file-like object containing the ARFF document and convert it into
    a Python object.

    :param fp: a file-like object.
    :param encode_nominal: boolean, if True perform a label encoding
        while reading the .arff file.
    :param return_type: determines the data structure used to store the
        dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
        `arff.DENSE_GEN` or `arff.LOD_GEN`.
        Consult the sections on `working with sparse data`_ and `loading
        progressively`_.
    :return: a dictionary.
    '''
    decoder = ArffDecoder()
    return decoder.decode(fp, encode_nominal=encode_nominal,
                          return_type=return_type)

# 将包含 ARFF 文档的字符串实例转换为 Python 对象
def loads(s, encode_nominal=False, return_type=DENSE):
    '''Convert a string instance containing the ARFF document into a Python
    object.

    :param s: a string object.
    :param encode_nominal: boolean, if True perform a label encoding
        while reading the .arff file.
    :param return_type: determines the data structure used to store the
        dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
        `arff.DENSE_GEN` or `arff.LOD_GEN`.
        Consult the sections on `working with sparse data`_ and `loading
        progressively`_.
    :return: a dictionary.
    '''
    decoder = ArffDecoder()
    return decoder.decode(s, encode_nominal=encode_nominal,
                          return_type=return_type)

# 将表示 ARFF 文档的对象序列化到给定的文件对象中
def dump(obj, fp):
    '''Serialize an object representing the ARFF document to a given file-like
    object.

    :param obj: a dictionary.
    :param fp: a file-like object.
    '''
    encoder = ArffEncoder()
    generator = encoder.iter_encode(obj)

    last_row = next(generator)
    for row in generator:
        fp.write(last_row + '\n')
        last_row = row
    fp.write(last_row)

    return fp

# 将表示 ARFF 文档的对象序列化为字符串并返回
def dumps(obj):
    '''Serialize an object representing the ARFF document, returning a string.

    :param obj: a dictionary.
    :return: a string with the ARFF document.
    '''
    encoder = ArffEncoder()
    return encoder.encode(obj)
# =============================================================================
```