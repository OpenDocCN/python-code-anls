# `D:\src\scipysrc\scipy\scipy\io\arff\_arffread.py`

```
# 导入正则表达式库
import re
# 导入日期时间库
import datetime

# 导入第三方库 numpy，并用 np 别名表示
import numpy as np

# 导入 CSV 处理库
import csv
# 导入 ctypes 库，用于处理 C 数据类型
import ctypes

"""A module to read arff files."""

# 定义模块中公开的接口列表
__all__ = ['MetaData', 'loadarff', 'ArffError', 'ParseArffError']

# ARFF 文件主要由两部分组成：头部（header）和数据（data）
#
# 头部包含以 @META 开头的各种组件，其中 META 是关键字之一（关系的属性，暂时为此）。
# 
# TODO:
#   - 整数和实数都被视为数值类型 -> 整数信息丢失！
#   - 将 ValueError 替换为 ParseError 或其他错误类型

# 正则表达式对象，用于匹配 ARFF 文件中的不同部分和结构
r_meta = re.compile(r'^\s*@')  # 匹配以 @ 开头的元数据行
r_comment = re.compile(r'^%')  # 匹配注释行
r_empty = re.compile(r'^\s+$')  # 匹配空行
r_headerline = re.compile(r'^\s*@\S*')  # 匹配以 @ 开头后跟单词的头部行
r_datameta = re.compile(r'^@[Dd][Aa][Tt][Aa]')  # 匹配 @DATA 或 @data 开头的行
r_relation = re.compile(r'^@[Rr][Ee][Ll][Aa][Tt][Ii][Oo][Nn]\s*(\S*)')  # 匹配 @RELATION 开头的行
r_attribute = re.compile(r'^\s*@[Aa][Tt][Tt][Rr][Ii][Bb][Uu][Tt][Ee]\s*(..*$)')  # 匹配以 @ATTRIBUTE 开头的行

r_nominal = re.compile(r'{(.+)}')  # 匹配形如 {value1,value2} 的标称属性值
r_date = re.compile(r"[Dd][Aa][Tt][Ee]\s+[\"']?(.+?)[\"']?$")  # 匹配日期类型的属性

# 匹配带有''的属性名称
r_comattrval = re.compile(r"'(..+)'\s+(..+$)")
# 匹配普通属性
r_wcomattrval = re.compile(r"(\S+)\s+(..+$)")

# ------------------------
# 自定义异常类
# ------------------------

# ARFF 文件解析异常类，继承自 OSError
class ArffError(OSError):
    pass

# 解析 ARFF 文件时的异常类，继承自 ArffError
class ParseArffError(ArffError):
    pass

# ----------
# 属性类定义
# ----------

# 属性基类
class Attribute:

    type_name = None

    def __init__(self, name):
        self.name = name
        self.range = None
        self.dtype = np.object_

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        如果可以解析属性行，则解析属性并返回解析后的属性对象，否则返回 None。
        """
        return None

    def parse_data(self, data_str):
        """
        解析此类型的值。
        """
        return None

    def __str__(self):
        """
        返回属性名称和类型名称的字符串表示。
        """
        return self.name + ',' + self.type_name

# 标称属性类，继承自 Attribute
class NominalAttribute(Attribute):

    type_name = 'nominal'

    def __init__(self, name, values):
        super().__init__(name)
        self.values = values
        self.range = values
        self.dtype = (np.bytes_, max(len(i) for i in values))
    def _get_nom_val(atrv):
        """
        Given a string containing a nominal type, returns a tuple of the
        possible values.

        A nominal type is defined as something framed between braces ({}).

        Parameters
        ----------
        atrv : str
           Nominal type definition

        Returns
        -------
        poss_vals : tuple
           Possible values extracted from the nominal type string

        Examples
        --------
        >>> from scipy.io.arff._arffread import NominalAttribute
        >>> NominalAttribute._get_nom_val("{floup, bouga, fl, ratata}")
        ('floup', 'bouga', 'fl', 'ratata')
        """
        # 使用正则表达式匹配大括号内的内容
        m = r_nominal.match(atrv)
        if m:
            # 如果匹配成功，提取大括号内部的内容并分割成列表
            attrs, _ = split_data_line(m.group(1))
            # 返回元组形式的可能取值
            return tuple(attrs)
        else:
            # 如果不是合法的标称类型字符串，抛出数值错误
            raise ValueError("This does not look like a nominal string")

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For nominal attributes, the attribute string would be like '{<attr_1>,
         <attr2>, <attr_3>}'.
        """
        if attr_string[0] == '{':
            # 如果属性字符串以大括号开头，调用_get_nom_val方法解析标称属性的可能取值
            values = cls._get_nom_val(attr_string)
            # 返回包含属性名称和可能取值的实例
            return cls(name, values)
        else:
            # 如果不是标称属性，返回None
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        if data_str in self.values:
            # 如果数据字符串在可能取值的集合中，返回数据字符串本身
            return data_str
        elif data_str == '?':
            # 如果数据字符串是问号，返回问号
            return data_str
        else:
            # 否则抛出数值错误，提示数据字符串不在可能取值集合中
            raise ValueError(f"{str(data_str)} value not in {str(self.values)}")

    def __str__(self):
        msg = self.name + ",{"
        # 构建属性名称和可能取值的字符串表示形式
        for i in range(len(self.values)-1):
            msg += self.values[i] + ","
        msg += self.values[-1]
        msg += "}"
        return msg
# 定义 NumericAttribute 类，继承自 Attribute 类
class NumericAttribute(Attribute):
    
    # 初始化方法，接受 name 参数作为属性名，并设置类型名称为 'numeric'，数据类型为 np.float64
    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'numeric'
        self.dtype = np.float64

    # 类方法，解析属性行，如果是数值型属性（'numeric', 'int', 'real'），则返回相应的 NumericAttribute 对象，否则返回 None
    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For numeric attributes, the attribute string would be like
        'numeric' or 'int' or 'real'.
        """
        attr_string = attr_string.lower().strip()

        if (attr_string[:len('numeric')] == 'numeric' or
           attr_string[:len('int')] == 'int' or
           attr_string[:len('real')] == 'real'):
            return cls(name)
        else:
            return None

    # 解析数据方法，根据 data_str 转换为相应的 float 值，如果数据为 '?'，则返回 np.nan
    def parse_data(self, data_str):
        """
        Parse a value of this type.

        Parameters
        ----------
        data_str : str
           string to convert

        Returns
        -------
        f : float
           where float can be nan

        Examples
        --------
        >>> from scipy.io.arff._arffread import NumericAttribute
        >>> atr = NumericAttribute('atr')
        >>> atr.parse_data('1')
        1.0
        >>> atr.parse_data('1\\n')
        1.0
        >>> atr.parse_data('?\\n')
        nan
        """
        if '?' in data_str:
            return np.nan
        else:
            return float(data_str)

    # 计算基本统计数据的方法，返回数据的最小值、最大值、均值和标准差乘以修正因子
    def _basic_stats(self, data):
        nbfac = data.size * 1. / (data.size - 1)
        return (np.nanmin(data), np.nanmax(data),
                np.mean(data), np.std(data) * nbfac)


# 定义 StringAttribute 类，继承自 Attribute 类
class StringAttribute(Attribute):

    # 初始化方法，接受 name 参数作为属性名，并设置类型名称为 'string'
    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'string'

    # 类方法，解析属性行，如果是字符串属性（'string'），则返回 StringAttribute 对象，否则返回 None
    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For string attributes, the attribute string would be like
        'string'.
        """
        attr_string = attr_string.lower().strip()

        if attr_string[:len('string')] == 'string':
            return cls(name)
        else:
            return None


# 定义 DateAttribute 类，继承自 Attribute 类
class DateAttribute(Attribute):

    # 初始化方法，接受 name、date_format 和 datetime_unit 参数，设置类型名称为 'date'，范围为 date_format，数据类型为 np.datetime64
    def __init__(self, name, date_format, datetime_unit):
        super().__init__(name)
        self.date_format = date_format
        self.datetime_unit = datetime_unit
        self.type_name = 'date'
        self.range = date_format
        self.dtype = np.datetime64(0, self.datetime_unit)

    # 静态方法，暂未提供具体实现
    @staticmethod
    def _get_date_format(atrv):
        # 使用正则表达式尝试匹配日期格式
        m = r_date.match(atrv)
        if m:
            # 提取匹配到的日期格式模式
            pattern = m.group(1).strip()
            # 将 Java 的 SimpleDateFormat 转换为 C 的格式
            datetime_unit = None
            if "yyyy" in pattern:
                pattern = pattern.replace("yyyy", "%Y")
                datetime_unit = "Y"
            elif "yy":
                pattern = pattern.replace("yy", "%y")
                datetime_unit = "Y"
            if "MM" in pattern:
                pattern = pattern.replace("MM", "%m")
                datetime_unit = "M"
            if "dd" in pattern:
                pattern = pattern.replace("dd", "%d")
                datetime_unit = "D"
            if "HH" in pattern:
                pattern = pattern.replace("HH", "%H")
                datetime_unit = "h"
            if "mm" in pattern:
                pattern = pattern.replace("mm", "%M")
                datetime_unit = "m"
            if "ss" in pattern:
                pattern = pattern.replace("ss", "%S")
                datetime_unit = "s"
            # 如果模式中包含时区信息，抛出异常，暂不支持
            if "z" in pattern or "Z" in pattern:
                raise ValueError("Date type attributes with time zone not "
                                 "supported, yet")
            
            # 如果没有识别出有效的时间单位，抛出异常
            if datetime_unit is None:
                raise ValueError("Invalid or unsupported date format")

            # 返回转换后的日期模式和时间单位
            return pattern, datetime_unit
        else:
            # 如果没有匹配到日期格式，抛出异常
            raise ValueError("Invalid or no date format")

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For date attributes, the attribute string would be like
        'date <format>'.
        """
        # 将属性字符串转换为小写并去除首尾空格
        attr_string_lower = attr_string.lower().strip()

        # 如果属性字符串以'date'开头
        if attr_string_lower[:len('date')] == 'date':
            # 解析日期格式和时间单位
            date_format, datetime_unit = cls._get_date_format(attr_string)
            # 返回解析后的日期属性对象
            return cls(name, date_format, datetime_unit)
        else:
            # 如果不是日期属性，返回None
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        # 去除数据字符串首尾空格以及可能的单引号或双引号
        date_str = data_str.strip().strip("'").strip('"')

        # 如果数据字符串是'?'，返回numpy中不确定的日期值
        if date_str == '?':
            return np.datetime64('NaT', self.datetime_unit)
        else:
            # 否则，根据指定的日期格式解析日期字符串为datetime对象，再转换为numpy的datetime64对象
            dt = datetime.datetime.strptime(date_str, self.date_format)
            return np.datetime64(dt).astype(
                "datetime64[%s]" % self.datetime_unit)

    def __str__(self):
        # 返回对象的字符串表示，包括继承的字符串表示和日期格式
        return super().__str__() + ',' + self.date_format
class RelationalAttribute(Attribute):
    # 定义一个继承自Attribute的关系属性类

    def __init__(self, name):
        # 初始化方法，设置属性名称，并初始化类型名称、数据类型、属性列表和方言
        super().__init__(name)
        self.type_name = 'relational'
        self.dtype = np.object_
        self.attributes = []
        self.dialect = None

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For date attributes, the attribute string would be like
        'date <format>'.
        """
        # 类方法：解析属性字符串，如果可解析则返回相应的属性对象，否则返回None

        attr_string_lower = attr_string.lower().strip()

        if attr_string_lower[:len('relational')] == 'relational':
            return cls(name)  # 如果属性字符串以'relational'开头，则返回RelationalAttribute对象
        else:
            return None  # 否则返回None

    def parse_data(self, data_str):
        # 解析数据方法

        elems = list(range(len(self.attributes)))  # 创建属性索引列表

        escaped_string = data_str.encode().decode("unicode-escape")  # 对数据字符串进行Unicode转义处理

        row_tuples = []

        for raw in escaped_string.split("\n"):
            row, self.dialect = split_data_line(raw, self.dialect)  # 使用split_data_line函数分割数据行

            row_tuples.append(tuple(
                [self.attributes[i].parse_data(row[i]) for i in elems]))  # 遍历数据行，解析每个属性的数据并构建元组列表

        return np.array(row_tuples,
                        [(a.name, a.dtype) for a in self.attributes])  # 返回包含解析数据的NumPy数组，属性名和数据类型由属性对象定义

    def __str__(self):
        return (super().__str__() + '\n\t' +
                '\n\t'.join(str(a) for a in self.attributes))
        # 返回属性对象的字符串表示形式，包括基类的字符串和所有属性对象的字符串表示形式


# -----------------
# Various utilities
# -----------------

def to_attribute(name, attr_string):
    """
    Convert an attribute string to an attribute object using available classes.
    """
    # 将属性字符串转换为属性对象，使用可用的属性类

    attr_classes = (NominalAttribute, NumericAttribute, DateAttribute,
                    StringAttribute, RelationalAttribute)  # 属性类的元组

    for cls in attr_classes:
        attr = cls.parse_attribute(name, attr_string)  # 尝试使用每个属性类解析属性字符串
        if attr is not None:
            return attr  # 如果解析成功，返回属性对象

    raise ParseArffError("unknown attribute %s" % attr_string)  # 如果无法识别属性字符串，则抛出解析错误


def csv_sniffer_has_bug_last_field():
    """
    Checks if the bug https://bugs.python.org/issue30157 is unpatched.
    """
    # 检查Python中的CSV嗅探器问题是否修复

    # 只进行一次计算
    has_bug = getattr(csv_sniffer_has_bug_last_field, "has_bug", None)

    if has_bug is None:
        dialect = csv.Sniffer().sniff("3, 'a'")
        csv_sniffer_has_bug_last_field.has_bug = dialect.quotechar != "'"
        has_bug = csv_sniffer_has_bug_last_field.has_bug

    return has_bug  # 返回是否存在CSV嗅探器的已知问题


def workaround_csv_sniffer_bug_last_field(sniff_line, dialect, delimiters):
    """
    Workaround for the bug https://bugs.python.org/issue30157 if is unpatched.
    """
    # 如果Python中的CSV嗅探器问题未修复，则提供的问题解决方法
    # 如果 csv_sniffer_has_bug_last_field() 函数返回 True，则执行以下逻辑
    if csv_sniffer_has_bug_last_field():
        # 定义正则表达式，用于匹配可能存在的 CSV 格式错误
        right_regex = r'(?P<delim>[^\w\n"\'])(?P<space> ?)(?P<quote>["\']).*?(?P=quote)(?:$|\n)'  # noqa: E501

        # 遍历不同的正则表达式模式，尝试匹配潜在的 CSV 格式错误
        for restr in (
            r'(?P<delim>[^\w\n"\'])(?P<space> ?)(?P<quote>["\']).*?(?P=quote)(?P=delim)',  # ,".*?",  # noqa: E501
            r'(?:^|\n)(?P<quote>["\']).*?(?P=quote)(?P<delim>[^\w\n"\'])(?P<space> ?)',  # .*?",  # noqa: E501
            right_regex,  # ,".*?"
            r'(?:^|\n)(?P<quote>["\']).*?(?P=quote)(?:$|\n)'  # ".*?" (no delim, no space)  # noqa: E501
        ):
            regexp = re.compile(restr, re.DOTALL | re.MULTILINE)
            # 在给定的字符串（sniff_line）中查找所有匹配的子串
            matches = regexp.findall(sniff_line)
            if matches:
                break

        # 如果匹配到的正则表达式不是 right_regex，则说明不存在该 bug，直接返回
        if restr != right_regex:
            return

        # 获取正则表达式中捕获组的索引
        groupindex = regexp.groupindex

        # 断言只有一个匹配结果
        assert len(matches) == 1
        m = matches[0]

        # 获取 quote 字符的索引并赋值给 quote 变量
        n = groupindex['quote'] - 1
        quote = m[n]

        # 获取 delim 字符的索引并赋值给 delim 变量
        n = groupindex['delim'] - 1
        delim = m[n]

        # 获取 space 标志位的索引并将其转换为布尔值赋值给 space 变量
        n = groupindex['space'] - 1
        space = bool(m[n])

        # 创建双引号包围的正则表达式对象，用于检测是否存在双引号转义情况
        dq_regexp = re.compile(
            rf"(({re.escape(delim)})|^)\W*{quote}[^{re.escape(delim)}\n]*{quote}[^{re.escape(delim)}\n]*{quote}\W*(({re.escape(delim)})|$)",
            re.MULTILINE  # noqa: E501
        )

        # 在给定字符串（sniff_line）中搜索是否存在双引号转义情况，返回结果赋值给 doublequote 变量
        doublequote = bool(dq_regexp.search(sniff_line))

        # 设置 dialect 对象的 quotechar 属性为 quote 变量的值
        dialect.quotechar = quote
        # 如果 delim 变量在 delimiters 集合中，则设置 dialect 对象的 delimiter 属性为 delim 变量的值
        if delim in delimiters:
            dialect.delimiter = delim
        # 设置 dialect 对象的 doublequote 和 skipinitialspace 属性为相应的布尔值
        dialect.doublequote = doublequote
        dialect.skipinitialspace = space
def split_data_line(line, dialect=None):
    # 定义可能的分隔符
    delimiters = ",\t"

    # 设置 CSV 字段大小限制为最大值的一半
    csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

    # 如果行末尾有换行符，则去除
    if line[-1] == '\n':
        line = line[:-1]
    
    # 去除行末尾的空白字符
    line = line.strip()
    
    sniff_line = line

    # 如果行中没有任何分隔符，则添加一个逗号，以避免 csv.Sniffer 报错
    if not any(d in line for d in delimiters):
        sniff_line += ","

    # 如果未指定方言，则使用 csv.Sniffer 推断方言
    if dialect is None:
        dialect = csv.Sniffer().sniff(sniff_line, delimiters=delimiters)
        workaround_csv_sniffer_bug_last_field(sniff_line=sniff_line,
                                              dialect=dialect,
                                              delimiters=delimiters)

    # 读取下一行数据并根据指定方言解析
    row = next(csv.reader([line], dialect))

    return row, dialect


# --------------
# Parsing header
# --------------
def tokenize_attribute(iterable, attribute):
    """Parse a raw string in header (e.g., starts by @attribute).

    Given a raw string attribute, try to get the name and type of the
    attribute. Constraints:

    * The first line must start with @attribute (case insensitive, and
      space like characters before @attribute are allowed)
    * Works also if the attribute is spread on multilines.
    * Works if empty lines or comments are in between

    Parameters
    ----------
    attribute : str
       the attribute string.

    Returns
    -------
    name : str
       name of the attribute
    value : str
       value of the attribute
    next : str
       next line to be parsed

    Examples
    --------
    If attribute is a string defined in python as r"floupi real", will
    return floupi as name, and real as value.

    >>> from scipy.io.arff._arffread import tokenize_attribute
    >>> iterable = iter([0] * 10) # dummy iterator
    >>> tokenize_attribute(iterable, r"@attribute floupi real")
    ('floupi', 'real', 0)

    If attribute is r"'floupi 2' real", will return 'floupi 2' as name,
    and real as value.

    >>> tokenize_attribute(iterable, r"  @attribute 'floupi 2' real   ")
    ('floupi 2', 'real', 0)

    """
    # 去除属性字符串两侧的空白字符
    sattr = attribute.strip()
    # 匹配属性字符串是否以 @attribute 开头
    mattr = r_attribute.match(sattr)
    if mattr:
        # atrv 是 @attribute 后面的内容
        atrv = mattr.group(1)
        if r_comattrval.match(atrv):
            name, type = tokenize_single_comma(atrv)
            next_item = next(iterable)
        elif r_wcomattrval.match(atrv):
            name, type = tokenize_single_wcomma(atrv)
            next_item = next(iterable)
        else:
            # 不确定是否支持多行属性
            raise ValueError("multi line not supported yet")
    else:
        raise ValueError("First line unparsable: %s" % sattr)

    attribute = to_attribute(name, type)
    # 如果 type 参数的小写形式等于 'relational'，则执行以下逻辑
    if type.lower() == 'relational':
        # 调用 read_relational_attribute 函数，获取下一个属性的值
        next_item = read_relational_attribute(iterable, attribute, next_item)
    #    raise ValueError("relational attributes not supported yet")
    
    # 返回 attribute 和 next_item 变量的值作为结果
    return attribute, next_item
# 使用正则表达式 r_comattrval 来对单个逗号分隔的字符串进行匹配和分析
def tokenize_single_comma(val):
    # 尝试匹配给定的字符串 val
    m = r_comattrval.match(val)
    if m:
        try:
            # 提取分组中的名称和类型，并去除两侧的空白字符
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            # 如果索引错误，抛出异常并附带更详细的错误信息
            raise ValueError("Error while tokenizing attribute") from e
    else:
        # 如果匹配失败，抛出带有详细错误信息的异常
        raise ValueError("Error while tokenizing single %s" % val)
    return name, type


# 使用正则表达式 r_wcomattrval 来对带逗号的字符串进行匹配和分析
def tokenize_single_wcomma(val):
    # 尝试匹配给定的字符串 val
    m = r_wcomattrval.match(val)
    if m:
        try:
            # 提取分组中的名称和类型，并去除两侧的空白字符
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            # 如果索引错误，抛出异常并附带更详细的错误信息
            raise ValueError("Error while tokenizing attribute") from e
    else:
        # 如果匹配失败，抛出带有详细错误信息的异常
        raise ValueError("Error while tokenizing single %s" % val)
    return name, type


# 读取关系属性的嵌套属性
def read_relational_attribute(ofile, relational_attribute, i):
    """Read the nested attributes of a relational attribute"""

    # 正则表达式匹配结束关系属性的标志
    r_end_relational = re.compile(r'^@[Ee][Nn][Dd]\s*' +
                                  relational_attribute.name + r'\s*$')

    # 循环读取直到遇到结束关系属性的标志
    while not r_end_relational.match(i):
        m = r_headerline.match(i)
        if m:
            # 如果匹配到头部行，则继续判断是否为属性行
            isattr = r_attribute.match(i)
            if isattr:
                # 如果是属性行，则进行属性解析并添加到关系属性的属性列表中
                attr, i = tokenize_attribute(ofile, i)
                relational_attribute.attributes.append(attr)
            else:
                # 如果不是属性行，则抛出异常
                raise ValueError("Error parsing line %s" % i)
        else:
            # 如果不是头部行，则继续读取下一行
            i = next(ofile)

    # 读取下一行，准备返回
    i = next(ofile)
    return i


# 读取可迭代对象 ofile 的头部信息
def read_header(ofile):
    """Read the header of the iterable ofile."""
    # 获取第一行数据
    i = next(ofile)

    # 跳过以 # 开头的注释行
    while r_comment.match(i):
        i = next(ofile)

    # 读取直到遇到 DATA 属性为止的所有信息
    relation = None
    attributes = []
    while not r_datameta.match(i):
        m = r_headerline.match(i)
        if m:
            # 如果匹配到头部行，则继续判断是否为属性行
            isattr = r_attribute.match(i)
            if isattr:
                # 如果是属性行，则进行属性解析并添加到属性列表中
                attr, i = tokenize_attribute(ofile, i)
                attributes.append(attr)
            else:
                # 如果不是属性行，则尝试匹配是否为关系行
                isrel = r_relation.match(i)
                if isrel:
                    # 如果是关系行，则获取关系名称
                    relation = isrel.group(1)
                else:
                    # 如果不是关系行，则抛出异常
                    raise ValueError("Error parsing line %s" % i)
                # 继续读取下一行
                i = next(ofile)
        else:
            # 如果不是头部行，则继续读取下一行
            i = next(ofile)

    # 返回关系名称和属性列表
    return relation, attributes


class MetaData:
    """Small container to keep useful information on a ARFF dataset.

    Knows about attributes names and types.

    Examples
    --------
    ::

        data, meta = loadarff('iris.arff')
        # This will print the attributes names of the iris.arff dataset
        for i in meta:
            print(i)
        # This works too
        meta.names()
        # Getting attribute type
        types = meta.types()

    Methods
    -------
    names
    types

    Notes
    """
    def __init__(self, rel, attr):
        # 初始化函数，接受关系名称 rel 和属性列表 attr
        self.name = rel
        # 使用字典推导式创建属性字典，键为属性名，值为属性对象本身
        self._attributes = {a.name: a for a in attr}

    def __repr__(self):
        # 返回对象的字符串表示形式
        msg = ""
        msg += "Dataset: %s\n" % self.name
        # 遍历属性字典，构建描述性信息并添加到 msg 中
        for i in self._attributes:
            msg += f"\t{i}'s type is {self._attributes[i].type_name}"
            # 如果属性有范围信息，则添加到 msg 中
            if self._attributes[i].range:
                msg += ", range is %s" % str(self._attributes[i].range)
            msg += '\n'
        return msg

    def __iter__(self):
        # 返回属性字典的迭代器
        return iter(self._attributes)

    def __getitem__(self, key):
        # 根据键获取属性对象，并返回其类型名和范围
        attr = self._attributes[key]
        return (attr.type_name, attr.range)

    def names(self):
        """Return the list of attribute names.

        Returns
        -------
        attrnames : list of str
            The attribute names.
        """
        # 返回属性字典的键列表，即属性名列表
        return list(self._attributes)

    def types(self):
        """Return the list of attribute types.

        Returns
        -------
        attr_types : list of str
            The attribute types.
        """
        # 返回属性字典中每个属性对象的类型名组成的列表
        attr_types = [self._attributes[name].type_name
                      for name in self._attributes]
        return attr_types
# 读取 ARFF 文件并返回其中的数据和元数据
def loadarff(f):
    """
    Read an arff file.

    The data is returned as a record array, which can be accessed much like
    a dictionary of NumPy arrays. For example, if one of the attributes is
    called 'pressure', then its first 10 data points can be accessed from the
    ``data`` record array like so: ``data['pressure'][0:10]``

    Parameters
    ----------
    f : file-like or str
       File-like object to read from, or filename to open.

    Returns
    -------
    data : record array
       The data of the arff file, accessible by attribute names.
    meta : `MetaData`
       Contains information about the arff file such as name and
       type of attributes, the relation (name of the dataset), etc.

    Raises
    ------
    ParseArffError
        This is raised if the given file is not ARFF-formatted.
    NotImplementedError
        The ARFF file has an attribute which is not supported yet.

    Notes
    -----
    This function should be able to read most arff files. Not
    implemented functionality include:
    * date type attributes
    * string type attributes

    It can read files with numeric and nominal attributes. It cannot read
    files with sparse data ({} in the file). However, this function can
    read files with missing data (? in the file), representing the data
    points as NaNs.

    Examples
    --------
    >>> from scipy.io import arff
    >>> from io import StringIO
    >>> content = \"\"\"
    ... @relation foo
    ... @attribute width  numeric
    ... @attribute height numeric
    ... @attribute color  {red,green,blue,yellow,black}
    ... @data
    ... 5.0,3.25,blue
    ... 4.5,3.75,green
    ... 3.0,4.00,red
    ... \"\"\"
    >>> f = StringIO(content)
    >>> data, meta = arff.loadarff(f)
    >>> data
    array([(5.0, 3.25, 'blue'), (4.5, 3.75, 'green'), (3.0, 4.0, 'red')],
          dtype=[('width', '<f8'), ('height', '<f8'), ('color', '|S6')])
    >>> meta
    Dataset: foo
        width's type is numeric
        height's type is numeric
        color's type is nominal, range is ('red', 'green', 'blue', 'yellow', 'black')

    """
    # 如果参数 f 是一个可以读取的对象，则直接使用；否则打开文件进行读取
    if hasattr(f, 'read'):
        ofile = f
    else:
        ofile = open(f)
    try:
        # 调用内部函数 _loadarff 处理文件读取
        return _loadarff(ofile)
    finally:
        # 如果 ofile 不是最初传入的 f，则关闭打开的文件
        if ofile is not f:  # only close what we opened
            ofile.close()


def _loadarff(ofile):
    # 解析 ARFF 文件的头部信息
    try:
        rel, attr = read_header(ofile)
    except ValueError as e:
        # 如果解析头部信息出错，则抛出 ParseArffError 异常
        msg = "Error while parsing header, error was: " + str(e)
        raise ParseArffError(msg) from e

    # 检查是否存在字符串类型的属性（目前不支持）
    hasstr = False
    for a in attr:
        if isinstance(a, StringAttribute):
            hasstr = True

    # 创建 MetaData 对象，保存关系和属性信息
    meta = MetaData(rel, attr)

    # XXX 以下代码实现并不完美
    # 构建类型描述符 descr 和转换器列表，用于将每个属性转换为适当的类型（应与 descr 中的类型匹配）
    # 如果我们希望支持整数作为整数值而不是数值类型（例如使用遮罩数组？），可以使用这部分代码。

    if hasstr:
        # 如何有效地支持字符串？理想情况下，在分配 NumPy 数组之前应该知道字符串的最大大小。
        raise NotImplementedError("String attributes not supported yet, sorry")

    ni = len(attr)

    def generator(row_iter, delim=','):
        # TODO: 这是我们花费时间的地方（约 80%）。我认为可以做得更有效：
        #   - 我们可以“编译”函数，因为这里有些值不会改变。
        #   - 可以动态生成将行转换为指定类型值的函数，并执行它，而不是循环执行。
        #   - 正则表达式用得太多了：对于注释，检查行是否以 % 开头应该足够快速，对于空行也是同样。
        #   --> 这似乎不会改变任何事情。

        # '编译'范围，因为它不会改变
        # 注意，我已经尝试过将转换器和行元素压缩在一起，但性能稍差。
        elems = list(range(ni))

        dialect = None
        for raw in row_iter:
            # 为了性能原因，我们不抽象跳过注释和空行。
            if r_comment.match(raw) or r_empty.match(raw):
                continue

            # 将数据行拆分并尝试解析数据
            row, dialect = split_data_line(raw, dialect)

            # 生成一个由元组组成的生成器，每个元组包含转换后的数据
            yield tuple([attr[i].parse_data(row[i]) for i in elems])

    # 使用生成器函数创建一个列表
    a = list(generator(ofile))
    # 这里不应该发生错误，否则就是一个 bug
    # 将列表转换为 NumPy 数组，使用属性的名称和数据类型作为字段描述
    data = np.array(a, [(a.name, a.dtype) for a in attr])
    return data, meta
# ----
# Misc
# ----

# 计算基本统计量：样本数除以自由度修正因子，防止样本数为1时的除零错误
def basic_stats(data):
    nbfac = data.size * 1. / (data.size - 1)
    # 返回数据的最小值、最大值、均值、标准差乘以自由度修正因子的元组
    return np.nanmin(data), np.nanmax(data), np.mean(data), np.std(data) * nbfac


# 打印属性信息，包括属性名、类型、最小值、最大值、均值、标准差
def print_attribute(name, tp, data):
    # 获取属性的类型名称
    type = tp.type_name
    # 如果类型是数值型（numeric、real、integer），计算并打印统计信息
    if type == 'numeric' or type == 'real' or type == 'integer':
        min, max, mean, std = basic_stats(data)
        print(f"{name},{type},{min:f},{max:f},{mean:f},{std:f}")
    else:
        # 否则，直接打印类型信息
        print(str(tp))


# 测试函数，加载指定文件的数据并进行测试
def test_weka(filename):
    # 载入文件数据和元数据
    data, meta = loadarff(filename)
    # 打印数据的dtype数量
    print(len(data.dtype))
    # 打印数据的元素总数
    print(data.size)
    # 遍历元数据，打印每个属性的统计信息
    for i in meta:
        print_attribute(i, meta[i], data[i])


# 确保该函数不被 nose 测试框架识别为测试用例
test_weka.__test__ = False


if __name__ == '__main__':
    import sys
    # 从命令行参数中获取文件名
    filename = sys.argv[1]
    # 调用测试函数
    test_weka(filename)
```