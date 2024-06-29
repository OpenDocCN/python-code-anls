# `D:\src\scipysrc\matplotlib\lib\matplotlib\_afm.py`

```py
def _parse_header(fh):
    """
    Read the font metrics header (up to the char metrics) and returns
    a dictionary mapping *key* to *val*.  *val* will be converted to the
    appropriate python type as necessary; e.g.:

        * 'False'->False
        * '0'->0
        * '-168 -218 1000 898'-> [-168, -218, 1000, 898]

    Dictionary keys are

      StartFontMetrics, FontName, FullName, FamilyName, Weight,
      ItalicAngle, IsFixedPitch, FontBBox, UnderlinePosition,
      UnderlineThickness, Version, Notice, EncodingScheme, CapHeight,
      XHeight, Ascender, Descender, StartCharMetrics
    """
    # 定义一个空字典，用于存储解析后的键值对
    metrics_dict = {}

    # 逐行读取文件内容
    for line in fh:
        # 将每行内容解码为字符串，去掉首尾空格
        line = line.decode('latin-1').strip()
        
        # 如果行为空，则跳过继续下一行
        if not line:
            continue
        
        # 使用正则表达式匹配行中的键值对，键在行首到第一个空格之间，值从空格之后到行尾
        match = re.match(br'([^ ]+) *(.*)', line)
        
        # 如果匹配成功
        if match:
            key = match.group(1)  # 键
            val = match.group(2)  # 值
            
            # 根据键名选择对应的解析函数，将值转换为合适的数据类型
            if key == b'IsFixedPitch':
                val = _to_bool(val)
            elif key in (b'FontBBox', b'CharWidth', b'CharWidths', b'CharMetrics'):
                val = _to_list_of_ints(val)
            elif key in (b'UnderlinePosition', b'UnderlineThickness', b'Ascender', b'Descender', b'CapHeight', b'XHeight', b'ItalicAngle'):
                val = _to_int(val)
            else:
                val = _to_str(val)
            
            # 将解析后的键值对存入字典中
            metrics_dict[_to_str(key)] = val
    
    # 返回解析后的字典
    return metrics_dict
    # 定义字典，将不同的 AFM 文件头关键字映射到其对应的转换函数
    header_converters = {
        b'StartFontMetrics': _to_float,         # 将 b'StartFontMetrics' 转换为浮点数
        b'FontName': _to_str,                   # 将 b'FontName' 转换为字符串
        b'FullName': _to_str,                   # 将 b'FullName' 转换为字符串
        b'FamilyName': _to_str,                 # 将 b'FamilyName' 转换为字符串
        b'Weight': _to_str,                     # 将 b'Weight' 转换为字符串
        b'ItalicAngle': _to_float,              # 将 b'ItalicAngle' 转换为浮点数
        b'IsFixedPitch': _to_bool,              # 将 b'IsFixedPitch' 转换为布尔值
        b'FontBBox': _to_list_of_ints,          # 将 b'FontBBox' 转换为整数列表
        b'UnderlinePosition': _to_float,        # 将 b'UnderlinePosition' 转换为浮点数
        b'UnderlineThickness': _to_float,       # 将 b'UnderlineThickness' 转换为浮点数
        b'Version': _to_str,                    # 将 b'Version' 转换为字符串
        # 一些 AFM 文件包含非 ASCII 字符（规范不允许）。因为没有公共 API 可以访问这个字段，
        # 直接返回原始字节。
        b'Notice': lambda x: x,                 # 不做任何转换，直接返回原始值
        b'EncodingScheme': _to_str,             # 将 b'EncodingScheme' 转换为字符串
        b'CapHeight': _to_float,                # 将 b'CapHeight' 转换为浮点数
        b'Capheight': _to_float,                # 将 b'Capheight' 转换为浮点数（一些 AFM 文件中可能存在 'Capheight'）
        b'XHeight': _to_float,                  # 将 b'XHeight' 转换为浮点数
        b'Ascender': _to_float,                 # 将 b'Ascender' 转换为浮点数
        b'Descender': _to_float,                # 将 b'Descender' 转换为浮点数
        b'StdHW': _to_float,                    # 将 b'StdHW' 转换为浮点数
        b'StdVW': _to_float,                    # 将 b'StdVW' 转换为浮点数
        b'StartCharMetrics': _to_int,           # 将 b'StartCharMetrics' 转换为整数
        b'CharacterSet': _to_str,               # 将 b'CharacterSet' 转换为字符串
        b'Characters': _to_int,                 # 将 b'Characters' 转换为整数
    }
    d = {}  # 创建空字典，用于存储 AFM 文件头部数据
    first_line = True  # 标记是否是文件的第一行
    for line in fh:  # 遍历文件句柄中的每一行
        line = line.rstrip()  # 去除行尾的空白字符
        if line.startswith(b'Comment'):  # 如果行以 b'Comment' 开头，则跳过该行
            continue
        lst = line.split(b' ', 1)  # 以空格分割行，最多分割一次
        key = lst[0]  # 第一个分割出来的部分作为关键字
        if first_line:
            # 根据 AFM 规范第 4 节，文件的第一行必须是 'StartFontMetrics'
            # 后面可能跟着版本号，最后一行必须是非空的 'EndFontMetrics'
            if key != b'StartFontMetrics':
                raise RuntimeError('Not an AFM file')
            first_line = False  # 不再是文件的第一行
        if len(lst) == 2:
            val = lst[1]  # 如果分割出来有两部分，则第二部分为值
        else:
            val = b''  # 否则值为空字节串
        try:
            converter = header_converters[key]  # 获取关键字对应的转换函数
        except KeyError:
            _log.error("Found an unknown keyword in AFM header (was %r)", key)
            continue  # 如果关键字未在 header_converters 中定义，记录错误并跳过
        try:
            d[key] = converter(val)  # 将关键字和对应值进行转换，并存入字典 d 中
        except ValueError:
            _log.error('Value error parsing header in AFM: %s, %s', key, val)
            continue  # 转换过程中发生错误，记录错误并跳过
        if key == b'StartCharMetrics':
            break  # 如果遇到 'StartCharMetrics'，停止解析
    else:
        raise RuntimeError('Bad parse')  # 如果没有遇到 'StartCharMetrics'，抛出解析错误
    return d  # 返回解析得到的字典
# 定义命名元组 CharMetrics，表示单个字符的度量信息
CharMetrics = namedtuple('CharMetrics', 'width, name, bbox')

# 设置 CharMetrics 的文档字符串，描述字符度量的子集信息
CharMetrics.__doc__ = """
    Represents the character metrics of a single character.

    Notes
    -----
    The fields do currently only describe a subset of character metrics
    information defined in the AFM standard.
    """

# 设置 CharMetrics.width 的文档字符串，描述字符宽度 (WX)
CharMetrics.width.__doc__ = """The character width (WX)."""

# 设置 CharMetrics.name 的文档字符串，描述字符名 (N)
CharMetrics.name.__doc__ = """The character name (N)."""

# 设置 CharMetrics.bbox 的文档字符串，描述字符边界框 (B) 的元组 (*llx*, *lly*, *urx*, *ury*)
CharMetrics.bbox.__doc__ = """
    The bbox of the character (B) as a tuple (*llx*, *lly*, *urx*, *ury*).
"""


def _parse_char_metrics(fh):
    """
    Parse the given filehandle for character metrics information and return
    the information as dicts.

    It is assumed that the file cursor is on the line behind
    'StartCharMetrics'.

    Returns
    -------
    ascii_d : dict
         A mapping "ASCII num of the character" to `.CharMetrics`.
    name_d : dict
         A mapping "character name" to `.CharMetrics`.

    Notes
    -----
    This function is incomplete per the standard, but thus far parses
    all the sample afm files tried.
    """

    # 需要的度量标识符
    required_keys = {'C', 'WX', 'N', 'B'}

    # 初始化 ASCII 数到 CharMetrics 的映射字典
    ascii_d = {}
    # 初始化 字符名 到 CharMetrics 的映射字典
    name_d = {}

    # 遍历文件句柄中的每一行
    for line in fh:
        # 将字节字符串转换为字符串，并去除右侧空白字符
        line = _to_str(line.rstrip())  # Convert from byte-literal

        # 如果遇到 'EndCharMetrics' 开头的行，则停止解析并返回结果字典
        if line.startswith('EndCharMetrics'):
            return ascii_d, name_d

        # 将每行度量信息按分号分隔，并转换成字典格式
        vals = dict(s.strip().split(' ', 1) for s in line.split(';') if s)

        # 检查是否包含所有需要的度量标识符，否则抛出异常
        if not required_keys.issubset(vals):
            raise RuntimeError('Bad char metrics line: %s' % line)

        # 将度量信息转换成相应的数据类型
        num = _to_int(vals['C'])  # ASCII 数字
        wx = _to_float(vals['WX'])  # 字符宽度
        name = vals['N']  # 字符名
        bbox = _to_list_of_floats(vals['B'])  # 边界框数据
        bbox = list(map(int, bbox))  # 转换边界框数据为整数列表
        metrics = CharMetrics(wx, name, bbox)  # 创建 CharMetrics 对象

        # 特殊处理：如果字符名是 'Euro'，则使用相应的字符编码 128
        if name == 'Euro':
            num = 128
        # 特殊处理：如果字符名是 'minus'，则使用 Unicode MINUS SIGN 的编码 0x2212
        elif name == 'minus':
            num = ord("\N{MINUS SIGN}")  # 0x2212

        # 如果 ASCII 数不为 -1，则将度量信息存入 ascii_d 字典
        if num != -1:
            ascii_d[num] = metrics

        # 将度量信息按字符名存入 name_d 字典
        name_d[name] = metrics

    # 如果未能正常返回结果，则抛出解析异常
    raise RuntimeError('Bad parse')


def _parse_kern_pairs(fh):
    """
    Return a kern pairs dictionary; keys are (*char1*, *char2*) tuples and
    values are the kern pair value.  For example, a kern pairs line like
    ``KPX A y -50``

    will be represented as::

      d[ ('A', 'y') ] = -50

    """

    # 读取下一行并检查是否以 'StartKernPairs' 开头，否则抛出异常
    line = next(fh)
    if not line.startswith(b'StartKernPairs'):
        raise RuntimeError('Bad start of kern pairs data: %s' % line)

    # 初始化空的 kern pairs 字典
    d = {}
    # 遍历文件句柄 fh 中的每一行
    for line in fh:
        # 去除行末尾的空白字符（包括换行符）
        line = line.rstrip()
        # 如果行为空，则跳过当前循环，继续处理下一行
        if not line:
            continue
        # 如果行以 'EndKernPairs' 开头
        if line.startswith(b'EndKernPairs'):
            # 跳过下一行，通常是 'EndKernData'，并返回字典 d
            next(fh)  # EndKernData
            return d
        # 将行按空格分割成列表 vals
        vals = line.split()
        # 如果列表 vals 的长度不为 4，或者第一个元素不是 'KPX'，抛出异常
        if len(vals) != 4 or vals[0] != b'KPX':
            raise RuntimeError('Bad kern pairs line: %s' % line)
        # 将 vals 列表中的第 1、2、3 个元素转换为字符串类型，并分别赋值给 c1、c2、val
        c1, c2, val = _to_str(vals[1]), _to_str(vals[2]), _to_float(vals[3])
        # 将 (c1, c2) 作为键，val 作为值，添加到字典 d 中
        d[(c1, c2)] = val
    # 如果未在循环中遇到以 'EndKernPairs' 开头的行或不符合条件的行，则抛出异常
    raise RuntimeError('Bad kern pairs parse')
# 创建一个命名元组 CompositePart，用于表示复合字符的组成部分信息
CompositePart = namedtuple('CompositePart', 'name, dx, dy')

# 设置命名元组 CompositePart 的文档字符串，说明它表示复合字符的组成元素信息
CompositePart.__doc__ = """
    Represents the information on a composite element of a composite char.
"""

# 设置命名元组 CompositePart 的 name 属性的文档字符串，说明它表示复合元素的名称
CompositePart.name.__doc__ = """Name of the part, e.g. 'acute'."""

# 设置命名元组 CompositePart 的 dx 属性的文档字符串，说明它表示相对于原点的 x 方向位移
CompositePart.dx.__doc__ = """x-displacement of the part from the origin."""

# 设置命名元组 CompositePart 的 dy 属性的文档字符串，说明它表示相对于原点的 y 方向位移
CompositePart.dy.__doc__ = """y-displacement of the part from the origin."""


def _parse_composites(fh):
    """
    Parse the given filehandle for composites information return them as a
    dict.

    It is assumed that the file cursor is on the line behind 'StartComposites'.

    Returns
    -------
    dict
        A dict mapping composite character names to a parts list. The parts
        list is a list of `.CompositePart` entries describing the parts of
        the composite.

    Examples
    --------
    A composite definition line::

      CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;

    will be represented as::

      composites['Aacute'] = [CompositePart(name='A', dx=0, dy=0),
                              CompositePart(name='acute', dx=160, dy=170)]

    """
    # 初始化一个空字典，用于存储复合字符的信息
    composites = {}
    
    # 遍历文件句柄中的每一行
    for line in fh:
        # 去除行末尾的换行符和空格
        line = line.rstrip()
        
        # 跳过空行
        if not line:
            continue
        
        # 如果当前行以 'EndComposites' 开头，则返回已解析的复合字符信息字典
        if line.startswith(b'EndComposites'):
            return composites
        
        # 将当前行按分号分割成多个部分
        vals = line.split(b';')
        cc = vals[0].split()
        name, _num_parts = cc[1], _to_int(cc[2])
        
        # 初始化一个空列表，用于存储当前复合字符的各部分信息
        pccParts = []
        
        # 遍历除第一个和最后一个部分外的所有部分
        for s in vals[1:-1]:
            pcc = s.split()
            # 创建 CompositePart 对象，表示当前复合字符的一个部分，并加入列表
            part = CompositePart(pcc[1], _to_float(pcc[2]), _to_float(pcc[3]))
            pccParts.append(part)
        
        # 将当前复合字符及其各部分信息加入 composites 字典
        composites[name] = pccParts
    
    # 如果遍历结束仍未遇到 'EndComposites'，则抛出运行时错误
    raise RuntimeError('Bad composites parse')


def _parse_optional(fh):
    """
    Parse the optional fields for kern pair data and composites.

    Returns
    -------
    kern_data : dict
        A dict containing kerning information. May be empty.
        See `._parse_kern_pairs`.
    composites : dict
        A dict containing composite information. May be empty.
        See `._parse_composites`.
    """
    # 定义可选字段的解析函数映射表
    optional = {
        b'StartKernData': _parse_kern_pairs,
        b'StartComposites':  _parse_composites,
    }

    # 初始化一个字典 d，用于存储解析结果，默认为空字典
    d = {b'StartKernData': {},
         b'StartComposites': {}}
    
    # 遍历文件句柄中的每一行
    for line in fh:
        # 去除行末尾的换行符和空格
        line = line.rstrip()
        
        # 跳过空行
        if not line:
            continue
        
        # 将当前行按空格分割，并取第一个部分作为 key
        key = line.split()[0]

        # 如果 key 在可选字段映射表中，则调用相应的解析函数解析后续内容，并将结果存入 d 中
        if key in optional:
            d[key] = optional[key](fh)

    # 返回解析后的 kern 数据字典和 composites 字典
    return d[b'StartKernData'], d[b'StartComposites']


class AFM:
    
    def __init__(self, fh):
        """Parse the AFM file in file object *fh*."""
        # 解析 AFM 文件的头部信息，并存储在 _header 属性中
        self._header = _parse_header(fh)
        
        # 解析 AFM 文件中的字符度量信息，分别存储在 _metrics 和 _metrics_by_name 属性中
        self._metrics, self._metrics_by_name = _parse_char_metrics(fh)
        
        # 解析 AFM 文件中的可选字段（kern 对数据和复合字符信息），分别存储在 _kern 和 _composite 属性中
        self._kern, self._composite = _parse_optional(fh)

    def get_bbox_char(self, c, isord=False):
        # 如果 isord 为 False，则将字符 c 转换为其 ASCII 码
        if not isord:
            c = ord(c)
        
        # 返回字符 c 在 AFM 文件中的度量信息中的边界框属性
        return self._metrics[c].bbox
    def string_width_height(self, s):
        """
        Return the string width (including kerning) and string height
        as a (*w*, *h*) tuple.
        """
        # 如果字符串长度为0，返回空宽度和高度
        if not len(s):
            return 0, 0
        total_width = 0  # 初始化总宽度为0
        namelast = None  # 上一个字符的名称初始为None
        miny = 1e9  # 最小y坐标初始化为一个很大的数
        maxy = 0  # 最大y坐标初始化为0
        for c in s:  # 遍历字符串中的每个字符
            if c == '\n':  # 如果字符是换行符，跳过
                continue
            wx, name, bbox = self._metrics[ord(c)]  # 获取字符c的宽度wx、名称name和边界框bbox

            total_width += wx + self._kern.get((namelast, name), 0)  # 累加字符宽度和可能的字符间距
            l, b, w, h = bbox  # 解构边界框
            miny = min(miny, b)  # 更新最小y坐标
            maxy = max(maxy, b + h)  # 更新最大y坐标

            namelast = name  # 更新上一个字符的名称为当前字符的名称

        return total_width, maxy - miny  # 返回总宽度和字符高度差作为元组

    def get_str_bbox_and_descent(self, s):
        """Return the string bounding box and the maximal descent."""
        # 如果字符串长度为0，返回空边界框和0的下降值
        if not len(s):
            return 0, 0, 0, 0, 0
        total_width = 0  # 初始化总宽度为0
        namelast = None  # 上一个字符的名称初始为None
        miny = 1e9  # 最小y坐标初始化为一个很大的数
        maxy = 0  # 最大y坐标初始化为0
        left = 0  # 最左边界初始化为0
        if not isinstance(s, str):  # 如果输入的s不是字符串类型，转换为字符串
            s = _to_str(s)
        for c in s:  # 遍历字符串中的每个字符
            if c == '\n':  # 如果字符是换行符，跳过
                continue
            name = uni2type1.get(ord(c), f"uni{ord(c):04X}")  # 获取字符c的名称或者使用默认的unicode名称
            try:
                wx, _, bbox = self._metrics_by_name[name]  # 尝试从名称获取宽度wx和边界框bbox
            except KeyError:
                name = 'question'  # 如果名称不存在，使用默认名称'question'
                wx, _, bbox = self._metrics_by_name[name]  # 获取默认名称的宽度wx和边界框bbox
            total_width += wx + self._kern.get((namelast, name), 0)  # 累加字符宽度和可能的字符间距
            l, b, w, h = bbox  # 解构边界框
            left = min(left, l)  # 更新最左边界
            miny = min(miny, b)  # 更新最小y坐标
            maxy = max(maxy, b + h)  # 更新最大y坐标

            namelast = name  # 更新上一个字符的名称为当前字符的名称

        return left, miny, total_width, maxy - miny, -miny  # 返回左边界、最小y坐标、总宽度、字符高度差和最大下降值作为元组

    def get_str_bbox(self, s):
        """Return the string bounding box."""
        return self.get_str_bbox_and_descent(s)[:4]  # 调用get_str_bbox_and_descent方法并返回前四个元素，即左边界、最小y坐标、总宽度、字符高度差

    def get_name_char(self, c, isord=False):
        """Get the name of the character, i.e., ';' is 'semicolon'."""
        if not isord:
            c = ord(c)  # 如果c不是字符编码，转换为字符编码
        return self._metrics[c].name  # 返回字符c的名称

    def get_width_char(self, c, isord=False):
        """
        Get the width of the character from the character metric WX field.
        """
        if not isord:
            c = ord(c)  # 如果c不是字符编码，转换为字符编码
        return self._metrics[c].width  # 返回字符c的宽度

    def get_width_from_char_name(self, name):
        """Get the width of the character from a type1 character name."""
        return self._metrics_by_name[name].width  # 返回给定类型1字符名称name的宽度

    def get_height_char(self, c, isord=False):
        """Get the bounding box (ink) height of character *c* (space is 0)."""
        if not isord:
            c = ord(c)  # 如果c不是字符编码，转换为字符编码
        return self._metrics[c].bbox[-1]  # 返回字符c的墨迹高度

    def get_kern_dist(self, c1, c2):
        """
        Return the kerning pair distance (possibly 0) for chars *c1* and *c2*.
        """
        name1, name2 = self.get_name_char(c1), self.get_name_char(c2)  # 获取字符c1和c2的名称
        return self.get_kern_dist_from_name(name1, name2)  # 调用get_kern_dist_from_name方法，返回字符c1和c2的字符间距距离
    # 返回给定字符对 *name1* 和 *name2* 的字距（可能为0）
    def get_kern_dist_from_name(self, name1, name2):
        return self._kern.get((name1, name2), 0)

    # 返回字体的名称，例如 'Times-Roman'
    def get_fontname(self):
        return self._header[b'FontName']

    # 对于与FT2Font一致性，返回字体的PostScript名称
    @property
    def postscript_name(self):
        return self.get_fontname()

    # 返回字体的完整名称，例如 'Times-Roman'
    def get_fullname(self):
        name = self._header.get(b'FullName')
        if name is None:  # 如果FullName未指定，使用FontName作为替代
            name = self._header[b'FontName']
        return name

    # 返回字体的家族名称，例如 'Times'
    def get_familyname(self):
        name = self._header.get(b'FamilyName')
        if name is not None:
            return name

        # 如果FamilyName未指定，尝试从FullName推断
        name = self.get_fullname()
        extras = (r'(?i)([ -](regular|plain|italic|oblique|bold|semibold|'
                  r'light|ultralight|extra|condensed))+$')
        return re.sub(extras, '', name)

    # 返回字体的家族名称，例如 'Times'
    @property
    def family_name(self):
        return self.get_familyname()

    # 返回字体的粗细，例如 'Bold' 或 'Roman'
    def get_weight(self):
        return self._header[b'Weight']

    # 返回字体的倾斜角度，作为浮点数
    def get_angle(self):
        return self._header[b'ItalicAngle']

    # 返回大写字母高度，作为浮点数
    def get_capheight(self):
        return self._header[b'CapHeight']

    # 返回x高度，作为浮点数
    def get_xheight(self):
        return self._header[b'XHeight']

    # 返回下划线粗细，作为浮点数
    def get_underline_thickness(self):
        return self._header[b'UnderlineThickness']

    # 返回标准水平笔画宽度，作为浮点数；如果在AFM文件中未指定，则返回None
    def get_horizontal_stem_width(self):
        return self._header.get(b'StdHW', None)

    # 返回标准垂直笔画宽度，作为浮点数；如果在AFM文件中未指定，则返回None
    def get_vertical_stem_width(self):
        return self._header.get(b'StdVW', None)
```