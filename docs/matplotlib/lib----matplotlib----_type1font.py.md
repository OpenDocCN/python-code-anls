# `D:\src\scipysrc\matplotlib\lib\matplotlib\_type1font.py`

```py
"""
A class representing a Type 1 font.

This version reads pfa and pfb files and splits them for embedding in
pdf files. It also supports SlantFont and ExtendFont transformations,
similarly to pdfTeX and friends. There is no support yet for subsetting.

Usage::

    font = Type1Font(filename)
    clear_part, encrypted_part, finale = font.parts
    slanted_font = font.transform({'slant': 0.167})
    extended_font = font.transform({'extend': 1.2})

Sources:

* Adobe Technical Note #5040, Supporting Downloadable PostScript
  Language Fonts.

* Adobe Type 1 Font Format, Adobe Systems Incorporated, third printing,
  v1.1, 1993. ISBN 0-201-57044-0.
"""

from __future__ import annotations  # 允许使用 annotations 作为类型提示

import binascii  # 提供二进制数据和ASCII字符串之间的转换
import functools  # 提供创建和管理装饰器的工具
import logging  # 提供日志记录功能
import re  # 提供正则表达式支持
import string  # 提供字符串处理相关的常量和函数
import struct  # 提供处理结构化数据的功能
import typing as T  # 提供类型提示的支持

import numpy as np  # 提供数值计算和数组处理的功能

from matplotlib.cbook import _format_approx  # 导入模块中的特定函数
from . import _api  # 导入当前包或模块的特定子模块或对象

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class _Token:
    """
    A token in a PostScript stream.

    Attributes
    ----------
    pos : int
        Position, i.e. offset from the beginning of the data.
    raw : str
        Raw text of the token.
    kind : str
        Description of the token (for debugging or testing).
    """
    __slots__ = ('pos', 'raw')  # 限定对象可以拥有的属性，节省内存
    kind = '?'  # 初始化默认的 token 类型为未知类型

    def __init__(self, pos, raw):
        _log.debug('type1font._Token %s at %d: %r', self.kind, pos, raw)
        self.pos = pos  # 设置 token 的位置信息
        self.raw = raw  # 设置 token 的原始文本信息

    def __str__(self):
        return f"<{self.kind} {self.raw} @{self.pos}>"  # 返回 token 的字符串表示形式

    def endpos(self):
        """Position one past the end of the token"""
        return self.pos + len(self.raw)  # 返回 token 结束位置的下一个位置的索引

    def is_keyword(self, *names):
        """Is this a name token with one of the names?"""
        return False  # 判断是否为关键字 token，默认返回 False

    def is_slash_name(self):
        """Is this a name token that starts with a slash?"""
        return False  # 判断是否为以斜杠开头的名称 token，默认返回 False

    def is_delim(self):
        """Is this a delimiter token?"""
        return False  # 判断是否为分隔符 token，默认返回 False

    def is_number(self):
        """Is this a number token?"""
        return False  # 判断是否为数字 token，默认返回 False

    def value(self):
        return self.raw  # 返回 token 的原始文本作为其值


class _NameToken(_Token):
    kind = 'name'  # 设置名称 token 的类型为 'name'

    def is_slash_name(self):
        return self.raw.startswith('/')  # 判断是否为以斜杠开头的名称 token

    def value(self):
        return self.raw[1:]  # 返回去掉开头斜杠后的名称 token 的值


class _BooleanToken(_Token):
    kind = 'boolean'  # 设置布尔值 token 的类型为 'boolean'

    def value(self):
        return self.raw == 'true'  # 返回布尔值 token 是否为 'true'


class _KeywordToken(_Token):
    kind = 'keyword'  # 设置关键字 token 的类型为 'keyword'

    def is_keyword(self, *names):
        return self.raw in names  # 判断关键字 token 是否在给定的名称列表中


class _DelimiterToken(_Token):
    kind = 'delimiter'  # 设置分隔符 token 的类型为 'delimiter'

    def is_delim(self):
        return True  # 始终返回 True，表示该 token 是分隔符 token

    def opposite(self):
        return {'[': ']', ']': '[', '{': '}', '}': '{', '<<': '>>', '>>': '<<'}[self.raw]
        # 返回与当前分隔符 token 对应的反向分隔符


class _WhitespaceToken(_Token):
    kind = 'whitespace'  # 设置空白符 token 的类型为 'whitespace'


class _StringToken(_Token):
    kind = 'string'  # 设置字符串 token 的类型为 'string'
    _escapes_re = re.compile(r'\\([\\()nrtbf]|[0-7]{1,3})')
    # 编译正则表达式，用于匹配字符串中的转义序列
    # 定义用于转义的字符映射字典，包括反斜杠和常见转义字符
    _replacements = {'\\': '\\', '(': '(', ')': ')', 'n': '\n',
                     'r': '\r', 't': '\t', 'b': '\b', 'f': '\f'}
    
    # 正则表达式，用于匹配空白字符和特殊控制字符
    _ws_re = re.compile('[\0\t\r\f\n ]')

    # 类方法装饰器，使用LRU缓存来优化性能
    @classmethod
    def _escape(cls, match):
        # 提取匹配的组
        group = match.group(1)
        try:
            # 尝试在替换字典中找到匹配的转义字符
            return cls._replacements[group]
        except KeyError:
            # 如果不在替换字典中，则将匹配的八进制字符转换为相应的ASCII字符
            return chr(int(group, 8))

    # 实例方法，返回处理后的值
    @functools.lru_cache
    def value(self):
        # 如果原始数据以 '(' 开头，则处理转义字符
        if self.raw[0] == '(':
            return self._escapes_re.sub(self._escape, self.raw[1:-1])
        else:
            # 否则，去除空白字符后，如果长度为奇数则补 '0'，再解码为二进制数据
            data = self._ws_re.sub('', self.raw[1:-1])
            if len(data) % 2 == 1:
                data += '0'
            return binascii.unhexlify(data)
class _BinaryToken(_Token):
    # 定义私有类 _BinaryToken，继承自 _Token，表示二进制类型的标记
    kind = 'binary'

    def value(self):
        # 返回该标记的值，即去除首个字符后的原始数据
        return self.raw[1:]


class _NumberToken(_Token):
    # 定义私有类 _NumberToken，继承自 _Token，表示数字类型的标记
    kind = 'number'

    def is_number(self):
        # 检查该标记是否为数字类型，始终返回 True
        return True

    def value(self):
        # 根据标记的原始数据内容，如果不包含小数点，则返回整数形式，否则返回浮点数形式
        if '.' not in self.raw:
            return int(self.raw)
        else:
            return float(self.raw)


def _tokenize(data: bytes, skip_ws: bool) -> T.Generator[_Token, int, None]:
    """
    从 Type-1 字体代码中生成 _Token 实例的生成器。

    使用者可以发送一个整数到这个分词器，以指示下一个标记应为给定长度的 _BinaryToken。

    Parameters
    ----------
    data : bytes
        要分词的字体数据。

    skip_ws : bool
        如果为 True，则生成器将从输出中丢弃任何 _WhitespaceToken。
    """
    text = data.decode('ascii', 'replace')
    # 匹配空白字符或注释
    whitespace_or_comment_re = re.compile(r'[\0\t\r\f\n ]+|%[^\r\n]*')
    # 匹配标记内容
    token_re = re.compile(r'/{0,2}[^]\0\t\r\f\n ()<>{}/%[]+')
    # 匹配字符串内部元素
    instring_re = re.compile(r'[()\\]')
    # 匹配十六进制字符串
    hex_re = re.compile(r'^<[0-9a-fA-F\0\t\r\f\n ]*>$')
    # 匹配八进制数字
    oct_re = re.compile(r'[0-7]{1,3}')
    pos = 0
    next_binary: int | None = None


class _BalancedExpression(_Token):
    # 定义私有类 _BalancedExpression，继承自 _Token
    pass


def _expression(initial, tokens, data):
    """
    消耗一些标记并返回一个平衡的 PostScript 表达式。

    Parameters
    ----------
    initial : _Token
        触发解析平衡表达式的初始标记。

    tokens : iterator of _Token
        后续的标记。

    data : bytes
        标记位置指向的底层数据。

    Returns
    -------
    _BalancedExpression
    """
    delim_stack = []
    token = initial
    while True:
        if token.is_delim():
            if token.raw in ('[', '{'):
                delim_stack.append(token)
            elif token.raw in (']', '}'):
                if not delim_stack:
                    raise RuntimeError(f"unmatched closing token {token}")
                match = delim_stack.pop()
                if match.raw != token.opposite():
                    raise RuntimeError(
                        f"opening token {match} closed by {token}"
                    )
                if not delim_stack:
                    break
            else:
                raise RuntimeError(f'unknown delimiter {token}')
        elif not delim_stack:
            break
        token = next(tokens)
    # 构造一个 _BalancedExpression 实例，包括起始位置和从起始位置到结束位置的数据内容
    return _BalancedExpression(
        initial.pos,
        data[initial.pos:token.endpos()].decode('ascii', 'replace')
    )


class Type1Font:
    """
    表示 Type-1 字体的类，供后端使用。

    Attributes
    ----------
    parts : tuple
        包含明文部分、加密部分和末尾零的三元组。

    decrypted : bytes
        “parts[1]” 的解密形式。
    """
    prop : dict[str, Any]
        A dictionary of font properties. Noteworthy keys include:

        - FontName: PostScript name of the font
        - Encoding: dict from numeric codes to glyph names
        - FontMatrix: bytes object encoding a matrix
        - UniqueID: optional font identifier, dropped when modifying the font
        - CharStrings: dict from glyph names to byte code
        - Subrs: array of byte code subroutines
        - OtherSubrs: bytes object encoding some PostScript code
    """
    __slots__ = ('parts', 'decrypted', 'prop', '_pos', '_abbr')
    """
    The _pos dict contains (begin, end) indices to parts[0] + decrypted
    so that they can be replaced when transforming the font;
    but since sometimes a definition appears in both parts[0] and decrypted,
    _pos[name] is an array of such pairs
    
    _abbr maps three standard abbreviations to their particular names in
    this font (e.g. 'RD' is named '-|' in some fonts)
    """

    def __init__(self, input):
        """
        Initialize a Type-1 font.

        Parameters
        ----------
        input : str or 3-tuple
            Either a pfb file name, or a 3-tuple of already-decoded Type-1
            font `~.Type1Font.parts`.
        """
        if isinstance(input, tuple) and len(input) == 3:
            self.parts = input
        else:
            with open(input, 'rb') as file:
                data = self._read(file)
            self.parts = self._split(data)

        self.decrypted = self._decrypt(self.parts[1], 'eexec')
        self._abbr = {'RD': 'RD', 'ND': 'ND', 'NP': 'NP'}
        self._parse()

    def _read(self, file):
        """
        Read the font from a file, decoding into usable parts.

        Parameters
        ----------
        file : file-like object
            The file object from which to read the font data.

        Returns
        -------
        bytes
            Decoded font data.

        Raises
        ------
        RuntimeError
            If the PFB file format is broken or contains unexpected segments.
        """
        rawdata = file.read()
        if not rawdata.startswith(b'\x80'):
            return rawdata

        data = b''
        while rawdata:
            if not rawdata.startswith(b'\x80'):
                raise RuntimeError('Broken pfb file (expected byte 128, '
                                   'got %d)' % rawdata[0])
            type = rawdata[1]
            if type in (1, 2):
                length, = struct.unpack('<i', rawdata[2:6])
                segment = rawdata[6:6 + length]
                rawdata = rawdata[6 + length:]

            if type == 1:       # ASCII text: include verbatim
                data += segment
            elif type == 2:     # binary data: encode in hexadecimal
                data += binascii.hexlify(segment)
            elif type == 3:     # end of file
                break
            else:
                raise RuntimeError('Unknown segment type %d in pfb file' % type)

        return data
    def _split(self, data):
        """
        Split the Type 1 font into its three main parts.

        The three parts are: (1) the cleartext part, which ends in a
        eexec operator; (2) the encrypted part; (3) the fixed part,
        which contains 512 ASCII zeros possibly divided on various
        lines, a cleartomark operator, and possibly something else.
        """

        # Cleartext part: just find the eexec and skip whitespace
        # 找到字体文件中 cleartext 部分的结束标志 eexec，并跳过空白字符
        idx = data.index(b'eexec')
        idx += len(b'eexec')
        while data[idx] in b' \t\r\n':
            idx += 1
        len1 = idx

        # Encrypted part: find the cleartomark operator and count
        # zeros backward
        # 找到字体文件中 encrypted 部分的起始点，通过 cleartomark 操作符来定位，并向前计算零的个数
        idx = data.rindex(b'cleartomark') - 1
        zeros = 512
        while zeros and (data[idx] in b'0' or data[idx] in b'\r\n'):
            if data[idx] in b'0':
                zeros -= 1
            idx -= 1
        if zeros:
            # this may have been a problem on old implementations that
            # used the zeros as necessary padding
            # 这可能是旧实现中的问题，这些零被用作必要的填充
            _log.info('Insufficiently many zeros in Type 1 font')

        # Convert encrypted part to binary (if we read a pfb file, we may end
        # up converting binary to hexadecimal to binary again; but if we read
        # a pfa file, this part is already in hex, and I am not quite sure if
        # even the pfb format guarantees that it will be in binary).
        # 将 encrypted 部分转换为二进制数据（如果读取的是 pfb 文件，可能会涉及二进制到十六进制再到二进制的转换；但如果读取的是 pfa 文件，这部分数据已经是十六进制了，而且我不确定即使 pfb 格式也能保证其为二进制数据）。
        idx1 = len1 + ((idx - len1 + 2) & ~1)  # ensure an even number of bytes
        binary = binascii.unhexlify(data[len1:idx1])

        return data[:len1], binary, data[idx+1:]
    def _encrypt(plaintext, key, ndiscard=4):
        """
        Encrypt plaintext using the Type-1 font algorithm.

        The algorithm is described in Adobe's "Adobe Type 1 Font Format".
        The key argument can be an integer, or one of the strings
        'eexec' and 'charstring', which map to the key specified for the
        corresponding part of Type-1 fonts.

        The ndiscard argument should be an integer, usually 4. That
        number of bytes is prepended to the plaintext before encryption.
        This function prepends NUL bytes for reproducibility, even though
        the original algorithm uses random bytes, presumably to avoid
        cryptanalysis.
        """

        # 根据传入的 key 参数选择正确的加密密钥
        key = _api.check_getitem({'eexec': 55665, 'charstring': 4330}, key=key)
        ciphertext = []
        # 将 ndiscard 个 NUL 字节和 plaintext 组合起来进行加密
        for byte in b'\0' * ndiscard + plaintext:
            # 计算加密后的字节
            c = byte ^ (key >> 8)
            ciphertext.append(c)
            # 更新加密密钥，参照 Type-1 字体算法
            key = ((key + c) * 52845 + 22719) & 0xffff

        return bytes(ciphertext)

    def _parse_subrs(self, tokens, _data):
        # 获取 /Subrs 后面的 token，应为一个数字
        count_token = next(tokens)
        if not count_token.is_number():
            # 如果不是数字，则抛出异常
            raise RuntimeError(
                f"Token following /Subrs must be a number, was {count_token}"
            )
        count = count_token.value()
        # 创建一个长度为 count 的数组，用于存储 subroutines 的内容
        array = [None] * count
        # 跳过数组定义的关键字
        next(t for t in tokens if t.is_keyword('array'))
        for _ in range(count):
            # 跳过 dup 关键字
            next(t for t in tokens if t.is_keyword('dup'))
            # 获取 subroutine 的索引值
            index_token = next(tokens)
            if not index_token.is_number():
                # 如果索引值不是数字，则抛出异常
                raise RuntimeError(
                    "Token following dup in Subrs definition must be a "
                    f"number, was {index_token}"
                )
            # 获取字节数量的 token
            nbytes_token = next(tokens)
            if not nbytes_token.is_number():
                # 如果不是数字，则抛出异常
                raise RuntimeError(
                    "Second token following dup in Subrs definition must "
                    f"be a number, was {nbytes_token}"
                )
            # 获取 subroutine 的定义 token
            token = next(tokens)
            if not token.is_keyword(self._abbr['RD']):
                # 如果不是预期的关键字，则抛出异常
                raise RuntimeError(
                    f"Token preceding subr must be {self._abbr['RD']}, "
                    f"was {token}"
                )
            # 读取 binary token，长度为 nbytes_token 的值加 1
            binary_token = tokens.send(1+nbytes_token.value())
            # 将 subroutine 的二进制内容存储到数组对应的索引位置
            array[index_token.value()] = binary_token.value()

        # 返回存储 subroutines 内容的数组和 tokens 的结束位置
        return array, next(tokens).endpos()
    def _parse_charstrings(tokens, _data):
        # 获取CharStrings的数量，并验证其后是否为数字，否则引发异常
        count_token = next(tokens)
        if not count_token.is_number():
            raise RuntimeError(
                "Token following /CharStrings must be a number, "
                f"was {count_token}"
            )
        count = count_token.value()
        # 初始化字符字典
        charstrings = {}
        # 跳过直到遇到'begin'关键字
        next(t for t in tokens if t.is_keyword('begin'))
        # 循环解析CharStrings定义
        while True:
            # 找到下一个关键字为'end'或者斜杠名字的token
            token = next(t for t in tokens
                         if t.is_keyword('end') or t.is_slash_name())
            # 如果token是'end'，则返回字符字典和token的结束位置
            if token.raw == 'end':
                return charstrings, token.endpos()
            # 获取glyphname
            glyphname = token.value()
            # 获取字符数据的字节数
            nbytes_token = next(tokens)
            if not nbytes_token.is_number():
                raise RuntimeError(
                    f"Token following /{glyphname} in CharStrings definition "
                    f"must be a number, was {nbytes_token}"
                )
            # 跳过下一个token，通常是RD或者|-，然后获取二进制数据
            next(tokens)  # usually RD or |-
            binary_token = tokens.send(1+nbytes_token.value())
            # 将glyphname和对应的二进制数据存入字符字典
            charstrings[glyphname] = binary_token.value()

    @staticmethod
    def _parse_encoding(tokens, _data):
        # 仅适用于遵循Adobe手册的编码，对于包含非兼容数据的旧字体，记录警告并返回可能不完整的编码
        encoding = {}
        while True:
            # 找到下一个关键字为'StandardEncoding'、'dup'或者'def'的token
            token = next(t for t in tokens
                         if t.is_keyword('StandardEncoding', 'dup', 'def'))
            # 如果token是'StandardEncoding'，返回标准编码和token的结束位置
            if token.is_keyword('StandardEncoding'):
                return _StandardEncoding, token.endpos()
            # 如果token是'def'，返回编码字典和token的结束位置
            if token.is_keyword('def'):
                return encoding, token.endpos()
            # 获取索引号token
            index_token = next(tokens)
            if not index_token.is_number():
                _log.warning(
                    f"Parsing encoding: expected number, got {index_token}"
                )
                continue
            # 获取名称token
            name_token = next(tokens)
            if not name_token.is_slash_name():
                _log.warning(
                    f"Parsing encoding: expected slash-name, got {name_token}"
                )
                continue
            # 将索引号和名称存入编码字典
            encoding[index_token.value()] = name_token.value()

    @staticmethod
    def _parse_othersubrs(tokens, data):
        init_pos = None
        while True:
            # 获取下一个token
            token = next(tokens)
            # 如果初始位置为None，则将其设置为当前token的位置
            if init_pos is None:
                init_pos = token.pos
            # 如果token是分隔符，则解析表达式
            if token.is_delim():
                _expression(token, tokens, data)
            # 如果token是'def'、'ND'或者'|-'，返回从初始位置到token结束位置的数据和token的结束位置
            elif token.is_keyword('def', 'ND', '|-'):
                return data[init_pos:token.endpos()], token.endpos()
    def transform(self, effects):
        """
        Return a new font that is slanted and/or extended.

        Parameters
        ----------
        effects : dict
            A dict with optional entries:

            - 'slant' : float, default: 0
                Tangent of the angle that the font is to be slanted to the
                right. Negative values slant to the left.
            - 'extend' : float, default: 1
                Scaling factor for the font width. Values less than 1 condense
                the glyphs.

        Returns
        -------
        `Type1Font`
        """
        # 获取字体名称和斜体角度
        fontname = self.prop['FontName']
        italicangle = self.prop['ItalicAngle']

        # 解析字体矩阵并创建变换前的矩阵
        array = [
            float(x) for x in (self.prop['FontMatrix']
                               .lstrip('[').rstrip(']').split())
        ]
        oldmatrix = np.eye(3, 3)
        oldmatrix[0:3, 0] = array[::2]
        oldmatrix[0:3, 1] = array[1::2]
        modifier = np.eye(3, 3)

        # 如果效果字典中包含'slant'条目，进行倾斜变换
        if 'slant' in effects:
            slant = effects['slant']
            fontname += f'_Slant_{int(1000 * slant)}'
            italicangle = round(
                float(italicangle) - np.arctan(slant) / np.pi * 180,
                5
            )
            modifier[1, 0] = slant

        # 如果效果字典中包含'extend'条目，进行宽度扩展变换
        if 'extend' in effects:
            extend = effects['extend']
            fontname += f'_Extend_{int(1000 * extend)}'
            modifier[0, 0] = extend

        # 计算新的字体矩阵
        newmatrix = np.dot(modifier, oldmatrix)
        array[::2] = newmatrix[0:3, 0]
        array[1::2] = newmatrix[0:3, 1]
        fontmatrix = (
            f"[{' '.join(_format_approx(x, 6) for x in array)}]"
        )

        # 准备替换内容，更新字体名称、斜体角度和字体矩阵
        replacements = (
            [(x, f'/FontName/{fontname} def')
             for x in self._pos['FontName']]
            + [(x, f'/ItalicAngle {italicangle} def')
               for x in self._pos['ItalicAngle']]
            + [(x, f'/FontMatrix {fontmatrix} readonly def')
               for x in self._pos['FontMatrix']]
            + [(x, '') for x in self._pos.get('UniqueID', [])]
        )

        # 构建修改后的字节流数据
        data = bytearray(self.parts[0])
        data.extend(self.decrypted)
        len0 = len(self.parts[0])
        for (pos0, pos1), value in sorted(replacements, reverse=True):
            data[pos0:pos1] = value.encode('ascii', 'replace')
            if pos0 < len(self.parts[0]):
                if pos1 >= len(self.parts[0]):
                    raise RuntimeError(
                        f"text to be replaced with {value} spans "
                        "the eexec boundary"
                    )
                len0 += len(value) - pos1 + pos0

        # 转换为字节数据并返回Type1Font对象
        data = bytes(data)
        return Type1Font((
            data[:len0],
            self._encrypt(data[len0:], 'eexec'),
            self.parts[2]
        ))
# 定义标准编码表，将 ASCII 字符映射到其对应的描述字符串
_StandardEncoding = {
    **{ord(letter): letter for letter in string.ascii_letters},
    0: '.notdef',           # 控制字符 0 映射到 ".notdef"
    32: 'space',            # ASCII 码 32（空格符）映射到 "space"
    33: 'exclam',           # ASCII 码 33（!）映射到 "exclam"
    34: 'quotedbl',         # ASCII 码 34（"）映射到 "quotedbl"
    35: 'numbersign',       # ASCII 码 35（#）映射到 "numbersign"
    36: 'dollar',           # ASCII 码 36（$）映射到 "dollar"
    37: 'percent',          # ASCII 码 37（%）映射到 "percent"
    38: 'ampersand',        # ASCII 码 38（&）映射到 "ampersand"
    39: 'quoteright',       # ASCII 码 39（'）映射到 "quoteright"
    40: 'parenleft',        # ASCII 码 40（(）映射到 "parenleft"
    41: 'parenright',       # ASCII 码 41（)）映射到 "parenright"
    42: 'asterisk',         # ASCII 码 42（*）映射到 "asterisk"
    43: 'plus',             # ASCII 码 43（+）映射到 "plus"
    44: 'comma',            # ASCII 码 44（,）映射到 "comma"
    45: 'hyphen',           # ASCII 码 45（-）映射到 "hyphen"
    46: 'period',           # ASCII 码 46（.）映射到 "period"
    47: 'slash',            # ASCII 码 47（/）映射到 "slash"
    48: 'zero',             # ASCII 码 48（0）映射到 "zero"
    49: 'one',              # ASCII 码 49（1）映射到 "one"
    50: 'two',              # ASCII 码 50（2）映射到 "two"
    51: 'three',            # ASCII 码 51（3）映射到 "three"
    52: 'four',             # ASCII 码 52（4）映射到 "four"
    53: 'five',             # ASCII 码 53（5）映射到 "five"
    54: 'six',              # ASCII 码 54（6）映射到 "six"
    55: 'seven',            # ASCII 码 55（7）映射到 "seven"
    56: 'eight',            # ASCII 码 56（8）映射到 "eight"
    57: 'nine',             # ASCII 码 57（9）映射到 "nine"
    58: 'colon',            # ASCII 码 58（:）映射到 "colon"
    59: 'semicolon',        # ASCII 码 59（;）映射到 "semicolon"
    60: 'less',             # ASCII 码 60（<）映射到 "less"
    61: 'equal',            # ASCII 码 61（=）映射到 "equal"
    62: 'greater',          # ASCII 码 62（>）映射到 "greater"
    63: 'question',         # ASCII 码 63（?）映射到 "question"
    64: 'at',               # ASCII 码 64（@）映射到 "at"
    91: 'bracketleft',      # ASCII 码 91（[）映射到 "bracketleft"
    92: 'backslash',        # ASCII 码 92（\）映射到 "backslash"
    93: 'bracketright',     # ASCII 码 93（]）映射到 "bracketright"
    94: 'asciicircum',      # ASCII 码 94（^）映射到 "asciicircum"
    95: 'underscore',       # ASCII 码 95（_）映射到 "underscore"
    96: 'quoteleft',        # ASCII 码 96（`）映射到 "quoteleft"
    123: 'braceleft',       # ASCII 码 123（{）映射到 "braceleft"
    124: 'bar',             # ASCII 码 124（|）映射到 "bar"
    125: 'braceright',      # ASCII 码 125（}）映射到 "braceright"
    126: 'asciitilde',      # ASCII 码 126（~）映射到 "asciitilde"
    161: 'exclamdown',      # ASCII 码 161（¡）映射到 "exclamdown"
    162: 'cent',            # ASCII 码 162（¢）映射到 "cent"
    163: 'sterling',        # ASCII 码 163（£）映射到 "sterling"
    164: 'fraction',        # ASCII 码 164（¤）映射到 "fraction"
    165: 'yen',             # ASCII 码 165（¥）映射到 "yen"
    166: 'florin',          # ASCII 码 166（ƒ）映射到 "florin"
    167: 'section',         # ASCII 码 167（§）映射到 "section"
    168: 'currency',        # ASCII 码 168（¨）映射到 "currency"
    169: 'quotesingle',     # ASCII 码 169（©）映射到 "quotesingle"
    170: 'quotedblleft',    # ASCII 码 170（ª）映射到 "quotedblleft"
    171: 'guillemotleft',   # ASCII 码 171（«）映射到 "guillemotleft"
    172: 'guilsinglleft',   # ASCII 码 172（¬）映射到 "guilsinglleft"
    173: 'guilsinglright',  # ASCII 码 173（­）映射到 "guilsinglright"
    174: 'fi',              # ASCII 码 174（®）映射到 "fi"
    175: 'fl',              # ASCII 码 175（¯）映射到 "fl"
    177: 'endash',          # ASCII 码 177（±）映射到 "endash"
    178: 'dagger',          # ASCII 码 178（²）映射到 "dagger"
    179: 'daggerdbl',       # ASCII 码 179（³）映射到 "daggerdbl"
    180: 'periodcentered',  # ASCII 码 180（·）映射到 "periodcentered"
    182: 'paragraph',       # ASCII 码 182（¶）映射到 "paragraph"
    183: 'bullet',          # ASCII 码 183（·）映射到 "bullet"
    184: 'quotesinglbase',  # ASCII 码 184（‚）映射到 "quotesinglbase"
    185: 'quotedblbase',    # ASCII 码 185（„）映射到 "quotedblbase"
    186: 'quotedblright',   # ASCII 码 186（”）映射到 "quotedblright"
    187: 'guillemotright',  # ASCII 码 187（»）映射到 "guillemotright"
    188: 'ellipsis',        # ASCII 码 188（…)）映射到 "ellipsis"
    189: 'perthousand',     # ASCII 码 189（‰）映射到 "perthousand"
    191: 'questiondown',    # ASCII 码 191（¿）映射到 "questiondown"
    193: 'grave',           # ASCII 码 193（Á）映射到 "grave"
    194: 'acute',           # ASCII 码 194（Â）映射到 "acute"
    195: 'circ
```