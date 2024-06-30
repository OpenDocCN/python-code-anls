# `D:\src\scipysrc\scipy\scipy\io\matlab\_byteordercodes.py`

```
''' Byteorder utilities for system - numpy byteorder encoding

Converts a variety of string codes for little endian, big endian,
native byte order and swapped byte order to explicit NumPy endian
codes - one of '<' (little endian) or '>' (big endian)

'''
import sys

# 导入系统模块 sys

__all__ = [
    'aliases', 'native_code', 'swapped_code',
    'sys_is_le', 'to_numpy_code'
]

# 定义 __all__ 列表，包含导出的变量名

sys_is_le = sys.byteorder == 'little'
# 检查系统字节顺序是否为小端，并赋值给 sys_is_le

native_code = sys_is_le and '<' or '>'
# 如果系统是小端，则 native_code 为 '<'，否则为 '>'

swapped_code = sys_is_le and '>' or '<'
# 如果系统是小端，则 swapped_code 为 '>'，否则为 '<'

aliases = {'little': ('little', '<', 'l', 'le'),
           'big': ('big', '>', 'b', 'be'),
           'native': ('native', '='),
           'swapped': ('swapped', 'S')}
# 定义别名字典，将不同的字节序描述映射到统一的表示方式

def to_numpy_code(code):
    """
    Convert various order codings to NumPy format.

    Parameters
    ----------
    code : str
        The code to convert. It is converted to lower case before parsing.
        Legal values are:
        'little', 'big', 'l', 'b', 'le', 'be', '<', '>', 'native', '=',
        'swapped', 's'.

    Returns
    -------
    out_code : {'<', '>'}
        Here '<' is the numpy dtype code for little endian,
        and '>' is the code for big endian.

    Examples
    --------
    >>> import sys
    >>> from scipy.io.matlab._byteordercodes import to_numpy_code
    >>> sys_is_le = (sys.byteorder == 'little')
    >>> sys_is_le
    True
    >>> to_numpy_code('big')
    '>'
    >>> to_numpy_code('little')
    '<'
    >>> nc = to_numpy_code('native')
    >>> nc == '<' if sys_is_le else nc == '>'
    True
    >>> sc = to_numpy_code('swapped')
    >>> sc == '>' if sys_is_le else sc == '<'
    True

    """
    code = code.lower()  # 将输入的代码转换为小写
    if code is None:
        return native_code  # 如果代码为空，则返回系统的本机代码
    if code in aliases['little']:
        return '<'  # 如果代码在小端别名中，则返回 '<'
    elif code in aliases['big']:
        return '>'  # 如果代码在大端别名中，则返回 '>'
    elif code in aliases['native']:
        return native_code  # 如果代码在本机别名中，则返回系统的本机代码
    elif code in aliases['swapped']:
        return swapped_code  # 如果代码在交换别名中，则返回系统的交换代码
    else:
        raise ValueError(
            'We cannot handle byte order %s' % code)  # 抛出值错误，表示无法处理给定的字节序代码
```