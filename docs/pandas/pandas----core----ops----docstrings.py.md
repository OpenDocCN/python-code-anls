# `D:\src\scipysrc\pandas\pandas\core\ops\docstrings.py`

```
"""
Templating for ops docstrings
"""

from __future__ import annotations

# 定义函数make_flex_doc，用于生成灵活的文档字符串，根据操作和类别类型进行适当的替换
def make_flex_doc(op_name: str, typ: str) -> str:
    """
    Make the appropriate substitutions for the given operation and class-typ
    into either _flex_doc_SERIES or _flex_doc_FRAME to return the docstring
    to attach to a generated method.

    Parameters
    ----------
    op_name : str {'__add__', '__sub__', ... '__eq__', '__ne__', ...}
        操作名称，例如'__add__', '__sub__'等
    typ : str {series, 'dataframe']}
        类别类型，可以是'series'或'dataframe'

    Returns
    -------
    doc : str
        生成的文档字符串
    """
    # 移除操作名称中的双下划线，例如'__add__'变为'add'
    op_name = op_name.replace("__", "")
    # 从_op_descriptions字典中获取操作描述
    op_desc = _op_descriptions[op_name]

    # 获取操作的描述和操作名
    op_desc_op = op_desc["op"]
    assert op_desc_op is not None  # for mypy
    # 如果操作名以'r'开头，构建相应的描述
    if op_name.startswith("r"):
        equiv = f"other {op_desc_op} {typ}"
    # 对于'divmod'操作，构建特定的描述
    elif op_name == "divmod":
        equiv = f"{op_name}({typ}, other)"
    else:
        equiv = f"{typ} {op_desc_op} other"

    # 如果类型为'series'
    if typ == "series":
        # 获取_series类型的基础文档
        base_doc = _flex_doc_SERIES
        # 如果操作支持反向，添加相应的反向参考信息
        if op_desc["reverse"]:
            base_doc += _see_also_reverse_SERIES.format(
                reverse=op_desc["reverse"], see_also_desc=op_desc["see_also_desc"]
            )
        # 格式化基础文档，插入操作描述、操作名、等价描述、series返回值描述等信息
        doc_no_examples = base_doc.format(
            desc=op_desc["desc"],
            op_name=op_name,
            equiv=equiv,
            series_returns=op_desc["series_returns"],
        )
        # 获取_series类型的操作示例
        ser_example = op_desc["series_examples"]
        # 如果有操作示例，将示例添加到文档末尾，否则保持文档不变
        if ser_example:
            doc = doc_no_examples + ser_example
        else:
            doc = doc_no_examples
    # 如果类型为'dataframe'
    elif typ == "dataframe":
        # 如果操作名在['eq', 'ne', 'le', 'lt', 'ge', 'gt']中，使用_flex_comp_doc_FRAME作为基础文档
        if op_name in ["eq", "ne", "le", "lt", "ge", "gt"]:
            base_doc = _flex_comp_doc_FRAME
            # 格式化_flex_comp_doc_FRAME文档，插入操作名和操作描述
            doc = _flex_comp_doc_FRAME.format(
                op_name=op_name,
                desc=op_desc["desc"],
            )
        # 否则使用_flex_doc_FRAME作为基础文档
        else:
            base_doc = _flex_doc_FRAME
            # 格式化_flex_doc_FRAME文档，插入操作描述、操作名、等价描述、反向描述等信息
            doc = base_doc.format(
                desc=op_desc["desc"],
                op_name=op_name,
                equiv=equiv,
                reverse=op_desc["reverse"],
            )
    # 如果类型参数既不是'series'也不是'dataframe'，引发断言错误
    else:
        raise AssertionError("Invalid typ argument.")
    # 返回生成的文档字符串
    return doc


# _common_examples_algebra_SERIES：包含了_series类型的通用代数操作示例的字符串
_common_examples_algebra_SERIES = """
Examples
--------
>>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
>>> a
a    1.0
b    1.0
c    1.0
d    NaN
dtype: float64
>>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
>>> b
a    1.0
b    NaN
d    1.0
e    NaN
dtype: float64"""

# _common_examples_comparison_SERIES：包含了_series类型的通用比较操作示例的字符串
_common_examples_comparison_SERIES = """
Examples
--------
>>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
>>> a
a    1.0
b    1.0
c    1.0
d    NaN
e    1.0
dtype: float64
>>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
>>> b
a    0.0
b    1.0
c    2.0
d    NaN
f    1.0
dtype: float64"""

# _add_example_SERIES：包含了_series类型的加法操作示例的字符串
_add_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.add(b, fill_value=0)
a    2.0
b    1.0
c    1.0
d    1.0
e    NaN
dtype: float64
"""
)

# _sub_example_SERIES：包含了_series类型的减法操作示例的字符串
_sub_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.subtract(b, fill_value=0)
a    0.0
"""
)
# Series 对象的乘法示例，使用 fill_value=0 处理缺失值
_mul_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.multiply(b, fill_value=0)
a    1.0
b    0.0
c    0.0
d    0.0
e    NaN
dtype: float64
"""
)

# Series 对象的除法示例，使用 fill_value=0 处理缺失值
_div_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.divide(b, fill_value=0)
a    1.0
b    inf
c    inf
d    0.0
e    NaN
dtype: float64
"""
)

# Series 对象的整除示例，使用 fill_value=0 处理缺失值
_floordiv_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.floordiv(b, fill_value=0)
a    1.0
b    inf
c    inf
d    0.0
e    NaN
dtype: float64
"""
)

# Series 对象的 divmod 示例，使用 fill_value=0 处理缺失值
_divmod_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.divmod(b, fill_value=0)
(a    1.0
 b    inf
 c    inf
 d    0.0
 e    NaN
 dtype: float64,
 a    0.0
 b    NaN
 c    NaN
 d    0.0
 e    NaN
 dtype: float64)
"""
)

# Series 对象的取模示例，使用 fill_value=0 处理缺失值
_mod_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.mod(b, fill_value=0)
a    0.0
b    NaN
c    NaN
d    0.0
e    NaN
dtype: float64
"""
)

# Series 对象的乘方示例，使用 fill_value=0 处理缺失值
_pow_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.pow(b, fill_value=0)
a    1.0
b    1.0
c    1.0
d    0.0
e    NaN
dtype: float64
"""
)

# Series 对象的不等于比较示例，使用 fill_value=0 处理缺失值
_ne_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.ne(b, fill_value=0)
a    False
b     True
c     True
d     True
e     True
dtype: bool
"""
)

# Series 对象的等于比较示例，使用 fill_value=0 处理缺失值
_eq_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.eq(b, fill_value=0)
a     True
b    False
c    False
d    False
e    False
dtype: bool
"""
)

# Series 对象的小于比较示例，使用 fill_value=0 处理缺失值
_lt_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.lt(b, fill_value=0)
a    False
b    False
c     True
d    False
e    False
f     True
dtype: bool
"""
)

# Series 对象的小于等于比较示例，使用 fill_value=0 处理缺失值
_le_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.le(b, fill_value=0)
a    False
b     True
c     True
d    False
e    False
f     True
dtype: bool
"""
)

# Series 对象的大于比较示例，使用 fill_value=0 处理缺失值
_gt_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.gt(b, fill_value=0)
a     True
b    False
c    False
d    False
e     True
f    False
dtype: bool
"""
)

# Series 对象的大于等于比较示例，使用 fill_value=0 处理缺失值
_ge_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.ge(b, fill_value=0)
a     True
b     True
c    False
d    False
e     True
f    False
dtype: bool
"""
)

# 描述返回结果为单个 Series 对象的字符串
_returns_series = """Series\n    The result of the operation."""

# 描述返回结果为一个包含两个 Series 对象的元组的字符串
_returns_tuple = """2-Tuple of Series\n    The result of the operation."""

# 包含各种操作的描述信息和示例的字典，每个操作包括运算符、描述、反向运算符、Series 示例和返回值说明
_op_descriptions: dict[str, dict[str, str | None]] = {
    # 算术运算符
    "add": {
        "op": "+",
        "desc": "Addition",
        "reverse": "radd",
        "series_examples": _add_example_SERIES,
        "series_returns": _returns_series,
    },
    "sub": {
        "op": "-",
        "desc": "Subtraction",
        "reverse": "rsub",
        "series_examples": _sub_example_SERIES,
        "series_returns": _returns_series,
    },
    "mul": {
        "op": "*",
        "desc": "Multiplication",
        "reverse": "rmul",
        "series_examples": _mul_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },  # 定义乘法操作，包括操作符、描述、反向操作、示例和返回值序列，没有 DataFrame 示例

    "mod": {
        "op": "%",
        "desc": "Modulo",
        "reverse": "rmod",
        "series_examples": _mod_example_SERIES,
        "series_returns": _returns_series,
    },  # 定义取模操作，包括操作符、描述、反向操作、示例和返回值序列

    "pow": {
        "op": "**",
        "desc": "Exponential power",
        "reverse": "rpow",
        "series_examples": _pow_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },  # 定义指数幂操作，包括操作符、描述、反向操作、示例和返回值序列，没有 DataFrame 示例

    "truediv": {
        "op": "/",
        "desc": "Floating division",
        "reverse": "rtruediv",
        "series_examples": _div_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },  # 定义浮点数除法操作，包括操作符、描述、反向操作、示例和返回值序列，没有 DataFrame 示例

    "floordiv": {
        "op": "//",
        "desc": "Integer division",
        "reverse": "rfloordiv",
        "series_examples": _floordiv_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },  # 定义整数除法操作，包括操作符、描述、反向操作、示例和返回值序列，没有 DataFrame 示例

    "divmod": {
        "op": "divmod",
        "desc": "Integer division and modulo",
        "reverse": "rdivmod",
        "series_examples": _divmod_example_SERIES,
        "series_returns": _returns_tuple,
        "df_examples": None,
    },  # 定义整数除法和取模操作，包括操作符、描述、反向操作、示例和返回值元组，没有 DataFrame 示例

    # Comparison Operators
    "eq": {
        "op": "==",
        "desc": "Equal to",
        "reverse": None,
        "series_examples": _eq_example_SERIES,
        "series_returns": _returns_series,
    },  # 定义相等比较操作，包括操作符、描述、示例和返回值序列，没有反向操作

    "ne": {
        "op": "!=",
        "desc": "Not equal to",
        "reverse": None,
        "series_examples": _ne_example_SERIES,
        "series_returns": _returns_series,
    },  # 定义不等比较操作，包括操作符、描述、示例和返回值序列，没有反向操作

    "lt": {
        "op": "<",
        "desc": "Less than",
        "reverse": None,
        "series_examples": _lt_example_SERIES,
        "series_returns": _returns_series,
    },  # 定义小于比较操作，包括操作符、描述、示例和返回值序列，没有反向操作

    "le": {
        "op": "<=",
        "desc": "Less than or equal to",
        "reverse": None,
        "series_examples": _le_example_SERIES,
        "series_returns": _returns_series,
    },  # 定义小于等于比较操作，包括操作符、描述、示例和返回值序列，没有反向操作

    "gt": {
        "op": ">",
        "desc": "Greater than",
        "reverse": None,
        "series_examples": _gt_example_SERIES,
        "series_returns": _returns_series,
    },  # 定义大于比较操作，包括操作符、描述、示例和返回值序列，没有反向操作

    "ge": {
        "op": ">=",
        "desc": "Greater than or equal to",
        "reverse": None,
        "series_examples": _ge_example_SERIES,
        "series_returns": _returns_series,
    },  # 定义大于等于比较操作，包括操作符、描述、示例和返回值序列，没有反向操作
# 反向操作名列表，包含所有操作的名称
_op_names = list(_op_descriptions.keys())

# 遍历操作名列表
for key in _op_names:
    # 获取当前操作的反向操作名
    reverse_op = _op_descriptions[key]["reverse"]
    
    # 如果存在反向操作名
    if reverse_op is not None:
        # 复制当前操作的描述信息到反向操作中
        _op_descriptions[reverse_op] = _op_descriptions[key].copy()
        
        # 设置反向操作的反向操作为当前操作名
        _op_descriptions[reverse_op]["reverse"] = key
        
        # 设置当前操作的“see_also_desc”描述信息，指向反向操作的反向描述
        _op_descriptions[key]["see_also_desc"] = (
            f"Reverse of the {_op_descriptions[key]['desc']} operator, {_py_num_ref}"
        )
        
        # 设置反向操作的“see_also_desc”描述信息，指向当前操作的描述
        _op_descriptions[reverse_op]["see_also_desc"] = (
            f"Element-wise {_op_descriptions[key]['desc']}, {_py_num_ref}"
        )

# 灵活文档字符串模板，用于描述 Series 的灵活操作
_flex_doc_SERIES = """
Return {desc} of series and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``, but with support to substitute a fill_value for
missing data in either one of the inputs.

Parameters
----------
other : Series or scalar value
    The second operand in this operation.
level : int or name
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.
fill_value : None or float value, default None (NaN)
    Fill existing missing (NaN) values, and any new element needed for
    successful Series alignment, with this value before computation.
    If data in both corresponding Series locations is missing
    the result of filling (at that location) will be missing.
axis : {{0 or 'index'}}
    Unused. Parameter needed for compatibility with DataFrame.

Returns
-------
{series_returns}
"""

# 反向操作的灵活文档字符串模板，用于描述 Series 的反向操作
_see_also_reverse_SERIES = """
See Also
--------
Series.{reverse} : {see_also_desc}.
"""

# 灵活文档字符串模板，用于描述 DataFrame 的灵活操作
_flex_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``, but with support to substitute a fill_value
for missing data in one of the inputs. With reverse version, `{reverse}`.

Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

Parameters
----------
other : scalar, sequence, Series, dict or DataFrame
    Any single or multiple element data structure, or list-like object.
axis : {{0 or 'index', 1 or 'columns'}}
    Whether to compare by the index (0 or 'index') or columns.
    (1 or 'columns'). For Series input, axis to match Series index on.
level : int or label
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.
fill_value : float or None, default None
    Fill existing missing (NaN) values, and any new element needed for
    successful DataFrame alignment, with this value before computation.
    If data in both corresponding DataFrame locations is missing
    the result will be missing.

Returns
-------
DataFrame
    Result of the arithmetic operation.

See Also
--------
DataFrame.add : Add DataFrames.
DataFrame.sub : Subtract DataFrames.
DataFrame.mul : Multiply DataFrames.

"""
# Divide DataFrames element-wise (float division).
DataFrame.div : Divide DataFrames (float division).

# Divide DataFrames element-wise (float division).
DataFrame.truediv : Divide DataFrames (float division).

# Divide DataFrames element-wise (integer division).
DataFrame.floordiv : Divide DataFrames (integer division).

# Calculate modulo (remainder after division).
DataFrame.mod : Calculate modulo (remainder after division).

# Calculate exponential power.
DataFrame.pow : Calculate exponential power.

# Mismatched indices will be unioned together.

# Example DataFrame creation with specified indices and columns.
>>> df = pd.DataFrame({'angles': [0, 3, 4],
...                    'degrees': [360, 180, 360]},
...                   index=['circle', 'triangle', 'rectangle'])

# Display the created DataFrame.
>>> df
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

# Example of adding a scalar value (1) to the DataFrame using operator version.
>>> df + 1
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

# Example of adding a scalar value (1) to the DataFrame using method add().
>>> df.add(1)
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

# Example of dividing each element in the DataFrame by a constant (10).
>>> df.div(10)
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

# Example of dividing a constant (10) by each element in the DataFrame (reverse division).
>>> df.rdiv(10)
             angles   degrees
circle          inf  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

# Example of subtracting a list ([1, 2]) from each column in the DataFrame using operator version.
>>> df - [1, 2]
           angles  degrees
circle         -1      358
triangle        2      178
rectangle       3      358

# Example of subtracting a list ([1, 2]) from each column in the DataFrame using method sub() with axis specified.
>>> df.sub([1, 2], axis='columns')
           angles  degrees
circle         -1      358
triangle        2      178
rectangle       3      358

# Example of subtracting a Series from each row in the DataFrame using method sub() with axis specified.
>>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
...        axis='index')
           angles  degrees
circle         -1      359
triangle        2      179
rectangle       3      359

# Example of multiplying each element in the DataFrame by a dictionary (element-wise multiplication).
>>> df.mul({'angles': 0, 'degrees': 2})
            angles  degrees
circle           0      720
triangle         0      360
rectangle        0      720

# Example of multiplying each element in the DataFrame by a dictionary with axis specified (row-wise multiplication).
>>> df.mul({'circle': 0, 'triangle': 2, 'rectangle': 3}, axis='index')
            angles  degrees
circle           0        0
triangle         6      360
rectangle       12     1080

# Example of multiplying each element in the DataFrame by a DataFrame of different shape using operator version.
>>> other = pd.DataFrame({'angles': [0, 3, 4]},
...                      index=['circle', 'triangle', 'rectangle'])

# Display the created DataFrame.
>>> other
           angles
circle          0
triangle        3
rectangle       4

# Perform element-wise multiplication between the original DataFrame and 'other' DataFrame.
>>> df * other
           angles  degrees
circle          0      NaN
triangle        9      NaN
rectangle      16      NaN

# Perform element-wise multiplication with 'other' DataFrame, filling NaN values with 0.
>>> df.mul(other, fill_value=0)
           angles  degrees
circle          0      0.0
triangle        9      0.0
rectangle      16      0.0

# Example of dividing each element in the DataFrame by a MultiIndex DataFrame by level.
>>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
...                              'degrees': [360, 180, 360, 360, 540, 720]},
...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
# 创建一个包含多级索引的 DataFrame，其中包括 'angles' 和 'degrees' 两列
>>> df_multindex = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6],
...                              'degrees': [360, 180, 360, 360, 540, 720]},
...                             index=pd.MultiIndex.from_arrays(
...                                 [['A', 'A', 'A', 'B', 'B', 'B'],
...                                  ['circle', 'triangle', 'rectangle',
...                                   'square', 'pentagon', 'hexagon']])
>>> df_multindex
             angles  degrees
A circle          0      360
  triangle        3      180
  rectangle       4      360
B square          4      360
  pentagon        5      540
  hexagon         6      720

# 将原 DataFrame df 按照指定的多级索引级别进行元素级除法运算，缺失值填充为 0
>>> df.div(df_multindex, level=1, fill_value=0)
             angles  degrees
A circle        NaN      1.0
  triangle      1.0      1.0
  rectangle     1.0      1.0
B square        0.0      0.0
  pentagon      0.0      0.0
  hexagon       0.0      0.0

# 创建一个包含多级索引的 DataFrame，包含 'A' 和 'B' 两列，每列包含一组数字列表
>>> df_pow = pd.DataFrame({'A': [2, 3, 4, 5],
...                        'B': [6, 7, 8, 9]})
>>> df_pow.pow(2)
    A   B
0   4  36
1   9  49
2  16  64
3  25  81


_flex_comp_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).

Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
operators.

Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
(rows or columns) and level for comparison.

Parameters
----------
other : scalar, sequence, Series, or DataFrame
    Any single or multiple element data structure, or list-like object.
axis : {{0 or 'index', 1 or 'columns'}}, default 'columns'
    Whether to compare by the index (0 or 'index') or columns
    (1 or 'columns').
level : int or label
    Broadcast across a level, matching Index values on the passed
    MultiIndex level.

Returns
-------
DataFrame of bool
    Result of the comparison.

See Also
--------
DataFrame.eq : Compare DataFrames for equality elementwise.
DataFrame.ne : Compare DataFrames for inequality elementwise.
DataFrame.le : Compare DataFrames for less than inequality
    or equality elementwise.
DataFrame.lt : Compare DataFrames for strictly less than
    inequality elementwise.
DataFrame.ge : Compare DataFrames for greater than inequality
    or equality elementwise.
DataFrame.gt : Compare DataFrames for strictly greater than
    inequality elementwise.

Notes
-----
Mismatched indices will be unioned together.
`NaN` values are considered different (i.e. `NaN` != `NaN`).

Examples
--------
>>> df = pd.DataFrame({{'cost': [250, 150, 100],
...                    'revenue': [100, 250, 300]}},
...                   index=['A', 'B', 'C'])
>>> df
   cost  revenue
A   250      100
B   150      250
C   100      300

Comparison with a scalar, using either the operator or method:

>>> df == 100
    cost  revenue
A  False     True
B  False    False
C   True    False

>>> df.eq(100)
    cost  revenue
A  False     True
B  False    False

When `other` is a :class:`Series`, the columns of a DataFrame are aligned
with the index of `other` and broadcast:

>>> df != pd.Series([100, 250], index=["cost", "revenue"])
    cost  revenue
A   True     True
B   True    False
C  False     True

Use the method to control the broadcast axis:
# 使用 `ne` 方法对 DataFrame 进行逐元素比较，返回布尔值 DataFrame。
# 在指定轴向进行比较，此处为行索引（index），与给定的 Series 对象进行比较。
>>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis='index')
   cost  revenue
A  True    False  # 行 A：cost 不等于 100，revenue 不等于 300
B  True     True  # 行 B：cost 不等于 100，revenue 不等于 300
C  True     True  # 行 C：cost 不等于 100，revenue 不等于 300
D  True     True  # 行 D：cost 不等于 100，revenue 不等于 300

# 当与任意序列进行比较时，必须保证列数与 `other` 中元素的数量相匹配。
>>> df == [250, 100]
    cost  revenue
A   True     True  # 行 A：cost 等于 250，revenue 等于 100
B  False    False  # 行 B：cost 不等于 250，revenue 不等于 100
C  False    False  # 行 C：cost 不等于 250，revenue 不等于 100

# 使用 `eq` 方法在指定轴向上进行比较。
>>> df.eq([250, 250, 100], axis='index')
    cost  revenue
A   True    False  # 行 A：cost 等于 250，revenue 不等于 250
B  False     True  # 行 B：cost 不等于 250，revenue 等于 250
C   True    False  # 行 C：cost 等于 100，revenue 不等于 100

# 与形状不同的 DataFrame 进行比较。
>>> other = pd.DataFrame({'revenue': [300, 250, 100, 150]}, index=['A', 'B', 'C', 'D'])
>>> other
   revenue
A      300
B      250
C      100
D      150

>>> df.gt(other)
    cost  revenue
A  False    False  # 行 A：cost 小于等于 250，revenue 小于等于 300
B  False    False  # 行 B：cost 小于等于 250，revenue 小于等于 250
C  False     True  # 行 C：cost 小于等于 250，revenue 大于 100
D  False    False  # 行 D：cost 小于等于 250，revenue 小于等于 150

# 对多级索引按级别进行比较。
>>> df_multindex = pd.DataFrame({'cost': [250, 150, 100, 150, 300, 220],
...                              'revenue': [100, 250, 300, 200, 175, 225]},
...                             index=[['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2'],
...                                    ['A', 'B', 'C', 'A', 'B', 'C']])
>>> df_multindex
      cost  revenue
Q1 A   250      100
   B   150      250
   C   100      300
Q2 A   150      200
   B   300      175
   C   220      225

>>> df.le(df_multindex, level=1)
       cost  revenue
Q1 A   True     True  # Q1 组的 A 行：cost 小于等于 250，revenue 小于等于 100
   B   True     True  # Q1 组的 B 行：cost 小于等于 150，revenue 小于等于 250
   C   True     True  # Q1 组的 C 行：cost 小于等于 100，revenue 小于等于 300
Q2 A  False     True  # Q2 组的 A 行：cost 小于等于 150，revenue 小于等于 200
   B   True    False  # Q2 组的 B 行：cost 小于等于 300，revenue 不小于等于 175
   C   True    False  # Q2 组的 C 行：cost 小于等于 220，revenue 不小于等于 225
```