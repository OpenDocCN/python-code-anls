# `.\numpy\numpy\_typing\_add_docstring.py`

```py
"""A module for creating docstrings for sphinx ``data`` domains."""

# 导入正则表达式和文本包装模块
import re
import textwrap

# 导入 NDArray 类型
from ._array_like import NDArray

# 全局变量，用于存储文档字符串的列表
_docstrings_list = []


def add_newdoc(name: str, value: str, doc: str) -> None:
    """Append ``_docstrings_list`` with a docstring for `name`.

    Parameters
    ----------
    name : str
        The name of the object.
    value : str
        A string-representation of the object.
    doc : str
        The docstring of the object.

    """
    # 将 name, value, doc 组成的元组添加到 _docstrings_list 中
    _docstrings_list.append((name, value, doc))


def _parse_docstrings() -> str:
    """Convert all docstrings in ``_docstrings_list`` into a single
    sphinx-legible text block.

    """
    # 初始化空列表，用于存储转换后的文本块
    type_list_ret = []
    
    # 遍历 _docstrings_list 中的每个元组 (name, value, doc)
    for name, value, doc in _docstrings_list:
        # 根据 docstring 的缩进，去除缩进并替换换行符
        s = textwrap.dedent(doc).replace("\n", "\n    ")

        # 将文本按行分割
        lines = s.split("\n")
        new_lines = []
        indent = ""

        # 遍历每一行文本
        for line in lines:
            # 使用正则表达式匹配 Rubric 或 Admonition 标记的文本行
            m = re.match(r'^(\s+)[-=]+\s*$', line)
            if m and new_lines:
                # 如果匹配成功且存在前一行，则进行相应的转换
                prev = textwrap.dedent(new_lines.pop())
                if prev == "Examples":
                    indent = ""
                    new_lines.append(f'{m.group(1)}.. rubric:: {prev}')
                else:
                    indent = 4 * " "
                    new_lines.append(f'{m.group(1)}.. admonition:: {prev}')
                new_lines.append("")
            else:
                new_lines.append(f"{indent}{line}")

        # 重新组合处理后的行文本
        s = "\n".join(new_lines)

        # 构建最终的 Sphinx 数据域文本块
        s_block = f""".. data:: {name}\n    :value: {value}\n    {s}"""
        type_list_ret.append(s_block)
    
    # 返回所有文本块组成的单个字符串
    return "\n".join(type_list_ret)


# 示例添加两个文档字符串到 _docstrings_list 中
add_newdoc('ArrayLike', 'typing.Union[...]',
    """
    A `~typing.Union` representing objects that can be coerced
    into an `~numpy.ndarray`.

    Among others this includes the likes of:

    * Scalars.
    * (Nested) sequences.
    * Objects implementing the `~class.__array__` protocol.

    .. versionadded:: 1.20

    See Also
    --------
    :term:`array_like`:
        Any scalar or sequence that can be interpreted as an ndarray.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> def as_array(a: npt.ArrayLike) -> np.ndarray:
        ...     return np.array(a)

    """)

add_newdoc('DTypeLike', 'typing.Union[...]',
    """
    A `~typing.Union` representing objects that can be coerced
    into a `~numpy.dtype`.

    Among others this includes the likes of:

    * :class:`type` objects.
    * Character codes or the names of :class:`type` objects.
    * Objects with the ``.dtype`` attribute.

    .. versionadded:: 1.20

    See Also
    --------
    :ref:`Specifying and constructing data types <arrays.dtypes.constructing>`
        A comprehensive overview of all objects that can be coerced
        into data types.

    Examples
    --------

    """
    # 导入 NumPy 库和 NumPy 的类型提示模块
    import numpy as np
    import numpy.typing as npt
    
    # 定义一个函数 `as_dtype`，用于将输入参数 `d` 转换为 NumPy 的数据类型 (`np.dtype`)
    def as_dtype(d: npt.DTypeLike) -> np.dtype:
        # 返回参数 `d` 对应的 NumPy 数据类型对象 (`np.dtype`)
        return np.dtype(d)
add_newdoc('NDArray', repr(NDArray),
    """
    定义一个 `np.ndarray[Any, np.dtype[+ScalarType]] <numpy.ndarray>` 类型的别名，
    其中 `dtype.type <numpy.dtype.type>` 是关于泛型类型的概念。

    可以在运行时用于定义具有给定 dtype 和未指定形状的数组。

    .. versionadded:: 1.21

    示例
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> print(npt.NDArray)
        numpy.ndarray[typing.Any, numpy.dtype[+_ScalarType_co]]

        >>> print(npt.NDArray[np.float64])
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]

        >>> NDArrayInt = npt.NDArray[np.int_]
        >>> a: NDArrayInt = np.arange(10)

        >>> def func(a: npt.ArrayLike) -> npt.NDArray[Any]:
        ...     return np.array(a)

    """
)

_docstrings = _parse_docstrings()
```