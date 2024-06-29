# `D:\src\scipysrc\pandas\pandas\io\xml.py`

```
"""
:mod:``pandas.io.xml`` is a module for reading XML.
"""

# 导入必要的模块和类
from __future__ import annotations

import io  # 导入用于处理文件流的模块
from os import PathLike  # 导入 PathLike，用于表示路径类的抽象基类
from typing import (  # 导入类型提示相关的模块
    TYPE_CHECKING,
    Any,
)

from pandas._libs import lib  # 导入 pandas 内部的 C 库
from pandas.compat._optional import import_optional_dependency  # 导入可选依赖项导入函数
from pandas.errors import (  # 导入异常类
    AbstractMethodError,
    ParserError,
)
from pandas.util._decorators import doc  # 导入用于文档字符串装饰器的模块
from pandas.util._validators import check_dtype_backend  # 导入检查数据类型后端的函数

from pandas.core.dtypes.common import is_list_like  # 导入检查对象是否可迭代的函数

from pandas.core.shared_docs import _shared_docs  # 导入共享文档字符串模块

from pandas.io.common import (  # 导入常用 I/O 函数
    get_handle,
    infer_compression,
    is_fsspec_url,
    is_url,
    stringify_path,
)
from pandas.io.parsers import TextParser  # 导入文本解析器类

if TYPE_CHECKING:
    from collections.abc import (  # 导入用于类型检查的抽象基类
        Callable,
        Sequence,
    )
    from xml.etree.ElementTree import Element  # 导入 ElementTree 中的 Element 类

    from lxml import etree  # 导入 lxml 库中的 etree 模块

    from pandas._typing import (  # 导入 pandas 中的类型提示
        CompressionOptions,
        ConvertersArg,
        DtypeArg,
        DtypeBackend,
        FilePath,
        ParseDatesArg,
        ReadBuffer,
        StorageOptions,
        XMLParsers,
    )

    from pandas import DataFrame  # 导入 DataFrame 类


@doc(
    storage_options=_shared_docs["storage_options"],  # 使用共享文档中的 storage_options 部分
    decompression_options=_shared_docs["decompression_options"] % "path_or_buffer",  # 使用共享文档中的 decompression_options 部分
)
class _XMLFrameParser:
    """
    Internal subclass to parse XML into DataFrames.

    Parameters
    ----------
    path_or_buffer : a valid JSON ``str``, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file.

    xpath : str or regex
        The ``XPath`` expression to parse required set of nodes for
        migration to :class:`~pandas.DataFrame`. ``etree`` supports limited ``XPath``.

    namespaces : dict
        The namespaces defined in XML document (``xmlns:namespace='URI'``)
        as dicts with key being namespace and value the URI.

    elems_only : bool
        Parse only the child elements at the specified ``xpath``.

    attrs_only : bool
        Parse only the attributes at the specified ``xpath``.

    names : list
        Column names for :class:`~pandas.DataFrame` of parsed XML data.

    dtype : dict
        Data type for data or columns. E.g. {{'a': np.float64,
        'b': np.int32, 'c': 'Int64'}}

        .. versionadded:: 1.5.0

    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels.

        .. versionadded:: 1.5.0

    parse_dates : bool or list of int or names or list of lists or dict
        Converts either index or select columns to datetimes

        .. versionadded:: 1.5.0

    encoding : str
        Encoding of xml object or document.

    stylesheet : str or file-like
        URL, file, file-like object, or a raw string containing XSLT,
        ``etree`` does not support XSLT but retained for consistency.
    """
    # 内部子类，用于将 XML 解析为 DataFrame

    def __init__(self, path_or_buffer, xpath, namespaces, elems_only, attrs_only, names, dtype, converters, parse_dates, encoding, stylesheet):
        self.path_or_buffer = path_or_buffer  # 初始化路径或缓冲区参数
        self.xpath = xpath  # 初始化 XPath 表达式参数
        self.namespaces = namespaces  # 初始化 XML 命名空间参数
        self.elems_only = elems_only  # 初始化是否仅解析元素参数
        self.attrs_only = attrs_only  # 初始化是否仅解析属性参数
        self.names = names  # 初始化列名参数
        self.dtype = dtype  # 初始化数据类型参数
        self.converters = converters  # 初始化值转换器参数
        self.parse_dates = parse_dates  # 初始化日期解析参数
        self.encoding = encoding  # 初始化编码参数
        self.stylesheet = stylesheet  # 初始化样式表参数
    """
    iterparse : dict, optional
        Dict with row element as key and list of descendant elements
        and/or attributes as value to be retrieved in iterparsing of
        XML document.

        .. versionadded:: 1.5.0

    {decompression_options}
        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    See also
    --------
    pandas.io.xml._EtreeFrameParser
    pandas.io.xml._LxmlFrameParser

    Notes
    -----
    To subclass this class effectively you must override the following methods:`
        * :func:`parse_data`
        * :func:`_parse_nodes`
        * :func:`_iterparse_nodes`
        * :func:`_parse_doc`
        * :func:`_validate_names`
        * :func:`_validate_path`

    See each method's respective documentation for details on their
    functionality.
    """

class XMLParser:
    def __init__(
        self,
        path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
        xpath: str,
        namespaces: dict[str, str] | None,
        elems_only: bool,
        attrs_only: bool,
        names: Sequence[str] | None,
        dtype: DtypeArg | None,
        converters: ConvertersArg | None,
        parse_dates: ParseDatesArg | None,
        encoding: str | None,
        stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None,
        iterparse: dict[str, list[str]] | None,
        compression: CompressionOptions,
        storage_options: StorageOptions,
    ) -> None:
        self.path_or_buffer = path_or_buffer  # 存储文件路径或缓冲区对象
        self.xpath = xpath  # 存储用于XPath查询的表达式
        self.namespaces = namespaces  # 存储命名空间字典
        self.elems_only = elems_only  # 仅解析元素节点的标志
        self.attrs_only = attrs_only  # 仅解析属性节点的标志
        self.names = names  # 要解析的节点名称列表
        self.dtype = dtype  # 数据类型
        self.converters = converters  # 数据转换器
        self.parse_dates = parse_dates  # 解析日期的标志
        self.encoding = encoding  # 文件编码方式
        self.stylesheet = stylesheet  # 样式表文件路径或缓冲区对象
        self.iterparse = iterparse  # 迭代解析配置字典
        self.compression: CompressionOptions = compression  # 压缩选项对象
        self.storage_options = storage_options  # 存储选项对象

    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, parse and return specific nodes.
        """

        raise AbstractMethodError(self)
    def _parse_nodes(self, elems: list[Any]) -> list[dict[str, str | None]]:
        """
        Parse xml nodes.

        This method will parse the children and attributes of elements
        in ``xpath``, conditionally for only elements, only attributes
        or both while optionally renaming node names.

        Raises
        ------
        ValueError
            * If only elements and only attributes are specified.

        Notes
        -----
        Namespace URIs will be removed from return node values. Also,
        elements with missing children or attributes compared to siblings
        will have optional keys filled with None values.
        """

        # Initialize list of dictionaries to store parsed XML node data
        dicts: list[dict[str, str | None]]

        # Check if both elems_only and attrs_only flags are True, raise ValueError if so
        if self.elems_only and self.attrs_only:
            raise ValueError("Either element or attributes can be parsed not both.")

        # Parse XML elements only if elems_only flag is True
        if self.elems_only:
            # If renaming of node names is specified
            if self.names:
                dicts = [
                    {
                        **(
                            {el.tag: el.text}  # Include element tag as key and text content as value if present
                            if el.text and not el.text.isspace()
                            else {}  # If no text content or only whitespace, include empty dictionary
                        ),
                        **{
                            nm: ch.text if ch.text else None  # Include child tag as key and text content as value, or None if no text
                            for nm, ch in zip(self.names, el.findall("*"))  # Iterate over renamed names and child elements
                        },
                    }
                    for el in elems  # Iterate over each XML element in elems list
                ]
            else:
                # If renaming is not specified, include only child elements' tag and text content
                dicts = [
                    {ch.tag: ch.text if ch.text else None for ch in el.findall("*")}
                    for el in elems
                ]

        # Parse XML attributes only if attrs_only flag is True
        elif self.attrs_only:
            dicts = [
                {k: v if v else None for k, v in el.attrib.items()}  # Include attribute key and value, or None if value is empty
                for el in elems  # Iterate over each XML element in elems list
            ]

        # Parse both elements and attributes, optionally renaming node names
        elif self.names:
            dicts = [
                {
                    **el.attrib,  # Include all attributes of the element
                    **({el.tag: el.text} if el.text and not el.text.isspace() else {}),  # Include element tag and text content if present
                    **{
                        nm: ch.text if ch.text else None  # Include renamed child tag and text content, or None if no text
                        for nm, ch in zip(self.names, el.findall("*"))  # Iterate over renamed names and child elements
                    },
                }
                for el in elems  # Iterate over each XML element in elems list
            ]

        # Parse both elements and attributes with original node names
        else:
            dicts = [
                {
                    **el.attrib,  # Include all attributes of the element
                    **({el.tag: el.text} if el.text and not el.text.isspace() else {}),  # Include element tag and text content if present
                    **{ch.tag: ch.text if ch.text else None for ch in el.findall("*")},  # Include child tag and text content, or None if no text
                }
                for el in elems  # Iterate over each XML element in elems list
            ]

        # Remove namespace URIs from dictionary keys, if present
        dicts = [
            {k.split("}")[1] if "}" in k else k: v for k, v in d.items()} for d in dicts
        ]

        # Get unique keys from all dictionaries in dicts list
        keys = list(dict.fromkeys([k for d in dicts for k in d.keys()]))

        # Ensure each dictionary in dicts has all keys present, fill with None if missing
        dicts = [{k: d[k] if k in d.keys() else None for k in keys} for d in dicts]

        # If renaming is specified, create dictionaries with keys as specified names
        if self.names:
            dicts = [dict(zip(self.names, d.values())) for d in dicts]

        # Return the list of dictionaries containing parsed XML node data
        return dicts
    def _validate_path(self) -> list[Any]:
        """
        Validate ``xpath``.

        This method checks the validity of the XPath expression (`xpath`).
        It ensures the syntax is correct and verifies that the evaluation
        of the XPath expression returns non-empty nodes.

        Raises
        ------
        SyntaxError
            * If `xpath` syntax is invalid or issues with namespaces.

        ValueError
            * If `xpath` evaluation does not return any nodes.
        """

        raise AbstractMethodError(self)

    def _validate_names(self) -> None:
        """
        Validate names.

        This method validates the `names` attribute.
        It checks if `names` is a list-like object and ensures it has
        the same length as the parsed nodes.

        Raises
        ------
        ValueError
            * If `names` is not a list or its length is less than the number of nodes.
        """
        raise AbstractMethodError(self)

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Element | etree._Element:
        """
        Build tree from path_or_buffer.

        This method parses the XML document from the `raw_doc` parameter,
        which can be a file path, bytes buffer, or string buffer, into
        an XML tree structure.

        Returns
        -------
        Element or etree._Element
            The root element of the parsed XML tree.
        """
        raise AbstractMethodError(self)
class _EtreeFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into DataFrames with the Python
    standard library XML module: `xml.etree.ElementTree`.
    """

    def parse_data(self) -> list[dict[str, str | None]]:
        # 导入 iterparse 函数
        from xml.etree.ElementTree import iterparse

        # 如果有样式表，抛出数值错误
        if self.stylesheet is not None:
            raise ValueError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )

        # 如果没有启用 iterparse，解析文档并验证路径
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)
            elems = self._validate_path()

        # 验证节点名称的有效性
        self._validate_names()

        # 解析节点并返回 XML 字典列表
        xml_dicts: list[dict[str, str | None]] = (
            self._parse_nodes(elems)
            if self.iterparse is None
            else self._iterparse_nodes(iterparse)
        )

        return xml_dicts

    def _validate_path(self) -> list[Any]:
        """
        Notes
        -----
        ``etree`` supports limited ``XPath``. If user attempts a more complex
        expression syntax error will raise.
        """

        # 错误消息内容
        msg = (
            "xpath does not return any nodes or attributes. "
            "Be sure to specify in `xpath` the parent nodes of "
            "children and attributes to parse. "
            "If document uses namespaces denoted with "
            "xmlns, be sure to define namespaces and "
            "use them in xpath."
        )
        try:
            # 查找匹配 XPath 的元素
            elems = self.xml_doc.findall(self.xpath, namespaces=self.namespaces)
            # 获取子元素和属性
            children = [ch for el in elems for ch in el.findall("*")]
            attrs = {k: v for el in elems for k, v in el.attrib.items()}

            # 如果 elems 为空，抛出数值错误
            if elems is None:
                raise ValueError(msg)

            # 如果 elems 不为空，根据条件抛出数值错误
            if elems is not None:
                if self.elems_only and children == []:
                    raise ValueError(msg)
                if self.attrs_only and attrs == {}:
                    raise ValueError(msg)
                if children == [] and attrs == {}:
                    raise ValueError(msg)

        except (KeyError, SyntaxError) as err:
            # 捕获错误并抛出语法错误
            raise SyntaxError(
                "You have used an incorrect or unsupported XPath "
                "expression for etree library or you used an "
                "undeclared namespace prefix."
            ) from err

        return elems
    # 验证给定的命名是否有效
    def _validate_names(self) -> None:
        children: list[Any]

        # 如果有指定命名
        if self.names:
            # 如果启用了迭代解析
            if self.iterparse:
                # 获取迭代解析的第一个元素，并解析其子元素列表
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                # 在 XML 文档中查找指定路径的父节点
                parent = self.xml_doc.find(self.xpath, namespaces=self.namespaces)
                # 获取父节点的所有子节点列表，如果没有父节点则为空列表
                children = parent.findall("*") if parent is not None else []

            # 如果 names 是类列表结构
            if is_list_like(self.names):
                # 检查 names 的长度是否小于子元素列表的长度
                if len(self.names) < len(children):
                    # 如果不匹配，抛出数值错误
                    raise ValueError(
                        "names does not match length of child elements in xpath."
                    )
            else:
                # 如果 names 不是列表结构，抛出类型错误
                raise TypeError(
                    f"{type(self.names).__name__} is not a valid type for names"
                )

    # 解析给定的文档并返回根元素
    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Element:
        # 导入需要的 XML 解析工具
        from xml.etree.ElementTree import (
            XMLParser,
            parse,
        )

        # 使用辅助函数获取文档数据的处理器
        handle_data = get_data_from_filepath(
            filepath_or_buffer=raw_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        # 使用处理器打开文档数据
        with handle_data as xml_data:
            # 使用指定编码创建 XML 解析器
            curr_parser = XMLParser(encoding=self.encoding)
            # 解析 XML 数据并返回文档对象
            document = parse(xml_data, parser=curr_parser)

        # 返回解析后的文档的根元素
        return document.getroot()
        # 内部类，用于使用第三方完整功能的 XML 库 `lxml` 解析 XML 数据到 Pandas 的 DataFrame 中
class _LxmlFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into :class:`~pandas.DataFrame` with third-party
    full-featured XML library, ``lxml``, that supports
    ``XPath`` 1.0 and XSLT 1.0.
    """

    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate ``xpath``, names, optionally parse and run XSLT,
        and parse original or transformed XML and return specific nodes.
        """
        # 导入 lxml.etree 模块的 iterparse 函数
        from lxml.etree import iterparse

        # 如果 iterparse 未定义，则解析传入的 XML 文件路径或缓冲区内容
        if self.iterparse is None:
            self.xml_doc = self._parse_doc(self.path_or_buffer)

            # 如果定义了样式表，则解析样式表，并对 XML 进行转换
            if self.stylesheet:
                self.xsl_doc = self._parse_doc(self.stylesheet)
                self.xml_doc = self._transform_doc()

            # 验证 XPath 路径，并返回元素列表
            elems = self._validate_path()

        # 验证 XML 元素名称
        self._validate_names()

        # 解析节点并返回字典列表
        xml_dicts: list[dict[str, str | None]] = (
            self._parse_nodes(elems)
            if self.iterparse is None
            else self._iterparse_nodes(iterparse)
        )

        return xml_dicts

    def _validate_path(self) -> list[Any]:
        # 错误消息，用于指示 XPath 未返回任何节点或属性的情况
        msg = (
            "xpath does not return any nodes or attributes. "
            "Be sure to specify in `xpath` the parent nodes of "
            "children and attributes to parse. "
            "If document uses namespaces denoted with "
            "xmlns, be sure to define namespaces and "
            "use them in xpath."
        )

        # 使用给定的 XPath 和命名空间从 XML 文档中获取元素列表
        elems = self.xml_doc.xpath(self.xpath, namespaces=self.namespaces)
        # 获取所有子元素列表
        children = [ch for el in elems for ch in el.xpath("*")]
        # 获取所有属性并组成字典
        attrs = {k: v for el in elems for k, v in el.attrib.items()}

        # 如果未找到任何元素，则抛出 ValueError 异常
        if elems == []:
            raise ValueError(msg)

        # 如果找到元素，根据选项进一步验证
        if elems != []:
            if self.elems_only and children == []:
                raise ValueError(msg)
            if self.attrs_only and attrs == {}:
                raise ValueError(msg)
            if children == [] and attrs == {}:
                raise ValueError(msg)

        return elems

    def _validate_names(self) -> None:
        # 声明 children 变量
        children: list[Any]

        # 如果定义了 names，则根据情况验证子元素的数量
        if self.names:
            if self.iterparse:
                children = self.iterparse[next(iter(self.iterparse))]
            else:
                children = self.xml_doc.xpath(
                    self.xpath + "[1]/*", namespaces=self.namespaces
                )

            # 如果 names 是类列表结构，但长度不匹配子元素数量，则抛出 ValueError 异常
            if is_list_like(self.names):
                if len(self.names) < len(children):
                    raise ValueError(
                        "names does not match length of child elements in xpath."
                    )
            else:
                # 如果 names 不是列表结构，则抛出 TypeError 异常
                raise TypeError(
                    f"{type(self.names).__name__} is not a valid type for names"
                )

    def _parse_doc(
        self, raw_doc: FilePath | ReadBuffer[bytes] | ReadBuffer[str]
    ) -> Any:
        # 省略了解析 XML 文档的方法的部分代码，因为未提供完整代码，无法准确注释
        pass
    # 定义一个方法 `_transform_doc`，返回类型为 `etree._XSLTResultTree`
    def _transform_doc(self) -> etree._XSLTResultTree:
        """
        Transform original tree using stylesheet.

        This method will transform original xml using XSLT script into
        am ideally flatter xml document for easier parsing and migration
        to Data Frame.
        """
        # 导入 XSLT 类，用于处理样式表转换
        from lxml.etree import XSLT
        
        # 使用给定的样式表 `self.xsl_doc` 创建一个转换器
        transformer = XSLT(self.xsl_doc)
        
        # 使用转换器将 `self.xml_doc` 进行转换，返回新的文档 `new_doc`
        new_doc = transformer(self.xml_doc)
        
        # 返回转换后的文档 `new_doc`
        return new_doc
# 定义函数 `get_data_from_filepath`，从文件路径或缓冲区中提取原始 XML 数据
def get_data_from_filepath(
    filepath_or_buffer: FilePath | bytes | ReadBuffer[bytes] | ReadBuffer[str],  # 接受文件路径或类文件对象作为输入参数
    encoding: str | None,  # 编码参数，用于解析文件内容
    compression: CompressionOptions,  # 压缩选项，指定如何处理压缩文件
    storage_options: StorageOptions,  # 存储选项，影响文件的存取行为
):
    """
    Extract raw XML data.

    The method accepts two input types:
        1. filepath (string-like)
        2. file-like object (e.g. open file object, StringIO)
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)  # 将文件路径或缓冲区对象转换为字符串路径
    with get_handle(  # 获取文件处理句柄
        filepath_or_buffer,  # 文件路径或缓冲区对象
        "r",  # 只读模式打开文件
        encoding=encoding,  # 指定文件编码方式
        compression=compression,  # 指定文件压缩方式
        storage_options=storage_options,  # 指定文件存储选项
    ) as handle_obj:
        return (
            preprocess_data(handle_obj.handle.read())  # 预处理从文件句柄中读取的数据
            if hasattr(handle_obj.handle, "read")  # 如果文件句柄具有 `read` 方法
            else handle_obj.handle  # 否则直接返回文件句柄
        )


# 定义函数 `preprocess_data`，用于根据数据类型将数据转换为 StringIO 或 BytesIO 对象
def preprocess_data(data) -> io.StringIO | io.BytesIO:
    """
    Convert extracted raw data.

    This method will return underlying data of extracted XML content.
    The data either has a `read` attribute (e.g. a file object or a
    StringIO/BytesIO) or is a string or bytes that is an XML document.
    """

    if isinstance(data, str):  # 如果数据是字符串
        data = io.StringIO(data)  # 转换为 StringIO 对象

    elif isinstance(data, bytes):  # 如果数据是字节流
        data = io.BytesIO(data)  # 转换为 BytesIO 对象

    return data  # 返回处理后的数据对象


# 定义函数 `_data_to_frame`，将解析后的数据转换为 DataFrame 对象
def _data_to_frame(data, **kwargs) -> DataFrame:
    """
    Convert parsed data to Data Frame.

    This method will bind xml dictionary data of keys and values
    into named columns of Data Frame using the built-in TextParser
    class that build Data Frame and infers specific dtypes.
    """

    tags = next(iter(data))  # 获取数据中的标签
    nodes = [list(d.values()) for d in data]  # 获取数据中所有节点的值列表

    try:
        with TextParser(nodes, names=tags, **kwargs) as tp:  # 使用 TextParser 类创建 Data Frame
            return tp.read()  # 返回解析后的 Data Frame
    except ParserError as err:  # 捕获解析错误
        raise ParserError(
            "XML document may be too complex for import. "
            "Try to flatten document and use distinct "
            "element and attribute names."
        ) from err  # 抛出解析错误信息


# 定义函数 `_parse`，调用内部解析器解析 XML 数据并返回 DataFrame 对象
def _parse(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],  # 文件路径或缓冲区对象
    xpath: str,  # XPath 查询表达式
    namespaces: dict[str, str] | None,  # XML 命名空间映射
    elems_only: bool,  # 是否仅处理元素
    attrs_only: bool,  # 是否仅处理属性
    names: Sequence[str] | None,  # 列名列表
    dtype: DtypeArg | None,  # 数据类型
    converters: ConvertersArg | None,  # 数据转换器
    parse_dates: ParseDatesArg | None,  # 是否解析日期
    encoding: str | None,  # 文件编码方式
    parser: XMLParsers,  # XML 解析器类型
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None,  # 样式表文件路径或缓冲区对象
    iterparse: dict[str, list[str]] | None,  # 迭代解析选项
    compression: CompressionOptions,  # 压缩选项
    storage_options: StorageOptions,  # 存储选项
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 数据类型后端
    **kwargs,  # 其他关键字参数
) -> DataFrame:
    """
    Call internal parsers.

    This method will conditionally call internal parsers:
    LxmlFrameParser and/or EtreeParser.

    Raises
    ------
    ImportError
        * If lxml is not installed if selected as parser.
    """
    ValueError
        * If parser is not lxml or etree.
    """

    p: _EtreeFrameParser | _LxmlFrameParser

    # 根据传入的 parser 参数选择相应的解析器对象
    if parser == "lxml":
        # 尝试导入 lxml.etree 模块，如果不存在则忽略错误
        lxml = import_optional_dependency("lxml.etree", errors="ignore")

        # 如果成功导入 lxml 模块，则创建 _LxmlFrameParser 对象
        if lxml is not None:
            p = _LxmlFrameParser(
                path_or_buffer,
                xpath,
                namespaces,
                elems_only,
                attrs_only,
                names,
                dtype,
                converters,
                parse_dates,
                encoding,
                stylesheet,
                iterparse,
                compression,
                storage_options,
            )
        else:
            # 如果未成功导入 lxml 模块，则抛出 ImportError 异常
            raise ImportError("lxml not found, please install or use the etree parser.")

    elif parser == "etree":
        # 创建 _EtreeFrameParser 对象
        p = _EtreeFrameParser(
            path_or_buffer,
            xpath,
            namespaces,
            elems_only,
            attrs_only,
            names,
            dtype,
            converters,
            parse_dates,
            encoding,
            stylesheet,
            iterparse,
            compression,
            storage_options,
        )
    else:
        # 如果 parser 参数既不是 "lxml" 也不是 "etree"，则抛出 ValueError 异常
        raise ValueError("Values for parser can only be lxml or etree.")

    # 使用解析器对象 p 解析数据，并返回数据字典列表
    data_dicts = p.parse_data()

    # 将数据字典列表转换为数据框架，并返回结果
    return _data_to_frame(
        data=data_dicts,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        dtype_backend=dtype_backend,
        **kwargs,
    )
@doc(
    storage_options=_shared_docs["storage_options"],  # 使用共享文档中的 storage_options 描述
    decompression_options=_shared_docs["decompression_options"] % "path_or_buffer",  # 使用共享文档中的 decompression_options 描述，其中 path_or_buffer 为占位符
)
def read_xml(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],  # path_or_buffer 参数可以是文件路径、字节读取缓冲区或字符串读取缓冲区
    *,
    xpath: str = "./*",  # XPath 表达式，默认为选择所有子元素
    namespaces: dict[str, str] | None = None,  # XML 文档中定义的命名空间字典，键为命名空间前缀，值为命名空间 URI
    elems_only: bool = False,  # 是否仅解析指定 XPath 下的子元素，默认为 False
    attrs_only: bool = False,  # 是否仅解析指定 XPath 下的属性，默认为 False
    names: Sequence[str] | None = None,  # 需要解析的列名列表，可为空
    dtype: DtypeArg | None = None,  # 数据类型参数或 None
    converters: ConvertersArg | None = None,  # 转换器参数或 None
    parse_dates: ParseDatesArg | None = None,  # 解析日期参数或 None
    # encoding can not be None for lxml and StringIO input
    encoding: str | None = "utf-8",  # 编码类型，默认为 UTF-8
    parser: XMLParsers = "lxml",  # XML 解析器，默认为 lxml
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None = None,  # 样式表文件路径或缓冲区，可为空
    iterparse: dict[str, list[str]] | None = None,  # 迭代解析选项字典或 None
    compression: CompressionOptions = "infer",  # 压缩选项，默认自动推断
    storage_options: StorageOptions | None = None,  # 存储选项或 None
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 数据类型后端或无默认值
) -> DataFrame:
    r"""
    Read XML document into a :class:`~pandas.DataFrame` object.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    path_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a ``read()`` function. The string can be a path.
        The string can further be a URL. Valid URL schemes
        include http, ftp, s3, and file.

        .. deprecated:: 2.1.0
            Passing xml literal strings is deprecated.
            Wrap literal xml input in ``io.StringIO`` or ``io.BytesIO`` instead.

    xpath : str, optional, default './\*'
        The ``XPath`` to parse required set of nodes for migration to
        :class:`~pandas.DataFrame`.``XPath`` should return a collection of elements
        and not a single element. Note: The ``etree`` parser supports limited ``XPath``
        expressions. For more complex ``XPath``, use ``lxml`` which requires
        installation.

    namespaces : dict, optional
        The namespaces defined in XML document as dicts with key being
        namespace prefix and value the URI. There is no need to include all
        namespaces in XML, only the ones used in ``xpath`` expression.
        Note: if XML document uses default namespace denoted as
        `xmlns='<URI>'` without a prefix, you must assign any temporary
        namespace prefix such as 'doc' to the URI in order to parse
        underlying nodes and/or attributes.

    elems_only : bool, optional, default False
        Parse only the child elements at the specified ``xpath``. By default,
        all child elements and non-empty text nodes are returned.

    attrs_only :  bool, optional, default False
        Parse only the attributes at the specified ``xpath``.
        By default, all attributes are returned.
    # names : list-like, optional
    #     Column names for DataFrame of parsed XML data. Use this parameter to
    #     rename original element names and distinguish same named elements and
    #     attributes.

    # dtype : Type name or dict of column -> type, optional
    #     Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32,
    #     'c': 'Int64'}
    #     Use `str` or `object` together with suitable `na_values` settings
    #     to preserve and not interpret dtype.
    #     If converters are specified, they will be applied INSTEAD
    #     of dtype conversion.
    #     
    #     .. versionadded:: 1.5.0

    # converters : dict, optional
    #     Dict of functions for converting values in certain columns. Keys can either
    #     be integers or column labels.
    #     
    #     .. versionadded:: 1.5.0

    # parse_dates : bool or list of int or names or list of lists or dict, default False
    #     Identifiers to parse index or columns to datetime. The behavior is as follows:
    #     
    #     - boolean. If True -> try parsing the index.
    #     - list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
    #       each as a separate date column.
    #     - list of lists. e.g. If [[1, 3]] -> combine columns 1 and 3 and parse as
    #       a single date column.
    #     - dict, e.g. {'foo' : [1, 3]} -> parse columns 1, 3 as date and call
    #       result 'foo'
    #     
    #     .. versionadded:: 1.5.0

    # encoding : str, optional, default 'utf-8'
    #     Encoding of XML document.

    # parser : {'lxml','etree'}, default 'lxml'
    #     Parser module to use for retrieval of data. Only 'lxml' and
    #     'etree' are supported. With 'lxml' more complex XPath searches
    #     and ability to use XSLT stylesheet are supported.

    # stylesheet : str, path object or file-like object
    #     A URL, file-like object, or a string path containing an XSLT script.
    #     This stylesheet should flatten complex, deeply nested XML documents
    #     for easier parsing. To use this feature you must have `lxml` module
    #     installed and specify 'lxml' as `parser`. The `xpath` must
    #     reference nodes of transformed XML document generated after XSLT
    #     transformation and not the original XML document. Only XSLT 1.0
    #     scripts and not later versions are currently supported.
    iterparse : dict, optional
        # `iterparse`参数，类型为字典，可选项
        # 指定在XML文档的迭代解析中要检索的节点或属性
        # 字典的键是重复元素的名称，值是重复元素的后代元素或属性名称列表
        # 注意：如果使用此选项，将替代`xpath`解析，与`xpath`不同的是，后代元素不需要彼此关联，可以存在于重复元素下的任何位置
        # 这种内存高效的方法适用于非常大的XML文件（500MB、1GB或5GB+）
        # 例如，{"row_element": ["child_elem", "attr", "grandchild_elem"]}
        
        .. versionadded:: 1.5.0
        # 自版本1.5.0起添加了此功能

    {decompression_options}
        # 解压选项，具体内容由外部传入

        .. versionchanged:: 1.4.0 Zstandard support.
        # 自版本1.4.0起添加了Zstandard支持

    {storage_options}
        # 存储选项，具体内容由外部传入

    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        # dtype_backend参数，类型为字符串，可选项，默认为'numpy_nullable'
        # 应用于生成的DataFrame的后端数据类型（仍处于试验阶段）
        # 行为如下：
        # "numpy_nullable": 返回支持可空dtype的DataFrame（默认选项）
        # "pyarrow": 返回基于pyarrow的可空ArrowDtype的DataFrame

        .. versionadded:: 2.0
        # 自版本2.0起添加了此功能

    Returns
    -------
    df
        # 返回一个DataFrame对象

    See Also
    --------
    read_json : 将JSON字符串转换为pandas对象。
    read_html : 将HTML表格读取为DataFrame对象。

    Notes
    -----
    # 关于本方法的注意事项：

    This method is best designed to import shallow XML documents in
    following format which is the ideal fit for the two-dimensions of a
    ``DataFrame`` (row by column). ::

            <root>
                <row>
                  <column1>data</column1>
                  <column2>data</column2>
                  <column3>data</column3>
                  ...
               </row>
               <row>
                  ...
               </row>
               ...
            </root>

    # 本方法最适合导入以下格式的浅层XML文档，这种格式最适合DataFrame的二维结构（行列）。

    As a file format, XML documents can be designed any way including
    layout of elements and attributes as long as it conforms to W3C
    specifications. Therefore, this method is a convenience handler for
    a specific flatter design and not all possible XML structures.

    # 作为文件格式，XML文档可以按任何方式设计，包括元素和属性的布局，只要符合W3C规范。
    # 因此，此方法是特定扁平设计的便捷处理程序，并非所有可能的XML结构。

    However, for more complex XML documents, ``stylesheet`` allows you to
    temporarily redesign original document with XSLT (a special purpose
    language) for a flatter version for migration to a DataFrame.

    # 但是，对于更复杂的XML文档，“stylesheet”允许您使用XSLT（一种特殊用途的语言）暂时重新设计原始文档，以便迁移到DataFrame的扁平版本。

    This function will *always* return a single :class:`DataFrame` or raise
    exceptions due to issues with XML document, ``xpath``, or other
    parameters.

    # 该函数始终返回单个DataFrame对象，或由于XML文档、`xpath`或其他参数问题而引发异常。

    See the :ref:`read_xml documentation in the IO section of the docs
    <io.read_xml>` for more information in using this method to parse XML
    files to DataFrames.

    # 有关使用此方法解析XML文件到DataFrame的更多信息，请参阅文档中IO部分的read_xml文档。

    Examples
    --------
    >>> from io import StringIO
    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data xmlns="http://example.com">
    ...  <row>
    # 定义包含 XML 数据的字符串
    xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data>
    ...   <row shape="square" degrees="360" sides="4.0"/>
    ...   <row shape="circle" degrees="360"/>
    ...   <row shape="triangle" degrees="180" sides="3.0"/>
    ... </data>'''

    # 使用 pandas 的 read_xml 函数读取 XML 数据，并将其转换为 DataFrame
    >>> df = pd.read_xml(StringIO(xml))
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    # 重新定义 xml 字符串，增加命名空间
    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data>
    ...   <row shape="square" degrees="360" sides="4.0"/>
    ...   <row shape="circle" degrees="360"/>
    ...   <row shape="triangle" degrees="180" sides="3.0"/>
    ... </data>'''

    # 使用 pandas 的 read_xml 函数读取带有 XPath 表达式的 XML 数据，并指定命名空间
    >>> df = pd.read_xml(
    ...     StringIO(xml), xpath=".//row"
    ... )
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    # 重新定义 xml_data 字符串，带有自定义命名空间
    >>> xml_data = '''
    ...         <data>
    ...            <row>
    ...               <index>0</index>
    ...               <a>1</a>
    ...               <b>2.5</b>
    ...               <c>True</c>
    ...               <d>a</d>
    ...               <e>2019-12-31 00:00:00</e>
    ...            </row>
    ...            <row>
    ...               <index>1</index>
    ...               <b>4.5</b>
    ...               <c>False</c>
    ...               <d>b</d>
    ...               <e>2019-12-31 00:00:00</e>
    ...            </row>
    ...         </data>
    ...         '''

    # 使用 pandas 的 read_xml 函数读取包含日期解析和空值处理的 XML 数据，并指定后端数据类型为 numpy_nullable
    >>> df = pd.read_xml(
    ...     StringIO(xml_data), dtype_backend="numpy_nullable", parse_dates=["e"]
    ... )
    >>> df
       index     a    b      c  d          e
    0      0     1  2.5   True  a 2019-12-31
    1      1  <NA>  4.5  False  b 2019-12-31

    """
    # 检查 dtype_backend 参数的数据类型
    check_dtype_backend(dtype_backend)
    # 调用 _parse 函数，并传递以下参数进行解析和处理
    # - path_or_buffer: 指定路径或者缓冲区对象，用于读取数据
    # - xpath: XML 或 HTML 文件中的 XPath 表达式，用于定位元素
    # - namespaces: XML 命名空间的字典，用于解析 XML 文件时指定命名空间
    # - elems_only: 是否仅解析元素，默认为 False
    # - attrs_only: 是否仅解析属性，默认为 False
    # - names: 列名的列表，用于指定返回的数据框的列名
    # - dtype: 数据类型的字典，用于指定数据各列的类型
    # - converters: 转换器的字典，用于自定义数据读取时的转换规则
    # - parse_dates: 解析日期的规则，用于将指定列的数据解析为日期类型
    # - encoding: 文件编码，用于读取文本文件时指定正确的编码格式
    # - parser: 文件解析器，指定处理 XML 或 HTML 文件的解析器
    # - stylesheet: XSLT 样式表文件路径，用于将 XML 文件转换为其他格式
    # - iterparse: 是否使用迭代方式解析 XML 文件，默认为 False
    # - compression: 压缩格式，指定读取文件时使用的压缩算法
    # - storage_options: 存储选项，用于传递给底层存储接口的额外参数
    # - dtype_backend: 数据类型的后端实现，用于指定不同的数据存储方式
    return _parse(
        path_or_buffer=path_or_buffer,
        xpath=xpath,
        namespaces=namespaces,
        elems_only=elems_only,
        attrs_only=attrs_only,
        names=names,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        encoding=encoding,
        parser=parser,
        stylesheet=stylesheet,
        iterparse=iterparse,
        compression=compression,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )
```