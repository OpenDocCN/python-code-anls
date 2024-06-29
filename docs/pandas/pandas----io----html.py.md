# `D:\src\scipysrc\pandas\pandas\io\html.py`

```
"""
:mod:`pandas.io.html` is a module containing functionality for dealing with
HTML IO.

"""

from __future__ import annotations

from collections import abc  # 导入标准库中的 collections.abc 模块
import errno  # 导入 errno 模块，用于处理错误码
import numbers  # 导入 numbers 模块，用于处理数字类型
import os  # 导入 os 模块，提供操作系统相关功能
import re  # 导入 re 模块，用于处理正则表达式
from re import Pattern  # 从 re 模块导入 Pattern 类型
from typing import (
    TYPE_CHECKING,  # 导入类型提示中的 TYPE_CHECKING 常量
    Literal,  # 导入 Literal 类型提示
    cast,  # 导入 cast 函数，用于强制类型转换
)

from pandas._libs import lib  # 导入 pandas 内部的 _libs 库
from pandas.compat._optional import import_optional_dependency  # 导入依赖可选的组件
from pandas.errors import (
    AbstractMethodError,  # 导入 AbstractMethodError 异常类
    EmptyDataError,  # 导入 EmptyDataError 异常类
)
from pandas.util._decorators import doc  # 导入 pandas 工具库中的 doc 装饰器
from pandas.util._validators import check_dtype_backend  # 导入数据类型后端验证函数

from pandas.core.dtypes.common import is_list_like  # 导入检查对象是否类列表的函数

from pandas import isna  # 导入 pandas 中的 isna 函数
from pandas.core.indexes.base import Index  # 导入索引基类
from pandas.core.indexes.multi import MultiIndex  # 导入多重索引类
from pandas.core.series import Series  # 导入序列类
from pandas.core.shared_docs import _shared_docs  # 导入共享文档

from pandas.io.common import (
    get_handle,  # 导入获取处理器函数
    is_url,  # 导入检查是否为 URL 的函数
    stringify_path,  # 导入路径转换为字符串函数
    validate_header_arg,  # 导入验证标头参数函数
)
from pandas.io.formats.printing import pprint_thing  # 导入格式化打印函数
from pandas.io.parsers import TextParser  # 导入文本解析器类

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,  # 导入 Iterable 类型
        Sequence,  # 导入 Sequence 类型
    )

    from pandas._typing import (
        BaseBuffer,  # 导入 BaseBuffer 类型
        DtypeBackend,  # 导入 DtypeBackend 类型
        FilePath,  # 导入 FilePath 类型
        HTMLFlavors,  # 导入 HTMLFlavors 类型
        ReadBuffer,  # 导入 ReadBuffer 类型
        StorageOptions,  # 导入 StorageOptions 类型
    )

    from pandas import DataFrame  # 导入 DataFrame 类型

#############
# READ HTML #
#############
_RE_WHITESPACE = re.compile(r"[\r\n]+|\s{2,}")  # 编译用于匹配空白字符和换行的正则表达式


def _remove_whitespace(s: str, regex: Pattern = _RE_WHITESPACE) -> str:
    """
    Replace extra whitespace inside of a string with a single space.

    Parameters
    ----------
    s : str or unicode
        The string from which to remove extra whitespace.
    regex : re.Pattern
        The regular expression to use to remove extra whitespace.

    Returns
    -------
    subd : str or unicode
        `s` with all extra whitespace replaced with a single space.
    """
    return regex.sub(" ", s.strip())  # 使用给定的正则表达式替换字符串中的额外空白为单个空格


def _get_skiprows(skiprows: int | Sequence[int] | slice | None) -> int | Sequence[int]:
    """
    Get an iterator given an integer, slice or container.

    Parameters
    ----------
    skiprows : int, slice, container
        The iterator to use to skip rows; can also be a slice.

    Raises
    ------
    TypeError
        * If `skiprows` is not a slice, integer, or Container

    Returns
    -------
    it : iterable
        A proper iterator to use to skip rows of a DataFrame.
    """
    if isinstance(skiprows, slice):
        start, step = skiprows.start or 0, skiprows.step or 1
        return list(range(start, skiprows.stop, step))  # 返回范围内的整数列表作为迭代器
    elif isinstance(skiprows, numbers.Integral) or is_list_like(skiprows):
        return cast("int | Sequence[int]", skiprows)  # 如果是整数或类列表，直接返回，否则进行类型强制转换
    elif skiprows is None:
        return 0  # 如果为 None，则返回 0
    raise TypeError(f"{type(skiprows).__name__} is not a valid type for skipping rows")  # 抛出类型错误异常


def _read(
    obj: FilePath | BaseBuffer,
    encoding: str | None,
    storage_options: StorageOptions | None,
) -> str | bytes:
    """
    Read from a file path or buffer and return contents as string or bytes.

    """
    # 尝试从 URL、文件或字符串读取内容。

    Parameters
    ----------
    obj : str, unicode, path object, or file-like object
        要读取的对象，可以是字符串、Unicode、路径对象或类文件对象。

    Returns
    -------
    raw_text : str
        返回读取的原始文本内容。
    """
    try:
        # 使用 get_handle 函数获取处理对象 handles，在只读模式下打开对象，并指定编码和存储选项
        with get_handle(
            obj, "r", encoding=encoding, storage_options=storage_options
        ) as handles:
            # 读取 handles 中的内容并返回
            return handles.handle.read()
    except OSError as err:
        # 如果不是 URL，则抛出文件未找到的错误
        if not is_url(obj):
            raise FileNotFoundError(
                f"[Errno {errno.ENOENT}] {os.strerror(errno.ENOENT)}: {obj}"
            ) from err
        # 如果是 URL，则继续抛出原始的 OSError
        raise
    """
    Base class for parsers that parse HTML into DataFrames.

    Parameters
    ----------
    io : str or file-like
        This can be either a string path, a valid URL using the HTTP,
        FTP, or FILE protocols or a file-like object.

    match : str or regex
        The text to match in the document.

    attrs : dict
        List of HTML <table> element attributes to match.

    encoding : str
        Encoding to be used by parser

    displayed_only : bool
        Whether or not items with "display:none" should be ignored

    extract_links : {None, "all", "header", "body", "footer"}
        Table elements in the specified section(s) with <a> tags will have their
        href extracted.

        .. versionadded:: 1.5.0

    storage_options : StorageOptions, optional
        Options for storage configuration.

    Attributes
    ----------
    io : str or file-like
        raw HTML, URL, or file-like object

    match : regex
        The text to match in the raw HTML

    attrs : dict-like
        A dictionary of valid table attributes to use to search for table
        elements.

    encoding : str
        Encoding to be used by parser

    displayed_only : bool
        Whether or not items with "display:none" should be ignored

    extract_links : {None, "all", "header", "body", "footer"}
        Table elements in the specified section(s) with <a> tags will have their
        href extracted.

        .. versionadded:: 1.5.0

    storage_options : StorageOptions, optional
        Options for storage configuration.

    Notes
    -----
    To subclass this class effectively you must override the following methods:
        * :func:`_build_doc`
        * :func:`_attr_getter`
        * :func:`_href_getter`
        * :func:`_text_getter`
        * :func:`_parse_td`
        * :func:`_parse_thead_tr`
        * :func:`_parse_tbody_tr`
        * :func:`_parse_tfoot_tr`
        * :func:`_parse_tables`
        * :func:`_equals_tag`
    See each method's respective documentation for details on their
    functionality.
    """

    def __init__(
        self,
        io: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
        match: str | Pattern,
        attrs: dict[str, str] | None,
        encoding: str,
        displayed_only: bool,
        extract_links: Literal[None, "header", "footer", "body", "all"],
        storage_options: StorageOptions = None,
    ) -> None:
        """
        Initialize the HTML frame parser with given parameters.

        Parameters
        ----------
        io : str or file-like
            Represents the input source: either a path, URL, or file-like object.
        match : str or regex
            Specifies the text or pattern to match within the document.
        attrs : dict, optional
            Attributes of HTML <table> elements to match.
        encoding : str
            The character encoding to be used for parsing.
        displayed_only : bool
            Determines if items marked with "display:none" should be ignored.
        extract_links : {None, "all", "header", "body", "footer"}
            Specifies sections where <a> tag hrefs should be extracted.
        storage_options : StorageOptions, optional
            Options for configuring storage.

        Returns
        -------
        None
        """
        self.io = io
        self.match = match
        self.attrs = attrs
        self.encoding = encoding
        self.displayed_only = displayed_only
        self.extract_links = extract_links
        self.storage_options = storage_options

    def parse_tables(self):
        """
        Parse and return all tables from the HTML document.

        Returns
        -------
        generator of parsed (header, body, footer) tuples from tables.
        """
        # Build the document structure from the input source
        tables = self._parse_tables(self._build_doc(), self.match, self.attrs)
        # Return a generator of parsed table components
        return (self._parse_thead_tbody_tfoot(table) for table in tables)
    def _attr_getter(self, obj, attr):
        """
        Return the attribute value of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        attr : str or unicode
            The attribute, such as "colspan"

        Returns
        -------
        str or unicode
            The attribute value.
        """
        # 从 DOM 节点中获取指定属性的值
        return obj.get(attr)

    def _href_getter(self, obj) -> str | None:
        """
        Return a href if the DOM node contains a child <a> or None.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        href : str or unicode
            The href from the <a> child of the DOM node.
        """
        # 抽象方法，用于获取 DOM 节点中 <a> 子节点的 href 属性值或者返回 None
        raise AbstractMethodError(self)

    def _text_getter(self, obj):
        """
        Return the text of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        text : str or unicode
            The text from an individual DOM node.
        """
        # 抽象方法，用于获取 DOM 节点中的文本内容
        raise AbstractMethodError(self)

    def _parse_td(self, obj):
        """
        Return the td elements from a row element.

        Parameters
        ----------
        obj : node-like
            A DOM <tr> node.

        Returns
        -------
        list of node-like
            These are the elements of each row, i.e., the columns.
        """
        # 抽象方法，用于解析 DOM 中 <tr> 节点的 <td> 元素列表
        raise AbstractMethodError(self)

    def _parse_thead_tr(self, table):
        """
        Return the list of thead row elements from the parsed table element.

        Parameters
        ----------
        table : a table element that contains zero or more thead elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        # 抽象方法，从解析后的表格元素中返回 thead 行元素列表
        raise AbstractMethodError(self)

    def _parse_tbody_tr(self, table):
        """
        Return the list of tbody row elements from the parsed table element.

        HTML5 table bodies consist of either 0 or more <tbody> elements (which
        only contain <tr> elements) or 0 or more <tr> elements. This method
        checks for both structures.

        Parameters
        ----------
        table : a table element that contains row elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        # 抽象方法，从解析后的表格元素中返回 tbody 行元素列表
        raise AbstractMethodError(self)

    def _parse_tfoot_tr(self, table):
        """
        Return the list of tfoot row elements from the parsed table element.

        Parameters
        ----------
        table : a table element that contains row elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        # 抽象方法，从解析后的表格元素中返回 tfoot 行元素列表
        raise AbstractMethodError(self)
    # 定义一个方法 `_parse_tables`，用于解析给定文档中与指定文本匹配的所有表格元素
    def _parse_tables(self, document, match, attrs):
        """
        Return all tables from the parsed DOM.

        Parameters
        ----------
        document : the DOM from which to parse the table element.

        match : str or regular expression
            The text to search for in the DOM tree.

        attrs : dict
            A dictionary of table attributes that can be used to disambiguate
            multiple tables on a page.

        Raises
        ------
        ValueError : `match` does not match any text in the document.

        Returns
        -------
        list of node-like
            HTML <table> elements to be parsed into raw data.
        """
        raise AbstractMethodError(self)

    # 定义一个方法 `_equals_tag`，用于检查给定的 DOM 节点是否与指定的标签名称匹配
    def _equals_tag(self, obj, tag) -> bool:
        """
        Return whether an individual DOM node matches a tag

        Parameters
        ----------
        obj : node-like
            A DOM node.

        tag : str
            Tag name to be checked for equality.

        Returns
        -------
        boolean
            Whether `obj`'s tag name is `tag`
        """
        raise AbstractMethodError(self)

    # 定义一个方法 `_build_doc`，返回一个类似树形结构的对象，用于迭代整个 DOM
    def _build_doc(self):
        """
        Return a tree-like object that can be used to iterate over the DOM.

        Returns
        -------
        node-like
            The DOM from which to parse the table element.
        """
        raise AbstractMethodError(self)
    def _parse_thead_tbody_tfoot(self, table_html):
        """
        给定一个表格，返回解析后的表头、表体和表尾。

        Parameters
        ----------
        table_html : node-like
            表格的 HTML 节点或类似结构

        Returns
        -------
        tuple of (header, body, footer), each a list of list-of-text rows.
            返回包含文本行列表的元组 (header, body, footer)

        Notes
        -----
        Header 和 body 都是列表的列表。顶层列表是行的列表。每行是一个字符串文本列表。

        Logic: Use <thead>, <tbody>, <tfoot> elements to identify
               header, body, and footer, otherwise:
               - Put all rows into body
               - Move rows from top of body to header only if
                 all elements inside row are <th>
               - Move rows from bottom of body to footer only if
                 all elements inside row are <th>
            逻辑：使用 <thead>、<tbody>、<tfoot> 元素来识别表头、表体和表尾，否则：
                 - 将所有行放入表体
                 - 如果行顶部的所有元素都是 <th>，则将行从表体移到表头
                 - 如果行底部的所有元素都是 <th>，则将行从表体移到表尾
        """
        # 解析表头行
        header_rows = self._parse_thead_tr(table_html)
        # 解析表体行
        body_rows = self._parse_tbody_tr(table_html)
        # 解析表尾行
        footer_rows = self._parse_tfoot_tr(table_html)

        # 判断一个行是否全部由 <th> 组成
        def row_is_all_th(row):
            return all(self._equals_tag(t, "th") for t in self._parse_td(row))

        if not header_rows:
            # 如果表格没有 <thead>，则将表体顶部的全为 <th> 的行移到表头
            while body_rows and row_is_all_th(body_rows[0]):
                header_rows.append(body_rows.pop(0))

        # 扩展表格中的 colspan 和 rowspan 属性
        header = self._expand_colspan_rowspan(header_rows, section="header")
        body = self._expand_colspan_rowspan(body_rows, section="body")
        footer = self._expand_colspan_rowspan(footer_rows, section="footer")

        return header, body, footer

    def _expand_colspan_rowspan(
        self, rows, section: Literal["header", "footer", "body"]
    ):
        """
        扩展表格行中的 colspan 和 rowspan 属性。

        Parameters
        ----------
        rows : list
            表格行的列表
        section : Literal["header", "footer", "body"]
            指定行所属的部分：表头、表尾或表体

        Returns
        -------
        list
            扩展后的行列表
        """
    ) -> list[list]:
        """
        Given a list of <tr>s, return a list of text rows.

        Parameters
        ----------
        rows : list of node-like
            List of <tr>s
        section : the section that the rows belong to (header, body or footer).

        Returns
        -------
        list of list
            Each returned row is a list of str text, or tuple (text, link)
            if extract_links is not None.

        Notes
        -----
        Any cell with ``rowspan`` or ``colspan`` will have its contents copied
        to subsequent cells.
        """
        # Initialize an empty list to store all rows as lists of strings or tuples
        all_texts = []  # list of rows, each a list of str
        
        # Type hinting for variables
        text: str | tuple
        remainder: list[
            tuple[int, str | tuple, int]
        ] = []  # list of (index, text, nrows)

        # Iterate over each <tr> element in the provided list of rows
        for tr in rows:
            # Initialize an empty list to store text data for the current row
            texts = []  # the output for this row
            # Initialize an empty list to store data that spans across rows
            next_remainder = []

            # Index to keep track of column positions
            index = 0
            # Parse <td> elements within the current <tr>
            tds = self._parse_td(tr)
            for td in tds:
                # Process contents from previous rows that have rowspan>1
                while remainder and remainder[0][0] <= index:
                    prev_i, prev_text, prev_rowspan = remainder.pop(0)
                    texts.append(prev_text)
                    if prev_rowspan > 1:
                        next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
                    index += 1

                # Extract text from the current <td> and handle colspan
                text = _remove_whitespace(self._text_getter(td))
                if self.extract_links in ("all", section):
                    href = self._href_getter(td)
                    text = (text, href)
                rowspan = int(self._attr_getter(td, "rowspan") or 1)
                colspan = int(self._attr_getter(td, "colspan") or 1)

                # Add the text to 'texts' list for the required number of columns
                for _ in range(colspan):
                    texts.append(text)
                    if rowspan > 1:
                        next_remainder.append((index, text, rowspan - 1))
                    index += 1

            # Add remaining content from previous rows to the current row
            for prev_i, prev_text, prev_rowspan in remainder:
                texts.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))

            # Append the fully processed row to 'all_texts' list
            all_texts.append(texts)
            # Update 'remainder' for the next iteration
            remainder = next_remainder

        # Add rows that only exist due to rowspan in the previous row
        while remainder:
            next_remainder = []
            texts = []
            for prev_i, prev_text, prev_rowspan in remainder:
                texts.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
            all_texts.append(texts)
            remainder = next_remainder

        # Return the final list of rows containing text or (text, link) tuples
        return all_texts
    # 处理隐藏表格的方法，根据条件过滤表格列表中的隐藏元素
    def _handle_hidden_tables(self, tbl_list, attr_name: str):
        """
        Return list of tables, potentially removing hidden elements

        Parameters
        ----------
        tbl_list : list of node-like
            Type of list elements will vary depending upon parser used
        attr_name : str
            Name of the accessor for retrieving HTML attributes

        Returns
        -------
        list of node-like
            Return type matches `tbl_list`
        """
        # 如果不仅显示可见元素，则直接返回原始表格列表
        if not self.displayed_only:
            return tbl_list

        # 使用列表推导式，过滤掉样式中包含"display:none"的元素
        return [
            x
            for x in tbl_list
            if "display:none" not in getattr(x, attr_name).get("style", "").replace(" ", "")
        ]
    # 定义一个私有类，继承自 _HtmlFrameParser 类，用于解析 HTML 到 DataFrame 的操作，内部使用 BeautifulSoup 实现。
    class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
        """
        HTML to DataFrame parser that uses BeautifulSoup under the hood.

        See Also
        --------
        pandas.io.html._HtmlFrameParser
        pandas.io.html._LxmlFrameParser

        Notes
        -----
        Documentation strings for this class are in the base class
        :class:`pandas.io.html._HtmlFrameParser`.
        """

        # 解析表格的方法，从给定的 document 中查找符合条件的表格元素
        def _parse_tables(self, document, match, attrs):
            element_name = "table"
            # 使用 BeautifulSoup 的 find_all 方法查找所有符合条件的表格元素
            tables = document.find_all(element_name, attrs=attrs)
            if not tables:
                raise ValueError("No tables found")

            result = []
            unique_tables = set()
            # 处理隐藏表格
            tables = self._handle_hidden_tables(tables, "attrs")

            for table in tables:
                # 如果设置了只显示显示的表格，则去除 style 和 display:none 的元素
                if self.displayed_only:
                    for elem in table.find_all("style"):
                        elem.decompose()

                    for elem in table.find_all(style=re.compile(r"display:\s*none")):
                        elem.decompose()

                # 如果表格不在唯一表格集合中并且匹配到了指定的字符串，则将其添加到结果中
                if table not in unique_tables and table.find(string=match) is not None:
                    result.append(table)
                unique_tables.add(table)
            if not result:
                raise ValueError(f"No tables found matching pattern {match.pattern!r}")
            return result

        # 从对象中获取 href 属性的方法
        def _href_getter(self, obj) -> str | None:
            a = obj.find("a", href=True)
            return None if not a else a["href"]

        # 获取对象文本内容的方法
        def _text_getter(self, obj):
            return obj.text

        # 判断对象是否与指定标签匹配的方法
        def _equals_tag(self, obj, tag) -> bool:
            return obj.name == tag

        # 解析表格行中的 td 或 th 元素的方法
        def _parse_td(self, row):
            return row.find_all(("td", "th"), recursive=False)

        # 解析表格 thead 中 tr 元素的方法
        def _parse_thead_tr(self, table):
            return table.select("thead tr")

        # 解析表格 tbody 中 tr 元素的方法
        def _parse_tbody_tr(self, table):
            from_tbody = table.select("tbody tr")
            from_root = table.find_all("tr", recursive=False)
            # HTML 规范：这两个列表最多只有一个包含内容
            return from_tbody + from_root

        # 解析表格 tfoot 中 tr 元素的方法
        def _parse_tfoot_tr(self, table):
            return table.select("tfoot tr")

        # 设置构建文档的方法
        def _setup_build_doc(self):
            # 从输入流中读取原始文本
            raw_text = _read(self.io, self.encoding, self.storage_options)
            if not raw_text:
                raise ValueError(f"No text parsed from document: {self.io}")
            return raw_text

        # 构建文档的方法，使用 BeautifulSoup 解析 HTML 文档并返回解析后的对象
        def _build_doc(self):
            from bs4 import BeautifulSoup

            # 设置构建文档
            bdoc = self._setup_build_doc()
            if isinstance(bdoc, bytes) and self.encoding is not None:
                udoc = bdoc.decode(self.encoding)
                from_encoding = None
            else:
                udoc = bdoc
                from_encoding = self.encoding

            # 使用 BeautifulSoup 解析 HTML 文档
            soup = BeautifulSoup(udoc, features="html5lib", from_encoding=from_encoding)

            # 替换所有 <br> 标签为换行符 + 其文本内容
            for br in soup.find_all("br"):
                br.replace_with("\n" + br.text)

            return soup


    # 构建 XPath 表达式的函数，模拟 bs4 在使用 lxml 解析器时传入关键字参数搜索属性的能力
    def _build_xpath_expr(attrs) -> str:
        """
        Build an xpath expression to simulate bs4's ability to pass in kwargs to
        search for attributes when using the lxml parser.

        Parameters
        ----------
        attrs : dict
            Dictionary of attributes to search for.

        Returns
        -------
        str
            Constructed XPath expression.
        """
    # 属性字典，用于存储 HTML 属性。这些属性不会进行有效性检查。
    attrs : dict
        A dict of HTML attributes. These are NOT checked for validity.
    
    # 返回
    # -------
    # expr : unicode
    #     返回一个 XPath 表达式，用于检查给定的 HTML 属性。
    """
    # 如果属性字典中包含 "class_" 键，则改为 "class" 键，因为 "class" 是 Python 的关键字
    if "class_" in attrs:
        attrs["class"] = attrs.pop("class_")
    
    # 创建一个 XPath 表达式字符串，将属性字典中的键值对转换为 XPath 属性选择器
    s = " and ".join([f"@{k}={v!r}" for k, v in attrs.items()])
    
    # 返回最终的 XPath 表达式，用于选取具有给定 HTML 属性的元素
    return f"[{s}]"
_re_namespace = {"re": "http://exslt.org/regular-expressions"}



# 定义一个命名空间字典，用于 XPath 表达式中的正则表达式扩展
_re_namespace = {"re": "http://exslt.org/regular-expressions"}



class _LxmlFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses lxml under the hood.

    Warning
    -------
    This parser can only handle HTTP, FTP, and FILE urls.

    See Also
    --------
    _HtmlFrameParser
    _BeautifulSoupLxmlFrameParser

    Notes
    -----
    Documentation strings for this class are in the base class
    :class:`_HtmlFrameParser`.
    """

    def _href_getter(self, obj) -> str | None:
        # 获取对象内部所有<a>标签的href属性值
        href = obj.xpath(".//a/@href")
        return None if not href else href[0]



        return obj.text_content()



    def _parse_td(self, row):
        # 仅查找直接子节点：这里的"row"元素可能是<thead>或<tfoot>（见_parse_thead_tr）。
        return row.xpath("./td|./th")



    def _parse_tables(self, document, match, kwargs):
        pattern = match.pattern

        # 1. 检查所有后代节点，寻找匹配给定模式的表格
        # GH 49929
        xpath_expr = f"//table[.//text()[re:test(., {pattern!r})]]"



        # 如果有任何表格属性被给出，则构建一个XPath表达式来搜索它们
        if kwargs:
            xpath_expr += _build_xpath_expr(kwargs)



        # 在文档中根据XPath表达式搜索匹配的表格
        tables = document.xpath(xpath_expr, namespaces=_re_namespace)



        tables = self._handle_hidden_tables(tables, "attrib")
        if self.displayed_only:
            for table in tables:
                # lxml使用XPath 1.0，不支持正则表达式。
                # 因此，我们找到所有具有style属性的元素，并迭代检查是否具有display:none
                for elem in table.xpath(".//style"):
                    elem.drop_tree()
                for elem in table.xpath(".//*[@style]"):
                    if "display:none" in elem.attrib.get("style", "").replace(" ", ""):
                        elem.drop_tree()



        # 如果未找到匹配模式的表格，则抛出值错误异常
        if not tables:
            raise ValueError(f"No tables found matching regex {pattern!r}")
        return tables



    def _equals_tag(self, obj, tag) -> bool:
        # 检查对象的标签是否与指定标签相等
        return obj.tag == tag
    def _build_doc(self):
        """
        Raises
        ------
        ValueError
            * If a URL that lxml cannot parse is passed.

        Exception
            * Any other ``Exception`` thrown. For example, trying to parse a
              URL that is syntactically correct on a machine with no internet
              connection will fail.

        See Also
        --------
        pandas.io.html._HtmlFrameParser._build_doc
        """
        from lxml.etree import XMLSyntaxError  # 导入 lxml 库中的 XMLSyntaxError 异常
        from lxml.html import (  # 导入 lxml 库中的 HTMLParser 和 parse 函数
            HTMLParser,
            parse,
        )

        parser = HTMLParser(recover=True, encoding=self.encoding)  # 创建一个 HTML 解析器对象

        if is_url(self.io):  # 如果 self.io 是一个 URL
            with get_handle(self.io, "r", storage_options=self.storage_options) as f:
                r = parse(f.handle, parser=parser)  # 解析 URL 对应的 HTML 内容
        else:
            # 否则尝试以最简单的方式解析输入
            try:
                r = parse(self.io, parser=parser)  # 解析 self.io 对应的 HTML 内容
            except OSError as err:
                raise FileNotFoundError(  # 如果解析失败，抛出 FileNotFoundError 异常
                    f"[Errno {errno.ENOENT}] {os.strerror(errno.ENOENT)}: {self.io}"
                ) from err

        try:
            r = r.getroot()  # 获取解析后的 HTML 根节点
        except AttributeError:
            pass  # 如果无法获取根节点，继续执行
        else:
            if not hasattr(r, "text_content"):  # 如果根节点没有 text_content 属性
                raise XMLSyntaxError("no text parsed from document", 0, 0, 0)  # 抛出 XMLSyntaxError 异常

        for br in r.xpath("*//br"):  # 遍历所有 <br> 标签
            br.tail = "\n" + (br.tail or "")  # 在每个 <br> 标签的尾部添加换行符

        return r  # 返回解析后的 HTML 根节点

    def _parse_thead_tr(self, table):
        rows = []

        for thead in table.xpath(".//thead"):  # 遍历表格中的所有 <thead> 元素
            rows.extend(thead.xpath("./tr"))  # 将每个 <thead> 元素下的所有 <tr> 元素添加到 rows 列表中

            # HACK: lxml does not clean up the clearly-erroneous
            # <thead><th>foo</th><th>bar</th></thead>. (Missing <tr>). Add
            # the <thead> and _pretend_ it's a <tr>; _parse_td() will find its
            # children as though it's a <tr>.
            #
            # Better solution would be to use html5lib.
            elements_at_root = thead.xpath("./td|./th")
            if elements_at_root:
                rows.append(thead)  # 如果 <thead> 下有直接的 <td> 或 <th> 元素，将整个 <thead> 当作 <tr> 处理

        return rows  # 返回解析后的 <thead> 下的所有 <tr> 元素列表

    def _parse_tbody_tr(self, table):
        from_tbody = table.xpath(".//tbody//tr")  # 获取表格中所有 <tbody> 下的 <tr> 元素
        from_root = table.xpath("./tr")  # 获取表格中根级别的 <tr> 元素
        # HTML spec: at most one of these lists has content
        return from_tbody + from_root  # 返回所有解析到的 <tbody> 下的 <tr> 元素和根级别的 <tr> 元素列表的合并结果

    def _parse_tfoot_tr(self, table):
        return table.xpath(".//tfoot//tr")  # 返回表格中所有 <tfoot> 下的 <tr> 元素列表
def _expand_elements(body) -> None:
    # 创建一个列表，包含每个元素的长度
    data = [len(elem) for elem in body]
    # 将长度列表转换为 Pandas Series 对象
    lens = Series(data)
    # 计算列表中元素的最大长度
    lens_max = lens.max()
    # 找出长度不是最大值的元素索引
    not_max = lens[lens != lens_max]

    # 创建一个包含空字符串的列表
    empty = [""]
    # 将不是最大长度的元素填充空字符串，使它们的长度等于最大长度
    for ind, length in not_max.items():
        body[ind] += empty * (lens_max - length)


def _data_to_frame(**kwargs):
    # 从传入参数中取出 head、body、foot 和 header
    head, body, foot = kwargs.pop("data")
    header = kwargs.pop("header")
    # 处理 skiprows 参数，确保它是一个列表
    kwargs["skiprows"] = _get_skiprows(kwargs["skiprows"])

    # 如果存在 head，将其与 body 连接
    if head:
        body = head + body

        # 当没有指定 header 时，根据情况推断是否有表头行
        if header is None:
            if len(head) == 1:
                header = 0
            else:
                # 忽略所有全为空文本行
                header = [i for i, row in enumerate(head) if any(text for text in row)]

    # 如果存在 foot，将其添加到 body
    if foot:
        body += foot

    # 填充 body 中长度不一致的元素
    _expand_elements(body)
    # 使用 TextParser 类处理 body，返回结果
    with TextParser(body, header=header, **kwargs) as tp:
        return tp.read()


_valid_parsers = {
    "lxml": _LxmlFrameParser,
    None: _LxmlFrameParser,
    "html5lib": _BeautifulSoupHtml5LibFrameParser,
    "bs4": _BeautifulSoupHtml5LibFrameParser,
}


def _parser_dispatch(flavor: HTMLFlavors | None) -> type[_HtmlFrameParser]:
    """
    根据输入的 flavor 选择相应的解析器。

    Parameters
    ----------
    flavor : {{"lxml", "html5lib", "bs4"}} or None
        使用的解析器类型。必须是有效的后端。

    Returns
    -------
    cls : _HtmlFrameParser 子类
        基于请求的 flavor 返回的解析器类。

    Raises
    ------
    ValueError
        * 如果 `flavor` 不是有效的后端。
    ImportError
        * 如果请求的 `flavor` 未安装。
    """
    valid_parsers = list(_valid_parsers.keys())
    # 如果 flavor 不在有效解析器列表中，抛出 ValueError 异常
    if flavor not in valid_parsers:
        raise ValueError(
            f"{flavor!r} is not a valid flavor, valid flavors are {valid_parsers}"
        )

    # 根据 flavor 导入相应的依赖
    if flavor in ("bs4", "html5lib"):
        import_optional_dependency("html5lib")
        import_optional_dependency("bs4")
    else:
        import_optional_dependency("lxml.etree")
    
    # 返回与 flavor 对应的解析器类
    return _valid_parsers[flavor]


def _print_as_set(s) -> str:
    # 将集合 s 中的每个元素转换为字符串并用逗号分隔，返回格式化后的字符串
    arg = ", ".join([pprint_thing(el) for el in s])
    return f"{{{arg}}}"


def _validate_flavor(flavor):
    # 如果 flavor 为 None，默认设置为 ("lxml", "bs4")
    if flavor is None:
        flavor = "lxml", "bs4"
    # 如果 flavor 是字符串，转换为元组
    elif isinstance(flavor, str):
        flavor = (flavor,)
    # 如果 flavor 是可迭代对象，确保所有元素都是字符串类型
    elif isinstance(flavor, abc.Iterable):
        if not all(isinstance(flav, str) for flav in flavor):
            raise TypeError(
                f"Object of type {type(flavor).__name__!r} "
                f"is not an iterable of strings"
            )
    else:
        # 如果 flavor 不是字符串或可迭代对象，抛出 ValueError 异常
        msg = repr(flavor) if isinstance(flavor, str) else str(flavor)
        msg += " is not a valid flavor"
        raise ValueError(msg)

    # 将 flavor 转换为元组并验证所有元素都是有效的解析器
    flavor = tuple(flavor)
    valid_flavors = set(_valid_parsers)
    flavor_set = set(flavor)
    # 如果 flavor_set 和 valid_flavors 的交集为空集
    if not flavor_set & valid_flavors:
        # 抛出 ValueError 异常，提示 flavor_set 不是有效的一组口味集合，显示有效口味集合
        raise ValueError(
            f"{_print_as_set(flavor_set)} is not a valid set of flavors, valid "
            f"flavors are {_print_as_set(valid_flavors)}"
        )
    # 返回 flavor 变量作为结果
    return flavor
# 定义一个私有函数 `_parse`，用于解析 HTML 或类似格式的数据并返回解析后的数据框列表
def _parse(
    flavor,  # 解析器的类型或风格，确定解析方式
    io,  # 输入数据流，可以是文件路径或读取缓冲区
    match,  # 匹配表格的正则表达式字符串或模式
    attrs,  # HTML 标签的属性字典，用于过滤解析的表格
    encoding,  # 数据编码方式
    displayed_only,  # 是否只解析可见的表格
    extract_links,  # 是否提取表格中的链接信息
    storage_options,  # 存储选项字典，用于处理输入流的额外参数
    **kwargs,  # 其他可选参数，用于传递给特定解析器
):
    # 确认并返回有效的解析器类型或风格
    flavor = _validate_flavor(flavor)
    
    # 编译正则表达式，用于匹配表格内容
    compiled_match = re.compile(match)  # 可以传入预编译的正则表达式对象
    
    retained = None
    # 遍历所有解析器类型或风格
    for flav in flavor:
        # 根据解析器类型获取相应的解析器对象
        parser = _parser_dispatch(flav)
        
        # 使用解析器对象对输入数据进行解析
        p = parser(
            io,
            compiled_match,
            attrs,
            encoding,
            displayed_only,
            extract_links,
            storage_options,
        )
        
        try:
            # 尝试解析表格数据
            tables = p.parse_tables()
        except ValueError as caught:
            # 如果输入数据流是类似文件对象且可回溯，则尝试回溯流以尝试下一个解析器
            if hasattr(io, "seekable") and io.seekable():
                io.seek(0)
            # 如果输入数据流无法回溯，则抛出错误提示用户无法尝试其他解析器
            elif hasattr(io, "seekable") and not io.seekable():
                raise ValueError(
                    f"The flavor {flav} failed to parse your input. "
                    "Since you passed a non-rewindable file "
                    "object, we can't rewind it to try "
                    "another parser. Try read_html() with a different flavor."
                ) from caught

            retained = caught
        else:
            break
    else:
        # 如果所有解析器均未成功解析，则断言并抛出最后一个捕获的异常
        assert retained is not None
        raise retained

    ret = []
    # 遍历所有成功解析的表格数据
    for table in tables:
        try:
            # 将表格数据转换为数据框，并应用额外参数 kwargs
            df = _data_to_frame(data=table, **kwargs)
            
            # 当提取表格链接信息时，将多级索引标题转换为元组索引
            if extract_links in ("all", "header") and isinstance(
                df.columns, MultiIndex
            ):
                df.columns = Index(
                    ((col[0], None if isna(col[1]) else col[1]) for col in df.columns),
                    tupleize_cols=False,
                )

            ret.append(df)  # 将处理后的数据框添加到结果列表中
        except EmptyDataError:  # 处理空表格的异常情况
            continue  # 忽略空表格，继续处理下一个表格

    return ret  # 返回所有解析后的数据框列表
    `
        # 设置数据类型后端，默认为 lib.no_default，类型为 DtypeBackend 或 lib.NoDefault
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        # 设置存储选项，默认为 None，类型为 StorageOptions
        storage_options: StorageOptions = None,
    ) -> list[DataFrame]:
    r"""
    Read HTML tables into a ``list`` of ``DataFrame`` objects.

    Parameters
    ----------
    io : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a string ``read()`` function.
        The string can represent a URL. Note that
        lxml only accepts the http, ftp and file url protocols. If you have a
        URL that starts with ``'https'`` you might try removing the ``'s'``.

        .. deprecated:: 2.1.0
            Passing html literal strings is deprecated.
            Wrap literal string/bytes input in ``io.StringIO``/``io.BytesIO`` instead.
    
    match : str or compiled regular expression, optional
        The set of tables containing text matching this regex or string will be
        returned. Unless the HTML is extremely simple you will probably need to
        pass a non-empty string here. Defaults to '.+' (match any non-empty
        string). The default value will return all tables contained on a page.
        This value is converted to a regular expression so that there is
        consistent behavior between Beautiful Soup and lxml.
    
    flavor : {{"lxml", "html5lib", "bs4"}} or list-like, optional
        The parsing engine (or list of parsing engines) to use. 'bs4' and
        'html5lib' are synonymous with each other, they are both there for
        backwards compatibility. The default of ``None`` tries to use ``lxml``
        to parse and if that fails it falls back on ``bs4`` + ``html5lib``.
    
    header : int or list-like, optional
        The row (or list of rows for a :class:`~pandas.MultiIndex`) to use to
        make the columns headers.
    
    index_col : int or list-like, optional
        The column (or list of columns) to use to create the index.
    
    skiprows : int, list-like or slice, optional
        Number of rows to skip after parsing the column integer. 0-based. If a
        sequence of integers or a slice is given, will skip the rows indexed by
        that sequence.  Note that a single element sequence means 'skip the nth
        row' whereas an integer means 'skip n rows'.
    """
    # attrs : dict, optional
    # 这是一个字典，包含用于在HTML中标识表格的属性。这些属性在传递给lxml或Beautiful Soup之前不会进行有效性检查。
    # 然而，这些属性必须是有效的HTML表格属性才能正确工作。
    # 例如，attrs = {"id": "table"} 是一个有效的属性字典，因为 'id' HTML标签属性是任何HTML标签的有效属性，
    # 参考这个文档 <https://html.spec.whatwg.org/multipage/dom.html#global-attributes>。
    # 而 attrs = {"asdf": "table"} 不是一个有效的属性字典，因为 'asdf' 不是有效的HTML属性，即使它是有效的XML属性。
    # HTML 4.01 的有效表格属性可以在这里找到 <http://www.w3.org/TR/REC-html40/struct/tables.html#h-11.2>。
    # HTML 5 规范草案可以在这里找到 <https://html.spec.whatwg.org/multipage/tables.html>，它包含了现代Web的最新表格属性信息。

    # parse_dates : bool, optional
    # 参见 read_csv 函数获取更多细节。

    # thousands : str, optional
    # 用于解析千位分隔符的分隔符。默认为 ','。

    # encoding : str, optional
    # 用于解码网页的编码方式。默认为 None。None 会保留先前的编码行为，依赖于底层解析库（例如，解析库将尝试使用文档提供的编码方式）。

    # decimal : str, default '.'
    # 用于识别十进制点的字符（例如，对于欧洲数据可以使用 ','）。

    # converters : dict, default None
    # 用于转换特定列中值的函数字典。键可以是整数或列标签，值是接受一个输入参数（单元格内容）并返回转换后内容的函数。

    # na_values : iterable, default None
    # 自定义的NA值（缺失值）列表。

    # keep_default_na : bool, default True
    # 如果指定了na_values并且keep_default_na为False，则默认的NaN值将被覆盖，否则它们将被附加到列表中。

    # displayed_only : bool, default True
    # 是否解析具有"display: none"的元素。

    # extract_links : {None, "all", "header", "body", "footer"}
    # 在指定的部分中，具有 <a> 标签的表格元素将提取其 href 属性。
    # .. versionadded:: 1.5.0
    # 检查 skiprows 是否为整数且非负，避免因为无效的 skiprows 值导致解析失败
    if isinstance(skiprows, numbers.Integral) and skiprows < 0:
        # 抛出数值错误，提示不能从数据末尾开始跳过行（传入了负值）
        raise ValueError(
            "cannot skip rows starting from the end of the "
            "data (you passed a negative value)"
        )
    
    # 检查 extract_links 是否为预期的取值之一
    if extract_links not in [None, "header", "footer", "body", "all"]:
        # 抛出数值错误，提示 extract_links 必须是 {None, "header", "footer", "body", "all"} 中的一个
        raise ValueError(
            "`extract_links` must be one of "
            '{None, "header", "footer", "body", "all"}, got '
            f'"{extract_links}"'
        )
    
    # 验证 header 参数的有效性
    validate_header_arg(header)
    
    # 检查并确保 dtype_backend 参数的有效性
    check_dtype_backend(dtype_backend)
    
    # 将 io 参数转换为字符串表示形式的路径
    io = stringify_path(io)
    # 调用_parse函数，并传递以下参数：
    # flavor: 数据解析的风格
    # io: 输入数据的源
    # match: 匹配解析数据的规则
    # header: 指定作为列名的行号，默认为0（第一行）
    # index_col: 指定作为行索引的列编号或列名
    # skiprows: 需要跳过的行数
    # parse_dates: 尝试解析为日期的列
    # thousands: 千位分隔符
    # attrs: 需要解析的属性
    # encoding: 文件的字符编码格式
    # decimal: 使用的小数点符号
    # converters: 要传递给解析器的列转换器的映射
    # na_values: 要替换为NA的值列表
    # keep_default_na: 是否保留默认的NA值列表
    # displayed_only: 仅显示的列
    # extract_links: 是否提取链接
    # dtype_backend: 后端的数据类型
    # storage_options: 存储选项
    return _parse(
        flavor=flavor,
        io=io,
        match=match,
        header=header,
        index_col=index_col,
        skiprows=skiprows,
        parse_dates=parse_dates,
        thousands=thousands,
        attrs=attrs,
        encoding=encoding,
        decimal=decimal,
        converters=converters,
        na_values=na_values,
        keep_default_na=keep_default_na,
        displayed_only=displayed_only,
        extract_links=extract_links,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
    )
```