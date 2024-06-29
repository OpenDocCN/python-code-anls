# `D:\src\scipysrc\pandas\pandas\io\formats\xml.py`

```
"""
:mod:`pandas.io.formats.xml` is a module for formatting data in XML.
"""

# 引入未来的注释语法以支持类型注释
from __future__ import annotations

# 引入所需的模块和类型
import codecs
import io
from typing import (
    TYPE_CHECKING,
    Any,
    final,
)

# 引入 pandas 的错误和装饰器模块
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

# 引入 pandas 的数据类型判断和缺失值处理模块
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna

# 引入 pandas 的共享文档和 IO 模块
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import get_data_from_filepath

# 如果是类型检查模式，引入额外的类型
if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )
    from pandas import DataFrame

# 对 _BaseXMLFormatter 类进行文档化
@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "path_or_buffer",
)
class _BaseXMLFormatter:
    """
    Subclass for formatting data in XML.

    Parameters
    ----------
    path_or_buffer : str or file-like
        This can be either a string of raw XML, a valid URL,
        file or file-like object.

    index : bool
        Whether to include index in xml document.

    row_name : str
        Name for root of xml document. Default is 'data'.

    root_name : str
        Name for row elements of xml document. Default is 'row'.

    na_rep : str
        Missing data representation.

    attrs_cols : list
        List of columns to write as attributes in row element.

    elem_cols : list
        List of columns to write as children in row element.

    namespaces : dict
        The namespaces to define in XML document as dicts with key
        being namespace and value the URI.

    prefix : str
        The prefix for each element in XML document including root.

    encoding : str
        Encoding of xml object or document.

    xml_declaration : bool
        Whether to include xml declaration at top line item in xml.

    pretty_print : bool
        Whether to write xml document with line breaks and indentation.

    stylesheet : str or file-like
        A URL, file, file-like object, or a raw string containing XSLT.

    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    See also
    --------
    pandas.io.formats.xml.EtreeXMLFormatter
    pandas.io.formats.xml.LxmlXMLFormatter

    """
    def __init__(
        self,
        frame: DataFrame,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        index: bool = True,
        root_name: str | None = "data",
        row_name: str | None = "row",
        na_rep: str | None = None,
        attr_cols: list[str] | None = None,
        elem_cols: list[str] | None = None,
        namespaces: dict[str | None, str] | None = None,
        prefix: str | None = None,
        encoding: str = "utf-8",
        xml_declaration: bool | None = True,
        pretty_print: bool | None = True,
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
    ) -> None:
        """
        Initialize XMLConverter object.

        Parameters:
        - frame: DataFrame to be converted.
        - path_or_buffer: Path or buffer to write XML data to.
        - index: Whether to include DataFrame index in XML output.
        - root_name: Root element name in XML.
        - row_name: Name for each row element in XML.
        - na_rep: String representation of NA/NaN values.
        - attr_cols: Columns to be treated as XML attributes.
        - elem_cols: Columns to be treated as XML elements.
        - namespaces: XML namespaces dictionary.
        - prefix: Prefix to be added to all XML elements.
        - encoding: Encoding format for XML output.
        - xml_declaration: Whether to include XML declaration.
        - pretty_print: Whether to prettify XML output.
        - stylesheet: Stylesheet file for XML transformation.
        - compression: Compression options for XML output.
        - storage_options: Options for storage operations.

        Returns:
        None
        """
        self.frame = frame
        self.path_or_buffer = path_or_buffer
        self.index = index
        self.root_name = root_name
        self.row_name = row_name
        self.na_rep = na_rep
        self.attr_cols = attr_cols
        self.elem_cols = elem_cols
        self.namespaces = namespaces
        self.prefix = prefix
        self.encoding = encoding
        self.xml_declaration = xml_declaration
        self.pretty_print = pretty_print
        self.stylesheet = stylesheet
        self.compression: CompressionOptions = compression
        self.storage_options = storage_options

        # Store original columns of the DataFrame
        self.orig_cols = self.frame.columns.tolist()

        # Process the DataFrame to create a list of dictionaries
        self.frame_dicts = self._process_dataframe()

        # Validate attribute and element columns
        self._validate_columns()

        # Validate encoding
        self._validate_encoding()

        # Determine prefix URI for namespaces
        self.prefix_uri = self._get_prefix_uri()

        # Handle indexes in the DataFrame
        self._handle_indexes()

    def _build_tree(self) -> bytes:
        """
        Build XML tree from data.

        This method initializes the root of the XML tree and constructs
        attributes and elements based on the DataFrame columns and data.

        Raises:
        AbstractMethodError: This method is meant to be overridden by subclasses.
        """
        raise AbstractMethodError(self)

    @final
    def _validate_columns(self) -> None:
        """
        Validate elem_cols and attr_cols parameters.

        This method ensures that elem_cols and attr_cols are list-like.

        Raises:
        TypeError: If elem_cols or attr_cols are not list-like.
        """
        if self.attr_cols and not is_list_like(self.attr_cols):
            raise TypeError(
                f"{type(self.attr_cols).__name__} is not a valid type for attr_cols"
            )

        if self.elem_cols and not is_list_like(self.elem_cols):
            raise TypeError(
                f"{type(self.elem_cols).__name__} is not a valid type for elem_cols"
            )

    @final
    def _validate_encoding(self) -> None:
        """
        Validate encoding parameter.

        This method checks if the specified encoding is available in the codecs module.

        Raises:
        LookupError: If the specified encoding is not found in the codecs.
        """
        codecs.lookup(self.encoding)
    def _process_dataframe(self) -> dict[int | str, dict[str, Any]]:
        """
        Adjust Data Frame to fit xml output.

        This method will adjust underlying data frame for xml output,
        including optionally replacing missing values and including indexes.
        """

        df = self.frame  # 将类属性 self.frame 赋给局部变量 df

        if self.index:  # 如果设置了 index 标志为 True
            df = df.reset_index()  # 重置数据框索引

        if self.na_rep is not None:  # 如果设置了缺失值替换标志
            df = df.fillna(self.na_rep)  # 使用指定的值填充缺失值

        return df.to_dict(orient="index")  # 将数据框转换为字典格式，并按行索引方式返回

    @final
    def _handle_indexes(self) -> None:
        """
        Handle indexes.

        This method will add indexes into attr_cols or elem_cols.
        """

        if not self.index:  # 如果 index 标志为 False，则直接返回
            return

        first_key = next(iter(self.frame_dicts))  # 获取第一个 frame_dicts 中的键值
        indexes: list[str] = [
            x for x in self.frame_dicts[first_key].keys() if x not in self.orig_cols
        ]  # 在 frame_dicts 的第一个键对应的字典中，找出不在 orig_cols 中的键作为索引列表

        if self.attr_cols:  # 如果 attr_cols 非空
            self.attr_cols = indexes + self.attr_cols  # 将索引列表与原有的 attr_cols 合并

        if self.elem_cols:  # 如果 elem_cols 非空
            self.elem_cols = indexes + self.elem_cols  # 将索引列表与原有的 elem_cols 合并

    def _get_prefix_uri(self) -> str:
        """
        Get uri of namespace prefix.

        This method retrieves corresponding URI to prefix in namespaces.

        Raises
        ------
        KeyError
            *If prefix is not included in namespace dict.
        """

        raise AbstractMethodError(self)  # 抛出抽象方法错误，子类需要实现具体逻辑

    @final
    def _other_namespaces(self) -> dict:
        """
        Define other namespaces.

        This method will build dictionary of namespaces attributes
        for root element, conditionally with optional namespaces and
        prefix.
        """

        nmsp_dict: dict[str, str] = {}  # 初始化命名空间字典为空字典
        if self.namespaces:  # 如果存在命名空间属性
            nmsp_dict = {
                f"xmlns{p if p=='' else f':{p}'}": n
                for p, n in self.namespaces.items()
                if n != self.prefix_uri[1:-1]
            }  # 构建命名空间属性字典，条件是命名空间不等于去掉前后引号的 self.prefix_uri

        return nmsp_dict  # 返回命名空间属性字典

    @final
    def _build_attribs(self, d: dict[str, Any], elem_row: Any) -> Any:
        """
        Create attributes of row.

        This method adds attributes using attr_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """

        if not self.attr_cols:  # 如果 attr_cols 为空列表
            return elem_row  # 直接返回元素行

        for col in self.attr_cols:  # 遍历 attr_cols 中的列名
            attr_name = self._get_flat_col_name(col)  # 获取平面化后的列名
            try:
                if not isna(d[col]):  # 如果列名在字典 d 中并且不为空
                    elem_row.attrib[attr_name] = str(d[col])  # 将列值转换为字符串并作为属性值赋给 elem_row
            except KeyError as err:
                raise KeyError(f"no valid column, {col}") from err  # 如果列名不在字典 d 中，抛出 KeyError

        return elem_row  # 返回处理后的 elem_row

    @final
    def _get_flat_col_name(self, col: str | tuple) -> str:
        """
        Get flat column name.

        This method returns flat column name by concatenating elements
        in tuple if provided.

        Parameters
        ----------
        col : str | tuple
            Column name or tuple of column names.

        Returns
        -------
        str
            Flat column name.
        """

        flat_col = col  # 初始化 flat_col 为输入的列名或列名元组
        if isinstance(col, tuple):  # 如果输入是元组
            flat_col = (
                "".join([str(c) for c in col]).strip()
                if "" in col
                else "_".join([str(c) for c in col]).strip()
            )  # 将元组中的元素连接为字符串，使用下划线分隔或合并，并去除首尾空格

        return f"{self.prefix_uri}{flat_col}"  # 返回加上前缀的平面化列名
    # 定义抽象方法，子类需要实现该方法
    def _sub_element_cls(self):
        raise AbstractMethodError(self)

    # 最终方法修饰器，确保方法不会被子类覆盖
    @final
    def _build_elems(self, d: dict[str, Any], elem_row: Any) -> None:
        """
        创建行的子元素。

        该方法使用 elem_cols 向行元素添加子元素，并处理多级索引或层次化列使用元组的情况。
        """
        sub_element_cls = self._sub_element_cls

        # 如果 elem_cols 为空，则直接返回
        if not self.elem_cols:
            return

        # 遍历 elem_cols 中的列名
        for col in self.elem_cols:
            # 获取扁平化后的列名作为元素名
            elem_name = self._get_flat_col_name(col)
            try:
                # 从字典 d 中获取列 col 对应的值，如果为 NaN 或空字符串，则置为 None
                val = None if isna(d[col]) or d[col] == "" else str(d[col])
                # 使用 sub_element_cls 创建子元素，并将值 val 设置为子元素的文本内容
                sub_element_cls(elem_row, elem_name).text = val
            except KeyError as err:
                # 如果 d 中不存在列 col，则抛出 KeyError 异常
                raise KeyError(f"no valid column, {col}") from err

    # 最终方法修饰器，确保方法不会被子类覆盖
    @final
    def write_output(self) -> str | None:
        # 调用 _build_tree 方法构建 XML 文档
        xml_doc = self._build_tree()

        # 如果指定了输出路径或缓冲区
        if self.path_or_buffer is not None:
            # 使用 get_handle 打开指定路径或缓冲区的句柄，以二进制写入模式
            with get_handle(
                self.path_or_buffer,
                "wb",
                compression=self.compression,
                storage_options=self.storage_options,
                is_text=False,
            ) as handles:
                # 将 XML 文档写入句柄中
                handles.handle.write(xml_doc)
            return None  # 返回 None 表示成功写入

        else:
            # 如果没有指定路径或缓冲区，则将 XML 文档解码为字符串并去除末尾的空白字符后返回
            return xml_doc.decode(self.encoding).rstrip()
# 定义一个 XML 格式化器类，继承自 _BaseXMLFormatter
class EtreeXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    # 构建 XML 树的方法，返回字节流
    def _build_tree(self) -> bytes:
        # 导入 ElementTree 模块的相关函数和类
        from xml.etree.ElementTree import (
            Element,
            SubElement,
            tostring,
        )

        # 创建根元素，带有命名空间前缀和根元素名，以及其他命名空间属性
        self.root = Element(
            f"{self.prefix_uri}{self.root_name}", attrib=self._other_namespaces()
        )

        # 遍历 frame_dicts 中的值
        for d in self.frame_dicts.values():
            # 创建行元素
            elem_row = SubElement(self.root, f"{self.prefix_uri}{self.row_name}")

            # 如果未指定属性列和元素列，则从字典中获取列名并构建元素
            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)
            else:
                # 否则先构建属性，再构建元素
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)

        # 将整个 XML 树转换为字节流
        self.out_xml = tostring(
            self.root,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )

        # 如果需要美化输出 XML 树
        if self.pretty_print:
            self.out_xml = self._prettify_tree()

        # 如果指定了样式表，但是未安装 lxml 解析器时，抛出 ValueError 异常
        if self.stylesheet is not None:
            raise ValueError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )

        # 返回格式化后的 XML 字节流
        return self.out_xml

    # 获取命名空间前缀的 URI
    def _get_prefix_uri(self) -> str:
        # 导入 register_namespace 函数
        from xml.etree.ElementTree import register_namespace

        uri = ""
        # 如果存在命名空间
        if self.namespaces:
            # 遍历命名空间字典
            for p, n in self.namespaces.items():
                # 如果命名空间和前缀都是字符串类型
                if isinstance(p, str) and isinstance(n, str):
                    # 注册命名空间
                    register_namespace(p, n)
            # 如果有指定默认前缀，获取对应的命名空间 URI
            if self.prefix:
                try:
                    uri = f"{{{self.namespaces[self.prefix]}}}"
                except KeyError as err:
                    raise KeyError(
                        f"{self.prefix} is not included in namespaces"
                    ) from err
            # 否则获取空命名空间的 URI
            elif "" in self.namespaces:
                uri = f'{{{self.namespaces[""]}}}'
            else:
                uri = ""

        # 返回命名空间 URI
        return uri

    # 缓存装饰器，返回 SubElement 类
    @cache_readonly
    def _sub_element_cls(self):
        # 导入 SubElement 类
        from xml.etree.ElementTree import SubElement

        return SubElement

    # 美化 XML 树的输出格式，返回字节流
    def _prettify_tree(self) -> bytes:
        """
        Output tree for pretty print format.

        This method will pretty print xml with line breaks and indentation.
        """

        # 导入 parseString 函数
        from xml.dom.minidom import parseString

        # 解析 XML 字符串并进行美化处理
        dom = parseString(self.out_xml)

        # 返回美化后的 XML 字节流
        return dom.toprettyxml(indent="  ", encoding=self.encoding)


# 定义一个 XML 格式化器类，继承自 _BaseXMLFormatter
class LxmlXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    # 初始化方法，调用父类的初始化方法，并执行 _convert_empty_str_key 方法
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._convert_empty_str_key()
    # 定义一个私有方法 `_build_tree`，返回类型为 bytes
    def _build_tree(self) -> bytes:
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """

        # 导入 lxml.etree 库中的 Element, SubElement, tostring 函数
        from lxml.etree import (
            Element,
            SubElement,
            tostring,
        )

        # 使用指定的命名空间映射创建根元素 self.root
        self.root = Element(f"{self.prefix_uri}{self.root_name}", nsmap=self.namespaces)

        # 遍历 self.frame_dicts 中的值
        for d in self.frame_dicts.values():
            # 在 self.root 下创建子元素 elem_row
            elem_row = SubElement(self.root, f"{self.prefix_uri}{self.row_name}")

            # 如果未定义属性列和元素列，则从字典 d 的键生成元素列并调用 _build_elems 方法
            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)
            else:
                # 否则调用 _build_attribs 方法生成属性列，并调用 _build_elems 方法生成元素列
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)

        # 将 self.root 转换为 XML 字符串并赋值给 self.out_xml
        self.out_xml = tostring(
            self.root,
            pretty_print=self.pretty_print,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )

        # 如果定义了样式表 stylesheet，则对 self.out_xml 进行转换
        if self.stylesheet is not None:
            self.out_xml = self._transform_doc()

        # 返回生成的 XML 字符串 self.out_xml
        return self.out_xml

    # 定义一个私有方法 `_convert_empty_str_key`，返回类型为 None
    def _convert_empty_str_key(self) -> None:
        """
        Replace zero-length string in `namespaces`.

        This method will replace '' with None to align to `lxml`
        requirement that empty string prefixes are not allowed.
        """

        # 如果 self.namespaces 存在且包含空字符串键 ''
        if self.namespaces and "" in self.namespaces.keys():
            # 将空字符串键 '' 替换为 None，以符合 lxml 要求空字符串前缀不允许的规定
            self.namespaces[None] = self.namespaces.pop("", "default")

    # 定义一个私有方法 `_get_prefix_uri`，返回类型为 str
    def _get_prefix_uri(self) -> str:
        uri = ""
        # 如果 self.namespaces 存在
        if self.namespaces:
            # 如果指定了前缀 self.prefix
            if self.prefix:
                try:
                    # 获取 self.prefix 对应的命名空间 URI，并格式化为 {URI} 形式
                    uri = f"{{{self.namespaces[self.prefix]}}}"
                except KeyError as err:
                    # 抛出 KeyError 异常，提示指定的前缀不在命名空间中
                    raise KeyError(
                        f"{self.prefix} is not included in namespaces"
                    ) from err
            # 如果命名空间中存在空字符串键 ''
            elif "" in self.namespaces:
                # 直接获取空字符串键对应的命名空间 URI，并格式化为 {URI} 形式
                uri = f'{{{self.namespaces[""]}}}'
            else:
                uri = ""

        # 返回获取的命名空间 URI
        return uri

    # 定义一个缓存只读属性 `_sub_element_cls`
    @cache_readonly
    def _sub_element_cls(self):
        # 导入 lxml.etree 库中的 SubElement 类
        from lxml.etree import SubElement

        # 返回 SubElement 类作为缓存只读属性值
        return SubElement
    def _transform_doc(self) -> bytes:
        """
        Parse stylesheet from file or buffer and run it.

        This method will parse a stylesheet object into a tree for parsing
        based on its specific object type, then transform
        the original tree with an XSLT script.
        """
        from lxml.etree import (
            XSLT,          # 导入 lxml 库中的 XSLT 模块，用于执行 XSLT 转换
            XMLParser,     # 导入 lxml 库中的 XMLParser 类，用于解析 XML
            fromstring,    # 导入 lxml 库中的 fromstring 函数，用于将字符串解析为 XML 对象
            parse,         # 导入 lxml 库中的 parse 函数，用于解析 XML 文件
        )

        style_doc = self.stylesheet
        assert style_doc is not None  # 断言确保 style_doc 不为 None，由调用者保证

        # 调用 get_data_from_filepath 函数获取样式表数据的处理句柄
        handle_data = get_data_from_filepath(
            filepath_or_buffer=style_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        with handle_data as xml_data:  # 使用处理句柄获取的数据，进入上下文管理器
            curr_parser = XMLParser(encoding=self.encoding)

            if isinstance(xml_data, io.StringIO):
                # 如果 xml_data 是 io.StringIO 类型，将其值编码为指定编码并解析为 XML 对象
                xsl_doc = fromstring(
                    xml_data.getvalue().encode(self.encoding), parser=curr_parser
                )
            else:
                # 否则，解析 XML 文件为 XML 对象
                xsl_doc = parse(xml_data, parser=curr_parser)

        transformer = XSLT(xsl_doc)  # 创建 XSLT 转换器对象，传入 XSL 样式表对象
        new_doc = transformer(self.root)  # 使用 XSLT 转换器对根节点进行转换

        return bytes(new_doc)  # 返回转换后的结果字节流
```