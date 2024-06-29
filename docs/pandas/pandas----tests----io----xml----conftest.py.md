# `D:\src\scipysrc\pandas\pandas\tests\io\xml\conftest.py`

```
# 导入 Path 类，用于处理文件路径
from pathlib import Path

# 导入 pytest 模块，用于编写和运行测试用例
import pytest


# 定义一个 pytest fixture，返回 XML 示例目录的 Path 对象
@pytest.fixture
def xml_data_path():
    """
    返回 XML 示例目录的 Path 对象。

    Examples
    --------
    >>> def test_read_xml(xml_data_path):
    ...     pd.read_xml(xml_data_path / "file.xsl")
    """
    return Path(__file__).parent.parent / "data" / "xml"


# 定义一个 pytest fixture，返回 books.xml 示例文件的路径（字符串形式）
@pytest.fixture
def xml_books(xml_data_path, datapath):
    """
    返回 books.xml 示例文件的路径（字符串形式）。

    Examples
    --------
    >>> def test_read_xml(xml_books):
    ...     pd.read_xml(xml_books)
    """
    return datapath(xml_data_path / "books.xml")


# 定义一个 pytest fixture，返回 doc_ch_utf.xml 示例文件的路径（字符串形式）
@pytest.fixture
def xml_doc_ch_utf(xml_data_path, datapath):
    """
    返回 doc_ch_utf.xml 示例文件的路径（字符串形式）。

    Examples
    --------
    >>> def test_read_xml(xml_doc_ch_utf):
    ...     pd.read_xml(xml_doc_ch_utf)
    """
    return datapath(xml_data_path / "doc_ch_utf.xml")


# 定义一个 pytest fixture，返回 baby_names.xml 示例文件的路径（字符串形式）
@pytest.fixture
def xml_baby_names(xml_data_path, datapath):
    """
    返回 baby_names.xml 示例文件的路径（字符串形式）。

    Examples
    --------
    >>> def test_read_xml(xml_baby_names):
    ...     pd.read_xml(xml_baby_names)
    """
    return datapath(xml_data_path / "baby_names.xml")


# 定义一个 pytest fixture，返回 cta_rail_lines.kml 示例文件的路径（字符串形式）
@pytest.fixture
def kml_cta_rail_lines(xml_data_path, datapath):
    """
    返回 cta_rail_lines.kml 示例文件的路径（字符串形式）。

    Examples
    --------
    >>> def test_read_xml(kml_cta_rail_lines):
    ...     pd.read_xml(
    ...         kml_cta_rail_lines,
    ...         xpath=".//k:Placemark",
    ...         namespaces={"k": "http://www.opengis.net/kml/2.2"},
    ...         stylesheet=xsl_flatten_doc,
    ...     )
    """
    return datapath(xml_data_path / "cta_rail_lines.kml")


# 定义一个 pytest fixture，返回 flatten_doc.xsl 示例文件的路径（字符串形式）
@pytest.fixture
def xsl_flatten_doc(xml_data_path, datapath):
    """
    返回 flatten_doc.xsl 示例文件的路径（字符串形式）。

    Examples
    --------
    >>> def test_read_xsl(xsl_flatten_doc, mode):
    ...     with open(
    ...         xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None
    ...     ) as f:
    ...         xsl_obj = f.read()
    """
    return datapath(xml_data_path / "flatten_doc.xsl")


# 定义一个 pytest fixture，返回 row_field_output.xsl 示例文件的路径（字符串形式）
@pytest.fixture
def xsl_row_field_output(xml_data_path, datapath):
    """
    返回 row_field_output.xsl 示例文件的路径（字符串形式）。

    Examples
    --------
    >>> def test_read_xsl(xsl_row_field_output, mode):
    ...     with open(
    ...         xsl_row_field_output, mode, encoding="utf-8" if mode == "r" else None
    ...     ) as f:
    ...         xsl_obj = f.read()
    """
    return datapath(xml_data_path / "row_field_output.xsl")
```