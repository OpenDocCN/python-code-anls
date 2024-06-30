# `D:\src\scipysrc\sympy\sympy\utilities\mathml\__init__.py`

```
"""Module with some functions for MathML, like transforming MathML
content in MathML presentation.

To use this module, you will need lxml.
"""

from pathlib import Path  # 导入 Path 类，用于处理文件路径

from sympy.utilities.decorator import doctest_depends_on  # 导入 SymPy 提供的装饰器


__doctest_requires__ = {('apply_xsl', 'c2p'): ['lxml']}  # 指定 doctest 需要的额外模块


def add_mathml_headers(s):
    """Adds MathML headers to a MathML string.

    Parameters
    ==========

    s
        A string containing MathML content.

    Returns
    =======

    str
        A string with MathML content wrapped in MathML headers.
    """
    return """<math xmlns:mml="http://www.w3.org/1998/Math/MathML"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.w3.org/1998/Math/MathML
        http://www.w3.org/Math/XMLSchema/mathml2/mathml2.xsd">""" + s + "</math>"


def _read_binary(pkgname, filename):
    """Reads binary data from a resource package.

    Parameters
    ==========

    pkgname
        Package name where the file is located.
    filename
        Name of the file to read from the package.

    Returns
    =======

    bytes
        Binary data read from the specified file.
    """
    import sys

    if sys.version_info >= (3, 10):
        # Python 3.10+ 以上版本可以使用 importlib.resources 的 files 函数
        from importlib.resources import files
        return files(pkgname).joinpath(filename).read_bytes()
    else:
        # Python 3.9 中的 read_binary 在 3.11 中已经被弃用
        from importlib.resources import read_binary
        return read_binary(pkgname, filename)


def _read_xsl(xsl):
    """Reads XSL stylesheet content either from a resource package or a file path.

    Parameters
    ==========

    xsl
        A string specifying the name or path of the XSL stylesheet.

    Returns
    =======

    bytes
        Binary content of the XSL stylesheet.
    """
    # 处理过时的 XSL 文件路径
    if xsl == 'mathml/data/simple_mmlctop.xsl':
        xsl = 'simple_mmlctop.xsl'
    elif xsl == 'mathml/data/mmlctop.xsl':
        xsl = 'mmlctop.xsl'
    elif xsl == 'mathml/data/mmltex.xsl':
        xsl = 'mmltex.xsl'

    if xsl in ['simple_mmlctop.xsl', 'mmlctop.xsl', 'mmltex.xsl']:
        # 从 sympy.utilities.mathml.data 包中读取 XSL 文件内容
        xslbytes = _read_binary('sympy.utilities.mathml.data', xsl)
    else:
        # 从指定路径中读取 XSL 文件内容
        xslbytes = Path(xsl).read_bytes()

    return xslbytes


@doctest_depends_on(modules=('lxml',))
def apply_xsl(mml, xsl):
    """Applies an XSL transformation to a MathML string.

    Parameters
    ==========

    mml
        A string containing MathML code.
    xsl
        A string specifying the name of an XSL stylesheet file or its full path.

    Returns
    =======

    str
        Transformed MathML string after applying the XSL stylesheet.
    """
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False)
    ac = etree.XSLTAccessControl.DENY_ALL

    s = etree.XML(_read_xsl(xsl), parser=parser)
    transform = etree.XSLT(s, access_control=ac)
    doc = etree.XML(mml, parser=parser)
    result = transform(doc)
    s = str(result)
    return s


@doctest_depends_on(modules=('lxml',))
def c2p(mml, simple=False):
    """Transforms MathML content into MathML presentation format.

    Parameters
    ==========

    mml
        A string containing MathML content.
    simple
        Boolean indicating whether to use a simplified transformation (default: False).

    Examples
    ========

    An example of how this function is used can be found in the module's doctest.
    """
    # 如果给定的 mml 字符串不以 '<math' 开头，就添加 mathml 头部信息
    if not mml.startswith('<math'):
        mml = add_mathml_headers(mml)
    
    # 根据 simple 参数的值选择不同的 XSL 样式表进行转换并返回结果
    if simple:
        # 使用简化的 XSL 样式表进行转换
        return apply_xsl(mml, 'mathml/data/simple_mmlctop.xsl')
    else:
        # 使用标准的 XSL 样式表进行转换
        return apply_xsl(mml, 'mathml/data/mmlctop.xsl')
```