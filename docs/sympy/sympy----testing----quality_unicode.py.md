# `D:\src\scipysrc\sympy\sympy\testing\quality_unicode.py`

```
# 导入正则表达式模块
import re
# 导入文件名匹配模块
import fnmatch

# 定义 Unicode 错误消息 B，用于包含未在白名单中的文件的 Unicode 字符报错信息
message_unicode_B = \
    "File contains a unicode character : %s, line %s. " \
    "But not in the whitelist. " \
    "Add the file to the whitelist in " + __file__
# 定义 Unicode 错误消息 D，用于包含在白名单中但不含 Unicode 字符的文件报错信息
message_unicode_D = \
    "File does not contain a unicode character : %s." \
    "but is in the whitelist. " \
    "Remove the file from the whitelist in " + __file__

# 定义编码头部的正则表达式模式，用于匹配文件头部的编码声明
encoding_header_re = re.compile(
    r'^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')

# 白名单模式列表，用于允许包含 Unicode 的文件
unicode_whitelist = [
    # 作者名字可以包含非 ASCII 字符
    r'*/bin/authors_update.py',
    r'*/bin/mailmap_check.py',

    # 这些文件包含有关 Unicode 输入和输出的功能和测试函数
    r'*/sympy/testing/tests/test_code_quality.py',
    r'*/sympy/physics/vector/tests/test_printing.py',
    r'*/physics/quantum/tests/test_printing.py',
    r'*/sympy/vector/tests/test_printing.py',
    r'*/sympy/parsing/tests/test_sympy_parser.py',
    r'*/sympy/printing/pretty/stringpict.py',
    r'*/sympy/printing/pretty/tests/test_pretty.py',
    r'*/sympy/printing/tests/test_conventions.py',
    r'*/sympy/printing/tests/test_preview.py',
    r'*/liealgebras/type_g.py',
    r'*/liealgebras/weyl_group.py',
    r'*/liealgebras/tests/test_type_G.py',

    # wigner.py 和 polarization.py 包含 Unicode doctest
    r'*/sympy/physics/wigner.py',
    r'*/sympy/physics/optics/polarization.py',

    # joint.py 在文档字符串中使用了一些 Unicode 变量名
    r'*/sympy/physics/mechanics/joint.py',

    # lll 方法在文档字符串和作者名字中使用了 Unicode
    r'*/sympy/polys/matrices/domainmatrix.py',
    r'*/sympy/matrices/repmatrix.py',

    # 符号解释中使用了希腊字母
    r'*/sympy/core/symbol.py',
]

# 严格白名单模式列表，用于强制要求包含 Unicode 的文件
unicode_strict_whitelist = [
    r'*/sympy/parsing/latex/_antlr/__init__.py',
    # test_mathematica.py 在测试希腊字符是否正常工作时使用了一些 Unicode
    r'*/sympy/parsing/tests/test_mathematica.py',
]

def _test_this_file_encoding(
    fname, test_file,
    unicode_whitelist=unicode_whitelist,
    unicode_strict_whitelist=unicode_strict_whitelist):
    """测试辅助函数，用于 Unicode 测试

    测试可能需要以文件为单位操作，因此将其移到单独的过程中。
    """
    # 是否包含 Unicode
    has_unicode = False

    # 是否在白名单中
    is_in_whitelist = False
    # 是否在严格白名单中
    is_in_strict_whitelist = False

    # 遍历普通白名单列表
    for patt in unicode_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_whitelist = True
            break

    # 遍历严格白名单列表
    for patt in unicode_strict_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_strict_whitelist = True
            is_in_whitelist = True
            break
    # 如果文件在白名单中
    if is_in_whitelist:
        # 遍历测试文件的每一行，同时记录行号（idx）
        for idx, line in enumerate(test_file):
            try:
                # 尝试将每一行编码为ASCII，如果有非ASCII字符会抛出异常
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # 如果捕获到Unicode编码错误或解码错误，则表示文件包含Unicode字符
                has_unicode = True

        # 如果文件中没有Unicode字符，并且不在严格白名单中
        if not has_unicode and not is_in_strict_whitelist:
            # 断言失败，输出包含文件名的错误消息（message_unicode_D），指示文件包含Unicode字符
            assert False, message_unicode_D % fname

    # 如果文件不在白名单中
    else:
        # 遍历测试文件的每一行，同时记录行号（idx）
        for idx, line in enumerate(test_file):
            try:
                # 尝试将每一行编码为ASCII，如果有非ASCII字符会抛出异常
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # 断言失败，输出包含文件名和行号的错误消息（message_unicode_B），指示具体行包含Unicode字符
                assert False, message_unicode_B % (fname, idx + 1)
```