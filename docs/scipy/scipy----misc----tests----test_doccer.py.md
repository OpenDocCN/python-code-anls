# `D:\src\scipysrc\scipy\scipy\misc\tests\test_doccer.py`

```
''' Some tests for the documenting decorator and support functions '''

# 导入系统模块 sys
import sys
# 导入 pytest 测试框架
import pytest
# 导入 numpy 的测试工具 assert_equal 和抑制警告的模块 suppress_warnings
from numpy.testing import assert_equal, suppress_warnings

# 导入 scipy 库中的文档处理模块 doccer
from scipy._lib import doccer

# python -OO 会剥离文档字符串
DOCSTRINGS_STRIPPED = sys.flags.optimize > 1

# 定义一个多行文档字符串
docstring = \
"""Docstring
    %(strtest1)s
        %(strtest2)s
     %(strtest3)s
"""

# 定义一些参数化文档字符串
param_doc1 = \
"""Another test
   with some indent"""

param_doc2 = \
"""Another test, one line"""

param_doc3 = \
"""    Another test
       with some indent"""

# 将参数化文档字符串存入字典
doc_dict = {'strtest1':param_doc1,
            'strtest2':param_doc2,
            'strtest3':param_doc3}

# 填充后的文档字符串
filled_docstring = \
"""Docstring
    Another test
       with some indent
        Another test, one line
     Another test
       with some indent
"""


# 测试函数：测试文本去缩进函数 unindent_string
def test_unindent():
    # 使用 suppress_warnings 抑制 DeprecationWarning
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        # 断言 unindent_string 函数对各参数化文档字符串的处理结果正确
        assert_equal(doccer.unindent_string(param_doc1), param_doc1)
        assert_equal(doccer.unindent_string(param_doc2), param_doc2)
        assert_equal(doccer.unindent_string(param_doc3), param_doc1)


# 测试函数：测试文本字典去缩进函数 unindent_dict
def test_unindent_dict():
    # 使用 suppress_warnings 抑制 DeprecationWarning
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        # 对 doc_dict 应用 unindent_dict 函数
        d2 = doccer.unindent_dict(doc_dict)
    # 断言 unindent_dict 函数处理后的结果与原始 doc_dict 一致
    assert_equal(d2['strtest1'], doc_dict['strtest1'])
    assert_equal(d2['strtest2'], doc_dict['strtest2'])
    assert_equal(d2['strtest3'], doc_dict['strtest1'])


# 测试函数：测试文档格式化函数 docformat
def test_docformat():
    # 使用 suppress_warnings 抑制 DeprecationWarning
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        # 对 doc_dict 应用 unindent_dict 函数
        udd = doccer.unindent_dict(doc_dict)
        # 对文档字符串 docstring 进行格式化
        formatted = doccer.docformat(docstring, udd)
        # 断言格式化后的结果与填充后的文档字符串 filled_docstring 相等
        assert_equal(formatted, filled_docstring)
        # 单行文档字符串格式化测试
        single_doc = 'Single line doc %(strtest1)s'
        formatted = doccer.docformat(single_doc, doc_dict)
        # 注意：格式化字符串的初始缩进不影响插入参数的后续缩进
        assert_equal(formatted, """Single line doc Another test
   with some indent""")


# 标记：如果文档字符串被剥离，则跳过测试
@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstrings stripped")
# 标记：如果 Python 版本大于等于 3.13，则跳过测试
@pytest.mark.skipif(sys.version_info >= (3, 13), reason='it fails on Py3.13')
# 测试函数：测试装饰器功能
def test_decorator():
    # 使用 suppress_warnings 抑制 DeprecationWarning
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        # 创建带有 unindentation 参数的装饰器
        decorator = doccer.filldoc(doc_dict, True)

        @decorator
        def func():
            """ Docstring
            %(strtest3)s
            """
        # 断言 func 函数的文档字符串被正确填充
        assert_equal(func.__doc__, """ Docstring
            Another test
               with some indent
            """)

        # 创建不带 unindentation 参数的装饰器
        decorator = doccer.filldoc(doc_dict, False)

        @decorator
        def func():
            """ Docstring
            %(strtest3)s
            """
        # 断言 func 函数的文档字符串被正确填充
        assert_equal(func.__doc__, """ Docstring
                Another test
                   with some indent
            """)


# 标记：如果文档字符串被剥离，则跳过测试
@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstrings stripped")
def test_inherit_docstring_from():

    # 使用 suppress_warnings 上下文管理器来抑制警告
    with suppress_warnings() as sup:
        # 过滤特定类别的 DeprecationWarning
        sup.filter(category=DeprecationWarning)

        # 定义一个名为 Foo 的类
        class Foo:
            # 定义一个函数 func，带有文档字符串 "Do something useful."
            def func(self):
                '''Do something useful.'''
                return

            # 定义一个函数 func2，带有文档字符串 "Something else."
            def func2(self):
                '''Something else.'''
        
        # 定义一个名为 Bar 的类，继承自 Foo
        class Bar(Foo):
            # 使用 doccer.inherit_docstring_from(Foo) 装饰器继承 func 函数的文档字符串
            @doccer.inherit_docstring_from(Foo)
            def func(self):
                '''%(super)sABC'''
                return

            # 使用 doccer.inherit_docstring_from(Foo) 装饰器继承 func2 函数的文档字符串
            @doccer.inherit_docstring_from(Foo)
            def func2(self):
                # 没有文档字符串
                return

    # 断言 Bar 类的 func 函数的文档字符串与 Foo 类的 func 函数的文档字符串拼接后相等
    assert_equal(Bar.func.__doc__, Foo.func.__doc__ + 'ABC')
    # 断言 Bar 类的 func2 函数的文档字符串与 Foo 类的 func2 函数的文档字符串相等
    assert_equal(Bar.func2.__doc__, Foo.func2.__doc__)
    
    # 创建 Bar 类的实例 bar
    bar = Bar()
    # 断言 bar 实例的 func 函数的文档字符串与 Foo 类的 func 函数的文档字符串拼接后相等
    assert_equal(bar.func.__doc__, Foo.func.__doc__ + 'ABC')
    # 断言 bar 实例的 func2 函数的文档字符串与 Foo 类的 func2 函数的文档字符串相等
    assert_equal(bar.func2.__doc__, Foo.func2.__doc__)
```