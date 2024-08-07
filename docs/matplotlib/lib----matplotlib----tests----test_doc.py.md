# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_doc.py`

```py
import pytest  # 导入 pytest 测试框架


def test_sphinx_gallery_example_header():
    """
    We have copied EXAMPLE_HEADER and modified it to include meta keywords.
    This test monitors that the version we have copied is still the same as
    the EXAMPLE_HEADER in sphinx-gallery. If sphinx-gallery changes its
    EXAMPLE_HEADER, this test will start to fail. In that case, please update
    the monkey-patching of EXAMPLE_HEADER in conf.py.
    """
    # 导入 pytest 并检查是否满足最低版本要求，如果不满足则跳过测试
    pytest.importorskip('sphinx_gallery', minversion='0.16.0')
    # 从 sphinx_gallery 模块中导入 gen_rst 函数
    from sphinx_gallery import gen_rst

    # 定义 EXAMPLE_HEADER 字符串，包含自动生成的文档头部信息模板
    EXAMPLE_HEADER = """
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "{0}"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_{1}>`
        to download the full example code.{2}

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_{1}:

"""
    # 断言 gen_rst 模块中的 EXAMPLE_HEADER 是否等于预定义的 EXAMPLE_HEADER
    assert gen_rst.EXAMPLE_HEADER == EXAMPLE_HEADER
```