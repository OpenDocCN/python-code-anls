# `D:\src\scipysrc\pandas\pandas\tests\util\test_doc.py`

```
# 导入需要的模块和函数
from textwrap import dedent
from pandas.util._decorators import doc

# 使用 @doc 装饰器为 cumsum 方法添加文档字符串
@doc(method="cumsum", operation="sum")
def cumsum(whatever):
    """
    This is the {method} method.

    It computes the cumulative {operation}.
    """

# 使用 @doc 装饰器为 cumavg 方法添加文档字符串
@doc(
    cumsum,
    dedent(
        """
        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    ),
    method="cumavg",
    operation="average",
)
def cumavg(whatever):
    pass

# 使用 @doc 装饰器为 cummax 方法添加文档字符串
@doc(cumsum, method="cummax", operation="maximum")
def cummax(whatever):
    pass

# 使用 @doc 装饰器为 cummin 方法添加文档字符串
@doc(cummax, method="cummin", operation="minimum")
def cummin(whatever):
    pass

# 测试 cumsum 方法的文档字符串格式
def test_docstring_formatting():
    docstr = dedent(
        """
        This is the cumsum method.

        It computes the cumulative sum.
        """
    )
    assert cumsum.__doc__ == docstr

# 测试 cumavg 方法的文档字符串格式和示例
def test_docstring_appending():
    docstr = dedent(
        """
        This is the cumavg method.

        It computes the cumulative average.

        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    )
    assert cumavg.__doc__ == docstr

# 测试从 cumsum 方法继承的文档字符串模板是否正确
def test_doc_template_from_func():
    docstr = dedent(
        """
        This is the cummax method.

        It computes the cumulative maximum.
        """
    )
    assert cummax.__doc__ == docstr

# 测试从 cummax 方法继承的文档字符串模板是否正确
def test_inherit_doc_template():
    docstr = dedent(
        """
        This is the cummin method.

        It computes the cumulative minimum.
        """
    )
    assert cummin.__doc__ == docstr
```