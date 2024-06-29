# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_to_markdown.py`

```
from io import StringIO  # 导入StringIO模块，用于创建内存中的文本缓冲区

import pytest  # 导入pytest模块，用于编写和运行测试用例

import pandas as pd  # 导入pandas库，并将其重命名为pd
import pandas._testing as tm  # 导入pandas内部测试模块

pytest.importorskip("tabulate")  # 确保tabulate模块存在，否则跳过测试


def test_keyword_deprecation():
    # 测试函数：检查关键字参数过时警告
    # GH 57280
    msg = (
        "Starting with pandas version 4.0 all arguments of to_markdown "
        "except for the argument 'buf' will be keyword-only."
    )
    s = pd.Series()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        s.to_markdown(None, "wt")


def test_simple():
    # 测试函数：简单的DataFrame转换为Markdown格式
    buf = StringIO()  # 创建一个StringIO对象，作为文本缓冲区
    df = pd.DataFrame([1, 2, 3])  # 创建一个DataFrame对象
    df.to_markdown(buf=buf)  # 将DataFrame转换为Markdown格式，输出到buf中
    result = buf.getvalue()  # 获取buf中的内容作为字符串
    assert (
        result == "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )


def test_empty_frame():
    # 测试函数：空DataFrame转换为Markdown格式
    buf = StringIO()  # 创建一个StringIO对象，作为文本缓冲区
    df = pd.DataFrame({"id": [], "first_name": [], "last_name": []}).set_index("id")  # 创建一个空DataFrame，并设置'id'列为索引
    df.to_markdown(buf=buf)  # 将DataFrame转换为Markdown格式，输出到buf中
    result = buf.getvalue()  # 获取buf中的内容作为字符串
    assert result == (
        "| id   | first_name   | last_name   |\n"
        "|------|--------------|-------------|"
    )


def test_other_tablefmt():
    # 测试函数：指定其他的tablefmt参数来转换DataFrame为Markdown格式
    buf = StringIO()  # 创建一个StringIO对象，作为文本缓冲区
    df = pd.DataFrame([1, 2, 3])  # 创建一个DataFrame对象
    df.to_markdown(buf=buf, tablefmt="jira")  # 将DataFrame转换为Markdown格式，输出到buf中，使用jira格式
    result = buf.getvalue()  # 获取buf中的内容作为字符串
    assert result == "||    ||   0 ||\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"


def test_other_headers():
    # 测试函数：指定其他的headers参数来转换DataFrame为Markdown格式
    buf = StringIO()  # 创建一个StringIO对象，作为文本缓冲区
    df = pd.DataFrame([1, 2, 3])  # 创建一个DataFrame对象
    df.to_markdown(buf=buf, headers=["foo", "bar"])  # 将DataFrame转换为Markdown格式，输出到buf中，使用指定的列标题
    result = buf.getvalue()  # 获取buf中的内容作为字符串
    assert result == (
        "|   foo |   bar |\n|------:|------:|\n|     0 "
        "|     1 |\n|     1 |     2 |\n|     2 |     3 |"
    )


def test_series():
    # 测试函数：将Series对象转换为Markdown格式
    buf = StringIO()  # 创建一个StringIO对象，作为文本缓冲区
    s = pd.Series([1, 2, 3], name="foo")  # 创建一个Series对象
    s.to_markdown(buf=buf)  # 将Series对象转换为Markdown格式，输出到buf中
    result = buf.getvalue()  # 获取buf中的内容作为字符串
    assert result == (
        "|    |   foo |\n|---:|------:|\n|  0 |     1 "
        "|\n|  1 |     2 |\n|  2 |     3 |"
    )


def test_no_buf():
    # 测试函数：不使用缓冲区，直接将DataFrame转换为Markdown格式
    df = pd.DataFrame([1, 2, 3])  # 创建一个DataFrame对象
    result = df.to_markdown()  # 将DataFrame转换为Markdown格式，返回转换后的字符串
    assert (
        result == "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )


@pytest.mark.parametrize("index", [True, False])
def test_index(index):
    # 测试函数：指定是否包含索引列来转换DataFrame为Markdown格式
    # GH 32667

    df = pd.DataFrame([1, 2, 3])  # 创建一个DataFrame对象

    result = df.to_markdown(index=index)  # 将DataFrame转换为Markdown格式，指定是否包含索引列

    if index:
        expected = (
            "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
        )
    else:
        expected = "|   0 |\n|----:|\n|   1 |\n|   2 |\n|   3 |"
    assert result == expected


def test_showindex_disallowed_in_kwargs():
    # 测试函数：测试在参数中使用showindex时是否会抛出错误
    # GH 32667; disallowing showindex in kwargs enforced in 2.0
    df = pd.DataFrame([1, 2, 3])  # 创建一个DataFrame对象
    with pytest.raises(ValueError, match="Pass 'index' instead of 'showindex"):
        df.to_markdown(index=True, showindex=True)  # 尝试使用showindex参数，应该抛出值错误异常
```