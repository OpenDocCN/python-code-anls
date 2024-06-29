# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_to_string.py`

```
# 从 textwrap 模块中导入 dedent 函数，用于处理多行字符串的缩进
# 导入 pytest 模块，用于编写和运行测试用例
# 从 pandas 库中导入 DataFrame 和 Series 类
pytest.importorskip("jinja2")  # 导入 jinja2 模块，如果不存在则跳过导入
# 从 pandas.io.formats.style 模块中导入 Styler 类
@pytest.fixture
# 定义一个 pytest 的 fixture，用于创建测试中常用的 DataFrame 实例 df
def df():
    return DataFrame(
        {"A": [0, 1], "B": [-0.61, -1.22], "C": Series(["ab", "cd"], dtype=object)}
    )


@pytest.fixture
# 定义一个 pytest 的 fixture，用于创建 Styler 实例 styler
def styler(df):
    return Styler(df, uuid_len=0, precision=2)


def test_basic_string(styler):
    # 测试 Styler 对象的 to_string 方法，返回结果并与预期结果进行比较
    result = styler.to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    """
    )
    assert result == expected


def test_string_delimiter(styler):
    # 测试 Styler 对象的 to_string 方法，使用指定的分隔符 delimiter 进行字符串格式化
    result = styler.to_string(delimiter=";")
    expected = dedent(
        """\
    ;A;B;C
    0;0;-0.61;ab
    1;1;-1.22;cd
    """
    )
    assert result == expected


def test_concat(styler):
    # 测试 Styler 对象的 concat 方法，将 Styler 对象与另一个 Styler 对象的样式合并后，返回结果并进行比较
    result = styler.concat(styler.data.agg(["sum"]).style).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830000 abcd
    """
    )
    assert result == expected


def test_concat_recursion(styler):
    # 测试 Styler 对象的 concat 方法，进行递归连接 Styler 对象的样式，并返回结果进行比较
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["sum"]), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(["sum"]), uuid_len=0, precision=4)
    result = styler1.concat(styler2.concat(styler3)).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830 abcd
    sum 1 -1.8300 abcd
    """
    )
    assert result == expected


def test_concat_chain(styler):
    # 测试 Styler 对象的 concat 方法，链式连接多个 Styler 对象的样式，并返回结果进行比较
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["sum"]), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(["sum"]), uuid_len=0, precision=4)
    result = styler1.concat(styler2).concat(styler3).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830 abcd
    sum 1 -1.8300 abcd
    """
    )
    assert result == expected
```