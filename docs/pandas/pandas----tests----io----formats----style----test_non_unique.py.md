# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_non_unique.py`

```
# 导入必要的模块和函数
from textwrap import dedent
import pytest
from pandas import (
    DataFrame,
    IndexSlice,
)
# 检查并跳过导入失败的情况，如果缺少 jinja2 模块
pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler

# 创建一个用于测试的 DataFrame fixture，包括非唯一的索引和列名
@pytest.fixture
def df():
    return DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["i", "j", "j"],  # 非唯一索引
        columns=["c", "d", "d"],  # 非唯一列名
        dtype=float,
    )

# 创建一个 Styler 对象的 fixture
@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)

# 测试函数：格式化非唯一的 DataFrame，验证 HTML 中的数据显示正确
def test_format_non_unique(df):
    # GH 41269
    
    # 使用字典格式化样式，生成 HTML
    html = df.style.format({"d": "{:.1f}"}).to_html()
    # 验证 HTML 中的特定数值
    for val in ["1.000000<", "4.000000<", "7.000000<"]:
        assert val in html
    for val in ["2.0<", "3.0<", "5.0<", "6.0<", "8.0<", "9.0<"]:
        assert val in html

    # 使用子集参数格式化样式，生成 HTML
    html = df.style.format(precision=1, subset=IndexSlice["j", "d"]).to_html()
    # 验证 HTML 中的特定数值
    for val in ["1.000000<", "4.000000<", "7.000000<", "2.000000<", "3.000000<"]:
        assert val in html
    for val in ["5.0<", "6.0<", "8.0<", "9.0<"]:
        assert val in html

# 参数化测试函数：验证 apply 和 map 方法在非唯一索引时引发 KeyError
@pytest.mark.parametrize("func", ["apply", "map"])
def test_apply_map_non_unique_raises(df, func):
    # GH 41269
    if func == "apply":
        op = lambda s: ["color: red;"] * len(s)  # 对 Series 应用操作函数
    else:
        op = lambda v: "color: red;"  # 对单个值应用操作函数

    # 使用 pytest.raises 检查是否引发 KeyError 异常
    with pytest.raises(KeyError, match="`Styler.apply` and `.map` are not"):
        getattr(df.style, func)(op)._compute()

# 测试函数：验证设置非唯一索引的表格样式字典
def test_table_styles_dict_non_unique_index(styler):
    # 使用 set_table_styles 方法设置表格样式
    styles = styler.set_table_styles(
        {"j": [{"selector": "td", "props": "a: v;"}]}, axis=1
    ).table_styles
    # 验证生成的样式字典
    assert styles == [
        {"selector": "td.row1", "props": [("a", "v")]},  # 第一行样式
        {"selector": "td.row2", "props": [("a", "v")]},  # 第二行样式
    ]

# 测试函数：验证设置非唯一列名的表格样式字典
def test_table_styles_dict_non_unique_columns(styler):
    # 使用 set_table_styles 方法设置表格样式
    styles = styler.set_table_styles(
        {"d": [{"selector": "td", "props": "a: v;"}]}, axis=0
    ).table_styles
    # 验证生成的样式字典
    assert styles == [
        {"selector": "td.col1", "props": [("a", "v")]},  # 第一列样式
        {"selector": "td.col2", "props": [("a", "v")]},  # 第二列样式
    ]

# 测试函数：验证设置非唯一索引或列名的工具提示时是否引发 KeyError
def test_tooltips_non_unique_raises(styler):
    # ttips 具有唯一的键，应该通过
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "b"])
    styler.set_tooltips(ttips=ttips)  # 正常情况

    # ttips 具有非唯一的列名，应该引发 KeyError
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "c"], index=["a", "b"])
    with pytest.raises(KeyError, match="Tooltips render only if `ttips` has unique"):
        styler.set_tooltips(ttips=ttips)

    # ttips 具有非唯一的索引，应该引发 KeyError
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "a"])
    with pytest.raises(KeyError, match="Tooltips render only if `ttips` has unique"):
        styler.set_tooltips(ttips=ttips)

# 测试函数：验证设置非唯一索引或列名的表格单元格类名时是否引发 KeyError
def test_set_td_classes_non_unique_raises(styler):
    # classes 具有唯一的键，应该通过
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "b"])
    styler.set_td_classes(classes=classes)  # 正常情况
    # 创建一个 DataFrame 对象 `classes`，包含两行两列的数据，列名为 ["c", "c"]，行索引为 ["a", "b"]
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "c"], index=["a", "b"])
    
    # 使用 pytest 的 `pytest.raises` 上下文管理器，期望捕获 KeyError 异常，并验证异常消息包含 "Classes render only if `classes` has unique"
    with pytest.raises(KeyError, match="Classes render only if `classes` has unique"):
        # 调用 styler 的 `set_td_classes` 方法，传入参数 classes=classes
        styler.set_td_classes(classes=classes)
    
    # 创建另一个 DataFrame 对象 `classes`，包含两行两列的数据，列名为 ["c", "d"]，行索引为 ["a", "a"]，索引不唯一
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "a"])
    
    # 使用 pytest 的 `pytest.raises` 上下文管理器，期望捕获 KeyError 异常，并验证异常消息包含 "Classes render only if `classes` has unique"
    with pytest.raises(KeyError, match="Classes render only if `classes` has unique"):
        # 再次调用 styler 的 `set_td_classes` 方法，传入参数 classes=classes
        styler.set_td_classes(classes=classes)
# 定义测试函数，用于验证隐藏非唯一列功能
def test_hide_columns_non_unique(styler):
    # 使用样式对象调用隐藏列方法，隐藏列"d"，并转换成内部数据表示
    ctx = styler.hide(["d"], axis="columns")._translate(True, True)

    # 断言头部第一行第二列的显示值为"c"
    assert ctx["head"][0][1]["display_value"] == "c"
    # 断言头部第一行第二列可见性为True
    assert ctx["head"][0][1]["is_visible"] is True

    # 断言头部第一行第三列的显示值为"d"
    assert ctx["head"][0][2]["display_value"] == "d"
    # 断言头部第一行第三列可见性为False
    assert ctx["head"][0][2]["is_visible"] is False

    # 断言头部第一行第四列的显示值为"d"
    assert ctx["head"][0][3]["display_value"] == "d"
    # 断言头部第一行第四列可见性为False
    assert ctx["head"][0][3]["is_visible"] is False

    # 断言正文部分第一行第二列可见性为True
    assert ctx["body"][0][1]["is_visible"] is True
    # 断言正文部分第一行第三列可见性为False
    assert ctx["body"][0][2]["is_visible"] is False
    # 断言正文部分第一行第四列可见性为False
    assert ctx["body"][0][3]["is_visible"] is False


# 定义测试函数，用于验证生成 LaTeX 表格功能（非唯一列）
def test_latex_non_unique(styler):
    # 调用样式对象的 to_latex 方法，生成 LaTeX 表格字符串
    result = styler.to_latex()
    # 断言生成的 LaTeX 字符串与预期的格式化字符串相符
    assert result == dedent(
        """\
        \\begin{tabular}{lrrr}
         & c & d & d \\\\
        i & 1.000000 & 2.000000 & 3.000000 \\\\
        j & 4.000000 & 5.000000 & 6.000000 \\\\
        j & 7.000000 & 8.000000 & 9.000000 \\\\
        \\end{tabular}
    """
    )
```