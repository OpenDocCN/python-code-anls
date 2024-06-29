# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_tooltip.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块
    DataFrame,  # 数据帧，用于表示二维数据结构
    MultiIndex,  # 多级索引，用于层次化数据表示
)

pytest.importorskip("jinja2")  # 导入 jinja2 库，如果不存在则跳过

from pandas.io.formats.style import Styler  # 从 pandas 库中导入 Styler 类，用于样式化数据帧的输出


@pytest.fixture
def df():
    return DataFrame(  # 创建一个数据帧对象
        data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],  # 数据为二维列表
        columns=["A", "B", "C"],  # 列名为 A、B、C
        index=["x", "y", "z"],  # 行索引为 x、y、z
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)  # 创建 Styler 对象，用于美化数据帧的输出，uuid_len 设置为 0


@pytest.mark.parametrize(
    "data, columns, index",
    [
        # 测试基本的重新索引和忽略空白
        ([["Min", "Max"], [np.nan, ""]], ["A", "C"], ["x", "y"]),
        # 测试非参考列，反转列名，短索引
        ([["Max", "Min", "Bad-Col"]], ["C", "A", "D"], ["x"]),
    ],
)
def test_tooltip_render(data, columns, index, styler):
    ttips = DataFrame(data=data, columns=columns, index=index)  # 创建包含提示信息的数据帧对象 ttips

    # GH 21266
    result = styler.set_tooltips(ttips).to_html()  # 使用 Styler 设置提示信息，并将结果转换为 HTML 字符串

    # 测试提示表的级别类
    assert "#T_ .pd-t {\n  visibility: hidden;\n" in result

    # 测试添加 'Min' 提示
    assert "#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}" in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' in result
    assert 'class="data row0 col0" >0<span class="pd-t"></span></td>' in result

    # 测试添加 'Max' 提示
    assert "#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}" in result
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' in result
    assert 'class="data row0 col2" >2<span class="pd-t"></span></td>' in result

    # 测试忽略 NaN、空字符串和坏列
    assert "#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "Bad-Col" not in result


def test_tooltip_ignored(styler):
    # GH 21266
    result = styler.to_html()  # 不使用 set_tooltips() 创建没有 <span>
    assert '<style type="text/css">\n</style>' in result
    assert '<span class="pd-t"></span>' not in result
    assert 'title="' not in result


def test_tooltip_css_class(styler):
    # GH 21266
    result = styler.set_tooltips(
        DataFrame([["tooltip"]], index=["x"], columns=["A"]),  # 创建包含提示信息的数据帧对象
        css_class="other-class",  # 设置 CSS 类名为 "other-class"
        props=[("color", "green")],  # 设置属性为颜色为绿色
    ).to_html()
    assert "#T_ .other-class {\n  color: green;\n" in result
    assert '#T_ #T__row0_col0 .other-class::after {\n  content: "tooltip";\n' in result

    # GH 39563
    result = styler.set_tooltips(  # set_tooltips 覆盖之前的设置
        DataFrame([["tooltip"]], index=["x"], columns=["A"]),  # 创建包含提示信息的数据帧对象
        css_class="another-class",  # 设置 CSS 类名为 "another-class"
        props="color:green;color:red;",  # 设置属性为颜色为绿色和红色
    ).to_html()
    assert "#T_ .another-class {\n  color: green;\n  color: red;\n}" in result
    [
        # 测试基本的重新索引和忽略空白值
        (["Min", "Max"], [np.nan, ""]),
        # 测试非引用列，反转列名，短索引
        (["Max", "Min", "Bad-Col"], ["C", "A", "D"], ["x"]),
    ],
# 定义函数，测试数据帧的工具提示渲染作为标题
def test_tooltip_render_as_title(data, columns, index, styler):
    # 创建数据帧对象ttips，用给定的数据、列和索引
    ttips = DataFrame(data=data, columns=columns, index=index)
    # 将styler对象设置为在HTML中将ttips作为标题属性的工具提示，并将结果转换为HTML字符串
    result = styler.set_tooltips(ttips, as_title_attribute=True).to_html()

    # 断言：检查是否未添加特定CSS样式
    assert "#T_ .pd-t {\n  visibility: hidden;\n" not in result

    # 断言：检查是否未将'Min'作为标题属性添加，并且相应的CSS不存在
    assert "#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' not in result
    assert 'class="data row0 col0"  title="Min">0</td>' in result

    # 断言：检查是否未将'Max'作为标题属性添加，并且相应的CSS不存在
    assert "#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' not in result
    assert 'class="data row0 col2"  title="Max">2</td>' in result

    # 断言：检查是否忽略了NaN、空字符串和错误列
    assert "#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "Bad-Col" not in result
    assert 'class="data row0 col1" >1</td>' in result
    assert 'class="data row1 col0" >3</td>' in result
    assert 'class="data row1 col1" >4</td>' in result
    assert 'class="data row1 col2" >5</td>' in result
    assert 'class="data row2 col0" >6</td>' in result
    assert 'class="data row2 col1" >7</td>' in result
    assert 'class="data row2 col2" >8</td>' in result


# 定义函数，测试带隐藏索引级别的工具提示渲染为标题
def test_tooltip_render_as_title_with_hidden_index_level():
    # 创建数据帧df，包括数据、列和多级索引
    df = DataFrame(
        data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        columns=["A", "B", "C"],
        index=MultiIndex.from_arrays(
            [["x", "y", "z"], [1, 2, 3], ["aa", "bb", "cc"]],
            names=["alpha", "num", "char"],
        ),
    )
    # 创建数据帧对象ttips，用于测试重新索引、忽略空值，并隐藏第二级别（num）的索引
    ttips = DataFrame(
        data=[["Min", "Max"], [np.nan, ""]],
        columns=["A", "C"],
        index=MultiIndex.from_arrays(
            [["x", "y"], [1, 2], ["aa", "bb"]], names=["alpha", "num", "char"]
        ),
    )
    # 创建Styler对象styler，并隐藏axis=0，level=-1（最后一级别），names=True的索引
    styler = Styler(df, uuid_len=0)
    styler = styler.hide(axis=0, level=-1, names=True)
    # 将styler对象设置为在HTML中将ttips作为标题属性的工具提示，并将结果转换为HTML字符串
    result = styler.set_tooltips(ttips, as_title_attribute=True).to_html()

    # 断言：检查是否未添加特定CSS样式
    assert "#T_ .pd-t {\n  visibility: hidden;\n" not in result

    # 断言：检查是否未将'Min'作为标题属性添加，并且相应的CSS不存在
    assert "#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' not in result
    assert 'class="data row0 col0"  title="Min">0</td>' in result
    # 测试 'Max' 工具提示作为 title 属性添加，并且不存在对应的 CSS 样式
    assert "#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    # 确保不会在结果中找到 'Max' 工具提示的伪元素样式定义
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' not in result
    # 确保结果中包含指定的数据单元格类和 title 属性值为 'Max'
    assert 'class="data row0 col2"  title="Max">2</td>' in result

    # 测试 NaN、空字符串和不良列是否被忽略
    # 确保不会在结果中找到对应列的 NaN 工具提示的伪元素样式定义
    assert "#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    # 确保不会在结果中找到对应列的 NaN 工具提示的伪元素样式定义
    assert "#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    # 确保不会在结果中找到对应列的 NaN 工具提示的伪元素样式定义
    assert "#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    # 确保不会在结果中找到对应列的 NaN 工具提示的伪元素样式定义
    assert "#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    # 确保结果中不包含 'Bad-Col'
    assert "Bad-Col" not in result
    # 确保结果中包含指定的数据单元格类
    assert 'class="data row0 col1" >1</td>' in result
    assert 'class="data row1 col0" >3</td>' in result
    assert 'class="data row1 col1" >4</td>' in result
    assert 'class="data row1 col2" >5</td>' in result
    assert 'class="data row2 col0" >6</td>' in result
    assert 'class="data row2 col1" >7</td>' in result
    assert 'class="data row2 col2" >8</td>' in result
```