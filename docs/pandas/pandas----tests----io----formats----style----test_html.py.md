# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\test_html.py`

```
# 从textwrap模块导入dedent和indent函数
from textwrap import (
    dedent,
    indent,
)

# 导入numpy库，并用np作为别名
import numpy as np

# 导入pytest库
import pytest

# 从pandas库中导入DataFrame、MultiIndex、option_context等
from pandas import (
    DataFrame,
    MultiIndex,
    option_context,
)

# 导入pytest并尝试导入jinja2库，如果导入失败则测试跳过
jinja2 = pytest.importorskip("jinja2")

# 从pandas.io.formats.style模块中导入Styler类
from pandas.io.formats.style import Styler


# 定义一个pytest fixture函数，用于设置Jinja2的环境
@pytest.fixture
def env():
    # 使用jinja2的PackageLoader从指定路径加载模板
    loader = jinja2.PackageLoader("pandas", "io/formats/templates")
    # 创建Jinja2的环境对象，设置为trim_blocks=True
    env = jinja2.Environment(loader=loader, trim_blocks=True)
    return env


# 定义一个pytest fixture函数，用于返回一个Styler对象
@pytest.fixture
def styler():
    # 创建一个简单的DataFrame并传递给Styler对象
    return Styler(DataFrame([[2.61], [2.69]], index=["a", "b"], columns=["A"]))


# 定义一个pytest fixture函数，用于返回一个带有MultiIndex的Styler对象
@pytest.fixture
def styler_mi():
    # 创建一个MultiIndex并传递给DataFrame，然后再传递给Styler对象
    midx = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    return Styler(DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=midx))


# 定义一个pytest fixture函数，用于返回一个带有多层MultiIndex的Styler对象
@pytest.fixture
def styler_multi():
    # 创建一个复杂的DataFrame并传递给Styler对象
    df = DataFrame(
        data=np.arange(16).reshape(4, 4),
        columns=MultiIndex.from_product([["A", "B"], ["a", "b"]], names=["A&", "b&"]),
        index=MultiIndex.from_product([["X", "Y"], ["x", "y"]], names=["X>", "y_"]),
    )
    return Styler(df)


# 定义一个pytest fixture函数，用于获取指定模板文件的Jinja2模板对象
@pytest.fixture
def tpl_style(env):
    return env.get_template("html_style.tpl")


# 定义一个pytest fixture函数，用于获取指定模板文件的Jinja2模板对象
@pytest.fixture
def tpl_table(env):
    return env.get_template("html_table.tpl")


# 定义一个测试函数，用于验证HTML模板是否正确引用了相关的子模板
def test_html_template_extends_options():
    # 打开并读取指定路径的html.tpl文件内容
    with open("pandas/io/formats/templates/html.tpl", encoding="utf-8") as file:
        result = file.read()
    # 断言文件内容中包含特定的Jinja2模板引用标记
    assert "{% include html_style_tpl %}" in result
    assert "{% include html_table_tpl %}" in result


# 定义一个测试函数，用于验证Styler对象生成的HTML是否正确排除了样式表，并包含了正确的doctype声明
def test_exclude_styles(styler):
    # 调用Styler对象的to_html方法生成HTML字符串，并设置了exclude_styles和doctype_html参数
    result = styler.to_html(exclude_styles=True, doctype_html=True)
    # 预期的HTML字符串，使用了dedent函数进行多行文本的缩进处理
    expected = dedent(
        """\
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        </head>
        <body>
        <table>
          <thead>
            <tr>
              <th >&nbsp;</th>
              <th >A</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th >a</th>
              <td >2.610000</td>
            </tr>
            <tr>
              <th >b</th>
              <td >2.690000</td>
            </tr>
          </tbody>
        </table>
        </body>
        </html>
        """
    )
    # 断言生成的HTML字符串与预期的一致
    assert result == expected


# 定义一个测试函数，用于验证Styler对象的多种格式化和样式设置操作
def test_w3_html_format(styler):
    # 设置Styler对象的UUID并链式调用多种格式和样式设置方法
    styler.set_uuid("").set_table_styles([{"selector": "th", "props": "att2:v2;"}]).map(
        lambda x: "att1:v1;"
    ).set_table_attributes('class="my-cls1" style="attr3:v3;"').set_td_classes(
        DataFrame(["my-cls2"], index=["a"], columns=["A"])
    ).format("{:.1f}").set_caption("A comprehensive test")
    # 定义预期的 HTML 字符串，使用 textwrap.dedent 方法移除多余的缩进
    expected = dedent(
        """\
        <style type="text/css">
        #T_ th {
          att2: v2;
        }
        #T__row0_col0, #T__row1_col0 {
          att1: v1;
        }
        </style>
        <table id="T_" class="my-cls1" style="attr3:v3;">
          <caption>A comprehensive test</caption>
          <thead>
            <tr>
              <th class="blank level0" >&nbsp;</th>
              <th id="T__level0_col0" class="col_heading level0 col0" >A</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th id="T__level0_row0" class="row_heading level0 row0" >a</th>
              <td id="T__row0_col0" class="data row0 col0 my-cls2" >2.6</td>
            </tr>
            <tr>
              <th id="T__level0_row1" class="row_heading level0 row1" >b</th>
              <td id="T__row1_col0" class="data row1 col0" >2.7</td>
            </tr>
          </tbody>
        </table>
        """
    )
    # 使用 styler.to_html() 方法将样式转换为 HTML，并与预期的 HTML 字符串比较
    assert expected == styler.to_html()
# 测试列跨度功能
def test_colspan_w3():
    # GH 36223 GitHub问题编号
    # 创建包含单元格和多级列索引的数据帧
    df = DataFrame(data=[[1, 2]], columns=[["l0", "l0"], ["l1a", "l1b"]])
    # 使用Styler对象渲染数据帧，禁用单元格ID
    styler = Styler(df, uuid="_", cell_ids=False)
    # 断言HTML输出中是否包含指定的列合并表头
    assert '<th class="col_heading level0 col0" colspan="2">l0</th>' in styler.to_html()


# 测试行跨度功能
def test_rowspan_w3():
    # GH 38533 GitHub问题编号
    # 创建包含单元格和多级行索引的数据帧
    df = DataFrame(data=[[1, 2]], index=[["l0", "l0"], ["l1a", "l1b"]])
    # 使用Styler对象渲染数据帧，禁用单元格ID
    styler = Styler(df, uuid="_", cell_ids=False)
    # 断言HTML输出中是否包含指定的行合并表头
    assert '<th class="row_heading level0 row0" rowspan="2">l0</th>' in styler.to_html()


# 测试样式设置功能
def test_styles(styler):
    # 设置Styler对象的唯一标识符为"abc"
    styler.set_uuid("abc")
    # 设置表格样式，将所有<td>元素的文本颜色设置为红色
    styler.set_table_styles([{"selector": "td", "props": "color: red;"}])
    # 生成包含完整HTML结构的表格并返回结果
    result = styler.to_html(doctype_html=True)
    # 预期的HTML结构，包含DOCTYPE声明、<html>、<head>、<body>等标签
    expected = dedent(
        """\
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style type="text/css">
        #T_abc td {
          color: red;
        }
        </style>
        </head>
        <body>
        <table id="T_abc">
          <thead>
            <tr>
              <th class="blank level0" >&nbsp;</th>
              <th id="T_abc_level0_col0" class="col_heading level0 col0" >A</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th id="T_abc_level0_row0" class="row_heading level0 row0" >a</th>
              <td id="T_abc_row0_col0" class="data row0 col0" >2.610000</td>
            </tr>
            <tr>
              <th id="T_abc_level0_row1" class="row_heading level0 row1" >b</th>
              <td id="T_abc_row1_col0" class="data row1 col0" >2.690000</td>
            </tr>
          </tbody>
        </table>
        </body>
        </html>
        """
    )
    # 断言生成的HTML与预期结果一致
    assert result == expected


# 测试不包含DOCTYPE声明的HTML输出
def test_doctype(styler):
    # 生成不包含DOCTYPE声明的HTML表格
    result = styler.to_html(doctype_html=False)
    # 断言结果中不包含<html>、<body>、<!DOCTYPE html>等标签
    assert "<html>" not in result
    assert "<body>" not in result
    assert "<!DOCTYPE html>" not in result
    assert "<head>" not in result


# 测试指定编码的HTML输出
def test_doctype_encoding(styler):
    # 使用ASCII编码生成HTML并断言是否包含<meta charset="ASCII">
    with option_context("styler.render.encoding", "ASCII"):
        result = styler.to_html(doctype_html=True)
        assert '<meta charset="ASCII">' in result
        # 使用指定编码"ANSI"生成HTML并断言是否包含<meta charset="ANSI">
        result = styler.to_html(doctype_html=True, encoding="ANSI")
        assert '<meta charset="ANSI">' in result


# 测试加粗表头选项
def test_bold_headers_arg(styler):
    # 生成加粗表头的HTML输出并断言是否包含相应的CSS样式
    result = styler.to_html(bold_headers=True)
    assert "th {\n  font-weight: bold;\n}" in result
    # 生成默认HTML输出并断言不包含加粗表头的CSS样式
    result = styler.to_html()
    assert "th {\n  font-weight: bold;\n}" not in result


# 测试添加表格标题选项
def test_caption_arg(styler):
    # 生成包含指定标题的HTML输出并断言是否包含<caption>标签
    result = styler.to_html(caption="foo bar")
    assert "<caption>foo bar</caption>" in result
    # 生成默认HTML输出并断言不包含指定标题的<caption>标签
    result = styler.to_html()
    assert "<caption>foo bar</caption>" not in result


# 测试检查块名称
def test_block_names(tpl_style, tpl_table):
    # 捕捉意外删除的块名，验证预期的样式名称集合是否存在
    expected_style = {
        "before_style",
        "style",
        "table_styles",
        "before_cellstyle",
        "cellstyle",
    }
    # 定义期望的表格元素集合，包括预期出现在模板中的所有元素
    expected_table = {
        "before_table",
        "table",
        "caption",
        "thead",
        "tbody",
        "after_table",
        "before_head_rows",
        "head_tr",
        "after_head_rows",
        "before_rows",
        "tr",
        "after_rows",
    }
    # 获取模板样式对象的块集合，并转换为集合类型
    result1 = set(tpl_style.blocks)
    # 使用断言检查模板样式块集合是否与期望的表格元素集合相等
    assert result1 == expected_style

    # 获取模板表格对象的块集合，并转换为集合类型
    result2 = set(tpl_table.blocks)
    # 使用断言检查模板表格块集合是否与期望的表格元素集合相等
    assert result2 == expected_table
# 测试从自定义模板创建表格模板
def test_from_custom_template_table(tmpdir):
    # 在临时目录下创建一个名为 tpl 的子目录，并在其中创建一个名为 myhtml_table.tpl 的文件
    p = tmpdir.mkdir("tpl").join("myhtml_table.tpl")
    # 向文件写入模板内容，模板继承自 html_table.tpl，重写了 table 块
    p.write(
        dedent(
            """\
            {% extends "html_table.tpl" %}
            {% block table %}
            <h1>{{custom_title}}</h1>
            {{ super() }}
            {% endblock table %}"""
        )
    )
    # 使用 Styler 类的方法从自定义模板创建结果
    result = Styler.from_custom_template(str(tmpdir.join("tpl")), "myhtml_table.tpl")
    # 断言结果类型是 Styler 的子类
    assert issubclass(result, Styler)
    # 断言结果的环境与 Styler 的环境不同
    assert result.env is not Styler.env
    # 断言结果的 HTML 表格模板与 Styler 的 HTML 表格模板不同
    assert result.template_html_table is not Styler.template_html_table
    # 创建一个新的 styler 对象，并检查生成的 HTML 是否包含特定标题
    styler = result(DataFrame({"A": [1, 2]}))
    assert "<h1>My Title</h1>\n\n\n<table" in styler.to_html(custom_title="My Title")


# 测试从自定义模板创建样式模板
def test_from_custom_template_style(tmpdir):
    # 在临时目录下创建一个名为 tpl 的子目录，并在其中创建一个名为 myhtml_style.tpl 的文件
    p = tmpdir.mkdir("tpl").join("myhtml_style.tpl")
    # 向文件写入样式模板内容，模板继承自 html_style.tpl，重写了 style 块
    p.write(
        dedent(
            """\
            {% extends "html_style.tpl" %}
            {% block style %}
            <link rel="stylesheet" href="mystyle.css">
            {{ super() }}
            {% endblock style %}"""
        )
    )
    # 使用 Styler 类的方法从自定义模板创建结果，指定 HTML 样式模板为 myhtml_style.tpl
    result = Styler.from_custom_template(
        str(tmpdir.join("tpl")), html_style="myhtml_style.tpl"
    )
    # 断言结果类型是 Styler 的子类
    assert issubclass(result, Styler)
    # 断言结果的环境与 Styler 的环境不同
    assert result.env is not Styler.env
    # 断言结果的 HTML 样式模板与 Styler 的 HTML 样式模板不同
    assert result.template_html_style is not Styler.template_html_style
    # 创建一个新的 styler 对象，并检查生成的 HTML 是否包含特定样式链接
    styler = result(DataFrame({"A": [1, 2]}))
    assert '<link rel="stylesheet" href="mystyle.css">\n\n<style' in styler.to_html()


# 测试使用序列作为标题创建 caption
def test_caption_as_sequence(styler):
    # 设置 styler 的 caption 为 ("full cap", "short cap")
    styler.set_caption(("full cap", "short cap"))
    # 断言生成的 HTML 包含指定的 caption 标签
    assert "<caption>full cap</caption>" in styler.to_html()


# 参数化测试基本粘性功能
@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("index_name", [True, False])
def test_sticky_basic(styler, index, columns, index_name):
    # 如果 index_name 为 True，将 styler 的索引名称设置为 "some text"
    if index_name:
        styler.index.name = "some text"
    # 如果 index 为 True，设置 styler 的垂直粘性
    if index:
        styler.set_sticky(axis=0)
    # 如果 columns 为 True，设置 styler 的水平粘性
    if columns:
        styler.set_sticky(axis=1)

    # 定义左侧 CSS 样式
    left_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  left: 0px;\n  z-index: {1};\n}}"
    )
    # 定义顶部 CSS 样式
    top_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  top: {1}px;\n  z-index: {2};\n{3}}}"
    )

    # 生成 styler 的 HTML 字符串
    res = styler.set_uuid("").to_html()

    # 测试索引粘性是否作用于 thead 和 tbody
    assert (left_css.format("thead tr th:nth-child(1)", "3 !important") in res) is index
    assert (left_css.format("tbody tr th:nth-child(1)", "1") in res) is index

    # 测试列粘性，包括是否存在名称行
    assert (
        top_css.format("thead tr:nth-child(1) th", "0", "2", "  height: 25px;\n") in res
    ) is (columns and index_name)
    assert (
        top_css.format("thead tr:nth-child(2) th", "25", "2", "  height: 25px;\n")
        in res
    ) is (columns and index_name)
    assert (top_css.format("thead tr:nth-child(1) th", "0", "2", "") in res) is (
        columns and not index_name
    )
@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("columns", [False, True])
def test_sticky_mi(styler_mi, index, columns):
    # 根据参数化的index和columns，测试样式是否正确设置为sticky
    if index:
        styler_mi.set_sticky(axis=0)
    if columns:
        styler_mi.set_sticky(axis=1)

    # 左侧样式表CSS，用于控制横向粘性（sticky）效果
    left_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  left: {1}px;\n  min-width: 75px;\n  max-width: 75px;\n  z-index: {2};\n}}"
    )
    # 顶部样式表CSS，用于控制纵向粘性（sticky）效果
    top_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  top: {1}px;\n  height: 25px;\n  z-index: {2};\n}}"
    )

    # 将样式设置后的表格内容转换为HTML字符串
    res = styler_mi.set_uuid("").to_html()

    # 对thead和tbody的index进行测试，检查粘性是否如预期
    assert (
        left_css.format("thead tr th:nth-child(1)", "0", "3 !important") in res
    ) is index
    assert (left_css.format("tbody tr th.level0", "0", "1") in res) is index
    assert (
        left_css.format("thead tr th:nth-child(2)", "75", "3 !important") in res
    ) is index
    assert (left_css.format("tbody tr th.level1", "75", "1") in res) is index

    # 对每个级别的列进行测试，检查列的粘性是否如预期
    assert (top_css.format("thead tr:nth-child(1) th", "0", "2") in res) is columns
    assert (top_css.format("thead tr:nth-child(2) th", "25", "2") in res) is columns


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("levels", [[1], ["one"], "one"])
def test_sticky_levels(styler_mi, index, columns, levels):
    # 设置index和columns的名称为["zero", "one"]，并根据参数化的levels设置粘性
    styler_mi.index.names, styler_mi.columns.names = ["zero", "one"], ["zero", "one"]
    if index:
        styler_mi.set_sticky(axis=0, levels=levels)
    if columns:
        styler_mi.set_sticky(axis=1, levels=levels)

    # 左侧样式表CSS，用于控制横向粘性（sticky）效果
    left_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  left: {1}px;\n  min-width: 75px;\n  max-width: 75px;\n  z-index: {2};\n}}"
    )
    # 顶部样式表CSS，用于控制纵向粘性（sticky）效果
    top_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  top: {1}px;\n  height: 25px;\n  z-index: {2};\n}}"
    )

    # 将样式设置后的表格内容转换为HTML字符串
    res = styler_mi.set_uuid("").to_html()

    # 检查是否没有应用level0的粘性
    assert "#T_ thead tr th:nth-child(1)" not in res
    assert "#T_ tbody tr th.level0" not in res
    assert "#T_ thead tr:nth-child(1) th" not in res

    # 检查是否正确应用了level1的粘性
    assert (
        left_css.format("thead tr th:nth-child(2)", "0", "3 !important") in res
    ) is index
    assert (left_css.format("tbody tr th.level1", "0", "1") in res) is index
    assert (top_css.format("thead tr:nth-child(2) th", "0", "2") in res) is columns


def test_sticky_raises(styler):
    # 测试异常情况：设置错误的轴参数时是否引发了ValueError
    with pytest.raises(ValueError, match="No axis named bad for object type DataFrame"):
        styler.set_sticky(axis="bad")


@pytest.mark.parametrize(
    "sparse_index, sparse_columns",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_sparse_options(sparse_index, sparse_columns):
    # 参数化测试：测试稀疏索引和列的各种组合选项
    # 创建一个多级索引对象，行索引为 [("Z", "a"), ("Z", "b"), ("Y", "c")]
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    # 创建一个多级索引对象，列索引为 [("A", "a"), ("A", "b"), ("B", "c")]
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    # 创建一个数据框对象，包含指定的数据和行、列索引
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=ridx, columns=cidx)
    # 使用数据框的样式对象创建一个样式器对象
    styler = df.style

    # 获取默认的 HTML 表示形式
    default_html = styler.to_html()  # defaults under pd.options to (True , True)

    # 使用选项上下文设置 styler 的稀疏索引和稀疏列选项
    with option_context(
        "styler.sparse.index", sparse_index, "styler.sparse.columns", sparse_columns
    ):
        # 根据当前选项上下文获取 HTML 表示形式
        html1 = styler.to_html()
        # 断言当前的 HTML 表示形式与默认的 HTML 表示形式是否相等，条件是 sparse_index 和 sparse_columns 都为 True
        assert (html1 == default_html) is (sparse_index and sparse_columns)
    # 使用指定的稀疏索引和稀疏列选项获取 HTML 表示形式
    html2 = styler.to_html(sparse_index=sparse_index, sparse_columns=sparse_columns)
    # 断言两个 HTML 表示形式是否相等
    assert html1 == html2
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("columns", [True, False])
def test_map_header_cell_ids(styler, index, columns):
    # 标记测试用例，用于测试映射表头单元格的标识符
    # GH 41893

    # 定义一个匿名函数 func，用于返回固定字符串 "attr: val;"
    func = lambda v: "attr: val;"

    # 初始化 styler 对象的 uuid 和 cell_ids 属性
    styler.uuid, styler.cell_ids = "", False

    # 如果 index 参数为 True，则在 styler 对象上映射索引轴上的 func 函数
    if index:
        styler.map_index(func, axis="index")

    # 如果 columns 参数为 True，则在 styler 对象上映射列轴上的 func 函数
    if columns:
        styler.map_index(func, axis="columns")

    # 将 styler 对象转换为 HTML 字符串
    result = styler.to_html()

    # 断言检查：测试不包含数据单元格的特定 HTML 片段存在性
    assert '<td class="data row0 col0" >2.610000</td>' in result
    assert '<td class="data row1 col0" >2.690000</td>' in result

    # 断言检查：测试索引头部标识符的存在性和 CSS 样式
    assert (
        '<th id="T__level0_row0" class="row_heading level0 row0" >a</th>' in result
    ) is index
    assert (
        '<th id="T__level0_row1" class="row_heading level0 row1" >b</th>' in result
    ) is index
    assert ("#T__level0_row0, #T__level0_row1 {\n  attr: val;\n}" in result) is index

    # 断言检查：测试列头部标识符的存在性和 CSS 样式
    assert (
        '<th id="T__level0_col0" class="col_heading level0 col0" >A</th>' in result
    ) is columns
    assert ("#T__level0_col0 {\n  attr: val;\n}" in result) is columns


@pytest.mark.parametrize("rows", [True, False])
@pytest.mark.parametrize("cols", [True, False])
def test_maximums(styler_mi, rows, cols):
    # 标记测试用例，用于测试最大值限制的显示
    result = styler_mi.to_html(
        max_rows=2 if rows else None,
        max_columns=2 if cols else None,
    )

    # 断言检查：检查特定字符串在结果 HTML 中的存在性，验证行和列的最大值限制
    assert ">5</td>" in result  # [[0,1], [4,5]] always visible
    assert (">8</td>" in result) is not rows  # first trimmed vertical element
    assert (">2</td>" in result) is not cols  # first trimmed horizontal element


def test_replaced_css_class_names():
    # 定义 CSS 类名的替换映射字典
    css = {
        "row_heading": "ROWHEAD",
        # "col_heading": "COLHEAD",
        "index_name": "IDXNAME",
        # "col": "COL",
        "row": "ROW",
        # "col_trim": "COLTRIM",
        "row_trim": "ROWTRIM",
        "level": "LEVEL",
        "data": "DATA",
        "blank": "BLANK",
    }

    # 创建一个 MultiIndex 示例对象 midx
    midx = MultiIndex.from_product([["a", "b"], ["c", "d"]])

    # 创建 Styler 对象 styler_mi，使用指定的 DataFrame 和 uuid_len 参数
    styler_mi = Styler(
        DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=midx),
        uuid_len=0,
    ).set_table_styles(css_class_names=css)

    # 设置索引名称为 ["n1", "n2"]
    styler_mi.index.names = ["n1", "n2"]

    # 隐藏索引中除第一个元素外的所有行
    styler_mi.hide(styler_mi.index[1:], axis=0)

    # 隐藏列索引中除第一个元素外的所有列
    styler_mi.hide(styler_mi.columns[1:], axis=1)

    # 在行轴上映射函数，设置样式 "color: red;"
    styler_mi.map_index(lambda v: "color: red;", axis=0)

    # 在列轴上映射函数，设置样式 "color: green;"
    styler_mi.map_index(lambda v: "color: green;", axis=1)

    # 在所有单元格映射函数，设置样式 "color: blue;"
    styler_mi.map(lambda v: "color: blue;")

    # 期望的 HTML 样式表字符串，使用了 Python 的 dedent 函数去除首行缩进
    expected = dedent(
        """\
    <style type="text/css">
    #T__ROW0_col0 {
      color: blue;
    }
    #T__LEVEL0_ROW0, #T__LEVEL1_ROW0 {
      color: red;
    }
    #T__LEVEL0_col0, #T__LEVEL1_col0 {
      color: green;
    }
    </style>
    """
    )
    <table id="T_">
      <thead>
        <tr>
          <th class="BLANK" >&nbsp;</th>
          <th class="IDXNAME LEVEL0" >n1</th>
          <th id="T__LEVEL0_col0" class="col_heading LEVEL0 col0" >a</th>
        </tr>
        <tr>
          <th class="BLANK" >&nbsp;</th>
          <th class="IDXNAME LEVEL1" >n2</th>
          <th id="T__LEVEL1_col0" class="col_heading LEVEL1 col0" >c</th>
        </tr>
        <tr>
          <th class="IDXNAME LEVEL0" >n1</th>
          <th class="IDXNAME LEVEL1" >n2</th>
          <th class="BLANK col0" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T__LEVEL0_ROW0" class="ROWHEAD LEVEL0 ROW0" >a</th>
          <th id="T__LEVEL1_ROW0" class="ROWHEAD LEVEL1 ROW0" >c</th>
          <td id="T__ROW0_col0" class="DATA ROW0 col0" >0</td>
        </tr>
      </tbody>
    </table>

这段代码定义了一个HTML表格的结构，其中包含表头和表体，具体如下：

- `<table id="T_">`：定义一个表格，使用ID "T_"。
- `<thead>`：表头部分的开始标签。
- `<tr>`：表格中的行。
  - `<th class="BLANK" >&nbsp;</th>`：空白表头单元格，显示一个空格。
  - `<th class="IDXNAME LEVEL0">n1</th>`：表头单元格，显示文本 "n1"，属于第一级索引。
  - `<th id="T__LEVEL0_col0" class="col_heading LEVEL0 col0">a</th>`：带有ID和类的表头单元格，显示文本 "a"，属于第一级索引和第一列。
- 其余两个`<tr>`元素类似地定义了表格的其余部分。


    """
    )
    result = styler_mi.to_html()
    assert result == expected

这部分代码可能是在Python脚本中，但是缺少前文的上下文。通常这种结构会用于测试或者生成HTML内容。具体的注释可能需要根据上下文来决定，比如`styler_mi`是一个用于样式化或处理HTML表格的对象，`to_html()`方法用于将表格转换为HTML格式的字符串，而`assert result == expected`用于断言生成的HTML结果与预期的结果相同。
# 定义测试函数，用于测试仅对可见单元格应用 CSS 样式规则
def test_include_css_style_rules_only_for_visible_cells(styler_mi):
    # GH 43619
    # 设置 UUID 为空字符串，返回样式器本身
    result = (
        styler_mi.set_uuid("")
        # 对每个值应用颜色为蓝的样式
        .map(lambda v: "color: blue;")
        # 隐藏除第一列之外的所有列
        .hide(styler_mi.data.columns[1:], axis="columns")
        # 隐藏除第一行之外的所有行
        .hide(styler_mi.data.index[1:], axis="index")
        # 将样式器内容转换为 HTML 格式
        .to_html()
    )
    # 期望的样式
    expected_styles = dedent(
        """\
        <style type="text/css">
        #T__row0_col0 {
          color: blue;
        }
        </style>
        """
    )
    # 断言期望的样式在结果中
    assert expected_styles in result


# 定义测试函数，用于测试仅对可见索引标签应用 CSS 样式规则
def test_include_css_style_rules_only_for_visible_index_labels(styler_mi):
    # GH 43619
    # 设置 UUID 为空字符串，返回样式器本身
    result = (
        styler_mi.set_uuid("")
        # 对每个索引标签应用颜色为蓝的样式
        .map_index(lambda v: "color: blue;", axis="index")
        # 隐藏所有列
        .hide(styler_mi.data.columns, axis="columns")
        # 隐藏除第一行之外的所有行
        .hide(styler_mi.data.index[1:], axis="index")
        # 将样式器内容转换为 HTML 格式
        .to_html()
    )
    # 期望的样式
    expected_styles = dedent(
        """\
        <style type="text/css">
        #T__level0_row0, #T__level1_row0 {
          color: blue;
        }
        </style>
        """
    )
    # 断言期望的样式在结果中
    assert expected_styles in result


# 定义测试函数，用于测试仅对可见列标签应用 CSS 样式规则
def test_include_css_style_rules_only_for_visible_column_labels(styler_mi):
    # GH 43619
    # 设置 UUID 为空字符串，返回样式器本身
    result = (
        styler_mi.set_uuid("")
        # 对每个列标签应用颜色为蓝的样式
        .map_index(lambda v: "color: blue;", axis="columns")
        # 隐藏除第一列之外的所有列
        .hide(styler_mi.data.columns[1:], axis="columns")
        # 隐藏所有行
        .hide(styler_mi.data.index, axis="index")
        # 将样式器内容转换为 HTML 格式
        .to_html()
    )
    # 期望的样式
    expected_styles = dedent(
        """\
        <style type="text/css">
        #T__level0_col0, #T__level1_col0 {
          color: blue;
        }
        </style>
        """
    )
    # 断言期望的样式在结果中
    assert expected_styles in result


# 定义测试函数，用于测试多级索引时的隐藏索引和列对齐
def test_hiding_index_columns_multiindex_alignment():
    # gh 43644
    # 创建多级索引
    midx = MultiIndex.from_product(
        [["i0", "j0"], ["i1"], ["i2", "j2"]], names=["i-0", "i-1", "i-2"]
    )
    # 创建多级列索引
    cidx = MultiIndex.from_product(
        [["c0"], ["c1", "d1"], ["c2", "d2"]], names=["c-0", "c-1", "c-2"]
    )
    # 创建 DataFrame
    df = DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=cidx)
    # 创建样式器
    styler = Styler(df, uuid_len=0)
    # 隐藏第一级索引的所有行，隐藏第一级列索引的所有列
    styler.hide(level=1, axis=0).hide(level=0, axis=1)
    # 隐藏指定的行索引组合，隐藏指定的列索引组合
    styler.hide([("j0", "i1", "j2")], axis=0)
    styler.hide([("c0", "d1", "d2")], axis=1)
    # 将样式器内容转换为 HTML 格式
    result = styler.to_html()
    # 期望的结果为空样式
    expected = dedent(
        """\
    <style type="text/css">
    </style>
    """
    )
    # 定义一个包含表格的 HTML 字符串，包括表头和表体内容
    <table id="T_">
      <thead>
        <tr>
          <th class="blank" >&nbsp;</th>
          <th class="index_name level1" >c-1</th>
          <th id="T__level1_col0" class="col_heading level1 col0" colspan="2">c1</th>
          <th id="T__level1_col2" class="col_heading level1 col2" >d1</th>
        </tr>
        <tr>
          <th class="blank" >&nbsp;</th>
          <th class="index_name level2" >c-2</th>
          <th id="T__level2_col0" class="col_heading level2 col0" >c2</th>
          <th id="T__level2_col1" class="col_heading level2 col1" >d2</th>
          <th id="T__level2_col2" class="col_heading level2 col2" >c2</th>
        </tr>
        <tr>
          <th class="index_name level0" >i-0</th>
          <th class="index_name level2" >i-2</th>
          <th class="blank col0" >&nbsp;</th>
          <th class="blank col1" >&nbsp;</th>
          <th class="blank col2" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T__level0_row0" class="row_heading level0 row0" rowspan="2">i0</th>
          <th id="T__level2_row0" class="row_heading level2 row0" >i2</th>
          <td id="T__row0_col0" class="data row0 col0" >0</td>
          <td id="T__row0_col1" class="data row0 col1" >1</td>
          <td id="T__row0_col2" class="data row0 col2" >2</td>
        </tr>
        <tr>
          <th id="T__level2_row1" class="row_heading level2 row1" >j2</th>
          <td id="T__row1_col0" class="data row1 col0" >4</td>
          <td id="T__row1_col1" class="data row1 col1" >5</td>
          <td id="T__row1_col2" class="data row1 col2" >6</td>
        </tr>
        <tr>
          <th id="T__level0_row2" class="row_heading level0 row2" >j0</th>
          <th id="T__level2_row2" class="row_heading level2 row2" >i2</th>
          <td id="T__row2_col0" class="data row2 col0" >8</td>
          <td id="T__row2_col1" class="data row2 col1" >9</td>
          <td id="T__row2_col2" class="data row2 col2" >10</td>
        </tr>
      </tbody>
    </table>
    """
    # 检查结果与期望值是否相等
    assert result == expected
# 定义一个测试函数，测试隐藏多级索引的列和修剪
def test_hiding_index_columns_multiindex_trimming():
    # GitHub 问题编号 44272
    # 创建一个 8x8 的 DataFrame，数据填充为 0 到 63
    df = DataFrame(np.arange(64).reshape(8, 8))
    # 将列设置为多级索引，第一级包含 [0, 1, 2, 3]，第二级包含 [0, 1]
    df.columns = MultiIndex.from_product([[0, 1, 2, 3], [0, 1]])
    # 将行设置为多级索引，第一级和第二级分别包含 [0, 1, 2, 3]
    df.index = MultiIndex.from_product([[0, 1, 2, 3], [0, 1]])
    # 设置索引的名称为 ['a', 'b']，列的名称为 ['c', 'd']
    df.index.names, df.columns.names = ["a", "b"], ["c", "d"]
    # 使用 Styler 类创建一个样式对象 styler，禁用单元格 ID，设置 uuid 长度为 0
    styler = Styler(df, cell_ids=False, uuid_len=0)
    
    # 隐藏列中的特定位置 [(0, 0), (0, 1), (1, 0)]
    styler.hide([(0, 0), (0, 1), (1, 0)], axis=1)
    # 隐藏行中的特定位置 [(0, 0), (0, 1), (1, 0)]
    styler.hide([(0, 0), (0, 1), (1, 0)], axis=0)
    
    # 使用 option_context 设置 styler.render.max_rows 和 styler.render.max_columns 的最大显示行数和列数为 4
    with option_context("styler.render.max_rows", 4, "styler.render.max_columns", 4):
        # 将 styler 对象转换为 HTML 格式的结果
        result = styler.to_html()

    # 期望的 HTML 结果，使用 dedent 方法进行格式化
    expected = dedent(
        """\
        <style type="text/css">
        </style>
        """
    )
    # 创建一个表格对象，用于展示数据
    <table id="T_">
      <thead>  <!-- 表头部分 -->
        <tr>  <!-- 第一行表头 -->
          <th class="blank" >&nbsp;</th>  <!-- 空白表头 -->
          <th class="index_name level0" >c</th>  <!-- 第一列表头 -->
          <th class="col_heading level0 col3" >1</th>  <!-- 第二列表头 -->
          <th class="col_heading level0 col4" colspan="2">2</th>  <!-- 第三、第四列表头，合并为两列 -->
          <th class="col_heading level0 col6" >3</th>  <!-- 第五列表头 -->
        </tr>
        <tr>  <!-- 第二行表头 -->
          <th class="blank" >&nbsp;</th>  <!-- 空白表头 -->
          <th class="index_name level1" >d</th>  <!-- 第一列子表头 -->
          <th class="col_heading level1 col3" >1</th>  <!-- 第二列子表头 -->
          <th class="col_heading level1 col4" >0</th>  <!-- 第三列子表头 -->
          <th class="col_heading level1 col5" >1</th>  <!-- 第四列子表头 -->
          <th class="col_heading level1 col6" >0</th>  <!-- 第五列子表头 -->
          <th class="col_heading level1 col_trim" >...</th>  <!-- 省略号列 -->
        </tr>
        <tr>  <!-- 第三行表头 -->
          <th class="index_name level0" >a</th>  <!-- 第一列表头 -->
          <th class="index_name level1" >b</th>  <!-- 第二列表头 -->
          <th class="blank col3" >&nbsp;</th>  <!-- 空白表头 -->
          <th class="blank col4" >&nbsp;</th>  <!-- 空白表头 -->
          <th class="blank col5" >&nbsp;</th>  <!-- 空白表头 -->
          <th class="blank col6" >&nbsp;</th>  <!-- 空白表头 -->
          <th class="blank col7 col_trim" >&nbsp;</th>  <!-- 空白表头，省略号列 -->
        </tr>
      </thead>
      <tbody>  <!-- 表格主体部分 -->
        <tr>  <!-- 第一行数据 -->
          <th class="row_heading level0 row3" >1</th>  <!-- 第一行第一列数据行头 -->
          <th class="row_heading level1 row3" >1</th>  <!-- 第一行第二列数据行头 -->
          <td class="data row3 col3" >27</td>  <!-- 第一行第三列数据 -->
          <td class="data row3 col4" >28</td>  <!-- 第一行第四列数据 -->
          <td class="data row3 col5" >29</td>  <!-- 第一行第五列数据 -->
          <td class="data row3 col6" >30</td>  <!-- 第一行第六列数据 -->
          <td class="data row3 col_trim" >...</td>  <!-- 第一行省略号列数据 -->
        </tr>
        <tr>  <!-- 第二行数据 -->
          <th class="row_heading level0 row4" rowspan="2">2</th>  <!-- 第二行第一列数据行头，占两行 -->
          <th class="row_heading level1 row4" >0</th>  <!-- 第二行第二列数据行头 -->
          <td class="data row4 col3" >35</td>  <!-- 第二行第三列数据 -->
          <td class="data row4 col4" >36</td>  <!-- 第二行第四列数据 -->
          <td class="data row4 col5" >37</td>  <!-- 第二行第五列数据 -->
          <td class="data row4 col6" >38</td>  <!-- 第二行第六列数据 -->
          <td class="data row4 col_trim" >...</td>  <!-- 第二行省略号列数据 -->
        </tr>
        <tr>  <!-- 第三行数据 -->
          <th class="row_heading level1 row5" >1</th>  <!-- 第三行第二列数据行头 -->
          <td class="data row5 col3" >43</td>  <!-- 第三行第三列数据 -->
          <td class="data row5 col4" >44</td>  <!-- 第三行第四列数据 -->
          <td class="data row5 col5" >45</td>  <!-- 第三行第五列数据 -->
          <td class="data row5 col6" >46</td>  <!-- 第三行第六列数据 -->
          <td class="data row5 col_trim" >...</td>  <!-- 第三行省略号列数据 -->
        </tr>
        <tr>  <!-- 第四行数据 -->
          <th class="row_heading level0 row6" >3</th>  <!-- 第四行第一列数据行头 -->
          <th class="row_heading level1 row6" >0</th>  <!-- 第四行第二列数据行头 -->
          <td class="data row6 col3" >51</td>  <!-- 第四行第三列数据 -->
          <td class="data row6 col4" >52</td>  <!-- 第四行第四列数据 -->
          <td class="data row6 col5" >53</td>  <!-- 第四行第五列数据 -->
          <td class="data row6 col6" >54</td>  <!-- 第四行第六列数据 -->
          <td class="data row6 col_trim" >...</td>  <!-- 第四行省略号列数据 -->
        </tr>
        <tr>  <!-- 省略号行 -->
          <th class="row_heading level0 row_trim" >...</th>  <!-- 省略号行第一列 -->
          <th class="row_heading level1 row_trim" >...</th>  <!-- 省略号行第二列 -->
          <td class="data col3 row_trim" >...</td>  <!-- 省略号行第三列 -->
          <td class="data col4 row_trim" >...</td>  <!-- 省略号行第四列 -->
          <td class="data col5 row_trim" >...</td>  <!-- 省略号行第五列 -->
          <td class="data col6 row_trim" >...</td>  <!-- 省略号行第六列 -->
          <td class="data row_trim col_trim" >...</td>  <!-- 省略号行省略号列 -->
        </tr>
      </tbody>
    </table>
    # 使用断言检查变量 result 是否等于变量 expected 的值
    assert result == expected
@pytest.mark.parametrize("type", ["data", "index"])
@pytest.mark.parametrize(
    "text, exp, found",
    [
        ("no link, just text", False, ""),  # 测试用例：没有链接，只有文本
        ("subdomain not www: sub.web.com", False, ""),  # 测试用例：子域名非 www: sub.web.com
        ("www subdomain: www.web.com other", True, "www.web.com"),  # 测试用例：www 子域名: www.web.com
        ("scheme full structure: http://www.web.com", True, "http://www.web.com"),  # 测试用例：完整结构的 scheme: http://www.web.com
        ("scheme no top-level: http://www.web", True, "http://www.web"),  # 测试用例：无顶级域的 scheme: http://www.web
        ("no scheme, no top-level: www.web", False, "www.web"),  # 测试用例：无 scheme，无顶级域: www.web
        ("https scheme: https://www.web.com", True, "https://www.web.com"),  # 测试用例：https scheme: https://www.web.com
        ("ftp scheme: ftp://www.web", True, "ftp://www.web"),  # 测试用例：ftp scheme: ftp://www.web
        ("ftps scheme: ftps://www.web", True, "ftps://www.web"),  # 测试用例：ftps scheme: ftps://www.web
        ("subdirectories: www.web.com/directory", True, "www.web.com/directory"),  # 测试用例：子目录: www.web.com/directory
        ("Multiple domains: www.1.2.3.4", True, "www.1.2.3.4"),  # 测试用例：多个域名: www.1.2.3.4
        ("with port: http://web.com:80", True, "http://web.com:80"),  # 测试用例：带端口号: http://web.com:80
        (
            "full net_loc scheme: http://user:pass@web.com",
            True,
            "http://user:pass@web.com",
        ),  # 测试用例：完整 net_loc 的 scheme: http://user:pass@web.com
        (
            "with valid special chars: http://web.com/,.':;~!@#$*()[]",
            True,
            "http://web.com/,.':;~!@#$*()[]",
        ),  # 测试用例：带有有效特殊字符: http://web.com/,.':;~!@#$*()[]
    ],
)
def test_rendered_links(type, text, exp, found):
    if type == "data":
        df = DataFrame([text])  # 如果类型是"data"，创建一个包含文本的数据框
        styler = df.style.format(hyperlinks="html")  # 使用 HTML 格式化样式
    else:
        df = DataFrame([0], index=[text])  # 如果类型是"index"，创建一个包含文本的带索引的数据框
        styler = df.style.format_index(hyperlinks="html")  # 使用 HTML 格式化索引样式

    rendered = f'<a href="{found}" target="_blank">{found}</a>'  # 创建预期的渲染结果链接
    result = styler.to_html()  # 获取样式化后的数据框 HTML
    assert (rendered in result) is exp  # 断言渲染结果中是否包含预期的链接结果与预期值相符
    assert (text in result) is not exp  # 断言渲染结果中是否包含原始文本与预期值不相符


def test_multiple_rendered_links():
    links = ("www.a.b", "http://a.c", "https://a.d", "ftp://a.e")  # 多个链接的字符串元组
    df = DataFrame(["text {} {} text {} {}".format(*links)])  # 创建包含链接文本的数据框
    result = df.style.format(hyperlinks="html").to_html()  # 使用 HTML 格式化样式并转换为 HTML
    href = '<a href="{0}" target="_blank">{0}</a>'  # 预期的链接 HTML 格式
    for link in links:
        assert href.format(link) in result  # 断言每个链接是否在渲染结果中
    assert href.format("text") not in result  # 断言不包含非链接文本的 HTML 结果


def test_concat(styler):
    other = styler.data.agg(["mean"]).style  # 对数据进行聚合并样式化
    styler.concat(other).set_uuid("X")  # 连接样式并设置唯一标识符
    result = styler.to_html()  # 获取样式化后的 HTML 结果
    fp = "foot0_"  # 期望结果的前缀
    expected = dedent(
        f"""\
    <tr>
      <th id="T_X_level0_row1" class="row_heading level0 row1" >b</th>
      <td id="T_X_row1_col0" class="data row1 col0" >2.690000</td>
    </tr>
    <tr>
      <th id="T_X_level0_{fp}row0" class="{fp}row_heading level0 {fp}row0" >mean</th>
      <td id="T_X_{fp}row0_col0" class="{fp}data {fp}row0 col0" >2.650000</td>
    </tr>
  </tbody>
</table>
    """
    )  # 期望的 HTML 结果
    assert expected in result  # 断言期望的 HTML 结果是否在实际结果中


def test_concat_recursion(styler):
    df = styler.data  # 获取样式化的数据
    styler1 = styler  # 复制样式化对象
    styler2 = Styler(df.agg(["mean"]), precision=3)  # 创建新的样式化对象，对数据进行聚合
    styler3 = Styler(df.agg(["mean"]), precision=4)  # 创建新的样式化对象，对数据进行聚合
    styler1.concat(styler2.concat(styler3)).set_uuid("X")  # 连接样式化对象并设置唯一标识符
    result = styler.to_html()  # 获取样式化后的 HTML 结果
    # 定义变量 fp1，用于存储字符串 "foot0_"
    fp1 = "foot0_"
    # 定义变量 fp2，用于存储字符串 "foot0_foot0_"
    fp2 = "foot0_foot0_"
    # 期望的 HTML 表格内容，使用了多行字符串，并且通过 dedent 函数进行了缩进处理
    expected = dedent(
        f"""\
    <tr>
      <th id="T_X_level0_row1" class="row_heading level0 row1" >b</th>
      <td id="T_X_row1_col0" class="data row1 col0" >2.690000</td>
    </tr>
    <tr>
      <th id="T_X_level0_{fp1}row0" class="{fp1}row_heading level0 {fp1}row0" >mean</th>
      <td id="T_X_{fp1}row0_col0" class="{fp1}data {fp1}row0 col0" >2.650</td>
    </tr>
    <tr>
      <th id="T_X_level0_{fp2}row0" class="{fp2}row_heading level0 {fp2}row0" >mean</th>
      <td id="T_X_{fp2}row0_col0" class="{fp2}data {fp2}row0 col0" >2.6500</td>
    </tr>
  </tbody>
    """
    )
def test_concat_combined():
    # 定义内部函数html_lines，用于生成HTML表格行的字符串，带有特定的foot_prefix前缀
    def html_lines(foot_prefix: str):
        # 断言foot_prefix以"_"结尾或为空字符串
        assert foot_prefix.endswith("_") or foot_prefix == ""
        fp = foot_prefix
        # 返回格式化的HTML字符串，缩进4个空格
        return indent(
            dedent(
                f"""\
                <tr>
                  <th id="T_X_level0_{fp}row0" class="{fp}row_heading level0 {fp}row0" >a</th>
                  <td id="T_X_{fp}row0_col0" class="{fp}data {fp}row0 col0" >2.610000</td>
                </tr>
                <tr>
                  <th id="T_X_level0_{fp}row1" class="{fp}row_heading level0 {fp}row1" >b</th>
                  <td id="T_X_{fp}row1_col0" class="{fp}data {fp}row1 col0" >2.690000</td>
                </tr>
                """
            ),
            prefix=" " * 4,
        )

    # 创建DataFrame对象df，包含两行数据"a"和"b"，列名为"A"
    df = DataFrame([[2.61], [2.69]], index=["a", "b"], columns=["A"])
    # 对df进行样式化，高亮每列的最大值，设置颜色为红色，生成Styler对象s1
    s1 = df.style.highlight_max(color="red")
    # 同上，生成Styler对象s2，设置颜色为绿色
    s2 = df.style.highlight_max(color="green")
    # 同上，生成Styler对象s3，设置颜色为蓝色
    s3 = df.style.highlight_max(color="blue")
    # 同上，生成Styler对象s4，设置颜色为黄色
    s4 = df.style.highlight_max(color="yellow")

    # 连接四个Styler对象，形成一个新的Styler对象，设置UUID为"X"，然后转换为HTML字符串
    result = s1.concat(s2).concat(s3.concat(s4)).set_uuid("X").to_html()

    # 预期的CSS样式表，用于着色不同的单元格
    expected_css = dedent(
        """\
        <style type="text/css">
        #T_X_row1_col0 {
          background-color: red;
        }
        #T_X_foot0_row1_col0 {
          background-color: green;
        }
        #T_X_foot1_row1_col0 {
          background-color: blue;
        }
        #T_X_foot1_foot0_row1_col0 {
          background-color: yellow;
        }
        </style>
        """
    )

    # 预期的HTML表格内容，包含表头和多个由html_lines生成的表格行
    expected_table = (
        dedent(
            """\
            <table id="T_X">
              <thead>
                <tr>
                  <th class="blank level0" >&nbsp;</th>
                  <th id="T_X_level0_col0" class="col_heading level0 col0" >A</th>
                </tr>
              </thead>
              <tbody>
            """
        )
        + html_lines("")
        + html_lines("foot0_")
        + html_lines("foot1_")
        + html_lines("foot1_foot0_")
        + dedent(
            """\
              </tbody>
            </table>
            """
        )
    )
    # 断言：验证 expected_css 和 expected_table 的组合是否等于 result
    assert expected_css + expected_table == result
# 定义一个测试函数，用于测试处理非标量数据时的 HTML 格式输出，处理 GH47103 问题
def test_to_html_na_rep_non_scalar_data(datapath):
    # 创建一个 DataFrame 对象，包含一个字典的列表，其中包括一个整数，一个列表和一个 NaN 值
    df = DataFrame([{"a": 1, "b": [1, 2, 3], "c": np.nan}])
    # 使用样式化对象 Styler 将 DataFrame 格式化为 HTML 表格，并将 NaN 值替换为 "-"
    result = df.style.format(na_rep="-").to_html(table_uuid="test")
    # 预期的 HTML 输出字符串，包含一个表格和相应的样式
    expected = """\
<style type="text/css">
</style>
<table id="T_test">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_test_level0_col0" class="col_heading level0 col0" >a</th>
      <th id="T_test_level0_col1" class="col_heading level0 col1" >b</th>
      <th id="T_test_level0_col2" class="col_heading level0 col2" >c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_test_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_test_row0_col0" class="data row0 col0" >1</td>
      <td id="T_test_row0_col1" class="data row0 col1" >[1, 2, 3]</td>
      <td id="T_test_row0_col2" class="data row0 col2" >-</td>
    </tr>
  </tbody>
</table>
"""
    # 断言结果与预期输出相符
    assert result == expected


# 使用参数化测试装饰器进行多组测试，测试 Styler.format_index_names 方法
@pytest.mark.parametrize("escape_axis_0", [True, False])
@pytest.mark.parametrize("escape_axis_1", [True, False])
def test_format_index_names(styler_multi, escape_axis_0, escape_axis_1):
    # 根据 escape_axis_0 的值选择不同的格式化索引名方式
    if escape_axis_0:
        styler_multi.format_index_names(axis=0, escape="html")
        expected_index = ["X&gt;", "y_"]
    else:
        expected_index = ["X>", "y_"]

    # 根据 escape_axis_1 的值选择不同的格式化列名方式
    if escape_axis_1:
        styler_multi.format_index_names(axis=1, escape="html")
        expected_columns = ["A&amp;", "b&amp;"]
    else:
        expected_columns = ["A&", "b&"]

    # 将格式化后的 Styler 对象转换为 HTML 表格
    result = styler_multi.to_html(table_uuid="test")
    # 断言预期的索引名和列名是否出现在结果中
    for expected_str in expected_index + expected_columns:
        assert f"{expected_str}</th>" in result
```